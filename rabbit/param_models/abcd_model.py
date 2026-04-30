"""
ABCD background estimation param model.

The ABCD method estimates a background process using four regions defined by
two independent discriminating variables x and y. The independence condition
A/B = C/D is enforced by construction: free parameters a_i, b_i, c_i scale
regions A, B, C per bin, and the signal region D is predicted as
D_i = a_i * c_i / b_i (up to an MC correction factor).
"""

import itertools

import numpy as np
import tensorflow as tf

from rabbit.param_models.param_model import ParamModel


def _get_global_indices(indata, channel_dict):
    """Convert one region's channel dict to a numpy array of global flat bin indices.

    Args:
        indata: FitInputData instance (channel_info must already have start/stop set).
        channel_dict: dict with exactly one entry {ch_name: {axis_name: bin_idx, ...}}.
            An empty axis selection dict means all bins of the channel are taken.

    Returns:
        numpy array of global flat bin indices for the selected bins.
    """
    if len(channel_dict) != 1:
        raise ValueError(
            f"Each ABCD region must specify exactly one channel, got {list(channel_dict.keys())}"
        )
    ch_name, axis_sel = next(iter(channel_dict.items()))

    if ch_name not in indata.channel_info:
        raise ValueError(
            f"Channel '{ch_name}' not found. Available channels: {list(indata.channel_info.keys())}"
        )

    info = indata.channel_info[ch_name]
    axes = info["axes"]
    start = info["start"]

    shape = [a.size for a in axes]
    all_indices = np.arange(int(np.prod(shape))).reshape(shape)

    slices = [slice(None)] * len(axes)
    for ax_name, ax_idx in axis_sel.items():
        ax_pos = next((i for i, a in enumerate(axes) if a.name == ax_name), None)
        if ax_pos is None:
            raise ValueError(
                f"Axis '{ax_name}' not found in channel '{ch_name}'. "
                f"Available axes: {[a.name for a in axes]}"
            )
        slices[ax_pos] = ax_idx

    local_indices = all_indices[tuple(slices)].flatten()
    return start + local_indices


def _build_param_names(process, ch_name, axes, axis_sel, shape):
    """Build parameter name strings for each selected bin of a region.

    Returns a list of byte strings like b'process_chname_ax0name0_ax1name2'.
    """
    # Full index grid over all axes
    all_axis_ranges = [range(a.size) for a in axes]
    names = []
    for idxs in itertools.product(*all_axis_ranges):
        # Apply axis selection: skip bins not matching the fixed selections
        skip = False
        for ax_name, ax_idx in axis_sel.items():
            ax_pos = next(i for i, a in enumerate(axes) if a.name == ax_name)
            if idxs[ax_pos] != ax_idx:
                skip = True
                break
        if skip:
            continue
        label = "_".join(f"{a.name}{i}" for a, i in zip(axes, idxs))
        names.append(f"{process}_{ch_name}_{label}".encode())
    return names


class ABCD(ParamModel):
    """
    ABCD background estimation model.

    Defines free parameters a_i, b_i, c_i for regions A, B, C per bin.
    Region D (signal region) is derived from the ABCD relation. The
    relation can be enforced at two levels:

    - rnorm-level (default, ``yield_correction=False``):
        ``rnorm_D = a · c / b``. Relation on yields then holds only when
        ``norm_A · norm_C = norm_B · norm_D`` (e.g. all four MC templates
        share the same per-bin shape, which is the typical case when the
        regions are different bin-slices of the same channel).
    - yield-level (``yield_correction=True``):
        ``rnorm_D = (a · c / b) · mc_factor_D`` with
        ``mc_factor_D = norm_A · norm_C / (norm_B · norm_D)``. This makes
        ``nexp_A · nexp_C = nexp_B · nexp_D`` hold for arbitrary MC
        templates, at the cost of folding the MC shape difference between
        regions into the parametrisation.

    Parameters are pure model nuisances (npoi=0, npou=3*n_bins). Positivity
    is enforced inside compute() via tf.square() on the raw fit variables.

    CLI syntax:
        --paramModel ABCD [yieldCorrection:0] <process> \\
                          <ch_A> [ax:val ...] <ch_B> [ax:val ...] \\
                          <ch_C> [ax:val ...] <ch_D> [ax:val ...]
    where ``yieldCorrection:0`` is the default (no MC factor); pass
    ``yieldCorrection:1`` to enable the yield-level form.

    Python constructor:
        ABCD(indata, "nonprompt",
                  channel_A={"ch_fakes": {"iso": 0}},
                  channel_B={"ch_fakes": {"iso": 1}},
                  channel_C={"ch_C": {}},
                  channel_D={"ch_D": {}})
    """

    def __init__(
        self,
        indata,
        abcd_process,
        channel_A,
        channel_B,
        channel_C,
        channel_D,
        yield_correction=False,
        **kwargs,
    ):
        """
        Args:
            indata: FitInputData instance.
            abcd_process: name of the background process to apply ABCD to (str).
            channel_A/B/C/D: dicts {ch_name: {axis_name: bin_idx, ...}} defining
                each ABCD region. An empty inner dict selects all bins of the channel.
            yield_correction: if True, multiply ``rnorm_D`` by ``mc_factor_D`` to
                enforce the ABCD relation on yields for arbitrary MC templates.
                Default False.
        """
        self.indata = indata
        self.yield_correction = yield_correction

        # Validate process
        proc_name = (
            abcd_process.encode() if isinstance(abcd_process, str) else abcd_process
        )
        if proc_name not in indata.procs:
            raise ValueError(
                f"Process '{abcd_process}' not found. Available: {list(indata.procs)}"
            )
        proc_idx = int(np.where(np.array(indata.procs) == proc_name)[0][0])

        # Get global flat indices for each region
        idx_A = _get_global_indices(indata, channel_A)
        idx_B = _get_global_indices(indata, channel_B)
        idx_C = _get_global_indices(indata, channel_C)
        idx_D = _get_global_indices(indata, channel_D)

        n = len(idx_A)
        if not (len(idx_B) == len(idx_C) == len(idx_D) == n):
            raise ValueError(
                f"All ABCD regions must have the same number of bins, got "
                f"A={len(idx_A)}, B={len(idx_B)}, C={len(idx_C)}, D={len(idx_D)}"
            )
        self.n_bins = n

        # Extract MC templates for the ABCD process
        norm_dense = tf.sparse.to_dense(indata.norm) if indata.sparse else indata.norm
        norm_proc = tf.cast(norm_dense[:, proc_idx], dtype=indata.dtype)

        mc_A = tf.gather(norm_proc, idx_A)
        mc_B = tf.gather(norm_proc, idx_B)
        mc_C = tf.gather(norm_proc, idx_C)
        mc_D = tf.gather(norm_proc, idx_D)

        # MC correction factor D = A*C / (B*D_MC); protect against zeros
        mc_B_safe = tf.where(mc_B == 0.0, tf.ones_like(mc_B), mc_B)
        mc_D_safe = tf.where(mc_D == 0.0, tf.ones_like(mc_D), mc_D)
        self.mc_factor_D = mc_A * mc_C / (mc_B_safe * mc_D_safe)

        # Scatter indices for compute(): stored as Python lists so they become
        # tf.constant at @tf.function trace time (not retraced per call)
        self._idx = {
            "A": [[int(i), proc_idx] for i in idx_A],
            "B": [[int(i), proc_idx] for i in idx_B],
            "C": [[int(i), proc_idx] for i in idx_C],
            "D": [[int(i), proc_idx] for i in idx_D],
        }

        # Per-region activity flags for non-full mode
        # (masked channels have bin indices >= indata.nbins)
        self._active_nonfull = {
            r: bool(all(int(i) < indata.nbins for i in idxs))
            for r, idxs in zip("ABCD", [idx_A, idx_B, idx_C, idx_D])
        }

        # Build parameter names: a_i for A, b_i for B, c_i for C
        names = []
        for ch_dict, region_label in [
            (channel_A, "A"),
            (channel_B, "B"),
            (channel_C, "C"),
        ]:
            ch_name, axis_sel = next(iter(ch_dict.items()))
            ch_axes = indata.channel_info[ch_name]["axes"]
            ch_shape = [a.size for a in ch_axes]
            names.extend(
                _build_param_names(abcd_process, ch_name, ch_axes, axis_sel, ch_shape)
            )

        # Model attributes
        self.npoi = 0
        self.npou = 3 * n
        self.params = np.array(names)
        self.is_linear = False
        self.allowNegativeParam = False
        self.xparamdefault = tf.ones([3 * n], dtype=indata.dtype)

    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        """Parse CLI arguments for ABCDModel.

        Syntax:
            --paramModel ABCD [yieldCorrection:0|1] <process> \\
                              <ch_A> [ax:val ...] <ch_B> [ax:val ...] \\
                              <ch_C> [ax:val ...] <ch_D> [ax:val ...]

        Each channel name is followed by zero or more axis:value pairs.
        Axis values are integers. ``yieldCorrection`` defaults to 0 (no MC
        factor); pass ``yieldCorrection:1`` to enable the yield-level form.
        """
        if len(args) < 5:
            raise ValueError(
                "ABCDModel expects: [yieldCorrection:0|1] process "
                "ch_A [ax:val ...] ch_B [ax:val ...] "
                "ch_C [ax:val ...] ch_D [ax:val ...]"
            )
        tokens = list(args)

        yield_correction = False
        if tokens and tokens[0].startswith("yieldCorrection:"):
            yield_correction = bool(int(tokens.pop(0).split(":", 1)[1]))

        process = tokens.pop(0)

        def _parse_one_region(tokens, region_name):
            if not tokens or ":" in tokens[0]:
                raise ValueError(
                    f"Expected channel name for region {region_name}, "
                    f"got '{tokens[0] if tokens else '<end>'}'"
                )
            ch_name = tokens.pop(0)
            axis_sel = {}
            while tokens and ":" in tokens[0]:
                ax, val = tokens.pop(0).split(":", 1)
                axis_sel[ax] = int(val)
            return {ch_name: axis_sel}

        channel_A = _parse_one_region(tokens, "A")
        channel_B = _parse_one_region(tokens, "B")
        channel_C = _parse_one_region(tokens, "C")
        channel_D = _parse_one_region(tokens, "D")

        if tokens:
            raise ValueError(f"Unexpected extra arguments after channel_D: {tokens}")

        return cls(
            indata,
            process,
            channel_A,
            channel_B,
            channel_C,
            channel_D,
            yield_correction=yield_correction,
            **kwargs,
        )

    def compute(self, param, full=False):
        """Compute per-bin, per-process yield scale factors.

        Args:
            param: 1D tensor of length 3*n_bins (raw fit variables for a, b, c).
            full: if True, return [nbinsfull, nproc]; else [nbins, nproc].

        Returns:
            2D tensor [nbins(_full), nproc] of yield scale factors.
            All entries are 1 except for the ABCD process bins which are set to
            a_i, b_i, c_i (regions A, B, C) and d_i = a_i*c_i/b_i*mc_factor (D).
        """
        n = self.n_bins
        # Enforce positivity via squaring; raw param=1 → physical value=1
        a = tf.square(param[:n])
        b = tf.square(param[n : 2 * n])
        c = tf.square(param[2 * n :])

        b_safe = tf.where(b == 0.0, tf.ones_like(b) * 1e-10, b)
        d = a * c / b_safe
        if self.yield_correction:
            d = d * self.mc_factor_D

        nbins = self.indata.nbinsfull if full else self.indata.nbins
        rnorm = tf.ones([nbins, self.indata.nproc], dtype=self.indata.dtype)

        # Python-level control flow resolved at @tf.function trace time
        indices = []
        updates = []
        for region, vals in [("A", a), ("B", b), ("C", c), ("D", d)]:
            active = True if full else self._active_nonfull[region]
            if active:
                indices.extend(self._idx[region])
                updates.append(vals)

        if indices:
            rnorm = tf.tensor_scatter_nd_update(
                rnorm,
                tf.constant(indices, dtype=tf.int64),
                tf.concat(updates, axis=0),
            )
        return rnorm
