"""
ExtendedABCD background estimation param model.

Uses 6 regions by adding a second sideband bin (Ax, Bx) in the x direction.
The fake rate is extrapolated log-linearly from the two sideband measurements
to the signal region:

             y=signal    y=sideband
x=extra-sideband:  Ax        Bx
x=sideband:         A         B
x=signal:           C         D  ← predicted

D = C * Ax * B² / (Bx * A²)

This follows from assuming the fake rate f(x) = B_x / A_x is log-linear in x,
so f(x_signal) = f(x_sideband)² / f(x_extra-sideband).
"""

import numpy as np
import tensorflow as tf

from rabbit.param_models.abcd_model import _build_param_names, _get_global_indices
from rabbit.param_models.param_model import ParamModel


class ExtendedABCD(ParamModel):
    """
    Extended ABCD background estimation model with 6 regions.

    Free parameters a_i, b_i, c_i, ax_i, bx_i for regions A, B, C, Ax, Bx.
    Region D is derived: ``rnorm_D = c · ax · b² / (bx · a²)`` (default,
    ``yield_correction=False``), or ``rnorm_D = (c · ax · b² / (bx · a²)) ·
    mc_factor_D`` with ``mc_factor_D = norm_C · norm_Ax · norm_B² /
    (norm_Bx · norm_A² · norm_D)`` when ``yield_correction=True``. The
    factor enforces the extended-ABCD relation on yields for arbitrary MC
    templates; see ABCD docstring for the rnorm-vs-yield discussion.

    Parameters are pure model nuisances (npoi=0, npou=5*n_bins). Positivity
    is enforced inside compute() via tf.square() on the raw fit variables.

    CLI syntax:
        --paramModel ExtendedABCD [yieldCorrection:0|1] <process> \\
            <ch_Ax> [ax:val ...] <ch_Bx> [ax:val ...] \\
            <ch_A>  [ax:val ...] <ch_B>  [ax:val ...] \\
            <ch_C>  [ax:val ...] <ch_D>  [ax:val ...]
    where ``yieldCorrection:0`` is the default; pass ``yieldCorrection:1`` to
    enable the yield-level form.

    Python constructor:
        ExtendedABCD(indata, "nonprompt",
                     channel_A={"ch_fakes": {"iso": 1}},
                     channel_B={"ch_fakes": {"iso": 2}},
                     channel_C={"ch_C": {}},
                     channel_D={"ch_D": {}},
                     channel_Ax={"ch_fakes": {"iso": 0}},
                     channel_Bx={"ch_fakes": {"iso": 3}})
    """

    def __init__(
        self,
        indata,
        abcd_process,
        channel_A,
        channel_B,
        channel_C,
        channel_D,
        channel_Ax,
        channel_Bx,
        yield_correction=False,
        **kwargs,
    ):
        """
        Args:
            indata: FitInputData instance.
            abcd_process: name of the background process (str).
            channel_A/B/C/D: dicts {ch_name: {axis_name: bin_idx, ...}}.
            channel_Ax/Bx: extra sideband regions (further from signal than A/B).
            yield_correction: if True, multiply ``rnorm_D`` by ``mc_factor_D`` to
                enforce the extended-ABCD relation on yields for arbitrary
                MC templates. Default False.
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

        # Get global flat indices for all six regions
        idx_A = _get_global_indices(indata, channel_A)
        idx_B = _get_global_indices(indata, channel_B)
        idx_C = _get_global_indices(indata, channel_C)
        idx_D = _get_global_indices(indata, channel_D)
        idx_Ax = _get_global_indices(indata, channel_Ax)
        idx_Bx = _get_global_indices(indata, channel_Bx)

        n = len(idx_A)
        for label, idx in [
            ("B", idx_B),
            ("C", idx_C),
            ("D", idx_D),
            ("Ax", idx_Ax),
            ("Bx", idx_Bx),
        ]:
            if len(idx) != n:
                raise ValueError(
                    f"All regions must have the same number of bins; "
                    f"A={n} but {label}={len(idx)}"
                )
        self.n_bins = n

        # Extract MC templates
        norm_dense = tf.sparse.to_dense(indata.norm) if indata.sparse else indata.norm
        norm_proc = tf.cast(norm_dense[:, proc_idx], dtype=indata.dtype)

        mc_A = tf.gather(norm_proc, idx_A)
        mc_B = tf.gather(norm_proc, idx_B)
        mc_C = tf.gather(norm_proc, idx_C)
        mc_D = tf.gather(norm_proc, idx_D)
        mc_Ax = tf.gather(norm_proc, idx_Ax)
        mc_Bx = tf.gather(norm_proc, idx_Bx)

        # MC correction factor: D = C * Ax * B² / (Bx * A² * D_MC) * (numerator MC)
        # rnorm_D = rnorm_C * rnorm_Ax * rnorm_B² / (rnorm_Bx * rnorm_A²) * mc_factor_D
        # mc_factor_D = N_C * N_Ax * N_B² / (N_Bx * N_A² * N_D)
        mc_A_safe = tf.where(mc_A == 0.0, tf.ones_like(mc_A), mc_A)
        mc_Bx_safe = tf.where(mc_Bx == 0.0, tf.ones_like(mc_Bx), mc_Bx)
        mc_D_safe = tf.where(mc_D == 0.0, tf.ones_like(mc_D), mc_D)
        self.mc_factor_D = (
            mc_C * mc_Ax * mc_B**2 / (mc_Bx_safe * mc_A_safe**2 * mc_D_safe)
        )

        # Scatter indices for compute()
        self._idx = {
            "A": [[int(i), proc_idx] for i in idx_A],
            "B": [[int(i), proc_idx] for i in idx_B],
            "C": [[int(i), proc_idx] for i in idx_C],
            "D": [[int(i), proc_idx] for i in idx_D],
            "Ax": [[int(i), proc_idx] for i in idx_Ax],
            "Bx": [[int(i), proc_idx] for i in idx_Bx],
        }

        # Per-region activity flags for non-full mode
        self._active_nonfull = {
            r: bool(all(int(i) < indata.nbins for i in idxs))
            for r, idxs in zip(
                ["A", "B", "C", "D", "Ax", "Bx"],
                [idx_A, idx_B, idx_C, idx_D, idx_Ax, idx_Bx],
            )
        }

        # Parameter names: A, B, C, Ax, Bx (5 regions × n_bins)
        names = []
        for ch_dict, region_label in [
            (channel_A, "A"),
            (channel_B, "B"),
            (channel_C, "C"),
            (channel_Ax, "Ax"),
            (channel_Bx, "Bx"),
        ]:
            ch_name, axis_sel = next(iter(ch_dict.items()))
            ch_axes = indata.channel_info[ch_name]["axes"]
            ch_shape = [a.size for a in ch_axes]
            names.extend(
                _build_param_names(abcd_process, ch_name, ch_axes, axis_sel, ch_shape)
            )

        # Model attributes
        self.npoi = 0
        self.npou = 5 * n
        self.params = np.array(names)
        self.is_linear = False
        self.allowNegativeParam = False
        self.xparamdefault = tf.ones([5 * n], dtype=indata.dtype)

    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        """Parse CLI arguments for ExtendedABCD.

        Syntax:
            --paramModel ExtendedABCD [yieldCorrection:0|1] <process> \\
                         <ch_Ax> [ax:val ...] <ch_Bx> [ax:val ...] \\
                         <ch_A>  [ax:val ...] <ch_B>  [ax:val ...] \\
                         <ch_C>  [ax:val ...] <ch_D>  [ax:val ...]
        ``yieldCorrection`` defaults to 0 (no MC factor); pass
        ``yieldCorrection:1`` to enable the yield-level form.
        """
        if len(args) < 7:
            raise ValueError(
                "ExtendedABCD expects: [yieldCorrection:0|1] process "
                "ch_Ax [ax:val ...] ch_Bx [ax:val ...] "
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

        channel_Ax = _parse_one_region(tokens, "Ax")
        channel_Bx = _parse_one_region(tokens, "Bx")
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
            channel_Ax,
            channel_Bx,
            yield_correction=yield_correction,
            **kwargs,
        )

    def compute(self, param, full=False):
        """Compute per-bin, per-process yield scale factors.

        Args:
            param: 1D tensor of length 5*n_bins (raw fit vars for a, b, c, ax, bx).
            full: if True return [nbinsfull, nproc]; else [nbins, nproc].
        """
        n = self.n_bins
        # Enforce positivity via squaring; raw param=1 → physical value=1
        a = tf.square(param[0 * n : 1 * n])
        b = tf.square(param[1 * n : 2 * n])
        c = tf.square(param[2 * n : 3 * n])
        ax = tf.square(param[3 * n : 4 * n])
        bx = tf.square(param[4 * n :])

        # D = C * Ax * B² / (Bx * A²) * mc_factor_D
        a_safe = tf.where(a == 0.0, tf.ones_like(a) * 1e-10, a)
        bx_safe = tf.where(bx == 0.0, tf.ones_like(bx) * 1e-10, bx)
        d = c * ax * b**2 / (bx_safe * a_safe**2)
        if self.yield_correction:
            d = d * self.mc_factor_D

        nbins = self.indata.nbinsfull if full else self.indata.nbins
        rnorm = tf.ones([nbins, self.indata.nproc], dtype=self.indata.dtype)

        # Python-level control flow resolved at @tf.function trace time
        indices = []
        updates = []
        for region, vals in [
            ("A", a),
            ("B", b),
            ("C", c),
            ("D", d),
            ("Ax", ax),
            ("Bx", bx),
        ]:
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
