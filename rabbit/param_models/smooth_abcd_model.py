"""
SmoothABCD background estimation param model.

Like ABCD, but the per-bin free parameters along one nominated smoothing axis
are replaced by an exponential Chebyshev polynomial:

    val_X(x) = exp(p_0·T_0(x̃) + p_1·T_1(x̃) + p_2·T_2(x̃) + ...)

where T_k is the Chebyshev polynomial of the first kind and x̃ is the bin
centre linearly rescaled so the smoothing axis edges [x_min, x_max] map to
[-1, +1].  This matches the basis used by the WRemnants spectrum regressor,
so pre-computed coefficients can be loaded directly as initial values.

This reduces parameters from 3·n_bins to 3·n_outer·(order+1), where
n_outer is the number of bins along the remaining (non-smooth) axes.
"""

import itertools

import numpy as np
import tensorflow as tf

from rabbit.param_models.abcd_model import _get_global_indices
from rabbit.param_models.param_model import ParamModel


class SmoothABCD(ParamModel):
    """
    Smooth ABCD background estimation model.

    Regions A, B, C have free parameters that follow an exponential polynomial
    along the smoothing axis; region D is derived as d = a·c/b · mc_factor_D.

    Parameters are pure model nuisances (npoi=0, npou=3·n_outer·(order+1)).

    CLI syntax:
        --paramModel SmoothABCD <axis> [order:N] <process> \\
                     <ch_A> [ax:val ...] <ch_B> [ax:val ...] \\
                     <ch_C> [ax:val ...] <ch_D> [ax:val ...]

    Python constructor:
        SmoothABCD(indata, "pt", "nonprompt",
                   channel_A={"ch_fakes": {"iso": 0}},
                   channel_B={"ch_fakes": {"iso": 1}},
                   channel_C={"ch_C": {}},
                   channel_D={"ch_D": {}},
                   order=1)
    """

    def __init__(
        self,
        indata,
        smoothing_axis,
        abcd_process,
        channel_A,
        channel_B,
        channel_C,
        channel_D,
        order=1,
        **kwargs,
    ):
        """
        Args:
            indata: FitInputData instance.
            smoothing_axis: name of the axis to parameterise smoothly.
            abcd_process: name of the background process (str).
            channel_A/B/C/D: dicts {ch_name: {axis_name: bin_idx, ...}}.
            order: polynomial degree in the exponent (default 1 = log-linear).
        """
        self.indata = indata
        self.order = order

        # Validate process
        proc_name = (
            abcd_process.encode() if isinstance(abcd_process, str) else abcd_process
        )
        if proc_name not in indata.procs:
            raise ValueError(
                f"Process '{abcd_process}' not found. Available: {list(indata.procs)}"
            )
        proc_idx = int(np.where(np.array(indata.procs) == proc_name)[0][0])

        # Build outer × smooth index structure for each region
        # outer_global_indices[region] = list of length n_outer,
        #   each entry is an array of n_smooth global flat indices
        outer_global_indices = {}
        outer_axes_by_region = {}
        smooth_size = None

        for region, ch_dict in zip(
            "ABCD", [channel_A, channel_B, channel_C, channel_D]
        ):
            ch_name, axis_sel = next(iter(ch_dict.items()))
            axes = indata.channel_info[ch_name]["axes"]

            # Free axes after the fixed axis selections
            remaining = [a for a in axes if a.name not in axis_sel]

            smooth_ax = next((a for a in remaining if a.name == smoothing_axis), None)
            if smooth_ax is None:
                raise ValueError(
                    f"Smoothing axis '{smoothing_axis}' not found among free axes "
                    f"of channel '{ch_name}' for region {region}. "
                    f"Free axes: {[a.name for a in remaining]}"
                )

            n_smooth = smooth_ax.size
            if smooth_size is None:
                smooth_size = n_smooth
            elif smooth_size != n_smooth:
                raise ValueError(
                    f"Smoothing axis '{smoothing_axis}' has inconsistent size: "
                    f"expected {smooth_size}, got {n_smooth} in region {region}"
                )

            outer_axes = [a for a in remaining if a.name != smoothing_axis]
            outer_axes_by_region[region] = outer_axes

            # For each outer-bin combination, gather the n_smooth global indices
            # (smooth axis is left free → _get_global_indices returns n_smooth values)
            region_outer_indices = []
            for outer_combo in itertools.product(*[range(a.size) for a in outer_axes]):
                extended_sel = dict(axis_sel)
                for ax, idx in zip(outer_axes, outer_combo):
                    extended_sel[ax.name] = idx
                region_outer_indices.append(
                    _get_global_indices(indata, {ch_name: extended_sel})
                )

            outer_global_indices[region] = region_outer_indices

        n_smooth = smooth_size
        n_outer = len(outer_global_indices["A"])
        self.n_outer = n_outer
        self.n_smooth = n_smooth

        for region in "BCD":
            if len(outer_global_indices[region]) != n_outer:
                raise ValueError(
                    f"Region {region} has {len(outer_global_indices[region])} outer bins, "
                    f"expected {n_outer}"
                )

        # Chebyshev basis matrix from bin centres of the smooth axis.
        # x̃ ∈ [-1, 1] with the axis edges mapped to the interval endpoints,
        # matching the WRemnants spectrum regressor's transform_chebyshev.
        ch_name_A, axis_sel_A = next(iter(channel_A.items()))
        axes_A = indata.channel_info[ch_name_A]["axes"]
        smooth_ax_A = next(a for a in axes_A if a.name == smoothing_axis)
        x_centers = np.array(smooth_ax_A.centers)
        edges = np.array(smooth_ax_A.edges)
        x_min, x_max = float(edges[0]), float(edges[-1])
        x_cheby = (
            2.0 * (x_centers - x_min) / (x_max - x_min) - 1.0
            if x_max > x_min
            else np.zeros_like(x_centers)
        )
        # basis[k, s] = T_k(x_cheby[s]),  shape [order+1, n_smooth]
        basis = np.zeros((order + 1, n_smooth))
        basis[0, :] = 1.0
        if order >= 1:
            basis[1, :] = x_cheby
        for k in range(2, order + 1):
            basis[k, :] = 2.0 * x_cheby * basis[k - 1, :] - basis[k - 2, :]
        self.basis = tf.constant(basis, dtype=indata.dtype)

        # MC correction factor for region D, shape [n_outer * n_smooth]
        norm_dense = tf.sparse.to_dense(indata.norm) if indata.sparse else indata.norm
        norm_proc = tf.cast(norm_dense[:, proc_idx], dtype=indata.dtype)

        def _gather_flat(region_outer_idx):
            return tf.gather(norm_proc, np.concatenate(region_outer_idx))

        mc_A = _gather_flat(outer_global_indices["A"])
        mc_B = _gather_flat(outer_global_indices["B"])
        mc_C = _gather_flat(outer_global_indices["C"])
        mc_D = _gather_flat(outer_global_indices["D"])

        mc_B_safe = tf.where(mc_B == 0.0, tf.ones_like(mc_B), mc_B)
        mc_D_safe = tf.where(mc_D == 0.0, tf.ones_like(mc_D), mc_D)
        self.mc_factor_D = mc_A * mc_C / (mc_B_safe * mc_D_safe)

        # Scatter indices in [bin, proc] format, ordered outer × smooth
        def _build_scatter_idx(region_outer_idx):
            return [
                [int(region_outer_idx[m][s]), proc_idx]
                for m in range(n_outer)
                for s in range(n_smooth)
            ]

        self._idx = {r: _build_scatter_idx(outer_global_indices[r]) for r in "ABCD"}

        # Per-region activity flags for non-full mode
        self._active_nonfull = {
            r: bool(
                all(
                    int(outer_global_indices[r][m][s]) < indata.nbins
                    for m in range(n_outer)
                    for s in range(n_smooth)
                )
            )
            for r in "ABCD"
        }

        # Parameter names: region × outer_bin × poly_coeff
        names = []
        for region, ch_dict in zip("ABC", [channel_A, channel_B, channel_C]):
            ch_name, _ = next(iter(ch_dict.items()))
            outer_axes = outer_axes_by_region[region]
            combos = list(itertools.product(*[range(a.size) for a in outer_axes])) or [
                ()
            ]
            for outer_combo in combos:
                if outer_axes:
                    outer_label = "_".join(
                        f"{a.name}{i}" for a, i in zip(outer_axes, outer_combo)
                    )
                else:
                    outer_label = "outer0"
                for k in range(order + 1):
                    names.append(
                        f"{abcd_process}_{ch_name}_{region}_{outer_label}_poly{k}".encode()
                    )

        # Model attributes
        self.npoi = 0
        self.npou = 3 * n_outer * (order + 1)
        self.params = np.array(names)
        self.is_linear = False
        self.allowNegativeParam = False
        # Default: all coefficients 0 → exp(0) = 1 (MC template as starting point)
        self.xparamdefault = tf.zeros([self.nparams], dtype=indata.dtype)

    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        """Parse CLI arguments for SmoothABCD.

        Syntax:
            --paramModel SmoothABCD <axis> [order:N] <process> \\
                         <ch_A> [ax:val ...] <ch_B> [ax:val ...] \\
                         <ch_C> [ax:val ...] <ch_D> [ax:val ...]
        """
        if len(args) < 6:
            raise ValueError(
                "SmoothABCD expects: axis [order:N] process "
                "ch_A [ax:val ...] ch_B [ax:val ...] ch_C [ax:val ...] ch_D [ax:val ...]"
            )
        tokens = list(args)
        smoothing_axis = tokens.pop(0)

        order = 1
        if tokens and tokens[0].startswith("order:"):
            order = int(tokens.pop(0).split(":", 1)[1])

        if not tokens:
            raise ValueError("SmoothABCD: expected process name after axis/order")
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
            smoothing_axis,
            process,
            channel_A,
            channel_B,
            channel_C,
            channel_D,
            order=order,
            **kwargs,
        )

    def compute(self, param, full=False):
        """Compute per-bin, per-process yield scale factors.

        Args:
            param: 1D tensor of length 3 * n_outer * (order+1).
            full: if True return [nbinsfull, nproc]; else [nbins, nproc].
        """
        n_coeffs = self.n_outer * (self.order + 1)

        # Reshape to [n_outer, order+1] for each region
        params_A = tf.reshape(param[:n_coeffs], [self.n_outer, self.order + 1])
        params_B = tf.reshape(
            param[n_coeffs : 2 * n_coeffs], [self.n_outer, self.order + 1]
        )
        params_C = tf.reshape(param[2 * n_coeffs :], [self.n_outer, self.order + 1])

        # Evaluate exp(polynomial): [n_outer, n_smooth]
        # self.basis is a constant of shape [order+1, n_smooth] with Chebyshev T_k rows
        a = tf.exp(tf.matmul(params_A, self.basis))
        b = tf.exp(tf.matmul(params_B, self.basis))
        c = tf.exp(tf.matmul(params_C, self.basis))

        # Flatten to [n_outer * n_smooth] = [n_total_bins per region]
        a_flat = tf.reshape(a, [-1])
        b_flat = tf.reshape(b, [-1])
        c_flat = tf.reshape(c, [-1])

        b_safe = tf.where(b_flat == 0.0, tf.ones_like(b_flat) * 1e-10, b_flat)
        d_flat = a_flat * c_flat / b_safe * self.mc_factor_D

        nbins = self.indata.nbinsfull if full else self.indata.nbins
        rnorm = tf.ones([nbins, self.indata.nproc], dtype=self.indata.dtype)

        # Python-level control flow resolved at @tf.function trace time
        indices = []
        updates = []
        for region, vals in [
            ("A", a_flat),
            ("B", b_flat),
            ("C", c_flat),
            ("D", d_flat),
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
