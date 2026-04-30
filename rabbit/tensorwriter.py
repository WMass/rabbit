import math
import os
from collections import defaultdict

import h5py
import numpy as np
from wums.sparse_hist import SparseHist  # noqa: F401  re-exported for convenience

from rabbit import common, h5pyutils_write

from wums import ioutils, logging  # isort: skip

logger = logging.child_logger(__name__)


class TensorWriter:
    def __init__(
        self,
        sparse=False,
        systematic_type="log_normal",
        allow_negative_expectation=False,
        add_bin_by_bin_stat_to_data_cov=False,
    ):
        self.allow_negative_expectation = allow_negative_expectation

        self.systematic_type = systematic_type

        self.symmetric_tensor = True  # If all shape systematics are symmetrized the systematic tensor is symmetric leading to reduced memory and improved efficiency
        self.add_bin_by_bin_stat_to_data_cov = add_bin_by_bin_stat_to_data_cov  # add bin by bin statistical uncertainty to data covariance matrix

        self.signals = set()
        self.bkgs = set()

        self.channels = {}
        self.nbinschan = {}
        self.pseudodata_names = set()

        self.dict_systgroups = defaultdict(lambda: set())

        self.systsstandard = set()
        self.systsnoi = set()
        self.systsnoconstraint = set()
        self.systscovariance = set()

        self.sparse = sparse
        self.idxdtype = "int64"

        # temporary data
        self.dict_data_obs = {}  # [channel]
        self.dict_data_var = {}  # [channel]
        self.data_covariance = None
        self.dict_pseudodata = {}  # [channel][pseudodata]
        self.dict_norm = {}  # [channel][process]
        self.dict_sumw2 = {}  # [channel][process]
        self.dict_logkavg = {}  # [channel][proc][syst]
        self.dict_logkhalfdiff = {}  # [channel][proc][syst]
        self.dict_logkavg_indices = {}
        self.dict_logkhalfdiff_indices = {}
        self.dict_beta_variations = {}  # [channel][syst][process]

        self.has_beta_variations = False

        # External likelihood terms. Each term is a dict with keys:
        #   name: identifier
        #   params: 1D ndarray of parameter name strings; both grad and hess
        #     refer to this same parameter list in the same order
        #   grad_values: 1D float ndarray (length == len(params)) or None
        #   hess_dense: 2D float ndarray of shape (len(params), len(params)) or None
        #   hess_sparse: tuple (rows, cols, values) for sparse hessian or None
        # Exactly one of hess_dense / hess_sparse may be set, or neither
        # (gradient-only term). Parameter names are resolved against the full
        # fit parameter list (POIs + systs) at fit time. See
        # add_external_likelihood_term for details.
        self.external_terms = []

        self.clipSystVariations = False
        if self.clipSystVariations > 0.0:
            self.clip = np.abs(np.log(self.clipSystVariations))

        self.logkepsilon = math.log(
            1e-3
        )  # numerical cutoff in case of zeros in systematic variations

        # settings for writing out hdf5 files
        self.dtype = "float64"
        self.chunkSize = 4 * 1024**2

    @staticmethod
    def _issparse(h):
        """Check if h is a scipy sparse array/matrix."""
        return hasattr(h, "toarray") and hasattr(h, "tocoo")

    @staticmethod
    def _sparse_to_flat_csr(h, dtype, flow=False):
        """Flatten a scipy sparse array/matrix to CSR with shape (1, prod(shape)).

        For SparseHist inputs, forwards ``flow`` to ``h.to_flat_csr`` so the
        wrapper can convert from its internal with-flow layout to the requested
        layout. For raw scipy sparse inputs, the row-major flatten of ``h.shape``
        is used directly (the user is responsible for matching the channel layout).

        The returned CSR array has sorted indices suitable for searchsorted lookups.
        """
        if hasattr(h, "to_flat_csr"):
            return h.to_flat_csr(dtype, flow=flow)

        import scipy.sparse

        size = int(np.prod(h.shape))
        coo = scipy.sparse.coo_array(h)
        if coo.ndim == 2:
            flat_indices = np.ravel_multi_index((coo.row, coo.col), h.shape)
        elif coo.ndim == 1:
            flat_indices = coo.coords[0]
        else:
            raise ValueError(
                f"Unsupported dimensionality {coo.ndim} for scipy sparse input"
            )
        sort_order = np.argsort(flat_indices)
        sorted_indices = flat_indices[sort_order].astype(np.int32)
        sorted_data = coo.data[sort_order].astype(dtype)
        indptr = np.array([0, len(sorted_data)], dtype=np.int32)
        return scipy.sparse.csr_array(
            (sorted_data, sorted_indices, indptr), shape=(1, size)
        )

    def _to_flat_dense(self, h, flow=False):
        """Convert any array-like (including scipy sparse) to a flat dense numpy array.

        For SparseHist inputs, ``flow`` selects the with-flow or no-flow layout.
        """
        if isinstance(h, SparseHist):
            return np.asarray(h.toarray(flow=flow)).flatten().astype(self.dtype)
        if self._issparse(h):
            return np.asarray(h.toarray()).flatten().astype(self.dtype)
        return np.asarray(h).flatten().astype(self.dtype)

    def get_flat_values(self, h, flow=False):
        if hasattr(h, "values"):
            values = h.values(flow=flow)
        elif isinstance(h, SparseHist):
            values = h.toarray(flow=flow)
        elif self._issparse(h):
            values = np.asarray(h.toarray())
        else:
            values = h
        return np.asarray(values).flatten().astype(self.dtype)

    def get_flat_variances(self, h, flow=False):
        if hasattr(h, "variances"):
            variances = h.variances(flow=flow)
        elif isinstance(h, SparseHist):
            variances = h.toarray(flow=flow)
        elif self._issparse(h):
            variances = np.asarray(h.toarray())
        else:
            variances = h

        variances = np.asarray(variances).flatten().astype(self.dtype)
        if (variances < 0.0).any():
            raise ValueError("Negative variances encountered")

        return variances

    def add_data(self, h, channel="ch0", variances=None):
        self._check_hist_and_channel(h, channel)
        if channel in self.dict_data_obs.keys():
            raise RuntimeError(f"Data histogram for channel '{channel}' already set.")
        self.dict_data_obs[channel] = self.get_flat_values(h)
        self.dict_data_var[channel] = self.get_flat_variances(
            h if variances is None else variances
        )

    def add_data_covariance(self, cov):
        self.data_covariance = cov if isinstance(cov, np.ndarray) else cov.values()

    def add_pseudodata(self, h, name=None, channel="ch0"):
        self._check_hist_and_channel(h, channel)
        if name is None:
            name = f"pseudodata_{len(self.pseudodata_names)}"
        self.pseudodata_names.add(name)
        if channel not in self.dict_pseudodata.keys():
            self.dict_pseudodata[channel] = {}
        if name in self.dict_pseudodata[channel].keys():
            raise RuntimeError(
                f"Pseudodata histogram '{name}' for channel '{channel}' already set."
            )
        self.dict_pseudodata[channel][name] = self.get_flat_values(h)

    def add_process(self, h, name, channel="ch0", signal=False, variances=None):
        self._check_hist_and_channel(h, channel)

        if name in self.dict_norm[channel].keys():
            raise RuntimeError(
                f"Nominal histogram for process '{name}' for channel '{channel}' already set."
            )

        if signal:
            self.signals.add(name)
        else:
            self.bkgs.add(name)

        self.dict_logkavg[channel][name] = {}
        self.dict_logkhalfdiff[channel][name] = {}
        if self.sparse:
            self.dict_logkavg_indices[channel][name] = {}
            self.dict_logkhalfdiff_indices[channel][name] = {}

        flow = self.channels[channel]["flow"]

        if self.sparse and self._issparse(h):
            # Store as flat CSR, avoiding full dense conversion
            norm = self._sparse_to_flat_csr(h, self.dtype, flow=flow)
            if not np.all(np.isfinite(norm.data)):
                raise RuntimeError(
                    f"NaN or Inf values encountered in nominal histogram for {name}!"
                )
            if not self.allow_negative_expectation:
                has_negative = np.any(norm.data < 0.0)
                if has_negative:
                    norm = norm.copy()
                    norm.data[:] = np.maximum(norm.data, 0.0)
                    norm.eliminate_zeros()
        else:
            norm = self.get_flat_values(h, flow)
            if not self.allow_negative_expectation:
                norm = np.maximum(norm, 0.0)
            if not np.all(np.isfinite(norm)):
                raise RuntimeError(
                    f"{len(norm)-sum(np.isfinite(norm))} NaN or Inf values encountered in nominal histogram for {name}!"
                )

        # variances are always stored dense (needed for sumw2 output assembly)
        if variances is not None:
            sumw2 = self.get_flat_variances(variances, flow)
        elif self._issparse(h):
            sumw2 = self._to_flat_dense(h, flow=flow)
        else:
            sumw2 = self.get_flat_variances(h, flow)

        if not np.all(np.isfinite(sumw2)):
            raise RuntimeError(
                f"{len(sumw2)-sum(np.isfinite(sumw2))} NaN or Inf values encountered in variances for {name}!"
            )

        self.dict_norm[channel][name] = norm
        self.dict_sumw2[channel][name] = sumw2

    def add_channel(self, axes, name=None, masked=False, flow=False):
        if flow and masked is False:
            raise NotImplementedError(
                "Keeping underflow/overflow is currently only supported for masked channels"
            )
        if name is None:
            name = f"ch{len(self.channels)}"
        logger.debug(f"Add new channel {name}")
        ibins = np.prod([a.extent if flow else a.size for a in axes])
        self.nbinschan[name] = ibins
        self.dict_norm[name] = {}
        self.dict_sumw2[name] = {}
        self.dict_beta_variations[name] = {}

        # add masked channels last
        this_channel = {"axes": [a for a in axes], "masked": masked, "flow": flow}
        if masked:
            self.channels[name] = this_channel
        else:
            self.channels = {
                **{k: v for k, v in self.channels.items() if not v["masked"]},
                name: this_channel,
                **{k: v for k, v in self.channels.items() if v["masked"]},
            }

        self.dict_logkavg[name] = {}
        self.dict_logkhalfdiff[name] = {}
        if self.sparse:
            self.dict_logkavg_indices[name] = {}
            self.dict_logkhalfdiff_indices[name] = {}

    def _check_hist_and_channel(self, h, channel):

        if channel not in self.channels.keys():
            raise RuntimeError(f"Channel {channel} not known!")

        if hasattr(h, "axes"):
            axes = [a for a in h.axes]
            channel_axes = self.channels[channel]["axes"]

            if not all(np.allclose(a, axes[i]) for i, a in enumerate(channel_axes)):
                raise RuntimeError(f"""
                    Histogram axes different have different edges from channel axes of channel {channel}
                    \nHistogram axes: {[a.edges for a in axes]}
                    \nChannel axes: {[a.edges for a in channel_axes]}
                    """)
        elif self._issparse(h):
            size_in = int(np.prod(h.shape))
            size_this = int(np.prod([len(a) for a in self.channels[channel]["axes"]]))
            if size_in != size_this:
                raise RuntimeError(
                    f"Total number of elements in sparse input different from channel size '{size_in}' != '{size_this}'"
                )
        else:
            shape_in = h.shape
            shape_this = tuple([len(a) for a in self.channels[channel]["axes"]])
            if shape_in != shape_this:
                raise RuntimeError(
                    f"Shape of input object different from channel axes '{shape_in}' != '{shape_this}'"
                )

    def _compute_asym_syst(
        self,
        logkup,
        logkdown,
        name,
        process,
        channel,
        symmetrize="average",
        add_to_data_covariance=False,
        _sparse_info=None,
        **kargs,
    ):
        """Compute symmetrized logk from asymmetric up/down variations.

        When _sparse_info is set to (nnz_indices, size), logkup/logkdown are value
        arrays at those indices and internal book_logk calls use sparse tuples.
        """
        var_name_out = name

        def _wrap(vals):
            """Wrap values as sparse tuple if in sparse mode."""
            if _sparse_info is not None:
                return (_sparse_info[0], vals, _sparse_info[1])
            return vals

        if symmetrize == "conservative":
            # symmetrize by largest magnitude of up and down variations
            logkavg_proc = np.where(
                np.abs(logkup) > np.abs(logkdown),
                logkup,
                logkdown,
            )
        elif symmetrize == "average":
            # symmetrize by average of up and down variations
            logkavg_proc = 0.5 * (logkup + logkdown)
        elif symmetrize in ["linear", "quadratic"]:
            # "linear" corresponds to a piecewise linear dependence of logk on theta
            # while "quadratic" corresponds to a quadratic dependence and leads
            # to a large variance
            diff_fact = np.sqrt(3.0) if symmetrize == "quadratic" else 1.0

            # split asymmetric variation into two symmetric variations
            logkavg_proc = 0.5 * (logkup + logkdown)
            logkdiffavg_proc = 0.5 * diff_fact * (logkup - logkdown)

            var_name_out = name + "SymAvg"
            var_name_out_diff = name + "SymDiff"

            # special case, book the extra systematic
            self.book_logk_avg(
                _wrap(logkdiffavg_proc), channel, process, var_name_out_diff
            )
            self.book_systematic(
                var_name_out_diff,
                add_to_data_covariance=add_to_data_covariance,
                **kargs,
            )
        else:
            if add_to_data_covariance:
                raise RuntimeError(
                    "add_to_data_covariance requires symmetric uncertainties"
                )

            self.symmetric_tensor = False

            logkavg_proc = 0.5 * (logkup + logkdown)
            logkhalfdiff_proc = 0.5 * (logkup - logkdown)

            self.book_logk_halfdiff(_wrap(logkhalfdiff_proc), channel, process, name)
        logkup = None
        logkdown = None

        return logkavg_proc, var_name_out

    def add_norm_systematic(
        self,
        name,
        process,
        channel,
        uncertainty,
        add_to_data_covariance=False,
        groups=None,
        symmetrize="average",
        **kargs,
    ):
        if not isinstance(process, (list, tuple, np.ndarray)):
            process = [process]

        if not isinstance(uncertainty, (list, tuple, np.ndarray)):
            uncertainty = [uncertainty]

        if len(uncertainty) != 1 and len(process) != len(uncertainty):
            raise RuntimeError(
                f"uncertainty must be either a scalar or list with the same length as the list of processes but len(process) = {len(process)} and len(uncertainty) = {len(uncertainty)}"
            )

        var_name_out = name

        systematic_type = "normal" if add_to_data_covariance else self.systematic_type

        for p, u in zip(process, uncertainty):
            norm = self.dict_norm[channel][p]

            if self._issparse(norm):
                # Sparse norm path: compute logk at nonzero positions only
                norm_vals = norm.data
                nnz_idx = norm.indices
                size = norm.shape[1]

                if isinstance(u, (list, tuple, np.ndarray)):
                    if len(u) != 2:
                        raise RuntimeError(
                            f"lnN uncertainty can only be a scalar for a symmetric or a list of 2 elements for asymmetric lnN uncertainties, but got a list of {len(u)} elements"
                        )
                    logkup_proc = self._get_logk_sparse(
                        norm_vals * u[0], norm_vals, 1.0, systematic_type
                    )
                    logkdown_proc = -self._get_logk_sparse(
                        norm_vals * u[1], norm_vals, 1.0, systematic_type
                    )
                    logkavg_proc, var_name_out = self._compute_asym_syst(
                        logkup_proc,
                        logkdown_proc,
                        name,
                        process,
                        channel,
                        symmetrize=symmetrize,
                        add_to_data_covariance=add_to_data_covariance,
                        _sparse_info=(nnz_idx, size),
                        **kargs,
                    )
                else:
                    logkavg_proc = self._get_logk_sparse(
                        norm_vals * u, norm_vals, 1.0, systematic_type
                    )

                self.book_logk_avg(
                    (nnz_idx, logkavg_proc, size), channel, p, var_name_out
                )
            else:
                if isinstance(u, (list, tuple, np.ndarray)):
                    if len(u) != 2:
                        raise RuntimeError(
                            f"lnN uncertainty can only be a scalar for a symmetric or a list of 2 elements for asymmetric lnN uncertainties, but got a list of {len(u)} elements"
                        )
                    # asymmetric lnN uncertainty
                    syst_up = norm * u[0]
                    syst_down = norm * u[1]

                    logkup_proc = self.get_logk(
                        syst_up, norm, systematic_type=systematic_type
                    )
                    logkdown_proc = -self.get_logk(
                        syst_down, norm, systematic_type=systematic_type
                    )

                    logkavg_proc, var_name_out = self._compute_asym_syst(
                        logkup_proc,
                        logkdown_proc,
                        name,
                        process,
                        channel,
                        symmetrize=symmetrize,
                        add_to_data_covariance=add_to_data_covariance,
                        **kargs,
                    )
                else:
                    syst = norm * u
                    logkavg_proc = self.get_logk(
                        syst, norm, systematic_type=systematic_type
                    )

                self.book_logk_avg(logkavg_proc, channel, p, var_name_out)

        self.book_systematic(
            var_name_out,
            groups=groups,
            add_to_data_covariance=add_to_data_covariance,
            **kargs,
        )

    def _add_systematic_sparse(
        self,
        h,
        name,
        process,
        channel,
        norm,
        kfactor,
        mirror,
        symmetrize,
        add_to_data_covariance,
        as_difference,
        **kargs,
    ):
        """Sparse path for add_systematic when norm is stored as scipy sparse CSR.

        Computes logk only at norm's nonzero positions, avoiding full-size dense
        intermediate arrays.  The logk result is a tuple (indices, values, size).
        """
        systematic_type = "normal" if add_to_data_covariance else self.systematic_type
        flow = self.channels[channel]["flow"]
        nnz_idx = norm.indices
        norm_vals = norm.data
        size = norm.shape[1]

        var_name_out = name

        if isinstance(h, (list, tuple)):
            self._check_hist_and_channel(h[0], channel)
            self._check_hist_and_channel(h[1], channel)

            syst_up_vals = self._get_syst_at_norm_nnz(h[0], norm, flow)
            syst_down_vals = self._get_syst_at_norm_nnz(h[1], norm, flow)

            if as_difference:
                syst_up_vals = norm_vals + syst_up_vals
                syst_down_vals = norm_vals + syst_down_vals

            logkup_vals = self._get_logk_sparse(
                syst_up_vals, norm_vals, kfactor, systematic_type
            )
            logkdown_vals = -self._get_logk_sparse(
                syst_down_vals, norm_vals, kfactor, systematic_type
            )

            logkavg_vals, var_name_out = self._compute_asym_syst(
                logkup_vals,
                logkdown_vals,
                name,
                process,
                channel,
                symmetrize,
                add_to_data_covariance,
                _sparse_info=(nnz_idx, size),
                **kargs,
            )
        elif mirror:
            self._check_hist_and_channel(h, channel)

            syst_vals = self._get_syst_at_norm_nnz(h, norm, flow)

            if as_difference:
                syst_vals = norm_vals + syst_vals

            logkavg_vals = self._get_logk_sparse(
                syst_vals, norm_vals, kfactor, systematic_type
            )
        else:
            raise RuntimeError(
                "Only one histogram given but mirror=False, can not construct a variation"
            )

        logkavg_proc = (nnz_idx, logkavg_vals, size)
        self.book_logk_avg(logkavg_proc, channel, process, var_name_out)
        self.book_systematic(
            var_name_out, add_to_data_covariance=add_to_data_covariance, **kargs
        )

    @staticmethod
    def _bin_label(ax, idx):
        """Return a string label for a hist axis bin, preferring string values."""
        v = ax.value(idx)
        if isinstance(v, (str, bytes)):
            return v.decode() if isinstance(v, bytes) else v
        return str(idx)

    @staticmethod
    def _make_empty_sparsehist(axes, size):
        """Construct an empty SparseHist over ``axes`` with the given flat size."""
        import scipy.sparse

        empty_csr = scipy.sparse.csr_array(
            (
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.int64),
                np.array([0, 0], dtype=np.int64),
            ),
            shape=(1, int(size)),
        )
        return SparseHist(empty_csr, axes)

    def _sparse_per_syst_slices(self, h, extra_axes, extra_axis_names, keep_axes):
        """Yield ``(linear_idx, sub_h)`` for every bin combination on the extra axes.

        Single-pass O(nnz log nnz) algorithm: extract all entries via the
        with-flow flat layout, compute a linear syst index from the extra-axis
        coordinates, sort once, then iterate contiguous per-bin runs. Empty
        slots yield an empty SparseHist over the kept axes so that callers can
        still book the corresponding systematic name (allowing it to be
        constrained externally even when the template variation is exactly
        zero).
        """
        import scipy.sparse

        extra_sizes = [int(len(a)) for a in extra_axes]
        n_total = int(np.prod(extra_sizes))
        keep_extent = tuple(int(a.extent) for a in keep_axes)
        keep_size = int(np.prod(keep_extent)) if keep_axes else 1

        h_axis_names = [a.name for a in h.axes]
        extra_positions = [h_axis_names.index(n) for n in extra_axis_names]
        keep_positions = [
            i for i in range(len(h_axis_names)) if i not in extra_positions
        ]

        csr = h.to_flat_csr(np.float64, flow=True)
        flat_idx = np.asarray(csr.indices, dtype=np.int64)
        values = np.asarray(csr.data, dtype=np.float64)

        if len(flat_idx) == 0:
            for linear_idx in range(n_total):
                yield linear_idx, self._make_empty_sparsehist(keep_axes, keep_size)
            return

        # SparseHist internal flat indices are in the with-flow layout; use
        # the per-axis extents (h.axes.extent, matching hist) — h.shape is
        # the no-flow shape and would unravel to the wrong coordinates.
        multi = np.unravel_index(flat_idx, h.axes.extent)

        # Drop entries that fall in flow bins of any extra axis (we only
        # iterate over the regular bins of those axes, matching the existing
        # multi-systematic dispatch convention).
        valid = np.ones(len(flat_idx), dtype=bool)
        per_extra_idx = []
        for ax_pos in extra_positions:
            ax = h.axes[ax_pos]
            u = SparseHist._underflow_offset(ax)
            s = int(len(ax))
            valid &= (multi[ax_pos] >= u) & (multi[ax_pos] < u + s)
            per_extra_idx.append(multi[ax_pos] - u)

        if not valid.all():
            multi = tuple(m[valid] for m in multi)
            values = values[valid]
            per_extra_idx = [arr[valid] for arr in per_extra_idx]

        if len(extra_positions) == 1:
            syst_linear = per_extra_idx[0]
        else:
            syst_linear = np.ravel_multi_index(per_extra_idx, extra_sizes)

        sort_order = np.argsort(syst_linear, kind="stable")
        sorted_syst = syst_linear[sort_order]
        sorted_values = values[sort_order]
        sorted_keep_multi = tuple(multi[i][sort_order] for i in keep_positions)

        boundaries = np.searchsorted(sorted_syst, np.arange(n_total + 1), side="left")

        for linear_idx in range(n_total):
            start = int(boundaries[linear_idx])
            end = int(boundaries[linear_idx + 1])
            if start == end:
                yield linear_idx, self._make_empty_sparsehist(keep_axes, keep_size)
                continue

            sub_keep_multi = tuple(arr[start:end] for arr in sorted_keep_multi)
            if len(keep_extent) == 1:
                sub_flat = sub_keep_multi[0]
            else:
                sub_flat = np.ravel_multi_index(sub_keep_multi, keep_extent)
            sub_vals = sorted_values[start:end]
            order = np.argsort(sub_flat)
            sub_csr = scipy.sparse.csr_array(
                (
                    sub_vals[order].astype(np.float64),
                    sub_flat[order].astype(np.int64),
                    np.array([0, len(sub_vals)], dtype=np.int64),
                ),
                shape=(1, keep_size),
            )
            yield linear_idx, SparseHist(sub_csr, keep_axes)

    def _get_systematic_slices(self, h, name, channel, syst_axes=None):
        """Detect extra axes in h beyond the channel and return list of (sub_name, sub_h) slices.

        Returns None if there are no extra axes (i.e. single-systematic case).

        h may be a single histogram or a list/tuple of two (up/down) histograms.
        Both elements of a pair must share the same extra-axis structure.

        For ``SparseHist`` inputs, an efficient single-pass algorithm is used
        that pre-extracts the underlying flat representation, partitions
        entries by their extra-axis indices via a global sort, and then yields
        one sub-``SparseHist`` per bin combination on the extra axes. This is
        O(nnz log nnz) total instead of O(nnz) per slice, and it always emits
        a (possibly empty) sub-hist for *every* combination so that the
        downstream booking sees all systematic names even where the
        per-bin variation is identically zero.

        syst_axes:
          - None (default): auto-detect any axes in h not present in the channel
          - list of axis names: use exactly these axes as systematic axes
          - empty list: disable detection entirely
        """
        extra_info = self._detect_extra_syst_axes(h, channel, syst_axes)
        if extra_info is None:
            return None
        h_ref, is_pair, extra_axis_names = extra_info

        extra_axes = [h_ref.axes[n] for n in extra_axis_names]
        extra_sizes = [len(a) for a in extra_axes]
        keep_axes_ref = [a for a in h_ref.axes if a.name not in extra_axis_names]

        # Fast path for SparseHist inputs (the slow per-slice loop below would
        # be O(nnz) per slice, which is prohibitive for large syst axes).
        if isinstance(h_ref, SparseHist):
            if is_pair:
                if not isinstance(h[1], SparseHist):
                    raise TypeError(
                        "Mixed SparseHist/non-SparseHist pair not supported"
                    )
                up_iter = self._sparse_per_syst_slices(
                    h[0], extra_axes, extra_axis_names, keep_axes_ref
                )
                dn_iter = self._sparse_per_syst_slices(
                    h[1], extra_axes, extra_axis_names, keep_axes_ref
                )
                paired = zip(up_iter, dn_iter)
            else:
                paired = (
                    (item, None)
                    for item in self._sparse_per_syst_slices(
                        h, extra_axes, extra_axis_names, keep_axes_ref
                    )
                )

            slices = []
            for linear_idx in range(int(np.prod(extra_sizes))):
                if is_pair:
                    (lu, sub_up), (ld, sub_dn) = next(paired)
                    assert lu == linear_idx and ld == linear_idx
                    sub_h = [sub_up, sub_dn]
                else:
                    (li, sub), _ = next(paired)
                    assert li == linear_idx
                    sub_h = sub
                # Decode the extra-axis multi-dim index for label construction
                if len(extra_axes) == 1:
                    idx_tuple = (linear_idx,)
                else:
                    idx_tuple = tuple(
                        int(x) for x in np.unravel_index(linear_idx, extra_sizes)
                    )
                labels = [
                    self._bin_label(ax, i) for ax, i in zip(extra_axes, idx_tuple)
                ]
                sub_name = "_".join([name, *labels])
                slices.append((sub_name, sub_h))
            return slices

        # Generic path: hist-like object using its own __getitem__ slicing.
        import itertools

        slices = []
        for idx_tuple in itertools.product(*[range(s) for s in extra_sizes]):
            labels = [self._bin_label(ax, i) for ax, i in zip(extra_axes, idx_tuple)]
            sub_name = "_".join([name, *labels])

            slice_dict = {n: i for n, i in zip(extra_axis_names, idx_tuple)}

            if is_pair:
                sub_h = [h[0][slice_dict], h[1][slice_dict]]
            else:
                sub_h = h[slice_dict]

            slices.append((sub_name, sub_h))

        return slices

    def _detect_extra_syst_axes(self, h, channel, syst_axes):
        """Determine the extra (systematic) axes of ``h`` for a given channel.

        Returns a tuple ``(h_ref, is_pair, extra_axis_names)`` if there are
        extra axes, or ``None`` otherwise. Shared by ``_get_systematic_slices``
        and the batched fast path so the detection logic stays in one place.
        """
        if syst_axes is not None and len(syst_axes) == 0:
            return None

        if isinstance(h, (list, tuple)):
            h_ref = h[0]
            is_pair = True
        else:
            h_ref = h
            is_pair = False

        if not hasattr(h_ref, "axes"):
            return None

        channel_axes = self.channels[channel]["axes"]

        # No extra axes possible when the axis count already matches the
        # channel. Short-circuit before touching ``.name`` so plain
        # boost_histogram inputs (whose axes have no ``name`` attribute)
        # still work in the common single-systematic case.
        if syst_axes is None and len(h_ref.axes) == len(channel_axes):
            return None

        h_axis_names = [getattr(a, "name", None) for a in h_ref.axes]
        channel_axis_names = [getattr(a, "name", None) for a in channel_axes]

        if syst_axes is None:
            if any(n is None for n in h_axis_names):
                raise RuntimeError(
                    "Cannot auto-detect systematic axes: histogram has more axes "
                    "than the channel but the extra axes are unnamed. Use named "
                    "axes (e.g. hist.Hist) or pass syst_axes explicitly."
                )
            extra_axis_names = [n for n in h_axis_names if n not in channel_axis_names]
        else:
            for n in syst_axes:
                if n not in h_axis_names:
                    raise RuntimeError(
                        f"Requested systematic axis '{n}' not found in histogram axes {h_axis_names}"
                    )
                if n in channel_axis_names:
                    raise RuntimeError(
                        f"Systematic axis '{n}' overlaps with channel axes {channel_axis_names}"
                    )
            extra_axis_names = list(syst_axes)

        if not extra_axis_names:
            return None

        return h_ref, is_pair, extra_axis_names

    def _add_systematics_sparsehist_batched(
        self,
        h,
        name,
        process,
        channel,
        kfactor,
        as_difference,
        extra_axis_names,
        **kargs,
    ):
        """Vectorized booking of one shape systematic per bin combination on
        the extra axes of a single ``SparseHist`` input.

        Used as a fast path for the multi-systematic dispatch when the input
        is a single (non-paired) :class:`wums.sparse_hist.SparseHist` with at
        least one extra axis beyond the channel axes and ``as_difference=True``.

        All per-entry math (channel-flat-index computation, norm lookup,
        sign-flip-protected logk evaluation) is done once over the entire
        ~nnz array using vectorised numpy operations. The result is then
        partitioned by linear systematic index via a single ``argsort`` +
        ``searchsorted`` and bulk-inserted into ``dict_logkavg`` /
        ``dict_logkavg_indices`` (sparse storage) or ``dict_logkavg`` (dense
        storage). Empty bin combinations on the extra axes still get an entry
        and a corresponding ``book_systematic`` call so they appear in the
        fit parameter list.
        """
        import scipy.sparse  # noqa: F401  used implicitly via SparseHist methods

        chan_info = self.channels[channel]
        chan_flow = chan_info["flow"]
        channel_axes_obj = chan_info["axes"]
        channel_axis_names = [a.name for a in channel_axes_obj]

        h_axis_names = [a.name for a in h.axes]
        for n in channel_axis_names:
            if n not in h_axis_names:
                raise RuntimeError(
                    f"Channel axis '{n}' not found in histogram axes {h_axis_names}"
                )

        extra_positions = [h_axis_names.index(n) for n in extra_axis_names]
        keep_positions = [h_axis_names.index(n) for n in channel_axis_names]
        keep_axes = [h.axes[i] for i in keep_positions]
        extra_axes = [h.axes[i] for i in extra_positions]

        extra_sizes = [int(len(a)) for a in extra_axes]
        n_total_systs = int(np.prod(extra_sizes))

        keep_extent = tuple(int(a.extent) for a in keep_axes)
        keep_no_flow = tuple(int(len(a)) for a in keep_axes)
        target_size = (
            int(np.prod(keep_extent)) if chan_flow else int(np.prod(keep_no_flow))
        )

        norm = self.dict_norm[channel][process]
        norm_is_sparse = self._issparse(norm)
        systematic_type = self.systematic_type

        # ---- Step 1: get h's flat with-flow (indices, values) ----
        # Access the SparseHist's internal flat buffers directly to skip the
        # O(nnz log nnz) sort that ``to_flat_csr`` would otherwise do (we do
        # our own sort later, by syst index, so pre-sorting by flat index
        # would be wasted work).
        flat_idx = np.asarray(h._flat_indices, dtype=np.int64)
        delta_vals = np.asarray(h._values, dtype=np.float64)

        if len(flat_idx) > 0:
            # SparseHist internal flat indices are in the with-flow layout;
            # use h.axes.extent (matching hist) — h.shape is the no-flow
            # shape and would unravel to the wrong coordinates.
            multi = np.unravel_index(flat_idx, h.axes.extent)
            # free flat_idx — we only need the per-axis multi-dim arrays now
            flat_idx = None

            # ---- Step 2: drop entries in flow bins ----
            # Build the validity mask in a single pass over all relevant axes
            # (extra axes always, channel axes only if the channel is no-flow).
            if chan_flow:
                check_positions = list(extra_positions)
            else:
                check_positions = list(extra_positions) + list(keep_positions)

            valid = None
            for ax_pos in check_positions:
                ax = h.axes[ax_pos]
                u = SparseHist._underflow_offset(ax)
                s = int(len(ax))
                ax_arr = multi[ax_pos]
                if u == 0:
                    cond = ax_arr < s
                else:
                    # 1 underflow bin: valid = ax_arr in [1, 1+s)
                    cond = (ax_arr >= u) & (ax_arr < u + s)
                if valid is None:
                    valid = cond
                else:
                    valid &= cond

            if valid is not None and not valid.all():
                multi = tuple(m[valid] for m in multi)
                delta_vals = delta_vals[valid]
            valid = None  # free

            # ---- Step 3: compute linear systematic index from extra axes ----
            if len(extra_positions) == 1:
                ax_pos = extra_positions[0]
                u = SparseHist._underflow_offset(h.axes[ax_pos])
                if u == 0:
                    syst_linear = multi[ax_pos].astype(np.int64, copy=False)
                else:
                    syst_linear = (multi[ax_pos] - u).astype(np.int64, copy=False)
            else:
                per = []
                for ax_pos in extra_positions:
                    u = SparseHist._underflow_offset(h.axes[ax_pos])
                    per.append(multi[ax_pos] - u if u else multi[ax_pos])
                syst_linear = np.ravel_multi_index(per, extra_sizes)

            # ---- Step 4: compute channel flat index in target layout ----
            if chan_flow:
                chan_flat = np.ravel_multi_index(
                    tuple(multi[i] for i in keep_positions), keep_extent
                )
            else:
                chan_no_flow_multi = []
                for ax_pos in keep_positions:
                    u = SparseHist._underflow_offset(h.axes[ax_pos])
                    chan_no_flow_multi.append(multi[ax_pos] - u if u else multi[ax_pos])
                chan_flat = np.ravel_multi_index(chan_no_flow_multi, keep_no_flow)
            multi = None  # free the per-axis arrays; we no longer need them

            # ---- Step 5: look up norm at chan_flat; drop where norm == 0 ----
            if norm_is_sparse:
                norm_indices_arr = np.asarray(norm.indices, dtype=np.int64)
                norm_data_arr = np.asarray(norm.data, dtype=np.float64)
                positions = np.searchsorted(norm_indices_arr, chan_flat)
                in_range = positions < len(norm_indices_arr)
                match = np.zeros(len(chan_flat), dtype=bool)
                match[in_range] = (
                    norm_indices_arr[positions[in_range]] == chan_flat[in_range]
                )
                chan_flat = chan_flat[match]
                delta_vals = delta_vals[match]
                syst_linear = syst_linear[match]
                norm_at_pos = norm_data_arr[positions[match]]
            else:
                norm_arr = np.asarray(norm, dtype=np.float64)
                norm_at_pos = norm_arr[chan_flat]
                nonzero_norm = norm_at_pos != 0.0
                if not nonzero_norm.all():
                    chan_flat = chan_flat[nonzero_norm]
                    delta_vals = delta_vals[nonzero_norm]
                    syst_linear = syst_linear[nonzero_norm]
                    norm_at_pos = norm_at_pos[nonzero_norm]

            # ---- Step 6: validate finiteness (mirrors get_logk's check) ----
            if not np.all(np.isfinite(delta_vals)):
                n_bad = int((~np.isfinite(delta_vals)).sum())
                raise RuntimeError(
                    f"{n_bad} NaN or Inf values encountered in systematic!"
                )

            # ---- Step 7: compute logk vectorized ----
            syst_at_pos = norm_at_pos + delta_vals  # as_difference=True

            if systematic_type == "log_normal":
                with np.errstate(divide="ignore", invalid="ignore"):
                    logk_vals = kfactor * np.log(syst_at_pos / norm_at_pos)
                logk_vals = np.where(
                    np.equal(np.sign(norm_at_pos * syst_at_pos), 1),
                    logk_vals,
                    self.logkepsilon,
                )
                if self.clipSystVariations > 0.0:
                    logk_vals = np.clip(logk_vals, -self.clip, self.clip)
            elif systematic_type == "normal":
                logk_vals = kfactor * (syst_at_pos - norm_at_pos)
            else:
                raise RuntimeError(
                    f"Invalid systematic_type {systematic_type}, valid choices are 'log_normal' or 'normal'"
                )

            # ---- Step 8: drop exactly-zero logk entries ----
            nonzero_logk = logk_vals != 0.0
            if not nonzero_logk.all():
                chan_flat = chan_flat[nonzero_logk]
                logk_vals = logk_vals[nonzero_logk]
                syst_linear = syst_linear[nonzero_logk]

            # ---- Step 9: sort by linear syst index for partitioning ----
            # Use the default (non-stable) quicksort since the intra-syst
            # order does not affect correctness.
            sort_order = np.argsort(syst_linear)
            sorted_syst = syst_linear[sort_order]
            sorted_chan_flat = chan_flat[sort_order]
            sorted_logk = logk_vals[sort_order]
            sort_order = None
            syst_linear = None
            chan_flat = None
            logk_vals = None

            # ---- Step 10: per-syst boundaries via searchsorted ----
            boundaries = np.searchsorted(
                sorted_syst, np.arange(n_total_systs + 1), side="left"
            )
            sorted_syst = None
        else:
            boundaries = np.zeros(n_total_systs + 1, dtype=np.int64)
            sorted_chan_flat = np.empty(0, dtype=np.int64)
            sorted_logk = np.empty(0, dtype=np.float64)

        # ---- Step 11: bulk insert into the writer's internal storage ----
        dict_logk_proc = self.dict_logkavg[channel][process]
        if self.sparse:
            dict_logk_idx_proc = self.dict_logkavg_indices[channel][process]

        # Pre-compute per-axis label lists once (much faster than calling
        # the generic _bin_label helper per combination, since the value()
        # method on boost_histogram axes has non-trivial per-call overhead).
        def _axis_labels(ax):
            # Check whether the axis stores string categories; if so, decode
            # them in bulk. Otherwise fall back to integer bin indices.
            n = int(len(ax))
            if n == 0:
                return []
            try:
                v0 = ax.value(0)
            except Exception:
                v0 = None
            if isinstance(v0, (str, bytes)):
                out = []
                for i in range(n):
                    v = ax.value(i)
                    if isinstance(v, bytes):
                        v = v.decode()
                    out.append(v)
                return out
            return [str(i) for i in range(n)]

        axis_label_lists = [_axis_labels(ax) for ax in extra_axes]

        if len(extra_axes) == 1:
            labels0 = axis_label_lists[0]
            sub_names = [f"{name}_{labels0[i]}" for i in range(extra_sizes[0])]
        else:
            sub_names = []
            for linear_idx in range(n_total_systs):
                multi_syst = np.unravel_index(linear_idx, extra_sizes)
                labels = [
                    axis_label_lists[k][int(multi_syst[k])]
                    for k in range(len(extra_axes))
                ]
                sub_names.append("_".join([name, *labels]))

        for linear_idx in range(n_total_systs):
            sub_name = sub_names[linear_idx]
            s = int(boundaries[linear_idx])
            e = int(boundaries[linear_idx + 1])

            if self.sparse:
                # Sparse storage: store views into the sorted buffers (they
                # keep the big arrays alive, but that is fine — we need the
                # data anyway and sharing storage avoids a full per-syst copy).
                dict_logk_idx_proc[sub_name] = sorted_chan_flat[s:e].reshape(-1, 1)
                dict_logk_proc[sub_name] = sorted_logk[s:e]
            else:
                # Dense storage: scatter into a full-size logk array
                logk_dense = np.zeros(target_size, dtype=np.float64)
                if e > s:
                    logk_dense[sorted_chan_flat[s:e]] = sorted_logk[s:e]
                dict_logk_proc[sub_name] = logk_dense

            self.book_systematic(sub_name, **kargs)

    def add_systematic(
        self,
        h,
        name,
        process,
        channel,
        kfactor=1,
        mirror=True,
        symmetrize="average",
        add_to_data_covariance=False,
        as_difference=False,
        syst_axes=None,
        **kargs,
    ):
        """
        h: either a single histogram with the systematic variation if mirror=True or a list of two histograms with the up and down variation
        as_difference: if True, interpret the histogram values as the difference with respect to the nominal (i.e. the absolute variation is norm + h)
        syst_axes: optional list of axis names in h that represent independent systematics.
                   If None (default) and h is a hist-like object with axes beyond the channel,
                   the extra axes are auto-detected and each bin combination becomes a separate
                   systematic with name "{name}_{label_0}_{label_1}_...". Pass an empty list
                   to disable auto-detection.
        """

        # Fast batched path for SparseHist multi-systematic input. Conditions:
        #   - extra (systematic) axes are present
        #   - input is a single SparseHist (not an asymmetric pair)
        #   - mirror=True (single-hist symmetric input)
        #   - as_difference=True (so missing entries cleanly mean "no variation"
        #     for both log_normal and normal systematic types)
        #   - not added to the data covariance (which goes through a different
        #     bookkeeping path)
        extra_info = self._detect_extra_syst_axes(h, channel, syst_axes)
        if (
            extra_info is not None
            and not extra_info[1]  # is_pair
            and isinstance(extra_info[0], SparseHist)
            and mirror
            and as_difference
            and not add_to_data_covariance
        ):
            _, _, extra_axis_names = extra_info
            self._add_systematics_sparsehist_batched(
                h,
                name,
                process,
                channel,
                kfactor=kfactor,
                as_difference=as_difference,
                extra_axis_names=extra_axis_names,
                **kargs,
            )
            return

        # multi-systematic dispatch: if h has extra axes beyond the channel,
        # iterate over those and book each combination as an independent systematic
        slices = self._get_systematic_slices(h, name, channel, syst_axes)
        if slices is not None:
            for sub_name, sub_h in slices:
                self.add_systematic(
                    sub_h,
                    sub_name,
                    process,
                    channel,
                    kfactor=kfactor,
                    mirror=mirror,
                    symmetrize=symmetrize,
                    add_to_data_covariance=add_to_data_covariance,
                    as_difference=as_difference,
                    syst_axes=[],
                    **kargs,
                )
            return

        norm = self.dict_norm[channel][process]

        # Use sparse path when norm is stored as scipy sparse CSR
        if self._issparse(norm):
            return self._add_systematic_sparse(
                h,
                name,
                process,
                channel,
                norm,
                kfactor,
                mirror,
                symmetrize,
                add_to_data_covariance,
                as_difference,
                **kargs,
            )

        var_name_out = name

        systematic_type = "normal" if add_to_data_covariance else self.systematic_type

        flow = self.channels[channel]["flow"]

        if isinstance(h, (list, tuple)):
            self._check_hist_and_channel(h[0], channel)
            self._check_hist_and_channel(h[1], channel)

            syst_up = self.get_flat_values(h[0], flow=flow)
            syst_down = self.get_flat_values(h[1], flow=flow)

            if as_difference:
                syst_up = norm + syst_up
                syst_down = norm + syst_down

            logkup_proc = self.get_logk(
                syst_up, norm, kfactor, systematic_type=systematic_type
            )
            logkdown_proc = -self.get_logk(
                syst_down, norm, kfactor, systematic_type=systematic_type
            )

            logkavg_proc, var_name_out = self._compute_asym_syst(
                logkup_proc,
                logkdown_proc,
                name,
                process,
                channel,
                symmetrize,
                add_to_data_covariance,
                **kargs,
            )
        elif mirror:
            self._check_hist_and_channel(h, channel)
            syst = self.get_flat_values(h, flow=flow)

            if as_difference:
                syst = norm + syst

            logkavg_proc = self.get_logk(
                syst, norm, kfactor, systematic_type=systematic_type
            )
        else:
            raise RuntimeError(
                "Only one histogram given but mirror=False, can not construct a variation"
            )

        self.book_logk_avg(logkavg_proc, channel, process, var_name_out)
        self.book_systematic(
            var_name_out, add_to_data_covariance=add_to_data_covariance, **kargs
        )

    def add_beta_variations(
        self,
        h,
        process,
        source_channel,
        dest_channel,
    ):
        """
        Adds a template variation in the destination channel that is correlated with the beta variation in the source channel for a given process
        h: must be a histogram with the axes of the source channel and destiation channel.
        """
        if self.sparse:
            raise NotImplementedError("Sparse implementation not yet implemented")

        if source_channel not in self.channels.keys():
            raise RuntimeError(f"Channel {source_channel} not known!")
        if dest_channel not in self.channels.keys():
            raise RuntimeError(f"Channel {dest_channel} not known!")
        if not self.channels[dest_channel]["masked"]:
            raise RuntimeError(
                f"Beta variations can only be applied to masked channels"
            )

        norm = self.dict_norm[dest_channel][process]

        source_axes = self.channels[source_channel]["axes"]
        dest_axes = self.channels[dest_channel]["axes"]

        source_axes_names = [a.name for a in source_axes]
        dest_axes_names = [a.name for a in dest_axes]

        for a in source_axes:
            if a.name not in h.axes.name:
                raise RuntimeError(
                    f"Axis {a.name} not found in histogram h with {h.axes.name}"
                )
        for a in dest_axes:
            if a.name not in h.axes.name:
                raise RuntimeError(
                    f"Axis {a.name} not found in histogram h with {h.axes.name}"
                )

        flow = self.channels[dest_channel]["flow"]
        variation = h.project(*dest_axes_names, *source_axes_names).values(flow=flow)
        variation = variation.reshape((*norm.shape, -1))

        if source_channel not in self.dict_beta_variations[dest_channel].keys():
            self.dict_beta_variations[dest_channel][source_channel] = {}

        self.dict_beta_variations[dest_channel][source_channel][process] = variation

        self.has_beta_variations = True

    @staticmethod
    def _strcategory_labels(ax):
        """Return the bin labels of a hist StrCategory axis as a numpy string array.

        Raises if ``ax`` is not a StrCategory axis.
        """
        import hist as _hist

        if not isinstance(ax, _hist.axis.StrCategory):
            raise TypeError(
                f"External term axes must be hist.axis.StrCategory; got {type(ax).__name__}"
            )
        return np.array([ax.value(i) for i in range(len(ax))], dtype=object)

    def add_external_likelihood_term(self, grad=None, hess=None, name=None):
        """Add an additive quadratic term to the negative log-likelihood.

        The term has the form

            L_ext(x) = g^T x_sub + 0.5 * x_sub^T H x_sub

        where ``x_sub`` is the slice of the full fit parameter vector
        corresponding to the parameters identified by the StrCategory axes
        of ``grad`` / ``hess``. Both ``grad`` and ``hess`` must use the same
        parameter list in the same order. The parameter names are stored as
        strings and resolved against the full parameter list (POIs + systs)
        at fit time.

        Parameters
        ----------
        grad : hist.Hist, optional
            1D histogram with one ``hist.axis.StrCategory`` axis whose bin
            labels are parameter names. Values are the gradient ``g``.
        hess : hist.Hist or wums.SparseHist, optional
            2D histogram with two ``hist.axis.StrCategory`` axes; both must
            have identical bin labels equal to the gradient parameter list
            (if ``grad`` is also given). Values are the hessian ``H``. May
            be a dense ``hist.Hist`` or a ``wums.SparseHist`` for sparse
            storage. ``H`` should be symmetric (the formula is
            ``0.5 x^T H x``); the user is responsible for symmetrizing.
        name : str, optional
            Identifier for this term. Auto-generated if not provided.
            Multiple terms can be added by calling this method repeatedly.
        """
        if grad is None and hess is None:
            raise ValueError(
                "add_external_likelihood_term requires at least one of grad or hess"
            )

        if name is None:
            name = f"ext{len(self.external_terms)}"
        if any(t["name"] == name for t in self.external_terms):
            raise RuntimeError(f"External likelihood term '{name}' already added")

        params = None

        # Process gradient
        grad_values = None
        if grad is not None:
            if not hasattr(grad, "axes") or len(grad.axes) != 1:
                raise ValueError(
                    f"grad must be a 1D histogram, got {type(grad).__name__} with "
                    f"{len(grad.axes) if hasattr(grad, 'axes') else 0} axes"
                )
            grad_params = self._strcategory_labels(grad.axes[0])
            grad_values = np.asarray(grad.values()).flatten().astype(self.dtype)
            if len(grad_values) != len(grad_params):
                raise RuntimeError(
                    f"grad values length {len(grad_values)} does not match params length {len(grad_params)}"
                )
            params = grad_params

        # Process hessian
        hess_dense = None
        hess_sparse = None
        if hess is not None:
            if len(hess.axes) != 2:
                raise ValueError(
                    f"hess must be a 2D histogram, got {len(hess.axes)} axes"
                )
            hess_params0 = self._strcategory_labels(hess.axes[0])
            hess_params1 = self._strcategory_labels(hess.axes[1])
            if not np.array_equal(hess_params0, hess_params1):
                raise ValueError(
                    "hess must have identical labels on both axes (since it is "
                    "indexed by the same parameter list)"
                )
            if params is not None:
                if not np.array_equal(params, hess_params0):
                    raise ValueError(
                        "grad and hess must use the same parameter list in the "
                        f"same order; got grad params {params.tolist()} vs "
                        f"hess params {hess_params0.tolist()}"
                    )
            else:
                params = hess_params0

            if isinstance(hess, SparseHist):
                # Access the SparseHist's internal flat (indices, values)
                # buffers directly. Going through ``to_flat_csr`` would do an
                # O(nnz log nnz) sort that we don't need here, since the
                # downstream representation is unordered (rows, cols, values).
                # The flat indices live in the with-flow layout of the dense
                # shape, but for StrCategory axes with overflow=False the
                # extents equal the sizes so there are no flow bins to drop.
                n = len(params)
                flat = np.asarray(hess._flat_indices, dtype=np.int64)
                vals = np.asarray(hess._values, dtype=self.dtype)
                rows, cols = np.divmod(flat, n)
                hess_sparse = (rows, cols, vals)
            elif self._issparse(hess):
                raise ValueError(
                    "raw scipy sparse hess inputs are not supported; "
                    "wrap in wums.SparseHist with the parameter axes attached"
                )
            else:
                hess_dense = np.asarray(hess.values()).astype(self.dtype)
                if hess_dense.shape != (len(params), len(params)):
                    raise RuntimeError(
                        f"hess shape {hess_dense.shape} does not match "
                        f"params length {len(params)}"
                    )

        self.external_terms.append(
            {
                "name": name,
                "params": np.asarray(params),
                "grad_values": grad_values,
                "hess_dense": hess_dense,
                "hess_sparse": hess_sparse,
            }
        )

    @staticmethod
    def _sparse_values_at(sparse_csr, indices):
        """Extract values from a flat CSR array at the given flat indices.

        Uses searchsorted on the sorted CSR indices to avoid any dense conversion.
        Returns a dense 1D array of values at the requested positions.
        """
        result = np.zeros(len(indices), dtype=sparse_csr.dtype)
        positions = np.searchsorted(sparse_csr.indices, indices)
        valid = (positions < len(sparse_csr.indices)) & (
            sparse_csr.indices[positions] == indices
        )
        result[valid] = sparse_csr.data[positions[valid]]
        return result

    def _get_syst_at_norm_nnz(self, h, norm_csr, flow):
        """Extract flat systematic values only at norm's nonzero positions.

        h can be a histogram, scipy sparse, SparseHist, or dense array.
        ``flow`` controls the flat layout (must match the channel/norm layout).
        Returns a 1D dense array of length norm_csr.nnz.
        """
        nnz_idx = norm_csr.indices
        if hasattr(h, "values"):
            values = h.values(flow=flow)
            return values.flatten().astype(self.dtype)[nnz_idx]
        elif self._issparse(h):
            syst_csr = self._sparse_to_flat_csr(h, self.dtype, flow=flow)
            return self._sparse_values_at(syst_csr, nnz_idx)
        else:
            return np.asarray(h).flatten().astype(self.dtype)[nnz_idx]

    def get_logk(self, syst, norm, kfac=1.0, systematic_type=None):
        if not np.all(np.isfinite(syst)):
            raise RuntimeError(
                f"{len(syst)-sum(np.isfinite(syst))} NaN or Inf values encountered in systematic!"
            )

        # TODO clean this up and avoid duplication
        if systematic_type == "log_normal":
            # check if there is a sign flip between systematic and nominal
            _logk = kfac * np.log(syst / norm)
            _logk_view = np.where(
                np.equal(np.sign(norm * syst), 1),
                _logk,
                self.logkepsilon * np.ones_like(_logk),
            )

            # FIXME does this actually take effect since _logk_view is normally returned?
            if self.clipSystVariations > 0.0:
                _logk = np.clip(_logk, -self.clip, self.clip)

            return _logk_view
        elif systematic_type == "normal":
            _logk = kfac * (syst - norm)
            return _logk
        else:
            raise RuntimeError(
                f"Invalid systematic_type {systematic_type}, valid choices are 'log_normal' or 'normal'"
            )

    def _get_logk_sparse(self, syst_vals, norm_vals, kfac, systematic_type):
        """Compute logk values at norm's nonzero positions only.

        syst_vals and norm_vals are dense 1D arrays of equal length (nnz of norm).
        Returns a 1D dense array of logk values at those positions.
        """
        if not np.all(np.isfinite(syst_vals)):
            raise RuntimeError(
                f"{len(syst_vals)-sum(np.isfinite(syst_vals))} NaN or Inf values encountered in systematic!"
            )

        if systematic_type == "log_normal":
            _logk = kfac * np.log(syst_vals / norm_vals)
            _logk = np.where(
                np.equal(np.sign(norm_vals * syst_vals), 1),
                _logk,
                self.logkepsilon,
            )
            if self.clipSystVariations > 0.0:
                _logk = np.clip(_logk, -self.clip, self.clip)
            return _logk
        elif systematic_type == "normal":
            return kfac * (syst_vals - norm_vals)
        else:
            raise RuntimeError(
                f"Invalid systematic_type {systematic_type}, valid choices are 'log_normal' or 'normal'"
            )

    def book_logk_avg(self, *args):
        self.book_logk(
            self.dict_logkavg,
            self.dict_logkavg_indices,
            *args,
        )

    def book_logk_halfdiff(self, *args):
        self.book_logk(
            self.dict_logkhalfdiff,
            self.dict_logkhalfdiff_indices,
            *args,
        )

    def book_logk(
        self,
        dict_logk,
        dict_logk_indices,
        logk,
        channel,
        process,
        syst_name,
    ):
        if isinstance(logk, tuple):
            # Sparse logk from _add_systematic_sparse: (indices, values, size)
            nnz_idx, logk_vals, size = logk
            nonzero_mask = logk_vals != 0.0
            indices = nnz_idx[nonzero_mask].reshape(-1, 1)
            values = logk_vals[nonzero_mask]
            dict_logk_indices[channel][process][syst_name] = indices
            dict_logk[channel][process][syst_name] = values
            return

        norm = self.dict_norm[channel][process]
        # ensure that systematic tensor is sparse where normalization matrix is sparse
        logk = np.where(np.equal(norm, 0.0), 0.0, logk)
        if self.sparse:
            indices = np.transpose(np.nonzero(logk))
            dict_logk_indices[channel][process][syst_name] = indices
            dict_logk[channel][process][syst_name] = np.reshape(logk[indices], [-1])
        else:
            dict_logk[channel][process][syst_name] = logk

    def book_systematic(
        self,
        name,
        noi=False,
        constrained=True,
        add_to_data_covariance=False,
        groups=None,
    ):

        if add_to_data_covariance:
            if noi:
                raise ValueError(
                    f"{name} is maked as 'noi' but an 'noi' can't be added to the data covariance matrix."
                )
            self.systscovariance.add(name)
        elif not constrained:
            self.systsnoconstraint.add(name)
        else:
            self.systsstandard.add(name)

        if noi:
            self.systsnoi.add(name)

        # below only makes sense if this is an explicit nuisance parameter
        if not add_to_data_covariance:
            if groups is None:
                groups = [name]

            for group in groups:
                self.dict_systgroups[group].add(name)

    def write(self, outfolder="./", outfilename="rabbit_input.hdf5", meta_data_dict={}):

        if self.signals.intersection(self.bkgs):
            raise RuntimeError(
                f"Processes '{self.signals.intersection(self.bkgs)}' found as signal and background"
            )

        procs = sorted(list(self.signals)) + sorted(list(self.bkgs))
        nproc = len(procs)

        nbins = sum(
            [v for c, v in self.nbinschan.items() if not self.channels[c]["masked"]]
        )
        # nbinsfull including masked channels
        nbinsfull = sum([v for v in self.nbinschan.values()])

        logger.info(f"Write out nominal arrays")
        sumw = np.zeros([nbinsfull, nproc], self.dtype)
        sumw2 = np.zeros([nbinsfull, nproc], self.dtype)
        data_obs = np.zeros([nbins], self.dtype)
        data_var = np.zeros([nbins], self.dtype)
        pseudodata = np.zeros([nbins, len(self.pseudodata_names)], self.dtype)
        ibin = 0
        for chan, chan_info in self.channels.items():
            nbinschan = self.nbinschan[chan]

            for iproc, proc in enumerate(procs):
                if proc not in self.dict_norm[chan]:
                    continue

                norm_proc = self.dict_norm[chan][proc]
                if self._issparse(norm_proc):
                    sumw[ibin + norm_proc.indices, iproc] = norm_proc.data
                else:
                    sumw[ibin : ibin + nbinschan, iproc] = norm_proc
                sumw2[ibin : ibin + nbinschan, iproc] = self.dict_sumw2[chan][proc]

            if not chan_info["masked"]:
                data_obs[ibin : ibin + nbinschan] = self.dict_data_obs[chan]
                data_var[ibin : ibin + nbinschan] = self.dict_data_var[chan]

                for idx, name in enumerate(self.pseudodata_names):
                    pseudodata[ibin : ibin + nbinschan, idx] = self.dict_pseudodata[
                        chan
                    ][name]

            ibin += nbinschan

        systs = self.get_systs()
        nsyst = len(systs)

        if self.symmetric_tensor:
            logger.info("No asymmetric systematics - write fully symmetric tensor")

        ibin = 0
        if self.sparse:
            logger.info(f"Write out sparse array")

            # Pre-compute total sizes so we can allocate the assembly buffers
            # once instead of growing them per (channel, process, syst) which
            # is O(N^2) total via np.ndarray.resize. This pass only touches
            # python dict structures and is essentially free.
            norm_sparse_size_total = 0
            logk_sparse_size_total = 0
            for chan_pre in self.channels.keys():
                dict_norm_chan_pre = self.dict_norm[chan_pre]
                dict_logkavg_chan_idx_pre = self.dict_logkavg_indices[chan_pre]
                dict_logkhalfdiff_chan_idx_pre = self.dict_logkhalfdiff_indices[
                    chan_pre
                ]
                for proc_pre in procs:
                    if proc_pre not in dict_norm_chan_pre:
                        continue
                    norm_proc_pre = dict_norm_chan_pre[proc_pre]
                    if self._issparse(norm_proc_pre):
                        norm_sparse_size_total += int(len(norm_proc_pre.indices))
                    else:
                        norm_sparse_size_total += int(np.count_nonzero(norm_proc_pre))
                    proc_logk_idx_pre = dict_logkavg_chan_idx_pre[proc_pre]
                    for syst_idx_arr in proc_logk_idx_pre.values():
                        logk_sparse_size_total += int(syst_idx_arr.shape[0])
                    proc_halfdiff_idx_pre = dict_logkhalfdiff_chan_idx_pre[proc_pre]
                    for syst_idx_arr in proc_halfdiff_idx_pre.values():
                        logk_sparse_size_total += int(syst_idx_arr.shape[0])

            norm_sparse_indices = np.empty([norm_sparse_size_total, 2], self.idxdtype)
            norm_sparse_values = np.empty([norm_sparse_size_total], self.dtype)
            logk_sparse_normindices = np.empty(
                [logk_sparse_size_total, 1], self.idxdtype
            )
            logk_sparse_systindices = np.empty(
                [logk_sparse_size_total, 1], self.idxdtype
            )
            logk_sparse_values = np.empty([logk_sparse_size_total], self.dtype)

            norm_sparse_size = 0
            logk_sparse_size = 0

            for chan in self.channels.keys():
                nbinschan = self.nbinschan[chan]
                dict_norm_chan = self.dict_norm[chan]
                dict_logkavg_chan_indices = self.dict_logkavg_indices[chan]
                dict_logkavg_chan_values = self.dict_logkavg[chan]

                for iproc, proc in enumerate(procs):
                    if proc not in dict_norm_chan:
                        continue
                    norm_proc = dict_norm_chan[proc]

                    if self._issparse(norm_proc):
                        # Use scipy sparse structure directly
                        norm_indices = norm_proc.indices.reshape(-1, 1)
                        norm_values = norm_proc.data

                        nvals = len(norm_values)
                        oldlength = norm_sparse_size
                        norm_sparse_size = oldlength + nvals

                        out_indices = np.array([[ibin, iproc]]) + np.pad(
                            norm_indices, ((0, 0), (0, 1)), "constant"
                        )
                        norm_indices = None

                        norm_sparse_indices[oldlength:norm_sparse_size] = out_indices
                        out_indices = None

                        norm_sparse_values[oldlength:norm_sparse_size] = norm_values
                        norm_values = None

                        # sorted CSR indices allow searchsorted in logk mapping below
                        norm_nnz_idx = norm_proc.indices
                        oldlength_norm = oldlength
                    else:
                        norm_indices = np.transpose(np.nonzero(norm_proc))
                        norm_values = np.reshape(norm_proc[norm_indices], [-1])

                        nvals = len(norm_values)
                        oldlength = norm_sparse_size
                        norm_sparse_size = oldlength + nvals

                        out_indices = np.array([[ibin, iproc]]) + np.pad(
                            norm_indices, ((0, 0), (0, 1)), "constant"
                        )
                        norm_indices = None

                        norm_sparse_indices[oldlength:norm_sparse_size] = out_indices
                        out_indices = None

                        norm_sparse_values[oldlength:norm_sparse_size] = norm_values
                        norm_values = None

                        norm_idx_map = (
                            np.cumsum(np.not_equal(norm_proc, 0.0)) - 1 + oldlength
                        )
                        norm_nnz_idx = None

                    dict_logkavg_proc_indices = dict_logkavg_chan_indices[proc]
                    dict_logkavg_proc_values = dict_logkavg_chan_values[proc]

                    for isyst, syst in enumerate(systs):
                        if syst not in dict_logkavg_proc_indices.keys():
                            continue

                        logkavg_proc_indices = dict_logkavg_proc_indices[syst]
                        logkavg_proc_values = dict_logkavg_proc_values[syst]

                        nvals_proc = len(logkavg_proc_values)
                        oldlength = logk_sparse_size
                        logk_sparse_size = oldlength + nvals_proc

                        # first dimension of output indices are NOT in the dense [nbin,nproc] space, but rather refer to indices in the norm_sparse vectors
                        # second dimension is flattened in the [2,nsyst] space, where logkavg corresponds to [0,isyst] flattened to isyst
                        # two dimensions are kept in separate arrays for now to reduce the number of copies needed later
                        if norm_nnz_idx is not None:
                            # scipy sparse norm: use searchsorted on sorted CSR indices
                            flat_positions = logkavg_proc_indices.flatten()
                            out_normindices = (
                                np.searchsorted(norm_nnz_idx, flat_positions)
                                + oldlength_norm
                            ).reshape(-1, 1)
                        else:
                            out_normindices = norm_idx_map[logkavg_proc_indices]
                        logkavg_proc_indices = None

                        logk_sparse_normindices[oldlength:logk_sparse_size] = (
                            out_normindices
                        )
                        logk_sparse_systindices[oldlength:logk_sparse_size] = isyst
                        out_normindices = None

                        logk_sparse_values[oldlength:logk_sparse_size] = (
                            logkavg_proc_values
                        )
                        logkavg_proc_values = None

                        if syst in self.dict_logkhalfdiff_indices[chan][proc].keys():
                            logkhalfdiff_proc_indices = self.dict_logkhalfdiff_indices[
                                chan
                            ][proc][syst]
                            logkhalfdiff_proc_values = self.dict_logkhalfdiff[chan][
                                proc
                            ][syst]

                            nvals_proc = len(logkhalfdiff_proc_values)
                            oldlength = logk_sparse_size
                            logk_sparse_size = oldlength + nvals_proc

                            # out_indices = np.array([[ibin,iproc,isyst,1]]) + np.pad(logkhalfdiff_proc_indices,((0,0),(0,3)),'constant')
                            # first dimension of output indices are NOT in the dense [nbin,nproc] space, but rather refer to indices in the norm_sparse vectors
                            # second dimension is flattened in the [2,nsyst] space, where logkhalfdiff corresponds to [1,isyst] flattened to nsyst + isyst
                            # two dimensions are kept in separate arrays for now to reduce the number of copies needed later
                            if norm_nnz_idx is not None:
                                flat_positions = logkhalfdiff_proc_indices.flatten()
                                out_normindices = (
                                    np.searchsorted(norm_nnz_idx, flat_positions)
                                    + oldlength_norm
                                ).reshape(-1, 1)
                            else:
                                out_normindices = norm_idx_map[
                                    logkhalfdiff_proc_indices
                                ]
                            logkhalfdiff_proc_indices = None

                            logk_sparse_normindices[oldlength:logk_sparse_size] = (
                                out_normindices
                            )
                            logk_sparse_systindices[oldlength:logk_sparse_size] = (
                                nsyst + isyst
                            )
                            out_normindices = None

                            logk_sparse_values[oldlength:logk_sparse_size] = (
                                logkhalfdiff_proc_values
                            )
                            logkhalfdiff_proc_values = None

                    # free memory
                    dict_logkavg_proc_indices = None
                    dict_logkavg_proc_values = None

                # free memory
                norm_proc = None
                norm_idx_map = None

                ibin += nbinschan

            logger.info(f"Sort sparse arrays into canonical order")
            assert norm_sparse_size == norm_sparse_size_total
            assert logk_sparse_size == logk_sparse_size_total

            # straightforward sorting of norm_sparse into canonical order
            norm_sparse_dense_shape = (nbinsfull, nproc)
            norm_sort_indices = np.argsort(
                np.ravel_multi_index(
                    np.transpose(norm_sparse_indices), norm_sparse_dense_shape
                )
            )
            norm_sparse_indices = norm_sparse_indices[norm_sort_indices]
            norm_sparse_values = norm_sparse_values[norm_sort_indices]

            # now permute the indices of the first dimension of logk_sparse corresponding to the resorting of norm_sparse

            # compute the inverse permutation from the sorting of norm_sparse
            # since the final indices are filled from here, need to ensure it has the correct data type
            logk_permute_indices = np.argsort(norm_sort_indices).astype(self.idxdtype)
            norm_sort_indices = None
            logk_sparse_normindices = logk_permute_indices[logk_sparse_normindices]
            logk_permute_indices = None
            logk_sparse_indices = np.concatenate(
                [logk_sparse_normindices, logk_sparse_systindices], axis=-1
            )

            # now straightforward sorting of logk_sparse into canonical order
            if self.symmetric_tensor:
                logk_sparse_dense_shape = (norm_sparse_indices.shape[0], nsyst)
            else:
                logk_sparse_dense_shape = (norm_sparse_indices.shape[0], 2 * nsyst)
            logk_sort_indices = np.argsort(
                np.ravel_multi_index(
                    np.transpose(logk_sparse_indices), logk_sparse_dense_shape
                )
            )
            logk_sparse_indices = logk_sparse_indices[logk_sort_indices]
            logk_sparse_values = logk_sparse_values[logk_sort_indices]
            logk_sort_indices = None

        else:
            logger.info(f"Write out dense array")
            # initialize with zeros, i.e. no variation
            norm = np.zeros([nbinsfull, nproc], self.dtype)
            if self.symmetric_tensor:
                logk = np.zeros([nbinsfull, nproc, nsyst], self.dtype)
            else:
                logk = np.zeros([nbinsfull, nproc, 2, nsyst], self.dtype)

            if self.has_beta_variations:
                beta_variations = np.zeros([nbinsfull, nbins, nproc], self.dtype)

            for chan in self.channels.keys():
                nbinschan = self.nbinschan[chan]
                dict_norm_chan = self.dict_norm[chan]

                for iproc, proc in enumerate(procs):
                    if proc not in dict_norm_chan:
                        continue

                    norm_proc = dict_norm_chan[proc]

                    norm[ibin : ibin + nbinschan, iproc] = norm_proc

                    dict_logkavg_proc = self.dict_logkavg[chan][proc]
                    dict_logkhalfdiff_proc = self.dict_logkhalfdiff[chan][proc]
                    for isyst, syst in enumerate(systs):
                        if syst not in dict_logkavg_proc.keys():
                            continue

                        if self.symmetric_tensor:
                            logk[ibin : ibin + nbinschan, iproc, isyst] = (
                                dict_logkavg_proc[syst]
                            )
                        else:
                            logk[ibin : ibin + nbinschan, iproc, 0, isyst] = (
                                dict_logkavg_proc[syst]
                            )
                            if syst in dict_logkhalfdiff_proc.keys():
                                logk[ibin : ibin + nbinschan, iproc, 1, isyst] = (
                                    dict_logkhalfdiff_proc[syst]
                                )

                    if not self.has_beta_variations:
                        continue

                    for (
                        source_channel,
                        source_channel_dict,
                    ) in self.dict_beta_variations[chan].items():
                        if proc not in source_channel_dict:
                            continue
                        # find the bins of the source channel
                        ibin_start = 0
                        for c, nb in self.nbinschan.items():
                            if self.channels[c]["masked"]:
                                continue  # masked channels can not be source channels
                            if c == source_channel:
                                ibin_end = ibin_start + nb
                                break
                            else:
                                ibin_start += nb
                        else:
                            raise RuntimeError(
                                f"Did not find source channel {source_channel} in list of channels {[k for k in self.nbinschan.keys()]}"
                            )

                        beta_vars = source_channel_dict[proc]
                        beta_variations[
                            ibin : ibin + nbinschan, ibin_start:ibin_end, iproc
                        ] = beta_vars

                ibin += nbinschan

        if self.data_covariance is None and (
            self.systscovariance or self.add_bin_by_bin_stat_to_data_cov
        ):
            # create data covariance
            self.data_covariance = np.diag(data_var)

        # write results to hdf5 file
        procSize = nproc * np.dtype(self.dtype).itemsize
        systSize = 2 * nsyst * np.dtype(self.dtype).itemsize
        amax = np.max([procSize, systSize])
        if amax > self.chunkSize:
            logger.info(
                f"Maximum chunk size in bytes was increased from {self.chunkSize} to {amax} to align with tensor sizes and allow more efficient reading/writing."
            )
            self.chunkSize = amax

        # create HDF5 file (chunk cache set to the chunk size since we can guarantee fully aligned writes
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)
        outpath = f"{outfolder}/{outfilename}"
        if len(outfilename.split(".")) < 2:
            outpath += ".hdf5"
        logger.info(f"Write output file {outpath}")
        f = h5py.File(outpath, rdcc_nbytes=self.chunkSize, mode="w")

        if meta_data_dict is not None:
            meta_data_dict.update(
                {
                    "channel_info": self.channels,
                    "symmetric_tensor": self.symmetric_tensor,
                    "systematic_type": self.systematic_type,
                }
            )
            if "meta_info" not in meta_data_dict:
                meta_data_dict["meta_info"] = {}

        ioutils.pickle_dump_h5py("meta", meta_data_dict, f)

        noiidxs = self.get_noiidxs()
        systsnoconstraint = self.get_systsnoconstraint()
        systgroups, systgroupidxs = self.get_systgroups()

        # save some lists of strings to the file for later use
        def create_dataset(
            name,
            content,
            length=None,
            dtype=h5py.special_dtype(vlen=str),
            compression="gzip",
        ):
            dimension = [len(content), length] if length else [len(content)]
            ds = f.create_dataset(
                f"h{name}", dimension, dtype=dtype, compression=compression
            )
            ds[...] = content

        create_dataset("procs", procs)
        create_dataset("signals", sorted(list(self.signals)))
        create_dataset("systs", systs)
        create_dataset("systsnoconstraint", systsnoconstraint)
        create_dataset("systgroups", systgroups)
        create_dataset(
            "systgroupidxs",
            systgroupidxs,
            dtype=h5py.special_dtype(vlen=np.dtype("int32")),
        )
        create_dataset("noiidxs", noiidxs, dtype="int32")
        create_dataset("pseudodatanames", [n for n in self.pseudodata_names])

        # create h5py datasets with optimized chunk shapes
        nbytes = 0

        constraintweights = self.get_constraintweights(self.dtype)
        nbytes += h5pyutils_write.writeFlatInChunks(
            constraintweights, f, "hconstraintweights", maxChunkBytes=self.chunkSize
        )
        constraintweights = None

        nbytes += h5pyutils_write.writeFlatInChunks(
            data_obs, f, "hdata_obs", maxChunkBytes=self.chunkSize
        )
        if np.any(data_var != data_obs):
            nbytes += h5pyutils_write.writeFlatInChunks(
                data_var, f, "hdata_var", maxChunkBytes=self.chunkSize
            )
            data_var = None
        data_obs = None

        nbytes += h5pyutils_write.writeFlatInChunks(
            pseudodata, f, "hpseudodata", maxChunkBytes=self.chunkSize
        )
        pseudodata = None

        if self.data_covariance is not None:
            for syst in self.systscovariance:
                systv = np.zeros(shape=(nbinsfull, 1), dtype=self.dtype)

                ibin = 0
                for chan in self.channels.keys():
                    nbinschan = self.nbinschan[chan]
                    dict_norm_chan = self.dict_norm[chan]

                    for proc in procs:
                        if proc not in dict_norm_chan:
                            continue

                        dict_logkavg_proc = self.dict_logkavg[chan][proc]

                        if syst not in dict_logkavg_proc.keys():
                            continue

                        systv[ibin : ibin + nbinschan, 0] += dict_logkavg_proc[syst]

                    ibin += nbinschan

                self.data_covariance[...] += systv @ systv.T

            full_cov = (
                np.add(self.data_covariance, np.diag(sumw2))
                if self.add_bin_by_bin_stat_to_data_cov
                else self.data_covariance
            )
            full_cov_inv = np.linalg.inv(full_cov)

            nbytes += h5pyutils_write.writeFlatInChunks(
                full_cov_inv,
                f,
                "hdata_cov_inv",
                maxChunkBytes=self.chunkSize,
            )

        nbytes += h5pyutils_write.writeFlatInChunks(
            sumw, f, "hsumw", maxChunkBytes=self.chunkSize
        )

        nbytes += h5pyutils_write.writeFlatInChunks(
            sumw2, f, "hsumw2", maxChunkBytes=self.chunkSize
        )

        if self.sparse:
            nbytes += h5pyutils_write.writeSparse(
                norm_sparse_indices,
                norm_sparse_values,
                norm_sparse_dense_shape,
                f,
                "hnorm_sparse",
                maxChunkBytes=self.chunkSize,
            )
            norm_sparse_indices = None
            norm_sparse_values = None
            nbytes += h5pyutils_write.writeSparse(
                logk_sparse_indices,
                logk_sparse_values,
                logk_sparse_dense_shape,
                f,
                "hlogk_sparse",
                maxChunkBytes=self.chunkSize,
            )
            logk_sparse_indices = None
            logk_sparse_values = None
        else:
            nbytes += h5pyutils_write.writeFlatInChunks(
                norm, f, "hnorm", maxChunkBytes=self.chunkSize
            )
            norm = None
            nbytes += h5pyutils_write.writeFlatInChunks(
                logk, f, "hlogk", maxChunkBytes=self.chunkSize
            )
            logk = None

            if self.has_beta_variations:
                nbytes += h5pyutils_write.writeFlatInChunks(
                    beta_variations, f, "hbetavariations", maxChunkBytes=self.chunkSize
                )
                beta_variations = None

        # Write external likelihood terms. Each term is written as a
        # subgroup under "external_terms"; the reader iterates the
        # subgroups directly, so no separate names list is needed.
        if self.external_terms:
            ext_group = f.create_group("external_terms")
            for term in self.external_terms:
                term_group = ext_group.create_group(term["name"])
                params_ds = term_group.create_dataset(
                    "params",
                    [len(term["params"])],
                    dtype=h5py.special_dtype(vlen=str),
                    compression="gzip",
                )
                params_ds[...] = [str(p) for p in term["params"]]

                if term["grad_values"] is not None:
                    nbytes += h5pyutils_write.writeFlatInChunks(
                        term["grad_values"],
                        term_group,
                        "grad_values",
                        maxChunkBytes=self.chunkSize,
                    )

                if term["hess_dense"] is not None:
                    nbytes += h5pyutils_write.writeFlatInChunks(
                        term["hess_dense"],
                        term_group,
                        "hess_dense",
                        maxChunkBytes=self.chunkSize,
                    )
                elif term["hess_sparse"] is not None:
                    rows, cols, vals = term["hess_sparse"]
                    n = len(term["params"])
                    rows = np.asarray(rows, dtype=self.idxdtype)
                    cols = np.asarray(cols, dtype=self.idxdtype)
                    vals = np.asarray(vals, dtype=self.dtype)
                    # Sort into canonical row-major order so the reader
                    # (and downstream tf.sparse / CSR consumers) can skip
                    # the reorder step. The fast path: if the input is
                    # already canonical (typical when the source is a
                    # SparseHist whose flat indices come in flat-index
                    # order), skip the O(nnz log nnz) argsort entirely.
                    # The check is a single vectorized O(nnz) pass and
                    # is essentially free compared to the sort it avoids
                    # (~50-150 s on 329M nnz).
                    if rows.size > 1:
                        drows = np.diff(rows)
                        dcols = np.diff(cols)
                        already_sorted = bool(
                            np.all((drows > 0) | ((drows == 0) & (dcols >= 0)))
                        )
                        del drows, dcols
                    else:
                        already_sorted = True
                    if not already_sorted:
                        flat = np.ravel_multi_index((rows, cols), (n, n))
                        sort_order = np.argsort(flat)
                        del flat
                        rows = rows[sort_order]
                        cols = cols[sort_order]
                        vals = vals[sort_order]
                    indices = np.stack([rows, cols], axis=-1)
                    nbytes += h5pyutils_write.writeSparse(
                        indices,
                        vals,
                        (n, n),
                        term_group,
                        "hess_sparse",
                        maxChunkBytes=self.chunkSize,
                    )

        logger.info(f"Total raw bytes in arrays = {nbytes}")

    def get_systsstandard(self):
        return list(common.natural_sort(self.systsstandard))

    def get_systsnoi(self):
        return list(common.natural_sort(self.systsnoi))

    def get_systsnoconstraint(self):
        return list(common.natural_sort(self.systsnoconstraint))

    def get_systs(self):
        return self.get_systsnoconstraint() + self.get_systsstandard()

    def get_constraintweights(self, dtype):
        systs = self.get_systs()
        constraintweights = np.ones([len(systs)], dtype=dtype)
        syst_to_idx = {s: i for i, s in enumerate(systs)}
        for syst in self.get_systsnoconstraint():
            constraintweights[syst_to_idx[syst]] = 0.0
        return constraintweights

    def get_groups(self, group_dict):
        systs = self.get_systs()
        # Pre-compute name -> index mapping once. The previous implementation
        # called ``systs.index(syst)`` per group member which is O(len(systs))
        # each, giving O(nsysts * nmembers) total -- prohibitive when both are
        # large (e.g. ~108k corparms in a single group).
        syst_to_idx = {s: i for i, s in enumerate(systs)}
        groups = []
        idxs = []
        for group, members in common.natural_sort_dict(group_dict).items():
            groups.append(group)
            idxs.append([syst_to_idx[syst] for syst in members])
        return groups, idxs

    def get_noiidxs(self):
        # list of indices of nois w.r.t. systs
        systs = self.get_systs()
        syst_to_idx = {s: i for i, s in enumerate(systs)}
        return [syst_to_idx[noi] for noi in self.get_systsnoi()]

    def get_systgroups(self):
        # list of groups of systematics (nuisances) and lists of indexes
        return self.get_groups(self.dict_systgroups)
