"""Multi-device (bins-sharded) evaluation of the likelihood.

The NLL is a plain sum over bins plus parameter-level terms, and every
large tensor in the fit (logk, norm, nobs, sumw/sumw2, beta) is
bins-proportional. Sharding the bins axis over N devices therefore
splits both the memory and the compute, while the only things that ever
cross devices per evaluation are the parameter vector going out and one
[nparams] partial (gradient / HVP) per shard coming back -- kilobytes
against the gigabytes each shard reads locally.

The two structural rules, established empirically (bench/poc_multigpu.py
on 4 GPUs):

1. Backward placement must be explicit. A single GradientTape over the
   combined multi-device loss computes correct values but the placer
   puts the backward ops on one device, dragging every shard's logk
   across the bus per call (measured 6.5 ms -> 135 ms at 4 shards).
   Hence: one tape per shard, differentiating with respect to a
   device-local copy of x, so each shard's backward subgraph is anchored
   to its device. The combined loss+grad returned to 4.5 ms.
2. XLA clusters are single-device: each shard's function is jit-compiled
   on its own, and the thin combiner stays a plain (non-jit) graph.

Rather than duplicating the yields/NLL mathematics, each shard is a
duck-typed evaluator object exposing the attributes the existing Fitter
methods read (indata view, logk slice, nobs, bbstat, param model, x);
the Fitter's own unbound methods are then bound onto it, so symmetric
and asymmetric interpolation, both systematic types, and the full
bin-by-bin-stat machinery are shared with the single-device path by
construction.

Not supported in sharded mode: sparse tensors and --covarianceFit
(checked at construction), regularizers whose penalty reads the predicted
yields (needs_observables=True -- no device holds the full yield vector;
checked in arm_regularizers, since self.regularizers is still empty when
the constructor runs), --fullNll, and the fwdrev HVP (falls back to
revrev). Parameter-only regularizers ARE supported: their penalty is a
function of the parameter vector, which every device has in full, and it
is evaluated once in the global term.
"""

import gc

import numpy as np
import tensorflow as tf
from wums import logging

from rabbit import external_likelihood
from rabbit import unbinned as unbinned_terms_mod
from rabbit.bbstat.bbstat import BinByBinStat
from rabbit.fitter import Fitter

logger = logging.child_logger(__name__)


# Buffered messages from pick_physical_gpus, drained by the driver once the
# logger exists (see drain_selection_log).
selection_log = []


def pick_physical_gpus(n, explicit=None):
    """Choose which physical GPUs to make visible, before TF initializes.

    ``explicit`` (a list of indices, from --devices) wins outright.
    Otherwise, when the node has more GPUs than requested, prefer the
    least-occupied ones by current memory use (via nvidia-smi), so a fit
    started on an interactively shared node lands next to nobody. Falls
    back to the first n on any query failure. NB nvidia-smi orders by PCI
    bus while CUDA defaults to fastest-first; on the homogeneous nodes
    here the orders coincide, and --devices remains the explicit escape
    hatch.
    """
    if n < 1:
        # Otherwise n = 0 hides every GPU (sorted(order[:0]) == [], which the
        # caller treats as a real selection) while make_fitter normalizes
        # 0 -> 1 and builds a single-device Fitter, so the run lands silently
        # on the CPU; n = -1 drops one GPU and leaves n_devices negative.
        raise ValueError(f"--nDevices must be >= 1, got {n}")
    # Validate the inputs before looking at the hardware: otherwise a bad
    # --devices is rejected on a GPU node and silently ignored on a CPU one
    # (the no-GPU early return below fires first), so CI never sees it.
    if explicit is not None:
        if len(explicit) != n:
            # Otherwise the mismatch falls through to select_devices' round
            # robin, which silently puts two shards on one GPU (or leaves one
            # idle) rather than doing what --devices promises.
            raise ValueError(
                f"--devices lists {len(explicit)} device(s) but --nDevices is "
                f"{n}; they must match."
            )
        if any(int(i) < 0 for i in explicit):
            # Python would wrap these: gpus[-1] silently selects the last GPU,
            # so a negative index picks a device the user never named while a
            # positive out-of-range one raises.
            raise ValueError(f"--devices indices must be >= 0, got {explicit}")

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return None
    if explicit is not None:
        try:
            return [gpus[int(i)] for i in explicit]
        except IndexError:
            raise ValueError(
                f"--devices {explicit} out of range: {len(gpus)} GPU(s) visible"
            )
    if len(gpus) <= n:
        return gpus

    import subprocess

    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        usage = {}
        for line in out.strip().splitlines():
            idx, used = line.split(",")
            usage[int(idx)] = int(used)
        order = sorted(range(len(gpus)), key=lambda i: usage.get(i, 0))
        chosen = sorted(order[:n])
        selection_log.append(
            (
                "info",
                f"Selecting GPU(s) {chosen} by occupancy "
                f"(memory used per GPU: {usage})",
            )
        )
        return [gpus[i] for i in chosen]
    except Exception as ex:
        selection_log.append(
            (
                "warning",
                f"Could not query GPU occupancy ({ex}); using the first {n} GPU(s).",
            )
        )
        return gpus[:n]


def drain_selection_log():
    """Emit and clear whatever pick_physical_gpus buffered.

    That runs before TF may touch a GPU, which is before the driver has set
    up the logger, so logging from inside it drops the records at every
    verbosity -- and the GPU choice is the one thing there worth a record.
    The caller drains this once the logger exists.
    """
    global selection_log
    for level, msg in selection_log:
        getattr(logger, level)(msg)
    selection_log = []


def select_devices(n):
    """Names of the n devices to shard over.

    Prefers distinct GPUs. With fewer GPUs than shards the shards are
    placed round-robin (still correct, still memory-split per device up
    to the number of GPUs); with no GPU at all every shard lands on the
    CPU -- functionally identical, which is what the correctness tests
    rely on.
    """
    gpus = tf.config.list_logical_devices("GPU")
    if len(gpus) >= n:
        return [g.name for g in gpus[:n]]
    if gpus:
        logger.warning(
            f"Requested {n} device shards but only {len(gpus)} GPU(s) are "
            "visible; placing shards round-robin."
        )
        return [gpus[i % len(gpus)].name for i in range(n)]
    cpus = tf.config.list_logical_devices("CPU")
    if len(cpus) >= n:
        # multiple logical CPU devices (tf.config.set_logical_device_
        # configuration): genuine cross-device placement without GPUs --
        # this is how the tests catch placement bugs XLA only raises
        # across devices, e.g. jit functions capturing a Variable resident
        # elsewhere
        return [c.name for c in cpus[:n]]
    logger.warning(
        f"Requested {n} device shards but no GPU is visible; running all "
        "shards on the CPU (correct, but no speedup)."
    )
    return [cpus[0].name] * n


class ShardIndataView:
    """The slice of FitInputData one shard sees, resident on its device.

    Only the attributes the yields/NLL/BBB code reads are provided. The
    view covers a contiguous range of *observed* bins; masked channels
    are excluded (the sharded path only serves the full=False likelihood
    evaluations), so nbins == nbinsfull and nbinsmasked == 0.
    """

    def __init__(self, indata, start, stop, device):
        self.start = int(start)
        self.stop = int(stop)
        nb = self.stop - self.start
        # Slice on the host, then copy only the slice, exactly as _build_shards
        # does for logk and nobs. Evaluating indata.norm[start:stop] inside the
        # device scope places the StridedSlice on that device, which requires
        # its input -- the whole [nbinsfull, nproc] tensor -- to be copied there
        # first, once per shard. On CPU-only runners that copy is free, which is
        # why the test suite cannot see this and why it only bites on the
        # hardware the feature exists for.
        with tf.device("/CPU:0"):
            norm = tf.identity(indata.norm[self.start : self.stop])
            sumw = tf.identity(indata.sumw[self.start : self.stop])
            sumw2 = tf.identity(indata.sumw2[self.start : self.stop])
        with tf.device(device):
            self.norm = tf.identity(norm)
            self.sumw = tf.identity(sumw)
            self.sumw2 = tf.identity(sumw2)
        self.nbins = nb
        self.nbinsfull = nb
        self.nbinsmasked = 0
        self.nproc = indata.nproc
        self.nsyst = indata.nsyst
        self.symmetric_tensor = indata.symmetric_tensor
        self.systematic_type = indata.systematic_type
        self.sparse = False
        self.dtype = indata.dtype
        # bbstat.profile_and_apply reads this before its `and full` short
        # circuits, so it has to exist even in lite mode. It is bin x bin
        # (the full-mode profile contracts over the bin axis with
        # einsum("ijk,jk->ik")), hence genuinely unshardable -- _build_shards
        # refuses the non-None case rather than slicing it wrongly.
        self.betavar = indata.betavar


class ShardParamModel:
    """Delegating wrapper that slices bin-dependent process rates.

    param_model.compute returns a tensor broadcastable against
    [nbins, nproc]; when a model makes it genuinely bin-dependent
    (leading dimension > 1) the shard must see only its own rows.
    Everything else (npoi, npou, nparams, defaults, ...) passes through.
    """

    def __init__(self, inner, start, stop):
        self._inner = inner
        self._start = int(start)
        self._stop = int(stop)

    def compute(self, param, full=False):
        rnorm = self._inner.compute(param, full)
        nlead = rnorm.shape[0]
        if nlead is not None and nlead > 1:
            rnorm = rnorm[self._start : self._stop]
        return rnorm

    def __getattr__(self, name):
        return getattr(self._inner, name)


class ShardEvaluator:
    """Duck-typed per-shard stand-in for the Fitter in the yields/NLL code.

    Plain attribute container; the Fitter binds its own unbound methods
    onto instances (see Fitter._build_shards), and the per-shard traced
    functions set ``x`` to the device-local parameter copy before
    calling them.
    """

    def __init__(self, device, indata_view):
        self.device = device
        self.indata = indata_view
        self.x = None  # set inside each traced shard function


def shard_edges(nbins, n):
    """Contiguous, balanced bin ranges. Bins are the unit of both memory
    and flops in the dense tensors, so equal bin counts balance the load."""
    edges = np.linspace(0, nbins, n + 1).astype(np.int64)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n)]


# The blinding offsets get_poi() and get_theta() read. The shard evaluators are
# duck-typed attribute containers, so every one of these has to be threaded to
# them explicitly -- and as a tensor, because XLA cannot read a Variable that
# lives on another device.
#
# Listed once rather than at each of the three sites that thread them: a Fitter
# that grows or drops an offset would otherwise have to be matched in all three,
# and missing one is silent until the first armed sharded fit. That is exactly
# how _blinding_offsets_poi_add arrived -- added to get_poi() on main, threaded
# nowhere. test_sharded_fit.py checks this tuple against what a blinded Fitter
# actually creates.
_BLINDING_OFFSET_ATTRS = (
    "_blinding_offsets_poi",
    "_blinding_offsets_poi_add",
    "_blinding_offsets_theta",
)


class MultiDeviceFitter(Fitter):
    """Bins-sharded Fitter for multi-device (typically multi-GPU) fits.

    The device layout is fixed at construction (use
    :func:`rabbit.fitter.make_fitter` to pick the class), so all sharding
    lives here and the base Fitter carries none of it. The likelihood
    mathematics is not reimplemented: shard evaluators borrow the base
    class's own unbound methods (see _SHARD_BORROWED_METHODS), so this
    class owns orchestration only -- building the per-device shards and
    assembling loss/grad/HVP/Hessian from per-shard partials.

    ``_make_tf_functions`` (called at the end of init_fit_parms and after
    a deepcopy) rebuilds the shards first, so parameter-layout changes and
    toy copies stay consistent without extra hooks in the base class.
    """

    _DYNAMIC_TF_FUNCS = Fitter._DYNAMIC_TF_FUNCS | {
        "shards",
        "_shard_graph_fns",
        "_hessp_batch_sharded",
        "_global_view",
        "_global_graph_fns",
        # Instance-level tf.function whose traced graph captures every
        # shard's logk, so it is dynamic machinery like the rest: stripped
        # on __deepcopy__ and dropped before a rebuild.
        "_profile_beta",
    }

    def __init__(self, indata, param_model, options, **kwargs):
        # set before super().__init__: the base constructor ends in
        # init_fit_parms -> _make_tf_functions, which builds the shards
        self.n_devices = int(getattr(options, "nDevices", 1) or 1)
        self._options = options
        self.shards = []
        super().__init__(indata, param_model, options, **kwargs)

    _SHARD_BORROWED_METHODS = (
        "get_poi",
        "get_model_nui",
        "get_theta",
        "get_x",
        "_compute_yields_noBBB",
        "_compute_yields_with_beta",
        "_compute_ln",
        "_compute_lbeta",
        "_compute_lc",
    )

    def _build_shards(self):
        """Construct the per-device shard evaluators (multi-device mode).

        Shards partition the observed bins contiguously; each holds its
        slice of logk / norm / nobs / sumw(2) and its own BinByBinStat
        (beta and kstat are per-bin, so BBB shards along with everything
        else). Parameter-level state (x, frozen mask, blinding offsets,
        param model) is shared -- those tensors are [nparams]-sized and
        cross devices for kilobytes per evaluation.
        """
        if self.indata.sparse:
            raise NotImplementedError(
                "Multi-device fits (--nDevices > 1) are not supported in "
                "sparse mode."
            )
        if getattr(self, "unbinned_terms", None):
            # An unbinned term is a sum over CANDIDATES; the shard views here
            # slice the bins axis and carry no candidate axis at all, so there
            # is nothing for `shard_edges` to cut. They are evaluated in the
            # global (unsharded) term instead, which is correct but puts their
            # per-candidate tensors -- the largest arrays in such a fit -- on
            # one device. Sharding them needs a `ShardUnbinnedView`: a
            # contiguous candidate range with the term's per-candidate tensors
            # `tf.identity`-copied onto the shard's device, one tape per shard
            # on a device-local x, and the partials added like everything else.
            logger.warning(
                f"{len(self.unbinned_terms)} unbinned term(s) are evaluated on "
                f"{self.shards[0].device if self.shards else 'the first device'} "
                "rather than sharded: the shard views partition bins, and an "
                "unbinned term has candidates instead."
            )
        if (
            self.bbstat.enabled
            and self.bbstat.binByBinStatMode == "full"
            and getattr(self.indata, "betavar", None) is not None
        ):
            # Caution, not necessity, and worth being honest about which:
            # bbstat reads betavar only under `and full`, the sharded loss only
            # ever evaluates full=False, and the full=True paths run host-pinned
            # on the intact self.indata -- so the term does not in fact enter the
            # sharded likelihood today, and this refuses a combination that
            # would work. It is kept because ShardIndataView hands every shard
            # the whole unsliced bin x bin tensor, which is correct only while
            # nothing reads it; if that changes the result is a silently wrong
            # answer rather than an error, and lite mode is the documented
            # multi-device path anyway. Drop it if the combination is wanted.
            raise NotImplementedError(
                "--binByBinStatMode full with a per-bin beta covariance "
                "(indata.betavar) is not supported in multi-device mode "
                "(--nDevices > 1): betavar is bin x bin and the full-mode "
                "profile contracts over the bin axis, so it cannot be split "
                "along with the rest. Use --binByBinStatMode lite, or run "
                "single-device."
            )
        # Drop the previous generation before allocating the next one. The
        # old shard tensors stay reachable through the old graph closures
        # otherwise, so a rebuild -- arm_regularizers with a regularizer
        # attached, deepcopy (the saturated-model path in save_hists), any
        # re-init_fit_parms -- transiently doubles device memory on exactly
        # the tensors this class exists to keep small.
        if self.covarianceFit:
            raise NotImplementedError(
                "Multi-device fits are not supported with --covarianceFit "
                "(the dense data covariance couples all bins)."
            )

        devices = select_devices(self.n_devices)
        edges = shard_edges(self.indata.nbins, self.n_devices)
        logger.info(
            f"Sharding {self.indata.nbins} bins over {self.n_devices} "
            f"device(s): {[f'{d}:[{a},{b})' for d, (a, b) in zip(devices, edges)]}"
        )

        self._shard_graph_fns = []
        self.shards = []
        for device, (a, b) in zip(devices, edges):
            view = ShardIndataView(self.indata, a, b, device)
            shard = ShardEvaluator(device, view)
            # Slice on the host, then copy only the shard to the device. Doing
            # the slice inside the tf.device block places the StridedSlice on
            # the GPU, which requires its *input* -- the full [nbins, 9, nsyst]
            # tensor -- to be copied there first, defeating the host pinning in
            # FitInputData and exhausting the card before any shard exists (a
            # 92144-bin, 4618-systematic model needs 30.6 GB for that one
            # tensor, against 7.66 GB for the shard actually wanted).
            #
            # rnorm_init is the systematic_type == "normal" param-model factor
            # the hot path applies to the [nbins, nproc] contraction result
            # (see Fitter._init_logk_scaled). It is bin-indexed, so it shards
            # exactly like logk; None for log_normal, where the multiplicative
            # form already carries the scaling. The sparse companion
            # (rnorm_init_at_norm) is deliberately not threaded -- sharding
            # refuses sparse mode above, and the hot path only reads it behind
            # `if self.indata.sparse`, which is False on every shard view.
            with tf.device("/CPU:0"):
                logk_shard = tf.identity(self.logk[a:b])
                rnorm_init_shard = (
                    None
                    if self.rnorm_init is None
                    else tf.identity(self.rnorm_init[a:b])
                )
            with tf.device(device):
                shard.logk = tf.identity(logk_shard)
                shard.rnorm_init = (
                    None if rnorm_init_shard is None else tf.identity(rnorm_init_shard)
                )
                shard.nobs = tf.Variable(
                    tf.zeros([b - a], dtype=self.indata.dtype),
                    trainable=False,
                    name=f"nobs_shard{a}",
                )
                shard.lognobs = tf.Variable(
                    tf.zeros([b - a], dtype=self.indata.dtype),
                    trainable=False,
                    name=f"lognobs_shard{a}",
                )
                shard.varnobs = (
                    tf.Variable(
                        tf.ones([b - a], dtype=self.indata.dtype),
                        trainable=False,
                        name=f"varnobs_shard{a}",
                    )
                    if self.chisqFit
                    else None
                )
                shard.bbstat = BinByBinStat(
                    view,
                    self._options,
                    chisqFit=self.chisqFit,
                    covarianceFit=False,
                    data_cov_inv=None,
                    nobs_template=shard.nobs,
                )
            # seed from the current observation state: init_fit_parms (and
            # thus shard construction) can re-run after set_nobs
            with tf.device("/CPU:0"):
                nobs_shard = tf.identity(self.nobs[a:b])
                lognobs_shard = tf.identity(self.lognobs[a:b])
                varnobs_shard = (
                    tf.identity(self.varnobs[a:b])
                    if shard.varnobs is not None
                    else None
                )
            shard.nobs.assign(nobs_shard)
            shard.lognobs.assign(lognobs_shard)
            if shard.varnobs is not None:
                shard.varnobs.assign(varnobs_shard)
            shard.param_model = ShardParamModel(self.param_model, a, b)
            # Parameter-level state. NB the fitter's tf.Variables
            # (frozen_params_mask, blinding offsets) must NOT be captured
            # here: XLA-compiled functions cannot read a Variable resident
            # on a different device, so their *values* are threaded into the
            # per-shard functions as arguments instead (see
            # _make_sharded_tf_functions), the same way x is.
            shard.do_blinding = self.do_blinding
            shard.chisqFit = self.chisqFit
            shard.covarianceFit = False
            shard.data_cov_inv = None
            for name in self._SHARD_BORROWED_METHODS:
                setattr(shard, name, getattr(Fitter, name).__get__(shard))
            self.shards.append(shard)

        # the parameter-level NLL terms (constraints + external likelihood
        # terms), evaluated once per call on the first device via the same
        # borrowed-method pattern
        gview = ShardEvaluator(devices[0], self.indata)
        gview.param_model = self.param_model
        gview.frozen_params_mask = self.frozen_params_mask
        gview.do_blinding = self.do_blinding
        if self.do_blinding:
            for _name in _BLINDING_OFFSET_ATTRS:
                setattr(gview, _name, getattr(self, _name))
        gview.cw = self.cw
        gview.x0 = self.x0
        for name in ("get_poi", "get_model_nui", "get_theta", "get_x", "_compute_lc"):
            setattr(gview, name, getattr(Fitter, name).__get__(gview))
        self._global_view = gview

    def _hessp_batch_dispatch(self, P):
        return self._hessp_batch_sharded(P)

    def arm_regularizers(self):
        """Accept parameter-only regularizers; refuse yield-dependent ones.

        A penalty that reads the predicted yields cannot work here: it needs
        the full (masked-channel-inclusive) yield vector and no single device
        holds one. A penalty on the parameters alone has no such problem --
        every device has the whole parameter vector -- so it is evaluated once
        in the global term, gnll_local.

        Refusing is deliberate rather than a missing feature. gnll_local is
        the only global term, so a penalty not evaluated there contributes
        nothing at all: the fit would minimise the unregularized likelihood
        while the command line said otherwise, converge, and write a result
        with the bound unenforced. Its loss would also compare favourably with
        a single-device fit's, having omitted a positive term. None of that is
        visible from the output, so it has to fail here instead.

        The graphs are rebuilt here because __init__ traces them before
        --regularizer has been read, so a graph cached from an earlier call
        would have no penalty in it.
        """
        unsupported = [
            type(reg).__name__
            for reg in self.regularizers
            if getattr(reg, "needs_observables", True)
        ]
        if unsupported:
            raise NotImplementedError(
                f"Regularizers {unsupported} read the predicted yields "
                "(needs_observables=True), which multi-device mode "
                "(--nDevices > 1) cannot provide: the sharded likelihood never "
                "builds the full yield vector on any one device. Run this fit "
                "single-device (drop --nDevices), or, if the penalty really "
                "only depends on the parameters, set needs_observables=False "
                "on the regularizer."
            )
        super().arm_regularizers()
        if len(self.regularizers):
            self._make_tf_functions()

    def _unsharded(self, what, flag):
        raise NotImplementedError(
            f"{what} is not supported in multi-device mode (--nDevices > 1): it "
            "is computed over all bins on one device, which is what the "
            "sharding exists to avoid. Drop --nDevices, or rerun this step "
            f"single-device from the fit output. ({flag})"
        )

    # Everything below is reachable from rabbit_fit.py and runs *after* the
    # minimiser, over all bins at once. Inherited unchanged they would raise an
    # allocation failure at the end of a multi-hour fit and take the completed
    # result down with them, so they refuse at the point of use instead, before
    # the fit starts.
    def global_impacts_parms(self, *args, **kwargs):
        self._unsharded("Global impacts", "--globalImpacts")

    def gaussian_global_impacts_parms(self, *args, **kwargs):
        self._unsharded("Gaussian global impacts", "--gaussianGlobalImpacts")

    def loss_val_grad_hess_beta(self, *args, **kwargs):
        """Beta-space EDM diagnostic (--diagnostics with bin-by-bin stat).

        The base implementation evaluates the loss over the full logk on one
        device and then takes a jacobian over the full-length ubeta, i.e. an
        [nbinsfull, nbinsfull] Hessian. Like the other entries here it would
        succeed on a small model and fail on the sizes --nDevices exists for,
        after the minimiser has already finished.
        """
        self._unsharded(
            "The beta-space EDM diagnostic", "--diagnostics with bin-by-bin stat"
        )

    def impacts_parms(self, *args, **kwargs):
        """Per-nuisance impacts. Refused only when bin-by-bin stat is on.

        With BBB enabled the base implementation needs a second Hessian at
        profile=False to split out the stat-only covariance, and the sharded
        loss_val_grad_hess supports profile=True only. With BBB off that
        branch is skipped and the sharded path works, so this must not be a
        blanket refusal -- --doImpacts --nDevices N --noBinByBinStat is a
        working combination.
        """
        if self.bbstat.enabled:
            self._unsharded(
                "Impacts with bin-by-bin statistical uncertainties",
                "--doImpacts without --noBinByBinStat",
            )
        return super().impacts_parms(*args, **kwargs)

    def toyassign(self, *args, **kwargs):
        self._unsharded("Toy generation", "-t > 0")

    def loss_val_valfull_grad_hess(self, *args, **kwargs):
        self._unsharded("The full-NLL value/gradient/Hessian", "--fullNll")

    def reduced_nll(self):
        """Sharded counterpart of Fitter.reduced_nll.

        The base version routes through _compute_nll over all bins on one
        device, which is the same class of failure as expected_yield and the
        dense Hessian: it OOMs gathering the full result. It computes exactly
        what the sharded loss_val already sums -- _compute_nll(full_nll=False)
        -- so reuse that. This runs after the minimiser has finished, so
        failing here throws away the whole fit: the last one cost 6.3 hours of
        completed minimisation.
        """
        return self.loss_val()

    def full_nll(self):
        raise NotImplementedError(
            "--fullNll is not supported in multi-device mode (--nDevices > 1): "
            "the sharded likelihood is built with full_nll=False, so the "
            "normalisation terms are not available per shard. Rerun the "
            "postfit single-device with --externalPostfit if it is needed."
        )

    def expected_yield(self, profile=False, full=False):
        """Global all-bins yield, computed on the host.

        This path is not sharded: it runs the base implementation over the
        whole [nbins, 9, nsyst] logk, and on the default device that
        materialises the full tensor on GPU:0 -- 30.6 GB for a 92144-bin,
        4618-systematic model, which is precisely what sharding exists to
        avoid. It is needed once at construction (nexpnom) and in postfit
        paths, never inside the minimiser loop, so paying for it in host
        memory costs one slow pass rather than the fit.
        """
        with tf.device("/CPU:0"):
            return super().expected_yield(profile=profile, full=full)

    def expected_events(self, *args, **kwargs):
        """Postfit yields and their variances, computed on the host.

        Same reasoning as expected_yield above, and the same tensor: this is
        the all-bins path, so on the default device it materialises the full
        logk on GPU:0. --saveHists calls it once per mapping immediately after
        the minimiser, so leaving it unpinned turns a finished multi-hour fit
        into an OOM at the point of writing the output.

        Pinned rather than refused: refusing would leave a sharded fit unable
        to write histograms at all, which is a large usability loss for a
        feature whose point is to make big fits runnable. The cost is one slow
        host pass per mapping, outside the minimiser loop.
        """
        with tf.device("/CPU:0"):
            return super().expected_events(*args, **kwargs)

    def expected_with_variance(self, *args, **kwargs):
        """Host-pinned; see expected_events."""
        with tf.device("/CPU:0"):
            return super().expected_with_variance(*args, **kwargs)

    def expected_variations(self, *args, **kwargs):
        """Host-pinned; see expected_events."""
        with tf.device("/CPU:0"):
            return super().expected_variations(*args, **kwargs)

    def chi2(self, *args, **kwargs):
        """Host-pinned; see expected_events."""
        with tf.device("/CPU:0"):
            return super().chi2(*args, **kwargs)

    def _dxdvars(self, *args, **kwargs):
        """Response of the postfit minimum to the constraint centers, on the host.

        An all-bins computation -- _compute_loss over every bin, then
        [npar, nbinsfull] jacobians -- so on the default device it
        materialises the full logk on GPU:0.

        Pinned rather than refused, because it is a shared primitive rather
        than an entry point: _dndvars calls it unconditionally, which puts it
        under chi2(profile=True) and the gaussian-global-impacts branch of
        expected_with_variance, i.e. the --saveHists path this class
        deliberately keeps working. Refusing here would take those down with
        it. The one entry point that is genuinely unsupported,
        --globalAsymImpacts with --globalAsymImpactsLinearWarmstart, is
        refused up front in rabbit_fit.py instead.
        """
        with tf.device("/CPU:0"):
            return super()._dxdvars(*args, **kwargs)

    def set_nobs(self, values, variances=None):
        super().set_nobs(values, variances)
        nobssafe = tf.where(values == 0.0, tf.constant(1.0, dtype=values.dtype), values)
        for shard in self.shards:
            a, b = shard.indata.start, shard.indata.stop
            shard.nobs.assign(values[a:b])
            shard.lognobs.assign(tf.math.log(nobssafe[a:b]))
            if shard.varnobs is not None:
                shard.varnobs.assign(self.varnobs[a:b])

    def _make_tf_functions(self):
        """Multi-device counterparts of the loss/grad/HVP/Hessian wrappers.

        Rebuilds the shards first: this method runs at the end of
        init_fit_parms and after a deepcopy, exactly the moments the shard
        evaluators must be reconstructed against the current parameter
        layout.

        The per-shard functions are jit-compiled on their own device and
        differentiate with respect to their local x copy (one tape per
        shard); the combiners below are plain graphs that broadcast x,
        collect the [nparams]-sized partials and sum them. See
        rabbit/sharding.py for why both choices are load-bearing.
        """
        # Release the previous generation before _build_shards allocates the
        # next one, or both exist at once and device memory peaks at 2x on
        # exactly the tensors this class is for. Rebuilds happen on
        # arm_regularizers with a regularizer attached, on deepcopy (the
        # saturated-model path in save_hists) and on any re-init_fit_parms.
        #
        # Both steps are required. self._profile_beta holds a live reference to
        # the old shards until it is replaced at the end of this method, so the
        # attributes have to go first; what remains is cyclic (a tf.function
        # graph references its captures and its siblings), so refcounting alone
        # cannot reclaim it and the device memory would be held until the
        # generational collector happened to run.
        for name in self._DYNAMIC_TF_FUNCS:
            self.__dict__.pop(name, None)
        gc.collect()

        self._build_shards()

        jit = self.jit_compile

        def make_shard_fns(shard):
            # aux = (frozen_params_mask value[, blinding offsets]) -- Variable
            # values threaded in as tensors because XLA cannot read a
            # Variable on another device
            def _pin(x, aux):
                shard.x = x
                shard.frozen_params_mask = aux[0]
                if shard.do_blinding:
                    for _k, _name in enumerate(_BLINDING_OFFSET_ATTRS, start=1):
                        setattr(shard, _name, aux[_k])

            def nll_local(x, aux):
                _pin(x, aux)
                nexp, _, beta = shard._compute_yields_with_beta(
                    profile=True, compute_norm=False, full=False
                )
                ln = shard._compute_ln(nexp[: shard.indata.nbins], full_nll=False)
                lbeta = shard._compute_lbeta(beta, full_nll=False)
                return ln + lbeta if lbeta is not None else ln

            def vg(x, aux):
                with tf.GradientTape() as t:
                    t.watch(x)
                    v = nll_local(x, aux)
                g = t.gradient(v, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                return v, g

            def vgp(x, aux, p):
                p = tf.stop_gradient(p)
                with tf.GradientTape() as t2:
                    t2.watch(x)
                    with tf.GradientTape() as t1:
                        t1.watch(x)
                        v = nll_local(x, aux)
                    g = t1.gradient(
                        v, x, unconnected_gradients=tf.UnconnectedGradients.ZERO
                    )
                hp = t2.gradient(
                    g,
                    x,
                    output_gradients=p,
                    unconnected_gradients=tf.UnconnectedGradients.ZERO,
                )
                return v, g, hp

            def beta_local(x, aux):
                _pin(x, aux)
                _, _, beta = shard._compute_yields_with_beta(
                    profile=True, compute_norm=False, full=False
                )
                return beta

            return (
                tf.function(nll_local, jit_compile=jit),
                tf.function(vg, jit_compile=jit),
                tf.function(vgp, jit_compile=jit),
                tf.function(beta_local, jit_compile=jit),
            )

        self._shard_graph_fns = [make_shard_fns(shard) for shard in self.shards]

        gview = self._global_view

        def gnll_local(x):
            gview.x = x
            total = gview._compute_lc(full_nll=False)
            lext = external_likelihood.compute_external_nll(
                self.external_terms, x, self.indata.dtype, full_nll=False
            )
            if lext is not None:
                total = total + lext
            # Unbinned terms are NOT sharded: they are a sum over candidates,
            # not over bins, and nothing in ShardIndataView carries a candidate
            # axis. Evaluating them here keeps the sharded likelihood CORRECT
            # (leaving them out silently drops them, which is what this path
            # did before), at the price of their per-candidate tensors living
            # on one device and their compute not being split. Sharding them
            # properly means a candidate-range shard view -- see the note in
            # `_build_shards`.
            lunb = unbinned_terms_mod.compute_unbinned_nll(
                self.unbinned_terms, gview.get_x(), self.indata.dtype, full_nll=False
            )
            if lunb is not None:
                total = total + lunb
            # Regularizer penalties belong here rather than in a shard: they
            # are global (a shard would double count them) and, for the ones
            # allowed on this path, depend only on the parameter vector, which
            # every device has in full. gvg and gvgp both differentiate
            # gnll_local, so gradient, HVP and Hessian follow automatically.
            if len(self.regularizers):
                # Same guard as Fitter._compute_nll_components: an unarmed
                # regularizer would otherwise contribute a penalty whose
                # internals were never set, or be omitted entirely if the
                # graphs were traced before it was attached -- either way the
                # objective drifts silently.
                if not getattr(self, "_regularizers_armed", False):
                    raise RuntimeError(
                        "Regularizers not armed; call arm_regularizers() "
                        "after attaching them (defaultassign does this)."
                    )
                penalty = tf.add_n(
                    [
                        reg.compute_nll_penalty(gview.get_x(), None)
                        for reg in self.regularizers
                    ]
                )
                total = total + penalty * tf.exp(2 * self.tau)
            return total

        def gvg(x):
            with tf.GradientTape() as t:
                t.watch(x)
                v = gnll_local(x)
            g = t.gradient(v, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            return v, g

        def gvgp(x, p):
            p = tf.stop_gradient(p)
            with tf.GradientTape() as t2:
                t2.watch(x)
                with tf.GradientTape() as t1:
                    t1.watch(x)
                    v = gnll_local(x)
                g = t1.gradient(
                    v, x, unconnected_gradients=tf.UnconnectedGradients.ZERO
                )
            hp = t2.gradient(
                g,
                x,
                output_gradients=p,
                unconnected_gradients=tf.UnconnectedGradients.ZERO,
            )
            return v, g, hp

        self._global_graph_fns = tuple(
            tf.function(f, jit_compile=jit) for f in (gnll_local, gvg, gvgp)
        )

        def _read_aux():
            aux = [tf.identity(self.frozen_params_mask)]
            if self.do_blinding:
                aux.extend(
                    tf.identity(getattr(self, _name))
                    for _name in _BLINDING_OFFSET_ATTRS
                )
            return tuple(aux)

        def _to_device(aux):
            return tuple(tf.identity(a) for a in aux)

        def _loss_val():
            x0 = tf.identity(self.x)
            aux = _read_aux()
            parts = []
            for shard, fns in zip(self.shards, self._shard_graph_fns):
                with tf.device(shard.device):
                    parts.append(fns[0](tf.identity(x0), _to_device(aux)))
            with tf.device(self.shards[0].device):
                parts.append(self._global_graph_fns[0](tf.identity(x0)))
            return tf.add_n(parts)

        def _loss_val_grad():
            x0 = tf.identity(self.x)
            aux = _read_aux()
            vs, gs = [], []
            for shard, fns in zip(self.shards, self._shard_graph_fns):
                with tf.device(shard.device):
                    v, g = fns[1](tf.identity(x0), _to_device(aux))
                vs.append(v)
                gs.append(g)
            with tf.device(self.shards[0].device):
                v, g = self._global_graph_fns[1](tf.identity(x0))
            vs.append(v)
            gs.append(g)
            return tf.add_n(vs), tf.add_n(gs)

        def _loss_val_grad_hessp(p):
            x0 = tf.identity(self.x)
            aux = _read_aux()
            vs, gs, hs = [], [], []
            for shard, fns in zip(self.shards, self._shard_graph_fns):
                with tf.device(shard.device):
                    v, g, hp = fns[2](tf.identity(x0), _to_device(aux), tf.identity(p))
                vs.append(v)
                gs.append(g)
                hs.append(hp)
            with tf.device(self.shards[0].device):
                v, g, hp = self._global_graph_fns[2](tf.identity(x0), tf.identity(p))
            vs.append(v)
            gs.append(g)
            hs.append(hp)
            return tf.add_n(vs), tf.add_n(gs), tf.add_n(hs)

        def _loss_val_grad_hessp_batch(P):
            """Batched HVP, kept sharded: [k, npar] directions in, sum out.

            Mirrors _loss_val_grad_hessp but vectorises over the batch inside
            each shard's graph, so the columns of the Hessian are built on the
            GPUs. vectorized_map is a pfor of width k, unlike tape.jacobian's
            non-pfor path whose while_loop XLA unrolls, so memory scales with k
            and compilation stays bounded.

            Deliberately NOT wrapped in tf.function, unlike every sibling
            here: hessian_from_hvps halves k and retries on
            ResourceExhaustedError, which needs the error to surface from the
            eager call rather than from a traced graph, and k changes between
            calls so a wrapper would retrace on every new batch width anyway.
            The retracing warning this produces is expected.
            """
            x0 = tf.identity(self.x)
            aux = _read_aux()
            parts = []
            for shard, fns in zip(self.shards, self._shard_graph_fns):
                with tf.device(shard.device):
                    a = _to_device(aux)
                    parts.append(
                        tf.vectorized_map(
                            lambda p, f=fns, a=a: f[2](tf.identity(x0), a, p)[2], P
                        )
                    )
            with tf.device(self.shards[0].device):
                parts.append(
                    tf.vectorized_map(
                        lambda p: self._global_graph_fns[2](tf.identity(x0), p)[2], P
                    )
                )
            return tf.add_n([tf.identity(x) for x in parts])

        self._hessp_batch_sharded = _loss_val_grad_hessp_batch

        def _profile_beta_sharded():
            x0 = tf.identity(self.x)
            aux = _read_aux()
            betas = []
            for shard, fns in zip(self.shards, self._shard_graph_fns):
                with tf.device(shard.device):
                    beta = fns[3](tf.identity(x0), _to_device(aux))
                shard.bbstat.beta.assign(beta)
                betas.append(beta)
            # mirror into the fitter-level beta variable, whose masked-bin
            # tail (never touched by the sharded loss) keeps its prior value
            nbins = self.indata.nbins
            full = tf.concat(betas + [self.bbstat.beta[nbins:]], axis=0)
            self.bbstat.beta.assign(full)

        self.loss_val = tf.function(_loss_val)
        self.loss_val_grad = tf.function(_loss_val_grad)
        self.loss_val_grad_hessp_revrev = tf.function(_loss_val_grad_hessp)
        if self.hvp_method == "fwdrev":
            logger.warning(
                "fwdrev HVP is not supported in multi-device mode; "
                "falling back to revrev."
            )
        self.loss_val_grad_hessp_fwdrev = self.loss_val_grad_hessp_revrev
        self.loss_val_grad_hessp = self.loss_val_grad_hessp_revrev

        # instance override of the class-level @tf.function
        def _loss_val_grad_hess_hvp(profile=True):
            """Value, gradient and dense Hessian for the sharded fit.

            The Hessian comes from batched HVP columns rather than the sharded
            jacobian. The jacobian path vectorises over every parameter inside
            each shard and needs ~70 GB on its device for a 6538-parameter
            model, so it cannot run at all here -- and it is what the postfit
            covariance calls, meaning the failure would land at the *end* of a
            multi-hour fit. The HVPs are the same ones the minimiser uses every
            iteration, so they stay distributed and cost one [k, npar] batch of
            memory.
            """
            if not profile:
                raise NotImplementedError(
                    "profile=False Hessians are not supported in multi-device "
                    "mode (--nDevices > 1)."
                )
            val, grad = self.loss_val_grad()
            hess = tf.constant(self.hessian_from_hvps(), dtype=self.indata.dtype)
            return val, grad, hess

        self.loss_val_grad_hess = _loss_val_grad_hess_hvp
        if self.bbstat.enabled:
            self._profile_beta = tf.function(_profile_beta_sharded)
