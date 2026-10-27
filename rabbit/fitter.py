import contextlib
import time

import h5py
import numpy as np
import scipy
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops as tf_sparse_csr
from wums import logging

from rabbit import external_likelihood, io_tools
from rabbit import preconditioner as precond
from rabbit import tfhelpers as tfh
from rabbit.bbstat.bbstat import BinByBinStat
from rabbit.blinding import Blinding
from rabbit.callbacks import RESTART_MIN_IMPROVEMENT, FitterCallback, merge_callbacks
from rabbit.common import match_regexp_params
from rabbit.constraints import postfit as constraints_postfit
from rabbit.constraints import scipy_adapter
from rabbit.constraints import warm_start as tc_warm
from rabbit.impacts import (
    asym_impacts,
    global_asym_impacts,
    global_impacts,
    nonprofiled_impacts,
    traditional_impacts,
)
from rabbit.minimizer import (
    minimize_trust_exact,
    minimize_trust_krylov,
    minimize_trust_ncg,
)
from rabbit.param_models.param_model import (
    compute_batched as param_model_compute_batched,
)
from rabbit.snapshot import Snapshotter, snapshot_on_signal
from rabbit.tfhelpers import edmval_cov

logger = logging.child_logger(__name__)

# Supported constraint-Hessian modes for contour_scan (see its docstring for
# what each mode does). Single source of truth for the CLI choices and the
# contour_scan validation.
CONTOUR_HESS_MODES = ("exact", "hvp", "frozen", "bfgs", "sr1")


def make_fitter(indata, param_model, options, **kwargs):
    """Construct the Fitter appropriate for the requested device count.

    The device layout is fixed at initialization, so the choice is a
    static one: --nDevices > 1 returns the bins-sharded
    :class:`rabbit.sharding.MultiDeviceFitter` subclass, anything else the
    plain single-device Fitter. Imported lazily to avoid a module cycle
    (sharding subclasses Fitter). All keyword arguments are forwarded
    verbatim, so this wrapper cannot drift from Fitter.__init__'s
    signature.
    """
    if int(getattr(options, "nDevices", 1) or 1) > 1:
        from rabbit.sharding import MultiDeviceFitter

        return MultiDeviceFitter(indata, param_model, options, **kwargs)
    return Fitter(indata, param_model, options, **kwargs)


# scipy methods that honour hard constraints (--constraints,
# rabbit.constraints.Constraints).
CONSTRAINED_METHODS = ("trust-constr",)

# Options from --minimizerMaxiter/--minimizerGtol/--minimizerFtol that each
# native minimizer actually reads. Anything else passed for these methods is
# warned about rather than silently dropped (see Fitter.fit).
NATIVE_MINIMIZER_OPTIONS = {
    "tf-trust-exact": {"gtol", "maxiter"},
    "tf-trust-ncg": {"gtol", "maxiter"},
    "tf-trust-krylov": {"gtol", "maxiter"},
}


def weighted_chi2_sf(x, weights):
    """P(sum_j w_j Z_j^2 > x) for independent standard normal Z_j and w_j > 0,
    by Imhof's (1961) inversion of the characteristic function (exact up to the
    numerical integration). Equal weights reduce to the chi2 survival function."""
    import scipy.integrate
    import scipy.stats

    w = np.asarray(weights, dtype=np.float64)
    w = w[w > 1e-12 * max(np.max(w, initial=0.0), 1e-300)]
    if len(w) == 0:
        return float(x < 0)
    if np.allclose(w, w[0], rtol=1e-12):
        return float(scipy.stats.chi2.sf(x / w[0], len(w)))

    def amp(u):
        return 1.0 / (u * np.prod((1.0 + (w * u) ** 2) ** 0.25))

    def half_atan(u):
        return 0.5 * np.sum(np.arctan(w * u))

    def integrand(u):
        return np.sin(half_atan(u) - 0.5 * x * u) * amp(u)

    # sin(A(u) - x u / 2) = sin A cos(x u / 2) - cos A sin(x u / 2) with a slowly
    # varying A: a smooth start, then Fourier-weighted quadrature (QAWF) for the
    # oscillatory tail
    split = 1.0 / np.max(w)
    head = scipy.integrate.quad(integrand, 0.0, split, limit=200)[0]
    if x > 0:
        tail = (
            scipy.integrate.quad(
                lambda u: np.sin(half_atan(u)) * amp(u),
                split,
                np.inf,
                weight="cos",
                wvar=0.5 * x,
            )[0]
            - scipy.integrate.quad(
                lambda u: np.cos(half_atan(u)) * amp(u),
                split,
                np.inf,
                weight="sin",
                wvar=0.5 * x,
            )[0]
        )
    else:
        tail = scipy.integrate.quad(integrand, split, np.inf, limit=500)[0]
    return float(min(max(0.5 + (head + tail) / np.pi, 0.0), 1.0))


def _is_mcstat_term(term):
    """External term written by TensorWriter.add_mc_stat_moment."""
    return str(term["name"]).startswith("mcstat")


class Fitter:
    # Dynamically-built tf.function wrappers holding un-copyable FuncGraph
    # state; stripped on deepcopy and rebuilt. Subclasses extend this with
    # their own dynamic machinery (see rabbit.sharding.MultiDeviceFitter).
    _DYNAMIC_TF_FUNCS = frozenset(
        {
            "loss_val",
            "loss_val_grad",
            "loss_val_grad_hessp",
            "loss_val_grad_hessp_fwdrev",
            "loss_val_grad_hessp_revrev",
            "_hessp_batch_graph",
            "_scalar_hessian_fns",
            # the traced functions of the other de-biasing setting
            # (_set_mcstat_debias_suspended)
            "_mcstat_tf_standby",
        }
    )
    valid_systematic_types = ["log_normal", "normal"]

    def __init__(
        self, indata, param_model, options, globalImpactsFromJVP=True, do_blinding=False
    ):
        self.indata = indata

        self.earlyStopping = options.earlyStopping
        self.stallRelTol = getattr(options, "stallRelTol", 0.0)
        self.globalImpactsFromJVP = globalImpactsFromJVP

        if self.indata.systematic_type not in Fitter.valid_systematic_types:
            raise RuntimeError(
                f"Invalid systematic_type {self.indata.systematic_type}, valid choices are {Fitter.valid_systematic_types}"
            )

        self.diagnostics = options.diagnostics
        self.minimizer_method = options.minimizerMethod
        # Optional scipy.optimize.minimize tolerances. None entries are
        # skipped at call site so each method falls back to scipy defaults
        # for what's not set.
        self.minimizer_maxiter = getattr(options, "minimizerMaxiter", None)
        self.minimizer_gtol = getattr(options, "minimizerGtol", None)
        self.minimizer_ftol = getattr(options, "minimizerFtol", None)
        self.minimizer_xtol = getattr(options, "minimizerXtol", None)
        self.minimizer_barrier_tol = getattr(options, "minimizerBarrierTol", None)
        # trust-constr interior-point start: None -> scipy's default
        self.minimizer_tc_init = {
            "initial_barrier_parameter": getattr(
                options, "minimizerInitialBarrier", None
            ),
            "initial_barrier_tolerance": getattr(
                options, "minimizerInitialBarrierTol", None
            ),
            "initial_tr_radius": getattr(options, "minimizerInitialTrRadius", None),
        }
        # trust-constr + --constraints only: how to start a fit whose starting
        # point is a previous minimum (rabbit.constraints.warm_start)
        self.minimizer_warm_start = getattr(options, "minimizerWarmStart", "auto")
        # what the last fit() did about it; None = a plain (cold) start
        self._warm_start_info = None
        self._warm_barrier = tc_warm.WARM_INITIAL_BARRIER
        # scipy's OptimizeResult from the last fit(), so the convergence
        # outcome can be written to the output. None if the minimizer raised.
        self.minimizer_result = None
        # whether the last fit() ended under a non-identity preconditioner:
        # scipy's norms (trust-constr's optimality) are then in its frame
        self._fit_preconditioned = False
        self.hvp_method = getattr(options, "hvpMethod", "revrev")
        hvp_batch = getattr(options, "hvpBatch", None)
        # Only an explicit --hvpBatch makes the single-device fit assemble its
        # dense Hessian from HVPs; by default it keeps the one-pass
        # tape.jacobian. The multi-device fit always uses HVPs, 256 at a time
        # unless told otherwise.
        self.hessian_from_hvp = hvp_batch is not None
        # `or 256` would turn an explicit 0 into the default; test None
        self.hvp_batch = 256 if hvp_batch is None else int(hvp_batch)
        # Optional parameter preconditioning (see rabbit/preconditioner.py).
        # getattr so callers that build options objects by hand keep working.
        # Parameter snapshots (see rabbit/snapshot.py). Everything the fit has
        # learned lives in memory until the output is written at the very end,
        # so without these an interrupted fit leaves nothing at all.
        self.snapshot_file = getattr(options, "snapshotFile", None)
        self.snapshot_interval = float(getattr(options, "snapshotInterval", 0.0) or 0.0)
        self.precondition = getattr(options, "precondition", False)
        self.precondition_params = getattr(options, "preconditionParams", None)
        self.precondition_from = getattr(options, "preconditionFrom", "hessian")
        self.precondition_blocks = getattr(options, "preconditionBlocks", "auto")
        self.precondition_transform = getattr(options, "preconditionTransform", "ridge")
        self.precondition_block_threshold = getattr(
            options, "preconditionBlockThreshold", 0.1
        )
        self.max_restarts = getattr(options, "maxRestarts", -1)
        self.precondition_ridge = getattr(options, "preconditionRidge", 1e-8)
        # jitCompile accepts "auto" (the default), "on", or "off".
        # True / False from programmatic callers are accepted as
        # aliases for "on" / "off". The tri-state is resolved to the
        # final boolean self.jit_compile right here, using the only
        # runtime condition it can depend on: whether the input is
        # sparse. Sparse mode uses SparseMatrixMatMul which has no
        # XLA kernel, so "auto" silently disables jit and "on" warns
        # and falls back.
        _jit_opt = getattr(options, "jitCompile", "auto")
        if _jit_opt is True:
            _jit_opt = "on"
        elif _jit_opt is False:
            _jit_opt = "off"
        if _jit_opt not in ("auto", "on", "off"):
            raise ValueError(
                f"jitCompile must be one of 'auto', 'on', 'off'; got {_jit_opt!r}"
            )
        if _jit_opt == "off":
            self.jit_compile = False
        elif _jit_opt == "on":
            if self.indata.sparse:
                logger.warning(
                    "--jitCompile=on requested but input data is sparse; "
                    "XLA has no kernel for the sparse matmul ops used in "
                    "sparse mode, so jit_compile will be disabled."
                )
                self.jit_compile = False
            else:
                self.jit_compile = True
        else:  # "auto"
            self.jit_compile = not self.indata.sparse
        # When --noHessian is requested the postfit Hessian is never
        # computed, so the dense [npar, npar] covariance matrix should
        # not be allocated. self.cov is set to None in that case and
        # callers must use self.var_prefit (the diagonal vector form)
        # for prefit uncertainties instead.
        self.compute_cov = not getattr(options, "noHessian", False)

        if options.covarianceFit and options.chisqFit:
            raise Exception(
                'Use either "--covarianceFit" for chi-squared fit using covariance or "--chisqFit" for diagonal chi-squared fit'
            )

        self.chisqFit = options.chisqFit
        self.covarianceFit = options.covarianceFit
        # the per-bin variance of the chi2 (--chisqVariance, see
        # _chisq_variance): "data" or "mc", None for the other fits
        chisq_variance = getattr(options, "chisqVariance", None)
        if chisq_variance is not None and not self.chisqFit:
            raise ValueError(
                f"--chisqVariance {chisq_variance} only applies to --chisqFit: "
                "--covarianceFit takes the data covariance of the input, and the "
                "Poisson likelihood has no variance to choose."
            )
        self.chisqVariance = (chisq_variance or "data") if self.chisqFit else None
        # When True, self.is_linear is forced True regardless of the actual
        # linearity of the param model, asymmetry tensor, or systematic type.
        # The minimize() path then takes a single Cholesky / Hessian-CG step
        # from the current x, i.e. a Gaussian approximation around that point.
        # Convergence to the true minimum then requires an outer iteration
        # that re-anchors the linearization point.
        self.force_linear = getattr(options, "forceLinear", False)

        # Mirrored by self.blinding.enabled, which is created in
        # init_fit_parms (it needs the param model). External callers and
        # the shard evaluators read this flag; the Fitter's own blinding
        # branches go through the collaborator.
        self.do_blinding = do_blinding
        # Read here rather than passed to init_fit_parms, so the saturated-test
        # re-init in bin/rabbit_fit.py keeps the same frame without plumbing.
        self.blinding_frame = getattr(options, "blindingFrame", "auto")
        self.prefit_unconstrained_nuisance_uncertainty = (
            options.prefitUnconstrainedNuisanceUncertainty
        )

        # --- observed number of events per bin
        self.nobs = tf.Variable(
            tf.zeros_like(self.indata.data_obs), trainable=False, name="nobs"
        )
        self.lognobs = tf.Variable(
            tf.zeros_like(self.indata.data_obs), trainable=False, name="lognobs"
        )

        self.varnobs = None
        self.data_cov_inv = None

        if self.chisqFit:
            self.varnobs = tf.Variable(
                tf.zeros_like(self.indata.data_obs), trainable=False, name="varnobs"
            )
        elif self.covarianceFit:
            if self.indata.data_cov_inv is None:
                logger.warning(
                    "No covariance provided, use reciproval of data variances"
                )
                self.data_cov_inv = tf.linalg.diag(
                    1.0 / self.indata.getattr("data_obs", "data_var")
                )
            else:
                # provided covariance
                self.data_cov_inv = self.indata.data_cov_inv

        # --- bin-by-bin statistical treatment (β nuisances + masks + kstat).
        # All BBB state is owned by the BinByBinStat helper. Constructed
        # here, before init_fit_parms, because init_fit_parms's is_linear
        # computation reads self.bbstat.enabled.
        self.bbstat = BinByBinStat(
            indata,
            options,
            chisqFit=self.chisqFit,
            covarianceFit=self.covarianceFit,
            data_cov_inv=self.data_cov_inv,
        )

        # Hard constraints (rabbit.constraints.Constraints, --constraints) for
        # the constrained minimizers. Separate from the regularizers: those are
        # terms of -ln(L) and stay in the loss under every minimizer. Set here,
        # NOT in init_fit_parms: like the param model they must survive a
        # re-init (the saturated test re-runs init_fit_parms with a composite
        # model, and its sub-fit must be constrained like the nominal one);
        # arm_regularizers re-resolves their positions for the new layout.
        self.constraints = []
        # what build_scipy_constraints kept (rows, bounds, labels); None = not built
        self._constraint_info = None
        # the faces active at the last constrained post-fit (constrained_postfit):
        # {"jacobian": A [m, n], "hess_constraints":
        # sum_i mu_i d2c_i [n, n]}, None when no face is active. The impacts
        # are then taken with these faces held.
        self.postfit_faces = None

        # --- limited-MC-statistics de-biasing options (see the block after
        # init_fit_parms below and docs/mcstat/RABBIT_MCSTAT_DESIGN.md)
        self.mcStatDebias = getattr(options, "mcStatDebias", "none")
        self.mcStatDebiasCov = getattr(options, "mcStatDebiasCov", "sandwich")
        self.covMode = getattr(options, "covMode", "observed")
        # the --covMode asked for; set_mcstat_asimov switches self.covMode to
        # fisher for an Asimov dataset of a de-biased fit
        self._covMode_requested = self.covMode
        self._mcstat_asimov_on = False
        # True while minimize() runs the ordinary-likelihood warm start
        self._mcstat_debias_suspended = False

        # the chi2 variance of --chisqVariance mc [nbins]; reads the bin-by-bin
        # stat and the de-biasing options above
        self.mc_variance = (
            self._mc_chisq_variance() if self.chisqVariance == "mc" else None
        )

        # --- fit params
        self.init_fit_parms(
            param_model,
            options.setConstraintMinimum,
            unblind=options.unblind,
            blinding_group=options.blindingGroup,
            freeze_parameters=options.freezeParameters,
        )

        self.nexpnom = tf.Variable(
            self.expected_yield(), trainable=False, name="nexpnom"
        )

        # --- limited-MC-statistics de-biasing (docs/mcstat/RABBIT_MCSTAT_DESIGN.md)
        # The options themselves are read before init_fit_parms (which needs
        # mcStatDebias to decide whether the M penalty term is kept).
        # self.mcstat_M (the continuous-M matrix) is built in init_fit_parms.
        self._check_mcstat_moment_variance()
        # Cross-fit (two-half / k-fold) fold templates, see _init_mcstat_folds.
        self._init_mcstat_folds()
        self.mcstat_debias_active = (
            self.mcStatDebias == "continuousM" and self.mcstat_M is not None
        ) or self._fold_debias_active()
        # Reference per-bin data variance D and total variance V of the
        # bin-by-bin stat compensator (_compute_mcstat_bb_compensator); None if
        # there is nothing to compensate.
        self.mcstat_bb_dref = None
        self.mcstat_bb_vref = None
        if self.mcstat_debias_active and self._bbstat_variance_moves():
            zeros = tf.zeros([self.indata.nbins], dtype=self.indata.dtype)
            self.mcstat_bb_dref = tf.Variable(
                zeros, trainable=False, name="mcstat_bb_dref"
            )
            self.mcstat_bb_vref = tf.Variable(
                zeros, trainable=False, name="mcstat_bb_vref"
            )
            self._update_mcstat_bb_vref()
        # betas of the separately profiled half terms (_profiled_half_terms),
        # held fixed when not profiling
        self.mcstat_half_beta = None
        if self._fold_debias_active() and self.bbstat.enabled:
            self.mcstat_half_beta = tf.Variable(
                tf.stack([self.bbstat.beta0, self.bbstat.beta0]),
                trainable=False,
                name="mcstat_half_beta",
            )

        if self.mcStatDebias == "kfold" and self.mcStatDebiasCov == "dataPropagated":
            # cov_dataprop_sandwich differentiates the full templates only; the
            # fold objectives depend on the separately stored fold templates,
            # whose own MC variances are not stored
            raise NotImplementedError(
                "--mcStatDebiasCov dataPropagated is only available for "
                "--mcStatDebias continuousM: it propagates the variance of the full "
                "templates, not of the fold templates a two-half / k-fold fit "
                "depends on."
            )
        if self.mcStatDebias == "kfold" and self.n_folds is None:
            logger.warning(
                f"--mcStatDebias {self.mcStatDebias} requested but no fold "
                "templates (hnorm_folds) found in the input; two-half de-biasing "
                "is inactive. Add processes with a fold_axis in the TensorWriter."
            )
        if self.n_folds is not None and self.n_folds >= 3 and self.covMode != "fisher":
            logger.info(
                f"k-fold de-biasing with k={self.n_folds} folds: the k-fold "
                "AVERAGING (complete U-statistic, lower curvature-estimate "
                "variance) is realized only in --covMode fisher. In observed mode "
                "the curvature is the Hessian of the single A/B-split objective "
                "(no averaging); pass --covMode fisher to average over all fold "
                "pairs."
            )
        for msg in self.mcstat_coverage_warnings():
            logger.warning(msg)
        if self.mcStatDebias == "continuousM":
            if self.mcstat_M is None:
                logger.warning(
                    "--mcStatDebias continuousM requested but no 'mcstat' external "
                    "moment term found in the input; continuous-M is inactive. Supply "
                    "M via TensorWriter.add_mc_stat_moment(M, param_names, "
                    "variance=...)."
                )
            elif not self._mcstat_M_params_linear():
                # The -1/2 theta^T M theta penalty only de-biases parameters that
                # enter the prediction LINEARLY. If any parameter that M touches
                # is nonlinear (e.g. the default x^2 POI transform, or log-normal
                # systematics), the data Fisher grows as the -M theta gradient
                # shifts the minimum and cancels the -M curvature subtraction
                # (docs/mcstat/RESULTS.md §9a) -> no de-bias of the point or
                # covariance for those parameters. Two-half de-biasing has no such
                # restriction.
                logger.warning(
                    "--mcStatDebias continuousM: the supplied M acts on parameters "
                    "that enter the prediction NONLINEARLY (e.g. the default rabbit "
                    "x^2 POI transform [use --allowNegativeParam for a linear POI], "
                    "or log_normal systematics). The -1/2 theta^T M theta penalty is "
                    "cancelled by Fisher growth at the shifted minimum and will NOT "
                    "de-bias the point or covariance for those parameters "
                    "(docs/mcstat/RESULTS.md §9a). Use a linear parametrization for "
                    "the de-biased parameters, or the cross-fit method "
                    "(--mcStatDebias kfold)."
                )

    def init_fit_parms(
        self,
        param_model,
        set_constraint_minimum=[],
        unblind=False,
        blinding_group=[],
        freeze_parameters=[],
    ):
        self.param_model = param_model

        # Param-model scaling of the systematic variations for the
        # yield-computation hot path; see _init_logk_scaled.
        self._init_logk_scaled()

        # All blinding state -- the drawn offsets, the Variables holding
        # the armed ones, and the hot-path application -- belongs to the
        # Blinding helper. Built here rather than in __init__ because the
        # offsets are sized by the param model, which a re-run of
        # init_fit_parms replaces (the saturated-test path in
        # bin/rabbit_fit.py does exactly that, and re-arms afterwards).
        self.blinding = Blinding(
            self.indata,
            self.param_model,
            enabled=self.do_blinding,
            unblind=unblind,
            blinding_group=blinding_group,
            frame=getattr(self, "blinding_frame", "auto"),
        )

        self.parms = np.concatenate([self.param_model.params, self.indata.systs])
        # Layout changed: anything a regularizer resolved is now stale. Marked
        # rather than re-armed here because this runs from __init__, before
        # regularizers are attached; discharged by arm_regularizers().
        self._regularizers_armed = False

        # tf tensor containing default constraint minima
        theta0default = np.zeros(self.indata.nsyst)
        for parm, val in set_constraint_minimum:
            idx = np.where(self.indata.systs.astype(str) == parm)[0]
            if len(idx) != 1:
                raise RuntimeError(
                    f"Expect to find exactly one match for {parm} to set constraint minimum, but found {len(idx)}"
                )
            theta0default[idx[0]] = val

        self.theta0default = tf.convert_to_tensor(
            theta0default, dtype=self.indata.dtype
        )

        # tf variable containing all fit parameters
        if self.param_model.nparams > 0:
            xdefault = tf.concat(
                [self.param_model.xparamdefault, self.theta0default], axis=0
            )
        else:
            xdefault = self.theta0default

        self.x = tf.Variable(xdefault, trainable=True, name="x")
        # Asimov recentring of an MC-stat de-biased objective
        # (set_mcstat_asimov): the gradient g0 at x0, zero unless set. Rebuilt
        # here with the parameter layout, so a re-init drops it.
        self.mcstat_asimov_g0 = tf.Variable(
            tf.zeros_like(xdefault), trainable=False, name="mcstat_asimov_g0"
        )
        self.mcstat_asimov_x0 = tf.Variable(
            tf.zeros_like(xdefault), trainable=False, name="mcstat_asimov_x0"
        )

        # ParamModel Gaussian priors (declared by the model via prior_sigmas /
        # prior_means; see ParamModel). They are folded into the same
        # constraint structure as the nuisances: a declared prior sets the
        # constraint weight (1/sigma^2) and center (prior mean) of the
        # ParamModel block in the full-length cw / x0 vectors built below.
        pm = self.param_model
        np_dtype = self.indata.dtype.as_numpy_dtype
        param_cw_np = np.zeros(pm.nparams, dtype=np_dtype)
        # Constraint centers default to each parameter's own default; declared
        # priors override the masked entries with their prior mean below.
        # Keeping the default (rather than 0) for the free entries matters for
        # consumers that read x0 as a parameter's natural center (e.g.
        # nonprofiled impacts), even though cw = 0 makes it irrelevant to the
        # likelihood itself.
        param_x0_np = pm.xparamdefault.numpy().astype(np_dtype)
        # The priors live ONLY on the model (prior_sigmas / prior_means); here
        # they are folded into the constraint weights / centers below, no copy
        # of the prior arrays is kept on the Fitter.
        sigmas = getattr(pm, "prior_sigmas", None)
        if sigmas is not None:
            sigmas = np.asarray(sigmas, dtype=np_dtype)
            if sigmas.shape != (pm.nparams,):
                raise ValueError(
                    f"param_model.prior_sigmas must have shape ({pm.nparams},); "
                    f"got {sigmas.shape}"
                )
            means = getattr(pm, "prior_means", None)
            if means is None:
                means = pm.xparamdefault.numpy().astype(np_dtype)
            else:
                means = np.asarray(means, dtype=np_dtype)
                if means.shape != (pm.nparams,):
                    raise ValueError(
                        f"param_model.prior_means must have shape ({pm.nparams},); "
                        f"got {means.shape}"
                    )
            mask = np.isfinite(sigmas) & (sigmas > 0)
            n_priored = int(mask.sum())
            if n_priored > 0:
                if mask[: pm.npoi].any() and not pm.allowNegativeParam:
                    raise ValueError(
                        "Gaussian priors on POIs require allowNegativeParam="
                        "True: with allowNegativeParam=False the stored "
                        "parameter is sqrt(poi), so the Gaussian penalty "
                        "would apply to sqrt(poi) rather than to the POI "
                        "itself."
                    )
                param_cw_np = np.where(mask, 1.0 / sigmas**2, 0.0)
                # priored entries -> prior mean; free entries keep their default
                param_x0_np = np.where(mask, means, param_x0_np)
                logger.info(
                    f"[paramPriors] applying Gaussian priors to "
                    f"{n_priored}/{pm.nparams} ParamModel params:"
                )
                for i, p in enumerate(pm.params):
                    if mask[i]:
                        name = p.decode() if isinstance(p, bytes) else str(p)
                        logger.info(f"    {name}: μ={means[i]:.4g} σ={sigmas[i]:.4g}")

        # Unified constraint structure over the full parameter vector,
        # index-aligned with x = [params | systs]:
        #   cw     constraint weights (1/sigma^2; 0 = unconstrained)
        #   x0     constraint centers (prior means / theta0), fluctuated in
        #          toys wherever cw > 0
        #   var_x0 prefit variance of the centers (1/cw; 0 where free)
        self.cw = tf.concat(
            [
                tf.constant(param_cw_np, dtype=self.indata.dtype),
                self.indata.constraintweights,
            ],
            axis=0,
        )
        self.x0default = tf.concat(
            [
                tf.constant(param_x0_np, dtype=self.indata.dtype),
                self.theta0default,
            ],
            axis=0,
        )
        self.x0 = tf.Variable(self.x0default, trainable=False, name="x0")
        self.var_x0 = tf.where(
            self.cw == 0.0,
            tf.zeros_like(self.cw),
            tf.math.reciprocal(self.cw),
        )

        # Indices (within the leading param block) of the priored params,
        # derived from cw so the priors stay defined only on the model. Used by
        # the asymmetric global impacts to scan the prior centers as sources.
        self.param_prior_idxs = tf.constant(
            np.where(param_cw_np > 0)[0].astype(np.int64), dtype=tf.int64
        )
        self.param_prior_active = bool(int(self.param_prior_idxs.shape[0]) > 0)

        # Per-parameter prefit variance vector. Always allocated; the
        # prefit covariance is intrinsically diagonal so this is the
        # only form needed for prefit uncertainties.
        self.var_prefit = tf.Variable(
            self.prefit_variance(
                unconstrained_err=self.prefit_unconstrained_nuisance_uncertainty
            ),
            trainable=False,
            name="var_prefit",
        )

        # Full parameter covariance matrix. Allocated only when the
        # postfit Hessian will actually be computed; otherwise None to
        # avoid the O(npar^2) allocation (94 GB for 108k parameters).
        if self.compute_cov:
            self.cov = tf.Variable(
                tf.linalg.diag(self.var_prefit),
                trainable=False,
                name="cov",
            )
            # curvature covariance of an MC-stat de-biased fit (self.cov holds
            # the reported one); the histogram global impacts decompose it
            self.mcstat_cov_curv = tf.Variable(
                tf.linalg.diag(self.var_prefit),
                trainable=False,
                name="mcstat_cov_curv",
            )
        else:
            self.cov = None
            self.mcstat_cov_curv = None

        # regularization
        self.regularizers = []
        # one common regularization strength parameter
        self.tau = tf.Variable(1.0, trainable=True, name="tau", dtype=tf.float64)

        # External likelihood terms (additive g^T x + 0.5 x^T H x
        # contributions to the NLL). See rabbit.external_likelihood for
        # the construction helper and the matching scalar evaluator.
        # The continuous-M moment (TensorWriter.add_mc_stat_moment) is stored as
        # an external term, but it is a fit option rather than a description of
        # the inputs: its -1/2 theta^T M theta penalty is applied only with
        # --mcStatDebias continuousM, so a file carrying M fits the plain NLL
        # otherwise.
        # It is kept separate from self.external_terms (self.mcstat_terms) so
        # that the warm start in minimize() can switch it off together with the
        # rest of the de-biasing.
        ext_terms = self.indata.external_terms
        mcstat_terms = [t for t in ext_terms if _is_mcstat_term(t)]
        ext_terms = [t for t in ext_terms if not _is_mcstat_term(t)]
        if mcstat_terms and getattr(self, "mcStatDebias", "none") != "continuousM":
            logger.info(
                f"Ignoring MC-stat moment term(s) {[t['name'] for t in mcstat_terms]}: "
                "they are applied only with --mcStatDebias continuousM."
            )
            mcstat_terms = []
        self.external_terms = external_likelihood.build_tf_external_terms(
            ext_terms,
            self.parms,
            self.indata.dtype,
        )
        self.mcstat_terms = external_likelihood.build_tf_external_terms(
            mcstat_terms,
            self.parms,
            self.indata.dtype,
        )
        # Frozen noise-floor matrix M over the full parameter vector, from the
        # term(s) written by TensorWriter.add_mc_stat_moment (stored as hess = -M,
        # so M = -scatter(hess_dense)), for the continuous-M sandwich
        # Sigma = A^-1 + A^-1 M A^-1 (H = A + M). Rebuilt here because a re-init
        # (e.g. the saturated projection test's composite model) changes the
        # parameter layout.
        self.mcstat_M = self._build_mcstat_M()

        # for freezing parameters
        self.frozen_params = []
        self.frozen_params_mask = tf.Variable(
            tf.zeros_like(self.x, dtype=tf.bool), trainable=False, dtype=tf.bool
        )

        self.frozen_indices = np.array([])
        self.freeze_params(freeze_parameters)

        # determine if problem is linear (ie likelihood is purely quadratic).
        # --forceLinear bypasses the structural checks and forces the
        # quadratic-solver path; the user is responsible for ensuring the
        # resulting Gaussian step is meaningful (e.g. via an outer iteration
        # that re-anchors the linearization point).
        self.is_linear = self.force_linear or (
            (self.chisqFit or self.covarianceFit)
            and self.param_model.is_linear
            and self.indata.symmetric_tensor
            and self.indata.systematic_type == "normal"
            and self.bbstat.is_linear
        )
        if self.force_linear:
            logger.info(
                "--forceLinear set: solving by a single Gaussian (Cholesky / "
                "Hessian-CG) step regardless of the actual likelihood shape."
            )

        # force retrace of @tf.function methods since self.x shape may have changed
        self._retrace_tf_functions()

    def _retrace_tf_functions(self):
        """Re-wrap every @tf.function method and rebuild the dynamic loss/grad/HVP
        wrappers, so the next call traces against the current Python state."""
        for name in dir(type(self)):
            val = getattr(type(self), name, None)
            if hasattr(val, "python_function"):
                setattr(
                    self,
                    name,
                    tf.function(val.python_function.__get__(self, type(self))),
                )

        # (re)build instance-level tf.function wrappers for loss/grad/HVP, which
        # are constructed dynamically so that jit_compile and the HVP autodiff
        # mode can be controlled via fit options.
        self._make_tf_functions()

    @contextlib.contextmanager
    def _ordinary_likelihood(self):
        """Temporarily drop the MC-stat de-biasing from the objective.

        Inside the block the loss is the ordinary likelihood that
        --mcStatDebias none minimizes: the data term on the full templates
        only, and no continuous-M penalty. Every other term (constraints,
        bin-by-bin stat, regularization, external terms, blinding) is shared.
        The traced loss functions read the switch at trace time, so each
        setting has its own set of them (_set_mcstat_debias_suspended).
        """
        self._set_mcstat_debias_suspended(True)
        try:
            yield
        finally:
            self._set_mcstat_debias_suspended(False)

    def _tf_function_attrs(self):
        """The instance attributes holding traced functions: the per-instance
        re-wraps of the @tf.function methods and the dynamic wrappers."""
        names = {
            name
            for name in dir(type(self))
            if hasattr(getattr(type(self), name, None), "python_function")
        }
        names |= set(self._DYNAMIC_TF_FUNCS) | {"loss_val_grad_hessp"}
        names.discard("_mcstat_tf_standby")
        return {name: self.__dict__[name] for name in names if name in self.__dict__}

    def _set_mcstat_debias_suspended(self, suspended):
        """Switch the de-biasing off (the warm start's ordinary likelihood) or
        back on. The set of traced functions of the other setting is kept, so
        each is traced once per fitter rather than twice per minimize(): XLA
        keeps compiled code mapped for the life of the process, and a retrace
        per fit exhausts the memory mappings over a run of toys. Any rebuild
        (_make_tf_functions) drops the kept set."""
        if suspended == self._mcstat_debias_suspended:
            return
        current = (self._mcstat_debias_suspended, self._tf_function_attrs())
        standby = self.__dict__.get("_mcstat_tf_standby")
        self._mcstat_debias_suspended = suspended
        if standby is not None and standby[0] == suspended:
            self.__dict__.update(standby[1])
        else:
            self._retrace_tf_functions()
        self._mcstat_tf_standby = current

    def __deepcopy__(self, memo):
        import copy

        # Instance-level tf.function overrides (set by init_fit_parms to force retracing)
        # contain FuncGraph objects that cannot be deepcopied. Strip them before copying
        # so the copy falls back to the class-level @tf.function methods and retraces.
        jit_overrides = {
            name
            for name in self.__dict__
            if hasattr(getattr(type(self), name, None), "python_function")
        }
        # Also strip the dynamically-built loss/grad/HVP tf.function wrappers,
        # which hold un-copyable FuncGraph state and will be rebuilt below.
        skip = jit_overrides | set(self._DYNAMIC_TF_FUNCS)
        state = {k: v for k, v in self.__dict__.items() if k not in skip}
        cls = type(self)
        obj = cls.__new__(cls)
        memo[id(self)] = obj
        for k, v in state.items():
            setattr(obj, k, copy.deepcopy(v, memo))
        obj._make_tf_functions()
        return obj

    def load_fitresult(self, fitresult_file, fitresult_key, profile=True):
        # load results from external fit and set postfit value and covariance elements for common parameters
        # external_cov_loaded records whether the loaded fitresult carried a
        # covariance (it does not when it was produced with --noHessian), so
        # callers can decide to recompute the Hessian at the loaded point.
        self.external_cov_loaded = False
        self.external_cov_mask = None
        cov_ext = None
        with h5py.File(fitresult_file, "r") as fext:
            if "x" in fext.keys():
                # fitresult from rabbit
                x_ext = fext["x"][...]
                parms_ext = fext["parms"][...].astype(str)
                if "cov" in fext.keys():
                    cov_ext = fext["cov"][...]
            else:
                # fitresult from rabbit
                h5results_ext = io_tools.get_fitresult(fext, fitresult_key)
                h_parms_ext = h5results_ext["parms"].get()

                x_ext = h_parms_ext.values()
                parms_ext = np.array(h_parms_ext.axes["parms"])
                if "cov" in h5results_ext.keys():
                    cov_ext = h5results_ext["cov"].get().values()

        xvals = self.x.numpy()
        parms = self.parms.astype(str)

        # Find common elements with their matching indices
        common_elements, idxs, idxs_ext = np.intersect1d(
            parms, parms_ext, assume_unique=True, return_indices=True
        )
        xvals[idxs] = x_ext[idxs_ext]

        self.x.assign(xvals)

        if cov_ext is not None:
            if self.cov is None:
                raise RuntimeError(
                    "load_fitresult: external covariance was provided but "
                    "the fitter was constructed with --noHessian (no full "
                    "covariance is allocated). Construct the fitter without "
                    "--noHessian to load an external covariance."
                )
            covval = self.cov.numpy()
            covval[np.ix_(idxs, idxs)] = cov_ext[np.ix_(idxs_ext, idxs_ext)]
            self.cov.assign(tf.constant(covval))
            self.external_cov_loaded = True
            # Which parameters the loaded covariance actually covers. Partial
            # overlap is supported (hence the intersect above), and the rest of
            # self.cov keeps the prefit diagonal it was initialized with, so a
            # caller reading diag(cov) as postfit uncertainties needs to know
            # which entries are real.
            mask = np.zeros(len(parms), dtype=bool)
            mask[idxs] = True
            self.external_cov_mask = mask

        if profile:
            self._profile_beta()

    def update_frozen_params(self):
        logger.debug(f"Updated list of frozen params: {self.frozen_params}")
        new_mask_np = np.isin(self.parms, self.frozen_params)

        self.frozen_params_mask.assign(new_mask_np)
        self.frozen_indices = np.where(new_mask_np)[0]
        self.floating_indices = np.where(~self.frozen_params_mask)[0]

    def freeze_params(self, frozen_parmeter_expressions):
        logger.debug(f"Freeze params with {frozen_parmeter_expressions}")
        self.frozen_params.extend(
            match_regexp_params(frozen_parmeter_expressions, self.parms)
        )
        self.update_frozen_params()

    def defreeze_params(self, unfrozen_parmeter_expressions):
        logger.debug(f"Freeze params with {unfrozen_parmeter_expressions}")
        unfrozen_parmeter = match_regexp_params(
            unfrozen_parmeter_expressions, self.parms
        )
        self.frozen_params = [
            x for x in self.frozen_params if x not in unfrozen_parmeter
        ]
        self.update_frozen_params()

    def _check_blinded_start_is_evaluable(self):
        """Refuse a blinded start the model cannot be evaluated at.

        Because the frame is not compensated, arming moves the PHYSICAL start
        to ``default + offset``. For a POI that merely scales yields that is a
        poor starting point and nothing worse. For a POI that feeds a
        calculation with a restricted domain it can be fatal: the yields go
        negative, the Poisson ``log`` returns NaN, and the minimiser has no
        finite point to descend from -- it does not converge slowly, it cannot
        start. That is the 2026-09-09 alpha_s failure.

        The wider the smearing the likelier this is, which is the awkward part:
        ``blind_additive_scale`` is the knob raised to make blinding effective,
        so the configurations that hide best are the ones most likely to land
        outside the domain. Rather than discover it as a NaN partway through,
        check once at arming and say so.

        The consequence is a real limitation, not a warning: a POI whose
        required smearing takes it out of its own evaluable range cannot be
        blinded this way at all. The message therefore names the knobs, and --
        like the weak-smearing warning -- no numbers, since the offset and the
        start point each give the other away.
        """
        try:
            val = float(self._compute_loss().numpy())
        except Exception as exc:
            raise RuntimeError(
                "The blinded starting point is outside the range this model can "
                "be evaluated at: computing the likelihood there raised "
                f"{type(exc).__name__}. Blinding is not frame-compensated -- on "
                "purpose, so that the offset is never written into a coordinate "
                "it could be recovered from -- so an armed fit opens at the "
                "model default plus the offset. Lower "
                "param_model.blind_additive_scale, exempt this parameter with "
                "--unblind, or declare blind_exempt on the model if it is not a "
                "result."
            ) from exc
        if not np.isfinite(val):
            raise RuntimeError(
                "The blinded starting point gives a non-finite likelihood, so "
                "the minimiser has nothing to start from. Blinding is not "
                "frame-compensated -- on purpose, so that the offset is never "
                "written into a coordinate it could be recovered from -- so an "
                "armed fit opens at the model default plus the offset, and for "
                "this model that point is outside its evaluable range (most "
                "often negative expected yields). Lower "
                "param_model.blind_additive_scale, exempt this parameter with "
                "--unblind, or declare blind_exempt on the model if it is not a "
                "result."
            )

    def set_blinding_offsets(self, blind=True):
        """Arm or disarm the blinding offsets. See :meth:`Blinding._arm` for
        what arming does to the frame, and why it never touches ``self.x``.

        Callers go through the Fitter rather than through ``self.blinding``
        because arming moves the PHYSICAL point the fit opens at, and whether
        the model can be evaluated there is a question about the likelihood,
        which the Fitter owns and the offsets know nothing about. So arming is
        followed by a check that fails loudly rather than leaving the minimiser
        with a NaN to start from.
        """
        if not self.blinding.enabled:
            return
        self.blinding._arm(blind)
        if not blind:
            return

        self._check_blinded_start_is_evaluable()

    # The parameter getters read self.x, or the B columns of x = [npar, B]
    # (the batched likelihood, _compute_nll_batched); the per-parameter masks
    # and offsets then broadcast as columns.
    def get_theta(self, x=None):
        x = self.x if x is None else x
        start = self.param_model.nparams
        theta = x[start : start + self.indata.nsyst]
        theta = tf.where(
            tfh.per_parameter(
                self.frozen_params_mask[start : start + self.indata.nsyst], theta
            ),
            tf.stop_gradient(theta),
            theta,
        )
        return self.blinding.apply_theta(theta)

    def get_model_nui(self, x=None):
        x = self.x if x is None else x
        npoi = self.param_model.npoi
        npou = self.param_model.npou
        nui = x[npoi : npoi + npou]
        # Apply frozen_params_mask the same way get_poi() and get_theta() do.
        # Without this, --freezeParameters silently fails to freeze POUs:
        # the param is registered as frozen but no stop_gradient is applied,
        # so the optimizer updates it anyway.
        nui = tf.where(
            tfh.per_parameter(self.frozen_params_mask[npoi : npoi + npou], nui),
            tf.stop_gradient(nui),
            nui,
        )
        return nui

    def get_poi(self, x=None):
        x = self.x if x is None else x
        xpoi = x[: self.param_model.npoi]
        # Offset BEFORE the positivity transform, not after. (x + off)**2 is
        # non-negative for ANY offset, so an additive offset composes with the
        # squared storage rather than fighting it -- x**2 + off is the form
        # that can go negative, which is why additive blinding used to require
        # allowNegativeParam. It also leaves the reported uncertainty exactly
        # unblinded: at the minimum d(poi)/dx is 2*sqrt(poi) in both frames, so
        # sigma_x is identical armed and disarmed. Identity when blinding is
        # off, so there is no branch here.
        xpoi = self.blinding.apply_poi(xpoi)
        if self.param_model.allowNegativeParam:
            poi = xpoi
        else:
            poi = tf.square(xpoi)
        return tf.where(
            tfh.per_parameter(self.frozen_params_mask[: self.param_model.npoi], poi),
            tf.stop_gradient(poi),
            poi,
        )

    def get_x(self, x=None):
        return tf.concat(
            [self.get_poi(x), self.get_model_nui(x), self.get_theta(x)], axis=0
        )

    @property
    def nfreeparms(self):
        """Number of fit parameters that cost a degree of freedom.

        A parameter with a Gaussian constraint adds one parameter AND one
        pseudo-measurement, so it costs NET ZERO degrees of freedom. Only
        genuinely free parameters reduce the ndf of a goodness-of-fit test.

        ``self.cw`` is the single place that records which parameters are
        constrained, over the whole vector ``[ParamModel params | systs]``:
        declared ParamModel priors on one side (``prior_sigmas`` entries that
        are finite and > 0, folded in by :meth:`init_fit_parms`) and
        ``indata.constraintweights`` on the other. ``cw == 0`` is exactly the
        set the likelihood treats as unconstrained, so counting it here cannot
        drift from what ``_compute_lc`` actually penalises.

        Deliberately NOT ``param_model.nparams + indata.nsystnoconstraint``:
        that charges every model parameter but only the unconstrained card
        systematics, i.e. it distinguishes parameters by WHERE THEY ARE
        DECLARED, which is not a statistical property. A model parameter with a
        sigma = 1 prior and a card nuisance with a sigma = 1 prior are the same
        object and must be charged the same way.

        FROZEN parameters are excluded as well. ``cw`` records constraints;
        frozen-ness lives in ``frozen_params_mask``, so an unconstrained frozen
        parameter has ``cw == 0`` yet is fixed and costs nothing. Counting it
        would make ``ndfsat`` too small and ``chi2.sf(chi2_val, ndfsat)``
        correspondingly too pessimistic, by exactly the number of such
        parameters. (The old formula ignored frozen parameters too, so this is
        not a regression -- but this is the property the name claims, so it
        should hold.)
        """
        return int(np.count_nonzero(self._free_parms_mask()))

    def _free_parms_mask(self):
        """Boolean mask over ``[ParamModel params | systs]``: unconstrained AND
        not frozen, i.e. the parameters that actually cost a degree of freedom.
        """
        return (self.cw.numpy() == 0.0) & ~self.frozen_params_mask.numpy()

    @property
    def nfreeparms_breakdown(self):
        """``(free ParamModel params, free systs)``, for logging.

        The two entries of :attr:`nfreeparms`, split at the ParamModel /
        systematics boundary. The second is ``indata.nsystnoconstraint`` minus
        any frozen unconstrained nuisances; the first is what the old ndf
        formula got wrong.
        """
        free = self._free_parms_mask()
        nparams = self.param_model.nparams
        return int(np.count_nonzero(free[:nparams])), int(
            np.count_nonzero(free[nparams:])
        )

    def prefit_variance(self, unconstrained_err=0.0):
        """Per-parameter prefit variance vector of length npar.

        Unconstrained entries (cw = 0: ParamModel params without a prior and
        unconstrained nuisances) are assigned a placeholder variance of
        unconstrained_err**2 (zero by default); constrained entries take
        their variance from the constraint term (1 / cw).
        """
        return tf.where(
            self.cw == 0.0,
            unconstrained_err**2 * tf.ones_like(self.cw),
            tf.math.reciprocal(self.cw),
        )

    def prefit_covariance(self, unconstrained_err=0.0):
        """Full prefit covariance as a tf.linalg.LinearOperatorDiag.

        The prefit covariance is intrinsically diagonal, so we return a
        LinearOperator that exposes a matrix-like interface (matvec, etc.)
        without ever allocating the dense [npar, npar] form. Callers that
        actually need a dense tensor can call .to_dense() explicitly.
        """
        return tf.linalg.LinearOperatorDiag(
            self.prefit_variance(unconstrained_err=unconstrained_err),
            is_self_adjoint=True,
            is_positive_definite=True,
        )

    @tf.function
    def val_jac(self, fun, *args, **kwargs):
        with tf.GradientTape() as t:
            val = fun(*args, **kwargs)
        jac = t.jacobian(val, self.x)

        return val, jac

    def _observations(self, nobs, variances=None):
        """(nobs, varnobs, lognobs) of the observed counts nobs, as the data
        term reads them; any shape, e.g. [nbins] or [nbins, B] for B toys.

        lognobs is log(nobs) with log 1 = 0 in empty bins: the offset that
        gives the saturated Poisson likelihood, for numerical precision in the
        minimizer (a plain log(nobs) makes -nobs (log nexp - log nobs) NaN
        there). varnobs is the variance of the chi2 (_chisq_variance), and None
        for the other fits, which do not read `variances` (--covarianceFit has
        its data covariance from the input). A chi2 fit needs nobs > 0.
        """
        nobs = tf.convert_to_tensor(nobs, dtype=self.indata.dtype)
        varnobs = None
        if self.chisqFit:
            # covariance from data stat
            message = "Bins in 'nobs <= 0' encountered, chi^2 fit can not be performed."
            if tf.executing_eagerly():
                if tf.math.reduce_any(nobs <= 0).numpy():
                    raise RuntimeError(message)
            else:
                tf.debugging.assert_positive(nobs, message=message)
            varnobs = self._chisq_variance(nobs, variances)
        nobssafe = tf.where(nobs == 0.0, tf.constant(1.0, dtype=nobs.dtype), nobs)
        return nobs, varnobs, tf.math.log(nobssafe)

    def _chisq_variance(self, nobs, variances=None):
        """The per-bin variance of the chi2 for the observations nobs (any
        shape, as in _observations), following --chisqVariance:

        - "data": `variances`, the data variances of the dataset (stored with
          the data or a pseudodata set, or those of a toy), else nobs
          themselves (Poisson); they must be > 0;
        - "mc": the MC statistical variance of the nominal templates
          (_mc_chisq_variance), whatever the dataset and `variances`.
        """
        if self.chisqVariance == "mc":
            shape = [-1] + [1] * (len(nobs.shape) - 1)
            return tf.broadcast_to(tf.reshape(self.mc_variance, shape), tf.shape(nobs))
        if variances is None:
            return nobs
        variances = tf.convert_to_tensor(variances, dtype=self.indata.dtype)
        message = (
            "Bins with data variance <= 0 encountered, chi^2 fit can not be "
            "performed."
        )
        if tf.executing_eagerly():
            if tf.math.reduce_any(variances <= 0).numpy():
                raise RuntimeError(message)
        else:
            tf.debugging.assert_positive(variances, message=message)
        return variances

    def _mc_chisq_variance(self):
        """The chi2 variance of --chisqVariance mc [nbins]: the MC statistical
        variance of the nominal templates in each data bin, the sum over the
        processes of their sumw2.

        A constant, like a data variance, rather than the variance of the
        templates at the fit point (scaled with the yields): a chi2 whose
        variance moves with the parameters is not the likelihood it stands for.
        Minimized as it is, it favours the parameters that inflate the variance;
        made a Gaussian likelihood by adding sum_b log V_b, it is no longer a
        chi2 and draws spurious information on the parameters from how the
        template statistics scale with them. The fitter also takes the chi2
        variance as a constant throughout (the bin-by-bin stat profiling, the
        data-stat global impacts, the residual covariances). The variance of the
        templates at the fit point is what --binByBinStatDynamic profiles.
        """
        if self.bbstat.enabled:
            raise ValueError(
                "--chisqVariance mc makes the MC statistical variance the chi2 "
                "variance; the bin-by-bin statistical uncertainties profile the "
                "same MC statistics and would count them twice. Add "
                "--noBinByBinStat."
            )
        if self.mcStatDebias != "none":
            raise ValueError(
                "--chisqVariance mc is not supported with --mcStatDebias: the "
                "de-biasing accounts for the MC statistical fluctuations of the "
                "templates, which the chi2 variance would then count again."
            )
        sumw2 = tf.convert_to_tensor(self.indata.sumw2)
        if len(sumw2.shape) > 1:
            sumw2 = tf.reduce_sum(sumw2, axis=-1)
        variance = sumw2[: self.indata.nbins]
        nonpositive = int(tf.math.count_nonzero(variance <= 0).numpy())
        if nonpositive:
            raise RuntimeError(
                f"{nonpositive} bins with MC statistical variance <= 0 (sum over "
                "the processes of sumw2) encountered, chi^2 fit with "
                "--chisqVariance mc can not be performed."
            )
        return variance

    def set_nobs(self, values, variances=None):
        """Set the observed data; `variances` are its data variances, for the
        chi2 (see _chisq_variance)."""
        values, varnobs, lognobs = self._observations(values, variances)
        if self.chisqFit:
            self.varnobs.assign(varnobs)

        self._check_mcstat_empty_halves(values)
        self.nobs.assign(values)
        self.lognobs.assign(lognobs)
        if self.chisqFit and getattr(self, "mcstat_bb_vref", None) is not None:
            # its per-bin variance is the data variance
            self._update_mcstat_bb_vref()

    def x0defaultassign(self):
        # reset all constraint centers
        self.x0.assign(self.x0default)

    def xdefaultassign(self):
        # start every parameter at its constraint center (prior mean / theta0
        # default, and the model default for unpriored params)
        #
        # Assigned raw, in whichever frame is armed: reframing x0default into
        # the blinded frame would compute the offset into x, and x0default is
        # public, so the offset would be one subtraction away from anyone who
        # reads x. See set_blinding_offsets. The physical start therefore
        # differs between an armed and a disarmed run, which is accepted --
        # the minimiser lands on the same minimum either way.
        self.x.assign(self.x0default)

    def defaultassign(self):
        var_pre = self.prefit_variance(
            unconstrained_err=self.prefit_unconstrained_nuisance_uncertainty
        )
        self.var_prefit.assign(var_pre)
        if self.cov is not None:
            self.cov.assign(tf.linalg.diag(var_pre))
        self.x0defaultassign()
        if self.bbstat.enabled:
            self.bbstat.beta0_default_assign()
            self.bbstat.beta_default_assign()
        self.xdefaultassign()
        if self.do_blinding:
            self.set_blinding_offsets(False)

        self.arm_regularizers()

    def _regularizers_need_observables(self):
        """Whether any attached regularizer reads the predicted yields.

        Penalties on the parameters alone (see Regularizer.needs_observables)
        do not, and then the full yields never have to be built for them.
        """
        return any(getattr(reg, "needs_observables", True) for reg in self.regularizers)

    def arm_regularizers(self):
        """Tell every regularizer about the current parameter layout.

        Must be called after anything that changes ``self.parms``, notably
        ``init_fit_parms``.
        """
        xinit = self.get_x()
        nexp0 = (
            self.expected_yield(full=True)
            if self._regularizers_need_observables()
            else None
        )
        for reg in self.regularizers:
            reg.set_expectations(
                xinit,
                nexp0 if getattr(reg, "needs_observables", True) else None,
                parms=self.parms,
            )
        self._regularizers_armed = True
        # The hard constraints resolve their positions by name too, so every
        # site that re-arms after a layout change (defaultassign, the saturated
        # test's re-init) covers them as well.
        for con in self.constraints:
            con.set_layout(self.parms)

    # ------------------------------------------------------------------
    # Functions of the PHYSICAL parameters, evaluated at a point given in the
    # fitter's own coordinate (the frame of self.x): fn(get_x()), so blinding
    # offsets, the POI positivity transform and the frozen-parameter
    # stop_gradient apply exactly as in the likelihood. Used by the hard
    # constraints (rabbit.constraints); the optimizer plumbing and the
    # post-fit algebra live there. self.x is restored afterwards: scipy calls
    # these at trial points it may reject.
    # ------------------------------------------------------------------

    def _with_x(self, xval, fn):
        """Evaluate ``fn()`` with self.x temporarily set to ``xval``."""
        xsave = self.x.numpy()
        try:
            self.x.assign(np.asarray(xval, dtype=xsave.dtype))
            return fn()
        finally:
            self.x.assign(xsave)

    def _zero_frozen_columns(self, a, axis=-1):
        if len(self.frozen_indices):
            idx = [slice(None)] * a.ndim
            idx[axis] = np.asarray(self.frozen_indices, dtype=int)
            a[tuple(idx)] = 0.0
        return a

    def eval_physical(self, fn, xval, jacobian=False):
        """``fn(get_x())`` at ``xval`` (``fn`` returns a 1-D tensor).

        With ``jacobian=True`` also its Jacobian [m, n] with respect to the
        fitter coordinate. Frozen columns are zero: besides the stop_gradient
        in get_x they are zeroed explicitly, because a constrained minimizer
        has a second handle on a variable -- its constraint Jacobian column --
        and with no objective gradient opposing it would move a "frozen"
        parameter freely (observed 2026-08-13: lambda_inf 1.0 -> 17.6).
        """

        def values():
            return tf.reshape(tf.cast(fn(self.get_x()), self.x.dtype), [-1])

        if not jacobian:
            return self._with_x(xval, lambda: values().numpy())

        def val_jac():
            with tf.GradientTape(persistent=True) as tape:
                vals = values()
            jac = tape.jacobian(
                vals,
                self.x,
                unconnected_gradients=tf.UnconnectedGradients.ZERO,
                experimental_use_pfor=False,
            )
            return vals.numpy(), jac.numpy()

        vals, jac = self._with_x(xval, val_jac)
        return vals, self._zero_frozen_columns(jac)

    def hvp_physical(self, fn, xval, v):
        """The Hessian of ``v . fn(get_x())`` at ``xval``, as a LinearOperator.

        Hessian-vector products by forward-over-back autodiff, frozen rows and
        columns zero. For the constraints this is the term trust-constr needs
        in the Hessian of the Lagrangian; without it scipy falls back to a
        BFGS estimate that starts as a scaled identity over ALL parameters
        (~37*I at the first update in the 2026-10-13 tests) while the true term
        lives on the few parameters the constraints read, so every step is
        over-damped and the barrier stalls.
        """
        xval = np.array(xval, dtype=np.float64, copy=True)
        w = tf.constant(np.asarray(v, dtype=np.float64), dtype=self.x.dtype)
        frozen = np.asarray(self.frozen_indices, dtype=int)
        n = int(self.x.shape[0])

        def matvec(direction):
            d = np.array(np.ravel(direction), dtype=np.float64, copy=True)
            if len(frozen):
                d[frozen] = 0.0

            def hv():
                dt = tf.constant(d, dtype=self.x.dtype)
                with tf.GradientTape() as outer:
                    with tf.GradientTape() as inner:
                        vals = tf.reshape(tf.cast(fn(self.get_x()), self.x.dtype), [-1])
                        weighted = tf.tensordot(vals, w, axes=1)
                    grad = inner.gradient(
                        weighted,
                        self.x,
                        unconnected_gradients=tf.UnconnectedGradients.ZERO,
                    )
                return outer.gradient(
                    grad,
                    self.x,
                    output_gradients=dt,
                    unconnected_gradients=tf.UnconnectedGradients.ZERO,
                ).numpy()

            out = self._with_x(xval, hv)
            if len(frozen):
                out[frozen] = 0.0
            return out

        return scipy.sparse.linalg.LinearOperator(
            (n, n), matvec=matvec, rmatvec=matvec, dtype=np.float64
        )

    def hess_physical(self, fn, xval, v, cols=None):
        """Dense Hessian of ``v . fn(get_x())`` at ``xval``, on ``cols``.

        [n, n], zero outside ``cols`` x ``cols`` (default: every floating
        parameter) and on frozen rows/columns. One backward pass per column.
        """
        n = int(self.x.shape[0])
        if cols is None:
            cols = np.asarray(self.floating_indices, dtype=int)
        cols = np.setdiff1d(np.asarray(cols, dtype=int), self.frozen_indices)
        out = np.zeros((n, n))
        if not len(cols):
            return out
        w = tf.constant(np.asarray(v, dtype=np.float64), dtype=self.x.dtype)
        idx = tf.constant(cols.astype(np.int32))

        def rows():
            with tf.GradientTape(persistent=True) as outer:
                with tf.GradientTape() as inner:
                    vals = tf.reshape(tf.cast(fn(self.get_x()), self.x.dtype), [-1])
                    weighted = tf.tensordot(vals, w, axes=1)
                grad = inner.gradient(
                    weighted, self.x, unconnected_gradients=tf.UnconnectedGradients.ZERO
                )
                gsub = tf.gather(grad, idx)
            return outer.jacobian(
                gsub,
                self.x,
                unconnected_gradients=tf.UnconnectedGradients.ZERO,
                experimental_use_pfor=False,
            ).numpy()

        out[cols] = self._zero_frozen_columns(self._with_x(xval, rows))
        return 0.5 * (out + out.T)

    def _constraints_fn(self, x):
        """All --constraints rows of the physical vector ``x``, concatenated."""
        return tf.concat(
            [
                tf.reshape(tf.cast(c.values(x), self.x.dtype), [-1])
                for c in self.constraints
            ],
            axis=0,
        )

    def _constraint_bounds(self):
        """(lower, upper, labels) of all --constraints rows."""
        lower = np.concatenate(
            [
                np.reshape(np.asarray(c.lower, np.float64), [-1])
                for c in self.constraints
            ]
        )
        upper = np.concatenate(
            [
                np.reshape(np.asarray(c.upper, np.float64), [-1])
                for c in self.constraints
            ]
        )
        labels = [str(lab) for c in self.constraints for lab in c.labels]
        return lower, upper, labels

    def _constraint_columns(self):
        """Union of the declared parameter_indices; None if any is undeclared."""
        cols = []
        for c in self.constraints:
            idx = c.parameter_indices()
            if idx is None:
                return None
            cols.extend(int(i) for i in idx)
        return np.unique(np.asarray(cols, dtype=int))

    def build_scipy_constraints(
        self, xstart, to_physical=None, transform=None, hold_feasible=True
    ):
        """The --constraints rows as scipy NonlinearConstraints, see
        :func:`rabbit.constraints.scipy_adapter.build_nonlinear_constraints`.

        ``transform``, if given, is a callable returning the CURRENT
        :class:`rabbit.preconditioner.Preconditioner` (looked up per call: it
        is rebuilt on restarts); the Jacobian and the constraint HVPs are then
        mapped to its internal coordinates by the chain rule, and
        ``to_physical`` defaults to its map."""
        if not len(self.constraints):
            self._constraint_info = None
            return []
        lower, upper, labels = self._constraint_bounds()
        pc_kw = {}
        if transform is not None:
            if to_physical is None:

                def to_physical(y):
                    return transform().to_physical(y)

            pc_kw = dict(
                jac_to_internal=lambda j: transform().jac_to_internal(j),
                hessp_to_internal=lambda p, mv: transform().hessp_to_internal(p, mv),
            )
        cons, info = scipy_adapter.build_nonlinear_constraints(
            lambda x: self.eval_physical(self._constraints_fn, x, jacobian=True),
            lambda x, v: self.hvp_physical(self._constraints_fn, x, v),
            xstart,
            lower,
            upper,
            labels,
            to_physical=to_physical,
            **pc_kw,
            hold_feasible=hold_feasible,
        )
        self._constraint_info = info
        return cons

    def constrained_postfit(self, grad, hess, active_tol=1e-5):
        """Face-held post-fit at the current self.x, see
        :func:`rabbit.constraints.postfit.face_held_covariance`."""
        x = self.x.numpy()
        cval, J = self.eval_physical(self._constraints_fn, x, jacobian=True)
        lower, upper, labels = self._constraint_bounds()
        cols = self._constraint_columns()
        if cols is None:
            logger.info(
                "[constraints] a Constraints object declares no parameter_indices; "
                "the post-fit constraint Hessian is taken over every floating "
                "parameter"
            )
        base = (
            self.cov.numpy()
            if getattr(self, "cov", None) is not None
            else np.diag(self.var_prefit.numpy())
        )
        cpf = constraints_postfit.face_held_covariance(
            grad,
            hess,
            cval,
            J,
            lower,
            upper,
            labels,
            lambda v: self.hess_physical(self._constraints_fn, x, v, cols),
            floating=(
                np.asarray(self.floating_indices, dtype=int)
                if len(self.frozen_params) > 0
                else None
            ),
            cov_base=base,
            active_tol=active_tol,
        )
        self._set_postfit_faces(cpf["jacobian_active"], cpf["hess_constraints"])
        return cpf

    def _set_postfit_faces(self, jac_active, hess_constraints):
        if len(jac_active):
            self.postfit_faces = {
                "jacobian": np.asarray(jac_active, dtype=np.float64),
                "hess_constraints": np.asarray(hess_constraints, dtype=np.float64),
            }
        else:
            self.postfit_faces = None

    def bayesassign(self):
        # Sample the parameter values from their priors: width sqrt(1/cw)
        # around the constraint centers wherever cw > 0; free entries (cw = 0)
        # stay at their default constraint center.
        sampled = self.x0 + tf.sqrt(self.var_x0) * tf.random.normal(
            shape=self.x0.shape, dtype=self.x0.dtype
        )
        self.x.assign(tf.where(self.cw > 0, sampled, self.x0default))

        self.bbstat.randomize_bayes()

    def frequentistassign(self):
        # Fluctuate the constraint centers around their defaults with the
        # prefit constraint widths; entries with cw = 0 (unconstrained) are
        # not randomized.
        self.x0.assign(
            self.x0default
            + tf.sqrt(self.var_x0)
            * tf.random.normal(shape=self.x0.shape, dtype=self.x0.dtype)
        )
        self.bbstat.randomize_frequentist()

    def toyassign(
        self,
        data_values=None,
        data_variances=None,
        syst_randomize="frequentist",
        data_randomize="poisson",
        data_mode="expected",
        randomize_parameters=False,
    ):
        if syst_randomize == "bayesian":
            # randomize actual values
            self.bayesassign()
        elif syst_randomize == "frequentist":
            # randomize nuisance constraint minima
            self.frequentistassign()

        if data_mode == "expected":
            data_nom = self.expected_yield()
        elif data_mode == "observed":
            data_nom = data_values

        if data_randomize == "poisson":
            if self.covarianceFit:
                raise RuntimeError(
                    "Toys with external covariance only possible with data_randomize=normal"
                )
            else:
                self.set_nobs(
                    tf.random.poisson(lam=data_nom, shape=[], dtype=self.nobs.dtype)
                )
        elif data_randomize == "normal":
            if self.covarianceFit:
                pdata = tfp.distributions.MultivariateNormalTriL(
                    loc=data_nom,
                    scale_tril=tf.linalg.cholesky(tf.linalg.inv(self.data_cov_inv)),
                )
                self.set_nobs(pdata.sample())
            else:
                if self.chisqFit:
                    # the width the chi2 assumes (--chisqVariance)
                    data_var = self._chisq_variance(data_nom, data_variances)
                else:
                    data_var = data_nom

                self.set_nobs(
                    tf.random.normal(
                        mean=data_nom,
                        stddev=tf.sqrt(data_var),
                        shape=[],
                        dtype=self.nobs.dtype,
                    ),
                    data_variances,
                )
        elif data_randomize == "none":
            self.set_nobs(data_nom, data_variances)

        # assign start values for nuisance parameters to constraint minima
        self.xdefaultassign()
        if self.bbstat.enabled:
            self.bbstat.beta_default_assign()
        # set likelihood offset
        self.nexpnom.assign(self.expected_yield())

        if randomize_parameters:
            # the special handling of the diagonal case here speeds things up, but is also required
            # in case the prefit covariance has zero for some uncertainties (which is the default
            # for unconstrained nuisances for example) since the multivariate normal distribution
            # requires a positive-definite covariance matrix.
            # Under --noHessian self.cov is None and only the diagonal
            # prefit variance vector is available, so we always take the
            # diagonal branch in that case (sourcing the variances from
            # var_prefit directly).
            cov_is_diag = self.cov is None or tfh.is_diag(self.cov)
            if cov_is_diag:
                stddev = (
                    tf.sqrt(self.var_prefit)
                    if self.cov is None
                    else tf.sqrt(tf.linalg.diag_part(self.cov))
                )
                self.x.assign(
                    tf.random.normal(
                        shape=[],
                        mean=self.x,
                        stddev=stddev,
                        dtype=self.x.dtype,
                    )
                )
            else:
                pparms = tfp.distributions.MultivariateNormalTriL(
                    loc=self.x, scale_tril=tf.linalg.cholesky(self.cov)
                )
                self.x.assign(pparms.sample())
            self.bbstat.randomize_postfit()

    def _mcstat_M_params_linear(self):
        """True iff every parameter the mcstat moment M touches enters the
        prediction/NLL linearly, so that -1/2 theta^T M theta de-biases exactly
        (constant curvature, no minimum-shift cancellation;
        docs/mcstat/RESULTS.md §9a).

        A parameter is "linear" here when:
          * POI (index < npoi): the param model is linear (allowNegativeParam,
            i.e. rnorm = x rather than the default rnorm = x^2), and
          * systematic: the systematics enter additively (systematic_type
            'normal') with a symmetric tensor (the asymmetric interpolation is
            nonlinear), and
        the per-bin NLL is Gaussian (chisqFit) so its curvature does not depend
        on the fit point (a Poisson l'' = nobs/nexp^2 varies with nexp(theta)
        even for a linear nexp). Returns True only if ALL touched parameters meet
        the relevant condition.
        """
        if self.mcstat_M is None:
            return True
        if not self.chisqFit:
            return False  # Poisson/other: curvature is theta-dependent
        npoi = self.param_model.npoi
        mcstat_terms = self.mcstat_terms
        touched = set()
        for t in mcstat_terms:
            touched.update(int(i) for i in t["indices"].numpy())
        for i in touched:
            if i < npoi:
                if not self.param_model.is_linear:
                    return False
            else:
                if not (
                    self.indata.systematic_type == "normal"
                    and self.indata.symmetric_tensor
                ):
                    return False
        return True

    def _build_mcstat_M(self):
        """Reconstruct the frozen noise-floor matrix M (nparams x nparams) from
        external terms whose name starts with 'mcstat'.

        TensorWriter.add_mc_stat_moment stores the term Hessian as hess = -M
        (so the objective contribution 0.5 x^T hess x = -1/2 x^T M x de-biases
        the curvature). Here we scatter each term's (-hess_dense) block back into
        the full nparams x nparams space, indexed by the term's parameter indices.
        Returns None if no such term is present (continuous-M inactive).
        """
        mcstat = self.mcstat_terms
        if not mcstat:
            return None
        n = int(self.x.shape[0])
        M = tf.zeros([n, n], dtype=self.indata.dtype)
        for t in mcstat:
            if t["hess_dense"] is None:
                raise NotImplementedError(
                    "mcstat moment term must use a dense Hessian (sparse M not "
                    "yet supported for the sandwich covariance)."
                )
            idx = t["indices"]
            coords = tf.reshape(
                tf.stack(tf.meshgrid(idx, idx, indexing="ij"), axis=-1), [-1, 2]
            )
            # objective uses hess = -M, so M = -hess_dense
            updates = tf.reshape(-t["hess_dense"], [-1])
            M = tf.tensor_scatter_nd_add(M, coords, updates)
        return M

    def _compute_nll_meat(self):
        """Plain full-sample NLL (no jackknife de-bias), used as the two-half
        sandwich meat H = grad^2 L_full. Mirrors _compute_nll -- every term of the
        objective, regularization and the bin-by-bin stat compensator included
        -- but always uses the full templates and the ordinary per-bin ln."""
        nexpfullcentral, _, beta = self._compute_yields_with_beta(
            profile=True,
            compute_norm=False,
            full=self._regularizers_need_observables(),
        )
        nexp = nexpfullcentral[: self.indata.nbins]
        l = self._compute_ln(nexp) + self._compute_lc()
        lbeta = self._compute_lbeta(beta)
        if lbeta is not None:
            l = l + lbeta
        lpenalty = self._compute_lpenalty(nexpfullcentral)
        if lpenalty is not None:
            l = l + lpenalty
        lext = self._compute_external_nll()
        if lext is not None:
            l = l + lext
        if self.mcstat_bb_vref is not None:
            l = l + self._compute_mcstat_bb_compensator(meat=True)
        return l

    def hess_meat(self):
        """Dense Hessian of _compute_nll_meat (batched under --hvpBatch)."""
        return self._scalar_hessian("meat", self._compute_nll_meat)

    def mcstat_coverage_warnings(self):
        """Warnings for the de-biasing configurations whose reported covariance
        is known to under- or over-cover (docs/mcstat/RESULTS.md §9p, §9u);
        empty otherwise. Logged when the fitter is built."""
        if not self.mcstat_debias_active:
            return []
        out = []
        folds = self._fold_debias_active()
        if self.mcStatDebiasCov == "curvature":
            out.append(
                "--mcStatDebiasCov curvature reports the inverse de-biased "
                "curvature, which UNDER-COVERS (it omits the template noise in the "
                "score variance); use --mcStatDebiasCov sandwich."
            )
        elif self.mcStatDebiasCov == "dataPropagated":
            out.append(
                "--mcStatDebiasCov dataPropagated is conservative and OVER-COVERS "
                "(about 0.8 for a 68% interval on the validation toys, "
                "docs/mcstat/RESULTS.md §9p); --mcStatDebiasCov sandwich covers "
                "correctly."
            )
        elif folds and not self.bbstat.enabled:
            out.append(
                f"--mcStatDebias {self.mcStatDebias} without bin-by-bin stat: the "
                "reported covariance lacks the first-order template-noise variance "
                "and UNDER-COVERS when the MC statistical variance is comparable to "
                "the data's (0.54 for a 68% interval at MC/data variance ratio 1 on "
                "the validation toy, docs/mcstat/RESULTS.md §9u); enable bin-by-bin "
                "stat."
            )
        if (
            folds
            and self.covMode == "fisher"
            and self.mcStatDebiasCov == "sandwich"
            and self.bbstat.enabled
            and self.bbstat.binByBinStatType == "gamma"
            and self._bbstat_variance_moves()
            and (self.bbstat.dynamic or self.bbstat.binByBinStatMode == "full")
        ):
            out.append(
                "--covMode fisher with --binByBinStatType gamma and a "
                "parameter-dependent bin-by-bin stat variance (full mode or "
                "--binByBinStatDynamic): the fold methods' covariance is mildly "
                "conservative and OVER-COVERS (0.73 against 0.69 in observed mode "
                "for a 68% interval at MC/data variance ratio 1 on the validation "
                "toy, docs/mcstat/RESULTS.md §9u); --covMode observed is recommended."
            )
        return out

    def _check_mcstat_moment_variance(self):
        """Refuse an M whose denominator convention does not match the fit.

        M_ij = sum_b (sum_e w_e^2 s_i s_j) / V_b depends on the per-bin variance
        V_b the fit itself uses: with bin-by-bin stat on (any type or mode), the
        inflated mu_b + sum_proc sumw2_b; without it, the bare data/expected
        variance (docs/mcstat/RABBIT_MCSTAT_DESIGN.md §1b). A mismatched M over- (or
        under-) subtracts the curvature, and H - M can go indefinite far from
        where M was produced, so the convention recorded by
        TensorWriter.add_mc_stat_moment is checked here rather than trusted.
        """
        if self.mcstat_M is None:
            return
        expected = "bb_inflated" if self.bbstat.enabled else "bare"
        conventions = getattr(self.indata, "mcstat_moment_variance", {})
        for t in self.mcstat_terms:
            got = conventions.get(str(t["name"]))
            # the name written before full mode and dynamic bin-by-bin stat
            # were supported
            got = "bb_inflated" if got == "bblite_inflated" else got
            if got is None:
                raise RuntimeError(
                    f"MC-stat moment term '{t['name']}' does not record the "
                    "variance used in its denominator; rewrite the input with "
                    "TensorWriter.add_mc_stat_moment(..., variance=...)."
                )
            if got != expected:
                raise RuntimeError(
                    f"MC-stat moment term '{t['name']}' was computed with the "
                    f"'{got}' per-bin variance in its denominator, but this fit "
                    f"uses the '{expected}' one (bin-by-bin stat "
                    f"{'on' if self.bbstat.enabled else 'off'}). Recompute M with "
                    f"variance='{expected}', or "
                    f"{'disable' if self.bbstat.enabled else 'enable'} "
                    "bin-by-bin stat to match it."
                )

    def _mcstat_bb_noise_scales(self):
        """[nbins, ncomp] standard deviations of the template noise that
        bin-by-bin stat models, as functions of the parameters: one component
        per beta, u = sigma * m(x) / m0, with m the yield beta scales.

        - multiplicative types, full mode: one beta per (bin, process), scaling
          that process's yield (the relative variance sumw2 / sumw^2 is fixed);
        - --binByBinStatDynamic (any type): the same per-process scaling, by
          construction (BinByBinStat._dynamic_variance);
        - multiplicative types, lite mode: one beta per bin, scaling the yield
          of the processes with MC variance;
        - normal-additive, static (lite or full): absolute shifts whose variance
          does not depend on the parameters (returns None).
        """
        bb = self.bbstat
        nb = self.indata.nbins
        _, norm = self._compute_yields_noBBB(full=False, compute_norm=True)
        norm = norm[:nb]
        betamask = bb.betamask[:nb]
        additive = bb.binByBinStatType == "normal-additive"
        if additive and not bb.dynamic:
            return None
        if bb.dynamic or bb.binByBinStatMode == "full":
            sumw = self.indata.sumw[:nb]
            sumw2 = self.indata.sumw2[:nb]
            ok = (
                (sumw != 0.0)
                & (sumw2 > 0.0)
                & ~(betamask if bb.binByBinStatMode == "full" else betamask[:, None])
            )
            scale = tf.sqrt(tf.where(ok, sumw2, 0.0)) / tf.where(ok, sumw, 1.0)
            return tf.where(ok, scale * norm, 0.0)
        if bb.proc_zero_var_mask is not None:
            zv = bb.proc_zero_var_mask[:nb]
            n_active = tf.reduce_sum(tf.where(zv, tf.zeros_like(norm), norm), axis=-1)
        else:
            n_active = tf.reduce_sum(norm, axis=-1)
        ok = ~betamask
        scale = tf.sqrt(tf.where(ok, bb.varbeta[:nb], 0.0)) / tf.where(
            ok, bb.sumw[:nb], 1.0
        )
        return tf.where(ok, scale * n_active, 0.0)[:, None]

    def _bbstat_variance_moves(self):
        """Whether the variance of the profiled bin-by-bin stat beta moves with
        the parameters (_mcstat_bb_noise_scales is not None)."""
        bb = self.bbstat
        return bb.enabled and (bb.dynamic or bb.binByBinStatType != "normal-additive")

    def _bb_model_variance(self):
        """Per-bin variance bin-by-bin stat assigns to the prediction at the
        current point (None without bin-by-bin stat): sum_c u_bc^2, or the
        fixed varbeta of a static normal-additive one."""
        if not self.bbstat.enabled:
            return None
        nb = self.indata.nbins
        if not self._bbstat_variance_moves():
            var = self.bbstat.varbeta[:nb]
            return tf.reduce_sum(var, axis=-1) if len(var.shape) > 1 else var
        return tf.reduce_sum(tf.square(self._mcstat_bb_noise_scales()), axis=-1)

    def _update_mcstat_bb_vref(self):
        """Set the reference per-bin variances of the compensator: D_b the data
        variance (chisq) or the prefit expectation (Poisson), and V_b = D_b +
        sum_proc sumw2_b (M's 'bb_inflated' convention)."""
        nb = self.indata.nbins
        if self.chisqFit:
            var_data = tf.convert_to_tensor(self.varnobs)[:nb]
        elif self.covarianceFit:
            var_data = tf.linalg.diag_part(tf.linalg.inv(self.data_cov_inv))[:nb]
        else:
            var_data = tf.convert_to_tensor(self.nexpnom)[:nb]
        sumw2 = self.indata.sumw2[:nb]
        var_mc = tf.reduce_sum(sumw2, axis=-1) if len(sumw2.shape) > 1 else sumw2
        self.mcstat_bb_dref.assign(var_data)
        self.mcstat_bb_vref.assign(var_data + var_mc)

    def _compute_mcstat_bb_compensator(self, meat=False):
        """Bin-by-bin stat compensator, with V_b(x) = D_b + sum_c u_bc(x)^2
        (u from _mcstat_bb_noise_scales, D_b and V_ref,b the reference
        variances of _update_mcstat_bb_vref):

            continuous-M, and the meat:  1/2 sum_b log(V_b(x) / V_ref,b)
            fold objective:              1/2 sum_b D_b (1 / V_ref,b - 1 / V_b(x))

        The profiled bin-by-bin stat term is ~ r_b^2 / (2 V_b(x)). Where V
        moves with the parameters, its gradient -r^2 dV / (2 V^2) is a
        de-attenuation of its own: per-process betas (full mode, dynamic) are
        the errors-in-variables profile, which removes the noise floor of the
        per-process templates; a lite multiplicative beta removes that of the
        total yield. The de-biasing (the continuous-M penalty, the cross-fit
        objective) removes the whole noise floor, so without this term it
        would be removed twice.

        With noisy templates E[r^2] = V, and the gradient of 1/2 log V,
        dV / (2 V), cancels it exactly. 1/2 log V is the difference between
        integrating beta out and profiling it (Gaussian case): continuous-M
        then minimizes the marginal bin-by-bin stat likelihood less
        1/2 theta^T M theta, and the meat is the marginal likelihood's
        curvature. After the jackknife the template noise is gone from the
        residual, E[r^2] = D, and -D / (2 V), of gradient D dV / (2 V^2),
        cancels it exactly instead: it centres each profiled term on its
        expectation, r^2 / (2 V) - D / (2 V), an estimating-equation correction
        rather than a likelihood (docs/mcstat/RABBIT_MCSTAT_DESIGN.md §1b,
        §2b). Both vanish at the reference point."""
        u = self._mcstat_bb_noise_scales()
        vref = self.mcstat_bb_vref
        ok = vref > 0.0
        vref_safe = tf.where(ok, vref, tf.ones_like(vref))
        V = tf.where(
            ok, self.mcstat_bb_dref + tf.reduce_sum(tf.square(u), -1), vref_safe
        )
        if self._fold_debias_active() and not meat:
            terms = self.mcstat_bb_dref * (1.0 / vref_safe - 1.0 / V)
        else:
            terms = tf.math.log(V / vref_safe)
        return 0.5 * tf.reduce_sum(tf.where(ok, terms, 0.0))

    def mcstat_bb_compensator_value(self):
        """Value of _compute_mcstat_bb_compensator at the current point (0 if
        there is none): not part of the goodness-of-fit statistic."""
        if self.mcstat_bb_vref is None:
            return 0.0
        return float(self._compute_mcstat_bb_compensator())

    @property
    def mcstat_asimov(self):
        """Whether the current data are marked as an Asimov dataset of a
        de-biased fit (set_mcstat_asimov)."""
        return self._mcstat_asimov_on

    def set_mcstat_asimov(self, asimov):
        """Mark the current data as an Asimov dataset of the current parameters
        (asimov=True) or as anything else (False). A no-op for a fit without
        MC-stat de-biasing.

        Asimov data are the expected yields of the (noisy) templates
        themselves, so they carry no attenuation, and the de-biasing would
        overshoot: the de-biased objective is not stationary at the generating
        point x0, although the de-biased estimator's expectation over data and
        template fluctuations is x0 by design. Its gradient g0 at x0 is
        recorded, and the objective carries -g0^T (x - x0) (_compute_mcstat_
        penalty), whose minimum is x0; being linear it leaves every Hessian
        unchanged. Scans, contours and limits on the Asimov data are then
        centred at x0.

        The observed curvature at Asimov is not the expected one either: the
        bin-by-bin stat compensator stands in for the expected squared residual
        while the residuals vanish (and the fold half terms keep residuals
        -Delta_h). The Gauss-Newton forms are the expectations of the observed
        bread and meat and do not depend on the data, so the covariance of an
        Asimov dataset uses --covMode fisher (self.covMode), restored when
        unset. (docs/mcstat/RABBIT_MCSTAT_DESIGN.md §5a)
        """
        if not self.mcstat_debias_active:
            return
        self.mcstat_asimov_g0.assign(tf.zeros_like(self.mcstat_asimov_g0))
        self.covMode = self._covMode_requested
        self._mcstat_asimov_on = False
        if not asimov:
            return
        _, g = self.loss_val_grad()
        self.mcstat_asimov_x0.assign(self.x)
        self.mcstat_asimov_g0.assign(g)
        self.covMode = "fisher"
        self._mcstat_asimov_on = True
        logger.info(
            "Asimov dataset with MC-stat de-biasing: the objective is recentred "
            "at the generating point and the covariance uses --covMode fisher."
        )

    def _check_mcstat_empty_halves(self, values):
        """Refuse data a Poisson fold de-biasing cannot fit: a bin with
        observed events where one fold half has no template for any process
        (all of the bin's MC events fell into the other half). That half's
        prediction is zero at any parameter values, so its Poisson term, and the
        de-biased objective, is infinite. The folds depend only on the
        simulation, so this is not handled by changing them."""
        mask = getattr(self, "_mcstat_empty_half_bins", None)
        if (
            mask is None
            or self.chisqFit
            or self.covarianceFit
            or self.mcStatDebias != "kfold"
        ):
            return
        bad = mask & (np.asarray(values)[: mask.shape[0]] > 0.0)
        if np.any(bad):
            idx = np.flatnonzero(bad)
            raise RuntimeError(
                f"--mcStatDebias {self.mcStatDebias}: {idx.size} bins with observed "
                "events have no MC-stat template in one fold half for any process "
                f"(first bins: {idx[:10].tolist()}), so that half predicts zero "
                "events at any parameter values and its Poisson likelihood is "
                "infinite. Use more MC statistics or fewer folds for these bins, "
                "or a coarser binning."
            )

    def _init_mcstat_folds(self):
        """Prepare the cross-fit (two-half / k-fold) fold templates.

        indata.norm_folds holds the k per-fold nominal templates: dense
        [k, nbinsfull, nproc], or in sparse mode [k, nnz] values on the sparsity
        pattern of indata.norm. Any k >= 2 is supported. The objective splits the
        folds into the halves G_A = {0..a-1}, G_B = {a..k-1}, a = k//2, b = k-a,
        each rescaled to the full-sample normalization, and combines them with
        the jackknife weights of _jackknife_terms; the curvature in
        --covMode fisher uses the raw folds (complete U-statistic,
        _fisher_data_terms).
        """
        self.n_folds = None
        self.n_half_folds = None
        self.norm_halves = None
        # fold templates replacing the stored ones while the split meat
        # differentiates the objective with respect to them (_fold_norm)
        self._mcstat_fold_override = None
        self.jackknife_weights = None
        # split-logk (systematics written with a fold_axis): per-fold logk,
        # dense [k, nbinsfull, nproc, (2,) nsyst] or, in sparse mode, a per-fold
        # delta against the shared logk over the folded systematics only,
        # [k, nnz, nfolded] (log_normal only).
        self.logk_folds = getattr(self.indata, "logk_folds", None)
        self.logk_folds_delta = getattr(self.indata, "logk_folds_delta", None)
        self.mcstat_folded_syst_idx = getattr(
            self.indata, "mcstat_folded_syst_idx", None
        )
        if (
            self.logk_folds_delta is not None
            and self.indata.systematic_type != "log_normal"
        ):
            raise NotImplementedError(
                "sparse split-logk is supported only for log_normal systematics."
            )
        norm_folds = getattr(self.indata, "norm_folds", None)
        if norm_folds is None:
            return
        k = int(norm_folds.shape[0])
        a = k // 2
        b = k - a
        self.n_folds = k
        self.n_half_folds = (a, b)
        dtype = self.indata.dtype
        self.norm_halves = tf.stack(
            [
                tf.constant(k / a, dtype) * tf.reduce_sum(norm_folds[:a], axis=0),
                tf.constant(k / b, dtype) * tf.reduce_sum(norm_folds[a:], axis=0),
            ]
        )
        # bins where a half's nominal template is zero for every process: with
        # multiplicative systematics its prediction is zero at any parameter
        # values, which a Poisson fit cannot accommodate if the bin has data
        # (checked in set_nobs). Additive systematics could fill it: no check.
        self._mcstat_empty_half_bins = None
        if self.indata.systematic_type == "log_normal":
            nb = self.indata.nbins
            if self.indata.sparse:
                rows = self.indata.norm.indices[:, 0]
                tot = tf.transpose(
                    tf.math.unsorted_segment_sum(
                        tf.transpose(self.norm_halves),
                        rows,
                        num_segments=self.indata.nbinsfull,
                    )
                )
            else:
                tot = tf.reduce_sum(self.norm_halves, axis=-1)
            self._mcstat_empty_half_bins = np.asarray(
                tf.reduce_any(tot[:, :nb] == 0.0, axis=0)
            )
        # L_cf = w_full L_full - w_A L_A - w_B L_B. With raw half sums S_A, S_B
        # (J_full = S_A + S_B, J_A = k/a S_A, J_B = k/b S_B) these are the unique
        # weights whose Gauss-Newton Hessian is the pure cross term
        # sym(J_A^T D J_B) -- the self terms S_A^T D S_A, S_B^T D S_B cancel --
        # and whose score is the half average 1/2 (J_A + J_B)^T D (y - n). For
        # even k (a = b) they reduce to the familiar (2, 1/2, 1/2).
        self.jackknife_weights = (k * k / (2.0 * a * b), a / (2.0 * b), b / (2.0 * a))

    def _fold_debias_active(self):
        return (
            self.mcStatDebias == "kfold"
            and self.n_folds is not None
            and not self._mcstat_debias_suspended
        )

    def _mthetaalpha(self, theta):
        """Systematic coefficient columns contracted with logk: theta for a
        symmetric tensor, [theta, theta*alpha] for the asymmetric interpolation.
        theta [nsyst] gives one column, [nsyst, 1]; theta [nsyst, B] gives B."""
        # the column count is explicit: with nsyst == 0, a -1 would be inferred
        # from the non-zero dimensions alone
        ncol = 1 if theta.shape.rank == 1 else tf.shape(theta)[-1]
        if self.indata.symmetric_tensor:
            return tf.reshape(theta, [self.indata.nsyst, ncol])
        twox = 2.0 * theta
        twox2 = twox * twox
        alpha = 0.125 * twox * (twox2 * (3.0 * twox2 - 10.0) + 15.0)
        alpha = tf.clip_by_value(alpha, -1.0, 1.0)
        thetaalpha = theta * alpha
        mthetaalpha = tf.stack([theta, thetaalpha], axis=0)  # [2, nsyst(, B)]
        return tf.reshape(mthetaalpha, [2 * self.indata.nsyst, ncol])

    def _fold_norm(self, halves):
        """The raw fold templates ([k, ...], halves=False) or the two rescaled
        halves (halves=True) the fold predictions are built from: the stored
        ones, or the tensor set in _mcstat_fold_override while the split meat
        differentiates the objective with respect to them."""
        F = self._mcstat_fold_override
        if F is None:
            return self.norm_halves if halves else self.indata.norm_folds
        if not halves:
            return F
        k = self.n_folds
        a, b = self.n_half_folds
        return tf.stack(
            [
                (k / a) * tf.reduce_sum(F[:a], axis=0),
                (k / b) * tf.reduce_sum(F[a:], axis=0),
            ]
        )

    def _compute_fold_yields(self, halves=False, full=False, return_norm=False):
        """Predicted yields of the MC-stat folds, without bin-by-bin stat.

        halves=False: the k RAW per-fold predictions n_i, [k, nbins], used by
        the U-statistic curvature. halves=True: the two rescaled half
        predictions [2, nbins] of the jackknife objective, n_A = k/a sum_{i<a} n_i
        and n_B = k/b sum_{i>=a} n_i.

        Systematics shared across folds apply the full-sample variation to every
        fold: the multiplicative (log_normal) factor unchanged, the additive
        (normal) variation split evenly -- 1/k per raw fold, so the folds sum to
        the full prediction, and the full variation for a rescaled half. Folded
        systematics (split logk) use each fold's own variation.

        All folds are evaluated in one batched contraction, so the graph does not
        grow with k.

        return_norm: also return the per-process predictions [m, nbins, nproc]
        (dense), which bin-by-bin stat needs in full mode, with zero-variance
        processes, or with --binByBinStatDynamic.
        """
        k = self.n_folds
        split = self.logk_folds is not None or self.logk_folds_delta is not None
        if halves and split:
            # the halves of a split-logk systematic are sums of fold predictions
            out = self._compute_fold_yields(
                halves=False, full=full, return_norm=return_norm
            )
            a, b = self.n_half_folds

            def halve(n):
                return tf.stack(
                    [
                        (k / a) * tf.reduce_sum(n[:a], axis=0),
                        (k / b) * tf.reduce_sum(n[a:], axis=0),
                    ]
                )

            if return_norm:
                return halve(out[0]), halve(out[1])
            return halve(out)
        norm = self._fold_norm(halves)
        # scale of a SHARED additive variation in each template of `norm`
        add_scale = 1.0 if halves else 1.0 / k

        poi = self.get_poi()
        model_nui = self.get_model_nui()
        theta = self.get_theta()
        all_params = tf.concat([poi, model_nui], axis=0)
        mthetaalpha = self._mthetaalpha(theta)
        log_normal = self.indata.systematic_type == "log_normal"

        if self.indata.sparse:
            # fold templates are values on the sparsity pattern of indata.norm
            idx = self.indata.norm.indices
            nbinsfull = self.indata.nbinsfull
            rnorm = self.param_model.compute(all_params, True)
            rnorm = tf.gather_nd(
                tf.broadcast_to(rnorm, [nbinsfull, self.indata.nproc]), idx
            )
            logsnorm = tf.squeeze(
                tf_sparse_csr.matmul(self.logk_csr, mthetaalpha), axis=-1
            )[None, :]
            if log_normal:
                if self.logk_folds_delta is not None:
                    theta_folded = tf.gather(theta, self.mcstat_folded_syst_idx)
                    logsnorm = logsnorm + tf.einsum(
                        "knj,j->kn", self.logk_folds_delta, theta_folded
                    )
                vals = norm * rnorm * tf.exp(logsnorm)
            else:
                if self.rnorm_init_at_norm is not None:
                    logsnorm = logsnorm * self.rnorm_init_at_norm
                vals = norm * rnorm + add_scale * logsnorm
            nexp = tf.transpose(
                tf.math.unsorted_segment_sum(
                    tf.transpose(vals), idx[:, 0], num_segments=nbinsfull
                )
            )
            if not full:
                nexp = nexp[:, : self.indata.nbins]
            if not return_norm:
                return nexp
            shape = [nbinsfull, self.indata.nproc]
            normcentral = tf.stack(
                [tf.scatter_nd(idx, vals[i], shape) for i in range(vals.shape[0])]
            )
            if not full:
                normcentral = normcentral[:, : self.indata.nbins]
            return nexp, normcentral

        rnorm = self.param_model.compute(all_params, full)
        if full or self.indata.nbinsmasked == 0:
            nbins = self.indata.nbinsfull
        else:
            nbins = self.indata.nbins
        nproc = self.indata.nproc
        norm = norm[:, :nbins]
        if split and not halves:
            # per-fold logk: each fold's own variation, already at fold scale
            logk = self.logk_folds[:, :nbins]
            mlogk = tf.reshape(logk, [k * nbins * nproc, -1])
            logsnorm = tf.reshape(tf.matmul(mlogk, mthetaalpha), [k, nbins, nproc])
            add_scale = 1.0
        else:
            mlogk = tf.reshape(self.logk[:nbins], [nbins * nproc, -1])
            logsnorm = tf.reshape(tf.matmul(mlogk, mthetaalpha), [1, nbins, nproc])
        if log_normal:
            normcentral = rnorm * tf.exp(logsnorm) * norm
        else:
            if self.rnorm_init is not None:
                logsnorm = logsnorm * self.rnorm_init[:nbins]
            normcentral = norm * rnorm + add_scale * logsnorm
        if return_norm:
            return tf.reduce_sum(normcentral, axis=-1), normcentral
        return tf.reduce_sum(normcentral, axis=-1)

    def _profiled_half_terms(self, profile=True):
        """[(n_h, beta_h)] for the two rescaled halves, each with its own
        bin-by-bin stat beta, profiled on that half's prediction with the
        full-sample kstat (beta_h None without bin-by-bin stat).

        The jackknife then combines the profiled likelihoods themselves, the
        smooth function of the templates whose O(1/N_MC) bias it removes. Giving
        the halves the full-sample beta instead weights their MC deviations by
        the data variance D rather than the profiled V = D + sigma_MC^2, which
        over-subtracts the noise floor by a factor 1 + rho
        (docs/mcstat/RESULTS.md §9u). The full-sample kstat is kept because it
        is part of the function being jackknifed: per-half kstat would remove
        the bin-by-bin stat variance itself as a 1/N_MC "bias".

        With profile=False the halves keep the betas stored by _profile_beta.
        Their betas are not the fit's bin-by-bin stat nuisances, so they carry
        no ubeta.
        """
        nb = self.indata.nbins
        if not self.bbstat.enabled:
            n_h = self._compute_fold_yields(halves=True)
            return [(n_h[0], None), (n_h[1], None)]
        if self.bbstat.needs_per_proc_norm():
            n_h, norm_h = self._compute_fold_yields(halves=True, return_norm=True)
        else:
            n_h, norm_h = self._compute_fold_yields(halves=True), None
        out = []
        for h in range(2):
            nexp_h, _, beta_h = self.bbstat.profile_and_apply(
                n_h[h],
                None if norm_h is None else norm_h[h],
                self.nobs,
                self.varnobs,
                self.lognobs,
                profile=profile,
                full=False,
                fixed_beta=None if profile else self.mcstat_half_beta[h],
                add_ubeta=False,
            )
            out.append((nexp_h[:nb], beta_h))
        return out

    def _jackknife_terms(self, nexp, beta, profile=True):
        """(prediction, weight, beta) terms of the cross-fit jackknife
        objective L_cf = w_full L_full - w_A L_A - w_B L_B (weights:
        _init_mcstat_folds), each term a profiled bin-by-bin stat likelihood:
        `nexp`, `beta` are the full term's, the halves' from
        _profiled_half_terms."""
        w_full, w_A, w_B = self.jackknife_weights
        (n_A, beta_A), (n_B, beta_B) = self._profiled_half_terms(profile)
        return [(nexp, w_full, beta), (n_A, -w_A, beta_A), (n_B, -w_B, beta_B)]

    def _profiled_data_nll(self, debiased):
        """The part of the objective (debiased) or of the meat whose observed
        Hessian --covMode fisher replaces by the Gauss-Newton form: the
        (jackknifed) profiled data terms, bin-by-bin stat constraint included,
        and the bin-by-bin stat compensator. Their residual-dependent curvature
        (residual * d2n, and the r^2 terms of a variance that moves with the
        parameters, which the compensator cancels) is what the Gauss-Newton
        form drops."""
        nexp_full, _, beta = self._compute_yields_with_beta(
            profile=True,
            compute_norm=False,
            full=self._regularizers_need_observables(),
        )
        nexp_full = nexp_full[: self.indata.nbins]
        if debiased and self._fold_debias_active():
            terms = self._jackknife_terms(nexp_full, beta)
        else:
            terms = [(nexp_full, 1.0, beta)]
        l = tf.add_n([c * self._compute_ln(n) for n, c, _ in terms])
        if self.bbstat.enabled:
            l = l + tf.add_n([c * self._compute_lbeta(b) for _, c, b in terms])
        if self.mcstat_bb_vref is not None and not (
            debiased and self._mcstat_debias_suspended
        ):
            l = l + self._compute_mcstat_bb_compensator(meat=not debiased)
        return l

    def _fisher_data_terms(self, debiased):
        """Terms (n, coeff) whose Gauss-Newton sum sum_i coeff_i J_i^T D J_i is
        the de-biased data Fisher F_data used as the GN curvature. `n` is
        [nbins] or a stack [m, nbins] whose rows share `coeff`.

        For a fold-based debias this is the COMPLETE k-fold U-statistic (k-fold
        AVERAGING): using J_sumfold = sum_i J_i^raw (raw, unrescaled folds),

            F_ch = k/(k-1) [ J_sumfold^T D J_sumfold - sum_i J_i^raw^T D J_i^raw ]
                 = k/(k-1) sum_{i!=j} J_i^raw^T D J_j^raw

        i.e. the average over ALL fold pairs, in O(k) via the full-minus-self
        identity, for any k >= 2. For k=2 it equals the single 2-half
        cross-half Fisher exactly; for larger k it reduces the curvature-
        estimate variance toward the continuous-M floor. The RAW folds
        (~n_full/k) are used only here (the GN form is residual-independent),
        never in the objective ln (which keeps the rescaled halves).

        The predictions are the raw ones, without bin-by-bin stat: its profile
        enters through the weight of _gauss_newton_surrogate (the Gauss-Newton
        form of a profiled term is J^T J / (D + sigma_bb^2)), which keeps the
        full-minus-self identity of the U-statistic exact."""
        nbins = self.indata.nbins
        if not self._fold_debias_active():
            nexp_full = self._compute_yields_noBBB(
                full=self._regularizers_need_observables(), compute_norm=False
            )[0][:nbins]
            return [(nexp_full, 1.0)]

        # Per-fold raw predictions n_i (each with its own logk for split-logk).
        # The "full" term is the SUM of the per-fold predictions,
        # J_sumfold = sum_i J_i, so the full-minus-self identity holds EXACTLY
        # for both shared logk (where sum_i n_i == nexp_full) and split logk
        # (where it does not). Using n_sumfold for the meat too keeps bread/meat
        # consistent (H - A = M) under split-logk.
        per_fold = self._compute_fold_yields(halves=False)
        n_sumfold = tf.reduce_sum(per_fold, axis=0)
        if debiased:
            coeff = self.n_folds / (self.n_folds - 1.0)
            return [(n_sumfold, coeff), (per_fold, -coeff)]
        return [(n_sumfold, 1.0)]

    def _gauss_newton_surrogate(self, fisher_terms):
        """Scalar whose Hessian at the current point is sum_i c_i J_i^T D J_i.

        S(x) = 1/2 sum_i c_i r_i^T D r_i with r_i = n_i(x) - stop_gradient(n_i(x)):
        the residual is identically zero at the evaluation point, so the
        residual * d2n piece of its Hessian vanishes and only J^T D J survives.
        D = the curvature of a profiled per-bin term, treated as constant:
        1 / (var + sigma_bb^2) with var the data variance (chisq) or the
        prediction (Poisson) and sigma_bb^2 the variance bin-by-bin stat
        assigns to the prediction (_bb_model_variance; 0 without it), and
        (C + diag(sigma_bb^2))^-1 for covFit. The Poisson weight is the SHARED
        full / sum-of-folds prediction (fisher_terms[0]) for every term: a
        per-term 1/n_i (~k/nexp_full for a raw fold) breaks the full-minus-self
        identity of the U-statistic and leaves a -sum_i F_i self-term that
        drives the bread indefinite.
        """
        nref = tf.stop_gradient(fisher_terms[0][0])
        var_bb = self._bb_model_variance()
        if var_bb is not None:
            var_bb = tf.stop_gradient(var_bb)
        if self.covarianceFit:
            Cinv = self.data_cov_inv
            if var_bb is not None:
                Cinv = tf.linalg.inv(tf.linalg.inv(Cinv) + tf.linalg.diag(var_bb))
        else:
            var = self.varnobs if self.chisqFit else nref
            if var_bb is not None:
                var = var + var_bb
            D = tf.math.divide_no_nan(tf.ones_like(nref), var)
        terms = []
        for n, c in fisher_terms:
            r = n - tf.stop_gradient(n)
            if self.covarianceFit:
                q = tf.reduce_sum(r * tf.linalg.matvec(Cinv, r))
            else:
                q = tf.reduce_sum(D * tf.square(r))
            terms.append(0.5 * c * q)
        return tf.add_n(terms)

    def _fisher_core(self, hess_obj, debiased):
        """Gauss-Newton (expected-information) curvature corresponding to the
        observed-Hessian `hess_obj`:

            F = hess_obj - H_ln_obs + F_data

        H_ln_obs is the observed Hessian of the OBJECTIVE's profiled data terms
        (_profiled_data_nll: bin-by-bin stat and its compensator included), so
        `hess_obj - H_ln_obs` is the remaining curvature (constraints,
        regularization, external terms, the continuous-M -M). F_data is the
        de-biased data Gauss-Newton Fisher from
        _fisher_data_terms (the complete k-fold U-statistic for the bread; the
        full J^T D J for the meat). Drops the residual-dependent pieces while
        keeping the rest (docs/mcstat/RABBIT_MCSTAT_DESIGN.md §2c/§2d).

        F_data - H_ln_obs is evaluated as ONE Hessian, of the Gauss-Newton
        surrogate minus the data term, the same way as hess_obj (one
        reverse-over-reverse jacobian, or batched HVPs under --hvpBatch):
        O(nparams) backward passes, each linear in nbins.
        No per-term [nbins, nparams] Jacobian is built (a reverse-mode jacobian
        of an [nbins] output costs one backward pass per bin, i.e. O(nbins^2)).
        """
        shift = self._scalar_hessian(
            f"gn_shift_{debiased}", lambda: self._gn_shift(debiased)
        )
        return hess_obj + tf.cast(shift, hess_obj.dtype)

    def fisher_curvature(self, hess_obj):
        """Gauss-Newton curvature of the ACTIVE objective (jackknife if a
        two-half debias is on; else the full term). Used as the sandwich bread
        and as the standard-covariance curvature under --covMode fisher."""
        return self._fisher_core(hess_obj, True)

    def fisher_curvature_full(self, hess_meat):
        """Gauss-Newton curvature of the FULL (undebiased) objective. Used as the
        sandwich meat under --covMode fisher."""
        return self._fisher_core(hess_meat, False)

    def cov_twohalf_sandwich(self, A, covMode="observed"):
        """Two-half / k-fold robust (sandwich) covariance Sigma = A^-1 H A^-1.

        Bread A = the de-biased (jackknife) curvature; meat H = the full-sample
        curvature. Both follow `covMode` (don't mix): 'observed' = autograd
        Hessians (A = grad^2 L_cf, H = grad^2 L_full); 'fisher' = Gauss-Newton
        (A = F_ch + nondata, H = F_full + nondata). See
        docs/mcstat/RABBIT_MCSTAT_DESIGN.md §2c. The passed `A` must already be
        in the requested mode.
        """
        H = self.mcstat_meat(covMode)
        Ainv = tf.linalg.inv(A)
        return Ainv @ tf.cast(H, Ainv.dtype) @ Ainv

    def mcstat_reported_cov(self, bread, cov_curv, covMode="observed"):
        """The covariance a fit reports, given its curvature covariance
        cov_curv = bread^-1: unchanged without MC-stat de-biasing or with
        --mcStatDebiasCov curvature; otherwise the sandwich (continuous-M:
        A^-1 + A^-1 M A^-1; fold methods: A^-1 H A^-1) or the data-propagated
        sandwich."""
        if not self.mcstat_debias_active or self.mcStatDebiasCov == "curvature":
            return cov_curv
        if self.mcStatDebiasCov == "dataPropagated":
            return self.cov_dataprop_sandwich(cov_curv)
        if self.mcStatDebias == "continuousM":
            return self.cov_mcstat_sandwich(bread)
        return self.cov_twohalf_sandwich(bread, covMode=covMode)

    def mcstat_postfit_covariances(self, grad, hess):
        """(edmval, bread, curvature covariance, reported covariance) at the
        current point, from the gradient and observed Hessian of the objective:
        the bread is the observed Hessian or, under --covMode fisher, its
        Gauss-Newton form."""
        bread = self.fisher_curvature(hess) if self.covMode == "fisher" else hess
        edmval, cov_curv = self.edmval_cov(grad, bread)
        return (
            edmval,
            bread,
            cov_curv,
            self.mcstat_reported_cov(bread, cov_curv, covMode=self.covMode),
        )

    def mcstat_curvature_and_meat(self):
        """Dense bread A (de-biased curvature, observed or Gauss-Newton per
        --covMode) and the score variance B at the current point, for the
        saturated projection test.

        B is the sandwich meat (continuous-M: A + M; fold methods: the ordinary
        likelihood's curvature) where bin-by-bin stat carries the MC variance of
        the data term. Without bin-by-bin stat that meat has none, so the data
        term's MC variance is added explicitly, sum_b rho_b J_b J_b^T / V_b with
        rho_b the MC/data variance ratio (_mcstat_gof_bin_inputs)."""
        _, _, hess = self.loss_val_grad_hess()
        bread = self.fisher_curvature(hess) if self.covMode == "fisher" else hess
        if self.mcStatDebias == "continuousM":
            meat = bread + tf.cast(self.mcstat_M, bread.dtype)
        else:
            meat = self.mcstat_meat(self.covMode)
        if not self.bbstat.enabled:
            meat = meat + self._mc_data_variance_curvature()
        return np.asarray(bread), np.asarray(meat)

    def _mc_data_variance_curvature(self):
        """sum_b rho_b J_b J_b^T / V_b, J = d nexp / dx: the score variance the
        MC fluctuations of the predictions add to the data term, as the Hessian
        of a zero-residual surrogate (cf. _gauss_newton_surrogate)."""

        def surrogate():
            rho = tf.stop_gradient(self._mcstat_gof_bin_inputs()[1])
            n = self._compute_yields_noBBB(full=False, compute_norm=False)[0][
                : self.indata.nbins
            ]
            r = n - tf.stop_gradient(n)
            if self.chisqFit:
                D = 1.0 / self.varnobs
            elif self.covarianceFit:
                D = tf.linalg.diag_part(self.data_cov_inv)
            else:
                D = tf.math.divide_no_nan(tf.ones_like(n), tf.stop_gradient(n))
            return 0.5 * tf.reduce_sum(rho * D * tf.square(r))

        return self._scalar_hessian("mc_data_variance", surrogate)

    def mcstat_projection_check(self, psi_indices, rtol=1e-8):
        """Whether the saturated projection test of a de-biased fit is defined.

        Directions of the bin scales psi that the data cannot constrain must
        also be flat for the de-biased objective. Both are judged from the
        Gauss-Newton forms of the data term plus the constraints, whose null
        spaces are exact (J v = 0; the observed Hessians, and bin-by-bin stat's
        observed curvature, add residual * d2n terms that are small but not zero
        there): the profiled Fisher of the ordinary likelihood,
        H~ = H_pp - H_pl H_ll^-1 H_lp, and the profiled cross-fit Fisher, A~. A common normalization is
        flat for both (the halves scale with the full prediction). A projection
        that integrates out what separates the processes is not: the bin scales
        then absorb a change of process composition in the full prediction while
        the half predictions, whose composition fluctuates, still change, and the
        saturated de-biased fit would follow template noise (or run to the
        singularity of the Poisson jackknife). Returns (ok, worst ratio)."""
        psi = np.asarray(psi_indices)
        lam = np.setdiff1d(np.asarray(self.floating_indices), psi)
        A, H = (np.asarray(m) for m in self._mcstat_gn_data_curvatures())

        def schur(M):
            R = np.linalg.solve(M[np.ix_(lam, lam)], M[np.ix_(psi, lam)].T).T
            St = M[np.ix_(psi, psi)] - R @ M[np.ix_(psi, lam)].T
            return 0.5 * (St + St.T)

        A_t, H_t = schur(A), schur(H)
        h, V = np.linalg.eigh(H_t)
        null = V[:, np.abs(h) <= rtol * np.max(np.abs(h))]
        if null.shape[1] == 0:
            return True, 0.0
        scale = np.max(np.abs(np.linalg.eigvalsh(A_t)))
        worst = float(np.max(np.abs(np.linalg.eigvalsh(null.T @ A_t @ null)))) / scale
        return worst <= 1e-6, worst

    def _mcstat_gn_data_curvatures(self):
        """(cross-fit, full-sample) data Gauss-Newton Fisher plus the constraint
        and regularization curvature: the Hessians of the surrogates of
        _gauss_newton_surrogate."""

        def surrogate(debiased):
            val = (
                self._gauss_newton_surrogate(self._fisher_data_terms(debiased))
                + self._compute_lc()
            )
            lpenalty = self._compute_lpenalty(
                self._compute_yields_with_beta(
                    profile=True,
                    compute_norm=False,
                    full=self._regularizers_need_observables(),
                )[0]
            )
            if lpenalty is not None:
                val = val + lpenalty
            return val

        return [
            self._scalar_hessian(f"gn_data_{d}", lambda d=d: surrogate(d))
            for d in (True, False)
        ]

    def mcstat_projection_test(self, dq, psi_indices, bread, meat):
        """p-value of the saturated projection test of a de-biased fit.

        dq = 2 (L(main fit) - L(saturated fit)) is the difference of the
        de-biased objective between two nested fits, the saturated one adding
        the bin scales psi to the analysis parameters lambda. The objective is
        not a likelihood, so dq is not chi2(dim psi); as a composite-likelihood
        ratio statistic (Kent 1982) it is asymptotically sum_j mu_j Z_j^2, with
        mu_j the eigenvalues of A~^+ B~: A~ = A_pp - A_pl A_ll^-1 A_lp the
        profiled curvature of psi (Schur complement of the bread A) and
        B~ = P H P^T, P = [1, -A_pl A_ll^-1], the variance of the profiled score
        from the meat H, both at the null point (the main fit's minimum, scales
        at 1). Only A_ll is inverted, so a direction in which the bin scales are
        degenerate with the analysis parameters (e.g. a common normalization)
        is simply not testable and drops out. Returns (pvalue, mu)."""
        psi = np.asarray(psi_indices)
        lam = np.setdiff1d(np.asarray(self.floating_indices), psi)
        A = np.asarray(bread)
        H = np.asarray(meat)
        if not np.isfinite(dq):
            return float("nan"), np.full(len(psi), np.nan)
        A_ll = A[np.ix_(lam, lam)]
        A_pl = A[np.ix_(psi, lam)]
        R = np.linalg.solve(A_ll, A_pl.T).T  # A_pl A_ll^-1
        A_t = A[np.ix_(psi, psi)] - R @ A_pl.T
        P = np.concatenate([np.eye(len(psi)), -R], axis=1)
        idx = np.concatenate([psi, lam])
        B_t = P @ H[np.ix_(idx, idx)] @ P.T
        a, V = np.linalg.eigh(0.5 * (A_t + A_t.T))
        keep = a > 1e-10 * np.max(np.abs(a))
        T = V[:, keep] / np.sqrt(a[keep])
        mu = np.linalg.eigvalsh(T.T @ (0.5 * (B_t + B_t.T)) @ T)
        return weighted_chi2_sf(dq, mu), np.sort(mu)

    def _mcstat_split_meat_hvp_batch(self, P):
        """B_split p for the rows p of P [m, npar] (mcstat_split_meat).

        With F the fold templates, X(F) = d^2 L_cf / dx dF and C the covariance
        of F given the full sample: X^T p is the gradient with respect to F of
        p . grad_x L, and X w the gradient with respect to x of w . grad_F L, two
        mixed second derivatives of the actual objective, so any likelihood,
        bin-by-bin stat profile and parameter is covered.

        Given the full sample, the score depends on the split noise z (the
        deviation of F from its fold mean) linearly and quadratically: g(z) =
        g_0 + G z + z^T K z. Propagating X(z) C X(z)^T would double the
        quadratic part (for Gaussian z, E[(2Kz)^T C (2Kz)] = 4 tr(KCKC) =
        2 Var(z^T K z); for two folds the dependence is purely quadratic). The
        mean split z = 0 (every fold the fold mean, so every half prediction the
        full one, valid for any k) separates them: L = X(0) = G,
        Q = X(z) - X(0) = 2Kz, and

            B_split = L C L^T + 1/2 Q C Q^T

        has the exact expectation Var(g | full sample) for any mix of the two.
        (The reflected split z -> -z would do the same, but for odd k its half
        predictions can be negative.) Per row: B_split p = X(0) (wl - wq / 2) +
        X(z) (wq / 2), wl = C L^T p, wq = C Q^T p; four mixed second
        derivatives."""
        v = self.indata.sumw2_folds
        vsum = tf.reduce_sum(v, axis=0, keepdims=True)

        def cov(u):
            # the fold covariance given their sum: diag(v) - v v^T / sum(v)
            vu = tf.reduce_sum(v * u, axis=0, keepdims=True)
            return v * u - v * tf.math.divide_no_nan(vu, vsum)

        def loss(F):
            self._mcstat_fold_override = F
            try:
                return self._compute_loss()
            finally:
                self._mcstat_fold_override = None

        def x_transpose(F, p):
            with tf.GradientTape() as tF:
                tF.watch(F)
                with tf.GradientTape() as tx:
                    L = loss(F)
                gx = tx.gradient(L, self.x)
                s = tf.reduce_sum(gx * p)
            return tF.gradient(s, F, unconnected_gradients="zero")

        def x_times(F, w):
            with tf.GradientTape() as tx:
                with tf.GradientTape() as tF:
                    tF.watch(F)
                    L = loss(F)
                gF = tF.gradient(L, F, unconnected_gradients="zero")
                s = tf.reduce_sum(gF * w)
            return tx.gradient(s, self.x, unconnected_gradients="zero")

        F0 = self.indata.norm_folds
        Fmean = tf.broadcast_to(tf.reduce_mean(F0, axis=0, keepdims=True), F0.shape)

        def one(p):
            p = tf.stop_gradient(p)
            F = tf.identity(F0)
            F_0 = tf.identity(Fmean)
            u_z = x_transpose(F, p)
            u_0 = x_transpose(F_0, p)
            wl = tf.stop_gradient(cov(u_0))
            wq = tf.stop_gradient(cov(u_z - u_0))
            return x_times(F_0, wl - 0.5 * wq) + x_times(F, 0.5 * wq)

        return tf.vectorized_map(one, P)

    def _mcstat_split_meat_fn(self):
        fn = self._scalar_hessian_fns.get("split_meat")
        if fn is None:
            fn = tf.function(self._mcstat_split_meat_hvp_batch)
            self._scalar_hessian_fns["split_meat"] = fn
        return fn

    def _at_data_sigma_points(self, compute):
        """compute() at the current data or, for an Asimov dataset of a
        de-biased fit (set_mcstat_asimov), averaged over the data y = mu +-
        sqrt(var), var the fit's per-bin data variance (the data variance for
        chisq, the yield for Poisson, diag(C) of the data covariance C for
        covarianceFit): the split meat's mixed derivatives depend on the
        residuals, which vanish at Asimov; its expectation over the data needs
        them. The two points give it exactly for a per-bin quadratic
        dependence; with a correlated data covariance (covarianceFit) only its
        diagonal is sampled."""
        if not self._mcstat_asimov_on:
            return compute()
        nobs = tf.identity(self.nobs)
        varnobs = None if self.varnobs is None else tf.identity(self.varnobs)
        if varnobs is not None:
            var = varnobs
        elif self.covarianceFit:
            var = tf.linalg.diag_part(tf.linalg.inv(self.data_cov_inv))
        else:
            var = nobs
        out = []
        try:
            for sign in (1.0, -1.0):
                y = nobs + sign * tf.sqrt(var)
                if self.chisqFit:
                    y = tf.maximum(y, 1e-3 * nobs)  # chisq needs y > 0
                elif not self.covarianceFit:
                    y = tf.maximum(y, 0.0)  # Poisson
                self.set_nobs(y, variances=varnobs)
                out.append(np.asarray(compute()))
        finally:
            self.set_nobs(nobs, variances=varnobs)
        return 0.5 * (out[0] + out[1])

    def mcstat_split_meat(self):
        """Dense split-contrast meat B_split of a fold de-biasing (zero matrix
        otherwise): the score variance from how the events divide between the
        folds, given the full sample.

        The cross-fit score depends on the realized split, through a product
        of the halves' deviations in the templates and in their derivatives
        (delta_h Delta_h). Its variance is not in the full-sample meat; it is
        second order in the template-noise floor, so it matters where that floor
        is comparable to the information (large rho along weakly constrained
        directions). It is propagated from the fold templates through the
        objective, with X = d^2 L_cf / dx dF evaluated at the realized and the
        mean split (_mcstat_split_meat_hvp_batch: B_split = L C L^T +
        1/2 Q C Q^T, exact for the linear and quadratic dependence on the split
        noise), C = diag(v) - v v^T / sum(v) the covariance of the fold
        templates F given their sum (v the per-fold MC-stat variances; zero for
        processes without folds). The fold sum, the full sample, is held fixed:
        its noise is the full-sample meat's.
        Systematics with per-fold variations (split logk) contribute only
        through the nominal fold templates. Assembled from batched columns
        (--hvpBatch, default 256). (docs/mcstat/RABBIT_MCSTAT_DESIGN.md §2c)"""
        n = int(self.x.shape[0])
        if not self._fold_debias_active():
            return tf.zeros([n, n], dtype=self.indata.dtype)
        fn = self._mcstat_split_meat_fn()

        def compute():
            return self._hessian_from_hvp_batches(
                fn, np.arange(n), max(1, self.hvp_batch)
            )

        B = self._at_data_sigma_points(compute)
        return tf.constant(0.5 * (B + B.T), dtype=self.indata.dtype)

    def mcstat_split_meat_rows(self, rows):
        """B_split c for the rows c of `rows` [m, npar] (Hessian-free path)."""
        fn = self._mcstat_split_meat_fn()
        return self._at_data_sigma_points(
            lambda: fn(tf.constant(rows, dtype=self.indata.dtype))
        )

    def mcstat_meat(self, covMode="observed"):
        """Dense sandwich meat H: the curvature of the ordinary (full-sample)
        likelihood at the current point, in the given --covMode, plus for the
        fold methods the split-contrast meat (mcstat_split_meat). Continuous-M
        uses the analytic H = A + M instead (cov_mcstat_sandwich)."""
        H_obs = self.hess_meat()
        H = self.fisher_curvature_full(H_obs) if covMode == "fisher" else H_obs
        if self._fold_debias_active():
            H = H + tf.cast(self.mcstat_split_meat(), H.dtype)
        return H

    # --- Hessian-free (--noHessian) counterparts --------------------------

    def _gn_shift(self, debiased):
        """Scalar whose Hessian is F_data - H_ln_obs (see _fisher_core): adding
        it to an objective turns that objective's observed Hessian into the
        Gauss-Newton one, plus, for a marginal bin-by-bin stat term, the
        expectation of what that drops (_gn_marginal_variance_term)."""
        val = self._gauss_newton_surrogate(
            self._fisher_data_terms(debiased)
        ) - self._profiled_data_nll(debiased)
        if self.mcstat_bb_vref is not None and not (
            debiased and self._fold_debias_active()
        ):
            val = val - self._gn_marginal_variance_term()
        return val

    def _gn_marginal_variance_term(self):
        """Scalar whose Hessian is 1/2 sum_b dV_b dV_b^T / V_b^2 (dV_b =
        2 sum_c u_bc du_bc, u from _mcstat_bb_noise_scales), the expectation of
        the curvature the Gauss-Newton form drops from a marginal bin-by-bin
        stat term (profile + 1/2 log V: the meat, the continuous-M objective)
        when the bin-by-bin stat variance moves with the parameters.

        The dropped pieces are the r^2 terms of V(x), whose expectation with
        the compensator's 1/2 log V is +1/2 dV dV^T / V^2 (E[r^2] = V), and the
        cross term 2 r (J dV^T)_sym / V^2, whose expectation is -dV dV^T / V^2
        because the template noise enters both the residual and the Jacobian:
        E[r J] = -sum_c u_c du_c = -dV / 2. Dropping them (setting them to zero)
        is the expectation only where they vanish on average; here they leave
        -1/2 dV dV^T / V^2. For the fold objective (centred compensator,
        E[r^2] = D) they cancel exactly and nothing is added. Computed with a
        zero-residual surrogate, W_b = 1/2 sum_c u_bc^2: the Hessian of
        sum_b (W_b - W_b|current)^2 / V_b^2 is 2 sum_b dW_b dW_b^T / V_b^2.
        (docs/mcstat/RABBIT_MCSTAT_DESIGN.md §2c)"""
        u = self._mcstat_bb_noise_scales()
        W = 0.5 * tf.reduce_sum(tf.square(u), axis=-1)
        nb = self.indata.nbins
        if self.chisqFit:
            var_data = self.varnobs[:nb]
        elif self.covarianceFit:
            var_data = tf.linalg.diag_part(tf.linalg.inv(self.data_cov_inv))[:nb]
        else:
            var_data = self._compute_yields_noBBB(full=False, compute_norm=False)[0][
                :nb
            ]
        V = tf.stop_gradient(var_data + 2.0 * W)
        r = W - tf.stop_gradient(W)
        return tf.reduce_sum(tf.math.divide_no_nan(tf.square(r), tf.square(V)))

    def _hvp_function(self, scalar_fn):
        """tf.function p -> (d^2 scalar_fn / dx^2) p, reverse-over-reverse."""

        def hvp(p):
            p = tf.stop_gradient(p)
            with tf.GradientTape() as t2:
                with tf.GradientTape() as t1:
                    val = scalar_fn()
                g = t1.gradient(val, self.x)
            return t2.gradient(g, self.x, output_gradients=p)

        return tf.function(hvp)

    def bread_hessp(self, covMode="observed"):
        """HVP of the curvature the covariance is built from (the de-biased one
        when de-biasing is active): the objective's observed Hessian, or its
        Gauss-Newton form under --covMode fisher. Matches the dense `bread`."""
        if covMode == "fisher":
            return self._hvp_function(
                lambda: self._compute_loss() + self._gn_shift(True)
            )

        def hvp(p):
            return self.loss_val_grad_hessp(p)[2]

        return hvp

    def meat_hessp(self, covMode="observed"):
        """HVP of the sandwich meat (mcstat_meat), for the fold methods."""
        if covMode == "fisher":
            return self._hvp_function(
                lambda: self._compute_nll_meat() + self._gn_shift(False)
            )
        return self._hvp_function(self._compute_nll_meat)

    def cov_block_hessfree(self, grad, row_indices, covMode="observed", rtol=1e-10):
        """Hessian-free edmval and covariance block for --noHessian.

        Solves A c_i = e_i by conjugate gradient with the bread HVP (A as in the
        dense path: de-biased, and Gauss-Newton under --covMode fisher). The
        curvature block is C_ij = c_i[j]. With MC-stat de-biasing and
        --mcStatDebiasCov sandwich the reported block is the sandwich

            Sigma_ij = e_i^T A^-1 H A^-1 e_j = c_i^T H c_j,

        which needs one meat HVP per row (fold methods), or none for
        continuous-M, where H = A + M gives Sigma_ij = C_ij + c_i^T M c_j.

        Returns (edmval, curvature block, reported block), blocks of shape
        [len(row_indices)]^2.
        """
        row_indices = np.asarray(list(row_indices), dtype=np.int64)
        bread = self.bread_hessp(covMode)
        edmval, rows = self.edmval_cov_rows_hessfree(
            grad, row_indices, rtol=rtol, hessp=bread
        )
        curv = rows[:, row_indices]
        if not (self.mcstat_debias_active and self.mcStatDebiasCov == "sandwich"):
            return edmval, curv, curv
        if self.mcStatDebias == "continuousM":
            M = np.asarray(self.mcstat_M)
            return edmval, curv, curv + rows @ M @ rows.T
        meat = self.meat_hessp(covMode)
        Hc = np.stack(
            [np.asarray(meat(tf.constant(c, dtype=self.indata.dtype))) for c in rows]
        )
        Hc = Hc + np.asarray(self.mcstat_split_meat_rows(rows))
        sand = rows @ Hc.T
        return edmval, curv, 0.5 * (sand + sand.T)

    # --- goodness of fit at the de-biased minimum -------------------------

    @tf.function
    def _mcstat_gof_bin_inputs(self):
        """Per-bin inputs of the goodness-of-fit reference (mcstat_gof):

        shrink: s_b = 1 - d nhat_b / d y_b, the fraction of a residual y_b - n_b
            that bin-by-bin stat leaves in the data term (nhat = the profiled
            prediction at fixed parameters; exactly 1 without bin-by-bin stat).
            Taken from the profiling itself, so it holds for any bin-by-bin
            stat type and mode.
        rho: MC variance of the prediction over the data variance.
        """
        nbins = self.indata.nbins
        with tf.GradientTape() as t:
            t.watch(self.nobs)
            nhat = self._compute_yields_with_beta(
                profile=True, compute_norm=False, full=False
            )[0][:nbins]
            total = tf.reduce_sum(nhat)
        dnhat = t.gradient(total, self.nobs, unconnected_gradients="zero")
        shrink = 1.0 - dnhat

        var_data, var_mc = self._mcstat_bin_variances()
        rho = tf.math.divide_no_nan(var_mc, var_data)
        return shrink, rho

    def _mcstat_bin_variances(self):
        """Per-bin data variance and MC variance of the prediction at the
        current point (each process keeping its relative MC variance)."""
        nbins = self.indata.nbins
        nexp, norm = self._compute_yields_noBBB(full=False, compute_norm=True)
        nexp = nexp[:nbins]
        norm = norm[:nbins]
        if len(self.indata.sumw2.shape) == 2:
            var_mc = self.bbstat._dynamic_variance(norm)
        else:
            sumw = tf.reduce_sum(self.indata.sumw[:nbins], axis=-1)
            var_mc = self.indata.sumw2[:nbins] * tf.math.divide_no_nan(nexp, sumw) ** 2
        if self.chisqFit:
            var_data = self.varnobs[:nbins]
        elif self.covarianceFit:
            var_data = tf.linalg.diag_part(tf.linalg.inv(self.data_cov_inv))
        else:
            var_data = nexp
        return var_data, var_mc

    def mcstat_gof(self, q, trace_AinvH=None):
        """p-value of the saturated statistic of a fold-de-biased fit.

        q = 2 (L_cf(theta_hat) - L_sat) = 2 * reduced_nll(): the de-biased
        objective at its own minimum, bin-by-bin stat included, exactly as the
        ordinary saturated test (the jackknife weights sum to one, so L_cf is
        zero at the saturated point). Only its reference distribution differs.

        Per bin, with e = y - n the full residual (variance V (1 + rho): data and
        MC), Delta_A,B = n_A,B - n the halves' MC deviations (independent of e,
        E[Delta_A Delta_B] = -sigma_MC^2, Var(Delta_A + Delta_B) =
        sigma_MC^2 (a - b)^2 / (a b)), and s the shrink of _mcstat_gof_bin_inputs
        (a profiled term's curvature in its residual is s / V: bin-by-bin stat
        absorbs the fraction 1 - s of it), each jackknife term being separately
        profiled, to leading order

            q_b V = s [e^2 - e (Delta_A + Delta_B) + Delta_A Delta_B].

        So q is a weighted chi2, sum_b lambda_b z_b^2 with lambda_b = s_b (1 +
        rho_b), minus the fitted parameters, plus an approximately Gaussian G of

            mean  -sum_b s_b rho_b
            var    sum_b s_b^2 [2 rho_b^2 + (1 + rho_b) rho_b (a - b)^2 / (a b)].

        Without bin-by-bin stat (s = 1) the mean is nbins: the MC fluctuations
        cancel in the cross-fit chi2 sum_b D_b (y - n_A)(y - n_B). With it the
        mean is sum_b s_b = sum_b 1 / (1 + rho_b) for s = 1 / (1 + rho); a
        mismodelling shift d_b still raises E[q_b] by d_b^2 / (V_b + sigma_b^2),
        as in the ordinary test. q excludes the bin-by-bin stat compensator
        (mcstat_bb_compensator_value), an offset of order sum_b rho_b. The weighted chi2 is moment-matched to
        c chi2_nu (Satterthwaite) with mean sum_b lambda_b + n_constrained -
        tr(A^-1 H), tr(A^-1 H) being the effective number of fitted parameters
        (constrained ones are balanced by their pseudo-measurements; without
        trace_AinvH, under --noHessian, the parameter count is used), and the
        p-value is P(c chi2_nu + G >= q).

        Returns a dict with q, mean, var, c, nu, g_mean, g_sigma, pvalue.
        """
        if not self._fold_debias_active():
            raise RuntimeError("mcstat_gof needs the fold (kfold) de-biasing.")
        from scipy.stats import chi2 as _chi2
        from scipy.stats import norm as _norm

        nfree = int(self.nfreeparms)
        nfloat = int(np.size(self.floating_indices))
        n_constrained = nfloat - nfree
        if trace_AinvH is None:
            trace_AinvH = float(nfloat)

        shrink, rho = (np.asarray(v) for v in self._mcstat_gof_bin_inputs())
        a, b = self.n_half_folds
        lam = shrink * (1.0 + rho)
        mean1 = float(np.sum(lam) + n_constrained - trace_AinvH)
        sum_lam = float(np.sum(lam)) + n_constrained
        sum_lam2 = float(np.sum(lam**2)) + n_constrained
        c = sum_lam2 / sum_lam
        nu = mean1 / c
        g_mean = -float(np.sum(shrink * rho))
        g_var = float(
            np.sum(shrink**2 * (2 * rho**2 + (1 + rho) * rho * (a - b) ** 2 / (a * b)))
        )
        g_sigma = np.sqrt(g_var)

        if g_sigma > 0:
            # P(c chi2_nu + G >= q) by integrating over G
            u = np.linspace(-9.0, 9.0, 2001)
            w = _norm.pdf(u)
            sf = _chi2.sf(np.maximum((q - g_mean - g_sigma * u) / c, 0.0), nu)
            pvalue = float(np.sum(w * sf) / np.sum(w))
        else:
            pvalue = float(_chi2.sf((q - g_mean) / c, nu))
        return dict(
            q=float(q),
            mean=mean1 + g_mean,
            var=2 * c * mean1 + g_var,
            c=c,
            nu=nu,
            g_mean=g_mean,
            g_sigma=float(g_sigma),
            pvalue=pvalue,
        )

    # --- adjusted profile-likelihood statistics for scans -----------------

    def set_mcstat_scan_covariances(self, indices, curvature, reported):
        """Record the curvature (A^-1) and reported (sandwich) covariance blocks
        of the parameters `indices`, for mcstat_scan_adjustment."""
        self._mcstat_scan_cov = dict(
            index={int(i): k for k, i in enumerate(np.asarray(indices))},
            curvature=np.asarray(curvature),
            reported=np.asarray(reported),
        )

    def mcstat_scan_needs_adjustment(self):
        return (
            self.mcstat_debias_active
            and getattr(self, "_mcstat_scan_cov", None) is not None
            and not np.allclose(
                self._mcstat_scan_cov["curvature"], self._mcstat_scan_cov["reported"]
            )
        )

    def mcstat_scan_adjustment(self, param_indices, deltas):
        """Factor turning the profiled statistic W(psi) = 2 Delta L of a
        de-biased objective into the adjusted statistic W_adj = W * factor.

        The de-biased objective is not a likelihood: its curvature is the
        de-biased A, whose inverse under-covers, while the reported covariance
        is the sandwich. Following the adjusted composite-likelihood ratio
        statistic (Pace, Salvan & Sartori 2011), along the direction
        d = psi - psi_hat

            factor(d) = (d^T Sigma_psi^-1 d) / (d^T C_psi^-1 d),

        with C_psi, Sigma_psi the curvature and sandwich blocks of the scanned
        parameters. W_adj then agrees with the sandwich Wald statistic to
        quadratic order (asymptotically chi2 with dim(psi) degrees of freedom)
        while keeping the shape of the profile along each ray. For one
        parameter the factor is the constant C_psi / Sigma_psi.

        deltas: [npoints, dim(psi)] offsets from the best fit. Returns
        [npoints] factors (1 at d = 0).
        """
        info = self._mcstat_scan_cov
        try:
            pos = [info["index"][int(i)] for i in param_indices]
        except KeyError:
            raise RuntimeError(
                "The adjusted scan needs the covariance of the scanned "
                "parameters, which was not computed (under --noHessian only the "
                "POIs and NOIs are)."
            )
        C = info["curvature"][np.ix_(pos, pos)]
        S = info["reported"][np.ix_(pos, pos)]
        d = np.atleast_2d(np.asarray(deltas, dtype=np.float64))
        num = np.einsum("ni,ij,nj->n", d, np.linalg.inv(S), d)
        den = np.einsum("ni,ij,nj->n", d, np.linalg.inv(C), d)
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0)

    def cov_dataprop_sandwich(self, Ainv):
        """Data-propagated Var(score) sandwich (Huber-White / delta-method).

        Propagates the per-bin DATA variance and the per-(bin,proc) TEMPLATE
        variance (sumw2 = the MC-stat variance) through the de-biased estimator
        theta_hat(nobs, norm) by implicit differentiation:

            Sigma = dx/dnobs . diag(Var[nobs]) . (dx/dnobs)^T
                  + dx/dnorm . diag(sumw2)     . (dx/dnorm)^T

        with dx/dv = -A^-1 d^2L/dx dv (A = de-biased bread; `Ainv` = A^-1). The
        FIRST term equals the analytic A^-1 H A^-1 (data Fisher meat); the SECOND
        adds the template-noise propagation. This is the conservative route: it
        OVER-covers (~0.81 in the numpy tests, docs/mcstat/RESULTS.md §7d) and
        needs a downward calibration k~0.81, vs the calibration-free
        analytic/BB-lite meat (~0.67).
        Dense only (watches indata.norm); the de-biased POINT (M in the
        minimization) is still required for an unbiased centre (§7d)."""
        if self.indata.sparse:
            raise NotImplementedError(
                "data-propagated sandwich is dense-only (it differentiates the "
                "loss w.r.t. indata.norm)."
            )
        Ainv = tf.cast(Ainv, self.indata.dtype)
        norm = self.indata.norm
        with tf.GradientTape() as t2:
            t2.watch([self.nobs, norm])
            with tf.GradientTape() as t1:
                t1.watch([self.nobs, norm])
                val = self._compute_loss()
            grad = t1.gradient(val, self.x)
        pd2ldxdnobs, pd2ldxdnorm = t2.jacobian(
            grad, [self.nobs, norm], unconnected_gradients="zero"
        )
        dxdnobs = -Ainv @ pd2ldxdnobs  # [npar, nbins]
        dxdnorm = -Ainv @ tf.reshape(
            pd2ldxdnorm, [tf.shape(pd2ldxdnorm)[0], -1]
        )  # [npar, nbinsfull*nproc]
        var_nobs = self.varnobs if self.varnobs is not None else self.nobs
        sumw2_flat = tf.reshape(self.indata.sumw2, [-1])
        return (dxdnobs * var_nobs[None, :]) @ tf.transpose(dxdnobs) + (
            dxdnorm * sumw2_flat[None, :]
        ) @ tf.transpose(dxdnorm)

    def cov_mcstat_sandwich(self, A):
        """Continuous-M robust (sandwich) covariance.

        Sigma = A^-1 H A^-1 with bread A = de-biased curvature (grad^2 of the
        objective WITH the penalty -1/2 theta^T M theta and the bin-by-bin stat
        compensator, whose Hessian is M_BB) and meat H = the score variance, the
        same-sample curvature WITHOUT the penalty plus M_BB, the part of the
        template noise bin-by-bin stat removes from it
        (_compute_mcstat_bb_compensator; zero without bin-by-bin stat or with a
        static normal-additive one). Because the penalty is a frozen quadratic,
        H = A + M exactly, so

            Sigma = A^-1 (A + M) A^-1 = A^-1 + A^-1 M A^-1.

        No second Hessian pass is needed. Frozen support only (the M-penalty
        treats M as constant; valid in the linear-Gaussian regime where
        continuous-M de-biases; docs/mcstat/RABBIT_MCSTAT_DESIGN.md §1c,
        docs/mcstat/RESULTS.md §9a).

        Parameters
        ----------
        A : tf.Tensor, shape [npar, npar]
            The de-biased objective Hessian (e.g. from loss_val_grad_hess()).

        Returns
        -------
        tf.Tensor, shape [npar, npar]
            The sandwich covariance.
        """
        if self.mcstat_M is None:
            raise RuntimeError(
                "cov_mcstat_sandwich requires an mcstat moment term (none found)."
            )
        Ainv = tf.linalg.inv(A)
        return Ainv + Ainv @ self.mcstat_M @ Ainv

    def edmval_cov(self, grad, hess):
        if len(self.frozen_params) > 0:
            # Only keep parameters that were floating in the fit
            subgrad = tf.gather(grad, self.floating_indices, axis=0)
            subhess = tf.gather(hess, self.floating_indices, axis=0)
            subhess = tf.gather(subhess, self.floating_indices, axis=1)
            edmval, cov = edmval_cov(subgrad, subhess)

            # update only the covariance entries for parameters that were floating in the fit
            coords = tf.stack(
                tf.meshgrid(
                    self.floating_indices, self.floating_indices, indexing="ij"
                ),
                axis=-1,
            )
            coords = tf.reshape(coords, [-1, 2])

            updates = tf.reshape(cov, [-1])

            cov = tf.tensor_scatter_nd_update(self.cov, coords, updates)
            return edmval, cov
        else:
            return edmval_cov(grad, hess)

    def edmval_cov_rows_hessfree(
        self, grad, row_indices, rtol=1e-10, maxiter=None, hessp=None
    ):
        """Hessian-free edmval + selected rows of the covariance matrix.

        Used under --noHessian to avoid allocating the dense [npar, npar]
        Hessian. Solves the linear systems

            H v = grad        ->  edmval = 0.5 * grad^T v
            H c_i = e_i       ->  c_i is the i-th column/row of cov

        iteratively via scipy's conjugate gradient, feeding it a
        LinearOperator backed by self.loss_val_grad_hessp. The Hessian
        must be positive-definite; that's the case for a converged NLL
        minimum (including the purely-quadratic --is_linear case).

        Parameters
        ----------
        grad : tf.Tensor or array-like, shape [npar]
            Gradient at the current x, already computed by the caller.
        row_indices : iterable of int
            Parameter indices to compute covariance rows for. Typically
            the POI indices [0, npoi) concatenated with the NOI indices
            (npoi + noiidxs).
        rtol : float
            Relative residual tolerance passed to scipy.sparse.linalg.cg.
        maxiter : int or None
            Maximum CG iterations per solve; None lets scipy choose.
        hessp : callable or None
            Hessian-vector product p -> H p (tf tensors) defining H; default the
            objective's (self.loss_val_grad_hessp). See bread_hessp.

        Returns
        -------
        edmval : float
        cov_rows : np.ndarray, shape [len(row_indices), npar]
            Row i is (H^{-1})[row_indices[i], :]; diag entries give the
            variances for those parameters.
        """
        import scipy.sparse.linalg as _spla

        n = int(self.x.shape[0])
        dtype = np.float64

        if hessp is None:

            def hessp(p):
                return self.loss_val_grad_hessp(p)[2]

        def _hvp_np(p_np):
            return np.asarray(hessp(tf.constant(p_np, dtype=self.x.dtype)))

        op = _spla.LinearOperator((n, n), matvec=_hvp_np, dtype=dtype)

        grad_np = grad.numpy() if hasattr(grad, "numpy") else np.asarray(grad)
        v, info = _spla.cg(op, grad_np, rtol=rtol, atol=0.0, maxiter=maxiter)
        if info != 0:
            raise ValueError(f"CG solver for edmval did not converge (info={info})")
        edmval = 0.5 * float(np.dot(grad_np, v))

        row_indices = np.asarray(list(row_indices), dtype=np.int64)
        cov_rows = np.empty((len(row_indices), n), dtype=dtype)
        for k, i in enumerate(row_indices):
            e = np.zeros(n, dtype=dtype)
            e[int(i)] = 1.0
            c, info = _spla.cg(op, e, rtol=rtol, atol=0.0, maxiter=maxiter)
            if info != 0:
                raise ValueError(
                    f"CG solver for cov row {int(i)} did not converge (info={info})"
                )
            cov_rows[k] = c

        return edmval, cov_rows

    def _resolved_param_impact_groups(self):
        """
        ParamModel impact groups resolved to floating full-x parameter indices.
        """
        groups = getattr(self.param_model, "param_impact_groups", None)
        if not groups:
            return []
        parms = self.parms.astype(str)
        name_to_idx = {p: i for i, p in enumerate(parms)}
        frozen = set(int(i) for i in np.atleast_1d(self.frozen_indices))
        resolved = []
        for label, pnames in groups.items():
            idxs = [
                name_to_idx[p]
                for p in pnames
                if p in name_to_idx and name_to_idx[p] not in frozen
            ]
            if idxs:
                resolved.append((label, np.array(idxs, dtype=np.int32)))
        return resolved

    def _cov_stat_floating(self, hess, nstat):
        """
        Invert the stat sub-Hessian hess[:nstat, :nstat], excluding frozen params.
        """
        hess_stat = hess[:nstat, :nstat]
        if len(self.frozen_params) == 0:
            return tf.linalg.inv(hess_stat)
        stat_float = self.floating_indices[self.floating_indices < nstat]
        sub = tf.gather(tf.gather(hess_stat, stat_float, axis=0), stat_float, axis=1)
        sub_inv = tf.linalg.inv(sub)
        coords = tf.reshape(
            tf.stack(tf.meshgrid(stat_float, stat_float, indexing="ij"), axis=-1),
            [-1, 2],
        )
        return tf.scatter_nd(
            tf.cast(coords, tf.int64),
            tf.reshape(sub_inv, [-1]),
            tf.constant([nstat, nstat], dtype=tf.int64),
        )

    def impacts_parms(self, hess, cov=None, mcstat_extra_var=None):
        # cov: the covariance matrix to DECOMPOSE (per-nuisance + grouped syst
        # impacts). Defaults to self.cov. For a de-biased fit pass the de-biased
        # CURVATURE covariance (A^-1, in the selected --covMode) and `hess` = the
        # matching de-biased curvature, so the whole decomposition (total / syst /
        # stat) is internally consistent; the reported covariance's extra term is
        # then reported as the `mcStatDebias` group, and with bin-by-bin stat as
        # the combined `mcStat` group, via mcstat_extra_var.
        #
        # The profile=False Hessian is taken here, outside the graph, because
        # assembling it from HVPs (--hvpBatch) runs a Python loop.
        hess_no_bbb = None
        if self.bbstat.enabled:
            _, _, hess_no_bbb = self.loss_val_grad_hess(profile=False)
        if cov is None:
            cov = self.cov
        if self.postfit_faces is not None:
            return self._face_held_impacts_parms(
                hess, hess_no_bbb, cov, mcstat_extra_var
            )
        return self._impacts_parms(hess, hess_no_bbb, cov, mcstat_extra_var)

    def _face_held_impacts_parms(self, hess, hess_no_bbb, cov, mcstat_extra_var=None):
        """Traditional impacts with the active constraint faces held.

        Every covariance that enters them is the face-held one (see
        rabbit.constraints.postfit): ``cov`` (default ``self.cov``) already is; the stat-block
        covariances (with and without the bin-by-bin stat) are the inverse of
        the Hessian of the Lagrangian on the stat block along the faces,
        W = H - sum_i mu_i d2c_i (the constrained nuisances frozen); and the
        grouped impacts condition on each group with the faces held. Exactly
        the impacts of a fit in which the faces are eliminated: for a face that
        is a bound on one parameter, those of the fit with that parameter
        frozen on the bound. Evaluated eagerly in numpy (a handful of small
        dense solves), not in the graph.
        """
        A = self.postfit_faces["jacobian"]
        Hc = self.postfit_faces["hess_constraints"]
        nstat = (
            self.param_model.npoi
            + self.param_model.npou
            + self.indata.nsystnoconstraint
        )
        floating = np.asarray(self.floating_indices, dtype=int)
        stat_floating = floating[floating < nstat]

        def cov_stat(h):
            if h is None:
                return None
            W = np.asarray(h)[:nstat, :nstat] - Hc[:nstat, :nstat]
            return tf.constant(
                constraints_postfit.face_held_inverse(W, A[:, :nstat], stat_floating),
                dtype=self.indata.dtype,
            )

        def group_impact(cov, v, idxs):
            return tf.constant(
                constraints_postfit.face_held_group_impact(
                    cov.numpy(), v.numpy(), idxs, A, floating
                ),
                dtype=self.indata.dtype,
            )

        param_groups = self._resolved_param_impact_groups()
        return traditional_impacts.impacts_parms(
            cov,
            cov_stat(hess),
            cov_stat(hess_no_bbb),
            self.param_model.npoi,
            self.indata.noiidxs,
            self.indata.systgroupidxs,
            nmodel_params=self.param_model.npoi + self.param_model.npou,
            param_groupidxs=[idxs for _, idxs in param_groups],
            mcstat_extra_var=mcstat_extra_var,
            group_impact_fn=group_impact,
        )

    @tf.function
    def _impacts_parms(self, hess, hess_no_bbb, cov, mcstat_extra_var=None):

        nstat = (
            self.param_model.npoi
            + self.param_model.npou
            + self.indata.nsystnoconstraint
        )
        cov_stat = self._cov_stat_floating(hess, nstat)

        if hess_no_bbb is not None:
            cov_stat_no_bbb = self._cov_stat_floating(hess_no_bbb, nstat)
        else:
            cov_stat_no_bbb = None

        param_groups = self._resolved_param_impact_groups()
        impacts, impacts_grouped = traditional_impacts.impacts_parms(
            cov,
            cov_stat,
            cov_stat_no_bbb,
            self.param_model.npoi,
            self.indata.noiidxs,
            self.indata.systgroupidxs,
            nmodel_params=self.param_model.npoi + self.param_model.npou,
            param_groupidxs=[idxs for _, idxs in param_groups],
            mcstat_extra_var=mcstat_extra_var,
        )

        return impacts, impacts_grouped

    @tf.function
    def global_impacts_parms(self, cov=None, mcstat_extra_var=None):
        # cov: covariance to decompose (default self.cov). For a de-biased fit
        # pass the de-biased CURVATURE covariance so the global-impact groups
        # (theta0 / nobs / beta0 / syst) decompose it consistently.
        if cov is None:
            cov = self.cov
        param_groupidxs = [idxs for _, idxs in self._resolved_param_impact_groups()]
        return global_impacts.global_impacts_parms(
            self.x,
            self.bbstat.ubeta,
            self.bbstat.beta_shape,
            self._compute_yields_with_beta,
            self._compute_lbeta,
            self._compute_lc,
            self.param_model.npoi,
            self.param_model.nparams,
            self.indata.noiidxs,
            self.indata.systgroupidxs,
            self.bbstat.enabled,
            self.bbstat.binByBinStatMode,
            self.globalImpactsFromJVP,
            cov,
            param_groupidxs=param_groupidxs,
            mcstat_extra_var=mcstat_extra_var,
        )

    @tf.function
    def gaussian_global_impacts_parms(self):
        dxdx0, dxdnobs, dxdbeta0 = self._dxdvars()
        param_groupidxs = [idxs for _, idxs in self._resolved_param_impact_groups()]

        impacts, impacts_grouped = global_impacts.gaussian_global_impacts_parms(
            dxdx0,
            dxdnobs,
            dxdbeta0,
            self.var_x0,
            self.nobs if self.varnobs is None else self.varnobs,
            (
                1.0
                if self.bbstat.binByBinStatType in ["normal-additive"]
                or not self.bbstat.enabled
                else 1.0 / self.bbstat.kstat
            ),
            self.param_model.npoi,
            self.param_model.nparams,
            self.indata.noiidxs,
            self.bbstat.enabled,
            self.bbstat.binByBinStatMode,
            self.bbstat.beta_shape,
            self.indata.systgroupidxs,
            param_groupidxs=param_groupidxs,
            data_cov_inv=self.data_cov_inv,
        )

        return impacts, impacts_grouped

    def asymmetric_nuisance_mask(self, atol=0.0):
        """Boolean mask of length nsyst, True where the nuisance has nonzero
        asymmetric (logkhalfdiff) tensor content."""
        return asym_impacts.asymmetric_nuisance_mask(self.indata, atol=atol)

    def asym_impacts_parms(
        self,
        nll_min=None,
        q=1,
        include=None,
        exclude=None,
        skip_symmetric=False,
        contour_xtol=1e-6,
        contour_gtol=1e-6,
        contour_maxiter=5000,
        hess_mode="exact",
    ):
        """Traditional asymmetric impacts via per-nuisance contour scan.

        All nuisances are scanned by default, including unconstrained ones —
        like the symmetric traditional impacts, the data still gives them a
        finite postfit uncertainty and hence a finite Delta(2NLL)=q contour.
        Use include/exclude to restrict the selection.

        Args:
            nll_min: postfit reduced NLL. Computed from the current fit state
                if None.
            q: contour level (q=1 -> 1 sigma).
            include: optional regex(es) restricting which nuisances to scan.
            exclude: optional regex(es) excluding nuisances from the scan.
            skip_symmetric: optionally skip nuisances whose template content is
                structurally symmetric (logkhalfdiff identically zero). Off by
                default since nonlinear effects can produce asymmetric impacts
                even for symmetric templates.
        """
        if nll_min is None:
            nll_min = float(self.reduced_nll().numpy())

        nsyst = self.indata.nsyst
        syst_names = np.array(self.indata.systs).astype(bytes)

        selected = np.ones(nsyst, dtype=bool)
        if skip_symmetric:
            selected &= self.asymmetric_nuisance_mask()
        if include is not None:
            keep = match_regexp_params(include, syst_names)
            keep_set = set(keep)
            selected &= np.array([n in keep_set for n in syst_names])
        if exclude is not None:
            drop = match_regexp_params(exclude, syst_names)
            drop_set = set(drop)
            selected &= np.array([n not in drop_set for n in syst_names])

        selected_idxs = np.where(selected)[0]
        selected_names = syst_names[selected_idxs]

        logger.info(
            f"asym_impacts_parms: selected {len(selected_idxs)}/{nsyst} nuisances "
            f"(skip_symmetric={skip_symmetric})"
        )

        # freeze-group grouped impacts are computed for the POIs and NOIs
        npoi = self.param_model.npoi
        targets = [
            p.decode() if isinstance(p, bytes) else str(p)
            for p in self.param_model.params[:npoi]
        ] + [
            (
                self.indata.systs[i].decode()
                if isinstance(self.indata.systs[i], bytes)
                else str(self.indata.systs[i])
            )
            for i in self.indata.noiidxs
        ]

        return asym_impacts.asym_impacts_parms(
            self,
            nll_min,
            selected_idxs,
            selected_names,
            targets=targets,
            q=q,
            contour_xtol=contour_xtol,
            contour_gtol=contour_gtol,
            contour_maxiter=contour_maxiter,
            hess_mode=hess_mode,
        )

    def global_asym_impacts_parms(
        self,
        include=None,
        exclude=None,
        sigma=1.0,
        linear_warmstart=False,
    ):
        """Fully likelihood-based asymmetric global impacts.

        For each selected source, shift its constraint center x0[idx] by +/-
        sigma (in units of the prefit constraint width) and re-run the full
        fit. POI/NOI shifts at each sign are the asymmetric global impacts.
        Sources are the constrained nuisances and any priored ParamModel
        params (exposed as <param>_prior, matching the source set of
        gaussian_global_impacts_parms so the two agree in the Gaussian limit).

        Unconstrained nuisances (constraintweight = 0) are always skipped:
        they have no prefit sigma, and their theta0 does not enter the NLL,
        so the shifted refit would reproduce the nominal minimum exactly
        (zero impact at the cost of two full fits).

        Args:
            include: optional regex(es) restricting which sources to scan
                (matched against nuisance names and bare priored-param names).
            exclude: optional regex(es) excluding sources from the scan.
            sigma: shift magnitude in prefit-sigma units.
            linear_warmstart: experimental, see
                global_asym_impacts.global_asym_impacts_parms.
        """
        nparams = self.param_model.nparams
        cw = self.indata.constraintweights.numpy()
        syst_names = np.array(self.indata.systs).astype(bytes)

        # theta0 of an unconstrained nuisance does not enter the NLL: no
        # finite prefit sigma to shift by, and the refit would be a no-op.
        selected = cw > 0
        if include is not None:
            keep_set = set(match_regexp_params(include, syst_names))
            selected &= np.array([n in keep_set for n in syst_names])
        if exclude is not None:
            drop_set = set(match_regexp_params(exclude, syst_names))
            selected &= np.array([n not in drop_set for n in syst_names])

        syst_sel = np.where(selected)[0]
        # nuisance sources, in full-x coordinates (nparams + syst index)
        src_x_idxs = [nparams + int(i) for i in syst_sel]
        src_names = [syst_names[int(i)] for i in syst_sel]
        n_nuis = len(src_x_idxs)

        # priored ParamModel params are sources too, labelled <param>_prior to
        # match the gaussian/likelihood global impacts; same include/exclude,
        # matched against the bare param name. param_prior_idxs are already
        # full-x indices (the param block leads x).
        if self.param_prior_active:
            parms = np.array(self.parms).astype(bytes)
            prior_idxs = self.param_prior_idxs.numpy().astype(int)
            prior_names = parms[prior_idxs]
            psel = np.ones(len(prior_idxs), dtype=bool)
            if include is not None:
                keep_set = set(match_regexp_params(include, prior_names))
                psel &= np.array([n in keep_set for n in prior_names])
            if exclude is not None:
                drop_set = set(match_regexp_params(exclude, prior_names))
                psel &= np.array([n not in drop_set for n in prior_names])
            for p, nm, ok in zip(prior_idxs, prior_names, psel):
                if ok:
                    src_x_idxs.append(int(p))
                    src_names.append(nm + b"_prior")

        # group membership in full-x indices: syst groups + ParamModel impact
        # groups (the module function keeps only the scanned members).
        group_members = {}
        for gname, gidxs in zip(self.indata.systgroups, self.indata.systgroupidxs):
            group_members[gname] = [nparams + int(i) for i in np.asarray(gidxs)]
        for label, idxs in self._resolved_param_impact_groups():
            key = label.encode() if isinstance(label, str) else label
            group_members.setdefault(key, [])
            group_members[key].extend(int(i) for i in idxs)

        logger.info(
            f"global_asym_impacts_parms: selected {len(src_x_idxs)} sources "
            f"({n_nuis} nuisances + {len(src_x_idxs) - n_nuis} priored params; "
            f"unconstrained nuisances always excluded)"
        )

        return global_asym_impacts.global_asym_impacts_parms(
            self,
            src_x_idxs,
            src_names,
            group_members=group_members,
            sigma=sigma,
            linear_warmstart=linear_warmstart,
        )

    def nonprofiled_impacts_parms(self, unconstrained_err=1.0):
        return nonprofiled_impacts.nonprofiled_impacts_parms(
            self.x,
            self.x0,
            self.frozen_indices,
            self.frozen_params,
            self.cw,
            self.indata.systgroups,
            self.indata.systgroupidxs,
            self.minimize,
            self.diagnostics,
            self.loss_val_grad_hess,
            unconstrained_err,
        )

    def _pd2ldbeta2(self, profile=False):
        with tf.GradientTape(watch_accessed_variables=False) as t2:
            t2.watch([self.bbstat.ubeta])
            with tf.GradientTape(watch_accessed_variables=False) as t1:
                t1.watch([self.bbstat.ubeta])
                if profile:
                    val = self._compute_loss(profile=True)
                else:
                    # TODO this principle can probably be generalized to other parts of the code
                    # to further reduce special cases

                    # if not profiling, likelihood doesn't include the data contribution
                    _1, _2, beta = self._compute_yields_with_beta(
                        profile=False, compute_norm=False, full=False
                    )
                    lbeta = self._compute_lbeta(beta)
                    val = lbeta

            pdldbeta = t1.gradient(val, self.bbstat.ubeta)
        if self.covarianceFit and profile:
            pd2ldbeta2_matrix = t2.jacobian(pdldbeta, self.bbstat.ubeta)
            pd2ldbeta2 = tf.linalg.LinearOperatorFullMatrix(
                pd2ldbeta2_matrix, is_self_adjoint=True
            )
        else:
            # pd2ldbeta2 is diagonal, so we can use gradient instead of jacobian
            pd2ldbeta2 = t2.gradient(pdldbeta, self.bbstat.ubeta)
        return pd2ldbeta2

    def _dxdvars(self):
        # Response of the postfit minimum to a unit shift of each constraint
        # center x0, over the whole parameter vector. x0 enters the NLL only
        # through cw * (x - x0)^2, so columns for unconstrained centers (cw = 0)
        # are exactly zero and carried along harmlessly.
        with tf.GradientTape() as t2:
            t2.watch([self.x0, self.nobs, self.bbstat.beta0])
            with tf.GradientTape() as t1:
                t1.watch([self.x0, self.nobs, self.bbstat.beta0])
                val = self._compute_loss()
            grad = t1.gradient(val, self.x)
        pd2ldxdx0, pd2ldxdnobs, pd2ldxdbeta0 = t2.jacobian(
            grad,
            [self.x0, self.nobs, self.bbstat.beta0],
            unconnected_gradients="zero",
        )

        # cov is inverse hesse, thus cov ~ d2xd2l
        dxdx0 = -self.cov @ pd2ldxdx0
        dxdnobs = -self.cov @ pd2ldxdnobs
        dxdbeta0 = -self.cov @ tf.reshape(pd2ldxdbeta0, [pd2ldxdbeta0.shape[0], -1])

        return dxdx0, dxdnobs, dxdbeta0

    def _dndvars(self, fun):
        with tf.GradientTape() as t:
            t.watch([self.x0, self.nobs, self.bbstat.beta0])
            n = fun()
            n_flat = tf.reshape(n, (-1,))

        pdndx, pdndx0, pdndnobs, pdndbeta0 = t.jacobian(
            n_flat,
            [self.x, self.x0, self.nobs, self.bbstat.beta0],
            unconnected_gradients="zero",
        )

        # apply chain rule to take into account correlations with the fit parameters
        dxdx0, dxdnobs, dxdbeta0 = self._dxdvars()

        dndx0 = pdndx0 + pdndx @ dxdx0
        dndnobs = pdndnobs + pdndx @ dxdnobs
        dndbeta0 = tf.reshape(pdndbeta0, [pdndbeta0.shape[0], -1]) + pdndx @ dxdbeta0

        return n, dndx0, dndnobs, dndbeta0

    def _compute_expected(
        self, fun_exp, inclusive=True, profile=False, full=True, need_observables=True
    ):
        if need_observables:
            observables = self._compute_yields(
                inclusive=inclusive, profile=profile, full=full
            )
            expected = fun_exp(self.x, observables)
        else:
            expected = fun_exp(self.x)

        return expected

    def _expected_with_variance(
        self,
        fun_exp,
        compute_cov=False,
        compute_global_impacts=False,
        compute_gaussian_global_impacts=False,
        profile=False,
        inclusive=True,
        full=True,
        need_observables=True,
    ):
        # compute uncertainty on expectation propagating through uncertainty on fit parameters using full covariance matrix
        # FIXME switch back to optimized version at some point?

        def compute_derivatives(dvars):
            with tf.GradientTape(watch_accessed_variables=False) as t:
                t.watch(dvars)
                expected = self._compute_expected(
                    fun_exp,
                    inclusive=inclusive,
                    profile=profile,
                    full=full,
                    need_observables=need_observables,
                )
                expected_flat = tf.reshape(expected, (-1,))
            jacs = t.jacobian(
                expected_flat,
                dvars,
            )
            return expected, *jacs

        if self.bbstat.enabled:
            dvars = [self.x, self.bbstat.ubeta]
            expected, dexpdx, pdexpdbeta = compute_derivatives(dvars)
        else:
            dvars = [self.x]
            expected, dexpdx = compute_derivatives(dvars)
            pdexpdbeta = None

        if compute_cov or compute_global_impacts:
            cov_dexpdx = tf.matmul(self.cov, dexpdx, transpose_b=True)
        # An MC-stat de-biased fit's global impacts decompose its curvature
        # covariance (the reported one, self.cov, is the sandwich) and add the
        # difference as the mcStatDebias / mcStat groups.
        mcstat_impacts = compute_global_impacts and self.mcstat_debias_active
        if mcstat_impacts:
            cov_dexpdx_curv = tf.matmul(self.mcstat_cov_curv, dexpdx, transpose_b=True)
            var_mcstat_extra = tf.einsum(
                "ij,ji->i", dexpdx, cov_dexpdx - cov_dexpdx_curv
            )

        if compute_cov:
            expcov = dexpdx @ cov_dexpdx
        else:
            # matrix free calculation
            expvar_flat = tf.einsum("ij,jk,ik->i", dexpdx, self.cov, dexpdx)
            expcov = None

        if pdexpdbeta is not None:
            pd2ldbeta2 = self._pd2ldbeta2(profile)

            if self.covarianceFit and profile:
                pd2ldbeta2_pdexpdbeta = pd2ldbeta2.solve(pdexpdbeta, adjoint_arg=True)
            else:
                if self.bbstat.binByBinStatType == "normal-additive":
                    pd2ldbeta2_pdexpdbeta = pdexpdbeta / pd2ldbeta2[None, :]
                else:
                    pd2ldbeta2_pdexpdbeta = tf.where(
                        self.bbstat.betamask[None, :],
                        tf.zeros_like(pdexpdbeta),
                        pdexpdbeta / pd2ldbeta2[None, :],
                    )

                # flatten all but first axes
                batch = tf.shape(pdexpdbeta)[0]
                pdexpdbeta = tf.reshape(pdexpdbeta, [batch, -1])
                pd2ldbeta2_pdexpdbeta = tf.transpose(
                    tf.reshape(pd2ldbeta2_pdexpdbeta, [batch, -1])
                )

            if compute_cov:
                expcov += pdexpdbeta @ pd2ldbeta2_pdexpdbeta
            else:
                expvar_flat += tf.einsum("ik,ki->i", pdexpdbeta, pd2ldbeta2_pdexpdbeta)

        if compute_cov:
            expvar_flat = tf.linalg.diag_part(expcov)

        expvar = tf.reshape(expvar_flat, tf.shape(expected))

        param_groupidxs = [idxs for _, idxs in self._resolved_param_impact_groups()]
        if compute_global_impacts:
            impacts, impacts_grouped = global_impacts.global_impacts_obs(
                self.x,
                self.bbstat.ubeta,
                self.bbstat.beta_shape,
                self._compute_yields_with_beta,
                self._compute_lbeta,
                self._compute_lc,
                self.param_model.npoi,
                self.param_model.nparams,
                self.indata.systgroupidxs,
                self.bbstat.enabled,
                self.bbstat.binByBinStatMode,
                self.globalImpactsFromJVP,
                cov_dexpdx_curv if mcstat_impacts else cov_dexpdx,
                (expvar_flat - var_mcstat_extra) if mcstat_impacts else expvar_flat,
                expvar.shape,
                profile,
                param_groupidxs=param_groupidxs,
                pdexpdbeta=pdexpdbeta,
                pd2ldbeta2_pdexpdbeta=(
                    pd2ldbeta2_pdexpdbeta if pdexpdbeta is not None else None
                ),
                prefit_unconstrained_nuisance_uncertainty=(
                    self.prefit_unconstrained_nuisance_uncertainty
                ),
                mcstat_extra_var=var_mcstat_extra if mcstat_impacts else None,
            )
        else:
            impacts = None
            impacts_grouped = None

        if compute_gaussian_global_impacts:

            def fun_n():
                return self._compute_expected(
                    fun_exp,
                    inclusive=inclusive,
                    profile=profile,
                    full=full,
                    need_observables=need_observables,
                )

            _, dndx0, dndnobs, dndbeta0 = self._dndvars(fun_n)
            impacts_gaussian, impacts_gaussian_grouped = (
                global_impacts.gaussian_global_impacts_obs(
                    dndx0,
                    dndnobs,
                    dndbeta0,
                    self.var_x0,
                    self.nobs if self.varnobs is None else self.varnobs,
                    (
                        1.0
                        if self.bbstat.binByBinStatType in ["normal-additive"]
                        or not self.bbstat.enabled
                        else 1.0 / self.bbstat.kstat
                    ),
                    self.bbstat.enabled,
                    self.bbstat.binByBinStatMode,
                    self.bbstat.beta_shape,
                    self.indata.systgroupidxs,
                    self.param_model.nparams,
                    param_groupidxs=param_groupidxs,
                    data_cov_inv=self.data_cov_inv,
                )
            )
        else:
            impacts_gaussian = None
            impacts_gaussian_grouped = None

        return (
            expected,
            expvar,
            expcov,
            impacts,
            impacts_grouped,
            impacts_gaussian,
            impacts_gaussian_grouped,
        )

    def _expected_variations(
        self,
        fun_exp,
        correlations,
        inclusive=True,
        full=True,
        need_observables=True,
    ):
        with tf.GradientTape() as t:
            # note that beta should only be profiled if correlations are taken into account
            expected = self._compute_expected(
                fun_exp,
                inclusive=inclusive,
                profile=correlations,
                full=full,
                need_observables=need_observables,
            )
            expected_flat = tf.reshape(expected, (-1,))
        dexpdx = t.jacobian(expected_flat, self.x)

        if correlations:
            # construct the matrix such that the columns represent
            # the variations associated with profiling a given parameter
            # taking into account its correlations with the other parameters
            dx = self.cov / tf.sqrt(tf.linalg.diag_part(self.cov))[None, :]

            dexp = dexpdx @ dx
        else:
            dexp = dexpdx * tf.sqrt(tf.linalg.diag_part(self.cov))[None, :]

        new_shape = tf.concat([tf.shape(expected), [-1]], axis=0)
        dexp = tf.reshape(dexp, new_shape)

        down = expected[..., None] - dexp
        up = expected[..., None] + dexp

        expvars = tf.stack([down, up], axis=-1)

        return expvars

    def _init_logk_scaled(self):
        """Prepare the param-model scaling of the systematic variations for
        the yield-computation hot path.

        For systematic_type == "log_normal" the multiplicative form
        ``rnorm * exp(θ·logk) * norm`` already carries the param-model
        scaling through to the variation, so nothing is needed and
        self.logk / self.logk_csr alias the indata tensors.

        For systematic_type == "normal" the linearized variation
        ``rnorm * norm + θ·logk`` does not scale with rnorm, so a constant
        rnorm_init = param_model.compute(xparamdefault) has to be absorbed
        into the variation, so that the relative size of an additive
        variation matches the multiplicative case at the linearization
        point.

        rnorm_init carries no systematic index, so it factors out of the
        contraction over systematics:

            sum_s (rnorm_init[b,p] * logk[b,p,s]) * theta[s]
              == rnorm_init[b,p] * sum_s logk[b,p,s] * theta[s]

        We therefore keep logk aliased and apply the factor to the
        [nbins, nproc] contraction *result* instead of to the
        [nbins, nproc, nsyst] input. This is exactly equivalent, and avoids
        allocating a second full-size copy of logk (and, in sparse mode,
        rebuilding the CSR matrix) at construction time. The scaling is
        still a constant, so the hot path remains strictly linear in θ.
        """
        # scaling factors applied to the contraction result; None when the
        # multiplicative form already carries the param-model scaling
        self.rnorm_init = None
        self.rnorm_init_at_norm = None

        self.logk = self.indata.logk
        if self.indata.sparse:
            self.logk_csr = self.indata.logk_csr

        if self.indata.systematic_type != "normal" or self.param_model.nparams == 0:
            return

        # compute() consumes PHYSICAL parameter values: everywhere else it is
        # fed get_poi(), which undoes the storage transform. xparamdefault is
        # in x-space, so for allowNegativeParam=False -- the default -- it
        # holds sqrt(mu), and handing it over raw evaluates the model at
        # sqrt(mu) instead of mu. The factor then comes out at 1/sqrt(mu) of
        # the value the yield computation uses at the very same point, which
        # is the linearization point this is supposed to match.
        #
        # Only visible when the POI default differs from 1 (in practice
        # --expectSignal != 1), since sqrt(1) == 1, which is why every test in
        # the repo bar the ones below runs straight past it.
        xdef = self.param_model.xparamdefault
        if not self.param_model.allowNegativeParam:
            npoi = self.param_model.npoi
            xdef = tf.concat([tf.square(xdef[:npoi]), xdef[npoi:]], axis=0)

        # Multi-device only: tf.broadcast_to is not lazy, so this is a real
        # [nbinsfull, nproc] allocation, made in __init__ before any shard
        # exists. Unpinned it lands on the default device, _build_shards then
        # slices it under /CPU:0 -- a device-to-host copy of the whole tensor --
        # and it stays resident on that one device for the life of the fit.
        # Smaller than logk by a factor nsyst, but the same failure mode, and it
        # breaks the invariant FitInputData(host_memory=True) and the host-side
        # slicing in _build_shards exist to hold. n_devices is set before
        # super().__init__ precisely so the base constructor can read it; the
        # getattr covers the plain Fitter, which has no such attribute.
        # Single-device keeps the default placement and nothing is duplicated.
        device = "/CPU:0" if int(getattr(self, "n_devices", 1)) > 1 else None
        with tf.device(device):
            rnorm_init = self.param_model.compute(xdef, full=True)
            rnorm_init = tf.broadcast_to(
                rnorm_init, [self.indata.nbinsfull, self.indata.nproc]
            )

            if self.indata.sparse:
                # The CSR contraction returns one value per non-zero of norm, so
                # the factor is gathered onto the same [norm_nnz] layout. This is
                # much smaller than the per-logk-entry scaling it replaces.
                self.rnorm_init_at_norm = tf.gather_nd(
                    rnorm_init, self.indata.norm.indices
                )
            else:
                # Dense: logsnorm is [nbins, nproc], matching rnorm_init directly
                # (sliced to nbins when masked channels are excluded).
                self.rnorm_init = rnorm_init

    def _compute_yields_noBBB(self, full=True, compute_norm=True):
        # full: compute yields inclduing masked channels
        # compute_norm: also build the dense [nbins, nproc] normcentral tensor.
        # In sparse mode this is expensive (forward + backward) and is only
        # needed when an external caller requests per-process yields, or for
        # binByBinStat in "full" mode. The default is True for backward
        # compatibility; the NLL/grad/HVP path passes compute_norm=False.
        nexpcentral, normcentral = self._compute_yields_noBBB_batched(
            self.x[:, None], full=full, compute_norm=compute_norm
        )
        if normcentral is not None:
            normcentral = normcentral[..., 0]
        return nexpcentral[:, 0], normcentral

    def _compute_yields_noBBB_batched(self, x, full=True, compute_norm=True):
        """Yields without bin-by-bin stat at the B columns of x [npar, B]:
        nexpcentral [nbins, B] and, with compute_norm, normcentral
        [nbins, nproc, B] (else None). The batch axis goes last, so the logk
        contraction is the same [.., nsyst] x [nsyst, B] product for B = 1
        and reads logk once for all B columns otherwise."""
        poi = self.get_poi(x)
        model_nui = self.get_model_nui(x)
        theta = self.get_theta(x)

        all_params = tf.concat([poi, model_nui], axis=0)
        rnorm = param_model_compute_batched(self.param_model, all_params, full)

        normcentral = None
        # [nsyst, B], or [2 * nsyst, B] with the asymmetric interpolation
        mthetaalpha = self._mthetaalpha(theta)

        if self.indata.sparse:
            # Inner contraction logk · mthetaalpha via tf.linalg.sparse's
            # CSR matmul. ~8x faster per call than gather + segment_sum
            # because SparseMatrixMatMul dispatches to a hand-tuned CSR
            # kernel. NOTE: SparseMatrixMatMul has no XLA kernel, so the
            # enclosing loss/grad/HVP tf.functions are built with
            # jit_compile=False in sparse mode (see _make_tf_functions).
            logsnorm = tf_sparse_csr.matmul(self.logk_csr, mthetaalpha)  # [nnz, B]

            # The values of a [nbinsfull, nproc] sparse tensor with the sparsity
            # pattern of self.indata.norm (without the masked channels' entries
            # unless full), which absorb the per-entry syst variation and the
            # per-(bin, proc) POI scaling rnorm.
            indices = self.indata.norm.indices
            normvalues = self.indata.norm.values[:, None]
            rnorm_init_at_norm = self.rnorm_init_at_norm
            nbins_out = int(self.indata.norm.dense_shape[0])
            if not full and self.indata.nbinsmasked:
                keep = indices[:, 0] < self.indata.nbins
                indices = tf.boolean_mask(indices, keep)
                normvalues = tf.boolean_mask(normvalues, keep)
                logsnorm = tf.boolean_mask(logsnorm, keep)
                if rnorm_init_at_norm is not None:
                    rnorm_init_at_norm = tf.boolean_mask(rnorm_init_at_norm, keep)
                nbins_out = self.indata.nbins

            # rnorm is [nbins or 1, nproc or 1, B]: its value at each entry,
            # broadcasting a singleton bin or process axis
            if rnorm.shape[0] == 1 and rnorm.shape[1] == 1:
                rnorm_at_norm = rnorm[0]
            elif rnorm.shape[0] == 1:
                rnorm_at_norm = tf.gather(rnorm[0], indices[:, 1])
            elif rnorm.shape[1] == 1:
                rnorm_at_norm = tf.gather(rnorm[:, 0], indices[:, 0])
            else:
                rnorm_at_norm = tf.gather_nd(rnorm, indices)
            if self.indata.systematic_type == "log_normal":
                # values[i] = norm[i] * exp(logsnorm[i]) * rnorm[bin, proc]
                values = tf.exp(logsnorm) * normvalues * rnorm_at_norm
            else:  # "normal"
                # values[i] = norm[i] * rnorm[bin, proc] + logsnorm[i]
                # rnorm_init factors out of the contraction (see
                # _init_logk_scaled), so it is applied here rather than
                # being baked into logk.
                if rnorm_init_at_norm is not None:
                    logsnorm = logsnorm * rnorm_init_at_norm[:, None]
                values = normvalues * rnorm_at_norm + logsnorm

            # Per-bin yields via unsorted_segment_sum on the sparse values
            # keyed by bin index, the dedicated segment_sum kernel with low
            # per-call overhead. The dense [nbins, nproc] grid is only
            # materialized when an external caller requested per-process
            # yields (compute_norm=True).
            nexpcentral = tf.math.unsorted_segment_sum(
                values, indices[:, 0], num_segments=nbins_out
            )
            if compute_norm:
                nproc = tf.cast(self.indata.nproc, indices.dtype)
                if values.shape[-1] == 1:
                    # B = 1: tf.sparse.to_dense, the fastest densification
                    # (on CPU about 4x faster than a scatter_nd)
                    dense_shape = tf.stack([tf.cast(nbins_out, indices.dtype), nproc])
                    normcentral = tf.sparse.to_dense(
                        tf.SparseTensor(indices, values[:, 0], dense_shape)
                    )[..., None]
                else:
                    # a segment sum over the flat (bin, proc) index, the
                    # kernel of nexpcentral above
                    normcentral = tf.reshape(
                        tf.math.unsorted_segment_sum(
                            values,
                            indices[:, 0] * nproc + indices[:, 1],
                            num_segments=tf.cast(nbins_out, indices.dtype) * nproc,
                        ),
                        [nbins_out, self.indata.nproc, -1],
                    )
        else:
            if full or self.indata.nbinsmasked == 0:
                nbins = self.indata.nbinsfull
                logk = self.logk
                norm = self.indata.norm
            else:
                nbins = self.indata.nbins
                logk = self.logk[:nbins]
                norm = self.indata.norm[:nbins]

            if self.indata.symmetric_tensor:
                mlogk = tf.reshape(
                    logk,
                    [nbins * self.indata.nproc, self.indata.nsyst],
                )
            else:
                mlogk = tf.reshape(
                    logk,
                    [nbins * self.indata.nproc, 2 * self.indata.nsyst],
                )

            logsnorm = tf.matmul(mlogk, mthetaalpha)
            logsnorm = tf.reshape(
                logsnorm, [nbins, self.indata.nproc, tf.shape(mthetaalpha)[-1]]
            )

            if self.indata.systematic_type == "log_normal":
                snorm = tf.exp(logsnorm)
                snormnorm = snorm * norm[..., None]
                normcentral = rnorm * snormnorm
            elif self.indata.systematic_type == "normal":
                # rnorm_init factors out of the contraction (see
                # _init_logk_scaled), so it is applied here rather than
                # being baked into logk.
                if self.rnorm_init is not None:
                    logsnorm = logsnorm * self.rnorm_init[:nbins][..., None]
                normcentral = norm[..., None] * rnorm + logsnorm

            nexpcentral = tf.reduce_sum(normcentral, axis=1)

        return nexpcentral, normcentral

    def _compute_yields_with_beta(self, profile=True, compute_norm=False, full=True):
        nexp, norm, beta = self._compute_yields_with_beta_batched(
            self.x[:, None], profile=profile, compute_norm=compute_norm, full=full
        )
        return tuple(None if t is None else t[..., 0] for t in (nexp, norm, beta))

    def _observed_data(self):
        """The observations (nobs, varnobs, lognobs) as columns [nbins, 1]
        shared by every element of a batch (varnobs None without one)."""
        return tuple(
            None if t is None else tf.convert_to_tensor(t)[:, None]
            for t in (self.nobs, self.varnobs, self.lognobs)
        )

    def _compute_yields_with_beta_batched(
        self, x, data=None, profile=True, compute_norm=False, full=True, beta0=None
    ):
        """_compute_yields_with_beta at the B columns of x [npar, B]: nexp
        [nbins, B], norm [nbins, nproc, B] or None, beta [.., B] or None.

        data: (nobs, varnobs, lognobs), each [nbins, 1] (shared) or [nbins, B]
        (see _observations); default the fitter's observations. beta0: the
        bin-by-bin stat constraint centres, of bbstat.beta_shape (masked
        channels included) plus the batch axis (default the stored ones). Bin-by-bin stat is element-wise in the bins:
        B = 1 calls profile_and_apply directly, larger batches vectorize the
        same call (tfh.map_batch)."""
        # Only materialize the dense [nbins, nproc] normcentral when an
        # external caller requested it, when BBB "full" mode needs per-process
        # yields for the analytic β solution, or when "lite" mode needs to
        # split finite-variance (sumw2>0) and zero-variance (sumw2==0)
        # contributions per bin.
        need_norm = compute_norm or self.bbstat.needs_per_proc_norm()
        nexp, norm = self._compute_yields_noBBB_batched(
            x, full=full, compute_norm=need_norm
        )
        if not self.bbstat.enabled:
            return nexp, norm, None
        nobs, varnobs, lognobs = self._observed_data() if data is None else data
        logbeta0 = None if beta0 is None else self.bbstat.log_beta0(beta0)

        def profile_and_apply(nexp, norm, nobs, varnobs, lognobs, beta0, logbeta0):
            return self.bbstat.profile_and_apply(
                nexp,
                norm,
                nobs,
                varnobs,
                lognobs,
                profile=profile,
                compute_norm=compute_norm,
                full=full,
                beta0=beta0,
                logbeta0=logbeta0,
            )

        return tfh.map_batch(
            profile_and_apply, (nexp, norm, nobs, varnobs, lognobs, beta0, logbeta0)
        )

    @tf.function
    def _profile_beta(self):
        nexp, norm, beta = self._compute_yields_with_beta(full=False)
        self.bbstat.beta.assign(beta)
        if self.mcstat_half_beta is not None and self._fold_debias_active():
            self.mcstat_half_beta.assign(
                tf.stack([b for _, b in self._profiled_half_terms(profile=True)])
            )

    def _compute_yields(self, inclusive=True, profile=True, full=True):
        nexpcentral, normcentral, beta = self._compute_yields_with_beta(
            profile=profile,
            compute_norm=not inclusive,
            full=full,
        )
        if inclusive:
            return nexpcentral
        else:
            return normcentral

    @tf.function
    def expected_with_variance(self, *args, **kwargs):
        return self._expected_with_variance(*args, **kwargs)

    @tf.function
    def expected_variations(self, *args, **kwagrs):
        return self._expected_variations(*args, **kwagrs)

    def _residuals_profiled(
        self,
        fun,
    ):

        def fun_res():
            expected = self._compute_expected(
                fun,
                inclusive=True,
                profile=True,
                full=False,
                need_observables=True,
            )
            observed = fun(None, self.nobs)
            return expected - observed

        residuals, dresdx0, dresdnobs, dresdbeta0 = self._dndvars(fun_res)

        # dresdx0 spans the full parameter vector; var_x0 weights each center
        # by its prefit variance (cw = 0 entries contribute exactly zero).
        res_cov = dresdx0 @ (self.var_x0[:, None] * tf.transpose(dresdx0))

        if self.covarianceFit:
            res_cov_stat = dresdnobs @ tf.linalg.solve(
                self.data_cov_inv, tf.transpose(dresdnobs)
            )
        elif self.varnobs is not None:
            res_cov_stat = dresdnobs @ (self.varnobs[:, None] * tf.transpose(dresdnobs))
        else:
            res_cov_stat = dresdnobs @ (self.nobs[:, None] * tf.transpose(dresdnobs))

        res_cov += res_cov_stat

        if self.bbstat.enabled:
            pd2ldbeta2 = self._pd2ldbeta2(profile=False)

            with tf.GradientTape() as t2:
                t2.watch([self.bbstat.ubeta, self.bbstat.beta0])
                with tf.GradientTape() as t1:
                    t1.watch([self.bbstat.ubeta, self.bbstat.beta0])
                    _1, _2, beta = self._compute_yields_with_beta(
                        profile=False, compute_norm=False, full=False
                    )
                    lbeta = self._compute_lbeta(beta)

                dlbetadbeta = t1.gradient(lbeta, self.bbstat.ubeta)
            pd2lbetadbetadbeta0 = t2.gradient(dlbetadbeta, self.bbstat.beta0)
            var_beta0 = pd2ldbeta2 / pd2lbetadbetadbeta0**2

            if self.bbstat.binByBinStatType in ["gamma", "normal-multiplicative"]:
                var_beta0 = tf.where(
                    self.bbstat.betamask, tf.zeros_like(var_beta0), var_beta0
                )

            res_cov_BBB = dresdbeta0 @ (
                tf.reshape(var_beta0, [-1])[:, None] * tf.transpose(dresdbeta0)
            )
            res_cov += res_cov_BBB

        if self.mcstat_debias_active:
            # The propagation above covers the parameters' response to the data,
            # constraint and beta0 fluctuations, Sigma_prop = sum dx/ds V_s dx/ds^T,
            # but not the template (MC) noise that the de-biased estimator's
            # reported covariance Sigma (the sandwich) carries. And unlike an
            # ordinary fit, whose residual is orthogonal to the fitted directions,
            # the de-biased minimum leaves a residual component along them, which
            # would otherwise read as having zero variance. Add the missing
            # J (Sigma - Sigma_prop) J^T, J = d(expected)/dx.
            with tf.GradientTape() as t:
                res_x = fun_res()
            J = tf.reshape(
                t.jacobian(res_x, self.x, unconnected_gradients="zero"),
                [-1, self.x.shape[0]],
            )
            dxdx0, dxdnobs, dxdbeta0 = self._dxdvars()
            if self.covarianceFit:
                cov_nobs = tf.linalg.inv(self.data_cov_inv)
            else:
                v = self.varnobs if self.varnobs is not None else self.nobs
                cov_nobs = tf.linalg.diag(v)
            sigma_prop = dxdx0 @ (self.var_x0[:, None] * tf.transpose(dxdx0))
            sigma_prop += dxdnobs @ cov_nobs @ tf.transpose(dxdnobs)
            if self.bbstat.enabled:
                sigma_prop += dxdbeta0 @ (
                    tf.reshape(var_beta0, [-1])[:, None] * tf.transpose(dxdbeta0)
                )
            # the missing piece is a covariance (the template-noise part of the
            # parameters' variance): keep its positive semi-definite part
            d = self.cov - sigma_prop
            evals, evecs = tf.linalg.eigh(0.5 * (d + tf.transpose(d)))
            sigma_mc = (evecs * tf.nn.relu(evals)[None, :]) @ tf.transpose(evecs)
            res_cov += J @ sigma_mc @ tf.transpose(J)

        return residuals, res_cov

    def _residuals(self, fun, fun_data):
        data, _0, data_cov = fun_data(self.nobs, self.varnobs, self.data_cov_inv)
        pred, _0, pred_cov, *_ = self._expected_with_variance(
            fun,
            profile=False,
            full=False,
            compute_cov=True,
            inclusive=True,
        )
        residuals = pred - data
        res_cov = pred_cov + data_cov
        return residuals, res_cov

    def _chi2(self, res, res_cov, ndf_reduction=0, pseudo_inverse=False):
        res = tf.reshape(res, (-1, 1))
        ndf = tf.size(res) - ndf_reduction

        if pseudo_inverse:
            # the fallback is for an exactly singular covariance only: a
            # non-finite one (e.g. NaN) still fails loudly, as the solve did
            res_cov = tf.debugging.assert_all_finite(
                res_cov, "chi2: the residual covariance is not finite"
            )
        if ndf_reduction > 0 or pseudo_inverse:
            # covariance matrix is in general non invertible with ndf < n
            # compute chi2 using pseudo inverse
            chi_square_value = tf.transpose(res) @ tf.linalg.pinv(res_cov) @ res
        else:
            chi_square_value = tf.transpose(res) @ tf.linalg.solve(res_cov, res)

        return tf.squeeze(chi_square_value), ndf

    @tf.function
    def chi2(
        self, fun, fun_data=None, ndf_reduction=0, profile=False, pseudo_inverse=False
    ):
        if profile:
            residuals, res_cov = self._residuals_profiled(fun)
        else:
            residuals, res_cov = self._residuals(fun, fun_data)
        return self._chi2(residuals, res_cov, ndf_reduction, pseudo_inverse)

    def expected_events(
        self,
        mapping,
        inclusive=True,
        compute_variance=True,
        compute_cov=False,
        compute_global_impacts=False,
        compute_gaussian_global_impacts=False,
        compute_variations=False,
        correlated_variations=False,
        profile=True,
        compute_chi2=False,
    ):

        if compute_variations and (
            compute_variance
            or compute_cov
            or compute_global_impacts
            or compute_gaussian_global_impacts
        ):
            raise NotImplementedError()

        fun = mapping.compute_flat if inclusive else mapping.compute_flat_per_process

        aux = [None] * 6
        if (
            compute_cov
            or compute_variance
            or compute_global_impacts
            or compute_gaussian_global_impacts
        ):
            out = self.expected_with_variance(
                fun,
                profile=profile,
                compute_cov=compute_cov,
                compute_global_impacts=compute_global_impacts,
                compute_gaussian_global_impacts=compute_gaussian_global_impacts,
                need_observables=mapping.need_observables,
                inclusive=inclusive and not mapping.need_processes,
            )
            exp = out[0]
            aux = [o for o in out[1:]]
        elif compute_variations:
            exp = self.expected_variations(
                fun,
                correlations=correlated_variations,
                inclusive=inclusive and not mapping.need_processes,
                need_observables=mapping.need_observables,
            )
        else:
            exp = self._compute_expected(
                fun,
                inclusive=inclusive and not mapping.need_processes,
                profile=profile,
                need_observables=mapping.need_observables,
            )

        if compute_chi2:
            chi2_args = (mapping.compute_flat, mapping._get_data, mapping.ndf_reduction)
            try:
                chi2val, ndf = self.chi2(*chi2_args, profile=profile)
            except tf.errors.InvalidArgumentError:
                # The residual covariance is exactly singular when the mapped bins obey
                # an exact linear relation (e.g. two projections of one channel), and
                # the solve can then fail depending on rounding: use the pseudo inverse.
                chi2val, ndf = self.chi2(
                    *chi2_args, profile=profile, pseudo_inverse=True
                )
            aux.append(chi2val)
            aux.append(ndf)
        else:
            aux.append(None)
            aux.append(None)

        return exp, aux

    @tf.function
    def expected_yield(self, profile=False, full=False):
        return self._compute_yields(inclusive=True, profile=profile, full=full)

    @tf.function
    def _expected_yield_noBBB(self, full=False):
        res, _ = self._compute_yields_noBBB(full=full, compute_norm=False)
        return res

    @tf.function
    def full_nll(self):
        return self._compute_nll(full_nll=True)

    @tf.function
    def reduced_nll(self):
        return self._compute_nll(full_nll=False)

    def _compute_lc(self, full_nll=False):
        return self._compute_lc_batched(self.x[:, None], full_nll=full_nll)[0]

    def _compute_lc_batched(self, x, x0=None, full_nll=False):
        """Constraint term [B] at the B columns of x [npar, B]; x0 [npar, 1]
        (default self.x0) or [npar, B]."""
        # One constraint term over the full effective parameter vector
        # [poi, model_nui, theta]: the ParamModel block is constrained by the
        # declared priors (cw = 0 -> free) and the nuisance block by
        # indata.constraintweights, all folded into cw / x0.
        cw = self.cw[:, None]
        x0 = self.x0[:, None] if x0 is None else x0
        lc = cw * 0.5 * tf.square(self.get_x(x) - x0)
        if full_nll:
            # normalization factor 0.5*log(2*pi*sigma^2) for constrained
            # entries, with sigma^2 = 1/cw and
            # log(1/sqrt(2*pi)) = -0.9189385332046727
            lc = lc + tf.where(
                cw > 0,
                0.9189385332046727
                - 0.5 * tf.math.log(tf.where(cw > 0, cw, tf.ones_like(cw))),
                tf.zeros_like(lc),
            )

        return tf.reduce_sum(lc, axis=0)

    def _compute_lbeta(self, beta, full_nll=False):
        lbeta = self._compute_lbeta_batched(
            None if beta is None else beta[..., None], full_nll=full_nll
        )
        return None if lbeta is None else lbeta[0]

    def _compute_lbeta_batched(self, beta, full_nll=False, beta0=None):
        """Bin-by-bin stat constraint [B] of beta [.., B] (None when disabled);
        beta0 [.., B] or [.., 1]: the constraint centres (default the stored
        ones)."""
        if not self.bbstat.enabled:
            return None
        logbeta0 = None if beta0 is None else self.bbstat.log_beta0(beta0)
        return tfh.map_batch(
            lambda beta, beta0, logbeta0: self.bbstat.lbeta(
                beta, full_nll=full_nll, beta0=beta0, logbeta0=logbeta0
            ),
            (beta, beta0, logbeta0),
        )

    def _compute_ln(self, nexp, full_nll=False):
        return self._compute_ln_batched(nexp[:, None], full_nll=full_nll)[0]

    def _compute_ln_batched(self, nexp, data=None, full_nll=False):
        """Data term [B] of the yields nexp [nbins, B]. data: (nobs, varnobs,
        lognobs), each [nbins, 1] (shared) or [nbins, B]; default the
        fitter's observations."""
        nobs, varnobs, lognobs = self._observed_data() if data is None else data
        if self.chisqFit:
            ln = 0.5 * tf.reduce_sum((nexp - nobs) ** 2 / varnobs, axis=0)
        elif self.covarianceFit:
            # Solve the system without inverting
            residual = nobs - nexp  # chi2 residual
            ln = 0.5 * tf.reduce_sum(
                residual * tf.matmul(self.data_cov_inv, residual), axis=0
            )
        else:
            nexpsafe = tf.where(nobs == 0.0, tf.constant(1.0, dtype=nexp.dtype), nexp)
            lognexp = tf.math.log(nexpsafe)

            # poisson term
            if full_nll:
                ldatafac = tf.math.lgamma(nobs + 1)
                ln = tf.reduce_sum(-nobs * lognexp + nexp + ldatafac, axis=0)
            else:
                # poisson w/o constant factorial part term and with offset to improve numerical precision
                ln = tf.reduce_sum(-nobs * (lognexp - lognobs) + nexp - nobs, axis=0)
        return ln

    def _compute_nll_components(
        self, x=None, data=None, x0=None, beta0=None, profile=True, full_nll=False
    ):
        """The terms of the negative log-likelihood, each [B] or None, at the B
        columns of x [npar, B]: (ln, lc, lbeta, lpenalty, lext, lmcstat, beta).

        x=None is the fitter's own point (self.x as one column). Per-element
        inputs, each with the batch axis last and of size 1 when all elements
        share it (default: the fitter's own, as for a fit):
        - data: the observed counts nobs [nbins, .], or (nobs, variances) with
          the chi2 data variances; the data term is derived from them as
          set_nobs does (_observations);
        - x0 [npar, .]: the constraint centres;
        - beta0 [.., .]: the bin-by-bin stat constraint centres, of the β shape
          (bbstat.beta_shape: [nbinsfull] in lite mode, [nbinsfull, nproc] in
          full mode, masked channels included).
        A frequentist toy changes all three, a Bayesian toy only data. The
        batch axis goes last throughout, so B = 1 runs the same operations as
        an unbatched evaluation.

        The regularization penalties and the MC-stat de-biasing terms (the
        cross-fit jackknife, _compute_mcstat_penalty) are built from self.x and
        the fitter's own observations, so they are only defined at the
        fitter's own point with its own inputs; elsewhere they are refused.
        """
        if (
            x is not None or data is not None or x0 is not None or beta0 is not None
        ) and (len(self.regularizers) or self._mcstat_terms_in_nll()):
            raise NotImplementedError(
                "The regularization penalties and the MC-stat de-biasing terms "
                "are defined at the fitter's own point and inputs only."
            )
        if x is None:
            x = self.x[:, None]
        if data is not None:
            data = self._observations(
                *(data if isinstance(data, (tuple, list)) else (data,))
            )
        nexpfullcentral, _, beta = self._compute_yields_with_beta_batched(
            x,
            data,
            profile=profile,
            compute_norm=False,
            # only build the full (masked-channel-inclusive) yields if some
            # regularizer actually reads them
            full=self._regularizers_need_observables(),
            beta0=beta0,
        )
        nexp = nexpfullcentral[: self.indata.nbins]

        if self._fold_debias_active():
            # Cross-fit jackknife combination L_cf = w_full L_full - w_A L_A -
            # w_B L_B (= 2 L_full - 1/2 L_A - 1/2 L_B for even k) of the
            # profiled likelihoods. The halves average to the full prediction,
            # so this de-biases the point (gradient = cross-fit score) and the
            # curvature (cross-half Fisher) regardless of nonlinearity; each
            # term carries its own bin-by-bin stat beta and constraint
            # (_jackknife_terms), weighted like its data term. Own point only,
            # B = 1.
            terms = self._jackknife_terms(
                nexp[:, 0], None if beta is None else beta[..., 0], profile=profile
            )
            ln = tf.add_n([c * self._compute_ln(n, full_nll) for n, c, _ in terms])[
                None
            ]
            lbeta = None
            if self.bbstat.enabled:
                lbeta = tf.add_n(
                    [c * self._compute_lbeta(b, full_nll) for _, c, b in terms]
                )[None]
        else:
            ln = self._compute_ln_batched(nexp, data, full_nll)
            lbeta = self._compute_lbeta_batched(beta, full_nll, beta0=beta0)

        # lc (theta priors) is counted once.
        lc = self._compute_lc_batched(x, x0, full_nll)

        # None without regularizers; own point only, B = 1
        lpenalty = None
        if len(self.regularizers):
            lpenalty = self._compute_lpenalty(nexpfullcentral[:, 0])[None]

        lext = external_likelihood.compute_external_nll(
            self.external_terms, x, self.indata.dtype, full_nll=full_nll
        )

        # None unless de-biasing; own point only, B = 1
        lmcstat = self._compute_mcstat_penalty(full_nll=full_nll)
        if lmcstat is not None:
            lmcstat = lmcstat[None]

        return ln, lc, lbeta, lpenalty, lext, lmcstat, beta

    def _compute_lpenalty(self, nexpfullcentral, x=None):
        """Sum of the regularization penalties (None without regularizers);
        `nexpfullcentral` are the profiled yields the regularizers read, at the
        physical parameters `x` (default get_x())."""
        if not len(self.regularizers):
            return None
        if not getattr(self, "_regularizers_armed", False):
            raise RuntimeError(
                "Regularizers not armed against the current parameter layout; "
                "call arm_regularizers() (or defaultassign()) first."
            )
        x = self.get_x() if x is None else x
        penalties = [
            reg.compute_nll_penalty(
                x,
                nexpfullcentral if getattr(reg, "needs_observables", True) else None,
            )
            * tf.exp(2 * self.tau)
            for reg in self.regularizers
        ]
        return tf.add_n(penalties)

    def _compute_external_nll(self, full_nll=False):
        """Sum of external likelihood term contributions.

        Each term contributes ``g_i^T x_sub + 0.5 x_sub^T H_i x_sub + const_i``,
        plus its Gaussian log-normalization when ``full_nll`` is set. The two
        scalars mirror the treatment of the native constraint term in
        :meth:`_compute_lc`: that one is written in centered form, so its
        "value at the prior mean is zero" property is automatic and only the
        log-normalization is gated on ``full_nll``. External terms are stored
        expanded, so the centering constant has to be added back explicitly.
        See :mod:`rabbit.external_likelihood`.
        """
        return external_likelihood.compute_external_nll(
            self.external_terms, self.x, self.indata.dtype, full_nll=full_nll
        )

    def _mcstat_terms_in_nll(self):
        """Whether the NLL carries MC-stat de-biasing terms: the cross-fit
        jackknife (_jackknife_terms) or _compute_mcstat_penalty. Their fold and
        half yields and compensators are built from self.x, so
        _compute_nll_components refuses any point or inputs but the fitter's
        own while they are active."""
        if self._fold_debias_active():
            return True
        if self._mcstat_debias_suspended:
            return False
        return (
            bool(self.mcstat_terms)
            or self.mcstat_bb_vref is not None
            or bool(self.mcstat_debias_active)
        )

    def _compute_nll_batched(
        self, x=None, data=None, x0=None, beta0=None, profile=True, full_nll=False
    ):
        """Negative log-likelihood [B] at the B columns of x [npar, B]: the sum
        of the terms of _compute_nll_components (see there for the inputs).
        _compute_nll is this at the fitter's own point."""
        ln, lc, *rest, _ = self._compute_nll_components(
            x, data=data, x0=x0, beta0=beta0, profile=profile, full_nll=full_nll
        )
        l = ln + lc  # main's order of summation, so B = 1 stays as it is
        for t in rest:
            if t is not None:
                l = l + t
        return l

    def _compute_nll(self, profile=True, full_nll=False):
        return self._compute_nll_batched(profile=profile, full_nll=full_nll)[0]

    def _compute_mcstat_penalty(self, full_nll=False):
        """The MC-stat de-biasing terms outside the data term: the continuous-M
        penalty -1/2 theta^T M theta (the external quadratic term with Hessian
        -M), the bin-by-bin stat compensator (_compute_mcstat_bb_compensator) of
        either method, and the Asimov recentring (set_mcstat_asimov). None if
        there are none, or while the de-biasing is suspended for the warm
        start."""
        if self._mcstat_debias_suspended:
            return None
        terms = []
        if self.mcstat_terms:
            terms.append(
                external_likelihood.compute_external_nll(
                    self.mcstat_terms, self.x, self.indata.dtype, full_nll=full_nll
                )
            )
        if self.mcstat_bb_vref is not None:
            terms.append(self._compute_mcstat_bb_compensator())
        if self.mcstat_debias_active:
            # Asimov recentring (set_mcstat_asimov); zero unless set
            terms.append(
                -tf.reduce_sum(self.mcstat_asimov_g0 * (self.x - self.mcstat_asimov_x0))
            )
        return tf.add_n(terms) if terms else None

    def _compute_loss(self, profile=True):
        return self._compute_nll(profile=profile)

    # Fitter methods that also run per shard: the shard evaluators are
    # duck-typed stand-ins (see rabbit.sharding), and binding the class's own
    # unbound methods onto them shares the yields/NLL mathematics -- both
    # interpolation forms, both systematic types, the BBB machinery -- with
    # the single-device path by construction.
    def _make_tf_functions(self):
        # Build tf.function wrappers at instance construction time so that
        # jit_compile and the HVP autodiff mode can be controlled via fit
        # options without redefining the class. self.jit_compile has
        # already been resolved to a plain bool in __init__ (tri-state
        # "auto"/"on"/"off" collapsed against self.indata.sparse), so
        # this body just reads it.
        jit = self.jit_compile
        # the kept set of the other de-biasing setting is now stale
        self._mcstat_tf_standby = None

        def _loss_val(self):
            return self._compute_loss()

        def _loss_val_grad(self):
            with tf.GradientTape() as t:
                val = self._compute_loss()
            grad = t.gradient(val, self.x)
            return val, grad

        def _loss_val_grad_hessp_fwdrev(self, p):
            p = tf.stop_gradient(p)
            with tf.autodiff.ForwardAccumulator(self.x, p) as acc:
                with tf.GradientTape() as grad_tape:
                    val = self._compute_loss()
                grad = grad_tape.gradient(val, self.x)
            hessp = acc.jvp(grad)
            return val, grad, hessp

        def _loss_val_grad_hessp_revrev(self, p):
            p = tf.stop_gradient(p)
            with tf.GradientTape() as t2:
                with tf.GradientTape() as t1:
                    val = self._compute_loss()
                grad = t1.gradient(val, self.x)
            hessp = t2.gradient(grad, self.x, output_gradients=p)
            return val, grad, hessp

        self.loss_val = tf.function(jit_compile=jit)(
            _loss_val.__get__(self, type(self))
        )
        self.loss_val_grad = tf.function(jit_compile=jit)(
            _loss_val_grad.__get__(self, type(self))
        )
        # NOTE: fwdrev HVP is NOT jit-compiled. tf.autodiff.ForwardAccumulator
        # does not propagate JVPs through XLA-compiled subgraphs (the JVP
        # comes back as zero), regardless of inner/outer placement. The
        # loss/grad and revrev HVP wrappers are unaffected.
        self.loss_val_grad_hessp_fwdrev = tf.function(
            _loss_val_grad_hessp_fwdrev.__get__(self, type(self))
        )
        self.loss_val_grad_hessp_revrev = tf.function(jit_compile=jit)(
            _loss_val_grad_hessp_revrev.__get__(self, type(self))
        )
        # Called eagerly, vectorized_map retraces its pfor graph on every
        # batch, which on a large model costs more than the HVPs themselves.
        # In a tf.function it is traced once per batch width (and profile).
        self._hessp_batch_graph = tf.function(self._hessp_batch)
        # traced Hessians of the auxiliary scalars (_scalar_hessian)
        self._scalar_hessian_fns = {}
        # tf.autodiff.ForwardAccumulator does not support tangent
        # propagation through SparseMatrixMatMul (no JVP rule for the
        # CSR variant), so the fwdrev HVP cannot be used in sparse mode.
        # Fall back to revrev with a warning.
        if self.hvp_method == "fwdrev" and self.indata.sparse:
            logger.warning(
                "fwdrev HVP is not supported in sparse mode "
                "(tf.autodiff.ForwardAccumulator cannot trace through "
                "tf.linalg.sparse's CSR matmul); falling back to revrev."
            )
            self.loss_val_grad_hessp = self.loss_val_grad_hessp_revrev
        elif self.hvp_method == "fwdrev":
            self.loss_val_grad_hessp = self.loss_val_grad_hessp_fwdrev
        else:
            self.loss_val_grad_hessp = self.loss_val_grad_hessp_revrev

    def loss_val_grad_hess(self, profile=True):
        """Value, gradient and dense Hessian.

        With --hvpBatch the Hessian is assembled from batches of that many
        HVPs (see hessian_from_hvps), which bounds the memory by the batch
        instead of the parameter count; otherwise it is one tape.jacobian.
        """
        if not self.hessian_from_hvp:
            return self._loss_val_grad_hess_jacobian(profile=profile)
        val, grad = self._loss_val_grad_profile(profile=profile)
        hess = tf.constant(
            self.hessian_from_hvps(profile=profile), dtype=self.indata.dtype
        )
        return val, grad, hess

    @tf.function
    def _loss_val_grad_hess_jacobian(self, profile=True):
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                val = self._compute_loss(profile=profile)
            grad = t1.gradient(val, self.x)
        hess = t2.jacobian(grad, self.x)
        return val, grad, hess

    @tf.function
    def _loss_val_grad_profile(self, profile=True):
        with tf.GradientTape() as t:
            val = self._compute_loss(profile=profile)
        grad = t.gradient(val, self.x)
        return val, grad

    @tf.function
    def loss_val_valfull_grad_hess(self, profile=True):
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                val, valfull = self._compute_nll(profile=profile)
            grad = t1.gradient(val, self.x)
        hess = t2.jacobian(grad, self.x)

        return val, valfull, grad, hess

    @tf.function
    def loss_val_grad_hess_beta(self, profile=True):
        with tf.GradientTape() as t2:
            t2.watch(self.bbstat.ubeta)
            with tf.GradientTape() as t1:
                t1.watch(self.bbstat.ubeta)
                val = self._compute_loss(profile=profile)
            grad = t1.gradient(val, self.bbstat.ubeta)
        hess = t2.jacobian(grad, self.bbstat.ubeta)

        grad = tf.reshape(grad, [-1])
        hess = tf.reshape(hess, [grad.shape[0], grad.shape[0]])

        betamask = ~tf.reshape(self.bbstat.betamask, [-1])
        grad = grad[betamask]
        hess = tf.boolean_mask(hess, betamask, axis=0)
        hess = tf.boolean_mask(hess, betamask, axis=1)

        return val, grad, hess

    def _reference_matrix(self):
        """Reference matrix the preconditioner is built from, as a numpy array.

        "hessian" is the exact Hessian at the current point, and is the default.
        On real data it is not guaranteed positive definite -- the term
        (1 - nobs/nexp) multiplying the second derivative of the prediction can
        go either way -- so the Cholesky needs a ridge big enough to make it so.
        That ridge is not a wart: it is what regularises the negative-curvature
        directions into small positive ones, which is exactly what lets the
        block be whitened into something the trust region can work with.

        "gaussnewton" is the Fisher information, i.e. the *expected* Hessian,
        obtained by evaluating the exact Hessian with the data replaced by the
        current prediction: at that point (1 - nobs/nexp) vanishes, the
        second-derivative term drops out and J^T W J + diag(cw) remains, which
        is positive semi-definite by construction and so factorises at the
        default ridge.

        That PSD-ness is a double-edged sword, and measurement says the edge
        usually points the wrong way. Being PSD, the Fisher matrix *cannot
        represent negative curvature at all*. Measured on a 2112-parameter
        in-situ efficiency block whose exact Hessian had 33 negative
        eigenvalues at the starting point: whitening with the Gauss-Newton
        transform left all 33 directions negative and the true Hessian at
        kappa ~1e4 in the new coordinates, and the fit froze immediately,
        while the ridged exact Hessian took the same fit from 145 min to
        4.6 min. Raising the ridge on the Gauss-Newton matrix did not help,
        so this is about the missing negative curvature, not about scaling.

        So prefer "hessian" unless the exact Hessian is positive definite where
        the fit starts -- near a minimum, or for a genuinely convex model --
        where "gaussnewton" is the cheaper and better-conditioned choice.
        """
        if self.precondition_from == "gaussnewton":
            saved_nobs = tf.identity(self.nobs)
            saved_varnobs = tf.identity(self.varnobs) if self.chisqFit else None
            try:
                self.set_nobs(self.expected_yield(), variances=saved_varnobs)
                _, _, hess = self.loss_val_grad_hess()
                return hess.__array__()
            finally:
                self.set_nobs(saved_nobs, variances=saved_varnobs)

        _, _, hess = self.loss_val_grad_hess()
        return hess.__array__()

    def _hessp_batch(self, P, profile=True):
        """Hessian-vector products for a batch of directions P: [k, npar].

        vectorized_map is a pfor of width k, not the while_loop that
        tape.jacobian's non-pfor path builds -- so XLA compiles it without the
        unrolling that made chunking unusable. Memory scales with k rather than
        npar, which is the whole point: k = 64 costs ~0.4 GB where vectorising
        over all 6538 parameters costs 43 GB.
        """

        def one(p):
            p = tf.stop_gradient(p)
            with tf.GradientTape() as t2:
                with tf.GradientTape() as t1:
                    val = self._compute_loss(profile=profile)
                grad = t1.gradient(val, self.x)
            return t2.gradient(grad, self.x, output_gradients=p)

        return tf.vectorized_map(one, P)

    def hessian_from_hvps(self, block=None, batch=None, profile=True):
        """Dense Hessian assembled column by column from Hessian-vector products.

        tape.jacobian vectorises over every parameter at once, holding one
        [nbins, 9] intermediate each: 43 GB on one device for a 6538-parameter,
        92144-bin model, and several times that in practice -- unallocatable on
        a GPU and enough to exhaust a 187 GB host. Yet the *result* is only
        [npar, npar] (342 MB here): the size is entirely in the intermediates.

        An HVP costs one [npar] vector instead, and in a multi-device fit
        loss_val_grad_hessp is already sharded -- it is what the minimiser calls
        every iteration -- so this distributes over the GPUs for free and needs
        no sharded jacobian of its own. The cost is npar HVPs.

        ``block`` restricts the columns computed, which is what preconditioning
        wants: it only needs the submatrix over the selected parameters.
        """
        n = int(self.x.shape[0])
        idx = np.arange(n) if block is None else np.asarray(block, dtype=np.int64)
        # an explicit --hvpBatch 0 means the sequential loop, like 1
        k = max(1, self.hvp_batch if batch is None else int(batch))
        out = np.empty((n, idx.size), dtype=np.float64)

        # the sequential loop below goes through loss_val_grad_hessp, which
        # only knows the profiled loss
        if k > 1 or not profile:
            return self._hessian_from_hvp_batches(
                lambda P: self._hessp_batch_dispatch(P, profile=profile), idx, k
            )

        v = np.zeros(n, dtype=np.float64)
        for c, j in enumerate(idx):
            v[j] = 1.0
            _, _, hp = self.loss_val_grad_hessp(tf.constant(v, dtype=self.indata.dtype))
            out[:, c] = np.asarray(hp)
            v[j] = 0.0
        return out

    def _hessian_from_hvp_batches(self, hessp_batch, idx, k):
        """Columns `idx` of a dense Hessian from `hessp_batch(P)`, the HVPs of a
        [k', npar] batch of directions, k columns at a time (halving k if the
        device runs out of memory)."""
        n = int(self.x.shape[0])
        out = np.empty((n, idx.size), dtype=np.float64)
        s0 = 0
        while s0 < idx.size:
            cols = idx[s0 : s0 + k]
            basis = np.zeros((cols.size, n), dtype=np.float64)
            basis[np.arange(cols.size), cols] = 1.0
            try:
                hp = hessp_batch(tf.constant(basis, dtype=self.indata.dtype))
            except tf.errors.ResourceExhaustedError:
                # The per-batch memory is k * nbins * 9 * 8 in principle,
                # but the loss graph holds several bins-shaped tensors at
                # once -- measured at ~6.5x that on one model -- so the
                # right k is model dependent. Halve and retry rather than
                # making the caller guess: this runs once per build, so a
                # couple of wasted attempts cost far less than falling back
                # to an unpreconditioned fit or losing the covariance.
                if k == 1:
                    raise
                k = max(1, k // 2)
                # Persist it: the limit is a property of this device and
                # model, not of this call, so the postfit Hessian and every
                # preconditioner rebuild in the restart loop would otherwise
                # re-discover it -- paying the same compile, allocate and
                # unwind each time. min() so an explicit oversized batch=
                # that shrinks cannot raise a lower standing default.
                self.hvp_batch = min(self.hvp_batch, k)
                logger.warning(
                    f"HVP batch too large for the device; retrying with "
                    f"--hvpBatch {k}"
                )
                continue
            out[:, s0 : s0 + cols.size] = np.asarray(hp).T
            s0 += cols.size
        return out

    def _scalar_hessian(self, key, scalar_fn):
        """Dense Hessian of an auxiliary scalar scalar_fn() (the sandwich meat,
        a Gauss-Newton surrogate) w.r.t. x. Like loss_val_grad_hess: one
        tape.jacobian, or with --hvpBatch assembled from batches of HVPs, which
        bounds the memory by the batch. `key` names the scalar; its traced
        functions are built once (and dropped with the others on retrace)."""
        fns = self._scalar_hessian_fns.get(key)
        if fns is None:

            def hess():
                with tf.GradientTape() as t2:
                    with tf.GradientTape() as t1:
                        val = scalar_fn()
                    g = t1.gradient(val, self.x)
                return t2.jacobian(g, self.x)

            def hessp_batch(P):
                def one(p):
                    p = tf.stop_gradient(p)
                    with tf.GradientTape() as t2:
                        with tf.GradientTape() as t1:
                            val = scalar_fn()
                        g = t1.gradient(val, self.x)
                    return t2.gradient(
                        g, self.x, output_gradients=p, unconnected_gradients="zero"
                    )

                return tf.vectorized_map(one, P)

            fns = (tf.function(hess), tf.function(hessp_batch))
            self._scalar_hessian_fns[key] = fns
        if not self.hessian_from_hvp:
            return fns[0]()
        n = int(self.x.shape[0])
        out = self._hessian_from_hvp_batches(
            fns[1], np.arange(n), max(1, self.hvp_batch)
        )
        return tf.constant(out, dtype=self.indata.dtype)

    def _hessp_batch_dispatch(self, P, profile=True):
        """Batched HVP; subclasses override to keep the work distributed."""
        return self._hessp_batch_graph(P, profile=profile)

    def _build_preconditioner(self):
        """Preconditioner for the upcoming :meth:`fit`, or an exact no-op.

        Built at the current parameter values, so the reference Hessian is the
        one the minimizer starts from.
        """
        theta_ref = self.x.numpy()
        if not self.precondition:
            return precond.Preconditioner.identity(theta_ref)

        index_blocks = precond.select_index_blocks(
            self.parms,
            self.cw.numpy(),
            self.frozen_params_mask.numpy(),
            expressions=self.precondition_params,
            match_fn=match_regexp_params,
            groups=self.indata.systgroups,
            group_idxs=self.indata.systgroupidxs,
        )
        # The dense [npar, npar] reference matrix is the one costly part; if it
        # cannot be
        # formed (memory, or a tracing failure on a large model) fall back to an
        # unpreconditioned fit rather than taking the whole job down.
        try:
            _t_ref = time.time()
            hess_np = self._reference_matrix()
            # Time it: this is a FULL Hessian at the current point, and it is
            # the entire cost of a preconditioner rebuild. Whether restarting
            # aggressively is worth it is exactly this number against the
            # remaining iterations, so it should not have to be guessed.
            logger.debug(
                f"Preconditioner reference matrix ({self.precondition_from}) "
                f"took {time.time() - _t_ref:.1f} s"
            )
        except Exception as ex:
            logger.warning(
                f"Could not compute the reference Hessian for preconditioning ({ex}); "
                "running unpreconditioned. If this is an out-of-memory error, "
                "assemble the Hessian from batched HVPs with a (lower) "
                "--hvpBatch."
            )
            return precond.Preconditioner.identity(theta_ref)

        if not index_blocks:
            logger.warning(
                "Preconditioning requested but no parameters were selected; "
                "running unpreconditioned."
            )
            return precond.Preconditioner.identity(theta_ref)

        if self.precondition_blocks in ("auto", "none"):
            # the expressions only set the scope
            scope = np.unique(np.concatenate([idx for _, idx in index_blocks]))
            if self.precondition_blocks == "auto":
                # the blocks come from the reference matrix itself
                index_blocks = precond.auto_blocks(
                    hess_np, scope, threshold=self.precondition_block_threshold
                )
            else:
                # no grouping: one factorisation over the whole scope
                index_blocks = [("all", scope)]

        return precond.Preconditioner.from_hessian(
            hess_np,
            theta_ref,
            index_blocks,
            ridge=self.precondition_ridge,
            transform=self.precondition_transform,
            # names so the per-block log says WHICH parameters each block holds;
            # "block of 14 parameters" alone leaves no way to tell from a log
            # which directions the transform actually helped. astype(str), not
            # str() per element: indata.systs comes out of h5py as an object
            # array of bytes, so str() would render every name as b'...'.
            names=self.parms.astype(str),
        )

    def _use_warm_start(self, warm_start):
        """Whether fit(warm_start) runs the trust-constr warm start."""
        mode = self.minimizer_warm_start
        if (
            mode == "off"
            or (mode == "auto" and not warm_start)
            or self.minimizer_method not in CONSTRAINED_METHODS
            or not len(self.constraints)
        ):
            return False
        ok, why = tc_warm.scipy_supports_slack_override()
        if not ok:
            logger.warning(
                f"[minimize] trust-constr warm start unavailable ({why}); "
                "starting from scipy's default interior-point start instead"
            )
            return False
        return True

    def _constraint_kkt_inputs(self):
        """(grad, cval, J, lower, upper, labels, floating) at the current x."""
        x = self.x.numpy()
        _, grad = self.loss_val_grad()
        cval, J = self.eval_physical(self._constraints_fn, x, jacobian=True)
        lower, upper, labels = self._constraint_bounds()
        floating = (
            np.asarray(self.floating_indices, dtype=int)
            if len(self.frozen_params) > 0
            else None
        )
        return grad.numpy(), cval, J, lower, upper, labels, floating

    def releasing_constraint_rows(self):
        """--constraints rows near a bound whose multiplier, at the current x,
        says the loss wants them released (see
        :func:`rabbit.constraints.warm_start.releasing_rows`)."""
        g, c, J, lo, up, labels, floating = self._constraint_kkt_inputs()
        return tc_warm.releasing_rows(g, c, J, lo, up, labels, floating=floating)

    #: release rounds (release step + warm leg) before the last-resort leg
    WARM_RELEASE_ROUNDS = 3

    def fit(self, warm_start=False):
        """Minimise from the current self.x.

        warm_start: the caller's statement that self.x is (near) the minimum of
        a closely related problem -- a previous fit (--externalPostfit), the
        main fit for the saturated sub-fit, the neighbouring scan point. It
        changes anything only for trust-constr with --constraints, under
        --minimizerWarmStart auto (the default; on/off override the caller).
        The warm leg holds no row feasible, starts the slacks at c - lb and
        the barrier on the start's central path (max multiplier * slack over
        the rows near a bound, rabbit.constraints.warm_start.centred_barrier).

        A small barrier cannot push off a face, and scipy's convergence test
        does not check the multipliers' sign, so a warm leg whose minimum has
        moved off a face stops ON it. The result is therefore checked
        (releasing_constraint_rows). If a face should be left, a release step
        moves off the releasing face(s) along their normals, holding the
        binding ones (a Newton step on the loss along that direction, cut back
        until the loss decreases and no row is violated), and a new warm leg
        starts from there. Only if that fails WARM_RELEASE_ROUNDS times does
        the fit fall back to scipy's own interior-point start (still holding
        no row feasible), which walks away from the point and is slow.
        See rabbit.constraints.warm_start.
        """
        if not self._use_warm_start(warm_start):
            self._warm_start_info = None
            return self._fit_leg("cold")
        info = {"legs": [], "releasing": [], "barrier": None}
        callback = None
        for rnd in range(self.WARM_RELEASE_ROUNDS + 1):
            g, c, J, lo, up, _, floating = self._constraint_kkt_inputs()
            self._warm_barrier = tc_warm.centred_barrier(
                g, c, J, lo, up, floating=floating
            )
            if info["barrier"] is None:
                info["barrier"] = self._warm_barrier
            logger.info(
                "[minimize] trust-constr warm start: barrier from "
                f"{self._warm_barrier:.3g} (the start's central path; "
                "--minimizerInitialBarrier* override it), slacks from the "
                "constraint values, no row held feasible"
            )
            callback = merge_callbacks(callback, self._fit_leg("warm"))
            info["legs"].append("warm")
            bad = self.releasing_constraint_rows()
            if not bad:
                break
            info["releasing"].extend(lab for lab, _, _ in bad)
            logger.warning(
                "[minimize] warm leg stopped on face(s) the loss wants to leave "
                "(multiplier of the releasing sign): "
                + "; ".join(
                    f"{lab} (mu={m:.3g}, distance {d:.2g})" for lab, m, d in bad
                )
            )
            if rnd == self.WARM_RELEASE_ROUNDS or not self._release_step():
                logger.warning(
                    "[minimize] no release step; continuing from scipy's own "
                    "interior-point start (expect a detour)"
                )
                callback = merge_callbacks(callback, self._fit_leg("release"))
                info["legs"].append("release")
                bad = self.releasing_constraint_rows()
                if bad:
                    logger.warning(
                        "[minimize] still on face(s) with a releasing multiplier: "
                        + "; ".join(lab for lab, _, _ in bad)
                    )
                break
            info["legs"].append("release-step")
        self._warm_start_info = info
        return callback

    def _release_step(self, max_halvings=40):
        """Move self.x off the --constraints faces whose multiplier says the
        loss wants them released, holding the binding ones; True if moved.

        The direction is rabbit.constraints.warm_start.release_direction; the
        length a Newton step on the loss along it (one HVP), halved until the
        loss decreases and no row is violated beyond what it was. This only
        restores the slack the interior point cannot grow by itself (a slack
        of ~1e-9 has a scaled gradient ~1e-17); the warm leg that follows
        finishes the job.
        """
        x0 = self.x.numpy()
        g, c, J, lo, up, labels, floating = self._constraint_kkt_inputs()
        rows, side, _, _, bad = tc_warm.classify_near_rows(
            g, c, J, lo, up, floating=floating
        )
        d = tc_warm.release_direction(
            J, rows[bad], side[bad], rows[~bad], floating=floating
        )
        if d is None:
            return False
        slope = float(g @ d)
        if not slope < 0:
            return False
        _, _, hd = self.loss_val_grad_hessp(tf.constant(d, dtype=self.x.dtype))
        curv = float(np.dot(d, hd.numpy()))
        # no positive curvature: cap the move at 1e-3 of the first face's scale
        t = (
            -slope / curv
            if curv > 0
            else 1e-3 / max(abs(float(J[rows[bad][0]] @ d)), 1e-300)
        )

        def violation(cv):
            return float(np.max(np.maximum(np.maximum(lo - cv, cv - up), 0.0)))

        f0 = float(self.loss_val())
        v0 = violation(c)
        for _ in range(max_halvings):
            self.x.assign(x0 + t * d)
            ft = float(self.loss_val())
            ct = self.eval_physical(self._constraints_fn, self.x.numpy())
            if ft < f0 and violation(ct) <= v0:
                logger.info(
                    f"[minimize] release step: {t:.3g} along the release direction, "
                    f"loss {f0:.10f} -> {ft:.10f}"
                )
                return True
            t *= 0.5
        self.x.assign(x0)
        return False

    def _fit_leg(self, tc_start="cold"):
        """One minimisation from self.x. tc_start (trust-constr + --constraints
        only): "cold" = scipy's interior-point start with rabbit's
        keep_feasible rule; "warm" = no row held, slacks at c - lb, centred
        barrier; "release" = no row held, scipy's start (last resort)."""
        logger.info("Perform iterative fit")

        # Optional reparameterisation. Built once at the starting point, it is
        # confined to the three scipy callbacks below: self.x always holds
        # *physical* parameters outside them, so the postfit Hessian,
        # covariance, impacts and pulls are unaffected and need no mapping
        # back. pc is an exact no-op when disabled, keeping one code path.
        # Held in a one-element cell because it is rebuilt at the current point
        # before every restart (see the restart loop) and the scipy callbacks
        # below must pick the new one up.
        pc_cell = [self._build_preconditioner()]

        def scipy_loss(yval):
            pc = pc_cell[0]
            self.x.assign(pc.to_physical(yval))
            val, grad = self.loss_val_grad()
            val = val.__array__()
            if not np.isfinite(val):
                # A trial point where the loss cannot be evaluated (typically a
                # negative expected yield) must read as infinitely bad. scipy's
                # trust-region update compares rho = actual/predicted reduction
                # with thresholds, and every comparison on a NaN rho is False: the
                # radius never shrinks and the same rejected step is proposed
                # until --earlyStopping fires, and a restart re-proposes it. +inf
                # gives rho = -inf, so the step is rejected and the radius
                # shrinks, as in the native minimizers (rabbit.minimizer.base).
                return np.inf, np.zeros_like(yval)
            grad_y = pc.grad_to_internal(grad.__array__())
            return val, grad_y

        def scipy_hessp(yval, pval):
            pc = pc_cell[0]
            self.x.assign(pc.to_physical(yval))

            def hvp(v):
                _, _, hessp = self.loss_val_grad_hessp(tf.convert_to_tensor(v))
                return hessp.__array__()

            return pc.hessp_to_internal(np.asarray(pval, dtype=np.float64), hvp)

        def scipy_hess(yval):
            pc = pc_cell[0]
            self.x.assign(pc.to_physical(yval))
            val, grad, hess = self.loss_val_grad_hess()
            if self.diagnostics:
                cond_number = tfh.cond_number(hess)
                logger.info(f"  - Condition number: {cond_number}")
                edmval = tfh.edmval(grad, hess)
                logger.info(f"  - edmval: {edmval}")
            return pc.hess_to_internal(hess.__array__())

        # Native (TF) minimizer counterparts of the callbacks above. Same
        # contract and the same internal coordinates, but the gradient and
        # Hessian stay tf tensors: with preconditioning off they never leave
        # the device, and the subproblem factorizes there either way.
        def native_loss(yval):
            pc = pc_cell[0]
            self.x.assign(pc.to_physical(yval))
            return float(self.loss_val())

        def native_closure(yval):
            pc = pc_cell[0]
            self.x.assign(pc.to_physical(yval))
            val, grad, hess = self.loss_val_grad_hess()
            if self.diagnostics:
                cond_number = tfh.cond_number(hess)
                logger.info(f"  - Condition number: {cond_number}")
                edmval = tfh.edmval(grad, hess)
                logger.info(f"  - edmval: {edmval}")
            if pc.enabled:
                grad = tf.constant(pc.grad_to_internal(grad.__array__()))
                hess = tf.constant(pc.hess_to_internal(hess.__array__()))
            return float(val), grad, hess

        def native_grad_closure(yval):
            pc = pc_cell[0]
            self.x.assign(pc.to_physical(yval))
            val, grad = self.loss_val_grad()
            if pc.enabled:
                grad = tf.constant(pc.grad_to_internal(grad.__array__()))
            return float(val), grad

        def native_set_point(yval):
            pc = pc_cell[0]
            self.x.assign(pc.to_physical(yval))

        def native_hessp():
            # internal-coordinate HVP, graph-compatible: the pc transform runs
            # inside the compiled CG loop (numpy per CG iteration would defeat
            # the on-device solve). Rebuilt per restart along with pc.
            pc = pc_cell[0]
            transforms = pc.tf_transforms()
            if transforms is None:

                def hessp(v):
                    _, _, hp = self.loss_val_grad_hessp(v)
                    return hp

            else:
                apply_T, apply_TT = transforms

                def hessp(v):
                    _, _, hp = self.loss_val_grad_hessp(apply_T(v))
                    return apply_TT(hp)

            return hessp

        # scipy works in internal coordinates throughout; y = 0 at the point the
        # transform was built.
        xval = pc_cell[0].from_physical(self.x.numpy())

        if len(self.constraints) and self.minimizer_method not in CONSTRAINED_METHODS:
            raise ValueError(
                f"--constraints needs a constrained minimizer "
                f"({', '.join(CONSTRAINED_METHODS)}); --minimizerMethod "
                f"{self.minimizer_method} would silently ignore them."
            )
        if self.minimizer_method in CONSTRAINED_METHODS:
            # Constrained trust region: the same hessp structure as
            # trust-krylov (so 3700 parameters stay tractable), plus the
            # --constraints rows as a NonlinearConstraint. Under
            # --precondition scipy iterates in y (theta = theta_ref + T y):
            # the constraint values only need the map, their Jacobian and
            # HVPs the chain rule (J T, T^T d2c T). scipy's trust-constr has
            # no preconditioner of its own -- its projected-CG subproblem sees
            # whatever Hessian it is given -- so this change of variables is
            # the only way to give it one.
            info_minimize = dict(hessp=scipy_hessp)
            cons = self.build_scipy_constraints(
                pc_cell[0].to_physical(xval),
                transform=lambda: pc_cell[0],
                hold_feasible=tc_start == "cold",
            )
            if cons:
                info_minimize["constraints"] = cons
            v0, _ = self.loss_val_grad()
            logger.info(
                f"[minimize] {self.minimizer_method} start loss = " f"{float(v0):.10f}"
            )
        elif self.minimizer_method in [
            "trust-krylov",
            "trust-ncg",
        ]:
            info_minimize = dict(hessp=scipy_hessp)
        elif self.minimizer_method in [
            "trust-exact",
            "dogleg",
        ]:
            info_minimize = dict(hess=scipy_hess)
        else:
            info_minimize = dict()

        # Build scipy.optimize.minimize options from --minimizerMaxiter/
        # --minimizerGtol/--minimizerFtol. Anything not set explicitly falls
        # back to the historical tol=0.0 baseline below (run to the tightest
        # internal criteria), since `tol` only fills options keys that are not
        # already present. Methods that don't recognize an option ignore it
        # with an OptimizeWarning (no crash).
        sci_opts = {}
        if self.minimizer_maxiter is not None:
            sci_opts["maxiter"] = int(self.minimizer_maxiter)
        if self.minimizer_gtol is not None:
            sci_opts["gtol"] = float(self.minimizer_gtol)
        if self.minimizer_ftol is not None:
            sci_opts["ftol"] = float(self.minimizer_ftol)
        if self.minimizer_method in CONSTRAINED_METHODS:
            if self.minimizer_xtol is not None:
                sci_opts["xtol"] = float(self.minimizer_xtol)
            if self.minimizer_barrier_tol is not None:
                sci_opts["barrier_tol"] = float(self.minimizer_barrier_tol)
            for k, v in self.minimizer_tc_init.items():
                if v is not None:
                    sci_opts[k] = float(v)
            if tc_start == "warm":
                # explicit --minimizerInitialBarrier* still win
                for k in ("initial_barrier_parameter", "initial_barrier_tolerance"):
                    sci_opts.setdefault(k, self._warm_barrier)
            if "xtol" not in sci_opts:
                logger.warning(
                    "[minimize] trust-constr without --minimizerXtol: a "
                    "collapsed trust radius cannot end a barrier subproblem, "
                    "so the barrier may stall and leave the fit ~barrier "
                    "parameter above the constrained minimum."
                )
        elif (
            self.minimizer_xtol is not None
            or self.minimizer_barrier_tol is not None
            or any(v is not None for v in self.minimizer_tc_init.values())
        ):
            logger.warning(
                "--minimizerXtol/--minimizerBarrierTol/--minimizerInitialBarrier/"
                "--minimizerInitialBarrierTol/--minimizerInitialTrRadius are "
                "trust-constr only; ignored"
            )
        logger.info(f"[minimize] method={self.minimizer_method} options={sci_opts}")

        # The native minimizers take gtol and maxiter and nothing else, so any
        # other key here is silently dropped -- scipy at least raises an
        # OptimizeWarning for an option its method does not recognize, and
        # without this the user sees the option echoed in the line above and
        # then has no signal that it did nothing. --minimizerFtol is the one
        # that reaches this today.
        if self.minimizer_method in NATIVE_MINIMIZER_OPTIONS:
            ignored = sorted(
                set(sci_opts) - NATIVE_MINIMIZER_OPTIONS[self.minimizer_method]
            )
            if ignored:
                logger.warning(
                    f"{self.minimizer_method} does not implement "
                    f"{', '.join(ignored)}; ignoring "
                    f"{', '.join(f'--minimizer{o.capitalize()}' for o in ignored)}. "
                    "Use a scipy --minimizerMethod if you need it."
                )

        # Restart loop. scipy's trust-region methods shrink the trust radius by
        # 4x on every rejected step with no lower bound, and the radius is a
        # local of scipy's loop -- so once it has collapsed the method takes
        # infinitesimal steps and the loss stops moving, far from any minimum.
        # A fresh minimize() call resets the radius to initial_trust_radius,
        # which is why restarting from a stalled point resumes progress. Loop
        # that here instead of making the caller chain --externalPostfit by
        # hand. Only an early-stopping stall is retried, and only while the
        # restarts keep buying loss.
        callback = None
        prev_loss = None
        attempt = 0

        # to_physical is the whole reason this is built here rather than in the
        # callback: under preconditioning the minimiser's iterate is in internal
        # coordinates, and a snapshot of those would load without complaint and
        # be wrong. pc_cell is read at save time, not captured, so a rebuilt
        # transform is picked up.
        snapshotter = Snapshotter(
            self.snapshot_file,
            self.parms,
            to_physical=lambda v: pc_cell[0].to_physical(v),
            interval_hours=self.snapshot_interval,
        )
        snapshotter.update(xval)

        with snapshot_on_signal(snapshotter):
            while True:
                cb = FitterCallback(
                    xval,
                    self.earlyStopping,
                    snapshotter=snapshotter,
                    stall_rel_tol=self.stallRelTol,
                )
                try:
                    if self.minimizer_method == "tf-trust-exact":
                        res = minimize_trust_exact(
                            native_loss,
                            native_closure,
                            xval,
                            gtol=sci_opts.get("gtol", 0.0),
                            maxiter=sci_opts.get("maxiter"),
                            callback=cb,
                        )
                    elif self.minimizer_method == "tf-trust-ncg":
                        res = minimize_trust_ncg(
                            native_loss,
                            native_grad_closure,
                            native_hessp(),
                            native_set_point,
                            xval,
                            gtol=sci_opts.get("gtol", 0.0),
                            maxiter=sci_opts.get("maxiter"),
                            callback=cb,
                        )
                    elif self.minimizer_method == "tf-trust-krylov":
                        res = minimize_trust_krylov(
                            native_loss,
                            native_grad_closure,
                            native_hessp(),
                            native_set_point,
                            xval,
                            gtol=sci_opts.get("gtol", 0.0),
                            maxiter=sci_opts.get("maxiter"),
                            callback=cb,
                        )
                    else:
                        with (
                            tc_warm.consistent_slacks(
                                tc_warm.WARM_SLACK_FLOOR_REL
                                * sci_opts["initial_barrier_parameter"]
                            )
                            if tc_start == "warm" and "constraints" in info_minimize
                            else contextlib.nullcontext()
                        ):
                            res = scipy.optimize.minimize(
                                scipy_loss,
                                xval,
                                method=self.minimizer_method,
                                jac=True,
                                tol=0.0,
                                callback=cb,
                                options=sci_opts,
                                **info_minimize,
                            )
                except Exception as ex:
                    # minimizer could have called the loss or hessp functions with "random" values, so restore the
                    # state from the end of the last iteration before the exception
                    xval = cb.xval
                    # An early stop throws scipy's result away; for a
                    # constrained fit keep the KKT measures of the last
                    # accepted iterate (the one xval holds) instead.
                    self.minimizer_result = getattr(cb, "last_state", None)
                    if not cb.stopped_early:
                        # a real failure, not a stall: surface it rather than
                        # letting a broken callback look like a converged fit
                        logger.warning(f"Minimizer raised: {ex}")
                        # the expensive case: a failure here used to discard the
                        # whole fit, however close to the minimum it had got
                        snapshotter.save(
                            cb.xval, "minimizer-failed", error=str(ex)[:200]
                        )
                    logger.debug(ex)
                else:
                    xval = res["x"]
                    self.minimizer_result = res
                    logger.debug(res)

                callback = merge_callbacks(callback, cb)
                last_loss = cb.loss_history[-1] if cb.loss_history else None

                if not cb.stopped_early:
                    break
                # The only reason to stop restarting is that the last restart
                # bought nothing: a round can never end above where it started
                # (the trust region accepts improving steps only), so "not below
                # the previous round" means the descent is genuinely exhausted.
                if (
                    prev_loss is not None
                    and last_loss is not None
                    and last_loss
                    >= prev_loss - RESTART_MIN_IMPROVEMENT * max(1.0, abs(prev_loss))
                ):
                    logger.info(
                        f"Restart did not reduce the loss further ({prev_loss} -> "
                        f"{last_loss}); stopping after {attempt} restart(s)."
                    )
                    break
                if 0 <= self.max_restarts <= attempt:
                    logger.warning(
                        f"Minimizer still stalling at loss {last_loss} after "
                        f"{self.max_restarts} restart(s) and the loss was still "
                        "coming down; raise --maxRestarts to let it continue."
                    )
                    break
                attempt += 1
                logger.info(
                    f"Minimizer stalled at loss {last_loss}; restarting "
                    f"(#{attempt}) to reset the trust radius."
                )
                prev_loss = last_loss

                # Rebuild the transform at the point we are restarting from. The
                # one built at the start whitens the Hessian *there*; by the time
                # the fit has stalled somewhere else that Hessian has changed and
                # the transform no longer conditions anything. Refreshing costs one
                # Hessian evaluation and is a no-op when preconditioning is off.
                self.x.assign(pc_cell[0].to_physical(xval))
                pc_cell[0] = self._build_preconditioner()
                xval = pc_cell[0].from_physical(self.x.numpy())

        # xval (and callback.xval) are internal coordinates; everything outside
        # fit() expects physical parameters.
        self.x.assign(pc_cell[0].to_physical(xval))
        self._fit_preconditioned = bool(pc_cell[0].enabled)

        # The minimum itself. Writing it costs nothing next to the fit and
        # covers the stretch that is otherwise unprotected: everything between
        # here and the output file being closed, which on a large model means
        # the Hessian, the impacts and the postfit histograms.
        snapshotter.save(xval, "converged")

        return callback

    def minimizer_status(self):
        """Convergence outcome of the last :meth:`fit`, or None if none ran.

        NB ``success`` is scipy's flag, not a convergence test on its own:
        BFGS reports False ("precision loss") at points with EDM ~1e-17.
        """
        res = self.minimizer_result
        if res is None:
            return None
        out = {
            "success": bool(getattr(res, "success", False)),
            "status": int(getattr(res, "status", -1)),
            "nit": int(getattr(res, "nit", -1)),
            "nfev": int(getattr(res, "nfev", -1)),
            "message": str(getattr(res, "message", "")),
        }
        # Constrained runs: at an active constraint grad f is balanced by the
        # constraint forces, so EDM says nothing. scipy's own test is
        #     optimality < gtol and constr_violation < gtol  ->  status 1
        # with optimality = ||grad f + J^T v||_inf (KKT residual, scipy's
        # Lagrangian gradient; measured in the coordinates scipy iterated, see
        # optimality_frame below) and constr_violation = max(0, lb - c). `v`
        # holds the Lagrange multipliers (strongly vs weakly active constraints).
        for k in ("optimality", "constr_violation", "barrier_parameter", "tr_radius"):
            v = getattr(res, k, None)
            if v is not None:
                out[k] = float(v)
        mult = getattr(res, "v", None)
        if mult is not None:
            try:
                out["multipliers"] = [
                    float(m) for m in np.concatenate([np.atleast_1d(m) for m in mult])
                ]
            except Exception:
                pass
        info = self._constraint_info
        if "optimality" in out:
            # scipy measures the KKT residual in the coordinates it iterated:
            # under --precondition that is T^T (grad f + J^T v), a different
            # norm (whitened: roughly per-sigma) from the fitter's own units.
            # The multipliers v are the same in both frames, so the residual is
            # recomputed in the fitter coordinate for a frame-free figure.
            preconditioned = getattr(self, "_fit_preconditioned", False)
            out["optimality_frame"] = "preconditioned" if preconditioned else "physical"
            if (
                preconditioned
                and "multipliers" in out
                and info is not None
                and "labels" in info
            ):
                _, grad = self.loss_val_grad()
                _, jac = self.eval_physical(
                    self._constraints_fn, self.x.numpy(), jacobian=True
                )
                lag = np.asarray(grad, dtype=np.float64) + jac[info["live"]].T @ (
                    np.asarray(out["multipliers"], dtype=np.float64)
                )
                out["optimality_physical"] = float(np.max(np.abs(lag)))
        if len(self.constraints) and info is not None and "labels" in info:
            vals = self.eval_physical(self._constraints_fn, self.x.numpy())
            out["constraint_values"] = [float(c) for c in vals[info["live"]]]
            out["constraint_lower_bounds"] = [float(c) for c in info["lower"]]
            out["constraint_upper_bounds"] = [float(c) for c in info["upper"]]
            out["constraint_labels"] = list(info["labels"])
        if self._warm_start_info is not None:
            out["warm_start_legs"] = list(self._warm_start_info["legs"])
            out["warm_start_releasing"] = list(self._warm_start_info["releasing"])
            out["warm_start_barrier"] = float(self._warm_start_info["barrier"])
        return out

    def minimize(self, warm_start=False, mcstat_warm_start=True):
        """Minimize the objective.

        warm_start: see :meth:`fit`; with the MC-stat warm start below it
        applies to both legs.

        With MC-stat de-biasing active, the ordinary likelihood (the
        --mcStatDebias none objective) is minimized first and the de-biased fit
        starts from its minimum (mcstat_warm_start). The de-biased objectives are
        only locally convex: the Poisson jackknife contains +w y ln n_A, which
        goes to -infinity as a half prediction n_A -> 0+ while the full one stays
        positive, and the continuous-M penalty -1/2 theta^T M theta outgrows a
        Poisson likelihood far from the minimum. A start far from the minimum
        (e.g. the blinding offsets) can then slide away from it. The ordinary
        minimum lies within O(sigma) of the de-biased one, inside its convex
        basin. Skipped for a quadratic objective, which is solved in closed
        form.
        """
        if (
            mcstat_warm_start
            and self.mcstat_debias_active
            and not self._mcstat_debias_suspended
            and not self.is_linear
        ):
            logger.info(
                "MC-stat de-biasing: minimizing the ordinary likelihood first and "
                "starting the de-biased fit from its minimum"
            )
            with self._ordinary_likelihood():
                self._minimize(warm_start=warm_start)
            logger.debug(f"ordinary-likelihood minimum (warm start): {self.x.numpy()}")
        return self._minimize(warm_start=warm_start)

    def _minimize(self, warm_start=False):
        if self.is_linear and len(self.constraints):
            if self.force_linear:
                raise ValueError(
                    "--forceLinear and --constraints: --forceLinear takes a single "
                    "Gaussian step even when the loss is not quadratic (for an "
                    "outer re-linearisation), which no constrained fit can "
                    "reproduce. Drop one of the two."
                )
            # A quadratic loss with constraints is a constrained problem the
            # closed-form solve below cannot honour: minimise it iteratively
            # instead (fit() requires a constrained minimizer).
            logger.info(
                "The likelihood is purely quadratic, but --constraints are set: "
                "minimising iteratively instead of the closed-form solve"
            )
        if self.is_linear and not len(self.constraints):
            if self.compute_cov:
                logger.info(
                    "Likelihood is purely quadratic, solving by Cholesky decomposition instead of iterative fit"
                )

                # no need to do a minimization, simple matrix solve is sufficient
                val, grad, hess = self.loss_val_grad_hess()

                # use a Cholesky decomposition to easily detect the non-positive-definite case
                chol = tf.linalg.cholesky(hess)

                # FIXME catch this exception to mark failed toys and continue
                if tf.reduce_any(tf.math.is_nan(chol)).numpy():
                    raise ValueError(
                        "Cholesky decomposition failed, Hessian is not positive-definite"
                    )

                del hess
                gradv = grad[..., None]
                dx = tf.linalg.cholesky_solve(chol, -gradv)[:, 0]
                del chol

                self.x.assign_add(dx)
            else:
                # --noHessian: we must not allocate the dense [npar, npar]
                # Hessian that the Cholesky path above builds. Solve the
                # normal equation H @ dx = -grad iteratively via conjugate
                # gradient using only Hessian-vector products, which is
                # already exposed as self.loss_val_grad_hessp. For a
                # purely quadratic NLL the Hessian is positive-definite
                # and CG converges to machine precision in at most npar
                # steps (typically far fewer for well-conditioned
                # problems).
                import scipy.sparse.linalg as _spla

                logger.info(
                    "Likelihood is purely quadratic, solving with "
                    "Hessian-free conjugate gradient (--noHessian)"
                )
                val, grad = self.loss_val_grad()
                grad_np = grad.numpy()
                n = int(grad_np.shape[0])
                dtype = grad_np.dtype

                def _hvp_np(p_np):
                    p_tf = tf.constant(p_np, dtype=self.x.dtype)
                    _, _, hessp = self.loss_val_grad_hessp(p_tf)
                    return hessp.numpy()

                op = _spla.LinearOperator((n, n), matvec=_hvp_np, dtype=dtype)
                dx_np, info = _spla.cg(op, -grad_np, rtol=1e-10, atol=0.0)
                if info != 0:
                    raise ValueError(
                        f"CG solver did not converge (info={info}); the "
                        "Hessian may not be positive-definite or the "
                        "problem may be ill-conditioned"
                    )
                self.x.assign_add(tf.constant(dx_np, dtype=self.x.dtype))

            callback = None
        else:
            callback = self.fit(warm_start=warm_start)

        return callback

    # Per-point diagnostics collected by nll_scan under save_detail, in the
    # column order of detail["diagnostics"]. rabbit_fit.py passes this tuple to
    # Workspace.add_nll_scan_detail_hists as the diagnostic axis labels.
    SCAN_DETAIL_FIELDS = (
        "edmval",
        "grad_max_abs",
        "success",
        "status",
        "nit",
        "nfev",
        "wall_s",
    )

    def nll_scan(
        self, param, scan_range, scan_points, use_prefit=False, save_detail=False
    ):
        # make a likelihood scan for a single parameter
        # assuming the likelihood is minimized

        # freeze minimize which mean to not update it in the fit
        self.freeze_params(param)

        idx = np.where(self.parms.astype(str) == param)[0][0]

        # store current state of x temporarily
        xval = tf.identity(self.x)

        param_offsets = np.linspace(0, scan_range, scan_points // 2 + 1)
        if not use_prefit:
            param_offsets *= self.cov[idx, idx].numpy() ** 0.5

        nscans = 2 * len(param_offsets) - 1
        dnlls = np.full(nscans, np.nan)
        scan_vals = np.zeros(nscans)

        nparms = int(self.x.shape[0])
        if save_detail:
            detail = {
                "x": np.full((nscans, nparms), np.nan),
                "diagnostics": np.full((nscans, len(self.SCAN_DETAIL_FIELDS)), np.nan),
                "messages": [""] * nscans,
            }
        else:
            detail = None

        def _record(islot, wall_s):
            """Per-point parameter vector and convergence diagnostics.

            Costs one gradient evaluation plus one CG solve against the
            Hessian -- small next to the point's own profile minimisation
            (tens of trust-region iterations), but not free, which is why it
            is opt-in.
            """
            detail["x"][islot] = self.x.numpy()
            _, grad = self.loss_val_grad()
            # frozen parameters enter the loss through tf.stop_gradient, so
            # their gradient entries and Hessian rows/columns are exactly zero:
            # both numbers below only see the parameters that were minimised
            try:
                edmval = self.edmval_cov_rows_hessfree(grad, [])[0]
            except ValueError as e:
                # one bad point should not lose the scan
                logger.warning(f"{e}; reporting NaN edmval for this scan point")
                edmval = float("nan")
            status = self.minimizer_status() or {}
            values = {
                "edmval": edmval,
                "grad_max_abs": float(np.max(np.abs(grad.numpy()), initial=0.0)),
                "success": float(status.get("success", np.nan)),
                "status": float(status.get("status", np.nan)),
                "nit": float(status.get("nit", np.nan)),
                "nfev": float(status.get("nfev", np.nan)),
                "wall_s": wall_s,
            }
            for k, field in enumerate(self.SCAN_DETAIL_FIELDS):
                detail["diagnostics"][islot, k] = values[field]
            detail["messages"][islot] = str(status.get("message", ""))

        # save delta nll w.r.t. global minimum
        nll_best = self.reduced_nll().numpy()
        # set central point
        dnlls[nscans // 2] = 0
        scan_vals[nscans // 2] = xval[idx].numpy()
        if save_detail:
            # no fit runs here: the caller's minimum IS the central point, so
            # its minimizer_status belongs to whatever fit produced it
            _record(nscans // 2, float("nan"))
        # scan positive side and negative side independently to profit from previous step
        for sign in [-1, 1]:
            param_scan_values = xval[idx].numpy() + sign * param_offsets
            for i, ixval in enumerate(param_scan_values):
                if i == 0:
                    continue

                logger.debug(f"Now at i={i} x={ixval}")
                self.x.assign(tf.tensor_scatter_nd_update(self.x, [[idx]], [ixval]))

                _t0 = time.perf_counter()
                self.fit(warm_start=True)
                _wall = time.perf_counter() - _t0

                islot = nscans // 2 + sign * i
                dnlls[islot] = self.reduced_nll().numpy() - nll_best

                scan_vals[islot] = ixval

                if save_detail:
                    _record(islot, _wall)

            # reset x to original state
            self.x.assign(xval)

        # let the parameter be free again
        self.defreeze_params(param)

        return scan_vals, dnlls, detail

    def nll_scan2D(self, param_tuple, scan_range, scan_points, use_prefit=False):

        # freeze minimize which mean to not update it in the fit
        self.freeze_params(param_tuple)

        idx0 = np.where(self.parms.astype(str) == param_tuple[0])[0][0]
        idx1 = np.where(self.parms.astype(str) == param_tuple[1])[0][0]

        xval = tf.identity(self.x)

        dsigs = np.linspace(-scan_range, scan_range, scan_points)
        if not use_prefit:
            x_scans = xval[idx0] + dsigs * self.cov[idx0, idx0] ** 0.5
            y_scans = xval[idx1] + dsigs * self.cov[idx1, idx1] ** 0.5
        else:
            x_scans = dsigs
            y_scans = dsigs

        best_fit = (scan_points + 1) // 2 - 1
        dnlls = np.full((len(x_scans), len(y_scans)), np.nan)
        nll_best = self.reduced_nll().numpy()
        dnlls[best_fit, best_fit] = 0
        # scan in a spiral around the best fit point
        dcol = -1
        drow = 0
        i = 0
        j = 0
        r = 1
        while r - 1 < best_fit:
            if i == r and drow == 1:
                drow = 0
                dcol = 1
            if j == r and dcol == 1:
                dcol = 0
                drow = -1
            elif i == -r and drow == -1:
                dcol = -1
                drow = 0
            elif j == -r and dcol == -1:
                drow = 1
                dcol = 0

            i += drow
            j += dcol

            if i == -r and j == -r:
                r += 1

            ix = best_fit - i
            iy = best_fit + j

            logger.debug(
                f"Now at (ix,iy) = ({ix},{iy}) (x,y)= ({x_scans[ix]},{y_scans[iy]})"
            )

            self.x.assign(
                tf.tensor_scatter_nd_update(
                    self.x, [[idx0], [idx1]], [x_scans[ix], y_scans[iy]]
                )
            )

            self.fit(warm_start=True)

            dnlls[ix, iy] = self.reduced_nll().numpy() - nll_best

        self.x.assign(xval)

        # let the parameter be free again
        self.defreeze_params(param_tuple)

        return x_scans, y_scans, dnlls

    def contour_scan(
        self,
        param,
        nll_min,
        q=1,
        signs=[-1, 1],
        fun=None,
        xtol=1e-6,
        gtol=1e-6,
        maxiter=5000,
        hess_mode="exact",
    ):
        # Layered cache: trust-constr calls scipy_loss many times during line
        # search (only val needed), and scipy_grad / scipy_hess on accepted
        # steps. The cache is keyed by x content so repeated requests at the
        # same point are free.
        lg_cache = {"x": None, "val": None, "grad": None}

        def _ensure_loss_grad(x):
            if lg_cache["x"] is not None and np.array_equal(lg_cache["x"], x):
                return
            self.x.assign(x)
            val, grad = self.loss_val_grad()
            lg_cache["x"] = np.array(x, copy=True)
            lg_cache["val"] = float(val.numpy()) - nll_min - 0.5 * q
            lg_cache["grad"] = grad.numpy()

        # Constraint Hessian. Modes:
        #   "exact": recompute the full NLL Hessian at every accepted iteration
        #       (~25 s/eval for thousands of params; dominant cost). Reference.
        #   "hvp": LinearOperator whose matvec computes one Hessian-vector
        #       product via a nested GradientTape (~2x the cost of a gradient).
        #       Exact (no approximation); avoids materializing the N x N matrix.
        #       trust-constr only multiplies H against trial directions in its
        #       inner CG, so HVP is typically much faster than "exact".
        #   "frozen": constant Hessian = postfit precision matrix (cov^-1),
        #       computed once. Cheapest, but the Lagrangian model is wrong off
        #       the postfit, so trust-constr's KKT/optimality criterion can be
        #       satisfied while the constraint violation is large -- producing
        #       silent failures on non-Gaussian profiles. Useful only as a
        #       speed reference, not as a production default.
        #   "bfgs" / "sr1": quasi-Newton Hessian estimate built up by
        #       trust-constr from the gradient sequence (no extra TF calls).
        #       Cheapest per iteration; may need more iterations to converge.
        #       SR1 is more robust for non-convex local geometry than BFGS.
        if hess_mode == "hvp":

            def scipy_hess(x, v):
                _ensure_loss_grad(x)
                n = len(x)
                scale = float(v[0])

                def _matvec(p):
                    self.x.assign(x)
                    p_tf = tf.convert_to_tensor(p, dtype=self.indata.dtype)
                    _, _, hp = self.loss_val_grad_hessp(p_tf)
                    return scale * hp.numpy()

                return scipy.sparse.linalg.LinearOperator(
                    shape=(n, n), matvec=_matvec, dtype=np.float64
                )

        elif hess_mode == "frozen":
            postfit_hess = np.linalg.inv(self.cov.numpy())

            def scipy_hess(x, v):
                return v[0] * postfit_hess

        elif hess_mode == "bfgs":
            scipy_hess = scipy.optimize.BFGS()

        elif hess_mode == "sr1":
            scipy_hess = scipy.optimize.SR1()

        elif hess_mode == "exact":
            h_cache = {"x": None, "hess": None}

            def _ensure_hess(x):
                if h_cache["x"] is not None and np.array_equal(h_cache["x"], x):
                    return
                self.x.assign(x)
                val, grad, hess = self.loss_val_grad_hess()
                h_cache["x"] = np.array(x, copy=True)
                h_cache["hess"] = hess.numpy()
                # opportunistically refresh the loss/grad cache.
                lg_cache["x"] = h_cache["x"]
                lg_cache["val"] = float(val.numpy()) - nll_min - 0.5 * q
                lg_cache["grad"] = grad.numpy()

            def scipy_hess(x, v):
                _ensure_hess(x)
                return v[0] * h_cache["hess"]

        else:
            raise ValueError(
                f"contour_scan: unknown hess_mode={hess_mode!r}; "
                f"expected one of {CONTOUR_HESS_MODES}."
            )

        def scipy_loss(x):
            _ensure_loss_grad(x)
            return np.array([lg_cache["val"]])

        def scipy_grad(x):
            _ensure_loss_grad(x)
            return lg_cache["grad"][None, :]

        nlc = scipy.optimize.NonlinearConstraint(
            fun=scipy_loss,
            lb=0,
            ub=0,
            jac=scipy_grad,
            hess=scipy_hess,
        )

        intervals = np.full((len(signs)), np.nan)
        params_values = np.full((len(signs), len(self.parms)), np.nan)

        xval = tf.identity(self.x)
        xval_np = xval.numpy()

        idx = np.where(self.parms.astype(str) == param)[0][0]
        x0 = xval_np[idx]

        # Gaussian-optimal warm start on the Delta(2NLL)=q contour:
        # maximizing +/- dx[idx] subject to dx^T H dx = q (with H = cov^{-1})
        # gives dx = sign * sqrt(q) * cov[:,idx] / sqrt(cov[idx,idx]).
        # This lands all parameters on the contour in the Gaussian limit and
        # is exact for nuisances with a near-quadratic likelihood, so the
        # constrained minimization typically converges in just a few steps.
        cov_col = self.cov[:, idx].numpy()
        sigma_idx = float(self.cov[idx, idx].numpy()) ** 0.5
        gauss_dx = (q**0.5) * cov_col / sigma_idx
        # Frozen parameters must stay at their current values: their gradients
        # are masked, so the optimizer would never move them back, and a
        # displaced frozen parameter shifts the NLL value and biases the
        # contour. Zero their warm-start displacement.
        if len(self.frozen_indices):
            gauss_dx[self.frozen_indices] = 0.0

        for i, sign in enumerate(signs):
            xval_init = xval_np + sign * gauss_dx
            t_side0 = time.perf_counter()

            opt = {}
            if fun is None:
                # contour scan on parameter
                def objective_val_grad(x):
                    self.x.assign(x)
                    val = -sign * (x[idx] - x0)
                    grad = np.zeros_like(x)
                    grad[idx] = -sign

                    # logger.info(f"val = {val}")
                    # logger.info(f"Grad = {grad}")
                    return val, grad

                from scipy.sparse import csr_matrix

                n_params = len(xval_init)
                obj_hess = csr_matrix((n_params, n_params))
                opt["hess"] = lambda x: obj_hess
            else:
                # contour scan on observable
                def objective_val_grad(x):
                    self.x.assign(x)
                    with tf.GradientTape() as t:
                        expected = self._compute_expected(
                            fun,
                            inclusive=True,
                            profile=True,
                            full=True,
                            need_observables=True,
                        )
                        val = -sign * tf.squeeze(expected)
                    grad = t.gradient(val, self.x)
                    return val.__array__(), grad.__array__()

                def objective_hessp(x, pval):
                    self.x.assign(x)
                    p = tf.convert_to_tensor(pval, dtype=self.indata.dtype)
                    p = tf.stop_gradient(p)
                    with tf.GradientTape() as t2:
                        with tf.GradientTape() as t1:
                            expected = self._compute_expected(
                                fun,
                                inclusive=True,
                                profile=True,
                                full=True,
                                need_observables=True,
                            )
                            val = -sign * tf.squeeze(expected)
                        grad = t1.gradient(val, self.x)
                    hessp = t2.gradient(grad, self.x, output_gradients=p)
                    return hessp.__array__()

                opt["hessp"] = objective_hessp

            res = scipy.optimize.minimize(
                objective_val_grad,
                xval_init,
                method="trust-constr",
                jac=True,
                constraints=[nlc],
                options={
                    "maxiter": maxiter,
                    "xtol": xtol,
                    "gtol": gtol,
                },
                **opt,
            )

            t_side = time.perf_counter() - t_side0
            logger.info(
                f"Success: {res.success} sign={sign} time={t_side:.2f}s "
                f"(nit={getattr(res, 'nit', '?')}, "
                f"nfev={getattr(res, 'nfev', '?')}, "
                f"njev={getattr(res, 'njev', '?')}, "
                f"nhev={getattr(res, 'nhev', '?')})"
            )
            logger.debug(f"Status: {res.status}")
            if not res.success:
                logger.warning(f"Message: {res.message}")
                logger.warning(f"Optimality (gtol): {res.optimality}")
                logger.warning(f"Constraint Violation: {res.constr_violation}")
                self.x.assign(xval)
                continue

            params_values[i] = res["x"] - xval

            if fun is None:
                val = res["x"][idx] - x0
            else:
                self.x.assign(res["x"])
                val = self._compute_expected(
                    fun,
                    inclusive=True,
                    profile=True,
                    full=True,
                    need_observables=True,
                )
            # reset the parameter values
            self.x.assign(xval)

            intervals[i] = val

        return intervals, params_values

    def contour_scan2D(self, param_tuple, nll_min, cl=1, n_points=16):
        # Not yet working
        def scipy_loss(xval):
            self.x.assign(xval)
            val, grad = self.loss_val_grad()
            return val.numpy()

        def scipy_grad(xval):
            self.x.assign(xval)
            val, grad = self.loss_val_grad()
            return grad.numpy()

        xval = tf.identity(self.x)

        # Constraint function and its derivatives
        delta_nll = 0.5 * cl**2

        def constraint(params):
            return scipy_loss(params) - nll_min - delta_nll

        nlc = scipy.optimize.NonlinearConstraint(
            fun=constraint,
            lb=-np.inf,
            ub=0,
            jac=scipy_grad,
            hess=scipy.optimize.SR1(),
        )

        # initial guess from covariance
        xval_init = xval.numpy()
        idx0 = np.where(self.parms.astype(str) == param_tuple[0])[0][0]
        idx1 = np.where(self.parms.astype(str) == param_tuple[1])[0][0]

        intervals = np.full((2, n_points), np.nan)
        for i, t in enumerate(np.linspace(0, 2 * np.pi, n_points, endpoint=False)):
            print(f"Now at {i} with angle={t}")

            # Objective function and its derivatives
            def objective(params):
                # coordinate center (best fit)
                x = params[idx0] - xval[idx0]
                y = params[idx1] - xval[idx1]
                return -(x**2 + y**2)

            def objective_jac(params):
                x = params[idx0] - xval[idx0]
                y = params[idx1] - xval[idx1]
                jac = np.zeros_like(params)
                jac[idx0] = -2 * x
                jac[idx1] = -2 * y
                return jac

            def objective_hessp(params, v):
                hessp = np.zeros_like(v)
                hessp[idx0] = -2 * v[idx0]
                hessp[idx1] = -2 * v[idx1]
                return hessp

            def constraint_angle(params):
                # coordinate center (best fit)
                x = params[idx0] - xval[idx0]
                y = params[idx1] - xval[idx1]
                return x * np.sin(t) - y * np.cos(t)

            def constraint_angle_jac(params):
                jac = np.zeros_like(params)
                jac[idx0] = np.sin(t)
                jac[idx1] = -np.cos(t)
                return jac

            # constraint on angle
            tc = scipy.optimize.NonlinearConstraint(
                fun=constraint_angle,
                lb=0,
                ub=0,
                jac=constraint_angle_jac,
                hess=scipy.optimize.SR1(),
            )

            res = scipy.optimize.minimize(
                objective,
                xval_init,
                method="trust-constr",
                jac=objective_jac,
                hessp=objective_hessp,
                constraints=[nlc, tc],
                options={
                    "maxiter": 10000,
                    "xtol": 1e-14,
                    "gtol": 1e-14,
                    # "verbose": 3
                },
            )

            print(res)

            if res["success"]:
                intervals[0, i] = res["x"][idx0]
                intervals[1, i] = res["x"][idx1]

            self.x.assign(xval)

        return intervals
