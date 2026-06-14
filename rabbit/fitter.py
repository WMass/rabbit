import hashlib
import re
import time

import h5py
import numpy as np
import scipy
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops as tf_sparse_csr
from wums import logging

from rabbit import external_likelihood, io_tools
from rabbit import tfhelpers as tfh
from rabbit.bbstat.bbstat import BinByBinStat
from rabbit.impacts import (
    asym_impacts,
    global_asym_impacts,
    global_impacts,
    nonprofiled_impacts,
    traditional_impacts,
)
from rabbit.tfhelpers import edmval_cov

logger = logging.child_logger(__name__)

# Supported constraint-Hessian modes for contour_scan (see its docstring for
# what each mode does). Single source of truth for the CLI choices and the
# contour_scan validation.
CONTOUR_HESS_MODES = ("exact", "hvp", "frozen", "bfgs", "sr1")


def match_regexp_params(regular_expressions, parameter_names):
    if isinstance(regular_expressions, str):
        regular_expressions = [regular_expressions]

    # Check for exact matches first
    exact_matches = [
        s for expr in regular_expressions for s in parameter_names if s.decode() == expr
    ]
    if exact_matches:
        return exact_matches

    # Fall back to regex matching
    compiled_expressions = [re.compile(expr) for expr in regular_expressions]
    return [
        s
        for s in parameter_names
        if any(regex.match(s.decode()) for regex in compiled_expressions)
    ]


class FitterCallback:
    def __init__(self, xv, early_stopping=-1):
        self.iiter = 0
        self.xval = xv

        self.loss_history = []
        self.time_history = []

        self.t0 = time.time()

        self.early_stopping = early_stopping

    def __call__(self, intermediate_result):
        loss = intermediate_result.fun

        logger.debug(f"Iteration {self.iiter}: loss value {loss}")
        if np.isnan(loss):
            raise ValueError(f"Loss value is NaN at iteration {self.iiter}")

        if (
            self.early_stopping > 0
            and len(self.loss_history) > self.early_stopping
            and self.loss_history[-self.early_stopping] <= loss
        ):
            raise ValueError(
                f"No reduction in loss after {self.early_stopping} iterations, early stopping."
            )

        self.loss_history.append(loss)
        self.time_history.append(time.time() - self.t0)

        self.xval = intermediate_result.x
        self.iiter += 1


class Fitter:
    valid_systematic_types = ["log_normal", "normal"]

    def __init__(
        self, indata, param_model, options, globalImpactsFromJVP=True, do_blinding=False
    ):
        self.indata = indata

        self.earlyStopping = options.earlyStopping
        self.globalImpactsFromJVP = globalImpactsFromJVP

        if self.indata.systematic_type not in Fitter.valid_systematic_types:
            raise RuntimeError(
                f"Invalid systematic_type {self.indata.systematic_type}, valid choices are {Fitter.valid_systematic_types}"
            )

        self.diagnostics = options.diagnostics
        self.minimizer_method = options.minimizerMethod
        self.hvp_method = getattr(options, "hvpMethod", "revrev")
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
        # When True, self.is_linear is forced True regardless of the actual
        # linearity of the param model, asymmetry tensor, or systematic type.
        # The minimize() path then takes a single Cholesky / Hessian-CG step
        # from the current x, i.e. a Gaussian approximation around that point.
        # Convergence to the true minimum then requires an outer iteration
        # that re-anchors the linearization point.
        self.force_linear = getattr(options, "forceLinear", False)

        self.do_blinding = do_blinding
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
            nobs_template=self.nobs,
        )

        # --- fit params
        self.init_fit_parms(
            param_model,
            options.setConstraintMinimum,
            unblind=options.unblind,
            freeze_parameters=options.freezeParameters,
        )

        self.nexpnom = tf.Variable(
            self.expected_yield(), trainable=False, name="nexpnom"
        )

        # --- limited-MC-statistics de-biasing options
        self.mcStatDebias = getattr(options, "mcStatDebias", "none")
        self.mcStatDebiasCov = getattr(options, "mcStatDebiasCov", "sandwich")
        self.mcStatKfold = getattr(options, "mcStatKfold", 2)
        self.covMode = getattr(options, "covMode", "observed")
        # Frozen noise-floor matrix M (nparams x nparams), reconstructed from any
        # external term named "mcstat*" (TensorWriter.add_mc_stat_moment stores
        # hess = -M, so M = -scatter(hess_dense)). Used for the continuous-M
        # sandwich covariance Sigma = A^-1 + A^-1 M A^-1 (H - A = M).
        self.mcstat_M = self._build_mcstat_M()
        # Two-half / k-fold cross-fit templates: precompute the half-sample
        # norm tensors norm_A, norm_B [nbinsfull, nproc] from the fold
        # templates (default groups = first/second half of the k folds, each
        # rescaled by k/(k/2)=2 so <norm_A>=<norm_B>=norm_full). Shared logk.
        self.norm_A = None
        self.norm_B = None
        norm_folds = getattr(self.indata, "norm_folds", None)
        if norm_folds is not None:
            k = int(norm_folds.shape[0])
            if k % 2 != 0:
                raise NotImplementedError(
                    f"two-half de-biasing needs an even number of folds; got k={k}."
                )
            half = k // 2
            scale = tf.constant(k / half, dtype=self.indata.dtype)  # = 2
            self.norm_A = scale * tf.reduce_sum(norm_folds[:half], axis=0)
            self.norm_B = scale * tf.reduce_sum(norm_folds[half:], axis=0)
        if self.mcStatDebias in ("twoHalf", "kfold") and self.norm_A is None:
            logger.warning(
                f"--mcStatDebias {self.mcStatDebias} requested but no fold "
                "templates (hnorm_folds) found in the input; two-half de-biasing "
                "is inactive. Add processes with a fold_axis in the TensorWriter."
            )
        if self.mcStatDebias == "continuousM" and self.mcstat_M is None:
            logger.warning(
                "--mcStatDebias continuousM requested but no 'mcstat' external "
                "moment term found in the input; continuous-M is inactive. Supply "
                "M via TensorWriter.add_mc_stat_moment(M, param_names)."
            )
        if self.mcStatDebias == "continuousM" and not self.param_model.is_linear:
            logger.warning(
                "--mcStatDebias continuousM: the param model is NONLINEAR "
                "(e.g. the default rabbit x^2 POI transform). The -1/2 theta^T M theta "
                "penalty is cancelled by Fisher growth at the shifted minimum and will "
                "NOT de-bias the point or covariance (RESULTS.md S9a). Use "
                "--allowNegativeParam (linear POI) or the two-half method instead."
            )

    def init_fit_parms(
        self,
        param_model,
        set_constraint_minimum=[],
        unblind=False,
        freeze_parameters=[],
    ):
        self.param_model = param_model

        # Internal (scaled) copy of indata.logk used by the yield-computation
        # hot path. For systematic_type == "normal" with a non-trivial param
        # model, the linearized additive variation Δ does not naturally scale
        # with the param-model factor rnorm(poi), so a ±20% variation defined
        # at the MC nominal becomes a different relative effect once rnorm
        # moves away from 1. Pre-multiplying logk by rnorm_init (the param
        # model evaluated at xparamdefault) restores the relative size of the
        # variation at the linearization point, without introducing a θ·poi
        # bilinearity in the hot path. For log_normal systematics the
        # multiplicative form already has this property so no copy is made.
        self._init_logk_scaled()

        if self.do_blinding:
            self._blinding_offsets_poi = tf.Variable(
                tf.ones([self.param_model.npoi], dtype=self.indata.dtype),
                trainable=False,
                name="offset_poi",
            )
            self._blinding_offsets_theta = tf.Variable(
                tf.zeros([self.indata.nsyst], dtype=self.indata.dtype),
                trainable=False,
                name="offset_theta",
            )
            self.init_blinding_values(unblind)

        self.parms = np.concatenate([self.param_model.params, self.indata.systs])

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
        else:
            self.cov = None

        # regularization
        self.regularizers = []
        # one common regularization strength parameter
        self.tau = tf.Variable(1.0, trainable=True, name="tau", dtype=tf.float64)

        # External likelihood terms (additive g^T x + 0.5 x^T H x
        # contributions to the NLL). See rabbit.external_likelihood for
        # the construction helper and the matching scalar evaluator.
        self.external_terms = external_likelihood.build_tf_external_terms(
            self.indata.external_terms,
            self.parms,
            self.indata.dtype,
        )

        # constraint minima for nuisance parameters
        self.theta0 = tf.Variable(
            self.theta0default,
            trainable=False,
            name="theta0",
        )
        self.var_theta0 = tf.where(
            self.indata.constraintweights == 0.0,
            tf.zeros_like(self.indata.constraintweights),
            tf.math.reciprocal(self.indata.constraintweights),
        )

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
        dynamic_tf_funcs = {
            "loss_val",
            "loss_val_grad",
            "loss_val_grad_hessp",
            "loss_val_grad_hessp_fwdrev",
            "loss_val_grad_hessp_revrev",
        }
        skip = jit_overrides | dynamic_tf_funcs
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

    def init_blinding_values(self, unblind_parameter_expressions=[]):
        logger.debug(f"Unblind parameters with {unblind_parameter_expressions}")
        unblind_parameters = match_regexp_params(
            unblind_parameter_expressions,
            [
                *self.param_model.params[: self.param_model.npoi],
                *[self.indata.systs[i] for i in self.indata.noiidxs],
            ],
        )

        # check if dataset is an integer (i.e. if it is real data or not) and use this to choose the random seed
        is_dataobs_int = np.sum(
            np.equal(self.indata.data_obs, np.floor(self.indata.data_obs))
        )

        def deterministic_random_from_string(s, mean=0.0, std=5.0):
            # random value with seed taken based on string of parameter name
            if isinstance(s, str):
                s = s.encode("utf-8")

            if is_dataobs_int:
                s += b"_data"

            # Hash the string
            hash = hashlib.sha256(s).hexdigest()

            seed_seq = np.random.SeedSequence(int(hash, 16))
            rng = np.random.default_rng(seed_seq)

            value = rng.normal(loc=mean, scale=std)
            return value

        # multiply offset to nois
        self._blinding_values_theta = np.zeros(self.indata.nsyst, dtype=np.float64)
        for i in self.indata.noiidxs:
            param = self.indata.systs[i]
            if param in unblind_parameters:
                continue
            logger.debug(f"Blind parameter {param}")
            value = deterministic_random_from_string(param)
            self._blinding_values_theta[i] = value

        # add offset to pois
        self._blinding_values_poi = np.ones(self.param_model.npoi, dtype=np.float64)
        for i in range(self.param_model.npoi):
            param = self.param_model.params[i]
            if param in unblind_parameters:
                continue
            logger.debug(f"Blind signal strength modifier for {param}")
            value = deterministic_random_from_string(param)
            self._blinding_values_poi[i] = np.exp(value)

    def set_blinding_offsets(self, blind=True):
        if not self.do_blinding:
            return
        if blind:
            self._blinding_offsets_poi.assign(self._blinding_values_poi)
            self._blinding_offsets_theta.assign(self._blinding_values_theta)
        else:
            self._blinding_offsets_poi.assign(
                np.ones(self.param_model.npoi, dtype=np.float64)
            )
            self._blinding_offsets_theta.assign(
                np.zeros(self.indata.nsyst, dtype=np.float64)
            )

    def get_theta(self):
        start = self.param_model.nparams
        theta = self.x[start : start + self.indata.nsyst]
        theta = tf.where(
            self.frozen_params_mask[start : start + self.indata.nsyst],
            tf.stop_gradient(theta),
            theta,
        )
        if self.do_blinding:
            return theta + self._blinding_offsets_theta
        else:
            return theta

    def get_model_nui(self):
        npoi = self.param_model.npoi
        npou = self.param_model.npou
        nui = self.x[npoi : npoi + npou]
        # Apply frozen_params_mask the same way get_poi() and get_theta() do.
        # Without this, --freezeParameters silently fails to freeze POUs:
        # the param is registered as frozen but no stop_gradient is applied,
        # so the optimizer updates it anyway.
        nui = tf.where(
            self.frozen_params_mask[npoi : npoi + npou],
            tf.stop_gradient(nui),
            nui,
        )
        return nui

    def get_poi(self):
        xpoi = self.x[: self.param_model.npoi]
        if self.param_model.allowNegativeParam:
            poi = xpoi
        else:
            poi = tf.square(xpoi)
        poi = tf.where(
            self.frozen_params_mask[: self.param_model.npoi], tf.stop_gradient(poi), poi
        )
        if self.do_blinding:
            return poi * self._blinding_offsets_poi
        else:
            return poi

    def get_x(self):
        return tf.concat(
            [self.get_poi(), self.get_model_nui(), self.get_theta()], axis=0
        )

    def prefit_variance(self, unconstrained_err=0.0):
        """Per-parameter prefit variance vector of length npar.

        Free parameters (POIs and unconstrained nuisances) are assigned a
        placeholder variance of unconstrained_err**2 (zero by default).
        Constrained nuisances take their variance from the constraint
        term (1 / constraintweight).
        """
        var_poi = (
            tf.ones([self.param_model.nparams], dtype=self.indata.dtype)
            * unconstrained_err**2
        )
        var_theta = tf.where(
            self.indata.constraintweights == 0.0,
            unconstrained_err**2,
            tf.math.reciprocal(self.indata.constraintweights),
        )
        return tf.concat([var_poi, var_theta], axis=0)

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

    def set_nobs(self, values, variances=None):
        if self.chisqFit:
            # covariance from data stat
            if tf.math.reduce_any(values <= 0).numpy():
                raise RuntimeError(
                    "Bins in 'nobs <= 0' encountered, chi^2 fit can not be performed."
                )
            self.varnobs.assign(values if variances is None else variances)

        self.nobs.assign(values)
        # compute offset for poisson nll improved numerical precision in minimizatoin
        # the offset is chosen to give the saturated likelihood
        nobssafe = tf.where(values == 0.0, tf.constant(1.0, dtype=values.dtype), values)
        self.lognobs.assign(tf.math.log(nobssafe))

    def theta0defaultassign(self):
        self.theta0.assign(self.theta0default)

    def xdefaultassign(self):
        if self.param_model.nparams == 0:
            self.x.assign(self.theta0)
        else:
            self.x.assign(
                tf.concat([self.param_model.xparamdefault, self.theta0], axis=0)
            )

    def defaultassign(self):
        var_pre = self.prefit_variance(
            unconstrained_err=self.prefit_unconstrained_nuisance_uncertainty
        )
        self.var_prefit.assign(var_pre)
        if self.cov is not None:
            self.cov.assign(tf.linalg.diag(var_pre))
        self.theta0defaultassign()
        if self.bbstat.enabled:
            self.bbstat.beta0_default_assign()
            self.bbstat.beta_default_assign()
        self.xdefaultassign()
        if self.do_blinding:
            self.set_blinding_offsets(False)

        xinit = self.get_x()
        nexp0 = self.expected_yield(full=True)
        for reg in self.regularizers:
            reg.set_expectations(xinit, nexp0)

    def bayesassign(self):
        # FIXME use theta0 as the mean and constraintweight to scale the width
        if self.param_model.nparams == 0:
            self.x.assign(
                self.theta0
                + tf.random.normal(shape=self.theta0.shape, dtype=self.theta0.dtype)
            )
        else:
            self.x.assign(
                tf.concat(
                    [
                        self.param_model.xparamdefault,
                        self.theta0
                        + tf.random.normal(
                            shape=self.theta0.shape, dtype=self.theta0.dtype
                        ),
                    ],
                    axis=0,
                )
            )

        self.bbstat.randomize_bayes()

    def frequentistassign(self):
        # FIXME use theta as the mean and constraintweight to scale the width
        self.theta0.assign(
            tf.random.normal(shape=self.theta0.shape, dtype=self.theta0.dtype)
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
                    data_var = data_nom if data_variances is None else data_variances
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

    def _build_mcstat_M(self):
        """Reconstruct the frozen noise-floor matrix M (nparams x nparams) from
        external terms whose name starts with 'mcstat'.

        TensorWriter.add_mc_stat_moment stores the term Hessian as hess = -M
        (so the objective contribution 0.5 x^T hess x = -1/2 x^T M x de-biases
        the curvature). Here we scatter each term's (-hess_dense) block back into
        the full nparams x nparams space, indexed by the term's parameter indices.
        Returns None if no such term is present (continuous-M inactive).
        """
        mcstat = [t for t in self.external_terms if str(t["name"]).startswith("mcstat")]
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
        sandwich meat H = grad^2 L_full. Mirrors _compute_nll but always uses the
        full templates and the ordinary per-bin ln."""
        nexpfullcentral, _, beta = self._compute_yields_with_beta(
            profile=True, compute_norm=False, full=len(self.regularizers)
        )
        nexp = nexpfullcentral[: self.indata.nbins]
        l = self._compute_ln(nexp) + self._compute_lc()
        lbeta = self._compute_lbeta(beta)
        if lbeta is not None:
            l = l + lbeta
        lext = self._compute_external_nll()
        if lext is not None:
            l = l + lext
        return l

    @tf.function
    def loss_val_grad_hess_meat(self):
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                val = self._compute_nll_meat()
            grad = t1.gradient(val, self.x)
        hess = t2.jacobian(grad, self.x)
        return val, grad, hess

    def cov_twohalf_sandwich(self, A):
        """Two-half / k-fold robust (sandwich) covariance Sigma = A^-1 H A^-1.

        Bread A = the de-biased (jackknife) objective Hessian; meat H = the
        full-sample NLL Hessian (loss_val_grad_hess_meat). covMode='observed'
        (autograd Hessians). See RABBIT_MCSTAT_DESIGN.md §2c.
        """
        _, _, H = self.loss_val_grad_hess_meat()
        Ainv = tf.linalg.inv(A)
        return Ainv @ tf.cast(H, Ainv.dtype) @ Ainv

    def cov_mcstat_sandwich(self, A):
        """Continuous-M robust (sandwich) covariance.

        Sigma = A^-1 H A^-1 with bread A = de-biased curvature (grad^2 of the
        objective WITH the -1/2 theta^T M theta penalty) and meat H = the
        same-sample curvature WITHOUT the penalty. Because the penalty is the
        frozen quadratic -1/2 theta^T M theta, H = A + M exactly, so

            Sigma = A^-1 (A + M) A^-1 = A^-1 + A^-1 M A^-1.

        No second Hessian pass is needed. Frozen support only (the M-penalty
        treats M as constant; valid in the linear-Gaussian regime where
        continuous-M de-biases, RESULTS.md S1c/S9a).

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

    def edmval_cov_rows_hessfree(self, grad, row_indices, rtol=1e-10, maxiter=None):
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

        def _hvp_np(p_np):
            p_tf = tf.constant(p_np, dtype=self.x.dtype)
            _, _, hessp = self.loss_val_grad_hessp(p_tf)
            return hessp.numpy()

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

    @tf.function
    def impacts_parms(self, hess):

        nstat = (
            self.param_model.npoi
            + self.param_model.npou
            + self.indata.nsystnoconstraint
        )
        cov_stat = self._cov_stat_floating(hess, nstat)

        if self.bbstat.enabled:
            val_no_bbb, grad_no_bbb, hess_no_bbb = self.loss_val_grad_hess(
                profile=False
            )
            cov_stat_no_bbb = self._cov_stat_floating(hess_no_bbb, nstat)
        else:
            cov_stat_no_bbb = None

        param_groups = self._resolved_param_impact_groups()
        impacts, impacts_grouped = traditional_impacts.impacts_parms(
            self.cov,
            cov_stat,
            cov_stat_no_bbb,
            self.param_model.npoi,
            self.indata.noiidxs,
            self.indata.systgroupidxs,
            nmodel_params=self.param_model.npoi + self.param_model.npou,
            param_groupidxs=[idxs for _, idxs in param_groups],
        )

        return impacts, impacts_grouped

    @tf.function
    def global_impacts_parms(self):
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
            self.cov,
        )

    @tf.function
    def gaussian_global_impacts_parms(self):
        dxdtheta0, dxdnobs, dxdbeta0 = self._dxdvars()

        impacts, impacts_grouped = global_impacts.gaussian_global_impacts_parms(
            dxdtheta0,
            dxdnobs,
            dxdbeta0,
            self.var_theta0,
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
            self.data_cov_inv,
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

        For each selected nuisance i, shift theta0[i] by +/- sigma (in units of
        the prefit constraint width) and re-run the full fit. POI shifts at
        each sign are the asymmetric global impacts.

        Unconstrained nuisances (constraintweight = 0) are always skipped:
        they have no prefit sigma, and their theta0 does not enter the NLL,
        so the shifted refit would reproduce the nominal minimum exactly
        (zero impact at the cost of two full fits).

        Args:
            include: optional regex(es) restricting which nuisances to scan.
            exclude: optional regex(es) excluding nuisances from the scan.
            sigma: shift magnitude in prefit-sigma units.
            linear_warmstart: experimental, see
                global_asym_impacts.global_asym_impacts_parms.
        """
        nsyst = self.indata.nsyst
        cw = self.indata.constraintweights.numpy()
        syst_names = np.array(self.indata.systs).astype(bytes)

        # theta0 of an unconstrained nuisance does not enter the NLL: no
        # finite prefit sigma to shift by, and the refit would be a no-op.
        selected = cw > 0
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
            f"global_asym_impacts_parms: selected {len(selected_idxs)}/{nsyst} "
            f"nuisances (unconstrained nuisances always excluded)"
        )

        return global_asym_impacts.global_asym_impacts_parms(
            self,
            selected_idxs,
            selected_names,
            sigma=sigma,
            linear_warmstart=linear_warmstart,
        )

    def nonprofiled_impacts_parms(self, unconstrained_err=1.0):
        return nonprofiled_impacts.nonprofiled_impacts_parms(
            self.x,
            self.theta0,
            self.frozen_indices,
            self.frozen_params,
            self.indata.constraintweights,
            self.indata.systgroups,
            self.indata.systgroupidxs,
            self.param_model.nparams,
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
        with tf.GradientTape() as t2:
            t2.watch([self.theta0, self.nobs, self.bbstat.beta0])
            with tf.GradientTape() as t1:
                t1.watch([self.theta0, self.nobs, self.bbstat.beta0])
                val = self._compute_loss()
            grad = t1.gradient(val, self.x)
        pd2ldxdtheta0, pd2ldxdnobs, pd2ldxdbeta0 = t2.jacobian(
            grad,
            [self.theta0, self.nobs, self.bbstat.beta0],
            unconnected_gradients="zero",
        )

        # cov is inverse hesse, thus cov ~ d2xd2l
        dxdtheta0 = -self.cov @ pd2ldxdtheta0
        dxdnobs = -self.cov @ pd2ldxdnobs
        dxdbeta0 = -self.cov @ tf.reshape(pd2ldxdbeta0, [pd2ldxdbeta0.shape[0], -1])

        return dxdtheta0, dxdnobs, dxdbeta0

    def _dndvars(self, fun):
        with tf.GradientTape() as t:
            t.watch([self.theta0, self.nobs, self.bbstat.beta0])
            n = fun()
            n_flat = tf.reshape(n, (-1,))

        pdndx, pdndtheta0, pdndnobs, pdndbeta0 = t.jacobian(
            n_flat,
            [self.x, self.theta0, self.nobs, self.bbstat.beta0],
            unconnected_gradients="zero",
        )

        # apply chain rule to take into account correlations with the fit parameters
        dxdtheta0, dxdnobs, dxdbeta0 = self._dxdvars()

        dndtheta0 = pdndtheta0 + pdndx @ dxdtheta0
        dndnobs = pdndnobs + pdndx @ dxdnobs
        dndbeta0 = tf.reshape(pdndbeta0, [pdndbeta0.shape[0], -1]) + pdndx @ dxdbeta0

        return n, dndtheta0, dndnobs, dndbeta0

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
                cov_dexpdx,
                expvar_flat,
                expvar.shape,
                profile,
                pdexpdbeta,
                pd2ldbeta2_pdexpdbeta if pdexpdbeta is not None else None,
                self.prefit_unconstrained_nuisance_uncertainty,
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

            _, dndtheta0, dndnobs, dndbeta0 = self._dndvars(fun_n)
            impacts_gaussian, impacts_gaussian_grouped = (
                global_impacts.gaussian_global_impacts_obs(
                    dndtheta0,
                    dndnobs,
                    dndbeta0,
                    self.var_theta0,
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
                    self.data_cov_inv,
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
        """Build an internal copy of indata.logk for the yield-computation
        hot path, pre-multiplied per (bin, proc) by the param-model factor
        evaluated at xparamdefault.

        For systematic_type == "log_normal" the multiplicative form
        ``rnorm * exp(θ·logk) * norm`` already carries the param-model
        scaling through to the variation, so no copy is needed and
        self.logk / self.logk_csr alias the indata tensors.

        For systematic_type == "normal" the linearized variation
        ``rnorm * norm + θ·logk`` does not scale with rnorm. We absorb a
        constant rnorm_init = param_model.compute(xparamdefault) into logk
        once, so the relative size of an additive variation matches the
        multiplicative case at the linearization point. The scaling is a
        constant, so the hot path remains strictly linear in θ.
        """
        if self.indata.systematic_type != "normal" or self.param_model.nparams == 0:
            self.logk = self.indata.logk
            if self.indata.sparse:
                self.logk_csr = self.indata.logk_csr
            return

        rnorm_init = self.param_model.compute(self.param_model.xparamdefault, full=True)
        rnorm_init = tf.broadcast_to(
            rnorm_init, [self.indata.nbinsfull, self.indata.nproc]
        )

        if self.indata.sparse:
            # logk dense shape is [norm_nnz, nsyst_or_2nsyst]; each value
            # at logk.indices[i] = (norm_pos, syst_pos) corresponds to the
            # (bin, proc) pair stored at norm.indices[norm_pos]. Gather
            # rnorm_init through this two-level mapping.
            rnorm_at_norm = tf.gather_nd(rnorm_init, self.indata.norm.indices)
            scale_per_logk = tf.gather(rnorm_at_norm, self.indata.logk.indices[:, 0])
            new_values = self.indata.logk.values * scale_per_logk
            self.logk = tf.SparseTensor(
                self.indata.logk.indices,
                new_values,
                self.indata.logk.dense_shape,
            )
            self.logk_csr = tf_sparse_csr.CSRSparseMatrix(self.logk)
        else:
            # Dense logk: [nbinsfull, nproc, nsyst] symmetric, or
            # [nbinsfull, nproc, 2, nsyst] asymmetric. Broadcast rnorm_init
            # over the trailing axes.
            if self.indata.symmetric_tensor:
                self.logk = self.indata.logk * rnorm_init[..., None]
            else:
                self.logk = self.indata.logk * rnorm_init[..., None, None]

    def _compute_yields_noBBB(self, full=True, compute_norm=True, templates="full"):
        # templates: "full" uses indata.norm; "A"/"B" use the precomputed
        # half-sample fold templates norm_A/norm_B (two-half de-biasing, shared
        # logk). Only supported in the dense path.
        # full: compute yields inclduing masked channels
        # compute_norm: also build the dense [nbins, nproc] normcentral tensor.
        # In sparse mode this is expensive (forward + backward) and is only
        # needed when an external caller requests per-process yields, or for
        # binByBinStat in "full" mode. The default is True for backward
        # compatibility; the NLL/grad/HVP path passes compute_norm=False.
        poi = self.get_poi()
        model_nui = self.get_model_nui()
        theta = self.get_theta()

        all_params = tf.concat([poi, model_nui], axis=0)
        rnorm = self.param_model.compute(all_params, full)

        normcentral = None
        if self.indata.symmetric_tensor:
            mthetaalpha = tf.reshape(theta, [self.indata.nsyst, 1])
        else:
            # interpolation for asymmetric log-normal
            twox = 2.0 * theta
            twox2 = twox * twox
            alpha = 0.125 * twox * (twox2 * (3.0 * twox2 - 10.0) + 15.0)
            alpha = tf.clip_by_value(alpha, -1.0, 1.0)

            thetaalpha = theta * alpha

            mthetaalpha = tf.stack(
                [theta, thetaalpha], axis=0
            )  # now has shape [2,nsyst]
            mthetaalpha = tf.reshape(mthetaalpha, [2 * self.indata.nsyst, 1])

        if self.indata.sparse:
            # Inner contraction logk · mthetaalpha via tf.linalg.sparse's
            # CSR matmul. ~8x faster per call than gather + segment_sum
            # because SparseMatrixMatMul dispatches to a hand-tuned CSR
            # kernel. NOTE: SparseMatrixMatMul has no XLA kernel, so the
            # enclosing loss/grad/HVP tf.functions are built with
            # jit_compile=False in sparse mode (see _make_tf_functions).
            logsnorm = tf.squeeze(
                tf_sparse_csr.matmul(self.logk_csr, mthetaalpha),
                axis=-1,
            )

            # Build a sparse [nbinsfull, nproc] tensor whose values absorb
            # the per-entry syst variation and the per-(bin, proc) POI
            # scaling rnorm. The sparsity pattern is unchanged from
            # self.indata.norm, so with_values lets us reuse the indices.
            if self.indata.systematic_type == "log_normal":
                # values[i] = norm[i] * exp(logsnorm[i]) * rnorm[bin, proc]
                snormnorm_sparse = self.indata.norm.with_values(
                    tf.exp(logsnorm) * self.indata.norm.values
                )
                snormnorm_sparse = snormnorm_sparse * rnorm
            else:  # "normal"
                # values[i] = norm[i] * rnorm[bin, proc] + logsnorm[i]
                snormnorm_sparse = self.indata.norm * rnorm
                snormnorm_sparse = snormnorm_sparse.with_values(
                    snormnorm_sparse.values + logsnorm
                )

            if not full and self.indata.nbinsmasked:
                snormnorm_sparse = tfh.simple_sparse_slice0end(
                    snormnorm_sparse, self.indata.nbins
                )

            # Per-bin yields via unsorted_segment_sum on the sparse values
            # keyed by bin index. Equivalent to tf.sparse.reduce_sum(...,
            # axis=-1) but uses the dedicated segment_sum kernel directly,
            # which has lower per-call overhead. The dense [nbinsfull,
            # nproc] grid is only materialized when an external caller
            # requested per-process yields (compute_norm=True).
            nbinsfull_int = int(snormnorm_sparse.dense_shape[0])
            nexpcentral = tf.math.unsorted_segment_sum(
                snormnorm_sparse.values,
                snormnorm_sparse.indices[:, 0],
                num_segments=nbinsfull_int,
            )
            if compute_norm:
                normcentral = tf.sparse.to_dense(snormnorm_sparse)
        else:
            if templates == "A":
                norm_src = self.norm_A
            elif templates == "B":
                norm_src = self.norm_B
            else:
                norm_src = self.indata.norm

            if full or self.indata.nbinsmasked == 0:
                nbins = self.indata.nbinsfull
                logk = self.logk
                norm = norm_src
            else:
                nbins = self.indata.nbins
                logk = self.logk[:nbins]
                norm = norm_src[:nbins]

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
            logsnorm = tf.reshape(logsnorm, [nbins, self.indata.nproc])

            if self.indata.systematic_type == "log_normal":
                snorm = tf.exp(logsnorm)
                snormnorm = snorm * norm
                normcentral = rnorm * snormnorm
            elif self.indata.systematic_type == "normal":
                normcentral = norm * rnorm + logsnorm

            nexpcentral = tf.reduce_sum(normcentral, axis=-1)

        return nexpcentral, normcentral

    def _compute_yields_with_beta(self, profile=True, compute_norm=False, full=True):
        # Only materialize the dense [nbins, nproc] normcentral when an
        # external caller requested it, when BBB "full" mode needs per-process
        # yields for the analytic β solution, or when "lite" mode needs to
        # split finite-variance (sumw2>0) and zero-variance (sumw2==0)
        # contributions per bin.
        need_norm = compute_norm or self.bbstat.needs_per_proc_norm()
        nexp, norm = self._compute_yields_noBBB(full=full, compute_norm=need_norm)
        return self.bbstat.profile_and_apply(
            nexp,
            norm,
            self.nobs,
            self.varnobs,
            self.lognobs,
            profile=profile,
            compute_norm=compute_norm,
            full=full,
        )

    @tf.function
    def _profile_beta(self):
        nexp, norm, beta = self._compute_yields_with_beta(full=False)
        self.bbstat.beta.assign(beta)

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

        residuals, dresdtheta0, dresdnobs, dresdbeta0 = self._dndvars(fun_res)

        res_cov = dresdtheta0 @ (self.var_theta0[:, None] * tf.transpose(dresdtheta0))

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

    def _chi2(self, res, res_cov, ndf_reduction=0):
        res = tf.reshape(res, (-1, 1))
        ndf = tf.size(res) - ndf_reduction

        if ndf_reduction > 0:
            # covariance matrix is in general non invertible with ndf < n
            # compute chi2 using pseudo inverse
            chi_square_value = tf.transpose(res) @ tf.linalg.pinv(res_cov) @ res
        else:
            chi_square_value = tf.transpose(res) @ tf.linalg.solve(res_cov, res)

        return tf.squeeze(chi_square_value), ndf

    @tf.function
    def chi2(self, fun, fun_data=None, ndf_reduction=0, profile=False):
        if profile:
            residuals, res_cov = self._residuals_profiled(fun)
        else:
            residuals, res_cov = self._residuals(fun, fun_data)
        return self._chi2(residuals, res_cov, ndf_reduction)

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
            chi2val, ndf = self.chi2(
                mapping.compute_flat,
                mapping._get_data,
                mapping.ndf_reduction,
                profile=profile,
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
        # constraints
        theta = self.get_theta()
        lc = self.indata.constraintweights * 0.5 * tf.square(theta - self.theta0)
        if full_nll:
            # normalization factor for normal distribution: log(1/sqrt(2*pi)) = -0.9189385332046727
            lc = lc + 0.9189385332046727 * self.indata.constraintweights

        return tf.reduce_sum(lc)

    def _compute_lbeta(self, beta, full_nll=False):
        return self.bbstat.lbeta(beta, full_nll=full_nll)

    def _compute_ln(self, nexp, full_nll=False):
        if self.chisqFit:
            ln = 0.5 * tf.reduce_sum((nexp - self.nobs) ** 2 / self.varnobs, axis=-1)
        elif self.covarianceFit:
            # Solve the system without inverting
            residual = tf.reshape(self.nobs - nexp, [-1, 1])  # chi2 residual
            ln = 0.5 * tf.reduce_sum(
                tf.matmul(
                    residual,
                    tf.matmul(self.data_cov_inv, residual),
                    transpose_a=True,
                )
            )
        else:
            nexpsafe = tf.where(
                self.nobs == 0.0, tf.constant(1.0, dtype=nexp.dtype), nexp
            )
            lognexp = tf.math.log(nexpsafe)

            # poisson term
            if full_nll:
                ldatafac = tf.math.lgamma(self.nobs + 1)
                ln = tf.reduce_sum(-self.nobs * lognexp + nexp + ldatafac, axis=-1)
            else:
                # poisson w/o constant factorial part term and with offset to improve numerical precision
                ln = tf.reduce_sum(
                    -self.nobs * (lognexp - self.lognobs) + nexp - self.nobs, axis=-1
                )
        return ln

    def _compute_nll_components(self, profile=True, full_nll=False):
        nexpfullcentral, _, beta = self._compute_yields_with_beta(
            profile=profile,
            compute_norm=False,
            full=len(self.regularizers),
        )

        nexp = nexpfullcentral[: self.indata.nbins]

        if (
            self.mcStatDebias in ("twoHalf", "kfold")
            and self.norm_A is not None
        ):
            # Cross-fit jackknife combination L_cf = 2 L_full - 1/2 L_A - 1/2 L_B.
            # Since mu_bar = 1/2(n_A + n_B) = n_full exactly, this de-biases the
            # point (gradient = cross-fit score) and curvature (cross-half
            # Fisher) regardless of nonlinearity. Shared logk; full templates
            # carry the BB-lite beta profiling (unchanged).
            nexp_A = self._compute_yields_noBBB(
                full=False, compute_norm=False, templates="A"
            )[0][: self.indata.nbins]
            nexp_B = self._compute_yields_noBBB(
                full=False, compute_norm=False, templates="B"
            )[0][: self.indata.nbins]
            ln = (
                2.0 * self._compute_ln(nexp, full_nll)
                - 0.5 * self._compute_ln(nexp_A, full_nll)
                - 0.5 * self._compute_ln(nexp_B, full_nll)
            )
        else:
            ln = self._compute_ln(nexp, full_nll)

        lc = self._compute_lc(full_nll)

        lbeta = self._compute_lbeta(beta, full_nll)

        if len(self.regularizers):
            x = self.get_x()
            penalties = [
                reg.compute_nll_penalty(x, nexpfullcentral) * tf.exp(2 * self.tau)
                for reg in self.regularizers
            ]
            lpenalty = tf.add_n(penalties)
        else:
            lpenalty = None

        return ln, lc, lbeta, lpenalty, beta

    def _compute_external_nll(self):
        """Sum of external likelihood term contributions: sum_i (g_i^T x_sub + 0.5 x_sub^T H_i x_sub)."""
        return external_likelihood.compute_external_nll(
            self.external_terms, self.x, self.indata.dtype
        )

    def _compute_nll(self, profile=True, full_nll=False):
        ln, lc, lbeta, lpenalty, beta = self._compute_nll_components(
            profile=profile, full_nll=full_nll
        )
        l = ln + lc

        if lbeta is not None:
            l = l + lbeta

        if lpenalty is not None:
            l = l + lpenalty

        lext = self._compute_external_nll()
        if lext is not None:
            l = l + lext
        return l

    def _compute_loss(self, profile=True):
        return self._compute_nll(profile=profile)

    def _make_tf_functions(self):
        # Build tf.function wrappers at instance construction time so that
        # jit_compile and the HVP autodiff mode can be controlled via fit
        # options without redefining the class. self.jit_compile has
        # already been resolved to a plain bool in __init__ (tri-state
        # "auto"/"on"/"off" collapsed against self.indata.sparse), so
        # this body just reads it.
        jit = self.jit_compile

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

    @tf.function
    def loss_val_grad_hess(self, profile=True):
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                val = self._compute_loss(profile=profile)
            grad = t1.gradient(val, self.x)
        hess = t2.jacobian(grad, self.x)
        return val, grad, hess

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

    def fit(self):
        logger.info("Perform iterative fit")

        def scipy_loss(xval):
            self.x.assign(xval)
            val, grad = self.loss_val_grad()
            return val.__array__(), grad.__array__()

        def scipy_hessp(xval, pval):
            self.x.assign(xval)
            p = tf.convert_to_tensor(pval)
            val, grad, hessp = self.loss_val_grad_hessp(p)
            return hessp.__array__()

        def scipy_hess(xval):
            self.x.assign(xval)
            val, grad, hess = self.loss_val_grad_hess()
            if self.diagnostics:
                cond_number = tfh.cond_number(hess)
                logger.info(f"  - Condition number: {cond_number}")
                edmval = tfh.edmval(grad, hess)
                logger.info(f"  - edmval: {edmval}")
            return hess.__array__()

        xval = self.x.numpy()

        callback = FitterCallback(xval, self.earlyStopping)

        if self.minimizer_method in [
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

        try:
            res = scipy.optimize.minimize(
                scipy_loss,
                xval,
                method=self.minimizer_method,
                jac=True,
                tol=0.0,
                callback=callback,
                **info_minimize,
            )
        except Exception as ex:
            # minimizer could have called the loss or hessp functions with "random" values, so restore the
            # state from the end of the last iteration before the exception
            xval = callback.xval
            logger.debug(ex)
        else:
            xval = res["x"]
            logger.debug(res)

        self.x.assign(xval)

        return callback

    def minimize(self):
        if self.is_linear:
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
            callback = self.fit()

        return callback

    def nll_scan(self, param, scan_range, scan_points, use_prefit=False):
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

        # save delta nll w.r.t. global minimum
        nll_best = self.reduced_nll().numpy()
        # set central point
        dnlls[nscans // 2] = 0
        scan_vals[nscans // 2] = xval[idx].numpy()
        # scan positive side and negative side independently to profit from previous step
        for sign in [-1, 1]:
            param_scan_values = xval[idx].numpy() + sign * param_offsets
            for i, ixval in enumerate(param_scan_values):
                if i == 0:
                    continue

                logger.debug(f"Now at i={i} x={ixval}")
                self.x.assign(tf.tensor_scatter_nd_update(self.x, [[idx]], [ixval]))

                self.fit()

                dnlls[nscans // 2 + sign * i] = self.reduced_nll().numpy() - nll_best

                scan_vals[nscans // 2 + sign * i] = ixval

            # reset x to original state
            self.x.assign(xval)

        # let the parameter be free again
        self.defreeze_params(param)

        return scan_vals, dnlls

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

            self.fit()

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
