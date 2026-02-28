import hashlib
import re
import time

import h5py
import numpy as np
import scipy
import tensorflow as tf
import tensorflow_probability as tfp
from wums import logging

from rabbit import io_tools
from rabbit import tfhelpers as tfh

logger = logging.child_logger(__name__)


def solve_quad_eq(a, b, c):
    return 0.5 * (-b + tf.sqrt(b**2 - 4.0 * a * c)) / a


def match_regexp_params(regular_expressions, parameter_names):
    if isinstance(regular_expressions, str):
        regular_expressions = [regular_expressions]
    # Find parameters that match any regex
    compiled_expressions = [re.compile(expr) for expr in regular_expressions]
    matched_parameters = [
        s
        for s in parameter_names
        if any(regex.match(s.decode()) for regex in compiled_expressions)
    ]
    return matched_parameters


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

        logger.debug(
            f"Iteration {self.iiter}: loss value {loss}"
        )  # ; status {intermediate_result.status}")
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
    valid_bin_by_bin_stat_types = ["gamma", "normal-additive", "normal-multiplicative"]
    valid_systematic_types = ["log_normal", "normal"]

    def __init__(
        self, indata, poi_model, options, globalImpactsFromJVP=True, do_blinding=False
    ):
        self.indata = indata

        self.earlyStopping = options.earlyStopping
        self.globalImpactsFromJVP = globalImpactsFromJVP
        self.binByBinStat = not options.noBinByBinStat
        self.binByBinStatMode = options.binByBinStatMode

        if options.binByBinStatType == "automatic":
            if options.covarianceFit:
                self.binByBinStatType = "normal-additive"
            elif options.binByBinStatMode == "full":
                self.binByBinStatType = "normal-multiplicative"
            else:
                self.binByBinStatType = "gamma"
        else:
            self.binByBinStatType = options.binByBinStatType

        if (
            self.binByBinStat
            and self.binByBinStatMode == "full"
            and not self.binByBinStatType.startswith("normal")
        ):
            raise Exception(
                'bin-by-bin stat only for option "--binByBinStatMode full" with "--binByBinStatType normal"'
            )

        if (
            options.covarianceFit
            and self.binByBinStat
            and not self.binByBinStatType.startswith("normal")
        ):
            raise Exception(
                'bin-by-bin stat only for option "--covarianceFit" with "--binByBinStatType normal"'
            )

        if self.binByBinStatType not in Fitter.valid_bin_by_bin_stat_types:
            raise RuntimeError(
                f"Invalid binByBinStatType {self.binByBinStatType}, valid choices are {Fitter.valid_bin_by_bin_stat_types}"
            )

        if self.indata.systematic_type not in Fitter.valid_systematic_types:
            raise RuntimeError(
                f"Invalid systematic_type {self.indata.systematic_type}, valid choices are {Fitter.valid_systematic_types}"
            )

        self.diagnostics = options.diagnostics
        self.minimizer_method = options.minimizerMethod

        if options.covarianceFit and options.chisqFit:
            raise Exception(
                'Use either "--covarianceFit" for chi-squared fit using covariance or "--chisqFit" for diagonal chi-squared fit'
            )

        self.chisqFit = options.chisqFit
        self.covarianceFit = options.covarianceFit
        self.prefitUnconstrainedNuisanceUncertainty = (
            options.prefitUnconstrainedNuisanceUncertainty
        )

        self.poi_model = poi_model

        self.do_blinding = do_blinding
        if self.do_blinding:
            self._blinding_offsets_poi = tf.Variable(
                tf.ones([self.poi_model.npoi], dtype=self.indata.dtype),
                trainable=False,
                name="offset_poi",
            )
            self._blinding_offsets_theta = tf.Variable(
                tf.zeros([self.indata.nsyst], dtype=self.indata.dtype),
                trainable=False,
                name="offset_theta",
            )
            self.init_blinding_values(options.unblind)

        self.parms = np.concatenate([self.poi_model.pois, self.indata.systs])

        # tf tensor containing default constraint minima
        theta0default = np.zeros(self.indata.nsyst)
        for parm, val in options.setConstraintMinimum:
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
        if self.poi_model.npoi > 0:
            xdefault = tf.concat(
                [self.poi_model.xpoidefault, self.theta0default], axis=0
            )
        else:
            xdefault = self.theta0default

        self.x = tf.Variable(xdefault, trainable=True, name="x")

        # for freezing parameters
        self.frozen_params = []
        self.frozen_params_mask = tf.Variable(
            tf.zeros_like(self.x, dtype=tf.bool), trainable=False, dtype=tf.bool
        )

        self.frozen_indices = np.array([])
        self.freeze_params(options.freezeParameters)

        # observed number of events per bin
        self.nobs = tf.Variable(
            tf.zeros_like(self.indata.data_obs), trainable=False, name="nobs"
        )
        self.lognobs = tf.Variable(
            tf.zeros_like(self.indata.data_obs), trainable=False, name="lognobs"
        )
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
                self.data_cov_inv = np.diag(
                    1.0 / self.indata.getattr("data_obs", "data_var")
                )
            else:
                # provided covariance
                self.data_cov_inv = self.indata.data_cov_inv

        # constraint minima for nuisance parameters
        self.theta0 = tf.Variable(
            self.theta0default,
            trainable=False,
            name="theta0",
        )

        # FIXME for now this is needed even if binByBinStat is off because of how it is used in the global impacts
        #  and uncertainty band computations (gradient is allowed to be zero or None and then propagated or skipped only later)

        # global observables for mc stat uncertainty
        if self.binByBinStatMode == "full":
            self.beta_shape = self.indata.sumw.shape
        elif self.binByBinStatMode == "lite":
            self.beta_shape = (self.indata.sumw.shape[0],)

        self.beta0 = tf.Variable(
            tf.zeros(self.beta_shape, dtype=self.indata.dtype),
            trainable=False,
            name="beta0",
        )
        self.logbeta0 = tf.Variable(
            tf.zeros(self.beta_shape, dtype=self.indata.dtype),
            trainable=False,
            name="logbeta0",
        )
        self.beta0defaultassign()

        # nuisance parameters for mc stat uncertainty
        self.beta = tf.Variable(self.beta0, trainable=False, name="beta")

        # dummy tensor to allow differentiation
        self.ubeta = tf.zeros_like(self.beta)

        if self.binByBinStat:
            if self.binByBinStatMode == "full":
                self.varbeta = self.indata.sumw2
                self.sumw = self.indata.sumw
            else:
                if self.indata.sumw2.ndim > 1:
                    self.varbeta = tf.reduce_sum(self.indata.sumw2, axis=-1)
                    self.sumw = tf.reduce_sum(self.indata.sumw, axis=-1)
                else:
                    self.varbeta = self.indata.sumw2
                    self.sumw = self.indata.sumw

            if self.binByBinStatType in ["gamma", "normal-multiplicative"]:
                self.kstat = self.sumw**2 / self.varbeta
                self.betamask = (self.varbeta == 0.0) | (self.kstat == 0.0)
                self.kstat = tf.where(self.betamask, 1.0, self.kstat)
            elif self.binByBinStatType == "normal-additive":
                # precompute decomposition of composite matrix to speed up
                # calculation of profiled beta values
                if self.covarianceFit:
                    sbeta = tf.math.sqrt(self.varbeta[: self.indata.nbins])

                    if self.binByBinStatMode == "lite":
                        sbeta = tf.linalg.LinearOperatorDiag(sbeta)
                        self.betaauxlu = tf.linalg.lu(
                            sbeta @ self.data_cov_inv @ sbeta
                            + tf.eye(
                                self.data_cov_inv.shape[0],
                                dtype=self.data_cov_inv.dtype,
                            )
                        )
                    elif self.binByBinStatMode == "full":
                        varbetasum = tf.reduce_sum(
                            self.varbeta[: self.indata.nbins], axis=1
                        )

                        varbetasum = tf.linalg.LinearOperatorDiag(varbetasum)

                        self.betaauxlu = tf.linalg.lu(
                            varbetasum @ self.data_cov_inv
                            + tf.eye(
                                self.data_cov_inv.shape[0],
                                dtype=self.data_cov_inv.dtype,
                            )
                        )

        self.nexpnom = tf.Variable(
            self.expected_yield(), trainable=False, name="nexpnom"
        )

        # parameter covariance matrix
        self.cov = tf.Variable(
            self.prefit_covariance(
                unconstrained_err=self.prefitUnconstrainedNuisanceUncertainty
            ),
            trainable=False,
            name="cov",
        )

        # determine if problem is linear (ie likelihood is purely quadratic)
        self.is_linear = (
            (self.chisqFit or self.covarianceFit)
            and self.poi_model.is_linear
            and self.indata.symmetric_tensor
            and self.indata.systematic_type == "normal"
            and ((not self.binByBinStat) or self.binByBinStatType == "normal-additive")
        )

    def load_fitresult(self, fitresult_file, fitresult_key):
        # load results from external fit and set postfit value and covariance elements for common parameters
        cov_ext = None
        with h5py.File(fitresult_file, "r") as fext:
            if "x" in fext.keys():
                # fitresult from combinetf
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
            covval = self.cov.numpy()
            covval[np.ix_(idxs, idxs)] = cov_ext[np.ix_(idxs_ext, idxs_ext)]
            self.cov.assign(tf.constant(covval))

    def update_frozen_params(self):
        new_mask_np = np.isin(self.parms, self.frozen_params)

        self.frozen_params_mask.assign(new_mask_np)
        self.frozen_indices = np.where(new_mask_np)[0]

    def freeze_params(self, frozen_parmeter_expressions):
        self.frozen_params.extend(
            match_regexp_params(frozen_parmeter_expressions, self.parms)
        )
        self.update_frozen_params()

    def defreeze_params(self, unfrozen_parmeter_expressions):
        unfrozen_parmeter = match_regexp_params(
            unfrozen_parmeter_expressions, self.parms
        )
        self.frozen_params = [
            x for x in self.frozen_params if x not in unfrozen_parmeter
        ]
        self.update_frozen_params()

    def init_blinding_values(self, unblind_parameter_expressions=[]):
        unblind_parameters = match_regexp_params(
            unblind_parameter_expressions,
            [
                *self.poi_model.pois,
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
        self._blinding_values_poi = np.ones(self.poi_model.npoi, dtype=np.float64)
        for i in range(self.poi_model.npoi):
            param = self.poi_model.pois[i]
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
                np.ones(self.poi_model.npoi, dtype=np.float64)
            )
            self._blinding_offsets_theta.assign(
                np.zeros(self.indata.nsyst, dtype=np.float64)
            )

    def get_theta(self):
        theta = self.x[self.poi_model.npoi : self.poi_model.npoi + self.indata.nsyst]
        theta = tf.where(
            self.frozen_params_mask[
                self.poi_model.npoi : self.poi_model.npoi + self.indata.nsyst
            ],
            tf.stop_gradient(theta),
            theta,
        )
        theta = self.x[self.poi_model.npoi :]
        if self.do_blinding:
            return theta + self._blinding_offsets_theta
        else:
            return theta

    def get_poi(self):
        xpoi = self.x[: self.poi_model.npoi]
        if self.poi_model.allowNegativePOI:
            poi = xpoi
        else:
            poi = tf.square(xpoi)
        poi = tf.where(
            self.frozen_params_mask[: self.poi_model.npoi], tf.stop_gradient(poi), poi
        )
        if self.do_blinding:
            return poi * self._blinding_offsets_poi
        else:
            return poi

    def _default_beta0(self):
        if self.binByBinStatType in ["gamma", "normal-multiplicative"]:
            return tf.ones(self.beta_shape, dtype=self.indata.dtype)
        elif self.binByBinStatType == "normal-additive":
            return tf.zeros(self.beta_shape, dtype=self.indata.dtype)

    def prefit_covariance(self, unconstrained_err=0.0):
        # free parameters are taken to have zero uncertainty for the purposes of prefit uncertainties
        var_poi = tf.zeros([self.poi_model.npoi], dtype=self.indata.dtype)

        # nuisances have their uncertainty taken from the constraint term, but unconstrained nuisances
        # are set to a placeholder uncertainty (zero by default) for the purposes of prefit uncertainties
        var_theta = tf.where(
            self.indata.constraintweights == 0.0,
            unconstrained_err**2,
            tf.math.reciprocal(self.indata.constraintweights),
        )

        invhessianprefit = tf.linalg.diag(tf.concat([var_poi, var_theta], axis=0))
        return invhessianprefit

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

    def set_beta0(self, values):
        self.beta0.assign(values)
        # compute offset for Gamma nll improved numerical precision in minimizatoin
        # the offset is chosen to give the saturated likelihood
        beta0safe = tf.where(
            values == 0.0, tf.constant(1.0, dtype=values.dtype), values
        )
        self.logbeta0.assign(tf.math.log(beta0safe))

    def theta0defaultassign(self):
        self.theta0.assign(self.theta0default)

    def xdefaultassign(self):
        if self.poi_model.npoi == 0:
            self.x.assign(self.theta0)
        else:
            self.x.assign(tf.concat([self.poi_model.xpoidefault, self.theta0], axis=0))

    def beta0defaultassign(self):
        self.set_beta0(self._default_beta0())

    def betadefaultassign(self):
        self.beta.assign(self.beta0)

    def defaultassign(self):
        self.cov.assign(
            self.prefit_covariance(
                unconstrained_err=self.prefitUnconstrainedNuisanceUncertainty
            )
        )
        self.theta0defaultassign()
        if self.binByBinStat:
            self.beta0defaultassign()
            self.betadefaultassign()
        self.xdefaultassign()
        if self.do_blinding:
            self.set_blinding_offsets(False)

    def bayesassign(self):
        # FIXME use theta0 as the mean and constraintweight to scale the width
        if self.poi_model.npoi == 0:
            self.x.assign(
                self.theta0
                + tf.random.normal(shape=self.theta0.shape, dtype=self.theta0.dtype)
            )
        else:
            self.x.assign(
                tf.concat(
                    [
                        self.poi_model.xpoidefault,
                        self.theta0
                        + tf.random.normal(
                            shape=self.theta0.shape, dtype=self.theta0.dtype
                        ),
                    ],
                    axis=0,
                )
            )

        if self.binByBinStat:
            if self.binByBinStatType == "gamma":
                # FIXME this is only valid for beta0=beta=1 (but this should always be the case when throwing toys)
                betagen = (
                    tf.random.gamma(
                        shape=[],
                        alpha=self.kstat * self.beta0 + 1.0,
                        beta=tf.ones_like(self.kstat),
                        dtype=self.beta.dtype,
                    )
                    / self.kstat
                )

                betagen = tf.where(self.kstat == 0.0, 0.0, betagen)
                self.beta.assign(betagen)
            else:
                if self.binByBinStatType == "normal-multiplicative":
                    stddev_beta0 = tf.sqrt(self.varbeta)
                elif self.binByBinStatType == "normal-additive":
                    stddev_beta0 = tf.ones_like(self.beta0)

                self.beta.assign(
                    tf.random.normal(
                        shape=[],
                        mean=self.beta0,
                        stddev=stddev_beta0,
                        dtype=self.beta.dtype,
                    )
                )

    def frequentistassign(self):
        # FIXME use theta as the mean and constraintweight to scale the width
        self.theta0.assign(
            tf.random.normal(shape=self.theta0.shape, dtype=self.theta0.dtype)
        )
        if self.binByBinStat:
            if self.binByBinStatType == "gamma":
                # FIXME this is only valid for beta0=beta=1 (but this should always be the case when throwing toys)
                beta0gen = (
                    tf.random.poisson(
                        shape=[],
                        lam=self.kstat * self.beta,
                        dtype=self.beta.dtype,
                    )
                    / self.kstat
                )

                beta0gen = tf.where(
                    self.kstat == 0.0,
                    tf.constant(0.0, dtype=self.kstat.dtype),
                    beta0gen,
                )
                self.set_beta0(beta0gen)
            else:
                if self.binByBinStatType == "normal-multiplicative":
                    stddev_beta = tf.sqrt(self.varbeta)
                elif self.binByBinStatType == "normal-additive":
                    stddev_beta = tf.ones_like(self.beta)

                self.set_beta0(
                    tf.random.normal(
                        shape=[],
                        mean=self.beta,
                        stddev=stddev_beta,
                        dtype=self.beta.dtype,
                    )
                )

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
        if self.binByBinStat:
            self.betadefaultassign()
        # set likelihood offset
        self.nexpnom.assign(self.expected_yield())

        if randomize_parameters:
            # the special handling of the diagonal case here speeds things up, but is also required
            # in case the prefit covariance has zero for some uncertainties (which is the default
            # for unconstrained nuisances for example) since the multivariate normal distribution
            # requires a positive-definite covariance matrix
            if tfh.is_diag(self.cov):
                self.x.assign(
                    tf.random.normal(
                        shape=[],
                        mean=self.x,
                        stddev=tf.sqrt(tf.linalg.diag_part(self.cov)),
                        dtype=self.x.dtype,
                    )
                )
            else:
                pparms = tfp.distributions.MultivariateNormalTriL(
                    loc=self.x, scale_tril=tf.linalg.cholesky(self.cov)
                )
                self.x.assign(pparms.sample())
            if self.binByBinStat:
                self.beta.assign(
                    tf.random.normal(
                        shape=[],
                        mean=self.beta0,
                        stddev=tf.sqrt(self.varbeta),
                        dtype=self.beta.dtype,
                    )
                )

    def nonprofiled_impacts_parms(self, unconstrained_err=1.0):
        x_tmp = tf.identity(self.x.value())
        x_tmp_tiled = tf.tile(
            tf.reshape(x_tmp, [1, 1, -1]), [len(self.frozen_indices), 2, 1]
        )
        nonprofiled_impacts = tf.Variable(x_tmp_tiled)

        theta0_tmp = tf.identity(self.theta0.value())

        err_theta = tf.where(
            self.indata.constraintweights == 0.0,
            unconstrained_err,
            tf.math.reciprocal(self.indata.constraintweights),
        )

        for i, idx in enumerate(self.frozen_indices):
            logger.info(f"Now at parameter {self.frozen_params[i]}")

            for j, sign in enumerate((1, -1)):
                variation = (
                    sign * err_theta[idx - self.poi_model.npoi]
                    + theta0_tmp[idx - self.poi_model.npoi]
                )
                # vary the non-profile parameter
                self.theta0[idx - self.poi_model.npoi].assign(variation)
                self.x[idx].assign(
                    variation
                )  # this should not be needed but should accelerates the minimization
                # minimize
                self.minimize()
                if self.diagnostics:
                    val, grad, hess = self.loss_val_grad_hess()
                    edmval, cov = tfh.edmval_cov(grad, hess)
                    logger.info(f"edmval: {edmval}")
                # difference w.r.t. nominal fit
                diff = x_tmp - self.x.value()
                nonprofiled_impacts[i, j].assign(diff)
                self.x.assign(x_tmp)

            # back to original value
            self.theta0[idx - self.poi_model.npoi].assign(
                theta0_tmp[idx - self.poi_model.npoi]
            )

        # grouped nonprofiled impacts
        @tf.function
        def envelope(values):
            zeros = tf.zeros(
                (tf.shape(values)[0], tf.shape(values)[-1]), dtype=values.dtype
            )
            vmin = tf.reduce_min(values, axis=1)
            vmax = tf.reduce_max(values, axis=1)
            lower = -tf.sqrt(tf.reduce_sum(tf.minimum(zeros, vmin) ** 2, axis=0))
            upper = tf.sqrt(tf.reduce_sum(tf.maximum(zeros, vmax) ** 2, axis=0))
            return tf.stack([lower, upper])

        impact_group_names = []
        impact_groups = []

        for group, idxs in zip(self.indata.systgroups, self.indata.systgroupidxs):
            frozen_mask = tf.constant(np.isin(self.frozen_indices, idxs))
            frozen_idxs = tf.where(frozen_mask)
            if tf.size(frozen_idxs) > 0:
                selected_impacts = tf.gather(nonprofiled_impacts, frozen_idxs[:, 0])
                group_env = envelope(selected_impacts)
                impact_groups.append(group_env)
                impact_group_names.append(group)

        # Add total envelope
        total_env = envelope(nonprofiled_impacts)
        impact_groups.append(total_env)
        impact_group_names.append(b"Total")

        impact_groups = tf.stack(impact_groups)

        return (
            self.frozen_params,
            nonprofiled_impacts.numpy(),
            impact_group_names,
            impact_groups.numpy(),
        )

    def _compute_impact_group(self, v, idxs):
        cov_reduced = tf.gather(
            self.cov[self.poi_model.npoi :, self.poi_model.npoi :], idxs, axis=0
        )
        cov_reduced = tf.gather(cov_reduced, idxs, axis=1)
        v_reduced = tf.gather(v, idxs, axis=1)
        invC_v = tf.linalg.solve(cov_reduced, tf.transpose(v_reduced))
        v_invC_v = tf.einsum("ij,ji->i", v_reduced, invC_v)
        return tf.sqrt(v_invC_v)

    def _gather_poi_noi_vector(self, v):
        v_poi = v[: self.poi_model.npoi]
        # protection for constained NOIs, set them to 0
        mask = (self.indata.noiidxs >= 0) & (
            self.indata.noiidxs < tf.shape(v[self.poi_model.npoi :])[0]
        )
        safe_idxs = tf.where(mask, self.indata.noiidxs, 0)
        mask = tf.cast(mask, v.dtype)
        mask = tf.reshape(
            mask,
            tf.concat(
                [tf.shape(mask), tf.ones(tf.rank(v) - 1, dtype=tf.int32)], axis=0
            ),
        )
        v_noi = tf.gather(v[self.poi_model.npoi :], safe_idxs) * mask
        v_gathered = tf.concat([v_poi, v_noi], axis=0)
        return v_gathered

    @tf.function
    def impacts_parms(self, hess):
        # impact for poi at index i in covariance matrix from nuisance with index j is C_ij/sqrt(C_jj) = <deltax deltatheta>/sqrt(<deltatheta^2>)
        v = self._gather_poi_noi_vector(self.cov)
        impacts = v / tf.reshape(tf.sqrt(tf.linalg.diag_part(self.cov)), [1, -1])

        nstat = self.poi_model.npoi + self.indata.nsystnoconstraint
        hess_stat = hess[:nstat, :nstat]
        inv_hess_stat = tf.linalg.inv(hess_stat)

        if self.binByBinStat:
            # impact bin-by-bin stat
            val_no_bbb, grad_no_bbb, hess_no_bbb = self.loss_val_grad_hess(
                profile=False
            )

            hess_stat_no_bbb = hess_no_bbb[:nstat, :nstat]
            inv_hess_stat_no_bbb = tf.linalg.inv(hess_stat_no_bbb)
            impacts_data_stat = tf.sqrt(tf.linalg.diag_part(inv_hess_stat_no_bbb))
            impacts_data_stat = self._gather_poi_noi_vector(impacts_data_stat)
            impacts_data_stat = tf.reshape(impacts_data_stat, (-1, 1))

            impacts_bbb_sq = tf.linalg.diag_part(inv_hess_stat - inv_hess_stat_no_bbb)
            impacts_bbb_sq = self._gather_poi_noi_vector(impacts_bbb_sq)
            impacts_bbb = tf.sqrt(tf.nn.relu(impacts_bbb_sq))  # max(0,x)
            impacts_bbb = tf.reshape(impacts_bbb, (-1, 1))
            impacts_grouped = tf.concat([impacts_data_stat, impacts_bbb], axis=1)
        else:
            impacts_data_stat = tf.sqrt(tf.linalg.diag_part(inv_hess_stat))
            impacts_data_stat = self._gather_poi_noi_vector(impacts_data_stat)
            impacts_data_stat = tf.reshape(impacts_data_stat, (-1, 1))
            impacts_grouped = impacts_data_stat

        if len(self.indata.systgroupidxs):
            impacts_grouped_syst = tf.map_fn(
                lambda idxs: self._compute_impact_group(
                    v[:, self.poi_model.npoi :], idxs
                ),
                tf.ragged.constant(self.indata.systgroupidxs, dtype=tf.int32),
                fn_output_signature=tf.TensorSpec(
                    shape=(impacts.shape[0],), dtype=tf.float64
                ),
            )
            impacts_grouped_syst = tf.transpose(impacts_grouped_syst)
            impacts_grouped = tf.concat([impacts_grouped_syst, impacts_grouped], axis=1)

        return impacts, impacts_grouped

    def _compute_global_impact_group(self, d_squared, idxs):
        gathered = tf.gather(d_squared, idxs, axis=-1)
        d_squared_summed = tf.reduce_sum(gathered, axis=-1)
        return tf.sqrt(d_squared_summed)

    def dbetadx_tangents(self, tangent_vector):
        """
        Computes JVP for a single tangent vector (column of cov_dexpdx).
        """
        # Setup Forward Accumulator for this tangent
        with tf.autodiff.ForwardAccumulator(self.x, tangent_vector) as acc:
            _1, _2, beta = self._compute_yields_with_beta(
                profile=True, compute_norm=False, full=False
            )
        return acc.jvp(beta)

    def _compute_global_impacts_beta0_jvp(self, cov_dexpdx, profile=True):
        """
        Computes global impacts from beta parameters via JVP in forward accumulator mode.
        This is fast in case of more beta parameters than explicit parameters (self.x) and 'cov_dexpdx' has only a few columns.
        It should always be more memory efficient
        """
        with tf.GradientTape() as t2:
            t2.watch(self.ubeta)
            with tf.GradientTape() as t1:
                t1.watch(self.ubeta)
                _1, _2, beta = self._compute_yields_with_beta(
                    profile=profile, compute_norm=False, full=False
                )
                lbeta = self._compute_lbeta(beta)
            pdlbetadbeta = t1.gradient(lbeta, self.ubeta)

        # pd2lbetadbeta2 is diagonal so we can use gradient instead of jacobian
        pd2lbetadbeta2_diag = t2.gradient(pdlbetadbeta, self.ubeta)

        # this the cholesky decomposition of pd2lbetadbeta2
        sbeta = tf.linalg.LinearOperatorDiag(
            tf.sqrt(tf.reshape(pd2lbetadbeta2_diag, [-1])), is_self_adjoint=True
        )

        impacts_beta_shape = (*self.beta_shape, cov_dexpdx.shape[-1])
        impacts_beta0 = tf.zeros(shape=impacts_beta_shape, dtype=cov_dexpdx.dtype)

        if profile:
            # dbeta/dx is None if not profiled (no relation)
            tangents = tf.transpose(cov_dexpdx)
            dbetadx_cov_dexpdx = tf.vectorized_map(self.dbetadx_tangents, tangents)

            # flatten all but first axes
            dbetadx_cov_dexpdx = tf.reshape(
                dbetadx_cov_dexpdx, [tf.shape(dbetadx_cov_dexpdx)[0], -1]
            )
            dbetadx_cov_dexpdx = tf.transpose(dbetadx_cov_dexpdx)

            impacts_beta0 += tf.reshape(sbeta @ dbetadx_cov_dexpdx, impacts_beta_shape)

        return impacts_beta0, sbeta

    def _compute_global_impacts_beta0(self, cov_dexpdx, profile=True):
        """
        Computes global impacts from beta parameters in the traditional mode.
        This is fast in case of less beta parameters than explicit parameters (self.x) or 'cov_dexpdx' has many columns.
        """
        with tf.GradientTape(persistent=True) as t2:
            t2.watch([self.x, self.ubeta])
            with tf.GradientTape(persistent=True) as t1:
                t1.watch([self.x, self.ubeta])
                _1, _2, beta = self._compute_yields_with_beta(
                    profile=profile, compute_norm=False, full=False
                )
                lbeta = self._compute_lbeta(beta)
            pdlbetadbeta = t1.gradient(lbeta, self.ubeta)
            dbetadx = t1.jacobian(beta, self.x)
        # pd2lbetadbeta2 is diagonal so we can use gradient instead of jacobian
        pd2lbetadbeta2_diag = t2.gradient(pdlbetadbeta, self.ubeta)

        # this the cholesky decomposition of pd2lbetadbeta2
        sbeta = tf.linalg.LinearOperatorDiag(
            tf.sqrt(tf.reshape(pd2lbetadbeta2_diag, [-1])), is_self_adjoint=True
        )

        impacts_beta_shape = (*self.beta_shape, cov_dexpdx.shape[-1])
        impacts_beta0 = tf.zeros(shape=impacts_beta_shape, dtype=cov_dexpdx.dtype)

        if profile:
            dbetadx_cov_dexpdx = dbetadx @ cov_dexpdx
            dbetadx_cov_dexpdx = tf.reshape(
                dbetadx_cov_dexpdx, [-1, tf.shape(dbetadx_cov_dexpdx)[-1]]
            )

            impacts_beta0 += tf.reshape(sbeta @ dbetadx_cov_dexpdx, impacts_beta_shape)

        return impacts_beta0, sbeta

    def _compute_global_impacts_x0(self, cov_dexpdx):
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                lc = self._compute_lc()
            dlcdx = t1.gradient(lc, self.x)
        # d2lcdx2 is diagonal so we can use gradient instead of jacobian
        d2lcdx2_diag = t2.gradient(dlcdx, self.x)

        # sc is the cholesky decomposition of d2lcdx2
        sc = tf.linalg.LinearOperatorDiag(tf.sqrt(d2lcdx2_diag), is_self_adjoint=True)
        return sc @ cov_dexpdx

    @tf.function
    def global_impacts_parms(self):
        # TODO migrate this to a mapping to avoid the below code which is largely duplicated

        idxs_poi = tf.range(self.poi_model.npoi, dtype=tf.int64)
        idxs_noi = tf.constant(
            self.poi_model.npoi + self.indata.noiidxs, dtype=tf.int64
        )
        idxsout = tf.concat([idxs_poi, idxs_noi], axis=0)

        dexpdx = tf.one_hot(idxsout, depth=self.cov.shape[0], dtype=self.cov.dtype)

        cov_dexpdx = tf.matmul(self.cov, dexpdx, transpose_b=True)

        var_total = tf.linalg.diag_part(self.cov)
        var_total = tf.gather(var_total, idxsout)

        if self.binByBinStat:
            if self.globalImpactsFromJVP:
                impacts_beta0, _ = self._compute_global_impacts_beta0_jvp(cov_dexpdx)
            else:
                impacts_beta0, _ = self._compute_global_impacts_beta0(cov_dexpdx)

            var_beta0 = tf.reduce_sum(tf.square(impacts_beta0), axis=0)

            if self.binByBinStatMode == "full":
                impacts_beta0_process = tf.sqrt(var_beta0)
                var_beta0 = tf.reduce_sum(var_beta0, axis=0)

            impacts_beta0_total = tf.sqrt(var_beta0)

        impacts_x0 = self._compute_global_impacts_x0(cov_dexpdx)
        impacts_theta0 = impacts_x0[self.poi_model.npoi :]

        impacts_theta0 = tf.transpose(impacts_theta0)
        impacts = impacts_theta0

        impacts_theta0_sq = tf.square(impacts_theta0)
        var_theta0 = tf.reduce_sum(impacts_theta0_sq, axis=-1)

        var_nobs = var_total - var_theta0

        if self.binByBinStat:
            var_nobs -= var_beta0

        impacts_nobs = tf.sqrt(var_nobs)

        if self.binByBinStat:
            impacts_grouped = tf.stack([impacts_nobs, impacts_beta0_total], axis=-1)
            if self.binByBinStatMode == "full":
                impacts_grouped = tf.concat(
                    [impacts_grouped, tf.transpose(impacts_beta0_process)], axis=-1
                )

        else:
            impacts_grouped = impacts_nobs[..., None]

        if len(self.indata.systgroupidxs):
            impacts_grouped_syst = tf.map_fn(
                lambda idxs: self._compute_global_impact_group(impacts_theta0_sq, idxs),
                tf.ragged.constant(self.indata.systgroupidxs, dtype=tf.int64),
                fn_output_signature=tf.TensorSpec(
                    shape=(impacts_theta0_sq.shape[0],), dtype=impacts_theta0_sq.dtype
                ),
            )
            impacts_grouped_syst = tf.transpose(impacts_grouped_syst)
            impacts_grouped = tf.concat([impacts_grouped_syst, impacts_grouped], axis=1)

        return impacts, impacts_grouped

    def _pd2ldbeta2(self, profile=False):
        with tf.GradientTape(watch_accessed_variables=False) as t2:
            t2.watch([self.ubeta])
            with tf.GradientTape(watch_accessed_variables=False) as t1:
                t1.watch([self.ubeta])
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

            pdldbeta = t1.gradient(val, self.ubeta)
        if self.covarianceFit and profile:
            pd2ldbeta2_matrix = t2.jacobian(pdldbeta, self.ubeta)
            pd2ldbeta2 = tf.linalg.LinearOperatorFullMatrix(
                pd2ldbeta2_matrix, is_self_adjoint=True
            )
        else:
            # pd2ldbeta2 is diagonal, so we can use gradient instead of jacobian
            pd2ldbeta2 = t2.gradient(pdldbeta, self.ubeta)
        return pd2ldbeta2

    def _dxdvars(self):
        with tf.GradientTape() as t2:
            t2.watch([self.theta0, self.nobs, self.beta0])
            with tf.GradientTape() as t1:
                t1.watch([self.theta0, self.nobs, self.beta0])
                val = self._compute_loss()
            grad = t1.gradient(val, self.x)
        pd2ldxdtheta0, pd2ldxdnobs, pd2ldxdbeta0 = t2.jacobian(
            grad, [self.theta0, self.nobs, self.beta0], unconnected_gradients="zero"
        )

        # cov is inverse hesse, thus cov ~ d2xd2l
        dxdtheta0 = -self.cov @ pd2ldxdtheta0
        dxdnobs = -self.cov @ pd2ldxdnobs
        dxdbeta0 = -self.cov @ tf.reshape(pd2ldxdbeta0, [pd2ldxdbeta0.shape[0], -1])

        return dxdtheta0, dxdnobs, dxdbeta0

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

        if self.binByBinStat:
            dvars = [self.x, self.ubeta]
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
                if self.binByBinStatType == "normal-additive":
                    pd2ldbeta2_pdexpdbeta = pdexpdbeta / pd2ldbeta2[None, :]
                else:
                    pd2ldbeta2_pdexpdbeta = tf.where(
                        self.betamask[None, :],
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
            # the fully general contribution to the covariance matrix
            # for a factorized likelihood L = sum_i L_i can be written as
            # cov_i = dexpdx @ cov_x @ d2L_i/dx2 @ cov_x @ dexpdx.T
            # This is totally general and always adds up to the total covariance matrix

            # This can be factorized into impacts only if the individual contributions
            # are rank 1.  This is not the case in general for the data stat uncertainties,
            # in particular where postfit nexpected != nobserved and nexpected is not a linear
            # function of the poi's and nuisance parameters x

            # For the systematic and MC stat uncertainties this is equivalent to the
            # more conventional global impact calculation (and without needing to insert the uncertainty on
            # the global observables "by hand", which can be non-trivial beyond the Gaussian case)

            if self.binByBinStat:
                if self.globalImpactsFromJVP:
                    impacts_beta0, sbeta = self._compute_global_impacts_beta0_jvp(
                        cov_dexpdx, profile
                    )
                else:
                    impacts_beta0, sbeta = self._compute_global_impacts_beta0(
                        cov_dexpdx, profile
                    )

                if pdexpdbeta is not None:
                    impacts_beta0 += tf.reshape(
                        sbeta @ pd2ldbeta2_pdexpdbeta, impacts_beta0.shape
                    )

                var_beta0 = tf.reduce_sum(tf.square(impacts_beta0), axis=0)
                if self.binByBinStatMode == "full":
                    impacts_beta0_process = tf.sqrt(var_beta0)
                    var_beta0 = tf.reduce_sum(var_beta0, axis=0)

                impacts_beta0_total = tf.sqrt(var_beta0)

            # protect against inconsistency
            # FIXME this should be handled more generally e.g. through modification of
            # the constraintweights for prefit vs postfit, though special handling of the zero
            # uncertainty case would still be needed
            if (not profile) and self.prefitUnconstrainedNuisanceUncertainty != 0.0:
                raise NotImplementedError(
                    "Global impacts calculation not implemented for prefit case where prefitUnconstrainedNuisanceUncertainty != 0."
                )

            impacts_x0 = self._compute_global_impacts_x0(cov_dexpdx)
            impacts_theta0 = impacts_x0[self.poi_model.npoi :]

            impacts_theta0 = tf.transpose(impacts_theta0)
            impacts = impacts_theta0

            impacts_theta0_sq = tf.square(impacts_theta0)
            var_theta0 = tf.reduce_sum(impacts_theta0_sq, axis=-1)

            var_nobs = expvar_flat - var_theta0

            if self.binByBinStat:
                var_nobs -= var_beta0

            impacts_nobs = tf.sqrt(var_nobs)

            if self.binByBinStat:
                impacts_grouped = tf.stack([impacts_nobs, impacts_beta0_total], axis=-1)
                if self.binByBinStatMode == "full":
                    impacts_grouped = tf.concat(
                        [impacts_grouped, tf.transpose(impacts_beta0_process)], axis=-1
                    )

            else:
                impacts_grouped = impacts_nobs[..., None]

            if len(self.indata.systgroupidxs):
                impacts_grouped_syst = tf.map_fn(
                    lambda idxs: self._compute_global_impact_group(
                        impacts_theta0_sq, idxs
                    ),
                    tf.ragged.constant(self.indata.systgroupidxs, dtype=tf.int64),
                    fn_output_signature=tf.TensorSpec(
                        shape=(impacts_theta0_sq.shape[0],),
                        dtype=impacts_theta0_sq.dtype,
                    ),
                )
                impacts_grouped_syst = tf.transpose(impacts_grouped_syst)

                impacts_grouped = tf.concat(
                    [impacts_grouped_syst, impacts_grouped], axis=-1
                )

            impacts = tf.reshape(impacts, [*expvar.shape, impacts.shape[-1]])
            impacts_grouped = tf.reshape(
                impacts_grouped, [*expvar.shape, impacts_grouped.shape[-1]]
            )
        else:
            impacts = None
            impacts_grouped = None

        return expected, expvar, expcov, impacts, impacts_grouped

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

    def _compute_yields_noBBB(self, full=True):
        # full: compute yields inclduing masked channels
        poi = self.get_poi()
        theta = self.get_theta()

        rnorm = self.poi_model.compute(poi, full)

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
            logsnorm = tf.sparse.sparse_dense_matmul(self.indata.logk, mthetaalpha)
            logsnorm = tf.squeeze(logsnorm, -1)

            if self.indata.systematic_type == "log_normal":
                snorm = tf.exp(logsnorm)
                snormnorm_sparse = self.indata.norm.with_values(
                    snorm * self.indata.norm.values
                )
            elif self.indata.systematic_type == "normal":
                snormnorm_sparse = self.indata.norm * rnorm
                snormnorm_sparse = snormnorm_sparse.with_values(
                    snormnorm_sparse.values + logsnorm
                )

            if not full and self.indata.nbinsmasked:
                snormnorm_sparse = tfh.simple_sparse_slice0end(
                    snormnorm_sparse, self.indata.nbins
                )

            if self.indata.systematic_type == "log_normal":
                snormnorm = tf.sparse.to_dense(snormnorm_sparse)
                normcentral = rnorm * snormnorm
            elif self.indata.systematic_type == "normal":
                normcentral = tf.sparse.to_dense(snormnorm_sparse)

            nexpcentral = tf.reduce_sum(normcentral, axis=-1)
        else:
            if full or self.indata.nbinsmasked == 0:
                nbins = self.indata.nbinsfull
                logk = self.indata.logk
                norm = self.indata.norm
            else:
                nbins = self.indata.nbins
                logk = self.indata.logk[:nbins]
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
        nexp, norm = self._compute_yields_noBBB(full=full)

        if self.binByBinStat:
            if profile:
                # analytic solution for profiled barlow-beeston lite parameters for each combination
                # of likelihood and uncertainty form

                nexp_profile = nexp[: self.indata.nbins]
                beta0 = self.beta0[: self.indata.nbins]

                if self.chisqFit:
                    if self.binByBinStatType == "gamma":
                        kstat = self.kstat[: self.indata.nbins]

                        abeta = nexp_profile**2
                        bbeta = kstat * self.varnobs - nexp_profile * self.nobs
                        cbeta = -kstat * self.varnobs * beta0
                        beta = solve_quad_eq(abeta, bbeta, cbeta)

                        betamask = self.betamask[: self.indata.nbins]
                        beta = tf.where(betamask, beta0, beta)
                    elif self.binByBinStatType == "normal-multiplicative":
                        kstat = self.kstat[: self.indata.nbins]
                        betamask = self.betamask[: self.indata.nbins]
                        if self.binByBinStatMode == "lite":
                            beta = (
                                nexp_profile * self.nobs / self.varnobs + kstat * beta0
                            ) / (kstat + nexp_profile * nexp_profile / self.varnobs)

                            beta = tf.where(betamask, beta0, beta)

                        elif self.binByBinStatMode == "full":
                            norm_profile = norm[: self.indata.nbins]
                            n2kstat = tf.square(norm_profile) / kstat
                            n2kstat = tf.where(
                                betamask,
                                tf.constant(0.0, dtype=self.indata.dtype),
                                n2kstat,
                            )
                            n2kstatsum = tf.reduce_sum(n2kstat, axis=-1)

                            nbeta = (
                                self.nobs / self.varnobs * n2kstatsum
                                + tf.reduce_sum(norm_profile * beta0, axis=-1)
                            ) / (1 + 1 / self.varnobs * n2kstatsum)
                            beta = (
                                beta0
                                + (1 / self.varnobs * (self.nobs - nbeta))[..., None]
                                * norm_profile
                                / kstat
                            )
                            beta = tf.where(betamask, beta0, beta)
                    elif self.binByBinStatType == "normal-additive":
                        varbeta = self.varbeta[: self.indata.nbins]
                        sbeta = tf.math.sqrt(varbeta)
                        if self.binByBinStatMode == "lite":
                            beta = (
                                sbeta * (self.nobs - nexp_profile)
                                + self.varnobs * beta0
                            ) / (self.varnobs + varbeta)
                        elif self.binByBinStatMode == "full":
                            varbetasum = tf.reduce_sum(varbeta, axis=-1)
                            nbeta = (
                                tf.reduce_sum(sbeta * beta0, axis=-1)
                                + varbetasum / self.varnobs * (self.nobs - nexp_profile)
                            ) / (1 + varbetasum / self.varnobs)
                            beta = (
                                beta0
                                - sbeta
                                * ((nexp_profile + nbeta - self.nobs) / self.varnobs)[
                                    :, None
                                ]
                            )
                elif self.covarianceFit:
                    if self.binByBinStatType == "normal-multiplicative":
                        kstat = self.kstat[: self.indata.nbins]
                        betamask = self.betamask[: self.indata.nbins]
                        if self.binByBinStatMode == "lite":

                            nexp_profile_m = tf.linalg.LinearOperatorDiag(nexp_profile)
                            A = (
                                nexp_profile_m @ self.data_cov_inv @ nexp_profile_m
                                + tf.linalg.diag(kstat)
                            )
                            b = (
                                nexp_profile_m
                                @ (self.data_cov_inv @ self.nobs[:, None])
                                + (kstat * beta0)[:, None]
                            )

                            # Cholesky solve sometimes does not give corret result
                            # chol = tf.linalg.cholesky(A)
                            # beta = tf.linalg.cholesky_solve(chol, b)

                            beta = tf.linalg.solve(A, b)

                            beta = tf.squeeze(beta, axis=-1)
                            beta = tf.where(betamask, beta0, beta)
                        elif self.binByBinStatMode == "full":
                            norm_profile = norm[: self.indata.nbins]

                            # first solve sum of processes
                            nbeta0 = tf.reduce_sum(norm_profile * beta0, axis=1)
                            n2kstat = tf.square(norm_profile) / kstat
                            n2kstat = tf.where(
                                betamask,
                                tf.constant(0.0, dtype=self.indata.dtype),
                                n2kstat,
                            )
                            n2kstatsum = tf.reduce_sum(n2kstat, axis=1)
                            n2kstatsum_m = tf.linalg.LinearOperatorDiag(n2kstatsum)

                            A = n2kstatsum_m @ self.data_cov_inv + tf.eye(
                                self.data_cov_inv.shape[0],
                                dtype=self.data_cov_inv.dtype,
                            )
                            b = (
                                n2kstatsum_m @ self.data_cov_inv @ (self.nobs[:, None])
                                + nbeta0[:, None]
                            )

                            # Cholesky solve sometimes does not give corret result
                            # chol = tf.linalg.cholesky(A)
                            # nbeta = tf.linalg.cholesky_solve(chol, b)

                            nbeta = tf.linalg.solve(A, b)

                            # now solve for beta [nprocesses x nbins]
                            beta = beta0 - norm_profile / kstat * (
                                self.data_cov_inv @ (nbeta - self.nobs[:, None])
                            )
                            beta = tf.where(betamask, beta0, beta)
                    elif self.binByBinStatType == "normal-additive":
                        varbeta = self.varbeta[: self.indata.nbins]
                        sbeta = tf.math.sqrt(varbeta)
                        if self.binByBinStatMode == "lite":
                            sbeta_m = tf.linalg.LinearOperatorDiag(sbeta)
                            beta = tf.linalg.lu_solve(
                                *self.betaauxlu,
                                sbeta_m
                                @ self.data_cov_inv
                                @ ((self.nobs - nexp_profile)[:, None])
                                + beta0[:, None],
                            )
                            beta = tf.squeeze(beta, axis=-1)
                        elif self.binByBinStatMode == "full":
                            # first solve for sum of processes
                            sbetabeta0sum = tf.reduce_sum(sbeta * beta0, axis=1)
                            varbetasum = tf.reduce_sum(varbeta, axis=1)
                            varbetasum = tf.linalg.LinearOperatorDiag(varbetasum)

                            nbeta = tf.linalg.lu_solve(
                                *self.betaauxlu,
                                varbetasum
                                @ self.data_cov_inv
                                @ ((self.nobs - nexp_profile)[:, None])
                                + sbetabeta0sum[:, None],
                            )
                            # second solve for beta
                            beta = beta0 - sbeta * (
                                self.data_cov_inv
                                @ (nbeta + nexp_profile[:, None] - self.nobs[:, None])
                            )
                else:
                    if self.binByBinStatType == "gamma":
                        kstat = self.kstat[: self.indata.nbins]
                        betamask = self.betamask[: self.indata.nbins]

                        beta = (self.nobs + kstat * beta0) / (nexp_profile + kstat)
                        beta = tf.where(betamask, beta0, beta)
                    elif self.binByBinStatType == "normal-multiplicative":
                        kstat = self.kstat[: self.indata.nbins]
                        betamask = self.betamask[: self.indata.nbins]
                        if self.binByBinStatMode == "lite":
                            abeta = kstat
                            bbeta = nexp_profile - beta0 * kstat
                            cbeta = -self.nobs
                            beta = solve_quad_eq(abeta, bbeta, cbeta)
                            beta = tf.where(betamask, beta0, beta)
                        elif self.binByBinStatMode == "full":
                            norm_profile = norm[: self.indata.nbins]
                            n2kstat = tf.square(norm_profile) / kstat
                            n2kstat = tf.where(
                                betamask,
                                tf.constant(0.0, dtype=self.indata.dtype),
                                n2kstat,
                            )
                            pbeta = tf.reduce_sum(
                                n2kstat - beta0 * norm_profile, axis=-1
                            )
                            qbeta = -self.nobs * tf.reduce_sum(n2kstat, axis=-1)
                            nbeta = solve_quad_eq(1, pbeta, qbeta)
                            beta = (
                                beta0
                                + (self.nobs / nbeta - 1)[..., None]
                                * norm_profile
                                / kstat
                            )
                            beta = tf.where(betamask, beta0, beta)
                    elif self.binByBinStatType == "normal-additive":
                        varbeta = self.varbeta[: self.indata.nbins]
                        sbeta = tf.math.sqrt(varbeta)
                        if self.binByBinStatMode == "lite":
                            abeta = sbeta
                            abeta = tf.where(
                                varbeta == 0.0,
                                tf.constant(1.0, dtype=varbeta.dtype),
                                abeta,
                            )
                            bbeta = varbeta + nexp_profile - sbeta * beta0
                            cbeta = (
                                sbeta * (nexp_profile - self.nobs)
                                - nexp_profile * beta0
                            )
                            beta = solve_quad_eq(abeta, bbeta, cbeta)
                            beta = tf.where(varbeta == 0.0, beta0, beta)
                        elif self.binByBinStatMode == "full":
                            norm_profile = norm[: self.indata.nbins]

                            qbeta = -self.nobs * tf.reduce_sum(varbeta, axis=-1)
                            pbeta = tf.reduce_sum(
                                varbeta - sbeta * beta0 - norm_profile, axis=-1
                            )
                            nbeta = solve_quad_eq(1, pbeta, qbeta)

                            beta = beta0 + (self.nobs / nbeta - 1)[..., None] * sbeta

                if self.indata.nbinsmasked:
                    beta = tf.concat([beta, self.beta0[self.indata.nbins :]], axis=0)
            else:
                beta = self.beta

            # Add dummy tensor to allow convenient differentiation by beta even when profiling
            beta = beta + self.ubeta

            betasel = beta[: nexp.shape[0]]

            if self.binByBinStatType in ["gamma", "normal-multiplicative"]:
                betamask = self.betamask[: nexp.shape[0]]
                if self.binByBinStatMode == "full":

                    if self.indata.betavar is not None and full:
                        # apply beta variations as normal scaling
                        n0 = self.indata.norm
                        sbeta = tf.math.sqrt(self.kstat[: self.indata.nbins])
                        dbeta = sbeta * (betasel[: self.indata.nbins] - 1)
                        dbeta = tf.where(
                            betamask[: self.indata.nbins], tf.zeros_like(dbeta), dbeta
                        )
                        var = tf.einsum("ijk,jk->ik", self.indata.betavar, dbeta)
                        safe_n0 = tf.where(
                            n0 > 0, n0, 1.0
                        )  # Use 1.0 as a dummy to avoid div by zero
                        ratio = var / safe_n0
                        norm = tf.where(n0 > 0, norm * (1 + ratio), norm)

                    norm = tf.where(betamask, norm, betasel * norm)
                    nexp = tf.reduce_sum(norm, -1)
                else:
                    nexp = tf.where(betamask, nexp, nexp * betasel)
                    if compute_norm:
                        norm = tf.where(
                            betamask[..., None], norm, betasel[..., None] * norm
                        )
            elif self.binByBinStatType == "normal-additive":
                varbeta = self.varbeta[: nexp.shape[0]]
                sbeta = tf.math.sqrt(varbeta)
                if self.binByBinStatMode == "full":
                    norm = norm + sbeta * betasel
                    nexp = tf.reduce_sum(norm, -1)
                else:
                    nexpnorm = nexp[..., None]
                    nexp = nexp + sbeta * betasel
                    if compute_norm:
                        # distribute the change in yields proportionally across processes
                        norm = (
                            norm
                            + sbeta[..., None] * betasel[..., None] * norm / nexpnorm
                        )
        else:
            beta = None

        return nexp, norm, beta

    @tf.function
    def _profile_beta(self):
        nexp, norm, beta = self._compute_yields_with_beta(full=False)
        self.beta.assign(beta)

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

        with tf.GradientTape() as t:
            t.watch([self.theta0, self.nobs, self.beta0])
            expected = self._compute_expected(
                fun,
                inclusive=True,
                profile=True,
                full=False,
                need_observables=True,
            )
            observed = fun(None, self.nobs)
            residuals = expected - observed

            residuals_flat = tf.reshape(residuals, (-1,))
        pdresdx, pdresdtheta0, pdresdnobs, pdresdbeta0 = t.jacobian(
            residuals_flat,
            [self.x, self.theta0, self.nobs, self.beta0],
            unconnected_gradients="zero",
        )

        # apply chain rule to take into account correlations with the fit parameters
        dxdtheta0, dxdnobs, dxdbeta0 = self._dxdvars()

        dresdtheta0 = pdresdtheta0 + pdresdx @ dxdtheta0
        dresdnobs = pdresdnobs + pdresdx @ dxdnobs
        dresdbeta0 = (
            tf.reshape(pdresdbeta0, [pdresdbeta0.shape[0], -1]) + pdresdx @ dxdbeta0
        )

        var_theta0 = tf.where(
            self.indata.constraintweights == 0.0,
            tf.zeros_like(self.indata.constraintweights),
            tf.math.reciprocal(self.indata.constraintweights),
        )

        res_cov = dresdtheta0 @ (var_theta0[:, None] * tf.transpose(dresdtheta0))

        if self.covarianceFit:
            res_cov_stat = dresdnobs @ tf.linalg.solve(
                self.data_cov_inv, tf.transpose(dresdnobs)
            )
        else:
            res_cov_stat = dresdnobs @ (self.nobs[:, None] * tf.transpose(dresdnobs))

        res_cov += res_cov_stat

        if self.binByBinStat:
            pd2ldbeta2 = self._pd2ldbeta2(profile=False)

            with tf.GradientTape() as t2:
                t2.watch([self.ubeta, self.beta0])
                with tf.GradientTape() as t1:
                    t1.watch([self.ubeta, self.beta0])
                    _1, _2, beta = self._compute_yields_with_beta(
                        profile=False, compute_norm=False, full=False
                    )
                    lbeta = self._compute_lbeta(beta)

                dlbetadbeta = t1.gradient(lbeta, self.ubeta)
            pd2lbetadbetadbeta0 = t2.gradient(dlbetadbeta, self.beta0)
            var_beta0 = pd2ldbeta2 / pd2lbetadbetadbeta0**2

            if self.binByBinStatType in ["gamma", "normal-multiplicative"]:
                var_beta0 = tf.where(self.betamask, tf.zeros_like(var_beta0), var_beta0)

            res_cov_BBB = dresdbeta0 @ (
                tf.reshape(var_beta0, [-1])[:, None] * tf.transpose(dresdbeta0)
            )
            res_cov += res_cov_BBB

        return residuals, res_cov

    def _residuals(self, fun, fun_data):
        data, _0, data_cov = fun_data(self.nobs, self.data_cov_inv)
        pred, _0, pred_cov, _1, _2 = self._expected_with_variance(
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
        compute_variations=False,
        correlated_variations=False,
        profile=True,
        compute_chi2=False,
    ):

        if compute_variations and (
            compute_variance or compute_cov or compute_global_impacts
        ):
            raise NotImplementedError()

        fun = mapping.compute_flat if inclusive else mapping.compute_flat_per_process

        aux = [None] * 4
        if compute_cov or compute_variance or compute_global_impacts:
            exp, exp_var, exp_cov, exp_impacts, exp_impacts_grouped = (
                self.expected_with_variance(
                    fun,
                    profile=profile,
                    compute_cov=compute_cov,
                    compute_global_impacts=compute_global_impacts,
                    need_observables=mapping.need_observables,
                    inclusive=inclusive and not mapping.need_processes,
                )
            )
            aux = [exp_var, exp_cov, exp_impacts, exp_impacts_grouped]
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
        res, _ = self._compute_yields_noBBB(full=full)
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
        if self.binByBinStat:
            beta0 = self.beta0
            if self.binByBinStatType == "gamma":
                kstat = self.kstat

                betasafe = tf.where(
                    beta0 == 0.0, tf.constant(1.0, dtype=beta.dtype), beta
                )
                logbeta = tf.math.log(betasafe)

                if full_nll:
                    # constant terms
                    lgammaalpha = tf.math.lgamma(kstat * beta0)
                    alphalntheta = -kstat * beta0 * tf.math.log(kstat)

                    lbeta = (
                        -kstat * beta0 * logbeta
                        + kstat * beta
                        + lgammaalpha
                        + alphalntheta
                    )
                else:
                    lbeta = -kstat * beta0 * (logbeta - self.logbeta0) + kstat * (
                        beta - beta0
                    )
            elif self.binByBinStatType == "normal-multiplicative":
                kstat = self.kstat
                betamask = self.betamask
                lbeta = tf.where(
                    betamask,
                    tf.constant(0.0, dtype=beta.dtype),
                    0.5 * tf.square(beta - beta0) * kstat,
                )
                if full_nll:
                    raise NotImplementedError()

            elif self.binByBinStatType == "normal-additive":
                lbeta = 0.5 * tf.square(beta - beta0)

                if full_nll:
                    # TODO: verify
                    sigma2 = self.varbeta / tf.square(self.sumw)

                    # normalization factor for normal distribution: log(1/sqrt(2*pi)) = -0.9189385332046727
                    lbeta = (
                        lbeta
                        + tf.cast(tf.shape(sigma2), tf.float64) * 0.9189385332046727
                        + 0.5 * tf.math.log(sigma2)
                    )

            return tf.reduce_sum(lbeta)

        return None

    def _compute_nll_components(self, profile=True, full_nll=False):
        nexpfullcentral, _, beta = self._compute_yields_with_beta(
            profile=profile,
            compute_norm=False,
            full=False,
        )

        nexp = nexpfullcentral

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

        lc = self._compute_lc(full_nll)

        lbeta = self._compute_lbeta(beta, full_nll)

        return ln, lc, lbeta, beta

    def _compute_nll(self, profile=True, full_nll=False):
        ln, lc, lbeta, beta = self._compute_nll_components(
            profile=profile, full_nll=full_nll
        )
        l = ln + lc

        if lbeta is not None:
            l = l + lbeta

        return l

    def _compute_loss(self, profile=True):
        l = self._compute_nll(profile=profile)
        return l

    @tf.function
    def loss_val(self):
        val = self._compute_loss()
        return val

    @tf.function
    def loss_val_grad(self):
        with tf.GradientTape() as t:
            val = self._compute_loss()
        grad = t.gradient(val, self.x)
        return val, grad

    # FIXME in principle this version of the function is preferred
    # but seems to introduce some small numerical non-reproducibility
    @tf.function
    def loss_val_grad_hessp_fwdrev(self, p):
        p = tf.stop_gradient(p)
        with tf.autodiff.ForwardAccumulator(self.x, p) as acc:
            with tf.GradientTape() as grad_tape:
                val = self._compute_loss()
            grad = grad_tape.gradient(val, self.x)
        hessp = acc.jvp(grad)
        return val, grad, hessp

    @tf.function
    def loss_val_grad_hessp_revrev(self, p):
        p = tf.stop_gradient(p)
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                val = self._compute_loss()
            grad = t1.gradient(val, self.x)
        hessp = t2.gradient(grad, self.x, output_gradients=p)
        return val, grad, hessp

    loss_val_grad_hessp = loss_val_grad_hessp_revrev

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
            t2.watch(self.ubeta)
            with tf.GradientTape() as t1:
                t1.watch(self.ubeta)
                val = self._compute_loss(profile=profile)
            grad = t1.gradient(val, self.ubeta)
        hess = t2.jacobian(grad, self.ubeta)

        grad = tf.reshape(grad, [-1])
        hess = tf.reshape(hess, [grad.shape[0], grad.shape[0]])

        betamask = ~tf.reshape(self.betamask, [-1])
        grad = grad[betamask]
        hess = tf.boolean_mask(hess, betamask, axis=0)
        hess = tf.boolean_mask(hess, betamask, axis=1)

        return val, grad, hess

    def minimize(self):
        if self.is_linear:
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

            callback = None
        else:

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

    def nll_scan(self, param, scan_range, scan_points, use_prefit=False):
        # make a likelihood scan for a single parameter
        # assuming the likelihood is minimized

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

                self.x.assign(tf.tensor_scatter_nd_update(self.x, [[idx]], [ixval]))

                def scipy_loss(xval):
                    self.x.assign(xval)
                    val, grad = self.loss_val_grad()
                    grad = grad.numpy()
                    grad[idx] = 0  # Zero out gradient for the frozen parameter
                    return val.numpy(), grad

                def scipy_hessp(xval, pval):
                    self.x.assign(xval)
                    pval[idx] = (
                        0  # Ensure the perturbation does not affect frozen parameter
                    )
                    p = tf.convert_to_tensor(pval)
                    val, grad, hessp = self.loss_val_grad_hessp(p)
                    hessp = hessp.numpy()
                    # TODO: worth testing modifying the loss/grad/hess functions to imply 1
                    # for the corresponding hessian element instead of 0,
                    # since this might allow the minimizer to converge more efficiently
                    hessp[idx] = (
                        0  # Zero out Hessian-vector product at the frozen index
                    )
                    return hessp

                res = scipy.optimize.minimize(
                    scipy_loss,
                    self.x,
                    method="trust-krylov",
                    jac=True,
                    hessp=scipy_hessp,
                )
                if res["success"]:
                    dnlls[nscans // 2 + sign * i] = (
                        self.reduced_nll().numpy() - nll_best
                    )
                    scan_vals[nscans // 2 + sign * i] = ixval

            # reset x to original state
            self.x.assign(xval)

        return scan_vals, dnlls

    def nll_scan2D(self, param_tuple, scan_range, scan_points, use_prefit=False):

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

            # print(f"i={i}, j={j}, r={r} drow={drow}, dcol={dcol} | ix={ix}, iy={iy}")

            self.x.assign(
                tf.tensor_scatter_nd_update(
                    self.x, [[idx0], [idx1]], [x_scans[ix], y_scans[iy]]
                )
            )

            def scipy_loss(xval):
                self.x.assign(xval)
                val, grad = self.loss_val_grad()
                grad = grad.numpy()
                grad[idx0] = 0
                grad[idx1] = 0
                return val.numpy(), grad

            def scipy_hessp(xval, pval):
                self.x.assign(xval)
                pval[idx0] = 0
                pval[idx1] = 0
                p = tf.convert_to_tensor(pval)
                val, grad, hessp = self.loss_val_grad_hessp(p)
                hessp = hessp.numpy()
                hessp[idx0] = 0
                hessp[idx1] = 0

                if np.allclose(hessp, 0, atol=1e-8):
                    return np.zeros_like(hessp)

                return hessp

            res = scipy.optimize.minimize(
                scipy_loss,
                self.x,
                method="trust-krylov",
                jac=True,
                hessp=scipy_hessp,
            )

            if res["success"]:
                dnlls[ix, iy] = self.reduced_nll().numpy() - nll_best

        self.x.assign(xval)
        return x_scans, y_scans, dnlls

    def contour_scan(self, param, nll_min, q=1, signs=[-1, 1], fun=None):

        def scipy_loss(x):
            self.x.assign(x)
            val = self.loss_val()
            loss = val.numpy() - nll_min - 0.5 * q
            return loss[None,]

        def scipy_grad(x):
            self.x.assign(x)
            val, grad = self.loss_val_grad()
            return grad.numpy()[None,]

        def scipy_hess(x, v):
            self.x.assign(x)
            val, grad, hess = self.loss_val_grad_hess()
            return v[0] * hess.numpy()

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
        xval_init = xval.numpy()

        idx = np.where(self.parms.astype(str) == param)[0][0]
        x0 = xval[idx]

        # initial guess from covariance
        initial_fit = False

        xup = xval[idx] + (self.cov[idx, idx] * q) ** 0.5
        xdn = xval[idx] - (self.cov[idx, idx] * q) ** 0.5

        for i, sign in enumerate(signs):
            # Objective function and its derivatives
            if sign == -1:
                xval_init[idx] = xdn
            else:
                xval_init[idx] = xup

            if initial_fit:
                # perform initial fit where contour is expected
                self.x.assign(xval_init)
                self.freeze_params(param)
                self.minimize()
                self.defreeze_params(param)

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
                    "maxiter": 50000,
                    "xtol": 1e-10,
                    "gtol": 1e-10,
                    # "barrier_tol": 1e-10,
                },
                **opt,
            )

            logger.info(f"Success: {res.success}")
            logger.debug(f"Status: {res.status}")
            if not res.success:
                logger.warning(f"Message: {res.message}")
                logger.warning(f"Optimality (gtol): {res.optimality}")
                logger.warning(f"Constraint Violation: {res.constr_violation}")
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
