"""Bin-by-bin (BBB) statistical treatment for the rabbit fitter.

Owns all β-related state (TF Variables and precomputed tensors) and
exposes a small API used by :class:`rabbit.fitter.Fitter`. See the
package docstring in :mod:`rabbit.bbstat` for the high-level layout.
"""

import tensorflow as tf
from wums import logging

from rabbit.bbstat.formulas import solve_quad_eq

logger = logging.child_logger(__name__)


VALID_BIN_BY_BIN_STAT_TYPES = ["gamma", "normal-additive", "normal-multiplicative"]


class BinByBinStat:
    """Encapsulates bin-by-bin statistical treatment.

    Constructed once by :class:`rabbit.fitter.Fitter`. Owns β/β0/log β0/uβ
    TF Variables, the per-bin (or per-(bin, proc)) ``kstat`` / ``betamask``
    / ``proc_zero_var_mask`` tensors, and any precomputed LU
    decompositions used by the covarianceFit profile.

    The flag :attr:`enabled` mirrors ``not options.noBinByBinStat``;
    callers should branch on it. ``proc_zero_var_mask`` flags
    ``(bin, process)`` entries with ``sumw2 == 0`` — those have no
    bin-by-bin statistical uncertainty, so β passes through them
    unchanged in lite mode rather than scaling them along with the
    finite-variance contributions.
    """

    valid_types = VALID_BIN_BY_BIN_STAT_TYPES

    def __init__(
        self,
        indata,
        options,
        *,
        chisqFit,
        covarianceFit,
        data_cov_inv,
        nobs_template,
    ):
        """
        Parameters
        ----------
        indata : FitInputData
            Loaded input tensor; only ``sumw``, ``sumw2``, ``nbins`` and
            ``dtype`` are read.
        options : argparse.Namespace
            CLI/options namespace with ``noBinByBinStat``, ``binByBinStatMode``,
            ``binByBinStatType``, and (optional) ``minBBKstat``.
        chisqFit, covarianceFit : bool
            Likelihood form selected on the Fitter.
        data_cov_inv : tf.Tensor or None
            Inverse data covariance matrix (only used by covarianceFit +
            normal-additive to precompute the auxiliary LU decomposition).
        nobs_template : tf.Tensor
            Template tensor with the same shape and dtype as
            ``Fitter.nobs``; used to size the per-bin ``nbeta`` Variable
            in the gamma + full mode.
        """
        self.indata = indata
        self.dtype = indata.dtype
        self.chisqFit = chisqFit
        self.covarianceFit = covarianceFit
        self.data_cov_inv = data_cov_inv

        self.enabled = not options.noBinByBinStat
        self.binByBinStatMode = options.binByBinStatMode
        self.minBBKstat = getattr(options, "minBBKstat", 0.0)

        if options.binByBinStatType == "automatic":
            if covarianceFit:
                self.binByBinStatType = "normal-additive"
            elif options.binByBinStatMode == "full":
                self.binByBinStatType = "normal-multiplicative"
            else:
                self.binByBinStatType = "gamma"
        else:
            self.binByBinStatType = options.binByBinStatType

        if (
            covarianceFit
            and self.enabled
            and not self.binByBinStatType.startswith("normal")
        ):
            raise Exception(
                'bin-by-bin stat only for option "--covarianceFit" with '
                '"--binByBinStatType normal"'
            )

        if self.binByBinStatType not in VALID_BIN_BY_BIN_STAT_TYPES:
            raise RuntimeError(
                f"Invalid binByBinStatType {self.binByBinStatType}, valid "
                f"choices are {VALID_BIN_BY_BIN_STAT_TYPES}"
            )

        # FIXME for now this is needed even if binByBinStat is off because
        # of how it is used in the global impacts and uncertainty band
        # computations (gradient is allowed to be zero or None and then
        # propagated or skipped only later).

        # Shape of the β tensor: per-bin in lite mode, per-(bin, proc) in full.
        if self.binByBinStatMode == "full":
            self.beta_shape = self.indata.sumw.shape
        elif self.binByBinStatMode == "lite":
            self.beta_shape = (self.indata.sumw.shape[0],)

        self.beta0 = tf.Variable(
            tf.zeros(self.beta_shape, dtype=self.dtype),
            trainable=False,
            name="beta0",
        )
        self.logbeta0 = tf.Variable(
            tf.zeros(self.beta_shape, dtype=self.dtype),
            trainable=False,
            name="logbeta0",
        )
        self.beta0_default_assign()

        # nuisance parameters for mc stat uncertainty
        self.beta = tf.Variable(self.beta0, trainable=False, name="beta")

        # dummy tensor to allow differentiation by β even when profiling
        self.ubeta = tf.zeros_like(self.beta)

        # Per-(bin, process) mask of zero-variance entries (sumw2 == 0).
        # Used in lite mode to exclude these from the merged sumw reduction
        # and from the β scaling. Defaults to None (no per-process axis).
        self.proc_zero_var_mask = None

        # Optional: gamma+full uses a per-bin Newton iterate variable.
        self.nbeta = None
        # Optional: covarianceFit + normal-additive precomputes an LU.
        self.betaauxlu = None

        if self.enabled:
            if self.binByBinStatMode == "full":
                self.varbeta = self.indata.sumw2
                self.sumw = self.indata.sumw
            else:
                if self.indata.sumw2.ndim > 1:
                    # Identify zero-variance entries per bin (sumw2 == 0,
                    # no bin-by-bin stat uncertainty to track). They are
                    # excluded from the merged sumw so the lite-mode
                    # kstat is computed from finite-variance contributions
                    # only.
                    self.proc_zero_var_mask = self.indata.sumw2 == 0.0
                    sumw_active = tf.where(
                        self.proc_zero_var_mask,
                        tf.zeros_like(self.indata.sumw),
                        self.indata.sumw,
                    )
                    self.varbeta = tf.reduce_sum(self.indata.sumw2, axis=-1)
                    self.sumw = tf.reduce_sum(sumw_active, axis=-1)
                else:
                    self.varbeta = self.indata.sumw2
                    self.sumw = self.indata.sumw

            self.betamask = (self.varbeta == 0.0) | (self.sumw == 0.0)
            if self.minBBKstat > 0.0:
                # Mask (bin, process) entries with effective MC stats below
                # threshold to avoid ill-conditioned profiles (e.g. from
                # mixed-sign-weight cancellations).
                varbeta_safe = tf.where(
                    self.betamask,
                    tf.ones_like(self.varbeta),
                    self.varbeta,
                )
                kstat_eff = self.sumw**2 / varbeta_safe
                low_stat = kstat_eff < tf.constant(
                    self.minBBKstat, dtype=self.varbeta.dtype
                )
                n_extra = int(
                    tf.reduce_sum(tf.cast(low_stat & ~self.betamask, tf.int32)).numpy()
                )
                if n_extra > 0:
                    logger.info(
                        f"--minBBKstat {self.minBBKstat}: masking {n_extra} "
                        "additional low-stat (bin, process) entries"
                    )
                self.betamask = self.betamask | low_stat
            self.kstat = tf.where(self.betamask, 1.0, self.sumw**2 / self.varbeta)

            if self.binByBinStatType in ["gamma", "normal-multiplicative"]:
                if self.binByBinStatType == "gamma" and self.binByBinStatMode == "full":
                    logger.warning(
                        "Running with '--binByBinStatType gamma "
                        "--binByBinStatMode full' is experimental and "
                        "results should be taken with care"
                    )
                    self.nbeta = tf.Variable(
                        tf.ones_like(nobs_template), trainable=True, name="nbeta"
                    )

            elif self.binByBinStatType == "normal-additive":
                # precompute decomposition of composite matrix to speed up
                # calculation of profiled beta values
                if covarianceFit:
                    sbeta = tf.math.sqrt(self.varbeta[: self.indata.nbins])

                    if self.binByBinStatMode == "lite":
                        sbeta = tf.linalg.LinearOperatorDiag(sbeta)
                        self.betaauxlu = tf.linalg.lu(
                            sbeta @ data_cov_inv @ sbeta
                            + tf.eye(
                                data_cov_inv.shape[0],
                                dtype=data_cov_inv.dtype,
                            )
                        )
                    elif self.binByBinStatMode == "full":
                        varbetasum = tf.reduce_sum(
                            self.varbeta[: self.indata.nbins], axis=1
                        )
                        varbetasum = tf.linalg.LinearOperatorDiag(varbetasum)
                        self.betaauxlu = tf.linalg.lu(
                            varbetasum @ data_cov_inv
                            + tf.eye(
                                data_cov_inv.shape[0],
                                dtype=data_cov_inv.dtype,
                            )
                        )

    # --- helpers -----------------------------------------------------------

    def default_beta0(self):
        """Default β0 tensor: 1 for multiplicative constraints, 0 for additive."""
        if self.binByBinStatType in ["gamma", "normal-multiplicative"]:
            return tf.ones(self.beta_shape, dtype=self.dtype)
        elif self.binByBinStatType == "normal-additive":
            return tf.zeros(self.beta_shape, dtype=self.dtype)

    def set_beta0(self, values):
        """Assign β0 and update the cached log β0 used in the gamma constraint."""
        self.beta0.assign(values)
        # Compute offset for Gamma NLL improved numerical precision in the
        # minimizer; the offset is chosen to give the saturated likelihood.
        beta0safe = tf.where(
            values == 0.0, tf.constant(1.0, dtype=values.dtype), values
        )
        self.logbeta0.assign(tf.math.log(beta0safe))

    def beta0_default_assign(self):
        self.set_beta0(self.default_beta0())

    def beta_default_assign(self):
        self.beta.assign(self.beta0)

    @property
    def is_linear(self):
        """Whether the BBB contribution to the loss is purely quadratic.

        True when BBB is disabled (no contribution) or the constraint is
        ``normal-additive`` (``0.5 (β-β0)²``). False for ``gamma`` and
        ``normal-multiplicative`` whose constraints involve ``log β`` or
        ``β·kstat``-like terms. Used by :attr:`Fitter.is_linear`.
        """
        return (not self.enabled) or self.binByBinStatType == "normal-additive"

    # --- toy randomization -------------------------------------------------

    def randomize_bayes(self):
        """Bayesian toy: randomize the *value* of β around β0.

        FIXME: only valid for β0 = β = 1 in the gamma case (which should
        always hold when throwing toys).
        """
        if not self.enabled:
            return

        if self.binByBinStatType == "gamma":
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

    def randomize_postfit(self):
        """Postfit toy: randomize β around β0 using a normal with stddev
        ``sqrt(varbeta)``.

        Used by :meth:`Fitter.toyassign` when ``randomize_parameters=True``
        — i.e., when the toy-throwing routine wants to perturb β alongside
        the fit parameters ``x``. No-op when BBB is disabled.

        Note: type-agnostic (unlike :meth:`randomize_bayes`); always uses
        the normal-multiplicative-form stddev. Preserves the legacy
        behaviour from ``Fitter.toyassign`` and is correct in expectation
        for ``normal-multiplicative``; the gamma and normal-additive cases
        re-use the same approximation as before.
        """
        if not self.enabled:
            return
        self.beta.assign(
            tf.random.normal(
                shape=[],
                mean=self.beta0,
                stddev=tf.sqrt(self.varbeta),
                dtype=self.beta.dtype,
            )
        )

    def randomize_frequentist(self):
        """Frequentist toy: randomize the *constraint minimum* β0 around β.

        FIXME: only valid for β0 = β = 1 in the gamma case.
        """
        if not self.enabled:
            return

        if self.binByBinStatType == "gamma":
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

    # --- constraint NLL term -----------------------------------------------

    def lbeta(self, beta, full_nll=False):
        """Constraint-NLL contribution from the β nuisance.

        Returns ``None`` when BBB is disabled. Otherwise returns the scalar
        ``Σ`` of per-(bin[, proc]) terms suitable for adding to the model
        NLL (when ``full_nll=False``, the constant terms that don't depend
        on β/β0 are dropped — this is the variant used inside the
        minimizer; ``full_nll=True`` adds them back, e.g. for absolute
        likelihood reporting).
        """
        if not self.enabled:
            return None

        beta0 = self.beta0
        if self.binByBinStatType == "gamma":
            kstat = self.kstat
            betasafe = tf.where(beta0 == 0.0, tf.constant(1.0, dtype=beta.dtype), beta)
            logbeta = tf.math.log(betasafe)

            if full_nll:
                # constant terms
                lgammaalpha = tf.math.lgamma(kstat * beta0)
                alphalntheta = -kstat * beta0 * tf.math.log(kstat)
                lbeta = (
                    -kstat * beta0 * logbeta + kstat * beta + lgammaalpha + alphalntheta
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
                sigma2 = 1.0 / self.kstat
                # log(1/sqrt(2π)) = -0.9189385332046727
                lbeta = (
                    lbeta
                    + tf.cast(tf.shape(sigma2), tf.float64) * 0.9189385332046727
                    + 0.5 * tf.math.log(sigma2)
                )

        return tf.reduce_sum(lbeta)

    # --- profile + application ---------------------------------------------

    def needs_per_proc_norm(self):
        """Whether the host should materialize per-process norms.

        ``full`` mode always needs them (its profile uses per-(bin, proc)
        β); ``lite`` mode only needs them when at least one process has
        zero variance so that β is applied selectively to the
        finite-variance contributions.
        """
        if not self.enabled:
            return False
        if self.binByBinStatMode == "full":
            return True
        return self.proc_zero_var_mask is not None

    def profile_and_apply(
        self,
        nexp,
        norm,
        nobs,
        varnobs,
        lognobs,
        *,
        profile=True,
        compute_norm=False,
        full=True,
    ):
        """Profile β (when ``profile=True``) and apply it to ``nexp`` / ``norm``.

        Returns ``(nexp, norm, beta)`` where ``beta`` is the (possibly
        profiled) β tensor. When BBB is disabled, returns the inputs
        untouched and ``beta=None``.
        """
        if not self.enabled:
            return nexp, norm, None

        if profile:
            # analytic solution for profiled barlow-beeston lite parameters for each combination
            # of likelihood and uncertainty form

            nexp_profile = nexp[: self.indata.nbins]
            beta0 = self.beta0[: self.indata.nbins]

            # In lite mode, split nexp_profile into per-bin sums over
            # processes with finite variance (sumw2 > 0) and zero variance
            # (sumw2 == 0) so that β only scales the finite-variance part.
            # When all processes have finite variances, n_active == nexp_profile
            # and n_zero_var == 0, so the lite formulas reduce to their original
            # form.
            if self.binByBinStatMode == "lite" and self.proc_zero_var_mask is not None:
                zv_mask = self.proc_zero_var_mask[: self.indata.nbins]
                norm_real = norm[: self.indata.nbins]
                n_active = tf.reduce_sum(
                    tf.where(zv_mask, tf.zeros_like(norm_real), norm_real), axis=-1
                )
                n_zero_var = tf.reduce_sum(
                    tf.where(zv_mask, norm_real, tf.zeros_like(norm_real)), axis=-1
                )
            else:
                n_active = nexp_profile
                n_zero_var = tf.zeros_like(nexp_profile)

            # Safe denominator for formulas that divide by n_active; bins with
            # n_active == 0 will be overridden by betamask afterwards.
            n_active_safe = tf.where(n_active > 0, n_active, tf.ones_like(n_active))

            # Real-bins slices used by every type×mode branch below.
            kstat = self.kstat[: self.indata.nbins]
            betamask = self.betamask[: self.indata.nbins]
            varbeta = self.varbeta[: self.indata.nbins]

            if self.chisqFit:
                if self.binByBinStatType == "gamma":

                    if self.binByBinStatMode == "lite":
                        abeta = n_active_safe**2
                        bbeta = kstat * varnobs + (n_zero_var - nobs) * n_active_safe
                        cbeta = -kstat * varnobs * beta0
                        beta = solve_quad_eq(abeta, bbeta, cbeta)
                    elif self.binByBinStatMode == "full":
                        norm_profile = norm[: self.indata.nbins]
                        logbeta0 = self.logbeta0[: self.indata.nbins]

                        # Minimum total expected yield for which all betas are
                        # positive. Optimise in log-space u = log(x - threshold)
                        # so that x = threshold + exp(u) > threshold for any
                        # real u, guaranteeing den > 0 and beta > 0 without any
                        # clipping. Protect against zero norm_profile for
                        # masked processes: kstat/0 = inf for the argmin
                        # gradient gives 0*(-inf) = NaN. Use a dummy norm=1
                        # for betamask bins so the division is finite, then
                        # set those entries to +inf to exclude them from the
                        # min.
                        norm_thresh = tf.where(
                            betamask, tf.ones_like(norm_profile), norm_profile
                        )
                        f_thresh = tf.where(
                            betamask,
                            tf.fill(
                                tf.shape(kstat),
                                tf.cast(float("inf"), kstat.dtype),
                            ),
                            kstat / norm_thresh,
                        )
                        threshold = nobs - varnobs * tf.reduce_min(f_thresh, axis=1)

                        # Initialise nbeta in log-space.
                        self.nbeta.assign(tf.zeros_like(nobs))

                        # solving nbeta numerically using newtons method
                        # (does not work with forward differentiation i.e.
                        # use --globalImpacts with --globalImpactsDisableJVP)
                        def fnll_nbeta(u):
                            x = threshold + tf.exp(u)
                            den = (
                                kstat + ((x - nobs) / varnobs)[..., None] * norm_profile
                            )
                            beta = kstat * beta0 / den
                            beta = tf.where(betamask, beta0, beta)
                            betasafe = tf.where(
                                beta0 == 0.0,
                                tf.constant(1.0, dtype=beta.dtype),
                                beta,
                            )
                            logbeta = tf.math.log(betasafe)
                            new_nexp = tf.reduce_sum(beta * norm_profile, axis=-1)
                            ln = 0.5 * (new_nexp - nobs) ** 2 / varnobs
                            lbeta = tf.reduce_sum(
                                kstat * (beta - beta0)
                                - kstat * beta0 * (logbeta - logbeta0),
                                axis=-1,
                            )
                            return ln + lbeta

                        beta = self._newton_solve_nbeta(
                            fnll_nbeta=fnll_nbeta,
                            kstat=kstat,
                            beta0=beta0,
                            norm_profile=norm_profile,
                            nobs=nobs,
                            varnobs=varnobs,
                            threshold=threshold,
                            chisq_fit=True,
                        )

                    beta = tf.where(betamask, beta0, beta)
                elif self.binByBinStatType == "normal-multiplicative":
                    if self.binByBinStatMode == "lite":
                        beta = (
                            (nobs - n_zero_var) * n_active_safe / varnobs
                            + kstat * beta0
                        ) / (kstat + n_active_safe * n_active_safe / varnobs)
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
                            nobs / varnobs * n2kstatsum
                            + tf.reduce_sum(norm_profile * beta0, axis=-1)
                        ) / (1 + 1 / varnobs * n2kstatsum)
                        beta = (
                            beta0
                            + (1 / varnobs * (nobs - nbeta))[..., None]
                            * norm_profile
                            / kstat
                        )
                        beta = tf.where(betamask, beta0, beta)
                elif self.binByBinStatType == "normal-additive":
                    sbeta = tf.math.sqrt(varbeta)
                    if self.binByBinStatMode == "lite":
                        beta = (sbeta * (nobs - nexp_profile) + varnobs * beta0) / (
                            varnobs + varbeta
                        )
                    elif self.binByBinStatMode == "full":
                        varbetasum = tf.reduce_sum(varbeta, axis=-1)
                        nbeta = (
                            tf.reduce_sum(sbeta * beta0, axis=-1)
                            + varbetasum / varnobs * (nobs - nexp_profile)
                        ) / (1 + varbetasum / varnobs)
                        beta = (
                            beta0
                            - sbeta * ((nexp_profile + nbeta - nobs) / varnobs)[:, None]
                        )
            elif self.covarianceFit:
                if self.binByBinStatType == "normal-multiplicative":
                    if self.binByBinStatMode == "lite":
                        n_active_m = tf.linalg.LinearOperatorDiag(n_active_safe)
                        A = (
                            n_active_m @ self.data_cov_inv @ n_active_m
                            + tf.linalg.diag(kstat)
                        )
                        b = (
                            n_active_m
                            @ (self.data_cov_inv @ (nobs - n_zero_var)[:, None])
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
                            n2kstatsum_m @ self.data_cov_inv @ (nobs[:, None])
                            + nbeta0[:, None]
                        )

                        # Cholesky solve sometimes does not give corret result
                        # chol = tf.linalg.cholesky(A)
                        # nbeta = tf.linalg.cholesky_solve(chol, b)
                        nbeta = tf.linalg.solve(A, b)

                        # now solve for beta [nprocesses x nbins]
                        beta = beta0 - norm_profile / kstat * (
                            self.data_cov_inv @ (nbeta - nobs[:, None])
                        )
                        beta = tf.where(betamask, beta0, beta)
                elif self.binByBinStatType == "normal-additive":
                    sbeta = tf.math.sqrt(varbeta)
                    if self.binByBinStatMode == "lite":
                        sbeta_m = tf.linalg.LinearOperatorDiag(sbeta)
                        beta = tf.linalg.lu_solve(
                            *self.betaauxlu,
                            sbeta_m
                            @ self.data_cov_inv
                            @ ((nobs - nexp_profile)[:, None])
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
                            @ ((nobs - nexp_profile)[:, None])
                            + sbetabeta0sum[:, None],
                        )
                        # second solve for beta
                        beta = beta0 - sbeta * (
                            self.data_cov_inv
                            @ (nbeta + nexp_profile[:, None] - nobs[:, None])
                        )
            else:
                if self.binByBinStatType == "gamma":

                    if self.binByBinStatMode == "lite":
                        # Quadratic profile when zero-variance processes
                        # contribute (β scales only the finite-variance
                        # part, leaving n_zero_var unchanged). Reduces to
                        # the original closed form
                        # (nobs + k·β0) / (n_active + k) when
                        # n_zero_var == 0.
                        abeta = n_active_safe * (n_active_safe + kstat)
                        bbeta = (
                            n_active_safe * (n_zero_var - nobs - beta0 * kstat)
                            + kstat * n_zero_var
                        )
                        cbeta = -beta0 * kstat * n_zero_var
                        beta = solve_quad_eq(abeta, bbeta, cbeta)
                    elif self.binByBinStatMode == "full":
                        norm_profile = norm[: self.indata.nbins]
                        logbeta0 = self.logbeta0[: self.indata.nbins]

                        # Minimum total expected yield for which all betas are
                        # positive. Optimise in log-space u = log(x - threshold)
                        # so that x = threshold + exp(u) > threshold for any
                        # real u, guaranteeing den > 0 and beta > 0 without any
                        # clipping. Protect against zero norm_profile for
                        # masked processes (see chisqFit branch above).
                        norm_thresh = tf.where(
                            betamask, tf.ones_like(norm_profile), norm_profile
                        )
                        f_thresh = tf.where(
                            betamask,
                            tf.fill(
                                tf.shape(kstat),
                                tf.cast(float("inf"), kstat.dtype),
                            ),
                            1.0 + kstat / norm_thresh,
                        )
                        threshold = nobs / tf.reduce_min(f_thresh, axis=1)

                        # Initialise nbeta in log-space from the current
                        # nexp_profile.
                        self.nbeta.assign(tf.zeros_like(nobs))

                        def fnll_nbeta(u):
                            x = threshold + tf.exp(u)
                            den = (1 - nobs / x)[..., None] * norm_profile + kstat
                            beta = kstat * beta0 / den
                            beta = tf.where(betamask, beta0, beta)
                            betasafe = tf.where(
                                beta0 == 0.0,
                                tf.constant(1.0, dtype=beta.dtype),
                                beta,
                            )
                            logbeta = tf.math.log(betasafe)
                            new_nexp = tf.reduce_sum(beta * norm_profile, axis=-1)
                            nexpsafe = tf.where(
                                nobs == 0.0,
                                tf.constant(1.0, dtype=new_nexp.dtype),
                                new_nexp,
                            )
                            lognexp = tf.math.log(nexpsafe)
                            ln = new_nexp - nobs - nobs * (lognexp - lognobs)
                            lbeta = tf.reduce_sum(
                                kstat * (beta - beta0)
                                - kstat * beta0 * (logbeta - logbeta0),
                                axis=-1,
                            )
                            return ln + lbeta

                        beta = self._newton_solve_nbeta(
                            fnll_nbeta=fnll_nbeta,
                            kstat=kstat,
                            beta0=beta0,
                            norm_profile=norm_profile,
                            nobs=nobs,
                            varnobs=None,
                            threshold=threshold,
                            chisq_fit=False,
                        )

                    beta = tf.where(betamask, beta0, beta)
                elif self.binByBinStatType == "normal-multiplicative":
                    if self.binByBinStatMode == "lite":
                        # Quadratic in β when zero-variance processes
                        # contribute. Reduces to a·β² + b·β + c = 0 with
                        # a = k, b = n_active - β0·k, c = -nobs when
                        # n_zero_var == 0.
                        abeta = kstat * n_active_safe
                        bbeta = (
                            n_active_safe**2
                            - kstat * beta0 * n_active_safe
                            + kstat * n_zero_var
                        )
                        cbeta = (
                            n_active_safe * (n_zero_var - nobs)
                            - kstat * beta0 * n_zero_var
                        )
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
                        pbeta = tf.reduce_sum(n2kstat - beta0 * norm_profile, axis=-1)
                        qbeta = -nobs * tf.reduce_sum(n2kstat, axis=-1)
                        nbeta = solve_quad_eq(1, pbeta, qbeta)
                        beta = (
                            beta0 + (nobs / nbeta - 1)[..., None] * norm_profile / kstat
                        )
                        beta = tf.where(betamask, beta0, beta)
                elif self.binByBinStatType == "normal-additive":
                    sbeta = tf.math.sqrt(varbeta)
                    if self.binByBinStatMode == "lite":
                        abeta = sbeta
                        abeta = tf.where(
                            varbeta == 0.0,
                            tf.constant(1.0, dtype=varbeta.dtype),
                            abeta,
                        )
                        bbeta = varbeta + nexp_profile - sbeta * beta0
                        cbeta = sbeta * (nexp_profile - nobs) - nexp_profile * beta0
                        beta = solve_quad_eq(abeta, bbeta, cbeta)
                        beta = tf.where(varbeta == 0.0, beta0, beta)
                    elif self.binByBinStatMode == "full":
                        norm_profile = norm[: self.indata.nbins]
                        qbeta = -nobs * tf.reduce_sum(varbeta, axis=-1)
                        pbeta = tf.reduce_sum(
                            varbeta - sbeta * beta0 - norm_profile, axis=-1
                        )
                        nbeta = solve_quad_eq(1, pbeta, qbeta)
                        beta = beta0 + (nobs / nbeta - 1)[..., None] * sbeta

            if self.indata.nbinsmasked:
                beta = tf.concat([beta, self.beta0[self.indata.nbins :]], axis=0)
        else:
            beta = self.beta

        # Add dummy tensor to allow convenient differentiation by beta even
        # when profiling
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
                    safe_n0 = tf.where(n0 > 0, n0, 1.0)
                    ratio = var / safe_n0
                    norm = tf.where(n0 > 0, norm * (1 + ratio), norm)

                norm = tf.where(betamask, norm, betasel * norm)
                nexp = tf.reduce_sum(norm, -1)
            elif self.proc_zero_var_mask is not None:
                # β scales only finite-variance entries; zero-variance
                # entries (sumw2==0) pass through unchanged.
                zv_mask = self.proc_zero_var_mask[: nexp.shape[0]]
                scale = tf.where(
                    zv_mask | betamask[..., None],
                    tf.ones_like(norm),
                    betasel[..., None],
                )
                norm = norm * scale
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
            elif self.proc_zero_var_mask is not None:
                # Additive correction sbeta·β is distributed only over
                # finite-variance entries, weighted by their share of
                # n_active per bin.
                zv_mask = self.proc_zero_var_mask[: nexp.shape[0]]
                active_norm = tf.where(zv_mask, tf.zeros_like(norm), norm)
                n_active_bin = tf.reduce_sum(active_norm, axis=-1)
                n_active_bin_safe = tf.where(
                    n_active_bin > 0, n_active_bin, tf.ones_like(n_active_bin)
                )
                addition = (
                    sbeta[..., None]
                    * betasel[..., None]
                    * active_norm
                    / n_active_bin_safe[..., None]
                )
                norm = norm + addition
                nexp = tf.reduce_sum(norm, -1)
            else:
                nexpnorm = nexp[..., None]
                nexp = nexp + sbeta * betasel
                if compute_norm:
                    # distribute the change in yields proportionally across processes
                    norm = (
                        norm + sbeta[..., None] * betasel[..., None] * norm / nexpnorm
                    )

        return nexp, norm, beta

    # --- gamma + full Newton helper ---------------------------------------

    def _newton_solve_nbeta(
        self,
        *,
        fnll_nbeta,
        kstat,
        beta0,
        norm_profile,
        nobs,
        varnobs,
        threshold,
        chisq_fit,
    ):
        """Numerical profile of β for the gamma + full mode.

        Uses Newton iteration on the per-bin scalar ``nbeta`` (in
        log-space ``u = log(x - threshold)``) to find the joint
        minimum of the data NLL + γ-constraint, then performs one
        differentiable Newton step at the converged value to restore
        ``du*/dz`` gradients otherwise lost through
        ``tf.Variable.assign_sub``.
        """

        def val_grad_hess_nbeta():
            with tf.GradientTape() as t2:
                with tf.GradientTape() as t1:
                    val = fnll_nbeta(self.nbeta)
                grad = t1.gradient(val, self.nbeta)
            hess = t2.gradient(grad, self.nbeta)
            return val, grad, hess

        def body(i, edm):
            val, grad, hess = val_grad_hess_nbeta()
            safe_hess = tf.maximum(hess, 1e-8)
            step = tf.clip_by_value(grad / safe_hess, -1.0, 1.0)
            self.nbeta.assign_sub(step)
            return i + 1, tf.reduce_max(0.5 * grad**2 / safe_hess)

        def cond(i, edm):
            return tf.logical_and(i < 50, edm > 1e-10)

        i0 = tf.constant(0)
        edm0 = tf.constant(tf.float64.max)
        # XLA needs a static upper bound on loop iterations to allocate
        # fixed-size tensor lists when the HVP is jit_compile=True.
        tf.while_loop(cond, body, loop_vars=(i0, edm0), maximum_iterations=50)

        # Implicit-function-theorem trick: one differentiable Newton step
        # at the converged value restores du*/dz gradients otherwise lost
        # through tf.Variable.assign_sub in the loop.
        u_stop = tf.stop_gradient(self.nbeta)
        with tf.GradientTape() as t2_imp:
            t2_imp.watch(u_stop)
            with tf.GradientTape() as t1_imp:
                t1_imp.watch(u_stop)
                val_imp = fnll_nbeta(u_stop)
            grad_imp = t1_imp.gradient(val_imp, u_stop)
        hess_imp = t2_imp.gradient(grad_imp, u_stop)
        safe_hess_imp = tf.maximum(hess_imp, 1e-8)
        u_diff = u_stop - grad_imp / safe_hess_imp

        x = threshold + tf.exp(u_diff)
        if chisq_fit:
            beta = (
                kstat
                * beta0
                / (kstat + ((x - nobs) / varnobs)[..., None] * norm_profile)
            )
        else:
            beta = kstat * beta0 / ((1 - nobs / x)[..., None] * norm_profile + kstat)
        return beta
