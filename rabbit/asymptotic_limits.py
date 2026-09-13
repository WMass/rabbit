import numpy as np
from scipy.optimize import root
from wums import logging

import tensorflow as tf  # isort: skip

from rabbit import tfhelpers as tfh  # isort: skip

logger = logging.child_logger(__name__)


def compute_likelihood_limit(
    fitter,
    fitter_asimov,
    nllvalreduced,
    nllvalreduced_asimov,
    param,
    cl_s,
    muhat=None,
    fun=None,
    r_init=None,
    allow_negative=True,
):
    """Observed asymptotic CLs upper limit from the profile likelihood.

    Reuses the same profiled-NLL machinery as the (working) expected limit:
    the profile-likelihood test statistic is evaluated with `fitter` fitted to
    data (q_mu) and with `fitter_asimov` fitted to the background-only Asimov
    dataset (q_{mu,A}). The limit is the value of `param` for which
    CLs(mu) = cl_s, following the asymptotic formulae of Cowan, Cranmer,
    Gross, Vitells, arXiv:1007.1727.

    For a negative best-fit signal strength the modified test statistic
    q-tilde_mu (Eq. 16) is used, taking the mu=0 conditional fit as reference
    instead of the global minimum, consistent with the `xobs < 0` branch of
    `compute_gaussian_limit`.

    If `fun` is given (channel/mapping case) the limit is reported on the
    mapped observable at the limit point, matching the convention of the
    expected channel limit produced via `contour_scan(fun=...)`.
    """
    idx = np.where(fitter.parms.astype(str) == param)[0][0]
    if muhat is None:
        # best-fit POI in the internal (self.x) space, consistent with
        # profiled_nll_at_poi (also correct for the channel/mapping case
        # where `param` is the POI driving the mapped observable)
        muhat = float(fitter.x.numpy()[idx])

    logger.info(
        f"Compute observed (Likelihood) limit for {param}, "
        f"mu_hat = {muhat}, CLs = {cl_s}"
    )

    # Reference NLL for the data test statistic q_mu (denominator of the
    # profile likelihood ratio). Modified statistics (mu_hat < 0): use the
    # mu = 0 conditional fit as reference instead of the global minimum.
    if allow_negative and muhat < 0:
        logger.debug("Use modified statistics (mu_hat < 0)")
        ref = fitter.profiled_nll_at_poi(param, 0.0)
        mu_lo = 0.0
    else:
        ref = nllvalreduced
        mu_lo = muhat

    def sqrt_qmu(mu):
        # one-sided test statistic for an upper limit
        if mu <= mu_lo:
            return 0.0
        dnll = fitter.profiled_nll_at_poi(param, mu) - ref
        return np.sqrt(max(0.0, 2.0 * dnll))

    def sqrt_qA(mu):
        # background-only Asimov: best fit at mu = 0, reference is its minimum
        if mu <= 0.0:
            return 0.0
        dnll = fitter_asimov.profiled_nll_at_poi(param, mu) - nllvalreduced_asimov
        return np.sqrt(max(0.0, 2.0 * dnll))

    def cls_minus_alpha(mu):
        mu = float(np.asarray(mu).reshape(-1)[0])
        # np.float64 (not python float) so tfh.normal_cdf can read .dtype
        qmu = np.float64(sqrt_qmu(mu))
        qA = np.float64(sqrt_qA(mu))
        # CLs+b - cl_s * CLb ; same algebra as compute_gaussian_limit
        val = tfh.normal_cdf(-qmu).numpy() - cl_s * tfh.normal_cdf(qA - qmu).numpy()
        logger.debug(
            f"mu = {mu}, sqrt(q_mu) = {qmu}, sqrt(q_A) = {qA}, "
            f"CLs+b - cl_s*CLb = {val}"
        )
        return val

    if r_init is None or not np.isfinite(r_init) or r_init <= mu_lo:
        # scale-aware default: ~2 sigma above the best fit
        sigma = float(fitter.cov[idx, idx].numpy()) ** 0.5
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0
        r_init = max(mu_lo, muhat) + 2.0 * sigma

    res = root(cls_minus_alpha, r_init)
    if not res.success:
        logger.warning(f"Root finding for observed limit on {param} did not converge")

    mu_limit = float(np.asarray(res.x).reshape(-1)[0])

    if fun is not None:
        # profile at the limit POI and report the mapped observable, same
        # convention as the expected channel limit (contour_scan(fun=...))
        xsave = tf.identity(fitter.x)
        xval = xsave.numpy()
        xval[idx] = mu_limit
        fitter.x.assign(xval)
        fitter.freeze_params(param)
        fitter.minimize()
        fitter.defreeze_params(param)
        exp, *_ = fitter.expected_with_variance(
            fun,
            profile=True,
            compute_cov=False,
            compute_global_impacts=False,
            need_observables=True,
            inclusive=True,
        )
        fitter.x.assign(xsave)
        limit = float(exp.numpy()[0])
    else:
        limit = mu_limit

    logger.info(f"Observed (Likelihood): {param} < {limit}")
    return limit


def compute_gaussian_limit(param, xobs, xobs_err, xerr, cl_s):
    logger.info(
        f"Measured xobs +/- xobs_err (asimov err) = {xobs} +/- {xobs_err} ({xerr})"
    )

    # In general the limit in Gaussian approximation is not analytically solveable but we have to find the root of f(x)
    if xobs < 0:
        logger.debug("Use modified statistics")
        # initial guess
        r_init = 0

        def qmu_sqrt(x):
            return tf.sqrt(((x - xobs) / xobs_err) ** 2 - (xobs / xobs_err) ** 2)

    else:
        # initial guess
        # Assume that the uncertainty in the asimov fit is the same as in the fit to data, then we can analytically solve for r
        r_init = xobs - xerr * (tfh.normal_pdf(cl_s * tfh.normal_cdf(xobs / xerr)))

        def qmu_sqrt(x):
            return (x - xobs) / xobs_err

    def qA_sqrt(x):
        return x / xerr

    def f(x):
        qmu = qmu_sqrt(x)
        qA = qA_sqrt(x)
        return tfh.normal_cdf(-qmu) - cl_s * tfh.normal_cdf(qA - qmu)

    res = root(f, r_init)
    limit = res.x[0]
    logger.info(f"Observed (Gaussian): {param} < {limit}")
    return limit
