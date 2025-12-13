import numpy as np
import scipy.optimize
import tensorflow as tf
from scipy.optimize import root
from wums import logging

from rabbit import tfhelpers as tfh
from rabbit.tfhelpers import edmval_cov

logger = logging.child_logger(__name__)


@tf.function
def limit_constraint(f):
    val = f()
    return val


@tf.function
def limit_constraint_val_grad(fitter, fitter_asimov, f):
    with tf.GradientTape() as t:
        t.watch([fitter.x, fitter_asimov.x])
        val = f()
    grad = t.gradient(val, [fitter.x, fitter_asimov.x])
    grad = tf.concat(grad, 0)

    return val, grad


def compute_likelihood_limit(
    fitter, fitter_asimov, nllvalreduced, nllvalreduced_asimov, param, cl
):

    # We have 2 fitters, one to fit the asimov dataset and one to fit the real data
    # The two are allowed to have different parameters; but the POI has to be the same

    # initial values
    idx = np.where(fitter.parms.astype(str) == param)[0][0]
    xval = tf.identity(fitter.x).numpy()

    xval_asimov = np.zeros_like(xval)
    xval_asimov[idx] = xval[idx]
    fitter_asimov.x.assign(xval_asimov)

    # perform an asumov fit at xobs to get good starting values
    fitter_asimov.freeze_params(param)
    fitter_asimov.minimize()
    fitter_asimov.defreeze_params(param)

    xval_asimov = tf.identity(fitter_asimov.x).numpy()
    xval_asimov = np.delete(xval_asimov, idx)
    xval_init = np.concatenate([xval, xval_asimov])

    # to find the upper limit we set the mu equal to mu_obs also for the asimov dataset
    # xval_init[len(xval_init)//2:][idx] = xval_init[idx]

    # TODO: modified statistics

    def _compute_limit_constraint():
        qmu = tf.sqrt(2 * (fitter._compute_nll() - nllvalreduced))
        qA = tf.sqrt(2 * (fitter_asimov._compute_nll() - nllvalreduced_asimov))
        constraint = cl * tfh.normal_cdf(qA - qmu) + tfh.normal_cdf(qmu) - 1

        logger.debug(f"q_mu = {qmu}")
        logger.debug(f"q_A = {qA}")
        logger.debug(f"constraint = {constraint}")

        logger.debug(f"fitter_asimov.x = {fitter_asimov.x}")
        logger.debug(f"fitter.x = {fitter.x}")

        return constraint

    def _compute_limit_constraint_grad():
        qmu = tf.sqrt(2 * (fitter._compute_nll() - nllvalreduced))
        qA = tf.sqrt(2 * (fitter_asimov._compute_nll() - nllvalreduced_asimov))

        phi_qmu = tfh.normal_pdf(qmu)
        phi_qA_qmu = tfh.normal_pdf(qA - qmu)

        with tf.GradientTape() as t:
            val = tf.sqrt(2 * fitter._compute_nll())
        dqmu_dx = t.gradient(val, fitter.x)

        with tf.GradientTape() as t:
            val = tf.sqrt(2 * fitter_asimov._compute_nll())
        dqA_dx = t.gradient(val, fitter_asimov.x)

        # from first fitter
        dconstraint_dx1 = (phi_qmu - cl * phi_qA_qmu) * dqmu_dx

        # from second fitter
        dconstraint_dx2 = cl * phi_qA_qmu * dqA_dx

        dconstraint_dx1 = dconstraint_dx1.numpy()
        dconstraint_dx2 = dconstraint_dx2.numpy()

        # common x[idx]
        dconstraint_dx1[idx] = dconstraint_dx1[idx] + dconstraint_dx2[idx]

        # drop x[idx] from second fitter
        dconstraint_dx2 = np.delete(dconstraint_dx2, idx)

        dconstraint_dx = np.concatenate([dconstraint_dx1, dconstraint_dx2])

        logger.debug(f"q_mu = {qmu}")
        logger.debug(f"q_A = {qA}")
        logger.debug(f"dqmu_dx = {dqmu_dx}")
        logger.debug(f"dqA_dx = {dqA_dx}")
        logger.debug(f"dconstraint_dx1 = {dconstraint_dx1}")
        logger.debug(f"dconstraint_dx2 = {dconstraint_dx2}")

        logger.debug(f"dconstraint/dx = {dconstraint_dx}")

        return dconstraint_dx

    def scipy_constraint(xval):
        logger.info(f"scipy_constraint: xval = {xval}")

        nx = (len(xval) + 1) // 2
        x1 = xval[:nx]
        x2 = xval[nx:]
        x2 = np.insert(x2, idx, x1[idx])

        fitter.x.assign(x1)
        fitter_asimov.x.assign(x2)
        val = _compute_limit_constraint()
        # val = limit_constraint(_compute_limit_constraint)
        return val.numpy()

    def scipy_constraint_grad(xval):
        logger.info(f"scipy_constraint_grad: xval = {xval}")

        nx = (len(xval) + 1) // 2
        x1 = xval[:nx]
        x2 = xval[nx:]
        x2 = np.insert(x2, idx, x1[idx])

        fitter.x.assign(x1)
        fitter_asimov.x.assign(x2)

        # val, grad = limit_constraint_val_grad(fitter, fitter_asimov, _compute_limit_constraint)
        grad = _compute_limit_constraint_grad()
        return grad

    nlc = scipy.optimize.NonlinearConstraint(
        fun=scipy_constraint,
        lb=-0.5,  # -0.5 at qmu = 0
        ub=0,
        jac=scipy_constraint_grad,
        hess=scipy.optimize.SR1(),  # TODO: use exact hessian or hessian vector product
    )

    # Objective function and its derivatives
    def objective(params):
        logger.debug(f"param[{idx}] = {params[idx]}")
        return -params[idx]

    def objective_jac(params):
        jac = np.zeros_like(params)
        jac[idx] = -0.001
        return jac

    def objective_hessp(params, v):
        return np.zeros_like(v)

    res = scipy.optimize.minimize(
        objective,
        xval_init,
        method="trust-constr",
        jac=objective_jac,
        hessp=objective_hessp,
        constraints=[nlc],
        options={
            "maxiter": 500,
            "xtol": 1e-10,
            "gtol": 1e-10,
            # "verbose": 3
        },
    )

    if not res["success"]:
        logger.warning("No success")

    limit = res["x"][idx]

    logger.info(f"Observed (Likelihood): {param} < {limit}")

    return limit


def compute_gaussian_limit(fitter, param, idx, xerr, cl_s):

    xobs = fitter.get_blinded_poi()[idx]
    val, grad, hess = fitter.loss_val_grad_hess()
    edmval, cov = edmval_cov(grad, hess)
    fitter.cov.assign(cov)
    xobs_err = fitter.cov[idx, idx] ** 0.5

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
