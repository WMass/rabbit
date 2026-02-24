import numpy as np
import tensorflow as tf
from wums import logging

logger = logging.child_logger(__name__)


def _compute_curvature(fitter):
    """
    Following Eq.(4.3) from https://iopscience.iop.org/article/10.1088/1748-0221/7/10/T10003/pdf
    """

    # the full derivative d(Li) / d(tau) = pd(Li)/pd(tau) + pd(Li)/pd(x) * d(x)/d(tau)
    #                                    = pd(Li)/pd(tau) - pd(Li)/pd(x) * (pd2(L)/pd(x^2))^-1 * pd2(L)/pd(x)pd(tau)
    # there is no dependency of Li on tau, thus, the first term is 0
    # (pd2(L)/pd(x^2))^-1 is the covariance matrix
    with tf.GradientTape(persistent=True) as t3:
        t3.watch(fitter.tau)

        # 1) compute dx/dtau
        with tf.GradientTape(persistent=True) as t2:
            t2.watch(fitter.tau)
            with tf.GradientTape() as t1:
                nll = fitter._compute_nll()

            pdLpdx = t1.gradient(nll, fitter.x)

        pd2Lpdx2 = t2.jacobian(pdLpdx, fitter.x)
        pd2Lpdxpdtau = t2.jacobian(pdLpdx, fitter.tau)

        chol = tf.linalg.cholesky(pd2Lpdx2)
        dxdtau = -tf.linalg.cholesky_solve(chol, pd2Lpdxpdtau[:, None])
        dxdtau = tf.reshape(dxdtau, [-1])

        # 2) compute pdLx/pdx, pdLy/pdx and pd^2Lx/pdx^2, pd^2Ly/pdx^2
        with tf.GradientTape(persistent=True) as t_inner:
            nexpfullcentral, _, beta = fitter._compute_yields_with_beta(
                profile=False,
                compute_norm=False,
                full=len(fitter.regularizers),
            )

            nexp = nexpfullcentral[: fitter.indata.nbins]

            ln = fitter._compute_ln(nexp)
            lc = fitter._compute_lc()
            lbeta = fitter._compute_lbeta(beta)
            lx = tf.math.log(ln + lc + lbeta)

            x = fitter.get_x()
            penalties = [
                reg.compute_nll_penalty(x, nexpfullcentral)
                for reg in fitter.regularizers
            ]
            ly = tf.math.log(tf.add_n(penalties))

        pdLxpdx = t_inner.gradient(lx, fitter.x)
        pdLypdx = t_inner.gradient(ly, fitter.x)

        pdLxpdx_dxdtau = tf.reduce_sum(pdLxpdx * dxdtau)
        pdLypdx_dxdtau = tf.reduce_sum(pdLypdx * dxdtau)

    pdLxpdx_d2xdtau2 = t3.gradient(pdLxpdx_dxdtau, fitter.tau)
    pdLypdx_d2xdtau2 = t3.gradient(pdLypdx_dxdtau, fitter.tau)

    pd2Lxpdx2 = t3.jacobian(pdLxpdx, fitter.x)
    pd2Lypdx2 = t3.jacobian(pdLypdx, fitter.x)

    dLxdtau = tf.reduce_sum(pdLxpdx * dxdtau)
    dLydtau = tf.reduce_sum(pdLypdx * dxdtau)

    d2Lxdtau2 = (
        tf.reduce_sum(dxdtau * tf.linalg.matvec(pd2Lxpdx2, dxdtau)) + pdLxpdx_d2xdtau2
    )
    d2Lydtau2 = (
        tf.reduce_sum(dxdtau * tf.linalg.matvec(pd2Lypdx2, dxdtau)) + pdLypdx_d2xdtau2
    )

    curvature = (d2Lydtau2 * dLxdtau - d2Lxdtau2 * dLydtau) / tf.pow(
        tf.square(dLxdtau) + tf.square(dLydtau), 1.5
    )

    return curvature


@tf.function
def compute_curvature(fitter):
    return _compute_curvature(fitter)


def l_curve_scan_tau(fitter, min=1.7, max=2.1, step=0.1):
    tau0 = fitter.tau.numpy()

    curvatures = []
    tau_steps = np.arange(min, max, step)

    for i, v in enumerate(tau_steps):
        logger.info(f"Iteration {i} with tau = {v}")

        fitter.tau.assign(v)
        cb = fitter.minimize()
        val = compute_curvature(fitter).numpy()
        curvatures.append(val)

        logger.info(f"Curvature (value) = {val}")

    # set tau back to the original value
    fitter.tau.assign(tau0)

    return tau_steps, np.array(curvatures)


@tf.function
def neg_curvature_val_grad_hess(fitter):
    with tf.GradientTape() as t2:
        t2.watch(fitter.tau)
        with tf.GradientTape() as t1:
            t1.watch(fitter.tau)
            val = -1 * _compute_curvature(fitter)
        grad = t1.gradient(val, fitter.tau)
    hess = t2.gradient(grad, fitter.tau)

    return val, grad, hess


@tf.function
def neg_curvature_val_grad(fitter):
    with tf.GradientTape() as t1:
        t1.watch(fitter.tau)
        val = -1 * _compute_curvature(fitter)
    grad = t1.gradient(val, fitter.tau)

    return val, grad


@tf.function
def neg_curvature_val_grad_hessp(fitter, p):
    p = tf.stop_gradient(p)
    with tf.GradientTape() as t2:
        t2.watch(fitter.tau)
        with tf.GradientTape() as t1:
            t1.watch(fitter.tau)
            val = -1 * _compute_curvature(fitter)
        grad = t1.gradient(fitter.tau)
    hessp = t2.gradient(grad, fitter.tau, output_gradients=p)
    return val, grad, hessp


def l_curve_optimize_tau(fitter):
    # FIXME: this does not work yet
    # find the tau where the curvature is maximum, minimize curvature w.r.t. tau
    logger.info(f"Run l curve optimization")

    # def scipy_loss(xval):
    #     logger.info(f"=======")
    #     logger.info(f"x = {xval}")
    #     tau.assign(xval[0])
    #     cb = fitter.minimize()
    #     val, grad = neg_curvature_val_grad(fitter, tau)
    #     logger.info(f"Curvature (value) = {-val}")
    #     logger.info(f"Curvature (gradient) = {grad}")
    #     return val.__array__(), grad.__array__()

    # def scipy_hessp(xval, pval):
    #     tau.assign(xval[0])
    #     p = tf.convert_to_tensor(pval)
    #     val, grad, hessp = neg_curvature_val_grad_hessp(fitter, tau, p)
    #     return hessp.__array__()

    # def scipy_hess(xval):
    #     tau.assign(xval[0])
    #     val, grad, hess = neg_curvature_val_grad_hess(fitter, tau)
    #     return hess.__array__()
    #     # logger.debug(f"xval = {xval}")
    #     # logger.debug(f"p = {p}")
    #     # logger.debug(f"val = {val}; grad = {grad}; hessp = {hessp}")

    # res = scipy.optimize.minimize(
    #     scipy_loss,
    #     [0],
    #     method="trust-exact",
    #     jac=True,
    #     # hessp=scipy_hessp,
    #     hess=scipy_hess,
    #     options=dict(
    #         disp = True
    #     ),
    # )

    # tau.assign(res["x"][0])
    # val = res["fun"]

    # logger.info(f"Optimization terminated")

    # logger.info(f"  maximum curvature: {-val}")
    # logger.info(f"  tau: {tau.numpy()}")

    # return tau.numpy(), val.numpy()

    edm = 1
    i = 0
    while i < 50 and (edm < 0 or edm > 1e-16):
        logger.info(f"Iteration {i}")

        cb = fitter.minimize()
        val, grad, hess = neg_curvature_val_grad_hess(fitter)

        logger.info(f"Curvature (value) = {-val}")
        logger.info(f"Curvature (gradient) = {grad}")
        logger.info(f"Curvature (hessian) = {hess}")

        eps = 1e-8
        hess_sign = tf.where(hess != 0, tf.sign(hess), tf.ones_like(hess))
        safe_hess = hess_sign * tf.maximum(tf.abs(hess), eps)
        step = grad / safe_hess

        logger.info(f"Curvature (step) = {-step}")
        fitter.tau.assign_sub(step)

        edm = 0.5 * grad * step
        i = i + 1

        logger.debug(f"Curvature edm = {edm}")
        logger.debug(f"Curvature tau = {fitter.tau}")

    tau = fitter.tau.numpy()
    curvature = val.numpy()

    logger.info(f"Optimization terminated")
    logger.info(f"  edm: {edm}")
    logger.info(f"  maximum curvature: {-val}")
    logger.info(f"  tau: {tau}")

    return tau, curvature
