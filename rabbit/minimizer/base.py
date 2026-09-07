"""Native trust-region outer loop.

Mirrors scipy's ``_minimize_trust_region`` (``_trustregion.py``) closely
enough that the fitter's callback / early-stopping / restart plumbing works
unchanged: the callback is invoked once per iteration with an
``OptimizeResult``-shaped intermediate result, and the returned
``OptimizeResult`` uses scipy's status codes. Running the loop in python is
deliberate -- it executes once per outer iteration, so its overhead is
irrelevant; the point of the native path is that the *subproblem* keeps the
Hessian and its factorizations on the TF device instead of round-tripping
through numpy/LAPACK per lambda trial.

One efficiency difference from scipy, with identical iterates: scipy's
``IterativeSubproblem`` computes the Hessian eagerly at every *proposed*
point (its constructor consumes Hessian norms), so rejected steps each pay
a full Hessian. Here a proposal is judged on its objective value alone and
the (val, grad, hess) closure runs only when a step is accepted.

The objective is split into two callables:

``fun(x)``      -> float                        (cheap, judges proposals)
``closure(x)``  -> (float, grad, hess) tensors  (expensive, accepted steps)
"""

import numpy as np
import tensorflow as tf
from scipy.optimize import OptimizeResult
from wums import logging

from .exact import IterativeSubproblem
from .gltr import GLTRSolver, GLTRSubproblem
from .krylov import CGSteihaugSubproblem, SteihaugCGSolver

logger = logging.child_logger(__name__)

_warned_no_gpu = False


def _warn_if_no_gpu():
    # TF's CPU Cholesky kernel is single-threaded Eigen; measured ~7x slower
    # than scipy's LAPACK trust-exact at n=2000. The native path is built for
    # devices where the factorization is fast and the transfer is not.
    global _warned_no_gpu
    if not _warned_no_gpu and not tf.config.list_logical_devices("GPU"):
        logger.warning(
            "tf-trust-exact without a visible GPU: the on-device factorizations "
            "fall back to TF's single-threaded CPU kernel, and scipy trust-exact "
            "is typically faster in that case"
        )
        _warned_no_gpu = True


_STATUS_MESSAGES = (
    "Optimization terminated successfully.",
    "Maximum number of iterations has been exceeded.",
    "A bad approximation caused failure to predict improvement.",
    "A linalg error occurred, such as a non-psd Hessian.",
)


def _minimize_trust_region(
    fun,
    closure,
    x0,
    subproblem_cls,
    initial_trust_radius=1.0,
    max_trust_radius=1000.0,
    eta=0.15,
    gtol=1e-4,
    maxiter=None,
    callback=None,
    subproblem_kwargs=None,
):
    if not (0 <= eta < 0.25):
        raise ValueError("invalid acceptance stringency")
    if max_trust_radius <= 0:
        raise ValueError("the max trust radius must be positive")
    if initial_trust_radius <= 0:
        raise ValueError("the initial trust radius must be positive")
    if initial_trust_radius >= max_trust_radius:
        raise ValueError(
            "the initial trust radius must be less than the max trust radius"
        )

    x = np.asarray(x0, dtype=np.float64).copy()
    if maxiter is None:
        maxiter = len(x) * 200
    subproblem_kwargs = subproblem_kwargs or {}

    m = subproblem_cls(*closure(x), **subproblem_kwargs)
    nfev = 1
    nhev = 1

    trust_radius = float(initial_trust_radius)
    warnflag = 1  # maxiter, unless something else ends the loop
    k = 0
    while k < maxiter:
        try:
            p, hits_boundary = m.solve(trust_radius)
        except (np.linalg.LinAlgError, ValueError) as ex:
            logger.warning(f"trust-region subproblem failed: {ex}")
            warnflag = 3
            break

        predicted_value = m.model_value(p)
        x_proposed = x + p
        fun_proposed = fun(x_proposed)
        nfev += 1

        actual_reduction = m.fun - fun_proposed
        predicted_reduction = m.fun - predicted_value

        # at the minimum the model cannot predict further improvement beyond
        # float cancellation; with gtol=0 this is the terminating criterion,
        # exactly as for scipy's trust-region methods under tol=0.0
        if predicted_reduction <= 0:
            warnflag = 2
            break
        rho = actual_reduction / predicted_reduction
        # A non-finite proposal value must count as a hard rejection. IEEE
        # comparisons on a NaN rho are all False, which would neither shrink
        # the radius nor accept the step -- freezing the loop at a fixed
        # radius until early stopping gives up far from the minimum.
        # Observed with preconditioned coordinates, where an internal step
        # of norm 1 can be an enormous physical step whose loss overflows.
        if not np.isfinite(rho):
            rho = -np.inf

        if rho < 0.25:
            trust_radius *= 0.25
        elif rho > 0.75 and hits_boundary:
            trust_radius = min(2 * trust_radius, max_trust_radius)

        if rho > eta:
            x = x_proposed
            m = subproblem_cls(*closure(x), **subproblem_kwargs)
            nfev += 1
            nhev += 1

        k += 1

        # the callback may raise (NaN loss, early stopping); the caller's
        # restart machinery relies on that propagating
        if callback is not None:
            callback(OptimizeResult(x=np.copy(x), fun=float(m.fun)))

        if m.jac_mag < gtol:
            warnflag = 0
            break

    success = warnflag == 0
    if warnflag == 2:
        # the standard end state of a converged fit run with gtol=0
        logger.debug(_STATUS_MESSAGES[warnflag])
    elif not success:
        logger.warning(_STATUS_MESSAGES[warnflag])

    return OptimizeResult(
        x=x,
        fun=float(m.fun),
        jac=np.asarray(m.jac),
        success=success,
        status=warnflag,
        nit=k,
        nfev=nfev,
        nhev=nhev,
        message=_STATUS_MESSAGES[warnflag],
    )


def minimize_trust_exact(fun, closure, x0, gtol=0.0, maxiter=None, callback=None):
    """Native nearly-exact trust-region minimization (cf. scipy trust-exact).

    Parameters
    ----------
    fun : callable
        x (numpy) -> float. Objective only, used to judge proposed steps.
    closure : callable
        x (numpy) -> (float, grad, hess) with gradient and dense Hessian as
        tf tensors (any coordinates, as long as fun/closure agree). Called
        once per accepted step.
    x0 : ndarray
        Starting point.
    gtol : float
        Gradient-norm termination threshold. The default 0.0 matches the
        fitter's historical tol=0.0 scipy setup: run until the quadratic
        model predicts no further improvement.
    maxiter : int or None
        Maximum outer iterations (None: 200 * len(x0), as scipy).
    callback : callable or None
        Called once per iteration with an OptimizeResult(x=..., fun=...).

    Returns
    -------
    scipy.optimize.OptimizeResult
    """
    _warn_if_no_gpu()
    return _minimize_trust_region(
        fun,
        closure,
        x0,
        subproblem_cls=IterativeSubproblem,
        gtol=gtol,
        maxiter=maxiter,
        callback=callback,
    )


def minimize_trust_ncg(
    fun,
    closure,
    hessp,
    set_point,
    x0,
    gtol=0.0,
    maxiter=None,
    callback=None,
    cg_maxiter=None,
):
    """Native matrix-free trust-region minimization (cf. scipy trust-ncg).

    Same outer loop as :func:`minimize_trust_exact`, with the Steihaug-CG
    subproblem running as one TF graph call per solve. ``closure`` here only
    needs (float, grad); the Hessian never materializes.

    Parameters
    ----------
    fun : callable
        x (numpy) -> float, judges proposed steps.
    closure : callable
        x (numpy) -> (float, grad[, ...]) with the gradient a tf tensor in
        the same coordinates as ``hessp``. Called once per accepted step.
    hessp : callable
        Graph-compatible v -> H @ v at the fitter's current point.
    set_point : callable or None
        x (numpy) -> None; re-pins the fitter state to the subproblem's
        linearization point before HVPs run (``fun`` evaluations at proposed
        points move that state in between).
    cg_maxiter : int or None
        Cap on CG iterations per solve (None: the dimension).
    """
    solver = SteihaugCGSolver(hessp)

    def closure2(x):
        out = closure(x)
        x_pinned = np.array(x, dtype=np.float64, copy=True)
        return out[0], out[1], x_pinned

    return _minimize_trust_region(
        fun,
        closure2,
        x0,
        subproblem_cls=CGSteihaugSubproblem,
        gtol=gtol,
        maxiter=maxiter,
        callback=callback,
        subproblem_kwargs=dict(
            solver=solver, set_point=set_point, cg_maxiter=cg_maxiter
        ),
    )


def minimize_trust_krylov(
    fun,
    closure,
    hessp,
    set_point,
    x0,
    gtol=0.0,
    maxiter=None,
    callback=None,
    cg_maxiter=None,
):
    """Native GLTR trust-region minimization (cf. scipy trust-krylov).

    Same contract as :func:`minimize_trust_ncg`; the subproblem is solved
    to optimality within the Krylov subspace (Lanczos on device,
    tridiagonal solves on host) instead of truncated at the boundary, and
    re-solves after rejected steps reuse the radius-independent Krylov
    data, often costing no new Hessian-vector products.
    """
    solver = GLTRSolver(hessp, kmax=cg_maxiter)

    def closure2(x):
        out = closure(x)
        x_pinned = np.array(x, dtype=np.float64, copy=True)
        return out[0], out[1], x_pinned

    return _minimize_trust_region(
        fun,
        closure2,
        x0,
        subproblem_cls=GLTRSubproblem,
        gtol=gtol,
        maxiter=maxiter,
        callback=callback,
        subproblem_kwargs=dict(
            solver=solver, set_point=set_point, cg_maxiter=cg_maxiter
        ),
    )
