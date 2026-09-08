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
    "The subproblem could not predict descent at any trust radius, but the "
    "gradient is not small: the quadratic MODEL failed, the fit did not "
    "converge, and the returned point is not a minimum.",
)

# `predicted_reduction <= 0` at a shrinking trust radius is the minimum only if
# the subproblem is solved accurately. Below this fraction of |fun| the
# first-order gain of ANY step is float noise, and stopping is right; above it
# the model is wrong, not the point.
MODEL_FAILURE_RTOL = 1e-12
# each retry costs a re-solve at a quarter of the radius; GLTR reuses its
# radius-independent Krylov data, so a retry is usually free of new HVPs
MAX_MODEL_SHRINKS = 20


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
    model_shrinks = 0
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

        # THE MODEL SAYS IT CANNOT IMPROVE. That is a statement about the
        # minimum only when the subproblem was solved accurately. For a Krylov
        # subproblem it is equally the signature of a model that has gone bad:
        # GLTR's Lanczos recurrence loses orthogonality at high condition
        # number, and the truncated model then predicts no descent while the
        # true objective still has a long way to fall.
        #
        # MEASURED. On a Z card with a 3.4e12-conditioned Hessian,
        # tf-trust-krylov stopped here and reported convergence **14.7 NLL
        # units above the minimum**, with its own EDM reading 14.72 -- the two
        # agreeing to three digits, i.e. the fit knew exactly how much it had
        # left to gain and stopped anyway. The POIs had barely left their
        # starting values. Treating that as success is the worst failure a
        # minimizer can have, because the answer looks clean.
        #
        # So: shrink and retry. At a small enough radius the model IS the local
        # quadratic and must predict descent whenever the gradient is non-zero,
        # so a retry either recovers the fit or proves the point stationary.
        # The Cauchy first-order gain `radius * |g|` is what separates them --
        # below float noise on `fun` no step of any kind can help, and stopping
        # is correct; above it the model is wrong, not the point, and the fit
        # must say so rather than return a number.
        if predicted_reduction <= 0:
            noise = MODEL_FAILURE_RTOL * max(abs(m.fun), 1.0)
            if trust_radius * m.jac_mag <= noise:
                warnflag = 2  # genuinely stationary: no step can gain anything
                break
            if model_shrinks >= MAX_MODEL_SHRINKS:
                warnflag = 4
                break
            model_shrinks += 1
            trust_radius *= 0.25
            logger.debug(
                f"subproblem predicted no descent at radius "
                f"{trust_radius * 4:.3g} with |g| = {m.jac_mag:.3g}; "
                f"retrying at {trust_radius:.3g}"
            )
            continue
        model_shrinks = 0
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
            # trust_radius travels with the iterate so a snapshot can record
            # it: resuming a long fit at radius 1.0 throws away however many
            # rejections it took to find the right scale, and on an expensive
            # objective those are the whole cost of the first iterations.
            callback(
                OptimizeResult(
                    x=np.copy(x), fun=float(m.fun), trust_radius=trust_radius
                )
            )

        if m.jac_mag < gtol:
            warnflag = 0
            break

    success = warnflag == 0
    if warnflag == 2:
        # the standard end state of a converged fit run with gtol=0
        logger.debug(_STATUS_MESSAGES[warnflag])
    elif warnflag == 4:
        logger.warning(
            _STATUS_MESSAGES[warnflag]
            + f" |g| = {m.jac_mag:.6g} after {MAX_MODEL_SHRINKS} radius "
            f"reductions to {trust_radius:.3g}. Re-run with "
            "--minimizerMethod tf-trust-exact, which solves the subproblem "
            "accurately, and compare the NLL."
        )
    elif not success:
        logger.warning(_STATUS_MESSAGES[warnflag])

    return OptimizeResult(
        x=x,
        fun=float(m.fun),
        trust_radius=trust_radius,
        jac=np.asarray(m.jac),
        success=success,
        status=warnflag,
        nit=k,
        nfev=nfev,
        nhev=nhev,
        message=_STATUS_MESSAGES[warnflag],
    )


def minimize_trust_exact(
    fun, closure, x0, gtol=0.0, maxiter=None, callback=None,
    initial_trust_radius=1.0,
):
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
        initial_trust_radius=initial_trust_radius,
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
    initial_trust_radius=1.0,
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
        initial_trust_radius=initial_trust_radius,
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
    initial_trust_radius=1.0,
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
        initial_trust_radius=initial_trust_radius,
        subproblem_kwargs=dict(
            solver=solver, set_point=set_point, cg_maxiter=cg_maxiter
        ),
    )
