"""Hard constraints as a scipy ``NonlinearConstraint`` for ``trust-constr``.

Optimizer plumbing only: what to evaluate comes in as callables from the
fitter (:meth:`rabbit.fitter.Fitter.eval_physical` /
:meth:`~rabbit.fitter.Fitter.hvp_physical` on the concatenated
:class:`~rabbit.constraints.constraints.Constraints` rows), so nothing here
touches the fitter's state.
"""

import numpy as np
import scipy
from wums import logging

logger = logging.child_logger(__name__)


def build_nonlinear_constraints(
    val_jac, hvp, xstart, lower, upper, labels, to_physical=None, hold_feasible=True
):
    """Build the scipy constraint for the rows ``lower <= c(x) <= upper``.

    Parameters
    ----------
    val_jac : callable
        ``x -> (c, J)``: all rows' values [m] and Jacobian [m, n] at the
        fitter coordinate ``x``, frozen columns already zero.
    hvp : callable
        ``(x, v) -> LinearOperator``: the Hessian of ``sum_i v_i c_i`` at
        ``x``, ``v`` of length m (all rows).
    xstart : array
        The starting point, in the fitter coordinate.
    lower, upper, labels : arrays / list, length m.
    to_physical : callable, optional
        Maps the minimizer's iterate to the fitter coordinate. fit() works in
        coordinates SHIFTED from it (y = x - x_start, even with
        preconditioning off) and rebuilds the shift on every restart, so the
        map is looked up at call time. Only pure shifts are supported
        (Jacobian dx/dy = 1, so neither J nor the Hessian needs a transform).
    hold_feasible : bool
        False: no row is held feasible (``keep_feasible`` all False), the
        warm-start rule, see :mod:`rabbit.constraints.warm_start`.

    Returns
    -------
    (constraints, info) : a list with zero or one ``NonlinearConstraint``,
    and a dict with the kept rows (``live``), their bounds, labels, the
    per-row ``keep_feasible`` and the start values.

    Rows that no free parameter can move (all-zero Jacobian row: every
    parameter they read is frozen) are dropped as constants, and the fit is
    refused if one of them is violated -- a frozen parameter parked outside
    the allowed region is a misconfiguration, not something to optimize
    around.
    """
    if to_physical is None:

        def to_physical(v):
            return v

    lower = np.reshape(np.asarray(lower, dtype=np.float64), [-1])
    upper = np.reshape(np.asarray(upper, dtype=np.float64), [-1])
    labels = [str(lab) for lab in labels]

    c_all, j_all = val_jac(np.asarray(xstart))
    if not (c_all.shape == lower.shape == upper.shape and len(labels) == len(lower)):
        raise ValueError(
            f"constraints returned {c_all.shape} values for {lower.shape} lower "
            f"bounds, {upper.shape} upper bounds and {len(labels)} labels"
        )
    m = len(lower)
    live = np.abs(j_all).sum(axis=1) > 0
    if not live.all():
        dead_bad = int(
            np.sum((~live) & ((c_all < lower - 1e-12) | (c_all > upper + 1e-12)))
        )
        if dead_bad:
            raise ValueError(
                f"{dead_bad} constraint(s) depending only on frozen parameters "
                "are VIOLATED at the start; the fit cannot satisfy them."
            )
        logger.info(
            f"[minimize] {int((~live).sum())} constraint(s) depend only on frozen "
            "parameters and are satisfied; dropped"
        )
    info = {"live": live}
    if not live.any():
        return [], info
    lb, ub, c0 = lower[live], upper[live], c_all[live]
    labels = [lab for lab, keep in zip(labels, live) if keep]

    # keep_feasible, per row. scipy holds a kept row's slack strictly
    # positive, so a row that starts EXACTLY on its bound can never move
    # (every trial step is rejected; for min (x-0.4)^2, x >= 0 from x = 0
    # scipy reports convergence at 0). Rows active at the start are therefore
    # not held feasible; the barrier still pulls them back. An infeasible
    # start cannot be held feasible at all (scipy would return it unchanged
    # and report success), so then no row is.
    finite_abs = np.maximum(
        np.abs(np.where(np.isfinite(lb), lb, 0.0)),
        np.abs(np.where(np.isfinite(ub), ub, 0.0)),
    )
    atol = 64.0 * np.finfo(np.float64).eps * np.maximum(1.0, finite_abs)
    viol = np.maximum(lb - c0, c0 - ub)
    if not hold_feasible:
        keep = np.zeros(len(lb), dtype=bool)
        logger.info(
            "[minimize] warm start: no constraint is held feasible, so a slack "
            "pinned at ~0 on a face of the previous minimum cannot freeze the fit"
        )
    elif np.any(viol > atol):
        keep = np.zeros(len(lb), dtype=bool)
        logger.warning(
            f"[minimize] starting point violates {int(np.sum(viol > atol))}/"
            f"{len(lb)} constraint(s) (worst by {float(np.max(viol)):.3g}); "
            "keep_feasible=False so the minimizer can walk INTO the allowed region."
        )
    else:
        on_bound = (np.abs(c0 - lb) <= atol) | (np.abs(ub - c0) <= atol)
        keep = ~on_bound
        if on_bound.any():
            logger.info(
                f"[minimize] {int(on_bound.sum())} constraint(s) start exactly on "
                "their bound and are not held feasible, so the first step can "
                "leave it: " + "; ".join(lab for lab, b in zip(labels, on_bound) if b)
            )
    logger.info(
        f"[minimize] {len(lb)} hard constraint(s); start values: "
        + "; ".join(f"{lab} = {c:.6g}" for lab, c in zip(labels, c0))
    )
    info.update(lower=lb, upper=ub, labels=labels, keep_feasible=keep, start_values=c0)

    # scipy calls fun and jac at the same point; evaluate once per point.
    cache = {"key": None, "val": None, "jac": None}

    def cached(y):
        x = np.asarray(to_physical(y))
        key = x.tobytes()
        if cache["key"] != key:
            c, j = val_jac(x)
            cache.update(key=key, val=c[live], jac=j[live])
        return cache["val"], cache["jac"]

    def hess(y, v):
        vfull = np.zeros(m)
        vfull[live] = np.asarray(v, dtype=np.float64)
        return hvp(np.asarray(to_physical(y)), vfull)

    return [
        scipy.optimize.NonlinearConstraint(
            fun=lambda y: cached(y)[0],
            lb=lb,
            ub=ub,
            jac=lambda y: cached(y)[1],
            hess=hess,
            keep_feasible=keep,
        )
    ], info
