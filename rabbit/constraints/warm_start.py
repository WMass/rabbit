"""Warm-starting ``trust-constr`` from (near) a constrained minimum.

scipy's interior-point method has no notion of a warm start, and three of its
start-up choices undo one (measured on a toy with 40 parameters and 8 rows at
production tolerances, and on rabbit's test model; numbers in the
ct18z-nominal-fits/261015-trustconstr-warmstart logbook):

1. The slack of every row NOT held feasible starts at
   ``s0 = max(1.5 * (c - lb), 1)`` (``tr_interior_point``), not at ``c - lb``.
   A row on its face (or a violated one) therefore starts with ``s = 1``,
   and the first iterations restore ``c - lb = s`` by moving ``x`` into the
   interior: a loss excursion of O(1) on the toy, +333 NLL on the real fit.
   (Rows held feasible are not affected: scipy overwrites their slack in place
   with ``c - lb``.)
2. A row held feasible (``keep_feasible``) has its slack pinned to ``c - lb``.
   At a previous minimum that is ~1e-10 on every active face. When the
   problem changed (a saturated sub-fit, a scan point, a toy), held CURVED
   rows reject the steps that cross them, the trust radius collapses, and
   scipy stops with status 2 ("success") or at maxiter, away from the new
   minimum.
3. The first barrier stage uses ``mu0 = 0.1``; ``mu`` falls only x0.2 per
   converged stage, so a start already at ``mu ~ 1e-9`` re-descends ~11
   stages, and on the way the barrier pushes the active faces ~mu/multiplier
   into the interior.

The warm start here holds no row feasible, starts each slack consistently
with ``x0`` (:func:`consistent_slacks`), and starts the barrier ON the central
path of the start: ``mu0 = max_i mu_i * s_i`` over the rows near a bound
(:func:`centred_barrier`), which for the restart of a converged fit is the
barrier parameter it ended on. Not a fixed small value: an off-centred start
(``mu_i * s_i`` far from ``mu0``, in either direction) leaves a slack residual
that scipy's projected CG, with its relative stopping rule, skips next to the
~1e-7 gradient left in the other directions, and with the noise of rabbit's
loss (~1e-12) every step is then rejected: the fit sits where it started until
the trust radius collapses (measured on rabbit's saturated test).

scipy's stopping test (the KKT residual with least-squares multipliers) does
not check the multipliers' SIGN, so a small barrier on a face the loss wants
to leave can stop there ("success"). It does, and not only in theory: in the
scaled interior-point step the loss gradient along a face's slack is
``mu_i * s_i``, ~1e-17 at a slack of 1e-9, and the barrier's push is ``mu``,
~1e-8, so neither survives scipy's relative CG stop next to the other
directions' residual, nor rabbit's loss noise; the trust radius then
collapses ON the face (status 2). Only a large barrier pushes off a face,
and scipy's own start (mu = 0.1) does that by walking the whole fit away.
:func:`releasing_rows` detects the case; :func:`release_direction` gives the
step that leaves the releasing faces while holding the binding ones, which
the caller line-searches before starting the next warm leg from there.

scipy exposes no ``s0``. :func:`consistent_slacks` sets it through a subclass
of the private ``BarrierSubproblem``, relying on two details of scipy's
``tr_interior_point`` (checked at run time by
:func:`scipy_supports_slack_override` and by
tests/test_trust_constr_warmstart.py): the ``s0`` array it passes to
``BarrierSubproblem`` is the same one it then stacks into the starting point
``z = hstack((x0, s0))``, and the constructor's arguments are named ``s0``
and ``constr_ineq0``. Written against scipy 1.18.
"""

import contextlib
import inspect

import numpy as np
from wums import logging

logger = logging.child_logger(__name__)

#: Barrier parameter of a warm start when no row near its bound carries a
#: binding multiplier (nothing to centre on): the scale a converged fit ends
#: on (rabbit's production --minimizerBarrierTol is 1e-9).
WARM_INITIAL_BARRIER = 1e-9
#: Range the centred barrier parameter is clipped to (see centred_barrier).
WARM_BARRIER_RANGE = (1e-12, 1e-3)
#: Slack floor for rows on or across their bound, relative to the barrier
#: parameter (the toy: 1e-3 * mu0 is fastest; 1 * mu0 costs 2-3x iterations).
WARM_SLACK_FLOOR_REL = 1e-3


def _tip_module():
    from scipy.optimize._trustregion_constr import tr_interior_point

    return tr_interior_point


def scipy_supports_slack_override():
    """(ok, reason): whether :func:`consistent_slacks` can work on this scipy."""
    try:
        tip = _tip_module()
        sub = tip.BarrierSubproblem
        names = list(inspect.signature(sub.__init__).parameters)
        src = inspect.getsource(tip.tr_interior_point)
    except Exception as e:  # private API moved or renamed
        return False, f"scipy trust-constr internals not found ({e})"
    for name in ("s0", "constr_ineq0"):
        if name not in names:
            return False, f"BarrierSubproblem.__init__ has no argument '{name}'"
    # the override needs the s0 handed to BarrierSubproblem to be the one
    # stacked into the start z afterwards
    i_sub = src.find("BarrierSubproblem(")
    i_z = src.find("z = np.hstack((x0, s0))")
    if i_sub < 0 or i_z < 0 or i_z < i_sub:
        return False, "tr_interior_point no longer builds z from the s0 it passes on"
    return True, ""


@contextlib.contextmanager
def consistent_slacks(floor):
    """Within the block, trust-constr starts each slack at ``max(c - lb, floor)``.

    ``floor`` is a scalar or one value per canonical inequality row. Only the
    START changes: the iteration itself is scipy's. Not thread safe (it swaps a
    module attribute of scipy for the duration of the block).
    """
    ok, why = scipy_supports_slack_override()
    if not ok:
        raise RuntimeError(f"cannot set trust-constr's starting slacks: {why}")
    tip = _tip_module()
    orig = tip.BarrierSubproblem
    sig = inspect.signature(orig.__init__)

    class _ConsistentSlackSubproblem(orig):
        def __init__(self, *args, **kwargs):
            bound = sig.bind(self, *args, **kwargs)
            s0 = bound.arguments["s0"]
            c_ineq0 = np.asarray(bound.arguments["constr_ineq0"], dtype=np.float64)
            # canonical rows are constr_ineq(x) <= 0, slack s = -constr_ineq;
            # in place: tr_interior_point stacks this very array into z
            s0[:] = np.maximum(-c_ineq0, floor)
            super().__init__(*args, **kwargs)

    tip.BarrierSubproblem = _ConsistentSlackSubproblem
    try:
        yield
    finally:
        tip.BarrierSubproblem = orig


def _near_multipliers(grad, cval, jac, lower, upper, floating, near_tol):
    """Least-squares multipliers of the rows near a bound.

    Returns (rows, side, mu, distance): side +1 for a lower bound, -1 for an
    upper one (a multiplier of the binding sign has side * mu > 0); distance
    to the bound, >= 0 inside the allowed region. ``near_tol`` is relative to
    ``max(1, |bound|)``; rows only frozen parameters move are left out.
    """
    g = np.asarray(grad, dtype=np.float64).reshape(-1)
    J = np.atleast_2d(np.asarray(jac, dtype=np.float64))
    c = np.asarray(cval, dtype=np.float64).reshape(-1)
    lb = np.asarray(lower, dtype=np.float64).reshape(-1)
    ub = np.asarray(upper, dtype=np.float64).reshape(-1)
    F = np.arange(len(g)) if floating is None else np.asarray(floating, dtype=int)
    empty = (np.zeros(0, dtype=int), np.zeros(0), np.zeros(0), np.zeros(0))
    if not len(c) or not len(F):
        return empty

    def near(b):
        fin = np.isfinite(b)
        scale = np.maximum(1.0, np.abs(np.where(fin, b, 0.0)))
        return fin & (np.abs(c - np.where(fin, b, 0.0)) <= near_tol * scale)

    near_lo, near_up = near(lb), near(ub)
    live = np.abs(J[:, F]).sum(axis=1) > 0
    rows = np.flatnonzero((near_lo ^ near_up) & live)  # equality rows: no sign
    if not len(rows):
        return empty
    mu = np.linalg.lstsq(J[np.ix_(rows, F)].T, g[F], rcond=None)[0]
    side = np.where(near_lo[rows], 1.0, -1.0)
    dist = np.where(side > 0, c[rows] - lb[rows], ub[rows] - c[rows])
    return rows, side, mu, dist


def centred_barrier(grad, cval, jac, lower, upper, floating=None, near_tol=1e-3):
    """The barrier parameter that puts the start on the central path.

    ``max_i mu_i * s_i`` over the rows within ``near_tol`` of a bound, ``mu_i``
    their least-squares multipliers of the binding sign and ``s_i`` the
    distance to the bound; clipped to WARM_BARRIER_RANGE, and
    WARM_INITIAL_BARRIER if no such row binds. For the restart of a converged
    trust-constr fit this is the barrier parameter it ended on. The max, not a
    mean: every other row then starts at or inside its own centre, which the
    barrier stages walk down together, as on a cold descent.

    Parameters as in :func:`rabbit.constraints.postfit.face_held_covariance`.
    """
    rows, side, mu, dist = _near_multipliers(
        grad, cval, jac, lower, upper, floating, near_tol
    )
    comp = np.maximum(side * mu, 0.0) * np.maximum(dist, 0.0)
    if not len(comp) or not np.any(comp > 0):
        return WARM_INITIAL_BARRIER
    return float(np.clip(np.max(comp), *WARM_BARRIER_RANGE))


def release_direction(jac, release, release_side, hold, floating=None):
    """Unit step direction that moves the ``release`` rows into the allowed region
    (``release_side`` +1 for a lower bound, -1 for an upper one) while holding
    the ``hold`` rows on their faces to first order (projected out of the span
    of their gradients). Zero on non-floating parameters; None if nothing is
    left after the projection.
    """
    J = np.atleast_2d(np.asarray(jac, dtype=np.float64))
    n = J.shape[1]
    F = np.arange(n) if floating is None else np.asarray(floating, dtype=int)
    JF = J[:, F]
    d = np.zeros(len(F))
    for i, sd in zip(release, release_side):
        norm = np.linalg.norm(JF[i])
        if norm > 0:
            d += sd * JF[i] / norm
    hold = np.asarray(hold, dtype=int)
    if len(hold):
        A = JF[hold]
        d -= A.T @ np.linalg.lstsq(A @ A.T, A @ d, rcond=None)[0]
    norm = np.linalg.norm(d)
    if not norm > 0:
        return None
    out = np.zeros(n)
    out[F] = d / norm
    return out


def classify_near_rows(
    grad, cval, jac, lower, upper, floating=None, near_tol=1e-6, rel_tol=1e-6
):
    """(rows, side, mu, distance, releasing) for the rows near a bound, see
    :func:`releasing_rows`; ``releasing`` is a boolean mask over ``rows``."""
    rows, side, mu, dist = _near_multipliers(
        grad, cval, jac, lower, upper, floating, near_tol
    )
    if not len(rows):
        return rows, side, mu, dist, np.zeros(0, dtype=bool)
    tol = rel_tol * max(1.0, float(np.max(np.abs(mu))))
    return rows, side, mu, dist, side * mu < -tol


def releasing_rows(
    grad,
    cval,
    jac,
    lower,
    upper,
    labels,
    floating=None,
    near_tol=1e-6,
    rel_tol=1e-6,
):
    """Rows on (near) a bound whose multiplier says the loss wants them released.

    At a constrained minimum ``grad = J_A^T mu`` over the rows A near a bound,
    with ``mu >= 0`` on a lower bound and ``mu <= 0`` on an upper one. A
    multiplier of the other sign means moving off that face lowers the loss,
    so the point is not a minimum, whatever scipy's status says (its KKT test
    takes the least-squares multipliers without their sign).

    Parameters as in :func:`rabbit.constraints.postfit.face_held_covariance`;
    ``near_tol`` is relative to ``max(1, |bound|)`` and ``rel_tol`` to the
    largest multiplier magnitude (at least 1).

    Returns a list of ``(label, multiplier, distance)`` for the offending rows.
    """
    rows, side, mu, dist, bad = classify_near_rows(
        grad, cval, jac, lower, upper, floating, near_tol, rel_tol
    )
    return [
        (str(labels[i]), float(m), float(d))
        for i, m, d in zip(rows[bad], mu[bad], dist[bad])
    ]
