"""GLTR (trust-krylov) subproblem: Lanczos on device, tridiagonal solves on host.

The Generalized Lanczos Trust Region method of Gould, Lucidi, Roma and
Toint [1]_ -- the algorithm behind trlib and therefore scipy's
trust-krylov. Where Steihaug-CG (``krylov.py``) stops at the first
boundary crossing or negative-curvature direction, GLTR keeps expanding
the Krylov subspace K_k = span{g, Hg, ...} and returns the *optimal* step
within it: with q_1 = g/||g|| the Lanczos basis Q_k tridiagonalizes H
(Q_k^T H Q_k = T_k) and maps the subproblem to

    min  gamma_0 e_1.h + h.T_k h / 2   s.t. ||h|| <= Delta,

a tridiagonal trust-region problem solved here by eigendecomposition plus
a safeguarded secular-equation solve (hard case included). The full-space
residual of the candidate p = Q_k h is available for free as
gamma_k |h_k|, which is the convergence test.

Division of labour: the Hessian-vector products and the Lanczos vector
updates (including reorthogonalization) run on the TF device through one
compiled step function; the k x k tridiagonal solves run in numpy on the
host, where they cost microseconds against HVPs costing milliseconds.
Full CGS2 reorthogonalization is used -- two matmuls against the stored
basis per step, cheap on device -- where trlib by default trusts the
three-term recurrence.

The Krylov data (T_k, Q_k) does not depend on the trust radius, so a
re-solve at a smaller radius -- the outer loop's rejected-step path --
first re-solves the projected problem on the subspace already built and
extends only if the residual test fails: often zero new HVPs.

References
----------
.. [1] N.I. Gould, S. Lucidi, M. Roma, P.L. Toint, "Solving the
       trust-region subproblem using the Lanczos method", SIAM J. Optim.
       9(2), pp. 504-525, 1999.
"""

import math

import numpy as np
import scipy.linalg
import tensorflow as tf
from wums import logging

logger = logging.child_logger(__name__)


def solve_tridiag_trust_region(diag, offdiag, gamma0, Delta):
    """Solve min gamma0*e1.h + h.T h/2, ||h|| <= Delta for tridiagonal T.

    Parameters are numpy: ``diag`` [k], ``offdiag`` [k-1], scalars. Returns
    (h, lam, hits_boundary). Exact up to the scalar secular solve, hard
    case included; k is the Krylov dimension so everything here is cheap.
    """
    diag = np.asarray(diag, dtype=np.float64)
    offdiag = np.asarray(offdiag, dtype=np.float64)
    k = diag.size
    if k == 1:
        w = diag.reshape(1)
        V = np.ones((1, 1))
    else:
        w, V = scipy.linalg.eigh_tridiagonal(diag, offdiag)
    gproj = gamma0 * V[0, :]  # g in the eigenbasis (g -> gamma0 e_1)
    wmin = w[0]

    # interior solution
    if wmin > 0:
        h0 = -gproj / w
        if np.linalg.norm(h0) <= Delta:
            return V @ h0, 0.0, False

    lam_lo = max(0.0, -wmin)

    def hnorm(lam):
        return np.linalg.norm(gproj / (w + lam))

    # Hard case: g has (numerically) no component on the bottom eigenspace
    # and the secular equation has no root above lam_lo. Move along the
    # bottom eigenvector to the boundary instead.
    lam_eps = lam_lo + 1e-12 * max(1.0, abs(wmin)) + 1e-300
    degenerate = (w - wmin) <= 1e-12 * max(1.0, abs(wmin))
    if hnorm(lam_eps) < Delta:
        denom = np.where(degenerate, 1.0, w - wmin)
        h = np.where(degenerate, 0.0, -gproj / denom)
        tau = math.sqrt(max(Delta**2 - float(h @ h), 0.0))
        # move to the boundary along the first degenerate eigen-coordinate
        idx = int(np.argmax(degenerate))
        h[idx] += tau
        return V @ h, lam_lo, True

    # secular equation phi(lam) = 1/Delta - 1/||h(lam)|| is increasing and
    # concave on (lam_lo, inf); bracket then bisect to machine precision
    lo = lam_eps
    hi = max(lam_eps * 2, lam_lo + gamma0 / Delta + abs(wmin) + 1.0)
    while hnorm(hi) > Delta:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if hnorm(mid) > Delta:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15 * max(1.0, hi):
            break
    lam = 0.5 * (lo + hi)
    h = -gproj / (w + lam)
    return V @ h, lam, True


class GLTRSolver:
    """Device-side Lanczos machinery shared across solves.

    Holds the stored basis Q as a [kmax, n] variable (unused rows zero, so
    reorthogonalization against the full buffer needs no masking) and one
    compiled step function with fixed shapes -- no retracing per iteration.
    """

    def __init__(self, hessp_fn, kmax=None):
        self.hessp_fn = hessp_fn
        self.kmax = kmax
        self.Q = None
        self._step = tf.function(self._step_impl)

    def ensure(self, n, dtype):
        kmax = self.kmax or min(n, 1000)
        if self.Q is None or self.Q.shape != (kmax, n):
            self.Q = tf.Variable(tf.zeros([kmax, n], dtype=dtype), trainable=False)
        self.kmax = kmax
        return kmax

    def reset(self):
        if self.Q is not None:
            self.Q.assign(tf.zeros_like(self.Q))

    def _step_impl(self, k, q_prev, q, gamma_prev):
        """One Lanczos step: store q as row k, return (delta, gamma, q_next)."""
        self.Q.scatter_nd_update(k[None, None], q[None, :])
        Hq = self.hessp_fn(q)
        delta = tf.tensordot(q, Hq, axes=1)
        w = Hq - delta * q - gamma_prev * q_prev
        # CGS2 against the whole stored basis (zero rows are no-ops)
        for _ in range(2):
            coeffs = tf.linalg.matvec(self.Q, w)
            w = w - tf.linalg.matvec(self.Q, coeffs, transpose_a=True)
        gamma = tf.norm(w)
        tiny = tf.constant(np.finfo(np.float64).tiny, dtype=w.dtype)
        q_next = w / tf.maximum(gamma, tiny)
        return delta, gamma, q_next

    def step(self, k, q_prev, q, gamma_prev):
        return self._step(
            tf.constant(k, tf.int32),
            q_prev,
            q,
            tf.constant(gamma_prev, q.dtype),
        )

    def retransform(self, h):
        """p = Q_k^T h, padding h to the buffer size."""
        h_pad = np.zeros(int(self.Q.shape[0]), dtype=np.float64)
        h_pad[: h.size] = h
        return tf.linalg.matvec(
            self.Q, tf.constant(h_pad, self.Q.dtype), transpose_a=True
        )


class GLTRSubproblem:
    """Trust-region subproblem solved by GLTR.

    Same interface as the other native subproblems. The Lanczos data lives
    for the lifetime of the subproblem (one linearization point), so
    re-solves at shrunken radii after rejected steps reuse it.
    """

    def __init__(
        self, fun_val, jac, x_point, solver, set_point=None, cg_maxiter=None, tol=None
    ):
        self.fun = float(fun_val)
        self.jac = tf.convert_to_tensor(jac)
        self.jac_mag = float(tf.norm(self.jac))
        self._x_point = x_point
        self._solver = solver
        self._set_point = set_point
        self._cg_maxiter = cg_maxiter
        self._tol = tol
        self._last_p = None
        self._last_model = None
        # Krylov state at this linearization point
        self._deltas = []
        self._gammas = []  # gammas[i] connects q_{i+1} and q_{i+2}
        self._q = None
        self._q_prev = None
        self._started = False
        self.niter = 0

    # breakdown threshold: an invariant subspace has been found and the
    # projected solution is exact within it
    BREAKDOWN = 1e-14

    def _extend(self):
        """One more Lanczos vector; returns False on breakdown."""
        k = len(self._deltas)
        gamma_prev = self._gammas[k - 1] if k > 0 else 0.0
        delta, gamma, q_next = self._solver.step(k, self._q_prev, self._q, gamma_prev)
        self._deltas.append(float(delta))
        self._gammas.append(float(gamma))
        self._q_prev, self._q = self._q, q_next
        self.niter += 1
        return self._gammas[-1] > self.BREAKDOWN * max(1.0, abs(self._deltas[-1]))

    def solve(self, tr_radius):
        if self._tol is not None:
            tolerance = self._tol
        else:
            tolerance = min(0.5, math.sqrt(self.jac_mag)) * self.jac_mag

        if self.jac_mag == 0.0:
            p = np.zeros(int(self.jac.shape[0]), dtype=np.float64)
            self._last_p, self._last_model = p, self.fun
            return p, False

        if self._set_point is not None:
            self._set_point(self._x_point)

        n = int(self.jac.shape[0])
        kmax = self._solver.ensure(n, self.jac.dtype)
        if self._cg_maxiter is not None:
            kmax = min(kmax, self._cg_maxiter)

        if not self._started:
            self._solver.reset()
            self._q_prev = tf.zeros_like(self.jac)
            self._q = self.jac / self.jac_mag
            self._started = True

        h = lam = hits_boundary = None
        can_extend = True
        while True:
            k = len(self._deltas)
            if k > 0:
                # projected problem on the subspace built so far; its
                # full-space residual is gamma_k |h_k|
                h, lam, hits_boundary = solve_tridiag_trust_region(
                    self._deltas, self._gammas[: k - 1], self.jac_mag, tr_radius
                )
                residual = self._gammas[k - 1] * abs(h[-1])
                # boundary solutions warrant a tighter test (as in trlib):
                # there the residual measures suboptimality of the returned
                # step within the full space, not just distance to a Newton
                # point the outer loop would refine anyway
                tol_eff = 0.1 * tolerance if hits_boundary else tolerance
                if residual <= tol_eff or not can_extend:
                    break
                if k >= kmax:
                    logger.warning(
                        f"GLTR hit the subspace cap kmax={kmax} "
                        f"(residual {residual:.2e} > tol {tolerance:.2e}); "
                        "returning the best step in the subspace"
                    )
                    break
            can_extend = self._extend()

        p = self._solver.retransform(h)
        p_np = p.__array__()

        # model value from the tridiagonal data
        Th = np.array(self._deltas[: h.size]) * h
        if h.size > 1:
            off = np.asarray(self._gammas[: h.size - 1])
            Th[:-1] += off * h[1:]
            Th[1:] += off * h[:-1]
        mval = self.jac_mag * h[0] + 0.5 * float(h @ Th)

        self._last_p = p_np
        self._last_model = self.fun + mval
        logger.debug(
            f"gltr: {len(self._deltas)} Lanczos vectors, lam={lam:.3e}, "
            f"|p|={np.linalg.norm(p_np):.3e}, hits_boundary={hits_boundary}"
        )
        return p_np, bool(hits_boundary)

    def model_value(self, p):
        if p is self._last_p:
            return self._last_model
        if self._set_point is not None:
            self._set_point(self._x_point)
        pt = tf.convert_to_tensor(p, dtype=self.jac.dtype)
        Hp = self._solver.hessp_fn(pt)
        return (
            self.fun
            + float(tf.tensordot(self.jac, pt, 1))
            + 0.5 * float(tf.tensordot(pt, Hp, 1))
        )
