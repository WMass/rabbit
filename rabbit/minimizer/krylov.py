"""Matrix-free trust-region subproblem: Steihaug-Toint CG in a TF graph.

The HVP-based counterpart of ``exact.py``, algorithmically the same
truncated-CG subproblem as scipy's trust-ncg (``CGSteihaugSubproblem`` in
``scipy/optimize/_trustregion_ncg.py``; scipy's trust-krylov solves the same
problem with the heavier GLTR/Lanczos machinery). What the port changes is
where it runs: the whole CG iteration is a single ``tf.while_loop`` inside
one ``tf.function`` call, so a solve costs one graph dispatch total instead
of one python round trip -- x assignment, numpy conversion both ways, and a
forced device sync -- per Hessian-vector product, which is what the scipy
callbacks pay today and what dominates when individual HVPs are fast.

Two further differences from scipy's loop, both exact:

- ``Hz`` (the Hessian applied to the current CG point) is carried through
  the recurrence (``Hz_next = Hz + alpha*Bd``), so the model value along
  ``z + t*d`` at the boundary and negative-curvature exits is a few dot
  products; scipy evaluates the full quadratic there, costing two extra
  HVPs on the exit iteration.
- the model value of the returned step falls out of the same bookkeeping
  and is handed back to the outer loop, which otherwise would need one more
  HVP per proposal to price it.
"""

import math

import numpy as np
import tensorflow as tf
from wums import logging

logger = logging.child_logger(__name__)


class SteihaugCGSolver:
    """Compiled Steihaug-CG solve for a fixed HVP callable.

    ``hessp_fn`` must be graph-compatible (a tf.function or plain tf ops)
    mapping a vector to H @ vector at the current linearization point; the
    caller re-pins that point before each solve. One instance per fit (or
    restart) so the traced graph is reused across outer iterations.
    """

    def __init__(self, hessp_fn):
        self.hessp_fn = hessp_fn
        self._graph = tf.function(self._solve_impl)

    def solve(self, jac, tr_radius, tolerance, maxiter):
        dtype = jac.dtype
        return self._graph(
            jac,
            tf.constant(tr_radius, dtype),
            tf.constant(tolerance, dtype),
            tf.constant(int(maxiter), tf.int32),
        )

    def _solve_impl(self, jac, trust_radius, tolerance, maxiter):
        dtype = jac.dtype
        zero = tf.zeros([], dtype)
        zeros = tf.zeros_like(jac)

        def dot(a, b):
            return tf.tensordot(a, b, axes=1)

        def cond(k, z, r, d, Hz, p, mval, hb, done):
            return tf.logical_and(tf.logical_not(done), k < maxiter)

        def body(k, z, r, d, Hz, p, mval, hb, done):
            Bd = self.hessp_fn(d)
            dBd = dot(d, Bd)
            rr = dot(r, r)

            # ||z + t d|| = trust_radius, numerically stable roots
            a = dot(d, d)
            b = 2.0 * dot(z, d)
            c = dot(z, z) - trust_radius * trust_radius
            disc = tf.sqrt(tf.maximum(b * b - 4.0 * a * c, zero))
            aux = tf.where(b >= zero, b + disc, b - disc)
            safe_aux = tf.where(tf.equal(aux, zero), tf.ones_like(aux), aux)
            safe_a = tf.where(tf.equal(a, zero), tf.ones_like(a), a)
            t1 = -aux / (2.0 * safe_a)
            t2 = -2.0 * c / safe_aux
            t_lo = tf.minimum(t1, t2)
            t_hi = tf.maximum(t1, t2)

            # model value along z + t*d from tracked quantities (H symmetric):
            # m(t) = g.z + t g.d + z.Hz/2 + t d.Hz + t^2 dBd/2
            gz = dot(jac, z)
            gd = dot(jac, d)
            zHz = dot(z, Hz)
            dHz = dot(d, Hz)

            def m_along(t):
                return gz + t * gd + 0.5 * zHz + t * dHz + 0.5 * t * t * dBd

            m_lo = m_along(t_lo)
            m_hi = m_along(t_hi)

            # exit 1: negative curvature -> best of the two boundary points
            neg = dBd <= zero
            t_neg = tf.where(m_lo < m_hi, t_lo, t_hi)
            p_neg = z + t_neg * d
            m_neg = tf.minimum(m_lo, m_hi)

            safe_dBd = tf.where(neg, tf.ones_like(dBd), dBd)
            alpha = rr / safe_dBd
            z_next = z + alpha * d
            Hz_next = Hz + alpha * Bd

            # exit 2: the CG step leaves the region -> stop at the boundary
            # (the positive root, as z is inside)
            crossed = tf.norm(z_next) >= trust_radius
            p_cross = z + t_hi * d
            m_cross = m_hi

            r_next = r + alpha * Bd
            interior = tf.norm(r_next) < tolerance
            m_int = dot(jac, z_next) + 0.5 * dot(z_next, Hz_next)

            new_done = neg | crossed | interior
            new_p = tf.where(neg, p_neg, tf.where(crossed, p_cross, z_next))
            new_m = tf.where(neg, m_neg, tf.where(crossed, m_cross, m_int))
            new_hb = neg | crossed

            safe_rr = tf.where(rr > zero, rr, tf.ones_like(rr))
            beta = dot(r_next, r_next) / safe_rr
            d_next = -r_next + beta * d

            return (
                k + 1,
                z_next,
                r_next,
                d_next,
                Hz_next,
                new_p,
                new_m,
                new_hb,
                new_done,
            )

        k, z, r, d, Hz, p, mval, hb, done = tf.while_loop(
            cond,
            body,
            (
                tf.constant(0, tf.int32),
                zeros,
                jac,
                -jac,
                zeros,
                zeros,
                zero,
                tf.constant(False),
                tf.constant(False),
            ),
        )
        # maxiter exhaustion returns the last interior CG point (tracked in
        # new_p on the continue path): a valid descent step, hb stays False
        return p, mval, hb, k


class CGSteihaugSubproblem:
    """Quadratic subproblem solved by Steihaug-Toint truncated CG.

    Matches the ``IterativeSubproblem`` interface consumed by the shared
    trust-region outer loop, but is matrix-free: constructed from the value,
    gradient and the *point* (so the linearization can be re-pinned before
    HVPs run -- the outer loop evaluates the objective at proposed points
    in between solves, which moves the fitter's parameter state).
    """

    def __init__(self, fun_val, jac, x_point, solver, set_point=None, cg_maxiter=None):
        self.fun = float(fun_val)
        self.jac = tf.convert_to_tensor(jac)
        self.jac_mag = float(tf.norm(self.jac))
        self._x_point = x_point
        self._solver = solver
        self._set_point = set_point
        self._cg_maxiter = cg_maxiter
        self._last_p = None
        self._last_model = None
        self.niter = 0

    def solve(self, tr_radius):
        # scipy's forcing sequence: superlinear local convergence
        tolerance = min(0.5, math.sqrt(self.jac_mag)) * self.jac_mag

        if self.jac_mag < tolerance:  # only at an exactly-zero gradient
            p = np.zeros(int(self.jac.shape[0]), dtype=np.float64)
            self._last_p, self._last_model = p, self.fun
            return p, False

        if self._set_point is not None:
            self._set_point(self._x_point)

        maxiter = self._cg_maxiter or int(self.jac.shape[0])
        p, mval, hb, k = self._solver.solve(self.jac, tr_radius, tolerance, maxiter)

        self.niter = int(k)
        p_np = p.__array__()
        self._last_p = p_np
        self._last_model = self.fun + float(mval)
        logger.debug(
            f"steihaug-cg: {self.niter} HVPs, |p|={np.linalg.norm(p_np):.3e}, "
            f"hits_boundary={bool(hb)}"
        )
        return p_np, bool(hb)

    def model_value(self, p):
        if p is self._last_p:
            # the outer loop prices exactly the step solve() returned; its
            # model value fell out of the CG bookkeeping
            return self._last_model
        # generic fallback: one HVP
        if self._set_point is not None:
            self._set_point(self._x_point)
        pt = tf.convert_to_tensor(p, dtype=self.jac.dtype)
        Hp = self._solver.hessp_fn(pt)
        return (
            self.fun
            + float(tf.tensordot(self.jac, pt, 1))
            + 0.5 * float(tf.tensordot(pt, Hp, 1))
        )
