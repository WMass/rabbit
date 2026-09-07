"""Nearly-exact trust-region subproblem in TensorFlow.

This is a port of scipy's ``IterativeSubproblem``
(``scipy/optimize/_trustregion_exact.py``, the More-Sorensen algorithm of
[3]_ as described in [1]_ ch. 7.3) with the linear algebra kept on the
TensorFlow device: building H + lambda*I, the Cholesky factorizations and
the triangular solves all run where the Hessian already lives, so the
n x n matrix never has to cross to the host per lambda trial. Only O(n)
vectors and control-flow scalars are synchronized.

The one structural difference from scipy: LAPACK's ``potrf`` reports the
index k of the first non-positive-definite leading minor on failure, which
scipy feeds to ``singular_leading_submatrix`` to tighten ``lambda_lb``
below the critical damping. ``tf.linalg.cholesky`` reports no such index
(it fills the factor with NaNs instead of raising, on CPU, GPU and under
XLA alike), and getting k out of TF would need a custom op. We don't need
it: a failed factorization of H + lambda*I proves lambda < -lambda_min(H),
so ``lambda_current`` itself is a valid lower bound, and the standard
safeguarded update max(sqrt(lb*ub), lb + theta*(ub - lb)) still converges,
just without the accelerated bound. The price is a few extra
factorizations per solve in the indefinite case -- each of which is the
thing this port makes cheap. With the fit preconditioner active the
internal-coordinate Hessian is near the identity and this branch is
essentially never taken.

The rarely-hit "hard case" refinement (interior solution with lambda > 0)
needs the smallest-singular-value estimate of the factor; that runs on the
host exactly as in scipy, paying one factor download when triggered.

References
----------
.. [1] A.R. Conn, N.I. Gould, and P.L. Toint, "Trust region methods",
       Siam, pp. 169-200, 2000.
.. [2] J. Nocedal and S. Wright, "Numerical optimization",
       Springer, pp. 83-91, 2006.
.. [3] J.J. More and D.C. Sorensen, "Computing a trust region step",
       SIAM J. Sci. Stat. Comput., vol. 4(3), pp. 553-572, 1983.
"""

import math

import numpy as np
import scipy.linalg
import tensorflow as tf
from wums import logging

logger = logging.child_logger(__name__)


def gershgorin_bounds(hess):
    """Lower and upper bounds on the eigenvalues of ``hess`` ([1]_ p. 19).

    Runs on device; returns python floats.
    """
    h_diag = tf.linalg.diag_part(hess)
    h_diag_abs = tf.abs(h_diag)
    row_sums = tf.reduce_sum(tf.abs(hess), axis=1)
    lb = tf.reduce_min(h_diag + h_diag_abs - row_sums)
    ub = tf.reduce_max(h_diag - h_diag_abs + row_sums)
    return float(lb), float(ub)


def estimate_smallest_singular_value(L):
    """Estimate the smallest singular value/vector of lower-triangular ``L``.

    Direct port of scipy's version (Cline et al. 1979), which works on the
    upper factor U; here U = L^T so U^T = L and the recurrence reads off
    columns of L. O(n^2), host-side numpy: it is inherently sequential and
    only runs in the rare hard case.
    """
    L = np.atleast_2d(L)
    n = L.shape[0]

    p = np.zeros(n)
    w = np.empty(n)

    for k in range(n):
        wp = (1 - p[k]) / L[k, k]
        wm = (-1 - p[k]) / L[k, k]
        pp = p[k + 1 :] + L[k + 1 :, k] * wp
        pm = p[k + 1 :] + L[k + 1 :, k] * wm

        if abs(wp) + np.linalg.norm(pp, 1) >= abs(wm) + np.linalg.norm(pm, 1):
            w[k] = wp
            p[k + 1 :] = pp
        else:
            w[k] = wm
            p[k + 1 :] = pm

    # w solves L w = e (e in {+1,-1}^n chosen for growth); now L^T v = w
    v = scipy.linalg.solve_triangular(L, w, lower=True, trans="T")
    v_norm = np.linalg.norm(v)

    s_min = np.linalg.norm(w) / v_norm
    z_min = v / v_norm
    return s_min, z_min


def estimate_smallest_singular_value_device(L, iters=8):
    """Device-side counterpart of :func:`estimate_smallest_singular_value`.

    Inverse iteration on L L^T: z <- (L L^T)^-1 z, normalized -- each pass
    two O(n^2) triangular solves on the device, so nothing but two scalars
    and one n-vector ever cross to the host, where the Cline et al.
    recurrence above is inherently sequential and needs the full factor
    downloaded (measured ~1 s/outer-iteration at n=4000 over PCIe).

    Convergence is governed by the eigenvalue separation of L L^T, which is
    extreme precisely in the near-singular regime the trust-region hard
    case lives in -- there a couple of iterations give machine-accurate
    estimates. Far from singularity the estimate is only an upper bound on
    sigma_min; every use in the solver is safe against that: the lambda_lb
    update only becomes looser, and the hard-case acceptance is guarded by
    an explicit model-value comparison at the call site.

    Returns (s_min, z_min) as (python float, numpy [n]) with ||z_min|| = 1.
    """
    n = tf.shape(L)[0]
    # deterministic start with broad spectral support (no randomness in
    # graphs; resume/reproducibility)
    z = tf.where(
        tf.range(n) % 2 == 0,
        tf.ones([n], dtype=L.dtype),
        -tf.ones([n], dtype=L.dtype),
    )
    z = z / tf.norm(z)
    for _ in range(iters):
        z = tf.squeeze(tf.linalg.cholesky_solve(L, z[:, None]), axis=-1)
        z = z / tf.norm(z)
    # Rayleigh quotient: sigma_min^2 ~ z.(L L^T)z = ||L^T z||^2
    s_min = float(tf.norm(tf.linalg.matvec(L, z, transpose_a=True)))
    return s_min, z.__array__()


def get_boundaries_intersections(z, d, trust_radius):
    """Solve ||z + t d|| == trust_radius for t; return [t_low, t_high]."""
    a = float(np.dot(d, d))
    b = 2 * float(np.dot(z, d))
    c = float(np.dot(z, z)) - trust_radius**2
    sqrt_discriminant = math.sqrt(b * b - 4 * a * c)

    # numerically stable form (avoids cancellation), as in scipy
    aux = b + math.copysign(sqrt_discriminant, b)
    ta = -aux / (2 * a)
    tb = -2 * c / aux
    return sorted([ta, tb])


class IterativeSubproblem:
    """Quadratic subproblem solved by the nearly-exact iterative method.

    Constructed from the objective value, gradient and dense Hessian at the
    current point (gradient and Hessian as tf tensors on device). One
    instance corresponds to one linearization point; ``solve`` may be
    called repeatedly with shrinking radii (rejected outer steps) and
    warm-starts its lambda search from the previous call, as in scipy.
    """

    # "theta" of [1]_ formula 7.3.14 (p. 190)
    UPDATE_COEFF = 0.01

    EPS = np.finfo(np.float64).eps

    # scipy caps the lambda search at 25 (gh-12513); ours approaches the
    # critical damping without the potrf-index acceleration, so allow more --
    # the factorizations are the operation this port makes cheap
    MAXITER_DEFAULT = 50

    def __init__(self, fun_val, jac, hess, k_easy=0.1, k_hard=0.2, maxiter=None):
        self.fun = float(fun_val)
        self.jac = tf.convert_to_tensor(jac)
        self.hess = tf.convert_to_tensor(hess)

        self.jac_mag = float(tf.norm(self.jac))

        # lambda-search warm start across solve() calls at this same point:
        # when the trust radius shrinks, the previous lower bound is reusable
        self.previous_tr_radius = -1.0
        self.lambda_lb = None

        # stop-criteria parameters, [1]_ pp. 194-197
        self.k_easy = k_easy
        self.k_hard = k_hard

        self.maxiter = self.MAXITER_DEFAULT if maxiter is None else maxiter

        self.dimension = int(self.hess.shape[0])
        self._eye = tf.eye(self.dimension, dtype=self.hess.dtype)
        self.hess_gersh_lb, self.hess_gersh_ub = gershgorin_bounds(self.hess)
        # NB axis=[-2, -1] requests the *matrix* norms; tf.norm's default
        # axis=None flattens the tensor, and the resulting max|H_ij| can sit
        # below |lambda_min|, silently invalidating the lambda_ub bracket
        self.hess_inf = float(tf.norm(self.hess, ord=np.inf, axis=[-2, -1]))
        self.hess_fro = float(tf.norm(self.hess, ord="fro", axis=[-2, -1]))
        self.CLOSE_TO_ZERO = self.dimension * self.EPS * self.hess_inf

    # --- model evaluation -------------------------------------------------

    def model_value(self, p):
        """m(p) = f + g.p + p.H p / 2 for a step ``p`` (numpy or tensor)."""
        p = tf.convert_to_tensor(p, dtype=self.hess.dtype)
        quad = tf.tensordot(p, tf.linalg.matvec(self.hess, p), axes=1)
        lin = tf.tensordot(self.jac, p, axes=1)
        return self.fun + float(lin) + 0.5 * float(quad)

    # --- linear algebra helpers ------------------------------------------

    def _factorize(self, lambda_current):
        """Cholesky of H + lambda*I. Returns (L, ok).

        tf.linalg.cholesky signals a non-positive-definite input by filling
        the factor with NaNs (never raising) on every backend, so success is
        a NaN check -- one scalar readback.
        """
        H = self.hess + lambda_current * self._eye
        L = tf.linalg.cholesky(H)
        ok = not bool(tf.reduce_any(tf.math.is_nan(L)))
        return L, ok

    @staticmethod
    def _cho_solve(L, b):
        """Solve (L L^T) x = b for vector b."""
        return tf.squeeze(
            tf.linalg.cholesky_solve(L, tf.expand_dims(b, axis=-1)), axis=-1
        )

    @staticmethod
    def _tri_solve_t(L, b):
        """Solve L^T x = b for vector b."""
        return tf.squeeze(
            tf.linalg.triangular_solve(
                L, tf.expand_dims(b, axis=-1), lower=True, adjoint=True
            ),
            axis=-1,
        )

    # --- lambda search ----------------------------------------------------

    def _initial_values(self, tr_radius):
        """Initial damping factor and bracket, [1]_ sec. 7.3.8 (p. 192)."""
        # upper bound
        hess_norm = min(self.hess_fro, self.hess_inf)
        lambda_ub = self.jac_mag / tr_radius + min(-self.hess_gersh_lb, hess_norm)
        lambda_ub = max(0.0, lambda_ub)

        # lower bound
        lambda_lb = self.jac_mag / tr_radius - min(self.hess_gersh_ub, hess_norm)
        lambda_lb = max(
            lambda_lb, -float(tf.reduce_min(tf.linalg.diag_part(self.hess)))
        )
        lambda_lb = max(0.0, lambda_lb)

        # improve the bracket with the previous solve at this point
        if tr_radius < self.previous_tr_radius and self.lambda_lb is not None:
            lambda_lb = max(self.lambda_lb, lambda_lb)

        if lambda_lb == 0.0:
            lambda_initial = 0.0
        else:
            lambda_initial = max(
                math.sqrt(lambda_lb * lambda_ub),
                lambda_lb + self.UPDATE_COEFF * (lambda_ub - lambda_lb),
            )
        return lambda_initial, lambda_lb, lambda_ub

    def solve(self, tr_radius):
        """Solve the subproblem for the given radius.

        Returns (p, hits_boundary) with ``p`` a numpy array.
        """
        lambda_current, lambda_lb, lambda_ub = self._initial_values(tr_radius)
        n = self.dimension
        hits_boundary = True
        already_factorized = False
        niter = 0

        p = None
        L = None
        factorized_ok = False

        while True:
            if already_factorized:
                already_factorized = False
            else:
                L, factorized_ok = self._factorize(lambda_current)

            if niter >= self.maxiter:
                # scipy caps here too (gh-12513). Return the best step
                # available rather than looping: the last computed p clipped
                # to the radius, or a boundary Cauchy step along -g.
                logger.warning(
                    f"trust-region subproblem lambda search hit maxiter="
                    f"{self.maxiter}; returning safeguarded step"
                )
                candidates = []
                if p is not None:
                    p_np = p.__array__()
                    p_norm = np.linalg.norm(p_np)
                    if p_norm > tr_radius:
                        p_np = p_np * (tr_radius / p_norm)
                    candidates.append(p_np)
                # Cauchy step: exact minimizer of the model along -g within
                # the radius; strict descent for any Hessian, so the outer
                # loop stays globally convergent even from this fallback
                g = self.jac.__array__()
                g_norm = np.linalg.norm(g)
                if g_norm > 0:
                    gHg = float(
                        tf.tensordot(
                            self.jac, tf.linalg.matvec(self.hess, self.jac), axes=1
                        )
                    )
                    if gHg <= 0:
                        t = tr_radius / g_norm
                    else:
                        t = min(tr_radius / g_norm, g_norm**2 / gHg)
                    candidates.append(-t * g)
                p = min(candidates, key=self.model_value)
                break
            niter += 1

            if factorized_ok and self.jac_mag > self.CLOSE_TO_ZERO:
                # successful factorization, general case
                p = self._cho_solve(L, -self.jac)
                p_norm = float(tf.norm(p))

                # interior convergence
                if p_norm <= tr_radius and lambda_current == 0.0:
                    hits_boundary = False
                    break

                # Newton step on the secular equation, [2]_ (4.44) p. 87
                w = self._tri_solve_t(L, p)
                w_norm = float(tf.norm(w))
                delta_lambda = (p_norm / w_norm) ** 2 * (p_norm - tr_radius) / tr_radius
                # The Newton correction is negative in the interior case and
                # can push lambda below zero (the true solution then being an
                # interior step); the search must stay on lambda >= 0 or the
                # sqrt(lb*ub) safeguards later see a negative bracket.
                lambda_new = max(lambda_current + delta_lambda, 0.0)

                if p_norm < tr_radius:
                    # inside the boundary with lambda > 0: hard-case territory
                    s_min, z_min = estimate_smallest_singular_value_device(L)

                    p_np = p.__array__()
                    ta, tb = get_boundaries_intersections(p_np, z_min, tr_radius)

                    # smallest-magnitude root, [3]_ p. 6
                    step_len = ta if abs(ta) < abs(tb) else tb

                    quadratic_term = float(
                        tf.tensordot(p, tf.linalg.matvec(self.hess, p), axes=1)
                    )

                    relative_error = (step_len**2 * s_min**2) / (
                        quadratic_term + lambda_current * tr_radius**2
                    )
                    if relative_error <= self.k_hard:
                        # Guard: only accept the corrected step if it actually
                        # lowers the model. The stop criterion trusts s_min,
                        # and any estimator (the LINPACK recurrence included)
                        # can be off far from singularity -- accepting a junk
                        # correction hands the outer loop a poor step it then
                        # rejects, which showed up as a doubled outer
                        # iteration count on GPU rounding.
                        p_hat = p_np + step_len * z_min
                        if self.model_value(p_hat) <= self.model_value(p_np):
                            p = p_hat
                            break

                    lambda_ub = lambda_current
                    lambda_lb = max(lambda_lb, lambda_current - s_min**2)

                    # refactorize at the Newton iterate (scipy rebuilds H
                    # with lambda_new here -- factorizing the stale matrix
                    # instead makes the interior case converge erratically)
                    L, factorized_ok = self._factorize(lambda_new)
                    if factorized_ok:
                        lambda_current = lambda_new
                        already_factorized = True
                    else:
                        lambda_lb = max(lambda_lb, lambda_new)
                        lambda_current = max(
                            math.sqrt(lambda_lb * lambda_ub),
                            lambda_lb + self.UPDATE_COEFF * (lambda_ub - lambda_lb),
                        )
                else:
                    # outside the boundary
                    relative_error = abs(p_norm - tr_radius) / tr_radius
                    if relative_error <= self.k_easy:
                        break

                    lambda_lb = lambda_current
                    lambda_current = lambda_new

            elif factorized_ok:
                # successful factorization but jac_mag ~ 0
                if lambda_current == 0.0:
                    p = tf.zeros([n], dtype=self.jac.dtype)
                    hits_boundary = False
                    break

                s_min, z_min = estimate_smallest_singular_value_device(L)
                step_len = tr_radius

                if (
                    step_len**2 * s_min**2
                    <= self.k_hard * lambda_current * tr_radius**2
                ):
                    # same guard as above: the step must descend the model
                    p_hat = step_len * z_min
                    if self.model_value(p_hat) <= self.fun:
                        p = p_hat
                        break

                lambda_ub = lambda_current
                lambda_lb = max(lambda_lb, lambda_current - s_min**2)
                lambda_current = max(
                    math.sqrt(lambda_lb * lambda_ub),
                    lambda_lb + self.UPDATE_COEFF * (lambda_ub - lambda_lb),
                )

            else:
                # Unsuccessful factorization: lambda_current is proven to lie
                # below the critical damping, so it is itself a valid lower
                # bound. scipy tightens the bound further using the potrf
                # failure index; without it the safeguarded update below
                # still converges linearly on the bracket.
                lambda_lb = max(lambda_lb, lambda_current)
                if lambda_ub - lambda_lb <= 1e-10 * max(1.0, lambda_ub):
                    # the bracket has collapsed onto a lambda that still
                    # fails, i.e. lambda_ub was not actually an upper bound;
                    # defensive (the matrix norms make it valid), rescue by
                    # doubling rather than looping to maxiter
                    lambda_ub = 2.0 * lambda_ub + 1.0
                lambda_current = max(
                    math.sqrt(lambda_lb * lambda_ub),
                    lambda_lb + self.UPDATE_COEFF * (lambda_ub - lambda_lb),
                )

        self.lambda_lb = lambda_lb
        self.lambda_current = lambda_current
        self.previous_tr_radius = tr_radius

        if isinstance(p, tf.Tensor):
            p = p.__array__()
        return np.asarray(p, dtype=np.float64), hits_boundary
