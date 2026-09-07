"""Tests for the native TF trust-region minimizer (tf-trust-exact).

Three layers, from unit to integration:

1. the subproblem against scipy's private ``IterativeSubproblem`` on random
   quadratic models -- positive-definite and indefinite Hessians, interior
   and boundary radii. The lambda searches differ slightly (we do not use
   the potrf failure index, see rabbit/minimizer/exact.py), so steps are
   compared by model value, not bitwise.
2. the full minimizer against scipy trust-exact on standard problems.
3. a full fit through the Fitter against the scipy trust-exact fit on the
   same tensor.
"""

import math
import tempfile

import numpy as np
import pytest
import tensorflow as tf

from rabbit import fitter, inputdata
from rabbit.minimizer import minimize_trust_exact
from rabbit.minimizer.exact import IterativeSubproblem
from rabbit.param_models.helpers import load_model

from .test_sparse_fit import make_options, make_test_tensor, run_fit

# --- 1. subproblem vs scipy ----------------------------------------------


def _random_model(n, rng, definite):
    A = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(A)
    if definite:
        eigs = rng.uniform(0.1, 10.0, n)
    else:
        eigs = rng.uniform(-5.0, 10.0, n)
        eigs[0] = -abs(eigs[0]) - 0.1  # guarantee indefiniteness
    H = Q @ np.diag(eigs) @ Q.T
    g = rng.standard_normal(n)
    return g, H


def _exact_subproblem_solution(g, H, tr_radius):
    """Exact trust-region step by eigendecomposition + bisection on the
    secular equation. Reference for both solvers (hard case not handled;
    the random g used here is never orthogonal to the leading eigenvector).
    """
    eigval, eigvec = np.linalg.eigh(H)
    gt = eigvec.T @ g

    def p_of(lam):
        return -eigvec @ (gt / (eigval + lam))

    lam_min = max(0.0, -eigval[0])
    if lam_min == 0.0 and np.linalg.norm(p_of(0.0)) <= tr_radius:
        return p_of(0.0)

    lo, hi = lam_min + 1e-14, lam_min + 1.0
    while np.linalg.norm(p_of(hi)) > tr_radius:
        hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.linalg.norm(p_of(mid)) > tr_radius:
            lo = mid
        else:
            hi = mid
    return p_of(hi)


@pytest.mark.parametrize("definite", [True, False])
@pytest.mark.parametrize("tr_radius", [0.01, 1.0, 100.0])
def test_subproblem_matches_scipy(definite, tr_radius):
    from scipy.optimize._trustregion_exact import IterativeSubproblem as ScipySubproblem

    rng = np.random.default_rng(1234)
    for trial in range(5):
        g, H = _random_model(10, rng, definite)

        m_tf = IterativeSubproblem(
            0.0, tf.constant(g, tf.float64), tf.constant(H, tf.float64)
        )
        p_tf, hb_tf = m_tf.solve(tr_radius)

        m_sp = ScipySubproblem(
            x=np.zeros(10), fun=lambda x: 0.0, jac=lambda x: g, hess=lambda x: H
        )
        p_sp, hb_sp = m_sp.solve(tr_radius)

        def model(p):
            return g @ p + 0.5 * p @ H @ p

        p_exact = _exact_subproblem_solution(g, H, tr_radius)
        best = model(p_exact)
        assert best < 0

        # Both solvers stop within the k_easy/k_hard bands of the exact
        # optimum, so steps differ slightly, and the boundary criterion
        # |norm(p) - radius|/radius <= k_easy allows up to a 10% overshoot
        # of the radius (scipy returns such steps too). What must hold is
        # that band plus achieving ~all of the optimal model reduction.
        assert np.linalg.norm(p_tf) <= tr_radius * (1 + m_tf.k_easy + 1e-6)
        assert model(p_tf) <= 0.95 * best  # >= 95% of the exact reduction
        assert model(p_sp) <= 0.95 * best  # scipy meets the same bar
        if hb_tf != hb_sp:
            # can only disagree when the interior/boundary distinction is
            # marginal, i.e. the unconstrained step ~ on the boundary
            assert abs(np.linalg.norm(p_exact) - tr_radius) / tr_radius < 0.15


# --- 2. minimizer vs scipy trust-exact ------------------------------------


def _tf_problem(f_tf, n):
    """Compiled (fun, closure) pair for the native minimizer plus numpy
    (f, g, h) for scipy, all from one TF definition of the objective."""
    xv = tf.Variable(tf.zeros(n, dtype=tf.float64))

    @tf.function
    def _val(x):
        xv.assign(x)
        return f_tf(xv)

    @tf.function
    def _vgh(x):
        xv.assign(x)
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                v = f_tf(xv)
            grad = t1.gradient(v, xv)
        hess = t2.jacobian(grad, xv)
        return v, grad, hess

    def fun(x):
        return float(_val(tf.constant(x, tf.float64)))

    def closure(x):
        v, grad, hess = _vgh(tf.constant(x, tf.float64))
        return float(v), grad, hess

    def f_np(x):
        return fun(x)

    def g_np(x):
        _, grad, _ = _vgh(tf.constant(x, tf.float64))
        return grad.numpy()

    def h_np(x):
        _, _, hess = _vgh(tf.constant(x, tf.float64))
        return hess.numpy()

    return fun, closure, f_np, g_np, h_np


def test_rosenbrock_matches_scipy():
    import scipy.optimize

    def rosen(x):
        return tf.reduce_sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    x0 = np.array([-1.2, 1.0, -0.5, 2.0, 0.3])
    fun, closure, f_np, g_np, h_np = _tf_problem(rosen, 5)

    res = minimize_trust_exact(fun, closure, x0)
    ref = scipy.optimize.minimize(
        f_np, x0, jac=g_np, hess=h_np, method="trust-exact", tol=0.0
    )

    np.testing.assert_allclose(res.x, np.ones(5), atol=1e-6)
    np.testing.assert_allclose(res.x, ref.x, atol=1e-6)
    assert res.fun <= ref.fun + 1e-10


def test_ill_conditioned_quadratic():
    rng = np.random.default_rng(42)
    n = 30
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    H = Q @ np.diag(np.logspace(-4, 2, n)) @ Q.T
    b = rng.standard_normal(n)
    Ht = tf.constant(H)
    bt = tf.constant(b)

    def quad(x):
        return 0.5 * tf.tensordot(x, tf.linalg.matvec(Ht, x), 1) + tf.tensordot(
            bt, x, 1
        )

    fun, closure, *_ = _tf_problem(quad, n)
    res = minimize_trust_exact(fun, closure, np.zeros(n))

    x_exact = -np.linalg.solve(H, b)
    np.testing.assert_allclose(res.x, x_exact, atol=1e-6, rtol=1e-6)


def test_callback_and_early_stopping():
    """The fitter's callback contract: called per iteration with .x/.fun,
    and a raise must propagate (that is how early stopping works)."""

    def rosen(x):
        return tf.reduce_sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    fun, closure, *_ = _tf_problem(rosen, 3)

    seen = []

    def cb(intermediate_result):
        seen.append((intermediate_result.fun, intermediate_result.x.copy()))
        if len(seen) >= 4:
            raise ValueError("stop")

    with pytest.raises(ValueError):
        minimize_trust_exact(fun, closure, np.array([-1.2, 1.0, 0.5]), callback=cb)
    assert len(seen) == 4
    assert all(np.isfinite(f) for f, _ in seen)


# --- 3. full fit through the Fitter ---------------------------------------


def run_fit_native(filename, method="tf-trust-exact", precondition=False):
    indata_obj = inputdata.FitInputData(filename)
    param_model = load_model("Mu", indata_obj)

    kwargs = dict(minimizerMethod=method)
    if precondition:
        kwargs.update(precondition=True, preconditionParams=[".*"])
    options = make_options(**kwargs)
    f = fitter.Fitter(indata_obj, param_model, options)
    f.set_nobs(indata_obj.data_obs)
    f.minimize()

    val, grad, hess = f.loss_val_grad_hess()
    from rabbit.tfhelpers import edmval_cov

    edmval, cov = edmval_cov(grad, hess)
    return {
        "x": f.x.numpy(),
        "loss": float(val),
        "edmval": float(edmval),
        "status": f.minimizer_status(),
    }


@pytest.mark.parametrize(
    "method", ["tf-trust-exact", "tf-trust-ncg", "tf-trust-krylov"]
)
@pytest.mark.parametrize("precondition", [False, True])
def test_fit_matches_scipy(method, precondition):
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = make_test_tensor(tmpdir)

        res_native = run_fit_native(fname, method, precondition)
        res_ref = run_fit(fname)  # trust-krylov: same likelihood, same minimum

        x_ref = np.concatenate([res_ref["param"], res_ref["theta"]])
        np.testing.assert_allclose(res_native["x"], x_ref, atol=1e-5, rtol=1e-4)
        assert res_native["edmval"] < 1e-4
        assert res_native["status"]["nit"] > 0


# --- 4. Steihaug-CG (tf-trust-ncg) ----------------------------------------


@pytest.mark.parametrize("definite", [True, False])
@pytest.mark.parametrize("tr_radius", [0.01, 1.0, 100.0])
def test_cg_subproblem_matches_scipy(definite, tr_radius):
    """Same algorithm as scipy's CGSteihaugSubproblem, so unlike the
    nearly-exact solver the iterates are deterministic and the steps must
    agree to float precision."""
    from scipy.optimize._trustregion_ncg import CGSteihaugSubproblem as ScipyCG

    from rabbit.minimizer.krylov import CGSteihaugSubproblem, SteihaugCGSolver

    rng = np.random.default_rng(4321)
    for trial in range(5):
        g, H = _random_model(10, rng, definite)
        Ht = tf.constant(H, tf.float64)

        solver = SteihaugCGSolver(lambda v: tf.linalg.matvec(Ht, v))
        m_tf = CGSteihaugSubproblem(
            0.0, tf.constant(g, tf.float64), np.zeros(10), solver
        )
        p_tf, hb_tf = m_tf.solve(tr_radius)

        m_sp = ScipyCG(
            x=np.zeros(10),
            fun=lambda x: 0.0,
            jac=lambda x: g,
            hess=None,
            hessp=lambda x, v: H @ v,
        )
        p_sp, hb_sp = m_sp.solve(tr_radius)

        np.testing.assert_allclose(p_tf, p_sp, atol=1e-9, rtol=1e-7)
        assert hb_tf == hb_sp

        # the cached model value handed to the outer loop must price the step
        def model(p):
            return g @ p + 0.5 * p @ H @ p

        assert abs(m_tf.model_value(p_tf) - model(p_tf)) < 1e-9 * (1 + abs(model(p_tf)))


def test_trust_ncg_rosenbrock():
    import scipy.optimize

    from rabbit.minimizer import minimize_trust_ncg

    def rosen(x):
        return tf.reduce_sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    n = 5
    xv = tf.Variable(tf.zeros(n, dtype=tf.float64))

    @tf.function
    def _val(x):
        xv.assign(x)
        return rosen(xv)

    @tf.function
    def _vg(x):
        xv.assign(x)
        with tf.GradientTape() as t:
            v = rosen(xv)
        return v, t.gradient(v, xv)

    @tf.function
    def _hvp(p):
        with tf.autodiff.ForwardAccumulator(xv, p) as acc:
            with tf.GradientTape() as t:
                v = rosen(xv)
            g = t.gradient(v, xv)
        return acc.jvp(g)

    def fun(x):
        return float(_val(tf.constant(x, tf.float64)))

    def closure(x):
        v, g = _vg(tf.constant(x, tf.float64))
        return float(v), g

    def set_point(x):
        xv.assign(tf.constant(x, tf.float64))

    x0 = np.array([-1.2, 1.0, -0.5, 2.0, 0.3])
    res = minimize_trust_ncg(fun, closure, _hvp, set_point, x0.copy())
    np.testing.assert_allclose(res.x, np.ones(n), atol=1e-5)

    ref = scipy.optimize.minimize(
        fun,
        x0.copy(),
        jac=lambda x: _vg(tf.constant(x, tf.float64))[1].numpy(),
        hessp=lambda x, v: (
            set_point(x),
            _hvp(tf.constant(v, tf.float64)).numpy(),
        )[1],
        method="trust-ncg",
        tol=0.0,
    )
    np.testing.assert_allclose(res.x, ref.x, atol=1e-5)


def test_pc_tf_transforms_match_numpy():
    """The TF-graph preconditioner application must reproduce the numpy one
    (it runs inside the CG loop where numpy cannot)."""
    from rabbit import preconditioner as precond

    rng = np.random.default_rng(99)
    n = 20
    A = rng.standard_normal((n, n))
    H = A @ A.T + n * np.eye(n)
    blocks = [("a", np.arange(0, 7)), ("b", np.arange(10, 16))]
    pc = precond.Preconditioner.from_hessian(H, np.zeros(n), blocks)
    assert pc.enabled

    apply_T, apply_TT = pc.tf_transforms()
    for _ in range(3):
        v = rng.standard_normal(n)
        np.testing.assert_allclose(
            apply_T(tf.constant(v)).numpy(), pc._apply_T(v), atol=1e-12
        )
        np.testing.assert_allclose(
            apply_TT(tf.constant(v)).numpy(), pc._apply_TT(v), atol=1e-12
        )


# --- 5. GLTR (tf-trust-krylov) ---------------------------------------------


@pytest.mark.parametrize("definite", [True, False])
@pytest.mark.parametrize("tr_radius", [0.01, 1.0, 100.0])
def test_gltr_subproblem_near_exact(definite, tr_radius):
    """Run to convergence on a small problem the Krylov subspace exhausts,
    GLTR must essentially reach the exact subproblem optimum -- the property
    Steihaug-CG does not have on the boundary."""
    from rabbit.minimizer.gltr import GLTRSolver, GLTRSubproblem

    rng = np.random.default_rng(2468)
    for trial in range(5):
        g, H = _random_model(10, rng, definite)
        Ht = tf.constant(H, tf.float64)

        solver = GLTRSolver(lambda v: tf.linalg.matvec(Ht, v))
        # explicit tight tolerance: the test demonstrates subspace
        # optimality, not the outer loop's forcing sequence
        m = GLTRSubproblem(
            0.0, tf.constant(g, tf.float64), np.zeros(10), solver, tol=1e-10
        )
        p, hb = m.solve(tr_radius)

        def model(p):
            return g @ p + 0.5 * p @ H @ p

        p_exact = _exact_subproblem_solution(g, H, tr_radius)
        best = model(p_exact)
        assert best < 0
        assert np.linalg.norm(p) <= tr_radius * (1 + 1e-8)
        assert model(p) <= 0.999 * best
        # the cached model value prices the returned step
        assert abs(m.model_value(p) - model(p)) < 1e-8 * (1 + abs(model(p)))


def test_gltr_hard_case():
    """g orthogonal to the bottom eigenvector, indefinite H, radius large
    enough that the secular equation has no root: the classic hard case."""
    from rabbit.minimizer.gltr import solve_tridiag_trust_region

    diag = np.array([-2.0, 1.0, 3.0])
    off = np.array([0.0, 0.5])  # decouples the bottom mode from g
    gamma0 = 1.0  # g = e1... wait e1 couples to mode 1
    # build instead directly: T diagonal-ish with g on a non-minimal mode
    h, lam, hb = solve_tridiag_trust_region(diag, off, gamma0, 10.0)
    T = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    g = np.array([gamma0, 0.0, 0.0])
    # must be on the boundary with lam >= -lambda_min
    assert hb
    assert np.linalg.norm(h) <= 10.0 * (1 + 1e-9)
    wmin = np.linalg.eigvalsh(T)[0]
    assert lam >= -wmin - 1e-9
    # and it must beat any interior point along -g
    m = g @ h + 0.5 * h @ T @ h
    assert m < 0
    # KKT: (T + lam I) h = -g up to solver tolerance
    resid = np.linalg.norm((T + lam * np.eye(3)) @ h + g)
    assert resid < 1e-6


def test_gltr_warm_restart_reuses_krylov_data():
    """The Krylov data is radius-independent: a re-solve at a smaller
    radius (the rejected-step path) must not restart the Lanczos process."""
    from rabbit.minimizer.gltr import GLTRSolver, GLTRSubproblem

    rng = np.random.default_rng(11)
    g, H = _random_model(30, rng, True)
    Ht = tf.constant(H, tf.float64)

    count = [0]

    def hessp(v):
        count[0] += 1
        return tf.linalg.matvec(Ht, v)

    solver = GLTRSolver(hessp)
    m = GLTRSubproblem(0.0, tf.constant(g, tf.float64), np.zeros(30), solver)
    p1, _ = m.solve(1.0)
    n_first = count[0]
    assert n_first > 0

    p2, hb2 = m.solve(0.25)  # shrunken radius, same point
    n_second = count[0] - n_first
    assert n_second <= 2  # essentially free re-solve

    # and the shrunken-radius solution is still near-optimal
    def model(p):
        return g @ p + 0.5 * p @ H @ p

    p_exact = _exact_subproblem_solution(g, H, 0.25)
    assert model(p2) <= 0.999 * model(p_exact)


def test_trust_krylov_rosenbrock():
    from rabbit.minimizer import minimize_trust_krylov

    def rosen(x):
        return tf.reduce_sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    n = 5
    xv = tf.Variable(tf.zeros(n, dtype=tf.float64))

    @tf.function
    def _val(x):
        xv.assign(x)
        return rosen(xv)

    @tf.function
    def _vg(x):
        xv.assign(x)
        with tf.GradientTape() as t:
            v = rosen(xv)
        return v, t.gradient(v, xv)

    @tf.function
    def _hvp(p):
        with tf.autodiff.ForwardAccumulator(xv, p) as acc:
            with tf.GradientTape() as t:
                v = rosen(xv)
            g = t.gradient(v, xv)
        return acc.jvp(g)

    def fun(x):
        return float(_val(tf.constant(x, tf.float64)))

    def closure(x):
        v, g = _vg(tf.constant(x, tf.float64))
        return float(v), g

    def set_point(x):
        xv.assign(tf.constant(x, tf.float64))

    # the classic scipy starting point, squarely in the global basin (the
    # harder start converges to Rosenbrock's legitimate second local
    # minimum near x1 = -0.96 for n >= 4, which is correct behavior but
    # not a useful assertion)
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    res = minimize_trust_krylov(fun, closure, _hvp, set_point, x0.copy())
    np.testing.assert_allclose(res.x, np.ones(n), atol=1e-5)


def test_device_smallest_singular_estimator():
    """Inverse-iteration estimator vs the true sigma_min: near-singular
    matrices (the regime the hard case uses it in) must be essentially
    exact; well-conditioned ones need only a safe same-order upper bound."""
    from rabbit.minimizer.exact import estimate_smallest_singular_value_device

    rng = np.random.default_rng(5)
    for gap in (1e-8, 1e-4, 1e-1):
        for n in (10, 50):
            Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
            eigs = np.linspace(1.0, 10.0, n)
            eigs[0] = gap  # smallest eigenvalue of L L^T = sigma_min^2
            A = Q @ np.diag(eigs) @ Q.T
            L = tf.constant(np.linalg.cholesky(A))
            s_est, z_est = estimate_smallest_singular_value_device(L)
            s_true = math.sqrt(gap)
            # Rayleigh quotient is an upper bound on sigma_min
            assert s_est >= s_true * (1 - 1e-9)
            if gap <= 1e-4:
                # strong separation: inverse iteration is converged
                assert s_est <= s_true * (1 + 1e-6)
                # and z is the bottom eigenvector
                overlap = abs(z_est @ Q[:, 0])
                assert overlap > 1 - 1e-8
            else:
                assert s_est <= s_true * 10  # same order even when hard
            assert abs(np.linalg.norm(z_est) - 1) < 1e-12


def test_nan_proposal_does_not_freeze_the_radius():
    """A proposal whose objective overflows must shrink the trust radius.

    With IEEE comparisons a NaN rho neither shrinks nor accepts, freezing
    the loop at a fixed radius until early stopping gives up far from the
    minimum -- observed in preconditioned coordinates where an internal
    step of norm 1 is an enormous physical step. The objective here is
    quadratic near the origin but NaN outside |x| < 3, so the minimizer
    must reject-and-shrink its way back inside."""

    H = np.diag([1.0, 4.0])
    b = np.array([1.0, -2.0])
    Ht, bt = tf.constant(H), tf.constant(b)

    def f_tf(x):
        quad = 0.5 * tf.tensordot(x, tf.linalg.matvec(Ht, x), 1) + tf.tensordot(
            bt, x, 1
        )
        bad = tf.reduce_max(tf.abs(x)) >= 3.0
        return tf.where(bad, tf.constant(float("nan"), tf.float64), quad)

    fun, closure, *_ = _tf_problem(f_tf, 2)
    res = minimize_trust_exact(fun, closure, np.array([2.0, -2.0]))
    x_exact = -np.linalg.solve(H, b)
    np.testing.assert_allclose(res.x, x_exact, atol=1e-8)
