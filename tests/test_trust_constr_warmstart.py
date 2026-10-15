"""Warm-starting trust-constr from a previous minimum (rabbit.constraints.warm_start).

scipy's interior-point start ignores where it starts: a row not held feasible
gets slack max(1.5*(c-lb), 1) (a row on its face: 1, so x walks into the
interior), the barrier restarts at 0.1, and a row held feasible keeps a slack
pinned at c - lb ~ 0 that can freeze a fit whose minimum moved. These tests pin
(1) the scipy internals the warm start relies on, so a scipy upgrade that
changes them fails HERE and loudly, (2) that a restart from the minimum stays
there, (3) that a moved minimum and a face that must be released are still
found, (4) the cold path is untouched, and (5) the saturated test end to end.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import scipy
from scipy.optimize import NonlinearConstraint, minimize
from scipy.optimize._trustregion_constr import tr_interior_point as tip

from rabbit.constraints import warm_start

from .test_sparse_fit import make_test_tensor
from .test_trust_constr import (
    CubicBound,
    LinearBound,
    _fitter,
    _free_fit,
    _names,
    _phys,
    _postfit,
)

# ---------------------------------------------------------------- scipy internals


class _Stop(Exception):
    pass


def _first_iterate(monkeypatch, keep_feasible, ctx=None):
    """The z = [x, s] scipy's interior point starts its first SQP from."""
    seen = []

    def spy(fun_and_constr, grad_and_jac, lagr_hess, z0, *args, **kwargs):
        seen.append(np.array(z0, dtype=np.float64))
        raise _Stop

    monkeypatch.setattr(tip, "equality_constrained_sqp", spy)
    # rows c - lb at the start: on the face (0), 0.3 inside, 11.3 inside
    con = NonlinearConstraint(
        lambda x: np.array([x[0], x[0] + x[1], x[1]]),
        [0.0, 1.0, -10.0],
        np.inf,
        jac=lambda x: np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        keep_feasible=keep_feasible,
    )
    with pytest.raises(_Stop), ctx if ctx is not None else _null():
        minimize(
            lambda x: np.sum((x - 3.0) ** 2),
            [0.0, 1.3],
            jac=lambda x: 2 * (x - 3.0),
            hess=lambda x: 2 * np.eye(2),
            method="trust-constr",
            constraints=[con],
        )
    monkeypatch.undo()
    return seen[0][2:]


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_scipy_internals_still_support_the_slack_override():
    ok, why = warm_start.scipy_supports_slack_override()
    assert ok, (
        f"scipy {scipy.__version__}: {why}. rabbit's trust-constr warm start "
        "(rabbit/constraints/warm_start.py) must be re-checked against this "
        "scipy's tr_interior_point before it can be used."
    )


def test_scipy_slack_init_is_what_the_warm_start_works_around(monkeypatch):
    """The premise, pinned: rows not held start at max(1.5*(c-lb), 1); held
    rows at c - lb. If scipy changes this, the warm start needs a second look."""
    s = _first_iterate(monkeypatch, [False, False, False])
    np.testing.assert_allclose(s, [1.0, 1.0, 1.5 * 11.3], rtol=1e-12)
    s = _first_iterate(monkeypatch, [False, True, True])
    np.testing.assert_allclose(s, [1.0, 0.3, 11.3], rtol=1e-12)


def test_consistent_slacks_reach_the_first_iterate(monkeypatch):
    s = _first_iterate(
        monkeypatch, [False, False, False], warm_start.consistent_slacks(1e-12)
    )
    np.testing.assert_allclose(s, [1e-12, 0.3, 11.3], rtol=1e-12)
    # and scipy is left as it was
    assert tip.BarrierSubproblem.__name__ == "BarrierSubproblem"


def test_releasing_rows():
    J = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    lb = np.array([0.0, -np.inf, 5.0])
    ub = np.array([np.inf, 2.0, np.inf])
    labels = ["x>=0", "y<=2", "x+y>=5"]
    # x on its lower bound, y on its upper bound, third row far
    c = np.array([0.0, 2.0, 2.0])
    # binding: g = mu_x * e_x - mu_y * e_y with mu_x > 0 (lower), mu_y > 0 (upper)
    assert warm_start.releasing_rows([3.0, -2.0], c, J, lb, ub, labels) == []
    bad = warm_start.releasing_rows([-3.0, -2.0], c, J, lb, ub, labels)
    assert [b[0] for b in bad] == ["x>=0"]
    bad = warm_start.releasing_rows([3.0, 2.0], c, J, lb, ub, labels)
    assert [b[0] for b in bad] == ["y<=2"]
    # a row only frozen parameters move is not judged
    assert warm_start.releasing_rows([-3.0, -2.0], c, J, lb, ub, labels, [1]) == []


def test_centred_barrier():
    J = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    lb = np.array([0.0, -np.inf, 5.0])
    ub = np.array([np.inf, 2.0, np.inf])
    # x 1e-8 above its lower bound (mu 3), y 2e-9 below its upper bound (mu 2)
    c = np.array([1e-8, 2.0 - 2e-9, 2.0])
    mu0 = warm_start.centred_barrier([3.0, -2.0], c, J, lb, ub)
    assert mu0 == pytest.approx(3e-8, rel=1e-6)  # max(3 * 1e-8, 2 * 2e-9)
    # a releasing multiplier does not count; nothing binding -> the default
    assert warm_start.centred_barrier([-3.0, 2.0], c, J, lb, ub) == (
        warm_start.WARM_INITIAL_BARRIER
    )
    # far from every bound -> the default
    assert warm_start.centred_barrier([3.0, -2.0], [1.0, 0.0, 2.0], J, lb, ub) == (
        warm_start.WARM_INITIAL_BARRIER
    )


# --------------------------------------------------------- rabbit, test model


def _setup(tmp):
    """Two active rows (a CURVED one and an upper bound) and one inactive."""
    fn = make_test_tensor(tmp)
    free = _free_fit(fn)
    names, xf = _names(free), _phys(free)
    s, b = xf[names.index("sig")], xf[names.index("bkg_norm")]
    lb_cubic, ub_bkg = 1.1 * (s**3 + s * b) + 0.5, b - 0.2

    def cons():
        return [
            CubicBound(lb=lb_cubic),
            LinearBound({"bkg_norm": 1.0}, ub=ub_bkg),
            LinearBound({"sig": 1.0}, lb=s - 5.0),
        ]

    return fn, names, cons, ub_bkg


def _cold(fn, cons, **kw):
    f = _fitter(fn, "trust-constr", cons=cons(), **kw)
    f.minimize()
    return f


def _restart(fn, cons, x, mode="auto", nobs=None):
    f = _fitter(fn, "trust-constr", cons=cons(), minimizerWarmStart=mode)
    if nobs is not None:
        f.set_nobs(nobs)
    f.x.assign(x)
    cb = f.minimize(warm_start=True)
    return f, cb


def test_restart_from_the_minimum_stays_there():
    with tempfile.TemporaryDirectory() as tmp:
        fn, names, cons, _ = _setup(tmp)
        cold = _cold(fn, cons)
        loss0, x0 = float(cold.loss_val()), cold.x.numpy().copy()
        warm, cb = _restart(fn, cons, x0)
        st = warm.minimizer_status()
        assert st["warm_start_legs"] == ["warm"]
        assert st["nit"] <= 3
        assert max(cb.loss_history) - loss0 < 1e-6  # no excursion
        # the same constrained minimum, multipliers and face-held covariance
        np.testing.assert_allclose(_phys(warm), _phys(cold), atol=1e-6)
        pc, pw = _postfit(cold), _postfit(warm)
        assert [a["label"] for a in pw["active"]] == [a["label"] for a in pc["active"]]
        assert len(pw["active"]) == 2
        np.testing.assert_allclose(
            [a["multiplier"] for a in pw["active"]],
            [a["multiplier"] for a in pc["active"]],
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            pw["cov_held"], pc["cov_held"], rtol=1e-4, atol=1e-12
        )


def test_start_exactly_on_a_face_does_not_walk_into_the_interior():
    """A row exactly on its bound is not held feasible, and scipy starts its
    slack at 1: the old start walked the loss up by ~0.2 here (TCA: +333 NLL)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn, names, cons, ub_bkg = _setup(tmp)
        cold = _cold(fn, cons)
        loss0 = float(cold.loss_val())
        x = cold.x.numpy().copy()
        x[names.index("bkg_norm")] = ub_bkg  # exactly on the face
        warm, cb = _restart(fn, cons, x)
        assert max(cb.loss_history) - loss0 < 1e-6
        assert warm.minimizer_status()["nit"] <= 6
        # (cold reaches it to ~ its final barrier parameter; warm goes deeper)
        assert abs(float(warm.loss_val()) - loss0) < 2e-5
        old, cb_old = _restart(fn, cons, x, mode="off")
        assert max(cb_old.loss_history) - loss0 > 1e-2  # what this fixes


def test_moved_minimum_is_found():
    """The start is the minimum of other data (a toy, a scan neighbour)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn, names, cons, _ = _setup(tmp)
        cold = _cold(fn, cons)
        nobs = np.random.default_rng(1).poisson(cold.indata.data_obs.numpy())
        nobs = nobs.astype(np.float64)
        ref = _fitter(fn, "trust-constr", cons=cons())
        ref.set_nobs(nobs)
        ref.minimize()
        warm, _ = _restart(fn, cons, cold.x.numpy(), nobs=nobs)
        assert abs(float(warm.loss_val()) - float(ref.loss_val())) < 2e-5
        np.testing.assert_allclose(_phys(warm), _phys(ref), atol=1e-4)


def test_a_face_that_must_be_released_is_released():
    """Start ON sig >= lb with the free minimum inside: whether the warm leg or
    the fallback leaves the face, the answer is the free minimum."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = make_test_tensor(tmp)
        free = _free_fit(fn)
        names = _names(free)
        i_sig = names.index("sig")
        lb = _phys(free)[i_sig] - 0.3

        def cons():
            return [LinearBound({"sig": 1.0}, lb=lb)]

        x = free.x.numpy().copy()
        x[i_sig] = np.sqrt(lb)  # POI stored as sqrt(mu): sig = lb, on the face
        warm, _ = _restart(fn, cons, x)
        st = warm.minimizer_status()
        assert st["warm_start_legs"][0] == "warm"
        np.testing.assert_allclose(_phys(warm), _phys(free), atol=1e-4)
        assert abs(float(warm.loss_val()) - float(free.loss_val())) < 1e-6
        assert warm.releasing_constraint_rows() == []


def test_cold_path_is_unchanged():
    """warm_start=False under the default 'auto' is the old fit, bit for bit."""
    with tempfile.TemporaryDirectory() as tmp:
        fn, names, cons, _ = _setup(tmp)
        a = _cold(fn, cons)
        b = _cold(fn, cons, minimizerWarmStart="off")
        np.testing.assert_array_equal(a.x.numpy(), b.x.numpy())
        sa, sb = a.minimizer_status(), b.minimizer_status()
        assert sa["nit"] == sb["nit"] and "warm_start_legs" not in sa


def test_unsupported_scipy_falls_back_to_the_cold_start(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        fn, names, cons, _ = _setup(tmp)
        f = _fitter(fn, "trust-constr", cons=cons())
        monkeypatch.setattr(
            warm_start, "scipy_supports_slack_override", lambda: (False, "test")
        )
        assert not f._use_warm_start(True)


def test_warm_start_is_inert_without_constraints():
    with tempfile.TemporaryDirectory() as tmp:
        fn = make_test_tensor(tmp)
        f = _fitter(fn, "trust-krylov", minimizerWarmStart="on")
        assert not f._use_warm_start(True)


# ------------------------------------------------------------ saturated test


def _rabbit_fit(fn, out, *extra):
    rabbit_fit = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "bin",
        "rabbit_fit.py",
    )
    subprocess.run(
        [
            sys.executable,
            rabbit_fit,
            fn,
            "-o",
            str(out),
            "-t",
            "0",
            "--saveHists",
            "--computeSaturatedProjectionTests",
            "-m",
            "Project",
            "ch0",
            "--minimizerMethod",
            "trust-constr",
            "--minimizerGtol",
            "1e-10",
            "--minimizerXtol",
            "1e-10",
            "--minimizerBarrierTol",
            "1e-10",
            "--minimizerMaxiter",
            "500",
            "--earlyStopping",
            "-1",
            *extra,
        ],
        check=True,
    )
    from rabbit import io_tools

    res = io_tools.get_fitresult(str(out / "fitresults.hdf5"))
    (sat,) = res["mappings"].values()
    return res, sat


def test_saturated_subfit_is_warm_and_agrees_with_the_cold_one(tmp_path):
    fn = make_test_tensor(str(tmp_path))
    free = _free_fit(fn)
    ub = float(_phys(free)[_names(free).index("bkg_norm")] - 0.3)
    cons = ["--constraints", "Bounds", f"bkg_norm<={ub!r}"]
    _, warm = _rabbit_fit(fn, tmp_path / "warm", *cons)
    _, cold = _rabbit_fit(fn, tmp_path / "cold", *cons, "--minimizerWarmStart", "off")
    sw = warm["saturated_fit"]["minimizer_status"]
    sc = cold["saturated_fit"]["minimizer_status"]
    assert sw["warm_start_legs"][0] == "warm" and "warm_start_legs" not in sc
    # centred on the main fit's end: its barrier parameter, not scipy's 0.1
    assert sw["warm_start_barrier"] < 1e-5
    assert sw["nit"] < sc["nit"]
    assert sw["constraint_values"][0] <= ub + 1e-8
    # the same saturated minimum, to the barrier gap ~ multiplier * slack that
    # a gtol exit leaves (the cold sub-fit happens to go deeper here)
    nw = float(warm["saturated_fit"]["nllvalreduced"])
    nc = float(cold["saturated_fit"]["nllvalreduced"])
    assert abs(nw - nc) < 1e-6
