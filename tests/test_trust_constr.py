"""--minimizerMethod trust-constr with hard constraints (rabbit.constraints).

Covers the ways this path has failed SILENTLY before:
a binding constraint not being honoured, a "frozen" parameter moved through
the constraint Jacobian (2026-08), a constraint depending only on frozen
parameters, a start exactly on a constraint's bound that never moves, and the
missing constraint Hessian that over-damped every step (2026-10). It also
checks that -r regularizers stay in the loss next to the constraints.

Constraints act on the PHYSICAL vector (Fitter.get_x()); the test model's POI
is stored as sqrt(mu), so everything below compares physical values.
"""

import tempfile

import numpy as np
import pytest
import tensorflow as tf

from rabbit import fitter, inputdata
from rabbit.constraints.constraints import Bounds, Constraints
from rabbit.param_models.helpers import load_model
from rabbit.regularization.regularizer import Regularizer

from .test_sparse_fit import make_options, make_test_tensor


class LinearBound(Constraints):
    """lb <= sum_j a_j x[name_j] <= ub."""

    def __init__(self, coeffs, lb=-np.inf, ub=np.inf):
        self.dtype = tf.float64
        self.coeffs = dict(coeffs)
        self.labels = ["linear"]
        self.lower = np.array([lb], dtype=np.float64)
        self.upper = np.array([ub], dtype=np.float64)
        self._idx = None

    def set_layout(self, parms):
        self._idx = self.resolve_indices(parms, list(self.coeffs), "LinearBound")

    def parameter_indices(self):
        return list(self._idx.values())

    def values(self, x):
        return tf.reshape(
            tf.add_n(
                [
                    tf.constant(a, dtype=self.dtype) * x[self._idx[n]]
                    for n, a in self.coeffs.items()
                ]
            ),
            [1],
        )


class CubicBound(Constraints):
    """sig^3 + sig * bkg_norm >= lb: nonlinear, with a non-trivial Hessian."""

    def __init__(self, lb):
        self.dtype = tf.float64
        self.labels = ["cubic"]
        self.lower = np.array([lb], dtype=np.float64)
        self.upper = np.array([np.inf])
        self._idx = None

    def set_layout(self, parms):
        self._idx = self.resolve_indices(parms, ["sig", "bkg_norm"], "CubicBound")

    # parameter_indices deliberately NOT declared: the post-fit must then fall
    # back to every floating parameter, and still be right

    def values(self, x):
        s, b = x[self._idx["sig"]], x[self._idx["bkg_norm"]]
        return tf.reshape(s**3 + s * b, [1])


class SigPenalty(Regularizer):
    """relu(lb - sig)^2: a -r term, which must stay in the loss."""

    needs_observables = False

    def __init__(self, lb):
        self.lb = float(lb)
        self._i = None

    def set_expectations(self, initial_params, initial_observables, parms=None):
        self._i = self.resolve_indices(parms, ["sig"], "SigPenalty")["sig"]

    def compute_nll_penalty(self, params, observables):
        return tf.square(tf.nn.relu(tf.constant(self.lb, tf.float64) - params[self._i]))


def _fitter(filename, method, cons=(), regs=(), tau=0.0, **kw):
    indata_obj = inputdata.FitInputData(filename)
    pm = load_model("Mu", indata_obj)
    opts = make_options(minimizerMethod=method, **kw)
    if method == "trust-constr":
        opts.minimizerGtol = 1e-10
        opts.minimizerXtol = 1e-10
        opts.minimizerBarrierTol = 1e-10
        opts.minimizerMaxiter = 500
    f = fitter.Fitter(indata_obj, pm, opts)
    f.set_nobs(indata_obj.data_obs)
    f.tau.assign(tau)
    f.regularizers = list(regs)
    f.constraints = list(cons)
    f.arm_regularizers()
    return f


def _names(f):
    return [p.decode() if isinstance(p, bytes) else str(p) for p in f.parms]


def _phys(f):
    return f.get_x().numpy()


def _free_fit(filename):
    f = _fitter(filename, "trust-krylov")
    f.minimize()
    return f


def test_regularizer_stays_in_the_loss_next_to_constraints():
    with tempfile.TemporaryDirectory() as tmp:
        fn = make_test_tensor(tmp)
        pen = SigPenalty(lb=50.0)  # violated at the default point
        ref = _fitter(fn, "trust-krylov", regs=[pen], tau=1.0)
        tc = _fitter(
            fn,
            "trust-constr",
            regs=[SigPenalty(lb=50.0)],
            cons=[LinearBound({"bkg_norm": 1.0}, lb=-10.0)],
            tau=1.0,
        )
        free = _fitter(fn, "trust-krylov")
        np.testing.assert_allclose(
            float(tc.loss_val()), float(ref.loss_val()), rtol=0, atol=1e-12
        )
        assert float(tc.loss_val()) > float(free.loss_val()) + 1.0


def test_constraints_need_a_constrained_minimizer():
    with tempfile.TemporaryDirectory() as tmp:
        fn = make_test_tensor(tmp)
        f = _fitter(fn, "trust-krylov", cons=[LinearBound({"sig": 1.0}, lb=0.0)])
        with pytest.raises(ValueError, match="constrained minimizer"):
            f.minimize()


def test_inactive_constraint_reproduces_the_free_fit():
    with tempfile.TemporaryDirectory() as tmp:
        fn = make_test_tensor(tmp)
        free = _free_fit(fn)
        sig = _phys(free)[_names(free).index("sig")]
        tc = _fitter(fn, "trust-constr", cons=[LinearBound({"sig": 1.0}, sig - 0.2)])
        tc.minimize()
        np.testing.assert_allclose(_phys(tc), _phys(free), atol=2e-5)
        st = tc.minimizer_status()
        assert st["constraint_values"][0] > st["constraint_lower_bounds"][0] + 0.1
        assert st["constraint_labels"] == ["linear"]


def test_binding_constraint_matches_the_profile_on_the_face():
    """An active constraint sig >= lb must land on the conditional minimum with
    sig fixed at lb (the exact constrained optimum when the face is active)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = make_test_tensor(tmp)
        free = _free_fit(fn)
        i_sig = _names(free).index("sig")
        lb = _phys(free)[i_sig] + 0.1
        tc = _fitter(fn, "trust-constr", cons=[LinearBound({"sig": 1.0}, lb)])
        tc.minimize()
        face = _fitter(fn, "trust-krylov", freezeParameters=["sig"])
        x0 = face.x.numpy().copy()
        x0[i_sig] = np.sqrt(lb)  # POI stored as sqrt(mu)
        face.x.assign(x0)
        face.minimize()
        assert abs(_phys(tc)[i_sig] - lb) < 1e-6
        np.testing.assert_allclose(_phys(tc), _phys(face), atol=1e-4)
        st = tc.minimizer_status()
        # scipy's gtol exit does not wait for the barrier parameter mu, and the
        # barrier optimum sits ~mu above the constrained minimum in the loss
        gap = float(tc.loss_val()) - float(face.loss_val())
        assert -1e-9 < gap < 2 * st["barrier_parameter"] + 1e-7
        assert st["constr_violation"] < 1e-8
        assert abs(st["multipliers"][0]) > 1e-3  # strongly active
        assert float(tc.loss_val()) > float(free.loss_val())


def test_bounds_on_a_nuisance_upper_bound():
    """The built-in Bounds, on a card nuisance, binding from ABOVE."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = make_test_tensor(tmp)
        free = _free_fit(fn)
        i_nrm = _names(free).index("bkg_norm")
        ub = _phys(free)[i_nrm] - 0.3
        indata_obj = inputdata.FitInputData(fn)
        tc = _fitter(
            fn, "trust-constr", cons=[Bounds(indata_obj, f"bkg_norm<={float(ub)!r}")]
        )
        tc.minimize()
        st = tc.minimizer_status()
        # the barrier optimum sits ~mu/|multiplier| inside the face
        assert 0 <= ub - _phys(tc)[i_nrm] < 1e-5
        assert st["constraint_upper_bounds"] == [ub]
        assert st["constr_violation"] < 1e-8


def test_bounds_parsing():
    with tempfile.TemporaryDirectory() as tmp:
        fn = make_test_tensor(tmp)
        indata_obj = inputdata.FitInputData(fn)
        b = Bounds(indata_obj, "sig>=0", "-1<=bkg_norm<=2.5e0")
        np.testing.assert_array_equal(b.lower, [0.0, -1.0])
        np.testing.assert_array_equal(b.upper, [np.inf, 2.5])
        for bad in ("sig>0", "2<=sig<=1", "sig"):
            with pytest.raises(ValueError):
                Bounds(indata_obj, bad)
        with pytest.raises(ValueError, match="twice"):
            Bounds(indata_obj, "sig>=0", "sig<=3")


def test_frozen_parameter_is_not_moved_through_the_constraint():
    with tempfile.TemporaryDirectory() as tmp:
        fn = make_test_tensor(tmp)
        free = _free_fit(fn)
        names = _names(free)
        i_sig, i_nrm = names.index("sig"), names.index("bkg_norm")
        xfree, pfree = free.x.numpy(), _phys(free)
        # bind on sig + bkg_norm with bkg_norm frozen at its free value: the
        # only legal way to satisfy it is to move sig
        lb = pfree[i_sig] + pfree[i_nrm] + 0.1
        tc = _fitter(
            fn,
            "trust-constr",
            cons=[LinearBound({"sig": 1.0, "bkg_norm": 1.0}, lb)],
            freezeParameters=["bkg_norm"],
        )
        tc.x.assign(xfree)
        _, jac = tc.eval_physical(tc._constraints_fn, xfree, jacobian=True)
        assert jac[0, i_nrm] == 0.0 and jac[0, i_sig] != 0.0
        tc.minimize()
        p = _phys(tc)
        assert tc.x.numpy()[i_nrm] == xfree[i_nrm]
        assert abs(p[i_sig] + p[i_nrm] - lb) < 1e-6


def test_constraint_on_frozen_parameters_only():
    with tempfile.TemporaryDirectory() as tmp:
        fn = make_test_tensor(tmp)
        # satisfied: dropped, and the fit is the free fit
        tc = _fitter(
            fn,
            "trust-constr",
            cons=[LinearBound({"bkg_norm": 1.0}, -10.0)],
            freezeParameters=["bkg_norm"],
        )
        assert tc.build_scipy_constraints(tc.x.numpy()) == []
        assert not tc._constraint_info["live"].any()
        # violated: refused
        tc = _fitter(
            fn,
            "trust-constr",
            cons=[LinearBound({"bkg_norm": 1.0}, 10.0)],
            freezeParameters=["bkg_norm"],
        )
        with pytest.raises(ValueError, match="VIOLATED"):
            tc.build_scipy_constraints(tc.x.numpy())


def test_start_exactly_on_a_bound_still_moves():
    """A row active at x0 must not be held feasible: with keep_feasible=True
    scipy rejects every step off the bound (or declares convergence on it)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = make_test_tensor(tmp)
        free = _free_fit(fn)
        i_nrm = _names(free).index("bkg_norm")
        nfree = _phys(free)[i_nrm]
        # start the nuisance 0.5 below its free value, with the bound exactly
        # there and the free minimum on the allowed side
        start = nfree - 0.5
        tc = _fitter(fn, "trust-constr", cons=[LinearBound({"bkg_norm": 1.0}, start)])
        x = tc.x.numpy().copy()
        x[i_nrm] = start  # a nuisance is stored linearly
        tc.x.assign(x)
        assert tc.build_scipy_constraints(x)[0].keep_feasible.tolist() == [False]
        tc.minimize()
        np.testing.assert_allclose(_phys(tc), _phys(free), atol=2e-5)


def test_constraint_hessian_is_exact():
    """The HVP handed to scipy equals finite differences of the Jacobian."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = make_test_tensor(tmp)
        tc = _fitter(fn, "trust-constr", cons=[CubicBound(lb=-100.0)])
        tc.build_scipy_constraints(tc.x.numpy())
        rng = np.random.default_rng(1)
        x = tc.x.numpy() + 0.1 * rng.standard_normal(tc.x.shape[0])
        v = np.array([0.7])
        d = rng.standard_normal(tc.x.shape[0])
        hv = tc.hvp_physical(tc._constraints_fn, x, v).matvec(d)
        eps = 1e-5
        jp = tc.eval_physical(tc._constraints_fn, x + eps * d, jacobian=True)[1]
        jm = tc.eval_physical(tc._constraints_fn, x - eps * d, jacobian=True)[1]
        fd = (v @ (jp - jm)) / (2 * eps)
        np.testing.assert_allclose(hv, fd, rtol=1e-6, atol=1e-8)
        assert np.count_nonzero(np.abs(hv) > 1e-12) <= 2  # sig and bkg_norm only


def test_constraint_hessian_reaches_scipy(monkeypatch):
    """The exact Hessian is passed (not scipy's default BFGS), and a nonlinear
    binding constraint is honoured."""
    seen = {}
    real = fitter.scipy.optimize.NonlinearConstraint

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(fitter.scipy.optimize, "NonlinearConstraint", spy)
    with tempfile.TemporaryDirectory() as tmp:
        fn = make_test_tensor(tmp)
        free = _free_fit(fn)
        names = _names(free)
        p = _phys(free)
        s, b = p[names.index("sig")], p[names.index("bkg_norm")]
        lb = s**3 + s * b + 0.5
        tc = _fitter(fn, "trust-constr", cons=[CubicBound(lb)])
        tc.minimize()
        assert callable(seen.get("hess"))
        st = tc.minimizer_status()
        assert abs(st["constraint_values"][0] - lb) < 1e-6
        assert st["constr_violation"] < 1e-8


def test_interior_point_start_options_reach_scipy(monkeypatch):
    """--minimizerInitialBarrier/-BarrierTol/-TrRadius are passed to scipy, and
    a small initial barrier still lands on the face of a binding constraint."""
    seen = {}
    real = fitter.scipy.optimize.minimize

    def spy(*args, **kwargs):
        if kwargs.get("method") == "trust-constr":
            seen.update(kwargs.get("options", {}))
        return real(*args, **kwargs)

    monkeypatch.setattr(fitter.scipy.optimize, "minimize", spy)
    with tempfile.TemporaryDirectory() as tmp:
        fn = make_test_tensor(tmp)
        free = _free_fit(fn)
        i_sig = _names(free).index("sig")
        lb = _phys(free)[i_sig] + 0.1
        tc = _fitter(
            fn,
            "trust-constr",
            cons=[LinearBound({"sig": 1.0}, lb)],
            minimizerInitialBarrier=1e-3,
            minimizerInitialBarrierTol=1e-3,
            minimizerInitialTrRadius=0.5,
        )
        tc.minimize()
        assert seen["initial_barrier_parameter"] == 1e-3
        assert seen["initial_barrier_tolerance"] == 1e-3
        assert seen["initial_tr_radius"] == 0.5
        assert abs(_phys(tc)[i_sig] - lb) < 1e-6
        assert tc.minimizer_status()["constr_violation"] < 1e-8
    # unset: scipy's defaults (the keys are not passed at all)
    seen.clear()
    with tempfile.TemporaryDirectory() as tmp:
        fn = make_test_tensor(tmp)
        tc = _fitter(fn, "trust-constr", cons=[LinearBound({"sig": 1.0}, -1e3)])
        tc.minimize()
        assert not {
            "initial_barrier_parameter",
            "initial_barrier_tolerance",
            "initial_tr_radius",
        } & set(seen)


# ---------------------------------------------------------------------------
# post-fit at an active face (Fitter.constrained_postfit, rabbit_fit's cov)
# ---------------------------------------------------------------------------


def _postfit(f):
    _, grad, hess = f.loss_val_grad_hess()
    return f.constrained_postfit(grad, hess)


def test_postfit_held_cov_matches_the_frozen_face_fit():
    """On a binding bound sig >= lb, the face-held covariance is the
    covariance of the fit with sig frozen on the face."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = make_test_tensor(tmp)
        free = _free_fit(fn)
        names = _names(free)
        i_sig = names.index("sig")
        lb = _phys(free)[i_sig] + 0.1
        tc = _fitter(fn, "trust-constr", cons=[LinearBound({"sig": 1.0}, lb)])
        tc.minimize()
        cpf = _postfit(tc)

        face = _fitter(fn, "trust-krylov", freezeParameters=["sig"])
        x0 = face.x.numpy().copy()
        x0[i_sig] = np.sqrt(lb)
        face.x.assign(x0)
        face.minimize()
        _, gf, hf = face.loss_val_grad_hess()
        _, cov_face = face.edmval_cov(gf, hf)
        cov_face = np.asarray(cov_face)

        other = [i for i in range(len(names)) if i != i_sig]
        held = cpf["cov_held"]
        np.testing.assert_allclose(
            held[np.ix_(other, other)], cov_face[np.ix_(other, other)], rtol=2e-3
        )
        assert abs(held[i_sig, i_sig]) < 1e-8 * cpf["cov_released"][i_sig, i_sig]
        # releasing the face gives the free covariance, which is wider on sig
        assert cpf["cov_released"][i_sig, i_sig] > 0
        (a,) = cpf["active"]
        assert a["label"] == "linear" and a["multiplier"] > 0
        assert a["release_dnll"] > 0
        assert cpf["edm_tangent"] < 1e-8 < cpf["edm_released"]


def test_postfit_without_an_active_face_is_the_usual_one():
    with tempfile.TemporaryDirectory() as tmp:
        fn = make_test_tensor(tmp)
        free = _free_fit(fn)
        sig = _phys(free)[_names(free).index("sig")]
        tc = _fitter(fn, "trust-constr", cons=[LinearBound({"sig": 1.0}, sig - 0.2)])
        tc.minimize()
        cpf = _postfit(tc)
        assert cpf["active"] == []
        np.testing.assert_array_equal(cpf["cov_held"], cpf["cov_released"])
        assert cpf["edm_tangent"] == cpf["edm_released"]


def test_postfit_uses_the_hessian_of_the_lagrangian():
    """On a NONLINEAR face, the curvature along the face is t^T W t with
    W = H - mu d2c, not t^T H t. Checked against finite differences of the loss
    along the face curve itself."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = make_test_tensor(tmp)
        free = _free_fit(fn)
        names = _names(free)
        i_sig, i_nrm = names.index("sig"), names.index("bkg_norm")
        p = _phys(free)
        lb = p[i_sig] ** 3 + p[i_sig] * p[i_nrm] + 0.5
        tc = _fitter(fn, "trust-constr", cons=[CubicBound(lb)])
        tc.minimize()
        cpf = _postfit(tc)
        (a,) = cpf["active"]
        W = cpf["hess_lagrangian"]
        _, _, H = tc.loss_val_grad_hess()
        H = np.asarray(H)

        xhat = tc.x.numpy().copy()
        b0 = xhat[i_nrm]

        def on_face(b):
            # physical s = x_sig^2 solves s^3 + s*b = c(xhat); keep the rest fixed
            target = (xhat[i_sig] ** 2) ** 3 + xhat[i_sig] ** 2 * b0
            s = xhat[i_sig] ** 2
            for _ in range(60):
                s -= (s**3 + s * b - target) / (3 * s**2 + b)
            x = xhat.copy()
            x[i_sig], x[i_nrm] = np.sqrt(s), b
            return x

        def loss_at(x):
            tc.x.assign(x)
            return float(tc.loss_val())

        h = 1e-3
        xm, x0, xp = on_face(b0 - h), on_face(b0), on_face(b0 + h)
        curv_fd = (loss_at(xp) - 2 * loss_at(x0) + loss_at(xm)) / h**2
        t = (xp - xm) / (2 * h)
        tc.x.assign(xhat)
        np.testing.assert_allclose(t @ W @ t, curv_fd, rtol=2e-3)
        # and the constraint-curvature term matters here
        assert abs(t @ H @ t - curv_fd) > 10 * abs(t @ W @ t - curv_fd)
        # held covariance is orthogonal to the face normal
        _, J = tc.eval_physical(tc._constraints_fn, xhat, jacobian=True)
        np.testing.assert_allclose(cpf["cov_held"] @ J[0], 0.0, atol=1e-9)


def test_rabbit_fit_stores_the_constrained_postfit(tmp_path):
    """End to end through bin/rabbit_fit.py: the face-held 'cov', the released
    one, the EDM on the face and the active-face summary are written."""
    import os
    import subprocess
    import sys

    from rabbit import io_tools

    fn = make_test_tensor(str(tmp_path))
    free = _free_fit(fn)
    ub = float(_phys(free)[_names(free).index("bkg_norm")] - 0.3)
    rabbit_fit = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "bin",
        "rabbit_fit.py",
    )
    out = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            rabbit_fit,
            fn,
            "-o",
            str(out),
            "-t",
            "0",
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
            "--constraints",
            "Bounds",
            f"bkg_norm<={ub!r}",
        ],
        check=True,
    )
    res = io_tools.get_fitresult(str(out / "fitresults.hdf5"))
    held, rel = res["cov"].get().values(), res["cov_released"].get().values()
    summary = res["constraints_postfit"]
    (face,) = summary["active"]
    assert face["label"].startswith("bkg_norm<=")
    assert face["multiplier"] < 0  # an upper bound binds with mu < 0
    i = list(np.array(res["cov"].get().axes["parms_x"])).index("bkg_norm")
    assert abs(held[i, i]) < 1e-8 * rel[i, i]
    assert res["edmval"] < 1e-8 < res["edmval_released"]


def test_hessian_impacts_are_skipped_at_an_active_face(tmp_path):
    """--doImpacts at an active face: skipped with a warning, and the fit and
    the rest of the post-fit are still written (not a crash after the fit)."""
    import os
    import subprocess
    import sys

    from rabbit import io_tools

    fn = make_test_tensor(str(tmp_path))
    free = _free_fit(fn)
    ub = float(_phys(free)[_names(free).index("bkg_norm")] - 0.3)
    rabbit_fit = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "bin",
        "rabbit_fit.py",
    )
    out = tmp_path / "out"
    r = subprocess.run(
        [
            sys.executable,
            rabbit_fit,
            fn,
            "-o",
            str(out),
            "-t",
            "0",
            "--doImpacts",
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
            "--constraints",
            "Bounds",
            f"bkg_norm<={ub!r}",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert "Skipping --doImpacts" in r.stdout + r.stderr
    res = io_tools.get_fitresult(str(out / "fitresults.hdf5"))
    assert "cov" in res and "constraints_postfit" in res
    assert not any(k.startswith("impacts") for k in res)


def test_constraints_refuse_multi_device():
    """--constraints with --nDevices > 1 is refused up front (not tested)."""
    import os
    import subprocess
    import sys

    with tempfile.TemporaryDirectory() as tmp:
        fn = make_test_tensor(tmp)
        rabbit_fit = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "bin",
            "rabbit_fit.py",
        )
        r = subprocess.run(
            [
                sys.executable,
                rabbit_fit,
                fn,
                "-o",
                tmp,
                "-t",
                "0",
                "--minimizerMethod",
                "trust-constr",
                "--nDevices",
                "2",
                "--constraints",
                "Bounds",
                "sig>=0",
            ],
            capture_output=True,
            text=True,
        )
        assert r.returncode != 0
        assert "--nDevices" in (r.stderr + r.stdout)


def test_postfit_with_an_indefinite_hessian_across_the_face():
    """Across a nonlinear face the loss Hessian can be indefinite while the fit
    is a constrained minimum. The face-held covariance needs W positive
    definite only ALONG the faces; the released view is then undefined, and
    releasing the face has no local minimum (an unbounded gain)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = make_test_tensor(tmp)
        free = _free_fit(fn)
        i_sig = _names(free).index("sig")
        lb = _phys(free)[i_sig] + 0.1
        tc = _fitter(fn, "trust-constr", cons=[LinearBound({"sig": 1.0}, lb)])
        tc.minimize()
        _, grad, hess = tc.loss_val_grad_hess()
        H = np.asarray(hess)
        ref = tc.constrained_postfit(grad, H)
        # bend the loss DOWN along the face normal only: Z^T H Z is unchanged
        _, J = tc.eval_physical(tc._constraints_fn, tc.x.numpy(), jacobian=True)
        a = J[0] / np.linalg.norm(J[0])
        Hbad = H - 10.0 * float(a @ H @ a) * np.outer(a, a)
        assert np.linalg.eigvalsh(Hbad).min() < 0
        cpf = tc.constrained_postfit(grad, Hbad)
        np.testing.assert_allclose(
            cpf["cov_held"], ref["cov_held"], rtol=1e-8, atol=1e-14
        )
        assert cpf["cov_released"] is None and cpf["edm_released"] is None
        (face,) = cpf["active"]
        assert face["release_dnll"] == np.inf and not face["weakly_binding"]
        assert cpf["release_dnll_joint"] == np.inf
        assert cpf["edm_tangent"] < 1e-8


def test_constraints_survive_a_layout_reinit():
    """The saturated test re-runs init_fit_parms (with a composite model) on a
    copy of the fitter; the constraints must survive it and be re-armed."""
    import copy

    with tempfile.TemporaryDirectory() as tmp:
        fn = make_test_tensor(tmp)
        free = _free_fit(fn)
        i_sig = _names(free).index("sig")
        lb = _phys(free)[i_sig] + 0.1
        tc = _fitter(fn, "trust-constr", cons=[LinearBound({"sig": 1.0}, lb)])
        fs = copy.deepcopy(tc)
        fs.init_fit_parms(
            fs.param_model, [], unblind=[], blinding_group=[], freeze_parameters=[]
        )
        fs.arm_regularizers()
        assert len(fs.constraints) == 1
        fs.minimize()
        assert abs(_phys(fs)[i_sig] - lb) < 1e-5


def _run_saturated(tmp_path, extra=()):
    """rabbit_fit.py with --computeSaturatedProjectionTests and a binding
    upper bound on bkg_norm; returns (fitresult, the mapping's results, ub)."""
    import os
    import subprocess
    import sys

    from rabbit import io_tools

    fn = make_test_tensor(str(tmp_path))
    free = _free_fit(fn)
    ub = float(_phys(free)[_names(free).index("bkg_norm")] - 0.3)
    rabbit_fit = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "bin",
        "rabbit_fit.py",
    )
    out = tmp_path / "out"
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
            "--constraints",
            "Bounds",
            f"bkg_norm<={ub!r}",
            *extra,
        ],
        check=True,
    )
    res = io_tools.get_fitresult(str(out / "fitresults.hdf5"))
    (sat,) = res["mappings"].values()
    return res, sat, ub


def test_saturated_test_is_constrained(tmp_path):
    """End to end: the saturated sub-fit of --computeSaturatedProjectionTests
    honours --constraints like the nominal fit (it used to run unconstrained)."""
    _, sat, ub = _run_saturated(tmp_path)
    st = sat["saturated_fit"]["minimizer_status"]
    assert st["constraint_labels"] == [f"bkg_norm<={ub!r}"]
    assert st["constraint_values"][0] <= ub + 1e-6


def test_saturated_test_is_unchanged_by_preconditioning(tmp_path):
    """End to end with the production flags (fitterAD.sh's full-scope spectral
    preconditioner next to --constraints): the nominal and the saturated
    sub-fit land where the unpreconditioned ones do, and say which frame
    their optimality is in."""
    flags = [
        "--precondition",
        "--preconditionParams",
        ".*",
        "--preconditionBlocks",
        "none",
        "--preconditionTransform",
        "spectral",
    ]
    res0, sat0, ub = _run_saturated(tmp_path / "plain")
    res1, sat1, _ = _run_saturated(tmp_path / "pre", flags)
    mu = {}
    for res, sat, frame in (
        (res0, sat0, "physical"),
        (res1, sat1, "preconditioned"),
    ):
        for key, st in (
            ("fit", res["minimizer_status"]),
            ("sat", sat["saturated_fit"]["minimizer_status"]),
        ):
            assert st["optimality_frame"] == frame
            assert st["constraint_values"][0] <= ub + 1e-6
            mu[key] = max(mu.get(key, 0.0), st["barrier_parameter"])
    # scipy's gtol exit does not wait for the barrier parameter, and the
    # barrier optimum sits ~mu above the constrained minimum in the loss. gtol
    # is tested in the iterated frame, so the two frames may exit at different
    # mu (seen: 6.4e-6 preconditioned, the same exit test passing in y while
    # the physical-frame residual was still 1.2e-10 > gtol).
    for key, a, b in (
        ("fit", res1["nllvalreduced"], res0["nllvalreduced"]),
        (
            "sat",
            sat1["saturated_fit"]["nllvalreduced"],
            sat0["saturated_fit"]["nllvalreduced"],
        ),
    ):
        assert abs(float(a) - float(b)) < 2 * mu[key] + 1e-9, (key, a, b, mu)


def _linear_chi2_fitter(tmp, method, cons=()):
    """A chi2 fit that is exactly quadratic (normal systematics, linear Mu)."""
    from tests.test_logk_no_copy import _normal_tensor

    ind = inputdata.FitInputData(_normal_tensor(tmp, False))
    pm = load_model("Mu", ind, allowNegativeParam=True)
    opts = make_options(minimizerMethod=method, chisqFit=True, noBinByBinStat=True)
    if method == "trust-constr":
        opts.minimizerGtol = 1e-10
        opts.minimizerXtol = 1e-10
        opts.minimizerBarrierTol = 1e-10
        opts.minimizerMaxiter = 500
    f = fitter.Fitter(ind, pm, opts)
    f.set_nobs(ind.data_obs)
    f.constraints = list(cons)
    f.arm_regularizers()
    return f


def test_quadratic_loss_with_constraints_is_minimised_iteratively():
    """A chi2 fit with a linear model takes the closed-form solve, which cannot
    honour constraints: with --constraints it must be minimised iteratively."""
    with tempfile.TemporaryDirectory() as tmp:
        free = _linear_chi2_fitter(tmp, "trust-krylov")
        assert free.is_linear
        free.minimize()
        i_nrm = _names(free).index("bkg_norm")
        ub = _phys(free)[i_nrm] - 0.3
        tc = _linear_chi2_fitter(
            tmp, "trust-constr", cons=[LinearBound({"bkg_norm": 1.0}, ub=ub)]
        )
        assert tc.is_linear
        tc.minimize()
        assert 0 <= ub - _phys(tc)[i_nrm] < 1e-5
        assert tc.minimizer_status()["constr_violation"] < 1e-8


def test_force_linear_with_constraints_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        fn = make_test_tensor(tmp)
        tc = _fitter(
            fn,
            "trust-constr",
            cons=[LinearBound({"sig": 1.0}, lb=0.0)],
            forceLinear=True,
        )
        with pytest.raises(ValueError, match="forceLinear"):
            tc.minimize()


def test_fun_and_jac_share_one_evaluation(monkeypatch):
    """scipy calls fun and jac at the same point: one constraint evaluation."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = make_test_tensor(tmp)
        tc = _fitter(fn, "trust-constr", cons=[LinearBound({"sig": 1.0}, lb=-1e3)])
        (nlc,) = tc.build_scipy_constraints(tc.x.numpy())
        calls = []
        real = tc.eval_physical

        def counting(*a, **k):
            calls.append(1)
            return real(*a, **k)

        monkeypatch.setattr(tc, "eval_physical", counting)
        y = tc.x.numpy() + 0.01
        nlc.fun(y)
        nlc.jac(y)
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# --precondition under trust-constr (theta = theta_ref + T y, T = L^-T)
# ---------------------------------------------------------------------------
#
# A change of variables: the constrained minimum, the multipliers and the
# face-held post-fit must not move; only the path (and its cost) may. The toy
# is the Vandermonde-like polynomial background of test_preconditioner (cond(H)
# ~1e5 at order 8), with a linear row on two coefficients and a nonlinear one
# on two others, both binding. The nonlinear row is a BALL: its feasible set is
# convex. With a non-convex row there can be several constrained minima, and
# since the two frames take different paths they may legitimately end in
# different ones (seen with poly3^2 + poly4 >= lb at order 8: loss 4.63 vs
# 5.33, both KKT points), which would test nothing about the transform.

PRECOND_FULL_SPECTRAL = dict(
    precondition=True,
    preconditionParams=[".*"],
    preconditionBlocks="none",
    preconditionTransform="spectral",
)


class BallBound(Constraints):
    """(x[a] - ca)^2 + (x[b] - cb)^2 <= ub: nonlinear, convex feasible set.

    Centred AWAY from the start point: a row whose gradient vanishes at the
    start is dropped as a constant by the adapter (see Open questions in the
    2026-10-15 trust-constr/precondition study), which would hide it.
    """

    def __init__(self, a, b, ca, cb, ub):
        self.dtype = tf.float64
        self.names, self.centre = (a, b), (float(ca), float(cb))
        self.labels = ["ball"]
        self.lower = np.array([-np.inf])
        self.upper = np.array([float(ub)])
        self._idx = None

    def set_layout(self, parms):
        self._idx = self.resolve_indices(parms, list(self.names), "BallBound")

    def parameter_indices(self):
        return list(self._idx.values())

    def values(self, x):
        (a, b), (ca, cb) = self.names, self.centre
        return tf.reshape(
            tf.square(x[self._idx[a]] - ca) + tf.square(x[self._idx[b]] - cb), [1]
        )


def _poly_constraints(filename):
    """A binding linear row and a binding ball row, placed from the free fit."""
    free = _fitter(filename, "trust-krylov")
    free.minimize()
    names, p = _names(free), _phys(free)
    _, _, hess = free.loss_val_grad_hess()
    cov = np.linalg.inv(np.asarray(hess))
    i1, i2, i3, i4 = (names.index(f"poly{k}") for k in (1, 2, 3, 4))
    # poly1 + poly2 <= its free value - 1 sigma
    lin_ub = p[i1] + p[i2] - np.sqrt(cov[i1, i1] + cov[i2, i2] + 2 * cov[i1, i2])
    # a ball containing the start (all zero) but not the free minimum
    ca, cb = 0.25 * p[i3], 0.25 * p[i4]
    radius = 0.5 * np.hypot(p[i3], p[i4])

    def make():
        return [
            LinearBound({"poly1": 1.0, "poly2": 1.0}, ub=lin_ub),
            BallBound("poly3", "poly4", ca, cb, radius**2),
        ]

    return make, np.sqrt(np.diag(cov))


def _counted_fit(filename, make_cons, **kw):
    """trust-constr fit, counting the loss HVPs the minimizer asks for."""
    f = _fitter(filename, "trust-constr", cons=make_cons(), **kw)
    count = [0]
    real = f.loss_val_grad_hessp

    def counting(v):
        count[0] += 1
        return real(v)

    f.loss_val_grad_hessp = counting
    f.minimize()
    f.loss_val_grad_hessp = real
    return f, count[0]


def test_preconditioned_constraint_jacobian_and_hvp_follow_the_chain_rule():
    """What scipy sees in y: c(theta(y)), J T and T^T d2c T -- checked against
    the dense transform and against finite differences in y."""
    from .test_preconditioner import make_polynomial_tensor

    with tempfile.TemporaryDirectory() as tmp:
        fn = make_polynomial_tensor(tmp, order=8)
        make_cons, _ = _poly_constraints(fn)
        f = _fitter(fn, "trust-constr", cons=make_cons(), **PRECOND_FULL_SPECTRAL)
        pc = f._build_preconditioner()
        assert pc.enabled and pc.cond_before > 1e4
        n = int(f.x.shape[0])
        T = np.stack([pc._apply_T(e) for e in np.eye(n)], axis=1)

        # the transform is looked up per call (fit() rebuilds it on restarts)
        cell = [pc]
        (nlc,) = f.build_scipy_constraints(f.x.numpy(), transform=lambda: cell[0])
        rng = np.random.default_rng(3)
        y = rng.normal(size=n)
        x = pc.to_physical(y)

        cx, jx = f.eval_physical(f._constraints_fn, x, jacobian=True)
        np.testing.assert_allclose(nlc.fun(y), cx, rtol=1e-13)
        np.testing.assert_allclose(nlc.jac(y), jx @ T, rtol=1e-10, atol=1e-12)

        v = np.array([0.7, -1.3])
        p = rng.normal(size=n)
        hp = nlc.hess(y, v).matvec(p)
        dense = T.T @ f.hess_physical(f._constraints_fn, x, v) @ T
        np.testing.assert_allclose(hp, dense @ p, rtol=1e-9, atol=1e-12)
        # and the y-space derivatives are those of the y-space function
        h = 1e-6
        jfd = np.stack(
            [(nlc.fun(y + h * e) - nlc.fun(y - h * e)) / (2 * h) for e in np.eye(n)],
            axis=1,
        )
        np.testing.assert_allclose(nlc.jac(y), jfd, rtol=1e-6, atol=1e-6)
        hfd = (nlc.jac(y + h * p) - nlc.jac(y - h * p)).T @ v / (2 * h)
        np.testing.assert_allclose(hp, hfd, rtol=1e-5, atol=1e-6)

        # swap in a rebuilt (here: identity) transform: the SAME constraint
        # object must follow it, although its cache holds this physical point
        from rabbit.preconditioner import Preconditioner

        cell[0] = Preconditioner.identity(x)
        np.testing.assert_allclose(nlc.fun(np.zeros(n)), cx, rtol=1e-13)
        np.testing.assert_allclose(nlc.jac(np.zeros(n)), jx, rtol=1e-13)


def test_preconditioning_does_not_move_the_constrained_fit():
    """Same minimum, multipliers and face-held covariance with and without the
    full-scope spectral preconditioner; the preconditioned fit is cheaper."""
    from .test_preconditioner import make_polynomial_tensor

    with tempfile.TemporaryDirectory() as tmp:
        fn = make_polynomial_tensor(tmp, order=8)
        make_cons, sigma = _poly_constraints(fn)
        plain, n_plain = _counted_fit(fn, make_cons)
        pre, n_pre = _counted_fit(fn, make_cons, **PRECOND_FULL_SPECTRAL)

        st_plain, st_pre = plain.minimizer_status(), pre.minimizer_status()
        for st in (st_plain, st_pre):
            assert st["status"] == 1, st["message"]  # gtol exit
            assert st["constr_violation"] < 1e-8
        assert st_plain["optimality_frame"] == "physical"
        assert "optimality_physical" not in st_plain
        assert st_pre["optimality_frame"] == "preconditioned"
        assert st_pre["optimality_physical"] < 1e-6

        # the minimum, in units of its own uncertainty, and the loss
        np.testing.assert_allclose(_phys(pre) / sigma, _phys(plain) / sigma, atol=1e-4)
        # up to the barrier offset (see test_saturated_test_is_unchanged_...)
        mu = max(st_plain["barrier_parameter"], st_pre["barrier_parameter"])
        assert abs(float(pre.loss_val()) - float(plain.loss_val())) < 2 * mu + 1e-9
        # both rows bind, with the same multipliers (frame-independent)
        np.testing.assert_allclose(
            st_pre["multipliers"], st_plain["multipliers"], rtol=1e-3
        )
        assert min(abs(m) for m in st_plain["multipliers"]) > 1e-3

        cp_plain, cp_pre = _postfit(plain), _postfit(pre)
        assert [a["label"] for a in cp_plain["active"]] == ["linear", "ball"]
        assert [a["label"] for a in cp_pre["active"]] == ["linear", "ball"]
        scale = np.max(np.abs(cp_plain["cov_held"]))
        np.testing.assert_allclose(
            cp_pre["cov_held"], cp_plain["cov_held"], rtol=0, atol=1e-4 * scale
        )

        # the point of it: fewer Hessian-vector products (the reference Hessian
        # the transform is built from is not counted; it costs one dense
        # Hessian, i.e. n HVPs at most, n = 9 here)
        print(
            f"trust-constr HVPs: plain {n_plain} (nit {st_plain['nit']}), "
            f"preconditioned {n_pre} (nit {st_pre['nit']})"
        )
        assert n_pre < n_plain
