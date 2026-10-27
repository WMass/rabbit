"""Tests for restarting the minimizer after an early-stopping stall.

scipy's trust-region loop shrinks the trust radius by 4x on every rejected step
with no lower bound, and holds the radius in a local variable. Once it has
collapsed the method takes infinitesimal steps and the loss stops changing far
from any minimum; a fresh minimize() call resets the radius and the descent
resumes. These tests cover the plumbing that turns that stall into a restart
instead of a give-up.
"""

import logging
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from rabbit import fitter, inputdata, io_tools
from rabbit.callbacks import (
    STATUS_MINIMIZER_RAISED,
    STATUS_STOPPED_EARLY,
    FitterCallback,
    merge_callbacks,
)
from rabbit.param_models.helpers import load_model

from .test_preconditioner import make_polynomial_tensor
from .test_sparse_fit import check_results, make_options, make_test_tensor


def _result(fun, x):
    return SimpleNamespace(fun=fun, x=np.asarray(x, dtype=float))


def test_stopped_early_flag_set_only_on_a_stall():
    """fit() keys the restart off this flag, so it must not fire otherwise."""
    cb = FitterCallback(np.zeros(2), early_stopping=3)
    # a steadily improving fit never stalls
    for i, loss in enumerate([10.0, 9.0, 8.0, 7.0, 6.0, 5.0]):
        cb(_result(loss, [i, i]))
    assert not cb.stopped_early

    # a flat one does
    cb = FitterCallback(np.zeros(2), early_stopping=3)
    raised = False
    try:
        for loss in [10.0, 9.0, 9.0, 9.0, 9.0, 9.0]:
            cb(_result(loss, [0, 0]))
    except ValueError:
        raised = True
    assert raised and cb.stopped_early


def test_disabled_early_stopping_never_stalls():
    cb = FitterCallback(np.zeros(2), early_stopping=-1)
    for _ in range(20):
        cb(_result(5.0, [0, 0]))
    assert not cb.stopped_early


def test_merge_callbacks_concatenates_into_one_continuous_run():
    """Callers report on the whole fit, so restarts must look continuous."""
    a = FitterCallback(np.zeros(2), early_stopping=-1)
    a.loss_history = [10.0, 9.0]
    a.time_history = [1.0, 2.0]
    a.iiter = 2

    b = FitterCallback(np.ones(2), early_stopping=-1)
    b.loss_history = [8.0, 7.0]
    b.time_history = [0.5, 1.5]  # clocked from its own construction
    b.iiter = 2
    b.xval = np.array([3.0, 4.0])

    merged = merge_callbacks(a, b)
    assert merged is a
    assert merged.loss_history == [10.0, 9.0, 8.0, 7.0]
    # second run's times offset by the first run's elapsed
    assert merged.time_history == [1.0, 2.0, 2.5, 3.5]
    assert merged.iiter == 4
    np.testing.assert_array_equal(merged.xval, [3.0, 4.0])


def test_merge_callbacks_with_no_accumulator_returns_the_first():
    cb = FitterCallback(np.zeros(2), early_stopping=-1)
    assert merge_callbacks(None, cb) is cb


def _fit(filename, **kw):
    indata_obj = inputdata.FitInputData(filename)
    param_model = load_model("Mu", indata_obj)
    options = make_options(**kw)
    f = fitter.Fitter(indata_obj, param_model, options)
    f.set_nobs(indata_obj.data_obs)
    f.minimize()
    val, grad, hess = f.loss_val_grad_hess()
    from rabbit.tfhelpers import edmval_cov

    edmval, cov = edmval_cov(grad, hess)
    cov_np = np.asarray(cov.numpy() if hasattr(cov, "numpy") else cov)
    return dict(
        param=f.x[: param_model.nparams].numpy(),
        theta=f.x[param_model.nparams :].numpy(),
        param_err=np.sqrt(np.diag(cov_np)[: param_model.nparams]),
        nll=f.reduced_nll().numpy(),
        edmval=edmval,
        parms=f.parms,
    )


def test_restarts_do_not_change_a_fit_that_does_not_stall():
    """--maxRestarts must be inert when nothing stalls."""
    with tempfile.TemporaryDirectory() as tmp:
        filename = make_test_tensor(tmp)
        plain = _fit(filename, maxRestarts=0)
        with_restarts = _fit(filename, earlyStopping=20, maxRestarts=5)
        assert check_results("no restarts", plain, "maxRestarts=5", with_restarts)


def test_restarting_is_on_by_default_and_unbounded():
    """Default is -1: keep restarting while the loss keeps dropping."""

    with tempfile.TemporaryDirectory() as tmp:
        filename = make_test_tensor(tmp)
        indata_obj = inputdata.FitInputData(filename)
        param_model = load_model("Mu", indata_obj)
        # options object without the attribute at all -> the getattr default
        options = make_options()
        del options.maxRestarts
        f = fitter.Fitter(indata_obj, param_model, options)
        assert f.max_restarts == -1


def test_unbounded_restarts_stop_when_a_restart_stops_improving():
    """The improvement check, not a counter, is what ends the loop."""
    with tempfile.TemporaryDirectory() as tmp:
        filename = make_polynomial_tensor(tmp, order=6)
        # unbounded restarts must still terminate
        res = _fit(filename, earlyStopping=10, maxRestarts=-1)
        assert np.isfinite(res["nll"])


def test_restarts_reach_at_least_as_low_a_loss_on_a_hard_model():
    """On a model with many correlated unconstrained params, restarting can
    only help: the restart is only taken after a stall, and is abandoned as
    soon as it stops reducing the loss."""
    with tempfile.TemporaryDirectory() as tmp:
        filename = make_polynomial_tensor(tmp, order=6)
        stop_only = _fit(filename, earlyStopping=10, maxRestarts=0)
        restarted = _fit(filename, earlyStopping=10, maxRestarts=5)
        assert restarted["nll"] <= stop_only["nll"] + 1e-6


def test_preconditioner_is_rebuilt_before_every_restart():
    """The transform whitens the Hessian at the point it was built.

    Once the fit has stalled somewhere else that Hessian has changed, so a
    restart must rebuild it there rather than reuse the one from the starting
    point -- otherwise the restart resets the trust radius but keeps a
    transform that no longer conditions anything.

    The stall is forced rather than coaxed out of the model: whether a real fit
    trips a given --earlyStopping threshold turns on differences far below the
    scale of the fit, and TF's multithreaded CPU reductions are not bitwise
    reproducible, so keying the test on that makes it flaky.
    """
    n_restarts = 2

    class StallEveryFewIterations(FitterCallback):
        def __call__(self, intermediate_result):
            self.iiter += 1
            self.loss_history.append(intermediate_result.fun)
            self.time_history.append(float(self.iiter))
            self.xval = intermediate_result.x
            if self.iiter >= 3:
                self.stopped_early = True
                raise ValueError("forced stall")

    with tempfile.TemporaryDirectory() as tmp:
        filename = make_polynomial_tensor(tmp, order=6)
        indata_obj = inputdata.FitInputData(filename)
        param_model = load_model("Mu", indata_obj)
        options = make_options(
            earlyStopping=3, maxRestarts=n_restarts, precondition=True
        )
        f = fitter.Fitter(indata_obj, param_model, options)
        f.set_nobs(indata_obj.data_obs)

        built_at = []
        original = f._build_preconditioner

        def counting():
            built_at.append(np.array(f.x.numpy(), copy=True))
            return original()

        f._build_preconditioner = counting
        monkeyed = fitter.FitterCallback
        fitter.FitterCallback = StallEveryFewIterations
        try:
            callback = f.fit()
        finally:
            fitter.FitterCallback = monkeyed

        assert callback is not None
        # one build up front, then one before each of the forced restarts
        assert len(built_at) == 1 + n_restarts, (
            f"expected {1 + n_restarts} builds (1 initial + {n_restarts} "
            f"refreshes), saw {len(built_at)}"
        )
        # each refresh must happen where the fit now is, not back at the start
        for i in range(1, len(built_at)):
            moved = np.linalg.norm(built_at[i] - built_at[i - 1])
            assert moved > 1e-6, (
                f"refresh {i} happened at the same point ({moved}); "
                "the transform was reused, not rebuilt"
            )


# -- the stall THRESHOLD -------------------------------------------------
#
# --stallRelTol is the only part of this that touches a default-on path
# (--earlyStopping defaults to 20), so the equivalence at 0.0 is what most
# wants pinning: it is what guarantees existing callers see no change.


def _feed(losses, early_stopping=3, stall_rel_tol=0.0):
    """Run a loss trajectory through the callback; did it call it a stall?"""
    cb = FitterCallback(
        np.zeros(2), early_stopping=early_stopping, stall_rel_tol=stall_rel_tol
    )
    for i, loss in enumerate(losses):
        try:
            cb(_result(loss, [i, i]))
        except ValueError:
            return True, cb
    return False, cb


# healthy descent, a hard flat, a crawl, one that gets worse, and the inputs
# where the algebra stops holding: ref == 0.0 (the relative form would divide
# by zero) and a loss pinned at an infinity (inf - inf and 0.0 * inf are both
# NaN, and every NaN comparison is False). +inf is REACHABLE -- __call__ raises
# on a NaN loss but not an infinite one, and a Poisson term with a
# non-positive prediction and non-zero data gives exactly +inf -- so an
# equivalence claim that excluded it would be the wrong claim.
TRAJECTORIES = [
    [10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
    [10.0, 9.0, 9.0, 9.0, 9.0, 9.0],
    [1e4, 1e4 * (1 - 1e-5), 1e4 * (1 - 2e-5), 1e4 * (1 - 3e-5), 1e4 * (1 - 4e-5)],
    [10.0, 9.0, 9.5, 10.0, 10.5, 11.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [np.inf] * 5,
    [-np.inf] * 5,
    [np.inf, np.inf, 5.0, 5.0, 5.0, 5.0],
    [np.inf, np.inf, np.inf, np.inf, 1.0, 1.0],
]


@pytest.mark.parametrize("losses", TRAJECTORIES)
def test_stall_rel_tol_zero_is_exactly_the_original_test(losses):
    """The default must be bit-identical to the test it generalises.

    ``gained <= 0`` is ``ref - loss <= 0`` is ``ref <= loss``, so this is
    algebraically guaranteed; pinned because it is the whole basis for calling
    the new option opt-in. Includes ref == 0.0, the one input where the
    relative form would divide by zero and has to fall back.
    """
    stalled, _ = _feed(losses, stall_rel_tol=0.0)

    # the predicate as it stood before --stallRelTol existed
    hist, exact = [], False
    for loss in losses:
        if len(hist) > 3 and hist[-3] <= loss:
            exact = True
            break
        hist.append(loss)

    assert stalled == exact


def test_stall_rel_tol_detects_a_crawl_the_exact_test_misses():
    """The gap the option exists for: improving, but not meaningfully."""
    crawl = [1e4 * (1 - 1e-5 * i) for i in range(8)]
    assert not _feed(crawl, stall_rel_tol=0.0)[0]  # invisible to the exact test
    assert _feed(crawl, stall_rel_tol=1e-3)[0]


def test_stall_rel_tol_leaves_a_healthy_descent_alone():
    """It must not fire on a fit that is still making real progress."""
    healthy = [1e4 * 0.5**i for i in range(8)]
    for tol in (0.0, 1e-4, 1e-3, 1e-2):
        assert not _feed(healthy, stall_rel_tol=tol)[0], f"fired at tol={tol}"


def test_stall_rel_tol_defaults_to_zero_everywhere_it_is_set():
    """Three places hold this default; a flip in any of them is a behaviour
    change on a default-on path, so pin all three."""
    assert FitterCallback(np.zeros(2)).stall_rel_tol == 0.0

    from rabbit import parsing

    assert parsing.common_parser().get_default("stallRelTol") == 0.0

    # and the Fitter's getattr fallback, for callers whose options predate it
    with tempfile.TemporaryDirectory() as tmp:
        filename = make_test_tensor(tmp)
        indata_obj = inputdata.FitInputData(filename)
        options = make_options()
        assert not hasattr(options, "stallRelTol")
        f = fitter.Fitter(indata_obj, load_model("Mu", indata_obj), options)
        assert f.stallRelTol == 0.0


def test_default_path_keeps_the_original_early_stopping_message():
    """Every existing user sees this string; the default path must not reword it."""
    stalled, cb = _feed([10.0, 9.0, 9.0, 9.0, 9.0, 9.0], stall_rel_tol=0.0)
    assert stalled
    with pytest.raises(ValueError, match=r"No reduction in loss after 3 iterations"):
        _raise(cb_losses=[10.0, 9.0, 9.0, 9.0, 9.0, 9.0], stall_rel_tol=0.0)

    # and the relative phrasing only appears when a threshold is in play
    with pytest.raises(ValueError, match=r"Loss improved only .* relative"):
        _raise(cb_losses=[1e4 * (1 - 1e-5 * i) for i in range(8)], stall_rel_tol=1e-3)


def _raise(cb_losses, stall_rel_tol):
    cb = FitterCallback(np.zeros(2), early_stopping=3, stall_rel_tol=stall_rel_tol)
    for i, loss in enumerate(cb_losses):
        cb(_result(loss, [i, i]))


def test_negative_stall_rel_tol_is_rejected():
    """A negative threshold would require the loss to WORSEN to count as
    stalled -- it weakens the test, so it is a footgun rather than an option."""
    with pytest.raises(ValueError, match=r"must be >= 0"):
        FitterCallback(np.zeros(2), early_stopping=3, stall_rel_tol=-1e-4)

    from rabbit import parsing

    with pytest.raises(SystemExit):
        parsing.common_parser().parse_args(["d.hdf5", "--stallRelTol", "-1e-4"])


# -- the minimizer status of a fit that does not return normally ------------
#
# scipy returns no result when the callback raises, which is how early
# stopping ends a round, so the fitter has to describe such a round itself.
# It used to keep the callback's trust-constr state, which every other method
# leaves unset: a fit ended by early stopping had no minimizer status, and
# fitresults.hdf5 no "minimizer_status". With --earlyStopping 20 on by default
# that is a common end for a converged fit, not a corner case.

# A --stallRelTol far above any relative gain declares a stall at the first
# check, the third callback of every round under --earlyStopping 1, whatever
# the loss does. A real stall turns on the last bits of the loss, which TF's
# multithreaded CPU reductions do not reproduce, so a test keyed on one is flaky.
FORCED_STALL = dict(earlyStopping=1, stallRelTol=1e10)


def test_stopped_result_describes_the_kept_iterate():
    """An early stop under trust-constr keeps scipy's counts and KKT measures
    of the last accepted iterate; a failure names the exception."""

    def state(fun, nit):
        return SimpleNamespace(
            fun=fun, x=np.zeros(2), nit=nit, nfev=nit + 1, optimality=0.5 / nit
        )

    cb = FitterCallback(np.zeros(2), early_stopping=1)
    with pytest.raises(ValueError) as stall:
        for nit, fun in enumerate([3.0, 2.0, 2.0], start=1):
            cb(state(fun, nit))
    st = cb.stopped_result(stall.value)
    assert (st.success, st.status) == (False, STATUS_STOPPED_EARLY)
    assert st.message == str(stall.value)
    assert (st.nit, st.nfev, st.optimality) == (2, 3, 0.25)

    st = FitterCallback(np.zeros(2)).stopped_result(RuntimeError())
    assert (st.success, st.status) == (False, STATUS_MINIMIZER_RAISED)
    assert (st.nit, st.nfev) == (0, -1)
    assert st.message == "Minimizer raised RuntimeError"


def _fitter(filename, **kw):
    indata_obj = inputdata.FitInputData(filename)
    f = fitter.Fitter(indata_obj, load_model("Mu", indata_obj), make_options(**kw))
    f.set_nobs(indata_obj.data_obs)
    return f


def _status_fit(filename, **kw):
    f = _fitter(filename, **kw)
    cb = f.minimize()
    return f, cb, f.minimizer_status()


def _failing_after(n):
    """A FitterCallback that raises a genuine error (not a stall) on its
    (n+1)-th call, after n iterations."""

    class Fails(FitterCallback):
        def __call__(self, intermediate_result):
            if self.iiter >= n:
                raise RuntimeError("simulated failure")
            super().__call__(intermediate_result)

    return Fails


@pytest.mark.parametrize("max_restarts", [0, 2])
def test_a_fit_ended_by_early_stopping_has_a_status(max_restarts):
    """No restarts allowed, and restarts that run out while the loss still
    falls (the hard model is far from its minimum after 2 iterations)."""
    with tempfile.TemporaryDirectory() as tmp:
        filename = make_polynomial_tensor(tmp, order=6)
        f, cb, st = _status_fit(filename, maxRestarts=max_restarts, **FORCED_STALL)
    assert cb.stopped_early
    assert st is not None, "a fit ended by early stopping has no minimizer status"
    assert st["success"] is False
    assert st["status"] == STATUS_STOPPED_EARLY
    assert st["restarts"] == max_restarts
    # the last round, up to the iterate the fit keeps; no evaluation count
    assert st["nit"] == 2 and st["nfev"] == -1
    # the callback's reason, then how the restarts ended
    assert st["message"].startswith("Loss improved only"), st["message"]
    assert "early stopping" in st["message"]
    if max_restarts == 0:
        assert st["message"].endswith("Restarting is disabled (--maxRestarts 0).")
    else:
        assert "The last restart still reduced the loss" in st["message"]
        assert st["message"].endswith("--maxRestarts 2 allows no more.")


def test_a_restart_that_buys_nothing_ends_the_fit_with_a_status(monkeypatch):
    """The other stall exit: the restart loop gives up because the last
    restart did not lower the loss (here every round ends at the same one)."""

    class FlatStall(FitterCallback):
        def __call__(self, intermediate_result):
            if self.iiter >= 2:
                self.stopped_early = True
                raise ValueError("forced stall.")
            self.iiter += 1
            self.loss_history.append(1.0)
            self.time_history.append(float(self.iiter))
            self.xval = intermediate_result.x

    monkeypatch.setattr(fitter, "FitterCallback", FlatStall)
    with tempfile.TemporaryDirectory() as tmp:
        f, cb, st = _status_fit(make_polynomial_tensor(tmp, order=6), earlyStopping=20)
    assert st["status"] == STATUS_STOPPED_EARLY and st["success"] is False
    assert st["restarts"] == 1 and st["nit"] == 2
    assert st["message"] == (
        "forced stall. The last restart did not reduce the loss further (1.0 -> 1.0)."
    )


@pytest.mark.parametrize("method", ["trust-krylov", "trust-constr"])
def test_a_minimizer_that_raises_after_a_step_keeps_the_last_iterate(
    monkeypatch, caplog, method
):
    """A genuine error, not a stall, once the fit has moved: the fit carries
    on from the last accepted iterate, with status -3 and a warning in the log.
    trust-constr used to report it as an early stop (status -2), the others
    not at all."""
    monkeypatch.setattr(fitter, "FitterCallback", _failing_after(3))
    with tempfile.TemporaryDirectory() as tmp, caplog.at_level(logging.WARNING):
        f, cb, st = _status_fit(
            make_polynomial_tensor(tmp, order=6),
            earlyStopping=20,
            minimizerMethod=method,
        )
    assert not cb.stopped_early
    assert st["status"] == STATUS_MINIMIZER_RAISED and st["success"] is False
    assert st["message"] == (
        "Minimizer raised RuntimeError: simulated failure; kept the last "
        "accepted iterate."
    )
    assert st["restarts"] == 0
    assert any(
        r.levelno == logging.WARNING
        and "its EDM tells whether it is a minimum" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]
    # the fit keeps the last accepted iterate, and the status describes it
    assert np.isclose(float(f.loss_val()), cb.loss_history[-1], rtol=1e-10)
    if method == "trust-constr":
        assert st["nit"] >= 1 and "optimality" in st  # scipy's count, KKT kept
    else:
        assert st["nit"] == 3 and st["nfev"] == -1


# -- a minimizer that raises before accepting a step -------------------------
#
# There is nothing to recover then: carrying on reported the start point as
# the fit result, with exit code 0 (seen: an op XLA cannot compile, raised by
# the first loss evaluation). It is an error now, raised with the status kept.


class _Injected(Exception):
    pass


def _reject_every_step(f):
    """Make every point but the start infinitely bad, so the minimizer
    rejects every step it proposes. Returns the start point."""
    x0 = f.x.numpy().copy()
    real = f.loss_val_grad

    def only_finite_at_the_start():
        val, grad = real()
        if not np.array_equal(f.x.numpy(), x0):
            val = np.asarray(np.inf)
        return val, grad

    f.loss_val_grad = only_finite_at_the_start
    return x0


def _stall_then_fail(rounds):
    """A callback that stalls after two iterations in the first round and
    raises a genuine error at once in the next; ``rounds`` collects them."""

    class StallThenFail(FitterCallback):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            rounds.append(self)

        def __call__(self, intermediate_result):
            if len(rounds) > 1:
                raise RuntimeError("the restart fails")
            if self.iiter >= 2:
                self.stopped_early = True
                raise ValueError("forced stall.")
            super().__call__(intermediate_result)

    return StallThenFail


def test_a_minimizer_that_fails_at_the_first_call_raises(tmp_path):
    snap = str(tmp_path / "snapshot.hdf5")
    f = _fitter(
        make_polynomial_tensor(str(tmp_path), order=6),
        earlyStopping=20,
        snapshotFile=snap,
    )
    x0 = f.x.numpy().copy()

    def fails():
        raise _Injected("cannot evaluate the loss")

    f.loss_val_grad = fails
    with pytest.raises(RuntimeError, match="never accepted a step") as err:
        f.minimize()
    assert isinstance(err.value.__cause__, _Injected)
    assert str(err.value) == (
        "Minimizer raised _Injected: cannot evaluate the loss; it never accepted "
        "a step, so the fit has no result."
    )
    st = f.minimizer_status()
    assert st["status"] == STATUS_MINIMIZER_RAISED and st["success"] is False
    assert st["message"] == str(err.value) and st["nit"] == 0
    # left where it started, not at a trial point of the failed minimizer
    np.testing.assert_array_equal(f.x.numpy(), x0)
    # and the failure snapshot is still written
    with h5py.File(snap, "r") as s:
        assert s.attrs["reason"] == "minimizer-failed"
        assert s.attrs["error"] == "cannot evaluate the loss"
        np.testing.assert_array_equal(s["x"][...], x0)


def test_raising_after_rejected_steps_only_is_an_error_too(monkeypatch, tmp_path):
    """The callback runs once per iteration, rejected steps included, so the
    iteration count is not what decides: three iterations that never left the
    start point are no more a result than none."""
    f = _fitter(make_polynomial_tensor(str(tmp_path), order=6), earlyStopping=20)
    x0 = _reject_every_step(f)
    monkeypatch.setattr(fitter, "FitterCallback", _failing_after(3))
    with pytest.raises(RuntimeError, match="never accepted a step"):
        f.minimize()
    assert f.minimizer_status()["nit"] == 3
    np.testing.assert_array_equal(f.x.numpy(), x0)


def test_a_restart_that_fails_at_once_keeps_the_earlier_rounds(monkeypatch):
    """Only the first round can leave nothing to keep: a restart follows a
    stall, a result in itself, so one that fails before its first step
    leaves the fit where the previous round stalled."""
    rounds = []
    monkeypatch.setattr(fitter, "FitterCallback", _stall_then_fail(rounds))
    with tempfile.TemporaryDirectory() as tmp:
        f, cb, st = _status_fit(make_polynomial_tensor(tmp, order=6), earlyStopping=20)
    assert len(rounds) == 2
    assert st["status"] == STATUS_MINIMIZER_RAISED and st["restarts"] == 1
    assert st["nit"] == 0
    assert st["message"].endswith("; kept the last accepted iterate.")
    assert np.isclose(float(f.loss_val()), rounds[0].loss_history[-1], rtol=1e-10)


def test_a_stall_in_place_is_kept_when_the_restart_fails(monkeypatch, tmp_path):
    """A first round that stalls without moving -- as a fit started at its
    minimum does -- has a result, its stall point, so a restart that then
    fails keeps it rather than raising."""
    f = _fitter(make_polynomial_tensor(str(tmp_path), order=6), earlyStopping=20)
    x0 = _reject_every_step(f)
    rounds = []
    monkeypatch.setattr(fitter, "FitterCallback", _stall_then_fail(rounds))
    f.minimize()
    st = f.minimizer_status()
    assert len(rounds) == 2
    assert st["status"] == STATUS_MINIMIZER_RAISED and st["restarts"] == 1
    np.testing.assert_array_equal(f.x.numpy(), x0)


def test_a_fit_that_returns_keeps_scipys_status(monkeypatch):
    """The normal exit is scipy's result unchanged, before and after restarts
    (forced here by a stall in the first round only)."""
    with tempfile.TemporaryDirectory() as tmp:
        filename = make_polynomial_tensor(tmp, order=6)
        f, cb, st = _status_fit(filename, minimizerMaxiter=30)
        rounds = []

        class StallsInTheFirstRound(FitterCallback):
            def __call__(self, intermediate_result):
                if self.iiter >= 2 and len(rounds) == 0:
                    rounds.append(self.iiter)
                    self.stopped_early = True
                    raise ValueError("forced stall")
                super().__call__(intermediate_result)

        monkeypatch.setattr(fitter, "FitterCallback", StallsInTheFirstRound)
        g, _, st_restarted = _status_fit(filename, minimizerMaxiter=30)

    for fit, status, restarts in ((f, st, 0), (g, st_restarted, 1)):
        res = fit.minimizer_result
        assert status["status"] == res.status >= 0
        assert status["message"] == res.message
        assert (status["nit"], status["nfev"]) == (res.nit, res.nfev)
        assert status["restarts"] == restarts


def test_the_cli_writes_the_status_of_a_fit_ended_by_early_stopping(tmp_path):
    """End to end: such a fit used to leave fitresults.hdf5 without a
    minimizer_status."""
    filename = make_polynomial_tensor(str(tmp_path), order=6)
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
            filename,
            "-o",
            str(out),
            "-t",
            "0",
            "--earlyStopping",
            str(FORCED_STALL["earlyStopping"]),
            "--stallRelTol",
            str(FORCED_STALL["stallRelTol"]),
            "--maxRestarts",
            "0",
        ],
        check=True,
    )
    res = io_tools.get_fitresult(str(out / "fitresults.hdf5"))
    assert "minimizer_status" in res, "fitresults.hdf5 has no minimizer_status"
    st = res["minimizer_status"]
    assert st["status"] == STATUS_STOPPED_EARLY and not st["success"]
    assert st["restarts"] == 0 and st["nit"] == 2
    assert st["message"].endswith("Restarting is disabled (--maxRestarts 0).")


# runs bin/rabbit_fit.py with the loss made to raise at its first evaluation
_FAILING_RABBIT_FIT = """
import runpy
import sys

from rabbit import fitter


class InjectedError(Exception):
    pass


def fails():
    raise InjectedError("injected first-call failure")


make_tf_functions = fitter.Fitter._make_tf_functions


def make_failing_tf_functions(self):
    make_tf_functions(self)  # binds the tf.functions per instance
    self.loss_val_grad = fails


fitter.Fitter._make_tf_functions = make_failing_tf_functions
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def test_the_cli_fails_when_the_minimizer_fails_at_the_first_call(tmp_path):
    """End to end: such a fit used to write the start point as its result
    and exit 0. Now it exits non-zero, writes no result and still leaves the
    failure snapshot."""
    filename = make_polynomial_tensor(str(tmp_path), order=6)
    rabbit_fit = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "bin",
        "rabbit_fit.py",
    )
    out = tmp_path / "out"
    snap = tmp_path / "snapshot.hdf5"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _FAILING_RABBIT_FIT,
            rabbit_fit,
            filename,
            "-o",
            str(out),
            "-t",
            "0",
            "--snapshotFile",
            str(snap),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0, proc.stderr[-2000:]
    assert (
        "RuntimeError: Minimizer raised InjectedError: injected first-call "
        "failure; it never accepted a step, so the fit has no result."
    ) in proc.stderr, proc.stderr[-2000:]
    with h5py.File(out / "fitresults.hdf5", "r") as res:
        assert "results" not in res
    with h5py.File(snap, "r") as s:
        assert s.attrs["reason"] == "minimizer-failed"
