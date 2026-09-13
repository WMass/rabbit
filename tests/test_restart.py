"""Tests for restarting the minimizer after an early-stopping stall.

scipy's trust-region loop shrinks the trust radius by 4x on every rejected step
with no lower bound, and holds the radius in a local variable. Once it has
collapsed the method takes infinitesimal steps and the loss stops changing far
from any minimum; a fresh minimize() call resets the radius and the descent
resumes. These tests cover the plumbing that turns that stall into a restart
instead of a give-up.
"""

import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

from rabbit import fitter, inputdata
from rabbit.callbacks import FitterCallback, merge_callbacks
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
