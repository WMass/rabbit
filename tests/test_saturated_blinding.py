"""The projected saturated test must run BLINDED, and must still warm-start.

Two properties of ``save_hists`` in ``bin/rabbit_fit.py``, both of which failed
silently before the commits this file ships with.

1. THE SATURATED FIT MUST NOT UNBLIND ITSELF. The composite re-init recreates
   the offset Variables at the composite size, which creates them at zero.
   Nothing re-armed them, so the saturated fit ran in an unblinded
   frame and wrote the ORIGINAL model's POI in the clear into
   ``results[...]["saturated_fit"]["parms"]`` -- an analysis asking for this
   test on data was unblinding itself in its own output file.

2. THE WARM START MUST ACTUALLY LAND ON THE MAIN FIT'S LOSS. The saturated
   bin scales are POIs of the composite and are not among the three blocks the
   warm start copies, so they keep ``x0default``. They come out of
   ``get_poi()`` as 1 only because ``SaturatedProjectModel`` declares them
   ``blind_exempt`` -- they are the test's own machinery, not a result, so they
   are never offset. Blind them and the composite opens far above the main fit
   and ``q = 2*(NLL_main - NLL_sat)`` starts NEGATIVE, which is not a deviance.

3. THE REGULARIZERS MUST BE ARMED BEFORE THE BLINDED-START CHECK. The composite
   re-init also discharges ``_regularizers_armed`` (the layout changed), and
   ``set_blinding_offsets(True)`` ends in
   ``_check_blinded_start_is_evaluable()``, which COMPUTES THE LOSS.
   ``_compute_nll_components()`` refuses to evaluate a loss whose regularizers
   are unarmed, so a blinded, regularized saturated test died outright -- and
   said "the blinded starting point is outside the range this model can be
   evaluated at", which is the outer half of a chained exception and blames the
   wrong thing entirely. It takes all three (blinding, a regularizer, this
   test) to reproduce, which is why the two properties above could be written
   without noticing it.

Every check is an INVARIANCE or a guard, and each one asserts that blinding is
genuinely armed first, so none of them can pass vacuously through an
accidentally unblinded fitter.
"""

import copy
import os
import pathlib
import tempfile
from types import SimpleNamespace

import hist
import numpy as np
import pytest

from rabbit import fitter, inputdata, tensorwriter
from rabbit.param_models import param_model as pm
from rabbit.regularization.regularizer import Regularizer

NBINS = 20


class OneParamPenalty(Regularizer):
    """A parameter-only penalty on a single named parameter.

    Deliberately minimal, and deliberately name-resolved: what the tests below
    need is only that ``len(fitter.regularizers)`` is nonzero, so that
    ``_compute_nll_components`` takes its armed branch. Name resolution (rather
    than positional indexing) is also what a real regularizer must do to
    survive this path at all -- the composite prepends a second model's POIs,
    so a cached index points at the wrong parameter.
    """

    needs_observables = False

    def __init__(self, name="bkgNorm", dtype=np.float64):
        self.name = name
        self.dtype = dtype
        self._idx = None

    def set_expectations(self, initial_params, initial_observables, parms=None):
        self._idx = self.resolve_indices(parms, [self.name], who="OneParamPenalty")[
            self.name
        ]

    def compute_nll_penalty(self, params, observables):
        import tensorflow as tf

        return tf.square(params[self._idx])


def make_tensor(path):
    np.random.seed(1234)
    ax = hist.axis.Regular(NBINS, -5, 5, name="x")
    h_data = hist.Hist(ax, storage=hist.storage.Double())
    h_sig = hist.Hist(ax, storage=hist.storage.Weight())
    h_bkg = hist.Hist(ax, storage=hist.storage.Weight())
    h_data.fill(
        np.concatenate([np.random.normal(0, 1, 8000), np.random.uniform(-5, 5, 4000)])
    )
    h_sig.fill(np.random.normal(0, 1, 8000))
    h_bkg.fill(np.random.uniform(-5, 5, 4000))

    w = tensorwriter.TensorWriter()
    w.add_channel(h_data.axes, "ch0")
    w.add_data(h_data, "ch0")
    w.add_process(h_sig, "sig", "ch0", signal=True)
    w.add_process(h_bkg, "bkg", "ch0", signal=False)
    w.add_norm_systematic("bkgNorm", ["bkg"], "ch0", 1.05)
    w.write(outfolder=os.path.dirname(path), outfilename=os.path.basename(path))


def make_options(**kwargs):
    defaults = dict(
        earlyStopping=-1,
        noBinByBinStat=True,
        binByBinStatMode="lite",
        binByBinStatType="automatic",
        covarianceFit=False,
        chisqFit=False,
        diagnostics=False,
        minimizerMethod="trust-krylov",
        prefitUnconstrainedNuisanceUncertainty=0.0,
        freezeParameters=[],
        setConstraintMinimum=[],
        unblind=[],
        blindingGroup=[],
        maxRestarts=-1,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


@pytest.fixture(scope="module")
def path():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "saturated_blinding_tensor.hdf5")
        make_tensor(p)
        yield p


def build_main(path, do_blinding=True, regularize=False):
    """A plain ``Mu``: the default model, with squared storage.

    ``regularize`` attaches an :class:`OneParamPenalty` the way the driver does
    (``tau`` then ``regularizers``), before ``defaultassign()`` arms it.
    """
    ind = inputdata.FitInputData(path)
    model = pm.Mu(ind)
    f = fitter.Fitter(ind, model, make_options(), do_blinding=do_blinding)
    if regularize:
        f.tau.assign(1.0)
        f.regularizers = [OneParamPenalty()]
    f.defaultassign()
    if do_blinding:
        f.set_blinding_offsets(True)
    return ind, model, f


def build_saturated(f, blind, rearm=True, arm_regularizers_early=True):
    """Mirror of the composite re-init in ``bin/rabbit_fit.py::save_hists``.

    Reproduced here rather than imported because ``rabbit_fit.py`` is a script;
    what is asserted below is a property of the Fitter and the models, not of
    the driver's argument parsing.
    """
    orig = f.param_model
    channel_info = {"ch0": {"axes": [hist.axis.Regular(NBINS, 0, NBINS, name="g")]}}
    sat = pm.SaturatedProjectModel(
        f.indata, channel_info, {"ch0": np.arange(NBINS, dtype=np.int64)}
    )
    composite = pm.CompositeParamModel([orig, sat])

    fs = copy.deepcopy(f)
    saved_regularizers = fs.regularizers
    saved_tau = float(fs.tau.numpy())
    fs.init_fit_parms(
        composite, [], unblind=[], blinding_group=[], freeze_parameters=[]
    )
    fs.regularizers = saved_regularizers
    fs.tau.assign(saved_tau)
    fs.xdefaultassign()
    # `arm_regularizers_early=False` is the pre-fix driver order, kept so the
    # tests below can show the early call is load bearing rather than defensive.
    if arm_regularizers_early:
        fs.arm_regularizers()
    if fs.do_blinding and rearm:
        fs.set_blinding_offsets(blind=blind)

    x_main = f.x.numpy()
    if orig.npoi > 0:
        fs.x[: orig.npoi].assign(x_main[: orig.npoi])
    if orig.npou > 0:
        fs.x[composite.npoi : composite.npoi + orig.npou].assign(
            x_main[orig.npoi : orig.nparams]
        )
    fs.x[composite.nparams :].assign(x_main[orig.nparams :])
    # The driver arms again here, at the warm-started point, which is where the
    # expectations a regularizer records should come from.
    fs.arm_regularizers()
    return orig, sat, composite, fs


def _armed(f):
    """Is this fitter's frame actually offset? Guards every test below."""
    return not np.allclose(f.blinding.offsets_poi_add.numpy(), 0.0, rtol=0, atol=1e-12)


# --- 1. the leak --------------------------------------------------------------


def test_composite_reinit_leaves_the_frame_unblinded(path):
    """The mechanism of the leak, pinned so the re-arm is never dropped.

    ``init_fit_parms`` recreates the offset Variables at the composite size,
    and it creates them at the IDENTITY. Nothing inside the Fitter re-arms
    them, so a driver that does not do it explicitly runs the saturated fit in
    a fully unblinded frame -- which is why the call in ``save_hists`` is load
    bearing rather than defensive.
    """
    _, _, f = build_main(path)
    assert _armed(f), "vacuous: the main fitter is not blinded"

    _, _, _, fs = build_saturated(f, blind=True, rearm=False)
    assert not _armed(fs), (
        "init_fit_parms no longer disarms the composite; if that is deliberate, "
        "the re-arm in save_hists can go -- but check get_poi() first"
    )


def test_saturated_fit_does_not_run_unblinded(path):
    """With the re-arm, the composite is back in a blinded frame.

    Without it the saturated fit writes the original model's POI in the clear
    into ``results[...]["saturated_fit"]["parms"]``, so any analysis asking for
    this test on data was unblinding itself in its own output file.
    """
    _, _, f = build_main(path)
    assert _armed(f), "vacuous: the main fitter is not blinded"

    _, _, _, fs = build_saturated(f, blind=True)
    assert _armed(fs)


def test_original_poi_is_written_blinded(path):
    """The coordinate stored for the analysis POI must differ from the physical
    one, i.e. what lands in the results file is still hidden."""
    _, orig, f = build_main(path)
    _, _, _, fs = build_saturated(f, blind=True)
    assert _armed(fs)

    x_stored = fs.x.numpy()[: orig.npoi]
    poi_physical = fs.get_poi().numpy()[: orig.npoi]
    assert not np.allclose(x_stored, poi_physical, rtol=0, atol=1e-12)


# --- 2. the warm start, which blinding is what makes non-trivial ---------------


def test_bin_scales_are_one_after_the_warm_start(path):
    """The warm start never copies the saturated slots, so this holds only
    because they are never offset -- SaturatedProjectModel declares them
    blind_exempt, so they keep the 1.0 their default stores."""
    _, orig, f = build_main(path)
    _, _, composite, fs = build_saturated(f, blind=True)
    assert _armed(fs), "vacuous: nothing is blinded at all in this fitter"

    scales = fs.get_poi().numpy()[orig.npoi : composite.npoi]
    assert scales.shape == (NBINS,)
    np.testing.assert_allclose(scales, 1.0, rtol=0, atol=1e-9)


def test_warm_start_sits_on_the_main_loss_while_blinded(path):
    """``q = 2*(NLL_main - NLL_sat)`` must start at 0, not below it.

    This is the whole point of the warm start: the saturated model is the exact
    identity at its default, so the composite opens exactly where the main fit
    is and the statistic reads as "what do these free bin scales buy from
    HERE". A negative start is not a deviance.
    """
    _, _, f = build_main(path)
    nll_main = float(f.reduced_nll().numpy())

    _, _, _, fs = build_saturated(f, blind=True)
    assert _armed(fs)
    nll_warm = float(fs.reduced_nll().numpy())

    assert np.isclose(nll_warm, nll_main, rtol=0, atol=1e-6), (nll_warm, nll_main)
    assert 2.0 * (nll_main - nll_warm) >= -1e-6


def test_the_same_holds_unblinded(path):
    """The warm start is a property of the LAYOUT, not of the offsets.

    Run the whole thing with blinding off -- both arms, since ``save_hists``
    passes the main fitter's own ``blind`` down and a disarmed composite fed
    armed coordinates is not a configuration the driver can produce. If this
    agrees and the blinded one above does too, the equality is not a
    cancellation between two offsets.
    """
    _, _, f = build_main(path, do_blinding=False)
    nll_main = float(f.reduced_nll().numpy())

    _, _, _, fs = build_saturated(f, blind=False)
    assert np.isclose(float(fs.reduced_nll().numpy()), nll_main, rtol=0, atol=1e-6)


# --- 3. blinding + a regularizer + this test ----------------------------------
#
# NOTE ON WHAT PINS WHAT. The mirror above is a copy of the driver's sequence,
# so a test that only exercises the mirror cannot catch the driver drifting away
# from it -- it would just test the copy. So this section has two kinds of check:
# the Fitter-level contract (the requirement itself, order-parameterised), and
# one source-level guard that reads bin/rabbit_fit.py and asserts it honours
# that contract. The guard is the regression test; the others document why.


def _save_hists_source():
    """The text of ``save_hists`` from the driver, via ast (never imported).

    ``bin/rabbit_fit.py`` is a script, so importing it to reach the function is
    not safe; parsing is. Consistent with the module docstring's reason for
    mirroring rather than importing the composite re-init.
    """
    import ast

    src = (
        pathlib.Path(__file__).resolve().parents[1] / "bin" / "rabbit_fit.py"
    ).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "save_hists":
            return ast.get_source_segment(src, node)
    raise AssertionError("save_hists not found in bin/rabbit_fit.py")


def test_driver_arms_regularizers_before_the_blinded_start_check():
    """THE REGRESSION TEST. ``save_hists`` must arm before it blinds.

    ``set_blinding_offsets(True)`` ends in
    ``_check_blinded_start_is_evaluable()``, which computes the loss, and
    ``_compute_nll_components()`` refuses a loss whose regularizers are unarmed
    after ``init_fit_parms`` changed the layout. The driver's own
    ``arm_regularizers()`` sits just before ``minimize()``, which is the right
    place for the expectations it records but far too late for this check --
    hence a second, earlier call.

    A source check rather than a behavioural one because the behaviour lives in
    a script's function; see ``_save_hists_source``.
    """
    src = _save_hists_source()
    arm = src.find("arm_regularizers()")
    blind = src.find("set_blinding_offsets(blind=blind)")

    assert arm != -1, "save_hists no longer arms the regularizers at all"
    assert blind != -1, (
        "save_hists no longer re-arms blinding; if that is deliberate this test "
        "can go, but check test_saturated_fit_does_not_run_unblinded first"
    )
    assert arm < blind, (
        "save_hists arms the regularizers only AFTER "
        "set_blinding_offsets(blind=blind), which computes the loss -- a "
        "blinded, regularized saturated test will die there, reported as "
        "'the blinded starting point is outside the range this model can be "
        "evaluated at'"
    )


def test_the_unarmed_order_really_does_raise(path):
    """Why the guard above matters: the wrong order is a hard crash.

    Pinned at the Fitter level so the requirement survives even if the driver
    is restructured -- and so the guard above cannot be dismissed as cosmetic.
    """
    _, _, f = build_main(path, regularize=True)
    assert _armed(f), "vacuous: the main fitter is not blinded"
    assert len(f.regularizers) == 1, "vacuous: no regularizer to arm"

    with pytest.raises(RuntimeError, match="blinded starting point"):
        build_saturated(f, blind=True, arm_regularizers_early=False)


def test_blinded_regularized_saturated_reinit_can_evaluate_its_loss(path):
    """And the driver's order works: all three at once, loss finite."""
    _, _, f = build_main(path, regularize=True)
    assert _armed(f), "vacuous: the main fitter is not blinded"
    assert len(f.regularizers) == 1, "vacuous: no regularizer to arm"

    _, _, _, fs = build_saturated(f, blind=True)
    assert _armed(fs)
    assert fs._regularizers_armed
    assert np.isfinite(float(fs._compute_loss().numpy()))


def test_the_penalty_follows_the_parameter_through_the_composite(path):
    """Arming after the re-init must re-resolve by NAME, not reuse an index.

    The composite prepends the saturated model's POIs, so the penalized
    parameter sits at a different position than it did in the main fit. If the
    index were cached the penalty would silently apply to another parameter.
    """
    _, _, f = build_main(path, regularize=True)
    idx_main = f.regularizers[0]._idx

    _, _, _, fs = build_saturated(f, blind=True)
    idx_comp = fs.regularizers[0]._idx

    names_main = np.asarray(f.parms).astype(str)
    names_comp = np.asarray(fs.parms).astype(str)
    assert names_main[idx_main] == "bkgNorm"
    assert names_comp[idx_comp] == "bkgNorm"
    assert idx_main != idx_comp, (
        "vacuous: the composite did not move bkgNorm, so a cached index would "
        "have worked and this asserts nothing"
    )


def test_unregularized_and_unblinded_paths_are_unaffected(path):
    """Neither arm that could not reproduce the bug regresses.

    These two are why it went unnoticed: with no regularizer there is nothing to
    arm, and unblinded ``set_blinding_offsets`` returns before the check.
    """
    _, _, f = build_main(path, regularize=False)
    _, _, _, fs = build_saturated(f, blind=True, arm_regularizers_early=False)
    assert np.isfinite(float(fs._compute_loss().numpy()))

    _, _, f = build_main(path, do_blinding=False, regularize=True)
    _, _, _, fs = build_saturated(f, blind=False, arm_regularizers_early=False)
    assert np.isfinite(float(fs._compute_loss().numpy()))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
