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

Every check is an INVARIANCE or a guard, and each one asserts that blinding is
genuinely armed first, so none of them can pass vacuously through an
accidentally unblinded fitter.
"""

import copy
import os
import tempfile
from types import SimpleNamespace

import hist
import numpy as np
import pytest

from rabbit import fitter, inputdata, tensorwriter
from rabbit.param_models import param_model as pm

NBINS = 20


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


def build_main(path, do_blinding=True):
    """A plain ``Mu``: the default model, with squared storage."""
    ind = inputdata.FitInputData(path)
    model = pm.Mu(ind)
    f = fitter.Fitter(ind, model, make_options(), do_blinding=do_blinding)
    f.defaultassign()
    if do_blinding:
        f.set_blinding_offsets(True)
    return ind, model, f


def build_saturated(f, blind, rearm=True):
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
    fs.init_fit_parms(
        composite, [], unblind=[], blinding_group=[], freeze_parameters=[]
    )
    fs.xdefaultassign()
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
    return orig, sat, composite, fs


def _armed(f):
    """Is this fitter's frame actually offset? Guards every test below."""
    return not np.allclose(f._blinding_offsets_poi_add.numpy(), 0.0, rtol=0, atol=1e-12)


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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
