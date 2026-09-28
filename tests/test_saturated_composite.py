"""The projected saturated test must compose with any analysis model.

``--computeSaturatedProjectionTests`` wraps the analysis model in a
``CompositeParamModel`` together with a ``SaturatedProjectModel``, one free
scale per output bin of the projection. The bin scales are auxiliary, not
results, so ``SaturatedProjectModel`` declares them as POUs (``npoi = 0``):

* they never enter the Fitter's POI block, so they are never blinded and never
  take part in its squaring transform -- and ``CompositeParamModel`` does not
  ask them to agree with the analysis model on ``allowNegativeParam``. The
  test is therefore available for analyses whose POI must be allowed negative
  (alpha_s, an additively blinded POI, plain ``Mu --allowNegativeParam``);
* positivity is owned by the model: the stored coordinate is sqrt(scale) and
  compute() squares it, for any coordinate the minimiser visits;
* the composite layout is ``[poi_o | pou_o | pou_sat]``, so the analysis
  model's block is contiguous at ``[0 : orig.nparams]``;
* the saturated model is the exact identity at its default, so the warm start
  of the saturated fit sits at the main fit's loss and the deviance is >= 0;
* the disagreement guard is still there for POI-carrying submodels.
"""

import os
import tempfile
from types import SimpleNamespace

import hist
import numpy as np
import pytest
import tensorflow as tf

from rabbit import fitter, inputdata, tensorwriter
from rabbit.param_models.param_model import (
    CompositeParamModel,
    Mu,
    SaturatedProjectModel,
)

NBINS = 20  # bins of the input channel
NGROUP = 4  # bins of the projection output -> npou of the saturated model
POI_START = 0.3  # deliberately not 1: a 1 would hide a sqrt/square mix-up
POU_START = 2.5

# flat index of the output bin each input bin contributes to, i.e. what
# Mapping.output_indices() returns for a projection that groups NBINS/NGROUP
# adjacent bins
GROUPS = np.repeat(np.arange(NGROUP), NBINS // NGROUP)


class ToyModel:
    """Analysis-model stand-in: one POI and one POU, POI allowed negative.

    Mirrors the shape of a physics param model (SCETlibADParamModel): a POI
    that must not be squared, plus model nuisances. The POU is what makes the
    composite layout check non-trivial -- with npou = 0 the permuted and the
    naive concatenation agree by accident.
    """

    def __init__(
        self,
        indata,
        allowNegativeParam=True,
        blind_additive_scale=None,
    ):
        self.indata = indata
        self.npoi = 1
        self.npou = 1
        self.nparams = 2
        self.params = np.array([b"alphaS", b"lambda2"])
        self.allowNegativeParam = allowNegativeParam
        self.is_linear = False
        start = POI_START if allowNegativeParam else np.sqrt(POI_START)
        self.xparamdefault = tf.constant([start, POU_START], dtype=indata.dtype)
        if blind_additive_scale is not None:
            self.blind_additive_scale = blind_additive_scale

    def compute(self, param, full=False):
        # one number per (bin, proc) is not needed: a per-process column is
        # enough to make the yields depend on both parameters
        col = tf.reshape(1.0 + 0.1 * param[0] + 0.01 * param[1], [1, 1])
        return tf.concat(
            [col, tf.ones([1, self.indata.nproc - 1], dtype=col.dtype)], axis=1
        )


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


def mapping_channel_info():
    """The 'channel_info of the mapping' argument: used only to name params."""
    return {"ch0": {"axes": [hist.axis.Regular(NGROUP, 0, NGROUP, name="g")]}}


def make_saturated(indata, **kwargs):
    return SaturatedProjectModel(
        indata, mapping_channel_info(), {"ch0": GROUPS}, **kwargs
    )


@pytest.fixture(scope="module")
def tensor_path():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "saturated_composite_tensor.hdf5")
        make_tensor(path)
        yield path


@pytest.fixture(scope="module")
def indata(tensor_path):
    return inputdata.FitInputData(tensor_path)


# --- the saturated model on its own ------------------------------------------


def test_bin_scales_are_pous_stored_as_sqrt(indata):
    """No POIs, one POU per output bin, stored as sqrt(scale).

    ``parms`` and its covariance are written out as ``fitter.x``, so the
    stored coordinate is what every saturated_* entry in the output file
    means.
    """
    m = make_saturated(indata)
    assert m.npoi == 0 and m.npou == NGROUP
    np.testing.assert_allclose(
        m.xparamdefault.numpy(), np.ones(NGROUP), rtol=0, atol=1e-14
    )
    # default scale is 1, and sqrt(1) == 1, so check the sqrt with a default
    # that is not 1 -- only expectSignal can move it
    m2 = make_saturated(indata, expectSignal=[("saturated_ch0_g0", 4.0)])
    np.testing.assert_allclose(
        m2.xparamdefault.numpy(), [2.0, 1.0, 1.0, 1.0], rtol=0, atol=1e-14
    )


def test_compute_squares_the_stored_coordinate(indata):
    xstored = np.array([1.3, 0.7, 1.0, 0.2])  # = sqrt(scale)
    r = make_saturated(indata).compute(tf.constant(xstored, dtype=indata.dtype))
    expected = np.repeat(xstored**2, NBINS // NGROUP).reshape(-1, 1)
    np.testing.assert_allclose(r.numpy(), expected, rtol=1e-15, atol=0)


def test_self_squaring_keeps_the_scales_positive(indata):
    """A negative stored coordinate is fine; a negative SCALE is not.

    The Fitter does not transform POUs, so this is the only thing standing
    between the data driving a bin scale negative and the Poisson log() going
    to NaN.
    """
    xstored = np.array([-1.5, 2.0, -0.4, 0.0])
    r = make_saturated(indata).compute(tf.constant(xstored, dtype=indata.dtype))
    r = r.numpy()
    assert np.all(r >= 0.0), r
    np.testing.assert_allclose(
        r, np.repeat(xstored**2, NBINS // NGROUP).reshape(-1, 1), rtol=1e-15, atol=0
    )


def test_never_advertised_as_linear(indata):
    """x -> x**2 in compute(), so the model is not linear.

    ``Fitter.is_linear`` short-circuits to a single Cholesky solve, which would
    be wrong for a quadratic parametrisation of the scales.
    """
    assert make_saturated(indata).is_linear is False


def test_extra_kwargs_are_ignored(indata):
    """The old ``allowNegativeParam`` argument no longer means anything."""
    m = make_saturated(indata, allowNegativeParam=False)
    assert m.npoi == 0 and m.npou == NGROUP


# --- composite construction and layout ---------------------------------------


@pytest.mark.parametrize("allow_negative", [True, False])
def test_composite_follows_the_analysis_model(indata, allow_negative):
    """The saturated model has no POIs, so it does not vote on
    allowNegativeParam: the composite takes the analysis model's flag,
    whichever it is."""
    toy = ToyModel(indata, allowNegativeParam=allow_negative)
    c = CompositeParamModel([toy, make_saturated(indata)])
    assert c.allowNegativeParam is allow_negative
    assert c.npoi == 1
    assert c.npou == 1 + NGROUP
    assert c.is_linear is False


def test_composite_layout_keeps_the_analysis_block_contiguous(indata):
    """[poi_o | pou_o | pou_sat]: the analysis model's parameters are the
    leading ``orig.nparams`` entries, in their own order."""
    toy = ToyModel(indata, allowNegativeParam=True)
    sat = make_saturated(indata)
    c = CompositeParamModel([toy, sat])

    assert list(c.params[: toy.nparams]) == list(toy.params)
    assert list(c.params[toy.nparams :]) == list(sat.params)
    assert list(c.params[: c.npoi]) == [b"alphaS"]
    np.testing.assert_allclose(
        c.xparamdefault.numpy(),
        np.concatenate([[POI_START, POU_START], np.ones(NGROUP)]),
        rtol=0,
        atol=1e-14,
    )


def test_composite_applies_the_transform_per_submodel(indata):
    """The analysis POI passes through raw while the saturated slice is squared.

    Written against the composite's own compute(), so it fails if the slicing
    and the per-submodel transform ever disagree.
    """
    toy = ToyModel(indata, allowNegativeParam=True)
    sat = make_saturated(indata)
    c = CompositeParamModel([toy, sat])

    xpoi_toy, xpou_toy, xsat = -0.7, 1.9, np.array([1.3, 0.7, 1.0, 0.2])
    param = tf.constant(
        np.concatenate([[xpoi_toy, xpou_toy], xsat]), dtype=indata.dtype
    )
    got = c.compute(param).numpy()
    expected = (
        toy.compute(tf.constant([xpoi_toy, xpou_toy], dtype=indata.dtype)).numpy()
        * sat.compute(tf.constant(xsat, dtype=indata.dtype)).numpy()
    )
    np.testing.assert_allclose(got, expected, rtol=1e-15, atol=0)

    # the analysis POI is NOT squared: flipping its sign must change the result
    param_flip = tf.constant(
        np.concatenate([[-xpoi_toy, xpou_toy], xsat]), dtype=indata.dtype
    )
    assert not np.allclose(c.compute(param_flip).numpy(), got)
    # a saturated coordinate IS squared: flipping its sign must not
    xsat_flip = xsat.copy()
    xsat_flip[0] *= -1
    param_sat_flip = tf.constant(
        np.concatenate([[xpoi_toy, xpou_toy], xsat_flip]), dtype=indata.dtype
    )
    np.testing.assert_allclose(
        c.compute(param_sat_flip).numpy(), got, rtol=1e-15, atol=0
    )


def test_disagreement_guard_still_refuses_a_model_that_cannot_self_enforce(indata):
    """`Mu` relies on the Fitter, so mixing it with a permissive POI model is
    still unrepresentable and must still raise rather than silently drop the
    positivity of the signal strengths."""
    mu = Mu(indata, allowNegativeParam=False)
    assert mu.npoi > 0 and mu.allowNegativeParam is False
    with pytest.raises(ValueError, match="allowNegativeParam"):
        CompositeParamModel([ToyModel(indata, allowNegativeParam=True), mu])


@pytest.mark.parametrize("allow_negative", [True, False])
def test_plain_mu_composes_either_way(indata, allow_negative):
    """rabbit's own default model, with and without --allowNegativeParam.

    The signal strengths follow the Fitter's transform (or not) while the bin
    scales are squared inside the saturated model in both cases.
    """
    mu = Mu(indata, allowNegativeParam=allow_negative)
    sat = make_saturated(indata)
    c = CompositeParamModel([mu, sat])
    assert c.allowNegativeParam is allow_negative
    assert c.npoi == mu.npoi and c.npou == NGROUP

    # compute() receives physical POIs (the Fitter squares them when needed)
    xsig = np.full(mu.npoi, -0.5 if allow_negative else 0.5)
    xsat = np.array([1.3, 0.7, 1.0, 0.2])
    param = tf.constant(np.concatenate([xsig, xsat]), dtype=indata.dtype)
    got = c.compute(param).numpy()
    expected = (
        mu.compute(tf.constant(xsig, dtype=indata.dtype)).numpy()
        * sat.compute(tf.constant(xsat, dtype=indata.dtype)).numpy()
    )
    np.testing.assert_allclose(got, expected, rtol=1e-15, atol=0)
    if allow_negative:
        assert np.any(got < 0.0), "vacuous: the negative signal strength was squared"


# --- through the Fitter, the way bin/rabbit_fit.py does it --------------------


def build_main(tensor_path, allowNegativeParam, do_blinding):
    ind = inputdata.FitInputData(tensor_path)
    model = ToyModel(ind, allowNegativeParam=allowNegativeParam)
    f = fitter.Fitter(ind, model, make_options(), do_blinding=do_blinding)
    f.defaultassign()
    if do_blinding:
        f.set_blinding_offsets(True)
    f.set_nobs(f.expected_yield())  # Asimov at the model default
    return ind, model, f


def build_saturated(f, do_blinding):
    """Mirror of the composite re-init in bin/rabbit_fit.py.

    Kept in the test rather than imported because rabbit_fit.py is a script;
    the assertions below are about the model and the Fitter, not the driver.
    """
    import copy

    orig = f.param_model
    sat = make_saturated(f.indata)
    composite = CompositeParamModel([orig, sat])

    fs = copy.deepcopy(f)
    fs.init_fit_parms(
        composite, [], unblind=[], blinding_group=[], freeze_parameters=[]
    )
    fs.xdefaultassign()
    if do_blinding:
        fs.set_blinding_offsets(blind=True)

    x_main = f.x.numpy()
    fs.x[: orig.nparams].assign(x_main[: orig.nparams])
    fs.x[composite.nparams :].assign(x_main[orig.nparams :])
    fs.arm_regularizers()
    return sat, composite, fs


@pytest.mark.parametrize(
    "allow_negative,do_blinding",
    [
        (True, True),  # a linearly stored POI, blinded: the clean configuration
        (True, False),
        (False, False),  # the legacy squared `Mu`-like path
        (False, True),  # squared and blinded: works, but warns about the covariance
    ],
)
def test_fitter_accepts_the_composite_and_keeps_the_layout(
    tensor_path, allow_negative, do_blinding
):
    _, orig, f = build_main(tensor_path, allow_negative, do_blinding)
    sat, composite, fs = build_saturated(f, do_blinding)

    # fitter-facing layout: [poi_o | pou_o | pou_sat | thetas]
    assert list(fs.parms[: composite.npoi]) == [b"alphaS"]
    assert list(fs.parms[composite.npoi : composite.nparams]) == [b"lambda2"] + list(
        sat.params
    )
    assert list(fs.parms[composite.nparams :]) == list(f.indata.systs)
    assert int(fs.x.shape[0]) == composite.nparams + f.indata.nsyst


@pytest.mark.parametrize(
    "allow_negative,do_blinding",
    [
        (True, True),
        (True, False),
        (False, False),
        (False, True),
    ],
)
def test_warm_start_sits_exactly_on_the_main_loss(
    tensor_path, allow_negative, do_blinding
):
    """The saturated model must be the exact identity at its default.

    That is what makes the deviance ``2*(NLL_main - NLL_sat)`` non-negative and
    readable as "what do these free bin scales buy from the main fit's point".
    It is also the check that catches a sqrt/square mismatch between
    xparamdefault and compute(): any mismatch moves the scales off 1 and the
    two losses apart.
    """
    _, _, f = build_main(tensor_path, allow_negative, do_blinding)
    nll_main = float(f.reduced_nll().numpy())
    _, _, fs = build_saturated(f, do_blinding)
    nll_warm = float(fs.reduced_nll().numpy())
    assert np.isclose(nll_warm, nll_main, rtol=0, atol=1e-9), (nll_warm, nll_main)


def test_bin_scales_are_not_blinded(tensor_path):
    """Only the analysis POI is offset; the bin scales are POUs and are not.

    This is the failure POUs fix by construction: a blinded bin scale opens at
    ``offset`` rather than 1, and the composite loss starts far from the main
    fit's.
    """
    _, orig, f = build_main(tensor_path, True, True)
    sat, composite, fs = build_saturated(f, True)

    add = fs.blinding.offsets_poi_add.numpy()
    assert add.shape == (composite.npoi,) == (orig.npoi,)
    # the analysis POI IS blinded, or the check below proves nothing
    assert abs(add[0]) > 1e-6

    scales = np.square(fs.get_model_nui().numpy()[orig.npou :])
    np.testing.assert_allclose(scales, 1.0, rtol=0, atol=1e-12)


def test_bin_scales_stay_positive_wherever_the_minimiser_goes(tensor_path):
    """Self-squaring guarantees positivity for any stored coordinate -- a
    negative scale sends the expected yield negative and the Poisson log to
    NaN."""
    _, orig, f = build_main(tensor_path, True, True)
    sat, composite, fs = build_saturated(f, True)
    sl = slice(orig.nparams, composite.nparams)

    x = fs.x.numpy()
    x[sl] += np.array([-30.0, 12.0, -3.0, 0.0])
    fs.x.assign(x)

    assert np.all(np.isfinite(fs.expected_yield().numpy()))
    assert np.all(fs.expected_yield().numpy() > 0.0)
    assert np.isfinite(float(fs.reduced_nll().numpy()))


def test_free_bin_scales_can_only_lower_the_loss(tensor_path):
    """A short minimization of the composite must not end above the main loss.

    The statistic is a difference of NLLs, so a saturated fit that stops higher
    than its own starting point would report a negative deviance.
    """
    _, _, f = build_main(tensor_path, True, True)
    nll_main = float(f.reduced_nll().numpy())
    _, _, fs = build_saturated(f, True)
    fs.minimize()
    nll_sat = float(fs.reduced_nll().numpy())
    assert nll_sat <= nll_main + 1e-7, (nll_sat, nll_main)


# --- the scale must survive the composite rabbit_fit.py builds ----------------


def test_declared_scale_survives_the_saturated_composite(tensor_path):
    """The composite here is built INSIDE rabbit_fit.py, so the user has no
    object to re-declare the scale on -- if it is dropped in the propagation
    there is no remedy, and for a free POI the weak-blinding warning cannot
    even report it. This is the configuration that makes that matter.
    """
    ind = inputdata.FitInputData(tensor_path)

    plain = ToyModel(ind)
    scaled = ToyModel(ind, blind_additive_scale=7.0)

    comp_plain = CompositeParamModel([plain, make_saturated(ind)])
    comp_scaled = CompositeParamModel([scaled, make_saturated(ind)])

    f_plain = fitter.Fitter(ind, comp_plain, make_options(), do_blinding=True)
    f_scaled = fitter.Fitter(ind, comp_scaled, make_options(), do_blinding=True)

    off_plain = f_plain.blinding.values_poi_add[0]
    off_scaled = f_scaled.blinding.values_poi_add[0]

    assert off_plain != 0.0, "vacuous: the analysis POI is not being offset"
    assert np.isclose(off_scaled, 7.0 * off_plain, rtol=1e-12, atol=0)

    # the saturated bin scales are not POIs, so they carry no scale at all
    assert comp_scaled.blind_additive_scale.shape == (1,)


def test_a_start_the_model_cannot_evaluate_is_refused(tensor_path):
    """The limitation of the uncompensated start, stated as a refusal.

    An uncompensated armed fit opens at default + offset, and an offset large
    enough can take the POI where the yields go negative and the likelihood is
    NaN. There is nothing for the minimiser to descend from, so arming raises
    rather than letting the fit start and fail later. Such a POI cannot be
    blinded this way -- a real limitation, reported as one.

    The offset is set explicitly rather than taken from the draw. Relying on
    the drawn value would make the test a property of the current
    BLINDING_SEED_SALT -- passing or not depending on the sign and size of one
    sample -- which is the same mistake the weak-smearing warning used to make.
    """
    ind = inputdata.FitInputData(tensor_path)
    model = ToyModel(ind)
    f = fitter.Fitter(ind, model, make_options(), do_blinding=True)
    f.defaultassign()
    f.set_nobs(f.expected_yield())

    # ToyModel scales by 1 + 0.1 * poi, so anything past -10 drives the yields
    # negative whatever the seed produced
    f.blinding.values_poi_add[0] = -50.0
    with pytest.raises(RuntimeError, match="non-finite likelihood"):
        f.set_blinding_offsets(True)


def test_an_evaluable_start_arms_cleanly(tensor_path):
    """The contrast to the refusal above: same machinery, evaluable start."""
    ind = inputdata.FitInputData(tensor_path)
    model = ToyModel(ind)
    f = fitter.Fitter(ind, model, make_options(), do_blinding=True)
    f.defaultassign()
    f.set_nobs(f.expected_yield())

    f.blinding.values_poi_add[0] = 2.0
    f.set_blinding_offsets(True)
    assert np.isfinite(float(f.reduced_nll().numpy()))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
