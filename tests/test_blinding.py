"""
Test that blinding is a change of variables that leaves the PHYSICS alone.

rabbit blinds by reparametrising the likelihood, not by masking output: the
getters apply the offset on the way INTO ParamModel.compute() and the constraint
term, so the model and the NLL see the physical value while ``fitter.x`` -- the
minimizer's coordinate, and what gets written out -- is the blinded one.

Blinding is ADDITIVE, always: a translation, whose Jacobian is the identity,
so the covariance, the uncertainties and every impact come out EXACTLY
unblinded while the central value stays hidden. The multiplicative form that
used to be the default divided all of those by the random factor -- leaving
only relative uncertainties usable, and making the reported sigma itself a
channel for the offset, since sigma_true / sigma_reported WAS the offset.

Every check below is an INVARIANCE, so no test prints, returns or asserts on an
offset value.
"""

import os
import tempfile
from types import SimpleNamespace

import hist
import numpy as np
import pytest
import tensorflow as tf

from rabbit import fitter, inputdata, tensorwriter
from rabbit.param_models.param_model import CompositeParamModel, ParamModel

# Deliberately NON-ZERO, so that the stored coordinate and the physical value
# are distinguishable: with a default of 0 an offset test cannot tell "x was
# left alone" from "x was set to the offset".
START = 0.3


class ToyModel(ParamModel):
    """One POI scaling the signal, linear so the fit solves exactly."""

    def __init__(self, indata, blind_additive_scale=None):
        super().__init__(indata)
        self.npoi = 1
        self.npou = 0
        self.params = np.array([b"alphaS"])
        self.xparamdefault = tf.constant([START], dtype=indata.dtype)
        self.is_linear = True
        self.allowNegativeParam = True
        if blind_additive_scale is not None:
            self.blind_additive_scale = blind_additive_scale

    def compute(self, param, full=False):
        # No numpy on `param`: compute() runs inside a tf.function, where it is
        # symbolic. The tests assert observable consequences instead.
        nproc = self.indata.nproc
        col = tf.reshape(1.0 + 0.1 * param[0], [1, 1])
        return tf.concat([col, tf.ones([1, nproc - 1], dtype=col.dtype)], axis=1)


class SecondToyModel(ToyModel):
    """A second POI-carrying model, with a DIFFERENT parameter name so its
    deterministic draw differs from ToyModel's."""

    def __init__(self, indata, **kwargs):
        super().__init__(indata, **kwargs)
        self.params = np.array([b"alphaS2"])


class PouOnlyModel(ParamModel):
    """Carries no POIs, like every model in the ABCD family."""

    def __init__(self, indata, name=b"nui"):
        super().__init__(indata)
        self.npoi = 0
        self.npou = 1
        self.params = np.array([name])
        self.xparamdefault = tf.constant([0.0], dtype=indata.dtype)
        self.is_linear = True
        self.allowNegativeParam = True

    def compute(self, param, full=False):
        return tf.ones([1, self.indata.nproc], dtype=self.indata.dtype)


def make_tensor(path):
    np.random.seed(1234)
    ax = hist.axis.Regular(20, -5, 5, name="x")
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
    # One ordinary constrained systematic, so the theta block is not empty.
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
        p = os.path.join(d, "blinding_tensor.hdf5")
        make_tensor(p)
        yield p


def build(path, do_blinding, **opts):
    ind = inputdata.FitInputData(path)
    model = ToyModel(ind)
    f = fitter.Fitter(ind, model, make_options(**opts), do_blinding=do_blinding)
    return ind, model, f


def _asimov(f):
    """Asimov data at the current point.

    Required for the Hessian checks to mean anything: with nobs = 0 the Poisson
    term ``sum(nexp - nobs*log nexp)`` collapses to ``sum(nexp)``, which is
    LINEAR in the yields, so the POI curvature is exactly zero and a comparison
    of Hessians passes vacuously via ``isclose(0, 0)``.
    """
    return f.expected_yield()


def test_model_sees_the_offset_while_x_does_not(path):
    """compute() gets x + offset; fitter.x -- what gets written out -- does not.

    The two frames differ by exactly the offset, and the stored one is the
    declared default: that is what makes the coordinate safe to write out and
    what stops the offset being recoverable from it.
    """
    _, _, fb = build(path, True)
    fb.defaultassign()
    fb.set_blinding_offsets(True)

    add = float(fb._blinding_offsets_poi_add[0].numpy())
    assert add != 0.0, "vacuous: POI was not blinded"

    assert np.isclose(float(fb.x[0].numpy()), START, rtol=0, atol=1e-12)
    assert np.isclose(float(fb.get_poi()[0].numpy()), START + add, rtol=0, atol=1e-12)


def test_x0_untouched_by_arming(path):
    """x0 is the model frame and must NOT be shifted.

    The constraint compares get_x() against x0, so offsetting both would
    double-count the shift and move the penalty's minimum.
    """
    _, _, f = build(path, True)
    f.defaultassign()
    before = f.x0.numpy().copy()
    f.set_blinding_offsets(True)
    assert np.array_equal(before, f.x0.numpy())


def test_determinism_across_fitters(path):
    """Same input, independent fitters: identical offsets, without printing one."""
    _, _, f1 = build(path, True)
    _, _, f2 = build(path, True)
    for f in (f1, f2):
        f.defaultassign()
        f.set_blinding_offsets(True)
    d1 = f1.get_poi().numpy() - f1.x[:1].numpy()
    d2 = f2.get_poi().numpy() - f2.x[:1].numpy()
    np.testing.assert_array_equal(d1, d2)
    assert not np.allclose(d1, 0.0)  # non-trivial


def test_unblind_disables_the_offset(path):
    """--unblind on the POI leaves the coordinate and the physical value equal."""
    _, _, f = build(path, True, unblind=["alphaS"])
    f.defaultassign()
    f.set_blinding_offsets(True)
    assert np.isclose(
        float(f.get_poi()[0].numpy()), float(f.x[0].numpy()), rtol=0, atol=1e-14
    )


def test_squared_storage_is_warned_about_not_refused(path, caplog):
    """Squared storage leaks the value through the covariance, and is reported.

    With ``allowNegativeParam=False`` the stored coordinate is ``sqrt(poi)``, so
    at the minimum ``d(poi)/dx = 2*sqrt(poi)`` and the reported uncertainty is
    ``sigma_poi / (2*sqrt(poi))`` -- a function of the TRUE value. The offset is
    not in it (sigma is preserved exactly), but an expected ``sigma_poi``, which
    an Asimov study gives for free, inverts it and the offset falls out.

    A warning rather than a refusal because the squaring is rabbit's default and
    blinding is on by default for a data fit: refusing would make every existing
    analysis unrunnable rather than merely leaky. The leak also predates any
    offset -- it is a property of reporting sqrt(poi) and its uncertainty.
    """
    ind = inputdata.FitInputData(path)
    model = ToyModel(ind)
    model.allowNegativeParam = False

    with caplog.at_level("WARNING"):
        fitter.Fitter(ind, model, make_options(), do_blinding=True)
    assert any("allowNegativeParam=False" in r.message for r in caplog.records)

    # and a linearly stored POI is not warned about
    caplog.clear()
    with caplog.at_level("WARNING"):
        fitter.Fitter(ind, ToyModel(ind), make_options(), do_blinding=True)
    assert not any("allowNegativeParam=False" in r.message for r in caplog.records)


# --- the SCALE of the draw, which carries the parameter's units --------------


def test_additive_scale_multiplies_the_draw(path):
    """``blind_additive_scale`` rescales the offset and nothing else.

    The offset is an ABSOLUTE shift in the POI's own fit units, so a POI whose
        sigma is O(1) in those units would be offset by well under a sigma. The
        model declares its units here; the draw itself (seed, sign, magnitude) must
        be untouched, which is what pins this as a rescale rather than a different
        random number.
    """
    ind = inputdata.FitInputData(path)
    base = fitter.Fitter(ind, ToyModel(ind), make_options(), do_blinding=True)
    scaled = fitter.Fitter(
        ind,
        ToyModel(ind, blind_additive_scale=1000.0),
        make_options(),
        do_blinding=True,
    )
    off_base = base._blinding_values_poi_add[0]
    off_scaled = scaled._blinding_values_poi_add[0]

    assert off_base != 0.0, "vacuous: the unscaled draw is already zero"
    assert np.isclose(off_scaled, 1000.0 * off_base, rtol=1e-12, atol=0)
    # a rescale, not a different random number: the sign is part of the draw
    assert np.sign(off_scaled) == np.sign(off_base)


def test_default_scale_is_the_historical_draw(path):
    """Not declaring a scale must reproduce the pre-existing offset exactly."""
    ind = inputdata.FitInputData(path)
    implicit = fitter.Fitter(ind, ToyModel(ind), make_options(), do_blinding=True)
    explicit = fitter.Fitter(
        ind,
        ToyModel(ind, blind_additive_scale=1.0),
        make_options(),
        do_blinding=True,
    )
    assert implicit._blinding_values_poi_add[0] == explicit._blinding_values_poi_add[0]


def _postfit_variances(f, ind):
    """Fit and hand back the variance vector the driver passes to the check.

    The driver builds this either as the covariance diagonal or, under
    --noHessian, from the CG-solved POI rows; the check only ever reads the
    diagonal, so it takes the vector.
    """
    f.set_nobs(ind.data_obs)
    f.minimize()
    _, grad, hess = f.loss_val_grad_hess()
    _, cov = f.edmval_cov(grad, hess)
    return np.diag(np.asarray(cov))


MSG = "Blinding was too narrow to hide"


def test_squared_storage_is_not_reported_when_nothing_is_blinded(path, caplog):
    """--unblind on every POI means there is no offset, so nothing to leak.

    The standard "now unblind my result" run still has do_blinding on and still
    uses the default squared storage, so a guard that looks only at the
    parametrisation warns about a leak that cannot happen -- and tells the
    reader to change parametrisation for no reason.
    """
    ind = inputdata.FitInputData(path)
    model = ToyModel(ind)
    model.allowNegativeParam = False

    caplog.clear()
    with caplog.at_level("WARNING"):
        fitter.Fitter(ind, model, make_options(unblind=["alphaS"]), do_blinding=True)
    assert not any("allowNegativeParam=False" in r.message for r in caplog.records)


def test_weak_blinding_is_reported_against_the_measured_uncertainty(path, caplog):
    """Judged on the smearing width against the sigma the fit measured.

    That is the only yardstick that decides whether a value is hidden: an
    offset of half a sigma leaves the truth recoverable whatever units it is
    in. A narrow smearing must be reported and an ample one must not, with the
    same model and the same data -- only the declared scale differs.
    """
    ind = inputdata.FitInputData(path)

    # a smearing far smaller than the fit's own resolution
    f = fitter.Fitter(
        ind, ToyModel(ind, blind_additive_scale=1e-9), make_options(), do_blinding=True
    )
    f.defaultassign()
    f.set_blinding_offsets(True)
    var = _postfit_variances(f, ind)
    with caplog.at_level("WARNING"):
        f.warn_if_blinding_is_weak(var)
    assert any(MSG in r.message for r in caplog.records)

    # and an ample one
    caplog.clear()
    f2 = fitter.Fitter(
        ind, ToyModel(ind, blind_additive_scale=1e4), make_options(), do_blinding=True
    )
    f2.defaultassign()
    f2.set_blinding_offsets(True)
    var2 = _postfit_variances(f2, ind)
    with caplog.at_level("WARNING"):
        f2.warn_if_blinding_is_weak(var2)
    assert not any(MSG in r.message for r in caplog.records)


def test_the_weak_blinding_warning_does_not_leak_the_secret(path, caplog):
    """The warning must not print the numbers it is reasoning about.

    A message carrying the drawn offset hands over the secret outright; one
    carrying sigma hands it over too once the declared scale is known, since
    the two determine each other. Only the parameter name and the knob to turn
    are safe, and those are the actionable part anyway.
    """
    ind = inputdata.FitInputData(path)
    f = fitter.Fitter(
        ind, ToyModel(ind, blind_additive_scale=1e-9), make_options(), do_blinding=True
    )
    f.defaultassign()
    f.set_blinding_offsets(True)
    var = _postfit_variances(f, ind)

    with caplog.at_level("WARNING"):
        f.warn_if_blinding_is_weak(var)
    msgs = [r.message for r in caplog.records if MSG in r.message]
    assert msgs, "warning did not fire; test is vacuous"
    text = " ".join(msgs)

    offset = abs(float(f._blinding_values_poi_add[0]))
    sigma = float(np.sqrt(var[0]))
    assert offset > 0.0 and sigma > 0.0, "vacuous: nothing drawn or no curvature"

    for forbidden, label in ((offset, "the drawn offset"), (sigma, "sigma")):
        for fmt in (f"{forbidden:.4g}", f"{forbidden:.3g}", f"{forbidden:g}"):
            assert fmt not in text, f"warning leaked {label} as {fmt!r}"

    assert "alphaS" in text
    assert "blind_additive_scale" in text


def test_the_verdict_follows_the_measured_sigma_not_just_the_scale(path, caplog):
    """Same declared scale, two different uncertainties, opposite verdicts.

    This is what makes the MEASURED sigma the yardstick rather than a fixed
    threshold on the declared scale: a smearing that amply hides a precise
    measurement is useless for a loose one, and the criterion has to notice.
    Without this, replacing ``5 * sigma`` with a bare constant passes the
    suite -- it did.

    The covariance is handed in directly rather than fitted twice, which keeps
    the two cases differing in exactly one thing. Building two datasets with
    different statistics would make the test depend on the fit as well as on
    the criterion.
    """
    from rabbit.fitter import BLINDING_DRAW_STD

    ind = inputdata.FitInputData(path)
    f = fitter.Fitter(
        ind, ToyModel(ind, blind_additive_scale=1.0), make_options(), do_blinding=True
    )
    f.defaultassign()
    f.set_blinding_offsets(True)
    assert float(f._blinding_offsets_poi_add[0].numpy()) != 0.0, "vacuous: not blinded"

    n = int(f.x.shape[0])
    smearing = BLINDING_DRAW_STD * 1.0

    # sigma well below the smearing: amply hidden
    tight = np.full(n, (smearing / 500.0) ** 2)
    caplog.clear()
    with caplog.at_level("WARNING"):
        f.warn_if_blinding_is_weak(tight)
    assert not any(MSG in r.message for r in caplog.records)

    # sigma well above it: the same smearing now hides nothing
    loose = np.full(n, (smearing * 20.0) ** 2)
    caplog.clear()
    with caplog.at_level("WARNING"):
        f.warn_if_blinding_is_weak(loose)
    assert any(MSG in r.message for r in caplog.records)


def test_an_unblinded_poi_is_not_reported_as_weakly_blinded(path, caplog):
    """--unblind leaves the offset at zero, which is not weak blinding.

    Blinded-ness is read off the offsets rather than remembered, so a POI with
    no offset must be skipped rather than judged -- otherwise every unblinded
    run would warn that its unblinded POI is insufficiently hidden.
    """
    ind = inputdata.FitInputData(path)
    f = fitter.Fitter(
        ind,
        ToyModel(ind, blind_additive_scale=1e-9),
        make_options(unblind=["alphaS"]),
        do_blinding=True,
    )
    f.defaultassign()
    f.set_blinding_offsets(True)
    assert float(f._blinding_offsets_poi_add[0].numpy()) == 0.0
    var = _postfit_variances(f, ind)

    with caplog.at_level("WARNING"):
        f.warn_if_blinding_is_weak(var)
    assert not any(MSG in r.message for r in caplog.records)


# --- the scale must SURVIVE compositing -------------------------------------


def _composite_offsets(path, models):
    """Offsets the Fitter actually draws for a CompositeParamModel of `models`."""
    ind = inputdata.FitInputData(path)
    composite = CompositeParamModel([m(ind) for m in models])
    f = fitter.Fitter(ind, composite, make_options(), do_blinding=True)
    return composite, f._blinding_values_poi_add


def test_composite_preserves_a_declared_scale(path):
    """A submodel declaring a scale must not be quietly reset to the default.

    The Fitter reads the scale off its EFFECTIVE model, so if the composite
    drops the declaration the draw silently reverts to 1.0 -- in the
    under-blinding direction, and with no warning possible for the free POIs
    this feature exists for.
    """
    _, off_plain = _composite_offsets(path, [lambda i: ToyModel(i)])
    _, off_scaled = _composite_offsets(
        path,
        [lambda i: ToyModel(i, blind_additive_scale=7.0)],
    )
    assert off_plain[0] != 0.0
    assert np.isclose(off_scaled[0], 7.0 * off_plain[0], rtol=1e-12, atol=0)


def test_scale_is_per_poi_not_one_composite_value(path):
    """Two submodels with different scales must each keep their own.

    The scale is in each parameter's own units, so there is no correct single
    composite value -- reducing to one (max, first, any) would silently
    rescale somebody.
    """
    composite, off = _composite_offsets(
        path,
        [
            lambda i: ToyModel(i, blind_additive_scale=7.0),
            lambda i: SecondToyModel(i, blind_additive_scale=0.5),
        ],
    )
    np.testing.assert_allclose(composite.blind_additive_scale, [7.0, 0.5])

    # each offset is its own scale times the draw for its own NAME
    ind = inputdata.FitInputData(path)
    solo_a = fitter.Fitter(
        ind, ToyModel(ind), make_options(), do_blinding=True
    )._blinding_values_poi_add[0]
    solo_b = fitter.Fitter(
        ind, SecondToyModel(ind), make_options(), do_blinding=True
    )._blinding_values_poi_add[0]
    assert np.isclose(off[0], 7.0 * solo_a, rtol=1e-12, atol=0)
    assert np.isclose(off[1], 0.5 * solo_b, rtol=1e-12, atol=0)


def test_composite_of_composites_keeps_the_vector(path):
    """The propagated vector is itself a legal declaration, so nesting works."""
    ind = inputdata.FitInputData(path)
    inner = CompositeParamModel(
        [
            ToyModel(ind, blind_additive_scale=7.0),
            SecondToyModel(ind, blind_additive_scale=0.5),
        ]
    )
    outer = CompositeParamModel([inner])
    np.testing.assert_allclose(outer.blind_additive_scale, [7.0, 0.5])


def test_a_composite_of_poi_less_models_constructs(path):
    """The blinding scale must not be built when there is no POI block.

    Every model in the ABCD family is POI-less, and load_models composes
    straight from --paramModel, so composing two of them is a reachable
    configuration -- and np.concatenate raises on an empty list, which turned a
    supported composition into a constructor crash. There is nothing to scale
    here, so the attribute is simply absent and the Fitter falls back to 1.0.
    """
    ind = inputdata.FitInputData(path)
    composite = CompositeParamModel([PouOnlyModel(ind), PouOnlyModel(ind, b"nui2")])
    assert composite.npoi == 0
    assert not hasattr(composite, "blind_additive_scale")


def test_undeclared_submodels_get_one(path):
    """A submodel that declares nothing contributes 1.0 over its own slice."""
    composite, _ = _composite_offsets(
        path,
        [
            lambda i: ToyModel(i, blind_additive_scale=7.0),
            lambda i: SecondToyModel(i),
        ],
    )
    np.testing.assert_allclose(composite.blind_additive_scale, [7.0, 1.0])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
