"""
Test that blinding is a change of variables that leaves the PHYSICS alone.

rabbit blinds by reparametrising the likelihood, not by masking output: the
getters apply the offset on the way INTO ParamModel.compute() and the constraint
term, so the model and the NLL see the physical value while ``fitter.x`` -- the
minimizer's coordinate, and what gets written out -- is the blinded one.

The two forms are not interchangeable, which is what these tests pin down:

* MULTIPLICATIVE (the default for POIs) suits a signal strength centred at 1
  that scales yields. But the reported coordinate is then ``poi_true / offset``,
  so the curvature scales as ``offset**2`` and the reported uncertainty, the POI
  row of the covariance and every impact on that POI are divided by the random
  factor. Only the RELATIVE uncertainty survives. (That Jacobian is intended and
  is unchanged; what the start point does is a separate question, covered in
  tests/test_blinding_multiplicative.py.)
* ADDITIVE, which a model opts into with ``blind_additive = True``, is a
  translation. Its Jacobian is the identity, so the covariance, the
  uncertainties and the impacts come out EXACTLY unblinded while the central
  value is still hidden.

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

# Deliberately NON-ZERO. A zero default would satisfy the start-invariance test
# by accident -- ``0 * offset == 0`` for the multiplicative form -- which is
# exactly the accident this machinery replaces with a guarantee.
START = 0.3


class ToyModel(ParamModel):
    """One POI scaling the signal, linear so the fit solves exactly."""

    def __init__(
        self, indata, blind_additive=False, blind_additive_scale=None, prior_sigma=None
    ):
        super().__init__(indata)
        self.npoi = 1
        self.npou = 0
        self.params = np.array([b"alphaS"])
        self.xparamdefault = tf.constant([START], dtype=indata.dtype)
        self.is_linear = True
        self.allowNegativeParam = True
        if blind_additive:
            self.blind_additive = True
        if blind_additive_scale is not None:
            self.blind_additive_scale = blind_additive_scale
        if prior_sigma is not None:
            self.prior_sigmas = np.array([prior_sigma], dtype=np.float64)
            self.prior_means = np.array([START], dtype=np.float64)

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


class PouOnlyAdditiveModel(ParamModel):
    """Declares blind_additive but carries no POIs, so it gets no vote."""

    def __init__(self, indata):
        super().__init__(indata)
        self.npoi = 0
        self.npou = 1
        self.params = np.array([b"nuisance"])
        self.xparamdefault = tf.constant([0.0], dtype=indata.dtype)
        self.is_linear = True
        self.allowNegativeParam = True
        self.blind_additive = True

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


def build(path, blind_additive, do_blinding, **opts):
    ind = inputdata.FitInputData(path)
    model = ToyModel(ind, blind_additive=blind_additive)
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


def test_model_sees_physical_while_x_is_blinded(path):
    """compute() gets the physical value; fitter.x holds the blinded one."""
    _, _, fu = build(path, True, False)
    fu.defaultassign()
    y_unblinded = fu.expected_yield().numpy()

    _, model, fb = build(path, True, True)
    fb.defaultassign()
    fb.set_blinding_offsets(True)
    # Zero the THETA offsets to isolate the POI: theta starts at 0, so arming
    # puts the physical NOIs at their offsets and moves the yields for reasons
    # that have nothing to do with the POI under test.
    fb._blinding_offsets_theta.assign(np.zeros(fb.indata.nsyst, dtype=np.float64))
    y_blinded = fb.expected_yield().numpy()

    assert np.isclose(float(fb.get_poi()[0].numpy()), START, rtol=0, atol=1e-12)
    assert np.allclose(y_blinded, y_unblinded, rtol=1e-12, atol=0)
    assert not np.isclose(float(fb.x[0].numpy()), START, rtol=0, atol=1e-9)


def test_additive_leaves_hessian_exactly_unblinded(path):
    """sigma = sqrt(diag(H^-1)), so 'sigma unblinded' IS 'H unchanged'."""

    def loss_and_hess(blind_additive, do_blinding, asimov):
        _, _, f = build(path, blind_additive, do_blinding)
        f.defaultassign()
        if do_blinding:
            f.set_blinding_offsets(True)
            f._blinding_offsets_theta.assign(np.zeros(f.indata.nsyst, dtype=np.float64))
        f.set_nobs(asimov)
        loss, _, hess = f.loss_val_grad_hess()
        return float(loss.numpy()), hess.numpy()

    _, _, f_ref = build(path, True, False)
    f_ref.defaultassign()
    asimov = _asimov(f_ref)

    l_u, h_u = loss_and_hess(True, False, asimov)
    l_a, h_a = loss_and_hess(True, True, asimov)

    # Guard against a vacuous pass before comparing.
    assert abs(h_u[0, 0]) > 1e-6, f"no POI curvature to compare: {h_u[0, 0]}"
    assert np.isclose(l_a, l_u, rtol=0, atol=1e-9)
    assert np.isclose(h_a[0, 0], h_u[0, 0], rtol=1e-10)
    assert np.allclose(h_a, h_u, rtol=1e-10, atol=0)

    # The defect being fixed: the multiplicative path scales the same element.
    _, h_m = loss_and_hess(False, True, asimov)
    assert not np.isclose(h_m[0, 0], h_u[0, 0], rtol=1e-6)


def test_physical_start_invariant_under_arming(path):
    """Arming must not move the physical point, and must be idempotent."""
    _, _, f = build(path, True, True)
    f.defaultassign()
    assert np.isclose(float(f.get_poi()[0].numpy()), START, rtol=0, atol=1e-14)

    f.set_blinding_offsets(True)
    p_armed = float(f.get_poi()[0].numpy())
    assert np.isclose(p_armed, START, rtol=0, atol=1e-12)
    assert not np.isclose(float(f.x[0].numpy()), p_armed, rtol=0, atol=1e-9)

    f.set_blinding_offsets(True)  # idempotent: shifts by zero
    assert np.isclose(float(f.get_poi()[0].numpy()), p_armed, rtol=0, atol=1e-14)

    f.set_blinding_offsets(False)
    assert np.isclose(float(f.get_poi()[0].numpy()), START, rtol=0, atol=1e-12)


def test_x0_untouched_by_arming(path):
    """x0 is the model frame and must NOT be shifted; cheapest guard against
    someone 'symmetrising' the compensation later."""
    _, _, f = build(path, True, True)
    f.defaultassign()
    before = f.x0.numpy().copy()
    f.set_blinding_offsets(True)
    assert np.array_equal(before, f.x0.numpy())


def test_multiplicative_path_keeps_its_arithmetic_but_is_reframed(path):
    """A model that does not opt into ADDITIVE blinding keeps exactly its
    ``get_poi`` arithmetic -- ``poi = x * offset``, no additive slot -- but ``x``
    is now reframed by the offset RATIO so that the physical start is preserved
    like the additive case.

    The two halves are separable and both matter: the Jacobian of the
    multiplicative form is intended and unchanged (only the RELATIVE
    uncertainty survives it), while the START POINT is not part of that
    bargain -- an uncompensated frame would open the fit at ``START * offset``
    with ``offset = exp(N(0, 5))``. tests/test_blinding_multiplicative.py
    carries the full set of start-point invariances.
    """
    _, _, f = build(path, False, True)
    f.defaultassign()
    f.set_blinding_offsets(True)
    assert float(f._blinding_offsets_poi_add[0].numpy()) == 0.0
    # the arithmetic of get_poi is untouched
    assert np.isclose(
        float(f.get_poi()[0].numpy()),
        float(f.x[0].numpy()) * float(f._blinding_offsets_poi[0].numpy()),
        rtol=1e-14,
    )
    # x IS frame-shifted now, and that is what keeps the physical start at START
    assert not np.isclose(float(f.x[0].numpy()), START, rtol=0, atol=1e-9)
    assert np.isclose(float(f.get_poi()[0].numpy()), START, rtol=1e-14, atol=0)


def test_determinism_across_fitters(path):
    """Same input, independent fitters: identical offsets, without printing one."""
    _, _, f1 = build(path, True, True)
    _, _, f2 = build(path, True, True)
    for f in (f1, f2):
        f.defaultassign()
        f.set_blinding_offsets(True)
    d1 = f1.get_poi().numpy() - f1.x[:1].numpy()
    d2 = f2.get_poi().numpy() - f2.x[:1].numpy()
    np.testing.assert_array_equal(d1, d2)
    assert not np.allclose(d1, 0.0)  # non-trivial


def test_unblind_disables_the_offset(path):
    """--unblind on the POI leaves the coordinate and the physical value equal."""
    _, _, f = build(path, True, True, unblind=["alphaS"])
    f.defaultassign()
    f.set_blinding_offsets(True)
    assert np.isclose(
        float(f.get_poi()[0].numpy()), float(f.x[0].numpy()), rtol=0, atol=1e-14
    )


def test_additive_requires_allow_negative_param(path):
    """With the squared storage an additive offset could hand compute() a
    negative POI, so the combination must be refused."""
    ind = inputdata.FitInputData(path)
    model = ToyModel(ind, blind_additive=True)
    model.allowNegativeParam = False
    try:
        fitter.Fitter(ind, model, make_options(), do_blinding=True)
    except ValueError as exc:
        assert "allowNegativeParam" in str(exc)
    else:
        raise AssertionError("expected a refusal for blind_additive + squared storage")


# --- the additive draw's SCALE, which the multiplicative form does not need ---


def test_additive_scale_multiplies_the_draw(path):
    """``blind_additive_scale`` rescales the offset and nothing else.

    exp(N(0, 5)) hides a POI by orders of magnitude whatever it means, but
    + N(0, 5) is an ABSOLUTE shift, so a POI whose sigma is O(1) in its own
    units would be offset by well under a sigma. The model declares its units
    here; the draw itself (seed, sign, magnitude) must be untouched, which is
    what pins this as a rescale rather than a different random number.
    """
    ind = inputdata.FitInputData(path)
    base = fitter.Fitter(
        ind, ToyModel(ind, blind_additive=True), make_options(), do_blinding=True
    )
    scaled = fitter.Fitter(
        ind,
        ToyModel(ind, blind_additive=True, blind_additive_scale=1000.0),
        make_options(),
        do_blinding=True,
    )
    off_base = base._blinding_values_poi_add[0]
    off_scaled = scaled._blinding_values_poi_add[0]

    assert off_base != 0.0, "vacuous: the unscaled draw is already zero"
    assert np.isclose(off_scaled, 1000.0 * off_base, rtol=1e-12, atol=0)
    # the multiplicative vector stays the identity in both
    assert np.allclose(base._blinding_values_poi, 1.0)
    assert np.allclose(scaled._blinding_values_poi, 1.0)


def test_default_scale_is_the_historical_draw(path):
    """Not declaring a scale must reproduce the pre-existing offset exactly."""
    ind = inputdata.FitInputData(path)
    implicit = fitter.Fitter(
        ind, ToyModel(ind, blind_additive=True), make_options(), do_blinding=True
    )
    explicit = fitter.Fitter(
        ind,
        ToyModel(ind, blind_additive=True, blind_additive_scale=1.0),
        make_options(),
        do_blinding=True,
    )
    assert implicit._blinding_values_poi_add[0] == explicit._blinding_values_poi_add[0]


def test_weak_additive_blinding_is_reported(path, caplog):
    """An offset small against the prefit sigma must not fail silently.

    The prefit sigma exists for a POI only where the model declared a prior on
    it, which is the only case the Fitter can judge. A prior width far larger
    than the drawn offset is the "POI in awkward units" case; a tight one is
    the alpha_s-like case that is fine.
    """
    ind = inputdata.FitInputData(path)

    with caplog.at_level("WARNING"):
        fitter.Fitter(
            ind,
            ToyModel(ind, blind_additive=True, prior_sigma=1e6),
            make_options(),
            do_blinding=True,
        )
    assert any(
        "Additive blinding may be INEFFECTIVE" in r.message for r in caplog.records
    )

    caplog.clear()
    with caplog.at_level("WARNING"):
        fitter.Fitter(
            ind,
            ToyModel(ind, blind_additive=True, prior_sigma=1e-4),
            make_options(),
            do_blinding=True,
        )
    assert not any(
        "Additive blinding may be INEFFECTIVE" in r.message for r in caplog.records
    )


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
    _, off_plain = _composite_offsets(
        path, [lambda i: ToyModel(i, blind_additive=True)]
    )
    _, off_scaled = _composite_offsets(
        path,
        [lambda i: ToyModel(i, blind_additive=True, blind_additive_scale=7.0)],
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
            lambda i: ToyModel(i, blind_additive=True, blind_additive_scale=7.0),
            lambda i: SecondToyModel(i, blind_additive=True, blind_additive_scale=0.5),
        ],
    )
    np.testing.assert_allclose(composite.blind_additive_scale, [7.0, 0.5])

    # each offset is its own scale times the draw for its own NAME
    ind = inputdata.FitInputData(path)
    solo_a = fitter.Fitter(
        ind, ToyModel(ind, blind_additive=True), make_options(), do_blinding=True
    )._blinding_values_poi_add[0]
    solo_b = fitter.Fitter(
        ind, SecondToyModel(ind, blind_additive=True), make_options(), do_blinding=True
    )._blinding_values_poi_add[0]
    assert np.isclose(off[0], 7.0 * solo_a, rtol=1e-12, atol=0)
    assert np.isclose(off[1], 0.5 * solo_b, rtol=1e-12, atol=0)


def test_composite_of_composites_keeps_the_vector(path):
    """The propagated vector is itself a legal declaration, so nesting works."""
    ind = inputdata.FitInputData(path)
    inner = CompositeParamModel(
        [
            ToyModel(ind, blind_additive=True, blind_additive_scale=7.0),
            SecondToyModel(ind, blind_additive=True, blind_additive_scale=0.5),
        ]
    )
    outer = CompositeParamModel([inner])
    np.testing.assert_allclose(outer.blind_additive_scale, [7.0, 0.5])


def test_undeclared_submodels_get_one(path):
    """A submodel that declares nothing contributes 1.0 over its own slice."""
    composite, _ = _composite_offsets(
        path,
        [
            lambda i: ToyModel(i, blind_additive=True, blind_additive_scale=7.0),
            lambda i: SecondToyModel(i),
        ],
    )
    np.testing.assert_allclose(composite.blind_additive_scale, [7.0, 1.0])


def test_a_poi_less_submodel_does_not_flip_the_form(path):
    """The form governs the POI block, so only POI-carrying submodels vote."""
    ind = inputdata.FitInputData(path)
    composite = CompositeParamModel([ToyModel(ind), PouOnlyAdditiveModel(ind)])
    assert not getattr(composite, "blind_additive", False)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
