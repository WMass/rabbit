"""MULTIPLICATIVE POI blinding must also be a change of variables.

``self.x`` is initialised to ``xparamdefault``, i.e. in the UNBLINDED frame, so
arming the offsets moves the PHYSICAL point unless ``x`` is compensated. The
additive form is compensated (``set_blinding_offsets`` shifts by
``off_old - off_new``); the multiplicative form was not, so a blinded fit opened
at ``default * exp(offset)`` with ``offset ~ N(0, 5)`` instead of at the start
value the model declared. For a signal strength that is a slow start; for a POI
fed into a calculation with a restricted domain it is an evaluation error.

The compensation is NOT the offset ratio in general. ``get_poi`` is affine in the
model frame,

    poi = T(x) * mul + add,     T = identity, or square when
                                allowNegativeParam is False,

so holding ``poi`` fixed gives ``x -> x * (mul_old/mul_new)`` only for a POI that
is not squared. For a squared POI -- which is the DEFAULT, and what ``Mu`` uses --
it is ``x -> x * sqrt(mul_old/mul_new)``. Both are checked below, because getting
the square root wrong would still leave the start point orders of magnitude out
for every ``Mu`` analysis.

Every check is an INVARIANCE or an idempotence, so no test prints, returns or
asserts on an offset value.
"""

import os
import tempfile
from types import SimpleNamespace

import hist
import numpy as np
import pytest
import tensorflow as tf

from rabbit import fitter, inputdata, tensorwriter
from rabbit.param_models.param_model import ParamModel

# Deliberately neither 0 nor 1. A 1 would satisfy the squared-branch test by
# accident (sqrt(1) == 1), and a 0 would satisfy the multiplicative invariance
# test by accident (0 * off == 0) -- both are exactly the accidents this
# machinery replaces with a guarantee.
MU_START = 1.7


class SquaredPoiModel(ParamModel):
    """`Mu`-like: one POI the FITTER keeps positive by squaring the block.

    ``allowNegativeParam = False`` is the default everywhere in rabbit, so this
    is the common case, and the stored coordinate is ``sqrt(poi)``.
    """

    def __init__(self, indata):
        super().__init__(indata)
        self.npoi = 1
        self.npou = 0
        self.params = np.array([b"mu"])
        self.allowNegativeParam = False
        self.xparamdefault = tf.constant([np.sqrt(MU_START)], dtype=indata.dtype)
        self.is_linear = False

    def compute(self, param, full=False):
        col = tf.reshape(1.0 + 0.1 * param[0], [1, 1])
        return tf.concat(
            [col, tf.ones([1, self.indata.nproc - 1], dtype=col.dtype)], axis=1
        )


class PermissivePoiModel(ParamModel):
    """A POI passed through raw, still blinded MULTIPLICATIVELY.

    ``allowNegativeParam = True`` without ``blind_additive``: the combination
    that exercises the ratio form rather than the square-root form.
    """

    def __init__(self, indata):
        super().__init__(indata)
        self.npoi = 1
        self.npou = 0
        self.params = np.array([b"mu"])
        self.allowNegativeParam = True
        self.xparamdefault = tf.constant([MU_START], dtype=indata.dtype)
        self.is_linear = True

    def compute(self, param, full=False):
        col = tf.reshape(1.0 + 0.1 * param[0], [1, 1])
        return tf.concat(
            [col, tf.ones([1, self.indata.nproc - 1], dtype=col.dtype)], axis=1
        )


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
        p = os.path.join(d, "blinding_mult_tensor.hdf5")
        make_tensor(p)
        yield p


def build(path, model_cls, **opts):
    ind = inputdata.FitInputData(path)
    f = fitter.Fitter(ind, model_cls(ind), make_options(**opts), do_blinding=True)
    f.defaultassign()
    return f


def _poi(f):
    return float(f.get_poi()[0].numpy())


def _blinded(f):
    """Is the fitter actually offsetting this POI? Guards against vacuity."""
    return not np.isclose(
        float(f._blinding_offsets_poi[0].numpy()), 1.0, rtol=0, atol=1e-9
    )


# --- the bug: arming must not move the physical start point ------------------


@pytest.mark.parametrize("model_cls", [SquaredPoiModel, PermissivePoiModel])
def test_physical_start_invariant_under_arming(path, model_cls):
    """The fit must open where the model said, not at default * offset.

    Fails before the fix by a factor exp(N(0, 5)) -- up to many orders of
    magnitude, which for a POI feeding a calculation with a restricted domain is
    not a slow start but an evaluation error.
    """
    f = build(path, model_cls)
    assert np.isclose(_poi(f), MU_START, rtol=0, atol=1e-12)

    f.set_blinding_offsets(True)
    assert _blinded(f), "vacuous: this POI was not blinded"
    assert np.isclose(
        _poi(f), MU_START, rtol=1e-12, atol=0
    ), f"arming moved the physical POI from {MU_START} to {_poi(f)}"


@pytest.mark.parametrize("model_cls", [SquaredPoiModel, PermissivePoiModel])
def test_x_is_still_blinded(path, model_cls):
    """Compensating the frame must not accidentally unblind the coordinate.

    The whole point of blinding is that ``fitter.x`` -- what gets written out --
    is not the physical value.
    """
    f = build(path, model_cls)
    f.set_blinding_offsets(True)
    assert _blinded(f)
    assert not np.isclose(float(f.x[0].numpy()), _poi(f), rtol=0, atol=1e-9)


@pytest.mark.parametrize("model_cls", [SquaredPoiModel, PermissivePoiModel])
def test_arming_is_idempotent(path, model_cls):
    """A second arm must reframe by exactly 1, on x as well as on get_poi.

    This is the property the additive fix already relies on; the driver arms
    once per (pseudo)data set and the saturated path arms again on a copy.
    """
    f = build(path, model_cls)
    f.set_blinding_offsets(True)
    x_once = f.x.numpy().copy()
    poi_once = _poi(f)

    f.set_blinding_offsets(True)
    np.testing.assert_array_equal(f.x.numpy(), x_once)
    assert _poi(f) == poi_once


@pytest.mark.parametrize("model_cls", [SquaredPoiModel, PermissivePoiModel])
def test_disarming_returns_to_the_starting_coordinate(path, model_cls):
    """Arm then disarm is the identity on both frames.

    With the offsets disarmed the two frames coincide, so this also pins that
    the reframing is invertible rather than merely self-consistent.
    """
    f = build(path, model_cls)
    x0 = f.x.numpy().copy()
    f.set_blinding_offsets(True)
    assert _blinded(f)
    f.set_blinding_offsets(False)
    np.testing.assert_allclose(f.x.numpy(), x0, rtol=1e-12, atol=0)
    assert np.isclose(_poi(f), MU_START, rtol=1e-12, atol=0)


@pytest.mark.parametrize("model_cls", [SquaredPoiModel, PermissivePoiModel])
def test_defaultassign_lands_on_the_default_while_armed(path, model_cls):
    """xdefaultassign must reframe too, not only set_blinding_offsets.

    The driver calls defaultassign() once per (pseudo)data set, and from the
    second set onwards the offsets are still armed from the previous one. Without
    the reframing in xdefaultassign the physical start of every set after the
    first depends on whether it happens to run armed.
    """
    f = build(path, model_cls)
    f.set_blinding_offsets(True)
    assert _blinded(f)

    # the loop body in bin/rabbit_fit.py: defaultassign() while armed, then arm
    f.defaultassign()
    assert np.isclose(_poi(f), MU_START, rtol=1e-12, atol=0)
    f.set_blinding_offsets(True)
    assert np.isclose(_poi(f), MU_START, rtol=1e-12, atol=0)

    # xdefaultassign() on its own, armed (the toy-throw path does this)
    f.set_blinding_offsets(True)
    f.xdefaultassign()
    assert np.isclose(_poi(f), MU_START, rtol=1e-12, atol=0)


# --- the likelihood itself must be untouched ---------------------------------


@pytest.mark.parametrize("model_cls", [SquaredPoiModel, PermissivePoiModel])
def test_loss_is_the_same_armed_and_disarmed(path, model_cls):
    """Blinding is a reparametrisation: at the same PHYSICAL point the loss,
    and hence every fit result, must be identical.

    Without the compensation the armed fitter sits at a different physical point
    and this comparison is meaningless -- which is exactly why the start point
    mattered.
    """
    f_ref = build(path, model_cls)
    asimov = f_ref.expected_yield()

    f_ref.set_nobs(asimov)
    loss_disarmed = float(f_ref.reduced_nll().numpy())

    f = build(path, model_cls)
    f.set_blinding_offsets(True)
    # isolate the POI: theta starts at 0, so arming moves the physical NOIs to
    # their offsets and the yields for reasons unrelated to the POI under test
    f._blinding_offsets_theta.assign(np.zeros(f.indata.nsyst, dtype=np.float64))
    f.set_nobs(asimov)
    loss_armed = float(f.reduced_nll().numpy())

    assert np.isclose(loss_armed, loss_disarmed, rtol=0, atol=1e-9)


def test_squared_branch_uses_the_square_root_of_the_ratio(path):
    """Pin the algebra, not just its consequence.

    For a squared POI, poi = x**2 * mul, so the frame-preserving factor on x is
    sqrt(mul_old/mul_new). Using the ratio itself -- the correct factor for a POI
    that is NOT squared -- would leave the physical start off by sqrt(offset),
    which for offset = exp(N(0, 5)) is still orders of magnitude. Read the
    factor off the coordinate rather than asserting on the offset value.
    """
    f = build(path, SquaredPoiModel)
    x_before = float(f.x[0].numpy())
    f.set_blinding_offsets(True)
    assert _blinded(f)
    x_after = float(f.x[0].numpy())
    mul = float(f._blinding_offsets_poi[0].numpy())

    factor = x_after / x_before
    assert np.isclose(factor, 1.0 / np.sqrt(mul), rtol=1e-12, atol=0)
    # and NOT the un-rooted ratio (guard against a plausible-looking wrong fix)
    assert not np.isclose(factor, 1.0 / mul, rtol=1e-6, atol=0)


def test_unblind_leaves_both_frames_equal(path):
    """--unblind on the POI makes the reframing the identity."""
    f = build(path, SquaredPoiModel, unblind=["mu"])
    x0 = f.x.numpy().copy()
    f.set_blinding_offsets(True)
    np.testing.assert_array_equal(f.x.numpy(), x0)
    assert np.isclose(_poi(f), MU_START, rtol=0, atol=1e-14)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
