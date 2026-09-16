"""A blinded NUISANCE OF INTEREST must also keep its declared start point.

Third instance of the same start-point bug. ``self.x`` is initialised to
``x0default``, i.e. in the UNBLINDED frame, so arming the offsets moves the
PHYSICAL point unless ``x`` is compensated. The additive POI slots are
compensated; the multiplicative POI slots were not (see
tests/test_blinding_multiplicative.py) and neither was the THETA block, so a
blinded fit opened every nuisance of interest at ``theta0default + N(0, 5)``
instead of at its declared centre. For a ``--poiAsNoi`` analysis that parameter
is the physics result itself.

THE FORMULA, DERIVED FROM THE CODE. ``get_theta`` is

    theta_physical = theta_stored + add

with NO transform in front of it -- unlike ``get_poi``, whose squaring branch is
what forces a square root there -- and no multiplicative offset exists for
nuisances at all. So the frame-preserving update is the plain additive shift

    theta -> theta + (add_old - add_new),

the same form the additive POI slots already used. No square root, no ratio.

THE CONSTRAINT TERM IS THE THING TO GET RIGHT. ``_compute_lc`` penalises
``get_x() - self.x0``: both sides are in the MODEL frame, and ``x0`` must not be
shifted. Holding ``theta_physical`` fixed therefore leaves the penalty's value
AND its minimum exactly where they were -- which is checked here for a
constrained NOI, not assumed.

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

# Deliberately NON-ZERO, via --setConstraintMinimum. theta0default is 0 by
# default, and 0 is the one value a sign error in the shift cannot be
# distinguished from by looking at the constraint alone.
THETA_START = 0.7


class NoPoiModel(ParamModel):
    """npoi = 0: the ``--poiAsNoi`` shape, where the POI block cannot help.

    This is also the case the old ``xdefaultassign`` guard
    (``do_blinding and npoi``) skipped entirely.
    """

    def __init__(self, indata):
        super().__init__(indata)
        self.npoi = 0
        self.npou = 0
        self.params = np.array([])
        self.allowNegativeParam = False
        self.xparamdefault = tf.zeros([0], dtype=indata.dtype)
        self.is_linear = True

    def compute(self, param, full=False):
        return tf.ones([1, self.indata.nproc], dtype=self.indata.dtype)


class OnePoiModel(NoPoiModel):
    """A POI alongside the NOI, so the two blocks cannot be confused.

    The POI block sits at [0, npoi) and the theta block at [nparams, ...), with
    the ParamModel's own nuisances in between; a reframe that got the offsets
    crossed would show up here and not in the npoi = 0 case.
    """

    def __init__(self, indata):
        super().__init__(indata)
        self.npoi = 1
        self.npou = 1
        self.params = np.array([b"mu", b"modelNui"])
        self.allowNegativeParam = True
        self.xparamdefault = tf.constant([1.3, 0.4], dtype=indata.dtype)
        self.is_linear = False

    def compute(self, param, full=False):
        col = tf.reshape(1.0 + 0.1 * param[0] + 0.01 * param[1], [1, 1])
        return tf.concat(
            [col, tf.ones([1, self.indata.nproc - 1], dtype=col.dtype)], axis=1
        )


def make_tensor(path, constrained=True):
    """One NOI (constrained or not) plus one ordinary constrained nuisance.

    The ordinary one is the control: ``init_blinding_values`` only offsets
    ``indata.noiidxs``, so it must come out of every reframe untouched.
    """
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
    w.add_norm_systematic(
        "massShift", ["sig"], "ch0", 1.01, noi=True, constrained=constrained
    )
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
        setConstraintMinimum=[("massShift", THETA_START)],
        unblind=[],
        blindingGroup=[],
        maxRestarts=-1,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


@pytest.fixture(scope="module")
def path():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "blinding_noi_tensor.hdf5")
        make_tensor(p, constrained=True)
        yield p


@pytest.fixture(scope="module")
def path_unconstrained():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "blinding_noi_free_tensor.hdf5")
        make_tensor(p, constrained=False)
        yield p


def build(path, model_cls=NoPoiModel, **opts):
    ind = inputdata.FitInputData(path)
    f = fitter.Fitter(ind, model_cls(ind), make_options(**opts), do_blinding=True)
    f.defaultassign()
    return f


def _inoi(f):
    return int(f.indata.noiidxs[0])


def _iother(f):
    """Index of the ordinary (non-NOI) nuisance, the untouched-control."""
    (other,) = [
        i for i in range(f.indata.nsyst) if i not in set(np.asarray(f.indata.noiidxs))
    ]
    return other


def _theta(f):
    return float(f.get_theta()[_inoi(f)].numpy())


def _xtheta(f):
    return float(f.x[f.param_model.nparams + _inoi(f)].numpy())


def _blinded(f):
    """Is this NOI actually offset? Guards every check below against vacuity."""
    return not np.isclose(
        float(f._blinding_offsets_theta[_inoi(f)].numpy()), 0.0, rtol=0, atol=1e-9
    )


# --- the bug -----------------------------------------------------------------


@pytest.mark.parametrize("model_cls", [NoPoiModel, OnePoiModel])
def test_physical_start_invariant_under_arming(path, model_cls):
    """The fit must open where the card said, not at theta0default + offset.

    Fails before the fix by the offset itself, drawn from N(0, 5) -- up to ~15
    prior widths for a constrained NOI, and for --poiAsNoi that is the physics
    parameter.
    """
    f = build(path, model_cls)
    assert np.isclose(_theta(f), THETA_START, rtol=0, atol=1e-12)

    f.set_blinding_offsets(True)
    assert _blinded(f), "vacuous: this NOI was not blinded"
    assert np.isclose(
        _theta(f), THETA_START, rtol=1e-12, atol=0
    ), f"arming moved the physical NOI from {THETA_START} to {_theta(f)}"


def test_npoi_zero_is_covered(path):
    """The old xdefaultassign guard was `do_blinding and npoi`, so a model with
    npoi = 0 -- the --poiAsNoi shape -- skipped the reframe entirely."""
    f = build(path, NoPoiModel)
    assert f.param_model.npoi == 0
    f.set_blinding_offsets(True)
    f.xdefaultassign()
    assert _blinded(f)
    assert np.isclose(_theta(f), THETA_START, rtol=1e-12, atol=0)


def test_theta_is_still_blinded(path):
    """Compensating the frame must not accidentally unblind the coordinate."""
    f = build(path, NoPoiModel)
    f.set_blinding_offsets(True)
    assert _blinded(f)
    assert not np.isclose(_xtheta(f), _theta(f), rtol=0, atol=1e-9)


def test_ordinary_nuisance_is_untouched(path):
    """Only nuisances of interest are offset, so only they may be reframed."""
    f = build(path, NoPoiModel)
    j = f.param_model.nparams + _iother(f)
    before = float(f.x[j].numpy())
    f.set_blinding_offsets(True)
    assert _blinded(f)
    assert float(f.x[j].numpy()) == before


@pytest.mark.parametrize("model_cls", [NoPoiModel, OnePoiModel])
def test_arming_is_idempotent(path, model_cls):
    """A second arm must reframe by exactly 0, on x as well as on get_theta."""
    f = build(path, model_cls)
    f.set_blinding_offsets(True)
    x_once = f.x.numpy().copy()
    theta_once = _theta(f)

    f.set_blinding_offsets(True)
    np.testing.assert_array_equal(f.x.numpy(), x_once)
    assert _theta(f) == theta_once


@pytest.mark.parametrize("model_cls", [NoPoiModel, OnePoiModel])
def test_disarming_returns_to_the_starting_coordinate(path, model_cls):
    """Arm then disarm is the identity on both frames."""
    f = build(path, model_cls)
    x0 = f.x.numpy().copy()
    f.set_blinding_offsets(True)
    assert _blinded(f)
    f.set_blinding_offsets(False)
    np.testing.assert_allclose(f.x.numpy(), x0, rtol=1e-12, atol=0)
    assert np.isclose(_theta(f), THETA_START, rtol=1e-12, atol=0)


@pytest.mark.parametrize("model_cls", [NoPoiModel, OnePoiModel])
def test_defaultassign_lands_on_the_default_while_armed(path, model_cls):
    """xdefaultassign must reframe the theta block too.

    The driver calls defaultassign() once per (pseudo)data set and from the
    second set onwards the offsets are still armed from the previous one; the
    toy-throw path calls xdefaultassign() armed for the same reason.
    """
    f = build(path, model_cls)
    f.set_blinding_offsets(True)
    assert _blinded(f)

    f.defaultassign()
    assert np.isclose(_theta(f), THETA_START, rtol=1e-12, atol=0)
    f.set_blinding_offsets(True)
    assert np.isclose(_theta(f), THETA_START, rtol=1e-12, atol=0)

    f.set_blinding_offsets(True)
    f.xdefaultassign()
    assert np.isclose(_theta(f), THETA_START, rtol=1e-12, atol=0)


# --- the constraint term -----------------------------------------------------


def test_x0_untouched_by_arming(path):
    """x0 is the model frame and must NOT be shifted -- the constraint compares
    get_x() against it, so shifting both would double-count."""
    f = build(path, NoPoiModel)
    before = f.x0.numpy().copy()
    f.set_blinding_offsets(True)
    assert _blinded(f)
    np.testing.assert_array_equal(before, f.x0.numpy())


def test_constraint_penalty_at_the_declared_start_is_zero(path):
    """A constrained NOI opening at its own constraint centre costs nothing.

    Before the fix the armed fit opened at theta0default + offset and carried
    0.5*offset**2 of prefit penalty -- on average 12.5 for offset ~ N(0, 5) --
    which is pure noise added to the prefit NLL of every blinded analysis.
    Compare armed against disarmed rather than asserting an absolute value, so
    the check does not depend on the rest of the likelihood.
    """
    f_ref = build(path, NoPoiModel)
    asimov = f_ref.expected_yield()
    f_ref.set_nobs(asimov)
    lc_disarmed = float(f_ref._compute_lc().numpy())

    f = build(path, NoPoiModel)
    f.set_blinding_offsets(True)
    assert _blinded(f)
    f.set_nobs(asimov)
    lc_armed = float(f._compute_lc().numpy())

    assert np.isclose(lc_armed, lc_disarmed, rtol=0, atol=1e-9)


@pytest.mark.parametrize("constrained", [True, False])
def test_constrained_minimum_is_not_moved(path, path_unconstrained, constrained):
    """The fit must land on the same PHYSICAL NOI armed or disarmed.

    This is the real question about the constraint term: the reframing may move
    the coordinate but must not move where the likelihood-plus-prior is
    minimised. Run it for a constrained NOI and for a free one, since only the
    former has a prior to move.
    """
    p = path if constrained else path_unconstrained

    f_ref = build(p, NoPoiModel)
    asimov = f_ref.expected_yield()
    f_ref.set_nobs(asimov)
    f_ref.minimize()
    theta_disarmed = _theta(f_ref)

    f = build(p, NoPoiModel)
    f.set_blinding_offsets(True)
    assert _blinded(f)
    f.set_nobs(asimov)
    f.minimize()

    assert np.isclose(_theta(f), theta_disarmed, rtol=0, atol=1e-6)


def test_loss_is_the_same_armed_and_disarmed(path):
    """Blinding is a reparametrisation: at the same PHYSICAL point the loss, and
    hence every fit result, must be identical."""
    f_ref = build(path, NoPoiModel)
    asimov = f_ref.expected_yield()
    f_ref.set_nobs(asimov)
    loss_disarmed = float(f_ref.reduced_nll().numpy())

    f = build(path, NoPoiModel)
    f.set_blinding_offsets(True)
    assert _blinded(f)
    f.set_nobs(asimov)
    loss_armed = float(f.reduced_nll().numpy())

    assert np.isclose(loss_armed, loss_disarmed, rtol=0, atol=1e-9)


def test_shift_is_additive_not_scaled(path):
    """Pin the algebra, not just its consequence.

    get_theta has no transform in front of it, so the frame-preserving update is
    a plain shift. Read the update off the coordinate and check it is a
    difference and not a ratio -- the mistake that a squared POI does require.
    """
    f = build(path, NoPoiModel)
    x_before = _xtheta(f)
    f.set_blinding_offsets(True)
    assert _blinded(f)
    x_after = _xtheta(f)
    off = float(f._blinding_offsets_theta[_inoi(f)].numpy())

    assert np.isclose(x_after, x_before - off, rtol=0, atol=1e-14)
    # the physical value is the stored one plus the offset, unchanged definition
    assert np.isclose(_theta(f), x_after + off, rtol=0, atol=1e-14)


def test_unblind_leaves_both_frames_equal(path):
    """--unblind on the NOI makes the reframing the identity."""
    f = build(path, NoPoiModel, unblind=["massShift"])
    x0 = f.x.numpy().copy()
    f.set_blinding_offsets(True)
    np.testing.assert_array_equal(f.x.numpy(), x0)
    assert np.isclose(_theta(f), THETA_START, rtol=0, atol=1e-14)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
