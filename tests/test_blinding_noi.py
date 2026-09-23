"""A blinded NUISANCE OF INTEREST is offset the same way a POI is.

``self.x`` is initialised to ``x0default``, i.e. in the UNBLINDED frame, and
arming is NOT compensated: it assigns the offsets and leaves ``x`` alone, so an
armed fit opens every nuisance of interest at ``theta0default + offset`` rather
than at its declared centre. That is deliberate -- compensating would set ``x``
from ``x0default`` and the offsets, and ``x0default`` is public, so the offset
would be one subtraction away from any prefit coordinate that is written or
inspected. See ``set_blinding_offsets``.

For a ``--poiAsNoi`` analysis the offset nuisance IS the physics result, which
is why the theta block gets the same treatment as the POI block and is tested
as carefully.

THE FORM, FROM THE CODE. ``get_theta`` is

    theta_physical = theta_stored + add

with no transform in front of it, unlike ``get_poi``, whose squaring branch
applies after the offset. Only nuisances OF INTEREST carry a non-zero offset --
``Blinding`` loops over ``indata.noiidxs`` -- so an ordinary
constrained nuisance is untouched.

THE CONSTRAINT TERM IS THE THING TO GET RIGHT. ``_compute_lc`` penalises
``get_x() - self.x0``: both sides are in the MODEL frame, and ``x0`` is not
shifted. The armed run therefore opens with a non-zero prior penalty, measured
here rather than hidden, and the MINIMUM is unmoved -- which is what matters
and is checked, not assumed.

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
    the ParamModel's own nuisances in between; an offset vector that got the
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

    The ordinary one is the control: ``Blinding`` only offsets
    ``indata.noiidxs``, so an ordinary nuisance must never be offset.
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
        float(f.blinding.offsets_theta[_inoi(f)].numpy()), 0.0, rtol=0, atol=1e-9
    )


# --- the bug -----------------------------------------------------------------


@pytest.mark.parametrize("model_cls", [NoPoiModel, OnePoiModel])
def test_arming_does_not_move_the_coordinate(path, model_cls):
    """Arming assigns offsets and leaves ``x`` alone.

    The physical NOI therefore DOES move, by the drawn offset. That is the
    accepted cost: compensating it would write the offset into ``x``, where the
    public theta0default recovers it by subtraction.
    """
    f = build(path, model_cls)
    x_before = f.x.numpy().copy()
    theta_before = _theta(f)

    f.set_blinding_offsets(True)
    assert _blinded(f), "vacuous: this NOI was not blinded"

    np.testing.assert_array_equal(f.x.numpy(), x_before)
    assert not np.isclose(
        _theta(f), theta_before, rtol=1e-9
    ), "the physical NOI did not move, so nothing was actually armed"


def test_npoi_zero_is_covered(path):
    """The --poiAsNoi shape, where the parameter of interest is a nuisance.

    npoi = 0 means the POI block is empty and everything rides on the theta
    offsets, so it is the configuration most easily missed by code that reaches
    for the POI block first.
    """
    f = build(path, NoPoiModel)
    assert f.param_model.npoi == 0
    x_before = f.x.numpy().copy()
    f.set_blinding_offsets(True)
    f.xdefaultassign()
    assert _blinded(f)
    np.testing.assert_array_equal(f.x.numpy(), x_before)
    assert not np.isclose(_xtheta(f), _theta(f), rtol=0, atol=1e-9)


def test_theta_is_still_blinded(path):
    """Compensating the frame must not accidentally unblind the coordinate."""
    f = build(path, NoPoiModel)
    f.set_blinding_offsets(True)
    assert _blinded(f)
    assert not np.isclose(_xtheta(f), _theta(f), rtol=0, atol=1e-9)


def test_ordinary_nuisance_is_untouched(path):
    """Only nuisances of interest are offset; an ordinary one is untouched."""
    f = build(path, NoPoiModel)
    j = f.param_model.nparams + _iother(f)
    before = float(f.x[j].numpy())
    f.set_blinding_offsets(True)
    assert _blinded(f)
    assert float(f.x[j].numpy()) == before


@pytest.mark.parametrize("model_cls", [NoPoiModel, OnePoiModel])
def test_arming_is_idempotent(path, model_cls):
    """A second arm must change nothing, on x or on get_theta."""
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
def test_defaultassign_lands_on_x0default_whether_armed_or_not(path, model_cls):
    """The stored start must not depend on whether the offsets happen to be armed.

    The driver calls defaultassign() once per (pseudo)data set and the offsets
    are still armed from the previous one, so a stored start that differed
    between the two would expose the offset as a difference between sets.
    """
    f = build(path, model_cls)
    f.defaultassign()
    disarmed = f.x.numpy().copy()

    f.set_blinding_offsets(True)
    assert _blinded(f)
    f.defaultassign()
    np.testing.assert_array_equal(f.x.numpy(), disarmed)

    f.set_blinding_offsets(True)
    f.xdefaultassign()
    np.testing.assert_array_equal(f.x.numpy(), disarmed)


# --- the constraint term -----------------------------------------------------


def test_x0_untouched_by_arming(path):
    """x0 is the model frame and must NOT be shifted -- the constraint compares
    get_x() against it, so shifting both would double-count."""
    f = build(path, NoPoiModel)
    before = f.x0.numpy().copy()
    f.set_blinding_offsets(True)
    assert _blinded(f)
    np.testing.assert_array_equal(before, f.x0.numpy())


def test_arming_costs_prefit_penalty_but_does_not_move_the_minimum(path):
    """The price of the uncompensated start, stated rather than hidden.

    Armed, the run opens at theta0default while the CONSTRAINT still compares
    the physical theta0default + offset against x0, so it carries
    0.5*cw*offset**2 of prefit penalty -- on average 12.5 per blinded NOI for
    offset ~ N(0, 5). That is real: the prefit NLL of a blinded run is not the
    prefit NLL of an unblinded one, and with many blinded NOIs the minimiser
    starts from a correspondingly worse point.

    It is accepted because it does not move the answer -- see
    test_constrained_minimum_is_not_moved, which is the assertion that matters.
    Compensating it away would mean writing the offset into x.
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

    off = float(f.blinding.offsets_theta[_inoi(f)].numpy())
    assert np.isclose(lc_armed - lc_disarmed, 0.5 * off**2, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("constrained", [True, False])
def test_constrained_minimum_is_not_moved(path, path_unconstrained, constrained):
    """The fit must land on the same PHYSICAL NOI armed or disarmed.

    This is the real question about the constraint term: arming moves
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


def test_unblind_leaves_both_frames_equal(path):
    """--unblind on the NOI leaves its offset at zero."""
    f = build(path, NoPoiModel, unblind=["massShift"])
    x0 = f.x.numpy().copy()
    f.set_blinding_offsets(True)
    np.testing.assert_array_equal(f.x.numpy(), x0)
    assert np.isclose(_theta(f), THETA_START, rtol=0, atol=1e-14)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
