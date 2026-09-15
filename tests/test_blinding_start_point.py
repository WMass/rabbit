"""A blinded fit starts somewhere else and lands in the same place.

Blinding is deliberately NOT compensated: arming the offsets leaves ``self.x``
untouched, so a blinded run opens at a different physical point than an
unblinded one. Two things must hold for that to be acceptable, and this file
pins both.

**The offset must not be recoverable from the fit's own state.** This is why
the compensation was removed. Compensating would set ``x`` to ``x0default``
mapped through the offsets, and ``x0default`` is public -- it is the default
the model declares -- so one subtraction would recover the secret from any
prefit coordinate that gets written, logged or inspected.

**The minimiser must reach the same physical minimum from the shifted start.**
Otherwise blinding would change the result rather than hide it. That is an
assumption about the minimiser, so it is tested rather than asserted.
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


def _fit(path, do_blinding):
    """Minimise once, armed or not, and report the PHYSICAL answer."""
    ind, _, f = build(path, do_blinding)
    f.defaultassign()
    if do_blinding:
        f.set_blinding_offsets(True)
    f.set_nobs(ind.data_obs)
    f.minimize()
    loss, _, hess = f.loss_val_grad_hess()
    return dict(
        x=f.x.numpy().copy(),
        physical=f.get_x().numpy().copy(),
        loss=float(loss.numpy()),
        hess=hess.numpy().copy(),
    )


def test_arming_does_not_touch_x(path):
    """The whole point: arming assigns offsets and nothing else.

    If this ever starts failing because a compensation was reintroduced, read
    test_offset_is_not_recoverable_from_the_prefit_coordinate below for why it
    must not be.
    """
    _, _, f = build(path, True)
    f.defaultassign()
    before = f.x.numpy().copy()
    f.set_blinding_offsets(True)
    after = f.x.numpy().copy()
    np.testing.assert_array_equal(before, after)

    # and disarming is equally inert
    f.set_blinding_offsets(False)
    np.testing.assert_array_equal(before, f.x.numpy())


def test_offset_is_not_recoverable_from_the_prefit_coordinate(path):
    """Stated as the attack it prevents.

    ``x0default`` is public, so an attacker who can read the prefit ``x``
    computes ``x0default - x``. That must carry no information about the
    offset, i.e. it must be identically zero -- while the offset itself is
    non-zero, or the test proves nothing.
    """
    _, _, f = build(path, True)
    f.defaultassign()
    f.set_blinding_offsets(True)

    npoi = f.param_model.npoi
    armed = f._blinding_offsets_poi_add.numpy()[:npoi]
    assert not np.allclose(armed, 0.0), "offset is the identity; test is vacuous"

    leak = f.x0default.numpy()[:npoi] - f.x.numpy()[:npoi]
    np.testing.assert_array_equal(leak, np.zeros_like(leak))


def test_physical_minimum_is_the_same_armed_and_disarmed(path):
    """The assumption the uncompensated start rests on.

    The blinded fit opens at a different physical point, so this is the claim
    that makes that harmless: the minimiser converges to the same physical
    minimum either way.

    Judged against the fit's OWN uncertainty rather than an absolute tolerance,
    which would be a property of whichever offset the seed happens to draw. A
    milli-sigma is the statement that matters: blinding does not move the answer
    by anything that could be mistaken for a result.
    """
    armed = _fit(path, True)
    plain = _fit(path, False)

    assert not np.allclose(
        armed["x"], plain["x"], rtol=1e-6
    ), "armed and disarmed fits used the same coordinates; test is vacuous"

    sigma = np.sqrt(np.diag(np.linalg.inv(plain["hess"])))
    assert np.all(sigma > 0)
    shift = np.abs(armed["physical"] - plain["physical"]) / sigma
    assert np.all(shift < 1e-3), (
        f"blinding moved the minimum by up to {shift.max():.2e} sigma, which is "
        "too much to call the same minimum"
    )
    assert np.isclose(armed["loss"], plain["loss"], rtol=0, atol=1e-6)


def test_blinding_leaves_the_postfit_hessian_unblinded(path):
    """sigma = sqrt(diag(H^-1)), so 'sigma unblinded' IS 'H unchanged'.

    The additive form is a translation, so its Jacobian is the identity and the
    whole Hessian -- not just the POI row -- is untouched. Checked AT THE
    MINIMUM rather than at a shared coordinate: without compensation the two
    runs pass through different points on the way, and it is the answer that has
    to match, not the path.
    """
    armed = _fit(path, True)
    plain = _fit(path, False)

    assert abs(plain["hess"][0, 0]) > 1e-6, f"no POI curvature: {plain['hess'][0, 0]}"
    np.testing.assert_allclose(armed["hess"], plain["hess"], rtol=1e-6, atol=0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
