"""--hvpBatch on a single device: the dense Hessian from batched HVPs.

By default the single-device fit takes its dense Hessian with one
tape.jacobian, whose memory scales with the parameter count. An explicit
--hvpBatch assembles it from HVPs instead, as the multi-device fit does. Both
must give the same matrix, profiled or not, and so the same impacts.
"""

import copy
import tempfile

import numpy as np
import pytest

from rabbit import fitter, inputdata
from rabbit.param_models.helpers import load_model
from tests.test_sparse_fit import make_options, make_test_tensor


def _fitter(fname, **kw):
    indata_obj = inputdata.FitInputData(fname)
    f = fitter.make_fitter(indata_obj, load_model("Mu", indata_obj), make_options(**kw))
    f.set_nobs(indata_obj.data_obs)
    # off the starting point, so every block of the Hessian is non-trivial
    f.x.assign(f.x + 0.05)
    return f


@pytest.fixture(scope="module")
def tensor():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield make_test_tensor(tmpdir)


@pytest.mark.parametrize("bb_mode", ["lite", "full"])
@pytest.mark.parametrize("batch", [1, 2, 1000])
def test_hessian_matches_jacobian(tensor, bb_mode, batch):
    opts = dict(binByBinStatType="gamma", binByBinStatMode=bb_mode)
    ref = _fitter(tensor, **opts)
    hvp = _fitter(tensor, hvpBatch=batch, **opts)
    assert not ref.hessian_from_hvp and hvp.hessian_from_hvp
    for profile in (True, False):
        v0, g0, h0 = ref.loss_val_grad_hess(profile=profile)
        v1, g1, h1 = hvp.loss_val_grad_hess(profile=profile)
        np.testing.assert_allclose(float(v1), float(v0), rtol=1e-14)
        np.testing.assert_allclose(g1.numpy(), g0.numpy(), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(h1.numpy(), h0.numpy(), rtol=1e-12, atol=1e-12)


def test_impacts_match_jacobian(tensor):
    """impacts_parms takes the profile=False Hessian through the same path."""
    out = []
    for kw in ({}, {"hvpBatch": 2}):
        f = _fitter(tensor, **kw)
        _, _, hess = f.loss_val_grad_hess()
        f.cov.assign(np.linalg.inv(hess.numpy()))
        out.append([np.asarray(r) for r in f.impacts_parms(hess)])
    for a, b in zip(*out):
        np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("batch", [0, 1])
def test_unbatched_profile_false(tensor, batch):
    """--hvpBatch 0 or 1 with profile=False, as --doImpacts with BBB asks for.

    The sequential loop only knows the profiled loss, so these go through the
    batched path one column at a time; an empty batch never advanced.
    """
    opts = dict(binByBinStatType="gamma", binByBinStatMode="lite")
    _, _, h0 = _fitter(tensor, **opts).loss_val_grad_hess(profile=False)
    _, _, h1 = _fitter(tensor, hvpBatch=batch, **opts).loss_val_grad_hess(profile=False)
    np.testing.assert_allclose(h1.numpy(), h0.numpy(), rtol=1e-12, atol=1e-12)


def test_deepcopy_after_hvp_hessian(tensor):
    """rabbit_limit.py and the saturated fit deepcopy a fitter after its
    Hessian, which traces the batched-HVP tf.function; that must be rebuilt,
    not copied."""
    f = _fitter(tensor, hvpBatch=2)
    _, _, h0 = f.loss_val_grad_hess()
    g = copy.deepcopy(f)
    _, _, h1 = g.loss_val_grad_hess()
    np.testing.assert_allclose(h1.numpy(), h0.numpy(), rtol=1e-12, atol=1e-12)
