"""Global impacts with --binByBinStatType gamma --binByBinStatMode full.

That mode profiles beta with a Newton loop. The default (JVP) global impacts
run it under a ForwardAccumulator inside tf.vectorized_map, which cannot
vectorise a loop that captures a tf.Variable, so this used to crash with
"Tried to take gradients (or similar) of a variable without handle data".
The JVP result must agree with the traditional backward-mode one.

The chi-square variant also covers a toy whose minimum fits every bin
exactly: the third derivative of r**2 is NaN at r == 0 in TensorFlow, which
made the whole postfit Hessian NaN there.
"""

import tempfile

import numpy as np
import pytest

from rabbit import fitter, inputdata
from rabbit.param_models.helpers import load_model
from tests.test_sparse_fit import make_options, make_test_tensor


def _global_impacts(fname, jvp, chisq):
    indata_obj = inputdata.FitInputData(fname)
    param_model = load_model("Mu", indata_obj)
    options = make_options(
        binByBinStatType="gamma", binByBinStatMode="full", chisqFit=chisq
    )
    f = fitter.make_fitter(indata_obj, param_model, options, globalImpactsFromJVP=jvp)
    f.set_nobs(indata_obj.data_obs)
    f.minimize()
    _, _, hess = f.loss_val_grad_hess()
    f.cov.assign(np.linalg.inv(hess.numpy()))
    return [np.asarray(r) for r in f.global_impacts_parms()]


@pytest.mark.parametrize("chisq", [False, True])
def test_jvp_matches_backward_mode(chisq):
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = make_test_tensor(tmpdir)
        ref = _global_impacts(fname, jvp=False, chisq=chisq)
        jvp = _global_impacts(fname, jvp=True, chisq=chisq)
    for r, j in zip(ref, jvp):
        assert np.all(np.isfinite(j))
        assert np.abs(r).max() > 0
        np.testing.assert_allclose(j, r, rtol=1e-9, atol=1e-14)
