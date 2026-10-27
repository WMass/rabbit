"""Fitter.toyassign draws its normal deviates independently per entry.

tf.random.normal(shape=[], mean=m, stddev=s) draws one deviate and scales it
by every entry's stddev, so the normal pseudo-data (--toysDataRandomize
normal) moved every bin by the same number of sigmas, and the randomized
starting values (--toysRandomizeParameters) every parameter alike. On chi2
toys the data fluctuations were fully correlated across bins, and the pulls
of the signal strength 1.43 wide (1.02 now).
"""

import numpy as np
import tensorflow as tf

from rabbit import fitter, inputdata
from rabbit.param_models.helpers import load_model

from .test_sparse_fit import make_options, make_test_tensor

NTOYS = 400


def _fitter(outdir, **opts):
    indata = inputdata.FitInputData(make_test_tensor(outdir))
    f = fitter.Fitter(
        indata, load_model("Mu", indata), make_options(earlyStopping=20, **opts)
    )
    f.set_nobs(indata.data_obs)
    return f


def _max_correlation(draws):
    corr = np.corrcoef(draws.T)
    return np.abs(corr[~np.eye(len(corr), dtype=bool)]).max()


def test_normal_pseudodata(tmp_path):
    f = _fitter(str(tmp_path), chisqFit=True, noBinByBinStat=True)
    nominal = f.expected_yield().numpy()
    tf.random.set_seed(3)
    pulls = []
    for _ in range(NTOYS):
        f.defaultassign()
        f.toyassign(syst_randomize="none", data_randomize="normal")
        pulls.append((f.nobs.numpy() - nominal) / np.sqrt(nominal))
    pulls = np.array(pulls)
    np.testing.assert_allclose(pulls.std(axis=0), 1.0, atol=0.15)
    assert _max_correlation(pulls) < 0.2


def test_randomized_starting_values(tmp_path):
    f = _fitter(str(tmp_path))
    nuis = slice(f.param_model.nparams, None)
    tf.random.set_seed(3)
    shifts = []
    for _ in range(NTOYS):
        f.defaultassign()
        f.toyassign(
            syst_randomize="none", data_randomize="none", randomize_parameters=True
        )
        shifts.append((f.x.numpy() - f.x0default.numpy())[nuis])
    shifts = np.array(shifts)
    assert shifts.shape[1] >= 2
    np.testing.assert_allclose(shifts.std(axis=0), 1.0, atol=0.15)
    assert _max_correlation(shifts) < 0.2
