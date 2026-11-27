"""Profiled residual covariance with bin-by-bin stat constraint centres 0.

Fitter._residuals_profiled, which the chi2 of a mapping reads, propagates
each constraint centre beta0 with the variance
(d2 lbeta / d beta2) / (d2 lbeta / d beta d beta0)^2, beta0 / kstat for the
gamma constraint. A frequentist toy draws beta0 = Poisson(kstat * beta) /
kstat = 0 where kstat is small; the constraint is then kstat * beta, both
derivatives vanish, and the 0/0 made the covariance (and the chi2) NaN. The
covariance at beta0 = 0 is the limit beta0 -> 0.
"""

import os

import hist
import numpy as np
import pytest
import tensorflow as tf

from rabbit import fitter, inputdata, tensorwriter
from rabbit.mappings import helpers as mh
from rabbit.param_models.helpers import load_model

from .test_sparse_fit import make_options

# Six bins: "sig", few simulated events of weight 30 (kstat 1 to 6), and
# "bkg", 400 events of weight 0.5 (kstat 400).
NSIG = np.array([1.0, 2.0, 4.0, 6.0, 3.0, 1.0])
SUMW = np.stack([30.0 * NSIG, np.full(6, 200.0)], axis=-1)
SUMW2 = np.stack([900.0 * NSIG, np.full(6, 100.0)], axis=-1)
NOBS = np.array([270.0, 300.0, 360.0, 405.0, 320.0, 215.0])
ZERO_BINS = [2, 4]


def _tensor(outdir):
    ax = hist.axis.Regular(len(NOBS), 0, len(NOBS), name="x")
    data = hist.Hist(ax, storage=hist.storage.Double())
    data.values()[...] = NOBS
    writer = tensorwriter.TensorWriter()
    writer.add_channel(data.axes, "ch0")
    writer.add_data(data, "ch0")
    for i, name in enumerate(["sig", "bkg"]):
        h = hist.Hist(ax, storage=hist.storage.Weight())
        h.values()[...] = SUMW[:, i]
        h.variances()[...] = SUMW2[:, i]
        writer.add_process(h, name, "ch0", signal=name == "sig")
    writer.add_norm_systematic("bkg_norm", "bkg", "ch0", 1.05)
    writer.write(outfolder=outdir, outfilename="residuals_zero_centres")
    return os.path.join(outdir, "residuals_zero_centres.hdf5")


def _residuals(f, mapping, beta0):
    f.bbstat.set_beta0(tf.constant(beta0))
    f._profile_beta()
    res, cov = f._residuals_profiled(mapping.compute_flat)
    return res.numpy(), cov.numpy()


@pytest.mark.parametrize("chisq", [False, True])
@pytest.mark.parametrize("mode", ["lite", "full"])
def test_residual_covariance_at_zero_centres(mode, chisq, tmp_path):
    indata = inputdata.FitInputData(_tensor(str(tmp_path)))
    options = make_options(
        binByBinStatType="gamma",
        binByBinStatMode=mode,
        chisqFit=chisq,
        earlyStopping=20,
    )
    f = fitter.Fitter(indata, load_model("Mu", indata), options)
    f.set_nobs(indata.data_obs)

    nominal = f.bbstat.beta0.numpy()
    zero = np.zeros(nominal.shape, dtype=bool)
    # full mode: bkg, which does not attain min(kstat / norm), so its profiled
    # beta is 0 (the profile's other regime is covered with its own fix)
    zero[ZERO_BINS] = True
    if mode == "full":
        zero[:, 0] = False

    beta0 = np.where(zero, 0.0, nominal)
    f.bbstat.set_beta0(tf.constant(beta0))
    f.minimize()
    _, _, hess = f.loss_val_grad_hess()
    f.cov.assign(np.linalg.inv(hess.numpy()))

    mapping = mh.load_mapping("BaseMapping", f.indata)
    res, cov = _residuals(f, mapping, beta0)
    assert np.isfinite(res).all() and np.isfinite(cov).all()

    # the beta0 -> 0 limit, at the same parameters and covariance: the
    # centre's variance beta0 / kstat and its profiled beta are linear in eps
    for eps in [1e-6, 1e-8]:
        _, cov_eps = _residuals(f, mapping, np.where(zero, eps, nominal))
        atol = 10 * eps * np.abs(cov).max()
        np.testing.assert_allclose(cov_eps, cov, rtol=0, atol=atol)
