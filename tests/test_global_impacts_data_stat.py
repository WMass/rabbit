"""Data stat group of the likelihood-based global impacts.

The data stat impact is the remainder var_total - var_x0 - var_beta0, the data
term's share of the curvature. That share is not positive semi-definite in
general: with large postfit residuals and a prediction that is not linear in
the parameters it can be negative, and its square root was NaN. Bin-by-bin
stat constraint centres far below 1, which frequentist toys draw at low kstat
(down to 0), pull this lite-mode fit enough for it, already at beta0 = 0.6 in
one bin. The negative remainder is reported as 0; a positive one is unchanged,
and then the groups still add up to the total.
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

# Six bins: "sig", few simulated events of weight 30, and "bkg", 400 events of
# weight 0.5; lite mode merges them into kstat 53 to 80 per bin.
NSIG = np.array([1.0, 2.0, 4.0, 6.0, 3.0, 1.0])
SUMW = np.stack([30.0 * NSIG, np.full(6, 200.0)], axis=-1)
SUMW2 = np.stack([900.0 * NSIG, np.full(6, 100.0)], axis=-1)
NOBS = np.array([270.0, 300.0, 360.0, 405.0, 320.0, 215.0])


def _fitter(outdir):
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
    writer.write(outfolder=outdir, outfilename="global_impacts_data_stat")

    indata = inputdata.FitInputData(
        os.path.join(outdir, "global_impacts_data_stat.hdf5")
    )
    options = make_options(binByBinStatType="gamma", earlyStopping=20)
    f = fitter.Fitter(indata, load_model("Mu", indata), options)
    f.set_nobs(indata.data_obs)
    return f


def _global_impacts(f, beta0):
    """Fit with the constraint centres beta0; the POI's per-source impacts,
    its grouped impacts and the remainder var_total - var_x0 - var_beta0."""
    f.bbstat.set_beta0(tf.constant(beta0))
    f.minimize()
    _, _, hess = f.loss_val_grad_hess()
    f.cov.assign(np.linalg.inv(hess.numpy()))
    impacts, grouped = [np.asarray(t)[0] for t in f.global_impacts_parms()]
    stat = len(f.indata.systgroups)  # [syst groups | stat | binByBinStat]
    remainder = f.cov.numpy()[0, 0] - np.sum(impacts**2) - grouped[stat + 1] ** 2
    return impacts, grouped, grouped[stat], remainder


@pytest.mark.parametrize("centre", [0.6, 0.0])
def test_negative_remainder(centre, tmp_path):
    f = _fitter(str(tmp_path))
    beta0 = f.bbstat.beta0.numpy()
    beta0[0] = centre
    impacts, grouped, stat, remainder = _global_impacts(f, beta0)
    assert np.isfinite(impacts).all() and np.isfinite(grouped).all()
    assert remainder < 0
    assert stat == 0

    # and the impacts on the observables
    mapping = mh.load_mapping("BaseMapping", f.indata)
    _, aux = f.expected_events(mapping, compute_global_impacts=True)
    assert np.isfinite(aux[2]).all() and np.isfinite(aux[3]).all()


def test_positive_remainder_unchanged(tmp_path):
    f = _fitter(str(tmp_path))
    _, _, stat, remainder = _global_impacts(f, f.bbstat.beta0.numpy())
    assert remainder > 0
    np.testing.assert_allclose(stat**2, remainder, rtol=1e-10)
