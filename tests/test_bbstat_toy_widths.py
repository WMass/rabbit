"""Bin-by-bin stat toys draw with the width of the constraint lbeta uses.

lbeta is kstat * (beta - beta0 * log(beta)) for gamma, 0.5 * kstat *
(beta - beta0)^2 for normal-multiplicative and 0.5 * (beta - beta0)^2 for
normal-additive (beta in units of sqrt(varbeta)). Frequentist toys draw the
centre beta0 around beta, Bayesian toys beta around beta0, from those
constraints. For normal-multiplicative the width is 1/sqrt(kstat) =
sqrt(varbeta) / sumw; the toys drew with sqrt(varbeta), the absolute MC stat
uncertainty of the yield, a factor sumw too wide (sqrt(varbeta) was the
relative width while varbeta was 1/kstat). Masked entries have no constraint
and keep their centre. The normal types also drew one deviate for all entries
(tf.random.normal with shape=[]), fully correlated MC stat fluctuations that
made the toy pulls of the signal strength 1.4 to 1.8 wide.
"""

import os
from types import SimpleNamespace

import hist
import numpy as np
import pytest
import tensorflow as tf

from rabbit import fitter, inputdata, tensorwriter
from rabbit.bbstat.bbstat import BinByBinStat
from rabbit.param_models.helpers import load_model

from .test_sparse_fit import make_options

# Four bins, three processes: "sig" (weight 5, kstat 4 to 400), "bkg"
# (weight 0.5, kstat 400) and "flat" without MC stat uncertainty (masked in
# full mode).
NSIG = np.array([4.0, 25.0, 100.0, 400.0])
SUMW = np.stack([5.0 * NSIG, np.full(4, 200.0), np.full(4, 40.0)], axis=-1)
SUMW2 = np.stack([25.0 * NSIG, np.full(4, 100.0), np.zeros(4)], axis=-1)
TYPES = ["gamma", "normal-multiplicative", "normal-additive"]
NDRAW = 2000


def _bbstat(stat_type, mode):
    indata = SimpleNamespace(
        sumw=tf.constant(SUMW),
        sumw2=tf.constant(SUMW2),
        nbins=SUMW.shape[0],
        nbinsmasked=0,
        nbinsfull=SUMW.shape[0],
        dtype=tf.float64,
        norm=None,
        betavar=None,
    )
    options = SimpleNamespace(
        noBinByBinStat=False,
        binByBinStatMode=mode,
        binByBinStatType=stat_type,
        minBBKstat=0.0,
    )
    return BinByBinStat(
        indata, options, chisqFit=False, covarianceFit=False, data_cov_inv=None
    )


def _draws(bb, randomize, variable):
    """NDRAW draws of `variable` by `randomize`, from the default state."""
    tf.random.set_seed(1)
    out = []
    for _ in range(NDRAW):
        bb.beta0_default_assign()
        bb.beta_default_assign()
        randomize()
        out.append(variable.numpy())
    return np.array(out)


def _check_spread(bb, stat_type, draws, centre, width):
    """Sample mean and standard deviation per entry against the centre and
    width. Masked entries have no constraint: normal-multiplicative leaves
    them at their centre, gamma draws them with its placeholder kstat (not
    checked here), normal-additive does not use betamask."""
    mask = bb.betamask.numpy()
    if stat_type == "normal-multiplicative":
        assert (draws[:, mask] == centre[mask][None]).all()
    if stat_type == "normal-additive":
        mask = np.zeros_like(mask)
    mean, std = draws.mean(axis=0)[~mask], draws.std(axis=0)[~mask]
    width, centre = width[~mask], centre[~mask]
    np.testing.assert_allclose(std, width, rtol=0.06)
    assert np.all(np.abs(mean - centre) < 5 * width / np.sqrt(NDRAW))


@pytest.mark.parametrize("mode", ["full", "lite"])
@pytest.mark.parametrize("stat_type", TYPES)
def test_frequentist_centres(stat_type, mode):
    bb = _bbstat(stat_type, mode)
    kstat = bb.kstat.numpy()
    centre = bb.default_beta0().numpy()
    # beta0 around beta (the default): Poisson(kstat * beta) / kstat for gamma
    width = {
        "gamma": np.sqrt(centre / kstat),
        "normal-multiplicative": 1.0 / np.sqrt(kstat),
        "normal-additive": np.ones_like(kstat),
    }[stat_type]
    draws = _draws(bb, bb.randomize_frequentist, bb.beta0)
    _check_spread(bb, stat_type, draws, centre, width)


@pytest.mark.parametrize("mode", ["full", "lite"])
@pytest.mark.parametrize("stat_type", TYPES)
def test_bayesian_values(stat_type, mode):
    bb = _bbstat(stat_type, mode)
    kstat = bb.kstat.numpy()
    beta0 = bb.default_beta0().numpy()
    # beta around beta0: Gamma(kstat * beta0 + 1) / kstat for gamma
    centre, width = {
        "gamma": ((kstat * beta0 + 1) / kstat, np.sqrt(kstat * beta0 + 1) / kstat),
        "normal-multiplicative": (beta0, 1.0 / np.sqrt(kstat)),
        "normal-additive": (beta0, np.ones_like(kstat)),
    }[stat_type]
    draws = _draws(bb, bb.randomize_bayes, bb.beta)
    _check_spread(bb, stat_type, draws, centre, width)


@pytest.mark.parametrize("randomize", ["frequentist", "bayes"])
@pytest.mark.parametrize("mode", ["full", "lite"])
@pytest.mark.parametrize("stat_type", ["normal-multiplicative", "normal-additive"])
def test_normal_draws_are_independent(stat_type, mode, randomize):
    """Each entry draws its own normal deviate: tf.random.normal with
    shape=[] drew one and scaled it by every entry's width."""
    bb = _bbstat(stat_type, mode)
    method = getattr(bb, f"randomize_{randomize}")
    variable = bb.beta0 if randomize == "frequentist" else bb.beta
    draws = _draws(bb, method, variable).reshape(NDRAW, -1)
    draws = draws[:, draws.std(axis=0) > 0.0]
    corr = np.corrcoef(draws.T)
    offdiag = corr[~np.eye(len(corr), dtype=bool)]
    assert np.abs(offdiag).max() < 0.1


@pytest.mark.parametrize("mode", ["full", "lite"])
@pytest.mark.parametrize("stat_type", TYPES)
def test_starting_values(stat_type, mode):
    """--toysRandomizeParameters: beta around beta0 with the width of its
    constraint at its minimum (it drew with sqrt(varbeta) for every type),
    each entry independently; masked entries keep beta0."""
    bb = _bbstat(stat_type, mode)
    kstat = bb.kstat.numpy()
    beta0 = bb.default_beta0().numpy()
    width = {
        "gamma": np.sqrt(beta0 / kstat),
        "normal-multiplicative": 1.0 / np.sqrt(kstat),
        "normal-additive": np.ones_like(kstat),
    }[stat_type]
    draws = _draws(bb, bb.randomize_postfit, bb.beta)
    mask = bb.betamask.numpy()
    if stat_type != "normal-additive":
        assert (draws[:, mask] == beta0[mask][None]).all()
    _check_spread(bb, stat_type, draws, beta0, width)
    flat = draws.reshape(NDRAW, -1)
    flat = flat[:, flat.std(axis=0) > 0.0]
    corr = np.corrcoef(flat.T)
    assert np.abs(corr[~np.eye(len(corr), dtype=bool)]).max() < 0.1


# Seven bins of a "sig" with weight 2 (kstat 10 to 100) and a "bkg" with
# weight 1 (kstat 150), data at the nominal yields.
FIT_NSIG = np.array([10.0, 30.0, 60.0, 100.0, 60.0, 30.0, 10.0])
FIT_SUMW = np.stack([2.0 * FIT_NSIG, np.full(7, 150.0)], axis=-1)
FIT_SUMW2 = np.stack([4.0 * FIT_NSIG, np.full(7, 150.0)], axis=-1)
NTOYS = 40


def _fitter(outdir, stat_type, mode):
    ax = hist.axis.Regular(len(FIT_NSIG), 0, len(FIT_NSIG), name="x")
    data = hist.Hist(ax, storage=hist.storage.Double())
    data.values()[...] = FIT_SUMW.sum(axis=-1)
    writer = tensorwriter.TensorWriter()
    writer.add_channel(data.axes, "ch0")
    writer.add_data(data, "ch0")
    for i, name in enumerate(["sig", "bkg"]):
        h = hist.Hist(ax, storage=hist.storage.Weight())
        h.values()[...] = FIT_SUMW[:, i]
        h.variances()[...] = FIT_SUMW2[:, i]
        writer.add_process(h, name, "ch0", signal=name == "sig")
    writer.add_norm_systematic("bkg_norm", "bkg", "ch0", 1.05)
    writer.write(outfolder=outdir, outfilename="toy_widths")

    indata = inputdata.FitInputData(os.path.join(outdir, "toy_widths.hdf5"))
    options = make_options(
        binByBinStatType=stat_type, binByBinStatMode=mode, earlyStopping=20
    )
    f = fitter.Fitter(indata, load_model("Mu", indata), options)
    f.set_nobs(indata.data_obs)
    return f


@pytest.mark.parametrize(
    "stat_type, mode",
    [
        ("normal-multiplicative", "full"),
        ("normal-multiplicative", "lite"),
        ("normal-additive", "full"),
    ],
)
def test_normal_toy_pulls(stat_type, mode, tmp_path):
    """Frequentist toys fit, and the pulls of the signal strength and the
    background normalization have unit width (within the statistics of NTOYS
    toys; 200 toys give 1.02 to 1.13 against 1.05 to 1.09 for gamma)."""
    f = _fitter(str(tmp_path), stat_type, mode)
    xtrue = f.x0default.numpy()
    tf.random.set_seed(11)
    pulls = []
    for _ in range(NTOYS):
        f.defaultassign()
        f.toyassign(syst_randomize="frequentist", data_randomize="poisson")
        f.minimize()
        val, grad, hess = [t.numpy() for t in f.loss_val_grad_hess()]
        assert np.isfinite(val) and np.max(np.abs(grad)) < 1e-3
        err = np.sqrt(np.diag(np.linalg.inv(hess)))
        pulls.append((f.x.numpy() - xtrue) / err)
    pulls = np.array(pulls)
    assert np.abs(pulls).max() < 4.5
    np.testing.assert_allclose(pulls.mean(axis=0), 0.0, atol=0.6)
    np.testing.assert_allclose(pulls.std(axis=0), 1.0, atol=0.45)
