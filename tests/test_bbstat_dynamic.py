"""Tests for --binByBinStatDynamic (lite mode with a yield-dependent variance).

In lite mode the per-bin MC-stat variance is by default fixed at the nominal
process composition. With the dynamic option it follows the current yields,
V = sum_p sumw2_p * (n_p / sumw_p)**2, while the constraint on beta keeps its
nominal width. The checks below are equivalences that hold exactly:

- at the nominal yields dynamic and static lite agree;
- for normal-multiplicative, dynamic lite equals full mode, whose per-process
  betas already carry the per-process relative variances (profiling the
  per-process betas at fixed total yield leaves a Gaussian in the total with
  variance V);
- normal-additive and normal-multiplicative dynamic lite give the same profiled
  NLL, since both are a Gaussian shift of the total yield with variance V.
"""

import os
import tempfile
from types import SimpleNamespace

import hist
import numpy as np
import pytest
import tensorflow as tf

from rabbit import fitter, inputdata, tensorwriter
from rabbit.bbstat.bbstat import BinByBinStat
from rabbit.param_models.helpers import load_model

# Three bins, two processes: a high-stat "signal" and a low-stat "background",
# so the relative variance of the total depends on the composition.
SUMW = np.array([[50.0, 20.0], [80.0, 10.0], [30.0, 40.0]])
SUMW2 = np.array([[5.0, 20.0], [4.0, 15.0], [3.0, 60.0]])
NOBS = np.array([140.0, 150.0, 120.0])
# Signal scaled up by 2.5, background down by 0.8.
SCALE = np.array([2.5, 0.8])
# Low-stat background scaled up: the relative variance of the total grows
# (s > 1), so the beta-independent part of the yield goes negative.
SCALES = [SCALE, np.array([0.3, 3.0])]


def _indata(sumw, sumw2):
    sumw = tf.constant(sumw, dtype=tf.float64)
    sumw2 = tf.constant(sumw2, dtype=tf.float64)
    nbins = int(sumw.shape[0])
    return SimpleNamespace(
        sumw=sumw,
        sumw2=sumw2,
        nbins=nbins,
        nbinsmasked=0,
        nbinsfull=nbins,
        dtype=tf.float64,
        norm=None,
        betavar=None,
    )


def _data_cov_inv(nbins):
    rng = np.random.default_rng(3)
    a = rng.normal(size=(nbins, nbins))
    cov = np.diag(NOBS[:nbins]) + 5.0 * (a @ a.T)
    return tf.constant(np.linalg.inv(cov), dtype=tf.float64)


def _bbstat(fit, stat_type, mode, dynamic, sumw=SUMW, sumw2=SUMW2):
    ind = _indata(sumw, sumw2)
    options = SimpleNamespace(
        noBinByBinStat=False,
        binByBinStatMode=mode,
        binByBinStatType=stat_type,
        binByBinStatDynamic=dynamic,
        minBBKstat=0.0,
    )
    return BinByBinStat(
        ind,
        options,
        chisqFit=fit == "chisq",
        covarianceFit=fit == "cov",
        data_cov_inv=_data_cov_inv(ind.nbins) if fit == "cov" else None,
        nobs_template=tf.zeros((ind.nbins,), dtype=tf.float64),
    )


def _nll(bb, fit, norm, nobs=NOBS):
    """Profiled NLL (data term + beta constraint) at the given process yields."""
    norm = tf.constant(norm, dtype=tf.float64)
    nobs = tf.constant(nobs, dtype=tf.float64)
    varnobs = nobs if fit == "chisq" else None
    nexp, _, beta = bb.profile_and_apply(
        tf.reduce_sum(norm, axis=-1), norm, nobs, varnobs, tf.math.log(nobs)
    )
    if fit == "poisson":
        ln = tf.reduce_sum(nexp - nobs * tf.math.log(nexp))
    elif fit == "chisq":
        ln = 0.5 * tf.reduce_sum((nexp - nobs) ** 2 / varnobs)
    else:
        r = (nobs - nexp)[:, None]
        ln = 0.5 * tf.reduce_sum(tf.transpose(r) @ bb.data_cov_inv @ r)
    return ln + bb.lbeta(beta), nexp, beta


FITS = ["poisson", "chisq", "cov"]
TYPES = ["gamma", "normal-multiplicative", "normal-additive"]
COMBOS = [(f, t) for f in FITS for t in TYPES if not (f == "cov" and t == "gamma")]


@pytest.mark.parametrize("fit, stat_type", COMBOS)
def test_nominal_yields_match_static(fit, stat_type):
    """At the nominal composition the dynamic variance is the static one."""
    static = _nll(_bbstat(fit, stat_type, "lite", False), fit, SUMW)
    dynamic = _nll(_bbstat(fit, stat_type, "lite", True), fit, SUMW)
    for a, b in zip(static, dynamic):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-12)


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("fit", FITS)
def test_normal_multiplicative_dynamic_lite_equals_full(fit, scale):
    """Dynamic lite reproduces full mode away from the nominal composition."""
    norm = SUMW * scale
    full, nexp_full, _ = _nll(
        _bbstat(fit, "normal-multiplicative", "full", False), fit, norm
    )
    lite, nexp_lite, _ = _nll(
        _bbstat(fit, "normal-multiplicative", "lite", True), fit, norm
    )
    static, _, _ = _nll(_bbstat(fit, "normal-multiplicative", "lite", False), fit, norm)

    np.testing.assert_allclose(lite.numpy(), full.numpy(), rtol=1e-10)
    np.testing.assert_allclose(nexp_lite.numpy(), nexp_full.numpy(), rtol=1e-10)
    # non-vacuous: the static composition gives a different answer here
    assert not np.isclose(static.numpy(), full.numpy(), rtol=1e-4)


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("fit", FITS)
def test_normal_additive_equals_multiplicative_when_dynamic(fit, scale):
    """Both normal types are a Gaussian shift of the total with variance V."""
    norm = SUMW * scale
    add, nexp_add, _ = _nll(_bbstat(fit, "normal-additive", "lite", True), fit, norm)
    mult, nexp_mult, _ = _nll(
        _bbstat(fit, "normal-multiplicative", "lite", True), fit, norm
    )
    np.testing.assert_allclose(add.numpy(), mult.numpy(), rtol=1e-10)
    np.testing.assert_allclose(nexp_add.numpy(), nexp_mult.numpy(), rtol=1e-10)


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("fit, stat_type", COMBOS)
def test_gradient_through_the_variance(fit, stat_type, scale):
    """The dependence of V on the yields is differentiated, and correctly."""
    bb = _bbstat(fit, stat_type, "lite", True)

    def f(mu):
        s = tf.stack([mu, tf.constant(scale[1], dtype=tf.float64)])
        return _nll(bb, fit, SUMW * s)[0]

    mu = tf.constant(scale[0], dtype=tf.float64)
    with tf.GradientTape() as t:
        t.watch(mu)
        val = f(mu)
    grad = t.gradient(val, mu).numpy()
    h = 1e-5
    fd = (f(mu + h).numpy() - f(mu - h).numpy()) / (2 * h)
    assert np.isfinite(grad)
    np.testing.assert_allclose(grad, fd, rtol=1e-6)


@pytest.mark.parametrize("fit, stat_type", COMBOS)
def test_zero_variance_process(fit, stat_type):
    """A process with sumw2 == 0 adds no variance and is not scaled by beta."""
    sumw = np.concatenate([SUMW, [[10.0], [5.0], [8.0]]], axis=1)
    sumw2 = np.concatenate([SUMW2, [[0.0], [0.0], [0.0]]], axis=1)
    static = _nll(_bbstat(fit, stat_type, "lite", False, sumw, sumw2), fit, sumw)
    dynamic = _nll(_bbstat(fit, stat_type, "lite", True, sumw, sumw2), fit, sumw)
    for a, b in zip(static, dynamic):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-12)

    # scaling only the zero-variance process leaves V unchanged
    bb = _bbstat(fit, stat_type, "lite", True, sumw, sumw2)
    var = bb._dynamic_variance(tf.constant(sumw * [1.0, 1.0, 3.0]))
    np.testing.assert_allclose(var.numpy(), SUMW2.sum(axis=-1), rtol=1e-12)


def test_full_mode_refused():
    with pytest.raises(ValueError, match="only applies"):
        _bbstat("poisson", "normal-multiplicative", "full", True)


def test_per_bin_sumw2_refused():
    with pytest.raises(ValueError, match="per-process"):
        _bbstat(
            "poisson",
            "gamma",
            "lite",
            True,
            sumw=SUMW.sum(axis=-1),
            sumw2=SUMW2.sum(axis=-1),
        )


# --- end-to-end through the Fitter ------------------------------------------


def _make_tensor(path):
    rng = np.random.default_rng(1234)
    ax = hist.axis.Regular(10, -5, 5, name="x")
    h_data = hist.Hist(ax, storage=hist.storage.Double())
    h_sig = hist.Hist(ax, storage=hist.storage.Weight())
    h_bkg = hist.Hist(ax, storage=hist.storage.Weight())
    # high-stat signal (small weights), low-stat background (large weights)
    x_sig = rng.normal(0, 1, 40000)
    x_bkg = rng.uniform(-5, 5, 400)
    h_sig.fill(x_sig, weight=np.full_like(x_sig, 0.1))
    h_bkg.fill(x_bkg, weight=np.full_like(x_bkg, 10.0))
    # data with twice the signal, so the fit moves away from the nominal mix
    h_data.view()[...] = 2.0 * h_sig.values() + h_bkg.values()

    w = tensorwriter.TensorWriter()
    w.add_channel(h_data.axes, "ch0")
    w.add_data(h_data, "ch0")
    w.add_process(h_sig, "sig", "ch0", signal=True)
    w.add_process(h_bkg, "bkg", "ch0", signal=False)
    w.add_norm_systematic("bkgNorm", ["bkg"], "ch0", 1.05)
    w.write(outfolder=os.path.dirname(path), outfilename=os.path.basename(path))


@pytest.fixture(scope="module")
def tensor_path():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "bbstat_dynamic.hdf5")
        _make_tensor(p)
        yield p


def _fit(path, **opts):
    options = dict(
        earlyStopping=-1,
        noBinByBinStat=False,
        binByBinStatMode="lite",
        binByBinStatType="normal-multiplicative",
        binByBinStatDynamic=False,
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
    options.update(opts)
    ind = inputdata.FitInputData(path)
    model = load_model("Mu", ind)
    model.allowNegativeParam = True
    f = fitter.Fitter(ind, model, SimpleNamespace(**options))
    f.set_nobs(ind.data_obs)
    f.minimize()
    _, _, hess = f.loss_val_grad_hess()
    cov = np.linalg.inv(hess.numpy())
    return f.x.numpy(), np.sqrt(np.diag(cov))


def test_fit_dynamic_lite_matches_full(tensor_path):
    """POI, nuisances and their uncertainties agree with full mode."""
    x_full, err_full = _fit(tensor_path, binByBinStatMode="full")
    x_dyn, err_dyn = _fit(tensor_path, binByBinStatDynamic=True)
    x_static, err_static = _fit(tensor_path)

    assert x_full[0] == pytest.approx(2.0, rel=0.05)
    np.testing.assert_allclose(x_dyn, x_full, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(err_dyn, err_full, rtol=1e-6)
    # non-vacuous: with the well-known signal doubled, the nominal relative
    # variance of the total (dominated by the background) overestimates the
    # MC-stat uncertainty
    assert err_static[0] > 1.1 * err_full[0]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
