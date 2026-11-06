"""--binByBinStatType gamma --binByBinStatMode full with constraint centres 0.

A frequentist toy draws the centres as beta0 = Poisson(kstat * beta) / kstat,
which is 0 with probability exp(-kstat * beta). The profiled beta of a process
with beta0 = 0 is 0, unless the process attains min(kstat / norm) in its bin
and the other processes cannot account for the data: then the minimum is on
the threshold of the Newton solve, where that process's beta is fixed by the
total yield (Barlow and Beeston's case of a source without simulated events in
a bin; rabbit.bbstat.formulas.gamma_full_threshold_profile). The solve cannot
reach the threshold, so the likelihood was NaN where every centre of a bin is 0
(Poisson: no predicted events) and otherwise jumped at beta0 = 0.

- the profile is the constrained minimum of the bin likelihood, by its KKT
  conditions (zero gradient where beta > 0, positive where beta = 0), in each
  regime: on the threshold, a zero centre with the minimum above the threshold,
  a zero centre not attaining min(kstat / norm), no zero centre;
- the likelihood, its gradient and the betas are the beta0 -> 0 limit;
- through the Fitter: finite value, gradient (also jit-compiled) and Hessian,
  equal to finite differences;
- a frequentist toy with zero centres converges to the minimum of the same toy
  with the zero centres replaced by a small epsilon;
- columns of centres with zeros in one batched evaluation equal the single
  evaluations.
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

# Six bins. Processes: "sig", few simulated events of weight 30 (kstat 1 to 6),
# "bkg", 400 events of weight 0.5 (kstat 400), and one without bin-by-bin stat
# uncertainty (sumw2 = 0, masked). sig has the smallest kstat / norm everywhere.
NSIG = np.array([1.0, 2.0, 4.0, 6.0, 3.0, 1.0])
SUMW = np.stack([30.0 * NSIG, np.full(6, 200.0), np.full(6, 20.0)], axis=-1)
SUMW2 = np.stack([900.0 * NSIG, np.full(6, 100.0), np.zeros(6)], axis=-1)
NOBS = np.array([270.0, 300.0, 360.0, 405.0, 320.0, 215.0])
# process yields away from the nominal ones
SCALE = np.array([1.1, 0.95, 1.0])
# Zero centres: in bin 0 every process with bin-by-bin stat, in bins 1 and 4
# sig (both on the threshold); in bin 5 sig, with too little data for the
# threshold; in bin 2 bkg, which does not attain min(kstat / norm); none in 3.
ZERO = np.array(
    [[1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=bool
)
FITS = ["poisson", "chisq"]


def _bbstat(fit):
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
        binByBinStatMode="full",
        binByBinStatType="gamma",
        minBBKstat=0.0,
    )
    return BinByBinStat(
        indata, options, chisqFit=fit == "chisq", covarianceFit=False, data_cov_inv=None
    )


def _profile(bb, fit, beta0):
    """Profiled NLL (data term + beta constraint) at the yields SUMW * SCALE,
    its gradient with respect to those yields, and the profiled beta."""
    bb.set_beta0(tf.constant(beta0))
    norm = tf.constant(SUMW * SCALE)
    nobs = tf.constant(NOBS)
    varnobs = nobs if fit == "chisq" else None
    with tf.GradientTape() as tape:
        tape.watch(norm)
        nexp, _, beta = bb.profile_and_apply(
            tf.reduce_sum(norm, axis=-1), norm, nobs, varnobs, tf.math.log(nobs)
        )
        if fit == "chisq":
            ln = 0.5 * tf.reduce_sum((nexp - nobs) ** 2 / varnobs)
        else:
            ln = tf.reduce_sum(nexp - nobs - nobs * tf.math.log(nexp / nobs))
        nll = ln + bb.lbeta(beta)
    return nll.numpy(), tape.gradient(nll, norm).numpy(), beta.numpy()


@pytest.mark.parametrize("fit", FITS)
def test_profile_is_the_constrained_minimum(fit):
    bb = _bbstat(fit)
    beta0 = np.where(ZERO, 0.0, 1.0)
    nll, grad, beta = _profile(bb, fit, beta0)
    assert np.isfinite(nll) and np.isfinite(grad).all()

    kstat, mask = bb.kstat.numpy(), bb.betamask.numpy()
    norm = SUMW * SCALE
    # the masked process keeps beta = beta0 = 1
    nexp = np.sum(beta * norm, axis=-1)
    dln = 1.0 - NOBS / nexp if fit == "poisson" else (nexp - NOBS) / NOBS
    # gradient of the (convex) bin likelihood with respect to beta
    dbeta = norm * dln[:, None] + kstat - kstat * beta0 / np.where(beta > 0, beta, 1)
    interior = ~mask & (beta > 0)
    boundary = ~mask & (beta == 0)
    np.testing.assert_allclose(dbeta[interior] / kstat[interior], 0.0, atol=1e-9)
    assert (dbeta[boundary] > 0).all()
    # the regimes the centres were chosen for
    assert (beta[[0, 1, 4], 0] > 0).all()
    assert beta[0, 1] == 0 and beta[2, 1] == 0 and beta[5, 0] == 0


@pytest.mark.parametrize("fit", FITS)
def test_zero_centres_are_the_limit(fit):
    """beta0 = eps -> 0: the NLL converges like kstat * eps * log(eps), its
    gradient and the betas like eps."""
    bb = _bbstat(fit)
    nll, grad, beta = _profile(bb, fit, np.where(ZERO, 0.0, 1.0))
    kzero = np.sum(bb.kstat.numpy()[ZERO])
    for eps in [1e-4, 1e-6]:
        nll_eps, grad_eps, beta_eps = _profile(bb, fit, np.where(ZERO, eps, 1.0))
        assert abs(nll_eps - nll) < kzero * eps * (1.0 - np.log(eps))
        np.testing.assert_allclose(grad_eps, grad, rtol=10 * eps, atol=10 * eps)
        np.testing.assert_allclose(beta_eps, beta, rtol=0, atol=100 * eps)


def _tensor(outdir):
    """sig and bkg of SUMW / SUMW2 (without the masked process), data NOBS."""
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
    writer.write(outfolder=outdir, outfilename="zero_centres")
    return os.path.join(outdir, "zero_centres.hdf5")


def _fitter(outdir, chisq):
    indata = inputdata.FitInputData(_tensor(outdir))
    options = make_options(
        binByBinStatType="gamma",
        binByBinStatMode="full",
        chisqFit=chisq,
        earlyStopping=20,
    )
    f = fitter.Fitter(indata, load_model("Mu", indata), options)
    f.set_nobs(indata.data_obs)
    return f


@pytest.mark.parametrize("chisq", [False, True])
def test_fitter_value_gradient_hessian(chisq, tmp_path):
    f = _fitter(str(tmp_path), chisq)
    beta0 = np.where(ZERO[:, :2], 0.0, 1.0)
    f.bbstat.set_beta0(tf.constant(beta0))
    f.x.assign([1.2, 0.3])
    x = f.x.numpy()

    val, grad, hess = [t.numpy() for t in f.loss_val_grad_hess()]
    assert np.isfinite(val) and np.isfinite(grad).all() and np.isfinite(hess).all()
    val_jit, grad_jit = [t.numpy() for t in f.loss_val_grad()]
    np.testing.assert_allclose(val_jit, val, rtol=1e-12)
    np.testing.assert_allclose(grad_jit, grad, rtol=1e-9)

    h = 1e-5
    grad_fd, hess_fd = np.zeros_like(grad), np.zeros_like(hess)
    for j in range(x.size):
        dx = h * np.eye(x.size)[j]
        f.x.assign(x + dx)
        val_up, grad_up = [t.numpy() for t in f.loss_val_grad()]
        f.x.assign(x - dx)
        val_dn, grad_dn = [t.numpy() for t in f.loss_val_grad()]
        grad_fd[j] = (val_up - val_dn) / (2 * h)
        hess_fd[:, j] = (grad_up - grad_dn) / (2 * h)
    f.x.assign(x)
    np.testing.assert_allclose(grad, grad_fd, rtol=1e-6)
    np.testing.assert_allclose(hess, hess_fd, rtol=1e-5)

    # the beta0 -> 0 limit
    f.bbstat.set_beta0(tf.constant(np.where(beta0 == 0, 1e-8, beta0)))
    assert abs(f.loss_val().numpy() - val) < 1e-4


@pytest.mark.parametrize("chisq", [False, True])
def test_frequentist_toy_fit(chisq, tmp_path):
    f = _fitter(str(tmp_path), chisq)
    # this seed draws a zero sig centre in a bin whose minimum is on the
    # threshold (asserted below)
    tf.random.set_seed(0)
    f.toyassign(
        syst_randomize="frequentist",
        data_randomize="normal" if chisq else "poisson",
    )
    beta0 = f.bbstat.beta0.numpy()
    zero = beta0 == 0
    assert zero.any()
    assert np.isfinite(f.loss_val().numpy())

    f.minimize()
    val, grad = [t.numpy() for t in f.loss_val_grad()]
    assert np.isfinite(val) and np.max(np.abs(grad)) < 1e-6
    # the minimum has a process with beta0 = 0 on the threshold
    beta = f._compute_nll_components()[-1].numpy()
    assert (beta[zero] > 0).any()
    x, nll = f.x.numpy(), f.reduced_nll().numpy()

    # the same toy with centres eps instead of 0, fit from the start
    f.bbstat.set_beta0(tf.constant(np.where(zero, 1e-8, beta0)))
    f.xdefaultassign()
    f.bbstat.beta_default_assign()
    f.minimize()
    np.testing.assert_allclose(f.x.numpy(), x, rtol=0, atol=1e-5)
    assert abs(f.reduced_nll().numpy() - nll) < 1e-4


@pytest.mark.parametrize("chisq", [False, True])
def test_batched_equals_single(chisq, tmp_path):
    """B columns of centres with zeros (vectorized over the batch, Newton solve
    and threshold profile included) at B points equal B single evaluations,
    value and gradient."""
    f = _fitter(str(tmp_path), chisq)
    nominal = f.bbstat.beta0.numpy()
    whole_bins = nominal.copy()
    whole_bins[[0, 3]] = 0.0
    sig = nominal.copy()
    sig[:, 0] = 0.0
    columns = [np.where(ZERO[:, :2], 0.0, nominal), whole_bins, sig, nominal]
    beta0 = np.stack(columns, axis=-1)
    rng = np.random.default_rng(5)
    X = f.x.numpy()[:, None] + 0.2 * rng.standard_normal((f.x.shape[0], len(columns)))

    Xt = tf.constant(X)
    with tf.GradientTape() as tape:
        tape.watch(Xt)
        batched = f._compute_nll_batched(Xt, beta0=tf.constant(beta0))
        total = tf.reduce_sum(batched)
    grads = tape.gradient(total, Xt).numpy()
    batched = batched.numpy()
    assert np.isfinite(batched).all() and np.isfinite(grads).all()

    xsave = f.x.numpy()
    for b in range(len(columns)):
        f.bbstat.set_beta0(tf.constant(beta0[..., b]))
        f.x.assign(X[:, b])
        val, grad = [t.numpy() for t in f.loss_val_grad()]
        np.testing.assert_allclose(batched[b], val, rtol=1e-11)
        np.testing.assert_allclose(grads[:, b], grad, rtol=1e-9, atol=1e-10)
    f.x.assign(xsave)
    f.bbstat.set_beta0(tf.constant(nominal))
