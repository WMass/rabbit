"""The chi2 of a mapping is taken over the rank of its residual covariance.

Residuals can obey exact linear identities, which make their covariance
exactly singular: two groupings of the same bins have the same total, and the
postfit residuals of the ABCD model sum to zero over each pair of A and B bins
and over all bins. The null eigenvalues are then rounding noise, and a solve
gave a chi2 that depended on the last bits of the parameters (the CI ABCD fit:
1.76 to 1.89 for shifts of 1e-14 sigma) or failed (two projections of one
channel). chi2_pseudo_inverse drops these directions and counts the others
as the ndf. Postfit, the covariance is the fit's response to its inputs, which
also nulls the directions the free parameters absorb (more in
tests/test_chi2_fit_response.py).
"""

import os
import tempfile

import hist
import numpy as np
import pytest
import tensorflow as tf

from rabbit import fitter, inputdata, tensorwriter
from rabbit.fitter import chi2_pseudo_inverse
from rabbit.mappings import helpers as mh
from rabbit.mappings import mapping as mp
from rabbit.param_models.helpers import load_models
from rabbit.tfhelpers import edmval_cov
from tests.test_sparse_fit import make_histograms, make_options

# --- chi2_pseudo_inverse on its own ------------------------------------------


def _spd(rng, n, cond=10.0):
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return (q * np.geomspace(1.0, 1.0 / cond, n)) @ q.T


def _solve_chi2(r, c):
    return float(r @ np.linalg.solve(c, r))


def test_regular_covariance_gives_the_solve():
    rng = np.random.default_rng(1)
    c = 7.0 * _spd(rng, 30, cond=1e3)
    r = rng.standard_normal(30)
    chi2, ndf = chi2_pseudo_inverse(r, c)
    assert ndf == 30
    np.testing.assert_allclose(chi2, _solve_chi2(r, c), rtol=1e-12)


def test_the_cutoff_does_not_depend_on_units():
    """A block of residuals with variances 1e-12 of the others (a normalized
    distribution next to event counts) is kept: the cutoff is relative to each
    residual's own variance, not to the largest variance."""
    rng = np.random.default_rng(2)
    small, large = 1e-12 * _spd(rng, 5), 1e3 * _spd(rng, 6)
    c = np.zeros((11, 11))
    c[:5, :5], c[5:, 5:] = small, large
    r = np.concatenate([1e-6 * rng.standard_normal(5), 30 * rng.standard_normal(6)])
    chi2, ndf = chi2_pseudo_inverse(r, c)
    assert ndf == 11
    np.testing.assert_allclose(chi2, _solve_chi2(r, c), rtol=1e-12)


def test_exact_identities_are_dropped_whatever_the_rounding():
    """C = A A^T of rank k < n, residuals r = A b in its range: chi2 = |b|^2,
    ndf = k, for any rounding noise on r and C."""
    rng = np.random.default_rng(3)
    n, k = 25, 19
    a = rng.standard_normal((n, k)) * rng.uniform(1, 100, n)[:, None]
    b = rng.standard_normal(k)
    c0, r0 = a @ a.T, a @ b
    values = []
    for _ in range(20):
        noise = 1e-15 * np.abs(c0).max() * rng.standard_normal((n, n))
        c = c0 + 0.5 * (noise + noise.T)
        r = r0 * (1 + 1e-15 * rng.standard_normal(n))
        chi2, ndf = chi2_pseudo_inverse(r, c)
        assert ndf == k
        values.append(chi2)
    np.testing.assert_allclose(values, b @ b, rtol=1e-9)


def test_the_rank_decides_not_the_declared_identities():
    """An identity a mapping declares (ndf_reduction) but its covariance does
    not have removes nothing: the ndf is the rank."""
    rng = np.random.default_rng(4)
    c = _spd(rng, 10)
    r = rng.standard_normal(10)
    chi2, ndf = chi2_pseudo_inverse(r, c, ndf_reduction=1)
    assert ndf == 10
    np.testing.assert_allclose(chi2, _solve_chi2(r, c), rtol=1e-12)


def test_non_finite_residuals_are_refused():
    """Not a NaN chi2 (the covariance: tests/test_chi2_singular.py)."""
    r = np.array([1.0, np.nan, 2.0])
    with pytest.raises(ValueError, match="residuals are not finite"):
        chi2_pseudo_inverse(r, np.eye(3))


def test_a_residual_without_variance_is_left_out_and_reported(caplog):
    rng = np.random.default_rng(5)
    c = np.zeros((6, 6))
    c[:5, :5] = _spd(rng, 5)
    r = np.concatenate([rng.standard_normal(5), [0.0]])
    chi2, ndf = chi2_pseudo_inverse(r, c)
    assert ndf == 5
    np.testing.assert_allclose(chi2, _solve_chi2(r[:5], c[:5, :5]), rtol=1e-12)

    r[5] = 1.0  # a bin with zero variance but a residual: not consistent
    with caplog.at_level("WARNING"):
        chi2_bad, ndf_bad = chi2_pseudo_inverse(r, c)
    assert (chi2_bad, ndf_bad) == (chi2, ndf)
    assert any("without variance" in rec.message for rec in caplog.records)


# --- through the Fitter --------------------------------------------------------


def _postfit(f):
    f.minimize()
    if f.bbstat.enabled:
        f._profile_beta()
    _, grad, hess = f.loss_val_grad_hess()
    _, cov = edmval_cov(grad, hess)
    f.cov.assign(cov)


def make_abcd_tensor(outdir, n_pt=5):
    """The CI ABCD tensor, tests/make_abcd_tensor.py: four regions of five pt
    bins, nonprompt in all of them and signal in C and D. More pt bins keep the
    events per bin."""
    rng = np.random.default_rng(42)
    ax_pt = hist.axis.Regular(n_pt, 0, 50, name="pt")
    scale = n_pt // 5

    def make_hist(n_events, mean_weight):
        h = hist.Hist(ax_pt, storage=hist.storage.Weight())
        pt = rng.uniform(0, 50, n_events)
        h.fill(pt=pt, weight=rng.normal(mean_weight, 0.1 * mean_weight, n_events))
        return h

    def make_data(sig_h, bkg_h):
        h = hist.Hist(ax_pt, storage=hist.storage.Double())
        sig = sig_h.values() if sig_h is not None else 0.0
        h.view()[:] = rng.poisson(np.maximum(sig + bkg_h.values(), 0))
        return h

    sig = {r: make_hist(5000 * scale, 1.0) for r in "CD"}
    npr = {
        "A": make_hist(8000 * scale, 2.0),
        "B": make_hist(8000 * scale, 4.0),
        "C": make_hist(6000 * scale, 1.5),
        "D": make_hist(3000 * scale, 0.75),
    }
    data = {r: make_data(sig.get(r), npr[r]) for r in "ABCD"}

    writer = tensorwriter.TensorWriter()
    for r in "ABCD":
        writer.add_channel(sig["C"].axes, f"ch_{r}")
    for r in "ABCD":
        writer.add_data(data[r], f"ch_{r}")
    for r in "CD":
        writer.add_process(sig[r], "signal", f"ch_{r}", signal=True)
    for r in "ABCD":
        writer.add_process(npr[r], "nonprompt", f"ch_{r}")
    writer.write(outfolder=outdir, outfilename=f"abcd{n_pt}")
    return os.path.join(outdir, f"abcd{n_pt}.hdf5")


@pytest.fixture(scope="module")
def abcd_fit():
    """The CI ABCD fit: --paramModel Mu --paramModel ABCD nonprompt ch_A ch_B
    ch_C ch_D, 16 free parameters for 20 bins."""
    with tempfile.TemporaryDirectory() as d:
        indata = inputdata.FitInputData(make_abcd_tensor(d))
        model = load_models(
            [["Mu"], ["ABCD", "nonprompt", "ch_A", "ch_B", "ch_C", "ch_D"]], indata
        )
        f = fitter.Fitter(indata, model, make_options())
        f.defaultassign()
        f.set_nobs(indata.data_obs)
        _postfit(f)
        yield f


def test_abcd_identities_are_not_counted(abcd_fit):
    """The six identities of the ABCD residuals (A + B in each pt bin, and the
    total) are null directions of the covariance propagated through the fit's
    covariance as well; the other ten directions the 16 free parameters absorb
    are null only in the fit response, whose rank is the saturated ndf, 4."""
    f = abcd_fit
    m = mh.load_mapping("BaseMapping", f.indata)
    res, cov = f.residuals(m.compute_flat, None, True)
    a, b, c, d = res.reshape(4, 5)
    scale = np.abs(res).max()
    np.testing.assert_allclose(a + b, 0, atol=1e-9 * scale)
    np.testing.assert_allclose(np.sum(c + d), 0, atol=1e-9 * scale)

    _, aux = f.expected_events(
        m, compute_variance=False, compute_chi2=True, profile=True
    )
    chi2, ndf = aux[-2:]
    assert ndf == 20 - f.nfreeparms == 4

    # the chi2 over the regular eigen-directions of the covariance
    lam, u = np.linalg.eigh(cov)
    keep = lam > 1e-10 * lam[-1]
    assert np.count_nonzero(keep) == 4
    np.testing.assert_allclose(
        chi2, np.sum((u.T @ res)[keep] ** 2 / lam[keep]), rtol=1e-8
    )

    res_prop, cov_prop = f._residual_terms(m.compute_flat, None, True)
    chi2_prop, ndf_prop = chi2_pseudo_inverse(res_prop.numpy(), cov_prop.numpy())
    assert ndf_prop == 20 - 6
    np.testing.assert_allclose(chi2_prop, chi2, rtol=1e-3)


def test_abcd_chi2_does_not_depend_on_rounding(abcd_fit):
    """Shifting the parameters by 1e-14 sigma moved the solved chi2 between
    1.76 and 1.89."""
    f = abcd_fit
    m = mh.load_mapping("BaseMapping", f.indata)
    fun = m.compute_flat
    x = f.x.numpy()
    sigma = np.sqrt(np.diag(f.cov.numpy()))
    rng = np.random.default_rng(6)
    results = []
    try:
        for _ in range(6):
            f.x.assign(x + 1e-14 * sigma * rng.standard_normal(x.shape))
            results.append(f.chi2(fun, None, 0, profile=True))
    finally:
        f.x.assign(x)
    chi2s, ndfs = np.array(results).T
    assert set(ndfs) == {4}
    np.testing.assert_allclose(chi2s, chi2s[0], rtol=1e-9)


@pytest.fixture(scope="module")
def fit_1d():
    """One channel of 20 bins: signal (free strength) and background with a
    norm and a shape systematic, bin-by-bin stat; fitted to fluctuated data."""
    h = make_histograms()
    with tempfile.TemporaryDirectory() as d:
        writer = tensorwriter.TensorWriter()
        writer.add_channel(h["data"].axes, "ch0")
        writer.add_data(h["data"], "ch0")
        writer.add_process(h["sig"], "sig", "ch0", signal=True)
        writer.add_process(h["bkg"], "bkg", "ch0")
        writer.add_norm_systematic("bkg_norm", "bkg", "ch0", 1.05)
        writer.add_systematic(
            [h["syst_up"], h["syst_dn"]],
            "bkg_shape",
            "bkg",
            "ch0",
            symmetrize="average",
        )
        writer.write(outfolder=d, outfilename="chi2_rank")
        indata = inputdata.FitInputData(os.path.join(d, "chi2_rank.hdf5"))
        model = load_models([["Mu"]], indata)
        f = fitter.Fitter(indata, model, make_options())
        f.defaultassign()
        nobs = np.random.default_rng(7).poisson(indata.data_obs.numpy())
        f.set_nobs(tf.constant(nobs, dtype=indata.dtype))
        _postfit(f)
        yield f


@pytest.mark.parametrize("profile", [False, True])
def test_two_groupings_of_the_same_bins(fit_1d, profile):
    """A projection next to itself, or next to its own total, adds no
    information: same chi2, same ndf. The solve failed on the duplicate.
    Postfit, the free signal strength absorbs one direction."""
    f = fit_1d
    single = mh.load_mapping("Project", f.indata, "ch0", "x")
    chi2, ndf = f.chi2(single.compute_flat, single._get_data, 0, profile)
    assert ndf == (20 - f.nfreeparms if profile else 20)
    for other in [
        mh.load_mapping("Project", f.indata, "ch0", "x"),
        mh.load_mapping("Project", f.indata, "ch0"),
    ]:
        both = mp.CompositeMapping([single, other])
        chi2_both, ndf_both = f.chi2(
            both.compute_flat, both._get_data, both.ndf_reduction, profile
        )
        assert ndf_both == ndf
        np.testing.assert_allclose(chi2_both, chi2, rtol=1e-9)


@pytest.mark.parametrize("profile", [False, True])
def test_a_regular_covariance_is_solved_as_before(fit_1d, profile):
    """Without identities nothing changes: r^T C^-1 r, ndf = n."""
    f = fit_1d
    m = mh.load_mapping("Select", f.indata, "ch0", "x:rebin(-5,-2,-1,0,1,2,5)")
    res, cov = f.residuals(m.compute_flat, m._get_data, profile)
    lam = np.linalg.eigvalsh(cov)
    assert lam[0] > 1e-4 * lam[-1]  # the test is about a well-conditioned case
    chi2, ndf = f.chi2(m.compute_flat, m._get_data, m.ndf_reduction, profile)
    assert ndf == 6
    np.testing.assert_allclose(chi2, _solve_chi2(res, cov), rtol=1e-12)


@pytest.mark.parametrize("profile", [False, True])
@pytest.mark.parametrize(
    "args,declared,ndf",
    [
        # sums to one: one null direction, as its ndf_reduction declares
        (["Normalize", "ch0", "x"], 1, 19),
        # a ratio of two normalized distributions obeys no fixed linear
        # identity: its covariance is regular (it declared one until the ndf
        # became the rank)
        (["Normratio", "ch0", "ch0", "x:slice(0,10)", "x:slice(10,20)"], 0, 10),
    ],
)
def test_declared_identities(fit_1d, profile, args, declared, ndf):
    """The value is the one of the previous pseudo-inverse, the ndf the rank."""
    f = fit_1d
    m = mh.load_mapping(args[0], f.indata, *args[1:])
    assert m.ndf_reduction == declared
    res, cov = f.residuals(m.compute_flat, m._get_data, profile)
    pinv_chi2 = float(res @ np.linalg.pinv(cov) @ res)
    chi2, ndf_found = f.chi2(m.compute_flat, m._get_data, m.ndf_reduction, profile)
    assert ndf_found == ndf
    np.testing.assert_allclose(chi2, pinv_chi2, rtol=1e-9)
