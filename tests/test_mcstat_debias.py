"""Limited-MC-statistics de-biasing: continuous-M, two-half and k-fold.

Each test builds a small toy tensor (module-level ``make_*_toy`` helpers, written
into pytest's tmp_path), fits it the way ``rabbit_fit.py`` would, and asserts the
de-biased point / curvature / sandwich either against an independent closed-form
numpy reference or against the documented qualitative behaviour:

* continuous-M (``add_mc_stat_moment``): with a LINEAR POI the point and
  covariance equal the closed form (H - M)^-1, inflating sigma; with rabbit's
  default x^2 POI the correction is cancelled (docs/mcstat/RESULTS.md §9a).
* two-half (k=2 folds): point, curvature and sandwich equal the numpy jackknife.
* k-fold: the --covMode fisher curvature equals the complete pairwise
  U-statistic A_k = k/(k-1) [F_full - sum_i F_i]; over an ensemble it is unbiased
  and its scatter shrinks as k grows.
* composition with bin-by-bin stat (BB-lite) converges positive-definite.
* sparse-mode fold storage gives exactly the dense result.
* a small fixed-seed template-fluctuating coverage ensemble
  (tests/mcstat_coverage.py) shows the de-bias direction.

The toy builders are parameterised by the number of folds ``k``, ``sparse`` and
``systematic_type`` so further configurations can reuse them.
"""

import functools
import importlib.util
import os
from pathlib import Path

import hist
import numpy as np
import pytest
import scipy.stats
import tensorflow as tf

from rabbit import fitter, inputdata
from rabbit.param_models import helpers as ph
from rabbit.tensorwriter import TensorWriter

REPO = Path(__file__).resolve().parents[1]

# The near-degenerate 2-process toy of the MC-stat studies: a flat process and a
# "step" process equal to it in the 2nd half and +3% in the 1st half, with
# finite (1:1) MC templates.
DEGEN_NBINS = 200
DEGEN_FLAT, DEGEN_STEP_HI, DEGEN_STEP_LO = 4962.0, 5112.0, 4962.0
DEGEN_SEED = 20240614

# POI directions: the degenerate difference and the well-constrained sum.
DDIF = np.array([1.0, -1.0]) / np.sqrt(2.0)

# Linear POIs, Gaussian (chisq) likelihood, no bin-by-bin stat: the setting in
# which the closed-form numpy references apply.
LINEAR_CHISQ = ("--chisqFit", "--noBinByBinStat", "--allowNegativeParam")


# ---------------------------------------------------------------------------
# fit driver
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _driver():
    """bin/rabbit_fit.py as an importable module (for its make_parser())."""
    spec = importlib.util.spec_from_file_location(
        "_rabbit_fit_for_mcstat_tests", REPO / "bin" / "rabbit_fit.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fit_toy(filename, outdir, *args):
    """Build and minimize a Fitter as `rabbit_fit.py filename -t 0 *args` would.

    Returns the minimized Fitter; callers take the loss/grad/hess and the
    de-biased covariances from it.
    """
    options = (
        _driver()
        .make_parser()
        .parse_args([str(filename), "-o", str(outdir), "-t", "0", *args])
    )
    indata = inputdata.FitInputData(str(filename), None)
    param_model = ph.load_models([["Mu"]], indata, **vars(options))
    f = fitter.Fitter(indata, param_model, options, do_blinding=False)
    f.defaultassign()
    f.set_nobs(indata.data_obs)
    f.minimize()
    return f


def sigma(cov, direction=DDIF):
    return float(np.sqrt(direction @ np.asarray(cov) @ direction))


# ---------------------------------------------------------------------------
# toy builders
# ---------------------------------------------------------------------------


def _hist1d(ax, values, variances=None):
    h = hist.Hist(ax, storage=hist.storage.Weight())
    h.view()["value"] = values
    h.view()["variance"] = values if variances is None else variances
    return h


def _fold_hist(ax, folds):
    """[k, nbins] fold counts -> hist with an `mcfold` axis (unit weights)."""
    k = folds.shape[0]
    axf = hist.axis.IntCategory(list(range(k)), name="mcfold")
    h = hist.Hist(ax, axf, storage=hist.storage.Weight())
    for i in range(k):
        h.view()["value"][:, i] = folds[i]
        h.view()["variance"][:, i] = folds[i]
    return h


def split_into_folds(rng, counts, k):
    """Assign each bin's integer counts to k folds, multinomial(1/k): [k, nbins]."""
    return rng.multinomial(counts, [1.0 / k] * k).T


def _write(tw, path, name):
    tw.write(outfolder=str(path), outfilename=name)
    return os.path.join(str(path), name)


def degenerate_toy_arrays(k=None, seed=DEGEN_SEED):
    """True yields, Asimov data and one finite-MC realization of the degenerate
    toy; with `k`, also each process's split into k folds.

    Returns a dict with ``n_true`` [2, nb], ``data`` [nb], ``sumw`` [2, nb]
    (unit-weight counts, so sumw2 == sumw) and ``folds`` [2, k, nb] (or None).
    """
    nb = DEGEN_NBINS
    rng = np.random.default_rng(seed)
    n_flat = np.full(nb, DEGEN_FLAT)
    n_step = np.concatenate(
        [np.full(nb // 2, DEGEN_STEP_HI), np.full(nb // 2, DEGEN_STEP_LO)]
    )
    sumw = np.stack([rng.poisson(n_flat), rng.poisson(n_step)])
    folds = None
    if k is not None:
        folds = np.stack([split_into_folds(rng, c, k) for c in sumw])
    return dict(
        n_true=np.stack([n_flat, n_step]),
        data=n_flat + n_step,
        sumw=sumw.astype(float),
        folds=folds,
    )


def make_mcstat_moment_toy(path, with_M=True, bbb_variance=False, seed=DEGEN_SEED):
    """Degenerate toy with plain finite-MC templates and, if `with_M`, the frozen
    noise-floor matrix M over the two POIs (continuous-M de-biasing).

    M_jj = sum_b sumw2[b, j] / var_b with var_b = data_b (chisq), or with the
    BB-lite-inflated var_b = data_b + sum_proc sumw2[b] if `bbb_variance`.
    Returns (filename, arrays) with ``arrays["M"]`` set (None without M).
    """
    arr = degenerate_toy_arrays(seed=seed)
    ax = hist.axis.Regular(DEGEN_NBINS, 0.0, 1.0, name="x")
    data, sumw = arr["data"], arr["sumw"]
    tw = TensorWriter(sparse=False)
    tw.add_channel([ax], "ch0")
    tw.add_data(_hist1d(ax, data), "ch0")
    tw.add_process(_hist1d(ax, sumw[0]), "flat", "ch0", signal=True)
    tw.add_process(_hist1d(ax, sumw[1]), "step", "ch0", signal=True)
    arr["M"] = None
    if with_M:
        var = data + sumw.sum(0) if bbb_variance else data
        arr["M"] = np.diag(np.sum(sumw / var, axis=1))
        tw.add_mc_stat_moment(
            arr["M"],
            ["flat", "step"],
            variance="bb_inflated" if bbb_variance else "bare",
        )
    name = f"toy_{'M' if with_M else 'noM'}{'_bb' if bbb_variance else ''}.hdf5"
    return _write(tw, path, name), arr


def make_twohalf_toy(
    path, k=2, sparse=False, systematic_type="log_normal", seed=DEGEN_SEED
):
    """Degenerate toy whose two processes (flat, step) carry k MC-stat folds
    (add_process(fold_axis=...)); the full template is the sum of the folds.
    k=2 is the two-half jackknife, k>2 the k-fold U-statistic.
    Returns (filename, arrays) as degenerate_toy_arrays(k)."""
    arr = degenerate_toy_arrays(k=k, seed=seed)
    ax = hist.axis.Regular(DEGEN_NBINS, 0.0, 1.0, name="x")
    tw = TensorWriter(sparse=sparse, systematic_type=systematic_type)
    tw.add_channel([ax], "ch0")
    tw.add_data(_hist1d(ax, arr["data"]), "ch0")
    for name, folds in zip(("flat", "step"), arr["folds"]):
        tw.add_process(
            _fold_hist(ax, folds), name, "ch0", signal=True, fold_axis="mcfold"
        )
    fname = f"toy_folds_k{k}_{'sp' if sparse else 'de'}_{systematic_type}.hdf5"
    return _write(tw, path, fname), arr


def make_folded_syst_toy(
    path,
    k=2,
    sparse=False,
    split=True,
    systematic_type="log_normal",
    nbins=50,
    nominal_range=(800.0, 1200.0),
    seed=7,
):
    """One folded signal process 'sig' (k folds) with a 'tilt' systematic whose
    up-variation reweights each fold's OWN events, so its logk carries fold-
    specific MC noise.

    split=True adds the systematic with fold_axis (split logk per fold);
    split=False adds the summed up template without folds (shared logk).
    Asimov data = the smooth nominal. Returns the filename.
    """
    rng = np.random.default_rng(seed)
    nom = np.linspace(*nominal_range, nbins)
    folds = split_into_folds(rng, rng.poisson(nom), k)
    w = 1.0 + 0.15 * (np.linspace(0.0, 1.0, nbins) - 0.5)
    upfolds = np.stack([rng.poisson(np.maximum(folds[i] * w, 0.0)) for i in range(k)])
    ax = hist.axis.Regular(nbins, 0.0, 1.0, name="x")
    tw = TensorWriter(sparse=sparse, systematic_type=systematic_type)
    tw.add_channel([ax], "ch0")
    tw.add_data(_hist1d(ax, nom), "ch0")
    tw.add_process(_fold_hist(ax, folds), "sig", "ch0", signal=True, fold_axis="mcfold")
    if split:
        tw.add_systematic(
            _fold_hist(ax, upfolds), "tilt", "sig", "ch0", fold_axis="mcfold"
        )
    else:
        tw.add_systematic(_hist1d(ax, upfolds.sum(0)), "tilt", "sig", "ch0")
    name = (
        f"toy_{'split' if split else 'shared'}logk_k{k}_nb{nbins}_s{seed}_"
        f"{'sp' if sparse else 'de'}_{systematic_type}.hdf5"
    )
    return _write(tw, path, name)


def make_folded_process_toy(
    path, k=4, sparse=False, systematic_type="log_normal", nbins=40, seed=13
):
    """One folded signal process 'sig' (k folds) with an ordinary (unfolded,
    shared-logk) linear shape systematic 'shp'. Asimov data = the nominal.
    Returns the filename."""
    rng = np.random.default_rng(seed)
    nom = np.linspace(900.0, 1100.0, nbins)
    folds = split_into_folds(rng, rng.poisson(nom), k)
    x = np.linspace(0.0, 1.0, nbins)
    ax = hist.axis.Regular(nbins, 0.0, 1.0, name="x")
    tw = TensorWriter(sparse=sparse, systematic_type=systematic_type)
    tw.add_channel([ax], "ch0")
    tw.add_data(_hist1d(ax, nom), "ch0")
    tw.add_process(_fold_hist(ax, folds), "sig", "ch0", signal=True, fold_axis="mcfold")
    tw.add_systematic(_hist1d(ax, nom * (1 + 0.1 * (x - 0.5))), "shp", "sig", "ch0")
    name = f"toy_foldproc_k{k}_{'sp' if sparse else 'de'}_{systematic_type}.hdf5"
    return _write(tw, path, name)


# ---------------------------------------------------------------------------
# numpy references
# ---------------------------------------------------------------------------


def _fisher(T, V):
    """sum_b T[b,i] T[b,j] / V[b] for templates T [nbins, nparams]."""
    return np.einsum("bi,b,bj->ij", T, 1.0 / V, T)


def kfold_ustatistic(fold_templates, V):
    """Complete pairwise U-statistic curvature A_k = k/(k-1) [F_full - sum F_i]
    for per-fold templates [k][nbins, nparams] (full = sum of folds)."""
    k = len(fold_templates)
    full = sum(fold_templates)
    return (k / (k - 1.0)) * (
        _fisher(full, V) - sum(_fisher(T, V) for T in fold_templates)
    )


# ---------------------------------------------------------------------------
# continuous-M (add_mc_stat_moment)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def moment_toys(tmp_path_factory):
    path = tmp_path_factory.mktemp("mcstat_moment")
    fn_noM, _ = make_mcstat_moment_toy(path, with_M=False)
    fn_M, arr = make_mcstat_moment_toy(path, with_M=True)
    return dict(noM=fn_noM, M=fn_M, arrays=arr, out=path)


def _continuous_m_fit(filename, outdir, linear, debias):
    """Fit with --mcStatDebias `debias` and return (rnorm, curvature cov,
    sandwich cov or None) in rnorm space."""
    args = ("--chisqFit", "--noBinByBinStat", "--mcStatDebias", debias)
    if linear:
        args += ("--allowNegativeParam",)
    f = fit_toy(filename, outdir, *args)
    _, grad, hess = f.loss_val_grad_hess()
    _, cov = f.edmval_cov(grad, hess)
    sand = None
    if f.mcstat_M is not None:
        sand = np.asarray(f.cov_mcstat_sandwich(hess))  # A^-1 + A^-1 M A^-1
    x = f.x.numpy()
    jac = np.diag(np.ones_like(x) if linear else 2.0 * x)  # d rnorm / d x
    rnorm = x if linear else x**2
    return (
        rnorm,
        jac @ np.asarray(cov) @ jac,
        None if sand is None else jac @ sand @ jac,
    )


def test_continuous_m_linear_poi_debiases(moment_toys):
    """Linear POI: the -1/2 theta^T M theta penalty gives the closed-form
    de-biased point (H - M)^-1 b, curvature (H - M)^-1 and sandwich
    A^-1 H A^-1, inflating sigma(dif) by > 20% (the scripts' DE-BIASED verdict)."""
    out = moment_toys["out"]
    r_no, C_no, S_no = _continuous_m_fit(moment_toys["noM"], out, True, "none")
    r_M, C_M, S_M = _continuous_m_fit(moment_toys["M"], out, True, "continuousM")
    assert S_no is None and S_M is not None

    arr = moment_toys["arrays"]
    T, V, d = arr["sumw"].T, arr["data"], arr["data"]
    H = _fisher(T, V)
    b = np.einsum("bi,b,b->i", T, 1.0 / V, d)
    A = H - arr["M"]
    Ainv = np.linalg.inv(A)
    np.testing.assert_allclose(r_no, np.linalg.solve(H, b), atol=1e-3)
    np.testing.assert_allclose(r_M, np.linalg.solve(A, b), atol=1e-3)
    assert sigma(C_no) == pytest.approx(sigma(np.linalg.inv(H)), abs=1e-3)
    assert sigma(C_M) == pytest.approx(sigma(Ainv), abs=1e-3)
    assert sigma(S_M) == pytest.approx(sigma(Ainv @ H @ Ainv), abs=1e-3)

    s_no, s_M, s_sand = sigma(C_no), sigma(C_M), sigma(S_M)
    assert s_M > 1.2 * s_no, f"M should inflate sigma(dif): {s_no} -> {s_M}"
    assert s_sand >= s_M, f"sandwich {s_sand} below curvature {s_M}"


def test_continuous_m_squared_poi_is_cancelled(moment_toys):
    """Default x^2 POI transform: the M correction is cancelled by Fisher growth
    at the shifted minimum (docs/mcstat/RESULTS.md §9a), so the curvature sigma(dif) does NOT
    inflate (the scripts' 'no de-bias (cancelled)' verdict). The sandwich still
    adds the A^-1 M A^-1 margin on top of the curvature."""
    out = moment_toys["out"]
    r_no, C_no, _ = _continuous_m_fit(moment_toys["noM"], out, False, "none")
    r_M, C_M, S_M = _continuous_m_fit(moment_toys["M"], out, False, "continuousM")
    s_no, s_M = sigma(C_no), sigma(C_M)
    assert s_M < 1.2 * s_no, f"x^2 POI should cancel the M de-bias: {s_no} -> {s_M}"
    np.testing.assert_allclose(r_M, r_no, atol=1e-3)
    assert sigma(S_M) >= s_M


# ---------------------------------------------------------------------------
# two-half (k=2 folds)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def twohalf_toy(tmp_path_factory):
    path = tmp_path_factory.mktemp("mcstat_twohalf")
    fn, arr = make_twohalf_toy(path, k=2)
    return dict(fn=fn, arrays=arr, out=path)


def twohalf_numpy_reference(arr):
    """Closed-form standard and jackknife (two-half) fit of the linear 2-POI toy:
    A = 2 H_full - (H_A + H_B)/2 with half templates scaled x2, sandwich
    A^-1 H_full A^-1. Returns (r_std, C_std, r_cf, C_curv, C_sand)."""
    data = arr["data"]
    V = data
    Tf = arr["sumw"].T
    TA = 2.0 * arr["folds"][:, 0, :].T
    TB = 2.0 * arr["folds"][:, 1, :].T

    def b(T):
        return np.einsum("bi,b,b->i", T, 1.0 / V, data)

    Hf = _fisher(Tf, V)
    A_cf = 2.0 * Hf - 0.5 * _fisher(TA, V) - 0.5 * _fisher(TB, V)
    b_cf = 2.0 * b(Tf) - 0.5 * b(TA) - 0.5 * b(TB)
    Ccurv = np.linalg.inv(A_cf)
    return (
        np.linalg.solve(Hf, b(Tf)),
        np.linalg.inv(Hf),
        np.linalg.solve(A_cf, b_cf),
        Ccurv,
        Ccurv @ Hf @ Ccurv,
    )


def _twohalf_fit(filename, outdir, debias):
    f = fit_toy(filename, outdir, *LINEAR_CHISQ, "--mcStatDebias", debias)
    _, grad, hess = f.loss_val_grad_hess()
    _, cov = f.edmval_cov(grad, hess)
    sand = None
    if debias in ("twoHalf", "kfold"):
        sand = np.asarray(f.cov_twohalf_sandwich(hess))
    return f.x.numpy(), np.asarray(cov), sand


def test_twohalf_matches_numpy_reference(twohalf_toy):
    """rabbit's two-half point, curvature and sandwich equal the independent numpy
    jackknife (atol 1e-3); the standard fit equals the plain least squares."""
    r_std, C_std, r_cf, C_curv, C_sand = twohalf_numpy_reference(twohalf_toy["arrays"])
    x_std, Cr_std, _ = _twohalf_fit(twohalf_toy["fn"], twohalf_toy["out"], "none")
    x_cf, Cr_curv, Cr_sand = _twohalf_fit(
        twohalf_toy["fn"], twohalf_toy["out"], "twoHalf"
    )

    np.testing.assert_allclose(x_std, r_std, atol=1e-3)
    assert sigma(Cr_std) == pytest.approx(sigma(C_std), abs=1e-3)

    np.testing.assert_allclose(x_cf, r_cf, atol=1e-3)
    assert sigma(Cr_curv) == pytest.approx(sigma(C_curv), abs=1e-3)
    assert sigma(Cr_sand) == pytest.approx(sigma(C_sand), abs=1e-3)
    # the de-bias is active: the point moved away from the attenuated fit
    assert not np.allclose(x_cf, x_std, atol=1e-2)


# ---------------------------------------------------------------------------
# k-fold U-statistic curvature
# ---------------------------------------------------------------------------


def test_kfold_fisher_curvature_matches_ustatistic(tmp_path_factory):
    """--mcStatDebias kfold --covMode fisher curvature on the k=4 fold toy equals
    the numpy complete pairwise U-statistic A_k (rtol 1e-4)."""
    k = 4
    path = tmp_path_factory.mktemp("mcstat_kfold")
    fn, arr = make_twohalf_toy(path, k=k)
    f = fit_toy(
        fn, path, *LINEAR_CHISQ, "--mcStatDebias", "kfold", "--covMode", "fisher"
    )
    _, _, hess = f.loss_val_grad_hess()
    curv = np.asarray(f.fisher_curvature(hess))

    ref = kfold_ustatistic(
        [arr["folds"][:, i, :].T.astype(float) for i in range(k)], arr["data"]
    )
    assert f.n_folds == k
    np.testing.assert_allclose(curv, ref, rtol=1e-4)


def _ustatistic_sigma_ensemble(k, ntoy, seed):
    """sigma(dif) from the U-statistic curvature over `ntoy` fresh MC
    realizations of the degenerate toy split into k folds (pure numpy)."""
    rng = np.random.default_rng(seed)
    base = degenerate_toy_arrays()
    sig = []
    for _ in range(ntoy):
        sumw = [rng.poisson(n) for n in base["n_true"]]
        folds = [split_into_folds(rng, c, k) for c in sumw]  # [2][k, nb]
        Ti = [np.stack([folds[0][i], folds[1][i]], 1).astype(float) for i in range(k)]
        v = DDIF @ np.linalg.inv(kfold_ustatistic(Ti, base["data"])) @ DDIF
        if v > 0:
            sig.append(np.sqrt(v))
    return np.median(sig), np.std(sig)


def test_kfold_ustatistic_unbiased_and_variance_reducing():
    """Over an ensemble (300 toys per k, fixed seeds) the U-statistic sigma(dif)
    has median within 0.02 of sigma_inf for k=4, 8, 16, and its RMS does not
    increase with k (k-fold averaging approaches the continuous-M floor)."""
    base = degenerate_toy_arrays()
    sinf = sigma(np.linalg.inv(_fisher(base["n_true"].T, base["data"])))
    prev = None
    for k in (4, 8, 16):
        med, rms = _ustatistic_sigma_ensemble(k, ntoy=300, seed=100 + k)
        assert abs(med - sinf) < 0.02, f"k={k}: median {med} vs sigma_inf {sinf}"
        if prev is not None:
            assert rms < prev + 1e-9, f"k={k}: RMS {rms} grew from {prev}"
        prev = rms


# ---------------------------------------------------------------------------
# composition with bin-by-bin stat (BB-lite)
# ---------------------------------------------------------------------------


def test_continuous_m_with_bblite_converges_pd(tmp_path):
    """continuous-M + BB-lite with M built on the BB-lite-inflated variance
    (data + sum sumw2): the fit converges (|grad| < 1e-3) to a positive-definite
    Hessian."""
    fn, _ = make_mcstat_moment_toy(tmp_path, with_M=True, bbb_variance=True)
    f = fit_toy(
        fn,
        tmp_path,
        "--chisqFit",
        "--allowNegativeParam",
        "--mcStatDebias",
        "continuousM",
    )
    _, grad, hess = f.loss_val_grad_hess()
    eig = np.linalg.eigvalsh(hess.numpy())
    gmax = float(np.max(np.abs(grad.numpy())))
    assert np.all(eig > 0), f"Hessian not PD: eigenvalues {eig}"
    assert gmax < 1e-3, f"not converged: max|grad| = {gmax}"
    S = np.asarray(f.cov_mcstat_sandwich(hess))
    assert np.all(np.isfinite(S)) and np.all(np.diag(S) > 0)


def test_twohalf_with_bblite_inflates_tilt(tmp_path):
    """two-half + BB-lite on the k=2 split-logk toy: the de-biased fit converges
    PD and its sandwich uncertainty on the folded 'tilt' systematic exceeds the
    un-de-biased one."""
    fn = make_folded_syst_toy(tmp_path, k=2, split=True)
    common = ("--chisqFit", "--allowNegativeParam", "--mcStatDebias")

    f0 = fit_toy(fn, tmp_path, *common, "none")
    _, g0, h0 = f0.loss_val_grad_hess()
    _, cov0 = f0.edmval_cov(g0, h0)
    sig0 = np.sqrt(np.diag(np.asarray(cov0)))

    f1 = fit_toy(fn, tmp_path, *common, "twoHalf")
    _, g1, h1 = f1.loss_val_grad_hess()
    eig = np.linalg.eigvalsh(h1.numpy())
    gmax = float(np.max(np.abs(g1.numpy())))
    S1 = np.asarray(f1.cov_twohalf_sandwich(h1, "observed"))

    itilt = list(f1.parms.astype(str)).index("tilt")
    assert np.all(eig > 0), f"Hessian not PD: eigenvalues {eig}"
    assert gmax < 1e-3, f"not converged: max|grad| = {gmax}"
    assert np.sqrt(S1[itilt, itilt]) > sig0[itilt], (
        f"two-half should inflate the tilt uncertainty: {sig0[itilt]} -> "
        f"{np.sqrt(S1[itilt, itilt])}"
    )


# ---------------------------------------------------------------------------
# sparse-mode fold storage == dense
# ---------------------------------------------------------------------------

SPARSE_TOYS = {
    # folded process, shared (unfolded) systematic: sparse per-fold yield path
    "folded_process": lambda path, sparse: make_folded_process_toy(
        path, k=4, sparse=sparse, nbins=40, seed=13
    ),
    # folded process AND folded (split-logk) systematic: sparse exp(delta.theta)
    "split_logk": lambda path, sparse: make_folded_syst_toy(
        path,
        k=4,
        sparse=sparse,
        split=True,
        nbins=40,
        nominal_range=(900.0, 1100.0),
        seed=21,
    ),
}


def _kfold_fisher_fit(filename, outdir):
    f = fit_toy(
        filename,
        outdir,
        *LINEAR_CHISQ,
        "--mcStatDebias",
        "kfold",
        "--covMode",
        "fisher",
    )
    _, _, hess = f.loss_val_grad_hess()
    bread = f.fisher_curvature(hess)
    sand = np.asarray(f.cov_twohalf_sandwich(bread, "fisher"))
    return f.indata.sparse, f.x.numpy(), np.asarray(bread), sand


@pytest.mark.parametrize("toy", sorted(SPARSE_TOYS))
def test_sparse_fold_debias_equals_dense(toy, tmp_path):
    """k=4 kfold/fisher on the same toy written sparse and dense: identical
    de-biased point, fisher curvature and sandwich (rtol 1e-6)."""
    build = SPARSE_TOYS[toy]
    is_sp, x_sp, b_sp, S_sp = _kfold_fisher_fit(build(tmp_path, True), tmp_path)
    is_de, x_de, b_de, S_de = _kfold_fisher_fit(build(tmp_path, False), tmp_path)
    assert is_sp and not is_de
    np.testing.assert_allclose(x_sp, x_de, rtol=1e-6)
    np.testing.assert_allclose(b_sp, b_de, rtol=1e-6)
    np.testing.assert_allclose(S_sp, S_de, rtol=1e-6)


# ---------------------------------------------------------------------------
# template-fluctuating coverage ensemble (tests/mcstat_coverage.py)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def coverage_harness():
    from tests import mcstat_coverage

    # the assertions below are calibrated on the harness's default toy
    if (mcstat_coverage.NB, mcstat_coverage.HI) != (200, 5112.0) or list(
        mcstat_coverage.RTRUE
    ) != [1.3, 0.7]:
        pytest.skip("NB/HI/R0/R1 overridden in the environment")
    return mcstat_coverage


def _check_coverage_result(r, ntoy):
    assert r["n"] + r["nbad"] == ntoy
    assert r["nbad"] <= ntoy // 10, f"{r['nbad']} of {ntoy} toy fits failed"
    for key in ("cov_dif", "cov_sum"):
        assert np.isfinite(r[key]) and 0.0 <= r[key] <= 1.0, (key, r[key])
    for key in ("bias_dif", "mean_res", "rms_res", "med_sig"):
        assert np.isfinite(r[key]), (key, r[key])
    assert r["med_sig"] > 0 and r["rms_res"] > 0


# 20 toys per method instead of the standalone tool's 50 to keep CI fast (~1 s per
# toy fit). With the fixed seed the result is deterministic; the statistical margin
# of the bias check is the smallest (the de-biased median residual has a scatter of
# ~0.1 at this size against the 0.4*|naive bias| ~ 0.11 threshold), so run
# tests/mcstat_coverage.py with a larger ensemble when changing the method.
COVERAGE_NTOY = 20
COVERAGE_SEED = 11


def test_coverage_continuous_m_corrects_naive(coverage_harness, tmp_path):
    """Template+data-fluctuating ensemble on the near-degenerate toy with a
    nonzero degenerate-direction truth (rnorm_true = [1.3, 0.7]):

    1. the naive fit is attenuated (median dif residual < -0.10);
    2. continuous-M removes the central-value bias (|bias| < 0.4 |naive bias|);
    3. its sandwich sigma(dif) is > 1.3x the naive one;
    4. it improves the dif coverage by > 0.10.

    Asserts the de-bias DIRECTION, not exact 0.683 coverage: the raw sandwich
    still underestimates the de-biased point's scatter (docs/mcstat/RESULTS.md §9n).
    """
    mc = coverage_harness
    kw = dict(ntoy=COVERAGE_NTOY, seed=COVERAGE_SEED, outdir=str(tmp_path))
    none = mc.run_coverage(debias="none", **kw)
    cm = mc.run_coverage(debias="continuousM", debiasCov="sandwich", **kw)
    _check_coverage_result(none, COVERAGE_NTOY)
    _check_coverage_result(cm, COVERAGE_NTOY)

    assert none["bias_dif"] < -0.10, f"naive should be attenuated: {none}"
    assert abs(cm["bias_dif"]) < 0.4 * abs(
        none["bias_dif"]
    ), f"de-bias should remove the bias: {none['bias_dif']} -> {cm['bias_dif']}"
    assert (
        cm["med_sig"] > 1.3 * none["med_sig"]
    ), f"de-bias should inflate sigma: {none['med_sig']} -> {cm['med_sig']}"
    assert (
        cm["cov_dif"] > none["cov_dif"] + 0.10
    ), f"de-bias should improve coverage: {none['cov_dif']} -> {cm['cov_dif']}"


@pytest.mark.parametrize(
    "config",
    [
        dict(debias="twoHalf"),
        dict(debias="kfold", covMode="fisher", k=4),
        dict(debias="continuousM", bbb=True),
    ],
    ids=["twoHalf", "kfold4-fisher", "continuousM-bblite"],
)
def test_coverage_harness_smoke(coverage_harness, tmp_path, config):
    """The harness runs the fold / BB-lite configurations end to end and returns
    finite coverage numbers in [0, 1] (tiny ensemble: smoke test only)."""
    ntoy = 3
    r = coverage_harness.run_coverage(
        ntoy=ntoy, seed=COVERAGE_SEED, outdir=str(tmp_path), **config
    )
    _check_coverage_result(r, ntoy)


# ---------------------------------------------------------------------------
# any k (odd k), additive systematics, sparse storage, M conventions
# ---------------------------------------------------------------------------

SMALL_NBINS = 60


def make_small_fold_toy(
    path, k, sparse=False, systematic_type="log_normal", syst=0, exempt=False
):
    """Well-conditioned 2-process toy with k MC-stat folds and optionally one
    SHARED (unfolded) shape systematic on `flat`, and (exempt) a third,
    unfolded background with zero MC variance (MC-stat-exempt, e.g.
    data-driven). Returns (filename, arrays)."""
    rng = np.random.default_rng(3)
    nb = SMALL_NBINS
    n_flat = np.full(nb, 400.0)
    n_step = np.linspace(100.0, 700.0, nb)
    folds = np.stack(
        [
            split_into_folds(rng, rng.poisson(n), k).astype(float)
            for n in (n_flat, n_step)
        ]
    )  # [2, k, nb]
    data = n_flat + n_step
    ax = hist.axis.Regular(nb, 0.0, 1.0, name="x")
    tw = TensorWriter(sparse=sparse, systematic_type=systematic_type)
    tw.add_channel([ax], "ch0")
    tw.add_data(_hist1d(ax, data), "ch0")
    for name, f in zip(("flat", "step"), folds):
        tw.add_process(_fold_hist(ax, f), name, "ch0", signal=True, fold_axis="mcfold")
    svar = None
    if syst:
        # additive variation (absolute, = logk for systematic_type "normal")
        svar = 0.05 * folds[0].sum(0) * np.linspace(-1.0, 1.0, nb)
        tw.add_systematic(
            _hist1d(ax, folds[0].sum(0) + svar), "s0", "flat", "ch0", mirror=True
        )
    if exempt:
        tw.add_process(
            _hist1d(ax, np.full(nb, 150.0), variances=np.zeros(nb)), "exempt", "ch0"
        )
    fname = f"small_k{k}_{int(sparse)}_{systematic_type}_{syst}_{int(exempt)}.hdf5"
    return _write(tw, path, fname), dict(folds=folds, data=data, svar=svar)


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("k", [2, 3, 5])
def test_any_k_matches_numpy(k, sparse, tmp_path):
    """Any k >= 2, odd included. The halves are G_A = first a = k//2 folds and
    G_B = the other b = k - a, rescaled to full normalization, and the jackknife
    L_cf = k^2/(2ab) L_full - a/(2b) L_A - b/(2a) L_B. Its Hessian is exactly the
    symmetrized cross term J_A^T D J_B and its point solves
    cross . x = 1/2 (J_A + J_B)^T D y; in fisher mode the curvature is the k-fold
    U-statistic."""
    fn, arr = make_small_fold_toy(tmp_path, k, sparse=sparse)
    Ti = [np.stack([arr["folds"][0][i], arr["folds"][1][i]], 1) for i in range(k)]
    y = arr["data"]
    D = 1.0 / y
    a, b = k // 2, k - k // 2
    TA, TB, Tf = (k / a) * sum(Ti[:a]), (k / b) * sum(Ti[a:]), sum(Ti)
    cross = 0.5 * (TA.T @ (D[:, None] * TB) + TB.T @ (D[:, None] * TA))
    x_ref = np.linalg.solve(cross, 0.5 * (TA + TB).T @ (D * y))
    Hfull = _fisher(Tf, y)

    f = fit_toy(fn, tmp_path, *LINEAR_CHISQ, "--mcStatDebias", "kfold")
    _, _, h = f.loss_val_grad_hess()
    np.testing.assert_allclose(f.x.numpy(), x_ref, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(h), cross, rtol=1e-8)
    Ci = np.linalg.inv(cross)
    np.testing.assert_allclose(
        np.asarray(f.cov_twohalf_sandwich(h)), Ci @ Hfull @ Ci, rtol=1e-6
    )

    f = fit_toy(
        fn, tmp_path, *LINEAR_CHISQ, "--mcStatDebias", "kfold", "--covMode", "fisher"
    )
    _, _, h = f.loss_val_grad_hess()
    U = kfold_ustatistic(Ti, y)
    A = np.asarray(f.fisher_curvature(h))
    np.testing.assert_allclose(A, U, rtol=1e-8)
    Ui = np.linalg.inv(U)
    np.testing.assert_allclose(
        np.asarray(f.cov_twohalf_sandwich(A, covMode="fisher")),
        Ui @ Hfull @ Ui,
        rtol=1e-6,
    )


@pytest.mark.parametrize("expect", [1.0, 2.0])
def test_kfold_fisher_shared_additive_systematic(expect, tmp_path):
    """A SHARED additive ('normal') systematic is split evenly over the raw folds
    (1/k each, so the folds sum to the full variation); the U-statistic must then
    treat it as MC-stat-exempt. The full variation per fold used to over-count the
    systematic's curvature by k (off-diagonal) and k^2 (diagonal). With
    --expectSignal != 1 the variation also carries the rnorm_init factor."""
    k = 4
    fn, arr = make_small_fold_toy(tmp_path, k, systematic_type="normal", syst=1)
    y = arr["data"]
    f = fit_toy(
        fn,
        tmp_path,
        *LINEAR_CHISQ,
        "--mcStatDebias",
        "kfold",
        "--covMode",
        "fisher",
        "--expectSignal",
        "flat",
        str(expect),
        "--expectSignal",
        "step",
        str(expect),
    )
    _, _, h = f.loss_val_grad_hess()
    Ti = [
        np.stack([arr["folds"][0][i], arr["folds"][1][i], expect * arr["svar"] / k], 1)
        for i in range(k)
    ]
    U = kfold_ustatistic(Ti, y)
    U[2, 2] += 1.0  # unit Gaussian constraint of s0
    np.testing.assert_allclose(np.asarray(f.fisher_curvature(h)), U, rtol=1e-8)


def test_sparse_fold_storage_follows_norm_sparsity(tmp_path):
    """In sparse mode the fold templates are stored as [k, nnz] values on the
    sparsity pattern of the nominal norm, not densified to [k, nbins, nproc]."""
    import h5py

    k = 4
    rng = np.random.default_rng(1)
    nb, nproc = 120, 6
    ax = hist.axis.Regular(nb, 0.0, 1.0, name="x")
    tw = TensorWriter(sparse=True)
    tw.add_channel([ax], "ch0")
    tw.add_data(_hist1d(ax, np.full(nb, 100.0)), "ch0")
    for p in range(nproc):
        counts = np.zeros(nb, dtype=int)
        lo = p * nb // nproc
        counts[lo : lo + nb // nproc] = rng.poisson(50.0, nb // nproc)
        tw.add_process(
            _fold_hist(ax, split_into_folds(rng, counts, k).astype(float)),
            f"p{p}",
            "ch0",
            signal=True,
            fold_axis="mcfold",
        )
    fn = _write(tw, tmp_path, "sparse_folds.hdf5")
    with h5py.File(fn) as h5:
        nnz = h5["hnorm_sparse"]["values"].shape[0]
        folds = h5["hnorm_folds"]
        assert folds.attrs["original_shape"].tolist() == [k, nnz]
        assert nnz <= nb  # one process per bin
    indata = inputdata.FitInputData(fn, None)
    np.testing.assert_allclose(
        np.asarray(indata.norm_folds).sum(0), np.asarray(indata.norm.values)
    )


def _moment_fit(fn, tmp_path, *args):
    return fit_toy(fn, tmp_path, "--chisqFit", "--allowNegativeParam", *args)


def test_continuous_m_variance_convention_enforced(tmp_path):
    """M's denominator depends on whether the fit uses bin-by-bin stat, so the
    convention is recorded with the term and a mismatch is refused."""
    fn_bare, _ = make_mcstat_moment_toy(tmp_path, bbb_variance=False)
    fn_bb, _ = make_mcstat_moment_toy(tmp_path, bbb_variance=True)
    _moment_fit(fn_bare, tmp_path, "--noBinByBinStat", "--mcStatDebias", "continuousM")
    _moment_fit(fn_bb, tmp_path, "--mcStatDebias", "continuousM")
    with pytest.raises(RuntimeError, match="'bare' per-bin variance"):
        _moment_fit(fn_bare, tmp_path, "--mcStatDebias", "continuousM")
    with pytest.raises(RuntimeError, match="'bb_inflated' per-bin variance"):
        _moment_fit(
            fn_bb, tmp_path, "--noBinByBinStat", "--mcStatDebias", "continuousM"
        )
    with pytest.raises(ValueError, match="variance must be one of"):
        TensorWriter().add_mc_stat_moment(np.eye(1), ["a"], variance="inflated")


def test_continuous_m_ignored_without_option(moment_toys, tmp_path):
    """A file carrying M fits the plain NLL unless --mcStatDebias continuousM."""
    f0 = _moment_fit(moment_toys["noM"], tmp_path, "--noBinByBinStat")
    f1 = _moment_fit(moment_toys["M"], tmp_path, "--noBinByBinStat")
    assert f1.mcstat_M is None and not f1.mcstat_debias_active
    np.testing.assert_allclose(f1.x.numpy(), f0.x.numpy(), rtol=1e-10)


# ---------------------------------------------------------------------------
# minimization robustness: non-finite trial points, ordinary-likelihood warm start
# ---------------------------------------------------------------------------

POISSON_LINEAR = ("--noBinByBinStat", "--allowNegativeParam")


def test_ordinary_fit_recovers_from_nonfinite_trial_step(tmp_path):
    """From this start, scipy's trust-krylov proposes a step into negative
    expected yields. A NaN loss there used to freeze the trust radius (every
    comparison on a NaN rho is False) until --earlyStopping, and the restart
    re-proposed the same step; it now reads as +inf and the radius shrinks."""
    fn, _ = make_small_fold_toy(tmp_path, 2)
    f = fit_toy(fn, tmp_path, *POISSON_LINEAR, "--mcStatDebias", "none")
    x_ref = f.x.numpy().copy()
    f.x.assign([2.4629323126223066, 9.486694480004758])
    f.minimize()
    np.testing.assert_allclose(f.x.numpy(), x_ref, atol=1e-5)


@pytest.mark.parametrize("bbb", [False, True], ids=["noBBB", "BBB"])
def test_debiased_fit_from_far_start_uses_ordinary_warm_start(bbb, tmp_path):
    """The Poisson jackknife objective is unbounded below where a half
    prediction goes to zero while the full one stays positive, so from a far
    start (e.g. blinding offsets) the de-biased fit could slide away from its
    minimum. minimize() first minimizes the ordinary likelihood, whose minimum
    lies in the convex basin of the de-biased one."""
    fn, _ = make_small_fold_toy(tmp_path, 4)
    args = ("--allowNegativeParam", "--mcStatDebias", "twoHalf")
    if not bbb:
        args = ("--noBinByBinStat",) + args
    f = fit_toy(fn, tmp_path, *args)
    x_ref = f.x.numpy().copy()
    f.x.assign([3.143434, 13.06379379])  # the blinded start of this toy
    f.minimize()
    np.testing.assert_allclose(f.x.numpy(), x_ref, atol=1e-5)
    _, _, h = f.loss_val_grad_hess()
    assert np.linalg.eigvalsh(np.asarray(h)).min() > 0


# ---------------------------------------------------------------------------
# --noHessian: Hessian-free curvature and sandwich
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "toy,extra",
    [
        ("folds", ("--mcStatDebias", "twoHalf")),
        ("folds", ("--mcStatDebias", "kfold", "--covMode", "fisher")),
        ("folds", ("--mcStatDebias", "none", "--covMode", "fisher")),
        ("moment", ("--mcStatDebias", "continuousM")),
        ("moment", ("--mcStatDebias", "continuousM", "--covMode", "fisher")),
    ],
    ids=["twoHalf", "kfold-fisher", "none-fisher", "continuousM", "continuousM-fisher"],
)
def test_hessfree_covariance_matches_dense(toy, extra, tmp_path):
    """cov_block_hessfree (conjugate gradient on Hessian-vector products) gives
    the dense curvature A^-1 and the dense sandwich: c_i^T H c_j with
    c_i = A^-1 e_i for the fold methods, C + c_i^T M c_j for continuous-M."""
    if toy == "folds":
        fn, _ = make_small_fold_toy(tmp_path, 3, systematic_type="normal", syst=1)
    else:
        fn, _ = make_mcstat_moment_toy(tmp_path, bbb_variance=True)
    f = fit_toy(fn, tmp_path, "--allowNegativeParam", *extra)
    _, g, h = f.loss_val_grad_hess()
    A = f.fisher_curvature(h) if f.covMode == "fisher" else h
    C = np.linalg.inv(np.asarray(A))
    if not f.mcstat_debias_active:
        S = C
    elif f.mcStatDebias == "continuousM":
        S = np.asarray(f.cov_mcstat_sandwich(A))
    else:
        S = np.asarray(f.cov_twohalf_sandwich(A, covMode=f.covMode))
    idx = np.arange(int(f.x.shape[0]))
    _, curv, rep = f.cov_block_hessfree(g, idx, covMode=f.covMode)
    np.testing.assert_allclose(curv, C, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(rep, S, rtol=1e-6, atol=1e-12)


# ---------------------------------------------------------------------------
# adjusted profile-likelihood scans
# ---------------------------------------------------------------------------


def test_adjusted_scans_follow_the_sandwich(tmp_path):
    """For a quadratic (linear chisq) de-biased objective the profiled
    statistic is d^T C^-1 d exactly; the adjusted statistic must then be the
    sandwich Wald form d^T Sigma^-1 d, in 1D and 2D, and the adjusted contour
    interval must be +-sigma_sandwich."""
    fn, _ = make_small_fold_toy(tmp_path, 2)
    f = fit_toy(fn, tmp_path, *LINEAR_CHISQ, "--mcStatDebias", "twoHalf")
    _, _, h = f.loss_val_grad_hess()
    C = np.linalg.inv(np.asarray(h))
    S = np.asarray(f.cov_twohalf_sandwich(h))
    assert not np.allclose(C, S)
    f.cov.assign(S)
    f.set_mcstat_scan_covariances(np.arange(2), C, S)
    assert f.mcstat_scan_needs_adjustment()
    xh = f.x.numpy().copy()

    xs, dnll, _ = f.nll_scan("flat", 2.0, 5)
    W = 2 * dnll * f.mcstat_scan_adjustment([0], (xs - xh[0])[:, None])
    np.testing.assert_allclose(W, (xs - xh[0]) ** 2 / S[0, 0], rtol=1e-6, atol=1e-9)

    xs2, ys2, dnll2 = f.nll_scan2D(("flat", "step"), 1.5, 5)
    gx, gy = np.meshgrid(
        np.asarray(xs2) - xh[0], np.asarray(ys2) - xh[1], indexing="ij"
    )
    d = np.stack([gx.ravel(), gy.ravel()], -1)
    W2 = 2 * dnll2.ravel() * f.mcstat_scan_adjustment([0, 1], d)
    wald = np.einsum("ni,ij,nj->n", d, np.linalg.inv(S), d)
    np.testing.assert_allclose(W2, wald, rtol=1e-6, atol=1e-9)

    q = 1.0 / float(f.mcstat_scan_adjustment([0], [[1.0]])[0])
    interval, _ = f.contour_scan("flat", float(f.reduced_nll().numpy()), q)
    np.testing.assert_allclose(np.abs(interval), np.sqrt(S[0, 0]), rtol=1e-4)


# ---------------------------------------------------------------------------
# goodness of fit at the de-biased minimum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3])
def test_gof_statistic_and_reference(k, tmp_path):
    """The goodness-of-fit statistic is 2 * reduced_nll() of the de-biased
    objective at its minimum, bin-by-bin stat included, as for an ordinary fit.
    For a chisq data term without bin-by-bin stat it is exactly the cross-fit
    chi2 sum_b (y-n_A)(y-n_B)/V, for any k, and its reference mean is
    nbins - p (the MC fluctuations cancel). With bin-by-bin stat the profiled
    beta leaves the fraction s_b ~ 1/(1+rho_b) of a residual, and the reference
    mean becomes sum_b s_b - p (each jackknife term separately profiled)."""
    fn, arr = make_small_fold_toy(tmp_path, k)
    y = arr["data"]
    a, b = k // 2, k - k // 2
    Ti = [np.stack([arr["folds"][0][i], arr["folds"][1][i]], 1) for i in range(k)]
    TA, TB = (k / a) * sum(Ti[:a]), (k / b) * sum(Ti[a:])

    f = fit_toy(fn, tmp_path, *LINEAR_CHISQ, "--mcStatDebias", "kfold")
    x = f.x.numpy()
    q = 2 * float(f.reduced_nll().numpy())
    np.testing.assert_allclose(q, np.sum((y - TA @ x) * (y - TB @ x) / y), rtol=1e-8)
    shrink, rho = (np.asarray(v) for v in f._mcstat_gof_bin_inputs())
    np.testing.assert_allclose(shrink, 1.0)
    g = f.mcstat_gof(q, trace_AinvH=2.0)
    assert g["mean"] == pytest.approx(SMALL_NBINS - 2)
    assert 0.0 <= g["pvalue"] <= 1.0

    fb = fit_toy(
        fn, tmp_path, "--chisqFit", "--allowNegativeParam", "--mcStatDebias", "kfold"
    )
    shrink_b, rho_b = (np.asarray(v) for v in fb._mcstat_gof_bin_inputs())
    np.testing.assert_allclose(shrink_b, 1.0 / (1.0 + rho_b), rtol=0.05)
    gb = fb.mcstat_gof(2 * float(fb.reduced_nll().numpy()), trace_AinvH=2.0)
    np.testing.assert_allclose(gb["mean"], np.sum(shrink_b) - 2.0, rtol=1e-8)
    assert 0.0 <= gb["pvalue"] <= 1.0


# ---------------------------------------------------------------------------
# impacts: mcStatDebias and the combined mcStat group
# ---------------------------------------------------------------------------


def _debiased_postfit(tmp_path, *extra):
    fn, _ = make_small_fold_toy(tmp_path, 4)
    f = fit_toy(
        fn, tmp_path, "--allowNegativeParam", "--mcStatDebias", "twoHalf", *extra
    )
    _, g, h = f.loss_val_grad_hess()
    _, bread, C, S = f.mcstat_postfit_covariances(g, h)
    f.cov.assign(S)
    f.mcstat_cov_curv.assign(C)
    return f, bread, np.asarray(C), np.asarray(S)


def test_impact_groups_add_up_to_the_reported_uncertainty(tmp_path):
    """The traditional and global impacts of a de-biased fit decompose its
    curvature covariance and append mcStatDebias (reported - curvature) and the
    combined mcStat (bin-by-bin stat and de-biasing in quadrature), so the
    groups add up to the reported (sandwich) uncertainty; the same holds for the
    histogram global impacts."""
    from rabbit.mappings import helpers as mh

    f, bread, C, S = _debiased_postfit(tmp_path)
    extra = tf.constant(np.diag(S) - np.diag(C))
    sigma = np.sqrt(np.diag(S))[:2]
    for _, grouped in (
        f.impacts_parms(bread, cov=tf.constant(C), mcstat_extra_var=extra),
        f.global_impacts_parms(cov=tf.constant(C), mcstat_extra_var=extra),
    ):
        g = np.asarray(grouped)  # [stat, binByBinStat, mcStatDebias, mcStat]
        assert g.shape[1] == 4
        np.testing.assert_allclose(np.sqrt(np.sum(g[:, :3] ** 2, axis=1)), sigma)
        np.testing.assert_allclose(g[:, 3], np.hypot(g[:, 1], g[:, 2]))
        assert np.all(g[:, 2] > 0)

    mapping = mh.load_mapping("BaseMapping", f.indata)
    exp, aux = f.expected_events(mapping, compute_global_impacts=True)
    var, grouped = np.asarray(aux[0]).ravel(), np.asarray(aux[3])
    grouped = grouped.reshape(-1, grouped.shape[-1])
    np.testing.assert_allclose(np.sum(grouped[:, :3] ** 2, axis=1), var, rtol=1e-8)
    np.testing.assert_allclose(grouped[:, 3], np.hypot(grouped[:, 1], grouped[:, 2]))


# ---------------------------------------------------------------------------
# saturated projection test of a de-biased fit
# ---------------------------------------------------------------------------


def test_weighted_chi2_sf():
    rng = np.random.default_rng(1)
    np.testing.assert_allclose(
        [fitter.weighted_chi2_sf(x, [1.0] * 5) for x in (1, 5, 12)],
        scipy.stats.chi2.sf([1, 5, 12], 5),
    )
    for w in ([0.5, 1.3, 2.0, 0.2], [3.0, 0.3], list(rng.uniform(0.5, 2, 30))):
        s = (rng.standard_normal((200000, len(w))) ** 2 * np.asarray(w)).sum(1)
        xs = np.quantile(s, [0.3, 0.9, 0.99])
        np.testing.assert_allclose(
            [fitter.weighted_chi2_sf(x, w) for x in xs], [0.7, 0.1, 0.01], atol=4e-3
        )


def test_projection_test_reference(tmp_path):
    """The reference of the de-biased saturated projection test,
    sum_j mu_j chi2_1 with mu = eig(A~^+ B~) of the bin scales' profiled
    curvature and score variance at the null point. For an ordinary fit (bread
    = meat) every mu is 1, and a bin-scale direction degenerate with the
    analysis parameters (here the common normalization, as both processes are
    free) is not testable and drops out: npou - 1 values."""
    import copy

    from rabbit.param_models.param_model import (
        CompositeParamModel,
        SaturatedProjectModel,
    )

    ngroup = 6
    groups = np.repeat(np.arange(ngroup), SMALL_NBINS // ngroup)
    chinfo = {"ch0": {"axes": [hist.axis.Regular(ngroup, 0, 1, name="g")]}}
    fn, _ = make_small_fold_toy(tmp_path, 2)
    for debias, n_mu in (("none", ngroup - 1), ("twoHalf", ngroup - 1)):
        f = fit_toy(fn, tmp_path, "--allowNegativeParam", "--mcStatDebias", debias)
        orig = f.param_model
        comp = CompositeParamModel(
            [orig, SaturatedProjectModel(f.indata, chinfo, {"ch0": groups})]
        )
        fs = copy.deepcopy(f)
        fs.init_fit_parms(comp, [], unblind=[], blinding_group=[], freeze_parameters=[])
        fs.xdefaultassign()
        xm = f.x.numpy()
        fs.x[: orig.nparams].assign(xm[: orig.nparams])
        fs.x[comp.nparams :].assign(xm[orig.nparams :])
        fs.arm_regularizers()
        psi = np.arange(orig.nparams, comp.nparams)
        if debias != "none":
            assert fs.mcstat_projection_check(psi)[0]
        if debias == "none":
            _, _, h = fs.loss_val_grad_hess()
            A0 = H0 = np.asarray(h)
        else:
            A0, H0 = fs.mcstat_curvature_and_meat()
        fs.minimize(mcstat_warm_start=False)
        dq = 2 * float(f.reduced_nll().numpy() - fs.reduced_nll().numpy())
        assert dq >= -1e-6
        p, mu = fs.mcstat_projection_test(dq, psi, A0, H0)
        assert len(mu) == n_mu
        assert np.all(mu > 0)
        if debias == "none":
            np.testing.assert_allclose(mu, 1.0, rtol=1e-6)
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# rabbit_limit.py
# ---------------------------------------------------------------------------


def test_rabbit_limit_uses_the_sandwich_and_adjusted_contours(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "_rabbit_limit_for_mcstat_tests", REPO / "bin" / "rabbit_limit.py"
    )
    rl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rl)
    fn, _ = make_small_fold_toy(tmp_path, 4)
    f = fit_toy(fn, tmp_path, "--allowNegativeParam", "--mcStatDebias", "twoHalf")
    rl._postfit_cov(f)
    _, g, h = f.loss_val_grad_hess()
    C = np.linalg.inv(np.asarray(h))
    S = np.asarray(f.cov_twohalf_sandwich(h))
    np.testing.assert_allclose(f.cov.numpy(), S, rtol=1e-8)
    np.testing.assert_allclose(
        rl._mcstat_contour_factor(f, idx=0), C[0, 0] / S[0, 0], rtol=1e-8
    )


def test_projection_test_refuses_an_ill_posed_projection(tmp_path):
    """With one bin scale per bin, the scales absorb any change of process
    composition in the full prediction, which the half predictions still feel:
    a direction the data cannot constrain but the de-biased objective is not
    flat along. The check must flag it (the saturated de-biased fit would
    follow template noise)."""
    import copy

    from rabbit.param_models.param_model import (
        CompositeParamModel,
        SaturatedProjectModel,
    )

    fn, _ = make_small_fold_toy(tmp_path, 2)
    f = fit_toy(fn, tmp_path, "--allowNegativeParam", "--mcStatDebias", "twoHalf")
    chinfo = {"ch0": {"axes": [hist.axis.Regular(SMALL_NBINS, 0, 1, name="x")]}}
    orig = f.param_model
    comp = CompositeParamModel(
        [
            orig,
            SaturatedProjectModel(f.indata, chinfo, {"ch0": np.arange(SMALL_NBINS)}),
        ]
    )
    fs = copy.deepcopy(f)
    fs.init_fit_parms(comp, [], unblind=[], blinding_group=[], freeze_parameters=[])
    fs.xdefaultassign()
    xm = f.x.numpy()
    fs.x[: orig.nparams].assign(xm[: orig.nparams])
    fs.x[comp.nparams :].assign(xm[orig.nparams :])
    fs.arm_regularizers()
    ok, worst = fs.mcstat_projection_check(np.arange(orig.nparams, comp.nparams))
    assert not ok and worst > 1e-3


# ---------------------------------------------------------------------------
# review follow-ups: regularization in the meat, refused combinations
# ---------------------------------------------------------------------------


def test_meat_includes_every_term_of_the_ordinary_objective(tmp_path):
    """The sandwich meat is the curvature of the ordinary (--mcStatDebias none)
    objective at the de-biased point, regularization included: with a
    regularizer it must equal the Hessian of that objective, plus that of the
    bin-by-bin stat compensator (marginal instead of profiled beta; here the
    small total-yield term of a lite multiplicative beta)."""
    from rabbit.regularization.regularizer import Regularizer

    class Penalty(Regularizer):
        needs_observables = False

        def __init__(self):
            self._idx = None

        def set_expectations(self, initial_params, initial_observables, parms=None):
            self._idx = self.resolve_indices(parms, ["flat"], who="Penalty")["flat"]

        def compute_nll_penalty(self, params, observables):
            return 50.0 * tf.square(params[self._idx] - 1.0)

    fn, _ = make_small_fold_toy(tmp_path, 2)
    f = fit_toy(fn, tmp_path, "--allowNegativeParam", "--mcStatDebias", "twoHalf")
    f.tau.assign(1.0)
    f.regularizers = [Penalty()]
    f.arm_regularizers()
    meat = np.asarray(f.mcstat_meat("observed"))
    with f._ordinary_likelihood():
        _, _, h_ordinary = f.loss_val_grad_hess()
    h_comp = f._scalar_hessian(
        "test", lambda: f._compute_mcstat_bb_compensator(meat=True)
    )
    assert np.abs(np.asarray(h_comp)).max() > 0
    np.testing.assert_allclose(
        meat, np.asarray(h_ordinary) + np.asarray(h_comp), rtol=1e-8
    )


def test_data_propagated_sandwich_refused_for_fold_methods(tmp_path):
    """cov_dataprop_sandwich differentiates the full templates only, while a
    fold objective depends on the fold templates: refused."""
    fn, _ = make_small_fold_toy(tmp_path, 2)
    with pytest.raises(NotImplementedError, match="only available for"):
        fit_toy(
            fn,
            tmp_path,
            "--allowNegativeParam",
            "--mcStatDebias",
            "twoHalf",
            "--mcStatDebiasCov",
            "dataPropagated",
        )


@pytest.mark.parametrize("exempt", [False, True], ids=["all-mc", "zerovar"])
@pytest.mark.parametrize("k", [2, 4, 5])
def test_jackknife_terms_are_separately_profiled(k, exempt, tmp_path):
    """With bin-by-bin stat each jackknife term is its own profiled likelihood
    at the full-sample kstat. For a chisq fit with normal-additive bin-by-bin
    stat the profile of a term is exactly (y - n_X)^2 / (2 V), V = y + sigma^2,
    so the de-biased curvature is sum_X c_X J_X^T J_X / V: the noise floor is
    removed with the profiled weight 1/V. Sharing the full-sample beta instead
    weighted the halves' MC deviations by 1/y (docs/mcstat/RESULTS.md §9u)."""
    fn, arr = make_small_fold_toy(tmp_path, k, exempt=exempt)
    args = ["--chisqFit", "--allowNegativeParam", "--mcStatDebias", "kfold"]
    f = fit_toy(fn, tmp_path, *args, "--binByBinStatType", "normal-additive")
    y = arr["data"]
    a, b = k // 2, k - k // 2
    Ti = [np.stack([arr["folds"][0][i], arr["folds"][1][i]], 1) for i in range(k)]
    TA, TB, Tf = (k / a) * sum(Ti[:a]), (k / b) * sum(Ti[a:]), sum(Ti)
    V = y + Tf.sum(1)  # unit weights: sumw2 = sumw; the exempt process has none
    w_full, w_A, w_B = f.jackknife_weights
    H_ref = sum(
        c * T.T @ (T / V[:, None]) for T, c in ((Tf, w_full), (TA, -w_A), (TB, -w_B))
    )
    _, _, h = f.loss_val_grad_hess()
    np.testing.assert_allclose(np.asarray(h)[:2, :2], H_ref, rtol=1e-8)


def make_exact_fold_toy(path, rho=0.5):
    """Two processes whose k = 2 folds are EXACT halves of the true templates,
    each fold claiming the MC variance rho * N / 2 (as weighted MC would): the
    template noise the jackknife removes is absent, so what is left is the
    jackknife's limit, the bin-by-bin stat likelihood at the true templates."""
    nb = SMALL_NBINS
    n = np.stack([np.full(nb, 400.0), np.linspace(100.0, 700.0, nb)])
    ax = hist.axis.Regular(nb, 0.0, 1.0, name="x")
    tw = TensorWriter(sparse=False)
    tw.add_channel([ax], "ch0")
    tw.add_data(_hist1d(ax, n.sum(0)), "ch0")
    axf = hist.axis.IntCategory([0, 1], name="mcfold")
    for name, t in zip(("flat", "step"), n):
        h = hist.Hist(ax, axf, storage=hist.storage.Weight())
        for i in range(2):
            h.view()["value"][:, i] = t / 2
            h.view()["variance"][:, i] = rho * t / 2
        tw.add_process(h, name, "ch0", signal=True, fold_axis="mcfold")
    return _write(tw, path, f"exact_folds_{rho}.hdf5"), n


@pytest.mark.parametrize(
    "bbb",
    [
        [
            "--binByBinStatType",
            "normal-multiplicative",
            "--binByBinStatMode",
            "full",
        ],
        ["--binByBinStatType", "normal-additive", "--binByBinStatDynamic"],
    ],
    ids=["nmult-full", "nadd-dynamic"],
)
def test_fold_compensator_unbiases_the_jackknife_limit(bbb, tmp_path):
    """Where the bin-by-bin stat variance moves with the parameters, the
    profile's -r^2 dV / (2 V^2) has mean -D dV / (2 V^2) once the jackknife has
    removed the template noise (E[r^2] = D): a bias of its own, large along
    weakly constrained directions. The compensator 1/2 sum D (1/V_ref - 1/V)
    cancels it at any point. With exact fold templates the jackknife limit is
    reached exactly, and for chisq with normal bin-by-bin stat the score is
    quadratic and separable in the data, so its mean over y ~ N(mu, D) is the
    average over y = mu +- sqrt(D): it must vanish at the truth."""
    fn, n = make_exact_fold_toy(tmp_path)
    args = ["--chisqFit", "--allowNegativeParam", "--mcStatDebias", "twoHalf"]
    f = fit_toy(fn, tmp_path, *args, *bbb)
    theta0 = np.array([1.3, 0.7])
    mu = theta0 @ n
    f.x.assign(theta0)
    grads = []
    for sign in (1.0, -1.0):
        f.set_nobs(tf.constant(mu + sign * np.sqrt(mu)), variances=tf.constant(mu))
        grads.append(np.asarray(f.loss_val_grad()[1]))
    scale = np.abs(grads[0]).max()
    np.testing.assert_allclose(0.5 * (grads[0] + grads[1]), 0.0, atol=1e-9 * scale)
    # without it the mean score is the compensator's gradient, which is not small
    with tf.GradientTape() as t:
        c = f._compute_mcstat_bb_compensator()
    g_comp = np.asarray(t.gradient(c, f.x))
    assert np.abs(g_comp).max() > 1e-3 * scale


@pytest.mark.parametrize("bbb", [[], ["--noBinByBinStat"]], ids=["bblite", "nobbb"])
def test_hvp_batch_assembles_the_extra_hessians(bbb, tmp_path):
    """--hvpBatch assembles every Hessian the de-biasing adds (sandwich meat,
    Gauss-Newton forms, MC-variance curvature) from batched HVPs, and they
    match the one-pass tape.jacobian."""
    fn, _ = make_small_fold_toy(tmp_path, 4, syst=1)
    args = ["--allowNegativeParam", "--mcStatDebias", "kfold", "--covMode", "fisher"]
    f = fit_toy(fn, tmp_path, *args, *bbb)
    fb = fit_toy(fn, tmp_path, *args, *bbb, "--hvpBatch", "2")
    assert fb.hessian_from_hvp and not f.hessian_from_hvp
    fb.x.assign(f.x)

    def extra(fit):
        _, _, h = fit.loss_val_grad_hess()
        out = [fit.fisher_curvature(h), fit.mcstat_meat("fisher")]
        out += [fit.hess_meat(), fit._mc_data_variance_curvature()]
        return out + list(fit._mcstat_gn_data_curvatures())

    for a, b in zip(extra(f), extra(fb)):
        a, b = np.asarray(a), np.asarray(b)
        np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-10 * np.abs(a).max())
    assert fb._scalar_hessian_fns, "the batched path was not used"


@pytest.mark.parametrize(
    "bbb,models",
    [
        (["--binByBinStatType", "gamma"], "total"),
        (["--binByBinStatType", "normal-additive"], "none"),
        (
            [
                "--binByBinStatType",
                "normal-multiplicative",
                "--binByBinStatMode",
                "full",
            ],
            "per-process",
        ),
        (
            ["--binByBinStatType", "normal-additive", "--binByBinStatMode", "full"],
            "none",
        ),
        (["--binByBinStatType", "gamma", "--binByBinStatDynamic"], "per-process"),
        (
            ["--binByBinStatType", "normal-additive", "--binByBinStatDynamic"],
            "per-process",
        ),
    ],
    ids=["gamma-lite", "nadd-lite", "nmult-full", "nadd-full", "gamma-dyn", "nadd-dyn"],
)
def test_continuous_m_removes_only_what_bin_by_bin_stat_does_not(bbb, models, tmp_path):
    """Continuous-M with every bin-by-bin stat type and mode. Where the
    bin-by-bin stat variance V(x) = D + sum_c u_c(x)^2 moves with the
    parameters, the profiled likelihood already removes part of M; the
    compensator 1/2 sum_b log(V_b(x) / V_ref,b) (marginal instead of profiled
    beta) restores it, with Hessian M_BB = sum du du^T / V at the reference.
    On the unit-weight toy (normalization POIs, D = data): per-process betas
    model all of M (u_p = sqrt(sumw_p) x_p); a lite multiplicative beta the
    noise of the total yield (u = n(x) / sqrt(sumw)); a static additive one
    none."""
    fn, arr = make_mcstat_moment_toy(tmp_path, bbb_variance=True)
    args = ["--chisqFit", "--allowNegativeParam", *bbb]
    f = fit_toy(fn, tmp_path, *args, "--mcStatDebias", "continuousM")
    sumw = arr["sumw"]
    tot = sumw.sum(0)
    x = f.x.numpy()[:2]
    if models == "none":
        assert f.mcstat_bb_vref is None
    else:
        # Hessian of 1/2 log V = 1/2 V'' / V - 1/2 V' V'^T / V^2 at x
        if models == "per-process":
            V = arr["data"] + x**2 @ sumw
            dV = 2 * sumw * x[:, None]
            d2V = 2 * np.einsum("ib,ij->ijb", sumw, np.eye(2))
        else:
            nx = x @ sumw
            V = arr["data"] + nx**2 / tot
            dV = 2 * nx * sumw / tot
            d2V = 2 * np.einsum("ib,jb->ijb", sumw, sumw) / tot
        ref = 0.5 * np.sum(d2V / V - np.einsum("ib,jb->ijb", dV, dV) / V**2, -1)
        H_c = f._scalar_hessian("test", f._compute_mcstat_bb_compensator)
        np.testing.assert_allclose(np.asarray(H_c)[:2, :2], ref, rtol=1e-10)

    _, g, h = f.loss_val_grad_hess()
    assert np.abs(np.asarray(g)).max() < 1e-3
    assert np.linalg.eigvalsh(np.asarray(h)).min() > 0
    if models == "per-process":
        # nothing left to remove: the point is the ordinary fit's (up to M being
        # frozen at the reference while V moves), and only the covariance is
        # the sandwich A^-1 + A^-1 M A^-1
        f0 = fit_toy(fn, tmp_path, *args)
        sigma0 = sigma(np.linalg.inv(np.asarray(f0.loss_val_grad_hess()[2])))
        np.testing.assert_allclose(f.x.numpy(), f0.x.numpy(), atol=0.01 * sigma0)
        S = np.asarray(f.cov_mcstat_sandwich(h))
        Ainv = np.linalg.inv(np.asarray(h))
        np.testing.assert_allclose(S, Ainv + Ainv @ arr["M"] @ Ainv, rtol=1e-8)


@pytest.mark.parametrize(
    "toy,args",
    [
        ("moment", ["--mcStatDebias", "continuousM", "--binByBinStatMode", "full"]),
        ("folds", ["--mcStatDebias", "twoHalf"]),
        ("folds", ["--mcStatDebias", "twoHalf", "--binByBinStatMode", "full"]),
        ("folds", ["--mcStatDebias", "none"]),
    ],
    ids=["continuousM-full", "twoHalf-gammalite", "twoHalf-full", "none"],
)
def test_asimov_is_recentred_with_the_expected_covariance(toy, args, tmp_path):
    """Asimov data are the expected yields of the noisy templates, so they carry
    no attenuation and the de-biased objective is not stationary at the
    generating point. set_mcstat_asimov recentres it there with a linear term
    (Hessian unchanged) and uses the Gauss-Newton covariance, which does not
    depend on the data: the Asimov sandwich equals that of real data at the
    same point. A fit without de-biasing is left untouched."""
    if toy == "moment":
        fn, _ = make_mcstat_moment_toy(tmp_path, bbb_variance=True)
    else:
        fn, _ = make_small_fold_toy(tmp_path, 2)
    common = ["--chisqFit", "--allowNegativeParam", *args]
    f = fit_toy(fn, tmp_path, *common)
    f.defaultassign()
    x0 = f.x.numpy()
    asimov = f.expected_yield()
    f.set_nobs(asimov)
    _, g_raw, h_raw = f.loss_val_grad_hess()
    f.set_mcstat_asimov(True)
    if not f.mcstat_debias_active:
        assert f.covMode == "observed"
        np.testing.assert_array_equal(np.asarray(f.loss_val_grad()[1]), g_raw)
        return
    assert np.abs(np.asarray(g_raw)).max() > 1e-3  # the overshoot it removes
    _, g, h = f.loss_val_grad_hess()
    np.testing.assert_allclose(np.asarray(g), 0.0, atol=1e-8)
    np.testing.assert_allclose(np.asarray(h), np.asarray(h_raw), rtol=1e-12)
    assert f.covMode == "fisher"
    f.minimize()
    np.testing.assert_allclose(f.x.numpy(), x0, atol=1e-6)
    S_asimov = np.asarray(f.mcstat_postfit_covariances(g, h)[3])

    # real data at the same point, same data variance: same fisher sandwich
    f.set_mcstat_asimov(False)
    assert f.covMode == "observed"
    rng = np.random.default_rng(1)
    y = rng.normal(np.asarray(asimov), np.sqrt(np.asarray(asimov)))
    f.set_nobs(tf.constant(y), variances=asimov)
    f.x.assign(x0)
    f.covMode = "fisher"
    _, g_d, h_d = f.loss_val_grad_hess()
    S_data = np.asarray(f.mcstat_postfit_covariances(g_d, h_d)[3])
    np.testing.assert_allclose(S_asimov, S_data, rtol=1e-6)
