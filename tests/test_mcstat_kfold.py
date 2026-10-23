"""Limited-MC-statistics de-biasing: the k-fold cross-fit (--mcStatDebias kfold).

* k = 2: point, curvature and sandwich equal the numpy jackknife.
* any k: the --covMode fisher curvature equals the complete pairwise
  U-statistic A_k = k/(k-1) [F_full - sum_i F_i]; over an ensemble it is
  unbiased and its scatter shrinks as k grows.
* Separately profiled jackknife terms, the compensator, the split-contrast
  meat and the Gauss-Newton marginal term against their closed forms.
* Sparse-mode fold storage gives exactly the dense result; the writer's
  per-fold variances and clipping; empty half templates.
"""

import hist
import numpy as np
import pytest
import tensorflow as tf

from rabbit import fitter, inputdata
from rabbit.param_models import helpers as ph
from rabbit.tensorwriter import TensorWriter
from tests.mcstat_helpers import (
    DDIF,
    DEGEN_NBINS,
    DEGEN_SEED,
    LINEAR_CHISQ,
    SMALL_NBINS,
    _driver,
    _fisher,
    _fold_hist,
    _hist1d,
    _write,
    degenerate_toy_arrays,
    fit_toy,
    make_mcstat_moment_toy,
    make_small_fold_toy,
    sigma,
    split_into_folds,
)

# ---------------------------------------------------------------------------
# toy builders
# ---------------------------------------------------------------------------


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


def kfold_ustatistic(fold_templates, V):
    """Complete pairwise U-statistic curvature A_k = k/(k-1) [F_full - sum F_i]
    for per-fold templates [k][nbins, nparams] (full = sum of folds)."""
    k = len(fold_templates)
    full = sum(fold_templates)
    return (k / (k - 1.0)) * (
        _fisher(full, V) - sum(_fisher(T, V) for T in fold_templates)
    )


# ---------------------------------------------------------------------------
# k = 2 (the two-half jackknife)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def twohalf_toy(tmp_path_factory):
    path = tmp_path_factory.mktemp("mcstat_twohalf")
    fn, arr = make_twohalf_toy(path, k=2)
    return dict(fn=fn, arrays=arr, out=path)


def split_meat_reference(F, y, V, th):
    """Closed-form split-contrast meat of a linear chisq fold fit whose terms
    are (y - n_h)^2 / (2V) (no bin-by-bin stat: V = data variance; static
    normal-additive: V = data + sigma^2), F the fold templates [k, nbins,
    nproc] (unit weights: variances = values), at the point th. X(F) =
    d^2 L_cf / dx dF at the realized split z and at the mean split 0, L = X(0),
    Q = X(z) - X(0), B = L C L^T + 1/2 Q C Q^T, C = diag(v) - v v^T / sum(v)."""
    k, nb, nproc = F.shape
    a, b = k // 2, k - k // 2
    w_A, w_B = a / (2.0 * b), b / (2.0 * a)

    def mixed(F):
        X = np.zeros((nproc, k, nb, nproc))
        for folds, size, c in ((range(a), a, -w_A), (range(a, k), b, -w_B)):
            J = (k / size) * F[list(folds)].sum(0)  # [nb, nproc]
            r = y - J @ th
            for i in folds:
                for q in range(nproc):
                    X[q, i, :, q] -= c * (k / size) * r / V
                    for p in range(nproc):
                        X[q, i, :, p] += c * J[:, q] * (k / size) * th[p] / V
        return X

    v = F

    def quad(X):
        return np.einsum("qibp,ibp,ribp->qr", X, v, X) - np.einsum(
            "qibp,ibp,jbp,rjbp->qr", X, v, v / v.sum(0, keepdims=True), X
        )

    Xz = mixed(F)
    X0 = mixed(np.broadcast_to(F.mean(0, keepdims=True), F.shape))
    return quad(X0) + 0.5 * quad(Xz - X0)


def twohalf_numpy_reference(arr):
    """Closed-form standard and jackknife (two-half) fit of the linear 2-POI toy:
    A = 2 H_full - (H_A + H_B)/2 with half templates scaled x2, sandwich
    A^-1 (H_full + B_split) A^-1 (split_meat_reference). Returns (r_std,
    C_std, r_cf, C_curv, C_sand)."""
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
    r_cf = np.linalg.solve(A_cf, b_cf)
    B = split_meat_reference(np.stack(arr["folds"], -1), data, V, r_cf)
    return (
        np.linalg.solve(Hf, b(Tf)),
        np.linalg.inv(Hf),
        r_cf,
        Ccurv,
        Ccurv @ (Hf + B) @ Ccurv,
    )


def _twohalf_fit(filename, outdir, debias):
    f = fit_toy(filename, outdir, *LINEAR_CHISQ, "--mcStatDebias", debias)
    _, grad, hess = f.loss_val_grad_hess()
    _, cov = f.edmval_cov(grad, hess)
    sand = None
    if debias == "kfold":
        sand = np.asarray(f.cov_twohalf_sandwich(hess))
    return f.x.numpy(), np.asarray(cov), sand


def test_twohalf_matches_numpy_reference(twohalf_toy):
    """rabbit's two-half point, curvature and sandwich equal the independent numpy
    jackknife (atol 1e-3); the standard fit equals the plain least squares."""
    r_std, C_std, r_cf, C_curv, C_sand = twohalf_numpy_reference(twohalf_toy["arrays"])
    x_std, Cr_std, _ = _twohalf_fit(twohalf_toy["fn"], twohalf_toy["out"], "none")
    x_cf, Cr_curv, Cr_sand = _twohalf_fit(
        twohalf_toy["fn"], twohalf_toy["out"], "kfold"
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

    f1 = fit_toy(fn, tmp_path, *common, "kfold")
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
# any k (odd k), additive systematics, sparse storage
# ---------------------------------------------------------------------------


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
    B = split_meat_reference(np.stack(arr["folds"], -1), y, y, f.x.numpy())
    np.testing.assert_allclose(
        np.asarray(f.cov_twohalf_sandwich(h)), Ci @ (Hfull + B) @ Ci, rtol=1e-6
    )

    f = fit_toy(
        fn, tmp_path, *LINEAR_CHISQ, "--mcStatDebias", "kfold", "--covMode", "fisher"
    )
    _, _, h = f.loss_val_grad_hess()
    U = kfold_ustatistic(Ti, y)
    A = np.asarray(f.fisher_curvature(h))
    np.testing.assert_allclose(A, U, rtol=1e-8)
    Ui = np.linalg.inv(U)
    B = split_meat_reference(np.stack(arr["folds"], -1), y, y, f.x.numpy())
    np.testing.assert_allclose(
        np.asarray(f.cov_twohalf_sandwich(A, covMode="fisher")),
        Ui @ (Hfull + B) @ Ui,
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


def _read_logk(fn):
    """(process names, syst names, hlogk [nbins, nproc, nsyst], hlogk_folds
    [k, nbins, nproc, nsyst]) of a symmetric dense tensor."""
    import h5py

    def read(d):
        return d[...].reshape(d.attrs["original_shape"])

    def names(d):
        return [x.decode() if isinstance(x, bytes) else x for x in d[...]]

    with h5py.File(fn) as h5:
        logk_folds = read(h5["hlogk_folds"]) if "hlogk_folds" in h5 else None
        return names(h5["hprocs"]), names(h5["hsysts"]), read(h5["hlogk"]), logk_folds


@pytest.mark.parametrize("symmetrize", ["linear", "quadratic"])
def test_folded_pair_split_symmetrization(symmetrize, tmp_path):
    """A folded up/down pair with the 'linear'/'quadratic' symmetrization is split
    into SymAvg and SymDiff in every fold, as for the full pair: per fold
    SymAvg = (lu + ld) / 2 and SymDiff = c (lu - ld) / 2 (c = sqrt(3) for
    quadratic) with lu = log(up/n), ld = -log(down/n) of that fold. The full
    systematics are those of the pair without folds."""
    k, nb = 4, 30
    rng = np.random.default_rng(3)
    ax = hist.axis.Regular(nb, 0.0, 1.0, name="x")
    folds = split_into_folds(rng, rng.poisson(np.full(nb, 2000.0)), k).astype(float)
    w = 1.0 + 0.2 * (np.linspace(0.0, 1.0, nb) - 0.5)
    up = np.stack([rng.poisson(f * w) for f in folds]).astype(float)
    down = np.stack([rng.poisson(f / w**0.5) for f in folds]).astype(float)

    fns = {}
    for folded in (True, False):
        tw = TensorWriter()
        tw.add_channel([ax], "ch0")
        tw.add_data(_hist1d(ax, folds.sum(0)), "ch0")
        tw.add_process(_fold_hist(ax, folds), "sig", "ch0", fold_axis="mcfold")
        if folded:
            pair = [_fold_hist(ax, up), _fold_hist(ax, down)]
            kwargs = {"fold_axis": "mcfold"}
        else:
            pair = [_hist1d(ax, up.sum(0)), _hist1d(ax, down.sum(0))]
            kwargs = {}
        tw.add_systematic(
            pair, "pair", "sig", "ch0", mirror=False, symmetrize=symmetrize, **kwargs
        )
        fns[folded] = _write(tw, tmp_path, f"pair_{symmetrize}_{folded}.hdf5")

    _, systs, logk, logk_folds = _read_logk(fns[True])
    _, systs_ref, logk_ref, _ = _read_logk(fns[False])
    assert systs == systs_ref
    assert sorted(systs) == ["pairSymAvg", "pairSymDiff"]
    np.testing.assert_allclose(logk, logk_ref, rtol=1e-12)

    c = np.sqrt(3.0) if symmetrize == "quadratic" else 1.0
    lu, ld = np.log(up / folds), -np.log(down / folds)
    iavg, idiff = systs.index("pairSymAvg"), systs.index("pairSymDiff")
    np.testing.assert_allclose(logk_folds[:, :, 0, iavg], 0.5 * (lu + ld), rtol=1e-10)
    np.testing.assert_allclose(
        logk_folds[:, :, 0, idiff], 0.5 * c * (lu - ld), rtol=1e-10
    )


def test_folded_syst_has_no_fold_structure_where_the_process_has_none(tmp_path):
    """In the bins a folded process is written without fold structure (here a
    negative fold), and everywhere for an MC-stat-exempt process, a folded
    systematic gets the full logk in every fold, so that the fold prediction
    stays the full one / k at any value of the nuisance. So does a single fold
    with too few effective MC events (here cancelling weights), whose own
    variation would be the ratio of two noise-dominated sums. An empty fold has
    no variation, as an empty full template."""
    k, nb, bad, low, empty = 4, 20, 7, 12, 15
    rng = np.random.default_rng(5)
    ax = hist.axis.Regular(nb, 0.0, 1.0, name="x")
    axf = hist.axis.IntCategory(list(range(k)), name="mcfold")
    folds = split_into_folds(rng, rng.poisson(np.full(nb, 1000.0)), k).astype(float)
    folds[0, bad] = -5.0  # negative weights
    up = np.stack([rng.poisson(np.abs(f) * 1.05) * np.sign(f) for f in folds]).astype(
        float
    )

    def fold_hist(values, variances):
        h = hist.Hist(ax, axf, storage=hist.storage.Weight())
        h.view()["value"] = values.T
        h.view()["variance"] = variances.T
        return h

    tw = TensorWriter()
    tw.add_channel([ax], "ch0")
    tw.add_data(_hist1d(ax, np.abs(folds).sum(0)), "ch0")
    variances = np.abs(folds)
    folds[1, low] = 0.5  # a near-cancellation: kstat = 0.25 / 100
    variances[1, low] = 100.0
    folds[2, empty] = variances[2, empty] = 0.0  # an empty fold
    tw.add_process(fold_hist(folds, variances), "neg", "ch0", fold_axis="mcfold")
    # MC-stat-exempt: no MC-stat variance at all
    tw.add_process(
        fold_hist(np.abs(folds), 0 * folds), "exempt", "ch0", fold_axis="mcfold"
    )
    for proc in ("neg", "exempt"):
        tw.add_systematic(
            fold_hist(np.abs(up) if proc == "exempt" else up, np.abs(up)),
            f"tilt_{proc}",
            proc,
            "ch0",
            fold_axis="mcfold",
        )
    procs, systs, logk, logk_folds = _read_logk(_write(tw, tmp_path, "flat.hdf5"))

    ineg, iexempt = procs.index("neg"), procs.index("exempt")
    isyst = systs.index("tilt_neg")
    # the flattened bin: every fold the full logk; elsewhere the folds differ
    np.testing.assert_allclose(
        logk_folds[:, bad, ineg, isyst], logk[bad, ineg, isyst], rtol=1e-12
    )
    others = np.arange(nb) != bad
    assert np.all(np.ptp(logk_folds[:, others, ineg, isyst], axis=0) > 0)
    # the fold with too few effective events: the full logk, the others their own
    np.testing.assert_allclose(
        logk_folds[1, low, ineg, isyst], logk[low, ineg, isyst], rtol=1e-12
    )
    assert np.all(logk_folds[[0, 2, 3], low, ineg, isyst] != logk[low, ineg, isyst])
    # an empty fold has no variation
    assert logk_folds[2, empty, ineg, isyst] == 0.0
    # the exempt process: every bin the full logk
    isyst = systs.index("tilt_exempt")
    np.testing.assert_allclose(
        logk_folds[:, :, iexempt, isyst],
        np.broadcast_to(logk[:, iexempt, isyst], (k, nb)),
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# bin-by-bin stat: separately profiled jackknife terms, compensator
# ---------------------------------------------------------------------------


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
    args = ["--chisqFit", "--allowNegativeParam", "--mcStatDebias", "kfold"]
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


# ---------------------------------------------------------------------------
# per-fold variances: writer, split-contrast meat, Gauss-Newton terms
# ---------------------------------------------------------------------------


def test_writer_stores_per_fold_variances(tmp_path):
    """The per-fold MC-stat variances are stored next to the fold templates
    (the fold histograms' variances); processes without folds get zero."""
    fn, arr = make_small_fold_toy(tmp_path, 3, exempt=True)
    ind = inputdata.FitInputData(fn, None)
    v = np.asarray(ind.sumw2_folds)
    nb = SMALL_NBINS
    for p in range(2):
        np.testing.assert_allclose(v[:, :nb, p], arr["folds"][p])  # unit weights
    np.testing.assert_array_equal(v[:, :nb, 2], 0.0)


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("k", [2, 4])
def test_split_meat_matches_the_closed_form(k, sparse, tmp_path):
    """The split-contrast meat: X(F) = d^2 L_cf / dx dF, the mixed derivative
    with respect to the fold templates F, at the realized split z and at the
    mean split 0 (every fold the fold mean), L = X(0), Q = X(z) - X(0),
    B_split = L C L^T + 1/2 Q C Q^T with C = diag(v) - v v^T / sum(v) the
    covariance of F given the full sample. For chisq with normal-additive
    bin-by-bin stat each separately profiled term is exactly
    (y - n_h)^2 / (2V), so X is analytic. The Hessian-free rows agree."""
    fn, arr = make_small_fold_toy(tmp_path, k, sparse=sparse)
    args = ["--chisqFit", "--allowNegativeParam", "--mcStatDebias", "kfold"]
    f = fit_toy(fn, tmp_path, *args, "--binByBinStatType", "normal-additive")
    y = arr["data"]
    F = np.stack(arr["folds"], -1)  # [k, nb, 2]
    V = y + F.sum((0, 2))
    B_ref = split_meat_reference(F, y, V, f.x.numpy()[:2])
    B = np.asarray(f.mcstat_split_meat())[:2, :2]
    np.testing.assert_allclose(B, B_ref, rtol=1e-8)
    rows = np.eye(int(f.x.shape[0]))[:2]
    np.testing.assert_allclose(
        np.asarray(f.mcstat_split_meat_rows(rows))[:, :2], B_ref, rtol=1e-8
    )


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_writer_keeps_the_folds_consistent_with_the_clipped_template(sparse, tmp_path):
    """add_process clips a template at zero (unless allow_negative_expectation);
    the folds must keep summing to the stored full template, and no fold (so no
    half prediction) may be negative: bins with a negative fold are written
    without fold structure (folds = clipped full / k, no split variance), the
    same in dense and sparse mode. A folded process without any MC-stat variance
    is MC-stat-exempt, and `variances=` scales the per-fold variances by the
    factor it applies to the full template's own variances."""
    nb, k = 6, 2
    ax = hist.axis.Regular(nb, 0.0, 1.0, name="x")
    axf = hist.axis.IntCategory(list(range(k)), name="mcfold")
    vals = np.array(
        [[5.0, 4.0], [-1.0, 3.0], [-2.0, 1.0], [2.0, 2.0], [0.0, 0.0], [3.0, 1.0]]
    )
    tw = TensorWriter(sparse=sparse)
    tw.add_channel([ax], "ch0")
    tw.add_data(_hist1d(ax, np.full(nb, 10.0)), "ch0")
    h = hist.Hist(ax, axf, storage=hist.storage.Weight())
    h.view()["value"] = vals
    h.view()["variance"] = np.abs(vals)
    tw.add_process(h, "sig", "ch0", signal=True, fold_axis="mcfold")
    hz = h.copy()
    hz.view()["variance"] = 0.0
    tw.add_process(hz, "nostat", "ch0", fold_axis="mcfold")
    hv = h.copy()
    tw.add_process(
        hv,
        "scaled",
        "ch0",
        fold_axis="mcfold",
        variances=_hist1d(ax, np.full(nb, 6.0)),
    )
    fn = _write(tw, tmp_path, f"clip_{int(sparse)}.hdf5")
    ind = inputdata.FitInputData(fn, None)
    if sparse:
        full = np.zeros((nb, 3))
        idx = np.asarray(ind.norm.indices)
        full[idx[:, 0], idx[:, 1]] = np.asarray(ind.norm.values)
        F = np.zeros((k, nb, 3))
        V = np.zeros((k, nb, 3))
        for i in range(k):
            F[i][idx[:, 0], idx[:, 1]] = np.asarray(ind.norm_folds)[i]
            V[i][idx[:, 0], idx[:, 1]] = np.asarray(ind.sumw2_folds)[i]
    else:
        full = np.asarray(ind.norm)[:nb]
        F = np.asarray(ind.norm_folds)[:, :nb]
        V = np.asarray(ind.sumw2_folds)[:, :nb]
    np.testing.assert_allclose(F.sum(0), full, atol=1e-12)
    assert np.all(F >= 0.0)
    sig = 0
    np.testing.assert_allclose(full[:, sig], [9, 2, 0, 4, 0, 4])
    for b in (1, 2):  # a negative fold: no fold structure, no split variance
        np.testing.assert_allclose(F[:, b, sig], full[b, sig] / k)
        np.testing.assert_array_equal(V[:, b, sig], 0.0)
    np.testing.assert_allclose(F[:, 0, sig], [5.0, 4.0])
    np.testing.assert_allclose(V[:, 0, sig], [5.0, 4.0])
    nostat = 1
    np.testing.assert_allclose(F[:, :, nostat], np.tile(full[:, nostat] / k, (k, 1)))
    np.testing.assert_array_equal(V[:, :, nostat], 0.0)
    scaled = 2
    np.testing.assert_allclose(V[:, 0, scaled], 6.0 * np.array([5.0, 4.0]) / 9.0)


@pytest.mark.parametrize("method", ["kfold", "continuousM"])
def test_gauss_newton_includes_the_expected_marginal_variance_term(method, tmp_path):
    """Where the bin-by-bin stat variance V(x) moves with the parameters, the
    Gauss-Newton form of a marginal bin-by-bin stat term (profile + 1/2 log V:
    the fold methods' meat, the continuous-M objective) adds the expectation of
    what it drops, -1/2 sum_b dV dV^T / V^2 (E[r^2] = V, E[r J] = -dV / 2). For
    chisq, normal-multiplicative full mode and unit weights: J_bp = sumw_bp,
    V_b = y_b + sum_p sumw_bp x_p^2, dV_bp = 2 sumw_bp x_p."""
    full = ["--binByBinStatType", "normal-multiplicative", "--binByBinStatMode", "full"]
    args = ["--chisqFit", "--allowNegativeParam", "--covMode", "fisher", *full]
    if method == "kfold":
        fn, arr = make_small_fold_toy(tmp_path, 2)
        sumw = np.stack([fp.sum(0) for fp in arr["folds"]])
    else:
        fn, arr = make_mcstat_moment_toy(tmp_path, bbb_variance=True)
        sumw = arr["sumw"]
    f = fit_toy(fn, tmp_path, *args, "--mcStatDebias", method)
    y = arr["data"]
    x = f.x.numpy()[:2]
    V = y + (x**2) @ sumw
    dV = 2.0 * sumw * x[:, None]
    ref = np.einsum("ib,jb->ij", sumw / V, sumw) - 0.5 * np.einsum(
        "ib,jb->ij", dV / V**2, dV
    )
    _, _, h = f.loss_val_grad_hess()
    if method == "kfold":
        got = np.asarray(f.fisher_curvature_full(f.hess_meat()))[:2, :2]
    else:
        got = np.asarray(f.fisher_curvature(h))[:2, :2]
        ref = ref - arr["M"]
    np.testing.assert_allclose(got, ref, rtol=1e-8)


# ---------------------------------------------------------------------------
# empty half templates, unbalanced odd-k splits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("poisson", [False, True], ids=["chisq", "poisson"])
def test_empty_half_template_keeps_the_gradient_finite(poisson, tmp_path):
    """A process with MC events in a bin can have none in one fold half, so the
    separately profiled half sees a zero template that is not masked (the
    masks come from the full sample). The gamma full-mode profile divides kstat
    by the template for its threshold; that must not turn the gradient NaN (WMass
    rabbit#141 review). Bin 2: the background's folds are [4, 0]. For chisq also
    a bin where the half is empty for every process."""
    nb = 6
    ax = hist.axis.Regular(nb, 0.0, 1.0, name="x")
    sig = np.array([[200, 150, 100, 75, 50, 40], [200, 150, 100, 75, 50, 40]], float)
    bkg = np.array([[30, 20, 4, 15, 10, 5], [30, 20, 0, 15, 10, 5]], float)
    if not poisson:
        sig[1, 4] = 0.0  # bin 4: half B empty for every process
        bkg[1, 4] = 0.0
    tw = TensorWriter(sparse=False)
    tw.add_channel([ax], "ch0")
    tw.add_data(_hist1d(ax, sig.sum(0) + bkg.sum(0)), "ch0")
    tw.add_process(_fold_hist(ax, sig), "sig", "ch0", signal=True, fold_axis="mcfold")
    tw.add_process(_fold_hist(ax, bkg), "bkg", "ch0", fold_axis="mcfold")
    tw.add_norm_systematic("bkg_norm", "bkg", "ch0", 1.1)
    fn = _write(tw, tmp_path, f"empty_half_{int(poisson)}.hdf5")
    for bbb in (
        ["--binByBinStatType", "gamma", "--binByBinStatMode", "full"],
        ["--binByBinStatType", "gamma"],
        ["--binByBinStatType", "normal-multiplicative", "--binByBinStatMode", "full"],
    ):
        args = ["--mcStatDebias", "kfold", *bbb] + ([] if poisson else ["--chisqFit"])
        options = (
            _driver()
            .make_parser()
            .parse_args([str(fn), "-o", str(tmp_path), "-t", "0", *args])
        )
        indata = inputdata.FitInputData(str(fn), None)
        pm = ph.load_models([["Mu"]], indata, **vars(options))
        f = fitter.Fitter(indata, pm, options, do_blinding=False)
        f.defaultassign()
        f.set_nobs(indata.data_obs)
        val, grad = f.loss_val_grad()
        assert np.isfinite(float(val)), bbb
        assert np.all(np.isfinite(np.asarray(grad))), bbb


def test_poisson_fold_fit_refuses_data_in_an_empty_half(tmp_path):
    """A bin where one fold half has no template for any process predicts zero
    events in that half at any parameter values: a Poisson fold fit cannot
    accommodate data there and refuses it explicitly (the folds depend only on
    the simulation). No data in the bin, or a chisq fit, is fine."""
    nb = 4
    ax = hist.axis.Regular(nb, 0.0, 1.0, name="x")
    sig = np.array([[100, 80, 5, 40], [100, 80, 0, 40]], float)
    bkg = np.array([[20, 10, 3, 5], [20, 10, 0, 5]], float)
    tw = TensorWriter(sparse=False)
    tw.add_channel([ax], "ch0")
    data = sig.sum(0) + bkg.sum(0)
    tw.add_data(_hist1d(ax, data), "ch0")
    tw.add_process(_fold_hist(ax, sig), "sig", "ch0", signal=True, fold_axis="mcfold")
    tw.add_process(_fold_hist(ax, bkg), "bkg", "ch0", fold_axis="mcfold")
    fn = _write(tw, tmp_path, "empty_half_data.hdf5")

    def make(*extra):
        options = (
            _driver()
            .make_parser()
            .parse_args(
                [
                    str(fn),
                    "-o",
                    str(tmp_path),
                    "-t",
                    "0",
                    "--mcStatDebias",
                    "kfold",
                    *extra,
                ]
            )
        )
        indata = inputdata.FitInputData(str(fn), None)
        pm = ph.load_models([["Mu"]], indata, **vars(options))
        f = fitter.Fitter(indata, pm, options, do_blinding=False)
        f.defaultassign()
        return f, indata

    f, indata = make()
    with pytest.raises(RuntimeError, match="no MC-stat template in one fold half"):
        f.set_nobs(indata.data_obs)
    empty = data.copy()
    empty[2] = 0.0
    f.set_nobs(tf.constant(empty))  # no data in that bin: fine
    f, indata = make("--chisqFit")
    f.set_nobs(indata.data_obs)  # chisq: the half term stays finite


@pytest.mark.parametrize(
    "bbb",
    [
        ["--binByBinStatType", "gamma", "--binByBinStatMode", "full"],
        ["--binByBinStatType", "gamma"],
    ],
    ids=["gamma-full", "gamma-lite"],
)
def test_split_meat_stays_finite_for_odd_k_unbalanced_folds(bbb, tmp_path):
    """With odd k the reflected split (F_i -> 2 mean(F) - F_i) can make a half
    prediction negative, e.g. folds [5, 1, 0]: half A = 2 * 6 - 3 * 5 = -3, which
    a Poisson fit cannot evaluate. The split meat's second point is the mean
    split, whose halves are the full prediction: finite for any k."""
    nb, k = 4, 3
    ax = hist.axis.Regular(nb, 0.0, 1.0, name="x")
    sig = np.array([[50, 40, 30, 20], [50, 40, 30, 20], [50, 40, 30, 20]], float)
    bkg = np.array([[10, 8, 5, 4], [10, 8, 1, 4], [10, 8, 0, 4]], float)
    tw = TensorWriter(sparse=False)
    tw.add_channel([ax], "ch0")
    tw.add_data(_hist1d(ax, sig.sum(0) + bkg.sum(0)), "ch0")
    tw.add_process(_fold_hist(ax, sig), "sig", "ch0", signal=True, fold_axis="mcfold")
    tw.add_process(_fold_hist(ax, bkg), "bkg", "ch0", fold_axis="mcfold")
    tw.add_norm_systematic("bkg_norm", "bkg", "ch0", 1.1)
    fn = _write(tw, tmp_path, "odd_k_unbalanced.hdf5")
    f = fit_toy(fn, tmp_path, "--mcStatDebias", "kfold", *bbb)
    B = np.asarray(f.mcstat_split_meat())
    assert np.all(np.isfinite(B))
    S = np.asarray(f.mcstat_postfit_covariances(*f.loss_val_grad_hess()[1:])[3])
    assert np.all(np.isfinite(S))


# ---------------------------------------------------------------------------
# bounded fold deviations (--mcStatFoldBound)
# ---------------------------------------------------------------------------


def _fold_fitter(fn, tmp_path, *extra):
    """The Fitter `rabbit_fit.py fn -t 0 --mcStatDebias kfold *extra` builds, at
    its default point with the data set (not minimized)."""
    options = (
        _driver()
        .make_parser()
        .parse_args(
            [str(fn), "-o", str(tmp_path), "-t", "0", "--mcStatDebias", "kfold"]
            + list(extra)
        )
    )
    indata = inputdata.FitInputData(str(fn), None)
    pm = ph.load_models([["Mu"]], indata, **vars(options))
    f = fitter.Fitter(indata, pm, options, do_blinding=False)
    f.defaultassign()
    f.set_nobs(indata.data_obs)
    return f


def _set_param(f, name, value):
    names = [p.decode() if isinstance(p, bytes) else str(p) for p in f.parms]
    x = f.x.numpy()
    x[names.index(name)] = value
    f.x.assign(x)


def make_one_fold_variation_toy(path, k=2, sparse=False):
    """A process 'sig' of ~100 unit-weight MC events per bin and fold, with a
    folded systematic 'onefold' that varies fold 0 only (+50%), as a variation
    carried by a few events of one fold would: its fold logk differ by log 1.5
    in every bin. Asimov data = the nominal. Returns the filename."""
    nb = 5
    folds = np.random.default_rng(3).poisson(100.0, size=(k, nb)).astype(float)
    up = folds.copy()
    up[0] *= 1.5
    ax = hist.axis.Regular(nb, 0.0, 1.0, name="x")
    tw = TensorWriter(sparse=sparse)
    tw.add_channel([ax], "ch0")
    tw.add_data(_hist1d(ax, folds.sum(0)), "ch0")
    tw.add_process(_fold_hist(ax, folds), "sig", "ch0", signal=True, fold_axis="mcfold")
    tw.add_systematic(_fold_hist(ax, up), "onefold", "sig", "ch0", fold_axis="mcfold")
    return _write(tw, path, f"onefold_k{k}_{'sp' if sparse else 'de'}.hdf5")


FORMS = ["taper", "tanh"]
# the bound alone, without the shared-logk fallback for noisy templates (the
# one-fold toy's fold deviation norm, ~0.2, is above the threshold of 0.1)
NO_SHARING = ("--mcStatFoldMaxDeviation", "0")
# the protections are off by default
BOUND = ("--mcStatFoldBound", "0.5")


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_fold_bound_leaves_the_nominal_point_unchanged(sparse, form, tmp_path):
    """Both bounds have value 0, slope 1 and curvature 0 at x = 0: the de-biased
    loss, gradient and Hessian at the nominal point (where every fold deviation x
    vanishes) are those of the unbounded objective, i.e. the noise-floor
    subtraction is untouched."""
    fn = make_folded_syst_toy(tmp_path, k=2, sparse=sparse, split=True)
    bound = ("--mcStatFoldBound", "0.5", "--mcStatFoldBoundForm", form)
    got = _fold_fitter(fn, tmp_path, *bound).loss_val_grad_hess()
    ref = _fold_fitter(fn, tmp_path, "--mcStatFoldBound", "0").loss_val_grad_hess()
    for g, r in zip(got, ref):
        np.testing.assert_allclose(np.asarray(g), np.asarray(r), rtol=1e-12, atol=1e-9)


def _fold_log_ratios(f):
    """log of (fold prediction / fold nominal) over (full prediction / full
    nominal) per (fold, bin, process): the bounded deviation d of each fold."""
    _, norm_full = f._compute_yields_noBBB(full=False, compute_norm=True)
    _, norm_folds = f._compute_fold_yields(halves=False, return_norm=True)
    nominal_folds = np.asarray(f.indata.norm_folds)[:, : f.indata.nbins]
    nominal = np.asarray(f.indata.norm)[: f.indata.nbins]
    ratio = (np.asarray(norm_folds) / nominal_folds) / (
        np.asarray(norm_full) / nominal
    )[None]
    return np.log(ratio)


@pytest.mark.parametrize(
    "form, bound, dmax",
    [("taper", 0.5, 0.25), ("tanh", 0.5, 0.5), ("taper", 0.0, None)],
    ids=["taper", "tanh", "unbounded"],
)
def test_fold_bound_keeps_each_fold_near_the_full_template(form, bound, dmax, tmp_path):
    """Far along the systematic each fold's template stays within e^(+-dmax) of
    the full-sample variation (dmax = kappa/2 for the taper, kappa for tanh);
    unbounded, fold 0 runs ahead by e^(x)."""
    fn = make_one_fold_variation_toy(tmp_path)
    args = ("--mcStatFoldBound", str(bound), "--mcStatFoldBoundForm", form)
    f = _fold_fitter(fn, tmp_path, *args, *NO_SHARING)
    _set_param(f, "onefold", 30.0)
    d = _fold_log_ratios(f)
    if dmax is None:
        assert np.abs(d).max() > 1.0
    else:
        assert np.abs(d).max() <= dmax * (1 + 1e-9)


def test_fold_taper_hands_the_fold_back_to_the_full_template(tmp_path):
    """The taper returns to the shared logk for |x| >> kappa: very far along the
    systematic the folds follow the full-sample variation, d ~ kappa^2 / x, so
    the objective there is the ordinary likelihood with no gain for the fit."""
    f = _fold_fitter(
        make_one_fold_variation_toy(tmp_path), tmp_path, *BOUND, *NO_SHARING
    )
    _set_param(f, "onefold", 1000.0)  # |x| ~ 400 kappa in every bin
    assert np.abs(_fold_log_ratios(f)).max() < 0.01 * 0.5


@pytest.mark.parametrize("form", FORMS)
def test_fold_bound_keeps_the_objective_bounded(form, tmp_path):
    """The one-fold variation drives the unbounded jackknife objective to -inf
    along the systematic (the half that carries it outgrows the full template:
    turnover at theta ~ 8 here); bounded it rises like an ordinary likelihood."""
    fn = make_one_fold_variation_toy(tmp_path)
    for bound in (0.5, 0.0):
        f = _fold_fitter(
            fn,
            tmp_path,
            "--mcStatFoldBound",
            str(bound),
            "--mcStatFoldBoundForm",
            form,
            "--noBinByBinStat",
            *NO_SHARING,
        )
        losses = []
        for theta in (0.0, 10.0, 30.0):
            _set_param(f, "onefold", theta)
            losses.append(float(f.loss_val_grad()[0]))
        if bound > 0:
            assert losses[0] < losses[1] < losses[2], losses
        else:
            assert losses[2] < losses[0] - 1e6, losses


@pytest.mark.parametrize("form", FORMS)
def test_fold_bound_dense_equals_sparse(form, tmp_path):
    """The bound acts on the dense per-fold logk and on the sparse per-fold delta
    alike: the same fold predictions where it is active."""
    preds = []
    for sparse in (False, True):
        fn = make_one_fold_variation_toy(tmp_path, sparse=sparse)
        f = _fold_fitter(
            fn, tmp_path, *BOUND, "--mcStatFoldBoundForm", form, *NO_SHARING
        )
        _set_param(f, "onefold", 30.0)
        preds.append(np.asarray(f._compute_fold_yields(halves=False)))
    np.testing.assert_allclose(preds[1], preds[0], rtol=1e-10)


def make_mixed_noise_toy(path, sparse=False):
    """'sig' (~1000 unit-weight MC events per bin and fold) with a folded 'tilt'
    whose fold variations differ by their MC noise only (fold deviation norm
    ~0.03), and 'bkg' (~100 per bin and fold) with a folded 'onefold' varying
    fold 0 only (+50%, fold deviation norm ~0.2). Asimov data. Returns the
    filename."""
    nb, k = 5, 2
    rng = np.random.default_rng(5)
    sig = rng.poisson(1000.0, size=(k, nb)).astype(float)
    w = 1.0 + 0.1 * (np.linspace(0.0, 1.0, nb) - 0.5)
    sig_up = np.stack([rng.poisson(sig[i] * w) for i in range(k)]).astype(float)
    bkg = rng.poisson(100.0, size=(k, nb)).astype(float)
    bkg_up = bkg.copy()
    bkg_up[0] *= 1.5
    ax = hist.axis.Regular(nb, 0.0, 1.0, name="x")
    tw = TensorWriter(sparse=sparse)
    tw.add_channel([ax], "ch0")
    tw.add_data(_hist1d(ax, sig.sum(0) + bkg.sum(0)), "ch0")
    tw.add_process(_fold_hist(ax, sig), "sig", "ch0", signal=True, fold_axis="mcfold")
    tw.add_process(_fold_hist(ax, bkg), "bkg", "ch0", fold_axis="mcfold")
    tw.add_systematic(_fold_hist(ax, sig_up), "tilt", "sig", "ch0", fold_axis="mcfold")
    tw.add_systematic(
        _fold_hist(ax, bkg_up), "onefold", "bkg", "ch0", fold_axis="mcfold"
    )
    return _write(tw, path, f"mixed_noise_{'sp' if sparse else 'de'}.hdf5")


def test_noisy_fold_templates_take_the_shared_logk(tmp_path):
    """Templates whose fold deviation norm exceeds --mcStatFoldMaxDeviation follow
    the full-sample variation in every fold (shared logk); the others keep their
    own fold variations."""
    fn = make_mixed_noise_toy(tmp_path)
    for tau, bkg_shared in (("0.1", True), ("0", False)):
        f = _fold_fitter(fn, tmp_path, "--mcStatFoldMaxDeviation", tau)
        procs = [p.decode() if isinstance(p, bytes) else str(p) for p in f.indata.procs]
        isig, ibkg = procs.index("sig"), procs.index("bkg")
        _set_param(f, "onefold", 30.0)
        _set_param(f, "tilt", 3.0)
        d = _fold_log_ratios(f)
        assert np.abs(d[:, :, isig]).max() > 1e-3  # split logk kept
        if bkg_shared:
            flags = f.mcstat_fold_shared_templates[: f.indata.nbins]
            assert flags[:, ibkg].all() and not flags[:, isig].any()
            np.testing.assert_allclose(d[:, :, ibkg], 0.0, atol=1e-12)
        else:
            assert f.mcstat_fold_shared_templates is None
            assert np.abs(d[:, :, ibkg]).max() > 1e-3


def test_fold_templates_below_the_threshold_are_untouched(tmp_path):
    """No template of a well-measured systematic crosses the threshold: the
    de-biased loss, gradient and Hessian away from the nominal point are those
    without --mcStatFoldMaxDeviation."""
    fn = make_folded_syst_toy(tmp_path, k=2, split=True)
    res = []
    for tau in ("0.1", "0"):
        f = _fold_fitter(fn, tmp_path, "--mcStatFoldMaxDeviation", tau)
        if tau != "0":
            assert not f.mcstat_fold_shared_templates.any()
        _set_param(f, "tilt", 1.5)
        res.append(f.loss_val_grad_hess())
    for g, r in zip(*res):
        np.testing.assert_allclose(np.asarray(g), np.asarray(r), rtol=1e-12, atol=1e-9)


def test_noisy_fold_templates_dense_equals_sparse(tmp_path):
    """The shared logk is given in dense (per-fold logk) and sparse (per-fold
    delta) storage alike: the same fold predictions."""
    preds = []
    for sparse in (False, True):
        f = _fold_fitter(
            make_mixed_noise_toy(tmp_path, sparse=sparse),
            tmp_path,
            *BOUND,
            "--mcStatFoldMaxDeviation",
            "0.1",
        )
        _set_param(f, "onefold", 30.0)
        _set_param(f, "tilt", 3.0)
        preds.append(np.asarray(f._compute_fold_yields(halves=False)))
    np.testing.assert_allclose(preds[1], preds[0], rtol=1e-10)


def make_few_event_toy(path, sparse=False, local=False):
    """'sig' (~1000 unit-weight MC events per bin and fold, folded 'tilt' with
    fold variations that differ by their MC noise only) and 'few', a background
    of a few unit-weight events per bin (kstat ~2, one fold often empty) with a
    folded 'onefew' that varies the events of one fold (+50%). With `local`,
    'few' is a single bin with four events per fold (above the writer's floor of
    3 for a fold's own variation, kstat 8 for the template), as a local nuisance
    (e.g. a scale factor of one eta-pt cell) sees it. Asimov data. Returns the
    filename."""
    nb, k = 5, 2
    rng = np.random.default_rng(7)
    sig = rng.poisson(1000.0, size=(k, nb)).astype(float)
    w = 1.0 + 0.1 * (np.linspace(0.0, 1.0, nb) - 0.5)
    sig_up = np.stack([rng.poisson(sig[i] * w) for i in range(k)]).astype(float)
    if local:
        few = np.zeros((k, nb))
        few[:, 1] = 4.0
    else:
        few = np.array([[0.0, 1.0, 2.0, 0.0, 1.0], [2.0, 1.0, 0.0, 3.0, 2.0]])
    few_up = few.copy()
    few_up[1] *= 1.5
    ax = hist.axis.Regular(nb, 0.0, 1.0, name="x")
    tw = TensorWriter(sparse=sparse)
    tw.add_channel([ax], "ch0")
    tw.add_data(_hist1d(ax, sig.sum(0) + few.sum(0)), "ch0")
    tw.add_process(_fold_hist(ax, sig), "sig", "ch0", signal=True, fold_axis="mcfold")
    tw.add_process(_fold_hist(ax, few), "few", "ch0", fold_axis="mcfold")
    tw.add_systematic(_fold_hist(ax, sig_up), "tilt", "sig", "ch0", fold_axis="mcfold")
    tw.add_systematic(
        _fold_hist(ax, few_up), "onefew", "few", "ch0", fold_axis="mcfold"
    )
    name = f"few_event_{'local_' if local else ''}{'sp' if sparse else 'de'}.hdf5"
    return _write(tw, path, name)


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_low_kstat_templates_leave_the_fold_structure(sparse, tmp_path):
    """With --mcStatFoldMinKstat, the few-event templates get 1/k of the full
    nominal in every fold and follow the full-sample variation: their half
    predictions equal the full one at any parameter values. The well-measured
    templates keep their folds and their own fold variations."""
    fn = make_few_event_toy(tmp_path, sparse=sparse)
    f = _fold_fitter(fn, tmp_path, "--mcStatFoldMinKstat", "10", *NO_SHARING)
    procs = [p.decode() if isinstance(p, bytes) else str(p) for p in f.indata.procs]
    isig, ifew = procs.index("sig"), procs.index("few")
    flags = np.asarray(f.mcstat_fold_optout_templates)
    if sparse:
        rows = np.asarray(f.indata.norm.indices)
        flagged_procs = set(rows[flags, 1].tolist())
        assert flagged_procs == {ifew}
    else:
        flags = flags[: f.indata.nbins]
        assert flags[:, ifew].all() and not flags[:, isig].any()
    _set_param(f, "onefew", 2.0)
    _set_param(f, "tilt", 3.0)
    _, norm_full = f._compute_yields_noBBB(full=False, compute_norm=True)
    _, norm_halves = f._compute_fold_yields(halves=True, return_norm=True)
    nb = f.indata.nbins
    full = np.asarray(norm_full)[:nb]
    halves = np.asarray(norm_halves)[:, :nb]
    np.testing.assert_allclose(
        halves[:, :, ifew], np.broadcast_to(full[None, :, ifew], (2, nb)), rtol=1e-12
    )
    assert np.abs(halves[:, :, isig] / full[None, :, isig] - 1.0).max() > 1e-4


def test_low_kstat_opt_out_keeps_the_objective_bounded(tmp_path):
    """Without protection the few-event template's fold variation makes the
    de-biased objective fall without bound along 'onefew'; once the template has
    left the fold structure the objective grows there like the ordinary one."""
    fn = make_few_event_toy(tmp_path, local=True)
    vals = {}
    for kmin in ("0", "10"):
        f = _fold_fitter(
            fn,
            tmp_path,
            "--mcStatFoldMinKstat",
            kmin,
            "--mcStatFoldBound",
            "0",
            *NO_SHARING,
        )
        out = []
        for th in (0.0, 5.0, 10.0, 20.0):
            _set_param(f, "onefew", th)
            out.append(float(f.loss_val_grad_hess()[0]))
        vals[kmin] = np.array(out)
    assert vals["0"][-1] < vals["0"][0] - 1e3
    assert np.all(np.diff(vals["10"]) > 0.0)
