"""Validate the complete k-fold U-statistic curvature (k-fold averaging).

(A) Correctness: rabbit's fisher curvature on the k=4 fold toy equals the numpy
    complete pairwise U-statistic  A_k = k/(k-1)[F_full - sum_i F_i_raw].
(B) Benefit: over an ensemble the U-statistic curvature is unbiased (median ~
    sigma_inf) and its RMS shrinks as k grows (toward the continuous-M floor).

Needs toy_kfold.hdf5 (tests/toy_kfold.py). Run with OUT=<dir>.
"""
import os, sys
import numpy as np
import importlib.util as _ilu

RABBIT_BASE = os.environ.get("RABBIT_BASE", ".")
sys.path.insert(0, os.path.join(RABBIT_BASE, "bin"))
_spec = _ilu.spec_from_file_location(
    "rabbit_fit_main", os.path.join(RABBIT_BASE, "bin", "rabbit_fit.py")
)
_rfm = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_rfm)
from rabbit import inputdata, fitter
from rabbit.param_models import helpers as ph

NB, A, H1, H2 = 200, 4962.0, 5112.0, 4962.0
ddif = np.array([1.0, -1.0]) / np.sqrt(2)
n_flat = np.full(NB, A)
n_step = np.concatenate([np.full(NB // 2, H1), np.full(NB // 2, H2)])
data = (n_flat + n_step).astype(float)


def ustat(Ti, V):
    Jf = sum(Ti)
    Ff = np.einsum("bi,b,bj->ij", Jf, 1 / V, Jf)
    Fi = sum(np.einsum("bi,b,bj->ij", T, 1 / V, T) for T in Ti)
    K = len(Ti)
    return (K / (K - 1.0)) * (Ff - Fi)


def numpy_ref_k4():
    rng = np.random.default_rng(20240614)  # same seed as toy_kfold.py
    swf = rng.poisson(n_flat); sws = rng.poisson(n_step)
    ff = np.stack([rng.multinomial(c, [0.25] * 4) for c in swf], 0).T
    sf = np.stack([rng.multinomial(c, [0.25] * 4) for c in sws], 0).T
    Ti = [np.stack([ff[i], sf[i]], 1).astype(float) for i in range(4)]
    return ustat(Ti, data)


def rabbit_fisher(fn):
    import tensorflow as tf
    a = _rfm.make_parser().parse_args(
        [fn, "-o", "/tmp/claude/vo", "-t", "0", "--chisqFit", "--noBinByBinStat",
         "--allowNegativeParam", "--mcStatDebias", "kfold", "--covMode", "fisher"])
    ind = inputdata.FitInputData(fn, None)
    f = fitter.Fitter(ind, ph.load_models([["Mu"]], ind, **vars(a)), a, do_blinding=False)
    f.defaultassign(); f.set_nobs(ind.data_obs); f.minimize()
    _, _, h = f.loss_val_grad_hess()
    return f.n_folds, np.asarray(f.fisher_curvature(h))


def ensemble_rms(K, ntoy=300):
    rng = np.random.default_rng(100 + K)
    s = []
    for _ in range(ntoy):
        swf = rng.poisson(n_flat); sws = rng.poisson(n_step)
        ff = np.stack([rng.multinomial(c, [1.0 / K] * K) for c in swf], 0).T
        sf = np.stack([rng.multinomial(c, [1.0 / K] * K) for c in sws], 0).T
        Ti = [np.stack([ff[i], sf[i]], 1).astype(float) for i in range(K)]
        Ak = ustat(Ti, data)
        v = ddif @ np.linalg.inv(Ak) @ ddif
        if v > 0:
            s.append(np.sqrt(v))
    return np.median(s), np.std(s)


if __name__ == "__main__":
    out = os.environ.get("OUT", "/tmp/claude")
    A4 = numpy_ref_k4()
    k, Fb = rabbit_fisher(f"{out}/toy_kfold.hdf5")
    print(f"(A) rabbit k={k} fisher curvature == numpy U-statistic A_k: "
          f"{np.allclose(Fb, A4, rtol=1e-4)}  "
          f"(sigma(dif) rabbit={np.sqrt(ddif@np.linalg.inv(Fb)@ddif):.4f}, "
          f"numpy={np.sqrt(ddif@np.linalg.inv(A4)@ddif):.4f})")
    assert np.allclose(Fb, A4, rtol=1e-4)

    Tt = np.stack([n_flat, n_step], 1)
    sinf = np.sqrt(ddif @ np.linalg.inv(np.einsum("bi,b,bj->ij", Tt, 1 / data, Tt)) @ ddif)
    print(f"(B) ensemble: sigma_inf={sinf:.4f}")
    prev = None
    for K in (4, 8, 16):
        m, r = ensemble_rms(K)
        print(f"    k={K:2d}  median={m:.4f}  RMS={r:.4f}")
        assert abs(m - sinf) < 0.02, "median should be ~unbiased"
        if prev is not None:
            assert r < prev + 1e-9, "RMS should not increase with k"
        prev = r
    print("\n  k-fold averaging checks passed (correct, unbiased, variance-reducing).")
