"""Validate rabbit's two-half (jackknife) MC-stat de-biasing against an
independent numpy reference, on the folded degenerate toy (toy_folds.hdf5).

Compares: standard fit (no debias), two-half curvature A^-1, two-half sandwich
A^-1 H A^-1.  Uses a LINEAR POI (--allowNegativeParam) so the closed-form numpy
reference applies.  Run with OUT=<dir> pointing at toy_folds.hdf5."""
import os, sys
import numpy as np
import importlib.util as _ilu

RABBIT_BASE = os.environ.get("RABBIT_BASE", ".")
sys.path.insert(0, os.path.join(RABBIT_BASE, "bin"))
_spec = _ilu.spec_from_file_location(
    "rabbit_fit_main", os.path.join(RABBIT_BASE, "bin", "rabbit_fit.py")
)
_rfm = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_rfm)
from rabbit import inputdata, fitter
from rabbit.param_models import helpers as ph

ddif = np.array([1.0, -1.0]) / np.sqrt(2)


def rabbit_fit(filename, debias):
    argv = [filename, "-o", "/tmp/claude/verify_out", "-t", "0",
            "--noBinByBinStat", "--chisqFit", "--allowNegativeParam",
            "--mcStatDebias", debias]
    args = _rfm.make_parser().parse_args(argv)
    indata = inputdata.FitInputData(filename, None)
    pm = ph.load_models([["Mu"]], indata, **vars(args))
    f = fitter.Fitter(indata, pm, args, do_blinding=False)
    f.defaultassign(); f.set_nobs(indata.data_obs); f.minimize()
    _, grad, hess = f.loss_val_grad_hess()
    _, cov = f.edmval_cov(grad, hess)
    curv = np.asarray(cov)
    sand = None
    if debias in ("twoHalf", "kfold"):
        sand = np.asarray(f.cov_twohalf_sandwich(hess))
    return f.x.numpy(), curv, sand


def numpy_ref():
    # regenerate the SAME folds as tests/toy_twohalf.py
    NB = 200; A, H1, H2 = 4962.0, 5112.0, 4962.0
    rng = np.random.default_rng(20240614)
    n_flat = np.full(NB, A)
    n_step = np.concatenate([np.full(NB // 2, H1), np.full(NB // 2, H2)])
    data = (n_flat + n_step).astype(float)
    sw_flat = rng.poisson(n_flat); sw_step = rng.poisson(n_step)
    flat_A = rng.binomial(sw_flat, 0.5); flat_B = sw_flat - flat_A
    step_A = rng.binomial(sw_step, 0.5); step_B = sw_step - step_A
    Tf = np.stack([sw_flat, sw_step], 1).astype(float)          # full
    TA = 2 * np.stack([flat_A, step_A], 1).astype(float)        # half A (x2)
    TB = 2 * np.stack([flat_B, step_B], 1).astype(float)        # half B (x2)
    V = data

    def H(T):  return np.einsum('bi,bj,b->ij', T, T, 1 / V)
    def b(T):  return np.einsum('bi,b,b->i', T, 1 / V, data)
    Hf = H(Tf)
    A_cf = 2 * Hf - 0.5 * H(TA) - 0.5 * H(TB)                   # jackknife Hessian
    b_cf = 2 * b(Tf) - 0.5 * b(TA) - 0.5 * b(TB)
    r_std = np.linalg.solve(Hf, b(Tf))
    r_cf = np.linalg.solve(A_cf, b_cf)
    Cstd = np.linalg.inv(Hf)
    Ccurv = np.linalg.inv(A_cf)
    Csand = Ccurv @ Hf @ Ccurv
    return r_std, Cstd, r_cf, Ccurv, Csand


if __name__ == "__main__":
    out = os.environ.get("OUT", "/tmp/claude")
    fn = f"{out}/toy_folds.hdf5"

    r_std, Cstd, r_cf, Ccurv, Csand = numpy_ref()
    print("NUMPY REFERENCE:")
    print(f"  std   r={np.round(r_std,4)}  sigma(dif)={np.sqrt(ddif@Cstd@ddif):.4f}")
    print(f"  cf    r={np.round(r_cf,4)}  curv(dif)={np.sqrt(ddif@Ccurv@ddif):.4f}  "
          f"sandwich(dif)={np.sqrt(ddif@Csand@ddif):.4f}")

    xr_std, Cr_std, _ = rabbit_fit(fn, "none")
    xr_cf, Cr_curv, Cr_sand = rabbit_fit(fn, "twoHalf")
    print("\nRABBIT:")
    print(f"  std   r={np.round(xr_std,4)}  sigma(dif)={np.sqrt(ddif@Cr_std@ddif):.4f}")
    print(f"  cf    r={np.round(xr_cf,4)}  curv(dif)={np.sqrt(ddif@Cr_curv@ddif):.4f}  "
          f"sandwich(dif)={np.sqrt(ddif@Cr_sand@ddif):.4f}")

    ok = (np.allclose(xr_cf, r_cf, atol=1e-3)
          and np.isclose(np.sqrt(ddif@Cr_curv@ddif), np.sqrt(ddif@Ccurv@ddif), atol=1e-3)
          and np.isclose(np.sqrt(ddif@Cr_sand@ddif), np.sqrt(ddif@Csand@ddif), atol=1e-3))
    print(f"\n  rabbit == numpy reference: {ok}")
