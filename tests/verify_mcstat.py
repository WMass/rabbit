"""Validate the continuous-M de-biasing (add_mc_stat_moment) in rabbit.

Builds a Fitter for toy_noM and toy_M, minimizes, computes the covariance, and
prints the POI uncertainties in physical signal-strength (rnorm) space.

KEY: the `-1/2 theta^T M theta` penalty only de-biases when the POI enters the
prediction LINEARLY.  rabbit's default POI transform is rnorm=x^2 (nonlinear) which
cancels the correction; pass --allowNegativeParam (LINEAR rnorm=x) to see the
de-bias.  See RESULTS.md S9a.  Run with OUT=<dir> pointing at toy_{noM,M}.hdf5.
"""

import importlib.util as _ilu
import os
import sys

import numpy as np

RABBIT_BASE = os.environ.get("RABBIT_BASE", ".")
sys.path.insert(0, os.path.join(RABBIT_BASE, "bin"))
_spec = _ilu.spec_from_file_location(
    "rabbit_fit_main", os.path.join(RABBIT_BASE, "bin", "rabbit_fit.py")
)
_rfm = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_rfm)

from rabbit import fitter, inputdata
from rabbit.param_models import helpers as ph


def run(filename, linear):
    argv = [
        filename,
        "-o",
        "/tmp/claude/verify_out",
        "-t",
        "0",
        "--noBinByBinStat",
        "--chisqFit",
    ]
    if linear:
        argv.append("--allowNegativeParam")
    args = _rfm.make_parser().parse_args(argv)
    indata = inputdata.FitInputData(filename, None)
    param_model = ph.load_models([["Mu"]], indata, **vars(args))
    f = fitter.Fitter(indata, param_model, args, do_blinding=False)
    f.defaultassign()
    f.set_nobs(indata.data_obs)
    f.minimize()
    _, grad, hess = f.loss_val_grad_hess()
    _, cov = f.edmval_cov(grad, hess)
    C = np.asarray(cov)  # curvature cov A^-1
    Csand = None
    if f.mcstat_M is not None:
        Csand = np.asarray(f.cov_mcstat_sandwich(hess))  # A^-1 + A^-1 M A^-1
    x = f.x.numpy()
    rnorm = x if linear else x**2  # physical signal strength
    J = np.diag(np.ones_like(x) if linear else 2 * x)  # d rnorm / d x
    Cr = J @ C @ J  # curvature cov in rnorm space
    Csr = None if Csand is None else J @ Csand @ J
    return f.parms.astype(str), rnorm, Cr, Csr


if __name__ == "__main__":
    out = os.environ.get("OUT", "/tmp/claude")
    ddif = np.array([1.0, -1.0]) / np.sqrt(2)
    for linear in (False, True):
        print(
            f"\n=== POI {'LINEAR (--allowNegativeParam)' if linear else 'SQUARED (rabbit default)'} ==="
        )
        res = {}
        for tag in ("noM", "M"):
            parms, rnorm, Cr, Csr = run(f"{out}/toy_{tag}.hdf5", linear)
            res[tag] = (rnorm, Cr, Csr)
            extra = ""
            if Csr is not None:
                extra = (
                    f"  sandwich(flat)={np.sqrt(Csr[0,0]):.4f}  "
                    f"sandwich(dif)={np.sqrt(ddif@Csr@ddif):.4f}"
                )
            print(
                f"  {tag:3s} rnorm={np.round(rnorm, 4)}  "
                f"curv_rnorm(flat)={np.sqrt(Cr[0,0]):.4f}  "
                f"curv_rnorm(dif)={np.sqrt(ddif@Cr@ddif):.4f}{extra}"
            )
        s_no = np.sqrt(ddif @ res["noM"][1] @ ddif)
        s_M = np.sqrt(ddif @ res["M"][1] @ ddif)
        verdict = (
            "DE-BIASED (sigma inflated)"
            if s_M > 1.2 * s_no
            else "no de-bias (cancelled)"
        )
        print(f"  -> curvature sigma(dif): {s_no:.4f} -> {s_M:.4f}   {verdict}")
        if res["M"][2] is not None:
            s_sand = np.sqrt(ddif @ res["M"][2] @ ddif)
            print(
                f"  -> sandwich sigma(dif):  {s_sand:.4f}   "
                f"(>= curvature {s_M:.4f}: robust over-coverage margin)"
            )
