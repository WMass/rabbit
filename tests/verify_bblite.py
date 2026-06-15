"""Validate MC-stat de-biasing composed with bin-by-bin stat (BB-lite) ON.

(A) continuous-M + BB-lite: M MUST use the BB-lite-inflated variance
    (mu_b + sum_proc sumw2) in its denominator, else H - M is non-PD and the fit
    diverges. Builds toy_Mbb with the correct denominator and checks it converges
    PD.
(B) two-half + BB-lite: the full-sample profiled beta is shared with the half
    predictions (fitter), so the jackknife stays bounded/PD; BB-lite inflates the
    baseline and two-half de-biases on top. Uses the well-posed split-logk toy.

Run with OUT=<dir>. Needs toy_splitlogk.hdf5 (tests/toy_splitlogk.py) present.
"""

import importlib.util as _ilu
import os
import sys

import hist
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
from rabbit.tensorwriter import TensorWriter

OUT = os.environ.get("OUT", "/tmp/claude")


def build_toy_Mbb():
    NB = 200
    A, H1, H2 = 4962.0, 5112.0, 4962.0
    rng = np.random.default_rng(20240614)
    n_flat = np.full(NB, A)
    n_step = np.concatenate([np.full(NB // 2, H1), np.full(NB // 2, H2)])
    data = (n_flat + n_step).astype(float)
    sw_flat = rng.poisson(n_flat).astype(float)
    sw_step = rng.poisson(n_step).astype(float)
    ax = hist.axis.Regular(NB, 0.0, 1.0, name="x")

    def wh(v, var):
        h = hist.Hist(ax, storage=hist.storage.Weight())
        h.view()["value"] = v
        h.view()["variance"] = var
        return h

    tw = TensorWriter(sparse=False)
    tw.add_channel([ax], "ch0")
    tw.add_data(wh(data, data), "ch0")
    tw.add_process(wh(sw_flat, sw_flat), "flat", "ch0", signal=True)
    tw.add_process(wh(sw_step, sw_step), "step", "ch0", signal=True)
    Vbb = data + (sw_flat + sw_step)  # BB-lite inflated variance
    M = np.diag([np.sum(sw_flat / Vbb), np.sum(sw_step / Vbb)])
    tw.add_mc_stat_moment(M, ["flat", "step"])
    tw.write(outfolder=OUT, outfilename="toy_Mbb.hdf5")


def fit(fn, debias, bbb, sandwich=None):
    argv = [
        fn,
        "-o",
        f"{OUT}/vo",
        "-t",
        "0",
        "--chisqFit",
        "--allowNegativeParam",
        "--mcStatDebias",
        debias,
    ]
    if not bbb:
        argv.append("--noBinByBinStat")
    a = _rfm.make_parser().parse_args(argv)
    ind = inputdata.FitInputData(fn, None)
    f = fitter.Fitter(
        ind, ph.load_models([["Mu"]], ind, **vars(a)), a, do_blinding=False
    )
    f.defaultassign()
    f.set_nobs(ind.data_obs)
    f.minimize()
    _, g, h = f.loss_val_grad_hess()
    _, cov = f.edmval_cov(g, h)
    C = np.asarray(cov)
    S = None
    if sandwich == "continuousM":
        S = np.asarray(f.cov_mcstat_sandwich(h))
    elif sandwich == "twoHalf":
        S = np.asarray(f.cov_twohalf_sandwich(h, "observed"))
    eig = np.linalg.eigvalsh(h.numpy())
    return f.x.numpy(), np.sqrt(np.diag(C)), S, float(np.max(np.abs(g.numpy()))), eig


if __name__ == "__main__":
    print("(A) continuous-M + BB-lite (toy_Mbb, BB-variance M):")
    build_toy_Mbb()
    x, sig, S, gmax, eig = fit(
        f"{OUT}/toy_Mbb.hdf5", "continuousM", True, "continuousM"
    )
    pd = bool(np.all(eig > 0))
    print(f"    x={np.round(x,4)} |grad|={gmax:.1e} PD={pd} hess_eig={np.round(eig,2)}")
    assert pd and gmax < 1e-3, "continuous-M + BB-lite did not converge PD"

    print("(B) two-half + BB-lite (toy_splitlogk):")
    fn = f"{OUT}/toy_splitlogk.hdf5"
    x0, sig0, _, g0, e0 = fit(fn, "none", True)
    x1, sig1, S1, g1, e1 = fit(fn, "twoHalf", True, "twoHalf")
    print(f"    none    x={np.round(x0,4)} tilt sigma={sig0[1]:.4f}")
    print(
        f"    twoHalf x={np.round(x1,4)} tilt curv={sig1[1]:.4f} sandwich={np.sqrt(S1[1,1]):.4f}"
    )
    assert bool(np.all(e1 > 0)) and g1 < 1e-3, "two-half + BB-lite not converged PD"
    assert np.sqrt(S1[1, 1]) > sig0[1], "two-half should inflate the tilt uncertainty"
    print("\n  ALL BB-lite composition checks passed.")
