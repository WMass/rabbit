"""Validate MC-stat fold de-biasing in SPARSE mode: the sparse per-fold yield path
(scatter the shared systematic factor into a dense [nbinsfull,nproc] grid, contract
with the dense fold norm) must give EXACTLY the same de-biased point, fisher
curvature, and sandwich as the equivalent DENSE tensor. Run with OUT=<dir>."""

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

NB, K = 40, 4
OUT = os.environ.get("OUT", "/tmp/claude")


def build(sparse):
    rng = np.random.default_rng(13)
    nom = np.linspace(900, 1100, NB)
    data = nom.copy()
    full = rng.poisson(nom)
    folds = np.stack([rng.multinomial(c, [1.0 / K] * K) for c in full], 0).T
    ax = hist.axis.Regular(NB, 0, 1, name="x")
    axf = hist.axis.IntCategory(list(range(K)), name="mcfold")

    def dh(v):
        h = hist.Hist(ax, storage=hist.storage.Weight())
        h.view()["value"] = v
        h.view()["variance"] = v
        return h

    def fh(fl):
        h = hist.Hist(ax, axf, storage=hist.storage.Weight())
        for f in range(K):
            h.view()["value"][:, f] = fl[f]
            h.view()["variance"][:, f] = fl[f]
        return h

    x = np.linspace(0, 1, NB)
    tw = TensorWriter(sparse=sparse)
    tw.add_channel([ax], "ch0")
    tw.add_data(dh(data), "ch0")
    tw.add_process(fh(folds), "sig", "ch0", signal=True, fold_axis="mcfold")
    tw.add_systematic(dh(nom * (1 + 0.1 * (x - 0.5))), "shp", "sig", "ch0")
    fn = f"{OUT}/toy_sf_{'sp' if sparse else 'de'}.hdf5"
    tw.write(outfolder=os.path.dirname(fn), outfilename=os.path.basename(fn))
    return fn


def run(fn):
    a = _rfm.make_parser().parse_args(
        [
            fn,
            "-o",
            f"{OUT}/vo",
            "-t",
            "0",
            "--chisqFit",
            "--noBinByBinStat",
            "--allowNegativeParam",
            "--mcStatDebias",
            "kfold",
            "--covMode",
            "fisher",
        ]
    )
    ind = inputdata.FitInputData(fn, None)
    f = fitter.Fitter(
        ind, ph.load_models([["Mu"]], ind, **vars(a)), a, do_blinding=False
    )
    f.defaultassign()
    f.set_nobs(ind.data_obs)
    f.minimize()
    _, _, h = f.loss_val_grad_hess()
    bread = f.fisher_curvature(h)
    S = np.asarray(f.cov_twohalf_sandwich(bread, "fisher"))
    return ind.sparse, f.x.numpy(), np.asarray(bread), S


if __name__ == "__main__":
    sp, xs, bs, Ss = run(build(True))
    de, xd, bd, Sd = run(build(False))
    print(f"sparse={sp} x={np.round(xs,5)} | dense={not de or de} x={np.round(xd,5)}")
    ok_x = np.allclose(xs, xd, rtol=1e-6)
    ok_b = np.allclose(bs, bd, rtol=1e-6)
    ok_s = np.allclose(Ss, Sd, rtol=1e-6)
    print(
        f"  point match: {ok_x}   fisher curvature match: {ok_b}   sandwich match: {ok_s}"
    )
    assert ok_x and ok_b and ok_s, "sparse fold de-bias must equal dense"
    print("\n  sparse fold de-biasing == dense (point, curvature, sandwich). passed.")
