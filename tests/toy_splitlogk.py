"""Small toy with a FOLDED process and a FOLDED systematic, to exercise the
split-logk path (add_systematic(fold_axis=...)).  The systematic's up-variation is
built from each fold's OWN events (a per-fold reweight), so its logk carries MC
noise that differs between folds -> split logk is meaningful.

Writes toy_splitlogk.hdf5 (folded syst) and toy_sharedlogk.hdf5 (same nominal
folds, systematic added WITHOUT fold_axis = shared logk) for comparison.  Set the
env var K to the number of folds (default 2; K>=3 exercises k>2 split-logk, whose
per-fold logk feeds the k-fold U-statistic curvature in --covMode fisher)."""

import os

import hist
import numpy as np

from rabbit.tensorwriter import TensorWriter

NB = 50
K = int(os.environ.get("K", "2"))
rng = np.random.default_rng(7)
nom = np.linspace(800.0, 1200.0, NB)  # smooth nominal shape
data = nom.copy()  # Asimov

# K independent folds of the nominal (per-bin multinomial split)
full = rng.poisson(nom)
folds = np.stack([rng.multinomial(c, [1.0 / K] * K) for c in full], axis=0).T  # [K, NB]

# systematic: a tilt reweight applied to each fold's OWN events -> per-fold up
# template with fold-specific Poisson noise
x = np.linspace(0, 1, NB)
w = 1.0 + 0.15 * (x - 0.5)
upfolds = np.stack(
    [rng.poisson(np.maximum(folds[f] * w, 0.0)) for f in range(K)], axis=0
)

ax = hist.axis.Regular(NB, 0.0, 1.0, name="x")
axf = hist.axis.IntCategory(list(range(K)), name="mcfold")


def dhist(v):
    h = hist.Hist(ax, storage=hist.storage.Weight())
    h.view()["value"] = v
    h.view()["variance"] = v
    return h


def fhist(fl):
    h = hist.Hist(ax, axf, storage=hist.storage.Weight())
    for f in range(K):
        h.view()["value"][:, f] = fl[f]
        h.view()["variance"][:, f] = fl[f]
    return h


def build(outname, split):
    tw = TensorWriter(sparse=False)
    tw.add_channel([ax], "ch0")
    tw.add_data(dhist(data), "ch0")
    tw.add_process(fhist(folds), "sig", "ch0", signal=True, fold_axis="mcfold")
    if split:
        tw.add_systematic(fhist(upfolds), "tilt", "sig", "ch0", fold_axis="mcfold")
    else:
        # shared logk: full up template (folds summed), no fold_axis
        tw.add_systematic(dhist(upfolds.sum(0)), "tilt", "sig", "ch0")
    tw.write(
        outfolder=os.path.dirname(outname) or ".", outfilename=os.path.basename(outname)
    )


if __name__ == "__main__":
    out = os.environ.get("OUT", "/tmp/claude")
    build(f"{out}/toy_splitlogk.hdf5", split=True)
    build(f"{out}/toy_sharedlogk.hdf5", split=False)
    print(
        f"wrote {out}/toy_splitlogk.hdf5 (split) and toy_sharedlogk.hdf5 (shared), K={K}"
    )
