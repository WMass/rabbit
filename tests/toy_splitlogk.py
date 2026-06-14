"""Small toy with a FOLDED process and a FOLDED systematic, to exercise the
split-logk two-half path (add_systematic(fold_axis=...)).  The systematic's
up-variation is built from each fold's OWN events (a per-fold reweight), so its
logk carries MC noise that differs between folds -> split logk is meaningful.

Writes toy_splitlogk.hdf5 (folded syst) and toy_sharedlogk.hdf5 (same nominal
folds, systematic added WITHOUT fold_axis = shared logk) for comparison."""
import os
import numpy as np
import hist
from rabbit.tensorwriter import TensorWriter

NB = 50
rng = np.random.default_rng(7)
nom = np.linspace(800.0, 1200.0, NB)            # smooth nominal shape
data = nom.copy()                               # Asimov

# two independent half-samples of the nominal (per-bin binomial split)
full = rng.poisson(nom)
A = rng.binomial(full, 0.5); B = full - A

# systematic: a tilt reweight applied to each fold's OWN events -> per-fold
# up template = fold * (1 + 0.15*(x-0.5)) with fold-specific Poisson noise
x = np.linspace(0, 1, NB)
w = 1.0 + 0.15 * (x - 0.5)
upA = rng.poisson(np.maximum(A * w, 0.0))
upB = rng.poisson(np.maximum(B * w, 0.0))

ax = hist.axis.Regular(NB, 0.0, 1.0, name="x")
axf = hist.axis.IntCategory([0, 1], name="mcfold")


def dhist(v):
    h = hist.Hist(ax, storage=hist.storage.Weight())
    h.view()["value"] = v; h.view()["variance"] = v
    return h


def fhist(f0, f1):
    h = hist.Hist(ax, axf, storage=hist.storage.Weight())
    h.view()["value"][:, 0] = f0; h.view()["value"][:, 1] = f1
    h.view()["variance"][:, 0] = f0; h.view()["variance"][:, 1] = f1
    return h


def build(outname, split):
    tw = TensorWriter(sparse=False)
    tw.add_channel([ax], "ch0")
    tw.add_data(dhist(data), "ch0")
    tw.add_process(fhist(A, B), "sig", "ch0", signal=True, fold_axis="mcfold")
    if split:
        tw.add_systematic(fhist(upA, upB), "tilt", "sig", "ch0", fold_axis="mcfold")
    else:
        # shared logk: full up template (folds summed), no fold_axis
        tw.add_systematic(dhist(upA + upB), "tilt", "sig", "ch0")
    tw.write(outfolder=os.path.dirname(outname) or ".",
             outfilename=os.path.basename(outname))


if __name__ == "__main__":
    out = os.environ.get("OUT", "/tmp/claude")
    build(f"{out}/toy_splitlogk.hdf5", split=True)
    build(f"{out}/toy_sharedlogk.hdf5", split=False)
    print(f"wrote {out}/toy_splitlogk.hdf5 (split) and toy_sharedlogk.hdf5 (shared)")
