"""Build the 2-process degenerate toy with k=4 MC-stat folds, to exercise the
complete k-fold U-statistic curvature (k-fold averaging). Each process's finite-MC
template is split into 4 independent folds (multinomial 1/4 per bin); the full =
sum of folds is derived by the writer. Writes toy_kfold.hdf5."""
import os
import numpy as np
import hist
from rabbit.tensorwriter import TensorWriter

NB = 200
A, H1, H2 = 4962.0, 5112.0, 4962.0
K = 4
rng = np.random.default_rng(20240614)

n_flat = np.full(NB, A)
n_step = np.concatenate([np.full(NB // 2, H1), np.full(NB // 2, H2)])
data = (n_flat + n_step).astype(float)

sw_flat = rng.poisson(n_flat)
sw_step = rng.poisson(n_step)
# split each bin's counts into K folds via a multinomial(count, [1/K]*K)
flat_folds = np.stack([rng.multinomial(c, [1.0 / K] * K) for c in sw_flat], axis=0).T  # [K, NB]
step_folds = np.stack([rng.multinomial(c, [1.0 / K] * K) for c in sw_step], axis=0).T

ax = hist.axis.Regular(NB, 0.0, 1.0, name="x")
axf = hist.axis.IntCategory(list(range(K)), name="mcfold")


def dhist(v):
    h = hist.Hist(ax, storage=hist.storage.Weight())
    h.view()["value"] = v; h.view()["variance"] = v
    return h


def fhist(folds):
    h = hist.Hist(ax, axf, storage=hist.storage.Weight())
    for f in range(K):
        h.view()["value"][:, f] = folds[f]
        h.view()["variance"][:, f] = folds[f]
    return h


if __name__ == "__main__":
    out = os.environ.get("OUT", "/tmp/claude")
    tw = TensorWriter(sparse=False)
    tw.add_channel([ax], "ch0")
    tw.add_data(dhist(data), "ch0")
    tw.add_process(fhist(flat_folds), "flat", "ch0", signal=True, fold_axis="mcfold")
    tw.add_process(fhist(step_folds), "step", "ch0", signal=True, fold_axis="mcfold")
    tw.write(outfolder=out, outfilename="toy_kfold.hdf5")
    print(f"wrote {out}/toy_kfold.hdf5  (k={K} folds)")
