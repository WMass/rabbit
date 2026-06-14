"""Build the 2-process degenerate toy with an MC-stat FOLD axis (k=2) and write a
rabbit tensor exercising the two-half de-biasing (add_process(fold_axis=...)).

Each process's finite-MC template is split into two independent halves (fold 0/1)
via a per-bin binomial(count, 0.5); the full template = fold0 + fold1 is derived
by the writer.  Writes toy_folds.hdf5."""
import os
import numpy as np
import hist
from rabbit.tensorwriter import TensorWriter

NB = 200
A, H1, H2 = 4962.0, 5112.0, 4962.0
rng = np.random.default_rng(20240614)

n_flat = np.full(NB, A)
n_step = np.concatenate([np.full(NB // 2, H1), np.full(NB // 2, H2)])
mu_true = n_flat + n_step
data = mu_true.copy()                       # Asimov data

# finite-MC full templates, then split into two folds per bin
sw_flat = rng.poisson(n_flat)
sw_step = rng.poisson(n_step)
flat_A = rng.binomial(sw_flat, 0.5); flat_B = sw_flat - flat_A
step_A = rng.binomial(sw_step, 0.5); step_B = sw_step - step_A

ax = hist.axis.Regular(NB, 0.0, 1.0, name="x")
axf = hist.axis.IntCategory([0, 1], name="mcfold")


def data_hist(values):
    h = hist.Hist(ax, storage=hist.storage.Weight())
    h.view()["value"] = values
    h.view()["variance"] = values
    return h


def folded_hist(fold0, fold1):
    h = hist.Hist(ax, axf, storage=hist.storage.Weight())
    h.view()["value"][:, 0] = fold0
    h.view()["value"][:, 1] = fold1
    h.view()["variance"][:, 0] = fold0
    h.view()["variance"][:, 1] = fold1
    return h


if __name__ == "__main__":
    out = os.environ.get("OUT", "/tmp/claude")
    tw = TensorWriter(sparse=False)
    tw.add_channel([ax], "ch0")
    tw.add_data(data_hist(data), "ch0")
    tw.add_process(folded_hist(flat_A, flat_B), "flat", "ch0",
                   signal=True, fold_axis="mcfold")
    tw.add_process(folded_hist(step_A, step_B), "step", "ch0",
                   signal=True, fold_axis="mcfold")
    tw.write(outfolder=out, outfilename="toy_folds.hdf5")
    print(f"wrote {out}/toy_folds.hdf5  (k=2 folds for flat, step)")
