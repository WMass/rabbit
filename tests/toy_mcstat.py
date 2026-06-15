"""Build the 2-process near-degenerate MC-stat toy as a rabbit input tensor and
test the continuous-M de-biasing (add_mc_stat_moment).  Writes two tensors:
  toy_noM.hdf5  : fluctuated finite-MC templates, no de-bias
  toy_M.hdf5    : same + frozen noise-floor matrix M over the 2 POIs
A chisqFit on each should give a *larger*, de-biased POI uncertainty with M
(curvature H - M; RABBIT_MCSTAT_DESIGN §1)."""

import os

import hist
import numpy as np

from rabbit.tensorwriter import TensorWriter

NB = 200
A, H1, H2 = 4962.0, 5112.0, 4962.0
rng = np.random.default_rng(20240614)

n_flat = np.full(NB, A)
n_step = np.concatenate([np.full(NB // 2, H1), np.full(NB // 2, H2)])
mu_true = n_flat + n_step  # true total per bin

# finite-MC (1:1) templates: one Poisson realization, sumw2 = counts (unit weights)
sw_flat = rng.poisson(n_flat).astype(float)
sw_step = rng.poisson(n_step).astype(float)
data = mu_true.copy()  # Asimov data (clean curvature comparison)

ax = hist.axis.Regular(NB, 0.0, 1.0, name="x")


def whist(values, variances):
    h = hist.Hist(ax, storage=hist.storage.Weight())
    h.view()["value"] = values
    h.view()["variance"] = variances
    return h


def build(outname, with_M):
    tw = TensorWriter(sparse=False)
    tw.add_channel([ax], "ch0")
    tw.add_data(whist(data, data), "ch0")
    tw.add_process(whist(sw_flat, sw_flat), "flat", "ch0", signal=True)
    tw.add_process(whist(sw_step, sw_step), "step", "ch0", signal=True)
    if with_M:
        # noise floor over the 2 POIs (process-norm scores -> diagonal):
        # M_jj = sum_b sumw2[b,j] / mu_b   (Gaussian/chisq weighting var_b = data_b)
        Mflat = np.sum(sw_flat / data)
        Mstep = np.sum(sw_step / data)
        M = np.diag([Mflat, Mstep])
        tw.add_mc_stat_moment(M, ["flat", "step"])
        print(f"  M = diag({Mflat:.2f}, {Mstep:.2f})")
    tw.write(
        outfolder=os.path.dirname(outname) or ".", outfilename=os.path.basename(outname)
    )


if __name__ == "__main__":
    out = os.environ.get("OUT", "/tmp/claude")
    print("building toy_noM.hdf5")
    build(f"{out}/toy_noM.hdf5", with_M=False)
    print("building toy_M.hdf5")
    build(f"{out}/toy_M.hdf5", with_M=True)
