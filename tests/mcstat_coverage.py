"""Native template-fluctuating ensemble coverage harness for the MC-stat de-biasing.

Unlike rabbit's built-in toys (which fluctuate the DATA only), this fluctuates the
TEMPLATES (the MC noise floor) AND the data per pseudo-experiment, builds a rabbit
tensor, fits it, and measures the coverage of a POI combination. This is the single
metric the de-bias is supposed to restore (none undercovers; debias+sandwich ~0.683).

Two near-degenerate processes (corr tunable via STEP), finite MC (oversample:1),
linear POIs (rnorm_true = 1). Coverage is measured for the degenerate direction
(rnorm0 - rnorm1, truth 0) and the well-constrained sum.

Usage:
    OUT=/tmp/cov python3 tests/mcstat_coverage.py            # prints a table
    from mcstat_coverage import run_coverage                  # programmatic
"""
import os, sys
import numpy as np
import hist
import importlib.util as _ilu

RABBIT_BASE = os.environ.get("RABBIT_BASE", ".")
sys.path.insert(0, os.path.join(RABBIT_BASE, "bin"))
_spec = _ilu.spec_from_file_location(
    "rabbit_fit_main", os.path.join(RABBIT_BASE, "bin", "rabbit_fit.py")
)
_rfm = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_rfm)
from rabbit import inputdata, fitter
from rabbit.tensorwriter import TensorWriter
from rabbit.param_models import helpers as ph

OUT = os.environ.get("OUT", "/tmp/claude")
# The near-degenerate 2-process toy from the slides (RESULTS §1): a flat process
# and a "step" process identical to it in the 2nd half and +3% in the 1st half.
# This sits in the regime where the de-biased CURVATURE ~ sigma_inf (M/lambda_min
# ~0.6, so H-M is comfortably positive and the de-biased point is well-behaved) --
# the naive sigma is ~60% of true so the naive interval undercovers (~0.45), and
# the de-bias restores it.
NB = int(os.environ.get("NB", "200"))
FLAT = 4962.0
HI = float(os.environ.get("HI", "5112"))   # step height; larger = less degenerate
N0 = np.full(NB, FLAT)
N1 = np.concatenate([np.full(NB // 2, HI), np.full(NB // 2, FLAT)])
# TRUE signal strengths. The degenerate (difference) direction must have a NONZERO
# true value, else the MC-stat attenuation (which biases toward 0) leaves it ~0 and
# there is no central-value bias for the de-bias to correct. R0!=R1 puts a real
# value on the difference so naive attenuates it (biased) and the de-bias restores it.
R0 = float(os.environ.get("R0", "1.3"))
R1 = float(os.environ.get("R1", "0.7"))
RTRUE = np.array([R0, R1])
MU = R0 * N0 + R1 * N1                          # true total (data expectation)
AX = hist.axis.Regular(NB, 0.0, 1.0, name="x")
DDIF = np.array([1.0, -1.0]) / np.sqrt(2.0)
DSUM = np.array([1.0, 1.0]) / np.sqrt(2.0)


def _whist(values, variances):
    h = hist.Hist(AX, storage=hist.storage.Weight())
    h.view()["value"] = values
    h.view()["variance"] = variances
    return h


def _fhist(folds):
    axf = hist.axis.IntCategory(list(range(folds.shape[0])), name="mcfold")
    h = hist.Hist(AX, axf, storage=hist.storage.Weight())
    for f in range(folds.shape[0]):
        h.view()["value"][:, f] = folds[f]
        h.view()["variance"][:, f] = folds[f]
    return h


def _build(fn, T0, T1, folds0, folds1, data, debias, bbb, poisson):
    tw = TensorWriter(sparse=False)
    tw.add_channel([AX], "ch0")
    tw.add_data(_whist(data, data), "ch0")
    if debias in ("twoHalf", "kfold"):
        tw.add_process(_fhist(folds0), "p0", "ch0", signal=True, fold_axis="mcfold")
        tw.add_process(_fhist(folds1), "p1", "ch0", signal=True, fold_axis="mcfold")
    else:
        tw.add_process(_whist(T0, T0), "p0", "ch0", signal=True)
        tw.add_process(_whist(T1, T1), "p1", "ch0", signal=True)
    if debias == "continuousM":
        # M_pp = sum_b sumw2_p / var_b ; var = data (chisq) or data+sumw2 (BB-lite)
        var = data + (T0 + T1) if bbb else data
        M = np.diag([np.sum(T0 / var), np.sum(T1 / var)])
        tw.add_mc_stat_moment(M, ["p0", "p1"])
    tw.write(outfolder=os.path.dirname(fn), outfilename=os.path.basename(fn))


def _fit(fn, debias, covMode, debiasCov, bbb, poisson):
    argv = [fn, "-o", f"{OUT}/vo", "-t", "0", "--allowNegativeParam",
            "--mcStatDebias", debias, "--covMode", covMode,
            "--mcStatDebiasCov", debiasCov]
    argv += [] if poisson else ["--chisqFit"]
    if not bbb:
        argv.append("--noBinByBinStat")
    a = _rfm.make_parser().parse_args(argv)
    ind = inputdata.FitInputData(fn, None)
    f = fitter.Fitter(ind, ph.load_models([["Mu"]], ind, **vars(a)), a, do_blinding=False)
    f.defaultassign(); f.set_nobs(ind.data_obs); f.minimize()
    _, grad, hess = f.loss_val_grad_hess()
    bread = f.fisher_curvature(hess) if covMode == "fisher" else hess
    _, cov_curv = f.edmval_cov(grad, bread)
    if debiasCov == "sandwich" and debias == "continuousM":
        C = np.asarray(f.cov_mcstat_sandwich(bread))
    elif debiasCov == "sandwich" and debias in ("twoHalf", "kfold"):
        C = np.asarray(f.cov_twohalf_sandwich(bread, covMode))
    else:
        C = np.asarray(cov_curv)
    return f.x.numpy(), C, float(np.max(np.abs(grad.numpy())))


def run_coverage(debias="none", covMode="observed", debiasCov="sandwich",
                 bbb=False, poisson=False, ntoy=200, k=2, seed=1):
    rng = np.random.default_rng(seed)
    fn = f"{OUT}/cov_{debias}_{covMode}_{debiasCov}_{int(bbb)}_{int(poisson)}.hdf5"
    cov_dif = cov_sum = 0
    bias_dif = []
    res_dif = []
    sig_dif = []
    nbad = 0
    for _ in range(ntoy):
        folds0 = np.stack([rng.poisson(N0 / k) for _ in range(k)], 0).astype(float)
        folds1 = np.stack([rng.poisson(N1 / k) for _ in range(k)], 0).astype(float)
        T0, T1 = folds0.sum(0), folds1.sum(0)
        data = rng.poisson(MU).astype(float)
        try:
            _build(fn, T0, T1, folds0, folds1, data, debias, bbb, poisson)
            x, C, gmax = _fit(fn, debias, covMode, debiasCov, bbb, poisson)
            if gmax > 1e-2 or not np.all(np.linalg.eigvalsh(C) > 0):
                nbad += 1; continue
        except Exception:
            nbad += 1; continue
        r = x - RTRUE                                 # residual vs true signal strengths
        s_dif = np.sqrt(DDIF @ C @ DDIF)
        s_sum = np.sqrt(DSUM @ C @ DSUM)
        cov_dif += abs(DDIF @ r) < s_dif
        cov_sum += abs(DSUM @ r) < s_sum
        res_dif.append(DDIF @ r); sig_dif.append(s_dif)
    n = ntoy - nbad
    res = np.array(res_dif)
    return dict(
        cov_dif=cov_dif / n, cov_sum=cov_sum / n, n=n, nbad=nbad,
        # median residual = robust bias (mean is corrupted by the heavy tails the
        # de-bias develops near the singular limit); rms_res = the actual point
        # scatter (vs med_sig = the reported uncertainty -> coverage = rms/med).
        bias_dif=float(np.median(res)), mean_res=float(np.mean(res)),
        rms_res=float(np.std(res)), med_sig=float(np.median(sig_dif)),
    )


if __name__ == "__main__":
    ntoy = int(os.environ.get("NTOY", "300"))
    configs = [
        ("none      observed sandwich  chisq", dict(debias="none")),
        ("continuousM observed sandwich chisq", dict(debias="continuousM")),
        ("continuousM observed curvature chisq", dict(debias="continuousM", debiasCov="curvature")),
        ("continuousM fisher  sandwich  chisq", dict(debias="continuousM", covMode="fisher")),
        ("twoHalf    observed sandwich  chisq", dict(debias="twoHalf")),
        ("twoHalf    fisher   sandwich  chisq", dict(debias="twoHalf", covMode="fisher")),
        ("kfold(8)   fisher   sandwich  chisq", dict(debias="kfold", covMode="fisher", k=8)),
        ("none       observed sandwich  chisq BBB", dict(debias="none", bbb=True)),
        ("continuousM observed sandwich chisq BBB", dict(debias="continuousM", bbb=True)),
        ("twoHalf    observed sandwich  chisq BBB", dict(debias="twoHalf", bbb=True)),
        ("none       observed sandwich  POISSON", dict(debias="none", poisson=True)),
        ("twoHalf    observed sandwich  POISSON", dict(debias="twoHalf", poisson=True)),
    ]
    print(f"MC-stat coverage ensemble (ntoy={ntoy}, target cov(dif)=0.683)\n")
    print(f"{'config':40s}{'cov(dif)':>9s}{'cov(sum)':>9s}{'bias(dif)':>11s}{'med_sig':>9s}{'nbad':>6s}")
    for label, kw in configs:
        r = run_coverage(ntoy=ntoy, **kw)
        print(f"{label:40s}{r['cov_dif']:9.3f}{r['cov_sum']:9.3f}"
              f"{r['bias_dif']:+11.4f}{r['med_sig']:9.4f}{r['nbad']:6d}")
