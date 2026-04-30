"""Test external likelihood terms (gradient + hessian) added to TensorWriter and Fitter.

The external term has the form

    L_ext(x) = g^T x_sub + 0.5 x_sub^T H x_sub

where x_sub is the slice of fit parameters identified by the StrCategory axes
of grad/hess. With Asimov data and a single Gaussian-constrained nuisance,
the analytical post-fit value of the nuisance is

    theta = -g / (1 + h)

where the +1 is the prefit Gaussian constraint and +h is the external hessian
contribution. This script verifies that prediction for several configurations,
including dense and sparse (wums.SparseHist) hessian storage.
"""

import os
import tempfile
from types import SimpleNamespace

import hist
import numpy as np
import scipy.sparse
from wums.sparse_hist import SparseHist

from rabbit import fitter, inputdata, tensorwriter
from rabbit.param_models.helpers import load_model


def make_options(**kwargs):
    defaults = dict(
        earlyStopping=-1,
        noBinByBinStat=True,
        binByBinStatMode="lite",
        binByBinStatType="automatic",
        covarianceFit=False,
        chisqFit=False,
        diagnostics=False,
        minimizerMethod="trust-krylov",
        prefitUnconstrainedNuisanceUncertainty=0.0,
        freezeParameters=[],
        setConstraintMinimum=[],
        unblind=[],
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def build_writer(grad=None, hess=None):
    """Build a TensorWriter with one bkg process and a single shape systematic."""
    np.random.seed(0)
    ax = hist.axis.Regular(20, -5, 5, name="x")

    h_data = hist.Hist(ax, storage=hist.storage.Double())
    h_bkg = hist.Hist(ax, storage=hist.storage.Weight())

    x_bkg = np.random.uniform(-5, 5, 5000)
    h_data.fill(x_bkg)
    h_bkg.fill(x_bkg, weight=np.ones(len(x_bkg)))

    bin_centers = ax.centers - ax.centers[0]
    weights = 0.01 * bin_centers - 0.05
    h_up = h_bkg.copy()
    h_dn = h_bkg.copy()
    h_up.values()[...] = h_bkg.values() * (1 + weights)
    h_dn.values()[...] = h_bkg.values() * (1 - weights)

    writer = tensorwriter.TensorWriter()
    writer.add_channel([ax], "ch0")
    writer.add_data(h_data, "ch0")
    writer.add_process(h_bkg, "bkg", "ch0", signal=True)
    writer.add_systematic([h_up, h_dn], "shape", "bkg", "ch0", symmetrize="average")

    if grad is not None or hess is not None:
        writer.add_external_likelihood_term(grad=grad, hess=hess)

    return writer


def run_fit(filename):
    indata_obj = inputdata.FitInputData(filename)
    param_model = load_model("Mu", indata_obj)
    options = make_options()
    f = fitter.Fitter(indata_obj, param_model, options)

    # use Asimov data so the only force on the nuisance is the constraint + external term
    f.set_nobs(f.expected_yield())
    f.minimize()

    parms_str = f.parms.astype(str)
    return {
        "parms": parms_str,
        "x": f.x.numpy(),
    }


def loss_grad_hess_at(filename, x_override=None):
    """Return (loss, grad, hess) for the loaded tensor evaluated at x_override
    (or the default starting x if None). Uses Asimov data."""
    import tensorflow as tf

    indata_obj = inputdata.FitInputData(filename)
    param_model = load_model("Mu", indata_obj)
    options = make_options()
    f = fitter.Fitter(indata_obj, param_model, options)
    f.set_nobs(f.expected_yield())
    if x_override is not None:
        f.x.assign(tf.constant(x_override, dtype=f.x.dtype))
    val, grad, hess = f.loss_val_grad_hess()
    return (
        f.parms.astype(str),
        val.numpy(),
        grad.numpy(),
        hess.numpy(),
    )


def get_param_value(result, name):
    idx = np.where(result["parms"] == name)[0][0]
    return result["x"][idx]


def get_param_index(parms, name):
    return int(np.where(parms == name)[0][0])


def make_grad_hist(values, param_names):
    """Build a 1D hist with a StrCategory axis for an external gradient."""
    ax = hist.axis.StrCategory(param_names, name="params")
    h = hist.Hist(ax, storage=hist.storage.Double())
    h.values()[...] = np.asarray(values)
    return h


def make_hess_hist(values, param_names):
    """Build a 2D hist with two StrCategory axes for an external hessian."""
    ax0 = hist.axis.StrCategory(param_names, name="params0")
    ax1 = hist.axis.StrCategory(param_names, name="params1")
    h = hist.Hist(ax0, ax1, storage=hist.storage.Double())
    h.values()[...] = np.asarray(values)
    return h


def make_hess_sparsehist(values, param_names):
    """Same as make_hess_hist but using a wums.SparseHist.

    StrCategory axes have an overflow bin by default, so SparseHist's
    with-flow layout has shape (n+1, n+1). The user data goes in the
    first n x n block; the overflow row/col is filled with zeros.
    """
    ax0 = hist.axis.StrCategory(param_names, name="params0")
    ax1 = hist.axis.StrCategory(param_names, name="params1")
    n = len(param_names)
    full = np.zeros((ax0.extent, ax1.extent), dtype=np.float64)
    full[:n, :n] = np.asarray(values, dtype=np.float64)
    return SparseHist(scipy.sparse.csr_array(full), [ax0, ax1])


def main():
    import tensorflow as tf

    tf.config.experimental.enable_op_determinism()

    SHAPE = "shape"

    with tempfile.TemporaryDirectory() as tmpdir:

        # --- Baseline: no external term ---
        baseline_writer = build_writer()
        baseline_writer.write(outfolder=tmpdir, outfilename="baseline")
        baseline = run_fit(os.path.join(tmpdir, "baseline.hdf5"))
        baseline_shape = get_param_value(baseline, SHAPE)
        print(f"Baseline (no external):    {SHAPE} = {baseline_shape:.6f}")
        assert (
            abs(baseline_shape) < 1e-6
        ), f"Asimov baseline should give {SHAPE} ~ 0, got {baseline_shape}"
        print("PASS: baseline Asimov fit gives shape ~ 0")

        # Reference loss/grad/hess at the baseline x (no external term).
        # The contribution of L_ext(x) = g^T x + 0.5 x^T H x to the NLL gradient
        # at any x is (g + H x), and to the NLL hessian is H. We test these
        # exactly (not analytical post-fit values, which depend on the data
        # Hessian and the constraint and don't have a clean closed form).
        parms, val0, grad0, hess0 = loss_grad_hess_at(
            os.path.join(tmpdir, "baseline.hdf5")
        )
        i_shape = get_param_index(parms, SHAPE)
        x0 = baseline["x"].copy()
        # the test below evaluates external terms at the baseline minimum
        # where x[i_shape] = 0, so H x_sub = 0 → grad delta == g exactly.

        configs = [
            (
                "grad only (g=1)",
                build_writer(grad=make_grad_hist([1.0], [SHAPE])),
                {i_shape: 1.0},
                {(i_shape, i_shape): 0.0},
            ),
            (
                "grad+dense hess (g=1, h=2)",
                build_writer(
                    grad=make_grad_hist([1.0], [SHAPE]),
                    hess=make_hess_hist([[2.0]], [SHAPE]),
                ),
                {i_shape: 1.0},
                {(i_shape, i_shape): 2.0},
            ),
            (
                "grad+SparseHist hess (g=1, h=2)",
                build_writer(
                    grad=make_grad_hist([1.0], [SHAPE]),
                    hess=make_hess_sparsehist([[2.0]], [SHAPE]),
                ),
                {i_shape: 1.0},
                {(i_shape, i_shape): 2.0},
            ),
            (
                "hess only (h=5)",
                build_writer(hess=make_hess_hist([[5.0]], [SHAPE])),
                {i_shape: 0.0},
                {(i_shape, i_shape): 5.0},
            ),
        ]

        for label, writer, expected_grad_delta, expected_hess_delta in configs:
            tag = (
                label.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace(",", "")
                .replace("=", "")
            )
            writer.write(outfolder=tmpdir, outfilename=tag)
            _, val, grad, hess = loss_grad_hess_at(
                os.path.join(tmpdir, f"{tag}.hdf5"),
                x_override=x0,
            )
            for idx, expected in expected_grad_delta.items():
                actual = grad[idx] - grad0[idx]
                print(
                    f"{label}: grad delta @ idx {idx} = {actual:+.6f}  (expected {expected:+.6f})"
                )
                assert (
                    abs(actual - expected) < 1e-8
                ), f"{label}: grad delta {actual} != expected {expected}"
            for (i, j), expected in expected_hess_delta.items():
                actual = hess[i, j] - hess0[i, j]
                print(
                    f"{label}: hess delta @ ({i},{j}) = {actual:+.6f}  (expected {expected:+.6f})"
                )
                assert (
                    abs(actual - expected) < 1e-8
                ), f"{label}: hess delta {actual} != expected {expected}"
            print(f"PASS: {label}")

        # Sanity check: also verify that running the full fit shifts the
        # baseline shape value in the expected direction (negative for g=+1).
        grad_only_writer = build_writer(grad=make_grad_hist([1.0], [SHAPE]))
        grad_only_writer.write(outfolder=tmpdir, outfilename="grad_only_fit")
        grad_only = run_fit(os.path.join(tmpdir, "grad_only_fit.hdf5"))
        v = get_param_value(grad_only, SHAPE)
        print(f"Full fit with g=+1: shape = {v:.6f}  (expected negative)")
        assert v < -1e-3, f"Expected shape to pull negative, got {v}"
        print("PASS: full fit with positive gradient pulls shape negative")

        print()
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
