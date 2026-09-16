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
from types import SimpleNamespace

import hist
import numpy as np
import pytest
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
        blindingGroup=[],
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


SHAPE = "shape"

# L_ext(x) = g^T x + 0.5 x^T H x contributes (g + H x) to the NLL gradient and
# H to its hessian. Evaluated at the baseline minimum, where x[shape] == 0, so
# H x is zero there and the gradient delta is exactly g. Asserting that rather
# than post-fit values, which depend on the data hessian and the constraint and
# have no clean closed form.
_TERMS = [
    ("grad only (g=1)", dict(grad=[1.0], hess=None), 1.0, 0.0),
    ("grad+dense hess (g=1, h=2)", dict(grad=[1.0], hess=[[2.0]]), 1.0, 2.0),
    (
        "grad+SparseHist hess (g=1, h=2)",
        dict(grad=[1.0], hess=[[2.0]], sparse_hess=True),
        1.0,
        2.0,
    ),
    ("hess only (h=5)", dict(grad=None, hess=[[5.0]]), 0.0, 5.0),
]


def _build(grad, hess, sparse_hess=False):
    make_hess = make_hess_sparsehist if sparse_hess else make_hess_hist
    return build_writer(
        grad=None if grad is None else make_grad_hist(grad, [SHAPE]),
        hess=None if hess is None else make_hess(hess, [SHAPE]),
    )


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    """The no-external-term fit, and its loss/grad/hess, built once."""
    import tensorflow as tf

    tf.config.experimental.enable_op_determinism()

    tmpdir = str(tmp_path_factory.mktemp("external_term"))
    build_writer().write(outfolder=tmpdir, outfilename="baseline")
    path = os.path.join(tmpdir, "baseline.hdf5")
    res = run_fit(path)
    parms, _, grad0, hess0 = loss_grad_hess_at(path)
    return SimpleNamespace(
        tmpdir=tmpdir,
        res=res,
        parms=parms,
        grad0=grad0,
        hess0=hess0,
        i_shape=get_param_index(parms, SHAPE),
        x0=res["x"].copy(),
    )


def test_asimov_baseline_sits_at_zero(baseline):
    """Without an external term the Asimov fit must not pull the parameter."""
    value = get_param_value(baseline.res, SHAPE)
    assert abs(value) < 1e-6, f"Asimov baseline should give {SHAPE} ~ 0, got {value}"


@pytest.mark.parametrize(
    "label,kwargs,expected_grad,expected_hess",
    _TERMS,
    ids=[t[0] for t in _TERMS],
)
def test_external_term_enters_grad_and_hess(
    baseline, label, kwargs, expected_grad, expected_hess
):
    tag = "".join(c if c.isalnum() else "_" for c in label)
    _build(**kwargs).write(outfolder=baseline.tmpdir, outfilename=tag)
    _, _, grad, hess = loss_grad_hess_at(
        os.path.join(baseline.tmpdir, f"{tag}.hdf5"), x_override=baseline.x0
    )
    i = baseline.i_shape
    assert (
        abs((grad[i] - baseline.grad0[i]) - expected_grad) < 1e-8
    ), f"{label}: grad delta {grad[i] - baseline.grad0[i]} != {expected_grad}"
    assert (
        abs((hess[i, i] - baseline.hess0[i, i]) - expected_hess) < 1e-8
    ), f"{label}: hess delta {hess[i, i] - baseline.hess0[i, i]} != {expected_hess}"


def test_positive_gradient_pulls_the_fit_negative(baseline):
    """End to end, not just the derivatives: a g = +1 term must move the minimum."""
    _build(grad=[1.0], hess=None).write(
        outfolder=baseline.tmpdir, outfilename="grad_only_fit"
    )
    value = get_param_value(
        run_fit(os.path.join(baseline.tmpdir, "grad_only_fit.hdf5")), SHAPE
    )
    assert value < -1e-3, f"expected {SHAPE} to pull negative, got {value}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
