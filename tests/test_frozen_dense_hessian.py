"""Frozen parameters under the dense-Hessian minimizers.

A frozen parameter is held by tf.stop_gradient alone: its gradient entry and
its Hessian row and column are exactly zero, but the loss value still depends
on it. trust-exact and tf-trust-exact used to move it anyway -- the singular
Hessian sends the More-Sorensen iteration into its hard case, which steps
along the frozen coordinate, and the step is accepted because the loss goes
down. A likelihood scan then came out flat (2dNLL ~ 0 where the profile is up
to 10) with every point at maxiter; dogleg's Cholesky failed on the same
Hessian and its scan points were not minimised at all. These tests pin down
that the scanned parameter stays exactly at the scan value and that the
profile is trust-krylov's, which never moved it. The closed-form solve of a
quadratic likelihood factorises the same Hessian and failed outright, and
contour_scan's hess_mode="frozen" took its constraint Hessian from cov^-1,
which couples a frozen parameter to the floating ones.
"""

import os
import tempfile

import numpy as np
import pytest

from rabbit import fitter, inputdata, tensorwriter
from rabbit.param_models.helpers import load_model
from rabbit.tfhelpers import edmval_cov

from .test_sparse_fit import make_histograms, make_options, make_test_tensor

SCAN_POINTS = 7
SCAN_RANGE = 2.0
METHODS = ["trust-exact", "tf-trust-exact", "dogleg"]


@pytest.fixture(scope="module")
def tensor():
    with tempfile.TemporaryDirectory() as outdir:
        yield make_test_tensor(outdir)


def _fitted(filename, method, precondition=False):
    indata_obj = inputdata.FitInputData(filename)
    param_model = load_model("Mu", indata_obj)
    opts = dict(minimizerMethod=method, earlyStopping=20)
    if precondition:
        opts.update(precondition=True, preconditionBlocks="none")
    f = fitter.Fitter(indata_obj, param_model, make_options(**opts))
    f.set_nobs(indata_obj.data_obs)
    f.minimize()
    # nll_scan scales its range by sqrt(cov[idx, idx])
    _, grad, hess = f.loss_val_grad_hess()
    _, cov = edmval_cov(grad, hess)
    f.cov.assign(cov)
    return f


@pytest.fixture(scope="module")
def reference(tensor):
    """trust-krylov profiles: its Krylov space never contains a frozen
    coordinate, so its scans are the reference."""
    f = _fitted(tensor, "trust-krylov")
    return {p: f.nll_scan(p, SCAN_RANGE, SCAN_POINTS)[:2] for p in ("sig", "bkg_norm")}


# precondition=True scans bkg_norm, so that the unconstrained POI is
# preconditioned while a constrained nuisance is frozen; frozen parameters
# never enter a preconditioner block, so scanning the POI would leave nothing
# to precondition.
@pytest.mark.parametrize(
    "param,precondition", [("sig", False), ("bkg_norm", False), ("bkg_norm", True)]
)
@pytest.mark.parametrize("method", METHODS)
def test_scan_keeps_the_frozen_parameter(
    tensor, reference, method, param, precondition
):
    f = _fitted(tensor, method, precondition=precondition)
    if precondition:
        # the transform is built per fit leg; check it really has a block here
        f.freeze_params(param)
        assert f._build_preconditioner().enabled
        f.defreeze_params(param)

    scan_vals, dnlls, detail = f.nll_scan(
        param, SCAN_RANGE, SCAN_POINTS, save_detail=True
    )

    idx = int(np.where(f.parms.astype(str) == param)[0][0])
    # exactly where the scan put it, at every point: not close, equal
    np.testing.assert_array_equal(detail["x"][:, idx], scan_vals)

    ref_vals, ref_dnlls = reference[param]
    np.testing.assert_allclose(scan_vals, ref_vals, rtol=0, atol=1e-6)
    np.testing.assert_allclose(dnlls, ref_dnlls, rtol=0, atol=1e-5)
    # a real profile, not the flat curve of a parameter walking back to its
    # best fit (the +-2 sigma points sit at dnll ~ 2)
    assert dnlls[0] > 1.0 and dnlls[-1] > 1.0, dnlls


@pytest.mark.parametrize("method", METHODS)
def test_freeze_parameters_option(tensor, method):
    """--freezeParameters: a fit from a displaced frozen value keeps it."""
    indata_obj = inputdata.FitInputData(tensor)
    param_model = load_model("Mu", indata_obj)
    f = fitter.Fitter(
        indata_obj,
        param_model,
        make_options(
            minimizerMethod=method, earlyStopping=20, freezeParameters=["sig"]
        ),
    )
    f.set_nobs(indata_obj.data_obs)
    idx = int(np.where(f.parms.astype(str) == "sig")[0][0])
    x = f.x.numpy().copy()
    x[idx] = 0.9  # sqrt storage: signal strength 0.81, far from its best fit
    f.x.assign(x)
    f.minimize()
    assert f.x.numpy()[idx] == 0.9
    # and the floating parameters are at their profile minimum
    _, grad = f.loss_val_grad()
    assert np.max(np.abs(grad.numpy()[f.floating_indices])) < 1e-4


def _linear_tensor(outdir):
    """Normal (additive) symmetric systematics: with chisqFit and a linear POI
    the likelihood is quadratic and minimize() solves it in closed form."""
    h = make_histograms()
    writer = tensorwriter.TensorWriter(systematic_type="normal")
    writer.add_channel(h["data"].axes, "ch0")
    writer.add_data(h["data"], "ch0")
    writer.add_process(h["sig"], "sig", "ch0", signal=True)
    writer.add_process(h["bkg"], "bkg", "ch0")
    writer.add_norm_systematic("bkg_norm", "bkg", "ch0", 1.05)
    writer.add_systematic(
        [h["syst_up"], h["syst_dn"]], "bkg_shape", "bkg", "ch0", symmetrize="average"
    )
    writer.write(outfolder=outdir, outfilename="linear")
    return os.path.join(outdir, "linear.hdf5")


@pytest.mark.parametrize("noHessian", [False, True])
def test_linear_solve_keeps_the_frozen_parameter(noHessian):
    """The Cholesky (and the --noHessian CG) solve on the floating block."""
    with tempfile.TemporaryDirectory() as outdir:
        indata_obj = inputdata.FitInputData(_linear_tensor(outdir))
        param_model = load_model("Mu", indata_obj, allowNegativeParam=True)
        f = fitter.Fitter(
            indata_obj,
            param_model,
            make_options(
                chisqFit=True,
                noBinByBinStat=True,
                noHessian=noHessian,
                freezeParameters=["bkg_norm"],
            ),
        )
        assert f.is_linear
        f.set_nobs(indata_obj.data_obs)
        idx = int(np.where(f.parms.astype(str) == "bkg_norm")[0][0])
        x = f.x.numpy().copy()
        x[idx] = 0.5
        f.x.assign(x)
        f.minimize()
        assert f.x.numpy()[idx] == 0.5
        # one exact Newton step: the floating gradient vanishes
        _, grad = f.loss_val_grad()
        assert np.max(np.abs(grad.numpy()[f.floating_indices])) < 1e-6


def _tensor_with_signal_norm(outdir):
    """make_test_tensor plus a 10% signal normalisation, degenerate with the
    POI: freezing it narrows the POI interval about fivefold."""
    h = make_histograms()
    writer = tensorwriter.TensorWriter()
    writer.add_channel(h["data"].axes, "ch0")
    writer.add_data(h["data"], "ch0")
    writer.add_process(h["sig"], "sig", "ch0", signal=True)
    writer.add_process(h["bkg"], "bkg", "ch0")
    writer.add_norm_systematic("bkg_norm", "bkg", "ch0", 1.05)
    writer.add_norm_systematic("sig_norm", "sig", "ch0", 1.10)
    writer.add_systematic(
        [h["syst_up"], h["syst_dn"]], "bkg_shape", "bkg", "ch0", symmetrize="average"
    )
    writer.write(outfolder=outdir, outfilename="signorm")
    return os.path.join(outdir, "signorm.hdf5")


def test_contour_frozen_hessian_mode_keeps_the_frozen_parameter():
    """contour_scan(hess_mode="frozen") (--asymImpactsHess frozen) uses cov^-1
    as the constraint Hessian, also when parameters are frozen (the
    freeze-group asymmetric impacts, --freezeParameters). Its frozen rows and
    columns must be zero, as in the exact Hessian: through the coupling,
    trust-constr moved a frozen sig_norm by about one sigma and returned a
    wrong, sign-flipped interval."""
    with tempfile.TemporaryDirectory() as outdir:
        f = _fitted(_tensor_with_signal_norm(outdir), "trust-krylov")
        nll_min = f.reduced_nll().numpy()
        ifz = int(np.where(f.parms.astype(str) == "sig_norm")[0][0])

        def contour(mode):
            intervals, dx = f.contour_scan("sig", nll_min, q=1, hess_mode=mode)
            # dx is res.x - xval per side, NaN where the side did not converge
            return intervals, dx[:, ifz]

        free_exact, _ = contour("exact")
        free_frozen, _ = contour("frozen")
        f.freeze_params("sig_norm")
        exact, dfz_exact = contour("exact")
        frozen, dfz_frozen = contour("frozen")
        f.defreeze_params("sig_norm")

    np.testing.assert_array_equal(dfz_exact, [0.0, 0.0])
    np.testing.assert_array_equal(dfz_frozen, [0.0, 0.0])
    np.testing.assert_allclose(frozen, exact, rtol=1e-4)
    # nothing frozen: the mode agrees with "exact", as it always did
    np.testing.assert_allclose(free_frozen, free_exact, rtol=1e-4)
    # signed [down, up], and freezing the degenerate normalisation narrows it
    assert exact[0] < 0 < exact[1]
    assert np.all(np.abs(exact) < 0.5 * np.abs(free_exact))
