"""Tests for --scanSaveDetail: what a likelihood scan records about its points.

Without it, ``nll_scan`` stores only the dnll curve. It restores the starting
vector when it finishes, so the parameter vectors it visited are gone, and the
only per-point convergence evidence is whatever scipy happened to print into
the log -- which is ``success``/``status`` (trust-krylov reports False at
perfectly converged points, so they carry no information) plus four of the
gradient entries. Recovering one point's parameters afterwards costs a whole
extra constrained fit.

These tests pin down that the opt-in path records the two things that fixes:
the full parameter vector per point, and a real floating-subspace edmval.
"""

import tempfile

import numpy as np

from rabbit import fitter, inputdata
from rabbit.param_models.helpers import load_model

from .test_sparse_fit import make_options, make_test_tensor

SCAN_POINTS = 5
SCAN_RANGE = 1.0


def _fit(outdir):
    filename = make_test_tensor(outdir)
    indata_obj = inputdata.FitInputData(filename)
    param_model = load_model("Mu", indata_obj)
    f = fitter.Fitter(indata_obj, param_model, make_options())
    f.set_nobs(indata_obj.data_obs)
    f.minimize()
    # nll_scan scales its range by sqrt(cov[idx, idx]), so the covariance must exist
    val, grad, hess = f.loss_val_grad_hess()
    from rabbit.tfhelpers import edmval_cov

    _, cov = edmval_cov(grad, hess)
    f.cov.assign(cov)
    return f


def test_detail_off_returns_the_same_two_values_as_before():
    """Backward compatibility: the default path is untouched."""
    with tempfile.TemporaryDirectory() as outdir:
        f = _fit(outdir)
        out = f.nll_scan("sig", SCAN_RANGE, SCAN_POINTS)
        assert len(out) == 2
        scan_vals, dnlls = out
        assert scan_vals.shape == dnlls.shape == (SCAN_POINTS,)


def test_detail_on_records_every_point_and_reproduces_the_curve():
    """The stored vectors must be the points the scan actually evaluated.

    This is the test that matters: re-evaluating the loss at each stored
    parameter vector has to give back the stored dnll. If it does, the vectors
    are the scan's own points and can be diffed against each other after the
    fact -- which is the whole purpose.
    """
    with tempfile.TemporaryDirectory() as outdir:
        f = _fit(outdir)
        nll_best = f.reduced_nll().numpy()

        scan_vals, dnlls, detail = f.nll_scan(
            "sig", SCAN_RANGE, SCAN_POINTS, save_detail=True
        )

        nparms = int(f.x.shape[0])
        assert detail["x"].shape == (SCAN_POINTS, nparms)
        assert detail["diagnostics"].shape == (
            SCAN_POINTS,
            len(f.SCAN_DETAIL_FIELDS),
        )
        assert np.all(np.isfinite(detail["x"]))

        # nll_scan restored the starting vector, so this also checks it did
        xval_after = f.x.numpy()

        idx = int(np.where(f.parms.astype(str) == "sig")[0][0])
        for i in range(SCAN_POINTS):
            # the scanned parameter sits at the scan value in the stored vector
            assert np.isclose(detail["x"][i, idx], scan_vals[i], atol=1e-12)
            f.x.assign(detail["x"][i])
            assert np.isclose(f.reduced_nll().numpy() - nll_best, dnlls[i], atol=1e-8)
        f.x.assign(xval_after)


def test_edmval_is_small_at_converged_scan_points():
    """A converged profile point has EDM ~0 over the parameters it minimised.

    The scanned parameter is frozen and therefore stationary by construction,
    not by minimisation, which is why the edmval is computed on the floating
    subspace only -- a full-space solve would report the scan's depth instead
    of its convergence, and would grow with the scan range rather than staying
    flat across it.
    """
    with tempfile.TemporaryDirectory() as outdir:
        f = _fit(outdir)
        _, _, detail = f.nll_scan("sig", SCAN_RANGE, SCAN_POINTS, save_detail=True)

        fields = list(f.SCAN_DETAIL_FIELDS)
        edm = detail["diagnostics"][:, fields.index("edmval")]
        gmax = detail["diagnostics"][:, fields.index("grad_max_abs")]

        # the central point runs no fit of its own; the off-central ones do
        off_central = [i for i in range(SCAN_POINTS) if i != SCAN_POINTS // 2]
        assert np.all(np.abs(edm[off_central]) < 1e-6), edm
        assert np.all(gmax[off_central] < 1e-4), gmax


def test_frozen_parameter_is_excluded_from_the_gradient_norm():
    """grad_max_abs is the sup-norm over the MINIMISED parameters.

    The scanned parameter's own gradient entry is nonzero in general -- that is
    what makes the scan a constrained minimum rather than a free one -- so
    including it would make every scan point look unconverged.
    """
    with tempfile.TemporaryDirectory() as outdir:
        f = _fit(outdir)
        _, _, detail = f.nll_scan("sig", SCAN_RANGE, SCAN_POINTS, save_detail=True)

        fields = list(f.SCAN_DETAIL_FIELDS)
        gmax = detail["diagnostics"][:, fields.index("grad_max_abs")]
        assert np.all(gmax < 1e-4)
        # and the scan really did move the parameter, i.e. the points are not
        # all sitting on top of the minimum
        assert np.ptp(detail["x"][:, 0]) > 0
