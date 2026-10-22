"""Tests for rabbit.batched_profile and --scanBatchSize (batched likelihood scans).

A batched scan profiles all points of a scan together, with its own vectorized NLL
and a batched Newton minimizer, instead of one sequential scipy fit per point. These
tests pin down that it is the same likelihood (its NLL equals Fitter.reduced_nll at
arbitrary parameter vectors, for every supported tensor layout), that it finds the
same profile (nll_scan and nll_scan2D give the sequential curves), that it leaves the
fitter as it found it, and that a fit it does not support falls back to the
sequential scan.
"""

import os

import hist
import numpy as np
import pytest
import tensorflow as tf

from rabbit import fitter, inputdata, tensorwriter
from rabbit.batched_profile import BatchedProfile
from rabbit.param_models.helpers import load_model
from rabbit.tfhelpers import edmval_cov

from .test_sparse_fit import make_options

SCAN_RANGE = 2.0


def make_tensor(outdir, sparse, symmetric):
    """Two channels, two signals and a background with a normalization and a shape
    systematic (asymmetric unless ``symmetric``), MC statistical uncertainties."""
    rng = np.random.default_rng(42)
    writer = tensorwriter.TensorWriter(sparse=sparse)
    for ich, channel in enumerate(["ch0", "ch1"]):
        ax = hist.axis.Regular(12, -3, 3, name="x")
        centers = ax.centers
        sig_a = 200 * np.exp(-0.5 * centers**2) * (1 + 0.5 * ich)
        sig_b = 80 * np.exp(-0.5 * (centers - 1) ** 2)
        bkg = 300 * np.exp(-0.2 * (centers + 3)) + 20
        h = {}
        for name, values in (("sig_a", sig_a), ("sig_b", sig_b), ("bkg", bkg)):
            h[name] = hist.Hist(ax, storage=hist.storage.Weight())
            h[name].values()[...] = values
            h[name].variances()[...] = values * rng.uniform(0.02, 0.2, len(values))
        data = hist.Hist(ax, storage=hist.storage.Double())
        data.values()[...] = rng.poisson(sig_a + sig_b + bkg)
        writer.add_channel(data.axes, channel)
        writer.add_data(data, channel)
        writer.add_process(h["sig_a"], "sig_a", channel, signal=True)
        writer.add_process(h["sig_b"], "sig_b", channel, signal=True)
        writer.add_process(h["bkg"], "bkg", channel)
        writer.add_norm_systematic("bkg_norm", "bkg", channel, 1.05)
        tilt = 0.04 * centers
        up, down = h["bkg"].copy(), h["bkg"].copy()
        up.values()[...] = bkg * (1 + tilt)
        down.values()[...] = bkg * (1 - 0.5 * tilt)  # asymmetric
        writer.add_systematic(
            [up, down],
            "bkg_shape",
            "bkg",
            channel,
            symmetrize="average" if symmetric else None,
        )
    name = f"batched_{'sparse' if sparse else 'dense'}_{int(symmetric)}"
    writer.write(outfolder=outdir, outfilename=name)
    return os.path.join(outdir, f"{name}.hdf5")


def make_fitter(
    outdir, sparse=False, symmetric=True, allow_negative=False, **option_overrides
):
    """A fitter at its minimum on the tensor's data, with its Hessian covariance."""
    indata_obj = inputdata.FitInputData(make_tensor(outdir, sparse, symmetric))
    param_model = load_model("Mu", indata_obj, allowNegativeParam=allow_negative)
    f = fitter.Fitter(indata_obj, param_model, make_options(**option_overrides))
    f.set_nobs(indata_obj.data_obs)
    f.minimize()
    _, grad, hess = f.loss_val_grad_hess()
    f.cov.assign(edmval_cov(grad, hess)[1])
    return f


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("bbstat", [True, False])
def test_batched_nll_is_reduced_nll(tmp_path, sparse, symmetric, bbstat):
    """One row per parameter vector, each equal to Fitter.reduced_nll there."""
    f = make_fitter(tmp_path, sparse, symmetric, noBinByBinStat=not bbstat)
    bp = BatchedProfile(f)
    xbest = f.x.numpy()
    X = xbest + 0.3 * np.random.default_rng(1).standard_normal((4, len(xbest)))
    X[0] = xbest
    batched = bp.nll(tf.constant(X, dtype=f.indata.dtype)).numpy()
    for row, value in zip(X, batched):
        f.x.assign(row)
        assert np.isclose(value, f.reduced_nll().numpy(), rtol=0, atol=1e-9)
    f.x.assign(xbest)


@pytest.mark.parametrize(
    "sparse,symmetric,allow_negative",
    [(False, True, False), (True, False, True)],
)
def test_batched_scan_matches_sequential(tmp_path, sparse, symmetric, allow_negative):
    """Same points, same profiled curve; x and the frozen set are restored."""
    f = make_fitter(tmp_path, sparse, symmetric, allow_negative)
    xbest = f.x.numpy().copy()
    x_seq, dnll_seq, _ = f.nll_scan("sig_a", SCAN_RANGE, 9)
    x_bat, dnll_bat, _ = f.nll_scan("sig_a", SCAN_RANGE, 9, batch_size=4)
    np.testing.assert_allclose(x_bat, x_seq, rtol=0, atol=1e-12)
    np.testing.assert_allclose(dnll_bat, dnll_seq, rtol=0, atol=1e-6)
    assert dnll_bat.max() > 1.0  # the scan left the minimum
    np.testing.assert_array_equal(f.x.numpy(), xbest)
    assert not np.any(f.frozen_params_mask.numpy())


def test_batched_scan2D_matches_sequential(tmp_path):
    f = make_fitter(tmp_path, sparse=True, symmetric=False)
    x_seq, y_seq, dnll_seq = f.nll_scan2D(("sig_a", "bkg_shape"), SCAN_RANGE, 5)
    x_bat, y_bat, dnll_bat = f.nll_scan2D(
        ("sig_a", "bkg_shape"), SCAN_RANGE, 5, batch_size=8
    )
    np.testing.assert_allclose(np.asarray(x_bat), np.asarray(x_seq), atol=1e-12)
    np.testing.assert_allclose(np.asarray(y_bat), np.asarray(y_seq), atol=1e-12)
    assert np.all(np.isfinite(dnll_bat))
    np.testing.assert_allclose(dnll_bat, dnll_seq, rtol=0, atol=1e-6)
    assert not np.any(f.frozen_params_mask.numpy())


def test_batched_scan_detail_reproduces_the_curve(tmp_path):
    """--scanSaveDetail with a batched scan: the stored vectors are the profiled
    points, and they converged."""
    f = make_fitter(tmp_path)
    nll_best = f.reduced_nll().numpy()
    x_scan, dnll, detail = f.nll_scan(
        "sig_a", SCAN_RANGE, 7, save_detail=True, batch_size=8
    )
    fields = list(f.SCAN_DETAIL_FIELDS)
    off_central = np.arange(7) != 3
    diagnostics = detail["diagnostics"][off_central]
    assert np.all(diagnostics[:, fields.index("success")] == 1.0)
    assert np.all(diagnostics[:, fields.index("edmval")] < 1e-6)
    xbest = f.x.numpy()
    idx = int(np.where(f.parms.astype(str) == "sig_a")[0][0])
    for i in range(7):
        assert np.isclose(detail["x"][i, idx], x_scan[i], atol=1e-12)
        f.x.assign(detail["x"][i])
        assert np.isclose(f.reduced_nll().numpy() - nll_best, dnll[i], atol=1e-8)
    f.x.assign(xbest)


def test_fixed_points_take_physical_poi_values(tmp_path):
    """A POI stored as a square root is fixed at the square root of its value."""
    f = make_fitter(tmp_path, allow_negative=False)
    bp = BatchedProfile(f)
    idx = int(np.where(f.parms.astype(str) == "sig_a")[0][0])
    physical = bp.fixed_points(["sig_a"], [[0.81], [1.21]])
    stored = bp.profile([idx], [[0.9], [1.1]])
    np.testing.assert_allclose(physical.x[:, idx], [0.9, 1.1], atol=1e-12)
    np.testing.assert_allclose(physical.dnll, stored.dnll, atol=1e-9)
    with pytest.raises(ValueError):
        bp.fixed_points(["sig_a"], [[-0.5]])


def test_saddle_at_square_root_poi_zero_is_left(tmp_path):
    """A POI stored as a square root has an identically vanishing gradient on the
    whole hyperplane where it is 0, so a start there is a saddle point (the fit
    wants the POI positive) that Newton steps alone never leave: the profile must
    still reach the minimum."""
    f = make_fitter(tmp_path, allow_negative=False)
    names = f.parms.astype(str)
    ia, ib = (int(np.where(names == p)[0][0]) for p in ("sig_a", "sig_b"))
    xbest = f.x.numpy()
    start = xbest.copy()
    start[ib] = 0.0
    res = BatchedProfile(f).profile([ia], [[xbest[ia]]], x_start=start[None, :])
    assert res.converged[0]
    assert abs(res.dnll[0]) < 1e-6
    assert np.isclose(abs(res.x[0, ib]), abs(xbest[ib]), atol=1e-3)


def test_undefined_points_are_given_up_at_once(tmp_path):
    """A point with a negative expected yield has no likelihood: it is reported
    as not finite and not converged after one evaluation, and does not hold up
    the other points of its batch."""
    f = make_fitter(tmp_path, allow_negative=True)
    idx = int(np.where(f.parms.astype(str) == "sig_a")[0][0])
    res = BatchedProfile(f).profile([idx], [[-50.0], [1.2]], batch_size=2)
    assert not np.isfinite(res.dnll[0]) and not res.converged[0]
    assert res.iterations[0] == 1
    assert np.isfinite(res.dnll[1]) and res.converged[1]


def test_unsupported_fit_falls_back_to_sequential(tmp_path):
    f = make_fitter(tmp_path, binByBinStatMode="full")
    with pytest.raises(NotImplementedError):
        BatchedProfile(f)
    assert f.batched_profile() is None
    x_seq, dnll_seq, _ = f.nll_scan("sig_a", SCAN_RANGE, 5)
    x_bat, dnll_bat, _ = f.nll_scan("sig_a", SCAN_RANGE, 5, batch_size=4)
    np.testing.assert_array_equal(x_bat, x_seq)
    np.testing.assert_allclose(dnll_bat, dnll_seq, rtol=0, atol=1e-9)
