"""AxisExpPolyModel: an exponential-polynomial background per cell.

rnorm = exp(lnAmpl_c + sum_k c_k * P_k(t)), with P_k orthonormalised on the
actual bin grid of the shape axis. Checks the basis, the parameterisation, the
nesting of the orders, and that a fit recovers the parameters it was built from.
"""

import os
import tempfile

import hist
import numpy as np
import pytest
import tensorflow as tf

from rabbit import fitter, inputdata, tensorwriter
from rabbit.param_models.helpers import load_model

from .test_sparse_fit import make_options

NMASS = 20
NETA = 2


def _write_tensor(outdir, nmass=NMASS, data_scale=None):
    axes = [
        hist.axis.Regular(nmass, 5.05, 5.45, name="mass"),
        hist.axis.Regular(NETA, -1.0, 1.0, name="eta"),
    ]
    h_bkg = hist.Hist(*axes, storage=hist.storage.Weight())
    h_bkg.values()[...] = 1000.0
    h_bkg.variances()[...] = 1000.0
    h_data = hist.Hist(*axes, storage=hist.storage.Double())
    h_data.values()[...] = h_bkg.values() * (1.0 if data_scale is None else data_scale)

    writer = tensorwriter.TensorWriter()
    writer.add_channel(h_data.axes, "ch0")
    writer.add_data(h_data, "ch0")
    writer.add_process(h_bkg, "bkg", "ch0", signal=False)
    writer.write(outfolder=outdir, outfilename="axis_exp_poly")
    return inputdata.FitInputData(os.path.join(outdir, "axis_exp_poly.hdf5"))


@pytest.fixture(scope="module")
def indata():
    with tempfile.TemporaryDirectory() as outdir:
        yield _write_tensor(outdir)


def _model(indata, order, *order_axes):
    return load_model(
        "AxisExpPolyModel", indata, "ch0", "bkg", "mass", "eta", str(order), *order_axes
    )


def _rnorm_bkg(model, param):
    """rnorm of the bkg process, reshaped to (mass, eta)."""
    rnorm = model.compute(tf.constant(param, dtype=model.indata.dtype)).numpy()
    return rnorm[:, 0].reshape(NMASS, NETA)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_basis_is_orthonormal_on_the_grid(indata, order):
    model = _model(indata, order)
    basis = np.array([p.numpy() for p in model.p_k])
    # orthogonal to the constant (lnAmpl) and to each other, unit rms
    np.testing.assert_allclose(basis.mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(basis @ basis.T / NMASS, np.eye(order), atol=1e-12)
    assert model.basis_max_overlap < 1e-12


def test_parameter_count_and_names(indata):
    # one lnAmpl and one coefficient per order, all per eta cell
    model = _model(indata, 2)
    assert model.npoi == 3 * NETA
    assert model.npou == 0
    names = [n.decode() for n in model.params]
    assert names == [
        "lnAmpl_bkg_eta0",
        "lnAmpl_bkg_eta1",
        "poly1_bkg_eta0",
        "poly1_bkg_eta1",
        "poly2_bkg_eta0",
        "poly2_bkg_eta1",
    ]


def test_zero_parameters_give_a_flat_unit_background(indata):
    model = _model(indata, 3)
    np.testing.assert_allclose(_rnorm_bkg(model, model.xparamdefault.numpy()), 1.0)


def test_order_one_is_an_exponential_slope(indata):
    model = _model(indata, 1)
    ln_ampl = np.array([0.3, -0.2])
    slope = np.array([0.5, -1.0])
    rnorm = _rnorm_bkg(model, np.concatenate([ln_ampl, slope]))

    p1 = model.p_k[0].numpy()
    expected = np.exp(ln_ampl[None, :] + slope[None, :] * p1[:, None])
    np.testing.assert_allclose(rnorm, expected, rtol=1e-12)


def test_each_order_nests_the_one_below(indata):
    low = _model(indata, 1)
    high = _model(indata, 2)
    p_low = np.array([0.1, 0.2, 0.7, -0.4])
    p_high = np.concatenate([p_low, np.zeros(NETA)])
    np.testing.assert_allclose(_rnorm_bkg(high, p_high), _rnorm_bkg(low, p_low))


def test_degenerate_order_raises():
    with tempfile.TemporaryDirectory() as outdir:
        small = _write_tensor(outdir, nmass=3)
        # 3 bins carry a constant and two shapes: order 3 has nothing left
        load_model("AxisExpPolyModel", small, "ch0", "bkg", "mass", "eta", "2")
        with pytest.raises(ValueError, match="degenerate"):
            load_model("AxisExpPolyModel", small, "ch0", "bkg", "mass", "eta", "3")


def test_bad_order_axes_raise(indata):
    with pytest.raises(ValueError, match="subset of cell_axes"):
        _model(indata, 1, "mass")


def test_fit_recovers_the_generating_parameters():
    truth_model_args = ("ch0", "bkg", "mass", "eta", "2")
    with tempfile.TemporaryDirectory() as outdir:
        flat = _write_tensor(outdir)
        truth = load_model("AxisExpPolyModel", flat, *truth_model_args)
        p_true = np.array([0.2, -0.1, -0.8, -0.5, 0.15, 0.3])
        shape = _rnorm_bkg(truth, p_true)

    with tempfile.TemporaryDirectory() as outdir:
        indata = _write_tensor(outdir, data_scale=shape)
        model = load_model("AxisExpPolyModel", indata, *truth_model_args)
        f = fitter.Fitter(indata, model, make_options(noBinByBinStat=True))
        f.set_nobs(indata.data_obs)
        f.minimize()
        np.testing.assert_allclose(f.x.numpy()[: model.npoi], p_true, atol=1e-6)
