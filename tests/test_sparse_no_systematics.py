"""A sparse tensor without systematics fits like the dense one.

The sparse likelihood contracts logk [nnz, nsyst] with the nuisances through
TF's CSR matmul. With nsyst = 0 the GPU kernel does not write its output, so
the product was whatever was left in GPU memory; the fitter now returns zeros
for an empty inner dimension. On a CPU-only machine the comparison passes
either way, on a GPU it failed before the guard.
"""

import os
import tempfile

import hist
import numpy as np
import pytest
import tensorflow as tf

from rabbit import fitter, inputdata, tensorwriter
from rabbit.param_models.helpers import load_model
from tests.test_sparse_fit import _to_scipy_sparse, make_options


def _write(path, sparse):
    rng = np.random.default_rng(3)
    ax = hist.axis.Regular(30, 0.0, 1.0, name="x")
    sig = hist.Hist(ax, storage=hist.storage.Weight())
    bkg = hist.Hist(ax, storage=hist.storage.Weight())
    x = ax.centers
    sig.values()[...] = 200.0 * np.exp(-0.5 * ((x - 0.5) / 0.1) ** 2)
    sig.variances()[...] = sig.values()
    bkg.values()[...] = 300.0 * (1.2 - x)
    bkg.variances()[...] = bkg.values()
    data = hist.Hist(ax, storage=hist.storage.Double())
    data.values()[...] = rng.poisson(sig.values() + bkg.values())

    w = tensorwriter.TensorWriter(sparse=sparse)
    w.add_channel(data.axes, "ch0")
    w.add_data(data, "ch0")
    if sparse:
        w.add_process(
            _to_scipy_sparse(sig),
            "sig",
            "ch0",
            signal=True,
            variances=sig.variances(),
        )
        w.add_process(_to_scipy_sparse(bkg), "bkg", "ch0", variances=bkg.variances())
    else:
        w.add_process(sig, "sig", "ch0", signal=True)
        w.add_process(bkg, "bkg", "ch0")
    name = "sparse" if sparse else "dense"
    w.write(outfolder=path, outfilename=name)
    return os.path.join(path, f"{name}.hdf5")


def _fit(fname):
    indata = inputdata.FitInputData(fname)
    f = fitter.Fitter(indata, load_model("Mu", indata), make_options())
    f.set_nobs(indata.data_obs)
    f.minimize()
    return f


def test_logk_csr_matmul_without_systematics_is_zero():
    with tempfile.TemporaryDirectory() as tmp:
        f = _fit(_write(tmp, sparse=True))
        assert f.logk.shape[1] == 0
        out = f._logk_csr_matmul(tf.zeros([0, 3], dtype=f.indata.dtype))
        assert out.shape == (f.logk.shape[0], 3)
        np.testing.assert_array_equal(out.numpy(), 0.0)


@pytest.mark.parametrize("dirty", [False, True], ids=["clean", "dirty-memory"])
def test_sparse_fit_without_systematics_matches_dense(dirty):
    with tempfile.TemporaryDirectory() as tmp:
        dense = _fit(_write(tmp, sparse=False))
        if dirty:
            # leave non-zero values in freed device memory, as a previous
            # computation would
            junk = tf.fill([4096], tf.constant(7.0, dense.indata.dtype)) * 1.0
            del junk
        sparse = _fit(_write(tmp, sparse=True))
        np.testing.assert_allclose(sparse.x.numpy(), dense.x.numpy(), rtol=1e-6)
        np.testing.assert_allclose(
            sparse.reduced_nll().numpy(), dense.reduced_nll().numpy(), rtol=1e-8
        )
