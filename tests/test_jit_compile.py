"""--jitCompile against the sparse (CSR) matmul, which XLA cannot compile.

SparseMatrixMatMul has no XLA kernel. The likelihood calls it for a sparse
tensor and for each external likelihood term whose Hessian is stored sparse,
with a dense tensor too. "auto" used to look at the tensor only, so a dense
tensor with a sparse external Hessian got jit-compiled loss/grad/HVP functions
that failed to compile on their first call. minimize() logs a minimizer
exception as a warning, so the fit stopped at its starting point and the job
wrote that point out as its result.

The forward-mode HVP (--hvpMethod fwdrev) cannot go through the CSR matmul
either, and failed the same way for a sparse external Hessian.
"""

import logging
import os

import hist
import numpy as np
import pytest
import scipy.sparse
from wums.sparse_hist import SparseHist

from rabbit import fitter, inputdata, tensorwriter
from rabbit.param_models.helpers import load_model
from tests.test_sparse_fit import make_histograms, make_options

NAMES = ["bkg_norm", "bkg_shape"]
HESS = np.array([[2.0, 0.4], [0.4, 1.0]])
MEAN = np.array([0.3, -0.2])


def write_tensor(outdir, hess, sparse=False, name="ext"):
    """The test_sparse_fit model (sig + bkg, two nuisances) with a Gaussian
    external term on the nuisances whose Hessian is stored as ``hess``:
    "dense", "sparse", or None for no term."""
    h = make_histograms()
    writer = tensorwriter.TensorWriter(sparse=sparse)
    writer.add_channel(h["data"].axes, "ch0")
    writer.add_data(h["data"], "ch0")
    writer.add_process(h["sig"], "sig", "ch0", signal=True)
    writer.add_process(h["bkg"], "bkg", "ch0")
    writer.add_norm_systematic("bkg_norm", "bkg", "ch0", 1.05)
    writer.add_systematic(
        [h["syst_up"], h["syst_dn"]], "bkg_shape", "bkg", "ch0", symmetrize="average"
    )
    if hess is not None:
        # the writer reads a SparseHist's flat indices without flow bins
        axes = [
            hist.axis.StrCategory(NAMES, name=f"params{i}", overflow=False)
            for i in range(2)
        ]
        if hess == "sparse":
            h_hess = SparseHist(scipy.sparse.csr_array(HESS), axes)
        else:
            h_hess = hist.Hist(*axes, storage=hist.storage.Double())
            h_hess.values()[...] = HESS
        # mean= stores the centering constant for a sparse Hessian too, so the
        # two storages give the same NLL, not only the same minimum
        writer.add_external_likelihood_term(hess=h_hess, mean=MEAN, name=name)
    fname = f"{'sparse' if sparse else 'dense'}_{hess}_{name}"
    writer.write(outfolder=outdir, outfilename=fname)
    return os.path.join(outdir, f"{fname}.hdf5")


def build_fitter(path, ndevices=1, **opts):
    indata = inputdata.FitInputData(path, host_memory=ndevices > 1)
    # earlyStopping as the CLI default; make_options switches it off
    options = make_options(earlyStopping=20, nDevices=ndevices, **opts)
    f = fitter.make_fitter(indata, load_model("Mu", indata), options)
    f.set_nobs(indata.data_obs)
    return f


def fit(f):
    f.minimize()
    _, _, hess = f.loss_val_grad_hess()
    return dict(x=f.x.numpy(), nll=float(f.reduced_nll()), hess=hess.numpy())


def jit_messages(caplog):
    return [
        (r.levelno, r.getMessage())
        for r in caplog.records
        if "--jitCompile" in r.getMessage()
    ]


@pytest.mark.parametrize("mode", ["auto", "on", "off"])
@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("hess", [None, "dense", "sparse"], ids=lambda h: f"{h}H")
def test_resolution(tmp_path, caplog, hess, sparse, mode):
    """jit_compile is on unless switched off or the likelihood uses the CSR
    matmul; "on" warns when it cannot be honoured."""
    with caplog.at_level(logging.INFO, logger="wums"):
        f = build_fitter(write_tensor(str(tmp_path), hess, sparse), jitCompile=mode)
    uses_csr = sparse or hess == "sparse"
    assert f.jit_compile is (mode != "off" and not uses_csr)
    warned = [m for lvl, m in jit_messages(caplog) if lvl >= logging.WARNING]
    assert bool(warned) == (mode == "on" and uses_csr), warned


def test_dropped_mcstat_term_keeps_jit(tmp_path):
    """An MC-stat moment term enters the loss only with --mcStatDebias
    continuousM, so a sparse one that is dropped must not switch jit off."""
    f = build_fitter(write_tensor(str(tmp_path), "sparse", name="mcstat_M"))
    assert not f.mcstat_terms
    assert f.jit_compile is True


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """The fit with the Hessian stored dense, jit-compiled."""
    tmpdir = str(tmp_path_factory.mktemp("jit_compile"))
    f = build_fitter(write_tensor(tmpdir, "dense"))
    assert f.jit_compile is True
    return tmpdir, fit(f)


@pytest.mark.parametrize(
    "mode,ndevices", [("auto", 1), ("on", 1), ("auto", 2)], ids=str
)
def test_sparse_external_hessian_fits_like_dense(reference, caplog, mode, ndevices):
    """Dense tensor, external Hessian stored sparse: the fit must reach the
    minimum of the same model with the Hessian stored dense. Two devices
    cover the sharded fitter, whose global term evaluates the external terms
    in its own jit-compiled functions.

    The tolerance on x is the minimizer's: the trust region stops at the
    rounding floor, which leaves the jit and non-jit fits of one model 1.4e-7
    apart. Before the fix the fit did not move from its start, 0.09 away."""
    tmpdir, ref = reference
    with caplog.at_level(logging.INFO, logger="wums"):
        f = build_fitter(
            write_tensor(tmpdir, "sparse"), ndevices=ndevices, jitCompile=mode
        )
    res = fit(f)
    np.testing.assert_allclose(res["x"], ref["x"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(res["nll"], ref["nll"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(res["hess"], ref["hess"], rtol=1e-6)

    assert f.jit_compile is False
    # the log says which term switched jit off, at a level matching the request
    msgs = jit_messages(caplog)
    assert len(msgs) == 1 and "['ext']" in msgs[0][1], msgs
    assert msgs[0][0] == (logging.WARNING if mode == "on" else logging.INFO)


def test_fwdrev_falls_back_for_sparse_external_hessian(reference, caplog):
    """ForwardAccumulator has no JVP rule for the CSR matmul: --hvpMethod
    fwdrev must fall back to revrev for a sparse external Hessian, as it does
    in sparse mode, rather than raise inside the minimizer. A dense one keeps
    fwdrev."""
    tmpdir, ref = reference
    dense = build_fitter(write_tensor(tmpdir, "dense"), hvpMethod="fwdrev")
    assert dense.loss_val_grad_hessp is dense.loss_val_grad_hessp_fwdrev

    with caplog.at_level(logging.WARNING, logger="wums"):
        f = build_fitter(write_tensor(tmpdir, "sparse"), hvpMethod="fwdrev")
    warned = [r.getMessage() for r in caplog.records if "fwdrev" in r.getMessage()]
    res = fit(f)
    np.testing.assert_allclose(res["x"], ref["x"], rtol=0, atol=1e-6)
    assert f.loss_val_grad_hessp is f.loss_val_grad_hessp_revrev
    assert len(warned) == 1 and "['ext']" in warned[0], warned


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
