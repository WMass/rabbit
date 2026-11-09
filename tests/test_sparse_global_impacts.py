"""Global impacts with bin-by-bin stat on a sparse tensor.

The default (JVP) global impacts differentiate the profiled beta in forward
mode, with tf.autodiff.ForwardAccumulator inside tf.vectorized_map. A sparse
tensor's yields contract logk with tf.linalg.sparse's CSR matmul, and the
accumulator gives every input without a tangent a zeros_like one -- the
CSRSparseMatrix variant included, for which TensorFlow has no CPU ZEROS_LIKE
kernel. So --globalImpacts on a sparse tensor failed whenever bin-by-bin stat
was on (CI's sparse fit runs with --noBinByBinStat):

  InternalError: No unary variant unary_op function found for op ZEROS_LIKE
  Variant type_name: tensorflow::CSRSparseMatrix for device type: CPU

At the same parameters and covariance, the sparse tensor must give the global
impacts of the dense tensor of the same model, on the parameters and on the
observable bins, in forward (default) and backward mode alike.
"""

import numpy as np
import pytest

from rabbit import fitter, inputdata, tensorwriter
from rabbit.mappings import helpers as mh
from rabbit.param_models.helpers import load_model
from tests.test_sparse_fit import make_histograms, make_options

_MODES = {
    "lite": dict(binByBinStatType="automatic", binByBinStatMode="lite"),
    "gamma_full": dict(binByBinStatType="gamma", binByBinStatMode="full"),
    "normal_additive_full": dict(
        binByBinStatType="normal-additive", binByBinStatMode="full"
    ),
}


def _make_tensor(outdir, sparse):
    hists = make_histograms()
    sig = hists["sig"]

    writer = tensorwriter.TensorWriter(sparse=sparse)
    writer.add_channel(hists["data"].axes, "ch0")
    writer.add_data(hists["data"], "ch0")
    writer.add_process(sig, "sig", "ch0", signal=True)
    writer.add_process(hists["bkg"], "bkg", "ch0")
    # beta is profiled on the unmasked bins only (full=False)
    writer.add_channel(sig.axes, "ch0_masked", masked=True)
    writer.add_process(sig, "sig", "ch0_masked", signal=True)

    writer.add_norm_systematic("bkg_norm", "bkg", "ch0", 1.05)
    writer.add_systematic(
        [hists["syst_up"], hists["syst_dn"]],
        "bkg_shape",
        "bkg",
        "ch0",
        symmetrize="average",
    )
    # asymmetric, so logk also carries the interpolation half
    tilt = np.linspace(-0.05, 0.05, sig.values().size)
    sig_up, sig_dn = sig.copy(), sig.copy()
    sig_up.values()[...] = sig.values() * (1 + tilt) ** 2
    sig_dn.values()[...] = sig.values() * (1 - tilt)
    writer.add_systematic(
        [sig_up, sig_dn], "sig_shape", "sig", "ch0", symmetrize=None, noi=True
    )

    name = f"test_global_impacts_{'sparse' if sparse else 'dense'}"
    writer.write(outfolder=outdir, outfilename=name)
    return f"{outdir}/{name}.hdf5"


@pytest.fixture(scope="module")
def tensors(tmp_path_factory):
    outdir = str(tmp_path_factory.mktemp("sparse_global_impacts"))
    return {sparse: _make_tensor(outdir, sparse) for sparse in (False, True)}


def _fitter(fname, mode, jvp=True):
    indata = inputdata.FitInputData(fname)
    options = make_options(earlyStopping=20, **_MODES[mode])
    f = fitter.make_fitter(
        indata, load_model("Mu", indata), options, globalImpactsFromJVP=jvp
    )
    f.set_nobs(indata.data_obs)
    return f


def _global_impacts(f):
    """Global impacts (per source, grouped) on the parameters and on all bins."""
    out = [np.asarray(r) for r in f.global_impacts_parms()]
    mapping = mh.load_mapping("BaseMapping", f.indata)
    _, aux = f.expected_events(mapping, compute_global_impacts=True)
    return out + [np.asarray(aux[2]), np.asarray(aux[3])]


@pytest.mark.parametrize("mode", list(_MODES))
def test_sparse_matches_dense(tensors, mode):
    dense = _fitter(tensors[False], mode)
    assert not dense.indata.sparse
    dense.minimize()
    _, _, hess = dense.loss_val_grad_hess()
    dense.cov.assign(np.linalg.inv(hess.numpy()))
    ref = _global_impacts(dense)

    # grouped impacts are [syst groups | stat | binByBinStat | ...]: the beta
    # part must contribute, or the comparison below would not test it
    ibbb = len(dense.indata.systgroupidxs) + 1
    assert ref[1][:, ibbb].min() > 0
    assert ref[3][:, ibbb].max() > 0

    for jvp in (True, False):
        sparse = _fitter(tensors[True], mode, jvp=jvp)
        assert sparse.indata.sparse
        sparse.x.assign(dense.x)
        sparse.cov.assign(dense.cov)
        for r, s in zip(ref, _global_impacts(sparse)):
            assert np.all(np.isfinite(s))
            np.testing.assert_allclose(
                s, r, rtol=1e-9, atol=1e-12 * np.abs(r).max(), err_msg=f"{jvp=}"
            )
