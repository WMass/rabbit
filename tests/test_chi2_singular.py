"""The postfit chi2 of a mapping whose residual covariance is exactly singular.

Bins that obey an exact linear relation (two projections of one channel, ABCD
regions) give a singular residual covariance. A solve then failed or not
depending on rounding; the chi2 is now taken over the rank of the covariance
(fitter.chi2_pseudo_inverse, more in tests/test_chi2_rank.py). Duplicating a
projection makes the covariance singular for any rounding.
"""

import os
import tempfile

import numpy as np
import pytest
import tensorflow as tf

from rabbit import fitter, inputdata, tensorwriter
from rabbit.mappings import helpers as mh
from rabbit.mappings import mapping as mp
from rabbit.param_models.helpers import load_model
from rabbit.tfhelpers import edmval_cov

from .test_sparse_fit import make_histograms, make_options


def _fitter(outdir):
    """A one-channel dense fit: signal, background with a norm and a shape
    systematic, bin-by-bin stat."""
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
    writer.write(outfolder=outdir, outfilename="chi2_singular")
    indata = inputdata.FitInputData(os.path.join(outdir, "chi2_singular.hdf5"))
    model = load_model("Mu", indata, expectSignal=[("sig", 1.3)])
    f = fitter.Fitter(indata, model, make_options())
    f.set_nobs(indata.data_obs)
    return f


@pytest.fixture(scope="module")
def postfit():
    with tempfile.TemporaryDirectory() as outdir:
        f = _fitter(outdir)
        # the test data are exactly sig / 0.9 + bkg, which the fit reproduces: fluctuate
        nobs = np.random.default_rng(3).poisson(f.indata.data_obs.numpy())
        f.set_nobs(tf.constant(nobs, dtype=f.indata.dtype))
        f.minimize()
        _, grad, hess = f.loss_val_grad_hess()
        _, cov = edmval_cov(grad, hess)
        f.cov.assign(cov)
        yield f


def test_duplicated_projection_counts_once(postfit):
    f = postfit
    single = mh.load_mapping("Project", f.indata, "ch0", "x")
    twice = mp.CompositeMapping(
        [single, mh.load_mapping("Project", f.indata, "ch0", "x")]
    )

    # the postfit residuals of all bins lose the direction the free signal
    # strength absorbs, and the duplicated covariance [[C, C], [C, C]] has the
    # rank of C
    ndf = 20 - f.nfreeparms
    _, cov = f.residuals(twice.compute_flat, twice._get_data, profile=True)
    lam = np.linalg.eigvalsh(cov)
    assert np.count_nonzero(lam > 1e-10 * lam[-1]) == ndf

    _, (*_, chi2_single, ndf_single) = f.expected_events(single, compute_chi2=True)
    _, (*_, chi2_twice, ndf_twice) = f.expected_events(twice, compute_chi2=True)

    # [r, r]^T pinv([[C, C], [C, C]]) [r, r] = r^T C^+ r
    np.testing.assert_allclose(chi2_twice, chi2_single, rtol=1e-8)
    assert int(ndf_single) == ndf == 19
    assert int(ndf_twice) == ndf  # the rank: the duplicate adds no information


def test_a_singular_but_not_a_non_finite_covariance():
    """An exactly singular covariance is handled, a NaN one refused."""
    res = np.ones(2)
    chi2, ndf = fitter.chi2_pseudo_inverse(res, np.ones([2, 2]))
    np.testing.assert_allclose(chi2, 1.0)  # [1, 1] pinv([[1, 1], [1, 1]]) [1, 1]
    assert ndf == 1
    nan = np.array([[1.0, np.nan], [np.nan, 1.0]])
    with pytest.raises(ValueError, match="covariance is not finite"):
        fitter.chi2_pseudo_inverse(res, nan)
