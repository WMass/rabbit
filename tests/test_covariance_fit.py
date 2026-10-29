"""--covarianceFit on an input tensor without a data covariance.

--covarianceFit fits the correlations of the data, so without a covariance in
the tensor the Fitter refuses it, and points to --chisqFit (uncorrelated data
variances) or to a covariance missing from the tensor.
"""

import hist
import numpy as np
import pytest

from rabbit.tensorwriter import TensorWriter
from tests.mcstat_helpers import _hist1d, _write, fit_toy

NBINS = 10
SIG = np.linspace(100.0, 300.0, NBINS)
BKG = np.full(NBINS, 200.0)
DATA = np.random.default_rng(1).poisson(1.1 * SIG + BKG).astype(float)


def _tensor(tmp_path, name, data, variances=None, cov=None):
    ax = hist.axis.Regular(NBINS, 0.0, 1.0, name="x")
    tw = TensorWriter()
    tw.add_channel([ax], "ch0")
    tw.add_data(_hist1d(ax, data, variances), "ch0")
    if cov is not None:
        tw.add_data_covariance(cov)
    tw.add_process(_hist1d(ax, SIG), "sig", "ch0", signal=True)
    tw.add_process(_hist1d(ax, BKG), "bkg", "ch0")
    tw.add_norm_systematic("bkg_norm", "bkg", "ch0", 1.05)
    return _write(tw, tmp_path, name)


@pytest.mark.parametrize(
    "variances",
    [None, DATA * np.linspace(1.5, 3.0, NBINS)],
    ids=["poisson", "weighted"],
)
def test_covariance_fit_without_data_covariance_is_refused(variances, tmp_path):
    fn = _tensor(tmp_path, "nocov.hdf5", DATA, variances)
    with pytest.raises(RuntimeError, match="--chisqFit") as err:
        fit_toy(fn, tmp_path, "--covarianceFit")
    assert "needs a data covariance" in str(err.value)


def test_a_diagonal_covariance_fits_as_the_chisq_fit(tmp_path):
    """The alternative the refusal points to: with uncorrelated data, the
    covariance fit and the chi-square fit are the same fit."""
    cov = np.diag(DATA)
    f = fit_toy(
        _tensor(tmp_path, "cov.hdf5", DATA, cov=cov),
        tmp_path,
        "--covarianceFit",
        "--noBinByBinStat",
    )
    np.testing.assert_allclose(f.data_cov_inv, np.linalg.inv(cov), rtol=1e-12)
    ref = fit_toy(
        _tensor(tmp_path, "nocov.hdf5", DATA),
        tmp_path,
        "--chisqFit",
        "--noBinByBinStat",
    )
    np.testing.assert_allclose(f.x, ref.x, rtol=1e-8, atol=1e-10)
