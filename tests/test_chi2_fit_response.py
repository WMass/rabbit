"""The postfit chi2 of a mapping from the fit's response to its inputs.

The residual covariance is B (1 - P) B^T in the whitened inputs (observations,
constraint centres, bin-by-bin stat centres), with P the projector on the scores
of the fitted parameters (Fitter._fit_response_covariance). It nulls exactly the
directions an unconstrained parameter absorbs, which a propagation through the
fit's covariance only nearly does, so the ndf of the chi2 of a mapping of all
bins is the saturated one, nbins - nfreeparms, and its value agrees with the
saturated 2 dNLL.
"""

import logging
import os
import subprocess
import sys
import tempfile

import hist
import numpy as np
import pytest
import tensorflow as tf

from rabbit import fitter, inputdata, tensorwriter
from rabbit.fitter import chi2_pseudo_inverse
from rabbit.mappings import helpers as mh
from rabbit.param_models.helpers import load_models
from rabbit.tfhelpers import edmval_cov
from tests.test_chi2_rank import make_abcd_tensor
from tests.test_external_term import make_hess_hist
from tests.test_sparse_fit import make_options, make_test_tensor

TESTS = os.path.dirname(os.path.abspath(__file__))

ABCD = ["nonprompt", "ch_A", "ch_B", "ch_C", "ch_D"]
EXTENDED_ABCD = ["nonprompt", "ch_Ax", "ch_Bx", "ch_A", "ch_B", "ch_C", "ch_D"]
# the CI fits (.github/workflows/main.yml) and a 2000-bin ABCD fit
CI_FITS = {
    "abcd": ("abcd", [["Mu"], ["ABCD", *ABCD]]),
    "smooth_abcd": ("abcd", [["Mu"], ["SmoothABCD", "pt", *ABCD]]),
    "extended_abcd": ("extended_abcd", [["Mu"], ["ExtendedABCD", *EXTENDED_ABCD]]),
    "smooth_extended_abcd": (
        "extended_abcd",
        [["Mu"], ["SmoothExtendedABCD", "pt", *EXTENDED_ABCD]],
    ),
    "nominal": ("nominal", [["Mu"]]),
    "abcd_2000_bins": ("abcd2000", [["Mu"], ["ABCD", *ABCD]]),
}


@pytest.fixture(scope="module")
def tensors():
    with tempfile.TemporaryDirectory() as d:
        for script in ["make_tensor.py", "make_abcd_tensor.py"]:
            subprocess.run(
                [sys.executable, os.path.join(TESTS, script), "-o", d], check=True
            )
        subprocess.run(
            [sys.executable, os.path.join(TESTS, "make_extended_abcd_tensor.py")]
            + ["-o", d],
            check=True,
        )
        yield {
            "nominal": os.path.join(d, "test_tensor.hdf5"),
            "abcd": os.path.join(d, "test_abcd_tensor", "rabbit_input.hdf5"),
            "extended_abcd": os.path.join(
                d, "test_extended_abcd_tensor", "rabbit_input.hdf5"
            ),
            "abcd2000": make_abcd_tensor(d, n_pt=500),
        }


def fit(path, models=(("Mu",),), nobs=None, **options):
    indata = inputdata.FitInputData(path)
    model = load_models([list(m) for m in models], indata)
    f = fitter.Fitter(indata, model, make_options(**options))
    f.defaultassign()
    f.set_nobs(indata.data_obs if nobs is None else nobs)
    f.minimize()
    if f.bbstat.enabled:
        f._profile_beta()
    return f


def with_covariance(f):
    """The parameter covariance, for the covariance propagated through it."""
    _, grad, hess = f.loss_val_grad_hess()
    _, cov = edmval_cov(grad, hess)
    f.cov.assign(cov)
    return f


def postfit_chi2(f, mapping=("BaseMapping",)):
    m = mh.load_mapping(mapping[0], f.indata, *mapping[1:])
    _, aux = f.expected_events(
        m, compute_variance=False, compute_chi2=True, profile=True
    )
    return aux[-2:]


def propagated_chi2(f, mapping=("BaseMapping",)):
    m = mh.load_mapping(mapping[0], f.indata, *mapping[1:])
    res, cov = f._residual_terms(m.compute_flat, m._get_data, True)
    return chi2_pseudo_inverse(res.numpy(), cov.numpy())


def saturated(f):
    """2 dNLL and ndf of the saturated test, as bin/rabbit_fit.py has them."""
    return 2 * float(f.reduced_nll()), int(f.nobs.shape[0]) - f.nfreeparms


@pytest.mark.parametrize("case", list(CI_FITS))
def test_the_postfit_ndf_is_the_saturated_one(tensors, case):
    """The propagated covariance's rank was 14 instead of 4 for the CI ABCD fit
    (20 bins, 16 free parameters), 60 instead of 57 for the nominal one."""
    tensor, models = CI_FITS[case]
    f = fit(tensors[tensor], models)
    chi2, ndf = postfit_chi2(f)
    dnll2, ndfsat = saturated(f)
    assert ndf == ndfsat
    np.testing.assert_allclose(chi2, dnll2, rtol=1e-2)


VARIANTS = {
    "no_bin_by_bin_stat": dict(noBinByBinStat=True),
    "gamma": dict(),
    "normal_multiplicative": dict(binByBinStatType="normal-multiplicative"),
    "normal_additive": dict(binByBinStatType="normal-additive"),
    "full": dict(binByBinStatMode="full"),
    "full_gamma": dict(binByBinStatMode="full", binByBinStatType="gamma"),
    "full_normal_additive": dict(
        binByBinStatMode="full", binByBinStatType="normal-additive"
    ),
    "dynamic": dict(binByBinStatDynamic=True),
    "chisq_gamma": dict(chisqFit=True),
    "chisq_normal_additive": dict(chisqFit=True, binByBinStatType="normal-additive"),
}


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_bin_by_bin_stat_modes_and_types(tensors, variant):
    """Profiling the bin-by-bin stat parameters makes the coefficients of the
    other scores depend on the data, except for multiplicative ones with a
    Poisson likelihood or additive ones with a Gaussian one; held at their
    profiled values and projected out per bin, they leave the free directions
    null in every mode. The closed form is the dense projection."""
    f = fit(tensors["abcd"], CI_FITS["abcd"][1], **VARIANTS[variant])
    chi2, ndf = postfit_chi2(f)
    dnll2, ndfsat = saturated(f)
    assert ndf == ndfsat
    np.testing.assert_allclose(chi2, dnll2, rtol=1e-2)
    if f.bbstat.enabled:
        fun = mh.load_mapping("BaseMapping", f.indata).compute_flat
        _, cov = f._fit_response_covariance(f._fit_response_terms(fun))
        _, cov_dense = f._fit_response_covariance(
            f._fit_response_terms(fun, dense_beta=True), dense_beta=True
        )
        np.testing.assert_allclose(cov, cov_dense, rtol=0, atol=1e-10 * cov.max())


@pytest.mark.parametrize("bin_by_bin_stat", [True, False])
def test_data_covariance(tensors, bin_by_bin_stat):
    """With a data covariance the bin-by-bin stat parameters see all bins: the
    dense projection. The likelihood is Gaussian, and so the chi2 is 2 dNLL."""
    f = fit(
        tensors["nominal"],
        covarianceFit=True,
        noBinByBinStat=not bin_by_bin_stat,
    )
    chi2, ndf = postfit_chi2(f)
    dnll2, ndfsat = saturated(f)
    assert ndf == ndfsat
    np.testing.assert_allclose(chi2, dnll2, rtol=1e-4)


@pytest.fixture(scope="module")
def nominal_fit(tensors):
    return with_covariance(fit(tensors["nominal"]))


@pytest.mark.parametrize(
    "mapping",
    [
        ("Select", "ch0", "x:rebin(-5,-2,2,5)"),
        ("Project", "ch1", "a"),
        ("Project", "ch1", "b"),
        ("Normalize", "ch0", "x"),
        ("Ratio", "ch0", "ch1", "None:None", "b:slice(0,2),b:sum"),
    ],
)
def test_mappings_without_free_directions_barely_change(nominal_fit, mapping):
    """The two covariances agree in the linear limit; for the CI nominal fit the
    chi2 of the mappings that do not resolve a free direction moves by < 1%."""
    f = nominal_fit
    chi2, ndf = postfit_chi2(f, mapping)
    chi2_prop, ndf_prop = propagated_chi2(f, mapping)
    assert ndf == ndf_prop
    np.testing.assert_allclose(chi2, chi2_prop, rtol=1e-2)


def write_tensor(path, bkg_free=True, external_sig_prior=False):
    """One channel of 20 bins, signal and background, no bin-by-bin stat
    variances; the background normalization unconstrained."""
    rng = np.random.default_rng(11)
    ax = hist.axis.Regular(20, -5, 5, name="x")
    h_sig = hist.Hist(ax, storage=hist.storage.Double())
    h_bkg = hist.Hist(ax, storage=hist.storage.Double())
    h_sig.fill(rng.normal(0, 1, 4000))
    h_bkg.fill(rng.uniform(-5, 5, 2000))
    h_data = hist.Hist(ax, storage=hist.storage.Double())
    h_data.values()[...] = rng.poisson(h_sig.values() + h_bkg.values())
    w = tensorwriter.TensorWriter()
    w.add_channel([ax], "ch0")
    w.add_data(h_data, "ch0")
    w.add_process(h_sig, "sig", "ch0", signal=True)
    w.add_process(h_bkg, "bkg", "ch0")
    if bkg_free:
        w.add_norm_systematic("bkg_norm", "bkg", "ch0", 1.1, constrained=False)
    if external_sig_prior:
        w.add_external_likelihood_term(
            hess=make_hess_hist([[1.0 / 0.05**2]], ["sig"]), mean=[1.0]
        )
    w.write(outfolder=os.path.dirname(path), outfilename=os.path.basename(path))
    return path + ".hdf5"


def pearson(f):
    mu = f.expected_yield().numpy()
    n = f.nobs.numpy()
    return float(np.sum((n - mu) ** 2 / mu))


def test_pearson():
    """A Poisson likelihood, all bins, only free parameters: the chi2 is
    sum (n - mu)^2 / mu exactly, with ndf nbins - nfree."""
    with tempfile.TemporaryDirectory() as d:
        f = fit(write_tensor(os.path.join(d, "pearson")), noBinByBinStat=True)
        chi2, ndf = postfit_chi2(f)
        assert f.nfreeparms == 2
        assert ndf == 20 - 2
        np.testing.assert_allclose(chi2, pearson(f), rtol=1e-9)


def test_an_empty_bin(caplog):
    """The prediction, not the observation, is the variance: an empty bin keeps
    its degree of freedom."""
    with tempfile.TemporaryDirectory() as d:
        path = write_tensor(os.path.join(d, "empty"))
        nobs = inputdata.FitInputData(path).data_obs.numpy().copy()
        nobs[0] = 0.0
        with caplog.at_level(logging.WARNING):
            f = fit(path, nobs=tf.constant(nobs), noBinByBinStat=True)
            chi2, ndf = postfit_chi2(f)
        assert ndf == 20 - 2
        np.testing.assert_allclose(chi2, pearson(f), rtol=1e-9)
        assert not [r for r in caplog.records if "chi2" in r.getMessage()]


def test_a_frozen_parameter_is_not_fitted():
    with tempfile.TemporaryDirectory() as d:
        path = write_tensor(os.path.join(d, "frozen"))
        f = fit(path, noBinByBinStat=True, freezeParameters=["sig"])
        chi2, ndf = postfit_chi2(f)
        assert f.nfreeparms == 1
        assert ndf == 20 - 1
        np.testing.assert_allclose(chi2, pearson(f), rtol=1e-9)


def test_a_gamma_centre_at_zero(tensors):
    """A frequentist toy can draw a bin-by-bin stat centre beta0 = 0, where the
    gamma constraint is flat: its parameter takes the bin (the closed form's
    limit), and nothing is NaN."""
    f = fit(tensors["abcd"], CI_FITS["abcd"][1])
    beta0 = f.bbstat.beta0.numpy().copy()
    beta0[3] = 0.0
    f.bbstat.set_beta0(beta0)
    f.minimize()
    f._profile_beta()
    chi2, ndf = postfit_chi2(f)
    assert np.isfinite(chi2)
    assert ndf == saturated(f)[1] - 1


def test_external_terms_fall_back_to_the_propagated_covariance(caplog):
    """Curvature without a fluctuating input: the response is not a projection,
    so the covariance is propagated through the fit's, as before."""
    with tempfile.TemporaryDirectory() as d:
        path = write_tensor(os.path.join(d, "external"), external_sig_prior=True)
        f = with_covariance(fit(path, noBinByBinStat=True))
        assert f._fit_response_unsupported() == "external likelihood terms"
        with caplog.at_level(logging.INFO):
            chi2, ndf = postfit_chi2(f)
        assert any(
            "propagated through the fit" in r.getMessage() for r in caplog.records
        )
        assert (chi2, ndf) == propagated_chi2(f)


def test_the_other_fallbacks(tensors):
    f = fit(tensors["abcd"], CI_FITS["abcd"][1])
    assert f._fit_response_unsupported() is None
    f.postfit_faces = {"jacobian": None, "hess_constraints": None}
    assert f._fit_response_unsupported() == "active constraint faces"
    f.postfit_faces = None
    f.regularizers = [object()]
    assert f._fit_response_unsupported() == "regularization"
    f.regularizers = []


def test_sharded():
    """The derivatives are all-bins computations, host-pinned like the chi2."""
    from tests.test_sharded_fit import _make_fitter

    with tempfile.TemporaryDirectory() as d:
        path = make_test_tensor(d)
        out = []
        for ndevices in (1, 2):
            f = _make_fitter(path, ndevices)
            f.minimize()
            f._profile_beta()
            out.append(postfit_chi2(f))
    (chi2_1, ndf_1), (chi2_2, ndf_2) = out
    assert ndf_1 == ndf_2 == 20 - 1
    np.testing.assert_allclose(chi2_2, chi2_1, rtol=1e-8)
