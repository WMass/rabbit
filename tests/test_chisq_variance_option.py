"""--chisqVariance, the per-bin variance of the --chisqFit chi2.

'data', the default, is tested with and without the option in
tests/test_chisq_data_variances.py. 'mc' is the MC statistical variance of the
nominal templates, the sum over the processes of their sumw2, for every dataset
the drivers fit: the data, the Asimov dataset, the toys (their normal
randomization width and their chi2), pseudodata and the fits of
rabbit_limit.py. The templates here are filled with weight 0.5, so that this
variance (half the expected yields) differs from the data, their variances
(2.5 times the data) and the expected yields.
"""

import hist
import numpy as np
import pytest
import tensorflow as tf

from rabbit import fitter, inputdata, tensorwriter
from rabbit.param_models import helpers as ph

from .test_chisq_data_variances import (
    _chisq_fitter,
    _direct_fit,
    _driver,
    _driver_fit,
    _make_tensor,
    record_chisq_variances,
    run_limit,
)

MC_WEIGHT = 0.5
MC = ("--chisqVariance", "mc", "--noBinByBinStat")


@pytest.fixture
def tensor(tmp_path):
    return _make_tensor(tmp_path, weighted=True, mc_weight=MC_WEIGHT, pseudodata=True)


def _mc_variance(indata):
    return np.sum(indata.sumw2.numpy(), axis=-1)[: indata.nbins]


def _hist_nobs(res):
    return res["mappings"]["BaseMapping"]["channels"]["ch0"]["hist_nobs"].get()


def test_the_mc_variance_is_the_sum_of_the_templates_sumw2(tensor, tmp_path):
    f = _chisq_fitter(tensor, tmp_path, *MC)
    expected = f.expected_yield().numpy()
    np.testing.assert_allclose(f.mc_variance.numpy(), _mc_variance(f.indata))
    np.testing.assert_allclose(f.mc_variance.numpy(), MC_WEIGHT * expected)
    for other in (f.indata.data_obs, f.indata.data_var, expected):
        assert not np.allclose(f.mc_variance.numpy(), other)


def test_observed_fit_with_the_mc_variance(tensor, tmp_path, monkeypatch):
    """The fit to the data is the one with set_nobs(data, sum sumw2): the data
    variances of the input are not used."""
    indata = inputdata.FitInputData(str(tensor))
    mc_variance = _mc_variance(indata)

    res, x, var = _driver_fit(monkeypatch, tensor, tmp_path / "mc", *MC, "--saveHists")
    x_ref, var_ref = _direct_fit(
        tensor, tmp_path / "direct", mc_variance, "--noBinByBinStat"
    )
    np.testing.assert_allclose(x, x_ref, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(var, var_ref, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(_hist_nobs(res).variances(), mc_variance)

    # with the data variances (2.5 times the data, against half the yields)
    _, _, var_data = _driver_fit(
        monkeypatch, tensor, tmp_path / "data", "--noBinByBinStat"
    )
    assert np.all(var_data > var) and var_data[0] > 3 * var[0], (var_data, var)


@pytest.mark.parametrize("variance", ["data", "mc"])
def test_asimov_fit_follows_the_option(variance, tensor, tmp_path, monkeypatch):
    """-t -1: the expected yields as the variances by default, the MC
    variance with mc."""
    options = MC if variance == "mc" else ("--noBinByBinStat",)
    # -t -1 replaces the -t 0 of CHISQ_DATA_FIT: the Asimov fit only
    res, x, var = _driver_fit(
        monkeypatch,
        tensor,
        tmp_path / "asimov",
        "-t",
        "-1",
        "--saveHists",
        *options,
        result="asimov",
    )

    f = _chisq_fitter(tensor, tmp_path, "--noBinByBinStat")
    expected = f.expected_yield().numpy()
    want = _mc_variance(f.indata) if variance == "mc" else expected
    np.testing.assert_allclose(_hist_nobs(res).values(), expected, rtol=1e-12)
    np.testing.assert_allclose(_hist_nobs(res).variances(), want, rtol=1e-12)

    x_ref, var_ref = _direct_fit(
        tensor, tmp_path / "direct", want, "--noBinByBinStat", nobs=expected
    )
    np.testing.assert_allclose(x, x_ref, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(var, var_ref, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("variance", ["data", "mc"])
def test_toys_follow_the_option(variance, tensor, tmp_path, monkeypatch):
    """Normal toys are drawn with the chi2 variance and fitted with it; a
    Poisson toy is fitted with its counts by default, with the MC variance with
    mc."""
    options = MC if variance == "mc" else ("--noBinByBinStat",)
    f = _chisq_fitter(tensor, tmp_path, *options)
    data, data_var = f.indata.data_obs, f.indata.data_var
    want = _mc_variance(f.indata) if variance == "mc" else data_var.numpy()

    stddevs = []
    normal = tf.random.normal

    def recording_normal(*args, **kwargs):
        stddevs.append(np.asarray(kwargs["stddev"]))
        return normal(*args, **kwargs)

    monkeypatch.setattr(tf.random, "normal", recording_normal)
    for data_mode in ("observed", "expected"):
        f.toyassign(
            data,
            data_var,
            syst_randomize="none",
            data_randomize="normal",
            data_mode=data_mode,
        )
        np.testing.assert_allclose(stddevs[-1] ** 2, want, rtol=1e-12)
        np.testing.assert_allclose(f.varnobs.numpy(), want, rtol=1e-12)

    f.toyassign(data, data_var, syst_randomize="none", data_randomize="poisson")
    np.testing.assert_allclose(
        f.varnobs.numpy(), want if variance == "mc" else f.nobs.numpy(), rtol=1e-12
    )


@pytest.mark.parametrize("variance", ["data", "mc"])
def test_pseudodata_fit_follows_the_option(variance, tensor, tmp_path, monkeypatch):
    """--pseudoData: the TensorWriter writes no pseudodata variances, so the
    pseudodata themselves by default, the MC variance with mc."""
    options = MC if variance == "mc" else ("--noBinByBinStat",)
    res, x, var = _driver_fit(
        monkeypatch,
        tensor,
        tmp_path / "pseudo",
        "--pseudoData",
        "pseudo",
        "--saveHists",
        *options,
        result="pseudo",
    )
    indata = inputdata.FitInputData(str(tensor), ["pseudo"])
    assert indata.pseudodata_var is None
    pseudo = indata.pseudodata_obs[:, 0].numpy()
    want = _mc_variance(indata) if variance == "mc" else pseudo
    np.testing.assert_allclose(_hist_nobs(res).values(), pseudo)
    np.testing.assert_allclose(_hist_nobs(res).variances(), want, rtol=1e-12)

    x_ref, var_ref = _direct_fit(
        tensor, tmp_path / "direct", want, "--noBinByBinStat", nobs=pseudo
    )
    np.testing.assert_allclose(x, x_ref, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(var, var_ref, rtol=1e-6, atol=1e-12)


def test_limit_fits_follow_the_option(tensor, tmp_path, monkeypatch):
    """rabbit_limit.py: the fits to the data and to the Asimov dataset of the
    background-only fit all take the MC variance."""
    mc_variance = _mc_variance(inputdata.FitInputData(str(tensor)))
    calls = record_chisq_variances(monkeypatch)
    run_limit(monkeypatch, tensor, tmp_path / "limit", *MC)
    assert len(calls) >= 3, f"expected 3 datasets, got {len(calls)}"
    for _, varnobs in calls:
        np.testing.assert_array_equal(varnobs, mc_variance)


@pytest.mark.parametrize(
    "variance", [("data",), ("mc", "--noBinByBinStat")], ids=["data", "mc"]
)
@pytest.mark.parametrize("fit", [(), ("--covarianceFit",)], ids=["poisson", "cov"])
def test_the_option_needs_chisqFit(variance, fit, tensor, tmp_path):
    options = (
        _driver("rabbit_fit")
        .make_parser()
        .parse_args(
            [str(tensor), "-o", str(tmp_path), "--chisqVariance", *variance, *fit]
        )
    )
    indata = inputdata.FitInputData(str(tensor))
    model = ph.load_models([["Mu"]], indata, **vars(options))
    with pytest.raises(ValueError, match="only applies to --chisqFit"):
        fitter.Fitter(indata, model, options)


@pytest.mark.parametrize(
    "extra, match",
    [
        ((), "--noBinByBinStat"),
        (("--binByBinStatType", "normal-additive"), "--noBinByBinStat"),
        (("--noBinByBinStat", "--mcStatDebias", "kfold"), "--mcStatDebias"),
    ],
    ids=["bbb", "bbb-normal-additive", "mcstat-debias"],
)
def test_the_mc_variance_refuses_counting_the_mc_statistics_twice(
    extra, match, tensor, tmp_path
):
    with pytest.raises(ValueError, match=match):
        _chisq_fitter(tensor, tmp_path, "--chisqVariance", "mc", *extra)


def test_the_mc_variance_refuses_bins_without_mc_statistics(tmp_path):
    ax = hist.axis.Regular(3, 0, 3, name="x")
    writer = tensorwriter.TensorWriter()
    writer.add_channel([ax], "ch0")
    for name, values in (("data", [3.0, 15.0, 25.0]), ("sig", [0.0, 10.0, 20.0])):
        h = hist.Hist(ax, storage=hist.storage.Weight())
        h.view()["value"] = values
        h.view()["variance"] = values
        if name == "data":
            writer.add_data(h, "ch0")
        else:
            writer.add_process(h, name, "ch0", signal=True)
    writer.write(outfolder=str(tmp_path), outfilename="empty")
    with pytest.raises(RuntimeError, match="1 bins with MC statistical variance"):
        _chisq_fitter(tmp_path / "empty.hdf5", tmp_path, *MC)
