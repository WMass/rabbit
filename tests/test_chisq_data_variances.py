"""--chisqFit uses the data variances of the input in the fit to the data.

The TensorWriter stores the data variances (``hdata_var``) when they differ
from the data, e.g. for weighted data, and the chi-square fit's per-bin
variance is then those rather than the data themselves: Fitter.toyassign passes
them on, and the data histograms (``hist_data_obs``) are drawn with them. The
fits to the observed data in rabbit_fit.py and rabbit_limit.py passed only the
values to Fitter.set_nobs, which then took the data themselves, so
``--chisqFit -t 0`` on a tensor with data variances was bit-identical to the
same fit without them. This is ``--chisqVariance data``, the default: the
driver tests run without the option and with it.

The drivers run in this process (loaded as modules, with their own parsers) on
a 20-bin signal + background tensor whose data are written once with variances
2.5 times the values and once without variances of their own.
"""

import functools
import importlib.util
import sys
from pathlib import Path

import hist
import numpy as np
import pytest

from rabbit import fitter, inputdata, io_tools, tensorwriter
from rabbit.param_models import helpers as ph

REPO = Path(__file__).resolve().parents[1]

VARSCALE = 2.5
CHISQ_DATA_FIT = ("-t", "0", "--chisqFit", "--unblind")
# the default and the option that selects it
DATA_VARIANCE = pytest.mark.parametrize(
    "variance_option", [(), ("--chisqVariance", "data")], ids=["default", "data"]
)


@functools.lru_cache(maxsize=None)
def _driver(name):
    """bin/<name>.py as an importable module."""
    spec = importlib.util.spec_from_file_location(
        f"_{name}_for_chisq_data_variances", REPO / "bin" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_driver(monkeypatch, name, *argv):
    monkeypatch.setattr(sys, "argv", [f"{name}.py", *map(str, argv)])
    _driver(name).main()


def _make_tensor(path, weighted, mc_weight=1.0, pseudodata=False):
    """Signal + background with a norm and a shape systematic; the data are a
    fixed Poisson draw, with variances VARSCALE times the values if
    ``weighted`` (else equal to them, so no hdata_var is written). The template
    events have weight ``mc_weight``; ``pseudodata`` adds a pseudodata set
    "pseudo" of 0.9 times the data, which the TensorWriter writes without
    variances."""
    rng = np.random.default_rng(42)
    ax = hist.axis.Regular(20, -5, 5, name="x")
    h_sig = hist.Hist(ax, storage=hist.storage.Weight())
    h_bkg = hist.Hist(ax, storage=hist.storage.Weight())
    h_sig.fill(rng.normal(0, 1, 10000), weight=mc_weight)
    h_bkg.fill(rng.uniform(-5, 5, 5000), weight=mc_weight)
    data = rng.poisson(1.1 * h_sig.values() + h_bkg.values()).astype(float)
    h_data = hist.Hist(ax, storage=hist.storage.Weight())
    h_data.view()["value"] = data
    h_data.view()["variance"] = VARSCALE * data if weighted else data

    writer = tensorwriter.TensorWriter()
    writer.add_channel(h_data.axes, "ch0")
    writer.add_data(h_data, "ch0")
    if pseudodata:
        h_pseudo = hist.Hist(ax, storage=hist.storage.Double())
        h_pseudo.view()[...] = 0.9 * data
        writer.add_pseudodata(h_pseudo, "pseudo", "ch0")
    writer.add_process(h_sig, "sig", "ch0", signal=True)
    writer.add_process(h_bkg, "bkg", "ch0")
    writer.add_norm_systematic("bkg_norm", "bkg", "ch0", 1.05)
    tilt = 0.01 * (ax.centers - ax.centers[0]) - 0.05
    up, dn = h_bkg.copy(), h_bkg.copy()
    up.values()[...] = h_bkg.values() * (1 + tilt)
    dn.values()[...] = h_bkg.values() * (1 - tilt)
    writer.add_systematic([up, dn], "bkg_shape", "bkg", "ch0", symmetrize="average")
    name = "weighted" if weighted else "unweighted"
    writer.write(outfolder=str(path), outfilename=name)
    return path / f"{name}.hdf5"


def _chisq_fitter(filename, outdir, *extra):
    """A Fitter with the options rabbit_fit.py parses for the chi2 data fit."""
    options = (
        _driver("rabbit_fit")
        .make_parser()
        .parse_args([str(filename), "-o", str(outdir), *CHISQ_DATA_FIT, *extra])
    )
    indata = inputdata.FitInputData(str(filename))
    param_model = ph.load_models([["Mu"]], indata, **vars(options))
    return fitter.Fitter(indata, param_model, options)


def _direct_fit(filename, outdir, variances, *extra, nobs=None):
    """The chi2 fit with set_nobs(nobs, variances), nobs the data by default."""
    f = _chisq_fitter(filename, outdir, *extra)
    f.defaultassign()
    f.set_nobs(f.indata.data_obs if nobs is None else nobs, variances)
    f.minimize()
    _, grad, hess = f.loss_val_grad_hess()
    _, cov = f.edmval_cov(grad, hess)
    return f.x.numpy(), np.diag(np.asarray(cov))


def _driver_fit(monkeypatch, filename, outdir, *extra, result=None):
    _run_driver(
        monkeypatch, "rabbit_fit", filename, "-o", outdir, *CHISQ_DATA_FIT, *extra
    )
    res = io_tools.get_fitresult(str(outdir / "fitresults.hdf5"), result=result)
    parms = res["parms"].get()
    return res, parms.values(), parms.variances()


@DATA_VARIANCE
def test_observed_chisq_fit_uses_the_data_variances(
    variance_option, tmp_path, monkeypatch
):
    filename = _make_tensor(tmp_path, weighted=True)
    indata = inputdata.FitInputData(str(filename))
    assert indata.data_var is not None, "no hdata_var written for weighted data"

    res, x, var = _driver_fit(
        monkeypatch, filename, tmp_path / "driver", "--saveHists", *variance_option
    )
    x_var, var_var = _direct_fit(filename, tmp_path / "direct", indata.data_var)
    x_pois, var_pois = _direct_fit(filename, tmp_path / "direct", None)

    # the observed fit is the fit with the data variances ...
    np.testing.assert_allclose(x, x_var, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(var, var_var, rtol=1e-6, atol=1e-12)
    # ... which is not the one with Poisson variances (what it was)
    assert np.all(var_var > 1.2 * var_pois), (var_var, var_pois)

    # and the data histogram of the fit has the input's variances
    channel = res["mappings"]["BaseMapping"]["channels"]["ch0"]
    data = indata.data_obs.numpy()
    np.testing.assert_allclose(channel["hist_nobs"].get().values(), data)
    np.testing.assert_allclose(channel["hist_nobs"].get().variances(), VARSCALE * data)
    np.testing.assert_allclose(
        channel["hist_data_obs"].get().variances(), VARSCALE * data
    )


@DATA_VARIANCE
def test_observed_chisq_fit_without_data_variances_is_unchanged(
    variance_option, tmp_path, monkeypatch
):
    """No hdata_var in the input: the variances are the data themselves."""
    filename = _make_tensor(tmp_path, weighted=False)
    assert inputdata.FitInputData(str(filename)).data_var is None

    _, x, var = _driver_fit(
        monkeypatch, filename, tmp_path / "driver", *variance_option
    )
    x_pois, var_pois = _direct_fit(filename, tmp_path / "direct", None)

    np.testing.assert_allclose(x, x_pois, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(var, var_pois, rtol=1e-6, atol=1e-12)


def record_chisq_variances(monkeypatch):
    """[(values, varnobs)] of every Fitter.set_nobs from here on."""
    calls = []
    set_nobs = fitter.Fitter.set_nobs

    def recording_set_nobs(self, values, variances=None):
        set_nobs(self, values, variances)
        calls.append((np.asarray(values).copy(), self.varnobs.numpy().copy()))

    monkeypatch.setattr(fitter.Fitter, "set_nobs", recording_set_nobs)
    return calls


def run_limit(monkeypatch, filename, outdir, *extra):
    _run_driver(
        monkeypatch,
        "rabbit_limit",
        filename,
        "-o",
        outdir,
        *CHISQ_DATA_FIT,
        "--allowNegativeParam",
        "--asymptoticLimits",
        "sig",
        "--modes",
        "gaussian",
        *extra,
    )


@DATA_VARIANCE
def test_limit_fits_to_the_data_use_the_data_variances(
    variance_option, tmp_path, monkeypatch
):
    """rabbit_limit.py sets the observed data twice: for the background-only
    fit and again for the unconditional fit after the Asimov dataset. Both
    times the chi2 variances must be the input's."""
    filename = _make_tensor(tmp_path, weighted=True)
    indata = inputdata.FitInputData(str(filename))
    data = indata.data_obs.numpy()

    calls = record_chisq_variances(monkeypatch)
    run_limit(monkeypatch, filename, tmp_path / "limit", *variance_option)

    to_data = [varnobs for values, varnobs in calls if np.array_equal(values, data)]
    assert len(to_data) >= 2, f"expected 2 fits to the data, got {len(to_data)}"
    for varnobs in to_data:
        np.testing.assert_array_equal(varnobs, indata.data_var.numpy())


def test_chisq_refuses_nonpositive_data_variances(tmp_path):
    f = _chisq_fitter(_make_tensor(tmp_path, weighted=True), tmp_path)
    variances = f.indata.data_var.numpy().copy()
    variances[3] = 0.0
    with pytest.raises(RuntimeError, match="data variance <= 0"):
        f.set_nobs(f.indata.data_obs, variances)
