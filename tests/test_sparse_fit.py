"""
Test that writes a simple tensor in both dense and sparse modes,
runs a fit on each, and verifies that sparse mode produces consistent results.
"""

import os
import tempfile
from types import SimpleNamespace

import hist
import numpy as np

from rabbit import fitter, inputdata, tensorwriter
from rabbit.param_models.helpers import load_model


def make_histograms():
    """Generate common histograms for test tensors."""
    np.random.seed(42)

    ax = hist.axis.Regular(20, -5, 5, name="x")

    h_data = hist.Hist(ax, storage=hist.storage.Double())
    h_sig = hist.Hist(ax, storage=hist.storage.Weight())
    h_bkg = hist.Hist(ax, storage=hist.storage.Weight())

    x_sig = np.random.normal(0, 1, 10000)
    x_bkg = np.random.uniform(-5, 5, 5000)

    h_data.fill(np.concatenate([x_sig, x_bkg]))
    h_sig.fill(x_sig, weight=np.ones(len(x_sig)))
    h_bkg.fill(x_bkg, weight=np.ones(len(x_bkg)))

    # scale signal down by 10% so the fit has something to recover
    h_sig.values()[...] = h_sig.values() * 0.9

    # shape systematic on background: linear tilt
    bin_centers = h_bkg.axes[0].centers
    bin_centers_shifted = bin_centers - bin_centers[0]
    weights = 0.01 * bin_centers_shifted - 0.05

    h_bkg_syst_up = h_bkg.copy()
    h_bkg_syst_dn = h_bkg.copy()
    h_bkg_syst_up.values()[...] = h_bkg.values() * (1 + weights)
    h_bkg_syst_dn.values()[...] = h_bkg.values() * (1 - weights)

    # difference histograms (variation - nominal)
    h_bkg_syst_up_diff = h_bkg.copy()
    h_bkg_syst_dn_diff = h_bkg.copy()
    h_bkg_syst_up_diff.values()[...] = h_bkg.values() * weights
    h_bkg_syst_dn_diff.values()[...] = h_bkg.values() * (-weights)

    return dict(
        data=h_data,
        sig=h_sig,
        bkg=h_bkg,
        syst_up=h_bkg_syst_up,
        syst_dn=h_bkg_syst_dn,
        syst_up_diff=h_bkg_syst_up_diff,
        syst_dn_diff=h_bkg_syst_dn_diff,
    )


def _to_scipy_sparse(h):
    """Convert a hist histogram to a scipy sparse CSR array of its values."""
    import scipy.sparse

    return scipy.sparse.csr_array(h.values())


def make_test_tensor(
    outdir, sparse=False, as_difference=False, scipy_sparse_input=False
):
    """Create a simple tensor with signal + background + one shape systematic."""

    hists = make_histograms()

    writer = tensorwriter.TensorWriter(sparse=sparse)

    writer.add_channel(hists["data"].axes, "ch0")
    writer.add_data(hists["data"], "ch0")

    if scipy_sparse_input:
        writer.add_process(
            _to_scipy_sparse(hists["sig"]),
            "sig",
            "ch0",
            signal=True,
            variances=hists["sig"].variances(),
        )
        writer.add_process(
            _to_scipy_sparse(hists["bkg"]),
            "bkg",
            "ch0",
            variances=hists["bkg"].variances(),
        )
    else:
        writer.add_process(hists["sig"], "sig", "ch0", signal=True)
        writer.add_process(hists["bkg"], "bkg", "ch0")

    writer.add_norm_systematic("bkg_norm", "bkg", "ch0", 1.05)

    if as_difference:
        syst_up = hists["syst_up_diff"]
        syst_dn = hists["syst_dn_diff"]
        if scipy_sparse_input:
            syst_up = _to_scipy_sparse(syst_up)
            syst_dn = _to_scipy_sparse(syst_dn)
        writer.add_systematic(
            [syst_up, syst_dn],
            "bkg_shape",
            "bkg",
            "ch0",
            symmetrize="average",
            as_difference=True,
        )
    else:
        syst_up = hists["syst_up"]
        syst_dn = hists["syst_dn"]
        if scipy_sparse_input:
            syst_up = _to_scipy_sparse(syst_up)
            syst_dn = _to_scipy_sparse(syst_dn)
        writer.add_systematic(
            [syst_up, syst_dn],
            "bkg_shape",
            "bkg",
            "ch0",
            symmetrize="average",
        )

    suffix = "sparse" if sparse else "dense"
    if as_difference:
        suffix += "_diff"
    if scipy_sparse_input:
        suffix += "_scipy"
    name = f"test_{suffix}"
    writer.write(outfolder=outdir, outfilename=name)
    return os.path.join(outdir, f"{name}.hdf5")


def make_options(**kwargs):
    """Create a minimal options namespace for the Fitter."""
    defaults = dict(
        earlyStopping=-1,
        noBinByBinStat=False,
        binByBinStatMode="lite",
        binByBinStatType="automatic",
        covarianceFit=False,
        chisqFit=False,
        diagnostics=False,
        minimizerMethod="trust-krylov",
        prefitUnconstrainedNuisanceUncertainty=0.0,
        freezeParameters=[],
        setConstraintMinimum=[],
        unblind=[],
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def run_fit(filename):
    """Load tensor, set up fitter, run fit to data, return results."""

    indata_obj = inputdata.FitInputData(filename)
    param_model = load_model("Mu", indata_obj)

    options = make_options()
    f = fitter.Fitter(indata_obj, param_model, options)

    # fit to observed data
    f.set_nobs(indata_obj.data_obs)
    f.minimize()

    # compute hessian covariance
    val, grad, hess = f.loss_val_grad_hess()
    from rabbit.tfhelpers import edmval_cov

    edmval, cov = edmval_cov(grad, hess)

    param_val = f.x[: param_model.nparams].numpy()
    theta_val = f.x[param_model.nparams :].numpy()
    cov_np = cov.numpy() if hasattr(cov, "numpy") else np.asarray(cov)
    param_err = np.sqrt(np.diag(cov_np)[: param_model.nparams])
    nll = f.reduced_nll().numpy()

    return dict(
        param=param_val,
        theta=theta_val,
        param_err=param_err,
        nll=nll,
        edmval=edmval,
        parms=f.parms,
    )


def check_results(label_a, res_a, label_b, res_b, atol=1e-5, rtol=1e-4):
    """Compare two fit results and return True if they match."""

    print(f"\n--- {label_a} vs {label_b} ---")

    for label, res in [(label_a, res_a), (label_b, res_b)]:
        print(f"\n{label}:")
        for i, name in enumerate(res["parms"][: len(res["param"])]):
            print(f"  {name}: {res['param'][i]:.6f} +/- {res['param_err'][i]:.6f}")
        for i, name in enumerate(res["parms"][len(res["param"]) :]):
            print(f"  {name}: {res['theta'][i]:.6f}")
        print(f"  reduced NLL: {res['nll']:.6f}")
        print(f"  EDM: {res['edmval']:.2e}")

    param_match = np.allclose(res_a["param"], res_b["param"], atol=atol, rtol=rtol)
    theta_match = np.allclose(res_a["theta"], res_b["theta"], atol=atol, rtol=rtol)
    err_match = np.allclose(
        res_a["param_err"], res_b["param_err"], atol=atol, rtol=rtol
    )
    nll_match = np.isclose(res_a["nll"], res_b["nll"], atol=atol, rtol=rtol)

    all_ok = param_match and theta_match and err_match and nll_match

    print(f"\n  param values match:        {param_match}")
    print(f"  Theta values match:      {theta_match}")
    print(f"  param uncertainties match:  {err_match}")
    print(f"  NLL values match:         {nll_match}")

    if not param_match:
        print(f"    {label_a} param:  {res_a['param']}")
        print(f"    {label_b} param:  {res_b['param']}")
        print(f"    diff:       {res_a['param'] - res_b['param']}")

    if not theta_match:
        print(f"    {label_a} theta:  {res_a['theta']}")
        print(f"    {label_b} theta:  {res_b['theta']}")
        print(f"    diff:         {res_a['theta'] - res_b['theta']}")

    if not nll_match:
        print(f"    {label_a} NLL:  {res_a['nll']}")
        print(f"    {label_b} NLL:  {res_b['nll']}")
        print(f"    diff:       {res_a['nll'] - res_b['nll']}")

    return all_ok


def main():
    import tensorflow as tf

    tf.config.experimental.enable_op_determinism()

    with tempfile.TemporaryDirectory() as tmpdir:
        # create tensors in all modes
        configs = [
            ("Dense", make_test_tensor(tmpdir, sparse=False)),
            ("Sparse", make_test_tensor(tmpdir, sparse=True)),
            (
                "Dense (as_difference)",
                make_test_tensor(tmpdir, sparse=False, as_difference=True),
            ),
            (
                "Sparse (as_difference)",
                make_test_tensor(tmpdir, sparse=True, as_difference=True),
            ),
            (
                "Sparse (scipy)",
                make_test_tensor(tmpdir, sparse=True, scipy_sparse_input=True),
            ),
            (
                "Sparse (scipy+diff)",
                make_test_tensor(
                    tmpdir, sparse=True, as_difference=True, scipy_sparse_input=True
                ),
            ),
        ]

        results = {}
        for label, fpath in configs:
            print("=" * 60)
            print(f"Running {label} fit...")
            print("=" * 60)
            results[label] = run_fit(fpath)
            print()

        # check consistency across all pairs vs the dense baseline
        print("=" * 60)
        print("Consistency checks")
        print("=" * 60)

        checks = [
            ("Dense", "Sparse"),
            ("Dense", "Dense (as_difference)"),
            ("Dense", "Sparse (as_difference)"),
            ("Dense", "Sparse (scipy)"),
            ("Dense", "Sparse (scipy+diff)"),
        ]

        all_ok = True
        for label_a, label_b in checks:
            ok = check_results(label_a, results[label_a], label_b, results[label_b])
            all_ok = all_ok and ok

        print()
        if all_ok:
            print("ALL CHECKS PASSED")
        else:
            print("SOME CHECKS FAILED")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
