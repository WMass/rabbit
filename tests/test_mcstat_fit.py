"""Limited-MC-statistics de-biasing: minimization and curvature machinery.

The ordinary-likelihood warm start, the Hessian-free (--noHessian) and
--hvpBatch curvatures, and the Asimov recentring.
"""

import hist
import numpy as np
import pytest
import tensorflow as tf

from rabbit.tensorwriter import TensorWriter
from tests.mcstat_helpers import (
    SMALL_NBINS,
    _fold_hist,
    _hist1d,
    _write,
    fit_toy,
    make_mcstat_moment_toy,
    make_small_fold_toy,
)

# ---------------------------------------------------------------------------
# minimization robustness: non-finite trial points, ordinary-likelihood warm start
# ---------------------------------------------------------------------------

POISSON_LINEAR = ("--noBinByBinStat", "--allowNegativeParam")


def test_ordinary_fit_recovers_from_nonfinite_trial_step(tmp_path):
    """From this start, scipy's trust-krylov proposes a step into negative
    expected yields. A NaN loss there used to freeze the trust radius (every
    comparison on a NaN rho is False) until --earlyStopping, and the restart
    re-proposed the same step; it now reads as +inf and the radius shrinks."""
    fn, _ = make_small_fold_toy(tmp_path, 2)
    f = fit_toy(fn, tmp_path, *POISSON_LINEAR, "--mcStatDebias", "none")
    x_ref = f.x.numpy().copy()
    f.x.assign([2.4629323126223066, 9.486694480004758])
    f.minimize()
    np.testing.assert_allclose(f.x.numpy(), x_ref, atol=1e-5)


@pytest.mark.parametrize("bbb", [False, True], ids=["noBBB", "BBB"])
def test_debiased_fit_from_far_start_uses_ordinary_warm_start(bbb, tmp_path):
    """The Poisson jackknife objective is unbounded below where a half
    prediction goes to zero while the full one stays positive, so from a far
    start (e.g. blinding offsets) the de-biased fit could slide away from its
    minimum. minimize() first minimizes the ordinary likelihood, whose minimum
    lies in the convex basin of the de-biased one."""
    fn, _ = make_small_fold_toy(tmp_path, 4)
    args = ("--allowNegativeParam", "--mcStatDebias", "kfold")
    if not bbb:
        args = ("--noBinByBinStat",) + args
    f = fit_toy(fn, tmp_path, *args)
    x_ref = f.x.numpy().copy()
    f.x.assign([3.143434, 13.06379379])  # the blinded start of this toy
    f.minimize()
    np.testing.assert_allclose(f.x.numpy(), x_ref, atol=1e-5)
    _, _, h = f.loss_val_grad_hess()
    assert np.linalg.eigvalsh(np.asarray(h)).min() > 0


def test_warm_start_reuses_traced_functions(tmp_path):
    """The warm start switches between the ordinary and the de-biased
    objective without retracing: each keeps its own traced functions, so a
    run of toys compiles them once (XLA keeps compiled code mapped for the
    life of the process). The kept set must still be the right objective,
    and any rebuild must drop it."""
    fn, _ = make_small_fold_toy(tmp_path, 4)
    f = fit_toy(fn, tmp_path, "--allowNegativeParam", "--mcStatDebias", "kfold")
    debiased = f.loss_val_grad
    ordinary_flag, ordinary = f._mcstat_tf_standby
    assert ordinary_flag is True
    traces = (
        debiased.experimental_get_tracing_count(),
        ordinary["loss_val_grad"].experimental_get_tracing_count(),
    )
    x_ref = f.x.numpy().copy()
    for _ in range(2):
        f.x.assign([3.143434, 13.06379379])
        f.minimize()
        np.testing.assert_allclose(f.x.numpy(), x_ref, atol=1e-5)
    assert f.loss_val_grad is debiased
    assert f._mcstat_tf_standby[1]["loss_val_grad"] is ordinary["loss_val_grad"]
    assert (
        debiased.experimental_get_tracing_count(),
        ordinary["loss_val_grad"].experimental_get_tracing_count(),
    ) == traces

    # the kept set is the ordinary likelihood, the live one the de-biased
    f0 = fit_toy(fn, tmp_path, "--allowNegativeParam", "--mcStatDebias", "none")
    f0.x.assign(f.x)
    with f._ordinary_likelihood():
        assert f.loss_val_grad is ordinary["loss_val_grad"]
        np.testing.assert_allclose(
            f.loss_val().numpy(), f0.loss_val().numpy(), rtol=1e-12
        )
    assert f.loss_val_grad is debiased
    assert not np.isclose(f.loss_val().numpy(), f0.loss_val().numpy())

    f._make_tf_functions()
    assert f._mcstat_tf_standby is None


# ---------------------------------------------------------------------------
# --noHessian: Hessian-free curvature and sandwich
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "toy,extra",
    [
        ("folds", ("--mcStatDebias", "kfold")),
        ("folds", ("--mcStatDebias", "kfold", "--covMode", "fisher")),
        ("folds", ("--mcStatDebias", "none", "--covMode", "fisher")),
        ("moment", ("--mcStatDebias", "continuousM")),
        ("moment", ("--mcStatDebias", "continuousM", "--covMode", "fisher")),
    ],
    ids=["kfold", "kfold-fisher", "none-fisher", "continuousM", "continuousM-fisher"],
)
def test_hessfree_covariance_matches_dense(toy, extra, tmp_path):
    """cov_block_hessfree (conjugate gradient on Hessian-vector products) gives
    the dense curvature A^-1 and the dense sandwich: c_i^T H c_j with
    c_i = A^-1 e_i for the fold methods, C + c_i^T M c_j for continuous-M."""
    if toy == "folds":
        fn, _ = make_small_fold_toy(tmp_path, 3, systematic_type="normal", syst=1)
    else:
        fn, _ = make_mcstat_moment_toy(tmp_path, bbb_variance=True)
    f = fit_toy(fn, tmp_path, "--allowNegativeParam", *extra)
    _, g, h = f.loss_val_grad_hess()
    A = f.fisher_curvature(h) if f.covMode == "fisher" else h
    C = np.linalg.inv(np.asarray(A))
    if not f.mcstat_debias_active:
        S = C
    elif f.mcStatDebias == "continuousM":
        S = np.asarray(f.cov_mcstat_sandwich(A))
    else:
        S = np.asarray(f.cov_twohalf_sandwich(A, covMode=f.covMode))
    idx = np.arange(int(f.x.shape[0]))
    _, curv, rep = f.cov_block_hessfree(g, idx, covMode=f.covMode)
    np.testing.assert_allclose(curv, C, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(rep, S, rtol=1e-6, atol=1e-12)


# ---------------------------------------------------------------------------
# --hvpBatch for the extra Hessians
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bbb", [[], ["--noBinByBinStat"]], ids=["bblite", "nobbb"])
def test_hvp_batch_assembles_the_extra_hessians(bbb, tmp_path):
    """--hvpBatch assembles every Hessian the de-biasing adds (sandwich meat,
    Gauss-Newton forms, MC-variance curvature) from batched HVPs, and they
    match the one-pass tape.jacobian."""
    fn, _ = make_small_fold_toy(tmp_path, 4, syst=1)
    args = ["--allowNegativeParam", "--mcStatDebias", "kfold", "--covMode", "fisher"]
    f = fit_toy(fn, tmp_path, *args, *bbb)
    fb = fit_toy(fn, tmp_path, *args, *bbb, "--hvpBatch", "2")
    assert fb.hessian_from_hvp and not f.hessian_from_hvp
    fb.x.assign(f.x)

    def extra(fit):
        _, _, h = fit.loss_val_grad_hess()
        out = [fit.fisher_curvature(h), fit.mcstat_meat("fisher")]
        out += [fit.hess_meat(), fit._mc_data_variance_curvature()]
        return out + list(fit._mcstat_gn_data_curvatures())

    for a, b in zip(extra(f), extra(fb)):
        a, b = np.asarray(a), np.asarray(b)
        np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-10 * np.abs(a).max())
    assert fb._scalar_hessian_fns, "the batched path was not used"


# ---------------------------------------------------------------------------
# Asimov recentring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "toy,args",
    [
        ("moment", ["--mcStatDebias", "continuousM", "--binByBinStatMode", "full"]),
        ("folds", ["--mcStatDebias", "kfold"]),
        ("folds", ["--mcStatDebias", "kfold", "--binByBinStatMode", "full"]),
        ("folds", ["--mcStatDebias", "none"]),
    ],
    ids=["continuousM-full", "kfold-gammalite", "kfold-full", "none"],
)
def test_asimov_is_recentred_with_the_expected_covariance(toy, args, tmp_path):
    """Asimov data are the expected yields of the noisy templates, so they carry
    no attenuation and the de-biased objective is not stationary at the
    generating point. set_mcstat_asimov recentres it there with a linear term
    (Hessian unchanged) and uses the Gauss-Newton covariance, which does not
    depend on the data: it equals that of real data at the same point. The
    fold methods' split meat depends on the residuals; at Asimov it is its
    expectation over the data, the average over y = mu +- sqrt(var). A fit
    without de-biasing is left untouched."""
    if toy == "moment":
        fn, _ = make_mcstat_moment_toy(tmp_path, bbb_variance=True)
    else:
        fn, _ = make_small_fold_toy(tmp_path, 2)
    common = ["--chisqFit", "--allowNegativeParam", *args]
    f = fit_toy(fn, tmp_path, *common)
    f.defaultassign()
    x0 = f.x.numpy()
    asimov = f.expected_yield()
    f.set_nobs(asimov)
    _, g_raw, h_raw = f.loss_val_grad_hess()
    f.set_mcstat_asimov(True)
    if not f.mcstat_debias_active:
        assert f.covMode == "observed"
        np.testing.assert_array_equal(np.asarray(f.loss_val_grad()[1]), g_raw)
        return
    assert np.abs(np.asarray(g_raw)).max() > 1e-3  # the overshoot it removes
    _, g, h = f.loss_val_grad_hess()
    np.testing.assert_allclose(np.asarray(g), 0.0, atol=1e-8)
    np.testing.assert_allclose(np.asarray(h), np.asarray(h_raw), rtol=1e-12)
    assert f.covMode == "fisher"
    f.minimize()
    np.testing.assert_allclose(f.x.numpy(), x0, atol=1e-6)
    f.x.assign(x0)
    S_asimov = np.asarray(f.mcstat_postfit_covariances(g, h)[3])
    folds = f._fold_debias_active()

    def gauss_newton_part():
        # the data-independent part of the sandwich: Gauss-Newton bread and
        # meat (continuous-M: A^-1 + A^-1 M A^-1), without the split meat
        _, g_, h_ = f.loss_val_grad_hess()
        A = np.asarray(f.fisher_curvature(h_))
        if not folds:
            return np.asarray(f.mcstat_postfit_covariances(g_, h_)[3])
        Ai = np.linalg.inv(A)
        return Ai @ np.asarray(f.fisher_curvature_full(f.hess_meat())) @ Ai

    S_gn_asimov = gauss_newton_part()
    B_asimov = np.asarray(f.mcstat_split_meat())

    f.set_mcstat_asimov(False)
    assert f.covMode == "observed"
    f.covMode = "fisher"
    if folds:
        # the split meat depends on the residuals: at Asimov it is its
        # expectation over the data, the average over y = mu +- sqrt(var)
        Ai = np.linalg.inv(np.asarray(f.fisher_curvature(h)))
        np.testing.assert_allclose(S_asimov, S_gn_asimov + Ai @ B_asimov @ Ai)
        Bs = []
        for sign in (1.0, -1.0):
            y = np.asarray(asimov) + sign * np.sqrt(np.asarray(asimov))
            f.set_nobs(tf.constant(y), variances=asimov)
            Bs.append(np.asarray(f.mcstat_split_meat()))
        np.testing.assert_allclose(B_asimov, 0.5 * (Bs[0] + Bs[1]), rtol=1e-8)

    # real data at the same point, same data variance: same Gauss-Newton part
    rng = np.random.default_rng(1)
    y = rng.normal(np.asarray(asimov), np.sqrt(np.asarray(asimov)))
    f.set_nobs(tf.constant(y), variances=asimov)
    np.testing.assert_allclose(S_gn_asimov, gauss_newton_part(), rtol=1e-6)


def test_asimov_recentring_carries_over_to_the_saturated_composite(tmp_path):
    """On an Asimov dataset the main de-biased objective is recentred at the
    generating point; the saturated projection test re-inits a copy with the
    composite model, which rebuilds the recentring at zero. Recentred again at
    its null point (as rabbit_fit.py does), the saturated fit stays there and
    the projection statistic vanishes, as for any Asimov dataset; without it the
    composite would drift to the raw de-biased minimum."""
    import copy

    from rabbit.param_models.param_model import (
        CompositeParamModel,
        SaturatedProjectModel,
    )

    ngroup = 6
    groups = np.repeat(np.arange(ngroup), SMALL_NBINS // ngroup)
    chinfo = {"ch0": {"axes": [hist.axis.Regular(ngroup, 0, 1, name="g")]}}
    fn, _ = make_small_fold_toy(tmp_path, 2)
    f = fit_toy(fn, tmp_path, "--allowNegativeParam", "--mcStatDebias", "kfold")
    f.defaultassign()
    f.set_nobs(f.expected_yield())
    f.set_mcstat_asimov(True)
    assert f.mcstat_asimov
    nll_main = float(f.reduced_nll().numpy())

    def saturated(recentre):
        orig = f.param_model
        comp = CompositeParamModel(
            [orig, SaturatedProjectModel(f.indata, chinfo, {"ch0": groups})]
        )
        fs = copy.deepcopy(f)
        # The saturated scales and the process strengths share one overall
        # normalisation (mu -> c mu, s_g -> s_g / sqrt(c) leaves every
        # prediction unchanged). The linear recentring term is not flat along
        # that direction, so with it free the recentred null point is a saddle
        # and the minimizer can run away from it. Fix one scale to remove it.
        fs.init_fit_parms(
            comp,
            [],
            unblind=[],
            blinding_group=[],
            freeze_parameters=["^saturated_ch0_g0$"],
        )
        fs.xdefaultassign()
        xm = f.x.numpy()
        fs.x[: orig.nparams].assign(xm[: orig.nparams])
        fs.x[comp.nparams :].assign(xm[orig.nparams :])
        fs.arm_regularizers()
        if recentre:
            fs.set_mcstat_asimov(True)
            # the recentred null point is a minimum, not a saddle
            _, g, h = fs.loss_val_grad_hess()
            i = fs.floating_indices
            np.testing.assert_allclose(np.asarray(g)[i], 0.0, atol=1e-8)
            assert np.linalg.eigvalsh(np.asarray(h)[np.ix_(i, i)]).min() > 0
        fs.minimize(mcstat_warm_start=False)
        return 2 * (nll_main - float(fs.reduced_nll().numpy()))

    assert abs(saturated(True)) < 1e-6
    assert saturated(False) > 1e-3


def test_asimov_split_meat_uses_the_data_covariance(tmp_path):
    """On Asimov the split meat is averaged over y = mu +- sqrt(var), var the
    fit's per-bin data variance: for --covarianceFit the diagonal of the data
    covariance, not the yields."""
    fn, arr = make_small_fold_toy(tmp_path, 2)
    tw_cov = np.diag(2.5 * arr["data"])  # a data covariance unlike the yields
    tw = TensorWriter(sparse=False)
    ax = hist.axis.Regular(SMALL_NBINS, 0.0, 1.0, name="x")
    tw.add_channel([ax], "ch0")
    tw.add_data(_hist1d(ax, arr["data"]), "ch0")
    tw.add_data_covariance(tw_cov)
    for name, folds in zip(("flat", "step"), arr["folds"]):
        tw.add_process(
            _fold_hist(ax, folds), name, "ch0", signal=True, fold_axis="mcfold"
        )
    fn = _write(tw, tmp_path, "covfit_folds.hdf5")
    args = ["--covarianceFit", "--noBinByBinStat", "--allowNegativeParam"]
    f = fit_toy(fn, tmp_path, *args, "--mcStatDebias", "kfold")
    f.defaultassign()
    mu = np.asarray(f.expected_yield())
    f.set_nobs(tf.constant(mu))
    f.set_mcstat_asimov(True)
    B_asimov = np.asarray(f.mcstat_split_meat())
    f.set_mcstat_asimov(False)
    Bs = []
    for sign in (1.0, -1.0):
        f.set_nobs(tf.constant(mu + sign * np.sqrt(np.diag(tw_cov))))
        Bs.append(np.asarray(f.mcstat_split_meat()))
    np.testing.assert_allclose(B_asimov, 0.5 * (Bs[0] + Bs[1]), rtol=1e-8)
