"""Tests for multi-device (bins-sharded) likelihood evaluation.

Without GPUs the shards all land on the CPU (sharding.select_devices
falls back with a warning), which exercises every piece of the sharded
machinery -- the bin slicing, the per-shard BinByBinStat, the borrowed
methods, the per-shard tapes, the partial-sum combiners -- with no
device separation. Device *placement* correctness was established
separately on a 4-GPU node (see rabbit/sharding.py).

The bar everywhere is near-bitwise agreement with the single-device
path: sharding is a memory layout, not an approximation.
"""

import re
import sys
import tempfile

import numpy as np
import pytest
import tensorflow as tf

from rabbit import fitter, inputdata
from rabbit.param_models.helpers import load_model
from tests.test_logk_no_copy import _normal_tensor
from tests.test_sparse_fit import make_options, make_test_tensor


def _make_fitter(filename, ndevices=1, do_blinding=False, expect_signal=None, **kw):
    indata_obj = inputdata.FitInputData(filename, host_memory=ndevices > 1)
    param_model = load_model("Mu", indata_obj, expectSignal=expect_signal)
    options = make_options(nDevices=ndevices, **kw)
    # pass the kwargs rabbit_fit passes, so the factory can never silently
    # drop one again (it did once: globalImpactsFromJVP)
    f = fitter.make_fitter(
        indata_obj,
        param_model,
        options,
        do_blinding=do_blinding,
        globalImpactsFromJVP=True,
    )
    f.set_nobs(indata_obj.data_obs)
    return f


RTOL = 1e-12


@pytest.mark.parametrize("ndevices", [2, 3])
def test_sharded_loss_grad_hvp_hess_match(ndevices):
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = make_test_tensor(tmpdir)
        f1 = _make_fitter(fname, 1)
        fn = _make_fitter(fname, ndevices)
        assert len(fn.shards) == ndevices

        # move off the trivial starting point so gradients are non-trivial
        rng = np.random.default_rng(3)
        xval = f1.x.numpy() + 0.1 * rng.standard_normal(f1.x.shape[0])
        f1.x.assign(xval)
        fn.x.assign(xval)

        v1, g1 = f1.loss_val_grad()
        vn, gn = fn.loss_val_grad()
        np.testing.assert_allclose(float(vn), float(v1), rtol=RTOL)
        np.testing.assert_allclose(gn.numpy(), g1.numpy(), rtol=1e-10, atol=1e-10)

        np.testing.assert_allclose(
            float(fn.loss_val()), float(f1.loss_val()), rtol=RTOL
        )

        p = tf.constant(rng.standard_normal(f1.x.shape[0]))
        _, _, h1 = f1.loss_val_grad_hessp(p)
        _, _, hn = fn.loss_val_grad_hessp(p)
        np.testing.assert_allclose(hn.numpy(), h1.numpy(), rtol=1e-9, atol=1e-9)

        _, _, H1 = f1.loss_val_grad_hess()
        _, _, Hn = fn.loss_val_grad_hess()
        np.testing.assert_allclose(Hn.numpy(), H1.numpy(), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("ndevices", [2, 3])
def test_sharded_loss_matches_single_device_with_blinding_armed(ndevices):
    """The sharded loss must agree with the single-device one while blinded.

    Blinding is armed by default for an observed-data fit and is not in the
    up-front refusal list, so `--nDevices N` on data is the headline use case,
    not an exotic one. Every other test in this file builds with
    do_blinding=False, which is why a get_poi() that grew a third offset on main
    could reach the shards unthreaded and crash the first loss evaluation with
    AttributeError -- the offsets are read whatever their value, so an
    all-identity run does not dodge it either.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname = make_test_tensor(tmp)
        f1 = _make_fitter(fname, 1, do_blinding=True)
        fn = _make_fitter(fname, ndevices, do_blinding=True)
        for f in (f1, fn):
            f.set_blinding_offsets(True)

        v1, g1 = f1.loss_val_grad()
        vn, gn = fn.loss_val_grad()
        assert np.isclose(float(v1.numpy()), float(vn.numpy()), rtol=RTOL, atol=0)
        np.testing.assert_allclose(g1.numpy(), gn.numpy(), rtol=1e-10, atol=0)


def test_every_blinding_offset_is_threaded_to_the_shards():
    """The shard evaluators are duck-typed, so a missed offset is an
    AttributeError at the first armed fit rather than a type error at import.

    _BLINDING_OFFSET_ATTRS is what the three threading sites iterate; this
    pins it against what a blinded Fitter's Blinding actually creates. Adding
    an offset without listing it, or listing one that no longer exists, fails
    here instead of in somebody's fit -- and the same names must exist on
    BlindingView, which is what each shard pins the threaded tensors onto.
    """
    from rabbit.blinding import BlindingView
    from rabbit.sharding import _BLINDING_OFFSET_ATTRS

    with tempfile.TemporaryDirectory() as tmp:
        f = _make_fitter(make_test_tensor(tmp), 1, do_blinding=True)

    created = {n for n in vars(f.blinding) if n.startswith("offsets_")}
    assert created, "no offsets created; test is vacuous"
    assert created == set(_BLINDING_OFFSET_ATTRS), (
        "the offsets a blinded Fitter creates and the ones sharding threads have "
        f"diverged: Blinding has {sorted(created)}, sharding threads "
        f"{sorted(_BLINDING_OFFSET_ATTRS)}"
    )
    view = set(vars(BlindingView(True)))
    missing = sorted(set(_BLINDING_OFFSET_ATTRS) - view)
    assert not missing, (
        f"BlindingView has no slot for {missing}, so a shard would evaluate "
        "with a stale or absent offset while the Fitter applies one"
    )


# rnorm_init is the identity unless the param model's default differs from 1,
# so a model left at mu = 1 cannot tell a threaded factor from a missing one.
NORMAL_EXPECT_SIGNAL = [("sig", 1.7)]


@pytest.mark.parametrize("ndevices", [2, 3])
def test_sharded_normal_systematics_match_single_device(ndevices):
    """systematic_type == "normal" reaches the shards through rnorm_init.

    Every other fixture in this file is log_normal, where the multiplicative
    form carries the param-model scaling itself and the hot path never reads
    rnorm_init. That blind spot is why the whole suite stayed green when the
    post-contraction scaling arrived on main and was threaded to no shard:
    the shard evaluators are duck-typed, so the first sharded "normal" fit
    died with AttributeError while all 38 tests here passed.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname = _normal_tensor(tmp, sparse=False)
        f1 = _make_fitter(fname, 1, expect_signal=NORMAL_EXPECT_SIGNAL)
        fn = _make_fitter(fname, ndevices, expect_signal=NORMAL_EXPECT_SIGNAL)
        assert f1.indata.systematic_type == "normal"
        assert f1.rnorm_init is not None, "no scaling to thread; test is vacuous"
        assert not np.allclose(
            f1.rnorm_init.numpy(), 1.0
        ), "rnorm_init is the identity; test is vacuous"

        rng = np.random.default_rng(11)
        xval = f1.x.numpy() + 0.1 * rng.standard_normal(f1.x.shape[0])
        f1.x.assign(xval)
        fn.x.assign(xval)

        v1, g1 = f1.loss_val_grad()
        vn, gn = fn.loss_val_grad()
        assert np.isclose(float(v1.numpy()), float(vn.numpy()), rtol=RTOL, atol=0)
        np.testing.assert_allclose(g1.numpy(), gn.numpy(), rtol=1e-10, atol=0)


def test_rnorm_init_is_sharded_to_each_shards_own_bins():
    """Each shard gets its own bin slice of rnorm_init, not the whole tensor.

    The hot path indexes it as rnorm_init[:nbins] against the shard's *local*
    bin count, so handing over the full [nbinsfull, nproc] tensor would silently
    scale every shard by the leading bins instead of its own. The behavioural
    test above cannot see that: Mu.compute returns one row broadcast over bins,
    so a mis-slice is numerically invisible for the only models this fixture can
    build. The shapes still differ, which is what this pins.
    """
    from rabbit.sharding import shard_edges

    ndevices = 3
    with tempfile.TemporaryDirectory() as tmp:
        fn = _make_fitter(
            _normal_tensor(tmp, sparse=False),
            ndevices,
            expect_signal=NORMAL_EXPECT_SIGNAL,
        )
        assert fn.rnorm_init is not None, "no scaling to thread; test is vacuous"
        edges = shard_edges(fn.indata.nbins, ndevices)
        assert len(edges) == len(fn.shards)
        for shard, (a, b) in zip(fn.shards, edges):
            assert shard.rnorm_init is not None, "shard lost the scaling factor"
            assert shard.rnorm_init.shape[0] == b - a == shard.indata.nbins
            np.testing.assert_array_equal(
                shard.rnorm_init.numpy(), fn.rnorm_init.numpy()[a:b]
            )


def test_sharded_profile_beta_matches():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = make_test_tensor(tmpdir)
        f1 = _make_fitter(fname, 1)
        fn = _make_fitter(fname, 2)

        rng = np.random.default_rng(7)
        xval = f1.x.numpy() + 0.05 * rng.standard_normal(f1.x.shape[0])
        f1.x.assign(xval)
        fn.x.assign(xval)

        f1._profile_beta()
        fn._profile_beta()
        np.testing.assert_allclose(
            fn.bbstat.beta.numpy(), f1.bbstat.beta.numpy(), rtol=1e-10, atol=1e-12
        )


@pytest.mark.parametrize("method", ["trust-krylov", "tf-trust-krylov"])
def test_sharded_fit_matches_single_device(method):
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = make_test_tensor(tmpdir)
        f1 = _make_fitter(fname, 1, minimizerMethod=method)
        fn = _make_fitter(fname, 2, minimizerMethod=method)

        f1.minimize()
        fn.minimize()

        np.testing.assert_allclose(fn.x.numpy(), f1.x.numpy(), rtol=1e-6, atol=1e-7)
        v1, g1, H1 = f1.loss_val_grad_hess()
        vn, gn, Hn = fn.loss_val_grad_hess()
        # the offset-form NLL is ~0 at the minimum of this Asimov-like fit,
        # so compare absolutely at float64 cancellation scale
        np.testing.assert_allclose(float(vn), float(v1), atol=1e-9)

        from rabbit.tfhelpers import edmval_cov

        edm1, _ = edmval_cov(g1, H1)
        edmn, _ = edmval_cov(gn, Hn)
        assert float(edmn) < 1e-4 and float(edm1) < 1e-4


def test_sharded_rejects_unsupported():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = make_test_tensor(tmpdir, sparse=True)
        indata_obj = inputdata.FitInputData(fname)
        param_model = load_model("Mu", indata_obj)
        options = make_options(nDevices=2)
        with pytest.raises(NotImplementedError):
            fitter.make_fitter(indata_obj, param_model, options)


def test_sharded_fitter_deepcopy():
    """Toys deepcopy the fitter; the shard machinery must be rebuilt."""
    import copy

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = make_test_tensor(tmpdir)
        fn = _make_fitter(fname, 2)
        v0 = float(fn.loss_val())
        fc = copy.deepcopy(fn)
        assert len(fc.shards) == 2
        np.testing.assert_allclose(float(fc.loss_val()), v0, rtol=RTOL)


def test_sharded_across_logical_cpu_devices():
    """Run the sharded path across two genuinely distinct (logical CPU)
    devices in a subprocess, where the TF context can still be configured.

    This is the regression net for the bug class the same-device CPU tests
    are structurally blind to: XLA-compiled shard functions capturing
    state resident on another device (first seen as jit functions reading
    the fitter's frozen_params_mask Variable from GPU:0)."""
    import os
    import subprocess
    import sys
    import textwrap

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = make_test_tensor(tmpdir)
        script = textwrap.dedent(f"""
            import numpy as np
            import tensorflow as tf

            cpu = tf.config.list_physical_devices("CPU")[0]
            tf.config.set_logical_device_configuration(
                cpu,
                [tf.config.LogicalDeviceConfiguration()] * 2,
            )

            from rabbit import fitter, inputdata
            from rabbit.param_models.helpers import load_model
            from tests.test_sparse_fit import make_options

            def build(nd):
                indata = inputdata.FitInputData({fname!r}, host_memory=nd > 1)
                pm = load_model("Mu", indata)
                f = fitter.make_fitter(indata, pm, make_options(nDevices=nd))
                f.set_nobs(indata.data_obs)
                return f

            f1 = build(1)
            f2 = build(2)
            devs = {{s.device for s in f2.shards}}
            assert len(devs) == 2, devs

            rng = np.random.default_rng(5)
            xval = f1.x.numpy() + 0.1 * rng.standard_normal(f1.x.shape[0])
            f1.x.assign(xval)
            f2.x.assign(xval)

            v1, g1 = f1.loss_val_grad()
            v2, g2 = f2.loss_val_grad()
            np.testing.assert_allclose(float(v2), float(v1), rtol=1e-12)
            np.testing.assert_allclose(g2.numpy(), g1.numpy(), rtol=1e-10, atol=1e-10)

            p = tf.constant(rng.standard_normal(f1.x.shape[0]))
            _, _, h1 = f1.loss_val_grad_hessp(p)
            _, _, h2 = f2.loss_val_grad_hessp(p)
            np.testing.assert_allclose(h2.numpy(), h1.numpy(), rtol=1e-9, atol=1e-9)

            _, _, H1 = f1.loss_val_grad_hess()
            _, _, H2 = f2.loss_val_grad_hess()
            np.testing.assert_allclose(H2.numpy(), H1.numpy(), rtol=1e-9, atol=1e-9)

            f2._profile_beta()
            print("LOGICAL-DEVICE SHARDING OK")
            """)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [os.getcwd()] + env.get("PYTHONPATH", "").split(os.pathsep)
        )
        env["TF_CPP_MIN_LOG_LEVEL"] = "3"
        res = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=420,
        )
        assert res.returncode == 0, res.stdout + "\n" + res.stderr
        assert "LOGICAL-DEVICE SHARDING OK" in res.stdout


class _YieldReadingPenalty:
    """Declares that it reads the yields, and checks that it is given them.

    Exists to make the L-curve list *mixed*: with only parameter-only penalties
    attached, `any(needs_observables)` is False and a global flag behaves
    identically to a per-regularizer one, so a single-penalty test cannot tell
    the two apart.
    """

    needs_observables = True

    def __init__(self, strength=1e-3):
        self.strength = strength

    def set_expectations(self, initial_params, initial_observables, parms=None):
        assert initial_observables is not None, "must be handed yields"

    def compute_nll_penalty(self, params, observables=None):
        assert observables is not None, "must be handed yields"
        return self.strength * tf.reduce_sum(observables[:1] ** 2)


class _ParamOnlyPenalty:
    """Quadratic pull on the first parameter; ignores the yields entirely."""

    needs_observables = False

    def __init__(self, strength=7.0):
        self.strength = strength
        self.armed = 0

    def set_expectations(self, initial_params, initial_observables, parms=None):
        assert initial_observables is None, "must not be handed yields"
        self.armed += 1

    def compute_nll_penalty(self, params, observables=None):
        assert observables is None, "must not be handed yields"
        return self.strength * tf.reduce_sum(params[:1] ** 2)


def test_parameter_only_penalty_is_actually_applied_when_sharded():
    """The regression test for the silent drop.

    A penalty on the parameters alone has to change the sharded loss by
    exactly the same amount it changes the single-device loss. Before this,
    the sharded loss was identical with and without the regularizer, because
    gnll_local had no penalty term in it -- which is invisible unless you
    compare against the unregularized value.
    """
    with tempfile.TemporaryDirectory() as tmp:
        filename = make_test_tensor(tmp)
        vals = {}
        for nd in (1, 2):
            for reg in (False, True):
                f = _make_fitter(filename, ndevices=nd)
                f.set_nobs(f.indata.data_obs)
                if reg:
                    f.regularizers = [_ParamOnlyPenalty()]
                    f.arm_regularizers()
                vals[(nd, reg)] = float(f.loss_val().numpy())

        d1 = vals[(1, True)] - vals[(1, False)]
        d2 = vals[(2, True)] - vals[(2, False)]
        assert d1 > 0.0, "the penalty must raise the single-device loss"
        assert d2 > 0.0, "the penalty must raise the sharded loss too (was 0)"
        assert np.isclose(
            d1, d2, rtol=1e-9, atol=1e-9
        ), f"penalty contributes {d2} sharded vs {d1} single-device"
        assert np.isclose(vals[(1, True)], vals[(2, True)], rtol=1e-9)


def test_parameter_only_penalty_reaches_the_gradient():
    """A penalty in the value but not the gradient would not steer the fit."""
    with tempfile.TemporaryDirectory() as tmp:
        filename = make_test_tensor(tmp)
        g = {}
        for reg in (False, True):
            f = _make_fitter(filename, ndevices=2)
            f.set_nobs(f.indata.data_obs)
            if reg:
                f.regularizers = [_ParamOnlyPenalty()]
                f.arm_regularizers()
            g[reg] = f.loss_val_grad()[1].numpy().copy()
        assert not np.allclose(g[True], g[False]), "penalty absent from gradient"


def test_regularizers_are_refused_rather_than_silently_dropped():
    """A sharded fit must not quietly minimise a different objective.

    The only global (non-shard) term is gnll_local, which sums the constraint
    and external-likelihood pieces and carries no penalty, so a regularizer
    that is accepted but not wired into it would be silently inert: the fit
    converges, writes a plausible result, and never enforces the bound. Its
    loss would even look *better* than a single-device fit's, a positive
    penalty term being absent -- which is why this asserts the shift is
    exactly the single-device shift rather than merely non-zero.

    The check cannot live in the constructor (self.regularizers is still empty
    there), which is why the docstring's "checked at construction" claim went
    unimplemented for so long.
    """

    class YieldDependent:
        needs_observables = True

        def set_expectations(self, *a, **k):
            raise AssertionError("must never be armed on the sharded path")

    with tempfile.TemporaryDirectory() as tmp:
        f = _make_fitter(make_test_tensor(tmp), ndevices=2)
        f.regularizers = [YieldDependent()]
        with pytest.raises(NotImplementedError, match="read the predicted yields"):
            f.arm_regularizers()
        # and with none configured, arming still works
        f.regularizers = []
        f.arm_regularizers()


@pytest.mark.parametrize(
    "method,flag",
    [
        ("global_impacts_parms", "--globalImpacts"),
        ("gaussian_global_impacts_parms", "--gaussianGlobalImpacts"),
        ("toyassign", "-t > 0"),
        # full_nll, not loss_val_valfull_grad_hess: the latter has no caller
        # in bin/, so testing it left the real --fullNll entry point uncovered
        ("full_nll", "--fullNll"),
    ],
)
def test_unsharded_postfit_steps_fail_before_the_fit_not_after(method, flag):
    """These run over all bins on one device, so they cannot work here.

    They are all reached only *after* the minimiser. Inherited unchanged they
    raise an allocation failure at the end of a multi-hour fit and take the
    result with it -- which is how reduced_nll was found, having discarded a
    19.9-hour run. Raising on call is the cheap version of that discovery.
    """
    with tempfile.TemporaryDirectory() as tmp:
        f = _make_fitter(make_test_tensor(tmp), ndevices=2)
        with pytest.raises(NotImplementedError, match=re.escape(flag)):
            getattr(f, method)()


@pytest.mark.parametrize("how", ["hessian", "gaussnewton"])
def test_reference_matrix_honours_precondition_from(how):
    """--preconditionFrom must mean the same thing sharded as unsharded.

    MultiDeviceFitter inherits _reference_matrix: the base implementation
    reaches the sharded machinery on its own, through loss_val_grad_hess and
    expected_yield, both of which the subclass replaces. An override that
    returned one of the two branches unconditionally would be invisible here
    without this comparison, since the fit itself proceeds normally either
    way.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = make_test_tensor(tmpdir)
        mats = []
        for ndevices in (1, 2):
            f = _make_fitter(fname, ndevices, preconditionFrom=how)
            f.defaultassign()
            f.set_nobs(f.indata.data_obs)
            mats.append(np.asarray(f._reference_matrix()))
        np.testing.assert_allclose(mats[1], mats[0], rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("mode", ["lite", "full"])
def test_bin_by_bin_stat_modes_match_single_device(mode):
    """Both BBB modes must run sharded and agree with the single-device loss.

    'full' exercises the branch of bbstat.profile_and_apply that reads
    indata.betavar, which the shard view has to expose even in lite mode
    because the read happens before the `and full` short-circuits.

    The Hessian is checked as well as the loss, because that is where the
    modes differ in kind rather than degree. 'full' profiles beta with a
    stateful Newton loop -- nbeta.assign_sub inside a tf.while_loop -- and
    _loss_val_grad_hessp_batch runs k directions of it inside a single
    tf.vectorized_map against that one shared variable. It is safe because the
    Newton state depends only on x, which is identical across the k
    directions, so the solves cannot disagree; comparing only loss_val leaves
    that unpinned, since the batching happens on the Hessian path alone.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = make_test_tensor(tmpdir)
        vals, hessians = [], []
        for ndevices in (1, 2):
            f = _make_fitter(
                fname, ndevices, noBinByBinStat=False, binByBinStatMode=mode
            )
            f.defaultassign()
            f.set_nobs(f.indata.data_obs)
            v = f.loss_val()
            vals.append(float(v[0] if isinstance(v, (tuple, list)) else v))
            _, _, hess = f.loss_val_grad_hess()
            hessians.append(np.asarray(hess))
        np.testing.assert_allclose(vals[1], vals[0], rtol=RTOL)
        assert hessians[1].shape == hessians[0].shape
        np.testing.assert_allclose(hessians[1], hessians[0], rtol=1e-9, atol=1e-11)
        # a batched vectorized_map that let the k directions interfere would
        # show up here first, since the interference is direction-dependent
        asym = np.abs(hessians[1] - hessians[1].T).max()
        assert asym < 1e-9, f"sharded Hessian is not symmetric: max|H - H^T| = {asym:g}"


def test_rebuild_frees_the_previous_shard_generation():
    """A rebuild must not hold two generations of shard tensors at once.

    Sampled at the moment the new shards are allocated, which is the only
    moment that matters for peak device memory: once _make_tf_functions has
    returned the old generation is unreachable either way, so checking there
    would pass vacuously.

    Both halves of the release matter -- dropping the attributes that still
    reference the old shards, and collecting the cycle that is left -- so
    the assertion fails if either is removed.
    """
    import weakref

    from rabbit.sharding import MultiDeviceFitter

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = make_test_tensor(tmpdir)
        f = _make_fitter(fname, 2, noBinByBinStat=False)
        f.defaultassign()
        f.set_nobs(f.indata.data_obs)

        # trace every graph, so each one really holds its captures
        f.loss_val()
        f.loss_val_grad()
        f.loss_val_grad_hessp(tf.constant(np.ones(f.x.shape[0])))
        f._profile_beta()

        refs = [weakref.ref(shard.logk) for shard in f.shards]
        alive = {}

        original = MultiDeviceFitter._build_shards

        def spy(self):
            # deliberately NO gc.collect() here: the production path has to do
            # it, and collecting in the probe would let this test pass with the
            # collect removed from _make_tf_functions
            alive["at_allocation"] = [r() is not None for r in refs]
            return original(self)

        MultiDeviceFitter._build_shards = spy
        try:
            f._make_tf_functions()
        finally:
            MultiDeviceFitter._build_shards = original

        assert alive["at_allocation"] == [False, False], alive["at_allocation"]
        # and the rebuild is still a no-op numerically
        np.testing.assert_allclose(float(f.loss_val()), 20.185670, rtol=1e-6)


@pytest.mark.parametrize("no_bbb", [False, True])
def test_impacts_refused_only_when_bin_by_bin_stat_needs_the_unsharded_hessian(no_bbb):
    """--doImpacts is refused sharded only with bin-by-bin stat enabled.

    With BBB on, impacts_parms takes a second Hessian at profile=False to
    split out the stat-only covariance, which the sharded loss does not
    provide. With BBB off that branch is skipped, so refusing unconditionally
    would take a working combination away.

    The refusal has to happen at the point of use rather than being inherited,
    because the inherited failure lands after the minimiser and the postfit
    Hessian have already run.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = make_test_tensor(tmpdir)
        f = _make_fitter(fname, 2, noBinByBinStat=no_bbb)
        f.defaultassign()
        f.set_nobs(f.indata.data_obs)
        _, _, hess = f.loss_val_grad_hess()

        if no_bbb:
            f.impacts_parms(hess)  # must not raise
        else:
            with pytest.raises(NotImplementedError, match="Impacts"):
                f.impacts_parms(hess)


def test_sharded_loss_refuses_unarmed_regularizers():
    """The sharded objective keeps the base class's armed check.

    An unarmed regularizer would otherwise contribute a penalty whose
    internals were never set, or none at all if the graphs were traced before
    it was attached -- the objective drifting with no error either way.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = make_test_tensor(tmpdir)
        f = _make_fitter(fname, 2)
        f.defaultassign()
        f.set_nobs(f.indata.data_obs)

        f.regularizers = [_ParamOnlyPenalty()]
        f._regularizers_armed = False
        f._make_tf_functions()
        with pytest.raises(RuntimeError, match="not armed"):
            f.loss_val()


def test_beta_edm_diagnostic_is_refused_when_sharded():
    """--diagnostics with bin-by-bin stat takes an [nbinsfull, nbinsfull]
    jacobian on one device, so it has to refuse rather than be inherited.

    Like the rest of the refusal list it would succeed on a model this size;
    refusing is about the sizes --nDevices exists for, and about failing
    before the minimiser rather than after it.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = make_test_tensor(tmpdir)
        f = _make_fitter(fname, 2, noBinByBinStat=False)
        f.defaultassign()
        f.set_nobs(f.indata.data_obs)
        with pytest.raises(NotImplementedError, match="beta-space EDM"):
            f.loss_val_grad_hess_beta()

        # single device keeps it
        f1 = _make_fitter(fname, 1, noBinByBinStat=False)
        f1.defaultassign()
        f1.set_nobs(f1.indata.data_obs)
        f1.loss_val_grad_hess_beta()


@pytest.mark.parametrize("n", [0, -1])
def test_pick_physical_gpus_rejects_degenerate_device_counts(n):
    """n < 1 desyncs GPU visibility from the fitter's own device count.

    n = 0 selects an empty GPU set, which the driver applies as a real
    selection and hides every GPU, while make_fitter normalizes 0 -> 1 and
    builds a single-device Fitter -- the run lands on the CPU with nothing
    said. n = -1 drops one GPU and leaves n_devices negative.
    """
    from rabbit.sharding import pick_physical_gpus

    with pytest.raises(ValueError, match="must be >= 1"):
        pick_physical_gpus(n)


@pytest.mark.parametrize("devices", [["-1"], ["0", "-2"]])
def test_explicit_devices_rejects_negative_indices(devices):
    """Negative --devices indices must not wrap to the tail.

    gpus[-1] is a valid Python index, so a negative value silently selects a
    GPU the user never named, while a positive out-of-range one raises.
    """
    from rabbit.sharding import pick_physical_gpus

    with pytest.raises(ValueError, match="must be >= 0"):
        pick_physical_gpus(len(devices), explicit=devices)


# Fitter methods rabbit_fit.py calls that are safe under sharding because they
# reach the likelihood only through primitives MultiDeviceFitter replaces
# (minimize / loss_val* / the HVP-assembled Hessian) or touch parameter-level
# state only. What the impacts helpers reach on their own behalf is checked
# separately, by the second test below.
_SHARDED_SAFE = {
    # re-minimise or evaluate through the sharded loss
    "minimize",
    "loss_val_grad",
    "loss_val_grad_hess",
    "asym_impacts_parms",
    # repeated minimize() calls; its one all-bins primitive, _dxdvars, is
    # host-pinned, and the --globalAsymImpactsLinearWarmstart combination that
    # makes it expensive is refused up front in rabbit_fit.py
    "global_asym_impacts_parms",
    "nonprofiled_impacts_parms",
    "contour_scan",
    "contour_scan2D",
    "nll_scan",
    "nll_scan2D",
    "edmval_cov_rows_hessfree",
    # parameter-level state only
    "defaultassign",
    "load_fitresult",
    "set_blinding_offsets",
    "prefit_covariance",
    "edmval_cov",
    # postfit verdict on the blinding width, reached as fitter.blinding.*:
    # reads the variance vector the driver already has, the POI offsets and
    # the model's declared scales. All [nparams]-sized, no bin axis anywhere,
    # so sharding cannot change it.
    "warn_if_weak",
}


def test_profiled_chi2_still_works_when_sharded():
    """--saveHists must survive sharding, not just the refusals.

    The suite asserted at length that unsupported paths refuse, and nothing
    asserted that the supported ones still run: a refusal added to _dxdvars
    passed all of it while breaking every sharded --saveHists, because
    _dndvars calls _dxdvars unconditionally and chi2(profile=True) goes
    through _dndvars.

    This checks that the path runs and agrees with single-device, not that it
    runs on the host: CI has no GPU, so dropping the tf.device("/CPU:0") is
    invisible here, as it is for the other pinned methods.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname = make_test_tensor(tmp)
        out = {}
        for nd in (1, 2):
            f = _make_fitter(fname, nd)
            f.minimize()
            _, grad, hess = f.loss_val_grad_hess()
            _, cov = f.edmval_cov(grad, hess)
            f.cov.assign(cov)
            out[nd] = np.asarray(f._dxdvars()[0])

        assert np.allclose(out[1], out[2], rtol=0, atol=1e-10), (
            "sharded dx/dx0 disagrees with single-device: "
            f"max|diff| = {np.abs(out[1] - out[2]).max():.3e}"
        )


# Methods MultiDeviceFitter keeps working by running them on the host rather
# than refusing them. Nothing they reach may refuse.
_HOST_PINNED = {
    "expected_yield",
    "expected_events",
    "expected_with_variance",
    "expected_variations",
    "chi2",
}


def test_no_refusal_is_reachable_from_a_path_that_is_kept_working():
    """A refusal on a shared primitive takes its supported callers down too.

    _dxdvars was first refused for the one entry point that needs it in bulk
    (--globalAsymImpactsLinearWarmstart) -- but it is a primitive, not an
    entry point, and _dndvars puts it under chi2 and expected_with_variance,
    both host-pinned on purpose. The refusal has to sit on the entry point,
    or up front in the driver, never on something a kept-working path calls.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sharding = (root / "rabbit" / "sharding.py").read_text()
    tree = ast.parse((root / "rabbit" / "fitter.py").read_text())

    refused = set()
    for node in ast.walk(ast.parse(sharding)):
        if isinstance(node, ast.FunctionDef) and "_unsharded" in ast.dump(node):
            refused.add(node.name)
    assert refused, "found no refusals to check -- the detector is broken"

    # self.<name>(...) called inside each Fitter method
    calls = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            calls[node.name] = {
                c.func.attr
                for c in ast.walk(node)
                if isinstance(c, ast.Call)
                and isinstance(c.func, ast.Attribute)
                and isinstance(c.func.value, ast.Name)
                and c.func.value.id == "self"
            }

    bad = []
    for entry in sorted(_HOST_PINNED):
        seen, stack = set(), [entry]
        while stack:
            cur = stack.pop()
            for callee in calls.get(cur, ()):
                if callee in seen:
                    continue
                seen.add(callee)
                stack.append(callee)
        for hit in sorted(seen & refused):
            bad.append(f"{entry} -> ... -> {hit}")

    assert not bad, (
        "MultiDeviceFitter refuses methods that a host-pinned path reaches, so "
        f"those paths now raise instead of running: {bad}. Move the refusal to "
        "the entry point, or to the up-front list in rabbit_fit.py."
    )


def test_refusal_messages_name_flags_the_fit_script_has():
    """A refusal that names the wrong flag sends the reader nowhere.

    _unsharded exists to tell the caller which option to drop, so the flag in
    it is the whole payload -- and it had drifted: global impacts pointed at
    --impactType, which add_impact_args defines for the print/plot scripts and
    rabbit_fit.py never sees, and gaussian global impacts at --doImpacts,
    which works sharded with --noBinByBinStat. CLI users are shielded by the
    up-front list in rabbit_fit.py; anyone driving MultiDeviceFitter directly
    is not.

    Asks the real parser rather than grepping for quoted strings, so a flag
    that exists only in a sibling script's parser does not count as known.
    """
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_rabbit_fit_for_flags", root / "bin" / "rabbit_fit.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    known = {opt for a in driver.make_parser()._actions for opt in a.option_strings}
    assert "--globalImpacts" in known, "parser introspection is broken"

    sharding = (root / "rabbit" / "sharding.py").read_text()
    named = set()
    for call in re.findall(r"_unsharded\(([^)]*)\)", sharding, re.S):
        named |= set(re.findall(r"--[a-zA-Z][a-zA-Z0-9]*", call))

    unknown = sorted(named - known)
    assert not unknown, (
        "refusal messages in rabbit/sharding.py name flags rabbit_fit.py does "
        f"not define: {unknown}. Name the option that actually reaches the "
        "refused path."
    )


@pytest.mark.parametrize(
    "argv,match",
    [
        (["--nDevices", "2"], "--nDevices > 1 is not supported"),
        (["--devices", "2"], "--devices is not supported"),
    ],
)
def test_rabbit_limit_rejects_device_flags_it_cannot_honour(argv, match, monkeypatch):
    """Both halves of the flag pair are inert here, so both must refuse.

    rabbit_limit.py builds a plain Fitter and never calls pick_physical_gpus,
    so --devices 2 would run on the default GPU -- silently colliding with
    whatever the user was trying to step around, which is the one thing the
    flag is for. --nDevices was already refused; --devices was not.
    """
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_rabbit_limit_for_guard", root / "bin" / "rabbit_limit.py"
    )
    limit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(limit)

    monkeypatch.setattr(sys, "argv", ["rabbit_limit.py", "nonexistent.hdf5"] + argv)
    with pytest.raises(Exception, match=re.escape(match)):
        limit.main()


# Reached from rabbit/impacts/*.py rather than from the driver, and safe:
# regexp bookkeeping on self.frozen_params, no tensor work.
_HELPER_SAFE = {"freeze_params", "defreeze_params"}


def test_every_fitter_method_reached_from_the_impacts_helpers_is_classified():
    """The same enumeration, one call level down.

    The test above walks driver -> Fitter. It cannot see driver -> helper
    module -> Fitter, and that gap hid _dxdvars: global_asym_impacts_parms is
    safe itself, but under --globalAsymImpactsLinearWarmstart it reaches a
    private Fitter method that runs _compute_loss over all bins and takes
    [npar, nbinsfull] jacobians. Pinning the helper layer down here means the
    safe list is checked rather than asserted.
    """
    from pathlib import Path

    from rabbit.fitter import Fitter

    root = Path(__file__).resolve().parents[1]
    sharding = (root / "rabbit" / "sharding.py").read_text()
    helpers = "\n".join(
        p.read_text() for p in sorted((root / "rabbit" / "impacts").glob("*.py"))
    )

    reached = {
        m
        for m in re.findall(r"\bfitter\.([a-z_][a-zA-Z_0-9]*)", helpers)
        if callable(getattr(Fitter, m, None))
    }
    handled = set(re.findall(r"^\s+def ([a-z_][a-zA-Z_0-9]*)", sharding, re.M))
    handled |= set(re.findall(r"self\.([a-z_][a-zA-Z_0-9]*)\s*=", sharding))

    unclassified = sorted(reached - handled - _SHARDED_SAFE - _HELPER_SAFE)
    assert not unclassified, (
        "fitter methods the impacts helpers reach that MultiDeviceFitter "
        f"neither handles nor declares safe: {unclassified}. Override, pin or "
        "refuse them in rabbit/sharding.py, or add them to _HELPER_SAFE with "
        "the reason they survive sharding."
    )


def test_every_driver_called_fitter_method_is_classified_for_sharding():
    """Each fitter method the driver calls must be handled or known safe.

    Three separate review rounds each found one more all-bins path reachable
    from rabbit_fit.py under --nDevices > 1 -- impacts_parms, then
    loss_val_grad_hess_beta, then the L-curve flags. They are the same defect
    found three times because nothing enumerates the surface.

    This closes the class: adding a fitter call to the driver, or a method to
    Fitter that the driver reaches, now fails here until it is either handled
    in MultiDeviceFitter (overridden, host-pinned or refused) or added to
    _SHARDED_SAFE with a reason.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    driver = (root / "bin" / "rabbit_fit.py").read_text()
    sharding = (root / "rabbit" / "sharding.py").read_text()

    # `fitter.` in the driver is both the module and the instance, so keep
    # only names that are genuinely Fitter methods (drops fitter.make_fitter)
    from rabbit.blinding import Blinding
    from rabbit.fitter import Fitter

    called = {
        m
        for m in re.findall(r"\bi?fitter\.([a-z_][a-zA-Z_0-9]*)\s*\(", driver)
        if callable(getattr(Fitter, m, None))
    }
    # Calls the driver makes on a Fitter COLLABORATOR reach the same fit state
    # and have to be classified too: without this, moving a method off the
    # Fitter -- as blinding's weak-smearing verdict was -- would drop it from
    # the enumeration rather than force a decision about it.
    called |= {
        m
        for m in re.findall(r"\bi?fitter\.blinding\.([a-z_][a-zA-Z_0-9]*)\s*\(", driver)
        if callable(getattr(Blinding, m, None))
    }
    # MultiDeviceFitter handles a name by defining it, or by rebinding it as an
    # instance attribute in _make_tf_functions
    handled = set(re.findall(r"^\s+def ([a-z_][a-zA-Z_0-9]*)", sharding, re.M))
    handled |= set(re.findall(r"self\.([a-z_][a-zA-Z_0-9]*)\s*=", sharding))

    unclassified = sorted(called - handled - _SHARDED_SAFE)
    assert not unclassified, (
        "fitter methods called by rabbit_fit.py that MultiDeviceFitter neither "
        f"handles nor declares safe: {unclassified}. Either override/pin/refuse "
        "them in rabbit/sharding.py, or add them to _SHARDED_SAFE with the "
        "reason they are safe when the bins are sharded."
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))


def test_lcurve_honours_the_needs_observables_contract():
    """The L-curve must build and pass yields on the same rule the fit does.

    Regularizer.compute_nll_penalty documents that ``observables`` is None when
    needs_observables is False, so a penalty that declared it does not read them
    raises rather than quietly consuming a vector. Fitter._compute_nll and
    arm_regularizers honour that; rabbit/regularization/lcurve.py was written
    before the contract and handed every penalty the yields unconditionally.

    That mattered beyond the raise: with the scan passing observables the fit
    path does not, the scan objective and the fit objective stop being the same
    function, and the tau it selects optimises something the fit never evaluates.

    Single-device only -- multi-device refuses --lCurveScan / --lCurveOptimize
    up front -- but it is the same contract, and nothing else in the suite
    executes lcurve.py at all.
    """
    from rabbit.regularization.lcurve import _compute_curvature

    with tempfile.TemporaryDirectory() as tmpdir:
        f = _make_fitter(make_test_tensor(tmpdir), 1)
        f.set_nobs(f.indata.data_obs)
        penalty = _ParamOnlyPenalty()
        # Mixed on purpose. The contract is per regularizer, so a list holding
        # only parameter-only penalties cannot detect a global any() -- both
        # forms then pass None to everything. With a yield-reading penalty
        # alongside, any() is True and only the per-regularizer form still
        # hands _ParamOnlyPenalty None.
        f.regularizers = [_YieldReadingPenalty(), penalty]
        f.arm_regularizers()
        f.tau.assign(0.5)

        # each penalty asserts, on every call, that it got what it declared, so
        # a violation surfaces here rather than as a wrong curvature
        curvature = _compute_curvature(f)

        assert penalty.armed >= 1, "regularizer never armed; test is vacuous"
        curvature = np.asarray(curvature)
        assert np.all(np.isfinite(curvature)), f"curvature is not finite: {curvature}"
