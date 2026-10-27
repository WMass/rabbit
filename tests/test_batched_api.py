"""The batched loss / gradient / HVP / Hessian API (Fitter.loss_val_batched and
friends): B independent likelihoods at the columns of X [npar, B] equal B
evaluations of the single API (loss_val_grad_hessp, loss_val_grad_hess) at the
same points, with the same per-element inputs, and leave the fitter's own state
unchanged."""

import copy
import tempfile

import numpy as np
import pytest
import tensorflow as tf

from .test_batched_nll import NBATCH, _fitter, _points

RTOL = 1e-10

# a cross-section of test_batched_nll.CASES: both tensor layouts, both
# interpolations, the bin-by-bin stat modes, chi2, frozen, blinded, external
CASES = [
    "dense",
    "sparse",
    "dense_asymmetric",
    "sparse_masked",
    "bbb_full",
    "bbb_dynamic",
    # entries without MC-stat variance: sqrt(varbeta) has an infinite
    # derivative there, NaN Hessians once differentiated in the batch
    "bbb_normal_additive",
    "bbb_full_additive",
    "chisq",
    "frozen",
    "blinded",
    "external_dense",
]


def _close(a, b):
    a, b = np.asarray(a), np.asarray(b)
    np.testing.assert_allclose(a, b, rtol=RTOL, atol=RTOL * np.abs(b).max())


def _directions(f, seed=2):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((f.x.shape[0], NBATCH))


def _single(f, X, fn):
    """fn() at each column of X, with self.x restored afterwards."""
    xsave = f.x.numpy()
    try:
        out = []
        for b in range(X.shape[-1]):
            f.x.assign(X[:, b])
            out.append(fn(b))
        return out
    finally:
        f.x.assign(xsave)


@pytest.mark.parametrize("hvp_method", ["revrev", "fwdrev"])
@pytest.mark.parametrize("case", CASES)
def test_batched_val_grad_hessp_equal_the_single_api(case, hvp_method):
    with tempfile.TemporaryDirectory() as tmp:
        f = _fitter(tmp, case)
        f.hvp_method = hvp_method
        X, P = _points(f), _directions(f)
        x = f.x.numpy()
        val, grad, hessp = f.loss_val_grad_hessp_batched(X, P)
        np.testing.assert_array_equal(f.x.numpy(), x)
        assert val.shape == (NBATCH,)
        assert grad.shape == hessp.shape == X.shape
        single = _single(
            f,
            X,
            lambda b: f.loss_val_grad_hessp_revrev(tf.constant(P[:, b])),
        )
        for k, got in enumerate((val, grad, hessp)):
            _close(np.stack([np.asarray(s[k]) for s in single], axis=-1), got)
        v, g = f.loss_val_grad_batched(X)
        _close(val, v)
        _close(grad, g)
        _close(val, f.loss_val_batched(X))


@pytest.mark.parametrize("from_hvp", [False, True], ids=["jacobian", "hvp"])
@pytest.mark.parametrize(
    "case", ["dense", "sparse", "bbb_full", "bbb_full_additive", "frozen"]
)
def test_batched_hessians_equal_the_single_api(case, from_hvp):
    with tempfile.TemporaryDirectory() as tmp:
        f = _fitter(tmp, case)
        f.hessian_from_hvp = from_hvp
        f.hvp_batch = 3  # several basis batches
        X = _points(f)
        val, grad, hess = f.loss_val_grad_hess_batched(X)
        n = f.x.shape[0]
        assert hess.shape == (n, n, NBATCH)
        f.hessian_from_hvp = False
        single = _single(f, X, lambda b: f.loss_val_grad_hess())
        for k, got in enumerate((val, grad, hess)):
            _close(np.stack([np.asarray(s[k]) for s in single], axis=-1), got)


# not gamma in full mode: a toy can draw a zero constraint centre, where that
# likelihood is NaN (on main too, see test_batched_nll)
@pytest.mark.parametrize("case", ["dense", "sparse", "bbb_full_additive", "chisq"])
def test_batched_toys_equal_the_single_api(case):
    """B frequentist toys (observations, x0 and β0 per element) in one call:
    loss, gradient and HVP of each equal those of the fit to it."""
    with tempfile.TemporaryDirectory() as tmp:
        f = _fitter(tmp, case)
        X, P = _points(f), _directions(f)
        state0 = (f.nobs.numpy(), f.x0.numpy(), f.bbstat.beta0.numpy())
        toys = []
        tf.random.set_seed(5)
        for _ in range(NBATCH):
            f.toyassign(
                syst_randomize="frequentist",
                data_randomize="normal" if f.chisqFit else "poisson",
            )
            nobs = f.nobs.numpy()
            if f.chisqFit:
                nobs = np.maximum(nobs, 1.0)
            toys.append((nobs, f.x0.numpy(), f.bbstat.beta0.numpy()))
        nobs, x0, beta0 = (np.stack(t, axis=-1) for t in zip(*toys))

        def set_state(n, c, b):
            f.set_nobs(tf.constant(n))
            f.x0.assign(c)
            f.bbstat.set_beta0(tf.constant(b))

        set_state(*state0)
        got = f.loss_val_grad_hessp_batched(X, P, data=nobs, x0=x0, beta0=beta0)
        np.testing.assert_array_equal(f.nobs.numpy(), state0[0])
        assert all(np.isfinite(t).all() for t in got)

        def one(b):
            set_state(nobs[:, b], x0[:, b], beta0[..., b])
            return f.loss_val_grad_hessp_revrev(tf.constant(P[:, b]))

        try:
            single = _single(f, X, one)
        finally:
            set_state(*state0)
        for k in range(3):
            _close(np.stack([np.asarray(s[k]) for s in single], axis=-1), got[k])


def test_batch_widths_do_not_retrace_each_time():
    with tempfile.TemporaryDirectory() as tmp:
        f = _fitter(tmp, "dense")
        f.hessian_from_hvp = True
        rng = np.random.default_rng(4)
        n = f.x.shape[0]
        for B in (1, 2, 3, 5, 8):
            X = f.x.numpy()[:, None] + 0.1 * rng.standard_normal((n, B))
            val, grad = f.loss_val_grad_batched(X)
            assert val.shape == (B,) and grad.shape == X.shape
            _close(val, f.loss_val_batched(X))
            _, _, hessp = f.loss_val_grad_hessp_batched(X, rng.standard_normal((n, B)))
            assert hessp.shape == X.shape
            assert f.loss_val_grad_hess_batched(X)[2].shape == (n, n, B)
        for name in ("val", "val_grad", f"val_grad_hessp_{f._batched_hvp_method()}"):
            assert f._batched_fn(name).experimental_get_tracing_count() <= 2


def test_unprofiled_batch_equals_the_single_api():
    with tempfile.TemporaryDirectory() as tmp:
        f = _fitter(tmp, "bbb_full")
        X = _points(f)
        val, grad = f.loss_val_grad_batched(X, profile=False)

        def one(b):
            with tf.GradientTape() as t:
                v = f._compute_loss(profile=False)
            return v, t.gradient(v, f.x)

        single = _single(f, X, one)
        for k, got in enumerate((val, grad)):
            _close(np.stack([np.asarray(s[k]) for s in single], axis=-1), got)
        assert not np.allclose(val, f.loss_val_batched(X))


def test_chisq_observations_with_variances():
    """data as (nobs, variances), one column per element."""
    with tempfile.TemporaryDirectory() as tmp:
        f = _fitter(tmp, "chisq")
        X, P = _points(f), _directions(f)
        nobs0 = f.nobs.numpy()
        rng = np.random.default_rng(6)
        nobs = nobs0[:, None] * rng.uniform(0.8, 1.2, (nobs0.size, NBATCH))
        var = nobs * rng.uniform(0.5, 2.0, nobs.shape)
        got = f.loss_val_grad_hessp_batched(X, P, data=(nobs, var))

        def one(b):
            f.set_nobs(tf.constant(nobs[:, b]), variances=tf.constant(var[:, b]))
            return f.loss_val_grad_hessp_revrev(tf.constant(P[:, b]))

        try:
            single = _single(f, X, one)
        finally:
            f.set_nobs(tf.constant(nobs0))
        for k in range(3):
            _close(np.stack([np.asarray(s[k]) for s in single], axis=-1), got[k])


@pytest.mark.parametrize("own_only", ["mcstat", "regularizers"])
def test_regularizers_and_mcstat_terms_are_refused(own_only, monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        f = _fitter(tmp, "dense")
        X = _points(f)
        if own_only == "mcstat":
            monkeypatch.setattr(f, "_mcstat_terms_in_nll", lambda: True)
        else:
            monkeypatch.setattr(f, "regularizers", [object()])
        with pytest.raises(NotImplementedError):
            f.loss_val_grad_batched(X)


def test_multi_device_fitters_refuse_the_batched_api():
    from rabbit.sharding import MultiDeviceFitter

    stub = MultiDeviceFitter.__new__(MultiDeviceFitter)
    with pytest.raises(NotImplementedError, match="multi-device"):
        stub.loss_val_batched(np.zeros((1, 2)))


def test_hessians_without_parameters_keep_the_batch_axis():
    with tempfile.TemporaryDirectory() as tmp:
        f = _fitter(tmp, "dense")
        f.hessian_from_hvp = True
        X = np.zeros((0, 3))
        # the loss itself needs the parameters; only the assembly is tested
        f._batched_tf_fns["val_grad"] = lambda *a: (tf.zeros([3], tf.float64), X)
        f.x = tf.Variable(tf.zeros([0], tf.float64))
        assert f.loss_val_grad_hess_batched(X)[2].shape == (0, 0, 3)


def test_deepcopy_after_use():
    with tempfile.TemporaryDirectory() as tmp:
        f = _fitter(tmp, "dense")
        X = _points(f)
        val = f.loss_val_batched(X)
        g = copy.deepcopy(f)
        assert g._batched_tf_fns == {}
        _close(val, g.loss_val_batched(X))
