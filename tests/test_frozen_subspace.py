"""`--freezeParameters` must hold a parameter fixed, not merely stop-gradient it.

Freezing is implemented with `tf.stop_gradient`: the parameter's gradient and
its Hessian row and column are zero, but the likelihood is still EVALUATED at
whatever the vector holds. While the minimiser was handed the FULL vector, each
frozen parameter was therefore a direction the model is exactly flat along --
and a trust-region method does not leave a flat direction alone. `trust-exact`'s
hard case walks to the trust-region boundary along the smallest-curvature
eigenvector, which after `hess_for_minimizer`'s unit diagonal is precisely the
frozen block; `trust-krylov` and `trust-ncg` see zero curvature there and do the
same. Observed on real fits: the four frozen resolution knobs of the Z-mass
campaign came back displaced by up to 0.126 from the value they were frozen at,
always along (1,1,1,1) or (1,-1,1,-1), i.e. an arbitrary vector of a degenerate
block.

The fix removes the frozen coordinates from the vector the minimiser sees. It is
exact rather than a regularisation -- the frozen gradient components are
identically zero, so no descent direction is lost -- and these tests pin both
halves of it: the minimiser's own vector has the floating dimension, and what
comes back is unchanged on the frozen entries and identical on the floating ones
to a fit where nothing was frozen at all (when the frozen values are the ones
that fit would have converged to anyway).
"""

import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.optimize

from rabbit import fitter, inputdata
from rabbit.callbacks import FitterCallback
from rabbit.param_models.helpers import load_model

from .test_sparse_fit import make_options, make_test_tensor


def _setup(filename, **kw):
    indata_obj = inputdata.FitInputData(filename)
    param_model = load_model("Mu", indata_obj)
    f = fitter.Fitter(indata_obj, param_model, make_options(**kw))
    f.set_nobs(indata_obj.data_obs)
    return f


def _frozen_names(f, n=2):
    """A couple of nuisance parameters to freeze, by name."""
    return [str(s) for s in np.asarray(f.parms).astype(str)[-n:]]


@pytest.mark.parametrize("method", ["trust-exact", "trust-krylov"])
def test_the_minimiser_vector_has_the_floating_dimension(method):
    """The load-bearing test: scipy must never be handed a flat direction.

    Spying on the vector rather than on the answer is deliberate. Whether a
    given toy model happens to trigger the hard case turns on details far below
    the scale of the fit, so keying the regression on a displacement would make
    it flaky; the dimension of the vector the minimiser optimises over is exact.
    """
    with tempfile.TemporaryDirectory() as tmp:
        filename = make_test_tensor(tmp)
        f = _setup(filename, minimizerMethod=method)
        frozen = _frozen_names(f, 2)
        f = _setup(filename, minimizerMethod=method, freezeParameters=frozen)
        npar = int(f.x.shape[0])

        seen = []
        real = scipy.optimize.minimize

        def spy(fun, x0, **kwargs):
            seen.append(np.asarray(x0).size)
            return real(fun, x0, **kwargs)

        scipy.optimize.minimize = spy
        try:
            f.minimize()
        finally:
            scipy.optimize.minimize = real

        assert seen, "the scipy minimiser was never called"
        assert all(n == npar - len(frozen) for n in seen), (
            f"the minimiser was handed {seen} coordinates; with {len(frozen)} "
            f"of {npar} frozen it must see {npar - len(frozen)}"
        )


@pytest.mark.parametrize(
    "method", ["trust-exact", "trust-krylov", "tf-trust-exact", "tf-trust-krylov"]
)
def test_a_frozen_parameter_comes_back_exactly_where_it_was_frozen(method):
    """Covers the NATIVE minimisers too, which take a different set of closures.

    Those hand back tf tensors rather than numpy, so the reduction has to gather
    BOTH axes of the Hessian there; gathering one axis twice is a mistake that
    the scipy tests cannot see.
    """
    with tempfile.TemporaryDirectory() as tmp:
        filename = make_test_tensor(tmp)
        f = _setup(filename, minimizerMethod=method)
        frozen = _frozen_names(f, 2)
        f = _setup(filename, minimizerMethod=method, freezeParameters=frozen)
        names = list(np.asarray(f.parms).astype(str))
        idx = [names.index(n) for n in frozen]
        before = f.x.numpy()[idx].copy()
        f.minimize()
        after = f.x.numpy()[idx]
        np.testing.assert_array_equal(after, before)


@pytest.mark.parametrize("method", ["trust-exact", "trust-krylov", "tf-trust-exact"])
def test_freezing_a_parameter_at_its_own_optimum_changes_nothing_else(method):
    """Freezing is not supposed to be a different fit.

    Fit everything, then re-fit with two parameters frozen AT THE VALUES the
    free fit found. The remaining parameters are at the same minimum of the same
    function, so they must come back to it.

    This is what tests the ARITHMETIC of the reduction rather than only its
    bookkeeping: a reduced gradient or Hessian that is wrong still leaves the
    frozen entries untouched -- they are not in the vector -- and so passes the
    other tests, while not arriving at the minimum. In particular the native
    closures hand back a Hessian whose BOTH axes have to be gathered, and
    gathering one axis twice happens to have a valid shape when the frozen
    parameters are the last ones, which is exactly the mistake this catches.
    """
    with tempfile.TemporaryDirectory() as tmp:
        filename = make_test_tensor(tmp)
        f = _setup(filename, minimizerMethod=method)
        f.minimize()
        xfree = f.x.numpy().copy()
        names = list(np.asarray(f.parms).astype(str))
        frozen = names[-2:]
        idx = [names.index(n) for n in frozen]

        g = _setup(filename, minimizerMethod=method, freezeParameters=frozen)
        x0 = g.x.numpy().copy()
        x0[idx] = xfree[idx]
        g.x.assign(x0)
        g.minimize()
        xfrozen = g.x.numpy()

        np.testing.assert_array_equal(xfrozen[idx], xfree[idx])
        np.testing.assert_allclose(xfrozen, xfree, atol=1e-5, rtol=1e-5)


def test_the_callback_reports_the_full_vector_under_reduction():
    """`cb.xval` feeds the snapshot and the rollback, so it must be full-length."""
    expand = lambda u: np.array([u[0], 7.0, u[1]])  # noqa: E731
    cb = FitterCallback(np.zeros(3), early_stopping=-1, expand=expand)
    cb(SimpleNamespace(fun=1.0, x=np.array([1.0, 2.0])))
    np.testing.assert_array_equal(cb.xval, [1.0, 7.0, 2.0])


def test_the_callback_is_unchanged_without_a_reduction():
    cb = FitterCallback(np.zeros(3), early_stopping=-1)
    cb(SimpleNamespace(fun=1.0, x=np.array([1.0, 2.0, 3.0])))
    np.testing.assert_array_equal(cb.xval, [1.0, 2.0, 3.0])
