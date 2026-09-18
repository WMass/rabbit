"""Test that an external likelihood term evaluates to a *proper* Gaussian.

An external term is stored expanded,

    -log L_ext = g^T x + 0.5 x^T H x   (+ const [+ lognorm])

so for a Gaussian prior N(mu, C) with H = C^-1 and g = -H mu it is short by
two additive scalars relative to the real thing:

    const   = 0.5 mu^T H mu                     -- centers the quadratic, i.e.
                                                   makes the term 0 at x = mu
    lognorm = 0.5 (k log 2pi - log det H)       -- normalizes the density

Without ``const`` the term reads -0.5 mu^T H mu at the prior mean, so every
absolute NLL on a card carrying an external term is offset by a card-dependent
amount -- invisible in any NLL *difference* taken within one fit (which is why
fits were always right) but wrong in ``nllvalreduced``, ``nllvalfull`` and the
saturated chi2 built as ``2 * nllvalreduced``.

The decisive check here compares two cards that are identical except for the
presence of the term, and asserts that the loss difference equals the analytic
Gaussian evaluated in numpy -- at several points, not just the minimum. That
is what pins both scalars; asserting only the postfit parameter value (as
test_external_term.py does) cannot see a constant at all.
"""

import os
import tempfile

import numpy as np
import pytest

from rabbit import external_likelihood
from tests.test_external_term import (
    build_writer,
    loss_grad_hess_at,
    make_grad_hist,
    make_hess_hist,
    make_hess_sparsehist,
)

PARAM = "shape"  # the single systematic in build_writer's model
TOL = 1e-9


def write(tmpdir, fname, **kwargs):
    """Build the standard test card, optionally with an external term."""
    writer = build_writer()  # no term; we add it ourselves to control kwargs
    if kwargs:
        writer.add_external_likelihood_term(**kwargs)
    path = os.path.join(tmpdir, fname)
    writer.write(outfolder=tmpdir, outfilename=fname)
    return path


def loss_at(filename, parms_ref, value, full=False):
    """Loss of ``filename`` with PARAM set to ``value``, everything else at 0."""
    import tensorflow as tf

    from rabbit import fitter, inputdata
    from rabbit.param_models.helpers import load_model
    from tests.test_external_term import make_options

    indata_obj = inputdata.FitInputData(filename)
    f = fitter.Fitter(indata_obj, load_model("Mu", indata_obj), make_options())
    f.set_nobs(f.expected_yield())
    x = f.x.numpy().copy()
    x[int(np.where(f.parms.astype(str) == PARAM)[0][0])] = value
    f.x.assign(tf.constant(x, dtype=f.x.dtype))
    return float((f.full_nll() if full else f.reduced_nll()).numpy())


def test_gaussian_scalars_closed_form():
    """0.5 g^T H^-1 g must equal 0.5 mu^T H mu, and the degenerate cases must
    not raise."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 3))
    H = A @ A.T + 3 * np.eye(3)  # symmetric positive definite
    mu = np.array([0.7, -1.3, 0.2])
    g = -H @ mu

    const, lognorm = external_likelihood.gaussian_scalars(g, H, "t")
    assert abs(const - 0.5 * mu @ H @ mu) < 1e-12, "const != 0.5 mu^T H mu"
    _, logdet = np.linalg.slogdet(H)
    assert abs(lognorm - 0.5 * (3 * np.log(2 * np.pi) - logdet)) < 1e-12

    # no hessian -> a linear tilt, no minimum, nothing to center
    assert external_likelihood.gaussian_scalars(g, None, "t") == (0.0, 0.0)
    # no gradient -> already centered at the origin
    c0, ln0 = external_likelihood.gaussian_scalars(None, H, "t")
    assert c0 == 0.0 and abs(ln0 - lognorm) < 1e-12
    # singular H -> pseudo-inverse path, must not raise
    Hs = np.diag([2.0, 0.0])
    external_likelihood.gaussian_scalars(np.array([1.0, 0.0]), Hs, "t")
    # indefinite H -> centering defined, normalization is not
    _, ln_bad = external_likelihood.gaussian_scalars(
        np.array([1.0, 0.0]), np.diag([2.0, -1.0]), "t"
    )
    assert ln_bad == 0.0, "indefinite H must not get a log-normalization"
    print("  test_gaussian_scalars_closed_form OK")


@pytest.mark.parametrize("sparse", [False, True])
def test_term_equals_analytic_gaussian(sparse):
    """loss(with term) - loss(without term) == the analytic Gaussian.

    Parametrized rather than defaulted: pytest drops arguments that have a
    default from its fixture lookup, so ``(sparse=False)`` collected as ONE
    test and the sparse external-hessian path -- the case with ``const=`` and
    ``lognorm=``, which this file exists to pin and test_external_term's
    SparseHist config does not cover -- ran nowhere once main() stopped being
    the CI entry point.
    """
    mu_val, sigma = 0.6, 0.25
    H = np.array([[1.0 / sigma**2]])
    g = -H @ np.array([mu_val])
    make_hess = make_hess_sparsehist if sparse else make_hess_hist

    with tempfile.TemporaryDirectory() as tmp:
        plain = write(tmp, "plain.hdf5")
        withterm = write(
            tmp,
            "ext.hdf5",
            grad=make_grad_hist(g, [PARAM]),
            hess=make_hess(H, [PARAM]),
            # a sparse H cannot be auto-centered at load time, so the writer
            # must be told the mean -- exercised properly in the mean= test.
            **({"const": float(0.5 * mu_val**2 / sigma**2)} if sparse else {}),
            **(
                {"lognorm": float(0.5 * (np.log(2 * np.pi) + 2 * np.log(sigma)))}
                if sparse
                else {}
            ),
        )

        for v in (0.0, mu_val, mu_val + sigma, -0.4):
            expect = 0.5 * (v - mu_val) ** 2 / sigma**2
            got = loss_at(withterm, None, v) - loss_at(plain, None, v)
            assert abs(got - expect) < TOL, (
                f"{'sparse' if sparse else 'dense'} reduced NLL at x={v}: "
                f"term contributed {got}, expected {expect}"
            )

        # at the mean the term must contribute exactly zero
        at_mu = loss_at(withterm, None, mu_val) - loss_at(plain, None, mu_val)
        assert abs(at_mu) < TOL, f"term is {at_mu} at x=mu, must be 0"

        # full NLL additionally carries the Gaussian normalization, which is
        # the same formula _compute_lc uses per constrained parameter
        expect_full = 0.5 * np.log(2 * np.pi) + np.log(sigma)
        got_full = loss_at(withterm, None, mu_val, full=True) - loss_at(
            plain, None, mu_val, full=True
        )
        assert (
            abs(got_full - expect_full) < TOL
        ), f"full NLL log-normalization: got {got_full}, expected {expect_full}"
    print(f"  test_term_equals_analytic_gaussian(sparse={sparse}) OK")


def test_mean_kwarg_matches_explicit_grad():
    """hess+mean must produce exactly the same term as the hand-built grad."""
    mu = np.array([0.6, -0.2])
    C = np.array([[0.09, 0.02], [0.02, 0.04]])
    H = np.linalg.inv(C)
    params = [PARAM]  # only one real param exists; use a 1D slice for the fit
    mu1, H1 = mu[:1], H[:1, :1]

    with tempfile.TemporaryDirectory() as tmp:
        by_grad = write(
            tmp,
            "bygrad.hdf5",
            grad=make_grad_hist(-H1 @ mu1, params),
            hess=make_hess_hist(H1, params),
        )
        by_mean = write(
            tmp,
            "bymean.hdf5",
            hess=make_hess_hist(H1, params),
            mean=make_grad_hist(mu1, params),
        )
        for v in (0.0, 0.5, -0.3):
            a, b = loss_at(by_grad, None, v), loss_at(by_mean, None, v)
            assert abs(a - b) < TOL, f"mean= and grad= disagree at x={v}: {a} vs {b}"
    print("  test_mean_kwarg_matches_explicit_grad OK")


def test_backward_compatible_when_scalars_absent():
    """A card whose term group has no const/lognorm must still load, and a
    dense Hessian must be centered automatically at load time."""
    mu_val, sigma = 0.6, 0.25
    H = np.array([[1.0 / sigma**2]])

    with tempfile.TemporaryDirectory() as tmp:
        path = write(
            tmp,
            "nolegacy.hdf5",
            grad=make_grad_hist(-H @ np.array([mu_val]), [PARAM]),
            hess=make_hess_hist(H, [PARAM]),
        )
        import h5py

        with h5py.File(path, "r") as fh:
            keys = set(fh["external_terms"]["ext0"].keys())
        # nothing was stored: the derivation happens at load
        assert "const" not in keys, f"const should not be written here, got {keys}"

        terms = None
        with h5py.File(path, "r") as fh:
            terms = external_likelihood.read_external_terms_from_h5(
                fh.get("external_terms")
            )
        assert terms[0]["const"] is None, "absent const must read back as None"

        built = external_likelihood.build_tf_external_terms(
            terms, np.array([PARAM]), __import__("tensorflow").float64
        )
        expect = 0.5 * mu_val**2 / sigma**2
        assert (
            abs(built[0]["const"] - expect) < 1e-10
        ), f"load-time derivation gave {built[0]['const']}, expected {expect}"
    print("  test_backward_compatible_when_scalars_absent OK")


def test_scalars_do_not_move_the_fit():
    """The scalars are constants, so they must not touch any derivative.

    Asserted as a difference against the term-free card, which isolates the
    term's own contribution from the data and native-constraint curvature
    that both cards share. At the default start (x = 0) the term contributes
    gradient ``g + H x = -H mu`` and Hessian ``H`` exactly.
    """
    mu_val, sigma = 0.6, 0.25
    H = np.array([[1.0 / sigma**2]])
    with tempfile.TemporaryDirectory() as tmp:
        plain = write(tmp, "plain.hdf5")
        withterm = write(
            tmp,
            "ext.hdf5",
            grad=make_grad_hist(-H @ np.array([mu_val]), [PARAM]),
            hess=make_hess_hist(H, [PARAM]),
        )
        parms_a, _, grad_a, hess_a = loss_grad_hess_at(plain)
        parms_b, _, grad_b, hess_b = loss_grad_hess_at(withterm)
        i = int(np.where(parms_b == PARAM)[0][0])
        j = int(np.where(parms_a == PARAM)[0][0])

        dgrad = grad_b[i] - grad_a[j]
        dhess = hess_b[i, i] - hess_a[j, j]
        assert (
            abs(dgrad - (-mu_val / sigma**2)) < 1e-8
        ), f"term changed the gradient by {dgrad}, expected {-mu_val / sigma**2}"
        assert (
            abs(dhess - 1.0 / sigma**2) < 1e-8
        ), f"term changed the hessian by {dhess}, expected {1.0 / sigma**2}"
    print("  test_scalars_do_not_move_the_fit OK")


def main():
    test_gaussian_scalars_closed_form()
    test_term_equals_analytic_gaussian(sparse=False)
    test_term_equals_analytic_gaussian(sparse=True)
    test_mean_kwarg_matches_explicit_grad()
    test_backward_compatible_when_scalars_absent()
    test_scalars_do_not_move_the_fit()
    print("all external-nll-constant tests passed")


if __name__ == "__main__":
    main()
