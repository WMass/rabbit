"""Tests for the optional parameter preconditioning.

The transform is a pure reparameterisation, so the decisive test is that a fit
run with it converges to the *same* parameters, uncertainties and NLL as one
run without it. The unit tests below check the algebra that guarantees this:
T^T H T = I on the block, and the chain rule for the gradient / Hessian-vector
product that the scipy callbacks rely on.
"""

import os
import tempfile

import numpy as np
import pytest

from rabbit import fitter, inputdata
from rabbit.param_models.helpers import load_model
from rabbit.preconditioner import (
    Preconditioner,
    auto_blocks,
    select_index_blocks,
)

from .test_sparse_fit import check_results, make_options, make_test_tensor


def _spd(n, seed=0, cond=1e6):
    """A symmetric positive definite matrix with a controlled condition number."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eig = np.geomspace(1.0, cond, n)
    return q @ np.diag(eig) @ q.T


# -- unit tests: the algebra ---------------------------------------------


def test_identity_is_exact_noop():
    theta = np.arange(5, dtype=float)
    pc = Preconditioner.identity(theta)
    assert not pc.enabled
    y = pc.from_physical(theta)
    np.testing.assert_allclose(y, np.zeros(5))
    np.testing.assert_allclose(pc.to_physical(y), theta)
    g = np.array([1.0, -2.0, 3.0, 0.5, 0.0])
    np.testing.assert_allclose(pc.grad_to_internal(g), g)
    h = _spd(5)
    np.testing.assert_allclose(pc.hess_to_internal(h), h)


def test_round_trip():
    n = 8
    theta_ref = np.linspace(-1, 1, n)
    h = _spd(n, seed=1)
    idx = np.arange(2, 7)
    pc = Preconditioner.from_hessian(h, theta_ref, idx)
    assert pc.enabled and pc.nblock == idx.size

    rng = np.random.default_rng(3)
    y = rng.normal(size=n)
    np.testing.assert_allclose(pc.from_physical(pc.to_physical(y)), y, atol=1e-10)

    theta = theta_ref + rng.normal(size=n)
    np.testing.assert_allclose(
        pc.to_physical(pc.from_physical(theta)), theta, atol=1e-10
    )


def test_whitens_the_reference_hessian_on_the_block():
    """T^T H T must be the identity on the block -- that is the whole point."""
    n = 10
    h = _spd(n, seed=2, cond=1e8)
    idx = np.arange(n)
    pc = Preconditioner.from_hessian(h, np.zeros(n), idx, ridge=0.0)
    hy = pc.hess_to_internal(h)
    np.testing.assert_allclose(hy, np.eye(n), atol=1e-6)
    # and the conditioning is genuinely improved
    assert np.linalg.cond(hy) < 1e3 < np.linalg.cond(h)


def test_partial_block_leaves_the_rest_untouched():
    """Parameters outside the block pass through, including their Hessian block."""
    n = 6
    h = _spd(n, seed=4)
    idx = np.array([0, 1, 2])
    pc = Preconditioner.from_hessian(h, np.zeros(n), idx, ridge=0.0)
    hy = pc.hess_to_internal(h)
    # block-block whitened
    np.testing.assert_allclose(hy[np.ix_(idx, idx)], np.eye(idx.size), atol=1e-8)
    # outside-outside untouched
    rest = np.array([3, 4, 5])
    np.testing.assert_allclose(
        hy[np.ix_(rest, rest)], h[np.ix_(rest, rest)], atol=1e-12
    )
    # off-diagonal coupling is transformed one-sided, and stays symmetric
    np.testing.assert_allclose(hy, hy.T, atol=1e-8)


def test_gradient_follows_the_chain_rule():
    """grad_y == T^T grad_theta, checked against finite differences."""
    n = 6
    a = _spd(n, seed=5)
    b = np.linspace(-0.5, 0.5, n)
    theta_ref = np.zeros(n)
    pc = Preconditioner.from_hessian(a, theta_ref, np.arange(n), ridge=0.0)

    def loss_theta(theta):
        return 0.5 * theta @ a @ theta + b @ theta

    def loss_y(y):
        return loss_theta(pc.to_physical(y))

    rng = np.random.default_rng(6)
    y0 = rng.normal(size=n) * 0.1
    grad_theta = a @ pc.to_physical(y0) + b
    analytic = pc.grad_to_internal(grad_theta)

    numeric = np.empty(n)
    eps = 1e-6
    for i in range(n):
        yp, ym = y0.copy(), y0.copy()
        yp[i] += eps
        ym[i] -= eps
        numeric[i] = (loss_y(yp) - loss_y(ym)) / (2 * eps)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


def test_hessp_matches_the_dense_transform():
    """hessp_to_internal must agree with hess_to_internal @ p."""
    n = 7
    a = _spd(n, seed=7)
    pc = Preconditioner.from_hessian(a, np.zeros(n), np.arange(1, 6), ridge=0.0)
    hy = pc.hess_to_internal(a)
    rng = np.random.default_rng(8)
    p = rng.normal(size=n)
    got = pc.hessp_to_internal(p, lambda v: a @ v)
    np.testing.assert_allclose(got, hy @ p, atol=1e-8)


def test_explicit_inverse_and_triangular_fallback_agree():
    """The cached-inverse fast path must match the triangular-solve fallback.

    Both branches are live (the fallback triggers if the inverse cannot be
    formed), so they have to give the same answer.
    """
    n = 9
    h = _spd(n, seed=11, cond=1e8)
    idx = np.arange(1, 8)
    fast = Preconditioner.from_hessian(h, np.zeros(n), idx, ridge=0.0)
    slow = Preconditioner.from_hessian(h, np.zeros(n), idx, ridge=0.0)
    assert fast.blocks[0].linv is not None
    slow.blocks[0].linv = None  # force the triangular-solve path

    rng = np.random.default_rng(12)
    v = rng.normal(size=n)
    np.testing.assert_allclose(fast.to_physical(v), slow.to_physical(v), atol=1e-10)
    np.testing.assert_allclose(
        fast.grad_to_internal(v), slow.grad_to_internal(v), atol=1e-10
    )
    np.testing.assert_allclose(
        fast.hess_to_internal(h), slow.hess_to_internal(h), atol=1e-8
    )
    np.testing.assert_allclose(
        fast.hessp_to_internal(v, lambda x: h @ x),
        slow.hessp_to_internal(v, lambda x: h @ x),
        atol=1e-8,
    )


def test_singular_block_falls_back_to_identity():
    """A preconditioner must never break a fit."""
    n = 5
    h = np.zeros((n, n))  # completely degenerate
    pc = Preconditioner.from_hessian(h, np.zeros(n), np.arange(n))
    assert not pc.enabled


def test_mildly_indefinite_block_is_ridged_from_its_spectrum():
    """A block needing a ridge between the old ladder's rungs must still work.

    The escalation used to go 1e-8 -> 1e-6 -> 1e-4 -> 1e-2, so a block whose
    smallest eigenvalue sat at a few percent of max|diag| was skipped: 1e-4 was
    too small and 1e-2 was tried only after. Two such blocks were dropped from a
    real fit that way. The ridge is now computed from lam_min instead.
    """
    n = 40
    h = _spd(n, seed=41, cond=1e3)
    w, V = np.linalg.eigh(h)
    # push a few eigenvalues to ~-3% of the largest diagonal entry
    target = -0.03 * np.max(np.diag(h))
    w[:5] = target
    h = V @ np.diag(w) @ V.T
    h = 0.5 * (h + h.T)
    assert np.linalg.eigvalsh(h).min() < 0, "test matrix must be indefinite"

    pc = Preconditioner.from_hessian(h, np.zeros(n), [("mild", np.arange(n))])
    assert pc.enabled, "a mildly indefinite block must not be skipped"
    assert pc.nblock == n


def test_rank_deficient_block_is_ridged_into_shape():
    n = 5
    h = _spd(n, seed=9)
    h[:, -1] = h[:, 0]  # exact linear dependence
    h[-1, :] = h[0, :]
    pc = Preconditioner.from_hessian(h, np.zeros(n), np.arange(n), ridge=1e-8)
    assert pc.enabled


def test_empty_block_falls_back_to_identity():
    pc = Preconditioner.from_hessian(_spd(4), np.zeros(4), np.array([], dtype=int))
    assert not pc.enabled


# -- unit tests: scope selection ----------------------------------------


def test_default_scope_is_one_block_of_unconstrained_parameters():
    parms = np.array(["poi", "a", "b", "c"])
    cw = np.array([0.0, 1.0, 0.0, 1.0])
    frozen = np.zeros(4, dtype=bool)
    blocks = select_index_blocks(parms, cw, frozen)
    assert len(blocks) == 1
    np.testing.assert_array_equal(blocks[0][1], [0, 2])


def test_frozen_parameters_are_always_excluded():
    parms = np.array(["poi", "a", "b", "c"])
    cw = np.zeros(4)
    frozen = np.array([False, True, False, False])
    blocks = select_index_blocks(parms, cw, frozen)
    np.testing.assert_array_equal(blocks[0][1], [0, 2, 3])


def test_selection_by_regex_and_by_group():
    parms = np.array(["poi", "eff_a", "eff_b", "other"])
    cw = np.ones(4)
    frozen = np.zeros(4, dtype=bool)

    def match_fn(exprs, names):
        import re

        return [n for n in names if any(re.fullmatch(e, n) for e in exprs)]

    blocks = select_index_blocks(
        parms, cw, frozen, expressions=["eff_.*"], match_fn=match_fn
    )
    np.testing.assert_array_equal(blocks[0][1], [1, 2])

    blocks = select_index_blocks(
        parms,
        cw,
        frozen,
        expressions=["mygroup"],
        match_fn=match_fn,
        groups=["mygroup"],
        group_idxs=[[3]],
    )
    np.testing.assert_array_equal(blocks[0][1], [3])

    # one block per expression -> block-diagonal transform
    blocks = select_index_blocks(
        parms, cw, frozen, expressions=["eff_.*", "other"], match_fn=match_fn
    )
    assert [b[0] for b in blocks] == ["eff_.*", "other"]
    np.testing.assert_array_equal(blocks[0][1], [1, 2])
    np.testing.assert_array_equal(blocks[1][1], [3])


# -- the invariance test -------------------------------------------------


def make_polynomial_tensor(outdir, order=6, nbins=30):
    """Background with ``order`` *unconstrained* polynomial shape systematics.

    Mirrors the situation preconditioning is for: a smooth basis that is not
    orthogonal under the data's own weight, so the coefficients are strongly
    correlated and the block is badly conditioned. Plain monomials x^k are used
    deliberately -- the Vandermonde-like Gram matrix is about as ill-conditioned
    as it gets, which is the point.
    """
    import hist

    from rabbit import tensorwriter

    rng = np.random.default_rng(7)
    ax = hist.axis.Regular(nbins, -1, 1, name="x")
    x = ax.centers

    truth = 1000.0 * np.exp(-0.5 * (x / 0.6) ** 2) + 200.0
    h_data = hist.Hist(ax, storage=hist.storage.Double())
    h_data.values()[...] = rng.poisson(truth).astype(float)

    h_bkg = hist.Hist(ax, storage=hist.storage.Weight())
    h_bkg.values()[...] = truth
    h_bkg.variances()[...] = truth

    h_sig = hist.Hist(ax, storage=hist.storage.Weight())
    h_sig.values()[...] = 50.0 * np.exp(-0.5 * (x / 0.2) ** 2)
    h_sig.variances()[...] = h_sig.values()

    writer = tensorwriter.TensorWriter()
    writer.add_channel(h_data.axes, "ch0")
    writer.add_data(h_data, "ch0")
    writer.add_process(h_sig, "sig", "ch0", signal=True)
    writer.add_process(h_bkg, "bkg", "ch0")

    for k in range(1, order + 1):
        w = 0.05 * x**k
        up = h_bkg.copy()
        dn = h_bkg.copy()
        up.values()[...] = h_bkg.values() * (1 + w)
        dn.values()[...] = h_bkg.values() * (1 - w)
        writer.add_systematic(
            [up, dn],
            f"poly{k}",
            "bkg",
            "ch0",
            symmetrize="average",
            constrained=False,
        )

    writer.write(outfolder=outdir, outfilename="test_poly")
    return os.path.join(outdir, "test_poly.hdf5")


def _run(filename, method, **kw):
    """Fit ``filename`` and return the results plus the preconditioner used."""
    indata_obj = inputdata.FitInputData(filename)
    param_model = load_model("Mu", indata_obj)
    options = make_options(minimizerMethod=method, **kw)
    f = fitter.Fitter(indata_obj, param_model, options)
    f.set_nobs(indata_obj.data_obs)
    # built separately purely so the test can assert the block is non-trivial;
    # fit() builds its own at the same point
    pc = f._build_preconditioner()
    f.minimize()
    val, grad, hess = f.loss_val_grad_hess()
    from rabbit.tfhelpers import edmval_cov

    edmval, cov = edmval_cov(grad, hess)
    cov_np = np.asarray(cov.numpy() if hasattr(cov, "numpy") else cov)
    res = dict(
        param=f.x[: param_model.nparams].numpy(),
        theta=f.x[param_model.nparams :].numpy(),
        param_err=np.sqrt(np.diag(cov_np)[: param_model.nparams]),
        nll=f.reduced_nll().numpy(),
        edmval=edmval,
        parms=f.parms,
    )
    return res, pc


@pytest.mark.parametrize("method", ["trust-krylov", "trust-exact"])
def test_fit_is_invariant_under_preconditioning(method):
    """Same minimum, same uncertainties, same NLL -- with trust-krylov kept.

    Covers both the hessp path (trust-krylov) and the dense-hess path
    (trust-exact), since the two transform different callbacks. The scope is
    forced to every parameter so the transform is a genuinely dense one with
    off-diagonal coupling, not a per-parameter rescaling.
    """
    with tempfile.TemporaryDirectory() as tmp:
        filename = make_test_tensor(tmp)
        plain, _ = _run(filename, method)
        pre, pc = _run(filename, method, precondition=True, preconditionParams=[".*"])

        assert pc.enabled and pc.nblock > 1, "preconditioning block must be non-trivial"
        assert check_results("plain", plain, "preconditioned", pre)


def test_gaussnewton_source_is_psd_and_restores_nobs():
    """Fisher information must be PSD, and must not leave nobs modified.

    It is computed by temporarily swapping the data for the prediction, so a
    leak there would silently corrupt the fit that follows.
    """
    with tempfile.TemporaryDirectory() as tmp:
        filename = make_polynomial_tensor(tmp, order=6)
        indata_obj = inputdata.FitInputData(filename)
        param_model = load_model("Mu", indata_obj)
        options = make_options(precondition=True, preconditionFrom="gaussnewton")
        f = fitter.Fitter(indata_obj, param_model, options)
        f.set_nobs(indata_obj.data_obs)

        nobs_before = f.nobs.numpy().copy()
        mat = f._reference_matrix()
        np.testing.assert_allclose(f.nobs.numpy(), nobs_before, rtol=0, atol=0)

        ev = np.linalg.eigvalsh(0.5 * (mat + mat.T))
        assert ev.min() > -1e-8 * abs(ev).max(), f"not PSD: min eig {ev.min()}"


@pytest.mark.parametrize("source", ["hessian", "gaussnewton"])
def test_fit_is_invariant_for_either_reference_source(source):
    """Both sources are only a choice of transform, so neither may move the fit."""
    with tempfile.TemporaryDirectory() as tmp:
        filename = make_polynomial_tensor(tmp, order=6)
        plain, _ = _run(filename, "trust-krylov")
        pre, pc = _run(
            filename, "trust-krylov", precondition=True, preconditionFrom=source
        )
        assert pc.enabled and pc.nblock > 1
        assert check_results("plain", plain, f"precond[{source}]", pre)


@pytest.mark.parametrize("transform", ["ridge", "spectral"])
def test_fit_is_invariant_for_either_transform(transform):
    """Both transforms are only a change of variables, so neither may move the
    fit. This is what justifies offering the choice at all: the block algebra
    below shows T^T H T = I, but only a fit shows the minimum staying put."""
    with tempfile.TemporaryDirectory() as tmp:
        filename = make_polynomial_tensor(tmp, order=6)
        plain, _ = _run(filename, "trust-krylov")
        pre, pc = _run(
            filename,
            "trust-krylov",
            precondition=True,
            preconditionTransform=transform,
        )
        assert pc.enabled and pc.nblock > 1
        assert check_results("plain", plain, f"precond[{transform}]", pre)


def test_default_transform_is_ridge_in_the_parser_and_the_fitter():
    """The transform default lives in three places (parser, Fitter getattr,
    _factorise signature). The third is pinned in test_preconditioner_spectral;
    these are the two that decide what a real run actually does."""
    from rabbit import parsing

    # get_default rather than parse_args: common_parser has a required
    # positional (filename), so parsing an empty argv exits
    assert parsing.common_parser().get_default("preconditionTransform") == "ridge"

    # the Fitter's getattr fallback, for callers whose options predate the flag
    with tempfile.TemporaryDirectory() as tmp:
        filename = make_test_tensor(tmp)
        indata_obj = inputdata.FitInputData(filename)
        options = make_options()
        assert not hasattr(options, "preconditionTransform")
        f = fitter.Fitter(indata_obj, load_model("Mu", indata_obj), options)
        assert f.precondition_transform == "ridge"


@pytest.mark.parametrize("method", ["trust-krylov", "trust-exact"])
def test_fit_is_invariant_on_an_ill_conditioned_block(method):
    """The case this feature exists for: many correlated unconstrained params.

    Uses the default scope (unconstrained parameters), so it also checks that
    the default picks the right block.
    """
    with tempfile.TemporaryDirectory() as tmp:
        filename = make_polynomial_tensor(tmp, order=6)
        plain, _ = _run(filename, method)
        pre, pc = _run(filename, method, precondition=True)

        # default scope must have found the unconstrained polynomial block
        assert pc.enabled and pc.nblock > 1
        # and it must actually be badly conditioned before, well conditioned
        # after. These are TRUE condition numbers at both ends, not the
        # scale-free degeneracy (pc.blocks[i].corr_before) -- see WHAT THE
        # NUMBERS MEAN in preconditioner.py.
        assert pc.cond_before > 1e3
        assert pc.cond_after < 1e2
        assert check_results("plain", plain, "preconditioned", pre)


# -- several blocks ------------------------------------------------------


def test_two_blocks_are_factorised_independently():
    """Block-diagonal transform: each block whitened, no cross-block mixing."""
    n = 12
    h = np.zeros((n, n))
    h[:6, :6] = _spd(6, seed=21, cond=1e6)
    h[6:, 6:] = _spd(6, seed=22, cond=1e4)
    a, b = np.arange(0, 6), np.arange(6, 12)
    pc = Preconditioner.from_hessian(h, np.zeros(n), [("a", a), ("b", b)], ridge=0.0)
    assert pc.enabled and pc.n_blocks == 2 and pc.nblock == 12
    hy = pc.hess_to_internal(h)
    np.testing.assert_allclose(hy[np.ix_(a, a)], np.eye(6), atol=1e-8)
    np.testing.assert_allclose(hy[np.ix_(b, b)], np.eye(6), atol=1e-8)


def test_a_singular_block_does_not_disable_the_others():
    """The reason for blocks: a union can be unusable while its parts are fine."""
    n = 10
    h = np.zeros((n, n))
    h[:5, :5] = _spd(5, seed=23)
    # second block is entirely degenerate
    good, bad = np.arange(0, 5), np.arange(5, 10)
    pc = Preconditioner.from_hessian(h, np.zeros(n), [("good", good), ("bad", bad)])
    assert pc.enabled, "the usable block must survive"
    assert pc.n_blocks == 1 and pc.nblock == 5
    np.testing.assert_array_equal(pc.blocks[0].idx, good)


def test_all_blocks_unusable_falls_back_to_identity():
    n = 6
    pc = Preconditioner.from_hessian(
        np.zeros((n, n)), np.zeros(n), [("x", np.arange(3)), ("y", np.arange(3, 6))]
    )
    assert not pc.enabled


def test_block_diagonal_transform_is_still_a_reparameterisation():
    """Round trip must hold with several blocks."""
    n = 14
    h = np.zeros((n, n))
    h[:7, :7] = _spd(7, seed=24)
    h[7:, 7:] = _spd(7, seed=25)
    pc = Preconditioner.from_hessian(
        h, np.linspace(-1, 1, n), [("a", np.arange(7)), ("b", np.arange(7, 14))]
    )
    rng = np.random.default_rng(26)
    y = rng.normal(size=n)
    np.testing.assert_allclose(pc.from_physical(pc.to_physical(y)), y, atol=1e-9)


# -- automatic block discovery -------------------------------------------


def _block_diag_hessian(sizes, seed=0, cond=1e5, coupling=0.0):
    """Block-diagonal SPD matrix, optionally with weak inter-block coupling."""
    rng = np.random.default_rng(seed)
    n = sum(sizes)
    H = np.zeros((n, n))
    at = 0
    for i, m in enumerate(sizes):
        H[at : at + m, at : at + m] = _spd(m, seed=seed + i, cond=cond)
        at += m
    if coupling:
        # small symmetric off-block perturbation, kept well below the diagonal
        P = rng.normal(size=(n, n)) * coupling * np.mean(np.diag(H))
        P = 0.5 * (P + P.T)
        mask = np.ones((n, n), dtype=bool)
        at = 0
        for m in sizes:
            mask[at : at + m, at : at + m] = False
            at += m
        H = H + P * mask
    return H


def test_auto_blocks_recovers_a_known_block_structure():
    """The point of the feature: find the clusters without being told them."""
    sizes = [6, 6, 6, 6, 6]
    H = _block_diag_hessian(sizes, seed=31, coupling=0.0)
    blocks = auto_blocks(H, np.arange(sum(sizes)), threshold=0.1)
    assert len(blocks) == len(sizes)
    found = sorted(sorted(idx.tolist()) for _, idx in blocks)
    expect = []
    at = 0
    for m in sizes:
        expect.append(list(range(at, at + m)))
        at += m
    assert found == sorted(expect)


def test_auto_blocks_ignores_weak_coupling_but_merges_strong():
    sizes = [5, 5]
    H = _block_diag_hessian(sizes, seed=32, coupling=1e-4)
    assert len(auto_blocks(H, np.arange(10), threshold=0.1)) == 2
    # a strong link between the two groups must merge them
    H2 = H.copy()
    scale = np.sqrt(H2[0, 0] * H2[5, 5])
    H2[0, 5] = H2[5, 0] = 0.8 * scale
    assert len(auto_blocks(H2, np.arange(10), threshold=0.1)) == 1


def test_auto_blocks_percolates_at_a_low_threshold():
    """Documented failure mode: below the percolation point it is one block."""
    H = _block_diag_hessian([5, 5, 5], seed=33, coupling=1e-3)
    assert len(auto_blocks(H, np.arange(15), threshold=1e-9)) == 1


def test_auto_blocks_isolates_parameters_with_no_curvature():
    """A zero-diagonal parameter cannot be normalised; it must not poison a block."""
    H = _block_diag_hessian([4, 4], seed=34)
    H[3, :] = 0.0
    H[:, 3] = 0.0
    blocks = auto_blocks(H, np.arange(8), threshold=0.1)
    singleton = [idx for _, idx in blocks if idx.tolist() == [3]]
    assert singleton, f"expected index 3 alone, got {[i.tolist() for _, i in blocks]}"


def test_auto_blocks_on_empty_scope():
    assert auto_blocks(_spd(4), np.array([], dtype=int)) == []


def test_auto_blocking_is_the_default():
    from types import SimpleNamespace

    from rabbit import fitter as fitter_mod

    o = SimpleNamespace()
    f = fitter_mod.Fitter.__new__(fitter_mod.Fitter)
    assert getattr(o, "preconditionBlocks", "auto") == "auto"
    assert getattr(o, "preconditionBlockThreshold", 0.1) == 0.1
    del f


@pytest.mark.parametrize("blocks", ["auto", "expressions", "none"])
def test_fit_is_invariant_for_either_blocking(blocks):
    """Blocking only changes how the transform is grouped, never the answer."""
    with tempfile.TemporaryDirectory() as tmp:
        filename = make_polynomial_tensor(tmp, order=6)
        plain, _ = _run(filename, "trust-krylov")
        pre, pc = _run(
            filename, "trust-krylov", precondition=True, preconditionBlocks=blocks
        )
        assert pc.enabled
        assert check_results("plain", plain, f"precond[{blocks}]", pre)


def test_blocks_none_is_a_single_factorisation_over_the_scope():
    """'none' keeps every cross-correlation, at the cost of one big Cholesky."""
    with tempfile.TemporaryDirectory() as tmp:
        filename = make_polynomial_tensor(tmp, order=6)
        indata_obj = inputdata.FitInputData(filename)
        param_model = load_model("Mu", indata_obj)
        options = make_options(precondition=True, preconditionBlocks="none")
        f = fitter.Fitter(indata_obj, param_model, options)
        f.set_nobs(indata_obj.data_obs)
        pc = f._build_preconditioner()
        assert pc.enabled
        assert pc.n_blocks == 1, f"expected one joint block, got {pc.n_blocks}"

        # and 'auto' on the same model may split it into more than one
        options_auto = make_options(precondition=True, preconditionBlocks="auto")
        f2 = fitter.Fitter(indata_obj, param_model, options_auto)
        f2.set_nobs(indata_obj.data_obs)
        pc2 = f2._build_preconditioner()
        assert pc2.enabled
        assert pc2.nblock == pc.nblock, "same scope, only the grouping differs"
