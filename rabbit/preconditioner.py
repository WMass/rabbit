"""Optional preconditioning of the fit parameters.

WHY. The trust-region minimizers solve their subproblem in a *spherical* trust
region, and the Krylov/CG inner solve converges in a number of Hessian-vector
products that grows like sqrt(kappa) of the Hessian. When a block of
parameters is strongly correlated -- typically many unconstrained,
weakly-identified coefficients of a smooth parameterisation, where the basis is
not orthogonal under the data's own weight -- kappa is huge, the inner solve
struggles, and the outer step is rejected. The symptom is outer iterations that
cost minutes and return a bit-identical loss.

WHAT. We reparameterise. With theta the physical parameters, y the internal
ones, theta_ref the point the transform was built at, and a reference Hessian
H0 = L L^T restricted to a selected block:

    theta = theta_ref + T y,     T = L^-T   ->   T^T H0 T = I

so the Hessian in y is the identity at theta_ref and the spherical trust region
in y is an H0-aligned ellipsoid in theta. That is exactly preconditioning, but
obtained as a change of variables, which means **the minimizer is untouched**:
scipy's trust-krylov (GLTR/_trlib) accepts no user preconditioner, and it does
not need to.

The chain rule gives what the scipy callbacks must return:

    loss_y(y)      = loss(theta(y))
    grad_y         = T^T grad_theta
    (H_y p)        = T^T H_theta (T p)

T and T^T are applied as dense matvecs against a cached L^-1, which is formed
once at construction. A triangular solve would do the same flops but is
inherently sequential and so cannot use more than one core; the matvec is a
parallel GEMV and measured ~20x faster at m=2112 (0.09 ms vs 1.9 ms). The
triangular solve is kept as a fallback. Forming the inverse is safe here:
against a direct solve it agrees to ~1e-15 even at cond(H)=1e12.

That is worth stating explicitly because the older *offline* in-situ basis had
to row-normalise its L^-1, which looks like a warning against explicit
inverses. It was not one. There the transform reshaped the basis in which the
histmaker precomputed its delta=0.01 finite variations, so a large T moved the
linearisation point far from where those templates were valid -- a modelling
constraint, not a numerical-stability one. Here T is applied to exact vectors
inside the fit, so that failure mode does not arise.

Parameters outside the selected blocks are passed through untouched, so the
transform is a no-op there and a block can be as small as one wants.

FINDING THE BLOCKS. The clusters can be read off the reference matrix instead
of being named by hand: threshold the correlation matrix and take connected
components (see :func:`auto_blocks`). On the in-situ efficiency fit that
recovers 239 components where the parameterisation has 240 (step, eta, charge)
blocks, with no knowledge of parameter names, and costs ~4e4 times fewer flops
to factorise than one joint block, since Cholesky is O(m^3).

SEVERAL BLOCKS. The transform is block diagonal: each selected group of
parameters gets its own factorisation and they are applied independently. That
is not only cheaper (m^3 per block instead of (sum m)^3) but often the only
thing that works -- a union of two individually well-behaved groups can be
singular, because the groups are nearly degenerate *with each other*, and one
joint Cholesky then fails where two separate ones succeed. The price is that
correlations between blocks are left alone, so blocks should be chosen to be
the strongly-correlated clusters.

SCOPE. Preconditioning buys little for constrained nuisances: their unit
Gaussian prior contributes the identity, H = I + J^T W J, so kappa is bounded.
The win is for *unconstrained* parameters and POIs, which is the default scope.

Everything here operates on numpy arrays at the scipy boundary; the fitter
keeps holding physical parameters in its tf.Variable, so nothing downstream
(covariance, impacts, pulls) needs to know that preconditioning happened.
"""

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.csgraph
from wums import logging

logger = logging.child_logger(__name__)

# Sources for the reference matrix, built by the fitter (see Fitter._reference_matrix).
PRECONDITION_SOURCES = ("hessian", "gaussnewton")


class Block:
    """One factorised group of parameters: indices, L, and a cached L^-1."""

    def __init__(self, idx, chol, cond_before=None, cond_after=None, label=""):
        self.idx = np.asarray(idx, dtype=np.int64)
        self.chol = np.asarray(chol, dtype=np.float64)
        if self.chol.shape != (self.idx.size,) * 2:
            raise ValueError(
                f"chol shape {self.chol.shape} does not match block size {self.idx.size}"
            )
        self.cond_before = cond_before
        self.cond_after = cond_after
        self.label = label
        # Explicit L^-1, formed once so the per-call transform is a dense matvec
        # (parallel GEMV) instead of a triangular solve (inherently sequential).
        # Measured at m=2112: 0.09 ms vs 1.9 ms per application. Falls back to
        # the solve if the inverse cannot be formed.
        self.linv = None
        try:
            self.linv = scipy.linalg.solve_triangular(
                self.chol, np.eye(self.chol.shape[0]), lower=True, trans="N"
            )
        except (scipy.linalg.LinAlgError, ValueError) as ex:
            logger.warning(
                f"Could not form the explicit inverse ({ex}); "
                "falling back to triangular solves."
            )


class Preconditioner:
    """Block-diagonal affine reparameterisation theta = theta_ref + T y.

    Each block contributes T = L^-T on its own indices; everything else is
    passed through. Use :meth:`identity` for the disabled case: it is an exact
    no-op, so the fitter has a single code path whether or not preconditioning
    is on.
    """

    def __init__(self, theta_ref, blocks=()):
        self.theta_ref = np.asarray(theta_ref, dtype=np.float64)
        self.n = self.theta_ref.size
        self.blocks = list(blocks)

    # -- construction ----------------------------------------------------

    @classmethod
    def identity(cls, theta_ref):
        """Disabled preconditioner: y == theta - theta_ref, no blocks."""
        return cls(theta_ref)

    @property
    def enabled(self):
        return bool(self.blocks)

    @property
    def nblock(self):
        """Total number of preconditioned parameters, over all blocks."""
        return int(sum(b.idx.size for b in self.blocks))

    @property
    def n_blocks(self):
        return len(self.blocks)

    @property
    def cond_before(self):
        c = [b.cond_before for b in self.blocks if b.cond_before is not None]
        return max(c) if c else None

    @property
    def cond_after(self):
        c = [b.cond_after for b in self.blocks if b.cond_after is not None]
        return max(c) if c else None

    @classmethod
    def from_hessian(cls, hess, theta_ref, index_blocks, ridge=1e-8, max_tries=4):
        """Build from a reference Hessian, one factorisation per index block.

        ``index_blocks`` is a list of index arrays (a single array is accepted
        and treated as one block). Each block is symmetrised, then a ridge
        proportional to its largest diagonal entry is added until the Cholesky
        succeeds. A block that cannot be factorised is dropped -- the others
        still apply, and a preconditioner must never break a fit.
        """
        if isinstance(index_blocks, np.ndarray) or (
            index_blocks and np.isscalar(index_blocks[0])
        ):
            index_blocks = [index_blocks]
        blocks = []
        for spec in index_blocks:
            label, idx = ("", spec) if not isinstance(spec, tuple) else spec
            blk = cls._factorise(
                hess, np.asarray(idx, dtype=np.int64), ridge, max_tries, label
            )
            if blk is not None:
                blocks.append(blk)
        if not blocks:
            logger.warning(
                "Preconditioning requested but no block could be used; "
                "running unpreconditioned."
            )
            return cls.identity(theta_ref)

        # One summary rather than a line per block: auto-blocking routinely
        # finds hundreds, and the per-block detail is at debug level.
        n_req = len(index_blocks)
        npar = sum(b.idx.size for b in blocks)
        conds = [b.cond_before for b in blocks if b.cond_before is not None]
        summary = f"Preconditioned {len(blocks)} of {n_req} block(s), {npar} parameters"
        if conds:
            summary += (
                f"; correlation condition number median {np.median(conds):.3g}, "
                f"worst {max(conds):.3g} -> 1 at the reference point"
            )
        if len(blocks) < n_req:
            summary += f" ({n_req - len(blocks)} block(s) not factorisable, skipped)"
        logger.info(summary)
        return cls(theta_ref, blocks)

    @staticmethod
    def _factorise(hess, idx, ridge, max_tries, label=""):
        """One block -> a :class:`Block`, or None if it is unusable."""
        tag = f"{label} " if label else ""
        if idx.size == 0:
            logger.warning(f"Preconditioning block {tag}is empty; skipping.")
            return None

        block = np.asarray(hess, dtype=np.float64)[np.ix_(idx, idx)]
        # symmetrise: the autodiff Hessian is symmetric only up to roundoff
        block = 0.5 * (block + block.T)

        diag = np.diag(block)
        scale = float(np.max(diag)) if diag.size else 0.0
        if not np.isfinite(scale) or scale <= 0.0:
            logger.warning(
                f"Preconditioning block {tag}has no positive diagonal "
                f"(max diag = {scale}); skipping."
            )
            return None

        cond_before = _cond_corr(block)
        # Ridge schedule: try the caller's value first, since for a positive
        # definite block that is all that is needed and the Cholesky is cheap.
        # Only if that fails is the spectrum worth the extra O(m^3): the ridge
        # required to restore definiteness is set by the most negative
        # eigenvalue, so it can be computed rather than guessed. Escalating by
        # powers of a hundred instead used to overshoot badly -- blocks needing
        # 0.03 were skipped after 1e-4 failed and 1e-2 was tried next.
        schedule = [ridge]
        for itry in range(max_tries):
            if itry >= len(schedule):
                if itry == 1:
                    w = np.linalg.eigvalsh(block)
                    lam_min, lam_max = float(w[0]), float(w[-1])
                    if lam_min < 0.0:
                        # enough to make it positive definite, plus a margin so
                        # the smallest eigenvalue is not left at zero
                        need = abs(lam_min) + max(
                            0.1 * abs(lam_min), 1e-8 * abs(lam_max)
                        )
                        schedule.append(need / scale)
                        logger.debug(
                            f"{tag}block has lam_min={lam_min:.3g} "
                            f"(max|diag|={scale:.3g}); ridge from the spectrum: "
                            f"{schedule[-1]:.3g} x max|diag|"
                        )
                    else:
                        schedule.append(max(schedule[-1], 1e-12) * 100.0)
                else:
                    schedule.append(max(schedule[-1], 1e-12) * 100.0)
            eps = schedule[itry]
            trial = block.copy()
            if eps > 0.0:
                trial[np.diag_indices_from(trial)] += eps * scale
            try:
                chol = scipy.linalg.cholesky(trial, lower=True)
            except scipy.linalg.LinAlgError:
                logger.debug(
                    f"Preconditioner Cholesky failed for {tag}block "
                    f"with ridge {eps:.3g} x max|diag| (try {itry + 1})"
                )
                continue
            # Conditioning actually achieved: L^-1 B L^-T for the *un-ridged*
            # block B. Using the ridged matrix here would return 1 by
            # construction and measure nothing.
            tb = scipy.linalg.solve_triangular(chol, block, lower=True, trans="N")
            tb = scipy.linalg.solve_triangular(chol, tb.T, lower=True, trans="N").T
            cond_after = _cond_corr(tb)
            logger.debug(
                f"Preconditioning {tag}block of {idx.size} parameters from the "
                f"reference Hessian (ridge {eps:.3g} x max|diag|): correlation "
                f"condition number {cond_before:.3g} -> {cond_after:.3g} at the "
                "reference point"
            )
            return Block(idx, chol, cond_before, cond_after, label)

        logger.warning(
            f"Preconditioning block {tag}is not factorisable; the largest ridge "
            f"tried was {max(schedule):.3g} x max|diag|. Skipping this block."
        )
        return None

    # -- the transform ---------------------------------------------------

    def _apply_T(self, v):
        """T v: L^-T on each block, identity elsewhere."""
        if not self.blocks:
            return np.asarray(v, dtype=np.float64)
        out = np.array(v, dtype=np.float64, copy=True)
        for b in self.blocks:
            if b.linv is not None:
                out[b.idx] = b.linv.T @ out[b.idx]
            else:
                out[b.idx] = scipy.linalg.solve_triangular(
                    b.chol, out[b.idx], lower=True, trans="T"
                )
        return out

    def _apply_TT(self, v):
        """T^T v: L^-1 on each block, identity elsewhere."""
        if not self.blocks:
            return np.asarray(v, dtype=np.float64)
        out = np.array(v, dtype=np.float64, copy=True)
        for b in self.blocks:
            if b.linv is not None:
                out[b.idx] = b.linv @ out[b.idx]
            else:
                out[b.idx] = scipy.linalg.solve_triangular(
                    b.chol, out[b.idx], lower=True, trans="N"
                )
        return out

    def tf_transforms(self):
        """TF-graph versions of ``_apply_T``/``_apply_TT``, or None if no-op.

        For the matrix-free (HVP) native subproblem the reparameterisation
        must run *inside* the compiled CG loop -- once per Hessian-vector
        product -- so the numpy implementations above cannot be used there.
        The block matrices are captured as tf constants once; each
        application is gather -> dense matvec -> scatter per block, the same
        parallel-GEMV rationale as the cached L^-1 on the numpy side.
        """
        if not self.blocks:
            return None
        import tensorflow as tf

        ops = []
        for b in self.blocks:
            gather_idx = tf.constant(b.idx, dtype=tf.int64)
            scatter_idx = tf.constant(b.idx.reshape(-1, 1), dtype=tf.int64)
            linv = tf.constant(b.linv) if b.linv is not None else None
            chol = tf.constant(b.chol) if b.linv is None else None
            ops.append((gather_idx, scatter_idx, linv, chol))

        def _apply(v, transpose):
            # transpose=True: T v (L^-T per block); False: T^T v (L^-1)
            out = v
            for gidx, sidx, linv, chol in ops:
                sub = tf.gather(out, gidx)
                if linv is not None:
                    new = tf.linalg.matvec(linv, sub, transpose_a=transpose)
                else:
                    new = tf.squeeze(
                        tf.linalg.triangular_solve(
                            chol, sub[:, None], lower=True, adjoint=transpose
                        ),
                        axis=-1,
                    )
                out = tf.tensor_scatter_nd_update(out, sidx, new)
            return out

        def apply_T(v):
            return _apply(v, True)

        def apply_TT(v):
            return _apply(v, False)

        return apply_T, apply_TT

    def to_physical(self, y):
        """theta = theta_ref + T y."""
        return self.theta_ref + self._apply_T(y)

    def from_physical(self, theta):
        """y = T^-1 (theta - theta_ref), i.e. L^T on each block."""
        d = np.asarray(theta, dtype=np.float64) - self.theta_ref
        if not self.blocks:
            return d
        out = np.array(d, dtype=np.float64, copy=True)
        for b in self.blocks:
            out[b.idx] = b.chol.T @ d[b.idx]
        return out

    def grad_to_internal(self, grad):
        """grad_y = T^T grad_theta."""
        return self._apply_TT(grad)

    def hessp_to_internal(self, p, hvp):
        """H_y p = T^T H_theta (T p); ``hvp`` maps a physical-space vector."""
        return self._apply_TT(hvp(self._apply_T(p)))

    def hess_to_internal(self, hess):
        """H_y = T^T H_theta T, for the dense-Hessian minimizers.

        Applied block by block: the left multiplication acts on the row index
        and the right one on the column index, so a block's off-diagonal
        coupling to the rest of the model transforms one-sided, as it should.
        """
        if not self.blocks:
            return np.asarray(hess, dtype=np.float64)
        out = np.array(hess, dtype=np.float64, copy=True)
        for b in self.blocks:
            if b.linv is not None:
                out[b.idx, :] = b.linv @ out[b.idx, :]
            else:
                out[b.idx, :] = scipy.linalg.solve_triangular(
                    b.chol, out[b.idx, :], lower=True, trans="N"
                )
        for b in self.blocks:
            if b.linv is not None:
                out[:, b.idx] = out[:, b.idx] @ b.linv.T
            else:
                out[:, b.idx] = scipy.linalg.solve_triangular(
                    b.chol, out[:, b.idx].T, lower=True, trans="N"
                ).T
        return out

    # -- diagnostics -----------------------------------------------------

    def summary(self):
        if not self.blocks:
            return "preconditioning: disabled"
        return (
            f"preconditioning: {self.n_blocks} block(s) covering "
            f"{self.nblock} of {self.n} parameters"
        )


def _cond_corr(mat):
    """Condition number of the *correlation* matrix of ``mat``.

    Scale-invariant, so it measures genuine degeneracy rather than a mismatch
    of units between parameters -- the quantity that actually governs how hard
    the block is to fit.
    """
    d = np.sqrt(np.abs(np.diag(mat)))
    good = d > 0
    if not np.any(good):
        return np.inf
    m = mat[np.ix_(good, good)] / np.outer(d[good], d[good])
    try:
        sv = np.linalg.svd(m, compute_uv=False)
    except np.linalg.LinAlgError:
        return np.inf
    return float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf


def auto_blocks(hess, idx, threshold=0.1, max_fraction=0.5):
    """Find the correlated clusters within ``idx`` from the reference matrix.

    Thresholds the correlation matrix at ``threshold`` and returns the connected
    components as blocks. Parameters correlate strongly with the others in their
    cluster and negligibly across clusters, which is exactly the structure a
    block-diagonal transform wants, and the components are far smaller than the
    union so each Cholesky is cheaper and likelier to succeed.

    ``threshold`` matters: too low and everything percolates into one component,
    too high and genuinely coupled parameters are split apart. Correlations below
    it are left unpreconditioned, which is the deliberate approximation. A
    component covering more than ``max_fraction`` of the parameters is warned
    about, since that usually means percolation.

    Parameters with a non-positive diagonal cannot be normalised and are
    returned as singletons, i.e. effectively left alone.
    """
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return []
    sub = np.asarray(hess, dtype=np.float64)[np.ix_(idx, idx)]
    sub = 0.5 * (sub + sub.T)
    d = np.diag(sub)
    good = d > 0
    n = idx.size
    corr = np.zeros((n, n))
    if np.any(good):
        g = np.where(good)[0]
        dd = np.sqrt(d[g])
        corr[np.ix_(g, g)] = np.abs(sub[np.ix_(g, g)] / np.outer(dd, dd))
    np.fill_diagonal(corr, 0.0)

    adj = scipy.sparse.csr_matrix(corr > threshold)
    ncomp, labels = scipy.sparse.csgraph.connected_components(adj, directed=False)
    sizes = np.bincount(labels, minlength=ncomp)
    biggest = int(sizes.max())
    if biggest > max_fraction * n:
        logger.warning(
            f"Auto-blocking at |rho| > {threshold} produced a component with "
            f"{biggest} of {n} parameters: the threshold is probably below the "
            "percolation point, so the blocks are not really separated."
        )
    logger.info(
        f"Auto-blocking {n} parameters at |rho| > {threshold}: {ncomp} block(s), "
        f"largest {biggest}, median {int(np.median(sizes))}"
    )
    return [(f"auto{c}", idx[np.where(labels == c)[0]]) for c in range(ncomp)]


def select_index_blocks(
    parms,
    cw,
    frozen_mask,
    expressions=None,
    match_fn=None,
    groups=None,
    group_idxs=None,
):
    """Parameter blocks to precondition, as a list of ``(label, indices)``.

    One block per entry in ``expressions``, which is what makes the transform
    block diagonal: each expression is expected to name a cluster of parameters
    that are correlated with each other. Grouping them into one factorisation
    instead is both more expensive and more fragile -- the union of two
    individually fine groups can be singular because the groups are nearly
    degenerate with each other.

    An entry may name parameters exactly, be a regex matched against the full
    parameter name (via ``match_fn``, the fitter's existing matcher), or name a
    systematic group. With no expressions there is a single block of every
    *unconstrained* parameter (cw == 0), which is where preconditioning helps.

    Frozen parameters are always excluded: a dense transform would otherwise
    mix a frozen parameter back into the fit through the other coordinates.
    """
    parms = np.asarray(parms).astype(str)
    n = parms.size
    frozen_mask = np.asarray(frozen_mask, dtype=bool)

    if not expressions:
        sel = (np.asarray(cw) == 0.0) & ~frozen_mask
        # debug, not info: this repeats identically on every restart's rebuild,
        # and the summary from from_hessian already reports how many parameters
        # ended up preconditioned
        logger.debug(
            f"No --preconditionParams given; selecting all {int(sel.sum())} "
            "unconstrained parameters (constraint weight 0). How they are grouped "
            "into blocks is set by --preconditionBlocks."
        )
        return [("unconstrained", np.where(sel)[0])]

    # NB explicit None checks: groups/group_idxs arrive as numpy arrays,
    # for which `groups or []` raises on the truth-value test.
    gnames = [] if groups is None else list(groups)
    gidxs = [] if group_idxs is None else list(group_idxs)
    by_group = {
        (k.decode() if isinstance(k, bytes) else str(k)): v
        for k, v in zip(gnames, gidxs)
    }

    out = []
    for expr in expressions:
        sel = np.zeros(n, dtype=bool)
        if expr in by_group:
            sel[np.asarray(by_group[expr], dtype=np.int64)] = True
        else:
            if match_fn is None:
                raise ValueError("no matcher available for regex selection")
            sel |= np.isin(parms, match_fn([expr], parms))
        sel &= ~frozen_mask
        idx = np.where(sel)[0]
        if idx.size == 0:
            logger.warning(f"--preconditionParams '{expr}' matched no parameters")
            continue
        out.append((expr, idx))
    return out
