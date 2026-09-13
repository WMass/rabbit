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
H0 = L L^T restricted to a selected block (away from a minimum H0 is
generally indefinite and has no Cholesky -- see WHITENING EACH BLOCK below for
the two ways to get an L):

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

WHITENING EACH BLOCK. There are two ways to produce the L above, selected by
--preconditionTransform (see :meth:`_factorise`):

    'ridge' (default)  factorise H + eps*I, raising eps until the Cholesky
                       succeeds. rabbit's original transform.
    'spectral'         factorise |H| = Q |Lambda| Q^T, giving every
                       eigendirection its own scale.

They agree exactly on a positive definite block -- both return the identity,
so nothing is lost on the easy ones -- and differ where H is indefinite. A
single scalar ridge must be at least |lam_min| to restore definiteness, so in
a block whose curvatures span orders of magnitude it swamps every direction
softer than itself and the whitening leaves those near-null: measured on
curvatures (3.3e9, -8.7e6, 1) the O(1) direction came out at 7e-08 and the
block's true condition number went 3.3e+09 -> 1.44e+08, where spectral gives
1. That is the genuine limitation of one scalar number, and the reason
spectral exists.

Spectral takes |Lambda| rather than clipping the spectrum to +eps, which keeps
the SIGN of each direction in the new coordinates. That matters because
trust-krylov exploits negative curvature to escape saddles, and flattening it
to +1 throws that away. It costs an eigendecomposition rather than a Cholesky:
measured 11x at m=200 and 44x at m=2112, i.e. ~1.7 s once per build, against a
fit of tens of minutes.

CHOOSING THE OPTIONS. The guidance for the neighbouring options *inverts* with
the transform, so they are worth reading together rather than one --help
string at a time:

--preconditionBlocks
    Under 'ridge' blocking is a *correctness* tool: one scalar ridge cannot
    serve a block whose curvatures span orders of magnitude, so splitting
    protects the soft directions. Under 'spectral' a larger block is always
    better or equal, since it whitens the cross-terms exactly where splitting
    discards them (measured: one block of 8 gives condition number 1, the same
    matrix split in two gives 4.0). There, blocking only limits the O(m^3)
    cost and 'none' is usually right.

--preconditionBlockThreshold
    Under 'spectral' this works against you: it splits away precisely the
    correlations spectral handles exactly, so "every parameter percolates into
    a single block" is the ideal there rather than the warning it is under
    'ridge'.

--preconditionFrom
    Under 'spectral' there is no reason to pick 'gaussnewton': its PSD-ness
    exists to guarantee the Cholesky succeeds, which spectral does not need,
    and it still cannot represent negative curvature. Prefer 'hessian'.

--preconditionRidge
    Applies to 'ridge' only; spectral derives its floor from the numerical
    rank of the block instead.

WHAT THE NUMBERS MEAN. Two different questions get asked about a block, and
conflating them is how a preconditioner comes to look like it worked:

    condition number   kappa of the block itself (:func:`_cond_true`). This is
                       what the minimizer feels -- the Krylov inner solve costs
                       ~sqrt(kappa) Hessian-vector products -- so it is the
                       number a transform has to reduce, and it is reported at
                       BOTH ends of the before/after arrow.
    degeneracy         kappa of the block's CORRELATION matrix
                       (:func:`_cond_corr`), i.e. after normalising to unit
                       diagonal. Scale-free, so it isolates genuine
                       near-linear-dependence from a mismatch of units.

The degeneracy is reported for the block as it ARRIVES only, where the units
are an arbitrary convention and factoring them out is the right thing to do.
It is deliberately not reported for the whitened block, for two reasons:

- the units there are not arbitrary -- they are precisely what the transform
  chose, and flattening them hides the failure mode of a single scalar ridge,
  which swamps every direction softer than |lam_min| and leaves them near-null
  (measured: 1.225 -> 1 by correlation, where the truth is 3.3e9 -> 1.4e8);
- on the spectral path it is not even an approximation, it is an artefact. B
  and |B| share eigenvectors, so with |B| = L L^T the congruence L^-1 B L^-T
  sends Lambda to its own signature: tb is symmetric AND orthogonal, tb^2 = I,
  and true kappa is identically 1 for any block size. But L is a Cholesky
  factor, not the eigenbasis, so diag(tb) is only bounded by [-1, 1] and the
  1/sqrt|diag| normalisation picks up the arbitrary orientation between the
  two. Measured on random indefinite blocks, the correlation number then reads
  1.7 at m=5, 7.7 at m=12, 21 at m=30, 34 at m=60 -- growing with block size,
  i.e. worst exactly where --preconditionBlocks none sends you -- while the
  true value is 1 throughout.

  The identity holds only where nothing was floored: a floored direction is one
  the congruence no longer preserves, so n_floored > 0 legitimately takes true
  kappa away from 1. That is why it is measured per block rather than assumed.

NEVER MAKE IT WORSE. A block is dropped if it cannot be factorised, and also
if whitening it would make its TRUE condition number worse by more than
DEGRADE_TOLERANCE. The second half matters because succeeding badly is not
obviously better than failing: whitening a block whose whole spectrum is
negative needs a ridge of at least |lam_min| ~ max|diag|, which swamps every
direction in it. Measured over 7094 such blocks across four off-diagonal
strengths, the ridge degraded the true condition number in 100% of them, median
10.8x, and improved none. Dropping restores exactly the unpreconditioned
behaviour, so a fit can never come out worse on that block than with
--precondition off.

The guard keys off the true condition number, not the correlation one, which is
blind to the clean case: a diagonal block has the identity as its correlation
matrix both before and after, so diag(-7.5e3, -6.3e4, -2.1e3) reports 1 -> 1
where the truth is 30 -> 320.

It does not act on a block that is already singular to working precision
(SINGULAR_COND). There the true condition number stops being a measurement --
one rank-deficient block reads 5.28e+16 -> 7.32e+16, a 1.39x "degradation"
between two values past 1/eps -- while the correlation number shows the ridge
genuinely helping it, 3.1e+16 -> 1.4e+12. Ridging a rank-deficient block into
shape is what the ridge was for in the first place, so that behaviour is left
alone. It is the mirror image of the all-negative case, and the reason both
metrics are kept rather than one being declared the right one.

One consequence worth being plain about. Expressing the ridge in units of
max|diag| rather than max(diag) means an all-negative block is now reachable at
all -- it used to be rejected before any factorisation was attempted -- but
since ridging such a block essentially never helps, the guard then drops it
again. The net behaviour for those blocks is the same as before; what changed is
that it is now arrived at by measurement, with a logged reason, rather than by a
scale test that was mislabelled (it always said "max|diag|"), and that the same
protection now covers every other block too. The spectral transform is what
actually rescues them: it reaches 1 on the same block.

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

# A block is dropped if whitening makes its true condition number worse by more
# than this factor. Not zero, so that a tie or a rounding difference does not
# churn a block in and out; small, because there is no reason to accept a
# transform that measurably degrades what it was applied to. See
# NEVER MAKE IT WORSE in the module docstring.
DEGRADE_TOLERANCE = 0.01

# Above this the block is singular to working precision and its condition
# number is not a measurement any more, so the degradation guard does not act
# on it: "worse" is not decidable between two values of order 1/eps, and
# ridging a rank-deficient block into shape is the ridge's original purpose
# (tests/test_preconditioner.py::test_rank_deficient_block_is_ridged_into_shape).
SINGULAR_COND = 1.0 / np.finfo(np.float64).eps


class Block:
    """One factorised group of parameters: indices, L, and a cached L^-1."""

    def __init__(
        self,
        idx,
        chol,
        cond_before=None,
        cond_after=None,
        label="",
        corr_before=None,
    ):
        self.idx = np.asarray(idx, dtype=np.int64)
        self.chol = np.asarray(chol, dtype=np.float64)
        if self.chol.shape != (self.idx.size,) * 2:
            raise ValueError(
                f"chol shape {self.chol.shape} does not match block size {self.idx.size}"
            )
        # cond_* are TRUE condition numbers, comparable at both ends.
        # corr_before is the scale-free degeneracy of the incoming block; there
        # is deliberately no corr_after (module docstring).
        self.cond_before = cond_before
        self.cond_after = cond_after
        self.corr_before = corr_before
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
    def from_hessian(
        cls,
        hess,
        theta_ref,
        index_blocks,
        ridge=1e-8,
        max_tries=4,
        names=None,
        transform="ridge",
    ):
        """Build from a reference Hessian, one factorisation per index block.

        ``index_blocks`` is a list of index arrays (a single array is accepted
        and treated as one block). Each block is symmetrised and then whitened
        by ``transform``:

        - ``'ridge'`` (the default) factorises ``H + eps*I``, escalating eps
          until the Cholesky succeeds;
        - ``'spectral'`` factorises ``|H| = Q |Lambda| Q^T``, giving each
          eigendirection its own scale.

        See :meth:`_factorise` for the dispatch and WHITENING EACH BLOCK in the
        module docstring for which to prefer. A block that cannot be factorised
        is dropped: the others still apply, and a preconditioner must never
        break a fit.
        """
        if isinstance(index_blocks, np.ndarray) or (
            index_blocks and np.isscalar(index_blocks[0])
        ):
            index_blocks = [index_blocks]
        blocks = []
        for spec in index_blocks:
            label, idx = ("", spec) if not isinstance(spec, tuple) else spec
            blk = cls._factorise(
                hess,
                np.asarray(idx, dtype=np.int64),
                ridge,
                max_tries,
                label,
                names=names,
                transform=transform,
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
        # Report the conditioning ACHIEVED, not a hardcoded 1. The literal "-> 1"
        # this used to print claimed the transform had whitened every block
        # perfectly, which is only true of the ridged matrix; measured on the
        # un-ridged block (Block.cond_after) it can be worse than where it
        # started -- seen at 1.8e+03 -> 1e+04 on a block whose ridge was forced
        # up to |lam_min| by a large negative eigenvalue. A summary that cannot
        # express that is worse than none, because it is read as success.
        #
        # The arrow is the TRUE condition number at both ends, so it compares
        # like with like; the scale-free degeneracy is a separate clause rather
        # than the other end of an arrow. See WHAT THE NUMBERS MEAN.
        #
        # Paired, so the two medians are guaranteed to be over the same set of
        # blocks. Both are set together on every success path, but filtering
        # them independently would not enforce that.
        pairs = [
            (b.cond_before, b.cond_after)
            for b in blocks
            if b.cond_before is not None and b.cond_after is not None
        ]
        summary = f"Preconditioned {len(blocks)} of {n_req} block(s), {npar} parameters"
        if pairs:
            conds = [c for c, _ in pairs]
            conds_after = [c for _, c in pairs]
            summary += (
                f"; condition number median {np.median(conds):.3g}, "
                f"worst {max(conds):.3g}"
                f" -> median {np.median(conds_after):.3g}, "
                f"worst {max(conds_after):.3g}"
            )
            if max(conds_after) > max(conds):
                summary += " (WORSE than unpreconditioned)"
            summary += " at the reference point"
            corrs = [b.corr_before for b in blocks if b.corr_before is not None]
            if corrs:
                summary += (
                    f"; of which degeneracy (scale-free) median "
                    f"{np.median(corrs):.3g}, worst {max(corrs):.3g}"
                )
        if len(blocks) < n_req:
            summary += f" ({n_req - len(blocks)} block(s) not factorisable, skipped)"
        logger.info(summary)
        return cls(theta_ref, blocks)

    @staticmethod
    def _factorise(
        hess, idx, ridge, max_tries, label="", names=None, transform="ridge"
    ):
        """Whiten one block: shared preparation, then the requested transform.

        Returns a :class:`Block`, or None if the block is unusable. The
        preparation (empty and finite checks, symmetrisation, and the
        conditioning it started at) is common to both transforms; only the way
        an L is obtained from an indefinite block differs. See
        --preconditionTransform, and WHITENING EACH BLOCK in the module
        docstring for which to prefer.
        """
        idx = np.asarray(idx, dtype=np.int64)
        tag = f"{label} " if label else ""
        if idx.size == 0:
            logger.warning(f"Preconditioning block {tag}is empty; skipping.")
            return None

        block = np.asarray(hess, dtype=np.float64)[np.ix_(idx, idx)]
        # symmetrise: the autodiff Hessian is symmetric only up to roundoff
        block = 0.5 * (block + block.T)
        if not np.all(np.isfinite(block)):
            logger.warning(
                f"Preconditioning block {tag}is non-finite; skipping."
                f"  [{_describe(idx, names)}]"
            )
            return None

        cond_before = _cond_true(block)
        corr_before = _cond_corr(block)

        if transform == "ridge":
            blk = Preconditioner._factorise_ridge(
                block,
                idx,
                cond_before,
                corr_before,
                ridge,
                max_tries,
                label=label,
                names=names,
            )
        elif transform == "spectral":
            blk = Preconditioner._factorise_spectral(
                block, idx, cond_before, corr_before, label=label, names=names
            )
        else:
            raise ValueError(f"unknown preconditioner transform {transform!r}")

        # NEVER MAKE IT WORSE. from_hessian already drops a block that cannot be
        # factorised, on the principle that a preconditioner must never break a
        # fit -- but that was applied only to factorisation FAILING, never to it
        # succeeding and making things worse. Whitening a block whose whole
        # spectrum is negative is exactly that case: the ridge has to be at
        # least |lam_min| ~ max|diag|, so it swamps every direction in the
        # block. Measured over 7094 such blocks across four off-diagonal
        # strengths, the ridge degraded the true condition number in 100% of
        # them, median 10.8x, and improved NONE.
        #
        # Dropping the block restores exactly the unpreconditioned behaviour
        # there, so the fit can never come out worse than with --precondition
        # off on that block, and the guard covers the pre-existing ridge path
        # too rather than only the blocks the max|diag| fix newly reaches.
        #
        # It has to key off the TRUE condition number. The correlation number
        # catches most of these but is blind to the clean case: a diagonal block
        # has the identity as its correlation matrix both before and after, so
        # diag(-7.5e3, -6.3e4, -2.1e3) reports 1 -> 1 while the truth is
        # 30 -> 320. It is also why the guard must not run on corr_before.
        # Not applied to an already-singular block: see SINGULAR_COND. There
        # the true condition number is noise at the 1e16 level -- one such
        # block measures 5.28e+16 -> 7.32e+16, a "degradation" of 1.39x between
        # two numbers past 1/eps -- while the correlation number shows the ridge
        # genuinely helping it (3.1e+16 -> 1.4e+12). That is the mirror image of
        # the all-negative case above, and the reason the two metrics are both
        # kept rather than one being declared correct.
        if (
            blk is not None
            and blk.cond_before is not None
            and blk.cond_after is not None
            and blk.cond_before < SINGULAR_COND
            and blk.cond_after > blk.cond_before * (1.0 + DEGRADE_TOLERANCE)
        ):
            logger.warning(
                f"Preconditioning block {tag}would make the conditioning WORSE "
                f"({blk.cond_before:.3g} -> {blk.cond_after:.3g}); leaving it "
                f"unpreconditioned.  [{_describe(idx, names)}]"
            )
            return None
        return blk

    @staticmethod
    def _factorise_ridge(
        block, idx, cond_before, corr_before, ridge, max_tries, label="", names=None
    ):
        """Ridge path: factorise ``B + eps*I``, escalating eps until it succeeds.

        rabbit's original transform, kept as the default. A single scalar ridge
        cannot serve a block whose curvatures span orders of magnitude -- see
        :meth:`_factorise_spectral` and the note on --preconditionTransform --
        so prefer spectral for heterogeneous blocks.
        """
        tag = f"{label} " if label else ""

        # The unit the ridge is expressed in, max|diag| rather than max(diag).
        # It is only a scale: nothing in the schedule below needs it to come
        # from a positive curvature, and requiring that used to skip every
        # block with an all-negative diagonal outright -- 21 of 34 on one real
        # fit, i.e. exactly the blocks most in need of help. Such a block fails
        # the first Cholesky, then the spectrum branch at itry == 1 sizes the
        # ridge from |lam_min| and it factorises (measured on diag(-7.5e3,
        # -6.3e4, -2.1e3): skipped before, true cond_after 320 after). It is
        # not as good as spectral, which reaches 1 on the same block, but a
        # usable transform beats none. The log line already called this
        # "max|diag|".
        diag = np.diag(block)
        scale = float(np.max(np.abs(diag))) if diag.size else 0.0
        if not np.isfinite(scale) or scale <= 0.0:
            logger.warning(
                f"Preconditioning block {tag}has an all-zero diagonal "
                f"(max|diag| = {scale}); skipping.  [{_describe(idx, names)}]"
            )
            return None

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
            cond_after = _cond_true(_whiten(chol, block))
            logger.debug(
                f"Preconditioning {tag}block of {idx.size} parameters from the "
                f"reference Hessian (ridge {eps:.3g} x max|diag|): condition "
                f"number {cond_before:.3g} -> {cond_after:.3g}, of which "
                f"degeneracy {corr_before:.3g}, at the reference point"
                f"  [{_describe(idx, names)}]"
            )
            return Block(
                idx, chol, cond_before, cond_after, label, corr_before=corr_before
            )

        logger.warning(
            f"Preconditioning block {tag}is not factorisable; the largest ridge "
            f"tried was {max(schedule):.3g} x max|diag|. Skipping this block."
            f"  [{_describe(idx, names)}]"
        )
        return None

    @staticmethod
    def _factorise_spectral(block, idx, cond_before, corr_before, label="", names=None):
        """Spectral path: whiten by the Cholesky of ``|B| = Q |Lambda| Q^T``.

        Gives every eigendirection its own scale, so unlike the ridge it serves
        a block whose curvatures span orders of magnitude, and it can start on
        a block with no positive curvature at all. Identical in result to the
        plain Cholesky where the block is positive definite. Takes |Lambda|
        rather than clipping, which preserves the sign of each direction --
        trust-krylov needs the negative ones. See the comment below and
        WHITENING EACH BLOCK in the module docstring.
        """
        tag = f"{label} " if label else ""

        # SPECTRAL whitening: factorise |B| = Q |Lambda| Q^T rather than
        # ridging B into definiteness.
        #
        # Why not a ridge. A ridge is ONE number and it has to be at least
        # |lam_min| to restore definiteness, so in a block whose curvatures span
        # orders of magnitude it swamps every direction softer than itself.
        # Those directions come back out of the whitening as near-null: measured
        # on a real 3-parameter case with curvatures (3.3e9, -8.7e6, 1), the
        # O(1) direction landed at 7e-08 and the block's true condition number
        # went 3.3e+09 -> 1.44e+08, where the spectral transform gives 1. That
        # is not a corner case -- it is what happens whenever a constrained
        # nuisance shares a block with an unconstrained parameter, and it made
        # the difference between a 3720-parameter fit stalling and converging.
        #
        # Taking |Lambda| keeps the SIGN of each direction in the new
        # coordinates (the true Hessian maps to diag(+1, -1, +1) above), which
        # matters because trust-krylov exploits negative curvature to escape
        # saddles; flattening it to +1 would throw that away, and is why the
        # Gauss-Newton reference matrix does badly here (see
        # Fitter._reference_matrix).
        #
        # For a positive-definite block this is IDENTICAL in result to the
        # plain Cholesky -- both give exactly the identity -- so nothing is lost
        # on the easy blocks; it only costs more (measured 11x at m=200, 44x at
        # m=2112, i.e. 1.7 s once per build, against a fit of tens of minutes).
        try:
            w, Q = np.linalg.eigh(block)
        except np.linalg.LinAlgError as ex:
            logger.warning(
                f"Preconditioning block {tag}eigendecomposition failed ({ex}); "
                f"skipping.  [{_describe(idx, names)}]"
            )
            return None

        aw = np.abs(w)
        wmax = float(np.max(aw))
        if not np.isfinite(wmax) or wmax <= 0.0:
            logger.warning(
                f"Preconditioning block {tag}has an all-zero spectrum; "
                f"skipping.  [{_describe(idx, names)}]"
            )
            return None
        # Floor the numerically-null directions. A direction indistinguishable
        # from zero curvature has no scale to whiten to, and 1/sqrt(~0) would
        # hand the minimizer an enormous spurious step.
        #
        # The threshold is the standard numerical-RANK cutoff, eps * m * wmax
        # (what pinv and lstsq use), NOT the `ridge` parameter. That distinction
        # is the whole point: whether a direction is resolvable is a question
        # about floating point, not about physics. Reusing `ridge` here was
        # tried and is wrong -- at its 1e-8 default and wmax = 3.3e9 the floor
        # lands at 33, which is ABOVE a genuine curvature of 1.0 in the very
        # block this transform exists for, so a physical direction gets flattened
        # and the block comes out at condition number 33 instead of 1
        # (tests/test_preconditioner_spectral.py pins exactly this).
        #
        # What this bounds is 1/sqrt(floor), so the protection is finite, not
        # absolute: on diag(1, 1e-18) it takes max|L^-1| from 1e9 to 4.7e7, a
        # factor of ~21. That is a step the trust region can still reject
        # rather than one that overflows it.
        floor = float(np.finfo(np.float64).eps) * float(idx.size) * wmax
        n_floored = int(np.count_nonzero(aw < floor))
        aw = np.maximum(aw, floor if floor > 0 else np.finfo(float).tiny)

        absH = (Q * aw) @ Q.T
        absH = 0.5 * (absH + absH.T)
        # Cholesky of |B|, rather than the symmetric square root Q sqrt(|Lambda|)
        # which is already in hand: Block caches an explicit L^-1 but falls back
        # to a triangular solve when it cannot be formed (see _apply_T), so L
        # has to be lower triangular. The cost is a second O(m^3), and it can
        # itself fail on a floored spectrum -- cond(|B|) is then ~1/(eps*m),
        # i.e. ~4e14 -- which is why this is caught and the block skipped rather
        # than assumed to succeed.
        try:
            chol = scipy.linalg.cholesky(absH, lower=True)
        except scipy.linalg.LinAlgError as ex:
            logger.warning(
                f"Preconditioning block {tag}Cholesky of |H| failed ({ex}); "
                f"skipping.  [{_describe(idx, names)}]"
            )
            return None

        # Conditioning actually achieved, on the UN-modified block: using |B|
        # here would return 1 by construction and measure nothing.
        #
        # On this path the answer is 1 by a stronger argument, and it is worth
        # knowing when reading the log: B and |B| share eigenvectors, so with
        # |B| = L L^T the congruence L^-1 B L^-T sends Lambda to its own
        # signature, i.e. tb is symmetric AND orthogonal (tb^2 = I) and true
        # kappa is identically 1 at any block size. It is still measured rather
        # than asserted, because n_floored > 0 breaks exactly that identity --
        # a floored direction is one the congruence no longer preserves.
        cond_after = _cond_true(_whiten(chol, block))
        n_neg = int(np.count_nonzero(w < 0.0))
        logger.debug(
            f"Preconditioning {tag}block of {idx.size} parameters by spectral "
            f"whitening of |H| (lam in [{w[0]:.3g}, {w[-1]:.3g}], {n_neg} "
            f"negative, {n_floored} floored): condition number "
            f"{cond_before:.3g} -> {cond_after:.3g}, of which degeneracy "
            f"{corr_before:.3g}, at the reference point"
            f"  [{_describe(idx, names)}]"
        )
        return Block(idx, chol, cond_before, cond_after, label, corr_before=corr_before)

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


def _describe(idx, names=None, limit=8):
    """Readable membership for a block: names when available, else indices."""
    idx = np.asarray(idx).ravel()
    if names is None:
        out = [str(int(i)) for i in idx[:limit]]
    else:
        out = [
            str(names[int(i)]) if int(i) < len(names) else str(int(i))
            for i in idx[:limit]
        ]
    if idx.size > limit:
        out.append(f"... +{idx.size - limit} more")
    return ", ".join(out)


def _whiten(chol, block):
    """``L^-1 B L^-T``: the block as the minimizer sees it after the transform."""
    t = scipy.linalg.solve_triangular(chol, block, lower=True, trans="N")
    return scipy.linalg.solve_triangular(chol, t.T, lower=True, trans="N").T


def _cond_true(mat):
    """Condition number of ``mat`` itself: what the minimizer actually feels.

    The Krylov inner solve converges in a number of Hessian-vector products
    growing like sqrt of this, so it -- not the correlation number below -- is
    the quantity a transform has to reduce. Reported for both ends of the
    before/after pair, since after the transform the parameter scales are
    precisely what was chosen rather than an arbitrary unit convention.
    """
    mat = np.asarray(mat, dtype=np.float64)
    if mat.size == 0 or not np.all(np.isfinite(mat)):
        return np.inf
    try:
        sv = np.linalg.svd(mat, compute_uv=False)
    except np.linalg.LinAlgError:
        return np.inf
    return float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf


def _cond_corr(mat):
    """Condition number of the *correlation* matrix of ``mat``.

    Scale-invariant, so it measures genuine degeneracy rather than a mismatch
    of units between parameters. Reported for the block as it arrives, where
    the units genuinely are arbitrary; NOT for the whitened block, where it is
    an artefact -- see WHAT THE NUMBERS MEAN in the module docstring.
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
