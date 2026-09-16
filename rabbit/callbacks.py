"""Minimizer callback and the bookkeeping around a stalled fit.

The callback is what scipy calls once per accepted iteration. Beyond logging it
does two things the fitter relies on: it keeps the last parameter vector, so a
minimizer that raises can be rolled back to the end of the last good iteration,
and it detects a *stall* -- no reduction in loss over a run of iterations.

A stall is worth singling out because it is usually recoverable. scipy's
trust-region methods shrink the trust radius 4x on every rejected step with no
lower bound and hold it in a local variable, so a run of rejections leaves the
minimizer taking infinitesimal steps far from any minimum, with the loss frozen
while the gradient is still large. A fresh minimize() call resets the radius, so
the fitter restarts rather than giving up; these helpers carry the state across
those restarts so the whole thing still reports as one continuous fit.
"""

import time

import numpy as np
from wums import logging

logger = logging.child_logger(__name__)


class FitterCallback:
    def __init__(
        self, xv, early_stopping=-1, snapshotter=None, expand=None, stall_rel_tol=0.0
    ):
        self.iiter = 0
        self.xval = xv
        # The minimiser may be working in a REDUCED coordinate vector (the
        # fitter removes the frozen subspace from the minimisation), while
        # everything that reads `xval` -- the snapshotter, the rollback after
        # an exception, the restart loop -- wants the full one. `expand` maps
        # the iterate back; None when the two coincide.
        self.expand = expand
        # Optional rabbit.snapshot.Snapshotter. The callback is the only place
        # that sees the accepted iterate every iteration, which is exactly what
        # a snapshot wants: trial points the trust region goes on to reject are
        # not states the fit was ever in.
        self.snapshotter = snapshotter

        self.loss_history = []
        self.time_history = []

        self.t0 = time.time()

        self.early_stopping = early_stopping
        # Relative improvement over the --earlyStopping window below which the
        # fit counts as stalled. 0.0, the default, is exactly the original test:
        # `gained <= 0` is `ref - loss <= 0` is `ref <= loss`, so the behaviour
        # of every existing caller is bit-identical.
        #
        # Why make it relative at all. The exact test fires only on LITERALLY no
        # improvement over the window, so a fit that crawls never satisfies it
        # and --maxRestarts never gets a chance to rebuild the preconditioner at
        # the point the fit has actually reached. That is a real gap in the
        # trigger, but it is NOT a bug we have observed: replayed against our
        # own trajectories they keep gaining 0.2-4% per 30 iterations, no
        # threshold up to 1e-3 fires, and the exact test was correctly reporting
        # "not stalled". Do not read this option as a fix for a known stall.
        #
        # And note what it cannot distinguish: "flat to 1e-4 over 20 iterations"
        # is also what APPROACHING A MINIMUM looks like, whereas the exact test
        # only fires once the trust radius has genuinely collapsed. So a
        # non-zero value will tend to fire near a good minimum and spend a full
        # reference-Hessian evaluation (measured 130-424 s) on each restart.
        # RESTART_MIN_IMPROVEMENT bounds the loop and scipy's gtol usually exits
        # first, so it is not unsafe -- but it is a behaviour change, which is
        # why it is off by default.
        # Negative would silently WEAKEN the test -- it would require the loss
        # to get worse before declaring a stall, i.e. a quieter trigger than
        # the default rather than a louder one.
        if float(stall_rel_tol) < 0.0:
            raise ValueError(
                f"stall_rel_tol must be >= 0, got {stall_rel_tol}. A negative "
                "threshold would require the loss to WORSEN before the fit "
                "counts as stalled."
            )
        self.stall_rel_tol = float(stall_rel_tol)
        # set just before raising, so fit() can tell a recoverable stall apart
        # from a genuine error and restart instead of giving up
        self.stopped_early = False

    def __call__(self, intermediate_result):
        loss = intermediate_result.fun

        elapsed = time.time() - self.t0
        prev = self.time_history[-1] if self.time_history else 0.0
        dt = elapsed - prev

        logger.debug(
            f"Iteration {self.iiter}: loss {loss}  "
            f"[dt={dt:.2f}s elapsed={elapsed:.2f}s]"
        )
        if np.isnan(loss):
            raise ValueError(f"Loss value is NaN at iteration {self.iiter}")

        if self.early_stopping > 0 and len(self.loss_history) > self.early_stopping:
            ref = self.loss_history[-self.early_stopping]
            # Scale by |ref| so the threshold means the same thing at loss 1e7
            # and at loss 1e4; ref == 0 falls back to the absolute test.
            #
            # Written as `loss >= ref - budget`, with `budget` short-circuited
            # at tol == 0, so the default is the ORIGINAL predicate for every
            # input including the infinities. `ref - loss <= tol * abs(ref)`
            # is not: with the loss pinned at an infinity both sides are NaN
            # (inf - inf, and 0.0 * inf), every NaN comparison is False, and
            # the stall goes undetected where `ref <= loss` sees inf <= inf and
            # fires. That case is reachable -- __call__ raises on a NaN loss
            # but not an infinite one, and a Poisson term with a non-positive
            # prediction and non-zero data gives exactly +inf -- and it is
            # precisely the collapsed-trust-radius stall --earlyStopping
            # exists to catch, so missing it means running to
            # maxiter = 200 * nparams instead.
            budget = self.stall_rel_tol * abs(ref) if self.stall_rel_tol else 0.0
            if loss >= ref - budget:
                self.stopped_early = True
                if not self.stall_rel_tol:
                    # unchanged wording on the default path: every existing
                    # user sees this string, and "improved only 0 relative" is
                    # an awkward way to say "did not improve"
                    raise ValueError(
                        f"No reduction in loss after {self.early_stopping} "
                        "iterations, early stopping."
                    )
                gained = ref - loss
                how = (
                    f"only {gained / abs(ref):.3g} relative"
                    if ref
                    else f"only {gained:.3g} absolute"
                )
                raise ValueError(
                    f"Loss improved {how} over {self.early_stopping} iterations "
                    f"(threshold {self.stall_rel_tol:.3g}), early stopping."
                )

        self.loss_history.append(loss)
        self.time_history.append(elapsed)

        self.xval = (
            intermediate_result.x
            if self.expand is None
            else self.expand(intermediate_result.x)
        )
        self.iiter += 1

        # After the update, so the snapshot and the loss recorded with it are
        # the same iterate. Placed after the early-stopping check too: that
        # path raises, and fit() snapshots on the way out.
        if self.snapshotter is not None:
            self.snapshotter.maybe_save(
                self.xval, elapsed, iteration=self.iiter, loss=float(loss)
            )


# Relative loss improvement below which a restart counts as having bought
# nothing. Loss values here are O(1e4), so float64 cancellation puts genuine
# improvements no finer than ~1e-9 relative.
RESTART_MIN_IMPROVEMENT = 1e-9


def merge_callbacks(acc, cb):
    """Fold one restart's callback into the accumulated one.

    Callers read loss_history/time_history/iiter to report on the whole fit, so
    the restarts have to look like a single continuous run. Times are offset by
    the elapsed time already accumulated, since each callback clocks from its
    own construction.
    """
    if acc is None:
        return cb
    offset = acc.time_history[-1] if acc.time_history else 0.0
    acc.loss_history.extend(cb.loss_history)
    acc.time_history.extend(t + offset for t in cb.time_history)
    acc.iiter += cb.iiter
    acc.xval = cb.xval
    acc.stopped_early = cb.stopped_early
    return acc
