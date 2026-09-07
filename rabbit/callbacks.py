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
    def __init__(self, xv, early_stopping=-1, snapshotter=None):
        self.iiter = 0
        self.xval = xv
        # Optional rabbit.snapshot.Snapshotter. The callback is the only place
        # that sees the accepted iterate every iteration, which is exactly what
        # a snapshot wants: trial points the trust region goes on to reject are
        # not states the fit was ever in.
        self.snapshotter = snapshotter

        self.loss_history = []
        self.time_history = []

        self.t0 = time.time()

        self.early_stopping = early_stopping
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

        if (
            self.early_stopping > 0
            and len(self.loss_history) > self.early_stopping
            and self.loss_history[-self.early_stopping] <= loss
        ):
            self.stopped_early = True
            raise ValueError(
                f"No reduction in loss after {self.early_stopping} iterations, early stopping."
            )

        self.loss_history.append(loss)
        self.time_history.append(elapsed)

        self.xval = intermediate_result.x
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
