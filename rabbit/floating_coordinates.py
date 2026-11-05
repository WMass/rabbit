"""Minimizer coordinates over the floating parameters only (FloatingCoordinates)."""

import numpy as np
import scipy
import tensorflow as tf


class FloatingCoordinates:
    """A dense-Hessian minimizer's coordinates: the floating parameters only.

    A frozen parameter is held by tf.stop_gradient alone, so its gradient
    entry and its Hessian row and column are exactly zero while the loss
    value still depends on it. The Krylov and gradient minimizers never leave
    the span of gradients and Hessian-vector products and so never move it.
    The dense-Hessian ones do: the zero rows make H singular, trust-exact's
    More-Sorensen iteration then never reaches lambda = 0 and ends in its
    hard case, adding a step along the smallest singular vector of
    H + lambda I -- the frozen coordinate. The model predicts no change along
    it, the loss does change, and a step walking the parameter back towards
    the best fit is accepted (nll_scan came out flat, every point at maxiter).
    dogleg's Cholesky of the singular H fails and the fit does not move.

    These minimizers therefore see the floating entries of the internal
    parameter vector and the floating block of the gradient and Hessian (the
    block edmval_cov inverts); the frozen entries stay at the anchor, the
    internal vector the leg starts from. The preconditioner never mixes a
    frozen parameter into a block (select_index_blocks), so its internal
    coordinate is the physical one shifted by theta_ref. ``floating=None``
    makes every method the identity.
    """

    def __init__(self, floating=None):
        self.idx = None if floating is None else np.asarray(floating, dtype=np.int64)
        self.anchor = None

    def set_anchor(self, y):
        """The full internal vector whose frozen entries embed() keeps."""
        if self.idx is not None:
            self.anchor = np.array(y, dtype=np.float64, copy=True)

    def embed(self, y):
        """Minimizer coordinates -> full internal vector."""
        if self.idx is None:
            return y
        out = self.anchor.copy()
        out[self.idx] = y
        return out

    def restrict(self, v):
        """Full vector (numpy, or a tf tensor kept on device) -> floating part."""
        if self.idx is None:
            return v
        if tf.is_tensor(v):
            return tf.gather(v, self.idx)
        return np.asarray(v)[self.idx]

    def restrict_hess(self, hess):
        """Full [n, n] Hessian (numpy or tf) -> its floating block."""
        if self.idx is None:
            return hess
        if tf.is_tensor(hess):
            return tf.gather(tf.gather(hess, self.idx, axis=0), self.idx, axis=1)
        return np.asarray(hess)[np.ix_(self.idx, self.idx)]

    def embed_result(self, res):
        """A minimizer's OptimizeResult -> the full vector: x embedded as by
        embed(), jac and hess zero in the frozen entries, rows and columns (as
        stop_gradient makes them)."""
        if self.idx is None:
            return res
        res = scipy.optimize.OptimizeResult(res)
        res.x = self.embed(res.x)
        n = len(self.anchor)
        if res.get("jac") is not None:
            jac = np.zeros(n)
            jac[self.idx] = np.asarray(res.jac)
            res.jac = jac
        if res.get("hess") is not None:
            hess = np.zeros((n, n))
            hess[np.ix_(self.idx, self.idx)] = np.asarray(res.hess)
            res.hess = hess
        return res

    def callback(self, cb):
        """``cb`` called with the iterate embedded in the full vector."""
        if self.idx is None or cb is None:
            return cb

        # scipy passes an OptimizeResult only to a callback whose single
        # parameter is named intermediate_result (inspect.signature)
        def embedded(intermediate_result):
            res = scipy.optimize.OptimizeResult(intermediate_result)
            res.x = self.embed(intermediate_result.x)
            return cb(res)

        return embedded
