"""Profile likelihood at many fixed points at once (vectorized likelihood scans).

``Fitter.nll_scan`` and ``Fitter.nll_scan2D`` evaluate a scan as a sequential Python
loop of independent minimizations, so a scan costs the number of points times the cost
of one profiled fit. Here the negative log-likelihood is written with a leading batch
axis, one row per point, built from the Fitter's own tensors (templates, sparse CSR or
dense logk, the parameter model, the bin-by-bin statistical treatment and the
constraints), and all points are profiled together by a batched Newton minimizer with
the exact Hessian: each iteration is one batched evaluation of the values, gradients
and Hessians of the unconverged points (with sparse inputs the systematic contraction
is one CSR matrix product with a column per point and direction), one batched
eigendecomposition and a backtracking line search on batched likelihood values.
Points start from the linear profile extrapolation of the minimum,
theta_hat + C_theta,r C_rr^-1 (r - r_hat), using the Fitter's covariance.

The batched loss is the sum of independent per-point losses, so its gradient is the
per-point gradient and its batch Jacobian is every point's Hessian. Each point keeps
its own convergence state and frozen parameters.

Supported: Poisson likelihood (no chisqFit / covarianceFit), log-normal systematics
(symmetric or asymmetric interpolation), sparse or dense inputs, bin-by-bin statistics
off or ``gamma`` in ``lite`` mode (the default), Gaussian constraints and any
ParamModel. Not supported (raises NotImplementedError): regularizers, external
likelihood terms, MC-stat de-biasing, blinding, masked channels, other bin-by-bin
modes.

Memory: the batched Hessian of the sparse contraction holds O(batch size x number of
parameters x number of non-zero template entries) intermediates.
"""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops as tf_sparse_csr
from wums import logging

logger = logging.child_logger(__name__)


def _pow2(n, cap):
    """Smallest power of two >= n, at most cap (the padded batch size of a call)."""
    return min(cap, 1 << max(n - 1, 0).bit_length())


@dataclass
class ProfileResult:
    """Profiled points. dnll: NLL - NLL at the Fitter's current x [N]; x: the
    profiled parameter vectors [N, npar]; converged, iterations [N]; edmval and
    grad_max_abs [N]: estimated distance to the minimum and largest gradient entry
    over the profiled parameters at the last Hessian evaluation."""

    dnll: np.ndarray
    x: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray
    edmval: np.ndarray
    grad_max_abs: np.ndarray


class BatchedProfile:
    def __init__(self, fitter):
        self.fitter = f = fitter
        indata = f.indata
        unsupported = []
        if f.chisqFit or f.covarianceFit:
            unsupported.append("chisqFit/covarianceFit")
        if indata.systematic_type != "log_normal":
            unsupported.append(f"systematic_type {indata.systematic_type}")
        if len(f.regularizers):
            unsupported.append("regularizers")
        if f.external_terms:
            unsupported.append("external likelihood terms")
        if f.mcstat_debias_active or f.mcstat_terms or f.mcstat_bb_vref is not None:
            unsupported.append("MC-stat de-biasing")
        if f.blinding.enabled:
            unsupported.append("blinding")
        if indata.nbinsmasked:
            unsupported.append("masked channels")
        bb = f.bbstat
        if bb.enabled and (
            bb.binByBinStatType != "gamma"
            or bb.binByBinStatMode != "lite"
            or bb.dynamic
        ):
            unsupported.append(
                f"bin-by-bin stat {bb.binByBinStatType}/{bb.binByBinStatMode}"
            )
        if unsupported:
            raise NotImplementedError(
                f"BatchedProfile does not support: {', '.join(unsupported)}"
            )

        self.dtype = indata.dtype
        self.nbins = indata.nbins
        self.nproc = indata.nproc
        self.nsyst = indata.nsyst
        self.npoi = f.param_model.npoi
        self.nparams = f.param_model.nparams
        self.npar = int(f.x.shape[0])
        self.sparse = indata.sparse
        if self.sparse:
            self.idx = indata.norm.indices
            self.norm_values = indata.norm.values
            self.bin_idx = self.idx[:, 0]
        self.zv = None
        if bb.enabled:
            self.zv = bb.proc_zero_var_mask
            if self.zv is not None and self.sparse:
                self.zv_entries = tf.gather_nd(self.zv, self.idx)
        self._make_tf_functions()

    # ------------------------------------------------------------------ loss
    def _poi(self, X):
        xpoi = X[:, : self.npoi]
        if self.fitter.param_model.allowNegativeParam:
            return xpoi
        return tf.square(xpoi)

    def _mthetaalpha(self, theta):
        """[nsyst or 2 nsyst, N]: the systematic coefficients of each point
        (Fitter._compute_yields_noBBB)."""
        if self.fitter.indata.symmetric_tensor:
            return tf.transpose(theta)
        twox = 2.0 * theta
        twox2 = twox * twox
        alpha = tf.clip_by_value(
            0.125 * twox * (twox2 * (3.0 * twox2 - 10.0) + 15.0), -1.0, 1.0
        )
        return tf.transpose(tf.concat([theta, theta * alpha], axis=1))

    def _yields(self, X):
        """Expected yields [N, nbins] without the bin-by-bin factors, and their
        zero-variance part [N, nbins] (None without bin-by-bin statistics)."""
        f = self.fitter
        params = tf.concat([self._poi(X), X[:, self.npoi : self.nparams]], axis=1)
        theta = X[:, self.nparams : self.nparams + self.nsyst]
        # [N, 1 or nbins, nproc]
        rnorm = tf.vectorized_map(lambda p: f.param_model.compute(p, False), params)
        rnorm = tf.broadcast_to(rnorm, [tf.shape(X)[0], self.nbins, self.nproc])
        mtheta = self._mthetaalpha(theta)
        zero_var = None
        if self.sparse:
            logsnorm = tf_sparse_csr.matmul(f.logk_csr, mtheta)  # [nnz, N]
            # [N, nnz]
            r = tf.transpose(tf.gather_nd(tf.transpose(rnorm, [1, 2, 0]), self.idx))
            vals = self.norm_values[None, :] * tf.exp(tf.transpose(logsnorm)) * r
            nexp = tf.transpose(
                tf.math.unsorted_segment_sum(
                    tf.transpose(vals), self.bin_idx, self.nbins
                )
            )
            if self.zv is not None:
                zvals = tf.where(self.zv_entries[None, :], vals, tf.zeros_like(vals))
                zero_var = tf.transpose(
                    tf.math.unsorted_segment_sum(
                        tf.transpose(zvals), self.bin_idx, self.nbins
                    )
                )
        else:
            logk = tf.reshape(f.logk[: self.nbins], [self.nbins * self.nproc, -1])
            logsnorm = tf.reshape(
                tf.transpose(logk @ mtheta), [-1, self.nbins, self.nproc]
            )
            norm = f.indata.norm[None, : self.nbins] * tf.exp(logsnorm) * rnorm
            nexp = tf.reduce_sum(norm, axis=-1)
            if self.zv is not None:
                zero_var = tf.reduce_sum(
                    tf.where(self.zv[None, : self.nbins], norm, tf.zeros_like(norm)),
                    axis=-1,
                )
        return nexp, zero_var

    def nll(self, X):
        """Reduced NLL (Fitter.reduced_nll) of every row of X [N, npar]: [N]."""
        f = self.fitter
        nobs, lognobs = tf.convert_to_tensor(f.nobs), tf.convert_to_tensor(f.lognobs)
        nexp, zero_var = self._yields(X)
        lbeta = 0.0
        bb = f.bbstat
        if bb.enabled:
            # Barlow-Beeston lite: the analytic minimum of the gamma-distributed
            # bin-by-bin factor beta of each bin (Fitter._compute_yields_with_beta)
            kstat, betamask = bb.kstat[: self.nbins], bb.betamask[: self.nbins]
            beta0, logbeta0 = bb.beta0[: self.nbins], bb.logbeta0[: self.nbins]
            n_zero = zero_var if zero_var is not None else tf.zeros_like(nexp)
            n_active = nexp - n_zero
            n_safe = tf.where(n_active > 0, n_active, tf.ones_like(n_active))
            a = n_safe * (n_safe + kstat)
            b = n_safe * (n_zero - nobs - beta0 * kstat) + kstat * n_zero
            c = -beta0 * kstat * n_zero
            beta = 0.5 * (-b + tf.sqrt(b**2 - 4.0 * a * c)) / a
            beta = tf.where(betamask, beta0, beta)
            nexp = tf.where(betamask, nexp, n_zero + n_active * beta)
            betasafe = tf.where(beta0 == 0.0, tf.ones_like(beta), beta)
            lbeta = tf.reduce_sum(
                -kstat * beta0 * (tf.math.log(betasafe) - logbeta0)
                + kstat * (beta - beta0),
                axis=-1,
            )
        nexpsafe = tf.where(nobs == 0.0, tf.ones_like(nexp), nexp)
        ln = tf.reduce_sum(
            -nobs * (tf.math.log(nexpsafe) - lognobs) + nexp - nobs, axis=-1
        )
        xphys = tf.concat([self._poi(X), X[:, self.npoi :]], axis=1)
        lc = tf.reduce_sum(f.cw * 0.5 * tf.square(xphys - f.x0), axis=-1)
        return ln + lc + lbeta

    def _make_tf_functions(self):
        def val(X):
            return self.nll(X)

        def val_grad_hess(X):
            # the batched loss is a sum of independent per-row losses: the gradient
            # of the sum is the per-row gradient, and its batch Jacobian is each
            # row's Hessian
            with tf.GradientTape() as t2:
                t2.watch(X)
                with tf.GradientTape() as t1:
                    t1.watch(X)
                    v = self.nll(X)
                    total = tf.reduce_sum(v)
                g = t1.gradient(total, X)
            return v, g, t2.batch_jacobian(g, X)

        self._val_fn = tf.function(val)
        self._val_grad_hess_fn = tf.function(val_grad_hess)

    def _padded(self, fn, X, size):
        """fn on the rows of X padded to ``size`` rows (static shapes: the sparse CSR
        product needs a known number of columns, and one trace serves every call)."""
        n = len(X)
        pad = np.concatenate([X, np.repeat(X[:1], size - n, axis=0)]) if n < size else X
        out = fn(tf.constant(pad, dtype=self.dtype))
        if isinstance(out, (tuple, list)):
            return tuple(t.numpy()[:n] for t in out)
        return out.numpy()[:n]

    # ------------------------------------------------------------- minimizer
    def minimize(
        self,
        X0,
        free,
        max_iter=100,
        edm_tol=1e-6,
        batch_size=8,
        max_halvings=20,
        max_component=5.0,
        negative_tol=1e-8,
        negative_step=1.0,
        fixed=None,
        scale=None,
    ):
        """Minimize every row of X0 [N, npar] over the entries where ``free`` [N,
        npar] is 1.

        Newton's method with the exact Hessian and a backtracking line search,
        batched: each iteration evaluates the value, gradient and Hessian of up to
        ``batch_size`` unconverged rows in one call, takes every row's Newton
        direction from one batched eigendecomposition (negative curvature is
        flipped, flat directions floored, and the step along any eigendirection
        capped at ``max_component``: near-flat directions, e.g. a nuisance that only
        acts on a signal fixed near zero, would otherwise propose huge steps the line
        search has to undo) and backtracks every row with batched likelihood values
        only, which cost a small fraction of a Hessian. A row has converged when its
        estimated distance to the minimum, edm = g^T |H|^-1 g / 2 in units of the
        NLL, is below ``edm_tol`` and its Hessian has no eigenvalue below
        -``negative_tol`` times its largest one. A row that would stop at such a
        saddle point moves ``negative_step`` downhill along the most negative
        curvature direction instead: the gradient vanishes, e.g., on the whole
        hyperplane where a POI stored as a square root (without allowNegativeParam)
        is 0, so Newton steps alone never leave it. A row whose likelihood is not
        finite where it is evaluated (e.g. a negative expected yield) is given up at
        once. Calls are padded to the next power of two up to ``batch_size`` rows, so
        the last unconverged rows do not pay for a full batch.

        Rows are started in order. With ``fixed`` (the columns held fixed in every
        row, at values that differ between rows), a row starts from the solution of
        the nearest converged row (distance over the fixed columns in units of
        ``scale``) where that has a lower NLL than its own starting point.

        Returns (X, nll, converged, iterations, edm, grad_max_abs).
        """
        X = np.array(X0, dtype=float)
        M = np.asarray(free, dtype=float)
        n = len(X)
        val = np.full(n, np.nan)
        edm = np.full(n, np.nan)
        gmax = np.full(n, np.nan)
        converged = np.zeros(n, dtype=bool)
        failed = np.zeros(n, dtype=bool)
        undefined = np.zeros(n, dtype=bool)
        iters = np.zeros(n, dtype=int)
        eye = np.eye(self.npar)
        while True:
            todo = np.where(~converged & ~failed & (iters < max_iter))[0][:batch_size]
            if not len(todo):
                break
            fresh = todo[iters[todo] == 0]
            if fixed is not None and len(fresh) and converged.any():
                self._neighbour_starts(X, fresh, converged, fixed, scale, batch_size)
            size = _pow2(len(todo), batch_size)
            v, g, H = self._padded(self._val_grad_hess_fn, X[todo], size)
            finite = np.isfinite(v) & np.isfinite(g).all(axis=1)
            finite &= np.isfinite(H).all(axis=(1, 2))
            if not finite.all():
                val[todo[~finite]] = v[~finite]
                iters[todo[~finite]] += 1
                undefined[todo[~finite]] = failed[todo[~finite]] = True
                todo, v, g, H = todo[finite], v[finite], g[finite], H[finite]
                if not len(todo):
                    continue
            m = M[todo]
            val[todo] = v
            g = g * m
            H = H * m[:, :, None] * m[:, None, :] + np.einsum(
                "ni,ij->nij", 1.0 - m, eye
            )
            # eigenvalues in ascending order
            w, V = np.linalg.eigh(0.5 * (H + np.swapaxes(H, 1, 2)))
            wmax = np.abs(w).max(axis=1)
            wabs = np.maximum(np.abs(w), 1e-10 * wmax[:, None])
            gv = np.einsum("nij,ni->nj", V, g)
            edm[todo] = 0.5 * np.einsum("nj,nj->n", gv, gv / wabs)
            gmax[todo] = np.abs(g).max(axis=1)
            iters[todo] += 1
            negative = w[:, 0] < -negative_tol * wmax
            small = edm[todo] < edm_tol
            done = small & ~negative
            converged[todo[done]] = True
            if done.all():
                continue
            rows = todo[~done]
            step = np.clip(gv[~done] / wabs[~done], -max_component, max_component)
            p = -np.einsum("nij,nj->ni", V[~done], step)
            nc = (small & negative)[~done]
            if nc.any():
                # a saddle point: downhill along the most negative curvature
                # direction (either way where the gradient has no component along it)
                sign = -np.sign(gv[~done][nc, 0])
                sign[sign == 0] = 1.0
                p[nc] += negative_step * sign[:, None] * V[~done][nc, :, 0]
            slope = np.einsum("ni,ni->n", g[~done], p)
            alpha = np.ones(len(rows))
            searching = np.ones(len(rows), dtype=bool)
            for _ in range(max_halvings):
                trial = X[rows][searching] + alpha[searching, None] * p[searching]
                vt = self._padded(self._val_fn, trial, _pow2(len(trial), batch_size))
                bound = (
                    val[rows][searching] + 1e-4 * alpha[searching] * slope[searching]
                )
                ok = np.isfinite(vt) & (vt <= bound)
                idx = np.where(searching)[0]
                X[rows[idx[ok]]] += alpha[idx[ok], None] * p[idx[ok]]
                searching[idx[ok]] = False
                alpha[idx[~ok]] *= 0.5
                if not searching.any():
                    break
            failed[rows[searching]] = True  # no decrease along the Newton direction
        last = np.where(~converged)[0]
        for i in range(0, len(last), batch_size):
            rows = last[i : i + batch_size]
            val[rows] = self._padded(
                self._val_fn, X[rows], _pow2(len(rows), batch_size)
            )
        if undefined.any():
            logger.warning(
                f"batched profile: the likelihood is not finite at {undefined.sum()} "
                f"of {n} points (e.g. a negative expected yield)"
            )
        if len(last) > undefined.sum():
            logger.warning(
                f"batched profile: {len(last) - undefined.sum()} of {n} points did "
                f"not converge (edm above {edm_tol} after {max_iter} iterations or no "
                "decrease along the Newton direction)"
            )
        return X, val, converged, iters, edm, gmax

    def _values(self, X, batch_size):
        """NLL of every row of X, batch_size rows per call."""
        out = np.empty(len(X))
        for i in range(0, len(X), batch_size):
            rows = X[i : i + batch_size]
            out[i : i + batch_size] = self._padded(
                self._val_fn, rows, _pow2(len(rows), batch_size)
            )
        return out

    def _neighbour_starts(self, X, fresh, solved, fixed, scale, batch_size):
        """Start the rows ``fresh`` of X from the solution of the nearest ``solved``
        row (with their own fixed values) where that has the lower NLL: the batched
        counterpart of the sequential scan's warm start from the previous point."""
        solved = np.where(solved)[0]
        d = (X[fresh][:, fixed][:, None, :] - X[solved][:, fixed][None, :, :]) / scale
        candidates = X[solved[np.argmin((d**2).sum(axis=-1), axis=1)]]
        candidates[:, fixed] = X[fresh][:, fixed]
        current = self._values(X[fresh], batch_size)
        new = self._values(candidates, batch_size)
        better = np.isfinite(new) & ~(new >= current)  # also where current is NaN
        X[fresh[better]] = candidates[better]

    # ------------------------------------------------------------ fixed points
    def profile(self, indices, values, x_start=None, **kwargs):
        """Profile the likelihood with the parameters ``indices`` (k positions in
        Fitter.x) fixed at ``values`` [N, k], given as stored in Fitter.x (a POI
        without allowNegativeParam is stored as the square root of its value).

        Everything else that is not frozen in the Fitter is profiled. Points are
        profiled from the inside out (by their distance to the current Fitter.x,
        assumed to be the minimum). Each starts from x_start [N, npar] if given;
        otherwise from the lower-NLL of the minimum itself and its linear profile
        extrapolation with the Fitter's covariance (the conditional mean of the
        Gaussian approximation, which can be far off where the likelihood is not
        Gaussian), or from the solution of the nearest point already profiled where
        that is lower still. ``kwargs`` go to ``minimize``. Returns a ProfileResult
        (in the order of ``values``) with dnll relative to the NLL at the current
        Fitter.x.
        """
        f = self.fitter
        batch_size = kwargs.get("batch_size", 8)
        idx = [int(i) for i in indices]
        values = np.asarray(values, dtype=float).reshape(-1, len(idx))
        xbest = f.x.numpy()
        C = f.cov.numpy() if f.cov is not None else None
        scale = np.sqrt(np.diag(C)[idx]) if C is not None else np.ones(len(idx))
        if x_start is not None:
            X0 = np.array(x_start, dtype=float)
            X0[:, idx] = values
        else:
            X0 = np.tile(xbest, (len(values), 1))
            X0[:, idx] = values
            if C is not None:
                shift = np.linalg.solve(C[np.ix_(idx, idx)], (values - xbest[idx]).T)
                linear = X0 + (C[:, idx] @ shift).T
                linear[:, idx] = values
                v_linear = self._values(linear, batch_size)
                use = np.isfinite(v_linear) & ~(
                    v_linear >= self._values(X0, batch_size)
                )
                X0[use] = linear[use]
        free = np.ones_like(X0)
        free[:, np.asarray(f.frozen_params_mask.numpy(), dtype=bool)] = 0.0
        free[:, idx] = 0.0
        order = np.argsort((((values - xbest[idx]) / scale) ** 2).sum(axis=1))
        nll_best = float(self._padded(self._val_fn, xbest[None, :], 1)[0])
        out = self.minimize(
            X0[order],
            free[order],
            fixed=None if x_start is not None else idx,
            scale=scale,
            **kwargs,
        )
        X, val, converged, iters, edm, gmax = (a[np.argsort(order)] for a in out)
        return ProfileResult(val - nll_best, X, converged, iters, edm, gmax)

    def fixed_points(self, names, values, x_start=None, **kwargs):
        """``profile`` with the fixed parameters given by name and their physical
        values [N, k] (POI values, not their stored square roots)."""
        f = self.fitter
        values = np.atleast_2d(np.asarray(values, dtype=float))
        parms = f.parms.astype(str)
        idx = [int(np.where(parms == p)[0][0]) for p in names]
        stored = values.copy()
        for j, i in enumerate(idx):
            if i < self.npoi and not f.param_model.allowNegativeParam:
                if (values[:, j] < 0).any():
                    raise ValueError(
                        f"{names[j]} is stored as a square root: it cannot be fixed "
                        "at a negative value"
                    )
                stored[:, j] = np.sqrt(values[:, j])
        return self.profile(idx, stored, x_start, **kwargs)
