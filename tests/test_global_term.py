#!/usr/bin/env python3
"""Joint quadratic (external) + unbinned term sharing parameters by name.

This is the structural test of the unified calibration objective: a global
calibration parameter vector ``theta`` that appears

* in a **quadratic external term** ``g^T theta + 0.5 theta^T H theta`` -- the
  hit-chi2 information of a track fit, already marginalized over the per-track
  parameters (in the CVH case: ``gradv`` / ``hesspackedv`` accumulated over
  candidates, see ``calibration_studies/global_corrections/fit_global_grads.py``),
  and
* in an **unbinned mass term** through the per-candidate Jacobian rows
  ``m_i(theta) = m_i^0 + D_i theta``,

with the two joined only by the parameter *names*.

Everything here has a closed form. The unbinned term is configured with a
single Gaussian ("hit") family, no kernel CF, no background and no positivity
floor, for which

    L_i = 1/(pi sigma_i) Int_0^inf exp(-k t^2 / 2) cos(t delta_i / sigma_i) dt
        = N(delta_i; 0, k sigma_i^2)        (for a t grid long enough)

so the total NLL is

    NLL(alpha, k, theta) = sum_i delta_i^2 / (2 k sigma_i^2)
                         + n/2 log(2 pi k) + sum_i log sigma_i
                         + g^T theta_e + 0.5 theta_e^T H theta_e + const
                         + Gaussian priors

with ``delta_i = mobs_i - alpha * unit * m_ref - (D theta)_i``. For fixed ``k``
this is quadratic in ``(alpha, theta)``, so the reference minimum is a linear
solve inside a 1D minimization over ``k``, and the reference covariance is the
analytic Hessian -- both exact to round-off. Nothing in the test depends on
external data files.

Run::

    python tests/test_global_term.py
"""

import os
import sys
import tempfile

import hist
import numpy as np
import scipy.optimize
import tensorflow as tf

from rabbit import fitter, inputdata, tensorwriter, unbinned
from rabbit.param_models.helpers import load_models

M_REF = 3.0969
ALPHA_UNIT = unbinned.ALPHA_UNIT
NT = 4096
TMAX = 30.0


# ---------------------------------------------------------------------------
# the toy problem
# ---------------------------------------------------------------------------
class Toy:
    """Random but reproducible joint problem, plus its analytic solution.

    ``theta`` splits into ``nshared`` parameters that both terms see and
    ``nextra`` that only the quadratic term sees (the case a card has to
    handle when the mass term constrains fewer directions than the hit term).
    """

    def __init__(self, n=600, nshared=4, nextra=2, seed=7, priors=None):
        rng = np.random.default_rng(seed)
        self.n = n
        self.nshared = nshared
        self.nextra = nextra
        self.ntheta = nshared + nextra
        self.names = [f"theta{j}" for j in range(self.ntheta)]
        # the mass term sees only the first nshared; the rest exist solely in
        # the quadratic term and must be declared by ExternalParams
        self.shared = self.names[:nshared]
        self.extra = self.names[nshared:]

        self.sigma = rng.uniform(0.020, 0.045, n)
        self.mobs = rng.normal(0.0005, 1.0, n) * self.sigma
        self.tgrid = np.linspace(0.0, TMAX, NT)
        self.vgf = np.ones(n)

        # sparse D: ~3 non-zero shared columns per candidate, entries of order
        # the per-mode mass response of a J/psi candidate (few MeV per unit)
        rows, cols, vals = [], [], []
        for i in range(n):
            for c in rng.choice(nshared, size=3, replace=False):
                rows.append(i)
                cols.append(int(c))
                vals.append(rng.normal(0.0, 3e-3))
        self.jac_idx = np.stack([rows, cols], axis=1).astype(np.int64)
        self.jac_val = np.asarray(vals, dtype=np.float64)
        self.D = np.zeros((n, self.ntheta))
        self.D[self.jac_idx[:, 0], self.jac_idx[:, 1]] = self.jac_val

        # a positive-definite external Hessian (NLL convention) with a
        # gradient that does not point at the origin
        L = rng.normal(0.0, 1.0, (self.ntheta, self.ntheta))
        self.H = L @ L.T + self.ntheta * np.eye(self.ntheta)
        self.g = rng.normal(0.0, 3.0, self.ntheta)

        # Gaussian priors, {name: (mean, sigma)}
        self.priors = dict(priors or {})

        # design matrix of the linear part: u = [alpha, theta]
        self.A = np.concatenate([np.full((n, 1), ALPHA_UNIT * M_REF), self.D], axis=1)
        self.W = 1.0 / self.sigma**2

    # -- reference ---------------------------------------------------------
    def _prior_arrays(self, order):
        """(cw, x0) over the parameter vector in the given name order."""
        cw = np.zeros(len(order))
        x0 = np.zeros(len(order))
        for i, name in enumerate(order):
            if name in self.priors:
                mean, sig = self.priors[name]
                cw[i] = 1.0 / sig**2
                x0[i] = mean
        return cw, x0

    def nll(self, alpha, k, theta, with_ext=True, with_mass=True):
        val = 0.0
        if with_mass:
            r = self.mobs - self.A @ np.concatenate([[alpha], theta])
            val += 0.5 * np.sum(self.W * r**2) / k
            val += 0.5 * self.n * np.log(2.0 * np.pi * k)
            val += np.sum(np.log(self.sigma))
        if with_ext:
            val += self.g @ theta + 0.5 * theta @ self.H @ theta
        names = ["alpha", "k"] + self.names
        cw, x0 = self._prior_arrays(names)
        vec = np.concatenate([[alpha, k], theta])
        val += 0.5 * np.sum(cw * (vec - x0) ** 2)
        return val

    def solve(self, with_ext=True, with_mass=True, fixed=None):
        """Exact minimum and covariance of the analytic NLL.

        ``fixed`` holds ``{name: value}`` for parameters to profile *around*
        rather than minimize over (the frozen-parameter case). Returns
        ``(x, cov, nll)`` with ``x = [alpha, k, theta...]``.
        """
        fixed = dict(fixed or {})
        nu = 1 + self.ntheta
        names = ["alpha", "k"] + self.names
        u_names = ["alpha"] + self.names
        cw, x0 = self._prior_arrays(names)
        cw_u = np.concatenate([cw[:1], cw[2:]])
        x0_u = np.concatenate([x0[:1], x0[2:]])
        Q = np.zeros((nu, nu))
        G = np.zeros(nu)
        if with_ext:
            Q[1:, 1:] += self.H
            G[:] += np.concatenate([[0.0], self.g])
        Q += np.diag(cw_u)
        G += -cw_u * x0_u
        AWA = self.A.T @ (self.W[:, None] * self.A)
        AWm = self.A.T @ (self.W * self.mobs)

        ufix = np.zeros(nu)
        free = []
        for i, nm in enumerate(u_names):
            if nm in fixed:
                ufix[i] = fixed[nm]
            else:
                free.append(i)
        free = np.asarray(free, dtype=int)

        def u_of_k(k):
            u = ufix.copy()
            if not with_mass:
                # alpha is touched by nothing here: solve the theta block only
                fr = free[free > 0]
                rhs = -G[fr] - Q[np.ix_(fr, free[free == 0])].sum(axis=1) * 0.0
                rhs = rhs - Q[np.ix_(fr, np.arange(nu))] @ ufix
                u[fr] = np.linalg.solve(
                    Q[np.ix_(fr, fr)], rhs + Q[np.ix_(fr, fr)] @ ufix[fr]
                )
                return u
            M = AWA / k + Q
            rhs = AWm / k - G - M @ ufix + M[:, free] @ ufix[free]
            u[free] = np.linalg.solve(M[np.ix_(free, free)], rhs[free])
            return u

        def f(k):
            u = u_of_k(k)
            return self.nll(u[0], k, u[1:], with_ext, with_mass)

        if with_mass and "k" not in fixed:
            res = scipy.optimize.minimize_scalar(
                f, bracket=(0.5, 1.0, 2.0), method="brent", options={"xtol": 1e-14}
            )
            k = float(res.x)
        else:
            k = float(fixed.get("k", 1.0))
        u = u_of_k(k)
        alpha, theta = u[0], u[1:]

        # analytic Hessian in the order [alpha, k, theta...]
        npar = 2 + self.ntheta
        Hf = np.zeros((npar, npar))
        idx_u = np.array([0] + list(range(2, npar)))
        if with_mass:
            r = self.mobs - self.A @ u
            Hf[np.ix_(idx_u, idx_u)] += AWA / k
            duk = (self.A.T @ (self.W * r)) / k**2
            Hf[idx_u, 1] += duk
            Hf[1, idx_u] += duk
            Hf[1, 1] += np.sum(self.W * r**2) / k**3 - 0.5 * self.n / k**2
        if with_ext:
            Hf[np.ix_(idx_u[1:], idx_u[1:])] += self.H
        Hf += np.diag(cw)
        if not with_mass:
            # alpha and k are unconstrained by anything here: report them as
            # fixed so the covariance of the theta block stays well defined
            Hf[0, 0] = 1.0
            Hf[1, 1] = 1.0
        cov = np.linalg.inv(Hf)
        x = np.concatenate([[alpha, k], theta])
        return x, cov, self.nll(alpha, k, theta, with_ext, with_mass)

    # -- card --------------------------------------------------------------
    def build_term(self):
        return unbinned.MassCFTerm(
            "mass",
            sigma=self.sigma,
            mobs=self.mobs,
            tgrid=self.tgrid,
            families=[{"name": "hit", "param": "k", "kind": "gauss"}],
            vgf=self.vgf,
            m_ref=M_REF,
            scale_param="alpha",
            bkg_frac=0.0,
            floor="none",
            chunk=256,
            jac=(self.jac_idx, self.jac_val, (self.n, self.nshared)),
            jac_params=self.shared,
            channel="toy",
        )

    def hists(self, names=None):
        """(grad, hess) histograms of the external term over ``names``."""
        names = list(names if names is not None else self.names)
        keep = [self.names.index(nm) for nm in names]
        ax0 = hist.axis.StrCategory(names, name="params0")
        ax1 = hist.axis.StrCategory(names, name="params1")
        hg = hist.Hist(hist.axis.StrCategory(names, name="params"))
        hg.values()[...] = self.g[keep]
        hh = hist.Hist(ax0, ax1)
        hh.values()[...] = self.H[np.ix_(keep, keep)]
        return hg, hh

    def param_bundle(self, names):
        """ExternalParams auxiliary bundle for the given parameter names."""
        sig = [self.priors.get(nm, (0.0, np.nan))[1] for nm in names]
        mean = [self.priors.get(nm, (0.0, np.nan))[0] for nm in names]
        return {
            "params": list(names),
            "defaults": np.zeros(len(names)),
            "prior_sigmas": np.asarray(sig, dtype=np.float64),
            "prior_means": np.asarray(mean, dtype=np.float64),
            "is_poi": np.zeros(len(names), dtype=np.int64),
        }

    def write_card(self, path, with_ext=True, with_mass=True):
        writer = tensorwriter.TensorWriter()
        writer.add_dummy_channel(name="toy_dummy")
        declared = []
        if with_mass:
            term = self.build_term()
            defaults = []
            sigmas = []
            means = []
            for p in term.param_names:
                defaults.append(1.0 if p == "k" else 0.0)
                mean, sig = self.priors.get(p, (0.0, np.nan))
                sigmas.append(sig)
                means.append(mean if np.isfinite(sig) else defaults[-1])
            writer.add_unbinned_term(
                "mass",
                term.config(),
                term.param_names,
                {
                    "sigma": self.sigma,
                    "mobs": self.mobs,
                    "vgf": self.vgf,
                    "tgrid": self.tgrid,
                    "jac_indices": self.jac_idx,
                    "jac_values": self.jac_val,
                    "jac_shape": np.array([self.n, self.nshared], dtype=np.int64),
                },
                param_defaults=defaults,
                param_prior_sigmas=sigmas,
                param_prior_means=means,
                param_is_poi=[1 if p == "alpha" else 0 for p in term.param_names],
            )
            declared = list(term.param_names)
        if with_ext:
            hg, hh = self.hists()
            writer.add_external_likelihood_term(grad=hg, hess=hh, name="hitchi2")
        undeclared = [nm for nm in self.names if nm not in declared]
        if undeclared:
            writer.add_auxiliary("global_params", self.param_bundle(undeclared))
        folder = os.path.dirname(os.path.abspath(path)) or "."
        name = os.path.basename(path)
        if name.endswith(".hdf5"):
            name = name[: -len(".hdf5")]
        writer.write(outfolder=folder, outfilename=name)
        models = []
        if with_mass:
            models.append(["UnbinnedParams"])
        if undeclared:
            models.append(["ExternalParams", "bundle:global_params"])
        return os.path.join(folder, name + ".hdf5"), models


# ---------------------------------------------------------------------------
# fitting helpers
# ---------------------------------------------------------------------------
class Options:
    def __init__(self, **kwargs):
        defaults = dict(
            earlyStopping=-1,
            noBinByBinStat=True,
            binByBinStatMode="lite",
            binByBinStatType="automatic",
            covarianceFit=False,
            chisqFit=False,
            diagnostics=False,
            minimizerMethod="trust-exact",
            prefitUnconstrainedNuisanceUncertainty=0.0,
            freezeParameters=[],
            setConstraintMinimum=[],
            unblind=[],
            blindingGroup=[],
        )
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


def make_fitter(filename, models, **opts):
    indata = inputdata.FitInputData(filename)
    pm = load_models(models, indata)
    f = fitter.Fitter(indata, pm, Options(**opts))
    f.set_nobs(f.indata.data_obs)
    return f


def cov_from_fitter(f):
    """Postfit covariance = inverse of the NLL Hessian at the minimum.

    ``Fitter.cov`` is only *allocated* in ``__init__``; it is filled by the
    output stage of ``rabbit_fit.py``, not by ``minimize()``.
    """
    _, _, hess = f.loss_val_grad_hess()
    return np.linalg.inv(hess.numpy())


def fit_and_read(filename, models, order, **opts):
    f = make_fitter(filename, models, **opts)
    f.minimize()
    parms = f.parms.astype(str)
    x = f.x.numpy()
    val = {nm: float(x[np.where(parms == nm)[0][0]]) for nm in order}
    cov = cov_from_fitter(f)
    err = {
        nm: float(np.sqrt(cov[i, i]))
        for nm, i in ((nm, int(np.where(parms == nm)[0][0])) for nm in order)
    }
    return f, val, err


def report(label, got, ref, tol):
    ok = True
    print(f"  {label}")
    for nm in ref:
        d = abs(got[nm] - ref[nm])
        rel = d / max(abs(ref[nm]), 1e-12)
        bad = rel > tol and d > tol
        ok &= not bad
        print(
            f"    {nm:10s} {got[nm]:14.9f} vs {ref[nm]:14.9f}  "
            f"(|d| {d:.2e}, rel {rel:.2e}){'  <-- FAIL' if bad else ''}"
        )
    return ok


# ---------------------------------------------------------------------------
# 1. the quadratic term alone
# ---------------------------------------------------------------------------
def test_quadratic_only(tmpdir):
    print("\n=== 1. quadratic term alone (ExternalParams) ===")
    toy = Toy()
    card, models = toy.write_card(
        os.path.join(tmpdir, "quad.hdf5"), with_ext=True, with_mass=False
    )
    assert models == [["ExternalParams", "bundle:global_params"]], models
    ref_x, ref_cov, ref_nll = toy.solve(with_ext=True, with_mass=False)
    ref = {nm: ref_x[2 + j] for j, nm in enumerate(toy.names)}
    ref_err = {
        nm: float(np.sqrt(ref_cov[2 + j, 2 + j])) for j, nm in enumerate(toy.names)
    }
    # the closed form of the pure quadratic
    direct = -np.linalg.solve(toy.H, toy.g)
    assert np.allclose(direct, ref_x[2:], atol=1e-12), "internal reference inconsistent"

    f, val, err = fit_and_read(card, models, toy.names)
    ok = report("minimum vs -H^-1 g", val, ref, 1e-9)
    ok &= report("errors vs sqrt(diag(H^-1))", err, ref_err, 1e-7)
    nll = float(f.full_nll().numpy())
    print(f"    NLL {nll:.9f} (reference {ref_nll:.9f} + lognorm + dummy channel)")
    print("  PASS" if ok else "  FAIL")
    return ok


# ---------------------------------------------------------------------------
# 2. the unbinned term alone
# ---------------------------------------------------------------------------
def test_mass_only(tmpdir):
    print("\n=== 2. unbinned mass term alone (Gaussian-equivalent) ===")
    toy = Toy()
    card, models = toy.write_card(
        os.path.join(tmpdir, "mass.hdf5"), with_ext=False, with_mass=True
    )
    order = ["alpha", "k"] + toy.names
    # No reference *minimum* here: the mass term alone leaves flat directions
    # (theta columns with no D entries, and alpha degenerate with the D column
    # sum), so the analytic linear solve is singular by construction. What is
    # tested is that rabbit's NLL equals the analytic one point-by-point.
    f = make_fitter(card, models)
    # the mass term alone leaves flat directions (D has rank <= nshared and
    # alpha is degenerate with the D column sum): check the NLL value, not the
    # minimum, plus the gradient of the analytic reference
    parms = f.parms.astype(str)
    x = np.zeros(len(parms))
    rng = np.random.default_rng(3)
    for i, nm in enumerate(parms):
        x[i] = 1.0 if nm == "k" else rng.normal(0.0, 0.3)
    f.x.assign(tf.constant(x, dtype=f.x.dtype))
    # reduced, not full: the 1-bin dummy channel contributes exactly +1 to the
    # full Poisson NLL (n = nexp = 1 -> nexp + log n!) and 0 to the reduced one
    got = float(f.reduced_nll().numpy())
    ref_val = toy.nll(
        x[list(parms).index("alpha")],
        x[list(parms).index("k")],
        np.array([x[list(parms).index(nm)] for nm in toy.names]),
        with_ext=False,
        with_mass=True,
    )
    rel = abs(got - ref_val) / abs(ref_val)
    ok = rel < 1e-9
    print(
        f"    NLL at a random point: {got:.9f} vs analytic {ref_val:.9f} (rel {rel:.2e})"
    )
    print("  PASS" if ok else "  FAIL")
    return ok


# ---------------------------------------------------------------------------
# 3. the joint fit
# ---------------------------------------------------------------------------
def test_joint(tmpdir):
    print("\n=== 3. joint fit: quadratic + unbinned sharing theta by name ===")
    toy = Toy()
    card, models = toy.write_card(os.path.join(tmpdir, "joint.hdf5"))
    assert models == [
        ["UnbinnedParams"],
        ["ExternalParams", "bundle:global_params"],
    ], models  # theta4/theta5 exist only in the quadratic term
    order = ["alpha", "k"] + toy.names
    ref_x, ref_cov, ref_nll = toy.solve()
    ref = {nm: ref_x[i] for i, nm in enumerate(order)}
    ref_err = {nm: float(np.sqrt(ref_cov[i, i])) for i, nm in enumerate(order)}

    f, val, err = fit_and_read(card, models, order)
    ok = report("minimum", val, ref, 1e-7)
    ok &= report("errors", err, ref_err, 1e-5)

    # NLL breakdown
    nll_ext = float(f._compute_external_nll(full_nll=True).numpy())
    nll_unb = float(f._compute_unbinned_nll(full_nll=True).numpy())
    ref_ext = toy.nll(0.0, 1.0, ref_x[2:], with_ext=True, with_mass=False)
    print(f"    external part {nll_ext:14.6f}  (analytic {ref_ext:14.6f} + lognorm)")
    print(f"    unbinned part {nll_unb:14.6f}")
    print("  PASS" if ok else "  FAIL")
    return ok


# ---------------------------------------------------------------------------
# 4. Gaussian priors on the shared parameters
# ---------------------------------------------------------------------------
def test_priors(tmpdir):
    print("\n=== 4. Gaussian priors on shared and external-only parameters ===")
    priors = {"theta0": (0.0, 0.05), "theta1": (0.2, 0.5), "theta4": (0.0, 0.1)}
    toy = Toy(priors=priors)
    card, models = toy.write_card(os.path.join(tmpdir, "priors.hdf5"))
    order = ["alpha", "k"] + toy.names
    ref_x, ref_cov, _ = toy.solve()
    ref = {nm: ref_x[i] for i, nm in enumerate(order)}
    ref_err = {nm: float(np.sqrt(ref_cov[i, i])) for i, nm in enumerate(order)}
    f, val, err = fit_and_read(card, models, order)
    ok = report("minimum with priors", val, ref, 1e-7)
    ok &= report("errors with priors", err, ref_err, 1e-5)
    print("  PASS" if ok else "  FAIL")
    return ok


# ---------------------------------------------------------------------------
# 5. a frozen shared parameter must be frozen in BOTH terms
# ---------------------------------------------------------------------------
def test_freeze(tmpdir):
    print("\n=== 5. freezing a shared parameter (external term at get_x) ===")
    toy = Toy()
    card, models = toy.write_card(os.path.join(tmpdir, "freeze.hdf5"))
    # trust-krylov, not trust-exact: freezing zeroes the parameter's whole
    # Hessian row and column, and the exact trust-region subproblem solver
    # cannot handle the resulting singular block (it wanders along the null
    # direction). Same reason likelihood scans use trust-krylov.
    f = make_fitter(
        card, models, freezeParameters=["theta0"], minimizerMethod="trust-krylov"
    )
    parms = list(f.parms.astype(str))
    i0 = parms.index("theta0")
    x = f.x.numpy()
    x[i0] = 0.37
    f.x.assign(tf.constant(x, dtype=f.x.dtype))
    f.minimize()
    got = float(f.x.numpy()[i0])
    ok = abs(got - 0.37) < 1e-12
    print(f"    theta0 after the fit: {got:.12f} (frozen at 0.370000000000)")

    # and the profiled rest must equal the analytic conditional minimum
    ref_x, _, _ = toy.solve(fixed={"theta0": 0.37})
    order = ["alpha", "k"] + toy.names
    val = {nm: float(f.x.numpy()[parms.index(nm)]) for nm in order}
    ref = {nm: ref_x[i] for i, nm in enumerate(order)}
    ok &= report("profiled minimum at theta0 = 0.37", val, ref, 1e-6)
    print("  PASS" if ok else "  FAIL")
    return ok


# ---------------------------------------------------------------------------
# 6. injection: shift the two terms consistently and recover
# ---------------------------------------------------------------------------
def test_injection(tmpdir):
    print("\n=== 6. injection into both terms, recovered jointly ===")
    toy = Toy(seed=11)
    inj = np.zeros(toy.ntheta)
    inj[0] = 2.0e-4 / 1.0  # one mode, in its own units
    inj[2] = -5.0e-4

    base_x, base_cov, _ = toy.solve()
    # inject: the mass term sees m_i^0 -> m_i^0 + D_i . inj, the quadratic
    # term sees g -> g - H inj. Both shift the minimum by exactly +inj.
    toy_i = Toy(seed=11)
    toy_i.mobs = toy.mobs + toy.D @ inj
    toy_i.g = toy.g - toy.H @ inj
    card, models = toy_i.write_card(os.path.join(tmpdir, "inject.hdf5"))
    order = ["alpha", "k"] + toy.names
    ref_x, ref_cov, _ = toy_i.solve()
    f, val, err = fit_and_read(card, models, order)

    ok = True
    print("    parameter      fitted        baseline      injected      pull")
    for j, nm in enumerate(toy.names):
        d = val[nm] - base_x[2 + j]
        pull = (d - inj[j]) / err[nm]
        bad = abs(pull) > 1e-3
        ok &= not bad
        print(
            f"    {nm:10s} {val[nm]:13.6e} {base_x[2+j]:13.6e} "
            f"{inj[j]:13.6e} {pull:8.2e}{'  <-- FAIL' if bad else ''}"
        )
    # and consistency of the analytic reference itself
    shift = ref_x[2:] - base_x[2:]
    # the analytic reference itself is only exact to the tolerance of its 1D
    # minimization over k (brent, xtol 1e-14 -> ~1e-9 on the linear block)
    print(f"    analytic shift - injection: max |d| = {np.abs(shift - inj).max():.2e}")
    ok &= np.abs(shift - inj).max() < 1e-8
    print("  PASS" if ok else "  FAIL")
    return ok


# ---------------------------------------------------------------------------
def main():
    tf.config.experimental.enable_op_determinism()
    with tempfile.TemporaryDirectory() as tmpdir:
        results = [
            ("quadratic only", test_quadratic_only(tmpdir)),
            ("mass only", test_mass_only(tmpdir)),
            ("joint", test_joint(tmpdir)),
            ("priors", test_priors(tmpdir)),
            ("freeze", test_freeze(tmpdir)),
            ("injection", test_injection(tmpdir)),
        ]
    print("\n=== summary ===")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not all(ok for _, ok in results):
        sys.exit(1)
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
