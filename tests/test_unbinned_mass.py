#!/usr/bin/env python3
"""Tests for the unbinned mass likelihood term (rabbit/unbinned.py).

Checks, in order:

1. **objective identity** -- ``MassCFTerm`` evaluated through the rabbit
   Fitter equals the step-1 reference objective
   (``cf_masslik_fit.MassNLL``, in
   ``/work/submit/david_w/ZMass/calibration_studies/resolution/``) at several
   parameter points, to better than 1e-6 relative. Both are fed the *same*
   per-candidate arrays -- the ones the datacard actually carries -- so this
   isolates the objective from the input assembly.
2. **gradient** -- the Fitter's analytic gradient against central finite
   differences of its own loss.
3. **fit** -- a full rabbit minimisation. In the quick (default) mode the
   result is compared against the reference minimiser run on the same
   subsample; with ``--full`` it is compared against the published step-1
   numbers for the J/psi-gun cache (see ``STEP1``).
4. **Breit-Wigner kernel** -- the density against ``scipy.special.
   voigt_profile`` (exact), then a closure fit recovering the generated mass
   and width.
5. **Bernstein background** -- normalisation on the window, non-negativity,
   and the degree-0 / flat-coefficient limits.
6. **sparse D rows** -- the per-candidate ``m_i(theta) = m_i^0 + D_i theta``
   hook against the same shift applied by hand.
7. **two channels** -- one term's candidates split into two terms sharing the
   same parameters: the NLLs must add up and the fits must agree.
8. **Gaussian priors** -- a prior declared with the term acts as
   ``0.5 ((p - mu) / sigma)^2`` and constrains the postfit uncertainty.

Quick mode builds a small datacard from the step-1 caches (default 20k
candidates, a 20k-sample kernel CF on 2048 points) and runs in a few minutes.
``--full`` uses the complete caches; the datacard is then ~1.6 GB and the
fits take a few minutes each.

Run (in the rabbit / wmassdev singularity)::

    python tests/test_unbinned_mass.py
    python tests/test_unbinned_mass.py --full --workdir /scratch/.../cards
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import tensorflow as tf

from rabbit import fitter, inputdata, unbinned
from rabbit.param_models.helpers import load_model

REF_DIR = "/work/submit/david_w/ZMass/calibration_studies/resolution"
RUNS = os.path.join(REF_DIR, "runs")
GUN_PAIRS = os.path.join(RUNS, "cf_masspairs_jpsigun_ul16_260902_m0_fixsign.npz")
GUN_KERNEL = os.path.join(RUNS, "cf_masskernel_jpsigun_ul16_260902_m0.npz")

# Published step-1 numbers for the full J/psi-gun cache (cf_masslik_fit.py,
# ~/public_html/cvh/260903_masslikfit/masslikfit_summary.txt). Checked only
# in --full mode.
STEP1 = {
    "r": {
        "alpha": (0.218078, 0.016728),
        "r": (0.992253, 0.002743),
        "nll": -588373.854307,
    },
    "families": {
        "alpha": (0.250017, 0.017658),
        "k_hit": (0.943484, 0.035370),
        "k_ms": (1.010843, 0.006067),
        "k_ioni": (0.688443, 0.054486),
        "nll": -588388.626654,
    },
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class Options:
    """Minimal stand-in for the rabbit_fit.py argparse namespace."""

    def __init__(self, **kwargs):
        defaults = dict(
            earlyStopping=-1,
            noBinByBinStat=False,
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


def make_fitter(filename, model="UnbinnedParams", **opts):
    indata = inputdata.FitInputData(filename)
    param_model = load_model(model, indata)
    f = fitter.Fitter(indata, param_model, Options(**opts))
    f.set_nobs(f.indata.data_obs)
    return f


def loss_at(f, x):
    f.x.assign(tf.constant(np.asarray(x, dtype=np.float64), dtype=f.x.dtype))
    return float(f.loss_val().numpy())


def build_card(args, model, outfile, extra=()):
    """Run the converter to produce a datacard, unless it is already there."""
    if os.path.exists(outfile):
        print(f"  reusing {outfile}")
        return outfile
    cmd = [
        sys.executable,
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "make_unbinned_mass_tensor.py"
        ),
        "--pairs-cache",
        args.pairs_cache,
        "--kernel-cache",
        args.kernel_cache,
        "--model",
        model,
        "-o",
        outfile,
        "--phik-cache",
        os.path.join(args.workdir, "phik_%s.npz" % args.tag),
        "--chunk",
        str(args.chunk),
    ]
    if not args.full:
        cmd += [
            "--maxn",
            str(args.maxn),
            "--maxk",
            str(args.maxk),
            "--phik-points",
            str(args.phik_points),
        ]
    cmd += list(extra)
    print("  " + " ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"  datacard built in {time.time()-t0:.1f} s")
    return outfile


def reference_families(term):
    """``{name: family}`` if the reference implementation knows this family set.

    ``cf_masslik_fit`` hard-codes the hit / ms / ioni split, optionally plus a
    radiative family; a cache with any other family cannot be compared against
    it, and the caller skips.
    """
    fams = {f["name"]: f for f in term.families}
    if not {"hit", "ms", "ioni"} <= set(fams):
        return None
    if set(fams) - {"hit", "ms", "ioni", "rad"}:
        return None
    return fams


def reference_order(term, model):
    """Parameter order of ``cf_masslik_fit.MassNLL`` for this term."""
    if model == "r":
        return ["alpha", "r"]
    order = ["alpha", "k_hit", "k_ms", "k_ioni"]
    if any(f["name"] == "rad" for f in term.families):
        order.append("k_rad")
    return order


def reference_objective(term, model, float_bkg=False):
    """A ``cf_masslik_fit.MassNLL`` fed from the term's own arrays."""
    import cf_masslik_fit

    fams = reference_families(term)
    inp = dict(
        n=term.n,
        TG=term.tgrid.numpy(),
        sig=term.sigma.numpy(),
        mobs=term.mobs.numpy(),
        vgf=term.vgf.numpy(),
        Sms=fams["ms"]["re"].numpy(),
        Sio_re=fams["ioni"]["re"].numpy(),
        Sio_im=fams["ioni"]["im"].numpy(),
        phiK_re=term.phik_re.numpy(),
        phiK_im=term.phik_im.numpy(),
        folded=False,
    )
    kwargs = {}
    if "rad" in fams:
        # the radiative family the parallel step-1 work adds: MassNLL picks it
        # up from has_rad and puts k_rad after k_ioni (and shares the single r
        # with every family in the 'r' model), which is the layout the
        # converter writes.
        inp["Srad_re"] = fams["rad"]["re"].numpy()
        inp["Srad_im"] = fams["rad"]["im"].numpy()
        inp["has_rad"] = True
        if model != "r":
            kwargs["float_krad"] = True
    return cf_masslik_fit.MassNLL(
        inp,
        model=model,
        float_bkg=float_bkg,
        floor=term.floor,
        floor_scale=term.floor_scale,
        chunk=term.chunk,
        fbkg0=term.bkg_frac,
        log=lambda *a: None,
        **kwargs,
    )


def cov_from_fitter(f):
    _, _, hess = f.loss_val_grad_hess()
    h = hess.numpy()
    return np.linalg.inv(h), h


# ---------------------------------------------------------------------------
# 1. objective identity
# ---------------------------------------------------------------------------
def test_identity(args, card, model):
    print(f"\n=== 1. objective identity vs cf_masslik_fit.MassNLL ({model}) ===")
    f = make_fitter(card)
    term = f.indata.unbinned_terms[0]
    names = list(f.parms.astype(str))
    print(f"  fit parameters: {names}")
    if reference_families(term) is None:
        print(
            "  SKIP: the reference implementation only knows the "
            "hit/ms/ioni(/rad) families; this cache has "
            f"{[fam['name'] for fam in term.families]}"
        )
        return True

    try:
        obj = reference_objective(term, model)
    except ImportError:
        print(f"  SKIP: cf_masslik_fit not importable (add {REF_DIR} to PYTHONPATH)")
        return True

    ref_order = reference_order(term, model)
    perm = [names.index(p) for p in ref_order]

    rng = np.random.default_rng(7)
    npar = len(ref_order)
    points = [
        [0.0] + [1.0] * (npar - 1),
        [0.2] + [1.0] * (npar - 1),
        ([0.25, 0.94, 1.01, 0.69, 1.05] + [1.0] * npar)[:npar],
        [-0.4] + [0.9] * (npar - 1),
        [1.3] + [1.1] * (npar - 1),
    ]
    points += [list(rng.normal([0.2] + [1.0] * (npar - 1), 0.05)) for _ in range(3)]

    ok = True
    print(f"  {'point':>34s} {'rabbit NLL':>18s} {'reference':>18s} {'rel':>10s}")
    for p in points:
        xr = np.array(p, dtype=np.float64)
        xfull = np.zeros(len(names))
        for v, i in zip(xr, perm):
            xfull[i] = v
        v_rabbit = loss_at(f, xfull)
        v_ref = obj.nll(xr)
        # the binned dummy channel contributes a constant; remove it
        v_rabbit -= args.dummy_offset
        rel = abs(v_rabbit - v_ref) / max(abs(v_ref), 1.0)
        flag = "" if rel < 1e-6 else "   <<< FAIL"
        ok &= rel < 1e-6
        print(
            f"  {np.array2string(xr, precision=3):>34s} {v_rabbit:18.6f} "
            f"{v_ref:18.6f} {rel:10.2e}{flag}"
        )
    print("  PASS" if ok else "  FAIL")
    return ok


# ---------------------------------------------------------------------------
# 2. gradient vs finite differences
# ---------------------------------------------------------------------------
def test_gradient(args, card):
    print("\n=== 2. gradient vs central finite differences ===")
    f = make_fitter(card)
    names = list(f.parms.astype(str))
    x0 = f.x.numpy().copy()
    f.x.assign(tf.constant(x0, dtype=f.x.dtype))
    _, grad = f.loss_val_grad()
    g = grad.numpy()
    ok = True
    print(f"  {'par':>10s} {'analytic':>16s} {'central FD':>16s} {'rel':>10s}")
    for i, nm in enumerate(names):
        h = 1e-4 * max(abs(x0[i]), 1e-2)
        xp, xm = x0.copy(), x0.copy()
        xp[i] += h
        xm[i] -= h
        fd = (loss_at(f, xp) - loss_at(f, xm)) / (2 * h)
        rel = abs(g[i] - fd) / max(abs(fd), 1e-9)
        ok &= rel < 1e-4
        print(f"  {nm:>10s} {g[i]:16.6f} {fd:16.6f} {rel:10.2e}")
    print("  PASS" if ok else "  FAIL")
    return ok


# ---------------------------------------------------------------------------
# 3. the fit
# ---------------------------------------------------------------------------
def test_fit(args, card, model):
    print(f"\n=== 3. rabbit fit ({model}) ===")
    f = make_fitter(card, minimizerMethod=args.minimizer)
    names = list(f.parms.astype(str))
    t0 = time.time()
    f.minimize()
    wall = time.time() - t0
    cov, hess = cov_from_fitter(f)
    x = f.x.numpy()
    err = np.sqrt(np.diag(cov))
    val, grad = f.loss_val_grad()
    nll = float(val.numpy()) - args.dummy_offset
    print(
        f"  {args.minimizer}: wall {wall:.1f} s, NLL = {nll:.6f}, "
        f"|grad|inf = {np.max(np.abs(grad.numpy())):.3e}"
    )
    for i, nm in enumerate(names):
        print(f"    {nm:>10s} = {x[i]:12.6f} +- {err[i]:.6f}")

    ok = True
    if args.full:
        ref = STEP1[model]
        print("  vs step-1 (cf_masslik_fit.py, full cache):")
        for nm, entry in ref.items():
            if nm == "nll":
                d = abs(nll - entry)
                ok &= d < 1e-2
                print(f"    {'NLL':>10s}: {nll:.6f} vs {entry:.6f}  (d = {d:.2e})")
                continue
            v, e = entry
            i = names.index(nm)
            dv = abs(x[i] - v) / max(abs(v), 1e-12)
            de = abs(err[i] - e) / e
            ok &= dv < 0.01 and de < 0.01
            print(
                f"    {nm:>10s}: {x[i]:.6f} +- {err[i]:.6f} vs {v:.6f} +- {e:.6f}"
                f"  (dval {dv:.2e}, derr {de:.2e})"
            )
    else:
        try:
            import cf_masslik_fit
        except ImportError:
            print("  SKIP reference comparison: cf_masslik_fit not importable")
            return ok
        term = f.indata.unbinned_terms[0]
        if reference_families(term) is None:
            print(
                "  SKIP reference comparison: unsupported family set "
                f"{[fam['name'] for fam in term.families]}"
            )
            return ok
        obj = reference_objective(term, model)
        ref_order = reference_order(term, model)
        x0 = [0.2] + [1.0] * (len(ref_order) - 1)
        res = cf_masslik_fit.minimize(
            obj, x0, method="trust-exact", log=lambda *a: None
        )
        cref = np.linalg.inv(res["hess"])
        eref = np.sqrt(np.diag(cref))
        print("  vs the reference minimiser on the same sample:")
        for j, nm in enumerate(ref_order):
            i = names.index(nm)
            dsig = abs(x[i] - res["x"][j]) / max(eref[j], 1e-12)
            de = abs(err[i] - eref[j]) / eref[j]
            ok &= dsig < 1e-3 and de < 1e-3
            print(
                f"    {nm:>10s}: {x[i]:.6f} +- {err[i]:.6f} vs "
                f"{res['x'][j]:.6f} +- {eref[j]:.6f}  "
                f"(d = {dsig:.2e} sigma, derr {de:.2e})"
            )
        d_nll = abs(nll - res["nll"])
        ok &= d_nll < 1e-4
        print(f"    {'NLL':>10s}: {nll:.6f} vs {res['nll']:.6f} (d = {d_nll:.2e})")
    print("  PASS" if ok else "  FAIL")
    return ok


# ---------------------------------------------------------------------------
# 4. Breit-Wigner kernel
# ---------------------------------------------------------------------------
def _bw_term(name, mobs, sigma, nt=1024, tmax=14.0, m_ref=9.46, chunk=8192):
    """A pure Breit-Wigner (x) per-candidate Gaussian term: a Voigt profile.

    The ``gauss`` family with ``vgf = 1`` and ``k_res = 1`` is exactly the
    Gaussian resolution CF ``exp(-t^2/2)`` in standardized units, and the
    Breit-Wigner kernel multiplies it by ``exp(-Gamma t / (2 sigma))``, so the
    density is the Voigt profile ``V(m - m_R; sigma, Gamma/2)``.
    """
    return unbinned.MassCFTerm(
        name,
        sigma=sigma,
        mobs=mobs,
        tgrid=np.linspace(0.0, tmax, nt),
        families=[{"name": "res", "param": "k_res", "kind": "gauss"}],
        vgf=np.ones_like(sigma),
        kernel=unbinned.BreitWignerKernel(
            width_param="gamma", mass_param="dm", width_unit=1e-3, mass_unit=1e-3
        ),
        m_ref=m_ref,
        scale_param=None,
        bkg_frac=0.0,
        chunk=chunk,
    )


def _trust_exact(nll_fn, x0, npar):
    """trust-exact with the exact TF gradient and Hessian of ``nll_fn``."""
    import scipy.optimize

    x = tf.Variable(np.asarray(x0, dtype=np.float64), dtype=tf.float64)

    @tf.function
    def _graph():
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                val = nll_fn(x)
            g = t1.gradient(val, x)
        h = t2.jacobian(g, x)
        return val, g, h

    def lgh(v):
        x.assign(tf.constant(np.asarray(v, dtype=np.float64), tf.float64))
        val, g, h = _graph()
        return float(val.numpy()), g.numpy(), h.numpy()

    cache = {}

    def _c(v):
        key = tuple(v)
        if key not in cache:
            cache.clear()
            cache[key] = lgh(v)
        return cache[key]

    res = scipy.optimize.minimize(
        lambda v: _c(v)[0],
        np.asarray(x0, dtype=np.float64),
        jac=lambda v: _c(v)[1],
        hess=lambda v: _c(v)[2],
        method="trust-exact",
        options=dict(gtol=1e-8, maxiter=300),
    )
    _, g, h = _c(res.x)
    return res, g, h


def test_breit_wigner(args):
    print("\n=== 4. Breit-Wigner kernel ===")
    from scipy.special import voigt_profile

    # fixed seed; the closure was checked to be unbiased over several seeds
    # (k_res and gamma are strongly anti-correlated, so single-sample pulls of
    # ~2 sigma on either are normal)
    rng = np.random.default_rng(21)
    m_ref = 9.4603  # Upsilon(1S)
    gamma_true = 5.0  # MeV; fitted in units of 1e-3 GeV
    dm_true = 5.0  # MeV offset of the resonance from m_ref
    ok = True

    # -- 4a. the density against an exact Voigt profile ------------------
    # The only difference is the trapezoid discretization of the inverse
    # Fourier transform, which is O(dt^2) -- checked here by refining the
    # grid, so the comparison tests the kernel and not the quadrature.
    n = 400
    sigma = np.full(n, 0.030)
    mobs = np.linspace(-0.15, 0.15, n)
    core = np.abs(mobs - dm_true * 1e-3) < 4 * sigma[0]
    ref = voigt_profile(mobs - dm_true * 1e-3, sigma, 0.5 * gamma_true * 1e-3)
    devs = {}
    for nt in (1024, 8192):
        term = _bw_term("bw%d" % nt, mobs, sigma, nt=nt, m_ref=m_ref)
        assert term.param_names == ["k_res", "dm", "gamma"], term.param_names
        li = term.raw_density(
            tf.constant([1.0, dm_true, gamma_true], tf.float64)
        ).numpy()
        devs[nt] = np.max(np.abs(li[core] - ref[core]) / ref[core])
        print(
            f"  density vs scipy voigt_profile, {nt:5d} t points: "
            f"max rel dev = {devs[nt]:.2e}"
        )
    ok &= devs[8192] < 1e-5
    ok &= devs[1024] / devs[8192] > 10  # O(dt^2): 64x expected, allow slack
    print(
        f"  refinement factor {devs[1024]/devs[8192]:.1f} (O(dt^2) -> 64) "
        f"{'PASS' if ok else 'FAIL'}"
    )

    # -- 4b. closure: recover the generated mass and width ---------------
    # A Breit-Wigner has an unbounded Cauchy tail, so any sample has to be cut
    # at some mass window -- and for a Voigt whose Gaussian core hides the
    # width, a large fraction of the information on Gamma sits between ~4
    # sigma and the cut. Fitting the *untruncated* density to a truncated
    # sample therefore biases Gamma by tens of percent, which has nothing to
    # do with the term. The test uses the correct truncated likelihood
    # instead: - sum_i log[L_i / Z], with Z the model's own integral over the
    # window. Z is built from a second MassCFTerm whose "candidates" are a
    # dense grid of masses at the same resolution, so it is exact (same
    # quadrature) and differentiable -- the pattern any real fit with a mass
    # window would use.
    n = 100000
    sig0 = 0.025
    half_window = 12 * sig0
    gamma_true = 20.0
    sigma = np.full(n, sig0)
    truth = dm_true * 1e-3 + 0.5 * gamma_true * 1e-3 * rng.standard_cauchy(n)
    mobs = truth + sigma * rng.standard_normal(n)
    keep = np.abs(mobs) < half_window
    nkeep = int(keep.sum())
    print(
        f"  generated {n} at sigma = {sig0*1e3:.0f} MeV, Gamma = "
        f"{gamma_true:.0f} MeV; kept {nkeep} in |m - m_ref| < "
        f"{half_window*1e3:.0f} MeV ({100*(1-keep.mean()):.2f} % outside, "
        f"normalised out of the likelihood)"
    )
    nt = 2048
    term = _bw_term("bwfit", mobs[keep], sigma[keep], nt=nt, m_ref=m_ref, chunk=16384)
    grid = np.linspace(-half_window, half_window, 4001)
    norm_term = _bw_term(
        "bwnorm", grid, np.full(len(grid), sig0), nt=nt, m_ref=m_ref, chunk=len(grid)
    )
    dgrid = tf.constant(np.diff(grid), tf.float64)

    def nll_truncated(x):
        dens = norm_term.raw_density(x)
        z = tf.reduce_sum(dgrid * (dens[1:] + dens[:-1]) * 0.5)
        return term.nll(x) + tf.constant(float(nkeep), tf.float64) * tf.math.log(z)

    t0 = time.time()
    res, grad, hess = _trust_exact(nll_truncated, [1.0, 0.0, 30.0], 3)
    err = np.sqrt(np.diag(np.linalg.inv(hess)))
    print(
        f"  fit in {time.time()-t0:.1f} s ({res.nit} iterations, "
        f"|grad|inf = {np.max(np.abs(grad)):.2e})"
    )
    labels = ["k_res", "dm [MeV]", "gamma [MeV]"]
    truths = [1.0, dm_true, gamma_true]
    for lab, v, e, tv in zip(labels, res.x, err, truths):
        pull = (v - tv) / e
        print(
            f"    {lab:>12s} = {v:9.4f} +- {e:.4f}   truth {tv:8.4f}   "
            f"pull {pull:+.2f}"
        )
        ok &= abs(pull) < 4.0
    print("  PASS" if ok else "  FAIL")
    return ok


# ---------------------------------------------------------------------------
# 5. Bernstein background
# ---------------------------------------------------------------------------
def test_bernstein():
    print("\n=== 5. Bernstein background pdf ===")
    rng = np.random.default_rng(3)
    ok = True
    window = (2.75, 3.45)
    for deg in (0, 1, 2, 4):
        names = [f"c{i}" for i in range(deg + 1)]
        bkg = unbinned.BernsteinBackground(window, names)
        for trial in range(3):
            p = rng.normal(0.0, 1.5, deg + 1)
            values = {n: tf.constant(v, tf.float64) for n, v in zip(names, p)}
            m = tf.constant(np.linspace(window[0], window[1], 200001), tf.float64)
            y = bkg.pdf(values, m).numpy()
            integral = np.trapezoid(y, np.linspace(window[0], window[1], 200001))
            neg = float(y.min())
            ok &= abs(integral - 1.0) < 1e-8 and neg >= 0.0
            if trial == 0:
                print(
                    f"  degree {deg}: integral = {integral:.12f}, "
                    f"min pdf = {neg:.4e}"
                )
        # flat coefficients must reproduce the uniform density
        flat = float(np.log(np.expm1(1.0)))
        values = {n: tf.constant(flat, tf.float64) for n in names}
        y = bkg.pdf(values, tf.constant([2.8, 3.0, 3.4], tf.float64)).numpy()
        d = np.max(np.abs(y - 1.0 / (window[1] - window[0])))
        ok &= d < 1e-12
        print(f"  degree {deg}: flat-coefficient limit vs uniform: {d:.2e}")
    print("  PASS" if ok else "  FAIL")
    return ok


# ---------------------------------------------------------------------------
# 6. sparse per-candidate parameter dependence
# ---------------------------------------------------------------------------
def test_jacobian_rows():
    print("\n=== 6. sparse D rows: m_i(theta) = m_i^0 + D_i theta ===")
    rng = np.random.default_rng(5)
    n, nt, ntheta = 500, 128, 3
    sigma = rng.uniform(0.02, 0.04, n)
    mobs = rng.normal(0.0, 0.03, n)
    tgrid = np.linspace(0.0, 14.0, nt)
    vgf = np.ones(n)
    theta = np.array([0.7, -1.3, 0.4])

    # a sparse D with ~2 non-zero columns per candidate
    rows, cols, vals = [], [], []
    for i in range(n):
        for c in rng.choice(ntheta, size=2, replace=False):
            rows.append(i)
            cols.append(int(c))
            vals.append(rng.normal(0.0, 1e-3))
    idx = np.stack([rows, cols], axis=1)
    vals = np.array(vals)
    dense = np.zeros((n, ntheta))
    dense[idx[:, 0], idx[:, 1]] = vals

    common = dict(
        tgrid=tgrid,
        families=[{"name": "res", "param": "k", "kind": "gauss"}],
        vgf=vgf,
        m_ref=3.0969,
        scale_param=None,
        bkg_frac=0.0,
        chunk=128,
    )
    t_jac = unbinned.MassCFTerm(
        "withD",
        sigma=sigma,
        mobs=mobs,
        jac=(idx, vals, (n, ntheta)),
        jac_params=["t0", "t1", "t2"],
        **common,
    )
    assert t_jac.param_names == ["k", "t0", "t1", "t2"], t_jac.param_names
    t_ref = unbinned.MassCFTerm(
        "shifted", sigma=sigma, mobs=mobs - dense @ theta, **common
    )
    v_jac = float(t_jac.nll(tf.constant([1.0, *theta], tf.float64)).numpy())
    v_ref = float(t_ref.nll(tf.constant([1.0], tf.float64)).numpy())
    rel = abs(v_jac - v_ref) / max(abs(v_ref), 1.0)
    ok = rel < 1e-12
    print(
        f"  NLL with D contraction {v_jac:.9f} vs pre-shifted {v_ref:.9f} "
        f"(rel {rel:.2e})"
    )

    # the gradient w.r.t. theta must be the sum of the per-candidate rows
    x = tf.constant([1.0, *theta], tf.float64)
    with tf.GradientTape() as t:
        t.watch(x)
        v = t_jac.nll(x)
    g = t.gradient(v, x).numpy()
    eps = 1e-6
    for j in range(ntheta):
        xp = np.array([1.0, *theta])
        xm = xp.copy()
        xp[1 + j] += eps
        xm[1 + j] -= eps
        fd = (
            float(t_jac.nll(tf.constant(xp, tf.float64)).numpy())
            - float(t_jac.nll(tf.constant(xm, tf.float64)).numpy())
        ) / (2 * eps)
        rel = abs(g[1 + j] - fd) / max(abs(fd), 1e-9)
        ok &= rel < 1e-5
        print(f"    d/dtheta{j}: {g[1+j]:14.6f} vs FD {fd:14.6f} (rel {rel:.2e})")
    print("  PASS" if ok else "  FAIL")
    return ok


# ---------------------------------------------------------------------------
# 7. two channels in one datacard, sharing parameters
# ---------------------------------------------------------------------------
def test_two_channels(args, card):
    """Split one term's candidates into two terms sharing the same parameters.

    This is the multi-channel case of the design (several resonance channels in
    one likelihood, tied together by common parameters) reduced to something
    with a known answer: the sum of the two terms' NLL must equal the single
    term's, and the fit must land on the same minimum. It also exercises the
    ``phik_grid`` storage variant (the already-interpolated per-candidate
    kernel CF) and the merging of parameter declarations across terms in
    UnbinnedParams.
    """
    print("\n=== 7. two unbinned terms in one datacard, shared parameters ===")
    from rabbit import tensorwriter

    f1 = make_fitter(card)
    term = f1.indata.unbinned_terms[0]
    names = list(f1.parms.astype(str))
    n = term.n
    half = n // 2

    def sub(sl):
        d = {
            "sigma": term.sigma.numpy()[sl],
            "mobs": term.mobs.numpy()[sl],
            "vgf": term.vgf.numpy()[sl],
            "tgrid": term.tgrid.numpy(),
            "phik_grid_re": term.phik_re.numpy()[sl],
            "phik_grid_im": term.phik_im.numpy()[sl],
        }
        for fam in term.families:
            for comp in ("re", "im"):
                if comp in fam:
                    d[f"S_{comp}_{fam['name']}"] = fam[comp].numpy()[sl]
        return d

    writer = tensorwriter.TensorWriter()
    writer.add_dummy_channel(name="dummy")
    cfg = term.config()
    for label, sl in (("first", slice(0, half)), ("second", slice(half, n))):
        c = dict(cfg)
        c["channel"] = label
        writer.add_unbinned_term(
            label,
            c,
            term.param_names,
            sub(sl),
            param_defaults=term.param_defaults,
            param_is_poi=term.param_is_poi,
        )
    out = os.path.join(args.workdir, f"unbinned_{args.tag}_twoterm.hdf5")
    if os.path.exists(out):
        os.remove(out)
    writer.write(
        outfolder=os.path.dirname(out),
        outfilename=os.path.basename(out)[: -len(".hdf5")],
    )

    f2 = make_fitter(out)
    assert list(f2.parms.astype(str)) == names, (f2.parms, names)
    print(
        f"  {len(f2.indata.unbinned_terms)} terms, "
        f"{[t.n for t in f2.indata.unbinned_terms]} candidates, "
        f"shared parameters {names}"
    )

    ok = True
    rng = np.random.default_rng(4)
    for k in range(3):
        x = f1.x.numpy() + rng.normal(0.0, 0.02, len(names))
        v1, v2 = loss_at(f1, x), loss_at(f2, x)
        rel = abs(v1 - v2) / max(abs(v1), 1.0)
        ok &= rel < 1e-12
        print(f"    NLL(one term) {v1:.9f}  NLL(two terms) {v2:.9f}  " f"rel {rel:.2e}")

    f1.minimize()
    f2.minimize()
    x1, x2 = f1.x.numpy(), f2.x.numpy()
    c1, _ = cov_from_fitter(f1)
    e1 = np.sqrt(np.diag(c1))
    for i, nm in enumerate(names):
        d = abs(x1[i] - x2[i]) / max(e1[i], 1e-12)
        ok &= d < 1e-4
        print(
            f"    {nm:>10s}: {x1[i]:.6f} (1 term) vs {x2[i]:.6f} (2 terms)  "
            f"d = {d:.1e} sigma"
        )
    print("  PASS" if ok else "  FAIL")
    return ok


# ---------------------------------------------------------------------------
# 8. Gaussian priors on unbinned parameters
# ---------------------------------------------------------------------------
def test_priors(args):
    """A prior declared with the term must act as 0.5 ((p - mu) / sigma)^2.

    Priors travel with the term (``param_prior_sigmas`` / ``param_prior_means``
    in the datacard), are declared by UnbinnedParams and applied by the Fitter
    through the ordinary ParamModel prior mechanism -- there is no separate
    code path for them here. This checks the arithmetic exactly and that the
    postfit uncertainty on the priored parameter shrinks accordingly.
    """
    print("\n=== 8. Gaussian prior on an unbinned parameter ===")
    mu, sigma = 1.0, 0.005
    card = build_card(
        args,
        "families",
        os.path.join(args.workdir, f"unbinned_{args.tag}_prior.hdf5"),
        extra=["--prior", f"k_ms:{mu}:{sigma}"],
    )
    f0 = make_fitter(os.path.join(args.workdir, f"unbinned_{args.tag}_families.hdf5"))
    f1 = make_fitter(card)
    names = list(f1.parms.astype(str))
    i = names.index("k_ms")
    print(
        f"  prior on k_ms: mu = {mu}, sigma = {sigma}; "
        f"constraint weight = {f1.cw.numpy()[i]:.1f} (1/sigma^2 = "
        f"{1/sigma**2:.1f})"
    )
    ok = abs(f1.cw.numpy()[i] - 1.0 / sigma**2) < 1e-6

    rng = np.random.default_rng(9)
    for k in range(3):
        x = f1.x.numpy() + rng.normal(0.0, 0.02, len(names))
        d = loss_at(f1, x) - loss_at(f0, x)
        expect = 0.5 * ((x[i] - mu) / sigma) ** 2
        rel = abs(d - expect) / max(abs(expect), 1e-12)
        ok &= rel < 1e-9
        print(
            f"    dNLL(prior) = {d:14.6f}  expected {expect:14.6f}  " f"rel {rel:.2e}"
        )

    f0.minimize()
    f1.minimize()
    e0 = np.sqrt(np.diag(cov_from_fitter(f0)[0]))[i]
    e1 = np.sqrt(np.diag(cov_from_fitter(f1)[0]))[i]
    ok &= e1 < e0 and e1 < sigma
    print(
        f"    k_ms = {f0.x.numpy()[i]:.6f} +- {e0:.6f} (free) -> "
        f"{f1.x.numpy()[i]:.6f} +- {e1:.6f} (priored)"
    )
    print("  PASS" if ok else "  FAIL")
    return ok


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="use the complete caches and compare against the "
        "published step-1 numbers",
    )
    p.add_argument("--pairs-cache", default=GUN_PAIRS)
    p.add_argument("--kernel-cache", default=GUN_KERNEL)
    p.add_argument(
        "--workdir",
        default=None,
        help="where the datacards are built (default: a temporary "
        "directory; give a path to reuse them between runs)",
    )
    p.add_argument("--maxn", type=int, default=20000)
    p.add_argument("--maxk", type=int, default=20000)
    p.add_argument("--phik-points", type=int, default=2048)
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--minimizer", default="trust-exact")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument(
        "--only", default=None, help="comma separated subset of tests to run (1..6)"
    )
    return p.parse_args()


def main():
    args = parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(max(1, args.threads // 8))
    args.tag = "full" if args.full else f"n{args.maxn}k{args.maxk}"
    # the 1-bin dummy channel contributes a constant Poisson term with Asimov
    # data (nobs = nexp = 1): -nobs (log nexp - log nobs) + nexp - nobs = 0
    args.dummy_offset = 0.0

    tmp = None
    if args.workdir is None:
        tmp = tempfile.TemporaryDirectory()
        args.workdir = tmp.name
    os.makedirs(args.workdir, exist_ok=True)
    print(f"work directory {args.workdir}")

    which = (
        set(args.only.split(","))
        if args.only
        else {"1", "2", "3", "4", "5", "6", "7", "8"}
    )
    results = {}

    if which & {"1", "2", "3", "7", "8"}:
        cards = {}
        for model in ("families", "r"):
            cards[model] = build_card(
                args,
                model,
                os.path.join(args.workdir, f"unbinned_{args.tag}_{model}.hdf5"),
            )
        if "1" in which:
            results["1 identity (families)"] = test_identity(
                args, cards["families"], "families"
            )
            results["1 identity (r)"] = test_identity(args, cards["r"], "r")
        if "2" in which:
            results["2 gradient"] = test_gradient(args, cards["families"])
        if "3" in which:
            results["3 fit (families)"] = test_fit(args, cards["families"], "families")
            results["3 fit (r)"] = test_fit(args, cards["r"], "r")
        if "7" in which:
            results["7 two channels"] = test_two_channels(args, cards["families"])
        if "8" in which:
            results["8 gaussian priors"] = test_priors(args)
    if "4" in which:
        results["4 breit-wigner"] = test_breit_wigner(args)
    if "5" in which:
        results["5 bernstein"] = test_bernstein()
    if "6" in which:
        results["6 sparse D rows"] = test_jacobian_rows()

    print("\n=== summary ===")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    if tmp is not None:
        tmp.cleanup()
    if not all(results.values()):
        raise SystemExit("SOME CHECKS FAILED")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
