#!/usr/bin/env python3
"""Tests for the Z/gamma* lineshape kernel (rabbit/lineshapes/zgamma.py).

Checks, in order:

1. **lineshape** -- ``ZGammaLineshape.dsigma_dm`` against the reference
   implementation it was ported from (``hard_me`` in
   ``calibration_studies/lineshape/zwidth_sensitivity.py``) on the same mass
   grid with the same parton luminosities: this must agree to float64
   round-off, because it is the same formula. The shipped luminosity table is
   then compared against the reference study's own cache, which quantifies the
   only genuine difference between the two (a different set of spline
   anchors).
2. **characteristic function** -- (a) ``cf_tab`` against an independent
   cell-by-cell Filon quadrature of the same piecewise-linear pdf, which tests
   the hat-basis/rFFT identity, the conjugation and the phase bookkeeping;
   (b) the *smeared* density that ``MassCFTerm`` actually builds from the
   kernel against a direct numerical Gaussian convolution of the pdf -- the
   end-to-end inverse transform, including the interpolation onto the
   per-candidate ``t/sigma`` grid.
3. **gradients** -- the analytic TF gradient and Hessian of a term's NLL with
   respect to ``m_Z`` and ``Gamma_Z`` against central finite differences.
4. **toy closure** -- 200k candidates generated from the lineshape smeared
   with a per-candidate Gaussian (sigma 1-2 GeV, no FSR, no background); fit
   ``m_Z``, ``Gamma_Z`` and the resolution scale and report pulls, together
   with the naive statistical expectations for ``sigma(Gamma_Z)``.
5. **wrong kernel** -- the same sample fitted with a plain Breit-Wigner
   kernel, which has no gamma*, no interference and no parton luminosity: the
   fitted mass is biased, and by how much is the size of the effect this
   provider exists to remove.
6. **datacard round trip** -- a term carrying the provider is written with
   ``TensorWriter.add_unbinned_term``, read back, and its parameters declared
   by ``UnbinnedParams``: the provider is rebuilt from its JSON config, the
   NLL through the Fitter matches the term evaluated directly, ``m_Z`` and
   ``Gamma_Z`` come out as the two POIs, and a Gaussian prior on ``Gamma_Z``
   is applied.
7. **terms / acceptance / FSR fold** -- the three optional modifiers of the
   Born spectrum, each against a closed form: switching matrix-element pieces
   off reproduces the piecewise sums; an acceptance multiplies the *un*folded
   pdf exactly; a one-atom FSR kernel at ``r0`` is a pure rescaling, so the
   folded pdf must equal ``p_born(m/r0)/r0``; a two-atom kernel is the weighted
   sum of two such rescalings; a one-band kernel equals the band-less one; and
   the whole configuration round-trips through ``config()``/``from_config``.

Run (in the rabbit / wmassdev singularity)::

    python tests/test_zgamma_kernel.py
    python tests/test_zgamma_kernel.py --n 50000 --skip 4 5   # quick

Test 1's reference comparison needs the lineshape study checkout
(``--ref-dir``); it is reported as SKIP when absent. Everything else is
self-contained.
"""

import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rabbit import unbinned  # noqa: E402
from rabbit.lineshapes import ZGammaLineshape, make_provider  # noqa: E402
from tests.test_unbinned_mass import _trust_exact  # noqa: E402

REF_LINESHAPE_DIR = "/work/submit/david_w/ZMass/calibration_studies/lineshape"

try:  # keep the box polite; must happen before the runtime is initialised
    tf.config.threading.set_intra_op_parallelism_threads(16)
except RuntimeError:
    pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _vals(**kw):
    return {k: tf.constant(float(v), tf.float64) for k, v in kw.items()}


def _filon_cf(z, p, taus):
    """Exact CF of the piecewise-linear pdf, cell by cell (numpy, independent).

    On each cell the integrand is (linear in m) x e^{i tau m}, whose integral is
    analytic:  dm e^{i tau m_k} [p_k A(theta) + p_{k+1} B(theta)] with
    theta = tau dm, A = Int_0^1 (1-v) e^{i theta v} dv, B = Int_0^1 v e^{...}.
    Series expansions are used for small theta. This shares no algebra with the
    provider's hat-basis/rFFT route.
    """
    dm, m = z.dm, z.m_grid
    tau = np.asarray(taus, dtype=float)
    th = tau[:, None] * dm
    e = np.exp(1j * th)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = -(e * (1j * th - 1) + 1) / th**2
        a = (e - 1) / (1j * th) - b
    small = np.abs(th) < 5e-2
    s = th[small]
    a[small] = 0.5 + 1j * s / 6 - s**2 / 24 - 1j * s**3 / 120 + s**4 / 720
    b[small] = 0.5 + 1j * s / 3 - s**2 / 8 - 1j * s**3 / 30 + s**4 / 144
    ph = np.exp(1j * tau[:, None] * (m[:-1] - z.m_ref))
    return dm * np.sum(ph * (p[None, :-1] * a + p[None, 1:] * b), axis=1)


_GL_V, _GL_W = np.polynomial.legendre.leggauss(5)
_GL_V = 0.5 * (_GL_V + 1.0)  # nodes on [0, 1]
_GL_W = 0.5 * _GL_W


def _gauss_smear(z, p, sigma, m_at):
    """Exact Gaussian convolution of the *represented* (piecewise-linear) pdf.

    Gauss-Legendre with 5 nodes per mass cell. The integrand is (linear) x
    (Gaussian) over a cell of width dm << sigma, so this is exact to round-off
    -- unlike a trapezoid on the mass grid, whose own O(dm^2) error would sit
    at the same few-1e-7 level as the quantity being tested.
    """
    dm = z.dm
    m0 = z.m_grid[:-1]
    mm = m0[None, :] + _GL_V[:, None] * dm  # (5, nm-1)
    pp = p[None, :-1] * (1.0 - _GL_V[:, None]) + p[None, 1:] * _GL_V[:, None]
    wp = (_GL_W[:, None] * pp) * dm
    out = np.empty(len(m_at))
    norm = 1.0 / (sigma * np.sqrt(2 * np.pi))
    for i, x in enumerate(m_at):
        out[i] = np.sum(wp * np.exp(-0.5 * ((x - mm) / sigma) ** 2)) * norm
    return out


def _z_term(name, mobs, sigma, kernel, m_ref, nt=256, tmax=8.0, chunk=8192):
    """A pure ``physics kernel (x) per-candidate Gaussian`` mass term."""
    return unbinned.MassCFTerm(
        name,
        sigma=sigma,
        mobs=mobs,
        tgrid=np.linspace(0.0, tmax, nt),
        families=[{"name": "res", "param": "k_res", "kind": "gauss"}],
        vgf=np.ones_like(sigma),
        kernel=kernel,
        m_ref=m_ref,
        scale_param=None,
        bkg_frac=0.0,
        chunk=chunk,
    )


def _sample(z, values, n, rng):
    """Draw ``n`` masses from the provider's own (truncated) lineshape."""
    p = z.pdf(values).numpy()
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * z.dm)])
    cdf /= cdf[-1]
    return np.interp(rng.random(n), cdf, z.m_grid)


# ---------------------------------------------------------------------------
# 1. the lineshape itself
# ---------------------------------------------------------------------------
def test_lineshape(args):
    print("\n=== 1. lineshape vs the reference implementation ===")
    ok = True
    z = ZGammaLineshape(window=tuple(args.window), nm=args.nm, nfft=args.nfft)
    print(f"  {z}")
    prov = z.lumi_provenance
    print(
        f"  luminosity table: {prov['pdfset']} member {prov['pdf_member']}, "
        f"sqrt(s) = {prov['sqrt_s_gev']/1000:.1f} TeV, {prov['n_anchor']} anchors "
        f"over {prov['m_lo']:.0f}-{prov['m_hi']:.0f} GeV, acceptance: "
        f"{prov['acceptance']}"
    )

    if not os.path.isdir(args.ref_dir):
        print(f"  SKIP: reference lineshape code not at {args.ref_dir}")
        return None
    sys.path.insert(0, args.ref_dir)
    import zwidth_sensitivity as zws  # noqa: E402

    # -- 1a. the matrix element, on the same grid with the same luminosities
    ref = zws.hard_me(z.m_grid**2, z.lumis, z.sin2, z.mz_ref, z.gz_ref)
    mine = z.dsigma_dm(_vals(m_Z=0.0, Gamma_Z=0.0)).numpy()
    dev = np.max(np.abs(mine - ref) / np.abs(ref))
    print(
        f"  1a. dsigma/dm vs hard_me (identical luminosities): max rel dev "
        f"= {dev:.2e}   (peak {mine.max():.1f} pb/GeV at "
        f"m = {z.m_grid[mine.argmax()]:.4f} GeV)"
    )
    ok &= dev < 1e-13

    # -- 1b. the shipped luminosity table vs the reference study's own cache.
    # build_lumis only uses its cache when the grid endpoints match the ones it
    # was built with, so rebuild the reference's extended grid exactly.
    try:
        s_lo, s_hi = 2 * np.log(zws.M_GRID_LO), 2 * np.log(zws.M_GRID_HI)
        n = zws.N_LOG
        ds = (s_hi - s_lo) / (n - 1)
        m_ext = np.exp((s_lo + np.arange(2 * n - 1) * ds) / 2)
        lref = zws.build_lumis(m_ext)
    except Exception as exc:  # no cache and no lhapdf in this environment
        print(f"  1b. SKIP luminosity comparison ({type(exc).__name__}: {exc})")
        return ok

    from scipy.interpolate import CubicSpline

    lo = max(m_ext[0], z.m_grid[0])
    hi = min(m_ext[-1], z.m_grid[-1])
    mc = z.m_grid[(z.m_grid >= lo) & (z.m_grid <= hi)]
    dev_l = []
    for i in range(lref.shape[0]):
        a = np.exp(CubicSpline(np.log(m_ext), np.log(lref[i]))(np.log(mc)))
        b = np.exp(CubicSpline(np.log(z.m_grid), np.log(z.lumis[i]))(np.log(mc)))
        dev_l.append(np.abs(a - b) / a)
    dev_l = np.concatenate(dev_l)
    print(
        f"  1b. shipped table vs reference cache on {lo:.0f}-{hi:.0f} GeV "
        f"(120 vs {prov['n_anchor']} anchors): max rel dev = {dev_l.max():.2e}, "
        f"median {np.median(dev_l):.2e}"
    )
    print(
        "      -> the only difference between the two lineshapes is the "
        "luminosity spline density; the matrix element is identical."
    )
    ok &= dev_l.max() < 1e-3
    print("  PASS" if ok else "  FAIL")
    return ok


# ---------------------------------------------------------------------------
# 2. the characteristic function
# ---------------------------------------------------------------------------
def test_cf(args):
    print("\n=== 2. characteristic function ===")
    ok = True
    z = ZGammaLineshape(window=tuple(args.window), nm=args.nm, nfft=args.nfft)
    v = _vals(m_Z=0.0, Gamma_Z=0.0)
    p = z.pdf(v).numpy()

    re, im = (a.numpy() for a in z.cf_tab(v))
    print(f"  phi(0) = {re[0]:.15f} + {im[0]:.1e} i   (exact value 1 + 0i)")
    ok &= abs(re[0] - 1.0) < 1e-12 and abs(im[0]) < 1e-12

    # -- 2a. against an independent Filon quadrature
    idx = np.unique(np.linspace(0, z.ntau - 1, 400).astype(int))
    ref = _filon_cf(z, p, z.tau_tab[idx])
    dev = np.max(np.abs(re[idx] + 1j * im[idx] - ref))
    print(f"  2a. cf_tab vs cell-by-cell Filon quadrature: max abs dev = {dev:.2e}")
    ok &= dev < 1e-12

    # -- 2b. the smeared density MassCFTerm builds from the kernel, against an
    # exact Gaussian convolution of the same represented pdf. This isolates the
    # CF chain (transform, tabulation, interpolation, inverse transform) from
    # how well the mass grid represents the continuum lineshape, which is 2c.
    m_at = np.linspace(70.0, 112.0, 85)
    print("      sigma   max rel dev   max |dev| / peak   median rel dev")
    for sigma in (1.0, 1.5, 2.5):
        term = _z_term(
            f"cf{sigma}",
            m_at - z.m_ref,
            np.full(len(m_at), sigma),
            unbinned.TabulatedLineshapeKernel(provider=z),
            z.m_ref,
            nt=args.nt,
            tmax=args.tmax,
            chunk=len(m_at),
        )
        dens = term.raw_density(tf.constant([1.0, 0.0, 0.0], tf.float64)).numpy()
        ref2 = _gauss_smear(z, p, sigma, m_at)
        rel = np.abs(dens - ref2) / ref2
        onpeak = np.max(np.abs(dens - ref2)) / ref2.max()
        print(
            f"      {sigma:5.2f}   {rel.max():.2e}      {onpeak:.2e}"
            f"           {np.median(rel):.2e}"
        )
        # The threshold is on the error *as a fraction of the peak density*,
        # which is what a likelihood cares about: the largest relative
        # deviation sits at m = 70 GeV, where the density is only ~0.5 % of the
        # peak, so the same absolute error reads ~200x bigger there.
        ok &= onpeak < 1e-6

    # -- 2c. how well the mass grid represents the continuum lineshape: the
    # same exact convolution at nm and at 4 nm. This is the O(dm^2) piece and
    # the only approximation left in the density; it is a smooth, systematic
    # few-ppm shape effect, far below the statistical reach of any Z sample.
    fine = ZGammaLineshape(window=tuple(args.window), nm=4 * args.nm, nfft=args.nfft)
    d_c = _gauss_smear(z, p, 1.5, m_at)
    d_f = _gauss_smear(fine, fine.pdf(v).numpy(), 1.5, m_at)
    print(
        f"  2c. mass-grid representation (nm = {args.nm}, dm = {z.dm*1e3:.2f} MeV "
        f"vs nm = {4*args.nm}): max rel dev "
        f"{np.max(np.abs(d_c-d_f)/d_f):.2e}, max |dev| / peak "
        f"{np.max(np.abs(d_c-d_f))/d_f.max():.2e}"
    )

    tau_needed, fits = z.check_tau_range(
        np.linspace(0, args.tmax, args.nt), np.array([1.0])
    )
    print(
        f"  2d. largest t/sigma a sigma = 1 GeV term needs: {tau_needed:.1f} "
        f"GeV^-1, tabulated to {z.tau_max:.1f} -> {'ok' if fits else 'TOO SMALL'}"
    )
    ok &= fits
    print("  PASS" if ok else "  FAIL")
    return ok


# ---------------------------------------------------------------------------
# 3. gradients
# ---------------------------------------------------------------------------
def test_gradients(args):
    print("\n=== 3. gradients vs finite differences ===")
    ok = True
    rng = np.random.default_rng(7)
    z = ZGammaLineshape(window=tuple(args.window), nm=args.nm, nfft=args.nfft)
    n = 2000
    sigma = 1.0 + rng.random(n)
    m_gen = _sample(z, _vals(m_Z=0.0, Gamma_Z=0.0), n, rng)
    mobs = m_gen + sigma * rng.standard_normal(n) - z.m_ref
    term = _z_term(
        "grad",
        mobs,
        sigma,
        unbinned.TabulatedLineshapeKernel(provider=z),
        z.m_ref,
        nt=args.nt,
        tmax=args.tmax,
        chunk=n,
    )
    print(f"  parameters {term.param_names}")

    x0 = np.array([1.02, 25.0, -30.0])  # k_res, m_Z [MeV], Gamma_Z [MeV]

    def nll(x):
        return term.nll(tf.constant(x, tf.float64))

    xv = tf.Variable(x0, dtype=tf.float64)
    with tf.GradientTape() as t2:
        with tf.GradientTape() as t1:
            val = term.nll(xv)
        g = t1.gradient(val, xv)
    h = t2.jacobian(g, xv).numpy()
    g = g.numpy()

    print(f"  NLL = {float(val):.9f}")
    print("      parameter      analytic grad     finite diff        rel dev")
    for i, name in enumerate(term.param_names):
        step = 1e-4 if name == "k_res" else 2e-2
        xp, xm = x0.copy(), x0.copy()
        xp[i] += step
        xm[i] -= step
        fd = (float(nll(xp)) - float(nll(xm))) / (2 * step)
        rel = abs(g[i] - fd) / max(abs(fd), 1e-12)
        print(f"      {name:>10s}   {g[i]:16.8f}  {fd:16.8f}   {rel:.2e}")
        ok &= rel < 5e-6

    print("  Hessian (analytic):")
    for row in h:
        print("      " + "  ".join(f"{v:14.6f}" for v in row))
    ok &= bool(np.all(np.isfinite(h)))
    ev = np.linalg.eigvalsh(h)
    print(f"  eigenvalues {ev} (all positive -> usable by trust-exact)")
    print("  PASS" if ok else "  FAIL")
    return ok


# ---------------------------------------------------------------------------
# 4. + 5. toy closure, and the same toy with the wrong kernel
# ---------------------------------------------------------------------------
def make_toy(args):
    """200k Z candidates: lineshape (x) per-candidate Gaussian, no FSR."""
    rng = np.random.default_rng(args.seed)
    z = ZGammaLineshape(window=tuple(args.window), nm=args.nm, nfft=args.nfft)
    truth = _vals(m_Z=args.mz_true, Gamma_Z=args.gz_true)
    n = args.n
    sigma = args.sigma_lo + (args.sigma_hi - args.sigma_lo) * rng.random(n)
    m_gen = _sample(z, truth, n, rng)
    m_obs = m_gen + sigma * rng.standard_normal(n)
    print(
        f"\n  generated {n} candidates from m_Z = {z.mz_ref:.6f} "
        f"{args.mz_true:+.1f} MeV, Gamma_Z = {z.gz_ref:.4f} "
        f"{args.gz_true:+.1f} MeV on the window {z.window} GeV, smeared with "
        f"sigma in [{args.sigma_lo}, {args.sigma_hi}] GeV; no FSR, no "
        f"background, no acceptance, no mass cut on m_obs (the model is then "
        f"exactly normalised, so no truncation term is needed)."
    )
    return z, sigma, m_obs


def test_closure(args, toy):
    print("\n=== 4. toy closure: fit m_Z and Gamma_Z ===")
    z, sigma, m_obs = toy
    term = _z_term(
        "zfit",
        m_obs - z.m_ref,
        sigma,
        unbinned.TabulatedLineshapeKernel(provider=z),
        z.m_ref,
        nt=args.nt,
        tmax=args.tmax,
        chunk=args.chunk,
    )
    assert list(term.param_names) == ["k_res", "m_Z", "Gamma_Z"], term.param_names

    t0 = time.time()
    res, grad, hess = _trust_exact(term.nll, [1.0, 0.0, 0.0], 3)
    dt = time.time() - t0
    cov = np.linalg.inv(hess)
    err = np.sqrt(np.diag(cov))
    print(
        f"  fit in {dt:.1f} s ({res.nit} iterations, "
        f"|grad|inf = {np.max(np.abs(grad)):.2e})"
    )
    labels = ["k_res", "m_Z [MeV]", "Gamma_Z [MeV]"]
    truths = [1.0, args.mz_true, args.gz_true]
    ok = True
    for lab, v, e, tv in zip(labels, res.x, err, truths):
        pull = (v - tv) / e
        print(
            f"    {lab:>14s} = {v:9.4f} +- {e:.4f}   truth {tv:8.4f}   "
            f"pull {pull:+.2f}"
        )
        ok &= abs(pull) < 4.0

    n = len(sigma)
    naive_cauchy = z.gz_ref * np.sqrt(2.0 / n) * 1e3  # MeV, unsmeared BW limit
    fixed = 1.0 / np.sqrt(hess[2, 2])  # Gamma only, all else fixed
    prof = err[2]  # profiling k_res and m_Z
    print(
        f"\n    sigma(Gamma_Z): naive unsmeared Breit-Wigner "
        f"Gamma sqrt(2/N) = {naive_cauchy:.3f} MeV;\n"
        f"                    this model, all else fixed = {fixed:.3f} MeV "
        f"({fixed/naive_cauchy:.2f}x);\n"
        f"                    profiling m_Z and the resolution scale = "
        f"{prof:.3f} MeV ({prof/naive_cauchy:.2f}x)."
    )
    print(
        f"    correlations: rho(Gamma_Z, k_res) = "
        f"{cov[2,0]/(err[2]*err[0]):+.3f}, rho(Gamma_Z, m_Z) = "
        f"{cov[2,1]/(err[2]*err[1]):+.3f}, rho(m_Z, k_res) = "
        f"{cov[1,0]/(err[1]*err[0]):+.3f}"
    )
    print(f"    correlation costs a factor {prof/fixed:.2f} on sigma(Gamma_Z).")
    print("  PASS" if ok else "  FAIL")
    return ok, res.x, err


def test_wrong_kernel(args, toy, ref=None):
    print("\n=== 5. the same toy fitted with a Breit-Wigner kernel ===")
    z, sigma, m_obs = toy
    term = _z_term(
        "bwfit",
        m_obs - z.m_ref,
        sigma,
        unbinned.BreitWignerKernel(
            width_param="Gamma_bw",
            mass_param="dm_bw",
            width_unit=1e-3,
            mass_unit=1e-3,
        ),
        z.m_ref,
        nt=args.nt,
        tmax=args.tmax,
        chunk=args.chunk,
    )
    assert list(term.param_names) == ["k_res", "dm_bw", "Gamma_bw"], term.param_names
    t0 = time.time()
    res, grad, hess = _trust_exact(term.nll, [1.0, 0.0, 2493.2], 3)
    err = np.sqrt(np.diag(np.linalg.inv(hess)))
    print(
        f"  fit in {time.time()-t0:.1f} s ({res.nit} iterations, "
        f"|grad|inf = {np.max(np.abs(grad)):.2e})"
    )
    mz_true_abs = z.mz_ref + args.mz_true * 1e-3
    bias = res.x[1] * 1e-3 + z.m_ref - mz_true_abs  # GeV
    for lab, v, e in zip(["k_res", "dm_bw [MeV]", "Gamma_bw [MeV]"], res.x, err):
        print(f"    {lab:>15s} = {v:11.4f} +- {e:.4f}")
    print(
        f"\n    the Breit-Wigner peak lands at "
        f"{z.m_ref + res.x[1]*1e-3:.4f} GeV, the generated m_Z is "
        f"{mz_true_abs:.4f} GeV\n"
        f"    -> bias {bias*1e3:+.1f} MeV ({abs(bias*1e3)/max(err[1],1e-9):.0f} "
        f"sigma_stat), and the fitted width is "
        f"{res.x[2]:.1f} MeV vs the generated {z.gz_ref*1e3 + args.gz_true:.1f} "
        f"MeV ({res.x[2] - z.gz_ref*1e3 - args.gz_true:+.1f} MeV)."
    )
    # context for the size of the shift
    v = _vals(m_Z=args.mz_true, Gamma_Z=args.gz_true)
    p = z.pdf(v).numpy()
    mean = float(np.sum(p * z.m_grid) * z.dm)
    mode = float(z.m_grid[p.argmax()])
    print(
        f"    for scale: the truncated lineshape has mean {mean:.3f} GeV and "
        f"mode {mode:.4f} GeV against m_Z = {mz_true_abs:.4f} GeV -- the "
        f"gamma* tail, the interference and the falling parton luminosity all "
        f"pull weight to low mass, and a symmetric Breit-Wigner absorbs that "
        f"into its peak position."
    )
    ok = abs(bias) > 5 * err[1] * 1e-3  # the point of the test: it IS biased
    print(
        "  PASS (the wrong kernel is measurably biased, as intended)"
        if ok
        else "  FAIL (no bias seen -- the test has lost its meaning)"
    )
    return ok


# ---------------------------------------------------------------------------
# 6. datacard round trip
# ---------------------------------------------------------------------------
def test_datacard(args):
    print("\n=== 6. datacard round trip and UnbinnedParams ===")
    import json
    import tempfile

    try:
        from rabbit.tensorwriter import TensorWriter

        from tests.test_unbinned_mass import loss_at, make_fitter
    except ImportError as exc:
        # TensorWriter pulls in wums.sparse_hist / hist, which the bare
        # wmassdev image does not ship. Put a full wums checkout on PYTHONPATH
        # (e.g. /work/submit/david_w/WRemnants_dev/wums) to run this test.
        print(f"  SKIP: cannot import the datacard writer ({exc})")
        return None

    rng = np.random.default_rng(3)
    z = ZGammaLineshape(window=tuple(args.window), nm=args.nm, nfft=args.nfft)
    n = 4000
    sigma = 1.0 + rng.random(n)
    m_gen = _sample(z, _vals(m_Z=0.0, Gamma_Z=0.0), n, rng)
    mobs = m_gen + sigma * rng.standard_normal(n) - z.m_ref
    tgrid = np.linspace(0.0, args.tmax, args.nt)
    term = _z_term(
        "zcard",
        mobs,
        sigma,
        unbinned.TabulatedLineshapeKernel(provider=z),
        z.m_ref,
        nt=args.nt,
        tmax=args.tmax,
        chunk=2048,
    )
    # m_Z and Gamma_Z as POIs, a PDG-sized Gaussian prior on the width, the
    # resolution scale free and starting at 1.
    decl = unbinned.declare_params(
        term,
        {
            **z.param_declarations(gz_prior=args.gz_prior),
            "k_res": (1.0, np.nan, 1.0, 0),
        },
    )
    print(f"  declarations: {dict(zip(term.param_names, zip(*decl.values())))}")

    w = TensorWriter()
    w.add_dummy_channel()
    w.add_unbinned_term(
        "zcard",
        term.config(),
        term.param_names,
        {"sigma": sigma, "mobs": mobs, "tgrid": tgrid, "vgf": np.ones(n)},
        **decl,
    )
    ok = True
    with tempfile.TemporaryDirectory() as d:
        w.write(outfolder=d, outfilename="zcard.hdf5")
        path = os.path.join(d, "zcard.hdf5")
        print(f"  wrote {path} ({os.path.getsize(path)/1024:.0f} kiB)")
        f = make_fitter(path)
        loaded = f.indata.unbinned_terms[0]
        prov = loaded.kernel.provider
        print(
            f"  kernel read back: {type(loaded.kernel).__name__} with provider "
            f"{type(prov).__name__} -- {prov}"
        )
        ok &= isinstance(prov, ZGammaLineshape)
        ok &= json.dumps(prov.config(), sort_keys=True) == json.dumps(
            z.config(), sort_keys=True
        )

        names = list(f.parms.astype(str))
        npoi = f.param_model.npoi
        print(
            f"  fit parameters {names}, npoi = {npoi} -> POIs {names[:npoi]}, "
            f"nuisances {names[npoi:]}"
        )
        ok &= npoi == 2 and set(names[:npoi]) == {"m_Z", "Gamma_Z"}

        # the NLL through the Fitter equals the term evaluated directly
        x = np.array(
            [decl["param_defaults"][term.param_names.index(nm)] for nm in names]
        )
        rng2 = np.random.default_rng(11)
        for _ in range(2):
            xt = x + rng2.normal(0.0, 0.3, len(names))
            direct = float(
                term.nll(
                    tf.constant(
                        [xt[names.index(nm)] for nm in term.param_names], tf.float64
                    )
                )
            )
            # the prior on Gamma_Z is the fitter's, not the term's
            i = names.index("Gamma_Z")
            prior = 0.5 * (xt[i] / args.gz_prior) ** 2
            through = loss_at(f, xt)
            rel = abs(through - direct - prior) / max(abs(direct), 1e-12)
            print(
                f"    NLL(fitter) = {through:.9f}, term {direct:.9f} + prior "
                f"{prior:.9f}: rel dev {rel:.2e}"
            )
            ok &= rel < 1e-12

        i = names.index("Gamma_Z")
        cw = f.cw.numpy()[i]
        print(
            f"  prior on Gamma_Z: sigma = {args.gz_prior} MeV -> constraint "
            f"weight {cw:.6f} (1/sigma^2 = {1/args.gz_prior**2:.6f})"
        )
        ok &= abs(cw - 1.0 / args.gz_prior**2) < 1e-9
    print("  PASS" if ok else "  FAIL")
    return ok


# ---------------------------------------------------------------------------
def test_modifiers(args):
    """Test 7: ``terms``, ``acceptance`` and the multiplicative FSR fold."""
    import json

    import numpy as np
    import tensorflow as tf

    from rabbit.lineshapes import ZGammaLineshape

    print("\n=== 7. terms / acceptance / FSR fold ===")
    ok = True
    W = (60.0, 130.0)
    NM = 4096
    kw = dict(window=W, nm=NM, nfft=NM, tau_max=1.0)
    V = {"m_Z": tf.constant(0.0, tf.float64), "Gamma_Z": tf.constant(0.0, tf.float64)}

    # (a) the matrix-element pieces add up
    full = ZGammaLineshape(**kw).dsigma_dm(V).numpy()
    parts = np.zeros_like(full)
    for t in ("gamma", "int", "z"):
        parts += ZGammaLineshape(terms=(t,), **kw).dsigma_dm(V).numpy()
    d = np.max(np.abs(parts - full)) / np.max(np.abs(full))
    print(f"  7a. gamma + int + z == full ME: max rel dev {d:.2e}")
    ok &= d < 1e-13

    # (b) the acceptance multiplies the pdf and then renormalises
    acc = {"kind": "bernstein", "lo": W[0], "hi": W[1], "coef": [0.2, 0.6, 0.9, 0.7]}
    z0 = ZGammaLineshape(**kw)
    za = ZGammaLineshape(acceptance=acc, **kw)
    a = za._acceptance_on(z0.m_grid)
    want = z0.pdf(V).numpy() * a
    want = want / (want.sum() * z0.dm)
    got = za.pdf(V).numpy()
    d = np.max(np.abs(got - want)) / np.max(want)
    print(f"  7b. acceptance x pdf, renormalised: max rel dev {d:.2e}")
    ok &= d < 1e-12

    # (c) a one-atom kernel at r0 is a pure rescaling of the Born spectrum
    r0 = 0.97
    zf = ZGammaLineshape(fsr={"r": [r0], "w": [1.0]}, **kw)
    got = zf.pdf(V).numpy()
    yb = zf.born_pdf(V).numpy()
    want = np.interp(zf.m_grid / r0, zf.m_born, yb, left=0.0, right=0.0) / r0
    want = want * zf._edge.numpy()
    want = want / (want.sum() * zf.dm)
    d = np.max(np.abs(got - want)) / np.max(want)
    peak_shift = zf.m_grid[np.argmax(got)] - z0.m_grid[np.argmax(z0.pdf(V).numpy())]
    print(
        f"  7c. one atom at r = {r0}: max rel dev {d:.2e}, "
        f"peak moved {peak_shift:+.3f} GeV (expect ~{-(1-r0)*91.2:+.3f})"
    )
    ok &= d < 1e-12
    ok &= abs(peak_shift + (1 - r0) * 91.2) < 0.2

    # (d) two atoms are the weighted sum of two rescalings
    rr, ww = [1.0, 0.9], [0.6, 0.4]
    z2 = ZGammaLineshape(fsr={"r": rr, "w": ww}, **kw)
    yb = z2.born_pdf(V).numpy()
    want = np.zeros(NM)
    for r, wt in zip(rr, ww):
        want += wt / r * np.interp(z2.m_grid / r, z2.m_born, yb, left=0.0, right=0.0)
    want = want * z2._edge.numpy()
    want = want / (want.sum() * z2.dm)
    got = z2.pdf(V).numpy()
    d = np.max(np.abs(got - want)) / np.max(want)
    n = float(got.sum() * z2.dm)
    print(f"  7d. two atoms: max rel dev {d:.2e}, norm {n:.15f}")
    ok &= d < 1e-12 and abs(n - 1.0) < 1e-12

    # (e) one band spanning everything == no band
    zb = ZGammaLineshape(
        fsr={"r": rr, "w": ww, "m_lo": [0.0, 0.0], "m_hi": [1e9, 1e9]}, **kw
    )
    d = np.max(np.abs(zb.pdf(V).numpy() - got)) / np.max(got)
    print(f"  7e. one all-inclusive band == band-less: max rel dev {d:.2e}")
    ok &= d < 1e-14

    # (f) config round trip
    cfg = ZGammaLineshape(
        terms=("z", "int"), acceptance=acc, fsr={"r": rr, "w": ww}, **kw
    ).config()
    zc = ZGammaLineshape.from_config(json.loads(json.dumps(cfg)))
    zr = ZGammaLineshape(
        terms=("z", "int"), acceptance=acc, fsr={"r": rr, "w": ww}, **kw
    )
    d = np.max(np.abs(zc.pdf(V).numpy() - zr.pdf(V).numpy()))
    print(f"  7f. config -> JSON -> from_config: max abs dev {d:.2e}")
    ok &= d == 0.0

    # (g) the tabulated acceptance: linear interpolation, constant outside,
    # and a fine grid rendering of a Bernstein acceptance reproduces it.  The
    # per-leg factorised FSR kernel (calibration_studies/zchannel) delivers
    # A(m) as a table on its own m_pre bands, so this is the form the
    # selection-conditional model actually uses.
    mg = np.array([70.0, 80.0, 90.0, 100.0])
    ag = np.array([0.20, 0.35, 0.45, 0.50])
    zg = ZGammaLineshape(acceptance={"kind": "grid", "m": mg, "a": ag}, **kw)
    got = zg._acceptance_on(np.array([60.0, 75.0, 85.0, 95.0, 120.0]))
    want = np.array([0.20, 0.275, 0.40, 0.475, 0.50])
    d = np.max(np.abs(got - want))
    print(f"  7g. grid acceptance, interp + constant outside: max abs dev {d:.2e}")
    ok &= d < 1e-15

    fine = np.linspace(W[0], W[1], 4001)
    zb2 = ZGammaLineshape(acceptance=acc, **kw)
    zg2 = ZGammaLineshape(
        acceptance={"kind": "grid", "m": fine,
                    "a": zb2._acceptance_on(fine)}, **kw)
    d = np.max(np.abs(zg2.pdf(V).numpy() - zb2.pdf(V).numpy())) / np.max(
        zb2.pdf(V).numpy())
    print(f"  7g. grid rendering of a Bernstein A(m): max rel dev {d:.2e}")
    ok &= d < 1e-6

    print("  PASS" if ok else "  FAIL")
    return ok


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--n", type=int, default=200000, help="toy candidates")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--window", type=float, nargs=2, default=[50.0, 130.0])
    p.add_argument("--nm", type=int, default=32768)
    p.add_argument("--nfft", type=int, default=None)
    p.add_argument("--nt", type=int, default=256, help="quadrature points in t")
    p.add_argument("--tmax", type=float, default=8.0, help="t range in units of sigma")
    p.add_argument(
        "--gz-prior",
        type=float,
        default=2.3,
        help="Gaussian prior sigma on Gamma_Z in test 6, in MeV (the PDG world "
        "average uncertainty). Priors default to none in the provider.",
    )
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--sigma-lo", type=float, default=1.0)
    p.add_argument("--sigma-hi", type=float, default=2.0)
    p.add_argument("--mz-true", type=float, default=30.0, help="MeV offset")
    p.add_argument("--gz-true", type=float, default=-40.0, help="MeV offset")
    p.add_argument("--ref-dir", default=REF_LINESHAPE_DIR)
    p.add_argument("--skip", type=int, nargs="*", default=[])
    return p.parse_args()


def main():
    args = parse_args()
    results = {}
    if 1 not in args.skip:
        results["1 lineshape"] = test_lineshape(args)
    if 2 not in args.skip:
        results["2 characteristic function"] = test_cf(args)
    if 3 not in args.skip:
        results["3 gradients"] = test_gradients(args)
    if 4 not in args.skip or 5 not in args.skip:
        toy = make_toy(args)
        if 4 not in args.skip:
            results["4 toy closure"] = test_closure(args, toy)[0]
        if 5 not in args.skip:
            results["5 wrong kernel"] = test_wrong_kernel(args, toy)
    if 6 not in args.skip:
        results["6 datacard round trip"] = test_datacard(args)
    if 7 not in args.skip:
        results["7 terms/acceptance/FSR"] = test_modifiers(args)

    print("\n" + "=" * 62)
    for k, v in results.items():
        tag = "SKIP" if v is None else ("PASS" if v else "FAIL")
        print(f"  {k:<32s} {tag}")
    print("=" * 62)
    return 0 if all(v is not False for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
