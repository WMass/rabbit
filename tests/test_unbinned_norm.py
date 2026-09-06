#!/usr/bin/env python3
"""Tests of the mass-window truncation normalisation of ``MassCFTerm``.

A sample selected in a mass window has the truncated likelihood
``prod_i L_i(m_i) / Z_i``, ``Z_i = Int_window L_i(m) dm``. ``norm_window``
switches that on; ``Z`` is evaluated on a mass grid for a small number of
resolution *classes* and gathered per candidate.

1. exact Gaussian -- ``Z`` against the error function
2. convergence in ``norm_tpoints``
3. convergence in the number of resolution classes
4. equivalence with the hand-rolled ``- sum log L + n log Z`` of
   ``test_unbinned_mass.py`` test 4b (on a fine ``t`` grid, where the mass-grid
   route is itself accurate), in value *and* gradient
5. closure: a Voigt toy cut to a window, fitted with and without the term
6. ``upsample``: in-graph expansion of the exponents equals building the term
   on the finer grid, and the density converges in the integration grid

Run: ``python tests/test_unbinned_norm.py``
"""

import sys
import time

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
from scipy.special import erf

from rabbit import unbinned

DTYPE = tf.float64


def gauss_term(name, mobs, sigma, nt=512, tmax=8.0, m_ref=0.0, window=None,
               nclass=1, gamma_param=None, chunk=100000, tpoints=8192):
    """Gaussian (+ optional Breit-Wigner) term, optionally window-normalised."""
    tgrid = np.linspace(0.0, tmax, nt)
    vgf = np.ones(len(sigma))
    families = [{"name": "res", "param": "k_res", "kind": "gauss"}]
    kernel = (
        unbinned.DeltaKernel()
        if gamma_param is None
        else unbinned.BreitWignerKernel(gamma_param, width_unit=1e-3)
    )
    norm = None
    if window is not None:
        edges = np.quantile(sigma, np.linspace(0, 1, nclass + 1))
        cls = np.clip(np.searchsorted(edges[1:-1], sigma, "right"), 0, nclass - 1)
        sig_c = np.array(
            [np.median(sigma[cls == c]) if np.any(cls == c) else sigma.mean()
             for c in range(nclass)]
        )
        norm = {"sigma": sig_c, "vgf": np.ones(nclass), "class": cls,
                "families": []}
    return unbinned.MassCFTerm(
        name, sigma=sigma, mobs=mobs, tgrid=tgrid, families=families, vgf=vgf,
        kernel=kernel, m_ref=m_ref, norm_window=window, norm_tpoints=tpoints,
        norm=norm, chunk=chunk, dtype=DTYPE,
    )


def phi(x):
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def test1():
    print("\n[1] exact Gaussian: Z against the error function")
    rng = np.random.default_rng(7)
    n = 2000
    m_ref = 91.1876
    sigma = rng.uniform(0.6, 2.5, n)
    mobs = rng.normal(0.0, 1.5, n)
    # a *narrow* window, so Z is genuinely fractional (a 60-120 GeV window is
    # 24 sigma wide and gives Z = 1 to round-off, which tests nothing)
    lo, hi = m_ref - 2.0, m_ref + 2.0
    ok = True
    for nodes in (1024, 4096, 16384):
        t = gauss_term("g", mobs, sigma, window=(lo, hi), nclass=8,
                       tpoints=nodes, m_ref=m_ref)
        z = t._norm_z({"k_res": tf.constant(1.0, DTYPE)}).numpy()
        sc = np.asarray(t._norm["sigma"])
        exact = phi((hi - m_ref) / sc) - phi((lo - m_ref) / sc)
        dev = np.max(np.abs(z - exact))
        print(f"    t points={nodes:6d}  max |Z - erf|  = {dev:.3e}   "
              f"(Z in [{z.min():.6f}, {z.max():.6f}])")
        if nodes == 16384:
            ok &= dev < 1e-6
    # a scaled resolution must still be exact
    t = gauss_term("g", mobs, sigma, window=(lo, hi), nclass=8, tpoints=16384,
                   m_ref=m_ref)
    for k in (0.5, 2.0):
        z = t._norm_z({"k_res": tf.constant(k, DTYPE)}).numpy()
        sc = np.asarray(t._norm["sigma"]) * np.sqrt(k)
        dev = np.max(np.abs(z - (phi((hi - m_ref) / sc) - phi((lo - m_ref) / sc))))
        print(f"    k_res={k:4.1f}   max |Z - erf|  = {dev:.3e}")
        ok &= dev < 1e-6
    print("    (the residual is the midpoint-rule error of the Fourier "
          "quadrature)")
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def test2():
    print("\n[2] convergence in norm_tpoints (Voigt, heavy tails)")
    rng = np.random.default_rng(8)
    n = 500
    sigma = rng.uniform(0.6, 2.5, n)
    mobs = rng.normal(0.0, 1.5, n)
    ref = None
    ok = True
    for nodes in (256, 512, 1024, 2048, 8192, 32768):
        t = gauss_term("v", mobs, sigma, window=(60.0, 120.0), nclass=4,
                       tpoints=nodes, m_ref=91.1876, gamma_param="gam")
        z = t._norm_z({"k_res": tf.constant(1.0, DTYPE),
                       "gam": tf.constant(2493.2, DTYPE)}).numpy()
        if ref is None:
            first = z
        ref = z
        print(f"    t points={nodes:6d}  Z = " + " ".join(f"{v:.9f}" for v in z))
    # the 32768-point answer is the reference; check 8192 is already there
    t = gauss_term("v", mobs, sigma, window=(60.0, 120.0), nclass=4, tpoints=8192,
                   m_ref=91.1876, gamma_param="gam")
    z257 = t._norm_z({"k_res": tf.constant(1.0, DTYPE),
                      "gam": tf.constant(2493.2, DTYPE)}).numpy()
    dev = np.max(np.abs(z257 - ref) / ref)
    print(f"    8192 vs 32768 t points: max rel dev = {dev:.3e}")
    ok &= dev < 1e-5
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def test3():
    print("\n[3] cost of the resolution-class approximation")
    print("    K classes replace the exact per-candidate Z. What matters is "
          "how fast Z\n    varies with sigma, i.e. how close the window edge "
          "is in units of sigma.")
    rng = np.random.default_rng(9)
    n = 5000
    sigma = rng.uniform(0.5, 3.0, n)
    mobs = rng.normal(0.0, 1.5, n)
    m_ref = 91.1876
    ok = True
    for half, tag in ((2.0, "narrow, edge at 1-4 sigma (worst case)"),
                      (8.0, "edge at 3-16 sigma"),
                      (30.0, "the real Z window, edge at 10-60 sigma")):
        lo, hi = m_ref - half, m_ref + half
        exact = phi((hi - m_ref) / sigma) - phi((lo - m_ref) / sigma)
        print(f"    window +-{half:4.1f} GeV ({tag}): Z in "
              f"[{exact.min():.6f}, {exact.max():.6f}]")
        devs = {}
        for nclass in (1, 4, 16, 32):
            t = gauss_term("g", mobs, sigma, window=(lo, hi), nclass=nclass,
                           tpoints=8192, m_ref=m_ref)
            z = t._norm_z({"k_res": tf.constant(1.0, DTYPE)}).numpy()
            per = np.asarray(tf.gather(z, t._norm_class))
            devs[nclass] = np.max(np.abs(per - exact) / exact)
            print(f"        K={nclass:3d}  max rel dev = {devs[nclass]:.3e}")
        # the class error must fall at least ~linearly with K
        ok &= devs[32] < 0.25 * devs[1] + 1e-12
        if half == 30.0:
            ok &= devs[16] < 1e-6
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def test4():
    print("\n[4] equivalence with the hand-rolled '- sum log L + n log Z'")
    rng = np.random.default_rng(10)
    n = 4000
    sig0 = 1.2
    sigma = np.full(n, sig0)
    mobs = rng.normal(0.0, 2.0, n)
    lo, hi = 60.0, 120.0
    m_ref = 91.1876
    keep = (mobs + m_ref > lo) & (mobs + m_ref < hi)
    mobs, sigma = mobs[keep], sigma[keep]
    nk = len(mobs)

    # the mass-grid reference has to be evaluated on a *fine* t grid: its own
    # integrand oscillates |m_edge - m| / sigma ~ 25 times, which the term's
    # working grid (64 in-maker points, 512 here) does not resolve. That is the
    # whole reason _norm_z does the integral in Fourier space instead.
    NT = 8192
    term = gauss_term("a", mobs, sigma, window=(lo, hi), nclass=1, tpoints=16384,
                      m_ref=m_ref, gamma_param="gam", nt=NT)
    plain = gauss_term("b", mobs, sigma, m_ref=m_ref, gamma_param="gam", nt=NT)
    grid = np.linspace(lo, hi, 16385)
    gterm = gauss_term("c", grid - m_ref, np.full(len(grid), sig0), m_ref=m_ref,
                       gamma_param="gam", chunk=len(grid), nt=NT)
    dg = tf.constant(np.diff(grid), DTYPE)
    names = ["k_res", "gam"]

    def f_new(x):
        return term.nll(tf.constant(x, DTYPE))

    def f_ref(x):
        v = tf.constant(x, DTYPE)
        d = gterm.raw_density(v)
        z = tf.reduce_sum(dg * (d[1:] + d[:-1]) * 0.5)
        return plain.nll(v) + tf.constant(float(nk), DTYPE) * tf.math.log(z)

    ok = True
    for x in ([1.0, 2493.2], [1.15, 2200.0], [0.9, 2800.0]):
        xv = tf.constant(x, DTYPE)
        with tf.GradientTape() as t1:
            t1.watch(xv)
            a = term.nll(xv)
        ga = t1.gradient(a, xv).numpy()
        with tf.GradientTape() as t2:
            t2.watch(xv)
            b = f_ref(xv)
        gb = t2.gradient(b, xv).numpy()
        dn = abs(a.numpy() - b.numpy()) / abs(b.numpy())
        dgr = np.max(np.abs(ga - gb) / (np.abs(gb) + 1e-30))
        print(f"    x={x}: NLL rel dev {dn:.3e}, grad rel dev {dgr:.3e}")
        ok &= dn < 1e-5 and dgr < 1e-4
    print("    (both routes are quadrature-limited at ~1e-6 relative here -- "
          "refining the\n     mass grid from 2049 to 16385 nodes does not move "
          "the number, and test 2 shows\n     the Fourier Z converging at the "
          "same 1e-6. The point is that they agree.)")
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def _fit(fun, x0, npar):
    def val_grad(x):
        xv = tf.constant(x, DTYPE)
        with tf.GradientTape() as t:
            t.watch(xv)
            v = fun(xv)
        return v.numpy(), t.gradient(v, xv).numpy()

    def hess(x):
        xv = tf.constant(x, DTYPE)
        with tf.GradientTape() as t2:
            t2.watch(xv)
            with tf.GradientTape() as t1:
                t1.watch(xv)
                v = fun(xv)
            g = t1.gradient(v, xv)
        return t2.jacobian(g, xv).numpy()

    r = minimize(val_grad, x0, jac=True, hess=hess, method="trust-exact")
    return r, hess(r.x)


def test5():
    print("\n[5] closure: Voigt sample cut to a window")
    rng = np.random.default_rng(11)
    n = 200000
    m_ref = 91.1876
    sig0 = 1.2
    gamma_true = 2493.2  # MeV
    lo, hi = 60.0, 120.0
    truth = 0.5 * gamma_true * 1e-3 * rng.standard_cauchy(n)
    mobs_all = truth + sig0 * rng.standard_normal(n)
    keep = (mobs_all + m_ref > lo) & (mobs_all + m_ref < hi)
    mobs = mobs_all[keep]
    sigma = np.full(len(mobs), sig0)
    print(f"    {len(mobs)} of {n} inside [{lo}, {hi}] "
          f"({100*(1-keep.mean()):.2f} % cut away)")

    ok = True
    for label, window in (("with norm_window", (lo, hi)), ("without", None)):
        t = gauss_term("f", mobs, sigma, window=window, nclass=1, tpoints=16384,
                       m_ref=m_ref, gamma_param="gam", chunk=200000)
        t0 = time.time()
        res, H = _fit(lambda x: t.nll(x), [1.0, 2400.0], 2)
        err = np.sqrt(np.diag(np.linalg.inv(H)))
        pull = (res.x[1] - gamma_true) / err[1]
        print(f"    {label:18s}: k_res = {res.x[0]:.5f} +- {err[0]:.5f}, "
              f"Gamma = {res.x[1]:8.2f} +- {err[1]:5.2f} MeV "
              f"(truth {gamma_true}, pull {pull:+.1f}) [{time.time()-t0:.0f} s]")
        if window is not None:
            ok &= abs(pull) < 3.0
        else:
            ok &= abs(pull) > 5.0  # the point of the test: it *must* be biased
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def test6():
    print("\n[6] in-graph upsampling")
    from scipy.interpolate import CubicSpline

    rng = np.random.default_rng(12)
    n, nt = 400, 64
    m_ref = 91.1876
    tmax = 7.8926
    t0 = np.linspace(0.0, tmax, nt)
    sigma = rng.uniform(0.5, 3.0, n)
    mobs = rng.uniform(-31.0, 29.0, n)
    vgf = rng.uniform(0.1, 0.5, n)
    # a smooth, physical-looking tabulated family plus an odd imaginary part
    v = rng.uniform(0.3, 0.9, n)[:, None]
    sre = -0.5 * v * t0[None, :] ** 2 * np.exp(-0.03 * t0[None, :])
    sim = 0.02 * v * t0[None, :] ** 3 / (1.0 + t0[None, :])
    ttab = np.linspace(0.0, 20.0, 4001)
    dmk = -np.abs(rng.standard_cauchy(4000)) * 0.5
    phik = np.mean(np.exp(1j * np.outer(ttab, dmk)), axis=1)

    def build(tg, sr, si, ups):
        return unbinned.MassCFTerm(
            "u", sigma=sigma, mobs=mobs, tgrid=tg, vgf=vgf,
            families=[{"name": "hit", "param": "k_hit", "kind": "gauss"},
                      {"name": "ms", "param": "k_ms", "kind": "tab",
                       "re": sr, "im": si}],
            phik=(ttab, phik.real.copy(), phik.imag.copy()),
            m_ref=m_ref, upsample=ups, chunk=n, dtype=DTYPE)

    vals = tf.constant([1.0, 1.0], DTYPE)
    ok = True
    for f in (4, 16):
        t1 = np.linspace(0.0, tmax, (nt - 1) * f + 1)
        pre = build(t1, CubicSpline(t0, sre, axis=1)(t1),
                    CubicSpline(t0, sim, axis=1)(t1), 1)
        ing = build(t0, sre, sim, f)
        a = pre.raw_density(vals).numpy()
        b = ing.raw_density(vals).numpy()
        dev = np.max(np.abs(a - b) / np.abs(a))
        print(f"    upsample {f:3d}: in-graph vs pre-splined density, "
              f"max rel dev = {dev:.3e}   (NLL {pre.nll(vals).numpy():.9f} vs "
              f"{ing.nll(vals).numpy():.9f})")
        ok &= dev < 1e-8
    base = build(t0, sre, sim, 1)
    prev = base.raw_density(vals).numpy()
    print("    convergence of the density with the integration grid "
          "(median |rel change|):")
    for f in (2, 4, 8, 16, 32):
        cur = build(t0, sre, sim, f).raw_density(vals).numpy()
        print(f"      {f:3d}x ({(nt-1)*f+1:5d} points): "
              f"{np.median(np.abs(cur - prev) / np.abs(cur)):.3e}")
        prev = cur
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    skip = set()
    if "--skip" in sys.argv:
        skip = {int(a) for a in sys.argv[sys.argv.index("--skip") + 1:]}
    results = {}
    for i, f in enumerate((test1, test2, test3, test4, test5, test6), 1):
        if i in skip:
            continue
        results[i] = f()
    print("\n" + "=" * 60)
    for i, r in results.items():
        print(f"  test {i}: {'PASS' if r else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)
