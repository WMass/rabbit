#!/usr/bin/env python3
"""`MassCFTerm(corr_form="fluctuation")`: the two corrections of
MASSCFTERM_SPEC applied INSIDE the convolution.

The residual form (`corr_form="residual"`, the historical one) evaluates the
self-consistent width at `delta_i(theta) = m_i - M(theta)` and inverts the
Jensen map on it.  Both are expansions in the RESOLUTION fluctuation, and at
the J/psi `delta_i` IS that fluctuation, which is where the spec's gates were
measured.  At the Z it is not: the window is +-27 sigma and what sits out there
is the Breit-Wigner tail and FSR.

The fluctuation form writes the two as ONE deterministic map of the
fluctuation, `m_i = m_true + u_i(x)` with `u_i(x) = sigma_i x + c_i x^2 + d_i`,
and integrates over `x` inside the convolution.  In Fourier space that is one
multiplicative factor `1 + w_i(tau)` on the resolution CF plus a shift `d_i` of
the residual.  Checks:

1. OFF (no `a_res`, no `jensen_s2`) is BIT-IDENTICAL to the term that has no
   correction at all, in either form.
2. The modelled density is right: sampled on a mass grid with a delta kernel it
   normalises to 1 and its MEAN is `v (c_i - a_i sigma_i) + d_i` with
   `v = Var(x)` -- the closed form the correction exists to install.  (`v` is 1
   in the real model, where the families together carry the whole `sigma_i^2`;
   here the single Gaussian family carries `vgf`, which is what makes this a
   test of the `phi'`/`phi''` construction rather than of a coincidence.)
3. With a DELTA kernel the fluctuation form reproduces the residual form's
   density -- the residual form is exact there, so this is the J/psi gate in
   miniature.  The two must agree to the order of the expansion.
4. The correction is BOUNDED: at Z-like `sigma/m` over a +-27 sigma window the
   residual form's Jensen map moves the residual by many GeV while the
   fluctuation form's deterministic shift stays at `d_i` (a few MeV) and its CF
   factor stays O(1e-2).
5. The analytic gradient of the corrected NLL matches central finite
   differences.
6. `c_i = -vgf_i sigma_i^2/m_i`: the two quadratic coefficients cancel, which
   is why the naive Z bias is -15...-27 MeV and not the full -35.
"""
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rabbit import unbinned  # noqa: E402

DT = tf.float64
MREF = 91.1876
NT = 96
TMAX = 7.8926


def build(sigma, mobs, *, vgf=None, a_res=None, jensen_s2=None, kernel=None,
          nt=NT, mref=MREF, chunk=4096, **kw):
    """A term whose ONLY resolution family is the analytic Gaussian, so `x` is
    exactly standard normal and every moment below is closed-form."""
    n = len(sigma)
    tgrid = np.linspace(0.0, TMAX, nt)
    if vgf is None:
        vgf = np.ones(n)
    fam = [{"name": "hit", "param": "k_hit", "kind": "gauss"}]
    t = unbinned.MassCFTerm(
        "f", sigma=np.asarray(sigma, float), mobs=np.asarray(mobs, float),
        tgrid=tgrid, families=fam, vgf=np.asarray(vgf, float),
        kernel=kernel if kernel is not None else unbinned.DeltaKernel(),
        m_ref=mref, scale_param="alpha", chunk=chunk,
        a_res=a_res, jensen_s2=jensen_s2, **{"floor": "none", **kw})
    decl = {"k_hit": (1.0, np.nan, 1.0, 0), "alpha": (0.0, np.nan, 0.0, 1)}
    for p in t.param_names:
        decl.setdefault(p, (0.0, np.nan, 0.0, 0))
    unbinned.declare_params(t, decl)
    return t


def values(t, **over):
    v = {p: float(d) for p, d in zip(t.param_names, t.param_defaults)}
    v.update(over)
    return {p: tf.constant(v[p], DT) for p in t.param_names}


def dens(t, **over):
    return t.raw_density(tf.constant(
        [float(values(t, **over)[p].numpy()) for p in t.param_names], DT)).numpy()


def nll(t, **over):
    return float(t.nll(tf.constant(
        [float(values(t, **over)[p].numpy()) for p in t.param_names], DT)).numpy())


# ---------------------------------------------------------------------------
def test_off_is_identical():
    print("\n=== 1. no correction -> bit-identical in both forms ===")
    rng = np.random.default_rng(1)
    n = 512
    sigma = 0.8 + 0.6 * rng.random(n)
    mobs = sigma * rng.standard_normal(n)
    base = build(sigma, mobs)
    res = build(sigma, mobs, corr_form="residual")
    flu = build(sigma, mobs, corr_form="fluctuation")
    a, b, c = nll(base), nll(res), nll(flu)
    print(f"  none {a!r}\n  residual {b!r}\n  fluctuation {c!r}")
    assert a == b == c, (a, b, c)
    assert flu._fluct_active is False
    # a_res present but all zero, jensen_s2 present but all zero
    z = build(sigma, mobs, a_res=np.zeros(n), jensen_s2=np.zeros(n),
              corr_form="fluctuation")
    d = nll(z)
    print(f"  a_res=0, jensen_s2=0 -> {d!r}   identical: {d == a}")
    assert d == a, (d, a)
    print("  PASS")


def test_density_moments():
    print("\n=== 2. the modelled density: norm 1 and mean c + d - a sigma ===")
    # one candidate repeated over a fine mass GRID: raw_density then IS
    # p_i(Delta) sampled, because the kernel is a delta.
    sig = 1.10
    m = MREF
    dgrid = np.linspace(-12.0 * sig, 12.0 * sig, 24001)
    n = len(dgrid)
    sigma = np.full(n, sig)
    vgf = np.full(n, 0.72)
    step = dgrid[1] - dgrid[0]

    def moments(a_val, s2_val, jensen):
        a_res = None if a_val is None else np.full(n, a_val)
        s2 = None if s2_val is None else np.full(n, s2_val)
        t = build(sigma, dgrid, vgf=vgf, a_res=a_res, jensen_s2=s2,
                  corr_form="fluctuation",
                  jensen_mode="exact" if jensen else "off")
        # ONE candidate's density sampled on a mass grid: pin `c_i` and `d_i`
        # to the reference mass, because the rows of this term are grid points
        # and the term would otherwise (correctly, for real candidates) give
        # each its own `m_i = mobs_i + m_ref` -- a +-15 % spread of c and d
        # across a +-13 GeV grid, which is not what a single density is.
        if t._fluct_active:
            t._fl_g = tf.constant(
                np.full(n, (0.0 if a_val is None else -a_val)
                        + (sig / m if jensen else 0.0)), tf.float64)
            t._fl_d = (None if not jensen
                       else tf.constant(np.full(n, 0.5 * s2_val * m), tf.float64))
        p = dens(t)
        z = p.sum() * step
        mu = (p * dgrid).sum() * step / z
        return z, mu, t

    aval = (1.0 + 0.72) * sig / m           # the make_card.py formula
    s2val = (sig / m) ** 2
    for label, a_val, s2_val, jensen in (
        ("a only            ", aval, None, False),
        ("jensen only       ", None, s2val, True),
        ("both (the default)", aval, s2val, True),
    ):
        z, mu, t = moments(a_val, s2_val, jensen)
        a_i = 0.0 if a_val is None else a_val
        c_i = (-a_i + (sig / m if jensen else 0.0)) * sig
        d_i = 0.5 * s2_val * m if jensen else 0.0
        # E[Delta] = Var(x) (c - a sigma) + d; Var(x) is the Gaussian family's
        # vgf here, and 1 in the real (all-family) model
        pred = vgf[0] * (c_i - a_i * sig) + d_i
        print(f"  {label}  norm {z:.10f}   mean {mu*1e3:+8.4f} MeV   "
              f"closed form {pred*1e3:+8.4f} MeV   "
              f"(c {c_i*1e3:+.3f}, d {d_i*1e3:+.3f}, -a.sig {-a_i*sig*1e3:+.3f}, "
              f"Var(x) {vgf[0]:.2f})")
        assert abs(z - 1.0) < 2e-6, z
        assert abs(mu - pred) < 3e-5 * sig, (mu, pred)
    print("  PASS")


def test_delta_kernel_matches_residual():
    print("\n=== 3. delta kernel: the two forms agree (the J/psi gate) ===")
    # J/psi-like: sigma/m = 0.0085, |delta| out to 10 sigma
    rng = np.random.default_rng(7)
    mj = 3.0969
    n = 20000
    sig = 0.0085 * mj
    sigma = np.full(n, sig)
    x = rng.standard_normal(n)
    mobs = sig * x
    vgf = np.full(n, 0.60)
    a_res = (1.0 + vgf) * sigma / (mobs + mj)
    s2 = (sigma / (mobs + mj)) ** 2
    common = dict(vgf=vgf, a_res=a_res, jensen_s2=s2, mref=mj)
    res = build(sigma, mobs, corr_form="residual", jensen_mode="exact", **common)
    flu = build(sigma, mobs, corr_form="fluctuation", jensen_mode="exact", **common)
    off = build(sigma, mobs, mref=mj, vgf=vgf)

    # the SHAPE the fit sees: -log L as a function of the scale alpha
    out = {}
    for name, t in (("uncorrected", off), ("residual", res), ("fluctuation", flu)):
        al = np.linspace(-1.5, 1.5, 13)
        y = np.array([nll(t, alpha=float(a)) for a in al])
        # parabola minimum
        c2 = np.polyfit(al, y, 2)
        out[name] = -0.5 * c2[1] / c2[0]
        print(f"  {name:12s} alpha_min = {out[name]:+.5f} e-3")
    d_res = out["residual"] - out["uncorrected"]
    d_flu = out["fluctuation"] - out["uncorrected"]
    print(f"  correction size: residual {d_res:+.5f} e-3, "
          f"fluctuation {d_flu:+.5f} e-3, difference "
          f"{d_flu - d_res:+.5f} e-3")
    assert abs(d_res) > 0.05, d_res                     # the correction is real
    assert abs(d_flu - d_res) < 0.05 * abs(d_res), (d_flu, d_res)
    print("  PASS")


def test_bounded_at_z():
    print("\n=== 4. at the Z the fluctuation form stays bounded ===")
    rng = np.random.default_rng(11)
    n = 4000
    sigma = 0.9 + 0.5 * rng.random(n)
    # a +-30 GeV window, i.e. +-27 sigma -- the real Z selection
    mobs = np.sort(rng.uniform(-30.0, 30.0, n))
    m = mobs + MREF
    vgf = 0.5 + 0.4 * rng.random(n)
    a_res = (1.0 + vgf) * sigma / np.abs(m)
    s2 = (sigma / np.abs(m)) ** 2
    # a Breit-Wigner kernel and a softplus floor: with a DELTA kernel a
    # candidate 27 sigma from the pole has no density at all, and the NLL of
    # such a toy is meaningless in either form
    common = dict(vgf=vgf, a_res=a_res, jensen_s2=s2, floor="softplus",
                  kernel=unbinned.BreitWignerKernel(width_param="G"))
    res = build(sigma, mobs, corr_form="residual", jensen_mode="exact", **common)
    flu = build(sigma, mobs, corr_form="fluctuation", jensen_mode="exact", **common)
    for t in (res, flu):
        t.param_defaults[list(t.param_names).index("G")] = 2.4952
    v = values(res)
    dres = res._chunk_residual(v, 0).numpy()
    moved_res = np.abs(dres - res.mobs[: len(dres)].numpy())
    vf = values(flu)
    dflu = flu._chunk_residual(vf, 0).numpy()
    moved_flu = np.abs(dflu - flu.mobs[: len(dflu)].numpy())
    wr, wi = flu._fluct_w(vf, 0)
    wmax = float(np.max(np.abs(wr.numpy()) + np.abs(wi.numpy())))
    sre, _ = flu._chunk_resolution(vf, 0)
    wsup = float(np.max((np.abs(wr.numpy()) + np.abs(wi.numpy()))
                        * np.exp(sre.numpy())))
    print(f"  residual form moves the residual by median "
          f"{np.median(moved_res)*1e3:8.1f} MeV, max {moved_res.max():.3f} GeV")
    print(f"  fluctuation form   d_i        median "
          f"{np.median(moved_flu)*1e3:8.1f} MeV, max "
          f"{moved_flu.max()*1e3:.1f} MeV")
    print(f"  fluctuation CF factor |w| max over the whole tau grid {wmax:.4f}, "
          f"and max |w| e^(Re S) (what the quadrature actually sees) {wsup:.4f}")
    assert moved_res.max() > 1.0                       # GeV: the known defect
    assert moved_flu.max() < 0.05                      # GeV: bounded by d_i
    assert wsup < 0.5, wsup
    assert np.isfinite(nll(flu))
    print("  PASS")


def test_gradient():
    print("\n=== 5. analytic gradient vs central finite differences ===")
    rng = np.random.default_rng(3)
    n = 800
    sigma = 0.9 + 0.4 * rng.random(n)
    mobs = 3.0 * sigma * rng.standard_normal(n)
    vgf = 0.4 + 0.4 * rng.random(n)
    m = mobs + MREF
    t = build(sigma, mobs, vgf=vgf,
              a_res=(1.0 + vgf) * sigma / np.abs(m),
              jensen_s2=(sigma / np.abs(m)) ** 2,
              corr_form="fluctuation",
              kernel=unbinned.BreitWignerKernel(width_param="G", mass_param="dM"))
    x0 = np.array([1.0 if p == "k_hit" else (2.5 if p == "G" else 0.0)
                   for p in t.param_names])
    xt = tf.Variable(x0, dtype=DT)
    with tf.GradientTape() as tape:
        v = t.nll(xt)
    g = tape.gradient(v, xt).numpy()
    ok = True
    for i, p in enumerate(t.param_names):
        h = 1e-4 * max(abs(x0[i]), 1.0)
        xp, xm = x0.copy(), x0.copy()
        xp[i] += h
        xm[i] -= h
        fd = (float(t.nll(tf.constant(xp, DT))) -
              float(t.nll(tf.constant(xm, DT)))) / (2 * h)
        rel = abs(g[i] - fd) / max(abs(fd), 1e-6)
        print(f"  {p:8s} analytic {g[i]:+14.6f}  fd {fd:+14.6f}  rel {rel:.2e}")
        ok &= rel < 2e-5
    assert ok
    print("  PASS")


def test_cancellation():
    print("\n=== 6. c_i = -vgf_i sigma_i^2/m_i (the two quadratics cancel) ===")
    rng = np.random.default_rng(5)
    n = 300
    sigma = 0.9 + 0.4 * rng.random(n)
    mobs = 2.0 * sigma * rng.standard_normal(n)
    vgf = 0.3 + 0.6 * rng.random(n)
    m = mobs + MREF
    a_res = (1.0 + vgf) * sigma / np.abs(m)
    t = build(sigma, mobs, vgf=vgf, a_res=a_res,
              jensen_s2=(sigma / np.abs(m)) ** 2, corr_form="fluctuation")
    c = t._fl_g.numpy() * sigma
    pred = -vgf * sigma ** 2 / np.abs(m)
    print(f"  max |c - (-vgf sigma^2/m)| = {np.max(np.abs(c - pred)):.3e} GeV")
    print(f"  c median {np.median(c)*1e3:+.3f} MeV, "
          f"d median {np.median(t._fl_d.numpy())*1e3:+.3f} MeV, "
          f"-a.sigma median {np.median(-a_res*sigma)*1e3:+.3f} MeV")
    assert np.max(np.abs(c - pred)) < 1e-12
    print("  PASS")


if __name__ == "__main__":
    test_off_is_identical()
    test_density_moments()
    test_delta_kernel_matches_residual()
    test_bounded_at_z()
    test_gradient()
    test_cancellation()
    print("\nALL PASS")
