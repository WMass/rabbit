#!/usr/bin/env python3
"""The EXACT second-order (Jensen) correction of `MassCFTerm`.

Reference: `calibration_studies/resolution/oddmoment/MASSCFTERM_SPEC.md` sec. 4b
and the validated offline implementation `oddmoment/masslik_np.py` (`_delta`,
`--jensen-mode exact`). The five checks are the ones that can fail silently:

1. OFF is bit-identical to a term that was never given `jensen_s2` -- the
   correction must be additive.
2. The map is INVERTED correctly: recomputing `r = u + u^2 + s^2/2` from the
   returned residual must give back `delta/m`.
3. The log-Jacobian is `-log(1 + 2u)` and it is actually applied -- switching
   it off changes the NLL by `sum_i log(1+2u_i)`.
4. The analytic gradient of the corrected NLL matches finite differences.
5. `shift` and `exact` differ, and `shift` moves the residual by exactly
   `1.5 s^2 m`, which is the form the spec measures to OVER-correct by 27-44 %.
"""

import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rabbit import unbinned  # noqa: E402

DT = tf.float64
MREF = 91.1876


def build(n=733, nt=64, seed=20260906, **kw):
    rng = np.random.default_rng(seed)
    sigma = 0.6 + 1.4 * rng.random(n)
    mobs = sigma * rng.standard_normal(n)
    tgrid = np.linspace(0.0, 7.8926, nt)
    vgf = 0.15 + 0.5 * rng.random(n)
    fam = [
        {"name": "hit", "param": "k_hit", "kind": "gauss"},
        {
            "name": "ms",
            "param": "k_ms",
            "kind": "tab",
            "re": (-0.02 * rng.random((n, 1)) * tgrid[None, :] ** 2),
        },
    ]
    t = unbinned.MassCFTerm(
        "j",
        sigma=sigma,
        mobs=mobs,
        tgrid=tgrid,
        families=fam,
        vgf=vgf,
        kernel=unbinned.BreitWignerKernel(width_param="G", mass_param="dM"),
        m_ref=MREF,
        scale_param="alpha",
        chunk=257,
        **kw,
    )
    unbinned.declare_params(
        t,
        {
            "k_hit": (1.0, np.nan, 1.0, 0),
            "k_ms": (1.0, np.nan, 1.0, 0),
            "G": (2.5, np.nan, 2.5, 0),
            "dM": (0.0, np.nan, 0.0, 0),
            "alpha": (0.0, np.nan, 0.0, 1),
        },
    )
    return t, sigma, mobs


def build_one(sig, deltas, s2v, **kw):
    """One candidate's map, scanned over `deltas` -- the monotonicity test."""
    n = len(deltas)
    tgrid = np.linspace(0.0, 7.8926, 64)
    fam = [{"name": "hit", "param": "k_hit", "kind": "gauss"}]
    t = unbinned.MassCFTerm(
        "one",
        sigma=np.full(n, sig),
        mobs=deltas,
        tgrid=tgrid,
        families=fam,
        vgf=np.full(n, 0.3),
        kernel=unbinned.BreitWignerKernel(width_param="G", mass_param="dM"),
        m_ref=MREF,
        scale_param="alpha",
        chunk=n,
        jensen_s2=np.full(n, s2v),
        jensen_mode="exact",
        **kw,
    )
    unbinned.declare_params(
        t,
        {
            "k_hit": (1.0, np.nan, 1.0, 0),
            "G": (2.5, np.nan, 2.5, 0),
            "dM": (0.0, np.nan, 0.0, 0),
            "alpha": (0.0, np.nan, 0.0, 1),
        },
    )
    return t, None, None


def nll(t, x):
    return float(t.nll(tf.constant(x, DT)).numpy())


def main():
    ok = True
    n = 733
    _, sigma, mobs = build(n)
    s2 = (sigma / np.abs(mobs + MREF)) ** 2

    # ---- 1. OFF is additive ------------------------------------------------
    a, _, _ = build(n)
    b, _, _ = build(n, jensen_s2=s2, jensen_mode="off")
    x = np.asarray(a.param_defaults, np.float64)
    va, vb = nll(a, x), nll(b, x)
    p = va == vb
    ok &= p
    print(
        f"1. OFF additive                 {'PASS' if p else 'FAIL'}   "
        f"{va!r} vs {vb!r}"
    )

    # ---- 2. the map is inverted --------------------------------------------
    e, _, _ = build(n, jensen_s2=s2, jensen_mode="exact")
    xe = np.asarray(e.param_defaults, np.float64)
    vals = e._values(tf.constant(xe, DT))
    dl = e._chunk_residual(vals, 0).numpy()
    lo, hi = e._chunks[0]
    m = mobs[lo:hi] + MREF
    u = dl / m
    r_back = u + u**2 + 0.5 * s2[lo:hi]
    r_lin = mobs[lo:hi] / m  # alpha = 0, dM = 0
    d = np.max(np.abs(r_back - r_lin))
    p = d < 1e-13
    ok &= p
    print(
        f"2. u + u^2 + s^2/2 == delta/m   {'PASS' if p else 'FAIL'}   "
        f"max |d| {d:.3e}"
    )

    # ---- 3. the Jacobian is applied ---------------------------------------
    lj = e._chunk_logjac(vals, 0).numpy()
    d = np.max(np.abs(lj + np.log1p(2.0 * u)))
    p = d < 1e-14
    ok &= p
    print(
        f"3. logjac == -log(1+2u)         {'PASS' if p else 'FAIL'}   "
        f"max |d| {d:.3e}"
    )

    # ---- 4. gradient vs finite differences --------------------------------
    v = tf.constant(xe, DT)
    with tf.GradientTape() as tp:
        tp.watch(v)
        f = e.nll(v)
    g = tp.gradient(f, v).numpy()
    worst = 0.0
    for i, nm in enumerate(e.param_names):
        h = 1e-5 * max(abs(xe[i]), 1.0)
        xp, xm = xe.copy(), xe.copy()
        xp[i] += h
        xm[i] -= h
        fd = (nll(e, xp) - nll(e, xm)) / (2 * h)
        worst = max(worst, abs(fd - g[i]) / max(abs(fd), 1e-8))
    p = worst < 3e-6
    ok &= p
    print(
        f"4. gradient vs FD               {'PASS' if p else 'FAIL'}   "
        f"worst rel {worst:.3e}"
    )

    # ---- 5. shift is the mean, and differs from exact ----------------------
    sh, _, _ = build(n, jensen_s2=s2, jensen_mode="shift")
    ms = sh._chunk_mean_shift(sh._values(tf.constant(xe, DT)), 0).numpy()
    d = np.max(np.abs(ms - 1.5 * s2[lo:hi] * m))
    p1 = d < 1e-13
    dn = abs(nll(sh, xe) - nll(e, xe))
    p2 = dn > 1e-6
    ok &= p1 and p2
    print(
        f"5. shift == 1.5 s^2 m, != exact {'PASS' if (p1 and p2) else 'FAIL'}   "
        f"max |d| {d:.3e}, |dNLL| {dn:.4f}"
    )

    # ---- 6. corr_clip: identity inside, saturation outside -----------------
    cl, _, _ = build(n, jensen_s2=s2, jensen_mode="exact", corr_clip=3.0)
    xc = np.asarray(cl.param_defaults, np.float64)
    vc = cl._values(tf.constant(xc, DT))
    dlc = cl._chunk_residual(vc, 0).numpy()
    ljc = cl._chunk_logjac(vc, 0).numpy()
    lo, hi = cl._chunks[0]
    inside = np.abs(mobs[lo:hi]) <= 3.0 * sigma[lo:hi]
    # inside the clip it must be the unclipped map, to the last bit
    p1 = np.array_equal(dlc[inside], dl[inside])
    # outside, the correction must SATURATE: the difference from the raw
    # residual is frozen at its value on the boundary, and the measure is
    # unchanged (unit slope), so the log-Jacobian is exactly zero
    corr = dlc - mobs[lo:hi]
    lim = 3.0 * sigma[lo:hi]
    ru = np.sign(mobs[lo:hi]) * lim / m
    du = 0.5 * (np.sqrt(np.maximum(1 + 4 * (ru - 0.5 * s2[lo:hi]), 0.1)) - 1)
    p2 = np.allclose(
        corr[~inside],
        (du * m - np.sign(mobs[lo:hi]) * lim)[~inside],
        rtol=0,
        atol=1e-11,
    )
    p3 = np.all(ljc[~inside] == 0.0) and np.array_equal(ljc[inside], lj[inside])
    # and it must still be monotone in the observable -- which is a
    # PER-CANDIDATE statement (each candidate has its own sigma and hence its
    # own clip), so it is checked by scanning one candidate's own delta
    grid = np.linspace(-60.0, 60.0, 4001)
    j = int(np.argmin(sigma))  # the tightest clip in the sample
    one, _, _ = build_one(sigma[j], grid, s2[j], corr_clip=3.0)
    xo = np.asarray(one.param_defaults, np.float64)
    go = one._chunk_residual(one._values(tf.constant(xo, DT)), 0).numpy()
    p4 = bool(np.all(np.diff(go) > 0))
    ok &= p1 and p2 and p3 and p4
    print(
        f"6. corr_clip in/out/monotone    "
        f"{'PASS' if (p1 and p2 and p3 and p4) else 'FAIL'}   "
        f"inside {int(inside.sum())}/{len(inside)}, "
        f"identity {p1}, saturates {p2}, logjac {p3}, monotone {p4}"
    )

    # ---- 7. corr_clip = 0 is the unclipped term, bit for bit ---------------
    z0, _, _ = build(n, jensen_s2=s2, jensen_mode="exact", corr_clip=0.0)
    p = nll(z0, xe) == nll(e, xe)
    ok &= p
    print(
        f"7. corr_clip = 0 is unclipped   {'PASS' if p else 'FAIL'}   "
        f"{nll(z0, xe)!r}"
    )

    print("ALL TESTS PASSED" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
