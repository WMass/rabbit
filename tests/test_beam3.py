#!/usr/bin/env python3
"""Tests for the LUMINOUS-REGION (``beam3``) block of ``MaterialCFTerm``.

The beam-line constraint is a Gaussian noise block whose covariance is a
physical 3x3 object, not a scale on a number.  ``beam3`` floats the covariance
itself: the two transverse variance scales, the x-y correlation, and the two
tilts.  A functional's variance share from the block is exactly
``sum_ab covBS_ab Q_ab`` with ``Q_ab = w_a w_b`` the outer product of its own
influence weights on the three beam rows, so the per-candidate input is six
numbers, the record, and the block's nominal share.

Checks, in order:

1. **nominal** -- at ``p = 0`` the block reproduces its nominal share BIT for
   BIT (the anchored form makes every difference identically zero), and the
   term equals one built with that share folded into ``vg_other``.
2. **share algebra** -- the in-graph share against an independent numpy
   evaluation of the CMS beam-spot-fitter covariance, over a grid of
   parameter values including large ones.
3. **gradient** -- analytic gradients of the NLL in all five parameters
   against central finite differences.
4. **hessian** -- the analytic Hessian against finite differences of the
   analytic gradient, full 5x5 block.
5. **width convention** -- ``d(share)/d(eps_x)`` of this block against the
   maker's exported ``*vbsx`` convention (``covBS -> D covBS D``): they differ
   only through ``dC_xz/dk_x`` and the difference is ``O(sigma_x^2/sigma_z^2)``
   of the term, which is what licenses reading the two against each other.
6. **units and freezing** -- ``beam3_units`` reproduces a rescaled parameter
   exactly; an empty role name freezes that role at the record.
7. **mean and covariance on ONE parameter** -- a tilt that appears in both
   ``beam3_params`` and ``jac_params`` is ONE parameter, and its gradient is
   the sum of the two contributions.
8. **truncation normalisation** -- with ``norm_window`` on, ``Z`` carries the
   beam parameters (its derivative is non-zero and matches finite differences).
9. **hdf5 round trip** -- write, read back, same parameter list and NLL.

Run (in the rabbit / wmassdev singularity)::

    python tests/test_beam3.py
"""

import os
import sys
import tempfile

import numpy as np
import tensorflow as tf

from rabbit import unbinned

RNG = np.random.default_rng(20260917)
TOL_EXACT = 1e-14

# a realistic Run-2 luminous region: cm
REC = np.array([10.83e-4, 10.39e-4, 3.6239, -5.97e-6, 4.72e-6])


# ---------------------------------------------------------------------------
# numpy reference
# ---------------------------------------------------------------------------
PACK = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
MULT = np.array([1.0, 2.0, 2.0, 1.0, 2.0, 1.0])


def cov_np(sx, sy, sz, dxdz, dydz, rho=0.0, kx=1.0, ky=1.0):
    """The CMS beam-spot-fitter covariance, packed (xx, xy, xz, yy, yz, zz)."""
    vx, vy, vz = kx * sx**2, ky * sy**2, sz**2
    cxy = rho * np.sqrt(vx) * np.sqrt(vy)
    cxz = dxdz * (vz - vx) - dydz * cxy
    cyz = dydz * (vz - vy) - dxdz * cxy
    return np.stack([vx, cxy, cxz, vy, cyz, np.broadcast_to(vz, np.shape(vx))], axis=-1)


def share_np(q, ref, v0, ex=0.0, ey=0.0, eta=0.0, tx=0.0, ty=0.0):
    c = cov_np(
        ref[:, 0],
        ref[:, 1],
        ref[:, 2],
        ref[:, 3] + tx,
        ref[:, 4] + ty,
        rho=np.tanh(eta),
        kx=1.0 + ex,
        ky=1.0 + ey,
    )
    c0 = cov_np(ref[:, 0], ref[:, 1], ref[:, 2], ref[:, 3], ref[:, 4])
    return v0 + ((c - c0) * q * MULT).sum(axis=-1)


# ---------------------------------------------------------------------------
# synthetic inputs
# ---------------------------------------------------------------------------
def make_inputs(n=300, nt=32, ncls=3, tmax=8.0, seed=3, wz_frac=1e-5):
    """A toy whose Gaussian share is split into hit classes and a beam block.

    ``w`` is drawn with a realistic hierarchy: the two transverse weights are
    O(1) in units where the beam block carries ~1/4 of the functional's
    variance, and the z weight is ~1e-5 of them (the z beam row is weightless
    against a 100 um vertex error), which is what makes the tilt's effect on
    the COVARIANCE second order.
    """
    rng = np.random.default_rng(seed)
    tgrid = np.linspace(0.0, tmax, nt)
    sigma = 0.02 + 0.01 * rng.random(n)

    # hit classes: 0.6 of the variance, as on the real beam functionals
    hptr = [0]
    hcls, hv = [], []
    for _ in range(n):
        k = rng.integers(1, ncls + 1)
        c = np.sort(rng.choice(ncls, size=k, replace=False))
        hcls.extend(c.tolist())
        hv.extend((0.6 / k * (0.8 + 0.4 * rng.random(k))).tolist())
        hptr.append(len(hcls))
    hptr = np.asarray(hptr, np.int64)
    hcls = np.asarray(hcls, np.int64)
    hv = np.asarray(hv, np.float64)

    ref = np.tile(REC, (n, 1))
    # the transverse weights, scaled so that the block carries ~0.25 of the
    # variance; `w` has units 1/cm for a dimensionless functional
    phi = 2 * np.pi * rng.random(n)
    amp = (0.9 + 0.2 * rng.random(n)) * np.sqrt(0.25) / REC[0]
    w = np.stack(
        [amp * np.cos(phi), amp * np.sin(phi), wz_frac * amp * (2 * rng.random(n) - 1)],
        axis=1,
    )
    q = np.stack([w[:, a] * w[:, b] for a, b in PACK], axis=-1)
    v0 = (cov_np(*[ref[:, i] for i in range(5)]) * q * MULT).sum(axis=-1)

    vg_other = 0.15 * (0.8 + 0.4 * rng.random(n))
    vgf = vg_other + np.add.reduceat(hv, hptr[:-1]) + v0
    mobs = rng.normal(0.0, sigma * np.sqrt(vgf))
    return dict(
        tgrid=tgrid,
        sigma=sigma,
        mobs=mobs,
        n=n,
        nt=nt,
        ncls=ncls,
        hit_ptr=hptr,
        hit_cls=hcls,
        hit_v=hv,
        vg_other=vg_other,
        vgf=vgf,
        q=q,
        ref=ref,
        v0=v0,
        w=w,
    )


B3 = ["beamwidth_x", "beamwidth_y", "beamcorr_xy", "beamtilt_x", "beamtilt_y"]


def build(inp, beam3=True, units=None, params=None, **kw):
    hp = [f"hitres_c{i}" for i in range(inp["ncls"])]
    share = (inp["hit_ptr"], inp["hit_cls"], inp["hit_v"], inp["vg_other"])
    b3 = None
    bp = ()
    if beam3:
        b3 = {"q": inp["q"], "ref": inp["ref"], "v0": inp["v0"]}
        bp = list(B3) if params is None else list(params)
    else:
        # the beam block folded into the share NOTHING scales
        share = (
            inp["hit_ptr"],
            inp["hit_cls"],
            inp["hit_v"],
            inp["vg_other"] + inp["v0"],
        )
    t = unbinned.MaterialCFTerm(
        "t",
        sigma=inp["sigma"],
        mobs=inp["mobs"],
        tgrid=inp["tgrid"],
        families=[],
        hit_params=hp,
        hit_share=share,
        hit_mode="linear",
        beam3_params=bp,
        beam3_units=units,
        beam3=b3,
        floor="none",
        param_defaults=np.zeros(
            len(hp)
            + len([p for p in bp if p])
            + len([p for p in kw.get("jac_params", ()) if p not in bp])
        ),
        **kw,
    )
    return t


def values(t, **over):
    v = {p: 0.0 for p in t.param_names}
    v.update(over)
    return v


def nll_at(t, v):
    x = tf.constant([v[p] for p in t.param_names], tf.float64)
    return float(t.nll(x).numpy())


def grad_at(t, v):
    x = tf.Variable([v[p] for p in t.param_names], dtype=tf.float64)
    with tf.GradientTape() as tp:
        y = t.nll(x)
    return np.asarray(tp.gradient(y, x).numpy())


def hess_at(t, v):
    x = tf.Variable([v[p] for p in t.param_names], dtype=tf.float64)
    with tf.GradientTape() as t2:
        with tf.GradientTape() as t1:
            y = t.nll(x)
        g = t1.gradient(y, x)
    return np.asarray(t2.jacobian(g, x).numpy())


# ---------------------------------------------------------------------------
def test_nominal():
    print("\n=== 1. nominal: the block reproduces its own nominal share ===")
    inp = make_inputs(seed=11)
    t = build(inp)
    lo, hi = t._chunks[0]
    v = values(t)
    sh = t._beam3_share(v, t._b3_q[lo:hi], t._b3_ref[lo:hi], t._b3_v0[lo:hi])
    d = np.abs(np.asarray(sh.numpy()) - inp["v0"][lo:hi])
    print(f"  max |share(0) - v0| = {d.max():.3e}  (bit-identical: {d.max() == 0.0})")
    assert d.max() == 0.0, "the anchored form must be exact at the nominal point"

    # the whole term against one with the block folded into vg_other
    t0 = build(inp, beam3=False)
    v0_ = values(t0)
    n1, n0 = nll_at(t, v), nll_at(t0, v0_)
    g1 = grad_at(t, v)[: inp["ncls"]]
    g0 = grad_at(t0, v0_)[: inp["ncls"]]
    rel = abs(n1 - n0) / max(abs(n0), 1e-300)
    print(f"  NLL beam3 {n1:.12f} vs folded {n0:.12f}  rel {rel:.3e}")
    print(f"  max |d NLL/d eps_hit difference| = {np.abs(g1 - g0).max():.3e}")
    assert rel < 1e-13
    assert np.abs(g1 - g0).max() < 1e-9
    print("  PASS")


def test_share_algebra():
    print("\n=== 2. share algebra vs an independent numpy evaluation ===")
    inp = make_inputs(seed=12)
    t = build(inp)
    lo, hi = t._chunks[0]
    worst = 0.0
    for ex, ey, eta, tx, ty in [
        (0.1, -0.2, 0.3, 1e-5, -2e-5),
        (-0.5, 0.7, -1.2, 5e-4, 3e-4),
        (2.0, 3.0, 0.05, -1e-3, 1e-3),
    ]:
        v = values(
            t,
            beamwidth_x=ex,
            beamwidth_y=ey,
            beamcorr_xy=eta,
            beamtilt_x=tx,
            beamtilt_y=ty,
        )
        sh = np.asarray(
            t._beam3_share(v, t._b3_q[lo:hi], t._b3_ref[lo:hi], t._b3_v0[lo:hi]).numpy()
        )
        ref = share_np(
            inp["q"][lo:hi], inp["ref"][lo:hi], inp["v0"][lo:hi], ex, ey, eta, tx, ty
        )
        r = np.abs(sh - ref).max() / np.abs(ref).max()
        worst = max(worst, r)
        print(
            f"  eps=({ex:+.2f},{ey:+.2f}) eta={eta:+.2f} "
            f"tilt=({tx:+.0e},{ty:+.0e})  rel {r:.3e}"
        )
    assert worst < 1e-12
    print("  PASS")


def test_gradient():
    print("\n=== 3. gradient vs central finite differences ===")
    inp = make_inputs(seed=13)
    t = build(inp)
    base = values(
        t,
        beamwidth_x=0.08,
        beamwidth_y=-0.05,
        beamcorr_xy=0.12,
        beamtilt_x=2e-5,
        beamtilt_y=-1e-5,
    )
    g = grad_at(t, base)
    names = list(t.param_names)
    worst = 0.0
    for p in B3:
        i = names.index(p)
        h = 1e-4 if not p.startswith("beamtilt") else 1e-7
        vp, vm = dict(base), dict(base)
        vp[p] += h
        vm[p] -= h
        fd = (nll_at(t, vp) - nll_at(t, vm)) / (2 * h)
        r = abs(g[i] - fd) / max(abs(fd), 1e-300)
        worst = max(worst, r)
        print(f"  {p:14s} analytic {g[i]:+.9e}  FD {fd:+.9e}  rel {r:.2e}")
    assert worst < 5e-6
    print("  PASS")


def test_hessian():
    print("\n=== 4. Hessian vs finite differences of the analytic gradient ===")
    inp = make_inputs(n=200, seed=14)
    t = build(inp)
    base = values(
        t,
        beamwidth_x=0.05,
        beamwidth_y=0.03,
        beamcorr_xy=-0.08,
        beamtilt_x=1e-5,
        beamtilt_y=1e-5,
    )
    H = hess_at(t, base)
    names = list(t.param_names)
    idx = [names.index(p) for p in B3]
    worst = 0.0
    for p in B3:
        i = names.index(p)
        h = 1e-4 if not p.startswith("beamtilt") else 1e-7
        vp, vm = dict(base), dict(base)
        vp[p] += h
        vm[p] -= h
        fd = (grad_at(t, vp) - grad_at(t, vm)) / (2 * h)
        rows = []
        for j in idx:
            r = abs(H[i, j] - fd[j]) / max(abs(fd[j]), abs(H[i, j]), 1e-300)
            rows.append(r)
            worst = max(worst, r)
        print(f"  row {p:14s} max rel {max(rows):.2e}")
    assert worst < 1e-4
    print("  PASS")


def test_width_convention():
    print("\n=== 5. d(share)/d(eps_x) vs the maker's D covBS D convention ===")
    inp = make_inputs(seed=15)
    t = build(inp)
    lo, hi = t._chunks[0]
    q, ref, v0 = inp["q"], inp["ref"], inp["v0"]
    h = 1e-5
    dp = share_np(q, ref, v0, ex=+h)
    dm = share_np(q, ref, v0, ex=-h)
    d_formula = (dp - dm) / (2 * h)
    # the maker's convention: covBS -> D covBS D with D = diag(rt kx, rt ky, 1)
    c0 = cov_np(*[ref[:, i] for i in range(5)])
    w = inp["w"]
    d_maker = (
        w[:, 0] ** 2 * c0[:, 0]
        + w[:, 0] * w[:, 1] * c0[:, 1]
        + w[:, 0] * w[:, 2] * c0[:, 2]
    )
    rel = np.abs(d_formula - d_maker) / np.abs(d_maker)
    # the analytic size of the difference: the two disagree only in dC_xz/dk_x,
    # -dxdz sigma_x^2 against +dxdz sigma_z^2/2, weighted by 2 w_x w_z
    pred = np.abs(
        2 * w[:, 0] * w[:, 2] * ref[:, 3] * (0.5 * ref[:, 2] ** 2 + ref[:, 0] ** 2)
    ) / np.abs(d_maker)
    print(
        f"  median rel difference {np.median(rel):.3e}  "
        f"(predicted {np.median(pred):.3e})"
    )
    print(f"  in units of the SHARE itself: {np.median(rel*np.abs(d_maker)/v0):.3e}")
    assert np.median(rel) < 1e-3
    assert np.abs(np.median(rel) / np.median(pred) - 1.0) < 0.2
    # and the in-graph derivative IS the formula one
    base = values(t)
    i = list(t.param_names).index("beamwidth_x")
    g = grad_at(t, base)[i]
    vp, vm = dict(base), dict(base)
    vp["beamwidth_x"] += 1e-4
    vm["beamwidth_x"] -= 1e-4
    fd = (nll_at(t, vp) - nll_at(t, vm)) / 2e-4
    print(f"  graph d NLL/d eps_x {g:+.9e}  FD {fd:+.9e}")
    assert abs(g - fd) / abs(fd) < 5e-6
    print("  PASS")


def test_units_and_freeze():
    print("\n=== 6. beam3_units, and freezing a role ===")
    inp = make_inputs(seed=16)
    t1 = build(inp)
    u = np.array([1.0, 1.0, 1.0, 1e-5, 1e-5])
    t2 = build(inp, units=u)
    v1 = values(t1, beamtilt_x=3e-5, beamtilt_y=-2e-5)
    v2 = values(t2, beamtilt_x=3.0, beamtilt_y=-2.0)
    n1, n2 = nll_at(t1, v1), nll_at(t2, v2)
    print(f"  NLL unit=1 {n1:.12f}  unit=1e-5 {n2:.12f}  diff {abs(n1-n2):.3e}")
    assert abs(n1 - n2) < 1e-12 * max(abs(n1), 1.0)

    tf_ = build(inp, params=["beamwidth_x", "", "", "beamtilt_x", ""])
    print(f"  frozen-role parameter list: {list(tf_.param_names)[-2:]}")
    assert "beamwidth_y" not in tf_.param_names
    assert "beamcorr_xy" not in tf_.param_names
    assert "beamtilt_y" not in tf_.param_names
    assert nll_at(tf_, values(tf_)) == nll_at(t1, values(t1))
    print("  PASS")


def test_one_parameter_two_roles():
    print("\n=== 7. a tilt in BOTH the mean and the covariance is ONE parameter ===")
    inp = make_inputs(seed=17)
    n = inp["n"]
    # D_card = -d(functional)/d(tilt): the mean response through the beam row
    zlever = 2.9 * (2 * RNG.random(n) - 1)
    dmean = -inp["w"][:, 0] * zlever
    idx = np.stack([np.arange(n), np.zeros(n, np.int64)], 1)
    t = build(
        inp,
        jac=(idx, dmean, (n, 1)),
        jac_params=["beamtilt_x"],
    )
    print(f"  parameters: {list(t.param_names)}")
    assert list(t.param_names).count("beamtilt_x") == 1
    base = values(t, beamtilt_x=4e-5)
    i = list(t.param_names).index("beamtilt_x")
    g = grad_at(t, base)[i]
    h = 1e-7
    vp, vm = dict(base), dict(base)
    vp["beamtilt_x"] += h
    vm["beamtilt_x"] -= h
    fd = (nll_at(t, vp) - nll_at(t, vm)) / (2 * h)
    print(f"  analytic {g:+.9e}  FD {fd:+.9e}  rel {abs(g-fd)/abs(fd):.2e}")
    assert abs(g - fd) / abs(fd) < 1e-5
    # mean-only and covariance-only terms must SUM to it -- evaluated at the
    # nominal point, where the two sub-models and the full one agree on
    # everything else (away from it they do not: the tilt moves the variance
    # too, so the mean-only model's dNLL/dmean is taken at a different width,
    # which is exactly the cross term this parameterisation exists to carry)
    tm = build(
        inp,
        params=["beamwidth_x", "beamwidth_y", "beamcorr_xy", "", ""],
        jac=(idx, dmean, (n, 1)),
        jac_params=["beamtilt_x"],
    )
    tc = build(inp)
    g0 = grad_at(t, values(t))[list(t.param_names).index("beamtilt_x")]
    gm = grad_at(tm, values(tm))[list(tm.param_names).index("beamtilt_x")]
    gc = grad_at(tc, values(tc))[list(tc.param_names).index("beamtilt_x")]
    print(
        f"  at nominal: mean-only {gm:+.9e} + cov-only {gc:+.9e} "
        f"= {gm+gc:+.9e}  vs {g0:+.9e}"
    )
    assert abs(gm + gc - g0) / max(abs(g0), 1e-300) < 1e-12
    print(f"  the covariance share of the tilt gradient: {gc/g0:.3e}")
    print("  PASS")


def test_norm_window():
    print("\n=== 8. the truncation normalisation carries the beam parameters ===")
    inp = make_inputs(n=200, seed=18)
    K = 4
    cls = np.clip(
        (inp["sigma"] - inp["sigma"].min()) / (np.ptp(inp["sigma"]) + 1e-12) * K,
        0,
        K - 1e-9,
    ).astype(np.int64)
    sig_c = np.array([inp["sigma"][cls == c].mean() for c in range(K)])
    hv_c = np.zeros((K, inp["ncls"]))
    seg = np.repeat(np.arange(inp["n"]), np.diff(inp["hit_ptr"]))
    for c in range(K):
        m = cls[seg] == c
        for j in range(inp["ncls"]):
            sel = m & (inp["hit_cls"] == j)
            hv_c[c, j] = inp["hit_v"][sel].sum() / max((cls == c).sum(), 1)
    norm = {
        "sigma": sig_c,
        "class": cls,
        "vgf": np.array([inp["vgf"][cls == c].mean() for c in range(K)]),
        "vg_other": np.array([inp["vg_other"][cls == c].mean() for c in range(K)]),
        "hit_v": hv_c,
        "group_families": [],
        "beam3_q": np.stack([inp["q"][cls == c].mean(0) for c in range(K)]),
        "beam3_ref": np.stack([inp["ref"][cls == c].mean(0) for c in range(K)]),
        "beam3_v0": np.array([inp["v0"][cls == c].mean() for c in range(K)]),
    }
    t = build(
        inp,
        norm=norm,
        norm_window=(-5.0, 5.0),
        norm_window_sigma=True,
        norm_tpoints=512,
    )
    base = values(t, beamwidth_x=0.05)
    g = grad_at(t, base)[list(t.param_names).index("beamwidth_x")]
    h = 1e-4
    vp, vm = dict(base), dict(base)
    vp["beamwidth_x"] += h
    vm["beamwidth_x"] -= h
    fd = (nll_at(t, vp) - nll_at(t, vm)) / (2 * h)
    print(f"  with Z: analytic {g:+.9e}  FD {fd:+.9e}  rel {abs(g-fd)/abs(fd):.2e}")
    assert abs(g - fd) / abs(fd) < 1e-5
    # and Z really does move: drop the beam3 rows from `norm` and the answer
    # changes
    norm2 = dict(norm)
    for k in ("beam3_q", "beam3_ref", "beam3_v0"):
        norm2.pop(k)
    try:
        build(
            inp,
            norm=norm2,
            norm_window=(-5.0, 5.0),
            norm_window_sigma=True,
            norm_tpoints=512,
        )
    except ValueError as e:
        print(f"  refused without the class-level block: {e}")
    else:
        raise AssertionError("a beam3 term with a window must demand norm beam3")
    print("  PASS")


def test_zero_record_row():
    print("\n=== 9b. a ZERO record row has a finite Hessian ===")
    # A truncation normalisation's unused class rows can carry a zero record
    # (an empty class takes whatever the builder puts there).  `rho sqrt(C_xx
    # C_yy)` would have an infinite derivative at `sigma = 0` and `rho = 0`
    # turns that into `0 * inf = NaN` -- finite VALUE, NaN Hessian, which is
    # exactly how it presents in a fit: the minimiser converges and the
    # covariance step dies.
    inp = make_inputs(n=80, nt=16, seed=21)
    inp["ref"] = inp["ref"].copy()
    inp["ref"][:10] = 0.0
    inp["v0"] = inp["v0"].copy()
    inp["v0"][:10] = 0.0
    t = build(inp)
    v = values(t, beamwidth_x=0.05, beamwidth_y=-0.03, beamcorr_xy=0.1)
    n = nll_at(t, v)
    g = grad_at(t, v)
    H = hess_at(t, v)
    print(
        f"  NLL {n:.6f} finite {np.isfinite(n)}; "
        f"grad finite {np.isfinite(g).all()}; Hessian finite "
        f"{np.isfinite(H).all()}"
    )
    assert np.isfinite(n) and np.isfinite(g).all() and np.isfinite(H).all()
    print("  PASS")


def test_rechunk_with_jac():
    print("\n=== 9c. re-chunking a term that carries a sparse D ===")
    inp = make_inputs(n=210, nt=16, seed=22)
    n = inp["n"]
    lev = 2.9 * (2 * RNG.random(n) - 1)
    dmean = -inp["w"][:, 0] * lev
    idx = np.stack([np.arange(n), np.zeros(n, np.int64)], 1)
    t = build(inp, jac=(idx, dmean, (n, 1)), jac_params=["beamtilt_x"])
    v = values(t, beamtilt_x=3e-5, beamwidth_x=0.04)
    n0, g0 = nll_at(t, v), grad_at(t, v)
    for c in (64, 7, 210, 1000):
        t.rechunk(c)
        n1, g1 = nll_at(t, v), grad_at(t, v)
        d = abs(n1 - n0) / max(abs(n0), 1e-300)
        dg = np.abs(g1 - g0).max()
        print(
            f"  chunk {c:5d} ({t.nchunk} chunk(s)): NLL rel {d:.2e}, "
            f"max |dgrad| {dg:.2e}"
        )
        assert d < 1e-12 and dg < 1e-6
    print("  PASS")


def test_hdf5_roundtrip():
    print("\n=== 9. datacard round trip ===")
    import h5py

    inp = make_inputs(n=120, nt=24, seed=19)
    t = build(inp, units=np.array([1.0, 1.0, 1.0, 1e-5, 1e-5]))
    v = {p: 0.01 * (i + 1) for i, p in enumerate(t.param_names)}
    before = nll_at(t, v)
    datasets = {
        "sigma": inp["sigma"],
        "mobs": inp["mobs"],
        "tgrid": inp["tgrid"],
        "hit_ptr": inp["hit_ptr"],
        "hit_cls": inp["hit_cls"],
        "hit_v": inp["hit_v"],
        "vg_other": inp["vg_other"],
        "beam3_q": inp["q"],
        "beam3_ref": inp["ref"],
        "beam3_v0": inp["v0"],
    }
    raw = [
        dict(
            name="t",
            config=t.config(),
            params=list(t.param_names),
            param_defaults=t.param_defaults,
            param_prior_sigmas=t.param_prior_sigmas,
            param_prior_means=t.param_prior_means,
            param_is_poi=t.param_is_poi,
            datasets=datasets,
        )
    ]
    with tempfile.TemporaryDirectory() as d:
        fn = os.path.join(d, "card.hdf5")
        with h5py.File(fn, "w") as f:
            unbinned.write_unbinned_terms_group(f, raw)
        with h5py.File(fn, "r") as f:
            back = unbinned.read_unbinned_terms_from_h5(f["unbinned_terms"])[0]
    print(f"  params {list(back.param_names)}")
    assert list(back.param_names) == list(t.param_names)
    after = nll_at(back, v)
    print(f"  NLL before {before:.12f} after {after:.12f}")
    assert abs(before - after) < 1e-12 * max(abs(before), 1.0)
    print("  PASS")


if __name__ == "__main__":
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ.get("OMP_NUM_THREADS", "8"))
    )
    only = sys.argv[1:] or None
    tests = [
        test_nominal,
        test_share_algebra,
        test_gradient,
        test_hessian,
        test_width_convention,
        test_units_and_freeze,
        test_one_parameter_two_roles,
        test_norm_window,
        test_zero_record_row,
        test_rechunk_with_jac,
        test_hdf5_roundtrip,
    ]
    for fn in tests:
        if only and not any(o in fn.__name__ for o in only):
            continue
        fn()
    print("\nALL TESTS PASSED")
