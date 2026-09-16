#!/usr/bin/env python3
"""Tests for ``MaterialCFTerm`` -- the mass likelihood whose resolution is
parameterised by the CVH fit's own PHYSICAL parameters (the parmtype-15
material-group amounts and the per-hit-class Gaussian shares) instead of the
four ad-hoc per-family knobs ``k_hit, k_ms, k_ioni, k_rad``.

Checks, in order:

1. **reduction** -- at ``k = eps = 0`` the term reproduces ``MassCFTerm``:
   bit-for-bit when every candidate touches exactly one group (identical
   summation order), and to float64 round-off in the realistic many-group
   case (the group sum is reassociated).
2. **legacy families** -- a ``MaterialCFTerm`` carrying only legacy families
   and no group/hit parameters is exactly ``MassCFTerm``.
3. **gradient** -- the analytic gradient of the NLL in ``k_g`` and ``eps_c``
   against central finite differences, in both ``exp`` and ``linear`` amount
   modes.
4. **units** -- ``group_units`` reproduces a rescaled parameter exactly (this
   is what carries the whitening of the quadratic card into the mass term).
5. **pruned baseline** -- a group folded into ``fix_re`` / ``fix_im`` behaves
   as a group whose parameter is frozen at 0.
6. **injection** -- multiply one group's DATA-side exponents by ``1 + d``
   (i.e. mimic ``d`` more material in that group) and check that the fit
   recovers it, in both amount modes.
7. **degeneracy** -- the Hessian of the mass NLL in the group parameters:
   rank, condition number, and the near-null directions, on a construction
   with two deliberately near-collinear groups.
8. **hdf5 round trip** -- write the term to a datacard, read it back, and
   check the NLL and the parameter list.

Run (in the rabbit / wmassdev singularity)::

    python tests/test_material_cf.py
"""

import os
import sys
import tempfile

import numpy as np
import tensorflow as tf

from rabbit import unbinned

RNG = np.random.default_rng(20260905)
TOL_EXACT = 1e-14


# ---------------------------------------------------------------------------
# synthetic inputs
# ---------------------------------------------------------------------------
def make_inputs(n=400, nt=48, ngroup=5, ncls=3, tmax=8.0, seed=1):
    """A small, well-conditioned toy: per-candidate per-group MS / ionization /
    radiative exponents with the right qualitative shape (Re S <= 0, growing
    like -tau^2 at small tau; Im S odd-ish), plus per-class hit variances."""
    rng = np.random.default_rng(seed)
    tgrid = np.linspace(0.0, tmax, nt)
    sigma = 0.02 + 0.01 * rng.random(n)

    # each candidate touches a random subset of the groups (2..ngroup)
    ptr = [0]
    gid = []
    for _ in range(n):
        k = rng.integers(min(2, ngroup), ngroup + 1)
        g = np.sort(rng.choice(ngroup, size=k, replace=False))
        gid.extend(g.tolist())
        ptr.append(len(gid))
    ptr = np.asarray(ptr, np.int64)
    gid = np.asarray(gid, np.int64)
    nnz = len(gid)

    # amplitude per (candidate, group) row, group-dependent so the groups are
    # distinguishable; MS dominates, ionization ~10x smaller, rad ~100x
    amp = 0.02 * (1.0 + 0.5 * gid.astype(float)) * (0.5 + rng.random(nnz))
    t2 = tgrid**2
    Sms = (-amp[:, None] * t2[None, :]).astype(np.float32)
    Sio_re = (-0.1 * amp[:, None] * t2[None, :] * np.exp(-0.05 * t2)[None, :]).astype(
        np.float32
    )
    Sio_im = (
        0.05 * amp[:, None] * (tgrid**3)[None, :] * np.exp(-0.05 * t2)[None, :]
    ).astype(np.float32)
    Srad_re = (-0.01 * amp[:, None] * t2[None, :]).astype(np.float32)
    Srad_im = (0.004 * amp[:, None] * (tgrid**3)[None, :]).astype(np.float32)

    # hit classes
    hptr = [0]
    hcls = []
    hv = []
    for _ in range(n):
        k = rng.integers(1, ncls + 1)
        c = np.sort(rng.choice(ncls, size=k, replace=False))
        hcls.extend(c.tolist())
        hv.extend((0.1 + 0.2 * rng.random(k)).tolist())
        hptr.append(len(hcls))
    hptr = np.asarray(hptr, np.int64)
    hcls = np.asarray(hcls, np.int64)
    hv = np.asarray(hv, np.float64)
    vg_other = 0.05 * rng.random(n)
    vgf = vg_other + np.add.reduceat(hv, hptr[:-1]) * (np.diff(hptr) > 0)

    # observed masses: a Gaussian draw of the right total width
    tot = vgf + 2.0 * np.add.reduceat(amp * 0.0 + amp, ptr[:-1]) * (np.diff(ptr) > 0)
    mobs = rng.normal(0.0, sigma * np.sqrt(np.maximum(tot, 0.2)))
    return dict(
        tgrid=tgrid,
        sigma=sigma,
        mobs=mobs,
        n=n,
        nt=nt,
        ngroup=ngroup,
        ncls=ncls,
        grp_ptr=ptr,
        grp_id=gid,
        fam={"ms": (Sms, None), "ioni": (Sio_re, Sio_im), "rad": (Srad_re, Srad_im)},
        hit_ptr=hptr,
        hit_cls=hcls,
        hit_v=hv,
        vg_other=vg_other,
        vgf=vgf,
    )


def group_names(ng):
    return [f"material_g{i}" for i in range(ng)]


def hit_names(nc):
    return [f"hitres_c{i}" for i in range(nc)]


def flatten(inp, w=None, hw=None):
    """Sum the per-group rows into flat ``(n, nt)`` per-family arrays -- what a
    ``MassCFTerm`` would carry -- with optional per-group weights ``w``."""
    ptr, gid, n, nt = inp["grp_ptr"], inp["grp_id"], inp["n"], inp["nt"]
    ww = np.ones(inp["ngroup"]) if w is None else np.asarray(w, float)
    out = {}
    for fam, (re, im) in inp["fam"].items():
        for comp, arr in (("re", re), ("im", im)):
            if arr is None:
                continue
            acc = np.zeros((n, nt))
            for i in range(n):
                for j in range(ptr[i], ptr[i + 1]):
                    acc[i] += ww[gid[j]] * arr[j]
            out[f"{fam}_{comp}"] = acc
    hh = np.ones(inp["ncls"]) if hw is None else np.asarray(hw, float)
    v = np.array(inp["vg_other"], float)
    for i in range(n):
        for j in range(inp["hit_ptr"][i], inp["hit_ptr"][i + 1]):
            v[i] += hh[inp["hit_cls"][j]] * inp["hit_v"][j]
    out["vgf"] = v
    return out


def build_mass(inp, flat, **kw):
    fams = [{"name": "hit", "param": "k_hit", "kind": "gauss"}]
    for fam in inp["fam"]:
        e = {"name": fam, "param": f"k_{fam}", "kind": "tab"}
        for comp in ("re", "im"):
            key = f"{fam}_{comp}"
            if key in flat:
                e[comp] = flat[key]
        fams.append(e)
    npar = 1 + len(inp["fam"])
    return unbinned.MassCFTerm(
        "mass",
        sigma=inp["sigma"],
        mobs=inp["mobs"],
        tgrid=inp["tgrid"],
        families=fams,
        vgf=flat["vgf"],
        floor="none",
        param_defaults=np.ones(npar),
        **kw,
    )


def build_material(
    inp,
    amount_mode="exp",
    hit_mode="linear",
    with_hits=True,
    group_units=None,
    hit_units=None,
    data_scale=None,
    prune=(),
    _fix_dtype=np.float64,
    **kw,
):
    """``data_scale`` multiplies the DATA-side exponents of the listed groups
    (the injection); ``prune`` folds the listed groups into the fixed
    baseline."""
    ptr, gid = inp["grp_ptr"], inp["grp_id"]
    keep = ~np.isin(gid, list(prune))
    gfam = []
    for fam, (re, im) in inp["fam"].items():
        e = {"name": fam}
        for comp, arr in (("re", re), ("im", im)):
            if arr is None:
                continue
            a = np.array(arr, dtype=np.float32)
            if data_scale is not None:
                for g, s in data_scale.items():
                    a[gid == g] *= np.float32(s)
            e[comp] = a[keep]
            if len(prune):
                # float64 baseline: this test is about the LOGIC of the fold,
                # not about float32 storage (a real card stores it float32
                # like every other exponent, which is a ~1e-11 NLL effect).
                fx = np.zeros((inp["n"], inp["nt"]), np.float64)
                for i in range(inp["n"]):
                    for j in range(ptr[i], ptr[i + 1]):
                        if gid[j] in prune:
                            fx[i] += a[j]
                e["fix_" + comp] = fx.astype(_fix_dtype)
        gfam.append(e)
    if len(prune):
        cnt = np.zeros(inp["n"], np.int64)
        for i in range(inp["n"]):
            cnt[i] = int(keep[ptr[i] : ptr[i + 1]].sum())
        ptr = np.concatenate([[0], np.cumsum(cnt)])
        gid = gid[keep]
    gp = group_names(inp["ngroup"])
    hp = hit_names(inp["ncls"]) if with_hits else []
    share = (
        (inp["hit_ptr"], inp["hit_cls"], inp["hit_v"], inp["vg_other"])
        if with_hits
        else None
    )
    if not with_hits:
        kw.setdefault("vgf", inp["vgf"])
    npar = len(gp) + len(hp)
    return unbinned.MaterialCFTerm(
        "mat",
        sigma=inp["sigma"],
        mobs=inp["mobs"],
        tgrid=inp["tgrid"],
        families=kw.pop("families", []),
        group_params=gp,
        group_families=gfam,
        grp_ptr=ptr,
        grp_id=gid,
        group_units=group_units,
        hit_params=hp,
        hit_share=share,
        hit_units=hit_units,
        amount_mode=amount_mode,
        hit_mode=hit_mode,
        floor="none",
        param_defaults=np.zeros(npar),
        **kw,
    )


# ---------------------------------------------------------------------------
# an EXACTLY GAUSSIAN toy, so the truth of a closure test is known in closed
# form: every group contributes ``Re S = -0.5 v_{i,g} tau^2`` and nothing else,
# so the density of candidate i is exactly N(0, sigma_i sqrt(V_i)) with
# ``V_i = v_other_i + sum_g A(k_g) v_{i,g}``.
# ---------------------------------------------------------------------------
def make_gauss_toy(
    n=4000, nt=40, ngroup=3, tmax=8.0, seed=101, scale=None, collinear=False
):
    rng = np.random.default_rng(seed)
    tgrid = np.linspace(0.0, tmax, nt)
    sigma = 0.02 + 0.01 * rng.random(n)

    ptr = [0]
    gid = []
    vrow = []
    for _ in range(n):
        k = rng.integers(2, ngroup + 1)
        g = np.sort(rng.choice(ngroup, size=k, replace=False))
        gid.extend(g.tolist())
        # amplitudes that VARY per candidate, which is what makes the groups
        # separable at all
        vrow.extend((0.05 + 0.45 * rng.random(k)).tolist())
        ptr.append(len(gid))
    ptr = np.asarray(ptr, np.int64)
    gid = np.asarray(gid, np.int64)
    vrow = np.asarray(vrow, np.float64)
    if collinear:
        # groups 0 and 1 ALWAYS appear together and ALWAYS with the same
        # amplitude, so only their sum is measurable and k_0 - k_1 is exactly
        # unconstrained
        ptr2, gid2, vrow2 = [0], [], []
        for i in range(n):
            sl = slice(ptr[i], ptr[i + 1])
            g = list(gid[sl])
            v = list(vrow[sl])
            a = v[g.index(0)] if 0 in g else (v[g.index(1)] if 1 in g else 0.2)
            rest = [(gg, vv) for gg, vv in zip(g, v) if gg not in (0, 1)]
            rows = [(0, a), (1, a)] + rest
            rows.sort()
            gid2.extend(x for x, _ in rows)
            vrow2.extend(y for _, y in rows)
            ptr2.append(len(gid2))
        ptr = np.asarray(ptr2, np.int64)
        gid = np.asarray(gid2, np.int64)
        vrow = np.asarray(vrow2, np.float64)
    v_other = 0.2 + 0.3 * rng.random(n)

    ss = np.ones(ngroup) if scale is None else np.asarray(scale, float)
    V = v_other.copy()
    np.add.at(V, np.repeat(np.arange(n), np.diff(ptr)), ss[gid] * vrow)
    # the standard-normal draw is taken BEFORE the scaling and depends only on
    # the seed, so an injected and an un-injected toy of the same seed share
    # their noise realisation exactly and their fitted difference is the
    # injection, not a fluctuation
    z0 = np.random.default_rng(seed + 999).standard_normal(n)
    mobs = z0 * sigma * np.sqrt(V)

    Sg = (-0.5 * vrow[:, None] * (tgrid**2)[None, :]).astype(np.float64)
    return dict(
        tgrid=tgrid,
        sigma=sigma,
        mobs=mobs,
        n=n,
        nt=nt,
        ngroup=ngroup,
        grp_ptr=ptr,
        grp_id=gid,
        Sg=Sg,
        v_other=v_other,
        V=V,
        scale=ss,
    )


def build_gauss_term(toy, amount_mode="exp", chunk=4096):
    # the un-scaled Gaussian remainder v_other rides in the FIXED baseline, so
    # that k = 0 is the truth of an un-injected toy
    fix = -0.5 * toy["v_other"][:, None] * (toy["tgrid"] ** 2)[None, :]
    return unbinned.MaterialCFTerm(
        "g",
        sigma=toy["sigma"],
        mobs=toy["mobs"],
        tgrid=toy["tgrid"],
        families=[],
        vgf=toy["v_other"],
        group_params=group_names(toy["ngroup"]),
        group_families=[{"name": "all", "re": toy["Sg"], "fix_re": fix}],
        grp_ptr=toy["grp_ptr"],
        grp_id=toy["grp_id"],
        amount_mode=amount_mode,
        floor="none",
        chunk=chunk,
        param_defaults=np.zeros(toy["ngroup"]),
    )


def _grad(term, z):
    xv = tf.Variable(z, dtype=tf.float64)
    with tf.GradientTape() as t1:
        v = term.nll(xv)
    return float(v.numpy()), np.asarray(t1.gradient(v, xv).numpy())


def _hess_fd(term, z, h=1e-4):
    """Hessian by central differences of the ANALYTIC gradient.

    The taped second derivative (``GradientTape.jacobian``) goes through
    ``pfor``, which retraces the segment-sum graph on every call and dominates
    the runtime of a test this size; with a handful of parameters, 2p gradient
    evaluations are cheaper and quite accurate enough for a rank /
    condition-number statement.
    """
    p = len(z)
    H = np.zeros((p, p))
    for i in range(p):
        up, dn = np.array(z, float), np.array(z, float)
        up[i] += h
        dn[i] -= h
        H[i] = 0.5 * (_grad(term, up)[1] - _grad(term, dn)[1]) / h
    return 0.5 * (H + H.T)


def fit_groups(term, start=None, tol=1e-8, maxit=12):
    """Damped-Newton minimisation over every group parameter, analytic
    gradient + finite-difference Hessian.  Returns (values, errors, H)."""
    x = np.zeros(len(term.param_names)) if start is None else np.array(start, float)
    f0, g = _grad(term, x)
    H = _hess_fd(term, x)
    for _ in range(maxit):
        if np.max(np.abs(g)) < tol:
            break
        step = -np.linalg.solve(H + 1e-10 * np.eye(len(x)) * max(np.trace(H), 1.0), g)
        lam = 1.0
        for _ in range(12):
            f1, g1 = _grad(term, x + lam * step)
            if np.isfinite(f1) and f1 <= f0 + 1e-12 * abs(f0):
                break
            lam *= 0.5
        x, f0, g = x + lam * step, f1, g1
        H = _hess_fd(term, x)
    cov = np.linalg.pinv(H)
    return x, np.sqrt(np.abs(np.diag(cov))), H


def nll_at(term, values):
    x = tf.constant([values[p] for p in term.param_names], tf.float64)
    return float(term.nll(x).numpy())


def grad_at(term, values):
    x = tf.Variable([values[p] for p in term.param_names], dtype=tf.float64)
    with tf.GradientTape() as tape:
        v = term.nll(x)
    return np.asarray(tape.gradient(v, x).numpy())


# ---------------------------------------------------------------------------
def test_reduction():
    print("\n=== 1. reduction to MassCFTerm at k = eps = 0 ===")
    # (a) many groups: reassociated sum, so float64 round-off
    inp = make_inputs()
    flat = flatten(inp)
    m = build_mass(inp, flat)
    t = build_material(inp)
    nm = nll_at(m, {p: 1.0 for p in m.param_names})
    nt_ = nll_at(t, {p: 0.0 for p in t.param_names})
    rel = abs(nm - nt_) / max(abs(nm), 1.0)
    print(f"  many-group   NLL mass {nm:.12f}  material {nt_:.12f}  rel {rel:.3e}")
    assert rel < 1e-13, rel

    # (b) one group per candidate: identical summation order -> bit-for-bit
    inp1 = make_inputs(n=200, ngroup=1, seed=7)
    inp1["grp_ptr"] = np.arange(inp1["n"] + 1, dtype=np.int64)
    inp1["grp_id"] = np.zeros(inp1["n"], np.int64)
    for fam in list(inp1["fam"]):
        re, im = inp1["fam"][fam]
        inp1["fam"][fam] = (re[: inp1["n"]], None if im is None else im[: inp1["n"]])
    flat1 = flatten(inp1)
    m1 = build_mass(inp1, flat1)
    t1 = build_material(inp1)
    a = nll_at(m1, {p: 1.0 for p in m1.param_names})
    b = nll_at(t1, {p: 0.0 for p in t1.param_names})
    print(
        f"  single-group NLL mass {a!r}\n               material {b!r}  "
        f"identical: {a == b}"
    )
    assert a == b, (a, b)
    print("  PASS")


def test_legacy_families():
    print("\n=== 2. legacy families only -> exactly MassCFTerm ===")
    inp = make_inputs(n=250, seed=3)
    flat = flatten(inp)
    fams = [{"name": "hit", "param": "k_hit", "kind": "gauss"}]
    for fam in inp["fam"]:
        e = {"name": fam, "param": f"k_{fam}", "kind": "tab"}
        for comp in ("re", "im"):
            if f"{fam}_{comp}" in flat:
                e[comp] = flat[f"{fam}_{comp}"]
        fams.append(e)
    npar = len(fams)
    common = dict(
        sigma=inp["sigma"],
        mobs=inp["mobs"],
        tgrid=inp["tgrid"],
        families=fams,
        vgf=flat["vgf"],
        floor="none",
        param_defaults=np.ones(npar),
    )
    m = unbinned.MassCFTerm("m", **common)
    t = unbinned.MaterialCFTerm("t", **common)
    v = {p: 0.7 + 0.1 * i for i, p in enumerate(m.param_names)}
    a, b = nll_at(m, v), nll_at(t, v)
    print(f"  params {m.param_names} -> {a!r} vs {b!r}  identical: {a == b}")
    assert list(m.param_names) == list(t.param_names)
    assert a == b, (a, b)
    print("  PASS")


def test_gradient():
    print("\n=== 3. analytic gradient vs central finite differences ===")
    inp = make_inputs(n=300, seed=11)
    for mode, hmode in (("exp", "linear"), ("linear", "exp")):
        t = build_material(inp, amount_mode=mode, hit_mode=hmode)
        base = {p: 0.03 * (i - 2) for i, p in enumerate(t.param_names)}
        g = grad_at(t, base)
        h = 1e-5
        worst = 0.0
        for i, p in enumerate(t.param_names):
            up, dn = dict(base), dict(base)
            up[p] += h
            dn[p] -= h
            fd = (nll_at(t, up) - nll_at(t, dn)) / (2 * h)
            rel = abs(fd - g[i]) / max(abs(fd), 1e-6)
            worst = max(worst, rel)
            print(
                f"  {mode:<6}/{hmode:<6} {p:<16} ana {g[i]:+12.6f}  "
                f"fd {fd:+12.6f}  rel {rel:.2e}"
            )
        assert worst < 3e-6, worst
    print("  PASS")


def test_units():
    print("\n=== 4. group_units carries the card whitening ===")
    inp = make_inputs(n=200, seed=5)
    u = np.array([0.02, 0.05, 0.1, 0.05, 0.02])[: inp["ngroup"]]
    t0 = build_material(inp)
    tu = build_material(inp, group_units=u)
    k = 0.4 * np.arange(inp["ngroup"]) - 0.6
    v0 = {p: 0.0 for p in t0.param_names}
    vu = {p: 0.0 for p in tu.param_names}
    for i, p in enumerate(group_names(inp["ngroup"])):
        v0[p] = k[i] * u[i]
        vu[p] = k[i]
    a, b = nll_at(t0, v0), nll_at(tu, vu)
    print(
        f"  NLL(k*u, units=1) {a!r}\n  NLL(k,   units=u) {b!r}  " f"identical: {a == b}"
    )
    assert a == b, (a, b)
    print("  PASS")


def test_pruned_baseline():
    print("\n=== 5. pruned group == group frozen at k = 0 ===")
    inp = make_inputs(n=200, seed=13)
    full = build_material(inp)
    pruned = build_material(inp, prune=(1, 3))
    k = {p: 0.0 for p in full.param_names}
    for i, p in enumerate(group_names(inp["ngroup"])):
        if i not in (1, 3):
            k[p] = 0.05 * (i + 1)
    a = nll_at(full, k)
    b = nll_at(pruned, k)
    rel = abs(a - b) / max(abs(a), 1.0)
    print(f"  full {a:.12f}  pruned {b:.12f}  rel {rel:.3e}")
    assert rel < 1e-13, rel
    # ... and with a float32 baseline (what a card actually stores) the
    # agreement is the float32 rounding of the folded sum
    pruned32 = build_material(inp, prune=(1, 3), _fix_dtype=np.float32)
    b32 = nll_at(pruned32, k)
    print(f"  float32 baseline {b32:.12f}  rel {abs(a-b32)/max(abs(a),1.):.3e}")
    assert abs(a - b32) / max(abs(a), 1.0) < 1e-8
    # ... and the pruned groups now have no gradient
    g = grad_at(pruned, k)
    for i in (1, 3):
        j = pruned.param_names.index(f"material_g{i}")
        print(f"  d(NLL)/d(material_g{i}) = {g[j]:.3e}")
        assert g[j] == 0.0
    print("  PASS")


def test_injection():
    print("\n=== 6. injection: 5 % more material in one group ===")
    d = 0.05
    for mode in ("linear", "exp"):
        truth = d if mode == "linear" else np.log1p(d)
        base = make_gauss_toy(n=4000, ngroup=3, seed=21)
        inj = make_gauss_toy(n=4000, ngroup=3, seed=21, scale=[1.0, 1.0 + d, 1.0])
        v0, e0, _ = fit_groups(build_gauss_term(base, amount_mode=mode))
        v1, e1, _ = fit_groups(build_gauss_term(inj, amount_mode=mode))
        print(
            f"  {mode:<6} truth shift = (0, {truth:+.5f}, 0);  "
            f"stat error on each parameter ~ {e0.mean():.4f}"
        )
        names = list(build_gauss_term(base).param_names)
        for i, nm in enumerate(names):
            tr = truth if i == 1 else 0.0
            print(
                f"      {nm:<14} base {v0[i]:+.5f}  inj {v1[i]:+.5f}  "
                f"shift {v1[i]-v0[i]:+.6f}  truth {tr:+.6f}  "
                f"(err {e0[i]:.5f})"
            )
        # The mode-independent statement: the AMOUNT factor A(k_1) must go up
        # by exactly the injected 1 + d.  The two toys share their noise, so
        # what is left is the estimator's own non-linearity -- a few % of the
        # injection, i.e. a few % of ONE statistical sigma.
        A = np.exp if mode == "exp" else (lambda z: 1.0 + z)
        r = A(v1[1]) / A(v0[1])
        print(
            f"      A(k_1) ratio {r:.6f}  injected {1+d:.6f}  "
            f"({100*(r/(1+d)-1):+.2f} %)"
        )
        assert abs(r / (1.0 + d) - 1.0) < 0.01, r
        assert abs((v1[1] - v0[1]) - truth) < 0.15 * abs(truth), (v0, v1)
        for i in (0, 2):
            assert abs(v1[i] - v0[i]) < 0.03 * abs(truth), (v0, v1)
    print("  PASS")


def test_degeneracy():
    print("\n=== 7. degeneracy structure of the group Hessian ===")
    for tag, collin in (("independent", False), ("groups 0,1 collinear", True)):
        toy = make_gauss_toy(n=4000, ngroup=4, seed=31, collinear=collin)
        t = build_gauss_term(toy)
        H = fit_groups(t)[2]
        w, V = np.linalg.eigh(H)
        rank = int((w > 1e-7 * w.max()).sum())
        print(f"  {tag}:")
        print("    eigenvalues " + "  ".join(f"{x:10.4g}" for x in w))
        print(
            f"    rank(1e-7) {rank}/{len(w)}   cond "
            f"{w.max()/max(w.min(), 1e-300):.3e}"
        )
        print(
            "    softest     "
            + "  ".join(
                f"{n.split('_')[-1]} {v:+.3f}" for n, v in zip(t.param_names, V[:, 0])
            )
        )
        if collin:
            # only the 0 <-> 1 antisymmetric combination is unmeasured
            v0 = V[:, 0]
            assert abs(v0[0]) > 0.4 and abs(v0[1]) > 0.4, v0
            assert np.sign(v0[0]) != np.sign(v0[1]), v0
            assert w[0] < 1e-7 * w.max(), w
        else:
            assert rank == len(w), (rank, w)
    print("  PASS")


def test_hdf5_roundtrip():
    print("\n=== 8. datacard round trip ===")
    import h5py

    inp = make_inputs(n=150, nt=32, ngroup=3, ncls=2, seed=41)
    t = build_material(inp)
    v = {p: 0.02 * (i + 1) for i, p in enumerate(t.param_names)}
    before = nll_at(t, v)

    datasets = {
        "sigma": inp["sigma"],
        "mobs": inp["mobs"],
        "tgrid": inp["tgrid"],
        "grp_ptr": t.g_ptr,
        "grp_id": np.asarray(inp["grp_id"]),
        "hit_ptr": inp["hit_ptr"],
        "hit_cls": inp["hit_cls"],
        "hit_v": inp["hit_v"],
        "vg_other": inp["vg_other"],
    }
    for fam, (re, im) in inp["fam"].items():
        datasets[f"Sg_re_{fam}"] = re
        if im is not None:
            datasets[f"Sg_im_{fam}"] = im
    raw = [
        dict(
            name="mat",
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
            back = unbinned.read_unbinned_terms_from_h5(f["unbinned_terms"])
    assert len(back) == 1
    t2 = back[0]
    assert list(t2.param_names) == list(t.param_names), t2.param_names
    after = nll_at(t2, v)
    print(
        f"  NLL before {before!r}\n      after  {after!r}  "
        f"identical: {before == after}"
    )
    assert before == after, (before, after)
    print("  PASS")


def test_self_consistent_sigma():
    """The MASSCFTERM_SPEC correction: sigma is the fit's own error, so it is a
    function of the residual it is measuring.

    Gate G1 of the spec -- ``a_i = 0`` reproduces the term bit-identically --
    plus the two things only a unit test can check: that the DYNAMIC path
    agrees with the static one at a = 0 (so the shortcut is not hiding a bug in
    the interpolator), and that the gradient in alpha is right when s depends
    on alpha through the prefactor, the tau grid AND the kernel CF.
    """
    print("\n=== 9. self-consistent sigma (MASSCFTERM_SPEC) ===")
    inp = make_inputs(n=400, seed=51)
    flat = flatten(inp)
    # keep the residuals inside the model's core: make_inputs draws `mobs` from
    # a width of its own invention, and a candidate parked at 3.5 sigma of THAT
    # sits where the truncated inverse-Fourier integral is small enough that a
    # few-% change of s can push it through zero.  That is a property of the
    # toy, not of the correction, and it would hide the thing being tested.
    inp["mobs"] = (
        0.3 * inp["sigma"] * np.random.default_rng(77).standard_normal(inp["n"])
    )
    # a mass term with a scale parameter, so `delta` actually moves
    tab = np.linspace(0.0, 400.0, 4096)
    pk = np.exp(-0.5 * (0.004 * tab) ** 2)
    common = dict(
        sigma=inp["sigma"],
        mobs=inp["mobs"],
        tgrid=inp["tgrid"],
        families=[{"name": "hit", "param": "k_hit", "kind": "gauss"}],
        # softplus, the production default: with floor="none" a
        # candidate whose truncated inverse-Fourier integral dips
        # negative in the tail gives log(<=0) = NaN, and moving s by
        # 1 % is enough to flip one
        vgf=flat["vgf"],
        floor="softplus",
        m_ref=3.0969,
        scale_param="alpha",
        phik=(tab, pk, np.zeros_like(pk)),
        param_defaults=np.zeros(2),
    )
    base = unbinned.MassCFTerm("b", **common)
    zero = unbinned.MassCFTerm("z", a_res=np.zeros(inp["n"]), **common)
    off = unbinned.MassCFTerm(
        "o", a_res=0.011 * np.ones(inp["n"]), self_consistent_sigma=False, **common
    )
    on = unbinned.MassCFTerm("c", a_res=0.011 * np.ones(inp["n"]), **common)
    v = {"alpha": 0.3, "k_hit": 1.0}
    nb, nz, no, nc = (nll_at(t, v) for t in (base, zero, off, on))
    print(f"  a=None      {nb!r}")
    print(f"  a=0         {nz!r}   identical: {nb == nz}")
    print(f"  a!=0, off   {no!r}   identical: {nb == no}")
    print(f"  a!=0, on    {nc!r}   d = {nc-nb:+.6f}")
    assert nb == nz, (nb, nz)  # G1
    assert nb == no, (nb, no)  # the switch
    assert nc != nb  # the correction does something

    # the dynamic path at a = 0 must agree with the static one: this is the
    # in-graph kernel-CF interpolation against np.interp
    forced = unbinned.MassCFTerm("f", a_res=1e-300 * np.ones(inp["n"]), **common)
    nf = nll_at(forced, v)
    print(f"  a=1e-300 (dynamic path) {nf!r}  rel {abs(nf-nb)/abs(nb):.3e}")
    assert abs(nf - nb) / abs(nb) < 1e-12, (nf, nb)

    # gradient in alpha with s depending on alpha
    g = grad_at(on, v)
    h = 1e-6
    up, dn = dict(v), dict(v)
    up["alpha"] += h
    dn["alpha"] -= h
    fd = (nll_at(on, up) - nll_at(on, dn)) / (2 * h)
    i = list(on.param_names).index("alpha")
    print(
        f"  d(NLL)/d(alpha)  ana {g[i]:+.6f}  fd {fd:+.6f}  "
        f"rel {abs(fd-g[i])/max(abs(fd),1e-9):.2e}"
    )
    assert abs(fd - g[i]) / max(abs(fd), 1e-9) < 1e-5

    # THE SIGN.  s_i = sigma_i - a_i delta_i, so a candidate whose observed
    # mass sits ABOVE the prediction must be given a SMALLER unconditional
    # resolution (its exported sigma was inflated by its own upward
    # fluctuation), and one below a larger.  Checked directly, because the
    # direction of the resulting alpha shift is what gates G2/G3 of the spec
    # measure on a toy generated WITH the defect -- this toy has no defect to
    # correct, so a fit here would test the toy, not the term.
    d = on._chunk_residual(
        on._values(tf.constant([v[p] for p in on.param_names], tf.float64)), 0
    ).numpy()
    ss = on._chunk_sigma(
        on._values(tf.constant([v[p] for p in on.param_names], tf.float64)),
        0,
        tf.constant(d),
    ).numpy()
    sig = np.asarray(on.sigma.numpy())[: len(d)]
    a = 0.011
    print(
        f"  s vs sigma: max |s/sigma - 1| = {np.abs(ss/sig - 1).max():.4e}, "
        f"corr(sign) = {np.sign(np.corrcoef(d, ss - sig)[0,1]):+.0f}"
    )
    assert np.allclose(ss, np.maximum(sig - a * d, on.sigma_floor * sig)), "s formula"
    assert np.corrcoef(d, ss - sig)[0, 1] < -0.99, "s must shrink where delta > 0"
    print("  PASS")


if __name__ == "__main__":
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ.get("OMP_NUM_THREADS", "8"))
    )
    only = sys.argv[1:] or None
    tests = [
        test_reduction,
        test_legacy_families,
        test_gradient,
        test_units,
        test_pruned_baseline,
        test_injection,
        test_degeneracy,
        test_hdf5_roundtrip,
        test_self_consistent_sigma,
    ]
    for fn in tests:
        if only and not any(o in fn.__name__ for o in only):
            continue
        fn()
    print("\nALL TESTS PASSED")
