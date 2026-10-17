"""Traditional and global impacts at an active constraint face (--constraints).

At an active face the impacts are taken with the faces held: every covariance
that enters them is the face-held one (the full one, the stat-block one with
and without the bin-by-bin stat, and each group's conditional one), from the
Hessian of the Lagrangian on the directions along the faces. Checked against
exact references that do not use that algebra:

(a) a bound on one parameter: the impacts of the fit with that parameter
    FROZEN on the bound (rabbit's ordinary path);
(b) a linear face mixing two parameters: the impacts of the fit in which one
    of them is ELIMINATED through the face (a smaller ParamModel);
(c) a nonlinear face: each group's conditional variance against the face-held
    covariance of the fit with that group frozen (rabbit's own freeze path),
    the global impacts against finite-difference refits and the data-stat
    remainder against diag(C H_data C);
(d) no active face: the impacts code path and its numbers are unchanged.

All comparisons are made at the SAME point (the reference fit's minimum), so
they test the algebra to round-off, not the minimizers.
"""

import os
import tempfile

import hist
import numpy as np
import pytest
import tensorflow as tf

from rabbit import fitter, inputdata, tensorwriter
from rabbit.constraints import postfit as cpostfit
from rabbit.constraints.constraints import Bounds, Constraints
from rabbit.param_models.param_model import ParamModel

from .test_sparse_fit import make_options

# ---------------------------------------------------------------------------
# a small model with a POI, two linear model nuisances (POUs, prior-free, so in
# the stat block), constrained card nuisances in overlapping groups, one
# unconstrained card nuisance and bin-by-bin stat
# ---------------------------------------------------------------------------


def make_face_tensor(outdir):
    rng = np.random.default_rng(7)
    ax = hist.axis.Regular(24, -3, 3, name="x")
    c = ax.centers
    nominal = {
        "sig": 400.0 * np.exp(-0.5 * c**2 / 0.6**2),
        "bkgA": 150.0 + 0.0 * c,
        "bkgB": 60.0 * (c + 3.5),
        "bkgC": 250.0 * np.exp(-0.6 * (c + 3.0)),
    }
    hists = {}
    for name, vals in nominal.items():
        h = hist.Hist(ax, storage=hist.storage.Weight())
        h.view().value = vals
        h.view().variance = 0.04 * vals
        hists[name] = h

    def tilted(h, slope):
        up, dn = h.copy(), h.copy()
        up.view().value = h.values() * (1 + slope * c)
        dn.view().value = h.values() * (1 - slope * c)
        return [up, dn]

    expect = (
        1.1 * nominal["sig"]
        + 0.9 * nominal["bkgA"]
        + nominal["bkgB"]
        + 1.2 * nominal["bkgC"]
    )
    h_data = hist.Hist(ax, storage=hist.storage.Double())
    h_data.view()[...] = rng.poisson(expect).astype(float)

    writer = tensorwriter.TensorWriter()
    writer.add_channel(h_data.axes, "ch0")
    writer.add_data(h_data, "ch0")
    writer.add_process(hists["sig"], "sig", "ch0", signal=True)
    for name in ("bkgA", "bkgB", "bkgC"):
        writer.add_process(hists[name], name, "ch0")
    writer.add_norm_systematic("normC", "bkgC", "ch0", 1.08, groups=["norms"])
    writer.add_norm_systematic("normB", "bkgB", "ch0", 1.05, groups=["norms", "bkg"])
    writer.add_systematic(
        tilted(hists["bkgA"], 0.03), "shapeA", "bkgA", "ch0", groups=["shapes", "bkg"]
    )
    writer.add_systematic(
        tilted(hists["sig"], 0.02), "shapeS", "sig", "ch0", groups=["shapes"]
    )
    writer.add_norm_systematic(
        "freeB", "bkgB", "ch0", 1.10, constrained=False, groups=["free"]
    )
    writer.write(outfolder=outdir, outfilename="face_tensor")
    return os.path.join(outdir, "face_tensor.hdf5")


class FacePOUs(ParamModel):
    """sig (POI, linear) scales the signal, a scales bkgA, b scales bkgC.

    With ``elim_lb`` set, ``a`` is eliminated through the face a + 2 b = lb:
    the parameters are (sig, b) and a = lb - 2 b. Groups map accordingly: with
    the face held, fixing a fixes b, so every one of the full model's groups
    {a}, {b}, {a, b} is the eliminated model's {b}.
    """

    def __init__(self, indata, elim_lb=None, disk=None, **kwargs):
        self.indata = indata
        self.elim_lb = elim_lb
        self.disk = disk
        self.npoi = 1
        if disk is not None:
            # (a, b) on the boundary of Disk(*disk): the face eliminated
            # exactly, a = a0 + r sa cos(phi), b = b0 + r sb sin(phi)
            self.npou = 1
            self.params = np.array(["sig", "phi"])
            self.param_impact_groups = {"phi": ["phi"]}
        elif elim_lb is None:
            self.npou = 2
            self.params = np.array(["sig", "a", "b"])
            self.param_impact_groups = {"a": ["a"], "b": ["b"], "ab": ["a", "b"]}
        else:
            self.npou = 1
            self.params = np.array(["sig", "b"])
            self.param_impact_groups = {"b": ["b"]}
        self.allowNegativeParam = True
        self.is_linear = False
        self.set_param_default(None, allowNegativeParam=True)
        procs = [p.decode() if isinstance(p, bytes) else str(p) for p in indata.procs]
        self._order = [procs.index(p) for p in ("sig", "bkgA", "bkgB", "bkgC")]

    def compute(self, param, full=False):
        sig = param[0]
        if self.disk is not None:
            a0, b0, sa, sb, r2 = self.disk
            r = np.sqrt(r2)
            a = a0 + r * sa * tf.cos(param[1])
            b = b0 + r * sb * tf.sin(param[1])
        elif self.elim_lb is None:
            a, b = param[1], param[2]
        else:
            b = param[1]
            a = tf.constant(self.elim_lb, dtype=param.dtype) - 2.0 * b
        vals = tf.stack([sig, a, tf.ones([], dtype=param.dtype), b])
        inv = np.argsort(self._order)
        return tf.reshape(tf.gather(vals, inv), [1, -1])


class Linear(Constraints):
    """sum_j coef_j x[name_j] >= lb."""

    def __init__(self, coeffs, lb):
        self.dtype = tf.float64
        self.coeffs = dict(coeffs)
        self.labels = ["linear"]
        self.lower = np.array([lb], dtype=np.float64)
        self.upper = np.array([np.inf])

    def set_layout(self, parms):
        self._idx = self.resolve_indices(parms, list(self.coeffs), "Linear")

    def parameter_indices(self):
        return list(self._idx.values())

    def values(self, x):
        return tf.reshape(
            tf.add_n([a * x[self._idx[n]] for n, a in self.coeffs.items()]), [1]
        )


class Disk(Constraints):
    """((a - a0)/sa)^2 + ((b - b0)/sb)^2 <= r2: a strongly curved face.

    With the centre a few sigma from the free minimum and a radius of 2 sigma,
    the constraint curvature is comparable to the likelihood's, so the
    Lagrangian term changes the curvature along the face at O(1).
    """

    def __init__(self, a0, b0, sa, sb, r2=4.0):
        self.dtype = tf.float64
        self.spec = (a0, b0, sa, sb)
        self.labels = ["disk"]
        self.lower = np.array([-np.inf])
        self.upper = np.array([r2], dtype=np.float64)

    def set_layout(self, parms):
        self._idx = self.resolve_indices(parms, ["a", "b"], "Disk")

    def parameter_indices(self):
        return list(self._idx.values())

    def values(self, x):
        a0, b0, sa, sb = self.spec
        return tf.reshape(
            ((x[self._idx["a"]] - a0) / sa) ** 2 + ((x[self._idx["b"]] - b0) / sb) ** 2,
            [1],
        )


def _fitter(fn, cons=(), elim_lb=None, disk=None, method="trust-krylov", **kw):
    indata_obj = inputdata.FitInputData(fn)
    pm = FacePOUs(indata_obj, elim_lb=elim_lb, disk=disk)
    opts = make_options(
        minimizerMethod=method, prefitUnconstrainedNuisanceUncertainty=1.0, **kw
    )
    if method == "trust-constr":
        opts.minimizerGtol = 1e-10
        opts.minimizerXtol = 1e-12
        opts.minimizerBarrierTol = 1e-10
        opts.minimizerMaxiter = 1000
    f = fitter.Fitter(indata_obj, pm, opts)
    f.set_nobs(indata_obj.data_obs)
    f.constraints = list(cons)
    f.arm_regularizers()
    return f


def _bounds(*tokens):
    from types import SimpleNamespace

    return Bounds(SimpleNamespace(dtype=tf.float64), *tokens)


def _names(f):
    return [p.decode() if isinstance(p, bytes) else str(p) for p in f.parms]


def _at(f, x):
    f.x.assign(np.asarray(x, dtype=np.float64))
    if f.bbstat.enabled:
        f._profile_beta()


def _postfit_plain(f):
    """rabbit_fit's unconstrained post-fit: cov, then every impact."""
    _, g, h = f.loss_val_grad_hess()
    _, cov = f.edmval_cov(g, h)
    f.cov.assign(cov)
    return _impacts(f, h)


def _postfit_held(f):
    """rabbit_fit's constrained post-fit: face-held cov, then every impact."""
    _, g, h = f.loss_val_grad_hess()
    cpf = f.constrained_postfit(g, h)
    f.cov.assign(tf.constant(cpf["cov_held"], dtype=f.indata.dtype))
    out = _impacts(f, h)
    out["cpf"] = cpf
    return out


def _impacts(f, h):
    imp, grp = f.impacts_parms(h)
    gimp, ggrp = f.global_impacts_parms()
    gaimp, gagrp = f.gaussian_global_impacts_parms()
    return {
        "cov": f.cov.numpy(),
        "impacts": np.asarray(imp),
        "grouped": np.asarray(grp),
        "global": np.asarray(gimp),
        "global_grouped": np.asarray(ggrp),
        "gaussian": np.asarray(gaimp),
        "gaussian_grouped": np.asarray(gagrp),
    }


def _newton_polish(f, iters=6):
    """Finish an unconstrained minimisation with exact Newton steps, to |grad|
    ~1e-13: the minimizer's own stop (|grad| 1e-8..1e-10, platform dependent)
    is too coarse for a finite-difference reference."""
    for _ in range(iters):
        _, g, h = f.loss_val_grad_hess()
        f.x.assign(f.x.numpy() - np.linalg.solve(np.asarray(h), np.asarray(g)))


def _free_minimum(fn):
    f = _fitter(fn)
    f.minimize()
    return f


# ---------------------------------------------------------------------------
# (a) a bound on one parameter == that parameter frozen on the bound
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,shift",
    [
        ("freeB", +0.3),  # unconstrained card nuisance: inside the stat block
        ("normC", -0.4),  # constrained nuisance: outside it
        ("a", +0.05),  # model nuisance (POU): inside the stat block
    ],
)
def test_bound_face_equals_frozen_parameter(name, shift):
    with tempfile.TemporaryDirectory() as tmp:
        fn = make_face_tensor(tmp)
        free = _free_minimum(fn)
        i = _names(free).index(name)
        bound = float(free.x.numpy()[i] + shift)

        ref = _fitter(fn, freezeParameters=[name])
        x0 = ref.x.numpy().copy()
        x0[i] = bound
        ref.x.assign(x0)
        ref.minimize()
        xref = ref.x.numpy().copy()
        _at(ref, xref)
        want = _postfit_plain(ref)

        op = ">=" if shift > 0 else "<="
        tc = _fitter(fn, cons=[_bounds(f"{name}{op}{bound!r}")])
        _at(tc, xref)
        got = _postfit_held(tc)
        assert len(got["cpf"]["active"]) == 1
        assert tc.postfit_faces is not None

        # the held parameter: exactly zero variance and covariance, as frozen
        assert np.all(got["cov"][i] == 0) and np.all(got["cov"][:, i] == 0)
        other = [k for k in range(len(xref)) if k != i]
        np.testing.assert_allclose(
            got["cov"][np.ix_(other, other)],
            want["cov"][np.ix_(other, other)],
            rtol=1e-9,
            atol=1e-14,
        )
        if name == "a":
            # the frozen fit drops the emptied param group {a}; held, it is 0
            for key in ("grouped", "global_grouped", "gaussian_grouped"):
                ncol = got[key].shape[1]
                assert np.all(got[key][:, ncol - 3] == 0), key
                got[key] = np.delete(got[key], ncol - 3, axis=1)
        for key in (
            "impacts",
            "grouped",
            "global",
            "global_grouped",
            "gaussian",
            "gaussian_grouped",
        ):
            assert np.all(np.isfinite(got[key])), key
            np.testing.assert_allclose(
                got[key], want[key], rtol=1e-9, atol=1e-12, err_msg=key
            )


# ---------------------------------------------------------------------------
# (b) a linear face mixing two parameters == the eliminated model
# ---------------------------------------------------------------------------


def test_linear_face_equals_eliminated_model():
    with tempfile.TemporaryDirectory() as tmp:
        fn = make_face_tensor(tmp)
        free = _free_minimum(fn)
        nf = _names(free)
        ia, ib = nf.index("a"), nf.index("b")
        xf = free.x.numpy()
        lb = float(xf[ia] + 2 * xf[ib] + 0.08)  # violated by the free fit

        ref = _fitter(fn, elim_lb=lb)
        ref.minimize()
        xe = ref.x.numpy().copy()
        _at(ref, xe)
        want = _postfit_plain(ref)

        # the same point in the full model: (sig, a = lb - 2 b, b, theta)
        xfull = np.concatenate([[xe[0], lb - 2 * xe[1], xe[1]], xe[2:]])
        tc = _fitter(fn, cons=[Linear({"a": 1.0, "b": 2.0}, lb)])
        _at(tc, xfull)
        got = _postfit_held(tc)
        (face,) = got["cpf"]["active"]
        assert face["multiplier"] > 0

        # full-model columns [sig, a, b, theta...] vs eliminated [sig, b, theta...]
        keep = [0, 2] + list(range(3, len(xfull)))
        np.testing.assert_allclose(
            got["cov"][np.ix_(keep, keep)], want["cov"], rtol=1e-8, atol=1e-13
        )
        np.testing.assert_allclose(
            got["impacts"][:, keep], want["impacts"], rtol=1e-8, atol=1e-12
        )
        # a = lb - 2 b: a's impact mirrors b's
        np.testing.assert_allclose(
            got["impacts"][:, 1], -want["impacts"][:, 1], rtol=1e-8
        )
        # grouped: [syst groups | stat | binByBinStat | param groups]; the
        # full model's param groups {a}, {b}, {a, b} are all the eliminated {b}
        g, w = got["grouped"], want["grouped"]
        nfix = w.shape[1] - 1
        np.testing.assert_allclose(g[:, :nfix], w[:, :nfix], rtol=1e-8, atol=1e-12)
        for k in range(3):
            np.testing.assert_allclose(g[:, nfix + k], w[:, nfix], rtol=1e-7)
        assert np.all(np.isfinite(g))
        # global impacts: the per-source ones are those of the card
        # nuisances, identical; the grouped data-stat remainder is finite
        for key in ("global", "gaussian"):
            np.testing.assert_allclose(
                got[key][:, keep], want[key], rtol=1e-8, atol=1e-12, err_msg=key
            )
        for key in ("global_grouped", "gaussian_grouped"):
            gg, ww = got[key], want[key]
            np.testing.assert_allclose(
                gg[:, :nfix], ww[:, :nfix], rtol=1e-8, atol=1e-12, err_msg=key
            )
            assert np.all(np.isfinite(gg)), key


# ---------------------------------------------------------------------------
# (c) a nonlinear face: the Lagrangian construction
# ---------------------------------------------------------------------------


def _curved_face_fit(fn, **kw):
    free = _free_minimum(fn)
    nf = _names(free)
    ia, ib = nf.index("a"), nf.index("b")
    xf = free.x.numpy()
    _, g, h = free.loss_val_grad_hess()
    _, cov = free.edmval_cov(g, h)
    sa, sb = np.sqrt(np.diag(np.asarray(cov))[[ia, ib]])
    # centre 3 sigma away along the diagonal, radius 2 sigma: the face normal
    # mixes a and b, and the free minimum is outside the disk
    spec = (
        float(xf[ia] + 3 / np.sqrt(2) * sa),
        float(xf[ib] + 3 / np.sqrt(2) * sb),
        float(sa),
        float(sb),
    )
    tc = _fitter(fn, cons=[Disk(*spec)], method="trust-constr", **kw)
    xs = xf.copy()
    # start strictly inside, but not at the centre: there the row's gradient
    # vanishes and build_scipy_constraints drops it
    xs[ia], xs[ib] = spec[0] - 0.5 * sa, spec[1] - 0.5 * sb
    tc.x.assign(xs)
    tc.minimize()
    _at(tc, tc.x.numpy().copy())
    tc.start = xs
    return tc, spec


def _held_variance_with_frozen(fn, lb, x, frozen, use_hess_no_bbb=False):
    """POI variance of the fit at x with ``frozen`` frozen and the face held,
    from rabbit's own freeze path + face_held_covariance (not the impacts
    code)."""
    f = _fitter(fn, cons=[Disk(*lb)], method="trust-constr", freezeParameters=frozen)
    _at(f, x)
    _, g, h = f.loss_val_grad_hess()
    if use_hess_no_bbb:
        _, _, h = f.loss_val_grad_hess(profile=False)
    cpf = f.constrained_postfit(g, h)
    return cpf["cov_held"][0, 0]


def test_curved_face_grouped_impacts_equal_frozen_group_fits():
    with tempfile.TemporaryDirectory() as tmp:
        fn = make_face_tensor(tmp)
        tc, lb = _curved_face_fit(fn)
        x = tc.x.numpy().copy()
        got = _postfit_held(tc)
        (face,) = got["cpf"]["active"]
        assert face["multiplier"] < 0  # an upper bound
        # the curvature term is there and matters: the stat-block variance
        # without it (H instead of W along the face) is off by more than 1%
        Hc = tc.postfit_faces["hess_constraints"]
        A = tc.postfit_faces["jacobian"]
        _, _, H = tc.loss_val_grad_hess()
        H = np.asarray(H)
        ns = 3 + tc.indata.nsystnoconstraint
        held = [
            cpostfit.face_held_inverse(M[:ns, :ns], A[:, :ns], np.arange(ns))[0, 0]
            for M in (H - Hc, H)
        ]
        assert abs(held[1] / held[0] - 1) > 1e-2
        var_tot = got["cov"][0, 0]

        # syst groups, in the order rabbit writes them
        systs = [
            s.decode() if isinstance(s, bytes) else str(s) for s in tc.indata.systs
        ]
        groups = [[systs[k] for k in idx] for idx in tc.indata.systgroupidxs]
        constrained = systs[tc.indata.nsystnoconstraint :]
        ref = [
            np.sqrt(var_tot - _held_variance_with_frozen(fn, lb, x, g)) for g in groups
        ]
        var_stat = _held_variance_with_frozen(fn, lb, x, constrained)
        var_stat_nobbb = _held_variance_with_frozen(
            fn, lb, x, constrained, use_hess_no_bbb=True
        )
        ref += [np.sqrt(var_stat_nobbb), np.sqrt(var_stat - var_stat_nobbb)]
        ref += [
            np.sqrt(var_tot - _held_variance_with_frozen(fn, lb, x, g))
            for g in (["a"], ["b"], ["a", "b"])
        ]
        np.testing.assert_allclose(got["grouped"][0], ref, rtol=1e-6, atol=1e-10)
        # holding the face fixes a*b, so freezing either of a, b fixes both
        np.testing.assert_allclose(got["grouped"][0, -3:], got["grouped"][0, -1])


def test_curved_face_global_impacts():
    """At a curved face, against the fit with the face ELIMINATED: (a, b) on
    the circle, parametrised by an angle phi. That is an unconstrained fit
    whose Hessian carries the face curvature through the chain rule rather
    than through the multiplier, so at its minimum the covariance and the
    global impacts must equal the face-held ones exactly, and its
    finite-difference refits give the response of the constrained minimum to
    a global observable.

    The eliminated fit (trust-krylov, then exact Newton steps) defines the
    point. Neither
    the point nor the finite differences come from trust-constr refits: those
    converge only to the barrier tolerance, ~1e-8 in x, i.e. ~1 % of the
    response at delta = 1e-3, and platform-dependently so (that made an
    earlier version of this check fail in CI at 0.9 %)."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = make_face_tensor(tmp)
        tc, lb = _curved_face_fit(fn, noBinByBinStat=True)
        a0, b0, sa, sb = lb
        spec = (a0, b0, sa, sb, 4.0)
        names = _names(tc)
        ia, ib, j = names.index("a"), names.index("b"), names.index("normC")
        n = len(names)
        keep = [0] + list(range(3, n))  # sig, card nuisances (full model)
        keep_ref = [0] + list(range(2, n - 1))  # the same, eliminated model
        jr = j - 1  # one parameter fewer before the card nuisances

        def ref_fitter():
            return _fitter(fn, disk=spec, noBinByBinStat=True, minimizerGtol=1e-14)

        xt = tc.x.numpy()
        phi0 = np.arctan2((xt[ib] - b0) / sb, (xt[ia] - a0) / sa)
        xstart = np.concatenate([[xt[0], phi0], xt[3:]])
        ref = ref_fitter()
        ref.x.assign(xstart)
        ref.minimize()
        _newton_polish(ref)
        xe = ref.x.numpy().copy()
        _at(ref, xe)
        want = _postfit_plain(ref)

        # the same point in the full model, on the face to round-off
        phi = xe[1]
        xfull = np.concatenate(
            [[xe[0], a0 + 2 * sa * np.cos(phi), b0 + 2 * sb * np.sin(phi)], xe[2:]]
        )
        _at(tc, xfull)
        got = _postfit_held(tc)
        (face,) = got["cpf"]["active"]
        assert face["multiplier"] < 0
        C = got["cov"]
        cw = tc.cw.numpy()

        np.testing.assert_allclose(
            C[np.ix_(keep, keep)],
            want["cov"][np.ix_(keep_ref, keep_ref)],
            rtol=1e-8,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            got["global"][:, keep], want["global"][:, keep_ref], rtol=1e-8, atol=1e-13
        )
        nfix = len(tc.indata.systgroupidxs) + 1  # syst groups + stat
        np.testing.assert_allclose(
            got["global_grouped"][:, :nfix], want["global_grouped"][:, :nfix], rtol=1e-8
        )

        # The data-stat remainder is the eliminated fit's (asserted above with
        # the grouped columns): the sources add up to diag(C), as in any fit,
        # and the face curvature term -diag(C Hc C), Hc = sum mu d2c, counts as
        # data -- in the eliminated coordinates it IS part of the data
        # Hessian (data gradient x curvature of the face). So stat^2 is
        # diag(C H_data C) minus that term, not diag(C H_data C).
        _, _, H = tc.loss_val_grad_hess()
        H = np.asarray(H)
        Hc = tc.postfit_faces["hess_constraints"]
        stat = got["global_grouped"][0, len(tc.indata.systgroupidxs)]
        np.testing.assert_allclose(
            np.sum(got["global"][0] ** 2) + stat**2, C[0, 0], rtol=1e-8
        )
        np.testing.assert_allclose(
            stat**2, C[0] @ (H - np.diag(cw) - Hc) @ C[0], rtol=1e-8
        )
        assert abs(C[0] @ Hc @ C[0]) > 1e-3 * C[0, 0]  # and that term matters

        # finite differences on the eliminated fit: shift x0_j by +-delta and
        # refit from its minimum (Newton-polished); dx_poi/dx0_j = C_pj cw_j.
        # Measured on this model: relative error -0.051 delta^2 (5e-8 at
        # delta = 1e-3, scaling as delta^2 over 3e-4..3e-2), i.e. truncation
        # only; refit noise ~1e-9. Without the polish the refits stop at
        # |grad| ~1e-8 and the error is ~1e-4, erratic in delta.
        delta = 1e-3
        resp = []
        for sgn in (+1, -1):
            f = ref_fitter()
            x0 = f.x0.numpy().copy()
            x0[jr] += sgn * delta
            f.x0.assign(x0)
            f.x.assign(xe)
            f.minimize()
            _newton_polish(f)
            resp.append(f.x.numpy()[0])
        fd = (resp[0] - resp[1]) / (2 * delta)
        np.testing.assert_allclose(fd, C[0, j] * cw[j], rtol=1e-6)
        Crel = got["cpf"]["cov_released"]
        assert abs(Crel[0, j] * cw[j] - fd) > 1e3 * abs(C[0, j] * cw[j] - fd)


# ---------------------------------------------------------------------------
# (d) no active face: unchanged
# ---------------------------------------------------------------------------


def test_inactive_face_leaves_impacts_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        fn = make_face_tensor(tmp)
        free = _free_minimum(fn)
        x = free.x.numpy().copy()
        i = _names(free).index("freeB")
        tc = _fitter(fn, cons=[_bounds(f"freeB>={float(x[i] - 1.0)!r}")])
        _at(tc, x)
        _, g, h = tc.loss_val_grad_hess()
        cpf = tc.constrained_postfit(g, h)
        assert cpf["active"] == [] and tc.postfit_faces is None

        plain = _fitter(fn)
        _at(plain, x)
        cov = tf.constant(cpf["cov_held"], dtype=tc.indata.dtype)
        tc.cov.assign(cov)
        plain.cov.assign(cov)
        a, b = _impacts(tc, h), _impacts(plain, h)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
