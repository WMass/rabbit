"""`corr_a_max`: the FIRST-order coefficient of the fluctuation form has a
domain too.

`u_i(x) = sigma_i x + c_i x^2 + d_i` is truncated at first order in BOTH
coefficients -- `a_i`, which carries the `(1 - a_i x)` Jacobian recovering the
unconditional width, and `g_i = c_i/sigma_i`, the quadratic one. Only `g_i` had
a declared domain (`corr_coeff_max`), because the Z leg never exercised the
other. The J/psi leg does: measured on `joint_ok_full` at the DEFAULT parameter
point, two candidates of 3 000 000 have a NEGATIVE modelled density, both at
`sigma/m ~ 11 %` and ~4 sigma above the peak, and both with `|g_i| = 0.022` --
a factor 3.6 INSIDE `corr_coeff_max = 0.08`, so the quadratic bound cannot
reach them. `log` of a negative density then takes the whole joint NLL,
gradient and Hessian non-finite in one step.

`corr_a_max` is the same device on `a_i`: a per-candidate constant computed
from observables, so theta-independent, applied BEFORE `g` is formed (because
`c_i = -a_i sigma_i + sigma_i^2/m_i` is defined in terms of `a_i`, and bounding
one without the other would leave the two coefficients describing different
maps).

It defaults to OFF and must stay that way until it is scanned and costed:
bounding `g` degrades a correction, bounding `a` degrades the thing that
removes a bias of order `a_i sigma_i` -- 27 MeV at the Z.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .test_fluctuation import build, dens  # noqa: E402


def _a_g(t):
    return np.asarray(t._fl_a), np.asarray(t._fl_g)


def _term(a_res, **kw):
    n = len(a_res)
    rng = np.random.default_rng(7)
    sigma = np.full(n, 0.9)
    mobs = rng.normal(0.0, 1.0, n)
    return build(
        sigma,
        mobs,
        a_res=np.asarray(a_res, float),
        jensen_s2=np.full(n, 1e-4),
        corr_form="fluctuation",
        **kw,
    )


def test_default_is_off_and_bit_identical():
    """A default term and one that names the default must be the same term."""
    a_res = [0.01, 0.05, 0.20]
    t0 = _term(a_res)
    t1 = _term(a_res, corr_a_max=0.0)
    assert t0.corr_a_max == 0.0
    for x, y in zip(_a_g(t0), _a_g(t1)):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(dens(t0), dens(t1))


def test_the_bound_clips_a_and_g_follows_it():
    """`g` must be rebuilt from the BOUNDED `a`, not left describing the old map."""
    # the quadratic bound is switched off so only the new one acts: with it on,
    # `g` for the widest candidate is clipped as well and the identity below is
    # the composition of two bounds rather than a test of one.
    a_res = [0.01, 0.05, 0.20]
    free = _term(a_res, corr_coeff_max=0.0)
    bound = _term(a_res, corr_a_max=0.03, corr_coeff_max=0.0)
    a_free, g_free = _a_g(free)
    a_bnd, g_bnd = _a_g(bound)

    assert np.all(np.abs(a_bnd) <= 0.03 + 1e-15)
    # untouched where it was already inside
    inside = np.abs(a_free) <= 0.03
    np.testing.assert_array_equal(a_bnd[inside], a_free[inside])
    np.testing.assert_array_equal(g_bnd[inside], g_free[inside])
    # and `g` moved by exactly the change in `a`, since g = -a + (terms in
    # sigma and m only)
    np.testing.assert_allclose(g_bnd - g_free, -(a_bnd - a_free), atol=1e-14)


def test_a_wide_candidate_that_the_quadratic_bound_cannot_reach():
    """The J/psi failure in miniature: |g| inside its bound, |a| outside.

    With the Jensen term on the two largely cancel, `g = -a + sigma/m`, so a
    WIDE candidate can carry a large first-order coefficient while its quadratic
    one stays small. That is exactly the configuration `corr_coeff_max` is blind
    to, and it is the one measured on the J/psi leg: `|a| ~ 0.045`, `|g| ~
    0.022`, `sigma/m ~ 11 %`.
    """
    rng = np.random.default_rng(7)
    # sigma/m ~ 9.9 %, i.e. the wide population, and a_res tuned so the
    # cancellation leaves |g| ~ 0.02 as measured
    t = build(
        np.array([9.0]),
        rng.normal(0.0, 1.0, 1),
        a_res=np.array([0.12]),
        jensen_s2=np.array([1e-4]),
        corr_form="fluctuation",
        corr_coeff_max=0.08,
    )
    a, g = _a_g(t)
    assert abs(a[0]) > 0.08 > abs(g[0]), (a[0], g[0])
    assert abs(g[0]) < 0.03, g[0]
    tb = build(
        np.array([9.0]),
        rng.normal(0.0, 1.0, 1),
        a_res=np.array([0.12]),
        jensen_s2=np.array([1e-4]),
        corr_form="fluctuation",
        corr_coeff_max=0.08,
        corr_a_max=0.05,
    )
    ab, _ = _a_g(tb)
    assert abs(ab[0]) == 0.05


def test_set_corr_bounds_reaches_the_same_state_as_construction():
    """The load-time override must not be a second, slightly different path."""
    a_res = [0.01, 0.05, 0.20]
    built = _term(a_res, corr_a_max=0.03, corr_coeff_max=0.05)
    late = _term(a_res).set_corr_bounds(a_max=0.03, coeff_max=0.05)
    assert (late.corr_a_max, late.corr_coeff_max) == (0.03, 0.05)
    for x, y in zip(_a_g(built), _a_g(late)):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(dens(built), dens(late))


def test_set_corr_bounds_is_reversible():
    """A scan re-uses one loaded card, so the operation has to be idempotent."""
    a_res = [0.01, 0.05, 0.20]
    t = _term(a_res)
    a0, g0 = (x.copy() for x in _a_g(t))
    t.set_corr_bounds(a_max=0.02)
    assert np.max(np.abs(_a_g(t)[0])) <= 0.02 + 1e-15
    t.set_corr_bounds(a_max=0.0)
    np.testing.assert_array_equal(_a_g(t)[0], a0)
    np.testing.assert_array_equal(_a_g(t)[1], g0)


def test_it_survives_the_card_round_trip():
    t = _term([0.01, 0.05, 0.20], corr_a_max=0.03)
    assert t.config()["corr_a_max"] == 0.03


# --------------------------------------------------------------------------
# `set_corr_form`: the same two corrections, applied where they are EXACT
# --------------------------------------------------------------------------
def test_set_corr_form_round_trips_to_the_constructed_term():
    """Flipping the form at load time must reach the same term as building it."""
    a_res = np.array([0.01, 0.05, 0.20])
    js = np.full(3, 1e-4)
    rng = np.random.default_rng(7)
    sigma = np.full(3, 0.9)
    mobs = rng.normal(0.0, 1.0, 3)
    built = build(sigma, mobs, a_res=a_res, jensen_s2=js, corr_form="residual")
    flipped = build(
        sigma, mobs, a_res=a_res, jensen_s2=js, corr_form="fluctuation"
    ).set_corr_form("residual")
    assert flipped.corr_form == "residual" and not flipped._fluct
    # the fluctuation block must be switched OFF, not left stale
    assert flipped._fl_a is None and flipped._fl_g is None
    assert not flipped._fluct_active
    assert flipped._dyn_sigma == built._dyn_sigma
    assert flipped._jensen == built._jensen
    np.testing.assert_array_equal(dens(flipped), dens(built))


def test_set_corr_form_is_reversible():
    a_res = np.array([0.01, 0.05, 0.20])
    js = np.full(3, 1e-4)
    rng = np.random.default_rng(7)
    t = build(
        np.full(3, 0.9),
        rng.normal(0.0, 1.0, 3),
        a_res=a_res,
        jensen_s2=js,
        corr_form="fluctuation",
    )
    d0 = dens(t)
    a0 = np.asarray(t._fl_a).copy()
    t.set_corr_form("residual")
    t.set_corr_form("fluctuation")
    np.testing.assert_array_equal(np.asarray(t._fl_a), a0)
    np.testing.assert_array_equal(dens(t), d0)


def test_the_residual_form_cannot_produce_a_negative_density():
    """The point of the switch, on a candidate that breaks the other form.

    Wide, far out in the tail, and with a first-order coefficient big enough
    that the truncated Fourier factor overshoots -- the configuration measured
    on the J/psi leg. The residual form has no such factor: it is a positive
    kernel evaluated at a shifted residual with a positive Jacobian.
    """
    sigma = np.array([9.0])
    mobs = np.array([36.0])  # 4 sigma out, where the density is tiny
    a_res = np.array([0.35])
    js = np.array([1e-4])
    fl = build(
        sigma,
        mobs,
        a_res=a_res,
        jensen_s2=js,
        corr_form="fluctuation",
        corr_coeff_max=0.0,
    )
    res = build(sigma, mobs, a_res=a_res, jensen_s2=js, corr_form="residual")
    assert dens(fl)[0] <= 0.0 < dens(res)[0], (dens(fl)[0], dens(res)[0])
    # and the load-time flip fixes it in place
    fl.set_corr_form("residual")
    assert dens(fl)[0] > 0.0
