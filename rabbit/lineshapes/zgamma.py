"""Z/gamma* physics-kernel provider for the unbinned mass term.

What this is
------------
:class:`~rabbit.unbinned.MassCFTerm` builds each candidate's density as the
inverse Fourier transform of a product of characteristic functions (CFs); the
*physics kernel* is the CF of the resonance lineshape itself. For the J/psi it
is 1 (:class:`~rabbit.unbinned.DeltaKernel`), for the Upsilon a Breit-Wigner
with the closed form ``e^{-Gamma|t|/2}``. The Z has no closed-form CF, so this
module produces one numerically and differentiably, as a function of the POIs
``m_Z`` and ``Gamma_Z``.

The physics
-----------
The (pre-FSR, Born-level) dilepton mass spectrum of neutral-current Drell-Yan
is, per quark flavour ``f`` and summed over the two lepton chiralities,

    dsigma/dm  =  2 m  * C * sum_f  L_f(m^2) *
        [   e_f^2 / (2 m^4)                                   (pure gamma*)
          + sum_{g in {g_L, g_R}}
              Re[chi] * (1 - 4 s2) / (4 s2 (1 - s2)) * e_f * g (interference)
            + |chi|^2 * (1 + (1 - 4 s2)^2)
                        / (32 s2^2 (1 - s2)^2) * g^2           (pure Z)
        ]

with ``s2 = sin^2(theta_W)``, ``e_f`` the quark charge, ``g_L = I3_f - e_f s2``,
``g_R = -e_f s2``, ``C = 4 pi alpha^2 / (3 N_c)`` and the propagator

    |chi|^2 = 1 / ((m^2 - m_Z^2)^2 + (m_Z Gamma_Z)^2)   ["fixed" width]
              1 / ((m^2 - m_Z^2)^2 + (m^2 Gamma_Z/m_Z)^2)  ["running" width]
    Re[chi] = (1 - m_Z^2 / m^2) * |chi|^2

``L_f`` is the LO parton luminosity at factorisation scale ``mu_F = m``,
tabulated once by :mod:`rabbit.lineshapes.make_lumi_table` (it does not depend
on any fitted parameter). The lepton-side factor is integrated inclusively, so
the lepton angular distribution -- and hence any acceptance -- is **not**
included; see "Approximations" below.

This is a direct port of ``hard_me`` in
``/work/submit/david_w/ZMass/calibration_studies/lineshape/zwidth_sensitivity.py``
(itself built on ``drell_yan_xsec.py`` from the same directory); given the same
luminosity table the two agree to float64 round-off, which
``tests/test_zgamma_kernel.py`` checks.

From the lineshape to the CF
----------------------------
The pdf is the above spectrum restricted to a mass window ``[m_lo, m_hi]`` and
renormalised, sampled on a uniform grid of ``nm`` points and represented by its
piecewise-linear (hat-basis) interpolant, with the outermost node on each side
forced to zero. Two consequences, both deliberate:

* **window truncation.** The kernel is the CF of the *truncated, renormalised*
  lineshape, i.e. of ``m_gen`` conditioned on ``m_lo < m_gen < m_hi``. The
  gamma* tail rises without bound as ``m -> 0``, so some window is unavoidable;
  it must be wide enough that the removed tails cannot migrate into the
  observed-mass fit range through the resolution (with sigma_m ~ 1.5 GeV a
  50-130 GeV window is ~27 sigma away from a Z peak at 91 GeV), and it should
  match the generator-level range of whatever sample the term is fitted to.
* **the truncation edge is a one-bin ramp**, not a step: the pdf goes linearly
  from 0 at ``m_lo`` to its physical value at ``m_lo + dm`` (``dm`` ~ 5 MeV by
  default). This makes the hat-basis representation exact -- the CF returned is
  then the *exact* CF of a genuine, everywhere-defined, exactly normalised pdf,
  which is what keeps ``phi(0) = 1`` to round-off and the inverse transform
  positive.

The CF of a hat-basis function on a uniform grid is analytic, so

    phi(tau) = dm * K(tau dm) * e^{i tau (m_lo - m_ref)}
                  * sum_k p_k e^{i tau k dm} ,
    K(x) = (sin(x/2) / (x/2))^2 ,

and the sum is a (conjugated) real DFT of the zero-padded ``p_k``: one
``tf.signal.rfft`` on ``nfft >= nm`` points gives ``phi`` on the uniform grid
``tau_j = 2 pi j / (nfft dm)``. Padding buys tau resolution only -- there is no
convolution and hence no aliasing; the transform is exact at every ``tau_j``.
``phi`` is then interpolated onto the per-candidate ``t_abs = t / sigma_i`` with
a 4-point Lagrange (cubic, O(dtau^4)) rule applied to ``Re phi`` and
``Im phi`` separately; ``log|phi|`` and ``arg phi`` -- what
:class:`~rabbit.unbinned.PhysicsKernel` returns -- are formed afterwards, so no
phase unwrapping is needed (only ``cos``/``sin`` of the phase are ever used
downstream, which are 2-pi periodic). The tabulation carries one extra node at
``tau = -dtau``, which costs nothing (``p`` is real, so ``phi(-tau) =
conj phi(tau)``) and lets the 4-point stencil reach ``tau = 0`` -- the most
heavily weighted point of the inverse transform, and the one place where
clamping instead would leave a visible O(dtau^2) error.

Accuracy is set by two grid spacings, both configurable and both cheap because
the transform happens once per NLL evaluation rather than once per candidate:
``dm`` (how well the piecewise-linear pdf represents the lineshape, O(dm^2))
and ``dtau`` (the interpolation, O(dtau^4) with a coefficient set by the fourth
moment of the *truncated* mass distribution, i.e. by how far the window
reaches). With ``nfft`` given as a multiple of ``nm`` the two are independent:
``dtau = 2 pi / (nfft dm)`` and ``dm = W / nm``, so ``dtau`` depends only on the
multiple and the window width. The defaults put both at or below ~1e-6 of the
peak smeared density for a 50-130 GeV window and a 1-2 GeV resolution;
``tests/test_zgamma_kernel.py`` measures them.

Everything from the matrix element to the interpolation is plain TensorFlow, so
value, gradient and Hessian with respect to ``m_Z`` and ``Gamma_Z`` come from
the same tapes as the rest of the likelihood; no custom gradient is needed.

Approximations
--------------
* **Born level, LO.** No QCD corrections beyond what the NNLO PDF absorbs, no
  EW loop corrections, no running of alpha or of sin^2(theta_W) with ``m``.
* **FSR** is *optional and multiplicative* (``fsr=``). Final-state radiation
  scales the mass, ``m_post = r m_pre``, and the distribution of
  ``u = -ln r`` is -- in this sample, verified from 60 to 150 GeV -- the same
  at every ``m_pre`` to a few per cent, so the fold is
  ``p_post(m) = sum_j w_j p_born(m / r_j) / r_j``.  With ``fsr`` given, ``pdf``
  (and hence the CF, and hence :class:`~rabbit.unbinned.MassCFTerm`) models the
  **post-FSR** mass as a function of the POIs alone, which is what makes the
  Z channel's FSR treatment exact rather than an additive convolution.  The
  atoms are a midpoint quadrature, so the residual bias scales as the square of
  the in-group spread of ``u``: on this sample the fitted ``Gamma_Z`` moves by
  +152 / +29 / +3.9 MeV for an in-group sd of 1e-2 / 3.3e-3 / 1e-3.  Optional
  per-atom ``m_lo``/``m_hi`` bands make the kernel piecewise constant in
  ``m_pre``, which is *required* once a lepton ``p_T`` cut is applied (the cut
  removes hard emission at an ``m_pre``-dependent rate).
  Without ``fsr`` the provider is the pre-FSR lineshape, and FSR has to be
  supplied as ``MassCFTerm``'s separate additive ``phi_K``.
* **Acceptance** is *optional* (``acceptance=``): a smooth multiplicative
  ``A(m)`` (Bernstein or a tabulated grid) applied to the Born spectrum
  *before* the FSR fold, i.e. the factorisation
  ``P(m_post, pass) = p_born(m_pre) A(m_pre) K_sel(m_post|m_pre)``.  Without it
  the luminosity table is inclusive in boson rapidity (a ``--y-cut`` option
  exists in the generator but is off by default) and the lepton angular
  distribution is integrated over, so no lepton pT/eta cuts are folded in.
* **No K-factor.** The hard ME and the parton luminosity are both LO.  Against
  the POWHEG-MiNNLO sample this leaves a smooth ratio that runs from 1.7 at
  55 GeV through 1.0 at the peak to 1.2 at 150 GeV; it is *six times* the full
  spread of PDF set, PDF order and mu_F in [Q/2, 2Q], so it is a genuine
  higher-order effect and not a luminosity choice.  It is **not** degenerate
  with the POIs: floating a 5-term smooth ``K(m)`` restores ``m_Z`` and
  ``Gamma_Z`` to the generator's inputs within 0.6 / 1.2 MeV on 29 M events and
  costs only 1.25x / 1.21x on their errors.
* **EW scheme.** Gmu, with ``sin^2(theta_W) = 1 - m_W^2/m_Z^2`` and
  ``alpha(m_Z) = sqrt(2) G_F m_W^2 sin^2 / pi`` evaluated once from the
  *reference* masses -- ``sin^2`` does not track a fitted ``m_Z``. It is a
  fixed input here (optionally a fitted nuisance, ``sin2_param``).
* **Width convention.** ``width_scheme="fixed"`` (default) is the constant-width
  propagator used by the reference study, POWHEG/MiNNLO and DYTurbo;
  ``"running"`` is the s-dependent-width form. The two schemes use *different*
  mass parameters, related by ``m_running = m_fixed sqrt(1 + (Gamma/m)^2)``
  (about +34 MeV at the Z); the defaults ``mz_ref``/``gz_ref`` follow the
  chosen scheme, and a fitted ``m_Z`` must be quoted in that scheme.

Usage
-----
::

    from rabbit import unbinned
    from rabbit.lineshapes import ZGammaLineshape

    provider = ZGammaLineshape(m_ref=91.1876, window=(50.0, 130.0))
    term = unbinned.MassCFTerm(
        "z", sigma=sigma, mobs=mass - 91.1876, tgrid=np.linspace(0, 10, 256),
        families=[{"name": "res", "param": "k_res", "kind": "gauss"}],
        vgf=np.ones_like(sigma), m_ref=91.1876,
        kernel=unbinned.TabulatedLineshapeKernel(provider=provider),
    )

``term.param_names`` then contains ``m_Z`` and ``Gamma_Z``; declare them as
POIs through :class:`~rabbit.param_models.unbinned_params.UnbinnedParams` by
passing ``param_is_poi`` to ``TensorWriter.add_unbinned_term`` (see
:meth:`ZGammaLineshape.param_declarations`).
"""

import json
import math
import os

import numpy as np
import tensorflow as tf

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ---------------------------------------------------------------------------
# Electroweak inputs.
#
# Verbatim from calibration_studies/lineshape/constants.py: the Gmu scheme of
# the DYTurbo CT18Z reference (ewscheme=1), whose independent inputs are G_F,
# m_W and m_Z. The Z mass and width quoted there are the *constant-width*
# scheme values used by POWHEG/MiNNLO and DYTurbo; the PDG numbers below are
# the running-width ones.
# ---------------------------------------------------------------------------
G_F = 1.1663787e-5  # GeV^-2
MW = 79.906853549493746  # GeV, constant-width scheme
MZ_FIXED = 91.153509740726733  # GeV, constant-width scheme
GZ_FIXED = 2.4932  # GeV, constant-width scheme
MZ_RUNNING = 91.1876  # GeV, PDG / s-dependent width
GZ_RUNNING = 2.4952  # GeV, PDG / s-dependent width

SIN2_DEFAULT = 1.0 - MW**2 / MZ_FIXED**2  # ~0.23152
ALPHA_EW = math.sqrt(2) * G_F * MW**2 * SIN2_DEFAULT / math.pi  # ~1/128.83
NC = 3
# GeV^-2 -> pb, including the 4 pi alpha^2 / (3 Nc) prefactor of the hard ME
GEV2PB = 4 * ALPHA_EW**2 * math.pi / (3 * NC) * 0.389379e9

# PDG id, electric charge, weak isospin -- the five flavours of the reference
# calculation (no top), in the order of the luminosity table's rows.
QUARKS = (
    (1, -1 / 3, -1 / 2),
    (2, 2 / 3, 1 / 2),
    (3, -1 / 3, -1 / 2),
    (4, 2 / 3, 1 / 2),
    (5, -1 / 3, -1 / 2),
)

DEFAULT_LUMI = "nnpdf31_nnlo_13tev"

# The three pieces of the neutral-current matrix element, selectable so that a
# fit can be repeated with the photon exchange and/or the gamma-Z interference
# switched off (the interference is what tilts the peak, so leaving it out
# moves the fitted mass by tens of MeV -- see tests/test_zgamma_kernel.py).
DEFAULT_TERMS = ("gamma", "int", "z")


def _load_fsr(fsr):
    """Normalise an FSR-kernel spec to ``{"r", "w", "m_lo", "m_hi"}`` float64.

    ``fsr`` is either such a mapping (lists accepted) or the path to an npz
    holding those arrays. ``r = m_post / m_pre`` are the kernel's support
    points and ``w`` the probabilities attached to them; ``r`` must lie in
    ``(0, 1]`` -- FSR only ever lowers the mass.

    ``m_lo``/``m_hi`` are optional and make the kernel **m-dependent**: an atom
    fires only for pre-FSR masses inside its own ``[m_lo, m_hi)`` band, and the
    weights are renormalised *within* each band, so the bands are a piecewise-
    constant-in-``m_pre`` conditional kernel rather than a single one.  A plain
    (band-less) kernel is stored as one band spanning ``(0, inf)``.

    An m-independent kernel is exact inclusively -- the ``u = -ln r`` spectrum
    of this sample is the same to a few per cent from 60 to 150 GeV -- but not
    once a lepton ``p_T`` cut is applied, because the cut removes hard emission
    at a rate that itself depends on ``m_pre``.
    """
    if isinstance(fsr, str):
        with np.load(fsr, allow_pickle=False) as d:
            fsr = {k: d[k] for k in ("r", "w", "m_lo", "m_hi") if k in d}
    r = np.asarray(fsr["r"], dtype=np.float64).ravel()
    w = np.asarray(fsr["w"], dtype=np.float64).ravel()
    if r.shape != w.shape or r.size == 0:
        raise ValueError("the FSR kernel needs equal-length, non-empty r and w")
    if not np.all((r > 0.0) & (r <= 1.0 + 1e-12)):
        raise ValueError("FSR kernel support must satisfy 0 < r <= 1")
    if np.any(w < 0.0):
        raise ValueError("FSR kernel weights must be non-negative")
    if "m_lo" in fsr and fsr["m_lo"] is not None:
        m_lo = np.asarray(fsr["m_lo"], dtype=np.float64).ravel()
        m_hi = np.asarray(fsr["m_hi"], dtype=np.float64).ravel()
        if m_lo.shape != r.shape or m_hi.shape != r.shape:
            raise ValueError("m_lo/m_hi must be per atom")
    else:
        m_lo = np.zeros_like(r)
        m_hi = np.full_like(r, np.inf)
    # renormalise each band to unit probability
    key = np.stack([m_lo, m_hi], axis=1)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    tot = np.bincount(inv, w, inv.max() + 1)
    if np.any(tot <= 0.0):
        raise ValueError("an FSR kernel band has zero total weight")
    return {"r": np.minimum(r, 1.0), "w": w / tot[inv], "m_lo": m_lo, "m_hi": m_hi}


def _bernstein(u, n):
    """The ``n + 1`` Bernstein basis polynomials of degree ``n`` at ``u``."""
    from math import comb

    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    return np.stack(
        [comb(n, k) * u**k * (1.0 - u) ** (n - k) for k in range(n + 1)], axis=-1
    )


def lumi_table_path(name):
    """Resolve a luminosity table given as a tag or as a path."""
    if os.path.sep in name or name.endswith(".npz"):
        if os.path.exists(name):
            return name
    cand = os.path.join(DATA_DIR, f"zlumi_{name}.npz")
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError(
        f"parton-luminosity table '{name}' not found (looked for '{cand}'). "
        "Available: "
        + ", ".join(
            sorted(
                f[len("zlumi_") : -len(".npz")]
                for f in os.listdir(DATA_DIR)
                if f.startswith("zlumi_") and f.endswith(".npz")
            )
        )
        + ". Regenerate one with rabbit/lineshapes/make_lumi_table.py."
    )


def load_lumi_table(name):
    """``(log_m, log_lumi (5, n), provenance dict)`` of a shipped npz table."""
    path = lumi_table_path(name)
    with np.load(path, allow_pickle=False) as d:
        log_m = np.asarray(d["log_m"], dtype=np.float64)
        log_lumi = np.asarray(d["log_lumi"], dtype=np.float64)
        prov = json.loads(str(d["provenance"][0]))
    prov["path"] = path
    return log_m, log_lumi, prov


def _next_pow2(n):
    return 1 << int(np.ceil(np.log2(max(int(n), 1))))


class ZGammaLineshape:
    """Differentiable Z/gamma* lineshape and its characteristic function.

    An instance is a ``provider`` for
    :class:`~rabbit.unbinned.TabulatedLineshapeKernel`: calling it as
    ``provider(values, t_abs)`` returns ``(log|phi|, arg phi)`` broadcast to the
    shape of ``t_abs``, where ``phi`` is the CF of the (window-truncated,
    renormalised) lineshape *relative to* ``m_ref``.

    Parameters
    ----------
    m_ref : float
        Reference mass of the term; ``mobs_i = m_i - m_ref`` and the CF is taken
        relative to it. Only shifts the phase, so its value is free -- use the
        term's ``m_ref``.
    window : (float, float)
        Generator-level mass window ``[m_lo, m_hi]`` the lineshape is truncated
        and renormalised to. Must lie inside the luminosity table's range.
    nm : int
        Uniform mass-grid points on the window. The default 32768 gives
        ``dm ~ 2.4 MeV`` on a 80 GeV window, for which the piecewise-linear
        representation error is ~1e-6 of the peak *smeared* density (it is
        O(dm^2): 16384 gives ~3e-6). Raising it costs only the transform --
        with the default ``nfft = 16 nm`` the CF grid spacing ``dtau =
        2 pi / (nfft dm) = 2 pi / (16 W)`` does not depend on ``nm`` at all,
        so the per-candidate interpolation is untouched.
    nfft : int, optional
        Zero-padded transform length (``>= nm``, rounded up to a power of two).
        Sets the CF grid spacing ``dtau = 2 pi / (nfft dm)`` and hence the
        interpolation error, which is O(dtau^4). Default ``16 * nm``.
    tau_max : float
        Largest ``|t| / sigma`` the tabulated CF covers, in GeV^-1. Beyond it
        the CF is continued as a constant; that region is always multiplied by
        a resolution factor ``exp(-sigma^2 tau^2 / 2) < e^{-300}`` for any
        realistic Z resolution, so the continuation is invisible. Use
        :meth:`check_tau_range` to assert it for a concrete term.
    lumi : str
        Luminosity table tag (looked up in ``rabbit/lineshapes/data``) or path.
    width_scheme : {"fixed", "running"}
        Propagator convention, see the module docstring.
    mz_param, gz_param : str
        Fit-parameter names. The fitted quantity is an *offset*:
        ``m_Z = mz_ref + mz_unit * value`` (and likewise for the width), so with
        the default ``*_unit = 1e-3`` the parameters are in MeV and start at 0.
    mz_ref, gz_ref : float, optional
        Central values; default to the scheme's own (``MZ_FIXED``/``GZ_FIXED``
        or ``MZ_RUNNING``/``GZ_RUNNING``).
    mz_unit, gz_unit : float
        Units of the fitted offsets (preconditioning, as elsewhere in
        :mod:`rabbit.unbinned`).
    sin2_param : str, optional
        Name of a fitted ``sin^2(theta_W)`` offset,
        ``s2 = sin2 + sin2_unit * value``. ``None`` (default) keeps it fixed.
    sin2, sin2_unit : float
        Central value and unit of ``sin^2(theta_W)``.
    dtype : tf.DType
        Graph dtype (float64 strongly recommended).
    """

    kind = "zgamma"

    def __init__(
        self,
        m_ref=MZ_RUNNING,
        window=(50.0, 130.0),
        nm=32768,
        nfft=None,
        tau_max=40.0,
        lumi=DEFAULT_LUMI,
        width_scheme="fixed",
        mz_param="m_Z",
        gz_param="Gamma_Z",
        mz_ref=None,
        gz_ref=None,
        mz_unit=1e-3,
        gz_unit=1e-3,
        sin2_param=None,
        sin2=SIN2_DEFAULT,
        sin2_unit=1e-3,
        terms=DEFAULT_TERMS,
        acceptance=None,
        fsr=None,
        fsr_mmax=None,
        dtype=tf.float64,
    ):
        if width_scheme not in ("fixed", "running"):
            raise ValueError(
                f"width_scheme must be 'fixed' or 'running', got '{width_scheme}'"
            )
        self.width_scheme = width_scheme
        self.m_ref = float(m_ref)
        self.window = (float(window[0]), float(window[1]))
        if not self.window[0] < self.window[1]:
            raise ValueError(f"empty mass window {self.window}")
        if self.window[0] <= 0:
            raise ValueError("the mass window must start above 0 GeV")
        self.nm = int(nm)
        if self.nm < 16:
            raise ValueError("nm must be at least 16")
        self.nfft = _next_pow2(16 * self.nm if nfft is None else nfft)
        if self.nfft < self.nm:
            raise ValueError(f"nfft ({self.nfft}) must be >= nm ({self.nm})")
        self.tau_max = float(tau_max)
        self.lumi_name = lumi
        self.dtype = dtype
        self.npdt = dtype.as_numpy_dtype

        self.mz_param = mz_param
        self.gz_param = gz_param
        self.mz_ref = float(
            (MZ_FIXED if width_scheme == "fixed" else MZ_RUNNING)
            if mz_ref is None
            else mz_ref
        )
        self.gz_ref = float(
            (GZ_FIXED if width_scheme == "fixed" else GZ_RUNNING)
            if gz_ref is None
            else gz_ref
        )
        self.mz_unit = float(mz_unit)
        self.gz_unit = float(gz_unit)
        self.sin2_param = sin2_param
        self.sin2 = float(sin2)
        self.sin2_unit = float(sin2_unit)
        self.param_names = tuple(
            p for p in (mz_param, gz_param, sin2_param) if p is not None
        )

        self.terms = tuple(terms)
        bad = set(self.terms) - set(DEFAULT_TERMS)
        if bad:
            raise ValueError(
                f"unknown lineshape term(s) {sorted(bad)}; "
                f"choose from {DEFAULT_TERMS}"
            )
        if not self.terms:
            raise ValueError("at least one lineshape term must be kept")
        self.acceptance = None if acceptance is None else dict(acceptance)
        self.fsr = None if fsr is None else _load_fsr(fsr)
        self.fsr_mmax = None if fsr_mmax is None else float(fsr_mmax)

        # ---- mass grid and luminosities (constants) -----------------------
        m_grid = np.linspace(self.window[0], self.window[1], self.nm)
        self.dm = float(m_grid[1] - m_grid[0])
        self.m_grid = m_grid

        log_m, log_lumi, self.lumi_provenance = load_lumi_table(lumi)
        lo, hi = float(np.exp(log_m[0])), float(np.exp(log_m[-1]))
        # The FSR fold needs the Born density ABOVE the output window; how far
        # above is set by the kernel's smallest r, which for an empirical kernel
        # runs down to O(1e-2) and would demand a 30 TeV grid.  The Born density
        # there is ~1e-5 of the peak and the kernel weight below r = 0.5 is
        # 0.7 %, so the grid is capped -- by default at the luminosity table's
        # own upper edge -- and the migration from beyond it is dropped.
        if fsr is None:
            m_hi_needed = self.window[1]
        else:
            cap = hi if self.fsr_mmax is None else self.fsr_mmax
            m_hi_needed = min(self.window[1] / float(self.fsr["r"].min()), cap)
        self.m_hi_born = m_hi_needed
        if self.window[0] < lo - 1e-9 or m_hi_needed > hi + 1e-9:
            raise ValueError(
                f"mass window {self.window} (Born grid up to "
                f"{m_hi_needed:.1f} GeV) is outside the luminosity table's "
                f"range [{lo:.3f}, {hi:.3f}] GeV ({self.lumi_provenance['path']}); "
                "regenerate the table with a wider --m-lo/--m-hi."
            )
        from scipy.interpolate import CubicSpline

        lg = np.log(m_grid)
        self.lumis = np.array([np.exp(CubicSpline(log_m, row)(lg)) for row in log_lumi])

        # Zero the outermost node on each side: the represented pdf then ramps
        # linearly to zero over one bin instead of stepping, which makes the
        # hat-basis CF below exact for the pdf actually being normalised.
        edge = np.ones(self.nm)
        edge[0] = 0.0
        edge[-1] = 0.0
        self._edge = tf.constant(edge, dtype)

        # ---- the FSR fold, the acceptance, and the extended Born grid ----
        # FSR is multiplicative -- m_post = r m_pre with r = 1 - x <= 1 -- so
        # the *post*-FSR density on the output grid needs the Born density
        # ABOVE the output window as well:
        #     p_post(m) = int_0^1 dr k(r) p_born(m/r) / r .
        # When a kernel is given the Born spectrum is therefore evaluated on an
        # extended grid of the same spacing, running from window[0] (the
        # generator's hard m_ll cut, below which p_born is genuinely zero) up to
        # window[1] / r_lo. p_born is linearly interpolated at m_i / r_j; the
        # indices and interpolation fractions do not depend on any fitted
        # parameter, so they are precomputed here and the fold costs two
        # gathers of shape (nm, n_kernel) per evaluation.
        if self.fsr is None:
            m_born = m_grid
        else:
            n_ext = int(np.ceil((self.m_hi_born - self.window[0]) / self.dm))
            n_ext = max(n_ext + 1, self.nm)
            m_born = self.window[0] + self.dm * np.arange(n_ext)
        self.m_born = m_born
        self.n_born = len(m_born)

        acc = self._acceptance_on(m_born)
        self._acc = None if acc is None else tf.constant(acc, dtype)

        if self.fsr is not None:
            # The fold is linear in the Born density and its coefficients do not
            # depend on any fitted parameter, so it collapses to one constant
            # (nm, n_born) matrix:  p_post = F p_born.  Building F sums the
            # atoms away at construction time -- a (nm, n_atoms) gather pair
            # would otherwise be re-materialised, and back-propagated through as
            # a scatter-add, on every Hessian evaluation, which is 30x slower
            # for a 3000-atom kernel and does not get cheaper as the kernel is
            # refined.
            r = self.fsr["r"]
            w = self.fsr["w"]
            F = np.zeros((self.nm, self.n_born))
            rows = np.arange(self.nm)
            for j in range(len(r)):
                m_src = m_grid / r[j]
                x = (m_src - self.window[0]) / self.dm
                i0 = np.floor(x).astype(np.int64)
                frac = x - i0
                ok = (i0 >= 0) & (i0 + 1 < self.n_born)
                ok &= (m_src >= self.fsr["m_lo"][j]) & (m_src < self.fsr["m_hi"][j])
                if not ok.any():
                    continue
                c = w[j] / r[j]
                idx = np.clip(i0, 0, self.n_born - 2)
                np.add.at(F, (rows[ok], idx[ok]), c * (1.0 - frac[ok]))
                np.add.at(F, (rows[ok], idx[ok] + 1), c * frac[ok])
            self._fsr_mat = tf.constant(F, dtype)

        self._m = tf.constant(m_born, dtype)
        self._q2 = tf.constant(m_born**2, dtype)
        lg_born = np.log(m_born)
        self._lumis = tf.constant(
            np.array([np.exp(CubicSpline(log_m, row)(lg_born)) for row in log_lumi]),
            dtype,
        )

        # ---- CF grid (constants) ------------------------------------------
        dtau = 2.0 * np.pi / (self.nfft * self.dm)
        # +4 guard points for the 4-point interpolation stencil at the top end
        ntau = min(int(np.ceil(self.tau_max / dtau)) + 4, self.nfft // 2 + 1)
        if ntau < 8:
            raise ValueError("tau grid too short; increase tau_max or decrease nfft/nm")
        self.dtau = dtau
        self.ntau = ntau
        self.tau_tab = np.arange(ntau) * dtau

        theta = self.tau_tab * self.dm
        with np.errstate(invalid="ignore"):
            khat = np.where(
                theta == 0.0, 1.0, (np.sin(0.5 * theta) / (0.5 * theta)) ** 2
            )
        psi = self.tau_tab * (self.window[0] - self.m_ref)
        self._pref_re = tf.constant(self.dm * khat * np.cos(psi), dtype)
        self._pref_im = tf.constant(self.dm * khat * np.sin(psi), dtype)

        self._npad = self.nfft - self.nm
        self._cache = None

    def _acceptance_on(self, m):
        """The acceptance factor ``A(m)`` on masses ``m``, or ``None``.

        ``self.acceptance`` is a mapping. Two forms:

        ``{"kind": "bernstein", "lo":, "hi":, "coef": [...]}``
            ``A(m) = sum_k coef_k B_{k,n}(u)``, ``u = (m - lo)/(hi - lo)``
            clipped to ``[0, 1]``, so ``A`` is constant outside ``[lo, hi]``.
            Bernstein because the coefficients are then bounded by the same
            interval as ``A`` itself, which keeps a fitted acceptance positive
            without a constraint.
        ``{"kind": "grid", "m": [...], "a": [...]}``
            linear interpolation of tabulated values, constant-extrapolated.

        The factor multiplies the *Born* spectrum, before the FSR fold: the
        probability that a candidate is selected is a property of the pre-FSR
        event (through the radiation it goes on to emit), and the FSR kernel
        that follows is the one measured *on selected events*. Together they
        are the factorisation
        ``P(m_post, pass) = p_born(m_pre) A(m_pre) K_sel(m_post|m_pre)``.
        """
        a = self.acceptance
        if a is None:
            return None
        kind = a.get("kind", "bernstein")
        m = np.asarray(m, dtype=np.float64)
        if kind == "bernstein":
            lo = float(a["lo"])
            hi = float(a["hi"])
            coef = np.asarray(a["coef"], dtype=np.float64)
            b = _bernstein((m - lo) / (hi - lo), len(coef) - 1)
            out = b @ coef
        elif kind == "grid":
            out = np.interp(m, np.asarray(a["m"], float), np.asarray(a["a"], float))
        else:
            raise ValueError(f"unknown acceptance kind '{kind}'")
        if np.any(out < 0.0):
            raise ValueError("the acceptance must be non-negative")
        return out

    # -- description -------------------------------------------------------
    def config(self):
        """JSON-serialisable description, round-tripped by :func:`from_config`."""
        return {
            "type": self.kind,
            "m_ref": self.m_ref,
            "window": list(self.window),
            "nm": self.nm,
            "nfft": self.nfft,
            "tau_max": self.tau_max,
            "lumi": self.lumi_name,
            "width_scheme": self.width_scheme,
            "mz_param": self.mz_param,
            "gz_param": self.gz_param,
            "mz_ref": self.mz_ref,
            "gz_ref": self.gz_ref,
            "mz_unit": self.mz_unit,
            "gz_unit": self.gz_unit,
            "sin2_param": self.sin2_param,
            "sin2": self.sin2,
            "sin2_unit": self.sin2_unit,
            "fsr_mmax": self.fsr_mmax,
            "terms": list(self.terms),
            "acceptance": (
                None
                if self.acceptance is None
                else {
                    k: (list(v) if isinstance(v, (list, tuple, np.ndarray)) else v)
                    for k, v in self.acceptance.items()
                }
            ),
            "fsr": (
                None
                if self.fsr is None
                else {
                    "r": self.fsr["r"].tolist(),
                    "w": self.fsr["w"].tolist(),
                    "m_lo": self.fsr["m_lo"].tolist(),
                    "m_hi": [
                        None if not np.isfinite(v) else float(v)
                        for v in self.fsr["m_hi"]
                    ],
                }
            ),
        }

    @classmethod
    def from_config(cls, cfg, dtype=tf.float64):
        cfg = dict(cfg)
        cfg.pop("type", None)
        f = cfg.get("fsr")
        if isinstance(f, dict) and "m_hi" in f:
            f = dict(f)
            f["m_hi"] = [np.inf if v is None else float(v) for v in f["m_hi"]]
            cfg["fsr"] = f
        return cls(dtype=dtype, **cfg)

    def param_declarations(self, mz_prior=None, gz_prior=None, sin2_prior=None):
        """Per-parameter ``(default, prior_sigma, prior_mean, is_poi)`` rows.

        Ordered like :attr:`param_names`. Defaults are 0 (i.e. the reference
        ``m_Z``/``Gamma_Z``), priors are ``NaN`` (free) unless given -- pass
        them in the *fitted* units, e.g. ``gz_prior=2.3`` for the PDG
        +-2.3 MeV world average on ``Gamma_Z``. ``m_Z`` and ``Gamma_Z`` are
        flagged as POIs, a floating ``sin^2(theta_W)`` as a nuisance.

        Feed the four columns straight to
        ``TensorWriter.add_unbinned_term(..., param_defaults=, ...)`` after
        merging with the term's other parameters (see
        :meth:`merge_declarations`).
        """
        rows = {
            self.mz_param: (0.0, mz_prior, 0.0, 1),
            self.gz_param: (0.0, gz_prior, 0.0, 1),
        }
        if self.sin2_param is not None:
            rows[self.sin2_param] = (0.0, sin2_prior, 0.0, 0)
        return {
            k: (v[0], np.nan if v[1] is None else float(v[1]), v[2], v[3])
            for k, v in rows.items()
        }

    def check_tau_range(self, tgrid, sigma):
        """Largest ``t/sigma`` a term will ask for, and whether it fits.

        Returns ``(tau_needed, ok)``. ``ok`` is False when the tabulated CF
        would be constant-continued in a region where the resolution factor has
        not yet killed the integrand.
        """
        tau_needed = float(np.max(tgrid) / np.min(sigma))
        return tau_needed, tau_needed <= self.tau_max

    # -- the physics -------------------------------------------------------
    def _values(self, values):
        one = tf.constant(1.0, self.dtype)
        mz = tf.constant(self.mz_ref, self.dtype) + values[self.mz_param] * self.npdt(
            self.mz_unit
        )
        gz = tf.constant(self.gz_ref, self.dtype) + values[self.gz_param] * self.npdt(
            self.gz_unit
        )
        if self.sin2_param is None:
            s2 = tf.constant(self.sin2, self.dtype)
        else:
            s2 = tf.constant(self.sin2, self.dtype) + values[
                self.sin2_param
            ] * self.npdt(self.sin2_unit)
        return mz, gz, s2, one

    def dsigma_dm(self, values=None, mz=None, gz=None, sin2=None, in_pb=True):
        """``dsigma/dm`` in pb/GeV on the mass grid (a TF tensor of length ``nm``).

        Either pass ``values`` (the ``{name: scalar}`` mapping the kernel gets)
        or explicit ``mz``/``gz``/``sin2``. ``in_pb=False`` drops the constant
        ``4 pi alpha^2 / (3 Nc)`` prefactor, which the normalisation removes
        anyway.
        """
        if values is not None:
            mz, gz, sin2, _ = self._values(values)
        else:
            mz = tf.constant(self.mz_ref if mz is None else mz, self.dtype)
            gz = tf.constant(self.gz_ref if gz is None else gz, self.dtype)
            sin2 = tf.constant(self.sin2 if sin2 is None else sin2, self.dtype)
        mz = tf.cast(mz, self.dtype)
        gz = tf.cast(gz, self.dtype)
        s2 = tf.cast(sin2, self.dtype)

        q2 = self._q2
        mz2 = mz * mz
        if self.width_scheme == "fixed":
            imag2 = mz2 * gz * gz
        else:  # running width: Gamma(s) = Gamma_Z s / m_Z^2
            imag2 = tf.square(q2 * gz / mz)
        prop_mod2 = tf.constant(1.0, self.dtype) / (tf.square(q2 - mz2) + imag2)
        prop_re = (tf.constant(1.0, self.dtype) - mz2 / q2) * prop_mod2

        int_c = (tf.constant(1.0, self.dtype) - self.npdt(4.0) * s2) / (
            self.npdt(4.0) * s2 * (tf.constant(1.0, self.dtype) - s2)
        )
        z_c = (
            tf.constant(1.0, self.dtype)
            + tf.square(tf.constant(1.0, self.dtype) - self.npdt(4.0) * s2)
        ) / (
            self.npdt(32.0)
            * tf.square(s2)
            * tf.square(tf.constant(1.0, self.dtype) - s2)
        )

        keep_g = "gamma" in self.terms
        keep_i = "int" in self.terms
        keep_z = "z" in self.terms

        out = None
        for i, (_, q_f, i3) in enumerate(QUARKS):
            e_f = self.npdt(q_f)
            if keep_g:
                me = tf.constant(float(q_f) ** 2 / 2.0, self.dtype) / tf.square(q2)
            else:
                me = tf.zeros_like(q2)
            for g in (-e_f * s2, self.npdt(i3) - e_f * s2):
                if keep_i:
                    me = me + prop_re * int_c * e_f * g
                if keep_z:
                    me = me + prop_mod2 * z_c * tf.square(g)
            term = me * self._lumis[i]
            out = term if out is None else out + term
        out = self.npdt(2.0) * self._m * out
        if in_pb:
            out = out * self.npdt(GEV2PB)
        return out

    def born_pdf(self, values=None, **kw):
        """The Born spectrum on :attr:`m_born`, times the acceptance.

        Not normalised -- :meth:`pdf` does that after the FSR fold.
        """
        y = self.dsigma_dm(values, in_pb=False, **kw)
        if self._acc is not None:
            y = y * self._acc
        return y

    def fold_fsr(self, y_born):
        """Apply the multiplicative FSR kernel, Born grid -> output grid.

        ``p_post(m_i) = sum_j w_j p_born(m_i / r_j) / r_j``, with ``p_born``
        linearly interpolated and taken to be zero outside :attr:`m_born`; the
        whole map is the constant matrix built in ``__init__``.
        """
        if self.fsr is None:
            return y_born
        return tf.linalg.matvec(self._fsr_mat, y_born)

    def pdf(self, values=None, **kw):
        """Normalised, window-truncated lineshape on :attr:`m_grid`.

        Born spectrum -> acceptance -> FSR fold -> window truncation ->
        normalisation, so with ``fsr`` given this is the density of the
        *post*-FSR mass and everything downstream (the CF, and hence
        :class:`~rabbit.unbinned.MassCFTerm`) models the post-FSR mass directly.

        ``sum_k pdf_k * dm == 1`` exactly (the outermost nodes are zero, so the
        trapezoid weight of every interior node is ``dm``).
        """
        y = self.fold_fsr(self.born_pdf(values, **kw)) * self._edge
        return y / (tf.reduce_sum(y) * self.npdt(self.dm))

    def dsigma_dm_np(self, mz=None, gz=None, sin2=None, in_pb=True):
        """Eager numpy convenience wrapper around :meth:`dsigma_dm`."""
        return self.dsigma_dm(mz=mz, gz=gz, sin2=sin2, in_pb=in_pb).numpy()

    # -- the characteristic function ---------------------------------------
    def cf_tab(self, values):
        """``(re, im)`` of ``phi(tau)`` on :attr:`tau_tab`."""
        re, im = self.cf_tab_ext(values)
        return re[1:], im[1:]

    def cf_tab_ext(self, values):
        """``(re, im)`` of ``phi`` on ``[-dtau] + tau_tab``, cached per call.

        The cache is keyed on the *identity* of the parameter tensors, which is
        what makes the one transform per NLL evaluation shared by all candidate
        chunks (``MassCFTerm`` builds ``values`` once in ``nll`` and calls the
        kernel once per chunk).
        """
        key = tuple(values[p] for p in self.param_names)
        cached = self._cache
        if cached is not None and len(cached[0]) == len(key):
            if all(a is b for a, b in zip(cached[0], key)):
                return cached[1]
        out = self._cf_tab(values)
        self._cache = (key, out)
        return out

    def _cf_tab(self, values):
        p = self.pdf(values)
        if self._npad:
            p = tf.concat([p, tf.zeros([self._npad], self.dtype)], axis=0)
        f = tf.signal.rfft(p)[: self.ntau]
        fr = tf.math.real(f)
        fi = tf.math.imag(f)
        # phi = dm K(tau dm) e^{i tau (m_lo - m_ref)} * conj(rfft(p))
        re = self._pref_re * fr + self._pref_im * fi
        im = self._pref_im * fr - self._pref_re * fi
        # Prepend the tau = -dtau point, which is free: p is real, so
        # phi(-tau) = conj(phi(tau)). Without it the 4-point stencil could not
        # reach below tau = dtau and everything in [0, dtau) -- which includes
        # t = 0, the most heavily weighted point of the inverse transform --
        # would be evaluated at dtau instead. That single clamped point is a
        # O(dtau^2) error on an O(1) integrand and dominated everything else.
        re = tf.concat([re[1:2], re], axis=0)
        im = tf.concat([-im[1:2], im], axis=0)
        return re, im

    def _interp(self, tab_re, tab_im, t_abs):
        """4-point Lagrange interpolation of ``(re, im)`` onto ``t_abs``.

        ``tab_*`` are indexed from ``tau = -dtau`` (see :meth:`_cf_tab`), so the
        node holding ``tau = x dtau`` is at position ``x + 1``. ``x`` is clamped
        to the tabulated range; beyond it the CF is continued as a constant,
        which is invisible behind the resolution factor (see ``tau_max``).
        """
        n = self.ntau
        x = t_abs / self.npdt(self.dtau)
        x = tf.clip_by_value(x, tf.constant(0.0, self.dtype), self.npdt(float(n - 3)))
        i0 = tf.floor(x)
        f = x - i0
        idx = tf.cast(i0, tf.int32) + 1

        w_m1 = -f * (f - 1.0) * (f - 2.0) / self.npdt(6.0)
        w_0 = (f + 1.0) * (f - 1.0) * (f - 2.0) / self.npdt(2.0)
        w_p1 = -(f + 1.0) * f * (f - 2.0) / self.npdt(2.0)
        w_p2 = (f + 1.0) * f * (f - 1.0) / self.npdt(6.0)

        def take(tab):
            return (
                w_m1 * tf.gather(tab, idx - 1)
                + w_0 * tf.gather(tab, idx)
                + w_p1 * tf.gather(tab, idx + 1)
                + w_p2 * tf.gather(tab, idx + 2)
            )

        return take(tab_re), take(tab_im)

    def log_cf(self, values, t_abs):
        """``(log|phi|, arg phi)`` at ``t_abs``: the ``PhysicsKernel`` contract."""
        tab_re, tab_im = self.cf_tab_ext(values)
        re, im = self._interp(tab_re, tab_im, t_abs)
        mod2 = tf.square(re) + tf.square(im)
        # The truncation edges give phi a 1/tau tail rather than an exponential
        # one, so |phi| stays well above any floor over the tabulated range;
        # the epsilon only guards pathological configurations.
        log_abs = self.npdt(0.5) * tf.math.log(mod2 + self.npdt(1e-300))
        return log_abs, tf.atan2(im, re)

    __call__ = log_cf

    # -- diagnostics -------------------------------------------------------
    def density_from_cf(self, values, m, tau_max=None, ntau=None):
        """Invert the tabulated CF back to a density at masses ``m``.

        Same quadrature family as :class:`~rabbit.unbinned.MassCFTerm` (a
        trapezoid over ``tau`` of ``Re[phi(tau) e^{-i tau (m - m_ref)}] / pi``),
        used by the tests to check that the transform round-trips onto
        :meth:`pdf`.
        """
        tau_max = self.tau_max if tau_max is None else float(tau_max)
        ntau = 200000 if ntau is None else int(ntau)
        tau = tf.constant(np.linspace(0.0, tau_max, ntau), self.dtype)
        tab_re, tab_im = self.cf_tab_ext(values)
        re, im = self._interp(tab_re, tab_im, tau[None, :])
        delta = tf.constant(np.asarray(m, dtype=np.float64) - self.m_ref, self.dtype)
        psi = -tau[None, :] * delta[:, None]
        integ = re * tf.cos(psi) - im * tf.sin(psi)
        dtau = tau[1] - tau[0]
        return tf.reduce_sum(
            dtau * (integ[:, 1:] + integ[:, :-1]) * self.npdt(0.5), axis=1
        ) / self.npdt(np.pi)

    def __repr__(self):
        return (
            f"ZGammaLineshape(window={self.window}, nm={self.nm}, "
            f"nfft={self.nfft}, dm={self.dm*1e3:.3f} MeV, "
            f"dtau={self.dtau:.2e}, ntau={self.ntau}, "
            f"scheme='{self.width_scheme}', lumi='{self.lumi_name}')"
        )
