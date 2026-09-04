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
reaches). The defaults put both below 1e-6 of the peak smeared density for a
50-130 GeV window and a 1-2 GeV resolution; ``tests/test_zgamma_kernel.py``
measures them.

Everything from the matrix element to the interpolation is plain TensorFlow, so
value, gradient and Hessian with respect to ``m_Z`` and ``Gamma_Z`` come from
the same tapes as the rest of the likelihood; no custom gradient is needed.

Approximations
--------------
* **Born level, LO.** No QCD corrections beyond what the NNLO PDF absorbs, no
  EW loop corrections, no running of alpha or of sin^2(theta_W) with ``m``.
* **No FSR.** The physics kernel is the *pre-FSR* lineshape by construction:
  final-state radiation is the separate empirical kernel CF ``phi_K`` of
  :class:`~rabbit.unbinned.MassCFTerm`, tabulated from the generator.
* **No acceptance.** The luminosity table is inclusive in boson rapidity (a
  ``--y-cut`` option exists in the generator but is off by default) and the
  lepton angular distribution is integrated over, so no lepton pT/eta cuts are
  folded in. A real data channel needs an ``m``-dependent acceptance
  ``A(m)`` multiplying the pdf.
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
        Uniform mass-grid points on the window. The default 16384 gives
        ``dm ~ 5 MeV`` on a 80 GeV window; the piecewise-linear representation
        error is then ~1e-7 of the peak density.
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
        nm=16384,
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

        # ---- mass grid and luminosities (constants) -----------------------
        m_grid = np.linspace(self.window[0], self.window[1], self.nm)
        self.dm = float(m_grid[1] - m_grid[0])
        self.m_grid = m_grid

        log_m, log_lumi, self.lumi_provenance = load_lumi_table(lumi)
        lo, hi = float(np.exp(log_m[0])), float(np.exp(log_m[-1]))
        if self.window[0] < lo - 1e-9 or self.window[1] > hi + 1e-9:
            raise ValueError(
                f"mass window {self.window} is outside the luminosity table's "
                f"range [{lo:.3f}, {hi:.3f}] GeV ({self.lumi_provenance['path']}); "
                "regenerate the table with a wider --m-lo/--m-hi."
            )
        from scipy.interpolate import CubicSpline

        lg = np.log(m_grid)
        self.lumis = np.array(
            [np.exp(CubicSpline(log_m, row)(lg)) for row in log_lumi]
        )

        # Zero the outermost node on each side: the represented pdf then ramps
        # linearly to zero over one bin instead of stepping, which makes the
        # hat-basis CF below exact for the pdf actually being normalised.
        edge = np.ones(self.nm)
        edge[0] = 0.0
        edge[-1] = 0.0
        self._edge = tf.constant(edge, dtype)

        self._m = tf.constant(m_grid, dtype)
        self._q2 = tf.constant(m_grid**2, dtype)
        self._lumis = tf.constant(self.lumis, dtype)

        # ---- CF grid (constants) ------------------------------------------
        dtau = 2.0 * np.pi / (self.nfft * self.dm)
        # +4 guard points for the 4-point interpolation stencil at the top end
        ntau = min(int(np.ceil(self.tau_max / dtau)) + 4, self.nfft // 2 + 1)
        if ntau < 8:
            raise ValueError(
                "tau grid too short; increase tau_max or decrease nfft/nm"
            )
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
        }

    @classmethod
    def from_config(cls, cfg, dtype=tf.float64):
        cfg = dict(cfg)
        cfg.pop("type", None)
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
        mz = tf.constant(self.mz_ref, self.dtype) + values[
            self.mz_param
        ] * self.npdt(self.mz_unit)
        gz = tf.constant(self.gz_ref, self.dtype) + values[
            self.gz_param
        ] * self.npdt(self.gz_unit)
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

        out = None
        for i, (_, q_f, i3) in enumerate(QUARKS):
            e_f = self.npdt(q_f)
            me = tf.constant(float(q_f) ** 2 / 2.0, self.dtype) / tf.square(q2)
            for g in (-e_f * s2, self.npdt(i3) - e_f * s2):
                me = me + prop_re * int_c * e_f * g
                me = me + prop_mod2 * z_c * tf.square(g)
            term = me * self._lumis[i]
            out = term if out is None else out + term
        out = self.npdt(2.0) * self._m * out
        if in_pb:
            out = out * self.npdt(GEV2PB)
        return out

    def pdf(self, values=None, **kw):
        """Normalised, window-truncated lineshape on the mass grid.

        ``sum_k pdf_k * dm == 1`` exactly (the outermost nodes are zero, so the
        trapezoid weight of every interior node is ``dm``).
        """
        y = self.dsigma_dm(values, in_pb=False, **kw) * self._edge
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
        x = tf.clip_by_value(
            x, tf.constant(0.0, self.dtype), self.npdt(float(n - 3))
        )
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
