"""Unbinned likelihood terms: per-candidate contributions to the fit NLL.

rabbit's native likelihood is binned (Poisson / chi2 over bins) plus the
quadratic *external* terms of :mod:`rabbit.external_likelihood`. An
**unbinned term** is a third, additive contribution

    NLL_unbinned(x) = - sum_i log L_i(x)

evaluated over ``n`` candidates rather than over bins. It is a plain
differentiable TF function of a *named slice* of the fit parameter vector, so
value, gradient, Hessian and Hessian-vector products all come from the same
tapes the binned likelihood uses, and every downstream consumer (minimizer,
covariance, impacts, scans) works unchanged.

The concrete implementation here is the CVH mass likelihood
(:class:`MassCFTerm`): each candidate contributes the inverse Fourier
transform of a product of characteristic functions (CFs),

    L_i = 1/(pi sigma_i)
          Int_0^inf Re[ phi_K(t/sigma_i) phi_res(t/sigma_i) e^{S_i(t)}
                        e^{-i t (m_i - m_pred_i)/sigma_i} ] dt

with

* ``phi_K``     an empirical (constant) kernel CF -- the FSR kernel, tabulated
                on absolute ``t`` and interpolated per candidate;
* ``phi_res``   the *physics* kernel of the resonance: a delta function
                (J/psi, K_S), a Breit-Wigner (Upsilon; analytic CF
                ``e^{i m t - Gamma |t| / 2}``), or a numerical lineshape CF
                (Z/gamma*, a function of m_Z, Gamma_Z, ..., supplied by a
                provider from :mod:`rabbit.lineshapes`) -- see
                :class:`PhysicsKernel`;
* ``S_i(t)``    the per-candidate resolution CF exponent, a sum over
                *families* ``S_i(t) = sum_f k_f S_{f,i}(t)``, each family a
                tabulated complex exponent with one free scale ``k_f``. The
                family list is data driven: any number of families, and
                several families may share the same parameter name (which is
                how the "single resolution scale r" model is expressed);
* ``m_pred_i``  the predicted mass of candidate ``i``,
                ``m_ref (1 + alpha) + dm_res + (D delta_theta)_i``, i.e. a
                momentum-scale parameter, the resonance's own mass parameter,
                and an optional per-candidate *sparse* linear dependence on
                global parameters (the CVH track-fit Jacobian rows
                ``d m_i / d theta_k``).

The extended likelihood adds an analytic background pdf with a floating
fraction,

    L_i -> (1 - f) max(L_i, 0) + f B(m_i) ,

with ``B`` uniform on the fit window (:class:`UniformBackground`) or a
Bernstein polynomial (:class:`BernsteinBackground`).

When the candidates were *selected* in a mass window -- as a Z channel's are
-- the density has to be renormalised over it, ``L_i -> L_i / Z_i`` with
``Z_i = Int_window L_i(m) dm``; ``norm_window`` switches that on. Omitting it
is not a constant offset: ``Z`` moves with the resonance mass, so the missing
term biases it (order 10 MeV on ``m_Z`` for a 60-120 GeV window, whose lower
edge sits in the FSR tail).

Reference implementation
------------------------
The J/psi objective reproduced here bit-for-bit (up to float64 summation
order) is
``/work/submit/david_w/ZMass/calibration_studies/resolution/cf_masslik_fit.py``
(class ``MassNLL``). Two conventions from there are kept deliberately:

* the **softplus positivity floor** ``s log(1 + exp(L/s))`` with
  ``s = 1e-9``, which equals ``max(L, 0)`` to ~1e-9 absolute but is
  everywhere differentiable (``np.clip``/``tf.maximum`` has identically zero
  gradient below zero and stalls the minimizer);
* the **preconditioning units**: the momentum scale ``alpha`` is fitted in
  units of 1e-3 and the background fraction ``f_bkg`` in units of 1e-3. These
  are *not* priors -- they equalise the curvatures (``d2NLL/df2 ~ n/f^2 ~
  1e9`` in absolute units against ~1e2 for the resolution scales), which
  otherwise collapses the trust region.

On-disk format
--------------
Terms are stored in the input HDF5 under a top-level ``unbinned_terms``
group, one subgroup per term, mirroring ``external_terms``. Written by
:meth:`rabbit.tensorwriter.TensorWriter.add_unbinned_term`, read by
:func:`read_unbinned_terms_from_h5` (exposed as
``FitInputData.unbinned_terms``). Each subgroup holds

===========================  =================================================
``config``                   1-element vlen-str dataset with a JSON blob: the
                             term ``kind``, ``channel`` label, the family
                             list, kernel / background / mass-shift
                             configuration, units, floor and chunk size.
``params``                   vlen-str, all parameter names the term consumes
``param_defaults``           float64 (nparams,) starting values
``param_prior_sigmas``       float64 (nparams,) NaN = free parameter
``param_prior_means``        float64 (nparams,)
``param_is_poi``             int8   (nparams,) 1 = report as POI
``sigma``, ``mobs``          float64 (n,) resolution and ``m_i - m_ref``
``weights``                  float64 (n,) optional per-candidate weight
``vgf``                      float64 (n,) Gaussian variance fraction, used by
                             a family of kind ``gauss``
``tgrid``                    float64 (nt,) standardized quadrature grid
``S_re_<f>``/``S_im_<f>``    float32 (n, nt) tabulated family exponents
``phik_t``                   float64 (nk,) absolute-t grid of the kernel CF
``phik_re``/``phik_im``      float64 (nk,) the kernel CF on that grid
``jac_indices``/``_values``  int64 (nnz, 2) / float64 (nnz,) optional sparse
                             ``D`` (n x n_jacparams) in row-major order
``norm_sigma``/``norm_vgf``  float64 (K,) resolution classes of the truncation
                             normalisation (present iff ``norm_window`` is set)
``norm_class``               int64 (n,) class index of every candidate
``S_re_<f>_norm`` etc.       float32 (K, nt) the classes' family exponents
``a_res``                    float64 (n,) optional coefficient of the exported
                             sigma's own dependence on the candidate's
                             fluctuation, ``sigma_i = sigma_bar_i (1 + a_i
                             x_i)``; drives the self-consistent resolution
``grp_ptr``/``grp_id``       int64 (n+1,) / (nnz,) CSR block-sparse index of a
                             ``MaterialCF`` term's per-material-group exponents
``Sg_re_<f>``/``Sg_im_<f>``  float32 (nnz, nt) those exponents
``Sgfix_re_<f>``/``_im_<f>`` float32 (n, nt) their pinned baselines
``group_units``              float64 (ngroups,) units of the group parameters
``hit_units``                float64 (ncls,) units of the hit-class parameters
``hit_ptr``/``hit_cls``      int64 (n+1,) / (nnz,) CSR index of the Gaussian
                             hit-class variance shares
``hit_v``/``vg_other``       float64 (nnz,) / (n,) those shares and the
                             Gaussian remainder no parameter scales
===========================  =================================================

The per-candidate kernel CF ``phi_K(t/sigma_i)`` is *not* stored: it is
rebuilt at load time by linear interpolation of the (small) tabulation onto
``tgrid / sigma_i``, exactly as the reference does, which keeps the datacard
~2 GB smaller for a 300k-candidate sample.

Parameter wiring
----------------
The term declares parameter *names*; they are resolved against the fit
parameter vector (``ParamModel`` params followed by systematics) by
:func:`build_tf_unbinned_terms`, exactly like an external term. The names
themselves are declared to the fitter through the ordinary ParamModel
mechanism -- see :class:`rabbit.param_models.unbinned_params.UnbinnedParams`,
which reads the declarations above straight out of the datacard.
"""

import json
import math

import numpy as np
import tensorflow as tf
from wums import logging

from rabbit import h5pyutils_write
from rabbit.h5pyutils_read import maketensor

logger = logging.child_logger(__name__)

# Default preconditioning units, see module docstring.
ALPHA_UNIT = 1e-3
FBKG_UNIT = 1e-3
# Default softplus floor scale of the reference implementation.
FLOOR_SCALE = 1e-9
# Lower bound on the self-consistent per-candidate resolution, as a fraction
# of the exported sigma_i: `s_i = max(sigma_i - a_i delta_i, SIGMA_FLOOR sigma_i)`.
# The linearisation sigma = sigma_bar (1 + a x) is only meaningful while
# 1 + a x > 0; far in the tail the floor keeps s positive and the density finite
# without touching anything within several sigma of the peak (a_i ~ 0.011 at
# J/psi momenta, so the floor binds only beyond |x| ~ 70).
SIGMA_FLOOR = 0.2


# ---------------------------------------------------------------------------
# physics kernels: the CF of the resonance lineshape itself
# ---------------------------------------------------------------------------
class PhysicsKernel:
    """CF of the resonance lineshape, relative to its central mass.

    A kernel contributes two things to :class:`MassCFTerm`:

    ``log_cf(params, t_abs)``
        the additive complex exponent ``(re, im)`` of its CF, evaluated at
        the *absolute* conjugate variable ``t_abs = t / sigma_i`` (shape
        ``(chunk, nt)``). ``None`` means "identically zero".
    ``mass_shift(params)``
        a shift of the predicted mass (scalar tensor), e.g. a floating
        resonance mass relative to ``m_ref``.

    Both are handed ``values``, the ``{name: scalar tensor}`` mapping that
    :meth:`MassCFTerm._values` builds from the gathered parameter vector, so a
    kernel owns whatever subset of the fit parameters it names in
    ``param_names``.
    """

    kind = "base"
    param_names = ()

    def log_cf(self, values, t_abs):
        return None, None

    def mass_shift(self, values):
        return None

    def config(self):
        return {"type": self.kind}


class DeltaKernel(PhysicsKernel):
    """Zero-width resonance: CF = 1. J/psi, K_S, and any fixed-mass state."""

    kind = "delta"


class BreitWignerKernel(PhysicsKernel):
    """Relativistic-agnostic (non-relativistic) Breit-Wigner of width Gamma.

    The BW pdf ``(Gamma/2pi) / ((m - m_R)^2 + Gamma^2/4)`` has the analytic
    CF ``e^{i m_R t - Gamma |t| / 2}``. The ``e^{i m_R t}`` factor is absorbed
    into the predicted mass (:meth:`mass_shift`), leaving the real exponent
    ``-Gamma |t| / 2`` here. ``t_abs`` is non-negative on the quadrature grid,
    so ``|t|`` is ``t_abs``.

    Parameters
    ----------
    width_param : str
        Name of the fit parameter holding the width.
    mass_param : str, optional
        Name of the fit parameter holding the resonance mass *offset* with
        respect to the term's ``m_ref``. Omit for a fixed mass.
    width_unit, mass_unit : float
        Multipliers applied to the fitted values (preconditioning; e.g. fit
        the width in MeV with ``width_unit=1e-3`` while masses are in GeV).
    """

    kind = "breitwigner"

    def __init__(self, width_param, mass_param=None, width_unit=1.0, mass_unit=1.0):
        self.width_param = width_param
        self.mass_param = mass_param
        self.width_unit = float(width_unit)
        self.mass_unit = float(mass_unit)
        self.param_names = tuple(p for p in (mass_param, width_param) if p is not None)

    def log_cf(self, values, t_abs):
        gamma = values[self.width_param] * self.width_unit
        return -0.5 * gamma * t_abs, None

    def mass_shift(self, values):
        if self.mass_param is None:
            return None
        return values[self.mass_param] * self.mass_unit

    def config(self):
        return {
            "type": self.kind,
            "width_param": self.width_param,
            "mass_param": self.mass_param,
            "width_unit": self.width_unit,
            "mass_unit": self.mass_unit,
        }


class TabulatedLineshapeKernel(PhysicsKernel):
    """Interface for a numerically tabulated lineshape CF (Z/gamma*).

    The Z/gamma* lineshape has no closed-form CF: it is produced from the
    theory prediction (running widths, gamma*-Z interference, PDF-weighted
    parton luminosity) as a numerical function of the POIs (m_Z, Gamma_Z,
    ...) and Fourier transformed once per parameter point. The intended
    implementation tabulates ``log CF(t; theta)`` on a grid in ``theta`` and
    interpolates differentiably inside the graph; both the tabulation and its
    interpolation are physics work that belongs with the lineshape provider,
    not here.

    This class fixes the *interface* that provider must satisfy so the
    channel abstraction is real: it is a drop-in for any other
    :class:`PhysicsKernel`, and :class:`MassCFTerm` needs no change to use
    it.

    A provider is any object with

    * ``provider(values, t_abs) -> (re, im)``, the additive complex exponent
      ``(log|phi|, arg phi)`` broadcast to the shape of ``t_abs``;
    * ``param_names``, the fit parameters it consumes;
    * optionally ``config()``, a JSON-serialisable dict with a ``"type"`` key,
      which is what lets the kernel survive the round trip through the
      datacard: :func:`rabbit.lineshapes.make_provider` rebuilds it from that
      dict on the read side.

    :class:`rabbit.lineshapes.zgamma.ZGammaLineshape` is the reference
    implementation (Z/gamma*, POIs ``m_Z`` and ``Gamma_Z``).

    Parameters
    ----------
    param_names : sequence of str, optional
        Fit parameters the kernel consumes. Defaults to the provider's own
        ``param_names``; giving both is allowed but they must agree.
    provider : callable or dict, optional
        The provider object, or its ``config()`` dict (as stored in the
        datacard), which is instantiated through
        :func:`rabbit.lineshapes.make_provider`.
    """

    kind = "tabulated"

    def __init__(self, param_names=None, provider=None):
        if isinstance(provider, dict):
            from rabbit.lineshapes import make_provider

            provider = make_provider(provider)
        self.provider = provider

        own = getattr(provider, "param_names", None)
        if param_names is None:
            if own is None:
                raise ValueError(
                    "TabulatedLineshapeKernel needs param_names, or a provider "
                    "that declares its own param_names"
                )
            param_names = own
        elif own is not None and tuple(param_names) != tuple(own):
            raise ValueError(
                f"TabulatedLineshapeKernel: param_names {tuple(param_names)} "
                f"disagree with the provider's {tuple(own)}"
            )
        self.param_names = tuple(param_names)

    def log_cf(self, values, t_abs):
        if self.provider is None:
            raise NotImplementedError(
                "TabulatedLineshapeKernel is an interface stub: supply a "
                "provider(values, t_abs) -> (re, im) returning the log-CF of "
                "the numerical lineshape (see the class docstring)."
            )
        return self.provider(values, t_abs)

    def config(self):
        cfg = {"type": self.kind, "param_names": list(self.param_names)}
        provider_config = getattr(self.provider, "config", None)
        if callable(provider_config):
            cfg["provider"] = provider_config()
        return cfg


# ---------------------------------------------------------------------------
# background pdfs
# ---------------------------------------------------------------------------
class BackgroundPdf:
    """Normalised background density on the fit window ``[lo, hi]``."""

    kind = "base"
    param_names = ()

    def __init__(self, window):
        self.window = (float(window[0]), float(window[1]))

    @property
    def width(self):
        return self.window[1] - self.window[0]

    def pdf(self, values, m):
        raise NotImplementedError

    def config(self):
        return {"type": self.kind, "window": list(self.window)}


class UniformBackground(BackgroundPdf):
    """Flat combinatoric floor, ``1 / (hi - lo)``.

    This is the reference implementation's ``f / MWIN`` term with
    ``MWIN = hi - lo``.
    """

    kind = "uniform"

    def pdf(self, values, m):
        return tf.constant(1.0 / self.width, dtype=m.dtype)


class BernsteinBackground(BackgroundPdf):
    """Bernstein polynomial density of degree ``deg`` on ``[lo, hi]``.

    With ``x = (m - lo) / (hi - lo)`` and basis
    ``B_{k,n}(x) = C(n,k) x^k (1-x)^{n-k}``,

        pdf(m) = sum_k c_k B_{k,n}(x) / [ (hi - lo) sum_k c_k / (n + 1) ]

    using ``Int_0^1 B_{k,n} dx = 1/(n+1)``, so the density integrates to one
    over the window for *any* coefficients. Coefficients are ``c_k =
    softplus(p_k)``, which keeps the density non-negative everywhere without
    a constrained minimizer; ``p_k = softplus^-1(1) ~ 0.5413`` gives the flat
    density, which is the default starting point (matching
    ``AxisBernsteinModel``). The overall normalisation is *not* a free
    parameter -- the mixture fraction ``f_bkg`` carries it -- so one
    coefficient is redundant by construction; that redundancy is harmless
    (it is a flat direction of the background shape alone, lifted as soon as
    ``f_bkg`` floats) and keeps the parameterisation symmetric.
    """

    kind = "bernstein"

    def __init__(self, window, param_names):
        super().__init__(window)
        self.param_names = tuple(param_names)
        self.degree = len(self.param_names) - 1
        if self.degree < 0:
            raise ValueError("BernsteinBackground needs at least one coefficient")
        n = self.degree
        self._binom = np.array([float(math.comb(n, k)) for k in range(n + 1)])

    def pdf(self, values, m):
        dtype = m.dtype
        lo, hi = self.window
        x = (m - tf.constant(lo, dtype)) / tf.constant(self.width, dtype)
        x = tf.clip_by_value(x, tf.constant(0.0, dtype), tf.constant(1.0, dtype))
        n = self.degree
        coeffs = [tf.math.softplus(values[p]) for p in self.param_names]
        num = None
        for k, c in enumerate(coeffs):
            b = (
                tf.constant(self._binom[k], dtype)
                * x**k
                * (tf.constant(1.0, dtype) - x) ** (n - k)
            )
            num = c * b if num is None else num + c * b
        norm = tf.add_n(coeffs) / tf.constant(float(n + 1), dtype)
        return num / (norm * tf.constant(self.width, dtype))

    def config(self):
        cfg = super().config()
        cfg["params"] = list(self.param_names)
        return cfg


_BACKGROUNDS = {
    "uniform": UniformBackground,
    "bernstein": BernsteinBackground,
}
_KERNELS = {
    "delta": DeltaKernel,
    "breitwigner": BreitWignerKernel,
    "tabulated": TabulatedLineshapeKernel,
}


def _make_kernel(cfg):
    cfg = dict(cfg or {"type": "delta"})
    typ = cfg.pop("type")
    if typ not in _KERNELS:
        raise ValueError(f"unknown physics kernel type '{typ}'")
    return _KERNELS[typ](**cfg)


def _make_background(cfg):
    cfg = dict(cfg or {"type": "uniform", "window": [0.0, 1.0]})
    typ = cfg.pop("type")
    if typ not in _BACKGROUNDS:
        raise ValueError(f"unknown background type '{typ}'")
    return _BACKGROUNDS[typ](**cfg)


# ---------------------------------------------------------------------------
# the term base class
# ---------------------------------------------------------------------------
class UnbinnedTerm:
    """Base class of an additive unbinned NLL term.

    Subclasses must set ``self.param_names`` (the ordered list of fit
    parameter *names* the term consumes) and implement :meth:`nll`.

    The fitter gathers ``x_sub = x[indices]`` in exactly that order and calls
    ``nll(x_sub)``; the return value is added to the total NLL inside the same
    ``tf.function`` as the binned likelihood, so gradients, Hessians and HVPs
    all come for free from rabbit's existing tapes.

    Implementations should accumulate over candidates in chunks (additively --
    value, gradient and Hessian are all sums over candidates) so that the
    materialised ``(chunk, nt)`` temporaries stay bounded.
    """

    kind = "base"

    def __init__(self, name, channel=None):
        self.name = name
        self.channel = channel
        self.param_names = []
        # parameter declarations consumed by UnbinnedParams; filled by
        # subclasses / the reader.
        self.param_defaults = None
        self.param_prior_sigmas = None
        self.param_prior_means = None
        self.param_is_poi = None

    @property
    def nparams(self):
        return len(self.param_names)

    def nll(self, params, full_nll=False):
        """Scalar NLL contribution. ``params`` is ordered as ``param_names``."""
        raise NotImplementedError

    def _values(self, params):
        """Map the gathered parameter vector to a ``{name: scalar}`` dict."""
        return {name: params[i] for i, name in enumerate(self.param_names)}


# ---------------------------------------------------------------------------
# the CVH mass likelihood
# ---------------------------------------------------------------------------
class MassCFTerm(UnbinnedTerm):
    """Unbinned per-candidate mass likelihood built from characteristic functions.

    See the module docstring for the physics.

    Parameters
    ----------
    name : str
        Term name (the HDF5 subgroup name).
    sigma, mobs : ndarray (n,)
        Per-candidate mass resolution and ``m_i - m_ref``.
    tgrid : ndarray (nt,)
        Standardized quadrature grid (``t`` in units of ``1/sigma_i``); the
        integral is done by the trapezoid rule on this grid.
    families : list[dict]
        ``{"name", "param", "kind"}`` plus tabulated exponents. ``kind`` is
        ``"gauss"`` (the analytic Gaussian/hit term ``-0.5 vgf t^2``, built
        from ``vgf`` -- no ``(n, nt)`` array needed) or ``"tab"`` (arrays
        ``re`` and/or ``im`` of shape ``(n, nt)``, kept in their stored dtype
        and promoted inside the graph). Several families may share one
        ``param`` name; that is the single-resolution-scale model.
    vgf : ndarray (n,), optional
        Gaussian variance fraction, required by a ``gauss`` family.
    phik : (ndarray, ndarray, ndarray), optional
        ``(t, re, im)`` tabulation of the constant kernel CF on absolute
        ``t``; interpolated onto ``tgrid / sigma_i`` at construction.
    phik_grid : (ndarray, ndarray), optional
        Alternative to ``phik``: the already-interpolated ``(n, nt)`` arrays.
    kernel : PhysicsKernel
    background : BackgroundPdf
    m_ref : float
        Reference mass; ``m_i = mobs_i + m_ref`` and the momentum-scale shift
        is ``m_ref * alpha``.
    scale_param : str or None
        Name of the momentum-scale parameter ``alpha``.
    scale_unit : float
        Units of ``alpha`` (default 1e-3, see module docstring).
    bkg_frac_param : str or None
        Name of the floating background fraction. When ``None`` the fraction
        is fixed to ``bkg_frac``.
    bkg_frac : float
        Fixed background fraction (ignored when ``bkg_frac_param`` is given);
        0 removes the background component entirely.
    bkg_frac_unit : float
        Units of the fitted fraction (default 1e-3).
    jac : (indices, values, shape), optional
        Sparse ``D`` with ``D[i, k] = d m_i / d theta_k`` contracted with the
        parameters named in ``jac_params``: adds ``(D theta)_i`` to the
        predicted mass of candidate ``i``.
    jac_params : list[str]
        Names of the global parameters the sparse ``D`` multiplies.
    floor : {"softplus", "clip", "none"}
        Positivity treatment of ``L_i`` before the background mixture.
    floor_scale : float
        Softness of the softplus floor.
    chunk : int
        Number of candidates per accumulation chunk.
    weights : ndarray (n,), optional
        Per-candidate weights.
    a_res : ndarray (n,), optional
        Per-candidate coefficient of the *exported* resolution's dependence on
        the candidate's own fluctuation, ``sigma_i = sigma_bar_i (1 + a_i
        x_i)``. When given (and non-zero) the width used in the likelihood
        becomes a function of the parameters, ``s_i = sigma_i - a_i delta_i``
        -- see :meth:`_chunk_sigma`. Absent or all-zero takes the static path,
        which is bit-identical to the code before the correction existed.
    self_consistent_sigma : bool
        Master switch for that correction; ``False`` ignores ``a_res``.
    sigma_floor : float
        Lower bound on ``s_i`` as a fraction of ``sigma_i`` (see
        :data:`SIGMA_FLOOR`).
    norm_window : (float, float), optional
        Mass window the candidates were *selected* in. When given, the density
        is divided by its own integral over the window,
        ``L_i -> L_i / Z_i`` with ``Z_i = Int_lo^hi L_i(m) dm``, i.e. the
        likelihood becomes the correct truncated one. ``Z`` is evaluated on a
        mass grid for a handful of resolution *classes* rather than per
        candidate (see ``norm``); the class assignment is stored in
        ``norm_class``. Omit for an untruncated sample.
    norm_tpoints : int
        Number of points of the (midpoint) ``t`` grid the truncation integral
        uses. It has to resolve oscillations at the window half-width, so it is
        much finer than the term's own ``tgrid``; the family exponents are
        resampled onto it with a cubic spline when the term is built.
    upsample : int
        Integrate the density on a ``tau`` grid this many times finer than the
        stored ``tgrid``, expanding the tabulated family exponents inside the
        graph with a fixed cubic-spline matrix. The density is an inverse
        Fourier transform whose integrand oscillates ``|m_i - m_pred|/sigma_i``
        times across the grid; over a Z window that reaches ~60 periods, which
        the in-maker's 64 exported points do not resolve. The exponents are
        smooth in ``tau``, so the expansion is faithful, and doing it in the
        graph keeps the *datacard* at the stored resolution -- only the
        per-chunk intermediates grow.
    norm : dict, optional
        The resolution classes, ``{"sigma": (K,), "vgf": (K,), "class": (n,),
        "families": [...]}``; the family entries mirror ``families`` but carry
        ``(K, nt)`` arrays. Required with ``norm_window``.
    dtype : tf.DType
        Graph dtype for the real arithmetic (float64 recommended).
    """

    kind = "MassCF"

    def __init__(
        self,
        name,
        sigma,
        mobs,
        tgrid,
        families,
        vgf=None,
        phik=None,
        phik_grid=None,
        kernel=None,
        background=None,
        m_ref=0.0,
        scale_param=None,
        scale_unit=ALPHA_UNIT,
        bkg_frac_param=None,
        bkg_frac=0.0,
        bkg_frac_unit=FBKG_UNIT,
        jac=None,
        jac_params=(),
        floor="softplus",
        floor_scale=FLOOR_SCALE,
        chunk=32768,
        weights=None,
        a_res=None,
        self_consistent_sigma=True,
        sigma_floor=SIGMA_FLOOR,
        norm_window=None,
        norm_tpoints=8192,
        upsample=1,
        norm=None,
        channel=None,
        dtype=tf.float64,
        param_defaults=None,
        param_prior_sigmas=None,
        param_prior_means=None,
        param_is_poi=None,
    ):
        super().__init__(name, channel=channel)
        self.dtype = dtype
        self.npdt = dtype.as_numpy_dtype
        self.m_ref = float(m_ref)
        self.scale_param = scale_param
        self.scale_unit = float(scale_unit)
        self.bkg_frac_param = bkg_frac_param
        self.bkg_frac = float(bkg_frac)
        self.bkg_frac_unit = float(bkg_frac_unit)
        self.floor = floor
        self.floor_scale = float(floor_scale)
        self.self_consistent_sigma = bool(self_consistent_sigma)
        self.sigma_floor = float(sigma_floor)
        self.kernel = kernel if kernel is not None else DeltaKernel()
        self.background = (
            background
            if background is not None
            else UniformBackground((m_ref - 0.5, m_ref + 0.5))
        )
        self.jac_params = list(jac_params)

        sigma = np.asarray(sigma, dtype=np.float64)
        mobs = np.asarray(mobs, dtype=np.float64)
        tgrid = np.asarray(tgrid, dtype=np.float64)
        self.n = len(sigma)
        self.nt = len(tgrid)
        if mobs.shape != sigma.shape:
            raise ValueError("sigma and mobs must have the same length")

        self.chunk = int(min(chunk, self.n)) if self.n else 1
        self.nchunk = int(np.ceil(self.n / self.chunk)) if self.n else 0
        self._chunks = [
            (i * self.chunk, min(self.n, (i + 1) * self.chunk))
            for i in range(self.nchunk)
        ]

        self.sigma = tf.constant(sigma, dtype)
        self.mobs = tf.constant(mobs, dtype)
        self.upsample = int(upsample)
        if self.upsample < 1:
            raise ValueError("upsample must be >= 1")
        self.tgrid_stored = tgrid
        if self.upsample > 1:
            from scipy.interpolate import CubicSpline

            tfine = np.linspace(tgrid[0], tgrid[-1],
                                (self.nt - 1) * self.upsample + 1)
            self._upmat = tf.constant(
                CubicSpline(tgrid, np.eye(self.nt), axis=0)(tfine), dtype
            )
            tgrid = tfine
        else:
            self._upmat = None
        self.nt_int = len(tgrid)
        self.tgrid = tf.constant(tgrid, dtype)
        # trapezoid weights: d = diff(t); trapz = sum(d * (y[1:] + y[:-1]) / 2)
        self.dtgrid = tf.constant(np.diff(tgrid), dtype)
        self.weights = (
            None
            if weights is None
            else tf.constant(np.asarray(weights, dtype=np.float64), dtype)
        )

        # SELF-CONSISTENT RESOLUTION (see `_chunk_sigma`).  `a_res` is the
        # per-candidate coefficient of the exported sigma's dependence on the
        # candidate's own fluctuation; an all-zero (or absent) `a_res` means the
        # correction is off and the STATIC path is taken, which is then
        # bit-identical to the code before it existed.
        self.a_res = None
        if a_res is not None:
            arr = np.asarray(a_res, dtype=np.float64).ravel()
            if arr.shape != (self.n,):
                raise ValueError(
                    f"a_res has shape {arr.shape}, expected {(self.n,)}"
                )
            self.a_res = tf.constant(arr, dtype)
            self._a_res_np = arr
        self._dyn_sigma = (
            self.a_res is not None
            and self.self_consistent_sigma
            and bool(np.any(self._a_res_np != 0.0))
        )

        # ---- families -----------------------------------------------------
        self.families = []
        for f in families:
            entry = {
                "name": f["name"],
                "param": f["param"],
                "kind": f.get("kind", "tab"),
            }
            if entry["kind"] == "gauss":
                if vgf is None:
                    raise ValueError(
                        f"family '{entry['name']}' is of kind 'gauss' but no vgf given"
                    )
            else:
                for comp in ("re", "im"):
                    arr = f.get(comp)
                    if arr is not None:
                        arr = np.asarray(arr)
                        if arr.shape != (self.n, self.nt):
                            raise ValueError(
                                f"family '{entry['name']}' component '{comp}' has "
                                f"shape {arr.shape}, expected {(self.n, self.nt)}"
                            )
                        # kept in the stored dtype (float32 halves the memory
                        # of the (n, nt) blocks) and promoted inside the graph
                        entry[comp] = tf.constant(arr, dtype=tf.as_dtype(arr.dtype))
                if "re" not in entry and "im" not in entry:
                    raise ValueError(
                        f"family '{entry['name']}' has neither 're' nor 'im'"
                    )
            self.families.append(entry)

        self.vgf = (
            None
            if vgf is None
            else tf.constant(np.asarray(vgf, dtype=np.float64), dtype)
        )

        # ---- kernel CF ----------------------------------------------------
        self.phik_tab = None
        if phik_grid is not None:
            pk_re, pk_im = phik_grid
            self.phik_re = tf.constant(np.asarray(pk_re), dtype)
            self.phik_im = tf.constant(np.asarray(pk_im), dtype)
        elif phik is not None:
            t_tab, re_tab, im_tab = (np.asarray(a, dtype=np.float64) for a in phik)
            self.phik_tab = (t_tab, re_tab, im_tab)
            # ALWAYS kept as tensors too (three (nk,) vectors, nothing), so a
            # parameter-dependent sigma or an upsampled tau grid can
            # re-interpolate the kernel CF inside the graph
            self._pk_t = tf.constant(t_tab, dtype)
            self._pk_re = tf.constant(re_tab, dtype)
            self._pk_im = tf.constant(im_tab, dtype)
            if self.upsample > 1 or self._dyn_sigma:
                # the pre-interpolated (n, nt) grid is skipped: with `upsample`
                # it would be that many times larger, and with a
                # parameter-dependent sigma it is simply wrong (it is pinned to
                # the exported sigma).  Both cases blend the tabulation per
                # chunk instead, which needs a uniform grid.
                d = np.diff(t_tab)
                if not np.allclose(d, d[0]):
                    raise ValueError(
                        "an upsampled tau grid or a parameter-dependent sigma "
                        "needs a uniformly spaced kernel CF tabulation"
                    )
                self.phik_re = None
                self.phik_im = None
            else:
                # absolute t per candidate, interpolated exactly as the reference
                tgi = tgrid[None, :] / sigma[:, None]
                self.phik_re = tf.constant(np.interp(tgi, t_tab, re_tab), dtype)
                self.phik_im = tf.constant(np.interp(tgi, t_tab, im_tab), dtype)
                del tgi
        else:
            self.phik_re = None
            self.phik_im = None

        # ---- truncation normalisation -------------------------------------
        self.norm_window = None if norm_window is None else (
            float(norm_window[0]), float(norm_window[1]))
        self.norm_tpoints = int(norm_tpoints)
        self._norm = None
        if self.norm_window is not None:
            if norm is None:
                raise ValueError(
                    "norm_window given without the 'norm' resolution classes"
                )
            self._build_norm(norm, phik)

        # ---- sparse per-candidate parameter dependence D ------------------
        self._jac_chunks = None
        if jac is not None and len(self.jac_params):
            idx, val, shape = jac
            idx = np.asarray(idx, dtype=np.int64).reshape(-1, 2)
            val = np.asarray(val, dtype=np.float64)
            if int(shape[1]) != len(self.jac_params):
                raise ValueError(
                    f"jac has {shape[1]} columns but {len(self.jac_params)} "
                    "jac_params were given"
                )
            self._jac_chunks = []
            for lo, hi in self._chunks:
                m = (idx[:, 0] >= lo) & (idx[:, 0] < hi)
                sub = idx[m].copy()
                sub[:, 0] -= lo
                order = np.lexsort((sub[:, 1], sub[:, 0]))
                self._jac_chunks.append(
                    tf.sparse.SparseTensor(
                        sub[order],
                        tf.constant(val[m][order], dtype),
                        [hi - lo, len(self.jac_params)],
                    )
                )
        elif jac is not None:
            raise ValueError("jac given without jac_params")

        # ---- parameter list ------------------------------------------------
        names = []
        if scale_param is not None:
            names.append(scale_param)
        for f in self.families:
            if f["param"] not in names:
                names.append(f["param"])
        for p in self._extra_param_names():
            if p not in names:
                names.append(p)
        for p in self.kernel.param_names:
            if p not in names:
                names.append(p)
        for p in self.background.param_names:
            if p not in names:
                names.append(p)
        if bkg_frac_param is not None and bkg_frac_param not in names:
            names.append(bkg_frac_param)
        for p in self.jac_params:
            if p not in names:
                names.append(p)
        self.param_names = names

        npar = len(names)
        self.param_defaults = (
            np.zeros(npar) if param_defaults is None else np.asarray(param_defaults)
        )
        self.param_prior_sigmas = (
            np.full(npar, np.nan)
            if param_prior_sigmas is None
            else np.asarray(param_prior_sigmas)
        )
        self.param_prior_means = (
            self.param_defaults.copy()
            if param_prior_means is None
            else np.asarray(param_prior_means)
        )
        self.param_is_poi = (
            np.zeros(npar, dtype=np.int8)
            if param_is_poi is None
            else np.asarray(param_is_poi).astype(np.int8)
        )
        for arr, label in (
            (self.param_defaults, "param_defaults"),
            (self.param_prior_sigmas, "param_prior_sigmas"),
            (self.param_prior_means, "param_prior_means"),
            (self.param_is_poi, "param_is_poi"),
        ):
            if arr.shape != (npar,):
                raise ValueError(
                    f"{label} has shape {arr.shape}, expected {(npar,)} for "
                    f"parameters {names}"
                )

    # -- internals ---------------------------------------------------------
    def _extra_param_names(self):
        """Parameter names a subclass contributes (see ``MaterialCFTerm``)."""
        return ()

    # ---- hooks a subclass overrides to change WHAT is evaluated, without
    # ---- touching the quadrature that evaluates it.  All three are no-ops
    # ---- here, so MassCFTerm's NLL is unchanged.
    def _chunk_exponent_scale(self, values, ci):
        """Optional per-candidate ``(nchunk,)`` multiplier of the resolution
        log-CF exponent, applied to Re S and Im S alike.

        Reserved for effects that scale a candidate's whole process-noise
        exponent by something other than the material amounts -- e.g. the
        dependence of the FITTED per-candidate sigma on the candidate's own
        fluctuation (the block variances are evaluated at the fitted state, so
        sigma_obs = sigma_bar (1 + a x)).  Returning ``None`` means 1.
        """
        return None

    def _chunk_mean_shift(self, values, ci):
        """Optional per-candidate ``(nchunk,)`` ADDITIVE shift of the model mean.

        Reserved for a deterministic per-candidate offset of the predicted mass
        that is not a global parameter -- e.g. the second-order Jensen term of
        the mass functional, ``0.5 tr(H Sigma)``.  ``None`` means zero.
        """
        return None

    def _chunk_sigma(self, values, ci, delta):
        """Per-candidate ``(nchunk,)`` resolution, which DEPENDS on the
        parameters when the self-consistent correction is on.

        ``sigma_i`` as exported is the FIT's own error, assembled from the block
        variances at the converged state, so it is a function of the very
        fluctuation the likelihood is measuring
        (``sigma_i = sigma_bar_i (1 + a_i x_i)``).  Treating it as a known
        constant fits a density whose width is correlated with its residual.
        The truth-free repair is to make the absolute scale a function of the
        parameters, ``s_i(theta) = sigma_i - a_i delta_i(theta)``, which is what
        a subclass returns here.

        THE DEFECT.  ``sigma_i`` as exported is the FIT's own error, assembled
        from the block variances at the converged state, and the process noise
        that dominates it scales with the momenta -- so it is a monotone
        function of the FITTED mass, i.e. of the very fluctuation the
        likelihood is measuring.  Writing ``sigma_i = sigma_bar_i (1 + a_i
        x_i)`` with ``x_i`` the truth-referenced standardized residual gives
        ``z_i = x_i/(1 + a_i x_i)`` and ``E[z_i] = -a_i``, so a likelihood that
        treats ``sigma_i`` as a known constant fits a density whose width is
        correlated with its residual.  The bias on a mass-scale parameter is
        ``-a_m F sigma_bar_eff / M`` with ``F = 1.624`` measured on the J/psi
        gun CF model (F = 2 for a Gaussian): ``-0.146e-3`` on the gun, and
        ``-24 to -43 MeV`` on ``m_Z`` at Z momenta.

        THE REPAIR, with no truth anywhere.  ``sigma_i = sigma_bar_i + a_i
        (m_i - mu_i)`` is the DEFINITION of ``a_i``, so

            s_i(theta) = max(sigma_i - a_i delta_i(theta), SIGMA_FLOOR sigma_i)

        recovers the unconditional resolution from observed quantities alone.
        At the true parameters it equals ``sigma_bar_i`` exactly, so the score
        has zero expectation and the estimator is unbiased to first order.

        Returning ``None`` means the stored, parameter-independent ``sigma`` --
        and then the pre-interpolated kernel CF grid is used, which is both the
        cheap path and bit-identical to the code before this hook existed.  That
        is what an absent or all-zero ``a_res`` selects, and it is exact rather
        than an approximation: ``a = 0`` makes ``s_i == sigma_i`` identically,
        so the two paths compute the same function.

        Spec, derivation and acceptance gates:
        ``calibration_studies/resolution/oddmoment/MASSCFTERM_SPEC.md``.

        When it is NOT None, ``s_i`` enters in the three places it appears:
        the ``1/(pi s_i)`` prefactor, the standardized-to-absolute map
        ``t_abs = tgrid / s_i``, and the kernel CF argument ``phi_K(t_abs)``,
        which is then interpolated IN GRAPH from ``phik_tab``.  The resolution
        EXPONENTS are functions of the standardized ``t`` and are untouched:
        they describe the shape, only the absolute scale is corrected.  The
        ``-ln s_i(theta)`` in ``log L_i`` becomes parameter-dependent and
        autodiff picks it up from the prefactor -- it must not be dropped.
        """
        if not self._dyn_sigma:
            return None
        lo, hi = self._chunks[ci]
        sig = self.sigma[lo:hi]
        return tf.maximum(sig - self.a_res[lo:hi] * delta,
                          self.npdt(self.sigma_floor) * sig)

    def _chunk_residual(self, values, ci):
        """Residual ``delta_i`` fed to the inverse-Fourier integral.

        Default: ``m_i^0 - m_ref - (scale + kernel shift) - (D theta)_i``, i.e.
        LINEAR in the parameters.  A subclass may return any per-candidate
        function of them -- in particular a nonlinear transform to a
        truth-referenced variable -- provided it also supplies the matching
        ``_chunk_logjac``.
        """
        lo, hi = self._chunks[ci]
        shift = self._mass_shift(values)
        delta = self.mobs[lo:hi] if shift is None else self.mobs[lo:hi] - shift
        if self._jac_chunks is not None:
            theta = tf.stack([values[p] for p in self.jac_params])
            dj = tf.squeeze(
                tf.sparse.sparse_dense_matmul(self._jac_chunks[ci], theta[:, None]),
                axis=-1,
            )
            delta = delta - dj
        ms = self._chunk_mean_shift(values, ci)
        if ms is not None:
            delta = delta - ms
        return delta

    def _chunk_logjac(self, values, ci):
        """Optional ``log |d(residual)/d(observable)|`` of chunk ``ci``.

        A nonlinear ``_chunk_residual`` changes the measure, and the density
        the likelihood needs is ``p_x(x_i) |dx_i/dm_i|``.  ``None`` means the
        transform is the identity (unit Jacobian), which is the linear default.
        """
        return None

    def _mass_shift(self, values):
        """Predicted mass minus ``m_ref``: the scalar (candidate-independent) part."""
        shift = None
        if self.scale_param is not None:
            shift = values[self.scale_param] * self.npdt(self.scale_unit * self.m_ref)
        dm = self.kernel.mass_shift(values)
        if dm is not None:
            shift = dm if shift is None else shift + dm
        return shift

    def _upsample_exponent(self, s):
        """Expand a *stored-grid* ``(nrow, nt)`` exponent onto the integration
        grid ``(nrow, nt_int)``.

        The identity when ``upsample == 1``; the fixed cubic-spline matrix
        otherwise. Applied to the TABULATED exponents only -- everything
        analytic in ``tau`` (the Gaussian family, the physics kernel) is
        evaluated on the integration grid directly.
        """
        if s is None or self._upmat is None:
            return s
        return tf.matmul(s, self._upmat, transpose_b=True)

    def _family_exponent(self, values, families, vgf):
        """``(Re S, Im S)`` of one block of rows from a (pre-sliced) family list.

        ``S = sum_f k_f S_f`` over the per-family scale knobs. The tabulated
        families are summed on their *stored* grid and the sum is expanded once
        (the spline is linear, so that is exact); the analytic Gaussian family
        is evaluated on the integration grid directly.
        """
        dtype = self.dtype
        s_re = None
        s_im = None
        gauss = None
        for f in families:
            k = values[f["param"]]
            if f["kind"] == "gauss":
                gauss = k * (self.npdt(-0.5) * vgf[:, None] * self.tgrid[None, :] ** 2)
                continue
            if "re" in f:
                contrib = k * tf.cast(f["re"], dtype)
                s_re = contrib if s_re is None else s_re + contrib
            if "im" in f:
                contrib = k * tf.cast(f["im"], dtype)
                s_im = contrib if s_im is None else s_im + contrib
        s_re = self._upsample_exponent(s_re)
        s_im = self._upsample_exponent(s_im)
        if gauss is not None:
            s_re = gauss if s_re is None else s_re + gauss
        return s_re, s_im

    def _chunk_resolution(self, values, ci):
        """Resolution log-CF exponent ``(Re S, Im S)`` of chunk ``ci``.

        THE single place the PER-CANDIDATE family sum is built: subclasses
        override this and only this to change the PARAMETERISATION of the
        resolution (see :class:`MaterialCFTerm`); the quadrature, the kernel,
        the mass shift and the background mixture in :meth:`_chunk_li` are
        untouched. The truncation normalisation is deliberately NOT routed
        through here -- :meth:`_norm_z` builds its own sum over resolution
        *classes*, whose rows are not candidates and to which the
        per-candidate hooks therefore do not apply.

        The returned exponents are already on the *integration* grid.
        """
        lo, hi = self._chunks[ci]
        families = [
            dict(f, **{c: f[c][lo:hi] for c in ("re", "im") if c in f})
            for f in self.families
        ]
        return self._family_exponent(
            values, families, None if self.vgf is None else self.vgf[lo:hi]
        )

    def _density(
        self,
        values,
        sigma,
        mobs,
        families,
        vgf,
        phik_re,
        phik_im,
        jac_sp=None,
        delta=None,
        s_re=None,
        s_im=None,
        escale=None,
    ):
        """Density ``L(m)`` of one block of rows, before the positivity floor.

        The quadrature itself, shared by the per-candidate chunks
        (:meth:`_chunk_li`) and callable on any other set of rows -- a mass
        grid, a set of resolution classes -- that wants exactly the same model.

        ``delta``, ``s_re``/``s_im`` and ``escale`` are the *already computed*
        residual, resolution log-CF exponent and exponent multiplier, which is
        how :meth:`_chunk_li` feeds in the subclass hooks (the residual has to
        exist before the self-consistent ``sigma`` that is then passed in as
        the ``sigma`` argument). Left at ``None`` they are built here from
        ``mobs``/``jac_sp`` and ``families``/``vgf``, which is the plain path.

        ``phik_re``/``phik_im`` are the pre-interpolated per-row kernel CF; pass
        ``None`` to have the tabulation blended at the actual ``t_abs`` inside
        the graph (needed once ``sigma`` moves with the parameters, or once the
        integration grid is finer than the stored one).
        """
        t_abs = self.tgrid[None, :] / sigma[:, None]

        # resolution CF exponent: sum over families, S = sum_f k_f S_f
        if s_re is None and s_im is None and families is not None:
            s_re, s_im = self._family_exponent(values, families, vgf)

        # optional per-candidate multiplier of the WHOLE process-noise exponent
        if escale is not None:
            if s_re is not None:
                s_re = s_re * escale[:, None]
            if s_im is not None:
                s_im = s_im * escale[:, None]

        # physics-kernel CF (delta: nothing; Breit-Wigner: -Gamma |t| / 2)
        k_re, k_im = self.kernel.log_cf(values, t_abs)
        if k_re is not None:
            s_re = k_re if s_re is None else s_re + k_re
        if k_im is not None:
            s_im = k_im if s_im is None else s_im + k_im

        if delta is None:
            # predicted mass: m_ref alpha + dm_res + (D theta)_i
            shift = self._mass_shift(values)
            delta = mobs if shift is None else mobs - shift
            if jac_sp is not None:
                theta = tf.stack([values[p] for p in self.jac_params])
                dj = tf.squeeze(
                    tf.sparse.sparse_dense_matmul(jac_sp, theta[:, None]), axis=-1
                )
                delta = delta - dj

        # Re[phi_K e^S e^{-i t delta}] = e^{Sre} (Re phi_K cos psi - Im phi_K sin psi)
        # with psi = Sim - t delta; all-real arithmetic, no complex gradients.
        psi = -t_abs * delta[:, None]
        if s_im is not None:
            psi = psi + s_im
        if phik_re is None and self.phik_tab is not None:
            phik_re, phik_im = self._interp_phik(t_abs)
        if phik_re is not None:
            integ = phik_re * tf.cos(psi) - phik_im * tf.sin(psi)
        else:
            integ = tf.cos(psi)
        if s_re is not None:
            integ = tf.exp(s_re) * integ

        return tf.reduce_sum(
            self.dtgrid[None, :] * (integ[:, 1:] + integ[:, :-1]) * self.npdt(0.5),
            axis=1,
        ) / (self.npdt(np.pi) * sigma)

    def _interp_phik(self, t_abs):
        """Linear blend of the (uniform) kernel-CF tabulation at arbitrary
        absolute ``t``, differentiable in ``t``.

        ONE implementation serving both callers: the parameter-dependent sigma,
        which has moved ``t_abs`` away from the grid the ``(n, nt)`` array was
        built on, and the upsampled ``tau`` grid, for which that array is never
        built at all. Both need the same thing -- the tabulation read at the
        actual ``t_abs`` -- and both tabulations are regular
        (``build_phik_table`` uses ``np.linspace``), so the lookup is
        ``floor(t/dt)`` plus a weight: no retabulation per parameter point, and
        numerically the same numbers ``np.interp`` gives. Out of range is
        clamped to the last sample, where the kernel CF has long decayed.
        """
        t_tab = self.phik_tab[0]
        t0 = self.npdt(t_tab[0])
        dt = self.npdt((t_tab[-1] - t_tab[0]) / (len(t_tab) - 1))
        n = len(t_tab)
        u = (tf.clip_by_value(t_abs, t0, self.npdt(t_tab[-1])) - t0) / dt
        i0 = tf.clip_by_value(tf.cast(tf.floor(u), tf.int32), 0, n - 2)
        w = u - tf.cast(i0, self.dtype)
        out = []
        for tab in (self._pk_re, self._pk_im):
            a = tf.gather(tab, i0)
            b = tf.gather(tab, i0 + 1)
            out.append(a + (b - a) * w)
        return out[0], out[1]

    def _chunk_li(self, values, ci):
        """Per-candidate density ``L_i`` for chunk ``ci``, before the floor."""
        lo, hi = self._chunks[ci]
        # the residual FIRST: the self-consistent resolution is a function of it
        delta = self._chunk_residual(values, ci)
        sigma = self._chunk_sigma(values, ci, delta)
        dyn_sigma = sigma is not None
        if not dyn_sigma:
            sigma = self.sigma[lo:hi]
        s_re, s_im = self._chunk_resolution(values, ci)
        # sigma moved, so the kernel CF has to be read at the NEW absolute t;
        # passing None lets `_density` blend the tabulation in graph
        use_tab = dyn_sigma and self.phik_tab is not None
        return self._density(
            values,
            sigma,
            None,
            None,
            None,
            None if use_tab or self.phik_re is None else self.phik_re[lo:hi],
            None if use_tab or self.phik_im is None else self.phik_im[lo:hi],
            delta=delta,
            s_re=s_re,
            s_im=s_im,
            escale=self._chunk_exponent_scale(values, ci),
        )

    def _norm_z(self, values):
        """``Z_c = Int_lo^hi L(m; class c) dm`` for every resolution class.

        The truncated likelihood of a sample selected in ``norm_window`` is
        ``prod_i L_i(m_i) / Z_i``. ``Z_i`` is a very flat function of the
        candidate's resolution, so it is evaluated once per resolution *class*
        and gathered per candidate rather than integrated for each of ``n``.

        The integral is done in Fourier space (Gil-Pelaez), not by sampling
        the density on a mass grid::

            Z = (1/pi) Int_0^inf Im[ phi(u) (e^{-i u d_lo} - e^{-i u d_hi}) ]/u du

        with ``d_x = x - mu_c``. Both routes need a ``t`` grid fine enough to
        resolve oscillations at the *window half-width* rather than at the
        candidate's own pull -- for a Z that is ``|d|/sigma ~ 30/1 = 30``, i.e.
        ~35 periods across the term's own ``tgrid``, which the 64-point in-maker
        grid samples 1.8 times per period. Upsampling is therefore unavoidable;
        doing it in Fourier space costs a ``(K, nt_norm)`` tensor, while the
        mass-grid route costs ``(K, n_mass, nt_norm)``. Hence
        :attr:`norm_tpoints` is large (thousands) and cheap.

        The exponents are resampled onto that grid when the term is built (they
        are smooth in ``t``: the largest second difference is <2 % of the range,
        and a cubic spline through every other in-maker point reproduces them to
        ~1e-4 absolute).

        The integrand ``G(t)/t`` is evaluated on a *midpoint* grid, which avoids
        the removable singularity at ``t = 0`` entirely.
        """
        dtype = self.dtype
        lo, hi = self.norm_window
        t = self._norm_tgrid
        sigma = self._norm_sigma
        t_abs = t[None, :] / sigma[:, None]

        s_re = None
        s_im = None
        for f in self._norm_families:
            k = values[f["param"]]
            if f["kind"] == "gauss":
                contrib = k * (
                    self.npdt(-0.5) * self._norm_vgf[:, None] * t[None, :] ** 2
                )
                s_re = contrib if s_re is None else s_re + contrib
                continue
            if "re" in f:
                contrib = k * tf.cast(f["re"], dtype)
                s_re = contrib if s_re is None else s_re + contrib
            if "im" in f:
                contrib = k * tf.cast(f["im"], dtype)
                s_im = contrib if s_im is None else s_im + contrib

        k_re, k_im = self.kernel.log_cf(values, t_abs)
        if k_re is not None:
            s_re = k_re if s_re is None else s_re + k_re
        if k_im is not None:
            s_im = k_im if s_im is None else s_im + k_im

        shift = self._mass_shift(values)
        d_lo = tf.constant(lo - self.m_ref, dtype)
        d_hi = tf.constant(hi - self.m_ref, dtype)
        if shift is not None:
            d_lo = d_lo - shift
            d_hi = d_hi - shift

        def edge(d):
            psi = -t_abs * d
            if s_im is not None:
                psi = psi + s_im
            if self._norm_phik_re is not None:
                return self._norm_phik_re * tf.sin(psi) + self._norm_phik_im * tf.cos(
                    psi
                )
            return tf.sin(psi)

        g = edge(d_lo) - edge(d_hi)
        if s_re is not None:
            g = tf.exp(s_re) * g
        return tf.reduce_sum(g / t[None, :], axis=1) * self.npdt(
            self._norm_dt / np.pi
        )

    def _build_norm(self, norm, phik):
        """Tabulate the resolution classes of the truncation normalisation."""
        lo, hi = self.norm_window
        if self.norm_tpoints < 16:
            raise ValueError("norm_tpoints must be at least 16")
        sig_c = np.asarray(norm["sigma"], dtype=np.float64).ravel()
        self._nclass = len(sig_c)
        cls = np.asarray(norm["class"], dtype=np.int64).ravel()
        if cls.shape != (self.n,):
            raise ValueError(
                f"norm class index has shape {cls.shape}, expected {(self.n,)}"
            )
        if cls.min() < 0 or cls.max() >= self._nclass:
            raise ValueError("norm class index out of range")
        self._norm_class = tf.constant(cls, tf.int32)
        self._norm_sigma = tf.constant(sig_c, self.dtype)

        tmax = float(np.asarray(self.tgrid_stored)[-1])
        nt = self.norm_tpoints
        self._norm_dt = tmax / nt
        tmid = (np.arange(nt) + 0.5) * self._norm_dt
        self._norm_tgrid = tf.constant(tmid, self.dtype)

        vgf_c = norm.get("vgf")
        self._norm_vgf = (
            None
            if vgf_c is None
            else tf.constant(np.asarray(vgf_c, dtype=np.float64).ravel(), self.dtype)
        )

        # resample the family exponents from the term's tgrid onto tmid
        from scipy.interpolate import CubicSpline

        tsrc = np.asarray(self.tgrid_stored, dtype=np.float64)
        by_name = {f["name"]: f for f in norm.get("families", [])}
        self._norm_families = []
        for f in self.families:
            entry = {"name": f["name"], "param": f["param"], "kind": f["kind"]}
            if entry["kind"] != "gauss":
                src = by_name.get(f["name"])
                if src is None:
                    raise ValueError(f"norm block is missing family '{f['name']}'")
                for comp in ("re", "im"):
                    if comp in f:
                        arr = np.asarray(src[comp], dtype=np.float64)
                        if arr.shape[0] != self._nclass:
                            raise ValueError(
                                f"norm family '{f['name']}' component '{comp}' has "
                                f"shape {arr.shape}, expected "
                                f"({self._nclass}, {self.nt})"
                            )
                        if arr.shape[1] == nt:
                            up = arr
                        elif arr.shape[1] == self.nt:
                            up = CubicSpline(tsrc, arr, axis=1)(tmid)
                        else:
                            raise ValueError(
                                f"norm family '{f['name']}' component '{comp}' has "
                                f"{arr.shape[1]} t points, expected {self.nt} or {nt}"
                            )
                        entry[comp] = tf.constant(up, self.dtype)
            self._norm_families.append(entry)

        if phik is None and (self.phik_re is not None or self.phik_tab is not None):
            raise ValueError(
                "norm_window needs the kernel CF *tabulation* (phik=(t, re, im)); "
                "it cannot be rebuilt from the per-candidate phik_grid"
            )
        if phik is not None:
            t_tab, re_tab, im_tab = (np.asarray(a, dtype=np.float64) for a in phik)
            tgi = tmid[None, :] / sig_c[:, None]
            if tgi.max() > t_tab[-1] * (1 + 1e-9):
                raise ValueError(
                    f"the kernel CF is tabulated to t = {t_tab[-1]:.3f} but the "
                    f"truncation normalisation needs {tgi.max():.3f} 1/GeV"
                )
            self._norm_phik_re = tf.constant(np.interp(tgi, t_tab, re_tab), self.dtype)
            self._norm_phik_im = tf.constant(np.interp(tgi, t_tab, im_tab), self.dtype)
        else:
            self._norm_phik_re = None
            self._norm_phik_im = None
        self._norm = norm

    def _mix(self, values, li, ci):
        """Positivity floor + background mixture + ``-sum log`` for chunk ``ci``."""
        lo, hi = self._chunks[ci]
        dtype = self.dtype
        if self.floor == "clip":
            lp = tf.maximum(li, self.npdt(0.0))
        elif self.floor == "softplus":
            s = self.npdt(self.floor_scale)
            lp = s * tf.math.softplus(li / s)
        else:
            lp = li

        if self.bkg_frac_param is not None:
            fb = values[self.bkg_frac_param] * self.npdt(self.bkg_frac_unit)
        elif self.bkg_frac:
            fb = tf.constant(self.bkg_frac, dtype)
        else:
            fb = None

        if fb is None:
            total = lp
        else:
            bkg = self.background.pdf(values, self.mobs[lo:hi] + self.npdt(self.m_ref))
            total = (tf.constant(1.0, dtype) - fb) * lp + fb * bkg

        logl = tf.math.log(total)
        lj = self._chunk_logjac(values, ci)
        if lj is not None:
            logl = logl + lj
        if self.weights is not None:
            logl = self.weights[lo:hi] * logl
        return -tf.reduce_sum(logl)

    # -- public ------------------------------------------------------------
    def nll(self, params, full_nll=False):
        """Scalar NLL contribution, accumulated over candidate chunks.

        ``full_nll`` is accepted for interface symmetry with the binned and
        external terms but has no effect: an unbinned term is already the
        exact ``-sum log(density)``, with no dropped normalisation constant.
        """
        values = self._values(params)
        z = None if self._norm is None else self._norm_z(values)
        total = None
        for ci in range(self.nchunk):
            li = self._chunk_li(values, ci)
            if z is not None:
                lo, hi = self._chunks[ci]
                li = li / tf.gather(z, self._norm_class[lo:hi])
            v = self._mix(values, li, ci)
            total = v if total is None else total + v
        if total is None:
            return tf.constant(0.0, self.dtype)
        return total

    def raw_density(self, params):
        """Per-candidate ``L_i`` before the positivity floor (diagnostics)."""
        values = self._values(params)
        return tf.concat(
            [self._chunk_li(values, ci) for ci in range(self.nchunk)], axis=0
        )

    def config(self):
        """JSON-serialisable structural description (written to the datacard)."""
        return {
            "kind": self.kind,
            "channel": self.channel,
            "m_ref": self.m_ref,
            "scale_param": self.scale_param,
            "scale_unit": self.scale_unit,
            "bkg_frac_param": self.bkg_frac_param,
            "bkg_frac": self.bkg_frac,
            "bkg_frac_unit": self.bkg_frac_unit,
            "floor": self.floor,
            "floor_scale": self.floor_scale,
            "self_consistent_sigma": self.self_consistent_sigma,
            "sigma_floor": self.sigma_floor,
            "chunk": self.chunk,
            "norm_window": None if self.norm_window is None else list(self.norm_window),
            "norm_tpoints": self.norm_tpoints,
            "upsample": self.upsample,
            "families": [
                {"name": f["name"], "param": f["param"], "kind": f["kind"]}
                for f in self.families
            ],
            "kernel": self.kernel.config(),
            "background": self.background.config(),
            "jac_params": list(self.jac_params),
        }


class MaterialCFTerm(MassCFTerm):
    """Mass likelihood whose resolution is parameterised by the PHYSICAL
    material and hit-resolution parameters of the CVH fit, not by ad-hoc
    per-family scale knobs.

    Physics
    -------
    The CVH propagator applies the parmtype-15 material-group parameter
    ``k_g`` to the AMOUNT of material of every Geant4 step:

        mean loss                dE      = exp(k_g) (dE/dx)_0 ds
        MS covariance            errMS  *= exp(k_g)
        ionization variance      errI   *= exp(k_g)

    Every step-level log-CF exponent of the resolution model is linear in that
    amount at fixed composition (Moliere ``chi_c^2 ~ x``, Urban ``a_j ~ x``,
    radiative mean emissions ``~ x``, delta-recoil ``xi ~ x``).  With the fit's
    influence weights held fixed -- the two-step convention the in-fit CGF
    block already uses -- the resolution exponent is therefore EXACTLY

        S_f(tau; k) = S_f^fix(tau) + sum_g A(k_g) S_{f,g}(tau)

    with ``A(k) = exp(k)`` (``amount_mode="exp"``, the convention of the C++
    ``matStepFact``) or ``A(k) = 1 + k`` (``"linear"``, its first-order form),
    and ``S^fix`` the pruned groups whose weight is pinned to 1.  ``k = 0``
    reproduces the production's own exponents bit-for-bit.

    The Gaussian hit share is split the same way but per HIT CLASS,

        v_i(eps) = v_other,i + sum_c H(eps_c) v_{c,i} ,
        Re S    += -0.5 v_i tau^2 ,

    with ``H(eps) = 1 + eps`` (``hit_mode="linear"``) or ``exp(eps)``.
    ``v_{c,i}`` is the summed exported influence variance ``resinfvarv`` of the
    candidate's parmtype-8/9 blocks of class ``c``, in units of ``sigma_i^2``,
    and ``v_other`` the Gaussian remainder (beamspot / vertex constraint) that
    no parameter scales.

    Field and alignment parameters enter this term ONLY through the mean, via
    the sparse ``D`` rows of :class:`MassCFTerm` -- they move where the mass
    sits, not how wide it is.

    Storage
    -------
    The per-group exponents are block-sparse (a candidate touches ~20 of the
    42 groups).  ``grp_ptr`` is a CSR row pointer over candidates into a flat
    ``nnz`` axis, ``grp_id`` the group index of each row, and each family's
    ``re`` / ``im`` arrays are ``(nnz, nt)``.

    Parameters
    ----------
    group_params : list[str]
        One parameter name per material group, in group-index order.  MUST be
        the names the quadratic external term uses (``material_<group>`` from
        ``make_global_term.name_params``) so that a joint fit floats one set.
    group_families : list[dict]
        ``{"name", "re"?, "im"?, "fix_re"?, "fix_im"?}``; ``re``/``im`` are
        ``(nnz, nt)``, the optional ``fix_*`` are ``(n, nt)`` baselines.
    grp_ptr : ndarray (n+1,)
    grp_id : ndarray (nnz,)
    group_units : ndarray (ngroups,), optional
        ``k_g = value * group_units[g]``.  1 by default; set to the whitening
        scale when the card is whitened, so both terms see the same physical k.
    hit_params : list[str]
        One parameter name per hit class, in class-index order.
    hit_share : (hit_ptr, hit_cls, hit_v, vg_other), optional
    hit_units : ndarray (ncls,), optional
    amount_mode, hit_mode : {"exp", "linear"}
    """

    kind = "MaterialCF"

    def __init__(
        self,
        name,
        *args,
        group_params=(),
        group_families=(),
        grp_ptr=None,
        grp_id=None,
        group_units=None,
        hit_params=(),
        hit_share=None,
        hit_units=None,
        amount_mode="exp",
        hit_mode="linear",
        amount_clip=5.0,
        hit_clip=50.0,
        **kwargs,
    ):
        if amount_mode not in ("exp", "linear"):
            raise ValueError(f"amount_mode {amount_mode!r} is not exp/linear")
        if hit_mode not in ("exp", "linear"):
            raise ValueError(f"hit_mode {hit_mode!r} is not exp/linear")
        self.group_params = list(group_params)
        self.hit_params = list(hit_params)
        self.amount_mode = amount_mode
        self.hit_mode = hit_mode
        # FINITENESS GUARD, not a physics choice.  A trust-region step early in
        # a fit can throw k to O(100); exp(k) is then +inf, the exponent -inf,
        # its exp() 0, and the GRADIENT inf*0 = NaN -- which is not a diverging
        # fit but a dead one (rabbit's Cholesky of the Hessian fails and the
        # covariance is lost).  Clipping k has a well-defined subgradient (0
        # outside), so the minimizer sees a flat region and steps back.
        # e^5 = 148x the material of a group: nothing physical is near it, and
        # a fit that ends ON the clip is telling you something is wrong.
        self.amount_clip = float(amount_clip)
        self.hit_clip = float(hit_clip)
        # set BEFORE super().__init__, which calls _extra_param_names()
        super().__init__(name, *args, **kwargs)

        npdt = self.npdt
        self.group_units = (
            np.ones(len(self.group_params))
            if group_units is None
            else np.asarray(group_units, dtype=np.float64)
        )
        if self.group_units.shape != (len(self.group_params),):
            raise ValueError("group_units must have one entry per group parameter")
        self.hit_units = (
            np.ones(len(self.hit_params))
            if hit_units is None
            else np.asarray(hit_units, dtype=np.float64)
        )
        if self.hit_units.shape != (len(self.hit_params),):
            raise ValueError("hit_units must have one entry per hit parameter")
        self._gunits = tf.constant(self.group_units, self.dtype)
        self._hunits = tf.constant(self.hit_units, self.dtype)

        # ---- block-sparse per-group exponents ----------------------------
        self.g_ptr = None
        self.group_families = []
        if len(self.group_params):
            if grp_ptr is None or grp_id is None:
                raise ValueError("group_params given without grp_ptr / grp_id")
            g_ptr = np.asarray(grp_ptr, dtype=np.int64).ravel()
            if g_ptr.shape != (self.n + 1,):
                raise ValueError(
                    f"grp_ptr has shape {g_ptr.shape}, expected {(self.n + 1,)}"
                )
            g_id = np.asarray(grp_id, dtype=np.int64).ravel()
            nnz = int(g_ptr[-1])
            if len(g_id) != nnz:
                raise ValueError(
                    f"grp_id has {len(g_id)} rows but grp_ptr ends at {nnz}"
                )
            if nnz and (g_id.min() < 0 or g_id.max() >= len(self.group_params)):
                raise ValueError("grp_id out of range of group_params")
            self.g_ptr = g_ptr
            self._g_id = tf.constant(g_id, tf.int32)
            self._g_seg = tf.constant(
                np.repeat(np.arange(self.n, dtype=np.int64), np.diff(g_ptr)),
                tf.int32,
            )
            for f in group_families:
                entry = {"name": f["name"]}
                for comp in ("re", "im"):
                    arr = f.get(comp)
                    if arr is not None:
                        arr = np.asarray(arr)
                        if arr.shape != (nnz, self.nt):
                            raise ValueError(
                                f"group family '{f['name']}' component '{comp}' "
                                f"has shape {arr.shape}, expected "
                                f"{(nnz, self.nt)}"
                            )
                        entry[comp] = tf.constant(arr, tf.as_dtype(arr.dtype))
                    fx = f.get("fix_" + comp)
                    if fx is not None:
                        fx = np.asarray(fx)
                        if fx.shape != (self.n, self.nt):
                            raise ValueError(
                                f"group family '{f['name']}' baseline "
                                f"'fix_{comp}' has shape {fx.shape}, expected "
                                f"{(self.n, self.nt)}"
                            )
                        entry["fix_" + comp] = tf.constant(
                            fx, tf.as_dtype(fx.dtype)
                        )
                if len(entry) == 1:
                    raise ValueError(
                        f"group family '{f['name']}' has no re/im/fix component"
                    )
                self.group_families.append(entry)
        elif len(group_families):
            raise ValueError("group_families given without group_params")

        # ---- Gaussian hit share ------------------------------------------
        # `hit_share` is required whenever the candidate has ANY Gaussian
        # variance, even with no floating class: `vg_other` alone is the hit +
        # beamspot remainder, which in the flat MassCFTerm rides as the `gauss`
        # family and IS the dominant part of the mass CF.  Dropping it makes
        # the model far too narrow, the density underflows, and log(0) = -inf
        # takes the gradient and the Hessian with it.
        self.h_ptr = None
        self.vg_other = None
        if hit_share is not None:
            h_ptr, h_cls, h_v, vg_other = hit_share
            h_ptr = np.asarray(h_ptr, dtype=np.int64).ravel()
            if h_ptr.shape != (self.n + 1,):
                raise ValueError(
                    f"hit_ptr has shape {h_ptr.shape}, expected {(self.n + 1,)}"
                )
            h_cls = np.asarray(h_cls, dtype=np.int64).ravel()
            h_v = np.asarray(h_v, dtype=np.float64).ravel()
            if len(h_cls) != int(h_ptr[-1]) or len(h_v) != len(h_cls):
                raise ValueError("hit_cls / hit_v inconsistent with hit_ptr")
            if len(h_cls) and (h_cls.min() < 0 or h_cls.max() >= len(self.hit_params)):
                raise ValueError("hit_cls out of range of hit_params")
            self.h_ptr = h_ptr
            self._h_cls = tf.constant(h_cls, tf.int32)
            self._h_v = tf.constant(h_v, self.dtype)
            self._h_seg = tf.constant(
                np.repeat(np.arange(self.n, dtype=np.int64), np.diff(h_ptr)),
                tf.int32,
            )
            self.vg_other = tf.constant(
                np.asarray(vg_other, dtype=np.float64).ravel(), self.dtype
            )
        elif len(self.hit_params):
            raise ValueError("hit_params given without hit_share")
        del npdt

    # -- internals ---------------------------------------------------------
    def _extra_param_names(self):
        return list(self.group_params) + list(self.hit_params)

    def _amount(self, values):
        k = tf.stack([values[p] for p in self.group_params]) * self._gunits
        if self.amount_clip > 0.0:
            k = tf.clip_by_value(k, self.npdt(-self.amount_clip),
                                 self.npdt(self.amount_clip))
        if self.amount_mode == "exp":
            return tf.exp(k)
        return tf.maximum(tf.constant(1.0, self.dtype) + k, self.npdt(0.0))

    def _hitscale(self, values):
        e = tf.stack([values[p] for p in self.hit_params]) * self._hunits
        if self.hit_clip > 0.0:
            e = tf.clip_by_value(e, self.npdt(-self.hit_clip),
                                 self.npdt(self.hit_clip))
        if self.hit_mode == "exp":
            return tf.exp(e)
        return tf.maximum(tf.constant(1.0, self.dtype) + e, self.npdt(0.0))

    def _chunk_resolution(self, values, ci):
        # any LEGACY per-family knobs first (empty in the physical model)
        s_re, s_im = super()._chunk_resolution(values, ci)
        lo, hi = self._chunks[ci]
        dtype = self.dtype

        if self.g_ptr is not None and len(self.group_families):
            a, b = int(self.g_ptr[lo]), int(self.g_ptr[hi])
            w = self._amount(values)
            wrow = tf.gather(w, self._g_id[a:b])[:, None]
            seg = self._g_seg[a:b] - np.int32(lo)
            nseg = hi - lo
            for f in self.group_families:
                for comp, tgt in (("re", 0), ("im", 1)):
                    arr = f.get(comp)
                    fx = f.get("fix_" + comp)
                    contrib = None
                    if arr is not None:
                        contrib = tf.math.unsorted_segment_sum(
                            wrow * tf.cast(arr[a:b], dtype), seg, nseg
                        )
                    if fx is not None:
                        c2 = tf.cast(fx[lo:hi], dtype)
                        contrib = c2 if contrib is None else contrib + c2
                    if contrib is None:
                        continue
                    # the per-group exponents are stored on the term's own
                    # tgrid, so they go through the same expansion the flat
                    # families do (a no-op at upsample == 1, which keeps this
                    # bit-identical to the un-upsampled term)
                    contrib = self._upsample_exponent(contrib)
                    if tgt == 0:
                        s_re = contrib if s_re is None else s_re + contrib
                    else:
                        s_im = contrib if s_im is None else s_im + contrib

        if self.h_ptr is not None:
            v = self.vg_other[lo:hi]
            if len(self.hit_params):
                a, b = int(self.h_ptr[lo]), int(self.h_ptr[hi])
                hw = self._hitscale(values)
                v = v + tf.math.unsorted_segment_sum(
                    tf.gather(hw, self._h_cls[a:b]) * self._h_v[a:b],
                    self._h_seg[a:b] - np.int32(lo),
                    hi - lo,
                )
            contrib = self.npdt(-0.5) * v[:, None] * self.tgrid[None, :] ** 2
            s_re = contrib if s_re is None else s_re + contrib

        return s_re, s_im

    def config(self):
        cfg = super().config()
        cfg.update(
            {
                "group_params": list(self.group_params),
                "hit_params": list(self.hit_params),
                "amount_mode": self.amount_mode,
                "hit_mode": self.hit_mode,
                "amount_clip": self.amount_clip,
                "hit_clip": self.hit_clip,
                "group_families": [{"name": f["name"]} for f in self.group_families],
            }
        )
        return cfg


def declare_params(term, declarations, default=(0.0, np.nan, 0.0, 0)):
    """Fill a term's parameter declaration arrays from a name-keyed dict.

    ``declarations`` maps a parameter name to
    ``(starting value, Gaussian prior sigma, prior mean, is_poi)``; anything not
    mentioned takes ``default`` (start at 0, free, not a POI). The arrays are
    written back onto the term *in its own parameter order* and returned as the
    four kwargs of :meth:`rabbit.tensorwriter.TensorWriter.add_unbinned_term`::

        term = MassCFTerm("z", ..., kernel=TabulatedLineshapeKernel(provider=zls))
        decl = unbinned.declare_params(term, {
            **zls.param_declarations(gz_prior=2.3),   # m_Z, Gamma_Z as POIs
            "alpha": (0.0, np.nan, 0.0, 1),
        })
        writer.add_unbinned_term(term.name, term.config(), term.param_names,
                                 datasets, **decl)

    Unknown names are an error -- a typo in a POI name would otherwise silently
    leave the parameter free and unreported.
    """
    unknown = set(declarations) - set(term.param_names)
    if unknown:
        raise ValueError(
            f"unbinned term '{term.name}': declarations for {sorted(unknown)} "
            f"which are not among its parameters {list(term.param_names)}"
        )
    rows = [declarations.get(n, default) for n in term.param_names]
    out = {
        "param_defaults": np.array([float(r[0]) for r in rows]),
        "param_prior_sigmas": np.array([float(r[1]) for r in rows]),
        "param_prior_means": np.array([float(r[2]) for r in rows]),
        "param_is_poi": np.array([int(r[3]) for r in rows], dtype=np.int8),
    }
    term.param_defaults = out["param_defaults"]
    term.param_prior_sigmas = out["param_prior_sigmas"]
    term.param_prior_means = out["param_prior_means"]
    term.param_is_poi = out["param_is_poi"]
    return out


_TERM_KINDS = {"MassCF": MassCFTerm, "MaterialCF": MaterialCFTerm}


# ---------------------------------------------------------------------------
# HDF5 io
# ---------------------------------------------------------------------------
def _write_str_dataset(group, key, values):
    import h5py

    ds = group.create_dataset(
        key, [len(values)], dtype=h5py.special_dtype(vlen=str), compression="gzip"
    )
    ds[...] = [str(s) for s in values]


def write_unbinned_terms_group(parent, terms, maxChunkBytes=1024**2):
    """Serialize the raw unbinned-term dicts collected by the TensorWriter.

    Each entry is ``{"name", "config", "params", "param_defaults",
    "param_prior_sigmas", "param_prior_means", "param_is_poi", "datasets"}``
    where ``datasets`` maps a name to a numeric ndarray. Returns the number of
    raw array bytes written.
    """
    if not terms:
        return 0

    nbytes = 0
    group = parent.create_group("unbinned_terms")
    for term in terms:
        g = group.create_group(term["name"])
        _write_str_dataset(g, "config", [json.dumps(term["config"])])
        _write_str_dataset(g, "params", term["params"])
        for key in (
            "param_defaults",
            "param_prior_sigmas",
            "param_prior_means",
            "param_is_poi",
        ):
            nbytes += h5pyutils_write.writeFlatInChunks(
                np.asarray(term[key]), g, key, maxChunkBytes=maxChunkBytes
            )
        for key, val in term["datasets"].items():
            arr = np.asarray(val)
            nbytes += h5pyutils_write.writeFlatInChunks(
                arr, g, key, maxChunkBytes=maxChunkBytes
            )
    return nbytes


def read_unbinned_terms_from_h5(group, dtype=tf.float64):
    """Decode an HDF5 ``unbinned_terms`` group into ready-to-use term objects.

    Unlike ``external_terms`` (raw numpy dicts at load time, tf objects built
    by the fitter) the terms are constructed here: they own large constant
    tensors, and building them once at load avoids a second copy. The fitter
    only resolves their parameter names against the fit parameter vector, see
    :func:`build_tf_unbinned_terms`.
    """
    if group is None:
        return []

    terms = []
    for name, g in group.items():
        cfg = json.loads(_read_str(g["config"])[0])
        kind = cfg.pop("kind", "MassCF")
        if kind not in _TERM_KINDS:
            raise RuntimeError(f"unknown unbinned term kind '{kind}' for '{name}'")
        params = _read_str(g["params"])
        data = {
            k: np.asarray(maketensor(g[k]))
            for k in g.keys()
            if k not in ("config", "params")
        }

        families = []
        for fam in cfg.pop("families"):
            entry = dict(fam)
            if entry.get("kind", "tab") != "gauss":
                for comp in ("re", "im"):
                    key = f"S_{comp}_{fam['name']}"
                    if key in data:
                        entry[comp] = data.pop(key)
            families.append(entry)

        norm = None
        if "norm_sigma" in data:
            norm = {
                "sigma": data.pop("norm_sigma"),
                "class": data.pop("norm_class"),
                "vgf": data.pop("norm_vgf", None),
                "families": [],
            }
            for fam in families:
                if fam.get("kind", "tab") == "gauss":
                    continue
                entry = {"name": fam["name"]}
                for comp in ("re", "im"):
                    key = f"S_{comp}_{fam['name']}_norm"
                    if key in data:
                        entry[comp] = data.pop(key)
                norm["families"].append(entry)

        phik = None
        if "phik_t" in data:
            phik = (data.pop("phik_t"), data.pop("phik_re"), data.pop("phik_im"))
        phik_grid = None
        if "phik_grid_re" in data:
            phik_grid = (data.pop("phik_grid_re"), data.pop("phik_grid_im"))

        extra = {}
        if kind == "MaterialCF":
            gfam = []
            for fam in cfg.pop("group_families", []):
                entry = {"name": fam["name"]}
                for comp in ("re", "im"):
                    for pref, key in (("", f"Sg_{comp}_{fam['name']}"),
                                      ("fix_", f"Sgfix_{comp}_{fam['name']}")):
                        if key in data:
                            entry[pref + comp] = data.pop(key)
                gfam.append(entry)
            extra["group_families"] = gfam
            for k in ("grp_ptr", "grp_id", "group_units", "hit_units"):
                if k in data:
                    extra[k] = data.pop(k)
            if "hit_ptr" in data:
                extra["hit_share"] = (
                    data.pop("hit_ptr"),
                    data.pop("hit_cls"),
                    data.pop("hit_v"),
                    data.pop("vg_other"),
                )

        jac = None
        if "jac_indices" in data:
            jac = (
                data.pop("jac_indices"),
                data.pop("jac_values"),
                data.pop("jac_shape"),
            )

        term = _TERM_KINDS[kind](
            name,
            sigma=data.pop("sigma"),
            mobs=data.pop("mobs"),
            tgrid=data.pop("tgrid"),
            families=families,
            vgf=data.pop("vgf", None),
            weights=data.pop("weights", None),
            a_res=data.pop("a_res", None),
            phik=phik,
            phik_grid=phik_grid,
            norm=norm,
            kernel=_make_kernel(cfg.pop("kernel", None)),
            background=_make_background(cfg.pop("background", None)),
            jac=jac,
            param_defaults=data.pop("param_defaults"),
            param_prior_sigmas=data.pop("param_prior_sigmas"),
            param_prior_means=data.pop("param_prior_means"),
            param_is_poi=data.pop("param_is_poi"),
            dtype=dtype,
            **extra,
            **cfg,
        )
        if list(term.param_names) != list(params):
            raise RuntimeError(
                f"unbinned term '{name}': stored parameter list {params} does "
                f"not match the one implied by its configuration "
                f"{term.param_names}"
            )
        logger.info(
            f"Loaded unbinned term '{name}' (kind {kind}, channel "
            f"{term.channel}): {term.n} candidates, {term.nchunk} chunk(s) of "
            f"{term.chunk}, parameters {term.param_names}"
        )
        terms.append(term)
    return terms


def _read_str(dset):
    return [s.decode() if isinstance(s, bytes) else str(s) for s in dset[...]]


# ---------------------------------------------------------------------------
# fitter-side helpers (mirroring rabbit.external_likelihood)
# ---------------------------------------------------------------------------
def build_tf_unbinned_terms(terms, parms):
    """Resolve each term's parameter names against the fit parameter vector.

    Returns a list of ``{"name", "term", "indices"}`` dicts; ``indices`` is an
    int64 tensor of positions in ``x`` ordered like ``term.param_names``.
    """
    if not terms:
        return []

    parms_str = np.asarray(parms).astype(str)
    parms_idx = {name: i for i, name in enumerate(parms_str)}
    if len(parms_idx) != len(parms_str):
        raise RuntimeError(
            "Duplicate parameter names in fitter parameter list; "
            "unbinned term resolution requires unique names."
        )

    out = []
    for term in terms:
        indices = np.empty(len(term.param_names), dtype=np.int64)
        for i, p in enumerate(term.param_names):
            j = parms_idx.get(p, -1)
            if j < 0:
                raise RuntimeError(
                    f"Unbinned term '{term.name}' parameter '{p}' not found in "
                    "fit parameters. Declare it with the UnbinnedParams param "
                    "model (--paramModel UnbinnedParams) or another model."
                )
            indices[i] = j
        out.append(
            {
                "name": term.name,
                "term": term,
                "indices": tf.constant(indices, dtype=tf.int64),
            }
        )
    return out


def compute_unbinned_nll(terms, x, dtype, full_nll=False):
    """Sum of the unbinned terms' NLL contributions at ``x``, or ``None``."""
    if not terms:
        return None
    total = None
    for entry in terms:
        sub = tf.gather(x, entry["indices"])
        val = entry["term"].nll(sub, full_nll=full_nll)
        total = val if total is None else total + val
    return tf.cast(total, dtype)
