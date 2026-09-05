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
        norm_window=None,
        norm_tpoints=8192,
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
        self.tgrid = tf.constant(tgrid, dtype)
        # trapezoid weights: d = diff(t); trapz = sum(d * (y[1:] + y[:-1]) / 2)
        self.dtgrid = tf.constant(np.diff(tgrid), dtype)
        self.weights = (
            None
            if weights is None
            else tf.constant(np.asarray(weights, dtype=np.float64), dtype)
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
    def _mass_shift(self, values):
        """Predicted mass minus ``m_ref``: the scalar (candidate-independent) part."""
        shift = None
        if self.scale_param is not None:
            shift = values[self.scale_param] * self.npdt(self.scale_unit * self.m_ref)
        dm = self.kernel.mass_shift(values)
        if dm is not None:
            shift = dm if shift is None else shift + dm
        return shift

    def _density(
        self, values, sigma, mobs, families, vgf, phik_re, phik_im, jac_sp=None
    ):
        """Density ``L(m)`` of one block of rows, before the positivity floor.

        Shared by the per-candidate chunks (:meth:`_chunk_li`) and by the
        truncation normalisation (:meth:`_norm_z`), which evaluates exactly the
        same model on a mass grid.
        """
        dtype = self.dtype
        t_abs = self.tgrid[None, :] / sigma[:, None]

        # resolution CF exponent: sum over families, S = sum_f k_f S_f
        s_re = None
        s_im = None
        for f in families:
            k = values[f["param"]]
            if f["kind"] == "gauss":
                contrib = k * (
                    self.npdt(-0.5) * vgf[:, None] * self.tgrid[None, :] ** 2
                )
                s_re = contrib if s_re is None else s_re + contrib
                continue
            if "re" in f:
                contrib = k * tf.cast(f["re"], dtype)
                s_re = contrib if s_re is None else s_re + contrib
            if "im" in f:
                contrib = k * tf.cast(f["im"], dtype)
                s_im = contrib if s_im is None else s_im + contrib

        # physics-kernel CF (delta: nothing; Breit-Wigner: -Gamma |t| / 2)
        k_re, k_im = self.kernel.log_cf(values, t_abs)
        if k_re is not None:
            s_re = k_re if s_re is None else s_re + k_re
        if k_im is not None:
            s_im = k_im if s_im is None else s_im + k_im

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

    def _chunk_li(self, values, ci):
        """Per-candidate density ``L_i`` for chunk ``ci``, before the floor."""
        lo, hi = self._chunks[ci]
        families = [
            dict(f, **{c: f[c][lo:hi] for c in ("re", "im") if c in f})
            for f in self.families
        ]
        return self._density(
            values,
            self.sigma[lo:hi],
            self.mobs[lo:hi],
            families,
            None if self.vgf is None else self.vgf[lo:hi],
            None if self.phik_re is None else self.phik_re[lo:hi],
            None if self.phik_im is None else self.phik_im[lo:hi],
            None if self._jac_chunks is None else self._jac_chunks[ci],
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

        tmax = float(np.asarray(self.tgrid)[-1])
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

        tsrc = np.asarray(self.tgrid, dtype=np.float64)
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

        if phik is None and self.phik_re is not None:
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
            "chunk": self.chunk,
            "norm_window": None if self.norm_window is None else list(self.norm_window),
            "norm_tpoints": self.norm_tpoints,
            "families": [
                {"name": f["name"], "param": f["param"], "kind": f["kind"]}
                for f in self.families
            ],
            "kernel": self.kernel.config(),
            "background": self.background.config(),
            "jac_params": list(self.jac_params),
        }


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


_TERM_KINDS = {"MassCF": MassCFTerm}


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
