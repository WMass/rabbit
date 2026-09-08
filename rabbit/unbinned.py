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


# ---------------------------------------------------------------------------
# candidate chunking
# ---------------------------------------------------------------------------
class ChunkTable:
    """``term._chunks``: the ``(lo, hi)`` bounds of candidate chunk ``ci``.

    A plain list of python pairs would do for the eager loop the terms have
    always run (``for ci in range(self.nchunk)``). This exists because the
    same loop has to be expressible as a ``tf.while_loop`` on the device: the
    loop variable is then a *traced int32 tensor*, and every ``arr[lo:hi]``
    in the chunk path has to become a dynamic ``strided_slice`` rather than a
    static one.

    Indexing with a python int returns python ints and is bit-identical to
    the list it replaces; indexing with a tensor returns tensors. Nothing
    else in the term has to know which mode it is in -- that is the whole
    point of putting the branch here.

    The last chunk is short (``n`` is not in general a multiple of
    ``chunk``), so the traced slice has a dynamic length. That is legal
    everywhere in the chunk path -- only the ``nt`` axis needs a static size
    -- and it means the graph is traced once, for all chunks, instead of once
    per chunk size.
    """

    def __init__(self, chunk, n, nchunk):
        self.chunk = int(chunk)
        self.n = int(n)
        self.nchunk = int(nchunk)

    def __len__(self):
        return self.nchunk

    def __iter__(self):
        for i in range(self.nchunk):
            yield self[i]

    def __getitem__(self, ci):
        if tf.is_tensor(ci):
            lo = tf.cast(ci, tf.int32) * tf.constant(self.chunk, tf.int32)
            return lo, tf.minimum(lo + self.chunk, tf.constant(self.n, tf.int32))
        ci = int(ci)
        if ci < 0:
            ci += self.nchunk
        if not 0 <= ci < self.nchunk:
            raise IndexError(ci)
        lo = ci * self.chunk
        return lo, min(self.n, lo + self.chunk)

    def __repr__(self):
        return f"ChunkTable(chunk={self.chunk}, n={self.n}, nchunk={self.nchunk})"


class JacChunkTable:
    """``term._jac_chunks``: the sparse ``D`` rows of candidate chunk ``ci``.

    The eager path keeps the pre-built per-chunk :class:`tf.SparseTensor`
    blocks it always had -- slicing a 46-million-nonzero sparse tensor once
    per chunk would be a real cost there. A *traced* index cannot index a
    python list, so it slices the whole-sample sparse tensor instead, which is
    assembled lazily on the first such access and only then (a card with no
    ``jac`` never builds it, and neither does a fit that stays on the host
    loop).
    """

    def __init__(self, blocks, chunks, njac):
        self.blocks = list(blocks)
        self.chunks = chunks
        self.njac = int(njac)
        self._whole = None

    def __len__(self):
        return len(self.blocks)

    def __iter__(self):
        return iter(self.blocks)

    def _whole_sparse(self):
        if self._whole is None:
            idx, val = [], []
            for ci, b in enumerate(self.blocks):
                lo, _ = self.chunks[ci]
                bi = b.indices.numpy().copy()
                bi[:, 0] += lo
                idx.append(bi)
                val.append(b.values.numpy())
            self._whole = tf.sparse.SparseTensor(
                np.concatenate(idx, axis=0) if idx else np.zeros((0, 2), np.int64),
                tf.constant(
                    np.concatenate(val, axis=0) if val else np.zeros(0),
                    self.blocks[0].values.dtype if self.blocks else tf.float64,
                ),
                [self.chunks.n, self.njac],
            )
        return self._whole

    def __getitem__(self, ci):
        if not tf.is_tensor(ci):
            return self.blocks[int(ci)]
        lo, hi = self.chunks[ci]
        return tf.sparse.slice(
            self._whole_sparse(),
            tf.stack([tf.cast(lo, tf.int64), tf.constant(0, tf.int64)]),
            tf.stack(
                [tf.cast(hi - lo, tf.int64), tf.constant(self.njac, tf.int64)]
            ),
        )

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

    @property
    def graph_chunkable(self):
        """Can the candidate-chunk loop run as a ``tf.while_loop``?

        True when every per-chunk slice in the term goes through
        :class:`ChunkTable` (or :class:`JacChunkTable`) and so accepts a
        *traced* chunk index. A subclass that indexes a numpy pointer array
        with ``int(...)`` -- :class:`MaterialCFTerm`'s CSR group / hit blocks
        do -- must say so, and the caller then keeps that term on the host
        loop. See ``calibration_studies/fullscale/devobj.py``.
        """
        return True

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
    corr_coeff_max : float
        Bound on the fluctuation form's quadratic coefficient ``|c_i/sigma_i|``
        -- the expansion parameter itself. A per-candidate constant, so it is
        parameter-independent; see :meth:`_build_fluct`. 0 disables it.
    corr_form : {"residual", "fluctuation"}
        WHERE the two corrections act. ``"residual"`` is the historical form
        measured on the J/psi: the width is evaluated at ``delta_i(theta)`` and
        the Jensen map is inverted on it, both bounded by ``corr_clip``.
        ``"fluctuation"`` is the treatment: both are one deterministic map of
        the resolution fluctuation, applied INSIDE the convolution, with no
        clip and no log-Jacobian -- see :meth:`_build_fluct`. At a delta kernel
        the two agree by construction; at the Z only the second is defined.
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
        jensen_s2=None,
        jensen_mode="exact",
        jensen_scale=1.0,
        jensen_disc_floor=0.1,
        corr_clip=0.0,
        corr_form="residual",
        corr_coeff_max=0.08,
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
        self._chunks = ChunkTable(self.chunk, self.n, self.nchunk)

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

        # ---- the second-order (Jensen) correction ------------------------
        if jensen_mode not in ("off", "shift", "exact"):
            raise ValueError(
                f"jensen_mode must be 'off', 'shift' or 'exact', "
                f"got '{jensen_mode}'"
            )
        # THE DOMAIN OF THE TWO CORRECTIONS.
        #
        # Both the self-consistent resolution and the Jensen map are
        # expansions in the RESOLUTION fluctuation: `sigma_i = sigma_bar_i
        # (1 + a_i x_i)` and `m_hat/m - 1 = u + u^2 + s^2/2` are statements
        # about `x`, `u` of order `s = sigma/m`. `delta_i` is the argument they
        # are fed, and `delta_i` is the deviation from the REFERENCE MASS.
        #
        # For a resonance of negligible width in a narrow window those are the
        # same thing: at the J/psi, `|r| = |delta|/m <= 0.113` and
        # `|a delta|/sigma` stays under 0.1, which is where the spec's gates
        # were measured. For the Z they are NOT: the window is +-30 GeV on
        # 91.19, so `|r|` reaches 0.52 and the deviation out there is FSR and
        # the Breit-Wigner tail, not a resolution fluctuation. Fed the full
        # `delta`, the exact Jensen map moves the residual by a MEDIAN of
        # 57.7 MeV and by up to 10.7 GeV, against the 20.6 MeV mean shift it
        # exists to apply, and `a_i delta_i` reaches a full `sigma_i`.
        #
        # `corr_clip` is the argument's domain, in units of `sigma_i`: both
        # corrections see `clip(delta, +-corr_clip sigma)` instead of `delta`,
        # so inside a few sigma nothing changes and outside they SATURATE
        # rather than extrapolate. The Jensen map is continued linearly with
        # unit slope beyond the clip, so the residual stays monotone in the
        # observable and the log-Jacobian vanishes there (it is the identity
        # map out there, by construction). `0` disables the clip and
        # reproduces the behaviour these corrections had when they were
        # measured on the J/psi.
        self.corr_clip = float(corr_clip)
        if self.corr_clip < 0.0:
            raise ValueError("corr_clip must be >= 0 (it is in units of sigma)")
        self.jensen_mode = jensen_mode
        self.jensen_scale = float(jensen_scale)
        self.jensen_disc_floor = float(jensen_disc_floor)
        self._jensen_s2_np = None
        if jensen_s2 is not None:
            arr = np.asarray(jensen_s2, dtype=np.float64).ravel()
            if arr.shape != (self.n,):
                raise ValueError(
                    f"jensen_s2 has shape {arr.shape}, expected {(self.n,)}"
                )
            if np.any(arr < 0.0):
                raise ValueError("jensen_s2 must be non-negative (it is a variance)")
            self._jensen_s2_np = arr
            self.jensen_s2 = tf.constant(arr, dtype)
        else:
            self.jensen_s2 = None
        if corr_form not in ("residual", "fluctuation"):
            raise ValueError(
                f"corr_form must be 'residual' or 'fluctuation', got '{corr_form}'"
            )
        self.corr_form = corr_form
        self.corr_coeff_max = float(corr_coeff_max)
        self._fluct = corr_form == "fluctuation"
        if self._fluct and jensen_mode == "shift":
            raise ValueError(
                "jensen_mode='shift' has no meaning in the fluctuation form: "
                "the deterministic part of the map is s^2/2 (the EXACT form's), "
                "not the 1.5 s^2 the shift form assumes the MLE responds to "
                "with weight 1"
            )
        self._jensen = (
            not self._fluct
            and self.jensen_mode != "off"
            and self.jensen_s2 is not None
            and self.jensen_scale != 0.0
            and bool(np.any(self._jensen_s2_np != 0.0))
        )
        if self._fluct:
            # the residual form's two devices are OFF: the width stays the
            # exported constant and the residual map is the identity.  Both
            # effects are carried instead by `_build_fluct`, INSIDE the
            # convolution.
            self._dyn_sigma = False
        # the OBSERVED mass, the denominator of r = delta/m. Truth-free.
        self._jensen_m = tf.constant(mobs + float(m_ref), dtype)
        self._build_fluct(sigma, mobs, jensen_mode, tgrid)
        # `_chunk_residual` computes u and `_chunk_logjac` needs it; both are
        # called once per chunk, residual first, inside one graph.
        self._jensen_u = {}
        self._jensen_clipped = {}

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
            blocks = []
            for lo, hi in self._chunks:
                m = (idx[:, 0] >= lo) & (idx[:, 0] < hi)
                sub = idx[m].copy()
                sub[:, 0] -= lo
                order = np.lexsort((sub[:, 1], sub[:, 0]))
                blocks.append(
                    tf.sparse.SparseTensor(
                        sub[order],
                        tf.constant(val[m][order], dtype),
                        [hi - lo, len(self.jac_params)],
                    )
                )
            self._jac_chunks = JacChunkTable(
                blocks, self._chunks, len(self.jac_params)
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

        The second-order (Jensen) term of the mass functional,
        ``0.5 tr(H Sigma) = 1.5 s^2 m``, in the ``shift`` form: a deterministic
        location offset the MLE is ASSUMED to respond to with weight 1.  It
        does not (the measured response is 0.73 at J/psi resolution and 0.56 at
        Z-like), so this form over-corrects by 27-44 % and exists only for
        comparison.  ``jensen_mode="exact"`` is the default and does the work
        in :meth:`_chunk_residual` instead.  ``None`` means zero.
        """
        if self._fluct:
            # the DETERMINISTIC part d_i = m_i s_i^2/2 of the fluctuation map
            # (see `_build_fluct`); the x-dependent part is the CF factor
            if self._fl_d is None:
                return None
            lo, hi = self._chunks[ci]
            return self._fl_d[lo:hi]
        if not self._jensen or self.jensen_mode != "shift":
            return None
        lo, hi = self._chunks[ci]
        return (self.npdt(1.5 * self.jensen_scale)
                * self.jensen_s2[lo:hi] * self._jensen_m[lo:hi])

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
        return tf.maximum(sig - self.a_res[lo:hi] * self._corr_delta(delta, ci),
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
        return self._jensen_exact(delta, ci)

    def _corr_delta(self, delta, ci):
        """``delta`` restricted to the corrections' domain of validity."""
        if self.corr_clip <= 0.0:
            return delta
        lo, hi = self._chunks[ci]
        lim = self.npdt(self.corr_clip) * self.sigma[lo:hi]
        return tf.clip_by_value(delta, -lim, lim)

    def _build_fluct(self, sigma, mobs, jensen_mode, tgrid):
        """The FLUCTUATION form of the two corrections (``corr_form``).

        THE DEFECT OF THE RESIDUAL FORM.  Both corrections are statements about
        the RESOLUTION fluctuation, and the residual form feeds them
        ``delta_i(theta) = m_i - M(theta)``.  At a narrow resonance in a narrow
        window those coincide; at the Z they do not -- the window is +-27 sigma
        and what sits out there is the Breit-Wigner tail and FSR, not
        resolution.  Fed the full ``delta``, the exact Jensen map moves the
        residual by a median 57.7 MeV and by up to 10.7 GeV against the 20.6 MeV
        it exists to apply.  ``corr_clip`` bounds that but is a saturation
        device, not a treatment.

        THE TREATMENT.  Put both corrections where they belong: INSIDE the
        convolution, as a deterministic per-candidate map of the fluctuation.
        With ``x`` the standardized fluctuation (CF ``phi_i(tau) = e^{S_i(tau)}``,
        the exported exponents) the observed mass is

            m_i = m_true + u_i(x),   u_i(x) = sigma_i x + c_i x^2 + d_i,

        so, integrating over the fluctuation rather than evaluating at it,

            L_i(theta) = Int K_theta(m_i - u_i(x)) p_i(x) dx ,

        which needs no log-Jacobian, no clip, and no assumption that
        ``delta_i`` is small.  The measure is ``p_i(x) dx = p_x(x)(1 - a_i x)dx``:
        the ``(1 - a_i x)`` is the Jacobian of recovering the UNCONDITIONAL
        width ``sigma_bar_i`` from the exported ``sigma_i = sigma_bar_i(1+a_i x)``
        -- profiling out ``sigma_bar_i`` against the observed ``sigma_i`` -- and
        it is NOT optional: dropping it leaves a score of ``+a_i/sigma_bar_i`` at
        the truth, i.e. a bias of order ``a_i sigma_i`` (27 MeV at the Z), and
        it is the piece that makes this form reduce EXACTLY to the residual
        form's validated density when the kernel is a delta.

        The coefficients, both truth-free:

            c_i = -a_i sigma_i + sigma_i^2 / m_i ,     d_i = m_i s_i^2 / 2

        -- the first term of ``c_i`` is the self-consistent width (sec. 2-3 of
        MASSCFTERM_SPEC), the second and ``d_i`` are the Jensen map's ``u^2``
        and ``s^2/2`` (sec. 4b).  With ``a_i = (1 + vgf_i) sigma_i/m_i`` the two
        largely CANCEL, ``c_i = -vgf_i sigma_i^2/m_i``, which is why the naive Z
        bias is -15...-27 MeV and not the full -35.

        IN FOURIER SPACE.  ``d_i`` is a shift of the residual (handled by
        :meth:`_chunk_mean_shift`).  The rest is one multiplicative factor on
        the resolution CF, on the same ``tau`` grid the term already integrates:

            Phi_i(tau)/phi_i(tau) = 1 + i a_i (S'(tau) - S'(0))
                                      - i (c_i/sigma_i) tau (S''(tau) + S'(tau)^2)

        from ``E[x e^{i tau x}] = -i phi'`` and ``E[x^2 e^{i tau x}] = -phi''``
        with ``phi'' = (S'' + S'^2) phi``.  The ``- S'(0)`` normalises
        ``Phi_i(0) = 1`` (it is ``1 - a_i E[x]``, a candidate constant, so it
        cannot bias anything -- but it keeps the density normalised to 1 and so
        keeps the truncation ``Z`` consistent).  ``c_i/sigma_i = -a_i +
        sigma_i/m_i`` needs no division at evaluation time.

        WHAT IS EXACT AND WHAT IS APPROXIMATED.  The truncation is first order
        in ``a_i`` and in ``c_i``.  It is EXACT for the first moment at that
        order: for a centred unit-variance ``x`` the modelled mean is

            E[Delta_i] = c_i + d_i - a_i sigma_i          (exact in a, c)

        -- which is the whole content of both corrections -- with the neglected
        pieces ``O(a_i^2, a_i c_i, c_i^2) ~ 1e-4`` of a correction that is
        itself ~1e-2 of the width.  The second moment loses
        ``c_i^2 Var(x^2) ~ 2 (c_i/sigma_i)^2 ~ 2e-4`` relative, well under
        0.1 MeV on ``Gamma_Z``.  Both are measured directly against the
        residual form on the J/psi, where the two must agree.
        """
        self._fl_a = None
        self._fl_g = None
        self._fl_d = None
        self._fluct_active = False
        self._dmat = {}
        if not self._fluct:
            return
        n = self.n
        a = np.zeros(n)
        if self.a_res is not None and self.self_consistent_sigma:
            a = self._a_res_np.astype(np.float64)
        m = np.asarray(mobs, dtype=np.float64) + float(self.m_ref)
        mden = np.maximum(np.abs(m), 1e-9)
        # the same convention the residual form uses: an all-zero `jensen_s2`
        # switches the Jensen map off entirely, so the term is bit-identical to
        # one that was never given it
        jen = (
            jensen_mode != "off"
            and self._jensen_s2_np is not None
            and self.jensen_scale != 0.0
            and bool(np.any(self._jensen_s2_np != 0.0))
        )
        sig = np.asarray(sigma, dtype=np.float64)
        # c_i / sigma_i  (the linear scale is sigma_i itself)
        g = -a + (sig / mden if jen else 0.0)
        # THE DOMAIN OF THE EXPANSION, as a bound on the COEFFICIENT (not on
        # any argument -- that was `corr_clip`'s mistake).  The quadratic term
        # of the map contributes `g_i x^2` against the linear `x`, so `g_i` IS
        # the expansion parameter, and where `|g_i| |x| ~ 1` the first-order
        # truncation stops being a correction: the modelled density can go
        # NEGATIVE in the tail.  Measured on the 300 k Z card: with both
        # corrections on, `|g|` reaches 0.097 and NOT ONE of 300 000 densities
        # is non-positive; with the Jensen term switched off the cancellation
        # `g = -vgf sigma/m` is gone, `|g|` reaches 0.197, and 19 candidates
        # (all with sigma_m/m > 0.066) go negative and take the NLL to -inf.
        # `corr_coeff_max` bounds `|g_i|`; it is a per-candidate CONSTANT
        # computed from observables, so it is theta-independent and cannot
        # deform the likelihood's dependence on the parameters.  Scanned on the
        # 300 k Z card at five parameter points (reference, m_Z +-30 MeV,
        # Gamma_Z +-60 MeV): 0.10 leaves 2-3 non-positive densities in the
        # `nojensen` arm, 0.08 leaves NONE anywhere, at the cost of bounding
        # 364 of 300 000 candidates (0.12 %) in the physics configuration --
        # all of them with sigma_m/m > 0.066, i.e. a factor 40 less weight in
        # the mass than a typical candidate.
        if self.corr_coeff_max > 0.0:
            nlim = int(np.sum(np.abs(g) > self.corr_coeff_max))
            if nlim:
                logger.info(
                    f"unbinned term '{self.name}': {nlim} of {n} candidates "
                    f"({100.0 * nlim / max(n, 1):.4f} %) have |c_i/sigma_i| "
                    f"above corr_coeff_max = {self.corr_coeff_max:g}; the "
                    f"quadratic coefficient is bounded there"
                )
            g = np.clip(g, -self.corr_coeff_max, self.corr_coeff_max)
        d = (
            0.5 * self.jensen_scale * self._jensen_s2_np * m
            if jen
            else np.zeros(n)
        )
        self._fl_a = tf.constant(a, self.dtype)
        self._fl_g = tf.constant(g, self.dtype)
        self._fl_d = None if not np.any(d != 0.0) else tf.constant(d, self.dtype)
        self._fluct_active = bool(np.any(a != 0.0) or np.any(g != 0.0))
        if not self._fluct_active:
            return
        # the CF-derivative form uses S'(0); the grid has to reach tau = 0
        if abs(float(self.tgrid_stored[0])) > 1e-12:
            raise ValueError(
                "corr_form='fluctuation' needs a tau grid that starts at 0 "
                f"(it starts at {float(self.tgrid_stored[0]):g}): the CF "
                "normalisation is fixed at S'(0)"
            )
        from scipy.interpolate import CubicSpline

        tsrc = np.asarray(self.tgrid_stored, dtype=np.float64)
        tfine = np.asarray(tgrid, dtype=np.float64)   # the INTEGRATION grid
        sp = CubicSpline(tsrc, np.eye(len(tsrc)), axis=0)
        for k in (1, 2):
            self._dmat[k] = tf.constant(sp(tfine, k), self.dtype)

    def set_corrections(self, self_consistent_sigma=None, jensen_mode=None):
        """Switch either correction on or off AFTER construction.

        The variant ladder (both on / no `a_res` / no Jensen / neither) runs off
        ONE card, and in the fluctuation form the two corrections are baked into
        per-candidate constants at construction, so flipping the flags by hand
        is not enough -- they have to be rebuilt.  Returns ``self``.
        """
        if self_consistent_sigma is not None:
            self.self_consistent_sigma = bool(self_consistent_sigma)
            if self.self_consistent_sigma and self.a_res is None:
                raise ValueError(
                    "self_consistent_sigma requested but the term has no a_res"
                )
        if jensen_mode is not None:
            if jensen_mode not in ("off", "shift", "exact"):
                raise ValueError(
                    f"jensen_mode must be 'off', 'shift' or 'exact', "
                    f"got '{jensen_mode}'"
                )
            if self._fluct and jensen_mode == "shift":
                raise ValueError(
                    "jensen_mode='shift' has no meaning in the fluctuation form"
                )
            if jensen_mode != "off" and self.jensen_s2 is None:
                raise ValueError(
                    f"jensen_mode='{jensen_mode}' but the term has no jensen_s2"
                )
            self.jensen_mode = jensen_mode
        self._dyn_sigma = bool(
            not self._fluct
            and self.a_res is not None
            and self.self_consistent_sigma
            and np.any(self._a_res_np != 0.0)
        )
        self._jensen = bool(
            not self._fluct
            and self.jensen_mode != "off"
            and self.jensen_s2 is not None
            and self.jensen_scale != 0.0
            and np.any(self._jensen_s2_np != 0.0)
        )
        if self._fluct:
            self._build_fluct(
                self.sigma.numpy(), self.mobs.numpy(),
                self.jensen_mode, self.tgrid.numpy(),
            )
        return self

    def _fluct_w(self, values, ci):
        """``(w_re, w_im)`` of the fluctuation-form correction factor
        ``Phi_i/phi_i = 1 + w_i(tau)`` -- see :meth:`_build_fluct`."""
        lo, hi = self._chunks[ci]
        d1re, d1im, d2re, d2im = self._chunk_resolution_derivs(values, ci)
        p1re = d1re - d1re[:, :1]
        p1im = d1im - d1im[:, :1]
        dre = d2re + d1re * d1re - d1im * d1im
        dim = d2im + self.npdt(2.0) * d1re * d1im
        a = self._fl_a[lo:hi][:, None]
        g = self._fl_g[lo:hi][:, None]
        gt = g * self.tgrid[None, :]
        return (-a * p1im + gt * dim, a * p1re - gt * dre)

    def _jensen_exact(self, delta, ci):
        """Invert the second-order mass map; identity unless mode is 'exact'.

        ``m_hat/m - 1 = u + u^2 + s^2/2`` (uncorrelated equal legs,
        ``m ~ (k1 k2)^{-1/2}``), so with ``r = delta/m``

            u = 1/2 (sqrt(max(1 + 4(r - s^2/2), floor)) - 1)

        and the density picks up ``du/dr = 1/(1 + 2u)``, which
        :meth:`_chunk_logjac` supplies.  The discriminant floor only bites
        where ``r < -1/4``, i.e. a candidate more than a quarter of its own
        mass below the pole -- the far tail, where the second-order expansion
        has no meaning either way.
        """
        if not self._jensen or self.jensen_mode != "exact":
            self._jensen_u.pop(ci, None)
            self._jensen_clipped.pop(ci, None)
            return delta
        lo, hi = self._chunks[ci]
        m = self._jensen_m[lo:hi]
        s2 = self.npdt(self.jensen_scale) * self.jensen_s2[lo:hi]
        dc = self._corr_delta(delta, ci)
        r = dc / m
        disc = tf.maximum(
            self.npdt(1.0) + self.npdt(4.0) * (r - self.npdt(0.5) * s2),
            self.npdt(self.jensen_disc_floor),
        )
        u = self.npdt(0.5) * (tf.sqrt(disc) - self.npdt(1.0))
        self._jensen_u[ci] = u
        self._jensen_clipped[ci] = (
            None if self.corr_clip <= 0.0
            else tf.abs(delta - dc) > self.npdt(0.0))
        # inside the clip this IS `u m`; outside, the map is continued with
        # unit slope from the boundary, so the correction saturates at the
        # value it had there and the transform stays monotone
        return u * m + (delta - dc)

    def _chunk_logjac(self, values, ci):
        """Optional ``log |d(residual)/d(observable)|`` of chunk ``ci``.

        A nonlinear ``_chunk_residual`` changes the measure, and the density
        the likelihood needs is ``p_x(x_i) |dx_i/dm_i|``.  ``None`` means the
        transform is the identity (unit Jacobian), which is the linear default.

        For the exact Jensen map that is ``-log(1 + 2u)``.  It must not be
        dropped: without it the transform is a rescaling, not a
        reparameterisation, and the correction is wrong at its own order.
        """
        if not self._jensen or self.jensen_mode != "exact":
            return None
        u = self._jensen_u.get(ci)
        if u is None:
            raise RuntimeError(
                "_chunk_logjac was called before _chunk_residual for chunk "
                f"{ci}; the Jensen Jacobian has nothing to report"
            )
        lj = -tf.math.log(self.npdt(1.0) + self.npdt(2.0) * u)
        clipped = self._jensen_clipped.get(ci)
        if clipped is not None:
            # unit slope beyond the clip -> no change of measure there
            lj = tf.where(clipped, tf.zeros_like(lj), lj)
        return lj

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

    def _family_parts(self, values, families, vgf):
        """``(tabulated Re S, tabulated Im S, Gaussian variance)`` of one block.

        The exponent is split into the two pieces that behave differently under
        differentiation in ``tau``: the TABULATED families, summed on their
        *stored* grid (a cubic spline maps them, and their tau-derivatives, onto
        the integration grid), and the analytic Gaussian family, returned as its
        per-row variance ``v`` -- the exponent is ``-v tau^2/2``, its first
        derivative ``-v tau`` and its second ``-v``, all exact.

        Splitting it here is what lets :meth:`_chunk_resolution` and
        :meth:`_chunk_resolution_derivs` be built from ONE traversal of the
        family list, so a subclass that changes the parameterisation
        (:class:`MaterialCFTerm`) overrides one method and gets both.
        """
        dtype = self.dtype
        s_re = None
        s_im = None
        gv = None
        for f in families:
            k = values[f["param"]]
            if f["kind"] == "gauss":
                contrib = k * vgf
                gv = contrib if gv is None else gv + contrib
                continue
            if "re" in f:
                contrib = k * tf.cast(f["re"], dtype)
                s_re = contrib if s_re is None else s_re + contrib
            if "im" in f:
                contrib = k * tf.cast(f["im"], dtype)
                s_im = contrib if s_im is None else s_im + contrib
        return s_re, s_im, gv

    def _assemble_exponent(self, s_re, s_im, gv):
        """Stored-grid tabulated parts + Gaussian variance -> ``(Re S, Im S)``
        on the INTEGRATION grid.

        ``S = sum_f k_f S_f`` over the per-family scale knobs. The tabulated sum
        is expanded once (the spline is linear, so that is exact); the Gaussian
        family is evaluated on the integration grid directly.
        """
        s_re = self._upsample_exponent(s_re)
        s_im = self._upsample_exponent(s_im)
        if gv is not None:
            gauss = self.npdt(-0.5) * gv[:, None] * self.tgrid[None, :] ** 2
            s_re = gauss if s_re is None else s_re + gauss
        return s_re, s_im

    def _family_exponent(self, values, families, vgf):
        """``(Re S, Im S)`` of one block of rows from a (pre-sliced) family list."""
        return self._assemble_exponent(*self._family_parts(values, families, vgf))

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
        return self._assemble_exponent(*self._chunk_resolution_parts(values, ci))

    def _chunk_resolution_parts(self, values, ci):
        """The chunk's exponent in the split form of :meth:`_family_parts`.

        THE method a subclass overrides to change the PARAMETERISATION of the
        resolution: both the exponent (:meth:`_chunk_resolution`) and its
        tau-derivatives (:meth:`_chunk_resolution_derivs`) are assembled from
        what this returns, so the two can never drift apart.
        """
        lo, hi = self._chunks[ci]
        families = [
            dict(f, **{c: f[c][lo:hi] for c in ("re", "im") if c in f})
            for f in self.families
        ]
        return self._family_parts(
            values, families, None if self.vgf is None else self.vgf[lo:hi]
        )

    def _deriv_exponent(self, s, order):
        """``d^order/dtau^order`` of a stored-grid ``(nrow, nt)`` exponent,
        evaluated on the INTEGRATION grid.

        A fixed cubic-spline differentiation matrix, built once in
        :meth:`_build_fluct`. The exponents are smooth in ``tau`` (the largest
        second difference is a couple of per cent of the range), which is the
        same property the ``upsample`` expansion already relies on.
        """
        if s is None:
            return None
        return tf.matmul(s, self._dmat[order], transpose_b=True)

    def _chunk_resolution_derivs(self, values, ci):
        """``(S'_re, S'_im, S''_re, S''_im)`` of the chunk on the integration
        grid, differentiated with respect to the standardized ``tau``.

        Only the RESOLUTION exponent: the physics kernel is a separate factor of
        the integrand and is not part of the fluctuation whose map the
        correction inverts.
        """
        s_re, s_im, gv = self._chunk_resolution_parts(values, ci)
        d1re = self._deriv_exponent(s_re, 1)
        d1im = self._deriv_exponent(s_im, 1)
        d2re = self._deriv_exponent(s_re, 2)
        d2im = self._deriv_exponent(s_im, 2)
        if gv is not None:
            g1 = -gv[:, None] * self.tgrid[None, :]
            g2 = -gv[:, None] * tf.ones_like(self.tgrid)[None, :]
            d1re = g1 if d1re is None else d1re + g1
            d2re = g2 if d2re is None else d2re + g2
        z = None
        if d1re is None or d1im is None or d2re is None or d2im is None:
            z = tf.zeros(
                [self._chunks[ci][1] - self._chunks[ci][0], self.nt_int], self.dtype
            )
        return (d1re if d1re is not None else z,
                d1im if d1im is not None else z,
                d2re if d2re is not None else z,
                d2im if d2im is not None else z)

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
        corr_w=None,
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

        ``corr_w`` is the fluctuation-form correction factor ``1 + w(tau)``
        (given as ``(Re w, Im w)``), which multiplies the RESOLUTION CF -- see
        :meth:`_build_fluct`.  ``None`` leaves the integrand bit-for-bit as it
        was before that form existed.
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
        if corr_w is None:
            if phik_re is not None:
                integ = phik_re * tf.cos(psi) - phik_im * tf.sin(psi)
            else:
                integ = tf.cos(psi)
        else:
            # Re[phi_K e^S (1 + w) e^{-i t delta}] = (1 + w_re) P - w_im Q with
            # P = Re[phi_K e^{i psi}] the uncorrected integrand and
            # Q = Im[phi_K e^{i psi}] its quadrature partner.
            w_re, w_im = corr_w
            cpsi, spsi = tf.cos(psi), tf.sin(psi)
            if phik_re is not None:
                p = phik_re * cpsi - phik_im * spsi
                q = phik_re * spsi + phik_im * cpsi
            else:
                p, q = cpsi, spsi
            integ = p + (w_re * p - w_im * q)
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
            corr_w=self._fluct_w(values, ci) if self._fluct_active else None,
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
            "jensen_mode": self.jensen_mode,
            "jensen_scale": self.jensen_scale,
            "jensen_disc_floor": self.jensen_disc_floor,
            "corr_clip": self.corr_clip,
            "corr_form": self.corr_form,
            "corr_coeff_max": self.corr_coeff_max,
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

    @property
    def graph_chunkable(self):
        # the CSR group / hit blocks are indexed as `int(self.g_ptr[lo])`,
        # i.e. a python int out of a numpy pointer array: a traced chunk index
        # cannot do that, so a term that carries them stays on the host loop.
        return self.g_ptr is None and self.h_ptr is None

    def _chunk_resolution_parts(self, values, ci):
        # any LEGACY per-family knobs first (empty in the physical model)
        s_re, s_im, gv = super()._chunk_resolution_parts(values, ci)
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
                    # tgrid, so they join the flat tabulated families there and
                    # go through the same expansion (a no-op at upsample == 1,
                    # which keeps this bit-identical to the un-upsampled term)
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
            gv = v if gv is None else gv + v

        return s_re, s_im, gv

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
            jensen_s2=data.pop("jensen_s2", None),
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
