"""Helpers for external likelihood terms (linear + quadratic parameter priors).

An "external likelihood term" is an additive contribution to the NLL. For a
Gaussian prior ``N(mu, C)`` on a subset of the fit parameters it is

    -log L_ext = 0.5 (x_sub - mu)^T H (x_sub - mu)  +  lognorm      (H = C^-1)

which this module stores in the expanded form

    -log L_ext = g^T x_sub + 0.5 x_sub^T H x_sub  +  const  +  lognorm

with ``g = -H mu``. The two additive scalars are what make that expansion a
proper Gaussian and are the reason they exist as separate stored quantities:

``const`` = ``0.5 mu^T H mu`` = ``0.5 g^T H^-1 g``
    Part of the *quadratic*, not a normalization: it is what makes the term
    vanish at ``x_sub = mu``. Without it the term reads ``-0.5 mu^T H mu`` at
    the prior mean, so any NLL containing an external term is offset by a
    card-dependent amount and cannot be compared against a card without one.
    Included in both the reduced and the full NLL, mirroring
    ``Fitter._compute_lc``, which writes its own constraint term in centered
    form ``cw * 0.5 * (x - x0)^2`` and therefore never loses this piece.

``lognorm`` = ``0.5 (k log(2 pi) - log det H)``
    The Gaussian normalization, i.e. ``0.5 log det(2 pi C)``. Depends only on
    the prior width, never on ``x``. Included in the *full* NLL only, again
    mirroring ``_compute_lc``, whose ``full_nll`` branch adds
    ``0.9189 + log(sigma)`` per constrained parameter -- exactly this formula
    in one dimension. Only defined for positive-definite ``H``; a term whose
    ``H`` is not positive definite is not a probability density and gets
    ``lognorm = 0`` with a warning.

Neither scalar depends on ``x``, so neither can change a fit result: the
minimum, gradient, Hessian, covariance, EDM and any NLL *difference* taken
within a single fit are all unaffected. They change reported absolute NLL
values (``nllvalreduced``, ``nllvalfull``, and the saturated chi2 built from
``2 * nllvalreduced``), which is precisely where the omission was visible.

Both scalars are derived automatically when the Hessian is dense. For a
sparse Hessian the ``0.5 g^T H^-1 g`` solve and the ``log det`` are not
cheap, so they must be supplied at write time instead -- see
``TensorWriter.add_external_likelihood_term``, whose ``mean=`` argument
computes ``const`` with a single matvec (``0.5 mu^T H mu`` needs no solve
when ``mu`` is known) and is the recommended way to declare a Gaussian
prior. Terms stored without them keep the historical uncorrected behaviour.

Both the linear (``grad``) and quadratic (``hess_dense`` / ``hess_sparse``)
parts are optional; the sparse Hessian is stored as a
``tf.sparse.SparseTensor`` whose indices are in canonical row-major order.

This module centralizes three things that were previously inlined in
``Fitter.__init__``, ``Fitter._compute_external_nll``, and
``FitInputData.__init__``:

* :func:`read_external_terms_from_h5` — load the raw numpy-level
  per-term dicts from an HDF5 group (used by FitInputData)
* :func:`build_tf_external_terms` — turn that list into tf-side per-term
  dicts (resolved parameter indices, tf.constant grads, CSRSparseMatrix
  Hessians, and the two scalars above). Used by the Fitter when it takes
  ownership of the input data.
* :func:`compute_external_nll` — evaluate the scalar NLL contribution
  of a list of tf-side terms at the current ``x``.
"""

import logging

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops as tf_sparse_csr

from rabbit.h5pyutils_read import makesparsetensor, maketensor

logger = logging.getLogger(__name__)


def gaussian_scalars(grad, hess, name="<unnamed>"):
    """Derive ``(const, lognorm)`` for a dense Gaussian external term.

    Parameters
    ----------
    grad : ndarray or None
        The linear part ``g``. ``None`` (or all-zero) means the quadratic is
        already centered at the origin and ``const`` is 0.
    hess : ndarray or None
        The dense quadratic part ``H``. ``None`` means there is no quadratic
        and hence neither scalar is defined.
    name : str
        Term name, used in warnings.

    Returns
    -------
    (float, float)
        ``const = 0.5 g^T H^-1 g`` and ``lognorm = 0.5 (k log 2pi - log det H)``.

    Notes
    -----
    ``const`` is computed as ``0.5 g^T H^-1 g`` rather than by forming
    ``mu = -H^-1 g`` first; the two are algebraically identical and this
    avoids a second solve.

    Degenerate cases are handled explicitly rather than raising, because the
    ``g^T x + 0.5 x^T H x`` form is general enough to express things that are
    not priors at all:

    * ``H`` absent or zero -- a pure linear tilt. It has no minimum, so there
      is no constant that centers it and none is added.
    * ``g`` absent or zero -- already centered; ``const`` is 0 exactly.
    * ``H`` singular -- a Gaussian on a subspace, flat elsewhere. The
      pseudo-inverse gives the correct constant for the constrained subspace.
      If ``g`` additionally has a component outside ``range(H)`` the term is
      unbounded below and *no* constant centers it; that is warned about.
    * ``H`` not positive definite -- a saddle rather than a density. The
      centering constant is still well defined, but the normalization is not,
      so ``lognorm`` is 0 with a warning.
    """
    if hess is None:
        return 0.0, 0.0

    H = np.asarray(hess, dtype=np.float64)
    k = H.shape[0]
    g = None if grad is None else np.asarray(grad, dtype=np.float64).ravel()

    # --- const -------------------------------------------------------------
    const = 0.0
    if g is not None and np.any(g != 0.0):
        # Symmetrize defensively: the loss uses 0.5 x^T H x, which only ever
        # sees the symmetric part, so the constant must be built from the same.
        Hs = 0.5 * (H + H.T)
        eigvals = np.linalg.eigvalsh(Hs)
        tol = max(k, 1) * np.finfo(np.float64).eps * max(abs(eigvals).max(), 1.0)
        if abs(eigvals).min() > tol:
            y = np.linalg.solve(Hs, g)
        else:
            y = np.linalg.pinv(Hs, rcond=tol) @ g
            residual = Hs @ y - g
            if np.linalg.norm(residual) > 1e-8 * max(np.linalg.norm(g), 1.0):
                logger.warning(
                    f"External likelihood term '{name}': the Hessian is singular and "
                    "the gradient has a component outside its range, so the term is "
                    "unbounded below and cannot be centered. Using the pseudo-inverse "
                    "for the in-range part; absolute NLL values remain offset."
                )
        const = float(0.5 * g @ y)

    # --- lognorm -----------------------------------------------------------
    sign, logdet = np.linalg.slogdet(0.5 * (H + H.T))
    if sign <= 0:
        logger.warning(
            f"External likelihood term '{name}': the Hessian is not positive "
            "definite, so the term is not a normalizable Gaussian. Skipping its "
            "log-normalization; the full NLL remains offset for this term."
        )
        lognorm = 0.0
    else:
        lognorm = float(0.5 * (k * np.log(2.0 * np.pi) - logdet))

    return const, lognorm


def read_external_terms_from_h5(ext_group):
    """Decode an HDF5 ``external_terms`` group into a list of raw dicts.

    Each entry has the keys used by the rest of the pipeline:

    * ``name``: term label (str, taken from the h5 subgroup name)
    * ``params``: 1D ndarray of parameter name strings
    * ``grad_values``: 1D float ndarray or ``None``
    * ``hess_dense``: 2D float ndarray or ``None``
    * ``hess_sparse``: :class:`tf.sparse.SparseTensor` or ``None`` (uses
      the same on-disk layout as ``hlogk_sparse`` / ``hnorm_sparse``)
    * ``const``, ``lognorm``: float or ``None`` -- the Gaussian scalars
      described in the module docstring. Optional on disk: absent means
      "not supplied", which for a dense Hessian is filled in by
      :func:`build_tf_external_terms` and for a sparse one leaves the term
      uncorrected (historical behaviour). Note that ``None`` and a stored
      ``0.0`` are deliberately distinguished, so a writer that has
      determined a scalar really is zero can say so.

    Parameters
    ----------
    ext_group : h5py.Group
        The ``external_terms`` group in the input HDF5 file, or ``None``.

    Returns
    -------
    list[dict]
        One entry per stored external term, or an empty list if
        ``ext_group`` is ``None``.
    """
    if ext_group is None:
        return []

    terms = []
    for tname, tg in ext_group.items():
        raw_params = tg["params"][...]
        params = np.array(
            [s.decode() if isinstance(s, bytes) else s for s in raw_params]
        )
        grad_values = (
            np.asarray(maketensor(tg["grad_values"]))
            if "grad_values" in tg.keys()
            else None
        )
        hess_dense = (
            np.asarray(maketensor(tg["hess_dense"]))
            if "hess_dense" in tg.keys()
            else None
        )
        hess_sparse = (
            makesparsetensor(tg["hess_sparse"]) if "hess_sparse" in tg.keys() else None
        )
        terms.append(
            {
                "name": tname,
                "params": params,
                "grad_values": grad_values,
                "hess_dense": hess_dense,
                "hess_sparse": hess_sparse,
                "const": (float(tg["const"][()]) if "const" in tg.keys() else None),
                "lognorm": (
                    float(tg["lognorm"][()]) if "lognorm" in tg.keys() else None
                ),
            }
        )
    return terms


def sparse_hessian_terms(terms):
    """Names of the raw terms whose Hessian is stored sparse.

    :func:`build_tf_external_terms` makes each such Hessian a
    ``CSRSparseMatrix``, which :func:`compute_external_nll` contracts with
    ``SparseMatrixMatMul``. That op has no XLA kernel, so a ``tf.function``
    evaluating one of these terms cannot be built with ``jit_compile=True``,
    whether the tensor is dense or sparse. The Fitter resolves
    ``--jitCompile`` against this list.

    Parameters
    ----------
    terms : iterable of dict
        Raw per-term dicts as returned by :func:`read_external_terms_from_h5`.

    Returns
    -------
    list[str]
    """
    return [
        str(t["name"])
        for t in terms
        if t["hess_dense"] is None and t["hess_sparse"] is not None
    ]


def build_tf_external_terms(terms, parms, dtype):
    """Turn raw external-term dicts into tf-side dicts ready for the fitter.

    * Parameter names are resolved against the full fit parameter list
      ``parms`` via a single ``name->index`` dict (O(n) rather than the
      naive O(n^2) per-parameter ``np.where`` that this replaces — the
      latter cost ~150 s on a 108k-parameter setup with a 108k-parameter
      external term).
    * Gradients are promoted to ``tf.constant`` in the fitter dtype.
    * Dense Hessians are promoted to ``tf.constant``.
    * Sparse Hessians are promoted to a :class:`CSRSparseMatrix` view
      for fast ``sm.matmul``.

    Parameters
    ----------
    terms : list[dict]
        Raw per-term dicts as returned by :func:`read_external_terms_from_h5`.
    parms : array-like of str
        Full ordered list of fit parameter names (POIs + systematics).
    dtype : tf.DType
        Fitter dtype for gradient / Hessian tensors.

    * The Gaussian scalars ``const`` / ``lognorm`` (see the module
      docstring) are filled in here when the term did not carry them and
      the Hessian is dense. A sparse Hessian that arrives without them is
      left uncorrected, with a warning: deriving ``const`` would need a
      sparse solve and ``lognorm`` a sparse log-determinant, neither of
      which belongs in a load path. Supply them at write time instead --
      ``TensorWriter.add_external_likelihood_term(hess=..., mean=...)``
      gets ``const`` from a single matvec.

    Returns
    -------
    list[dict]
        One entry per term with keys ``name``, ``indices``, ``grad``,
        ``hess_dense``, ``hess_csr``, ``const``, ``lognorm``. Empty if
        ``terms`` is empty.
    """
    parms_str = np.asarray(parms).astype(str)
    parms_idx = {name: i for i, name in enumerate(parms_str)}
    if len(parms_idx) != len(parms_str):
        raise RuntimeError(
            "Duplicate parameter names in fitter parameter list; "
            "external term resolution requires unique names."
        )

    out = []
    for term in terms:
        params = np.asarray(term["params"]).astype(str)
        indices = np.empty(len(params), dtype=np.int64)
        for i, p in enumerate(params):
            j = parms_idx.get(p, -1)
            if j < 0:
                raise RuntimeError(
                    f"External likelihood term '{term['name']}' parameter "
                    f"'{p}' not found in fit parameters"
                )
            indices[i] = j
        tf_indices = tf.constant(indices, dtype=tf.int64)

        tf_grad = (
            tf.constant(term["grad_values"], dtype=dtype)
            if term["grad_values"] is not None
            else None
        )

        tf_hess_dense = None
        tf_hess_csr = None
        if term["hess_dense"] is not None:
            tf_hess_dense = tf.constant(term["hess_dense"], dtype=dtype)
        elif term["hess_sparse"] is not None:
            # Build a CSRSparseMatrix view of the stored sparse Hessian
            # for use in the closed-form external gradient/HVP path via
            # sm.matmul. The Hessian is assumed symmetric, so the loss
            # L = 0.5 x_sub^T H x_sub has gradient H @ x_sub and HVP
            # H @ p_sub, each a single sm.matmul call. NOTE:
            # SparseMatrixMatMul has no XLA kernel, so any tf.function
            # that calls sm.matmul must be built with jit_compile=False
            # (the Fitter checks sparse_hessian_terms for this).
            # The TensorWriter sorts the indices into canonical row-major
            # order at write time, so we can feed the SparseTensor
            # straight to the CSR builder without an additional reorder
            # step.
            tf_hess_csr = tf_sparse_csr.CSRSparseMatrix(term["hess_sparse"])

        const = term.get("const")
        lognorm = term.get("lognorm")
        if const is None or lognorm is None:
            if term["hess_dense"] is not None:
                d_const, d_lognorm = gaussian_scalars(
                    term["grad_values"], term["hess_dense"], term["name"]
                )
                const = d_const if const is None else const
                lognorm = d_lognorm if lognorm is None else lognorm
            elif term["hess_sparse"] is not None:
                logger.warning(
                    f"External likelihood term '{term['name']}' has a sparse Hessian "
                    "and no stored const/lognorm, so it cannot be centered or "
                    "normalized here (that would need a sparse solve and "
                    "log-determinant). Absolute NLL values will be offset for this "
                    "term. Pass mean= to TensorWriter.add_external_likelihood_term "
                    "to have them computed at write time."
                )
                const = 0.0 if const is None else const
                lognorm = 0.0 if lognorm is None else lognorm
            else:
                # grad-only: a linear tilt with no minimum, nothing to center.
                const = 0.0 if const is None else const
                lognorm = 0.0 if lognorm is None else lognorm

        out.append(
            {
                "name": term["name"],
                "indices": tf_indices,
                "grad": tf_grad,
                "hess_dense": tf_hess_dense,
                "hess_csr": tf_hess_csr,
                "const": float(const),
                "lognorm": float(lognorm),
            }
        )
    return out


def compute_external_nll(terms, x, dtype, full_nll=False):
    """Evaluate the scalar NLL contribution of a list of external terms.

    For each term, adds ``g^T x_sub + 0.5 * x_sub^T H x_sub + const`` to the
    running total, plus ``lognorm`` when ``full_nll`` is set. The two scalars
    are what turn the expanded quadratic back into a proper Gaussian; see the
    module docstring. They are pre-computed (per term, at build time), so
    adding them here costs one scalar add and cannot perturb the closed-form
    gradient / HVP paths below. Sparse Hessian terms use ``sm.matmul`` for the
    ``H @ x_sub`` product, which dispatches to a multi-threaded CSR
    kernel and is much faster per call than the previous element-wise
    gather-based form. The autodiff gradient and HVP of
    ``0.5 x^T H x`` via ``sm.matmul`` are themselves single
    ``sm.matmul`` calls, so reverse-over-reverse autodiff no longer
    rematerializes a 2D gather/scatter chain in the second-order tape
    — that was the dominant cost on large external-Hessian problems
    (e.g. jpsi: 329M-nnz prefit Hessian).

    Parameters
    ----------
    terms : list[dict]
        tf-side per-term dicts as returned by :func:`build_tf_external_terms`.
    x : tf.Tensor
        Current full parameter vector [npar], or a batch of them [npar, B]
        (batch axis last, see ``Fitter._compute_nll_batched``).
    dtype : tf.DType
        Dtype for the accumulator.
    full_nll : bool
        If set, also add each term's Gaussian log-normalization. Matches
        ``Fitter._compute_lc``, which adds ``0.9189 + log(sigma)`` per
        constrained parameter under the same flag.

    Returns
    -------
    tf.Tensor or None
        Scalar contribution to the NLL ([B] for a batch ``x``), or ``None`` if
        ``terms`` is empty.
    """
    if not terms:
        return None
    offset = sum(
        t.get("const", 0.0) + (t.get("lognorm", 0.0) if full_nll else 0.0)
        for t in terms
    )
    batched = x.shape.rank == 2
    if not batched:
        x = x[:, None]
    total = tf.fill(tf.shape(x)[1:], tf.constant(offset, dtype=dtype))
    for term in terms:
        x_sub = tf.gather(x, term["indices"])  # [k, B]
        if term["grad"] is not None:
            total = total + tf.reduce_sum(term["grad"][:, None] * x_sub, axis=0)
        if term["hess_dense"] is not None:
            # 0.5 * x_sub^T H x_sub
            total = total + 0.5 * tf.reduce_sum(
                x_sub * tf.matmul(term["hess_dense"], x_sub), axis=0
            )
        elif term["hess_csr"] is not None:
            # Loss = 0.5 * x_sub^T H x_sub via CSR matmul (H symmetric).
            Hx = tf_sparse_csr.matmul(term["hess_csr"], x_sub)
            total = total + 0.5 * tf.reduce_sum(x_sub * Hx, axis=0)
    return total if batched else total[0]
