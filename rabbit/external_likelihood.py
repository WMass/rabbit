"""Helpers for external likelihood terms (linear + quadratic parameter priors).

An "external likelihood term" is an additive contribution to the NLL of
the form

    -log L_ext = g^T x_sub + 0.5 * x_sub^T H x_sub

where ``x_sub`` is the subset of the fit parameters the term constrains.
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
  Hessians). Used by the Fitter when it takes ownership of the input
  data.
* :func:`compute_external_nll` — evaluate the scalar NLL contribution
  of a list of tf-side terms at the current ``x``.
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops as tf_sparse_csr

from rabbit.h5pyutils_read import makesparsetensor, maketensor


def read_external_terms_from_h5(ext_group):
    """Decode an HDF5 ``external_terms`` group into a list of raw dicts.

    Each entry has the keys used by the rest of the pipeline:

    * ``name``: term label (str, taken from the h5 subgroup name)
    * ``params``: 1D ndarray of parameter name strings
    * ``grad_values``: 1D float ndarray or ``None``
    * ``hess_dense``: 2D float ndarray or ``None``
    * ``hess_sparse``: :class:`tf.sparse.SparseTensor` or ``None`` (uses
      the same on-disk layout as ``hlogk_sparse`` / ``hnorm_sparse``)

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
            }
        )
    return terms


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

    Returns
    -------
    list[dict]
        One entry per term with keys ``name``, ``indices``, ``grad``,
        ``hess_dense``, ``hess_csr``. Empty if ``terms`` is empty.
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
            # that calls sm.matmul must be built with jit_compile=False.
            # The TensorWriter sorts the indices into canonical row-major
            # order at write time, so we can feed the SparseTensor
            # straight to the CSR builder without an additional reorder
            # step.
            tf_hess_csr = tf_sparse_csr.CSRSparseMatrix(term["hess_sparse"])

        out.append(
            {
                "name": term["name"],
                "indices": tf_indices,
                "grad": tf_grad,
                "hess_dense": tf_hess_dense,
                "hess_csr": tf_hess_csr,
            }
        )
    return out


def compute_external_nll(terms, x, dtype):
    """Evaluate the scalar NLL contribution of a list of external terms.

    For each term, adds ``g^T x_sub + 0.5 * x_sub^T H x_sub`` to the
    running total. Sparse Hessian terms use ``sm.matmul`` for the
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
        Current full parameter vector.
    dtype : tf.DType
        Dtype for the accumulator.

    Returns
    -------
    tf.Tensor or None
        Scalar contribution to the NLL, or ``None`` if ``terms`` is empty.
    """
    if not terms:
        return None
    total = tf.zeros([], dtype=dtype)
    for term in terms:
        x_sub = tf.gather(x, term["indices"])
        if term["grad"] is not None:
            total = total + tf.reduce_sum(term["grad"] * x_sub)
        if term["hess_dense"] is not None:
            # 0.5 * x_sub^T H x_sub
            total = total + 0.5 * tf.reduce_sum(
                x_sub * tf.linalg.matvec(term["hess_dense"], x_sub)
            )
        elif term["hess_csr"] is not None:
            # Loss = 0.5 * x_sub^T H x_sub via CSR matvec (H symmetric).
            Hx = tf.squeeze(
                tf_sparse_csr.matmul(term["hess_csr"], x_sub[:, None]),
                axis=-1,
            )
            total = total + 0.5 * tf.reduce_sum(x_sub * Hx)
    return total
