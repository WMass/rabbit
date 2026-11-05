import math

import tensorflow as tf

from .scipyhelpers import scipy_cond_number, scipy_edmval, scipy_edmval_cov


def simple_sparse_slice0end(in_sparse, end):
    """
    Slice a tf.sparse.SparseTensor along axis 0 starting 0 to the 'end'.
    """

    # Convert dense_shape, indices, and values to tensors if they aren't already
    dense_shape = in_sparse.dense_shape
    indices = in_sparse.indices
    values = in_sparse.values

    # Compute output dense shape after slicing
    out_shape = tf.concat([[end], dense_shape[1:]], axis=0)

    # Filter rows: select entries where indices[:, 0] < end
    mask = indices[:, 0] < end
    selected_indices = tf.boolean_mask(indices, mask)
    selected_values = tf.boolean_mask(values, mask)

    # Return the sliced sparse tensor
    return tf.sparse.SparseTensor(
        indices=selected_indices, values=selected_values, dense_shape=out_shape
    )


def per_parameter(v, like):
    """A per-parameter vector `v` of shape [n], laid out against `like`.

    `like` is [n], or [n, B] with the batch axis last (the batched likelihood,
    Fitter._compute_nll_batched); `v` then becomes the column [n, 1].
    """
    return v if like.shape.rank == 1 else v[:, None]


def _batch_first(a):
    rank = a.shape.rank
    return tf.transpose(a, [rank - 1] + list(range(rank - 1)))


def _batch_last(a):
    rank = a.shape.rank
    return tf.transpose(a, list(range(1, rank)) + [0])


def map_batch(fn, args):
    """Apply a per-element function over the trailing batch axis of `args`.

    `args` is a sequence of tensors whose last axis is the batch axis, of size
    B or of size 1 for an argument all B elements share, or None. `fn` takes
    the per-element tensors (the batch axis dropped, None passed through) and
    returns a nest of tensors or None; they come back with the batch axis last.

    Without a batch axis of size other than 1 this is one direct call of `fn`,
    so B == 1 runs exactly the per-element code. Otherwise the same `fn` is
    vectorized with tf.vectorized_map, the shared arguments captured as they
    are.
    """
    args = [None if a is None else tf.convert_to_tensor(a) for a in args]
    batched = [a is not None and a.shape[-1] != 1 for a in args]

    def call(elems):
        it = iter(elems)
        return fn(
            *[
                next(it) if b else (None if a is None else a[..., 0])
                for a, b in zip(args, batched)
            ]
        )

    if not any(batched):
        out = call(())
        return tf.nest.map_structure(lambda o: None if o is None else o[..., None], out)

    # vectorized_map cannot return None: map the tensor outputs only and put
    # the Nones back in place afterwards
    structure = []

    def one(elems):
        out = call(elems)
        flat = tf.nest.flatten(out)
        structure[:] = [out, [o is None for o in flat]]
        return tuple(o for o in flat if o is not None)

    mapped = tf.vectorized_map(
        one, tuple(_batch_first(a) for a, b in zip(args, batched) if b)
    )
    out, nones = structure
    it = iter(mapped)
    flat = [None if none else _batch_last(next(it)) for none in nones]
    return tf.nest.pack_sequence_as(out, flat)


def is_diag(x):
    return tf.equal(
        tf.math.count_nonzero(x), tf.math.count_nonzero(tf.linalg.diag_part(x))
    )


def is_on_gpu(tensor):
    """
    Check if tensor is on a GPU device
    """

    device = tensor.device
    device_type = device.split(":")[-2]
    return device_type == "GPU"


def tf_edmval_cov(grad, hess):
    # use a Cholesky decomposition to easily detect the non-positive-definite case
    chol = tf.linalg.cholesky(hess)

    # FIXME catch this exception to mark failed toys and continue
    if tf.reduce_any(tf.math.is_nan(chol)).numpy():
        raise ValueError(
            "Cholesky decomposition failed, Hessian is not positive-definite"
        )

    gradv = grad[..., None]
    edmval = 0.5 * tf.linalg.matmul(
        gradv, tf.linalg.cholesky_solve(chol, gradv), transpose_a=True
    )
    edmval = edmval[0, 0].numpy()

    cov = tf.linalg.cholesky_solve(chol, tf.eye(chol.shape[0], dtype=chol.dtype))

    return edmval, cov


def edmval_cov(grad, hess):
    # scipy is faster than tensorflow on CPU so use it as appropriate
    if is_on_gpu(hess):
        return tf_edmval_cov(grad, hess)
    else:
        return scipy_edmval_cov(grad.__array__(), hess.__array__())


def tf_edmval(grad, hess):
    # Ensure proper shapes
    grad = tf.reshape(grad, (-1, 1))  # shape (n, 1)

    # Solve H x = g for x
    x = tf.linalg.solve(hess, grad)  # shape (n, 1)

    # Compute EDM = 0.5 * g^T x
    edm = 0.5 * tf.squeeze(tf.matmul(tf.transpose(grad), x))

    return edm


def edmval(grad, hess):
    # scipy is faster than tensorflow on CPU so use it as appropriate
    if is_on_gpu(hess):
        return tf_edmval(grad, hess)
    else:
        return scipy_edmval(grad.__array__(), hess.__array__())


def cond_number(hess):
    if is_on_gpu(hess):
        # 2-norm condition number, as numpy.linalg.cond (tf has no cond)
        s = tf.linalg.svd(hess, compute_uv=False)
        return s[0] / s[-1]
    else:
        return scipy_cond_number(hess.__array__())


def segment_sum_along_axis(x, segment_ids, idx, num_segments):
    # Move the target axis to the front
    perm = [idx] + [i for i in range(len(x.shape)) if i != idx]
    x_transposed = tf.transpose(x, perm)

    # Count leading/trailing -1s and cut them off
    nleading = tf.reduce_sum(
        tf.cast(
            tf.cumsum(tf.cast(segment_ids == -1, tf.int32))
            == tf.range(1, len(segment_ids) + 1),
            tf.int32,
        )
    )
    ntrailing = tf.reduce_sum(
        tf.cast(
            tf.cumsum(tf.cast(segment_ids[::-1] == -1, tf.int32))
            == tf.range(1, len(segment_ids) + 1),
            tf.int32,
        )
    )
    x_transposed = x_transposed[nleading : len(x_transposed) - ntrailing]
    segment_ids_valid = segment_ids[nleading : len(segment_ids) - ntrailing]

    # Apply segment_sum along axis 0
    rebinned = tf.math.segment_sum(x_transposed, segment_ids_valid)

    # Update static shape if possible
    static_shape = rebinned.shape.as_list()
    static_shape[0] = num_segments
    rebinned.set_shape(static_shape)

    # Undo the transposition
    reverse_perm = [perm.index(i) for i in range(len(perm))]
    rebinned = tf.transpose(rebinned, reverse_perm)

    return rebinned


def normal_pdf(x):
    x = tf.convert_to_tensor(x)
    two_pi = tf.constant(2.0 * math.pi, dtype=x.dtype)
    return tf.exp(-0.5 * tf.square(x)) / tf.sqrt(two_pi)


def normal_cdf(x):
    return 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(tf.constant(2.0, dtype=x.dtype))))
