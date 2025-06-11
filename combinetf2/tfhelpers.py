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


def is_diag(x):
    return tf.math.equal(
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

    return edmval


def edmval(grad, hess):
    # scipy is faster than tensorflow on CPU so use it as appropriate
    if is_on_gpu(hess):
        return tf_edmval(grad, hess)
    else:
        return scipy_edmval(grad.__array__(), hess.__array__())


def cond_number(hess):
    if is_on_gpu(hess):
        return tf.linalg.cond(hess)
    else:
        return scipy_cond_number(hess.__array__())


def segment_sum_along_axis(x, segment_ids, idx, num_segments):
    # Move the target axis to the front
    perm = [idx] + [i for i in range(len(x.shape)) if i != idx]
    x_transposed = tf.transpose(x, perm)

    # Apply segment_sum along axis 0
    rebinned = tf.math.segment_sum(x_transposed, segment_ids)

    # Update static shape if possible
    static_shape = rebinned.shape.as_list()
    static_shape[0] = num_segments
    rebinned.set_shape(static_shape)

    # Undo the transposition
    reverse_perm = [perm.index(i) for i in range(len(perm))]
    rebinned = tf.transpose(rebinned, reverse_perm)

    return rebinned
