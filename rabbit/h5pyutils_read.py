import tensorflow as tf


def maketensor(h5dset):
    if "original_shape" in h5dset.attrs:
        shape = h5dset.attrs["original_shape"]
    else:
        shape = h5dset.shape

    if h5dset.size == 0:
        return tf.zeros(shape, h5dset.dtype)

    # read directly from hdf5 dataset to the underlying buffer of a tensor
    # this requires that the tensor is located on the CPU, so force the device
    with tf.device(tf.config.list_logical_devices("CPU")[0]):
        atensor = tf.zeros(h5dset.shape, h5dset.dtype)
        # zero tensors have a special flag set, using the identity clears this implicitly
        atensor = tf.identity(atensor)

    # read into the underlying array
    h5dset.read_direct(atensor.__array__())

    # the reshape operation is needed in case the hdf5 dataset was flattened
    # this also triggers a copy of the tensor to the default device (e.g. GPU)
    # if needed (ie if not the default CPU device)
    atensor = tf.reshape(atensor, shape)
    return atensor


def makesparsetensor(h5group):
    indices = maketensor(h5group["indices"])
    values = maketensor(h5group["values"])
    dense_shape = h5group.attrs["dense_shape"]

    return tf.sparse.SparseTensor(indices, values, dense_shape)
