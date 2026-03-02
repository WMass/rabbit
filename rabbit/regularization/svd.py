import tensorflow as tf

from rabbit.regularization.regularizer import Regularizer


class SVD(Regularizer):
    """
    Singular Value Decomposition (SVD) see: https://arxiv.org/abs/hep-ph/9509307
    """

    def __init__(self, mapping, dtype):
        if len(mapping.channel_info) > 1:
            raise NotImplementedError(
                "Regularization currently only works for 1 channel at a time; use multiple regularizers if you want to penalize multiple channels."
            )

        self.mapping = mapping

        # there is an embiguity about what to do with the flow bins.
        #   they are not part of the fit, thus, the flow bins are not taken except for masked channels
        self.input_shape = [
            a.extent if v["flow"] else a.size
            for v in mapping.channel_info.values()
            for a in v["axes"]
        ]

        self.ndims = len(self.input_shape)

        if self.ndims == 1:
            kernel = tf.constant([1, -2, 1], dtype=dtype)
            self.kernel = kernel[:, tf.newaxis, tf.newaxis]  # (W, In, Out)
        elif self.ndims == 2:
            kernel = tf.constant([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=dtype)
            self.kernel = kernel[:, :, tf.newaxis, tf.newaxis]  # (H, W, In, Out)
        elif self.ndims == 3:
            # Axial neighbors are 1, center is -6
            kernel = tf.zeros((3, 3, 3), dtype=dtype)
            indices = [
                [1, 1, 1],
                [0, 1, 1],
                [2, 1, 1],
                [1, 0, 1],
                [1, 2, 1],
                [1, 1, 0],
                [1, 1, 2],
            ]
            values = [-6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            kernel = tf.tensor_scatter_nd_update(kernel, indices, values)
            self.kernel = kernel[:, :, :, tf.newaxis, tf.newaxis]
        else:
            raise NotImplementedError("SVD regularization only implemented in up to 3D")

        self.paddings = [[0, 0]] + [[1, 1]] * self.ndims + [[0, 0]]

    def set_expectations(self, initial_params, initial_observables):
        # TODO: do we need to include this in autodiff for global impacts, since initial_params = (poi0, theta0)?
        nexp0 = self.mapping.compute_flat(initial_params, initial_observables)
        self.nexp0 = tf.reshape(nexp0, self.input_shape)

    def compute_nll_penalty(self, params, observables):
        mask = self.nexp0 != 0
        nexp0_safe = tf.where(mask, self.nexp0, tf.cast(1.0, self.nexp0.dtype))

        nexp = self.mapping.compute_flat(params, observables)
        nexp = tf.reshape(nexp, self.input_shape)

        dexp = nexp / nexp0_safe
        dexp = tf.where(mask, dexp, tf.ones_like(dexp))

        # add batch (first) and channel (last) dimensions
        dexp = dexp[tf.newaxis, ..., tf.newaxis]

        # padding 'SYMMETRIC' means copy the element at the edge, i.e. apply the kernel (1 -2 1) to (x x y) -> x -2x + y = -x y
        #   which is equivalent to applying a "modified kernal" of (-1, 1) to (x y) -> -x y
        padded_input = tf.pad(dexp, self.paddings, mode="SYMMETRIC")

        if self.ndims == 1:
            curvature_map = tf.nn.conv1d(
                padded_input, self.kernel, stride=1, padding="VALID"
            )
        elif self.ndims == 2:
            curvature_map = tf.nn.conv2d(
                padded_input, self.kernel, strides=[1, 1, 1, 1], padding="VALID"
            )
        elif self.ndims == 3:
            curvature_map = tf.nn.conv3d(
                padded_input, self.kernel, strides=[1, 1, 1, 1, 1], padding="VALID"
            )

        penalty = tf.reduce_sum(tf.square(curvature_map))

        return penalty
