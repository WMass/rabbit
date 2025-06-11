import tensorflow as tf

from combinetf2.physicsmodels.physicsmodel import Channelmodel


class Project(Channelmodel):
    """
    A class to project a histogram to lower dimensions.
    The normalization is done to the integral of all processes or data.

    Parameters
    ----------
    channel_name : str
        Name of the channel. Required.
    axes_names : list of str, optional
        Names of the axes to keep. If empty, the histogram will be projected to a single bin.
    """

    def __init__(self, indata, key, channel, *axes_names):
        super().__init__(indata, key, channel)
        self.channel = channel

        info = indata.channel_info[channel]
        channel_axes = {a.name: a for a in info["axes"]}

        hist_axes = [channel_axes[n] for n in axes_names]

        if len(hist_axes) != len(axes_names):
            raise ValueError(
                f"Hist axes {[h.name for h in hist_axes]} != {axes_names} not found"
            )

        channel_axes_names = [axis.name for axis in channel_axes.values()]

        axis_idxs = [channel_axes_names.index(axis) for axis in axes_names]

        self.proj_idxs = [i for i in range(len(channel_axes)) if i not in axis_idxs]

        post_proj_axes_names = [
            axis for axis in channel_axes_names if axis in axes_names
        ]

        self.transpose_idxs = [post_proj_axes_names.index(axis) for axis in axes_names]

        self.has_data = not info.get("masked", False)

        self.channel_info = {channel: {"axes": hist_axes, "processes": indata.procs}}

    def project(self, values):
        exp = tf.reduce_sum(values, axis=self.proj_idxs)
        perm = self.transpose_idxs[:]
        if len(exp.shape) > len(self.transpose_idxs):
            # last is process axis
            perm += list(range(len(self.transpose_idxs), len(exp.shape)))
        exp = tf.transpose(exp, perm=perm)
        return exp

    def compute(self, param, observables):
        return self.project(observables)


class Normalize(Project):
    """
    Same as project but also normalize
    """

    ndf_reduction = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def project(self, values, *args):
        norm = tf.reduce_sum(values)
        out = values / norm
        return super().project(out, *args)
