import tensorflow as tf

from combinetf2.physicsmodels import helpers
from combinetf2.physicsmodels.physicsmodel import PhysicsModel


class AngularCoefficients(PhysicsModel):
    """
    A class to compute ratios of channels, processes, or bins.
    Optionally the numerator and denominator can be normalized.

    Parameters
    ----------
        indata: Input data used for analysis (e.g., histograms or data structures).
        channel: str
            Name of the channel.
        channel: str
            Name of the process.
        selection: dict, optional
            Dictionary specifying selection criteria. Keys are axis names, and values are slices or conditions.
            Defaults to an empty dictionary meaning no selection.
            E.g. {"charge":0, "ptVgen":slice(0,2), "absYVgen": hist.sum, "massVgen": hist.rebin(2)}
            Selected axes are summed before the ratio is computed. To integrate over one axis before the ratio, use `slice(None)`
    """

    def __init__(
        self,
        indata,
        key,
        channel,
        processes=[],
        selection={},
        rebin_axes={},
        sum_axes=[],
        helicity_axis="helicitySig",
    ):
        self.key = key

        self.num = helpers.Term(
            indata,
            channel,
            processes,
            {helicity_axis: slice(1, None), **selection},
            rebin_axes,
            sum_axes,
        )  # sigma_i
        self.den = helpers.Term(
            indata,
            channel,
            processes,
            {helicity_axis: 0, **selection},
            rebin_axes,
            sum_axes,
        )  # sigma_UL

        self.has_data = False

        self.need_processes = False

        # The output of ratios will always be without process axis
        self.skip_per_process = True

        self.has_processes = False  # The result has no process axis

        self.helicity_index = [
            i for i, a in enumerate(self.num.channel_axes) if a.name == helicity_axis
        ][0]

        channel_axes = []
        for a in self.num.channel_axes:
            if a.name == helicity_axis:
                a.__dict__["name"] = "ai"
            channel_axes.append(a)

        self.channel_info = {
            channel: {
                "axes": channel_axes,
            }
        }

    @classmethod
    def parse_args(cls, indata, *args):
        """
        parsing the input arguments into the ratio constructor, is has to be called as
        -m AngularCoefficients
            <ch > <ch den>
            <proc_num_0>,<proc_num_1>,... <proc_num_0>,<proc_num_1>,...
            <axis_num_0>:<slice_num_0>,<axis_num_1>,<slice_num_1>... <axis_den_0>,<slice_den_0>,<axis_den_1>,<slice_den_1>...

        Processes selections are optional. But in case on is given for the numerator, the denominator must be specified as well and vice versa.
        Use 'None' if you don't want to select any for either numerator xor denominator.

        Axes selections are optional. But in case one is given for the numerator, the denominator must be specified as well and vice versa.
        Use 'None:None' if you don't want to do any for either numerator xor denominator.
        """

        if len(args) > 2 and ":" not in args[1]:
            procs = [p for p in args[1].split(",") if p != "None"]
        else:
            procs = []

        # find axis selections
        if any(a for a in args if ":" in a):
            sel_args = [a for a in args if ":" in a]

            axis_selection, axes_rebin, axes_sum = helpers.parse_axis_selection(
                sel_args[0]
            )
        else:
            axis_selection = {}
            axes_sum = []
            axes_rebin = {}

        key = " ".join([cls.__name__, *args])

        return cls(
            indata,
            key,
            args[0],
            procs,
            axis_selection,
            axes_rebin,
            axes_sum,
        )

    def compute_ais(self, observables, inclusive=False):
        num = self.num.select(observables, inclusive=inclusive)
        den = self.den.select(observables, inclusive=inclusive)

        den = tf.expand_dims(den, axis=self.helicity_index)

        return num / den

    def compute_flat(self, params, observables):
        return self.compute_ais(observables, True)

    def compute_flat_per_process(self, params, observables):
        return self.compute_ais(observables, False)


class LamTung(AngularCoefficients):

    def __init__(self, indata, key, channel, *args, **kwargs):
        super().__init__(indata, key, channel, *args, **kwargs)

        self.channel_info[channel]["axes"] = [
            c for c in self.channel_info[channel]["axes"] if c.name != "ai"
        ]

    def compute_ais(self, observables, inclusive=False):
        ais = super().compute_ais(observables, inclusive)
        a0 = tf.gather(ais, indices=0, axis=self.helicity_index)
        a2 = tf.gather(ais, indices=2, axis=self.helicity_index)
        return a0 - a2
