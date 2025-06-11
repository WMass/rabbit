import hist
import tensorflow as tf

from combinetf2.physicsmodels import helpers
from combinetf2.physicsmodels.physicsmodel import PhysicsModel


class Ratio(PhysicsModel):
    """
    A class to compute ratios of channels, processes, or bins.
    Optionally the numerator and denominator can be normalized.

    Parameters
    ----------
        indata: Input data used for analysis (e.g., histograms or data structures).
        num_channel: str
            Name of the numerator channel.
        den_channel: str
            Name of the denominator channel.
        num_processes: list of str, optional
            List of process names for the numerator channel. Defaults to None, meaning all processes will be considered.
            Selected processes are summed before the ratio is computed.
        den_processes: list of str, optional
            Same as num_processes but for denumerator
        num_selection: dict, optional
            Dictionary specifying selection criteria for the numerator. Keys are axis names, and values are slices or conditions.
            Defaults to an empty dictionary meaning no selection.
            E.g. {"charge":0, "ai":slice(0,2)}
            Selected axes are summed before the ratio is computed. To integrate over one axis before the ratio, use `slice(None)`
        den_selection: dict, optional
            Same as num_selection but for denumerator
        normalize: bool, optional
            Whether to normalize the numerator and denominator before the ratio. Defaults to False.
    """

    def __init__(
        self,
        indata,
        key,
        num_channel,
        den_channel,
        num_processes=[],
        den_processes=[],
        num_selection={},
        den_selection={},
        num_axes_rebin=[],
        den_axes_rebin=[],
        num_axes_sum=[],
        den_axes_sum=[],
    ):
        self.key = key

        self.num = helpers.Term(
            indata,
            num_channel,
            num_processes,
            num_selection,
            num_axes_rebin,
            num_axes_sum,
        )
        self.den = helpers.Term(
            indata,
            den_channel,
            den_processes,
            den_selection,
            den_axes_rebin,
            den_axes_sum,
        )

        self.has_data = self.num.has_data and self.den.has_data

        self.need_processes = len(num_processes) or len(
            den_processes
        )  # the fun_flat will be by processes

        # The output of ratios will always be without process axis
        self.skip_per_process = True

        if [a.size for a in self.num.channel_axes] != [
            a.size for a in self.den.channel_axes
        ]:
            raise RuntimeError(
                "Channel axes for numerator and denominator must have the same number of bins"
            )
        elif self.num.channel_axes != self.den.channel_axes:
            # same number of bins but different axis name, make new integer axes with axis names a0, a1, ...
            hist_axes = [
                hist.axis.IntCategory(range(a.size), name=f"a{i}", overflow=False)
                for i, a in enumerate(self.num.channel_axes)
            ]
        else:
            hist_axes = self.num.channel_axes

        if num_channel == den_channel:
            channel = num_channel
        else:
            channel = f"{num_channel}_{den_channel}"

        self.has_processes = False  # The result has no process axis

        self.channel_info = {
            channel: {
                "axes": hist_axes,
            }
        }

    @classmethod
    def parse_args(cls, indata, *args):
        """
        parsing the input arguments into the ratio constructor, is has to be called as
        -m Ratio
            <ch num> <ch den>
            <proc_num_0>,<proc_num_1>,... <proc_num_0>,<proc_num_1>,...
            <axis_num_0>:<slice_num_0>,<axis_num_1>,<slice_num_1>... <axis_den_0>,<slice_den_0>,<axis_den_1>,<slice_den_1>...

        Processes selections are optional. But in case on is given for the numerator, the denominator must be specified as well and vice versa.
        Use 'None' if you don't want to select any for either numerator xor denominator.

        Axes selections are optional. But in case one is given for the numerator, the denominator must be specified as well and vice versa.
        Use 'None:None' if you don't want to do any for either numerator xor denominator.
        """

        if len(args) > 2 and ":" not in args[2]:
            procs_num = [p for p in args[2].split(",") if p != "None"]
            procs_den = [p for p in args[3].split(",") if p != "None"]
        else:
            procs_num = []
            procs_den = []

        # find axis selections
        if any(a for a in args if ":" in a):
            sel_args = [a for a in args if ":" in a]
        else:
            sel_args = ["None:None", "None:None"]

        axis_selection_num, axes_rebin_num, axes_sum_num = helpers.parse_axis_selection(
            sel_args[0]
        )
        axis_selection_den, axes_rebin_den, axes_sum_den = helpers.parse_axis_selection(
            sel_args[1]
        )

        key = " ".join([cls.__name__, *args])

        return cls(
            indata,
            key,
            args[0],
            args[1],
            procs_num,
            procs_den,
            axis_selection_num,
            axis_selection_den,
            axes_rebin_num,
            axes_rebin_den,
            axes_sum_num,
            axes_sum_den,
        )

    def compute_flat(self, params, observables):
        num = self.num.select(observables, inclusive=True)
        den = self.den.select(observables, inclusive=True)

        return num / den

    def compute_flat_per_process(self, params, observables):
        return self.compute_flat(params, observables)


class Normratio(Ratio):
    """
    Same as Ratio but the numerator and denominator are normalized
    """

    ndf_reduction = 1

    def init(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_flat(self, params, observables):
        num = self.num.select(observables, normalize=True, inclusive=True)
        den = self.den.select(observables, normalize=True, inclusive=True)
        exp = num / den
        exp = tf.reshape(exp, [-1])
        return exp
