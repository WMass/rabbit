import hist
import tensorflow as tf

from rabbit.mappings.mapping import Mapping


class Param(Mapping):
    """
    Custom mapping of fit parameters using f(x) = scale*x + offset

    Parameters
    ----------
        indata: Input data used for analysis (e.g., histograms or data structures).
        params: str or list of string
            Parameter names
        scale: float or list of floats, optional
            Scale the parameters
        offset: float or list of floats, optional
            add an offset to the parameters
    """

    need_observables = False
    has_data = False
    has_processes = False
    skip_per_process = True
    skip_prefit = True

    def __init__(
        self,
        indata,
        key,
        params,
        scales=None,
        offsets=None,
    ):
        self.key = key

        # TODO: account for generic poi model
        param_names = list(indata.signals.astype(str)) + list(indata.systs.astype(str))

        if isinstance(params, str):
            idxs = [param_names.index(params)]
        else:
            idxs = [param_names.index(p) for p in params]

        self.idxs = tf.constant(idxs, dtype=tf.int32)

        self.params = [param_names[i] for i in idxs]

        scales = [scales] if isinstance(int, float) else scales
        offsets = [offsets] if isinstance(int, float) else offsets

        self.scales = tf.constant(scales, tf.float64)
        self.offsets = tf.constant(offsets, tf.float64)

        self.channel_info = {
            "param": {
                "axes": [
                    hist.axis.StrCategory(self.params, name="params"),
                ],
                "start": None,
                "stop": None,
            }
        }

        self.instance = "param"

    @classmethod
    def parse_args(cls, indata, params, scales=None, offsets=None):
        key = " ".join([cls.__name__, params, scales, offsets])
        params = params.split(",")
        if scales is not None:
            scales = [float(s) for s in scales.split(",")]
        if offsets is not None:
            offsets = [float(o) for o in offsets.split(",")]

        return cls(indata, key, params, scales, offsets)

    def compute_flat(self, params):
        params = tf.gather(params, indices=self.idxs, axis=-1)
        return params * self.scales + self.offsets
