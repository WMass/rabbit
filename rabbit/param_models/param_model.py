import functools
import itertools

import numpy as np
import tensorflow as tf


class ParamModel:

    def __init__(self, indata, *args, **kwargs):
        self.indata = indata

        # # a param model must set these attributes
        # self.npoi = # number of true parameters of interest (POIs), reported as POIs in outputs
        # self.npou = # number of model nuisance parameters (parameters of uninterest)
        # self.params = # list of names for all parameters (POIs first, then model nuisances)
        # self.xparamdefault = # default values for all parameters (length nparams)
        # self.is_linear = # define if the model is linear in the parameters
        # self.allowNegativeParam = # define if the POI parameters can be negative or not

    @property
    def nparams(self):
        """Total number of parameters: npoi + npou."""
        return self.npoi + self.npou

    # class function to parse strings as given by the argparse input e.g. --paramModel <Model> <arg[0]> <args[1]> ...
    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        return cls(indata, *args, **kwargs)

    def compute(self, param, full=False):
        """
        Compute an array for the rate per process
        :param param: 1D tensor of explicit parameters in the fit (length nparams)
        :return 2D tensor to be multiplied with [proc,bin] tensor
        """

    def set_param_default(self, expectSignal, allowNegativeParam=False):
        """
        Set default parameter values, used by different param models.
        Only the first npoi entries (true POIs) support the squaring transform;
        model nuisance parameters (npou entries) are always stored directly.
        """
        paramdefault = tf.ones([self.nparams], dtype=self.indata.dtype)
        if expectSignal is not None:
            indices = []
            updates = []
            for signal, value in expectSignal:
                if signal.encode() not in self.params:
                    raise ValueError(
                        f"{signal.encode()} not in list of params: {self.params}"
                    )
                idx = np.where(np.isin(self.params, signal.encode()))[0][0]

                indices.append([idx])
                updates.append(float(value))

            paramdefault = tf.tensor_scatter_nd_update(paramdefault, indices, updates)

        # squaring transform applies only to the npoi true POI entries
        poi_part = paramdefault[: self.npoi]
        nui_part = paramdefault[self.npoi :]

        if allowNegativeParam:
            xpoi_part = poi_part
        else:
            xpoi_part = tf.sqrt(poi_part)

        self.xparamdefault = tf.concat([xpoi_part, nui_part], axis=0)


class CompositeParamModel(ParamModel):
    """
    multiply different param models together
    """

    def __init__(
        self,
        param_models,
        allowNegativeParam=False,
    ):

        self.param_models = param_models

        self.npoi = sum([m.npoi for m in param_models])
        self.npou = sum([m.npou for m in param_models])

        self.params = np.concatenate([m.params for m in param_models])

        self.allowNegativeParam = allowNegativeParam

        self.is_linear = self.nparams == 0 or self.allowNegativeParam

        self.xparamdefault = tf.concat([m.xparamdefault for m in param_models], axis=0)

    def compute(self, param, full=False):
        start = 0
        results = []
        for m in self.param_models:
            results.append(m.compute(param[start : start + m.nparams], full))
            start += m.nparams

        rnorm = functools.reduce(lambda a, b: a * b, results)
        return rnorm


class Ones(ParamModel):
    """
    multiply all processes with ones
    """

    def __init__(self, indata, **kwargs):
        self.indata = indata
        self.npoi = 0
        self.npou = 0
        self.params = np.array([])
        self.xparamdefault = tf.zeros([0], dtype=self.indata.dtype)

        self.allowNegativeParam = False
        self.is_linear = True

    def compute(self, param, full=False):
        rnorm = tf.ones(self.indata.nproc, dtype=self.indata.dtype)
        rnorm = tf.reshape(rnorm, [1, -1])
        return rnorm


class Mu(ParamModel):
    """
    multiply unconstrained parameter to signal processes, and ones otherwise
    """

    def __init__(self, indata, expectSignal=None, allowNegativeParam=False, **kwargs):
        self.indata = indata

        self.npoi = self.indata.nsignals
        self.npou = 0

        self.params = np.array([s for s in self.indata.signals])

        self.allowNegativeParam = allowNegativeParam

        self.is_linear = self.nparams == 0 or self.allowNegativeParam

        self.set_param_default(expectSignal, allowNegativeParam)

    def compute(self, param, full=False):
        rnorm = tf.concat(
            [
                param,
                tf.ones([self.indata.nproc - param.shape[0]], dtype=self.indata.dtype),
            ],
            axis=0,
        )

        rnorm = tf.reshape(rnorm, [1, -1])
        return rnorm


class Mixture(ParamModel):
    """
    Based on unconstrained parameters x_i
    multiply `primary` process by x_i
    multiply `complementary` process by 1-x_i
    """

    def __init__(
        self,
        indata,
        primary_processes,
        complementary_processes,
        expectSignal=None,
        allowNegativeParam=False,
        **kwargs,
    ):
        self.indata = indata

        if type(primary_processes) == str:
            primary_processes = [primary_processes]

        if type(complementary_processes) == str:
            complementary_processes = [complementary_processes]

        primary_processes = np.array(primary_processes).astype("S")
        complementary_processes = np.array(complementary_processes).astype("S")

        if len(primary_processes) != len(complementary_processes):
            raise ValueError(
                f"Length of pimary and complementary processes has to be the same, but got {len(primary_processes)} and {len(complementary_processes)}"
            )

        if any(n not in self.indata.procs for n in primary_processes):
            not_found = [n for n in primary_processes if n not in self.indata.procs]
            raise ValueError(f"{not_found} not found in processes {self.indata.procs}")

        if any(n not in self.indata.procs for n in complementary_processes):
            not_found = [
                n for n in complementary_processes if n not in self.indata.procs
            ]
            raise ValueError(f"{not_found} not found in processes {self.indata.procs}")

        self.primary_idxs = np.where(np.isin(self.indata.procs, primary_processes))[0]
        self.complementary_idxs = np.where(
            np.isin(self.indata.procs, complementary_processes)
        )[0]
        self.all_idx = np.concatenate([self.primary_idxs, self.complementary_idxs])

        self.npoi = len(primary_processes)
        self.npou = 0
        self.params = np.array(
            [
                f"{p}_{c}_mixing".encode()
                for p, c in zip(
                    primary_processes.astype(str), complementary_processes.astype(str)
                )
            ]
        )

        self.allowNegativeParam = allowNegativeParam
        self.is_linear = False

        self.set_param_default(expectSignal, allowNegativeParam)

    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        """
        parsing the input arguments into the constructor, is has to be called as
        --paramModel Mixture <proc_0>,<proc_1>,... <proc_a>,<proc_b>,...
        to introduce a mixing parameter for proc_0 with proc_a, and proc_1 with proc_b, etc.
        """

        if len(args) != 2:
            raise ValueError(
                f"Expected exactly 2 arguments for Mixture model but got {len(args)}"
            )

        primaries = args[0].split(",")
        complementaries = args[1].split(",")

        return cls(indata, primaries, complementaries, **kwargs)

    def compute(self, param, full=False):

        ones = tf.ones(self.nparams, dtype=self.indata.dtype)
        updates = tf.concat([ones * param, ones * (1 - param)], axis=0)

        # Single scatter update
        rnorm = tf.tensor_scatter_nd_update(
            tf.ones(self.indata.nproc, dtype=self.indata.dtype),
            self.all_idx[:, None],
            updates,
        )

        rnorm = tf.reshape(rnorm, [1, -1])
        return rnorm


class SaturatedProjectModel(ParamModel):
    """
    For computing the saturated test statistic of a projection.
    Add one free parameter for each projected bin
    """

    def __init__(
        self,
        indata,
        channel_info,
        expectSignal=None,
        allowNegativeParam=False,
        **kwargs,
    ):
        self.indata = indata
        self.channel_info_mapping = channel_info

        self.npoi = int(
            np.sum(
                [
                    np.prod([a.size for a in v["axes"]]) if len(v["axes"]) else 1
                    for v in channel_info.values()
                ]
            )
        )
        self.npou = 0

        names = []
        for k, v in self.channel_info_mapping.items():
            for idxs in itertools.product(*[range(a.size) for a in v["axes"]]):
                label = "_".join(f"{a.name}{i}" for a, i in zip(v["axes"], idxs))
                names.append(f"saturated_{k}_{label}".encode())

        self.params = np.array(names)

        self.allowNegativeParam = allowNegativeParam

        self.is_linear = self.nparams == 0 or self.allowNegativeParam

        self.set_param_default(expectSignal, allowNegativeParam)

    def compute(self, param, full=False):
        start = 0
        rnorms = []
        for k, v in self.indata.channel_info.items():
            if v["masked"] and not full:
                continue
            shape_input = [a.size for a in v["axes"]]

            irnorm = tf.ones(shape_input, dtype=self.indata.dtype)
            if k in self.channel_info_mapping.keys():
                mapping_axes = self.channel_info_mapping[k]["axes"]
                shape_mapping = [a.size if a in mapping_axes else 1 for a in v["axes"]]
                n_mapping_params = np.prod([a.size for a in mapping_axes])
                iparam = param[start : start + n_mapping_params]
                irnorm *= tf.reshape(iparam, shape_mapping)
                start += n_mapping_params

            irnorm = tf.reshape(
                irnorm,
                [
                    -1,
                ],
            )
            rnorms.append(irnorm)

        rnorm = tf.concat(rnorms, axis=0)
        rnorm = tf.reshape(rnorm, [-1, 1])

        return rnorm
