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


class AxisNormModel(ParamModel):
    """
    One independent normalization parameter per (process, bin-combination) of a
    caller-specified set of axes, within a named channel.  Each process in
    proc_spec gets its own set of per-cell parameters; they are never shared
    across processes.  All other channels and processes are left at scale factor 1.

    Usage::

        --paramModel AxisNormModel <channel> <proc_spec> <axes>

    where proc_spec is ``all`` or a comma-separated list of process names,
    and axes is a comma-separated list of axis names.

    Example (btojpsik: independent per-cell norms for signal and flat bkg)::

        --paramModel AxisNormModel btojpsik_stuff signal,flatBkg bkmm_kaon_pt,bkmm_kaon_eta,bkmm_kaon_charge
    """

    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        if len(args) != 3:
            raise ValueError(
                f"AxisNormModel requires exactly 3 positional arguments "
                f"(channel, proc_spec, axes) but got {len(args)}: {args}"
            )
        channel, proc_spec, axes_csv = args
        return cls(indata, channel, proc_spec, axes_csv, **kwargs)

    def __init__(
        self,
        indata,
        channel,
        proc_spec,
        axes_csv,
        expectSignal=None,
        allowNegativeParam=False,
        **kwargs,
    ):
        self.indata = indata

        if channel not in indata.channel_info:
            raise ValueError(
                f"Channel '{channel}' not found in tensor. "
                f"Available: {list(indata.channel_info.keys())}"
            )
        self.channel = channel
        axes = indata.channel_info[channel]["axes"]
        axis_by_name = {a.name: a for a in axes}

        requested_names = [n.strip() for n in axes_csv.split(",")]
        for name in requested_names:
            if name not in axis_by_name:
                raise ValueError(
                    f"Axis '{name}' not found in channel '{channel}'. "
                    f"Available: {list(axis_by_name.keys())}"
                )
        self.requested_axis_names = set(requested_names)
        self.requested_axes = [axis_by_name[n] for n in requested_names]

        if proc_spec == "all":
            target_encoded = list(indata.procs)
        else:
            target_encoded = []
            for name in [p.strip() for p in proc_spec.split(",")]:
                encoded = name.encode() if isinstance(name, str) else name
                if encoded not in indata.procs:
                    raise ValueError(
                        f"Process '{name}' not found in tensor. "
                        f"Available: {[p.decode() if isinstance(p, bytes) else p for p in indata.procs]}"
                    )
                target_encoded.append(encoded)
        self.proc_idxs = [int(np.where(indata.procs == p)[0][0]) for p in target_encoded]

        cell_shape = [a.size for a in self.requested_axes]
        self.n_cell = int(np.prod(cell_shape))
        self.npoi = len(self.proc_idxs) * self.n_cell

        names = []
        for proc_encoded in target_encoded:
            proc_name = proc_encoded.decode() if isinstance(proc_encoded, bytes) else str(proc_encoded)
            for idxs in itertools.product(*[range(s) for s in cell_shape]):
                label = "_".join(f"{a.name}{i}" for a, i in zip(self.requested_axes, idxs))
                names.append(f"norm_{proc_name}_{label}".encode())
        self.params = np.array(names)

        self.npou = 0
        self.allowNegativeParam = allowNegativeParam
        self.is_linear = self.npoi == 0 or self.allowNegativeParam

        self.set_param_default(expectSignal, allowNegativeParam)

    def compute(self, param, full=False):
        reshape = [
            a.size if a.name in self.requested_axis_names else 1
            for a in self.indata.channel_info[self.channel]["axes"]
        ]
        shape_input = [a.size for a in self.indata.channel_info[self.channel]["axes"]]

        rnorms = []
        for k, v in self.indata.channel_info.items():
            if v["masked"] and not full:
                continue
            nbins_channel = int(np.prod([a.size for a in v["axes"]]))
            irnorm = tf.ones([nbins_channel, self.indata.nproc], dtype=self.indata.dtype)
            if k == self.channel:
                for i, proc_idx in enumerate(self.proc_idxs):
                    ipoi = param[i * self.n_cell : (i + 1) * self.n_cell]
                    scaling = tf.reshape(
                        tf.broadcast_to(tf.reshape(ipoi, reshape), shape_input), [-1, 1]
                    )
                    proc_col = tf.one_hot(proc_idx, self.indata.nproc, dtype=self.indata.dtype)
                    irnorm = irnorm + (scaling - 1.0) * tf.reshape(proc_col, [1, -1])
            rnorms.append(irnorm)

        return tf.concat(rnorms, axis=0)


class AxisExpModel(ParamModel):
    """
    Per-(process, cell) exponential background param model.

    For each process in proc_spec and each bin of the cell axes, assigns two
    independent parameters (lnAmpl, slope).  In compute() produces::

        rnorm = exp(lnAmpl_ijk + slope_ijk · x_m)

    where x_m is the normalized center of shape-axis bin m (range [0, 1]).
    Both parameters are unconstrained reals (allowNegativeParam always True):
      lnAmpl controls the per-cell log-amplitude (exp(lnAmpl) is the yield at x=0).
      slope < 0 gives a falling exponential, slope = 0 is flat, slope > 0 is rising.
    The flat-background case (slope = 0) is an interior point, so the Hessian is
    non-degenerate there.  All other channels and processes are left at 1.0.

    Usage::

        --paramModel AxisExpModel <channel> <proc_spec> <shape_axis> <cell_axes>

    Example::

        --paramModel AxisExpModel btojpsik_stuff bkgExp \\
            bkmm_jpsimc_mass \\
            bkmm_kaon_pt,bkmm_kaon_eta,bkmm_kaon_charge
    """

    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        if len(args) not in (4, 5):
            raise ValueError(
                f"AxisExpModel requires 4 or 5 positional arguments "
                f"(channel, proc_spec, shape_axis, cell_axes[, slope_axes]) "
                f"but got {len(args)}: {args}"
            )
        channel, proc_spec, shape_axis, cell_axes_csv = args[:4]
        slope_axes_csv = args[4] if len(args) == 5 else None
        return cls(
            indata, channel, proc_spec, shape_axis, cell_axes_csv,
            slope_axes_csv=slope_axes_csv, **kwargs
        )

    def __init__(
        self,
        indata,
        channel,
        proc_spec,
        shape_axis,
        cell_axes_csv,
        slope_axes_csv=None,
        expectSignal=None,
        allowNegativeParam=False,
        **kwargs,
    ):
        self.indata = indata

        if channel not in indata.channel_info:
            raise ValueError(
                f"Channel '{channel}' not found in tensor. "
                f"Available: {list(indata.channel_info.keys())}"
            )
        self.channel = channel
        channel_axes = indata.channel_info[channel]["axes"]
        axis_by_name = {a.name: a for a in channel_axes}

        if shape_axis not in axis_by_name:
            raise ValueError(
                f"Shape axis '{shape_axis}' not found in channel '{channel}'. "
                f"Available: {list(axis_by_name.keys())}"
            )

        cell_names = [n.strip() for n in cell_axes_csv.split(",")]
        for name in cell_names:
            if name not in axis_by_name:
                raise ValueError(
                    f"Cell axis '{name}' not found in channel '{channel}'. "
                    f"Available: {list(axis_by_name.keys())}"
                )
            if name == shape_axis:
                raise ValueError(
                    f"Axis '{name}' appears in both shape_axis and cell_axes."
                )
        self.cell_axis_names = set(cell_names)
        self.cell_axes = [axis_by_name[n] for n in cell_names]
        self.shape_axis = shape_axis

        # Slope axes: subset of cell axes; default = all cell axes (per-cell slopes)
        if slope_axes_csv is None:
            slope_names = cell_names
        else:
            slope_names = [n.strip() for n in slope_axes_csv.split(",")]
            bad = [n for n in slope_names if n not in self.cell_axis_names]
            if bad:
                raise ValueError(
                    f"Slope axes {bad} are not in cell_axes '{cell_axes_csv}'. "
                    f"Slope axes must be a subset of cell axes."
                )
        self.slope_axis_names = set(slope_names)
        self.slope_axes = [axis_by_name[n] for n in slope_names]
        slope_shape = [a.size for a in self.slope_axes]
        self.n_slope_groups = int(np.prod(slope_shape))

        if proc_spec == "all":
            target_encoded = list(indata.procs)
        else:
            target_encoded = []
            for name in [p.strip() for p in proc_spec.split(",")]:
                encoded = name.encode() if isinstance(name, str) else name
                if encoded not in indata.procs:
                    raise ValueError(
                        f"Process '{name}' not found in tensor. "
                        f"Available: {[p.decode() if isinstance(p, bytes) else p for p in indata.procs]}"
                    )
                target_encoded.append(encoded)
        self.proc_idxs = [int(np.where(indata.procs == p)[0][0]) for p in target_encoded]

        cell_shape = [a.size for a in self.cell_axes]
        self.n_cell = int(np.prod(cell_shape))
        self.npoi = len(self.proc_idxs) * (self.n_cell + self.n_slope_groups)

        names = []
        for proc_encoded in target_encoded:
            proc_name = proc_encoded.decode() if isinstance(proc_encoded, bytes) else str(proc_encoded)
            for idxs in itertools.product(*[range(s) for s in cell_shape]):
                label = "_".join(f"{a.name}{i}" for a, i in zip(self.cell_axes, idxs))
                names.append(f"lnAmpl_{proc_name}_{label}".encode())
            for idxs in itertools.product(*[range(s) for s in slope_shape]):
                label = "_".join(f"{a.name}{i}" for a, i in zip(self.slope_axes, idxs))
                names.append(f"slope_{proc_name}_{label}".encode())
        self.params = np.array(names)

        # Normalized shape-axis bin centers in [0, 1]
        centers = np.asarray(axis_by_name[shape_axis].centers, dtype=np.float32)
        span = max(float(centers[-1] - centers[0]), 1e-6)
        x_m = (centers - centers[0]) / span
        self.x_m = tf.constant(x_m, dtype=indata.dtype)

        # Reshape helpers built from channel axis ordering
        full_shape = [a.size for a in channel_axes]
        self.full_shape = full_shape
        self.cell_reshape = [
            a.size if a.name in self.cell_axis_names else 1 for a in channel_axes
        ]
        self.slope_cell_reshape = [
            a.size if a.name in self.slope_axis_names else 1 for a in channel_axes
        ]
        self.shape_reshape = [
            a.size if a.name == shape_axis else 1 for a in channel_axes
        ]

        # Always unconstrained: exp(lnAmpl + slope*x) is positive for any real (lnAmpl, slope).
        self.npou = 0
        self.allowNegativeParam = True
        self.is_linear = False
        # Default: lnAmpl=0 → amplitude=1, slope=0 → flat shape.
        self.xparamdefault = tf.zeros([self.npoi], dtype=indata.dtype)

    def compute(self, param, full=False):
        x_reshaped = tf.reshape(self.x_m, self.shape_reshape)

        rnorms = []
        for k, v in self.indata.channel_info.items():
            if v["masked"] and not full:
                continue
            nbins_channel = int(np.prod([a.size for a in v["axes"]]))
            irnorm = tf.ones([nbins_channel, self.indata.nproc], dtype=self.indata.dtype)
            if k == self.channel:
                for i, proc_idx in enumerate(self.proc_idxs):
                    stride = self.n_cell + self.n_slope_groups
                    a_poi = param[i * stride : i * stride + self.n_cell]
                    b_poi = param[i * stride + self.n_cell : (i + 1) * stride]
                    a = tf.reshape(a_poi, self.cell_reshape)
                    b = tf.reshape(b_poi, self.slope_cell_reshape)
                    scaling = tf.reshape(
                        tf.broadcast_to(tf.exp(a + b * x_reshaped), self.full_shape),
                        [-1, 1],
                    )
                    proc_col = tf.one_hot(proc_idx, self.indata.nproc, dtype=self.indata.dtype)
                    irnorm = irnorm + (scaling - 1.0) * tf.reshape(proc_col, [1, -1])
            rnorms.append(irnorm)

        return tf.concat(rnorms, axis=0)


class AxisBernsteinModel(ParamModel):
    """
    Per-(process, cell) first-order Bernstein background param model.

    For each process in proc_spec and each cell, assigns two non-negative
    parameters (c0, c1).  In compute() produces::

        rnorm(x_m) = c0 · (1 − x_m) + c1 · x_m

    where x_m is the normalized center of shape-axis bin m (range [0, 1]).
    c0 is the relative rate at the low edge of the mass window; c1 at the high
    edge.  Non-negativity is enforced via softplus applied inside compute().
    Default c0=c1=1 gives a flat unit background.
    All other channels and processes are left at 1.0.

    Usage::

        --paramModel AxisBernsteinModel <channel> <proc_spec> <shape_axis> <cell_axes>

    Example::

        --paramModel AxisBernsteinModel btojpsik_stuff bkgBernstein \\
            bkmm_jpsimc_mass \\
            bkmm_kaon_pt,bkmm_kaon_eta,bkmm_kaon_charge
    """

    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        if len(args) != 4:
            raise ValueError(
                f"AxisBernsteinModel requires exactly 4 positional arguments "
                f"(channel, proc_spec, shape_axis, cell_axes) but got {len(args)}: {args}"
            )
        channel, proc_spec, shape_axis, cell_axes_csv = args
        return cls(indata, channel, proc_spec, shape_axis, cell_axes_csv, **kwargs)

    def __init__(
        self,
        indata,
        channel,
        proc_spec,
        shape_axis,
        cell_axes_csv,
        expectSignal=None,
        allowNegativeParam=False,
        **kwargs,
    ):
        self.indata = indata

        if channel not in indata.channel_info:
            raise ValueError(
                f"Channel '{channel}' not found in tensor. "
                f"Available: {list(indata.channel_info.keys())}"
            )
        self.channel = channel
        channel_axes = indata.channel_info[channel]["axes"]
        axis_by_name = {a.name: a for a in channel_axes}

        if shape_axis not in axis_by_name:
            raise ValueError(
                f"Shape axis '{shape_axis}' not found in channel '{channel}'. "
                f"Available: {list(axis_by_name.keys())}"
            )

        cell_names = [n.strip() for n in cell_axes_csv.split(",")]
        for name in cell_names:
            if name not in axis_by_name:
                raise ValueError(
                    f"Cell axis '{name}' not found in channel '{channel}'. "
                    f"Available: {list(axis_by_name.keys())}"
                )
            if name == shape_axis:
                raise ValueError(
                    f"Axis '{name}' appears in both shape_axis and cell_axes."
                )
        self.cell_axis_names = set(cell_names)
        self.cell_axes = [axis_by_name[n] for n in cell_names]
        self.shape_axis = shape_axis

        if proc_spec == "all":
            target_encoded = list(indata.procs)
        else:
            target_encoded = []
            for name in [p.strip() for p in proc_spec.split(",")]:
                encoded = name.encode() if isinstance(name, str) else name
                if encoded not in indata.procs:
                    raise ValueError(
                        f"Process '{name}' not found in tensor. "
                        f"Available: {[p.decode() if isinstance(p, bytes) else p for p in indata.procs]}"
                    )
                target_encoded.append(encoded)
        self.proc_idxs = [int(np.where(indata.procs == p)[0][0]) for p in target_encoded]

        cell_shape = [a.size for a in self.cell_axes]
        self.n_cell = int(np.prod(cell_shape))
        self.npoi = len(self.proc_idxs) * 2 * self.n_cell

        names = []
        for proc_encoded in target_encoded:
            proc_name = proc_encoded.decode() if isinstance(proc_encoded, bytes) else str(proc_encoded)
            for prefix in ("c0", "c1"):
                for idxs in itertools.product(*[range(s) for s in cell_shape]):
                    label = "_".join(f"{a.name}{i}" for a, i in zip(self.cell_axes, idxs))
                    names.append(f"{prefix}_{proc_name}_{label}".encode())
        self.params = np.array(names)

        # Normalized shape-axis bin centers in [0, 1]
        centers = np.asarray(axis_by_name[shape_axis].centers, dtype=np.float32)
        span = max(float(centers[-1] - centers[0]), 1e-6)
        x_m = (centers - centers[0]) / span
        self.x_m = tf.constant(x_m, dtype=indata.dtype)

        # Reshape helpers built from channel axis ordering
        full_shape = [a.size for a in channel_axes]
        self.full_shape = full_shape
        self.cell_reshape = [
            a.size if a.name in self.cell_axis_names else 1 for a in channel_axes
        ]
        self.shape_reshape = [
            a.size if a.name == shape_axis else 1 for a in channel_axes
        ]

        # Non-negative Bernstein coefficients via softplus: c = log(1 + exp(x)).
        # softplus is always > 0 and has nonzero gradient everywhere, unlike x²
        # which has zero gradient at x=0 and produces a degenerate (zero-pivot)
        # Hessian when background → 0 at best fit (boundary of parameter space).
        # allowNegativeParam=True tells the fitter to pass raw x; softplus
        # is applied inside compute() below.
        # Default x = softplus_inv(1) ≈ 0.5413 so c starts at 1 (flat background).
        self.npou = 0
        self.allowNegativeParam = True
        self.is_linear = False
        if expectSignal is not None:
            raise ValueError(
                "AxisBernsteinModel does not support expectSignal; "
                "set initial Bernstein coefficients via --expectSignal on another model."
            )
        _softplus_inv_1 = float(np.log(np.exp(1.0) - 1.0))  # ≈ 0.5413
        self.xparamdefault = tf.constant(
            _softplus_inv_1 * np.ones(self.npoi), dtype=self.indata.dtype
        )

    def compute(self, param, full=False):
        x_reshaped = tf.reshape(self.x_m, self.shape_reshape)

        rnorms = []
        for k, v in self.indata.channel_info.items():
            if v["masked"] and not full:
                continue
            nbins_channel = int(np.prod([a.size for a in v["axes"]]))
            irnorm = tf.ones([nbins_channel, self.indata.nproc], dtype=self.indata.dtype)
            if k == self.channel:
                for i, proc_idx in enumerate(self.proc_idxs):
                    c0_poi = param[i * 2 * self.n_cell : i * 2 * self.n_cell + self.n_cell]
                    c1_poi = param[i * 2 * self.n_cell + self.n_cell : (i + 1) * 2 * self.n_cell]
                    c0 = tf.reshape(tf.nn.softplus(c0_poi), self.cell_reshape)
                    c1 = tf.reshape(tf.nn.softplus(c1_poi), self.cell_reshape)
                    scaling = tf.reshape(
                        tf.broadcast_to(
                            c0 * (1.0 - x_reshaped) + c1 * x_reshaped,
                            self.full_shape,
                        ),
                        [-1, 1],
                    )
                    proc_col = tf.one_hot(proc_idx, self.indata.nproc, dtype=self.indata.dtype)
                    irnorm = irnorm + (scaling - 1.0) * tf.reshape(proc_col, [1, -1])
            rnorms.append(irnorm)

        return tf.concat(rnorms, axis=0)
