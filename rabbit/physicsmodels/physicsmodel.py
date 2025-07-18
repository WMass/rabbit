import tensorflow as tf

from rabbit.physicsmodels import helpers


class PhysicsModel:
    """
    Processing the flat input vector. can be used to inherit custom physics models from.
    """

    need_observables = True  # if observables should be provided to the compute function
    has_data = True  # if data histograms are stored or not, and if chi2 is calculated

    skip_prefit = False  # if the physics model doesn't implement the prefit case
    skip_incusive = False  # if the physics model doesn't implement compute_flat
    skip_per_process = (
        False  # if the physics model doesn't implement compute_flat_per_process
    )
    need_processes = False  # if observables are required by process in compute_flat

    ndf_reduction = 0  # how much will be subtracted from the ndf / number of bins, e.g. for chi2 calculation

    def __init__(self, indata, key):
        # The result of a model in the output dictionary is stored under 'result = fitresult[cls.name]'
        #   if the model can have different instances 'self.instance' must be set to a unique string and the result will be stored in 'result = result[self.instance]'
        #   each model can have different channels that are the same or different from channels from the input data. All channel specific results are stored under 'result["channels"]'
        self.key = (
            key  # where to store the results of this model in the results dictionary
        )

    # class function to parse strings as given by the argparse input e.g. -m PhysicsModel <arg[0]> <args[1]> ...
    @classmethod
    def parse_args(cls, indata, *args):
        key = " ".join([cls.__name__, *args])
        return cls(indata, key, *args)

    # function to compute the transformation of the physics model, has to be differentiable.
    #    For custom physics models, this function should be overridden.
    #    observables are the provided histograms inclusive in processes: nbins unless 'need_processes=True'
    #    params are the fit parameters
    def compute_flat(self, params, observables=None):
        return observables

    # function to compute the transformation of the physics model, has to be differentiable.
    #    For custom physics models, this function can be overridden.
    #    observables are the provided histograms per process: nbins x nprocesses and the result is expected to be nbins x nprocesses
    #    params are the fit parameters
    def compute_flat_per_process(self, params, observables=None):
        return self.compute_flat(params, observables)

    # generic version which should not need to be overridden
    @tf.function
    def get_data(self, *args, **kwargs):
        return self._get_data(*args, **kwargs)

    def _get_data(self, data, data_cov_inv=None):
        with tf.GradientTape() as t:
            t.watch(data)
            output = self.compute_flat(None, data)

        jacobian = t.jacobian(output, data)

        # Ensure the Jacobian has at least 2 dimensions (expand in case output is a scalar)
        if len(jacobian.shape) == 1:
            jacobian = tf.expand_dims(jacobian, axis=0)

        if data_cov_inv is None:
            # Assume poisson uncertainties on data
            cov_output = (jacobian * data) @ tf.transpose(jacobian)
        else:
            # General case with full covariance matrix
            # the following is equivalent to, but faster than: cov_output = jacobian @ tf.linalg.inv(data_cov_inv) @ tf.transpose(jacobian)
            cov_output = jacobian @ tf.linalg.solve(
                data_cov_inv, tf.transpose(jacobian)
            )

        variances_output = tf.linalg.diag_part(cov_output)

        return output, variances_output, cov_output


class CompositeModel(PhysicsModel):
    """
    A composition of different physics models e.g. to compute the covariance across them
    """

    def __init__(
        self,
        models,
    ):
        self.key = self.__class__.__name__

        self.channel_info = {}

        self.models = models

        self.ndf_reduction = 0
        self.has_data = True

        self.need_processes = False
        self.skip_per_process = False

        # make a composite model with unique channel names
        for m in models:
            for k, c in m.channel_info.items():
                self.channel_info[f"{m.key}_{k}"] = c

            self.ndf_reduction += m.ndf_reduction
            # if any of the submodels does not process data, the composite model also does not
            if not m.has_data:
                self.has_data = False
            # if any of the submodels needs processes, the composite model also needs processes
            if m.need_processes:
                self.need_processes = True
            # if any of the submodels skips the processes step, the composite model also skip the processes
            if m.skip_per_process:
                self.skip_per_process = True

    def compute_flat(self, params, observables=None):
        exp_list = []
        for m in self.models:
            if self.need_processes and not m.need_processes:
                # in case the composite model requires processes but one of the submodels does not, sum them up again
                observables = tf.reduce_sum(observables, axis=1)
            result = m.compute_flat(params, observables)
            exp_list.append(result)

        exp = tf.concat(exp_list, axis=0)
        return exp

    def compute_flat_per_process(self, params, observables=None):
        exp = tf.concat(
            [m.compute_flat_per_process(params, observables) for m in self.models],
            axis=0,
        )
        return exp


class Basemodel(PhysicsModel):
    """
    A class to output histograms without any transformation, can be used as base class to inherit custom physics models from.
    """

    def __init__(self, indata, key):
        super().__init__(indata, key)
        self.channel_info = indata.channel_info
        for i, c in self.channel_info.items():
            c["processes"] = indata.procs


class Channelmodel(PhysicsModel):
    """
    Abstract physics model to process a specific channel
    """

    def __init__(
        self,
        indata,
        key,
        channel,
        processes=[],
        **kwargs,
    ):
        super().__init__(indata, key)

        self.term = helpers.Term(indata, channel, processes, **kwargs)

        channel_info = indata.channel_info[channel]

        self.channel_info = {
            channel: {
                "axes": self.term.channel_axes,
                "flow": channel_info.get("flow", False),
                "processes": processes if len(processes) else indata.procs,
            }
        }

        self.has_data = not channel_info["masked"]

    def compute(self, params, observables):
        return observables

    def compute_per_process(self, params, observables):
        return self.compute(params, observables)

    def compute_flat(self, params, observables):
        exp = self.term.select(observables, inclusive=True)
        exp = self.compute(params, exp)
        exp = tf.reshape(exp, [-1])  # flatten again
        return exp

    def compute_flat_per_process(self, params, observables):
        exp = self.term.select(observables, inclusive=False)
        exp = self.compute_per_process(params, exp)
        # flatten again
        flat_shape = (-1, exp.shape[-1])
        exp = tf.reshape(exp, flat_shape)
        return exp


class Select(Channelmodel):
    """
    A class to output histograms without any transformation for a given channel, can be used as base class to inherit custom physics models from.
    """

    def __init__(
        self,
        indata,
        key,
        *args,
        **kwargs,
    ):
        super().__init__(indata, key, *args, **kwargs)

    @classmethod
    def parse_args(cls, indata, channel, *args):
        """
        parsing the input arguments into the ratio constructor, is has to be called as
        -m Select <ch num>
            <proc_0>,<proc_1>,...
            <axis_0>:<selection_0>,<axis_1>,<selection_1>...

        Processes selections are optional.
        Axes selections are optional.
        """

        if len(args) and ":" not in args[0]:
            procs = [p for p in args[0].split(",") if p != "None"]
        else:
            procs = []

        # find axis selections
        if any(":" in a for a in args):
            sel_args = [a for a in args if ":" in a][0]
        else:
            sel_args = "None:None"

        axis_selection, axes_rebin, axes_sum = helpers.parse_axis_selection(sel_args)

        key = " ".join([cls.__name__, *args])

        return cls(
            indata,
            key,
            channel,
            procs,
            selections=axis_selection,
            rebin_axes=axes_rebin,
            sum_axes=axes_sum,
        )

    def compute(self, params, observables):
        return observables

    def compute_per_process(self, params, observables):
        return self.compute(params, observables)
