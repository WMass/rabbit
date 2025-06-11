import math
import os
from collections import defaultdict

import h5py
import numpy as np

from combinetf2 import common, h5pyutils

from wums import ioutils, logging, output_tools  # isort: skip

logger = logging.child_logger(__name__)


class TensorWriter:
    def __init__(
        self,
        sparse=False,
        systematic_type="log_normal",
        allow_negative_expectation=False,
        add_bin_by_bin_stat_to_data_cov=False,
    ):
        self.allow_negative_expectation = allow_negative_expectation

        self.systematic_type = systematic_type

        self.symmetric_tensor = True  # If all shape systematics are symmetrized the systematic tensor is symmetric leading to reduced memory and improved efficiency
        self.add_bin_by_bin_stat_to_data_cov = add_bin_by_bin_stat_to_data_cov  # add bin by bin statistical uncertainty to data covariance matrix

        self.signals = set()
        self.bkgs = set()

        self.channels = {}
        self.nbinschan = {}
        self.pseudodata_names = set()

        self.dict_noigroups = defaultdict(lambda: set())
        self.dict_systgroups = defaultdict(lambda: set())

        self.systsstandard = set()
        self.systsnoi = set()
        self.systsnoconstraint = set()
        self.systsnoprofile = set()
        self.systscovariance = set()

        self.sparse = sparse
        self.idxdtype = "int64"

        # temporary data
        self.dict_data_obs = {}  # [channel]
        self.data_covariance = None
        self.dict_pseudodata = {}  # [channel][pseudodata]
        self.dict_norm = {}  # [channel][process]
        self.dict_sumw2 = {}  # [channel]
        self.dict_logkavg = {}  # [channel][proc][syst]
        self.dict_logkhalfdiff = {}  # [channel][proc][syst]
        self.dict_logkavg_indices = {}
        self.dict_logkhalfdiff_indices = {}

        self.clipSystVariations = False
        if self.clipSystVariations > 0.0:
            self.clip = np.abs(np.log(self.clipSystVariations))

        self.logkepsilon = math.log(
            1e-3
        )  # numerical cutoff in case of zeros in systematic variations

        # settings for writing out hdf5 files
        self.dtype = "float64"
        self.chunkSize = 4 * 1024**2

    def get_flat_values(self, h):
        if hasattr(h, "values"):
            values = h.values()
        else:
            values = h
        return values.flatten().astype(self.dtype)

    def get_flat_variances(self, h):
        if hasattr(h, "values"):
            variances = h.variances()
        else:
            variances = h
        return variances.flatten().astype(self.dtype)

    def add_data(self, h, channel="ch0"):
        self._check_hist_and_channel(h, channel)
        if channel in self.dict_data_obs.keys():
            raise RuntimeError(f"Data histogram for channel '{channel}' already set.")
        self.dict_data_obs[channel] = self.get_flat_values(h)

    def add_data_covariance(self, cov):
        self.data_covariance = cov if isinstance(cov, np.ndarray) else cov.values()

    def add_pseudodata(self, h, name=None, channel="ch0"):
        self._check_hist_and_channel(h, channel)
        if name is None:
            name = f"pseudodata_{len(self.pseudodata_names)}"
        self.pseudodata_names.add(name)
        if channel not in self.dict_pseudodata.keys():
            self.dict_pseudodata[channel] = {}
        if name in self.dict_pseudodata[channel].keys():
            raise RuntimeError(
                f"Pseudodata histogram '{name}' for channel '{channel}' already set."
            )
        self.dict_pseudodata[channel][name] = self.get_flat_values(h)

    def add_process(self, h, name, channel="ch0", signal=False, variances=None):
        self._check_hist_and_channel(h, channel)

        if name in self.dict_norm[channel].keys():
            raise RuntimeError(
                f"Nominal histogram for process '{name}' for channel '{channel}' already set."
            )

        if signal:
            self.signals.add(name)
        else:
            self.bkgs.add(name)

        self.dict_logkavg[channel][name] = {}
        self.dict_logkhalfdiff[channel][name] = {}
        if self.sparse:
            self.dict_logkavg_indices[channel][name] = {}
            self.dict_logkhalfdiff_indices[channel][name] = {}

        norm = self.get_flat_values(h)
        sumw2 = self.get_flat_variances(h if variances is None else variances)

        if not self.allow_negative_expectation:
            norm = np.maximum(norm, 0.0)
        if not np.all(np.isfinite(sumw2)):
            raise RuntimeError(
                f"{len(sumw2)-sum(np.isfinite(sumw2))} NaN or Inf values encountered in variances for {name}!"
            )
        if not np.all(np.isfinite(norm)):
            raise RuntimeError(
                f"{len(norm)-sum(np.isfinite(norm))} NaN or Inf values encountered in nominal histogram for {name}!"
            )

        self.dict_norm[channel][name] = norm
        self.dict_sumw2[channel] += sumw2

    def add_channel(self, axes, name=None, masked=False):
        if name is None:
            name = f"ch{len(self.channels)}"
        logger.debug(f"Add new channel {name}")
        ibins = np.prod([len(a) for a in axes])
        self.nbinschan[name] = ibins
        self.dict_norm[name] = {}
        self.dict_sumw2[name] = np.zeros(ibins)

        # add masked channels last and not masked channels first
        this_channel = {"axes": [a for a in axes], "masked": masked}
        if masked:
            self.channels[name] = this_channel
        else:
            self.channels = {name: this_channel, **self.channels}

        self.dict_logkavg[name] = {}
        self.dict_logkhalfdiff[name] = {}
        if self.sparse:
            self.dict_logkavg_indices[name] = {}
            self.dict_logkhalfdiff_indices[name] = {}

    def _check_hist_and_channel(self, h, channel):

        if channel not in self.channels.keys():
            raise RuntimeError(f"Channel {channel} not known!")

        if hasattr(h, "axes"):
            axes = [a for a in h.axes]
            channel_axes = self.channels[channel]["axes"]

            if not all(np.allclose(a, axes[i]) for i, a in enumerate(channel_axes)):
                raise RuntimeError(
                    f"""
                    Histogram axes different have different edges from channel axes of channel {channel}
                    \nHistogram axes: {[a.edges for a in axes]}
                    \nChannel axes: {[a.edges for a in channel_axes]}
                    """
                )
        else:
            shape_in = h.shape
            shape_this = tuple([len(a) for a in self.channels[channel]["axes"]])
            if shape_in != shape_this:
                raise RuntimeError(
                    f"Shape of input object different from channel axes '{shape_in}' != '{shape_this}'"
                )

    def _compute_asym_syst(
        self,
        logkup,
        logkdown,
        name,
        process,
        channel,
        symmetrize="average",
        add_to_data_covariance=False,
        **kargs,
    ):
        var_name_out = name

        if symmetrize == "conservative":
            # symmetrize by largest magnitude of up and down variations
            logkavg_proc = np.where(
                np.abs(logkup) > np.abs(logkdown),
                logkup,
                logkdown,
            )
        elif symmetrize == "average":
            # symmetrize by average of up and down variations
            logkavg_proc = 0.5 * (logkup + logkdown)
        elif symmetrize in ["linear", "quadratic"]:
            # "linear" corresponds to a piecewise linear dependence of logk on theta
            # while "quadratic" corresponds to a quadratic dependence and leads
            # to a large variance
            diff_fact = np.sqrt(3.0) if symmetrize == "quadratic" else 1.0

            # split asymmetric variation into two symmetric variations
            logkavg_proc = 0.5 * (logkup + logkdown)
            logkdiffavg_proc = 0.5 * diff_fact * (logkup - logkdown)

            var_name_out = name + "SymAvg"
            var_name_out_diff = name + "SymDiff"

            # special case, book the extra systematic
            self.book_logk_avg(logkdiffavg_proc, channel, process, var_name_out_diff)
            self.book_systematic(
                var_name_out_diff,
                add_to_data_covariance=add_to_data_covariance,
                **kargs,
            )
        else:
            if add_to_data_covariance:
                raise RuntimeError(
                    "add_to_data_covariance requires symmetric uncertainties"
                )

            self.symmetric_tensor = False

            logkavg_proc = 0.5 * (logkup + logkdown)
            logkhalfdiff_proc = 0.5 * (logkup - logkdown)

            self.book_logk_halfdiff(logkhalfdiff_proc, channel, process, name)
        logkup = None
        logkdown = None

        return logkavg_proc, var_name_out

    def add_lnN_systematic(
        self,
        name,
        process,
        channel,
        uncertainty,
        profile=True,
        add_to_data_covariance=False,
        groups=None,
        symmetrize="average",
        **kargs,
    ):
        if not isinstance(process, (list, tuple, np.ndarray)):
            process = [process]

        if not isinstance(uncertainty, (list, tuple, np.ndarray)):
            uncertainty = [uncertainty]

        if len(uncertainty) != 1 and len(process) != len(uncertainty):
            raise RuntimeError(
                f"uncertainty must be either a scalar or list with the same length as the list of processes but len(process) = {len(process)} and len(uncertainty) = {len(uncertainty)}"
            )

        var_name_out = name

        systematic_type = "normal" if add_to_data_covariance else self.systematic_type

        for p, u in zip(process, uncertainty):
            norm = self.dict_norm[channel][p]
            if isinstance(u, (list, tuple, np.ndarray)):
                if len(u) != 2:
                    raise RuntimeError(
                        f"lnN uncertainty can only be a scalar for a symmetric or a list of 2 elements for asymmetric lnN uncertainties, but got a list of {len(u)} elements"
                    )
                # asymmetric lnN uncertainty
                syst_up = norm * u[0]
                syst_down = norm * u[1]

                logkup_proc = self.get_logk(
                    syst_up, norm, systematic_type=systematic_type
                )
                logkdown_proc = -self.get_logk(
                    syst_down, norm, systematic_type=systematic_type
                )

                logkavg_proc, var_name_out = self._compute_asym_syst(
                    logkup_proc,
                    logkdown_proc,
                    name,
                    process,
                    channel,
                    symmetrize=symmetrize,
                    add_to_data_covariance=add_to_data_covariance,
                    **kargs,
                )
            else:
                syst = norm * u
                logkavg_proc = self.get_logk(
                    syst, norm, systematic_type=systematic_type
                )

            self.book_logk_avg(logkavg_proc, channel, p, var_name_out)

        self.book_systematic(
            var_name_out,
            groups=groups,
            profile=profile,
            add_to_data_covariance=add_to_data_covariance,
            **kargs,
        )

    def add_systematic(
        self,
        h,
        name,
        process,
        channel,
        kfactor=1,
        mirror=True,
        symmetrize="average",
        add_to_data_covariance=False,
        **kargs,
    ):
        """
        h: either a single histogram with the systematic variation if mirror=True or a list of two histograms with the up and down variation
        """

        norm = self.dict_norm[channel][process]

        var_name_out = name

        systematic_type = "normal" if add_to_data_covariance else self.systematic_type

        if isinstance(h, (list, tuple, np.ndarray)):
            self._check_hist_and_channel(h[0], channel)
            self._check_hist_and_channel(h[1], channel)

            syst_up = self.get_flat_values(h[0])
            syst_down = self.get_flat_values(h[1])

            logkup_proc = self.get_logk(
                syst_up, norm, kfactor, systematic_type=systematic_type
            )
            logkdown_proc = -self.get_logk(
                syst_down, norm, kfactor, systematic_type=systematic_type
            )

            logkavg_proc, var_name_out = self._compute_asym_syst(
                logkup_proc,
                logkdown_proc,
                name,
                process,
                channel,
                symmetrize,
                add_to_data_covariance,
                **kargs,
            )
        elif mirror:
            self._check_hist_and_channel(h, channel)
            syst = self.get_flat_values(h)
            logkavg_proc = self.get_logk(
                syst, norm, kfactor, systematic_type=systematic_type
            )
        else:
            raise RuntimeError(
                "Only one histogram given but mirror=False, can not construct a variation"
            )

        self.book_logk_avg(logkavg_proc, channel, process, var_name_out)
        self.book_systematic(
            var_name_out, add_to_data_covariance=add_to_data_covariance, **kargs
        )

    def get_logk(self, syst, norm, kfac=1.0, systematic_type=None):
        if not np.all(np.isfinite(syst)):
            raise RuntimeError(
                f"{len(syst)-sum(np.isfinite(syst))} NaN or Inf values encountered in systematic!"
            )

        # TODO clean this up and avoid duplication
        if systematic_type == "log_normal":
            # check if there is a sign flip between systematic and nominal
            _logk = kfac * np.log(syst / norm)
            _logk_view = np.where(
                np.equal(np.sign(norm * syst), 1),
                _logk,
                self.logkepsilon * np.ones_like(_logk),
            )

            # FIXME does this actually take effect since _logk_view is normally returned?
            if self.clipSystVariations > 0.0:
                _logk = np.clip(_logk, -self.clip, self.clip)

            return _logk_view
        elif systematic_type == "normal":
            _logk = kfac * (syst - norm)
            return _logk
        else:
            raise RuntimeError(
                f"Invalid systematic_type {systematic_type}, valid choices are 'log_normal' or 'normal'"
            )

    def book_logk_avg(self, *args):
        self.book_logk(
            self.dict_logkavg,
            self.dict_logkavg_indices,
            *args,
        )

    def book_logk_halfdiff(self, *args):
        self.book_logk(
            self.dict_logkhalfdiff,
            self.dict_logkhalfdiff_indices,
            *args,
        )

    def book_logk(
        self,
        dict_logk,
        dict_logk_indices,
        logk,
        channel,
        process,
        syst_name,
    ):
        norm = self.dict_norm[channel][process]
        # ensure that systematic tensor is sparse where normalization matrix is sparse
        logk = np.where(np.equal(norm, 0.0), 0.0, logk)
        if self.sparse:
            indices = np.transpose(np.nonzero(logk))
            dict_logk_indices[channel][process][syst_name] = indices
            dict_logk[channel][process][syst_name] = np.reshape(logk[indices], [-1])
        else:
            dict_logk[channel][process][syst_name] = logk

    def book_systematic(
        self,
        name,
        profile=True,
        noi=False,
        constrained=True,
        add_to_data_covariance=False,
        groups=None,
    ):

        if add_to_data_covariance:
            self.systscovariance.add(name)
        elif not profile:
            self.systsnoprofile.add(name)
        elif not constrained:
            self.systsnoconstraint.add(name)
        else:
            self.systsstandard.add(name)

        # below only makes sense if this is an explicit nuisance parameter
        if not add_to_data_covariance:
            if groups is None:
                groups = [name]

            if noi:
                target_dict = self.dict_noigroups
            else:
                target_dict = self.dict_systgroups

            for group in groups:
                target_dict[group].add(name)

    def write(self, outfolder="./", outfilename="combinetf2_input.hdf5", args={}):

        if self.signals.intersection(self.bkgs):
            raise RuntimeError(
                f"Processes '{self.signals.intersection(self.bkgs)}' found as signal and background"
            )

        procs = sorted(list(self.signals)) + sorted(list(self.bkgs))
        nproc = len(procs)

        nbins = sum(
            [v for c, v in self.nbinschan.items() if not self.channels[c]["masked"]]
        )
        # nbinsfull including masked channels
        nbinsfull = sum([v for v in self.nbinschan.values()])

        logger.info(f"Write out nominal arrays")
        sumw = np.zeros([nbinsfull], self.dtype)
        sumw2 = np.zeros([nbinsfull], self.dtype)
        data_obs = np.zeros([nbins], self.dtype)
        pseudodata = np.zeros([nbins, len(self.pseudodata_names)], self.dtype)
        ibin = 0
        for chan, chan_info in self.channels.items():
            nbinschan = self.nbinschan[chan]

            sumw2[ibin : ibin + nbinschan] = self.dict_sumw2[chan]

            for iproc, proc in enumerate(procs):
                if proc not in self.dict_norm[chan]:
                    continue

                sumw[ibin : ibin + nbinschan] += self.dict_norm[chan][proc]

            if not chan_info["masked"]:
                data_obs[ibin : ibin + nbinschan] = self.dict_data_obs[chan]

                for idx, name in enumerate(self.pseudodata_names):
                    pseudodata[ibin : ibin + nbinschan, idx] = self.dict_pseudodata[
                        chan
                    ][name]

            ibin += nbinschan

        systs = self.get_systs()
        nsyst = len(systs)

        if self.symmetric_tensor:
            logger.info("No asymmetric systematics - write fully symmetric tensor")

        ibin = 0
        if self.sparse:
            logger.info(f"Write out sparse array")
            norm_sparse_size = 0
            norm_sparse_indices = np.zeros([norm_sparse_size, 2], self.idxdtype)
            norm_sparse_values = np.zeros([norm_sparse_size], self.dtype)

            logk_sparse_size = 0
            logk_sparse_normindices = np.zeros([logk_sparse_size, 1], self.idxdtype)
            logk_sparse_systindices = np.zeros([logk_sparse_size, 1], self.idxdtype)
            logk_sparse_values = np.zeros([logk_sparse_size], self.dtype)

            for chan in self.channels.keys():
                nbinschan = self.nbinschan[chan]
                dict_norm_chan = self.dict_norm[chan]
                dict_logkavg_chan_indices = self.dict_logkavg_indices[chan]
                dict_logkavg_chan_values = self.dict_logkavg[chan]

                for iproc, proc in enumerate(procs):
                    if proc not in dict_norm_chan:
                        continue
                    norm_proc = dict_norm_chan[proc]

                    norm_indices = np.transpose(np.nonzero(norm_proc))
                    norm_values = np.reshape(norm_proc[norm_indices], [-1])

                    nvals = len(norm_values)
                    oldlength = norm_sparse_size
                    norm_sparse_size = oldlength + nvals
                    norm_sparse_indices.resize([norm_sparse_size, 2])
                    norm_sparse_values.resize([norm_sparse_size])

                    out_indices = np.array([[ibin, iproc]]) + np.pad(
                        norm_indices, ((0, 0), (0, 1)), "constant"
                    )
                    norm_indices = None

                    norm_sparse_indices[oldlength:norm_sparse_size] = out_indices
                    out_indices = None

                    norm_sparse_values[oldlength:norm_sparse_size] = norm_values
                    norm_values = None

                    norm_idx_map = (
                        np.cumsum(np.not_equal(norm_proc, 0.0)) - 1 + oldlength
                    )

                    dict_logkavg_proc_indices = dict_logkavg_chan_indices[proc]
                    dict_logkavg_proc_values = dict_logkavg_chan_values[proc]

                    for isyst, syst in enumerate(systs):
                        if syst not in dict_logkavg_proc_indices.keys():
                            continue

                        logkavg_proc_indices = dict_logkavg_proc_indices[syst]
                        logkavg_proc_values = dict_logkavg_proc_values[syst]

                        nvals_proc = len(logkavg_proc_values)
                        oldlength = logk_sparse_size
                        logk_sparse_size = oldlength + nvals_proc
                        logk_sparse_normindices.resize([logk_sparse_size, 1])
                        logk_sparse_systindices.resize([logk_sparse_size, 1])
                        logk_sparse_values.resize([logk_sparse_size])

                        # first dimension of output indices are NOT in the dense [nbin,nproc] space, but rather refer to indices in the norm_sparse vectors
                        # second dimension is flattened in the [2,nsyst] space, where logkavg corresponds to [0,isyst] flattened to isyst
                        # two dimensions are kept in separate arrays for now to reduce the number of copies needed later
                        out_normindices = norm_idx_map[logkavg_proc_indices]
                        logkavg_proc_indices = None

                        logk_sparse_normindices[oldlength:logk_sparse_size] = (
                            out_normindices
                        )
                        logk_sparse_systindices[oldlength:logk_sparse_size] = isyst
                        out_normindices = None

                        logk_sparse_values[oldlength:logk_sparse_size] = (
                            logkavg_proc_values
                        )
                        logkavg_proc_values = None

                        if syst in self.dict_logkhalfdiff_indices[chan][proc].keys():
                            logkhalfdiff_proc_indices = self.dict_logkhalfdiff_indices[
                                chan
                            ][proc][syst]
                            logkhalfdiff_proc_values = self.dict_logkhalfdiff[chan][
                                proc
                            ][syst]

                            nvals_proc = len(logkhalfdiff_proc_values)
                            oldlength = logk_sparse_size
                            logk_sparse_size = oldlength + nvals_proc
                            logk_sparse_normindices.resize([logk_sparse_size, 1])
                            logk_sparse_systindices.resize([logk_sparse_size, 1])
                            logk_sparse_values.resize([logk_sparse_size])

                            # out_indices = np.array([[ibin,iproc,isyst,1]]) + np.pad(logkhalfdiff_proc_indices,((0,0),(0,3)),'constant')
                            # first dimension of output indices are NOT in the dense [nbin,nproc] space, but rather refer to indices in the norm_sparse vectors
                            # second dimension is flattened in the [2,nsyst] space, where logkhalfdiff corresponds to [1,isyst] flattened to nsyst + isyst
                            # two dimensions are kept in separate arrays for now to reduce the number of copies needed later
                            out_normindices = norm_idx_map[logkhalfdiff_proc_indices]
                            logkhalfdiff_proc_indices = None

                            logk_sparse_normindices[oldlength:logk_sparse_size] = (
                                out_normindices
                            )
                            logk_sparse_systindices[oldlength:logk_sparse_size] = (
                                nsyst + isyst
                            )
                            out_normindices = None

                            logk_sparse_values[oldlength:logk_sparse_size] = (
                                logkhalfdiff_proc_values
                            )
                            logkhalfdiff_proc_values = None

                    # free memory
                    dict_logkavg_proc_indices = None
                    dict_logkavg_proc_values = None

                # free memory
                norm_proc = None
                norm_idx_map = None

                ibin += nbinschan

            logger.info(f"Resize and sort sparse arrays into canonical order")
            # resize sparse arrays to actual length
            norm_sparse_indices.resize([norm_sparse_size, 2])
            norm_sparse_values.resize([norm_sparse_size])
            logk_sparse_normindices.resize([logk_sparse_size, 1])
            logk_sparse_systindices.resize([logk_sparse_size, 1])
            logk_sparse_values.resize([logk_sparse_size])

            # straightforward sorting of norm_sparse into canonical order
            norm_sparse_dense_shape = (nbinsfull, nproc)
            norm_sort_indices = np.argsort(
                np.ravel_multi_index(
                    np.transpose(norm_sparse_indices), norm_sparse_dense_shape
                )
            )
            norm_sparse_indices = norm_sparse_indices[norm_sort_indices]
            norm_sparse_values = norm_sparse_values[norm_sort_indices]

            # now permute the indices of the first dimension of logk_sparse corresponding to the resorting of norm_sparse

            # compute the inverse permutation from the sorting of norm_sparse
            # since the final indices are filled from here, need to ensure it has the correct data type
            logk_permute_indices = np.argsort(norm_sort_indices).astype(self.idxdtype)
            norm_sort_indices = None
            logk_sparse_normindices = logk_permute_indices[logk_sparse_normindices]
            logk_permute_indices = None
            logk_sparse_indices = np.concatenate(
                [logk_sparse_normindices, logk_sparse_systindices], axis=-1
            )

            # now straightforward sorting of logk_sparse into canonical order
            if self.symmetric_tensor:
                logk_sparse_dense_shape = (norm_sparse_indices.shape[0], nsyst)
            else:
                logk_sparse_dense_shape = (norm_sparse_indices.shape[0], 2 * nsyst)
            logk_sort_indices = np.argsort(
                np.ravel_multi_index(
                    np.transpose(logk_sparse_indices), logk_sparse_dense_shape
                )
            )
            logk_sparse_indices = logk_sparse_indices[logk_sort_indices]
            logk_sparse_values = logk_sparse_values[logk_sort_indices]
            logk_sort_indices = None

        else:
            logger.info(f"Write out dense array")
            # initialize with zeros, i.e. no variation
            norm = np.zeros([nbinsfull, nproc], self.dtype)
            if self.symmetric_tensor:
                logk = np.zeros([nbinsfull, nproc, nsyst], self.dtype)
            else:
                logk = np.zeros([nbinsfull, nproc, 2, nsyst], self.dtype)

            for chan in self.channels.keys():
                nbinschan = self.nbinschan[chan]
                dict_norm_chan = self.dict_norm[chan]

                for iproc, proc in enumerate(procs):
                    if proc not in dict_norm_chan:
                        continue

                    norm_proc = dict_norm_chan[proc]

                    norm[ibin : ibin + nbinschan, iproc] = norm_proc

                    dict_logkavg_proc = self.dict_logkavg[chan][proc]
                    dict_logkhalfdiff_proc = self.dict_logkhalfdiff[chan][proc]
                    for isyst, syst in enumerate(systs):
                        if syst not in dict_logkavg_proc.keys():
                            continue

                        if self.symmetric_tensor:
                            logk[ibin : ibin + nbinschan, iproc, isyst] = (
                                dict_logkavg_proc[syst]
                            )
                        else:
                            logk[ibin : ibin + nbinschan, iproc, 0, isyst] = (
                                dict_logkavg_proc[syst]
                            )
                            if syst in dict_logkhalfdiff_proc.keys():
                                logk[ibin : ibin + nbinschan, iproc, 1, isyst] = (
                                    dict_logkhalfdiff_proc[syst]
                                )

                ibin += nbinschan

        if self.data_covariance is None and (
            self.systscovariance or self.add_bin_by_bin_stat_to_data_cov
        ):
            # create data covariance assuming poisson statistics
            self.data_covariance = np.diag(data_obs)

        # write results to hdf5 file
        procSize = nproc * np.dtype(self.dtype).itemsize
        systSize = 2 * nsyst * np.dtype(self.dtype).itemsize
        amax = np.max([procSize, systSize])
        if amax > self.chunkSize:
            logger.info(
                f"Maximum chunk size in bytes was increased from {self.chunkSize} to {amax} to align with tensor sizes and allow more efficient reading/writing."
            )
            self.chunkSize = amax

        # create HDF5 file (chunk cache set to the chunk size since we can guarantee fully aligned writes
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)
        outpath = f"{outfolder}/{outfilename}"
        if len(outfilename.split(".")) < 2:
            outpath += ".hdf5"
        logger.info(f"Write output file {outpath}")
        f = h5py.File(outpath, rdcc_nbytes=self.chunkSize, mode="w")

        # propagate meta info into result file
        meta = {
            "meta_info": output_tools.make_meta_info_dict(
                args=args, wd=common.base_dir
            ),
            "channel_info": self.channels,
            "symmetric_tensor": self.symmetric_tensor,
            "systematic_type": self.systematic_type,
        }

        ioutils.pickle_dump_h5py("meta", meta, f)

        systsnoprofile = self.get_systsnoprofile()
        systsnoconstraint = self.get_systsnoconstraint()
        noigroups, noigroupidxs = self.get_noigroups()
        systgroups, systgroupidxs = self.get_systgroups()

        # save some lists of strings to the file for later use
        def create_dataset(
            name,
            content,
            length=None,
            dtype=h5py.special_dtype(vlen=str),
            compression="gzip",
        ):
            dimension = [len(content), length] if length else [len(content)]
            ds = f.create_dataset(
                f"h{name}", dimension, dtype=dtype, compression=compression
            )
            ds[...] = content

        create_dataset("procs", procs)
        create_dataset("signals", sorted(list(self.signals)))
        create_dataset("systs", systs)
        create_dataset("systsnoprofile", systsnoprofile)
        create_dataset("systsnoconstraint", systsnoconstraint)
        create_dataset("systgroups", systgroups)
        create_dataset(
            "systgroupidxs",
            systgroupidxs,
            dtype=h5py.special_dtype(vlen=np.dtype("int32")),
        )
        create_dataset("noigroups", noigroups)
        create_dataset("noigroupidxs", noigroupidxs, dtype="int32")
        create_dataset("pseudodatanames", [n for n in self.pseudodata_names])

        # create h5py datasets with optimized chunk shapes
        nbytes = 0

        constraintweights = self.get_constraintweights(self.dtype)
        nbytes += h5pyutils.writeFlatInChunks(
            constraintweights, f, "hconstraintweights", maxChunkBytes=self.chunkSize
        )
        constraintweights = None

        nbytes += h5pyutils.writeFlatInChunks(
            data_obs, f, "hdata_obs", maxChunkBytes=self.chunkSize
        )
        data_obs = None

        nbytes += h5pyutils.writeFlatInChunks(
            pseudodata, f, "hpseudodata", maxChunkBytes=self.chunkSize
        )
        pseudodata = None

        if self.data_covariance is not None:
            for syst in self.systscovariance:
                systv = np.zeros(shape=(nbinsfull, 1), dtype=self.dtype)

                ibin = 0
                for chan in self.channels.keys():
                    nbinschan = self.nbinschan[chan]
                    dict_norm_chan = self.dict_norm[chan]

                    for proc in procs:
                        if proc not in dict_norm_chan:
                            continue

                        dict_logkavg_proc = self.dict_logkavg[chan][proc]

                        if syst not in dict_logkavg_proc.keys():
                            continue

                        systv[ibin : ibin + nbinschan, 0] += dict_logkavg_proc[syst]

                    ibin += nbinschan

                self.data_covariance[...] += systv @ systv.T

            full_cov = (
                np.add(self.data_covariance, np.diag(sumw2))
                if self.add_bin_by_bin_stat_to_data_cov
                else self.data_covariance
            )
            full_cov_inv = np.linalg.inv(full_cov)

            nbytes += h5pyutils.writeFlatInChunks(
                full_cov_inv,
                f,
                "hdata_cov_inv",
                maxChunkBytes=self.chunkSize,
            )

        nbytes += h5pyutils.writeFlatInChunks(
            sumw, f, "hsumw", maxChunkBytes=self.chunkSize
        )

        nbytes += h5pyutils.writeFlatInChunks(
            sumw2, f, "hsumw2", maxChunkBytes=self.chunkSize
        )

        if self.sparse:
            nbytes += h5pyutils.writeSparse(
                norm_sparse_indices,
                norm_sparse_values,
                norm_sparse_dense_shape,
                f,
                "hnorm_sparse",
                maxChunkBytes=self.chunkSize,
            )
            norm_sparse_indices = None
            norm_sparse_values = None
            nbytes += h5pyutils.writeSparse(
                logk_sparse_indices,
                logk_sparse_values,
                logk_sparse_dense_shape,
                f,
                "hlogk_sparse",
                maxChunkBytes=self.chunkSize,
            )
            logk_sparse_indices = None
            logk_sparse_values = None
        else:
            nbytes += h5pyutils.writeFlatInChunks(
                norm, f, "hnorm", maxChunkBytes=self.chunkSize
            )
            norm = None
            nbytes += h5pyutils.writeFlatInChunks(
                logk, f, "hlogk", maxChunkBytes=self.chunkSize
            )
            logk = None

        logger.info(f"Total raw bytes in arrays = {nbytes}")

    def get_systsstandard(self):
        return list(common.natural_sort(self.systsstandard))

    def get_systsnoprofile(self):
        return list(common.natural_sort(self.systsnoprofile))

    def get_systsnoi(self):
        return list(common.natural_sort(self.systsnoi))

    def get_systsnoconstraint(self):
        return list(common.natural_sort(self.systsnoconstraint))

    def get_systs(self):
        return (
            self.get_systsnoconstraint()
            + self.get_systsstandard()
            + self.get_systsnoprofile()
        )

    def get_constraintweights(self, dtype):
        systs = self.get_systs()
        constraintweights = np.ones([len(systs)], dtype=dtype)
        for syst in self.get_systsnoconstraint():
            constraintweights[systs.index(syst)] = 0.0
        return constraintweights

    def get_groups(self, group_dict):
        systs = self.get_systs()
        groups = []
        idxs = []
        for group, members in common.natural_sort_dict(group_dict).items():
            groups.append(group)
            idx = []
            for syst in members:
                idx.append(systs.index(syst))
            idxs.append(idx)
        return groups, idxs

    def get_noigroups(self):
        # list of groups of systematics to be treated as additional outputs for impacts, etc (aka "nuisances of interest")
        systs = self.get_systs()
        groups = []
        idxs = []
        for group, members in common.natural_sort_dict(self.dict_noigroups).items():
            groups.append(group)
            for syst in members:
                idxs.append(systs.index(syst))
        return groups, idxs

    def get_systgroups(self):
        # list of groups of systematics (nuisances) and lists of indexes
        return self.get_groups(self.dict_systgroups)
