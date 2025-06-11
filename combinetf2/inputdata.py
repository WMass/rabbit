import h5py
import hist
import numpy as np
import tensorflow as tf

from combinetf2.h5pyutils import makesparsetensor, maketensor


class FitInputData:
    def __init__(self, filename, pseudodata=None):
        with h5py.File(filename, mode="r") as f:

            # load text arrays from file
            self.procs = f["hprocs"][...]
            self.signals = f["hsignals"][...]
            self.systs = f["hsysts"][...]
            self.systsnoprofile = f["hsystsnoprofile"][...]
            self.systsnoconstraint = f["hsystsnoconstraint"][...]
            self.systgroups = f["hsystgroups"][...]
            self.systgroupidxs = f["hsystgroupidxs"][...]

            self.noigroups = f["hnoigroups"][...]
            self.noigroupidxs = f["hnoigroupidxs"][...]
            if "hpseudodatanames" in f.keys():
                self.pseudodatanames = f["hpseudodatanames"][...].astype(str)
            else:
                self.pseudodatanames = []

            # load arrays from file

            if "hdata_cov_inv" in f.keys():
                hdata_cov_inv = f["hdata_cov_inv"]
                self.data_cov_inv = maketensor(hdata_cov_inv)
            else:
                self.data_cov_inv = None

            # load data/pseudodata
            if pseudodata is not None:
                if pseudodata in self.pseudodatanames:
                    pseudodata_idx = np.where(self.pseudodatanames == pseudodata)[0][0]
                else:
                    raise Exception(
                        "Pseudodata %s not found, available pseudodata sets are %s"
                        % (pseudodata, self.pseudodatanames)
                    )
                print("Run pseudodata fit for index %i: " % (pseudodata_idx))
                print(self.pseudodatanames[pseudodata_idx])
                hdata_obs = f["hpseudodata"]

                data_obs = maketensor(hdata_obs)
                self.data_obs = data_obs[:, pseudodata_idx]
            else:
                self.data_obs = maketensor(f["hdata_obs"])

            # start by creating tensors which read in the hdf5 arrays (optimized for memory consumption)
            self.constraintweights = maketensor(f["hconstraintweights"])

            self.sparse = not "hnorm" in f

            if self.sparse:
                print(
                    "WARNING: The sparse tensor implementation is experimental and probably slower than with a dense tensor!"
                )
                self.norm = makesparsetensor(f["hnorm_sparse"])
                self.logk = makesparsetensor(f["hlogk_sparse"])
            else:
                self.norm = maketensor(f["hnorm"])
                self.logk = maketensor(f["hlogk"])

            # infer some metadata from loaded information
            self.dtype = self.data_obs.dtype
            self.nbins = self.data_obs.shape[-1]
            self.nbinsfull = self.norm.shape[0]
            self.nbinsmasked = self.nbinsfull - self.nbins
            self.nproc = len(self.procs)
            self.nsyst = len(self.systs)
            self.nsystnoprofile = len(self.systsnoprofile)
            self.nsystnoconstraint = len(self.systsnoconstraint)
            self.nsignals = len(self.signals)
            self.nsystgroups = len(self.systgroups)
            self.nnoigroups = len(self.noigroups)

            # reference meta data if available
            self.metadata = {}
            if "meta" in f.keys():
                from wums.ioutils import pickle_load_h5py

                self.metadata = pickle_load_h5py(f["meta"])
                self.channel_info = self.metadata["channel_info"]
            else:
                self.channel_info = {
                    "ch0": {
                        "axes": [
                            hist.axis.Integer(
                                0,
                                self.nbins,
                                underflow=False,
                                overflow=False,
                                name="obs",
                            )
                        ]
                    }
                }
                if self.nbinsmasked:
                    self.channel_info["ch1_masked"] = {
                        "axes": [
                            hist.axis.Integer(
                                0,
                                self.nbinsmasked,
                                underflow=False,
                                overflow=False,
                                name="masked",
                            )
                        ]
                    }

            self.symmetric_tensor = self.metadata.get("symmetric_tensor", False)

            if self.metadata.get("exponential_transform", False):
                raise NotImplementedError(
                    "exponential_transform functionality has been removed.   Please use systematic_type normal instead"
                )

            self.systematic_type = self.metadata.get("systematic_type", "log_normal")

            if "hsumw2" in f.keys():
                self.sumw = maketensor(f["hsumw"])
                self.sumw2 = maketensor(f["hsumw2"])
            else:
                # fallback for older datacards
                kstat = maketensor(f["hkstat"])

                self.sumw = self.expected_events_nominal()
                self.sumw2 = self.sumw**2 / kstat

                self.sumw2 = tf.where(kstat == 0.0, self.sumw, self.sumw2)

            # compute indices for channels
            ibin = 0
            for channel, info in self.channel_info.items():
                axes = info["axes"]
                shape = tuple([len(a) for a in axes])
                size = int(np.prod(shape))

                start = ibin
                stop = start + size

                info["start"] = start
                info["stop"] = stop

                ibin = stop

            for channel, info in self.channel_info.items():
                print(channel, info)

            self.axis_procs = hist.axis.StrCategory(self.procs, name="processes")

    @tf.function
    def expected_events_nominal(self):
        rnorm = tf.ones(self.nproc, dtype=self.dtype)
        mrnorm = tf.expand_dims(rnorm, -1)

        if self.sparse:
            nexpfullcentral = tf.sparse.sparse_dense_matmul(self.norm, mrnorm)
            nexpfullcentral = tf.squeeze(nexpfullcentral, -1)
        else:
            nexpfullcentral = tf.matmul(self.norm, mrnorm)
            nexpfullcentral = tf.squeeze(nexpfullcentral, -1)

        return nexpfullcentral
