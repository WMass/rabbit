import numpy as np
import tensorflow as tf


class POIModel:

    def __init__(self, indata, *args, **kwargs):
        self.indata = indata

        # # a POI model must set these attribues
        # self.npoi = # number of parameters of interest (POIs)
        # self.pois = # list of names for the POIs
        # self.xpoidefault = # default values for the POIs
        # self.is_linear = # define if the model is linear in the POIs
        # self.allowNegativePOI = # define if the POI can be negative or not

    # class function to parse strings as given by the argparse input e.g. --poiModel <Model> <arg[0]> <args[1]> ...
    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        return cls(indata, *args, **kwargs)

    def compute(self, poi, full=False):
        """
        Compute an array for the rate per process
        :param params: 1D tensor of explicit parameters in the fit
        :return 2D tensor to be multiplied with [proc,bin] tensor
        """

    def set_poi_default(self, expectSignal, allowNegativePOI=False):
        """
        Set default POI values, used by different POI models
        """
        poidefault = tf.ones([self.npoi], dtype=self.indata.dtype)
        if expectSignal is not None:
            indices = []
            updates = []
            for signal, value in expectSignal:
                if signal.encode() not in self.pois:
                    raise ValueError(
                        f"{signal.encode()} not in list of POIs: {self.pois}"
                    )
                idx = np.where(np.isin(self.pois, signal.encode()))[0][0]

                indices.append([idx])
                updates.append(float(value))

            poidefault = tf.tensor_scatter_nd_update(poidefault, indices, updates)

        if allowNegativePOI:
            self.xpoidefault = poidefault
        else:
            self.xpoidefault = tf.sqrt(poidefault)


class Ones(POIModel):
    """
    multiply all processes with ones
    """

    def __init__(self, indata, **kwargs):
        self.indata = indata
        self.npoi = 0
        self.pois = np.array([])
        self.poidefault = tf.zeros([], dtype=self.indata.dtype)

        self.allowNegativePOI = False
        self.is_linear = True

    def compute(self, poi, full=False):
        rnorm = tf.ones(self.indata.nproc, dtype=self.indata.dtype)
        rnorm = tf.reshape(rnorm, [1, -1])
        return rnorm


class Mu(POIModel):
    """
    multiply unconstrained parameter to signal processes, and ones otherwise
    """

    def __init__(self, indata, expectSignal=None, allowNegativePOI=False, **kwargs):
        self.indata = indata

        self.npoi = self.indata.nsignals

        self.pois = np.array([s for s in self.indata.signals])

        self.allowNegativePOI = allowNegativePOI

        self.is_linear = self.npoi == 0 or self.allowNegativePOI

        self.set_poi_default(expectSignal, allowNegativePOI)

    def compute(self, poi, full=False):
        rnorm = tf.concat(
            [poi, tf.ones([self.indata.nproc - poi.shape[0]], dtype=self.indata.dtype)],
            axis=0,
        )

        rnorm = tf.reshape(rnorm, [1, -1])
        return rnorm


class Mixture(POIModel):
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
        allowNegativePOI=False,
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
        self.pois = np.array(
            [
                f"{p}_{c}_mixing".encode()
                for p, c in zip(
                    primary_processes.astype(str), complementary_processes.astype(str)
                )
            ]
        )

        self.allowNegativePOI = allowNegativePOI
        self.is_linear = False

        self.set_poi_default(expectSignal, allowNegativePOI)

    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        """
        parsing the input arguments into the constructor, is has to be called as
        --poiModel Mixture <proc_0>,<proc_1>,... <proc_a>,<proc_b>,...
        to introduce a mixing parameter for proc_0 with proc_a, and proc_1 with proc_b, etc.
        """

        if len(args) != 2:
            raise ValueError(
                f"Expected exactly 2 arguments for Mixture model but got {len(args)}"
            )

        primaries = args[0].split(",")
        complementaries = args[1].split(",")

        return cls(indata, primaries, complementaries, **kwargs)

    def compute(self, poi, full=False):

        ones = tf.ones(self.npoi, dtype=self.indata.dtype)
        updates = tf.concat([ones * poi, ones * (1 - poi)], axis=0)

        # Single scatter update
        rnorm = tf.tensor_scatter_nd_update(
            tf.ones(self.indata.nproc, dtype=self.indata.dtype),
            self.all_idx[:, None],
            updates,
        )

        rnorm = tf.reshape(rnorm, [1, -1])
        return rnorm
