import math

import hdf5plugin
import numpy as np

# Compression strategy for the HDF5 write path.
#
# By default dense arrays are written with Blosc2 + LZ4 byte-shuffle. This is
# much faster than gzip (typically ~5x on write) while achieving equal or
# better ratios, and works well for dense tensor buffers that often contain
# lots of structural zeros from sparsity patterns.
#
# Callers that know the data is already densely packed with unstructured
# nonzero values (e.g. the ``values`` payload of an explicitly sparse tensor)
# can pass ``compress=False`` to skip compression entirely. For those inputs
# Blosc2 LZ4 buys only ~4-5% of file size at ~5x the write cost, so turning
# it off is a strict win.
#
# The HDF5 filter pipeline is fundamentally single-threaded per chunk, so
# multi-threaded compression via BLOSC2_NTHREADS does not take effect through
# h5py; the main speedup comes from switching compressor and skipping the
# uncompressible buffers.
#
# Reading requires the hdf5plugin filter to be registered, which happens
# automatically via the ``import hdf5plugin`` at module import time in both
# h5pyutils_write and h5pyutils_read.
_DEFAULT_COMPRESSION_KWARGS = hdf5plugin.Blosc2(cname="lz4", clevel=5)


def writeFlatInChunks(arr, h5group, outname, maxChunkBytes=1024**2, compress=True):
    arrflat = arr.reshape(-1)

    esize = np.dtype(arrflat.dtype).itemsize
    nbytes = arrflat.size * esize

    # Empty datasets must not use chunked storage or compression.
    if arrflat.size == 0:
        chunksize = 1
        extra_kwargs = {"chunks": None}
    else:
        chunksize = int(min(arrflat.size, max(1, math.floor(maxChunkBytes / esize))))
        extra_kwargs = {"chunks": (chunksize,)}
        if compress:
            extra_kwargs.update(_DEFAULT_COMPRESSION_KWARGS)

    h5dset = h5group.create_dataset(
        outname,
        arrflat.shape,
        dtype=arrflat.dtype,
        **extra_kwargs,
    )

    # write in chunks, preserving sparsity if relevant
    for ielem in range(0, arrflat.size, chunksize):
        aout = arrflat[ielem : ielem + chunksize]
        if np.count_nonzero(aout):
            h5dset[ielem : ielem + chunksize] = aout

    h5dset.attrs["original_shape"] = np.array(arr.shape, dtype="int64")

    return nbytes


def writeSparse(indices, values, dense_shape, h5group, outname, maxChunkBytes=1024**2):
    outgroup = h5group.create_group(outname)

    nbytes = 0
    # Index arrays compress extremely well (~10x for the tensor-sparse
    # structures used by rabbit), so keep the default compression.
    nbytes += writeFlatInChunks(indices, outgroup, "indices", maxChunkBytes)
    # Values of a sparse tensor are already densely packed nonzeros; real
    # physics values typically give only ~4% compression gain at 5x the
    # write cost, so skip compression here.
    nbytes += writeFlatInChunks(
        values, outgroup, "values", maxChunkBytes, compress=False
    )
    outgroup.attrs["dense_shape"] = np.array(dense_shape, dtype="int64")

    return nbytes
