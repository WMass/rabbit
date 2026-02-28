import math

import numpy as np


def writeFlatInChunks(arr, h5group, outname, maxChunkBytes=1024**2):
    arrflat = arr.reshape(-1)

    esize = np.dtype(arrflat.dtype).itemsize
    nbytes = arrflat.size * esize

    # special handling for empty datasets, which should not use chunked storage or compression
    if arrflat.size == 0:
        chunksize = 1
        chunks = None
        compression = None
    else:
        chunksize = int(min(arrflat.size, max(1, math.floor(maxChunkBytes / esize))))
        chunks = (chunksize,)
        compression = "gzip"

    h5dset = h5group.create_dataset(
        outname,
        arrflat.shape,
        chunks=chunks,
        dtype=arrflat.dtype,
        compression=compression,
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
    nbytes += writeFlatInChunks(indices, outgroup, "indices", maxChunkBytes)
    nbytes += writeFlatInChunks(values, outgroup, "values", maxChunkBytes)
    outgroup.attrs["dense_shape"] = np.array(dense_shape, dtype="int64")

    return nbytes
