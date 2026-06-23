"""Generic ``auxiliary`` array bundles carried through the fit HDF5.

An auxiliary entry is a named bundle of arbitrary arrays — numeric ndarrays
and/or 1-D string lists — stashed in the input HDF5 under a top-level
``auxiliary`` group. It is **not** used by the fit itself; it is a side channel
for ParamModels to carry pre-computed inputs (e.g. a reco x gen response matrix)
that must stay consistent with the datacard they are fit against, rather than
being read from a separate, independently-versioned file at fit time.

This mirrors the ``external_terms`` mechanism (see
:mod:`rabbit.external_likelihood`): :class:`rabbit.tensorwriter.TensorWriter`
collects bundles via ``add_auxiliary`` and serializes them in ``write()``;
:class:`rabbit.inputdata.FitInputData` exposes them as ``self.auxiliary``.

Round-trip guarantees:

* numeric arrays survive bit-for-bit including dtype and shape, via
  ``writeFlatInChunks`` (which stamps an ``original_shape`` attr) and
  ``maketensor`` (which restores it);
* 1-D string lists (e.g. axis names) survive as a Python ``list[str]``, stored
  as a vlen-str dataset like ``external_terms``' ``params``.
"""

import h5py
import numpy as np

from rabbit import h5pyutils_write
from rabbit.h5pyutils_read import maketensor


def _is_string_array(arr):
    """True if ``arr`` should be stored as strings rather than numbers."""
    return arr.dtype.kind in ("U", "S", "O")


def write_auxiliary_group(parent, auxiliary, maxChunkBytes=1024**2):
    """Serialize auxiliary bundles under ``parent``'s ``auxiliary`` group.

    Parameters
    ----------
    parent : h5py.Group
        Open HDF5 group/file to create the ``auxiliary`` subgroup in.
    auxiliary : list[dict]
        Bundles ``{"name": str, "datasets": {key: ndarray | list[str]}}`` as
        collected by ``TensorWriter.add_auxiliary``.
    maxChunkBytes : int
        Chunk size passed through to ``writeFlatInChunks`` for numeric arrays.

    Returns
    -------
    int
        Number of raw array bytes written (0 if ``auxiliary`` is empty).
    """
    if not auxiliary:
        return 0

    nbytes = 0
    aux_group = parent.create_group("auxiliary")
    for aux in auxiliary:
        g = aux_group.create_group(aux["name"])
        for key, val in aux["datasets"].items():
            arr = np.asarray(val)
            if _is_string_array(arr):
                # 1-D list of strings (e.g. axis names) -> vlen-str dataset,
                # mirroring external_terms' "params".
                flat = arr.reshape(-1)
                ds = g.create_dataset(
                    key,
                    [flat.size],
                    dtype=h5py.special_dtype(vlen=str),
                    compression="gzip",
                )
                ds[...] = [str(s) for s in flat]
            else:
                # numeric array -> flat chunked; shape recovered by maketensor.
                nbytes += h5pyutils_write.writeFlatInChunks(
                    arr, g, key, maxChunkBytes=maxChunkBytes
                )
    return nbytes


def read_auxiliary_from_h5(aux_group):
    """Decode an HDF5 ``auxiliary`` group.

    Parameters
    ----------
    aux_group : h5py.Group or None
        The ``auxiliary`` group in the input HDF5 file, or ``None``.

    Returns
    -------
    dict
        ``{name: {key: ndarray | list[str]}}``. Numeric datasets are decoded via
        ``maketensor`` (shape restored from ``original_shape``); string datasets
        are decoded to a Python ``list[str]``. Empty dict if ``aux_group`` is
        ``None``.
    """
    if aux_group is None:
        return {}

    out = {}
    for name, g in aux_group.items():
        bundle = {}
        for key, ds in g.items():
            if h5py.check_string_dtype(ds.dtype):
                bundle[key] = [
                    s.decode() if isinstance(s, bytes) else str(s) for s in ds[...]
                ]
            else:
                bundle[key] = np.asarray(maketensor(ds))
        out[name] = bundle
    return out
