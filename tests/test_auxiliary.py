"""Test the generic ``auxiliary`` array bundles carried through the fit HDF5.

Covers:
  * round-trip through the real path: TensorWriter.add_auxiliary -> write() ->
    FitInputData.auxiliary, asserting numeric arrays survive bit-for-bit
    (dtype + shape, incl. >2D) and string lists survive as list[str];
  * a datacard with no auxiliary reads back as an empty dict;
  * add_auxiliary rejects a duplicate bundle name;
  * a direct write_auxiliary_group/read_auxiliary_from_h5 round-trip with
    multiple bundles and mixed dtypes (no full datacard needed).

This mirrors the scetlib_np response-matrix use case (R reco x gen, N_gen,
axis names + edges) without depending on WRemnants.
"""

import os
import tempfile

import h5py
import hist
import numpy as np

from rabbit import inputdata, tensorwriter
from rabbit.auxiliary import read_auxiliary_from_h5, write_auxiliary_group


def build_minimal_writer():
    """A minimal valid TensorWriter: one channel, data, one signal process,
    one shape systematic (matches the proven test_external_term setup)."""
    np.random.seed(0)
    ax = hist.axis.Regular(20, -5, 5, name="x")

    h_data = hist.Hist(ax, storage=hist.storage.Double())
    h_bkg = hist.Hist(ax, storage=hist.storage.Weight())

    x_bkg = np.random.uniform(-5, 5, 5000)
    h_data.fill(x_bkg)
    h_bkg.fill(x_bkg, weight=np.ones(len(x_bkg)))

    weights = 0.01 * (ax.centers - ax.centers[0]) - 0.05
    h_up = h_bkg.copy()
    h_dn = h_bkg.copy()
    h_up.values()[...] = h_bkg.values() * (1 + weights)
    h_dn.values()[...] = h_bkg.values() * (1 - weights)

    writer = tensorwriter.TensorWriter()
    writer.add_channel([ax], "ch0")
    writer.add_data(h_data, "ch0")
    writer.add_process(h_bkg, "bkg", "ch0", signal=True)
    writer.add_systematic([h_up, h_dn], "shape", "bkg", "ch0", symmetrize="average")
    return writer


def make_scetlib_np_bundle():
    """A scetlib_np-shaped bundle: multi-dim float64 R, 1-D float64 N_gen,
    reco/gen axis name lists, and one edges array per axis."""
    np.random.seed(42)
    reco_axes = ["ptll", "yll"]
    gen_axes = ["ptVGen", "absYVGen"]
    reco_shape = (4, 3)
    gen_shape = (5, 2)
    R = np.random.uniform(0.0, 1.0, size=reco_shape + gen_shape).astype(np.float64)
    N_gen = np.random.uniform(1.0, 10.0, size=gen_shape).astype(np.float64)
    edges = {
        "edges__ptll": np.array([0.0, 5.0, 10.0, 20.0, 44.0], dtype=np.float64),
        "edges__yll": np.array([0.0, 1.0, 2.0, 2.5], dtype=np.float64),
        "edges__ptVGen": np.array([0.0, 4.0, 8.0, 16.0, 44.0, 100.0], dtype=np.float64),
        "edges__absYVGen": np.array([0.0, 1.25, 2.5], dtype=np.float64),
    }
    datasets = {
        "R": R,
        "N_gen": N_gen,
        "reco_axes": reco_axes,
        "gen_axes": gen_axes,
        **edges,
    }
    return datasets


def assert_bundle_equal(got, expected):
    assert set(got.keys()) == set(
        expected.keys()
    ), f"keys differ: {sorted(got)} != {sorted(expected)}"
    for key, exp in expected.items():
        val = got[key]
        if isinstance(exp, list):  # string list
            assert val == exp, f"{key}: {val} != {exp}"
        else:  # numeric array, bit-for-bit incl dtype + shape
            exp_arr = np.asarray(exp)
            assert (
                val.shape == exp_arr.shape
            ), f"{key}: shape {val.shape} != {exp_arr.shape}"
            assert (
                val.dtype == exp_arr.dtype
            ), f"{key}: dtype {val.dtype} != {exp_arr.dtype}"
            assert np.array_equal(val, exp_arr), f"{key}: values differ"


def test_through_writer(tmpdir):
    datasets = make_scetlib_np_bundle()

    writer = build_minimal_writer()
    writer.add_auxiliary("scetlib_np", datasets)
    writer.write(outfolder=tmpdir, outfilename="with_aux")

    indata_obj = inputdata.FitInputData(os.path.join(tmpdir, "with_aux.hdf5"))
    assert "scetlib_np" in indata_obj.auxiliary, "scetlib_np bundle missing on read"
    assert_bundle_equal(indata_obj.auxiliary["scetlib_np"], datasets)
    print("PASS: add_auxiliary -> write -> FitInputData round-trip (scetlib_np)")


def test_no_auxiliary(tmpdir):
    writer = build_minimal_writer()
    writer.write(outfolder=tmpdir, outfilename="no_aux")
    indata_obj = inputdata.FitInputData(os.path.join(tmpdir, "no_aux.hdf5"))
    assert (
        indata_obj.auxiliary == {}
    ), f"expected empty auxiliary, got {indata_obj.auxiliary}"
    print("PASS: datacard with no auxiliary reads back as empty dict")


def test_duplicate_name_guard():
    writer = build_minimal_writer()
    writer.add_auxiliary("dup", {"a": np.zeros(3)})
    try:
        writer.add_auxiliary("dup", {"b": np.ones(3)})
    except RuntimeError as exc:
        assert "already added" in str(exc)
        print("PASS: add_auxiliary rejects a duplicate bundle name")
    else:
        raise AssertionError("expected RuntimeError on duplicate auxiliary name")


def test_direct_roundtrip(tmpdir):
    """write_auxiliary_group / read_auxiliary_from_h5 directly, no datacard.
    Exercises multiple bundles and mixed dtypes (float32, int, >2D)."""
    bundles = [
        {
            "name": "b0",
            "datasets": {
                "arr3d": np.arange(24, dtype=np.float64).reshape(2, 3, 4),
                "f32": np.array([1.5, 2.5], dtype=np.float32),
                "ints": np.array([[1, 2], [3, 4]], dtype=np.int64),
                "names": ["alpha", "beta", "gamma"],
            },
        },
        {
            "name": "b1",
            "datasets": {"x": np.linspace(0, 1, 7, dtype=np.float64)},
        },
    ]
    path = os.path.join(tmpdir, "direct.hdf5")
    with h5py.File(path, "w") as f:
        write_auxiliary_group(f, bundles)
    with h5py.File(path, "r") as f:
        out = read_auxiliary_from_h5(f.get("auxiliary"))

    assert set(out.keys()) == {"b0", "b1"}
    for b in bundles:
        assert_bundle_equal(out[b["name"]], b["datasets"])
    # explicit dtype checks (assert_bundle_equal already covers, but be loud)
    assert out["b0"]["f32"].dtype == np.float32
    assert out["b0"]["ints"].dtype == np.int64
    assert out["b0"]["arr3d"].shape == (2, 3, 4)
    # empty group -> {}
    assert read_auxiliary_from_h5(None) == {}
    print("PASS: direct write_auxiliary_group/read_auxiliary_from_h5 round-trip")


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_through_writer(tmpdir)
        test_no_auxiliary(tmpdir)
        test_duplicate_name_guard()
        test_direct_roundtrip(tmpdir)
    print()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
