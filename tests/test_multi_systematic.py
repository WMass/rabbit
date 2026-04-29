"""
Test that add_systematic correctly handles a histogram with extra axes
representing multiple independent systematics. The result should be identical
to booking each systematic individually.
"""

import os
import tempfile

import h5py
import hist
import numpy as np

from rabbit import tensorwriter


def make_base_histograms(nsyst):
    """Build a nominal background plus per-syst variation histograms."""
    np.random.seed(42)

    ax_x = hist.axis.Regular(20, -5, 5, name="x")

    h_bkg = hist.Hist(ax_x, storage=hist.storage.Weight())
    h_bkg.fill(np.random.uniform(-5, 5, 5000), weight=np.ones(5000))

    bin_centers = ax_x.centers - ax_x.centers[0]
    base_weights = 0.01 * bin_centers - 0.05

    # build a different variation per systematic
    variations_up = []
    variations_dn = []
    for i in range(nsyst):
        scale = 1.0 + 0.5 * i
        h_up = h_bkg.copy()
        h_dn = h_bkg.copy()
        h_up.values()[...] = h_bkg.values() * (1 + scale * base_weights)
        h_dn.values()[...] = h_bkg.values() * (1 - scale * base_weights)
        variations_up.append(h_up)
        variations_dn.append(h_dn)

    return h_bkg, variations_up, variations_dn


def make_writer_with_individual_systs(
    h_bkg, variations_up, variations_dn, name_prefix, sparse=False
):
    """Reference: book each systematic separately via the existing API."""
    writer = tensorwriter.TensorWriter(sparse=sparse)
    writer.add_channel(h_bkg.axes, "ch0")
    writer.add_data(h_bkg, "ch0")
    writer.add_process(h_bkg, "bkg", "ch0", signal=True)

    for i, (h_up, h_dn) in enumerate(zip(variations_up, variations_dn)):
        writer.add_systematic(
            [h_up, h_dn],
            f"{name_prefix}_{i}",
            "bkg",
            "ch0",
            symmetrize="average",
        )
    return writer


def make_writer_with_multi_axis(
    h_bkg, variations_up, variations_dn, name_prefix, sparse=False
):
    """New path: pack the variations into a single histogram with an extra 'syst' axis."""
    nsyst = len(variations_up)
    ax_x = h_bkg.axes[0]
    ax_syst = hist.axis.Integer(0, nsyst, underflow=False, overflow=False, name="syst")

    h_up_combined = hist.Hist(ax_x, ax_syst, storage=hist.storage.Weight())
    h_dn_combined = hist.Hist(ax_x, ax_syst, storage=hist.storage.Weight())
    for i in range(nsyst):
        h_up_combined.values()[:, i] = variations_up[i].values()
        h_dn_combined.values()[:, i] = variations_dn[i].values()

    writer = tensorwriter.TensorWriter(sparse=sparse)
    writer.add_channel(h_bkg.axes, "ch0")
    writer.add_data(h_bkg, "ch0")
    writer.add_process(h_bkg, "bkg", "ch0", signal=True)
    writer.add_systematic(
        [h_up_combined, h_dn_combined],
        name_prefix,
        "bkg",
        "ch0",
        symmetrize="average",
    )
    return writer


def _embed_no_flow_into_with_flow(values_no_flow, axes):
    """Embed a no-flow dense array into a with-flow dense array of the given axes.

    Flow bins are filled with zeros. Used to construct SparseHist data when the
    user only has values for the regular bins.
    """
    full_shape = tuple(int(ax.extent) for ax in axes)
    full = np.zeros(full_shape, dtype=values_no_flow.dtype)
    slices = tuple(
        slice(
            tensorwriter.SparseHist._underflow_offset(ax),
            tensorwriter.SparseHist._underflow_offset(ax) + len(ax),
        )
        for ax in axes
    )
    full[slices] = values_no_flow
    return full


def make_writer_with_sparsehist_multi_axis(
    h_bkg, variations_up, variations_dn, name_prefix
):
    """Sparse mode + with-flow SparseHist input on a no-flow channel.

    Exercises the conversion from SparseHist's internal with-flow layout to the
    no-flow CSR layout used by the channel.
    """
    import scipy.sparse

    nsyst = len(variations_up)
    ax_x = h_bkg.axes[0]
    ax_syst = hist.axis.Integer(0, nsyst, underflow=False, overflow=False, name="syst")

    # Build no-flow (x_size, nsyst) data, then embed into with-flow shape
    up_no_flow = np.zeros((len(ax_x), nsyst))
    dn_no_flow = np.zeros((len(ax_x), nsyst))
    for i in range(nsyst):
        up_no_flow[:, i] = variations_up[i].values()
        dn_no_flow[:, i] = variations_dn[i].values()

    up_full = _embed_no_flow_into_with_flow(up_no_flow, [ax_x, ax_syst])
    dn_full = _embed_no_flow_into_with_flow(dn_no_flow, [ax_x, ax_syst])
    bkg_full = _embed_no_flow_into_with_flow(h_bkg.values(), [ax_x])

    sh_up = tensorwriter.SparseHist(scipy.sparse.csr_array(up_full), [ax_x, ax_syst])
    sh_dn = tensorwriter.SparseHist(scipy.sparse.csr_array(dn_full), [ax_x, ax_syst])
    sh_bkg = tensorwriter.SparseHist(
        scipy.sparse.csr_array(bkg_full.reshape(1, -1)), [ax_x]
    )

    writer = tensorwriter.TensorWriter(sparse=True)
    writer.add_channel(h_bkg.axes, "ch0")  # flow=False
    writer.add_data(h_bkg, "ch0")
    writer.add_process(sh_bkg, "bkg", "ch0", signal=True, variances=h_bkg.variances())
    writer.add_systematic(
        [sh_up, sh_dn],
        name_prefix,
        "bkg",
        "ch0",
        symmetrize="average",
    )
    return writer


def make_writer_masked_flow_individual(
    h_bkg, variations_up, variations_dn, name_prefix
):
    """Reference: masked channel with flow=True, hist process and individual hist systematics."""
    ax_x = h_bkg.axes[0]
    writer = tensorwriter.TensorWriter(sparse=True)

    # Regular non-masked data channel (needed because every TensorWriter must have data)
    writer.add_channel(h_bkg.axes, "ch0")
    writer.add_data(h_bkg, "ch0")
    writer.add_process(h_bkg, "bkg", "ch0", signal=True)

    # Masked channel with flow=True
    writer.add_channel([ax_x], "masked0", masked=True, flow=True)
    writer.add_process(h_bkg, "bkg", "masked0", signal=True)

    for i, (h_up, h_dn) in enumerate(zip(variations_up, variations_dn)):
        writer.add_systematic(
            [h_up, h_dn],
            f"{name_prefix}_{i}",
            "bkg",
            "masked0",
            symmetrize="average",
        )

    return writer


def make_writer_masked_flow_sparsehist(
    h_bkg, variations_up, variations_dn, name_prefix
):
    """SparseHist (always with-flow internally) on a masked channel with flow=True."""
    import scipy.sparse

    nsyst = len(variations_up)
    ax_x = h_bkg.axes[0]  # Regular axis with under/overflow
    ax_syst = hist.axis.Integer(0, nsyst, underflow=False, overflow=False, name="syst")

    # Build no-flow data and embed into with-flow shape via the helper.
    up_no_flow = np.zeros((len(ax_x), nsyst))
    dn_no_flow = np.zeros((len(ax_x), nsyst))
    for i in range(nsyst):
        up_no_flow[:, i] = variations_up[i].values()
        dn_no_flow[:, i] = variations_dn[i].values()

    up_full = _embed_no_flow_into_with_flow(up_no_flow, [ax_x, ax_syst])
    dn_full = _embed_no_flow_into_with_flow(dn_no_flow, [ax_x, ax_syst])
    bkg_full = _embed_no_flow_into_with_flow(h_bkg.values(), [ax_x])

    sh_up = tensorwriter.SparseHist(scipy.sparse.csr_array(up_full), [ax_x, ax_syst])
    sh_dn = tensorwriter.SparseHist(scipy.sparse.csr_array(dn_full), [ax_x, ax_syst])
    sh_bkg = tensorwriter.SparseHist(
        scipy.sparse.csr_array(bkg_full.reshape(1, -1)), [ax_x]
    )

    writer = tensorwriter.TensorWriter(sparse=True)
    writer.add_channel(h_bkg.axes, "ch0")
    writer.add_data(h_bkg, "ch0")
    writer.add_process(h_bkg, "bkg", "ch0", signal=True)

    writer.add_channel([ax_x], "masked0", masked=True, flow=True)
    writer.add_process(
        sh_bkg, "bkg", "masked0", signal=True, variances=np.zeros(int(ax_x.extent))
    )
    writer.add_systematic(
        [sh_up, sh_dn],
        name_prefix,
        "bkg",
        "masked0",
        symmetrize="average",
    )
    return writer


def make_writer_with_str_category(h_bkg, variations_up, variations_dn):
    """Variant using a StrCategory axis to verify name labels come from bin values."""
    nsyst = len(variations_up)
    ax_x = h_bkg.axes[0]
    labels = [f"var{i}" for i in range(nsyst)]
    ax_syst = hist.axis.StrCategory(labels, name="kind")

    h_up_combined = hist.Hist(ax_x, ax_syst, storage=hist.storage.Weight())
    h_dn_combined = hist.Hist(ax_x, ax_syst, storage=hist.storage.Weight())
    for i in range(nsyst):
        h_up_combined.values()[:, i] = variations_up[i].values()
        h_dn_combined.values()[:, i] = variations_dn[i].values()

    writer = tensorwriter.TensorWriter()
    writer.add_channel(h_bkg.axes, "ch0")
    writer.add_data(h_bkg, "ch0")
    writer.add_process(h_bkg, "bkg", "ch0", signal=True)
    writer.add_systematic(
        [h_up_combined, h_dn_combined],
        "shape",
        "bkg",
        "ch0",
        symmetrize="average",
    )
    return writer, labels


def read_hdf5_arrays(path):
    """Load systs, dense norm, and dense logk from a written tensor file.

    Materializes dense arrays from the sparse storage format if needed so that
    sparse-mode and dense-mode outputs can be compared.
    """
    with h5py.File(path, "r") as f:
        systs = [s.decode() for s in f["hsysts"][...]]
        nproc = len(f["hprocs"][...])
        nsyst = len(systs)

        if "hnorm" in f:
            hnorm = np.asarray(f["hnorm"]).reshape(
                tuple(f["hnorm"].attrs["original_shape"])
            )
            hlogk = np.asarray(f["hlogk"]).reshape(
                tuple(f["hlogk"].attrs["original_shape"])
            )
            return {"systs": systs, "hnorm": hnorm, "hlogk": hlogk}

        # Sparse format: reconstruct dense (nbinsfull, nproc) and (nbinsfull, nproc, nsyst)
        # writeFlatInChunks stores the original shape as an attribute
        norm_idx_dset = f["hnorm_sparse"]["indices"]
        norm_indices = np.asarray(norm_idx_dset).reshape(
            tuple(norm_idx_dset.attrs["original_shape"])
        )
        norm_values = np.asarray(f["hnorm_sparse"]["values"])
        nbinsfull, _ = f["hnorm_sparse"].attrs["dense_shape"]
        hnorm = np.zeros((int(nbinsfull), int(nproc)))
        hnorm[norm_indices[:, 0], norm_indices[:, 1]] = norm_values

        logk_idx_dset = f["hlogk_sparse"]["indices"]
        logk_indices = np.asarray(logk_idx_dset).reshape(
            tuple(logk_idx_dset.attrs["original_shape"])
        )
        logk_values = np.asarray(f["hlogk_sparse"]["values"])
        # logk_indices[:, 0] indexes into norm_sparse; [:, 1] is syst (or syst*2 for asym)
        logk_nsyst_dim = f["hlogk_sparse"].attrs["dense_shape"][1]
        symmetric = logk_nsyst_dim == nsyst
        if symmetric:
            hlogk = np.zeros((int(nbinsfull), int(nproc), int(nsyst)))
        else:
            hlogk = np.zeros((int(nbinsfull), int(nproc), 2, int(nsyst)))

        for k in range(len(logk_indices)):
            ni = logk_indices[k, 0]  # index into norm_sparse
            si = logk_indices[k, 1]  # syst dim index
            bin_idx, proc_idx = norm_indices[ni]
            if symmetric:
                hlogk[bin_idx, proc_idx, si] = logk_values[k]
            else:
                if si < nsyst:
                    hlogk[bin_idx, proc_idx, 0, si] = logk_values[k]
                else:
                    hlogk[bin_idx, proc_idx, 1, si - nsyst] = logk_values[k]

        return {"systs": systs, "hnorm": hnorm, "hlogk": hlogk}


def main():
    nsyst = 4

    with tempfile.TemporaryDirectory() as tmpdir:
        h_bkg, var_up, var_dn = make_base_histograms(nsyst)

        # Reference path: individual systematics
        ref_writer = make_writer_with_individual_systs(h_bkg, var_up, var_dn, "shape")
        ref_path = os.path.join(tmpdir, "ref.hdf5")
        ref_writer.write(outfolder=tmpdir, outfilename="ref")

        # New path: single histogram with extra axis
        multi_writer = make_writer_with_multi_axis(h_bkg, var_up, var_dn, "shape")
        multi_writer.write(outfolder=tmpdir, outfilename="multi")
        multi_path = os.path.join(tmpdir, "multi.hdf5")

        ref = read_hdf5_arrays(ref_path)
        multi = read_hdf5_arrays(multi_path)

        # Auto-generated names for an Integer axis named "syst" should be "shape_0", "shape_1", ...
        expected_names = [f"shape_{i}" for i in range(nsyst)]

        print("Reference systs:    ", ref["systs"])
        print("Multi-axis systs:   ", multi["systs"])
        print("Expected names:     ", expected_names)

        assert (
            ref["systs"] == expected_names
        ), f"Reference systs {ref['systs']} != expected {expected_names}"
        assert (
            multi["systs"] == expected_names
        ), f"Multi-axis systs {multi['systs']} != expected {expected_names}"

        assert np.allclose(ref["hnorm"], multi["hnorm"]), "norm mismatch"
        assert np.allclose(ref["hlogk"], multi["hlogk"]), "logk mismatch"

        print("PASS: multi-axis Integer matches individual systematics")

        # StrCategory axis: names should come from the string bin labels
        cat_writer, cat_labels = make_writer_with_str_category(h_bkg, var_up, var_dn)
        cat_writer.write(outfolder=tmpdir, outfilename="cat")
        cat_path = os.path.join(tmpdir, "cat.hdf5")
        cat = read_hdf5_arrays(cat_path)

        expected_cat_names = sorted([f"shape_{lbl}" for lbl in cat_labels])
        print("StrCategory systs:  ", cat["systs"])
        assert (
            cat["systs"] == expected_cat_names
        ), f"Category systs {cat['systs']} != expected {expected_cat_names}"
        # Same logk values, just different names
        assert np.allclose(ref["hnorm"], cat["hnorm"]), "norm mismatch (cat)"
        assert np.allclose(ref["hlogk"], cat["hlogk"]), "logk mismatch (cat)"

        print("PASS: multi-axis StrCategory matches individual systematics")

        # Sparse mode: hist with extra axis
        sparse_ref_writer = make_writer_with_individual_systs(
            h_bkg, var_up, var_dn, "shape", sparse=True
        )
        sparse_ref_writer.write(outfolder=tmpdir, outfilename="sparse_ref")
        sparse_ref = read_hdf5_arrays(os.path.join(tmpdir, "sparse_ref.hdf5"))

        sparse_multi_writer = make_writer_with_multi_axis(
            h_bkg, var_up, var_dn, "shape", sparse=True
        )
        sparse_multi_writer.write(outfolder=tmpdir, outfilename="sparse_multi")
        sparse_multi = read_hdf5_arrays(os.path.join(tmpdir, "sparse_multi.hdf5"))

        print("Sparse-mode multi-axis systs:", sparse_multi["systs"])
        assert sparse_multi["systs"] == expected_names
        assert np.allclose(
            sparse_ref["hnorm"], sparse_multi["hnorm"]
        ), "norm mismatch (sparse multi vs sparse ref)"
        assert np.allclose(
            sparse_ref["hlogk"], sparse_multi["hlogk"]
        ), "logk mismatch (sparse multi vs sparse ref)"
        # Sparse and dense paths should agree
        assert np.allclose(
            ref["hnorm"], sparse_ref["hnorm"]
        ), "norm mismatch (sparse ref vs dense ref)"
        assert np.allclose(
            ref["hlogk"], sparse_ref["hlogk"]
        ), "logk mismatch (sparse ref vs dense ref)"
        print("PASS: sparse mode multi-axis matches sparse mode individual")

        # SparseHist input + sparse mode + multi-axis
        sh_writer = make_writer_with_sparsehist_multi_axis(
            h_bkg, var_up, var_dn, "shape"
        )
        sh_writer.write(outfolder=tmpdir, outfilename="sparsehist_multi")
        sh = read_hdf5_arrays(os.path.join(tmpdir, "sparsehist_multi.hdf5"))

        print("SparseHist multi-axis systs:", sh["systs"])
        assert sh["systs"] == expected_names
        assert np.allclose(
            sparse_ref["hnorm"], sh["hnorm"]
        ), "norm mismatch (SparseHist multi vs sparse ref)"
        assert np.allclose(
            sparse_ref["hlogk"], sh["hlogk"]
        ), "logk mismatch (SparseHist multi vs sparse ref)"
        print("PASS: SparseHist multi-axis matches sparse mode individual")

        # Flow test: SparseHist with flow=True on a masked channel with flow=True
        masked_ref_writer = make_writer_masked_flow_individual(
            h_bkg, var_up, var_dn, "shape"
        )
        masked_ref_writer.write(outfolder=tmpdir, outfilename="masked_ref")
        masked_ref = read_hdf5_arrays(os.path.join(tmpdir, "masked_ref.hdf5"))

        masked_sh_writer = make_writer_masked_flow_sparsehist(
            h_bkg, var_up, var_dn, "shape"
        )
        masked_sh_writer.write(outfolder=tmpdir, outfilename="masked_sh")
        masked_sh = read_hdf5_arrays(os.path.join(tmpdir, "masked_sh.hdf5"))

        print("Masked-flow individual systs:", masked_ref["systs"])
        print("Masked-flow SparseHist systs:", masked_sh["systs"])
        assert masked_ref["systs"] == expected_names
        assert masked_sh["systs"] == expected_names
        assert np.allclose(
            masked_ref["hnorm"], masked_sh["hnorm"]
        ), "norm mismatch (masked SparseHist flow vs masked individual)"
        assert np.allclose(
            masked_ref["hlogk"], masked_sh["hlogk"]
        ), "logk mismatch (masked SparseHist flow vs masked individual)"
        print("PASS: SparseHist on masked flow=True channel matches individual")

        # --- Batched SparseHist path: single hist, mirror=True, as_difference=True ---
        # This exercises the vectorized fast path in add_systematic which
        # bypasses the per-slice dispatch entirely. We compare byte-for-byte
        # against the equivalent per-syst manual booking (which goes through
        # the regular single-syst path) using log_normal systematic type on
        # a dense process, on data that includes positions where the delta
        # pushes the bin negative (so the logkepsilon fallback is exercised).
        import scipy.sparse as _sp
        from wums.sparse_hist import SparseHist as _SH

        nbatch = 12
        ax_bx = hist.axis.Regular(8, -4, 4, name="x")
        ax_by = hist.axis.Regular(6, 0, 3, name="y")
        ax_bs = hist.axis.Integer(
            0, nbatch, underflow=False, overflow=False, name="syst"
        )

        rng = np.random.default_rng(17)
        h_bproc = hist.Hist(ax_bx, ax_by, storage=hist.storage.Weight())
        x_v = rng.normal(0, 1, 1000)
        y_v = rng.uniform(0, 3, 1000)
        h_bproc.fill(x_v, y_v, weight=np.ones(1000))
        h_bdata = hist.Hist(ax_bx, ax_by, storage=hist.storage.Double())
        h_bdata.fill(x_v, y_v)

        ext_shape = (ax_bx.extent, ax_by.extent, ax_bs.extent)
        dense_systs = rng.normal(0, 0.1, ext_shape)
        sparse_mask = rng.random(ext_shape) < 0.5
        dense_systs[sparse_mask] = 0
        flat_data = dense_systs.reshape(1, -1)
        sh_batch = _SH(_sp.csr_array(flat_data), [ax_bx, ax_by, ax_bs])

        def make_batch_writer(use_batched):
            w = tensorwriter.TensorWriter(sparse=True, systematic_type="log_normal")
            w.add_channel([ax_bx, ax_by], "ch0")
            w.add_data(h_bdata, "ch0")
            w.add_process(h_bproc, "proc", "ch0", signal=True)
            if use_batched:
                w.add_systematic(
                    sh_batch,
                    "syst",
                    "proc",
                    "ch0",
                    mirror=True,
                    as_difference=True,
                    constrained=False,
                    groups=["g"],
                )
            else:
                for i in range(nbatch):
                    sub_dense = dense_systs[:, :, i]
                    sub_flat = _sp.csr_array(sub_dense.reshape(1, -1))
                    sub_sh = _SH(sub_flat, [ax_bx, ax_by])
                    w.add_systematic(
                        sub_sh,
                        f"syst_{i}",
                        "proc",
                        "ch0",
                        mirror=True,
                        as_difference=True,
                        constrained=False,
                        groups=["g"],
                        syst_axes=[],
                    )
            return w

        wb = make_batch_writer(True)
        wb.write(outfolder=tmpdir, outfilename="batch_fast")
        wm = make_batch_writer(False)
        wm.write(outfolder=tmpdir, outfilename="batch_manual")

        bf = read_hdf5_arrays(os.path.join(tmpdir, "batch_fast.hdf5"))
        bm = read_hdf5_arrays(os.path.join(tmpdir, "batch_manual.hdf5"))

        print("Batched-path fast systs:    ", bf["systs"])
        print("Batched-path manual systs:  ", bm["systs"])
        assert (
            bf["systs"] == bm["systs"]
        ), f"syst lists differ: fast {bf['systs']} vs manual {bm['systs']}"
        assert np.allclose(
            bf["hnorm"], bm["hnorm"]
        ), "hnorm mismatch (batched fast vs manual)"
        assert np.allclose(
            bf["hlogk"], bm["hlogk"]
        ), "hlogk mismatch (batched fast vs manual)"
        print("PASS: batched SparseHist path matches per-syst manual booking")

        print()
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
