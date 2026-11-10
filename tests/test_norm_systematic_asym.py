"""Asymmetric lnN uncertainties from add_norm_systematic, for every symmetrization.

Each process of the list books its own logk values. Passing the whole process
list to _compute_asym_syst used to raise "unhashable type: 'list'" for every
mode that books inside it (None, "linear", "quadratic"), even for one process.
Covers the dense path and the sparse-norm path (SparseHist processes).
"""

import hist
import numpy as np
import pytest
import scipy.sparse

from rabbit import tensorwriter

UNCERTAINTIES = {"sig": [1.05, 0.97], "bkg": [1.10, 0.80]}


def make_writer(sparse_input):
    ax = hist.axis.Regular(4, 0, 4, name="x", underflow=False, overflow=False)
    h_data = hist.Hist(ax, storage=hist.storage.Double())
    h_data.values()[...] = [10.0, 20.0, 30.0, 40.0]
    writer = tensorwriter.TensorWriter(sparse=sparse_input)
    writer.add_channel([ax], "ch0")
    writer.add_data(h_data, "ch0")
    for i, name in enumerate(UNCERTAINTIES):
        h = hist.Hist(ax, storage=hist.storage.Weight())
        h.values()[...] = (i + 1) * np.array([1.0, 2.0, 0.0, 4.0])
        h.variances()[...] = h.values()
        if sparse_input:
            values = scipy.sparse.csr_array(h.values().reshape(1, -1))
            writer.add_process(
                tensorwriter.SparseHist(values, [ax]),
                name,
                "ch0",
                signal=name == "sig",
                variances=h.variances(),
            )
        else:
            writer.add_process(h, name, "ch0", signal=name == "sig")
    return writer


def booked(values):
    values = np.asarray(values)
    return np.unique(values[values != 0.0])


@pytest.mark.parametrize("sparse_input", [False, True])
@pytest.mark.parametrize(
    "symmetrize", [None, "linear", "quadratic", "average", "conservative"]
)
def test_asymmetric_norm_systematic_per_process(sparse_input, symmetrize):
    writer = make_writer(sparse_input)
    writer.add_norm_systematic(
        "norm",
        list(UNCERTAINTIES),
        "ch0",
        list(UNCERTAINTIES.values()),
        symmetrize=symmetrize,
    )
    avg_name = "normSymAvg" if symmetrize in ("linear", "quadratic") else "norm"
    for process, (up, down) in UNCERTAINTIES.items():
        logkup, logkdown = np.log(up), -np.log(down)
        if symmetrize == "conservative":
            expected = logkup if abs(logkup) > abs(logkdown) else logkdown
        else:
            expected = 0.5 * (logkup + logkdown)
        np.testing.assert_allclose(
            booked(writer.dict_logkavg["ch0"][process][avg_name]), [expected]
        )
        halfdiff = 0.5 * (logkup - logkdown)
        if symmetrize is None:
            np.testing.assert_allclose(
                booked(writer.dict_logkhalfdiff["ch0"][process]["norm"]), [halfdiff]
            )
        elif symmetrize in ("linear", "quadratic"):
            factor = np.sqrt(3.0) if symmetrize == "quadratic" else 1.0
            np.testing.assert_allclose(
                booked(writer.dict_logkavg["ch0"][process]["normSymDiff"]),
                [factor * halfdiff],
            )


@pytest.mark.parametrize("sparse_input", [False, True])
def test_asymmetric_norm_systematic_single_process(sparse_input):
    writer = make_writer(sparse_input)
    writer.add_norm_systematic("x", "bkg", "ch0", [1.05, 0.97], symmetrize=None)
    np.testing.assert_allclose(
        booked(writer.dict_logkhalfdiff["ch0"]["bkg"]["x"]),
        [0.5 * (np.log(1.05) + np.log(0.97))],
    )


@pytest.mark.parametrize(
    "uncertainty, expected",
    [(1.05, np.log(1.05)), ([1.05], np.log(1.05)), ([(1.05, 0.97)], None)],
)
def test_single_process_existing_forms(uncertainty, expected):
    writer = make_writer(False)
    writer.add_norm_systematic("x", "bkg", "ch0", uncertainty)
    if expected is None:
        expected = 0.5 * (np.log(1.05) - np.log(0.97))
    np.testing.assert_allclose(
        booked(writer.dict_logkavg["ch0"]["bkg"]["x"]), [expected]
    )
