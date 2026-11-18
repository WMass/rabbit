"""add_norm_systematic with one uncertainty for several processes books it on
every process, not only the first."""

import hist
import numpy as np
import pytest

from rabbit.tensorwriter import TensorWriter

PROCS = ["sig", "bkgA", "bkgB"]


def _writer(sparse=False):
    ax = hist.axis.Regular(4, 0.0, 1.0, name="x")
    w = TensorWriter(sparse=sparse)
    w.add_channel([ax], "ch")
    data = hist.Hist(ax, storage=hist.storage.Double())
    data.values()[...] = 100.0
    w.add_data(data, "ch")
    for i, p in enumerate(PROCS):
        h = hist.Hist(ax, storage=hist.storage.Weight())
        h.values()[...] = 10.0 * (i + 1)
        h.variances()[...] = 10.0 * (i + 1)
        w.add_process(h, p, "ch", signal=(p == "sig"))
    return w


@pytest.mark.parametrize(
    "uncertainty",
    [1.02, [1.02], (1.02,), np.array([1.02])],
    ids=["scalar", "list", "tuple", "array"],
)
def test_single_kappa_books_every_process(uncertainty):
    w = _writer()
    w.add_norm_systematic("norm", PROCS, "ch", uncertainty)
    for p in PROCS:
        assert "norm" in w.dict_logkavg["ch"][p], p
        np.testing.assert_allclose(
            w.dict_logkavg["ch"][p]["norm"], np.log(1.02), rtol=1e-12
        )


def test_single_asymmetric_kappa_books_every_process():
    w = _writer()
    w.add_norm_systematic("norm", PROCS, "ch", [(1.05, 0.97)])
    for p in PROCS:
        np.testing.assert_allclose(
            w.dict_logkavg["ch"][p]["norm"],
            0.5 * (np.log(1.05) - np.log(0.97)),
            rtol=1e-12,
        )


def test_per_process_kappas_unchanged():
    w = _writer()
    w.add_norm_systematic("norm", PROCS, "ch", [1.01, 1.02, 1.03])
    for p, kappa in zip(PROCS, [1.01, 1.02, 1.03]):
        np.testing.assert_allclose(
            w.dict_logkavg["ch"][p]["norm"], np.log(kappa), rtol=1e-12
        )


def test_wrong_length_still_raises():
    w = _writer()
    with pytest.raises(RuntimeError):
        w.add_norm_systematic("norm", PROCS, "ch", [1.01, 1.02])
