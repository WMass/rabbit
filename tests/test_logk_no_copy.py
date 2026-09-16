"""The 'normal' systematic scaling is applied after the contraction, not baked
into a copy of logk.

For systematic_type == "normal" the param-model factor rnorm_init used to be
absorbed into logk at construction, allocating a second full-size
[nbins, nproc, nsyst] tensor (and rebuilding the CSR matrix in sparse mode).
rnorm_init carries no systematic index, so it factors out of the contraction:

    sum_s (rnorm_init[b,p] * logk[b,p,s]) * theta[s]
      == rnorm_init[b,p] * sum_s logk[b,p,s] * theta[s]

and it is applied to the [nbins, nproc] result instead. These tests pin both
halves of that: that no copy is made, and that dropping it changes nothing.
"""

import os
import tempfile

import numpy as np
import pytest

from rabbit import fitter, inputdata, tensorwriter
from rabbit.param_models.helpers import load_model
from tests.test_sparse_fit import make_histograms, make_options

# The scaling is the identity unless the param model's default differs from 1,
# so a model at its usual mu = 1 cannot tell the two forms apart.
EXPECT_SIGNAL = [("sig", 1.7)]


def _normal_tensor(outdir, sparse):
    h = make_histograms()
    w = tensorwriter.TensorWriter(sparse=sparse, systematic_type="normal")
    w.add_channel(h["data"].axes, "ch0")
    w.add_data(h["data"], "ch0")
    w.add_process(h["sig"], "sig", "ch0", signal=True)
    w.add_process(h["bkg"], "bkg", "ch0")
    w.add_norm_systematic("bkg_norm", "bkg", "ch0", 1.05)
    # rnorm_init is non-unit only on the signal process, so the scaling is the
    # identity unless a systematic actually touches sig -- see the control in
    # test_post_contraction_scaling_equals_premultiplying_logk.
    w.add_norm_systematic("sig_norm", "sig", "ch0", 1.10)
    w.add_systematic(
        [h["syst_up"], h["syst_dn"]], "bkg_shape", "bkg", "ch0", symmetrize="average"
    )
    name = f"normal_{'sparse' if sparse else 'dense'}"
    w.write(outfolder=outdir, outfilename=name)
    return os.path.join(outdir, f"{name}.hdf5")


def _make(outdir, sparse):
    indata = inputdata.FitInputData(_normal_tensor(outdir, sparse))
    model = load_model("Mu", indata, expectSignal=EXPECT_SIGNAL)
    f = fitter.Fitter(indata, model, make_options())
    f.set_nobs(indata.data_obs)
    return indata, f


@pytest.mark.parametrize("sparse", [False, True])
def test_logk_is_aliased_not_copied(sparse):
    """The point of the change: no second full-size tensor is allocated."""
    with tempfile.TemporaryDirectory() as tmp:
        indata, f = _make(tmp, sparse)
        assert f.logk is indata.logk, "logk was copied rather than aliased"
        if sparse:
            assert f.logk_csr is indata.logk_csr, "logk_csr was rebuilt"


def test_rnorm_init_is_the_value_the_hot_path_uses():
    """The factor must be the model evaluated where the yields evaluate it.

    ``compute()`` consumes PHYSICAL parameter values -- everywhere else it is
    handed ``get_poi()``, which undoes the storage transform. ``xparamdefault``
    is in x-space, so with the default ``allowNegativeParam=False`` it holds
    ``sqrt(mu)``; passing it raw evaluates the model at ``sqrt(mu)`` and the
    factor lands at ``1/sqrt(mu)`` of what the yield computation uses at the
    same point.

    Compared against ``compute(get_poi())`` rather than against a literal,
    because the claim is an agreement between two call sites and not a
    particular number. The check above cannot see this: ``rnorm_init != 1``
    is satisfied by ``sqrt(1.7) = 1.304`` exactly as happily as by ``1.7``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        indata, f = _make(tmp, sparse=False)
        assert f.rnorm_init is not None, "scaling not armed; test is vacuous"
        assert not f.param_model.allowNegativeParam, "squared storage is the case"

        f.xdefaultassign()
        hot = np.asarray(f.param_model.compute(f.get_poi(), True))
        init = np.asarray(f.rnorm_init)

        # and the default really is off 1, or sqrt() is the identity here
        assert not np.allclose(hot[0], 1.0), "POI default is 1; test is vacuous"

        np.testing.assert_allclose(init[0], hot[0], rtol=1e-12, atol=0)


def test_scaling_factors_out_of_the_contraction():
    """The algebraic premise, on the tensors the fitter actually holds.

    Pre-multiplying logk by rnorm_init and contracting must equal contracting
    and then scaling the result -- which is what licenses applying the factor
    to the [nbins, nproc] result instead of the [nbins, nproc, nsyst] input.
    Checked on the real logk/rnorm_init rather than on invented arrays, so a
    misaligned or wrongly-shaped rnorm_init fails here: the identity holds
    elementwise for the true [nbins, nproc] layout and not for a transposed
    or mis-sliced one.

    This pins the premise, not the wiring. The yield path bakes rnorm_init in
    at construction, so it cannot be swapped out afterwards to compare the
    two forms end to end.
    """
    with tempfile.TemporaryDirectory() as tmp:
        indata, f = _make(tmp, sparse=False)
        assert f.rnorm_init is not None, "scaling not armed; test is vacuous"

        logk = np.asarray(indata.logk)
        rnorm_init = np.asarray(f.rnorm_init)
        assert rnorm_init.shape == logk.shape[:2], (
            f"rnorm_init {rnorm_init.shape} does not match the [nbins, nproc] "
            f"prefix of logk {logk.shape}"
        )
        # the factor must actually bite somewhere a systematic reaches, or the
        # identity below is 1 == 1
        touched = np.abs(logk).sum(axis=2) > 0
        assert np.any(touched & (rnorm_init != 1.0)), (
            "no systematic touches a process whose rnorm_init differs from 1, "
            "so this tensor cannot distinguish the two forms"
        )

        rng = np.random.default_rng(0)
        theta = rng.normal(size=logk.shape[2])

        baked = np.einsum("bps,s->bp", logk * rnorm_init[:, :, None], theta)
        after = rnorm_init * np.einsum("bps,s->bp", logk, theta)
        np.testing.assert_allclose(baked, after, rtol=1e-13, atol=0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
