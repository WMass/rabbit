"""CompositeParamModel must preserve the fitter-facing [POIs | POUs] layout.

Regression test for the bug where a POU-carrying model placed before a
POI-carrying one (e.g. a custom npou>0 param model composited with
SaturatedProjectModel by --computeSaturatedProjectionTests) leaked its POUs
into the composite's POI slice — the fitter then silently squared and
blinded them.
"""

import numpy as np
import pytest
import tensorflow as tf

from rabbit.param_models.param_model import CompositeParamModel


class PouOnly:
    """Mimics a POU-only model (npoi=0), e.g. a theory-parameter model."""

    npoi = 0
    npou = 2
    nparams = 2
    params = np.array([b"a1", b"a2"])
    xparamdefault = tf.constant([0.4, 2.0], dtype=tf.float64)
    allowNegativeParam = True
    is_linear = False
    prior_sigmas = np.array([0.5, np.nan])
    param_impact_groups = {"groupA": [b"a1", b"a2"]}
    w = tf.constant([1.0, 10.0], dtype=tf.float64)

    def compute(self, param, full=False):
        return 1.0 + tf.reduce_sum(param * self.w)


class PoiAndPou:
    """Mimics a POI-carrying model (e.g. SaturatedProjectModel + one POU)."""

    npoi = 2
    npou = 1
    nparams = 3
    params = np.array([b"b1", b"b2", b"bp"])
    xparamdefault = tf.constant([1.0, 1.0, 3.0], dtype=tf.float64)
    allowNegativeParam = False
    is_linear = False
    w = tf.constant([100.0, 1000.0, 10000.0], dtype=tf.float64)

    def compute(self, param, full=False):
        return 1.0 + tf.reduce_sum(param * self.w)


def test_pou_model_first_layout():
    # the configuration that was broken: POU-carrying model first
    c = CompositeParamModel([PouOnly(), PoiAndPou()])

    assert c.npoi == 2 and c.npou == 3
    assert list(c.params) == [b"b1", b"b2", b"a1", b"a2", b"bp"]
    # the first npoi names must be exactly the POIs (this is the slice that
    # get_poi squaring, blinding, and output reporting act on)
    assert list(c.params[: c.npoi]) == [b"b1", b"b2"]
    np.testing.assert_allclose(c.xparamdefault.numpy(), [1.0, 1.0, 0.4, 2.0, 3.0])


def test_compute_reassembles_native_order():
    A, B = PouOnly(), PoiAndPou()
    c = CompositeParamModel([A, B])
    param = tf.constant([1.5, 2.5, -0.3, 1.1, 4.0], dtype=tf.float64)
    expected = A.compute(tf.constant([-0.3, 1.1], dtype=tf.float64)) * B.compute(
        tf.constant([1.5, 2.5, 4.0], dtype=tf.float64)
    )
    np.testing.assert_allclose(c.compute(param).numpy(), expected.numpy(), rtol=1e-14)

    # gradient flows through the permutation
    with tf.GradientTape() as t:
        t.watch(param)
        val = c.compute(param)
    g = t.gradient(val, param).numpy()
    assert np.all(np.isfinite(g)) and np.all(g != 0)


def test_legacy_valid_ordering_unchanged():
    A, B = PouOnly(), PoiAndPou()
    c = CompositeParamModel([B, A])
    assert list(c.params) == [b"b1", b"b2", b"bp", b"a1", b"a2"]
    param = tf.constant([1.5, 2.5, 4.0, -0.3, 1.1], dtype=tf.float64)
    expected = A.compute(tf.constant([-0.3, 1.1], dtype=tf.float64)) * B.compute(
        tf.constant([1.5, 2.5, 4.0], dtype=tf.float64)
    )
    np.testing.assert_allclose(c.compute(param).numpy(), expected.numpy(), rtol=1e-14)


def test_allow_negative_derived_from_poi_models_only():
    # the POU-only model has allowNegativeParam=True, but only the
    # POI-carrying model's flag matters for the squaring transform
    c = CompositeParamModel([PouOnly(), PoiAndPou()])
    assert c.allowNegativeParam is False


def test_priors_and_impact_groups_propagated():
    c = CompositeParamModel([PouOnly(), PoiAndPou()])
    np.testing.assert_allclose(c.prior_sigmas, [np.nan, np.nan, 0.5, np.nan, np.nan])
    np.testing.assert_allclose(c.prior_means, [1.0, 1.0, 0.4, 2.0, 3.0])
    assert c.param_impact_groups == {"groupA": [b"a1", b"a2"]}


def test_conflicting_flags_raise():
    class PoiNeg(PoiAndPou):
        allowNegativeParam = True

    with pytest.raises(ValueError):
        CompositeParamModel([PoiAndPou(), PoiNeg()])

    with pytest.raises(ValueError):
        CompositeParamModel([PouOnly(), PoiAndPou()], allowNegativeParam=True)

    # explicit flag consistent with submodels is accepted
    c = CompositeParamModel([PouOnly(), PoiAndPou()], allowNegativeParam=False)
    assert c.allowNegativeParam is False
    # POU-only composite: explicit flag respected (no POI carrier to conflict)
    assert (
        CompositeParamModel([PouOnly()], allowNegativeParam=True).allowNegativeParam
        is True
    )
