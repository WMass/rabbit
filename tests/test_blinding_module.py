"""Unit tests for the Blinding collaborator, with no Fitter in sight.

Everything here constructs :class:`rabbit.blinding.Blinding` directly against
stand-in ``indata`` / ``param_model`` objects: the draw, the group and
exemption policy, arming, the hot-path application and the weak-smearing
verdict are all reachable without a tensor file, a likelihood or a fit.

That is what the extraction bought. tests/test_blinding.py has to build a toy
model, write a tensor and run a fit before it can assert that an offset
cancels -- the right shape for the INVARIANCES it checks, and far too much
machinery for "is this scale wide enough to hide this parameter", which is a
question about the offsets alone.

The fitter-level behaviour (that the offsets cancel out of the physics, the
covariance and the start point) stays in tests/test_blinding.py,
tests/test_blinding_noi.py and tests/test_blinding_start_point.py.

As there, no test asserts on an offset VALUE: the checks are about which
parameters are offset, whether two offsets agree, and whether a declared scale
is wide enough -- never what the number is.
"""

import numpy as np
import pytest
import tensorflow as tf

from rabbit.blinding import BLINDING_DRAW_STD, Blinding, BlindingView


class FakeIndata:
    """The handful of input-tensor attributes Blinding reads.

    ``data_obs`` is read only to decide whether the fit is on real data, which
    the draw folds into its seed so a toy does not share a data fit's offsets.
    """

    def __init__(
        self,
        systs=(b"bkgNorm", b"noi1", b"noi2"),
        noiidxs=(1, 2),
        integer_data=True,
    ):
        self.systs = np.array(list(systs))
        self.nsyst = len(self.systs)
        self.noiidxs = np.array(list(noiidxs), dtype=np.int64)
        self.data_obs = np.array([3.0, 7.0] if integer_data else [3.5, 7.5])
        self.dtype = tf.float64


class FakeModel:
    """The POI-side declarations Blinding reads off a param model."""

    def __init__(self, params=(b"mu",), allowNegativeParam=True, **declarations):
        self.params = np.array(list(params))
        self.npoi = len(self.params)
        self.allowNegativeParam = allowNegativeParam
        # blind_additive_scale / blind_exempt / blind_exempt_params are all
        # optional: the baseline model declares none of them
        for name, value in declarations.items():
            setattr(self, name, value)


def make(indata=None, model=None, **kwargs):
    return Blinding(
        indata if indata is not None else FakeIndata(),
        model if model is not None else FakeModel(),
        enabled=True,
        **kwargs,
    )


def test_the_draw_is_deterministic_in_the_parameter_name():
    """Two runs of the same analysis must blind identically, or results from
    different jobs cannot be compared -- and two different parameters must not
    share an offset, or their difference gives one of them away."""
    assert make().values_poi_add[0] == make().values_poi_add[0]
    assert (
        make().values_poi_add[0] != make(model=FakeModel((b"mu2",))).values_poi_add[0]
    )


def test_the_draw_separates_a_real_data_card_from_an_mc_card():
    """The seed folds in whether the card's data_obs is integer, so a fit to an
    Asimov or MC card does not run in a real-data card's frame. It is the
    card that decides: --pseudoData on a real-data card shares its frame."""
    data = Blinding(FakeIndata(integer_data=True), FakeModel(), enabled=True)
    toy = Blinding(FakeIndata(integer_data=False), FakeModel(), enabled=True)
    assert data.values_poi_add[0] != toy.values_poi_add[0]


def test_an_empty_bin_does_not_make_an_mc_card_look_like_data():
    """An exactly-0.0 bin is integral, and empty bins are common in an MC card.
    One of them must not put the card in the real-data frame, or a closure fit
    against known truth reads the data offsets straight off."""
    data = Blinding(FakeIndata(integer_data=True), FakeModel(), enabled=True)
    ind = FakeIndata(integer_data=False)
    ind.data_obs = np.array([0.0, 7.5])
    mixed = Blinding(ind, FakeModel(), enabled=True)
    assert data.values_poi_add[0] != mixed.values_poi_add[0]


def test_unblind_leaves_exactly_the_named_parameters_in_their_true_frame():
    """An offset of exactly zero is how "not blinded" is represented; every
    later decision (the weak-smearing verdict, the squared-storage warning)
    reads it back that way."""
    b = make(unblind=["mu"])
    assert b.values_poi_add[0] == 0.0
    assert np.count_nonzero(b.values_theta) == 2, "the NOIs must be untouched"

    n = make(unblind=["noi1"])
    assert n.values_theta[1] == 0.0
    assert n.values_theta[2] != 0.0
    assert n.values_poi_add[0] != 0.0


def test_only_the_nois_among_the_nuisances_are_offset():
    """theta carries every systematic, but a constrained nuisance is not a
    result: only indata.noiidxs entries are drawn for."""
    b = make()
    assert b.values_theta[0] == 0.0, "constrained systematic must not be offset"
    assert (b.values_theta[[1, 2]] != 0.0).all()


def test_a_blinding_group_shares_one_offset():
    """The point of --blindingGroup: members keep their relative differences,
    which a per-name draw destroys, while the absolute values stay hidden."""
    ind = FakeIndata(systs=(b"c", b"noi_y0", b"noi_y1"), noiidxs=(1, 2))
    grouped = Blinding(ind, FakeModel(), enabled=True, blinding_group=[r"noi_y\d+"])
    assert grouped.values_theta[1] == grouped.values_theta[2] != 0.0

    ungrouped = Blinding(ind, FakeModel(), enabled=True)
    assert ungrouped.values_theta[1] != ungrouped.values_theta[2]


def test_unblind_overlapping_a_group_is_refused():
    """Ambiguous intent, and the wrong guess unblinds something sensitive."""
    with pytest.raises(RuntimeError, match="both --unblind and --blindingGroup"):
        make(unblind=["mu"], blinding_group=["mu"])


def test_a_model_can_declare_its_own_parameters_exempt():
    """Auxiliary quantities that are not results (the saturated test's per-bin
    scales) are exempt without the user passing --unblind for them."""
    assert make(model=FakeModel(blind_exempt=True)).values_poi_add[0] == 0.0

    named = make(
        model=FakeModel((b"mu", b"aux"), blind_exempt_params=[b"aux"]),
    )
    assert named.values_poi_add[1] == 0.0
    assert named.values_poi_add[0] != 0.0


def test_the_declared_scale_multiplies_the_draw():
    """blind_additive_scale is in the POI's own fit units, scalar for a whole
    model or per-POI as CompositeParamModel produces."""
    plain = make().values_poi_add[0]
    assert make(model=FakeModel(blind_additive_scale=7.0)).values_poi_add[
        0
    ] == pytest.approx(7.0 * plain)

    base = make(model=FakeModel((b"mu", b"mu2"))).values_poi_add
    per_poi = make(
        model=FakeModel((b"mu", b"mu2"), blind_additive_scale=[7.0, 0.5])
    ).values_poi_add
    np.testing.assert_allclose(per_poi, [7.0 * base[0], 0.5 * base[1]])


def test_arming_is_what_puts_the_drawn_offsets_into_the_variables():
    """Constructed disarmed on purpose: nothing is in the frame until a caller
    asks for it, and disarming has to restore exact zeros rather than
    "something small"."""
    b = make()
    np.testing.assert_array_equal(b.offsets_poi_add.numpy(), np.zeros(1))
    np.testing.assert_array_equal(b.offsets_theta.numpy(), np.zeros(3))

    b._arm(True)
    np.testing.assert_allclose(b.offsets_poi_add.numpy(), b.values_poi_add)
    np.testing.assert_allclose(b.offsets_theta.numpy(), b.values_theta)

    b._arm(False)
    np.testing.assert_array_equal(b.offsets_poi_add.numpy(), np.zeros(1))
    np.testing.assert_array_equal(b.offsets_theta.numpy(), np.zeros(3))


def test_apply_shifts_by_the_armed_offset_and_nothing_else():
    b = make()
    xpoi = tf.constant([0.3], dtype=tf.float64)
    theta = tf.constant([0.1, 0.2, 0.3], dtype=tf.float64)

    np.testing.assert_array_equal(b.apply_poi(xpoi).numpy(), xpoi.numpy())
    np.testing.assert_array_equal(b.apply_theta(theta).numpy(), theta.numpy())

    b._arm(True)
    np.testing.assert_allclose(
        b.apply_poi(xpoi).numpy(), xpoi.numpy() + b.values_poi_add
    )
    np.testing.assert_allclose(
        b.apply_theta(theta).numpy(), theta.numpy() + b.values_theta
    )


def test_a_disabled_instance_owns_nothing_and_applies_nothing():
    """--unblind everything, or a fit that was never blinded: the collaborator
    still exists, so the Fitter's callers need no branch of their own."""
    b = Blinding(FakeIndata(), FakeModel(), enabled=False)
    assert b.offsets_poi_add is None and b.values_poi_add is None
    assert b.offsets_theta is None and b.values_theta is None

    xpoi = tf.constant([0.3], dtype=tf.float64)
    assert b.apply_poi(xpoi) is xpoi
    b._arm(True)  # a no-op, not an AttributeError
    assert b.apply_poi(xpoi) is xpoi


def test_the_shard_view_applies_what_the_fitter_applies():
    """rabbit.sharding pins the offset VALUES on a BlindingView because XLA
    cannot read a Variable on another device. The two must agree, or a sharded
    fit silently minimises in a different frame from a single-device one."""
    b = make()
    b._arm(True)

    view = BlindingView(True)
    view.offsets_poi_add = tf.identity(b.offsets_poi_add)
    view.offsets_theta = tf.identity(b.offsets_theta)

    xpoi = tf.constant([0.3], dtype=tf.float64)
    theta = tf.constant([0.1, 0.2, 0.3], dtype=tf.float64)
    np.testing.assert_array_equal(
        view.apply_poi(xpoi).numpy(), b.apply_poi(xpoi).numpy()
    )
    np.testing.assert_array_equal(
        view.apply_theta(theta).numpy(), b.apply_theta(theta).numpy()
    )


MSG = "too narrow to hide"


def test_a_smearing_narrower_than_the_measured_sigma_is_reported(caplog):
    """The verdict is BLINDING_DRAW_STD * scale against 5 sigma, and sigma is
    whatever the fit measured. Same draw, same parameter -- only the
    uncertainty it is judged against differs."""
    b = make()
    b._arm(True)

    sigma_wide = BLINDING_DRAW_STD / 5.0 * 1.1  # smearing < 5 sigma
    with caplog.at_level("WARNING"):
        b.warn_if_weak(np.array([sigma_wide**2]))
    assert any(MSG in r.message for r in caplog.records)

    caplog.clear()
    sigma_tight = BLINDING_DRAW_STD / 5.0 * 0.9  # smearing > 5 sigma
    with caplog.at_level("WARNING"):
        b.warn_if_weak(np.array([sigma_tight**2]))
    assert not any(MSG in r.message for r in caplog.records)


def test_the_weak_smearing_verdict_prints_no_numbers(caplog):
    """The offset and the width each give the other away, so the warning says
    which POIs and which knob, never how much."""
    b = make(model=FakeModel(blind_additive_scale=1e-9))
    b._arm(True)
    with caplog.at_level("WARNING"):
        b.warn_if_weak(np.array([1.0]))

    text = " ".join(r.message for r in caplog.records)
    assert MSG in text
    assert "blind_additive_scale" in text
    assert str(abs(float(b.values_poi_add[0])))[:6] not in text
    assert "1e-09" not in text and "1e-9" not in text


def test_nothing_is_reported_for_a_parameter_that_was_never_blinded(caplog):
    """An --unblind'ed POI has an offset of exactly zero; warning that its
    (absent) smearing is too narrow would send the reader after a fix for a
    parameter they deliberately opened."""
    b = make(model=FakeModel(blind_additive_scale=1e-9), unblind=["mu"])
    b._arm(True)
    with caplog.at_level("WARNING"):
        b.warn_if_weak(np.array([1.0]))
    assert not any(MSG in r.message for r in caplog.records)


def test_a_sigma_that_was_not_computed_is_skipped(caplog):
    """Under --noHessian only the POI and NOI variances are solved for and the
    rest are left NaN; a non-finite entry means "no verdict", not "weak"."""
    b = make()
    b._arm(True)
    with caplog.at_level("WARNING"):
        b.warn_if_weak(np.array([np.nan]))
    assert not any(MSG in r.message for r in caplog.records)
