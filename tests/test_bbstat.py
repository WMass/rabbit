"""Unit tests for the standalone BinByBinStat module.

Constructs a minimal stand-in for ``FitInputData`` (``sumw``, ``sumw2``,
``dtype``, ``nbins``) so the BBB class can be exercised without
spinning up the full Fitter.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from rabbit.bbstat.bbstat import VALID_BIN_BY_BIN_STAT_TYPES, BinByBinStat
from rabbit.bbstat.formulas import solve_quad_eq


def _make_indata(sumw, sumw2):
    """Tiny FitInputData stand-in carrying just what BinByBinStat reads."""
    sumw = tf.constant(sumw, dtype=tf.float64)
    sumw2 = tf.constant(sumw2, dtype=tf.float64)
    nbins = int(sumw.shape[0])
    return SimpleNamespace(
        sumw=sumw,
        sumw2=sumw2,
        nbins=nbins,
        nbinsmasked=0,
        nbinsfull=nbins,
        dtype=tf.float64,
        norm=None,
        betavar=None,
    )


def _make_options(**kwargs):
    defaults = dict(
        noBinByBinStat=False,
        binByBinStatMode="lite",
        binByBinStatType="automatic",
        minBBKstat=0.0,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_bbstat(sumw, sumw2, **opts):
    indata = _make_indata(sumw, sumw2)
    options = _make_options(**opts)
    nobs_template = tf.zeros((indata.nbins,), dtype=tf.float64)
    return BinByBinStat(
        indata,
        options,
        chisqFit=False,
        covarianceFit=False,
        data_cov_inv=None,
        nobs_template=nobs_template,
    )


# --- solve_quad_eq -----------------------------------------------------------


def test_solve_quad_eq_simple_root():
    # x² - 5x + 6 = 0 → roots 2 and 3; positive root the function returns is 3.
    a, b, c = 1.0, -5.0, 6.0
    assert float(solve_quad_eq(a, b, c)) == pytest.approx(3.0)


def test_solve_quad_eq_lite_gamma_all_finite_variance():
    # Closed form for gamma+lite when all processes have finite variance:
    # β = (n + k·β0) / (N + k).
    # The new formula is: a·β² + b·β + c = 0 with
    #   a = N·(N + k), b = N·(0 - n - β0·k) + k·0 = -N·(n + β0·k), c = 0.
    # Positive root = (n + β0·k) / (N + k).
    n, N, k, beta0 = 50.0, 100.0, 25.0, 1.0
    a = N * (N + k)
    b = N * (0.0 - n - beta0 * k) + k * 0.0
    c = -beta0 * k * 0.0
    expected = (n + beta0 * k) / (N + k)
    got = float(solve_quad_eq(a, b, c))
    assert got == pytest.approx(expected)


# --- default_beta0 -----------------------------------------------------------


@pytest.mark.parametrize(
    "stat_type, expected_value",
    [("gamma", 1.0), ("normal-multiplicative", 1.0), ("normal-additive", 0.0)],
)
def test_default_beta0(stat_type, expected_value):
    bb = _make_bbstat(
        sumw=[10.0, 20.0],
        sumw2=[10.0, 20.0],
        binByBinStatType=stat_type,
    )
    assert np.allclose(bb.default_beta0().numpy(), expected_value)


# --- lbeta closed forms ------------------------------------------------------


def test_lbeta_at_beta_equal_beta0_is_zero_for_normal_types():
    """At β = β0 the constraint contribution is exactly 0 for normal forms."""
    for stat_type in ("normal-multiplicative", "normal-additive"):
        bb = _make_bbstat(
            sumw=[10.0, 20.0],
            sumw2=[10.0, 20.0],
            binByBinStatType=stat_type,
        )
        beta = bb.beta0
        val = bb.lbeta(beta).numpy()
        assert val == pytest.approx(0.0)


def test_lbeta_at_beta_equal_beta0_is_zero_for_gamma():
    """For gamma at β = β0 = 1 the offset form k·(β-β0) - k·β0·(log β - log β0)
    evaluates to k·0 - k·1·0 = 0."""
    bb = _make_bbstat(
        sumw=[10.0, 20.0],
        sumw2=[10.0, 20.0],
        binByBinStatType="gamma",
    )
    val = bb.lbeta(bb.beta0).numpy()
    assert val == pytest.approx(0.0)


def test_lbeta_normal_additive_quadratic():
    """L = 0.5 Σ (β - β0)² for additive normal."""
    bb = _make_bbstat(
        sumw=[10.0, 20.0],
        sumw2=[10.0, 20.0],
        binByBinStatType="normal-additive",
    )
    beta = tf.constant([0.5, -1.0], dtype=tf.float64)
    val = bb.lbeta(beta).numpy()
    expected = 0.5 * (0.5**2 + (-1.0) ** 2)
    assert val == pytest.approx(expected)


def test_lbeta_normal_multiplicative_quadratic_with_kstat():
    """L = Σ 0.5·(β - β0)²·k for multiplicative normal (in non-masked bins)."""
    bb = _make_bbstat(
        sumw=[10.0, 20.0],
        sumw2=[5.0, 10.0],
        binByBinStatType="normal-multiplicative",
    )
    # kstat = sumw² / sumw2 = [100/5, 400/10] = [20, 40]
    np.testing.assert_allclose(bb.kstat.numpy(), [20.0, 40.0])
    beta = tf.constant([1.1, 0.9], dtype=tf.float64)
    val = bb.lbeta(beta).numpy()
    expected = 0.5 * (0.01 * 20.0 + 0.01 * 40.0)
    assert val == pytest.approx(expected)


# --- masking and minBBKstat --------------------------------------------------


def test_betamask_when_sumw_or_sumw2_zero():
    bb = _make_bbstat(
        sumw=[10.0, 0.0, 20.0],
        sumw2=[10.0, 5.0, 0.0],
    )
    expected_mask = [False, True, True]
    np.testing.assert_array_equal(bb.betamask.numpy(), expected_mask)


def test_min_bb_kstat_extends_betamask():
    # kstat = sumw²/sumw2 = [4, 100, 100] → minBBKstat=10 masks first bin.
    bb = _make_bbstat(
        sumw=[2.0, 10.0, 10.0],
        sumw2=[1.0, 1.0, 1.0],
        minBBKstat=10.0,
    )
    expected_mask = [True, False, False]
    np.testing.assert_array_equal(bb.betamask.numpy(), expected_mask)


# --- selective lite-mode for zero-variance processes -------------------------


def test_proc_zero_var_mask_detection():
    """sumw2 == 0 entries are flagged as zero-variance and excluded from sumw."""
    sumw = np.array([[1.0, 5.0], [2.0, 0.0]], dtype=np.float64)
    sumw2 = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    bb = _make_bbstat(sumw=sumw, sumw2=sumw2, binByBinStatMode="lite")
    # First proc has finite variance (sumw2>0 in both rows), second is fully
    # zero-variance (sumw2=0).
    assert bb.proc_zero_var_mask is not None
    expected = sumw2 == 0.0
    np.testing.assert_array_equal(bb.proc_zero_var_mask.numpy(), expected)
    # Merged sumw should NOT include the zero-variance contribution.
    np.testing.assert_allclose(bb.sumw.numpy(), [1.0, 2.0])


def test_full_mode_skips_zero_var_detection():
    """In full mode there's no merging; per-(bin, proc) betamask handles the
    sumw2==0 entries individually."""
    sumw = np.array([[1.0, 5.0]], dtype=np.float64)
    sumw2 = np.array([[1.0, 0.0]], dtype=np.float64)
    bb = _make_bbstat(sumw=sumw, sumw2=sumw2, binByBinStatMode="full")
    assert bb.proc_zero_var_mask is None
    # betamask is (bin, proc) shape and True where sumw2==0 OR sumw==0.
    assert bb.betamask.shape == (1, 2)
    np.testing.assert_array_equal(bb.betamask.numpy(), [[False, True]])


# --- valid-types validation --------------------------------------------------


def test_invalid_stat_type_raises():
    with pytest.raises(RuntimeError):
        _make_bbstat(
            sumw=[10.0],
            sumw2=[10.0],
            binByBinStatType="banana",
        )


def test_valid_types_constant():
    assert "gamma" in VALID_BIN_BY_BIN_STAT_TYPES
    assert "normal-additive" in VALID_BIN_BY_BIN_STAT_TYPES
    assert "normal-multiplicative" in VALID_BIN_BY_BIN_STAT_TYPES


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
