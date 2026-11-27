"""Toys leave the masked bin-by-bin stat entries at their defaults.

Entries without bin-by-bin stat uncertainty (betamask: sumw2 == 0 or
sumw == 0, or below --minBBKstat) have no constraint in the multiplicative
types (gamma, normal-multiplicative) and their beta scales nothing; kstat is a
placeholder 1 there. Frequentist toys drew their centres anyway, the gamma
type as Poisson(beta) / 1, and full mode's profile then counted such a
process's yield as beta0 * norm while the prediction keeps norm: the profiled
betas were not the minimum of the likelihood the fit evaluates. Bayesian toys
drew their beta, which the constraint term then counted when not profiling.
normal-additive does not use betamask (an entry with sumw == 0 but sumw2 > 0
shifts its bin there), so its toys are unchanged.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from rabbit.bbstat.bbstat import BinByBinStat

# Six bins, three processes: "sig" (weight 3, kstat 50 to 300), "bkg"
# (weight 0.5, kstat 400) and "flat" without MC stat uncertainty (masked).
NSIG = np.array([50.0, 100.0, 200.0, 300.0, 150.0, 50.0])
SUMW = np.stack([3.0 * NSIG, np.full(6, 200.0), np.full(6, 40.0)], axis=-1)
SUMW2 = np.stack([9.0 * NSIG, np.full(6, 100.0), np.zeros(6)], axis=-1)
NOBS = np.array([400.0, 560.0, 830.0, 1150.0, 700.0, 380.0])
SCALE = np.array([1.1, 0.95, 1.0])


def _bbstat(stat_type, mode, sumw=SUMW, sumw2=SUMW2, min_kstat=0.0):
    indata = SimpleNamespace(
        sumw=tf.constant(sumw),
        sumw2=tf.constant(sumw2),
        nbins=sumw.shape[0],
        nbinsmasked=0,
        nbinsfull=sumw.shape[0],
        dtype=tf.float64,
        norm=None,
        betavar=None,
    )
    options = SimpleNamespace(
        noBinByBinStat=False,
        binByBinStatMode=mode,
        binByBinStatType=stat_type,
        minBBKstat=min_kstat,
    )
    return BinByBinStat(
        indata, options, chisqFit=False, covarianceFit=False, data_cov_inv=None
    )


# Masked by sumw2 == 0 (full: the flat process; lite: bins 1 and 4 without
# any MC stat uncertainty) or by --minBBKstat (full: sig where its kstat is
# 50; lite: the bins whose merged kstat is below 300). normal-multiplicative
# toys draw with the width sqrt(varbeta), 0 where sumw2 == 0, so only entries
# masked by --minBBKstat moved there.
MASKINGS = {
    ("full", "sumw2"): dict(),
    ("full", "minBBKstat"): dict(min_kstat=60.0),
    ("lite", "sumw2"): dict(sumw2_zero_bins=[1, 4]),
    ("lite", "minBBKstat"): dict(min_kstat=300.0),
}


@pytest.mark.parametrize("masking", ["sumw2", "minBBKstat"])
@pytest.mark.parametrize("mode", ["full", "lite"])
@pytest.mark.parametrize("stat_type", ["gamma", "normal-multiplicative"])
def test_masked_entries_keep_their_defaults(stat_type, mode, masking):
    spec = dict(MASKINGS[(mode, masking)])
    sumw2 = SUMW2.copy()
    sumw2[spec.pop("sumw2_zero_bins", [])] = 0.0
    bb = _bbstat(stat_type, mode, SUMW, sumw2, **spec)
    mask = bb.betamask.numpy()
    assert mask.any() and not mask.all()

    tf.random.set_seed(1)
    bb.randomize_frequentist()
    beta0 = bb.beta0.numpy()
    np.testing.assert_array_equal(beta0[mask], 1.0)
    np.testing.assert_array_equal(bb.logbeta0.numpy()[mask], 0.0)
    assert (beta0[~mask] != 1.0).any()

    bb.beta0_default_assign()
    bb.beta_default_assign()
    bb.randomize_bayes()
    beta = bb.beta.numpy()
    np.testing.assert_array_equal(beta[mask], 1.0)
    assert (beta[~mask] != 1.0).any()


def test_normal_additive_unchanged():
    """sumw == 0 with sumw2 > 0 is masked, but shifts its bin in
    normal-additive: its centre is still drawn."""
    sumw = SUMW.copy()
    sumw[2, 0] = 0.0
    bb = _bbstat("normal-additive", "full", sumw, SUMW2)
    assert bb.betamask.numpy()[2, 0]
    tf.random.set_seed(1)
    bb.randomize_frequentist()
    active = bb.varbeta.numpy() > 0.0
    assert (bb.beta0.numpy()[active] != 0.0).all()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gamma_full_toy_profile_is_the_minimum(seed):
    """After a frequentist toy the profiled betas are stationary for the
    likelihood the fit evaluates, in which the masked process keeps its yield."""
    bb = _bbstat("gamma", "full")
    mask = bb.betamask.numpy()
    tf.random.set_seed(seed)
    bb.randomize_frequentist()
    beta0 = bb.beta0.numpy()

    norm = SUMW * SCALE
    nobs = tf.constant(NOBS)
    nexp, _, beta = bb.profile_and_apply(
        tf.constant(norm.sum(axis=-1)),
        tf.constant(norm),
        nobs,
        None,
        tf.math.log(nobs),
    )
    nexp, beta = nexp.numpy(), beta.numpy()
    np.testing.assert_allclose(nexp, np.sum(np.where(mask, 1.0, beta) * norm, -1))
    kstat = bb.kstat.numpy()
    dbeta = norm * (1.0 - NOBS / nexp)[:, None] + kstat - kstat * beta0 / beta
    np.testing.assert_allclose(dbeta[~mask] / kstat[~mask], 0.0, atol=1e-9)
