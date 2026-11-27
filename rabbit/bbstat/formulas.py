"""Closed-form helpers for the bin-by-bin (BBB) statistical treatment.

Pure functions only — no TF Variables, no Fitter state. Used by
:mod:`rabbit.bbstat.bbstat`.
"""

import tensorflow as tf


def solve_quad_eq(a, b, c):
    """Positive root of ``a x² + b x + c = 0`` (TF-broadcast)."""
    return 0.5 * (-b + tf.sqrt(b**2 - 4.0 * a * c)) / a


def gamma_full_beta(kstat, beta0, den):
    """Profiled gamma + full β = kstat·β0 / den, exactly 0 where β0 == 0.

    Also at den == 0: a process with β0 == 0 attaining min(kstat/norm) has
    den == 0 on the profile threshold, where kstat·β0 / den would be 0/0.
    """
    return kstat * beta0 / tf.where(beta0 == 0.0, tf.ones_like(den), den)


def gamma_full_threshold_profile(
    kstat, beta0, norm, betamask, zero_norm, ratio, threshold
):
    """Gamma + full profile of the bins whose minimum is on the threshold.

    Per (bin, process) the profiled β solves kstat·β0 / β = kstat + norm·t at
    the bin's total yield x, with t = 1 - nobs/x for the Poisson and
    (x - nobs)/varnobs for the chi2 data term. The Newton solve searches x
    above the threshold, where t = -min(kstat/norm) and the smallest
    denominator vanishes; a process attaining the minimum with β0 > 0 has
    β → ∞ there, so the minimum is above it. With β0 == 0 (which a frequentist
    toy draws, Poisson(kstat·β)/kstat) its β is 0 above the threshold
    instead. If every process attaining the minimum has β0 == 0 and the
    others' yield at the threshold, ν_rest, is below it, the minimum is on
    the threshold (Barlow and Beeston's case of a source without simulated
    events in a bin; the β0 → 0 limit): x = threshold, the other betas at
    t = -min(kstat/norm), where den = kstat - norm·min(kstat/norm), and for the
    processes attaining the minimum, whose stationarity condition (den = 0,
    β0 = 0) leaves them free, β = (threshold - ν_rest) / Σ norm (any split
    gives the same likelihood). Otherwise the minimum is above the threshold
    and those processes keep β = 0.

    ratio: kstat/norm, +inf where zero_norm (masked or zero template).
    threshold: the total yield x at which t = -min(ratio).

    Returns ``(on_threshold, beta, newton_beta0)``: the [nbins] mask of those
    bins, their betas [nbins, nproc] (only meaningful where on_threshold),
    and β0 with the zero centres of the processes attaining the minimum set
    to 1 in those bins, for the Newton solve whose result is discarded there:
    with only zero centres it would predict no events, and log(0) in the
    Poisson term would make its convergence criterion, a maximum over all
    bins, NaN.
    """
    r_min = tf.reduce_min(ratio, axis=-1)
    has_min = ~tf.math.is_inf(r_min)
    r_min = tf.where(has_min, r_min, tf.zeros_like(r_min))
    at_min = ~zero_norm & (ratio == r_min[..., None])
    zero_centre = beta0 == 0.0

    # betas at the threshold of the processes not attaining the minimum
    free = ~betamask & ~at_min & ~zero_centre
    den = kstat - norm * r_min[..., None]
    # not attaining the minimum in floating point but (nearly) tied with it:
    # its β diverges at the threshold, so the minimum is above it
    diverges = free & (den <= 0.0)
    den = tf.where(free & ~diverges, den, tf.ones_like(den))
    beta = tf.where(free, kstat * beta0 / den, beta0)
    beta = tf.where(at_min, tf.zeros_like(beta), beta)
    nu_rest = tf.reduce_sum(beta * norm, axis=-1)

    on_threshold = (
        has_min
        & tf.reduce_all(~at_min | zero_centre, axis=-1)
        & ~tf.reduce_any(diverges, axis=-1)
        & (nu_rest < threshold)
    )

    norm_min = tf.reduce_sum(tf.where(at_min, norm, tf.zeros_like(norm)), axis=-1)
    beta_min = tf.where(
        on_threshold, threshold - nu_rest, tf.zeros_like(nu_rest)
    ) / tf.where(on_threshold, norm_min, tf.ones_like(norm_min))
    beta = tf.where(at_min, beta_min[..., None], beta)

    newton_beta0 = tf.where(
        on_threshold[..., None] & at_min, tf.ones_like(beta0), beta0
    )
    return on_threshold, beta, newton_beta0
