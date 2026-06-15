"""Fast assertion test for the MC-stat de-biasing, using the native template-
fluctuating coverage harness (tests/mcstat_coverage.py). Asserts the ROBUST,
reproducible facts on the near-degenerate slides toy with a NONZERO degenerate-
direction truth (rnorm_true=[1.3,0.7]):

  1. naive (no de-bias) is badly BIASED by the MC-stat attenuation and undercovers;
  2. continuous-M REMOVES the central-value bias;
  3. it INFLATES the uncertainty estimate toward sigma_inf;
  4. and IMPROVES the coverage of the degenerate POI direction.

Note (RESULTS §9n): the raw sandwich sigma still underestimates the de-biased
point's ensemble scatter by ~20-25% (the MC-noise variance in the score is not
captured by the single-toy meat), so exact 68.3% coverage needs the Bartlett/
calibration factor -- this test asserts the de-bias DIRECTION (bias removed, sigma
inflated, coverage improved), not exact 0.683. Run with OUT=<dir>.
"""

import os

os.environ.setdefault("NB", "200")
os.environ.setdefault("HI", "5112")  # near-degenerate (large attenuation)
os.environ.setdefault("R0", "1.3")
os.environ.setdefault("R1", "0.7")

import tests.mcstat_coverage as mc

NTOY = int(os.environ.get("NTOY", "50"))

if __name__ == "__main__":
    print(
        f"coverage harness, slides toy, diff_true={mc.DDIF @ mc.RTRUE:.3f}, ntoy={NTOY}"
    )
    none = mc.run_coverage(ntoy=NTOY, seed=11, debias="none")
    cmS = mc.run_coverage(
        ntoy=NTOY, seed=11, debias="continuousM", debiasCov="sandwich"
    )
    for lab, r in [("none", none), ("continuousM sand", cmS)]:
        print(
            f"  {lab:18s} cov(dif)={r['cov_dif']:.3f} med_bias={r['bias_dif']:+.4f} "
            f"med_sig={r['med_sig']:.4f} rms_res={r['rms_res']:.4f}"
        )

    # 1. naive is badly biased by attenuation (degenerate direction pulled to 0)
    assert (
        none["bias_dif"] < -0.10
    ), f"naive should be attenuated, got {none['bias_dif']}"
    # 2. de-bias removes the central-value bias
    assert abs(cmS["bias_dif"]) < 0.4 * abs(
        none["bias_dif"]
    ), f"de-bias should remove the bias: {none['bias_dif']} -> {cmS['bias_dif']}"
    # 3. de-bias inflates the uncertainty estimate toward sigma_inf
    assert (
        cmS["med_sig"] > 1.3 * none["med_sig"]
    ), f"de-bias should inflate sigma: {none['med_sig']} -> {cmS['med_sig']}"
    # 4. de-bias improves coverage of the degenerate POI
    assert (
        cmS["cov_dif"] > none["cov_dif"] + 0.10
    ), f"de-bias should improve coverage: {none['cov_dif']} -> {cmS['cov_dif']}"
    print(
        "\n  coverage de-bias checks passed (bias removed, sigma inflated, "
        "coverage improved)."
    )
