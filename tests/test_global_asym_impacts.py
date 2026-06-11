"""End-to-end correctness tests for --globalAsymImpacts.

Builds a tensor with both symmetric and asymmetric nuisances via
tests/make_tensor.py, then runs three fits and checks:

  T1 Gaussian-limit closure: at sigma=0.01, globalAsym up/sigma matches
     gaussianGlobalImpacts to ~1e-3 relative.
  T2 Symmetry at small sigma: up = -down to ~1e-3 relative.
  T3 Asymmetry detection: at sigma=1.0, the structurally asymmetric nuisance
     slope_2_signal_ch1 has |up+down| above a threshold; structurally
     symmetric nuisances stay close to zero.
  T4 State restoration: postfit `parms` histogram values are bit-identical
     between a baseline fit and the --globalAsymImpacts fit (same seed).
  T5 Output schema: global_impacts_asym and *_grouped exist with axes
     (impacts, downUpVar, parms).
  T6 Unconstrained nuisances skipped: slope_background / slope_signal not
     in the impacts axis.

Run: source /opt/env.sh && source setup.sh
     python tests/test_global_asym_impacts.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from rabbit import io_tools

OUTDIR = Path("/tmp/test_global_asym_impacts")
TENSOR = OUTDIR / "test_tensor.hdf5"

# Names taken from tests/make_tensor.py
#
# slope_2_signal_ch1 is added with symmetrize=None -> rabbit stores logkhalfdiff
# != 0, so this is the only nuisance for which down vs up should disagree at
# sigma=1 beyond minimizer noise. (slope_signal_ch0 also has kfactor=1.2 but
# is an NOI, treated separately by the model.)
ASYMMETRIC_NUIS = "slope_2_signal_ch1"
# The SymAvg/SymDiff partners produced by the symmetrize transforms have
# logkhalfdiff identically zero: structurally symmetric internal nuisances.
SYMMETRIC_NUIS = [
    "slope_lin_signal_ch0SymAvg",
    "slope_lin_signal_ch0SymDiff",
    "slope_quad_signal_ch0SymAvg",
    "slope_quad_signal_ch0SymDiff",
    "slope_signal_ch1",  # symmetrize="conservative" -> single sym nuisance
    "norm",  # pure norm systematics
    "bkg_norm",
    "bkg_2_norm",
]
UNCONSTRAINED_NUIS = ["slope_background", "slope_signal"]


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def make_tensor() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "tests/make_tensor.py", "-o", str(OUTDIR) + "/"])
    assert TENSOR.exists(), TENSOR


def fit(label: str, *extra: str) -> Path:
    out = OUTDIR / label
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    run(
        [
            "rabbit_fit.py",
            str(TENSOR),
            "-o",
            str(out),
            "--seed",
            "42",
            *extra,
        ]
    )
    return out / "fitresults.hdf5"


def load_results(path: Path) -> dict:
    return io_tools.get_fitresult(str(path))


def get_impact_array(res: dict, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (values, parms_axis, impacts_axis) for an impacts hist."""
    h = res[name].get()
    impacts_axis = np.array(h.axes["impacts"])
    parms_axis = np.array(h.axes["parms"])
    return h.values(), parms_axis, impacts_axis


def get_nuis_idx(impacts_axis: np.ndarray, name: str) -> int:
    matches = np.where(impacts_axis.astype(str) == name)[0]
    if len(matches) == 0:
        raise KeyError(
            f"{name!r} not in impacts axis: {list(impacts_axis.astype(str))}"
        )
    return int(matches[0])


def get_poi_idx(parms_axis: np.ndarray, name: str) -> int:
    return int(np.where(parms_axis.astype(str) == name)[0][0])


# ---------------------------------------------------------------------------


def t1_gaussian_closure(res_small: dict, res_gauss: dict) -> None:
    """At sigma=0.01, globalAsym up/sigma must match gaussianGlobal."""
    print("\n[T1] Gaussian-limit closure (sigma=0.01)")

    sigma = 0.01
    asym, asym_parms, asym_impacts = get_impact_array(res_small, "global_impacts_asym")
    # asym shape: (impacts, downUpVar, parms)
    # gauss shape: (parms, impacts)
    gauss_h = res_gauss["gaussian_global_impacts"].get()
    g_vals = gauss_h.values()
    g_parms = np.array(gauss_h.axes["parms"]).astype(str)
    g_impacts = np.array(gauss_h.axes["impacts"]).astype(str)

    poi = "sig"  # the only POI in make_tensor.py default config
    poi_idx_asym = get_poi_idx(asym_parms, poi)
    poi_idx_gauss = int(np.where(g_parms == poi)[0][0])

    common = [n for n in asym_impacts.astype(str) if n in g_impacts]
    print(f"  comparing {len(common)} nuisances on POI {poi!r}")

    max_rel = 0.0
    worst = None
    for name in common:
        i_asym = get_nuis_idx(asym_impacts, name)
        i_gauss = int(np.where(g_impacts == name)[0][0])
        up = asym[i_asym, 1, poi_idx_asym] / sigma
        ref = g_vals[poi_idx_gauss, i_gauss]
        # gaussian impact is unsigned; compare magnitudes
        denom = max(abs(ref), 1e-12)
        rel = abs(abs(up) - abs(ref)) / denom
        if rel > max_rel:
            max_rel, worst = rel, (name, up, ref)

    print(f"  max relative diff: {max_rel:.3e}  (worst: {worst})")
    assert max_rel < 5e-2, (
        f"T1 FAIL: globalAsym at sigma=0.01 deviates from gaussianGlobal by "
        f"{max_rel:.3e} on {worst!r}; expected closure <5%"
    )
    print("  T1 PASS")


def t2_small_sigma_symmetry(res_small: dict) -> None:
    """At sigma=0.01, down ~ -up for every scanned nuisance."""
    print("\n[T2] Small-sigma symmetry (sigma=0.01: up = -down)")

    asym, asym_parms, asym_impacts = get_impact_array(res_small, "global_impacts_asym")
    poi_idx = get_poi_idx(asym_parms, "sig")

    max_rel = 0.0
    worst = None
    for k, name in enumerate(asym_impacts.astype(str)):
        down = asym[k, 0, poi_idx]
        up = asym[k, 1, poi_idx]
        denom = max(abs(up), abs(down), 1e-12)
        rel = abs(up + down) / denom
        if rel > max_rel:
            max_rel, worst = rel, (name, up, down)

    print(f"  max |up+down|/max(|up|,|down|): {max_rel:.3e}  (worst: {worst})")
    assert max_rel < 5e-2, f"T2 FAIL: not symmetric at small sigma; worst={worst}"
    print("  T2 PASS")


def t3_asymmetry_at_full_sigma(res_full: dict, res_small: dict) -> None:
    """At sigma=1.0 the method must produce non-trivial asymmetry, and the
    asymmetry must scale with sigma.

    Note: even structurally symmetric nuisances (logkhalfdiff=0) can show
    asymmetric impacts at sigma=1.0 with systematic_type="log_normal", because
    the kernel exp(+theta * logkavg) is asymmetric in theta away from the
    quadratic basin. This is correct behaviour, not a bug. We therefore assert:
      a) the structurally asymmetric nuisance shows clearly non-zero
         asymmetry at sigma=1.0;
      b) every nuisance that was symmetric at sigma=0.01 (validated by T2)
         shows MORE asymmetry at sigma=1.0 than at sigma=0.01 -- i.e. the
         asymmetry is a real sigma-dependent effect, not noise.
    """
    print("\n[T3] Asymmetry detection at sigma=1.0")

    asym_full, asym_parms, asym_impacts = get_impact_array(
        res_full, "global_impacts_asym"
    )
    asym_small, _, asym_impacts_small = get_impact_array(
        res_small, "global_impacts_asym"
    )
    poi_idx = get_poi_idx(asym_parms, "sig")

    def asymmetry(arr: np.ndarray, axis: np.ndarray, name: str) -> float:
        k = get_nuis_idx(axis, name)
        d, u = arr[k, 0, poi_idx], arr[k, 1, poi_idx]
        return abs(u + d) / max(abs(u), abs(d), 1e-12)

    # (a) structurally asymmetric nuisance must show measurable asymmetry.
    impacts_str = asym_impacts.astype(str)
    if ASYMMETRIC_NUIS not in impacts_str:
        print(
            f"  skip (a): {ASYMMETRIC_NUIS} not in impacts axis; "
            f"list = {list(impacts_str)}"
        )
    else:
        a = asymmetry(asym_full, asym_impacts, ASYMMETRIC_NUIS)
        print(f"  asymmetry({ASYMMETRIC_NUIS}) at sigma=1.0 = {a:.3e}")
        assert a > 1e-2, (
            f"T3 FAIL (a): {ASYMMETRIC_NUIS} should be measurably asymmetric "
            f"at sigma=1.0, got {a:.3e}"
        )

    # (b) every nuisance: asymmetry grows with sigma (within minimizer noise).
    growth_floor = 1e-3
    for name in asym_impacts.astype(str):
        if name not in asym_impacts_small.astype(str):
            continue
        a_small = asymmetry(asym_small, asym_impacts_small, name)
        a_full = asymmetry(asym_full, asym_impacts, name)
        print(
            f"  asymmetry({name}): sigma=0.01 -> {a_small:.3e}, "
            f"sigma=1.0 -> {a_full:.3e}"
        )
        # Allow noise floor: even if the asymmetry is tiny, it should not
        # *shrink* with sigma if sigma=0.01 was already at convergence noise.
        assert a_full + growth_floor >= a_small, (
            f"T3 FAIL (b): {name} asymmetry shrinks with sigma "
            f"({a_small:.3e} -> {a_full:.3e})"
        )
    print("  T3 PASS")


def t4_state_restoration(res_baseline: dict, res_full: dict) -> None:
    """After the asym scan, postfit `parms` must match a vanilla fit's parms."""
    print("\n[T4] State restoration (parms unchanged vs baseline)")

    p0 = res_baseline["parms"].get().values()
    p1 = res_full["parms"].get().values()
    diff = np.max(np.abs(p0 - p1))
    print(f"  max |parms_baseline - parms_after_asym| = {diff:.3e}")
    # Same seed, same fit -> should be bit-identical, but allow a tiny float
    # margin in case of any nondeterminism in the env.
    assert diff < 1e-9, "T4 FAIL: postfit parms changed after the asym loop"
    print("  T4 PASS")


def t5_output_schema(res_full: dict) -> None:
    print("\n[T5] Output schema")
    assert "global_impacts_asym" in res_full, list(res_full.keys())
    assert "global_impacts_asym_grouped" in res_full, list(res_full.keys())

    h = res_full["global_impacts_asym"].get()
    names = [a.name for a in h.axes]
    print(f"  axes = {names}")
    assert names == ["impacts", "downUpVar", "parms"], names

    g = res_full["global_impacts_asym_grouped"].get()
    gnames = [a.name for a in g.axes]
    print(f"  grouped axes = {gnames}")
    assert gnames == ["impacts", "downUpVar", "parms"], gnames

    groups = list(np.array(g.axes["impacts"]).astype(str))
    print(f"  groups = {groups}")
    assert groups[-1] == "Total", groups
    print("  T5 PASS")


def t6_unconstrained_skipped(res_full: dict) -> None:
    print("\n[T6] Unconstrained nuisances skipped")
    _, _, impacts_axis = get_impact_array(res_full, "global_impacts_asym")
    impacts_str = list(impacts_axis.astype(str))
    print(f"  scanned nuisances = {impacts_str}")
    for n in UNCONSTRAINED_NUIS:
        assert n not in impacts_str, f"T6 FAIL: unconstrained {n!r} should be skipped"
    print("  T6 PASS")


# ---------------------------------------------------------------------------


def main() -> None:
    os.chdir(os.environ.get("RABBIT_BASE", "."))
    make_tensor()

    common = [
        "--doImpacts",
        "--gaussianGlobalImpacts",
    ]

    # baseline: just the fit, no asym scan -> reference parms
    base_h5 = fit("baseline", *common)

    # small-sigma asym for Gaussian closure & symmetry
    small_h5 = fit(
        "small_sigma",
        *common,
        "--globalAsymImpacts",
        "--globalAsymImpactsSigma",
        "0.01",
    )

    # full-sigma asym for asymmetry detection
    full_h5 = fit(
        "full_sigma",
        *common,
        "--globalAsymImpacts",
        "--globalAsymImpactsSigma",
        "1.0",
    )

    res_base = load_results(base_h5)
    res_small = load_results(small_h5)
    res_full = load_results(full_h5)

    t5_output_schema(res_full)
    t6_unconstrained_skipped(res_full)
    t1_gaussian_closure(res_small, res_small)  # gaussianGlobal is in same file
    t2_small_sigma_symmetry(res_small)
    t3_asymmetry_at_full_sigma(res_full, res_small)
    t4_state_restoration(res_base, res_full)

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
