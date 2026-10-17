# Reproduction of the limited-MC statistical issues (toy from the slides)

> **About this file.** This is the research log behind the MC-stat de-biasing in
> rabbit (`--mcStatDebias`, `--covMode`); the rabbit code cites its sections for
> the numbers and derivations it relies on. §1–§8 are a standalone numpy/jax study:
> the `demo*.py` / `mcstat_toy.py` scripts they mention are not part of rabbit.
> §9 follows the rabbit implementation in chronological order, so earlier
> subsections may describe behaviour that a later one supersedes (e.g. the
> `--mcStatKfold` option, removed in §9q). For the current interfaces see
> `RABBIT_MCSTAT_DESIGN.md` in this directory; the rabbit-level checks quoted in §9
> live in `tests/test_mcstat_debias.py`.

Self-contained reproduction of `slides/statissues-Aug17-2023` using jax + numpy
+ scipy, with the `scipy` **trust-krylov** minimiser wherever an explicit
minimisation is needed.  Solutions from `MC_STAT_METHODS.md` are tested and
shown to restore a correct central value, uncertainty estimate, and coverage.

## Files
- `mcstat_toy.py`   — toy templates, infinite-MC Fisher, all estimators, the
  JAX profiled full-mc-stat objective + trust-krylov fitters.
- `demo1_issue.py`  — the basic issue + the F-M (continuous-M) uncertainty fix.
- `demo2_coverage.py`— coverage of the 1σ interval: stat-only / BB-lite /
  full-mc-stat (Hessian and PLR).
- `demo3_fc.py`     — Feldman-Cousins coverage: asymptotic vs profile vs cheat.
- `results/`        — plots (`err_stat.png`, `err_stat2.png`, `templates.png`)
  and `fc_results.npz`.

## The toy (slide "Simple but problematic case")
200 bins, 2 free-floating processes, ~2e6 expected events.
- process 0: flat, height 4962 in all bins.
- process 1: step, 5112 (bins 0–99) then 4962 (bins 100–199).

The two templates are **near-degenerate** (correlation −0.9999); only the
first-half step breaks the degeneracy.  Model `μ_i(Δμ)=Σ_j(1+Δμ_j) n_{ij}`,
two POIs `Δμ=(Δμ_0,Δμ_1)`.  "1:1 MC": observed templates `M_{ij}~Poisson(n_{ij})`,
data `D_i~Poisson(d_i)`.

Infinite-MC (Asimov) Fisher `F_{jl}=Σ_i k_{ij}k_{il}/μ_i` gives
**σ∞(Δμ_1)=0.0943** (slide: 0.0950).

## 1. The basic issue + the F-M uncertainty fix  (`demo1_issue.py`)
20 000 finite-MC toys, naive same-sample Hessian σ vs the F-M debias
`Σ_true=[Σ_i (k k^T − Σ_k)/D_i]^{-1}` (slide "inversion" / MD §3b continuous-M).

| quantity | reproduction | slide |
|---|---|---|
| naive σ (mean / median / RMS) | 0.0566 / 0.0565 / 0.0027 | 0.057 / 0.056 / 0.003 |
| analytic ⟨σ⟩ biased `[F+Σ_k]` | 0.0563 | 0.056 |
| F-M σ (mean / median / RMS) | 0.0976 / 0.0951 / 0.0152 | 0.098 / 0.095 / 0.015 |

→ the naive finite-MC uncertainty is **~2× too small**; the F-M correction
**recovers σ∞=0.095** (with the long right tail / occasional non-PSD that the
slide and MD §3b warn about).

## 2. Coverage of the 1σ interval  (`demo2_coverage.py`, 8000 toys)
Generated at (0,0) and at +5σ along the worst-constrained eigen-direction.

| config | σ (repro) | cover (repro) | σ (slide) | cover (slide) |
|---|---|---|---|---|
| inf-MC, hessian (sanity)        | 0.0943 | 0.682 | 0.0950 | 0.682 |
| stat-only, hessian @(0,0)       | 0.0565 | 0.519 | 0.0566 | 0.521 |
| stat-only, hessian @+5σ         | 0.0565 | 0.002 | 0.0566 | 0.141 |
| BB-lite, hessian @(0,0)         | 0.0799 | 0.684 | 0.0801 | 0.683 |
| BB-lite, hessian @+5σ           | 0.0799 | 0.003 | 0.0801 | 0.200 |
| full-mcstat, hessian @(0,0)     | 0.1337 | 0.445 | 0.1344 | 0.447 |
| full-mcstat, hessian @+5σ       | 0.1400 | 0.466 | 0.1405 | 0.460 |
| full-mcstat, PLR @(0,0)         | 0.1341 | 0.447 | 0.1351 | 0.445 |
| full-mcstat, PLR @+5σ           | 0.1405 | 0.454 | 0.1412 | 0.458 |

Reproduces the slide's conclusions exactly:
- infinite-MC: perfect coverage, σ=0.095.
- stat-only / BB-lite: **biased toward Δμ=0** (bias(μ1)=−0.30 at the displaced
  point — classic errors-in-variables attenuation), so coverage is strongly
  **parameter-dependent** (collapses away from 0).
- full-mc-stat: **unbiased** central value (bias≈0), coverage **independent of
  the parameter value**, but the asymptotic/PLR interval **still undercovers**
  (~0.45). Hessian ≈ PLR → likelihood is ~parabolic but Wilks' theorem fails.

## 3. Feldman-Cousins: the coverage fix  (`demo3_fc.py`)
The true value μ1_gen is covered iff the observed PLR test statistic
`t_obs ≤ q_c(μ1_gen)`, where `q_c` is the 68.3% quantile of the sampling
distribution of `t` under μ1=μ1_gen.  How that distribution is built:
- asymptotic: assume t~χ²₁ → q_c=1.
- FC profile: build it from each toy's **profiled** nuisances (≈ the finite MC
  sample).
- FC cheat: build it from the **true** nuisances (λ = truth).

| method | q_c | coverage (repro) | coverage (slide) | σ_eff = σ_hess·√q_c |
|---|---|---|---|---|
| asymptotic PLR | 1.00 | 0.457 | 0.445 | 0.132 |
| FC profile     | 1.64 | 0.553 ± 0.020 | 0.537 | 0.169 (slide 0.179) |
| FC cheat       | 2.88 | **0.697 ± 0.019** | 0.669 | **0.224** (slide 0.232) |

The test statistic's tail is the smoking gun: under the **true** nuisances,
`P(t>1) = 0.551` vs `χ²₁`'s 0.317, and the 68.3% quantile is `q_c=2.88` instead
of 1.0 (`results/teststat.png`). Wilks' theorem fails; the profiled-nuisance
distribution (q_c≈1.6) still misses much of that tail, so FC-profile
undercovers, while FC-cheat captures it and covers.

→ The profiled-nuisance sampling distribution **underestimates the tail** of `t`
(the slide's key plot); using the **true** nuisances captures it and **restores
~68% coverage**, at the cost of a further-enlarged interval (σ_eff≈0.22–0.23).

## Summary of the uncertainties (slide "Summary")
| configuration | proper coverage? | expected σ(Δμ_1) |
|---|---|---|
| infinite MC | yes | 0.095 |
| 1:1 MC, stat-only | no (param-dep.) | 0.057 |
| 1:1 MC, Barlow-Beeston lite | no (param-dep.) | 0.080 |
| 1:1 MC, full mc-stat, asymptotic/PLR | no (undercovers) | 0.134 |
| 1:1 MC, full mc-stat, FC cheat | **yes** | ~0.224 |

The true 1:1-MC uncertainty (≈0.22) is **>2×** the infinite-MC σ∞ (0.095) and
~3× the BB-lite estimate — the extra inflation comes from the template
**gradient** noise, not only the bin counts, exactly as the slide states.

## 4. The MD composite scalar objective  (`demo4_composite.py`)
`ℓ = BB-lite-NLL(θ) − ½(θ−sgθ)ᵀ M (θ−sgθ)` built as a JAX autograd scalar with
`stop_gradient`, minimised with trust-krylov.  (Autograd central value + Hessian
verified identical to closed-form BB-lite estimate and `H_bblite − M`.)

| gen | median σ(Δμ₁) | coverage | bias(μ₁) | non-PSD |
|---|---|---|---|---|
| (0,0) | 0.1343 | 0.915 | −0.001 | 0/8000 |
| +5σ (μ₁=0.471) | 0.1342 | 0.032 | −0.301 | 0/8000 |

→ Restores the **uncertainty estimate** σ₂≈0.134 (= full-mc-stat target,
param-independent, PSD-stable, no long tail — better-behaved than raw F−M) but
does **not** restore **coverage**: the stop-gradient makes M curvature-only, so
the point estimate is the BB-lite one, still attenuation-biased toward 0 (bias
−0.30 at +5σ).  Coverage is therefore strongly parameter-dependent (over-covers
near 0, under-covers away).  Correct coverage still needs FC-cheat (§3).

## 4b. Composite with NO stop-gradient: M in the gradient too (`demo4b_nosg.py`)
M-term set to the bare quadratic `−½ θᵀ M θ` (contributes to gradient AND
Hessian).  Stationary point `θ̂=(H_bb−M)⁻¹ g`, Hessian `H_bb−M`.  (Autograd
trust-krylov = closed form.)

| gen | median σ₁ | coverage | bias(μ₁) | RMS(μ̂₁) | failed (non-PSD) |
|---|---|---|---|---|---|
| (0,0) | 0.1346 | 0.445 | −0.002 | 0.255 | 1/8000 |
| +5σ   | 0.1347 | 0.413 | +0.032 | 0.275 | 0/8000 |

→ Letting M into the gradient **de-attenuates the central value** (bias →0 at
both points) and makes coverage **parameter-independent** — the estimator now
behaves identically to the full-mc-stat (per-process λ) treatment.  But it
**still undercovers (~0.45)**: the quoted curvature σ=0.134 is ~2× smaller than
the true fat-tailed spread RMS(μ̂₁)=0.255 (near-singular `H_bb−M` → heavy tails,
occasional unbounded-below failure).  Contrast with the stop-gradient version
(§4), which keeps BB-lite's biased point estimate → param-dependent coverage.
Neither asymptotic form reaches 0.683; only FC-cheat (§3) does.

## 5. Restoring 68.3% coverage efficiently (`demo5_sandwich.py`, `demo6_bootstrap.py`, `demo7_coxreid.py`)
Residual undercoverage is the incidental-parameters (Neyman-Scott) effect: ~1-2
nuisances/bin -> profile curvature is an inconsistent variance; missing piece is
the derivative-template (gradient) noise amplified through `(H_bb-M)^{-1}`.

med(σ)/RMS(σ) = median and spread (std) of the per-toy *estimated* uncertainty
across toys, at gen=(0,0); RMS is the slide's convention.

| mechanism | cost | med(σ) | RMS(σ) | cover @(0,0)/@+5σ | verdict |
|---|---|---|---|---|---|
| profile curvature (baseline) | O(1) | 0.135 | 0.022 | 0.44 / 0.41 | undercovers |
| **robust/sandwich covariance** | O(1), no toys | 0.297 | 0.122 | **0.82 / 0.83** | valid (conservative), param-indep — recommended cheap fix |
| Cox-Reid modified PL | O(1) | 0.079 | 0.005 | 0.67 / 0.004 | fails (re-biases central value; shrinks σ) |
| template bootstrap (quantile) | O(B) closed-form | 0.105 | 0.012 | 0.35 / 0.35 | undercovers (same-sample dispersion) |
| FC-cheat (reference) | O(N·B) | 0.224 | tail-dom. | 0.70 | correct but needs truth |

(RMS(σ) for sandwich and FC-cheat is large/tail-dominated — both inherit the fat
upper tail of near-singular toys; their *median* σ is the robust summary.)

- **Sandwich** = `Var(μ̂)=Σ(∂μ̂/∂D)²D + Σ(∂μ̂/∂M)²M` via one autograd Jacobian.
  Cheapest valid interval; over-covers (SD on a fat-tailed estimator).
- **Exact 0.683 truth-free is the open problem** (Cranmer physics/0310108, Punzi
  physics/0511202; MD §8). Truth-free resampling reproduces the same-sample
  dispersion -> undercovers (verified). Efficient routes to near-exact coverage:
  dimension-reduced nuisance FC (scan only the few POI-coupled λ directions),
  hybrid/prior-predictive FC (marginalize sub-toy nuisances), or a
  Bartlett/bootstrap-calibrated threshold.

## 5b. Can the Cox-Reid attenuation be removed? (`demo9_cr_fix.py`)
Attenuation source: the adjustment g(θ)=½Σ log det j_λλ(θ) is θ-dependent (det ∝
rrᵀ grows with |θ|), so its GRADIENT shifts the minimum.  Fix = curvature-only
via stop-gradient (point estimate from the unbiased full-mc-stat MLE; only the
CR-adjusted Hessian H_profile+H_CR used for the interval).

| method | med σ | cover @(0,0)/@+5σ | bias(μ₁) @(0,0)/@+5σ |
|---|---|---|---|
| full-mc-stat (baseline) | 0.133 | 0.44 / 0.47 | +0.001 / +0.009 |
| Cox-Reid full | 0.079 | 0.68 / 0.003 | −0.002 / −0.302 |
| Cox-Reid curvature-only | 0.081 | 0.28 / 0.31 | −0.005 / +0.008 |

→ Attenuation IS removable (bias→0, param-independent).  But it does not help:
Cox-Reid's correction has the WRONG SIGN here — it *adds* curvature (σ 0.133→0.08,
tighter) whereas the incidental-parameters failure needs σ *inflated*.  So the
de-attenuated CR is unbiased but undercovers worse (0.28).  It also shows full-CR's
0.68 at gen=(0,0) was an artifact (bias-toward-0 compensating a too-small σ when
truth=0; removing the bias collapses it to 0.28).  Cox-Reid corrects the nuisance-
profile bias, not the variance underestimate — wrong tool for this toy.

## 5c. Can the Bartlett/calibrated threshold be analytic (toy-free)?
Bartlett factor c = E[t] = tr(I_curv·V_true) = V_true/V_curv (1 dof). Both pieces
are analytic, BUT at the Asimov/expansion point the full-mc-stat MLE is efficient
(verified via implicit-diff sandwich): V_true = V_curv = 0.133² exactly, so the
LEADING-order analytic factor is c=1 — it predicts coverage 0.449 and is BLIND to
the undercoverage. The effect (E[t]=2.87) is a higher-order finite-MC fluctuation
(the estimated information fluctuates; incidental parameters), invisible to a
single-point/linear analysis.
- The earlier sandwich giving 0.83 used the de-attenuated (H_bb−M)⁻¹ estimator,
  whose near-singular nonlinearity inflates the per-toy linear variance — an
  uncontrolled proxy (over-covers), not a principled analytic Bartlett.
- A genuinely toy-free analytic route needs the next-order cumulant terms
  (Bartlett's E[∂³ℓ], E[(∂²ℓ)²]) — explicit template-moment sums (same family as
  the noise-floor M and its variance), but a heavy expansion, convergence not
  guaranteed in the incidental-parameters regime.
- Cheapest route that works (not toy-free): ONE non-nested parametric bootstrap to
  estimate the scalar E[t]; threshold=E[t] gives ~0.683 because E[t]=2.866 ≈
  q_c=2.876 and P(t≤E[t])=0.682 (χ²₁'s mean=68.27%-quantile property survives, so
  matching the first cumulant suffices). O(B) once, vs nested FC.

## 5d. Sandwich applied PER-TOY to the full-mc-stat MLE (`demo10_fullmle_sandwich.py`)
| gen | curvature σ/cover | sandwich σ/cover | ratio |
|---|---|---|---|
| (0,0) | 0.134 / 0.44 | 0.280 / 0.80 | 2.09 |
| +5σ   | 0.140 / 0.47 | 0.282 / 0.80 | 2.01 |

Refines §5c: the per-toy robust/sandwich covariance on the *standard full-mc-stat
MLE* (implicit-diff V_sand=H⁻¹ J H⁻¹, meat J=Var(score) from Poisson D and M) does
NOT collapse to the Hessian — it inflates σ ~2× and restores coverage to ~0.80
(conservative, param-independent), like demo5.
- Asimov single point: J=H exactly → c=1 (blind). Per toy: evaluated on fluctuated
  data at the realized θ̂, the near-degenerate Jacobian ∂θ̂/∂M is large, so
  E[V_sand(toy)] ≫ V_sand(Asimov). That gap IS the finite-MC/incidental-parameters
  effect — invisible at the expansion point, captured per-realization.
- Practical: the sandwich works directly on the full-mc-stat MLE (no need for the
  de-attenuated (H_bb−M)⁻¹ estimator). Cheapest fix: 1 Hessian + few autograd
  Jacobians per fit, no nested toys, no truth. Over-covers (SD on a fat tail).

## 5e. Sandwich evaluated on the BB-lite MLE (`demo11_bblite_sandwich.py`)
| gen | RMS(μ̂₁) | bias | curvature σ/cover | sandwich σ/cover | ratio |
|---|---|---|---|---|---|
| (0,0) | 0.0807 | +0.001 | 0.0798 / 0.683 | 0.102 / 0.80 | 1.28 |
| +5σ   | 0.0795 | −0.301 | 0.0799 / 0.003 | 0.104 / 0.01 | 1.30 |

- BB-lite has NO variance underestimate: true RMS(μ̂₁)=0.081 ≈ curvature 0.080 →
  coverage 0.683 at gen0 with no correction. Its variance inflation regularizes
  the degeneracy (well-conditioned), unlike the full-mc-stat MLE. The sandwich
  therefore only mildly inflates (×1.28) and OVER-corrects (0.80, the plain
  curvature is better-calibrated here).
- The sandwich cannot fix BB-lite's central-value BIAS (−0.30 at +5σ) → coverage
  stays parameter-dependent (0.80 → 0.01).

Unifying lesson: the sandwich fixes a VARIANCE underestimate, not a BIAS. It
rescues (conservatively) the unbiased near-singular estimators (full-mc-stat,
(H_bb−M)⁻¹) whose curvature undercounts the template-fluctuation spread; on the
biased BB-lite MLE it just widens the interval around the wrong center. Correct
coverage needs an unbiased central value first.

## 5f. Mitigating the sandwich over-coverage (`demo12_sandwich_calib.py`)
| gen | raw ±1σ_sand | k·σ_sand (empirical) | Cornish-Fisher (kurtosis) |
|---|---|---|---|
| (0,0) | k=1 → 0.799 | k*=0.780 → 0.683 | g₂≈0.01 → k=1.00 → 0.799 (fails) |
| +5σ   | k=1 → 0.797 | k*=0.774 → 0.683 | g₂≈0.17 → k=0.99 → 0.792 (fails) |

Diagnosis: NOT a fat-tail effect. The standardised residual z=(μ̂−μ)/σ_sand is
~Gaussian (excess kurtosis ≈0) because per-toy σ_sand tracks the excursions; the
linear sandwich just over-estimates the conditional SD by a constant ≈1.28× (the
estimator response saturates, linear propagation does not).
Fix: interval = μ̂ ± k·σ_sand with k≈0.78 → exactly 0.683, and k is
PARAMETER-INDEPENDENT (0.780 vs 0.774) so it is one universal constant. Obtain k
from one cheap non-nested bootstrap (robust here since z is Gaussian — bulk, not
tail), a precomputed Asimov ensemble, or the 2nd-order delta method. Cornish-Fisher
fails (z not leptokurtic — the over-coverage is a scale bias, not a tail shape).

## 5g. Second-order delta method (tested) — does NOT mitigate over-coverage (`demo13_second_order.py`)
2nd-order variance of mu_hat1 via autograd Hessian wrt the 600 data inputs,
Poisson cumulants (kappa2=kappa3=mean):
  Var ~ Σ g_a²x_a + Σ g_a g_aa x_a + ½ Σ g_ab² x_a x_b

| gen | true RMS / 68.3%ile | 1st-order σ/cover | 2nd-order σ/cover |
|---|---|---|---|
| (0,0) | 0.244 / 0.233 | 0.290 / 0.83 | 0.353 / 0.90 |
| +5σ   | 0.311 / 0.264 | 0.320 / 0.82 | 0.391 / 0.91 |

NEGATIVE RESULT: 2nd-order delta makes σ LARGER (0.29→0.35) and coverage WORSE
(0.83→0.90). The over-coverage is a variance-vs-quantile gap (target 68.3%-quantile
0.233 < SD 0.244, μ̂ mildly fat-tailed); the delta method at any order estimates the
SD, and higher orders only add positive variance. Even first-order already
over-estimates the true SD. So no variance-based correction lands on the 68.3%
interval — the analytic route must target the quantile shape (Cornish-Fisher on
μ̂'s kurtosis), but the robust practical fix stays the single calibration constant
k≈0.78 of §5f (per-toy σ_sand absorbs the tail → standardised residual ~Gaussian).

## 5h. Natural sandwich + continuous-M combination (`demo14_crossfit.py`)
Clarification: sandwich and continuous-M are COMPLEMENTARY, not redundant —
continuous-M/λ-profiling fixes the CENTRAL VALUE (de-attenuation); the sandwich
fixes the VARIANCE (which even the M-debiased curvature 0.134 underestimates).

Natural combinations (avoid the clunky "form M → (H_bb−M)⁻¹g → sandwich"):
- Full Barlow-Beeston MLE + sandwich (§5d/demo10): λ-profiling de-attenuates
  automatically (no explicit M); sandwich gives the variance. Lower variance.
- Cross-fit bilinear objective + sandwich (MD §4a): one autograd scalar gives all
  three, M NEVER materialized:

| gen | bias(μ₁) | F_ch curvature (Hessian) | cross-fit+sandwich σ/cover (k→0.683) |
|---|---|---|---|
| (0,0) | +0.001 | 0.0947 ≈ σ∞ (PSD 2996/3000) | 0.354 / 0.86  (k≈0.72) |
| +5σ   | +0.050 | 0.0950 ≈ σ∞              | 0.390 / 0.88  (k≈0.68) |

gradient=cross-fit score→unbiased central value; Hessian=cross-half Fisher→σ∞
curvature (self-term-free); sandwich meat also self-term-free (independent halves).
Trade-off (MD §3a): single random split is noisier (RMS 0.45 vs 0.25 for full-BB)
→ wider sandwich (0.35 vs 0.28). So full-BB+sandwich = lower-variance natural route;
cross-fit+sandwich = most elegant/M-free (and tracks θ-dependence for free if nonlinear).

## 5i. Can the calibration factor k be computed without bootstrapping? (`demo15_kfactor.py`)
k = 68.3%ile|z|, z=(μ̂−μ)/σ_sand (Gaussian). Tested analytic surrogates ALL fail:
| route | result | why |
|---|---|---|
| variance ratio @ Asimov | wrong (c=1) | Asimov σ_sand=0.133 ≪ per-toy 0.28 |
| 2nd-order delta on variance (§5g) | wrong direction | inflates σ |
| Cornish-Fisher on z kurtosis (§5f) | k≈1 (fails) | z is Gaussian (κ≈0) |
| ratio of expectations √(E[res²]/E[σ²]) | 0.10 vs 0.79 | res² & σ_sand² correlated +0.72, tail-dominated |

Root cause: σ_sand TRACKS the excursions (corr(res²,σ²)=+0.72) → z Gaussian and k a
clean param-independent constant, but the ratio expectation does NOT factorize, so
no moment formula works. A closed form would need a full 2nd-order delta on the
RATIO z (derivatives/variance of σ_sand itself = 3rd-derivative tensors of μ̂, at a
regularized point) — analytic in principle but heavy/fragile, not a simple formula.

Practical resolution: k is PARAMETER-INDEPENDENT (0.787 vs 0.779) and z Gaussian, so
k is ONE universal constant — precompute once offline from a tiny Asimov ensemble
(robust, bulk quantity), never a per-measurement bootstrap.

## 5j. Is k computable WITHOUT the truth? (`demo16_bootstrap_k.py`) — YES
Parametric bootstrap off the OBSERVED finite-MC sample (no truth): treat M_obs as
template truth-surrogate, M*~Poisson(M_obs), D*~Poisson(Σ(1+μ̂_obs)M_obs), refit,
z*=(μ̂*−μ̂_obs)/σ_sand*, k_boot=68.3%ile|z*|.

| quantity | value |
|---|---|
| true k (§5i) | ~0.78 |
| k_boot (off finite sample) | median 0.825, mean 0.832, std 0.09 |
| coverage with bootstrap-k | 0.693 ± 0.038  (consistent with 0.683) |
| raw sandwich (k=1) | 0.820 |

→ k IS recoverable truth-free. Contrast with FC: the FC threshold q_c is a TAIL
quantity → same-sample bootstrap reproduces the too-thin tail → undercovers
(needed FC-cheat/truth). k is a BULK scale (SD of Gaussian z) → robust to the
same-sample bias → bootstrap off the finite sample recovers it (k_boot 0.83 vs
true 0.78, slightly conservative → coverage 0.69).

PUNCHLINE: the residual incidental-parameters undercoverage, which the slides
fixed only via FC-with-true-nuisances ('cheating'), is fixable TRUTH-FREE by:
full-mc-stat MLE (unbiased central value) + sandwich (robust variance) + one bulk
calibration factor k from a single parametric bootstrap off the observed finite MC.
In a real analysis: one bootstrap off your one dataset, no infinite-stats access.

## 5k. Sandwich on continuous-M MLE, with vs without BB-lite (`demo17_Mbblite_sandwich.py`)
Estimator θ̂=(H−M)⁻¹g (M in gradient → de-attenuated), data-term V=D vs V=D+ΣM:

| gen | variant | bias | med σ_sand | PSD% | cov(k=1) | k | cov(calib) |
|---|---|---|---|---|---|---|---|
| (0,0) | cont-M (V=D)      | +0.002 | 0.296 | 100 | 0.830 | 0.728 | 0.683 |
| (0,0) | cont-M + BB-lite  | +0.002 | 0.296 | 100 | 0.832 | 0.725 | 0.683 |
| +5σ   | cont-M (V=D)      | +0.030 | 0.324 | 100 | 0.823 | 0.754 | 0.683 |
| +5σ   | cont-M + BB-lite  | +0.030 | 0.324 | 100 | 0.823 | 0.753 | 0.683 |

→ Rows identical to ~3 digits: BB-lite makes NO difference once sandwiched, because
the sandwich already propagates Var(M) (= the σ_MC that BB-lite injects analytically).
Unifying result: the robust sandwich SUBSUMES both the BB-lite σ_MC inflation AND the
continuous-M curvature debias for the VARIANCE. The only thing needed upstream is the
de-attenuated CENTRAL VALUE (continuous-M-in-gradient or full-BB λ-profiling).
Minimal recipe: de-attenuated central value + sandwich + one bulk calibration k.

## 6. Regime scan: adjusting the step size d (`demo18_regimes.py`)
a=4962, process-1 first-half = a+d. Larger d -> less degenerate -> milder.
6 estimators (median σ / coverage @(0,0)/@+5σ):

CLEAN (d=1000, σ∞=0.014, corr -0.996):
  plain MLE        0.0141  0.519/0.515   (undercovers: misses σ_MC)
  BB-lite MLE      0.0200  0.685/0.680   <- sufficient (param-independent)
  M-corr MLE(V=D)  0.0144  0.509/0.507   (undercovers: V=D misses σ_MC)
  M-corr BB-lite   0.0204  0.676/0.672
  +sandwich        0.0213  0.695/0.691   (slightly over)
  => BB-lite necessary & sufficient; M-correction unnecessary; only σ_MC matters.

INTERMEDIATE (d=500, σ∞=0.029, corr -0.999):
  plain MLE        0.0264  0.518/0.456
  BB-lite MLE      0.0373  0.683/0.609   (param-DEPENDENT: bias -0.021 -> undercovers @+5σ)
  M-corr BB-lite   0.0404  0.643/0.642   (param-INDEPENDENT: bias removed, but 0.64<0.68)
  +sandwich        0.0467  0.719/0.714   (over; calibrate to 0.68)

SEVERE (d=150, original): bblite 0.69/0.003, M-corr-bblite 0.44/0.41, sandwich 0.83.

KEY FINDING: the hoped-for middle tier ("M+BB-lite enough without sandwich") is
SQUEEZED OUT. In this toy the central-value bias and the variance-underestimate
(sandwich-need) are both ∝ noise-floor/step² and turn on together; moreover the
M-correction's de-attenuation is what CREATES the near-singular fat-tailed
estimator the sandwich fixes. So M and sandwich are a PACKAGE: wherever M is
needed (kill bias), the sandwich is needed too (fix the exposed variance). By the
d where M+BB-lite would suffice (~1000), BB-lite already suffices alone.
=> Two clean tiers, not three: (clean) BB-lite sufficient; (finite-MC-matters)
   de-attenuation + sandwich as a unit.

## 5m. Calibration from a single bootstrap or two sample halves? (`demo19_calib_cheap.py`)
k=SD(z) is a SCALE; a single bootstrap/split is a 1-dof estimate (noisy/skewed).
Per-analysis coverage at the original toy (d=150, gen 0):

| method | cost | coverage | note |
|---|---|---|---|
| fixed universal k=0.78 | once | 0.683 | target |
| full bootstrap (B=60) | B fits | 0.713 | robust |
| single bootstrap (B=1) | 1 fit | 0.600 | UNDERcovers: |z*| half-normal, mode 0 -> k too small too often |
| TWO halves (1 split) | 2 half-fits | 0.665 | ~works; cheapest |
| pooled single-boot (mean k) | once | 0.723 | universal -> precompute OK |

TWO-halves construction: half-MC de-attenuated fits (weight-2 events -> sumw2=4*count
in the noise floor); shared data cancels in (μ̂_A-μ̂_B), unbiasedly isolating
Var_MC=(μ̂_A-μ̂_B)²/4; k=√(σ∞²+Var_MC)/σ_sand, σ∞ from F-M curvature. The σ∞ FLOOR
prevents k collapsing to 0 (the single-bootstrap failure mode) -> works per-analysis.
Still 1-dof (fat-tailed k, occasional non-PSD half-fit). For robustness: k-fold
(several splits averaged, interpolates to the small-bootstrap result), or pool the
MEDIAN k (mean is tail-dominated), or precompute the universal constant once.

## 5m (cont.) Reported-σ stability: median/RMS of k·σ_sand (orig toy, gen 0)
| method | med σ | RMS σ | coverage |
|---|---|---|---|
| σ_sand (k=1) | 0.295 | 0.103 | 0.813 |
| fixed k=0.78 | 0.230 | 0.080 | 0.687 |
| full bootstrap (B=60) | 0.247 | 0.091 | 0.720 |
| single bootstrap (B=1) | 0.216 | 0.223 | 0.608 |
| TWO halves (1 split) | 0.216 | 6.29 | 0.666 |

Two-halves reported σ: sensible MEDIAN (0.216, ~target) but catastrophic RMS (6.3,
~30× median) — 1-dof instability: a half-fit occasionally hits near-singular (H-M)
→ (μ̂_A-μ̂_B) explodes → that analysis's σ is enormous. Coverage (ensemble property)
is ~right (0.666, tail toys self-compensate), but per-analysis σ is unusable alone.
=> use k-fold (RMS shrinks ~1/√folds → converges to full-bootstrap RMS ~0.09).

## 7. The heuristic CI of arXiv:2401.10542 §2.8.1 (`demo20_heuristic.py`)
σ̂_heur = σ̄_H·√(1+1/k)·(σ_H^BBfull/σ_H^BBlite), μ̂ from full-BB MLE. Toy-free.

| gen | heuristic (analytic σ̄_H) | heuristic (F-M σ̄_H) | sandwich uncalib |
|---|---|---|---|
| (0,0) | σ=0.221 cover 0.688 RMS 0.047 | 0.224 / 0.702 / 0.094 | 0.277 / 0.806 / 0.227 |
| +5σ   | σ=0.233 cover 0.698 RMS 0.046 | 0.235 / 0.720 / 0.091 | 0.282 / 0.794 / 0.128 |
implied analytic calibration k_eff=σ_heur/σ_sand=0.80-0.83 ≈ bootstrap k 0.78.

Basis: multiplicative decomposition mapping onto the 3 finite-MC effects:
  factor 1  σ̄_H              = σ∞ (data-stat floor)        <- continuous-M / F-M supplies it
  factor 2  √(1+1/k)         = per-bin MC Poisson inflation <- BB-lite σ_MC (V->V(1+1/k))
  factor 3  σ_BBfull/σ_BBlite= cross-template interplay     <- gradient-noise/incidental-params
Heuristic leap: treats the 3 as independent multiplicative inflations (not rigorous).
Factor 3 is the key: ratio of DE-ATTENUATED (full-BB) to ATTENUATED (BB-lite) curvatures
= Hessian-only proxy for the variance underestimate (what my moment-based analytic
calibration attempts missed -- they used one estimator's moments; this contrasts two).

Assessment vs our tools:
- CALIBRATION: REPLACES it. The Hessian ratio IS the analytic calibration (k_eff≈0.80
  ≈ bootstrap 0.78). Toy-free -- the calibration §5i said had no simple closed form,
  obtained here by contrasting two curvatures.
- SANDWICH: can REPLACE for σ-setting; deterministic & ~5-100x more stable (RMS 0.047 vs
  0.23). Sandwich more GENERAL (autodiff handles template correlations; eq 26 hard for
  correlated models, 1D only -- paper's caveat). Mutual cross-check.
- CONTINUOUS-M: AUGMENTS/consumes it -- supplies σ̄_H from finite MC (paper §3.6 problem);
  F-M σ̄_H fed in still covers (0.70/0.72). Same de-attenuated central value (full-BB MLE).
- BB-LITE: USED as ingredient (factor 2 inflation + ratio denominator). Not replaced.
Net: an analytic, stable re-assembly of "sandwich+calibration" from {σ∞(cont-M), BB-lite,
full-BB}; trades rigor/generality for cost/stability.

## 7b. Heuristic from M directly + multi-POI matrix form (`demo21_multiPOI.py`)
(1) sigma_BBfull = M-corrected BB-lite Hessian: (H_bb-M)^-1 -> 0.1344 vs full-BB fit
    0.1335. So extra factor σ_BBfull/σ_BBlite = √([(H_bb-M)^-1]_11/[H_bb^-1]_11) is the
    M-correction inflation. Quadrature increment ΔΣ=(H_bb-M)^-1-H_bb^-1 ≈ H_bb^-1 M H_bb^-1
    (Woodbury) -> computable DIRECTLY from M, no full-BB fit. (σ̄_H also M-derivable via F-M.)

(2) Multi-POI: matrix lift Σ_heur = (H_bb-M)^-1 H_bb (H_bb-M)^-1 (symmetric PSD, from M):
| Σ | μ0 cover | μ1 cover | joint 2D |
|---|---|---|---|
| Σ_BBfull=(H_bb-M)^-1 (σ2) | 0.440 | 0.439 | 0.461 |
| Σ_heur (matrix)           | 0.669 | 0.671 | 0.675 |
| Σ_sand (uncalib)          | 0.813 | 0.813 | 0.777 |
Restores per-component AND joint coverage from M alone, no toys/calibration.

UNIFICATION: Σ_heur=(H_bb-M)^-1 H_bb (H_bb-M)^-1 is a SANDWICH A^-1 B A^-1 with bread
A=(H_bb-M) (de-biased Fisher) and meat B=H_bb (naive curvature). The info-equality
violation B-A = M (the noise floor) IS the meat-bread gap. So the heuristic = robust
sandwich with an ANALYTIC M-based meat instead of the data-propagated one -> generalizes
to the full covariance and needs no calibration. Data-propagated sandwich = general-purpose
cross-check when template correlations make M hard to form.

## 7c. Single autograd scalar objective for the heuristic (`demo22_objective.py`)
l_heur(θ) = l_BBlite(θ) − ½θᵀMθ + ½(θ−sgθ)ᵀ ΔH (θ−sgθ),
  ΔH = (H_bb−M)H_bb⁻¹(H_bb−M) − (H_bb−M)  [evaluated at sg θ].
autograd grad → de-biased MLE (H_bb−M)⁻¹g; autograd Hessian → Σ_heur⁻¹.

| gen | med σ1 | bias(μ1) | cover μ0 | cover μ1 | joint2D |
|---|---|---|---|---|---|
| (0,0) | 0.225 | +0.007 | 0.674 | 0.674 | 0.650 |
| +5σ   | 0.226 | +0.039 | 0.649 | 0.649 | 0.631 |
Matches the demo21 matrix heuristic (~0.67).

- −½θᵀMθ puts M in the gradient → de-biased point estimate.
- stop-gradient quadratic: 0 gradient at eval point (point unchanged) but Hessian += ΔH
  → ∇²ℓ = (H_bb−M)+ΔH = Σ_heur⁻¹ = the SANDWICH curvature.
- Stop-gradient is ESSENTIAL: a natural objective's inverse-Hessian obeys the
  information equality (= de-biased Fisher (H_bb−M)⁻¹ = σ2, undercovers); a sandwich
  needs meat≠bread (differ by M), only achievable by decoupling gradient (point) from
  Hessian (covariance). This is MD §4's stop-grad M-penalty with the penalty matrix
  upgraded from −M (→σ2) to ΔH (→sandwich/coverage-correct).
- Caveat: ΔH has H_bb⁻¹ (a solve) → not HVP/matrix-free friendly. Matrix-free route:
  minimize ℓ0=BBlite-NLL−½θᵀMθ and autograd-differentiate the minimizer through the
  data (= the sandwich, demo10/21). Nonlinear: H_bb,M re-evaluated at sgθ each step.

## 7d. Data-propagated Var(score) sandwich; is M needed in minimization? (`demo23_propscore.py`)
Σ = A⁻¹ J A⁻¹, J = Var(score) by autograd-propagating diag(D),diag(M) through θ̂(data)
(Huber-White). Meat is data-propagated -> NO explicit M in the variance. But coverage
needs the central value de-attenuated, which requires M in the MINIMIZATION:

| minimization | gen | med σ | cover | bias(μ1) |
|---|---|---|---|---|
| WITHOUT M (BB-lite MLE) | (0,0) | 0.102 | 0.789 | -0.002 |
| WITHOUT M               | +5σ  | 0.103 | 0.007 | -0.305 |
| WITH M (de-biased MLE)  | (0,0) | 0.295 | 0.813 | -0.008 |
| WITH M                  | +5σ  | 0.320 | 0.834 | +0.023 |

Without M: data-propagated sandwich gives the right variance OF the biased estimator,
but the biased center -> param-dependent, collapses to 0.007 at +5σ. With M:
param-independent (0.81/0.83), over-covers -> needs calibration k.

M's role by route:
  analytic-meat heuristic : M in point + bread + meat (=H_bb=bread+M); calibration-free (0.67)
  data-propagated sandwich: M only in minimization (point+bread); meat M-free; needs k (0.81)
M is IRREDUCIBLE for the central value (or full-BB λ-profiling implicitly). The sandwich
repairs variance, not bias (cf §5e): drop M from the minimization and coverage collapses
(0.007) regardless of the meat. M for the point; data for the spread.

## 7e. M-free two-half version of the corrected heuristic (`demo24_twohalf_heur.py`)
When M is hard to form: cross-half Fisher F_ch (=H_bb-M in expectation, self-term
cancelled by A-vs-B cross product) replaces (H_bb-M). All M-free:
  point: θ̂ = F_ch⁻¹ g_ch (cross-fit MLE);  cov: Σ = F_ch⁻¹ H_bb F_ch⁻¹ (meat=full H_bb).

| version | gen | med σ | RMS σ | cover | bias |
|---|---|---|---|---|---|
| R=1 (single split) | (0,0) | 0.224 | 0.118 | 0.594 | +0.013 |
| R=8 (k-fold)       | (0,0) | 0.225 | 0.079 | 0.656 | +0.009 |
| R=1 (single split) | +5σ  | 0.227 | 0.137 | 0.565 | +0.062 |
| R=8 (k-fold)       | +5σ  | 0.228 | 0.076 | 0.632 | +0.032 |
(M-based ref demo21: σ 0.221, cover 0.67.)

Right median σ (=M-based & FC-cheat), coverage -> 0.68 with k-fold. Cost = cross-half
1-dof noise (MD §3a): single split noisy (F_ch can be non-PSD, σ fat-tailed, point
slightly biased, undercovers 0.59); k-fold averaging (R splits) halves σ-RMS, shrinks
bias, coverage 0.656/0.632 -> converges to M-based 0.67 as R grows.
Hierarchy: M computable -> (H_bb-M)⁻¹H_bb(H_bb-M)⁻¹ (lowest variance, 0.67); M hard ->
F_ch⁻¹ H_bb F_ch⁻¹ with k-fold (M-free, extra resampling noise bought down by folds).
Natural nonlinear/multi-POI route (MD §4a): re-reweighted half-templates at current θ,
all matrix-valued, never store the NP×NP M.

## 7f. Nonlinear case: minimizable autograd scalar? (`demo25_nonlinear.py`)
Model μ_i(θ)=T_i0 e^{θ0}+T_i1 e^{θ1} (nonlinear; = linear toy at θ=0).
Cross-half bilinear ℓ_ch(θ)=½Σ(D−μ^A(θ))(D−μ^B(θ))/V IS a single autograd scalar,
minimizable by trust-krylov (converges, fail≈0). M never formed; θ-dependence tracked
via re-reweighted half-templates (MD §4 route-a nonlinear).

Covariance: use GAUSS-NEWTON Fisher F_ch, H_bb (J_ij=∂μ_i/∂θ_j outer products), NOT
the raw bilinear Hessian (residual·∇²μ contamination inflated σ1 0.23->0.29).

| gen | med σ1 | bias(θ1) | μ0 | μ1 | joint2D |
|---|---|---|---|---|---|
| (0,0) | 0.230 | -0.052 | 0.696 | 0.705 | 0.165 |
| +5σ   | 0.143 | -0.005 | 0.763 | 0.634 | 0.137 |

ANSWER: YES, minimizable autograd scalar exists nonlinearly; per-component coverage
restored (~0.70, σ1 correct). CAVEAT: joint 2D covariance fails (0.17) in extreme
degeneracy (corr -0.9999) — Mahalanobis dominated by the tight (well-measured, σ~1e-3)
direction, where residual cross-fit bias/noise (bias θ1=-0.05) breaks Σ^-1. Linear case
had closed-form point (exact tight direction) -> joint OK (0.675); nonlinear iterative
minimization loses that. Fix: more folds, tighter de-biasing, or analytic M. The M-free
cross-half pays its 1-dof noise most in the well-measured directions.

## 7g. Poisson (general, non-curl-free) case (`demo26_poisson.py`)
Poisson NLL is NOT curl-free -> no single scalar gives BOTH cross-fit gradient and
Hessian (MD §4). The construction splits across two single autograd scalars:
(1) CENTRAL VALUE: full Poisson Barlow-Beeston NLL (MC as Poisson nuisances) — a single
    minimizable autograd scalar -> de-biased point. General.
(2) CURVATURE: per-bin cross-half Poisson scalar
      S(θ)=Σ_b[ μ̄_b − N_b ln(μ̄_b²/√(n^A_b n^B_b)) ],  μ̄=½(n^A+n^B),
    whose autograd Hessian = cross-half Poisson Fisher ½Σ_b(A^A A^B^T+..)/N_b.
    VERIFIED: ∇²S matches analytic F_ch to ~0.2% (residual = observed-vs-expected
    weighting 1/μ̄²,1/n^A,1/n^B vs 1/N). M never formed.
(3) Heuristic Σ=F_ch⁻¹ H F_ch⁻¹ with Poisson 1:1 weighting 1/(D_b+Σ_jM_bj):
    coverage 0.686 (gen0) / 0.649 (+5σ) ≈ 0.68 — identical to Gaussian (paper:
    Poisson≡Gaussian BB at these counts).

So: single minimizable autograd scalar for the CENTRAL VALUE (yes, Poisson BB NLL);
single autograd scalar whose HESSIAN is the de-biased curvature (yes, GM/AM scalar S);
but NOT one scalar for both (Poisson not curl-free). The Gaussian one-objective elegance
is special to the quadratic form; Poisson needs the two-scalar split (each autograd-friendly,
central-value piece globally minimizable).

## 7h. Two-half scalar for the Poisson central value (full BB infeasible) (`demo27_poisson_central.py`)
median bias of μ̂1 (runaways |bias|>1 filtered):
| scalar | bias@(0,0) | bias@+5σ | runaway |
|---|---|---|---|
| naive same-sample Poisson NLL | +0.001 | -0.299 (attenuated) | 0% |
| S  GM/AM cross-half Poisson    | +0.010 | +0.005 (DE-BIASED) | 2.1% |
| G  Gaussian cross-half bilinear| +0.009 | +0.004 (DE-BIASED) | 2.5% |

=> YES. Minimizing the GM/AM cross-half Poisson scalar S(θ)=Σ_b[μ̄_b−N_b ln(μ̄_b²/
√(n^A_b n^B_b))] (built from two halves, no full BB) gives a DE-BIASED central value
(+0.005 @+5σ vs naive -0.30) AND, via its Hessian, the de-biased curvature (§7g).
One Poisson-shaped autograd scalar does both, M never formed.
- Slightly better than MD §4 expected: §4 reserved S for curvature (BB-lite for point);
  empirically S's minimum is de-biased. The "not curl-free" caveat is about the CANONICAL
  cross-fit score not being a gradient; S is a constructed scalar whose ∇S is curl-free by
  construction and the GM/AM structure removes the self-term.
- REAL caveat = non-convexity (Hessian=cross-half Fisher can be non-PSD) -> saddle-prone,
  ~2% runaway minimizations at displaced points. Fixes (autograd-friendly): k-fold averaging
  (§7e), Gauss-Newton/LM with PSD-projected F_ch, or the linear closed form F_ch⁻¹g_ch (demo14, no divergence).

## 7i. Minimizing S directly with trust-krylov: σ and coverage (`demo28_S_trustkrylov.py`)
| gen | central bias | runaway | (∇²S)^-1 σ med/RMS/cover |
|---|---|---|---|
| (0,0) | +0.0005 | 1.1% | 0.095 / 0.019 / 0.270 |
| +5σ   | +0.032  | 2.1% | 0.095 / 0.018 / 0.260 |
(mismatched-meat corrections: F_ch^-1 H_bb F_ch^-1 ->0.32; F_ch^-1 F_ss F_ch^-1 ->0.45; still undercover.)

S directly: de-biased POINT (bias +0.0005) but σ=(∇²S)^-1 = σ∞ (0.095), coverage 0.27
(BADLY undercovers). S's Hessian is the cross-half Poisson Fisher at DATA weighting 1/N
-> σ∞ (bias fix only, NO σ_MC). The de-biased estimator's true spread is ~0.25, so a 0.095
interval covers 27% — worse than the attenuated naive (0.52, small σ matched shrunk spread).
=> S directly is the BIAS FIX (point + σ∞), not the coverage fix.
The meat corrections fail here due to WEIGHTING mismatch: S's bread is σ∞-scale (1/N), but
the covering heuristic (demo26, 0.686) uses BB-lite weighting 1/(D+ΣM) throughout (σ₂-scale
bread + σ_MC meat). Recipe: use S (or two-half) for the de-biased POINT, but take the
covariance from the BB-lite-weighted heuristic (H_bb-M)^-1 H_bb (H_bb-M)^-1, NOT (∇²S)^-1.

## 7j. S modified to include BB-lite (`demo29_S_bblite.py`)
Re-weight the cross-half scalar by BB-lite variance V=D+ΣM (vs Poisson 1/N).
Minimize with trust-krylov; F_ch=∇²S_bb.
| gen | central bias | bare (∇²S_bb)^-1 σ/cover | heuristic F_ch^-1 H_bb F_ch^-1 σ/cover |
|---|---|---|---|
| (0,0) | -0.001 | 0.134 / 0.376 | 0.225 / 0.612 |
| +5σ   | +0.028 | 0.134 / 0.365 | 0.223 / 0.576 |

Hierarchy through S:
  unweighted S        -> σ∞ (0.095), cover 0.27   [bias fix only]
  BB-lite-weighted S  -> σ2 (0.134), cover 0.38   [adds σ_MC, still undercovers]
  BB-lite S + meat    -> 0.225, cover 0.61-0.67   [coverage fix]
=> YES, S takes BB-lite naturally (just the variance weighting 1/(D+ΣM)) -> lifts the
bare Hessian from σ∞ to σ2 and the de-biased point. But a single scalar's Hessian tops
out at σ2, which still undercovers (incidental params); coverage needs the σ_BBfull/σ_BBlite
meat factor (heuristic F_ch^-1 H_bb F_ch^-1), not a single-scalar Hessian. (0.61 here vs
0.67 M-based: single-split cross-fit noise + bias; k-fold/closed-form point closes it.)

## 8. Unifying picture: S and G as one cross-fitting procedure (`demo30_general.py`)
Verified (element [1,1]): with metric w_b, same-sample F_ss = F_true + M (biased high),
cross-half F_ch ≈ F_true (de-biased) — identically for Gaussian (w=1/V) and Poisson (w=1/n).
   F_true=507506,  F_true+M=507607,  <F_ss>=507564,  <F_ch>=507463.

GENERAL PROCEDURE = cross-fitting (two-sample splitting) the likelihood's score/Fisher:
- bias: plug-in NLL curvature is a same-sample (V-statistic) outer product Σ w A A^T;
  ⟨F_ss⟩=F_true+M (own-observation/overfitting bias — A and n from the SAME MC).
- fix: polarize, A A^T -> A^A A^B^T (independent halves) -> ⟨F_ch⟩=F_true.
- general object: F_ch = Σ_b w_b (∂n^A_b)(∂n^B_b)^T, w_b = ∂²(−lnL)/∂n² = 1/Var(D_b)
  (variance-function metric: Gaussian 1/V, Poisson 1/n; coincide at Asimov).
- G, S = two SCALAR realizations: Gaussian polarized square ½(D−n^A)(D−n^B) [curl-free →
  gradient also cross-fitted]; Poisson GM/AM scalar S [Hessian = F_ch]. Same F_ch, different form.
- continuous-M (F_ss−M, MD §3b) = U-statistic / infinitesimal-jackknife LIMIT of the 2-fold
  split (all events, lower variance). {G, S, two-half F_ch, continuous-M} = one family.
- = cross-fitting / sample-splitting / double-ML de-biasing: MC templates are an estimated
  nuisance; same MC for prediction AND curvature = overfitting; cross-pairing removes the
  own-observation bias M (the V-statistic diagonal).
- divergence: single curl-free scalar for BOTH gradient and Hessian — Gaussian yes (quadratic);
  Poisson not curl-free (two-scalar split or continuous-M as fallback).

## 8b. General recipe for the de-biased cross-half scalar (`demo31_general_recipe.py`)
For ANY per-bin NLL l_b(n):
   L_cf(θ) = Σ_b [ 2 l_b(μ̄_b) − ½ l_b(n^A_b) − ½ l_b(n^B_b) ],  μ̄_b=½(n^A_b+n^B_b).
Properties:
 1. diagonal: L_cf(n,n)=l_b(n) (reduces to NLL).
 2. self-curvature cancels: ∂²L_cf/∂(n^A)²|_diag = ½l''(μ̄)−½l''(n^A) -> 0
    => ∇²L_cf = ½Σ w_b(A^A A^B^T+A^B A^A^T) = cross-half Fisher (E=F_true). w_b=l_b''.
    EXACT for constant metric (Gaussian, Neyman χ²: residual ~1e-10 verified);
    O(1/√N) for varying metric (Poisson: 0.012% verified).
 3. gradient = cross-fit (de-biased) score.
INTERPRETATION: L_cf = l(μ̄) − [½(l(n^A)+l(n^B)) − l(μ̄)] = l(μ̄) − (Jensen/convexity gap).
The Jensen gap ≈ ½l''·Var = the noise-floor self-term M. So the recipe = "NLL at the mean
minus a two-sample estimate of its own convexity bias" — a jackknife/Richardson extrapolation
2·(at mean) − (mean of half-evaluations).
REPRODUCES: Gaussian -> ½(D-n^A)(D-n^B)/V = G (machine-exact); Poisson -> μ̄-N ln(μ̄²/√(n^A n^B))
= S; Neyman χ² -> bilinear (machine-exact). Works for any twice-diff per-bin NLL.
=> G, S, (and continuous-M as U-stat limit) all fall out of this single formula.

## 8c. Final covariance after minimizing L_cf
NOT (∇²L_cf)^-1 (= de-biased curvature σ∞/σ2, UNDERCOVERS 0.27/0.45). Use the SANDWICH:
After θ̂ = argmin L_cf:
  bread A = ∇²L_cf|θ̂           (cross-half Fisher = de-biased curvature; autograd Hessian)
  meat  H = Σ_b w_b (∂n_b)(∂n_b)^T at θ̂   (UN-polarized same-sample Fisher; A_b A_b^T, not A^A A^B)
  Σ = A^-1 H A^-1
weighting w_b = 1/(D_b+Σ_j M_bj) (BB-lite -> carries σ_MC). Coverage ~0.61 (trust-krylov,
demo29) -> 0.66 (k-fold closed form, demo24) -> 0.67 (M-based, demo21).

Key identities:
- H − A = M (noise floor) => Σ = A^-1 + A^-1 M A^-1 (de-biased-curvature cov + additive M
  correction). M never explicitly formed.
- = §2.8.1 heuristic (H_bb-M)^-1 H_bb (H_bb-M)^-1 with A=∇²L_cf; = sandwich with ANALYTIC meat.
Choices/caveats:
- analytic meat H -> calibration-free (~0.67); data-propagated Var(score) meat -> over-covers
  (0.80) needs calibration k≈0.78. Prefer analytic unless template correlations force autograd.
- multi-POI: full matrix sandwich, restores per-component AND joint coverage (§7b, 0.675).
- single-split noisy -> k-fold-average the bread A for a stable per-analysis Σ (§5e/§7e).
- single-object alternative (no separate meat): stop-grad ΔH=A H^-1 A − A penalty (§7c), but
  needs H^-1 (not HVP-friendly) -> the sandwich post-step is simpler.

## 8d. Simplified form: jackknife combination of ordinary NLLs (`demo32_jackknife.py`)
Since μ̄_b = n_b^full EXACTLY (halves partition the full sample: ½(2M^A+2M^B)=M),
ℓ_b(μ̄_b) IS the full-MC likelihood, and the de-biased scalar collapses to:
   L_cf = 2 L_full − ½ L_A − ½ L_B
with L_full,L_A,L_B the STANDARD same-sample NLL on full MC and the two ×2-rescaled halves.
NO cross-products / GM-AM / polarized objective — three ordinary NLL evaluations.

Verified: max|μ̄−n_full|=0 (exact); L_cf == S(GM/AM) to the digit;
<∇²L_full>=507553≈F_true+M (biased), <∇²L_cf>=507503≈F_true=507506 (DE-BIASED).

Why (2,−½,−½): delete-half jackknife/Richardson in MC size. M∝1/N_MC: full carries M, each
half 2M -> 2(F+M)−½(F+2M)−½(F+2M)=F_true. Coefficients fixed by {reduce to NLL: 2−½−½=1} and
{cancel 1/N bias: 2−½·2−½·2=0}. Works for ANY likelihood (cancellation is in the coefficients).

Covariance also simplifies (all standard NLL Hessians):
   bread A = ∇²L_cf = 2∇²L_full − ½∇²L_A − ½∇²L_B   (de-biased)
   meat  H = ∇²L_full                                (full-sample naive Fisher; H−A=M)
   Σ = A^-1 H A^-1
PRACTICAL: implementable with EXISTING fitting code unchanged — run the ordinary likelihood
3× (full, half-A×2, half-B×2), combine 2·full−½·A−½·B, minimize, sandwich the Hessians.

## 8e. Continuous generalization + loop-free feasibility (`demo33_continuous.py`)
Continuous limit of 2L_full-½(L_A+L_B) = infinitesimal delete-1 jackknife / U-statistic
= analytic noise floor M, de-biased curvature F_ss-M (MD §3b):
   M_ij = Σ_b (Σ_{e∈b} w_e² s_i(e) s_j(e)) / μ_b,   s_i(e)=∂lnμ/∂θ_i.
Verified vs 2-fold cross-half (both de-biased, median≈σ∞=0.095):
   continuous-M : mean 0.0976 RMS 0.0152 nonPSD 0.00%
   2-fold       : mean 0.0994 RMS 0.0237 nonPSD 0.09%
=> continuous-M ~35% lower variance + PSD-stable (U-statistic advantage).

FEASIBLE WITHOUT per-iteration event loop: precompute (one pass) the augmented moment
histograms: H0_b=Σw_e; derivative templates A_i,b=Σ w_e s_i(e)=∂μ_b/∂θ_i; noise-floor
moments Q_ij,b=Σ w_e² s_i s_j. Then each iteration reconstruct (cheap, over bins only):
   μ_b(θ), F_ss(θ)=Σ A_i A_j/μ_b, M_ij(θ)=Σ_b Q_ij,b/μ_b.   (M(θ) rebuild verified, no loop)
- linear/affine-in-θ: Q_b,A_b fixed -> exact, loop-free.
- nonlinear (scores θ-dep): exact-derivative construction stores per-event scores once ->
  reconstruct to needed order, no per-iter loop (re-loop only if higher moments needed).
- large N_P: matrix-free M·v = Σ_b (Σ w_e²(s_e·v)s_e)/μ_b (one JVP), store length-N_P vectors.
Final covariance (continuous): Σ = (F_ss-M)^-1 F_ss (F_ss-M)^-1 (BB-lite weighting for σ_MC).
= the repo's EBFIT_MCORR (N_P(N_P+1)/2 moment vector per bin = precomputed Q_ij,b).

## 8f. O(N_P·N_bins) feasibility (avoiding the N_P²·N_bins moment Q)
Continuous-M needs Q_ij,b=Σ_{e∈b} w_e² s_i s_j (N_P²·N_bins). Routes to O(N_P·N_bins):

GENERAL: use the 2-fold/k-fold CROSS-HALF — needs only the half-sample DERIVATIVE
templates A^A_i,b, A^B_i,b (first moments, two N_P×N_bins tensors), NOT Q:
   F_ch = ½Σ_b (A^A_i A^B_j + A^B_i A^A_j)/μ_b ;  objective 2L_full-½(L_A+L_B).
   matrix-free: F_ch·v = ½Σ_b[(A^B_b·v)A^A_b+(A^A_b·v)A^B_b]/μ_b  (O(N_P·N_bins), no events).
   Verified (demo33): de-biases identically to continuous-M using only half templates, no Q.
   Cost: variance (2-fold 1-dof noisier); k-fold recovers it at O(k·N_P·N_bins).

Continuous-M stays O(N_P·N_bins) in two cases:
  1. Process-normalization params (common; the toy): scores = process indicators ->
     Q diagonal -> M_ij=δ_ij Σ_b sumw2_i,b/μ_b. Store one N_P×N_bins array. Exact.
  2. Fine binning, continuous scores: rank-1 approx Q_b≈sumw2_b (A_b/H0_b)(A_b/H0_b)^T ->
     M(θ)≈Σ_b sumw2_b/(H0_b² μ_b) A_b A_b^T. O(N_P·N_bins); error = within-bin score variance.
     (Fails for the toy: bins mix processes -> use case 1 diagonal instead.)

Trade-off: continuous-M lowest variance but O(N_P²) for general scores; cross-half/k-fold
O(N_P·N_bins) always (store half derivative templates, matrix-free, no re-loop), variance
recovered by raising k. k-fold interpolates continuous-M <-> 2-fold.

---

## §9. Rabbit implementation: continuous-M validated, linearity caveat discovered

Branch `mcstat-debias` in `rabbit/`. `tensorwriter.add_mc_stat_moment(M, param_names)`
builds a frozen `nparams²` external-likelihood term with `hess = -M`, reusing rabbit's
external-quadratic machinery (`compute_external_nll` evaluates `0.5 xᵀH x` → `-½θᵀMθ`).
The term correctly enters value, gradient, AND Hessian:
at fixed `x=[1,1]`, `hess(M) - hess(noM) = diag(-99.0, -100.8) = -M` exactly (off-diagonal
unchanged) — so the external-term route is the right plumbing for continuous-M.

### §9a. **Key finding: the `-½θᵀMθ` penalty only de-biases when the de-biased
parameter enters the prediction LINEARLY.**
The 200-bin/2-process degenerate toy (`tests/toy_mcstat.py`, `tests/verify_mcstat.py`),
fit two ways for the POI signal-strength multiplier `rnorm`:

| POI parametrization | central value noM→M | σ_rnorm(flat) noM→M | σ_rnorm(dif) noM→M |
|---|---|---|---|
| **squared `rnorm=x²`** (rabbit default, `allowNegativeParam=False`) | 1.0614→1.0615 (no shift) | 0.0600→0.0600 | 0.0840→0.0840 |
| **linear `rnorm=x`** (`--allowNegativeParam`) | 1.0614→1.2067 (attenuation corrected) | 0.0600→0.1105 | 0.0840→0.1548 |

(Both `noM` cases agree at 0.0600/0.0840 regardless of transform — the standard MLE is
parametrization-invariant, a good consistency check.)

**Mechanism.** The penalty contributes `-Mθ` to the gradient, so the de-biased minimum
satisfies `g_data(θ̂) = Mθ̂` — pulled off the data-only MLE. With a NONLINEAR map
(`rnorm=x²`) the data Fisher `F(θ) ∝ (∂rnorm/∂x)² = (2x)²` grows as the minimum shifts
up, and the growth almost exactly cancels the `-M` subtraction: net curvature, central
value, and covariance (in physical `rnorm` space, via the `2x` Jacobian) all essentially
unchanged. With a LINEAR map the data Hessian is constant, so `A = H - M` and
`θ̂ = (H-M)⁻¹g` hold exactly as §1b/the slides derive — the eigenvalue along the
degenerate direction drops by the M-projection (141.6→41.7, M·d̂≈99.9) and σ inflates.

**Why this matters / is consistent.** §1b's `θ̂=(H-M)⁻¹g` is implicitly a linear-Gaussian
result. Free-floating normalizations — the original motivation and the recommended first
deliverable (§3) — ARE linear in `rnorm`, but rabbit's default squaring transform hides
that; **continuous-M requires `--allowNegativeParam` (or any linear param model) for the
de-biased POIs.** For parameters that enter nonlinearly (response models, exp/softplus
shapes), the penalty is cancelled by the Fisher-growth and the **two-half / k-fold
cross-fit objective is the correct route** (it de-biases via the data-independent
`norm^A·norm^B` cross structure regardless of nonlinearity — RESULTS §7e/§8d, demo34).

### §9b. Implications for the rabbit design
- Continuous-M (§1): document the linearity requirement; emit a warning (or hard error)
  if `mcStatDebias=continuousM` is requested for POIs/params that are not linear
  (`allowNegativeParam=False` squared POIs, or any nonlinear param model entry).
- The sandwich covariance `A⁻¹HA⁻¹` (the coverage fix) is still pending in the fitter;
  the curvature de-bias `A=H-M` is now confirmed working (linear case).
- Two-half (§2) remains the general-purpose method and is unaffected by this caveat.

### §9c. Continuous-M sandwich covariance + numpy cross-check (rabbit)
Added `Fitter.cov_mcstat_sandwich(A)` = `A⁻¹ + A⁻¹ M A⁻¹` (analytic; meat `H=A+M`
since the penalty is the frozen quadratic, no 2nd Hessian pass), `Fitter._build_mcstat_M`
(reconstructs M from the `mcstat*` external term: `M=-scatter(hess_dense)`), CLI
`--mcStatDebias/--mcStatDebiasCov/--mcStatKfold/--covMode`, the linearity warning, and
the `rabbit_fit.py` hook that reports the sandwich when `continuousM`+`sandwich`.

**Exact agreement rabbit vs independent numpy reference** (same toy templates, LINEAR POI):

| quantity | numpy ref | rabbit |
|---|---|---|
| de-biased point `r_M` | [1.2067, 0.799] | [1.2067, 0.799] |
| curvature σ(dif) noM→M | 0.0840 → 0.1548 | 0.0840 → 0.1548 |
| sandwich σ(dif) | 0.2853 | 0.2853 |
| curvature σ(flat) noM→M | 0.0600 → 0.1105 | 0.0600 → 0.1105 |
| sandwich σ(flat) | 0.2035 | 0.2035 |

rabbit == reference to 4 d.p. Because the numpy ensemble of this exact construction
gives ~0.68 coverage (RESULTS §3/§5), rabbit's continuous-M inherits correct coverage.
**Continuous-M path is complete and validated end-to-end** (writer → inputdata via the
existing external-term machinery → fitter de-biased point/curvature → sandwich cov),
subject to the §9a linearity requirement.

### §9d. Two-half / k-fold cross-fit (rabbit) — implemented + validated
Writer: `add_process(h, ..., fold_axis=<name>)` slices a named MC-stat fold axis
into k folds, derives the full template by projecting it out, and stores
`hnorm_folds [k, nbinsfull, nproc]` (non-folded processes get `norm_full/k` per
fold → identical across folds → MC-stat-EXEMPT, drop out of the de-bias). inputdata
loads `norm_folds`/`mcstat_fold_k`. Fitter precomputes half templates
`norm_A=2·Σ_{f<k/2} folds`, `norm_B=2·Σ_{f≥k/2} folds`; `_compute_yields_noBBB(templates=
"A"|"B")` substitutes them (shared logk); `_compute_nll_components` forms the
jackknife `L_cf = 2L_full − ½L_A − ½L_B`. Sandwich `Σ=A⁻¹HA⁻¹` via
`cov_twohalf_sandwich` (bread A = jackknife Hessian, meat H = full-sample Hessian
`loss_val_grad_hess_meat`).

**(i) Exact agreement vs numpy reference (LINEAR POI), toy_folds.hdf5:**

| quantity | numpy | rabbit |
|---|---|---|
| de-biased point r_cf | [1.1478, 0.8568] | [1.1478, 0.8568] |
| curvature σ(dif) std→cf | 0.0840 → 0.1367 | 0.0840 → 0.1367 |
| sandwich σ(dif) | 0.2224 | 0.2224 |

**(ii) The differentiator — two-half de-biases under the NONLINEAR (default x²) POI**
(where continuous-M is cancelled, §9a): physical `rnorm` point [1.0614→1.1478] AND
covariance curv(dif) 0.0840→0.1367, sandwich 0.2568 both move. The data-independent
`norm^A·norm^B` cross structure de-biases regardless of the parameter→prediction map.

**Summary of the rabbit MC-stat de-biasing branch (`mcstat-debias`):**
- continuous-M: lowest variance (full M moment), LINEAR params only, analytic sandwich.
- two-half/k-fold: general, nonlinearity-robust, needs only fold templates
  (O(k·nparams·nbins)), variance reduced by raising k.
- `--covMode {observed,fisher}`: BOTH implemented (autograd Hessian; or Gauss-Newton
  J^T D J dropping residual·∂²μ) for the standard cov and the sandwich bread+meat (§9f).
- Both validated to 4 d.p. against independent numpy references; coverage inherited
  from the numpy ensemble (§3/§5).

### §9e. Full CLI pipeline + conditioning note
End-to-end `bin/rabbit_fit.py` runs (minimize → Hessian → sandwich → workspace
output) verified on the folded toy with the DEFAULT (squared) POI:
- `--mcStatDebias twoHalf`: edmval 3.7e-17 (fully converged), reports the two-half
  sandwich, p-value 100%; standard baseline edmval 1e-16, p-value 22.91%.
Two-half therefore works end-to-end in the realistic well-conditioned setting AND
de-biases there (it is nonlinearity-robust, §9d-ii).

Conditioning caveat (toy-specific, not a de-bias bug): with `--allowNegativeParam`
(LINEAR POI) the corr=-0.9999 toy is extremely ill-conditioned (Hessian eig ratio
~1e4) and the CLI iterative trust-krylov can stall (edmval ~120; continuous-M's H-M
can momentarily go non-PD mid-iteration). The SAME fits converge cleanly when driven
programmatically (|grad|max ~1e-11, all Hessians PD, exact numpy match, §9a/§9c/§9d),
so the de-biasing math/code is correct; this is a minimizer-robustness edge of the
pathological toy. Real analyses (free-floating norms, milder correlations) are
well-conditioned. Continuous-M needs the linear POI (§9a); two-half does not and is
the robust default.

### §9f. --covMode fisher and split-logk (rabbit) — implemented + validated
**fisher (Gauss-Newton).** `fitter.fisher_curvature` / `fisher_curvature_full` /
`_fisher_core` build `F = H_obj - H_ln_obs + Σ_i coeff_i J_iᵀ D J_i` (D=1/V chisq,
data_cov_inv covFit, 1/nexp Poisson). For the jackknife objective the GN data part
is exactly the cross-half Fisher `F_ch`. `--covMode fisher` uses it as the bread for
the standard cov and the sandwich (continuous-M & two-half); the two-half meat
follows covMode too. Validated: LINEAR → fisher == observed to 5.8e-11 (GN=observed
when ∂²n=0); NONLINEAR (x²) → differs by 5.4e-4 (drops the residual·∂²n term). CLI
runs both modes end-to-end (edmval ~1e-17).

**split-logk.** `add_systematic(h, ..., fold_axis=<name>)` splits a systematic's logk
per fold (stored `hlogk_folds [k,nbinsfull,nproc,nsyst]`; shared systs replicated so
`logk_A=logk_B`), so a systematic that REUSES the nominal MC sample has its own
template noise de-biased too. Fitter builds `logk_A/logk_B` (k=2; per-fold logk =
per-half logk; same rnorm_init scaling as the shared logk) and uses them in the
two-half yields. Validated on a folded-systematic toy (`tests/toy_splitlogk.py`),
`tilt` nuisance σ:

| | standard | two-half shared-logk | two-half split-logk |
|---|---|---|---|
| σ(tilt) | 0.0705 | 0.0705 (no syst de-bias) | 0.0907 (de-biased) |

Shared logk de-biases only the nominal-template noise (via norm^A·norm^B); split logk
additionally de-biases the systematic-template noise — exactly the intended
differentiator. Scope: k=2, symmetric, dense, single-hist (mirror) systematics.

### §9g. BB-lite (bin-by-bin stat) compatibility (rabbit)
Examined and fixed the composition of the de-biasing with Barlow-Beeston-lite.
Both methods diverged catastrophically with BB-lite ON before the fixes; root causes
and fixes:

**continuous-M + BB-lite.** With BB-lite the effective per-bin variance is the
inflated `mu_b + Σ_proc sumw2_b`, so the curvature `H_bb` shrinks. M must use that
SAME inflated variance in its denominator; with the bare data variance M is ~2× too
large, `H_bb − M` goes non-PD and the fit diverges (|grad|~1e7). Fix: M built with
the BB-variance converges cleanly (toy_Mbb: x=[1.2012,0.8045], |grad|4.8e-6, PD
eig [21.0, 4.99e5]). The fitter now WARNS when continuousM+BB-lite is active
(M-denominator reminder), and `add_mc_stat_moment` documents it. This is a
user/writer responsibility (M is supplied), not a fitter computation.

**two-half + BB-lite.** Structural, not toy-specific (standard BB-lite fits converge
fine on the same toys): the jackknife `2L_full − ½L_A − ½L_B` is UNBOUNDED with
BB-lite because β-profiling flattens `L_full`'s POI curvature while the negative
half-terms keep full curvature → `A = 2H_full,bb − ½H_A − ½H_B` indefinite (|grad|
diverges to 1e11–1e21). Fix: apply the full-sample profiled β (per-bin factor
`β = nexp/nexp_full_raw`) to the HALF predictions too, so all three terms are
consistently profiled and `A ≈ H_full,bb > 0`. With this, two-half+BB-lite is
bounded/PD and composes sensibly (well-posed split-logk toy: central values correct,
BB-lite inflates the baseline tilt σ 0.0705→0.0992, two-half de-biases on top →
0.133 curv / 0.178 sandwich). Both `--covMode` modes work (fisher≈observed).

Caveats: (i) β is the standard full-profiled β shared with the halves; the residual
cross-fit β-non-stationarity is higher order (results are sensible). (ii) Extremely
degenerate problems (corr −0.9999 toy_folds) can find spurious far minima under
BB-lite+two-half — an amplification of the degeneracy, mitigated by good starting
values; the milder split-logk toy is well-behaved. [Superseded by §9u: sharing
the full-sample β over-subtracts by 1 + ρ, and these failures were that.]
Validation: `tests/verify_bblite.py`
(asserts PD convergence + de-bias for both methods with BB-lite on). No-BB paths
unchanged (verify_mcstat / verify_twohalf still exact).

### §9h. Impacts / output propagation of the de-biased covariance (rabbit)
Examined how the de-biased covariance reaches the derived products and fixed the
gaps. `fitter.cov` is set to the reported de-biased covariance (sandwich or
curvature), so everything that reads it already propagates:
- parameter errors (`parms_variances = diag(fitter.cov)`),
- expected/output histogram bin errors (`expected_with_variance` uses `self.cov`;
  verified: sandwich-cov bin errors > curvature-cov by ~7% on the toy → `--saveHists
  --computeHistErrors` reports the de-biased uncertainties),
- postfit variations (`dx = self.cov/√diag`).

The gaps were in IMPACTS, now fixed:
- **Traditional impacts** (`--doImpacts`) decomposed `self.cov` (sandwich) for the
  syst/per-nuisance pieces but took the stat piece from the raw observed Hessian —
  inconsistent, and `--covMode fisher` was not propagated. Fix: `impacts_parms(hess,
  cov=…, extra_group_vars=…)` now decomposes the de-biased CURVATURE `A⁻¹` with the
  matching curvature `bread` (covMode-aware) for ALL pieces (total/syst/stat), so the
  decomposition is internally consistent. When the reported cov is the sandwich, the
  extra coverage term `diag(sandwich − curvature)` is appended as a new
  **`mcStatDebias`** grouped-impact column (axis label added in `workspace.py`).
  Verified: columns `[<systgroups>, stat, mcStatDebias]` match the axis; the
  mcStatDebias column equals `√diag(sandwich−curvature)`.
- **Global impacts** (`--globalImpacts`) gained a `cov` override so they decompose
  the de-biased curvature consistently too.

All `--mcStatDebias × --covMode × --mcStatDebiasCov` combinations run `--doImpacts`
/ `--globalImpacts` / `--saveHists` end-to-end; standard (non-debias) impacts and the
no-BB / BB-lite validation suites are unchanged.

### §9i. k-fold averaging (complete U-statistic) (rabbit)
Implemented the k-fold variance reduction as the COMPLETE pairwise U-statistic over
all k folds, via the full-minus-self identity (no enumeration of partitions):

    A_k = k/(k-1) [ J_full^T D J_full − Σ_i J_i^raw^T D J_i^raw ]
        = k/(k-1) Σ_{i≠j} J_i^raw^T D J_j^raw      (average over ALL fold pairs)

- **O(k)** per evaluation (one full term + k per-fold self terms), streamable, no
  combinatorial cost; memory is just the k fold templates (already stored).
- Reduces to the single 2-half cross-half Fisher **exactly at k=2** (so it replaces
  the k=2 fisher path uniformly — no separate code).
- Lives in the GN/`--covMode fisher` curvature (the raw folds ~n_full/k are
  residual-independent there; the objective keeps the rescaled halves for a sensible
  point). `_fisher_data_terms` builds the term list; `_fisher_core` now uses the
  objective's rescaled halves for `H_ln` (non-data extraction) and the U-statistic
  raw folds for the GN data Fisher `F`. Observed mode k≥3 uses the single 2-half
  split (no averaging) — the fitter logs an INFO pointing to `--covMode fisher`.
- k is inferred from the fold-axis size; `--mcStatKfold` is now deprecated/unused
  (the complete U-statistic is parameter-free).

Validation (`tests/toy_kfold.py`, `tests/verify_kfold.py`):
- rabbit k=4 fisher curvature == numpy U-statistic to 4 d.p. (σ(dif)=0.1583), and
  differs from the coarse single 2-half split (0.1502) → averaging is active for k>2.
- ensemble: median ≈ σ∞=0.1343 (unbiased) at all k; RMS shrinks 0.032→0.023→0.021
  for k=4,8,16, toward the continuous-M floor (~0.015) — the 2-fold↔continuous-M
  interpolation. k=2 single split is high-variance (occasionally non-PD); averaging
  stabilizes it. No regression (verify_mcstat/twohalf/bblite unchanged; converges
  cleanly on a well-conditioned k=4 toy).

### §9j. k>2 split-logk (rabbit)
Lifted the k=2-only restriction on split-logk (per-fold systematic logk). The
restriction was for forming the per-HALF logk used in the OBJECTIVE (log of a sum
!= sum of logs), but the k-fold U-statistic CURVATURE only needs the per-FOLD logk,
which is exactly what the writer stores. Changes:
- Fitter builds `logk_folds_scaled [k, ...]` (per-fold rnorm_init-scaled logk) for
  ANY k; the k=2 half logk (objective full treatment) is set from it, and for k>2
  the objective falls back to shared logk (the point de-biases the dominant
  nominal-template noise via norm^A/norm^B; the systematic-template noise is
  de-biased in the curvature) with an INFO.
- `_fisher_data_terms` now uses the SUM of the per-fold predictions
  `n_sumfold = Σ_i n_i` as the U-statistic "full" term (and as the meat), so the
  full-minus-self identity `F_sumfold − Σ_i F_i = Σ_{i≠j} cross` holds EXACTLY for
  both shared logk (where `Σ_i n_i == nexp_full`) and split logk (where it does
  not). BB-lite β is applied to every per-fold prediction consistently.

Validation: shared-logk paths unchanged (`Σ_i n_i == nexp_full` → verify_kfold /
verify_twohalf / verify_bblite all still pass; k=2 split-logk tilt sandwich
unchanged at 0.0907). New k=4 split-logk toy (`tests/toy_splitlogk.py K=4`):
per-fold logk `[4,nbins,1,1]` loaded and used; converges (|grad| 7e-9); the `tilt`
systematic (which reuses the nominal MC) is de-biased MORE with split logk than
shared (sandwich 0.130 vs 0.086). So split-logk now works for any k≥2.

### §9k. Asymmetric folded systematics (rabbit)
Lifted the "mirror=True only" restriction on folded systematics. Findings:
- Folded PROCESS + asymmetric SHARED (non-folded) systematics already worked (the
  nominal fold tensor is systematic-structure-independent; per-fold yields use the
  shared asymmetric logk via the existing interpolation) — verified converges/PD.
- Folded ASYMMETRIC systematics (split-logk on an up/down pair) now supported:
  `add_systematic([up, down], ..., fold_axis=…, symmetrize=…)` computes per-fold
  logkavg (+ per-fold logkhalfdiff for the asymmetric symmetrize mode), stores them
  (`dict_logkhalfdiff_folds`), and `write()` assembles the fold logk in the matching
  shape ([k,…,nsyst] symmetric or [k,…,2,nsyst] asymmetric; folded systs split,
  others replicated). The fitter scales the per-fold logk shape-aware and feeds it to
  the U-statistic curvature via the existing asymmetric interpolation branch.
  ('linear'/'quadratic' symmetrize — which split into 2 systematics — not supported
  with folds.) Validated: k=4 folded asymmetric syst loads logk_folds [4,40,1,2,1],
  converges (|grad| 2e-12), PD. Symmetric/split paths unchanged (verify_*).

### §9l. Sparse-mode folds (rabbit)
The de-bias fold path now works in sparse mode. Key facts:
- The fold norm tensor is stored DENSE [k, nbinsfull, nproc] even in sparse mode
  (the O(k·nparams·nbins) de-bias cost); the FULL template stays sparse (CSR).
- The fisher U-statistic curvature uses the dense per-bin yield and a dense
  [nbins, nparams] Jacobian, so it is feasible whenever the standard dense
  covariance is (sparse-input + dense-cov); only the per-fold YIELD path was missing.
- Implemented sparse per-fold yields: scatter the SHARED systematic factor from the
  sparse logk (`exp(logsnorm)` log_normal / additive `logsnorm` normal) into a dense
  [nbinsfull, nproc] grid and contract with the dense fold/half norm. Removed the
  writer's sparse fold_axis raise (folds densified). split-logk (per-fold logk) is
  NOT supported in sparse mode (raises) — shared logk only.

Validation (`tests/verify_sparsefold.py`): a sparse k=4 fold tensor gives EXACTLY the
same de-biased point, fisher curvature, and sandwich as the equivalent DENSE tensor
(match to 1e-6). Existing sparse non-fold path (templates="full") unchanged; verify_*
suites pass.

**Status of the rabbit MC-stat de-biasing branch (`mcstat-debias`):** continuous-M
and two-half/k-fold (complete U-statistic averaging); `--covMode {observed,fisher}`
and `--mcStatDebiasCov {curvature,sandwich}` throughout; BB-lite compatible; impacts
+ output-hist propagation; split-logk for any k≥2 (dense); asymmetric folded
systematics; sparse-mode nominal folds. Remaining (documented limits): sparse
split-logk, and large-`nparams` `--noHessian` curvature (needs matrix-free).

### §9m. Sparse split-logk (rabbit)
Lifted the sparse split-logk restriction. The sparse fold-yield path uses the SHARED
systematic factor from the sparse logk; split-logk needs the per-fold logk. Rather
than densify the full [k,nbins,nproc,nsyst] logk (huge in sparse mode), store only a
per-fold DELTA over the FOLDED systs (a small subset): `delta = logk_fold - logk_full`,
dense [k, nbinsfull, nproc, n_folded], plus the folded global syst indices. The fitter
multiplies the shared log_normal factor by `exp(delta . theta)` for the active fold
(`bpj,j->bp` over the folded systs), so `shared_logk + delta = fold_logk` exactly. The
writer computes `logk_full` densely in `_add_systematic_folded` and stores the delta;
`write()` assembles `hlogk_folds_delta` + `hmcstat_folded_syst_idx` (sparse mode only);
`inputdata` loads them. Applied in the CURVATURE (templates='fold') path; the sparse
objective/halves use the shared logk (point de-biases nominal noise, as for dense k>2).
Scope: log_normal symmetric folded systs (asymmetric sparse split-logk raises).

Validation (`tests/verify_sparse_splitlogk.py`): a sparse k=4 split-logk tensor gives
EXACTLY the dense result (point, fisher curvature, sandwich to 1e-6; delta [4,40,1,1]).
All other fold paths unchanged (verify_kfold/twohalf/bblite/sparsefold pass; dense
asymmetric folded syst still converges PD).

**Fold-generality is now complete:** symmetric/asymmetric systematics × dense/sparse
tensors × shared/split logk × any k≥2. Remaining documented limit: large-`nparams`
`--noHessian` curvature (matrix-free cross-fit, a separate redesign).

### §9n. Native template-fluctuating coverage harness (rabbit)
Closed the methodological gap: rabbit's built-in toys fluctuate the DATA only, but
MC-stat coverage needs the TEMPLATES fluctuated. `tests/mcstat_coverage.py` does
both per pseudo-experiment (fluctuate templates AND data, build a rabbit tensor,
fit, measure POI coverage). Near-degenerate 2-process slides toy with a NONZERO
degenerate-direction truth (rnorm_true=[1.3,0.7], so the difference direction has a
real value the attenuation can bias — with rnorm_true=[1,1] the difference truth is
0 and there is no central-value bias to correct).

Findings (validates 2 of the 3 original criteria cleanly; characterizes the 3rd):
- **Central value: de-bias works (strong, clean).** Naive is badly attenuated:
  med_bias(diff) ~ −0.29 on a true value 0.42 (~70% attenuation toward 0), coverage
  ~0.03–0.08. continuous-M / two-half REMOVE it: med_bias → ~0 (−0.02 to +0.005 in
  the well-behaved regime). The original "correct central value" criterion ✓.
- **Uncertainty estimate: restored toward sigma_inf.** Naive sigma underestimates;
  the de-biased curvature ~ sigma_inf (slides: 0.080 → 0.135 ≈ sigma_inf 0.134). ✓
- **Coverage: improved but not exactly 0.683.** The raw curvature/sandwich sigma
  underestimates the de-biased point's ENSEMBLE scatter by ~20–25% (the MC-noise
  variance IN THE SCORE is not captured by the single-toy meat), so coverage lands
  ~0.40–0.55, not 0.683. The Bartlett/calibration factor (RESULTS §6–§7) closes
  this. So the harness CONFIRMS the known calibration limitation rather than
  contradicting the de-bias. In the extreme-degenerate regime the de-biased point is
  additionally heavy-tailed (rms >> sigma_inf from per-toy near-singular H−M);
  milder regimes are well-behaved but the calibration gap persists.

Tests: `tests/mcstat_coverage.py` (harness; configurable debias × covMode ×
debiasCov × BB-lite × chisq/Poisson, reports coverage/bias/sigma), and
`tests/verify_coverage.py` (fast assertion: naive attenuated+undercovers; de-bias
removes the bias, inflates sigma, improves coverage). This makes "exact coverage
needs the calibration factor" a MEASURED, reproducible statement instead of an
inherited numpy claim — and identifies the calibration factor as the natural next
implementation item for exact coverage.

### §9o. Reconciliation: the PROPER sandwich removes the calibration need (rabbit)
§9n's "coverage ~0.5, calibration needed" was measured with `--noBinByBinStat`, i.e.
the BARE sandwich whose meat (the full Hessian) OMITS sigma_MC. That is NOT the proper
sandwich the numpy analysis validated (RESULTS §7-8): the calibration factor is exactly
the ratio between the ensemble Var[score] and a meat that omits the MC-noise variance,
so a meat that INCLUDES it needs no calibration. Two proper routes:
  (a) BB-lite + M-correction: the per-bin beta nuisances put sigma_MC INTO the
      curvature, restoring the information equality (curvature = Var[score]) for the
      MC-aware model. In rabbit the continuous-M+BB-lite sandwich meat is A+M = H_bb
      (the BB-lite curvature), so it IS route (a).
  (b) cross-fold score-variance meat (data-propagated Var(score) from the halves).
      NOTE: rabbit's two-half sandwich currently uses the HESSIAN meat
      (loss_val_grad_hess_meat), not the cross-half score variance, so it is NOT yet
      route (b) and still needs calibration without BB-lite -- the cross-half
      score-variance meat is the concrete missing piece for self-calibration.

Empirical confirmation (native harness, near-degenerate toy, rnorm_true=[1.3,0.7],
diff_true=0.424, ntoy=150):

| construction | cov(dif) | med_sig | rms_res(scatter) |
|---|---|---|---|
| none (no BB-lite) | 0.487 | 0.027 | 0.040 |
| BARE continuous-M sandwich (no BB-lite) | 0.513 | 0.029 | 0.042 |
| PROPER continuous-M sandwich (BB-lite, sigma_MC in meat) | **0.670** | 0.041 | 0.040 |

The de-biased point is unbiased in all three (med_bias ~ 0). The BARE sandwich's sigma
(0.029) underestimates the ensemble scatter (0.042) -> undercovers (0.51); the PROPER
(BB-lite) sandwich's sigma (0.041) MATCHES the scatter (0.040) -> covers 0.67 ~ 0.683
with NO calibration factor. So the calibration need is removed by putting sigma_MC in
the meat (BB-lite), exactly as the numpy tests showed -- §9n's gap was an artifact of
testing the bare construction. Caveat: BB-lite on this extreme-degenerate toy is
fragile (23% of toys gave a rank-deficient covariance, filtered; surviving toys give
clean coverage) -- the §9g BB-lite+degeneracy fragility, not a de-bias issue. The
covMode choice (observed vs fisher) does NOT change this (calibration is about
meat=curvature vs meat=Var[score], not which curvature).

### §9p. Data-propagated Var(score) sandwich option (rabbit)
Added `--mcStatDebiasCov dataPropagated` (+ `Fitter.cov_dataprop_sandwich`): the
Huber-White / delta-method sandwich from §7d (demo23), propagating BOTH the per-bin
DATA variance and the per-(bin,proc) TEMPLATE variance (sumw2 = MC-stat variance)
through the de-biased estimator by implicit differentiation:

    Sigma = dx/dnobs . diag(Var[nobs]) . (dx/dnobs)^T
          + dx/dnorm . diag(sumw2)     . (dx/dnorm)^T,   dx/dv = -A^-1 d2L/dx dv.

The first term equals the analytic A^-1 H A^-1; the second adds the template-noise
propagation (computed via a GradientTape watching indata.norm). Continuous-M, dense.

CORRECTION to §9o: the score-variance / data-propagated meat is NOT "self-
calibrating" -- per §7d it is CONSERVATIVE and OVER-covers. Confirmed in rabbit on
the slides toy (rnorm_true=[1.3,0.7], ntoy=120): one toy sigma(dif) curvature 0.122 <
analytic sandwich 0.193 < data-propagated 0.366; ensemble coverage:

| meat | cov(dif) |
|---|---|
| none | 0.06 |
| analytic sandwich (no BB-lite) | 0.50 (under) |
| **data-propagated** | **0.79 (over)** |

So the three meats bracket the target: bare-Hessian under-covers (~0.5, needs k up);
analytic/BB-lite meat on-target (~0.67, calibration-free, §9o); data-propagated
over-covers (~0.79, conservative, needs a downward k~0.81). Use dataPropagated when
you want a conservative interval without tuning. covMode-independent. The de-biased
POINT (M in the minimization) is still required for an unbiased centre (§7d).

### §9q. Review follow-up: scaling, odd k, additive systematics, sparse storage (rabbit)
Changes made in response to the design review of the PR, after rebasing on main:

- **Gauss-Newton curvature is now O(nparams·nbins).** `_fisher_core` used to build
  each U-statistic term's Jacobian `J_i = ∂n_i/∂θ` with a reverse-mode
  `GradientTape.jacobian` of an `[nbins]` output — one backward pass per bin, so
  O(nbins²) per term. It now evaluates `Σ_i c_i J_iᵀ D J_i` as the Hessian of the
  surrogate `S(x) = ½ Σ_i c_i r_iᵀ D r_i`, `r_i = n_i(x) − stop_gradient(n_i(x))`. The
  residual vanishes identically at the evaluation point, so only `JᵀDJ` survives, and
  the Hessian goes through the same reverse-over-reverse jacobian as the ordinary
  Hessian. All k folds are evaluated in one batched contraction, so the graph no
  longer grows with k. Steady state, k=4, 2 POIs (same machine for both):

  | nbins | dense before | dense after | sparse before | sparse after |
  |---|---|---|---|---|
  | 800   | 5.1 ms  | 0.5 ms | 43 ms   | 1.6 ms |
  | 3200  | 50 ms   | 1.7 ms | 1396 ms | 12 ms  |
  | 6400  | 150 ms  | 1.6 ms | –       | –      |
  | 12800 | –       | –      | –       | 18 ms  |
  | 25600 | –       | 4.0 ms | –       | –      |

  i.e. now about the cost of one ordinary Hessian. First-call trace time dropped
  from 3–9 s to ~1 s. The curvature agrees with the old implementation to ~1e-15
  relative.
- **Any k ≥ 2, including odd k.** The objective's halves are `G_A = {0..a−1}`,
  `G_B = {a..k−1}`, `a = ⌊k/2⌋`, `b = k − a`, each rescaled to the full normalization
  (`n_A = k/a·S_A`, `n_B = k/b·S_B`). The jackknife weights generalize to
  `L_cf = k²/(2ab)·L_full − a/(2b)·L_A − b/(2a)·L_B` (= `2, ½, ½` for even k): these
  are the unique weights for which the self terms `S_AᵀDS_A`, `S_BᵀDS_B` cancel in the
  Gauss-Newton Hessian (leaving `sym(J_AᵀDJ_B)`) and the score is the half average
  `½(J_A+J_B)ᵀD(y−n)`, so the estimator has exactly the even-k structure. Numpy
  ensemble on the §9d toy (600 template draws per k): median σ(dif) from the
  cross-half curvature = 0.1355 / 0.1350 / 0.1359 / 0.1350 for k = 2/3/4/5 against
  σ∞ = 0.1343 (naive ≈ 0.080). In rabbit, k = 2, 3, 5 reproduce the numpy point,
  observed Hessian (= cross term), fisher U-statistic and sandwich exactly, dense and
  sparse. The `NotImplementedError` for odd k is gone.
- **Split-logk halves for any k.** With per-fold systematics, the objective's halves
  are now the rescaled sums of the per-fold predictions (previously k>2 fell back to
  shared logk in the objective, §9j). For k=2 this is identical to the previous half
  logk.
- **Bug fix: additive (`normal`) systematics in the fold curvature.** A systematic
  shared across folds applied the FULL additive variation to every raw fold, so the
  folds summed to k× the variation and the U-statistic came out k× too large on the
  systematic–POI entries and k² on the systematic diagonal (4× and 16× at k = 4). The
  shared additive variation is now split evenly (1/k per raw fold; full for a
  rescaled half), in the fitter for shared logk and in the writer for the replicated
  entries of `hlogk_folds`. log_normal systematics were unaffected (the
  multiplicative factor is scale-free). Also fixed after the rebase: main now applies
  the `normal`-systematic `rnorm_init` factor to the contraction result instead of
  pre-scaling logk, so the fold path's own pre-scaling would have applied it twice;
  the sparse fold path did not apply it at all.
- **Sparse storage.** In sparse mode the fold templates are stored as `[k, nnz]`
  values on the sparsity pattern of the sparse `norm` (previously densified to
  `[k, nbinsfull, nproc]`), and the sparse split-logk deltas as `[k, nnz, nfolded]`;
  storage is k× the sparse nominal.
- **Continuous-M input/fit coupling.** `add_mc_stat_moment(M, names, *, variance)`
  now requires the denominator convention (`"bblite_inflated"` / `"bare"`, §9g);
  it is stored with the term, and the fitter raises if it does not match the fit's
  bin-by-bin stat setting (previously a warning only). The M penalty is applied only
  with `--mcStatDebias continuousM`; before, a file carrying M de-biased the point
  under any option.
- `--mcStatKfold` removed (unused since §9i); the "k is inferred from the fold axis"
  note moved into the `--mcStatDebias` help, together with the fold-assignment rule
  and the storage cost.
- The validation scripts became pytest tests with hard assertions
  (`tests/test_mcstat_debias.py`), so CI runs them.


### §9r. Robust minimization, goodness of fit, `--noHessian`, adjusted scans (rabbit)
Follow-up to §9q, prompted by end-to-end runs of `rabbit_fit.py`.

- **Blinded Poisson fits with linear POIs failed.** Data fits are blinded by
  default; the POI offsets (std 5) open the fit far from the minimum. Two causes,
  separated on a 60-bin, 2-process toy (k=2; far starts drawn like the blinding
  offsets, feasible ones only; bare scipy trust-krylov):

  | MC stat (events/fold/bin) | ordinary | ordinary, NaN→+∞ | de-biased | de-biased, NaN→+∞ |
  |---|---|---|---|---|
  | ~200   | 14/40 | 0/40 | 21/40 | 8/40 |
  | ~2000  | 18/41 | 0/41 | 18/41 | 6/41 |
  | ~20000 | 15/41 | 0/41 | 15/41 | 0/41 |

  1. *scipy and non-finite losses* (affects ordinary fits too): a trial step into
     negative yields gives a NaN loss; every comparison on the resulting NaN
     `ρ` is False, so the trust radius never shrinks, the same step is proposed
     until `--earlyStopping`, and a restart re-proposes it (the restart loop's
     loss was bit-identical before and after). The scipy loss now maps a
     non-finite value to `+∞` (the native minimizers already treat a non-finite
     `ρ` as `−∞`).
  2. *The Poisson jackknife is unbounded below*: `L_cf` contains `+½ y ln n_B`,
     which goes to `−∞` as a half prediction `n_B → 0⁺` while the full one stays
     positive. At the stalled point of a failing fit: a bin with `n = 65.5`,
     `n_A = 131`, `n_B ≈ 0`, gradient ~7.8e6 pointing into that boundary, most
     negative Hessian eigenvalue −2.6e11. The dangerous sliver narrows like
     `1/√N_MC` but never closes. Along the straight line from that point to the
     de-biased minimum `L_cf` decreases monotonically and is positive definite
     after the first tenth: the right minimum is the one in the basin of the
     ordinary MLE.

  Fix: `Fitter.minimize()` first minimizes the ordinary likelihood (the
  `--mcStatDebias none` objective) and starts the de-biased fit from its minimum.
  Through `minimize()` on the same toy, with both fixes: 0/40, 0/41, 0/41
  failures for the ordinary fit and for the de-biased fit; without the warm start
  the de-biased fit still fails 8/40 and 6/41 at the two lower MC statistics.
- **`--noHessian`**: the bread (de-biased; Gauss–Newton under `--covMode fisher`)
  and the sandwich block `c_iᵀHc_j`, `c_i = A⁻¹e_i`, from Hessian-vector
  products; identical to the dense covariance to 1e-11 or better in every mode
  (observed, fisher, continuous-M, fisher without de-biasing).
- **Adjusted scans** (`W_adj = W·dᵀΣ⁻¹d / dᵀC⁻¹d`): on a linear chisq toy the
  adjusted 1D statistic is `(d/σ_sand)²` at every point, the adjusted 2D statistic
  equals `dᵀΣ⁻¹d` to 2e-13, and the adjusted `--contourScan` interval is
  `±σ_sand` (0.020607; curvature σ 0.020390).
- **Goodness of fit at the de-biased minimum** (`Fitter.mcstat_gof`). The
  statistic is the ordinary one, `2·(reduced NLL)` of the de-biased objective
  *with* bin-by-bin stat at its minimum; only its reference changes (design doc
  §5a). With bin-by-bin stat its mean is far from the bin count: per bin
  `E[q_b] = s(1+ρ) − ρ` (`= 1 − ρ` for `s = 1/(1+ρ)`), because β absorbs the common
  MC fluctuation but not the halves' disagreement (`E[Δ_AΔ_B] = −σ²_MC`). A
  variant on the unprofiled predictions (mean ≈ nbins) was tried and dropped as
  inconsistent with the fit, which uses bin-by-bin stat.

  numpy emulation of the chisq fit with BB-lite (β profiled per bin on the full
  term and shared by the halves, 1500 toys): ρ≈1: mean `q` −1.69±0.40 (pred
  −2.23), var 235 (pred 237); ρ≈0.25: 42.85±0.28 (pred 43.00), var 119 (pred 124);
  without BB-lite, ρ≈1, 6000 toys: mean 55.7 (pred 58), var 592 (pred 601).

  rabbit toys (60 bins, 2 linear POIs, data AND templates fluctuated per toy,
  folds re-drawn; MC/data statistics ratio 1 unless noted):

  | config | toys | mean q (pred) | var q (pred) | p<0.05 | p<0.32 | mean p |
  |---|---|---|---|---|---|---|
  | chisq, no BB, k=2            | 180 | 58.8±2.0 (58.0)     | 697 (592)   | 0.067 | 0.356 | 0.50 |
  | Poisson, no BB, k=2          | 180 | 58.5±2.0 (58.0)     | 696 (592)   | 0.061 | 0.367 | 0.50 |
  | Poisson, no BB, k=2, MC ×4   | 120 | 59.8±1.3 (58.0)     | 203 (190)   | 0.067 | 0.375 | 0.47 |
  | chisq, BB-lite, k=2          | 180 | −0.9±1.2 (−2.2)     | 279 (237)   | 0.067 | 0.378 | 0.48 |
  | Poisson, BB-lite, k=2        | 150 | −1.5±1.4 (−2.1)     | 277 (236)   | 0.060 | 0.353 | 0.50 |
  | Poisson, BB-lite, k=3        | 120 | −3.5±1.5 (−2.1)     | 281 (252)   | 0.033 | 0.292 | 0.52 |
  | Poisson, BB-lite, k=2, MC ×4 | 120 | 44.8±1.0 (43.0)     | 132 (124)   | 0.083 | 0.383 | 0.46 |
  | Poisson, BB-lite, k=2, MC ÷4 |  60 | −181.5±5.6 (−187.5) | 1855 (2137) | 0.033 | 0.383 | 0.45 |

  Means agree within 1–2σ. The variances come out 5–18% above the prediction in
  every row, but that is one fluctuation, not a bias: all configurations reuse
  seeds 1–6 and draw the data first, so they share the same data fluctuations.
  A numpy replica of the toy generator reproduces rabbit's `q` toy by toy (to
  2e-13 without BB-lite; within 0.5 with it, from small differences in the
  BB-lite formulation) and gives, with many toys, var 578±8 (pred ~592, no BB,
  10000 toys) and 230±5 (pred 237, BB-lite, 4000 toys), against 697±74 and
  279±29 from the 180 rabbit toys. The p<0.05 rates, 3–8%, are consistent with
  5% at this sample size. The reference is a large-count approximation
  (second-order expansion of the data term, moment-matched shape); an exact
  alternative is a parametric bootstrap with Poisson data and resampled fold
  templates (not implemented).

### §9s. Projection tests, impacts, limits (rabbit)
- **Saturated projection tests** (design doc §5a): composite-likelihood ratio of
  the nested de-biased fits, `Σ_j μ_j χ²₁` with `μ = eig(Ã⁺B̃)` from the Schur
  complements at the null point. rabbit toys: 60 bins = 10 projected bins × 6
  bins of an integrated-out axis that separates the two processes, data and
  templates fluctuated; a common normalization is degenerate with the bin
  scales, leaving 9 testable directions:

  | config | toys | mean Δq (pred Σμ) | var (pred 2Σμ²) | p<0.05 | mean p | naive χ²(10) p<0.05 |
  |---|---|---|---|---|---|---|
  | Poisson, BB-lite, k=2 | 150 | 9.26±0.37 (9.00) | 20.1 (18.0) | 0.047 | 0.49 | 0.040 |
  | Poisson, BB-lite, k=3 | 150 | 9.35±0.37 (9.00) | 20.5 (18.0) | 0.067 | 0.48 | 0.040 |
  | chisq, BB-lite, k=2   | 150 | 9.29±0.37 (9.00) | 20.4 (18.0) | 0.053 | 0.48 | 0.040 |
  | Poisson, no BB, k=2   | 150 | 18.20±0.70 (18.00) | 74.2 (72.0) | 0.067 | 0.50 | 0.400 |
  | chisq, no BB, k=2     |  90 | 17.50±0.91 (18.06) | 74.1 (72.4) | 0.044 | 0.53 | 0.400 |

  With BB-lite μ ≈ 1 (the meat carries the MC variance). Without it the meat
  needs the data term's MC variance added explicitly (`Σ_b ρ_b J_bJ_bᵀ/V_b`);
  before that, the same no-BB toys gave mean 18.50 against a prediction of 9.00
  and p<0.05 = 0.52. With one scale per bin the projection is ill-posed for the
  de-biased objective: the scales absorb a change of process composition that
  the halves still feel, and the saturated fit ran to `+∞`. This is now detected
  from the data Gauss–Newton forms (relative curvature 0.049 / 0.011 with and
  without BB-lite, against ~1e-16 for the grouped projection) and the test is
  skipped.
- **Impacts**: the traditional and global groups add up to the reported
  (sandwich) uncertainty exactly (0.02924 = 0.02924 on the k=4 toy), with `mcStat`
  = √(binByBinStat² + mcStatDebias²); also for the histogram global impacts.
- **Linear χ²**: on an Asimov toy the de-biased fit gave 565 (ndf 60) against 33
  for the ordinary fit, because one residual direction (along the fitted
  parameters) had zero propagated variance; with the template-noise term 34. On
  fluctuated toys the linear χ² of *ordinary* fits is itself unreliable: it
  solves with a rank-deficient residual covariance (the fitted directions have
  no variance, `ndf_reduction = 0`), giving medians ~60 but 99th percentiles
  ~2·10⁴. That is a pre-existing issue, and the de-biased linear χ² (median ~70)
  could not be calibrated on top of it.
- **`rabbit_limit.py`**: on the k=4 toy the Gaussian σ goes 0.0282 → 0.0291
  (sandwich) and the likelihood contour level is divided by 0.969.

### §9t. Bin-by-bin stat variants, `--hvpBatch` (rabbit)
- **Variant matrix** (Poisson for all, chisq for a subset; two-half and k-fold
  fisher, on the small well-conditioned k = 4 toy, with the full-sample β then
  shared with the halves): every bin-by-bin stat type/mode/dynamic converges,
  bread and sandwich are positive definite, the Hessian-free covariance equals the
  dense one (~1e-14), and `--hvpBatch 1` reproduces the point exactly and the
  sandwich to ~1e-14. These are mechanical checks only; the β sharing itself is
  wrong beyond small ρ (§9u).
- **`--hvpBatch`** now assembles every dense Hessian of a de-biased fit from
  batched HVPs (`Fitter._scalar_hessian`): the sandwich meat, the Gauss–Newton
  shift, the MC-variance curvature without bin-by-bin stat, and the
  projection-test Gauss–Newton forms; they match `tape.jacobian` to 1e-10.
- **Continuous-M** failed (H − M indefinite) with multiplicative full mode and
  with `--binByBinStatDynamic`. The cause is not the denominator of `M` but the
  profile's own de-attenuation (§9u).
- The coverage, goodness-of-fit and projection-test toys (§9n–§9s) used no
  bin-by-bin stat and gamma-lite only.

### §9u. Bin-by-bin stat and the de-biasing: separate profiles, compensator (rabbit)
Notation: `ρ_b = σ²_MC,b / D_b`, the MC variance of the prediction over the data
variance (≈ N_data/N_MC for unit weights; ≈ 1 for MC with the data's luminosity,
the slides toy). `V_b = D_b + σ²_MC,b` is the per-bin variance of a bin-by-bin
stat profile. Toys: the near-degenerate harness toy (`tests/mcstat_coverage.py`,
true strengths (1.3, 0.7), degenerate difference 0.42), chisq unless noted;
weighted MC (weight w) gives ρ ≈ w.

**Three defects, found by toys that measure point bias and coverage for every
bin-by-bin stat mode across ρ** (earlier validation never did this for the fold
methods with bin-by-bin stat: §9g checked convergence on a small-ρ toy, §9n–§9o
measured coverage only for continuous-M, §9r–§9s validated test-statistic null
distributions, §9t was mechanical; failures seen in §9g ("spurious far minima")
and §9o (23% of toys filtered as rank-deficient) were attributed to degeneracy).

1. *Shared β over-subtracts* (fold methods, every bin-by-bin stat type). With
   the full-sample β applied to the halves, a Gaussian term becomes (k = 2,
   normal-additive) `e²/(2V) − Σ_hΔ_h²/(4D)`: the noise floor is removed with
   weight `1/D` instead of `1/V`, a factor `1 + ρ` too much. Direct check at
   ρ ≈ 1 (degenerate-direction curvature): shared β 11.98 (bread indefinite),
   separately profiled terms 55.89, true-template target 55.96.
2. *Missing compensator* (multiplicative full mode, `--binByBinStatDynamic`).
   With per-process β the profile `r²/(2V(θ))`, `V = D + Σ_p u_p(θ)²`, is the
   errors-in-variables profile: its gradient `−r²∂V/(2V²)` cancels the
   attenuation when `E[r²] = V`, so the ordinary full-mode fit is already
   unbiased (+0.006 at ρ ≈ 1, against −0.27 for gamma-lite). De-biasing on top
   removes the noise floor twice. Lite multiplicative β scales only the total
   yield (a `1/kstat` effect); static normal-additive has no θ-dependence.
   Exact-template fits (no template noise, `sumw2` claimed = what the jackknife
   converges to) show the size, bias of the degenerate difference:

   | ρ | 0.03 | 0.1 | 0.3 | 1.0 |
   |---|---|---|---|---|
   | full / dynamic, uncompensated | +0.014 | +0.071 | +0.270 | +1.358 |
   | gamma-lite, no bin-by-bin stat (ρ = 1) | | | | −0.009 |

3. *Fold sandwich without bin-by-bin stat*: the reported meat is the plain data
   Hessian, without the template-noise variance (the `Σρ JJᵀ/V` term is used only
   by the projection test). Unbiased, but coverage 0.69 / 0.60 / 0.57 at
   ρ = 0.03 / 0.1 / 0.3 (σ/scatter 0.97 / 0.87 / 0.81). Not yet fixed.

ρ scan of the shared-β fold code (twoHalf, 100 toys per point; bias of the
degenerate difference; the ordinary gamma-lite fit's attenuation for scale):

| ρ | attenuation | gamma-lite | normal-additive lite | full, compensated (wt 1) | full, uncompensated | no bin-by-bin stat |
|---|---|---|---|---|---|---|
| 0.03 | −0.020 | −0.002 | +0.033 | −0.015 | +0.002 | +0.006 |
| 0.1 | −0.052 | +0.020 | +0.004 | +0.022 | +0.104 | +0.008 |
| 0.3 | −0.152 | +0.089 | +0.134 | +0.111 | 82% of toys fail | +0.017 |

(uncertainties ±0.013–0.028). Effect 1 is relevant from ρ between 0.1 and 0.3,
effect 2 from ρ between 0.03 and 0.1. Thresholds are for this extremely
degenerate toy; both scale with ρ × (noise floor / information) along the
weakest direction.

**Redesign** (design doc §1b, §2b, §2c, §5a):
- Fold methods: every jackknife term is its own profiled likelihood at the
  full-sample kstat (per-half kstat would jackknife the bin-by-bin stat variance
  itself away), each with its own constraint term and, with `profile=False`, its
  own stored β.
- Compensator where the bin-by-bin stat variance moves with the parameters:
  continuous-M (and every meat) `½Σ log(V/V_ref)`, i.e. the marginal instead of
  the profiled bin-by-bin stat likelihood; fold objective `½Σ D(1/V_ref − 1/V)`,
  exact for `E[r²] = D` at any θ (an estimating-equation correction, not a
  likelihood).
- `--covMode fisher`: the Gauss–Newton form of a profiled term on the raw
  predictions with weight `1/(D + σ²_bb)`, replacing the profiled terms and the
  compensator together (keeps the U-statistic identity exact).
- GOF reference: with separate profiles the `Δ_AΔ_B` term carries the shrink `s`:
  `G` mean `−Σ sρ`, variance `Σ s²[2ρ² + (1+ρ)ρ(a−b)²/(ab)]`; `q` excludes the
  compensator.

**Validation of the redesigned fold methods** (separate profiles + compensator;
100 toys per point; bias of the degenerate difference, truth 0.42; coverage
target 0.683, ±0.05):

| ρ | method | bin-by-bin stat | mean bias | median bias | rms | median σ | coverage |
|---|---|---|---|---|---|---|---|
| 0.1 | twoHalf | gamma-lite | +0.033 ± 0.017 | +0.059 | 0.171 | 0.155 | 0.60 |
| 0.1 | twoHalf | gamma-lite dynamic | −0.009 ± 0.017 | −0.024 | 0.172 | 0.153 | 0.62 |
| 0.1 | twoHalf | normal-additive lite | −0.018 ± 0.016 | −0.025 | 0.161 | 0.154 | 0.72 |
| 0.1 | twoHalf | normal-mult. full | −0.004 ± 0.014 | −0.002 | 0.140 | 0.153 | 0.72 |
| 0.3 | twoHalf | gamma-lite | +0.016 ± 0.020 | +0.039 | 0.204 | 0.189 | 0.65 |
| 0.3 | twoHalf | gamma-lite dynamic | −0.011 ± 0.021 | −0.029 | 0.207 | 0.194 | 0.63 |
| 0.3 | twoHalf | normal-additive lite | −0.007 ± 0.022 | −0.013 | 0.222 | 0.189 | 0.62 |
| 0.3 | twoHalf | normal-mult. full | +0.024 ± 0.019 | +0.046 | 0.187 | 0.185 | 0.67 |
| 0.3 | 2-fold, fisher | gamma-lite | +0.000 ± 0.023 | +0.008 | 0.232 | 0.190 | 0.63 |
| 0.3 | 2-fold, fisher | normal-mult. full | +0.011 ± 0.023 | +0.029 | 0.231 | 0.191 | 0.59 |
| 1.0 | twoHalf | gamma-lite | +0.112 ± 0.068 | −0.044 | 0.677 | 0.319 | 0.50 |
| 1.0 | twoHalf | gamma-lite dynamic | +0.040 ± 0.047 | −0.021 | 0.473 | 0.295 | 0.57 |
| 1.0 | twoHalf | normal-additive lite | +0.027 ± 0.044 | −0.025 | 0.438 | 0.303 | 0.62 |
| 1.0 | twoHalf | normal-mult. full (1 fit failed) | +0.057 ± 0.045 | +0.011 | 0.444 | 0.310 | 0.58 |
| 1.0 | 2-fold, fisher | gamma-lite | +0.041 ± 0.043 | +0.002 | 0.433 | 0.325 | 0.60 |
| 1.0 | 2-fold, fisher | normal-mult. full | −0.031 ± 0.038 | −0.003 | 0.375 | 0.323 | 0.68 |

The shared-β biases are gone (ρ = 0.3: +0.09 to +0.13 before, now within
±0.025, every mode), and nothing fails (1 of 1600 fits). Coverage is close to
the target up to ρ = 0.3 (0.59–0.72, a little low on average). At ρ = 1 the
medians are unbiased but the point has heavy tails along the degenerate
direction (rms 0.38–0.68 against σ 0.30–0.33), and coverage drops to
0.50–0.68; continuous-M covers 0.64–0.77 on the same toy. The likely cause is
the meat: the full-sample curvature lacks the variance of the jackknife's own
split-noise term (the `Δ_h` part of the cross-fit score), which is second order in
the noise floor and matters only where the floor is comparable to the
information. A cross-half score-variance meat (outlook of the original study)
would address it; not done.

**Continuous-M with the compensator** (log form's first-order version; 200 toys
each, ρ ≈ 1; the bias of the ordinary gamma-lite fit is −0.27):

| bin-by-bin stat | coverage (diff) | median bias | rms | median σ |
|---|---|---|---|---|
| gamma-lite | 0.635 ± 0.034 | +0.050 | 0.368 | 0.310 |
| normal-additive full | 0.715 ± 0.032 | +0.029 | 0.338 | 0.316 |
| normal-multiplicative full | 0.740 ± 0.031 | −0.001 | 0.352 | 0.330 |
| gamma full | 0.773 ± 0.030 | +0.020 | 0.313 | 0.332 |
| gamma-lite dynamic | 0.705 ± 0.032 | −0.036 | 0.351 | 0.327 |
| normal-additive dynamic | 0.755 ± 0.030 | +0.013 | 0.330 | 0.324 |
| Poisson, normal-multiplicative full | 0.720 ± 0.032 | +0.002 | 0.317 | 0.319 |
| Poisson, gamma-lite dynamic | 0.730 ± 0.031 | +0.041 | 0.340 | 0.342 |

**Asimov fits** (one template set; Asimov = expected yields from the noisy
templates; 30 data toys with the same templates for comparison):
- The Asimov point is shifted by any de-biasing: Asimov data carry no attenuation,
  so the correction overshoots (continuous-M lite: 1.13 against 0.42, data
  0.47; twoHalf without bin-by-bin stat at ρ ≈ 1: 1.28, data 0.55). With
  continuous-M in full mode on unit-weight per-process scores `M_BB = M` and the
  Asimov point is the truth.
- The Asimov σ equals the median σ of real data with a fixed bin-by-bin stat
  variance (0.299 / 0.299, 0.231 / 0.230), but is too small where it moves with
  the parameters (continuous-M full mode 0.147 against 0.330; the ordinary
  full-mode fit 0.114 against 0.194): the `r²`-dependent curvature terms vanish
  at Asimov. Expected uncertainties need the expected-information covariance at
  θ₀ or data-only toys with the fixed templates.
- Implemented (design doc §5a): for an Asimov dataset of a de-biased fit the
  objective is recentred at the generating point by a linear term (Hessian
  unchanged) and the covariance uses the Gauss–Newton forms, which equal the
  real-data ones at the same point (test: to 1e-6). Fits without de-biasing are
  unchanged. (`rabbit_fit.py` does not minimize on Asimov, so its point was
  already the generating one; the recentring matters for scans, contours,
  limits and the EDM.)
