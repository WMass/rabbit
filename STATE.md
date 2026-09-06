# material-resolution — state (2026-09-05)

Branch **`material-resolution`** off `global-term-card` (worktree
`/work/submit/david_w/ZMass/rabbit-material`), 3 commits, nothing pushed.

Goal: replace the four ad-hoc per-family resolution scales of the CVH unbinned
mass term (`k_hit`, `k_ms`, `k_ioni`, `k_rad`) by the PHYSICAL parameters the
fit already carries — the 42 parmtype-15 material-group amounts and a set of
per-hit-class Gaussian shares — so that one fit floats one set of parameters
across the quadratic hit-chi2 term, the mass-term width, and the mass-term
mean.

---

## The parameterisation, as implemented

The CVH propagator applies the parmtype-15 parameter `k_g` to the AMOUNT of
material of every Geant4 step:

| where | code | effect |
|---|---|---|
| mean energy loss | `G4ErrorEnergyLossForCVH::AlongStepDoIt` | `xifact = exp(dxieff)`, `dxieff = <leg offset> + k_g`; `dE = xifact (dE/dx)_0 ds` |
| step MS covariance | `Geant4ePropagator::PropagateError` (M1 block) | `errMSIout *= exp(dms) * exp(k_g)` |
| step ionization variance | same | `errMSIout(0,0) *= exp(dioni) * exp(k_g)` |

Every step-level log-CF exponent of the offline resolution model is **linear in
the step's material amount at fixed composition** (Moliere `chi_c^2 ∝ x` and
`Omega_0 ∝ x` at fixed `chi_a`; Urban `a_1, a_2, a_3 ∝ x`; radiative mean
emissions `∝ x`; delta-recoil `xi ∝ x`), and each of `ms_step_exponent`,
`ioni_step_exponent`, `rad_exponent`, `delta_step_exponent` is a plain sum over
step rows.

So with the fit's influence weights **held fixed** — the two-step convention
the in-fit CGF block already uses ("the fit never differentiates through the
weights", `Geant4ePropagator.cc` M1 comment; `w = sqrt(v_pool/sq2)/sigma` is a
property of the CONVERGED fit) —

```
    S_f(tau; k) = S_f^fix(tau) + sum_g A(k_g) S_{f,g}(tau)
    A(k) = exp(k)     "exp"     (the C++ matStepFact convention)
    A(k) = 1 + k      "linear"  (its first-order form)
```

is EXACT for an amount change at fixed composition, and `k = 0` reproduces the
production's own exponents. The Gaussian hit share is split the same way but
per HIT CLASS,

```
    v_i(eps) = v_other,i + sum_c H(eps_c) v_{c,i} ,   Re S += -0.5 v_i tau^2
    H(eps) = 1 + eps  (default)  or  exp(eps)
```

with `v_{c,i}` the summed exported influence variance `resinfvarv` of the
candidate's parmtype-8/9 blocks of class `c` in units of `sigma_i^2`, and
`v_other` the Gaussian remainder (beamspot / vertex constraint) that no
parameter scales. **`v_other` is always present, with or without a floating
class** — in the flat `MassCFTerm` it rides as the `gauss` family and it is the
dominant part of the mass CF.

**Field and alignment enter the mass term only through the MEAN**, via the
existing sparse `D` rows `m_i(theta) = m_i^0 + D_i theta`. The material
parameters enter all three places at once.

Offline validation of the decomposition (`matres/extract_groups.py --validate`,
`max |sum_g S_g - S_flat|` against `max |S_ms| ~ 23`): `ms` 3.6e-14, `io_re`
5.6e-16, `rad_re` 8.7e-18 — float64 round-off, as the linearity argument
requires.

---

## Commits

### `e81bba3` — `MaterialCFTerm`

1. **`_chunk_resolution` factored out of `MassCFTerm._chunk_li`.** That method
   mixed the PARAMETERISATION of the exponent with the inverse-Fourier
   quadrature that turns it into a density; only the first ever changes.
   `_extra_param_names()` is the hook a subclass uses to add its parameters to
   `param_names`.
2. **`MaterialCFTerm`.** Per-group exponents block-sparse (a J/psi candidate
   touches ~22 of the 42 groups): CSR `grp_ptr` over candidates, `grp_id` per
   row, `(nnz, nt)` `re`/`im` arrays per family, contracted in-graph with
   `tf.math.unsorted_segment_sum`; optional `fix_re`/`fix_im` `(n, nt)`
   baselines carry the pruned groups at weight 1. `group_params` MUST be the
   names the quadratic external term uses (`material_<group>` from
   `make_global_term.name_params`), so a joint fit floats one set;
   `group_units` carries the card's whitening in (`k_phys = value * units`).
   Legacy per-family knobs still work, behind the card's `--legacy-families`.

### `2029a6d` — hooks for a later per-candidate transform

Three no-op hooks, so a later subclass can change WHAT is evaluated without
touching the quadrature that evaluates it:

* `_chunk_exponent_scale(values, ci)` — optional `(nchunk,)` multiplier of the
  resolution exponent. Reserved for effects that scale a candidate's whole
  process-noise exponent by something other than the material amounts: the
  block variances are evaluated at the FITTED state, so the fitted
  per-candidate sigma depends on the candidate's own fluctuation
  (`sigma_obs = sigma_bar (1 + a x)`).
* `_chunk_residual(values, ci)` — the residual fed to the integral. Default is
  the linear `m_i^0 - m_ref - shift - (D theta)_i` that was inline; a subclass
  may return any per-candidate function of the parameters, in particular a
  nonlinear transform to a truth-referenced variable.
* `_chunk_logjac(values, ci)` — the matching `log |d(residual)/d(observable)|`,
  added to `log L` in `_mix`, because a nonlinear transform changes the measure.

All three return `None` / the previous expression, so `MassCFTerm` is
unchanged: tests 1, 2, 4 and 5 report a **bit-identical** NLL.

### `d051c54` — hook for a parameter-dependent per-candidate sigma

`_chunk_sigma(values, ci)`; `None` (the default) means the stored, constant
`sigma`, which is the cheap and bit-identical path.  `Jpsi_sigmamass` is the
FIT's own error, assembled at the converged state, so it is a function of the
very fluctuation the likelihood is measuring — treating it as known constant
fits a density whose width is correlated with its residual.  When the hook is
overridden, `s_i` enters the `1/(pi s_i)` prefactor, `t_abs = tgrid/s_i` AND
the kernel CF argument, so `_interp_phik` re-reads the tabulated `phi_K` in
graph (regular grid → gather + lerp, differentiable in `t`, the same numbers
`np.interp` gives).  The exponents are functions of the standardized `t` and
stay untouched.  The correction itself is another agent's
(`calibration_studies/resolution/oddmoment/MASSCFTERM_SPEC.md`) and is NOT
implemented here.

### `ca6f732` — the Gaussian remainder, and two finiteness guards

`hit_share` is honoured whenever it is given, even with zero floating classes
(it was previously gated on `hit_params`, which silently removed the Gaussian
term from every two-track card and made the NLL `-inf`). Plus `amount_clip` /
`hit_clip` (default `k in [-5, 5]`, `e^5` = 148x a group's material): a
trust-region step early in a fit can throw `k` to O(100), `exp(k)` is `+inf`,
the exponent `-inf`, its `exp()` 0, and the gradient `inf*0 = NaN`.
`tf.clip_by_value` has a well-defined subgradient, so the minimizer sees a flat
region and steps back. A fit that ENDS on the clip is telling you something
else is wrong.

---

## Tests — `tests/test_material_cf.py`, ALL EIGHT PASS

| # | check | result |
|---|---|---|
| 1 | reduction to `MassCFTerm` at `k = eps = 0`, many groups | **rel 0.0** |
| 1b | ... one group per candidate (identical summation order) | **bit-identical** |
| 2 | legacy families only == `MassCFTerm` | **bit-identical** |
| 3 | analytic gradient vs central FD in `k_g` and `eps_c`, both amount modes | max rel **4e-9** over 16 parameters |
| 4 | `group_units` == a rescaled parameter | **bit-identical** |
| 5 | pruned group == group frozen at 0 | **rel 0.0** (float32 baseline 3.4e-11) |
| 6 | injection of 5 % more material in one group | `A(k_1)` ratio **1.045765** vs 1.05 injected (**-0.40 %**), identical in both amount modes; the other two groups move by <6e-4 against a 0.11 stat sigma |
| 7 | degeneracy, two collinear groups | eigenvalue 4.3e-7 vs 268 (**1.6e-9**), softest direction exactly `(-0.707, +0.707, 0, 0)` = `k_0 - k_1`; independent case rank 4/4, cond 7.4 |
| 8 | HDF5 round trip through a datacard | **bit-identical NLL** |

Test 6 measures a SHIFT: the injected and un-injected toys share their
standard-normal draw, so what is left after the difference is the estimator's
own non-linearity, not a fluctuation. The amount factor `A(k)` is the
mode-independent quantity and both modes agree on it to the last digit, which
is what "exp and linear are two parameterisations of one likelihood" means.

Runtime ~12 min at `OMP_NUM_THREADS=8`; the Hessians are finite differences of
the analytic gradient because `GradientTape.jacobian` goes through `pfor`,
which retraces the segment-sum graph on every call.

---

## The joint fit on the J/psi gun (real data, 2026-09-05)

Card: the quadratic hit-chi2 term over 299,069 candidates of
`resolution_trackres_jpsigun_ul16_260905d_m0` (`chi2/ndof < 3`,
`hessmax < 1e8`, `gradmax < 1e6`, whitened) + a `MaterialCFTerm` over 24,000
candidates of the same production, on the SAME 92 parameters (50 field modes +
42 material groups) with the sparse D rows. Truth is zero (MC, ideal geometry).
`trust-krylov`, edm 9.4e-11, 2467 s.

| | quadratic only | joint | ratio |
|---|---|---|---|
| **`bfield_mode0`** = the momentum scale | 0.0339 T of RMS \|dB\| = **dB/B 8.9e-3** | 0.00025 T = **dB/B 6.6e-5** | **0.0074 (136x)** |
| `material_tib_support` sigma(k) | 0.01543 | 0.01203 | 0.780 |
| `material_tec_structure` | 0.02929 | 0.02421 | 0.827 |
| `material_bpix_support` | 0.04477 | 0.03806 | 0.850 |
| `material_tob_support` | 0.01982 | 0.01769 | 0.893 |
| `material_tec_services` | 0.06818 | 0.06716 | 0.985 |
| every ACTIVE-silicon group | 0.0199 (its prior) | 0.0199 | 1.000 |

The hit chi2 measures the material and is nearly blind to the overall scale;
the masses measure the scale and add 2-22 % on the material. `bfield_mode0`
moves from +1.36 sigma to -0.30 sigma, i.e. onto zero, as it must on MC.
A mass-only fit of the same 24k candidates gives dB/B 1.02e-4 on the scale --
**87x better than the hit chi2 on 299k candidates** -- and 1.0-2.1x WORSE than
it on every material group, so the joint fit beats both everywhere.

**Injection through both terms**, `k = ln(1.05)` (5 % more material) into
`material_tib_support`, applied to the quadratic gradient (`G -> G - K dtheta`),
the mass mean (via the D rows) and the mass-term exponents of that group:
recovered at **pull -0.505** vs truth, shift 88.7 % of the injection
(+4.42 % material against +5.00 %), with the residual absorbed by the
correlated `bpix_support` (-0.37 sigma).  Leakage onto the other 91 parameters:
**rms 0.057 sigma**, and the momentum scale moves 0.31 sigma.

---

## Open

* the per-hit-class parameters have never been FITTED: the two-track maker
  exports no parmtype-8/9 resolution blocks, so a ditrack candidate's
  `reseigidx` has none and `vgf` is one scalar. Implemented, gradient-checked
  and round-tripped; not measured. The C++ fix is to port the four
  `resfamily_.push_back(8)/(9)` pushes from
  `ResidualGlobalCorrectionMakerG4e.cc:3454/3476` into the two-track hit loop.
* the analysis side lives in `calibration_studies/resolution/matres/`
  (`extract_groups.py`, `make_material_card.py`, `analyze_groups.py`,
  `report_fit.py`, `cmp_scale.py`, `run_joint.sh`).
* `--legacy-families` exists; the "physical vs four-knob" comparison fit has
  not been run.
