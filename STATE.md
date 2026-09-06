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

### `2047b07` — the self-consistent-resolution correction (MASSCFTERM_SPEC)

`sigma_i` as exported is the FIT's own error at the converged state, so it is a
monotone function of the fitted mass — of the very fluctuation the likelihood is
measuring.  `s_i(theta) = max(sigma_i - a_i delta_i(theta), 0.2 sigma_i)`
recovers the unconditional resolution from observed quantities alone and enters
the `1/(pi s)` prefactor, `t_abs = tgrid/s` and `phi_K(t_abs)`; the exponents
are untouched; the `-ln s(theta)` log-Jacobian rides in the prefactor.
`a_res` is the new per-candidate input, `self_consistent_sigma=True` the
default, `False` reproduces the old behaviour off the same card, and an
absent/all-zero `a_res` takes the static path — exact, not a shortcut, since
`a = 0` makes `s == sigma` identically.  Plus `_chunk_mean_shift`, a no-op hook
for the second-order Jensen term.  The spec, derivation and toy gates are
another agent's
(`calibration_studies/resolution/oddmoment/MASSCFTERM_SPEC.md`); only the
implementation is here.

---

## Tests — `tests/test_material_cf.py`, ALL NINE PASS

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
| 9a | G1: `a_res` None / all-zero / non-zero with the switch off | all three **bit-identical** to the pre-spec NLL |
| 9b | the DYNAMIC sigma path at `a = 1e-300` vs the static one | **rel 0.0** (`_interp_phik` reproduces `np.interp` exactly) |
| 9c | `d(NLL)/d(alpha)` vs central FD with `s(alpha)` in all three places | rel **5.4e-9** |
| 9d | the formula and its sign | `s == max(sigma - a delta, 0.2 sigma)` exactly, `corr(delta, s - sigma) = -1` |

**The gate on the real gun** (two cards over the same 24k J/psi-gun candidates,
identical but for `self_consistent_sigma`, with `alpha` the ONLY parameter that
moves the mean — `--no-jac`, so the 50 field modes cannot absorb the shift):

| | alpha [1e-3] |
|---|---|
| naive | -0.026120 ± 0.058680 |
| corrected | +0.121160 ± 0.058677 |
| **shift** | **+0.147280** against the spec's **+0.146 ± 0.01** |

A first attempt on a card that ALSO floated the field modes gave +0.0121, and
that was the setup being wrong rather than the term: `alpha` is nearly
degenerate with `bfield_mode0`, which took +0.068e-3 of the shift. A gate on a
scale parameter has to be run where that parameter is the only scale.

`a_i` has a median of 0.011527 on this sample — exactly the spec's 0.011 — and
a maximum of 1.8e5, because `Jpsi_sigmamass` itself reaches 2.7e5 GeV on 26 of
24,000 candidates. The card builder clips `|a_i|` at `--max-ares` (0.5); the
real conclusion is that the mass-term selection wants a `Jpsi_sigmamass` cut,
which none of `chi2/ndof`, `hessmax` or `gradmax` supplies.

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

---

# (merged in) STATE of the z-lineshape-kernel branch

# Z/gamma* lineshape kernel — branch `z-lineshape-kernel`

Worktree: `/work/submit/david_w/ZMass/rabbit-zlineshape` (branched off
`unbinned-mass-term` @ `8edb2c2`). **Do not work in
`/work/submit/david_w/ZMass/rabbit`** — another agent owns `global-term-card`
there.

Goal: a Z→μμ channel fitted for `m_Z` and `Γ_Z` with the same machinery as the
J/ψ channel, i.e. a physics kernel supplying the characteristic function of the
Z/γ* lineshape to `rabbit.unbinned.MassCFTerm`.

**Status: complete and all seven tests pass.** FSR and the acceptance are now
*inside* the provider and closed at generator level against the real sample (see
the 2026-09-05 section at the end); what is left is not this kernel but the rest
of a data channel (background, CVH resolution, EW nuisances) plus the one thing
the generator-level study added: the card must float a smooth `K(m)`.

---

## Commits

| commit | what |
|---|---|
| `af6ab5e` | `make_lumi_table.py` + the shipped NNPDF3.1 table |
| `de1edcd` | `zgamma.py`, `lineshapes/__init__.py`, `TabulatedLineshapeKernel` wiring, `unbinned.declare_params`, package-data, first cut of the tests |
| `0218bd6` | test threshold fix, graceful skip in test 6, first STATE.md |
| `bb85bef` | default `nm = 32768`, `f.param_model` fix in test 6 |
| `4cf512e` | full test results in STATE.md |
| `93f74f4`…`3275383` | truncated likelihood (`norm_window`), Fourier-space `Z`, `upsample` |
| (this one) | **`terms`, `acceptance`, `fsr` in the provider** — see below |

## Files

```
rabbit/lineshapes/__init__.py                        provider registry, make_provider()
rabbit/lineshapes/zgamma.py                          ZGammaLineshape (the provider)
rabbit/lineshapes/make_lumi_table.py                 LHAPDF generator for the table
rabbit/lineshapes/data/zlumi_nnpdf31_nnlo_13tev.npz  the shipped table (15 kiB)
rabbit/unbinned.py                                   TabulatedLineshapeKernel + declare_params
tests/test_zgamma_kernel.py                          the 6 tests
pyproject.toml                                       package-data for the npz
```

## API

```python
from rabbit import unbinned
from rabbit.lineshapes import ZGammaLineshape

z = ZGammaLineshape(m_ref=91.1876, window=(50., 130.))      # see defaults below
term = unbinned.MassCFTerm(
    "z", sigma=sigma, mobs=mass - 91.1876, tgrid=np.linspace(0, 8, 256),
    families=[{"name": "res", "param": "k_res", "kind": "gauss"}],
    vgf=vgf, m_ref=91.1876,
    kernel=unbinned.TabulatedLineshapeKernel(provider=z),   # param_names from z
)
decl = unbinned.declare_params(term, {
    **z.param_declarations(gz_prior=2.3),        # m_Z, Gamma_Z as POIs (MeV)
    "k_res": (1.0, np.nan, 1.0, 0),
})
writer.add_unbinned_term("z", term.config(), term.param_names,
                         {"sigma": ..., "mobs": ..., "tgrid": ..., "vgf": ...},
                         **decl)
```

Provider surface: `dsigma_dm(values|mz,gz,sin2, in_pb)`, `pdf(values)`,
`cf_tab(values)` / `cf_tab_ext(values)`, `log_cf(values, t_abs)` (= `__call__`,
the `PhysicsKernel` contract), `density_from_cf(values, m)` (diagnostic),
`param_declarations(...)`, `check_tau_range(tgrid, sigma)`, `config()` /
`from_config(cfg)`.

Constructor knobs: `m_ref`, `window`, `nm`, `nfft`, `tau_max`, `lumi`,
`width_scheme`, `mz_param`/`gz_param`/`sin2_param`, `mz_ref`/`gz_ref`/`sin2`,
`mz_unit`/`gz_unit`/`sin2_unit`, `terms`, `acceptance`, `fsr`, `fsr_mmax`,
`dtype`. New surface: `born_pdf(values)` (Born x acceptance on the extended
grid `m_born`) and `fold_fsr(y)` (Born grid -> output grid).

Defaults: `nm=32768` (dm = 2.44 MeV), `nfft=16*nm` (dtau = 4.91e-3 GeV^-1),
`tau_max=40` GeV^-1, `window=(50,130)`, `m_ref=91.1876`,
`width_scheme="fixed"`, `lumi="nnpdf31_nnlo_13tev"`,
`mz_unit=gz_unit=1e-3` (parameters fitted in MeV as offsets from `mz_ref`,
`gz_ref`, starting at 0).

## Physics and provenance

Born-level neutral-current Drell-Yan mass spectrum, ported verbatim from
`hard_me` in
`/work/submit/david_w/ZMass/calibration_studies/lineshape/zwidth_sensitivity.py`
(built on `drell_yan_xsec.py` / `constants.py` in the same directory):

```
dsigma/dm = 2 m C sum_f L_f(m^2) [ e_f^2/(2 m^4)
            + sum_{g in gL,gR} ( Re[chi] (1-4s2)/(4 s2 (1-s2)) e_f g
                               + |chi|^2 (1+(1-4s2)^2)/(32 s2^2 (1-s2)^2) g^2 ) ]
|chi|^2 = 1/((m^2-m_Z^2)^2 + (m_Z Gamma_Z)^2)     "fixed"   (default)
          1/((m^2-m_Z^2)^2 + (m^2 Gamma_Z/m_Z)^2) "running"
Re[chi] = (1 - m_Z^2/m^2) |chi|^2
g_L = I3_f - e_f s2,  g_R = -e_f s2,  C = 4 pi alpha^2/(3 N_c)
```

* EW scheme: Gmu. `G_F = 1.1663787e-5`, `m_W = 79.906853549493746`,
  `m_Z(fixed) = 91.153509740726733`, `Γ_Z(fixed) = 2.4932`,
  `sin^2 = 1 - m_W^2/m_Z^2 = 0.23152`, `alpha(m_Z) = sqrt(2) G_F m_W^2 sin^2/pi
  = 1/128.83`. DYTurbo CT18Z `.in` values, copied from `constants.py`.
  `sin^2` does **not** track a fitted `m_Z` (fixed input, optionally a fitted
  nuisance via `sin2_param`).
* Width convention: the two schemes use different mass parameters,
  `m_running = m_fixed sqrt(1+(Gamma/m)^2)` (+34 MeV at the Z). `mz_ref`/`gz_ref`
  follow the chosen scheme; a fitted `m_Z` must be quoted in that scheme.
* Parton luminosity: LO, flavours d,u,s,c,b, `mu_F = Q`, NNPDF3.1 NNLO
  `as=0.118` member 0, sqrt(s) = 13 TeV, 300 log-spaced anchors 40–200 GeV,
  **inclusive in rapidity** (`--y-cut` exists, off by default). Generated by
  `make_lumi_table.py`, which imports
  `drell_yan_xsec.integrate_sigma_hat_prime_sm` so the table is bit-identical
  to the reference study's.
* **Not included**: FSR (that is `MassCFTerm`'s separate `phi_K`), acceptance,
  EW loop corrections, running alpha/sin^2, QCD beyond what the PDF absorbs.

## Numerics

pdf = spectrum truncated to `window`, renormalised, sampled on `nm` uniform
points, represented by its piecewise-linear interpolant with the outermost node
on each side forced to 0 (so the truncation edge is a one-bin ramp, ~2.4 MeV,
and the represented pdf is exactly normalised: `phi(0) = 1` to round-off).
The hat-basis CF is analytic:

```
phi(tau) = dm K(tau dm) e^{i tau (m_lo - m_ref)} conj(rfft(pad(p))[j])
K(x) = (sin(x/2)/(x/2))^2,   tau_j = 2 pi j / (nfft dm)
```

then 4-point Lagrange onto the per-candidate `t/sigma`, returning
`(log|phi|, arg phi)`. The tabulation carries an extra node at `tau = -dtau`
(free: `phi(-tau) = conj phi(tau)`) so the stencil reaches `tau = 0`.

Padding buys tau resolution only — there is no convolution, so no aliasing; the
transform is exact at every `tau_j`. `dm = W/nm` and `dtau = 2 pi/(nfft dm)`, so
with `nfft` given as a multiple of `nm` the two error scales are independent
(`dtau = 2 pi/(16 W)` regardless of `nm`).

**The one real bug found**: clamping the interpolation stencil at `tau = dtau`
instead of reaching `tau = 0` left an O(dtau^2) error on the most heavily
weighted point of the inverse transform — smeared density error 1.5e-3 instead
of 1.5e-8, a factor 1e5.

## Measured accuracy (defaults, window 50–130 GeV)

| quantity | value |
|---|---|
| `dsigma_dm` vs reference `hard_me`, same luminosities | 2.82e-16 rel |
| shipped luminosity table vs reference cache, 60–130 GeV | max 3.98e-4, median 2.15e-7 rel |
| `phi(0)` | 1.000000000000000 + 0i |
| `cf_tab` vs independent cell-by-cell Filon quadrature | 3.41e-13 abs |
| smeared density vs exact Gaussian convolution, max abs / peak | 1.5e-8 / 1.5e-8 / 2.2e-8 (sigma 1.0 / 1.5 / 2.5 GeV) |
| … same, max **relative** (at m = 70 GeV, ~0.5 % of peak) | 1.5e-6 |
| mass-grid representation, nm=32768 vs 131072, max abs / peak | 1.8e-6 (was 3.4e-6 at nm=16384) |
| gradient vs central FD, `m_Z` / `Gamma_Z` | 2.8e-9 / 7.8e-9 rel |

tau-interpolation convergence is clean O(dtau^4):
9.17e-6 → 5.81e-7 → 3.63e-8 → 1.91e-9 for `nfft` = 8/16/32/64 × `nm`.

## Test results (all PASS)

```
python tests/test_zgamma_kernel.py
```

| test | result |
|---|---|
| 1 lineshape vs reference | PASS — ME identical to 2.8e-16 |
| 2 characteristic function | PASS — CF exact to 3.4e-13, smeared density 2e-8 of peak |
| 3 gradients vs finite differences | PASS — 3e-9 / 8e-9, Hessian positive definite |
| 4 toy closure, 200k | PASS — pulls +0.03 / +0.79 / +0.90 |
| 5 Breit-Wigner-only kernel | PASS — mass biased by −43.2 MeV (7 sigma_stat) |
| 6 datacard round trip | PASS — 2 POIs, NLL 6e-17, prior weight exact |
| 7 terms / acceptance / FSR fold | PASS — all six checks at float64 round-off |

### Test 4 — 200k toy closure

200k candidates from the lineshape at `m_Z = 91.153510 + 30 MeV`,
`Γ_Z = 2.4932 − 40 MeV`, smeared with per-candidate Gaussian
`sigma ∈ [1, 2] GeV`. No FSR, no background, no acceptance, and **no cut on
m_obs** — the model is then exactly normalised, so no truncation term is needed
(a real window would need the truncated likelihood, as
`tests/test_unbinned_mass.py` test 4 does for the Breit-Wigner).

| parameter | fitted | error | truth | pull |
|---|---|---|---|---|
| `k_res` (variance scale) | 1.0004 | 0.0139 | 1.0 | +0.03 |
| `m_Z` [MeV] | 34.96 | 6.28 | 30.0 | +0.79 |
| `Gamma_Z` [MeV] | −26.73 | 14.70 | −40.0 | +0.90 |

7 trust-exact iterations, 1569 s, `|grad|_inf = 1.4e-5`.

`sigma(Gamma_Z)` against the naive expectations:

| | MeV | ratio |
|---|---|---|
| unsmeared Breit-Wigner limit, `Gamma sqrt(2/N)` | 7.88 | 1.00 |
| this model, `m_Z` and `k_res` fixed | 10.82 | 1.37 |
| profiling `m_Z` and `k_res` | 14.70 | 1.86 |

So a 1–2 GeV Gaussian resolution costs 1.37x on the width, and floating the
resolution scale a further 1.36x. The cost is entirely
`rho(Gamma_Z, k_res) = −0.677`; `rho(Gamma_Z, m_Z) = +0.015` and
`rho(m_Z, k_res) = −0.016` are negligible, i.e. the mass is not degraded by
either.

### Test 5 — the same toy with a Breit-Wigner-only kernel

| parameter | fitted | error |
|---|---|---|
| `k_res` | 0.9634 | 0.0142 |
| `dm_bw` [MeV] | −47.31 | 6.29 |
| `Gamma_bw` [MeV] | 2544.6 | 15.2 |

The Breit-Wigner peak lands at 91.1403 GeV against a generated
`m_Z = 91.1835 GeV`: **−43.2 MeV, 7 sigma_stat**, and the width comes out
+91.4 MeV too large. For scale, the truncated lineshape has mean 90.342 GeV
(0.84 GeV below `m_Z`) but mode 91.1853 GeV (+1.8 MeV above it): the gamma*
tail, the interference and the falling parton luminosity all pull weight to low
mass, and a symmetric Breit-Wigner splits the difference between mode and mean.
That −43 MeV is the size of the effect this provider exists to remove.

## Timings (16 threads, wmassdev image)

| | |
|---|---|
| build a provider | ~2 s (mostly the luminosity spline) |
| one CF table (nm=32768, nfft=2^19) | ~133 ms |
| 200k-candidate `MassCFTerm.nll`, nt=256 | ~2 s |
| same, value + grad + Hessian (3 params) | ~220 s, ~88 GB RSS |
| test 4 fit (200k, 7 iterations) | 1569 s |
| test 5 fit (200k, Breit-Wigner kernel, 8 iterations) | 189 s |

The Z kernel costs ~9x the Breit-Wigner kernel per iteration. That is the
per-candidate path, not the FFT: four `tf.gather`s of the tau interpolation plus
their scatter-add adjoints over `n*nt = 5e7` indices per parameter.

## How to run

```bash
IMG=/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling:latest
export APPTAINERENV_PYTHONPATH=/work/submit/david_w/ZMass/rabbit-zlineshape:/work/submit/david_w/WRemnants_dev/wums:/work/submit/david_w/ZMass/calibration_studies/env_tf/pypath
export APPTAINERENV_TF_CPP_MIN_LOG_LEVEL=2 APPTAINERENV_OMP_NUM_THREADS=16
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
cd /work/submit/david_w/ZMass/rabbit-zlineshape
singularity exec -B /work/submit,/home/submit,/scratch/submit,/tmp "$IMG" \
    python -u tests/test_zgamma_kernel.py                # all six, ~40 min
singularity exec -B /work/submit,/home/submit,/scratch/submit,/tmp "$IMG" \
    python -u tests/test_zgamma_kernel.py --skip 4 5     # fast, ~8 min
```

The `/work/submit/david_w/WRemnants_dev/wums` entry is **required** for test 6:
the wums shim in `calibration_studies/env_tf/pypath` has no `sparse_hist`, so
`rabbit.tensorwriter` cannot import without it. `/ceph/submit` is not mountable
from this account, so the stock `calibration_studies/env_tf/run_tf.sh` fails;
a working copy without that bind is in the session scratchpad at
`.../scratchpad/zlineshape/run_tf.sh`.

Regenerating the luminosity table (only to change PDF set / sqrt(s) / mass
range / rapidity cut — the shipped table is committed):

```bash
cd /work/submit/david_w/ZMass/calibration_studies && source setup_env.sh
python /work/submit/david_w/ZMass/rabbit-zlineshape/rabbit/lineshapes/make_lumi_table.py \
    --m-lo 40 --m-hi 200 --n-anchor 300 --tag nnpdf31_nnlo_13tev
```

## Optimisations not pursued

* The kernel returns `(log|phi|, arg phi)` and `MassCFTerm` immediately does
  `exp(log|phi|) cos(arg phi + ...)` — a `log`/`atan2`/`exp` round trip on
  `n*nt` elements. An optional `(re, im)` kernel contract would drop four
  transcendentals per element. That is a change to `MassCFTerm`, so it was left
  alone.
* The Hessian memory (88 GB at n=200k, nt=256) is the autodiff tape over all
  chunks. `trust-krylov` with HVPs avoids materialising it.
* `tau_max=40` is generous: a Z term with `sigma >= 1 GeV` and `tmax = 8` needs
  only 8 GeV^-1. Lowering it shrinks the tabulation but not the gather cost.

## Follow-up: the truncated likelihood (`norm_window`), commit `93f74f4`

The last section's *"a mass window on `m_obs`* needs the truncated-likelihood
normalisation" is now done, in `MassCFTerm` itself rather than around it.

`norm_window=(lo, hi)` divides the density by its own integral over the window,
`L_i -> L_i / Z_i`, which is the correct likelihood for a sample *selected* in
that window. `Z` is evaluated on a mass grid for a handful of resolution
*classes* (`norm={"sigma", "vgf", "class", "families"}`, `norm_nodes` grid
points) and gathered per candidate; the datasets round-trip through the
datacard as `norm_sigma` / `norm_vgf` / `norm_class` / `S_<c>_<f>_norm`.

Why a mass grid and not the Fourier identity
`Z = (1/pi) Int Im[phi(u)(e^{-iu d_lo} - e^{-iu d_hi})]/u du`: that integrand
oscillates at the window *half-width* (~30 GeV for a Z), needing a `t` grid an
order of magnitude finer than the density itself does -- and the family
exponents only exist on the term's own `tgrid`.

`_chunk_li` was split into a reusable `_density()` so the normalisation runs
the identical model; the per-candidate path is untouched.

`tests/test_unbinned_norm.py`, all five pass:

| test | result |
|---|---|
| 1 Gaussian `Z` vs the error function | PASS -- 2.9e-7 at 1025 nodes (the O(h^2) mass quadrature), `Z` in 0.60-0.99 on a +-2 GeV window |
| 2 convergence in `norm_nodes` | PASS -- 257 vs 4097 nodes agree to 2.9e-7 relative on a Voigt over 60-120 GeV |
| 3 cost of the class approximation | PASS -- exact to 2e-16 at 16 classes for a +-30 GeV window; 1.6e-2 at 32 classes when the edge sits 1-4 sigma away |
| 4 vs the hand-rolled `- sum log L + n log Z` | PASS -- NLL 2e-16, gradient 1.6e-14 |
| 5 closure on a truncated Voigt | PASS -- see below |

Test 5 is the reason this exists. 400k Voigt candidates cut to 60-120 GeV,
only **2.67 %** outside:

| | `k_res` | `Gamma` [MeV] | pull on `Gamma` |
|---|---|---|---|
| with `norm_window` | 1.00620 +- 0.01356 | 2483.79 +- 11.44 | **-0.8** |
| without | 1.45564 +- 0.01422 | 2014.65 +- 9.82 | **-48.7** |

i.e. a 2.7 % truncation, ignored, costs 479 MeV on the width and 46 % on the
resolution scale.

## Follow-up 2: the quadrature. Two bugs, one cause. `0940e43`, `a5a1df3`

Both were found by pointing the machinery at the real Z smoke (459 candidates
of `dymc_8p5M_260905`) rather than at a toy.

**The density is an inverse Fourier transform, and its integrand oscillates
`|m_obs - m_pred| / sigma` times across `tgrid`.** For a J/psi in a +-0.35 GeV
window that is a few periods. For a Z in a 60-120 GeV window it reaches 61,
against the 64 points the in-maker exports (`tau=stride4of448<=8`).

1. **`_norm_z` was sampling the density on a mass grid.** The window edge is
   30 GeV from the model centre, so every one of those samples was in the
   unresolved regime. It showed up as `Z` between 0.75 and **2.8** -- an
   integral of a density over a sub-interval, larger than one. Replaced by the
   Gil-Pelaez form, which needs the same fine `t` grid but only a `(K, nt)`
   tensor rather than `(K, n_mass, nt)`, so it can afford to be fine
   (`norm_tpoints`, default 8192). `Z` on the smoke is now in [0.748, 0.967]:
   3.4 % mean leakage out of the window, 25 % for the worst-resolved pairs.

2. **The candidate density itself is under-resolved on 64 points.** Rebuilding
   the same term on a 16x finer grid moves the NLL by **-33.7** over 449
   candidates and individual densities by up to 270 %; the fitted `m_Z` moves
   by **-29.3 MeV**, which is 20 sigma at 3.9 M candidates. `MassCFTerm` now
   takes `upsample=N` and expands the tabulated exponents with a fixed
   cubic-spline matrix **inside the graph** -- the exponents are smooth in tau
   (largest second difference <2 % of the range), so the expansion is faithful,
   and keeping it in the graph leaves the datacard at 64 points where a
   16x-finer stored array would be a 79 GB card at 3.9 M candidates.
   4x and 16x agree to 0.4 MeV on `m_Z`, so 4x is already converged.

`tests/test_unbinned_norm.py` (six tests, all pass):

| test | result |
|---|---|
| 1 Gaussian `Z` vs the error function | PASS -- 2.9e-7 on a +-2 GeV window where `Z` is 0.60-0.99 |
| 2 convergence in `norm_tpoints` | PASS -- 8192 vs 32768 agree to 1.2e-6 |
| 3 cost of the class approximation | PASS -- exact to 4e-16 on a +-30 GeV window |
| 4 vs the hand-rolled `- sum log L + n log Z` | PASS -- 6.8e-7 / 9.1e-6, both routes quadrature-limited there |
| 5 closure on a truncated Voigt | PASS -- pull -0.2 with the term, -34.5 without |
| 6 in-graph upsampling | PASS -- 1.2e-11 vs a pre-splined term; density converges 7.4e-2 -> 2.4e-4 from 2x to 32x |

### Z channel status

The channel itself lives in
`/work/submit/david_w/ZMass/calibration_studies/zchannel` (FSR kernel from the
MiniAOD gen record, datacard builder, fit driver, systematics scan, README).
On the 459-candidate smoke, resolution scales fixed:

    m_Z     = -107.7 +- 142.0 MeV      (truth 0 in the fixed-width scheme)
    Gamma_Z = +332.5 +- 293.5 MeV

projecting to **sigma(m_Z) = 1.45 MeV, sigma(Gamma_Z) = 2.76 MeV** at 3.9 M
candidates -- against 1.42 / 3.33 MeV from test 4's 200k toy scaled by sqrt(N).
`rho(m_Z, Gamma_Z) = -0.01`. Floating the four resolution scales *freely*
leaves the information indefinite at 449 candidates: the Z alone does not
determine them. Constrained at 1e-2 or 1e-3 -- which is what the J/psi channel
supplies -- the errors are unchanged from holding them fixed, so profiling the
resolution costs the Z nothing *given* an external constraint.

## What a data Z channel still needs

* **FSR kernel** — `phi_K` tabulated from the generator
  (POWHEG+MiNNLO+Photos), as the J/psi channel does. The provider is pre-FSR by
  construction; the reference study's `FSRKernel` (LL collinear radiator in
  `log m^2`) is the natural starting point.
* **Acceptance `A(m)`** — lepton pT/eta cuts make the observed spectrum
  `A(m) dsigma/dm`; the provider integrates the lepton angular distribution out
  and has no angular information. The generator's `--y-cut` covers only boson
  rapidity.
* **Background** — `UniformBackground` / `BernsteinBackground` already exist in
  `MassCFTerm`; a Z channel needs a real shape and a floating fraction.
* **Per-candidate resolution from CVH on Z tracks** — the `families` / `sigma` /
  `vgf` arrays, i.e. the same export path as the J/psi channel. Test 4 shows
  the width measurement is limited by `rho(Gamma_Z, k_res) = −0.68`, so the
  resolution model fidelity is the thing that matters most.
* **Theory nuisances** — PDF (regenerate the table per replica/eigenvector and
  add a luminosity-shape nuisance), EW corrections, the running-vs-fixed width
  convention, running alpha and sin^2.
* ~~A mass window on `m_obs` needs the truncated-likelihood normalisation.~~
  Done -- `norm_window`, see the section above.
* `m_Z` and the momentum-scale parameter `alpha` are exactly degenerate in a
  single-resonance fit — a Z channel measures `m_Z` only jointly with the
  J/psi (or another) channel that pins the scale.


---

# 2026-09-05 — the generator's own parameters, and the FSR fold moved into the provider

Everything below is measured at **generator level** on 29.3 M events
(N_eff 19.9 M) of `DYJetsToMuMu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos`
UL16 MiniAODv2 — the sample the detector-level Z channel will be fitted on.
Driver, kernels and figures: `calibration_studies/zchannel/`
(`fit_gen.py`, `README.md` §"Generator-level closure"), figures in
`~/public_html/cvh/260905_zgen/`.

## 1. The width convention is settled, and the provider already had it right

The gridpack's `powheg.input` sets **no** EW inputs, so running its own
`pwhg_main` for one initialisation prints what POWHEG used. It reads the PDG
(running-width) values and converts them to the **constant-width** scheme,
unconditionally — there is no `runningwidth` flag in this process:

| | POWHEG | provider | difference |
|---|---|---|---|
| `m_Z` (used) | 91.153509740726733 | `MZ_FIXED` | 0 |
| `Gamma_Z` (used) | 2.4932018986110700 | `GZ_FIXED` = 2.4932 | −1.9 keV |
| `m_W` | 79.906853549493746 | `MW` | 0 |
| `sin^2` | 0.23153999447822571 | `1 − MW²/MZ_FIXED²` | 1e-16 |
| `1/alpha(m_Z)` | 128.82531590804655 | derived | 0 |
| `G_F` | 1.1663787e-5 | same | 0 |
| lumi PDF | `lhaid 306000` | `NNPDF31_nnlo_as_0118` mem 0 | same set |
| `m_ll` range | `min_Z_mass 50` | window lower edge 50 | same |

So `width_scheme="fixed"` and every EW constant of `ZGammaLineshape` **is** the
generator's. Fitting in either convention against its own reference agrees to
0.2 MeV; the 34 MeV is entirely in the reference value.

## 2. What the provider was missing, and now has

* **`terms=("gamma","int","z")`** — select the matrix-element pieces.
* **`acceptance=`** — a smooth multiplicative `A(m)` (Bernstein or a grid)
  applied to the Born spectrum *before* the FSR fold.
* **`fsr=`** — the **multiplicative** FSR fold,
  `p_post(m) = sum_j w_j p_born(m/r_j)/r_j`, on a Born grid extended above the
  window (capped by `fsr_mmax`, default the luminosity table's edge). Optional
  per-atom `m_lo`/`m_hi` bands make it piecewise constant in `m_pre`.
  Everything downstream — `pdf`, `cf_tab`, `MassCFTerm` — then models the
  **post**-FSR mass as a function of the POIs alone, which is what the earlier
  note asked for (`dm/m_pre` is nearly `m_pre`-independent while `dm` is not, so
  `MassCFTerm`'s additive `phi_K` was an approximation).
  The fold is one constant `(nm, n_born)` matrix built in `__init__`; a
  gather-based version was 30x slower once the kernel had a few thousand atoms
  (its backward pass is a scatter-add over `nm x n_atoms`).

## 3. Closure (window 60–120 GeV, all errors are the weighted sandwich)

| model | Δ`m_Z` [MeV] | Δ`Gamma_Z` [MeV] |
|---|---|---|
| pre-FSR, 2 parameters | −2.47 ± 0.40 | **+75.83 ± 0.85** |
| pre-FSR + 5-term smooth `K(m)` | −0.45 ± 0.50 | +1.14 ± 0.97 |
| post-FSR, no fold | −227.1 ± 0.48 | +741.3 ± 1.19 |
| post-FSR, no fold, + `K(m)` | −30.7 ± 0.56 | +317.0 ± 1.19 |
| **post-FSR, folded, + `K(m)`** | **+0.15 ± 0.56** | **+1.31 ± 1.14** |
| fiducial (pT>25, \|eta\|<2.4), folded + `A(m)` + `K(m)` | **−0.31 ± 0.87** | **+2.61 ± 1.75** |

## 4. The one thing the provider does *not* model: the NNLO K-factor

The hard ME and the parton luminosity are both LO, and the sample is MiNNLO.
The generated/model ratio runs 1.7 (55 GeV) → 1.0 (peak) → 1.2 (150 GeV), which
is why the 2-parameter fit puts +76 MeV on the width. It is **not** a luminosity
choice — over NNPDF3.1 NNLO / its replica 1 / NNPDF3.1 LO / CT18 NNLO and
mu_F in [Q/2, 2Q] the bias moves by only ±3 MeV (`m_Z`) and ±4 MeV (`Gamma_Z`),
about 5 % of the effect — and it is **not degenerate with the POIs**: floating
5 Legendre terms restores the inputs, costs 1.25x / 1.21x on the errors, has
every rho(POI, c_k) below 0.40, and makes the fitted POIs agree across all five
luminosities to 0.25 MeV / 0.03 MeV.

**Therefore the detector-level Z card must float a smooth `K(m)`.** Without one
the kernel is wrong by 76 MeV on the width — 50x its projected statistical
error.

## 5. FSR-kernel systematics that carry over to data

* **Quadrature.** The atoms are a midpoint rule, so the bias goes as the square
  of the in-group spread of `u = -ln r`: 1e-2 → +152, 3.3e-3 → +29,
  1e-3 → +3.9, 3.3e-4 → +1.3, 1e-4 → +0.7 MeV on `Gamma_Z`. `build_kernel`'s
  default is now 3.3e-4 (~3200 atoms).
* **`m_pre`-dependence.** Inclusively the kernel is multiplicative to a few per
  cent (⟨u⟩ 26.2e-3 → 28.8e-3 from 60–80 to 110–150 GeV) and banding moves the
  fit by 0.3 / 0.7 MeV. Under a **tight fiducial cut it is not**: ⟨u⟩ runs
  7.1e-3 → 14.9e-3, a factor two, and using the inclusive kernel on the
  fiducial sample costs 3.2 MeV on `m_Z`. Bands are mandatory there.
* **Kernel statistics.** Half-sample split: 0.4 MeV on `Gamma_Z` at 15 M gen
  events, i.e. negligible.
* **Numerics.** `nm` 4096/8192/16384 and a Born grid capped at 160 or 200 GeV
  all agree to 0.06 MeV.

## 6. What this leaves for the detector-level card

1. a floating `K(m)` (5 Legendre terms, or the same idea as a Bernstein);
2. `fsr=` a banded kernel built with the **analysis** selection, `sigma_cap`
   3.3e-4 or finer;
3. `acceptance=` the reco acceptance (at gen level `A(m)` is degenerate with
   `K(m)` — dropping it changed the fit by 0.3 / 0.9 MeV — so it matters only
   inasmuch as it is *not* smooth);
4. everything above is *in addition to* the resolution, the background and the
   `m_Z`–`alpha` degeneracy already listed.

All seven tests re-run and pass against the final code
(`tests/test_zgamma_kernel.py`; test 4's 200 k toy is unchanged at pulls
+0.03 / +0.79 / +0.90, and test 7's six checks are at float64 round-off).
