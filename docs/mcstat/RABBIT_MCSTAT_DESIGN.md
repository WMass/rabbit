# Implementing MC-stat de-biasing in `rabbit`: continuous-M and two-half

Interfaces for the cross-half / continuous-M de-biasing (see `RESULTS.md`
§3,§5–§8) inside `wmass/rabbit`. Goal: de-biased POI central value **and** a
covering covariance, with autograd-friendly objectives that fit rabbit's existing
TF-NLL + SciPy design and its matrix-free HVP.

This started as the design proposal; it has been updated to describe what is
implemented (`rabbit/fitter.py`, `rabbit/tensorwriter.py`, `rabbit/inputdata.py`,
`bin/rabbit_fit.py`). The validation history is in `RESULTS.md` §9; the checks are
`tests/test_mcstat_debias.py`.

## 0. How rabbit maps onto the math

| math (RESULTS.md) | rabbit object |
|---|---|
| per-bin prediction `n_b(θ)` | `nexp = _compute_yields_noBBB()[0]` (= `Σ_proc rnorm·exp(θ·logk)·norm`) |
| per-(bin,proc) template `M_{ib}` | `indata.sumw` / `indata.norm` |
| noise-floor moment `Q_{i,b}=Σ_e w_e² s_i s_j` (diagonal) | `indata.sumw2` [nbins, nproc] — **already stored** |
| per-bin NLL `ℓ_b(n)` | `_compute_ln(nexp)` (chisq / covariance / Poisson) |
| BB-lite σ_MC (β nuisances) | `BinByBinStat` (`bbstat`) |
| Fisher / curvature | TF autograd Hessian of `_compute_nll`; matrix-free HVP `loss_val_grad_hessp` |
| de-biased objective | extra term(s) in `_compute_nll_components` |
| final covariance | post-fit, in `cov` / `_cov_stat_floating` |

rabbit minimizes a single scalar `_compute_nll()` and gets grad/Hessian/HVP by
autodiff — exactly the "single autograd scalar" setting. So both methods inject
at the **objective** level and let autodiff produce the de-biased gradient/Hessian.

Fit options (in `Fitter.__init__` / CLI), **three orthogonal axes**:
```
--mcStatDebias {none, continuousM, twoHalf, kfold}   # de-bias the point/curvature; default none
--mcStatDebiasCov {curvature, sandwich, dataPropagated}  # default sandwich; dataPropagated: continuousM only
--covMode {observed, fisher}                         # curvature primitive; applies EVERYWHERE
```
`twoHalf` and `kfold` are the same method: the number of folds `k` is inferred from
the fold axis of the input (§2a), so there is no fold-count option.
`--covMode` is orthogonal and applies uniformly to **all** curvatures — the
standard inverse-Hessian covariance *and* both the bread and meat of the de-biased
sandwich. `observed` = the autograd Hessian `∇²L` (data-conditional; Efron–Hinkley);
`fisher` = the Gauss–Newton/expected information `JᵀD J` (`J=∂n/∂θ` JVP, `D=diag(ℓ'')`;
lower-variance, cheaper, finite-MC-cleaner). Both are valid and consistent in every
case — the only rule is to use the **same** mode for bread and meat (and the
standard cov), so the sandwich is a coherent robust covariance and `H−A=M` holds
cleanly (exactly in `fisher` mode; with matching residual·∂²μ terms in `observed`).
Compose with existing `--binByBinStat*` (BB-lite supplies σ_MC; the debias supplies
the curvature de-bias + coverage). Keep BB-lite ON.

---

## 1. Continuous-M

### 1a. Supply a single frozen `M_ij` matrix to the writer
The per-process yields are usually a *small* subset of the parameters (most are
systematics/POIs from a response model), so the `sumw2` block alone is not enough;
the general per-event score `s_i(e)=∂lnμ/∂θ_i` lives only in the event loop. It is
**sufficient to support only the frozen `M_ij` form** — a single `nparams²` matrix
accumulated at the expansion point in the event loop and stored once:
```python
# tensorwriter.py
def add_mc_stat_moment(self, M, param_names, *, variance, name="mcstat_M"):
    # M[i,j] = Σ_b (Σ_{e∈b} w_e² s_i s_j)/V_b at nominal
    # variance: "bb_inflated" (V_b = μ_b + Σ_proc sumw2_b) | "bare" (V_b = μ_b)
```
It is stored as an external quadratic term (`hess = -M`) with the `variance`
convention recorded next to it. The fitter consumes it directly as a constant, and
only with `--mcStatDebias continuousM` (otherwise the term is ignored, so the file
does not de-bias a fit that did not ask for it). Because the correct denominator
depends on the fit configuration (§1b), the fitter **raises** if the recorded
convention does not match its bin-by-bin stat setting. **Frozen is enough** because:
(i) the only θ-dependence of `M(θ)=Σ_b Q_b/μ_b(θ)` is the `1/μ_b` weight, and `M`
is itself the leading `O(1/N_MC)` bias correction, so its θ-variation is a
negligible correction-to-a-correction `O(1/N_MC · dlnμ/dθ · Δθ)`;
(ii) the stop-gradient curvature penalty (§1b) already treats `M` as frozen at the
reference each Newton step. No per-bin `Q` (`nparams²·nbins`) and no `1/μ(θ)`
reconstruction at fit time. (`sumw2` gives the diagonal process-norm block as a
free special case if no `M` is supplied.) `M` is symmetric → store the
`nparams(nparams+1)/2` upper triangle; sparse/low-rank if structured.
### 1b. De-biasing the objective (frozen `M`, in the gradient)
`self.M` is the supplied frozen `nparams²` matrix (loaded once; `tf.constant`).
Add to `_compute_nll_components` (RESULTS.md §4b/§7d):
```python
def _compute_mcstat_penalty(self):
    if self.mcStatDebias != "continuousM":
        return None
    th = self.get_x()                                   # full parameter vector
    return -0.5 * tf.reduce_sum(th * tf.linalg.matvec(self.M, th))   # -½ θᵀMθ
```
- `-½ θᵀMθ` (M in the gradient) gives the **de-biased point** `θ̂ = (H − M)⁻¹ g`;
  autodiff produces the de-biased gradient and the Hessian `H − M` automatically.
- **LINEARITY REQUIREMENT (verified in rabbit, RESULTS §9a).** This `θ̂=(H−M)⁻¹g`
  result is linear-Gaussian: it holds only when the de-biased parameters enter the
  prediction `n(θ)` **linearly** (`H` constant). If the map is nonlinear, the data
  Fisher `F(θ)∝(∂n/∂θ)²` grows as the `-Mθ` gradient shifts the minimum and
  *cancels* the `-M` subtraction — no de-bias of point or covariance. rabbit's
  **default POI transform is `rnorm=x²`** (`allowNegativeParam=False`), which is
  nonlinear and defeats continuous-M; the de-biased POIs must use a **linear**
  parametrization (`--allowNegativeParam`, or a linear param-model). Free-floating
  normalizations (the §3 motivation) are linear in `rnorm`, so this is just a flag.
  For genuinely nonlinear parameters use **two-half/k-fold (§2)**, which de-biases
  via the data-independent `norm^A·norm^B` cross structure regardless of nonlinearity.
  The fitter should warn/error if `continuousM` is requested for nonlinear params.
- `M` is constant → the penalty is a plain quadratic form, `O(nparams²)` per eval
  (`O(nparams)` matrix-free if `M` is supplied factored). No bin sums, no
  reconstruction. If BB-lite is on, the writer should compute `M` with the BB-lite
  variance `μ_b+Σsumw2` in the denominator so the curvature lands at σ₂ (RESULTS.md §7j).
  With the bare variance under BB-lite, `M` is ~2× too large and `H − M` goes
  indefinite (RESULTS.md §9g); hence the recorded `variance` convention (§1a).
  Storing a configuration-independent quantity instead would need the per-bin
  moments `Q_b` (`nparams²·nbins`), which is exactly what the frozen `M` avoids.
  (`"bblite_inflated"`, the name used before full mode and dynamic bin-by-bin
  stat were supported, is read as `"bb_inflated"`.)
- **What bin-by-bin stat already removes.** `M` is the moment of the true
  template noise. When the variance of the profiled β moves with the parameters,
  the profiled likelihood already removes part of it. With
  `V_b(θ) = D_b + Σ_c u_bc(θ)²`, where `u_bc` is the noise standard deviation β
  component `c` models, the profiled per-bin term is `r_b²/(2V_b(θ))`. Its
  gradient `−r²∂V/(2V²)` has mean `−∂V/(2V)` (with noisy templates
  `E[r²] = V`), which de-attenuates on its own: with per-process β (full mode,
  dynamic) this is the errors-in-variables profile, which removes the noise
  floor of the per-process templates, `M_BB = Σ_bΣ_c ∂u_bc∂u_bcᵀ/V_b` (for linear
  parameters; with unit-weight events and per-process scores, all of `M`).
  Penalizing `M` on top removes it twice: `H − M` went indefinite with
  multiplicative full mode and dynamic bin-by-bin stat (RESULTS.md §9t).
  The fitter adds the **compensator** `½Σ_b log(V_b(θ)/V_b,ref)`
  (`Fitter._compute_mcstat_bb_compensator`), whose gradient `∂V/(2V)` cancels
  the profile's term exactly. In the Gaussian case `½ log V` is exactly the
  difference between integrating β out and profiling it, so continuous-M
  minimizes the **marginal** bin-by-bin stat likelihood less `½θᵀMθ`:

      −log L_marginal(θ) − ½ θᵀMθ,   −log L_marginal = −log L_profile + ½ Σ_b log V_b(θ)

  The marginal likelihood is attenuated, and the M penalty removes exactly that.
  `u` per configuration (`_mcstat_bb_noise_scales`):
  - multiplicative types in full mode, and `--binByBinStatDynamic` (any type):
    one component per (bin, process), `u = √sumw2 · n_p(θ)/sumw`;
  - multiplicative types in lite mode: one component per bin, scaling the yield
    of the processes with MC variance (the noise of the total yield only, a
    relative `1/kstat` correction along the well-measured directions);
  - normal-additive, static (lite or full): fixed absolute shifts, no term.

  `D_b` and `V_b,ref = D_b + Σ_proc sumw2_b` are frozen at the reference (the data
  variance for chisq, the prefit expectation for Poisson), as for `M`. The meat
  is the marginal likelihood's curvature, `A + M` (§1c).

### 1c. Covariance (sandwich) — RESULTS.md §8c
The minimized-objective Hessian `A = ∇²L = H_bb − M` is the de-biased *curvature*
(σ₂) and **undercovers**. Report the sandwich:
```python
def cov_mcstat_sandwich(self):
    A = self.hess(...)                 # ∇² of the de-biased objective (bread)
    H = self.hess_full()               # ∇² of the SAME-SAMPLE NLL (no M penalty) = meat
    Ainv = tf.linalg.inv(A)
    return Ainv @ H @ Ainv             # = A⁻¹ + A⁻¹ M A⁻¹ ; H−A = M
```
With bin-by-bin stat whose variance moves with the parameters, `A` includes the
compensator's curvature (§1b) and the meat is that of the marginal likelihood,
again `A + M`.
`hess_full` = the existing `loss_val_grad_hess` evaluated with the M-penalty
disabled (rabbit already toggles BBB off for `impacts_parms`, line ~934 — reuse
that pattern). Matrix-free variant: `Σ·v = A⁻¹ H (A⁻¹ v)` via two HVP solves of
the existing `loss_val_grad_hessp`. Analytic meat (calibration-free, RESULTS.md
§8c); use data-propagated meat only if you accept the calibration factor.

---

## 2. Two-half  (general: any POI / systematic / param-model parameter)

The robust route. Needs only first moments (half-sample derivative templates),
`O(nparams·nbins)`, and handles arbitrary parameters because the half-templates
carry the full response structure (RESULTS.md §7e/§8d/§8f).

### 2a. Input: `TensorWriter` fold interface

rabbit's `TensorWriter` consumes **filled histograms**, not events — so it cannot
split a process itself. The user (who has the events) fills histograms with an
extra **fold axis** and passes them; the writer slices that axis into folds and
derives the full by projecting it out. Folding is **opt-in per call** via a
`fold_axis` argument (default `None` = no folding); the number of folds `k` is
**inferred from the size of that axis** — no writer-level fold count. `k=2` is the
two-half case; `k>2` enables k-fold variance reduction (re-group into halves
multiple ways at fit time, RESULTS.md §7e).

**Fold-assignment rule (user's responsibility, documented):** when filling, assign
each *generated* event to a fold by something independent of the fitted
observables and constant across oversampled copies — e.g. `fold = hash(gen_event_id)
% k` — added as the fold-axis fill value. All oversampled/smeared copies of one
generated event must land in the **same** fold, so the folds are statistically
independent at the generated-event level (RESULTS.md/MD §5). The writer can't
verify this; the fold axis is named by the user (any categorical/int axis).

**`add_process(h, name, ..., fold_axis=None)` — folds sliced from a named axis;
full DERIVED (never provided twice).** Since `h = Σ_f h[fold]`, the writer projects
the fold axis out for the full template and `sumw2` (`= Σ_f sumw2_fold`, still
needed by BB-lite), and slices it for the folds:
```python
def add_process(self, h, name, channel="ch0", signal=False, variances=None,
                fold_axis=None):
    if fold_axis is not None:
        k = len(h.axes[fold_axis])                       # k inferred from the axis
        folds = [h[{fold_axis: f}] for f in range(k)]    # per-fold slices
        h     = h[{fold_axis: sum}]                      # project out -> full (rest unchanged)
        self.dict_sumw_folds[channel][name] = stack([flat_dense(hf) for hf in folds])
        self.fold_k[channel][name] = k                   # record per-process k for the fitter
    # ... existing full-template / sumw2 handling on h ...
```
`fold_axis=None` (default) → ordinary unfolded process (existing behaviour
unchanged). A process passed with `fold_axis=None` while *other* processes are
folded is **MC-stat-EXEMPT** (data-driven/analytic): stored as `norm_full/k` in every
fold, identical across folds, so it drops out of the de-bias (correct — no MC
noise). All folded processes must share a common `k` (any `k ≥ 2`). The writer
never asks for, nor checks, a separately-supplied full against the folds.

**`add_systematic(..., fold_axis=None)` — same convention; split only if it shares
the MC.** Pass `up_hist`/`down_hist` carrying the (named) `fold_axis` to split the
variation per fold (full up/down derived by projecting it out). Omit `fold_axis`
(default) → the usual axis-free up/down, logk **shared** across folds — the common
case (smooth / analytic / independent-sample systematics; the nominal fold split
already de-biases the dominant nominal-template noise via `norm^A·norm^B`).
Sharing logk is the recommended default: the dominant MC-stat effect is the
nominal-template (and its derivative) noise; with shared logk, `∂n^{A/B}/∂θ =
logk · norm^{A/B}`, so the cross product `norm^A·norm^B` already de-biases the
nominal noise. Split logk only for systematics whose up/down histograms reuse the
nominal MC sample.

**On `write()`** — extra HDF5 datasets (attribute `mcstat_fold_k` = `k`):
- `hnorm_folds`: dense mode `[k, nbinsfull, nproc]`; sparse mode `[k, nnz]`, the
  fold values on the sparsity pattern of the sparse `hnorm_sparse` (so the folds
  keep the sparsity of the nominal templates; a fold entry where the full template
  is exactly zero is dropped with a warning);
- split logk (systematics added *with* a `fold_axis`; else logk is shared): dense
  `hlogk_folds` `[k, nbinsfull, nproc, (2,) nsyst]`, where shared entries replicate
  the full logk — unchanged for `log_normal`, `1/k` of it for the additive `normal`
  form so the folds sum to the full variation; sparse `hlogk_folds_delta`
  `[k, nnz, nfolded]`, the per-fold delta against the full logk over the folded
  systematics only (`log_normal`, symmetric), plus `hmcstat_folded_syst_idx`;
- the existing full `hsumw`/`hnorm`/`hlogk` unchanged (`= Σ_f folds`).
`inputdata.py` loads `indata.norm_folds`, `indata.logk_folds`,
`indata.logk_folds_delta` (`None` ⇒ unavailable). Storage: `k×` the nominal
templates (`k×nnz` in sparse mode); logk only ×k for the systematics actually
folded.

**Fitter consumption (how halves are formed from folds):**
- halves `G_A={0..a-1}`, `G_B={a..k-1}`, `a=⌊k/2⌋`, `b=k−a`, each rescaled to the
  full normalization: `norm_A = (k/a)·Σ_{f∈G_A} norm_folds[f]`, `norm_B = (k/b)·Σ_{G_B}`
  (for `k=2` the ×2 rescale). With split logk the half predictions are the rescaled
  sums of the per-fold predictions.
- A systematic shared across folds applies the full-sample variation to every fold:
  the multiplicative (`log_normal`) factor unchanged, the additive (`normal`)
  variation split evenly (`1/k` per raw fold, the full variation for a rescaled half).
- `Fitter._compute_fold_yields(halves=...)` evaluates all folds (or both halves) in
  one batched contraction, dense or sparse.
- k-fold averaging: the complete pairwise U-statistic over all `k` folds in
  `--covMode fisher` (§2b), which averages over every fold pair in `O(k)`.

This needs **no `nparams²` moment** — only first-moment fold templates,
`O(k·nparams·nbins)`.

### 2b. Objective: the jackknife combination (RESULTS.md §8d)
Because `μ̄_b = ½(n^A_b+n^B_b) = n^full_b` exactly, `L_cf = 2L_full − ½L_A − ½L_B`.

*Any k (as implemented).* With raw half sums `S_A`, `S_B` (`J_full = S_A+S_B`,
`J_A = (k/a)S_A`, `J_B = (k/b)S_B`) the weights generalize to

    L_cf = k²/(2ab)·L_full − a/(2b)·L_A − b/(2a)·L_B,

the unique choice for which the self terms `S_AᵀDS_A`, `S_BᵀDS_B` cancel in the
Gauss–Newton Hessian (leaving `sym(J_AᵀDJ_B)`) and the score is the half average
`½(J_A+J_B)ᵀD(y−n)`. For even k (`a=b`) this is `(2, ½, ½)`; odd k works the same
way (RESULTS.md §9q). The sketch below is the original even-k form.
A ~10-line change to `_compute_nll_components`:
```python
def _compute_ln_debiased(self, full_nll=False):
    nexp_full, _ = self._compute_yields_noBBB()                 # full templates
    if self.mcStatDebias in ("twoHalf", "kfold"):
        nexp_A = self._compute_yields_noBBB(templates="A")[0]    # half-A (×2-rescaled)
        nexp_B = self._compute_yields_noBBB(templates="B")[0]
        ln = (2.0 * self._compute_ln(nexp_full[:nbins])
              - 0.5 * self._compute_ln(nexp_A[:nbins])
              - 0.5 * self._compute_ln(nexp_B[:nbins]))
        return ln
    return self._compute_ln(nexp_full[:nbins])
```
`_compute_yields_noBBB` gains a `templates="full"|"A"|"B"` switch selecting
`norm`/`logk` vs `norm_A`/`logk_A` vs `norm_B`/`logk_B`. Everything else
(penalties, autodiff grad/HVP) is unchanged; bin-by-bin stat is profiled per term
(below). Autodiff of this scalar gives:
- gradient = cross-fit score → **de-biased point** (RESULTS.md §8d verified);
- Hessian = cross-half Fisher `F_ch` → de-biased curvature.
*Bin-by-bin stat.* Each jackknife term is its own **profiled** likelihood,
`L_X,bb(θ) = min_β [ln(n_X β) + lbeta(β)]`, all with the full-sample kstat
(`Fitter._profiled_half_terms`), and the constraint term of each enters with its
term's weight. The jackknife removes the `O(1/N_MC)` bias of a smooth function of
the templates, and the profiled likelihood (its minimum over β included) is that
function. The kstat stays the full sample's because it is part of the function:
per-half kstat would make the bin-by-bin stat variance itself a `1/N_MC` "bias"
that the jackknife removes, leaving the likelihood without bin-by-bin stat.
For a Gaussian term with normal-additive β, `L_X,bb = (y − n_X)²/(2V)`,
`V = D + σ²_MC`, and the de-biased curvature `Σ_X c_X J_XᵀJ_X/V` has the noise floor
removed with weight `1/V`. The earlier design (§9g) instead applied the
full-sample β to the halves; that gives `e²/(2V) − Σ_hΔ_h²/(4D)` (k = 2), which
removes the floor with `1/D`, over-subtracting by `1 + ρ`; at `ρ ≈ 1` the bread
went indefinite for every bin-by-bin stat type (RESULTS.md §9u).

Where the bin-by-bin stat variance moves with the parameters (§1b), the
jackknife's limit is the profiled likelihood at the *true* templates, whose
residual has variance `D`, not `V`: its gradient `−r²∂V/(2V²)` then has mean
`−D∂V/(2V²)`, a bias of its own (anti-attenuation, large along weakly constrained
directions: +1.4 on a true 0.42 at `ρ = 1`, RESULTS.md §9u). The fold objective
adds the compensator `½Σ_b D_b (1/V_b,ref − 1/V_b(θ))`, of gradient
`D∂V/(2V²)`, which cancels it exactly at any `θ`. It centres each profiled term on
its expectation, `(r² − D)/(2V)`; unlike the continuous-M compensator it is not a
log-likelihood, but an estimating-equation correction. The meat is the curvature of
the full-sample marginal likelihood (`½Σ log V`), whose score variance keeps the
template noise.

k-fold: the objective keeps the single A/B split; the averaging over fold pairs is
done in the `--covMode fisher` curvature by the complete U-statistic
`F_ch = k/(k−1)[J_ΣᵀDJ_Σ − Σ_i J_iᵀDJ_i] = k/(k−1)Σ_{i≠j}J_iᵀDJ_j` over the raw folds
(RESULTS.md §9i), computed in `O(k)` via the full-minus-self identity.

### 2c. Covariance (sandwich) — same as §1c, both bread & meat follow `--covMode`
`Σ = A⁻¹ H A⁻¹` with bread `A` = de-biased curvature, meat `H` = full-sample
curvature, **both in the selected `--covMode`** (don't mix):
- `fisher`: `A = F_ch = ½Σ_b(J^A_b J^B_b{}^T + J^B_b J^A_b{}^T)/V_b`,
  `H = Σ_b J_b J_b{}^T/V_b` (`J=∂nexp/∂θ` JVPs) → `H−A=M` exactly;
- `observed`: `A = ∇²L_cf` (naive jackknife Hessian), `H = ∇²L_full` → consistent
  (matching residual·∂²μ in both).

Both modes are valid robust covariances; they differ as in §2d (nonlinear toy,
curvature bias / RMS vs `F_true`):
- observed full Hessian `H_full`: bias +358 (M=157 + same-sample residual·∂²μ=201), RMS 1950;
- observed jackknife Hessian `∇²L_cf`: bias +156 (M removed; residual·∂²μ → *cross* +100), RMS 1950;
- fisher cross-half `F_ch`: bias +56, RMS 1078.
So `observed` is fully valid (the residual·∂²μ contamination is *milder* than the
undebiased Hessian — the cross structure halves it, +201→+100); `fisher` drops it
entirely (no residual bias, ~2× lower variance) and is cheaper. Matrix-free `Σ·v`
via the existing HVP machinery in either mode.

*Implementation of the GN curvature.* `F = ∇²L_obj − ∇²ln_obj + Σ_i c_i J_iᵀDJ_i`,
where `ln_obj` is the objective's profiled data terms (bin-by-bin stat constraint
and compensator included, `Fitter._profiled_data_nll`) and the last two pieces
are evaluated together as the Hessian of `S(x) − ln_obj(x)`, with the
Gauss–Newton surrogate
`S(x) = ½Σ_i c_i r_iᵀDr_i`, `r_i = n_i(x) − stop_gradient(n_i(x))`. The residual
vanishes identically at the evaluation point, so only `JᵀDJ` survives; the Hessian
goes through the same reverse-over-reverse jacobian as the ordinary one
(`O(nparams)` backward passes, each linear in nbins), and no `[nbins, nparams]`
Jacobian is materialized. `D` is constant, the curvature of a profiled term:
`1/(V + σ²_bb)` (chisq), `(C + diag σ²_bb)⁻¹` (covariance fit), `1/(n_ref + σ²_bb)`
(Poisson, with the shared full/sum-of-folds prediction for every term), with
`σ²_bb` the variance bin-by-bin stat assigns to the prediction (0 without it). The
`J_i` are the raw predictions, without β, so the full-minus-self identity of the
U-statistic stays exact. This drops the residual-dependent curvature of the
profile (including the `r²` terms of a parameter-dependent variance) together
with the compensator that cancels them, so `F` matches the expectation of the
observed bread.

### 2d. `--covMode` on the *standard* (undebiased) covariance
`--covMode` applies here too (it's the `mcStatDebias=none` instance of the same
axis). rabbit currently inverts the observed Hessian `∇²L_full` (= `observed`).
The `fisher` option `JᵀD J` (`J=∂n/∂θ` via `ForwardAccumulator` JVP, `D=diag(ℓ'')`)
is a cheap win even with no de-biasing (`demo34_residual.py`): it drops the
residual·∂²μ term → ~2× lower curvature variance (more stable σ) and removes its
`+201` self-correlation bias (so it undercovers slightly *less*), and is *cheaper*
than the full Hessian (no ∂²μ). Caveat: `fisher` does **not** remove the dominant
`M` Fisher-self-term bias (`+157`) — that still needs the de-bias (§1/§2); it is a
curvature cleanup, not a coverage fix. The case is strongest with finite MC (the
residual·∂²μ is MC-contaminated, not the benign Efron–Hinkley observed-vs-expected
difference of the infinite-MC limit). Because `--covMode` is uniform across the
standard cov and the de-biased bread/meat (§1c/§2c), the two stay consistent.

---

## 3. What to store, by parameter type

| parameter kind | continuous-M | two-half |
|---|---|---|
| free-floating process norms | **`sumw2` (already stored)** → diagonal M, exact, O(np·nbins) | half norms `norm_A/B` |
| systematics (logk), param models | derivative-template variances (NEW input) OR matrix-free M·v | half logk `logk_A/B` (carries full response, no extra moments) |
| large `nparams` | matrix-free `M·v` (one JVP), store nothing beyond `sumw2` | half templates O(R·np·nbins); matrix-free `F_ch·v` |

Recommendation: ship **continuous-M (diagonal, from `sumw2`)** first — it's a pure
fitter+covariance change, no input format change, and covers the
free-floating-normalization analyses (the original motivation). Add **two-half**
for the general case (requires the input-format extension but no `nparams²` moment
and handles arbitrary parameters + nonlinearity).

---

## 4. Patch surface (as implemented)

- `tensorwriter.py`: `add_process(..., fold_axis=)`, `add_systematic(..., fold_axis=)`,
  `add_mc_stat_moment(M, param_names, *, variance)`; `write()` adds `hnorm_folds`,
  `hlogk_folds` / `hlogk_folds_delta` and the moment's `mcstat_variance` attribute.
- `inputdata.py`: load `norm_folds`, `logk_folds`, `logk_folds_delta`,
  `mcstat_moment_variance`.
- `fitter.py`:
  - `__init__`: `--mcStatDebias`, `--mcStatDebiasCov`, `--covMode`; `_build_mcstat_M`,
    `_check_mcstat_moment_variance`, `_init_mcstat_folds`.
  - `_compute_fold_yields(halves=)`: batched per-fold / per-half predictions.
  - `_compute_nll_components`: the jackknife ln (`_jackknife_ln_terms`); the M
    penalty is an external term, kept only for `continuousM`.
  - `fisher_curvature`, `fisher_curvature_full` (`_fisher_core`), `cov_mcstat_sandwich`,
    `cov_twohalf_sandwich`, `cov_dataprop_sandwich`.
- `bin/rabbit_fit.py`: bread = observed or GN curvature; reports the sandwich and
  decomposes the de-biased curvature in the impacts, with the sandwich excess as the
  `mcStatDebias` impact group (`workspace.py`).
- `sharding.py`: multi-device fits refuse these options (the fold yields are not
  sharded).

All differentiable, all through the existing `loss_val_grad`/`loss_val_grad_hessp`
paths — no change to the minimizer interface. The de-biasing is purely additive
terms in the scalar NLL + a post-fit sandwich.

## 5a. Minimization, goodness of fit, `--noHessian`, scans

**Warm start.** The de-biased objectives are only *locally* convex. The Poisson
jackknife contains `+w_A y ln n_A` (and the same for B), which goes to `−∞` as a
half prediction `n_A → 0⁺` while the full one stays positive; the continuous-M
penalty `−½θᵀMθ` outgrows a Poisson likelihood far from the minimum. From a far
start, such as the blinding offsets (std 5), the minimizer can slide toward that
boundary. `Fitter.minimize()` therefore first minimizes the ordinary likelihood
(the `--mcStatDebias none` objective: the data term on the full templates, no M;
every other term shared) and starts the de-biased fit from its minimum, which
lies within O(σ) of the de-biased one, inside its convex basin. Skipped for a
quadratic objective (closed-form solve). Separately, the scipy loss maps a
non-finite value to `+∞`: scipy's trust-region update compares
`ρ = actual/predicted reduction` with thresholds, and every comparison on a NaN
`ρ` is False, so the radius never shrank and the same rejected step was proposed
until `--earlyStopping` (and again after each restart). Far-start scan on a
60-bin toy (RESULTS.md §9r): with both, no failures out of ~40 feasible starts
at any MC statistics; without the warm start, 8/40 and 6/41 at 200 and 2000
events per fold per bin.

**Asimov datasets** (`Fitter.set_mcstat_asimov`, called by `rabbit_fit.py` for
`-t -1` and by `rabbit_limit.py` for its Asimov datasets; a no-op without
de-biasing). Asimov data are the expected yields of the noisy templates
themselves, so they carry no attenuation and the de-biasing overshoots: the
de-biased objective is not stationary at the generating point `x₀` (continuous-M
gamma-lite at ρ ≈ 1: minimum at 1.13 for a true 0.42; RESULTS.md §9u), although
the estimator's expectation over data and template fluctuations is `x₀` by
design. The objective gets `−g₀ᵀ(x − x₀)`, `g₀` its gradient at `x₀` on the Asimov
data: the minimum is `x₀`, and being linear the term leaves every Hessian
unchanged, so scans, contours and limits on Asimov are centred at `x₀`. The
observed curvature at Asimov is not the expected one either: the bin-by-bin stat
compensator stands in for `E[r²]` while the residuals vanish, and the fold half
terms keep residuals `−Δ_h`. The Gauss–Newton bread and meat (§2c) are the
expectations of the observed ones and do not depend on the data, so an Asimov
dataset of a de-biased fit uses `--covMode fisher`; its sandwich equals that of
real data at the same point. (Omitting the compensator instead would neither
remove the overshoot, which is there without any compensator, nor fix the fold
methods' curvature.)

**Goodness of fit at the de-biased minimum.** The statistic is the same as for
an ordinary fit: `q = 2·(L_cf(θ̂) − L_sat)` = 2·(reduced NLL), the de-biased
objective *including* bin-by-bin stat at its own minimum (the jackknife weights
sum to one, so `L_cf` vanishes at the saturated point like the ordinary
likelihood). Only its reference distribution differs (`Fitter.mcstat_gof`).
Per bin, with `e = y − n` the full residual (variance `V(1+ρ)`, data and MC),
`Δ_{A,B} = n_{A,B} − n` the halves' MC deviations (independent of `e`,
`E[Δ_AΔ_B] = −σ²_MC`, `Var(Δ_A+Δ_B) = σ²_MC (a−b)²/(ab)`), and `s` the fraction of
`e` that bin-by-bin stat leaves in the data term (a profiled term's curvature in
its residual is `s/V`), each jackknife term being separately profiled (§2b), to
leading order

    q_b·V = s·[e² − e·(Δ_A + Δ_B) + Δ_A·Δ_B]

- without bin-by-bin stat (`s = 1`) this is the cross-fit χ²
  `Σ_b D_b (y−n_A)(y−n_B)` (exact for chisq, any k): the MC fluctuations cancel in
  its mean, `E[q_b] = 1`;
- with it (`s ≈ 1/(1+ρ)`), `E[q_b] = s(1+ρ) − sρ = s`: β absorbs part of both the
  common MC fluctuation and the halves' disagreement. A mismodelling shift `d` in the
  bin still raises `E[q_b]` by `d²/(V+σ²)`, exactly as in the ordinary
  bin-by-bin-stat saturated test.

So `q` is a weighted χ² `Σ_b λ_b z_b²` (`λ_b = s_b(1+ρ_b)`, minus the fitted
parameters: mean `Σλ_b + n_constrained − tr(A⁻¹H)`, `tr(A⁻¹H)` the effective
number of fitted parameters from the sandwich bread and meat), moment-matched to
`c·χ²_ν`, plus an approximately Gaussian `G` with mean `−Σs_bρ_b` and variance
`Σs_b²[2ρ_b² + (1+ρ_b)ρ_b(a−b)²/(ab)]`; the p-value is `P(c·χ²_ν + G ≥ q)`. `q`
excludes the bin-by-bin stat compensator (an offset of order `Σρ_b`). `s_b`
is taken from the fit itself, `s_b = 1 − ∂n̂_b/∂y_b` of the profiled prediction,
so it holds for any bin-by-bin stat type and mode; `ρ_b` is the MC variance of
the prediction over the data variance. The fit result stores
`mcstat_gof_pvalue`/`_mean`/`_var`, and `rabbit_plot_hists.py` uses the p-value.

*Approximations.* This reference is a large-count (leading-order) result: the
per-bin decomposition is the second-order expansion of the data term around `y`
(exact for chisq, leading order for Poisson), and the shape is moment-matched
(weighted χ² plus a Gaussian `G`). The ordinary saturated test also relies on
large counts (Wilks), but as a genuine likelihood ratio its χ² limit is generic;
the de-biased objective is not a likelihood. An approximation-free alternative,
left as a follow-up, is a parametric bootstrap of the whole sampling process:
Poisson pseudo-data from the fitted prediction, MC fold templates resampled from
their own (Poisson or effective-count) distribution around the observed ones,
refit, recompute `q`.
Under `--noHessian` the parameter count stands in for `tr(A⁻¹H)`. Not available
for continuous-M, whose objective contains `−½θᵀMθ`, which depends on the origin
of θ and is no goodness-of-fit measure. Calibration on rabbit toys: RESULTS.md
§9r.

The sandwich meat is the curvature of the whole ordinary objective, every term
of `_compute_nll` (constraints, bin-by-bin stat, external terms, regularization)
with the data term on the full templates; the bread is the same for the
de-biased objective. `--mcStatDebiasCov dataPropagated` propagates the variance
of the full templates (`sumw2`), so it is refused for the fold methods, whose
objective depends on the fold templates.

**Saturated projection tests** (`--computeSaturatedProjectionTests`). The
statistic is the difference of the de-biased objective between the main fit and
the composite fit with one free scale per projected bin, `Δq = 2(L − L_sat)`, the
saturated fit started at the main fit's minimum (without the ordinary-likelihood
warm start, so `Δq ≥ 0`). As a composite-likelihood ratio (Kent 1982) it is
asymptotically `Σ_j μ_j χ²₁`, with `μ_j` the eigenvalues of `Ã⁺B̃` at the null
point: `Ã = A_ψψ − A_ψλ A_λλ⁻¹ A_λψ` the profiled curvature of the bin scales ψ
(Schur complement of the bread), `B̃ = P H Pᵀ`, `P = [1, −A_ψλ A_λλ⁻¹]`, the
variance of the profiled score from the meat; p-values by Imhof's inversion
(`weighted_chi2_sf`). Only `A_λλ` is inverted, so a direction in which the scales
are degenerate with the analysis parameters (a common normalization) is not
testable and drops out. Without bin-by-bin stat the meat carries no MC variance,
so for this test the data term's `Σ_b ρ_b J_bJ_bᵀ/V_b` is added. The test is
skipped (p = NaN, with a warning) when a bin-scale direction the data cannot
constrain is not flat for the de-biased objective (`mcstat_projection_check`,
from the data Gauss–Newton forms): a projection that integrates out what
separates the processes lets the scales absorb a change of process composition
that the half predictions still feel. Refused for continuous-M (the free scales
open directions without data curvature, along which `−½θᵀMθ` is unbounded), and
with `--noHessian`.

**Impacts.** The traditional and global impacts (parameters and histograms) of a
de-biased fit decompose its curvature covariance and append `mcStatDebias` (the
variance the reported covariance adds, `diag(Σ − C)`) and `mcStat`, the whole
MC-statistical contribution (`binByBinStat` and `mcStatDebias` in quadrature), so
the groups add up to the reported uncertainty. The Gaussian global impacts
propagate the data, constraint and β0 variances through the de-biased estimator,
so the de-biasing changes their data-stat term itself; they get no extra groups.

**Linear χ² of a mapping.** Its residual covariance propagates the data,
constraint and β0 variances through the fit; for a de-biased fit the parameters'
template-noise variance, the positive part of `Σ − Σ_prop`, is added through
`J(·)Jᵀ`. Without it the de-biased minimum, which unlike an ordinary one leaves
a residual component along the fitted directions, read as a zero-variance
direction (χ² 565 instead of 33 on a 60-bin toy). Without bin-by-bin stat the
sandwich itself lacks the MC variance, and so does this term.

**`rabbit_limit.py`.** Gaussian limits use the reported (sandwich) uncertainty;
likelihood limits use the contour at `q_μ,A / factor`, factor = curvature /
reported variance of the parameter or mapping output.

**`--noHessian`.** The covariance rows of the POIs and NOIs come from conjugate
gradient on the bread HVP (`c_i = A⁻¹e_i`; the Gauss–Newton HVP under
`--covMode fisher`, via the same surrogate as §2c). The sandwich block is
`Σ_ij = c_iᵀ H c_j`: one meat HVP per row for the fold methods, none for
continuous-M (`Σ_ij = C_ij + c_iᵀ M c_j`). Identical to the dense path
(`Fitter.cov_block_hessfree`). `--mcStatDebiasCov dataPropagated` is refused.

**Adjusted scans.** Profiling the de-biased objective gives intervals calibrated
to its curvature `A`, not to the sandwich. The scans report the adjusted
composite-likelihood statistic (Pace, Salvan & Sartori 2011): along the
direction `d = ψ − ψ̂`

    W_adj(ψ) = W(ψ) · (dᵀ Σ_ψ⁻¹ d) / (dᵀ C_ψ⁻¹ d)

with `C_ψ`, `Σ_ψ` the curvature and sandwich blocks of the scanned parameters;
for one parameter the factor is the constant `C_ψ/Σ_ψ`. It agrees with the
sandwich Wald statistic to quadratic order while keeping the shape of the profile
along each ray (exactly `dᵀΣ⁻¹d` for a quadratic objective). `--scan` and
`--scan2D` write the adjusted values, and the raw profiles as
`nll_scan_unadjusted_*`; `--contourScan` uses the threshold `cl²/factor`.
`--asymImpacts` (contour-based) is refused under de-biasing, as is
`--externalPostfit` without recomputing the covariance when global or histogram
impacts are requested (the curvature covariance is not stored).

## 5. Validation hooks (port the toy)
- `tests/test_mcstat_debias.py`: the 2-process / 200-bin near-degenerate toy (and
  smaller ones) as rabbit input tensors; rabbit vs closed-form numpy references for
  continuous-M, two-half, any-k jackknife and the k-fold U-statistic (dense and
  sparse), BB-lite composition, and a small coverage ensemble. The long coverage
  study is `tests/mcstat_coverage.py`.
- Invariants (MD §6): `M` independent of `N_MC`; `F−M ≈ F_ch`; sandwich σ ≈ k·curvature.
