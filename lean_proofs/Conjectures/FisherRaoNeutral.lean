import Mathlib

/-!
# Characterization of Neyman–Pearson Optimality for Fisher–Rao Neutral Zones
(arXiv:2511.20960, Conjecture 1)

We formalize in Lean 4 + Mathlib the exact Fisher–Rao bijection, calibration odds,
range bound, discrete Neyman–Pearson criterion, and minimal exact-rational counterexamples
from Soumojit Das, Nairanjana Dasgupta, and Prashanta Dutta, *Geometric Calibration and
Neutral Zones for Uncertainty-Aware Multi-Class Classification* (arXiv:2511.20960):

1. **Exact reduction of the Fisher–Rao score to Maximum Class Probability (Proposition 1)**:
   - `bhattacharyya_simplex_vertex`: For any simplex vector `P : Fin c → ℝ` and predicted
     class `J : Fin c`, `∑ j, √(P j · e_J(j)) = √(P J)`.
   - `max_prob_ge_inv_card`: Every simplex vector in `Δ^(c-1)` satisfies `1/c ≤ max_j P_j ≤ 1`.
   - `frScore`, `frInv`: `R = h(S) = exp(-2λ arccos(√S))` and inverse
     `s(r) = cos²((-log r)/(2λ))`.
   - `frScore_strictMono`: For `λ > 0`, `h` is strictly increasing on `[0, 1]`.
   - `frInv_frScore`, `frScore_frInv`: Exact inverse identities `s(h(S)) = S` on `[0, 1]`
     and `h(s(r)) = r` on `[exp(-λ π), 1]`, giving exact calibrated correctness
     `Pr(correct | R = r) = cos²((-log r)/(2λ))`.
   - `frScore_sharper_lower_bound`: For `c ≥ 2`, `λ > 0`, `S ∈ [1/c, 1]`,
     `exp(-λ π) < exp(-2λ arccos(1/√c)) ≤ frScore λ S ≤ 1` (strictly sharper than
     Theorem 2's diameter bound `exp(-λ π)`).
   - `calibrated_odds_eq_cot_sq`, `calibrated_odds_strictMono`: Under perfect/confidence
     calibration, posterior correctness odds equal `cot²((-log r)/(2λ))` and are strictly
     increasing on `[exp(-2λ arccos(1/√c)), 1)`.
2. **Finite-space Neyman–Pearson optimality criterion (Theorem 1 linear identity & sufficiency)**:
   - `np_linear_identity`: `∑ ω, p ω * (U ω - q) * φ ω = (1 - q) * π * power - q * (1 - π) * size`.
   - `np_threshold_optimal_of_monotone`: If `U ω = g (S ω)` with `g` monotone, any score-threshold
     rule at cutoff `τ` maximizes power among all tests of no greater size.
3. **Minimal exact-rational counterexamples (Section 4)**:
   - `counterexample1_bayes_optimal_miscalibrated`: Binary Bayes-optimal classifier with
     scores `3/5 < 9/10` and true correctness `9/10 > 3/5`: odds fall from `9` to `3/2`,
     low-score acceptance achieves exact NP size `1/5` and power `3/5`, whereas the upper
     score threshold of exact NP size `1/5` achieves power only `1/10` (`1/10 < 3/5`), and
     at deferral budget `1/2`, low-score deferral captures `1/5` vs. `4/5` of errors.
   - `counterexample2_confidence_calibrated_loses_to_vector`: Exact confidence calibration
     (`m(3/5) = 3/5`, `m(4/5) = 4/5`) and strict score MLR, yet full-vector test gets
     correct mass `9/40` vs. score-threshold `1/10` at error mass `1/40`.
   - `counterexample3_side_info_beats_calibrated`: Perfectly calibrated constant score
     `(3/4, 1/4)` beaten by side information (size `0`, power `2/3` vs. `0`).
   - `counterexample4_class_conditional_reversal`: Perfectly calibrated classifier
     `(3/5, 2/5)` and `(1/10, 9/10)` with `Y ~ P`: pooled correctness rises from `3/5` to `9/10`,
     yet conditioned on true class `Y = 1`, `Pr(C = 1 | S = 3/5, Y = 1) = 1 > 0 = Pr(C = 1 | S = 9/10, Y = 1)`
     with positive mass on both states (`3/10` and `1/20`) and class Bayes errors `1/7` and `4/13`.

Machine-checked vs prose-only:
- Machine-checked: All exact geometric/trigonometric formulas, strict monotonicity of score
  and calibrated odds, sharper lower bound, discrete NP linear identity and sufficiency, and
  all four exact-rational counterexamples.
- Prose-only (in `solutions/fisher_rao_neutral/proof.pdf`): General measure-theoretic
  necessity via dense atom-free cutoffs (Theorems 1–2) and the 62-case rational frontier script.
-/

open BigOperators Finset

namespace FisherRaoNeutral

/-! ### 1. Exact Bijection, Sharper Lower Bound, and Calibrated Odds (Proposition 1) -/

/-- Fisher–Rao distance `d_FR(P, e_J) = 2 arccos(√S)` as a function of `S = P_J`. -/
noncomputable def dFR (s : ℝ) : ℝ := 2 * Real.arccos (Real.sqrt s)

/-- Fisher–Rao reliability score `R = h(S) = exp(-2λ arccos(√S))`. -/
noncomputable def frScore (lam s : ℝ) : ℝ :=
  Real.exp (-2 * lam * Real.arccos (Real.sqrt s))

/-- Exact closed-form inverse `s(r) = cos²((-log r)/(2λ))` (calibrated `Pr(correct | R = r)`). -/
noncomputable def frInv (lam r : ℝ) : ℝ :=
  (Real.cos (-Real.log r / (2 * lam))) ^ 2

/-- Bhattacharyya coefficient between simplex vector `P` and vertex `e_J` equals `√(P J)`. -/
theorem bhattacharyya_simplex_vertex {c : ℕ} (P : Fin c → ℝ) (J : Fin c) :
    ∑ j : Fin c, Real.sqrt (P j * if j = J then 1 else 0) = Real.sqrt (P J) := by
  rw [Finset.sum_eq_single J]
  · simp
  · intro j _ hne
    simp [hne]
  · intro h; exact (h (Finset.mem_univ J)).elim

/-- On the `(c-1)`-simplex (`c ≥ 1`), the maximum class probability lies in `[1/c, 1]`. -/
theorem max_prob_bounds {c : ℕ} (hc : 0 < c) {P : Fin c → ℝ}
    (hnonneg : ∀ j, 0 ≤ P j) (hsum : ∑ j : Fin c, P j = 1)
    {J : Fin c} (hmax : ∀ j, P j ≤ P J) :
    1 / (c : ℝ) ≤ P J ∧ P J ≤ 1 := by
  constructor
  · have hle : (1 : ℝ) ≤ (c : ℝ) * P J := by
      calc (1 : ℝ) = ∑ j : Fin c, P j := hsum.symm
        _ ≤ ∑ _j : Fin c, P J := Finset.sum_le_sum fun j _ => hmax j
        _ = (c : ℝ) * P J := by simp [Finset.sum_const, nsmul_eq_mul]
    rw [div_le_iff₀ (Nat.cast_pos.mpr hc)]
    linarith
  · calc P J = ∑ j ∈ ({J} : Finset (Fin c)), P j := by simp
      _ ≤ ∑ j : Fin c, P j := Finset.sum_le_univ_sum_of_nonneg hnonneg
      _ = 1 := hsum

/-- **Proposition 1(a)**: For `λ > 0`, `frScore lam` is strictly increasing on `[0, 1]`. -/
theorem frScore_strictMonoOn {lam : ℝ} (hlam : 0 < lam) :
    StrictMonoOn (frScore lam) (Set.Icc 0 1) := by
  intro s₁ hs₁ s₂ hs₂ hlt
  unfold frScore
  apply Real.exp_strictMono
  have hsqrt_lt : Real.sqrt s₁ < Real.sqrt s₂ := Real.sqrt_lt_sqrt hs₁.1 hlt
  have hsqrt1_ge : -1 ≤ Real.sqrt s₁ := by linarith [Real.sqrt_nonneg s₁]
  have hsqrt2_le : Real.sqrt s₂ ≤ 1 := by
    rw [← Real.sqrt_one]; exact Real.sqrt_le_sqrt hs₂.2
  have harccos_lt : Real.arccos (Real.sqrt s₂) < Real.arccos (Real.sqrt s₁) :=
    Real.arccos_lt_arccos hsqrt1_ge hsqrt_lt hsqrt2_le
  nlinarith

/-- **Proposition 1(b)**: Exact left-inverse identity `cos²((-log (frScore λ s))/(2λ)) = s`
for every `λ > 0` and `s ∈ [0, 1]`. -/
theorem frInv_frScore {lam s : ℝ} (hlam : 0 < lam) (hs0 : 0 ≤ s) (hs1 : s ≤ 1) :
    frInv lam (frScore lam s) = s := by
  unfold frInv frScore
  rw [Real.log_exp]
  have harg : -(-2 * lam * Real.arccos (Real.sqrt s)) / (2 * lam) = Real.arccos (Real.sqrt s) := by
    have hne : lam ≠ 0 := ne_of_gt hlam
    field_simp [hne]
  rw [harg, Real.cos_arccos (by linarith [Real.sqrt_nonneg s])
    (by rw [← Real.sqrt_one]; exact Real.sqrt_le_sqrt hs1)]
  exact Real.sq_sqrt hs0

/-- **Proposition 1(c)**: Exact right-inverse identity `frScore λ (frInv λ r) = r` for every
`λ > 0` and `r ∈ [exp(-λ π), 1]`. -/
theorem frScore_frInv {lam r : ℝ} (hlam : 0 < lam)
    (hr_lo : Real.exp (-lam * Real.pi) ≤ r) (hr_hi : r ≤ 1) :
    frScore lam (frInv lam r) = r := by
  have hr_pos : 0 < r := lt_of_lt_of_le (Real.exp_pos _) hr_lo
  set θ : ℝ := -Real.log r / (2 * lam)
  have hlog_le : Real.log r ≤ 0 := Real.log_nonpos (le_of_lt hr_pos) hr_hi
  have hlog_ge : -lam * Real.pi ≤ Real.log r := by
    rw [← Real.log_exp (-lam * Real.pi)]
    exact Real.log_le_log (Real.exp_pos _) hr_lo
  have hθ0 : 0 ≤ θ := div_nonneg (by linarith) (by positivity)
  have hθ_hi : θ ≤ Real.pi / 2 := by
    dsimp [θ]
    rw [div_le_div_iff₀ (by positivity) (by norm_num : (0 : ℝ) < 2)]
    nlinarith
  have hcos_nonneg : 0 ≤ Real.cos θ :=
    Real.cos_nonneg_of_mem_Icc ⟨by linarith [Real.pi_pos], hθ_hi⟩
  unfold frScore frInv
  change Real.exp (-2 * lam * Real.arccos (Real.sqrt (Real.cos θ ^ 2))) = r
  rw [Real.sqrt_sq hcos_nonneg, Real.arccos_cos hθ0 (by linarith [Real.pi_pos])]
  have harg : -2 * lam * θ = Real.log r := by
    dsimp [θ]
    have hne : lam ≠ 0 := ne_of_gt hlam
    field_simp [hne]
  rw [harg, Real.exp_log hr_pos]

/-- **Sharper lower bound**: For `c ≥ 2`, `λ > 0`, and `s ∈ [1/c, 1]`,
`exp(-λ π) < exp(-2λ arccos(1/√c)) ≤ frScore λ s ≤ 1`. -/
theorem frScore_sharper_lower_bound {c : ℕ} (hc : 2 ≤ c) {lam s : ℝ}
    (hlam : 0 < lam) (hs_lo : 1 / (c : ℝ) ≤ s) (hs_hi : s ≤ 1) :
    Real.exp (-lam * Real.pi) < Real.exp (-2 * lam * Real.arccos (1 / Real.sqrt c)) ∧
    Real.exp (-2 * lam * Real.arccos (1 / Real.sqrt c)) ≤ frScore lam s ∧
    frScore lam s ≤ 1 := by
  have hc_pos : (0 : ℝ) < c := by exact_mod_cast (by omega : 0 < c)
  have hsqrtc_pos : 0 < Real.sqrt c := Real.sqrt_pos.mpr hc_pos
  have hinv_pos : 0 < 1 / Real.sqrt c := div_pos one_pos hsqrtc_pos
  have hinv_le_one : 1 / Real.sqrt c ≤ 1 := by
    rw [div_le_one hsqrtc_pos, ← Real.sqrt_one]
    exact Real.sqrt_le_sqrt (by exact_mod_cast (by omega : 1 ≤ c))
  have hinv_sq : Real.sqrt (1 / (c : ℝ)) = 1 / Real.sqrt c := by
    rw [Real.sqrt_div zero_le_one, Real.sqrt_one]
  refine ⟨?_, ?_, ?_⟩
  · apply Real.exp_strictMono
    have harc_lt : Real.arccos (1 / Real.sqrt c) < Real.pi / 2 := by
      rw [← Real.arccos_zero]
      exact Real.arccos_lt_arccos (by linarith) hinv_pos hinv_le_one
    nlinarith
  · have hfr_lo : frScore lam (1 / (c : ℝ)) = Real.exp (-2 * lam * Real.arccos (1 / Real.sqrt c)) := by
      unfold frScore; rw [hinv_sq]
    rw [← hfr_lo]
    exact (frScore_strictMonoOn hlam).monotoneOn
      ⟨by positivity, by rw [div_le_one hc_pos]; exact_mod_cast (by omega : 1 ≤ c)⟩
      ⟨le_trans (by positivity) hs_lo, hs_hi⟩ hs_lo
  · have hfr1 : frScore lam 1 = 1 := by
      unfold frScore; simp [Real.arccos_one]
    rw [← hfr1]
    exact (frScore_strictMonoOn hlam).monotoneOn
      ⟨le_trans (by positivity) hs_lo, hs_hi⟩ ⟨zero_le_one, le_rfl⟩ hs_hi

/-- **Proposition 1(d)**: Under confidence calibration (`Pr(C = 1 | R = r) = s(r)`),
the posterior correctness odds equal `cot²((-log r)/(2λ))` and are strictly increasing
on `[exp(-2λ arccos(1/√c)), 1)`. -/
theorem calibrated_odds_eq_cot_sq (lam r : ℝ) :
    let θ := -Real.log r / (2 * lam)
    let s := frInv lam r
    s / (1 - s) = (Real.cos θ / Real.sin θ) ^ 2 := by
  intro θ s
  have hsin_sq : 1 - Real.cos θ ^ 2 = Real.sin θ ^ 2 := by
    have := Real.sin_sq_add_cos_sq θ; linarith
  dsimp [s, frInv, θ]
  rw [hsin_sq, div_pow]

theorem calibrated_odds_strictMonoOn {lam : ℝ} (hlam : 0 < lam) :
    StrictMonoOn (fun r => frInv lam r / (1 - frInv lam r))
      (Set.Ioo (Real.exp (-lam * Real.pi)) 1) := by
  intro r₁ hr₁ r₂ hr₂ hlt
  have h1pos : 0 < r₁ := lt_trans (Real.exp_pos _) hr₁.1
  have h2pos : 0 < r₂ := lt_trans h1pos hlt
  set θ₁ := -Real.log r₁ / (2 * lam)
  set θ₂ := -Real.log r₂ / (2 * lam)
  have hlog_lt : Real.log r₁ < Real.log r₂ := Real.log_lt_log h1pos hlt
  have hθ_lt : θ₂ < θ₁ := by
    dsimp [θ₁, θ₂]
    exact div_lt_div_of_pos_right (by linarith) (by positivity)
  have hθ2_pos : 0 < θ₂ := by
    dsimp [θ₂]
    have hlog2 : Real.log r₂ < 0 := Real.log_neg h2pos hr₂.2
    exact div_pos (by linarith) (by positivity)
  have hθ1_lt : θ₁ < Real.pi / 2 := by
    dsimp [θ₁]
    have hlog1 : -lam * Real.pi < Real.log r₁ := by
      rw [← Real.log_exp (-lam * Real.pi)]
      exact Real.log_lt_log (Real.exp_pos _) hr₁.1
    rw [div_lt_div_iff₀ (by positivity) (by norm_num : (0 : ℝ) < 2)]
    nlinarith
  have hcos_pos1 : 0 < Real.cos θ₁ :=
    Real.cos_pos_of_mem_Ioo ⟨by linarith [Real.pi_pos], hθ1_lt⟩
  have hcos_pos2 : 0 < Real.cos θ₂ :=
    Real.cos_pos_of_mem_Ioo ⟨by linarith [Real.pi_pos], lt_trans hθ_lt hθ1_lt⟩
  have hcos_lt : Real.cos θ₁ < Real.cos θ₂ :=
    Real.strictAntiOn_cos ⟨le_of_lt hθ2_pos, by linarith [Real.pi_pos]⟩
      ⟨le_of_lt (lt_trans hθ2_pos hθ_lt), by linarith [Real.pi_pos]⟩ hθ_lt
  have hs_lt : frInv lam r₁ < frInv lam r₂ := by
    unfold frInv
    change Real.cos θ₁ ^ 2 < Real.cos θ₂ ^ 2
    nlinarith
  have hs2_lt1 : frInv lam r₂ < 1 := by
    unfold frInv
    change Real.cos θ₂ ^ 2 < 1
    have hsin2 : 0 < Real.sin θ₂ :=
      Real.sin_pos_of_pos_of_lt_pi hθ2_pos (by linarith [Real.pi_pos])
    nlinarith [Real.sin_sq_add_cos_sq θ₂]
  have hs1_lt1 : frInv lam r₁ < 1 := lt_trans hs_lt hs2_lt1
  rw [div_lt_div_iff₀ (sub_pos.mpr hs1_lt1) (sub_pos.mpr hs2_lt1)]
  nlinarith

/-! ### 2. Finite-Space Neyman–Pearson Linear Identity and Sufficiency (Theorem 1) -/

/-- **Theorem 1 linear identity**: For any finite probability space `(Ω, p)`, correctness
probability `U : Ω → ℝ`, prior accuracy `π`, and test `φ : Ω → ℝ`,
`∑ ω, p ω (U ω - q) φ ω = (1 - q) (∑ ω, p ω U ω φ ω) - q (∑ ω, p ω (1 - U ω) φ ω)`. -/
theorem np_linear_identity {Ω : Type*} [Fintype Ω] (p U φ : Ω → ℝ) (q : ℝ) :
    ∑ ω : Ω, p ω * (U ω - q) * φ ω =
      (1 - q) * (∑ ω : Ω, p ω * U ω * φ ω) -
      q * (∑ ω : Ω, p ω * (1 - U ω) * φ ω) := by
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun ω _ => ?_
  ring

/-- **Theorem 1 sufficiency (discrete)**: If `U ω = g (S ω)` with `g` monotone, and `φ*` is a
score-threshold rule at cutoff `τ` (`S ω > τ → φ* ω = 1` and `S ω < τ → φ* ω = 0`) with
`q = g τ ∈ (0, 1)`, then `φ*` achieves maximum correct mass among all tests `φ : Ω → [0, 1]`
with no greater error mass. -/
theorem np_threshold_optimal_of_monotone {Ω : Type*} [Fintype Ω]
    {p S U φStar φ : Ω → ℝ} {g : ℝ → ℝ} {τ : ℝ}
    (hp : ∀ ω, 0 ≤ p ω)
    (hU : ∀ ω, U ω = g (S ω))
    (hg : Monotone g)
    (hq0 : 0 < g τ) (hq1 : g τ < 1)
    (_hstar_bounds : ∀ ω, 0 ≤ φStar ω ∧ φStar ω ≤ 1)
    (hstar_hi : ∀ ω, τ < S ω → φStar ω = 1)
    (hstar_lo : ∀ ω, S ω < τ → φStar ω = 0)
    (hφ_bounds : ∀ ω, 0 ≤ φ ω ∧ φ ω ≤ 1)
    (hsize : ∑ ω : Ω, p ω * (1 - U ω) * φ ω ≤ ∑ ω : Ω, p ω * (1 - U ω) * φStar ω) :
    ∑ ω : Ω, p ω * U ω * φ ω ≤ ∑ ω : Ω, p ω * U ω * φStar ω := by
  set q := g τ
  have hpoint : ∀ ω : Ω, p ω * (U ω - q) * φ ω ≤ p ω * (U ω - q) * φStar ω := by
    intro ω
    rw [hU ω]
    rcases lt_trichotomy (S ω) τ with hlt | heq | hgt
    · rw [hstar_lo ω hlt, mul_zero]
      have hgle : g (S ω) ≤ q := hg (le_of_lt hlt)
      have h1 : g (S ω) - q ≤ 0 := sub_nonpos.mpr hgle
      have h2 : 0 ≤ p ω * φ ω := mul_nonneg (hp ω) (hφ_bounds ω).1
      nlinarith
    · rw [heq]; simp [q]
    · rw [hstar_hi ω hgt, mul_one]
      have hgge : q ≤ g (S ω) := hg (le_of_lt hgt)
      have h1 : 0 ≤ g (S ω) - q := sub_nonneg.mpr hgge
      have h2 : 0 ≤ p ω * (g (S ω) - q) := mul_nonneg (hp ω) h1
      nlinarith [(hφ_bounds ω).2]
  have hsum := Finset.sum_le_sum fun ω (_ : ω ∈ Finset.univ) => hpoint ω
  rw [np_linear_identity p U φ q, np_linear_identity p U φStar q] at hsum
  have h1q : 0 < 1 - q := sub_pos.mpr hq1
  nlinarith

/-! ### 3. Minimal Exact-Rational Counterexamples (Section 4) -/

/-- **Counterexample 1 (Reversed odds despite Bayes-optimal predicted labels)**:
Two equiprobable states with reported probabilities `(3/5, 2/5)` and `(9/10, 1/10)`,
and true class-1 probabilities `9/10` and `3/5`:
- Class 1 is Bayes-optimal at both states (`9/10 > 1/2`, `3/5 > 1/2`).
- Score strictly increases (`3/5 < 9/10`, and `frScore λ (3/5) < frScore λ (9/10)`),
  yet correctness odds strictly decrease (`9 > 3/2`).
- Accepting only state 0 achieves exact NP size `1/5` and power `3/5`, whereas the
  upper score threshold of exact NP size `1/5` (accepting `1/4` of state 1) achieves
  power only `1/10 < 3/5`.
- At deferral budget `1/2`, low-score deferral captures `1/5` of errors vs. `4/5` for
  high-score deferral. -/
theorem counterexample1_bayes_optimal_miscalibrated {lam : ℝ} (hlam : 0 < lam) :
    let p : Fin 2 → ℝ := ![1 / 2, 1 / 2]
    let S : Fin 2 → ℝ := ![3 / 5, 9 / 10]
    let U : Fin 2 → ℝ := ![9 / 10, 3 / 5]
    let π := ∑ i, p i * U i
    let φLow : Fin 2 → ℝ := ![1, 0]
    let φUpper : Fin 2 → ℝ := ![0, 1 / 4]
    -- Bayes-optimal predicted class at both states
    (1 / 2 < U 0 ∧ 1 / 2 < U 1) ∧
    -- Fisher-Rao score strictly increases while correctness odds strictly decrease
    (frScore lam (S 0) < frScore lam (S 1)) ∧
    (U 0 / (1 - U 0) = 9 ∧ U 1 / (1 - U 1) = 3 / 2 ∧ U 1 / (1 - U 1) < U 0 / (1 - U 0)) ∧
    -- Exact NP size 1/5 for both tests, but power 3/5 vs 1/10
    ((∑ i, p i * (1 - U i) * φLow i) / (1 - π) = 1 / 5) ∧
    ((∑ i, p i * U i * φLow i) / π = 3 / 5) ∧
    ((∑ i, p i * (1 - U i) * φUpper i) / (1 - π) = 1 / 5) ∧
    ((∑ i, p i * U i * φUpper i) / π = 1 / 10) ∧
    ((1 / 10 : ℝ) < 3 / 5) ∧
    -- Deferral budget 1/2: low-score captures 1/5 of errors vs 4/5 for high-score
    ((∑ i, p i * (1 - U i) * φLow i) / (1 - π) = 1 / 5 ∧
     (∑ i, p i * (1 - U i) * (![0, 1] : Fin 2 → ℝ) i) / (1 - π) = 4 / 5) := by
  intro p S U π φLow φUpper
  have hπ : π = 3 / 4 := by
    dsimp [π, p, U]; simp [Fin.sum_univ_two]; norm_num
  refine ⟨⟨by simp [U]; norm_num, by simp [U]; norm_num⟩, ?_, ?_, ?_, ?_, ?_, ?_, by norm_num, ?_⟩
  · exact frScore_strictMonoOn hlam ⟨by simp [S]; norm_num, by simp [S]; norm_num⟩
      ⟨by simp [S]; norm_num, by simp [S]; norm_num⟩ (by simp [S]; norm_num)
  · simp [U]; norm_num
  · rw [hπ]; simp [p, U, φLow, Fin.sum_univ_two]; norm_num
  · rw [hπ]; simp [p, U, φLow, Fin.sum_univ_two]; norm_num
  · rw [hπ]; simp [p, U, φUpper, Fin.sum_univ_two]; norm_num
  · rw [hπ]; simp [p, U, φUpper, Fin.sum_univ_two]; norm_num
  · rw [hπ]; simp [p, U, φLow, Fin.sum_univ_two]; norm_num

/-- **Counterexample 2 (Exact confidence calibration + strict score MLR loses to full vector)**:
Three states with masses `1/4, 1/4, 1/2`, scores `3/5, 3/5, 4/5`, and true correctness
`3/10, 9/10, 4/5`. Confidence calibration `m(3/5) = 3/5`, `m(4/5) = 4/5` holds identically,
yet accepting only state 1 attains error mass `1/40` and correct mass `9/40`, strictly beating
the upper score threshold (error mass `1/40`, correct mass `1/10 = 4/40`). -/
theorem counterexample2_confidence_calibrated_loses_to_vector :
    let p : Fin 3 → ℝ := ![1 / 4, 1 / 4, 1 / 2]
    let U : Fin 3 → ℝ := ![3 / 10, 9 / 10, 4 / 5]
    let φVec : Fin 3 → ℝ := ![0, 1, 0]
    let φThr : Fin 3 → ℝ := ![0, 0, 1 / 4]
    -- Exact confidence calibration at both score levels 3/5 and 4/5
    ((p 0 * U 0 + p 1 * U 1) / (p 0 + p 1) = 3 / 5 ∧ U 2 = 4 / 5) ∧
    -- Equal error mass 1/40, strictly higher correct mass 9/40 > 1/10
    (∑ i, p i * (1 - U i) * φVec i = 1 / 40) ∧
    (∑ i, p i * U i * φVec i = 9 / 40) ∧
    (∑ i, p i * (1 - U i) * φThr i = 1 / 40) ∧
    (∑ i, p i * U i * φThr i = 1 / 10) ∧
    ((1 / 10 : ℝ) < 9 / 40) := by
  intro p U φVec φThr
  refine ⟨⟨by simp [p, U]; norm_num, by simp [U]⟩, ?_, ?_, ?_, ?_, by norm_num⟩ <;>
    (simp [p, U, φVec, φThr, Fin.sum_univ_three]; norm_num)

/-- **Counterexample 3 (Side information beats perfectly calibrated score)**:
Two equally likely subpopulations with true class-1 probs `1` and `1/2`, always reporting `(3/4, 1/4)`.
Perfect calibration `(1 + 1/2)/2 = 3/4` holds, yet accepting subpopulation 0 attains exact
NP size `0` and power `2/3`, whereas any score-only test of size `0` has power `0`. -/
theorem counterexample3_side_info_beats_calibrated :
    let p : Fin 2 → ℝ := ![1 / 2, 1 / 2]
    let U : Fin 2 → ℝ := ![1, 1 / 2]
    let π := ∑ i, p i * U i
    let φX : Fin 2 → ℝ := ![1, 0]
    π = 3 / 4 ∧
    (∑ i, p i * (1 - U i) * φX i) / (1 - π) = 0 ∧
    (∑ i, p i * U i * φX i) / π = 2 / 3 ∧
    (∀ c : ℝ, 0 ≤ c → (∑ i, p i * (1 - U i) * c) / (1 - π) = 0 →
      (∑ i, p i * U i * c) / π = 0) := by
  intro p U π φX
  have hπ : π = 3 / 4 := by dsimp [π, p, U]; simp [Fin.sum_univ_two]; norm_num
  refine ⟨hπ, ?_, ?_, ?_⟩
  · rw [hπ]; simp [p, U, φX, Fin.sum_univ_two]
  · rw [hπ]; simp [p, U, φX, Fin.sum_univ_two]; norm_num
  · intro c _ hc
    rw [hπ] at hc ⊢
    simp [p, U, Fin.sum_univ_two] at hc ⊢
    have hc0 : c = 0 := by
      rcases hc with (h1 | h2) | h3 <;> linarith
    simp [hc0]

/-- **Counterexample 4 (Class-conditional ordering reversal under perfect calibration)**:
Two equiprobable states with true posteriors `P₀ = (3/5, 2/5)` (predicts class 1, `S = 3/5`)
and `P₁ = (1/10, 9/10)` (predicts class 2, `S = 9/10`) and `Y ~ P` (perfectly calibrated).
Pooled correctness rises from `3/5` to `9/10`, yet conditioned on true class `Y = 1`:
both scores occur with positive probability (`3/10` and `1/20`), and
`Pr(C = 1 | S = 3/5, Y = 1) = 1 > 0 = Pr(C = 1 | S = 9/10, Y = 1)`.
Moreover, the class-conditional Bayes errors are `1/7` (class 1) and `4/13` (class 2). -/
theorem counterexample4_class_conditional_reversal {lam : ℝ} (hlam : 0 < lam) :
    let w0 : ℝ := 1 / 2
    let w1 : ℝ := 1 / 2
    let p0_cls1 : ℝ := 3 / 5
    let p0_cls2 : ℝ := 2 / 5
    let p1_cls1 : ℝ := 1 / 10
    let p1_cls2 : ℝ := 9 / 10
    -- Scores are 3/5 and 9/10, with frScore strictly increasing
    (frScore lam (3 / 5) < frScore lam (9 / 10)) ∧
    -- Positive joint mass of (S = 3/5, Y = 1) and (S = 9/10, Y = 1)
    (w0 * p0_cls1 = 3 / 10 ∧ 0 < w0 * p0_cls1) ∧
    (w1 * p1_cls1 = 1 / 20 ∧ 0 < w1 * p1_cls1) ∧
    -- Conditional correctness given Y = 1 reverses from 1 to 0
    ((w0 * p0_cls1) / (w0 * p0_cls1) = 1 ∧ (0 : ℝ) / (w1 * p1_cls1) = 0) ∧
    -- Class-conditional Bayes errors are 1/7 (class 1) and 4/13 (class 2)
    ((w1 * p1_cls1) / (w0 * p0_cls1 + w1 * p1_cls1) = 1 / 7) ∧
    ((w0 * p0_cls2) / (w0 * p0_cls2 + w1 * p1_cls2) = 4 / 13) := by
  refine ⟨frScore_strictMonoOn hlam ⟨by norm_num, by norm_num⟩ ⟨by norm_num, by norm_num⟩
    (by norm_num), ?_, ?_, ?_, ?_, ?_⟩ <;> norm_num

end FisherRaoNeutral
