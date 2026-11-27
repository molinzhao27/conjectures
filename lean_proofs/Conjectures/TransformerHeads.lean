import Mathlib

/-!
# Lean 4 Formalization of the Refutation of Conjecture 1 (Yu et al., ICLR 2026, arXiv:2510.06662)

We formalize in Lean 4 + Mathlib (zero `sorry`, zero `axiom`) all core lemmas and both explicit
multilayer transformer constructions refuting Conjecture 1 of Yu et al. (ICLR 2026,
arXiv:2510.06662) for standard multilayer bidirectional transformer encoders (where all tokens
issue queries and residual connections around attention are present; the paper leaves the
multilayer architecture unspecified, so CLS-only and residual-free variants remain open).
Conjecture 1 asks whether `L * h ≥ D` is necessary to avoid `T`-dependent exponential growth in
FFN parameter count `M` (with softmax inverse temperature `β > 0` allowed to grow with `T / ε`,
as in the paper's own Theorem 2(1)):

1. **Lemma 3 (Two-Sided Softmin Error Bound, `softmin_error_bound` & `softmin_error_bound_univ`)**:
   For any `T ≥ 1`, values `a : Fin T → ℝ` in `[0, 1]`, nonempty subset `S ⊆ Fin T`, inverse
   temperature `β > 0`, and logits `-β * a t` (`t ∈ S`), `-β * (a t + 2)` (`t ∉ S`), `-2 * β`
   (CLS token), the softmax-weighted average `z̃ = ∑_{t ∈ Fin T} w(t) a(t)` satisfies
   `|z̃ - min_{t ∈ S} a(t)| ≤ T / (e * β) + (T + 1) * exp (-β)` (and for `S = Finset.univ`,
   `|z̃ - min_t a(t)| ≤ T / (e * β) + exp (-β)`).
2. **Exact Theorem B Attention Construction (`L = 2, h = 1, D = 3, n = 9`, NoPE + NoLN + zero
   intermediate FFNs)**:
   Explicit `9 × 9` attention matrices `WQ1, WK1, WV1, WO1, WQ2, WK2, WV2, WO2` with all entries in
   `{-1, 0, 1} ⊆ [-1, 1]` (`theoremB_matrices_bounded`, independent of `T`), proving unit-norm
   hidden states `∑ j, (x^(ℓ)(s)_j)² ≤ 1` at Layers 0, 1, 2 (`nopeEmbed_unit_norm`,
   `theoremB_L2_h1_D3`) and the exact attention-layer readout error bound
   `≤ 3 * (T / (e * β̃) + 2 * exp (-β̃))` (`theoremB_L2_h1_D3`).
3. **Lemma 1 (Piecewise-Linear ReLU Squaring Error Bound, `pwlSquare`, `pwlSquare_error`,
   `pwlSquare_mem_Icc`, `pwlSquare_weights_bounded`)**:
   Explicit 2-layer ReLU interpolant of `(u - c₀)²` on uniform knots `k / N` (`k = 0..N`) using
   `N + 2` ReLU neurons with every weight and bias in `[-1, 1]` (`N ≥ 2`, `c₀ ∈ [0, 1]`),
   staying in `[0, 1]` and approximating `(u - c₀)²` within `[0, 1 / (4 * N²)]`.
4. **Lemma 2 (Neuron-Splitting Rescaling & `[-1, 1]`-Bounded Readout, `relu_rescale_splitting`,
   `readoutB`, `readoutB_eq_three_sum`)**:
   Exact rescaling `v * max 0 (w · (α • x) + b) = ∑_{Fin ⌈α⌉} (v * α / ⌈α⌉) * max 0 (w · x + b / α)`
   preserving `[-1, 1]`-bounded weights and biases for any `α ≥ 1`, plus the explicit 3-neuron
   `[-1, 1]`-bounded ReLU readout `readoutB` realizing `3 * (x 3 + x 4 + x 6)` on nonnegative
   coordinates.
5. **End-to-End Theorem B (`H3`, `approxFeat`, `theoremB_end_to_end_L2_h1_D3`)**:
   Composes `approxFeat` (`d * (N + 2)` ReLU neurons, `N` independent of `T`) → `nopeEmbed` →
   Layer 1 → Layer 2 → `readoutB` to uniformly approximate the exact 3-task target
   `H3(X) = ∑_{i=0}^2 min_t (‖X_t - x^(i)‖₂² / d)` within any `ε > 0` for all `d ≥ 1` and `T ≥ 1`,
   with all hidden-state norms `≤ 1` and all attention/FFN weights and biases in `[-1, 1]`.
6. **Theorem A (`L = 2, h = 1`, PE + attention residuals + exact Layer 1 ReLU gating for
   arbitrary `D ≥ 1` and `T ≥ D`, `card_slotA`, `theoremA_matrices_bounded`,
   `theoremA_L2_h1_arbitrary_D`)**:
   Explicit `E = 5D + 2` embedding (`Fintype.card (SlotA D) = 5 * D + 2`), `{0, 1}` attention
   matrices, and `D`-neuron Layer 1 ReLU gate `max 0 (u_{F,i} + u_{B,i} - c)` with
   `c = 1 / √(11D + 3)`, proving unit-norm bounds at Layers 0, 1, 2, exact Layer 1 parallel
   retrieval and gating, and the Layer 2 CLS readout bound
   `|(D / c) * ∑_i X2(CLS)_{G,i} - ∑_i min_t P t i| ≤ D * (T / (e * β̃₁) + exp(-β̃₁)) + (T + 1 - D) / exp(β̃₂)`.

(Note: the general `(L, h)` family `D = (2L - 1)h` of Theorem B and the FFN parameter-count
asymptotics `M = O_{d,D}(1/ε^γ)` are proved in prose in `solution.md` and verified numerically in
`verify_transformer_heads.py`.)
-/

open Finset BigOperators

namespace TransformerHeadCount

/-! ### Part 1: Lemma 3 — Uniform Two-Sided Softmin Error Bound -/

/-- Elementary inequality `x * exp (-β * x) ≤ 1 / (e * β)` for all `x ≥ 0` and `β > 0`. -/
lemma mul_exp_neg_le {x β : ℝ} (_hx : 0 ≤ x) (hβ : 0 < β) :
    x * Real.exp (-(β * x)) ≤ 1 / (Real.exp 1 * β) := by
  have h1 : β * x ≤ Real.exp (β * x - 1) := by
    have := Real.add_one_le_exp (β * x - 1)
    linarith
  have hexp_eq : Real.exp (β * x - 1) = Real.exp (β * x) / Real.exp 1 := by
    rw [Real.exp_sub]
  rw [hexp_eq] at h1
  have he_pos : 0 < Real.exp 1 := Real.exp_pos 1
  have hbx_pos : 0 < Real.exp (β * x) := Real.exp_pos (β * x)
  have h2 : Real.exp 1 * β * x ≤ Real.exp (β * x) := by
    have := (le_div_iff₀ he_pos).mp h1
    linarith
  have h3 : x * Real.exp (-(β * x)) = x / Real.exp (β * x) := by
    rw [Real.exp_neg, div_eq_mul_inv]
  rw [h3, div_le_div_iff₀ hbx_pos (mul_pos he_pos hβ)]
  linarith

/-- Unnormalized softmax weight for token `t ∈ Fin T` with mask `S : Finset (Fin T)`. -/
noncomputable def tokWeight {T : ℕ} (S : Finset (Fin T)) (a : Fin T → ℝ) (β : ℝ) (t : Fin T) : ℝ :=
  if t ∈ S then Real.exp (-(β * a t)) else Real.exp (-(β * (a t + 2)))

/-- Normalization denominator including CLS token with logit `-2 * β`. -/
noncomputable def softminDenom {T : ℕ} (S : Finset (Fin T)) (a : Fin T → ℝ) (β : ℝ) : ℝ :=
  (∑ t : Fin T, tokWeight S a β t) + Real.exp (-(2 * β))

/-- Softmax-weighted average `z̃ = (∑ t, tokWeight t * a t) / softminDenom`. -/
noncomputable def softminAvg {T : ℕ} (S : Finset (Fin T)) (a : Fin T → ℝ) (β : ℝ) : ℝ :=
  (∑ t : Fin T, tokWeight S a β t * a t) / softminDenom S a β

lemma tokWeight_pos {T : ℕ} (S : Finset (Fin T)) (a : Fin T → ℝ) (β : ℝ) (t : Fin T) :
    0 < tokWeight S a β t := by
  unfold tokWeight; split_ifs <;> exact Real.exp_pos _

lemma softminDenom_pos {T : ℕ} (S : Finset (Fin T)) (a : Fin T → ℝ) (β : ℝ) :
    0 < softminDenom S a β := by
  unfold softminDenom
  have h1 : 0 ≤ ∑ t : Fin T, tokWeight S a β t :=
    Finset.sum_nonneg fun t _ => le_of_lt (tokWeight_pos S a β t)
  linarith [Real.exp_pos (-(2 * β))]

/-- Softmin average stays in `[0, 1]` whenever `a t ∈ [0, 1]`. -/
lemma softminAvg_mem_Icc {T : ℕ} (S : Finset (Fin T)) (a : Fin T → ℝ) (β : ℝ)
    (ha : ∀ t, 0 ≤ a t ∧ a t ≤ 1) :
    0 ≤ softminAvg S a β ∧ softminAvg S a β ≤ 1 := by
  have hZ : 0 < softminDenom S a β := softminDenom_pos S a β
  constructor
  · apply div_nonneg _ (le_of_lt hZ)
    exact Finset.sum_nonneg fun t _ =>
      mul_nonneg (le_of_lt (tokWeight_pos S a β t)) (ha t).1
  · unfold softminAvg
    rw [div_le_one hZ]
    have hsum : ∑ t : Fin T, tokWeight S a β t * a t ≤ ∑ t : Fin T, tokWeight S a β t :=
      Finset.sum_le_sum fun t _ => by
        have hw := le_of_lt (tokWeight_pos S a β t)
        nlinarith [(ha t).2]
    unfold softminDenom
    linarith [Real.exp_pos (-(2 * β))]

/-- **Lemma 3 (Uniform Two-Sided Softmin Error Bound)**:
For any nonempty `S ⊆ Fin T`, `a : Fin T → [0, 1]`, and `β > 0`, letting `a_* = S.inf' hS a`:
`|softminAvg S a β - a_*| ≤ T / (e * β) + (T + 1) * exp (-β)`. -/
theorem softmin_error_bound {T : ℕ} (S : Finset (Fin T)) (hS : S.Nonempty)
    (a : Fin T → ℝ) (ha : ∀ t, 0 ≤ a t ∧ a t ≤ 1) {β : ℝ} (hβ : 0 < β) :
    let a_star := S.inf' hS a
    |softminAvg S a β - a_star| ≤
      (T : ℝ) / (Real.exp 1 * β) + ((T : ℝ) + 1) * Real.exp (-β) := by
  intro a_star
  obtain ⟨t_star, ht_mem, ht_eq⟩ := Finset.exists_mem_eq_inf' hS a
  have ha_star_le : ∀ t ∈ S, a_star ≤ a t := fun t ht => Finset.inf'_le a ht
  have ha_star_Icc : 0 ≤ a_star ∧ a_star ≤ 1 := by
    dsimp only [a_star]; rw [ht_eq]; exact ha t_star
  have hZ_pos : 0 < softminDenom S a β := softminDenom_pos S a β
  -- Denominator lower bound: Z ≥ exp(-β * a_star)
  have hZ_ge : Real.exp (-(β * a_star)) ≤ softminDenom S a β := by
    unfold softminDenom
    have hterm : tokWeight S a β t_star ≤ ∑ t : Fin T, tokWeight S a β t :=
      Finset.single_le_sum (fun t _ => le_of_lt (tokWeight_pos S a β t)) (Finset.mem_univ t_star)
    have htw : tokWeight S a β t_star = Real.exp (-(β * a_star)) := by
      dsimp only [a_star]; rw [ht_eq]; simp [tokWeight, ht_mem]
    linarith [Real.exp_pos (-(2 * β))]
  have hexp_star_pos : 0 < Real.exp (-(β * a_star)) := Real.exp_pos _
  -- Rewrite z̃ - a_* = (∑_t w_t (a_t - a_*) - exp(-2β) * a_*) / Z
  have hdiff : softminAvg S a β - a_star =
      ((∑ t : Fin T, tokWeight S a β t * (a t - a_star)) - Real.exp (-(2 * β)) * a_star) /
        softminDenom S a β := by
    unfold softminAvg
    rw [sub_eq_iff_eq_add]
    field_simp [ne_of_gt hZ_pos]
    unfold softminDenom
    have hdist : ∑ t : Fin T, tokWeight S a β t * (a t - a_star) =
        (∑ t : Fin T, tokWeight S a β t * a t) -
          (∑ t : Fin T, tokWeight S a β t) * a_star := by
      rw [Finset.sum_mul, ← Finset.sum_sub_distrib]
      refine Finset.sum_congr rfl fun t _ => by ring
    linarith
  rw [hdiff, abs_div, abs_of_pos hZ_pos, div_le_iff₀ hZ_pos]
  -- Bound |numerator| ≤ (T / (e β) + (T + 1) exp(-β)) * exp(-β a_star) ≤ (...) * Z
  have hterm_bound : ∀ t : Fin T,
      |tokWeight S a β t * (a t - a_star)| ≤
        (1 / (Real.exp 1 * β) + Real.exp (-β)) * Real.exp (-(β * a_star)) := by
    intro t
    by_cases htS : t ∈ S
    · -- t ∈ S: a_t - a_* ≥ 0, w_t * (a_t - a_*) = exp(-β a_*) * ((a_t - a_*) * exp(-β (a_t - a_*)))
      have hge : 0 ≤ a t - a_star := sub_nonneg.mpr (ha_star_le t htS)
      have htw : tokWeight S a β t = Real.exp (-(β * a_star)) * Real.exp (-(β * (a t - a_star))) := by
        simp only [tokWeight, htS, ↓reduceIte, ← Real.exp_add]
        congr 1; ring
      rw [abs_of_nonneg (mul_nonneg (le_of_lt (tokWeight_pos S a β t)) hge), htw]
      have hme := mul_exp_neg_le hge hβ
      have hexp_nn : 0 ≤ Real.exp (-β) := le_of_lt (Real.exp_pos _)
      have hcalc : Real.exp (-(β * a_star)) * Real.exp (-(β * (a t - a_star))) * (a t - a_star) =
          ((a t - a_star) * Real.exp (-(β * (a t - a_star)))) * Real.exp (-(β * a_star)) := by ring
      rw [hcalc]
      nlinarith
    · -- t ∉ S: |a_t - a_*| ≤ 1 and w_t = exp(-β (a_t + 2)) ≤ exp(-β) * exp(-β a_*)
      have habs : |a t - a_star| ≤ 1 := by
        rw [abs_le]; constructor <;> linarith [(ha t).1, (ha t).2, ha_star_Icc.1, ha_star_Icc.2]
      have hw_le : tokWeight S a β t ≤ Real.exp (-β) * Real.exp (-(β * a_star)) := by
        simp only [tokWeight, htS, ↓reduceIte, ← Real.exp_add]
        apply Real.exp_le_exp.mpr
        nlinarith [(ha t).1, ha_star_Icc.2]
      rw [abs_mul, abs_of_pos (tokWeight_pos S a β t)]
      have hinv_nn : 0 ≤ 1 / (Real.exp 1 * β) := by positivity
      calc tokWeight S a β t * |a t - a_star|
          ≤ tokWeight S a β t * 1 := mul_le_mul_of_nonneg_left habs (le_of_lt (tokWeight_pos S a β t))
        _ = tokWeight S a β t := mul_one _
        _ ≤ Real.exp (-β) * Real.exp (-(β * a_star)) := hw_le
        _ ≤ (1 / (Real.exp 1 * β) + Real.exp (-β)) * Real.exp (-(β * a_star)) := by nlinarith
  have hcls_bound : |Real.exp (-(2 * β)) * a_star| ≤ Real.exp (-β) * Real.exp (-(β * a_star)) := by
    rw [abs_of_nonneg (mul_nonneg (le_of_lt (Real.exp_pos _)) ha_star_Icc.1)]
    have hexp_le : Real.exp (-(2 * β)) ≤ Real.exp (-β) * Real.exp (-(β * a_star)) := by
      rw [← Real.exp_add]
      apply Real.exp_le_exp.mpr
      nlinarith [ha_star_Icc.2]
    nlinarith [ha_star_Icc.2, Real.exp_pos (-(2 * β))]
  have hnum : |(∑ t : Fin T, tokWeight S a β t * (a t - a_star)) - Real.exp (-(2 * β)) * a_star| ≤
      ((T : ℝ) / (Real.exp 1 * β) + ((T : ℝ) + 1) * Real.exp (-β)) * Real.exp (-(β * a_star)) := by
    have htri := abs_sub (∑ t : Fin T, tokWeight S a β t * (a t - a_star)) (Real.exp (-(2 * β)) * a_star)
    have hsum_abs : |∑ t : Fin T, tokWeight S a β t * (a t - a_star)| ≤
        ∑ t : Fin T, |tokWeight S a β t * (a t - a_star)| := Finset.abs_sum_le_sum_abs _ _
    have hsum_le : ∑ t : Fin T, |tokWeight S a β t * (a t - a_star)| ≤
        (T : ℝ) * ((1 / (Real.exp 1 * β) + Real.exp (-β)) * Real.exp (-(β * a_star))) := by
      calc ∑ t : Fin T, |tokWeight S a β t * (a t - a_star)|
          ≤ ∑ _t : Fin T, ((1 / (Real.exp 1 * β) + Real.exp (-β)) * Real.exp (-(β * a_star))) :=
            Finset.sum_le_sum fun t _ => hterm_bound t
        _ = (T : ℝ) * ((1 / (Real.exp 1 * β) + Real.exp (-β)) * Real.exp (-(β * a_star))) := by
            simp
    have hlin : (T : ℝ) * ((1 / (Real.exp 1 * β) + Real.exp (-β)) * Real.exp (-(β * a_star))) +
        Real.exp (-β) * Real.exp (-(β * a_star)) =
        ((T : ℝ) / (Real.exp 1 * β) + ((T : ℝ) + 1) * Real.exp (-β)) * Real.exp (-(β * a_star)) := by
      ring
    linarith
  have hcoef_nn : 0 ≤ (T : ℝ) / (Real.exp 1 * β) + ((T : ℝ) + 1) * Real.exp (-β) := by
    positivity
  exact le_trans hnum (mul_le_mul_of_nonneg_left hZ_ge hcoef_nn)

/-- Tighter specialization for full-sequence retrieval `S = Finset.univ`:
`|softminAvg univ a β - min_t a t| ≤ T / (e * β) + exp (-β)`. -/
theorem softmin_error_bound_univ {T : ℕ} (hT : 0 < T)
    (a : Fin T → ℝ) (ha : ∀ t, 0 ≤ a t ∧ a t ≤ 1) {β : ℝ} (hβ : 0 < β) :
    let hne : (Finset.univ : Finset (Fin T)).Nonempty := ⟨⟨0, hT⟩, Finset.mem_univ _⟩
    let a_star := Finset.univ.inf' hne a
    |softminAvg Finset.univ a β - a_star| ≤
      (T : ℝ) / (Real.exp 1 * β) + Real.exp (-β) := by
  intro hne a_star
  obtain ⟨t_star, _, ht_eq⟩ := Finset.exists_mem_eq_inf' hne a
  have ha_star_le : ∀ t : Fin T, a_star ≤ a t := fun t => Finset.inf'_le a (Finset.mem_univ t)
  have ha_star_Icc : 0 ≤ a_star ∧ a_star ≤ 1 := by
    dsimp only [a_star]; rw [ht_eq]; exact ha t_star
  have hZ_pos : 0 < softminDenom Finset.univ a β := softminDenom_pos _ _ _
  have hZ_ge : Real.exp (-(β * a_star)) ≤ softminDenom Finset.univ a β := by
    unfold softminDenom
    have hterm := Finset.single_le_sum (fun t _ => le_of_lt (tokWeight_pos Finset.univ a β t))
      (Finset.mem_univ t_star)
    have htw : tokWeight Finset.univ a β t_star = Real.exp (-(β * a_star)) := by
      dsimp only [a_star]; rw [ht_eq]; simp [tokWeight]
    linarith [Real.exp_pos (-(2 * β))]
  have hdiff : softminAvg Finset.univ a β - a_star =
      ((∑ t : Fin T, tokWeight Finset.univ a β t * (a t - a_star)) - Real.exp (-(2 * β)) * a_star) /
        softminDenom Finset.univ a β := by
    unfold softminAvg
    rw [sub_eq_iff_eq_add]
    field_simp [ne_of_gt hZ_pos]
    unfold softminDenom
    have hdist : ∑ t : Fin T, tokWeight Finset.univ a β t * (a t - a_star) =
        (∑ t : Fin T, tokWeight Finset.univ a β t * a t) -
          (∑ t : Fin T, tokWeight Finset.univ a β t) * a_star := by
      rw [Finset.sum_mul, ← Finset.sum_sub_distrib]
      refine Finset.sum_congr rfl fun t _ => by ring
    linarith
  rw [hdiff, abs_div, abs_of_pos hZ_pos, div_le_iff₀ hZ_pos]
  have hsum_nn : 0 ≤ ∑ t : Fin T, tokWeight Finset.univ a β t * (a t - a_star) :=
    Finset.sum_nonneg fun t _ =>
      mul_nonneg (le_of_lt (tokWeight_pos _ _ _ t)) (sub_nonneg.mpr (ha_star_le t))
  have hsum_ub : ∑ t : Fin T, tokWeight Finset.univ a β t * (a t - a_star) ≤
      ((T : ℝ) / (Real.exp 1 * β)) * Real.exp (-(β * a_star)) := by
    calc ∑ t : Fin T, tokWeight Finset.univ a β t * (a t - a_star)
        ≤ ∑ _t : Fin T, ((1 / (Real.exp 1 * β)) * Real.exp (-(β * a_star))) := by
          refine Finset.sum_le_sum fun t _ => ?_
          have hge : 0 ≤ a t - a_star := sub_nonneg.mpr (ha_star_le t)
          have htw : tokWeight Finset.univ a β t =
              Real.exp (-(β * a_star)) * Real.exp (-(β * (a t - a_star))) := by
            simp only [tokWeight, Finset.mem_univ, ↓reduceIte, ← Real.exp_add]
            congr 1; ring
          rw [htw]
          have hme := mul_exp_neg_le hge hβ
          have hcalc : Real.exp (-(β * a_star)) * Real.exp (-(β * (a t - a_star))) * (a t - a_star) =
              ((a t - a_star) * Real.exp (-(β * (a t - a_star)))) * Real.exp (-(β * a_star)) := by ring
          rw [hcalc]
          exact mul_le_mul_of_nonneg_right hme (le_of_lt (Real.exp_pos _))
      _ = ((T : ℝ) / (Real.exp 1 * β)) * Real.exp (-(β * a_star)) := by
          simp; ring
  have hcls_nn : 0 ≤ Real.exp (-(2 * β)) * a_star :=
    mul_nonneg (le_of_lt (Real.exp_pos _)) ha_star_Icc.1
  have hcls_ub : Real.exp (-(2 * β)) * a_star ≤ Real.exp (-β) * Real.exp (-(β * a_star)) := by
    have hexp_le : Real.exp (-(2 * β)) ≤ Real.exp (-β) * Real.exp (-(β * a_star)) := by
      rw [← Real.exp_add]
      apply Real.exp_le_exp.mpr
      nlinarith [ha_star_Icc.2]
    nlinarith [ha_star_Icc.2, Real.exp_pos (-(2 * β))]
  have hnum : |(∑ t : Fin T, tokWeight Finset.univ a β t * (a t - a_star)) - Real.exp (-(2 * β)) * a_star| ≤
      ((T : ℝ) / (Real.exp 1 * β) + Real.exp (-β)) * Real.exp (-(β * a_star)) := by
    rw [abs_le]
    constructor <;> linarith
  have hcoef_nn : 0 ≤ (T : ℝ) / (Real.exp 1 * β) + Real.exp (-β) := by positivity
  exact le_trans hnum (mul_le_mul_of_nonneg_left hZ_ge hcoef_nn)

/-! ### Part 2: Exact Theorem B (`L = 2, h = 1, D = 3, n = 9`) NoPE Counterexample -/

/-- Exact attention output for a single head `(WQ, WK, WV, WO)` on a `(T+1)`-token sequence
`X : Fin (T + 1) → Fin E → ℝ` with inverse temperature `β`. -/
noncomputable def attnLayer {T E : ℕ} (WQ WK WV WO : Matrix (Fin E) (Fin E) ℝ) (β : ℝ)
    (X : Fin (T + 1) → Fin E → ℝ) : Fin (T + 1) → Fin E → ℝ :=
  fun s =>
    let q := WQ.mulVec (X s)
    let weights := fun t : Fin (T + 1) => Real.exp (β * dotProduct q (WK.mulVec (X t)))
    let Z := ∑ t : Fin (T + 1), weights t
    let attnOut := fun i => (∑ t : Fin (T + 1), weights t * (WV.mulVec (X t) i)) / Z
    X s + WO.mulVec attnOut

/-- Helper: entry-wise `[-1, 1]` bound on a matrix. -/
def BoundedMatrix {E : ℕ} (M : Matrix (Fin E) (Fin E) ℝ) : Prop :=
  ∀ i j, -1 ≤ M i j ∧ M i j ≤ 1

/-- Theorem B (`L = 2, h = 1, D = 3, E = 9`) Layer 1 matrices:
Slots: `0,1,2` = `V` (`P_0, P_1, P_2`); `3,4,5` = `M` (`M_0, M_1, M_2`);
`6` = `M'_0`; `7` = `cls`; `8` = `tok`. -/
def WQ1 : Matrix (Fin 9) (Fin 9) ℝ := Matrix.of fun i j =>
  if (i = 0 ∧ j = 7) ∨ (i = 1 ∧ j = 8) ∨ (i = 2 ∧ (j = 7 ∨ j = 8)) ∨ (i = 3 ∧ (j = 7 ∨ j = 8))
  then 1 else 0

def WK1 : Matrix (Fin 9) (Fin 9) ℝ := Matrix.of fun i j =>
  if (i = 0 ∧ j = 0) ∨ (i = 1 ∧ j = 2) ∨ (i = 2 ∧ j = 7) ∨ (i = 3 ∧ j = 7)
  then -1 else 0

def WV1 : Matrix (Fin 9) (Fin 9) ℝ := Matrix.of fun i j =>
  if (i = 0 ∧ j = 0) ∨ (i = 1 ∧ j = 2) then 1 else 0

def WO1 : Matrix (Fin 9) (Fin 9) ℝ := Matrix.of fun i j =>
  if (i = 3 ∧ j = 0) ∨ (i = 5 ∧ j = 1) then 1 else 0

/-- Theorem B (`L = 2, h = 1, D = 3, E = 9`) Layer 2 matrices:
Retrieves task `1` into slot `4` (`M_1`) AND piggybacks token-stored task `2` from slot `5` (`M_2`)
into slot `6` (`M'_0`). -/
def WQ2 : Matrix (Fin 9) (Fin 9) ℝ := Matrix.of fun i j =>
  if (i = 0 ∧ j = 7) ∨ (i = 1 ∧ j = 7) ∨ (i = 2 ∧ j = 7) then 1 else 0

def WK2 : Matrix (Fin 9) (Fin 9) ℝ := Matrix.of fun i j =>
  if (i = 0 ∧ j = 1) ∨ (i = 1 ∧ j = 7) ∨ (i = 2 ∧ j = 7) then -1 else 0

def WV2 : Matrix (Fin 9) (Fin 9) ℝ := Matrix.of fun i j =>
  if (i = 0 ∧ j = 1) ∨ (i = 1 ∧ j = 5) then 1 else 0

def WO2 : Matrix (Fin 9) (Fin 9) ℝ := Matrix.of fun i j =>
  if (i = 4 ∧ j = 0) ∨ (i = 6 ∧ j = 1) then 1 else 0

theorem theoremB_matrices_bounded :
    BoundedMatrix WQ1 ∧ BoundedMatrix WK1 ∧ BoundedMatrix WV1 ∧ BoundedMatrix WO1 ∧
    BoundedMatrix WQ2 ∧ BoundedMatrix WK2 ∧ BoundedMatrix WV2 ∧ BoundedMatrix WO2 := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    (intro i j; dsimp [BoundedMatrix, WQ1, WK1, WV1, WO1, WQ2, WK2, WV2, WO2]; split_ifs <;> norm_num)

lemma WQ1_mulVec (x : Fin 9 → ℝ) :
    WQ1.mulVec x = ![x 7, x 8, x 7 + x 8, x 7 + x 8, 0, 0, 0, 0, 0] := by
  ext i; fin_cases i <;>
    (simp [Matrix.mulVec, dotProduct, WQ1, Fin.sum_univ_succ]; try ring)

lemma WK1_mulVec (x : Fin 9 → ℝ) :
    WK1.mulVec x = ![-x 0, -x 2, -x 7, -x 7, 0, 0, 0, 0, 0] := by
  ext i; fin_cases i <;>
    (simp [Matrix.mulVec, dotProduct, WK1]; try ring)

lemma WV1_mulVec (x : Fin 9 → ℝ) :
    WV1.mulVec x = ![x 0, x 2, 0, 0, 0, 0, 0, 0, 0] := by
  ext i; fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, WV1]

lemma WO1_mulVec (x : Fin 9 → ℝ) :
    WO1.mulVec x = ![0, 0, 0, x 0, 0, x 1, 0, 0, 0] := by
  ext i; fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, WO1]

lemma WQ2_mulVec (x : Fin 9 → ℝ) :
    WQ2.mulVec x = ![x 7, x 7, x 7, 0, 0, 0, 0, 0, 0] := by
  ext i; fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, WQ2]

lemma WK2_mulVec (x : Fin 9 → ℝ) :
    WK2.mulVec x = ![-x 1, -x 7, -x 7, 0, 0, 0, 0, 0, 0] := by
  ext i; fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, WK2]

lemma WV2_mulVec (x : Fin 9 → ℝ) :
    WV2.mulVec x = ![x 1, x 5, 0, 0, 0, 0, 0, 0, 0] := by
  ext i; fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, WV2]

lemma WO2_mulVec (x : Fin 9 → ℝ) :
    WO2.mulVec x = ![0, 0, 0, 0, x 0, 0, x 1, 0, 0] := by
  ext i; fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, WO2]

/-- Strictly NoPE input embedding for Theorem B (`c = 1/3`, so `c² = 1/9`):
- Input tokens `t : Fin T` (embedded via `Fin.castSucc t`): store `c * P t 0, c * P t 1, c * P t 2`
  in slots `0, 1, 2` and `c` in slot `8` (`tok`). Notice ZERO dependence on `t`!
- CLS token (`Fin.last T`): stores `c` in slot `7` (`cls`), all else `0`. -/
noncomputable def nopeEmbed {T : ℕ} (P : Fin T → Fin 3 → ℝ) : Fin (T + 1) → Fin 9 → ℝ :=
  Fin.snoc
    (fun t : Fin T => ![P t 0 / 3, P t 1 / 3, P t 2 / 3, 0, 0, 0, 0, 0, 1 / 3])
    ![0, 0, 0, 0, 0, 0, 0, 1 / 3, 0]

/-- Exact unit-norm verification for the NoPE embeddings: `∑ j, (X0 s j)² ≤ 1` for every token `s`. -/
theorem nopeEmbed_unit_norm {T : ℕ} (P : Fin T → Fin 3 → ℝ)
    (hP : ∀ t i, 0 ≤ P t i ∧ P t i ≤ 1) (s : Fin (T + 1)) :
    ∑ j : Fin 9, (nopeEmbed P s j) ^ 2 ≤ 1 := by
  refine Fin.lastCases ?_ ?_ s
  · simp [nopeEmbed, Fin.sum_univ_succ]; norm_num
  · intro t
    simp only [nopeEmbed, Fin.snoc_castSucc, Fin.sum_univ_succ, Fin.isValue, Matrix.cons_val_zero,
      Matrix.cons_val_succ, Matrix.cons_val_fin_one, Fin.sum_univ_zero, add_zero]
    have h0 := hP t 0; have h1 := hP t 1; have h2 := hP t 2
    nlinarith

set_option maxHeartbeats 600000 in
/-- **Exact Attention-Mechanism Verification for Theorem B (`L = 2, h = 1, D = 3, n = 9`)**:
On ANY sequence length `T ≥ 1`, ANY feature matrix `P : Fin T → Fin 3 → [0, 1]`, and ANY
effective temperature `β̃ > 0` (setting `β = 9 * β̃` since `c² = 1/9`), the strictly NoPE,
NoLN, zero-intermediate-FFN 2-layer 1-head bidirectional encoder with residuals
`X2 = attnLayer WQ2 WK2 WV2 WO2 β (attnLayer WQ1 WK1 WV1 WO1 β (nopeEmbed P))` satisfies:
1. Every hidden state at Layers 0, 1, and 2 has `‖x^(ℓ)(s)‖₂² ≤ 1`;
2. The linear CLS readout `3 * (X2 (last T) 3 + X2 (last T) 4 + X2 (last T) 6)` approximates
   `∑_{i=0}^2 min_t P t i` within `3 * (T / (e * β̃) + 2 * exp (-β̃))` (so choosing
   `β̃ = Θ(T / ε)` drives the error below any `ε > 0` with fixed `9 × 9` attention matrices;
   the factor `3` in the readout is realized with `[-1, 1]`-bounded weights via Lemma 2 in
   `solution.md`). -/
theorem theoremB_L2_h1_D3 {T : ℕ} (hT : 0 < T) (P : Fin T → Fin 3 → ℝ)
    (hP : ∀ t i, 0 ≤ P t i ∧ P t i ≤ 1) {btilde : ℝ} (hbt : 0 < btilde) :
    let β := 9 * btilde
    let X0 := nopeEmbed P
    let X1 := attnLayer WQ1 WK1 WV1 WO1 β X0
    let X2 := attnLayer WQ2 WK2 WV2 WO2 β X1
    let hne : (Finset.univ : Finset (Fin T)).Nonempty := ⟨⟨0, hT⟩, Finset.mem_univ _⟩
    (∀ s, ∑ j : Fin 9, (X0 s j) ^ 2 ≤ 1) ∧
    (∀ s, ∑ j : Fin 9, (X1 s j) ^ 2 ≤ 1) ∧
    (∀ s, ∑ j : Fin 9, (X2 s j) ^ 2 ≤ 1) ∧
    |3 * (X2 (Fin.last T) 3 + X2 (Fin.last T) 4 + X2 (Fin.last T) 6) -
        (Finset.univ.inf' hne (fun t => P t 0) +
         Finset.univ.inf' hne (fun t => P t 1) +
         Finset.univ.inf' hne (fun t => P t 2))| ≤
      3 * ((T : ℝ) / (Real.exp 1 * btilde) + 2 * Real.exp (-btilde)) := by
  intro β X0 X1 X2 hne
  set z0 := softminAvg Finset.univ (fun t => P t 0) btilde
  set z1 := softminAvg Finset.univ (fun t => P t 1) btilde
  set z2 := softminAvg Finset.univ (fun t => P t 2) btilde
  have hz0 : 0 ≤ z0 ∧ z0 ≤ 1 := softminAvg_mem_Icc _ _ _ (fun t => hP t 0)
  have hz1 : 0 ≤ z1 ∧ z1 ≤ 1 := softminAvg_mem_Icc _ _ _ (fun t => hP t 1)
  have hz2 : 0 ≤ z2 ∧ z2 ≤ 1 := softminAvg_mem_Icc _ _ _ (fun t => hP t 2)
  -- Helper: compute exact dot products for Layer 1 using our fast mulVec lemmas
  have hL1_tok : ∀ t : Fin T,
      X1 (Fin.castSucc t) =
        ![P t 0 / 3, P t 1 / 3, P t 2 / 3,
          (∑ u : Fin T, tokWeight Finset.univ (fun u => P u 2) btilde u * (P u 0 / 3)) /
            softminDenom Finset.univ (fun u => P u 2) btilde,
          0, z2 / 3, 0, 0, 1 / 3] := by
    intro t
    have hdot_u : ∀ u : Fin T,
        β * dotProduct (WQ1.mulVec (X0 (Fin.castSucc t))) (WK1.mulVec (X0 (Fin.castSucc u))) =
          -(btilde * P u 2) := by
      intro u
      rw [WQ1_mulVec, WK1_mulVec]
      simp [β, X0, nopeEmbed, dotProduct, Fin.sum_univ_succ]
      ring
    have hdot_cls :
        β * dotProduct (WQ1.mulVec (X0 (Fin.castSucc t))) (WK1.mulVec (X0 (Fin.last T))) =
          -(2 * btilde) := by
      rw [WQ1_mulVec, WK1_mulVec]
      simp [β, X0, nopeEmbed, dotProduct, Fin.sum_univ_succ]
      ring
    have hZeq : (∑ s : Fin (T + 1),
        Real.exp (β * dotProduct (WQ1.mulVec (X0 (Fin.castSucc t))) (WK1.mulVec (X0 s)))) =
        softminDenom Finset.univ (fun u => P u 2) btilde := by
      rw [Fin.sum_univ_castSucc, hdot_cls]
      simp_rw [hdot_u]
      simp [softminDenom, tokWeight]
    have hWV1_u : ∀ u : Fin T, WV1.mulVec (X0 (Fin.castSucc u)) = ![P u 0 / 3, P u 2 / 3, 0, 0, 0, 0, 0, 0, 0] := by
      intro u; rw [WV1_mulVec]; simp [X0, nopeEmbed]
    have hWV1_cls : WV1.mulVec (X0 (Fin.last T)) = ![0, 0, 0, 0, 0, 0, 0, 0, 0] := by
      rw [WV1_mulVec]; simp [X0, nopeEmbed]
    have hattn0 : (∑ s : Fin (T + 1),
        Real.exp (β * dotProduct (WQ1.mulVec (X0 (Fin.castSucc t))) (WK1.mulVec (X0 s))) *
          WV1.mulVec (X0 s) 0) / softminDenom Finset.univ (fun u => P u 2) btilde =
        (∑ u : Fin T, tokWeight Finset.univ (fun u => P u 2) btilde u * (P u 0 / 3)) /
          softminDenom Finset.univ (fun u => P u 2) btilde := by
      rw [Fin.sum_univ_castSucc, hdot_cls]
      simp_rw [hdot_u, hWV1_u, hWV1_cls]
      simp [tokWeight]
    have hattn1 : (∑ s : Fin (T + 1),
        Real.exp (β * dotProduct (WQ1.mulVec (X0 (Fin.castSucc t))) (WK1.mulVec (X0 s))) *
          WV1.mulVec (X0 s) 1) / softminDenom Finset.univ (fun u => P u 2) btilde =
        z2 / 3 := by
      rw [Fin.sum_univ_castSucc, hdot_cls]
      simp_rw [hdot_u, hWV1_u, hWV1_cls]
      have hpull : (∑ x : Fin T, Real.exp (-(btilde * P x 2)) * (P x 2 / 3)) =
          (∑ x : Fin T, Real.exp (-(btilde * P x 2)) * P x 2) / 3 := by
        rw [Finset.sum_div]; refine Finset.sum_congr rfl fun x _ => by ring
      simp only [z2, softminAvg, tokWeight, Finset.mem_univ, ↓reduceIte]
      change (∑ x : Fin T, Real.exp (-(btilde * P x 2)) * (P x 2 / 3) +
          Real.exp (-(2 * btilde)) * 0) / softminDenom Finset.univ (fun u => P u 2) btilde = _
      rw [mul_zero, add_zero, hpull]; ring
    funext j
    simp only [X1, attnLayer, hZeq, WO1_mulVec, Pi.add_apply]
    change X0 (Fin.castSucc t) j +
        ![0, 0, 0,
          (∑ s : Fin (T + 1),
            Real.exp (β * dotProduct (WQ1.mulVec (X0 (Fin.castSucc t))) (WK1.mulVec (X0 s))) *
              WV1.mulVec (X0 s) 0) / softminDenom Finset.univ (fun u => P u 2) btilde,
          0,
          (∑ s : Fin (T + 1),
            Real.exp (β * dotProduct (WQ1.mulVec (X0 (Fin.castSucc t))) (WK1.mulVec (X0 s))) *
              WV1.mulVec (X0 s) 1) / softminDenom Finset.univ (fun u => P u 2) btilde,
          0, 0, 0] j = _
    rw [hattn0, hattn1]
    fin_cases j <;> simp [X0, nopeEmbed]
  have hL1_cls :
      X1 (Fin.last T) =
        ![0, 0, 0, z0 / 3, 0,
          (∑ u : Fin T, tokWeight Finset.univ (fun u => P u 0) btilde u * (P u 2 / 3)) /
            softminDenom Finset.univ (fun u => P u 0) btilde,
          0, 1 / 3, 0] := by
    have hdot_u : ∀ u : Fin T,
        β * dotProduct (WQ1.mulVec (X0 (Fin.last T))) (WK1.mulVec (X0 (Fin.castSucc u))) =
          -(btilde * P u 0) := by
      intro u
      rw [WQ1_mulVec, WK1_mulVec]
      simp [β, X0, nopeEmbed, dotProduct, Fin.sum_univ_succ]
      ring
    have hdot_cls :
        β * dotProduct (WQ1.mulVec (X0 (Fin.last T))) (WK1.mulVec (X0 (Fin.last T))) =
          -(2 * btilde) := by
      rw [WQ1_mulVec, WK1_mulVec]
      simp [β, X0, nopeEmbed, dotProduct, Fin.sum_univ_succ]
      ring
    have hZeq : (∑ s : Fin (T + 1),
        Real.exp (β * dotProduct (WQ1.mulVec (X0 (Fin.last T))) (WK1.mulVec (X0 s)))) =
        softminDenom Finset.univ (fun u => P u 0) btilde := by
      rw [Fin.sum_univ_castSucc, hdot_cls]
      simp_rw [hdot_u]
      simp [softminDenom, tokWeight]
    have hWV1_u : ∀ u : Fin T, WV1.mulVec (X0 (Fin.castSucc u)) = ![P u 0 / 3, P u 2 / 3, 0, 0, 0, 0, 0, 0, 0] := by
      intro u; rw [WV1_mulVec]; simp [X0, nopeEmbed]
    have hWV1_cls : WV1.mulVec (X0 (Fin.last T)) = ![0, 0, 0, 0, 0, 0, 0, 0, 0] := by
      rw [WV1_mulVec]; simp [X0, nopeEmbed]
    have hattn0 : (∑ s : Fin (T + 1),
        Real.exp (β * dotProduct (WQ1.mulVec (X0 (Fin.last T))) (WK1.mulVec (X0 s))) *
          WV1.mulVec (X0 s) 0) / softminDenom Finset.univ (fun u => P u 0) btilde =
        z0 / 3 := by
      rw [Fin.sum_univ_castSucc, hdot_cls]
      simp_rw [hdot_u, hWV1_u, hWV1_cls]
      simp only [Matrix.cons_val_zero, mul_zero, add_zero, z0, softminAvg,
        tokWeight, Finset.mem_univ, ↓reduceIte]
      have hpull : (∑ x : Fin T, Real.exp (-(btilde * P x 0)) * (P x 0 / 3)) =
          (∑ x : Fin T, Real.exp (-(btilde * P x 0)) * P x 0) / 3 := by
        rw [Finset.sum_div]; refine Finset.sum_congr rfl fun x _ => by ring
      rw [hpull]; ring
    have hattn1 : (∑ s : Fin (T + 1),
        Real.exp (β * dotProduct (WQ1.mulVec (X0 (Fin.last T))) (WK1.mulVec (X0 s))) *
          WV1.mulVec (X0 s) 1) / softminDenom Finset.univ (fun u => P u 0) btilde =
        (∑ u : Fin T, tokWeight Finset.univ (fun u => P u 0) btilde u * (P u 2 / 3)) /
          softminDenom Finset.univ (fun u => P u 0) btilde := by
      rw [Fin.sum_univ_castSucc, hdot_cls]
      simp_rw [hdot_u, hWV1_u, hWV1_cls]
      simp [tokWeight]
    funext j
    simp only [X1, attnLayer, hZeq, WO1_mulVec, Pi.add_apply]
    change X0 (Fin.last T) j +
        ![0, 0, 0,
          (∑ s : Fin (T + 1),
            Real.exp (β * dotProduct (WQ1.mulVec (X0 (Fin.last T))) (WK1.mulVec (X0 s))) *
              WV1.mulVec (X0 s) 0) / softminDenom Finset.univ (fun u => P u 0) btilde,
          0,
          (∑ s : Fin (T + 1),
            Real.exp (β * dotProduct (WQ1.mulVec (X0 (Fin.last T))) (WK1.mulVec (X0 s))) *
              WV1.mulVec (X0 s) 1) / softminDenom Finset.univ (fun u => P u 0) btilde,
          0, 0, 0] j = _
    rw [hattn0, hattn1]
    fin_cases j <;> simp [X0, nopeEmbed]
  -- Helper bound: any convex-like weighted average with weights ≤ denominator is in [0, 1/3]
  have havg_bound : ∀ (a b : Fin T → ℝ), (∀ u, 0 ≤ b u ∧ b u ≤ 1) →
      0 ≤ (∑ u : Fin T, tokWeight Finset.univ a btilde u * (b u / 3)) /
            softminDenom Finset.univ a btilde ∧
      (∑ u : Fin T, tokWeight Finset.univ a btilde u * (b u / 3)) /
            softminDenom Finset.univ a btilde ≤ 1 / 3 := by
    intro a b hb
    have hZ := softminDenom_pos Finset.univ a btilde
    constructor
    · apply div_nonneg _ (le_of_lt hZ)
      exact Finset.sum_nonneg fun u _ =>
        mul_nonneg (le_of_lt (tokWeight_pos _ _ _ u)) (by linarith [(hb u).1])
    · rw [div_le_div_iff₀ hZ (by norm_num)]
      have h1 : (∑ u : Fin T, tokWeight Finset.univ a btilde u * (b u / 3)) * 3 ≤
          ∑ u : Fin T, tokWeight Finset.univ a btilde u := by
        rw [Finset.sum_mul]
        refine Finset.sum_le_sum fun u _ => ?_
        have hw := le_of_lt (tokWeight_pos Finset.univ a btilde u)
        nlinarith [(hb u).2]
      unfold softminDenom
      linarith [Real.exp_pos (-(2 * btilde))]
  -- Layer 1 unit norm
  have hX1_norm : ∀ s, ∑ j : Fin 9, (X1 s j) ^ 2 ≤ 1 := by
    intro s
    refine Fin.lastCases ?_ ?_ s
    · rw [hL1_cls]
      simp only [Fin.sum_univ_succ, Fin.isValue, Matrix.cons_val_zero, Matrix.cons_val_succ,
        Matrix.cons_val_fin_one, Fin.sum_univ_zero]
      have h5 := havg_bound (fun u => P u 0) (fun u => P u 2) (fun u => hP u 2)
      nlinarith [hz0.1, hz0.2, h5.1, h5.2]
    · intro t
      rw [hL1_tok t]
      simp only [Fin.sum_univ_succ, Fin.isValue, Matrix.cons_val_zero, Matrix.cons_val_succ,
        Matrix.cons_val_fin_one, Fin.sum_univ_zero]
      have h0 := hP t 0; have h1 := hP t 1; have h2 := hP t 2
      have h3 := havg_bound (fun u => P u 2) (fun u => P u 0) (fun u => hP u 0)
      nlinarith [hz2.1, hz2.2, h3.1, h3.2]
  -- Layer 2: for input tokens t, Q2(X1 t) = 0, so uniform weights over all T+1 tokens!
  -- For CLS token, Q2(X1 cls) = (1/3, 1/3, 1/3, 0...), giving exact softmin on task 1!
  have hL2_cls_coords :
      X2 (Fin.last T) 3 = z0 / 3 ∧
      X2 (Fin.last T) 4 = z1 / 3 ∧
      X2 (Fin.last T) 6 =
        ((∑ u : Fin T, tokWeight Finset.univ (fun u => P u 1) btilde u) * (z2 / 3) +
         Real.exp (-(2 * btilde)) *
           ((∑ u : Fin T, tokWeight Finset.univ (fun u => P u 0) btilde u * (P u 2 / 3)) /
            softminDenom Finset.univ (fun u => P u 0) btilde)) /
          softminDenom Finset.univ (fun u => P u 1) btilde := by
    have hdot_u : ∀ u : Fin T,
        β * dotProduct (WQ2.mulVec (X1 (Fin.last T))) (WK2.mulVec (X1 (Fin.castSucc u))) =
          -(btilde * P u 1) := by
      intro u
      rw [WQ2_mulVec, WK2_mulVec, hL1_cls, hL1_tok u]
      simp [β, dotProduct, Fin.sum_univ_succ]
      ring
    have hdot_cls :
        β * dotProduct (WQ2.mulVec (X1 (Fin.last T))) (WK2.mulVec (X1 (Fin.last T))) =
          -(2 * btilde) := by
      rw [WQ2_mulVec, WK2_mulVec, hL1_cls]
      simp [β, dotProduct, Fin.sum_univ_succ]
      ring
    have hZeq : (∑ s : Fin (T + 1),
        Real.exp (β * dotProduct (WQ2.mulVec (X1 (Fin.last T))) (WK2.mulVec (X1 s)))) =
        softminDenom Finset.univ (fun u => P u 1) btilde := by
      rw [Fin.sum_univ_castSucc, hdot_cls]
      simp_rw [hdot_u]
      simp [softminDenom, tokWeight]
    have hattn0 : (∑ s : Fin (T + 1),
        Real.exp (β * dotProduct (WQ2.mulVec (X1 (Fin.last T))) (WK2.mulVec (X1 s))) *
          WV2.mulVec (X1 s) 0) / softminDenom Finset.univ (fun u => P u 1) btilde =
        z1 / 3 := by
      rw [Fin.sum_univ_castSucc, hdot_cls]
      simp_rw [hdot_u, WV2_mulVec, hL1_tok, hL1_cls]
      have hpull : (∑ x : Fin T, Real.exp (-(btilde * P x 1)) * (P x 1 / 3)) =
          (∑ x : Fin T, Real.exp (-(btilde * P x 1)) * P x 1) / 3 := by
        rw [Finset.sum_div]; refine Finset.sum_congr rfl fun x _ => by ring
      simp only [z1, softminAvg, tokWeight, Finset.mem_univ, ↓reduceIte]
      change (∑ x : Fin T, Real.exp (-(btilde * P x 1)) * (P x 1 / 3) +
          Real.exp (-(2 * btilde)) * 0) / softminDenom Finset.univ (fun u => P u 1) btilde = _
      rw [mul_zero, add_zero, hpull]; ring
    have hattn1 : (∑ s : Fin (T + 1),
        Real.exp (β * dotProduct (WQ2.mulVec (X1 (Fin.last T))) (WK2.mulVec (X1 s))) *
          WV2.mulVec (X1 s) 1) / softminDenom Finset.univ (fun u => P u 1) btilde =
        ((∑ u : Fin T, tokWeight Finset.univ (fun u => P u 1) btilde u) * (z2 / 3) +
         Real.exp (-(2 * btilde)) *
           ((∑ u : Fin T, tokWeight Finset.univ (fun u => P u 0) btilde u * (P u 2 / 3)) /
            softminDenom Finset.univ (fun u => P u 0) btilde)) /
          softminDenom Finset.univ (fun u => P u 1) btilde := by
      rw [Fin.sum_univ_castSucc, hdot_cls]
      simp_rw [hdot_u, WV2_mulVec, hL1_tok, hL1_cls]
      change (∑ x : Fin T, Real.exp (-(btilde * P x 1)) * (z2 / 3) +
          Real.exp (-(2 * btilde)) *
            ((∑ u : Fin T, tokWeight Finset.univ (fun u => P u 0) btilde u * (P u 2 / 3)) /
              softminDenom Finset.univ (fun u => P u 0) btilde)) /
          softminDenom Finset.univ (fun u => P u 1) btilde = _
      simp only [tokWeight, Finset.mem_univ, ↓reduceIte, ← Finset.sum_mul]
    refine ⟨?_, ?_, ?_⟩
    · -- slot 3 unchanged by WO2
      simp only [X2, attnLayer, WO2_mulVec, Pi.add_apply]
      rw [hL1_cls]; simp
    · -- slot 4 = z1 / 3
      simp only [X2, attnLayer, hZeq, WO2_mulVec, Pi.add_apply]
      change X1 (Fin.last T) 4 +
          ((∑ s : Fin (T + 1),
            Real.exp (β * dotProduct (WQ2.mulVec (X1 (Fin.last T))) (WK2.mulVec (X1 s))) *
              WV2.mulVec (X1 s) 0) / softminDenom Finset.univ (fun u => P u 1) btilde) = z1 / 3
      rw [hattn0, hL1_cls]; simp
    · -- slot 6 = piggyback readout of z2 / 3
      simp only [X2, attnLayer, hZeq, WO2_mulVec, Pi.add_apply]
      change X1 (Fin.last T) 6 +
          ((∑ s : Fin (T + 1),
            Real.exp (β * dotProduct (WQ2.mulVec (X1 (Fin.last T))) (WK2.mulVec (X1 s))) *
              WV2.mulVec (X1 s) 1) / softminDenom Finset.univ (fun u => P u 1) btilde) = _
      rw [hattn1, hL1_cls]; simp
  -- Layer 2 unit norm for all tokens (every added slot 4 or 6 is a convex combination of values in [0, 1/3]!)
  have hX2_norm : ∀ s, ∑ j : Fin 9, (X2 s j) ^ 2 ≤ 1 := by
    intro s
    have hcoord_bound : ∀ j : Fin 9, 0 ≤ X2 s j ∧ X2 s j ≤ 1 / 3 := by
      -- First show 0 ≤ X1 t j ≤ 1/3 for all t, j
      have hX1_coord : ∀ t : Fin (T + 1), ∀ j : Fin 9, 0 ≤ X1 t j ∧ X1 t j ≤ 1 / 3 := by
        intro t j
        refine Fin.lastCases ?_ ?_ t
        · rw [hL1_cls]
          have h5 := havg_bound (fun u => P u 0) (fun u => P u 2) (fun u => hP u 2)
          fin_cases j
          · simp
          · simp
          · simp
          · simp; exact ⟨by linarith [hz0.1], by linarith [hz0.2]⟩
          · simp
          · simpa using h5
          · simp
          · simp
          · simp
        · intro u
          rw [hL1_tok u]
          have h0 := hP u 0; have h1 := hP u 1; have h2 := hP u 2
          have h3 := havg_bound (fun v => P v 2) (fun v => P v 0) (fun v => hP v 0)
          fin_cases j
          · simp; exact ⟨by linarith [h0.1], by linarith [h0.2]⟩
          · simp; exact ⟨by linarith [h1.1], by linarith [h1.2]⟩
          · simp; exact ⟨by linarith [h2.1], by linarith [h2.2]⟩
          · simpa using h3
          · simp
          · simp; exact ⟨by linarith [hz2.1], by linarith [hz2.2]⟩
          · simp
          · simp
          · simp
      -- Softmax weights in Layer 2 are strictly positive and sum to Z > 0
      set w2 : Fin (T + 1) → ℝ :=
        fun t => Real.exp (β * dotProduct (WQ2.mulVec (X1 s)) (WK2.mulVec (X1 t)))
      have hw2_pos : ∀ t, 0 < w2 t := fun t => Real.exp_pos _
      have hZ2_pos : 0 < ∑ t : Fin (T + 1), w2 t :=
        Finset.sum_pos (fun t _ => hw2_pos t) ⟨Fin.last T, Finset.mem_univ _⟩
      -- Any weighted average of a slot of X1 lies in [0, 1/3]
      have hconv : ∀ k : Fin 9,
          0 ≤ (∑ t : Fin (T + 1), w2 t * X1 t k) / (∑ t : Fin (T + 1), w2 t) ∧
          (∑ t : Fin (T + 1), w2 t * X1 t k) / (∑ t : Fin (T + 1), w2 t) ≤ 1 / 3 := by
        intro k
        constructor
        · apply div_nonneg _ (le_of_lt hZ2_pos)
          exact Finset.sum_nonneg fun t _ =>
            mul_nonneg (le_of_lt (hw2_pos t)) (hX1_coord t k).1
        · rw [div_le_iff₀ hZ2_pos, Finset.mul_sum]
          refine Finset.sum_le_sum fun t _ => ?_
          have hw := le_of_lt (hw2_pos t)
          nlinarith [(hX1_coord t k).2]
      -- X1 s 4 = 0 and X1 s 6 = 0 for all s
      have hX1_4 : X1 s 4 = 0 := by
        refine Fin.lastCases ?_ ?_ s
        · rw [hL1_cls]; simp
        · intro u; rw [hL1_tok u]; simp
      have hX1_6 : X1 s 6 = 0 := by
        refine Fin.lastCases ?_ ?_ s
        · rw [hL1_cls]; simp
        · intro u; rw [hL1_tok u]; simp
      intro j
      simp only [X2, attnLayer, WO2_mulVec, Pi.add_apply]
      fin_cases j
      · simpa using hX1_coord s 0
      · simpa using hX1_coord s 1
      · simpa using hX1_coord s 2
      · simpa using hX1_coord s 3
      · -- slot 4: X1 s 4 (=0) + avg of X1 t 1
        simp only [WV2_mulVec]
        simpa [hX1_4, w2] using hconv 1
      · simpa using hX1_coord s 5
      · -- slot 6: X1 s 6 (=0) + avg of X1 t 5
        simp only [WV2_mulVec]
        simpa [hX1_6, w2] using hconv 5
      · simpa using hX1_coord s 7
      · simpa using hX1_coord s 8
    calc ∑ j : Fin 9, (X2 s j) ^ 2
        ≤ ∑ _j : Fin 9, ((1 / 3 : ℝ) ^ 2) := by
          refine Finset.sum_le_sum fun j _ => ?_
          have h := hcoord_bound j
          nlinarith
      _ = 1 := by simp; norm_num
  -- Finally, bound the readout error for all three tasks!
  refine ⟨nopeEmbed_unit_norm P hP, hX1_norm, hX2_norm, ?_⟩
  rcases hL2_cls_coords with ⟨h3, h4, h6⟩
  have he0 := softmin_error_bound_univ hT (fun t => P t 0) (fun t => hP t 0) hbt
  have he1 := softmin_error_bound_univ hT (fun t => P t 1) (fun t => hP t 1) hbt
  have he2 := softmin_error_bound_univ hT (fun t => P t 2) (fun t => hP t 2) hbt
  -- Show |3 * X2 (last T) 6 - z2| ≤ exp (-btilde)
  have h6_err : |3 * X2 (Fin.last T) 6 - z2| ≤ Real.exp (-btilde) := by
    rw [h6]
    set S1 := ∑ u : Fin T, tokWeight Finset.univ (fun u => P u 1) btilde u
    set Z1 := softminDenom Finset.univ (fun u => P u 1) btilde
    set e2b := Real.exp (-(2 * btilde))
    set c_cls := (∑ u : Fin T, tokWeight Finset.univ (fun u => P u 0) btilde u * (P u 2 / 3)) /
        softminDenom Finset.univ (fun u => P u 0) btilde
    have hZ1_eq : Z1 = S1 + e2b := rfl
    have hZ1_pos : 0 < Z1 := softminDenom_pos _ _ _
    have hc_Icc := havg_bound (fun u => P u 0) (fun u => P u 2) (fun u => hP u 2)
    have hdiff : 3 * ((S1 * (z2 / 3) + e2b * c_cls) / Z1) - z2 =
        e2b * (3 * c_cls - z2) / Z1 := by
      field_simp [ne_of_gt hZ1_pos]
      rw [hZ1_eq]; ring
    rw [hdiff, abs_div, abs_of_pos hZ1_pos, abs_mul, abs_of_pos (Real.exp_pos _), div_le_iff₀ hZ1_pos]
    have hdiff_le1 : |3 * c_cls - z2| ≤ 1 := by
      rw [abs_le]; constructor <;> linarith [hc_Icc.1, hc_Icc.2, hz2.1, hz2.2]
    -- Lower bound Z1 ≥ exp(-btilde) (since P 0 1 ≤ 1)
    have hZ1_ge : Real.exp (-btilde) ≤ Z1 := by
      have hterm := Finset.single_le_sum (fun u _ => le_of_lt (tokWeight_pos Finset.univ (fun u => P u 1) btilde u))
        (Finset.mem_univ (⟨0, hT⟩ : Fin T))
      have htw : Real.exp (-btilde) ≤ tokWeight Finset.univ (fun u => P u 1) btilde ⟨0, hT⟩ := by
        simp only [tokWeight, Finset.mem_univ, ↓reduceIte]
        apply Real.exp_le_exp.mpr
        nlinarith [(hP ⟨0, hT⟩ 1).2]
      unfold Z1 softminDenom
      linarith [Real.exp_pos (-(2 * btilde))]
    have he2b_eq : e2b = Real.exp (-btilde) * Real.exp (-btilde) := by
      simp only [e2b, ← Real.exp_add]; congr 1; ring
    calc e2b * |3 * c_cls - z2|
        ≤ e2b * 1 := mul_le_mul_of_nonneg_left hdiff_le1 (le_of_lt (Real.exp_pos _))
      _ = Real.exp (-btilde) * Real.exp (-btilde) := by rw [mul_one, he2b_eq]
      _ ≤ Real.exp (-btilde) * Z1 := mul_le_mul_of_nonneg_left hZ1_ge (le_of_lt (Real.exp_pos _))
  -- Combine all four triangle-inequality pieces!
  have hrewrite :
      3 * (X2 (Fin.last T) 3 + X2 (Fin.last T) 4 + X2 (Fin.last T) 6) -
        (Finset.univ.inf' hne (fun t => P t 0) +
         Finset.univ.inf' hne (fun t => P t 1) +
         Finset.univ.inf' hne (fun t => P t 2)) =
      (z0 - Finset.univ.inf' hne (fun t => P t 0)) +
      (z1 - Finset.univ.inf' hne (fun t => P t 1)) +
      (z2 - Finset.univ.inf' hne (fun t => P t 2)) +
      (3 * X2 (Fin.last T) 6 - z2) := by
    rw [h3, h4]; ring
  rw [hrewrite]
  have htri1 := abs_add_le
    ((z0 - Finset.univ.inf' hne (fun t => P t 0)) +
     (z1 - Finset.univ.inf' hne (fun t => P t 1)) +
     (z2 - Finset.univ.inf' hne (fun t => P t 2)))
    (3 * X2 (Fin.last T) 6 - z2)
  have htri2 := abs_add_le
    ((z0 - Finset.univ.inf' hne (fun t => P t 0)) +
     (z1 - Finset.univ.inf' hne (fun t => P t 1)))
    (z2 - Finset.univ.inf' hne (fun t => P t 2))
  have htri3 := abs_add_le
    (z0 - Finset.univ.inf' hne (fun t => P t 0))
    (z1 - Finset.univ.inf' hne (fun t => P t 1))
  have hexp_nn : 0 ≤ Real.exp (-btilde) := le_of_lt (Real.exp_pos _)
  linarith

/-! ### Part 3: Lemma 1 — Piecewise-Linear ReLU Squaring Error Bound -/

/-- Explicit 2-layer ReLU piecewise-linear interpolant of `(u - c₀)^2` on uniform knots `k / N`
(`k = 0..N`) using `N + 2` ReLU neurons:
- two neurons `-c₀ * max 0 u` realizing initial slope `-2 * c₀` with `[-1, 1]`-bounded weights;
- one neuron `(1 / N) * max 0 u` giving the first secant slope `1 / N - 2 * c₀` on `[0, 1/N]`;
- `N - 1` neurons `(2 / N) * max 0 (u - k / N)` (`k = 1..N-1`) adding slope increment `2 / N` at
  each interior knot `k / N`;
- output bias `c₀^2`. -/
noncomputable def pwlSquare (N : ℕ) (c₀ u : ℝ) : ℝ :=
  c₀ ^ 2 - c₀ * max 0 u - c₀ * max 0 u + (1 / (N : ℝ)) * max 0 u +
    ∑ k ∈ Finset.Ico 1 N, (2 / (N : ℝ)) * max 0 (u - (k : ℝ) / (N : ℝ))

/-- All first-layer weights (`1`), first-layer biases (`0`, `-k / N`), second-layer weights
(`-c₀`, `1 / N`, `2 / N`), and output bias (`c₀^2`) of `pwlSquare` lie in `[-1, 1]`. -/
theorem pwlSquare_weights_bounded {N : ℕ} (hN : 2 ≤ N) {c₀ : ℝ} (hc₀ : 0 ≤ c₀ ∧ c₀ ≤ 1) :
    (-1 ≤ (1 : ℝ) ∧ (1 : ℝ) ≤ 1) ∧
    (-1 ≤ (0 : ℝ) ∧ (0 : ℝ) ≤ 1) ∧
    (∀ k ∈ Finset.Ico 1 N, -1 ≤ -((k : ℝ) / (N : ℝ)) ∧ -((k : ℝ) / (N : ℝ)) ≤ 1) ∧
    (-1 ≤ -c₀ ∧ -c₀ ≤ 1) ∧
    (-1 ≤ (1 : ℝ) / (N : ℝ) ∧ (1 : ℝ) / (N : ℝ) ≤ 1) ∧
    (-1 ≤ (2 : ℝ) / (N : ℝ) ∧ (2 : ℝ) / (N : ℝ) ≤ 1) ∧
    (-1 ≤ c₀ ^ 2 ∧ c₀ ^ 2 ≤ 1) := by
  have hN_pos : (0 : ℝ) < (N : ℝ) := by exact_mod_cast (by omega : 0 < N)
  have hN_ge2 : (2 : ℝ) ≤ (N : ℝ) := by exact_mod_cast hN
  refine ⟨by norm_num, by norm_num, ?_, ?_, ?_, ?_, ?_⟩
  · intro k hk
    have hk_le : (k : ℝ) ≤ (N : ℝ) := by exact_mod_cast le_of_lt (Finset.mem_Ico.mp hk).2
    have hk_nn : (0 : ℝ) ≤ (k : ℝ) := Nat.cast_nonneg k
    have hdiv_le : (k : ℝ) / (N : ℝ) ≤ 1 := (div_le_one hN_pos).mpr hk_le
    have hdiv_nn : 0 ≤ (k : ℝ) / (N : ℝ) := div_nonneg hk_nn (le_of_lt hN_pos)
    constructor <;> linarith
  · constructor <;> linarith [hc₀.1, hc₀.2]
  · constructor
    · linarith [div_nonneg (by norm_num : (0 : ℝ) ≤ 1) (le_of_lt hN_pos)]
    · rw [div_le_one hN_pos]; linarith
  · constructor
    · linarith [div_nonneg (by norm_num : (0 : ℝ) ≤ 2) (le_of_lt hN_pos)]
    · rw [div_le_one hN_pos]; exact hN_ge2
  · constructor <;> nlinarith [hc₀.1, hc₀.2]

private lemma sum_Ico_slope_eval (N m : ℕ) (hN : 0 < N) (u : ℝ) :
    ∑ k ∈ Finset.Ico 1 (m + 1), (2 / (N : ℝ)) * (u - (k : ℝ) / (N : ℝ)) =
      (2 * (m : ℝ) / (N : ℝ)) * u - (m : ℝ) * ((m : ℝ) + 1) / (N : ℝ) ^ 2 := by
  have hN_ne : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (ne_of_gt hN)
  induction m with
  | zero => simp
  | succ m ih =>
    rw [Finset.sum_Ico_succ_top (by omega : 1 ≤ m + 1), ih]
    push_cast
    field_simp [hN_ne]
    ring

private lemma pwlSquare_on_interval {N m : ℕ} (hN : 0 < N) (hm : m < N) (c₀ u : ℝ)
    (hu_lo : (m : ℝ) / (N : ℝ) ≤ u) (hu_hi : u ≤ ((m : ℝ) + 1) / (N : ℝ)) :
    pwlSquare N c₀ u =
      c₀ ^ 2 - 2 * c₀ * u + ((2 * (m : ℝ) + 1) / (N : ℝ)) * u -
        (m : ℝ) * ((m : ℝ) + 1) / (N : ℝ) ^ 2 := by
  have hN_pos : (0 : ℝ) < (N : ℝ) := Nat.cast_pos.mpr hN
  have hN_ne : (N : ℝ) ≠ 0 := ne_of_gt hN_pos
  have hu_nn : 0 ≤ u := le_trans (div_nonneg (Nat.cast_nonneg m) (le_of_lt hN_pos)) hu_lo
  have hsplit :
      ∑ k ∈ Finset.Ico 1 N, (2 / (N : ℝ)) * max 0 (u - (k : ℝ) / (N : ℝ)) =
        (∑ k ∈ Finset.Ico 1 (m + 1), (2 / (N : ℝ)) * max 0 (u - (k : ℝ) / (N : ℝ))) +
        (∑ k ∈ Finset.Ico (m + 1) N, (2 / (N : ℝ)) * max 0 (u - (k : ℝ) / (N : ℝ))) :=
    (Finset.sum_Ico_consecutive _ (by omega : 1 ≤ m + 1) (by omega : m + 1 ≤ N)).symm
  have hpart1 :
      ∑ k ∈ Finset.Ico 1 (m + 1), (2 / (N : ℝ)) * max 0 (u - (k : ℝ) / (N : ℝ)) =
        ∑ k ∈ Finset.Ico 1 (m + 1), (2 / (N : ℝ)) * (u - (k : ℝ) / (N : ℝ)) := by
    refine Finset.sum_congr rfl fun k hk => ?_
    have hk_le : (k : ℝ) ≤ (m : ℝ) := by exact_mod_cast Nat.le_of_lt_succ (Finset.mem_Ico.mp hk).2
    have hsub : 0 ≤ u - (k : ℝ) / (N : ℝ) := by
      have hdiv : (k : ℝ) / (N : ℝ) ≤ (m : ℝ) / (N : ℝ) :=
        div_le_div_of_nonneg_right hk_le (le_of_lt hN_pos)
      linarith
    rw [max_eq_right hsub]
  have hpart2 :
      ∑ k ∈ Finset.Ico (m + 1) N, (2 / (N : ℝ)) * max 0 (u - (k : ℝ) / (N : ℝ)) = 0 := by
    refine Finset.sum_eq_zero fun k hk => ?_
    have hk_ge : ((m : ℝ) + 1) ≤ (k : ℝ) := by exact_mod_cast (Finset.mem_Ico.mp hk).1
    have hsub : u - (k : ℝ) / (N : ℝ) ≤ 0 := by
      have hdiv : ((m : ℝ) + 1) / (N : ℝ) ≤ (k : ℝ) / (N : ℝ) :=
        div_le_div_of_nonneg_right hk_ge (le_of_lt hN_pos)
      linarith
    rw [max_eq_left hsub, mul_zero]
  unfold pwlSquare
  rw [max_eq_right hu_nn, hsplit, hpart1, hpart2, add_zero, sum_Ico_slope_eval N m hN u]
  field_simp [hN_ne]
  ring

private lemma exists_knot_interval {N : ℕ} (hN : 0 < N) {u : ℝ} (hu : 0 ≤ u ∧ u ≤ 1) :
    ∃ m < N, (m : ℝ) / (N : ℝ) ≤ u ∧ u ≤ ((m : ℝ) + 1) / (N : ℝ) := by
  have hN_pos : (0 : ℝ) < (N : ℝ) := Nat.cast_pos.mpr hN
  by_cases hu1 : u = 1
  · refine ⟨N - 1, by omega, ?_, ?_⟩
    · rw [hu1, div_le_one hN_pos]
      exact_mod_cast (by omega : N - 1 ≤ N)
    · have hcast : ((N - 1 : ℕ) : ℝ) + 1 = (N : ℝ) := by exact_mod_cast (by omega : N - 1 + 1 = N)
      rw [hu1, hcast, div_self (ne_of_gt hN_pos)]
  · have hu_lt : u < 1 := lt_of_le_of_ne hu.2 hu1
    set m := Nat.floor (u * (N : ℝ))
    have hm_lt : m < N := by
      rw [Nat.floor_lt (mul_nonneg hu.1 (le_of_lt hN_pos))]
      nlinarith
    refine ⟨m, hm_lt, ?_, ?_⟩
    · rw [div_le_iff₀ hN_pos]
      exact Nat.floor_le (mul_nonneg hu.1 (le_of_lt hN_pos))
    · rw [le_div_iff₀ hN_pos]
      exact le_of_lt (Nat.lt_floor_add_one (u * (N : ℝ)))

/-- **Lemma 1 (Piecewise-Linear ReLU Squaring Error Bound)**:
For `N ≥ 2`, `c₀ ∈ [0, 1]`, and `u ∈ [0, 1]`, `pwlSquare N c₀ u` over-approximates `(u - c₀)^2`
with uniform error at most `1 / (4 * N^2)`. -/
theorem pwlSquare_error {N : ℕ} (hN : 2 ≤ N) {c₀ u : ℝ} (_hc₀ : 0 ≤ c₀ ∧ c₀ ≤ 1)
    (hu : 0 ≤ u ∧ u ≤ 1) :
    0 ≤ pwlSquare N c₀ u - (u - c₀) ^ 2 ∧
    pwlSquare N c₀ u - (u - c₀) ^ 2 ≤ 1 / (4 * (N : ℝ) ^ 2) := by
  have hN_pos_nat : 0 < N := by omega
  have hN_pos : (0 : ℝ) < (N : ℝ) := Nat.cast_pos.mpr hN_pos_nat
  have hN_ne : (N : ℝ) ≠ 0 := ne_of_gt hN_pos
  obtain ⟨m, hm_lt, hu_lo, hu_hi⟩ := exists_knot_interval hN_pos_nat hu
  rw [pwlSquare_on_interval hN_pos_nat hm_lt c₀ u hu_lo hu_hi]
  have hfact :
      c₀ ^ 2 - 2 * c₀ * u + ((2 * (m : ℝ) + 1) / (N : ℝ)) * u -
        (m : ℝ) * ((m : ℝ) + 1) / (N : ℝ) ^ 2 - (u - c₀) ^ 2 =
      (u - (m : ℝ) / (N : ℝ)) * (((m : ℝ) + 1) / (N : ℝ) - u) := by
    field_simp [hN_ne]
    ring
  rw [hfact]
  set a := u - (m : ℝ) / (N : ℝ)
  set b := ((m : ℝ) + 1) / (N : ℝ) - u
  have ha : 0 ≤ a := sub_nonneg.mpr hu_lo
  have hb : 0 ≤ b := sub_nonneg.mpr hu_hi
  have hab : a + b = 1 / (N : ℝ) := by
    dsimp only [a, b]; field_simp [hN_ne]; ring
  constructor
  · exact mul_nonneg ha hb
  · have h4ab : 4 * (a * b) ≤ (a + b) ^ 2 := by nlinarith [sq_nonneg (a - b)]
    rw [hab] at h4ab
    have h4N2_pos : (0 : ℝ) < 4 * (N : ℝ) ^ 2 := by positivity
    rw [le_div_iff₀ h4N2_pos]
    have hsq : (1 / (N : ℝ)) ^ 2 * (N : ℝ) ^ 2 = 1 := by field_simp [hN_ne]
    nlinarith [mul_le_mul_of_nonneg_right h4ab (sq_nonneg (N : ℝ))]

/-- The piecewise-linear interpolant `pwlSquare N c₀ u` remains in `[0, 1]` whenever
`N ≥ 2`, `c₀ ∈ [0, 1]`, and `u ∈ [0, 1]`. -/
theorem pwlSquare_mem_Icc {N : ℕ} (hN : 2 ≤ N) {c₀ u : ℝ} (hc₀ : 0 ≤ c₀ ∧ c₀ ≤ 1)
    (hu : 0 ≤ u ∧ u ≤ 1) :
    0 ≤ pwlSquare N c₀ u ∧ pwlSquare N c₀ u ≤ 1 := by
  have hN_pos_nat : 0 < N := by omega
  have hN_pos : (0 : ℝ) < (N : ℝ) := Nat.cast_pos.mpr hN_pos_nat
  have hN_ne : (N : ℝ) ≠ 0 := ne_of_gt hN_pos
  constructor
  · have herr := (pwlSquare_error hN hc₀ hu).1
    nlinarith [sq_nonneg (u - c₀)]
  · obtain ⟨m, hm_lt, hu_lo, hu_hi⟩ := exists_knot_interval hN_pos_nat hu
    rw [pwlSquare_on_interval hN_pos_nat hm_lt c₀ u hu_lo hu_hi]
    set t := (N : ℝ) * u - (m : ℝ)
    have ht0 : 0 ≤ t := by
      have := (div_le_iff₀ hN_pos).mp hu_lo
      dsimp only [t]; linarith
    have ht1 : t ≤ 1 := by
      have := (le_div_iff₀ hN_pos).mp hu_hi
      dsimp only [t]; linarith
    have hconv :
        c₀ ^ 2 - 2 * c₀ * u + ((2 * (m : ℝ) + 1) / (N : ℝ)) * u -
          (m : ℝ) * ((m : ℝ) + 1) / (N : ℝ) ^ 2 =
        (1 - t) * ((m : ℝ) / (N : ℝ) - c₀) ^ 2 +
          t * (((m : ℝ) + 1) / (N : ℝ) - c₀) ^ 2 := by
      dsimp only [t]
      field_simp [hN_ne]
      ring
    rw [hconv]
    have hm_le : (m : ℝ) / (N : ℝ) ≤ 1 :=
      (div_le_one hN_pos).mpr (by exact_mod_cast le_of_lt hm_lt)
    have hm1_le : ((m : ℝ) + 1) / (N : ℝ) ≤ 1 :=
      (div_le_one hN_pos).mpr (by exact_mod_cast (by omega : m + 1 ≤ N))
    have hm_nn : 0 ≤ (m : ℝ) / (N : ℝ) := div_nonneg (Nat.cast_nonneg m) (le_of_lt hN_pos)
    have hm1_nn : 0 ≤ ((m : ℝ) + 1) / (N : ℝ) := by positivity
    have hsq1 : ((m : ℝ) / (N : ℝ) - c₀) ^ 2 ≤ 1 := by
      nlinarith [show -1 ≤ (m : ℝ) / (N : ℝ) - c₀ by linarith [hc₀.2, hm_nn],
        show (m : ℝ) / (N : ℝ) - c₀ ≤ 1 by linarith [hc₀.1, hm_le]]
    have hsq2 : (((m : ℝ) + 1) / (N : ℝ) - c₀) ^ 2 ≤ 1 := by
      nlinarith [show -1 ≤ ((m : ℝ) + 1) / (N : ℝ) - c₀ by linarith [hc₀.2, hm1_nn],
        show ((m : ℝ) + 1) / (N : ℝ) - c₀ ≤ 1 by linarith [hc₀.1, hm1_le]]
    nlinarith [show 0 ≤ 1 - t by linarith, ht0, hsq1, hsq2]

/-! ### Part 4: Lemma 2 — Neuron-Splitting Rescaling & `[-1, 1]`-Bounded Readout -/

/-- **Lemma 2 (Neuron-Splitting Rescaling)**:
For any rescaling factor `α ≥ 1` with `m = ⌈α⌉₊`, any `[-1, 1]`-bounded ReLU neuron applied to
`α • x` equals the sum of `m` identical ReLU neurons applied to `x` with rescaled output weight
`v * (α / m) ∈ [-1, 1]` and rescaled bias `b / α ∈ [-1, 1]`. -/
theorem relu_rescale_splitting {k : ℕ} {α v b : ℝ} (hα : 1 ≤ α)
    (hv : |v| ≤ 1) (hb : |b| ≤ 1) (w x : Fin k → ℝ) :
    let m : ℕ := Nat.ceil α
    v * max 0 (dotProduct w (α • x) + b) =
      ∑ _ : Fin m, (v * (α / (m : ℝ))) * max 0 (dotProduct w x + b / α) ∧
    |v * (α / (m : ℝ))| ≤ 1 ∧
    |b / α| ≤ 1 := by
  intro m
  have hα_pos : 0 < α := by linarith
  have hα_le_m : α ≤ (m : ℝ) := Nat.le_ceil α
  have hm_pos : (0 : ℝ) < (m : ℝ) := lt_of_lt_of_le hα_pos hα_le_m
  have hm_ne : (m : ℝ) ≠ 0 := ne_of_gt hm_pos
  refine ⟨?_, ?_, ?_⟩
  · have hdot_smul : dotProduct w (α • x) = α * dotProduct w x := by
      simp only [dotProduct, Pi.smul_apply, smul_eq_mul, Finset.mul_sum]
      refine Finset.sum_congr rfl fun i _ => by ring
    have hdot : dotProduct w (α • x) + b = α * (dotProduct w x + b / α) := by
      rw [hdot_smul, mul_add, mul_div_cancel₀ _ (ne_of_gt hα_pos)]
    have hmax_hom : ∀ y : ℝ, max 0 (α * y) = α * max 0 y := by
      intro y
      rcases le_total 0 y with hy | hy
      · rw [max_eq_right (mul_nonneg (le_of_lt hα_pos) hy), max_eq_right hy]
      · rw [max_eq_left (mul_nonpos_of_nonneg_of_nonpos (le_of_lt hα_pos) hy),
          max_eq_left hy, mul_zero]
    rw [hdot, hmax_hom, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    field_simp [hm_ne]
  · rw [abs_mul, abs_of_pos (div_pos hα_pos hm_pos)]
    have hdiv : α / (m : ℝ) ≤ 1 := (div_le_one hm_pos).mpr hα_le_m
    have hdiv_nn : 0 ≤ α / (m : ℝ) := le_of_lt (div_pos hα_pos hm_pos)
    nlinarith [abs_nonneg v]
  · rw [abs_div, abs_of_pos hα_pos]
    calc |b| / α ≤ 1 / α := div_le_div_of_nonneg_right hb (le_of_lt hα_pos)
      _ ≤ 1 := (div_le_one hα_pos).mpr hα

/-- Explicit 3-neuron `[-1, 1]`-bounded ReLU readout for Theorem B (all weights `1`, bias `0`). -/
noncomputable def readoutB (x : Fin 9 → ℝ) : ℝ :=
  ∑ _ : Fin 3, max 0 (x 3 + x 4 + x 6)

/-- On nonnegative coordinates `x 3, x 4, x 6 ≥ 0`, `readoutB x = 3 * (x 3 + x 4 + x 6)`. -/
theorem readoutB_eq_three_sum (x : Fin 9 → ℝ)
    (h3 : 0 ≤ x 3) (h4 : 0 ≤ x 4) (h6 : 0 ≤ x 6) :
    readoutB x = 3 * (x 3 + x 4 + x 6) := by
  unfold readoutB
  have hnn : 0 ≤ x 3 + x 4 + x 6 := by linarith
  simp only [max_eq_right hnn, Fin.sum_univ_three]
  ring

/-! ### Part 5: End-to-End Theorem B (`L = 2, h = 1, D = 3`, NoPE + NoLN + `[-1, 1]` Weights) -/

/-- Exact 3-task retrieval target `H3(X) = ∑_{i=0}^2 min_t (1/d) ∑_r (X t r - (i+1)/4)^2`. -/
noncomputable def H3 {d T : ℕ} (hT : 0 < T) (X : Fin T → Fin d → ℝ) : ℝ :=
  ∑ i : Fin 3,
    Finset.univ.inf' ⟨⟨0, hT⟩, Finset.mem_univ _⟩
      (fun t => (∑ r : Fin d, (X t r - ((i : ℕ) + 1 : ℝ) / 4) ^ 2) / (d : ℝ))

/-- 2-layer ReLU FFN feature approximator for task `i : Fin 3`, using `d * (N + 2)` ReLU neurons
with all weights and biases in `[-1, 1]` (completely independent of sequence length `T`). -/
noncomputable def approxFeat (d N : ℕ) (i : Fin 3) (x : Fin d → ℝ) : ℝ :=
  (∑ r : Fin d, pwlSquare N (((i : ℕ) + 1 : ℝ) / 4) (x r)) / (d : ℝ)

private lemma center_mem_Icc (i : Fin 3) :
    0 ≤ ((i : ℕ) + 1 : ℝ) / 4 ∧ ((i : ℕ) + 1 : ℝ) / 4 ≤ 1 := by
  fin_cases i <;> norm_num

private lemma approxFeat_mem_Icc {d N : ℕ} (hd : 1 ≤ d) (hN : 2 ≤ N) (i : Fin 3)
    {x : Fin d → ℝ} (hx : ∀ r, 0 ≤ x r ∧ x r ≤ 1) :
    0 ≤ approxFeat d N i x ∧ approxFeat d N i x ≤ 1 := by
  have hd_pos : (0 : ℝ) < (d : ℝ) := Nat.cast_pos.mpr (by omega)
  unfold approxFeat
  constructor
  · apply div_nonneg _ (le_of_lt hd_pos)
    exact Finset.sum_nonneg fun r _ => (pwlSquare_mem_Icc hN (center_mem_Icc i) (hx r)).1
  · rw [div_le_one hd_pos]
    calc ∑ r : Fin d, pwlSquare N (((i : ℕ) + 1 : ℝ) / 4) (x r)
        ≤ ∑ _r : Fin d, (1 : ℝ) :=
          Finset.sum_le_sum fun r _ => (pwlSquare_mem_Icc hN (center_mem_Icc i) (hx r)).2
      _ = (d : ℝ) := by simp

private lemma approxFeat_error {d N : ℕ} (hd : 1 ≤ d) (hN : 2 ≤ N) (i : Fin 3)
    {x : Fin d → ℝ} (hx : ∀ r, 0 ≤ x r ∧ x r ≤ 1) :
    |approxFeat d N i x - (∑ r : Fin d, (x r - ((i : ℕ) + 1 : ℝ) / 4) ^ 2) / (d : ℝ)| ≤
      1 / (4 * (N : ℝ) ^ 2) := by
  have hd_pos : (0 : ℝ) < (d : ℝ) := Nat.cast_pos.mpr (by omega)
  have hdiff :
      approxFeat d N i x - (∑ r : Fin d, (x r - ((i : ℕ) + 1 : ℝ) / 4) ^ 2) / (d : ℝ) =
      (∑ r : Fin d, (pwlSquare N (((i : ℕ) + 1 : ℝ) / 4) (x r) -
        (x r - ((i : ℕ) + 1 : ℝ) / 4) ^ 2)) / (d : ℝ) := by
    unfold approxFeat
    rw [← sub_div, ← Finset.sum_sub_distrib]
  rw [hdiff]
  have hsum_nn :
      0 ≤ ∑ r : Fin d, (pwlSquare N (((i : ℕ) + 1 : ℝ) / 4) (x r) -
        (x r - ((i : ℕ) + 1 : ℝ) / 4) ^ 2) :=
    Finset.sum_nonneg fun r _ => (pwlSquare_error hN (center_mem_Icc i) (hx r)).1
  rw [abs_of_nonneg (div_nonneg hsum_nn (le_of_lt hd_pos)), div_le_iff₀ hd_pos]
  calc ∑ r : Fin d, (pwlSquare N (((i : ℕ) + 1 : ℝ) / 4) (x r) -
          (x r - ((i : ℕ) + 1 : ℝ) / 4) ^ 2)
      ≤ ∑ _r : Fin d, (1 / (4 * (N : ℝ) ^ 2)) :=
        Finset.sum_le_sum fun r _ => (pwlSquare_error hN (center_mem_Icc i) (hx r)).2
    _ = 1 / (4 * (N : ℝ) ^ 2) * (d : ℝ) := by simp [mul_comm]

private lemma abs_inf'_sub_inf'_le {α : Type*} (S : Finset α) (hS : S.Nonempty)
    (f g : α → ℝ) {δ : ℝ} (hfg : ∀ x ∈ S, |f x - g x| ≤ δ) :
    |S.inf' hS f - S.inf' hS g| ≤ δ := by
  obtain ⟨xf, hxf_mem, hxf_eq⟩ := Finset.exists_mem_eq_inf' hS f
  obtain ⟨xg, hxg_mem, hxg_eq⟩ := Finset.exists_mem_eq_inf' hS g
  have h1 : S.inf' hS f ≤ S.inf' hS g + δ := by
    have hle := Finset.inf'_le f hxg_mem
    have hdiff := (abs_le.mp (hfg xg hxg_mem)).2
    linarith
  have h2 : S.inf' hS g ≤ S.inf' hS f + δ := by
    have hle := Finset.inf'_le g hxf_mem
    have hdiff := (abs_le.mp (hfg xf hxf_mem)).1
    linarith
  rw [abs_le]
  constructor <;> linarith

private lemma attnLayer_nonneg {T E : ℕ} {WQ WK WV WO : Matrix (Fin E) (Fin E) ℝ} {β : ℝ}
    {X : Fin (T + 1) → Fin E → ℝ}
    (hWV : ∀ i j, 0 ≤ WV i j) (hWO : ∀ i j, 0 ≤ WO i j) (hX : ∀ s j, 0 ≤ X s j)
    (s : Fin (T + 1)) (j : Fin E) :
    0 ≤ attnLayer WQ WK WV WO β X s j := by
  unfold attnLayer
  dsimp only [Pi.add_apply]
  have hZ_pos : 0 < ∑ t : Fin (T + 1),
      Real.exp (β * dotProduct (WQ.mulVec (X s)) (WK.mulVec (X t))) :=
    Finset.sum_pos (fun t _ => Real.exp_pos _) ⟨Fin.last T, Finset.mem_univ _⟩
  have hattn_nn : ∀ i : Fin E,
      0 ≤ (∑ t : Fin (T + 1),
        Real.exp (β * dotProduct (WQ.mulVec (X s)) (WK.mulVec (X t))) *
          WV.mulVec (X t) i) /
        ∑ t : Fin (T + 1),
          Real.exp (β * dotProduct (WQ.mulVec (X s)) (WK.mulVec (X t))) := by
    intro i
    apply div_nonneg _ (le_of_lt hZ_pos)
    refine Finset.sum_nonneg fun t _ => mul_nonneg (le_of_lt (Real.exp_pos _)) ?_
    rw [Matrix.mulVec, dotProduct]
    exact Finset.sum_nonneg fun k _ => mul_nonneg (hWV i k) (hX t k)
  have hWO_nn : 0 ≤ WO.mulVec (fun i =>
      (∑ t : Fin (T + 1),
        Real.exp (β * dotProduct (WQ.mulVec (X s)) (WK.mulVec (X t))) *
          WV.mulVec (X t) i) /
        ∑ t : Fin (T + 1),
          Real.exp (β * dotProduct (WQ.mulVec (X s)) (WK.mulVec (X t)))) j := by
    rw [Matrix.mulVec, dotProduct]
    exact Finset.sum_nonneg fun k _ => mul_nonneg (hWO j k) (hattn_nn k)
  linarith [hX s j]

/-- **End-to-End Theorem B (`L = 2, h = 1, D = 3`, NoPE + NoLN + `[-1, 1]` Weights)**:
For every input dimension `d ≥ 1` and accuracy `ε > 0`, there is a fixed knot resolution `N ≥ 2`
(independent of `T`, giving `d * (N + 2)` ReLU neurons per feature) such that for every sequence
length `T ≥ 1`, there is an inverse temperature `β > 0` for which the strictly NoPE 2-layer
1-head transformer `approxFeat → nopeEmbed → Layer 1 → Layer 2 → readoutB` satisfies:
1. Unit-norm hidden states `∑ j, (X^(ℓ) s j)^2 ≤ 1` at Layers 0, 1, and 2;
2. Uniform approximation `|readoutB (X2 (Fin.last T)) - H3 hT X| < ε` for all `X ∈ [0, 1]^{T × d}`. -/
theorem theoremB_end_to_end_L2_h1_D3 :
    ∀ (d : ℕ), 1 ≤ d → ∀ (ε : ℝ), 0 < ε →
      ∃ N : ℕ, 2 ≤ N ∧
        ∀ (T : ℕ) (hT : 0 < T),
          ∃ β : ℝ, 0 < β ∧
            ∀ (X : Fin T → Fin d → ℝ),
              (∀ t r, 0 ≤ X t r ∧ X t r ≤ 1) →
              let P : Fin T → Fin 3 → ℝ := fun t i => approxFeat d N i (X t)
              let X0 := nopeEmbed P
              let X1 := attnLayer WQ1 WK1 WV1 WO1 β X0
              let X2 := attnLayer WQ2 WK2 WV2 WO2 β X1
              (∀ s, ∑ j : Fin 9, (X0 s j) ^ 2 ≤ 1) ∧
              (∀ s, ∑ j : Fin 9, (X1 s j) ^ 2 ≤ 1) ∧
              (∀ s, ∑ j : Fin 9, (X2 s j) ^ 2 ≤ 1) ∧
              |readoutB (X2 (Fin.last T)) - H3 hT X| < ε := by
  intro d hd ε hε
  refine ⟨Nat.ceil (Real.sqrt (3 / (2 * ε))) + 2, by omega, ?_⟩
  intro T hT
  set N : ℕ := Nat.ceil (Real.sqrt (3 / (2 * ε))) + 2
  have hN2 : 2 ≤ N := by omega
  set btilde : ℝ := 6 * ((T : ℝ) + 2) / ε + 1
  have hbt_pos : 0 < btilde := by positivity
  refine ⟨9 * btilde, by positivity, ?_⟩
  intro X hX P X0 X1 X2
  have hP : ∀ t i, 0 ≤ P t i ∧ P t i ≤ 1 := fun t i => approxFeat_mem_Icc hd hN2 i (hX t)
  have hthmB := theoremB_L2_h1_D3 hT P hP hbt_pos
  rcases hthmB with ⟨hX0_norm, hX1_norm, hX2_norm, hattn_err⟩
  refine ⟨hX0_norm, hX1_norm, hX2_norm, ?_⟩
  have hX0_nn : ∀ s j, 0 ≤ X0 s j := by
    intro s j
    refine Fin.lastCases ?_ ?_ s
    · fin_cases j <;> (simp [X0, nopeEmbed]; try norm_num)
    · intro t
      fin_cases j
      · simp [X0, nopeEmbed]; linarith [(hP t 0).1]
      · simp [X0, nopeEmbed]; linarith [(hP t 1).1]
      · simp [X0, nopeEmbed]; linarith [(hP t 2).1]
      · simp [X0, nopeEmbed]
      · simp [X0, nopeEmbed]
      · simp [X0, nopeEmbed]
      · simp [X0, nopeEmbed]
      · simp [X0, nopeEmbed]
      · simp [X0, nopeEmbed]
  have hWV1_nn : ∀ i j, 0 ≤ WV1 i j := fun i j => by dsimp [WV1]; split_ifs <;> norm_num
  have hWO1_nn : ∀ i j, 0 ≤ WO1 i j := fun i j => by dsimp [WO1]; split_ifs <;> norm_num
  have hWV2_nn : ∀ i j, 0 ≤ WV2 i j := fun i j => by dsimp [WV2]; split_ifs <;> norm_num
  have hWO2_nn : ∀ i j, 0 ≤ WO2 i j := fun i j => by dsimp [WO2]; split_ifs <;> norm_num
  have hX1_nn : ∀ s j, 0 ≤ X1 s j := attnLayer_nonneg hWV1_nn hWO1_nn hX0_nn
  have hX2_nn : ∀ s j, 0 ≤ X2 s j := attnLayer_nonneg hWV2_nn hWO2_nn hX1_nn
  rw [readoutB_eq_three_sum _ (hX2_nn _ 3) (hX2_nn _ 4) (hX2_nn _ 6)]
  let hne : (Finset.univ : Finset (Fin T)).Nonempty := ⟨⟨0, hT⟩, Finset.mem_univ _⟩
  let Q : Fin T → Fin 3 → ℝ := fun t i =>
    (∑ r : Fin d, (X t r - ((i : ℕ) + 1 : ℝ) / 4) ^ 2) / (d : ℝ)
  have hH3_eq : H3 hT X =
      Finset.univ.inf' hne (fun t => Q t 0) +
      Finset.univ.inf' hne (fun t => Q t 1) +
      Finset.univ.inf' hne (fun t => Q t 2) := by
    unfold H3
    rw [Fin.sum_univ_three]
  have hinf_err : ∀ i : Fin 3,
      |Finset.univ.inf' hne (fun t => P t i) - Finset.univ.inf' hne (fun t => Q t i)| ≤
        1 / (4 * (N : ℝ) ^ 2) := fun i =>
    abs_inf'_sub_inf'_le Finset.univ hne (fun t => P t i) (fun t => Q t i)
      (fun t _ => approxFeat_error hd hN2 i (hX t))
  have hffn_lt : 3 / (4 * (N : ℝ) ^ 2) < ε / 2 := by
    have hN_gt : Real.sqrt (3 / (2 * ε)) < (N : ℝ) := by
      have hceil : Real.sqrt (3 / (2 * ε)) ≤ (Nat.ceil (Real.sqrt (3 / (2 * ε))) : ℝ) :=
        Nat.le_ceil _
      have hN_cast : (N : ℝ) = (Nat.ceil (Real.sqrt (3 / (2 * ε))) : ℝ) + 2 := by
        dsimp only [N]; push_cast; ring
      linarith
    have hsqrt_nn : 0 ≤ Real.sqrt (3 / (2 * ε)) := Real.sqrt_nonneg _
    have hN_sq : 3 / (2 * ε) < (N : ℝ) ^ 2 := by
      have hsq : (Real.sqrt (3 / (2 * ε))) ^ 2 = 3 / (2 * ε) :=
        Real.sq_sqrt (by positivity)
      nlinarith
    have h4N2_pos : (0 : ℝ) < 4 * (N : ℝ) ^ 2 := by positivity
    rw [div_lt_div_iff₀ h4N2_pos (by norm_num : (0 : ℝ) < 2)]
    have hmul : 3 < 2 * ε * (N : ℝ) ^ 2 := by
      have := (div_lt_iff₀ (by positivity : (0 : ℝ) < 2 * ε)).mp hN_sq
      linarith
    linarith
  have hattn_lt : 3 * ((T : ℝ) / (Real.exp 1 * btilde) + 2 * Real.exp (-btilde)) < ε / 2 := by
    have he_ge1 : (1 : ℝ) ≤ Real.exp 1 := by linarith [Real.add_one_le_exp (1 : ℝ)]
    have hT_term : (T : ℝ) / (Real.exp 1 * btilde) ≤ (T : ℝ) / btilde := by
      apply div_le_div_of_nonneg_left (Nat.cast_nonneg T) hbt_pos
      nlinarith
    have hexp_le : Real.exp (-btilde) ≤ 1 / btilde := by
      rw [Real.exp_neg, ← one_div]
      apply one_div_le_one_div_of_le hbt_pos
      linarith [Real.add_one_le_exp btilde]
    have hsum_le : 3 * ((T : ℝ) / (Real.exp 1 * btilde) + 2 * Real.exp (-btilde)) ≤
        3 * ((T : ℝ) + 2) / btilde := by
      have h1 : 3 * ((T : ℝ) / btilde + 2 * (1 / btilde)) = 3 * ((T : ℝ) + 2) / btilde := by ring
      linarith
    have hbt_gt : 6 * ((T : ℝ) + 2) / ε < btilde := by dsimp only [btilde]; linarith
    have hdiv_lt : 3 * ((T : ℝ) + 2) / btilde < ε / 2 := by
      rw [div_lt_div_iff₀ hbt_pos (by norm_num : (0 : ℝ) < 2)]
      have := (div_lt_iff₀ hε).mp hbt_gt
      linarith
    linarith
  rw [hH3_eq]
  have hsplit :
      3 * (X2 (Fin.last T) 3 + X2 (Fin.last T) 4 + X2 (Fin.last T) 6) -
        (Finset.univ.inf' hne (fun t => Q t 0) +
         Finset.univ.inf' hne (fun t => Q t 1) +
         Finset.univ.inf' hne (fun t => Q t 2)) =
      (3 * (X2 (Fin.last T) 3 + X2 (Fin.last T) 4 + X2 (Fin.last T) 6) -
        (Finset.univ.inf' hne (fun t => P t 0) +
         Finset.univ.inf' hne (fun t => P t 1) +
         Finset.univ.inf' hne (fun t => P t 2))) +
      ((Finset.univ.inf' hne (fun t => P t 0) - Finset.univ.inf' hne (fun t => Q t 0)) +
       (Finset.univ.inf' hne (fun t => P t 1) - Finset.univ.inf' hne (fun t => Q t 1)) +
       (Finset.univ.inf' hne (fun t => P t 2) - Finset.univ.inf' hne (fun t => Q t 2))) := by
    ring
  rw [hsplit]
  have htri1 := abs_add_le
    (3 * (X2 (Fin.last T) 3 + X2 (Fin.last T) 4 + X2 (Fin.last T) 6) -
      (Finset.univ.inf' hne (fun t => P t 0) +
       Finset.univ.inf' hne (fun t => P t 1) +
       Finset.univ.inf' hne (fun t => P t 2)))
    ((Finset.univ.inf' hne (fun t => P t 0) - Finset.univ.inf' hne (fun t => Q t 0)) +
     (Finset.univ.inf' hne (fun t => P t 1) - Finset.univ.inf' hne (fun t => Q t 1)) +
     (Finset.univ.inf' hne (fun t => P t 2) - Finset.univ.inf' hne (fun t => Q t 2)))
  have htri2 := abs_add_le
    ((Finset.univ.inf' hne (fun t => P t 0) - Finset.univ.inf' hne (fun t => Q t 0)) +
     (Finset.univ.inf' hne (fun t => P t 1) - Finset.univ.inf' hne (fun t => Q t 1)))
    (Finset.univ.inf' hne (fun t => P t 2) - Finset.univ.inf' hne (fun t => Q t 2))
  have htri3 := abs_add_le
    (Finset.univ.inf' hne (fun t => P t 0) - Finset.univ.inf' hne (fun t => Q t 0))
    (Finset.univ.inf' hne (fun t => P t 1) - Finset.univ.inf' hne (fun t => Q t 1))
  have he0 := hinf_err 0; have he1 := hinf_err 1; have he2 := hinf_err 2
  have h3frac : 3 * (1 / (4 * (N : ℝ) ^ 2)) = 3 / (4 * (N : ℝ) ^ 2) := by ring
  linarith

/-! ### Part 6: Theorem A (`L = 2, h = 1`, PE + Residuals + Exact ReLU Gating, Arbitrary `D ≥ 1`) -/

/-- Coordinate slots for Theorem A with `E = 5 * D + 2` dimensions:
blocks `A, B, C, F, G : Fin D` plus scalar flags `H, I`. -/
inductive SlotA (D : ℕ) where
  | A : Fin D → SlotA D
  | B : Fin D → SlotA D
  | C : Fin D → SlotA D
  | F : Fin D → SlotA D
  | G : Fin D → SlotA D
  | H : SlotA D
  | I : SlotA D
  deriving DecidableEq

/-- Explicit bijection identifying `SlotA D` with `5` copies of `Fin D` plus `2` scalars. -/
def slotAEquiv (D : ℕ) :
    SlotA D ≃ (Fin D ⊕ Fin D ⊕ Fin D ⊕ Fin D ⊕ Fin D ⊕ Unit ⊕ Unit) where
  toFun
    | .A i => Sum.inl i
    | .B i => Sum.inr (Sum.inl i)
    | .C i => Sum.inr (Sum.inr (Sum.inl i))
    | .F i => Sum.inr (Sum.inr (Sum.inr (Sum.inl i)))
    | .G i => Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inl i))))
    | .H   => Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inl ())))))
    | .I   => Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr ())))))
  invFun
    | Sum.inl i => .A i
    | Sum.inr (Sum.inl i) => .B i
    | Sum.inr (Sum.inr (Sum.inl i)) => .C i
    | Sum.inr (Sum.inr (Sum.inr (Sum.inl i))) => .F i
    | Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inl i)))) => .G i
    | Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inl ()))))) => .H
    | Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr ()))))) => .I
  left_inv s := by cases s <;> rfl
  right_inv s := by
    rcases s with i | (i | (i | (i | (i | (u | u))))) <;> rfl

instance (D : ℕ) : Fintype (SlotA D) := Fintype.ofEquiv _ (slotAEquiv D).symm

/-- Exact embedding dimension `|SlotA D| = 5 * D + 2`. -/
theorem card_slotA (D : ℕ) : Fintype.card (SlotA D) = 5 * D + 2 := by
  rw [Fintype.card_congr (slotAEquiv D)]
  simp
  ring

private lemma sum_slotA {D : ℕ} (f : SlotA D → ℝ) :
    ∑ s : SlotA D, f s =
      (∑ i : Fin D, f (.A i)) +
      (∑ i : Fin D, f (.B i)) +
      (∑ i : Fin D, f (.C i)) +
      (∑ i : Fin D, f (.F i)) +
      (∑ i : Fin D, f (.G i)) +
      f .H + f .I := by
  rw [Fintype.sum_equiv (slotAEquiv D) f (fun y => f ((slotAEquiv D).symm y))
    (fun s => by simp)]
  simp [Fintype.sum_sum_type, slotAEquiv]
  ring

/-- Single-head attention layer with residual connection on an arbitrary finite coordinate type `ι`
(identical to `attnLayer` when `ι = Fin E`). -/
noncomputable def attnLayerOn {T : ℕ} {ι : Type*} [Fintype ι]
    (WQ WK WV WO : Matrix ι ι ℝ) (β : ℝ)
    (X : Fin (T + 1) → ι → ℝ) : Fin (T + 1) → ι → ℝ :=
  fun s =>
    let q := WQ.mulVec (X s)
    let weights := fun t : Fin (T + 1) => Real.exp (β * dotProduct q (WK.mulVec (X t)))
    let Z := ∑ t : Fin (T + 1), weights t
    let attnOut := fun i => (∑ t : Fin (T + 1), weights t * (WV.mulVec (X t) i)) / Z
    X s + WO.mulVec attnOut

/-- Entry-wise `[-1, 1]` bound on a matrix over an arbitrary index type `ι`. -/
def BoundedMatrixOn {ι : Type*} (M : Matrix ι ι ℝ) : Prop :=
  ∀ i j, -1 ≤ M i j ∧ M i j ≤ 1

/-- Scaling factor `c = 1 / √(11 * D + 3)` for Theorem A. -/
noncomputable def cA (D : ℕ) : ℝ :=
  1 / Real.sqrt (11 * (D : ℝ) + 3)

private lemma cA_pos (D : ℕ) : 0 < cA D := by
  unfold cA; positivity

private lemma cA_sq (D : ℕ) : (cA D) ^ 2 = 1 / (11 * (D : ℝ) + 3) := by
  unfold cA
  rw [div_pow, one_pow, Real.sq_sqrt (by positivity)]

/-- Positional-encoding input embedding for Theorem A (`S_i = [T]`):
- Input token `t : Fin T` stores `-c * P t i` in `A i`, `c * 1_{t = i}` in `B i`,
  `c * P t i` in `C i`, `c * 1_{t < D}` in `H`, and `0` in `F, G, I`.
- CLS token (`Fin.last T`) stores `-2 * c` in `A i`, `c` in `I`, and `0` elsewhere. -/
noncomputable def peEmbed {D T : ℕ} (P : Fin T → Fin D → ℝ) : Fin (T + 1) → SlotA D → ℝ :=
  Fin.snoc
    (fun t : Fin T => fun s : SlotA D =>
      match s with
      | .A i => -(cA D * P t i)
      | .B i => if (t : ℕ) = (i : ℕ) then cA D else 0
      | .C i => cA D * P t i
      | .F _ => 0
      | .G _ => 0
      | .H   => if (t : ℕ) < D then cA D else 0
      | .I   => 0)
    (fun s : SlotA D =>
      match s with
      | .A _ => -(2 * cA D)
      | .I   => cA D
      | _    => 0)

/-- Theorem A Layer 1 attention matrices (all entries in `{0, 1}`). -/
def WQ1A (D : ℕ) : Matrix (SlotA D) (SlotA D) ℝ := Matrix.of fun s t =>
  match s, t with
  | .A i, .B j => if i = j then 1 else 0
  | _, _ => 0

def WK1A (D : ℕ) : Matrix (SlotA D) (SlotA D) ℝ := Matrix.of fun s t =>
  match s, t with
  | .A i, .A j => if i = j then 1 else 0
  | _, _ => 0

def WV1A (D : ℕ) : Matrix (SlotA D) (SlotA D) ℝ := Matrix.of fun s t =>
  match s, t with
  | .C i, .C j => if i = j then 1 else 0
  | _, _ => 0

def WO1A (D : ℕ) : Matrix (SlotA D) (SlotA D) ℝ := Matrix.of fun s t =>
  match s, t with
  | .F i, .C j => if i = j then 1 else 0
  | _, _ => 0

/-- Theorem A Layer 1 FFN (residual + `D` ReLU gating neurons `max 0 (u_{F,i} + u_{B,i} - c)`). -/
noncomputable def ffn1A {D : ℕ} (u : SlotA D → ℝ) : SlotA D → ℝ :=
  fun s =>
    match s with
    | .G i => u (.G i) + max 0 (u (.F i) + u (.B i) - cA D)
    | _ => u s

/-- Theorem A Layer 2 attention matrices (all entries in `{0, 1}`). -/
def WQ2A (D : ℕ) : Matrix (SlotA D) (SlotA D) ℝ := Matrix.of fun s t =>
  match s, t with
  | .H, .I => 1
  | _, _ => 0

def WK2A (D : ℕ) : Matrix (SlotA D) (SlotA D) ℝ := Matrix.of fun s t =>
  match s, t with
  | .H, .H => 1
  | _, _ => 0

def WV2A (D : ℕ) : Matrix (SlotA D) (SlotA D) ℝ := Matrix.of fun s t =>
  match s, t with
  | .G i, .G j => if i = j then 1 else 0
  | _, _ => 0

def WO2A (D : ℕ) : Matrix (SlotA D) (SlotA D) ℝ := Matrix.of fun s t =>
  match s, t with
  | .G i, .G j => if i = j then 1 else 0
  | _, _ => 0

/-- All 8 attention matrices and the Layer 1 FFN gating bias `-cA D` of Theorem A are in `[-1, 1]`. -/
theorem theoremA_matrices_bounded (D : ℕ) :
    BoundedMatrixOn (WQ1A D) ∧ BoundedMatrixOn (WK1A D) ∧
    BoundedMatrixOn (WV1A D) ∧ BoundedMatrixOn (WO1A D) ∧
    BoundedMatrixOn (WQ2A D) ∧ BoundedMatrixOn (WK2A D) ∧
    BoundedMatrixOn (WV2A D) ∧ BoundedMatrixOn (WO2A D) ∧
    (-1 ≤ -(cA D) ∧ -(cA D) ≤ 1) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro s t; dsimp [WQ1A]; split <;> (try split_ifs) <;> norm_num
  · intro s t; dsimp [WK1A]; split <;> (try split_ifs) <;> norm_num
  · intro s t; dsimp [WV1A]; split <;> (try split_ifs) <;> norm_num
  · intro s t; dsimp [WO1A]; split <;> (try split_ifs) <;> norm_num
  · intro s t; dsimp [WQ2A]; split <;> norm_num
  · intro s t; dsimp [WK2A]; split <;> norm_num
  · intro s t; dsimp [WV2A]; split <;> (try split_ifs) <;> norm_num
  · intro s t; dsimp [WO2A]; split <;> (try split_ifs) <;> norm_num
  · have hc_pos : 0 < cA D := cA_pos D
    have hsqrt_ge1 : 1 ≤ Real.sqrt (11 * (D : ℝ) + 3) := by
      rw [← Real.sqrt_one]
      exact Real.sqrt_le_sqrt (by linarith [Nat.cast_nonneg (α := ℝ) D])
    have hc_le1 : cA D ≤ 1 := by
      unfold cA; exact (div_le_one (by positivity)).mpr hsqrt_ge1
    constructor <;> linarith

private lemma WQ1A_mulVec {D : ℕ} (u : SlotA D → ℝ) (s : SlotA D) :
    (WQ1A D).mulVec u s = match s with | .A i => u (.B i) | _ => 0 := by
  simp only [Matrix.mulVec, dotProduct, sum_slotA]
  cases s <;> simp [WQ1A]

private lemma WK1A_mulVec {D : ℕ} (u : SlotA D → ℝ) (s : SlotA D) :
    (WK1A D).mulVec u s = match s with | .A i => u (.A i) | _ => 0 := by
  simp only [Matrix.mulVec, dotProduct, sum_slotA]
  cases s <;> simp [WK1A]

private lemma WV1A_mulVec {D : ℕ} (u : SlotA D → ℝ) (s : SlotA D) :
    (WV1A D).mulVec u s = match s with | .C i => u (.C i) | _ => 0 := by
  simp only [Matrix.mulVec, dotProduct, sum_slotA]
  cases s <;> simp [WV1A]

private lemma WO1A_mulVec {D : ℕ} (u : SlotA D → ℝ) (s : SlotA D) :
    (WO1A D).mulVec u s = match s with | .F i => u (.C i) | _ => 0 := by
  simp only [Matrix.mulVec, dotProduct, sum_slotA]
  cases s <;> simp [WO1A]

private lemma WQ2A_mulVec {D : ℕ} (u : SlotA D → ℝ) (s : SlotA D) :
    (WQ2A D).mulVec u s = match s with | .H => u .I | _ => 0 := by
  simp only [Matrix.mulVec, dotProduct, sum_slotA]
  cases s <;> simp [WQ2A]

private lemma WK2A_mulVec {D : ℕ} (u : SlotA D → ℝ) (s : SlotA D) :
    (WK2A D).mulVec u s = match s with | .H => u .H | _ => 0 := by
  simp only [Matrix.mulVec, dotProduct, sum_slotA]
  cases s <;> simp [WK2A]

private lemma WV2A_mulVec {D : ℕ} (u : SlotA D → ℝ) (s : SlotA D) :
    (WV2A D).mulVec u s = match s with | .G i => u (.G i) | _ => 0 := by
  simp only [Matrix.mulVec, dotProduct, sum_slotA]
  cases s <;> simp [WV2A]

private lemma WO2A_mulVec {D : ℕ} (u : SlotA D → ℝ) (s : SlotA D) :
    (WO2A D).mulVec u s = match s with | .G i => u (.G i) | _ => 0 := by
  simp only [Matrix.mulVec, dotProduct, sum_slotA]
  cases s <;> simp [WO2A]

private lemma dot_Q1A_K1A {D : ℕ} (u v : SlotA D → ℝ) :
    dotProduct ((WQ1A D).mulVec u) ((WK1A D).mulVec v) =
      ∑ i : Fin D, u (.B i) * v (.A i) := by
  simp [dotProduct, sum_slotA, WQ1A_mulVec, WK1A_mulVec]

private lemma dot_Q2A_K2A {D : ℕ} (u v : SlotA D → ℝ) :
    dotProduct ((WQ2A D).mulVec u) ((WK2A D).mulVec v) = u .I * v .H := by
  simp [dotProduct, sum_slotA, WQ2A_mulVec, WK2A_mulVec]

set_option maxHeartbeats 800000 in
/-- **Theorem A (`L = 2, h = 1`, PE + Attention Residuals + Exact ReLU Gating, Arbitrary `D ≥ 1`)**:
For ANY number of tasks `D ≥ 1`, ANY sequence length `T ≥ D`, ANY feature matrix
`P : Fin T → Fin D → [0, 1]`, and ANY effective inverse temperatures `btilde1, btilde2 > 0`
(with `β₁ = (11D + 3) * btilde1`, `β₂ = (11D + 3) * btilde2`, `c = 1 / √(11D + 3)`), the 2-layer
1-head transformer `X0 → X1_attn → X1 = ffn1A X1_attn → X2` satisfies:
1. Unit-norm hidden states `∑ j, (X s j)² ≤ 1` at `X0`, `X1_attn`, `X1`, and `X2`;
2. Exact Layer 1 parallel retrieval: each token `i < D` retrieves `c * softminAvg univ (P · i) btilde1`
   into slot `F i`;
3. Exact Layer 1 ReLU gating: `X1 s (G i)` equals `c * softminAvg univ (P · i) btilde1` when
   `s = i` and `0` otherwise (isolating task `i` on token `i` and zeroing out all `s ≥ D` and CLS);
4. Exact Layer 2 CLS readout bound:
   `|(D / c) * ∑_i X2 (last T) (G i) - ∑_i min_t P t i| ≤ D * (T / (e * btilde1) + exp (-btilde1)) + (T + 1 - D) / exp btilde2`. -/
theorem theoremA_L2_h1_arbitrary_D {D T : ℕ} (hD : 1 ≤ D) (hDT : D ≤ T)
    (P : Fin T → Fin D → ℝ) (hP : ∀ t i, 0 ≤ P t i ∧ P t i ≤ 1)
    {btilde1 btilde2 : ℝ} (hbt1 : 0 < btilde1) (_hbt2 : 0 < btilde2) :
    let c := cA D
    let β1 := (11 * (D : ℝ) + 3) * btilde1
    let β2 := (11 * (D : ℝ) + 3) * btilde2
    let X0 := peEmbed P
    let X1_attn := attnLayerOn (WQ1A D) (WK1A D) (WV1A D) (WO1A D) β1 X0
    let X1 := fun s => ffn1A (X1_attn s)
    let X2 := attnLayerOn (WQ2A D) (WK2A D) (WV2A D) (WO2A D) β2 X1
    let hT : 0 < T := lt_of_lt_of_le (by omega : 0 < D) hDT
    let hne : (Finset.univ : Finset (Fin T)).Nonempty := ⟨⟨0, hT⟩, Finset.mem_univ _⟩
    (∀ s, ∑ j : SlotA D, (X0 s j) ^ 2 ≤ 1) ∧
    (∀ s, ∑ j : SlotA D, (X1_attn s j) ^ 2 ≤ 1) ∧
    (∀ s, ∑ j : SlotA D, (X1 s j) ^ 2 ≤ 1) ∧
    (∀ s, ∑ j : SlotA D, (X2 s j) ^ 2 ≤ 1) ∧
    (∀ i : Fin D,
      X1_attn (Fin.castSucc ⟨i, Nat.lt_of_lt_of_le i.2 hDT⟩) (.F i) =
        c * softminAvg Finset.univ (fun t => P t i) btilde1) ∧
    (∀ (s : Fin (T + 1)) (i : Fin D),
      X1 s (.G i) =
        if (s : ℕ) = (i : ℕ) then c * softminAvg Finset.univ (fun t => P t i) btilde1 else 0) ∧
    |(D : ℝ) / c * (∑ i : Fin D, X2 (Fin.last T) (.G i)) -
        ∑ i : Fin D, Finset.univ.inf' hne (fun t => P t i)| ≤
      (D : ℝ) * ((T : ℝ) / (Real.exp 1 * btilde1) + Real.exp (-btilde1)) +
        ((T : ℝ) + 1 - (D : ℝ)) / Real.exp btilde2 := by
  intro c β1 β2 X0 X1_attn X1 X2 hT hne
  have hc_pos : 0 < c := cA_pos D
  have hc_nn : 0 ≤ c := le_of_lt hc_pos
  have hc_sq : c ^ 2 = 1 / (11 * (D : ℝ) + 3) := cA_sq D
  have h11D_pos : (0 : ℝ) < 11 * (D : ℝ) + 3 := by positivity
  have hβ1_c2 : β1 * (c * c) = btilde1 := by
    dsimp only [β1]
    have : c * c = c ^ 2 := (sq c).symm
    rw [this, hc_sq]
    field_simp [ne_of_gt h11D_pos]
  have hβ2_c2 : β2 * (c * c) = btilde2 := by
    dsimp only [β2]
    have : c * c = c ^ 2 := (sq c).symm
    rw [this, hc_sq]
    field_simp [ne_of_gt h11D_pos]
  have hB_sq : ∀ t : Fin T,
      ∑ i : Fin D, (if (t : ℕ) = (i : ℕ) then c else 0) ^ 2 ≤ c ^ 2 := by
    intro t
    by_cases ht : (t : ℕ) < D
    · have hsum : ∑ i : Fin D, (if (t : ℕ) = (i : ℕ) then c else 0) ^ 2 = c ^ 2 := by
        rw [Finset.sum_eq_single (⟨t, ht⟩ : Fin D)]
        · simp
        · intro i _ hi
          have hne_val : (t : ℕ) ≠ (i : ℕ) := fun h => hi (Fin.ext h.symm)
          simp [hne_val]
        · intro h; exact absurd (Finset.mem_univ _) h
      linarith
    · have hsum : ∑ i : Fin D, (if (t : ℕ) = (i : ℕ) then c else 0) ^ 2 = 0 := by
        refine Finset.sum_eq_zero fun i _ => ?_
        have hne_val : (t : ℕ) ≠ (i : ℕ) := by have := i.2; omega
        simp [hne_val]
      nlinarith [sq_nonneg c]
  have hconv_Icc : ∀ (w : Fin (T + 1) → ℝ) (v : Fin (T + 1) → ℝ),
      (∀ t, 0 < w t) → (∀ t, 0 ≤ v t ∧ v t ≤ c) →
      0 ≤ (∑ t : Fin (T + 1), w t * v t) / (∑ t : Fin (T + 1), w t) ∧
      (∑ t : Fin (T + 1), w t * v t) / (∑ t : Fin (T + 1), w t) ≤ c := by
    intro w v hw hv
    have hZ : 0 < ∑ t : Fin (T + 1), w t :=
      Finset.sum_pos (fun t _ => hw t) ⟨Fin.last T, Finset.mem_univ _⟩
    constructor
    · apply div_nonneg _ (le_of_lt hZ)
      exact Finset.sum_nonneg fun t _ => mul_nonneg (le_of_lt (hw t)) (hv t).1
    · rw [div_le_iff₀ hZ, Finset.mul_sum]
      refine Finset.sum_le_sum fun t _ => ?_
      have := le_of_lt (hw t)
      nlinarith [(hv t).2]
  have hX0_C : ∀ (t : Fin (T + 1)) (i : Fin D), 0 ≤ X0 t (.C i) ∧ X0 t (.C i) ≤ c := by
    intro t i
    refine Fin.lastCases ?_ ?_ t
    · simp [X0, peEmbed]; exact hc_nn
    · intro u
      simp only [X0, peEmbed, Fin.snoc_castSucc]
      constructor
      · exact mul_nonneg hc_nn (hP u i).1
      · nlinarith [(hP u i).2]
  have hX1_attn_nonF : ∀ (s : Fin (T + 1)) (j : SlotA D),
      (∀ i, j ≠ .F i) → X1_attn s j = X0 s j := by
    intro s j hj
    unfold X1_attn attnLayerOn
    dsimp only [Pi.add_apply]
    rw [WO1A_mulVec]
    cases j with
    | F i => exact absurd rfl (hj i)
    | _ => simp
  have hX1_attn_F_Icc : ∀ (s : Fin (T + 1)) (i : Fin D),
      0 ≤ X1_attn s (.F i) ∧ X1_attn s (.F i) ≤ c := by
    intro s i
    have hX0_F : X0 s (.F i) = 0 := by
      refine Fin.lastCases ?_ ?_ s <;> simp [X0, peEmbed]
    unfold X1_attn attnLayerOn
    dsimp only [Pi.add_apply]
    rw [hX0_F, zero_add, WO1A_mulVec]
    simp only [WV1A_mulVec]
    exact hconv_Icc _ (fun t => X0 t (.C i)) (fun _ => Real.exp_pos _) (fun t => hX0_C t i)
  have hL1_ret : ∀ i : Fin D,
      X1_attn (Fin.castSucc ⟨i, Nat.lt_of_lt_of_le i.2 hDT⟩) (.F i) =
        c * softminAvg Finset.univ (fun t => P t i) btilde1 := by
    intro i
    set si : Fin T := ⟨i, Nat.lt_of_lt_of_le i.2 hDT⟩
    have hdot_u : ∀ u : Fin T,
        β1 * dotProduct ((WQ1A D).mulVec (X0 (Fin.castSucc si)))
          ((WK1A D).mulVec (X0 (Fin.castSucc u))) = -(btilde1 * P u i) := by
      intro u
      rw [dot_Q1A_K1A]
      have hsum : ∑ k : Fin D, X0 (Fin.castSucc si) (.B k) * X0 (Fin.castSucc u) (.A k) =
          c * (-(c * P u i)) := by
        rw [Finset.sum_eq_single i]
        · simp only [X0, peEmbed, Fin.snoc_castSucc]; simp [si, c]
        · intro k _ hk
          have hne_val : (si : ℕ) ≠ (k : ℕ) := fun h => hk (Fin.ext h.symm)
          simp only [X0, peEmbed, Fin.snoc_castSucc]; simp [hne_val]
        · intro h; exact absurd (Finset.mem_univ _) h
      rw [hsum]
      calc β1 * (c * -(c * P u i)) = -(β1 * (c * c) * P u i) := by ring
        _ = -(btilde1 * P u i) := by rw [hβ1_c2]
    have hdot_cls :
        β1 * dotProduct ((WQ1A D).mulVec (X0 (Fin.castSucc si)))
          ((WK1A D).mulVec (X0 (Fin.last T))) = -(2 * btilde1) := by
      rw [dot_Q1A_K1A]
      have hsum : ∑ k : Fin D, X0 (Fin.castSucc si) (.B k) * X0 (Fin.last T) (.A k) =
          c * (-(2 * c)) := by
        rw [Finset.sum_eq_single i]
        · simp only [X0, peEmbed, Fin.snoc_castSucc, Fin.snoc_last]; simp [si, c]
        · intro k _ hk
          have hne_val : (si : ℕ) ≠ (k : ℕ) := fun h => hk (Fin.ext h.symm)
          simp only [X0, peEmbed, Fin.snoc_castSucc, Fin.snoc_last]; simp [hne_val]
        · intro h; exact absurd (Finset.mem_univ _) h
      rw [hsum]
      calc β1 * (c * -(2 * c)) = -(2 * (β1 * (c * c))) := by ring
        _ = -(2 * btilde1) := by rw [hβ1_c2]
    have hZeq :
        (∑ t : Fin (T + 1),
          Real.exp (β1 * dotProduct ((WQ1A D).mulVec (X0 (Fin.castSucc si)))
            ((WK1A D).mulVec (X0 t)))) =
        softminDenom Finset.univ (fun u => P u i) btilde1 := by
      rw [Fin.sum_univ_castSucc, hdot_cls]
      simp_rw [hdot_u]
      simp [softminDenom, tokWeight]
    have hnum_eq :
        (∑ t : Fin (T + 1),
          Real.exp (β1 * dotProduct ((WQ1A D).mulVec (X0 (Fin.castSucc si)))
            ((WK1A D).mulVec (X0 t))) * X0 t (.C i)) =
        c * ∑ u : Fin T, tokWeight Finset.univ (fun u => P u i) btilde1 u * P u i := by
      rw [Fin.sum_univ_castSucc, hdot_cls]
      simp_rw [hdot_u]
      simp only [X0, peEmbed, Fin.snoc_castSucc, Fin.snoc_last, mul_zero, add_zero,
        tokWeight, Finset.mem_univ, ↓reduceIte, Finset.mul_sum]
      refine Finset.sum_congr rfl fun u _ => by dsimp only [c]; ring
    unfold X1_attn attnLayerOn
    dsimp only [Pi.add_apply]
    have hX0_F : X0 (Fin.castSucc si) (.F i) = 0 := by simp [X0, peEmbed]
    rw [hX0_F, zero_add, WO1A_mulVec]
    simp only [WV1A_mulVec]
    rw [hZeq, hnum_eq]
    unfold softminAvg
    ring
  have hL1_gate : ∀ (s : Fin (T + 1)) (i : Fin D),
      X1 s (.G i) =
        if (s : ℕ) = (i : ℕ) then c * softminAvg Finset.univ (fun t => P t i) btilde1 else 0 := by
    intro s i
    have hG0 : X1_attn s (.G i) = 0 := by
      rw [hX1_attn_nonF s (.G i) (fun _ h => by cases h)]
      refine Fin.lastCases ?_ ?_ s <;> simp [X0, peEmbed]
    have hB_eq : X1_attn s (.B i) = X0 s (.B i) :=
      hX1_attn_nonF s (.B i) (fun _ h => by cases h)
    dsimp only [X1, ffn1A]
    rw [hG0, zero_add, hB_eq]
    refine Fin.lastCases ?_ ?_ s
    · have hne_last : (Fin.last T : ℕ) ≠ (i : ℕ) := by simp; have := i.2; omega
      rw [if_neg hne_last]
      have hB_last : X0 (Fin.last T) (.B i) = 0 := by simp [X0, peEmbed]
      rw [hB_last, add_zero]
      exact max_eq_left (sub_nonpos.mpr (hX1_attn_F_Icc (Fin.last T) i).2)
    · intro u
      simp only [Fin.coe_castSucc]
      by_cases hui : (u : ℕ) = (i : ℕ)
      · rw [if_pos hui]
        have hu_eq : u = ⟨i, Nat.lt_of_lt_of_le i.2 hDT⟩ := Fin.ext hui
        have hB_u : X0 (Fin.castSucc u) (.B i) = c := by simp [X0, peEmbed, hui, c]
        rw [hB_u, hu_eq, hL1_ret i]
        have hnn : 0 ≤ c * softminAvg Finset.univ (fun t => P t i) btilde1 :=
          mul_nonneg hc_nn (softminAvg_mem_Icc _ _ _ (fun t => hP t i)).1
        have hsub : c * softminAvg Finset.univ (fun t => P t i) btilde1 + c - c =
            c * softminAvg Finset.univ (fun t => P t i) btilde1 := by ring
        rw [hsub, max_eq_right hnn]
      · rw [if_neg hui]
        have hB_u : X0 (Fin.castSucc u) (.B i) = 0 := by simp [X0, peEmbed, hui]
        rw [hB_u, add_zero]
        exact max_eq_left (sub_nonpos.mpr (hX1_attn_F_Icc (Fin.castSucc u) i).2)
  have hX1_G_Icc : ∀ (s : Fin (T + 1)) (i : Fin D), 0 ≤ X1 s (.G i) ∧ X1 s (.G i) ≤ c := by
    intro s i
    rw [hL1_gate s i]
    split_ifs
    · have hz := softminAvg_mem_Icc Finset.univ (fun t => P t i) btilde1 (fun t => hP t i)
      exact ⟨mul_nonneg hc_nn hz.1, by nlinarith [hz.2]⟩
    · exact ⟨le_refl 0, hc_nn⟩
  have hX1_nonG : ∀ (s : Fin (T + 1)) (j : SlotA D),
      (∀ i, j ≠ .G i) → X1 s j = X1_attn s j := by
    intro s j hj
    dsimp only [X1, ffn1A]
    cases j with
    | G i => exact absurd rfl (hj i)
    | _ => rfl
  have hX2_nonG : ∀ (s : Fin (T + 1)) (j : SlotA D),
      (∀ i, j ≠ .G i) → X2 s j = X1 s j := by
    intro s j hj
    unfold X2 attnLayerOn
    dsimp only [Pi.add_apply]
    rw [WO2A_mulVec]
    cases j with
    | G i => exact absurd rfl (hj i)
    | _ => simp
  have hX2_G_Icc : ∀ (s : Fin (T + 1)) (i : Fin D), 0 ≤ X2 s (.G i) ∧ X2 s (.G i) ≤ 2 * c := by
    intro s i
    unfold X2 attnLayerOn
    dsimp only [Pi.add_apply]
    rw [WO2A_mulVec]
    simp only [WV2A_mulVec]
    have havg := hconv_Icc
      (fun t => Real.exp (β2 * dotProduct ((WQ2A D).mulVec (X1 s)) ((WK2A D).mulVec (X1 t))))
      (fun t => X1 t (.G i)) (fun _ => Real.exp_pos _) (fun t => hX1_G_Icc t i)
    have hres := hX1_G_Icc s i
    constructor <;> linarith
  have hX0_norm : ∀ s, ∑ j : SlotA D, (X0 s j) ^ 2 ≤ 1 := by
    intro s
    rw [sum_slotA]
    refine Fin.lastCases ?_ ?_ s
    · simp only [X0, peEmbed, Fin.snoc_last, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true,
        zero_pow, Finset.sum_const_zero, add_zero]
      have hA : ∑ _i : Fin D, (-(2 * c)) ^ 2 = 4 * (D : ℝ) * c ^ 2 := by simp; ring
      change (∑ _i : Fin D, (-(2 * c)) ^ 2) + c ^ 2 ≤ 1
      rw [hA, show 4 * (D : ℝ) * c ^ 2 + c ^ 2 = (4 * (D : ℝ) + 1) * c ^ 2 by ring, hc_sq]
      rw [mul_one_div, div_le_one h11D_pos]
      linarith
    · intro t
      simp only [X0, peEmbed, Fin.snoc_castSucc, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true,
        zero_pow, Finset.sum_const_zero, add_zero]
      have hA : ∑ i : Fin D, (-(c * P t i)) ^ 2 ≤ (D : ℝ) * c ^ 2 := by
        calc ∑ i : Fin D, (-(c * P t i)) ^ 2
            ≤ ∑ _i : Fin D, c ^ 2 := Finset.sum_le_sum fun i _ => by
              have hnn : 0 ≤ c * P t i := mul_nonneg hc_nn (hP t i).1
              have hle : c * P t i ≤ c := by nlinarith [(hP t i).2, hc_nn]
              nlinarith
          _ = (D : ℝ) * c ^ 2 := by simp
      have hC : ∑ i : Fin D, (c * P t i) ^ 2 ≤ (D : ℝ) * c ^ 2 := by
        calc ∑ i : Fin D, (c * P t i) ^ 2
            ≤ ∑ _i : Fin D, c ^ 2 := Finset.sum_le_sum fun i _ => by
              have hnn : 0 ≤ c * P t i := mul_nonneg hc_nn (hP t i).1
              have hle : c * P t i ≤ c := by nlinarith [(hP t i).2, hc_nn]
              nlinarith
          _ = (D : ℝ) * c ^ 2 := by simp
      have hH : (if (t : ℕ) < D then c else 0) ^ 2 ≤ c ^ 2 := by
        split_ifs <;> nlinarith [sq_nonneg c]
      have hsum_le :
          (∑ i : Fin D, (-(c * P t i)) ^ 2) +
          (∑ i : Fin D, (if (t : ℕ) = (i : ℕ) then c else 0) ^ 2) +
          (∑ i : Fin D, (c * P t i) ^ 2) +
          (if (t : ℕ) < D then c else 0) ^ 2 ≤ (2 * (D : ℝ) + 2) * c ^ 2 := by
        linarith [hB_sq t]
      have h1 : (2 * (D : ℝ) + 2) * c ^ 2 ≤ 1 := by
        rw [hc_sq, mul_one_div, div_le_one h11D_pos]; linarith
      change (∑ i : Fin D, (-(c * P t i)) ^ 2) +
          (∑ i : Fin D, (if (t : ℕ) = (i : ℕ) then c else 0) ^ 2) +
          (∑ i : Fin D, (c * P t i) ^ 2) +
          (if (t : ℕ) < D then c else 0) ^ 2 ≤ 1
      linarith
  have hF_sq : ∀ s : Fin (T + 1), ∑ i : Fin D, (X1_attn s (.F i)) ^ 2 ≤ (D : ℝ) * c ^ 2 := by
    intro s
    calc ∑ i : Fin D, (X1_attn s (.F i)) ^ 2
        ≤ ∑ _i : Fin D, c ^ 2 := Finset.sum_le_sum fun i _ => by
          have h := hX1_attn_F_Icc s i; nlinarith
      _ = (D : ℝ) * c ^ 2 := by simp
  have hbase_sq : ∀ s : Fin (T + 1),
      (∑ i : Fin D, (X0 s (.A i)) ^ 2) + (∑ i : Fin D, (X0 s (.B i)) ^ 2) +
      (∑ i : Fin D, (X0 s (.C i)) ^ 2) + (X0 s .H) ^ 2 + (X0 s .I) ^ 2 ≤
      (4 * (D : ℝ) + 2) * c ^ 2 := by
    intro s
    refine Fin.lastCases ?_ ?_ s
    · simp only [X0, peEmbed, Fin.snoc_last, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true,
        zero_pow, Finset.sum_const_zero, add_zero]
      have hA : ∑ _i : Fin D, (-(2 * c)) ^ 2 = 4 * (D : ℝ) * c ^ 2 := by simp; ring
      change (∑ _i : Fin D, (-(2 * c)) ^ 2) + c ^ 2 ≤ (4 * (D : ℝ) + 2) * c ^ 2
      rw [hA]; nlinarith [sq_nonneg c]
    · intro t
      simp only [X0, peEmbed, Fin.snoc_castSucc, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true,
        zero_pow, add_zero]
      have hA : ∑ i : Fin D, (-(c * P t i)) ^ 2 ≤ (D : ℝ) * c ^ 2 := by
        calc ∑ i : Fin D, (-(c * P t i)) ^ 2
            ≤ ∑ _i : Fin D, c ^ 2 := Finset.sum_le_sum fun i _ => by
              have hnn : 0 ≤ c * P t i := mul_nonneg hc_nn (hP t i).1
              have hle : c * P t i ≤ c := by nlinarith [(hP t i).2, hc_nn]
              nlinarith
          _ = (D : ℝ) * c ^ 2 := by simp
      have hC : ∑ i : Fin D, (c * P t i) ^ 2 ≤ (D : ℝ) * c ^ 2 := by
        calc ∑ i : Fin D, (c * P t i) ^ 2
            ≤ ∑ _i : Fin D, c ^ 2 := Finset.sum_le_sum fun i _ => by
              have hnn : 0 ≤ c * P t i := mul_nonneg hc_nn (hP t i).1
              have hle : c * P t i ≤ c := by nlinarith [(hP t i).2, hc_nn]
              nlinarith
          _ = (D : ℝ) * c ^ 2 := by simp
      have hH : (if (t : ℕ) < D then c else 0) ^ 2 ≤ c ^ 2 := by
        split_ifs <;> nlinarith [sq_nonneg c]
      change (∑ i : Fin D, (-(c * P t i)) ^ 2) +
          (∑ i : Fin D, (if (t : ℕ) = (i : ℕ) then c else 0) ^ 2) +
          (∑ i : Fin D, (c * P t i) ^ 2) +
          (if (t : ℕ) < D then c else 0) ^ 2 ≤ (4 * (D : ℝ) + 2) * c ^ 2
      have hD_nn : (0 : ℝ) ≤ (D : ℝ) := Nat.cast_nonneg D
      nlinarith [hB_sq t, sq_nonneg c]
  have hX1_attn_norm : ∀ s, ∑ j : SlotA D, (X1_attn s j) ^ 2 ≤ 1 := by
    intro s
    rw [sum_slotA]
    have hA : ∑ i : Fin D, (X1_attn s (.A i)) ^ 2 = ∑ i : Fin D, (X0 s (.A i)) ^ 2 :=
      Finset.sum_congr rfl fun i _ => by rw [hX1_attn_nonF s (.A i) (fun _ h => by cases h)]
    have hB : ∑ i : Fin D, (X1_attn s (.B i)) ^ 2 = ∑ i : Fin D, (X0 s (.B i)) ^ 2 :=
      Finset.sum_congr rfl fun i _ => by rw [hX1_attn_nonF s (.B i) (fun _ h => by cases h)]
    have hC : ∑ i : Fin D, (X1_attn s (.C i)) ^ 2 = ∑ i : Fin D, (X0 s (.C i)) ^ 2 :=
      Finset.sum_congr rfl fun i _ => by rw [hX1_attn_nonF s (.C i) (fun _ h => by cases h)]
    have hG : ∑ i : Fin D, (X1_attn s (.G i)) ^ 2 = 0 := by
      refine Finset.sum_eq_zero fun i _ => ?_
      rw [hX1_attn_nonF s (.G i) (fun _ h => by cases h)]
      refine Fin.lastCases ?_ ?_ s <;> simp [X0, peEmbed]
    have hH : X1_attn s .H = X0 s .H := hX1_attn_nonF s .H (fun _ h => by cases h)
    have hI : X1_attn s .I = X0 s .I := hX1_attn_nonF s .I (fun _ h => by cases h)
    rw [hA, hB, hC, hG, hH, hI]
    have h1 : (5 * (D : ℝ) + 2) * c ^ 2 ≤ 1 := by
      rw [hc_sq, mul_one_div, div_le_one h11D_pos]; linarith
    linarith [hbase_sq s, hF_sq s]
  have hX1_norm : ∀ s, ∑ j : SlotA D, (X1 s j) ^ 2 ≤ 1 := by
    intro s
    rw [sum_slotA]
    have hA : ∑ i : Fin D, (X1 s (.A i)) ^ 2 = ∑ i : Fin D, (X0 s (.A i)) ^ 2 :=
      Finset.sum_congr rfl fun i _ => by
        rw [hX1_nonG s (.A i) (fun _ h => by cases h), hX1_attn_nonF s (.A i) (fun _ h => by cases h)]
    have hB : ∑ i : Fin D, (X1 s (.B i)) ^ 2 = ∑ i : Fin D, (X0 s (.B i)) ^ 2 :=
      Finset.sum_congr rfl fun i _ => by
        rw [hX1_nonG s (.B i) (fun _ h => by cases h), hX1_attn_nonF s (.B i) (fun _ h => by cases h)]
    have hC : ∑ i : Fin D, (X1 s (.C i)) ^ 2 = ∑ i : Fin D, (X0 s (.C i)) ^ 2 :=
      Finset.sum_congr rfl fun i _ => by
        rw [hX1_nonG s (.C i) (fun _ h => by cases h), hX1_attn_nonF s (.C i) (fun _ h => by cases h)]
    have hF : ∑ i : Fin D, (X1 s (.F i)) ^ 2 = ∑ i : Fin D, (X1_attn s (.F i)) ^ 2 :=
      Finset.sum_congr rfl fun i _ => by rw [hX1_nonG s (.F i) (fun _ h => by cases h)]
    have hG : ∑ i : Fin D, (X1 s (.G i)) ^ 2 ≤ (D : ℝ) * c ^ 2 := by
      calc ∑ i : Fin D, (X1 s (.G i)) ^ 2
          ≤ ∑ _i : Fin D, c ^ 2 := Finset.sum_le_sum fun i _ => by
            have h := hX1_G_Icc s i; nlinarith
        _ = (D : ℝ) * c ^ 2 := by simp
    have hH : X1 s .H = X0 s .H := by
      rw [hX1_nonG s .H (fun _ h => by cases h), hX1_attn_nonF s .H (fun _ h => by cases h)]
    have hI : X1 s .I = X0 s .I := by
      rw [hX1_nonG s .I (fun _ h => by cases h), hX1_attn_nonF s .I (fun _ h => by cases h)]
    rw [hA, hB, hC, hF, hH, hI]
    have h1 : (6 * (D : ℝ) + 2) * c ^ 2 ≤ 1 := by
      rw [hc_sq, mul_one_div, div_le_one h11D_pos]; linarith
    linarith [hbase_sq s, hF_sq s]
  have hX2_norm : ∀ s, ∑ j : SlotA D, (X2 s j) ^ 2 ≤ 1 := by
    intro s
    rw [sum_slotA]
    have hA : ∑ i : Fin D, (X2 s (.A i)) ^ 2 = ∑ i : Fin D, (X0 s (.A i)) ^ 2 :=
      Finset.sum_congr rfl fun i _ => by
        rw [hX2_nonG s (.A i) (fun _ h => by cases h), hX1_nonG s (.A i) (fun _ h => by cases h),
          hX1_attn_nonF s (.A i) (fun _ h => by cases h)]
    have hB : ∑ i : Fin D, (X2 s (.B i)) ^ 2 = ∑ i : Fin D, (X0 s (.B i)) ^ 2 :=
      Finset.sum_congr rfl fun i _ => by
        rw [hX2_nonG s (.B i) (fun _ h => by cases h), hX1_nonG s (.B i) (fun _ h => by cases h),
          hX1_attn_nonF s (.B i) (fun _ h => by cases h)]
    have hC : ∑ i : Fin D, (X2 s (.C i)) ^ 2 = ∑ i : Fin D, (X0 s (.C i)) ^ 2 :=
      Finset.sum_congr rfl fun i _ => by
        rw [hX2_nonG s (.C i) (fun _ h => by cases h), hX1_nonG s (.C i) (fun _ h => by cases h),
          hX1_attn_nonF s (.C i) (fun _ h => by cases h)]
    have hF : ∑ i : Fin D, (X2 s (.F i)) ^ 2 = ∑ i : Fin D, (X1_attn s (.F i)) ^ 2 :=
      Finset.sum_congr rfl fun i _ => by
        rw [hX2_nonG s (.F i) (fun _ h => by cases h), hX1_nonG s (.F i) (fun _ h => by cases h)]
    have hG : ∑ i : Fin D, (X2 s (.G i)) ^ 2 ≤ 4 * (D : ℝ) * c ^ 2 := by
      calc ∑ i : Fin D, (X2 s (.G i)) ^ 2
          ≤ ∑ _i : Fin D, (4 * c ^ 2) := Finset.sum_le_sum fun i _ => by
            have h := hX2_G_Icc s i; nlinarith
        _ = 4 * (D : ℝ) * c ^ 2 := by simp; ring
    have hH : X2 s .H = X0 s .H := by
      rw [hX2_nonG s .H (fun _ h => by cases h), hX1_nonG s .H (fun _ h => by cases h),
        hX1_attn_nonF s .H (fun _ h => by cases h)]
    have hI : X2 s .I = X0 s .I := by
      rw [hX2_nonG s .I (fun _ h => by cases h), hX1_nonG s .I (fun _ h => by cases h),
        hX1_attn_nonF s .I (fun _ h => by cases h)]
    rw [hA, hB, hC, hF, hH, hI]
    have h1 : (9 * (D : ℝ) + 2) * c ^ 2 ≤ 1 := by
      rw [hc_sq, mul_one_div, div_le_one h11D_pos]; linarith
    linarith [hbase_sq s, hF_sq s]
  refine ⟨hX0_norm, hX1_attn_norm, hX1_norm, hX2_norm, hL1_ret, hL1_gate, ?_⟩
  have hX1_cls_I : X1 (Fin.last T) .I = c := by
    rw [hX1_nonG _ .I (fun _ h => by cases h), hX1_attn_nonF _ .I (fun _ h => by cases h)]
    simp [X0, peEmbed, c]
  have hX1_H : ∀ t : Fin (T + 1), X1 t .H = if (t : ℕ) < D then c else 0 := by
    intro t
    rw [hX1_nonG _ .H (fun _ h => by cases h), hX1_attn_nonF _ .H (fun _ h => by cases h)]
    refine Fin.lastCases ?_ ?_ t
    · have hnot : ¬ (Fin.last T : ℕ) < D := by simp; omega
      rw [if_neg hnot]; simp [X0, peEmbed]
    · intro u; simp [X0, peEmbed, c]
  have hw2_eq : ∀ t : Fin (T + 1),
      Real.exp (β2 * dotProduct ((WQ2A D).mulVec (X1 (Fin.last T)))
        ((WK2A D).mulVec (X1 t))) = if (t : ℕ) < D then Real.exp btilde2 else 1 := by
    intro t
    rw [dot_Q2A_K2A, hX1_cls_I, hX1_H t]
    split_ifs
    · rw [hβ2_c2]
    · simp
  have hcard_lt : (Finset.univ.filter (fun t : Fin (T + 1) => (t : ℕ) < D)).card = D := by
    have hbij : (Finset.univ.filter (fun t : Fin (T + 1) => (t : ℕ) < D)).card =
        (Finset.univ : Finset (Fin D)).card := by
      refine Finset.card_bij (fun t ht => ⟨t.1, (Finset.mem_filter.mp ht).2⟩) ?_ ?_ ?_
      · intro _ _; exact Finset.mem_univ _
      · intro a _ b _ h; exact Fin.ext (by simpa using congrArg Fin.val h)
      · intro i _
        refine ⟨⟨i.1, by have := i.2; omega⟩, ?_, by ext; rfl⟩
        simp [i.2]
    simpa using hbij
  have hcard_ge : (Finset.univ.filter (fun t : Fin (T + 1) => ¬ (t : ℕ) < D)).card = T + 1 - D := by
    have hsum_card := Finset.filter_card_add_filter_neg_card_eq_card
      (s := (Finset.univ : Finset (Fin (T + 1))))
      (p := fun t : Fin (T + 1) => (t : ℕ) < D)
    simp only [Finset.card_univ, Fintype.card_fin] at hsum_card
    omega
  set Z2 : ℝ := (D : ℝ) * Real.exp btilde2 + ((T : ℝ) + 1 - (D : ℝ))
  have hZ2_eq :
      (∑ t : Fin (T + 1),
        Real.exp (β2 * dotProduct ((WQ2A D).mulVec (X1 (Fin.last T)))
          ((WK2A D).mulVec (X1 t)))) = Z2 := by
    simp_rw [hw2_eq]
    rw [Finset.sum_ite, Finset.sum_const, Finset.sum_const, nsmul_eq_mul, nsmul_eq_mul,
      hcard_lt, hcard_ge, mul_one]
    have hsub_cast : ((T + 1 - D : ℕ) : ℝ) = (T : ℝ) + 1 - (D : ℝ) := by
      rw [Nat.cast_sub (by omega : D ≤ T + 1), Nat.cast_add, Nat.cast_one]
    rw [hsub_cast]
  have hZ2_pos : 0 < Z2 := by
    have hDT_r : (D : ℝ) ≤ (T : ℝ) := Nat.cast_le.mpr hDT
    have hD_pos : (0 : ℝ) < (D : ℝ) := Nat.cast_pos.mpr (by omega)
    have hexp_pos : 0 < Real.exp btilde2 := Real.exp_pos _
    dsimp only [Z2]
    nlinarith
  have hL2_cls_G : ∀ i : Fin D,
      X2 (Fin.last T) (.G i) =
        (c * Real.exp btilde2 / Z2) * softminAvg Finset.univ (fun t => P t i) btilde1 := by
    intro i
    have hX1_cls_G : X1 (Fin.last T) (.G i) = 0 := by
      rw [hL1_gate (Fin.last T) i, if_neg (by simp; have := i.2; omega)]
    unfold X2 attnLayerOn
    dsimp only [Pi.add_apply]
    rw [hX1_cls_G, zero_add, WO2A_mulVec]
    simp only [WV2A_mulVec]
    rw [hZ2_eq]
    simp_rw [hw2_eq, hL1_gate]
    have hnum :
        (∑ t : Fin (T + 1),
          (if (t : ℕ) < D then Real.exp btilde2 else 1) *
            (if (t : ℕ) = (i : ℕ) then c * softminAvg Finset.univ (fun u => P u i) btilde1 else 0)) =
        Real.exp btilde2 * (c * softminAvg Finset.univ (fun u => P u i) btilde1) := by
      rw [Finset.sum_eq_single (⟨i.1, by have := i.2; omega⟩ : Fin (T + 1))]
      · simp [i.2]
      · intro t _ ht
        have hne_val : (t : ℕ) ≠ (i : ℕ) := fun h => ht (Fin.ext h)
        simp [hne_val]
      · intro h; exact absurd (Finset.mem_univ _) h
    rw [hnum]
    ring
  have hD_pos : (0 : ℝ) < (D : ℝ) := Nat.cast_pos.mpr (by omega)
  have hD_ne : (D : ℝ) ≠ 0 := ne_of_gt hD_pos
  have hc_ne : c ≠ 0 := ne_of_gt hc_pos
  have hL2_err : ∀ i : Fin D,
      |(D : ℝ) / c * X2 (Fin.last T) (.G i) -
          softminAvg Finset.univ (fun t => P t i) btilde1| ≤
        ((T : ℝ) + 1 - (D : ℝ)) / ((D : ℝ) * Real.exp btilde2) := by
    intro i
    set zi := softminAvg Finset.univ (fun t => P t i) btilde1
    have hzi := softminAvg_mem_Icc Finset.univ (fun t => P t i) btilde1 (fun t => hP t i)
    rw [hL2_cls_G i]
    have hsimp :
        (D : ℝ) / c * ((c * Real.exp btilde2 / Z2) * zi) - zi =
        -(((T : ℝ) + 1 - (D : ℝ)) / Z2 * zi) := by
      field_simp [hc_ne, ne_of_gt hZ2_pos]
      dsimp only [Z2]
      ring
    rw [hsimp, abs_neg]
    have hrem_nn : 0 ≤ (T : ℝ) + 1 - (D : ℝ) := by
      have : (D : ℝ) ≤ (T : ℝ) := Nat.cast_le.mpr hDT
      linarith
    rw [abs_of_nonneg (mul_nonneg (div_nonneg hrem_nn (le_of_lt hZ2_pos)) hzi.1)]
    have hZ2_ge : (D : ℝ) * Real.exp btilde2 ≤ Z2 := by dsimp only [Z2]; linarith
    have hDexp_pos : 0 < (D : ℝ) * Real.exp btilde2 := mul_pos hD_pos (Real.exp_pos _)
    calc ((T : ℝ) + 1 - (D : ℝ)) / Z2 * zi
        ≤ ((T : ℝ) + 1 - (D : ℝ)) / Z2 * 1 :=
          mul_le_mul_of_nonneg_left hzi.2 (div_nonneg hrem_nn (le_of_lt hZ2_pos))
      _ = ((T : ℝ) + 1 - (D : ℝ)) / Z2 := mul_one _
      _ ≤ ((T : ℝ) + 1 - (D : ℝ)) / ((D : ℝ) * Real.exp btilde2) :=
          div_le_div_of_nonneg_left hrem_nn hDexp_pos hZ2_ge
  have hrewrite :
      (D : ℝ) / c * (∑ i : Fin D, X2 (Fin.last T) (.G i)) -
        ∑ i : Fin D, Finset.univ.inf' hne (fun t => P t i) =
      (∑ i : Fin D, ((D : ℝ) / c * X2 (Fin.last T) (.G i) -
          softminAvg Finset.univ (fun t => P t i) btilde1)) +
      (∑ i : Fin D, (softminAvg Finset.univ (fun t => P t i) btilde1 -
          Finset.univ.inf' hne (fun t => P t i))) := by
    rw [Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun i _ => by ring
  rw [hrewrite]
  have htri := abs_add_le
    (∑ i : Fin D, ((D : ℝ) / c * X2 (Fin.last T) (.G i) -
        softminAvg Finset.univ (fun t => P t i) btilde1))
    (∑ i : Fin D, (softminAvg Finset.univ (fun t => P t i) btilde1 -
        Finset.univ.inf' hne (fun t => P t i)))
  have hpart1 :
      |∑ i : Fin D, ((D : ℝ) / c * X2 (Fin.last T) (.G i) -
          softminAvg Finset.univ (fun t => P t i) btilde1)| ≤
      ((T : ℝ) + 1 - (D : ℝ)) / Real.exp btilde2 := by
    calc |∑ i : Fin D, ((D : ℝ) / c * X2 (Fin.last T) (.G i) -
            softminAvg Finset.univ (fun t => P t i) btilde1)|
        ≤ ∑ i : Fin D, |(D : ℝ) / c * X2 (Fin.last T) (.G i) -
            softminAvg Finset.univ (fun t => P t i) btilde1| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _i : Fin D, (((T : ℝ) + 1 - (D : ℝ)) / ((D : ℝ) * Real.exp btilde2)) :=
            Finset.sum_le_sum fun i _ => hL2_err i
      _ = ((T : ℝ) + 1 - (D : ℝ)) / Real.exp btilde2 := by
            simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
            field_simp [hD_ne, ne_of_gt (Real.exp_pos btilde2)]
  have hpart2 :
      |∑ i : Fin D, (softminAvg Finset.univ (fun t => P t i) btilde1 -
          Finset.univ.inf' hne (fun t => P t i))| ≤
      (D : ℝ) * ((T : ℝ) / (Real.exp 1 * btilde1) + Real.exp (-btilde1)) := by
    calc |∑ i : Fin D, (softminAvg Finset.univ (fun t => P t i) btilde1 -
            Finset.univ.inf' hne (fun t => P t i))|
        ≤ ∑ i : Fin D, |softminAvg Finset.univ (fun t => P t i) btilde1 -
            Finset.univ.inf' hne (fun t => P t i)| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _i : Fin D, ((T : ℝ) / (Real.exp 1 * btilde1) + Real.exp (-btilde1)) :=
            Finset.sum_le_sum fun i _ =>
              softmin_error_bound_univ hT (fun t => P t i) (fun t => hP t i) hbt1
      _ = (D : ℝ) * ((T : ℝ) / (Real.exp 1 * btilde1) + Real.exp (-btilde1)) := by simp; ring
  linarith

end TransformerHeadCount
