import Mathlib

/-!
# Formalization of Core Lemmas and the `L = 2, h = 1, D = 3` Attention Construction for Conjecture 1 (arXiv:2510.06662)

We formalize in Lean 4 + Mathlib (zero `sorry`) the softmin error bound and the explicit
`L = 2, h = 1, D = 3, n = 9` attention-layer construction underlying our refutation of Conjecture 1
in Yu et al. (ICLR 2026, arXiv:2510.06662) for standard multilayer bidirectional transformer
encoders (where all tokens issue queries and residual connections around attention are present;
the paper leaves the multilayer model unspecified, so CLS-only and residual-free variants remain
open). Conjecture 1 asks whether `L * h ≥ D` is necessary to avoid `T`-dependent exponential growth
in FFN parameter count `M` (with softmax inverse temperature `β > 0` allowed to grow with `T / ε`,
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
   `{-1, 0, 1} ⊆ [-1, 1]` (`theoremB_matrices_bounded`, independent of `T`), proving:
   - Every layer satisfies the unit-norm constraint `∑ j, (x^(ℓ)(s)_j)² ≤ 1` (`nopeEmbed_unit_norm`
     and the first three conjuncts of `theoremB_L2_h1_D3`).
   - Exact attention-logit decomposition at Layers 1 and 2, proving that the linear CLS readout
     `3 * (X2 (last T) 3 + X2 (last T) 4 + X2 (last T) 6)` approximates `∑_{i=0}^2 min_t P t i`
     within `3 * (T / (e * β̃) + 2 * exp (-β̃))` for every `T ≥ 1` and `β̃ > 0` (`theoremB_L2_h1_D3`),
     so choosing `β̃ = Θ(T / ε)` achieves accuracy `ε` with fixed `9 × 9` attention matrices.

(Note: the full statement of Conjecture 1 — FFN approximation of `f_i` via Lemma 1, neuron-splitting
Lemma 2 to implement the `3×` readout with `[-1, 1]` weights, FFN parameter count `M`, Theorem A,
and the general `(L, h)` family — is proved in prose in `solution.md` and verified numerically in
`verify_transformer_heads.py`; this Lean file formally certifies Lemma 3 and the `L = 2, h = 1,
D = 3` attention mechanism with exact `P ∈ [0, 1]` inputs and `F₀ = sum`.)
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

end TransformerHeadCount
