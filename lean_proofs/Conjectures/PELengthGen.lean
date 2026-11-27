import Mathlib

/-!
# General Length-Generalization Impossibility for a Positional Head
(arXiv:2610.41218)

We formalize in Lean 4 + Mathlib (zero `sorry`, standard axioms only) the three core results on
Conjecture `conj:general` and its growing-offset quartic counterexample:

1. **Positive-scaling invariance proving `conj:general` under S1–S3**:
   - `logit_smul`: Bilinear positional logits `s_W(i, j) = φ(i)ᵀ W φ(j)` are positively
     homogeneous in `W`.
   - `positive_scaling_to_margin`: Any head `W` strictly realizing a finite set of old
     comparisons can be positively rescaled to `c • W` (`c > 0`) attaining any prescribed margin
     `γ > 0` while preserving every strict comparison on every length.
   - `conj_general_under_S1_S3`: Under the displayed separation hypothesis (for some `γ > 0`,
     every head with old margin `≥ γ` fails at least one new comparison), NO head `W` can
     strictly realize both the old and new comparisons.
   - `dual_certificate_obstruction`: Any nonnegative dual certificate `y` with positive weight
     satisfying `∑ y_h gap_h(W) = 0` for all `W` rules out strict joint realization.
2. **Explicit quartic copy counterexample `g(d) = d - 2(d - 3)^2 (d - 5)^2`**:
   - `quartic_bilinear_repr`: The explicit `5 × 5` matrix `W₄` with degree-4 monomial features
     `φ₄(i) = (1, i, i^2, i^3, i^4)` realizes `logit φ₄ W₄ i j = quarticScore (i - j)`.
   - `quartic_old_margin`, `quartic_old_margin_exact`: At `L₀ = 5` (scored rows `i ∈ {3, 4}`,
     target `i - 3`), `W₄` achieves exact minimum margin `1` (attained at `(i, j) = (4, 0)`).
   - `quartic_new_margin`, `quartic_new_margin_exact`: At `L = 10` (scored rows `i ∈ {5..9}`,
     target `i - 5`), `W₄` achieves exact minimum margin `2` (attained at `(i, j) = (5, 2)`).
3. **5th-forward-difference non-realizability witness (`-5 ≠ 0`)**:
   - `fwdDiff5_annihilates_quartic`: The 5th forward difference on keys `0..5` annihilates
     `j ↦ logit φ₄ W i j` for EVERY `5 × 5` matrix `W` and EVERY row `i`.
   - `fwdDiff5_targetRow9`: On the one-hot target row `j ↦ 1[j = 4]` (row `9` at `L = 10`), the
     same 5th forward difference equals `-5`.
   - `one_hot_not_realizable`: No `W : Matrix (Fin 5) (Fin 5) ℝ` matches the one-hot target on
     row `9` (so the projection residual is strictly nonzero even while `W₄` strictly realizes
     argmax at both `L₀ = 5` and `L = 10`).

Machine-checked vs prose-only:
- Machine-checked: Positive-scaling invariance and `conj:general` under S1–S3, the Gordan dual
  obstruction direction, exact bilinear representation of `g(d)`, exact margins `1` at `L₀ = 5`
  and `2` at `L = 10`, and the `-5 ≠ 0` 5th-forward-difference non-realizability witness.
- Prose-only (in `solutions/pe_length_gen/results.md`): The primal-dual converse direction of the
  finite Gordan alternative (nearest-point convex hull projection), the Vandermonde minimum-norm
  interpolant gaps `(-15, 15, -35, -70, -126)`, and the 12-case LP verification suite.
-/

open BigOperators Finset

namespace PELengthGen

/-! ### 1. Positive-Scaling Invariance and `conj:general` under S1–S3 -/

/-- Bilinear positional logit `s_W(i, j) = φ(i)ᵀ W φ(j)`. -/
def logit {D : ℕ} (φ : ℕ → Fin D → ℝ) (W : Matrix (Fin D) (Fin D) ℝ) (i j : ℕ) : ℝ :=
  ∑ a : Fin D, ∑ b : Fin D, φ i a * W a b * φ j b

lemma logit_smul {D : ℕ} (φ : ℕ → Fin D → ℝ) (c : ℝ) (W : Matrix (Fin D) (Fin D) ℝ) (i j : ℕ) :
    logit φ (c • W) i j = c * logit φ W i j := by
  unfold logit
  simp only [Matrix.smul_apply, smul_eq_mul]
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun a _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun b _ => ?_
  ring

/-- Target-minus-competitor gap for a comparison triple `(i, t, j)`. -/
def compGap {D : ℕ} (φ : ℕ → Fin D → ℝ) (W : Matrix (Fin D) (Fin D) ℝ) (comp : ℕ × ℕ × ℕ) : ℝ :=
  logit φ W comp.1 comp.2.1 - logit φ W comp.1 comp.2.2

lemma compGap_smul {D : ℕ} (φ : ℕ → Fin D → ℝ) (c : ℝ) (W : Matrix (Fin D) (Fin D) ℝ)
    (comp : ℕ × ℕ × ℕ) :
    compGap φ (c • W) comp = c * compGap φ W comp := by
  unfold compGap
  rw [logit_smul, logit_smul]
  ring

/-- Strict argmax realization of a finite set of comparisons (assumption S2). -/
def RealizesComps {D : ℕ} (φ : ℕ → Fin D → ℝ) (W : Matrix (Fin D) (Fin D) ℝ)
    (comps : Finset (ℕ × ℕ × ℕ)) : Prop :=
  ∀ comp ∈ comps, 0 < compGap φ W comp

/-- Uniform margin `γ` on a finite set of comparisons. -/
def HasMargin {D : ℕ} (φ : ℕ → Fin D → ℝ) (W : Matrix (Fin D) (Fin D) ℝ)
    (comps : Finset (ℕ × ℕ × ℕ)) (γ : ℝ) : Prop :=
  ∀ comp ∈ comps, γ ≤ compGap φ W comp

/-- Positive scaling preserves strict realization on any comparison set. -/
theorem realizesComps_smul {D : ℕ} (φ : ℕ → Fin D → ℝ) {c : ℝ} (hc : 0 < c)
    (W : Matrix (Fin D) (Fin D) ℝ) (comps : Finset (ℕ × ℕ × ℕ)) :
    RealizesComps φ (c • W) comps ↔ RealizesComps φ W comps := by
  constructor <;> intro h comp hcomp
  · have hpos := h comp hcomp
    rw [compGap_smul] at hpos
    exact (mul_pos_iff_of_pos_left hc).1 hpos
  · rw [compGap_smul]
    exact mul_pos hc (h comp hcomp)

/-- **Positive-scaling to prescribed margin**: if `W` strictly realizes `oldComps` and `γ > 0`,
there exists `c > 0` such that `c • W` attains margin `γ` on `oldComps` while preserving every
strict comparison sign. -/
theorem positive_scaling_to_margin {D : ℕ} (φ : ℕ → Fin D → ℝ)
    (W : Matrix (Fin D) (Fin D) ℝ) (oldComps : Finset (ℕ × ℕ × ℕ))
    (hold : RealizesComps φ W oldComps) (γ : ℝ) :
    ∃ c : ℝ, 0 < c ∧ HasMargin φ (c • W) oldComps γ := by
  by_cases hne : oldComps.Nonempty
  · obtain ⟨comp₀, hcomp₀_mem, hcomp₀_min⟩ :=
      Finset.exists_min_image oldComps (compGap φ W) hne
    set m₀ : ℝ := compGap φ W comp₀
    have hm₀_pos : 0 < m₀ := hold comp₀ hcomp₀_mem
    refine ⟨max 1 (γ / m₀), lt_of_lt_of_le one_pos (le_max_left _ _), ?_⟩
    intro comp hcomp
    rw [compGap_smul]
    have hgap_ge : m₀ ≤ compGap φ W comp := hcomp₀_min comp hcomp
    have hc_ge : γ / m₀ ≤ max 1 (γ / m₀) := le_max_right _ _
    have h1 : γ / m₀ * m₀ = γ := div_mul_cancel₀ γ (ne_of_gt hm₀_pos)
    have h2 : γ / m₀ * m₀ ≤ max 1 (γ / m₀) * compGap φ W comp :=
      mul_le_mul hc_ge hgap_ge (le_of_lt hm₀_pos) (le_trans (le_of_lt one_pos) (le_max_left _ _))
    linarith
  · rw [Finset.not_nonempty_iff_eq_empty] at hne
    refine ⟨1, one_pos, ?_⟩
    intro comp hcomp
    simp [hne] at hcomp

/-- **`conj:general` under S1–S3**: if for some `γ > 0`, every parameter `W` achieving old margin
at least `γ` fails some new comparison (`compGap ≤ 0`), then NO parameter `W` can strictly realize
both `oldComps` and `newComps`. -/
theorem conj_general_under_S1_S3 {D : ℕ} (φ : ℕ → Fin D → ℝ)
    (oldComps newComps : Finset (ℕ × ℕ × ℕ))
    {γ : ℝ} (hγ : 0 < γ)
    (hsep : ∀ W : Matrix (Fin D) (Fin D) ℝ,
      HasMargin φ W oldComps γ → ∃ comp ∈ newComps, compGap φ W comp ≤ 0) :
    ¬ ∃ W : Matrix (Fin D) (Fin D) ℝ,
      RealizesComps φ W oldComps ∧ RealizesComps φ W newComps := by
  rintro ⟨W, hold, hnew⟩
  obtain ⟨c, hc_pos, hmargin⟩ := positive_scaling_to_margin φ W oldComps hold γ
  have _ := hγ
  obtain ⟨comp, hcomp_mem, hfail⟩ := hsep (c • W) hmargin
  have hstrict : 0 < compGap φ (c • W) comp := (realizesComps_smul φ hc_pos W newComps).2 hnew comp hcomp_mem
  linarith

/-- **Gordan dual obstruction**: if nonnegative weights `y` on `comps` sum to `1` and cancel the
weighted comparison gaps identically for all `W`, then no `W` strictly realizes `comps`. -/
theorem dual_certificate_obstruction {D : ℕ} (φ : ℕ → Fin D → ℝ)
    (comps : Finset (ℕ × ℕ × ℕ)) (y : (ℕ × ℕ × ℕ) → ℝ)
    (hy_nonneg : ∀ comp ∈ comps, 0 ≤ y comp)
    (hy_sum : ∑ comp ∈ comps, y comp = 1)
    (hdual : ∀ W : Matrix (Fin D) (Fin D) ℝ, ∑ comp ∈ comps, y comp * compGap φ W comp = 0) :
    ¬ ∃ W : Matrix (Fin D) (Fin D) ℝ, RealizesComps φ W comps := by
  rintro ⟨W, hreal⟩
  have hpos_exists : ∃ comp ∈ comps, 0 < y comp := by
    by_contra h
    push_neg at h
    have hsum0 : ∑ comp ∈ comps, y comp ≤ 0 := Finset.sum_nonpos h
    linarith
  rcases hpos_exists with ⟨comp₀, hcomp₀, hy₀⟩
  have hsum_pos : 0 < ∑ comp ∈ comps, y comp * compGap φ W comp :=
    Finset.sum_pos' (fun comp hcomp => mul_nonneg (hy_nonneg comp hcomp) (le_of_lt (hreal comp hcomp)))
      ⟨comp₀, hcomp₀, mul_pos hy₀ (hreal comp₀ hcomp₀)⟩
  linarith [hdual W]

/-! ### 2. Explicit Quartic Copy Counterexample (`L₀ = 5`, `L = 10`) -/

/-- The quartic relative score function `g(d) = d - 2 (d - 3)^2 (d - 5)^2`. -/
def quarticScore (d : ℝ) : ℝ :=
  d - 2 * (d - 3) ^ 2 * (d - 5) ^ 2

/-- Degree-4 monomial positional feature map `φ₄(i) = (1, i, i^2, i^3, i^4)`. -/
def φ₄ (i : ℕ) : Fin 5 → ℝ :=
  ![1, (i : ℝ), (i : ℝ) ^ 2, (i : ℝ) ^ 3, (i : ℝ) ^ 4]

/-- Explicit `5 × 5` bilinear weight matrix realizing `quarticScore (i - j)`. -/
def W₄ : Matrix (Fin 5) (Fin 5) ℝ :=
  ![ ![-450, -481, -188, -32, -2],
     ![ 481,  376,   96,   8,  0],
     ![-188,  -96,  -12,   0,  0],
     ![  32,    8,    0,   0,  0],
     ![  -2,    0,    0,   0,  0] ]

/-- Exact bilinear realization: `logit φ₄ W₄ i j = g(i - j)` for all `i, j : ℕ`. -/
theorem quartic_bilinear_repr (i j : ℕ) :
    logit φ₄ W₄ i j = quarticScore ((i : ℝ) - (j : ℝ)) := by
  unfold logit quarticScore φ₄ W₄
  simp only [Fin.sum_univ_five, Fin.isValue, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.cons_val]
  ring

/-- Margin check on all offsets `d ∈ {0, 1, 2, 4}` versus target offset `3` (for `L₀ = 5`). -/
lemma quarticScore_old_offsets {d : ℕ} (hd : d ≤ 4) (hne : d ≠ 3) :
    1 ≤ quarticScore 3 - quarticScore (d : ℝ) := by
  have hcases : d = 0 ∨ d = 1 ∨ d = 2 ∨ d = 4 := by omega
  rcases hcases with rfl | rfl | rfl | rfl <;> unfold quarticScore <;> norm_num

/-- **Old-length margin `1` at `L₀ = 5`**: on every scored row `i ∈ {3, 4}` with target `i - 3`
and every causal competitor `0 ≤ j ≤ i` (`j ≠ i - 3`), `s(i, i - 3) - s(i, j) ≥ 1`. -/
theorem quartic_old_margin {i j : ℕ} (hi : i = 3 ∨ i = 4) (hj : j ≤ i) (hne : j ≠ i - 3) :
    1 ≤ logit φ₄ W₄ i (i - 3) - logit φ₄ W₄ i j := by
  rw [quartic_bilinear_repr, quartic_bilinear_repr]
  have htar : (i : ℝ) - ((i - 3 : ℕ) : ℝ) = 3 := by
    have : 3 ≤ i := by omega
    rw [Nat.cast_sub this]; ring
  have hcomp : (i : ℝ) - (j : ℝ) = ((i - j : ℕ) : ℝ) := by
    rw [Nat.cast_sub hj]
  rw [htar, hcomp]
  exact quarticScore_old_offsets (by omega) (by omega)

/-- The old-length minimum margin `1` is attained at `(i, j) = (4, 0)` (`g(3) - g(4) = 1`). -/
theorem quartic_old_margin_exact :
    logit φ₄ W₄ 4 1 - logit φ₄ W₄ 4 0 = 1 := by
  rw [quartic_bilinear_repr, quartic_bilinear_repr]
  unfold quarticScore
  norm_num

/-- Margin check on all offsets `d ∈ {0..9} \ {5}` versus target offset `5` (for `L = 10`). -/
lemma quarticScore_new_offsets {d : ℕ} (hd : d ≤ 9) (hne : d ≠ 5) :
    2 ≤ quarticScore 5 - quarticScore (d : ℝ) := by
  have hcases : d = 0 ∨ d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 7 ∨ d = 8 ∨ d = 9 := by omega
  rcases hcases with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    unfold quarticScore <;> norm_num

/-- **New-length margin `2` at `L = 10`**: on every scored row `i ∈ {5, 6, 7, 8, 9}` with target
`i - 5` and every causal competitor `0 ≤ j ≤ i` (`j ≠ i - 5`), `s(i, i - 5) - s(i, j) ≥ 2`. -/
theorem quartic_new_margin {i j : ℕ} (hi5 : 5 ≤ i) (hi9 : i ≤ 9) (hj : j ≤ i)
    (hne : j ≠ i - 5) :
    2 ≤ logit φ₄ W₄ i (i - 5) - logit φ₄ W₄ i j := by
  rw [quartic_bilinear_repr, quartic_bilinear_repr]
  have htar : (i : ℝ) - ((i - 5 : ℕ) : ℝ) = 5 := by
    rw [Nat.cast_sub hi5]; ring
  have hcomp : (i : ℝ) - (j : ℝ) = ((i - j : ℕ) : ℝ) := by
    rw [Nat.cast_sub hj]
  rw [htar, hcomp]
  exact quarticScore_new_offsets (by omega) (by omega)

/-- The new-length minimum margin `2` is attained at `(i, j) = (5, 2)` (`g(5) - g(3) = 2`). -/
theorem quartic_new_margin_exact :
    logit φ₄ W₄ 5 0 - logit φ₄ W₄ 5 2 = 2 := by
  rw [quartic_bilinear_repr, quartic_bilinear_repr]
  unfold quarticScore
  norm_num

/-! ### 3. Fifth-Forward-Difference Non-Realizability Witness (`-5 ≠ 0`) -/

/-- Fifth forward difference operator `Δ⁵ f(0) = f(5) - 5 f(4) + 10 f(3) - 10 f(2) + 5 f(1) - f(0)`. -/
def fwdDiff5 (f : ℕ → ℝ) : ℝ :=
  f 5 - 5 * f 4 + 10 * f 3 - 10 * f 2 + 5 * f 1 - f 0

/-- The 5th forward difference annihilates every row `j ↦ logit φ₄ W i j` for every `5 × 5`
weight matrix `W` and every query position `i`. -/
theorem fwdDiff5_annihilates_quartic (W : Matrix (Fin 5) (Fin 5) ℝ) (i : ℕ) :
    fwdDiff5 (fun j => logit φ₄ W i j) = 0 := by
  unfold fwdDiff5 logit φ₄
  simp only [Fin.sum_univ_five, Fin.isValue, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.cons_val, Nat.cast_zero, Nat.cast_one, Nat.cast_ofNat]
  ring

/-- One-hot target row on query `i = 9` at length `L = 10` (target key `j = 4`). -/
def targetRow9 (j : ℕ) : ℝ :=
  if j = 4 then 1 else 0

/-- Exact evaluation of `Δ⁵` on `targetRow9`: `Δ⁵(targetRow9) = -5`. -/
theorem fwdDiff5_targetRow9 : fwdDiff5 targetRow9 = -5 := by
  unfold fwdDiff5 targetRow9
  norm_num

/-- **Exact non-realizability witness**: no `5 × 5` matrix `W` can realize `targetRow9` on keys
`0..5` (and hence not on causal keys `0..9`), because `0 = Δ⁵(logit) ≠ Δ⁵(targetRow9) = -5`. -/
theorem one_hot_not_realizable :
    ¬ ∃ W : Matrix (Fin 5) (Fin 5) ℝ, ∀ j ≤ 5, logit φ₄ W 9 j = targetRow9 j := by
  rintro ⟨W, hW⟩
  have h1 : fwdDiff5 (fun j => logit φ₄ W 9 j) = fwdDiff5 targetRow9 := by
    unfold fwdDiff5
    dsimp only
    rw [hW 5 (by omega), hW 4 (by omega), hW 3 (by omega), hW 2 (by omega),
      hW 1 (by omega), hW 0 (by omega)]
  rw [fwdDiff5_annihilates_quartic W 9, fwdDiff5_targetRow9] at h1
  norm_num at h1

end PELengthGen
