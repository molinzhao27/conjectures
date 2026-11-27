import Mathlib

/-!
# Bounds, Exact Optima, and Dual Formulations for the Seven-Atom Circle Moment Problem
(Guangjian Zhang, arXiv:2608.30254)

We formalize in Lean 4 + Mathlib (complete kernel-checked proofs, standard foundations only) the exact algebraic,
variational, KKT, and rational-certificate theorems from `solutions/weighted_data_vector/solution.md`:

1. **Planar covariance mean lemma `H(S) = det(S) / tr(S)` (Lemma 1)**:
   - `H_variational_le`, `H_variational_attained`: For every symmetric PSD `2 × 2` matrix `S` and
     every `L : Matrix (Fin 2) (Fin 2) ℝ` with `L.trace = 1`, `H S ≤ (Lᵀ * S * L).trace`,
     with equality attained by an explicit `L` of trace `1`.
   - `H_mono_loewner`: Loewner monotonicity `S₁ ⪯ S₂ → H S₁ ≤ H S₂`.
   - `H_concave`: Concavity `t H(S₁) + (1 - t) H(S₂) ≤ H(t S₁ + (1 - t) S₂)` for `t ∈ [0, 1]`.
   - `H_le_quarter_trace`: `H S ≤ S.trace / 4`.

2. **Anisotropic 3-point expectation inequality (Theorem 2)**:
   - `expectation_H_bound`: For any finite probability distribution `(pᵢ, yᵢ)` in `ℝ²` with
     `Σ = ∑ pᵢ yᵢ yᵢᵀ`, `tr(Σ) = 1`, and anisotropy `d = √((Σ₀₀ - Σ₁₁)² + 4 Σ₀₁²)`,
     `∑ pᵢ H(2 Σ + yᵢ yᵢᵀ) ≤ 2/3 - (7/12) d²`.
   - `three_point_expectation_bound`:
     `(1 / 2) * (1 + ∑ pᵢ H(2 Σ + yᵢ yᵢᵀ)) ≤ 5/6 - (7/24) d² ≤ 5/6`
     (giving `F_weighted(2, 2, 3) ≤ 11 / 6`), and the concavity bound `≤ 7/8 - (3/8) d²`
     (giving `≤ 5/8` whenever `d² ≥ 2/3`).

3. **Two real output lines 3-point bound (Theorem 4)**:
   - `two_output_lines_bound`: For `a, b ≥ 0`, `t ∈ (0, 1)`, `(1 - t) a + t b ≤ 1`,
     `min (b / 2) (t * a) ≤ t / (1 - t + 2 t²) ≤ 1 / (2√2 - 1) < 5 / 8`.

4. **Isotropic equal-energy 6-atom algebraic identity & bound (Theorem 5)**:
   - `isotropic_six_atom_identity`: For `f(t) = t / (1 - t)²` and `a² ≠ 4`,
     `(1/6)(f((1+a)/3) + f((1-a)/3) - a(f((1+a)/3) - f((1-a)/3))) - 1/4 = -a²(5a² + 4) / (4(4 - a²)²) ≤ 0`.
   - `isotropic_six_atom_le_quarter`: Under `|a| ≤ 1`, `a + b + c = 0`, `0 ≤ a * c`,
     `(1/6)(f((1+a)/3) + f((1-a)/3) + b(f((1+a)/3) - f((1-a)/3))) ≤ 1/4`.
   - `six_vectors_nonneg_inner_product`: Rankin's (1955) 6-vectors-in-`ℝ⁴` lemma: any 6 vectors
     in `ℝ⁴` contain a distinct pair with nonnegative inner product.

5. **Simultaneous 7-atom original-weight obstruction at both budgets (Proposition 10)**:
   - `seven_atom_moment_conditions`: Exact verification of zero conditional mean and conditional
     energy `e' = 1 + 1/40500` on the second feature line.
   - `seven_atom_orig_weight_k3`, `seven_atom_orig_weight_k4`: Exhaustive verification over all
     spanning original-weight selections showing exact optima `700205 / 1107014 > 5 / 8` (at `k = 3`)
     and `295205 / 1107014 > 1 / 4` (at `k = 4`).
   - `seven_atom_reweighted_beats_both`: Equal-weight reweighting of one cross pair achieves
     `3320249 / 5535070 < 5 / 8` (at `k = 3`) and `1295249 / 5535070 < 1 / 4` (at `k = 4`).

6. **Exact rational dual certificates for a 7-atom continuum (Proposition 8)**:
   - `dual_cert_orthogonality`: Exact rational verification `Xᵀ * Y = 0`.
   - `dual_cert_k3`: Exact `k = 3` mixed certificate (supports `{1,5,6}` and `{3,5,6}` in ratio
     `93 : 7`) certifies max dual ratio `560659 / 7066875 < 1 / 10` across EVERY compatible
     nonnegative test mass vector `p` with `∑ pᵢ yᵢ xᵢᵀ = 0`.
   - `dual_cert_k4`: Exact `k = 4` mixed certificate (supports `{1,3,4,5}` and `{1,3,5,6}` in ratio
     `33 : 67`) certifies max dual ratio `381447541 / 7312728225 < 1 / 16` across EVERY compatible
     nonnegative test mass vector `p` with `∑ pᵢ yᵢ xᵢᵀ = 0`.

Compactness/uniformity for `Γ₄ < 1/2` (Theorem 3), Sion's minimax strong duality (Theorem 7),
and the full 5-atom classification & cyclotomic dihedral KKT tables (Propositions 6 & 9) are
proved in prose and verified in `solutions/weighted_data_vector/solution.md`, `proof.pdf`, and `explore.py`.
-/

open Matrix Finset BigOperators

namespace WeightedDataVector

/-! ### 1. Planar Covariance Mean Lemma: `H(S) = det(S) / tr(S)` (Lemma 1) -/

/-- Symmetric `2 × 2` positive semidefinite predicate. -/
def IsSymPSD2 (S : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  S 0 1 = S 1 0 ∧ 0 ≤ S 0 0 ∧ 0 ≤ S 1 1 ∧ 0 ≤ S 0 0 * S 1 1 - S 0 1 * S 1 0

/-- Loewner order `S₁ ⪯ S₂` on `2 × 2` real matrices. -/
def LoewnerLE (S₁ S₂ : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  IsSymPSD2 (S₂ - S₁)

/-- `H(S) = det(S) / tr(S)` for a `2 × 2` matrix (equals `0` when `tr(S) = 0` or `det(S) = 0`). -/
noncomputable def H (S : Matrix (Fin 2) (Fin 2) ℝ) : ℝ :=
  S.det / S.trace

lemma quadForm_nonneg_of_isSymPSD2 {S : Matrix (Fin 2) (Fin 2) ℝ} (hS : IsSymPSD2 S)
    (x y : ℝ) : 0 ≤ S 0 0 * x ^ 2 + 2 * S 0 1 * x * y + S 1 1 * y ^ 2 := by
  obtain ⟨hsym, ha, hc, hdet⟩ := hS
  rw [← hsym] at hdet
  by_cases ha0 : S 0 0 = 0
  · rw [ha0, zero_mul] at hdet
    have hb0 : S 0 1 = 0 := by nlinarith
    rw [ha0, hb0]; nlinarith [sq_nonneg y]
  · have hap : 0 < S 0 0 := lt_of_le_of_ne ha (Ne.symm ha0)
    have hid : S 0 0 * (S 0 0 * x ^ 2 + 2 * S 0 1 * x * y + S 1 1 * y ^ 2) =
        (S 0 0 * x + S 0 1 * y) ^ 2 + (S 0 0 * S 1 1 - S 0 1 * S 0 1) * y ^ 2 := by ring
    have hrhs : 0 ≤ (S 0 0 * x + S 0 1 * y) ^ 2 + (S 0 0 * S 1 1 - S 0 1 * S 0 1) * y ^ 2 :=
      add_nonneg (sq_nonneg _) (mul_nonneg hdet (sq_nonneg _))
    nlinarith

lemma trace_conj_nonneg_of_isSymPSD2 {S : Matrix (Fin 2) (Fin 2) ℝ} (hS : IsSymPSD2 S)
    (R : Matrix (Fin 2) (Fin 2) ℝ) : 0 ≤ (Rᵀ * S * R).trace := by
  have h1 := quadForm_nonneg_of_isSymPSD2 hS (R 0 0) (R 1 0)
  have h2 := quadForm_nonneg_of_isSymPSD2 hS (R 0 1) (R 1 1)
  have hexp : (Rᵀ * S * R).trace =
      (S 0 0 * (R 0 0) ^ 2 + 2 * S 0 1 * (R 0 0) * (R 1 0) + S 1 1 * (R 1 0) ^ 2) +
      (S 0 0 * (R 0 1) ^ 2 + 2 * S 0 1 * (R 0 1) * (R 1 1) + S 1 1 * (R 1 1) ^ 2) := by
    simp only [Matrix.trace, Matrix.diag_apply, Fin.sum_univ_two, Matrix.mul_apply,
      Matrix.transpose_apply, hS.1]
    ring
  linarith

/-- **Lemma 1 (Variational lower bound)**: For every symmetric PSD `2 × 2` matrix `S` and every
`L` with `tr(L) = 1`, `H(S) ≤ tr(Lᵀ S L)`. -/
theorem H_variational_le {S : Matrix (Fin 2) (Fin 2) ℝ} (hS : IsSymPSD2 S)
    {L : Matrix (Fin 2) (Fin 2) ℝ} (hL : L.trace = 1) :
    H S ≤ (Lᵀ * S * L).trace := by
  obtain ⟨hsym, ha, hc, hdet⟩ := hS
  have htr_eq : S.trace = S 0 0 + S 1 1 := by simp [Matrix.trace, Fin.sum_univ_two]
  have hdet_eq : S.det = S 0 0 * S 1 1 - S 0 1 * S 1 0 := Matrix.det_fin_two S
  have hLtr : L 0 0 + L 1 1 = 1 := by simpa [Matrix.trace, Fin.sum_univ_two] using hL
  by_cases htr0 : S 0 0 + S 1 1 = 0
  · have hH0 : H S = 0 := by simp [H, htr_eq, htr0]
    rw [hH0]
    exact trace_conj_nonneg_of_isSymPSD2 ⟨hsym, ha, hc, hdet⟩ L
  · have htrp : 0 < S 0 0 + S 1 1 := lt_of_le_of_ne (by linarith) (Ne.symm htr0)
    -- Decompose `L = L₀ + R` where `L₀ = adj(S) / tr(S)` has `tr(L₀) = 1`, `tr(R) = 0`.
    let t := S 0 0 + S 1 1
    let L₀ : Matrix (Fin 2) (Fin 2) ℝ := !![S 1 1 / t, -S 0 1 / t; -S 0 1 / t, S 0 0 / t]
    let R : Matrix (Fin 2) (Fin 2) ℝ := L - L₀
    have hR_nn : 0 ≤ (Rᵀ * S * R).trace :=
      trace_conj_nonneg_of_isSymPSD2 ⟨hsym, ha, hc, hdet⟩ R
    have hdiff : (Lᵀ * S * L).trace - H S = (Rᵀ * S * R).trace := by
      simp only [H, hdet_eq, Matrix.trace, Matrix.diag_apply, Fin.sum_univ_two,
        Matrix.mul_apply, Matrix.transpose_apply, R, L₀, Matrix.sub_apply, Matrix.of_apply,
        Matrix.cons_val_zero, Matrix.cons_val_one, t]
      have h11 : L 1 1 = 1 - L 0 0 := by linarith
      rw [← hsym, h11]
      field_simp [ne_of_gt htrp]
      ring
    linarith

/-- **Lemma 1 (Variational attainment)**: For every symmetric PSD `2 × 2` matrix `S`, there exists
`L` with `tr(L) = 1` achieving `tr(Lᵀ S L) = H(S)`. -/
theorem H_variational_attained {S : Matrix (Fin 2) (Fin 2) ℝ} (hS : IsSymPSD2 S) :
    ∃ L : Matrix (Fin 2) (Fin 2) ℝ, L.trace = 1 ∧ (Lᵀ * S * L).trace = H S := by
  obtain ⟨hsym, ha, hc, hdet⟩ := hS
  have hdet_eq : S.det = S 0 0 * S 1 1 - S 0 1 * S 1 0 := Matrix.det_fin_two S
  by_cases htr0 : S 0 0 + S 1 1 = 0
  · have ha0 : S 0 0 = 0 := by linarith
    have hc0 : S 1 1 = 0 := by linarith
    have hb0 : S 0 1 = 0 := by rw [← hsym, ha0, zero_mul] at hdet; nlinarith
    refine ⟨!![1, 0; 0, 0], by simp [Matrix.trace, Fin.sum_univ_two], ?_⟩
    simp [H, Matrix.trace, Fin.sum_univ_two, Matrix.mul_apply, ha0, hc0, hb0, ← hsym]
  · have htrp : 0 < S 0 0 + S 1 1 := lt_of_le_of_ne (by linarith) (Ne.symm htr0)
    let t := S 0 0 + S 1 1
    refine ⟨!![S 1 1 / t, -S 0 1 / t; -S 0 1 / t, S 0 0 / t], ?_, ?_⟩
    · simp only [Matrix.trace, Matrix.diag_apply, Fin.sum_univ_two, Matrix.of_apply,
        Matrix.cons_val_zero, Matrix.cons_val_one, t]
      field_simp [ne_of_gt htrp]; ring
    · simp only [H, hdet_eq, Matrix.trace, Matrix.diag_apply, Fin.sum_univ_two,
        Matrix.mul_apply, Matrix.transpose_apply, Matrix.of_apply, Matrix.cons_val_zero,
        Matrix.cons_val_one, t]
      rw [← hsym]
      field_simp [ne_of_gt htrp]
      ring

/-- **Lemma 1 (Loewner monotonicity of `H`)**: If `S₁, S₂` are symmetric PSD and `S₁ ⪯ S₂`,
then `H(S₁) ≤ H(S₂)`. -/
theorem H_mono_loewner {S₁ S₂ : Matrix (Fin 2) (Fin 2) ℝ}
    (h1 : IsSymPSD2 S₁) (h2 : IsSymPSD2 S₂) (hle : LoewnerLE S₁ S₂) :
    H S₁ ≤ H S₂ := by
  obtain ⟨L, hLtr, hLeq⟩ := H_variational_attained h2
  have hlow := H_variational_le h1 hLtr
  have hdiff : (Lᵀ * S₂ * L).trace - (Lᵀ * S₁ * L).trace = (Lᵀ * (S₂ - S₁) * L).trace := by
    rw [Matrix.mul_sub, Matrix.sub_mul, Matrix.trace_sub]
  have hnn : 0 ≤ (Lᵀ * (S₂ - S₁) * L).trace := trace_conj_nonneg_of_isSymPSD2 hle L
  linarith

/-- **Lemma 1 (Concavity of `H`)**: For symmetric PSD `S₁, S₂` and `t ∈ [0, 1]`,
`t H(S₁) + (1 - t) H(S₂) ≤ H(t S₁ + (1 - t) S₂)`. -/
theorem H_concave {S₁ S₂ : Matrix (Fin 2) (Fin 2) ℝ} (h1 : IsSymPSD2 S₁) (h2 : IsSymPSD2 S₂)
    {t : ℝ} (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    t * H S₁ + (1 - t) * H S₂ ≤ H (t • S₁ + (1 - t) • S₂) := by
  -- First show `t • S₁ + (1 - t) • S₂` is symmetric PSD.
  have hconv_psd : IsSymPSD2 (t • S₁ + (1 - t) • S₂) := by
    obtain ⟨hs1, ha1, hc1, hd1⟩ := h1
    obtain ⟨hs2, ha2, hc2, hd2⟩ := h2
    refine ⟨?_, ?_, ?_, ?_⟩
    · simp [hs1, hs2]
    · simp only [Matrix.add_apply, Matrix.smul_apply, smul_eq_mul]; nlinarith
    · simp only [Matrix.add_apply, Matrix.smul_apply, smul_eq_mul]; nlinarith
    · simp only [Matrix.add_apply, Matrix.smul_apply, smul_eq_mul]
      rw [← hs1] at hd1 ⊢
      rw [← hs2] at hd2 ⊢
      -- `det(t S₁ + (1-t) S₂) = t² det S₁ + (1-t)² det S₂ + t(1-t)(a₁ c₂ + a₂ c₁ - 2 b₁ b₂)`.
      have hcross : 0 ≤ S₁ 0 0 * S₂ 1 1 + S₂ 0 0 * S₁ 1 1 - 2 * S₁ 0 1 * S₂ 0 1 := by
        have hsq1 : (S₁ 0 1) ^ 2 ≤ S₁ 0 0 * S₁ 1 1 := by linarith
        have hsq2 : (S₂ 0 1) ^ 2 ≤ S₂ 0 0 * S₂ 1 1 := by linarith
        have hprod : (S₁ 0 1 * S₂ 0 1) ^ 2 ≤ (S₁ 0 0 * S₂ 1 1) * (S₂ 0 0 * S₁ 1 1) := by
          have : (S₁ 0 1 * S₂ 0 1) ^ 2 = (S₁ 0 1) ^ 2 * (S₂ 0 1) ^ 2 := by ring
          have : (S₁ 0 0 * S₂ 1 1) * (S₂ 0 0 * S₁ 1 1) = (S₁ 0 0 * S₁ 1 1) * (S₂ 0 0 * S₂ 1 1) := by ring
          nlinarith [mul_nonneg ha1 hc1, mul_nonneg ha2 hc2]
        have ham : 4 * ((S₁ 0 0 * S₂ 1 1) * (S₂ 0 0 * S₁ 1 1)) ≤
            (S₁ 0 0 * S₂ 1 1 + S₂ 0 0 * S₁ 1 1) ^ 2 := by
          nlinarith [sq_nonneg (S₁ 0 0 * S₂ 1 1 - S₂ 0 0 * S₁ 1 1)]
        have hnn : 0 ≤ S₁ 0 0 * S₂ 1 1 + S₂ 0 0 * S₁ 1 1 := by nlinarith
        nlinarith [sq_nonneg (2 * (S₁ 0 1 * S₂ 0 1))]
      have hexp : (t * S₁ 0 0 + (1 - t) * S₂ 0 0) * (t * S₁ 1 1 + (1 - t) * S₂ 1 1) -
          (t * S₁ 0 1 + (1 - t) * S₂ 0 1) * (t * S₁ 0 1 + (1 - t) * S₂ 0 1) =
          t ^ 2 * (S₁ 0 0 * S₁ 1 1 - S₁ 0 1 * S₁ 0 1) +
          (1 - t) ^ 2 * (S₂ 0 0 * S₂ 1 1 - S₂ 0 1 * S₂ 0 1) +
          t * (1 - t) * (S₁ 0 0 * S₂ 1 1 + S₂ 0 0 * S₁ 1 1 - 2 * S₁ 0 1 * S₂ 0 1) := by ring
      rw [hexp]
      nlinarith [mul_nonneg (sq_nonneg t) hd1, mul_nonneg (sq_nonneg (1 - t)) hd2,
        mul_nonneg (mul_nonneg ht0 (sub_nonneg.mpr ht1)) hcross]
  obtain ⟨L, hLtr, hLeq⟩ := H_variational_attained hconv_psd
  have hlow1 := H_variational_le h1 hLtr
  have hlow2 := H_variational_le h2 hLtr
  have hlin : (Lᵀ * (t • S₁ + (1 - t) • S₂) * L).trace =
      t * (Lᵀ * S₁ * L).trace + (1 - t) * (Lᵀ * S₂ * L).trace := by
    rw [Matrix.mul_add, Matrix.add_mul, Matrix.mul_smul, Matrix.smul_mul,
      Matrix.mul_smul, Matrix.smul_mul, Matrix.trace_add, Matrix.trace_smul, Matrix.trace_smul]
    simp [smul_eq_mul]
  rw [← hLeq, hlin]
  nlinarith

/-- `H(S) ≤ tr(S) / 4` for every symmetric PSD `2 × 2` matrix `S`. -/
theorem H_le_quarter_trace {S : Matrix (Fin 2) (Fin 2) ℝ} (hS : IsSymPSD2 S) :
    H S ≤ S.trace / 4 := by
  have hL : ((1 / 2 : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℝ)).trace = 1 := by
    simp [Matrix.trace]
  have h := H_variational_le hS hL
  have hrhs : (((1 / 2 : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℝ))ᵀ * S *
      ((1 / 2 : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℝ))).trace = S.trace / 4 := by
    simp [Matrix.trace, Fin.sum_univ_two]
    ring
  linarith

/-! ### 2. Exact Anisotropic 3-Point Expectation Inequality (Theorem 2) -/

/-- Rank-one outer product `y yᵀ` of a planar vector `y : ℝ × ℝ`. -/
def outer2 (y : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![y.1 * y.1, y.1 * y.2; y.2 * y.1, y.2 * y.2]

/-- Helper for Theorem 2: AM-GM square completion `(7/12) dsq ≤ U/9 + dsq*(A + 1/3) - (1/3)*X`. -/
private lemma amgm_complete_square {A U V X dsq : ℝ}
    (hU : 0 ≤ U) (hV : 0 ≤ V) (hdsq : 0 ≤ dsq)
    (hVeq : V = 4 * A - 1) (hcs : X ^ 2 ≤ dsq * U * V) :
    (7 / 12 : ℝ) * dsq ≤ U / 9 + dsq * (A + 1 / 3) - (1 / 3) * X := by
  have h1 : X ^ 2 ≤ (U / 3 + 3 / 4 * dsq * V) ^ 2 := by
    have hsq : 0 ≤ (U / 3 - 3 / 4 * dsq * V) ^ 2 := sq_nonneg _
    have hid : (U / 3 + 3 / 4 * dsq * V) ^ 2 - dsq * U * V = (U / 3 - 3 / 4 * dsq * V) ^ 2 := by ring
    linarith
  have h2 : 0 ≤ U / 3 + 3 / 4 * dsq * V := by positivity
  have h3 : X ≤ U / 3 + 3 / 4 * dsq * V := by
    by_contra hgt; push_neg at hgt
    have hlt : (U / 3 + 3 / 4 * dsq * V) ^ 2 < X ^ 2 := by
      have : 0 < (X - (U / 3 + 3 / 4 * dsq * V)) * (X + (U / 3 + 3 / 4 * dsq * V)) :=
        mul_pos (by linarith) (by linarith)
      nlinarith
    linarith
  have h4 : (1 / 3) * X ≤ U / 9 + (1 / 4) * dsq * V := by linarith
  have hEq : U / 9 + dsq * (A + 1 / 3) - (1 / 3) * X - (7 / 12) * dsq =
      (U / 9 + (1 / 4) * dsq * V - (1 / 3) * X) + dsq * (A - 1 / 4 - V / 4) := by ring
  have hAV : A - 1 / 4 - V / 4 = 0 := by linarith
  rw [hAV, mul_zero, add_zero] at hEq
  linarith

/-- Helper for Theorem 2: concavity bound `∑ p_i H(2 Σ + y_i y_iᵀ) ≤ (3/4)(1 - d²)`. -/
private lemma expectation_H_concavity_bound {ι : Type*} [Fintype ι]
    (p : ι → ℝ) (y : ι → ℝ × ℝ)
    (hp_nn : ∀ i, 0 ≤ p i) (hp_sum : ∑ i, p i = 1)
    (hE : ∑ i, p i * ((y i).1 ^ 2 + (y i).2 ^ 2) = 1) :
    let Sm : Matrix (Fin 2) (Fin 2) ℝ := ∑ i, p i • outer2 (y i)
    let dsq : ℝ := (Sm 0 0 - Sm 1 1) ^ 2 + 4 * (Sm 0 1) ^ 2
    ∑ i, p i * H ((2 : ℝ) • Sm + outer2 (y i)) ≤ 3 / 4 * (1 - dsq) := by
  intro Sm dsq
  have hS00 : Sm 0 0 = ∑ i, p i * (y i).1 ^ 2 := by simp [Sm, outer2, Matrix.sum_apply, sq]
  have hS11 : Sm 1 1 = ∑ i, p i * (y i).2 ^ 2 := by simp [Sm, outer2, Matrix.sum_apply, sq]
  have hS01 : Sm 0 1 = ∑ i, p i * ((y i).1 * (y i).2) := by simp [Sm, outer2, Matrix.sum_apply]
  have hS10 : Sm 1 0 = Sm 0 1 := by simp [Sm, outer2, Matrix.sum_apply, mul_comm]
  have hStr : Sm 0 0 + Sm 1 1 = 1 := by
    rw [hS00, hS11, ← Finset.sum_add_distrib]
    simpa [mul_add] using hE
  have hS_psd : IsSymPSD2 Sm := by
    refine ⟨hS10.symm, ?_, ?_, ?_⟩
    · rw [hS00]; exact Finset.sum_nonneg fun i _ => mul_nonneg (hp_nn i) (sq_nonneg _)
    · rw [hS11]; exact Finset.sum_nonneg fun i _ => mul_nonneg (hp_nn i) (sq_nonneg _)
    · rw [hS10, hS00, hS11, hS01]
      have hcs := Finset.sum_mul_sq_le_sq_mul_sq Finset.univ
        (fun i => Real.sqrt (p i) * (y i).1) (fun i => Real.sqrt (p i) * (y i).2)
      have hf : ∑ i, (Real.sqrt (p i) * (y i).1) ^ 2 = ∑ i, p i * (y i).1 ^ 2 := by
        refine Finset.sum_congr rfl fun i _ => by rw [mul_pow, Real.sq_sqrt (hp_nn i)]
      have hg : ∑ i, (Real.sqrt (p i) * (y i).2) ^ 2 = ∑ i, p i * (y i).2 ^ 2 := by
        refine Finset.sum_congr rfl fun i _ => by rw [mul_pow, Real.sq_sqrt (hp_nn i)]
      have hfg : ∑ i, (Real.sqrt (p i) * (y i).1) * (Real.sqrt (p i) * (y i).2) =
          ∑ i, p i * ((y i).1 * (y i).2) := by
        refine Finset.sum_congr rfl fun i _ => by
          calc (Real.sqrt (p i) * (y i).1) * (Real.sqrt (p i) * (y i).2)
              = (Real.sqrt (p i)) ^ 2 * ((y i).1 * (y i).2) := by ring
            _ = p i * ((y i).1 * (y i).2) := by rw [Real.sq_sqrt (hp_nn i)]
      rw [hf, hg, hfg] at hcs
      nlinarith
  have h3S_psd : IsSymPSD2 ((3 : ℝ) • Sm) := by
    obtain ⟨hs, ha, hc, hd⟩ := hS_psd
    refine ⟨by simp [hs], by simp; linarith, by simp; linarith, ?_⟩
    simp only [Matrix.smul_apply, smul_eq_mul]; nlinarith
  obtain ⟨L, hLtr, hLeq⟩ := H_variational_attained h3S_psd
  have hM_psd : ∀ i, IsSymPSD2 ((2 : ℝ) • Sm + outer2 (y i)) := by
    intro i
    have ⟨hs, ha, hc, hd⟩ := hS_psd
    refine ⟨?_, ?_, ?_, ?_⟩
    · simp [outer2, hs, mul_comm]
    · simp [outer2]; nlinarith [sq_nonneg (y i).1]
    · simp [outer2]; nlinarith [sq_nonneg (y i).2]
    · simp [outer2]
      rw [← hs] at hd ⊢
      have hq := quadForm_nonneg_of_isSymPSD2 hS_psd (y i).2 (-(y i).1)
      nlinarith
  have hle_H3 : ∑ i, p i * H ((2 : ℝ) • Sm + outer2 (y i)) ≤ H ((3 : ℝ) • Sm) := by
    calc ∑ i, p i * H ((2 : ℝ) • Sm + outer2 (y i))
        ≤ ∑ i, p i * (Lᵀ * ((2 : ℝ) • Sm + outer2 (y i)) * L).trace :=
          Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (H_variational_le (hM_psd i) hLtr) (hp_nn i)
      _ = (Lᵀ * (∑ i, p i • ((2 : ℝ) • Sm + outer2 (y i))) * L).trace := by
          rw [Matrix.mul_sum, Matrix.sum_mul, Matrix.trace_sum]
          refine Finset.sum_congr rfl fun i _ => ?_
          rw [Matrix.mul_smul, Matrix.smul_mul, Matrix.trace_smul, smul_eq_mul]
      _ = (Lᵀ * ((3 : ℝ) • Sm) * L).trace := by
          congr 2
          have hsum_eq : ∑ i, p i • ((2 : ℝ) • Sm + outer2 (y i)) = (3 : ℝ) • Sm := by
            simp only [smul_add, Finset.sum_add_distrib, ← Finset.sum_smul, hp_sum, one_smul]
            change (2 : ℝ) • Sm + Sm = (3 : ℝ) • Sm
            ext r c; simp; ring
          rw [hsum_eq]
      _ = H ((3 : ℝ) • Sm) := hLeq
  have hH3_eq : H ((3 : ℝ) • Sm) = 3 / 4 * (1 - dsq) := by
    simp only [H, Matrix.det_fin_two, Matrix.trace, Matrix.diag_apply, Fin.sum_univ_two,
      Matrix.smul_apply, smul_eq_mul, hS10, dsq]
    have hS11_eq : Sm 1 1 = 1 - Sm 0 0 := by linarith [hStr]
    rw [hS11_eq]
    ring
  linarith

/-- Cauchy–Schwarz cross-term bound helper for `three_point_expectation_bound`. -/
private lemma cs_cross_bound {ι : Type*} [Fintype ι]
    (p s w₁ w₂ : ι → ℝ) (z₁ z₂ : ℝ)
    (hp_nn : ∀ i, 0 ≤ p i) (hs_nn : ∀ i, 0 ≤ s i)
    (hnorm_w : ∀ i, (w₁ i) ^ 2 + (w₂ i) ^ 2 = (s i) ^ 2) :
    (∑ i, p i * ((z₁ * w₁ i + z₂ * w₂ i) * (s i - 1) / (2 + s i))) ^ 2 ≤
      (z₁ ^ 2 + z₂ ^ 2) *
        (∑ i, p i * ((s i - 1) ^ 2 / (2 + s i))) *
        (∑ i, p i * ((s i) ^ 2 / (2 + s i))) := by
  set dsq := z₁ ^ 2 + z₂ ^ 2
  set U := ∑ i, p i * ((s i - 1) ^ 2 / (2 + s i))
  set V := ∑ i, p i * ((s i) ^ 2 / (2 + s i))
  have h2s_pos : ∀ i, 0 < 2 + s i := fun i => by linarith [hs_nn i]
  have hdsq_nn : 0 ≤ dsq := by positivity
  have hpt_abs : ∀ i, |z₁ * w₁ i + z₂ * w₂ i| ≤ Real.sqrt dsq * s i := by
    intro i
    have hsq : (z₁ * w₁ i + z₂ * w₂ i) ^ 2 ≤ dsq * (s i) ^ 2 := by
      rw [← hnorm_w i]; simp only [dsq]
      nlinarith [sq_nonneg (z₁ * w₂ i - z₂ * w₁ i)]
    have hnn : 0 ≤ Real.sqrt dsq * s i := mul_nonneg (Real.sqrt_nonneg _) (hs_nn i)
    have hr : (Real.sqrt dsq * s i) ^ 2 = dsq * (s i) ^ 2 := by
      rw [mul_pow, Real.sq_sqrt hdsq_nn]
    rw [← sq_abs] at hsq
    nlinarith [abs_nonneg (z₁ * w₁ i + z₂ * w₂ i)]
  have habs_sum : |∑ i, p i * ((z₁ * w₁ i + z₂ * w₂ i) * (s i - 1) / (2 + s i))| ≤
      Real.sqrt dsq * ∑ i, p i * (s i * |s i - 1| / (2 + s i)) := by
    calc |∑ i, p i * ((z₁ * w₁ i + z₂ * w₂ i) * (s i - 1) / (2 + s i))|
        ≤ ∑ i, |p i * ((z₁ * w₁ i + z₂ * w₂ i) * (s i - 1) / (2 + s i))| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ i, Real.sqrt dsq * (p i * (s i * |s i - 1| / (2 + s i))) := by
        refine Finset.sum_le_sum fun i _ => ?_
        rw [abs_mul, abs_div, abs_mul, abs_of_nonneg (hp_nn i), abs_of_pos (h2s_pos i)]
        have h1 := hpt_abs i
        have h3 : |z₁ * w₁ i + z₂ * w₂ i| * |s i - 1| / (2 + s i) ≤
            Real.sqrt dsq * s i * |s i - 1| / (2 + s i) :=
          div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right h1 (abs_nonneg _)) (le_of_lt (h2s_pos i))
        have h4 := mul_le_mul_of_nonneg_left h3 (hp_nn i)
        have hrew : p i * (Real.sqrt dsq * s i * |s i - 1| / (2 + s i)) =
            Real.sqrt dsq * (p i * (s i * |s i - 1| / (2 + s i))) := by ring
        linarith only [h4, hrew]
      _ = Real.sqrt dsq * ∑ i, p i * (s i * |s i - 1| / (2 + s i)) := (Finset.mul_sum _ _ _).symm
  have hcs_uv : (∑ i, p i * (s i * |s i - 1| / (2 + s i))) ^ 2 ≤ V * U := by
    let f : ι → ℝ := fun i => Real.sqrt (p i / (2 + s i)) * s i
    let g : ι → ℝ := fun i => Real.sqrt (p i / (2 + s i)) * |s i - 1|
    have hfg : ∑ i, f i * g i = ∑ i, p i * (s i * |s i - 1| / (2 + s i)) := by
      refine Finset.sum_congr rfl fun i _ => ?_
      have hnn : 0 ≤ p i / (2 + s i) := div_nonneg (hp_nn i) (le_of_lt (h2s_pos i))
      calc f i * g i = (Real.sqrt (p i / (2 + s i))) ^ 2 * (s i * |s i - 1|) := by dsimp [f, g]; ring
        _ = (p i / (2 + s i)) * (s i * |s i - 1|) := by rw [Real.sq_sqrt hnn]
        _ = p i * (s i * |s i - 1| / (2 + s i)) := by ring
    have hf2 : ∑ i, (f i) ^ 2 = V := by
      refine Finset.sum_congr rfl fun i _ => ?_
      have hnn : 0 ≤ p i / (2 + s i) := div_nonneg (hp_nn i) (le_of_lt (h2s_pos i))
      calc (f i) ^ 2 = (Real.sqrt (p i / (2 + s i))) ^ 2 * (s i) ^ 2 := by dsimp [f]; rw [mul_pow]
        _ = (p i / (2 + s i)) * (s i) ^ 2 := by rw [Real.sq_sqrt hnn]
        _ = p i * ((s i) ^ 2 / (2 + s i)) := by ring
    have hg2 : ∑ i, (g i) ^ 2 = U := by
      refine Finset.sum_congr rfl fun i _ => ?_
      have hnn : 0 ≤ p i / (2 + s i) := div_nonneg (hp_nn i) (le_of_lt (h2s_pos i))
      calc (g i) ^ 2 = (Real.sqrt (p i / (2 + s i))) ^ 2 * |s i - 1| ^ 2 := by dsimp [g]; rw [mul_pow]
        _ = (p i / (2 + s i)) * (s i - 1) ^ 2 := by rw [Real.sq_sqrt hnn, sq_abs]
        _ = p i * ((s i - 1) ^ 2 / (2 + s i)) := by ring
    rw [← hfg, ← hf2, ← hg2]
    exact Finset.sum_mul_sq_le_sq_mul_sq Finset.univ f g
  calc (∑ i, p i * ((z₁ * w₁ i + z₂ * w₂ i) * (s i - 1) / (2 + s i))) ^ 2
      = |∑ i, p i * ((z₁ * w₁ i + z₂ * w₂ i) * (s i - 1) / (2 + s i))| ^ 2 := (sq_abs _).symm
    _ ≤ (Real.sqrt dsq * ∑ i, p i * (s i * |s i - 1| / (2 + s i))) ^ 2 :=
      pow_le_pow_left₀ (abs_nonneg _) habs_sum 2
    _ = dsq * (∑ i, p i * (s i * |s i - 1| / (2 + s i))) ^ 2 := by
      rw [mul_pow, Real.sq_sqrt hdsq_nn]
    _ ≤ dsq * (V * U) := mul_le_mul_of_nonneg_left hcs_uv hdsq_nn
    _ = dsq * U * V := by ring

/-- **Theorem 2 (Exact expectation inequality)**:
For any finite probability distribution `(pᵢ, yᵢ)` in `ℝ²` with covariance
`Σ = ∑ pᵢ • outer2 (yᵢ)` of trace `1` and normalized anisotropy
`d = √((Σ 0 0 - Σ 1 1)² + 4 * (Σ 0 1)²)`:
1. `∑ pᵢ H(2 Σ + yᵢ yᵢᵀ) ≤ 2/3 - (7/12) d²`.
2. `(1/2) * (1 + ∑ pᵢ H(2 Σ + yᵢ yᵢᵀ)) ≤ 5/6 - (7/24) d² ≤ 5/6`
   (yielding `F_weighted(2, 2, 3) ≤ 11 / 6`).
3. `(1/2) * (1 + ∑ pᵢ H(2 Σ + yᵢ yᵢᵀ)) ≤ 7/8 - (3/8) d²`
   (yielding `≤ 5/8` whenever `d² ≥ 2/3`). -/
theorem three_point_expectation_bound {ι : Type*} [Fintype ι]
    (p : ι → ℝ) (y : ι → ℝ × ℝ)
    (hp_nn : ∀ i, 0 ≤ p i) (hp_sum : ∑ i, p i = 1)
    (hE : ∑ i, p i * ((y i).1 ^ 2 + (y i).2 ^ 2) = 1) :
    let Sm : Matrix (Fin 2) (Fin 2) ℝ := ∑ i, p i • outer2 (y i)
    let dsq : ℝ := (Sm 0 0 - Sm 1 1) ^ 2 + 4 * (Sm 0 1) ^ 2
    (∑ i, p i * H ((2 : ℝ) • Sm + outer2 (y i))) ≤ 2 / 3 - 7 / 12 * dsq ∧
    (1 / 2) * (1 + ∑ i, p i * H ((2 : ℝ) • Sm + outer2 (y i))) ≤ 5 / 6 - 7 / 24 * dsq ∧
    (1 / 2) * (1 + ∑ i, p i * H ((2 : ℝ) • Sm + outer2 (y i))) ≤ 7 / 8 - 3 / 8 * dsq := by
  intro Sm dsq
  -- Extract entries of `Sm`.
  have hS00 : Sm 0 0 = ∑ i, p i * (y i).1 ^ 2 := by
    simp [Sm, outer2, Matrix.sum_apply, sq]
  have hS11 : Sm 1 1 = ∑ i, p i * (y i).2 ^ 2 := by
    simp [Sm, outer2, Matrix.sum_apply, sq]
  have hS01 : Sm 0 1 = ∑ i, p i * ((y i).1 * (y i).2) := by
    simp [Sm, outer2, Matrix.sum_apply]
  have hS10 : Sm 1 0 = Sm 0 1 := by
    simp [Sm, outer2, Matrix.sum_apply, mul_comm]
  have hStr : Sm 0 0 + Sm 1 1 = 1 := by
    rw [hS00, hS11, ← Finset.sum_add_distrib]
    simpa [mul_add] using hE
  -- Vector representation: `z₁ = S₀₀ - S₁₁`, `z₂ = 2 S₀₁`, `w₁ i = y₁² - y₂²`, `w₂ i = 2 y₁ y₂`, `s i = y₁² + y₂²`.
  let z₁ : ℝ := Sm 0 0 - Sm 1 1
  let z₂ : ℝ := 2 * Sm 0 1
  let s : ι → ℝ := fun i => (y i).1 ^ 2 + (y i).2 ^ 2
  let w₁ : ι → ℝ := fun i => (y i).1 ^ 2 - (y i).2 ^ 2
  let w₂ : ι → ℝ := fun i => 2 * (y i).1 * (y i).2
  have hs_nn : ∀ i, 0 ≤ s i := fun i => by positivity
  have h2s_pos : ∀ i, 0 < 2 + s i := fun i => by linarith [hs_nn i]
  have hdsq_eq : dsq = z₁ ^ 2 + z₂ ^ 2 := by simp only [dsq, z₁, z₂]; ring
  have hEs : ∑ i, p i * s i = 1 := hE
  have hEw1 : ∑ i, p i * w₁ i = z₁ := by
    simp only [z₁, hS00, hS11, ← Finset.sum_sub_distrib, w₁]
    refine Finset.sum_congr rfl fun i _ => by ring
  have hEw2 : ∑ i, p i * w₂ i = z₂ := by
    simp only [z₂, hS01, Finset.mul_sum, w₂]
    refine Finset.sum_congr rfl fun i _ => by ring
  have hEdot : ∑ i, p i * (z₁ * w₁ i + z₂ * w₂ i) = dsq := by
    calc ∑ i, p i * (z₁ * w₁ i + z₂ * w₂ i)
        = z₁ * (∑ i, p i * w₁ i) + z₂ * (∑ i, p i * w₂ i) := by
          rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
          refine Finset.sum_congr rfl fun i _ => by ring
      _ = z₁ ^ 2 + z₂ ^ 2 := by rw [hEw1, hEw2]; ring
      _ = dsq := hdsq_eq.symm
  -- Exact formula: `4 * H(2 S + y_i y_iᵀ) = (2 + s_i) - ((2 z₁ + w₁ i)² + (2 z₂ + w₂ i)²) / (2 + s_i)`.
  have h4H : ∀ i, 4 * H ((2 : ℝ) • Sm + outer2 (y i)) =
      (2 + s i) - ((2 * z₁ + w₁ i) ^ 2 + (2 * z₂ + w₂ i) ^ 2) / (2 + s i) := by
    intro i
    have htr : ((2 : ℝ) • Sm + outer2 (y i)).trace = 2 + s i := by
      simp only [Matrix.trace, Matrix.diag_apply, Fin.sum_univ_two, Matrix.add_apply,
        Matrix.smul_apply, smul_eq_mul, outer2, Matrix.of_apply, Matrix.cons_val_zero,
        Matrix.cons_val_one, s]
      linarith [hStr]
    have h4det : 4 * ((2 : ℝ) • Sm + outer2 (y i)).det =
        (2 + s i) ^ 2 - ((2 * z₁ + w₁ i) ^ 2 + (2 * z₂ + w₂ i) ^ 2) := by
      rw [Matrix.det_fin_two]
      simp only [Matrix.add_apply, Matrix.smul_apply, smul_eq_mul, outer2, Matrix.of_apply,
        Matrix.cons_val_zero, Matrix.cons_val_one, hS10, z₁, z₂, w₁, w₂, s]
      have hS11_eq : Sm 1 1 = 1 - Sm 0 0 := by linarith [hStr]
      rw [hS11_eq]
      ring
    simp only [H, htr]
    have hp := h2s_pos i
    field_simp [ne_of_gt hp]
    linarith [h4det]
  -- Define `A, U, V` as in Theorem 2.
  let A : ℝ := ∑ i, p i / (2 + s i)
  let U : ℝ := ∑ i, p i * ((s i - 1) ^ 2 / (2 + s i))
  let V : ℝ := ∑ i, p i * ((s i) ^ 2 / (2 + s i))
  have hU_nn : 0 ≤ U := Finset.sum_nonneg fun i _ => mul_nonneg (hp_nn i) (div_nonneg (sq_nonneg _) (le_of_lt (h2s_pos i)))
  have hV_nn : 0 ≤ V := Finset.sum_nonneg fun i _ => mul_nonneg (hp_nn i) (div_nonneg (sq_nonneg _) (le_of_lt (h2s_pos i)))
  have hU_eq : U = 9 * A - 3 := by
    have hpt : ∀ i, p i * ((s i - 1) ^ 2 / (2 + s i)) =
        p i * s i - 4 * p i + 9 * (p i / (2 + s i)) := by
      intro i; field_simp [ne_of_gt (h2s_pos i)]; ring
    simp only [U, hpt, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, hEs, hp_sum]
    simp only [A]; ring
  have hV_eq : V = 4 * A - 1 := by
    have hpt : ∀ i, p i * ((s i) ^ 2 / (2 + s i)) =
        p i * s i - 2 * p i + 4 * (p i / (2 + s i)) := by
      intro i; field_simp [ne_of_gt (h2s_pos i)]; ring
    simp only [V, hpt, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, hEs, hp_sum]
    simp only [A]; ring
  -- Expand `E[‖2 z + w‖² / (2 + s)]`.
  have hnorm_w : ∀ i, (w₁ i) ^ 2 + (w₂ i) ^ 2 = (s i) ^ 2 := fun i => by
    simp only [w₁, w₂, s]; ring
  have hmain_exp : ∑ i, p i * (((2 * z₁ + w₁ i) ^ 2 + (2 * z₂ + w₂ i) ^ 2) / (2 + s i)) =
      4 * dsq * A + 4 / 3 * dsq - 4 / 3 * ∑ i, p i * ((z₁ * w₁ i + z₂ * w₂ i) * (s i - 1) / (2 + s i)) + V := by
    have hpt : ∀ i, p i * (((2 * z₁ + w₁ i) ^ 2 + (2 * z₂ + w₂ i) ^ 2) / (2 + s i)) =
        4 * dsq * (p i / (2 + s i)) +
        4 / 3 * (p i * (z₁ * w₁ i + z₂ * w₂ i)) -
        4 / 3 * (p i * ((z₁ * w₁ i + z₂ * w₂ i) * (s i - 1) / (2 + s i))) +
        p i * ((s i) ^ 2 / (2 + s i)) := by
      intro i
      have hpos := h2s_pos i
      rw [hdsq_eq, ← hnorm_w i]
      field_simp [ne_of_gt hpos]
      ring
    simp only [hpt, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, hEdot]
    rfl
  -- Cauchy-Schwarz bound on the cross term:
  -- `|∑ p_i (z · w_i)(s_i - 1) / (2 + s_i)| ≤ √(dsq * U * V)`.
  have hcs_cross : (∑ i, p i * ((z₁ * w₁ i + z₂ * w₂ i) * (s i - 1) / (2 + s i))) ^ 2 ≤
      dsq * U * V := by
    rw [hdsq_eq]
    exact cs_cross_bound p s w₁ w₂ z₁ z₂ hp_nn hs_nn hnorm_w
  -- Complete the square: `U / 9 - (1 / 3) * cross + dsq * (A + 1/3) ≥ (7 / 12) * dsq`.
  set X := ∑ i, p i * ((z₁ * w₁ i + z₂ * w₂ i) * (s i - 1) / (2 + s i))
  have hcomplete : (7 / 12 : ℝ) * dsq ≤ U / 9 + dsq * (A + 1 / 3) - (1 / 3) * X :=
    amgm_complete_square hU_nn hV_nn (by rw [hdsq_eq]; positivity) hV_eq hcs_cross
  -- Deduce `∑ p_i H(2 Σ + y_i y_iᵀ) ≤ 2/3 - (7/12) dsq`.
  have hbound1 : ∑ i, p i * H ((2 : ℝ) • Sm + outer2 (y i)) ≤ 2 / 3 - 7 / 12 * dsq := by
    have h4sum : 4 * ∑ i, p i * H ((2 : ℝ) • Sm + outer2 (y i)) =
        3 - ∑ i, p i * (((2 * z₁ + w₁ i) ^ 2 + (2 * z₂ + w₂ i) ^ 2) / (2 + s i)) := by
      rw [Finset.mul_sum]
      have hpt : ∀ i, 4 * (p i * H ((2 : ℝ) • Sm + outer2 (y i))) =
          2 * p i + p i * s i - p i * (((2 * z₁ + w₁ i) ^ 2 + (2 * z₂ + w₂ i) ^ 2) / (2 + s i)) := by
        intro i
        have := h4H i
        calc 4 * (p i * H ((2 : ℝ) • Sm + outer2 (y i)))
            = p i * (4 * H ((2 : ℝ) • Sm + outer2 (y i))) := by ring
          _ = p i * ((2 + s i) - ((2 * z₁ + w₁ i) ^ 2 + (2 * z₂ + w₂ i) ^ 2) / (2 + s i)) := by rw [this]
          _ = 2 * p i + p i * s i - p i * (((2 * z₁ + w₁ i) ^ 2 + (2 * z₂ + w₂ i) ^ 2) / (2 + s i)) := by ring
      simp only [hpt, Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, hp_sum, hEs]
      ring
    have hVU : V - 1 / 3 = 4 / 9 * U := by linarith only [hU_eq, hV_eq]
    have hrel : (1 / 4 : ℝ) * (∑ i, p i * (((2 * z₁ + w₁ i) ^ 2 + (2 * z₂ + w₂ i) ^ 2) / (2 + s i)) - 1 / 3) =
        U / 9 + dsq * (A + 1 / 3) - (1 / 3) * X := by
      rw [hmain_exp]
      linarith only [hVU]
    linarith only [h4sum, hrel, hcomplete]
  -- Also prove the concavity bound `≤ 7/8 - (3/8) dsq` via `expectation_H_concavity_bound`.
  have hbound3 : (1 / 2) * (1 + ∑ i, p i * H ((2 : ℝ) • Sm + outer2 (y i))) ≤ 7 / 8 - 3 / 8 * dsq := by
    have h := expectation_H_concavity_bound p y hp_nn hp_sum hE
    dsimp only at h
    linarith only [h]
  exact ⟨hbound1, by linarith only [hbound1], hbound3⟩

/-! ### 3. Two Real Output Lines 3-Point Bound (Theorem 4) -/

/-- **Theorem 4 (Explicit two-output-line 3-point bound)**:
For `a, b ≥ 0`, `t ∈ (0, 1)`, and `(1 - t) a + t b ≤ 1`,
`min (b / 2) (t * a) ≤ t / (1 - t + 2 t²) ≤ 1 / (2√2 - 1) < 5 / 8`. -/
theorem two_output_lines_bound {a b t : ℝ}
    (ht0 : 0 < t) (ht1 : t < 1) (hE : (1 - t) * a + t * b ≤ 1) :
    min (b / 2) (t * a) ≤ t / (1 - t + 2 * t ^ 2) ∧
    t / (1 - t + 2 * t ^ 2) ≤ 1 / (2 * Real.sqrt 2 - 1) ∧
    1 / (2 * Real.sqrt 2 - 1) < 5 / 8 := by
  have hden1 : 0 < 1 - t + 2 * t ^ 2 := by nlinarith
  have hs2_sq : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  have hs2_gt : (13 : ℝ) / 10 < Real.sqrt 2 := by
    rw [Real.lt_sqrt (by norm_num)]; norm_num
  have hden2 : 0 < 2 * Real.sqrt 2 - 1 := by linarith
  refine ⟨?_, ?_, ?_⟩
  · rw [le_div_iff₀ hden1]
    set m := min (b / 2) (t * a)
    have hm1 : m ≤ b / 2 := min_le_left _ _
    have hm2 : m ≤ t * a := min_le_right _ _
    nlinarith
  · rw [div_le_div_iff₀ hden1 hden2]
    have hsq : 0 ≤ (Real.sqrt 2 * t - 1) ^ 2 := sq_nonneg _
    nlinarith [hs2_sq]
  · rw [div_lt_div_iff₀ hden2 (by norm_num : (0 : ℝ) < 8)]
    linarith

/-! ### 4. Isotropic 6-Atom Algebraic Identity & Rankin Lemma (Theorem 5) -/

/-- Auxiliary function `f₆(t) = t / (1 - t)²` from Theorem 5. -/
noncomputable def f6 (t : ℝ) : ℝ := t / (1 - t) ^ 2

/-- **Theorem 5 (Exact isotropic 6-atom algebraic identity)**:
For every `a ∈ ℝ` with `a² ≠ 4`,
`(1/6)(f₆((1+a)/3) + f₆((1-a)/3) - a(f₆((1+a)/3) - f₆((1-a)/3))) - 1/4 = -a²(5a² + 4) / (4(4 - a²)²) ≤ 0`. -/
theorem isotropic_six_atom_identity {a : ℝ} (ha : a ^ 2 ≠ 4) :
    (1 / 6) * (f6 ((1 + a) / 3) + f6 ((1 - a) / 3) - a * (f6 ((1 + a) / 3) - f6 ((1 - a) / 3))) - 1 / 4 =
      -(a ^ 2 * (5 * a ^ 2 + 4)) / (4 * (4 - a ^ 2) ^ 2) ∧
    -(a ^ 2 * (5 * a ^ 2 + 4)) / (4 * (4 - a ^ 2) ^ 2) ≤ 0 := by
  have hm : 2 - a ≠ 0 := by intro h; apply ha; nlinarith
  have hp : 2 + a ≠ 0 := by intro h; apply ha; nlinarith
  have h3 : 4 - a ^ 2 ≠ 0 := sub_ne_zero.mpr ha.symm
  have hf1 : f6 ((1 + a) / 3) = 3 * (1 + a) / (2 - a) ^ 2 := by
    simp only [f6]
    have : 1 - (1 + a) / 3 = (2 - a) / 3 := by ring
    rw [this]; field_simp [hm]
  have hf2 : f6 ((1 - a) / 3) = 3 * (1 - a) / (2 + a) ^ 2 := by
    simp only [f6]
    have : 1 - (1 - a) / 3 = (2 + a) / 3 := by ring
    rw [this]; field_simp [hp]
  constructor
  · rw [hf1, hf2]
    field_simp [hm, hp, h3]
    ring
  · have hnum : 0 ≤ a ^ 2 * (5 * a ^ 2 + 4) := by positivity
    have hden : 0 ≤ 4 * (4 - a ^ 2) ^ 2 := by positivity
    rw [neg_div]
    exact neg_nonpos.mpr (div_nonneg hnum hden)

/-- **Theorem 5 (Full inequality under `a + b + c = 0` and `0 ≤ a * c`)**:
For `|a| ≤ 1`, `a + b + c = 0`, and `0 ≤ a * c`,
`(1/6)(f₆((1+a)/3) + f₆((1-a)/3) + b(f₆((1+a)/3) - f₆((1-a)/3))) ≤ 1 / 4`. -/
theorem isotropic_six_atom_le_quarter {a b c : ℝ} (ha : |a| ≤ 1)
    (hsum : a + b + c = 0) (hac : 0 ≤ a * c) :
    (1 / 6) * (f6 ((1 + a) / 3) + f6 ((1 - a) / 3) + b * (f6 ((1 + a) / 3) - f6 ((1 - a) / 3))) ≤
      1 / 4 := by
  have hal := (abs_le.mp ha).1; have hah := (abs_le.mp ha).2
  have ha4 : a ^ 2 ≠ 4 := by nlinarith
  have ⟨hid, hneg⟩ := isotropic_six_atom_identity ha4
  have hdiff : f6 ((1 + a) / 3) - f6 ((1 - a) / 3) =
      6 * a * (8 + a ^ 2) / (4 - a ^ 2) ^ 2 := by
    have hm : 2 - a ≠ 0 := by linarith
    have hp : 2 + a ≠ 0 := by linarith
    have h3 : 4 - a ^ 2 ≠ 0 := by nlinarith
    have hf1 : f6 ((1 + a) / 3) = 3 * (1 + a) / (2 - a) ^ 2 := by
      simp only [f6]
      have : 1 - (1 + a) / 3 = (2 - a) / 3 := by ring
      rw [this]; field_simp [hm]
    have hf2 : f6 ((1 - a) / 3) = 3 * (1 - a) / (2 + a) ^ 2 := by
      simp only [f6]
      have : 1 - (1 - a) / 3 = (2 + a) / 3 := by ring
      rw [this]; field_simp [hp]
    rw [hf1, hf2]
    field_simp [hm, hp, h3]
    ring
  have hb_eq : b = -a - c := by linarith
  have hstep : (1 / 6) * (f6 ((1 + a) / 3) + f6 ((1 - a) / 3) + b * (f6 ((1 + a) / 3) - f6 ((1 - a) / 3))) -
      (1 / 6) * (f6 ((1 + a) / 3) + f6 ((1 - a) / 3) - a * (f6 ((1 + a) / 3) - f6 ((1 - a) / 3))) =
      -(a * c) * (8 + a ^ 2) / (4 - a ^ 2) ^ 2 := by
    rw [hb_eq, hdiff]
    ring
  have hnonpos : -(a * c) * (8 + a ^ 2) / (4 - a ^ 2) ^ 2 ≤ 0 := by
    have h1 : 0 ≤ a * c * (8 + a ^ 2) := mul_nonneg hac (by positivity)
    have h2 : 0 ≤ (4 - a ^ 2) ^ 2 := sq_nonneg _
    have h3 : -(a * c) * (8 + a ^ 2) / (4 - a ^ 2) ^ 2 = -(a * c * (8 + a ^ 2) / (4 - a ^ 2) ^ 2) := by ring
    rw [h3]
    exact neg_nonpos.mpr (div_nonneg h1 h2)
  linarith

/-- **Rankin's (1955) 6-vectors-in-`ℝ⁴` lemma (Theorem 5)**:
Any 6 vectors in `ℝ⁴` have a distinct pair `i ≠ j` with nonnegative inner product `∑ k, v i k * v j k ≥ 0`. -/
theorem six_vectors_nonneg_inner_product (v : Fin 6 → Fin 4 → ℝ) :
    ∃ i j : Fin 6, i ≠ j ∧ 0 ≤ ∑ k : Fin 4, v i k * v j k := by
  -- Map `c : Fin 6 → ℝ` to `(∑ c_i, ∑ c_i v_i) ∈ ℝ × (Fin 4 → ℝ)` (dim 5 < 6), so kernel is nontrivial!
  let L : (Fin 6 → ℝ) →ₗ[ℝ] (ℝ × (Fin 4 → ℝ)) :=
    { toFun := fun c => (∑ i, c i, fun k => ∑ i, c i * v i k)
      map_add' := fun c₁ c₂ => by
        ext
        · simp [Finset.sum_add_distrib]
        · simp [add_mul, Finset.sum_add_distrib]
      map_smul' := fun r c => by
        ext
        · simp [Finset.mul_sum]
        · simp [mul_assoc, Finset.mul_sum] }
  have hker_pos : 0 < Module.finrank ℝ (LinearMap.ker L) := by
    have hrn := LinearMap.finrank_range_add_finrank_ker L
    have hdom : Module.finrank ℝ (Fin 6 → ℝ) = 6 := by simp
    have hcod : Module.finrank ℝ (ℝ × (Fin 4 → ℝ)) = 5 := by simp [Module.finrank_prod]
    have hrange_le : Module.finrank ℝ (LinearMap.range L) ≤ 5 := by
      rw [← hcod]
      exact Submodule.finrank_le _
    omega
  have hnt : Nontrivial (LinearMap.ker L) := Module.nontrivial_of_finrank_pos hker_pos
  obtain ⟨⟨c, hc⟩, hc_ne⟩ := exists_ne (0 : LinearMap.ker L)
  have hc_nz : c ≠ 0 := fun h => hc_ne (Subtype.ext h)
  have hcsum : ∑ i, c i = 0 := congrArg Prod.fst hc
  have hcv : ∀ k : Fin 4, ∑ i, c i * v i k = 0 := fun k => congrFun (congrArg Prod.snd hc) k
  -- Split `Fin 6` into `P = {i | 0 < c i}` and `N = {i | c i < 0}`; both are nonempty!
  let P : Finset (Fin 6) := Finset.univ.filter (fun i => 0 < c i)
  let N : Finset (Fin 6) := Finset.univ.filter (fun i => c i < 0)
  have hPN_disj : Disjoint P N := Finset.disjoint_filter.mpr (fun _ _ h1 h2 => by linarith)
  have hP_ne : P.Nonempty := by
    by_contra hemp
    rw [Finset.not_nonempty_iff_eq_empty] at hemp
    have hle0 : ∀ i, c i ≤ 0 := fun i => by
      by_contra hgt; push_neg at hgt
      exact (Finset.notMem_empty i) (hemp ▸ Finset.mem_filter.mpr ⟨Finset.mem_univ i, hgt⟩)
    have hall0 : ∀ i, c i = 0 := fun i => by
      have := Finset.sum_eq_zero_iff_of_nonpos (fun j _ => hle0 j) |>.mp hcsum i (Finset.mem_univ i)
      exact this
    exact hc_nz (funext hall0)
  have hN_ne : N.Nonempty := by
    by_contra hemp
    rw [Finset.not_nonempty_iff_eq_empty] at hemp
    have hge0 : ∀ i, 0 ≤ c i := fun i => by
      by_contra hlt; push_neg at hlt
      exact (Finset.notMem_empty i) (hemp ▸ Finset.mem_filter.mpr ⟨Finset.mem_univ i, hlt⟩)
    have hall0 : ∀ i, c i = 0 := fun i =>
      Finset.sum_eq_zero_iff_of_nonneg (fun j _ => hge0 j) |>.mp hcsum i (Finset.mem_univ i)
    exact hc_nz (funext hall0)
  -- Define `w_P k := ∑ i ∈ P, c i * v i k` and `w_N k := ∑ j ∈ N, (-c j) * v j k`; note `w_P = w_N`!
  have hw_eq : ∀ k : Fin 4, ∑ i ∈ P, c i * v i k = ∑ j ∈ N, (-c j) * v j k := by
    intro k
    have h0 := hcv k
    have hpart : ∑ i : Fin 6, c i * v i k = (∑ i ∈ P, c i * v i k) + (∑ j ∈ N, c j * v j k) := by
      rw [← Finset.sum_union hPN_disj]
      symm
      refine Finset.sum_subset (Finset.subset_univ _) fun i _ hi => ?_
      simp only [Finset.mem_union, P, N, Finset.mem_filter, Finset.mem_univ, true_and,
        not_or, not_lt] at hi
      have : c i = 0 := le_antisymm hi.1 hi.2
      simp [this]
    have hneg : ∑ j ∈ N, (-c j) * v j k = -(∑ j ∈ N, c j * v j k) := by
      rw [← Finset.sum_neg_distrib]
      refine Finset.sum_congr rfl fun j _ => by ring
    linarith
  -- Therefore `0 ≤ ∑ k, w_P k * w_N k = ∑ i ∈ P, ∑ j ∈ N, (c i * (-c j)) * ⟨v i, v j⟩`!
  have hdouble_nn : 0 ≤ ∑ i ∈ P, ∑ j ∈ N, (c i * (-c j)) * (∑ k : Fin 4, v i k * v j k) := by
    have hEq : ∑ i ∈ P, ∑ j ∈ N, (c i * (-c j)) * (∑ k : Fin 4, v i k * v j k) =
        ∑ k : Fin 4, (∑ i ∈ P, c i * v i k) ^ 2 := by
      calc ∑ i ∈ P, ∑ j ∈ N, (c i * (-c j)) * (∑ k : Fin 4, v i k * v j k)
          = ∑ i ∈ P, ∑ j ∈ N, ∑ k : Fin 4, (c i * v i k) * ((-c j) * v j k) := by
            refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
            rw [Finset.mul_sum]; refine Finset.sum_congr rfl fun k _ => by ring
        _ = ∑ k : Fin 4, (∑ i ∈ P, c i * v i k) * (∑ j ∈ N, (-c j) * v j k) := by
            rw [Finset.sum_comm]
            trans ∑ j ∈ N, ∑ k : Fin 4, (∑ i ∈ P, c i * v i k) * ((-c j) * v j k)
            · refine Finset.sum_congr rfl fun j _ => ?_
              rw [Finset.sum_comm]
              refine Finset.sum_congr rfl fun k _ => (Finset.sum_mul _ _ _).symm
            · rw [Finset.sum_comm]
              refine Finset.sum_congr rfl fun k _ => (Finset.mul_sum _ _ _).symm
        _ = ∑ k : Fin 4, (∑ i ∈ P, c i * v i k) ^ 2 := by
            refine Finset.sum_congr rfl fun k _ => by rw [← hw_eq k, sq]
    rw [hEq]
    exact Finset.sum_nonneg fun k _ => sq_nonneg _
  by_contra hall_neg
  push_neg at hall_neg
  have hstrict : ∑ i ∈ P, ∑ j ∈ N, (c i * (-c j)) * (∑ k : Fin 4, v i k * v j k) < 0 := by
    have hlt_i : ∑ i ∈ P, ∑ j ∈ N, (c i * (-c j)) * (∑ k : Fin 4, v i k * v j k) <
        ∑ _i ∈ P, (0 : ℝ) := by
      refine Finset.sum_lt_sum_of_nonempty hP_ne fun i hi => ?_
      have hci : 0 < c i := (Finset.mem_filter.mp hi).2
      have hlt_j : ∑ j ∈ N, (c i * (-c j)) * (∑ k : Fin 4, v i k * v j k) < ∑ _j ∈ N, (0 : ℝ) := by
        refine Finset.sum_lt_sum_of_nonempty hN_ne fun j hj => ?_
        have hcj : c j < 0 := (Finset.mem_filter.mp hj).2
        have hne : i ≠ j := fun heq => by rw [heq] at hci; linarith
        have hdot := hall_neg i j hne
        have hprod : 0 < c i * (-c j) := mul_pos hci (by linarith)
        nlinarith
      simpa using hlt_j
    simpa using hlt_i
  linarith

/-! ### 5. Exact 7-Atom Original-Weight Obstruction at Both Budgets (Proposition 10) -/

/-- Exact rational parameters of the 4 atoms on the second feature line in Proposition 10:
each entry is `(x_coord, y_in_units_of_s, integer_weight)` where `s² = 56 / 81`. -/
def obsAtoms : Fin 4 → ℚ × ℚ × ℚ
  | 0 => (-1, 1 / 100, 5)
  | 1 => (-1, -1 / 100, 5)
  | 2 => (5 / 9, 1, 9)
  | 3 => (5 / 9, -1, 9)

/-- Vertical scale `s² = 56 / 81` in Proposition 10. -/
def obsVertScale : ℚ := 56 / 81

/-- First-class conditional energy `e = 20 / 21`. -/
def obsE1 : ℚ := 20 / 21

/-- Second-class conditional energy `e' = 1 + 1 / 40500`. -/
def obsE2 : ℚ := 1 + 1 / 40500

/-- Original-weight squared conditional mean of a nonempty subset `S ⊆ Fin 4` on line 2. -/
def obsSqMean (S : Finset (Fin 4)) : ℚ :=
  let W := ∑ i ∈ S, (obsAtoms i).2.2
  let mx := (∑ i ∈ S, (obsAtoms i).2.2 * (obsAtoms i).1) / W
  let my := (∑ i ∈ S, (obsAtoms i).2.2 * (obsAtoms i).2.1) / W
  mx ^ 2 + obsVertScale * my ^ 2

/-- First-class equilateral triangle cost for `i ∈ {1, 2, 3}` atoms. -/
def obsFirstCost : Fin 3 → ℚ
  | 0 => obsE1        -- 1 atom
  | 1 => obsE1 / 4    -- 2 atoms
  | 2 => 0            -- 3 atoms

/-- **Proposition 10 (Moment conditions on line 2)**: The 4 atoms on the second feature line
have zero conditional mean and exact conditional energy `e' = 1 + 1 / 40500`. -/
theorem seven_atom_moment_conditions :
    (∑ i : Fin 4, (obsAtoms i).2.2) = 28 ∧
    (∑ i : Fin 4, (obsAtoms i).2.2 * (obsAtoms i).1) = 0 ∧
    (∑ i : Fin 4, (obsAtoms i).2.2 * (obsAtoms i).2.1) = 0 ∧
    (∑ i : Fin 4, ((obsAtoms i).2.2 / 28) * ((obsAtoms i).1 ^ 2 + obsVertScale * (obsAtoms i).2.1 ^ 2)) =
      obsE2 := by
  simp only [Fin.sum_univ_four, obsAtoms, obsVertScale, obsE2]
  norm_num

private lemma fin4_nonempty_subsets (S : Finset (Fin 4)) (hS : S.Nonempty) :
    S ∈ ([{0}, {1}, {2}, {3},
          {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3},
          {0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3},
          {0, 1, 2, 3}] : List (Finset (Fin 4))) := by
  revert S hS
  decide

/-- **Proposition 10 (Original-weight obstruction at budget `k = 3`)**:
Every spanning original-weight selection `(i₁ + 1) + |S| ≤ 3` (`S ⊆ Fin 4` nonempty) has cost ratio
at least `700205 / 1107014 > 5 / 8`, attained at `i₁ = 0` (1 atom on line 1) and `S = {0, 3}`. -/
theorem seven_atom_orig_weight_k3 :
    (700205 : ℚ) / 1107014 > 5 / 8 ∧
    (obsFirstCost 0 + obsSqMean {0, 3}) / (obsE1 + obsE2) = 700205 / 1107014 ∧
    ∀ (i₁ : Fin 3) (S : Finset (Fin 4)), S.Nonempty → (i₁.val + 1) + S.card ≤ 3 →
      (700205 : ℚ) / 1107014 ≤ (obsFirstCost i₁ + obsSqMean S) / (obsE1 + obsE2) := by
  refine ⟨by norm_num, ?_, ?_⟩
  · simp only [obsFirstCost, obsSqMean, obsAtoms, obsVertScale, obsE1, obsE2,
      Finset.sum_insert, Finset.sum_singleton, Finset.mem_singleton, Fin.reduceEq,
      not_false_eq_true]
    norm_num
  · intro i₁ S hSne hcard
    have hmem := fin4_nonempty_subsets S hSne
    fin_cases i₁ <;>
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hmem <;>
      rcases hmem with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      first
      | (exfalso; revert hcard; decide)
      | (simp [obsFirstCost, obsSqMean, obsAtoms, obsVertScale, obsE1, obsE2]; norm_num)

/-- **Proposition 10 (Original-weight obstruction at budget `k = 4`)**:
Every spanning original-weight selection `(i₁ + 1) + |S| ≤ 4` (`S ⊆ Fin 4` nonempty) has cost ratio
at least `295205 / 1107014 > 1 / 4`, attained at `i₁ = 1` (2 atoms on line 1) and `S = {0, 3}`. -/
theorem seven_atom_orig_weight_k4 :
    (295205 : ℚ) / 1107014 > 1 / 4 ∧
    (obsFirstCost 1 + obsSqMean {0, 3}) / (obsE1 + obsE2) = 295205 / 1107014 ∧
    ∀ (i₁ : Fin 3) (S : Finset (Fin 4)), S.Nonempty → (i₁.val + 1) + S.card ≤ 4 →
      (295205 : ℚ) / 1107014 ≤ (obsFirstCost i₁ + obsSqMean S) / (obsE1 + obsE2) := by
  refine ⟨by norm_num, ?_, ?_⟩
  · simp only [obsFirstCost, obsSqMean, obsAtoms, obsVertScale, obsE1, obsE2,
      Finset.sum_insert, Finset.sum_singleton, Finset.mem_singleton, Fin.reduceEq,
      not_false_eq_true]
    norm_num
  · intro i₁ S hSne hcard
    have hmem := fin4_nonempty_subsets S hSne
    fin_cases i₁ <;>
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hmem <;>
      rcases hmem with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      first
      | (exfalso; revert hcard; decide)
      | (simp [obsFirstCost, obsSqMean, obsAtoms, obsVertScale, obsE1, obsE2]; norm_num)

/-- **Proposition 10 (Equal-weight cross-pair reweighting beats both targets)**:
Assigning equal weights `(1/2, 1/2)` to the opposite-vertical-sign cross pair `{0, 3}` lowers its
squared mean to `d' = 88607 / 405000`, giving ratios `3320249 / 5535070 < 5 / 8` (at `k = 3`) and
`1295249 / 5535070 < 1 / 4` (at `k = 4`). -/
theorem seven_atom_reweighted_beats_both :
    let mx := ((obsAtoms 0).1 + (obsAtoms 3).1) / 2
    let my := ((obsAtoms 0).2.1 + (obsAtoms 3).2.1) / 2
    let d' := mx ^ 2 + obsVertScale * my ^ 2
    d' = 88607 / 405000 ∧
    (obsE1 + d') / (obsE1 + obsE2) = 3320249 / 5535070 ∧
    (3320249 : ℚ) / 5535070 < 5 / 8 ∧
    (obsE1 / 4 + d') / (obsE1 + obsE2) = 1295249 / 5535070 ∧
    (1295249 : ℚ) / 5535070 < 1 / 4 := by
  dsimp only
  norm_num [obsAtoms, obsVertScale, obsE1, obsE2]

/-! ### 6. Exact Rational Dual Certificates for a 7-Atom Continuum (Proposition 8) -/

/-- Exact `7 × 2` input matrix `X` of Proposition 8. -/
def certX : Matrix (Fin 7) (Fin 2) ℚ :=
  !![1, 0; 0, 1; 1, 1; 1, -1; 2, 1; 1, 2; 2, -1]

/-- Exact `7 × 2` raw matrix `A` of Proposition 8. -/
def certA : Matrix (Fin 7) (Fin 2) ℚ :=
  !![1, -2; -1, 3; 2, 1; 0, 3; -1, -1; 2, -2; -3, 1]

/-- Exact inverse of a `2 × 2` rational matrix via adjugate / determinant. -/
def inv2Q (M : Matrix (Fin 2) (Fin 2) ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  (M.det)⁻¹ • !![M 1 1, -M 0 1; -M 1 0, M 0 0]

/-- Projected residual matrix `Y = (I - X (Xᵀ X)⁻¹ Xᵀ) A` of Proposition 8. -/
def certY : Matrix (Fin 7) (Fin 2) ℚ :=
  certA - certX * inv2Q (certXᵀ * certX) * certXᵀ * certA

/-- Equal-row-multiplier least-squares fit `B_S = (X_Sᵀ X_S)⁻¹ X_Sᵀ Y_S` on support `S ⊆ Fin 7`. -/
def certFit (S : Finset (Fin 7)) : Matrix (Fin 2) (Fin 2) ℚ :=
  let G : Matrix (Fin 2) (Fin 2) ℚ := Matrix.of fun r c => ∑ i ∈ S, certX i r * certX i c
  let R : Matrix (Fin 2) (Fin 2) ℚ := Matrix.of fun r c => ∑ i ∈ S, certX i r * certY i c
  inv2Q G * R

/-- Dual ratio for atom `i : Fin 7` given mixed second-moment matrix `H` and dual multiplier `D`:
`(x_iᵀ H x_i - y_iᵀ D x_i) / ‖y_i‖²`. -/
def certDualRatio (Hmat Dmat : Matrix (Fin 2) (Fin 2) ℚ) (i : Fin 7) : ℚ :=
  let xHx := ∑ r : Fin 2, ∑ c : Fin 2, certX i r * Hmat r c * certX i c
  let yDx := ∑ r : Fin 2, ∑ c : Fin 2, certY i r * Dmat r c * certX i c
  let ySq := certY i 0 ^ 2 + certY i 1 ^ 2
  (xHx - yDx) / ySq

/-- Mixed matrix `H₃` for the `k = 3` dual certificate: supports `{1, 5, 6}` (93%) and `{3, 5, 6}` (7%). -/
def certH3 : Matrix (Fin 2) (Fin 2) ℚ :=
  let B₁ := certFit {1, 5, 6}
  let B₂ := certFit {3, 5, 6}
  (93 / 100 : ℚ) • (B₁ * B₁ᵀ) + (7 / 100 : ℚ) • (B₂ * B₂ᵀ)

/-- Dual multiplier `D₃ = (1/100) !![-22, 24; 1, -20]` for `k = 3`. -/
def certD3 : Matrix (Fin 2) (Fin 2) ℚ :=
  (1 / 100 : ℚ) • !![-22, 24; 1, -20]

/-- Mixed matrix `H₄` for the `k = 4` dual certificate: supports `{1, 3, 4, 5}` (33%) and `{1, 3, 5, 6}` (67%). -/
def certH4 : Matrix (Fin 2) (Fin 2) ℚ :=
  let B₁ := certFit {1, 3, 4, 5}
  let B₂ := certFit {1, 3, 5, 6}
  (33 / 100 : ℚ) • (B₁ * B₁ᵀ) + (67 / 100 : ℚ) • (B₂ * B₂ᵀ)

/-- Dual multiplier `D₄ = (1/100) !![-14, 15; 2, -12]` for `k = 4`. -/
def certD4 : Matrix (Fin 2) (Fin 2) ℚ :=
  (1 / 100 : ℚ) • !![-14, 15; 2, -12]

private lemma certXtX_eq : certXᵀ * certX = !![12, 2; 2, 9] := by
  ext r c
  fin_cases r <;> fin_cases c <;>
    simp [certX, Matrix.mul_apply, Fin.sum_univ_seven] <;> norm_num

private lemma certXtA_eq : certXᵀ * certA = !![-3, 0; 7, -5] := by
  ext r c
  fin_cases r <;> fin_cases c <;>
    simp [certX, certA, Matrix.mul_apply, Fin.sum_univ_seven] <;> norm_num

private lemma certY_eq :
    certY = !![145 / 104, -109 / 52;
               -97 / 52, 93 / 26;
               159 / 104, 77 / 52;
               131 / 104, 121 / 52;
               -14 / 13, -8 / 13;
               69 / 104, -49 / 52;
               -35 / 26, 3 / 13] := by
  have hmul : certX * inv2Q (certXᵀ * certX) * certXᵀ * certA =
      certX * (inv2Q !![12, 2; 2, 9] * !![-3, 0; 7, -5]) := by
    rw [Matrix.mul_assoc, Matrix.mul_assoc, certXtX_eq, certXtA_eq]
  have hcoef : inv2Q !![12, 2; 2, 9] * !![-3, 0; 7, -5] =
      !![-41 / 104, 5 / 52; 45 / 52, -15 / 26] := by
    ext r c
    fin_cases r <;> fin_cases c <;>
      simp [inv2Q, Matrix.mul_apply, Matrix.det_fin_two, Fin.sum_univ_two] <;> norm_num
  rw [certY, hmul, hcoef]
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [certX, certA] <;> norm_num

/-- **Proposition 8 (Orthogonality & nondegeneracy)**: `Xᵀ Y = 0` (so uniform positive masses are
compatible), and every residual row `y_i` has strictly positive squared norm. -/
theorem dual_cert_orthogonality :
    certXᵀ * certY = 0 ∧
    ∀ i : Fin 7, 0 < certY i 0 ^ 2 + certY i 1 ^ 2 := by
  rw [certY_eq]
  constructor
  · ext r c
    fin_cases r <;> fin_cases c <;>
      simp [certX, Matrix.mul_apply, Fin.sum_univ_seven] <;> norm_num
  · intro i
    fin_cases i <;> simp <;> norm_num

/-- Helper: weak duality connecting pointwise dual ratios to the excess ratio over ANY compatible
nonnegative mass vector `p : Fin 7 → ℚ` with `∑ p_i y_i x_iᵀ = 0`. -/
private lemma weak_duality_of_pointwise {Hmat Dmat : Matrix (Fin 2) (Fin 2) ℚ} {bound : ℚ}
    (hpt : ∀ i : Fin 7, certDualRatio Hmat Dmat i ≤ bound)
    {p : Fin 7 → ℚ} (hp_nn : ∀ i, 0 ≤ p i)
    (hcompat : ∀ r c : Fin 2, ∑ i : Fin 7, p i * certY i r * certX i c = 0)
    (hEpos : 0 < ∑ i : Fin 7, p i * (certY i 0 ^ 2 + certY i 1 ^ 2)) :
    (∑ i : Fin 7, p i * (∑ r : Fin 2, ∑ c : Fin 2, certX i r * Hmat r c * certX i c)) /
      (∑ i : Fin 7, p i * (certY i 0 ^ 2 + certY i 1 ^ 2)) ≤ bound := by
  have hypos : ∀ i : Fin 7, 0 < certY i 0 ^ 2 + certY i 1 ^ 2 := dual_cert_orthogonality.2
  -- Show `∑ p_i (y_iᵀ D x_i) = 0` from compatibility `∑ p_i y_i x_iᵀ = 0`.
  have hDzero : ∑ i : Fin 7, p i * (∑ r : Fin 2, ∑ c : Fin 2, certY i r * Dmat r c * certX i c) = 0 := by
    calc ∑ i : Fin 7, p i * (∑ r : Fin 2, ∑ c : Fin 2, certY i r * Dmat r c * certX i c)
        = ∑ i : Fin 7, ∑ r : Fin 2, ∑ c : Fin 2, Dmat r c * (p i * certY i r * certX i c) := by
          refine Finset.sum_congr rfl fun i _ => ?_
          rw [Finset.mul_sum]; refine Finset.sum_congr rfl fun r _ => ?_
          rw [Finset.mul_sum]; refine Finset.sum_congr rfl fun c _ => by ring
      _ = ∑ r : Fin 2, ∑ c : Fin 2, Dmat r c * (∑ i : Fin 7, p i * certY i r * certX i c) := by
          rw [Finset.sum_comm]
          refine Finset.sum_congr rfl fun r _ => ?_
          rw [Finset.sum_comm]
          refine Finset.sum_congr rfl fun c _ => (Finset.mul_sum _ _ _).symm
      _ = 0 := by
          refine Finset.sum_eq_zero fun r _ => Finset.sum_eq_zero fun c _ => ?_
          rw [hcompat r c, mul_zero]
  rw [div_le_iff₀ hEpos]
  have hsum_le : ∑ i : Fin 7, p i *
      ((∑ r : Fin 2, ∑ c : Fin 2, certX i r * Hmat r c * certX i c) -
       (∑ r : Fin 2, ∑ c : Fin 2, certY i r * Dmat r c * certX i c)) ≤
      bound * ∑ i : Fin 7, p i * (certY i 0 ^ 2 + certY i 1 ^ 2) := by
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun i _ => ?_
    have hi := (div_le_iff₀ (hypos i)).mp (hpt i)
    have hmul := mul_le_mul_of_nonneg_left hi (hp_nn i)
    linarith
  have hsplit : ∑ i : Fin 7, p i *
      ((∑ r : Fin 2, ∑ c : Fin 2, certX i r * Hmat r c * certX i c) -
       (∑ r : Fin 2, ∑ c : Fin 2, certY i r * Dmat r c * certX i c)) =
      (∑ i : Fin 7, p i * (∑ r : Fin 2, ∑ c : Fin 2, certX i r * Hmat r c * certX i c)) -
      (∑ i : Fin 7, p i * (∑ r : Fin 2, ∑ c : Fin 2, certY i r * Dmat r c * certX i c)) := by
    rw [← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun i _ => by ring
  linarith

/-- **Proposition 8 (`k = 3` exact rational dual certificate)**:
Mixing equal-weight fits on supports `{1, 5, 6}` (93%) and `{3, 5, 6}` (7%) with
`100 D₃ = !![-22, 24; 1, -20]` certifies exact maximum dual ratio `560659 / 7066875 < 1 / 10`
across EVERY compatible nonnegative test mass vector `p`. -/
theorem dual_cert_k3 :
    (560659 : ℚ) / 7066875 < 1 / 10 ∧
    (∃ i : Fin 7, certDualRatio certH3 certD3 i = 560659 / 7066875) ∧
    (∀ i : Fin 7, certDualRatio certH3 certD3 i ≤ 560659 / 7066875) ∧
    (∀ (p : Fin 7 → ℚ), (∀ i, 0 ≤ p i) →
      (∀ r c : Fin 2, ∑ i : Fin 7, p i * certY i r * certX i c = 0) →
      0 < ∑ i : Fin 7, p i * (certY i 0 ^ 2 + certY i 1 ^ 2) →
      (∑ i : Fin 7, p i * (∑ r : Fin 2, ∑ c : Fin 2, certX i r * certH3 r c * certX i c)) /
        (∑ i : Fin 7, p i * (certY i 0 ^ 2 + certY i 1 ^ 2)) ≤ 560659 / 7066875) := by
  have hB1 : certFit {1, 5, 6} = !![-211 / 520, -5 / 52; 7 / 52, 19 / 78] := by
    ext r c
    fin_cases r <;> fin_cases c <;>
      simp [certFit, certY_eq, certX, inv2Q, Matrix.det_fin_two, Matrix.of_apply,
        Matrix.vecHead, Matrix.vecTail] <;> norm_num
  have hB2 : certFit {3, 5, 6} = !![-333 / 3640, 69 / 364; 401 / 1820, -129 / 182] := by
    ext r c
    fin_cases r <;> fin_cases c <;>
      simp [certFit, certY_eq, certX, inv2Q, Matrix.det_fin_two, Matrix.of_apply,
        Matrix.vecHead, Matrix.vecTail] <;> norm_num
  have hH3 : certH3 = !![119991 / 728000, -18973 / 227500;
                         -18973 / 227500, 301939 / 2730000] := by
    ext r c
    fin_cases r <;> fin_cases c <;>
      simp only [certH3, hB1, hB2, Matrix.add_apply, Matrix.smul_apply, Matrix.mul_apply,
        Matrix.transpose_apply, Fin.sum_univ_two, Matrix.of_apply, smul_eq_mul] <;> norm_num
  have hpt : ∀ i : Fin 7, certDualRatio certH3 certD3 i ≤ 560659 / 7066875 := by
    intro i
    fin_cases i <;>
      simp [certDualRatio, hH3, certD3, certY_eq, certX, Fin.sum_univ_two] <;> norm_num
  have hatt : ∃ i : Fin 7, certDualRatio certH3 certD3 i = 560659 / 7066875 := by
    refine ⟨2, ?_⟩
    simp [certDualRatio, hH3, certD3, certY_eq, certX, Fin.sum_univ_two]
    norm_num
  refine ⟨by norm_num, hatt, hpt, fun p hp hcompat hEpos =>
    weak_duality_of_pointwise hpt hp hcompat hEpos⟩

/-- **Proposition 8 (`k = 4` exact rational dual certificate)**:
Mixing equal-weight fits on supports `{1, 3, 4, 5}` (33%) and `{1, 3, 5, 6}` (67%) with
`100 D₄ = !![-14, 15; 2, -12]` certifies exact maximum dual ratio `381447541 / 7312728225 < 1 / 16`
across EVERY compatible nonnegative test mass vector `p`. -/
theorem dual_cert_k4 :
    (381447541 : ℚ) / 7312728225 < 1 / 16 ∧
    (∃ i : Fin 7, certDualRatio certH4 certD4 i = 381447541 / 7312728225) ∧
    (∀ i : Fin 7, certDualRatio certH4 certD4 i ≤ 381447541 / 7312728225) ∧
    (∀ (p : Fin 7 → ℚ), (∀ i, 0 ≤ p i) →
      (∀ r c : Fin 2, ∑ i : Fin 7, p i * certY i r * certX i c = 0) →
      0 < ∑ i : Fin 7, p i * (certY i 0 ^ 2 + certY i 1 ^ 2) →
      (∑ i : Fin 7, p i * (∑ r : Fin 2, ∑ c : Fin 2, certX i r * certH4 r c * certX i c)) /
        (∑ i : Fin 7, p i * (certY i 0 ^ 2 + certY i 1 ^ 2)) ≤ 381447541 / 7312728225) := by
  have hB1 : certFit {1, 3, 4, 5} = !![243 / 1144, 251 / 1716; -287 / 572, -69 / 286] := by
    ext r c
    fin_cases r <;> fin_cases c <;>
      simp [certFit, certY_eq, certX, inv2Q, Matrix.det_fin_two, Matrix.of_apply,
        Matrix.vecHead, Matrix.vecTail] <;> norm_num
  have hB2 : certFit {1, 3, 5, 6} = !![-607 / 4264, 627 / 2132; -181 / 2132, -87 / 1066] := by
    ext r c
    fin_cases r <;> fin_cases c <;>
      simp [certFit, certY_eq, certX, inv2Q, Matrix.det_fin_two, Matrix.of_apply,
        Matrix.vecHead, Matrix.vecTail] <;> norm_num
  have hH4 : certH4 = !![21570941 / 230767680, -10538371 / 192306400;
                         -10538371 / 192306400, 10728539 / 96153200] := by
    ext r c
    fin_cases r <;> fin_cases c <;>
      simp only [certH4, hB1, hB2, Matrix.add_apply, Matrix.smul_apply, Matrix.mul_apply,
        Matrix.transpose_apply, Fin.sum_univ_two, Matrix.of_apply, smul_eq_mul] <;> norm_num
  have hpt : ∀ i : Fin 7, certDualRatio certH4 certD4 i ≤ 381447541 / 7312728225 := by
    intro i
    fin_cases i <;>
      simp [certDualRatio, hH4, certD4, certY_eq, certX, Fin.sum_univ_two] <;> norm_num
  have hatt : ∃ i : Fin 7, certDualRatio certH4 certD4 i = 381447541 / 7312728225 := by
    refine ⟨0, ?_⟩
    simp [certDualRatio, hH4, certD4, certY_eq, certX, Fin.sum_univ_two]
    norm_num
  refine ⟨by norm_num, hatt, hpt, fun p hp hcompat hEpos =>
    weak_duality_of_pointwise hpt hp hcompat hEpos⟩

end WeightedDataVector
