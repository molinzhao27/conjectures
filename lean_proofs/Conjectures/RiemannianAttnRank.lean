import Mathlib

/-!
# Refutation of Riemannian Attention Rank-Preservation Conjectures and Exact Two-Token Criterion
(Song, arXiv:2608.01283, Section 3, Conjectures `conj:rankpreserve` and `conj:rankstrong`)

We formalize in Lean 4 + Mathlib (complete proofs, standard axioms only) the mathematical cores of
all four main theorems settling `conj:rankpreserve` and `conj:rankstrong`:

1. **Theorem 1 (Single-layer exact rank-one collapse with full-rank active diverse metrics)**:
   - `theorem1_single_layer_collapse`: At `L = d = r = 2`, `H₀ = I₂`, for EVERY `s > 0` and
     `τ > 0`, the queries `q₀ = 0, q₁ = (0, 2)`, keys `k₀ = 0, k₁ = (1, 0)`, and full-rank
     diagonal metrics `U₀ = √s diag(√5, 1)`, `U₁ = √s diag(1, √2)` span `D = ℝ²`, achieve
     metric strength `Ū² = 6s`, positive diversity `σ_U² = (9/4) s² > 0`, within-row metric
     contrast `3s` in every row, and strictly alter Euclidean attention probabilities, yet produce
     two IDENTICAL softmax attention rows `(p, 1 - p)` with `det A = 0` and `effRank2 A = 1`.
2. **Theorem 2 (Deep collapse despite distinct preferred keys and arbitrarily large fixed margins)**:
   - `theorem2_layer_matrix`: For every margin `b > 1` (`s = (b - 1)/2 > 0`), the symmetric
     two-token Riemannian layer with full-rank metrics `U₀ = diag(√s, 1)`, `U₁ = diag(√(3s), 1)`
     has `Ū² = 1 + 3s`, `σ_U² = s² > 0`, metric contrast `2s`, and exact softmax matrix
     `A(b) = J + tanh(b/2) • T`, where `J = (1/2) 11ᵀ` and `T = I₂ - J`.
   - `theorem2_product_formula` & `theorem2_constant_margin_collapse`: For any margin schedule
     `b : ℕ → ℝ`, the depth-`n` product satisfies `P_n = J + (∏_{j < n} tanh(b_j/2)) • T` with
     `effRank2 P_n = 1 + (∏_{j < n} tanh(b_j/2))²`. At any constant margin `b > 1`,
     `effRank2 P_n → 1` as `n → ∞` even as `σ₂(A_n) = tanh(b/2)` is arbitrarily close to `1`.
3. **Theorem 3 core (General Dobrushin ergodicity contraction)**:
   - `dobrushin_pair_eq_one_sub_min`: Exact identity
     `(1/2) ∑_j |A_{ij} - A_{kj}| = 1 - ∑_j min(A_{ij}, A_{kj})` for row-stochastic matrices.
   - `softmax_entry_ge_exp_neg_range` & `softmax_dobrushin_le`: If every row of a score matrix
     has within-row range `≤ R`, then every softmax entry is `≥ exp(-R) / L` and the Dobrushin
     contraction coefficient is `≤ 1 - exp(-R) ≤ exp(-exp(-R))`.
4. **Theorem 4 (Exact two-token rank-preservation criterion)**:
   - `stoch2_dobrushin_eq_abs_det` & `stoch2_prod_abs_det`: For any `2 × 2` row-stochastic
     matrices `A_j`, `δ(A_j) = |1 - a_j - b_j| = |det A_j|` and `|det P_n| = ∏_{j < n} u_j`.
   - `inf_prod_pos_iff_pos_and_summable`: For any sequence `u_j ∈ [0, 1]`,
     `(∃ c > 0, ∀ n, c ≤ ∏_{j < n} u_j) ↔ (∀ j, 0 < u_j) ∧ Summable (fun j => 1 - u_j)`.
   - `theorem4_two_token_exact_criterion`: For any sequence of `2 × 2` row-stochastic matrices
     `A_j` and any invertible `2 × 2` input `H₀`,
     `(∃ δ > 0, ∀ n, 1 + δ ≤ effRank2 (P_n * H₀)) ↔ (∀ j, 0 < u_j) ∧ Summable (fun j => 1 - u_j)`.

Machine-checked vs prose-only:
- Machine-checked: Complete Theorem 1 (all metrics, scores, softmax identity, rank-1 collapse),
  complete Theorem 2 (projection algebra, induction, exact `effRank2 = 1 + t_n²`, convergence to 1),
  Theorem 3 Dobrushin contraction & softmax range bound, and complete Theorem 4 (`iff` criterion
  for arbitrary invertible `2 × 2` input `H₀`).
- Prose-only (in `solutions/riemannian_attn_rank/proof.pdf`): Higher-dimensional `2 ≤ r ≤ d`
  padding of Theorem 1, SiLU MetricNet open-loop decoder realization `B_n = J + t_{n-1}^{-1} T`,
  doubly exponential bounded-weight collapse in Theorem 3(2), and the equilateral-triangle
  zero-limit exception in `ℝ²`/`ℝ³`.
-/

open Matrix BigOperators Finset Filter Topology

namespace RiemannianAttnRank

/-! ### Common Helpers: 2×2 Frobenius Norm, Exact Effective Rank, and Row-Wise Softmax -/

/-- Squared Frobenius norm of a `2 × 2` real matrix. -/
def frobSq2 (M : Matrix (Fin 2) (Fin 2) ℝ) : ℝ :=
  (M 0 0) ^ 2 + (M 0 1) ^ 2 + (M 1 0) ^ 2 + (M 1 1) ^ 2

/-- For any `2 × 2` matrix `M`, `(‖M‖_F²)² ≥ 4 (det M)²` (equivalently `(σ₁² - σ₂²)² ≥ 0`). -/
lemma frobSq2_sq_ge_four_det_sq (M : Matrix (Fin 2) (Fin 2) ℝ) :
    4 * (M.det) ^ 2 ≤ (frobSq2 M) ^ 2 := by
  unfold frobSq2
  rw [Matrix.det_fin_two]
  have h : ((M 0 0) ^ 2 + (M 0 1) ^ 2 + (M 1 0) ^ 2 + (M 1 1) ^ 2) ^ 2 -
      4 * (M 0 0 * M 1 1 - M 0 1 * M 1 0) ^ 2 =
      ((M 0 0) ^ 2 + (M 0 1) ^ 2 - (M 1 0) ^ 2 - (M 1 1) ^ 2) ^ 2 +
      4 * (M 0 0 * M 1 0 + M 0 1 * M 1 1) ^ 2 := by ring
  nlinarith [sq_nonneg ((M 0 0) ^ 2 + (M 0 1) ^ 2 - (M 1 0) ^ 2 - (M 1 1) ^ 2),
             sq_nonneg (M 0 0 * M 1 0 + M 0 1 * M 1 1)]

lemma frobSq2_nonneg (M : Matrix (Fin 2) (Fin 2) ℝ) : 0 ≤ frobSq2 M := by
  unfold frobSq2
  positivity

/-- Exact stable / effective rank `‖M‖_F² / σ₁(M)²` of a `2 × 2` real matrix `M`, using the
exact closed form `2 σ₁² = ‖M‖_F² + √((‖M‖_F²)² - 4 (det M)²)`. -/
noncomputable def effRank2 (M : Matrix (Fin 2) (Fin 2) ℝ) : ℝ :=
  2 * frobSq2 M / (frobSq2 M + Real.sqrt ((frobSq2 M) ^ 2 - 4 * (M.det) ^ 2))

/-- If `M` is a nonzero `2 × 2` matrix with `det M = 0`, then `effRank2 M = 1`. -/
lemma effRank2_eq_one_of_det_zero (M : Matrix (Fin 2) (Fin 2) ℝ)
    (hpos : 0 < frobSq2 M) (hdet : M.det = 0) :
    effRank2 M = 1 := by
  unfold effRank2
  rw [hdet]
  simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, mul_zero, sub_zero,
    Real.sqrt_sq (le_of_lt hpos)]
  have h2 : frobSq2 M + frobSq2 M = 2 * frobSq2 M := by ring
  rw [h2]
  exact div_self (by positivity)

/-- Row-wise softmax of a score matrix `S : Matrix (Fin L) (Fin L) ℝ`:
`softmaxRow S i j = exp(S i j) / ∑_k exp(S i k)`. -/
noncomputable def softmaxRow {L : ℕ} (S : Matrix (Fin L) (Fin L) ℝ) : Matrix (Fin L) (Fin L) ℝ :=
  fun i j => Real.exp (S i j) / ∑ k : Fin L, Real.exp (S i k)

/-- Riemannian squared distance score `-(E_{ij} + G_{ij}) / τ` for `d = 2` queries, keys,
and diagonal metric factors `U_i = diag(u_{i,0}, u_{i,1})`. -/
def riemEuclid2 (q k : Fin 2 → Fin 2 → ℝ) (i j : Fin 2) : ℝ :=
  (q i 0 - k j 0) ^ 2 + (q i 1 - k j 1) ^ 2

noncomputable def riemMetric2 (q k u : Fin 2 → Fin 2 → ℝ) (i j : Fin 2) : ℝ :=
  let v0 := q i 0 - k j 0
  let v1 := q i 1 - k j 1
  ((u i 0 * v0) ^ 2 + (u i 1 * v1) ^ 2 + (u j 0 * v0) ^ 2 + (u j 1 * v1) ^ 2) / 2

noncomputable def riemAttn2 (q k u : Fin 2 → Fin 2 → ℝ) (τ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  softmaxRow (fun i j => -(riemEuclid2 q k i j + riemMetric2 q k u i j) / τ)

/-! ### 1. Theorem 1: Single-Layer Active Full-Rank Collapse -/

/-- **Theorem 1 (Single-layer exact rank-one collapse refuting `conj:rankpreserve` and
`conj:rankstrong`)**: For every `s > 0` and `τ > 0`, with `H₀ = I₂`, queries `q₀ = 0, q₁ = (0, 2)`,
keys `k₀ = 0, k₁ = (1, 0)`, and full-rank diagonal metrics
`U₀ = √s diag(√5, 1)`, `U₁ = √s diag(1, √2)`:
- Active differences `v₀₁ = (-1, 0)` and `v₁₀ = (0, 2)` span `ℝ²` (`det = -2 ≠ 0`);
- Metric Frobenius norms squared are `‖U₀‖_F² = 6s` and `‖U₁‖_F² = 3s`, giving
  `Ū² = 6s` and positive diversity `σ_U² = (9/4) s² > 0`;
- Within-row metric contrast equals `3s` in both rows;
- Yet both rows of the Riemannian softmax attention matrix `A` equal `(p, 1 - p)` with
  `p = (1 + exp(-(1 + 3s)/τ))⁻¹` (strictly different from Euclidean `p_E = (1 + exp(-1/τ))⁻¹`),
  so `A.det = 0` and `effRank2 (A * 1) = 1`. -/
theorem theorem1_single_layer_collapse {s τ : ℝ} (hs : 0 < s) (hτ : 0 < τ) :
    let q : Fin 2 → Fin 2 → ℝ := ![![0, 0], ![0, 2]]
    let k : Fin 2 → Fin 2 → ℝ := ![![0, 0], ![1, 0]]
    let u : Fin 2 → Fin 2 → ℝ := ![![Real.sqrt (5 * s), Real.sqrt s], ![Real.sqrt s, Real.sqrt (2 * s)]]
    let A := riemAttn2 q k u τ
    let p := 1 / (1 + Real.exp (-(1 + 3 * s) / τ))
    let pE := 1 / (1 + Real.exp (-1 / τ))
    let norm0Sq := (u 0 0) ^ 2 + (u 0 1) ^ 2
    let norm1Sq := (u 1 0) ^ 2 + (u 1 1) ^ 2
    let meanNormSq := (norm0Sq + norm1Sq) / 2
    let varNormSq := ((norm0Sq - meanNormSq) ^ 2 + (norm1Sq - meanNormSq) ^ 2) / 2
    norm0Sq = 6 * s ∧
    norm1Sq = 3 * s ∧
    varNormSq = 9 / 4 * s ^ 2 ∧
    riemMetric2 q k u 0 1 - riemMetric2 q k u 0 0 = 3 * s ∧
    riemMetric2 q k u 1 1 - riemMetric2 q k u 1 0 = 3 * s ∧
    A 0 0 = p ∧ A 0 1 = 1 - p ∧ A 1 0 = p ∧ A 1 1 = 1 - p ∧
    p ≠ pE ∧
    A.det = 0 ∧
    effRank2 (A * (1 : Matrix (Fin 2) (Fin 2) ℝ)) = 1 := by
  intro q k u A p pE norm0Sq norm1Sq meanNormSq varNormSq
  have h5s : (Real.sqrt (5 * s)) ^ 2 = 5 * s := Real.sq_sqrt (by linarith)
  have h2s : (Real.sqrt (2 * s)) ^ 2 = 2 * s := Real.sq_sqrt (by linarith)
  have h1s : (Real.sqrt s) ^ 2 = s := Real.sq_sqrt (by linarith)
  have hn0 : norm0Sq = 6 * s := by
    dsimp only [norm0Sq, u]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
    linarith [h5s, h1s]
  have hn1 : norm1Sq = 3 * s := by
    dsimp only [norm1Sq, u]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
    linarith [h2s, h1s]
  have hvar : varNormSq = 9 / 4 * s ^ 2 := by
    dsimp only [varNormSq, meanNormSq]
    rw [hn0, hn1]
    ring
  have hE00 : riemEuclid2 q k 0 0 = 0 := by simp [riemEuclid2, q, k]
  have hE01 : riemEuclid2 q k 0 1 = 1 := by simp [riemEuclid2, q, k]
  have hE10 : riemEuclid2 q k 1 0 = 4 := by simp [riemEuclid2, q, k]; norm_num
  have hE11 : riemEuclid2 q k 1 1 = 5 := by simp [riemEuclid2, q, k]; norm_num
  have hG00 : riemMetric2 q k u 0 0 = 0 := by simp [riemMetric2, q, k]
  have hG01 : riemMetric2 q k u 0 1 = 3 * s := by
    simp only [riemMetric2, q, k, u, Matrix.cons_val_zero, Matrix.cons_val_one, sub_zero, zero_sub,
      mul_neg, mul_one, neg_sq, mul_zero, zero_pow (by norm_num : 2 ≠ 0), add_zero]
    linarith [h5s, h1s]
  have hG10 : riemMetric2 q k u 1 0 = 6 * s := by
    simp only [riemMetric2, q, k, u, Matrix.cons_val_zero, Matrix.cons_val_one, sub_zero,
      mul_zero, zero_pow (by norm_num : 2 ≠ 0), zero_add]
    have hmul : (Real.sqrt s * 2) ^ 2 = 4 * (Real.sqrt s) ^ 2 := by ring
    have hmul2 : (Real.sqrt (2 * s) * 2) ^ 2 = 4 * (Real.sqrt (2 * s)) ^ 2 := by ring
    linarith [h1s, h2s, hmul, hmul2]
  have hG11 : riemMetric2 q k u 1 1 = 9 * s := by
    simp only [riemMetric2, q, k, u, Matrix.cons_val_zero, Matrix.cons_val_one, sub_zero, zero_sub,
      mul_neg, mul_one, neg_sq]
    have hmul : (Real.sqrt (2 * s) * 2) ^ 2 = 4 * (Real.sqrt (2 * s)) ^ 2 := by ring
    linarith [h1s, h2s, h5s, hmul]
  have hden0 : 0 < 1 + Real.exp (-(1 + 3 * s) / τ) := by positivity
  have hA00 : A 0 0 = p := by
    simp only [A, riemAttn2, softmaxRow, Fin.sum_univ_two, hE00, hE01, hG00, hG01,
      add_zero, neg_zero, zero_div, Real.exp_zero, p]
  have hA01 : A 0 1 = 1 - p := by
    simp only [A, riemAttn2, softmaxRow, Fin.sum_univ_two, hE00, hE01, hG00, hG01,
      add_zero, neg_zero, zero_div, Real.exp_zero, p]
    field_simp [ne_of_gt hden0]
    ring
  have hshift : Real.exp (-(5 + 9 * s) / τ) =
      Real.exp (-(4 + 6 * s) / τ) * Real.exp (-(1 + 3 * s) / τ) := by
    rw [← Real.exp_add]
    congr 1
    ring
  have hA10 : A 1 0 = p := by
    simp only [A, riemAttn2, softmaxRow, Fin.sum_univ_two, hE10, hE11, hG10, hG11, p]
    have hexp_pos : 0 < Real.exp (-(4 + 6 * s) / τ) := Real.exp_pos _
    rw [hshift]
    field_simp [ne_of_gt hexp_pos, ne_of_gt hden0]
  have hA11 : A 1 1 = 1 - p := by
    simp only [A, riemAttn2, softmaxRow, Fin.sum_univ_two, hE10, hE11, hG10, hG11, p]
    have hexp_pos : 0 < Real.exp (-(4 + 6 * s) / τ) := Real.exp_pos _
    rw [hshift]
    field_simp [ne_of_gt hexp_pos, ne_of_gt hden0]
    ring
  have hp_ne : p ≠ pE := by
    intro heq
    have hdenE : 0 < 1 + Real.exp (-1 / τ) := by positivity
    have h1 : Real.exp (-(1 + 3 * s) / τ) = Real.exp (-1 / τ) := by
      dsimp only [p, pE] at heq
      have hden0' : (1 + Real.exp (-(1 + 3 * s) / τ)) ≠ 0 := ne_of_gt hden0
      have hdenE' : (1 + Real.exp (-1 / τ)) ≠ 0 := ne_of_gt hdenE
      have hcross : 1 + Real.exp (-1 / τ) = 1 + Real.exp (-(1 + 3 * s) / τ) := by
        have := (div_eq_div_iff hden0' hdenE').mp heq
        linarith
      linarith
    have h2 : -(1 + 3 * s) / τ = -1 / τ := Real.exp_injective h1
    have h3 : -(1 + 3 * s) = -1 := (div_left_inj' (ne_of_gt hτ)).mp h2
    linarith
  have hdet : A.det = 0 := by
    rw [Matrix.det_fin_two, hA00, hA01, hA10, hA11]
    ring
  have hfrob_pos : 0 < frobSq2 A := by
    unfold frobSq2
    rw [hA00, hA01, hA10, hA11]
    have hp_pos : 0 < p := by dsimp only [p]; positivity
    nlinarith [sq_nonneg (1 - p)]
  refine ⟨hn0, hn1, hvar, ?_, ?_, hA00, hA01, hA10, hA11, hp_ne, hdet, ?_⟩
  · linarith [hG01, hG00]
  · linarith [hG11, hG10]
  · rw [Matrix.mul_one]
    exact effRank2_eq_one_of_det_zero A hfrob_pos hdet

/-! ### 2. Theorem 2: Deep Collapse Despite Persistent Large Margins -/

/-- Orthogonal projections `J = (1/2) 11ᵀ` and `T = I₂ - J`. -/
noncomputable def projJ : Matrix (Fin 2) (Fin 2) ℝ := !![1 / 2, 1 / 2; 1 / 2, 1 / 2]
noncomputable def projT : Matrix (Fin 2) (Fin 2) ℝ := !![1 / 2, -(1 / 2); -(1 / 2), 1 / 2]

lemma projJ_mul_projJ : projJ * projJ = projJ := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [projJ, Matrix.mul_apply, Fin.sum_univ_two] <;> norm_num

lemma projT_mul_projT : projT * projT = projT := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [projT, Matrix.mul_apply, Fin.sum_univ_two] <;> norm_num

lemma projJ_mul_projT : projJ * projT = 0 := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [projJ, projT, Matrix.mul_apply, Fin.sum_univ_two]

lemma projT_mul_projJ : projT * projJ = 0 := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [projJ, projT, Matrix.mul_apply, Fin.sum_univ_two]

lemma linCombJT_mul (t₁ t₂ : ℝ) :
    (projJ + t₁ • projT) * (projJ + t₂ • projT) = projJ + (t₁ * t₂) • projT := by
  simp only [Matrix.add_mul, Matrix.mul_add, Matrix.smul_mul, Matrix.mul_smul,
    projJ_mul_projJ, projJ_mul_projT, projT_mul_projJ, projT_mul_projT,
    smul_zero, add_zero, zero_add, smul_smul, mul_comm]

/-- Exact effective rank of `J + t • T` for `0 ≤ t ≤ 1` is `1 + t²`. -/
lemma effRank2_JT {t : ℝ} (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    effRank2 (projJ + t • projT) = 1 + t ^ 2 := by
  have hfrob : frobSq2 (projJ + t • projT) = 1 + t ^ 2 := by
    unfold frobSq2 projJ projT
    simp only [Matrix.add_apply, Matrix.smul_apply, smul_eq_mul, Matrix.of_apply,
      Matrix.cons_val', Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_fin_one]
    ring
  have hdet : (projJ + t • projT).det = t := by
    rw [Matrix.det_fin_two]
    unfold projJ projT
    simp only [Matrix.add_apply, Matrix.smul_apply, smul_eq_mul, Matrix.of_apply,
      Matrix.cons_val', Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_fin_one]
    ring
  unfold effRank2
  rw [hfrob, hdet]
  have hsq : (1 + t ^ 2) ^ 2 - 4 * t ^ 2 = (1 - t ^ 2) ^ 2 := by ring
  rw [hsq, Real.sqrt_sq (by nlinarith)]
  ring

/-- Key hyperbolic identity: `(1 + tanh(b/2)) / 2 = 1 / (1 + exp(-b))`. -/
lemma half_one_add_tanh_half (b : ℝ) :
    (1 + Real.tanh (b / 2)) / 2 = 1 / (1 + Real.exp (-b)) := by
  rw [Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq]
  have hpos1 : 0 < Real.exp (b / 2) := Real.exp_pos _
  have hpos2 : 0 < Real.exp (-(b / 2)) := Real.exp_pos _
  have hsum_pos : 0 < Real.exp (b / 2) + Real.exp (-(b / 2)) := by positivity
  have hden_pos : 0 < 1 + Real.exp (-b) := by positivity
  have hexp : Real.exp (-(b / 2)) = Real.exp (b / 2) * Real.exp (-b) := by
    rw [← Real.exp_add]; congr 1; ring
  field_simp [ne_of_gt hsum_pos, ne_of_gt hden_pos]
  rw [hexp]
  ring

/-- **Theorem 2(a) (Exact symmetric layer matrix `A(b) = J + tanh(b/2) • T`)**:
For every margin `b > 1` (`s = (b - 1)/2 > 0`), with `τ = 1`, `q₀ = k₀ = 0`, `q₁ = k₁ = (1, 0)`,
and full-rank metrics `U₀ = diag(√s, 1)`, `U₁ = diag(√(3s), 1)`:
`Ū² = 1 + 3s`, `σ_U² = s² > 0`, metric contrast equals `2s`, and `A = J + tanh(b/2) • T`. -/
theorem theorem2_layer_matrix {b : ℝ} (hb : 1 < b) :
    let s := (b - 1) / 2
    let q : Fin 2 → Fin 2 → ℝ := ![![0, 0], ![1, 0]]
    let k : Fin 2 → Fin 2 → ℝ := ![![0, 0], ![1, 0]]
    let u : Fin 2 → Fin 2 → ℝ := ![![Real.sqrt s, 1], ![Real.sqrt (3 * s), 1]]
    let A := riemAttn2 q k u 1
    let norm0Sq := (u 0 0) ^ 2 + (u 0 1) ^ 2
    let norm1Sq := (u 1 0) ^ 2 + (u 1 1) ^ 2
    let meanNormSq := (norm0Sq + norm1Sq) / 2
    let varNormSq := ((norm0Sq - meanNormSq) ^ 2 + (norm1Sq - meanNormSq) ^ 2) / 2
    norm0Sq = 1 + s ∧
    norm1Sq = 1 + 3 * s ∧
    varNormSq = s ^ 2 ∧
    riemMetric2 q k u 0 1 = 2 * s ∧
    A = projJ + Real.tanh (b / 2) • projT := by
  intro s q k u A norm0Sq norm1Sq meanNormSq varNormSq
  have hs : 0 < s := by dsimp only [s]; linarith
  have h1s : (Real.sqrt s) ^ 2 = s := Real.sq_sqrt (le_of_lt hs)
  have h3s : (Real.sqrt (3 * s)) ^ 2 = 3 * s := Real.sq_sqrt (by linarith)
  have hn0 : norm0Sq = 1 + s := by
    dsimp only [norm0Sq, u]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, one_pow]
    linarith [h1s]
  have hn1 : norm1Sq = 1 + 3 * s := by
    dsimp only [norm1Sq, u]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, one_pow]
    linarith [h3s]
  have hvar : varNormSq = s ^ 2 := by
    dsimp only [varNormSq, meanNormSq]
    rw [hn0, hn1]
    ring
  have hE00 : riemEuclid2 q k 0 0 = 0 := by simp [riemEuclid2, q, k]
  have hE01 : riemEuclid2 q k 0 1 = 1 := by simp [riemEuclid2, q, k]
  have hE10 : riemEuclid2 q k 1 0 = 1 := by simp [riemEuclid2, q, k]
  have hE11 : riemEuclid2 q k 1 1 = 0 := by simp [riemEuclid2, q, k]
  have hG00 : riemMetric2 q k u 0 0 = 0 := by simp [riemMetric2, q, k]
  have hG11 : riemMetric2 q k u 1 1 = 0 := by simp [riemMetric2, q, k]
  have hG01 : riemMetric2 q k u 0 1 = 2 * s := by
    simp only [riemMetric2, q, k, u, Matrix.cons_val_zero, Matrix.cons_val_one, sub_self, zero_sub,
      mul_neg, mul_one, neg_sq, mul_zero, zero_pow (by norm_num : 2 ≠ 0), add_zero]
    linarith [h1s, h3s]
  have hG10 : riemMetric2 q k u 1 0 = 2 * s := by
    simp only [riemMetric2, q, k, u, Matrix.cons_val_zero, Matrix.cons_val_one, sub_self, sub_zero,
      mul_one, mul_zero, zero_pow (by norm_num : 2 ≠ 0), add_zero]
    linarith [h1s, h3s]
  have h12s : 1 + 2 * s = b := by dsimp only [s]; ring
  have hth := half_one_add_tanh_half b
  have hden_pos : 0 < 1 + Real.exp (-b) := by positivity
  have hth2 : (1 - Real.tanh (b / 2)) / 2 = Real.exp (-b) / (1 + Real.exp (-b)) := by
    have : (1 - Real.tanh (b / 2)) / 2 = 1 - (1 + Real.tanh (b / 2)) / 2 := by ring
    rw [this, hth]
    field_simp [ne_of_gt hden_pos]
    ring
  refine ⟨hn0, hn1, hvar, hG01, ?_⟩
  ext i j
  fin_cases i <;> fin_cases j
  · change A 0 0 = (projJ + Real.tanh (b / 2) • projT) 0 0
    simp only [A, riemAttn2, softmaxRow, Fin.sum_univ_two, hE00, hE01, hG00, hG01, h12s,
      add_zero, neg_zero, div_one, Real.exp_zero, projJ, projT, Matrix.add_apply,
      Matrix.smul_apply, smul_eq_mul, Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero]
    linarith [hth]
  · change A 0 1 = (projJ + Real.tanh (b / 2) • projT) 0 1
    simp only [A, riemAttn2, softmaxRow, Fin.sum_univ_two, hE00, hE01, hG00, hG01, h12s,
      add_zero, neg_zero, div_one, Real.exp_zero, projJ, projT, Matrix.add_apply,
      Matrix.smul_apply, smul_eq_mul, Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero,
      Matrix.cons_val_one]
    linarith [hth2]
  · change A 1 0 = (projJ + Real.tanh (b / 2) • projT) 1 0
    simp only [A, riemAttn2, softmaxRow, Fin.sum_univ_two, hE10, hE11, hG10, hG11, h12s,
      add_zero, neg_zero, div_one, Real.exp_zero, projJ, projT, Matrix.add_apply,
      Matrix.smul_apply, smul_eq_mul, Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero,
      Matrix.cons_val_one]
    rw [add_comm (Real.exp (-b)) 1]; linarith [hth2]
  · change A 1 1 = (projJ + Real.tanh (b / 2) • projT) 1 1
    simp only [A, riemAttn2, softmaxRow, Fin.sum_univ_two, hE10, hE11, hG10, hG11, h12s,
      add_zero, neg_zero, div_one, Real.exp_zero, projJ, projT, Matrix.add_apply,
      Matrix.smul_apply, smul_eq_mul, Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero,
      Matrix.cons_val_one]
    rw [add_comm (Real.exp (-b)) 1]; linarith [hth]

/-- Left-ordered matrix product `P_n = A_{n-1} * ... * A_0` of a matrix sequence `A : ℕ → Matrix`. -/
def matProdLeft {L : ℕ} (A : ℕ → Matrix (Fin L) (Fin L) ℝ) : ℕ → Matrix (Fin L) (Fin L) ℝ
  | 0 => 1
  | n + 1 => A n * matProdLeft A n

/-- **Theorem 2(b) (Exact product formula and constant-margin collapse to effective rank `1`)**:
For any margin sequence `b : ℕ → ℝ` with `b j > 0`:
1. `P_n = J + (∏_{j < n} tanh(b_j / 2)) • T` for every `n ≥ 1`;
2. `effRank2 P_n = 1 + (∏_{j < n} tanh(b_j / 2))²` for every `n ≥ 1`;
3. For any constant margin `b > 0`, `effRank2 P_n = 1 + tanh(b / 2)^{2n} → 1` as `n → ∞`. -/
theorem theorem2_product_and_collapse (b : ℕ → ℝ) (hb : ∀ j, 0 < b j) :
    let A : ℕ → Matrix (Fin 2) (Fin 2) ℝ := fun j => projJ + Real.tanh (b j / 2) • projT
    (∀ n ≥ 1, matProdLeft A n = projJ + (∏ j ∈ Finset.range n, Real.tanh (b j / 2)) • projT) ∧
    (∀ n ≥ 1, effRank2 (matProdLeft A n) = 1 + (∏ j ∈ Finset.range n, Real.tanh (b j / 2)) ^ 2) := by
  intro A
  have hJT1 : projJ + projT = (1 : Matrix (Fin 2) (Fin 2) ℝ) := by
    ext i j; fin_cases i <;> fin_cases j <;> simp [projJ, projT] <;> norm_num
  have hprod : ∀ n, matProdLeft A n = projJ + (∏ j ∈ Finset.range n, Real.tanh (b j / 2)) • projT := by
    intro n
    induction n with
    | zero =>
      simp only [matProdLeft, Finset.range_zero, Finset.prod_empty, one_smul]
      exact hJT1.symm
    | succ k ih =>
      simp only [matProdLeft, ih, A, linCombJT_mul, Finset.prod_range_succ, mul_comm]
  have htanh_bounds : ∀ j, 0 ≤ Real.tanh (b j / 2) ∧ Real.tanh (b j / 2) ≤ 1 := by
    intro j
    have hx : 0 < b j / 2 := half_pos (hb j)
    rw [Real.tanh_eq_sinh_div_cosh]
    have hch : 0 < Real.cosh (b j / 2) := Real.cosh_pos _
    have hsh : 0 ≤ Real.sinh (b j / 2) := le_of_lt (Real.sinh_pos_iff.mpr hx)
    have hlt : Real.sinh (b j / 2) < Real.cosh (b j / 2) := by
      rw [Real.sinh_eq, Real.cosh_eq]
      have : 0 < Real.exp (-(b j / 2)) := Real.exp_pos _
      linarith
    exact ⟨div_nonneg hsh (le_of_lt hch), (div_le_one hch).mpr (le_of_lt hlt)⟩
  refine ⟨fun n _ => hprod n, fun n _ => ?_⟩
  rw [hprod n]
  apply effRank2_JT
  · exact Finset.prod_nonneg fun j _ => (htanh_bounds j).1
  · exact Finset.prod_le_one (fun j _ => (htanh_bounds j).1) (fun j _ => (htanh_bounds j).2)

/-- At constant margin `b > 0`, `effRank2 (matProdLeft (fun _ => J + tanh(b/2) T) n) → 1`
as `n → ∞`. -/
theorem theorem2_constant_margin_collapse {b : ℝ} (hb : 0 < b) :
    let A : ℕ → Matrix (Fin 2) (Fin 2) ℝ := fun _ => projJ + Real.tanh (b / 2) • projT
    Tendsto (fun n => effRank2 (matProdLeft A n)) atTop (𝓝 1) := by
  intro A
  have hcore := (theorem2_product_and_collapse (fun _ => b) (fun _ => hb)).2
  have hth_pos : 0 < Real.tanh (b / 2) := by
    rw [Real.tanh_eq_sinh_div_cosh]
    exact div_pos (Real.sinh_pos_iff.mpr (half_pos hb)) (Real.cosh_pos _)
  have hth_lt1 : Real.tanh (b / 2) < 1 := by
    rw [Real.tanh_eq_sinh_div_cosh, div_lt_one (Real.cosh_pos _), Real.sinh_eq, Real.cosh_eq]
    have : 0 < Real.exp (-(b / 2)) := Real.exp_pos _
    linarith
  have hsq_lt1 : (Real.tanh (b / 2)) ^ 2 < 1 := by nlinarith
  have hsq_nonneg : 0 ≤ (Real.tanh (b / 2)) ^ 2 := sq_nonneg _
  have htend : Tendsto (fun n : ℕ => 1 + ((Real.tanh (b / 2)) ^ 2) ^ n) atTop (𝓝 (1 + 0)) :=
    tendsto_const_nhds.add (tendsto_pow_atTop_nhds_zero_of_lt_one hsq_nonneg hsq_lt1)
  rw [add_zero] at htend
  refine htend.congr' ?_
  filter_upwards [Filter.eventually_ge_atTop 1] with n hn
  rw [hcore n hn, Finset.prod_const, Finset.card_range]
  ring

/-! ### 3. Theorem 3 Core: General Dobrushin Ergodicity Contraction -/

/-- Exact pairwise Dobrushin identity for any two probability distributions `p, q` on `Fin L`:
`(1/2) ∑_j |p_j - q_j| = 1 - ∑_j min(p_j, q_j)`. -/
theorem dobrushin_pair_eq_one_sub_min {L : ℕ} (p q : Fin L → ℝ)
    (hp : ∑ j : Fin L, p j = 1) (hq : ∑ j : Fin L, q j = 1) :
    (1 / 2 : ℝ) * ∑ j : Fin L, |p j - q j| = 1 - ∑ j : Fin L, min (p j) (q j) := by
  have hpt : ∀ j : Fin L, |p j - q j| = p j + q j - 2 * min (p j) (q j) := by
    intro j
    rcases le_total (p j) (q j) with h | h
    · rw [abs_of_nonpos (sub_nonpos.mpr h), min_eq_left h]; ring
    · rw [abs_of_nonneg (sub_nonneg.mpr h), min_eq_right h]; ring
  simp_rw [hpt, Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, hp, hq]
  ring

/-- Softmax entry lower bound: if within row `i` all score differences satisfy
`S i k - S i j ≤ R`, then `softmaxRow S i j ≥ exp(-R) / L`. -/
theorem softmax_entry_ge_exp_neg_range {L : ℕ} (hL : 1 ≤ L)
    (S : Matrix (Fin L) (Fin L) ℝ) (R : ℝ)
    (hrange : ∀ i j k : Fin L, S i k - S i j ≤ R) (i j : Fin L) :
    Real.exp (-R) / (L : ℝ) ≤ softmaxRow S i j := by
  unfold softmaxRow
  have hLpos : (0 : ℝ) < (L : ℝ) := by positivity
  have hsum_pos : 0 < ∑ k : Fin L, Real.exp (S i k) := by
    apply Finset.sum_pos (fun k _ => Real.exp_pos _)
    exact ⟨⟨0, by omega⟩, Finset.mem_univ _⟩
  have hbound : ∑ k : Fin L, Real.exp (S i k) ≤ (L : ℝ) * (Real.exp R * Real.exp (S i j)) := by
    calc ∑ k : Fin L, Real.exp (S i k)
        ≤ ∑ _k : Fin L, Real.exp R * Real.exp (S i j) := by
          apply Finset.sum_le_sum
          intro k _
          rw [← Real.exp_add]
          exact Real.exp_le_exp.mpr (by linarith [hrange i j k])
      _ = (L : ℝ) * (Real.exp R * Real.exp (S i j)) := by
          simp [Finset.sum_const, nsmul_eq_mul]
  rw [div_le_div_iff₀ hLpos hsum_pos]
  have hcancel : Real.exp (-R) * Real.exp R = 1 := by
    rw [← Real.exp_add, neg_add_cancel, Real.exp_zero]
  have hmul := mul_le_mul_of_nonneg_left hbound (le_of_lt (Real.exp_pos (-R)))
  calc Real.exp (-R) * ∑ k : Fin L, Real.exp (S i k)
      ≤ Real.exp (-R) * ((L : ℝ) * (Real.exp R * Real.exp (S i j))) := hmul
    _ = (Real.exp (-R) * Real.exp R) * (Real.exp (S i j) * (L : ℝ)) := by ring
    _ = Real.exp (S i j) * (L : ℝ) := by rw [hcancel, one_mul]

/-- **Theorem 3(1) Dobrushin contraction bound**: If every row of `S` has score range `≤ R`,
then for every pair of rows `(i, k)` of `A = softmaxRow S`,
`(1/2) ∑_j |A_{ij} - A_{kj}| ≤ 1 - exp(-R) ≤ exp(-exp(-R))`. -/
theorem softmax_dobrushin_le {L : ℕ} (hL : 1 ≤ L)
    (S : Matrix (Fin L) (Fin L) ℝ) (R : ℝ)
    (hrange : ∀ i j k : Fin L, S i k - S i j ≤ R) (i k : Fin L) :
    let A := softmaxRow S
    (1 / 2 : ℝ) * ∑ j : Fin L, |A i j - A k j| ≤ 1 - Real.exp (-R) ∧
    1 - Real.exp (-R) ≤ Real.exp (-Real.exp (-R)) := by
  intro A
  have hrowsum : ∀ r : Fin L, ∑ j : Fin L, A r j = 1 := by
    intro r
    unfold A softmaxRow
    rw [← Finset.sum_div]
    have hpos : 0 < ∑ j : Fin L, Real.exp (S r j) := by
      apply Finset.sum_pos (fun j _ => Real.exp_pos _)
      exact ⟨⟨0, by omega⟩, Finset.mem_univ _⟩
    exact div_self (ne_of_gt hpos)
  rw [dobrushin_pair_eq_one_sub_min (A i) (A k) (hrowsum i) (hrowsum k)]
  have hmin : Real.exp (-R) ≤ ∑ j : Fin L, min (A i j) (A k j) := by
    have hpt : ∀ j : Fin L, Real.exp (-R) / (L : ℝ) ≤ min (A i j) (A k j) := fun j =>
      le_min (softmax_entry_ge_exp_neg_range hL S R hrange i j)
             (softmax_entry_ge_exp_neg_range hL S R hrange k j)
    calc Real.exp (-R)
        = (L : ℝ) * (Real.exp (-R) / (L : ℝ)) := by
          have hL0 : (L : ℝ) ≠ 0 := by positivity
          field_simp [hL0]
      _ = ∑ _j : Fin L, Real.exp (-R) / (L : ℝ) := by
          simp [Finset.sum_const, nsmul_eq_mul]
      _ ≤ ∑ j : Fin L, min (A i j) (A k j) := Finset.sum_le_sum (fun j _ => hpt j)
  constructor
  · linarith
  · linarith [Real.add_one_le_exp (-Real.exp (-R))]

/-! ### 4. Theorem 4: Exact Two-Token Rank-Preservation Criterion -/

/-- A `2 × 2` row-stochastic matrix parameterized by off-diagonals `a, b ∈ [0, 1]`. -/
def stoch2 (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1 - a, a; b, 1 - b]

/-- Predicate asserting that a `2 × 2` matrix is row-stochastic. -/
def IsRowStoch2 (A : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  (∀ i j, 0 ≤ A i j) ∧ A 0 0 + A 0 1 = 1 ∧ A 1 0 + A 1 1 = 1

lemma isRowStoch2_one : IsRowStoch2 (1 : Matrix (Fin 2) (Fin 2) ℝ) := by
  refine ⟨fun i j => by fin_cases i <;> fin_cases j <;> simp, by simp, by simp⟩

lemma isRowStoch2_mul {A B : Matrix (Fin 2) (Fin 2) ℝ}
    (hA : IsRowStoch2 A) (hB : IsRowStoch2 B) : IsRowStoch2 (A * B) := by
  refine ⟨?_, ?_, ?_⟩
  · intro i j
    rw [Matrix.mul_apply, Fin.sum_univ_two]
    exact add_nonneg (mul_nonneg (hA.1 i 0) (hB.1 0 j)) (mul_nonneg (hA.1 i 1) (hB.1 1 j))
  · simp only [Matrix.mul_apply, Fin.sum_univ_two]
    calc A 0 0 * B 0 0 + A 0 1 * B 1 0 + (A 0 0 * B 0 1 + A 0 1 * B 1 1)
        = A 0 0 * (B 0 0 + B 0 1) + A 0 1 * (B 1 0 + B 1 1) := by ring
      _ = A 0 0 * 1 + A 0 1 * 1 := by rw [hB.2.1, hB.2.2]
      _ = 1 := by linarith [hA.2.1]
  · simp only [Matrix.mul_apply, Fin.sum_univ_two]
    calc A 1 0 * B 0 0 + A 1 1 * B 1 0 + (A 1 0 * B 0 1 + A 1 1 * B 1 1)
        = A 1 0 * (B 0 0 + B 0 1) + A 1 1 * (B 1 0 + B 1 1) := by ring
      _ = A 1 0 * 1 + A 1 1 * 1 := by rw [hB.2.1, hB.2.2]
      _ = 1 := by linarith [hA.2.2]

lemma isRowStoch2_matProdLeft {A : ℕ → Matrix (Fin 2) (Fin 2) ℝ}
    (hA : ∀ j, IsRowStoch2 (A j)) (n : ℕ) : IsRowStoch2 (matProdLeft A n) := by
  induction n with
  | zero => exact isRowStoch2_one
  | succ k ih => exact isRowStoch2_mul (hA k) ih

/-- Exact `2 × 2` Dobrushin-determinant identity: for any `2 × 2` row-stochastic matrix `A`,
`δ(A) = (1/2) ∑_j |A_{0j} - A_{1j}| = |1 - A_{01} - A_{10}| = |det A| ∈ [0, 1]`. -/
theorem stoch2_dobrushin_eq_abs_det (A : Matrix (Fin 2) (Fin 2) ℝ) (hA : IsRowStoch2 A) :
    (1 / 2 : ℝ) * (|A 0 0 - A 1 0| + |A 0 1 - A 1 1|) = |A.det| ∧
    |A.det| = |1 - A 0 1 - A 1 0| ∧
    |A.det| ≤ 1 := by
  have h00 : A 0 0 = 1 - A 0 1 := by linarith [hA.2.1]
  have h11 : A 1 1 = 1 - A 1 0 := by linarith [hA.2.2]
  have hdet : A.det = 1 - A 0 1 - A 1 0 := by
    rw [Matrix.det_fin_two, h00, h11]; ring
  have hd0 : A 0 0 - A 1 0 = 1 - A 0 1 - A 1 0 := by linarith [h00]
  have hd1 : A 0 1 - A 1 1 = -(1 - A 0 1 - A 1 0) := by linarith [h11]
  refine ⟨?_, ?_, ?_⟩
  · rw [hd0, hd1, abs_neg, hdet]; ring
  · rw [hdet]
  · rw [hdet, abs_le]
    have ha0 : 0 ≤ A 0 1 := hA.1 0 1
    have hb0 : 0 ≤ A 1 0 := hA.1 1 0
    have ha1 : A 0 1 ≤ 1 := by linarith [hA.1 0 0, hA.2.1]
    have hb1 : A 1 0 ≤ 1 := by linarith [hA.1 1 1, hA.2.2]
    constructor <;> linarith

lemma matProdLeft_abs_det (A : ℕ → Matrix (Fin 2) (Fin 2) ℝ) (n : ℕ) :
    |(matProdLeft A n).det| = ∏ j ∈ Finset.range n, |(A j).det| := by
  induction n with
  | zero => simp [matProdLeft]
  | succ k ih =>
    rw [matProdLeft, Matrix.det_mul, abs_mul, ih, Finset.prod_range_succ, mul_comm]

/-- For any `2 × 2` row-stochastic matrix `P`, `1 ≤ frobSq2 P ≤ 2`. -/
lemma isRowStoch2_frobSq2_bounds (P : Matrix (Fin 2) (Fin 2) ℝ) (hP : IsRowStoch2 P) :
    1 ≤ frobSq2 P ∧ frobSq2 P ≤ 2 := by
  unfold frobSq2
  have h0 : 1 / 2 ≤ (P 0 0) ^ 2 + (P 0 1) ^ 2 := by
    nlinarith [sq_nonneg (P 0 0 - P 0 1), hP.2.1]
  have h1 : 1 / 2 ≤ (P 1 0) ^ 2 + (P 1 1) ^ 2 := by
    nlinarith [sq_nonneg (P 1 0 - P 1 1), hP.2.2]
  have hu0 : (P 0 0) ^ 2 + (P 0 1) ^ 2 ≤ 1 := by
    nlinarith [hP.1 0 0, hP.1 0 1, hP.2.1]
  have hu1 : (P 1 0) ^ 2 + (P 1 1) ^ 2 ≤ 1 := by
    nlinarith [hP.1 1 0, hP.1 1 1, hP.2.2]
  exact ⟨by linarith, by linarith⟩

/-- **Exact infinite-product criterion for `[0, 1]`-valued sequences**:
For any sequence `u : ℕ → ℝ` with `0 ≤ u j ≤ 1`,
`(∃ c > 0, ∀ n, c ≤ ∏ j ∈ Finset.range n, u j) ↔ (∀ j, 0 < u j) ∧ Summable (fun j => 1 - u j)`. -/
theorem inf_prod_pos_iff_pos_and_summable (u : ℕ → ℝ) (hu0 : ∀ j, 0 ≤ u j) (hu1 : ∀ j, u j ≤ 1) :
    (∃ c > 0, ∀ n, c ≤ ∏ j ∈ Finset.range n, u j) ↔
    (∀ j, 0 < u j) ∧ Summable (fun j => 1 - u j) := by
  constructor
  · rintro ⟨c, hc, hprod⟩
    have hpos : ∀ j, 0 < u j := by
      intro j
      by_contra hle
      have h0 : u j = 0 := le_antisymm (not_lt.mp hle) (hu0 j)
      have hp0 : ∏ k ∈ Finset.range (j + 1), u k = 0 :=
        Finset.prod_eq_zero (Finset.mem_range.mpr (Nat.lt_succ_self j)) h0
      linarith [hprod (j + 1)]
    refine ⟨hpos, ?_⟩
    apply summable_of_sum_range_le (c := -Real.log c) (fun j => sub_nonneg.mpr (hu1 j))
    intro n
    have hterm : ∀ j, 1 - u j ≤ -Real.log (u j) := by
      intro j
      have h := Real.log_le_sub_one_of_pos (hpos j)
      linarith
    have hsum_log : ∑ j ∈ Finset.range n, Real.log (u j) =
        Real.log (∏ j ∈ Finset.range n, u j) :=
      (Real.log_prod (Finset.range n) u (fun j _ => ne_of_gt (hpos j))).symm
    have hlog_ge : Real.log c ≤ Real.log (∏ j ∈ Finset.range n, u j) :=
      Real.log_le_log hc (hprod n)
    calc ∑ j ∈ Finset.range n, (1 - u j)
        ≤ ∑ j ∈ Finset.range n, (-Real.log (u j)) := Finset.sum_le_sum (fun j _ => hterm j)
      _ = -(∑ j ∈ Finset.range n, Real.log (u j)) := Finset.sum_neg_distrib (f := fun j => Real.log (u j))
      _ = -Real.log (∏ j ∈ Finset.range n, u j) := by rw [hsum_log]
      _ ≤ -Real.log c := by linarith
  · rintro ⟨hpos, hsum⟩
    have htend : Tendsto (fun j => 1 - u j) atTop (𝓝 0) := hsum.tendsto_atTop_zero
    have hev : ∀ᶠ j in atTop, 1 - u j < 1 / 2 :=
      htend.eventually (gt_mem_nhds (by norm_num))
    rw [Filter.eventually_atTop] at hev
    obtain ⟨N, hN⟩ := hev
    have hPN_pos : 0 < ∏ j ∈ Finset.range N, u j :=
      Finset.prod_pos (fun j _ => hpos j)
    let S : ℝ := ∑' j, (1 - u j)
    have hS_nonneg : 0 ≤ S := tsum_nonneg (fun j => sub_nonneg.mpr (hu1 j))
    refine ⟨(∏ j ∈ Finset.range N, u j) * Real.exp (-2 * S), by positivity, ?_⟩
    intro n
    by_cases hn : n ≤ N
    · have hmono : ∏ j ∈ Finset.range N, u j ≤ ∏ j ∈ Finset.range n, u j := by
        have hsub : Finset.range n ⊆ Finset.range N := Finset.range_mono hn
        have hsplit' : (∏ j ∈ Finset.range N \ Finset.range n, u j) *
            (∏ j ∈ Finset.range n, u j) = ∏ j ∈ Finset.range N, u j :=
          Finset.prod_sdiff hsub
        have htail_le : ∏ j ∈ Finset.range N \ Finset.range n, u j ≤ 1 :=
          Finset.prod_le_one (fun j _ => hu0 j) (fun j _ => hu1 j)
        have hhead_nn : 0 ≤ ∏ j ∈ Finset.range n, u j :=
          Finset.prod_nonneg (fun j _ => hu0 j)
        nlinarith
      have hexp_le1 : Real.exp (-2 * S) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
      nlinarith
    · push_neg at hn
      have hsub : Finset.range N ⊆ Finset.range n := Finset.range_mono (le_of_lt hn)
      have hsplit : (∏ j ∈ Finset.range n \ Finset.range N, u j) *
          (∏ j ∈ Finset.range N, u j) = ∏ j ∈ Finset.range n, u j :=
        Finset.prod_sdiff hsub
      have hlog_tail : -2 * S ≤ ∑ j ∈ Finset.range n \ Finset.range N, Real.log (u j) := by
        have hpt : ∀ j ∈ Finset.range n \ Finset.range N, -2 * (1 - u j) ≤ Real.log (u j) := by
          intro j hj
          have hjN : N ≤ j := by
            have := (Finset.mem_sdiff.mp hj).2
            rw [Finset.mem_range] at this
            omega
          have hhalf : 1 / 2 ≤ u j := by linarith [hN j hjN]
          have hinv := Real.log_le_sub_one_of_pos (inv_pos.mpr (hpos j))
          rw [Real.log_inv] at hinv
          have hdiv : (u j)⁻¹ - 1 ≤ 2 * (1 - u j) := by
            have h1 : (u j)⁻¹ - 1 = (1 - u j) / u j := by
              field_simp [ne_of_gt (hpos j)]
            rw [h1, div_le_iff₀ (hpos j)]
            nlinarith [hu1 j]
          linarith
        have hsum_le : ∑ j ∈ Finset.range n \ Finset.range N, (1 - u j) ≤ S :=
          hsum.sum_le_tsum _ (fun j _ => sub_nonneg.mpr (hu1 j))
        calc -2 * S
            ≤ -2 * ∑ j ∈ Finset.range n \ Finset.range N, (1 - u j) := by linarith
          _ = ∑ j ∈ Finset.range n \ Finset.range N, (-2 * (1 - u j)) := by
              rw [Finset.mul_sum]
          _ ≤ ∑ j ∈ Finset.range n \ Finset.range N, Real.log (u j) :=
              Finset.sum_le_sum hpt
      have htail_ge : Real.exp (-2 * S) ≤ ∏ j ∈ Finset.range n \ Finset.range N, u j := by
        have hlog_eq : ∑ j ∈ Finset.range n \ Finset.range N, Real.log (u j) =
            Real.log (∏ j ∈ Finset.range n \ Finset.range N, u j) :=
          (Real.log_prod (Finset.range n \ Finset.range N) u (fun j _ => ne_of_gt (hpos j))).symm
        have htail_pos : 0 < ∏ j ∈ Finset.range n \ Finset.range N, u j :=
          Finset.prod_pos (fun j _ => hpos j)
        rw [← Real.exp_log htail_pos, ← hlog_eq]
        exact Real.exp_le_exp.mpr hlog_tail
      rw [← hsplit, mul_comm (∏ j ∈ Finset.range N, u j)]
      exact mul_le_mul_of_nonneg_right htail_ge (le_of_lt hPN_pos)

/-- Algebraic comparison between `effRank2 M - 1` and `(det M)² / (frobSq2 M)²` for any
nonzero `2 × 2` matrix `M`. -/
lemma effRank2_sub_one_bounds (M : Matrix (Fin 2) (Fin 2) ℝ) (hpos : 0 < frobSq2 M) :
    (M.det) ^ 2 / (frobSq2 M) ^ 2 ≤ effRank2 M - 1 ∧
    effRank2 M - 1 ≤ 4 * (M.det) ^ 2 / (frobSq2 M) ^ 2 := by
  set F := frobSq2 M
  set D := M.det
  set R := Real.sqrt (F ^ 2 - 4 * D ^ 2)
  have hF2 : 0 ≤ F ^ 2 - 4 * D ^ 2 := sub_nonneg.mpr (frobSq2_sq_ge_four_det_sq M)
  have hR0 : 0 ≤ R := Real.sqrt_nonneg _
  have hRle : R ≤ F := by
    have : R ^ 2 ≤ F ^ 2 := by
      rw [Real.sq_sqrt hF2]; nlinarith [sq_nonneg D]
    nlinarith
  have hRsq : R ^ 2 = F ^ 2 - 4 * D ^ 2 := Real.sq_sqrt hF2
  have hden_pos : 0 < F + R := by linarith
  have hexact : effRank2 M - 1 = 4 * D ^ 2 / (F + R) ^ 2 := by
    unfold effRank2
    change 2 * F / (F + R) - 1 = 4 * D ^ 2 / (F + R) ^ 2
    field_simp [ne_of_gt hden_pos]
    nlinarith [hRsq]
  rw [hexact]
  have hden_lo : F ^ 2 ≤ (F + R) ^ 2 := by nlinarith
  have hden_hi : (F + R) ^ 2 ≤ 4 * F ^ 2 := by nlinarith
  constructor
  · rw [div_le_div_iff₀ (sq_pos_of_pos hpos) (sq_pos_of_pos hden_pos)]
    nlinarith [sq_nonneg D]
  · rw [div_le_div_iff₀ (sq_pos_of_pos hden_pos) (sq_pos_of_pos hpos)]
    nlinarith [sq_nonneg D]

/-- Submultiplicativity of `frobSq2` for `2 × 2` matrices: `frobSq2 (A * B) ≤ frobSq2 A * frobSq2 B`. -/
lemma frobSq2_mul_le (A B : Matrix (Fin 2) (Fin 2) ℝ) :
    frobSq2 (A * B) ≤ frobSq2 A * frobSq2 B := by
  unfold frobSq2
  simp only [Matrix.mul_apply, Fin.sum_univ_two]
  have h00 : (A 0 0 * B 0 0 + A 0 1 * B 1 0) ^ 2 ≤ ((A 0 0) ^ 2 + (A 0 1) ^ 2) * ((B 0 0) ^ 2 + (B 1 0) ^ 2) := by
    nlinarith [sq_nonneg (A 0 0 * B 1 0 - A 0 1 * B 0 0)]
  have h01 : (A 0 0 * B 0 1 + A 0 1 * B 1 1) ^ 2 ≤ ((A 0 0) ^ 2 + (A 0 1) ^ 2) * ((B 0 1) ^ 2 + (B 1 1) ^ 2) := by
    nlinarith [sq_nonneg (A 0 0 * B 1 1 - A 0 1 * B 0 1)]
  have h10 : (A 1 0 * B 0 0 + A 1 1 * B 1 0) ^ 2 ≤ ((A 1 0) ^ 2 + (A 1 1) ^ 2) * ((B 0 0) ^ 2 + (B 1 0) ^ 2) := by
    nlinarith [sq_nonneg (A 1 0 * B 1 0 - A 1 1 * B 0 0)]
  have h11 : (A 1 0 * B 0 1 + A 1 1 * B 1 1) ^ 2 ≤ ((A 1 0) ^ 2 + (A 1 1) ^ 2) * ((B 0 1) ^ 2 + (B 1 1) ^ 2) := by
    nlinarith [sq_nonneg (A 1 0 * B 1 1 - A 1 1 * B 0 1)]
  linarith

/-- **Theorem 4 (Exact two-token rank-preservation criterion)**:
Let `A : ℕ → Matrix (Fin 2) (Fin 2) ℝ` be any sequence of `2 × 2` row-stochastic matrices with
`u_j = |det (A j)| = δ(A_j)`, and let `H₀` be any invertible `2 × 2` input (`det H₀ ≠ 0`). Then
`(∃ δ > 0, ∀ n, 1 + δ ≤ effRank2 (P_n * H₀))` holds IF AND ONLY IF
`(∀ j, 0 < u_j) ∧ Summable (fun j => 1 - u_j)`. -/
theorem theorem4_two_token_exact_criterion
    (A : ℕ → Matrix (Fin 2) (Fin 2) ℝ) (hA : ∀ j, IsRowStoch2 (A j))
    (H₀ : Matrix (Fin 2) (Fin 2) ℝ) (hH : H₀.det ≠ 0) :
    (∃ δ > 0, ∀ n, 1 + δ ≤ effRank2 (matProdLeft A n * H₀)) ↔
    (∀ j, 0 < |(A j).det|) ∧ Summable (fun j => 1 - |(A j).det|) := by
  have hu0 : ∀ j, 0 ≤ |(A j).det| := fun j => abs_nonneg _
  have hu1 : ∀ j, |(A j).det| ≤ 1 := fun j => (stoch2_dobrushin_eq_abs_det (A j) (hA j)).2.2
  rw [← inf_prod_pos_iff_pos_and_summable (fun j => |(A j).det|) hu0 hu1]
  have hH_frob_pos : 0 < frobSq2 H₀ := by
    have h4 := frobSq2_sq_ge_four_det_sq H₀
    have hd : 0 < (H₀.det) ^ 2 := sq_pos_of_ne_zero hH
    nlinarith [frobSq2_nonneg H₀]
  have hHinv_mul : H₀ * ((H₀.det)⁻¹ • H₀.adjugate) = 1 := by
    rw [Matrix.mul_smul, Matrix.mul_adjugate, smul_smul, inv_mul_cancel₀ hH, one_smul]
  have hHinv_frob_pos : 0 < frobSq2 ((H₀.det)⁻¹ • H₀.adjugate) := by
    have hdet_inv : ((H₀.det)⁻¹ • H₀.adjugate).det ≠ 0 := by
      intro h0
      have h1 : (H₀ * ((H₀.det)⁻¹ • H₀.adjugate)).det = 0 := by rw [Matrix.det_mul, h0, mul_zero]
      rw [hHinv_mul, Matrix.det_one] at h1
      exact one_ne_zero h1
    have h4 := frobSq2_sq_ge_four_det_sq ((H₀.det)⁻¹ • H₀.adjugate)
    have hd : 0 < (((H₀.det)⁻¹ • H₀.adjugate).det) ^ 2 := sq_pos_of_ne_zero hdet_inv
    nlinarith [frobSq2_nonneg ((H₀.det)⁻¹ • H₀.adjugate)]
  -- Uniform positive lower and upper bounds on frobSq2 (P_n * H₀)
  have hfrob_bounds : ∀ n,
      1 / frobSq2 ((H₀.det)⁻¹ • H₀.adjugate) ≤ frobSq2 (matProdLeft A n * H₀) ∧
      frobSq2 (matProdLeft A n * H₀) ≤ 2 * frobSq2 H₀ := by
    intro n
    have hPn := isRowStoch2_matProdLeft hA n
    obtain ⟨hlo, hhi⟩ := isRowStoch2_frobSq2_bounds (matProdLeft A n) hPn
    constructor
    · have hrec : matProdLeft A n = (matProdLeft A n * H₀) * ((H₀.det)⁻¹ • H₀.adjugate) := by
        rw [Matrix.mul_assoc, hHinv_mul, Matrix.mul_one]
      have hsub := frobSq2_mul_le (matProdLeft A n * H₀) ((H₀.det)⁻¹ • H₀.adjugate)
      rw [← hrec] at hsub
      exact (div_le_iff₀ hHinv_frob_pos).mpr (by linarith)
    · calc frobSq2 (matProdLeft A n * H₀)
          ≤ frobSq2 (matProdLeft A n) * frobSq2 H₀ := frobSq2_mul_le _ _
        _ ≤ 2 * frobSq2 H₀ := mul_le_mul_of_nonneg_right hhi (le_of_lt hH_frob_pos)
  constructor
  · rintro ⟨δ, hδ, heff⟩
    let c₀ : ℝ := 1 / frobSq2 ((H₀.det)⁻¹ • H₀.adjugate)
    have hc₀ : 0 < c₀ := div_pos one_pos hHinv_frob_pos
    refine ⟨Real.sqrt (δ * c₀ ^ 2 / (4 * (H₀.det) ^ 2)), ?_, ?_⟩
    · apply Real.sqrt_pos.mpr
      have : 0 < (H₀.det) ^ 2 := sq_pos_of_ne_zero hH
      positivity
    · intro n
      have hFn_lo : c₀ ≤ frobSq2 (matProdLeft A n * H₀) := (hfrob_bounds n).1
      have hFn_pos : 0 < frobSq2 (matProdLeft A n * H₀) := lt_of_lt_of_le hc₀ hFn_lo
      have hup := (effRank2_sub_one_bounds (matProdLeft A n * H₀) hFn_pos).2
      have hδ_le : δ ≤ 4 * ((matProdLeft A n * H₀).det) ^ 2 / (frobSq2 (matProdLeft A n * H₀)) ^ 2 := by
        linarith [heff n]
      have hFn_sq : c₀ ^ 2 ≤ (frobSq2 (matProdLeft A n * H₀)) ^ 2 := by nlinarith
      have hdet_sq : δ * c₀ ^ 2 ≤ 4 * ((matProdLeft A n).det) ^ 2 * (H₀.det) ^ 2 := by
        have h1 : δ * (frobSq2 (matProdLeft A n * H₀)) ^ 2 ≤ 4 * ((matProdLeft A n * H₀).det) ^ 2 :=
          (le_div_iff₀ (sq_pos_of_pos hFn_pos)).mp hδ_le
        rw [Matrix.det_mul, mul_pow] at h1
        nlinarith
      rw [← matProdLeft_abs_det]
      have hsq : δ * c₀ ^ 2 / (4 * (H₀.det) ^ 2) ≤ (|(matProdLeft A n).det|) ^ 2 := by
        rw [sq_abs, div_le_iff₀ (by positivity)]
        linarith
      have hroot := Real.sqrt_le_sqrt hsq
      rwa [Real.sqrt_sq (abs_nonneg _)] at hroot
  · rintro ⟨c, hc, hprod⟩
    let M₀ : ℝ := 2 * frobSq2 H₀
    have hM₀ : 0 < M₀ := by positivity
    refine ⟨c ^ 2 * (H₀.det) ^ 2 / M₀ ^ 2, ?_, ?_⟩
    · have : 0 < (H₀.det) ^ 2 := sq_pos_of_ne_zero hH
      positivity
    · intro n
      have hFn_pos : 0 < frobSq2 (matProdLeft A n * H₀) :=
        lt_of_lt_of_le (div_pos one_pos hHinv_frob_pos) (hfrob_bounds n).1
      have hFn_hi : frobSq2 (matProdLeft A n * H₀) ≤ M₀ := (hfrob_bounds n).2
      have hlo := (effRank2_sub_one_bounds (matProdLeft A n * H₀) hFn_pos).1
      have hdet_ge : c ≤ |(matProdLeft A n).det| := by
        rw [matProdLeft_abs_det]; exact hprod n
      have hnum : c ^ 2 * (H₀.det) ^ 2 ≤ ((matProdLeft A n * H₀).det) ^ 2 := by
        rw [Matrix.det_mul, mul_pow]
        have hcsq : c ^ 2 ≤ ((matProdLeft A n).det) ^ 2 := by
          have := sq_le_sq' (by linarith [abs_nonneg (matProdLeft A n).det]) hdet_ge
          rwa [sq_abs] at this
        exact mul_le_mul_of_nonneg_right hcsq (sq_nonneg _)
      have hden : (frobSq2 (matProdLeft A n * H₀)) ^ 2 ≤ M₀ ^ 2 := by nlinarith
      have hdiv : c ^ 2 * (H₀.det) ^ 2 / M₀ ^ 2 ≤
          ((matProdLeft A n * H₀).det) ^ 2 / (frobSq2 (matProdLeft A n * H₀)) ^ 2 := by
        rw [div_le_div_iff₀ (sq_pos_of_pos hM₀) (sq_pos_of_pos hFn_pos)]
        nlinarith [sq_nonneg (H₀.det)]
      linarith

end RiemannianAttnRank
