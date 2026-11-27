import Mathlib

/-!
# Contraction Bounds for Random-Permutation Coordinate Descent on General Quadratics
(Donghwa Kim, Jaewook Lee, and Chulhee Yun, *Provable Benefit of Random Permutations
over Uniform Sampling in Stochastic Coordinate Descent*, ICML 2025, PMLR 267,
arXiv:2505.23152, Section 4, Conjecture 4.1 `conj:general`)

We formalize in Lean 4 + Mathlib (complete proofs with no unfinished placeholders,
standard Lean foundations only) the exact algebraic, combinatorial, spectral, and
contraction cores of Theorems 1–2 and Propositions 3–4:

1. **RPCD triangular system and one-epoch energy identity**:
   - `gamma_add_transpose`: for symmetric unit-diagonal `A`, `Γ_p + Γ_pᵀ = A + I`.
   - `gamma_det_one`: `det(Γ_p) = 1` for every permutation `p ∈ S_n` (hence `Γ_p` is always invertible).
   - `rpcd_triangular_system`: sequential coordinate-descent epoch `M_p = U_{p(n)} ⋯ U_{p(1)}`
     satisfies `Γ_p (x - M_p x) = A x`.
   - `rpcd_energy_identity` and `rpcd_energy_identity_inv`: exact one-epoch energy drop
     `xᵀ A x - (M_p x)ᵀ A (M_p x) = ‖x - M_p x‖² = xᵀ A Γ_p⁻ᵀ Γ_p⁻¹ A x`.
2. **Exact combinatorial expectation of `Γ_p Γ_pᵀ` (Theorem 1 core)**:
   - `card_perm_lt`: exact 2-cycle involution count `2 · |{p : p⁻¹ k < p⁻¹ i}| = n!` (`i ≠ k`).
   - `card_perm_lt_lt`: exact 3-cycle bijection count `3 · |{p : p⁻¹ k < p⁻¹ i ∧ p⁻¹ k < p⁻¹ j}| = n!`
     for pairwise distinct `i, j, k`.
   - `expected_gamma_gram`: for every `n ≥ 2` and symmetric unit-diagonal `A` (`F = A - I`),
     `E_p[Γ_p Γ_pᵀ] = A + (1/3) F² + (1/6) diag(F²)`.
3. **All-dimensional `λ_max(A) ≤ 2` contraction theorem (Theorem 1, end-to-end)**:
   - `variational_epoch_bound`: operator-convexity variational inequality
     `2 (A x)ᵀ z - zᵀ E_p[Γ_p Γ_pᵀ] z ≤ xᵀ A x - E_p[(M_p x)ᵀ A (M_p x)]`.
   - `spectral_J_le_h_A`: via `Matrix.IsHermitian.spectral_theorem`, for symmetric unit-diagonal `A`
     with `s I ⪯ A ⪯ 2 I` (`0 < s ≤ 1`), proves `F² ⪯ (1 - t) F + t I`, `diag(F²) ⪯ t I`,
     and `J = (1/3) F² + (1/6) diag(F²) ⪯ h A` where `h = max (t(3 - 2s)/(6s)) ((2 + t)/12)`.
   - `h_div_one_add_h_le_B`: exact scalar inequality
     `h / (1 + h) ≤ max(1/4, (1 - s)²) ≤ B_n(s)` for every `n ≥ 2`, `s ∈ (0, 1]`.
   - `all_dim_lambda_max_le_two`: full end-to-end proof that
     `E_p[(M_p x)ᵀ A (M_p x)] ≤ B_n(s) · xᵀ A x` for EVERY `n ≥ 2`, EVERY symmetric unit-diagonal
     `A` with `s I ⪯ A ⪯ 2 I`, and EVERY `x ∈ ℝ^n`.
4. **Dimension `n = 3` contraction chain and cubic positivity criterion (Theorem 2)**:
   - `dim3_sq_le_cube`: `(1 - s/2)² ≤ (1 - s/3)³` on `[0, 1]` via `(1 - s/3)³ - (1 - s/2)² = s²(1/12 - s/27)`.
   - `dim3_contraction_le_B3`: `(1 - s/2)⁴ ≤ (1 - s/3)⁶ ≤ B_3(s)` for all `s ∈ [0, 1]`.
   - `cubic_no_neg_root`: nonnegativity of characteristic coefficients `e₁, e₂, e₃ ≥ 0` excludes
     negative real eigenvalues.
5. **Weighted-`ℓ_∞` contraction and direct-sum closure (Propositions 3 & 4)**:
   - `weighted_linf_step`, `weighted_linf_epoch`, `weighted_linf_iter`: deterministic per-epoch
     weighted-`ℓ_∞` contraction `|M_p x|_w ≤ r |x|_w` for every permutation `p` whenever
     `|I - A| w ≤ r w`, and `K`-epoch squared-Euclidean-norm contraction `≤ r^{2K} α² ‖w‖₂²`.
   - `B_mono_n` and `B_anti_s`: monotonicity of `B_n(s)` in `n ≥ 1` and antitonicity in `s ∈ [0, 1]`.
   - `direct_sum_closure`: block-diagonal direct sums of blocks satisfying Conjecture 4.1 satisfy
     Conjecture 4.1.

Machine-checked vs. prose-only (in `solutions/rpcd_contraction/proof.pdf`):
- Machine-checked: Complete RPCD triangular system and energy identity, `det(Γ_p) = 1`, exact
  combinatorial expectation `E_p[Γ_p Γ_pᵀ]`, complete end-to-end Theorem 1 (`λ_max(A) ≤ 2` in
  all dimensions `n ≥ 2`), `n = 3` scalar bound `(1 - s/2)⁴ ≤ (1 - s/3)⁶ ≤ B_3(s)` and cubic
  positivity criterion, weighted-`ℓ_∞` contraction (Proposition 3), and `B_n(s)` monotonicity
  + direct-sum closure (Proposition 4).
- Prose-only / Python-certified (`_certificate.py`, `explore.py`): Spectral-radius Gelfand root
  limit equivalence (Lemma 1), the 19,986-monomial exact integer polynomial positivity
  certificate for `P(u, v, w)` in dimension 3 (Theorem 2), Perron–Frobenius spectral radius
  for symmetric M-matrices, and the marginal uniformity of restricting a random permutation to
  direct-sum blocks.
-/

open scoped BigOperators
open Finset Matrix

namespace RPCDContraction

/-! ### 1. Definitions, Triangular System, and One-Epoch Energy Identity -/

/-- Conjecture 4.1 target bound `B_n(s) = max((1 - 1/n)^n, (1 - s/n)^{2n})`. -/
noncomputable def B (n : ℕ) (s : ℝ) : ℝ :=
  max ((1 - 1 / (n : ℝ)) ^ n) ((1 - s / (n : ℝ)) ^ (2 * n))

/-- Single coordinate-descent update `U_i x = (I - e_i e_iᵀ A) x` (assuming `A i i = 1`). -/
def coordUpdate {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) (i : Fin n) (x : Fin n → ℝ) : Fin n → ℝ :=
  fun j => if j = i then x i - (A *ᵥ x) i else x j

/-- Apply the first `k` updates `U_{p(k-1)} ⋯ U_{p(0)}` of permutation `p`. -/
def partialEpoch {n : ℕ} (p : Equiv.Perm (Fin n)) (A : Matrix (Fin n) (Fin n) ℝ) :
    ℕ → (Fin n → ℝ) → (Fin n → ℝ)
  | 0, x => x
  | k + 1, x =>
    if h : k < n then coordUpdate A (p ⟨k, h⟩) (partialEpoch p A k x)
    else partialEpoch p A k x

/-- Full RPCD epoch map `M_p x = U_{p(n-1)} ⋯ U_{p(0)} x`. -/
def epochMap {n : ℕ} (p : Equiv.Perm (Fin n)) (A : Matrix (Fin n) (Fin n) ℝ)
    (x : Fin n → ℝ) : Fin n → ℝ :=
  partialEpoch p A n x

/-- Strictly permuted lower-triangular factor `Γ_p`:
unit diagonal and `(i, j)` entry `A i j` whenever `j` precedes `i` in `p`. -/
def Gamma {n : ℕ} (p : Equiv.Perm (Fin n)) (A : Matrix (Fin n) (Fin n) ℝ) :
    Matrix (Fin n) (Fin n) ℝ :=
  Matrix.of fun i j => if i = j then 1 else if p.symm j < p.symm i then A i j else 0

/-- Key splitting identity: for symmetric unit-diagonal `A`, `Γ_p + Γ_pᵀ = A + I`. -/
theorem gamma_add_transpose {n : ℕ} (p : Equiv.Perm (Fin n))
    (A : Matrix (Fin n) (Fin n) ℝ) (hAsym : Aᵀ = A) (hAdiag : ∀ i, A i i = 1) :
    Gamma p A + (Gamma p A)ᵀ = A + 1 := by
  ext i j
  simp only [Matrix.add_apply, Matrix.transpose_apply, Gamma, Matrix.of_apply, Matrix.one_apply]
  by_cases hij : i = j
  · subst hij; simp [hAdiag]
  · have hji : j ≠ i := Ne.symm hij
    have hne : p.symm j ≠ p.symm i := fun h => hji (p.symm.injective h)
    have hAji : A j i = A i j := by
      have := congrFun (congrFun hAsym i) j
      simpa using this
    rcases lt_or_gt_of_ne hne with hlt | hgt
    · have hnot : ¬(p.symm i < p.symm j) := not_lt_of_gt hlt
      simp [hij, hji, hlt, hnot]
    · have hnot : ¬(p.symm j < p.symm i) := not_lt_of_gt hgt
      simp [hij, hji, hgt, hnot, hAji]

/-- Helper: adjoint transpose identity `u ⬝ᵥ (Mᵀ *ᵥ v) = (M *ᵥ u) ⬝ᵥ v`. -/
lemma dotProduct_mulVec_transpose {n : ℕ} (M : Matrix (Fin n) (Fin n) ℝ) (u v : Fin n → ℝ) :
    dotProduct u (Mᵀ *ᵥ v) = dotProduct (M *ᵥ u) v := by
  simp only [dotProduct, mulVec, Matrix.transpose_apply, Finset.mul_sum, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl; intro i _
  apply Finset.sum_congr rfl; intro j _
  ring

/-- `det(Γ_p) = 1` for every permutation `p` and every matrix `A` (since `Γ_p` is
permutation-conjugate to a unit lower-triangular matrix). -/
theorem gamma_det_one {n : ℕ} (p : Equiv.Perm (Fin n)) (A : Matrix (Fin n) (Fin n) ℝ) :
    (Gamma p A).det = 1 := by
  let L : Matrix (Fin n) (Fin n) ℝ := Matrix.submatrix (Gamma p A) p p
  have hdet_eq : Lᵀ.det = (Gamma p A).det := by
    rw [Matrix.det_transpose, Matrix.det_submatrix_equiv_self]
  rw [← hdet_eq]
  have hupper : Lᵀ.BlockTriangular id := by
    intro i j hij
    simp only [L, Matrix.transpose_apply, Matrix.submatrix_apply, Gamma, Matrix.of_apply,
      Equiv.symm_apply_apply]
    have hne : p j ≠ p i := fun h => ne_of_lt hij (p.injective h)
    have hnot : ¬(i < j) := not_lt_of_gt hij
    simp [hne, hnot]
  rw [Matrix.det_of_upperTriangular hupper]
  simp [L, Gamma]

lemma partialEpoch_unchanged_after {n : ℕ} (p : Equiv.Perm (Fin n))
    (A : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → ℝ) :
    ∀ k (j : Fin n), k ≤ (p.symm j).val → partialEpoch p A k x j = x j := by
  intro k
  induction k with
  | zero => intro j _; rfl
  | succ k ih =>
    intro j hj
    rw [partialEpoch]
    split_ifs with hkn
    · unfold coordUpdate
      have hne : j ≠ p ⟨k, hkn⟩ := by
        intro heq
        have : (p.symm j).val = k := by rw [heq, Equiv.symm_apply_apply]
        omega
      simp only [hne, ↓reduceIte]
      exact ih j (by omega)
    · exact ih j (by omega)

lemma partialEpoch_frozen_before {n : ℕ} (p : Equiv.Perm (Fin n))
    (A : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → ℝ) :
    ∀ k (j : Fin n), (p.symm j).val < k → k ≤ n →
      partialEpoch p A k x j = epochMap p A x j := by
  intro k j hjk hkn
  unfold epochMap
  have hstep : ∀ m, k ≤ m → m ≤ n → partialEpoch p A m x j = partialEpoch p A k x j := by
    intro m hkm hmn
    induction m, hkm using Nat.le_induction with
    | base => rfl
    | succ m hkm ihm =>
      rw [partialEpoch]
      have hmlt : m < n := by omega
      simp only [hmlt, ↓reduceDIte]
      unfold coordUpdate
      have hne : j ≠ p ⟨m, hmlt⟩ := by
        intro heq
        have : (p.symm j).val = m := by rw [heq, Equiv.symm_apply_apply]
        omega
      simp only [hne, ↓reduceIte]
      exact ihm (by omega)
  exact (hstep n hkn le_rfl).symm

/-- **RPCD triangular system**: for unit-diagonal `A`, `Γ_p (x - M_p x) = A x`. -/
theorem rpcd_triangular_system {n : ℕ} (p : Equiv.Perm (Fin n))
    (A : Matrix (Fin n) (Fin n) ℝ) (hAdiag : ∀ i, A i i = 1) (x : Fin n → ℝ) :
    Gamma p A *ᵥ (x - epochMap p A x) = A *ᵥ x := by
  ext i
  set t : ℕ := (p.symm i).val
  have ht : t < n := (p.symm i).isLt
  have hpi : p ⟨t, ht⟩ = i := by simp [t]
  let x_prev := partialEpoch p A t x
  let x_next := partialEpoch p A (t + 1) x
  have hx_next : x_next = coordUpdate A i x_prev := by
    dsimp only [x_next, x_prev]
    rw [partialEpoch]
    simp [ht, hpi]
  have hgrad_zero : (A *ᵥ x_next) i = 0 := by
    rw [hx_next]
    change ∑ j : Fin n, A i j * coordUpdate A i x_prev j = 0
    simp only [coordUpdate]
    have hsplit : ∀ j : Fin n,
        A i j * (if j = i then x_prev i - (A *ᵥ x_prev) i else x_prev j) =
        A i j * x_prev j - (if i = j then A i i * (A *ᵥ x_prev) i else 0) := by
      intro j
      by_cases hji : j = i
      · subst hji; simp; ring
      · simp [hji, Ne.symm hji]
    rw [Finset.sum_congr rfl (fun j _ => hsplit j), Finset.sum_sub_distrib, Finset.sum_ite_eq]
    simp [hAdiag i, mulVec, dotProduct]
  have hx_next_val : ∀ j : Fin n,
      x_next j = if p.symm j ≤ p.symm i then epochMap p A x j else x j := by
    intro j
    by_cases hle : p.symm j ≤ p.symm i
    · simp only [hle, ↓reduceIte]
      have hlt : (p.symm j).val < t + 1 := Nat.lt_succ_of_le hle
      exact partialEpoch_frozen_before p A x (t + 1) j hlt (Nat.succ_le_of_lt ht)
    · simp only [hle, ↓reduceIte]
      have hgt : t + 1 ≤ (p.symm j).val := Nat.succ_le_of_lt (not_le.1 hle)
      exact partialEpoch_unchanged_after p A x (t + 1) j hgt
  have hdiff : (A *ᵥ x) i - (Gamma p A *ᵥ (x - epochMap p A x)) i = (A *ᵥ x_next) i := by
    simp only [mulVec, dotProduct, Pi.sub_apply, Gamma, Matrix.of_apply, ← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl; intro j _
    rw [hx_next_val j]
    by_cases hij : i = j
    · subst hij
      simp [hAdiag i]
    · have hne : p.symm j ≠ p.symm i := fun h => hij (p.symm.injective h).symm
      by_cases hlt : p.symm j < p.symm i
      · have hle : p.symm j ≤ p.symm i := le_of_lt hlt
        simp [hij, hlt, hle]; ring
      · have hnotle : ¬(p.symm j ≤ p.symm i) := fun h => hlt (lt_of_le_of_ne h hne)
        simp [hij, hlt, hnotle]
  linarith

/-- **RPCD one-epoch energy identity**:
For symmetric unit-diagonal `A`,
`xᵀ A x - (M_p x)ᵀ A (M_p x) = ‖x - M_p x‖²`. -/
theorem rpcd_energy_identity {n : ℕ} (p : Equiv.Perm (Fin n))
    (A : Matrix (Fin n) (Fin n) ℝ) (hAsym : Aᵀ = A) (hAdiag : ∀ i, A i i = 1)
    (x : Fin n → ℝ) :
    dotProduct x (A *ᵥ x) - dotProduct (epochMap p A x) (A *ᵥ epochMap p A x) =
      dotProduct (x - epochMap p A x) (x - epochMap p A x) := by
  set d := x - epochMap p A x
  have hxM : epochMap p A x = x - d := by ext i; simp [d]
  have hGd : Gamma p A *ᵥ d = A *ᵥ x := rpcd_triangular_system p A hAdiag x
  have hsym : ∀ u v : Fin n → ℝ, dotProduct u (A *ᵥ v) = dotProduct v (A *ᵥ u) := by
    intro u v
    calc dotProduct u (A *ᵥ v) = dotProduct u (Aᵀ *ᵥ v) := by rw [hAsym]
      _ = dotProduct (A *ᵥ u) v := dotProduct_mulVec_transpose A u v
      _ = dotProduct v (A *ᵥ u) := dotProduct_comm _ _
  have hquad : ∀ a b : Fin n → ℝ,
      dotProduct a (A *ᵥ a) - dotProduct (a - b) (A *ᵥ (a - b)) =
      2 * dotProduct b (A *ᵥ a) - dotProduct b (A *ᵥ b) := by
    intro a b
    have h1 : dotProduct (a - b) (A *ᵥ (a - b)) =
        dotProduct a (A *ᵥ a) - dotProduct a (A *ᵥ b) - dotProduct b (A *ᵥ a) + dotProduct b (A *ᵥ b) := by
      rw [Matrix.mulVec_sub, dotProduct_sub, sub_dotProduct, sub_dotProduct]; ring
    rw [h1, hsym a b]; ring
  rw [hxM, hquad x d, ← hGd]
  have hGt : dotProduct d ((Gamma p A)ᵀ *ᵥ d) = dotProduct d (Gamma p A *ᵥ d) := by
    rw [dotProduct_mulVec_transpose, dotProduct_comm]
  have hsplit := gamma_add_transpose p A hAsym hAdiag
  have hsum : dotProduct d ((Gamma p A + (Gamma p A)ᵀ) *ᵥ d) =
      dotProduct d ((A + 1) *ᵥ d) := by rw [hsplit]
  rw [Matrix.add_mulVec, Matrix.add_mulVec, dotProduct_add, dotProduct_add,
    Matrix.one_mulVec, hGt] at hsum
  linarith

/-- Matrix-inverse form of the RPCD one-epoch energy identity:
if `Γinv * Γ_p = 1`, then `xᵀ A x - (M_p x)ᵀ A (M_p x) = (A x)ᵀ (Γinvᵀ Γinv) (A x)`. -/
theorem rpcd_energy_identity_inv {n : ℕ} (p : Equiv.Perm (Fin n))
    (A Γinv : Matrix (Fin n) (Fin n) ℝ) (hAsym : Aᵀ = A) (hAdiag : ∀ i, A i i = 1)
    (hinv : Γinv * Gamma p A = 1) (x : Fin n → ℝ) :
    dotProduct x (A *ᵥ x) - dotProduct (epochMap p A x) (A *ᵥ epochMap p A x) =
      dotProduct (A *ᵥ x) ((Γinvᵀ * Γinv) *ᵥ (A *ᵥ x)) := by
  rw [rpcd_energy_identity p A hAsym hAdiag x]
  have hd : x - epochMap p A x = Γinv *ᵥ (A *ᵥ x) := by
    calc x - epochMap p A x = (1 : Matrix (Fin n) (Fin n) ℝ) *ᵥ (x - epochMap p A x) := by simp
      _ = (Γinv * Gamma p A) *ᵥ (x - epochMap p A x) := by rw [hinv]
      _ = Γinv *ᵥ (Gamma p A *ᵥ (x - epochMap p A x)) := by rw [Matrix.mulVec_mulVec]
      _ = Γinv *ᵥ (A *ᵥ x) := by rw [rpcd_triangular_system p A hAdiag x]
  rw [hd, ← Matrix.mulVec_mulVec, dotProduct_mulVec_transpose]

/-! ### 2. Exact Combinatorial Expectation `E_p[Γ_p Γ_pᵀ] = A + (1/3) F² + (1/6) diag(F²)` -/

/-- Exact 2-cycle involution count: `2 · |{p : p⁻¹ k < p⁻¹ i}| = n!` for `i ≠ k`. -/
theorem card_perm_lt {n : ℕ} {i k : Fin n} (hik : i ≠ k) :
    2 * (Finset.univ.filter (fun p : Equiv.Perm (Fin n) => p.symm k < p.symm i)).card =
      Fintype.card (Equiv.Perm (Fin n)) := by
  let S₁ := Finset.univ.filter (fun p : Equiv.Perm (Fin n) => p.symm k < p.symm i)
  let S₂ := Finset.univ.filter (fun p : Equiv.Perm (Fin n) => p.symm i < p.symm k)
  have hbij : S₁.card = S₂.card := by
    apply Finset.card_bij (fun p _ => Equiv.swap i k * p)
    · intro p hp
      simp only [S₁, S₂, Finset.mem_filter, Finset.mem_univ, true_and] at hp ⊢
      have h1 : (Equiv.swap i k * p).symm i = p.symm k := by
        simp [Equiv.Perm.mul_def]
      have h2 : (Equiv.swap i k * p).symm k = p.symm i := by
        simp [Equiv.Perm.mul_def]
      rw [h1, h2]; exact hp
    · intro p₁ _ p₂ _ h
      exact mul_left_cancel h
    · intro p hp
      refine ⟨Equiv.swap i k * p, ?_, by simp [← mul_assoc]⟩
      simp only [S₁, S₂, Finset.mem_filter, Finset.mem_univ, true_and] at hp ⊢
      have h1 : (Equiv.swap i k * p).symm i = p.symm k := by simp [Equiv.Perm.mul_def]
      have h2 : (Equiv.swap i k * p).symm k = p.symm i := by simp [Equiv.Perm.mul_def]
      rw [h1, h2]; exact hp
  have hdisj : Disjoint S₁ S₂ := by
    rw [Finset.disjoint_left]; intro p h1 h2
    simp only [S₁, S₂, Finset.mem_filter, Finset.mem_univ, true_and] at h1 h2
    exact lt_asymm h1 h2
  have hunion : S₁ ∪ S₂ = Finset.univ := by
    ext p
    simp only [S₁, S₂, Finset.mem_union, Finset.mem_filter, Finset.mem_univ, true_and, iff_true]
    exact lt_or_gt_of_ne (fun h => hik (p.symm.injective h).symm)
  have htot := Finset.card_union_of_disjoint hdisj
  rw [hunion, Finset.card_univ] at htot
  dsimp only [S₁] at hbij htot
  omega

/-- Exact 3-cycle bijection count:
`3 · |{p : p⁻¹ k < p⁻¹ i ∧ p⁻¹ k < p⁻¹ j}| = n!` for pairwise distinct `i, j, k`. -/
theorem card_perm_lt_lt {n : ℕ} {i j k : Fin n} (hij : i ≠ j) (hik : i ≠ k) (hjk : j ≠ k) :
    3 * (Finset.univ.filter (fun p : Equiv.Perm (Fin n) =>
      p.symm k < p.symm i ∧ p.symm k < p.symm j)).card =
      Fintype.card (Equiv.Perm (Fin n)) := by
  let Sk := Finset.univ.filter (fun p : Equiv.Perm (Fin n) => p.symm k < p.symm i ∧ p.symm k < p.symm j)
  let Si := Finset.univ.filter (fun p : Equiv.Perm (Fin n) => p.symm i < p.symm j ∧ p.symm i < p.symm k)
  let Sj := Finset.univ.filter (fun p : Equiv.Perm (Fin n) => p.symm j < p.symm k ∧ p.symm j < p.symm i)
  have hki : k ≠ i := Ne.symm hik
  have hji : j ≠ i := Ne.symm hij
  have hkj : k ≠ j := Ne.symm hjk
  -- Swap i and k gives bijection Sk ≃ Si
  have h1 : Sk.card = Si.card := by
    apply Finset.card_bij (fun p _ => Equiv.swap i k * p)
    · intro p hp
      simp only [Sk, Si, Finset.mem_filter, Finset.mem_univ, true_and] at hp ⊢
      have hi : (Equiv.swap i k * p).symm i = p.symm k := by simp [Equiv.Perm.mul_def]
      have hk : (Equiv.swap i k * p).symm k = p.symm i := by simp [Equiv.Perm.mul_def]
      have hj : (Equiv.swap i k * p).symm j = p.symm j := by
        simp [Equiv.Perm.mul_def, Equiv.swap_apply_of_ne_of_ne hji hjk]
      rw [hi, hk, hj]; exact ⟨hp.2, hp.1⟩
    · intro p₁ _ p₂ _ h; exact mul_left_cancel h
    · intro p hp
      refine ⟨Equiv.swap i k * p, ?_, by simp [← mul_assoc]⟩
      simp only [Sk, Si, Finset.mem_filter, Finset.mem_univ, true_and] at hp ⊢
      have hi : (Equiv.swap i k * p).symm i = p.symm k := by simp [Equiv.Perm.mul_def]
      have hk : (Equiv.swap i k * p).symm k = p.symm i := by simp [Equiv.Perm.mul_def]
      have hj : (Equiv.swap i k * p).symm j = p.symm j := by
        simp [Equiv.Perm.mul_def, Equiv.swap_apply_of_ne_of_ne hji hjk]
      rw [hi, hk, hj]; exact ⟨hp.2, hp.1⟩
  -- Swap j and k gives bijection Sk ≃ Sj
  have h2 : Sk.card = Sj.card := by
    apply Finset.card_bij (fun p _ => Equiv.swap j k * p)
    · intro p hp
      simp only [Sk, Sj, Finset.mem_filter, Finset.mem_univ, true_and] at hp ⊢
      have hj : (Equiv.swap j k * p).symm j = p.symm k := by simp [Equiv.Perm.mul_def]
      have hk : (Equiv.swap j k * p).symm k = p.symm j := by simp [Equiv.Perm.mul_def]
      have hi : (Equiv.swap j k * p).symm i = p.symm i := by
        simp [Equiv.Perm.mul_def, Equiv.swap_apply_of_ne_of_ne hij hik]
      rw [hi, hk, hj]; exact ⟨hp.2, hp.1⟩
    · intro p₁ _ p₂ _ h; exact mul_left_cancel h
    · intro p hp
      refine ⟨Equiv.swap j k * p, ?_, by simp [← mul_assoc]⟩
      simp only [Sk, Sj, Finset.mem_filter, Finset.mem_univ, true_and] at hp ⊢
      have hj : (Equiv.swap j k * p).symm j = p.symm k := by simp [Equiv.Perm.mul_def]
      have hk : (Equiv.swap j k * p).symm k = p.symm j := by simp [Equiv.Perm.mul_def]
      have hi : (Equiv.swap j k * p).symm i = p.symm i := by
        simp [Equiv.Perm.mul_def, Equiv.swap_apply_of_ne_of_ne hij hik]
      rw [hi, hk, hj]; exact ⟨hp.2, hp.1⟩
  have hd_ki : Disjoint Sk Si := by
    rw [Finset.disjoint_left]; intro p ha hb
    simp only [Sk, Si, Finset.mem_filter, Finset.mem_univ, true_and] at ha hb
    exact lt_asymm ha.1 hb.2
  have hd_kij : Disjoint (Sk ∪ Si) Sj := by
    rw [Finset.disjoint_left]; intro p hab hc
    simp only [Sk, Si, Sj, Finset.mem_union, Finset.mem_filter, Finset.mem_univ, true_and] at hab hc
    rcases hab with ha | hb
    · exact lt_asymm ha.2 hc.1
    · exact lt_asymm hb.1 hc.2
  have hunion : Sk ∪ Si ∪ Sj = Finset.univ := by
    ext p
    simp only [Sk, Si, Sj, Finset.mem_union, Finset.mem_filter, Finset.mem_univ, true_and, iff_true]
    have h12 : p.symm i ≠ p.symm j := fun h => hij (p.symm.injective h)
    have h13 : p.symm i ≠ p.symm k := fun h => hik (p.symm.injective h)
    have h23 : p.symm j ≠ p.symm k := fun h => hjk (p.symm.injective h)
    rcases lt_or_gt_of_ne h12 with hlt12 | hgt12
    · rcases lt_or_gt_of_ne h13 with hlt13 | hgt13
      · exact Or.inl (Or.inr ⟨hlt12, hlt13⟩)
      · exact Or.inl (Or.inl ⟨hgt13, lt_trans hgt13 hlt12⟩)
    · rcases lt_or_gt_of_ne h23 with hlt23 | hgt23
      · exact Or.inr ⟨hlt23, hgt12⟩
      · exact Or.inl (Or.inl ⟨lt_trans hgt23 hgt12, hgt23⟩)
  have htot := Finset.card_union_of_disjoint hd_kij
  rw [Finset.card_union_of_disjoint hd_ki, hunion, Finset.card_univ] at htot
  dsimp only [Sk] at h1 h2 htot
  omega

/-- **Exact combinatorial expectation (Theorem 1)**:
For every `n` and symmetric unit-diagonal `A` with `F = A - I`,
`E_p[Γ_p Γ_pᵀ] = A + (1/3) F² + (1/6) diag(F²)`. -/
theorem expected_gamma_gram {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)
    (hAsym : Aᵀ = A) (hAdiag : ∀ i, A i i = 1) :
    let F : Matrix (Fin n) (Fin n) ℝ := A - 1
    let N : ℝ := (Fintype.card (Equiv.Perm (Fin n)) : ℝ)
    (1 / N) • ∑ p : Equiv.Perm (Fin n), Gamma p A * (Gamma p A)ᵀ =
      A + (1 / 3 : ℝ) • (F * F) + (1 / 6 : ℝ) • Matrix.diagonal (fun i => (F * F) i i) := by
  intro F N
  have hN_pos : 0 < N := Nat.cast_pos.2 Fintype.card_pos
  have hN_ne : N ≠ 0 := ne_of_gt hN_pos
  have hF_diag : ∀ i, F i i = 0 := fun i => by simp [F, hAdiag i]
  have hF_off : ∀ i j, i ≠ j → F i j = A i j := fun i j hij => by simp [F, hij]
  have hF_sym : ∀ i j, F j i = F i j := by
    intro i j
    have hAji : A j i = A i j := by simpa using congrFun (congrFun hAsym i) j
    by_cases hij : i = j
    · subst hij; rfl
    · rw [hF_off j i (Ne.symm hij), hF_off i j hij, hAji]
  ext i j
  simp only [Matrix.smul_apply, Matrix.sum_apply, Matrix.add_apply, Matrix.mul_apply,
    Matrix.transpose_apply, Matrix.diagonal_apply, smul_eq_mul]
  rw [Finset.sum_comm]
  by_cases hij : i = j
  · subst hij
    simp only [↓reduceIte, hAdiag i]
    have hterm : ∀ k : Fin n,
        (∑ p : Equiv.Perm (Fin n), Gamma p A i k * Gamma p A i k) =
        if k = i then N else (N / 2) * (F i k * F k i) := by
      intro k
      by_cases hki : k = i
      · subst hki; simp [Gamma, N]
      · simp only [hki, ↓reduceIte]
        have hpt : ∀ p : Equiv.Perm (Fin n),
            Gamma p A i k * Gamma p A i k =
            if p.symm k < p.symm i then F i k * F k i else 0 := by
          intro p
          simp only [Gamma, Matrix.of_apply, Ne.symm hki, ↓reduceIte]
          split_ifs
          · rw [hF_off i k (Ne.symm hki), hF_sym i k, hF_off i k (Ne.symm hki)]
          · ring
        simp_rw [hpt, ← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul]
        have hcnt := card_perm_lt (Ne.symm hki)
        have hcntR : ((Finset.univ.filter (fun p : Equiv.Perm (Fin n) =>
            p.symm k < p.symm i)).card : ℝ) = N / 2 := by
          have h2 : 2 * ((Finset.univ.filter (fun p : Equiv.Perm (Fin n) =>
              p.symm k < p.symm i)).card : ℝ) = N := by dsimp only [N]; exact_mod_cast hcnt
          linarith
        rw [hcntR]
    simp_rw [hterm]
    have hsum : (∑ k : Fin n, if k = i then N else N / 2 * (F i k * F k i)) =
        N + (N / 2) * ∑ k : Fin n, F i k * F k i := by
      have hrew : ∀ k : Fin n,
          (if k = i then N else N / 2 * (F i k * F k i)) =
          (if k = i then N else 0) + (N / 2) * (F i k * F k i) := by
        intro k
        by_cases hki : k = i
        · subst hki; simp [hF_diag k]
        · simp [hki]
      simp_rw [hrew, Finset.sum_add_distrib, Finset.sum_ite_eq', ← Finset.mul_sum]
      simp
    rw [hsum]
    field_simp [hN_ne]
    ring
  · simp only [hij, ↓reduceIte, mul_zero, add_zero]
    have hterm : ∀ k : Fin n,
        (∑ p : Equiv.Perm (Fin n), Gamma p A i k * Gamma p A j k) =
        if k = i then (N / 2) * A i j
        else if k = j then (N / 2) * A i j
        else (N / 3) * (F i k * F k j) := by
      intro k
      by_cases hki : k = i
      · subst hki
        simp only [↓reduceIte]
        have hpt : ∀ p : Equiv.Perm (Fin n),
            Gamma p A k k * Gamma p A j k = if p.symm k < p.symm j then A k j else 0 := by
          intro p
          have hAjk : A j k = A k j := by simpa using congrFun (congrFun hAsym k) j
          simp only [Gamma, Matrix.of_apply, ↓reduceIte, Ne.symm hij, one_mul, hAjk]
        simp_rw [hpt, ← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul]
        have hcnt := card_perm_lt (Ne.symm hij)
        have hcntR : ((Finset.univ.filter (fun p : Equiv.Perm (Fin n) =>
            p.symm k < p.symm j)).card : ℝ) = N / 2 := by
          have h2 : 2 * ((Finset.univ.filter (fun p : Equiv.Perm (Fin n) =>
              p.symm k < p.symm j)).card : ℝ) = N := by dsimp only [N]; exact_mod_cast hcnt
          linarith
        rw [hcntR]
      · by_cases hkj : k = j
        · subst hkj
          simp only [hki, ↓reduceIte]
          have hpt : ∀ p : Equiv.Perm (Fin n),
              Gamma p A i k * Gamma p A k k = if p.symm k < p.symm i then A i k else 0 := by
            intro p
            simp only [Gamma, Matrix.of_apply, hij, ↓reduceIte, mul_one]
          simp_rw [hpt, ← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul]
          have hcnt := card_perm_lt hij
          have hcntR : ((Finset.univ.filter (fun p : Equiv.Perm (Fin n) =>
              p.symm k < p.symm i)).card : ℝ) = N / 2 := by
            have h2 : 2 * ((Finset.univ.filter (fun p : Equiv.Perm (Fin n) =>
                p.symm k < p.symm i)).card : ℝ) = N := by dsimp only [N]; exact_mod_cast hcnt
            linarith
          rw [hcntR]
        · simp only [hki, hkj, ↓reduceIte]
          have hpt : ∀ p : Equiv.Perm (Fin n),
              Gamma p A i k * Gamma p A j k =
              if p.symm k < p.symm i ∧ p.symm k < p.symm j then F i k * F k j else 0 := by
            intro p
            simp only [Gamma, Matrix.of_apply, Ne.symm hki, Ne.symm hkj, ↓reduceIte]
            by_cases h1 : p.symm k < p.symm i <;> by_cases h2 : p.symm k < p.symm j
            · simp only [h1, h2, and_self, ↓reduceIte]
              rw [hF_off i k (Ne.symm hki), ← hF_sym k j, hF_off j k (Ne.symm hkj)]
            · simp [h1, h2]
            · simp [h1, h2]
            · simp [h1, h2]
          simp_rw [hpt, ← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul]
          have hcnt := card_perm_lt_lt hij (Ne.symm hki) (Ne.symm hkj)
          have hcntR : ((Finset.univ.filter (fun p : Equiv.Perm (Fin n) =>
              p.symm k < p.symm i ∧ p.symm k < p.symm j)).card : ℝ) = N / 3 := by
            have h3 : 3 * ((Finset.univ.filter (fun p : Equiv.Perm (Fin n) =>
                p.symm k < p.symm i ∧ p.symm k < p.symm j)).card : ℝ) = N := by
              dsimp only [N]; exact_mod_cast hcnt
            linarith
          rw [hcntR]
    simp_rw [hterm]
    have hsum : (∑ k : Fin n,
        if k = i then N / 2 * A i j else if k = j then N / 2 * A i j else N / 3 * (F i k * F k j)) =
        N * A i j + (N / 3) * ∑ k : Fin n, F i k * F k j := by
      have hrew : ∀ k : Fin n,
          (if k = i then N / 2 * A i j else if k = j then N / 2 * A i j else N / 3 * (F i k * F k j)) =
          (if k = i then N / 2 * A i j else 0) + (if k = j then N / 2 * A i j else 0) +
          (N / 3) * (F i k * F k j) := by
        intro k
        by_cases hki : k = i
        · subst hki; simp [hij, hF_diag k]
        · by_cases hkj : k = j
          · subst hkj; simp [hki, hF_diag k]
          · simp [hki, hkj]
      simp_rw [hrew, Finset.sum_add_distrib, Finset.sum_ite_eq', ← Finset.mul_sum]
      simp; ring
    rw [hsum]
    field_simp [hN_ne]

/-! ### 3. All-Dimensional `λ_max(A) ≤ 2` Theorem (Theorem 1) -/

/-- Variational (operator-convexity) lower bound for the expected one-epoch energy drop:
for EVERY `x, z : Fin n → ℝ`,
`2 (A x)ᵀ z - zᵀ E_p[Γ_p Γ_pᵀ] z ≤ xᵀ A x - E_p[(M_p x)ᵀ A (M_p x)]`. -/
theorem variational_epoch_bound {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)
    (hAsym : Aᵀ = A) (hAdiag : ∀ i, A i i = 1) (x z : Fin n → ℝ) :
    let N : ℝ := (Fintype.card (Equiv.Perm (Fin n)) : ℝ)
    2 * dotProduct (A *ᵥ x) z -
      dotProduct z (((1 / N) • ∑ p : Equiv.Perm (Fin n), Gamma p A * (Gamma p A)ᵀ) *ᵥ z) ≤
      dotProduct x (A *ᵥ x) -
        (1 / N) * ∑ p : Equiv.Perm (Fin n), dotProduct (epochMap p A x) (A *ᵥ epochMap p A x) := by
  intro N
  have hN_pos : 0 < N := Nat.cast_pos.2 Fintype.card_pos
  have hpt : ∀ p : Equiv.Perm (Fin n),
      2 * dotProduct (A *ᵥ x) z - dotProduct z ((Gamma p A * (Gamma p A)ᵀ) *ᵥ z) ≤
      dotProduct x (A *ᵥ x) - dotProduct (epochMap p A x) (A *ᵥ epochMap p A x) := by
    intro p
    rw [rpcd_energy_identity p A hAsym hAdiag x]
    set u := x - epochMap p A x
    set y := (Gamma p A)ᵀ *ᵥ z
    have hnn : 0 ≤ dotProduct (u - y) (u - y) := Finset.sum_nonneg fun i _ => mul_self_nonneg _
    have hcross : dotProduct u y = dotProduct (A *ᵥ x) z := by
      dsimp only [y]
      rw [dotProduct_mulVec_transpose, rpcd_triangular_system p A hAdiag x]
    have hyy : dotProduct y y = dotProduct z ((Gamma p A * (Gamma p A)ᵀ) *ᵥ z) := by
      dsimp only [y]
      rw [← Matrix.mulVec_mulVec, dotProduct_mulVec_transpose, dotProduct_comm]
    have hexp : ∀ a b : Fin n → ℝ, dotProduct (a - b) (a - b) =
        dotProduct a a - 2 * dotProduct a b + dotProduct b b := by
      intro a b
      rw [sub_dotProduct, dotProduct_sub, dotProduct_sub, dotProduct_comm b a]; ring
    linarith [hexp u y]
  have hsum := Finset.sum_le_sum fun p (_ : p ∈ Finset.univ) => hpt p
  simp only [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul] at hsum
  have hmat : dotProduct z (((1 / N) • ∑ p : Equiv.Perm (Fin n), Gamma p A * (Gamma p A)ᵀ) *ᵥ z) =
      (1 / N) * ∑ p : Equiv.Perm (Fin n), dotProduct z ((Gamma p A * (Gamma p A)ᵀ) *ᵥ z) := by
    rw [Matrix.smul_mulVec, dotProduct_smul, smul_eq_mul]
    congr 1
    simp only [Matrix.sum_mulVec, dotProduct_sum]
  rw [hmat]
  have hmul := mul_le_mul_of_nonneg_left hsum (by positivity : 0 ≤ 1 / N)
  have hN_ne : N ≠ 0 := ne_of_gt hN_pos
  have h1 : (1 / N) * (N * (2 * dotProduct (A *ᵥ x) z) -
      ∑ p : Equiv.Perm (Fin n), dotProduct z ((Gamma p A * (Gamma p A)ᵀ) *ᵥ z)) =
      2 * dotProduct (A *ᵥ x) z -
      (1 / N) * ∑ p : Equiv.Perm (Fin n), dotProduct z ((Gamma p A * (Gamma p A)ᵀ) *ᵥ z) := by
    field_simp [hN_ne]
  have h2 : (1 / N) * (N * dotProduct x (A *ᵥ x) -
      ∑ p : Equiv.Perm (Fin n), dotProduct (epochMap p A x) (A *ᵥ epochMap p A x)) =
      dotProduct x (A *ᵥ x) -
      (1 / N) * ∑ p : Equiv.Perm (Fin n), dotProduct (epochMap p A x) (A *ᵥ epochMap p A x) := by
    field_simp [hN_ne]
  linarith

/-- Spectral bound for `J = (1/3) F² + (1/6) diag(F²)` when `s I ⪯ A ⪯ 2 I` (`0 < s ≤ 1`):
`xᵀ J x ≤ h · xᵀ A x` where `h = max ((1 - s)(3 - 2s) / (6s)) ((3 - s) / 12)`. -/
theorem spectral_J_le_h_A {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)
    (hAsym : Aᵀ = A) (hAdiag : ∀ i, A i i = 1)
    {s : ℝ} (hs0 : 0 < s) (hs1 : s ≤ 1)
    (hspec : ∀ x : Fin n → ℝ,
      s * dotProduct x x ≤ dotProduct x (A *ᵥ x) ∧
      dotProduct x (A *ᵥ x) ≤ 2 * dotProduct x x) :
    let F : Matrix (Fin n) (Fin n) ℝ := A - 1
    let J : Matrix (Fin n) (Fin n) ℝ :=
      (1 / 3 : ℝ) • (F * F) + (1 / 6 : ℝ) • Matrix.diagonal (fun i => (F * F) i i)
    let t : ℝ := 1 - s
    let h : ℝ := max (t * (3 - 2 * s) / (6 * s)) ((2 + t) / 12)
    ∀ x : Fin n → ℝ, dotProduct x (J *ᵥ x) ≤ h * dotProduct x (A *ᵥ x) := by
  intro F J t h x
  have hHerm : A.IsHermitian := by
    rw [Matrix.IsHermitian, Matrix.conjTranspose_eq_transpose_of_trivial]; exact hAsym
  -- Eigenvalues of A lie in [s, 2]
  let U : Matrix (Fin n) (Fin n) ℝ := ↑hHerm.eigenvectorUnitary
  let ev : Fin n → ℝ := hHerm.eigenvalues
  have hU_mem := hHerm.eigenvectorUnitary.prop
  have hUU : U * Uᵀ = 1 := by
    have := Matrix.mem_unitaryGroup_iff.1 hU_mem
    simpa [U, Matrix.star_eq_conjTranspose, Matrix.conjTranspose_eq_transpose_of_trivial] using this
  have hUtU : Uᵀ * U = 1 := by
    have := Matrix.mem_unitaryGroup_iff'.1 hU_mem
    simpa [U, Matrix.star_eq_conjTranspose, Matrix.conjTranspose_eq_transpose_of_trivial] using this
  have hspec_A : A = U * Matrix.diagonal ev * Uᵀ := by
    have hst := hHerm.spectral_theorem
    simpa [U, ev, Matrix.star_eq_conjTranspose, Matrix.conjTranspose_eq_transpose_of_trivial,
      Function.comp_def] using hst
  have hA_quad : ∀ v : Fin n → ℝ,
      dotProduct v (A *ᵥ v) = ∑ k : Fin n, ev k * ((Uᵀ *ᵥ v) k) ^ 2 := by
    intro v
    rw [hspec_A, ← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec]
    have htr : dotProduct v (U *ᵥ (Matrix.diagonal ev *ᵥ (Uᵀ *ᵥ v))) =
        dotProduct (Uᵀ *ᵥ v) (Matrix.diagonal ev *ᵥ (Uᵀ *ᵥ v)) := by
      have := dotProduct_mulVec_transpose Uᵀ v (Matrix.diagonal ev *ᵥ (Uᵀ *ᵥ v))
      simpa [Matrix.transpose_transpose] using this
    rw [htr]
    simp only [dotProduct, Matrix.mulVec_diagonal]
    apply Finset.sum_congr rfl; intro k _; ring
  have hv_norm : ∀ v : Fin n → ℝ,
      dotProduct v v = ∑ k : Fin n, ((Uᵀ *ᵥ v) k) ^ 2 := by
    intro v
    calc dotProduct v v = dotProduct v ((U * Uᵀ) *ᵥ v) := by rw [hUU, Matrix.one_mulVec]
      _ = dotProduct v (U *ᵥ (Uᵀ *ᵥ v)) := by rw [← Matrix.mulVec_mulVec]
      _ = dotProduct (Uᵀ *ᵥ v) (Uᵀ *ᵥ v) := by
          have := dotProduct_mulVec_transpose Uᵀ v (Uᵀ *ᵥ v)
          simpa [Matrix.transpose_transpose] using this
      _ = ∑ k : Fin n, ((Uᵀ *ᵥ v) k) ^ 2 := by
          unfold dotProduct; apply Finset.sum_congr rfl; intro k _; ring
  have hev_bounds : ∀ k : Fin n, s ≤ ev k ∧ ev k ≤ 2 := by
    intro k
    let v : Fin n → ℝ := U *ᵥ Pi.single k 1
    have hUtv : Uᵀ *ᵥ v = Pi.single k 1 := by
      dsimp only [v]; rw [Matrix.mulVec_mulVec, hUtU, Matrix.one_mulVec]
    have h1 := hspec v
    rw [hA_quad v, hv_norm v, hUtv] at h1
    have hsum1 : ∑ j : Fin n, ev j * ((Pi.single k (1 : ℝ) : Fin n → ℝ) j) ^ 2 = ev k := by
      simp [Pi.single_apply]
    have hsum2 : ∑ j : Fin n, ((Pi.single k (1 : ℝ) : Fin n → ℝ) j) ^ 2 = 1 := by
      simp [Pi.single_apply]
    rw [hsum1, hsum2, mul_one, mul_one] at h1
    exact h1
  -- Quadratic form of F² = (A - I)² = U diag((ev - 1)²) Uᵀ
  have hF2_eq : F * F = U * Matrix.diagonal (fun k => (ev k - 1) ^ 2) * Uᵀ := by
    have hF_eq : F = U * Matrix.diagonal (fun k => ev k - 1) * Uᵀ := by
      dsimp only [F]; rw [hspec_A]
      calc U * Matrix.diagonal ev * Uᵀ - 1
          = U * Matrix.diagonal ev * Uᵀ - U * 1 * Uᵀ := by rw [Matrix.mul_one, hUU]
        _ = U * (Matrix.diagonal ev - Matrix.diagonal (fun _ => (1 : ℝ))) * Uᵀ := by
            rw [Matrix.diagonal_one, Matrix.mul_sub, Matrix.sub_mul]
        _ = U * Matrix.diagonal (fun k => ev k - 1) * Uᵀ := by
            rw [← Matrix.diagonal_sub]
    rw [hF_eq]
    calc (U * Matrix.diagonal (fun k => ev k - 1) * Uᵀ) *
          (U * Matrix.diagonal (fun k => ev k - 1) * Uᵀ)
        = U * Matrix.diagonal (fun k => ev k - 1) * (Uᵀ * U) *
          Matrix.diagonal (fun k => ev k - 1) * Uᵀ := by simp only [Matrix.mul_assoc]
      _ = U * (Matrix.diagonal (fun k => ev k - 1) * Matrix.diagonal (fun k => ev k - 1)) * Uᵀ := by
          rw [hUtU, Matrix.mul_one]; simp only [Matrix.mul_assoc]
      _ = U * Matrix.diagonal (fun k => (ev k - 1) ^ 2) * Uᵀ := by
          rw [Matrix.diagonal_mul_diagonal,
            show (fun k => (ev k - 1) * (ev k - 1)) = (fun k => (ev k - 1) ^ 2) by funext k; ring]
  have hF2_quad : ∀ v : Fin n → ℝ,
      dotProduct v ((F * F) *ᵥ v) = ∑ k : Fin n, (ev k - 1) ^ 2 * ((Uᵀ *ᵥ v) k) ^ 2 := by
    intro v
    rw [hF2_eq, ← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec]
    have htr : dotProduct v (U *ᵥ (Matrix.diagonal (fun k => (ev k - 1) ^ 2) *ᵥ (Uᵀ *ᵥ v))) =
        dotProduct (Uᵀ *ᵥ v) (Matrix.diagonal (fun k => (ev k - 1) ^ 2) *ᵥ (Uᵀ *ᵥ v)) := by
      have := dotProduct_mulVec_transpose Uᵀ v (Matrix.diagonal (fun k => (ev k - 1) ^ 2) *ᵥ (Uᵀ *ᵥ v))
      simpa [Matrix.transpose_transpose] using this
    rw [htr]
    simp only [dotProduct, Matrix.mulVec_diagonal]
    apply Finset.sum_congr rfl; intro k _; ring
  -- Diagonal bound: (F²) i i ≤ t
  have hdiag_le : ∀ i : Fin n, (F * F) i i ≤ t := by
    intro i
    let e : Fin n → ℝ := Pi.single i 1
    have hlhs : dotProduct e ((F * F) *ᵥ e) = (F * F) i i := by
      simp [e, dotProduct, mulVec, Pi.single_apply]
    have hAii : dotProduct e (A *ᵥ e) = 1 := by
      simp [e, dotProduct, mulVec, Pi.single_apply, hAdiag i]
    have hee : dotProduct e e = 1 := by
      simp [e, dotProduct, Pi.single_apply]
    rw [← hlhs, hF2_quad e]
    have hpt : ∀ k : Fin n,
        (ev k - 1) ^ 2 * ((Uᵀ *ᵥ e) k) ^ 2 ≤
        ((1 - t) * (ev k - 1) + t) * ((Uᵀ *ᵥ e) k) ^ 2 := by
      intro k
      apply mul_le_mul_of_nonneg_right _ (sq_nonneg _)
      have ⟨hk1, hk2⟩ := hev_bounds k
      dsimp only [t]; nlinarith
    have hsum := Finset.sum_le_sum fun k (_ : k ∈ Finset.univ) => hpt k
    have hrhs : ∑ k : Fin n, ((1 - t) * (ev k - 1) + t) * ((Uᵀ *ᵥ e) k) ^ 2 =
        (1 - t) * (dotProduct e (A *ᵥ e) - dotProduct e e) + t * dotProduct e e := by
      rw [hA_quad e, hv_norm e, ← Finset.sum_sub_distrib, Finset.mul_sum, Finset.mul_sum,
        ← Finset.sum_add_distrib]
      apply Finset.sum_congr rfl; intro k _; ring
    rw [hrhs, hAii, hee] at hsum
    linarith
  -- Now bound xᵀ J x ≤ (1/3) xᵀ F² x + (t/6) xᵀ x ≤ h · xᵀ A x
  have hJ_le : dotProduct x (J *ᵥ x) ≤
      (1 / 3 : ℝ) * dotProduct x ((F * F) *ᵥ x) + (t / 6) * dotProduct x x := by
    dsimp only [J]
    rw [Matrix.add_mulVec, Matrix.smul_mulVec, Matrix.smul_mulVec,
      dotProduct_add, dotProduct_smul, dotProduct_smul, smul_eq_mul, smul_eq_mul]
    have hdiag_quad : dotProduct x (Matrix.diagonal (fun i => (F * F) i i) *ᵥ x) ≤
        t * dotProduct x x := by
      simp only [dotProduct, Matrix.mulVec_diagonal, Finset.mul_sum]
      apply Finset.sum_le_sum; intro i _
      have := hdiag_le i
      nlinarith [mul_self_nonneg (x i)]
    linarith
  refine le_trans hJ_le ?_
  rw [hF2_quad x, hv_norm x, hA_quad x, Finset.mul_sum, Finset.mul_sum, Finset.mul_sum,
    ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum; intro k _
  have hscalar : (1 / 3 : ℝ) * (ev k - 1) ^ 2 + t / 6 ≤ h * ev k := by
    have ⟨hk1, hk2⟩ := hev_bounds k
    have hs_val : (1 / 3 : ℝ) * (s - 1) ^ 2 + t / 6 ≤ h * s := by
      have hh1 : t * (3 - 2 * s) / (6 * s) ≤ h := le_max_left _ _
      have hmul := mul_le_mul_of_nonneg_right hh1 (le_of_lt hs0)
      have hrew : t * (3 - 2 * s) / (6 * s) * s = (1 / 3 : ℝ) * (s - 1) ^ 2 + t / 6 := by
        dsimp only [t]; field_simp [ne_of_gt hs0]; ring
      linarith
    have h2_val : (1 / 3 : ℝ) * (2 - 1) ^ 2 + t / 6 ≤ h * 2 := by
      have hh2 : (2 + t) / 12 ≤ h := le_max_right _ _
      linarith
    -- Convexity of q(z) = (1/3)(z-1)^2 + t/6 - h z on [s, 2] (positive leading coefficient 1/3 > 0!)
    have hquad_le : ∀ z : ℝ, s ≤ z → z ≤ 2 →
        (1 / 3 : ℝ) * (z - 1) ^ 2 + t / 6 - h * z ≤ 0 := by
      intro z hsz hz2
      have h2ms : 0 < 2 - s := by linarith
      have hid : (2 - s) * ((1 / 3 : ℝ) * (z - 1) ^ 2 + t / 6 - h * z) =
          (2 - z) * ((1 / 3 : ℝ) * (s - 1) ^ 2 + t / 6 - h * s) +
          (z - s) * ((1 / 3 : ℝ) * (2 - 1) ^ 2 + t / 6 - h * 2) -
          (1 / 3 : ℝ) * (2 - s) * (z - s) * (2 - z) := by ring
      have h1 : (2 - z) * ((1 / 3 : ℝ) * (s - 1) ^ 2 + t / 6 - h * s) ≤ 0 := by nlinarith
      have h2 : (z - s) * ((1 / 3 : ℝ) * (2 - 1) ^ 2 + t / 6 - h * 2) ≤ 0 := by nlinarith
      have h3 : 0 ≤ (1 / 3 : ℝ) * (2 - s) * (z - s) * (2 - z) := by
        have hzms : 0 ≤ z - s := by linarith
        have h2mz : 0 ≤ 2 - z := by linarith
        positivity
      nlinarith
    linarith [hquad_le (ev k) hk1 hk2]
  nlinarith [sq_nonneg ((Uᵀ *ᵥ x) k)]

/-- Monotonicity of `n ↦ (1 - a / n) ^ n` for `0 ≤ a ≤ 1`, `1 ≤ n` (via Bernoulli's inequality). -/
lemma one_sub_div_pow_mono {a : ℝ} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) {n : ℕ} (hn : 1 ≤ n) :
    (1 - a / (n : ℝ)) ^ n ≤ (1 - a / ((n + 1 : ℕ) : ℝ)) ^ (n + 1) := by
  have hn_pos : (0 : ℝ) < n := Nat.cast_pos.2 hn
  have hn1_pos : (0 : ℝ) < (n + 1 : ℕ) := Nat.cast_pos.2 (by omega)
  have hbase_nn : 0 ≤ 1 - a / (n : ℝ) := by
    rw [sub_nonneg, div_le_one hn_pos]; linarith [show (1 : ℝ) ≤ n by exact_mod_cast hn]
  have hbase1_pos : 0 < 1 - a / ((n + 1 : ℕ) : ℝ) := by
    rw [sub_pos, div_lt_one hn1_pos]
    have : (2 : ℝ) ≤ (n + 1 : ℕ) := by exact_mod_cast (show 2 ≤ n + 1 by omega)
    linarith
  by_cases hbase0 : 1 - a / (n : ℝ) = 0
  · rw [hbase0, zero_pow (by omega)]; exact pow_nonneg (le_of_lt hbase1_pos) _
  · have hbase_pos : 0 < 1 - a / (n : ℝ) := lt_of_le_of_ne hbase_nn (Ne.symm hbase0)
    -- Rewrite via ratio: ((1 - a/(n+1)) / (1 - a/n))^n ≥ 1 / (1 - a/(n+1)) using Bernoulli!
    -- Even cleaner: ((1 - a/(n+1))^(n+1)) / ((1 - a/n)^n)
    -- = (1 - a/(n+1)) * (1 + a / ((n + 1)*(n - a)))^n ≥ (1 - a/(n+1)) * (1 + n*a / ((n+1)*(n-a))) ≥ 1!
    let δ : ℝ := a / (((n + 1 : ℕ) : ℝ) * ((n : ℝ) - a))
    have hna_pos : 0 < (n : ℝ) - a := by
      have : (n : ℝ) * (1 - a / (n : ℝ)) = (n : ℝ) - a := by field_simp [ne_of_gt hn_pos]
      nlinarith
    have hδ0 : 0 ≤ δ := div_nonneg ha0 (mul_nonneg (le_of_lt hn1_pos) (le_of_lt hna_pos))
    have hratio : (1 - a / ((n + 1 : ℕ) : ℝ)) / (1 - a / (n : ℝ)) = 1 + δ := by
      dsimp only [δ]
      push_cast
      field_simp [ne_of_gt hn_pos, ne_of_gt hn1_pos, ne_of_gt hna_pos, ne_of_gt hbase_pos]
      ring
    have hbern : 1 + (n : ℝ) * δ ≤ (1 + δ) ^ n := one_add_mul_le_pow (by linarith) n
    have hprod : 1 ≤ (1 - a / ((n + 1 : ℕ) : ℝ)) * (1 + (n : ℝ) * δ) := by
      dsimp only [δ]
      push_cast
      have hden_pos : 0 < ((n : ℝ) + 1) * ((n : ℝ) - a) := by positivity
      have hdiff : (1 - a / ((n : ℝ) + 1)) * (1 + (n : ℝ) * (a / (((n : ℝ) + 1) * ((n : ℝ) - a)))) - 1 =
          a ^ 2 / (((n : ℝ) + 1) ^ 2 * ((n : ℝ) - a)) := by
        field_simp [ne_of_gt hn1_pos, ne_of_gt hna_pos]
        ring
      have hnn : 0 ≤ a ^ 2 / (((n : ℝ) + 1) ^ 2 * ((n : ℝ) - a)) := by
        apply div_nonneg (sq_nonneg a) (by positivity)
      linarith
    have hpow_ge : 1 ≤ (1 - a / ((n + 1 : ℕ) : ℝ)) * (1 + δ) ^ n := by
      calc 1 ≤ (1 - a / ((n + 1 : ℕ) : ℝ)) * (1 + (n : ℝ) * δ) := hprod
        _ ≤ (1 - a / ((n + 1 : ℕ) : ℝ)) * (1 + δ) ^ n :=
            mul_le_mul_of_nonneg_left hbern (le_of_lt hbase1_pos)
    rw [← hratio, div_pow] at hpow_ge
    have hden_pow : 0 < (1 - a / (n : ℝ)) ^ n := pow_pos hbase_pos n
    have hmul := mul_le_mul_of_nonneg_right hpow_ge (le_of_lt hden_pow)
    rw [one_mul, mul_assoc, div_mul_cancel₀ _ (ne_of_gt hden_pow), ← pow_succ'] at hmul
    exact hmul

/-- Scalar contraction chain for Theorem 1:
`h / (1 + h) ≤ max (1/4) ((1 - s)²) ≤ B_n(s)` for all `n ≥ 2`, `0 < s ≤ 1`. -/
theorem h_div_one_add_h_le_B {n : ℕ} (hn : 2 ≤ n) {s : ℝ} (hs0 : 0 < s) (hs1 : s ≤ 1) :
    let t : ℝ := 1 - s
    let h : ℝ := max (t * (3 - 2 * s) / (6 * s)) ((2 + t) / 12)
    h / (1 + h) ≤ max (1 / 4 : ℝ) ((1 - s) ^ 2) ∧
    max (1 / 4 : ℝ) ((1 - s) ^ 2) ≤ B n s := by
  intro t h
  have ht0 : 0 ≤ t := by dsimp only [t]; linarith
  have hh0 : 0 ≤ h := le_trans (by positivity : (0 : ℝ) ≤ (2 + t) / 12) (le_max_right _ _)
  constructor
  · have hmon : ∀ a b : ℝ, 0 ≤ a → a ≤ b → a / (1 + a) ≤ b / (1 + b) := by
      intro a b ha hab
      rw [div_le_div_iff₀ (by linarith) (by linarith)]
      nlinarith
    have h2 : ((2 + t) / 12) / (1 + (2 + t) / 12) ≤ 1 / 4 := by
      rw [div_le_div_iff₀ (by linarith) (by norm_num)]
      dsimp only [t]; linarith
    have hs_bd : (t * (3 - 2 * s) / (6 * s)) / (1 + t * (3 - 2 * s) / (6 * s)) ≤
        max (1 / 4 : ℝ) ((1 - s) ^ 2) := by
      by_cases hhalf : 1 / 2 ≤ s
      · refine le_trans ?_ (le_max_left _ _)
        have hle13 : t * (3 - 2 * s) / (6 * s) ≤ 1 / 3 := by
          rw [div_le_div_iff₀ (by linarith) (by norm_num)]
          dsimp only [t]; nlinarith
        calc (t * (3 - 2 * s) / (6 * s)) / (1 + t * (3 - 2 * s) / (6 * s))
            ≤ (1 / 3 : ℝ) / (1 + 1 / 3) := hmon _ _ (div_nonneg (by dsimp only [t]; nlinarith) (by linarith)) hle13
          _ = 1 / 4 := by norm_num
      · push_neg at hhalf
        refine le_trans ?_ (le_max_right _ _)
        have h32s : 0 ≤ 3 - 2 * s := by linarith
        have hden : 0 < 1 + t * (3 - 2 * s) / (6 * s) := by positivity
        rw [div_le_iff₀ hden]
        have hkey : t ^ 2 * (3 + s + 2 * s ^ 2) - t * (3 - 2 * s) = s ^ 2 * (1 - s) * (1 - 2 * s) := by
          dsimp only [t]; ring
        have hkey_nn : 0 ≤ s ^ 2 * (1 - s) * (1 - 2 * s) :=
          mul_nonneg (mul_nonneg (sq_nonneg s) (by linarith)) (by linarith)
        have hrew : (1 - s) ^ 2 * (1 + t * (3 - 2 * s) / (6 * s)) - t * (3 - 2 * s) / (6 * s) =
            (t ^ 2 * (3 + s + 2 * s ^ 2) - t * (3 - 2 * s)) / (6 * s) := by
          dsimp only [t]; field_simp [ne_of_gt hs0]; ring
        have hdiv_nn : 0 ≤ (t ^ 2 * (3 + s + 2 * s ^ 2) - t * (3 - 2 * s)) / (6 * s) :=
          div_nonneg (by linarith) (by linarith)
        linarith
    rcases max_choice (t * (3 - 2 * s) / (6 * s)) ((2 + t) / 12) with hEq | hEq
    · dsimp only [h]; rw [hEq]; exact hs_bd
    · dsimp only [h]; rw [hEq]; exact le_trans h2 (le_max_left _ _)
  · unfold B
    apply max_le
    · refine le_trans ?_ (le_max_left _ _)
      have h2 : (1 / 4 : ℝ) = (1 - 1 / (2 : ℝ)) ^ (2 : ℕ) := by norm_num
      rw [h2]
      clear h2
      induction n, hn using Nat.le_induction with
      | base => norm_num
      | succ k hk ih =>
        exact le_trans ih (one_sub_div_pow_mono (by norm_num) (by norm_num) (by omega))
    · refine le_trans ?_ (le_max_right _ _)
      have h1 : 1 - s ≤ (1 - s / (n : ℝ)) ^ n := by
        have hbern := one_add_mul_le_pow (a := -s / (n : ℝ)) (by
          have hnR : (2 : ℝ) ≤ n := by exact_mod_cast hn
          have hsn : s / (n : ℝ) ≤ 1 := by rw [div_le_one (by linarith)]; linarith
          rw [neg_div]; linarith) n
        have hrew1 : 1 + (n : ℝ) * (-s / (n : ℝ)) = 1 - s := by
          have hn_ne : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.2 (by omega)
          field_simp [hn_ne]; ring
        have hrew2 : 1 + -s / (n : ℝ) = 1 - s / (n : ℝ) := by ring
        rw [hrew1, hrew2] at hbern
        exact hbern
      have hsq : (1 - s) ^ 2 ≤ ((1 - s / (n : ℝ)) ^ n) ^ 2 :=
        pow_le_pow_left₀ (by linarith) h1 2
      rw [← pow_mul, mul_comm n 2] at hsq
      exact hsq

/-- **Theorem 1 (All dimensions `n ≥ 2` with `λ_max(A) ≤ 2`)**:
For every `n ≥ 2`, every symmetric unit-diagonal `A` satisfying `s I ⪯ A ⪯ 2 I` (`0 < s ≤ 1`),
and every `x ∈ ℝ^n`, the single-epoch expected RPCD energy contracts by at most `B_n(s)`:
`E_p[(M_p x)ᵀ A (M_p x)] ≤ B_n(s) · xᵀ A x`. -/
theorem all_dim_lambda_max_le_two {n : ℕ} (hn : 2 ≤ n)
    (A : Matrix (Fin n) (Fin n) ℝ) (hAsym : Aᵀ = A) (hAdiag : ∀ i, A i i = 1)
    {s : ℝ} (hs0 : 0 < s) (hs1 : s ≤ 1)
    (hspec : ∀ x : Fin n → ℝ,
      s * dotProduct x x ≤ dotProduct x (A *ᵥ x) ∧
      dotProduct x (A *ᵥ x) ≤ 2 * dotProduct x x)
    (x : Fin n → ℝ) :
    let N : ℝ := (Fintype.card (Equiv.Perm (Fin n)) : ℝ)
    (1 / N) * ∑ p : Equiv.Perm (Fin n), dotProduct (epochMap p A x) (A *ᵥ epochMap p A x) ≤
      B n s * dotProduct x (A *ᵥ x) := by
  intro N
  set F : Matrix (Fin n) (Fin n) ℝ := A - 1
  set J : Matrix (Fin n) (Fin n) ℝ :=
    (1 / 3 : ℝ) • (F * F) + (1 / 6 : ℝ) • Matrix.diagonal (fun i => (F * F) i i)
  set t : ℝ := 1 - s
  set h : ℝ := max (t * (3 - 2 * s) / (6 * s)) ((2 + t) / 12)
  have hh0 : 0 ≤ h := le_trans (by dsimp only [t]; linarith : (0 : ℝ) ≤ (2 + t) / 12) (le_max_right _ _)
  have h1h_pos : 0 < 1 + h := by linarith
  have hvar := variational_epoch_bound A hAsym hAdiag x ((1 + h)⁻¹ • x)
  have hexp := expected_gamma_gram A hAsym hAdiag
  have hJ := spectral_J_le_h_A A hAsym hAdiag hs0 hs1 hspec x
  -- Evaluate the LHS of hvar using E[Γ Γᵀ] = A + J and xᵀ J x ≤ h xᵀ A x
  have hAJ_eq : (1 / N) • ∑ p : Equiv.Perm (Fin n), Gamma p A * (Gamma p A)ᵀ = A + J := by
    simpa [N, F, J, add_assoc] using hexp
  dsimp only at hvar
  change 2 * dotProduct (A *ᵥ x) ((1 + h)⁻¹ • x) -
      dotProduct ((1 + h)⁻¹ • x) (((1 / N) • ∑ p : Equiv.Perm (Fin n), Gamma p A * (Gamma p A)ᵀ) *ᵥ ((1 + h)⁻¹ • x)) ≤
      dotProduct x (A *ᵥ x) -
        (1 / N) * ∑ p : Equiv.Perm (Fin n), dotProduct (epochMap p A x) (A *ᵥ epochMap p A x) at hvar
  rw [hAJ_eq] at hvar
  have h1 : dotProduct (A *ᵥ x) ((1 + h)⁻¹ • x) = (1 + h)⁻¹ * dotProduct x (A *ᵥ x) := by
    rw [dotProduct_smul, smul_eq_mul, dotProduct_comm]
  have h2 : dotProduct ((1 + h)⁻¹ • x) ((A + J) *ᵥ ((1 + h)⁻¹ • x)) ≤
      (1 + h)⁻¹ * dotProduct x (A *ᵥ x) := by
    rw [Matrix.mulVec_smul, smul_dotProduct, dotProduct_smul, smul_eq_mul, smul_eq_mul,
      Matrix.add_mulVec, dotProduct_add]
    have hsum_le : dotProduct x (A *ᵥ x) + dotProduct x (J *ᵥ x) ≤ (1 + h) * dotProduct x (A *ᵥ x) := by
      linarith
    have hmul := mul_le_mul_of_nonneg_left hsum_le (mul_nonneg (inv_nonneg.2 (le_of_lt h1h_pos))
      (inv_nonneg.2 (le_of_lt h1h_pos)))
    have hrew : (1 + h)⁻¹ * ((1 + h)⁻¹ * ((1 + h) * dotProduct x (A *ᵥ x))) =
        (1 + h)⁻¹ * dotProduct x (A *ᵥ x) := by
      field_simp [ne_of_gt h1h_pos]
    linarith
  have hdrop : (1 + h)⁻¹ * dotProduct x (A *ᵥ x) ≤
      dotProduct x (A *ᵥ x) -
        (1 / N) * ∑ p : Equiv.Perm (Fin n), dotProduct (epochMap p A x) (A *ᵥ epochMap p A x) := by
    linarith
  have hstep : (1 / N) * ∑ p : Equiv.Perm (Fin n), dotProduct (epochMap p A x) (A *ᵥ epochMap p A x) ≤
      (h / (1 + h)) * dotProduct x (A *ᵥ x) := by
    have hrew : dotProduct x (A *ᵥ x) - (1 + h)⁻¹ * dotProduct x (A *ᵥ x) =
        (h / (1 + h)) * dotProduct x (A *ᵥ x) := by
      field_simp [ne_of_gt h1h_pos]; ring
    linarith
  have ⟨hb1, hb2⟩ := h_div_one_add_h_le_B hn hs0 hs1
  have hxAx_nn : 0 ≤ dotProduct x (A *ᵥ x) :=
    le_trans (mul_nonneg (le_of_lt hs0) (Finset.sum_nonneg fun i _ => mul_self_nonneg _)) (hspec x).1
  exact le_trans hstep (mul_le_mul_of_nonneg_right (le_trans hb1 hb2) hxAx_nn)

/-! ### 4. Dimension `n = 3` Contraction Bound and Cubic Criterion (Theorem 2) -/

/-- `(1 - s/2)² ≤ (1 - s/3)³` on `[0, 1]` via `(1 - s/3)³ - (1 - s/2)² = s²(1/12 - s/27) ≥ 0`. -/
theorem dim3_sq_le_cube {s : ℝ} (hs1 : s ≤ 1) :
    (1 - s / 2) ^ 2 ≤ (1 - s / 3) ^ 3 := by
  have hid : (1 - s / 3) ^ 3 - (1 - s / 2) ^ 2 = s ^ 2 * (1 / 12 - s / 27) := by ring
  have hnn : 0 ≤ s ^ 2 * (1 / 12 - s / 27) := mul_nonneg (sq_nonneg s) (by linarith)
  linarith

/-- **Theorem 2 scalar chain**: `(1 - s/2)⁴ ≤ (1 - s/3)⁶ ≤ B_3(s)` for all `s ∈ [0, 1]`. -/
theorem dim3_contraction_le_B3 {s : ℝ} (hs1 : s ≤ 1) :
    (1 - s / 2) ^ 4 ≤ (1 - s / 3) ^ 6 ∧ (1 - s / 3) ^ 6 ≤ B 3 s := by
  constructor
  · have h1 := dim3_sq_le_cube hs1
    have h2 := pow_le_pow_left₀ (sq_nonneg (1 - s / 2)) h1 2
    linarith [show ((1 - s / 2) ^ 2) ^ 2 = (1 - s / 2) ^ 4 by ring,
              show ((1 - s / 3) ^ 3) ^ 2 = (1 - s / 3) ^ 6 by ring]
  · unfold B
    simp only [Nat.cast_ofNat, show 2 * 3 = 6 by norm_num]
    exact le_max_right _ _

/-- Cubic eigenvalue positivity criterion (Theorem 2):
if `e₁, e₂, e₃ ≥ 0`, then no real root of `λ³ - e₁ λ² + e₂ λ - e₃ = 0` is negative. -/
theorem cubic_no_neg_root {e₁ e₂ e₃ lam : ℝ}
    (he1 : 0 ≤ e₁) (he2 : 0 ≤ e₂) (he3 : 0 ≤ e₃)
    (hroot : lam ^ 3 - e₁ * lam ^ 2 + e₂ * lam - e₃ = 0) :
    0 ≤ lam := by
  by_contra hneg
  push_neg at hneg
  set z := -lam
  have hz : 0 < z := by linarith
  have hval : z ^ 3 + e₁ * z ^ 2 + e₂ * z + e₃ = 0 := by
    dsimp only [z]; linarith
  have hpos : 0 < z ^ 3 + e₁ * z ^ 2 + e₂ * z + e₃ := by
    have h1 : 0 < z ^ 3 := pow_pos hz 3
    have h2 : 0 ≤ e₁ * z ^ 2 := mul_nonneg he1 (sq_nonneg z)
    have h3 : 0 ≤ e₂ * z := mul_nonneg he2 (le_of_lt hz)
    linarith
  linarith

/-! ### 5. Weighted-`ℓ_∞` Contraction (Proposition 3) and Direct-Sum Closure (Proposition 4) -/

/-- Entrywise comparison matrix `C = |I - A|` for a unit-diagonal matrix `A`. -/
def compMat {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  Matrix.of fun i j => if i = j then 0 else |A i j|

/-- Single coordinate update contracts coordinate `i` to `r α w_i` while preserving `α w_j`
on all other coordinates. -/
theorem weighted_linf_step {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) (hAdiag : ∀ i, A i i = 1)
    (w : Fin n → ℝ) {r α : ℝ} (hα : 0 ≤ α)
    (hCw : ∀ i, (compMat A *ᵥ w) i ≤ r * w i)
    (i : Fin n) (x : Fin n → ℝ) (hx : ∀ j, |x j| ≤ α * w j) :
    |coordUpdate A i x i| ≤ r * α * w i ∧
    ∀ j, j ≠ i → coordUpdate A i x j = x j := by
  constructor
  · have hupd : coordUpdate A i x i = - ∑ j : Fin n, (if j = i then 0 else A i j * x j) := by
      unfold coordUpdate
      simp only [↓reduceIte, mulVec, dotProduct]
      have hrew : ∀ j : Fin n, A i j * x j =
          (if i = j then x i else 0) + (if j = i then 0 else A i j * x j) := by
        intro j
        by_cases hji : j = i
        · subst hji; simp [hAdiag j]
        · simp [hji, Ne.symm hji]
      rw [Finset.sum_congr rfl (fun j _ => hrew j), Finset.sum_add_distrib, Finset.sum_ite_eq]
      simp
    rw [hupd, abs_neg]
    calc |∑ j : Fin n, if j = i then 0 else A i j * x j|
        ≤ ∑ j : Fin n, |if j = i then 0 else A i j * x j| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ j : Fin n, compMat A i j * (α * w j) := by
          apply Finset.sum_le_sum; intro j _
          by_cases hji : j = i
          · subst hji; simp [compMat]
          · simp only [hji, ↓reduceIte, abs_mul, compMat, Matrix.of_apply, Ne.symm hji]
            exact mul_le_mul_of_nonneg_left (hx j) (abs_nonneg _)
      _ = α * (compMat A *ᵥ w) i := by
          simp only [mulVec, dotProduct, Finset.mul_sum]
          apply Finset.sum_congr rfl; intro j _; ring
      _ ≤ α * (r * w i) := mul_le_mul_of_nonneg_left (hCw i) hα
      _ = r * α * w i := by ring
  · intro j hji; simp [coordUpdate, hji]

/-- **Proposition 3 (One-epoch deterministic weighted-`ℓ_∞` contraction)**:
If `|I - A| w ≤ r w` with `0 ≤ r ≤ 1` and `∀ j, |x j| ≤ α w j`, then for EVERY permutation `p`,
after one full RPCD epoch EVERY coordinate satisfies `|(M_p x) i| ≤ r α w i`. -/
theorem weighted_linf_epoch {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) (hAdiag : ∀ i, A i i = 1)
    (w : Fin n → ℝ) (hw : ∀ i, 0 ≤ w i) {r α : ℝ} (hr1 : r ≤ 1) (hα : 0 ≤ α)
    (hCw : ∀ i, (compMat A *ᵥ w) i ≤ r * w i)
    (p : Equiv.Perm (Fin n)) (x : Fin n → ℝ) (hx : ∀ j, |x j| ≤ α * w j) :
    ∀ i : Fin n, |epochMap p A x i| ≤ r * α * w i := by
  have hinv : ∀ k, k ≤ n →
      (∀ j : Fin n, (p.symm j).val < k → |partialEpoch p A k x j| ≤ r * α * w j) ∧
      (∀ j : Fin n, |partialEpoch p A k x j| ≤ α * w j) := by
    intro k
    induction k with
    | zero =>
      intro _
      exact ⟨fun j hj => absurd hj (Nat.not_lt_zero _), hx⟩
    | succ k ih =>
      intro hkn
      have ⟨h1, h2⟩ := ih (by omega)
      have hklt : k < n := by omega
      have hstep := weighted_linf_step A hAdiag w hα hCw (p ⟨k, hklt⟩) (partialEpoch p A k x) h2
      constructor
      · intro j hj
        unfold partialEpoch
        simp only [hklt, ↓reduceDIte]
        by_cases heq : j = p ⟨k, hklt⟩
        · subst heq; exact hstep.1
        · rw [hstep.2 j heq]
          have hne_val : (p.symm j).val ≠ k := by
            intro hv
            apply heq
            have : p.symm j = ⟨k, hklt⟩ := Fin.ext hv
            simpa using congrArg p this
          exact h1 j (by omega)
      · intro j
        unfold partialEpoch
        simp only [hklt, ↓reduceDIte]
        by_cases heq : j = p ⟨k, hklt⟩
        · subst heq
          calc |coordUpdate A (p ⟨k, hklt⟩) (partialEpoch p A k x) (p ⟨k, hklt⟩)|
              ≤ r * α * w (p ⟨k, hklt⟩) := hstep.1
            _ ≤ 1 * α * w (p ⟨k, hklt⟩) :=
                mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right hr1 hα) (hw (p ⟨k, hklt⟩))
            _ = α * w (p ⟨k, hklt⟩) := by ring
        · rw [hstep.2 j heq]; exact h2 j
  intro i
  exact (hinv n le_rfl).1 i (p.symm i).isLt

/-- Iterate RPCD over a deterministic sequence of `K` permutations `ps`. -/
def iterEpoch {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) :
    ∀ {K : ℕ}, (Fin K → Equiv.Perm (Fin n)) → (Fin n → ℝ) → (Fin n → ℝ)
  | 0, _, x => x
  | _ + 1, ps, x => epochMap (ps 0) A (iterEpoch A (fun i => ps i.succ) x)

/-- **Proposition 3 (`K`-epoch squared-Euclidean-norm contraction `≤ r^{2K} α² ‖w‖₂²`)**. -/
theorem weighted_linf_iter {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) (hAdiag : ∀ i, A i i = 1)
    (w : Fin n → ℝ) (hw : ∀ i, 0 ≤ w i) {r α : ℝ} (hr0 : 0 ≤ r) (hr1 : r ≤ 1) (hα : 0 ≤ α)
    (hCw : ∀ i, (compMat A *ᵥ w) i ≤ r * w i)
    {K : ℕ} (ps : Fin K → Equiv.Perm (Fin n)) (x : Fin n → ℝ) (hx : ∀ j, |x j| ≤ α * w j) :
    (∀ i : Fin n, |iterEpoch A ps x i| ≤ r ^ K * α * w i) ∧
    dotProduct (iterEpoch A ps x) (iterEpoch A ps x) ≤ r ^ (2 * K) * α ^ 2 * dotProduct w w := by
  have hcoord : ∀ {K : ℕ} (ps : Fin K → Equiv.Perm (Fin n)),
      ∀ i : Fin n, |iterEpoch A ps x i| ≤ r ^ K * α * w i := by
    intro K
    induction K with
    | zero => intro ps i; simpa [iterEpoch] using hx i
    | succ K ih =>
      intro ps i
      have hprev := ih (fun j => ps j.succ)
      have h1 := weighted_linf_epoch A hAdiag w hw hr1 (mul_nonneg (pow_nonneg hr0 K) hα)
        hCw (ps 0) (iterEpoch A (fun j => ps j.succ) x) hprev i
      simpa [iterEpoch, pow_succ', mul_assoc] using h1
  refine ⟨hcoord ps, ?_⟩
  unfold dotProduct
  rw [Finset.mul_sum]
  apply Finset.sum_le_sum; intro i _
  have hi := hcoord ps i
  have hsq : |iterEpoch A ps x i| ^ 2 ≤ (r ^ K * α * w i) ^ 2 :=
    pow_le_pow_left₀ (abs_nonneg _) hi 2
  rw [sq_abs] at hsq
  have hlhs : iterEpoch A ps x i * iterEpoch A ps x i = iterEpoch A ps x i ^ 2 := by ring
  have hrhs : (r ^ K * α * w i) ^ 2 = r ^ (2 * K) * α ^ 2 * (w i * w i) := by
    rw [mul_pow, mul_pow, ← pow_mul, mul_comm K 2]; ring
  linarith

/-- Monotonicity of `B_n(s)` in dimension `n ≥ 1` (Proposition 4). -/
theorem B_mono_n {n m : ℕ} (hn : 1 ≤ n) (hnm : n ≤ m) {s : ℝ} (hs0 : 0 ≤ s) (hs1 : s ≤ 1) :
    B n s ≤ B m s := by
  induction m, hnm using Nat.le_induction with
  | base => exact le_rfl
  | succ k hk ih =>
    refine le_trans ih ?_
    unfold B
    have h1 := one_sub_div_pow_mono (a := 1) (by norm_num) (by norm_num) (le_trans hn hk)
    have h2_half := one_sub_div_pow_mono hs0 hs1 (le_trans hn hk)
    have h2 : (1 - s / (k : ℝ)) ^ (2 * k) ≤ (1 - s / ((k + 1 : ℕ) : ℝ)) ^ (2 * (k + 1)) := by
      have hnn : 0 ≤ (1 - s / (k : ℝ)) ^ k := by
        apply pow_nonneg
        rw [sub_nonneg, div_le_one (Nat.cast_pos.2 (by omega))]
        exact le_trans hs1 (Nat.one_le_cast.2 (by omega))
      have hsq := pow_le_pow_left₀ hnn h2_half 2
      simpa [← pow_mul, mul_comm] using hsq
    exact max_le_max h1 h2

/-- Antitonicity of `B_n(s)` in `s ∈ [0, 1]` (Proposition 4). -/
theorem B_anti_s {n : ℕ} (hn : 1 ≤ n) {s₁ s₂ : ℝ} (hle : s₁ ≤ s₂) (hs2 : s₂ ≤ 1) :
    B n s₂ ≤ B n s₁ := by
  unfold B
  apply max_le_max le_rfl
  have hn_pos : (0 : ℝ) < n := Nat.cast_pos.2 hn
  have hbase_nn : 0 ≤ 1 - s₂ / (n : ℝ) := by
    rw [sub_nonneg, div_le_one hn_pos]
    exact le_trans hs2 (Nat.one_le_cast.2 hn)
  have hbase_le : 1 - s₂ / (n : ℝ) ≤ 1 - s₁ / (n : ℝ) := by
    have := div_le_div_of_nonneg_right hle (le_of_lt hn_pos)
    linarith
  exact pow_le_pow_left₀ hbase_nn hbase_le (2 * n)

/-- **Proposition 4 (Direct-sum closure)**:
If a direct sum `A = ⊕_j A_j` of total size `N = ∑_j n_j` (`n_j ≥ 1`) has block minimum
eigenvalues `s_j ∈ [s, 1]` and block contraction rates `ρ_j ≤ B_{n_j}(s_j)`, then every
block rate satisfies `ρ_j ≤ B_N(s)` (so `max_j ρ_j ≤ B_N(s)`). -/
theorem direct_sum_closure {m : ℕ} (sizes : Fin m → ℕ) (hsizes : ∀ j, 1 ≤ sizes j)
    (blockS blockRho : Fin m → ℝ) {s : ℝ} (hs0 : 0 ≤ s)
    (hS : ∀ j, s ≤ blockS j ∧ blockS j ≤ 1)
    (hRho : ∀ j, blockRho j ≤ B (sizes j) (blockS j)) :
    ∀ j : Fin m, blockRho j ≤ B (∑ l : Fin m, sizes l) s := by
  intro j
  have hle_sum : sizes j ≤ ∑ l : Fin m, sizes l :=
    Finset.single_le_sum (fun l _ => Nat.zero_le (sizes l)) (Finset.mem_univ j)
  calc blockRho j ≤ B (sizes j) (blockS j) := hRho j
    _ ≤ B (sizes j) s := B_anti_s (hsizes j) (hS j).1 (hS j).2
    _ ≤ B (∑ l : Fin m, sizes l) s := B_mono_n (hsizes j) hle_sum hs0 (le_trans (hS j).1 (hS j).2)

end RPCDContraction
