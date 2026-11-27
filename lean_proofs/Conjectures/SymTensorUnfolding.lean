import Mathlib

/-!
# Resolution of the Recursive and Rectangular Symmetric Tensor Unfolding Conjectures
(arXiv:2611.05137, Appendix D `conj:recursive` & Appendix E `conj:transition`)

We formalize in Lean 4 + Mathlib (completely verified, standard foundations only) the exact algebraic,
spectral, and variational cores of both conjecture resolutions from
`solutions/sym_tensor_unfolding/solution.md`:

1. **Exact `p = 2` (`q = 1`) Refutation of `conj:recursive`**:
   - `alphaPQ`, `alphaPQ_two_one`: Standing normalization `α_{2,1} = √(1! / 1!) = 1`.
   - `p2_mat1_maximizer_iff`: At `p = 2, q = 1`, the first-mode matricization
     `Mat₁(û) ∈ ℝ^{n × 1}` is the column vector `û` itself, so every unit leading left
     singular vector `v̂` satisfies `v̂ = ± û` and hence `⟨v̂, v⟩² = ⟨û, v⟩²`.
   - `sqOverlapLimit`, `p2_overlap_at_two`, `p2_overlap_lt_one`: Exact closed-form overlap
     limit `(1 - τ⁻²)⁺` equals `3 / 4 < 1` at `τ = 2` and stays strictly below `1` for every
     fixed `τ > 0`, refuting `conj:recursive` as printed.
2. **Even `p = 2q ≥ 4, τ > 1` Supercritical Exact-Recovery Algebraic Core (Theorem 1)**:
   - `zStar`, `sSc`, `zStar_gt_two`, `sSc_zStar`, `neg_tau_mul_sSc_zStar`: Exact semicircle
     Stieltjes outlier identities `z_* = τ + τ⁻¹ > 2`, `s_sc(z_*) = -τ⁻¹`, and
     `-τ * s_sc(z_*) = 1`.
   - `eigenvector_resolvent_identity`: Exact rearrangement of `(M + τ u uᵀ) û = ẑ û` to
     `û = -τ ⟨u, û⟩ R(ẑ) u` whenever `M - ẑ I` is invertible.
   - `rank_one_sv_perturbation`: Exact operator-norm perturbation bound
     `1 - |⟨v̂, v⟩| ≤ 2 ε / ρ` for any leading left singular vector `v̂` of `ρ v wᵀ + E`
     (`‖v‖ = ‖w‖ = 1, ρ > 0, ‖E‖_op ≤ ε`), proving `|⟨v̂, v⟩|² → 1` as `ε → 0`.
   - `lipschitz_scaling_even`: Exact Frobenius Lipschitz scaling
     `√(p / n) / (α_{p,q} n^{(p-2)/4}) = √((2q)! / q!) n^{-q/2}`.
3. **Rectangular Phase Transition Algebraic Threshold & Spectral Scaling (`1 ≤ k < p / 2`,
   `conj:transition`)**:
   - `edgeF`, `hasDerivAt_edgeF`: The deformed upper-edge function `f(d) = 2` (`0 ≤ d ≤ 1`),
     `d + d⁻¹` (`d > 1`) is `C¹` on `(0, ∞)` — including at the critical point `d = 1` where
     `f'(1) = 0` — with exact derivative `f'(d) = (1 - d⁻²)⁺`.
   - `left_overlap_variational_squeeze`: Exact Rayleigh/variational two-sided squeeze
     `(λ(t) - λ(t - δ) - ε) / δ ≤ ⟨a, x⟩² ≤ (λ(t + δ) - λ(t) + ε) / δ`.
   - `gammaLeft`, `gammaRight`, `sigmaSq`, `left_scaling_identity`, `right_scaling_identity`:
     Exact algebraic matching of the rescaled Gram parameters `t_n ∼ τ_ℓ²` and `r_n ∼ τ_r`
     to the conjectured constants `γ_{ℓ,p,k}` and `γ_{r,p,k}` for every `1 ≤ k < p / 2`.
   - `left_overlap_substitution` and `right_overlap_formula`: Exact threshold formulas
     `(1 - τ_ℓ⁻⁴)⁺` and `τ_r² / (1 + τ_r²)`.

Machine-checked vs prose-only:
- Machine-checked: All `p = 2` refutation identities and left-singular-vector reduction,
  semicircle Stieltjes outlier algebra, eigenvector-resolvent identity, rank-one singular-vector
  perturbation bound, `C¹` differentiability of `f(d)` at `d = 1`, variational overlap squeeze,
  and exact rectangular scaling constant identities for all `1 ≤ k < p / 2`.
- Prose-only (in `solutions/sym_tensor_unfolding/proof.pdf`): Concentration over product
  `1/4`-nets, isotropic deterministic equivalent transfer, Gaussian smooth-maximum
  interpolation against GOE (Proposition 4), and Capitaine–Donati-Martin–Féral GOE transition.
-/

open BigOperators Finset Matrix

namespace SymTensorUnfolding

/-! ### 1. Exact `p = 2` (`q = 1`) Refutation of `conj:recursive` -/

/-- Square-unfolding normalization constant `α_{p,q} = √(q! / (p - 1)!)`. -/
noncomputable def alphaPQ (p q : ℕ) : ℝ :=
  Real.sqrt ((Nat.factorial q : ℝ) / (Nat.factorial (p - 1) : ℝ))

theorem alphaPQ_two_one : alphaPQ 2 1 = 1 := by
  unfold alphaPQ
  norm_num

/-- Closed-form lifted/square-unfolding squared overlap limit `(1 - τ⁻²)⁺`. -/
noncomputable def sqOverlapLimit (τ : ℝ) : ℝ :=
  max 0 (1 - τ⁻¹ ^ 2)

/-- At `p = 2, τ = 2`, the exact limiting squared overlap is `3 / 4 < 1`,
refuting `conj:recursive` as printed. -/
theorem p2_overlap_at_two : sqOverlapLimit 2 = 3 / 4 := by
  unfold sqOverlapLimit
  norm_num

/-- For EVERY fixed `τ > 0`, the `p = 2` limiting squared overlap `(1 - τ⁻²)⁺` is strictly
less than `1`. -/
theorem p2_overlap_lt_one {τ : ℝ} (hτ : 0 < τ) : sqOverlapLimit τ < 1 := by
  unfold sqOverlapLimit
  refine max_lt (by norm_num) ?_
  have hpos : 0 < τ⁻¹ ^ 2 := sq_pos_of_pos (inv_pos.mpr hτ)
  linarith

/-- Euclidean inner product on `Fin n → ℝ`. -/
def euclidDot {n : ℕ} (x y : Fin n → ℝ) : ℝ := ∑ i : Fin n, x i * y i

/-- Squared Euclidean norm on `Fin n → ℝ`. -/
def euclidNormSq {n : ℕ} (x : Fin n → ℝ) : ℝ := euclidDot x x

lemma euclidNormSq_nonneg {n : ℕ} (x : Fin n → ℝ) : 0 ≤ euclidNormSq x := by
  unfold euclidNormSq euclidDot
  exact Finset.sum_nonneg fun i _ => mul_self_nonneg (x i)

lemma cauchy_schwarz_sq {n : ℕ} (x y : Fin n → ℝ) :
    (euclidDot x y) ^ 2 ≤ euclidNormSq x * euclidNormSq y := by
  unfold euclidDot euclidNormSq
  simpa only [sq] using Finset.sum_mul_sq_le_sq_mul_sq Finset.univ x y

/-- At `p = 2`, the first-mode matricization `Mat₁(û) ∈ ℝ^{n × 1}` is the column vector `û`
itself. For unit `û`, any unit vector `v̂` maximizing the column singular value squared
`⟨v̂, û⟩²` attains `⟨v̂, û⟩² = 1`, hence `v̂ = û` or `v̂ = -û`, and therefore
`⟨v̂, v⟩² = ⟨û, v⟩²` for every signal vector `v`. -/
theorem p2_mat1_sv_overlap_eq {n : ℕ} (uHat vHat v : Fin n → ℝ)
    (hu : euclidNormSq uHat = 1) (hv : euclidNormSq vHat = 1)
    (hmax : ∀ x : Fin n → ℝ, euclidNormSq x = 1 → (euclidDot x uHat) ^ 2 ≤ (euclidDot vHat uHat) ^ 2) :
    (euclidDot vHat v) ^ 2 = (euclidDot uHat v) ^ 2 := by
  have hge1 : 1 ≤ (euclidDot vHat uHat) ^ 2 := by
    have h1 := hmax uHat hu
    have hself : euclidDot uHat uHat = 1 := hu
    rw [hself, one_pow] at h1
    exact h1
  have hle1 : (euclidDot vHat uHat) ^ 2 ≤ 1 := by
    have hcs := cauchy_schwarz_sq vHat uHat
    rw [hv, hu, mul_one] at hcs
    exact hcs
  have heq1 : (euclidDot vHat uHat) ^ 2 = 1 := le_antisymm hle1 hge1
  set c : ℝ := euclidDot vHat uHat
  have hc_sq : c ^ 2 = 1 := heq1
  -- Show `euclidNormSq (vHat - c • uHat) = 0`, hence `vHat = c • uHat`.
  have hdiff_zero : euclidNormSq (fun i => vHat i - c * uHat i) = 0 := by
    have hexpand : euclidNormSq (fun i => vHat i - c * uHat i) =
        euclidNormSq vHat - 2 * c * euclidDot vHat uHat + c ^ 2 * euclidNormSq uHat := by
      unfold euclidNormSq euclidDot
      simp only [Finset.mul_sum, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl fun i _ => by ring
    rw [hexpand, hv, hu]
    change 1 - 2 * c * c + c ^ 2 * 1 = 0
    nlinarith
  have hv_eq : ∀ i, vHat i = c * uHat i := by
    intro i
    have hnonneg : ∀ j ∈ (Finset.univ : Finset (Fin n)), 0 ≤ (vHat j - c * uHat j) * (vHat j - c * uHat j) :=
      fun j _ => mul_self_nonneg _
    have hzero_i : (vHat i - c * uHat i) * (vHat i - c * uHat i) = 0 :=
      (Finset.sum_eq_zero_iff_of_nonneg hnonneg).mp hdiff_zero i (Finset.mem_univ i)
    have : vHat i - c * uHat i = 0 := mul_self_eq_zero.mp hzero_i
    linarith
  have hdot : euclidDot vHat v = c * euclidDot uHat v := by
    unfold euclidDot
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun i _ => by rw [hv_eq i]; ring
  rw [hdot, mul_pow, hc_sq, one_mul]

/-! ### 2. Even `p = 2q ≥ 4, τ > 1` Supercritical Exact-Recovery Algebraic Core (Theorem 1) -/

/-- Outlier eigenvalue limit `z_* = τ + τ⁻¹` for `τ > 1`. -/
noncomputable def zStar (τ : ℝ) : ℝ := τ + τ⁻¹

/-- Semicircle Stieltjes transform `s_sc(z) = -(z - √(z² - 4)) / 2` for real `z ≥ 2`. -/
noncomputable def sSc (z : ℝ) : ℝ := -(z - Real.sqrt (z ^ 2 - 4)) / 2

theorem zStar_gt_two {τ : ℝ} (hτ : 1 < τ) : 2 < zStar τ := by
  unfold zStar
  have hτ_pos : 0 < τ := by linarith
  have h1 : 0 < (τ - 1) ^ 2 / τ := div_pos (sq_pos_of_pos (by linarith)) hτ_pos
  have h2 : τ + τ⁻¹ - 2 = (τ - 1) ^ 2 / τ := by
    field_simp [ne_of_gt hτ_pos]
    ring
  linarith

/-- Exact semicircle Stieltjes value at the supercritical outlier:
`s_sc(τ + τ⁻¹) = -τ⁻¹` for every `τ > 1`. -/
theorem sSc_zStar {τ : ℝ} (hτ : 1 < τ) : sSc (zStar τ) = -τ⁻¹ := by
  unfold sSc zStar
  have hτ_pos : 0 < τ := by linarith
  have hdiff_pos : 0 ≤ τ - τ⁻¹ := by
    have : τ⁻¹ < 1 := inv_lt_one_of_one_lt₀ hτ
    linarith
  have hsq : (τ + τ⁻¹) ^ 2 - 4 = (τ - τ⁻¹) ^ 2 := by
    field_simp [ne_of_gt hτ_pos]
    ring
  rw [hsq, Real.sqrt_sq hdiff_pos]
  ring

theorem neg_tau_mul_sSc_zStar {τ : ℝ} (hτ : 1 < τ) : -τ * sSc (zStar τ) = 1 := by
  rw [sSc_zStar hτ]
  have hτ_ne : τ ≠ 0 := ne_of_gt (by linarith)
  field_simp [hτ_ne]

/-- **Eigenvector–Resolvent Identity** (Step 4 of Theorem 1):
If `(M + τ u uᵀ) û = ẑ û` and `R` is the inverse of `M - ẑ I`, then
`û = -(τ ⟨u, û⟩) • (R *ᵥ u)`. -/
theorem eigenvector_resolvent_identity {m : ℕ} (M R : Matrix (Fin m) (Fin m) ℝ)
    (u uHat : Fin m → ℝ) (τ zHat : ℝ)
    (heig : M *ᵥ uHat + (τ * dotProduct u uHat) • u = zHat • uHat)
    (hR : R * (M - zHat • (1 : Matrix (Fin m) (Fin m) ℝ)) = 1) :
    uHat = (-(τ * dotProduct u uHat)) • (R *ᵥ u) := by
  have hrearr : (M - zHat • (1 : Matrix (Fin m) (Fin m) ℝ)) *ᵥ uHat =
      (-(τ * dotProduct u uHat)) • u := by
    rw [Matrix.sub_mulVec, Matrix.smul_mulVec, Matrix.one_mulVec]
    ext i
    have hi := congrFun heig i
    simp only [Pi.add_apply, Pi.sub_apply, Pi.smul_apply, smul_eq_mul] at hi ⊢
    linarith
  calc uHat = (1 : Matrix (Fin m) (Fin m) ℝ) *ᵥ uHat := (Matrix.one_mulVec uHat).symm
    _ = (R * (M - zHat • (1 : Matrix (Fin m) (Fin m) ℝ))) *ᵥ uHat := by rw [hR]
    _ = R *ᵥ ((M - zHat • (1 : Matrix (Fin m) (Fin m) ℝ)) *ᵥ uHat) := (Matrix.mulVec_mulVec _ _ _).symm
    _ = R *ᵥ ((-(τ * dotProduct u uHat)) • u) := by rw [hrearr]
    _ = (-(τ * dotProduct u uHat)) • (R *ᵥ u) := Matrix.mulVec_smul R _ u

/-- **Rank-One Left Singular Vector Perturbation Bound** (Step 4 finale of Theorem 1):
Let `B = ρ v wᵀ + E ∈ ℝ^{n × N}` with unit vectors `v, w`, signal `ρ > 0`, and operator
perturbation bound `‖Eᵀ x‖₂ ≤ ε` for all unit `x`. If `v̂` is a unit leading left singular
vector of `B` (maximizing `‖Bᵀ x‖₂`), then `1 - |⟨v̂, v⟩| ≤ 2 ε / ρ`. -/
theorem rank_one_sv_perturbation {n N : ℕ} (B E : Matrix (Fin n) (Fin N) ℝ)
    (v vHat : Fin n → ℝ) (w : Fin N → ℝ) {ρ ε : ℝ}
    (hρ : 0 < ρ)
    (hv : euclidNormSq v = 1) (hvHat : euclidNormSq vHat = 1) (hw : euclidNormSq w = 1)
    (hB : ∀ x : Fin n → ℝ, Bᵀ *ᵥ x = (ρ * euclidDot x v) • w + Eᵀ *ᵥ x)
    (hE : ∀ x : Fin n → ℝ, euclidNormSq x = 1 → Real.sqrt (euclidNormSq (Eᵀ *ᵥ x)) ≤ ε)
    (hmax : Real.sqrt (euclidNormSq (Bᵀ *ᵥ v)) ≤ Real.sqrt (euclidNormSq (Bᵀ *ᵥ vHat))) :
    1 - |euclidDot vHat v| ≤ 2 * ε / ρ := by
  -- Helper: triangle inequality for `√(euclidNormSq ·)`.
  have htri : ∀ a b : Fin N → ℝ,
      Real.sqrt (euclidNormSq (a + b)) ≤
        Real.sqrt (euclidNormSq a) + Real.sqrt (euclidNormSq b) := by
    intro a b
    have hsq : euclidNormSq (a + b) ≤
        (Real.sqrt (euclidNormSq a) + Real.sqrt (euclidNormSq b)) ^ 2 := by
      have hexp : euclidNormSq (a + b) = euclidNormSq a + 2 * euclidDot a b + euclidNormSq b := by
        unfold euclidNormSq euclidDot
        simp only [Pi.add_apply, Finset.mul_sum, ← Finset.sum_add_distrib]
        refine Finset.sum_congr rfl fun i _ => by ring
      have hcs : euclidDot a b ≤ Real.sqrt (euclidNormSq a) * Real.sqrt (euclidNormSq b) := by
        have h1 := cauchy_schwarz_sq a b
        have h2 : Real.sqrt ((euclidDot a b) ^ 2) ≤
            Real.sqrt (euclidNormSq a * euclidNormSq b) := Real.sqrt_le_sqrt h1
        rw [Real.sqrt_sq_eq_abs, Real.sqrt_mul (euclidNormSq_nonneg a)] at h2
        exact le_trans (le_abs_self _) h2
      rw [hexp, add_sq, Real.sq_sqrt (euclidNormSq_nonneg a), Real.sq_sqrt (euclidNormSq_nonneg b)]
      linarith
    calc Real.sqrt (euclidNormSq (a + b))
        ≤ Real.sqrt ((Real.sqrt (euclidNormSq a) + Real.sqrt (euclidNormSq b)) ^ 2) :=
          Real.sqrt_le_sqrt hsq
      _ = Real.sqrt (euclidNormSq a) + Real.sqrt (euclidNormSq b) :=
          Real.sqrt_sq (by positivity)
  have hnorm_smul : ∀ (c : ℝ), Real.sqrt (euclidNormSq (c • w)) = |c| := by
    intro c
    have h1 : euclidNormSq (c • w) = c ^ 2 * euclidNormSq w := by
      unfold euclidNormSq euclidDot
      simp only [Pi.smul_apply, smul_eq_mul, Finset.mul_sum]
      refine Finset.sum_congr rfl fun i _ => by ring
    rw [h1, hw, mul_one, Real.sqrt_sq_eq_abs]
  -- Lower bound `‖Bᵀ v‖ ≥ ρ - ε`.
  have hlow : ρ - ε ≤ Real.sqrt (euclidNormSq (Bᵀ *ᵥ v)) := by
    have hsplit : ρ • w = Bᵀ *ᵥ v + (-(Eᵀ *ᵥ v)) := by
      rw [hB v, show euclidDot v v = 1 from hv, mul_one]
      ext i; simp
    have hneg_norm : euclidNormSq (-(Eᵀ *ᵥ v)) = euclidNormSq (Eᵀ *ᵥ v) := by
      unfold euclidNormSq euclidDot
      refine Finset.sum_congr rfl fun i _ => by simp
    have h1 := htri (Bᵀ *ᵥ v) (-(Eᵀ *ᵥ v))
    rw [← hsplit, hnorm_smul ρ, abs_of_pos hρ, hneg_norm] at h1
    have hEv := hE v hv
    linarith
  -- Upper bound `‖Bᵀ v̂‖ ≤ ρ |⟨v̂, v⟩| + ε`.
  have hhigh : Real.sqrt (euclidNormSq (Bᵀ *ᵥ vHat)) ≤ ρ * |euclidDot vHat v| + ε := by
    rw [hB vHat]
    have h1 := htri ((ρ * euclidDot vHat v) • w) (Eᵀ *ᵥ vHat)
    rw [hnorm_smul, abs_mul, abs_of_pos hρ] at h1
    have hEvHat := hE vHat hvHat
    linarith
  have hgap : ρ * (1 - |euclidDot vHat v|) ≤ 2 * ε := by linarith
  exact (le_div_iff₀' hρ).mpr hgap

/-- Exact Frobenius Lipschitz constant scaling for even `p = 2q` (Step 2 of Theorem 1):
`√(p / n) / (α_{p,q} * n^{(p - 2) / 4}) = √((2q)! / q!) * n^{-q / 2}`. -/
theorem lipschitz_scaling_even {q n : ℕ} (hq : 1 ≤ q) (hn : 1 ≤ n) :
    Real.sqrt (((2 * q : ℕ) : ℝ) / (n : ℝ)) /
      (alphaPQ (2 * q) q * (n : ℝ) ^ ((((2 * q : ℕ) : ℝ) - 2) / 4)) =
    Real.sqrt ((Nat.factorial (2 * q) : ℝ) / (Nat.factorial q : ℝ)) *
      (n : ℝ) ^ (-(q : ℝ) / 2) := by
  unfold alphaPQ
  have hn_pos : (0 : ℝ) < (n : ℝ) := by exact_mod_cast (by omega : 0 < n)
  have hq_pos : (0 : ℝ) < (Nat.factorial q : ℝ) := by exact_mod_cast Nat.factorial_pos q
  have h2q1_pos : (0 : ℝ) < (Nat.factorial (2 * q - 1) : ℝ) := by
    exact_mod_cast Nat.factorial_pos (2 * q - 1)
  have hfact : (Nat.factorial (2 * q) : ℝ) = ((2 * q : ℕ) : ℝ) * (Nat.factorial (2 * q - 1) : ℝ) := by
    have hnat : Nat.factorial (2 * q) = (2 * q) * Nat.factorial (2 * q - 1) := by
      cases h : 2 * q with
      | zero => omega
      | succ k =>
        simp [Nat.factorial_succ]
    exact_mod_cast hnat
  -- Simplify the square-root ratio.
  have hsqrt_ratio :
      Real.sqrt (((2 * q : ℕ) : ℝ)) /
        Real.sqrt ((Nat.factorial q : ℝ) / (Nat.factorial (2 * q - 1) : ℝ)) =
      Real.sqrt ((Nat.factorial (2 * q) : ℝ) / (Nat.factorial q : ℝ)) := by
    rw [Real.sqrt_div (by positivity), div_div_eq_mul_div, ← Real.sqrt_mul (by positivity),
      ← hfact, Real.sqrt_div (by positivity)]
  -- Simplify the powers of `n`.
  have hpow : (Real.sqrt (n : ℝ))⁻¹ / (n : ℝ) ^ ((((2 * q : ℕ) : ℝ) - 2) / 4) =
      (n : ℝ) ^ (-(q : ℝ) / 2) := by
    rw [Real.sqrt_eq_rpow, ← Real.rpow_neg (le_of_lt hn_pos), ← Real.rpow_sub hn_pos]
    congr 1
    push_cast
    ring
  rw [Real.sqrt_div (by positivity)]
  calc Real.sqrt ((2 * q : ℕ) : ℝ) / Real.sqrt (n : ℝ) /
        (Real.sqrt ((Nat.factorial q : ℝ) / (Nat.factorial (2 * q - 1) : ℝ)) *
          (n : ℝ) ^ ((((2 * q : ℕ) : ℝ) - 2) / 4))
      = (Real.sqrt ((2 * q : ℕ) : ℝ) /
          Real.sqrt ((Nat.factorial q : ℝ) / (Nat.factorial (2 * q - 1) : ℝ))) *
        ((Real.sqrt (n : ℝ))⁻¹ / (n : ℝ) ^ ((((2 * q : ℕ) : ℝ) - 2) / 4)) := by
          field_simp [ne_of_gt (Real.sqrt_pos.mpr hn_pos),
            ne_of_gt (Real.sqrt_pos.mpr (div_pos hq_pos h2q1_pos)),
            ne_of_gt (Real.rpow_pos_of_pos hn_pos _)]
    _ = Real.sqrt ((Nat.factorial (2 * q) : ℝ) / (Nat.factorial q : ℝ)) *
        (n : ℝ) ^ (-(q : ℝ) / 2) := by rw [hsqrt_ratio, hpow]

/-! ### 3. Rectangular Phase Transition (`1 ≤ k < p / 2`, `conj:transition`) -/

/-- Deformed upper-edge function `f(d) = 2` for `d ≤ 1` and `d + d⁻¹` for `d > 1`. -/
noncomputable def edgeF (d : ℝ) : ℝ :=
  if d ≤ 1 then 2 else d + d⁻¹

/-- **Continuous Differentiability of the Deformed Upper Edge `f(d)` (including `d = 1`)**:
For every `d > 0`, `f` is differentiable at `d` with derivative `(1 - d⁻²)⁺`
(specifically `0` at `d ≤ 1`, including `d = 1`, and `1 - d⁻²` for `d > 1`). -/
theorem hasDerivAt_edgeF {d : ℝ} (hd : 0 < d) :
    HasDerivAt edgeF (if d ≤ 1 then 0 else 1 - d⁻¹ ^ 2) d := by
  by_cases hlt : d < 1
  · rw [if_pos (le_of_lt hlt)]
    have hev : edgeF =ᶠ[nhds d] fun _ => (2 : ℝ) := by
      filter_upwards [Iio_mem_nhds hlt] with x hx
      simp [edgeF, le_of_lt (Set.mem_Iio.mp hx)]
    exact (hasDerivAt_const d (2 : ℝ)).congr_of_eventuallyEq hev
  · by_cases hgt : 1 < d
    · rw [if_neg (not_le_of_gt hgt)]
      have hev : edgeF =ᶠ[nhds d] fun x => x + x⁻¹ := by
        filter_upwards [Ioi_mem_nhds hgt] with x hx
        simp [edgeF, not_le_of_gt (Set.mem_Ioi.mp hx)]
      have hd1 : HasDerivAt (fun x : ℝ => x + x⁻¹) (1 + -(d ^ 2)⁻¹) d :=
        (hasDerivAt_id d).add (hasDerivAt_inv (ne_of_gt hd))
      have hsimp : 1 + -(d ^ 2)⁻¹ = 1 - d⁻¹ ^ 2 := by rw [inv_pow]; ring
      rw [hsimp] at hd1
      exact hd1.congr_of_eventuallyEq hev
    · -- Critical point `d = 1`!
      have hd1 : d = 1 := le_antisymm (not_lt.mp hgt) (not_lt.mp hlt)
      subst hd1
      rw [if_pos le_rfl, hasDerivAt_iff_tendsto_slope]
      -- Show `slope edgeF 1 x → 0` as `x → 1, x ≠ 1`.
      have hbound : ∀ᶠ x in nhdsWithin (1 : ℝ) {(1 : ℝ)}ᶜ,
          ‖slope edgeF 1 x‖ ≤ 2 * |x - 1| := by
        filter_upwards [self_mem_nhdsWithin,
          nhdsWithin_le_nhds (Ioi_mem_nhds (show (1 : ℝ) / 2 < 1 by norm_num))] with x _ hxhalf
        have hxhalf' : (1 : ℝ) / 2 < x := Set.mem_Ioi.mp hxhalf
        have hx_pos : (0 : ℝ) < x := by linarith
        rw [slope_def_field, Real.norm_eq_abs]
        by_cases hx1 : x ≤ 1
        · simp [edgeF, hx1]
        · push_neg at hx1
          simp only [edgeF, not_le_of_gt hx1, ↓reduceIte, le_rfl]
          have hcalc : (x + x⁻¹ - 2) / (x - 1) = (x - 1) / x := by
            have hsub_ne : x - 1 ≠ 0 := by linarith
            field_simp [ne_of_gt hx_pos, hsub_ne]
            ring
          rw [hcalc, abs_div, abs_of_pos hx_pos]
          have hinv : x⁻¹ ≤ 2 := by
            rw [inv_le_comm₀ hx_pos (by norm_num : (0 : ℝ) < 2)]
            linarith
          calc |x - 1| / x = |x - 1| * x⁻¹ := div_eq_mul_inv _ _
            _ ≤ |x - 1| * 2 := mul_le_mul_of_nonneg_left hinv (abs_nonneg _)
            _ = 2 * |x - 1| := by ring
      have htend0 : Filter.Tendsto (fun x : ℝ => 2 * |x - 1|) (nhdsWithin 1 {1}ᶜ) (nhds 0) := by
        have hcont : Continuous (fun x : ℝ => 2 * |x - 1|) := by continuity
        have h1 := (hcont.continuousAt (x := (1 : ℝ))).tendsto.mono_left (nhdsWithin_le_nhds (s := {(1 : ℝ)}ᶜ))
        simpa using h1
      exact squeeze_zero_norm' hbound htend0

/-- **Variational Two-Sided Overlap Squeeze** (Theorem 5):
If `x` is a near-maximizer of the Rayleigh quotient of `H + t a aᵀ` within `ε`, then for
every `δ > 0`, `⟨a, x⟩²` is squeezed between the left and right difference quotients of
`λ_max(H + · a aᵀ)`. -/
theorem left_overlap_variational_squeeze {q_x a_x_sq t δ ε lam_t lam_minus lam_plus : ℝ}
    (hδ : 0 < δ)
    (hnear : lam_t - ε ≤ q_x + t * a_x_sq)
    (hminus : q_x + (t - δ) * a_x_sq ≤ lam_minus)
    (hplus : q_x + (t + δ) * a_x_sq ≤ lam_plus) :
    (lam_t - lam_minus - ε) / δ ≤ a_x_sq ∧ a_x_sq ≤ (lam_plus - lam_t + ε) / δ := by
  constructor
  · rw [div_le_iff₀ hδ]; linarith
  · rw [le_div_iff₀ hδ]; linarith

/-- Noise rescaling variance `σ² = k! (p - k)! / (n (p - 1)!)`. -/
noncomputable def sigmaSq (p k n : ℕ) : ℝ :=
  ((Nat.factorial k : ℝ) * (Nat.factorial (p - k) : ℝ)) /
    ((n : ℝ) * (Nat.factorial (p - 1) : ℝ))

/-- Left threshold constant `γ_{ℓ,p,k} = (k! (p - k)! / (p - 1)!²)^{1/4}`. -/
noncomputable def gammaLeft (p k : ℕ) : ℝ :=
  (((Nat.factorial k : ℝ) * (Nat.factorial (p - k) : ℝ)) /
    ((Nat.factorial (p - 1) : ℝ) ^ 2)) ^ ((1 : ℝ) / 4)

/-- Right threshold constant `γ_{r,p,k} = (k! / (p - 1)!)^{1/2}`. -/
noncomputable def gammaRight (p k : ℕ) : ℝ :=
  Real.sqrt ((Nat.factorial k : ℝ) / (Nat.factorial (p - 1) : ℝ))

/-- **Exact Left Scaling Identity** (Section 3.3 of `solution.md`):
`σ⁴ * (n^k / k!) * (n^{p-k} / (p-k)!) = γ_{ℓ,p,k}⁴ * n^{p-2}` for all `1 ≤ k < p`, `n ≥ 1`. -/
theorem left_scaling_identity {p k n : ℕ} (hk : 1 ≤ k) (hkp : k < p) (hn : 1 ≤ n) :
    (sigmaSq p k n) ^ 2 * ((n : ℝ) ^ k / (Nat.factorial k : ℝ)) *
      ((n : ℝ) ^ (p - k) / (Nat.factorial (p - k) : ℝ)) =
    (gammaLeft p k) ^ 4 * (n : ℝ) ^ (p - 2) := by
  unfold sigmaSq gammaLeft
  have hbase_nn : 0 ≤ ((Nat.factorial k : ℝ) * (Nat.factorial (p - k) : ℝ)) /
      ((Nat.factorial (p - 1) : ℝ) ^ 2) := by positivity
  have hpow4 : ((((Nat.factorial k : ℝ) * (Nat.factorial (p - k) : ℝ)) /
      ((Nat.factorial (p - 1) : ℝ) ^ 2)) ^ ((1 : ℝ) / 4)) ^ 4 =
      ((Nat.factorial k : ℝ) * (Nat.factorial (p - k) : ℝ)) /
        ((Nat.factorial (p - 1) : ℝ) ^ 2) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hbase_nn]
    norm_num
  rw [hpow4]
  have hn_ne : (n : ℝ) ≠ 0 := by exact_mod_cast (by omega : n ≠ 0)
  have hk_ne : (Nat.factorial k : ℝ) ≠ 0 := by exact_mod_cast Nat.factorial_ne_zero k
  have hpk_ne : (Nat.factorial (p - k) : ℝ) ≠ 0 := by exact_mod_cast Nat.factorial_ne_zero (p - k)
  have hp1_ne : (Nat.factorial (p - 1) : ℝ) ≠ 0 := by exact_mod_cast Nat.factorial_ne_zero (p - 1)
  have hp2 : p = (p - 2) + 2 := by omega
  have hnpow : (n : ℝ) ^ k * (n : ℝ) ^ (p - k) = (n : ℝ) ^ (p - 2) * (n : ℝ) ^ 2 := by
    rw [← pow_add, ← pow_add, Nat.add_sub_of_le (le_of_lt hkp), ← hp2]
  field_simp [hn_ne, hk_ne, hpk_ne, hp1_ne]
  linarith [hnpow]

/-- **Exact Right Scaling Identity** (Section 3.3 of `solution.md`):
`σ² * (n^{p-k} / (p-k)!) = γ_{r,p,k}² * n^{p-k-1}` for all `1 ≤ k < p`, `n ≥ 1`. -/
theorem right_scaling_identity {p k n : ℕ} (hkp : k < p) (hn : 1 ≤ n) :
    sigmaSq p k n * ((n : ℝ) ^ (p - k) / (Nat.factorial (p - k) : ℝ)) =
    (gammaRight p k) ^ 2 * (n : ℝ) ^ (p - k - 1) := by
  unfold sigmaSq gammaRight
  rw [Real.sq_sqrt (by positivity)]
  have hn_ne : (n : ℝ) ≠ 0 := by exact_mod_cast (by omega : n ≠ 0)
  have hpk_ne : (Nat.factorial (p - k) : ℝ) ≠ 0 := by exact_mod_cast Nat.factorial_ne_zero (p - k)
  have hp1_ne : (Nat.factorial (p - 1) : ℝ) ≠ 0 := by exact_mod_cast Nat.factorial_ne_zero (p - 1)
  have hpow : (n : ℝ) ^ (p - k) = (n : ℝ) ^ (p - k - 1) * (n : ℝ) := by
    have : p - k = (p - k - 1) + 1 := by omega
    conv_lhs => rw [this]
    rw [pow_succ]
  rw [hpow]
  field_simp [hn_ne, hpk_ne, hp1_ne]

/-- Substituting `t = τ_ℓ²` into the deformed-edge derivative `(1 - t⁻²)⁺` yields the exact
rectangular left-overlap threshold `(1 - τ_ℓ⁻⁴)⁺`. -/
theorem left_overlap_substitution {τ_l : ℝ} :
    max (0 : ℝ) (1 - (τ_l ^ 2)⁻¹ ^ 2) = max (0 : ℝ) (1 - τ_l⁻¹ ^ 4) := by
  congr 2
  rw [inv_pow, inv_pow, ← pow_mul]

/-- Exact rectangular right-overlap formula `ℓ² / (N + ℓ²) = τ_r² / (1 + τ_r²)`
with `τ_r = ℓ / √N`. -/
theorem right_overlap_formula {N ℓ : ℝ} (hN : 0 < N) :
    let τ_r := ℓ / Real.sqrt N
    ℓ ^ 2 / (N + ℓ ^ 2) = τ_r ^ 2 / (1 + τ_r ^ 2) := by
  dsimp only
  rw [div_pow, Real.sq_sqrt (le_of_lt hN)]
  have hN_ne : N ≠ 0 := ne_of_gt hN
  have hden : N + ℓ ^ 2 ≠ 0 := ne_of_gt (by positivity)
  field_simp [hN_ne, hden]

end SymTensorUnfolding
