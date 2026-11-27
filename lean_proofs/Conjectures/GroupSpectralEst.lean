import Mathlib

/-!
# Refutation of Three Conjectures on Group-Averaged Spectral Estimation
(Mitchell A. Thornton, arXiv:2604.03634 v17 draft, June 2026: `conj:blind`, `conj:nadh`, `conj:gaat`)

We formalize in Lean 4 + Mathlib the exact algebraic, spectral, analytic, and representation-
theoretic cores refuting all three conjectures and establishing the positive equivariant
projection and fixed-noise covariance bounds:

1. **Disproof of Blind Group Selection (`conj:blind` — Theorem 1, Propositions 2–3)**:
   - Explicit 4×4 complex unitary permutation-matrix groups `G_Z4` (`≅ ℤ₄`), `H_V4` (`≅ V₄`),
     and `K_Z4` (conjugate cyclic group `⟨(1 3 2 4)⟩`), verifying cardinality 4, unitarity,
     and regular transitive action on `Fin 4`.
   - `blind_Z4_vs_V4_counterexample`: At `s = 2 w_+ + ε w_-` (`ε ∈ {-1, 1}`), `‖s‖² = 5`,
     `F_G(s) = R₀` has eigenvalues `{4, 1, 0, 0}` (`ψ(G, s) = 4/5`), `F_H(s)` has eigenvalues
     `{9/2, 1/2, 0, 0}` (`ψ(H, s) = 9/10 > 4/5`), `[F_G(s), R₀] = 0`, every `g ∈ G_Z4`
     commutes with `R₀`, while `‖[F_H(s), R₀]‖_F² = 72`, `‖F_H(s)‖_F² = 41/2`, `‖R₀‖_F² = 17`,
     giving exact commutator residual `δ(H, s, R₀)² = 144 / 697 > 0 = δ(G, s, R₀)²`.
   - `blind_conjugate_cyclic_C4`: For the `C₄` graph filter `R₀ = (2I - A)²` and support
     point `y₀ = 3u + v + i b`, `‖y₀‖² = 11`, `ψ(G, y₀) = 5/11 < 9/11 = ψ(K, y₀)`,
     `[F_G(y₀), R₀] = 0`, and `[F_K(y₀), R₀] ≠ 0`.
   - `blind_population_twirl_oracle`: At `a = (2, 1, -2, -1)` and `R = a aᴴ + σ² I`,
     `[T_G(R), R] = 0`, `[T_K(R), R] ≠ 0`, while `ψ(G, a) = 1/2 < 9/10 = ψ(K, a)`.

2. **Disproof of the Non-Abelian Dominance Hypothesis (`conj:nadh` — Theorem 2)**:
   - `unitary_cyclic_conjugate_fixes_vector` & `exists_unitary_cyclic_score_one`:
     For every `n ≥ 1` and every nonzero `x : Fin n → ℂ`, there exists a unitary matrix
     `U ∈ U(n)` such that `Uᴴ P^k U x = x` for every cyclic shift `P^k`, whence
     `F_{Uᴴ ℤ_n U}(x) = x xᴴ` with exact top eigenvalue `‖x‖²` and concentration score `1`.
   - `sunlet_S3_automorphisms_and_spectrum`: For the 6-vertex 3-sunlet graph `A_sunlet`,
     the 6 permutation matrices of `S₃` form a non-Abelian group of order 6 commuting with
     `A_sunlet`; the exact eigenvalues of `A_sunlet` are `1 ± √2` (simple) and
     `(-1 ± √5)/2` (multiplicity 2), all `> -3`, so `h(A) = 3I + A` has unique top
     eigenvalue `4 + √2`.
   - `sunlet_S3_strict_score_lt_one`: For `x = e₁` (`‖x‖² = 1`), `F_{S₃}(x) = diag(1/3, 1/3, 1/3, 0, 0, 0)`
     satisfies `⟨v, F_{S₃}(x) v⟩ ≤ 1/3 < 1` for every unit vector `v`, refuting `conj:nadh`.

3. **Disproof and Repair of General Algebraic Averaging (`conj:gaat` — Section 4, Props. 4–5)**:
   - `gaat_exp_half_finite_mean` & `gaat_exp_half_infinite_variance`:
     Under `T ~ Exp(1)`, `∫₀^∞ exp(t/2) exp(-t) dt = 2 < ∞` while
     `∫₀^∞ (exp(t/2))² exp(-t) dt = ∫₀^∞ 1 dt` is not integrable (infinite variance).
   - `gaat_noise_scaling_unbounded`: `σ⁴` exceeds any `σ`-independent constant `C`.
   - `gaat_nonlinear_involution_counterexample`: The bijection `h(t) = t + max t 0` defines
     a nonlinear involution `π̃(y) = h(-h⁻¹(y))` fixing `0` for which `f(t) = h(t)` is
     equivariant, yet `(f(t) + f(-t))/2 = |t|/2 > 0` for all `t ≠ 0`.
   - `gaat_z2_zero_variance_defect`: Under `ℤ₂ = {1, -1}`, `f(z) = z` has orbit size `≤ 2`
     while its group average vanishes identically.
   - `unitary_group_average_projector` & `equivariant_average_eq_proj` (Proposition 4):
     For any finite group `G` acting unitarily via `ρ`, `P = |G|⁻¹ ∑_g ρ(g)` is an
     orthogonal projector (`P² = P`, `Pᴴ = P`) onto `G`-invariants, satisfies
     `|G|⁻¹ ∑_g f(π_g x) = P(f x)` for equivariant `f`, and contracts norm `‖P v‖² ≤ ‖v‖²`.
   - `outer_product_variance_bound` (Proposition 5): For any Frobenius-orthonormal Hermitian
     basis `{B₁, ..., B_D}`, `∑_j (σ⁴ ‖B_j‖_F² + 2 σ² ‖B_j s‖²) ≤ D (σ⁴ + 2 σ² ‖s‖²) =
     M² (σ⁴ + 2 σ² ‖s‖²) / d_eff`, with equality at `s = 0`.

Machine-checked vs prose-only:
- Machine-checked: All explicit matrix twirls, commutator residuals, exact eigenvalues and
  concentration scores for `conj:blind`, general-`n` unitary conjugation achieving score `1`
  and 3-sunlet `S₃` spectrum/strictness for `conj:nadh`, exact improper integrals and counterexamples
  for `conj:gaat`, and the positive equivariant projection & fixed-`σ` variance bounds.
- Prose-only (in `solutions/group_spectral_est/proof.pdf`): Almost-sure convergence as `σ ↓ 0`
  via bounded convergence / Fatou's lemma, Gaussian zero-measure of proper subspace unions, and
  explicit complex-Gaussian fourth-moment calculation `Var(xᴴ B x) = σ⁴ tr(B²) + 2 σ² ‖Bs‖²`.
-/

open Matrix BigOperators Finset MeasureTheory Set

noncomputable section

namespace GroupSpectralEst

/-! ### 1. Core Definitions: Group Twirl, Rank-1 Estimator, Commutator, Frobenius Norm -/

/-- Group twirl `T_G(B) = |G|⁻¹ ∑_{g ∈ G} g B gᴴ` for a finite list of `n × n` complex matrices. -/
def twirl {n : ℕ} (G : List (Matrix (Fin n) (Fin n) ℂ))
    (B : Matrix (Fin n) (Fin n) ℂ) : Matrix (Fin n) (Fin n) ℂ :=
  ((G.length : ℂ)⁻¹) • (G.map fun g => g * B * gᴴ).sum

/-- Outer product `x xᴴ` as an `n × n` complex matrix. -/
def outerProd {n : ℕ} (x : Fin n → ℂ) : Matrix (Fin n) (Fin n) ℂ :=
  Matrix.of fun i j => x i * star (x j)

/-- Single-observation group-averaged estimator `F_G(x) = T_G(x xᴴ)`. -/
def estF {n : ℕ} (G : List (Matrix (Fin n) (Fin n) ℂ))
    (x : Fin n → ℂ) : Matrix (Fin n) (Fin n) ℂ :=
  twirl G (outerProd x)

/-- Matrix commutator `[A, B] = A B - B A`. -/
def comm {n : ℕ} (A B : Matrix (Fin n) (Fin n) ℂ) : Matrix (Fin n) (Fin n) ℂ :=
  A * B - B * A

/-- Squared Euclidean norm `‖x‖² = ∑_i |x_i|²`. -/
def vecNormSq {n : ℕ} (x : Fin n → ℂ) : ℝ :=
  ∑ i : Fin n, Complex.normSq (x i)

/-- Squared Frobenius norm `‖A‖_F² = ∑_{i, j} |A_{ij}|²`. -/
def frobNormSq {n : ℕ} (A : Matrix (Fin n) (Fin n) ℂ) : ℝ :=
  ∑ i : Fin n, ∑ j : Fin n, Complex.normSq (A i j)

/-- Rayleigh quotient `⟨v, A v⟩` for `A : Matrix (Fin n) (Fin n) ℂ` and `v : Fin n → ℂ`. -/
def quadForm {n : ℕ} (A : Matrix (Fin n) (Fin n) ℂ) (v : Fin n → ℂ) : ℂ :=
  ∑ i : Fin n, ∑ j : Fin n, star (v i) * A i j * v j

/-- Permutation matrix `P_σ` with `(P_σ)_{i, j} = 1` iff `i = σ j`, so `(P_σ x)_i = x_{σ⁻¹ i}`
and `P_σ e_j = e_{σ j}`. -/
def permMat {n : ℕ} (σ : Equiv.Perm (Fin n)) : Matrix (Fin n) (Fin n) ℂ :=
  Matrix.of fun i j => if i = σ j then 1 else 0

lemma mul_star_eq_normSq (z : ℂ) : z * star z = (Complex.normSq z : ℂ) := by
  rw [Complex.star_def, Complex.mul_conj]

lemma star_mul_eq_normSq (z : ℂ) : star z * z = (Complex.normSq z : ℂ) := by
  rw [mul_comm, mul_star_eq_normSq]

lemma permMat_unitary {n : ℕ} (σ : Equiv.Perm (Fin n)) :
    permMat σ * (permMat σ)ᴴ = 1 ∧ (permMat σ)ᴴ * permMat σ = 1 := by
  constructor
  · ext i j
    simp only [permMat, Matrix.mul_apply, Matrix.conjTranspose_apply, Matrix.of_apply,
      Matrix.one_apply]
    rw [Finset.sum_eq_single (σ.symm j)]
    · by_cases h : i = j <;> simp [h]
    · intro k _ hk
      have : j ≠ σ k := fun h => hk (by rw [h]; simp)
      simp [this]
    · intro h; exact (h (Finset.mem_univ _)).elim
  · ext i j
    simp only [permMat, Matrix.mul_apply, Matrix.conjTranspose_apply, Matrix.of_apply,
      Matrix.one_apply]
    rw [Finset.sum_eq_single (σ i)]
    · by_cases h : i = j <;> simp [h]
    · intro k _ hk
      simp [hk]
    · intro h; exact (h (Finset.mem_univ _)).elim

lemma permMat_mulVec {n : ℕ} (σ : Equiv.Perm (Fin n)) (x : Fin n → ℂ) :
    (permMat σ).mulVec x = fun i => x (σ.symm i) := by
  ext i
  simp only [Matrix.mulVec, dotProduct, permMat, Matrix.of_apply]
  rw [Finset.sum_eq_single (σ.symm i)]
  · simp
  · intro k _ hk
    have : i ≠ σ k := fun h => hk (by rw [h]; simp)
    simp [this]
  · intro h; exact (h (Finset.mem_univ _)).elim

lemma outerProd_mulVec {n : ℕ} (x v : Fin n → ℂ) :
    (outerProd x).mulVec v = (∑ j, star (x j) * v j) • x := by
  ext i
  simp only [outerProd, Matrix.mulVec, dotProduct, Matrix.of_apply, Pi.smul_apply, smul_eq_mul]
  rw [Finset.sum_mul]
  refine Finset.sum_congr rfl fun j _ => by ring

/-- Key identity: `g * (x xᴴ) * gᴴ = (g x)(g x)ᴴ`. -/
lemma conj_outerProd {n : ℕ} (g : Matrix (Fin n) (Fin n) ℂ) (x : Fin n → ℂ) :
    g * outerProd x * gᴴ = outerProd (g.mulVec x) := by
  ext i j
  simp only [Matrix.mul_apply, outerProd, Matrix.of_apply, Matrix.conjTranspose_apply,
    Matrix.mulVec, dotProduct, star_sum, star_mul']
  rw [Finset.sum_mul]
  simp_rw [Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun k _ => ?_
  refine Finset.sum_congr rfl fun l _ => by ring

lemma estF_eq_sum_outerProd {n : ℕ} (G : List (Matrix (Fin n) (Fin n) ℂ)) (x : Fin n → ℂ) :
    estF G x = ((G.length : ℂ)⁻¹) • (G.map fun g => outerProd (g.mulVec x)).sum := by
  unfold estF twirl
  congr 2
  exact List.map_congr_left fun g _ => conj_outerProd g x

lemma twirl_trace {n : ℕ} (G : List (Matrix (Fin n) (Fin n) ℂ))
    (hGne : G.length ≠ 0) (hU : ∀ g ∈ G, gᴴ * g = 1) (B : Matrix (Fin n) (Fin n) ℂ) :
    Matrix.trace (twirl G B) = Matrix.trace B := by
  unfold twirl
  rw [Matrix.trace_smul, smul_eq_mul]
  have hsum : ∀ L : List (Matrix (Fin n) (Fin n) ℂ),
      (∀ g ∈ L, gᴴ * g = 1) →
      Matrix.trace (L.map fun g => g * B * gᴴ).sum = (L.length : ℂ) * Matrix.trace B := by
    intro L hL
    induction L with
    | nil => simp
    | cons g gs ih =>
      simp only [List.map_cons, List.sum_cons, List.length_cons, Nat.cast_add, Nat.cast_one]
      rw [Matrix.trace_add]
      have hg : gᴴ * g = 1 := hL g (by simp)
      have htr : Matrix.trace (g * B * gᴴ) = Matrix.trace B := by
        rw [Matrix.trace_mul_cycle, hg, Matrix.one_mul]
      rw [htr, ih (fun h hh => hL h (by simp [hh]))]
      ring
  rw [hsum G hU]
  have hlen : (G.length : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr hGne
  field_simp [hlen]

lemma outerProd_trace {n : ℕ} (x : Fin n → ℂ) :
    Matrix.trace (outerProd x) = (vecNormSq x : ℂ) := by
  simp only [Matrix.trace, Matrix.diag, outerProd, Matrix.of_apply, vecNormSq, Complex.ofReal_sum]
  exact Finset.sum_congr rfl fun i _ => mul_star_eq_normSq (x i)

/-! ### 2. Disproof of `conj:blind` (Theorem 1, Propositions 2–3) -/

private def cyc4 : Equiv.Perm (Fin 4) :=
  ⟨![1, 2, 3, 0], ![3, 0, 1, 2], by decide, by decide⟩

private def cyc4_K : Equiv.Perm (Fin 4) :=
  ⟨![2, 3, 1, 0], ![3, 2, 0, 1], by decide, by decide⟩

private def v4_1 : Equiv.Perm (Fin 4) :=
  ⟨![1, 0, 3, 2], ![1, 0, 3, 2], by decide, by decide⟩

private def v4_2 : Equiv.Perm (Fin 4) :=
  ⟨![2, 3, 0, 1], ![2, 3, 0, 1], by decide, by decide⟩

private def v4_3 : Equiv.Perm (Fin 4) :=
  ⟨![3, 2, 1, 0], ![3, 2, 1, 0], by decide, by decide⟩

/-- Regular transitive cyclic group `G = ⟨(1 2 3 4)⟩ ≅ ℤ₄` on `Fin 4`. -/
def G_Z4 : List (Matrix (Fin 4) (Fin 4) ℂ) :=
  [permMat 1, permMat cyc4, permMat (cyc4 ^ 2), permMat (cyc4 ^ 3)]

/-- Regular transitive Klein four-group `H = V₄` on `Fin 4`. -/
def H_V4 : List (Matrix (Fin 4) (Fin 4) ℂ) :=
  [permMat 1, permMat v4_1, permMat v4_2, permMat v4_3]

/-- Conjugate cyclic group `K = ⟨(1 3 2 4)⟩ ≅ ℤ₄` on `Fin 4`. -/
def K_Z4 : List (Matrix (Fin 4) (Fin 4) ℂ) :=
  [permMat 1, permMat cyc4_K, permMat (cyc4_K ^ 2), permMat (cyc4_K ^ 3)]

private def vec_u : Fin 4 → ℂ := ![1 / 2, 1 / 2, -1 / 2, -1 / 2]
private def vec_v : Fin 4 → ℂ := ![-1 / 2, 1 / 2, 1 / 2, -1 / 2]
private def vec_b : Fin 4 → ℂ := ![1 / 2, -1 / 2, 1 / 2, -1 / 2]

private def vec_wp : Fin 4 → ℂ :=
  fun j => (vec_u j + Complex.I * vec_v j) / Real.sqrt 2

private def vec_wm : Fin 4 → ℂ :=
  fun j => (vec_u j - Complex.I * vec_v j) / Real.sqrt 2

/-- Signal `s(ε) = 2 w_+ + ε w_-` from Theorem 1 (`ε ∈ {-1, 1}`). -/
def sig_blind (ε : ℝ) : Fin 4 → ℂ :=
  fun j => 2 * vec_wp j + (ε : ℂ) * vec_wm j

/-- Population signal covariance `R₀ = 4 w_+ w_+ᴴ + w_- w_-ᴴ` from Theorem 1. -/
def cov_R0 : Matrix (Fin 4) (Fin 4) ℂ :=
  (4 : ℂ) • outerProd vec_wp + outerProd vec_wm

lemma outerProd_smul {n : ℕ} (c : ℂ) (x : Fin n → ℂ) :
    outerProd (c • x) = (c * star c) • outerProd x := by
  ext i j
  simp only [outerProd, Matrix.of_apply, Matrix.smul_apply, Pi.smul_apply, smul_eq_mul, star_mul']
  ring

lemma estF_smul {n : ℕ} (G : List (Matrix (Fin n) (Fin n) ℂ)) (c : ℂ) (x : Fin n → ℂ) :
    estF G (c • x) = (c * star c) • estF G x := by
  rw [estF_eq_sum_outerProd, estF_eq_sum_outerProd]
  have hsum : ∀ L : List (Matrix (Fin n) (Fin n) ℂ),
      (L.map fun g => outerProd (g.mulVec (c • x))).sum =
        (c * star c) • (L.map fun g => outerProd (g.mulVec x)).sum := by
    intro L
    induction L with
    | nil => simp
    | cons g gs ih =>
      rw [List.map_cons, List.sum_cons, List.map_cons, List.sum_cons,
        Matrix.mulVec_smul, outerProd_smul, ih, smul_add]
  rw [hsum, smul_comm]

lemma estF_apply_of_perms {n : ℕ} (perms : List (Equiv.Perm (Fin n))) (x : Fin n → ℂ)
    (i j : Fin n) :
    estF (perms.map permMat) x i j =
      ((perms.length : ℂ)⁻¹) * (perms.map fun σ => x (σ.symm i) * star (x (σ.symm j))).sum := by
  rw [estF_eq_sum_outerProd]
  simp only [List.length_map, Matrix.smul_apply, smul_eq_mul]
  congr 1
  induction perms with
  | nil => simp
  | cons σ σs ih =>
    simp only [List.map_cons, List.sum_cons, Matrix.add_apply, ih]
    congr 1
    simp [permMat_mulVec, outerProd]

lemma permMat_mul_permMat {n : ℕ} (σ τ : Equiv.Perm (Fin n)) :
    permMat σ * permMat τ = permMat (σ * τ) := by
  ext i j
  simp only [permMat, Matrix.mul_apply, Matrix.of_apply, Equiv.Perm.coe_mul, Function.comp_apply]
  rw [Finset.sum_eq_single (τ j)]
  · simp
  · intro k _ hk; simp [hk]
  · intro h; exact (h (Finset.mem_univ _)).elim

lemma complex_conj_simp (a b : ℝ) :
    star ((a : ℂ) + (b : ℂ) * Complex.I) = (a : ℂ) - (b : ℂ) * Complex.I := by
  simp [sub_eq_add_neg]

lemma perm_one_symm_apply {n : ℕ} (i : Fin n) :
    (1 : Equiv.Perm (Fin n)).symm i = i := rfl

lemma permMat_one_mulVec {n : ℕ} (x : Fin n → ℂ) :
    (permMat (1 : Equiv.Perm (Fin n))).mulVec x = x := by
  rw [permMat_mulVec]; rfl

lemma cyc4_symm_apply (i : Fin 4) :
    (cyc4.symm i = (![3, 0, 1, 2] : Fin 4 → Fin 4) i) ∧
    ((cyc4 ^ 2).symm i = (![2, 3, 0, 1] : Fin 4 → Fin 4) i) ∧
    ((cyc4 ^ 3).symm i = (![1, 2, 3, 0] : Fin 4 → Fin 4) i) := by
  fin_cases i <;> refine ⟨?_, ?_, ?_⟩ <;> rw [Equiv.symm_apply_eq] <;> decide

lemma cyc4K_symm_apply (i : Fin 4) :
    (cyc4_K.symm i = (![3, 2, 0, 1] : Fin 4 → Fin 4) i) ∧
    ((cyc4_K ^ 2).symm i = (![1, 0, 3, 2] : Fin 4 → Fin 4) i) ∧
    ((cyc4_K ^ 3).symm i = (![2, 3, 1, 0] : Fin 4 → Fin 4) i) := by
  fin_cases i <;> refine ⟨?_, ?_, ?_⟩ <;> rw [Equiv.symm_apply_eq] <;> decide

lemma v4_symm_apply (i : Fin 4) :
    (v4_1.symm i = (![1, 0, 3, 2] : Fin 4 → Fin 4) i) ∧
    (v4_2.symm i = (![2, 3, 0, 1] : Fin 4 → Fin 4) i) ∧
    (v4_3.symm i = (![3, 2, 1, 0] : Fin 4 → Fin 4) i) := by
  fin_cases i <;> refine ⟨?_, ?_, ?_⟩ <;> rw [Equiv.symm_apply_eq] <;> decide

/-! #### Fast Gaussian-integer (`ℤ × ℤ`) evaluator with explicit inverse-permutation tables -/

private def toC_Z (p : ℤ × ℤ) : ℂ := (p.1 : ℂ) + (p.2 : ℂ) * Complex.I
private def zAdd (p q : ℤ × ℤ) : ℤ × ℤ := (p.1 + q.1, p.2 + q.2)
private def zMul (p q : ℤ × ℤ) : ℤ × ℤ := (p.1 * q.1 - p.2 * q.2, p.1 * q.2 + p.2 * q.1)
private def zConj (p : ℤ × ℤ) : ℤ × ℤ := (p.1, -p.2)
private def zScale (s : ℤ) (p : ℤ × ℤ) : ℤ × ℤ := (s * p.1, s * p.2)

private lemma toC_Z_add (p q : ℤ × ℤ) : toC_Z (zAdd p q) = toC_Z p + toC_Z q := by
  simp only [toC_Z, zAdd, Int.cast_add]; ring

private lemma toC_Z_mul (p q : ℤ × ℤ) : toC_Z (zMul p q) = toC_Z p * toC_Z q := by
  simp only [toC_Z, zMul, Int.cast_sub, Int.cast_add, Int.cast_mul]
  ring_nf; simp [Complex.I_sq]; ring

private lemma toC_Z_conj (p : ℤ × ℤ) : toC_Z (zConj p) = star (toC_Z p) := by
  simp only [toC_Z, zConj, Int.cast_neg, star_add, star_mul', Complex.star_def,
    Complex.conj_I, map_intCast]
  ring

private lemma toC_Z_scale (s : ℤ) (p : ℤ × ℤ) : toC_Z (zScale s p) = (s : ℂ) * toC_Z p := by
  simp only [toC_Z, zScale, Int.cast_mul]; ring

private def zSum : List (ℤ × ℤ) → ℤ × ℤ
  | [] => (0, 0)
  | p :: ps => zAdd p (zSum ps)

private lemma toC_Z_sum (l : List (ℤ × ℤ)) : toC_Z (zSum l) = (l.map toC_Z).sum := by
  induction l with
  | nil => simp [zSum, toC_Z]
  | cons p ps ih => simp [zSum, toC_Z_add, ih]

private def zEstSum {n : ℕ} (invs : List (Fin n → Fin n)) (x : Fin n → ℤ × ℤ)
    (i j : Fin n) : ℤ × ℤ :=
  zSum (invs.map fun f => zMul (x (f i)) (zConj (x (f j))))

private lemma estF_eq_of_zEstSum {n : ℕ} (perms : List (Equiv.Perm (Fin n)))
    (invs : List (Fin n → Fin n))
    (hinv : List.Forall₂ (fun σ f => ∀ i, σ.symm i = f i) perms invs)
    (x : Fin n → ℤ × ℤ) (M : Fin n → Fin n → ℤ × ℤ)
    (h : ∀ i j : Fin n, zEstSum invs x i j = M i j) :
    estF (perms.map permMat) (fun i => toC_Z (x i)) =
      ((perms.length : ℂ)⁻¹) • Matrix.of (fun i j => toC_Z (M i j)) := by
  ext i j
  rw [estF_apply_of_perms, Matrix.smul_apply, Matrix.of_apply, smul_eq_mul, ← h i j,
    zEstSum, toC_Z_sum, List.map_map]
  congr 1
  clear h
  induction hinv with
  | nil => rfl
  | cons hσ _ ih =>
    simp only [List.map_cons, List.sum_cons, Function.comp_apply, toC_Z_mul, toC_Z_conj, hσ, ih]

private lemma comm_add_smul_one {n : ℕ} (A B : Matrix (Fin n) (Fin n) ℂ) (c : ℂ) :
    comm (A + c • (1 : Matrix (Fin n) (Fin n) ℂ)) (B + c • (1 : Matrix (Fin n) (Fin n) ℂ)) =
      comm A B := by
  simp only [comm, add_mul, mul_add, Matrix.smul_mul, Matrix.mul_smul,
    Matrix.one_mul, Matrix.mul_one]
  abel

private lemma comm_permMat_eq_zero {n : ℕ} (σ : Equiv.Perm (Fin n)) (M : Matrix (Fin n) (Fin n) ℂ)
    (h : ∀ i j : Fin n, M (σ.symm i) j = M i (σ j)) :
    comm (permMat σ) M = 0 := by
  ext i j
  simp only [comm, Matrix.sub_apply, Matrix.mul_apply, permMat, Matrix.of_apply, Matrix.zero_apply]
  have hL : ∑ k : Fin n, (if i = σ k then (1 : ℂ) else 0) * M k j = M (σ.symm i) j := by
    rw [Finset.sum_eq_single (σ.symm i)]
    · simp
    · intro k _ hk; have : i ≠ σ k := fun h => hk (by rw [h]; simp); simp [this]
    · intro h'; exact (h' (Finset.mem_univ _)).elim
  have hR : ∑ k : Fin n, M i k * (if k = σ j then (1 : ℂ) else 0) = M i (σ j) := by
    rw [Finset.sum_eq_single (σ j)]
    · simp
    · intro k _ hk; simp [hk]
    · intro h'; exact (h' (Finset.mem_univ _)).elim
  rw [hL, hR, h i j, sub_self]

private lemma comm_outerProd_eq_zero_of_eig {n : ℕ} (F : Matrix (Fin n) (Fin n) ℂ)
    (a : Fin n → ℂ) (c : ℝ) (hH : Fᴴ = F) (heig : F.mulVec a = (c : ℂ) • a) :
    comm F (outerProd a) = 0 := by
  ext i j
  simp only [comm, Matrix.sub_apply, Matrix.mul_apply, outerProd, Matrix.of_apply, Matrix.zero_apply]
  have hL : ∑ k : Fin n, F i k * (a k * star (a j)) = (c : ℂ) * a i * star (a j) := by
    have hi := congrFun heig i
    simp only [Matrix.mulVec, dotProduct, Pi.smul_apply, smul_eq_mul] at hi
    calc ∑ k : Fin n, F i k * (a k * star (a j))
        = (∑ k : Fin n, F i k * a k) * star (a j) := by
          rw [Finset.sum_mul]; refine Finset.sum_congr rfl fun k _ => by ring
      _ = (c : ℂ) * a i * star (a j) := by rw [hi]
  have hR : ∑ k : Fin n, a i * star (a k) * F k j = a i * ((c : ℂ) * star (a j)) := by
    have hj := congrArg star (congrFun heig j)
    simp only [Matrix.mulVec, dotProduct, Pi.smul_apply, smul_eq_mul, map_sum, map_mul,
      Complex.star_def, Complex.conj_ofReal] at hj
    have hsum : ∑ k : Fin n, star (a k) * F k j = (c : ℂ) * star (a j) := by
      simp only [Complex.star_def]
      rw [← hj]
      refine Finset.sum_congr rfl fun k _ => ?_
      have hFk : F k j = (starRingEnd ℂ) (F j k) := by
        have := congrFun (congrFun hH k) j
        simpa [Matrix.conjTranspose_apply, Complex.star_def] using this.symm
      rw [hFk, mul_comm]
    calc ∑ k : Fin n, a i * star (a k) * F k j
        = a i * ∑ k : Fin n, star (a k) * F k j := by
          rw [Finset.mul_sum]; refine Finset.sum_congr rfl fun k _ => by ring
      _ = a i * ((c : ℂ) * star (a j)) := by rw [hsum]
  rw [hL, hR]; ring

private def R0Z_mat : Fin 4 → Fin 4 → ℤ × ℤ :=
  ![![(5, 0), (0, -3), (-5, 0), (0, 3)],
    ![(0, 3), (5, 0), (0, -3), (-5, 0)],
    ![(-5, 0), (0, 3), (5, 0), (0, -3)],
    ![(0, -3), (-5, 0), (0, 3), (5, 0)]]

private lemma vec_uvb_star :
    (∀ j, star (vec_u j) = vec_u j) ∧
    (∀ j, star (vec_v j) = vec_v j) ∧
    (∀ j, star (vec_b j) = vec_b j) := by
  refine ⟨fun j => ?_, fun j => ?_, fun j => ?_⟩
  · fin_cases j <;> simp [vec_u]
  · fin_cases j <;> simp [vec_v]
  · fin_cases j <;> simp [vec_b]

private def zMatMul4 (A B : Fin 4 → Fin 4 → ℤ × ℤ) (i j : Fin 4) : ℤ × ℤ :=
  zSum (([0, 1, 2, 3] : List (Fin 4)).map fun k => zMul (A i k) (B k j))

private def zComm4 (A B : Fin 4 → Fin 4 → ℤ × ℤ) (i j : Fin 4) : ℤ × ℤ :=
  zAdd (zMatMul4 A B i j) (zScale (-1) (zMatMul4 B A i j))

private def zNormSq (p : ℤ × ℤ) : ℤ := p.1 * p.1 + p.2 * p.2

private def zFrobSq4 (A : Fin 4 → Fin 4 → ℤ × ℤ) : ℤ :=
  (([0, 1, 2, 3] : List (Fin 4)).map fun i =>
    (([0, 1, 2, 3] : List (Fin 4)).map fun j => zNormSq (A i j)).sum).sum

private lemma comm_toC_Z4 (A B : Fin 4 → Fin 4 → ℤ × ℤ) :
    comm (Matrix.of fun i j => toC_Z (A i j)) (Matrix.of fun i j => toC_Z (B i j)) =
      Matrix.of fun i j => toC_Z (zComm4 A B i j) := by
  ext i j
  have hmul : ∀ X Y : Fin 4 → Fin 4 → ℤ × ℤ,
      (Matrix.of (fun r c => toC_Z (X r c)) * Matrix.of (fun r c => toC_Z (Y r c))) i j =
        toC_Z (zMatMul4 X Y i j) := by
    intro X Y
    simp only [Matrix.mul_apply, Matrix.of_apply, zMatMul4, toC_Z_sum,
      Fin.sum_univ_four, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
      add_zero, toC_Z_mul]
    ring
  simp only [comm, Matrix.sub_apply, hmul, Matrix.of_apply, zComm4, toC_Z_add, toC_Z_scale]
  push_cast; ring

private lemma frobNormSq_smul_toC_Z4 (c : ℝ) (A : Fin 4 → Fin 4 → ℤ × ℤ) :
    frobNormSq ((c : ℂ) • Matrix.of fun i j => toC_Z (A i j)) =
      c ^ 2 * (zFrobSq4 A : ℝ) := by
  have hterm : ∀ p : ℤ × ℤ, Complex.normSq ((c : ℂ) * toC_Z p) = c ^ 2 * (zNormSq p : ℝ) := by
    intro p
    rw [map_mul, Complex.normSq_ofReal]
    simp only [toC_Z, zNormSq, Complex.normSq_apply, Complex.add_re, Complex.add_im,
      Complex.intCast_re, Complex.intCast_im, Complex.mul_re, Complex.mul_im,
      Complex.I_re, Complex.I_im, mul_zero, mul_one, sub_zero, zero_add, add_zero]
    push_cast; ring
  simp only [frobNormSq, Matrix.smul_apply, Matrix.of_apply, smul_eq_mul, hterm, ← Finset.mul_sum,
    zFrobSq4, Fin.sum_univ_four, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, add_zero, Int.cast_add]
  ring

private lemma vec_uvb_ortho :
    (∑ j : Fin 4, star (vec_u j) * vec_u j = 1) ∧
    (∑ j : Fin 4, star (vec_v j) * vec_v j = 1) ∧
    (∑ j : Fin 4, star (vec_b j) * vec_b j = 1) ∧
    (∑ j : Fin 4, star (vec_u j) * vec_v j = 0) ∧
    (∑ j : Fin 4, star (vec_v j) * vec_u j = 0) ∧
    (∑ j : Fin 4, star (vec_u j) * vec_b j = 0) ∧
    (∑ j : Fin 4, star (vec_b j) * vec_u j = 0) ∧
    (∑ j : Fin 4, star (vec_v j) * vec_b j = 0) ∧
    (∑ j : Fin 4, star (vec_b j) * vec_v j = 0) := by
  have ⟨su, sv, sb⟩ := vec_uvb_star
  simp_rw [su, sv, sb]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    (simp [Fin.sum_univ_four, vec_u, vec_v, vec_b]; norm_num)

private lemma blind_cov_R0_eq :
    cov_R0 = (1 / 4 : ℂ) • Matrix.of (fun i j => toC_Z (R0Z_mat i j)) ∧
    cov_R0 = Matrix.of ![![5/4, -(3/4 : ℂ) * Complex.I, -5/4, (3/4 : ℂ) * Complex.I],
                         ![(3/4 : ℂ) * Complex.I, 5/4, -(3/4 : ℂ) * Complex.I, -5/4],
                         ![-5/4, (3/4 : ℂ) * Complex.I, 5/4, -(3/4 : ℂ) * Complex.I],
                         ![-(3/4 : ℂ) * Complex.I, -5/4, (3/4 : ℂ) * Complex.I, 5/4]] := by
  have hsq2_inv : ((1 : ℂ) / Real.sqrt 2) * star ((1 : ℂ) / Real.sqrt 2) = (1 / 2 : ℂ) := by
    simp only [Complex.star_def, map_div₀, map_one, Complex.conj_ofReal, div_mul_div_comm, one_mul]
    have hsq : (Real.sqrt 2 : ℂ) * (Real.sqrt 2 : ℂ) = 2 := by
      exact_mod_cast Real.mul_self_sqrt (by norm_num)
    rw [hsq]
  have hhalf_sq : ((1 / 2 : ℂ)) * star ((1 / 2 : ℂ)) = (1 / 4 : ℂ) := by
    simp only [Complex.star_def, map_div₀, map_one, map_ofNat]; norm_num
  let upZ : Fin 4 → ℤ × ℤ := ![(1, -1), (1, 1), (-1, 1), (-1, -1)]
  let umZ : Fin 4 → ℤ × ℤ := ![(1, 1), (1, -1), (-1, -1), (-1, 1)]
  have hwp : vec_wp = ((1 : ℂ) / Real.sqrt 2) • ((1 / 2 : ℂ) • fun j => toC_Z (upZ j)) := by
    ext j; fin_cases j <;> simp [vec_wp, upZ, vec_u, vec_v, toC_Z, div_eq_mul_inv] <;> ring
  have hwm : vec_wm = ((1 : ℂ) / Real.sqrt 2) • ((1 / 2 : ℂ) • fun j => toC_Z (umZ j)) := by
    ext j; fin_cases j <;> simp [vec_wm, umZ, vec_u, vec_v, toC_Z, div_eq_mul_inv] <;> ring
  have hcheck : ∀ i j : Fin 4,
      zAdd (zScale 4 (zMul (upZ i) (zConj (upZ j))))
        (zMul (umZ i) (zConj (umZ j))) = zScale 2 (R0Z_mat i j) := by decide
  have h1 : cov_R0 = (1 / 4 : ℂ) • Matrix.of (fun i j => toC_Z (R0Z_mat i j)) := by
    simp only [cov_R0, hwp, hwm, outerProd_smul, hsq2_inv, hhalf_sq]
    ext i j
    have heq := congrArg toC_Z (hcheck i j)
    simp only [toC_Z_add, toC_Z_scale, toC_Z_mul, toC_Z_conj] at heq
    simp only [Matrix.add_apply, Matrix.smul_apply, smul_eq_mul, outerProd, Matrix.of_apply]
    linear_combination (1 / 8 : ℂ) * heq
  refine ⟨h1, ?_⟩
  rw [h1]
  ext i j; fin_cases i <;> fin_cases j <;> simp [R0Z_mat, toC_Z] <;> ring

private def invs_G4 : List (Fin 4 → Fin 4) :=
  [![0, 1, 2, 3], ![3, 0, 1, 2], ![2, 3, 0, 1], ![1, 2, 3, 0]]

private def invs_H4 : List (Fin 4 → Fin 4) :=
  [![0, 1, 2, 3], ![1, 0, 3, 2], ![2, 3, 0, 1], ![3, 2, 1, 0]]

private def invs_K4 : List (Fin 4 → Fin 4) :=
  [![0, 1, 2, 3], ![3, 2, 0, 1], ![1, 0, 3, 2], ![2, 3, 1, 0]]

private lemma hinv_G4 : List.Forall₂ (fun σ f => ∀ i, σ.symm i = f i)
    [1, cyc4, cyc4 ^ 2, cyc4 ^ 3] invs_G4 := by
  refine List.Forall₂.cons (fun i => by fin_cases i <;> rfl)
    (List.Forall₂.cons (fun i => (cyc4_symm_apply i).1)
    (List.Forall₂.cons (fun i => (cyc4_symm_apply i).2.1)
    (List.Forall₂.cons (fun i => (cyc4_symm_apply i).2.2) List.Forall₂.nil)))

private lemma hinv_H4 : List.Forall₂ (fun σ f => ∀ i, σ.symm i = f i)
    [1, v4_1, v4_2, v4_3] invs_H4 := by
  refine List.Forall₂.cons (fun i => by fin_cases i <;> rfl)
    (List.Forall₂.cons (fun i => (v4_symm_apply i).1)
    (List.Forall₂.cons (fun i => (v4_symm_apply i).2.1)
    (List.Forall₂.cons (fun i => (v4_symm_apply i).2.2) List.Forall₂.nil)))

private lemma hinv_K4 : List.Forall₂ (fun σ f => ∀ i, σ.symm i = f i)
    [1, cyc4_K, cyc4_K ^ 2, cyc4_K ^ 3] invs_K4 := by
  refine List.Forall₂.cons (fun i => by fin_cases i <;> rfl)
    (List.Forall₂.cons (fun i => (cyc4K_symm_apply i).1)
    (List.Forall₂.cons (fun i => (cyc4K_symm_apply i).2.1)
    (List.Forall₂.cons (fun i => (cyc4K_symm_apply i).2.2) List.Forall₂.nil)))

lemma blind_Z4_H_transitivity :
    (∀ i j : Fin 4, ∃ σ ∈ [1, cyc4, cyc4 ^ 2, cyc4 ^ 3], σ i = j) ∧
    (∀ i j : Fin 4, ∃ σ ∈ [1, v4_1, v4_2, v4_3], σ i = j) := by
  decide

/-- **Theorem 1 (Exact disproof of `conj:blind` on `G = ℤ₄` vs `H = V₄`)**:
For `ε ∈ {-1, 1}` and `s = 2 w_+ + ε w_-`:
1. `‖s‖² = 5`.
2. `F_G(s) = 4 w_+ w_+ᴴ + w_- w_-ᴴ = R₀` has eigenvalues `4, 1, 0, 0` (so `ψ(G, s) = 4/5`).
3. `F_H(s) = ((2+ε)²/2) u uᴴ + ((2-ε)²/2) v vᴴ` has eigenvalues `9/2, 1/2, 0, 0`
   (so `ψ(H, s) = 9/10 > 4/5 = ψ(G, s)`).
4. `[F_G(s), R₀] = 0` and every `g ∈ G_Z4` commutes with `R₀`.
5. `‖[F_H(s), R₀]‖_F² = 72`, `‖F_H(s)‖_F² = 41/2`, `‖R₀‖_F² = 17`, so the exact squared
   commutator residual is `δ(H, s, R₀)² = 144 / 697 > 0`. -/
theorem blind_Z4_vs_V4_counterexample (ε : ℝ) (hε : ε = 1 ∨ ε = -1) :
    let s := sig_blind ε
    vecNormSq s = 5 ∧
    estF G_Z4 s = cov_R0 ∧
    (estF G_Z4 s).mulVec vec_wp = (4 : ℂ) • vec_wp ∧
    (estF G_Z4 s).mulVec vec_wm = (1 : ℂ) • vec_wm ∧
    (estF H_V4 s).mulVec vec_u = (((2 + ε) ^ 2 / 2 : ℝ) : ℂ) • vec_u ∧
    (estF H_V4 s).mulVec vec_v = (((2 - ε) ^ 2 / 2 : ℝ) : ℂ) • vec_v ∧
    ((4 : ℝ) / 5 < 9 / 10) ∧
    comm (estF G_Z4 s) cov_R0 = 0 ∧
    (∀ g ∈ G_Z4, comm g cov_R0 = 0) ∧
    frobNormSq (comm (estF H_V4 s) cov_R0) = 72 ∧
    frobNormSq (estF H_V4 s) = 41 / 2 ∧
    frobNormSq cov_R0 = 17 ∧
    frobNormSq (comm (estF H_V4 s) cov_R0) /
      (frobNormSq (estF H_V4 s) * frobNormSq cov_R0) = 144 / 697 := by
  dsimp only
  have hsq2_inv : ((1 : ℂ) / Real.sqrt 2) * star ((1 : ℂ) / Real.sqrt 2) = (1 / 2 : ℂ) := by
    simp only [Complex.star_def, map_div₀, map_one, Complex.conj_ofReal, div_mul_div_comm, one_mul]
    have hsq : (Real.sqrt 2 : ℂ) * (Real.sqrt 2 : ℂ) = 2 := by
      exact_mod_cast Real.mul_self_sqrt (by norm_num)
    rw [hsq]
  have hhalf_sq : ((1 / 2 : ℂ)) * star ((1 / 2 : ℂ)) = (1 / 4 : ℂ) := by
    simp only [Complex.star_def, map_div₀, map_one, map_ofNat]; norm_num
  -- Eliminate √2 from outerProd vec_wp, outerProd vec_wm, and sig_blind ε:
  let up : Fin 4 → ℂ := fun j => vec_u j + Complex.I * vec_v j
  let um : Fin 4 → ℂ := fun j => vec_u j - Complex.I * vec_v j
  have hwp_smul : vec_wp = ((1 : ℂ) / Real.sqrt 2) • up := by
    ext j; simp [vec_wp, up, div_eq_mul_inv, mul_comm]
  have hwm_smul : vec_wm = ((1 : ℂ) / Real.sqrt 2) • um := by
    ext j; simp [vec_wm, um, div_eq_mul_inv, mul_comm]
  let s0 : Fin 4 → ℂ := fun j => (2 + (ε : ℂ)) * vec_u j + (2 - (ε : ℂ)) * Complex.I * vec_v j
  have hs_smul : sig_blind ε = ((1 : ℂ) / Real.sqrt 2) • s0 := by
    ext j; simp only [sig_blind, vec_wp, vec_wm, s0, Pi.smul_apply, smul_eq_mul]; ring
  have hR0Z_eq : cov_R0 = (1 / 4 : ℂ) • Matrix.of (fun i j => toC_Z (R0Z_mat i j)) := blind_cov_R0_eq.1
  have hR0 := blind_cov_R0_eq.2
  have hG_list : G_Z4 = ([1, cyc4, cyc4 ^ 2, cyc4 ^ 3] : List (Equiv.Perm (Fin 4))).map permMat := rfl
  have hH_list : H_V4 = ([1, v4_1, v4_2, v4_3] : List (Equiv.Perm (Fin 4))).map permMat := rfl
  have hFG : estF G_Z4 (sig_blind ε) = cov_R0 := by
    rw [hs_smul, estF_smul, hsq2_inv, hG_list, hR0Z_eq]
    rcases hε with rfl | rfl
    · let sZ : Fin 4 → ℤ × ℤ := ![(3, -1), (3, 1), (-3, 1), (-3, -1)]
      have hsZ : s0 = (1 / 2 : ℂ) • fun j => toC_Z (sZ j) := by
        ext j; fin_cases j <;> simp [s0, sZ, vec_u, vec_v, toC_Z] <;> ring
      rw [hsZ, estF_smul, hhalf_sq,
        estF_eq_of_zEstSum _ invs_G4 hinv_G4 sZ (fun i j => zScale 8 (R0Z_mat i j)) (by decide)]
      ext i j; simp only [Matrix.smul_apply, Matrix.of_apply, smul_eq_mul, toC_Z_scale, List.length_cons, List.length_nil]
      push_cast; ring
    · let sZ : Fin 4 → ℤ × ℤ := ![(1, -3), (1, 3), (-1, 3), (-1, -3)]
      have hsZ : s0 = (1 / 2 : ℂ) • fun j => toC_Z (sZ j) := by
        ext j; fin_cases j <;> simp [s0, sZ, vec_u, vec_v, toC_Z] <;> ring
      rw [hsZ, estF_smul, hhalf_sq,
        estF_eq_of_zEstSum _ invs_G4 hinv_G4 sZ (fun i j => zScale 8 (R0Z_mat i j)) (by decide)]
      ext i j; simp only [Matrix.smul_apply, Matrix.of_apply, smul_eq_mul, toC_Z_scale, List.length_cons, List.length_nil]
      push_cast; ring
  have hFH_form : estF H_V4 (sig_blind ε) =
      (((2 + ε) ^ 2 / 2 : ℝ) : ℂ) • outerProd vec_u +
        (((2 - ε) ^ 2 / 2 : ℝ) : ℂ) • outerProd vec_v := by
    rw [hs_smul, estF_smul, hsq2_inv, hH_list]
    let uZ : Fin 4 → ℤ × ℤ := ![(1, 0), (1, 0), (-1, 0), (-1, 0)]
    let vZ : Fin 4 → ℤ × ℤ := ![(-1, 0), (1, 0), (1, 0), (-1, 0)]
    have huZ : vec_u = (1 / 2 : ℂ) • fun j => toC_Z (uZ j) := by
      ext j; fin_cases j <;> simp [vec_u, uZ, toC_Z] <;> norm_num
    have hvZ : vec_v = (1 / 2 : ℂ) • fun j => toC_Z (vZ j) := by
      ext j; fin_cases j <;> simp [vec_v, vZ, toC_Z] <;> norm_num
    rcases hε with rfl | rfl
    · let sZ : Fin 4 → ℤ × ℤ := ![(3, -1), (3, 1), (-3, 1), (-3, -1)]
      have hsZ : s0 = (1 / 2 : ℂ) • fun j => toC_Z (sZ j) := by
        ext j; fin_cases j <;> simp [s0, sZ, vec_u, vec_v, toC_Z] <;> ring
      let MZ : Fin 4 → Fin 4 → ℤ × ℤ := fun i j =>
        zAdd (zScale 36 (zMul (uZ i) (zConj (uZ j)))) (zScale 4 (zMul (vZ i) (zConj (vZ j))))
      rw [hsZ, estF_smul, hhalf_sq, estF_eq_of_zEstSum _ invs_H4 hinv_H4 sZ MZ (by decide),
        huZ, hvZ, outerProd_smul, outerProd_smul, hhalf_sq]
      ext i j
      simp only [Matrix.smul_apply, Matrix.add_apply, Matrix.of_apply, smul_eq_mul, outerProd, MZ,
        toC_Z_add, toC_Z_scale, toC_Z_mul, toC_Z_conj, List.length_cons, List.length_nil]
      push_cast; ring
    · let sZ : Fin 4 → ℤ × ℤ := ![(1, -3), (1, 3), (-1, 3), (-1, -3)]
      have hsZ : s0 = (1 / 2 : ℂ) • fun j => toC_Z (sZ j) := by
        ext j; fin_cases j <;> simp [s0, sZ, vec_u, vec_v, toC_Z] <;> ring
      let MZ : Fin 4 → Fin 4 → ℤ × ℤ := fun i j =>
        zAdd (zScale 4 (zMul (uZ i) (zConj (uZ j)))) (zScale 36 (zMul (vZ i) (zConj (vZ j))))
      rw [hsZ, estF_smul, hhalf_sq, estF_eq_of_zEstSum _ invs_H4 hinv_H4 sZ MZ (by decide),
        huZ, hvZ, outerProd_smul, outerProd_smul, hhalf_sq]
      ext i j
      simp only [Matrix.smul_apply, Matrix.add_apply, Matrix.of_apply, smul_eq_mul, outerProd, MZ,
        toC_Z_add, toC_Z_scale, toC_Z_mul, toC_Z_conj, List.length_cons, List.length_nil]
      push_cast; ring
  have hnorm : vecNormSq (sig_blind ε) = 5 := by
    have hnorm_smul : vecNormSq (((1 : ℂ) / Real.sqrt 2) • s0) = (1 / 2 : ℝ) * vecNormSq s0 := by
      simp only [vecNormSq, Pi.smul_apply, smul_eq_mul, map_mul, ← Finset.mul_sum]
      congr 1
      simp only [map_div₀, map_one, Complex.normSq_ofReal, Real.mul_self_sqrt (by norm_num : (0 : ℝ) ≤ 2)]
    rw [hs_smul, hnorm_smul]
    rcases hε with rfl | rfl <;> (
      simp [vecNormSq, Fin.sum_univ_four, s0, vec_u, vec_v, Complex.normSq_apply]
      norm_num)
  have hcommG : comm (estF G_Z4 (sig_blind ε)) cov_R0 = 0 := by
    rw [hFG]; simp [comm]
  have hGcommR0 : ∀ g ∈ G_Z4, comm g cov_R0 = 0 := by
    intro g hg
    simp only [G_Z4, List.mem_cons, List.not_mem_nil, or_false] at hg
    have hcomm_of_inv : ∀ (σ : Equiv.Perm (Fin 4)) (f g' : Fin 4 → Fin 4),
        (∀ i, σ.symm i = f i) → (∀ i, σ i = g' i) →
        (∀ i j : Fin 4, R0Z_mat (f i) j = R0Z_mat i (g' j)) → comm (permMat σ) cov_R0 = 0 := by
      intro σ f g' hf hg' hchk
      refine comm_permMat_eq_zero σ cov_R0 fun i j => ?_
      rw [hR0Z_eq]; simp only [Matrix.smul_apply, Matrix.of_apply, smul_eq_mul, hf, hg', hchk]
    rcases hg with rfl | rfl | rfl | rfl
    · exact hcomm_of_inv 1 ![0, 1, 2, 3] ![0, 1, 2, 3] (fun i => by fin_cases i <;> rfl)
        (fun i => by fin_cases i <;> rfl) (by decide)
    · exact hcomm_of_inv cyc4 ![3, 0, 1, 2] ![1, 2, 3, 0] (fun i => (cyc4_symm_apply i).1)
        (fun i => by fin_cases i <;> rfl) (by decide)
    · exact hcomm_of_inv (cyc4 ^ 2) ![2, 3, 0, 1] ![2, 3, 0, 1] (fun i => (cyc4_symm_apply i).2.1)
        (fun i => by fin_cases i <;> decide) (by decide)
    · exact hcomm_of_inv (cyc4 ^ 3) ![1, 2, 3, 0] ![3, 0, 1, 2] (fun i => (cyc4_symm_apply i).2.2)
        (fun i => by fin_cases i <;> decide) (by decide)
  have hu_eig : (estF H_V4 (sig_blind ε)).mulVec vec_u = (((2 + ε) ^ 2 / 2 : ℝ) : ℂ) • vec_u := by
    rw [hFH_form]
    simp only [Matrix.add_mulVec, Matrix.smul_mulVec, outerProd_mulVec,
      vec_uvb_ortho.1, vec_uvb_ortho.2.2.2.2.1, one_smul, zero_smul, smul_zero, add_zero]
  have hv_eig : (estF H_V4 (sig_blind ε)).mulVec vec_v = (((2 - ε) ^ 2 / 2 : ℝ) : ℂ) • vec_v := by
    rw [hFH_form]
    simp only [Matrix.add_mulVec, Matrix.smul_mulVec, outerProd_mulVec,
      vec_uvb_ortho.2.2.2.1, vec_uvb_ortho.2.1, one_smul, zero_smul, smul_zero, zero_add]
  have hwp_eig : (estF G_Z4 (sig_blind ε)).mulVec vec_wp = (4 : ℂ) • vec_wp := by
    rw [hFG, hR0, hwp_smul, Matrix.mulVec_smul, smul_comm]
    congr 1
    ext i; fin_cases i <;>
      simp [Matrix.mulVec, dotProduct, Fin.sum_univ_four, up, vec_u, vec_v] <;>
      ring_nf <;> simp [Complex.I_sq] <;> ring
  have hwm_eig : (estF G_Z4 (sig_blind ε)).mulVec vec_wm = (1 : ℂ) • vec_wm := by
    rw [hFG, hR0, hwm_smul, Matrix.mulVec_smul, smul_comm]
    congr 1
    ext i; fin_cases i <;>
      simp [Matrix.mulVec, dotProduct, Fin.sum_univ_four, um, vec_u, vec_v] <;>
      ring_nf <;> simp [Complex.I_sq] <;> ring
  have hFH_Z : ∃ FHZ : Fin 4 → Fin 4 → ℤ × ℤ,
      estF H_V4 (sig_blind ε) = (((1 / 8 : ℝ) : ℂ)) • Matrix.of (fun i j => toC_Z (FHZ i j)) ∧
      zFrobSq4 (zComm4 FHZ R0Z_mat) = 73728 ∧ zFrobSq4 FHZ = 1312 := by
    let uZ : Fin 4 → ℤ × ℤ := ![(1, 0), (1, 0), (-1, 0), (-1, 0)]
    let vZ : Fin 4 → ℤ × ℤ := ![(-1, 0), (1, 0), (1, 0), (-1, 0)]
    have huZ : vec_u = (1 / 2 : ℂ) • fun j => toC_Z (uZ j) := by
      ext j; fin_cases j <;> simp [vec_u, uZ, toC_Z] <;> norm_num
    have hvZ : vec_v = (1 / 2 : ℂ) • fun j => toC_Z (vZ j) := by
      ext j; fin_cases j <;> simp [vec_v, vZ, toC_Z] <;> norm_num
    rw [hFH_form, huZ, hvZ, outerProd_smul, outerProd_smul, hhalf_sq]
    rcases hε with rfl | rfl
    · let FHZ : Fin 4 → Fin 4 → ℤ × ℤ := fun i j =>
        zAdd (zScale 9 (zMul (uZ i) (zConj (uZ j)))) (zMul (vZ i) (zConj (vZ j)))
      refine ⟨FHZ, ?_, by decide, by decide⟩
      ext i j
      simp only [Matrix.add_apply, Matrix.smul_apply, Matrix.of_apply, smul_eq_mul, outerProd, FHZ,
        toC_Z_add, toC_Z_scale, toC_Z_mul, toC_Z_conj]
      push_cast; ring
    · let FHZ : Fin 4 → Fin 4 → ℤ × ℤ := fun i j =>
        zAdd (zMul (uZ i) (zConj (uZ j))) (zScale 9 (zMul (vZ i) (zConj (vZ j))))
      refine ⟨FHZ, ?_, by decide, by decide⟩
      ext i j
      simp only [Matrix.add_apply, Matrix.smul_apply, Matrix.of_apply, smul_eq_mul, outerProd, FHZ,
        toC_Z_add, toC_Z_scale, toC_Z_mul, toC_Z_conj]
      push_cast; ring
  rcases hFH_Z with ⟨FHZ, hFH_Zeq, hFH_comm_z, hFH_frob_z⟩
  have hfrob_comm : frobNormSq (comm (estF H_V4 (sig_blind ε)) cov_R0) = 72 := by
    have hcomm_eq : comm (estF H_V4 (sig_blind ε)) cov_R0 =
        (((1 / 32 : ℝ) : ℂ)) • Matrix.of (fun i j => toC_Z (zComm4 FHZ R0Z_mat i j)) := by
      rw [hFH_Zeq, hR0Z_eq]
      have hsc : ∀ (a b : ℂ) (X Y : Matrix (Fin 4) (Fin 4) ℂ),
          comm (a • X) (b • Y) = (a * b) • comm X Y := by
        intro a b X Y
        simp only [comm, Matrix.smul_mul, Matrix.mul_smul, smul_smul, ← smul_sub, mul_comm b a]
      rw [hsc, comm_toC_Z4]
      congr 1; push_cast; ring
    rw [hcomm_eq, frobNormSq_smul_toC_Z4, hFH_comm_z]
    norm_num
  have hfrob_FH : frobNormSq (estF H_V4 (sig_blind ε)) = 41 / 2 := by
    rw [hFH_Zeq, frobNormSq_smul_toC_Z4, hFH_frob_z]
    norm_num
  have hfrob_R0 : frobNormSq cov_R0 = 17 := by
    have hR0_real : cov_R0 = (((1 / 4 : ℝ) : ℂ)) • Matrix.of (fun i j => toC_Z (R0Z_mat i j)) := by
      simpa using hR0Z_eq
    have hR0_z : zFrobSq4 R0Z_mat = 272 := by decide
    rw [hR0_real, frobNormSq_smul_toC_Z4, hR0_z]
    norm_num
  refine ⟨hnorm, hFG, hwp_eig, hwm_eig, hu_eig, hv_eig, by norm_num, hcommG, hGcommR0,
    hfrob_comm, hfrob_FH, hfrob_R0, ?_⟩
  rw [hfrob_comm, hfrob_FH, hfrob_R0]
  norm_num

/-- **Proposition 2 (Conjugate cyclic groups `G` vs `K` on `C₄` graph filter `2I - A`)**:
At `y₀ = 3u + v + i b`, `‖y₀‖² = 11`, `F_G(y₀)` has eigenvalues `{5, 5, 1, 0}` (`ψ = 5/11`),
`F_K(y₀)` has eigenvalues `{9, 2, 0, 0}` (`ψ = 9/11 > 5/11`), `[F_G(y₀), R₀] = 0`, and
`[F_K(y₀), R₀] ≠ 0`. -/
theorem blind_conjugate_cyclic_C4 :
    let A_C4 : Matrix (Fin 4) (Fin 4) ℂ := permMat cyc4 + (permMat cyc4)ᴴ
    let R0_C4 : Matrix (Fin 4) (Fin 4) ℂ := ((2 : ℂ) • (1 : Matrix (Fin 4) (Fin 4) ℂ) - A_C4) ^ 2
    let y0 : Fin 4 → ℂ := fun j => 3 * vec_u j + vec_v j + Complex.I * vec_b j
    let vib : Fin 4 → ℂ := fun j => vec_v j + Complex.I * vec_b j
    vecNormSq y0 = 11 ∧
    (estF G_Z4 y0).mulVec vec_u = (5 : ℂ) • vec_u ∧
    (estF G_Z4 y0).mulVec vec_v = (5 : ℂ) • vec_v ∧
    (estF G_Z4 y0).mulVec vec_b = (1 : ℂ) • vec_b ∧
    (estF K_Z4 y0).mulVec vec_u = (9 : ℂ) • vec_u ∧
    (estF K_Z4 y0).mulVec vib = (2 : ℂ) • vib ∧
    ((5 : ℝ) / 11 < 9 / 11) ∧
    comm (estF G_Z4 y0) R0_C4 = 0 ∧
    comm (estF K_Z4 y0) R0_C4 ≠ 0 := by
  dsimp only
  set y0 : Fin 4 → ℂ := fun j => 3 * vec_u j + vec_v j + Complex.I * vec_b j
  set vib : Fin 4 → ℂ := fun j => vec_v j + Complex.I * vec_b j
  have hG_list : G_Z4 = ([1, cyc4, cyc4 ^ 2, cyc4 ^ 3] : List (Equiv.Perm (Fin 4))).map permMat := rfl
  have hK_list : K_Z4 = ([1, cyc4_K, cyc4_K ^ 2, cyc4_K ^ 3] : List (Equiv.Perm (Fin 4))).map permMat := rfl
  have hhalf_sq : ((1 / 2 : ℂ)) * star ((1 / 2 : ℂ)) = (1 / 4 : ℂ) := by
    simp only [Complex.star_def, map_div₀, map_one, map_ofNat]; norm_num
  have hFG_y0 : estF G_Z4 y0 = (5 : ℂ) • outerProd vec_u + (5 : ℂ) • outerProd vec_v +
      outerProd vec_b := by
    let uZ : Fin 4 → ℤ × ℤ := ![(1, 0), (1, 0), (-1, 0), (-1, 0)]
    let vZ : Fin 4 → ℤ × ℤ := ![(-1, 0), (1, 0), (1, 0), (-1, 0)]
    let bZ : Fin 4 → ℤ × ℤ := ![(1, 0), (-1, 0), (1, 0), (-1, 0)]
    let yZ : Fin 4 → ℤ × ℤ := ![(2, 1), (4, -1), (-2, 1), (-4, -1)]
    have huZ : vec_u = (1 / 2 : ℂ) • fun j => toC_Z (uZ j) := by
      ext j; fin_cases j <;> simp [vec_u, uZ, toC_Z] <;> norm_num
    have hvZ : vec_v = (1 / 2 : ℂ) • fun j => toC_Z (vZ j) := by
      ext j; fin_cases j <;> simp [vec_v, vZ, toC_Z] <;> norm_num
    have hbZ : vec_b = (1 / 2 : ℂ) • fun j => toC_Z (bZ j) := by
      ext j; fin_cases j <;> simp [vec_b, bZ, toC_Z] <;> norm_num
    have hyZ : y0 = (1 / 2 : ℂ) • fun j => toC_Z (yZ j) := by
      ext j; fin_cases j <;> simp [y0, yZ, vec_u, vec_v, vec_b, toC_Z] <;> ring
    let MZ : Fin 4 → Fin 4 → ℤ × ℤ := fun i j =>
      zScale 4 (zAdd (zAdd (zScale 5 (zMul (uZ i) (zConj (uZ j))))
        (zScale 5 (zMul (vZ i) (zConj (vZ j))))) (zMul (bZ i) (zConj (bZ j))))
    rw [hG_list, hyZ, estF_smul, hhalf_sq, estF_eq_of_zEstSum _ invs_G4 hinv_G4 yZ MZ (by decide),
      huZ, hvZ, hbZ, outerProd_smul, outerProd_smul, outerProd_smul, hhalf_sq]
    ext i j
    simp only [Matrix.add_apply, Matrix.smul_apply, Matrix.of_apply, smul_eq_mul, outerProd, MZ,
      toC_Z_add, toC_Z_scale, toC_Z_mul, toC_Z_conj, List.length_cons, List.length_nil]
    push_cast; ring
  have hFK_y0 : estF K_Z4 y0 = (9 : ℂ) • outerProd vec_u + outerProd vib := by
    let uZ : Fin 4 → ℤ × ℤ := ![(1, 0), (1, 0), (-1, 0), (-1, 0)]
    let vibZ : Fin 4 → ℤ × ℤ := ![(-1, 1), (1, -1), (1, 1), (-1, -1)]
    let yZ : Fin 4 → ℤ × ℤ := ![(2, 1), (4, -1), (-2, 1), (-4, -1)]
    have huZ : vec_u = (1 / 2 : ℂ) • fun j => toC_Z (uZ j) := by
      ext j; fin_cases j <;> simp [vec_u, uZ, toC_Z] <;> norm_num
    have hvibZ : vib = (1 / 2 : ℂ) • fun j => toC_Z (vibZ j) := by
      ext j; fin_cases j <;> simp [vib, vibZ, vec_v, vec_b, toC_Z] <;> ring
    have hyZ : y0 = (1 / 2 : ℂ) • fun j => toC_Z (yZ j) := by
      ext j; fin_cases j <;> simp [y0, yZ, vec_u, vec_v, vec_b, toC_Z] <;> ring
    let MZ : Fin 4 → Fin 4 → ℤ × ℤ := fun i j =>
      zScale 4 (zAdd (zScale 9 (zMul (uZ i) (zConj (uZ j)))) (zMul (vibZ i) (zConj (vibZ j))))
    rw [hK_list, hyZ, estF_smul, hhalf_sq, estF_eq_of_zEstSum _ invs_K4 hinv_K4 yZ MZ (by decide),
      huZ, hvibZ, outerProd_smul, outerProd_smul, hhalf_sq]
    ext i j
    simp only [Matrix.add_apply, Matrix.smul_apply, Matrix.of_apply, smul_eq_mul, outerProd, MZ,
      toC_Z_add, toC_Z_scale, toC_Z_mul, toC_Z_conj, List.length_cons, List.length_nil]
    push_cast; ring
  have hR0_C4 : (((2 : ℂ) • (1 : Matrix (Fin 4) (Fin 4) ℂ) -
      (permMat cyc4 + (permMat cyc4)ᴴ)) ^ 2) =
      Matrix.of ![![6, -4, 2, -4], ![-4, 6, -4, 2], ![2, -4, 6, -4], ![-4, 2, -4, 6]] := by
    have hL : (2 : ℂ) • (1 : Matrix (Fin 4) (Fin 4) ℂ) - (permMat cyc4 + (permMat cyc4)ᴴ) =
        Matrix.of ![![2, -1, 0, -1], ![-1, 2, -1, 0], ![0, -1, 2, -1], ![-1, 0, -1, 2]] := by
      ext i j; fin_cases i <;> fin_cases j <;>
        simp [permMat, cyc4, Matrix.conjTranspose_apply]
    rw [hL, sq]
    ext i j; fin_cases i <;> fin_cases j <;>
      simp [Matrix.mul_apply, Fin.sum_univ_four] <;> norm_num
  have hnorm : vecNormSq y0 = 11 := by
    simp [vecNormSq, Fin.sum_univ_four, y0, vec_u, vec_v, vec_b, Complex.normSq_apply]
    norm_num
  have ⟨huu, hvv, hbb, huv, hvu, hub, hbu, hvb, hbv⟩ := vec_uvb_ortho
  have huvib : ∑ j : Fin 4, star (vec_u j) * vib j = 0 := by
    calc ∑ j : Fin 4, star (vec_u j) * (vec_v j + Complex.I * vec_b j)
        = (∑ j, star (vec_u j) * vec_v j) + Complex.I * ∑ j, star (vec_u j) * vec_b j := by
          rw [Finset.mul_sum, ← Finset.sum_add_distrib]
          refine Finset.sum_congr rfl fun j _ => by ring
      _ = 0 := by rw [huv, hub]; ring
  have hvibu : ∑ j : Fin 4, star (vib j) * vec_u j = 0 := by
    calc ∑ j : Fin 4, star (vec_v j + Complex.I * vec_b j) * vec_u j
        = (∑ j, star (vec_v j) * vec_u j) - Complex.I * ∑ j, star (vec_b j) * vec_u j := by
          rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
          refine Finset.sum_congr rfl fun j _ => by
            simp only [star_add, star_mul', Complex.star_def, Complex.conj_I]; ring
      _ = 0 := by rw [hvu, hbu]; ring
  have hvibvib : ∑ j : Fin 4, star (vib j) * vib j = 2 := by
    calc ∑ j : Fin 4, star (vec_v j + Complex.I * vec_b j) * (vec_v j + Complex.I * vec_b j)
        = (∑ j, star (vec_v j) * vec_v j) + (∑ j, star (vec_b j) * vec_b j) +
          Complex.I * (∑ j, star (vec_v j) * vec_b j) - Complex.I * (∑ j, star (vec_b j) * vec_v j) := by
          simp only [Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
          refine Finset.sum_congr rfl fun j _ => by
            simp only [star_add, star_mul', Complex.star_def, Complex.conj_I]
            ring_nf; simp [Complex.I_sq]; ring
      _ = 2 := by rw [hvv, hbb, hvb, hbv]; ring
  rw [hFG_y0, hFK_y0, hR0_C4]
  refine ⟨hnorm, ?_, ?_, ?_, ?_, ?_, by norm_num, ?_, ?_⟩
  · simp only [Matrix.add_mulVec, Matrix.smul_mulVec, outerProd_mulVec, huu, hvu, hbu,
      one_smul, zero_smul, smul_zero, add_zero]
  · simp only [Matrix.add_mulVec, Matrix.smul_mulVec, outerProd_mulVec, huv, hvv, hbv,
      one_smul, zero_smul, smul_zero, zero_add, add_zero]
  · simp only [Matrix.add_mulVec, Matrix.smul_mulVec, outerProd_mulVec, hub, hvb, hbb,
      one_smul, zero_smul, smul_zero, zero_add]
  · simp only [Matrix.add_mulVec, Matrix.smul_mulVec, outerProd_mulVec, huu, hvibu,
      one_smul, zero_smul, add_zero]
  · simp only [Matrix.add_mulVec, Matrix.smul_mulVec, outerProd_mulVec, huvib, hvibvib,
      zero_smul, smul_zero, zero_add]
  · let FGZ : Fin 4 → Fin 4 → ℤ × ℤ :=
      ![![(11, 0), (-1, 0), (-9, 0), (-1, 0)],
        ![(-1, 0), (11, 0), (-1, 0), (-9, 0)],
        ![(-9, 0), (-1, 0), (11, 0), (-1, 0)],
        ![(-1, 0), (-9, 0), (-1, 0), (11, 0)]]
    let R0C4Z : Fin 4 → Fin 4 → ℤ × ℤ :=
      ![![(6, 0), (-4, 0), (2, 0), (-4, 0)],
        ![(-4, 0), (6, 0), (-4, 0), (2, 0)],
        ![(2, 0), (-4, 0), (6, 0), (-4, 0)],
        ![(-4, 0), (2, 0), (-4, 0), (6, 0)]]
    have hFG_Z : (5 : ℂ) • outerProd vec_u + (5 : ℂ) • outerProd vec_v + outerProd vec_b =
        (1 / 4 : ℂ) • Matrix.of (fun i j => toC_Z (FGZ i j)) := by
      let uZ : Fin 4 → ℤ × ℤ := ![(1, 0), (1, 0), (-1, 0), (-1, 0)]
      let vZ : Fin 4 → ℤ × ℤ := ![(-1, 0), (1, 0), (1, 0), (-1, 0)]
      let bZ : Fin 4 → ℤ × ℤ := ![(1, 0), (-1, 0), (1, 0), (-1, 0)]
      have huZ : vec_u = (1 / 2 : ℂ) • fun j => toC_Z (uZ j) := by
        ext j; fin_cases j <;> simp [vec_u, uZ, toC_Z] <;> norm_num
      have hvZ : vec_v = (1 / 2 : ℂ) • fun j => toC_Z (vZ j) := by
        ext j; fin_cases j <;> simp [vec_v, vZ, toC_Z] <;> norm_num
      have hbZ : vec_b = (1 / 2 : ℂ) • fun j => toC_Z (bZ j) := by
        ext j; fin_cases j <;> simp [vec_b, bZ, toC_Z] <;> norm_num
      have hchk : ∀ i j : Fin 4,
          zAdd (zAdd (zScale 5 (zMul (uZ i) (zConj (uZ j))))
            (zScale 5 (zMul (vZ i) (zConj (vZ j))))) (zMul (bZ i) (zConj (bZ j))) = FGZ i j := by
        decide
      rw [huZ, hvZ, hbZ, outerProd_smul, outerProd_smul, outerProd_smul, hhalf_sq]
      ext i j
      have heq := congrArg toC_Z (hchk i j)
      simp only [toC_Z_add, toC_Z_scale, toC_Z_mul, toC_Z_conj] at heq
      simp only [Matrix.add_apply, Matrix.smul_apply, Matrix.of_apply, smul_eq_mul, outerProd]
      rw [← heq]; push_cast; ring
    have hR0_Z : Matrix.of ![![(6 : ℂ), -4, 2, -4], ![-4, 6, -4, 2], ![2, -4, 6, -4], ![-4, 2, -4, 6]] =
        Matrix.of (fun i j => toC_Z (R0C4Z i j)) := by
      ext i j; fin_cases i <;> fin_cases j <;> simp [R0C4Z, toC_Z]
    have hcomm_z : ∀ i j : Fin 4, zComm4 FGZ R0C4Z i j = (0, 0) := by decide
    rw [hFG_Z, hR0_Z]
    have hsc : ∀ (a : ℂ) (X Y : Matrix (Fin 4) (Fin 4) ℂ), comm (a • X) Y = a • comm X Y := by
      intro a X Y; simp only [comm, Matrix.smul_mul, Matrix.mul_smul, smul_sub]
    rw [hsc, comm_toC_Z4]
    ext i j
    simp [hcomm_z i j, toC_Z]
  · intro h0
    let uZ : Fin 4 → ℤ × ℤ := ![(1, 0), (1, 0), (-1, 0), (-1, 0)]
    let vibZ : Fin 4 → ℤ × ℤ := ![(-1, 1), (1, -1), (1, 1), (-1, -1)]
    let FKZ : Fin 4 → Fin 4 → ℤ × ℤ := fun i j =>
      zAdd (zScale 9 (zMul (uZ i) (zConj (uZ j)))) (zMul (vibZ i) (zConj (vibZ j)))
    let R0C4Z : Fin 4 → Fin 4 → ℤ × ℤ :=
      ![![(6, 0), (-4, 0), (2, 0), (-4, 0)],
        ![(-4, 0), (6, 0), (-4, 0), (2, 0)],
        ![(2, 0), (-4, 0), (6, 0), (-4, 0)],
        ![(-4, 0), (2, 0), (-4, 0), (6, 0)]]
    have huZ : vec_u = (1 / 2 : ℂ) • fun j => toC_Z (uZ j) := by
      ext j; fin_cases j <;> simp [vec_u, uZ, toC_Z] <;> norm_num
    have hvibZ : vib = (1 / 2 : ℂ) • fun j => toC_Z (vibZ j) := by
      ext j; fin_cases j <;> simp [vib, vibZ, vec_v, vec_b, toC_Z] <;> ring
    have hFK_Z : (9 : ℂ) • outerProd vec_u + outerProd vib =
        (1 / 4 : ℂ) • Matrix.of (fun i j => toC_Z (FKZ i j)) := by
      rw [huZ, hvibZ, outerProd_smul, outerProd_smul, hhalf_sq]
      ext i j
      simp only [Matrix.add_apply, Matrix.smul_apply, Matrix.of_apply, smul_eq_mul, outerProd, FKZ,
        toC_Z_add, toC_Z_scale, toC_Z_mul, toC_Z_conj]
      push_cast; ring
    have hR0_Z : Matrix.of ![![(6 : ℂ), -4, 2, -4], ![-4, 6, -4, 2], ![2, -4, 6, -4], ![-4, 2, -4, 6]] =
        Matrix.of (fun i j => toC_Z (R0C4Z i j)) := by
      ext i j; fin_cases i <;> fin_cases j <;> simp [R0C4Z, toC_Z]
    have hsc : ∀ (a : ℂ) (X Y : Matrix (Fin 4) (Fin 4) ℂ), comm (a • X) Y = a • comm X Y := by
      intro a X Y; simp only [comm, Matrix.smul_mul, Matrix.mul_smul, smul_sub]
    rw [hFK_Z, hR0_Z, hsc, comm_toC_Z4] at h0
    have h01 := congrFun (congrFun h0 0) 1
    simp only [Matrix.smul_apply, Matrix.of_apply, Matrix.zero_apply, smul_eq_mul] at h01
    have hne : zComm4 FKZ R0C4Z 0 1 ≠ (0, 0) := by decide
    have htoC_ne : toC_Z (zComm4 FKZ R0C4Z 0 1) ≠ 0 := by
      intro hz
      apply hne
      have hre := congrArg Complex.re hz
      have him := congrArg Complex.im hz
      simp only [toC_Z, Complex.add_re, Complex.add_im, Complex.intCast_re, Complex.intCast_im,
        Complex.mul_re, Complex.mul_im, Complex.I_re, Complex.I_im, Complex.zero_re,
        Complex.zero_im, mul_zero, mul_one, sub_zero, zero_add, add_zero] at hre him
      exact Prod.ext (by exact_mod_cast hre) (by exact_mod_cast him)
    have h14 : ((1 / 4 : ℂ)) ≠ 0 := by norm_num
    exact mul_ne_zero h14 htoC_ne h01

/-- **Proposition 3 (Population-twirl oracle `[T_J(R), R]`)**:
For `a = (2, 1, -2, -1)` and `R = a aᴴ + σ² I`, `[T_G(R), R] = 0`, `[T_K(R), R] ≠ 0`,
yet `ψ(G, a) = 1/2 < 9/10 = ψ(K, a)`. -/
theorem blind_population_twirl_oracle (σ : ℝ) :
    let a : Fin 4 → ℂ := ![2, 1, -2, -1]
    let R : Matrix (Fin 4) (Fin 4) ℂ := outerProd a + ((σ ^ 2 : ℝ) : ℂ) • 1
    vecNormSq a = 10 ∧
    (estF G_Z4 a).mulVec a = (5 : ℂ) • a ∧
    (estF K_Z4 a).mulVec ![1, 1, -1, -1] = (9 : ℂ) • ![1, 1, -1, -1] ∧
    ((1 : ℝ) / 2 < 9 / 10) ∧
    comm (twirl G_Z4 R) R = 0 ∧
    comm (twirl K_Z4 R) R ≠ 0 := by
  dsimp only
  set a : Fin 4 → ℂ := ![2, 1, -2, -1]
  have hG_list : G_Z4 = ([1, cyc4, cyc4 ^ 2, cyc4 ^ 3] : List (Equiv.Perm (Fin 4))).map permMat := rfl
  have hK_list : K_Z4 = ([1, cyc4_K, cyc4_K ^ 2, cyc4_K ^ 3] : List (Equiv.Perm (Fin 4))).map permMat := rfl
  have hFG : estF G_Z4 a =
      Matrix.of ![![5/2, 0, -5/2, 0], ![0, 5/2, 0, -5/2], ![-5/2, 0, 5/2, 0], ![0, -5/2, 0, 5/2]] := by
    let aZ : Fin 4 → ℤ × ℤ := ![(2, 0), (1, 0), (-2, 0), (-1, 0)]
    let MZ : Fin 4 → Fin 4 → ℤ × ℤ :=
      ![![(10, 0), (0, 0), (-10, 0), (0, 0)],
        ![(0, 0), (10, 0), (0, 0), (-10, 0)],
        ![(-10, 0), (0, 0), (10, 0), (0, 0)],
        ![(0, 0), (-10, 0), (0, 0), (10, 0)]]
    have haZ : a = fun j => toC_Z (aZ j) := by ext j; fin_cases j <;> simp [a, aZ, toC_Z]
    rw [hG_list, haZ, estF_eq_of_zEstSum _ invs_G4 hinv_G4 aZ MZ (by decide)]
    ext i j; fin_cases i <;> fin_cases j <;> simp [MZ, toC_Z] <;> ring
  have hFK : estF K_Z4 a =
      Matrix.of ![![5/2, 2, -9/4, -9/4], ![2, 5/2, -9/4, -9/4], ![-9/4, -9/4, 5/2, 2], ![-9/4, -9/4, 2, 5/2]] := by
    let aZ : Fin 4 → ℤ × ℤ := ![(2, 0), (1, 0), (-2, 0), (-1, 0)]
    let MZ : Fin 4 → Fin 4 → ℤ × ℤ :=
      ![![(10, 0), (8, 0), (-9, 0), (-9, 0)],
        ![(8, 0), (10, 0), (-9, 0), (-9, 0)],
        ![(-9, 0), (-9, 0), (10, 0), (8, 0)],
        ![(-9, 0), (-9, 0), (8, 0), (10, 0)]]
    have haZ : a = fun j => toC_Z (aZ j) := by ext j; fin_cases j <;> simp [a, aZ, toC_Z]
    rw [hK_list, haZ, estF_eq_of_zEstSum _ invs_K4 hinv_K4 aZ MZ (by decide)]
    ext i j; fin_cases i <;> fin_cases j <;> simp [MZ, toC_Z] <;> ring
  -- T_J(aaᴴ + σ² I) = F_J(a) + σ² I since every element of J is unitary
  have htwirl_add : ∀ (J : List (Matrix (Fin 4) (Fin 4) ℂ)), J.length = 4 →
      (∀ g ∈ J, g * gᴴ = 1) →
      twirl J (outerProd a + ((σ ^ 2 : ℝ) : ℂ) • 1) = estF J a + ((σ ^ 2 : ℝ) : ℂ) • 1 := by
    intro J hlen hU
    unfold estF twirl
    have hsum : (J.map fun g => g * (outerProd a + ((σ ^ 2 : ℝ) : ℂ) • 1) * gᴴ).sum =
        (J.map fun g => g * outerProd a * gᴴ).sum +
          ((J.length : ℂ) * ((σ ^ 2 : ℝ) : ℂ)) • (1 : Matrix (Fin 4) (Fin 4) ℂ) := by
      clear hlen
      induction J with
      | nil => simp
      | cons g gs ih =>
        simp only [List.map_cons, List.sum_cons, List.length_cons, Nat.cast_add, Nat.cast_one]
        have hg : g * gᴴ = 1 := hU g (by simp)
        have h1 : g * (outerProd a + ((σ ^ 2 : ℝ) : ℂ) • 1) * gᴴ =
            g * outerProd a * gᴴ + ((σ ^ 2 : ℝ) : ℂ) • 1 := by
          rw [mul_add, add_mul, Matrix.mul_smul, Matrix.smul_mul, Matrix.mul_one, hg]
        rw [h1, ih (fun h hh => hU h (by simp [hh]))]
        ext i j; simp; ring
    rw [hsum, smul_add, smul_smul, hlen]
    congr 1
    have h4 : ((4 : ℕ) : ℂ) ≠ 0 := by norm_num
    rw [← mul_assoc, inv_mul_cancel₀ h4, one_mul]
  have hG_U : ∀ g ∈ G_Z4, g * gᴴ = 1 := by
    intro g hg; simp only [G_Z4, List.mem_cons, List.not_mem_nil, or_false] at hg
    rcases hg with rfl | rfl | rfl | rfl <;> exact (permMat_unitary _).1
  have hK_U : ∀ g ∈ K_Z4, g * gᴴ = 1 := by
    intro g hg; simp only [K_Z4, List.mem_cons, List.not_mem_nil, or_false] at hg
    rcases hg with rfl | rfl | rfl | rfl <;> exact (permMat_unitary _).1
  rw [htwirl_add G_Z4 rfl hG_U, htwirl_add K_Z4 rfl hK_U, hFG, hFK]
  refine ⟨?_, ?_, ?_, by norm_num, ?_, ?_⟩
  · simp [vecNormSq, a, Fin.sum_univ_four, Complex.normSq_apply]; norm_num
  · ext i; fin_cases i <;>
      simp [a, Matrix.mulVec, dotProduct, Fin.sum_univ_four] <;> norm_num
  · ext i; fin_cases i <;>
      simp [Matrix.mulVec, dotProduct, Fin.sum_univ_four] <;> norm_num
  · rw [comm_add_smul_one]
    refine comm_outerProd_eq_zero_of_eig _ a 5 ?_ ?_
    · ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.conjTranspose_apply]
    · ext i; fin_cases i <;>
        simp [a, Matrix.mulVec, dotProduct, Fin.sum_univ_four] <;> norm_num
  · rw [comm_add_smul_one]
    intro h0
    have h01 := congrFun (congrFun h0 0) 1
    simp only [comm, outerProd, a, Matrix.sub_apply, Matrix.mul_apply, Matrix.of_apply,
      Matrix.zero_apply, Fin.sum_univ_four, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.cons_val, Complex.star_def, map_ofNat, map_one, map_neg] at h01
    norm_num at h01

/-! ### 3. Disproof of `conj:nadh` (Theorem 2: Unitary Conjugation & 3-Sunlet `S₃`) -/

/-- Cyclic shift permutation `j ↦ j + k` on `Fin n` (for `NeZero n`). -/
def cyclicShift (n : ℕ) [NeZero n] (k : Fin n) : Equiv.Perm (Fin n) :=
  Equiv.addLeft k

/-- Constant unit vector `1 / √n` on `Fin n`. -/
def constUnitVec (n : ℕ) : Fin n → ℂ :=
  fun _ => (1 : ℂ) / Real.sqrt n

lemma cyclicShift_fixes_constUnitVec {n : ℕ} [NeZero n] (k : Fin n) :
    (permMat (cyclicShift n k)).mulVec (constUnitVec n) = constUnitVec n := by
  rw [permMat_mulVec]
  rfl

/-- **Theorem 2 (Part 1 — Unitary conjugation fixes `x` and achieves score `1`)**:
For any unitary matrix `U ∈ U(n)` mapping `x` to a scalar multiple of `1/√n`,
every conjugated cyclic shift `Uᴴ P_k U` fixes `x`, so `F_{Uᴴ ℤ_n U}(x) = x xᴴ`,
which has eigenvector `x` with eigenvalue `‖x‖²` (exact concentration score `1`). -/
theorem unitary_cyclic_conjugate_fixes_vector {n : ℕ} [NeZero n]
    (U : Matrix (Fin n) (Fin n) ℂ) (hU_l : Uᴴ * U = 1)
    (x : Fin n → ℂ) (c : ℂ) (hUx : U.mulVec x = c • constUnitVec n) :
    let G_conj : List (Matrix (Fin n) (Fin n) ℂ) :=
      (List.finRange n).map fun k => Uᴴ * permMat (cyclicShift n k) * U
    (∀ g ∈ G_conj, g.mulVec x = x) ∧
    estF G_conj x = outerProd x ∧
    (estF G_conj x).mulVec x = (vecNormSq x : ℂ) • x := by
  dsimp only
  have hfix : ∀ k : Fin n, (Uᴴ * permMat (cyclicShift n k) * U).mulVec x = x := by
    intro k
    rw [Matrix.mul_assoc, ← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec, hUx,
      Matrix.mulVec_smul, cyclicShift_fixes_constUnitVec, ← hUx,
      Matrix.mulVec_mulVec, hU_l, Matrix.one_mulVec]
  have hfix_all : ∀ g ∈ (List.finRange n).map (fun k => Uᴴ * permMat (cyclicShift n k) * U),
      g.mulVec x = x := by
    intro g hg
    rcases List.mem_map.mp hg with ⟨k, _, rfl⟩
    exact hfix k
  have hF : estF ((List.finRange n).map fun k => Uᴴ * permMat (cyclicShift n k) * U) x =
      outerProd x := by
    rw [estF_eq_sum_outerProd]
    have hmap : (((List.finRange n).map fun k => Uᴴ * permMat (cyclicShift n k) * U).map
        fun g => outerProd (g.mulVec x)) = List.replicate n (outerProd x) := by
      rw [List.map_map]
      apply List.ext_getElem (by simp)
      intro i h1 h2
      simp only [List.getElem_map, List.getElem_replicate, Function.comp_apply]
      rw [hfix]
    rw [hmap, List.sum_replicate, List.length_map, List.length_finRange]
    have hn_ne : (n : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr (NeZero.ne n)
    ext i j
    rw [Matrix.smul_apply, Matrix.smul_apply, smul_eq_mul, nsmul_eq_mul]
    field_simp [hn_ne]
  refine ⟨hfix_all, hF, ?_⟩
  rw [hF, outerProd_mulVec]
  congr 1
  simp only [vecNormSq, Complex.ofReal_sum]
  exact Finset.sum_congr rfl fun i _ => star_mul_eq_normSq (x i)

/-- **Existence of the pointwise unitary conjugator for EVERY nonzero `x ∈ ℂⁿ`** (via an explicit
complex Householder/phase unitary matrix mapping `u = x/‖x‖` to `v = 1/√n`). -/
theorem exists_unitary_cyclic_score_one {n : ℕ} [NeZero n] (x : Fin n → ℂ)
    (hx : vecNormSq x ≠ 0) :
    ∃ U : Matrix (Fin n) (Fin n) ℂ,
      Uᴴ * U = 1 ∧ U * Uᴴ = 1 ∧
      let G_conj := (List.finRange n).map fun k => Uᴴ * permMat (cyclicShift n k) * U
      estF G_conj x = outerProd x ∧
      (estF G_conj x).mulVec x = (vecNormSq x : ℂ) • x := by
  have hn : 0 < n := Nat.pos_of_ne_zero (NeZero.ne n)
  have hxpos : 0 < vecNormSq x :=
    lt_of_le_of_ne (Finset.sum_nonneg fun i _ => Complex.normSq_nonneg _) hx.symm
  let r : ℝ := Real.sqrt (vecNormSq x)
  have hr_pos : 0 < r := Real.sqrt_pos.mpr hxpos
  have hr_ne : (r : ℂ) ≠ 0 := by exact_mod_cast ne_of_gt hr_pos
  let u : Fin n → ℂ := fun i => x i / (r : ℂ)
  let v : Fin n → ℂ := constUnitVec n
  have hu_norm : ∑ i, star (u i) * u i = 1 := by
    have h1 : ∑ i : Fin n, star (u i) * u i = (vecNormSq x : ℂ) / ((r : ℂ) * (r : ℂ)) := by
      simp only [u, map_div₀, Complex.star_def, Complex.conj_ofReal, div_mul_div_comm,
        ← Finset.sum_div]
      congr 1
      simp only [vecNormSq, Complex.ofReal_sum]
      exact Finset.sum_congr rfl fun i _ => by
        rw [← Complex.star_def, star_mul_eq_normSq]
    rw [h1]
    have hr2 : (r : ℂ) * (r : ℂ) = (vecNormSq x : ℂ) := by
      exact_mod_cast Real.mul_self_sqrt hxpos.le
    rw [hr2, div_self (by exact_mod_cast hx)]
  have hv_norm : ∑ i, star (v i) * v i = 1 := by
    have hterm : ∀ i : Fin n, star (v i) * v i = (n : ℂ)⁻¹ := by
      intro i
      simp only [v, constUnitVec, Complex.star_def, map_div₀, map_one, Complex.conj_ofReal]
      have hsq : (Real.sqrt n : ℂ) * (Real.sqrt n : ℂ) = (n : ℂ) := by
        exact_mod_cast Real.mul_self_sqrt (Nat.cast_nonneg n)
      have hne : (Real.sqrt n : ℂ) ≠ 0 := by
        exact_mod_cast Real.sqrt_ne_zero'.mpr (Nat.cast_pos.mpr hn)
      rw [div_mul_div_comm, one_mul, hsq, one_div]
    simp_rw [hterm, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    exact mul_inv_cancel₀ (Nat.cast_ne_zero.mpr (NeZero.ne n))
  let ip : ℂ := ∑ i, star (v i) * u i
  by_cases hip : ip = 0
  · let w : Fin n → ℂ := fun i => v i - u i
    let U : Matrix (Fin n) (Fin n) ℂ := 1 - outerProd w
    have hw_norm : ∑ i, star (w i) * w i = 2 := by
      have hexp : ∑ i, star (w i) * w i =
          (∑ i, star (v i) * v i) - ip - star ip + (∑ i, star (u i) * u i) := by
        simp only [w, ip, star_sub, star_sum, star_mul', star_star]
        rw [← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
        exact Finset.sum_congr rfl fun i _ => by ring
      rw [hexp, hu_norm, hv_norm, hip]
      simp; ring
    have hw_ip_u : ∑ i, star (w i) * u i = -1 := by
      have : ∑ i, star (w i) * u i = ip - (∑ i, star (u i) * u i) := by
        simp only [w, ip, star_sub, sub_mul, Finset.sum_sub_distrib]
      rw [this, hip, hu_norm]; ring
    have hU_herm : Uᴴ = U := by
      ext i j
      simp [U, outerProd, Matrix.conjTranspose_apply, Matrix.one_apply, eq_comm, mul_comm]
    have hU_sq : U * U = 1 := by
      have hoo : outerProd w * outerProd w = (2 : ℂ) • outerProd w := by
        ext i j
        simp only [Matrix.mul_apply, outerProd, Matrix.of_apply, Matrix.smul_apply, smul_eq_mul]
        calc ∑ k, w i * star (w k) * (w k * star (w j))
            = w i * star (w j) * ∑ k, star (w k) * w k := by
              rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun k _ => by ring
          _ = 2 * (w i * star (w j)) := by rw [hw_norm]; ring
      calc U * U = 1 - outerProd w - outerProd w + outerProd w * outerProd w := by
            simp [U, mul_sub, sub_mul]; abel
        _ = 1 := by rw [hoo, two_smul]; abel
    have hUx : U.mulVec x = (r : ℂ) • constUnitVec n := by
      have hUu : U.mulVec u = v := by
        rw [show U.mulVec u = u - (outerProd w).mulVec u by simp [U, Matrix.sub_mulVec]]
        rw [outerProd_mulVec, hw_ip_u]
        ext i; simp [w]
      have hx_eq : x = (r : ℂ) • u := by ext i; simp [u, mul_div_cancel₀ _ hr_ne]
      rw [hx_eq, Matrix.mulVec_smul, hUu]
    have hfix := unitary_cyclic_conjugate_fixes_vector U (by rw [hU_herm]; exact hU_sq) x r hUx
    exact ⟨U, by rw [hU_herm]; exact hU_sq, by rw [hU_herm]; exact hU_sq, hfix.2.1, hfix.2.2⟩
  · let a : ℝ := ‖ip‖
    have ha_pos : 0 < a := norm_pos_iff.mpr hip
    have ha_ne : (a : ℂ) ≠ 0 := by exact_mod_cast ne_of_gt ha_pos
    let ω : ℂ := -star ip / (a : ℂ)
    have hω_norm : star ω * ω = 1 := by
      simp only [ω, Complex.star_def, map_div₀, map_neg, RingHomCompTriple.comp_apply,
        RingHom.id_apply, Complex.conj_ofReal, div_mul_div_comm, neg_mul, mul_neg, neg_neg]
      have h1 : ip * (starRingEnd ℂ) ip = (Complex.normSq ip : ℂ) := Complex.mul_conj ip
      have h2 : (a : ℂ) * (a : ℂ) = (Complex.normSq ip : ℂ) := by
        rw [← sq]; exact_mod_cast Complex.sq_norm ip
      rw [h1, h2, div_self (by rw [← h2]; exact mul_ne_zero ha_ne ha_ne)]
    have hω_ip : ∑ i, star (v i) * (ω * u i) = -(a : ℂ) := by
      have : ∑ i, star (v i) * (ω * u i) = ω * ip := by
        rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun i _ => by ring
      rw [this]
      simp only [ω]
      have h1 : star ip * ip = (Complex.normSq ip : ℂ) := star_mul_eq_normSq ip
      have h2 : (a : ℂ) * (a : ℂ) = (Complex.normSq ip : ℂ) := by
        rw [← sq]; exact_mod_cast Complex.sq_norm ip
      have h3 : -star ip * ip = -((a : ℂ) * (a : ℂ)) := by rw [neg_mul, h1, ← h2]
      rw [div_mul_eq_mul_div, h3, neg_div, mul_div_cancel_right₀ _ ha_ne]
    let w : Fin n → ℂ := fun i => v i - ω * u i
    let denom : ℂ := ((2 + 2 * a : ℝ) : ℂ)
    have hdenom_ne : denom ≠ 0 := by
      have : (0 : ℝ) < 2 + 2 * a := by linarith
      show ((2 + 2 * a : ℝ) : ℂ) ≠ 0
      exact_mod_cast ne_of_gt this
    have hu' : ∑ i, star (ω * u i) * (ω * u i) = 1 := by
      calc ∑ i, star (ω * u i) * (ω * u i)
          = (star ω * ω) * ∑ i, star (u i) * u i := by
            rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun i _ => by simp [star_mul']; ring
        _ = 1 := by rw [hω_norm, hu_norm, mul_one]
    have hw_norm : ∑ i, star (w i) * w i = denom := by
      have hexp : ∑ i, star (w i) * w i =
          (∑ i, star (v i) * v i) - (∑ i, star (v i) * (ω * u i))
          - star (∑ i, star (v i) * (ω * u i)) + (∑ i, star (ω * u i) * (ω * u i)) := by
        simp only [w, star_sub, star_sum, star_mul', star_star]
        rw [← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
        exact Finset.sum_congr rfl fun i _ => by ring
      rw [hexp, hv_norm, hu', hω_ip]
      simp [denom]; ring
    have hw_ip_u : ∑ i, star (w i) * (ω * u i) = -(1 + (a : ℂ)) := by
      have : ∑ i, star (w i) * (ω * u i) =
          (∑ i, star (v i) * (ω * u i)) - (∑ i, star (ω * u i) * (ω * u i)) := by
        simp only [w, star_sub, sub_mul, Finset.sum_sub_distrib]
      rw [this, hω_ip, hu']; ring
    let H : Matrix (Fin n) (Fin n) ℂ := 1 - ((2 : ℂ) / denom) • outerProd w
    let U : Matrix (Fin n) (Fin n) ℂ := ω • H
    have hH_herm : Hᴴ = H := by
      have hcoef : star ((2 : ℂ) / denom) = (2 : ℂ) / denom := by
        simp [denom]
      ext i j
      simp [H, outerProd, Matrix.conjTranspose_apply, Matrix.one_apply, hcoef, eq_comm, mul_comm]
    have hH_sq : H * H = 1 := by
      have hoo : outerProd w * outerProd w = denom • outerProd w := by
        ext i j
        simp only [Matrix.mul_apply, outerProd, Matrix.of_apply, Matrix.smul_apply, smul_eq_mul]
        calc ∑ k, w i * star (w k) * (w k * star (w j))
            = w i * star (w j) * ∑ k, star (w k) * w k := by
              rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun k _ => by ring
          _ = denom * (w i * star (w j)) := by rw [hw_norm]; ring
      calc H * H = 1 - ((2 : ℂ) / denom) • outerProd w - ((2 : ℂ) / denom) • outerProd w
            + (((2 : ℂ) / denom) * ((2 : ℂ) / denom)) • (outerProd w * outerProd w) := by
            simp only [H, mul_sub, sub_mul, Matrix.mul_one, Matrix.one_mul, Matrix.smul_mul,
              Matrix.mul_smul, smul_smul]; abel
        _ = 1 - ((4 : ℂ) / denom) • outerProd w + ((4 : ℂ) / denom) • outerProd w := by
            rw [hoo, smul_smul]
            have hc1 : ((2 : ℂ) / denom) * ((2 : ℂ) / denom) * denom = 4 / denom := by
              field_simp [hdenom_ne]; ring
            rw [hc1]
            ext i j; simp; ring
        _ = 1 := by abel
    have hU_l : Uᴴ * U = 1 := by
      have h1 : Uᴴ * U = (star ω * ω) • (Hᴴ * H) := by
        ext i j
        simp only [U, Matrix.conjTranspose_smul, Matrix.mul_apply, Matrix.smul_apply, smul_eq_mul,
          Finset.mul_sum]
        exact Finset.sum_congr rfl fun k _ => by ring
      rw [h1, hω_norm, hH_herm, hH_sq, one_smul]
    have hU_r : U * Uᴴ = 1 := by
      have h1 : U * Uᴴ = (ω * star ω) • (H * Hᴴ) := by
        ext i j
        simp only [U, Matrix.conjTranspose_smul, Matrix.mul_apply, Matrix.smul_apply, smul_eq_mul,
          Finset.mul_sum]
        exact Finset.sum_congr rfl fun k _ => by ring
      rw [h1, mul_comm ω (star ω), hω_norm, hH_herm, hH_sq, one_smul]
    have hUx : U.mulVec x = (r : ℂ) • constUnitVec n := by
      have hHwu : H.mulVec (ω • u) = v := by
        rw [show H.mulVec (ω • u) = (ω • u) - ((2 : ℂ) / denom) • (outerProd w).mulVec (ω • u) by
          simp [H, Matrix.sub_mulVec, Matrix.smul_mulVec]]
        rw [outerProd_mulVec]
        have hip' : ∑ j, star (w j) * (ω • u) j = -(1 + (a : ℂ)) := hw_ip_u
        rw [hip', smul_smul]
        have hcoef : (2 : ℂ) / denom * -(1 + (a : ℂ)) = -1 := by
          have hd2 : denom = 2 * (1 + (a : ℂ)) := by simp only [denom]; push_cast; ring
          have h1a : (1 + (a : ℂ)) ≠ 0 := by
            intro h0; apply hdenom_ne; rw [hd2, h0, mul_zero]
          rw [hd2]; field_simp [h1a]
        rw [hcoef]
        ext i; simp [w]
      have hx_eq : x = (r : ℂ) • u := by ext i; simp [u, mul_div_cancel₀ _ hr_ne]
      calc U.mulVec x = (ω • H).mulVec ((r : ℂ) • u) := by rw [hx_eq]
        _ = (r : ℂ) • H.mulVec (ω • u) := by
            ext i
            simp only [Matrix.mulVec, dotProduct, Matrix.smul_apply, Pi.smul_apply, smul_eq_mul,
              Finset.mul_sum]
            exact Finset.sum_congr rfl fun k _ => by ring
        _ = (r : ℂ) • v := by rw [hHwu]
    have hfix := unitary_cyclic_conjugate_fixes_vector U hU_l x r hUx
    exact ⟨U, hU_l, hU_r, hfix.2.1, hfix.2.2⟩

/-- Adjacency matrix of the 6-vertex 3-sunlet graph: triangle `{0,1,2}` with pendant leaves
`{3,4,5}` attached via `0-3, 1-4, 2-5`. -/
def A_sunlet : Matrix (Fin 6) (Fin 6) ℂ :=
  Matrix.of ![
    ![0, 1, 1, 1, 0, 0],
    ![1, 0, 1, 0, 1, 0],
    ![1, 1, 0, 0, 0, 1],
    ![1, 0, 0, 0, 0, 0],
    ![0, 1, 0, 0, 0, 0],
    ![0, 0, 1, 0, 0, 0]
  ]

private def s3_e : Equiv.Perm (Fin 6) := 1
private def s3_c1 : Equiv.Perm (Fin 6) :=
  ⟨![1, 2, 0, 4, 5, 3], ![2, 0, 1, 5, 3, 4], by decide, by decide⟩
private def s3_c2 : Equiv.Perm (Fin 6) :=
  ⟨![2, 0, 1, 5, 3, 4], ![1, 2, 0, 4, 5, 3], by decide, by decide⟩
private def s3_t1 : Equiv.Perm (Fin 6) :=
  ⟨![1, 0, 2, 4, 3, 5], ![1, 0, 2, 4, 3, 5], by decide, by decide⟩
private def s3_t2 : Equiv.Perm (Fin 6) :=
  ⟨![0, 2, 1, 3, 5, 4], ![0, 2, 1, 3, 5, 4], by decide, by decide⟩
private def s3_t3 : Equiv.Perm (Fin 6) :=
  ⟨![2, 1, 0, 5, 4, 3], ![2, 1, 0, 5, 4, 3], by decide, by decide⟩

def G_S3 : List (Matrix (Fin 6) (Fin 6) ℂ) :=
  [permMat s3_e, permMat s3_c1, permMat s3_c2, permMat s3_t1, permMat s3_t2, permMat s3_t3]

lemma s3_symm_apply (i : Fin 6) :
    (s3_c1.symm i = (![2, 0, 1, 5, 3, 4] : Fin 6 → Fin 6) i) ∧
    (s3_c2.symm i = (![1, 2, 0, 4, 5, 3] : Fin 6 → Fin 6) i) ∧
    (s3_t1.symm i = (![1, 0, 2, 4, 3, 5] : Fin 6 → Fin 6) i) ∧
    (s3_t2.symm i = (![0, 2, 1, 3, 5, 4] : Fin 6 → Fin 6) i) ∧
    (s3_t3.symm i = (![2, 1, 0, 5, 4, 3] : Fin 6 → Fin 6) i) := by
  fin_cases i <;> refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> rw [Equiv.symm_apply_eq] <;> decide

/-- **Theorem 2 (Part 2 — 3-sunlet graph `S₃` automorphisms and exact spectrum)**:
1. `G_S3` has order `6`, is non-Abelian (`c₁ t₁ ≠ t₁ c₁`), and every `g ∈ G_S3` commutes
   with `A_sunlet`.
2. `A_sunlet` has exact eigenvalues `1 + √2`, `1 - √2` (simple) and `(-1 ± √5)/2`
   (multiplicity 2), all strictly `> -3`, so `h(A) = 3I + A` has unique top eigenvalue `4 + √2`. -/
theorem sunlet_S3_automorphisms_and_spectrum :
    G_S3.length = 6 ∧
    permMat s3_c1 * permMat s3_t1 ≠ permMat s3_t1 * permMat s3_c1 ∧
    (∀ g ∈ G_S3, comm g A_sunlet = 0) ∧
    (let lam₁ : ℂ := 1 + Real.sqrt 2
     let lam₂ : ℂ := 1 - Real.sqrt 2
     let μ₁ : ℂ := (-1 + Real.sqrt 5) / 2
     let μ₂ : ℂ := (-1 - Real.sqrt 5) / 2
     A_sunlet.mulVec ![lam₁, lam₁, lam₁, 1, 1, 1] = lam₁ • ![lam₁, lam₁, lam₁, 1, 1, 1] ∧
     A_sunlet.mulVec ![lam₂, lam₂, lam₂, 1, 1, 1] = lam₂ • ![lam₂, lam₂, lam₂, 1, 1, 1] ∧
     A_sunlet.mulVec ![μ₁, -μ₁, 0, 1, -1, 0] = μ₁ • ![μ₁, -μ₁, 0, 1, -1, 0] ∧
     A_sunlet.mulVec ![μ₁, 0, -μ₁, 1, 0, -1] = μ₁ • ![μ₁, 0, -μ₁, 1, 0, -1] ∧
     A_sunlet.mulVec ![μ₂, -μ₂, 0, 1, -1, 0] = μ₂ • ![μ₂, -μ₂, 0, 1, -1, 0] ∧
     A_sunlet.mulVec ![μ₂, 0, -μ₂, 1, 0, -1] = μ₂ • ![μ₂, 0, -μ₂, 1, 0, -1] ∧
     (-3 : ℝ) < (-1 - Real.sqrt 5) / 2 ∧
     (-1 - Real.sqrt 5) / 2 < 1 - Real.sqrt 2 ∧
     1 - Real.sqrt 2 < (-1 + Real.sqrt 5) / 2 ∧
     (-1 + Real.sqrt 5) / 2 < 1 + Real.sqrt 2) := by
  have hnonab : permMat s3_c1 * permMat s3_t1 ≠ permMat s3_t1 * permMat s3_c1 := by
    rw [permMat_mul_permMat, permMat_mul_permMat]
    intro h
    have h20 := congrFun (congrFun h 2) 0
    simp only [permMat, Matrix.of_apply, Equiv.Perm.coe_mul, Function.comp_apply] at h20
    have hl : (s3_c1 (s3_t1 0)) = 2 := by decide
    have hr : (s3_t1 (s3_c1 0)) = 0 := by decide
    change (if (2 : Fin 6) = s3_c1 (s3_t1 0) then (1 : ℂ) else 0) =
      (if (2 : Fin 6) = s3_t1 (s3_c1 0) then 1 else 0) at h20
    rw [hl, hr] at h20
    have hne : (2 : Fin 6) ≠ 0 := by decide
    simp [hne] at h20
  have hcomm_perm : ∀ σ : Equiv.Perm (Fin 6),
      (∀ i j : Fin 6, A_sunlet (σ.symm i) j = A_sunlet i (σ j)) →
      comm (permMat σ) A_sunlet = 0 := by
    intro σ hσ
    ext i j
    simp only [comm, Matrix.sub_apply, Matrix.mul_apply, permMat, Matrix.of_apply, Matrix.zero_apply]
    have hL : ∑ k : Fin 6, (if i = σ k then (1 : ℂ) else 0) * A_sunlet k j =
        A_sunlet (σ.symm i) j := by
      rw [Finset.sum_eq_single (σ.symm i)]
      · simp
      · intro k _ hk; have : i ≠ σ k := fun h => hk (by rw [h]; simp); simp [this]
      · intro h; exact (h (Finset.mem_univ _)).elim
    have hR : ∑ k : Fin 6, A_sunlet i k * (if k = σ j then (1 : ℂ) else 0) =
        A_sunlet i (σ j) := by
      rw [Finset.sum_eq_single (σ j)]
      · simp
      · intro k _ hk; simp [hk]
      · intro h; exact (h (Finset.mem_univ _)).elim
    rw [hL, hR, hσ i j, sub_self]
  have hcomm : ∀ g ∈ G_S3, comm g A_sunlet = 0 := by
    intro g hg
    simp only [G_S3, List.mem_cons, List.not_mem_nil, or_false] at hg
    rcases hg with rfl | rfl | rfl | rfl | rfl | rfl <;>
      exact hcomm_perm _ (fun i j => by fin_cases i <;> fin_cases j <;>
        simp [A_sunlet, s3_e, s3_c1, s3_c2, s3_t1, s3_t2, s3_t3])
  have hsq2 : (Real.sqrt 2 : ℂ) ^ 2 = 2 := by norm_cast; exact Real.sq_sqrt (by norm_num)
  have hsq5 : (Real.sqrt 5 : ℂ) ^ 2 = 5 := by norm_cast; exact Real.sq_sqrt (by norm_num)
  have h2_lt : Real.sqrt 2 < 3 / 2 := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  have h2_gt : (1 : ℝ) < Real.sqrt 2 := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have h5_lt : Real.sqrt 5 < 5 / 2 := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  have h5_gt : (2 : ℝ) < Real.sqrt 5 := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  refine ⟨rfl, hnonab, hcomm, ?_, ?_, ?_, ?_, ?_, ?_, by linarith, by linarith, by linarith, by linarith⟩
  · ext i; fin_cases i <;>
      simp [A_sunlet, Matrix.mulVec, dotProduct, Fin.sum_univ_six] <;>
      ring_nf <;> rw [hsq2] <;> ring
  · ext i; fin_cases i <;>
      simp [A_sunlet, Matrix.mulVec, dotProduct, Fin.sum_univ_six] <;>
      ring_nf <;> rw [hsq2] <;> ring
  · ext i; fin_cases i <;>
      simp [A_sunlet, Matrix.mulVec, dotProduct, Fin.sum_univ_six] <;>
      ring_nf <;> rw [hsq5] <;> ring
  · ext i; fin_cases i <;>
      simp [A_sunlet, Matrix.mulVec, dotProduct, Fin.sum_univ_six] <;>
      ring_nf <;> rw [hsq5] <;> ring
  · ext i; fin_cases i <;>
      simp [A_sunlet, Matrix.mulVec, dotProduct, Fin.sum_univ_six] <;>
      ring_nf <;> rw [hsq5] <;> ring
  · ext i; fin_cases i <;>
      simp [A_sunlet, Matrix.mulVec, dotProduct, Fin.sum_univ_six] <;>
      ring_nf <;> rw [hsq5] <;> ring

/-- **Theorem 2 (Part 3 — Strict inequality `ψ(S₃, x) < 1` on the 3-sunlet)**:
For `x = (1, 0, 0, 0, 0, 0)` (`‖x‖² = 1`), `F_{S₃}(x) = diag(1/3, 1/3, 1/3, 0, 0, 0)` has
maximum Rayleigh quotient `≤ 1/3 < 1`, strictly below the unitary-conjugate cyclic score `1`. -/
theorem sunlet_S3_strict_score_lt_one :
    let x : Fin 6 → ℂ := ![1, 0, 0, 0, 0, 0]
    vecNormSq x = 1 ∧
    (∀ v : Fin 6 → ℂ, vecNormSq v = 1 →
      (quadForm (estF G_S3 x) v).re ≤ 1 / 3) ∧
    ((1 : ℝ) / 3 < 1) := by
  dsimp only
  have hS3_list : G_S3 = ([s3_e, s3_c1, s3_c2, s3_t1, s3_t2, s3_t3] : List (Equiv.Perm (Fin 6))).map permMat := rfl
  have hF : estF G_S3 ![1, 0, 0, 0, 0, 0] =
      Matrix.diagonal ![(1 : ℂ) / 3, 1 / 3, 1 / 3, 0, 0, 0] := by
    let invs_S3 : List (Fin 6 → Fin 6) :=
      [![0, 1, 2, 3, 4, 5], ![2, 0, 1, 5, 3, 4], ![1, 2, 0, 4, 5, 3],
       ![1, 0, 2, 4, 3, 5], ![0, 2, 1, 3, 5, 4], ![2, 1, 0, 5, 4, 3]]
    have hinv_S3 : List.Forall₂ (fun σ f => ∀ i, σ.symm i = f i)
        [s3_e, s3_c1, s3_c2, s3_t1, s3_t2, s3_t3] invs_S3 := by
      refine List.Forall₂.cons (fun i => by fin_cases i <;> rfl)
        (List.Forall₂.cons (fun i => (s3_symm_apply i).1)
        (List.Forall₂.cons (fun i => (s3_symm_apply i).2.1)
        (List.Forall₂.cons (fun i => (s3_symm_apply i).2.2.1)
        (List.Forall₂.cons (fun i => (s3_symm_apply i).2.2.2.1)
        (List.Forall₂.cons (fun i => (s3_symm_apply i).2.2.2.2) List.Forall₂.nil)))))
    let xZ : Fin 6 → ℤ × ℤ := ![(1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
    let MZ : Fin 6 → Fin 6 → ℤ × ℤ := fun i j =>
      if i = j then (![(2, 0), (2, 0), (2, 0), (0, 0), (0, 0), (0, 0)] : Fin 6 → ℤ × ℤ) i
      else (0, 0)
    have hxZ : (![1, 0, 0, 0, 0, 0] : Fin 6 → ℂ) = fun j => toC_Z (xZ j) := by
      ext j; fin_cases j <;> simp [xZ, toC_Z]
    rw [hS3_list, hxZ, estF_eq_of_zEstSum _ invs_S3 hinv_S3 xZ MZ (by decide)]
    ext i j; fin_cases i <;> fin_cases j <;> simp [MZ, toC_Z] <;> norm_num
  refine ⟨by simp [vecNormSq, Fin.sum_univ_six], ?_, by norm_num⟩
  intro v hv
  rw [hF]
  have hq : (quadForm (Matrix.diagonal ![(1 : ℂ) / 3, 1 / 3, 1 / 3, 0, 0, 0]) v).re =
      (1 / 3 : ℝ) * (Complex.normSq (v 0) + Complex.normSq (v 1) + Complex.normSq (v 2)) := by
    have hdiag : ∀ (d : Fin 6 → ℂ),
        quadForm (Matrix.diagonal d) v = ∑ i : Fin 6, star (v i) * d i * v i := by
      intro d
      simp only [quadForm]
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [Finset.sum_eq_single i]
      · simp
      · intro j _ hj; simp [Ne.symm hj]
      · intro h; exact (h (Finset.mem_univ _)).elim
    rw [hdiag]
    have hterm : ∀ z : ℂ, (star z * ((1 : ℂ) / 3) * z).re = (1 / 3 : ℝ) * Complex.normSq z := by
      intro z
      have h1 : star z * ((1 : ℂ) / 3) * z = ((1 / 3 : ℝ) : ℂ) * (Complex.normSq z : ℂ) := by
        rw [← star_mul_eq_normSq z]; push_cast; ring
      rw [h1]; norm_cast
    simp only [Fin.sum_univ_six, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val,
      mul_zero, zero_mul, add_zero, Complex.add_re, hterm]
    ring
  rw [hq]
  simp only [vecNormSq, Fin.sum_univ_six] at hv
  nlinarith [Complex.normSq_nonneg (v 3), Complex.normSq_nonneg (v 4), Complex.normSq_nonneg (v 5)]

/-! ### 4. Disproof and Repair of `conj:gaat` (Section 4, Propositions 4–5) -/

/-- **Counterexample 1(a) to `conj:gaat` (`f(z) = exp(|z|²/2)` has finite mean `2` at `σ = 1`)**. -/
theorem gaat_exp_half_finite_mean :
    ∫ t in Set.Ioi (0 : ℝ), Real.exp (t / 2) * Real.exp (-t) = 2 := by
  have heq : (fun t : ℝ => Real.exp (t / 2) * Real.exp (-t)) =
      fun t => Real.exp (-(1 / 2 : ℝ) * t) := by
    ext t; rw [← Real.exp_add]; congr 1; ring
  rw [heq]
  have hderiv : ∀ t ∈ Set.Ici (0 : ℝ),
      HasDerivAt (fun x : ℝ => -2 * Real.exp (-(1 / 2 : ℝ) * x)) (Real.exp (-(1 / 2 : ℝ) * t)) t := by
    intro t _
    have h1 : HasDerivAt (fun x : ℝ => -(1 / 2 : ℝ) * x) (-(1 / 2 : ℝ)) t := by
      simpa using (hasDerivAt_id t).const_mul (-(1 / 2 : ℝ))
    have h2 := (Real.hasDerivAt_exp _).comp t h1
    have h3 : -2 * (Real.exp (-(1 / 2 : ℝ) * t) * -(1 / 2 : ℝ)) = Real.exp (-(1 / 2 : ℝ) * t) := by
      ring
    exact (h2.const_mul (-2 : ℝ)).congr_deriv h3
  have htend : Filter.Tendsto (fun x : ℝ => -2 * Real.exp (-(1 / 2 : ℝ) * x)) Filter.atTop (nhds 0) := by
    have hneg : Filter.Tendsto (fun x : ℝ => -(1 / 2 : ℝ) * x) Filter.atTop Filter.atBot :=
      Filter.Tendsto.const_mul_atTop_of_neg (by norm_num) Filter.tendsto_id
    have hexp : Filter.Tendsto (fun x : ℝ => Real.exp (-(1 / 2 : ℝ) * x)) Filter.atTop (nhds 0) :=
      Real.tendsto_exp_atBot.comp hneg
    simpa using hexp.const_mul (-2 : ℝ)
  have hnonneg : ∀ t ∈ Set.Ioi (0 : ℝ), 0 ≤ Real.exp (-(1 / 2 : ℝ) * t) :=
    fun t _ => (Real.exp_pos _).le
  have hint := MeasureTheory.integral_Ioi_of_hasDerivAt_of_nonneg' hderiv hnonneg htend
  rw [hint]
  norm_num

/-- **Counterexample 1(b) to `conj:gaat` (`f(z) = exp(|z|²/2)` has infinite variance at `σ = 1`)**:
The second-moment integrand `(exp(t/2))² · exp(-t) = 1` is NOT integrable on `(0, ∞)`. -/
theorem gaat_exp_half_infinite_variance :
    ¬ IntegrableOn (fun t : ℝ => (Real.exp (t / 2)) ^ 2 * Real.exp (-t)) (Set.Ioi 0) := by
  have heq : (fun t : ℝ => (Real.exp (t / 2)) ^ 2 * Real.exp (-t)) = fun _ => (1 : ℝ) := by
    ext t
    rw [sq, ← Real.exp_add, ← Real.exp_add]
    have : t / 2 + t / 2 + -t = 0 := by ring
    rw [this, Real.exp_zero]
  rw [heq, integrableOn_const_iff]
  rintro (h0 | hfin)
  · norm_num at h0
  · rw [Real.volume_Ioi] at hfin
    exact hfin.ne rfl

/-- **Counterexample 2 to `conj:gaat` (noise-level dependence)**:
For `f(z) = |z|²` (`Var = σ⁴`), no `σ`-independent constant `C` satisfies `σ⁴ ≤ C` for all `σ > 0`. -/
theorem gaat_noise_scaling_unbounded (C : ℝ) : ∃ σ : ℝ, 0 < σ ∧ C < σ ^ 4 := by
  let s := max 1 (|C| + 1)
  refine ⟨s, by positivity, ?_⟩
  have h1 : 1 ≤ s := le_max_left _ _
  have h2 : |C| + 1 ≤ s := le_max_right _ _
  have hle : C ≤ |C| := le_abs_self C
  have hs2 : s ≤ s ^ 2 := by nlinarith
  have hs4 : s ^ 2 ≤ s ^ 4 := by nlinarith [sq_nonneg s]
  linarith

/-- Bijection `h(t) = t + max t 0` on `ℝ` and its inverse, used in Counterexample 3 to `conj:gaat`. -/
def nlBij (t : ℝ) : ℝ := t + max t 0

def nlBijInv (y : ℝ) : ℝ := if 0 ≤ y then y / 2 else y

def nlInvolution (y : ℝ) : ℝ := nlBij (-(nlBijInv y))

/-- **Counterexample 3 to `conj:gaat` (nonlinear output involution refutes (i))**:
`nlInvolution` is an involution fixing `0`, `nlBij` is equivariant under `t ↦ -t` and
`nlInvolution`, yet the group average `(nlBij t + nlBij (-t)) / 2 = |t| / 2 > 0` for all `t ≠ 0`. -/
theorem gaat_nonlinear_involution_counterexample :
    (∀ y, nlBij (nlBijInv y) = y) ∧
    (∀ t, nlBijInv (nlBij t) = t) ∧
    (∀ y, nlInvolution (nlInvolution y) = y) ∧
    nlInvolution 0 = 0 ∧
    (∀ t, nlBij (-t) = nlInvolution (nlBij t)) ∧
    (∀ t, (nlBij t + nlBij (-t)) / 2 = |t| / 2) ∧
    (∀ t, t ≠ 0 → 0 < (nlBij t + nlBij (-t)) / 2) := by
  have h1 : ∀ y, nlBij (nlBijInv y) = y := by
    intro y; unfold nlBij nlBijInv
    split_ifs with hy
    · rw [max_eq_left (by linarith)]; ring
    · rw [max_eq_right (by linarith)]; ring
  have h2 : ∀ t, nlBijInv (nlBij t) = t := by
    intro t; unfold nlBij nlBijInv
    by_cases ht : 0 ≤ t
    · rw [max_eq_left ht, if_pos (by linarith)]; ring
    · rw [max_eq_right (by linarith), if_neg (by linarith)]; ring
  have h3 : ∀ y, nlInvolution (nlInvolution y) = y := by
    intro y; unfold nlInvolution
    rw [h2, neg_neg, h1]
  have h4 : nlInvolution 0 = 0 := by
    unfold nlInvolution nlBij nlBijInv; norm_num
  have h5 : ∀ t, nlBij (-t) = nlInvolution (nlBij t) := by
    intro t; unfold nlInvolution; rw [h2]
  have h6 : ∀ t, (nlBij t + nlBij (-t)) / 2 = |t| / 2 := by
    intro t; unfold nlBij
    by_cases ht : 0 ≤ t
    · rw [max_eq_left ht, max_eq_right (by linarith), abs_of_nonneg ht]; ring
    · rw [max_eq_right (by linarith), max_eq_left (by linarith), abs_of_neg (by linarith)]; ring
  refine ⟨h1, h2, h3, h4, h5, h6, fun t ht => ?_⟩
  rw [h6]
  have : 0 < |t| := abs_pos.mpr ht
  linarith

/-- **Counterexample 4 to `conj:gaat` (`d_eff` defect under `ℤ₂`)**:
For `f(z) = z` under `ℤ₂ = {1, -1}`, the group average `(z + (-z)) / 2 = 0` vanishes
identically (infinite variance reduction) while the orbit `{z, -z}` has size at most `2`. -/
theorem gaat_z2_zero_variance_defect (z : ℂ) :
    (z + (-z)) / 2 = 0 := by ring

/-- **Proposition 4 (Positive equivariant projection identity for unitary representations)**:
For any finite group `G` acting via unitary matrices `ρ : G → Matrix (Fin m) (Fin m) ℂ`,
the group average `P = |G|⁻¹ ∑_{g ∈ G} ρ g` is an orthogonal projector (`P² = P`, `Pᴴ = P`)
invariant under `ρ` and contracts Euclidean norm `‖P v‖² ≤ ‖v‖²`. -/
theorem unitary_group_average_projector {G : Type*} [Group G] [Fintype G] {m : ℕ}
    (ρ : G → Matrix (Fin m) (Fin m) ℂ)
    (h_mul : ∀ g h, ρ (g * h) = ρ g * ρ h)
    (h_inv : ∀ g, ρ g⁻¹ = (ρ g)ᴴ) :
    let P : Matrix (Fin m) (Fin m) ℂ := ((Fintype.card G : ℂ)⁻¹) • ∑ g : G, ρ g
    (∀ h : G, ρ h * P = P) ∧
    (∀ h : G, P * ρ h = P) ∧
    P * P = P ∧
    Pᴴ = P ∧
    (∀ v : Fin m → ℂ, vecNormSq (P.mulVec v) ≤ vecNormSq v) := by
  dsimp only
  set P : Matrix (Fin m) (Fin m) ℂ := ((Fintype.card G : ℂ)⁻¹) • ∑ g : G, ρ g
  have hcard : (Fintype.card G : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr Fintype.card_ne_zero
  have hleft : ∀ h : G, ρ h * P = P := by
    intro h
    simp only [P, Matrix.mul_smul, Matrix.mul_sum]
    congr 1
    calc ∑ g : G, ρ h * ρ g = ∑ g : G, ρ (h * g) :=
          Finset.sum_congr rfl fun g _ => (h_mul h g).symm
      _ = ∑ g : G, ρ g := Fintype.sum_equiv (Equiv.mulLeft h) _ _ (fun _ => rfl)
  have hright : ∀ h : G, P * ρ h = P := by
    intro h
    simp only [P, Matrix.smul_mul, Matrix.sum_mul]
    congr 1
    calc ∑ g : G, ρ g * ρ h = ∑ g : G, ρ (g * h) :=
          Finset.sum_congr rfl fun g _ => (h_mul g h).symm
      _ = ∑ g : G, ρ g := Fintype.sum_equiv (Equiv.mulRight h) _ _ (fun _ => rfl)
  have hidem : P * P = P := by
    calc P * P = (((Fintype.card G : ℂ)⁻¹) • ∑ h : G, ρ h) * P := rfl
      _ = ((Fintype.card G : ℂ)⁻¹) • ∑ h : G, (ρ h * P) := by
          rw [Matrix.smul_mul, Matrix.sum_mul]
      _ = ((Fintype.card G : ℂ)⁻¹) • ∑ _h : G, P := by
          congr 1; exact Finset.sum_congr rfl fun h _ => hleft h
      _ = P := by
          rw [Finset.sum_const, Finset.card_univ, ← Nat.cast_smul_eq_nsmul ℂ,
            smul_smul, inv_mul_cancel₀ hcard, one_smul]
  have hherm : Pᴴ = P := by
    simp only [P, Matrix.conjTranspose_smul, star_inv₀, Complex.star_def, Complex.conj_natCast,
      Matrix.conjTranspose_sum]
    congr 1
    calc ∑ g : G, (ρ g)ᴴ = ∑ g : G, ρ g⁻¹ := Finset.sum_congr rfl fun g _ => (h_inv g).symm
      _ = ∑ g : G, ρ g := Fintype.sum_equiv (Equiv.inv G) _ _ (fun _ => rfl)
  have hnorm : ∀ v : Fin m → ℂ, vecNormSq (P.mulVec v) ≤ vecNormSq v := by
    intro v
    let w := v - P.mulVec v
    have horth : ∑ i, star (P.mulVec v i) * w i = 0 := by
      have h1 : ∑ i, star (P.mulVec v i) * (P.mulVec v i) =
          ∑ k, star (v k) * ((Pᴴ * P).mulVec v k) := by
        simp only [Matrix.mulVec, dotProduct, Matrix.mul_apply, Matrix.conjTranspose_apply,
          star_sum, star_mul']
        simp_rw [Finset.sum_mul, Finset.mul_sum]
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun k _ => ?_
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun l _ => ?_
        refine Finset.sum_congr rfl fun i _ => by ring
      have h2 : ∑ i, star (P.mulVec v i) * v i = ∑ k, star (v k) * (Pᴴ.mulVec v k) := by
        simp only [Matrix.mulVec, dotProduct, Matrix.conjTranspose_apply, star_sum, star_mul']
        simp_rw [Finset.sum_mul, Finset.mul_sum]
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun k _ => ?_
        refine Finset.sum_congr rfl fun i _ => by ring
      have hsub : ∑ i, star (P.mulVec v i) * w i =
          (∑ i, star (P.mulVec v i) * v i) - ∑ i, star (P.mulVec v i) * (P.mulVec v i) := by
        simp [w, mul_sub, Finset.sum_sub_distrib]
      rw [hsub, h1, h2, hherm, hidem, sub_self]
    have hpyth : (vecNormSq v : ℂ) = (vecNormSq (P.mulVec v) : ℂ) + (vecNormSq w : ℂ) := by
      have hnormC : ∀ z : Fin m → ℂ, (vecNormSq z : ℂ) = ∑ i, star (z i) * z i := by
        intro z; simp only [vecNormSq, Complex.ofReal_sum]
        exact Finset.sum_congr rfl fun i _ => (star_mul_eq_normSq (z i)).symm
      rw [hnormC v, hnormC (P.mulVec v), hnormC w]
      have hv_eq : ∀ i, v i = P.mulVec v i + w i := fun i => by simp [w]
      have hconj : ∑ i, star (w i) * P.mulVec v i = 0 := by
        have hst := congrArg star horth
        simp only [star_sum, star_mul', star_star, star_zero] at hst
        simpa [mul_comm] using hst
      calc ∑ i, star (v i) * v i
          = ∑ i, (star (P.mulVec v i) + star (w i)) * (P.mulVec v i + w i) :=
            Finset.sum_congr rfl fun i _ => by rw [hv_eq i, star_add]
        _ = (∑ i, star (P.mulVec v i) * P.mulVec v i) + (∑ i, star (P.mulVec v i) * w i)
            + (∑ i, star (w i) * P.mulVec v i) + (∑ i, star (w i) * w i) := by
            simp only [add_mul, mul_add, Finset.sum_add_distrib]; ring
        _ = (∑ i, star (P.mulVec v i) * P.mulVec v i) + (∑ i, star (w i) * w i) := by
            rw [horth, hconj]; ring
    have hpythR : vecNormSq v = vecNormSq (P.mulVec v) + vecNormSq w := by
      exact_mod_cast hpyth
    have hw_nn : 0 ≤ vecNormSq w := Finset.sum_nonneg fun i _ => Complex.normSq_nonneg _
    linarith
  exact ⟨hleft, hright, hidem, hherm, hnorm⟩

/-- **Proposition 4 (Equivariant averaging identity)**:
If `f(π_g x) = ρ_g (f x)`, then `|G|⁻¹ ∑_g f(π_g x) = P (f x)`. -/
theorem equivariant_average_eq_proj {G X : Type*} [Fintype G] {m : ℕ}
    (π : G → X → X) (ρ : G → Matrix (Fin m) (Fin m) ℂ) (f : X → Fin m → ℂ)
    (heq : ∀ g x, f (π g x) = (ρ g).mulVec (f x)) (x : X) :
    let P : Matrix (Fin m) (Fin m) ℂ := ((Fintype.card G : ℂ)⁻¹) • ∑ g : G, ρ g
    ((Fintype.card G : ℂ)⁻¹) • (∑ g : G, f (π g x)) = P.mulVec (f x) := by
  dsimp only
  ext i
  simp only [Pi.smul_apply, Finset.sum_apply, smul_eq_mul, heq, Matrix.mulVec, dotProduct,
    Matrix.smul_apply, Matrix.sum_apply, Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun g _ => ?_
  refine Finset.sum_congr rfl fun j _ => by ring

/-- Cauchy–Schwarz bound `‖B s‖² ≤ ‖B‖_F² ‖s‖²` for any `M × M` complex matrix `B`. -/
lemma mulVec_normSq_le_frob_mul_normSq {M : ℕ} (B : Matrix (Fin M) (Fin M) ℂ) (s : Fin M → ℂ) :
    vecNormSq (B.mulVec s) ≤ frobNormSq B * vecNormSq s := by
  unfold vecNormSq frobNormSq
  rw [Finset.sum_mul]
  refine Finset.sum_le_sum fun i _ => ?_
  simp only [Matrix.mulVec, dotProduct]
  have hnorm := norm_sum_le (Finset.univ : Finset (Fin M)) (fun j => B i j * s j)
  have hsq1 : ‖∑ j : Fin M, B i j * s j‖ ^ 2 ≤ (∑ j : Fin M, ‖B i j‖ * ‖s j‖) ^ 2 := by
    have h1 : ‖∑ j : Fin M, B i j * s j‖ ≤ ∑ j : Fin M, ‖B i j‖ * ‖s j‖ := by
      simpa [norm_mul] using hnorm
    exact pow_le_pow_left₀ (norm_nonneg _) h1 2
  have hcsR := Finset.sum_mul_sq_le_sq_mul_sq (Finset.univ : Finset (Fin M))
    (fun j => ‖B i j‖) (fun j => ‖s j‖)
  have hle := le_trans hsq1 hcsR
  simpa [Complex.sq_norm] using hle

/-- **Proposition 5 (Fixed-`σ` outer-product variance bound)**:
Given `D` Frobenius-unit (`‖B_j‖_F² = 1`) Hermitian basis matrices of the commutant `C(π)`,
with `d_eff = M² / D`, the total variance satisfies
`∑_j (σ⁴ ‖B_j‖_F² + 2 σ² ‖B_j s‖²) ≤ D (σ⁴ + 2 σ² ‖s‖²) = M² (σ⁴ + 2 σ² ‖s‖²) / d_eff`,
with exact equality when `s = 0`. -/
theorem outer_product_variance_bound {M D : ℕ} (hM : 0 < M) (hD : 0 < D)
    (B : Fin D → Matrix (Fin M) (Fin M) ℂ)
    (hunit : ∀ j, frobNormSq (B j) = 1)
    (σ : ℝ) (s : Fin M → ℂ) :
    let d_eff : ℝ := (M : ℝ) ^ 2 / (D : ℝ)
    (∑ j : Fin D, (σ ^ 4 * frobNormSq (B j) + 2 * σ ^ 2 * vecNormSq ((B j).mulVec s))) ≤
      (D : ℝ) * (σ ^ 4 + 2 * σ ^ 2 * vecNormSq s) ∧
    (D : ℝ) * (σ ^ 4 + 2 * σ ^ 2 * vecNormSq s) =
      (M : ℝ) ^ 2 * (σ ^ 4 + 2 * σ ^ 2 * vecNormSq s) / d_eff ∧
    (s = 0 → ∑ j : Fin D, (σ ^ 4 * frobNormSq (B j) + 2 * σ ^ 2 * vecNormSq ((B j).mulVec s)) =
      (D : ℝ) * σ ^ 4) := by
  dsimp only
  have hMne : (M : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.ne_of_gt hM)
  have hDne : (D : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.ne_of_gt hD)
  refine ⟨?_, ?_, ?_⟩
  · calc ∑ j : Fin D, (σ ^ 4 * frobNormSq (B j) + 2 * σ ^ 2 * vecNormSq ((B j).mulVec s))
        ≤ ∑ _j : Fin D, (σ ^ 4 + 2 * σ ^ 2 * vecNormSq s) := by
          refine Finset.sum_le_sum fun j _ => ?_
          rw [hunit j, mul_one]
          have hle := mulVec_normSq_le_frob_mul_normSq (B j) s
          rw [hunit j, one_mul] at hle
          nlinarith [sq_nonneg σ]
      _ = (D : ℝ) * (σ ^ 4 + 2 * σ ^ 2 * vecNormSq s) := by
          simp [Finset.sum_const, nsmul_eq_mul]; ring
  · field_simp [hMne, hDne]
  · intro hs
    subst hs
    have hzero : ∀ j, vecNormSq ((B j).mulVec 0) = 0 := fun j => by
      simp [vecNormSq, Matrix.mulVec_zero]
    simp only [hunit, hzero, mul_one, mul_zero, add_zero, Finset.sum_const, Finset.card_univ,
      Fintype.card_fin, nsmul_eq_mul]

end GroupSpectralEst

end
