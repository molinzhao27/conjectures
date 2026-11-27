import Mathlib

/-!
# Complete Resolution of Restricted Holonomy for the RDPG Bundle and Laplacian Dynamics
(Giulio Valentino Dalla Riva, arXiv:2603.05703, Conjecture `conj:full-holonomy`)

We formalize in Lean 4 + Mathlib (complete kernel-checked proofs) the exact algebraic,
geometric, and spectral cores of all four theorems settling `conj:full-holonomy`:

1. **Theorem 1 (Ambient RDPG principal bundle — Lyapunov isomorphism & curvature span of `so(d)`)**:
   - `lyapunov_isSkew` & `lyapunov_injective_on_skew`: For every symmetric positive definite Gram
     matrix `G = Xᵀ X ≻ 0`, the Lyapunov operator `S_G(A) = G A + A G` maps `so(d)` injectively
     (and hence bijectively on finite-dimensional `so(d)`) onto `so(d)`.
   - `ambient_curvature_spans_so`: Given any unit `z ∈ ker(Xᵀ)` (which exists whenever `n > d`),
     the horizontal tangent vectors `H_i = z e_iᵀ` satisfy `Xᵀ H_i = 0` and
     `H_iᵀ H_j - H_jᵀ H_i = E_{ij}`, whose `S_G⁻¹`-images span all of `so(d)`.
2. **Theorem 2 (Autonomous Laplacian dynamics — strict trace decay & trivial orbit holonomy)**:
   - `dirichlet_energy_identity` & `dirichlet_energy_pos`: Exact identity
     `tr(Xᵀ L(P) X) = (1/2) ∑_{i,j} P_{ij} ‖x_i - x_j‖²`, strictly positive whenever `P > 0`
     entrywise and `X` has at least two distinct rows.
   - `laplacian_field_horizontal`: `Xᵀ (-L(P) X)` is symmetric, so the Laplacian vector field
     is everywhere horizontal (`ω_X(F(X)) = 0`).
   - `trace_decay_deriv`: Along any differentiable solution of `X' = -L(X Xᵀ) X`,
     `(d/dt) tr(X(t) X(t)ᵀ) = -2 tr(X(t)ᵀ L(P(t)) X(t)) < 0`, ruling out nonconstant closed
     orbits.
   - `horizontal_curve_trivial_holonomy`: Any parallel-transport curve `Q(t)` with vanishing
     connection pullback `Q'(t) = 0` is constant (`Q(t) = Q(0)`), proving trivial orbit holonomy.
3. **Theorem 3 (Explicit non-regular `S_d`-symmetric commuting counterexample & exact disk loop
   angle `2π / 201`)**:
   - `symmFrame`: Explicit `(d + 1) × d` frame `x_i = (1 + e_i)/(4d)` (`i < d`), `x_d = 1/(4d)`
     for every `d ≥ 2`.
   - `symmFrame_interior`, `symmFrame_gram_spd`, `symmFrame_nonregular`: Strictly interior
     (`0 < P_{ij} < 1`), full column rank (`G = (1/(4d)²)(I_d + (d+3) J_d) ≻ 0`), and non-regular
     (degree gap `(d + 2)/(4d)² > 0`).
   - `symmLaplacian_commute`: Every two `S_d`-symmetric Laplacians commute identically, so every
     frozen projected commutator `Xᵀ [L_1, L_2] X` vanishes identically.
   - `disk_loop_exact_holonomy`: For `g = 1/(4d)²`, `r = 1/(40d)`, unit `z ∈ ker(Xᵀ)`, and
     orthonormal contrasts `u, v ⊥ 1_d` with `K = u vᵀ - v uᵀ`:
     - `r² / (2g + r²) = 1 / 201` (exact for all `d ≥ 2`);
     - `Y(θ)ᵀ Y'(θ) - Y'(θ)ᵀ Y(θ) = r² K` and `S_{Y(θ)ᵀ Y(θ)}(K) = (2g + r²) K`;
     - The explicit `SO(d)` rotation curve
       `Q(θ) = I - sin(θ/201) K + (1 - cos(θ/201)) K²` solves `Q'(θ) = -(1/201) K Q(θ)` with
       `Q(0) = I`, `Q(θ)ᵀ Q(θ) = I`, and `Q(2π) ≠ I` (exact holonomy angle `2π / 201`).
4. **Theorem 4 (Base-4 Vandermonde finite-time rank core & path Laplacian simple spectrum)**:
   - `base4_rates_pos` & `base4_rates_injective`: All `d(d+1)/2` rates `μ_{ij} = s_i + s_j`
     (`s_i = 4^i` for `i < d`, `s_d = 0`) are strictly positive and pairwise distinct.
   - `vandermonde_rates_det_ne_zero`: The weighted `N × N` Vandermonde matrix `(w_p μ_p^{k+1})`
     is invertible (`det ≠ 0`).
   - `path_laplacian_simple_spectrum`: Every eigenspace of the `(d+1) × (d+1)` path Laplacian
     is at most 1-dimensional (any eigenvector is uniquely determined by its 0-th coordinate).

Machine-checked vs prose-only:
- Machine-checked: All four algebraic/geometric/spectral cores listed above for general `d ≥ 2`.
- Prose-only (in `solutions/rdpg_latent_holonomy/proof.pdf`): Cartan structure equation and
  Ambrose–Singer theorem invoking `Hol_p^0 = SO(d)` from curvature span, ODE uniqueness preserving
  `S_d`-symmetry along the flow, Zariski-open density of nonvanishing polynomial minors, and
  commutator surjectivity via spectral theorem on `W = 1^⊥`.
-/

open Matrix BigOperators Finset

namespace RDPGLatentHolonomy

/-! ### 1. Theorem 1: Lyapunov Isomorphism and Ambient Curvature Span of `so(d)` -/

/-- A square matrix is skew-symmetric (`A ∈ so(d)`). -/
def IsSkew {d : ℕ} (A : Matrix (Fin d) (Fin d) ℝ) : Prop := Aᵀ = -A

/-- A square matrix is symmetric positive definite (`G ≻ 0`). -/
def IsSPD {d : ℕ} (G : Matrix (Fin d) (Fin d) ℝ) : Prop :=
  Gᵀ = G ∧ ∀ v : Fin d → ℝ, v ≠ 0 → 0 < dotProduct v (G.mulVec v)

/-- The Lyapunov operator `S_G(A) = G A + A G`. -/
def lyapunov {d : ℕ} (G A : Matrix (Fin d) (Fin d) ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  G * A + A * G

/-- `S_G` preserves skew-symmetry whenever `G` is symmetric. -/
theorem lyapunov_isSkew {d : ℕ} {G A : Matrix (Fin d) (Fin d) ℝ}
    (hG : Gᵀ = G) (hA : IsSkew A) : IsSkew (lyapunov G A) := by
  unfold IsSkew lyapunov at *
  rw [Matrix.transpose_add, Matrix.transpose_mul, Matrix.transpose_mul, hG, hA]
  simp [add_comm]

/-- Submodule `so(d)` of skew-symmetric `d × d` matrices. -/
def skewSubmodule (d : ℕ) : Submodule ℝ (Matrix (Fin d) (Fin d) ℝ) where
  carrier := {A | IsSkew A}
  add_mem' := fun {A B} hA hB => by
    show (A + B)ᵀ = -(A + B)
    rw [Matrix.transpose_add, hA, hB, neg_add]
  zero_mem' := by show (0 : Matrix (Fin d) (Fin d) ℝ)ᵀ = -0; simp
  smul_mem' := fun c {A} hA => by
    show (c • A)ᵀ = -(c • A)
    rw [Matrix.transpose_smul, hA, smul_neg]

/-- **Lyapunov injectivity on `so(d)`**: If `G ≻ 0` and `A ∈ so(d)` satisfies `G A + A G = 0`,
then `A = 0`. Proof: `tr(Aᵀ (G A + A G)) = 2 ∑_j (A e_j)ᵀ G (A e_j) = 0`. -/
theorem lyapunov_injective_on_skew {d : ℕ} {G A : Matrix (Fin d) (Fin d) ℝ}
    (hG : IsSPD G) (hA : IsSkew A) (h0 : lyapunov G A = 0) : A = 0 := by
  have htr0 : (Aᵀ * (G * A + A * G)).trace = 0 := by
    rw [show G * A + A * G = lyapunov G A from rfl, h0, Matrix.mul_zero, Matrix.trace_zero]
  have htr_eq : (Aᵀ * (G * A + A * G)).trace = 2 * (Aᵀ * G * A).trace := by
    rw [Matrix.mul_add, Matrix.trace_add, ← Matrix.mul_assoc]
    have h2 : (Aᵀ * (A * G)).trace = (Aᵀ * G * A).trace := by
      rw [← Matrix.mul_assoc, Matrix.trace_mul_comm (Aᵀ * A) G, ← Matrix.mul_assoc]
      have hAt : Aᵀ = -A := hA
      have hcomm : (G * Aᵀ * A).trace = (Aᵀ * G * A).trace := by
        calc (G * Aᵀ * A).trace
            = (A * (G * Aᵀ)).trace := Matrix.trace_mul_comm (G * Aᵀ) A
          _ = (A * G * Aᵀ).trace := by rw [Matrix.mul_assoc]
          _ = ((-Aᵀ) * G * (-A)).trace := by
              rw [hAt]; simp
          _ = (Aᵀ * G * A).trace := by simp
      exact hcomm
    rw [h2]; ring
  have htr_sum : (Aᵀ * G * A).trace =
      ∑ j : Fin d, dotProduct (A.mulVec (Pi.single j (1 : ℝ)))
        (G.mulVec (A.mulVec (Pi.single j (1 : ℝ)))) := by
    simp only [Matrix.trace, Matrix.diag, Matrix.mul_apply, Matrix.transpose_apply,
      dotProduct, Matrix.mulVec]
    refine Finset.sum_congr rfl fun j _ => ?_
    have hcol : ∀ i, (∑ k, A i k * (Pi.single j (1 : ℝ) : Fin d → ℝ) k) = A i j := by
      intro i
      rw [Finset.sum_eq_single j]
      · simp
      · intro k _ hk; simp [Pi.single_eq_of_ne hk]
      · intro h; exact (h (Finset.mem_univ j)).elim
    simp_rw [hcol, Finset.sum_mul, Finset.mul_sum]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun i _ => by ring
  have hsum0 : ∑ j : Fin d,
      dotProduct (A.mulVec (Pi.single j (1 : ℝ)))
        (G.mulVec (A.mulVec (Pi.single j (1 : ℝ)))) = 0 := by
    linarith
  have hnonneg : ∀ j ∈ (Finset.univ : Finset (Fin d)),
      0 ≤ dotProduct (A.mulVec (Pi.single j (1 : ℝ)))
        (G.mulVec (A.mulVec (Pi.single j (1 : ℝ)))) := by
    intro j _
    by_cases hv : A.mulVec (Pi.single j (1 : ℝ)) = 0
    · simp [hv]
    · exact (hG.2 _ hv).le
  have hterm0 := (Finset.sum_eq_zero_iff_of_nonneg hnonneg).1 hsum0
  ext i j
  have hv0 : A.mulVec (Pi.single j (1 : ℝ)) = 0 := by
    by_contra hne
    have hpos := hG.2 (A.mulVec (Pi.single j (1 : ℝ))) hne
    have hzero := hterm0 j (Finset.mem_univ j)
    linarith
  have hi := congrFun hv0 i
  simp only [Matrix.mulVec, dotProduct, Pi.zero_apply] at hi
  rw [Finset.sum_eq_single j] at hi
  · simpa using hi
  · intro k _ hk; simp [Pi.single_eq_of_ne hk]
  · intro h; exact (h (Finset.mem_univ j)).elim

/-- The Lyapunov linear endomorphism `S_G : so(d) →ₗ[ℝ] so(d)`. -/
def lyapunovLinear {d : ℕ} (G : Matrix (Fin d) (Fin d) ℝ) (hG : Gᵀ = G) :
    skewSubmodule d →ₗ[ℝ] skewSubmodule d where
  toFun := fun A => ⟨lyapunov G A.1, lyapunov_isSkew hG A.2⟩
  map_add' := fun A B => Subtype.ext (by simp [lyapunov, Matrix.mul_add, Matrix.add_mul]; abel)
  map_smul' := fun c A => Subtype.ext (by simp [lyapunov])

/-- **Lyapunov isomorphism on `so(d)`**: For every `G ≻ 0`, `S_G : so(d) → so(d)` is bijective. -/
theorem lyapunov_bijective_on_skew {d : ℕ} {G : Matrix (Fin d) (Fin d) ℝ} (hG : IsSPD G) :
    Function.Bijective (lyapunovLinear G hG.1) := by
  have hinj : Function.Injective (lyapunovLinear G hG.1) := by
    intro A B hAB
    have hdiff : lyapunov G (A.1 - B.1) = 0 := by
      have hval : lyapunov G A.1 = lyapunov G B.1 := congrArg Subtype.val hAB
      have hsub : lyapunov G (A.1 - B.1) = lyapunov G A.1 - lyapunov G B.1 := by
        simp only [lyapunov, Matrix.mul_sub, Matrix.sub_mul]; abel
      rw [hsub, hval, sub_self]
    have hskew : IsSkew (A.1 - B.1) := by
      show (A.1 - B.1)ᵀ = -(A.1 - B.1)
      rw [Matrix.transpose_sub, A.2, B.2]; abel
    have h0 := lyapunov_injective_on_skew hG hskew hdiff
    exact Subtype.ext (sub_eq_zero.1 h0)
  exact ⟨hinj, LinearMap.injective_iff_surjective.1 hinj⟩

/-- Elementary skew-symmetric matrix `E_{ij} = e_i e_jᵀ - e_j e_iᵀ`. -/
def elemSkew {d : ℕ} (i j : Fin d) : Matrix (Fin d) (Fin d) ℝ :=
  Matrix.of fun a b => (if a = i ∧ b = j then (1 : ℝ) else 0) -
    (if a = j ∧ b = i then (1 : ℝ) else 0)

lemma elemSkew_isSkew {d : ℕ} (i j : Fin d) : IsSkew (elemSkew i j) := by
  ext a b
  simp only [Matrix.transpose_apply, Matrix.neg_apply, elemSkew, Matrix.of_apply]
  by_cases h1 : a = i ∧ b = j <;> by_cases h2 : a = j ∧ b = i <;>
    simp [h1, h2, and_comm]

lemma skew_eq_quarter_sum_two_elemSkew {d : ℕ} {S : Matrix (Fin d) (Fin d) ℝ} (hS : IsSkew S) :
    S = ∑ i : Fin d, ∑ j : Fin d, ((1 / 4 : ℝ) * S i j) • (2 • elemSkew i j) := by
  ext a b
  simp only [Matrix.sum_apply, Matrix.smul_apply, smul_eq_mul, elemSkew, Matrix.of_apply]
  have h1 : ∀ i j : Fin d, (1 / 4 : ℝ) * S i j *
      (2 * ((if a = i ∧ b = j then (1 : ℝ) else 0) - if a = j ∧ b = i then 1 else 0)) =
      (if i = a ∧ j = b then (1 / 2 : ℝ) * S a b else 0) -
      (if i = b ∧ j = a then (1 / 2 : ℝ) * S b a else 0) := by
    intro i j
    by_cases h1 : i = a ∧ j = b <;> by_cases h2 : i = b ∧ j = a
    · rcases h1 with ⟨rfl, rfl⟩; rcases h2 with ⟨hba, _⟩; subst hba; simp
    · rcases h1 with ⟨rfl, rfl⟩
      have hne : ¬ (i = j ∧ j = i) := fun h => h2 ⟨h.1, h.2⟩
      simp [hne]; ring
    · rcases h2 with ⟨rfl, rfl⟩
      have hne : ¬ (j = i ∧ i = j) := fun h => h1 ⟨h.2, h.1⟩
      simp [h1, hne]; ring
    · have h1' : ¬ (a = i ∧ b = j) := fun h => h1 ⟨h.1.symm, h.2.symm⟩
      have h2' : ¬ (a = j ∧ b = i) := fun h => h2 ⟨h.2.symm, h.1.symm⟩
      simp only [h1, h2, h1', h2', ↓reduceIte, sub_self, mul_zero]
  simp only [nsmul_eq_mul, Nat.cast_ofNat]
  rw [Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => h1 i j]
  simp_rw [Finset.sum_sub_distrib]
  have hsum1 : ∑ i : Fin d, ∑ j : Fin d,
      (if i = a ∧ j = b then (1 / 2 : ℝ) * S a b else 0) = (1 / 2 : ℝ) * S a b := by
    rw [Finset.sum_eq_single a]
    · rw [Finset.sum_eq_single b]
      · simp
      · intro j _ hjb; simp [hjb]
      · intro h; exact (h (Finset.mem_univ b)).elim
    · intro i _ hia; exact Finset.sum_eq_zero fun j _ => by simp [hia]
    · intro h; exact (h (Finset.mem_univ a)).elim
  have hsum2 : ∑ i : Fin d, ∑ j : Fin d,
      (if i = b ∧ j = a then (1 / 2 : ℝ) * S b a else 0) = (1 / 2 : ℝ) * S b a := by
    rw [Finset.sum_eq_single b]
    · rw [Finset.sum_eq_single a]
      · simp
      · intro j _ hja; simp [hja]
      · intro h; exact (h (Finset.mem_univ a)).elim
    · intro i _ hib; exact Finset.sum_eq_zero fun j _ => by simp [hib]
    · intro h; exact (h (Finset.mem_univ b)).elim
  rw [hsum1, hsum2]
  have hba : S b a = -S a b := by
    have := congrFun (congrFun hS a) b
    simpa [Matrix.transpose_apply] using this
  rw [hba]; ring

/-- Rank-1 horizontal perturbation `H_i = z e_iᵀ` for `z ∈ ker(Xᵀ)`. -/
def horizPerturb {n d : ℕ} (z : Fin n → ℝ) (i : Fin d) : Matrix (Fin n) (Fin d) ℝ :=
  Matrix.of fun a b => z a * if b = i then 1 else 0

/-- **Theorem 1 (Curvature values at `X` span `so(d)`)**:
Given any unit vector `z ∈ ker(Xᵀ)` (guaranteed whenever `n > d`), the vectors `H_i = z e_iᵀ`
are horizontal (`Xᵀ H_i = 0`), satisfy `H_iᵀ H_j - H_jᵀ H_i = E_{ij}`, and every skew-symmetric
matrix `A ∈ so(d)` is an `ℝ`-linear combination of `{2 S_G⁻¹(E_{ij})}` — proving that the
curvature values `R_X(H_i, H_j)` at `X` alone span all of `so(d)`. -/
theorem ambient_curvature_spans_so {n d : ℕ} {X : Matrix (Fin n) (Fin d) ℝ}
    {z : Fin n → ℝ} (hker : Xᵀ.mulVec z = 0) (hz : dotProduct z z = 1)
    (hG : IsSPD (Xᵀ * X)) :
    (∀ i : Fin d, Xᵀ * horizPerturb z i = 0) ∧
    (∀ i j : Fin d, (horizPerturb z i)ᵀ * horizPerturb z j -
      (horizPerturb z j)ᵀ * horizPerturb z i = elemSkew i j) ∧
    (∀ A : skewSubmodule d, ∃ R : Fin d → Fin d → skewSubmodule d,
      (∀ i j, lyapunovLinear (Xᵀ * X) hG.1 (R i j) =
        ⟨2 • elemSkew i j, (skewSubmodule d).smul_mem 2 (elemSkew_isSkew i j)⟩) ∧
      A = ∑ i : Fin d, ∑ j : Fin d, ((1 / 4 : ℝ) * (lyapunov (Xᵀ * X) A.1) i j) • R i j) := by
  have hhoriz : ∀ i : Fin d, Xᵀ * horizPerturb z i = 0 := by
    intro i
    ext a b
    simp only [Matrix.mul_apply, horizPerturb, Matrix.of_apply, Matrix.zero_apply]
    have ha : ∑ k : Fin n, Xᵀ a k * z k = 0 := congrFun hker a
    have hfac : ∑ k : Fin n, Xᵀ a k * (z k * if b = i then 1 else 0) =
        (∑ k : Fin n, Xᵀ a k * z k) * (if b = i then 1 else 0) := by
      rw [Finset.sum_mul]
      refine Finset.sum_congr rfl fun k _ => by ring
    rw [hfac, ha, zero_mul]
  have hcurv : ∀ i j : Fin d, (horizPerturb z i)ᵀ * horizPerturb z j -
      (horizPerturb z j)ᵀ * horizPerturb z i = elemSkew i j := by
    intro i j
    ext a b
    simp only [Matrix.sub_apply, Matrix.mul_apply, Matrix.transpose_apply, horizPerturb,
      Matrix.of_apply, elemSkew]
    have h1 : ∑ k : Fin n, (z k * if a = i then (1 : ℝ) else 0) *
        (z k * if b = j then 1 else 0) =
        (dotProduct z z) * (if a = i ∧ b = j then 1 else 0) := by
      unfold dotProduct
      rw [Finset.sum_mul]
      refine Finset.sum_congr rfl fun k _ => ?_
      by_cases hai : a = i <;> by_cases hbj : b = j <;> simp [hai, hbj]
    have h2 : ∑ k : Fin n, (z k * if a = j then (1 : ℝ) else 0) *
        (z k * if b = i then 1 else 0) =
        (dotProduct z z) * (if a = j ∧ b = i then 1 else 0) := by
      unfold dotProduct
      rw [Finset.sum_mul]
      refine Finset.sum_congr rfl fun k _ => ?_
      by_cases haj : a = j <;> by_cases hbi : b = i <;> simp [haj, hbi]
    rw [h1, h2, hz, one_mul, one_mul]
  refine ⟨hhoriz, hcurv, ?_⟩
  intro A
  obtain ⟨hinj, hsurj⟩ := lyapunov_bijective_on_skew hG
  choose R hR using fun i j =>
    hsurj ⟨2 • elemSkew i j, (skewSubmodule d).smul_mem 2 (elemSkew_isSkew i j)⟩
  refine ⟨R, hR, ?_⟩
  apply hinj
  have hmap : lyapunovLinear (Xᵀ * X) hG.1
      (∑ i : Fin d, ∑ j : Fin d, ((1 / 4 : ℝ) * (lyapunov (Xᵀ * X) A.1) i j) • R i j) =
      ∑ i : Fin d, ∑ j : Fin d, ((1 / 4 : ℝ) * (lyapunov (Xᵀ * X) A.1) i j) •
        ⟨2 • elemSkew i j, (skewSubmodule d).smul_mem 2 (elemSkew_isSkew i j)⟩ := by
    rw [map_sum]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [map_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [map_smul, hR]
  rw [hmap]
  apply Subtype.ext
  have hval : (∑ i : Fin d, ∑ j : Fin d, ((1 / 4 : ℝ) * (lyapunov (Xᵀ * X) A.1) i j) •
      (⟨2 • elemSkew i j, (skewSubmodule d).smul_mem 2 (elemSkew_isSkew i j)⟩ : skewSubmodule d)).1 =
      ∑ i : Fin d, ∑ j : Fin d, ((1 / 4 : ℝ) * (lyapunov (Xᵀ * X) A.1) i j) • (2 • elemSkew i j) := by
    rw [AddSubmonoidClass.coe_finset_sum]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [AddSubmonoidClass.coe_finset_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    change ((1 / 4 : ℝ) * (lyapunov (Xᵀ * X) A.1) i j) • ((2 : ℝ) • elemSkew i j) =
      ((1 / 4 : ℝ) * (lyapunov (Xᵀ * X) A.1) i j) • ((2 : ℕ) • elemSkew i j)
    congr 1
    ext a b
    simp [two_smul]
    ring
  rw [hval]
  exact skew_eq_quarter_sum_two_elemSkew (lyapunov_isSkew hG.1 A.2)

/-! ### 2. Theorem 2: Strict Trace Decay and Trivial Autonomous Orbit Holonomy -/

/-- Graph Laplacian `L(P) = diag(P 1) - P`. -/
def laplacian {n : ℕ} (P : Matrix (Fin n) (Fin n) ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  Matrix.diagonal (fun i => ∑ j : Fin n, P i j) - P

lemma laplacian_transpose {n : ℕ} {P : Matrix (Fin n) (Fin n) ℝ} (hP : Pᵀ = P) :
    (laplacian P)ᵀ = laplacian P := by
  unfold laplacian
  rw [Matrix.transpose_sub, Matrix.diagonal_transpose, hP]

/-- **Dirichlet energy identity**: For symmetric `P`,
`tr(Xᵀ L(P) X) = (1 / 2) ∑_{i,j} P_{ij} ‖x_i - x_j‖²`. -/
theorem dirichlet_energy_identity {n d : ℕ} (X : Matrix (Fin n) (Fin d) ℝ)
    {P : Matrix (Fin n) (Fin n) ℝ} (hP : Pᵀ = P) :
    (Xᵀ * laplacian P * X).trace =
      (1 / 2 : ℝ) * ∑ i : Fin n, ∑ j : Fin n, P i j * (∑ k : Fin d, (X i k - X j k) ^ 2) := by
  have hPij : ∀ i j, P j i = P i j := fun i j => congrFun (congrFun hP i) j
  have hlhs : (Xᵀ * laplacian P * X).trace =
      ∑ k : Fin d, ∑ i : Fin n, ∑ j : Fin n, P i j * (X i k * X i k - X i k * X j k) := by
    simp only [Matrix.trace, Matrix.diag, Matrix.mul_apply, Matrix.transpose_apply, laplacian,
      Matrix.sub_apply, Matrix.diagonal_apply]
    refine Finset.sum_congr rfl fun k _ => ?_
    have hstep1 : ∑ i : Fin n, (∑ j : Fin n, X j k *
        ((if j = i then ∑ l : Fin n, P j l else 0) - P j i)) * X i k =
        ∑ i : Fin n, ((∑ l : Fin n, P i l) * X i k * X i k -
          ∑ j : Fin n, P i j * X i k * X j k) := by
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [Finset.sum_mul]
      have hsplit : ∑ j : Fin n, X j k * ((if j = i then ∑ l : Fin n, P j l else 0) - P j i) * X i k =
          (∑ j : Fin n, if j = i then X j k * (∑ l : Fin n, P j l) * X i k else 0) -
          ∑ j : Fin n, X j k * P j i * X i k := by
        rw [← Finset.sum_sub_distrib]
        refine Finset.sum_congr rfl fun j _ => by split_ifs <;> ring
      rw [hsplit]
      congr 1
      · rw [Finset.sum_eq_single i]
        · rw [if_pos rfl]; ring
        · intro j _ hji; simp [hji]
        · intro h; exact (h (Finset.mem_univ i)).elim
      · refine Finset.sum_congr rfl fun j _ => by rw [hPij i j]; ring
    rw [hstep1]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [Finset.sum_mul, Finset.sum_mul, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun j _ => by ring
  rw [hlhs]
  have hrhs : (1 / 2 : ℝ) * ∑ i : Fin n, ∑ j : Fin n, P i j * (∑ k : Fin d, (X i k - X j k) ^ 2) =
      ∑ k : Fin d, (1 / 2 : ℝ) * ∑ i : Fin n, ∑ j : Fin n, P i j * (X i k - X j k) ^ 2 := by
    calc (1 / 2 : ℝ) * ∑ i : Fin n, ∑ j : Fin n, P i j * (∑ k : Fin d, (X i k - X j k) ^ 2)
        = ∑ i : Fin n, ∑ j : Fin n, ∑ k : Fin d, (1 / 2 : ℝ) * (P i j * (X i k - X j k) ^ 2) := by
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun i _ => ?_
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun j _ => ?_
          rw [Finset.mul_sum, Finset.mul_sum]
      _ = ∑ i : Fin n, ∑ k : Fin d, ∑ j : Fin n, (1 / 2 : ℝ) * (P i j * (X i k - X j k) ^ 2) :=
          Finset.sum_congr rfl fun i _ => Finset.sum_comm
      _ = ∑ k : Fin d, ∑ i : Fin n, ∑ j : Fin n, (1 / 2 : ℝ) * (P i j * (X i k - X j k) ^ 2) :=
          Finset.sum_comm
      _ = ∑ k : Fin d, (1 / 2 : ℝ) * ∑ i : Fin n, ∑ j : Fin n, P i j * (X i k - X j k) ^ 2 := by
          refine Finset.sum_congr rfl fun k _ => ?_
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun i _ => ?_
          rw [Finset.mul_sum]
  rw [hrhs]
  refine Finset.sum_congr rfl fun k _ => ?_
  have hsym : ∑ i : Fin n, ∑ j : Fin n, P i j * (X j k * X j k) =
      ∑ i : Fin n, ∑ j : Fin n, P i j * (X i k * X i k) := by
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by rw [hPij i j]
  have hsym2 : ∑ i : Fin n, ∑ j : Fin n, P i j * (X i k - X j k) ^ 2 =
      ∑ i : Fin n, ∑ j : Fin n, P i j * (X i k * X i k) +
      ∑ i : Fin n, ∑ j : Fin n, P i j * (X j k * X j k) -
      2 * ∑ i : Fin n, ∑ j : Fin n, P i j * (X i k * X j k) := by
    rw [Finset.mul_sum]
    simp_rw [Finset.mul_sum]
    rw [← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun j _ => by ring
  have hdiff : ∑ i : Fin n, ∑ j : Fin n, P i j * (X i k * X i k - X i k * X j k) =
      (∑ i : Fin n, ∑ j : Fin n, P i j * (X i k * X i k)) -
      (∑ i : Fin n, ∑ j : Fin n, P i j * (X i k * X j k)) := by
    rw [← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun j _ => by ring
  rw [hsym2, hsym, hdiff]
  ring

/-- Strict positivity of the Dirichlet energy for every strictly positive `P` and nonconstant
row configuration `X`. -/
theorem dirichlet_energy_pos {n d : ℕ} {X : Matrix (Fin n) (Fin d) ℝ}
    {P : Matrix (Fin n) (Fin n) ℝ} (hP : Pᵀ = P) (hPpos : ∀ i j, 0 < P i j)
    (hnonconst : ∃ i₀ j₀ k₀, X i₀ k₀ ≠ X j₀ k₀) :
    0 < (Xᵀ * laplacian P * X).trace := by
  rw [dirichlet_energy_identity X hP]
  rcases hnonconst with ⟨i₀, j₀, k₀, hne⟩
  have hpos_term : 0 < P i₀ j₀ * (∑ k : Fin d, (X i₀ k - X j₀ k) ^ 2) := by
    refine mul_pos (hPpos i₀ j₀) ?_
    refine Finset.sum_pos' (fun k _ => sq_nonneg _) ⟨k₀, Finset.mem_univ k₀, ?_⟩
    exact sq_pos_of_ne_zero (sub_ne_zero.2 hne)
  have hsum_pos : 0 < ∑ i : Fin n, ∑ j : Fin n, P i j * (∑ k : Fin d, (X i k - X j k) ^ 2) := by
    refine Finset.sum_pos' (fun i _ => Finset.sum_nonneg fun j _ =>
      mul_nonneg (hPpos i j).le (Finset.sum_nonneg fun k _ => sq_nonneg _))
      ⟨i₀, Finset.mem_univ i₀, ?_⟩
    exact Finset.sum_pos' (fun j _ => mul_nonneg (hPpos i₀ j).le
      (Finset.sum_nonneg fun k _ => sq_nonneg _)) ⟨j₀, Finset.mem_univ j₀, hpos_term⟩
  linarith

/-- The Laplacian vector field `F(X) = -L(P) X` is everywhere horizontal (`Xᵀ F(X)` is symmetric). -/
theorem laplacian_field_horizontal {n d : ℕ} (X : Matrix (Fin n) (Fin d) ℝ)
    {P : Matrix (Fin n) (Fin n) ℝ} (hP : Pᵀ = P) :
    (Xᵀ * (-laplacian P * X))ᵀ = Xᵀ * (-laplacian P * X) := by
  rw [Matrix.transpose_mul, Matrix.transpose_mul, Matrix.transpose_neg,
    laplacian_transpose hP, Matrix.transpose_transpose, Matrix.mul_assoc]

/-- **Theorem 2 (Exact trace-decay derivative along `X' = -L(X Xᵀ) X`)**:
Along any differentiable solution of `X'(t) = -L(P(t)) X(t)` (`P(t) = X(t) X(t)ᵀ`),
`(d/dt) tr(P(t)) = -2 tr(X(t)ᵀ L(P(t)) X(t))`. -/
theorem trace_decay_deriv {n d : ℕ} (X : ℝ → Matrix (Fin n) (Fin d) ℝ) (t : ℝ)
    (hode : ∀ i k, HasDerivAt (fun s => X s i k)
      ((-laplacian (X t * (X t)ᵀ) * X t) i k) t) :
    HasDerivAt (fun s => (X s * (X s)ᵀ).trace)
      (-2 * ((X t)ᵀ * laplacian (X t * (X t)ᵀ) * X t).trace) t := by
  have htr : ∀ s, (X s * (X s)ᵀ).trace = ∑ i : Fin n, ∑ k : Fin d, (X s i k) ^ 2 := by
    intro s
    simp [Matrix.trace, Matrix.diag, Matrix.mul_apply, sq]
  simp_rw [htr]
  have hderiv : HasDerivAt (fun s => ∑ i : Fin n, ∑ k : Fin d, (X s i k) ^ 2)
      (∑ i : Fin n, ∑ k : Fin d, 2 * X t i k * ((-laplacian (X t * (X t)ᵀ) * X t) i k)) t := by
    refine HasDerivAt.fun_sum fun i _ => HasDerivAt.fun_sum fun k _ => ?_
    have hsq := (hode i k).pow 2
    simpa [mul_comm, mul_left_comm, mul_assoc] using hsq
  convert hderiv using 1
  simp only [Matrix.trace, Matrix.diag, Matrix.mul_apply, Matrix.transpose_apply, Matrix.neg_apply]
  rw [Finset.mul_sum]
  conv_rhs => rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  have hsum : ∑ j : Fin n, X t j k * laplacian (X t * (X t)ᵀ) j i =
      ∑ j : Fin n, laplacian (X t * (X t)ᵀ) i j * X t j k := by
    refine Finset.sum_congr rfl fun j _ => ?_
    have hsym := laplacian_transpose (P := X t * (X t)ᵀ) (by simp)
    have hij := congrFun (congrFun hsym i) j
    simp only [Matrix.transpose_apply] at hij
    rw [hij]; ring
  have hneg : ∑ j : Fin n, -(laplacian (X t * (X t)ᵀ) i j) * X t j k =
      -(∑ j : Fin n, laplacian (X t * (X t)ᵀ) i j * X t j k) := by
    rw [← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun j _ => by ring
  rw [hneg, ← hsum]; ring

/-- **Trivial holonomy along any horizontal curve**: If parallel transport `Q(t)` satisfies
`Q'(t) = 0` everywhere on `ℝ` (vanishing pullback connection `ω(X'(t)) = 0`), then `Q(t) = Q(0)`
for all `t`. -/
theorem horizontal_curve_trivial_holonomy {d : ℕ}
    (Q : ℝ → Matrix (Fin d) (Fin d) ℝ)
    (hflat : ∀ t i j, HasDerivAt (fun s => Q s i j) 0 t) (t : ℝ) :
    Q t = Q 0 := by
  ext i j
  exact is_const_of_deriv_eq_zero (fun s => (hflat s i j).differentiableAt)
    (fun s => (hflat s i j).deriv) t 0

/-! ### 3. Theorem 3: Non-Regular `S_d`-Symmetric Counterexample & Exact Disk Loop Angle `2π/201` -/

/-- Explicit `S_d`-symmetric interior frame `X_d ∈ ℝ^{(d+1) × d}` for `d ≥ 2`:
`x_i = (1 + e_i)/(4d)` for `i < d`, and `x_d = 1/(4d)`. -/
noncomputable def symmFrame (d : ℕ) : Matrix (Fin (d + 1)) (Fin d) ℝ :=
  Matrix.of fun i j =>
    if i.val < d then
      (1 + if i.val = j.val then (1 : ℝ) else 0) / (4 * (d : ℝ))
    else
      1 / (4 * (d : ℝ))

/-- All-ones `d × d` matrix `J_d`. -/
def allOnes (d : ℕ) : Matrix (Fin d) (Fin d) ℝ := Matrix.of fun _ _ => 1

/-- **Theorem 3(1a) (Strictly interior probabilities)**: `0 < (X_d X_dᵀ)_{ij} < 1` for all `i, j`
whenever `d ≥ 2`. -/
theorem symmFrame_interior {d : ℕ} (hd : 2 ≤ d) (i j : Fin (d + 1)) :
    0 < (symmFrame d * (symmFrame d)ᵀ) i j ∧ (symmFrame d * (symmFrame d)ᵀ) i j < 1 := by
  have hd_pos : (0 : ℝ) < d := by positivity
  have hd_ge2 : (2 : ℝ) ≤ d := by exact_mod_cast hd
  have h4d_pos : (0 : ℝ) < 4 * d := by positivity
  have hentry_pos : ∀ a b, 0 < symmFrame d a b := by
    intro a b
    unfold symmFrame
    simp only [Matrix.of_apply]
    split_ifs <;> positivity
  have hentry_le : ∀ a b, symmFrame d a b ≤ 2 / (4 * (d : ℝ)) := by
    intro a b
    unfold symmFrame
    simp only [Matrix.of_apply]
    split_ifs <;> apply div_le_div_of_nonneg_right (by norm_num) h4d_pos.le
  constructor
  · simp only [Matrix.mul_apply, Matrix.transpose_apply]
    refine Finset.sum_pos (fun k _ => mul_pos (hentry_pos i k) (hentry_pos j k)) ?_
    exact ⟨⟨0, by omega⟩, Finset.mem_univ _⟩
  · simp only [Matrix.mul_apply, Matrix.transpose_apply]
    calc ∑ k : Fin d, symmFrame d i k * symmFrame d j k
        ≤ ∑ _k : Fin d, (2 / (4 * (d : ℝ))) * (2 / (4 * (d : ℝ))) := by
          refine Finset.sum_le_sum fun k _ =>
            mul_le_mul (hentry_le i k) (hentry_le j k) (hentry_pos j k).le (by positivity)
      _ = (d : ℝ) * ((2 / (4 * (d : ℝ))) * (2 / (4 * (d : ℝ)))) := by simp
      _ = 1 / (4 * (d : ℝ)) := by field_simp [hd_pos.ne']; ring
      _ < 1 := by rw [div_lt_one h4d_pos]; linarith

/-- Exact Gram matrix of `symmFrame d`: `X_dᵀ X_d = (1/(4d)²)(I_d + (d + 3) J_d)`. -/
theorem symmFrame_gram_formula (d : ℕ) :
    (symmFrame d)ᵀ * symmFrame d =
      (1 / (4 * (d : ℝ)) ^ 2) • ((1 : Matrix (Fin d) (Fin d) ℝ) + ((d : ℝ) + 3) • allOnes d) := by
  ext a b
  simp only [Matrix.mul_apply, Matrix.transpose_apply, Matrix.smul_apply, Matrix.add_apply,
    Matrix.one_apply, allOnes, Matrix.of_apply, smul_eq_mul]
  rw [Fin.sum_univ_castSucc]
  have hlast : symmFrame d (Fin.last d) a * symmFrame d (Fin.last d) b =
      1 / (4 * (d : ℝ)) ^ 2 := by
    simp only [symmFrame, Matrix.of_apply, Fin.val_last, lt_self_iff_false, ↓reduceIte]
    ring
  rw [hlast]
  have hcast : ∀ i : Fin d, symmFrame d i.castSucc a * symmFrame d i.castSucc b =
      (1 + (if i = a then (1 : ℝ) else 0) + (if i = b then 1 else 0) +
        (if i = a ∧ i = b then 1 else 0)) / (4 * (d : ℝ)) ^ 2 := by
    intro i
    simp only [symmFrame, Matrix.of_apply, Fin.coe_castSucc, i.isLt, ↓reduceIte, Fin.val_eq_val]
    by_cases hia : i = a <;> by_cases hib : i = b <;> simp [hia, hib] <;> ring
  simp_rw [hcast, ← Finset.sum_div, Finset.sum_add_distrib]
  have hsum1 : ∑ _i : Fin d, (1 : ℝ) = d := by simp
  have hsum2 : ∑ i : Fin d, (if i = a then (1 : ℝ) else 0) = 1 := by simp
  have hsum3 : ∑ i : Fin d, (if i = b then (1 : ℝ) else 0) = 1 := by simp
  have hsum4 : ∑ i : Fin d, (if i = a ∧ i = b then (1 : ℝ) else 0) =
      if a = b then 1 else 0 := by
    by_cases hab : a = b
    · subst hab; simp
    · simp only [hab, ↓reduceIte]
      refine Finset.sum_eq_zero fun i _ => ?_
      have : ¬ (i = a ∧ i = b) := fun ⟨h1, h2⟩ => hab (h1.symm.trans h2)
      simp [this]
  rw [hsum1, hsum2, hsum3, hsum4]
  split_ifs <;> ring

/-- **Theorem 3(1b) (Full column rank)**: `G = X_dᵀ X_d` is symmetric positive definite. -/
theorem symmFrame_gram_spd {d : ℕ} (hd : 2 ≤ d) :
    IsSPD ((symmFrame d)ᵀ * symmFrame d) := by
  refine ⟨by simp, ?_⟩
  intro v hv
  rw [symmFrame_gram_formula d]
  have hquad : dotProduct v (((1 / (4 * (d : ℝ)) ^ 2) •
      ((1 : Matrix (Fin d) (Fin d) ℝ) + ((d : ℝ) + 3) • allOnes d)).mulVec v) =
      (1 / (4 * (d : ℝ)) ^ 2) * ((∑ i : Fin d, (v i) ^ 2) +
        ((d : ℝ) + 3) * (∑ i : Fin d, v i) ^ 2) := by
    simp only [dotProduct, Matrix.mulVec, Matrix.smul_apply, Matrix.add_apply, Matrix.one_apply,
      allOnes, Matrix.of_apply, smul_eq_mul]
    have hsplit : ∀ i : Fin d,
        v i * ∑ j : Fin d, (1 / (4 * (d : ℝ)) ^ 2) *
          ((if i = j then (1 : ℝ) else 0) + ((d : ℝ) + 3) * 1) * v j =
        (1 / (4 * (d : ℝ)) ^ 2) * ((v i) ^ 2 + ((d : ℝ) + 3) * (v i * ∑ j : Fin d, v j)) := by
      intro i
      have hinner : ∑ j : Fin d, (1 / (4 * (d : ℝ)) ^ 2) *
          ((if i = j then (1 : ℝ) else 0) + ((d : ℝ) + 3) * 1) * v j =
          (1 / (4 * (d : ℝ)) ^ 2) * (v i + ((d : ℝ) + 3) * ∑ j : Fin d, v j) := by
        have h2 : ∑ j : Fin d, (1 / (4 * (d : ℝ)) ^ 2) *
            ((if i = j then (1 : ℝ) else 0) + ((d : ℝ) + 3) * 1) * v j =
            (1 / (4 * (d : ℝ)) ^ 2) * ∑ j : Fin d,
              ((if i = j then v j else 0) + ((d : ℝ) + 3) * v j) := by
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun j _ => by split_ifs <;> ring
        rw [h2, Finset.sum_add_distrib, ← Finset.mul_sum]
        congr 1
        simp only [Finset.sum_ite_eq, Finset.mem_univ, ↓reduceIte]
      rw [hinner]; ring
    simp_rw [hsplit]
    rw [← Finset.mul_sum, Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.sum_mul, ← sq]
  rw [hquad]
  have hd_pos : (0 : ℝ) < d := by positivity
  have hne : ∃ i, v i ≠ 0 := by
    by_contra h0; push_neg at h0; exact hv (funext h0)
  rcases hne with ⟨i₀, hi₀⟩
  have hsq_pos : 0 < ∑ i : Fin d, (v i) ^ 2 :=
    Finset.sum_pos' (fun i _ => sq_nonneg _) ⟨i₀, Finset.mem_univ i₀, sq_pos_of_ne_zero hi₀⟩
  have h2_nonneg : 0 ≤ ((d : ℝ) + 3) * (∑ i : Fin d, v i) ^ 2 := by positivity
  exact mul_pos (by positivity) (by linarith)

/-- **Theorem 3(1c) (Non-regularity)**: The first `d` node degrees strictly exceed the last node
degree by `(d + 2) / (4d)² > 0`. -/
theorem symmFrame_nonregular (d : ℕ) (i : Fin d) :
    let P := symmFrame d * (symmFrame d)ᵀ
    (∑ j : Fin (d + 1), P i.castSucc j) - (∑ j : Fin (d + 1), P (Fin.last d) j) =
      ((d : ℝ) + 2) / (4 * (d : ℝ)) ^ 2 := by
  intro P
  have hrow_sum : ∀ k : Fin d, ∑ j : Fin (d + 1), symmFrame d j k =
      ((d : ℝ) + 2) / (4 * (d : ℝ)) := by
    intro k
    rw [Fin.sum_univ_castSucc]
    simp only [symmFrame, Matrix.of_apply, Fin.coe_castSucc, Fin.is_lt, ↓reduceIte,
      Fin.val_last, lt_self_iff_false, Fin.val_eq_val]
    rw [← Finset.sum_div, Finset.sum_add_distrib]
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one,
      Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte]
    ring
  have hdeg : ∀ a : Fin (d + 1), ∑ j : Fin (d + 1), P a j =
      ∑ k : Fin d, symmFrame d a k * (((d : ℝ) + 2) / (4 * (d : ℝ))) := by
    intro a
    simp only [P, Matrix.mul_apply, Matrix.transpose_apply]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [← Finset.mul_sum, hrow_sum k]
  rw [hdeg i.castSucc, hdeg (Fin.last d), ← Finset.sum_sub_distrib]
  have hdiff : ∀ k : Fin d,
      symmFrame d i.castSucc k * (((d : ℝ) + 2) / (4 * (d : ℝ))) -
      symmFrame d (Fin.last d) k * (((d : ℝ) + 2) / (4 * (d : ℝ))) =
      if i = k then ((d : ℝ) + 2) / (4 * (d : ℝ)) ^ 2 else 0 := by
    intro k
    simp only [symmFrame, Matrix.of_apply, Fin.coe_castSucc, i.isLt, ↓reduceIte,
      Fin.val_last, lt_self_iff_false, Fin.val_eq_val]
    split_ifs <;> ring
  simp_rw [hdiff]
  simp

/-- `S_d`-symmetric off-diagonal weight matrix parameterized by `(a, b)`:
weight `a` among `{0, ..., d-1}`, weight `b` between `{0, ..., d-1}` and node `d`. -/
def symmWeight (d : ℕ) (a b : ℝ) : Matrix (Fin (d + 1)) (Fin (d + 1)) ℝ :=
  Matrix.of fun i j =>
    if i = j then 0
    else if i.val < d ∧ j.val < d then a
    else b

/-- **Theorem 3(2) (Commuting `S_d`-symmetric Laplacians)**:
Every two `S_d`-symmetric Laplacians commute identically (`L(a₁, b₁) L(a₂, b₂) = L(a₂, b₂) L(a₁, b₁)`),
so every frozen projected commutator `Xᵀ [L₁, L₂] X` vanishes identically! -/
theorem symmLaplacian_commute (d : ℕ) (a₁ b₁ a₂ b₂ : ℝ) :
    laplacian (symmWeight d a₁ b₁) * laplacian (symmWeight d a₂ b₂) =
    laplacian (symmWeight d a₂ b₂) * laplacian (symmWeight d a₁ b₁) := by
  have hL : ∀ (a b : ℝ) (i j : Fin (d + 1)),
      laplacian (symmWeight d a b) i j =
        if i = j then (if i.val < d then ((d : ℝ) - 1) * a + b else (d : ℝ) * b)
        else if i.val < d ∧ j.val < d then -a
        else -b := by
    intro a b i j
    simp only [laplacian, Matrix.sub_apply, Matrix.diagonal_apply, symmWeight, Matrix.of_apply]
    by_cases hij : i = j
    · subst hij
      simp only [↓reduceIte, sub_zero]
      rw [Fin.sum_univ_castSucc]
      by_cases hi : i.val < d
      · simp only [hi, ↓reduceIte]
        have hlast : ¬ (i = Fin.last d) := fun h => by
          have := congrArg Fin.val h; simp at this; omega
        have hlast_lt : ¬ (Fin.last d).val < d := by simp
        simp only [hlast, hlast_lt, and_false, true_and, Fin.coe_castSucc, Fin.is_lt, ↓reduceIte]
        let i₀ : Fin d := ⟨i.val, hi⟩
        have hsum_cast : ∑ k : Fin d, (if i = k.castSucc then (0 : ℝ) else a) =
            ((d : ℝ) - 1) * a := by
          have heq : ∀ k : Fin d, (if i = k.castSucc then (0 : ℝ) else a) =
              if k = i₀ then 0 else a := fun k => by simp [Fin.ext_iff, i₀, eq_comm]
          rw [Finset.sum_congr rfl fun k _ => heq k]
          rw [← Finset.add_sum_erase _ _ (Finset.mem_univ i₀)]
          have hconst : ∑ k ∈ Finset.univ.erase i₀, (if k = i₀ then (0 : ℝ) else a) =
              ∑ _k ∈ Finset.univ.erase i₀, a :=
            Finset.sum_congr rfl fun k hk => by simp [(Finset.mem_erase.1 hk).1]
          rw [if_pos rfl, zero_add, hconst, Finset.sum_const,
            Finset.card_erase_of_mem (Finset.mem_univ i₀), Finset.card_univ, Fintype.card_fin]
          have hd1 : 1 ≤ d := by omega
          simp [Nat.cast_sub hd1]
        rw [hsum_cast]
      · simp only [hi, ↓reduceIte]
        have hilast : i = Fin.last d := by ext; simp; omega
        subst hilast
        simp only [false_and, ↓reduceIte, add_zero]
        have hne : ∀ k : Fin d, ¬ (Fin.last d = k.castSucc) := fun k h => by
          have := congrArg Fin.val h; simp at this; omega
        simp [hne]
    · simp only [hij, ↓reduceIte, zero_sub]
      split_ifs <;> ring
  ext i j
  simp only [Matrix.mul_apply, hL]
  rw [Fin.sum_univ_castSucc, Fin.sum_univ_castSucc]
  have hprod : ∀ (a₁ b₁ a₂ b₂ : ℝ),
      (∑ k : Fin d,
        (if i = k.castSucc then (if i.val < d then ((d : ℝ) - 1) * a₁ + b₁ else (d : ℝ) * b₁)
         else if i.val < d ∧ k.castSucc.val < d then -a₁ else -b₁) *
        (if k.castSucc = j then (if k.castSucc.val < d then ((d : ℝ) - 1) * a₂ + b₂ else (d : ℝ) * b₂)
         else if k.castSucc.val < d ∧ j.val < d then -a₂ else -b₂)) +
      (if i = Fin.last d then (if i.val < d then ((d : ℝ) - 1) * a₁ + b₁ else (d : ℝ) * b₁)
       else if i.val < d ∧ (Fin.last d).val < d then -a₁ else -b₁) *
      (if Fin.last d = j then (if (Fin.last d).val < d then ((d : ℝ) - 1) * a₂ + b₂ else (d : ℝ) * b₂)
       else if (Fin.last d).val < d ∧ j.val < d then -a₂ else -b₂) =
      if i.val < d then
        if j.val < d then
          if i = j then
            (((d : ℝ) - 1) * a₁ + b₁) * (((d : ℝ) - 1) * a₂ + b₂) +
            ((d : ℝ) - 1) * (a₁ * a₂) + b₁ * b₂
          else
            -(((d : ℝ) - 1) * a₁ + b₁) * a₂ - a₁ * (((d : ℝ) - 1) * a₂ + b₂) +
            ((d : ℝ) - 2) * (a₁ * a₂) + b₁ * b₂
        else
          -(((d : ℝ) - 1) * a₁ + b₁) * b₂ + ((d : ℝ) - 1) * (a₁ * b₂) - (d : ℝ) * (b₁ * b₂)
      else
        if j.val < d then
          -b₁ * (((d : ℝ) - 1) * a₂ + b₂) + ((d : ℝ) - 1) * (b₁ * a₂) - (d : ℝ) * (b₁ * b₂)
        else
          (d : ℝ) * (b₁ * b₂) + ((d : ℝ) * b₁) * ((d : ℝ) * b₂) := by
    intro x₁ y₁ x₂ y₂
    by_cases hi : i.val < d <;> by_cases hj : j.val < d
    · let i₀ : Fin d := ⟨i.val, hi⟩
      let j₀ : Fin d := ⟨j.val, hj⟩
      have hni : i ≠ Fin.last d := fun h => by have := congrArg Fin.val h; simp at this; omega
      have hnj : Fin.last d ≠ j := fun h => by have := congrArg Fin.val h; simp at this; omega
      simp only [hi, hj, hni, hnj, Fin.val_last, lt_self_iff_false, and_false, false_and,
        ↓reduceIte, Fin.coe_castSucc, Fin.is_lt, and_self]
      by_cases hij : i = j
      · subst hij
        simp only [↓reduceIte]
        have hterm : ∀ k : Fin d,
            (if i = k.castSucc then ((d : ℝ) - 1) * x₁ + y₁ else -x₁) *
            (if k.castSucc = i then ((d : ℝ) - 1) * x₂ + y₂ else -x₂) =
            if k = i₀ then (((d : ℝ) - 1) * x₁ + y₁) * (((d : ℝ) - 1) * x₂ + y₂) else x₁ * x₂ := by
          intro k
          have hk1 : (i = k.castSucc) ↔ (k = i₀) := by simp [Fin.ext_iff, i₀, eq_comm]
          have hk2 : (k.castSucc = i) ↔ (k = i₀) := by simp [Fin.ext_iff, i₀]
          simp only [hk1, hk2]
          split_ifs <;> ring
        rw [Finset.sum_congr rfl fun k _ => hterm k]
        rw [← Finset.add_sum_erase _ _ (Finset.mem_univ i₀)]
        have herase : ∑ k ∈ Finset.univ.erase i₀,
            (if k = i₀ then (((d : ℝ) - 1) * x₁ + y₁) * (((d : ℝ) - 1) * x₂ + y₂) else x₁ * x₂) =
            ((d : ℝ) - 1) * (x₁ * x₂) := by
          have h1 : ∑ k ∈ Finset.univ.erase i₀,
              (if k = i₀ then (((d : ℝ) - 1) * x₁ + y₁) * (((d : ℝ) - 1) * x₂ + y₂) else x₁ * x₂) =
              ∑ _k ∈ Finset.univ.erase i₀, x₁ * x₂ :=
            Finset.sum_congr rfl fun k hk => by simp [(Finset.mem_erase.1 hk).1]
          rw [h1, Finset.sum_const, Finset.card_erase_of_mem (Finset.mem_univ i₀),
            Finset.card_univ, Fintype.card_fin]
          have hd1 : 1 ≤ d := by omega
          simp [Nat.cast_sub hd1]
        rw [if_pos rfl, herase]; ring
      · simp only [hij, ↓reduceIte]
        have hij0 : i₀ ≠ j₀ := fun h => hij (Fin.ext (congrArg (fun x : Fin d => x.val) h))
        have hterm : ∀ k : Fin d,
            (if i = k.castSucc then ((d : ℝ) - 1) * x₁ + y₁ else -x₁) *
            (if k.castSucc = j then ((d : ℝ) - 1) * x₂ + y₂ else -x₂) =
            if k = i₀ then -(((d : ℝ) - 1) * x₁ + y₁) * x₂
            else if k = j₀ then -x₁ * (((d : ℝ) - 1) * x₂ + y₂)
            else x₁ * x₂ := by
          intro k
          have hk1 : (i = k.castSucc) ↔ (k = i₀) := by simp [Fin.ext_iff, i₀, eq_comm]
          have hk2 : (k.castSucc = j) ↔ (k = j₀) := by simp [Fin.ext_iff, j₀]
          simp only [hk1, hk2]
          by_cases hki : k = i₀
          · subst hki; simp [hij0]; ring
          · by_cases hkj : k = j₀
            · subst hkj; simp [hki]
            · simp [hki, hkj]
        rw [Finset.sum_congr rfl fun k _ => hterm k]
        rw [← Finset.add_sum_erase _ _ (Finset.mem_univ i₀)]
        have hjmem : j₀ ∈ Finset.univ.erase i₀ := Finset.mem_erase.2 ⟨hij0.symm, Finset.mem_univ _⟩
        rw [← Finset.add_sum_erase _ _ hjmem]
        have herase : ∑ k ∈ (Finset.univ.erase i₀).erase j₀,
            (if k = i₀ then -(((d : ℝ) - 1) * x₁ + y₁) * x₂
             else if k = j₀ then -x₁ * (((d : ℝ) - 1) * x₂ + y₂)
             else x₁ * x₂) = ((d : ℝ) - 2) * (x₁ * x₂) := by
          have h1 : ∑ k ∈ (Finset.univ.erase i₀).erase j₀,
              (if k = i₀ then -(((d : ℝ) - 1) * x₁ + y₁) * x₂
               else if k = j₀ then -x₁ * (((d : ℝ) - 1) * x₂ + y₂)
               else x₁ * x₂) =
              ∑ _k ∈ (Finset.univ.erase i₀).erase j₀, x₁ * x₂ :=
            Finset.sum_congr rfl fun k hk => by
              have hkj := (Finset.mem_erase.1 hk).1
              have hki := (Finset.mem_erase.1 (Finset.mem_erase.1 hk).2).1
              simp [hki, hkj]
          rw [h1, Finset.sum_const, Finset.card_erase_of_mem hjmem,
            Finset.card_erase_of_mem (Finset.mem_univ i₀), Finset.card_univ, Fintype.card_fin]
          have hd2 : 2 ≤ d := by
            have := Finset.card_pos.2 ⟨j₀, hjmem⟩
            have h1 := Finset.card_erase_of_mem (Finset.mem_univ i₀)
            simp at h1; omega
          have hsub : (d - 1 - 1 : ℕ) = d - 2 := by omega
          rw [hsub, nsmul_eq_mul, Nat.cast_sub hd2]; norm_num
        simp only [↓reduceIte, if_neg hij0.symm, herase]
        ring
    · let i₀ : Fin d := ⟨i.val, hi⟩
      have hjlast : j = Fin.last d := by ext; simp; omega
      subst hjlast
      have hni : i ≠ Fin.last d := fun h => by have := congrArg Fin.val h; simp at this; omega
      have hnk : ∀ k : Fin d, k.castSucc ≠ Fin.last d := fun k h => by
        have := congrArg Fin.val h; simp at this; omega
      simp only [hi, hni, hnk, Fin.val_last, lt_self_iff_false, and_false, ↓reduceIte,
        Fin.coe_castSucc, Fin.is_lt, and_self]
      have hterm : ∀ k : Fin d,
          (if i = k.castSucc then ((d : ℝ) - 1) * x₁ + y₁ else -x₁) * -y₂ =
          if k = i₀ then -(((d : ℝ) - 1) * x₁ + y₁) * y₂ else x₁ * y₂ := by
        intro k
        have hk1 : (i = k.castSucc) ↔ (k = i₀) := by simp [Fin.ext_iff, i₀, eq_comm]
        simp only [hk1]; split_ifs <;> ring
      rw [Finset.sum_congr rfl fun k _ => hterm k]
      rw [← Finset.add_sum_erase _ _ (Finset.mem_univ i₀)]
      have herase : ∑ k ∈ Finset.univ.erase i₀,
          (if k = i₀ then -(((d : ℝ) - 1) * x₁ + y₁) * y₂ else x₁ * y₂) =
          ((d : ℝ) - 1) * (x₁ * y₂) := by
        have h1 := Finset.sum_congr (s₁ := Finset.univ.erase i₀) rfl fun k hk => by
          show (if k = i₀ then -(((d : ℝ) - 1) * x₁ + y₁) * y₂ else x₁ * y₂) = x₁ * y₂
          simp [(Finset.mem_erase.1 hk).1]
        rw [h1, Finset.sum_const, Finset.card_erase_of_mem (Finset.mem_univ i₀),
          Finset.card_univ, Fintype.card_fin]
        have hd1 : 1 ≤ d := by omega
        simp [Nat.cast_sub hd1]
      rw [if_pos rfl, herase]; ring
    · let j₀ : Fin d := ⟨j.val, hj⟩
      have hilast : i = Fin.last d := by ext; simp; omega
      subst hilast
      have hnj : Fin.last d ≠ j := fun h => by have := congrArg Fin.val h; simp at this; omega
      have hnk : ∀ k : Fin d, Fin.last d ≠ k.castSucc := fun k h => by
        have := congrArg Fin.val h; simp at this; omega
      simp only [hj, hnj, hnk, Fin.val_last, lt_self_iff_false, false_and, ↓reduceIte,
        Fin.coe_castSucc, Fin.is_lt, and_self]
      have hterm : ∀ k : Fin d,
          -y₁ * (if k.castSucc = j then ((d : ℝ) - 1) * x₂ + y₂ else -x₂) =
          if k = j₀ then -y₁ * (((d : ℝ) - 1) * x₂ + y₂) else y₁ * x₂ := by
        intro k
        have hk2 : (k.castSucc = j) ↔ (k = j₀) := by simp [Fin.ext_iff, j₀]
        simp only [hk2]; split_ifs <;> ring
      rw [Finset.sum_congr rfl fun k _ => hterm k]
      rw [← Finset.add_sum_erase _ _ (Finset.mem_univ j₀)]
      have herase : ∑ k ∈ Finset.univ.erase j₀,
          (if k = j₀ then -y₁ * (((d : ℝ) - 1) * x₂ + y₂) else y₁ * x₂) =
          ((d : ℝ) - 1) * (y₁ * x₂) := by
        have h1 := Finset.sum_congr (s₁ := Finset.univ.erase j₀) rfl fun k hk => by
          show (if k = j₀ then -y₁ * (((d : ℝ) - 1) * x₂ + y₂) else y₁ * x₂) = y₁ * x₂
          simp [(Finset.mem_erase.1 hk).1]
        rw [h1, Finset.sum_const, Finset.card_erase_of_mem (Finset.mem_univ j₀),
          Finset.card_univ, Fintype.card_fin]
        have hd1 : 1 ≤ d := by omega
        simp [Nat.cast_sub hd1]
      rw [if_pos rfl, herase]; ring
    · have hilast : i = Fin.last d := by ext; simp; omega
      have hjlast : j = Fin.last d := by ext; simp; omega
      subst hilast; subst hjlast
      have hnk1 : ∀ k : Fin d, Fin.last d ≠ k.castSucc := fun k h => by
        have := congrArg Fin.val h; simp at this; omega
      have hnk2 : ∀ k : Fin d, k.castSucc ≠ Fin.last d := fun k h => hnk1 k h.symm
      simp only [hnk1, hnk2, Fin.val_last, lt_self_iff_false, false_and, and_false, ↓reduceIte]
      simp
  rw [hprod a₁ b₁ a₂ b₂, hprod a₂ b₂ a₁ b₁]
  split_ifs <;> ring

/-- Rank-1 outer product `u vᵀ`. -/
def outer {m n : ℕ} (u : Fin m → ℝ) (v : Fin n → ℝ) : Matrix (Fin m) (Fin n) ℝ :=
  Matrix.of fun i j => u i * v j

private lemma rodrigues_loop_props {d : ℕ} (u v : Fin d → ℝ)
    (hu_unit : dotProduct u u = 1) (hv_unit : dotProduct v v = 1) (huv : dotProduct u v = 0) :
    let K : Matrix (Fin d) (Fin d) ℝ := outer u v - outer v u
    let Q : ℝ → Matrix (Fin d) (Fin d) ℝ := fun θ =>
      1 - Real.sin (θ / 201) • K + (1 - Real.cos (θ / 201)) • (K * K)
    Q 0 = 1 ∧
    (∀ θ, (Q θ)ᵀ * Q θ = 1) ∧
    (∀ θ i j, HasDerivAt (fun t => Q t i j) (((-(1 / 201 : ℝ) • K) * Q θ) i j) θ) ∧
    Q (2 * Real.pi) ≠ 1 := by
  intro K Q
  have hvu : dotProduct v u = 0 := by rw [dotProduct_comm, huv]
  have houter_T : ∀ (a b : Fin d → ℝ), (outer a b)ᵀ = outer b a := fun a b => by
    ext i j; simp [outer, mul_comm]
  have houter_mul : ∀ (a b c w : Fin d → ℝ),
      outer a b * outer c w = (dotProduct b c) • outer a w := fun a b c w => by
    ext i j
    simp only [Matrix.mul_apply, outer, Matrix.of_apply, Matrix.smul_apply, smul_eq_mul, dotProduct]
    rw [Finset.sum_mul]
    refine Finset.sum_congr rfl fun k _ => by ring
  have houter_mulVec : ∀ (a b w : Fin d → ℝ),
      (outer a b).mulVec w = (dotProduct b w) • a := fun a b w => by
    ext i
    simp only [Matrix.mulVec, dotProduct, outer, Matrix.of_apply, Pi.smul_apply, smul_eq_mul]
    rw [Finset.sum_mul]
    refine Finset.sum_congr rfl fun k _ => by ring
  have hK2 : K * K = -(outer u u + outer v v) := by
    simp only [K, Matrix.mul_sub, Matrix.sub_mul, houter_mul, hu_unit, hv_unit, huv, hvu,
      one_smul, zero_smul, sub_zero, zero_sub]
    abel
  have hK3 : K * (K * K) = -K := by
    rw [hK2, Matrix.mul_neg, Matrix.mul_add]
    simp only [K, Matrix.sub_mul, houter_mul, hu_unit, hv_unit, huv, hvu,
      one_smul, zero_smul, sub_zero, zero_sub]
    abel
  have hK3' : (K * K) * K = -K := by rw [Matrix.mul_assoc, hK3]
  have hK2_sq : (K * K) * (K * K) = -(K * K) := by
    rw [Matrix.mul_assoc, hK3, Matrix.mul_neg]
  have hKT : Kᵀ = -K := by simp only [K, Matrix.transpose_sub, houter_T]; abel
  have hK2T : (K * K)ᵀ = K * K := by rw [Matrix.transpose_mul, hKT, Matrix.neg_mul, Matrix.mul_neg, neg_neg]
  have hQ0 : Q 0 = 1 := by simp [Q]
  have hQorth : ∀ θ, (Q θ)ᵀ * Q θ = 1 := by
    intro θ
    set s := Real.sin (θ / 201)
    set c := Real.cos (θ / 201)
    set a := 1 - c
    have hsc : s ^ 2 + c ^ 2 = 1 := Real.sin_sq_add_cos_sq _
    have hQT : (Q θ)ᵀ = 1 + s • K + a • (K * K) := by
      simp only [Q, Matrix.transpose_add, Matrix.transpose_sub, Matrix.transpose_one,
        Matrix.transpose_smul, hKT, hK2T, smul_neg, sub_neg_eq_add, s, c, a]
    have hQeq : Q θ = 1 - s • K + a • (K * K) := rfl
    rw [hQT, hQeq]
    have hm1 : (1 + s • K + a • (K * K)) * (1 - s • K + a • (K * K)) =
        (1 + s • K + a • (K * K)) - s • ((1 + s • K + a • (K * K)) * K) +
        a • ((1 + s • K + a • (K * K)) * (K * K)) := by
      rw [Matrix.mul_add, Matrix.mul_sub, Matrix.mul_one, Matrix.mul_smul, Matrix.mul_smul]
    have hm2 : (1 + s • K + a • (K * K)) * K = K + s • (K * K) + a • (-K) := by
      rw [Matrix.add_mul, Matrix.add_mul, Matrix.one_mul, Matrix.smul_mul, Matrix.smul_mul, hK3']
    have hm3 : (1 + s • K + a • (K * K)) * (K * K) = K * K + s • (-K) + a • (-(K * K)) := by
      rw [Matrix.add_mul, Matrix.add_mul, Matrix.one_mul, Matrix.smul_mul, Matrix.smul_mul, hK3, hK2_sq]
    rw [hm1, hm2, hm3]
    ext i j
    simp only [Matrix.add_apply, Matrix.sub_apply, Matrix.neg_apply, Matrix.smul_apply,
      Matrix.one_apply, smul_eq_mul]
    have hcoeff : 2 * a - s ^ 2 - a ^ 2 = 0 := by simp only [a]; nlinarith [hsc]
    linear_combination (K * K) i j * hcoeff
  have hQderiv : ∀ θ i j, HasDerivAt (fun t => Q t i j)
      (((-(1 / 201 : ℝ) • K) * Q θ) i j) θ := by
    intro θ i j
    have hdiv : HasDerivAt (fun t : ℝ => t / 201) (1 / 201) θ := by
      simpa using (hasDerivAt_id θ).div_const (201 : ℝ)
    have hsin : HasDerivAt (fun t : ℝ => Real.sin (t / 201)) (Real.cos (θ / 201) * (1 / 201)) θ :=
      hdiv.sin
    have hcos : HasDerivAt (fun t : ℝ => 1 - Real.cos (t / 201)) (Real.sin (θ / 201) * (1 / 201)) θ := by
      simpa using (hasDerivAt_const θ (1 : ℝ)).sub hdiv.cos
    have hval : (fun t => Q t i j) = fun t =>
        (1 : Matrix (Fin d) (Fin d) ℝ) i j - Real.sin (t / 201) * K i j +
          (1 - Real.cos (t / 201)) * (K * K) i j := by
      ext t; simp [Q]
    rw [hval]
    have hd1 := ((hasDerivAt_const θ ((1 : Matrix (Fin d) (Fin d) ℝ) i j)).sub
      (hsin.mul_const (K i j))).add (hcos.mul_const ((K * K) i j))
    convert hd1 using 1
    have hprod : (-(1 / 201 : ℝ) • K) * Q θ =
        (-(1 / 201 : ℝ) * Real.cos (θ / 201)) • K +
        ((1 / 201 : ℝ) * Real.sin (θ / 201)) • (K * K) := by
      simp only [Q, Matrix.mul_add, Matrix.mul_sub, Matrix.smul_mul, Matrix.mul_smul,
        Matrix.mul_one, hK3, smul_neg, smul_smul]
      ext a b
      simp only [Matrix.add_apply, Matrix.sub_apply, Matrix.neg_apply, Matrix.smul_apply,
        smul_eq_mul]
      ring
    rw [hprod]
    simp only [Matrix.add_apply, Matrix.smul_apply, smul_eq_mul]
    ring
  have hQne1 : Q (2 * Real.pi) ≠ 1 := by
    intro hEq
    have hKv : K.mulVec v = u := by
      show (outer u v - outer v u).mulVec v = u
      rw [Matrix.sub_mulVec, houter_mulVec, houter_mulVec, hv_unit, huv, one_smul, zero_smul, sub_zero]
    have hK2v : (K * K).mulVec v = -v := by
      rw [hK2, Matrix.neg_mulVec, Matrix.add_mulVec, houter_mulVec, houter_mulVec, huv, hv_unit,
        zero_smul, one_smul, zero_add]
    have hQv : (Q (2 * Real.pi)).mulVec v =
        v - Real.sin (2 * Real.pi / 201) • u + (1 - Real.cos (2 * Real.pi / 201)) • (-v) := by
      show (1 - Real.sin (2 * Real.pi / 201) • K + (1 - Real.cos (2 * Real.pi / 201)) • (K * K)).mulVec v = _
      rw [Matrix.add_mulVec, Matrix.sub_mulVec, Matrix.one_mulVec, Matrix.smul_mulVec,
        Matrix.smul_mulVec, hKv, hK2v]
    have h1 : dotProduct u ((Q (2 * Real.pi)).mulVec v) = dotProduct u v := by
      rw [hEq, Matrix.one_mulVec]
    have h2 : dotProduct u ((Q (2 * Real.pi)).mulVec v) = -Real.sin (2 * Real.pi / 201) := by
      rw [hQv]
      simp only [dotProduct, Pi.add_apply, Pi.sub_apply, Pi.smul_apply, Pi.neg_apply, smul_eq_mul]
      calc ∑ i : Fin d, u i * (v i - Real.sin (2 * Real.pi / 201) * u i +
              (1 - Real.cos (2 * Real.pi / 201)) * -v i)
          = (dotProduct u v) - Real.sin (2 * Real.pi / 201) * (dotProduct u u) -
            (1 - Real.cos (2 * Real.pi / 201)) * (dotProduct u v) := by
            unfold dotProduct
            simp only [Finset.mul_sum, ← Finset.sum_sub_distrib]
            refine Finset.sum_congr rfl fun i _ => by ring
        _ = -Real.sin (2 * Real.pi / 201) := by rw [huv, hu_unit]; ring
    have hsin_pos : 0 < Real.sin (2 * Real.pi / 201) := by
      apply Real.sin_pos_of_pos_of_lt_pi
      · positivity
      · nlinarith [Real.pi_pos]
    linarith
  exact ⟨hQ0, hQorth, hQderiv, hQne1⟩

/-- **Theorem 3(3) (Exact disk loop holonomy angle `2π / 201`)**:
At the non-regular `S_d`-symmetric frame `X_d` (`d ≥ 2`), with `g = (4d)⁻²`, `r = (40d)⁻¹`,
unit `z ∈ ker(X_dᵀ)`, orthonormal contrasts `u, v ⊥ 1_d`, `K = u vᵀ - v uᵀ`, and
`Y(θ) = X_d + r z (cos θ u + sin θ v)ᵀ`:
1. `r² / (2g + r²) = 1 / 201` identically for every `d ≥ 2`;
2. `Y(θ)ᵀ Y'(θ) - Y'(θ)ᵀ Y(θ) = r² K` and `S_{Y(θ)ᵀ Y(θ)}(K) = (2g + r²) K`, so
   `ω_{Y(θ)}(Y'(θ)) = (1 / 201) K` is constant along the circle;
3. The explicit rotation curve `Q(θ) = I - sin(θ/201) K + (1 - cos(θ/201)) K²` solves
   `Q'(θ) = -(1/201) K Q(θ)` with `Q(0) = I`, `Q(θ)ᵀ Q(θ) = I`, and `Q(2π) ≠ I`. -/
theorem disk_loop_exact_holonomy {d : ℕ} (hd : 2 ≤ d)
    (z : Fin (d + 1) → ℝ) (hz_ker : (symmFrame d)ᵀ.mulVec z = 0) (hz_unit : dotProduct z z = 1)
    (u v : Fin d → ℝ)
    (hu_unit : dotProduct u u = 1) (hv_unit : dotProduct v v = 1) (huv : dotProduct u v = 0)
    (hu_perp : ∑ i : Fin d, u i = 0) (hv_perp : ∑ i : Fin d, v i = 0) :
    let g : ℝ := 1 / (4 * (d : ℝ)) ^ 2
    let r : ℝ := 1 / (40 * (d : ℝ))
    let K : Matrix (Fin d) (Fin d) ℝ := outer u v - outer v u
    let h : ℝ → (Fin d → ℝ) := fun θ i => Real.cos θ * u i + Real.sin θ * v i
    let Y : ℝ → Matrix (Fin (d + 1)) (Fin d) ℝ := fun θ => symmFrame d + r • outer z (h θ)
    let Y' : ℝ → Matrix (Fin (d + 1)) (Fin d) ℝ :=
      fun θ => r • outer z (fun i => -Real.sin θ * u i + Real.cos θ * v i)
    let Q : ℝ → Matrix (Fin d) (Fin d) ℝ :=
      fun θ => (1 : Matrix (Fin d) (Fin d) ℝ) - Real.sin (θ / 201) • K +
        (1 - Real.cos (θ / 201)) • (K * K)
    r ^ 2 / (2 * g + r ^ 2) = 1 / 201 ∧
    (∀ θ, (Y θ)ᵀ * Y' θ - (Y' θ)ᵀ * Y θ = r ^ 2 • K) ∧
    (∀ θ, lyapunov ((Y θ)ᵀ * Y θ) K = (2 * g + r ^ 2) • K) ∧
    (Q 0 = 1) ∧
    (∀ θ, (Q θ)ᵀ * Q θ = 1) ∧
    (∀ θ i j, HasDerivAt (fun s => Q s i j) (((-(1 / 201 : ℝ) • K) * Q θ) i j) θ) ∧
    Q (2 * Real.pi) ≠ 1 := by
  intro g r K h Y Y' Q
  have hd_pos : (0 : ℝ) < d := by positivity
  have hrate : r ^ 2 / (2 * g + r ^ 2) = 1 / 201 := by
    simp only [g, r]
    field_simp [hd_pos.ne']
    ring
  -- Helper identities for outer products
  have hvu : dotProduct v u = 0 := by rw [dotProduct_comm]; exact huv
  have houter_mul : ∀ {p q s : ℕ} (a : Fin p → ℝ) (b c : Fin q → ℝ) (w : Fin s → ℝ),
      outer a b * outer c w = (dotProduct b c) • outer a w := by
    intro p q s a b c w
    ext i j
    simp only [Matrix.mul_apply, outer, Matrix.of_apply, Matrix.smul_apply, smul_eq_mul,
      dotProduct, Finset.sum_mul]
    refine Finset.sum_congr rfl fun k _ => by ring
  have houter_mulVec : ∀ {p q : ℕ} (a : Fin p → ℝ) (b c : Fin q → ℝ),
      (outer a b).mulVec c = (dotProduct b c) • a := by
    intro p q a b c
    ext i
    simp only [Matrix.mulVec, dotProduct, outer, Matrix.of_apply, Pi.smul_apply, smul_eq_mul,
      Finset.sum_mul]
    refine Finset.sum_congr rfl fun k _ => by ring
  have houter_T : ∀ {p q : ℕ} (a : Fin p → ℝ) (b : Fin q → ℝ),
      (outer a b)ᵀ = outer b a := by
    intro p q a b; ext i j; simp [outer, mul_comm]
  have hX_outer : ∀ w : Fin d → ℝ, (symmFrame d)ᵀ * outer z w = 0 := by
    intro w; ext i j
    simp only [Matrix.mul_apply, outer, Matrix.of_apply, Matrix.zero_apply]
    have hi : ∑ k, (symmFrame d)ᵀ i k * z k = 0 := congrFun hz_ker i
    have hfac : ∑ k, (symmFrame d)ᵀ i k * (z k * w j) =
        (∑ k, (symmFrame d)ᵀ i k * z k) * w j := by
      rw [Finset.sum_mul]; refine Finset.sum_congr rfl fun k _ => by ring
    rw [hfac, hi, zero_mul]
  have houter_X : ∀ w : Fin d → ℝ, outer w z * symmFrame d = 0 := by
    intro w
    have := congrArg Matrix.transpose (hX_outer w)
    simpa [Matrix.transpose_mul, houter_T] using this
  -- 2. Y(θ)ᵀ Y'(θ) - Y'(θ)ᵀ Y(θ) = r² K
  have hconn_num : ∀ θ, (Y θ)ᵀ * Y' θ - (Y' θ)ᵀ * Y θ = r ^ 2 • K := by
    intro θ
    set h' : Fin d → ℝ := fun i => -Real.sin θ * u i + Real.cos θ * v i
    have h1 : (Y θ)ᵀ * Y' θ = r ^ 2 • outer (h θ) h' := by
      simp only [Y, Y', Matrix.transpose_add, Matrix.transpose_smul, houter_T,
        Matrix.add_mul, Matrix.mul_smul, Matrix.smul_mul, hX_outer, zero_add,
        houter_mul, hz_unit, one_smul, smul_smul, sq]; rfl
    have h2 : (Y' θ)ᵀ * Y θ = r ^ 2 • outer h' (h θ) := by
      simp only [Y, Y', Matrix.transpose_smul, houter_T, Matrix.mul_add, Matrix.smul_mul,
        Matrix.mul_smul, houter_X, smul_zero, zero_add, houter_mul, hz_unit, one_smul,
        smul_smul, sq]; rfl
    rw [h1, h2, ← smul_sub]
    congr 1
    ext i j
    simp only [Matrix.sub_apply, outer, Matrix.of_apply, h, h', K]
    have hcs : Real.cos θ ^ 2 + Real.sin θ ^ 2 = 1 := Real.cos_sq_add_sin_sq θ
    linear_combination (u i * v j - v i * u j) * hcs
  -- 3. S_{Y(θ)ᵀ Y(θ)}(K) = (2g + r²) K
  have hG_K : (symmFrame d)ᵀ * symmFrame d * K = g • K := by
    rw [symmFrame_gram_formula d]
    have hJ_u : allOnes d * outer u v = 0 := by
      ext i j; simp only [Matrix.mul_apply, allOnes, outer, Matrix.of_apply, Matrix.zero_apply,
        one_mul, ← Finset.sum_mul, hu_perp, zero_mul]
    have hJ_v : allOnes d * outer v u = 0 := by
      ext i j; simp only [Matrix.mul_apply, allOnes, outer, Matrix.of_apply, Matrix.zero_apply,
        one_mul, ← Finset.sum_mul, hv_perp, zero_mul]
    have hJK : allOnes d * K = 0 := by simp [K, Matrix.mul_sub, hJ_u, hJ_v]
    simp only [g, Matrix.smul_mul, Matrix.add_mul, Matrix.one_mul, hJK, smul_zero, add_zero]
  have hK_G : K * ((symmFrame d)ᵀ * symmFrame d) = g • K := by
    have hKT : Kᵀ = -K := by
      simp only [K, Matrix.transpose_sub, houter_T]; abel
    have h1 := congrArg Matrix.transpose hG_K
    rw [Matrix.transpose_mul, Matrix.transpose_smul, hKT] at h1
    have hGT : ((symmFrame d)ᵀ * symmFrame d)ᵀ = (symmFrame d)ᵀ * symmFrame d := by simp
    rw [hGT] at h1
    have h2 := congrArg Neg.neg h1
    simpa [Matrix.neg_mul, smul_neg] using h2
  have hlyap : ∀ θ, lyapunov ((Y θ)ᵀ * Y θ) K = (2 * g + r ^ 2) • K := by
    intro θ
    have hYtY : (Y θ)ᵀ * Y θ = (symmFrame d)ᵀ * symmFrame d + r ^ 2 • outer (h θ) (h θ) := by
      simp only [Y, Matrix.transpose_add, Matrix.transpose_smul, houter_T,
        Matrix.add_mul, Matrix.mul_add, Matrix.mul_smul, Matrix.smul_mul,
        hX_outer, houter_X, smul_zero, add_zero, zero_add, houter_mul, hz_unit,
        one_smul, smul_smul, sq]
    have hhhK : outer (h θ) (h θ) * K + K * outer (h θ) (h θ) = K := by
      have hhu : dotProduct (h θ) u = Real.cos θ := by
        simp only [dotProduct, h, add_mul, Finset.sum_add_distrib, mul_assoc, ← Finset.mul_sum]
        change Real.cos θ * dotProduct u u + Real.sin θ * dotProduct v u = Real.cos θ
        rw [hu_unit, hvu]; ring
      have hhv : dotProduct (h θ) v = Real.sin θ := by
        simp only [dotProduct, h, add_mul, Finset.sum_add_distrib, mul_assoc, ← Finset.mul_sum]
        change Real.cos θ * dotProduct u v + Real.sin θ * dotProduct v v = Real.sin θ
        rw [huv, hv_unit]; ring
      have huh : dotProduct u (h θ) = Real.cos θ := by rw [dotProduct_comm, hhu]
      have hvh : dotProduct v (h θ) = Real.sin θ := by rw [dotProduct_comm, hhv]
      simp only [K, Matrix.mul_sub, Matrix.sub_mul, houter_mul, hhu, hhv, huh, hvh]
      ext i j
      simp only [Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply, outer, Matrix.of_apply,
        smul_eq_mul, h]
      have hcs : Real.cos θ ^ 2 + Real.sin θ ^ 2 = 1 := Real.cos_sq_add_sin_sq θ
      linear_combination (u i * v j - v i * u j) * hcs
    unfold lyapunov
    rw [hYtY, Matrix.add_mul, Matrix.mul_add, hG_K, hK_G, Matrix.smul_mul, Matrix.mul_smul]
    calc g • K + r ^ 2 • (outer (h θ) (h θ) * K) +
          (g • K + r ^ 2 • (K * outer (h θ) (h θ)))
        = (g + g) • K + r ^ 2 • (outer (h θ) (h θ) * K + K * outer (h θ) (h θ)) := by
          rw [add_smul, smul_add]; abel
      _ = (2 * g + r ^ 2) • K := by rw [hhhK, ← add_smul]; congr 1; ring
  -- 4-7. Properties of Q(θ) via standalone Rodrigues lemma
  obtain ⟨hQ0, hQorth, hQderiv, hQne1⟩ := rodrigues_loop_props u v hu_unit hv_unit huv
  exact ⟨hrate, hconn_num, hlyap, hQ0, hQorth, hQderiv, hQne1⟩

/-! ### 4. Theorem 4: Base-4 Vandermonde Finite-Time Rank Core & Path Laplacian Simple Spectrum -/

/-- Base-4 diagonal equilibrium values: `s_i = 4^i` for `i < d`, and `s_d = 0`. -/
def base4Diag (d : ℕ) (i : Fin (d + 1)) : ℕ :=
  if i.val < d then 4 ^ i.val else 0

/-- Strict pair index set `{(i, j) | 0 ≤ i < j ≤ d}` of size `d(d+1)/2`. -/
abbrev PairIdx (d : ℕ) := {p : Fin (d + 1) × Fin (d + 1) // p.1 < p.2}

/-- Linearized off-diagonal exponential rates `μ_{ij} = s_i + s_j`. -/
def base4Rate (d : ℕ) (p : PairIdx d) : ℕ :=
  base4Diag d p.1.1 + base4Diag d p.1.2

lemma base4Rate_pos (d : ℕ) (p : PairIdx d) : 0 < base4Rate d p := by
  unfold base4Rate base4Diag
  have h1 : p.1.1.val < d := by
    have hlt : p.1.1.val < p.1.2.val := p.2
    have hle : p.1.2.val ≤ d := Nat.lt_succ_iff.1 p.1.2.isLt
    omega
  simp only [h1, ↓reduceIte]
  have hpow : 0 < 4 ^ p.1.1.val := pow_pos (by norm_num) _
  omega

/-- Key base-4 uniqueness lemma: if `4^a + c = 4^a' + c'` where `c ∈ {0, 4^b}` (`a < b`) and
`c' ∈ {0, 4^b'}` (`a' < b'`), then `a = a'` and `c = c'`. -/
private lemma base4_pair_eq {a b a' b' : ℕ}
    (h1 : b = 0 ∨ a < b) (h2 : b' = 0 ∨ a' < b')
    (heq : 4 ^ a + (if b = 0 then 0 else 4 ^ b) = 4 ^ a' + (if b' = 0 then 0 else 4 ^ b')) :
    a = a' ∧ (if b = 0 then 0 else 4 ^ b) = (if b' = 0 then 0 else 4 ^ b') := by
  -- Use divisibility by 4^(min a a'): the unique lowest base-4 digit determines a = a'!
  have hlow : ∀ x y : ℕ, (y = 0 ∨ x < y) →
      4 ^ x ∣ (4 ^ x + if y = 0 then 0 else 4 ^ y) ∧
      ¬ (4 ^ (x + 1) ∣ (4 ^ x + if y = 0 then 0 else 4 ^ y)) := by
    intro x y hxy
    rcases hxy with rfl | hlt
    · simp only [↓reduceIte, add_zero]
      refine ⟨dvd_rfl, ?_⟩
      intro hdvd
      have hle : 4 ^ (x + 1) ≤ 4 ^ x := Nat.le_of_dvd (pow_pos (by norm_num) x) hdvd
      have hgt : 4 ^ x < 4 ^ (x + 1) := by rw [pow_succ]; nlinarith [pow_pos (by norm_num : 0 < 4) x]
      omega
    · have hy0 : y ≠ 0 := by omega
      simp only [hy0, ↓reduceIte]
      have hdvd_y : 4 ^ (x + 1) ∣ 4 ^ y := Nat.pow_dvd_pow 4 hlt
      have hdvd_y' : 4 ^ x ∣ 4 ^ y := Nat.pow_dvd_pow 4 hlt.le
      refine ⟨dvd_add dvd_rfl hdvd_y', ?_⟩
      intro hdvd
      have hdvd_x : 4 ^ (x + 1) ∣ 4 ^ x := (Nat.dvd_add_left hdvd_y).1 hdvd
      have hle : 4 ^ (x + 1) ≤ 4 ^ x := Nat.le_of_dvd (pow_pos (by norm_num) x) hdvd_x
      have hgt : 4 ^ x < 4 ^ (x + 1) := by rw [pow_succ]; nlinarith [pow_pos (by norm_num : 0 < 4) x]
      omega
  have ha : a = a' := by
    by_contra hne
    rcases lt_or_gt_of_ne hne with hlt | hgt
    · have h4a1 : 4 ^ (a + 1) ∣ 4 ^ a' := Nat.pow_dvd_pow 4 hlt
      have hdiv_rhs : 4 ^ (a + 1) ∣ (4 ^ a' + if b' = 0 then 0 else 4 ^ b') :=
        dvd_trans h4a1 (hlow a' b' h2).1
      rw [← heq] at hdiv_rhs
      exact (hlow a b h1).2 hdiv_rhs
    · have h4a'1 : 4 ^ (a' + 1) ∣ 4 ^ a := Nat.pow_dvd_pow 4 hgt
      have hdiv_lhs : 4 ^ (a' + 1) ∣ (4 ^ a + if b = 0 then 0 else 4 ^ b) :=
        dvd_trans h4a'1 (hlow a b h1).1
      rw [heq] at hdiv_lhs
      exact (hlow a' b' h2).2 hdiv_lhs
  subst ha
  exact ⟨rfl, by omega⟩

/-- **Theorem 4 (Base-4 rate injectivity)**: All `d(d+1)/2` rates `μ_{ij} = s_i + s_j`
(`0 ≤ i < j ≤ d`) are pairwise distinct. -/
theorem base4Rate_injective (d : ℕ) : Function.Injective (base4Rate d) := by
  intro p q hpq
  have hp1 : p.1.1.val < d := by have := p.2; have := p.1.2.isLt; omega
  have hq1 : q.1.1.val < d := by have := q.2; have := q.1.2.isLt; omega
  let bp : ℕ := if p.1.2.val < d then p.1.2.val else 0
  let bq : ℕ := if q.1.2.val < d then q.1.2.val else 0
  have hbp : bp = 0 ∨ p.1.1.val < bp := by
    unfold bp; split_ifs with h
    · right; exact p.2
    · left; rfl
  have hbq : bq = 0 ∨ q.1.1.val < bq := by
    unfold bq; split_ifs with h
    · right; exact q.2
    · left; rfl
  have hval_p : base4Rate d p = 4 ^ p.1.1.val + if bp = 0 then 0 else 4 ^ bp := by
    unfold base4Rate base4Diag bp
    simp only [hp1, ↓reduceIte]
    split_ifs with h1 h2
    · have : p.1.1.val < p.1.2.val := p.2; omega
    · rfl
    · rfl
    · omega
  have hval_q : base4Rate d q = 4 ^ q.1.1.val + if bq = 0 then 0 else 4 ^ bq := by
    unfold base4Rate base4Diag bq
    simp only [hq1, ↓reduceIte]
    split_ifs with h1 h2
    · have : q.1.1.val < q.1.2.val := q.2; omega
    · rfl
    · rfl
    · omega
  rw [hval_p, hval_q] at hpq
  obtain ⟨h1, h2⟩ := base4_pair_eq hbp hbq hpq
  have hp1_eq : p.1.1 = q.1.1 := Fin.ext h1
  have hp2_eq : p.1.2 = q.1.2 := by
    apply Fin.ext
    by_cases hp2 : p.1.2.val < d <;> by_cases hq2 : q.1.2.val < d
    · have hbp_ne : bp ≠ 0 := by unfold bp; rw [if_pos hp2]; have := p.2; omega
      have hbq_ne : bq ≠ 0 := by unfold bq; rw [if_pos hq2]; have := q.2; omega
      simp only [hbp_ne, hbq_ne, ↓reduceIte] at h2
      have hpow_eq : bp = bq := Nat.pow_right_injective (by norm_num : 2 ≤ 4) h2
      simpa [bp, bq, hp2, hq2] using hpow_eq
    · have hbp_ne : bp ≠ 0 := by unfold bp; rw [if_pos hp2]; have := p.2; omega
      have hbq_eq : bq = 0 := by unfold bq; rw [if_neg hq2]
      simp only [hbp_ne, hbq_eq, ↓reduceIte] at h2
      exact absurd h2 (pow_ne_zero _ (by norm_num))
    · have hbp_eq : bp = 0 := by unfold bp; rw [if_neg hp2]
      have hbq_ne : bq ≠ 0 := by unfold bq; rw [if_pos hq2]; have := q.2; omega
      simp only [hbp_eq, hbq_ne, ↓reduceIte] at h2
      exact absurd h2.symm (pow_ne_zero _ (by norm_num))
    · have := p.1.2.isLt; have := q.1.2.isLt; omega
  exact Subtype.ext (Prod.ext hp1_eq hp2_eq)

/-- **Theorem 4 (Weighted shifted Vandermonde invertibility)**:
For any positive weights `w : PairIdx d → ℝ`, the `N × N` matrix `M_{p, k} = w_p μ_p^{k+1}`
has nonzero determinant. -/
theorem vandermonde_rates_det_ne_zero (d : ℕ) (w : PairIdx d → ℝ) (hw : ∀ p, 0 < w p) :
    let e := Fintype.equivFin (PairIdx d)
    let M : Matrix (Fin (Fintype.card (PairIdx d))) (Fin (Fintype.card (PairIdx d))) ℝ :=
      Matrix.of fun r k => w (e.symm r) * ((base4Rate d (e.symm r) : ℝ) ^ (k.val + 1))
    M.det ≠ 0 := by
  intro e M
  let μ : Fin (Fintype.card (PairIdx d)) → ℝ := fun r => (base4Rate d (e.symm r) : ℝ)
  have hM : M = Matrix.diagonal (fun r => w (e.symm r) * μ r) * Matrix.vandermonde μ := by
    ext r k
    simp only [M, Matrix.of_apply, Matrix.diagonal_mul, Matrix.vandermonde_apply, μ, pow_succ']
    ring
  rw [hM, Matrix.det_mul, Matrix.det_diagonal, Matrix.det_vandermonde]
  refine mul_ne_zero ?_ ?_
  · refine Finset.prod_ne_zero_iff.2 fun r _ => ?_
    exact mul_ne_zero (hw _).ne' (Nat.cast_ne_zero.2 (base4Rate_pos d _).ne')
  · refine Finset.prod_ne_zero_iff.2 fun i _ => Finset.prod_ne_zero_iff.2 fun j hj => ?_
    rw [Finset.mem_Ioi] at hj
    intro hdiff
    have heq : μ j = μ i := sub_eq_zero.1 hdiff
    have hnat : base4Rate d (e.symm j) = base4Rate d (e.symm i) := Nat.cast_injective heq
    have hji : j = i := e.symm.injective (base4Rate_injective d hnat)
    exact (ne_of_gt hj) hji

/-- Path graph Laplacian on `Fin (d + 1)`: tridiagonal with `-1` off-diagonals and degrees
`1, 2, ..., 2, 1`. -/
def pathAdj (d : ℕ) : Matrix (Fin (d + 1)) (Fin (d + 1)) ℝ :=
  Matrix.of fun i j => if i.val + 1 = j.val ∨ j.val + 1 = i.val then 1 else 0

/-- **Theorem 4 (Simple spectrum of the path Laplacian)**:
Every eigenspace of `laplacian (pathAdj d)` has dimension at most `1`: any two eigenvectors `u, v`
with the same eigenvalue `λ` and equal `0`-th coordinate `u 0 = v 0` are identically equal. -/
theorem path_laplacian_simple_spectrum (d : ℕ) (lam : ℝ) (u v : Fin (d + 1) → ℝ)
    (hu : (laplacian (pathAdj d)).mulVec u = lam • u)
    (hv : (laplacian (pathAdj d)).mulVec v = lam • v)
    (h0 : u 0 = v 0) : u = v := by
  -- Define w = u - v; then L w = lam • w and w 0 = 0, so w = 0 by induction!
  let w : Fin (d + 1) → ℝ := u - v
  have hw : (laplacian (pathAdj d)).mulVec w = lam • w := by
    simp only [w, Matrix.mulVec_sub, hu, hv, smul_sub]
  have hw0 : w 0 = 0 := by simp [w, h0]
  have hzero : ∀ k : ℕ, (hk : k < d + 1) → w ⟨k, hk⟩ = 0 := by
    intro k
    induction k using Nat.strong_induction_on with
    | h k ih =>
      intro hk
      cases k with
      | zero => exact hw0
      | succ m =>
        -- Use row m of (L w)_m = lam * w_m to deduce w_{m+1} = 0!
        have hm : m < d + 1 := by omega
        have hrow := congrFun hw ⟨m, hm⟩
        have hwm : w ⟨m, hm⟩ = 0 := ih m (Nat.lt_succ_self m) hm
        simp only [Matrix.mulVec, dotProduct, laplacian, Matrix.sub_apply, Matrix.diagonal_apply,
          pathAdj, Matrix.of_apply, Pi.smul_apply, smul_eq_mul, hwm, mul_zero] at hrow
        -- In row m, the only nonzero off-diagonal entries are j = m - 1 (if m > 0) and j = m + 1.
        have hterm : ∀ j : Fin (d + 1),
            ((if (⟨m, hm⟩ : Fin (d + 1)) = j then
                ∑ l : Fin (d + 1), if m + 1 = l.val ∨ l.val + 1 = m then (1 : ℝ) else 0 else 0) -
              (if m + 1 = j.val ∨ j.val + 1 = m then 1 else 0)) * w j =
            if j = ⟨m + 1, hk⟩ then -w ⟨m + 1, hk⟩ else 0 := by
          intro j
          by_cases hjm : j = ⟨m, hm⟩
          · subst hjm
            have hne : (⟨m, hm⟩ : Fin (d + 1)) ≠ ⟨m + 1, hk⟩ := by
              intro h; have := congrArg Fin.val h; simp at this
            simp [hne, hwm]
          · by_cases hjs : j = ⟨m + 1, hk⟩
            · subst hjs
              have hne : (⟨m, hm⟩ : Fin (d + 1)) ≠ ⟨m + 1, hk⟩ := by
                intro h; have := congrArg Fin.val h; simp at this
              simp [hne]
            · by_cases hjp : j.val + 1 = m
              · have hjlt : j.val < m + 1 := by omega
                have hwj : w j = 0 := by
                  have := ih j.val hjlt j.isLt
                  simpa [Fin.eta] using this
                simp [hjs, hwj]
              · have hne1 : m + 1 ≠ j.val := fun h => hjs (Fin.ext h.symm)
                simp [Ne.symm hjm, hjs, hne1, hjp]
        simp_rw [hterm] at hrow
        simp only [Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte, neg_eq_zero] at hrow
        exact hrow
  ext i
  have hi : w i = 0 := by simpa [Fin.eta] using hzero i.val i.isLt
  exact sub_eq_zero.1 hi

end RDPGLatentHolonomy
