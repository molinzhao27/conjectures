import Mathlib

/-!
# Rank-Eight Counterexamples to Graph Coherence Stability under Random Edge Union
(Tommaso d'Orsi and Gleb Novikov, SODA 2026 / arXiv:2510.26679, Conjectures
`conj:coherence-simple` (1.7) and `conj:coherence-normalized` (1.8))

We formalize in Lean 4 + Mathlib the exact deterministic spectral core of the simultaneous
rank-8 `K₄`-with-private-leaves + `s = ⌊t^(3/4)⌋` isolates construction refuting both
`conj:coherence-simple` and `conj:coherence-normalized`:

1. **Input Graph `K` Rank and Exact Coherence (Lemma 1)**:
   - `B_input_invertible`: The 8×8 quotient matrix `B = [[J₄ - I₄, I₄], [I₄, 0]]` has exact
     inverse `B⁻¹ = [[0, I₄], [I₄, -(J₄ - I₄)]]` (`B * B⁻¹ = 1` and `B⁻¹ * B = 1`),
     certifying `rank(A_K) = rank(Ā_K) = 8`.
   - `input_projector_coherence`: The 8-class indicator orthogonal projector `P₈` on
     `Fin (8 * t + s)` satisfies `P₈ᵀ = P₈`, `P₈ * P₈ = P₈`, `tr(P₈) = 8`, `max_j (P₈)_{jj} = 1/t`,
     and exact input coherence `coh₈(A_K) = coh₈(Ā_K) = n / (8 t)`.

2. **Exact 9-Dimensional Quotient Reduction at `p = 1/5` (Lemma 2)**:
   - `critical_det_collapse`: `det [[3 + p, 1 + 3p], [1 + 3p, 4p]] = -(5p - 1)(p - 1)`,
     vanishing at `p = 1/5` so the center–leaf 2×2 block collapses to rank 1: `(4/5) v vᵀ`.
   - `quotient_block_orthogonal_reduction`: Explicit orthogonal similarity `Qᵀ M₃(c,l,i,ε) Q = L₃`
     where `L₃ = [[a, 0, ε h], [0, 0, ε b], [ε h, ε b, ε² d]]`.
   - `L3_charpoly_exact`: Exact characteristic polynomial
     `p_L(λ) = λ³ - (a + ε² d) λ² - ε² (h² + b² - a d) λ + a ε² b²`,
     confirming `tr(L₃) = a + ε² d` and `det(L₃) = -a ε² b²`.
   - `gap_coeff_identities`, `gap_coeff_raw_and_normalized`:
     `d - h² / a = (i² / 5) * (1 - ((2c² + l²) / (4c² + l²))²)`, evaluating to `16 / 125 > 0`
     (raw: `c = l = i = 1`) and `5 / 72 > 0` (normalized: `c² = 5/24, l² = 5/12, i² = 5/8`).
   - `difference_block_det`: Each of the three 2×2 difference blocks
     `(4/5) [[-c², c l], [c l, 0]]` has nonzero determinant `-(16/25) c² l²`.
   - `strict_eighth_ninth_gap_and_isolate_mass`: For ANY `a > 0, b > 0, d > h²/a`, there exists
     `ε₀ > 0` such that for all `ε ∈ (0, ε₀)`, `L₃` has three exact eigenvalues
     `λ_- < 0 < λ_+ < λ_0` satisfying:
     * strict 8th–9th singular gap `λ_+ - |λ_-| = λ_+ + λ_- ≥ (1/2) ε² (d - h²/a) > 0`,
     * explicit eigenvector `(x, y, z)` of `L₃` for `λ_+` with squared isolate coordinate
       `z² / (x² + y² + z²) ≥ 1/3` (in fact `> 1/2 - δ` for any `δ > 0`).

3. **Eigenvector Perturbation & `Ω(n^(1/4))` Output-Coherence Blow-Up (Theorem 1 core)**:
   - `projector_diag_ge_sq_entry`: For any symmetric idempotent projector `P` (`Pᵀ = P`, `P² = P`)
     and any unit vector `v̂` in its range (`P v̂ = v̂`), `(P)_{jj} ≥ v̂_j²` for every coordinate `j`.
   - `pigeonhole_isolate_max_sq`: On an isolate set `S` of cardinality `s > 0`,
     `∑_{j ∈ S} v̂_j² ≥ m₀` implies `∃ j ∈ S, v̂_j² ≥ m₀ / s`.
   - `perturbation_isolate_mass_lower`: If `∑_{j ∈ S} v_{+,j}² ≥ m` and `∑_{j ∈ S} (v̂_j - v_{+,j})² ≤ δ²`
     with `δ ≤ √m`, then `∑_{j ∈ S} v̂_j² ≥ (√m - δ)²`.
   - `output_coherence_omega_n_quarter`: With `s = ⌊t^(3/4)⌋`, `n = 8t + s` (`t ≥ 1`),
     * input coherence is bounded: `n / (8 t) ≤ 9 / 8`,
     * output coherence blows up polynomially: for any projector `P` containing a unit vector `v̂`
       with isolate mass `≥ m₀ > 0`, `(n / 8) * max_j P_{jj} ≥ m₀ * (n / 9)^(1/4)`, proving
       simultaneous `Ω(n^(1/4))` blow-up for both `A_{K̂}` and `Ā_{K̂}`!

Machine-checked vs prose-only:
- Machine-checked: Exact invertibility of the 8×8 input block `B`, exact input projector
  coherence `n/(8t)`, exact `p = 1/5` determinant collapse and orthogonal 3×3 reduction,
  exact characteristic polynomial, exact gap coefficients `16/125` and `5/72`, rigorous IVT
  existence and quantitative strict 8th–9th gap `λ_+ + λ_- ≥ (1/2) ε² (d - h²/a)` plus
  eigenvector isolate mass lower bound, projector diagonal inequality, perturbation mass
  transfer, pigeonhole, and the `Ω(n^(1/4))` blow-up vs `O(1)` input coherence.
- Prose-only (in `solutions/dp_pca_coherence/proof.pdf`): Random matrix Hoeffding + `1/4`-net
  concentration `‖W‖ = O(√n)` and degree concentration `O(√(n log n))` with probability `1 - o(1)`,
  plus Weyl's inequality linking `‖E‖ ≤ η` to the eigenvalue perturbation bound.
-/

open Matrix BigOperators Finset

noncomputable section

namespace DPPCACoherence

/-! ### 1. Input Graph `K` Rank and Exact Coherence (Lemma 1) -/

/-- Integer entries of the 8×8 quotient adjacency matrix `B = [[J₄ - I₄, I₄], [I₄, 0]]`. -/
def B_input_int : Matrix (Fin 8) (Fin 8) ℤ :=
  Matrix.of ![
    ![0, 1, 1, 1, 1, 0, 0, 0],
    ![1, 0, 1, 1, 0, 1, 0, 0],
    ![1, 1, 0, 1, 0, 0, 1, 0],
    ![1, 1, 1, 0, 0, 0, 0, 1],
    ![1, 0, 0, 0, 0, 0, 0, 0],
    ![0, 1, 0, 0, 0, 0, 0, 0],
    ![0, 0, 1, 0, 0, 0, 0, 0],
    ![0, 0, 0, 1, 0, 0, 0, 0]
  ]

/-- Integer entries of the exact inverse `B⁻¹ = [[0, I₄], [I₄, -(J₄ - I₄)]]`. -/
def B_input_inv_int : Matrix (Fin 8) (Fin 8) ℤ :=
  Matrix.of ![
    ![0, 0, 0, 0, 1, 0, 0, 0],
    ![0, 0, 0, 0, 0, 1, 0, 0],
    ![0, 0, 0, 0, 0, 0, 1, 0],
    ![0, 0, 0, 0, 0, 0, 0, 1],
    ![1, 0, 0, 0, 0, -1, -1, -1],
    ![0, 1, 0, 0, -1, 0, -1, -1],
    ![0, 0, 1, 0, -1, -1, 0, -1],
    ![0, 0, 0, 1, -1, -1, -1, 0]
  ]

/-- The 8×8 quotient adjacency matrix `B = [[J₄ - I₄, I₄], [I₄, 0]]` over `ℝ`. -/
def B_input : Matrix (Fin 8) (Fin 8) ℝ := B_input_int.map (Int.cast : ℤ → ℝ)

/-- Exact inverse `B⁻¹ = [[0, I₄], [I₄, -(J₄ - I₄)]]` over `ℝ`. -/
def B_input_inv : Matrix (Fin 8) (Fin 8) ℝ := B_input_inv_int.map (Int.cast : ℤ → ℝ)

/-- **Lemma 1 (Part 1 — Exact rank 8 of the input quotient matrix `B`)**:
`B_input` is symmetric and invertible with explicit inverse `B_input_inv`, certifying
`rank(A_K) = rank(Ā_K) = 8`. -/
theorem B_input_invertible :
    B_inputᵀ = B_input ∧ B_input * B_input_inv = 1 ∧ B_input_inv * B_input = 1 := by
  have h1 : B_input_intᵀ = B_input_int := by decide
  have h2 : B_input_int * B_input_inv_int = 1 := by decide
  have h3 : B_input_inv_int * B_input_int = 1 := by decide
  refine ⟨?_, ?_, ?_⟩
  · ext i j
    have := congrFun (congrFun h1 i) j
    simp only [B_input, Matrix.transpose_apply, Matrix.map_apply] at this ⊢
    exact congrArg (Int.cast : ℤ → ℝ) this
  · ext i j
    have := congrFun (congrFun h2 i) j
    simp only [B_input, B_input_inv, Matrix.mul_apply, Matrix.map_apply, Matrix.one_apply] at this ⊢
    have hcast : (∑ k : Fin 8, (B_input_int i k : ℝ) * (B_input_inv_int k j : ℝ)) =
        ((∑ k : Fin 8, B_input_int i k * B_input_inv_int k j : ℤ) : ℝ) := by push_cast; rfl
    rw [hcast, this]
    split_ifs <;> simp
  · ext i j
    have := congrFun (congrFun h3 i) j
    simp only [B_input, B_input_inv, Matrix.mul_apply, Matrix.map_apply, Matrix.one_apply] at this ⊢
    have hcast : (∑ k : Fin 8, (B_input_inv_int i k : ℝ) * (B_input_int k j : ℝ)) =
        ((∑ k : Fin 8, B_input_inv_int i k * B_input_int k j : ℤ) : ℝ) := by push_cast; rfl
    rw [hcast, this]
    split_ifs <;> simp

/-- The 8-class indicator orthogonal projector `P₈` on `n = 8t + s` vertices. -/
noncomputable def inputProj (t s : ℕ) : Matrix (Fin (8 * t + s)) (Fin (8 * t + s)) ℝ :=
  Matrix.of fun i j =>
    if i.val < 8 * t ∧ j.val < 8 * t ∧ i.val / t = j.val / t then (t : ℝ)⁻¹ else 0

/-- **Lemma 1 (Part 2 — Exact input rank-8 projector properties and coherence `n / (8 t)`)**:
For every `t ≥ 1` and `s : ℕ`, `P₈ = inputProj t s` is symmetric, idempotent (`P₈² = P₈`),
has trace `8`, diagonal entries `≤ 1/t` (attained on non-isolates), and exact rank-8
coherence `(n / 8) * (1 / t) = n / (8 t)`. -/
theorem input_projector_coherence (t s : ℕ) (ht : 0 < t) :
    let n := 8 * t + s
    let P := inputProj t s
    Pᵀ = P ∧
    P * P = P ∧
    Matrix.trace P = 8 ∧
    (∀ j : Fin n, P j j ≤ (t : ℝ)⁻¹) ∧
    (∃ j : Fin n, P j j = (t : ℝ)⁻¹) ∧
    ((n : ℝ) / 8) * (t : ℝ)⁻¹ = (n : ℝ) / (8 * (t : ℝ)) := by
  dsimp only
  have hsym : (inputProj t s)ᵀ = inputProj t s := by
    ext i j
    simp only [Matrix.transpose_apply, inputProj, Matrix.of_apply]
    by_cases h : j.val < 8 * t ∧ i.val < 8 * t ∧ j.val / t = i.val / t
    · rw [if_pos h, if_pos ⟨h.2.1, h.1, h.2.2.symm⟩]
    · rw [if_neg h, if_neg (fun h' => h ⟨h'.2.1, h'.1, h'.2.2.symm⟩)]
  have hblock_card : ∀ c : ℕ, c < 8 →
      (Finset.univ.filter fun k : Fin (8 * t + s) => k.val < 8 * t ∧ k.val / t = c).card = t := by
    intro c hc
    have hbij : (Finset.univ.filter fun k : Fin (8 * t + s) => k.val < 8 * t ∧ k.val / t = c) =
        Finset.image (fun r : Fin t => ⟨c * t + r.val, by nlinarith [r.isLt]⟩) Finset.univ := by
      ext k
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_image]
      constructor
      · rintro ⟨hk, hdiv⟩
        refine ⟨⟨k.val % t, Nat.mod_lt _ ht⟩, ?_⟩
        ext; dsimp only
        have := Nat.div_add_mod k.val t
        rw [hdiv] at this
        linarith
      · rintro ⟨r, rfl⟩
        refine ⟨by nlinarith [r.isLt], ?_⟩
        show (c * t + r.val) / t = c
        rw [Nat.mul_comm, Nat.mul_add_div ht, Nat.div_eq_of_lt r.isLt, add_zero]
    rw [hbij, Finset.card_image_of_injective]
    · simp
    · intro r₁ r₂ h
      have hval : c * t + r₁.val = c * t + r₂.val := congrArg Fin.val h
      exact Fin.ext (by omega)
  have hidem : inputProj t s * inputProj t s = inputProj t s := by
    ext i j
    simp only [Matrix.mul_apply, inputProj, Matrix.of_apply]
    by_cases hij : i.val < 8 * t ∧ j.val < 8 * t ∧ i.val / t = j.val / t
    · rw [if_pos hij]
      have hterm : ∀ k : Fin (8 * t + s),
          (if i.val < 8 * t ∧ k.val < 8 * t ∧ i.val / t = k.val / t then (t : ℝ)⁻¹ else 0) *
          (if k.val < 8 * t ∧ j.val < 8 * t ∧ k.val / t = j.val / t then (t : ℝ)⁻¹ else 0) =
          if k.val < 8 * t ∧ k.val / t = i.val / t then (t : ℝ)⁻¹ * (t : ℝ)⁻¹ else 0 := by
        intro k
        by_cases hk : k.val < 8 * t ∧ k.val / t = i.val / t
        · rw [if_pos ⟨hij.1, hk.1, hk.2.symm⟩, if_pos ⟨hk.1, hij.2.1, hk.2.trans hij.2.2⟩, if_pos hk]
        · rw [if_neg hk]
          by_cases h1 : i.val < 8 * t ∧ k.val < 8 * t ∧ i.val / t = k.val / t
          · exact (hk ⟨h1.2.1, h1.2.2.symm⟩).elim
          · simp [h1]
      simp_rw [hterm, Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const, nsmul_eq_mul]
      have hc_lt : i.val / t < 8 := Nat.div_lt_of_lt_mul (by linarith [hij.1])
      rw [hblock_card (i.val / t) hc_lt]
      have htne : (t : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.ne_of_gt ht)
      field_simp [htne]
    · rw [if_neg hij]
      refine Finset.sum_eq_zero fun k _ => ?_
      by_cases h1 : i.val < 8 * t ∧ k.val < 8 * t ∧ i.val / t = k.val / t
      · by_cases h2 : k.val < 8 * t ∧ j.val < 8 * t ∧ k.val / t = j.val / t
        · exact (hij ⟨h1.1, h2.2.1, h1.2.2.trans h2.2.2⟩).elim
        · rw [if_neg h2, mul_zero]
      · rw [if_neg h1, zero_mul]
  have htr : Matrix.trace (inputProj t s) = 8 := by
    simp only [Matrix.trace, Matrix.diag, inputProj, Matrix.of_apply, and_self, and_true]
    rw [Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const, nsmul_eq_mul]
    have hcard : (Finset.univ.filter fun i : Fin (8 * t + s) => i.val < 8 * t).card = 8 * t := by
      have hbij : (Finset.univ.filter fun i : Fin (8 * t + s) => i.val < 8 * t) =
          Finset.image (fun r : Fin (8 * t) => (⟨r.val, by omega⟩ : Fin (8 * t + s))) Finset.univ := by
        ext i
        simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_image]
        exact ⟨fun hi => ⟨⟨i.val, hi⟩, rfl⟩, fun ⟨r, hr⟩ => by rw [← hr]; exact r.isLt⟩
      rw [hbij, Finset.card_image_of_injective]
      · simp
      · intro r₁ r₂ h; exact Fin.ext (by simpa using congrArg Fin.val h)
    rw [hcard]
    have htne : (t : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.ne_of_gt ht)
    push_cast
    field_simp [htne]
  have hle : ∀ j : Fin (8 * t + s), inputProj t s j j ≤ (t : ℝ)⁻¹ := by
    intro j
    simp only [inputProj, Matrix.of_apply]
    split_ifs
    · exact le_rfl
    · positivity
  have hex : ∃ j : Fin (8 * t + s), inputProj t s j j = (t : ℝ)⁻¹ := by
    refine ⟨⟨0, by omega⟩, ?_⟩
    simp only [inputProj, Matrix.of_apply]
    split_ifs with h
    · rfl
    · exfalso; exact h ⟨by omega, by omega, trivial⟩
  refine ⟨hsym, hidem, htr, hle, hex, by ring⟩

/-! ### 2. Exact 9-Dimensional Quotient Reduction at `p = 1/5` (Lemma 2) -/

/-- Critical determinant collapse of the center–leaf 2×2 block at `p = 1/5`:
`det [[3 + p, 1 + 3p], [1 + 3p, 4p]] = -(5p - 1)(p - 1)`, which vanishes at `p = 1/5`. -/
theorem critical_det_collapse (p : ℝ) :
    (3 + p) * (4 * p) - (1 + 3 * p) ^ 2 = -(5 * p - 1) * (p - 1) := by
  ring

/-- The 3×3 block `M₃(c, l, i, ε)` on the normalized indicators of centers, leaves, and `S`. -/
def M3 (c l i ε : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.of ![
    ![(16 / 5) * c ^ 2, (8 / 5) * c * l, (2 / 5) * ε * c * i],
    ![(8 / 5) * c * l, (4 / 5) * l ^ 2, (2 / 5) * ε * l * i],
    ![(2 / 5) * ε * c * i, (2 / 5) * ε * l * i, (1 / 5) * ε ^ 2 * i ^ 2]
  ]

/-- The reduced 3×3 matrix `L₃(a, b, h, d, ε) = [[a, 0, ε h], [0, 0, ε b], [ε h, ε b, ε² d]]`. -/
def L3 (a b h d ε : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.of ![
    ![a, 0, ε * h],
    ![0, 0, ε * b],
    ![ε * h, ε * b, ε ^ 2 * d]
  ]

/-- **Lemma 2 (Orthogonal reduction of `M₃` to `L₃`)**:
In the orthonormal basis `{v/‖v‖, u/‖v‖, e₃}` where `v = (2c, l)ᵀ` and `u = (-l, 2c)ᵀ`,
`Qᵀ * M₃(c, l, i, ε) * Q = L₃(a, b, h, d, ε)` with `Qᵀ Q = 1`. -/
theorem quotient_block_orthogonal_reduction (c l i ε : ℝ) (hv : 0 < 4 * c ^ 2 + l ^ 2) :
    let r := Real.sqrt (4 * c ^ 2 + l ^ 2)
    let Q : Matrix (Fin 3) (Fin 3) ℝ := Matrix.of ![
      ![2 * c / r, -l / r, 0],
      ![l / r, 2 * c / r, 0],
      ![0, 0, 1]
    ]
    let a := (4 / 5) * (4 * c ^ 2 + l ^ 2)
    let h := (2 * i / 5) * (2 * c ^ 2 + l ^ 2) / r
    let b := (2 * i / 5) * (c * l) / r
    let d := (1 / 5) * i ^ 2
    Qᵀ * Q = 1 ∧ Qᵀ * M3 c l i ε * Q = L3 a b h d ε := by
  dsimp only
  set r := Real.sqrt (4 * c ^ 2 + l ^ 2)
  have hr2 : r ^ 2 = 4 * c ^ 2 + l ^ 2 := Real.sq_sqrt hv.le
  have hrne : r ≠ 0 := Real.sqrt_ne_zero'.mpr hv
  constructor
  · ext p q; fin_cases p <;> fin_cases q <;>
      simp [Matrix.mul_apply, Matrix.transpose_apply, Fin.sum_univ_three] <;>
      field_simp [hrne] <;> nlinarith [hr2]
  · ext p q; fin_cases p <;> fin_cases q <;>
      simp [M3, L3, Matrix.mul_apply, Matrix.transpose_apply, Fin.sum_univ_three] <;>
      field_simp [hrne] <;> nlinarith [hr2]

/-- Exact characteristic polynomial of `L₃(a, b, h, d, ε)`:
`det(λ I - L₃) = λ³ - (a + ε² d) λ² - ε² (h² + b² - a d) λ + a ε² b²`. -/
def L3Poly (a b h d ε lam : ℝ) : ℝ :=
  lam ^ 3 - (a + ε ^ 2 * d) * lam ^ 2 - ε ^ 2 * (h ^ 2 + b ^ 2 - a * d) * lam +
    a * ε ^ 2 * b ^ 2

theorem L3_charpoly_exact (a b h d ε lam : ℝ) :
    Matrix.det (lam • (1 : Matrix (Fin 3) (Fin 3) ℝ) - L3 a b h d ε) =
      L3Poly a b h d ε lam := by
  rw [Matrix.det_fin_three]
  simp [L3, L3Poly]
  ring

/-- **Lemma 2 (Exact gap coefficient formula and both raw & normalized evaluations)**:
`d - h² / a = (i² / 5) * (1 - ((2c² + l²) / (4c² + l²))²)`,
evaluating to `16 / 125` (raw) and `5 / 72` (normalized). -/
theorem gap_coeff_identities (c l i : ℝ) :
    let r2 := 4 * c ^ 2 + l ^ 2
    let a := (4 / 5) * r2
    let h2 := ((2 * i / 5) * (2 * c ^ 2 + l ^ 2)) ^ 2 / r2
    let d := (1 / 5) * i ^ 2
    d - h2 / a = (i ^ 2 / 5) * (1 - ((2 * c ^ 2 + l ^ 2) / (4 * c ^ 2 + l ^ 2)) ^ 2) := by
  dsimp only
  by_cases hv : 4 * c ^ 2 + l ^ 2 = 0
  · simp [hv]; ring
  · field_simp [hv]; ring

theorem gap_coeff_raw_and_normalized :
    ((1 : ℝ) ^ 2 / 5) * (1 - ((2 * 1 ^ 2 + 1 ^ 2) / (4 * 1 ^ 2 + 1 ^ 2)) ^ 2) = 16 / 125 ∧
    ((5 / 8 : ℝ) / 5) * (1 - ((2 * (5 / 24) + 5 / 12) / (4 * (5 / 24) + 5 / 12)) ^ 2) = 5 / 72 := by
  constructor <;> norm_num

/-- **Lemma 2 (Difference subspace 2×2 block determinant)**:
At `p = 1/5`, each of the three 2×2 difference blocks `(4/5) [[-c², c l], [c l, 0]]`
has exact nonzero determinant `-(16/25) c² l²`. -/
theorem difference_block_det (c l : ℝ) :
    Matrix.det (Matrix.of ![![-(4 / 5 : ℝ) * c ^ 2, (4 / 5) * c * l],
                            ![(4 / 5) * c * l, 0]]) = -(16 / 25 : ℝ) * c ^ 2 * l ^ 2 := by
  rw [Matrix.det_fin_two]
  simp
  ring

lemma L3Poly_continuous (a b h d ε : ℝ) : Continuous (L3Poly a b h d ε) := by
  unfold L3Poly; continuity

lemma L3Poly_vieta_sum {a b h d ε lam_neg lam_pos lam_0 : ℝ}
    (hne1 : lam_neg ≠ lam_pos) (hne2 : lam_neg ≠ lam_0) (hne3 : lam_pos ≠ lam_0)
    (e1 : L3Poly a b h d ε lam_neg = 0)
    (e2 : L3Poly a b h d ε lam_pos = 0)
    (e3 : L3Poly a b h d ε lam_0 = 0) :
    lam_neg + lam_pos + lam_0 = a + ε ^ 2 * d := by
  simp only [L3Poly] at e1 e2 e3
  have d12 : lam_pos - lam_neg ≠ 0 := sub_ne_zero.mpr hne1.symm
  have d13 : lam_0 - lam_neg ≠ 0 := sub_ne_zero.mpr hne2.symm
  have d23 : lam_0 - lam_pos ≠ 0 := sub_ne_zero.mpr hne3.symm
  have q12 : (lam_pos ^ 2 + lam_pos * lam_neg + lam_neg ^ 2) -
      (a + ε ^ 2 * d) * (lam_pos + lam_neg) - ε ^ 2 * (h ^ 2 + b ^ 2 - a * d) = 0 := by
    have hdiff : (lam_pos - lam_neg) * ((lam_pos ^ 2 + lam_pos * lam_neg + lam_neg ^ 2) -
        (a + ε ^ 2 * d) * (lam_pos + lam_neg) - ε ^ 2 * (h ^ 2 + b ^ 2 - a * d)) = 0 := by
      linarith
    exact (mul_eq_zero.mp hdiff).resolve_left d12
  have q23 : (lam_0 ^ 2 + lam_0 * lam_pos + lam_pos ^ 2) -
      (a + ε ^ 2 * d) * (lam_0 + lam_pos) - ε ^ 2 * (h ^ 2 + b ^ 2 - a * d) = 0 := by
    have hdiff : (lam_0 - lam_pos) * ((lam_0 ^ 2 + lam_0 * lam_pos + lam_pos ^ 2) -
        (a + ε ^ 2 * d) * (lam_0 + lam_pos) - ε ^ 2 * (h ^ 2 + b ^ 2 - a * d)) = 0 := by
      linarith
    exact (mul_eq_zero.mp hdiff).resolve_left d23
  have qdiff : (lam_0 - lam_neg) * (lam_neg + lam_pos + lam_0 - (a + ε ^ 2 * d)) = 0 := by
    linarith
  linarith [(mul_eq_zero.mp qdiff).resolve_left d13]

lemma L3Poly_eval_signs {a b h d ε : ℝ}
    (ha : 0 < a) (hb : 0 < b) (hε : 0 < ε)
    (hd_C1 : ε ^ 2 * |d| < a / 8) (hb_C1 : ε ^ 2 * b ^ 2 < a ^ 2 / 8)
    (hd_C2 : ε * |d| < b / 8) (hh_C2 : ε * h ^ 2 < a * b / 8)
    (hsmall_b : ε * b < a / 16) :
    0 < L3Poly a b h d ε 0 ∧
    0 < L3Poly a b h d ε ((4 / 5) * ε * b) ∧
    L3Poly a b h d ε (a / 2) < 0 ∧
    L3Poly a b h d ε (-2 * ε * b) < 0 ∧
    L3Poly a b h d ε a ≤ 0 ∧
    0 < L3Poly a b h d ε (a + ε ^ 2 * |d| + ε ^ 2 * (h ^ 2 + b ^ 2) / a + 1) := by
  have hp0 : 0 < L3Poly a b h d ε 0 := by simp only [L3Poly]; ring_nf; positivity
  have hp_lo : 0 < L3Poly a b h d ε ((4 / 5) * ε * b) := by
    simp only [L3Poly]
    have hfac : ((4 / 5) * ε * b) ^ 3 - (a + ε ^ 2 * d) * ((4 / 5) * ε * b) ^ 2 -
        ε ^ 2 * (h ^ 2 + b ^ 2 - a * d) * ((4 / 5) * ε * b) + a * ε ^ 2 * b ^ 2 =
        ε ^ 2 * ((9 / 25) * a * b ^ 2 + (64 / 125) * ε * b ^ 3 - (16 / 25) * ε ^ 2 * d * b ^ 2 -
          (4 / 5) * ε * b * h ^ 2 - (4 / 5) * ε * b ^ 3 + (4 / 5) * ε * b * a * d) := by ring
    rw [hfac]
    have hd_lt : ε ^ 2 * d < a / 8 :=
      lt_of_le_of_lt (mul_le_mul_of_nonneg_left (le_abs_self d) (sq_nonneg ε)) hd_C1
    have h1 : (16 / 25) * ε ^ 2 * d * b ^ 2 < (2 / 25) * a * b ^ 2 := by
      have := mul_lt_mul_of_pos_right hd_lt (sq_pos_of_pos hb); linarith
    have h2 : (4 / 5) * ε * b * h ^ 2 < (1 / 10) * a * b ^ 2 := by
      have := mul_lt_mul_of_pos_right hh_C2 hb; linarith
    have h3 : (4 / 5) * ε * b ^ 3 < (1 / 20) * a * b ^ 2 := by
      have := mul_lt_mul_of_pos_right hsmall_b (sq_pos_of_pos hb); linarith
    have h4 : -((4 / 5) * ε * b * a * d) < (1 / 10) * a * b ^ 2 := by
      have hm : ε * (-d) < b / 8 :=
        lt_of_le_of_lt (mul_le_mul_of_nonneg_left (neg_le_abs d) hε.le) hd_C2
      have := mul_lt_mul_of_pos_right hm (mul_pos ha hb); linarith
    have hpos3 : 0 < (64 / 125) * ε * b ^ 3 := by positivity
    have hab2 : 0 < a * b ^ 2 := mul_pos ha (sq_pos_of_pos hb)
    exact mul_pos (sq_pos_of_pos hε) (by linarith)
  have hpa2 : L3Poly a b h d ε (a / 2) < 0 := by
    simp only [L3Poly]
    have hrew : (a / 2) ^ 3 - (a + ε ^ 2 * d) * (a / 2) ^ 2 -
        ε ^ 2 * (h ^ 2 + b ^ 2 - a * d) * (a / 2) + a * ε ^ 2 * b ^ 2 =
        -(a ^ 3) / 8 + (1 / 4) * a ^ 2 * (ε ^ 2 * d) - (1 / 2) * a * (ε ^ 2 * h ^ 2) +
          (1 / 2) * a * (ε ^ 2 * b ^ 2) := by ring
    rw [hrew]
    have hd_lt : ε ^ 2 * d < a / 8 :=
      lt_of_le_of_lt (mul_le_mul_of_nonneg_left (le_abs_self d) (sq_nonneg ε)) hd_C1
    have ht1 : (1 / 4) * a ^ 2 * (ε ^ 2 * d) < a ^ 3 / 32 := by
      have := mul_lt_mul_of_pos_left hd_lt (sq_pos_of_pos ha); linarith
    have ht2 : 0 ≤ (1 / 2) * a * (ε ^ 2 * h ^ 2) := by positivity
    have ht3 : (1 / 2) * a * (ε ^ 2 * b ^ 2) < a ^ 3 / 16 := by
      have := mul_lt_mul_of_pos_left hb_C1 ha; linarith
    have ha3 : 0 < a ^ 3 := pow_pos ha 3
    linarith
  have hp_neg : L3Poly a b h d ε (-2 * ε * b) < 0 := by
    simp only [L3Poly]
    have hfac : (-2 * ε * b) ^ 3 - (a + ε ^ 2 * d) * (-2 * ε * b) ^ 2 -
        ε ^ 2 * (h ^ 2 + b ^ 2 - a * d) * (-2 * ε * b) + a * ε ^ 2 * b ^ 2 =
        ε ^ 2 * (-3 * a * b ^ 2 - 6 * ε * b ^ 3 - 4 * ε ^ 2 * d * b ^ 2 +
          2 * ε * b * h ^ 2 - 2 * ε * b * a * d) := by ring
    rw [hfac]
    have h1 : -4 * ε ^ 2 * d * b ^ 2 < (1 / 2) * a * b ^ 2 := by
      have hm : ε ^ 2 * (-d) < a / 8 :=
        lt_of_le_of_lt (mul_le_mul_of_nonneg_left (neg_le_abs d) (sq_nonneg ε)) hd_C1
      have := mul_lt_mul_of_pos_right hm (sq_pos_of_pos hb); linarith
    have h2 : -2 * ε * b * a * d < (1 / 4) * a * b ^ 2 := by
      have hm : ε * (-d) < b / 8 :=
        lt_of_le_of_lt (mul_le_mul_of_nonneg_left (neg_le_abs d) hε.le) hd_C2
      have := mul_lt_mul_of_pos_right hm (mul_pos ha hb); linarith
    have h3 : 2 * ε * b * h ^ 2 < (1 / 4) * a * b ^ 2 := by
      have := mul_lt_mul_of_pos_right hh_C2 hb; linarith
    have h4 : 0 < 6 * ε * b ^ 3 := by positivity
    have hab2 : 0 < a * b ^ 2 := mul_pos ha (sq_pos_of_pos hb)
    exact mul_neg_of_pos_of_neg (sq_pos_of_pos hε) (by linarith)
  have hpa : L3Poly a b h d ε a ≤ 0 := by
    simp only [L3Poly]
    have : a ^ 3 - (a + ε ^ 2 * d) * a ^ 2 - ε ^ 2 * (h ^ 2 + b ^ 2 - a * d) * a + a * ε ^ 2 * b ^ 2 =
        -(a * ε ^ 2 * h ^ 2) := by ring
    rw [this]; exact neg_nonpos.mpr (by positivity)
  have hpM : 0 < L3Poly a b h d ε (a + ε ^ 2 * |d| + ε ^ 2 * (h ^ 2 + b ^ 2) / a + 1) := by
    set M := a + ε ^ 2 * |d| + ε ^ 2 * (h ^ 2 + b ^ 2) / a + 1
    simp only [L3Poly]
    have hM_gt : a < M := by
      simp only [M]; linarith [show (0 : ℝ) ≤ ε ^ 2 * |d| + ε ^ 2 * (h ^ 2 + b ^ 2) / a by positivity]
    have hMpos : 0 < M := lt_trans ha hM_gt
    have hdiff_ge : ε ^ 2 * (-d) + ε ^ 2 * (h ^ 2 + b ^ 2) / a < M - (a + ε ^ 2 * d) := by
      simp only [M]
      have h1 : ε ^ 2 * (-d) ≤ ε ^ 2 * |d| := mul_le_mul_of_nonneg_left (neg_le_abs d) (sq_nonneg ε)
      have h2 : 0 ≤ ε ^ 2 * |d| + ε ^ 2 * (-d) := by
        have := mul_le_mul_of_nonneg_left (le_abs_self d) (sq_nonneg ε); linarith
      linarith
    have hmul : a * (ε ^ 2 * (-d) + ε ^ 2 * (h ^ 2 + b ^ 2) / a) =
        ε ^ 2 * (h ^ 2 + b ^ 2 - a * d) := by field_simp [ha.ne']; ring
    have hdiff_pos : 0 < M - (a + ε ^ 2 * d) := by
      simp only [M]
      have h1 : ε ^ 2 * d ≤ ε ^ 2 * |d| := mul_le_mul_of_nonneg_left (le_abs_self d) (sq_nonneg ε)
      have h2 : 0 ≤ ε ^ 2 * (h ^ 2 + b ^ 2) / a := by positivity
      linarith
    have h2 : ε ^ 2 * (h ^ 2 + b ^ 2 - a * d) < M * (M - (a + ε ^ 2 * d)) := by
      have hs1 : a * (ε ^ 2 * (-d) + ε ^ 2 * (h ^ 2 + b ^ 2) / a) < a * (M - (a + ε ^ 2 * d)) :=
        mul_lt_mul_of_pos_left hdiff_ge ha
      have hs2 : a * (M - (a + ε ^ 2 * d)) < M * (M - (a + ε ^ 2 * d)) :=
        mul_lt_mul_of_pos_right hM_gt hdiff_pos
      linarith
    have h3 : 0 < a * ε ^ 2 * b ^ 2 := by positivity
    have h4 : M * (ε ^ 2 * (h ^ 2 + b ^ 2 - a * d)) < M * (M * (M - (a + ε ^ 2 * d))) :=
      mul_lt_mul_of_pos_left h2 hMpos
    linarith
  exact ⟨hp0, hp_lo, hpa2, hp_neg, hpa, hpM⟩

private lemma L3_eig_and_mass {a b h d ε lam_pos : ℝ}
    (ha : 0 < a) (hb : 0 < b) (hε : 0 < ε)
    (hp_pos_eq : L3Poly a b h d ε lam_pos = 0)
    (hpos_lo : (4 / 5) * ε * b < lam_pos)
    (hpos_hi : lam_pos < a / 2)
    (hpos_gt0 : 0 < lam_pos)
    (hsmall_h : ε * |h| < a / 8) :
    let x := -ε * h / (a - lam_pos)
    let y := ε * b / lam_pos
    let z := (1 : ℝ)
    (L3 a b h d ε).mulVec ![x, y, z] = lam_pos • ![x, y, z] ∧
    (1 / 3 : ℝ) ≤ z ^ 2 / (x ^ 2 + y ^ 2 + z ^ 2) := by
  dsimp only
  constructor
  · have haml : a - lam_pos ≠ 0 := by linarith only [ha, hpos_hi]
    have hlp : lam_pos ≠ 0 := hpos_gt0.ne'
    ext i; fin_cases i
    · simp [L3, Matrix.mulVec, dotProduct, Fin.sum_univ_three]
      field_simp [haml]; ring
    · simp [L3, Matrix.mulVec, dotProduct, Fin.sum_univ_three]
      field_simp [hlp]
    · simp [L3, Matrix.mulVec, dotProduct, Fin.sum_univ_three]
      have e2 := hp_pos_eq
      simp only [L3Poly] at e2
      field_simp [haml, hlp]
      linarith only [e2]
  · have hx_bound : (-ε * h / (a - lam_pos)) ^ 2 ≤ 1 / 16 := by
      have habs : |-ε * h / (a - lam_pos)| ≤ 1 / 4 := by
        rw [abs_div, abs_mul, abs_neg, abs_of_pos hε,
          abs_of_pos (show 0 < a - lam_pos by linarith only [ha, hpos_hi])]
        rw [div_le_iff₀ (show 0 < a - lam_pos by linarith only [ha, hpos_hi])]
        linarith only [hsmall_h, hpos_hi]
      rw [← sq_abs]
      have hnn := abs_nonneg (-ε * h / (a - lam_pos))
      have hmul := mul_le_mul habs habs hnn (by norm_num)
      linarith only [hmul, sq |-ε * h / (a - lam_pos)|]
    have hy_bound : (ε * b / lam_pos) ^ 2 ≤ 25 / 16 := by
      have hy_pos : 0 ≤ ε * b / lam_pos := by positivity
      have hy_le : ε * b / lam_pos ≤ 5 / 4 := by
        rw [div_le_iff₀ hpos_gt0]; linarith only [hpos_lo]
      have hmul := mul_le_mul hy_le hy_le hy_pos (by norm_num)
      linarith only [hmul, sq (ε * b / lam_pos)]
    have hden_pos : 0 < (-ε * h / (a - lam_pos)) ^ 2 + (ε * b / lam_pos) ^ 2 + 1 ^ 2 := by
      positivity
    rw [div_le_div_iff₀ (by norm_num : (0 : ℝ) < 3) hden_pos]
    linarith only [hx_bound, hy_bound]

private lemma L3_lam0_upper_bound {a b h d ε C₃ lam_0 : ℝ}
    (ha : 0 < a) (hε : 0 < ε) (h0_lo : a ≤ lam_0)
    (hp_0_eq : L3Poly a b h d ε lam_0 = 0)
    (hC3_ge : (4 / 3) * (h ^ 2 / a) * (|d| / a + b ^ 2 / a ^ 2) ≤ C₃)
    (hsmall_C3 : ε ^ 2 * C₃ < (d - h ^ 2 / a) / 2)
    (hd_C1 : ε ^ 2 * |d| < a / 8)
    (hb_C1 : ε ^ 2 * b ^ 2 < a ^ 2 / 8) :
    lam_0 - a ≤ ε ^ 2 * (h ^ 2 / a) + (1 / 2) * ε ^ 2 * (d - h ^ 2 / a) := by
  have e3 := hp_0_eq
  simp only [L3Poly] at e3
  have hlam0_pos : 0 < lam_0 := lt_of_lt_of_le ha h0_lo
  have hre : (lam_0 - a) * (lam_0 ^ 2 - ε ^ 2 * d * lam_0 - ε ^ 2 * b ^ 2) =
      ε ^ 2 * h ^ 2 * lam_0 := by linarith only [e3]
  have hden_ge : (3 / 4) * lam_0 ^ 2 ≤ lam_0 ^ 2 - ε ^ 2 * d * lam_0 - ε ^ 2 * b ^ 2 := by
    have hd_le : ε ^ 2 * d ≤ a / 8 :=
      le_trans (mul_le_mul_of_nonneg_left (le_abs_self d) (sq_nonneg ε)) hd_C1.le
    have h1 : ε ^ 2 * d * lam_0 ≤ (1 / 8) * lam_0 ^ 2 := by
      calc ε ^ 2 * d * lam_0 ≤ (a / 8) * lam_0 := mul_le_mul_of_nonneg_right hd_le hlam0_pos.le
        _ ≤ (lam_0 / 8) * lam_0 := mul_le_mul_of_nonneg_right (by linarith only [h0_lo]) hlam0_pos.le
        _ = (1 / 8) * lam_0 ^ 2 := by ring
    have h2 : ε ^ 2 * b ^ 2 ≤ (1 / 8) * lam_0 ^ 2 := by
      have ha2 : a ^ 2 ≤ lam_0 ^ 2 := pow_le_pow_left₀ ha.le h0_lo 2
      linarith only [hb_C1, ha2]
    linarith only [h1, h2]
  have hrough : lam_0 - a ≤ (4 / 3) * (ε ^ 2 * h ^ 2 / a) := by
    have h1 : (lam_0 - a) * ((3 / 4) * lam_0 ^ 2) ≤ ε ^ 2 * h ^ 2 * lam_0 := by
      calc (lam_0 - a) * ((3 / 4) * lam_0 ^ 2)
          ≤ (lam_0 - a) * (lam_0 ^ 2 - ε ^ 2 * d * lam_0 - ε ^ 2 * b ^ 2) :=
            mul_le_mul_of_nonneg_left hden_ge (by linarith only [h0_lo])
        _ = ε ^ 2 * h ^ 2 * lam_0 := hre
    have h2 : (lam_0 - a) * ((3 / 4) * lam_0) ≤ ε ^ 2 * h ^ 2 := by
      have : ((lam_0 - a) * ((3 / 4) * lam_0)) * lam_0 ≤ (ε ^ 2 * h ^ 2) * lam_0 := by linarith only [h1]
      exact le_of_mul_le_mul_right this hlam0_pos
    have h3 : (lam_0 - a) * ((3 / 4) * a) ≤ (lam_0 - a) * ((3 / 4) * lam_0) :=
      mul_le_mul_of_nonneg_left (by linarith only [h0_lo]) (by linarith only [h0_lo])
    rw [mul_div_assoc', le_div_iff₀ ha]; linarith only [h2, h3]
  have hexact : lam_0 - a = ε ^ 2 * h ^ 2 / lam_0 +
      (lam_0 - a) * (ε ^ 2 * d / lam_0 + ε ^ 2 * b ^ 2 / lam_0 ^ 2) := by
    have hne : lam_0 ≠ 0 := hlam0_pos.ne'
    field_simp [hne]; linarith only [e3]
  have hterm1 : ε ^ 2 * h ^ 2 / lam_0 ≤ ε ^ 2 * (h ^ 2 / a) := by
    rw [mul_div_assoc', div_le_div_iff₀ hlam0_pos ha]
    exact mul_le_mul_of_nonneg_left h0_lo (by positivity)
  have hterm2_br : ε ^ 2 * d / lam_0 + ε ^ 2 * b ^ 2 / lam_0 ^ 2 ≤
      ε ^ 2 * (|d| / a + b ^ 2 / a ^ 2) := by
    have hd1 : ε ^ 2 * d / lam_0 ≤ ε ^ 2 * |d| / a := by
      rw [div_le_div_iff₀ hlam0_pos ha]
      have h1 : ε ^ 2 * d * a ≤ ε ^ 2 * |d| * a :=
        mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left (le_abs_self d) (sq_nonneg ε)) ha.le
      have h2 : ε ^ 2 * |d| * a ≤ ε ^ 2 * |d| * lam_0 :=
        mul_le_mul_of_nonneg_left h0_lo (by positivity)
      exact h1.trans h2
    have hb1 : ε ^ 2 * b ^ 2 / lam_0 ^ 2 ≤ ε ^ 2 * b ^ 2 / a ^ 2 := by
      rw [div_le_div_iff₀ (sq_pos_of_pos hlam0_pos) (sq_pos_of_pos ha)]
      have : a ^ 2 ≤ lam_0 ^ 2 := pow_le_pow_left₀ ha.le h0_lo 2
      exact mul_le_mul_of_nonneg_left this (by positivity)
    have hsum := add_le_add hd1 hb1
    have heq : ε ^ 2 * |d| / a + ε ^ 2 * b ^ 2 / a ^ 2 = ε ^ 2 * (|d| / a + b ^ 2 / a ^ 2) := by ring
    exact hsum.trans_eq heq
  have hterm2 : (lam_0 - a) * (ε ^ 2 * d / lam_0 + ε ^ 2 * b ^ 2 / lam_0 ^ 2) ≤
      (1 / 2) * ε ^ 2 * (d - h ^ 2 / a) := by
    have h1 : (lam_0 - a) * (ε ^ 2 * d / lam_0 + ε ^ 2 * b ^ 2 / lam_0 ^ 2) ≤
        ((4 / 3) * (ε ^ 2 * h ^ 2 / a)) * (ε ^ 2 * (|d| / a + b ^ 2 / a ^ 2)) := by
      by_cases hneg_br : ε ^ 2 * d / lam_0 + ε ^ 2 * b ^ 2 / lam_0 ^ 2 ≤ 0
      · calc (lam_0 - a) * (ε ^ 2 * d / lam_0 + ε ^ 2 * b ^ 2 / lam_0 ^ 2) ≤ 0 :=
              mul_nonpos_of_nonneg_of_nonpos (by linarith only [h0_lo]) hneg_br
          _ ≤ _ := by positivity
      · push_neg at hneg_br
        exact mul_le_mul hrough hterm2_br hneg_br.le (by positivity)
    have h2 : ((4 / 3) * (ε ^ 2 * h ^ 2 / a)) * (ε ^ 2 * (|d| / a + b ^ 2 / a ^ 2)) ≤
        ε ^ 2 * (ε ^ 2 * C₃) := by
      calc ((4 / 3) * (ε ^ 2 * h ^ 2 / a)) * (ε ^ 2 * (|d| / a + b ^ 2 / a ^ 2))
          = ε ^ 2 * (ε ^ 2 * ((4 / 3) * (h ^ 2 / a) * (|d| / a + b ^ 2 / a ^ 2))) := by ring
        _ ≤ ε ^ 2 * (ε ^ 2 * C₃) :=
            mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hC3_ge (sq_nonneg ε)) (sq_nonneg ε)
    have h3 : ε ^ 2 * (ε ^ 2 * C₃) ≤ (1 / 2) * ε ^ 2 * (d - h ^ 2 / a) := by
      have h3' := mul_le_mul_of_nonneg_left hsmall_C3.le (sq_nonneg ε)
      have heq : ε ^ 2 * ((d - h ^ 2 / a) / 2) = (1 / 2) * ε ^ 2 * (d - h ^ 2 / a) := by ring
      exact h3'.trans_eq heq
    exact h1.trans (h2.trans h3)
  exact hexact.le.trans (add_le_add hterm1 hterm2)

private lemma L3_root_gap_and_eig {a b h d ε C₃ : ℝ}
    (ha : 0 < a) (hb : 0 < b) (hε : 0 < ε)
    (hC3_ge : (4 / 3) * (h ^ 2 / a) * (|d| / a + b ^ 2 / a ^ 2) ≤ C₃)
    (hsmall_C3 : ε ^ 2 * C₃ < (d - h ^ 2 / a) / 2)
    (hsmall_b : ε * b < a / 16)
    (hsmall_h : ε * |h| < a / 8)
    (hd_C1 : ε ^ 2 * |d| < a / 8)
    (hb_C1 : ε ^ 2 * b ^ 2 < a ^ 2 / 8)
    (hd_C2 : ε * |d| < b / 8)
    (hh_C2 : ε * h ^ 2 < a * b / 8) :
    ∃ lam_neg lam_pos lam_0 : ℝ,
      L3Poly a b h d ε lam_neg = 0 ∧
      L3Poly a b h d ε lam_pos = 0 ∧
      L3Poly a b h d ε lam_0 = 0 ∧
      lam_neg < 0 ∧ 0 < lam_pos ∧ lam_pos < a / 2 ∧ a ≤ lam_0 ∧
      (1 / 2) * ε ^ 2 * (d - h ^ 2 / a) ≤ lam_pos + lam_neg ∧
      (let x := -ε * h / (a - lam_pos)
       let y := ε * b / lam_pos
       let z := (1 : ℝ)
       (L3 a b h d ε).mulVec ![x, y, z] = lam_pos • ![x, y, z] ∧
       (1 / 3 : ℝ) ≤ z ^ 2 / (x ^ 2 + y ^ 2 + z ^ 2)) := by
  set p : ℝ → ℝ := L3Poly a b h d ε
  have hp_cont : Continuous p := L3Poly_continuous a b h d ε
  rcases L3Poly_eval_signs ha hb hε hd_C1 hb_C1 hd_C2 hh_C2 hsmall_b with
    ⟨hp0, hp_lo_pos, hpa2, hp_neg, hpa, hpM⟩
  set M_big : ℝ := a + ε ^ 2 * |d| + ε ^ 2 * (h ^ 2 + b ^ 2) / a + 1
  have hM_gt : a < M_big := by
    simp only [M_big]; linarith [show (0 : ℝ) ≤ ε ^ 2 * |d| + ε ^ 2 * (h ^ 2 + b ^ 2) / a by positivity]
  have hroot_neg : ∃ lam_neg ∈ Set.Ioo (-2 * ε * b) 0, p lam_neg = 0 :=
    intermediate_value_Ioo (by nlinarith) hp_cont.continuousOn ⟨hp_neg, hp0⟩
  have hroot_pos : ∃ lam_pos ∈ Set.Ioo ((4 / 5) * ε * b) (a / 2), p lam_pos = 0 :=
    intermediate_value_Ioo' (by linarith only [hsmall_b, ha]) hp_cont.continuousOn ⟨hpa2, hp_lo_pos⟩
  have hroot_0 : ∃ lam_0 ∈ Set.Icc a M_big, p lam_0 = 0 :=
    intermediate_value_Icc hM_gt.le hp_cont.continuousOn ⟨hpa, hpM.le⟩
  rcases hroot_neg with ⟨lam_neg, ⟨_, hneg_hi⟩, hp_neg_eq⟩
  rcases hroot_pos with ⟨lam_pos, ⟨hpos_lo, hpos_hi⟩, hp_pos_eq⟩
  rcases hroot_0 with ⟨lam_0, ⟨h0_lo, _⟩, hp_0_eq⟩
  have hpos_gt0 : 0 < lam_pos := lt_trans (by positivity) hpos_lo
  have hne1 : lam_neg ≠ lam_pos := ne_of_lt (lt_trans hneg_hi hpos_gt0)
  have hne2 : lam_neg ≠ lam_0 := ne_of_lt (lt_trans hneg_hi (lt_of_lt_of_le ha h0_lo))
  have hne3 : lam_pos ≠ lam_0 := ne_of_lt (lt_of_lt_of_le (by linarith only [ha, hpos_hi]) h0_lo)
  have hvieta_sum : lam_neg + lam_pos + lam_0 = a + ε ^ 2 * d :=
    L3Poly_vieta_sum hne1 hne2 hne3 hp_neg_eq hp_pos_eq hp_0_eq
  have hlam0_upper : lam_0 - a ≤ ε ^ 2 * (h ^ 2 / a) + (1 / 2) * ε ^ 2 * (d - h ^ 2 / a) :=
    L3_lam0_upper_bound ha hε h0_lo hp_0_eq hC3_ge hsmall_C3 hd_C1 hb_C1
  have hgap_out : (1 / 2) * ε ^ 2 * (d - h ^ 2 / a) ≤ lam_pos + lam_neg := by
    have h1 : lam_pos + lam_neg = a + ε ^ 2 * d - lam_0 := by
      calc lam_pos + lam_neg = (lam_neg + lam_pos + lam_0) - lam_0 := by ring
        _ = a + ε ^ 2 * d - lam_0 := by rw [hvieta_sum]
    have h2 : (1 / 2) * ε ^ 2 * (d - h ^ 2 / a) ≤ ε ^ 2 * d - (lam_0 - a) := by
      calc (1 / 2) * ε ^ 2 * (d - h ^ 2 / a)
          = ε ^ 2 * d - (ε ^ 2 * (h ^ 2 / a) + (1 / 2) * ε ^ 2 * (d - h ^ 2 / a)) := by ring
        _ ≤ ε ^ 2 * d - (lam_0 - a) := sub_le_sub_left hlam0_upper _
    calc (1 / 2) * ε ^ 2 * (d - h ^ 2 / a) ≤ ε ^ 2 * d - (lam_0 - a) := h2
      _ = a + ε ^ 2 * d - lam_0 := by ring
      _ = lam_pos + lam_neg := h1.symm
  have heig := L3_eig_and_mass ha hb hε hp_pos_eq hpos_lo hpos_hi hpos_gt0 hsmall_h
  exact ⟨lam_neg, lam_pos, lam_0, hp_neg_eq, hp_pos_eq, hp_0_eq, hneg_hi, hpos_gt0, hpos_hi,
    h0_lo, hgap_out, heig.1, heig.2⟩

/-- **Lemma 2 (Strict 8th–9th singular gap and eigenvector isolate mass of `L₃`)**:
For ANY `a > 0, b > 0` and `d > h²/a` (so `κ := d - h²/a > 0`), there exists `ε₀ > 0`
such that for EVERY `ε ∈ (0, ε₀)`, `L₃(a, b, h, d, ε)` has three exact eigenvalues
`λ_- < 0 < λ_+ < a / 2 < a ≤ λ_0` satisfying:
1. Strict 8th–9th singular value gap: `λ_+ - |λ_-| = λ_+ + λ_- ≥ (1/2) ε² (d - h²/a) > 0`,
2. Explicit eigenvector `(x, y, z)` of `L₃` for `λ_+` with squared isolate coordinate
   `z² / (x² + y² + z²) ≥ 1/3` (approaching `1/2` as `ε → 0`). -/
theorem strict_eighth_ninth_gap_and_isolate_mass {a b h d : ℝ}
    (ha : 0 < a) (hb : 0 < b) (hgap : 0 < d - h ^ 2 / a) :
    ∃ ε₀ : ℝ, 0 < ε₀ ∧ ∀ ε : ℝ, 0 < ε → ε < ε₀ →
      ∃ lam_neg lam_pos lam_0 : ℝ,
        L3Poly a b h d ε lam_neg = 0 ∧
        L3Poly a b h d ε lam_pos = 0 ∧
        L3Poly a b h d ε lam_0 = 0 ∧
        lam_neg < 0 ∧ 0 < lam_pos ∧ lam_pos < a / 2 ∧ a ≤ lam_0 ∧
        (1 / 2) * ε ^ 2 * (d - h ^ 2 / a) ≤ lam_pos + lam_neg ∧
        (let x := -ε * h / (a - lam_pos)
         let y := ε * b / lam_pos
         let z := (1 : ℝ)
         (L3 a b h d ε).mulVec ![x, y, z] = lam_pos • ![x, y, z] ∧
         (1 / 3 : ℝ) ≤ z ^ 2 / (x ^ 2 + y ^ 2 + z ^ 2)) := by
  set κ := d - h ^ 2 / a
  let C₁ : ℝ := |d| + h ^ 2 / a + b ^ 2 / a + 1
  let C₂ : ℝ := |d| + h ^ 2 / a + b + 1
  let C₃ : ℝ := (4 / 3) * (h ^ 2 / a) * (|d| / a + b ^ 2 / a ^ 2) + 1
  let C₄ : ℝ := |h| + 1
  have hC1 : 0 < C₁ := by positivity
  have hC2 : 0 < C₂ := by positivity
  have hC3 : 0 < C₃ := by positivity
  have hC4 : 0 < C₄ := by positivity
  let ε₀ : ℝ := min 1 (min (a / (4 * b)) (min (Real.sqrt (a / (8 * C₁)))
    (min (b / (8 * C₂)) (min (Real.sqrt (κ / (2 * C₃)))
      (min (a / (8 * C₄)) (a / (16 * b)))))))
  have hε₀ : 0 < ε₀ := by
    simp only [ε₀]
    refine lt_min (by norm_num) (lt_min (by positivity) (lt_min (Real.sqrt_pos.mpr (by positivity))
      (lt_min (by positivity) (lt_min (Real.sqrt_pos.mpr (by positivity))
        (lt_min (by positivity) (by positivity))))))
  refine ⟨ε₀, hε₀, fun ε hε hε_lt => ?_⟩
  have hε_C1 : ε < Real.sqrt (a / (8 * C₁)) :=
    lt_of_lt_of_le hε_lt (le_trans (min_le_right _ _) (le_trans (min_le_right _ _) (min_le_left _ _)))
  have hε_C2 : ε < b / (8 * C₂) :=
    lt_of_lt_of_le hε_lt (le_trans (min_le_right _ _) (le_trans (min_le_right _ _)
      (le_trans (min_le_right _ _) (min_le_left _ _))))
  have hε_C3 : ε < Real.sqrt (κ / (2 * C₃)) :=
    lt_of_lt_of_le hε_lt (le_trans (min_le_right _ _) (le_trans (min_le_right _ _)
      (le_trans (min_le_right _ _) (le_trans (min_le_right _ _) (min_le_left _ _)))))
  have hε_C4 : ε < a / (8 * C₄) :=
    lt_of_lt_of_le hε_lt (le_trans (min_le_right _ _) (le_trans (min_le_right _ _)
      (le_trans (min_le_right _ _) (le_trans (min_le_right _ _)
        (le_trans (min_le_right _ _) (min_le_left _ _))))))
  have hε_b16 : ε < a / (16 * b) :=
    lt_of_lt_of_le hε_lt (le_trans (min_le_right _ _) (le_trans (min_le_right _ _)
      (le_trans (min_le_right _ _) (le_trans (min_le_right _ _)
        (le_trans (min_le_right _ _) (min_le_right _ _))))))
  have hsmall_C1 : ε ^ 2 * C₁ < a / 8 := by
    have hsq : ε ^ 2 < a / (8 * C₁) := by
      have := pow_lt_pow_left₀ hε_C1 hε.le (by norm_num : 2 ≠ 0)
      rwa [Real.sq_sqrt (by positivity)] at this
    have := mul_lt_mul_of_pos_right hsq hC1
    have hr : a / (8 * C₁) * C₁ = a / 8 := by field_simp [hC1.ne']
    linarith
  have hsmall_C2 : ε * C₂ < b / 8 := by
    have := mul_lt_mul_of_pos_right hε_C2 hC2
    have hr : b / (8 * C₂) * C₂ = b / 8 := by field_simp [hC2.ne']
    linarith
  have hsmall_C3 : ε ^ 2 * C₃ < κ / 2 := by
    have hsq : ε ^ 2 < κ / (2 * C₃) := by
      have := pow_lt_pow_left₀ hε_C3 hε.le (by norm_num : 2 ≠ 0)
      rwa [Real.sq_sqrt (by positivity)] at this
    have := mul_lt_mul_of_pos_right hsq hC3
    have hr : κ / (2 * C₃) * C₃ = κ / 2 := by field_simp [hC3.ne']
    linarith
  have hsmall_b : ε * b < a / 16 := by
    have := mul_lt_mul_of_pos_right hε_b16 hb
    have hr : a / (16 * b) * b = a / 16 := by field_simp [hb.ne']
    linarith
  have hsmall_h : ε * |h| < a / 8 := by
    have := mul_lt_mul_of_pos_right hε_C4 hC4
    have hr : a / (8 * C₄) * C₄ = a / 8 := by field_simp [hC4.ne']
    have hle : ε * |h| ≤ ε * C₄ := mul_le_mul_of_nonneg_left (by simp only [C₄]; linarith) hε.le
    linarith
  have hd_C1 : ε ^ 2 * |d| < a / 8 := by
    have hle : |d| ≤ C₁ := by
      have : 0 ≤ h ^ 2 / a + b ^ 2 / a + 1 := by positivity
      simp only [C₁]; linarith
    exact lt_of_le_of_lt (mul_le_mul_of_nonneg_left hle (sq_nonneg ε)) hsmall_C1
  have hb_C1 : ε ^ 2 * b ^ 2 < a ^ 2 / 8 := by
    have h1 : ε ^ 2 * (b ^ 2 / a) < a / 8 := by
      have hle : b ^ 2 / a ≤ C₁ := by
        have : 0 ≤ |d| + h ^ 2 / a + 1 := by positivity
        simp only [C₁]; linarith
      exact lt_of_le_of_lt (mul_le_mul_of_nonneg_left hle (sq_nonneg ε)) hsmall_C1
    have := mul_lt_mul_of_pos_right h1 ha
    have heq : ε ^ 2 * (b ^ 2 / a) * a = ε ^ 2 * b ^ 2 := by field_simp [ha.ne']
    linarith
  have hd_C2 : ε * |d| < b / 8 := by
    have hle : |d| ≤ C₂ := by
      have : 0 ≤ h ^ 2 / a + b + 1 := by positivity
      simp only [C₂]; linarith
    exact lt_of_le_of_lt (mul_le_mul_of_nonneg_left hle hε.le) hsmall_C2
  have hh_C2 : ε * h ^ 2 < a * b / 8 := by
    have h1 : ε * (h ^ 2 / a) < b / 8 := by
      have hle : h ^ 2 / a ≤ C₂ := by
        have : 0 ≤ |d| + b + 1 := by positivity
        simp only [C₂]; linarith
      exact lt_of_le_of_lt (mul_le_mul_of_nonneg_left hle hε.le) hsmall_C2
    have := mul_lt_mul_of_pos_right h1 ha
    have heq : ε * (h ^ 2 / a) * a = ε * h ^ 2 := by field_simp [ha.ne']
    linarith
  exact L3_root_gap_and_eig ha hb hε (by simp only [C₃]; linarith) hsmall_C3 hsmall_b hsmall_h
    hd_C1 hb_C1 hd_C2 hh_C2

/-! ### 3. Eigenvector Perturbation & `Ω(n^(1/4))` Output-Coherence Blow-Up (Theorem 1 Core) -/

/-- **Projector diagonal lower bound**:
For any symmetric idempotent projector `P` (`Pᵀ = P`, `P² = P`) and any unit vector `v`
in its range (`P v = v`), every diagonal entry satisfies `v_j² ≤ P_{jj}`. -/
theorem projector_diag_ge_sq_entry {n : ℕ}
    (P : Matrix (Fin n) (Fin n) ℝ) (hPsym : Pᵀ = P) (hPidem : P * P = P)
    (v : Fin n → ℝ) (hv_unit : ∑ i : Fin n, v i ^ 2 = 1) (hv_range : P.mulVec v = v)
    (j : Fin n) :
    v j ^ 2 ≤ P j j := by
  -- v j = (P v)_j = ∑_k P_{jk} v_k; by Cauchy–Schwarz,
  -- (v j)² ≤ (∑_k P_{jk}²) * (∑_k v_k²) = (P * Pᵀ)_{jj} * 1 = P_{jj}
  have hvj : v j = ∑ k : Fin n, P j k * v k := by
    have := congrFun hv_range j
    simpa [Matrix.mulVec, dotProduct] using this.symm
  have hcs := Finset.sum_mul_sq_le_sq_mul_sq (Finset.univ : Finset (Fin n)) (fun k => P j k) v
  have hPjj : ∑ k : Fin n, P j k ^ 2 = P j j := by
    have hpp := congrFun (congrFun hPidem j) j
    simp only [Matrix.mul_apply] at hpp
    rw [← hpp]
    refine Finset.sum_congr rfl fun k _ => ?_
    have hkj : P k j = P j k := by
      have := congrFun (congrFun hPsym j) k
      simpa [Matrix.transpose_apply] using this
    rw [hkj, sq]
  rw [hvj]
  rw [hPjj, hv_unit, mul_one] at hcs
  exact hcs

/-- **Pigeonhole bound on isolate mass**:
If `∑_{j ∈ S} v_j² ≥ m₀` on a nonempty isolate set `S` of size `s = S.card`,
then some `j ∈ S` satisfies `v_j² ≥ m₀ / s`. -/
theorem pigeonhole_isolate_max_sq {n : ℕ} (S : Finset (Fin n)) (hS : S.Nonempty)
    (v : Fin n → ℝ) (m₀ : ℝ) (hmass : m₀ ≤ ∑ j ∈ S, v j ^ 2) :
    ∃ j ∈ S, m₀ / (S.card : ℝ) ≤ v j ^ 2 := by
  by_contra h
  push_neg at h
  have hsum : ∑ j ∈ S, v j ^ 2 < ∑ _j ∈ S, m₀ / (S.card : ℝ) :=
    Finset.sum_lt_sum_of_nonempty hS fun j hj => h j hj
  rw [Finset.sum_const, nsmul_eq_mul] at hsum
  have hcard : (S.card : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hS.card_pos.ne'
  rw [mul_div_cancel₀ _ hcard] at hsum
  linarith

/-- **Perturbation isolate-mass lower bound (reverse triangle inequality on `S`)**:
If the population eigenvector `v₊` has isolate mass `∑_{j ∈ S} v_{+,j}² ≥ m` (`m ≥ 0`)
and the perturbed eigenvector `v̂` satisfies `∑_{j ∈ S} (v̂_j - v_{+,j})² ≤ δ²` with `0 ≤ δ ≤ √m`,
then `∑_{j ∈ S} v̂_j² ≥ (√m - δ)²`. -/
theorem perturbation_isolate_mass_lower {n : ℕ} (S : Finset (Fin n))
    (v_pop v_hat : Fin n → ℝ) {m δ : ℝ} (hδ0 : 0 ≤ δ) (hδ : δ ≤ Real.sqrt m)
    (hpop : m ≤ ∑ j ∈ S, v_pop j ^ 2)
    (hpert : ∑ j ∈ S, (v_hat j - v_pop j) ^ 2 ≤ δ ^ 2) :
    (Real.sqrt m - δ) ^ 2 ≤ ∑ j ∈ S, v_hat j ^ 2 := by
  set A := Real.sqrt (∑ j ∈ S, v_hat j ^ 2)
  set B := Real.sqrt (∑ j ∈ S, v_pop j ^ 2)
  set E := Real.sqrt (∑ j ∈ S, (v_hat j - v_pop j) ^ 2)
  have hA2 : A ^ 2 = ∑ j ∈ S, v_hat j ^ 2 :=
    Real.sq_sqrt (Finset.sum_nonneg fun _ _ => sq_nonneg _)
  have hB2 : B ^ 2 = ∑ j ∈ S, v_pop j ^ 2 :=
    Real.sq_sqrt (Finset.sum_nonneg fun _ _ => sq_nonneg _)
  have hE2 : E ^ 2 = ∑ j ∈ S, (v_hat j - v_pop j) ^ 2 :=
    Real.sq_sqrt (Finset.sum_nonneg fun _ _ => sq_nonneg _)
  have hB_ge : Real.sqrt m ≤ B := Real.sqrt_le_sqrt hpop
  have hE_le : E ≤ δ := by
    rw [← Real.sqrt_sq hδ0]
    exact Real.sqrt_le_sqrt hpert
  -- Triangle inequality: B ≤ A + E, proved via Cauchy–Schwarz on ∑ v_pop_j = ∑ v_hat_j - ∑ (v_hat_j - v_pop_j)
  have htri : B ≤ A + E := by
    have hcs := Finset.sum_mul_sq_le_sq_mul_sq S v_hat (fun j => v_pop j - v_hat j)
    have hdot : (∑ j ∈ S, v_hat j * (v_pop j - v_hat j)) ≤ A * E := by
      have hsq : (∑ j ∈ S, v_hat j * (v_pop j - v_hat j)) ^ 2 ≤ (A * E) ^ 2 := by
        rw [mul_pow, hA2, hE2]
        have heq : ∑ j ∈ S, (v_pop j - v_hat j) ^ 2 = ∑ j ∈ S, (v_hat j - v_pop j) ^ 2 :=
          Finset.sum_congr rfl fun j _ => by ring
        rw [← heq]; exact hcs
      have hAE : 0 ≤ A * E := by positivity
      nlinarith
    have hexpand : B ^ 2 = A ^ 2 + E ^ 2 + 2 * ∑ j ∈ S, v_hat j * (v_pop j - v_hat j) := by
      rw [hB2, hA2, hE2, Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun j _ => by ring
    have hsq_le : B ^ 2 ≤ (A + E) ^ 2 := by nlinarith
    nlinarith [Real.sqrt_nonneg (∑ j ∈ S, v_hat j ^ 2),
      Real.sqrt_nonneg (∑ j ∈ S, (v_hat j - v_pop j) ^ 2)]
  have hsub : Real.sqrt m - δ ≤ A := by linarith
  have hsub_nn : 0 ≤ Real.sqrt m - δ := by linarith
  rw [← hA2]
  exact pow_le_pow_left₀ hsub_nn hsub 2

/-- **Theorem 1 (Deterministic `Ω(n^(1/4))` output coherence blow-up vs `O(1)` input coherence)**:
For `t ≥ 1`, `s = ⌊t^(3/4)⌋`, `n = 8t + s`:
1. `1 ≤ s`, `(s : ℝ) ≤ t^(3/4)`, and `n ≤ 9 t`.
2. Input coherence is uniformly bounded: `n / (8 t) ≤ 9 / 8`.
3. For any symmetric idempotent rank-8 projector `P` on `Fin n` whose range contains a unit
   vector `v̂` with isolate mass `∑_{j ∈ S} v̂_j² ≥ m₀ > 0` on a set `S` of size `s`,
   some vertex `j ∈ S` satisfies
   `(n / 8) * P_{jj} ≥ m₀ * t^(1/4) ≥ m₀ * (n / 9)^(1/4)`,
   establishing simultaneous `Ω(n^(1/4))` blow-up for both `A_{K̂}` and `Ā_{K̂}`! -/
theorem output_coherence_omega_n_quarter (t : ℕ) (ht : 1 ≤ t)
    (s : ℕ) (hs_pos : 1 ≤ s) (hs_le : (s : ℝ) ≤ (t : ℝ) ^ ((3 : ℝ) / 4))
    (S : Finset (Fin (8 * t + s))) (hS_card : S.card = s)
    (P : Matrix (Fin (8 * t + s)) (Fin (8 * t + s)) ℝ)
    (hPsym : Pᵀ = P) (hPidem : P * P = P)
    (v_hat : Fin (8 * t + s) → ℝ) (hv_unit : ∑ i, v_hat i ^ 2 = 1) (hv_range : P.mulVec v_hat = v_hat)
    {m₀ : ℝ} (hm₀ : 0 < m₀) (hmass : m₀ ≤ ∑ j ∈ S, v_hat j ^ 2) :
    let n : ℕ := 8 * t + s
    ((n : ℝ) / (8 * (t : ℝ)) ≤ 9 / 8) ∧
    ∃ j ∈ S,
      m₀ * ((n : ℝ) / 9) ^ ((1 : ℝ) / 4) ≤ ((n : ℝ) / 8) * P j j := by
  dsimp only
  have ht_pos : (0 : ℝ) < (t : ℝ) := Nat.cast_pos.mpr (by omega)
  have hs_posR : (0 : ℝ) < (s : ℝ) := Nat.cast_pos.mpr (by omega)
  have ht1 : (1 : ℝ) ≤ (t : ℝ) := by exact_mod_cast ht
  have ht34_le_t : (t : ℝ) ^ ((3 : ℝ) / 4) ≤ (t : ℝ) := by
    calc (t : ℝ) ^ ((3 : ℝ) / 4) ≤ (t : ℝ) ^ (1 : ℝ) :=
          Real.rpow_le_rpow_of_exponent_le ht1 (by norm_num)
      _ = (t : ℝ) := Real.rpow_one _
  have hs_le_t : (s : ℝ) ≤ (t : ℝ) := le_trans hs_le ht34_le_t
  have hn_le_9t : ((8 * t + s : ℕ) : ℝ) ≤ 9 * (t : ℝ) := by
    push_cast; linarith
  have hcoh_in : ((8 * t + s : ℕ) : ℝ) / (8 * (t : ℝ)) ≤ 9 / 8 := by
    rw [div_le_div_iff₀ (by positivity) (by norm_num : (0 : ℝ) < 8)]
    linarith
  have hS_ne : S.Nonempty := Finset.card_pos.mp (by omega)
  rcases pigeonhole_isolate_max_sq S hS_ne v_hat m₀ hmass with ⟨j, hjS, hvj⟩
  have hPjj : v_hat j ^ 2 ≤ P j j :=
    projector_diag_ge_sq_entry P hPsym hPidem v_hat hv_unit hv_range j
  have ht_le_n8 : (t : ℝ) ≤ ((8 * t + s : ℕ) : ℝ) / 8 := by
    push_cast; linarith
  have hts_ge : (t : ℝ) ^ ((1 : ℝ) / 4) ≤ (t : ℝ) / (s : ℝ) := by
    rw [le_div_iff₀ hs_posR]
    calc (t : ℝ) ^ ((1 : ℝ) / 4) * (s : ℝ)
        ≤ (t : ℝ) ^ ((1 : ℝ) / 4) * (t : ℝ) ^ ((3 : ℝ) / 4) :=
          mul_le_mul_of_nonneg_left hs_le (Real.rpow_nonneg ht_pos.le _)
      _ = (t : ℝ) ^ ((1 : ℝ) / 4 + (3 : ℝ) / 4) := (Real.rpow_add ht_pos _ _).symm
      _ = (t : ℝ) := by norm_num
  have hn9_le_t : (((8 * t + s : ℕ) : ℝ) / 9) ^ ((1 : ℝ) / 4) ≤ (t : ℝ) ^ ((1 : ℝ) / 4) := by
    apply Real.rpow_le_rpow (by positivity) _ (by norm_num)
    linarith
  refine ⟨hcoh_in, j, hjS, ?_⟩
  rw [hS_card] at hvj
  calc m₀ * (((8 * t + s : ℕ) : ℝ) / 9) ^ ((1 : ℝ) / 4)
      ≤ m₀ * (t : ℝ) ^ ((1 : ℝ) / 4) := mul_le_mul_of_nonneg_left hn9_le_t hm₀.le
    _ ≤ m₀ * ((t : ℝ) / (s : ℝ)) := mul_le_mul_of_nonneg_left hts_ge hm₀.le
    _ = (t : ℝ) * (m₀ / (s : ℝ)) := by ring
    _ ≤ (((8 * t + s : ℕ) : ℝ) / 8) * (m₀ / (s : ℝ)) :=
        mul_le_mul_of_nonneg_right ht_le_n8 (div_nonneg hm₀.le hs_posR.le)
    _ ≤ (((8 * t + s : ℕ) : ℝ) / 8) * v_hat j ^ 2 := mul_le_mul_of_nonneg_left hvj (by positivity)
    _ ≤ (((8 * t + s : ℕ) : ℝ) / 8) * P j j := mul_le_mul_of_nonneg_left hPjj (by positivity)

end DPPCACoherence

end
