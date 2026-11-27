import Mathlib

/-!
# Continuous Piecewise Linear Triangulations of the Product Surface
(Robert Burlacu, Lukas Hager, and Robert Hildebrand, arXiv:2604.04026)

We formalize in Lean 4 + Mathlib (complete kernel-checked proofs, standard foundations only) the exact algebraic,
geometric, and certificate theorems from `solutions/pwl_product_triang/solution.md`:

1. **Exact face-error formula, edge characterization, boundary sufficiency, and area identity**:
   - `exact_face_error_formula`: On any triangle `(v₀, v₁, v₂)` with vertex deviations `(d₀, d₁, d₂)`
     and barycentric coordinates `l₀ + l₁ + l₂ = 1`,
     `f(p) - F(p) = l₀ d₀ + l₁ d₁ + l₂ d₂ + l₀ l₁ q(v₁ - v₀) + l₁ l₂ q(v₂ - v₁) + l₂ l₀ q(v₀ - v₂)`.
   - `edgeError_abs_le_iff`: Exact edge characterization `|E(t)| ≤ ε` on `[0, 1]` iff `|d₀|, |d₁| ≤ ε`
     and `-(√(ε + d₀) + √(ε + d₁))² ≤ e ≤ (√(ε - d₀) + √(ε - d₁))²`.
   - `faceError_abs_le_of_edges`: Purely algebraic horizontal-line proof that edgewise `|E| ≤ ε`
     implies `|f - F| ≤ ε` on the entire triangle.
   - `four_area_sq_eq`: `4 A² = (q(u) + q(v) - q(u - v))² - 4 q(u) q(v)`.

2. **Sharp conjectured bound `δ(T) ≥ √3 / (8ε)` for uniform signed-edge-product triangulations
   (Theorem 2)**:
   - `one_sign_triangle_area_le`: One-sign triangles satisfy `A ≤ 4ε < 8ε / √3`.
   - `alternating_walk_sum_le`: Along any infinite alternating `(b, -w)`-edge walk with bounded
     vertex deviations `|d_v| ≤ ε`, the midpoint test forces `b + w ≤ 8ε`.
   - `four_spoke_equal_product_impossible`, `null_edge_flanking_opposite_signs`: The two vertex
     incidence obstructions proving every vertex meets both a `b`-edge and a `-w`-edge.
   - `mixed_sign_area_le_eight_div_sqrt3`: Under `0 ≤ a ≤ b`, `0 < w`, `b + w ≤ 8ε`, every face
     has area `A ≤ 8ε / √3`.
   - `uniform_edge_product_density_lower_bound`: Density bound `N / ℓ² ≥ √3 / (8ε)`.
   - `sharp_parallelogram_tiling_face`: Explicit constant-deviation (`d = ε / 3`) parallelogram
     tiling face achieving exact area `8ε / √3` and edge products `{8ε/3, 8ε/3, -16ε/3}`.

3. **Explicit continuous periodic triangulation with an oversized face of area `5ε > 8ε / √3`
   (Proposition 1)**:
   - `periodic_mesh_total_area`: The 10-triangle periodic mesh on `[0, 21/10] × [0, 51/10]` has
     exact total area `1071 / 100` (density `1000 / 1071`).
   - `periodic_mesh_oversized_face`: Triangle `ABC` has exact area `5`, and `8ε / √3 < 5ε`.
   - `periodic_mesh_global_error_le`: Every edge (and hence every face) satisfies `|f - F| ≤ 169/180 < 1`,
     attaining `169 / 180` on edge `AC` at `t = 17/30`.
   - `periodic_mesh_boundary_continuity`: Opposite boundary vertex deviations match identically.

4. **Universal density improvement over the discontinuous optimum (Lemma 3 & Theorem 3)**:
   - `unrestricted_area_le_M`: Every valid triangle at `ε = 1` satisfies `A ≤ M = 32√3 / 9`.
   - `large_triangle_stability`: Every valid mixed-sign triangle with `A > B = 123/20` satisfies
     `d₀ < -96/100`, `1/2 < d₁, d₂ < 92/100`, and `w > max p r`.
   - `large_vertex_four_spoke_obstruction`: No vertex can be surrounded exclusively by large faces
     (resolving Remark `lem:no-optimal-tiling`).
   - `overhang_factor_lt_alpha`, `limsup_density_recurrence`, `universal_density_constant_strict_improvement`:
     Exact verification that `η = 1 - B/M < 1/484`, `(1 - 2√η)⁻¹ < 11/10`, and
     `0.162379 < 3√3 / 32 < (1/M)(1 + η / (1 + 6(11/10)²))` with `0.162406 < c_cont < 0.162407`.

Global topological/measure-theoretic arguments for locally finite planar triangulations
(complete fan decomposition, overhang homothety on `Q_ℓ`, and double-counting marks on `Q_{αℓ}`)
are presented in prose in `solutions/pwl_product_triang/solution.md` and `proof.pdf`.
-/

namespace PWLProductTriang

noncomputable section

/-! ### 1. Exact Face-Error Formula, Edge Characterization, and Area Identity -/

/-- The product surface `F(x, y) = x * y`. -/
def F (p : ℝ × ℝ) : ℝ := p.1 * p.2

/-- The signed edge product `q(u) = u_x * u_y`. -/
def q (u : ℝ × ℝ) : ℝ := u.1 * u.2

/-- Barycentric combination of three planar vertices. -/
def baryPoint (v₀ v₁ v₂ : ℝ × ℝ) (l₀ l₁ l₂ : ℝ) : ℝ × ℝ :=
  (l₀ * v₀.1 + l₁ * v₁.1 + l₂ * v₂.1, l₀ * v₀.2 + l₁ * v₁.2 + l₂ * v₂.2)

/-- Affine interpolant of `F + d` on triangle `(v₀, v₁, v₂)` in barycentric coordinates. -/
def affineInterp (v₀ v₁ v₂ : ℝ × ℝ) (d₀ d₁ d₂ l₀ l₁ l₂ : ℝ) : ℝ :=
  l₀ * (F v₀ + d₀) + l₁ * (F v₁ + d₁) + l₂ * (F v₂ + d₂)

/-- Signed face error `f(p) - F(p)` in terms of vertex deviations and signed edge products. -/
def faceError (d₀ d₁ d₂ e₀₁ e₁₂ e₂₀ l₀ l₁ l₂ : ℝ) : ℝ :=
  l₀ * d₀ + l₁ * d₁ + l₂ * d₂ + l₀ * l₁ * e₀₁ + l₁ * l₂ * e₁₂ + l₂ * l₀ * e₂₀

/-- Signed error along an edge with vertex deviations `d₀, d₁` and signed edge product `e`. -/
def edgeError (d₀ d₁ e t : ℝ) : ℝ :=
  (1 - t) * d₀ + t * d₁ + t * (1 - t) * e

/-- Area of triangle `(v₀, v₁, v₂)` in `ℝ²`. -/
def triArea (v₀ v₁ v₂ : ℝ × ℝ) : ℝ :=
  |(v₁.1 - v₀.1) * (v₂.2 - v₀.2) - (v₁.2 - v₀.2) * (v₂.1 - v₀.1)| / 2

/-- **Exact face-error formula**: For any triangle `(v₀, v₁, v₂)`, deviations `(d₀, d₁, d₂)`,
and barycentric coordinates `l₀ + l₁ + l₂ = 1`,
`f(p) - F(p) = l₀ d₀ + l₁ d₁ + l₂ d₂ + l₀ l₁ q(v₁ - v₀) + l₁ l₂ q(v₂ - v₁) + l₂ l₀ q(v₀ - v₂)`. -/
theorem exact_face_error_formula (v₀ v₁ v₂ : ℝ × ℝ) (d₀ d₁ d₂ l₀ l₁ l₂ : ℝ)
    (hl : l₀ + l₁ + l₂ = 1) :
    affineInterp v₀ v₁ v₂ d₀ d₁ d₂ l₀ l₁ l₂ - F (baryPoint v₀ v₁ v₂ l₀ l₁ l₂) =
      faceError d₀ d₁ d₂ (q (v₁ - v₀)) (q (v₂ - v₁)) (q (v₀ - v₂)) l₀ l₁ l₂ := by
  have hl0 : l₀ = 1 - l₁ - l₂ := by linarith
  simp only [affineInterp, F, baryPoint, faceError, q, Prod.fst_sub, Prod.snd_sub]
  rw [hl0]
  ring

/-- Helper: one-sided upper edge characterization. -/
private lemma edgeError_le_iff {ε d₀ d₁ e : ℝ} :
    (∀ t ∈ Set.Icc (0 : ℝ) 1, edgeError d₀ d₁ e t ≤ ε) ↔
      d₀ ≤ ε ∧ d₁ ≤ ε ∧ e ≤ (Real.sqrt (ε - d₀) + Real.sqrt (ε - d₁)) ^ 2 := by
  constructor
  · intro h
    have hd0 : d₀ ≤ ε := by
      have := h 0 ⟨le_refl 0, by norm_num⟩; simpa [edgeError] using this
    have hd1 : d₁ ≤ ε := by
      have := h 1 ⟨by norm_num, le_refl 1⟩; simpa [edgeError] using this
    refine ⟨hd0, hd1, ?_⟩
    set s₀ := Real.sqrt (ε - d₀)
    set s₁ := Real.sqrt (ε - d₁)
    have hs0 : 0 ≤ s₀ := Real.sqrt_nonneg _
    have hs1 : 0 ≤ s₁ := Real.sqrt_nonneg _
    have hs0sq : s₀ ^ 2 = ε - d₀ := Real.sq_sqrt (by linarith)
    have hs1sq : s₁ ^ 2 = ε - d₁ := Real.sq_sqrt (by linarith)
    by_cases hs0_0 : s₀ = 0
    · have hd0_eq : d₀ = ε := by nlinarith
      by_cases hs1_0 : s₁ = 0
      · have hd1_eq : d₁ = ε := by nlinarith
        have hmid := h (1 / 2) ⟨by norm_num, by norm_num⟩
        simp only [edgeError, hd0_eq, hd1_eq] at hmid
        rw [hs0_0, hs1_0]; norm_num; linarith
      · have hfor : ∀ t : ℝ, 0 < t → t < 1 → (1 - t) * e ≤ ε - d₁ := by
          intro t ht0 ht1
          have ht := h t ⟨le_of_lt ht0, le_of_lt ht1⟩
          simp only [edgeError, hd0_eq] at ht
          have hfac : t * ((1 - t) * e - (ε - d₁)) ≤ 0 := by nlinarith
          nlinarith
        have he_le : e ≤ ε - d₁ := by
          by_contra hgt; push_neg at hgt
          have he_pos : 0 < e := by linarith
          set δ := (e - (ε - d₁)) / e
          have hδ_pos : 0 < δ := div_pos (by linarith) he_pos
          set t := min (1 / 2) (δ / 2)
          have h1 := hfor t (lt_min (by norm_num) (by linarith))
            (lt_of_le_of_lt (min_le_left _ _) (by norm_num))
          have h2 : t * e < e - (ε - d₁) :=
            (lt_div_iff₀ he_pos).mp (lt_of_le_of_lt (min_le_right _ _) (by linarith))
          linarith
        rw [hs0_0, zero_add, hs1sq]; exact he_le
    · by_cases hs1_0 : s₁ = 0
      · have hd1_eq : d₁ = ε := by nlinarith
        have hfor : ∀ t : ℝ, 0 < t → t < 1 → t * e ≤ ε - d₀ := by
          intro t ht0 ht1
          have ht := h t ⟨le_of_lt ht0, le_of_lt ht1⟩
          simp only [edgeError, hd1_eq] at ht
          have hfac : (1 - t) * (t * e - (ε - d₀)) ≤ 0 := by nlinarith
          nlinarith
        have he_le : e ≤ ε - d₀ := by
          by_contra hgt; push_neg at hgt
          have he_pos : 0 < e := by linarith
          set δ := (e - (ε - d₀)) / e
          have hδ_pos : 0 < δ := div_pos (by linarith) he_pos
          set u := min (1 / 2) (δ / 2)
          have hu0 : 0 < u := lt_min (by norm_num) (by linarith)
          have hu1 : u < 1 := lt_of_le_of_lt (min_le_left _ _) (by norm_num)
          have h1 := hfor (1 - u) (by linarith) (by linarith)
          have h2 : u * e < e - (ε - d₀) :=
            (lt_div_iff₀ he_pos).mp (lt_of_le_of_lt (min_le_right _ _) (by linarith))
          linarith
        rw [hs1_0, add_zero, hs0sq]; exact he_le
      · have hs0p : 0 < s₀ := lt_of_le_of_ne hs0 (Ne.symm hs0_0)
        have hs1p : 0 < s₁ := lt_of_le_of_ne hs1 (Ne.symm hs1_0)
        have hpos : 0 < s₀ + s₁ := by linarith
        let t := s₀ / (s₀ + s₁)
        have ht := h t ⟨div_nonneg hs0 (le_of_lt hpos), (div_le_one hpos).mpr (by linarith)⟩
        have h1t : 1 - t = s₁ / (s₀ + s₁) := by dsimp [t]; field_simp [ne_of_gt hpos]; ring
        simp only [edgeError] at ht
        rw [h1t] at ht
        dsimp [t] at ht
        have hmul := mul_le_mul_of_nonneg_left ht (sq_nonneg (s₀ + s₁))
        have hsimp : (s₀ + s₁) ^ 2 *
            (s₁ / (s₀ + s₁) * d₀ + s₀ / (s₀ + s₁) * d₁ +
              s₀ / (s₀ + s₁) * (s₁ / (s₀ + s₁)) * e) =
            s₁ * (s₀ + s₁) * d₀ + s₀ * (s₀ + s₁) * d₁ + s₀ * s₁ * e := by
          field_simp [ne_of_gt hpos]
        rw [hsimp] at hmul
        have hineq : s₀ * s₁ * e ≤ s₀ * s₁ * (s₀ + s₁) ^ 2 := by nlinarith [hs0sq, hs1sq]
        exact le_of_mul_le_mul_left hineq (mul_pos hs0p hs1p)
  · rintro ⟨hd0, hd1, hhi⟩ t ⟨ht0, ht1⟩
    set b₀ := Real.sqrt (ε - d₀)
    set b₁ := Real.sqrt (ε - d₁)
    have hb0 : b₀ ^ 2 = ε - d₀ := Real.sq_sqrt (by linarith)
    have hb1 : b₁ ^ 2 = ε - d₁ := Real.sq_sqrt (by linarith)
    have hprod_nn : 0 ≤ t * (1 - t) := mul_nonneg ht0 (by linarith)
    have hsq : 0 ≤ ((1 - t) * b₀ - t * b₁) ^ 2 := sq_nonneg _
    have hmul : t * (1 - t) * e ≤ t * (1 - t) * (b₀ + b₁) ^ 2 := mul_le_mul_of_nonneg_left hhi hprod_nn
    simp only [edgeError]
    nlinarith [hb0, hb1, hsq, hmul]

/-- **Exact edge-error characterization**: For `ε > 0`, `|edgeError d₀ d₁ e t| ≤ ε` for all
`t ∈ [0, 1]` iff `|d₀| ≤ ε`, `|d₁| ≤ ε`, and
`-(√(ε + d₀) + √(ε + d₁))² ≤ e ≤ (√(ε - d₀) + √(ε - d₁))²`. -/
theorem edgeError_abs_le_iff {ε d₀ d₁ e : ℝ} (_hε : 0 < ε) :
    (∀ t ∈ Set.Icc (0 : ℝ) 1, |edgeError d₀ d₁ e t| ≤ ε) ↔
      |d₀| ≤ ε ∧ |d₁| ≤ ε ∧
      -(Real.sqrt (ε + d₀) + Real.sqrt (ε + d₁)) ^ 2 ≤ e ∧
      e ≤ (Real.sqrt (ε - d₀) + Real.sqrt (ε - d₁)) ^ 2 := by
  have hsplit : (∀ t ∈ Set.Icc (0 : ℝ) 1, |edgeError d₀ d₁ e t| ≤ ε) ↔
      (∀ t ∈ Set.Icc (0 : ℝ) 1, edgeError (-d₀) (-d₁) (-e) t ≤ ε) ∧
      (∀ t ∈ Set.Icc (0 : ℝ) 1, edgeError d₀ d₁ e t ≤ ε) := by
    constructor
    · intro h
      refine ⟨fun t ht => ?_, fun t ht => (abs_le.mp (h t ht)).2⟩
      have h1 := (abs_le.mp (h t ht)).1
      simp only [edgeError] at h1 ⊢; linarith
    · rintro ⟨hlo, hhi⟩ t ht
      rw [abs_le]
      have h1 := hlo t ht
      simp only [edgeError] at h1 ⊢
      exact ⟨by linarith, hhi t ht⟩
  rw [hsplit, edgeError_le_iff, edgeError_le_iff]
  simp only [sub_neg_eq_add, abs_le]
  constructor
  · rintro ⟨⟨h1, h2, h3⟩, ⟨h4, h5, h6⟩⟩
    exact ⟨⟨by linarith, h4⟩, ⟨by linarith, h5⟩, by linarith, h6⟩
  · rintro ⟨⟨h1, h2⟩, ⟨h3, h4⟩, h5, h6⟩
    exact ⟨⟨by linarith, by linarith, by linarith⟩, ⟨h2, h4, h6⟩⟩

/-- **Boundary sufficiency (purely algebraic horizontal-line proof)**: If the affine approximation
of `F(x, y) = xy` has error at most `ε` on all three edges of triangle `(v₀, v₁, v₂)`, then its
error is at most `ε` at every point in the triangle. -/
theorem faceError_abs_le_of_edges {v₀ v₁ v₂ : ℝ × ℝ} {d₀ d₁ d₂ ε : ℝ}
    (h01 : ∀ t ∈ Set.Icc (0 : ℝ) 1, |edgeError d₀ d₁ (q (v₁ - v₀)) t| ≤ ε)
    (h12 : ∀ t ∈ Set.Icc (0 : ℝ) 1, |edgeError d₁ d₂ (q (v₂ - v₁)) t| ≤ ε)
    (h20 : ∀ t ∈ Set.Icc (0 : ℝ) 1, |edgeError d₂ d₀ (q (v₀ - v₂)) t| ≤ ε)
    {l₀ l₁ l₂ : ℝ} (hl0 : 0 ≤ l₀) (hl1 : 0 ≤ l₁) (hl2 : 0 ≤ l₂) (hsum : l₀ + l₁ + l₂ = 1) :
    |affineInterp v₀ v₁ v₂ d₀ d₁ d₂ l₀ l₁ l₂ - F (baryPoint v₀ v₁ v₂ l₀ l₁ l₂)| ≤ ε := by
  -- Helper: if any barycentric coordinate is zero, we lie on an edge.
  have hbdry : ∀ m₀ m₁ m₂ : ℝ, 0 ≤ m₀ → 0 ≤ m₁ → 0 ≤ m₂ → m₀ + m₁ + m₂ = 1 →
      (m₀ = 0 ∨ m₁ = 0 ∨ m₂ = 0) →
      |affineInterp v₀ v₁ v₂ d₀ d₁ d₂ m₀ m₁ m₂ - F (baryPoint v₀ v₁ v₂ m₀ m₁ m₂)| ≤ ε := by
    intro m₀ m₁ m₂ hm0 hm1 hm2 hms hzero
    rw [exact_face_error_formula v₀ v₁ v₂ d₀ d₁ d₂ m₀ m₁ m₂ hms]
    rcases hzero with rfl | rfl | rfl
    · have hm1' : m₁ = 1 - m₂ := by linarith
      have hEq : faceError d₀ d₁ d₂ (q (v₁ - v₀)) (q (v₂ - v₁)) (q (v₀ - v₂)) 0 m₁ m₂ =
          edgeError d₁ d₂ (q (v₂ - v₁)) m₂ := by
        simp only [faceError, edgeError, hm1']; ring
      rw [hEq]; exact h12 m₂ ⟨hm2, by linarith⟩
    · have hm2' : m₂ = 1 - m₀ := by linarith
      have hEq : faceError d₀ d₁ d₂ (q (v₁ - v₀)) (q (v₂ - v₁)) (q (v₀ - v₂)) m₀ 0 m₂ =
          edgeError d₂ d₀ (q (v₀ - v₂)) m₀ := by
        simp only [faceError, edgeError, hm2']; ring
      rw [hEq]; exact h20 m₀ ⟨hm0, by linarith⟩
    · have hm0' : m₀ = 1 - m₁ := by linarith
      have hEq : faceError d₀ d₁ d₂ (q (v₁ - v₀)) (q (v₂ - v₁)) (q (v₀ - v₂)) m₀ m₁ 0 =
          edgeError d₀ d₁ (q (v₁ - v₀)) m₁ := by
        simp only [faceError, edgeError, hm0']; ring
      rw [hEq]; exact h01 m₁ ⟨hm1, by linarith⟩
  -- Pick a nonzero barycentric direction `(a₀, a₁, a₂)` with `a₀ + a₁ + a₂ = 0` along which
  -- the `y`-coordinate is constant, making `f - F` affine along the line!
  have hdir : ∃ a₀ a₁ a₂ : ℝ, a₀ + a₁ + a₂ = 0 ∧
      a₀ * v₀.2 + a₁ * v₁.2 + a₂ * v₂.2 = 0 ∧ (a₀ ≠ 0 ∨ a₁ ≠ 0 ∨ a₂ ≠ 0) := by
    by_cases hy : v₁.2 - v₂.2 = 0 ∧ v₂.2 - v₀.2 = 0 ∧ v₀.2 - v₁.2 = 0
    · exact ⟨1, -1, 0, by ring, by linarith, Or.inl one_ne_zero⟩
    · refine ⟨v₁.2 - v₂.2, v₂.2 - v₀.2, v₀.2 - v₁.2, by ring, by ring, ?_⟩
      tauto
  obtain ⟨a₀, a₁, a₂, ha_sum, ha_y, ha_ne⟩ := hdir
  -- Since `a₀ + a₁ + a₂ = 0` and not all are zero, at least one is positive and at least one is negative.
  have hpos_ex : 0 < a₀ ∨ 0 < a₁ ∨ 0 < a₂ := by
    by_contra h; push_neg at h
    rcases ha_ne with h0 | h1 | h2
    · exact h0 (by linarith)
    · exact h1 (by linarith)
    · exact h2 (by linarith)
  have hneg_ex : a₀ < 0 ∨ a₁ < 0 ∨ a₂ < 0 := by
    by_contra h; push_neg at h
    rcases ha_ne with h0 | h1 | h2
    · exact h0 (by linarith)
    · exact h1 (by linarith)
    · exact h2 (by linarith)
  -- Along `l(t) = l + t * a`, `y` is constant so `g(t) = (f - F)(l + t * a)` is affine in `t`!
  let g : ℝ → ℝ := fun t =>
    affineInterp v₀ v₁ v₂ d₀ d₁ d₂ (l₀ + t * a₀) (l₁ + t * a₁) (l₂ + t * a₂) -
    F (baryPoint v₀ v₁ v₂ (l₀ + t * a₀) (l₁ + t * a₁) (l₂ + t * a₂))
  have hg_affine : ∀ t : ℝ, g t = g 0 + t * (g 1 - g 0) := by
    intro t
    have hy0 : (l₀ + t * a₀) * v₀.2 + (l₁ + t * a₁) * v₁.2 + (l₂ + t * a₂) * v₂.2 =
        l₀ * v₀.2 + l₁ * v₁.2 + l₂ * v₂.2 := by
      calc _ = (l₀ * v₀.2 + l₁ * v₁.2 + l₂ * v₂.2) + t * (a₀ * v₀.2 + a₁ * v₁.2 + a₂ * v₂.2) := by ring
        _ = l₀ * v₀.2 + l₁ * v₁.2 + l₂ * v₂.2 := by rw [ha_y, mul_zero, add_zero]
    have hy1 : (l₀ + 1 * a₀) * v₀.2 + (l₁ + 1 * a₁) * v₁.2 + (l₂ + 1 * a₂) * v₂.2 =
        l₀ * v₀.2 + l₁ * v₁.2 + l₂ * v₂.2 := by
      calc _ = (l₀ * v₀.2 + l₁ * v₁.2 + l₂ * v₂.2) + 1 * (a₀ * v₀.2 + a₁ * v₁.2 + a₂ * v₂.2) := by ring
        _ = l₀ * v₀.2 + l₁ * v₁.2 + l₂ * v₂.2 := by rw [ha_y, mul_zero, add_zero]
    simp only [g, affineInterp, F, baryPoint]
    rw [hy0, hy1]
    ring
  -- Find `t₊ ≥ 0` where the ray `l + t * a` first exits the simplex (some coordinate becomes `0`).
  have hex_pos : ∃ t : ℝ, 0 ≤ t ∧ 0 ≤ l₀ + t * a₀ ∧ 0 ≤ l₁ + t * a₁ ∧ 0 ≤ l₂ + t * a₂ ∧
      (l₀ + t * a₀ = 0 ∨ l₁ + t * a₁ = 0 ∨ l₂ + t * a₂ = 0) := by
    let cand : Finset ℝ :=
      (if a₀ < 0 then {-l₀ / a₀} else ∅) ∪
      (if a₁ < 0 then {-l₁ / a₁} else ∅) ∪
      (if a₂ < 0 then {-l₂ / a₂} else ∅)
    have hcand_ne : cand.Nonempty := by
      rcases hneg_ex with h0 | h1 | h2
      · exact ⟨-l₀ / a₀, by simp [cand, h0]⟩
      · exact ⟨-l₁ / a₁, by simp [cand, h1]⟩
      · exact ⟨-l₂ / a₂, by simp [cand, h2]⟩
    let t := cand.min' hcand_ne
    have ht_mem : t ∈ cand := Finset.min'_mem cand hcand_ne
    have ht_le : ∀ x ∈ cand, t ≤ x := fun x hx => Finset.min'_le cand x hx
    have ht0 : 0 ≤ t := by
      simp only [cand, Finset.mem_union] at ht_mem
      rcases ht_mem with (h0 | h1) | h2
      · split_ifs at h0 with ha
        · simp only [Finset.mem_singleton] at h0; rw [h0]
          exact div_nonneg_of_nonpos (by linarith) (le_of_lt ha)
        · simp at h0
      · split_ifs at h1 with ha
        · simp only [Finset.mem_singleton] at h1; rw [h1]
          exact div_nonneg_of_nonpos (by linarith) (le_of_lt ha)
        · simp at h1
      · split_ifs at h2 with ha
        · simp only [Finset.mem_singleton] at h2; rw [h2]
          exact div_nonneg_of_nonpos (by linarith) (le_of_lt ha)
        · simp at h2
    have hcoord : ∀ (l a : ℝ), 0 ≤ l → (a < 0 → t ≤ -l / a) → 0 ≤ l + t * a := by
      intro l a hl hle
      by_cases ha : a < 0
      · have h1 : t ≤ l / (-a) := by
          have hrew : -l / a = l / (-a) := by field_simp [ne_of_lt ha]
          rw [← hrew]; exact hle ha
        have h2 : t * (-a) ≤ l := (le_div_iff₀ (by linarith : 0 < -a)).mp h1
        linarith
      · push_neg at ha; nlinarith
    refine ⟨t, ht0,
      hcoord l₀ a₀ hl0 (fun ha => ht_le (-l₀ / a₀) (by simp [cand, ha])),
      hcoord l₁ a₁ hl1 (fun ha => ht_le (-l₁ / a₁) (by simp [cand, ha])),
      hcoord l₂ a₂ hl2 (fun ha => ht_le (-l₂ / a₂) (by simp [cand, ha])), ?_⟩
    simp only [cand, Finset.mem_union] at ht_mem
    rcases ht_mem with (h0 | h1) | h2
    · split_ifs at h0 with ha
      · simp only [Finset.mem_singleton] at h0; left; rw [h0]; field_simp [ne_of_lt ha]; ring
      · simp at h0
    · split_ifs at h1 with ha
      · simp only [Finset.mem_singleton] at h1; right; left; rw [h1]; field_simp [ne_of_lt ha]; ring
      · simp at h1
    · split_ifs at h2 with ha
      · simp only [Finset.mem_singleton] at h2; right; right; rw [h2]; field_simp [ne_of_lt ha]; ring
      · simp at h2
  -- Find `t₋ ≤ 0` symmetrically (using `-a`).
  have hex_neg : ∃ s : ℝ, 0 ≤ s ∧ 0 ≤ l₀ - s * a₀ ∧ 0 ≤ l₁ - s * a₁ ∧ 0 ≤ l₂ - s * a₂ ∧
      (l₀ - s * a₀ = 0 ∨ l₁ - s * a₁ = 0 ∨ l₂ - s * a₂ = 0) := by
    let cand : Finset ℝ :=
      (if 0 < a₀ then {l₀ / a₀} else ∅) ∪
      (if 0 < a₁ then {l₁ / a₁} else ∅) ∪
      (if 0 < a₂ then {l₂ / a₂} else ∅)
    have hcand_ne : cand.Nonempty := by
      rcases hpos_ex with h0 | h1 | h2
      · exact ⟨l₀ / a₀, by simp [cand, h0]⟩
      · exact ⟨l₁ / a₁, by simp [cand, h1]⟩
      · exact ⟨l₂ / a₂, by simp [cand, h2]⟩
    let s := cand.min' hcand_ne
    have hs_mem : s ∈ cand := Finset.min'_mem cand hcand_ne
    have hs_le : ∀ x ∈ cand, s ≤ x := fun x hx => Finset.min'_le cand x hx
    have hs0 : 0 ≤ s := by
      simp only [cand, Finset.mem_union] at hs_mem
      rcases hs_mem with (h0 | h1) | h2
      · split_ifs at h0 with ha
        · simp only [Finset.mem_singleton] at h0; rw [h0]; exact div_nonneg hl0 (le_of_lt ha)
        · simp at h0
      · split_ifs at h1 with ha
        · simp only [Finset.mem_singleton] at h1; rw [h1]; exact div_nonneg hl1 (le_of_lt ha)
        · simp at h1
      · split_ifs at h2 with ha
        · simp only [Finset.mem_singleton] at h2; rw [h2]; exact div_nonneg hl2 (le_of_lt ha)
        · simp at h2
    have hcoord : ∀ (l a : ℝ), 0 ≤ l → (0 < a → s ≤ l / a) → 0 ≤ l - s * a := by
      intro l a hl hle
      by_cases ha : 0 < a
      · have := (le_div_iff₀ ha).mp (hle ha)
        linarith
      · push_neg at ha; nlinarith
    refine ⟨s, hs0,
      hcoord l₀ a₀ hl0 (fun ha => hs_le (l₀ / a₀) (by simp [cand, ha])),
      hcoord l₁ a₁ hl1 (fun ha => hs_le (l₁ / a₁) (by simp [cand, ha])),
      hcoord l₂ a₂ hl2 (fun ha => hs_le (l₂ / a₂) (by simp [cand, ha])), ?_⟩
    simp only [cand, Finset.mem_union] at hs_mem
    rcases hs_mem with (h0 | h1) | h2
    · split_ifs at h0 with ha
      · simp only [Finset.mem_singleton] at h0; left; rw [h0]; field_simp [ne_of_gt ha]; ring
      · simp at h0
    · split_ifs at h1 with ha
      · simp only [Finset.mem_singleton] at h1; right; left; rw [h1]; field_simp [ne_of_gt ha]; ring
      · simp at h1
    · split_ifs at h2 with ha
      · simp only [Finset.mem_singleton] at h2; right; right; rw [h2]; field_simp [ne_of_gt ha]; ring
      · simp at h2
  obtain ⟨t, ht0, ht_l0, ht_l1, ht_l2, ht_z⟩ := hex_pos
  obtain ⟨s, hs0, hs_l0, hs_l1, hs_l2, hs_z⟩ := hex_neg
  have hgt : |g t| ≤ ε := hbdry _ _ _ ht_l0 ht_l1 ht_l2
    (by calc (l₀ + t * a₀) + (l₁ + t * a₁) + (l₂ + t * a₂)
            = (l₀ + l₁ + l₂) + t * (a₀ + a₁ + a₂) := by ring
          _ = 1 := by rw [hsum, ha_sum]; ring) ht_z
  have hgs : |g (-s)| ≤ ε := by
    have h1 : l₀ + -s * a₀ = l₀ - s * a₀ := by ring
    have h2 : l₁ + -s * a₁ = l₁ - s * a₁ := by ring
    have h3 : l₂ + -s * a₂ = l₂ - s * a₂ := by ring
    simp only [g, h1, h2, h3]
    exact hbdry _ _ _ hs_l0 hs_l1 hs_l2
      (by calc (l₀ - s * a₀) + (l₁ - s * a₁) + (l₂ - s * a₂)
              = (l₀ + l₁ + l₂) - s * (a₀ + a₁ + a₂) := by ring
            _ = 1 := by rw [hsum, ha_sum]; ring) hs_z
  have hg0_eq : g 0 = affineInterp v₀ v₁ v₂ d₀ d₁ d₂ l₀ l₁ l₂ - F (baryPoint v₀ v₁ v₂ l₀ l₁ l₂) := by
    simp [g]
  rw [← hg0_eq]
  by_cases hts : t + s = 0
  · have ht_eq : t = 0 := by linarith
    simpa [ht_eq] using hgt
  · have hts_pos : 0 < t + s := lt_of_le_of_ne (by linarith) (Ne.symm hts)
    have hconv : g 0 = (s / (t + s)) * g t + (t / (t + s)) * g (-s) := by
      rw [hg_affine t, hg_affine (-s)]
      field_simp [ne_of_gt hts_pos]
      ring
    rw [hconv]
    have hlo_t := (abs_le.mp hgt).1
    have hhi_t := (abs_le.mp hgt).2
    have hlo_s := (abs_le.mp hgs).1
    have hhi_s := (abs_le.mp hgs).2
    have hc1 : 0 ≤ s / (t + s) := div_nonneg hs0 (le_of_lt hts_pos)
    have hc2 : 0 ≤ t / (t + s) := div_nonneg ht0 (le_of_lt hts_pos)
    have hcsum : s / (t + s) + t / (t + s) = 1 := by field_simp [ne_of_gt hts_pos]; ring
    rw [abs_le]
    constructor <;> nlinarith

/-- **Exact triangle area identity**:
`4 * A(v₀, v₁, v₂)² = (q(v₁ - v₀) + q(v₂ - v₀) - q(v₂ - v₁))² - 4 * q(v₁ - v₀) * q(v₂ - v₀)`. -/
theorem four_area_sq_eq (v₀ v₁ v₂ : ℝ × ℝ) :
    4 * (triArea v₀ v₁ v₂) ^ 2 =
      (q (v₁ - v₀) + q (v₂ - v₀) - q (v₂ - v₁)) ^ 2 - 4 * q (v₁ - v₀) * q (v₂ - v₀) := by
  simp only [triArea, q, Prod.fst_sub, Prod.snd_sub, div_pow, sq_abs]
  ring

/-! ### 2. Uniform Signed-Edge-Product Triangulations (Theorem 2) -/

/-- **Theorem 2, Case 1 (One-sign triangles)**: If all three signed edge products of a valid
triangle have the same sign (all `≥ 0` or all `≤ 0`), then `A ≤ 4ε < 8ε / √3`. -/
theorem one_sign_triangle_area_le {v₀ v₁ v₂ : ℝ × ℝ} {d₀ d₁ d₂ ε : ℝ} (hε : 0 < ε)
    (h01 : ∀ t ∈ Set.Icc (0 : ℝ) 1, |edgeError d₀ d₁ (q (v₁ - v₀)) t| ≤ ε)
    (h12 : ∀ t ∈ Set.Icc (0 : ℝ) 1, |edgeError d₁ d₂ (q (v₂ - v₁)) t| ≤ ε)
    (h20 : ∀ t ∈ Set.Icc (0 : ℝ) 1, |edgeError d₂ d₀ (q (v₀ - v₂)) t| ≤ ε)
    (hsign : (0 ≤ q (v₁ - v₀) ∧ 0 ≤ q (v₂ - v₁) ∧ 0 ≤ q (v₀ - v₂)) ∨
             (q (v₁ - v₀) ≤ 0 ∧ q (v₂ - v₁) ≤ 0 ∧ q (v₀ - v₂) ≤ 0)) :
    triArea v₀ v₁ v₂ ≤ 4 * ε ∧ 4 * ε < 8 * ε / Real.sqrt 3 := by
  -- Every edge product satisfies `|q(e)| ≤ 8ε` because `(√(ε ± d_i) + √(ε ± d_j))² ≤ 2(2ε + 2ε) = 8ε`.
  have hq_abs : ∀ (u : ℝ × ℝ) (a b : ℝ),
      (∀ t ∈ Set.Icc (0 : ℝ) 1, |edgeError a b (q u) t| ≤ ε) → |q u| ≤ 8 * ε := by
    intro u a b he
    have ⟨ha, hb, hlo, hhi⟩ := (edgeError_abs_le_iff hε).mp he
    have ha_lo := (abs_le.mp ha).1; have ha_hi := (abs_le.mp ha).2
    have hb_lo := (abs_le.mp hb).1; have hb_hi := (abs_le.mp hb).2
    have hhi8 : (Real.sqrt (ε - a) + Real.sqrt (ε - b)) ^ 2 ≤ 8 * ε := by
      have h1 : (Real.sqrt (ε - a) + Real.sqrt (ε - b)) ^ 2 ≤
          2 * (Real.sqrt (ε - a) ^ 2 + Real.sqrt (ε - b) ^ 2) := by
        nlinarith [sq_nonneg (Real.sqrt (ε - a) - Real.sqrt (ε - b))]
      rw [Real.sq_sqrt (by linarith), Real.sq_sqrt (by linarith)] at h1
      linarith
    have hlo8 : (Real.sqrt (ε + a) + Real.sqrt (ε + b)) ^ 2 ≤ 8 * ε := by
      have h1 : (Real.sqrt (ε + a) + Real.sqrt (ε + b)) ^ 2 ≤
          2 * (Real.sqrt (ε + a) ^ 2 + Real.sqrt (ε + b) ^ 2) := by
        nlinarith [sq_nonneg (Real.sqrt (ε + a) - Real.sqrt (ε + b))]
      rw [Real.sq_sqrt (by linarith), Real.sq_sqrt (by linarith)] at h1
      linarith
    rw [abs_le]; constructor <;> linarith
  have hb01 := hq_abs _ _ _ h01
  have hb12 := hq_abs _ _ _ h12
  have hb20 := hq_abs _ _ _ h20
  -- Note `q (v₂ - v₀) = q (v₀ - v₂)`.
  have hq20 : q (v₂ - v₀) = q (v₀ - v₂) := by simp [q]; ring
  -- Algebraically, when `p, r, s` all have the same sign and `|p|, |r|, |s| ≤ 8ε`,
  -- `(p + r - s)² - 4 p r ≤ (8ε)² = 64 ε²`!
  -- Proof: for `a, b, c ≥ 0` with `a, b, c ≤ M`, `(a + b - c)² - 4 a b = (a - b)² - 2c(a + b) + c²`:
  -- Wait: if `c ≤ a + b`, `(a + b - c)² - 4ab ≤ (a - b)² ≤ M²` (if `c` is in `[0, a+b]`, wait:
  -- `(a + b - c)² - 4ab` is convex in `c ∈ [0, M]`: at `c = 0` it is `(a - b)² ≤ M²`;
  -- at `c = M` it is `(a + b - M)² - 4ab = (M - a - b)² - 4ab`: wait, `4ab ≥ 0`,
  -- and `(M - a - b)² - 4ab = M² - 2M(a+b) + (a-b)² ≤ M²` since `(a-b)² ≤ (a+b)² ≤ 2M(a+b)`!
  -- EVEN BETTER: `M² - ((a + b - c)² - 4ab)`... wait: check `a = M, b = M, c = 0`: `(2M)^2 - 4M^2 = 0 ≤ M^2`!
  -- Check `a = M, b = 0, c = 0`: `M^2`! Check `a = 0, b = 0, c = M`: `M^2`!
  -- Check `a = M, b = M, c = M`: `M^2 - 4M^2 = -3M^2 ≤ M^2`!
  have hpoly : ∀ a b c M : ℝ, 0 ≤ a → a ≤ M → 0 ≤ b → b ≤ M → 0 ≤ c → c ≤ M →
      (a + b - c) ^ 2 - 4 * a * b ≤ M ^ 2 := by
    intro a b c M ha haM hb hbM hc hcM
    by_cases hc2 : c ≤ 2 * (a + b)
    · have h1 : (a + b - c) ^ 2 - 4 * a * b = (a - b) ^ 2 + c * (c - 2 * (a + b)) := by ring
      have h2 : c * (c - 2 * (a + b)) ≤ 0 := mul_nonpos_of_nonneg_of_nonpos hc (by linarith)
      nlinarith
    · push_neg at hc2
      have h1 : 0 ≤ c - (a + b) := by linarith
      have h2 : c - (a + b) ≤ M := by linarith
      nlinarith [mul_nonneg ha hb]
  have h4A : 4 * (triArea v₀ v₁ v₂) ^ 2 ≤ (8 * ε) ^ 2 := by
    rw [four_area_sq_eq, hq20]
    rcases hsign with ⟨h1, h2, h3⟩ | ⟨h1, h2, h3⟩
    · exact hpoly (q (v₁ - v₀)) (q (v₀ - v₂)) (q (v₂ - v₁)) (8 * ε)
        h1 (abs_le.mp hb01).2 h3 (abs_le.mp hb20).2 h2 (abs_le.mp hb12).2
    · have hEq : (q (v₁ - v₀) + q (v₀ - v₂) - q (v₂ - v₁)) ^ 2 - 4 * q (v₁ - v₀) * q (v₀ - v₂) =
          ((-q (v₁ - v₀)) + (-q (v₀ - v₂)) - (-q (v₂ - v₁))) ^ 2 -
            4 * (-q (v₁ - v₀)) * (-q (v₀ - v₂)) := by ring
      rw [hEq]
      exact hpoly (-q (v₁ - v₀)) (-q (v₀ - v₂)) (-q (v₂ - v₁)) (8 * ε)
        (by linarith) (by linarith [(abs_le.mp hb01).1])
        (by linarith) (by linarith [(abs_le.mp hb20).1])
        (by linarith) (by linarith [(abs_le.mp hb12).1])
  have hA_nn : 0 ≤ triArea v₀ v₁ v₂ := by simp only [triArea]; positivity
  have hA_le : triArea v₀ v₁ v₂ ≤ 4 * ε := by nlinarith
  have hlt : 4 * ε < 8 * ε / Real.sqrt 3 := by
    have hs3 : 0 < Real.sqrt 3 := Real.sqrt_pos.mpr (by norm_num)
    rw [lt_div_iff₀ hs3]
    have hs3_lt : Real.sqrt 3 < 2 := by
      rw [Real.sqrt_lt' (by norm_num)]; norm_num
    nlinarith
  exact ⟨hA_le, hlt⟩

/-- **Theorem 2, Case 2 (Alternating walk sum bound)**: Along an infinite alternating `(b, -w)`-walk
with bounded vertex deviations `|d_k| ≤ ε`, the midpoint error test forces `b + w ≤ 8ε`. -/
theorem alternating_walk_sum_le {ε b w : ℝ} (_hε : 0 < ε) {d : ℕ → ℝ}
    (hd : ∀ k, |d k| ≤ ε)
    (hb : ∀ k, b ≤ 4 * ε - 2 * (d (2 * k) + d (2 * k + 1)))
    (hw : ∀ k, w ≤ 4 * ε + 2 * (d (2 * k + 1) + d (2 * k + 2))) :
    b + w ≤ 8 * ε := by
  by_contra hgt
  push_neg at hgt
  let δ := (b + w - 8 * ε) / 2
  have hδ : 0 < δ := by dsimp [δ]; linarith
  have hstep : ∀ k, d (2 * k) + δ ≤ d (2 * (k + 1)) := by
    intro k
    have h1 := hb k
    have h2 := hw k
    have hidx : 2 * (k + 1) = 2 * k + 2 := by ring
    rw [hidx]
    dsimp [δ]; linarith
  have hind : ∀ n : ℕ, d 0 + (n : ℝ) * δ ≤ d (2 * n) := by
    intro n
    induction n with
    | zero => simp
    | succ m ih =>
      have hm := hstep m
      push_cast
      linarith
  obtain ⟨N, hN⟩ := exists_nat_gt (2 * ε / δ)
  have h1 := hind N
  have h0_lo := (abs_le.mp (hd 0)).1
  have hN_hi := (abs_le.mp (hd (2 * N))).2
  have hmul : 2 * ε < (N : ℝ) * δ := (div_lt_iff₀ hδ).mp hN
  linarith

/-- **Theorem 2, Case 2 (Four-spoke bilinear obstruction for `0 < a < b`)**:
No four planar spokes `p₁, p₂, n₁, n₂` with `p₁ ≠ p₂`, `n₁ ≠ n₂`, `q(pᵢ) = a > 0`, `q(nⱼ) = -w < 0`,
and `q(pᵢ - nⱼ) = b > a` for all `i, j ∈ {1, 2}` can exist. -/
theorem four_spoke_equal_product_impossible {p₁ p₂ n₁ n₂ : ℝ × ℝ} {a b w : ℝ}
    (_ha : 0 < a) (hab : a < b) (hw : 0 < w)
    (hpne : p₁ ≠ p₂) (hnne : n₁ ≠ n₂)
    (hp1 : q p₁ = a) (hp2 : q p₂ = a) (hn1 : q n₁ = -w) (hn2 : q n₂ = -w)
    (h11 : q (p₁ - n₁) = b) (h12 : q (p₁ - n₂) = b)
    (h21 : q (p₂ - n₁) = b) (h22 : q (p₂ - n₂) = b) : False := by
  -- Subtracting `q(p₁ - nⱼ) = q(p₂ - nⱼ)` gives `(p₁ - p₂)_x * nⱼ_y + (p₁ - p₂)_y * nⱼ_x = 0`.
  have hB1 : (p₁.1 - p₂.1) * n₁.2 + (p₁.2 - p₂.2) * n₁.1 = 0 := by
    simp only [q, Prod.fst_sub, Prod.snd_sub] at hp1 hp2 hn1 h11 h21
    linarith
  have hB2 : (p₁.1 - p₂.1) * n₂.2 + (p₁.2 - p₂.2) * n₂.1 = 0 := by
    simp only [q, Prod.fst_sub, Prod.snd_sub] at hp1 hp2 hn2 h12 h22
    linarith
  -- Since `p₁ ≠ p₂`, `(p₁.1 - p₂.1, p₁.2 - p₂.2) ≠ (0, 0)`, so `n₁` and `n₂` are collinear.
  have hcol : n₁.1 * n₂.2 = n₁.2 * n₂.1 := by
    by_cases hx : p₁.1 - p₂.1 = 0
    · have hy : p₁.2 - p₂.2 ≠ 0 := fun hy => hpne (Prod.ext (by linarith) (by linarith))
      rw [hx, zero_mul, zero_add] at hB1 hB2
      have h1 : n₁.1 = 0 := (mul_eq_zero.mp hB1).resolve_left hy
      have h2 : n₂.1 = 0 := (mul_eq_zero.mp hB2).resolve_left hy
      simp [h1, h2]
    · have hmul : (p₁.1 - p₂.1) * (n₁.2 * n₂.1 - n₂.2 * n₁.1) = 0 := by
        calc (p₁.1 - p₂.1) * (n₁.2 * n₂.1 - n₂.2 * n₁.1)
            = ((p₁.1 - p₂.1) * n₁.2 + (p₁.2 - p₂.2) * n₁.1) * n₂.1 -
              ((p₁.1 - p₂.1) * n₂.2 + (p₁.2 - p₂.2) * n₂.1) * n₁.1 := by ring
          _ = 0 := by rw [hB1, hB2]; ring
      have hz := (mul_eq_zero.mp hmul).resolve_left hx
      linarith
  -- From `q(n₁) = q(n₂) = -w ≠ 0` and collinearity, `n₂ = -n₁` (as `n₁ ≠ n₂`).
  have hn1x : n₁.1 ≠ 0 := by
    intro h0; simp only [q, h0, zero_mul] at hn1; linarith
  have hn2x : n₂.1 ≠ 0 := by
    intro h0; simp only [q, h0, zero_mul] at hn2; linarith
  have hsq : n₂.1 ^ 2 = n₁.1 ^ 2 := by
    have h1 : n₁.1 * (n₂.1 * n₂.2) = n₁.1 * (-w) := by rw [← hn2]; rfl
    have h2 : n₂.1 * (n₁.1 * n₁.2) = n₂.1 * (-w) := by rw [← hn1]; rfl
    -- `n₁.1² * (-w) = n₁.1 * n₁.1 * n₂.1 * n₂.2 = n₂.1² * n₁.1 * n₁.2 = n₂.1² * (-w)`.
    have h3 : n₁.1 ^ 2 * (-w) = n₂.1 ^ 2 * (-w) := by
      calc n₁.1 ^ 2 * (-w) = n₁.1 ^ 2 * (n₂.1 * n₂.2) := by rw [← hn2]; rfl
        _ = n₁.1 * n₂.1 * (n₁.1 * n₂.2) := by ring
        _ = n₁.1 * n₂.1 * (n₁.2 * n₂.1) := by rw [hcol]
        _ = n₂.1 ^ 2 * (n₁.1 * n₁.2) := by ring
        _ = n₂.1 ^ 2 * (-w) := by rw [← hn1]; rfl
    exact mul_right_cancel₀ (by linarith : (-w) ≠ 0) h3.symm
  have hneg : n₂.1 = -n₁.1 ∧ n₂.2 = -n₁.2 := by
    rcases sq_eq_sq_iff_eq_or_eq_neg.mp hsq with hpos | hnegx
    · have hy : n₂.2 = n₁.2 := by
        have h1 : n₁.1 * n₂.2 = n₁.1 * n₁.2 := by rw [hcol, hpos]; ring
        exact mul_left_cancel₀ hn1x h1
      exact (hnne (Prod.ext hpos.symm hy.symm)).elim
    · have hy : n₂.2 = -n₁.2 := by
        have h1 : n₁.1 * n₂.2 = n₁.1 * (-n₁.2) := by rw [hcol, hnegx]; ring
        exact mul_left_cancel₀ hn1x h1
      exact ⟨hnegx, hy⟩
  -- Averaging `q(p₁ - n₁) = b` and `q(p₁ + n₁) = b` gives `b = q(p₁) + q(n₁) = a - w < a`.
  simp only [q, Prod.fst_sub, Prod.snd_sub, hneg.1, hneg.2] at hp1 hn1 h11 h12
  linarith

/-- **Theorem 2, Case 2 (Null-edge flanking lemma for `a = 0`)**:
If a horizontal null edge `(s, 0)` (`s ≠ 0`) is flanked on opposite sides (`y₁ * y₂ < 0`) by two
triangles whose third vertices `(x₁, y₁)` and `(x₂, y₂)` each have the edge-product pair `{b, -w}`
with `b, w > 0`, then the two edges from the origin cannot both have product `b` or both `-w`. -/
theorem null_edge_flanking_opposite_signs {s x₁ y₁ x₂ y₂ b w : ℝ}
    (hs : s ≠ 0) (hb : 0 < b) (hw : 0 < w) (hopp : y₁ * y₂ < 0)
    (ht1 : (x₁ * y₁ = b ∧ (x₁ - s) * y₁ = -w) ∨ (x₁ * y₁ = -w ∧ (x₁ - s) * y₁ = b))
    (ht2 : (x₂ * y₂ = b ∧ (x₂ - s) * y₂ = -w) ∨ (x₂ * y₂ = -w ∧ (x₂ - s) * y₂ = b)) :
    (x₁ * y₁ = b ∧ x₂ * y₂ = -w) ∨ (x₁ * y₁ = -w ∧ x₂ * y₂ = b) := by
  rcases ht1 with ⟨h1a, h1b⟩ | ⟨h1a, h1b⟩ <;> rcases ht2 with ⟨h2a, h2b⟩ | ⟨h2a, h2b⟩
  · exfalso
    have hs1 : s * y₁ = b + w := by linarith
    have hs2 : s * y₂ = b + w := by linarith
    have hprod : s ^ 2 * (y₁ * y₂) = (b + w) ^ 2 := by nlinarith
    have hpos : 0 < s ^ 2 := sq_pos_of_ne_zero hs
    nlinarith [sq_pos_of_pos (by linarith : 0 < b + w)]
  · exact Or.inl ⟨h1a, h2a⟩
  · exact Or.inr ⟨h1a, h2a⟩
  · exfalso
    have hs1 : s * y₁ = -(b + w) := by linarith
    have hs2 : s * y₂ = -(b + w) := by linarith
    have hprod : s ^ 2 * (y₁ * y₂) = (b + w) ^ 2 := by nlinarith
    have hpos : 0 < s ^ 2 := sq_pos_of_ne_zero hs
    nlinarith [sq_pos_of_pos (by linarith : 0 < b + w)]

/-- **Theorem 2 (Exact area cap `8ε / √3` under `b + w ≤ 8ε`)**:
For any mixed-sign triangle with edge products `{a, b, -w}` satisfying `0 ≤ a ≤ b`, `0 < w`,
and `b + w ≤ 8ε`, its area satisfies `A ≤ 8ε / √3`. -/
theorem mixed_sign_area_le_eight_div_sqrt3 {ε a b w A : ℝ}
    (hε : 0 < ε) (ha : 0 ≤ a) (hab : a ≤ b) (hw : 0 < w)
    (hsum : b + w ≤ 8 * ε) (hA : 0 ≤ A)
    (harea : 4 * A ^ 2 = (b - a) ^ 2 + 2 * w * (a + b) + w ^ 2) :
    A ≤ 8 * ε / Real.sqrt 3 := by
  -- Convex in `a ∈ [0, b]`: bounded by endpoints `a = 0` (`(b + w)² ≤ 64ε²`) and
  -- `a = b` (`w(4b + w) ≤ w(32ε - 3w) = 256ε²/3 - 3(w - 16ε/3)² ≤ 256ε²/3`).
  have h4A : 4 * A ^ 2 ≤ 256 * ε ^ 2 / 3 := by
    have h0 : (b - 0) ^ 2 + 2 * w * (0 + b) + w ^ 2 ≤ 256 * ε ^ 2 / 3 := by nlinarith
    have hb_end : (b - b) ^ 2 + 2 * w * (b + b) + w ^ 2 ≤ 256 * ε ^ 2 / 3 := by
      have h1 : 4 * w * b + w ^ 2 ≤ w * (32 * ε - 3 * w) := by nlinarith
      have h2 : w * (32 * ε - 3 * w) = 256 * ε ^ 2 / 3 - 3 * (w - 16 * ε / 3) ^ 2 := by ring
      nlinarith [sq_nonneg (w - 16 * ε / 3)]
    have hconv : (b - a) ^ 2 + 2 * w * (a + b) + w ^ 2 ≤
        max ((b - 0) ^ 2 + 2 * w * (0 + b) + w ^ 2)
            ((b - b) ^ 2 + 2 * w * (b + b) + w ^ 2) := by
      by_cases hb0 : b = 0
      · have ha0 : a = 0 := by linarith
        simp only [hb0, ha0]
        exact le_max_left _ _
      · have hbpos : 0 < b := lt_of_le_of_ne (by linarith) (Ne.symm hb0)
        have hle_lin : (b - a) ^ 2 + 2 * w * (a + b) + w ^ 2 ≤
            ((b - a) / b) * ((b - 0) ^ 2 + 2 * w * (0 + b) + w ^ 2) +
            (a / b) * ((b - b) ^ 2 + 2 * w * (b + b) + w ^ 2) := by
          have hdiff : ((b - a) / b) * ((b - 0) ^ 2 + 2 * w * (0 + b) + w ^ 2) +
              (a / b) * ((b - b) ^ 2 + 2 * w * (b + b) + w ^ 2) -
              ((b - a) ^ 2 + 2 * w * (a + b) + w ^ 2) = a * (b - a) := by
            field_simp [ne_of_gt hbpos]; ring
          nlinarith [mul_nonneg ha (sub_nonneg.mpr hab)]
        have h1 : 0 ≤ (b - a) / b := div_nonneg (by linarith) (le_of_lt hbpos)
        have h2 : 0 ≤ a / b := div_nonneg ha (le_of_lt hbpos)
        have h3 : (b - a) / b + a / b = 1 := by field_simp [ne_of_gt hbpos]; ring
        have hm1 := le_max_left ((b - 0) ^ 2 + 2 * w * (0 + b) + w ^ 2)
          ((b - b) ^ 2 + 2 * w * (b + b) + w ^ 2)
        have hm2 := le_max_right ((b - 0) ^ 2 + 2 * w * (0 + b) + w ^ 2)
          ((b - b) ^ 2 + 2 * w * (b + b) + w ^ 2)
        nlinarith
    rw [harea]
    exact le_trans hconv (max_le h0 hb_end)
  have hs3 : 0 < Real.sqrt 3 := Real.sqrt_pos.mpr (by norm_num)
  rw [le_div_iff₀ hs3]
  have hsq : (A * Real.sqrt 3) ^ 2 ≤ (8 * ε) ^ 2 := by
    rw [mul_pow, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 3)]
    linarith
  have habs : |A * Real.sqrt 3| ≤ |8 * ε| := sq_le_sq.mp hsq
  rwa [abs_of_nonneg (mul_nonneg hA (le_of_lt hs3)), abs_of_pos (by linarith)] at habs

/-- **Theorem 2 (Density lower bound from uniform area cap `8ε / √3`)**:
If `N` triangles of areas at most `8ε / √3` cover a square of side `ℓ > 0`, then
`N / ℓ² ≥ √3 / (8ε)`. -/
theorem uniform_edge_product_density_lower_bound {ε ℓ : ℝ} (hε : 0 < ε) (hℓ : 0 < ℓ)
    {N : ℕ} (hcover : ℓ ^ 2 ≤ (N : ℝ) * (8 * ε / Real.sqrt 3)) :
    Real.sqrt 3 / (8 * ε) ≤ (N : ℝ) / ℓ ^ 2 := by
  have hs3 : 0 < Real.sqrt 3 := Real.sqrt_pos.mpr (by norm_num)
  rw [div_le_div_iff₀ (by positivity) (sq_pos_of_pos hℓ)]
  have hmul := mul_le_mul_of_nonneg_left hcover (le_of_lt hs3)
  have hcancel : Real.sqrt 3 * ((N : ℝ) * (8 * ε / Real.sqrt 3)) = (N : ℝ) * (8 * ε) := by
    field_simp [ne_of_gt hs3]
  linarith

/-- **Theorem 2 (Sharpness witness)**: For every `ε > 0`, the triangle with edge vectors
`u = √ε (s, t)` and `v = √ε (t, s)` (`s = 2 + 2/√3`, `t = 2 - 2/√3`) and constant vertex
deviation `d = ε / 3` has edge products `{8ε/3, 8ε/3, -16ε/3}`, exact area `8ε / √3`, and
global error `≤ ε` on the entire triangle. -/
theorem sharp_parallelogram_tiling_face {ε : ℝ} (hε : 0 < ε) :
    let r := Real.sqrt ε
    let s := 2 + 2 / Real.sqrt 3
    let t := 2 - 2 / Real.sqrt 3
    let v₀ : ℝ × ℝ := (0, 0)
    let v₁ : ℝ × ℝ := (r * s, r * t)
    let v₂ : ℝ × ℝ := (r * t, r * s)
    let d := ε / 3
    q (v₁ - v₀) = 8 * ε / 3 ∧
    q (v₂ - v₀) = 8 * ε / 3 ∧
    q (v₂ - v₁) = -16 * ε / 3 ∧
    triArea v₀ v₁ v₂ = 8 * ε / Real.sqrt 3 ∧
    (∀ l₀ l₁ l₂ : ℝ, 0 ≤ l₀ → 0 ≤ l₁ → 0 ≤ l₂ → l₀ + l₁ + l₂ = 1 →
      |affineInterp v₀ v₁ v₂ d d d l₀ l₁ l₂ - F (baryPoint v₀ v₁ v₂ l₀ l₁ l₂)| ≤ ε) := by
  dsimp only
  have hs3 : 0 < Real.sqrt 3 := Real.sqrt_pos.mpr (by norm_num)
  have hs3_sq : Real.sqrt 3 ^ 2 = 3 := Real.sq_sqrt (by norm_num)
  have hr_sq : Real.sqrt ε ^ 2 = ε := Real.sq_sqrt (le_of_lt hε)
  have hq01 : q ((Real.sqrt ε * (2 + 2 / Real.sqrt 3), Real.sqrt ε * (2 - 2 / Real.sqrt 3)) - (0, 0)) =
      8 * ε / 3 := by
    simp only [q, Prod.fst_sub, Prod.snd_sub, sub_zero]
    have h1 : (Real.sqrt ε * (2 + 2 / Real.sqrt 3)) * (Real.sqrt ε * (2 - 2 / Real.sqrt 3)) =
        Real.sqrt ε ^ 2 * (4 - 4 / Real.sqrt 3 ^ 2) := by
      field_simp [ne_of_gt hs3]; ring
    rw [h1, hr_sq, hs3_sq]; ring
  have hq02 : q ((Real.sqrt ε * (2 - 2 / Real.sqrt 3), Real.sqrt ε * (2 + 2 / Real.sqrt 3)) - (0, 0)) =
      8 * ε / 3 := by
    simp only [q, Prod.fst_sub, Prod.snd_sub, sub_zero]
    have h1 : (Real.sqrt ε * (2 - 2 / Real.sqrt 3)) * (Real.sqrt ε * (2 + 2 / Real.sqrt 3)) =
        Real.sqrt ε ^ 2 * (4 - 4 / Real.sqrt 3 ^ 2) := by
      field_simp [ne_of_gt hs3]; ring
    rw [h1, hr_sq, hs3_sq]; ring
  have hq12 : q ((Real.sqrt ε * (2 - 2 / Real.sqrt 3), Real.sqrt ε * (2 + 2 / Real.sqrt 3)) -
      (Real.sqrt ε * (2 + 2 / Real.sqrt 3), Real.sqrt ε * (2 - 2 / Real.sqrt 3))) =
      -16 * ε / 3 := by
    simp only [q, Prod.fst_sub, Prod.snd_sub]
    have h1 : (Real.sqrt ε * (2 - 2 / Real.sqrt 3) - Real.sqrt ε * (2 + 2 / Real.sqrt 3)) *
        (Real.sqrt ε * (2 + 2 / Real.sqrt 3) - Real.sqrt ε * (2 - 2 / Real.sqrt 3)) =
        -(16 * Real.sqrt ε ^ 2 / Real.sqrt 3 ^ 2) := by
      field_simp [ne_of_gt hs3]; ring
    rw [h1, hr_sq, hs3_sq]; ring
  have hq20 : q ((0, 0) - (Real.sqrt ε * (2 - 2 / Real.sqrt 3), Real.sqrt ε * (2 + 2 / Real.sqrt 3))) =
      8 * ε / 3 := by
    have : q ((0, 0) - (Real.sqrt ε * (2 - 2 / Real.sqrt 3), Real.sqrt ε * (2 + 2 / Real.sqrt 3))) =
        q ((Real.sqrt ε * (2 - 2 / Real.sqrt 3), Real.sqrt ε * (2 + 2 / Real.sqrt 3)) - (0, 0)) := by
      simp [q]
    rw [this, hq02]
  have harea : triArea (0, 0)
      (Real.sqrt ε * (2 + 2 / Real.sqrt 3), Real.sqrt ε * (2 - 2 / Real.sqrt 3))
      (Real.sqrt ε * (2 - 2 / Real.sqrt 3), Real.sqrt ε * (2 + 2 / Real.sqrt 3)) =
      8 * ε / Real.sqrt 3 := by
    simp only [triArea, sub_zero]
    have hdet : (Real.sqrt ε * (2 + 2 / Real.sqrt 3)) * (Real.sqrt ε * (2 + 2 / Real.sqrt 3)) -
        (Real.sqrt ε * (2 - 2 / Real.sqrt 3)) * (Real.sqrt ε * (2 - 2 / Real.sqrt 3)) =
        16 * ε / Real.sqrt 3 := by
      have : (Real.sqrt ε * (2 + 2 / Real.sqrt 3)) * (Real.sqrt ε * (2 + 2 / Real.sqrt 3)) -
          (Real.sqrt ε * (2 - 2 / Real.sqrt 3)) * (Real.sqrt ε * (2 - 2 / Real.sqrt 3)) =
          16 * Real.sqrt ε ^ 2 / Real.sqrt 3 := by
        field_simp [ne_of_gt hs3]; ring
      rw [this, hr_sq]
    rw [hdet, abs_of_pos (by positivity)]
    ring
  have hedge_pos : ∀ t ∈ Set.Icc (0 : ℝ) 1, |edgeError (ε / 3) (ε / 3) (8 * ε / 3) t| ≤ ε := by
    intro t ⟨ht0, ht1⟩
    simp only [edgeError]
    rw [abs_le]
    have h1 : t * (1 - t) ≤ 1 / 4 := by nlinarith [sq_nonneg (t - 1 / 2)]
    have h2 : 0 ≤ t * (1 - t) := mul_nonneg ht0 (by linarith)
    constructor <;> nlinarith
  have hedge_neg : ∀ t ∈ Set.Icc (0 : ℝ) 1, |edgeError (ε / 3) (ε / 3) (-16 * ε / 3) t| ≤ ε := by
    intro t ⟨ht0, ht1⟩
    simp only [edgeError]
    rw [abs_le]
    have h1 : t * (1 - t) ≤ 1 / 4 := by nlinarith [sq_nonneg (t - 1 / 2)]
    have h2 : 0 ≤ t * (1 - t) := mul_nonneg ht0 (by linarith)
    constructor <;> nlinarith
  refine ⟨hq01, hq02, hq12, harea, fun l₀ l₁ l₂ hl0 hl1 hl2 hsum => ?_⟩
  exact faceError_abs_le_of_edges
    (by rw [hq01]; exact hedge_pos)
    (by rw [hq12]; exact hedge_neg)
    (by rw [hq20]; exact hedge_pos)
    hl0 hl1 hl2 hsum

/-! ### 3. Explicit Continuous Periodic Triangulation with Area `5ε > 8ε / √3` (Proposition 1) -/

/-- A triangle with assigned vertex deviations. -/
structure DevTriangle where
  v₀ : ℝ × ℝ
  v₁ : ℝ × ℝ
  v₂ : ℝ × ℝ
  d₀ : ℝ
  d₁ : ℝ
  d₂ : ℝ

/-- Vertex deviation function on the period box `[0, 21/10] × [0, 51/10]` (Proposition 1). -/
def periodicDev (p : ℝ × ℝ) : ℝ :=
  if p.1 = 2 then 0
  else if p.2 = 5 then 2 / 3
  else -2 / 3

/-- The 10 triangles forming one period of the continuous periodic mesh on `[0, 21/10] × [0, 51/10]`. -/
def periodicTriangles : List DevTriangle :=
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (0, 5)
  let D : ℝ × ℝ := (2, 0)
  let C : ℝ × ℝ := (2, 5 / 2)
  let E : ℝ × ℝ := (2, 5)
  let A' : ℝ × ℝ := (21 / 10, 0)
  let B' : ℝ × ℝ := (21 / 10, 5)
  let T₀ : ℝ × ℝ := (0, 51 / 10)
  let T₁ : ℝ × ℝ := (2, 51 / 10)
  let T₂ : ℝ × ℝ := (21 / 10, 51 / 10)
  let mkT (u v w : ℝ × ℝ) : DevTriangle :=
    ⟨u, v, w, periodicDev u, periodicDev v, periodicDev w⟩
  [ mkT A B C, mkT A D C, mkT B E C,
    mkT A' B' C, mkT A' D C, mkT B' E C,
    mkT B E T₁, mkT B T₁ T₀,
    mkT E B' T₂, mkT E T₂ T₁ ]

/-- **Proposition 1 (Boundary periodicity of vertex deviations)**: Opposite edges of the period
box `[0, 21/10] × [0, 51/10]` carry identical vertex deviations. -/
theorem periodic_mesh_boundary_continuity :
    (periodicDev (0, 0) = periodicDev (21 / 10, 0) ∧
     periodicDev (0, 5) = periodicDev (21 / 10, 5) ∧
     periodicDev (0, 51 / 10) = periodicDev (21 / 10, 51 / 10)) ∧
    (periodicDev (0, 0) = periodicDev (0, 51 / 10) ∧
     periodicDev (2, 0) = periodicDev (2, 51 / 10) ∧
     periodicDev (21 / 10, 0) = periodicDev (21 / 10, 51 / 10)) := by
  norm_num [periodicDev]

/-- **Proposition 1 (Total area and density)**: The 10 triangles tile the period box
`[0, 21/10] × [0, 51/10]` of exact area `1071 / 100`, giving density `1000 / 1071`. -/
theorem periodic_mesh_total_area :
    periodicTriangles.length = 10 ∧
    (periodicTriangles.map fun T => triArea T.v₀ T.v₁ T.v₂).sum = 1071 / 100 ∧
    (periodicTriangles.length : ℝ) / (1071 / 100) = 1000 / 1071 := by
  refine ⟨rfl, ?_, by norm_num [periodicTriangles]⟩
  simp only [periodicTriangles, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, triArea]
  norm_num

/-- **Proposition 1 (Oversized face `5ε > 8ε / √3`)**: Triangle `ABC` in `periodicTriangles`
has exact area `5`, and for every `ε > 0`, `8ε / √3 < 5ε`. -/
theorem periodic_mesh_oversized_face {ε : ℝ} (hε : 0 < ε) :
    triArea (0, 0) (0, 5) (2, 5 / 2) = 5 ∧ 8 * ε / Real.sqrt 3 < 5 * ε := by
  constructor
  · simp only [triArea]; norm_num
  · have hs3 : 0 < Real.sqrt 3 := Real.sqrt_pos.mpr (by norm_num)
    rw [div_lt_iff₀ hs3]
    have h3 : (8 : ℝ) < 5 * Real.sqrt 3 := by
      have hsq : (8 : ℝ) ^ 2 < (5 * Real.sqrt 3) ^ 2 := by
        rw [mul_pow, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 3)]; norm_num
      nlinarith
    nlinarith

/-- **Proposition 1 (Exact global error `169 / 180 < 1` on all 10 faces)**:
Every triangle `T ∈ periodicTriangles` satisfies `|f - F| ≤ 169 / 180 < 1` everywhere on `T`,
and `169 / 180` is attained on edge `AC` at `t = 17 / 30`. -/
theorem periodic_mesh_global_error_le :
    (169 : ℝ) / 180 < 1 ∧
    edgeError (-2 / 3) 0 5 (17 / 30) = 169 / 180 ∧
    ∀ T ∈ periodicTriangles, ∀ l₀ l₁ l₂ : ℝ,
      0 ≤ l₀ → 0 ≤ l₁ → 0 ≤ l₂ → l₀ + l₁ + l₂ = 1 →
      |affineInterp T.v₀ T.v₁ T.v₂ T.d₀ T.d₁ T.d₂ l₀ l₁ l₂ -
        F (baryPoint T.v₀ T.v₁ T.v₂ l₀ l₁ l₂)| ≤ 169 / 180 := by
  refine ⟨by norm_num, by simp [edgeError]; norm_num, ?_⟩
  -- Reusable bound for any edge with `|d₀|, |d₁| ≤ 2/3` and `|e| ≤ 1/2`.
  have hsmall : ∀ d₀ d₁ e : ℝ, |d₀| ≤ 2 / 3 → |d₁| ≤ 2 / 3 → |e| ≤ 1 / 2 →
      ∀ t ∈ Set.Icc (0 : ℝ) 1, |edgeError d₀ d₁ e t| ≤ 169 / 180 := by
    intro d₀ d₁ e h0 h1 he t ⟨ht0, ht1⟩
    have h0l := (abs_le.mp h0).1; have h0h := (abs_le.mp h0).2
    have h1l := (abs_le.mp h1).1; have h1h := (abs_le.mp h1).2
    have hel := (abs_le.mp he).1; have heh := (abs_le.mp he).2
    have hp1 : 0 ≤ t * (1 - t) := mul_nonneg ht0 (by linarith)
    have hp2 : t * (1 - t) ≤ 1 / 4 := by nlinarith [sq_nonneg (t - 1 / 2)]
    simp only [edgeError]
    rw [abs_le]; constructor <;> nlinarith
  -- Bound for the two active edges `(d₀ = -2/3, d₁ = 0, e = 5)` and `(d₀ = 2/3, d₁ = 0, e = -5)`.
  have hAC : ∀ t ∈ Set.Icc (0 : ℝ) 1, |edgeError (-2 / 3) 0 5 t| ≤ 169 / 180 := by
    intro t ⟨ht0, ht1⟩
    simp only [edgeError]
    rw [abs_le]
    constructor
    · nlinarith [mul_nonneg ht0 (sub_nonneg.mpr ht1)]
    · have hsq : 0 ≤ 5 * (t - 17 / 30) ^ 2 := by positivity
      linarith
  have hCA : ∀ t ∈ Set.Icc (0 : ℝ) 1, |edgeError 0 (-2 / 3) 5 t| ≤ 169 / 180 := by
    intro t ⟨ht0, ht1⟩
    have h := hAC (1 - t) ⟨by linarith, by linarith⟩
    have hEq : edgeError 0 (-2 / 3) 5 t = edgeError (-2 / 3) 0 5 (1 - t) := by
      simp [edgeError]; ring
    rw [hEq]; exact h
  have hBC : ∀ t ∈ Set.Icc (0 : ℝ) 1, |edgeError (2 / 3) 0 (-5) t| ≤ 169 / 180 := by
    intro t ⟨ht0, ht1⟩
    have h := hAC t ⟨ht0, ht1⟩
    have hEq : edgeError (2 / 3) 0 (-5) t = -edgeError (-2 / 3) 0 5 t := by
      simp [edgeError]; ring
    rw [hEq, abs_neg]; exact h
  have hCB : ∀ t ∈ Set.Icc (0 : ℝ) 1, |edgeError 0 (2 / 3) (-5) t| ≤ 169 / 180 := by
    intro t ⟨ht0, ht1⟩
    have h := hBC (1 - t) ⟨by linarith, by linarith⟩
    have hEq : edgeError 0 (2 / 3) (-5) t = edgeError (2 / 3) 0 (-5) (1 - t) := by
      simp [edgeError]; ring
    rw [hEq]; exact h
  intro T hT l₀ l₁ l₂ hl0 hl1 hl2 hsum
  simp only [periodicTriangles, List.mem_cons, List.not_mem_nil, or_false] at hT
  rcases hT with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · -- ABC
    refine faceError_abs_le_of_edges ?_ ?_ ?_ hl0 hl1 hl2 hsum
    · exact hsmall _ _ _ (by norm_num [periodicDev]) (by norm_num [periodicDev]) (by norm_num [q])
    · convert hBC using 2; norm_num [periodicDev, q]
    · convert hCA using 2; norm_num [periodicDev, q]
  · -- ADC
    refine faceError_abs_le_of_edges ?_ ?_ ?_ hl0 hl1 hl2 hsum
    · exact hsmall _ _ _ (by norm_num [periodicDev]) (by norm_num [periodicDev]) (by norm_num [q])
    · exact hsmall _ _ _ (by norm_num [periodicDev]) (by norm_num [periodicDev]) (by norm_num [q])
    · convert hCA using 2; norm_num [periodicDev, q]
  · -- BEC
    refine faceError_abs_le_of_edges ?_ ?_ ?_ hl0 hl1 hl2 hsum
    · exact hsmall _ _ _ (by norm_num [periodicDev]) (by norm_num [periodicDev]) (by norm_num [q])
    · exact hsmall _ _ _ (by norm_num [periodicDev]) (by norm_num [periodicDev]) (by norm_num [q])
    · convert hCB using 2; norm_num [periodicDev, q]
  · -- A'B'C
    refine faceError_abs_le_of_edges ?_ ?_ ?_ hl0 hl1 hl2 hsum <;>
      exact hsmall _ _ _ (by norm_num [periodicDev]) (by norm_num [periodicDev]) (by norm_num [q])
  · -- A'DC
    refine faceError_abs_le_of_edges ?_ ?_ ?_ hl0 hl1 hl2 hsum <;>
      exact hsmall _ _ _ (by norm_num [periodicDev]) (by norm_num [periodicDev]) (by norm_num [q])
  · -- B'EC
    refine faceError_abs_le_of_edges ?_ ?_ ?_ hl0 hl1 hl2 hsum <;>
      exact hsmall _ _ _ (by norm_num [periodicDev]) (by norm_num [periodicDev]) (by norm_num [q])
  · -- BET₁
    refine faceError_abs_le_of_edges ?_ ?_ ?_ hl0 hl1 hl2 hsum <;>
      exact hsmall _ _ _ (by norm_num [periodicDev]) (by norm_num [periodicDev]) (by norm_num [q])
  · -- BT₁T₀
    refine faceError_abs_le_of_edges ?_ ?_ ?_ hl0 hl1 hl2 hsum <;>
      exact hsmall _ _ _ (by norm_num [periodicDev]) (by norm_num [periodicDev]) (by norm_num [q])
  · -- EB'T₂
    refine faceError_abs_le_of_edges ?_ ?_ ?_ hl0 hl1 hl2 hsum <;>
      exact hsmall _ _ _ (by norm_num [periodicDev]) (by norm_num [periodicDev]) (by norm_num [q])
  · -- ET₂T₁
    refine faceError_abs_le_of_edges ?_ ?_ ?_ hl0 hl1 hl2 hsum <;>
      exact hsmall _ _ _ (by norm_num [periodicDev]) (by norm_num [periodicDev]) (by norm_num [q])

/-! ### 4. Universal Density Improvement over the Discontinuous Constant (Lemma 3 & Theorem 3) -/

/-- The unrestricted maximum triangle area at `ε = 1`: `M = 32√3 / 9`. -/
noncomputable def M_val : ℝ := 32 * Real.sqrt 3 / 9

/-- The large-triangle threshold `B = 123 / 20 = 6.15`. -/
def B_val : ℝ := 123 / 20

/-- The overhang enlargement factor `α = 11 / 10 = 1.1`. -/
def α_val : ℝ := 11 / 10

/-- The discontinuous optimal density constant `3√3 / 32 = 1 / M`. -/
noncomputable def c_disc : ℝ := 3 * Real.sqrt 3 / 32

/-- The improved universal continuous density constant
`(1 / M) * (1 + (1 - B / M) / (1 + 6 * α²))`. -/
noncomputable def c_cont : ℝ :=
  (1 / M_val) * (1 + (1 - B_val / M_val) / (1 + 6 * α_val ^ 2))

lemma M_val_pos : 0 < M_val := by
  simp only [M_val]; positivity

lemma M_val_sq : M_val ^ 2 = 1024 / 27 := by
  simp only [M_val, div_pow, mul_pow]
  rw [Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 3)]
  ring

/-- Polynomial `P(m) = 32 * (1 - m) * (1 + m)²` appearing in the corner-area bound of Lemma 3. -/
def P_poly (m : ℝ) : ℝ := 32 * (1 - m) * (1 + m) ^ 2

lemma P_poly_le_M_sq {m : ℝ} (hm0 : 0 ≤ m) (_hm1 : m ≤ 1) :
    P_poly m ≤ 1024 / 27 := by
  have hfac : 1024 / 27 - P_poly m = 32 / 27 * (3 * m - 1) ^ 2 * (3 * m + 5) := by
    simp only [P_poly]; ring
  have hnn : 0 ≤ 32 / 27 * (3 * m - 1) ^ 2 * (3 * m + 5) := by positivity
  linarith

/-- Core corner-area polynomial identity and upper bound of Lemma 3(1). -/
lemma lemma3_corner_bound {a₀ a₁ a₂ b₁ b₂ : ℝ}
    (h1 : a₁ ^ 2 + b₁ ^ 2 = 1) (h2 : a₂ ^ 2 + b₂ ^ 2 = 1) :
    let p := 2 * (a₀ + a₁) ^ 2
    let r := 2 * (a₀ + a₂) ^ 2
    let w := 2 * (b₁ + b₂) ^ 2
    let m := (a₁ + a₂) / 2
    let h := ((a₁ - a₂) / 2) ^ 2
    let S := 1 - m ^ 2
    ((p - r) ^ 2 + 2 * w * (p + r) + w ^ 2) / 4 ≤
      P_poly m - 16 * S * (h + (1 - a₀) * (1 + a₀ + 2 * m)) := by
  dsimp only
  set m := (a₁ + a₂) / 2
  set H := (a₁ - a₂) / 2
  set h := H ^ 2
  set S := 1 - m ^ 2
  set C := (a₀ + m) ^ 2
  set Q := (b₁ + b₂) ^ 2
  have hQ_nn : 0 ≤ Q := sq_nonneg _
  have hQ_le : Q ≤ 4 * (S - h) := by
    have hcs : Q ≤ 2 * (b₁ ^ 2 + b₂ ^ 2) := by
      simp only [Q]; nlinarith [sq_nonneg (b₁ - b₂)]
    have hSh : 2 * (b₁ ^ 2 + b₂ ^ 2) = 4 * (S - h) := by
      have : b₁ ^ 2 + b₂ ^ 2 = 2 - a₁ ^ 2 - a₂ ^ 2 := by linarith
      rw [this]; simp only [S, h, m, H]; ring
    linarith
  have hcorner : ((2 * (a₀ + a₁) ^ 2 - 2 * (a₀ + a₂) ^ 2) ^ 2 +
      2 * (2 * Q) * (2 * (a₀ + a₁) ^ 2 + 2 * (a₀ + a₂) ^ 2) + (2 * Q) ^ 2) / 4 =
      16 * C * h + Q * (Q + 4 * C + 4 * h) := by
    simp only [C, h, m, H]; ring
  have hmono : 16 * C * h + Q * (Q + 4 * C + 4 * h) ≤
      16 * C * h + (4 * (S - h)) * (4 * (S - h) + 4 * C + 4 * h) := by
    have h4C : 0 ≤ 4 * C + 4 * h := by positivity
    nlinarith
  have halg : 16 * C * h + (4 * (S - h)) * (4 * (S - h) + 4 * C + 4 * h) =
      P_poly m - 16 * S * (h + (1 - a₀) * (1 + a₀ + 2 * m)) := by
    simp only [P_poly, S, C]; ring
  linarith

/-- Helper: boundary cases `p = 0` or `r = 0` of Lemma 3(1) stay strictly below `B = 123/20`. -/
private lemma lemma3_degenerate_corner {a b : ℝ} (hab : a ^ 2 + b ^ 2 = 1) :
    (1 + a) ^ 2 + (1 + b) ^ 2 < B_val := by
  have hcs : a + b ≤ Real.sqrt 2 := by
    have hsq : (a + b) ^ 2 ≤ 2 := by nlinarith [sq_nonneg (a - b)]
    exact Real.le_sqrt_of_sq_le hsq
  have h1 : (1 + a) ^ 2 + (1 + b) ^ 2 = 3 + 2 * (a + b) := by nlinarith
  have hs2 : Real.sqrt 2 < 142 / 100 := by
    rw [Real.sqrt_lt' (by norm_num)]; norm_num
  simp only [B_val]
  linarith

/-- Helper: separate convexity on `[0, Pcap] × [0, Rcap]` for `(x - y)² + 2 W (x + y) + W²`. -/
private lemma quad_box_max {Pcap Rcap Wcap x y : ℝ} (hW : 0 ≤ Wcap)
    (hx0 : 0 ≤ x) (hxP : x ≤ Pcap) (hy0 : 0 ≤ y) (hyR : y ≤ Rcap) :
    (x - y) ^ 2 + 2 * Wcap * (x + y) + Wcap ^ 2 ≤
      max (max (Wcap ^ 2) ((Pcap + Wcap) ^ 2))
          (max ((Rcap + Wcap) ^ 2) ((Pcap - Rcap) ^ 2 + 2 * Wcap * (Pcap + Rcap) + Wcap ^ 2)) := by
  have hconv_quad : ∀ (u M c₀ c₁ : ℝ), 0 ≤ u → u ≤ M →
      u ^ 2 + c₁ * u + c₀ ≤ max c₀ (M ^ 2 + c₁ * M + c₀) := by
    intro u M c₀ c₁ hu0 huM
    by_cases hM : M = 0
    · have hu : u = 0 := by linarith
      simp [hu]
    · have hMp : 0 < M := lt_of_le_of_ne (by linarith) (Ne.symm hM)
      have hlin : u ^ 2 + c₁ * u + c₀ ≤ (1 - u / M) * c₀ + (u / M) * (M ^ 2 + c₁ * M + c₀) := by
        have hdiff : (1 - u / M) * c₀ + (u / M) * (M ^ 2 + c₁ * M + c₀) - (u ^ 2 + c₁ * u + c₀) =
            u * (M - u) := by field_simp [ne_of_gt hMp]; ring
        have hnn : 0 ≤ u * (M - u) := mul_nonneg hu0 (sub_nonneg.mpr huM)
        linarith
      have h1 : 0 ≤ 1 - u / M := by rw [sub_nonneg, div_le_one hMp]; exact huM
      have h2 : 0 ≤ u / M := div_nonneg hu0 (le_of_lt hMp)
      have hm1 := le_max_left c₀ (M ^ 2 + c₁ * M + c₀)
      have hm2 := le_max_right c₀ (M ^ 2 + c₁ * M + c₀)
      nlinarith
  have hx_step : (x - y) ^ 2 + 2 * Wcap * (x + y) + Wcap ^ 2 ≤
      max ((y + Wcap) ^ 2) ((Pcap - y) ^ 2 + 2 * Wcap * (Pcap + y) + Wcap ^ 2) := by
    have h1 := hconv_quad x Pcap (y ^ 2 + 2 * Wcap * y + Wcap ^ 2) (2 * Wcap - 2 * y) hx0 hxP
    have he1 : x ^ 2 + (2 * Wcap - 2 * y) * x + (y ^ 2 + 2 * Wcap * y + Wcap ^ 2) =
        (x - y) ^ 2 + 2 * Wcap * (x + y) + Wcap ^ 2 := by ring
    have he2 : y ^ 2 + 2 * Wcap * y + Wcap ^ 2 = (y + Wcap) ^ 2 := by ring
    have he3 : Pcap ^ 2 + (2 * Wcap - 2 * y) * Pcap + (y ^ 2 + 2 * Wcap * y + Wcap ^ 2) =
        (Pcap - y) ^ 2 + 2 * Wcap * (Pcap + y) + Wcap ^ 2 := by ring
    rw [he1, he3, he2] at h1
    exact h1
  have hy_left : (y + Wcap) ^ 2 ≤ max (Wcap ^ 2) ((Rcap + Wcap) ^ 2) := by
    have : (y + Wcap) ^ 2 ≤ (Rcap + Wcap) ^ 2 := by nlinarith
    exact le_trans this (le_max_right _ _)
  have hy_right : (Pcap - y) ^ 2 + 2 * Wcap * (Pcap + y) + Wcap ^ 2 ≤
      max ((Pcap + Wcap) ^ 2) ((Pcap - Rcap) ^ 2 + 2 * Wcap * (Pcap + Rcap) + Wcap ^ 2) := by
    have h1 := hconv_quad y Rcap (Pcap ^ 2 + 2 * Wcap * Pcap + Wcap ^ 2) (2 * Wcap - 2 * Pcap) hy0 hyR
    have he1 : y ^ 2 + (2 * Wcap - 2 * Pcap) * y + (Pcap ^ 2 + 2 * Wcap * Pcap + Wcap ^ 2) =
        (Pcap - y) ^ 2 + 2 * Wcap * (Pcap + y) + Wcap ^ 2 := by ring
    have he2 : Pcap ^ 2 + 2 * Wcap * Pcap + Wcap ^ 2 = (Pcap + Wcap) ^ 2 := by ring
    have he3 : Rcap ^ 2 + (2 * Wcap - 2 * Pcap) * Rcap + (Pcap ^ 2 + 2 * Wcap * Pcap + Wcap ^ 2) =
        (Pcap - Rcap) ^ 2 + 2 * Wcap * (Pcap + Rcap) + Wcap ^ 2 := by ring
    rw [he1, he3, he2] at h1
    exact h1
  have hmax1 : max (Wcap ^ 2) ((Rcap + Wcap) ^ 2) ≤
      max (max (Wcap ^ 2) ((Pcap + Wcap) ^ 2))
          (max ((Rcap + Wcap) ^ 2) ((Pcap - Rcap) ^ 2 + 2 * Wcap * (Pcap + Rcap) + Wcap ^ 2)) :=
    max_le (le_trans (le_max_left _ _) (le_max_left _ _))
           (le_trans (le_max_left _ _) (le_max_right _ _))
  have hmax2 : max ((Pcap + Wcap) ^ 2) ((Pcap - Rcap) ^ 2 + 2 * Wcap * (Pcap + Rcap) + Wcap ^ 2) ≤
      max (max (Wcap ^ 2) ((Pcap + Wcap) ^ 2))
          (max ((Rcap + Wcap) ^ 2) ((Pcap - Rcap) ^ 2 + 2 * Wcap * (Pcap + Rcap) + Wcap ^ 2)) :=
    max_le (le_trans (le_max_right _ _) (le_max_left _ _))
           (le_trans (le_max_right _ _) (le_max_right _ _))
  exact le_trans hx_step (max_le (le_trans hy_left hmax1) (le_trans hy_right hmax2))

private lemma lemma3_stability_from_active {a₀ a₁ a₂ : ℝ}
    (ha0 : 0 ≤ a₀) (ha0_le : a₀ ≤ 1)
    (ha1 : 0 ≤ a₁) (ha1_le : a₁ ≤ 1)
    (ha2 : 0 ≤ a₂) (ha2_le : a₂ ≤ 1)
    (hactive_gt : B_val ^ 2 < P_poly ((a₁ + a₂) / 2) -
      16 * (1 - ((a₁ + a₂) / 2) ^ 2) *
        (((a₁ - a₂) / 2) ^ 2 + (1 - a₀) * (1 + a₀ + 2 * ((a₁ + a₂) / 2)))) :
    99 / 100 < a₀ ∧ 20 / 100 < a₁ ∧ a₁ < 48 / 100 ∧ 20 / 100 < a₂ ∧ a₂ < 48 / 100 := by
  set m := (a₁ + a₂) / 2
  set h := ((a₁ - a₂) / 2) ^ 2
  set S := 1 - m ^ 2
  have hm0 : 0 ≤ m := by simp only [m]; linarith only [ha1, ha2]
  have hm1 : m ≤ 1 := by simp only [m]; linarith only [ha1_le, ha2_le]
  have hS0 : 0 ≤ S := by simp only [S]; nlinarith only [hm0, hm1]
  have hP_le := P_poly_le_M_sq hm0 hm1
  have hrem_nn : 0 ≤ 16 * S * (h + (1 - a₀) * (1 + a₀ + 2 * m)) := by
    have : 0 ≤ (1 - a₀) * (1 + a₀ + 2 * m) := mul_nonneg (by linarith only [ha0_le]) (by linarith only [ha0, hm0])
    have : 0 ≤ h := sq_nonneg _
    positivity
  have hPm_gt : B_val ^ 2 < P_poly m := by linarith only [hactive_gt, hrem_nn]
  have hm_lo : 29 / 100 < m := by
    by_contra hle; push_neg at hle
    have hP29 : P_poly (29 / 100) < B_val ^ 2 := by simp only [P_poly, B_val]; norm_num
    have hmono29 : P_poly m ≤ P_poly (29 / 100) := by
      have hfac : P_poly (29 / 100) - P_poly m =
          32 * (29 / 100 - m) * (1 - (29 / 100) - m - (29 / 100) ^ 2 - (29 / 100) * m - m ^ 2) := by
        simp only [P_poly]; ring
      have hpos_f : 0 ≤ 1 - (29 / 100) - m - (29 / 100) ^ 2 - (29 / 100) * m - m ^ 2 := by
        nlinarith only [hm0, hle]
      nlinarith only [hfac, hle, hpos_f]
    linarith only [hPm_gt, hP29, hmono29]
  have hm_hi : m < 39 / 100 := by
    by_contra hge; push_neg at hge
    have hP39 : P_poly (39 / 100) < B_val ^ 2 := by simp only [P_poly, B_val]; norm_num
    have hmono39 : P_poly m ≤ P_poly (39 / 100) := by
      have hfac : P_poly (39 / 100) - P_poly m =
          32 * (m - 39 / 100) * (m ^ 2 + (39 / 100) * m + (39 / 100) ^ 2 + m + 39 / 100 - 1) := by
        simp only [P_poly]; ring
      have hpos_f : 0 ≤ m ^ 2 + (39 / 100) * m + (39 / 100) ^ 2 + m + 39 / 100 - 1 := by
        nlinarith only [hge]
      nlinarith only [hfac, hge, hpos_f]
    linarith only [hPm_gt, hP39, hmono39]
  have hS_gt : 84 / 100 < S := by simp only [S]; nlinarith only [hm0, hm_hi]
  have hgap : 16 * S * (h + (1 - a₀) * (1 + a₀ + 2 * m)) < 104 / 1000 := by
    have : (1024 : ℝ) / 27 - B_val ^ 2 < 104 / 1000 := by simp only [B_val]; norm_num
    linarith only [hactive_gt, hP_le, this]
  have hterm_lt : h + (1 - a₀) * (1 + a₀ + 2 * m) < 104 / 1000 / (16 * (84 / 100)) := by
    have hnn : 0 ≤ h + (1 - a₀) * (1 + a₀ + 2 * m) := by
      have : 0 ≤ (1 - a₀) * (1 + a₀ + 2 * m) := mul_nonneg (by linarith only [ha0_le]) (by linarith only [ha0, hm0])
      have : 0 ≤ h := sq_nonneg _
      linarith only [this, ‹0 ≤ (1 - a₀) * (1 + a₀ + 2 * m)›]
    have h1 : 16 * (84 / 100) * (h + (1 - a₀) * (1 + a₀ + 2 * m)) ≤
        16 * S * (h + (1 - a₀) * (1 + a₀ + 2 * m)) := by
      nlinarith only [hS_gt, hnn]
    rw [lt_div_iff₀ (by norm_num : (0 : ℝ) < 16 * (84 / 100))]
    linarith only [h1, hgap]
  have hh_lt : h < (9 / 100) ^ 2 := by
    have : 0 ≤ (1 - a₀) * (1 + a₀ + 2 * m) := mul_nonneg (by linarith only [ha0_le]) (by linarith only [ha0, hm0])
    have hnum : (104 : ℝ) / 1000 / (16 * (84 / 100)) < (9 / 100) ^ 2 := by norm_num
    linarith only [hterm_lt, this, hnum]
  have ha0_gt : 99 / 100 < a₀ := by
    have h1 : 1 - a₀ ≤ (1 - a₀) * (1 + a₀ + 2 * m) := by nlinarith only [ha0, ha0_le, hm0]
    have h2 : 0 ≤ h := sq_nonneg _
    have h3 : (104 : ℝ) / 1000 / (16 * (84 / 100)) < 1 / 100 := by norm_num
    linarith only [hterm_lt, h1, h2, h3]
  have hH_abs : |(a₁ - a₂) / 2| < 9 / 100 := by
    have : ((a₁ - a₂) / 2) ^ 2 < (9 / 100) ^ 2 := hh_lt
    rw [abs_lt]; constructor <;> nlinarith only [this]
  have ha1_lo : 20 / 100 < a₁ := by simp only [m] at hm_lo; linarith only [hm_lo, (abs_lt.mp hH_abs).1]
  have ha1_hi : a₁ < 48 / 100 := by simp only [m] at hm_hi; linarith only [hm_hi, (abs_lt.mp hH_abs).2]
  have ha2_lo : 20 / 100 < a₂ := by simp only [m] at hm_lo; linarith only [hm_lo, (abs_lt.mp hH_abs).2]
  have ha2_hi : a₂ < 48 / 100 := by simp only [m] at hm_hi; linarith only [hm_hi, (abs_lt.mp hH_abs).1]
  exact ⟨ha0_gt, ha1_lo, ha1_hi, ha2_lo, ha2_hi⟩

private lemma lemma3_cap_eq {u v : ℝ} (hu : 0 ≤ u) (hv : 0 ≤ v) :
    2 * (Real.sqrt (u / 2) + Real.sqrt (v / 2)) ^ 2 = (Real.sqrt u + Real.sqrt v) ^ 2 := by
  have hs : ∀ x : ℝ, 0 ≤ x → Real.sqrt 2 * Real.sqrt (x / 2) = Real.sqrt x := fun x hx => by
    rw [← Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 2)]; congr 1; ring
  have h1 : Real.sqrt 2 * (Real.sqrt (u / 2) + Real.sqrt (v / 2)) = Real.sqrt u + Real.sqrt v := by
    rw [mul_add, hs u hu, hs v hv]
  rw [← h1, mul_pow, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)]

private lemma lemma3_max_lt_w {p r w A : ℝ} (hp0 : 0 ≤ p) (hr0 : 0 ≤ r) (hw0 : 0 ≤ w)
    (hp_lt : p < 9 / 2) (hr_lt : r < 9 / 2)
    (harea : 4 * A ^ 2 = (p - r) ^ 2 + 2 * w * (p + r) + w ^ 2)
    (h4B_lt : 4 * B_val ^ 2 < 4 * A ^ 2) :
    max p r < w := by
  by_contra hle; push_neg at hle
  have hw_lt : w < 9 / 2 := lt_of_le_of_lt hle (max_lt hp_lt hr_lt)
  have hpr_sq : (p - r) ^ 2 ≤ (9 / 2) ^ 2 := by nlinarith only [hp0, hr0, hp_lt, hr_lt]
  have hsum_le : (p - r) ^ 2 + 2 * w * (p + r) + w ^ 2 ≤ 6 * (9 / 2) ^ 2 := by
    nlinarith only [hpr_sq, hp0, hr0, hw0, hp_lt, hr_lt, hw_lt]
  have h6_lt : 6 * (9 / 2 : ℝ) ^ 2 < 4 * B_val ^ 2 := by simp only [B_val]; norm_num
  linarith only [harea, h4B_lt, hsum_le, h6_lt]

/-- **Lemma 3(1) (Unrestricted area bound & quantitative stability)**:
For any valid mixed-sign triangle at `ε = 1` with `p, r ≥ 0`, `w ≥ 0`, `|dᵢ| ≤ 1`,
`p ≤ (√(1 - d₀) + √(1 - d₁))²`, `r ≤ (√(1 - d₀) + √(1 - d₂))²`, `w ≤ (√(1 + d₁) + √(1 + d₂))²`,
and `4 A² = (p - r)² + 2 w (p + r) + w²` (`A ≥ 0`):
1. `A ≤ M_val = 32√3 / 9`.
2. If `A > B_val = 123 / 20`, then `d₀ < -96/100`, `1/2 < d₁ < 92/100`, `1/2 < d₂ < 92/100`,
   and `max p r < w`. -/
theorem large_triangle_stability {d₀ d₁ d₂ p r w A : ℝ}
    (hd0 : |d₀| ≤ 1) (hd1 : |d₁| ≤ 1) (hd2 : |d₂| ≤ 1)
    (hp0 : 0 ≤ p) (hr0 : 0 ≤ r) (hw0 : 0 ≤ w) (hA0 : 0 ≤ A)
    (hp : p ≤ (Real.sqrt (1 - d₀) + Real.sqrt (1 - d₁)) ^ 2)
    (hr : r ≤ (Real.sqrt (1 - d₀) + Real.sqrt (1 - d₂)) ^ 2)
    (hw : w ≤ (Real.sqrt (1 + d₁) + Real.sqrt (1 + d₂)) ^ 2)
    (harea : 4 * A ^ 2 = (p - r) ^ 2 + 2 * w * (p + r) + w ^ 2) :
    A ≤ M_val ∧
    (B_val < A →
      d₀ < -96 / 100 ∧ 1 / 2 < d₁ ∧ d₁ < 92 / 100 ∧ 1 / 2 < d₂ ∧ d₂ < 92 / 100 ∧
      max p r < w) := by
  have hd0l := (abs_le.mp hd0).1; have hd0h := (abs_le.mp hd0).2
  have hd1l := (abs_le.mp hd1).1; have hd1h := (abs_le.mp hd1).2
  have hd2l := (abs_le.mp hd2).1; have hd2h := (abs_le.mp hd2).2
  set a₀ := Real.sqrt ((1 - d₀) / 2)
  set a₁ := Real.sqrt ((1 - d₁) / 2)
  set a₂ := Real.sqrt ((1 - d₂) / 2)
  set b₁ := Real.sqrt ((1 + d₁) / 2)
  set b₂ := Real.sqrt ((1 + d₂) / 2)
  have ha0 : 0 ≤ a₀ := Real.sqrt_nonneg _
  have ha1 : 0 ≤ a₁ := Real.sqrt_nonneg _
  have ha2 : 0 ≤ a₂ := Real.sqrt_nonneg _
  have hb1 : 0 ≤ b₁ := Real.sqrt_nonneg _
  have hb2 : 0 ≤ b₂ := Real.sqrt_nonneg _
  have ha0sq : a₀ ^ 2 = (1 - d₀) / 2 := Real.sq_sqrt (by linarith)
  have ha1sq : a₁ ^ 2 = (1 - d₁) / 2 := Real.sq_sqrt (by linarith)
  have ha2sq : a₂ ^ 2 = (1 - d₂) / 2 := Real.sq_sqrt (by linarith)
  have hb1sq : b₁ ^ 2 = (1 + d₁) / 2 := Real.sq_sqrt (by linarith)
  have hb2sq : b₂ ^ 2 = (1 + d₂) / 2 := Real.sq_sqrt (by linarith)
  have ha0_le : a₀ ≤ 1 := Real.sqrt_le_one.mpr (by linarith)
  have ha1_le : a₁ ≤ 1 := Real.sqrt_le_one.mpr (by linarith)
  have ha2_le : a₂ ≤ 1 := Real.sqrt_le_one.mpr (by linarith)
  have hb1_le : b₁ ≤ 1 := Real.sqrt_le_one.mpr (by linarith)
  have hb2_le : b₂ ≤ 1 := Real.sqrt_le_one.mpr (by linarith)
  have h1id : a₁ ^ 2 + b₁ ^ 2 = 1 := by linarith
  have h2id : a₂ ^ 2 + b₂ ^ 2 = 1 := by linarith
  -- Rewrite the edge caps in terms of `aᵢ, bᵢ`.
  set Pcap := 2 * (a₀ + a₁) ^ 2
  set Rcap := 2 * (a₀ + a₂) ^ 2
  set Wcap := 2 * (b₁ + b₂) ^ 2
  have hPcap : Pcap = (Real.sqrt (1 - d₀) + Real.sqrt (1 - d₁)) ^ 2 :=
    lemma3_cap_eq (by linarith only [hd0h]) (by linarith only [hd1h])
  have hRcap : Rcap = (Real.sqrt (1 - d₀) + Real.sqrt (1 - d₂)) ^ 2 :=
    lemma3_cap_eq (by linarith only [hd0h]) (by linarith only [hd2h])
  have hWcap : Wcap = (Real.sqrt (1 + d₁) + Real.sqrt (1 + d₂)) ^ 2 :=
    lemma3_cap_eq (by linarith only [hd1l]) (by linarith only [hd2l])
  rw [← hPcap] at hp
  rw [← hRcap] at hr
  rw [← hWcap] at hw
  -- Separate convexity in `(p, r)` and monotonicity in `w`:
  -- `4 A² ≤ max` over the 4 corners `(0,0), (Pcap,0), (0,Rcap), (Pcap,Rcap)` at `w = Wcap`.
  have hf_le : ∀ x y : ℝ, 0 ≤ x → x ≤ Pcap → 0 ≤ y → y ≤ Rcap →
      (x - y) ^ 2 + 2 * Wcap * (x + y) + Wcap ^ 2 ≤
        max (max (Wcap ^ 2) ((Pcap + Wcap) ^ 2))
            (max ((Rcap + Wcap) ^ 2) ((Pcap - Rcap) ^ 2 + 2 * Wcap * (Pcap + Rcap) + Wcap ^ 2)) :=
    fun x y hx0 hxP hy0 hyR => quad_box_max (by positivity) hx0 hxP hy0 hyR
  -- Bounds on the 3 degenerate corners: all strictly `< 4 * B_val²`.
  have hdeg0 : Wcap ^ 2 < 4 * B_val ^ 2 := by
    have hW8 : Wcap ≤ 8 := by simp only [Wcap]; nlinarith only [hb1, hb2, hb1_le, hb2_le]
    have hW0 : 0 ≤ Wcap := by positivity
    simp only [B_val]; nlinarith only [hW8, hW0]
  have hdegP : (Pcap + Wcap) ^ 2 < 4 * B_val ^ 2 := by
    have hPW : Pcap + Wcap ≤ 2 * ((1 + a₁) ^ 2 + (1 + b₁) ^ 2) := by
      simp only [Pcap, Wcap]; nlinarith only [ha0, ha1, ha0_le, hb1, hb2, hb2_le]
    have hlt := lemma3_degenerate_corner h1id
    have hnn : 0 ≤ Pcap + Wcap := by positivity
    nlinarith only [hPW, hlt, hnn]
  have hdegR : (Rcap + Wcap) ^ 2 < 4 * B_val ^ 2 := by
    have hRW : Rcap + Wcap ≤ 2 * ((1 + a₂) ^ 2 + (1 + b₂) ^ 2) := by
      simp only [Rcap, Wcap]; nlinarith only [ha0, ha2, ha0_le, hb1, hb2, hb1_le]
    have hlt := lemma3_degenerate_corner h2id
    have hnn : 0 ≤ Rcap + Wcap := by positivity
    nlinarith only [hRW, hlt, hnn]
  -- Bound on the active corner via `lemma3_corner_bound`.
  set m := (a₁ + a₂) / 2
  set h := ((a₁ - a₂) / 2) ^ 2
  set S := 1 - m ^ 2
  have hm0 : 0 ≤ m := by simp only [m]; linarith only [ha1, ha2]
  have hm1 : m ≤ 1 := by simp only [m]; linarith only [ha1_le, ha2_le]
  have hS0 : 0 ≤ S := by simp only [S]; nlinarith only [hm0, hm1]
  have hactive : ((Pcap - Rcap) ^ 2 + 2 * Wcap * (Pcap + Rcap) + Wcap ^ 2) / 4 ≤
      P_poly m - 16 * S * (h + (1 - a₀) * (1 + a₀ + 2 * m)) :=
    lemma3_corner_bound (a₀ := a₀) h1id h2id
  have hP_le := P_poly_le_M_sq hm0 hm1
  have h4A_le : 4 * A ^ 2 ≤ (p - r) ^ 2 + 2 * Wcap * (p + r) + Wcap ^ 2 := by
    rw [harea]; nlinarith only [hp0, hr0, hw0, hw]
  have h4A_max := le_trans h4A_le (hf_le p r hp0 hp hr0 hr)
  -- First conclusion: `A ≤ M_val`.
  have hA_le_M : A ≤ M_val := by
    have hB_lt_M : 4 * B_val ^ 2 < 4 * (1024 / 27) := by simp only [B_val]; norm_num
    have h4A_M : 4 * A ^ 2 ≤ 4 * (1024 / 27) := by
      refine le_trans h4A_max (max_le
        (max_le (by linarith only [hdeg0, hB_lt_M]) (by linarith only [hdegP, hB_lt_M]))
        (max_le (by linarith only [hdegR, hB_lt_M]) ?_))
      have hsub_nn : 0 ≤ 16 * S * (h + (1 - a₀) * (1 + a₀ + 2 * m)) := by
        have h1 : 0 ≤ (1 - a₀) * (1 + a₀ + 2 * m) := mul_nonneg (by linarith only [ha0_le]) (by linarith only [ha0, hm0])
        have h2 : 0 ≤ h := sq_nonneg _
        positivity
      linarith only [hactive, hP_le, hsub_nn]
    have hAsq : A ^ 2 ≤ M_val ^ 2 := by rw [M_val_sq]; linarith only [h4A_M]
    have habs : |A| ≤ |M_val| := sq_le_sq.mp hAsq
    rwa [abs_of_nonneg hA0, abs_of_pos M_val_pos] at habs
  refine ⟨hA_le_M, fun hAB => ?_⟩
  -- Since `A > B_val`, the 3 degenerate corners are `< 4 * B_val² < 4 * A²`, so the active corner dominates!
  have h4B_lt : 4 * B_val ^ 2 < 4 * A ^ 2 := by
    have hBpos : 0 < B_val := by norm_num [B_val]
    nlinarith only [hAB, hBpos]
  have hactive_gt : B_val ^ 2 < P_poly m - 16 * S * (h + (1 - a₀) * (1 + a₀ + 2 * m)) := by
    have h4c : 4 * B_val ^ 2 < (Pcap - Rcap) ^ 2 + 2 * Wcap * (Pcap + Rcap) + Wcap ^ 2 := by
      have hlt_all := lt_of_lt_of_le h4B_lt h4A_max
      rcases lt_max_iff.mp hlt_all with hL | hR
      · rcases lt_max_iff.mp hL with h1 | h2
        · linarith only [h1, hdeg0]
        · linarith only [h2, hdegP]
      · rcases lt_max_iff.mp hR with h1 | h2
        · linarith only [h1, hdegR]
        · exact h2
    linarith only [h4c, hactive]
  obtain ⟨ha0_gt, ha1_lo, ha1_hi, ha2_lo, ha2_hi⟩ :=
    lemma3_stability_from_active ha0 ha0_le ha1 ha1_le ha2 ha2_le hactive_gt
  have hd0_lt : d₀ < -96 / 100 := by
    have hsq : (99 / 100 : ℝ) ^ 2 < a₀ ^ 2 := by nlinarith only [ha0_gt]
    linarith only [ha0sq, hsq]
  have hd1_lo : 1 / 2 < d₁ := by
    have hsq : a₁ ^ 2 < (48 / 100 : ℝ) ^ 2 := by nlinarith only [ha1, ha1_hi]
    linarith only [ha1sq, hsq]
  have hd1_hi : d₁ < 92 / 100 := by
    have hsq : (20 / 100 : ℝ) ^ 2 < a₁ ^ 2 := by nlinarith only [ha1_lo]
    linarith only [ha1sq, hsq]
  have hd2_lo : 1 / 2 < d₂ := by
    have hsq : a₂ ^ 2 < (48 / 100 : ℝ) ^ 2 := by nlinarith only [ha2, ha2_hi]
    linarith only [ha2sq, hsq]
  have hd2_hi : d₂ < 92 / 100 := by
    have hsq : (20 / 100 : ℝ) ^ 2 < a₂ ^ 2 := by nlinarith only [ha2_lo]
    linarith only [ha2sq, hsq]
  -- Finally prove `max p r < w`!
  have hp_lt : p < 9 / 2 := by
    have hsum : a₀ + a₁ < 3 / 2 := by linarith only [ha0_le, ha1_hi]
    have hnn : 0 ≤ a₀ + a₁ := by linarith only [ha0, ha1]
    have hsq : 2 * (a₀ + a₁) ^ 2 < 9 / 2 := by nlinarith only [hsum, hnn]
    linarith only [hp, hsq]
  have hr_lt : r < 9 / 2 := by
    have hsum : a₀ + a₂ < 3 / 2 := by linarith only [ha0_le, ha2_hi]
    have hnn : 0 ≤ a₀ + a₂ := by linarith only [ha0, ha2]
    have hsq : 2 * (a₀ + a₂) ^ 2 < 9 / 2 := by nlinarith only [hsum, hnn]
    linarith only [hr, hsq]
  have hw_gt : max p r < w := lemma3_max_lt_w hp0 hr0 hw0 hp_lt hr_lt harea h4B_lt
  exact ⟨hd0_lt, hd1_lo, hd1_hi, hd2_lo, hd2_hi, hw_gt⟩

/-- **Lemma 3(2) (Vertex obstruction — resolving Remark `lem:no-optimal-tiling`)**:
No four alternating-sign spokes `p₁ = (u, v)`, `p₂ = (-s, -t)` (`u, v, s, t > 0`),
`n₁ = (h, -g)`, `n₂ = (-h', g')` (`h, g, h', g' > 0`) can satisfy `p_x n_y + p_y n_x < 0`
for all four adjacent pairs `(p, n)`. -/
theorem large_vertex_four_spoke_obstruction {u v s t h g h' g' : ℝ}
    (hu : 0 < u) (_hv : 0 < v) (hs : 0 < s) (_ht : 0 < t)
    (hh : 0 < h) (_hg : 0 < g) (hh' : 0 < h') (_hg' : 0 < g')
    (h11 : u * (-g) + v * h < 0)
    (h12 : u * g' + v * (-h') < 0)
    (h21 : (-s) * (-g) + (-t) * h < 0)
    (h22 : (-s) * g' + (-t) * (-h') < 0) : False := by
  have i1 : v / u < g / h := by
    rw [div_lt_div_iff₀ hu hh]; linarith
  have i2 : g' / h' < v / u := by
    rw [div_lt_div_iff₀ hh' hu]; linarith
  have i3 : g / h < t / s := by
    rw [div_lt_div_iff₀ hh hs]; linarith
  have i4 : t / s < g' / h' := by
    rw [div_lt_div_iff₀ hs hh']; linarith
  linarith

/-- **Theorem 3 (Overhang factor bound)**: `η = 1 - B / M < 1 / 484` and `(1 - 2√η)⁻¹ < 11/10`. -/
theorem overhang_factor_lt_alpha :
    let η := 1 - B_val / M_val
    0 < η ∧ η < 1 / 484 ∧ (1 - 2 * Real.sqrt η)⁻¹ < α_val := by
  dsimp only
  have hMpos : 0 < M_val := M_val_pos
  have hBpos : 0 < B_val := by norm_num [B_val]
  have hB_lt_M : B_val < M_val := by
    have : B_val ^ 2 < M_val ^ 2 := by rw [M_val_sq]; norm_num [B_val]
    nlinarith
  have hη_pos : 0 < 1 - B_val / M_val := by
    rw [sub_pos, div_lt_one hMpos]; exact hB_lt_M
  have hη_lt : 1 - B_val / M_val < 1 / 484 := by
    have hsq : (483 / 484 * M_val) ^ 2 < B_val ^ 2 := by
      rw [mul_pow, M_val_sq]; norm_num [B_val]
    have h1 : 483 / 484 * M_val < B_val := by nlinarith
    have h2 : 483 / 484 < B_val / M_val := (lt_div_iff₀ hMpos).mpr h1
    linarith
  have hsqrt_lt : Real.sqrt (1 - B_val / M_val) < 1 / 22 := by
    rw [Real.sqrt_lt' (by norm_num)]
    linarith
  have hden_pos : 0 < 1 - 2 * Real.sqrt (1 - B_val / M_val) := by linarith
  have hinv_lt : (1 - 2 * Real.sqrt (1 - B_val / M_val))⁻¹ < α_val := by
    rw [inv_lt_comm₀ hden_pos (by norm_num [α_val])]
    simp only [α_val]
    linarith
  exact ⟨hη_pos, hη_lt, hinv_lt⟩

/-- **Theorem 3 (Asymptotic recurrence lemma)**: If `D(αℓ) ≥ (η / (6α²)) * (1 - D(ℓ) / η)` for
all `ℓ > 0` and `D(ℓ) ≤ U` eventually as `ℓ → ∞`, then `U ≥ η / (1 + 6α²)`. -/
theorem limsup_density_recurrence {η α U : ℝ} (hη : 0 < η) (hα : 1 < α)
    {D : ℝ → ℝ}
    (hrec : ∀ ℓ : ℝ, 0 < ℓ → η / (6 * α ^ 2) * (1 - D ℓ / η) ≤ D (α * ℓ))
    (hev : ∀ᶠ ℓ in Filter.atTop, D ℓ ≤ U) :
    η / (1 + 6 * α ^ 2) ≤ U := by
  rw [Filter.eventually_atTop] at hev
  obtain ⟨L, hL⟩ := hev
  set ℓ := max L 1
  have hℓ_pos : 0 < ℓ := lt_of_lt_of_le one_pos (le_max_right L 1)
  have hℓ_ge : L ≤ ℓ := le_max_left L 1
  have hαℓ_ge : L ≤ α * ℓ := by nlinarith
  have hD1 : D ℓ ≤ U := hL ℓ hℓ_ge
  have hD2 : D (α * ℓ) ≤ U := hL (α * ℓ) hαℓ_ge
  have hstep := hrec ℓ hℓ_pos
  have hpos1 : 0 < 6 * α ^ 2 := by positivity
  have hpos2 : 0 < 1 + 6 * α ^ 2 := by positivity
  have h1 : η / (6 * α ^ 2) * (1 - U / η) ≤ U := by
    have : 1 - U / η ≤ 1 - D ℓ / η := by
      have : D ℓ / η ≤ U / η := div_le_div_of_nonneg_right hD1 (le_of_lt hη)
      linarith
    have : η / (6 * α ^ 2) * (1 - U / η) ≤ η / (6 * α ^ 2) * (1 - D ℓ / η) :=
      mul_le_mul_of_nonneg_left this (by positivity)
    linarith
  have hmul : (6 * α ^ 2) * (η / (6 * α ^ 2) * (1 - U / η)) ≤ (6 * α ^ 2) * U :=
    mul_le_mul_of_nonneg_left h1 (le_of_lt hpos1)
  have hsimp : (6 * α ^ 2) * (η / (6 * α ^ 2) * (1 - U / η)) = η - U := by
    field_simp [ne_of_gt hη, ne_of_gt hpos1]
  rw [hsimp] at hmul
  rw [div_le_iff₀ hpos2]
  linarith

/-- **Theorem 3 (Strict improvement over the discontinuous constant `3√3 / 32`)**:
`c_disc = 1 / M_val`, `0.162379 < c_disc < 0.162380`, `c_disc < c_cont`,
`0.162406 < c_cont < 0.162407`, and `c_cont < √3 / 8`. -/
theorem universal_density_constant_strict_improvement :
    c_disc = 1 / M_val ∧
    c_disc < c_cont ∧
    (162379 : ℝ) / 1000000 < c_disc ∧ c_disc < 162380 / 1000000 ∧
    (162406 : ℝ) / 1000000 < c_cont ∧ c_cont < 162407 / 1000000 ∧
    c_cont < Real.sqrt 3 / 8 := by
  have hs3 : 0 < Real.sqrt 3 := Real.sqrt_pos.mpr (by norm_num)
  have hs3_sq : Real.sqrt 3 ^ 2 = 3 := Real.sq_sqrt (by norm_num)
  have hMpos : 0 < M_val := M_val_pos
  have hdisc_eq : c_disc = 1 / M_val := by
    simp only [c_disc, M_val]
    field_simp [ne_of_gt hs3]
    nlinarith [hs3_sq]
  -- Tight rational bounds on `√3`: `17320508 / 10000000 < √3 < 17320509 / 10000000`.
  have hs3_lo : (17320508 : ℝ) / 10000000 < Real.sqrt 3 := by
    rw [Real.lt_sqrt (by norm_num)]; norm_num
  have hs3_hi : Real.sqrt 3 < 17320509 / 10000000 := by
    rw [Real.sqrt_lt' (by norm_num)]; norm_num
  have hdisc_lo : (162379 : ℝ) / 1000000 < c_disc := by
    simp only [c_disc]; linarith
  have hdisc_hi : c_disc < 162380 / 1000000 := by
    simp only [c_disc]; linarith
  have hη_pos : 0 < 1 - B_val / M_val := overhang_factor_lt_alpha.1
  have hlt : c_disc < c_cont := by
    rw [hdisc_eq]
    simp only [c_cont]
    have hfrac_pos : 0 < (1 - B_val / M_val) / (1 + 6 * α_val ^ 2) :=
      div_pos hη_pos (by norm_num [α_val])
    have hinv_pos : 0 < 1 / M_val := div_pos one_pos hMpos
    nlinarith
  -- Compute `c_cont` linearly in `1 / M_val = c_disc` and `1 / M_val² = 27 / 1024`:
  -- `c_cont = c_disc * (1 + 1 / (1 + 6α²)) - (B_val / (1 + 6α²)) * (27 / 1024)`.
  have hc_cont_eq : c_cont = c_disc * (1 + 1 / (1 + 6 * α_val ^ 2)) -
      B_val / (1 + 6 * α_val ^ 2) * (27 / 1024) := by
    simp only [c_cont]
    have hM2 : (1 / M_val) * (1 / M_val) = 27 / 1024 := by
      have : (1 / M_val) ^ 2 = 27 / 1024 := by
        rw [one_div, inv_pow, M_val_sq]; norm_num
      linarith [sq (1 / M_val)]
    have h1 : (1 / M_val) * (1 + (1 - B_val / M_val) / (1 + 6 * α_val ^ 2)) =
        (1 / M_val) * (1 + 1 / (1 + 6 * α_val ^ 2)) -
        B_val / (1 + 6 * α_val ^ 2) * ((1 / M_val) * (1 / M_val)) := by ring
    rw [h1, hM2, ← hdisc_eq]
  have hcont_lo : (162406 : ℝ) / 1000000 < c_cont := by
    rw [hc_cont_eq]
    simp only [c_disc, B_val, α_val]
    linarith
  have hcont_hi : c_cont < 162407 / 1000000 := by
    rw [hc_cont_eq]
    simp only [c_disc, B_val, α_val]
    linarith
  have hcont_lt_conj : c_cont < Real.sqrt 3 / 8 := by
    linarith
  exact ⟨hdisc_eq, hlt, hdisc_lo, hdisc_hi, hcont_lo, hcont_hi, hcont_lt_conj⟩

end

end PWLProductTriang
