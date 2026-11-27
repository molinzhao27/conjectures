import Mathlib

/-!
# Weak Identifiability of Softmax Multi-Head Attention Networks
(Shahverdi, Marchetti, Bökman, and Kohn, arXiv:2601.21645, Section 5.2)

We formalize in Lean 4 + Mathlib (complete proofs, standard axioms only) the mathematical cores of
both the expanding-width refutation and the constant-width generic identifiability theorem:

1. **Expanding-width `(1, 3, 3)` refutation (Theorem 1 & Proposition 2)**:
   - `expanding133_continuous_non_gauge_family`: At `n ≥ 2`, `(d₀, d₁, d₂) = (1, 3, 3)`,
     `(h₀, h₁, h₂) = (1, 2, 1)`, the continuous family `P₀(t) = diag(1, t, t)`,
     `P₁(t) = diag(t, 1, t)` preserves the end-to-end function IDENTICALLY for all `t : ℝ` and
     all inputs `X : Matrix (Fin n) (Fin 1) ℝ`, yet at `t = 0` vs `t ≠ 0` the layer-2
     constant-token linear maps have incompatible determinant/rank multisets (`{0, 0}` vs
     invertible `{t², t²}`), so NO latent symmetry in `K₁ = GL(3)² ⋊ S₂` relates them.
   - `expanding133_output_span_2d`: For every `n ≥ 2`, the outputs on `X₁ = 1` and `X₂ = e₀`
     have linearly independent rows (nonzero `2 × 2` minor), certifying the 2D output span.
   - `expanding133_equivariance_no_lift`: Proposition 2 — the explicit `(1, 3, 3)` parameter is
     end-to-end equivariant under `g = diag(1, 1, 2)` on outputs (trivial on inputs), yet admits
     NO invertible token-wise hidden lift `M ∈ GL(6)` (in particular no lift in `K₁`).
2. **Single-head two-type probe and exact score determination (§3)**:
   - `attn_const_rows`: Row-wise softmax attention fixes every constant-row matrix.
   - `two_type_probe_formula` & `single_head_determines_score`: For `n ≥ 2`, two-type probes
     (`n - 1` rows `x` and one row `x + v`) yield `x D + σ_n(x E vᵀ) • (v D)` on common rows,
     where `σ_n(s) = (1 + (n - 1) e^{-s})⁻¹` is strictly increasing; hence whenever `D ≠ 0`, the
     single-head map uniquely determines `E`.
   - `opposite_score_sum_linear`: At `n = 2`, `T_A(X) + T_{-A}(X)` outputs the row sum in both
     rows (the exact `n = 2` opposite-score degeneracy).
3. **Constant-width generic identifiability algebraic & analytic core (Theorem 3)**:
   - `block_output_normalization`: Output-coordinate change `Y = Z U` transforms each head via
     `T_A(Z) V P U' = T_{U⁻¹ A U⁻¹ᵀ}(Y) (U⁻¹ V P U')`.
   - `sigmoid_freq_linear_independent`: For every `n ≥ 2` and any finite set `S` of distinct
     strictly positive frequencies `q > 0`, the functions `{1} ∪ {s ↦ σ_n(q s) : q ∈ S}` are
     linearly independent over `ℝ` (proved completely elementarily by asymptotic peeling as
     `s → +∞`).
   - `constant_width_unique_lift`: Step 5 — once normalized blocks `(a_{ij}, b_{ij})` are
     recovered up to a head permutation with invertible `b_{ij} = D_{ij} P_{ij} U_{i+1}` and
     distinct scores `a_{ij}`, the layerwise transporter in `K_i = GL(d)^{h_i} ⋊ S_{h_i}`
     exists (`M_{ij} = D_{ij}⁻¹ D'_{i, π(j)}`), preserves normalized downstream inputs on all
     hidden arrays, and is UNIQUE.

Machine-checked vs prose-only:
- Machine-checked: Complete expanding `(1, 3, 3)` continuous non-gauge family, 2D output-span
  minor, Proposition 2 equivariance-lift refutation, two-type probe formula and score recovery,
  `n = 2` opposite-score cancellation, output-coordinate normalization, real-asymptotic sigmoid
  linear independence, and Step 5 unique `K_i` transporter lift.
- Prose-only (in `solutions/equivariant_mha_ident/proof.pdf`): Real-analyticity / Lebesgue-null
  complement arguments for open-dense genericity in Steps 1, 3, and 4 (inverse function theorem
  for peeling subsequent blocks) and the general-`(h, d)` orthogonal-projector wording of Theorem 1.
-/

open Matrix BigOperators Finset Filter Topology

namespace EquivariantMHAIdent

/-! ### Core Attention Definitions -/

/-- Row-wise softmax of an `n × n` score matrix. -/
noncomputable def softmaxRow {n : ℕ} (S : Matrix (Fin n) (Fin n) ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  fun i j => Real.exp (S i j) / ∑ k : Fin n, Real.exp (S i k)

/-- Single-head pure softmax attention `T_A(Z) = softmaxRow(Z A Zᵀ) * Z`. -/
noncomputable def attnHead {n d : ℕ} (A : Matrix (Fin d) (Fin d) ℝ)
    (Z : Matrix (Fin n) (Fin d) ℝ) : Matrix (Fin n) (Fin d) ℝ :=
  softmaxRow (Z * A * Zᵀ) * Z

/-- Constant-row matrix with every row equal to `x : Fin d → ℝ`. -/
def constRows (n : ℕ) {d : ℕ} (x : Fin d → ℝ) : Matrix (Fin n) (Fin d) ℝ :=
  Matrix.of (fun _ j => x j)

/-- Row-wise softmax of a row-constant matrix has every entry equal to `1 / n` (for `n ≥ 1`). -/
lemma softmaxRow_const {n : ℕ} (hn : 1 ≤ n) (c : Fin n → ℝ) :
    softmaxRow (Matrix.of (fun i (_ : Fin n) => c i)) =
      Matrix.of (fun _ _ => (1 : ℝ) / (n : ℝ)) := by
  ext i j
  simp only [softmaxRow, Matrix.of_apply, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul]
  have hn0 : (n : ℝ) ≠ 0 := by positivity
  have hexp : Real.exp (c i) ≠ 0 := ne_of_gt (Real.exp_pos _)
  field_simp [hn0, hexp]

/-- **Constant-row invariance**: For any `n ≥ 1`, `A`, and `x`, `T_A(constRows n x) = constRows n x`. -/
theorem attn_const_rows {n d : ℕ} (hn : 1 ≤ n) (A : Matrix (Fin d) (Fin d) ℝ) (x : Fin d → ℝ) :
    attnHead A (constRows n x) = constRows n x := by
  unfold attnHead
  have hS : constRows n x * A * (constRows n x)ᵀ =
      Matrix.of (fun _ _ => ∑ a : Fin d, ∑ b : Fin d, x a * A a b * x b) := by
    ext i j
    simp only [Matrix.mul_apply, constRows, Matrix.of_apply, Matrix.transpose_apply]
    simp_rw [Finset.sum_mul]
    rw [Finset.sum_comm]
  rw [hS, softmaxRow_const hn]
  ext i j
  simp only [Matrix.mul_apply, Matrix.of_apply, constRows, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul]
  have hn0 : (n : ℝ) ≠ 0 := by positivity
  field_simp [hn0]

/-- Sigmoid function `σ_n(s) = 1 / (1 + (n - 1) * exp(-s))` governing two-type attention probes. -/
noncomputable def sigmaN (n : ℕ) (s : ℝ) : ℝ :=
  1 / (1 + ((n - 1 : ℕ) : ℝ) * Real.exp (-s))

lemma sigmaN_strictMono {n : ℕ} (hn : 2 ≤ n) : StrictMono (sigmaN n) := by
  intro s₁ s₂ hlt
  unfold sigmaN
  have hm_pos : (0 : ℝ) < ((n - 1 : ℕ) : ℝ) := by
    have : 1 ≤ n - 1 := by omega
    positivity
  have hd1 : 0 < 1 + ((n - 1 : ℕ) : ℝ) * Real.exp (-s₁) := by positivity
  have hd2 : 0 < 1 + ((n - 1 : ℕ) : ℝ) * Real.exp (-s₂) := by positivity
  rw [one_div_lt_one_div hd1 hd2]
  have hexp : Real.exp (-s₂) < Real.exp (-s₁) := Real.exp_lt_exp.mpr (by linarith)
  nlinarith

/-! ### 1. Expanding-Width `(1, 3, 3)` Refutation (Theorem 1 & Proposition 2) -/

/-- Explicit depth-2 `(1, 3, 3)` network with head counts `(1, 2, 1)`:
- Layer 1: `O₁ = [1]`, `A₁₁ = [1]`, `A₁₂ = [2]`, `V₁₁ = e₀`, `V₁₂ = e₁`;
- Layer 2: aggregate blocks `P₀, P₁ : Matrix (Fin 3) (Fin 3) ℝ`, `A₂₁ = I₃`, `V₂₁ = V`. -/
noncomputable def exp133Head1 {n : ℕ} (a : ℝ) (X : Matrix (Fin n) (Fin 1) ℝ) :
    Matrix (Fin n) (Fin 1) ℝ :=
  attnHead (Matrix.of fun _ _ => a) X

/-- Row vector `e_j ∈ ℝ^{1 × 3}`. -/
def rowE3 (j : Fin 3) : Matrix (Fin 1) (Fin 3) ℝ :=
  Matrix.of fun _ k => if k = j then 1 else 0

/-- Hidden representation of layer 1 (pair of `n × 3` head matrices). -/
noncomputable def exp133Layer1 {n : ℕ} (X : Matrix (Fin n) (Fin 1) ℝ) :
    Matrix (Fin n) (Fin 3) ℝ × Matrix (Fin n) (Fin 3) ℝ :=
  (exp133Head1 1 X * rowE3 0, exp133Head1 2 X * rowE3 1)

/-- Layer-2 map parameterized by `P₀, P₁, V : Matrix (Fin 3) (Fin 3) ℝ` and `A₂₁ = I₃`. -/
noncomputable def exp133Layer2 {n : ℕ} (P₀ P₁ V : Matrix (Fin 3) (Fin 3) ℝ)
    (H : Matrix (Fin n) (Fin 3) ℝ × Matrix (Fin n) (Fin 3) ℝ) :
    Matrix (Fin n) (Fin 3) ℝ :=
  attnHead (1 : Matrix (Fin 3) (Fin 3) ℝ) (H.1 * P₀ + H.2 * P₁) * V

/-- Continuous diagonal perturbation blocks `P₀(t) = diag(1, t, t)` and `P₁(t) = diag(t, 1, t)`. -/
def blockP0 (t : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := Matrix.diagonal ![1, t, t]
def blockP1 (t : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := Matrix.diagonal ![t, 1, t]

lemma rowE3_mul_blockP0 (t : ℝ) : rowE3 0 * blockP0 t = rowE3 0 := by
  ext i j
  fin_cases i; fin_cases j <;>
    simp [rowE3, blockP0, Matrix.mul_apply, Matrix.diagonal]

lemma rowE3_mul_blockP1 (t : ℝ) : rowE3 1 * blockP1 t = rowE3 1 := by
  ext i j
  fin_cases i; fin_cases j <;>
    simp [rowE3, blockP1, Matrix.mul_apply, Matrix.diagonal]

/-- A latent symmetry in `K₁ = GL(3)² ⋊ S₂` acts on `(H₀, H₁)` by applying invertible matrices
`G₀, G₁` within each head and optionally swapping the two heads. -/
def K1Layer2Related (n : ℕ) (P₀ P₁ P₀' P₁' : Matrix (Fin 3) (Fin 3) ℝ) : Prop :=
  ∃ (swap : Bool) (G₀ G₁ : Matrix (Fin 3) (Fin 3) ℝ),
    G₀.det ≠ 0 ∧ G₁.det ≠ 0 ∧
    ∀ H : Matrix (Fin n) (Fin 3) ℝ × Matrix (Fin n) (Fin 3) ℝ,
      exp133Layer2 P₀' P₁' 1 H =
        exp133Layer2 P₀ P₁ 1 (if swap then (H.2 * G₁, H.1 * G₀) else (H.1 * G₀, H.2 * G₁))

/-- **Theorem 1 (Explicit `(1, 3, 3)` continuous non-gauge symmetry family & rank-multiset
obstruction)**:
1. For EVERY `t : ℝ` and EVERY input `X : Matrix (Fin n) (Fin 1) ℝ`, replacing `(I₃, I₃)` by
   `(blockP0 t, blockP1 t)` preserves the end-to-end map identically;
2. At `t = 1` (blocks `I₃, I₃`, determinants `1, 1`) versus `t = 0` (rank-1 blocks
   `diag(1,0,0), diag(0,1,0)`, determinants `0, 0`), NO hidden symmetry in `K₁ = GL(3)² ⋊ S₂`
   relates their layer-2 maps (for any `n ≥ 1`). -/
theorem expanding133_continuous_non_gauge_family {n : ℕ} (hn : 1 ≤ n) :
    (∀ (t : ℝ) (X : Matrix (Fin n) (Fin 1) ℝ),
      exp133Layer2 (blockP0 t) (blockP1 t) 1 (exp133Layer1 X) =
      exp133Layer2 1 1 1 (exp133Layer1 X)) ∧
    ¬ K1Layer2Related n 1 1 (blockP0 0) (blockP1 0) := by
  constructor
  · intro t X
    unfold exp133Layer2 exp133Layer1
    dsimp only
    rw [Matrix.mul_assoc, rowE3_mul_blockP0 t, Matrix.mul_assoc, rowE3_mul_blockP1 t,
        Matrix.mul_one, Matrix.mul_one, Matrix.mul_one, Matrix.mul_one]
  · rintro ⟨swap, G₀, G₁, hG₀, hG₁, hrel⟩
    -- Probe layer 2 on constant-token hidden arrays with head 0 = constRows n x and head 1 = 0
    have hprobe : ∀ x : Fin 3 → ℝ,
        constRows n x * blockP0 0 =
          constRows n x * (if swap then G₀ else G₀) := by
      intro x
      have hH := hrel (constRows n x, 0)
      unfold exp133Layer2 at hH
      simp only [Matrix.zero_mul, add_zero, Matrix.mul_one] at hH
      have hL : constRows n x * blockP0 0 = constRows n (fun j => ∑ k : Fin 3, x k * blockP0 0 k j) := by
        ext i j; simp [constRows, Matrix.mul_apply]
      rw [hL, attn_const_rows hn] at hH
      cases swap
      · simp only [Bool.false_eq_true, ↓reduceIte, add_zero] at hH
        have hR : constRows n x * G₀ = constRows n (fun j => ∑ k : Fin 3, x k * G₀ k j) := by
          ext i j; simp [constRows, Matrix.mul_apply]
        rw [hR, attn_const_rows hn] at hH
        rw [← hL, ← hR] at hH
        simpa using hH
      · simp only [↓reduceIte, zero_add] at hH
        have hR : constRows n x * G₀ = constRows n (fun j => ∑ k : Fin 3, x k * G₀ k j) := by
          ext i j; simp [constRows, Matrix.mul_apply]
        rw [hR, attn_const_rows hn] at hH
        rw [← hL, ← hR] at hH
        simpa using hH
    -- Hence blockP0 0 = G₀, contradicting det (blockP0 0) = 0 vs det G₀ ≠ 0
    have hmat : blockP0 0 = G₀ := by
      ext a b
      have hx := congrFun (congrFun (hprobe (fun k => if k = a then 1 else 0)) ⟨0, by omega⟩) b
      simp only [Matrix.mul_apply, constRows, Matrix.of_apply, ite_mul, one_mul, zero_mul] at hx
      simpa using hx
    have hdet0 : (blockP0 0).det = 0 := by
      simp [blockP0, Matrix.det_diagonal, Fin.prod_univ_three]
    rw [← hmat] at hG₀
    exact hG₀ hdet0

/-- **2D output span of the explicit `(1, 3, 3)` counterexample**:
For every `n ≥ 2`:
- On `X₁ = 1` (all ones), every output row equals `(1, 1, 0)`;
- On `X₂ = e₀` (first entry `1`, remaining `n - 1` entries `0`), every output row has strictly
  larger coordinate 1 than coordinate 0 (`y₀ < y₁`), so the `2 × 2` minor
  `1 * y₁ - 1 * y₀ = y₁ - y₀ > 0` is strictly nonzero. -/
theorem expanding133_output_span_2d {n : ℕ} (hn : 2 ≤ n) :
    let X₁ : Matrix (Fin n) (Fin 1) ℝ := Matrix.of fun _ _ => 1
    let X₂ : Matrix (Fin n) (Fin 1) ℝ := Matrix.of fun i _ => if i = ⟨0, by omega⟩ then 1 else 0
    let Y₁ := exp133Layer2 1 1 1 (exp133Layer1 X₁)
    let Y₂ := exp133Layer2 1 1 1 (exp133Layer1 X₂)
    (∀ i : Fin n, Y₁ i 0 = 1 ∧ Y₁ i 1 = 1 ∧ Y₁ i 2 = 0) ∧
    (∀ i : Fin n, Y₂ i 0 < Y₂ i 1 ∧ Y₂ i 2 = 0) := by
  intro X₁ X₂ Y₁ Y₂
  have hn1 : 1 ≤ n := by omega
  -- Helper: compute exp133Head1 a X₂ explicitly
  have hhead2 : ∀ (a : ℝ) (i : Fin n),
      exp133Head1 a X₂ i 0 =
        if i = ⟨0, by omega⟩ then Real.exp a / (Real.exp a + ((n - 1 : ℕ) : ℝ))
        else 1 / (n : ℝ) := by
    intro a i
    unfold exp133Head1 attnHead softmaxRow
    rw [Matrix.mul_apply]
    simp only [X₂, Matrix.of_apply]
    let A1 : Matrix (Fin 1) (Fin 1) ℝ := Matrix.of fun _ _ => a
    have hscore : ∀ r c : Fin n,
        (X₂ * A1 * X₂ᵀ) r c =
          if r = ⟨0, by omega⟩ ∧ c = ⟨0, by omega⟩ then a else 0 := by
      intro r c
      simp only [Matrix.mul_apply, Fin.sum_univ_one, X₂, A1, Matrix.of_apply, Matrix.transpose_apply]
      by_cases hr : r = ⟨0, by omega⟩ <;> by_cases hc : c = ⟨0, by omega⟩ <;> simp [hr, hc]
    -- Only j = 0 contributes to ∑_j softmax(r, j) * X₂(j, 0)
    have honly : ∑ j : Fin n,
        (Real.exp ((X₂ * A1 * X₂ᵀ) i j) /
          ∑ k : Fin n, Real.exp ((X₂ * A1 * X₂ᵀ) i k)) *
        (if j = ⟨0, by omega⟩ then (1 : ℝ) else 0) =
        Real.exp ((X₂ * A1 * X₂ᵀ) i ⟨0, by omega⟩) /
          ∑ k : Fin n, Real.exp ((X₂ * A1 * X₂ᵀ) i k) := by
      rw [Finset.sum_eq_single ⟨0, by omega⟩]
      · simp
      · intro j _ hj; simp [hj]
      · intro h; exact absurd (Finset.mem_univ _) h
    change ∑ j : Fin n,
        (Real.exp ((X₂ * A1 * X₂ᵀ) i j) /
          ∑ k : Fin n, Real.exp ((X₂ * A1 * X₂ᵀ) i k)) *
        (if j = ⟨0, by omega⟩ then (1 : ℝ) else 0) = _
    rw [honly]
    simp_rw [hscore]
    by_cases hi : i = ⟨0, by omega⟩
    · subst hi
      simp only [true_and, ↓reduceIte]
      have hden : ∑ k : Fin n, Real.exp (if k = ⟨0, by omega⟩ then a else 0) =
          Real.exp a + ((n - 1 : ℕ) : ℝ) := by
        rw [← Finset.add_sum_erase _ _ (Finset.mem_univ (⟨0, by omega⟩ : Fin n))]
        simp only [↓reduceIte]
        have hrest : ∑ k ∈ Finset.univ.erase (⟨0, by omega⟩ : Fin n),
            Real.exp (if k = ⟨0, by omega⟩ then a else 0) = ((n - 1 : ℕ) : ℝ) := by
          have h1 : ∀ k ∈ Finset.univ.erase (⟨0, by omega⟩ : Fin n),
              Real.exp (if k = ⟨0, by omega⟩ then a else 0) = 1 := by
            intro k hk
            have hne : k ≠ ⟨0, by omega⟩ := (Finset.mem_erase.mp hk).1
            simp [hne]
          rw [Finset.sum_congr rfl h1, Finset.sum_const, Finset.card_erase_of_mem (Finset.mem_univ _),
              Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one]
        rw [hrest]
      rw [hden]
    · simp only [hi, false_and, ↓reduceIte, Real.exp_zero, Finset.sum_const, Finset.card_univ,
        Fintype.card_fin, nsmul_eq_mul, mul_one]
  constructor
  · -- X₁ = all ones
    have hX1 : X₁ = constRows n (fun _ => 1) := rfl
    have h1a : exp133Head1 1 X₁ = X₁ := by
      unfold exp133Head1; rw [hX1]; exact attn_const_rows hn1 _ _
    have h1b : exp133Head1 2 X₁ = X₁ := by
      unfold exp133Head1; rw [hX1]; exact attn_const_rows hn1 _ _
    have hZ1 : (exp133Layer1 X₁).1 * (1 : Matrix (Fin 3) (Fin 3) ℝ) +
        (exp133Layer1 X₁).2 * (1 : Matrix (Fin 3) (Fin 3) ℝ) = constRows n ![1, 1, 0] := by
      unfold exp133Layer1
      dsimp only
      rw [Matrix.mul_one, Matrix.mul_one, h1a, h1b]
      ext i j
      fin_cases j <;> simp [X₁, rowE3, constRows, Matrix.mul_apply]
    intro i
    have hY1 : Y₁ = constRows n ![1, 1, 0] := by
      dsimp only [Y₁]; unfold exp133Layer2
      rw [hZ1, attn_const_rows hn1, Matrix.mul_one]
    rw [hY1]
    simp [constRows]
  · -- X₂ = e₀: strict positivity of softmax weights preserves strict inequality y₀ < y₁
    set Z₂ : Matrix (Fin n) (Fin 3) ℝ :=
      (exp133Layer1 X₂).1 * (1 : Matrix (Fin 3) (Fin 3) ℝ) +
      (exp133Layer1 X₂).2 * (1 : Matrix (Fin 3) (Fin 3) ℝ)
    have hZ2_0 : ∀ j : Fin n, Z₂ j 0 = exp133Head1 1 X₂ j 0 := by
      intro j; simp [Z₂, exp133Layer1, rowE3, Matrix.mul_apply]
    have hZ2_1 : ∀ j : Fin n, Z₂ j 1 = exp133Head1 2 X₂ j 0 := by
      intro j; simp [Z₂, exp133Layer1, rowE3, Matrix.mul_apply]
    have hZ2_2 : ∀ j : Fin n, Z₂ j 2 = 0 := by
      intro j; simp [Z₂, exp133Layer1, rowE3, Matrix.mul_apply]
    have hp_lt : Z₂ ⟨0, by omega⟩ 0 < Z₂ ⟨0, by omega⟩ 1 := by
      rw [hZ2_0, hZ2_1, hhead2 1, hhead2 2]
      simp only [↓reduceIte]
      have hm_pos : (0 : ℝ) < ((n - 1 : ℕ) : ℝ) := by
        have : 1 ≤ n - 1 := by omega
        positivity
      have hd1 : 0 < Real.exp 1 + ((n - 1 : ℕ) : ℝ) := by positivity
      have hd2 : 0 < Real.exp 2 + ((n - 1 : ℕ) : ℝ) := by positivity
      rw [div_lt_div_iff₀ hd1 hd2]
      have hexp : Real.exp 1 < Real.exp 2 := Real.exp_lt_exp.mpr (by norm_num)
      nlinarith
    have hp_le : ∀ j : Fin n, Z₂ j 0 ≤ Z₂ j 1 := by
      intro j
      by_cases hj : j = ⟨0, by omega⟩
      · subst hj; exact le_of_lt hp_lt
      · rw [hZ2_0, hZ2_1, hhead2 1, hhead2 2]
        simp [hj]
    intro i
    have hY2_eq : Y₂ = softmaxRow (Z₂ * (1 : Matrix (Fin 3) (Fin 3) ℝ) * Z₂ᵀ) * Z₂ := by
      dsimp only [Y₂]; unfold exp133Layer2 attnHead; rw [Matrix.mul_one]
    have hw_pos : ∀ j : Fin n, 0 < softmaxRow (Z₂ * (1 : Matrix (Fin 3) (Fin 3) ℝ) * Z₂ᵀ) i j := by
      intro j
      unfold softmaxRow
      apply div_pos (Real.exp_pos _)
      exact Finset.sum_pos (fun k _ => Real.exp_pos _) ⟨⟨0, by omega⟩, Finset.mem_univ _⟩
    constructor
    · rw [hY2_eq, Matrix.mul_apply, Matrix.mul_apply]
      apply Finset.sum_lt_sum
      · intro j _
        exact mul_le_mul_of_nonneg_left (hp_le j) (le_of_lt (hw_pos j))
      · exact ⟨⟨0, by omega⟩, Finset.mem_univ _, mul_lt_mul_of_pos_left hp_lt (hw_pos _)⟩
    · rw [hY2_eq, Matrix.mul_apply]
      apply Finset.sum_eq_zero
      intro j _
      rw [hZ2_2 j, mul_zero]

/-- **Proposition 2 (Failure of layerwise equivariance lift)**:
The explicit `(1, 3, 3)` network is end-to-end equivariant under `g = diag(1, 1, 2)` on outputs
(`F(X) * g = F(X)` for every `X`), yet admits NO matrix `M : Matrix (Fin 6) (Fin 6) ℝ` (invertible
or not) intertwining layer 2 with `g` on all hidden arrays `H : Matrix (Fin n) (Fin 6) ℝ`
(for any `n ≥ 2`). -/
theorem expanding133_equivariance_no_lift {n : ℕ} (hn : 2 ≤ n) :
    let g : Matrix (Fin 3) (Fin 3) ℝ := Matrix.diagonal ![1, 1, 2]
    let O₂ : Matrix (Fin 6) (Fin 3) ℝ := Matrix.of fun i j =>
      if i.val % 3 = j.val then 1 else 0
    (∀ X : Matrix (Fin n) (Fin 1) ℝ,
      exp133Layer2 1 1 1 (exp133Layer1 X) * g = exp133Layer2 1 1 1 (exp133Layer1 X)) ∧
    ¬ ∃ M : Matrix (Fin 6) (Fin 6) ℝ,
      ∀ H : Matrix (Fin n) (Fin 6) ℝ,
        attnHead (1 : Matrix (Fin 3) (Fin 3) ℝ) (H * M * O₂) =
        attnHead (1 : Matrix (Fin 3) (Fin 3) ℝ) (H * O₂) * g := by
  intro g O₂
  have hn1 : 1 ≤ n := by omega
  constructor
  · intro X
    have hZ2 : ∀ i : Fin n,
        ((exp133Layer1 X).1 * (1 : Matrix (Fin 3) (Fin 3) ℝ) +
         (exp133Layer1 X).2 * (1 : Matrix (Fin 3) (Fin 3) ℝ)) i 2 = 0 := by
      intro i; simp [exp133Layer1, rowE3, Matrix.mul_apply]
    have hY2 : ∀ i : Fin n, exp133Layer2 1 1 1 (exp133Layer1 X) i 2 = 0 := by
      intro i
      unfold exp133Layer2 attnHead
      rw [Matrix.mul_one, Matrix.mul_apply]
      exact Finset.sum_eq_zero fun j _ => by rw [hZ2 j, mul_zero]
    ext i j
    fin_cases j <;>
      simp [g, Matrix.mul_apply, Matrix.diagonal, hY2 i]
  · rintro ⟨M, hM⟩
    -- Constant-token hidden inputs force M * O₂ = O₂ * g
    have hMO : M * O₂ = O₂ * g := by
      ext a b
      have hconst := hM (constRows n (fun k => if k = a then 1 else 0))
      have hL : constRows n (fun k => if k = a then 1 else 0) * M * O₂ =
          constRows n (fun j => (M * O₂) a j) := by
        ext i j
        rw [Matrix.mul_assoc]
        simp [Matrix.mul_apply, constRows]
      have hR : constRows n (fun k => if k = a then 1 else 0) * O₂ =
          constRows n (fun j => O₂ a j) := by
        ext i j; simp [Matrix.mul_apply, constRows]
      rw [hL, attn_const_rows hn1, hR, attn_const_rows hn1] at hconst
      have hab := congrFun (congrFun hconst ⟨0, by omega⟩) b
      simpa [constRows, Matrix.mul_apply] using hab
    -- Now evaluate at H with row 0 = e₂ (in the first block) and all other rows 0,
    -- so H * O₂ has row 0 = e₂ and other rows 0, while H * M * O₂ = (H * O₂) * g has row 0 = 2 e₂
    let H₀ : Matrix (Fin n) (Fin 6) ℝ :=
      Matrix.of fun i k => if i = ⟨0, by omega⟩ ∧ k = (2 : Fin 6) then 1 else 0
    have hHO : H₀ * O₂ =
        Matrix.of fun i j => if i = ⟨0, by omega⟩ ∧ j = (2 : Fin 3) then 1 else 0 := by
      ext i j
      simp only [Matrix.mul_apply, H₀, O₂, Matrix.of_apply]
      rw [Finset.sum_eq_single (2 : Fin 6)]
      · by_cases hi : i = ⟨0, by omega⟩ <;> fin_cases j <;> simp [hi]
      · intro k _ hk; simp [hk]
      · intro h; exact absurd (Finset.mem_univ _) h
    have hHMO : H₀ * M * O₂ =
        Matrix.of fun i j => if i = ⟨0, by omega⟩ ∧ j = (2 : Fin 3) then 2 else 0 := by
      rw [Matrix.mul_assoc, hMO, ← Matrix.mul_assoc, hHO]
      ext i j
      simp only [Matrix.mul_apply, g, Matrix.diagonal, Matrix.of_apply]
      rw [Finset.sum_eq_single (2 : Fin 3)]
      · by_cases hi : i = ⟨0, by omega⟩ <;> fin_cases j <;> simp [hi]
      · intro k _ hk; simp [hk]
      · intro h; exact absurd (Finset.mem_univ _) h
    -- Compute the (0, 2) entry of both sides of hM H₀
    have heval_scale : ∀ (c : ℝ),
        attnHead (1 : Matrix (Fin 3) (Fin 3) ℝ)
          (Matrix.of fun (i : Fin n) (j : Fin 3) => if i = ⟨0, by omega⟩ ∧ j = 2 then c else 0)
          ⟨0, by omega⟩ 2 =
        c * Real.exp (c ^ 2) / (Real.exp (c ^ 2) + ((n - 1 : ℕ) : ℝ)) := by
      intro c
      set Z : Matrix (Fin n) (Fin 3) ℝ :=
        Matrix.of fun i j => if i = ⟨0, by omega⟩ ∧ j = 2 then c else 0
      have hS : ∀ r k : Fin n, (Z * (1 : Matrix (Fin 3) (Fin 3) ℝ) * Zᵀ) r k =
          if r = ⟨0, by omega⟩ ∧ k = ⟨0, by omega⟩ then c ^ 2 else 0 := by
        intro r k
        rw [Matrix.mul_one, Matrix.mul_apply]
        rw [Finset.sum_eq_single (2 : Fin 3)]
        · by_cases hr : r = ⟨0, by omega⟩ <;> by_cases hk : k = ⟨0, by omega⟩ <;>
            simp [Z, hr, hk, sq]
        · intro j _ hj; simp [Z, hj]
        · intro h; exact absurd (Finset.mem_univ _) h
      unfold attnHead softmaxRow
      rw [Matrix.mul_apply]
      rw [Finset.sum_eq_single ⟨0, by omega⟩]
      · simp_rw [hS]
        simp only [true_and, ↓reduceIte, Z, Matrix.of_apply]
        have hden : ∑ k : Fin n, Real.exp (if k = ⟨0, by omega⟩ then c ^ 2 else 0) =
            Real.exp (c ^ 2) + ((n - 1 : ℕ) : ℝ) := by
          rw [← Finset.add_sum_erase _ _ (Finset.mem_univ (⟨0, by omega⟩ : Fin n))]
          simp only [↓reduceIte]
          have hrest : ∑ k ∈ Finset.univ.erase (⟨0, by omega⟩ : Fin n),
              Real.exp (if k = ⟨0, by omega⟩ then c ^ 2 else 0) = ((n - 1 : ℕ) : ℝ) := by
            have h1 : ∀ k ∈ Finset.univ.erase (⟨0, by omega⟩ : Fin n),
                Real.exp (if k = ⟨0, by omega⟩ then c ^ 2 else 0) = 1 := by
              intro k hk
              have hne : k ≠ ⟨0, by omega⟩ := (Finset.mem_erase.mp hk).1
              simp [hne]
            rw [Finset.sum_congr rfl h1, Finset.sum_const,
                Finset.card_erase_of_mem (Finset.mem_univ _), Finset.card_univ, Fintype.card_fin,
                nsmul_eq_mul, mul_one]
          rw [hrest]
        rw [hden]; ring
      · intro k _ hk; simp [Z, hk]
      · intro h; exact absurd (Finset.mem_univ _) h
    have h02 := congrFun (congrFun (hM H₀) ⟨0, by omega⟩) 2
    rw [hHMO, hHO] at h02
    have hL := heval_scale 2
    have hR_raw := heval_scale 1
    have hR : (attnHead (1 : Matrix (Fin 3) (Fin 3) ℝ)
        (Matrix.of fun (i : Fin n) (j : Fin 3) => if i = ⟨0, by omega⟩ ∧ j = 2 then 1 else 0) * g)
        ⟨0, by omega⟩ 2 =
        2 * Real.exp 1 / (Real.exp 1 + ((n - 1 : ℕ) : ℝ)) := by
      rw [Matrix.mul_apply, Fin.sum_univ_three]
      simp only [g, Matrix.diagonal, Matrix.of_apply,
        show (0 : Fin 3) ≠ 2 by decide, show (1 : Fin 3) ≠ 2 by decide, ↓reduceIte, mul_zero,
        zero_add]
      have hg2 : (![1, 1, 2] : Fin 3 → ℝ) 2 = 2 := rfl
      rw [hR_raw, hg2]
      norm_num
      ring
    rw [hL, hR] at h02
    norm_num at h02
    have hm_pos : (0 : ℝ) < ((n - 1 : ℕ) : ℝ) := by
      have : 1 ≤ n - 1 := by omega
      positivity
    have hd4 : 0 < Real.exp 4 + ((n - 1 : ℕ) : ℝ) := by positivity
    have hd1 : 0 < Real.exp 1 + ((n - 1 : ℕ) : ℝ) := by positivity
    rw [div_eq_div_iff (ne_of_gt hd4) (ne_of_gt hd1)] at h02
    have hexp : Real.exp 1 < Real.exp 4 := Real.exp_lt_exp.mpr (by norm_num)
    nlinarith

/-! ### 2. Single-Head Two-Type Probe, Score Determination, and `n = 2` Opposite-Score Degeneracy -/

/-- Two-type token matrix: row `0` equals `x + v`, all other rows equal `x`. -/
def twoTypeMat {n d : ℕ} (hn : 2 ≤ n) (x v : Fin d → ℝ) : Matrix (Fin n) (Fin d) ℝ :=
  Matrix.of fun i j => if i = ⟨0, by omega⟩ then x j + v j else x j

/-- Row-vector inner product `x E vᵀ`. -/
def bilin {d : ℕ} (x : Fin d → ℝ) (E : Matrix (Fin d) (Fin d) ℝ) (v : Fin d → ℝ) : ℝ :=
  ∑ a : Fin d, ∑ b : Fin d, x a * E a b * v b

/-- **Two-type probe formula**: On `twoTypeMat hn x v` (`n ≥ 2`), every common row `i ≠ 0` of
`T_E(X) * D` equals `x D + σ_n(x E vᵀ) • (v D)`. -/
theorem two_type_probe_formula {n d m : ℕ} (hn : 2 ≤ n)
    (E : Matrix (Fin d) (Fin d) ℝ) (D : Matrix (Fin d) (Fin m) ℝ)
    (x v : Fin d → ℝ) (i : Fin n) (hi : i ≠ ⟨0, by omega⟩) (c : Fin m) :
    (attnHead E (twoTypeMat hn x v) * D) i c =
      (∑ k : Fin d, x k * D k c) +
        sigmaN n (bilin x E v) * (∑ k : Fin d, v k * D k c) := by
  set X := twoTypeMat hn x v
  have hS : ∀ j : Fin n, (X * E * Xᵀ) i j =
      if j = ⟨0, by omega⟩ then bilin x E x + bilin x E v else bilin x E x := by
    intro j
    simp only [Matrix.mul_apply, X, twoTypeMat, Matrix.of_apply, Matrix.transpose_apply, hi,
      ↓reduceIte, bilin]
    by_cases hj : j = ⟨0, by omega⟩
    · simp only [hj, ↓reduceIte]
      calc ∑ a : Fin d, (∑ b : Fin d, x b * E b a) * (x a + v a)
          = ∑ a : Fin d, (∑ b : Fin d, x b * E b a * x a + ∑ b : Fin d, x b * E b a * v a) := by
            refine Finset.sum_congr rfl fun a _ => ?_
            rw [Finset.sum_mul, ← Finset.sum_add_distrib]
            refine Finset.sum_congr rfl fun b _ => by ring
        _ = (∑ a : Fin d, ∑ b : Fin d, x b * E b a * x a) +
            (∑ a : Fin d, ∑ b : Fin d, x b * E b a * v a) := Finset.sum_add_distrib
        _ = (∑ b : Fin d, ∑ a : Fin d, x b * E b a * x a) +
            (∑ b : Fin d, ∑ a : Fin d, x b * E b a * v a) := by
            rw [Finset.sum_comm (s := Finset.univ) (t := Finset.univ) (f := fun a b => x b * E b a * x a),
                Finset.sum_comm (s := Finset.univ) (t := Finset.univ) (f := fun a b => x b * E b a * v a)]
    · simp only [hj, ↓reduceIte]
      calc ∑ a : Fin d, (∑ b : Fin d, x b * E b a) * x a
          = ∑ a : Fin d, ∑ b : Fin d, x b * E b a * x a :=
            Finset.sum_congr rfl fun a _ => Finset.sum_mul _ _ _
        _ = ∑ b : Fin d, ∑ a : Fin d, x b * E b a * x a := Finset.sum_comm
  -- Softmax weight on row 0 is sigmaN n (bilin x E v)
  have hden : ∑ k : Fin n, Real.exp ((X * E * Xᵀ) i k) =
      Real.exp (bilin x E x) * (Real.exp (bilin x E v) + ((n - 1 : ℕ) : ℝ)) := by
    simp_rw [hS]
    rw [← Finset.add_sum_erase _ _ (Finset.mem_univ (⟨0, by omega⟩ : Fin n))]
    simp only [↓reduceIte, Real.exp_add]
    have hrest : ∑ k ∈ Finset.univ.erase (⟨0, by omega⟩ : Fin n),
        Real.exp (if k = ⟨0, by omega⟩ then bilin x E x + bilin x E v else bilin x E x) =
        ((n - 1 : ℕ) : ℝ) * Real.exp (bilin x E x) := by
      have h1 : ∀ k ∈ Finset.univ.erase (⟨0, by omega⟩ : Fin n),
          Real.exp (if k = ⟨0, by omega⟩ then bilin x E x + bilin x E v else bilin x E x) =
            Real.exp (bilin x E x) := by
        intro k hk
        have hne : k ≠ ⟨0, by omega⟩ := (Finset.mem_erase.mp hk).1
        simp [hne]
      rw [Finset.sum_congr rfl h1, Finset.sum_const,
          Finset.card_erase_of_mem (Finset.mem_univ _), Finset.card_univ, Fintype.card_fin,
          nsmul_eq_mul]
    rw [hrest]; ring
  have hw0 : softmaxRow (X * E * Xᵀ) i ⟨0, by omega⟩ = sigmaN n (bilin x E v) := by
    unfold softmaxRow sigmaN
    rw [hS, if_pos rfl, Real.exp_add, hden]
    have hxx : Real.exp (bilin x E x) ≠ 0 := ne_of_gt (Real.exp_pos _)
    have hvv : Real.exp (bilin x E v) ≠ 0 := ne_of_gt (Real.exp_pos _)
    have h1 : 0 < Real.exp (bilin x E v) + ((n - 1 : ℕ) : ℝ) := by positivity
    have h2 : 0 < 1 + ((n - 1 : ℕ) : ℝ) * Real.exp (-bilin x E v) := by positivity
    have hcancel : Real.exp (bilin x E v) * Real.exp (-bilin x E v) = 1 := by
      rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
    field_simp [hxx, ne_of_gt h1, ne_of_gt h2]
    calc Real.exp (bilin x E v) * (1 + ((n - 1 : ℕ) : ℝ) * Real.exp (-bilin x E v))
        = Real.exp (bilin x E v) + ((n - 1 : ℕ) : ℝ) * (Real.exp (bilin x E v) * Real.exp (-bilin x E v)) := by ring
      _ = Real.exp (bilin x E v) + ((n - 1 : ℕ) : ℝ) := by rw [hcancel, mul_one]
  have hsum_w : ∑ j : Fin n, softmaxRow (X * E * Xᵀ) i j = 1 := by
    unfold softmaxRow
    rw [← Finset.sum_div]
    have hpos : 0 < ∑ k : Fin n, Real.exp ((X * E * Xᵀ) i k) :=
      Finset.sum_pos (fun k _ => Real.exp_pos _) ⟨⟨0, by omega⟩, Finset.mem_univ _⟩
    exact div_self (ne_of_gt hpos)
  -- Compute row i of attnHead E X
  have hrow : ∀ k : Fin d, attnHead E X i k = x k + sigmaN n (bilin x E v) * v k := by
    intro k
    unfold attnHead
    rw [Matrix.mul_apply]
    have hsplit : ∑ j : Fin n, softmaxRow (X * E * Xᵀ) i j * X j k =
        (∑ j : Fin n, softmaxRow (X * E * Xᵀ) i j * x k) +
        softmaxRow (X * E * Xᵀ) i ⟨0, by omega⟩ * v k := by
      rw [← Finset.add_sum_erase _ _ (Finset.mem_univ (⟨0, by omega⟩ : Fin n)),
          ← Finset.add_sum_erase (s := Finset.univ)
            (f := fun j => softmaxRow (X * E * Xᵀ) i j * x k) (Finset.mem_univ (⟨0, by omega⟩ : Fin n))]
      have hrest : ∑ j ∈ Finset.univ.erase (⟨0, by omega⟩ : Fin n),
          softmaxRow (X * E * Xᵀ) i j * X j k =
          ∑ j ∈ Finset.univ.erase (⟨0, by omega⟩ : Fin n),
            softmaxRow (X * E * Xᵀ) i j * x k := by
        refine Finset.sum_congr rfl fun j hj => ?_
        have hne : j ≠ ⟨0, by omega⟩ := (Finset.mem_erase.mp hj).1
        simp [X, twoTypeMat, hne]
      rw [hrest]
      simp only [X, twoTypeMat, Matrix.of_apply, ↓reduceIte]
      ring
    rw [hsplit, ← Finset.sum_mul, hsum_w, one_mul, hw0]
  rw [Matrix.mul_apply]
  simp_rw [hrow, add_mul, mul_assoc, Finset.sum_add_distrib, ← Finset.mul_sum]

/-- **Single-head exact score determination**: For `n ≥ 2` and any nonzero `D ≠ 0`, if
`T_E(X) * D = T_{E'}(X) * D` for all `X : Matrix (Fin n) (Fin d) ℝ`, then `E = E'`. -/
theorem single_head_determines_score {n d m : ℕ} (hn : 2 ≤ n)
    {E E' : Matrix (Fin d) (Fin d) ℝ} {D : Matrix (Fin d) (Fin m) ℝ}
    (hD : D ≠ 0)
    (heq : ∀ X : Matrix (Fin n) (Fin d) ℝ, attnHead E X * D = attnHead E' X * D) :
    E = E' := by
  have hex : ∃ (v₀ : Fin d → ℝ) (c₀ : Fin m), ∑ k : Fin d, v₀ k * D k c₀ ≠ 0 := by
    by_contra hall
    push_neg at hall
    apply hD
    ext a c
    have h := hall (fun k => if k = a then 1 else 0) c
    simpa using h
  obtain ⟨v₀, c₀, hv₀⟩ := hex
  have hbilin_ne : ∀ (x v : Fin d → ℝ), (∃ c : Fin m, ∑ k : Fin d, v k * D k c ≠ 0) →
      bilin x E v = bilin x E' v := by
    rintro x v ⟨c, hvc⟩
    have hrow := congrFun (congrFun (heq (twoTypeMat hn x v)) ⟨1, by omega⟩) c
    have hne1 : (⟨1, by omega⟩ : Fin n) ≠ ⟨0, by omega⟩ := by
      intro h; exact Nat.one_ne_zero (congrArg Fin.val h)
    rw [two_type_probe_formula hn E D x v _ hne1 c,
        two_type_probe_formula hn E' D x v _ hne1 c] at hrow
    have hsig : sigmaN n (bilin x E v) = sigmaN n (bilin x E' v) :=
      mul_right_cancel₀ hvc (by linarith)
    exact (sigmaN_strictMono hn).injective hsig
  have hbilin_all : ∀ x w : Fin d → ℝ, bilin x E w = bilin x E' w := by
    intro x w
    by_cases hw : ∑ k : Fin d, w k * D k c₀ ≠ 0
    · exact hbilin_ne x w ⟨c₀, hw⟩
    · push_neg at hw
      have hvw : ∑ k : Fin d, (v₀ k + w k) * D k c₀ ≠ 0 := by
        simp_rw [add_mul, Finset.sum_add_distrib, hw, add_zero]
        exact hv₀
      have h1 := hbilin_ne x (fun k => v₀ k + w k) ⟨c₀, hvw⟩
      have h2 := hbilin_ne x v₀ ⟨c₀, hv₀⟩
      have hadd : ∀ M : Matrix (Fin d) (Fin d) ℝ,
          bilin x M (fun k => v₀ k + w k) = bilin x M v₀ + bilin x M w := by
        intro M; unfold bilin
        simp_rw [mul_add, Finset.sum_add_distrib]
      rw [hadd E, hadd E'] at h1
      linarith
  ext a b
  have hab := hbilin_all (fun k => if k = a then 1 else 0) (fun k => if k = b then 1 else 0)
  simpa [bilin] using hab

/-- Exact `n = 2` opposite-score cancellation: `T_A(X) + T_{-A}(X)` outputs the row sum
`X 0 + X 1` in both rows (making `X ↦ T_A(X) + T_{-A}(X)` purely linear!). -/
theorem opposite_score_sum_linear {d : ℕ} (A : Matrix (Fin d) (Fin d) ℝ)
    (X : Matrix (Fin 2) (Fin d) ℝ) (i : Fin 2) (j : Fin d) :
    (attnHead A X + attnHead (-A) X) i j = X 0 j + X 1 j := by
  have hneg : X * (-A) * Xᵀ = -(X * A * Xᵀ) := by
    simp [Matrix.mul_neg, Matrix.neg_mul]
  have hpair : ∀ s t : ℝ,
      Real.exp s / (Real.exp s + Real.exp t) + Real.exp (-s) / (Real.exp (-s) + Real.exp (-t)) = 1 := by
    intro s t
    have h1 : 0 < Real.exp s + Real.exp t := by positivity
    have h2 : 0 < Real.exp (-s) + Real.exp (-t) := by positivity
    have hs : Real.exp s * Real.exp (-s) = 1 := by rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
    have ht : Real.exp t * Real.exp (-t) = 1 := by rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
    field_simp [ne_of_gt h1, ne_of_gt h2]
    linear_combination hs - ht
  set S : Matrix (Fin 2) (Fin 2) ℝ := X * A * Xᵀ
  have hSneg : X * (-A) * Xᵀ = -S := hneg
  unfold attnHead softmaxRow
  rw [hSneg]
  change (∑ k : Fin 2, (Real.exp (S i k) / ∑ m : Fin 2, Real.exp (S i m)) * X k j) +
      (∑ k : Fin 2, (Real.exp ((-S) i k) / ∑ m : Fin 2, Real.exp ((-S) i m)) * X k j) =
      X 0 j + X 1 j
  simp only [Fin.sum_univ_two, Matrix.neg_apply]
  have h0 := hpair (S i 0) (S i 1)
  have h1 := hpair (S i 1) (S i 0)
  rw [add_comm (Real.exp (S i 1)) (Real.exp (S i 0)),
      add_comm (Real.exp (-(S i 1))) (Real.exp (-(S i 0)))] at h1
  linear_combination X 0 j * h0 + X 1 j * h1

/-! ### 3. Constant-Width Generic Identifiability Core (Theorem 3) -/

/-- **Step 1 (Output-coordinate normalization)**:
If `U` is invertible (`U * Uinv = 1`) and `Y = X * U`, then
`T_A(X) * V * P * U' = T_{Uinv * A * Uinvᵀ}(Y) * (Uinv * V * P * U')`. -/
theorem block_output_normalization {n d : ℕ}
    (A V P U Uinv U' : Matrix (Fin d) (Fin d) ℝ) (X : Matrix (Fin n) (Fin d) ℝ)
    (hUR : U * Uinv = 1) :
    let Y := X * U
    let a := Uinv * A * Uinvᵀ
    let b := Uinv * V * P * U'
    attnHead A X * V * P * U' = attnHead a Y * b := by
  intro Y a b
  have hscore : Y * a * Yᵀ = X * A * Xᵀ := by
    calc Y * a * Yᵀ
        = (X * U) * (Uinv * A * Uinvᵀ) * (Uᵀ * Xᵀ) := by
            simp only [Y, a, Matrix.transpose_mul]
      _ = X * (U * Uinv) * A * (Uinvᵀ * Uᵀ) * Xᵀ := by simp only [Matrix.mul_assoc]
      _ = X * (1 : Matrix (Fin d) (Fin d) ℝ) * A * (U * Uinv)ᵀ * Xᵀ := by
            rw [← Matrix.transpose_mul, hUR]
      _ = X * A * Xᵀ := by
            rw [hUR, Matrix.transpose_one, Matrix.mul_one, Matrix.mul_one]
  unfold attnHead
  rw [hscore]
  calc softmaxRow (X * A * Xᵀ) * X * V * P * U'
      = softmaxRow (X * A * Xᵀ) * X * (1 * V) * P * U' := by rw [Matrix.one_mul]
    _ = softmaxRow (X * A * Xᵀ) * X * ((U * Uinv) * V) * P * U' := by rw [hUR]
    _ = softmaxRow (X * A * Xᵀ) * Y * b := by simp only [Y, b, Matrix.mul_assoc]

/-- **Step 3 (Real-asymptotic linear independence of distinct positive sigmoid frequencies)**:
For every `n ≥ 2`, any finite set `S : Finset ℝ` of strictly positive frequencies (`∀ q ∈ S, 0 < q`),
any constant `c₀ : ℝ`, and any coefficients `c : ℝ → ℝ`, if
`∀ s : ℝ, c₀ + ∑ q ∈ S, c q * sigmaN n (q * s) = 0`,
then `c₀ = 0` and `∀ q ∈ S, c q = 0`. -/
theorem sigmoid_freq_linear_independent {n : ℕ} (hn : 2 ≤ n) :
    ∀ (S : Finset ℝ), (∀ q ∈ S, 0 < q) →
      ∀ (c₀ : ℝ) (c : ℝ → ℝ),
        (∀ s : ℝ, c₀ + ∑ q ∈ S, c q * sigmaN n (q * s) = 0) →
        c₀ = 0 ∧ ∀ q ∈ S, c q = 0 := by
  have hm_pos : (0 : ℝ) < ((n - 1 : ℕ) : ℝ) := by
    have : 1 ≤ n - 1 := by omega
    positivity
  have hsig_tend : ∀ {q : ℝ}, 0 < q → Tendsto (fun s : ℝ => sigmaN n (q * s)) atTop (𝓝 1) := by
    intro q hq
    unfold sigmaN
    have hexp : Tendsto (fun s : ℝ => Real.exp (-(q * s))) atTop (𝓝 0) := by
      have hlin : Tendsto (fun s : ℝ => -(q * s)) atTop atBot := by
        simpa [neg_mul] using tendsto_id.const_mul_atTop_of_neg (neg_lt_zero.mpr hq)
      exact Real.tendsto_exp_atBot.comp hlin
    have hden : Tendsto (fun s : ℝ => 1 + ((n - 1 : ℕ) : ℝ) * Real.exp (-(q * s))) atTop (𝓝 1) := by
      have hmul := hexp.const_mul ((n - 1 : ℕ) : ℝ)
      rw [mul_zero] at hmul
      simpa using (tendsto_const_nhds (x := (1 : ℝ))).add hmul
    have hdiv : Tendsto (fun s : ℝ => (1 : ℝ) / (1 + ((n - 1 : ℕ) : ℝ) * Real.exp (-(q * s))))
        atTop (𝓝 (1 / 1)) :=
      tendsto_const_nhds.div hden one_ne_zero
    simpa only [div_one] using hdiv
  -- Strong induction on S.card
  intro S
  induction hcard : S.card using Nat.strong_induction_on generalizing S with
  | h k ih =>
    intro hSpos c₀ c hzero
    by_cases hempty : S = ∅
    · subst hempty
      have h0 := hzero 0
      simp only [Finset.sum_empty, add_zero] at h0
      exact ⟨h0, fun q hq => absurd hq (Finset.notMem_empty q)⟩
    · have hne : S.Nonempty := Finset.nonempty_iff_ne_empty.mpr hempty
      -- First take s → +∞ to get c₀ + ∑_{q ∈ S} c q = 0
      have hlim_sum : Tendsto (fun s : ℝ => c₀ + ∑ q ∈ S, c q * sigmaN n (q * s)) atTop
          (𝓝 (c₀ + ∑ q ∈ S, c q * 1)) := by
        refine tendsto_const_nhds.add ?_
        exact tendsto_finset_sum S (fun q hq => (hsig_tend (hSpos q hq)).const_mul (c q))
      have hlim_zero : Tendsto (fun s : ℝ => c₀ + ∑ q ∈ S, c q * sigmaN n (q * s)) atTop (𝓝 0) := by
        simp [hzero]
      have hsum_eq0 : c₀ + ∑ q ∈ S, c q = 0 := by
        have h := tendsto_nhds_unique hlim_sum hlim_zero
        simpa using h
      -- Let q₀ = min' S be the smallest positive frequency
      let q₀ := S.min' hne
      have hq₀_mem : q₀ ∈ S := Finset.min'_mem S hne
      have hq₀_pos : 0 < q₀ := hSpos q₀ hq₀_mem
      -- Rewrite identity as ∑_{q ∈ S} c q * (sigmaN n (q * s) - 1) = 0, then multiply by exp(q₀ * s)
      have hdiff_zero : ∀ s : ℝ,
          ∑ q ∈ S, c q * (Real.exp (q₀ * s) * (sigmaN n (q * s) - 1)) = 0 := by
        intro s
        have h1 : ∑ q ∈ S, c q * (sigmaN n (q * s) - 1) = 0 := by
          have := hzero s
          have hsub : ∑ q ∈ S, c q * (sigmaN n (q * s) - 1) =
              (∑ q ∈ S, c q * sigmaN n (q * s)) - ∑ q ∈ S, c q := by
            simp_rw [mul_sub, mul_one, Finset.sum_sub_distrib]
          linarith
        have h2 : ∑ q ∈ S, c q * (Real.exp (q₀ * s) * (sigmaN n (q * s) - 1)) =
            Real.exp (q₀ * s) * ∑ q ∈ S, c q * (sigmaN n (q * s) - 1) := by
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun q _ => by ring
        rw [h2, h1, mul_zero]
      -- Compute limit of exp(q₀ * s) * (sigmaN n (q * s) - 1) as s → +∞
      have hrew : ∀ q s : ℝ,
          Real.exp (q₀ * s) * (sigmaN n (q * s) - 1) =
          -(((n - 1 : ℕ) : ℝ) * Real.exp ((q₀ - q) * s)) /
            (1 + ((n - 1 : ℕ) : ℝ) * Real.exp (-(q * s))) := by
        intro q s
        unfold sigmaN
        have hden : 0 < 1 + ((n - 1 : ℕ) : ℝ) * Real.exp (-(q * s)) := by positivity
        have hexp : Real.exp (q₀ * s) * Real.exp (-(q * s)) = Real.exp ((q₀ - q) * s) := by
          rw [← Real.exp_add]; congr 1; ring
        have hnum : Real.exp (q₀ * s) * (1 - (1 + ((n - 1 : ℕ) : ℝ) * Real.exp (-(q * s)))) =
            -(((n - 1 : ℕ) : ℝ) * Real.exp ((q₀ - q) * s)) := by
          calc Real.exp (q₀ * s) * (1 - (1 + ((n - 1 : ℕ) : ℝ) * Real.exp (-(q * s))))
              = -(((n - 1 : ℕ) : ℝ) * (Real.exp (q₀ * s) * Real.exp (-(q * s)))) := by ring
            _ = -(((n - 1 : ℕ) : ℝ) * Real.exp ((q₀ - q) * s)) := by rw [hexp]
        rw [eq_div_iff (ne_of_gt hden)]
        have h1 : (1 / (1 + ((n - 1 : ℕ) : ℝ) * Real.exp (-(q * s))) - 1) *
            (1 + ((n - 1 : ℕ) : ℝ) * Real.exp (-(q * s))) =
            1 - (1 + ((n - 1 : ℕ) : ℝ) * Real.exp (-(q * s))) := by
          field_simp [ne_of_gt hden]
        rw [mul_assoc, h1, hnum]
      have hterm_tend : ∀ q ∈ S,
          Tendsto (fun s : ℝ => c q * (Real.exp (q₀ * s) * (sigmaN n (q * s) - 1))) atTop
            (𝓝 (if q = q₀ then c q₀ * (-((n - 1 : ℕ) : ℝ)) else 0)) := by
        intro q hq
        simp_rw [hrew q]
        have hden : Tendsto (fun s : ℝ => 1 + ((n - 1 : ℕ) : ℝ) * Real.exp (-(q * s))) atTop (𝓝 1) := by
          have hexp : Tendsto (fun s : ℝ => Real.exp (-(q * s))) atTop (𝓝 0) :=
            Real.tendsto_exp_atBot.comp (by
              simpa [neg_mul] using tendsto_id.const_mul_atTop_of_neg (neg_lt_zero.mpr (hSpos q hq)))
          simpa using (tendsto_const_nhds (x := (1 : ℝ))).add (hexp.const_mul ((n - 1 : ℕ) : ℝ))
        by_cases hqq : q = q₀
        · subst hqq
          simp only [↓reduceIte, sub_self, zero_mul, Real.exp_zero, mul_one]
          have hdiv := (tendsto_const_nhds (x := -((n - 1 : ℕ) : ℝ))).div hden (by norm_num)
          simpa using hdiv.const_mul (c q₀)
        · simp only [hqq, ↓reduceIte]
          have hq_gt : q₀ < q := lt_of_le_of_ne (Finset.min'_le S q hq) (Ne.symm hqq)
          have hnum : Tendsto (fun s : ℝ => -(((n - 1 : ℕ) : ℝ) * Real.exp ((q₀ - q) * s))) atTop (𝓝 0) := by
            have hexp : Tendsto (fun s : ℝ => Real.exp ((q₀ - q) * s)) atTop (𝓝 0) :=
              Real.tendsto_exp_atBot.comp (tendsto_id.const_mul_atTop_of_neg (sub_neg.mpr hq_gt))
            simpa using (hexp.const_mul ((n - 1 : ℕ) : ℝ)).neg
          have hdiv := hnum.div hden (by norm_num)
          simpa using hdiv.const_mul (c q)
      have hlim_peel := tendsto_finset_sum S hterm_tend
      have hlim_zero2 : Tendsto (fun s : ℝ => ∑ q ∈ S, c q * (Real.exp (q₀ * s) * (sigmaN n (q * s) - 1)))
          atTop (𝓝 0) := by
        simp [hdiff_zero]
      have hpeel_eq := tendsto_nhds_unique hlim_peel hlim_zero2
      rw [Finset.sum_ite_eq' S q₀, if_pos hq₀_mem] at hpeel_eq
      have hcq₀ : c q₀ = 0 := by
        have hneg_ne : -((n - 1 : ℕ) : ℝ) ≠ 0 := neg_ne_zero.mpr (ne_of_gt hm_pos)
        exact (mul_eq_zero.mp hpeel_eq).resolve_right hneg_ne
      -- Apply induction hypothesis to S.erase q₀
      have hcard_lt : (S.erase q₀).card < k := by
        rw [Finset.card_erase_of_mem hq₀_mem, hcard]
        have hpos : 0 < S.card := hne.card_pos
        omega
      have hzero_erase : ∀ s : ℝ, c₀ + ∑ q ∈ S.erase q₀, c q * sigmaN n (q * s) = 0 := by
        intro s
        have hsplit := Finset.add_sum_erase S (fun q => c q * sigmaN n (q * s)) hq₀_mem
        dsimp only at hsplit
        rw [hcq₀, zero_mul, zero_add] at hsplit
        rw [hsplit]
        exact hzero s
      obtain ⟨hc₀, herase⟩ := ih (S.erase q₀).card hcard_lt (S.erase q₀) rfl
        (fun q hq => hSpos q (Finset.mem_of_mem_erase hq)) c₀ c hzero_erase
      refine ⟨hc₀, fun q hq => ?_⟩
      by_cases hqq : q = q₀
      · subst hqq; exact hcq₀
      · exact herase q (Finset.mem_erase.mpr ⟨hqq, hq⟩)

/-- **Step 5 (Existence and uniqueness of the constant-width `K_i` transporter lift)**:
For a layer with `h` heads, invertible normalized factorizations `b_j = D_j * R_j` and
`b'_j = D'_j * R'_j` (with `R_j = P_j U_{i+1}`), distinct scores `a_j`, and a matched head
permutation `π : Equiv.Perm (Fin h)` with `a'_{π j} = a_j` and `b'_{π j} = b_j`:
1. Each `D_j` and `D'_j` is invertible, and the head feature transporter `M_j := D_j⁻¹ * D'_{π j}`
   is invertible and satisfies `D_j * M_j = D'_{π j}` and `M_j * R'_{π j} = R_j` (preserving
   normalized downstream input on all hidden arrays);
2. Uniqueness: any other `(π', M')` with `a'_{π' j} = a_j` and `D_j * M'_j = D'_{π' j}` must
   satisfy `π' = π` and `M' = M`. -/
theorem constant_width_unique_lift {h d : ℕ}
    (a a' D D' R R' : Fin h → Matrix (Fin d) (Fin d) ℝ)
    (ha_inj : Function.Injective a)
    (hbinv : ∀ j, IsUnit (D j * R j))
    (π : Equiv.Perm (Fin h))
    (ha_match : ∀ j, a' (π j) = a j)
    (hb_match : ∀ j, D' (π j) * R' (π j) = D j * R j) :
    ∃! (pair : Equiv.Perm (Fin h) × (Fin h → Matrix (Fin d) (Fin d) ℝ)),
      (∀ j, a' (pair.1 j) = a j) ∧
      (∀ j, D j * pair.2 j = D' (pair.1 j)) ∧
      (∀ j, pair.2 j * R' (pair.1 j) = R j) ∧
      (∀ j, IsUnit (pair.2 j)) := by
  -- From IsUnit (D j * R j), both D j and R j are units (in finite-dim Matrix algebra)
  have hD_unit : ∀ j, IsUnit (D j) := fun j =>
    (Matrix.isUnit_iff_isUnit_det _).mpr <|
      isUnit_of_mul_isUnit_left (by rw [← Matrix.det_mul]; exact (Matrix.isUnit_iff_isUnit_det _).mp (hbinv j))
  have hD'_unit : ∀ j, IsUnit (D' (π j)) := fun j =>
    (Matrix.isUnit_iff_isUnit_det _).mpr <|
      isUnit_of_mul_isUnit_left (by
        rw [← Matrix.det_mul, hb_match j]
        exact (Matrix.isUnit_iff_isUnit_det _).mp (hbinv j))
  let Dinv : Fin h → Matrix (Fin d) (Fin d) ℝ := fun j => ↑((hD_unit j).unit⁻¹)
  have hD_mul_inv : ∀ j, D j * Dinv j = 1 := fun j => by
    have h := (hD_unit j).unit.mul_inv
    simpa [Dinv, (hD_unit j).unit_spec] using h
  have hD_inv_mul : ∀ j, Dinv j * D j = 1 := fun j => by
    have h := (hD_unit j).unit.inv_mul
    simpa [Dinv, (hD_unit j).unit_spec] using h
  let M : Fin h → Matrix (Fin d) (Fin d) ℝ := fun j => Dinv j * D' (π j)
  refine ⟨(π, M), ⟨ha_match, ?_, ?_, ?_⟩, ?_⟩
  · intro j
    dsimp only [M]
    rw [← Matrix.mul_assoc, hD_mul_inv j, Matrix.one_mul]
  · intro j
    dsimp only [M]
    rw [Matrix.mul_assoc, hb_match j, ← Matrix.mul_assoc, hD_inv_mul j, Matrix.one_mul]
  · intro j
    dsimp only [M, Dinv]
    exact IsUnit.mul ((hD_unit j).unit⁻¹.isUnit) (hD'_unit j)
  · rintro ⟨π', M'⟩ ⟨ha', hDM', _, _⟩
    have hπ : π' = π := by
      ext j
      have h1 : a' (π' j) = a j := ha' j
      have h2 : a' (π (π.symm (π' j))) = a (π.symm (π' j)) := ha_match (π.symm (π' j))
      rw [Equiv.apply_symm_apply] at h2
      have h3 : a (π.symm (π' j)) = a j := h2.symm.trans h1
      have h4 : π.symm (π' j) = j := ha_inj h3
      have h5 : π' j = π j := by simpa using congrArg π h4
      exact congrArg Fin.val h5
    refine Prod.ext hπ (funext fun j => ?_)
    dsimp only [M]
    calc M' j = 1 * M' j := (Matrix.one_mul _).symm
      _ = (Dinv j * D j) * M' j := by rw [hD_inv_mul j]
      _ = Dinv j * (D j * M' j) := Matrix.mul_assoc _ _ _
      _ = Dinv j * D' (π' j) := by rw [hDM' j]
      _ = Dinv j * D' (π j) := by rw [hπ]

end EquivariantMHAIdent
