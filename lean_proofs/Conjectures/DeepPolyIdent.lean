import Mathlib

/-!
# Eight Inequivalent Representations of a Generic Quadratic hPNN
(Usevich, Borsoi, Dérand, and Clausel, arXiv:2506.17093)

We formalize in Lean 4 + Mathlib (zero `sorry`, standard axioms only) the cube Cayley-octad
counterexample disproving the global identifiability conjecture for quadratic hPNNs of widths
`(4, 7, 7)` and every depth `L ≥ 3` in the family `(4, 7, ..., 7)`:

1. **Cube Walsh identity**:
   - `cube_walsh_identity`:
     `∑ ε ∈ {±1}³, ε₁ ε₂ ε₃ (x₀ + ε₁ x₁ + ε₂ x₂ + ε₃ x₃)² = 0` for all `x : Fin 4 → ℝ`.
2. **Uniqueness of the relation and linear independence of every 7-of-8 subset**:
   - `cube_unique_relation`: Any linear relation `∑ ε, c(ε) (cubeForm ε x)² = 0` holding for all
     `x : Fin 4 → ℝ` is a scalar multiple of `walshSign` (`∃ λ, ∀ ε, c ε = λ * walshSign ε`).
   - `cube_seven_linearIndependent`: For EVERY omitted vertex `e₀ ∈ {±1}³`, the 7 squared linear
     forms `{x ↦ (cubeForm ε x)² | ε ≠ e₀}` are linearly independent over `ℝ`.
   - `cube_seven_quadCoeff_linearIndependent`: Equivalently, their 7 coefficient vectors in `ℝ¹⁰`
     are linearly independent over `ℝ`.
3. **Eight inequivalent `(4, 7, 7)` quadratic hPNN realizations**:
   - `shallowEquiv_eval_eq`: Hidden-neuron permutation and coupled scaling `(s_j, s_j⁻²)` preserve
     the realized polynomial map `evalShallow W₁ W₂`.
   - `shallow_realization_eq`: For all 8 choices of omitted vertex `k ∈ {±1}³`, the explicit
     `(4, 7, 7)` quadratic hPNN `(cubeW₁ k, cubeW₂ k)` computes the exact same map
     `Fin 4 → Fin 7 → ℝ`.
   - `shallow_realization_inequivalent`: For `k₁ ≠ k₂`, `(cubeW₁ k₁, cubeW₂ k₁)` and
     `(cubeW₁ k₂, cubeW₂ k₂)` are NOT equivalent modulo hidden-neuron permutation and scaling.
   - `shallow_eight_realizations`: Combines the 8 pairwise-inequivalent realizations (indexed by
     `Sign3`, `Fintype.card Sign3 = 8`) into one summary theorem.
4. **Lift to arbitrary depth `L ≥ 3`**:
   - `deep_eight_inequivalent_realizations`: For every `d ≥ 1` (giving `d + 2 ≥ 3` linear layers
     in architecture `(4, 7, ..., 7)`) and every tail of `7 × 7` weight matrices `tail`, the 8
     deep networks `(cubeW₁ k, cubeW₂ k, tail)` compute the exact same polynomial map yet remain
     pairwise inequivalent under any deep equivalence preserving first-layer projective rows.

Machine-checked vs prose-only:
- Machine-checked: The cube Walsh identity, uniqueness of the relation, linear independence of
  every 7-of-8 subset (both as functions and in `ℝ¹⁰`), all 8 exact `(4, 7, 7)` realizations,
  their pairwise inequivalence, and the lift to arbitrary depth `L ≥ 3`.
- Prose-only (in `solutions/deep_poly_ident/results.md`): The Zariski-open genericity transfer
  from the cube witness to almost all real/complex parameters via Bézout's theorem and finite
  étale covers, the exact fiber count `= 8` on the generic locus, and the width-two checks.
-/

open BigOperators Finset Matrix

namespace DeepPolyIdent

/-! ### 1. Cube Sign Vectors and the Walsh Identity -/

/-- The 8 vertices of the cube `{±1}³`, represented as `Bool × Bool × Bool`. -/
abbrev Sign3 : Type := Bool × Bool × Bool

lemma card_sign3 : Fintype.card Sign3 = 8 := by decide

/-- Map `Bool` to `{+1, -1} ⊂ ℝ`. -/
def sgn (b : Bool) : ℝ :=
  if b then 1 else -1

lemma sgn_sq (b : Bool) : sgn b * sgn b = 1 := by
  cases b <;> simp [sgn]

/-- Walsh parity character `ε₁ ε₂ ε₃ ∈ {±1}`. -/
def walshSign (e : Sign3) : ℝ :=
  sgn e.1 * sgn e.2.1 * sgn e.2.2

lemma walshSign_sq (e : Sign3) : walshSign e * walshSign e = 1 := by
  rcases e with ⟨b1, b2, b3⟩
  cases b1 <;> cases b2 <;> cases b3 <;> simp [walshSign, sgn]

lemma walshSign_ne_zero (e : Sign3) : walshSign e ≠ 0 := by
  rcases e with ⟨b1, b2, b3⟩
  cases b1 <;> cases b2 <;> cases b3 <;> simp [walshSign, sgn]

/-- First-layer weight row `(1, ε₁, ε₂, ε₃) ∈ ℝ⁴` at vertex `e`. -/
def cubeRow (e : Sign3) : Fin 4 → ℝ :=
  ![1, sgn e.1, sgn e.2.1, sgn e.2.2]

/-- Linear form `ℓ_e(x) = x₀ + ε₁ x₁ + ε₂ x₂ + ε₃ x₃`. -/
def cubeForm (e : Sign3) (x : Fin 4 → ℝ) : ℝ :=
  x 0 + sgn e.1 * x 1 + sgn e.2.1 * x 2 + sgn e.2.2 * x 3

lemma cubeRow_dot (e : Sign3) (x : Fin 4 → ℝ) :
    dotProduct (cubeRow e) x = cubeForm e x := by
  unfold dotProduct cubeRow cubeForm
  simp only [Fin.sum_univ_four, Fin.isValue, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.cons_val]
  ring

lemma sum_sign3 (f : Sign3 → ℝ) :
    ∑ e : Sign3, f e =
      f (false, false, false) + f (false, false, true) +
      f (false, true, false) + f (false, true, true) +
      f (true, false, false) + f (true, false, true) +
      f (true, true, false) + f (true, true, true) := by
  simp only [Fintype.sum_prod_type, Fintype.sum_bool]
  ring

/-- **Cube Walsh identity**:
`∑_{ε ∈ {±1}³} ε₁ ε₂ ε₃ (x₀ + ε₁ x₁ + ε₂ x₂ + ε₃ x₃)² = 0` for every `x : Fin 4 → ℝ`. -/
theorem cube_walsh_identity (x : Fin 4 → ℝ) :
    ∑ e : Sign3, walshSign e * (cubeForm e x) ^ 2 = 0 := by
  rw [sum_sign3]
  simp only [walshSign, cubeForm, sgn, ↓reduceIte, Bool.false_eq_true]
  ring

/-! ### 2. Uniqueness of the Relation and 7-of-8 Linear Independence -/

/-- **Uniqueness of the linear relation among the 8 cube squares**: if
`∑ e, c e * (cubeForm e x)² = 0` for all `x : Fin 4 → ℝ`, then `c` is a scalar multiple of
`walshSign`. -/
theorem cube_unique_relation (c : Sign3 → ℝ)
    (hrel : ∀ x : Fin 4 → ℝ, ∑ e : Sign3, c e * (cubeForm e x) ^ 2 = 0) :
    ∃ lam : ℝ, ∀ e : Sign3, c e = lam * walshSign e := by
  -- Evaluate the vanishing quadratic form at 7 symmetric test pairs to extract the 7 Walsh sums
  have h000 := hrel ![1, 0, 0, 0]
  have h1p := hrel ![1, 1, 0, 0]
  have h1m := hrel ![1, -1, 0, 0]
  have h2p := hrel ![1, 0, 1, 0]
  have h2m := hrel ![1, 0, -1, 0]
  have h3p := hrel ![1, 0, 0, 1]
  have h3m := hrel ![1, 0, 0, -1]
  have h12p := hrel ![0, 1, 1, 0]
  have h12m := hrel ![0, 1, -1, 0]
  have h13p := hrel ![0, 1, 0, 1]
  have h13m := hrel ![0, 1, 0, -1]
  have h23p := hrel ![0, 0, 1, 1]
  have h23m := hrel ![0, 0, 1, -1]
  rw [sum_sign3] at h000 h1p h1m h2p h2m h3p h3m h12p h12m h13p h13m h23p h23m
  simp only [cubeForm, sgn, ↓reduceIte, Bool.false_eq_true, Fin.isValue,
    Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val] at h000 h1p h1m h2p h2m h3p h3m h12p h12m h13p h13m h23p h23m
  refine ⟨c (true, true, true), fun e => ?_⟩
  rcases e with ⟨b1, b2, b3⟩
  cases b1 <;> cases b2 <;> cases b3 <;>
    simp only [walshSign, sgn, ↓reduceIte, Bool.false_eq_true] <;>
    linarith

/-- **Linear independence of every 7-of-8 subset (as functions `(Fin 4 → ℝ) → ℝ`)**:
omitting ANY vertex `e₀ ∈ {±1}³`, the remaining 7 squared linear forms are linearly independent
over `ℝ`. -/
theorem cube_seven_linearIndependent (e₀ : Sign3) :
    LinearIndependent ℝ (fun e : {e : Sign3 // e ≠ e₀} => fun x : Fin 4 → ℝ => (cubeForm e.1 x) ^ 2) := by
  rw [Fintype.linearIndependent_iff]
  intro g hg
  let c : Sign3 → ℝ := fun e => if h : e = e₀ then 0 else g ⟨e, h⟩
  have hsum : ∀ x : Fin 4 → ℝ, ∑ e : Sign3, c e * (cubeForm e x) ^ 2 = 0 := by
    intro x
    have h1 : ∑ e : Sign3, c e * (cubeForm e x) ^ 2 =
        ∑ e : {e : Sign3 // e ≠ e₀}, g e * (cubeForm e.1 x) ^ 2 := by
      rw [← Finset.add_sum_erase _ _ (Finset.mem_univ e₀)]
      have hc0 : c e₀ = 0 := by simp [c]
      rw [hc0, zero_mul, zero_add]
      symm
      refine Finset.sum_bij (fun e _ => e.1) ?_ ?_ ?_ ?_
      · intro e _
        exact Finset.mem_erase.2 ⟨e.2, Finset.mem_univ _⟩
      · intro e₁ _ e₂ _ h
        exact Subtype.ext h
      · intro e he
        exact ⟨⟨e, (Finset.mem_erase.1 he).1⟩, Finset.mem_univ _, rfl⟩
      · intro e _
        simp [c, e.2]
    rw [h1]
    have hx := congrFun hg x
    simpa [Finset.sum_apply, Pi.smul_apply, smul_eq_mul] using hx
  obtain ⟨lam, hlam⟩ := cube_unique_relation c hsum
  have hlam0 : lam = 0 := by
    have h0 := hlam e₀
    have hc0 : c e₀ = 0 := by simp [c]
    rw [hc0] at h0
    rcases mul_eq_zero.1 h0.symm with h | h
    · exact h
    · exact absurd h (walshSign_ne_zero e₀)
  intro e
  have he := hlam e.1
  rw [hlam0, zero_mul] at he
  simpa [c, e.2] using he

/-- Ten-coordinate quadratic monomial representation of `(a · x)²` in basis
`(x₀², x₁², x₂², x₃², 2 x₀ x₁, 2 x₀ x₂, 2 x₀ x₃, 2 x₁ x₂, 2 x₁ x₃, 2 x₂ x₃)`. -/
def quadCoeff (a : Fin 4 → ℝ) : Fin 10 → ℝ :=
  ![a 0 * a 0, a 1 * a 1, a 2 * a 2, a 3 * a 3,
    a 0 * a 1, a 0 * a 2, a 0 * a 3,
    a 1 * a 2, a 1 * a 3, a 2 * a 3]

/-- **Linear independence of every 7-of-8 subset in `ℝ¹⁰`**: for every `e₀ : Sign3`, the 7 vectors
`{quadCoeff (cubeRow e) | e ≠ e₀}` are linearly independent in `Fin 10 → ℝ`. -/
theorem cube_seven_quadCoeff_linearIndependent (e₀ : Sign3) :
    LinearIndependent ℝ (fun e : {e : Sign3 // e ≠ e₀} => quadCoeff (cubeRow e.1)) := by
  rw [Fintype.linearIndependent_iff]
  intro g hg
  have hfun : ∑ e : {e : Sign3 // e ≠ e₀}, g e • (fun x : Fin 4 → ℝ => (cubeForm e.1 x) ^ 2) = 0 := by
    funext x
    simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, Pi.zero_apply]
    have hcoord : ∀ idx : Fin 10, ∑ e : {e : Sign3 // e ≠ e₀}, g e * quadCoeff (cubeRow e.1) idx = 0 := by
      intro idx
      have := congrFun hg idx
      simpa [Finset.sum_apply, Pi.smul_apply, smul_eq_mul] using this
    have hexpand : ∀ e : {e : Sign3 // e ≠ e₀},
        (cubeForm e.1 x) ^ 2 =
          quadCoeff (cubeRow e.1) 0 * (x 0 * x 0) +
          quadCoeff (cubeRow e.1) 1 * (x 1 * x 1) +
          quadCoeff (cubeRow e.1) 2 * (x 2 * x 2) +
          quadCoeff (cubeRow e.1) 3 * (x 3 * x 3) +
          quadCoeff (cubeRow e.1) 4 * (2 * x 0 * x 1) +
          quadCoeff (cubeRow e.1) 5 * (2 * x 0 * x 2) +
          quadCoeff (cubeRow e.1) 6 * (2 * x 0 * x 3) +
          quadCoeff (cubeRow e.1) 7 * (2 * x 1 * x 2) +
          quadCoeff (cubeRow e.1) 8 * (2 * x 1 * x 3) +
          quadCoeff (cubeRow e.1) 9 * (2 * x 2 * x 3) := by
      intro e
      unfold cubeForm quadCoeff cubeRow
      simp only [Fin.isValue, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val]
      ring
    simp_rw [hexpand]
    have h0 := hcoord 0; have h1 := hcoord 1; have h2 := hcoord 2; have h3 := hcoord 3
    have h4 := hcoord 4; have h5 := hcoord 5; have h6 := hcoord 6; have h7 := hcoord 7
    have h8 := hcoord 8; have h9 := hcoord 9
    calc ∑ e : {e : Sign3 // e ≠ e₀}, g e * _
        = (∑ e, g e * quadCoeff (cubeRow e.1) 0) * (x 0 * x 0) +
          (∑ e, g e * quadCoeff (cubeRow e.1) 1) * (x 1 * x 1) +
          (∑ e, g e * quadCoeff (cubeRow e.1) 2) * (x 2 * x 2) +
          (∑ e, g e * quadCoeff (cubeRow e.1) 3) * (x 3 * x 3) +
          (∑ e, g e * quadCoeff (cubeRow e.1) 4) * (2 * x 0 * x 1) +
          (∑ e, g e * quadCoeff (cubeRow e.1) 5) * (2 * x 0 * x 2) +
          (∑ e, g e * quadCoeff (cubeRow e.1) 6) * (2 * x 0 * x 3) +
          (∑ e, g e * quadCoeff (cubeRow e.1) 7) * (2 * x 1 * x 2) +
          (∑ e, g e * quadCoeff (cubeRow e.1) 8) * (2 * x 1 * x 3) +
          (∑ e, g e * quadCoeff (cubeRow e.1) 9) * (2 * x 2 * x 3) := by
            simp_rw [mul_add, Finset.sum_add_distrib, ← mul_assoc, ← Finset.sum_mul]
      _ = 0 := by rw [h0, h1, h2, h3, h4, h5, h6, h7, h8, h9]; ring
  exact (Fintype.linearIndependent_iff.1 (cube_seven_linearIndependent e₀)) g hfun

/-! ### 3. Eight Inequivalent `(4, 7, 7)` Quadratic hPNN Realizations -/

/-- Evaluation of a 2-layer `(4, 7, 7)` quadratic hPNN: `x ↦ W₂ * (W₁ * x)^⊙2`. -/
def evalShallow (W₁ : Matrix (Fin 7) (Fin 4) ℝ) (W₂ : Matrix (Fin 7) (Fin 7) ℝ)
    (x : Fin 4 → ℝ) : Fin 7 → ℝ :=
  W₂.mulVec (fun j => (W₁.mulVec x j) ^ 2)

/-- Standard hidden-neuron equivalence of two `(4, 7, 7)` quadratic hPNNs: permutation `σ` and
nonzero scalings `s_j` on rows of `W₁` coupled with `s_j⁻²` on columns of `W₂`. -/
def ShallowEquiv (W₁ : Matrix (Fin 7) (Fin 4) ℝ) (W₂ : Matrix (Fin 7) (Fin 7) ℝ)
    (W₁' : Matrix (Fin 7) (Fin 4) ℝ) (W₂' : Matrix (Fin 7) (Fin 7) ℝ) : Prop :=
  ∃ (σ : Equiv.Perm (Fin 7)) (s : Fin 7 → ℝ),
    (∀ j, s j ≠ 0) ∧
    (∀ j, W₁' j = s j • W₁ (σ j)) ∧
    (∀ i j, W₂' i j = (s j ^ 2)⁻¹ * W₂ i (σ j))

/-- Equivalent `(4, 7, 7)` networks realize the exact same polynomial map. -/
theorem shallowEquiv_eval_eq {W₁ W₁' : Matrix (Fin 7) (Fin 4) ℝ}
    {W₂ W₂' : Matrix (Fin 7) (Fin 7) ℝ} (heq : ShallowEquiv W₁ W₂ W₁' W₂') :
    evalShallow W₁' W₂' = evalShallow W₁ W₂ := by
  rcases heq with ⟨σ, s, hs, hW₁, hW₂⟩
  funext x i
  unfold evalShallow Matrix.mulVec dotProduct
  have hterm : ∀ j : Fin 7,
      W₂' i j * (∑ c : Fin 4, W₁' j c * x c) ^ 2 =
      W₂ i (σ j) * (∑ c : Fin 4, W₁ (σ j) c * x c) ^ 2 := by
    intro j
    have hrow : ∑ c : Fin 4, W₁' j c * x c = s j * ∑ c : Fin 4, W₁ (σ j) c * x c := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun c _ => ?_
      have hc := congrFun (hW₁ j) c
      simp only [Pi.smul_apply, smul_eq_mul] at hc
      rw [hc]; ring
    rw [hW₂ i j, hrow, mul_pow]
    field_simp [hs j]
  simp_rw [hterm]
  exact Equiv.sum_comp σ (fun j => W₂ i j * (∑ c : Fin 4, W₁ j c * x c) ^ 2)

/-- Fixed enumeration of the 7 vertices other than `eBase = (false, false, false)`. -/
def baseVert : Fin 7 → Sign3
  | 0 => (false, false, true)
  | 1 => (false, true, false)
  | 2 => (false, true, true)
  | 3 => (true, false, false)
  | 4 => (true, false, true)
  | 5 => (true, true, false)
  | 6 => (true, true, true)

def eBase : Sign3 := (false, false, false)

lemma baseVert_injective : Function.Injective baseVert := by
  intro i j h
  fin_cases i <;> fin_cases j <;> simp [baseVert] at h ⊢

lemma baseVert_ne_eBase (j : Fin 7) : baseVert j ≠ eBase := by
  fin_cases j <;> decide

lemma sign3_eq_eBase_or_baseVert (e : Sign3) : e = eBase ∨ ∃ j : Fin 7, baseVert j = e := by
  rcases e with ⟨b1, b2, b3⟩
  cases b1 <;> cases b2 <;> cases b3
  · left; rfl
  · right; exact ⟨0, rfl⟩
  · right; exact ⟨1, rfl⟩
  · right; exact ⟨2, rfl⟩
  · right; exact ⟨3, rfl⟩
  · right; exact ⟨4, rfl⟩
  · right; exact ⟨5, rfl⟩
  · right; exact ⟨6, rfl⟩

/-- For each omitted vertex `k : Sign3`, the 7 hidden vertices of realization `k`:
if `baseVert j = k`, substitute `eBase`; otherwise keep `baseVert j`. -/
def hiddenVert (k : Sign3) (j : Fin 7) : Sign3 :=
  if baseVert j = k then eBase else baseVert j

lemma hiddenVert_ne (k : Sign3) (j : Fin 7) : hiddenVert k j ≠ k := by
  unfold hiddenVert
  split_ifs with h
  · rw [← h]; exact (baseVert_ne_eBase j).symm
  · exact h

lemma hiddenVert_injective (k : Sign3) : Function.Injective (hiddenVert k) := by
  intro i j hij
  unfold hiddenVert at hij
  split_ifs at hij with hi hj hj
  · exact baseVert_injective (hi.trans hj.symm)
  · exact absurd hij.symm (baseVert_ne_eBase j)
  · exact absurd hij (baseVert_ne_eBase i)
  · exact baseVert_injective hij

lemma hiddenVert_surj_ne (k : Sign3) {e : Sign3} (he : e ≠ k) :
    ∃ j : Fin 7, hiddenVert k j = e := by
  rcases sign3_eq_eBase_or_baseVert e with rfl | ⟨j, rfl⟩
  · rcases sign3_eq_eBase_or_baseVert k with rfl | ⟨j₀, rfl⟩
    · exact absurd rfl he
    · exact ⟨j₀, by simp [hiddenVert]⟩
  · refine ⟨j, ?_⟩
    simp [hiddenVert, he]

/-- First-layer `7 × 4` weight matrix for realization `k : Sign3`. -/
def cubeW₁ (k : Sign3) : Matrix (Fin 7) (Fin 4) ℝ :=
  fun j => cubeRow (hiddenVert k j)

/-- Second-layer `7 × 7` weight matrix for realization `k : Sign3` so that
`evalShallow (cubeW₁ k) (cubeW₂ k) x i = (cubeForm (baseVert i) x)²` for every `i : Fin 7`. -/
def cubeW₂ (k : Sign3) : Matrix (Fin 7) (Fin 7) ℝ :=
  fun i j =>
    if baseVert i = k then
      -(walshSign k * walshSign (hiddenVert k j))
    else
      if i = j then 1 else 0

lemma cubeRow_eq_smul_iff {e₁ e₂ : Sign3} {s : ℝ} :
    cubeRow e₁ = s • cubeRow e₂ ↔ s = 1 ∧ e₁ = e₂ := by
  constructor
  · intro h
    have h0 := congrFun h 0
    have h1 := congrFun h 1
    have h2 := congrFun h 2
    have h3 := congrFun h 3
    simp only [cubeRow, Pi.smul_apply, smul_eq_mul, Fin.isValue, Matrix.cons_val_zero,
      Matrix.cons_val_one, Matrix.cons_val] at h0 h1 h2 h3
    have hs : s = 1 := by linarith
    subst hs
    refine ⟨rfl, ?_⟩
    rcases e₁ with ⟨a1, a2, a3⟩
    rcases e₂ with ⟨b1, b2, b3⟩
    have hsgn : ∀ a b : Bool, sgn a = 1 * sgn b → a = b := by
      intro a b hab; cases a <;> cases b <;> simp [sgn] at hab ⊢ <;> linarith
    exact Prod.ext (hsgn a1 b1 h1) (Prod.ext (hsgn a2 b2 h2) (hsgn a3 b3 h3))
  · rintro ⟨rfl, rfl⟩
    simp

/-- **All 8 realizations compute the exact same `(4, 7, 7)` quadratic hPNN map**. -/
theorem shallow_realization_eq (k : Sign3) (x : Fin 4 → ℝ) :
    evalShallow (cubeW₁ k) (cubeW₂ k) x = fun i : Fin 7 => (cubeForm (baseVert i) x) ^ 2 := by
  funext i
  have hmul : ∀ j : Fin 7, (cubeW₁ k).mulVec x j = cubeForm (hiddenVert k j) x :=
    fun j => cubeRow_dot (hiddenVert k j) x
  simp only [evalShallow, hmul]
  unfold Matrix.mulVec dotProduct
  by_cases hik : baseVert i = k
  · -- Row `i` corresponds to the omitted vertex `k`: reconstruct `(cubeForm k x)²` via Walsh identity
    simp only [cubeW₂, hik, ↓reduceIte]
    have hsum_all : ∑ e : Sign3, walshSign e * (cubeForm e x) ^ 2 = 0 := cube_walsh_identity x
    have hsplit : ∑ e : Sign3, walshSign e * (cubeForm e x) ^ 2 =
        walshSign k * (cubeForm k x) ^ 2 +
          ∑ j : Fin 7, walshSign (hiddenVert k j) * (cubeForm (hiddenVert k j) x) ^ 2 := by
      rw [← Finset.add_sum_erase _ _ (Finset.mem_univ k)]
      congr 1
      symm
      refine Finset.sum_bij (fun j _ => hiddenVert k j) ?_ ?_ ?_ ?_
      · intro j _
        exact Finset.mem_erase.2 ⟨hiddenVert_ne k j, Finset.mem_univ _⟩
      · intro j₁ _ j₂ _ h
        exact hiddenVert_injective k h
      · intro e he
        obtain ⟨j, hj⟩ := hiddenVert_surj_ne k (Finset.mem_erase.1 he).1
        exact ⟨j, Finset.mem_univ j, hj⟩
      · intro j _; rfl
    have hwalsh_sq : walshSign k * walshSign k = 1 := walshSign_sq k
    have hmul_k : walshSign k * (∑ e : Sign3, walshSign e * (cubeForm e x) ^ 2) = 0 := by
      rw [hsum_all, mul_zero]
    rw [hsplit, mul_add, ← mul_assoc, hwalsh_sq, one_mul, Finset.mul_sum] at hmul_k
    have hrew : ∑ j : Fin 7, -(walshSign k * walshSign (hiddenVert k j)) *
          (cubeForm (hiddenVert k j) x) ^ 2 =
        -(∑ j : Fin 7, walshSign k * (walshSign (hiddenVert k j) *
          (cubeForm (hiddenVert k j) x) ^ 2)) := by
      rw [← Finset.sum_neg_distrib]
      refine Finset.sum_congr rfl fun j _ => by ring
    rw [hrew]
    linarith
  · -- Row `i` is not omitted: `hiddenVert k i = baseVert i` and row `i` of `cubeW₂ k` is `e_i`
    simp only [cubeW₂, hik, ↓reduceIte, ite_mul, one_mul, zero_mul,
      Finset.sum_ite_eq, Finset.mem_univ, ↓reduceIte]
    simp [hiddenVert, hik]

/-- **Pairwise inequivalence of the 8 realizations**: if `k₁ ≠ k₂`, no hidden-neuron permutation
and scaling maps `cubeW₁ k₁` to `cubeW₁ k₂`. -/
theorem firstLayer_inequivalent {k₁ k₂ : Sign3} (hne : k₁ ≠ k₂) :
    ¬ ∃ (σ : Equiv.Perm (Fin 7)) (s : Fin 7 → ℝ),
      (∀ j, s j ≠ 0) ∧ (∀ j, cubeW₁ k₂ j = s j • cubeW₁ k₁ (σ j)) := by
  rintro ⟨σ, s, _, hW₁⟩
  -- Since `k₁ ≠ k₂`, `k₁` appears as a hidden vertex in realization `k₂`
  obtain ⟨j, hj⟩ := hiddenVert_surj_ne k₂ hne
  have hrow := hW₁ j
  unfold cubeW₁ at hrow
  rw [hj] at hrow
  have heq := (cubeRow_eq_smul_iff.1 hrow).2
  exact hiddenVert_ne k₁ (σ j) heq.symm

/-- Pairwise inequivalence in the full `(4, 7, 7)` shallow equivalence relation. -/
theorem shallow_realization_inequivalent {k₁ k₂ : Sign3} (hne : k₁ ≠ k₂) :
    ¬ ShallowEquiv (cubeW₁ k₁) (cubeW₂ k₁) (cubeW₁ k₂) (cubeW₂ k₂) := by
  rintro ⟨σ, s, hs, hW₁, _⟩
  exact firstLayer_inequivalent hne ⟨σ, s, hs, hW₁⟩

/-- **Summary theorem (depth `L = 2`)**: the 8 networks `(cubeW₁ k, cubeW₂ k)` indexed by
`k : Sign3` (`Fintype.card Sign3 = 8`) all realize the same `(4, 7, 7)` quadratic hPNN map and are
pairwise inequivalent. -/
theorem shallow_eight_realizations :
    Fintype.card Sign3 = 8 ∧
    (∀ k₁ k₂ : Sign3, evalShallow (cubeW₁ k₁) (cubeW₂ k₁) = evalShallow (cubeW₁ k₂) (cubeW₂ k₂)) ∧
    (∀ k₁ k₂ : Sign3, k₁ ≠ k₂ → ¬ ShallowEquiv (cubeW₁ k₁) (cubeW₂ k₁) (cubeW₁ k₂) (cubeW₂ k₂)) := by
  refine ⟨card_sign3, ?_, fun _ _ h => shallow_realization_inequivalent h⟩
  intro k₁ k₂
  funext x
  rw [shallow_realization_eq k₁ x, shallow_realization_eq k₂ x]

/-! ### 4. Lift to Arbitrary Depth `L ≥ 3` (`(4, 7, ..., 7)`) -/

/-- Apply `d` additional quadratic layers `(tail 0, ..., tail (d - 1))` of width `7`. -/
def evalTail : {d : ℕ} → (Fin d → Matrix (Fin 7) (Fin 7) ℝ) → (Fin 7 → ℝ) → (Fin 7 → ℝ)
  | 0, _, v => v
  | _ + 1, tail, v =>
      let nextV := (tail 0).mulVec (fun j => (v j) ^ 2)
      evalTail (fun ℓ => tail ℓ.succ) nextV

/-- Evaluation of a `(d + 2)`-layer quadratic hPNN of widths `(4, 7, ..., 7)`. -/
def evalDeep {d : ℕ} (W₁ : Matrix (Fin 7) (Fin 4) ℝ) (W₂ : Matrix (Fin 7) (Fin 7) ℝ)
    (tail : Fin d → Matrix (Fin 7) (Fin 7) ℝ) (x : Fin 4 → ℝ) : Fin 7 → ℝ :=
  evalTail tail (evalShallow W₁ W₂ x)

/-- Necessary condition of deep hidden-neuron equivalence at any depth: first-layer rows agree up
to permutation and nonzero rescaling. -/
def DeepFirstLayerEquiv (W₁ W₁' : Matrix (Fin 7) (Fin 4) ℝ) : Prop :=
  ∃ (σ : Equiv.Perm (Fin 7)) (s : Fin 7 → ℝ),
    (∀ j, s j ≠ 0) ∧ (∀ j, W₁' j = s j • W₁ (σ j))

/-- **Lift to arbitrary depth `L = d + 2 ≥ 3`**: for every `d ≥ 1` and every sequence of higher
layer matrices `tail : Fin d → Matrix (Fin 7) (Fin 7) ℝ`, the 8 deep networks
`(cubeW₁ k, cubeW₂ k, tail)` realize the exact same polynomial map yet are pairwise inequivalent. -/
theorem deep_eight_inequivalent_realizations {d : ℕ} (tail : Fin d → Matrix (Fin 7) (Fin 7) ℝ) :
    Fintype.card Sign3 = 8 ∧
    (∀ k₁ k₂ : Sign3, evalDeep (cubeW₁ k₁) (cubeW₂ k₁) tail =
      evalDeep (cubeW₁ k₂) (cubeW₂ k₂) tail) ∧
    (∀ k₁ k₂ : Sign3, k₁ ≠ k₂ → ¬ DeepFirstLayerEquiv (cubeW₁ k₁) (cubeW₁ k₂)) := by
  refine ⟨card_sign3, ?_, fun _ _ h => firstLayer_inequivalent h⟩
  intro k₁ k₂
  funext x
  unfold evalDeep
  rw [shallow_realization_eq k₁ x, shallow_realization_eq k₂ x]

end DeepPolyIdent
