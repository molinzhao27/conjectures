import Mathlib

/-!
# Resolution of Curto–Morrison CTLN Open Questions 11, 13, and 14
(Carina Curto and Katherine Morrison, *Graph rules for recurrent neural network dynamics*,
arXiv:2301.12638 / Notices of the AMS 70(4), Apr. 2023)

We formalize in Lean 4 + Mathlib (zero `sorry`, standard axioms only):
1. **Open Question 13 ("Proper sources die" — Complete algebraic & ODE-inequality core)**:
   - `ctln_sum_lower_bound`: Total activity satisfies `∑ ([u_i]_+ - x_i) ≥ n*θ - (1+(n-1)*c)*S`.
   - `ctln_asymptotic_D_neg`: For `n ≥ 1, δ > 0, θ > 0`, the floor `n*θ/(1+(n-1)*c)` strictly
     exceeds `θ/c` (so eventually `D = θ - c*S ≤ -η < 0`).
   - `oq13_source_decay_trap`: When `D ≤ -η` and `x_j ≤ η/c`, preactivation `D + c*x_j ≤ 0`,
     yielding exact exponential decay `x_j' = -x_j`.
   - `oq13_diff_on_E`, `oq13_diff_off_E`: For an out-neighbor `j → k`, `d = x_k - x_j` satisfies
     `d' ≥ δ*x_k + ε*x_j` when `u_j > 0` and `d' ≥ -d` when `u_j ≤ 0`.
   - `oq13_ratio_growth_on_E`, `oq13_ratio_nonneg_off_E`: For `x_k ≥ x_j > 0` and `D < 0`,
     the ratio `r = x_k / x_j` satisfies `r' ≥ δ + ε` on `E` and `r' ≥ 0` off `E`.
2. **Open Question 11 (Constant-sum rigidity lemma — Nonnegative chamber defect)**:
   - `oq11_q_nonneg_and_zero_iff`: For any active mask `σ` and preactivation `y`,
     `q_i = max 0 (y i) - (if σ i then y i else 0) ≥ 0`, with `q = 0` iff `y` lies in the
     closed linear chamber `σ`.
3. **Open Question 14 (Strict interpolation refutation — 5-node TLN & 7-node CTLN)**:
   - `oq14_params_legal`: `ε = 3/10, δ = 3/2` satisfy `0 < ε < δ / (δ + 1)`.
   - `oq14_no_self_loops`, `oq14_non_isomorphic`: Both 7-node CTLN graphs are loopless and
     non-isomorphic (block-`A` nonedge selection matrices `P` and `Q` have different numbers
     of zero columns: 2 vs. 1).
   - `oq14_synchronous_reduction`: Exact matrix identity `W_s * L = L * W_red` for all `s : ℝ`.
   - `oq14_static_tln_fixed_point`: Exact shared fixed point `x* = (2/5) • 1` for the 5-node
     competitive TLN segment `W_s = -(3/2) * ((1-s) P + s Q)` for all `s : ℝ`.
   - `oq14_charpoly_P`, `oq14_charpoly_Q`, `oq14_charpoly_midpoint`: Exact characteristic
     polynomials of `P`, `Q` (`X^2 * (X - 1) * (X^2 + X + 1)`) and `(P + Q)/2`
     (`(X - 1) * (16 X^4 + 16 X^3 + 8 X^2 + 2 X - 1) / 16`).
   - `oq14_midpoint_real_root`: By IVT on `[-716/1000, -715/1000]`, the midpoint quartic has
     a real root `μ ∈ (-0.716, -0.715)`.
   - `oq14_floquet_exponent_flip`, `oq14_static_abscissa_flip`: For any block-on fraction
     `p ∈ (0.5074, 0.5076)` and `μ ∈ (-0.716, -0.715)`, endpoint normal Floquet exponent is
     strictly negative (`-1 + 1.6 * p < 0`) while midpoint exponent is strictly positive
     (`-1 + p * (0.7 - 1.8 * μ) > 0`), and similarly for the static TLN abscissa (`-1/4 < 0`
     vs. `-1 - 1.5 * μ > 0`).

The Arb 448-bit interval-arithmetic periodic-orbit certificates (Gaudi period/excursion and
7-node quotient cycle `p ∈ (0.5074, 0.5076)`), ODE integration steps, and Baire/variation-of-
constants arguments are proved in `solutions/ctln_dynamics/solution.md` and certified in
`solutions/ctln_dynamics/explore.py` (`uv run solutions/ctln_dynamics/explore.py certificate`).
-/

open Matrix Finset BigOperators Polynomial

namespace CTLNDynamics

/-- Standard ReLU `max 0 u`. -/
def relu (u : ℝ) : ℝ := max 0 u

/-! ### 1. Open Question 13: Proper Sources Die -/

/-- **Total-activity differential inequality**:
If `x ≥ 0` and every off-diagonal weight satisfies `W i j ≥ -c` (`W i i = 0`), then
`∑_i ([θ + (W x)_i]_+ - x_i) ≥ n * θ - (1 + (n - 1) * c) * ∑_i x_i`. -/
theorem ctln_sum_lower_bound {n : ℕ} (c θ : ℝ)
    (W : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → ℝ)
    (hx : ∀ i, 0 ≤ x i)
    (hdiag : ∀ i, W i i = 0)
    (hoff : ∀ i j, i ≠ j → -c ≤ W i j) :
    (n : ℝ) * θ - (1 + ((n : ℝ) - 1) * c) * (∑ i : Fin n, x i) ≤
      ∑ i : Fin n, (relu (θ + W.mulVec x i) - x i) := by
  have hrow : ∀ i : Fin n, θ - c * ((∑ j : Fin n, x j) - x i) ≤ θ + W.mulVec x i := by
    intro i
    have hmul : -c * ((∑ j : Fin n, x j) - x i) ≤ W.mulVec x i := by
      simp only [Matrix.mulVec, dotProduct]
      rw [← Finset.add_sum_erase _ (fun j => W i j * x j) (Finset.mem_univ i),
        hdiag i, zero_mul, zero_add]
      have hsub : (∑ j : Fin n, x j) - x i = ∑ j ∈ Finset.univ.erase i, x j := by
        have := Finset.add_sum_erase Finset.univ x (Finset.mem_univ i)
        linarith
      rw [hsub, Finset.mul_sum]
      refine Finset.sum_le_sum fun j hj => ?_
      have hne : i ≠ j := (Finset.ne_of_mem_erase hj).symm
      exact mul_le_mul_of_nonneg_right (hoff i j hne) (hx j)
    linarith
  have hterm : ∀ i : Fin n,
      θ - c * ((∑ j : Fin n, x j) - x i) - x i ≤ relu (θ + W.mulVec x i) - x i := by
    intro i
    have h1 := hrow i
    have h2 : θ + W.mulVec x i ≤ relu (θ + W.mulVec x i) := le_max_right 0 _
    linarith
  have hsum := Finset.sum_le_sum fun i (_ : i ∈ Finset.univ) => hterm i
  have hlhs :
      ∑ i : Fin n, (θ - c * ((∑ j : Fin n, x j) - x i) - x i) =
        (n : ℝ) * θ - (1 + ((n : ℝ) - 1) * c) * (∑ j : Fin n, x j) := by
    simp only [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
      nsmul_eq_mul, ← Finset.mul_sum]
    ring
  linarith

/-- **Asymptotic inhibition excess (`L > θ / c`)**:
For every `n ≥ 1`, `δ > 0` (`c = 1 + δ > 1`), and `θ > 0`,
`θ / c < n * θ / (1 + (n - 1) * c)`. -/
theorem ctln_asymptotic_D_neg {n : ℕ} (hn : 1 ≤ n) {δ θ : ℝ} (hδ : 0 < δ) (hθ : 0 < θ) :
    let c := 1 + δ
    θ / c < (n : ℝ) * θ / (1 + ((n : ℝ) - 1) * c) := by
  intro c
  have hc : 1 < c := by linarith
  have hcpos : 0 < c := by linarith
  have hn1 : (1 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  have hden : 0 < 1 + ((n : ℝ) - 1) * c := by nlinarith
  rw [div_lt_div_iff₀ hcpos hden]
  nlinarith

/-- **Permanent decay trap for a source**:
If `D ≤ -η`, `c > 0`, and `x_j ≤ η / c`, then `u_j = D + c * x_j ≤ 0`, so `relu u_j - x_j = -x_j`. -/
theorem oq13_source_decay_trap {c η D xj : ℝ} (hc : 0 < c)
    (hD : D ≤ -η) (hx : xj ≤ η / c) :
    relu (D + c * xj) - xj = -xj := by
  have hcx : c * xj ≤ η := (le_div_iff₀' hc).mp hx
  have hu : D + c * xj ≤ 0 := by linarith
  simp [relu, max_eq_left hu]

/-- **Difference inequality on `E = {D + c * x_j > 0}`**:
For an out-neighbor `j → k`, `u_k ≥ D + c * x_k + (δ + ε) * x_j` and `u_j = D + c * x_j > 0`
imply `(relu u_k - x_k) - (relu u_j - x_j) ≥ δ * x_k + ε * x_j`. -/
theorem oq13_diff_on_E {δ ε D xj xk uk : ℝ}
    (huk : D + (1 + δ) * xk + (δ + ε) * xj ≤ uk)
    (hE : 0 < D + (1 + δ) * xj) :
    δ * xk + ε * xj ≤ (relu uk - xk) - (relu (D + (1 + δ) * xj) - xj) := by
  have hrj : relu (D + (1 + δ) * xj) = D + (1 + δ) * xj := max_eq_right (le_of_lt hE)
  have hrk : uk ≤ relu uk := le_max_right 0 uk
  rw [hrj]
  linarith

/-- **Difference inequality off `E` (`D + c * x_j ≤ 0`)**:
`(relu u_k - x_k) - (relu u_j - x_j) ≥ -(x_k - x_j)`. -/
theorem oq13_diff_off_E {D c xj xk uk : ℝ} (hoff : D + c * xj ≤ 0) :
    -(xk - xj) ≤ (relu uk - xk) - (relu (D + c * xj) - xj) := by
  have hrj : relu (D + c * xj) = 0 := max_eq_left hoff
  have hrk : 0 ≤ relu uk := le_max_left 0 uk
  rw [hrj]
  linarith

/-- **Ratio growth on `E` (`r = x_k / x_j ≥ 1`, `D < 0`)**:
When `x_j > 0`, `x_k ≥ x_j`, `D ≤ 0`, `u_k ≥ D + c * x_k + (δ + ε) * x_j`, and `u_j = D + c * x_j > 0`,
the quotient-rule derivative `(relu u_k - (x_k / x_j) * relu u_j) / x_j` is at least `δ + ε`. -/
theorem oq13_ratio_growth_on_E {δ ε D xj xk uk : ℝ}
    (hxj : 0 < xj) (hle : xj ≤ xk) (hD : D ≤ 0)
    (huk : D + (1 + δ) * xk + (δ + ε) * xj ≤ uk)
    (hE : 0 < D + (1 + δ) * xj) :
    δ + ε ≤ (relu uk - (xk / xj) * relu (D + (1 + δ) * xj)) / xj := by
  have hrj : relu (D + (1 + δ) * xj) = D + (1 + δ) * xj := max_eq_right (le_of_lt hE)
  have hrk : uk ≤ relu uk := le_max_right 0 uk
  rw [hrj, le_div_iff₀ hxj]
  have hr1 : 1 ≤ xk / xj := (one_le_div hxj).mpr hle
  have hmul : (xk / xj) * (D + (1 + δ) * xj) = (xk / xj) * D + (1 + δ) * xk := by
    field_simp [ne_of_gt hxj]
  rw [hmul]
  have hDterm : (xk / xj) * D ≤ D := by nlinarith
  linarith

/-- **Ratio non-decrease off `E` (`u_j ≤ 0`)**:
`(relu u_k - r * relu u_j) / x_j ≥ 0`. -/
theorem oq13_ratio_nonneg_off_E {D c xj uk r : ℝ} (hxj : 0 < xj) (hoff : D + c * xj ≤ 0) :
    0 ≤ (relu uk - r * relu (D + c * xj)) / xj := by
  have hrj : relu (D + c * xj) = 0 := max_eq_left hoff
  have hrk : 0 ≤ relu uk := le_max_left 0 uk
  rw [hrj, mul_zero, sub_zero]
  exact div_nonneg hrk (le_of_lt hxj)

/-! ### 2. Open Question 11: Nonnegative Chamber Defect for Constant-Sum Rigidity -/

/-- **Chamber defect `q(y) = [y]_+ - P_σ y` is componentwise nonnegative and vanishes
exactly on the closed linear chamber `σ`**. -/
theorem oq11_q_nonneg_and_zero_iff {n : ℕ} (σ : Fin n → Bool) (y : Fin n → ℝ) :
    let q : Fin n → ℝ := fun i => relu (y i) - (if σ i then y i else 0)
    (∀ i, 0 ≤ q i) ∧
    (q = 0 ↔ ∀ i, (σ i = true → 0 ≤ y i) ∧ (σ i = false → y i ≤ 0)) := by
  intro q
  constructor
  · intro i
    dsimp only [q, relu]
    cases h : σ i
    · simp only [Bool.false_eq_true, ↓reduceIte, sub_zero]; exact le_max_left 0 (y i)
    · simp only [↓reduceIte, sub_nonneg]; exact le_max_right 0 (y i)
  · constructor
    · intro hq0 i
      have hi : q i = 0 := congrFun hq0 i
      dsimp only [q, relu] at hi
      constructor
      · intro ht
        rw [ht] at hi
        simp only [↓reduceIte] at hi
        have hle := le_max_left 0 (y i)
        linarith
      · intro hf
        rw [hf] at hi
        simp only [Bool.false_eq_true, ↓reduceIte, sub_zero] at hi
        have hle := le_max_right 0 (y i)
        linarith
    · intro hch
      funext i
      dsimp only [q, relu, Pi.zero_apply]
      cases h : σ i
      · simp only [Bool.false_eq_true, ↓reduceIte, sub_zero]
        exact max_eq_left ((hch i).2 h)
      · simp only [↓reduceIte]
        rw [max_eq_right ((hch i).1 h), sub_self]

/-! ### 3. Open Question 14: Exact 5-Node Static TLN & 7-Node CTLN Counterexamples -/

/-- Legal parameter check for the 7-node CTLN counterexample: `ε = 3/10, δ = 3/2`. -/
theorem oq14_params_legal :
    let ε : ℝ := 3 / 10
    let δ : ℝ := 3 / 2
    0 < ε ∧ 0 < δ ∧ ε < δ / (δ + 1) := by
  norm_num

/-- Missing-edge receiver map `f₀ = (2, 3, 1, 1, 2)` (0-based: `1, 2, 0, 0, 1`) inside block `A`. -/
def f0 : Fin 5 → Fin 5
  | 0 => 1
  | 1 => 2
  | 2 => 0
  | 3 => 0
  | 4 => 1

/-- Missing-edge receiver map `f₁ = (3, 5, 4, 1, 4)` (0-based: `2, 4, 3, 0, 3`) inside block `A`. -/
def f1 : Fin 5 → Fin 5
  | 0 => 2
  | 1 => 4
  | 2 => 3
  | 3 => 0
  | 4 => 3

/-- Row-selection matrix `P` of missing edges inside block `A` for the left CTLN. -/
def P_mat : Matrix (Fin 5) (Fin 5) ℝ :=
  Matrix.of fun i j => if j = f0 i then 1 else 0

/-- Row-selection matrix `Q` of missing edges inside block `A` for the right CTLN. -/
def Q_mat : Matrix (Fin 5) (Fin 5) ℝ :=
  Matrix.of fun i j => if j = f1 i then 1 else 0

/-- Neither `f0` nor `f1` has fixed points (no self-loops in either CTLN). -/
theorem oq14_no_self_loops : (∀ i : Fin 5, f0 i ≠ i) ∧ (∀ i : Fin 5, f1 i ≠ i) := by
  constructor <;> intro i <;> fin_cases i <;> decide

/-- `P` and `Q` are non-isomorphic directed graphs: `f0` misses 2 columns (`{3, 4}`) while `f1`
misses only 1 column (`{1}`), so no permutation `π` of `Fin 5` conjugates `f0` to `f1`. -/
theorem oq14_non_isomorphic :
    ¬ ∃ π : Equiv.Perm (Fin 5), ∀ i : Fin 5, π (f0 i) = f1 (π i) := by
  rintro ⟨π, hπ⟩
  -- Every image element `f1 (π i)` equals `π (f0 i)`, so `image f0` and `image f1` have equal card.
  have hcard : (Finset.univ.image f0).card = (Finset.univ.image f1).card := by
    have heq : (Finset.univ.image f0).image π = Finset.univ.image f1 := by
      ext y
      simp only [Finset.mem_image, Finset.mem_univ, true_and]
      constructor
      · rintro ⟨_, ⟨i, rfl⟩, rfl⟩
        exact ⟨π i, (hπ i).symm⟩
      · rintro ⟨j, rfl⟩
        refine ⟨f0 (π.symm j), ⟨π.symm j, rfl⟩, ?_⟩
        simpa using hπ (π.symm j)
    rw [← heq, Finset.card_image_of_injective _ π.injective]
  have hc0 : (Finset.univ.image f0).card = 3 := by decide
  have hc1 : (Finset.univ.image f1).card = 4 := by decide
  omega

/-- 7-node CTLN weight matrix for a block-`A` nonedge map `f` at `(ε, δ) = (3/10, 3/2)`:
off-diagonal edge weight `-1 + ε = -7/10`, nonedge weight `-1 - δ = -5/2`. -/
noncomputable def ctln7 (f : Fin 5 → Fin 5) : Matrix (Fin 7) (Fin 7) ℝ :=
  Matrix.of fun i j =>
    if i = j then 0
    else if hi : i.1 < 5 then
      if hj : j.1 < 5 then
        if (⟨j.1, hj⟩ : Fin 5) = f ⟨i.1, hi⟩ then -(5 : ℝ) / 2 else -(7 : ℝ) / 10
      else if j.1 = 5 then -(5 : ℝ) / 2
      else -(7 : ℝ) / 10
    else if i.1 = 5 then
      if j.1 < 5 then -(7 : ℝ) / 10 else -(5 : ℝ) / 2
    else
      if j.1 = 5 then -(7 : ℝ) / 10 else -(5 : ℝ) / 2

/-- 3-cell quotient weight matrix `W_red` on the synchronous manifold `x_0 = ... = x_4`. -/
noncomputable def W_red : Matrix (Fin 3) (Fin 3) ℝ :=
  !![-(23 : ℝ) / 5, -(5 : ℝ) / 2, -(7 : ℝ) / 10;
     -(7 : ℝ) / 2, 0, -(5 : ℝ) / 2;
     -(25 : ℝ) / 2, -(7 : ℝ) / 10, 0]

/-- Synchrony lift matrix `L : ℝ^{7 × 3}` sending `(a, b, c) ↦ (a, a, a, a, a, b, c)`. -/
def L_lift : Matrix (Fin 7) (Fin 3) ℝ :=
  Matrix.of fun i j =>
    if i.1 < 5 then (if j.1 = 0 then 1 else 0)
    else if i.1 = 5 then (if j.1 = 1 then 1 else 0)
    else (if j.1 = 2 then 1 else 0)

/-- **Exact synchronous reduction for the entire interpolated CTLN segment**:
For EVERY `s : ℝ`, `((1 - s) • ctln7 f0 + s • ctln7 f1) * L_lift = L_lift * W_red`. -/
theorem oq14_synchronous_reduction (s : ℝ) :
    ((1 - s) • ctln7 f0 + s • ctln7 f1) * L_lift = L_lift * W_red := by
  have h0 : ctln7 f0 * L_lift = L_lift * W_red := by
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp [ctln7, f0, L_lift, W_red, Matrix.mul_apply, Fin.sum_univ_seven, Fin.sum_univ_three] <;>
      norm_num
  have h1 : ctln7 f1 * L_lift = L_lift * W_red := by
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp [ctln7, f1, L_lift, W_red, Matrix.mul_apply, Fin.sum_univ_seven, Fin.sum_univ_three] <;>
      norm_num
  rw [Matrix.add_mul, Matrix.smul_mul, Matrix.smul_mul, h0, h1, ← add_smul]
  simp

/-- **5-node static TLN convex-hull counterexample (exact shared fixed point)**:
For every `s : ℝ`, `W_s = -(3/2) * ((1 - s) • P + s • Q)` has the exact strictly positive
fixed point `x* = (2/5) • 1` satisfying `(W_s - I) x* + 1 = 0`. -/
theorem oq14_static_tln_fixed_point (s : ℝ) :
    let Rs : Matrix (Fin 5) (Fin 5) ℝ := (1 - s) • P_mat + s • Q_mat
    let A : Matrix (Fin 5) (Fin 5) ℝ := -1 - ((3 : ℝ) / 2) • Rs
    let xstar : Fin 5 → ℝ := fun _ => (2 : ℝ) / 5
    A.mulVec xstar + (fun _ => (1 : ℝ)) = 0 := by
  intro Rs A xstar
  funext i
  fin_cases i <;>
    simp [A, Rs, P_mat, Q_mat, f0, f1, xstar, Matrix.mulVec, dotProduct, Fin.sum_univ_five] <;>
    ring

private lemma det_fin_four {R : Type*} [CommRing R] (M : Matrix (Fin 4) (Fin 4) R) :
    M.det =
      M 0 0 * (M.submatrix Fin.succ (Fin.succAbove 0)).det -
      M 0 1 * (M.submatrix Fin.succ (Fin.succAbove 1)).det +
      M 0 2 * (M.submatrix Fin.succ (Fin.succAbove 2)).det -
      M 0 3 * (M.submatrix Fin.succ (Fin.succAbove 3)).det := by
  rw [Matrix.det_succ_row_zero, Fin.sum_univ_four]
  norm_num [Fin.val_zero, Fin.val_one, Fin.val_two]
  ring

private lemma det_fin_five {R : Type*} [CommRing R] (M : Matrix (Fin 5) (Fin 5) R) :
    M.det =
      M 0 0 * (M.submatrix Fin.succ (Fin.succAbove 0)).det -
      M 0 1 * (M.submatrix Fin.succ (Fin.succAbove 1)).det +
      M 0 2 * (M.submatrix Fin.succ (Fin.succAbove 2)).det -
      M 0 3 * (M.submatrix Fin.succ (Fin.succAbove 3)).det +
      M 0 4 * (M.submatrix Fin.succ (Fin.succAbove 4)).det := by
  rw [Matrix.det_succ_row_zero, Fin.sum_univ_five]
  norm_num [Fin.val_zero, Fin.val_one, Fin.val_two]
  ring

private lemma det_fin_four_explicit {R : Type*} [CommRing R] (M : Matrix (Fin 4) (Fin 4) R) :
    M.det =
      M 0 0 * (!![M 1 1, M 1 2, M 1 3; M 2 1, M 2 2, M 2 3; M 3 1, M 3 2, M 3 3] : Matrix (Fin 3) (Fin 3) R).det -
      M 0 1 * (!![M 1 0, M 1 2, M 1 3; M 2 0, M 2 2, M 2 3; M 3 0, M 3 2, M 3 3] : Matrix (Fin 3) (Fin 3) R).det +
      M 0 2 * (!![M 1 0, M 1 1, M 1 3; M 2 0, M 2 1, M 2 3; M 3 0, M 3 1, M 3 3] : Matrix (Fin 3) (Fin 3) R).det -
      M 0 3 * (!![M 1 0, M 1 1, M 1 2; M 2 0, M 2 1, M 2 2; M 3 0, M 3 1, M 3 2] : Matrix (Fin 3) (Fin 3) R).det := by
  have h0 : M.submatrix Fin.succ (Fin.succAbove 0) =
      !![M 1 1, M 1 2, M 1 3; M 2 1, M 2 2, M 2 3; M 3 1, M 3 2, M 3 3] := by
    ext i j; fin_cases i <;> fin_cases j <;> rfl
  have h1 : M.submatrix Fin.succ (Fin.succAbove 1) =
      !![M 1 0, M 1 2, M 1 3; M 2 0, M 2 2, M 2 3; M 3 0, M 3 2, M 3 3] := by
    ext i j; fin_cases i <;> fin_cases j <;> rfl
  have h2 : M.submatrix Fin.succ (Fin.succAbove 2) =
      !![M 1 0, M 1 1, M 1 3; M 2 0, M 2 1, M 2 3; M 3 0, M 3 1, M 3 3] := by
    ext i j; fin_cases i <;> fin_cases j <;> rfl
  have h3 : M.submatrix Fin.succ (Fin.succAbove 3) =
      !![M 1 0, M 1 1, M 1 2; M 2 0, M 2 1, M 2 2; M 3 0, M 3 1, M 3 2] := by
    ext i j; fin_cases i <;> fin_cases j <;> rfl
  rw [det_fin_four, h0, h1, h2, h3]

private lemma det_fin_five_explicit {R : Type*} [CommRing R] (M : Matrix (Fin 5) (Fin 5) R) :
    M.det =
      M 0 0 * (!![M 1 1, M 1 2, M 1 3, M 1 4; M 2 1, M 2 2, M 2 3, M 2 4;
                  M 3 1, M 3 2, M 3 3, M 3 4; M 4 1, M 4 2, M 4 3, M 4 4] : Matrix (Fin 4) (Fin 4) R).det -
      M 0 1 * (!![M 1 0, M 1 2, M 1 3, M 1 4; M 2 0, M 2 2, M 2 3, M 2 4;
                  M 3 0, M 3 2, M 3 3, M 3 4; M 4 0, M 4 2, M 4 3, M 4 4] : Matrix (Fin 4) (Fin 4) R).det +
      M 0 2 * (!![M 1 0, M 1 1, M 1 3, M 1 4; M 2 0, M 2 1, M 2 3, M 2 4;
                  M 3 0, M 3 1, M 3 3, M 3 4; M 4 0, M 4 1, M 4 3, M 4 4] : Matrix (Fin 4) (Fin 4) R).det -
      M 0 3 * (!![M 1 0, M 1 1, M 1 2, M 1 4; M 2 0, M 2 1, M 2 2, M 2 4;
                  M 3 0, M 3 1, M 3 2, M 3 4; M 4 0, M 4 1, M 4 2, M 4 4] : Matrix (Fin 4) (Fin 4) R).det +
      M 0 4 * (!![M 1 0, M 1 1, M 1 2, M 1 3; M 2 0, M 2 1, M 2 2, M 2 3;
                  M 3 0, M 3 1, M 3 2, M 3 3; M 4 0, M 4 1, M 4 2, M 4 3] : Matrix (Fin 4) (Fin 4) R).det := by
  have h0 : M.submatrix Fin.succ (Fin.succAbove 0) =
      !![M 1 1, M 1 2, M 1 3, M 1 4; M 2 1, M 2 2, M 2 3, M 2 4;
         M 3 1, M 3 2, M 3 3, M 3 4; M 4 1, M 4 2, M 4 3, M 4 4] := by
    ext i j; fin_cases i <;> fin_cases j <;> rfl
  have h1 : M.submatrix Fin.succ (Fin.succAbove 1) =
      !![M 1 0, M 1 2, M 1 3, M 1 4; M 2 0, M 2 2, M 2 3, M 2 4;
         M 3 0, M 3 2, M 3 3, M 3 4; M 4 0, M 4 2, M 4 3, M 4 4] := by
    ext i j; fin_cases i <;> fin_cases j <;> rfl
  have h2 : M.submatrix Fin.succ (Fin.succAbove 2) =
      !![M 1 0, M 1 1, M 1 3, M 1 4; M 2 0, M 2 1, M 2 3, M 2 4;
         M 3 0, M 3 1, M 3 3, M 3 4; M 4 0, M 4 1, M 4 3, M 4 4] := by
    ext i j; fin_cases i <;> fin_cases j <;> rfl
  have h3 : M.submatrix Fin.succ (Fin.succAbove 3) =
      !![M 1 0, M 1 1, M 1 2, M 1 4; M 2 0, M 2 1, M 2 2, M 2 4;
         M 3 0, M 3 1, M 3 2, M 3 4; M 4 0, M 4 1, M 4 2, M 4 4] := by
    ext i j; fin_cases i <;> fin_cases j <;> rfl
  have h4 : M.submatrix Fin.succ (Fin.succAbove 4) =
      !![M 1 0, M 1 1, M 1 2, M 1 3; M 2 0, M 2 1, M 2 2, M 2 3;
         M 3 0, M 3 1, M 3 2, M 3 3; M 4 0, M 4 1, M 4 2, M 4 3] := by
    ext i j; fin_cases i <;> fin_cases j <;> rfl
  rw [det_fin_five, h0, h1, h2, h3, h4]

/-- **Exact characteristic polynomial of `P`**: `z^2 * (z - 1) * (z^2 + z + 1)`. -/
theorem oq14_charpoly_P :
    P_mat.charpoly = X ^ 2 * (X - 1) * (X ^ 2 + X + 1) := by
  have hM : P_mat.charmatrix =
      !![(X : ℝ[X]), -1, 0, 0, 0;
         0, X, -1, 0, 0;
         -1, 0, X, 0, 0;
         -1, 0, 0, X, 0;
         0, -1, 0, 0, X] := by
    ext i j
    fin_cases i <;> fin_cases j <;> simp [P_mat, f0]
  rw [Matrix.charpoly, hM, det_fin_five_explicit]
  simp only [Matrix.of_apply, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val,
    zero_mul, add_zero, sub_zero]
  simp only [det_fin_four_explicit, Matrix.of_apply, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.cons_val, zero_mul, add_zero, sub_zero]
  simp only [Matrix.det_fin_three, Matrix.of_apply, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.cons_val]
  ring

/-- **Exact characteristic polynomial of `Q`**: `z^2 * (z - 1) * (z^2 + z + 1)`. -/
theorem oq14_charpoly_Q :
    Q_mat.charpoly = X ^ 2 * (X - 1) * (X ^ 2 + X + 1) := by
  have hM : Q_mat.charmatrix =
      !![(X : ℝ[X]), 0, -1, 0, 0;
         0, X, 0, 0, -1;
         0, 0, X, -1, 0;
         -1, 0, 0, X, 0;
         0, 0, 0, -1, X] := by
    ext i j
    fin_cases i <;> fin_cases j <;> simp [Q_mat, f1]
  rw [Matrix.charpoly, hM, det_fin_five_explicit]
  simp only [Matrix.of_apply, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val,
    zero_mul, add_zero, sub_zero]
  simp only [det_fin_four_explicit, Matrix.of_apply, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.cons_val, zero_mul, add_zero, sub_zero]
  simp only [Matrix.det_fin_three, Matrix.of_apply, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.cons_val]
  ring

private lemma mid_mat_eq :
    ((1 : ℝ) / 2) • (P_mat + Q_mat) =
      !![(0 : ℝ), 1 / 2, 1 / 2, 0, 0;
         0, 0, 1 / 2, 0, 1 / 2;
         1 / 2, 0, 0, 1 / 2, 0;
         1, 0, 0, 0, 0;
         0, 1 / 2, 0, 1 / 2, 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;> (simp [P_mat, Q_mat, f0, f1]; try norm_num)

private lemma det_B_mid (y : ℝ[X]) :
    (!![y, -1, -1, 0, 0;
        0, y, -1, 0, -1;
        -1, 0, y, -1, 0;
        -2, 0, 0, y, 0;
        0, -1, 0, -1, y] : Matrix (Fin 5) (Fin 5) ℝ[X]).det =
      (y - 2) * (y ^ 4 + 2 * y ^ 3 + 2 * y ^ 2 + y - 1) := by
  set B : Matrix (Fin 5) (Fin 5) ℝ[X] :=
    !![y, -1, -1, 0, 0;
       0, y, -1, 0, -1;
       -1, 0, y, -1, 0;
       -2, 0, 0, y, 0;
       0, -1, 0, -1, y]
  let B30 : Matrix (Fin 4) (Fin 4) ℝ[X] :=
    !![-1, -1, 0, 0;
       y, -1, 0, -1;
       0, y, -1, 0;
       -1, 0, -1, y]
  let B33 : Matrix (Fin 4) (Fin 4) ℝ[X] :=
    !![y, -1, -1, 0;
       0, y, -1, -1;
       -1, 0, y, 0;
       0, -1, 0, y]
  have hB30_sub : B.submatrix (Fin.succAbove 3) (Fin.succAbove 0) = B30 := by
    ext i j; fin_cases i <;> fin_cases j <;> rfl
  have hB33_sub : B.submatrix (Fin.succAbove 3) (Fin.succAbove 3) = B33 := by
    ext i j; fin_cases i <;> fin_cases j <;> rfl
  have hdetB30 : B30.det = -y ^ 2 - 2 * y + 1 := by
    rw [Matrix.det_succ_row B30 0, Fin.sum_univ_four]
    have e0 : B30 0 0 = -1 := rfl
    have e1 : B30 0 1 = -1 := rfl
    have e2 : B30 0 2 = 0 := rfl
    have e3 : B30 0 3 = 0 := rfl
    have h0 : B30.submatrix (Fin.succAbove 0) (Fin.succAbove 0) =
        !![-1, 0, -1; y, -1, 0; 0, -1, y] := by
      ext i j; fin_cases i <;> fin_cases j <;> rfl
    have h1 : B30.submatrix (Fin.succAbove 0) (Fin.succAbove 1) =
        !![y, 0, -1; 0, -1, 0; -1, -1, y] := by
      ext i j; fin_cases i <;> fin_cases j <;> rfl
    rw [e0, e1, e2, e3, h0, h1]
    simp only [Matrix.det_fin_three, Matrix.of_apply, Matrix.cons_val_zero,
      Matrix.cons_val_one, Matrix.cons_val, Fin.val_zero, Fin.val_one, Fin.val_two]
    ring
  have hdetB33 : B33.det = y ^ 4 - 2 * y ^ 2 - y + 1 := by
    rw [Matrix.det_succ_row B33 3, Fin.sum_univ_four]
    have e0 : B33 3 0 = 0 := rfl
    have e1 : B33 3 1 = -1 := rfl
    have e2 : B33 3 2 = 0 := rfl
    have e3 : B33 3 3 = y := rfl
    have h1 : B33.submatrix (Fin.succAbove 3) (Fin.succAbove 1) =
        !![y, -1, 0; 0, -1, -1; -1, y, 0] := by
      ext i j; fin_cases i <;> fin_cases j <;> rfl
    have h3 : B33.submatrix (Fin.succAbove 3) (Fin.succAbove 3) =
        !![y, -1, -1; 0, y, -1; -1, 0, y] := by
      ext i j; fin_cases i <;> fin_cases j <;> rfl
    have hv1 : ((1 : Fin 4) : ℕ) = 1 := rfl
    have hv3 : ((3 : Fin 4) : ℕ) = 3 := rfl
    rw [e0, e1, e2, e3, h1, h3, hv1, hv3]
    simp only [Matrix.det_fin_three, Matrix.of_apply, Matrix.cons_val_zero,
      Matrix.cons_val_one, Matrix.cons_val]
    ring
  rw [Matrix.det_succ_row B 3, Fin.sum_univ_five]
  have b0 : B 3 0 = -2 := rfl
  have b1 : B 3 1 = 0 := rfl
  have b2 : B 3 2 = 0 := rfl
  have b3 : B 3 3 = y := rfl
  have b4 : B 3 4 = 0 := rfl
  have hv0 : ((0 : Fin 5) : ℕ) = 0 := rfl
  have hv3 : ((3 : Fin 5) : ℕ) = 3 := rfl
  rw [b0, b1, b2, b3, b4, hB30_sub, hB33_sub, hdetB30, hdetB33, hv0, hv3]
  simp only [mul_zero, zero_mul, add_zero]
  ring

/-- **Exact characteristic polynomial of the midpoint `(P + Q) / 2`**:
`(X - 1) * (16 X^4 + 16 X^3 + 8 X^2 + 2 X - 1) / 16`. -/
theorem oq14_charpoly_midpoint :
    (((1 : ℝ) / 2) • (P_mat + Q_mat)).charpoly =
      C ((1 : ℝ) / 16) * (X - 1) * (16 * X ^ 4 + 16 * X ^ 3 + 8 * X ^ 2 + 2 * X - 1) := by
  have h2half : (2 : ℝ[X]) * C (2⁻¹ : ℝ) = 1 := by
    rw [← map_ofNat C 2, ← map_mul]; norm_num
  have h2M : (2 : ℝ[X]) • (((1 : ℝ) / 2) • (P_mat + Q_mat)).charmatrix =
      !![(2 * X : ℝ[X]), -1, -1, 0, 0;
         0, 2 * X, -1, 0, -1;
         -1, 0, 2 * X, -1, 0;
         -2, 0, 0, 2 * X, 0;
         0, -1, 0, -1, 2 * X] := by
    rw [mid_mat_eq]
    funext i j
    fin_cases i <;> fin_cases j <;> simp [h2half]
  have hdet_eq : (32 : ℝ[X]) * (((1 : ℝ) / 2) • (P_mat + Q_mat)).charpoly =
      (2 * X - 2) * ((2 * X) ^ 4 + 2 * (2 * X) ^ 3 + 2 * (2 * X) ^ 2 + (2 * X) - 1) := by
    have hsmul : ((2 : ℝ[X]) • (((1 : ℝ) / 2) • (P_mat + Q_mat)).charmatrix).det =
        (32 : ℝ[X]) * (((1 : ℝ) / 2) • (P_mat + Q_mat)).charpoly := by
      rw [Matrix.det_smul, Fintype.card_fin, Matrix.charpoly]; norm_num
    rw [← hsmul, h2M, det_B_mid (2 * X)]
  have h32 : C ((1 : ℝ) / 32) * (32 : ℝ[X]) = 1 := by
    rw [← map_ofNat C 32, ← map_mul]; norm_num
  have h16 : C ((1 : ℝ) / 32) * (2 : ℝ[X]) = C ((1 : ℝ) / 16) := by
    rw [← map_ofNat C 2, ← map_mul]; norm_num
  calc (((1 : ℝ) / 2) • (P_mat + Q_mat)).charpoly
      = C ((1 : ℝ) / 32) * ((32 : ℝ[X]) * (((1 : ℝ) / 2) • (P_mat + Q_mat)).charpoly) := by
        rw [← mul_assoc, h32, one_mul]
    _ = C ((1 : ℝ) / 32) *
          ((2 * X - 2) * ((2 * X) ^ 4 + 2 * (2 * X) ^ 3 + 2 * (2 * X) ^ 2 + (2 * X) - 1)) := by
        rw [hdet_eq]
    _ = (C ((1 : ℝ) / 32) * 2) * (X - 1) *
          (16 * X ^ 4 + 16 * X ^ 3 + 8 * X ^ 2 + 2 * X - 1) := by ring
    _ = C ((1 : ℝ) / 16) * (X - 1) *
          (16 * X ^ 4 + 16 * X ^ 3 + 8 * X ^ 2 + 2 * X - 1) := by rw [h16]

/-- **Existence of the destabilizing real eigenvalue `μ ∈ (-0.716, -0.715)`**:
The quartic `q(z) = 16 z^4 + 16 z^3 + 8 z^2 + 2 z - 1` changes sign on `[-716/1000, -715/1000]`,
so by the Intermediate Value Theorem it has a real root `μ ∈ (-716/1000, -715/1000)`. -/
theorem oq14_midpoint_real_root :
    ∃ μ : ℝ, -(716 : ℝ) / 1000 < μ ∧ μ < -(715 : ℝ) / 1000 ∧
      16 * μ ^ 4 + 16 * μ ^ 3 + 8 * μ ^ 2 + 2 * μ - 1 = 0 := by
  let q : ℝ → ℝ := fun z => 16 * z ^ 4 + 16 * z ^ 3 + 8 * z ^ 2 + 2 * z - 1
  have hcont : Continuous q := by fun_prop
  have hlo : 0 < q (-(716 : ℝ) / 1000) := by dsimp [q]; norm_num
  have hhi : q (-(715 : ℝ) / 1000) < 0 := by dsimp [q]; norm_num
  have h0mem : (0 : ℝ) ∈ Set.Icc (q (-(715 : ℝ) / 1000)) (q (-(716 : ℝ) / 1000)) :=
    ⟨le_of_lt hhi, le_of_lt hlo⟩
  obtain ⟨μ, hμ_mem, hμ0⟩ :=
    intermediate_value_Icc' (by norm_num : -(716 : ℝ) / 1000 ≤ -(715 : ℝ) / 1000)
      hcont.continuousOn h0mem
  have hne_lo : μ ≠ -(716 : ℝ) / 1000 := by
    intro h; rw [h] at hμ0; linarith
  have hne_hi : μ ≠ -(715 : ℝ) / 1000 := by
    intro h; rw [h] at hμ0; linarith
  exact ⟨μ, lt_of_le_of_ne hμ_mem.1 hne_lo.symm, lt_of_le_of_ne hμ_mem.2 hne_hi, hμ0⟩

/-- **Floquet exponent sign flip (7-node CTLN)**:
For any certified block-on fraction `p ∈ (0.5074, 0.5076)` and any midpoint root
`μ ∈ (-0.716, -0.715)`, both endpoints are strictly attracting transversely
(`-1 + p * (0.7 - 1.8 * (-1/2)) < 0`) while the midpoint is strictly unstable
(`-1 + p * (0.7 - 1.8 * μ) > 0`). -/
theorem oq14_floquet_exponent_flip {p μ : ℝ}
    (hp_lo : (5074 : ℝ) / 10000 < p) (hp_hi : p < (5076 : ℝ) / 10000)
    (hμ_hi : μ < -(715 : ℝ) / 1000) :
    -1 + p * ((7 : ℝ) / 10 - ((9 : ℝ) / 5) * (-(1 : ℝ) / 2)) < 0 ∧
    0 < -1 + p * ((7 : ℝ) / 10 - ((9 : ℝ) / 5) * μ) := by
  constructor <;> nlinarith

/-- **Spectral abscissa sign flip (5-node static TLN)**:
At both endpoints (`Re λ ≥ -1/2`), `-1 - (3/2) * (-1/2) = -1/4 < 0`; at the midpoint
(`μ < -0.715`), `-1 - (3/2) * μ > 0`. -/
theorem oq14_static_abscissa_flip {μ : ℝ} (hμ_hi : μ < -(715 : ℝ) / 1000) :
    -1 - ((3 : ℝ) / 2) * (-(1 : ℝ) / 2) < 0 ∧
    0 < -1 - ((3 : ℝ) / 2) * μ := by
  constructor <;> linarith

end CTLNDynamics
