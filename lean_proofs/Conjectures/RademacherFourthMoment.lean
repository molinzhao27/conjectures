import Mathlib

/-!
# Sharp Cubic Rademacher Stability Constant
(Peigan Gao and Jian Qian, *Fourth-Moment Geometry of Rademacher Sums*,
arXiv:2608.17802, August 2026, Conjecture `eq:conjecture`)

We formalize in Lean 4 + Mathlib (complete proofs with no unfinished placeholders,
standard Lean foundations only) the exact algebraic, sharpness, constant-separation,
`π`-bracketed comparison, curvature, and analytical reduction cores of the proof that
for all `n ≥ 1` and all unit vectors `a ∈ S^{n-1}`,
`E |S̄_n|³ - E |∑ a_i ε_i|³ ≥ (5√3 - 6√2) Δ_n(a)`:

1. **Exact finite Rademacher cubic moments and sharp equality at `n = 3, a = (1,1,0)/√2`**:
   - `M_one`, `M_two`, `M_three`, `M_four`, `M_five`: exact evaluations
     `M₁ = 1`, `M₂ = √2`, `M₃ = 5/(2√3)`, `M₄ = 3/2`, `M₅ = 27√5/40`.
   - `sharp_n3_unit`, `sharp_n3_moment`, `sharp_n3_deficit`, `sharp_n3_delta`,
     `sharp_n3_equality`, `sharp_n3_ratio`: at `n = 3`, `a = (1, 1, 0)/√2`,
     `M₃ - E|a·ε|³ = 5/(2√3) - √2`, `Δ₃(a) = 1/6`, and the exact ratio is `5√3 - 6√2`.
   - `secant_C_2_1`, `secant_C_4_2`, `secant_C_5_4`: exact secant formulas for `n = 2, 4, 5`.
2. **Exact integer-squaring constant separation inequalities**:
   - `C_opt_lt_seven_fortieths`: `C = 5√3 - 6√2 < 7/40` (via `4751² = 22572001 < 22579200 = 2·3360²`).
   - `C_2_1_gt_seven_fortieths`: `C_{2,1} = 2(√2 - 1) > 7/40`.
   - `C_4_2_gt_seven_fortieths`: `C_{4,2} = 6 - 4√2 > 7/40`.
   - `C_5_4_gt_seven_fortieths`: `C_{5,4} = 27√5/2 - 30 > 7/40`.
3. **`π`-bracketed large-`n` / Gaussian comparison bounds (from Mathlib's `Real.pi` bounds)**:
   - `four_div_pi_curv_gt`: `(4/π)(0.64/4 - 1/32) > 0.16`.
   - `sqrt_eight_div_pi_bounds`: `1.595 < √(8/π) < 1.596`.
   - `sqrt_two_div_pi_gt`: `√(2/π) > 0.797`.
4. **Fourier majorant & curvature/far-region reduction inequalities (Sections 2–3)**:
   - `fourier_recurrence_telescope`: exact polynomial telescoping identity
     `(u + j + 1) P(j) - (u - j) P(j + 1) = (u + j + 1) j³`.
   - `cos_prod_le_weighted_rpow`: weighted AM–GM pointwise Fourier majorant
     `∏ cos(√x_i t) ≤ ∑_{x_i > 0} x_i |cos(√x_i t)|^{1/x_i}`.
   - `L0_deriv_neg`, `L0_second_deriv_neg`, `L0_strictAntiOn`: monotonicity and strict
     concavity of `L₀(x) = 1.595 (1 - x/4) ((1 - x²)/4 - x/32)` on `[0, 1/2]`.
   - `T6_gt_c`, `T7_gt_02219`, `far_odd_D_ge_6`: exact tangent-deficit bounds for even `n ≥ 6`
     and `D ≥ 6` for odd `n ≥ 7`.
   - `delta_ub_strictAnti`, `delta_ub_7`, `delta_ub_9`, `delta_ub_ge_11`: odd defect upper bounds
     `0.234`, `0.225`, `0.220`.
   - `psi_second_deriv_neg` and `far_odd_six_endpoints`: concavity of `ψ_N(D)` on `(0, 6]`
     and exact verification of all six endpoints `ψ_N(D) > e_N` for `N ∈ {7, 9, 11}`,
     `D ∈ {H(N), 6}`.
5. **Two-coordinate kernel & equalization-path endpoint comparison (Section 4)**:
   - `K_nonneg`, `K_antitoneOn`, `K_sqrt_convexOn`: nonnegativity, monotonicity, and exact
     convexity of `v ↦ K(√v)` on `[0, ∞)`.
   - `centralBinom_sqrt_succ_strictAnti` and `centralBinom_sqrt_strictMono`: exact monotonicity
     of `P_m √(m + 1)` (decreasing) and `P_k √k` (increasing) along even indices.
   - `equalization_quotient_strictMono`: strict monotonicity of `t ↦ t √(k + t²) / (1 + t)` on `(0, 1)`.
   - `secant_factor_strictAnti`, `secant_slope_strictMono`, `secant_slope_at_two`: strict
     monotonicity of the odd-to-even secant slope `P_k √k · r / (1 + r)²` (`r = √(1 + 1/k)`),
     attaining `C = 5√3 - 6√2` at `k = 2`.

Machine-checked vs. prose-only (in `solutions/rademacher_fourth_moment/proof.pdf`):
- Machine-checked: Exact finite Rademacher expectations `M₁..M₅`, sharpness at `n = 3`,
  integer-squaring constant separations, `π`-bracketed Gaussian bounds, weighted AM–GM
  Fourier majorant, Fourier telescoping identity, `L₀` and `ψ_N` concavity and all six
  far-region endpoint certificates, kernel convexity `v ↦ K(√v)`, central-binomial
  monotonicity, and the equalization-path derivative quotient and odd-to-even secant
  monotonicity.
- Prose-only / Python-certified (`explore.py`): Analytic continuation of the `|cos t|^s`
  integral identity (3) and Laplace representation (4), the 25-term alternating bound
  `A₁ > 0.64`, the 6,609 integer-arithmetic interval boxes for near-flat odd `n ≥ 7`,
  and the 47 simplicial polynomial cone certificates for `2 ≤ n ≤ 5`.
-/

open scoped BigOperators
open Finset

namespace RademacherFourthMoment

/-! ### 1. Rademacher Cubic Moments and Sharp Equality at `n = 3` -/

/-- Exact Rademacher cubic moment `E |∑_i a_i ε_i|³` over `ε ∈ {±1}^n`. -/
noncomputable def radCubicMoment {n : ℕ} (a : Fin n → ℝ) : ℝ :=
  (1 / (2 : ℝ) ^ n) *
    ∑ σ : Fin n → Bool, |∑ i : Fin n, (if σ i then (1 : ℝ) else -1) * a i| ^ 3

/-- Flat unit vector `(1/√n, ..., 1/√n) ∈ S^{n-1}`. -/
noncomputable def flatVec (n : ℕ) : Fin n → ℝ := fun _ => 1 / Real.sqrt n

/-- Flat cubic moment `M_n = E |S̄_n|³`. -/
noncomputable def M (n : ℕ) : ℝ := radCubicMoment (flatVec n)

/-- Dispersion `Δ_n(a) = ∑_i (a_i² - 1/n)²`. -/
noncomputable def Delta {n : ℕ} (a : Fin n → ℝ) : ℝ :=
  ∑ i : Fin n, (a i ^ 2 - 1 / (n : ℝ)) ^ 2

/-- Optimal dimension-free constant `C = 5√3 - 6√2`. -/
noncomputable def C_opt : ℝ := 5 * Real.sqrt 3 - 6 * Real.sqrt 2

/-- Dimension-specific optimal constants for `n = 2, 4, 5`. -/
noncomputable def C_2_1 : ℝ := 2 * (Real.sqrt 2 - 1)
noncomputable def C_4_2 : ℝ := 6 - 4 * Real.sqrt 2
noncomputable def C_5_4 : ℝ := 27 * Real.sqrt 5 / 2 - 30

/-- Exact unnormalized integer sum `∑_{σ ∈ {±1}^n} |∑ σ_i|³`. -/
def radIntSum (n : ℕ) : ℕ :=
  ∑ σ : Fin n → Bool, (Int.natAbs (∑ i : Fin n, if σ i then (1 : ℤ) else -1)) ^ 3

lemma M_eq_radIntSum (n : ℕ) (hn : 0 < n) :
    M n = (radIntSum n : ℝ) / ((2 : ℝ) ^ n * (n : ℝ) * Real.sqrt n) := by
  have hsqrt_pos : 0 < Real.sqrt n := Real.sqrt_pos.2 (Nat.cast_pos.2 hn)
  have hsq : Real.sqrt n ^ 2 = (n : ℝ) := Real.sq_sqrt (Nat.cast_nonneg n)
  unfold M radCubicMoment flatVec radIntSum
  have hterm : ∀ σ : Fin n → Bool,
      |∑ i : Fin n, (if σ i then (1 : ℝ) else -1) * (1 / Real.sqrt n)| ^ 3 =
      ((Int.natAbs (∑ i : Fin n, if σ i then (1 : ℤ) else -1) : ℝ) ^ 3) /
        ((n : ℝ) * Real.sqrt n) := by
    intro σ
    rw [← Finset.sum_mul, abs_mul, abs_of_pos (one_div_pos.2 hsqrt_pos), mul_pow]
    have hcast : (∑ i : Fin n, if σ i then (1 : ℝ) else -1) =
        ((∑ i : Fin n, if σ i then (1 : ℤ) else -1 : ℤ) : ℝ) := by
      push_cast
      apply Finset.sum_congr rfl; intro i _; split_ifs <;> simp
    rw [hcast, ← Int.cast_abs, Int.abs_eq_natAbs, Int.cast_natCast]
    have hcube : (1 / Real.sqrt n) ^ 3 = 1 / ((n : ℝ) * Real.sqrt n) := by
      have : Real.sqrt n ^ 3 = (n : ℝ) * Real.sqrt n := by
        rw [pow_succ, hsq]
      rw [one_div_pow, this]
    rw [hcube]
    ring
  simp_rw [hterm, ← Finset.sum_div]
  push_cast
  ring

lemma radIntSum_1 : radIntSum 1 = 2 := by decide
lemma radIntSum_2 : radIntSum 2 = 16 := by decide
lemma radIntSum_3 : radIntSum 3 = 60 := by decide
lemma radIntSum_4 : radIntSum 4 = 192 := by decide
lemma radIntSum_5 : radIntSum 5 = 540 := by decide

theorem M_one : M 1 = 1 := by
  rw [M_eq_radIntSum 1 (by norm_num), radIntSum_1]
  norm_num

theorem M_two : M 2 = Real.sqrt 2 := by
  rw [M_eq_radIntSum 2 (by norm_num), radIntSum_2]
  have h2 : Real.sqrt 2 ≠ 0 := ne_of_gt (Real.sqrt_pos.2 (by norm_num))
  have hsq : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  norm_num
  field_simp [h2]
  nlinarith [hsq]

theorem M_three : M 3 = 5 / (2 * Real.sqrt 3) := by
  rw [M_eq_radIntSum 3 (by norm_num), radIntSum_3]
  have h3 : Real.sqrt 3 ≠ 0 := ne_of_gt (Real.sqrt_pos.2 (by norm_num))
  norm_num
  field_simp [h3]
  ring

theorem M_four : M 4 = 3 / 2 := by
  rw [M_eq_radIntSum 4 (by norm_num), radIntSum_4]
  have h4 : Real.sqrt 4 = 2 := by
    rw [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
  norm_num [h4]

theorem M_five : M 5 = 27 * Real.sqrt 5 / 40 := by
  rw [M_eq_radIntSum 5 (by norm_num), radIntSum_5]
  have h5 : Real.sqrt 5 ≠ 0 := ne_of_gt (Real.sqrt_pos.2 (by norm_num))
  have hsq : Real.sqrt 5 ^ 2 = 5 := Real.sq_sqrt (by norm_num)
  norm_num
  field_simp [h5]
  nlinarith [hsq]

/-- Sharp extremizer `a = (1, 1, 0) / √2` in dimension `n = 3`. -/
noncomputable def sharpVec3 : Fin 3 → ℝ :=
  ![1 / Real.sqrt 2, 1 / Real.sqrt 2, 0]

theorem sharp_n3_unit : ∑ i : Fin 3, sharpVec3 i ^ 2 = 1 := by
  simp only [Fin.sum_univ_three, sharpVec3, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.cons_val_two, Matrix.head_cons, Matrix.tail_cons]
  have hsq : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  rw [div_pow, one_pow, hsq]
  norm_num

theorem sharp_n3_moment : radCubicMoment sharpVec3 = Real.sqrt 2 := by
  unfold radCubicMoment
  have hsqrt_pos : 0 < Real.sqrt 2 := Real.sqrt_pos.2 (by norm_num)
  have hsq : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  let hequiv : (Fin 3 → Bool) ≃ Bool × Bool × Bool :=
    { toFun := fun σ => (σ 0, σ 1, σ 2)
      invFun := fun t => ![t.1, t.2.1, t.2.2]
      left_inv := fun σ => by ext i; fin_cases i <;> rfl
      right_inv := fun ⟨_, _, _⟩ => rfl }
  rw [← Fintype.sum_equiv hequiv.symm _ _ (fun _ => rfl)]
  simp only [Fintype.sum_prod_type, Fintype.sum_bool, Fin.sum_univ_three,
    hequiv, Equiv.coe_fn_symm_mk, sharpVec3, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.cons_val_two, Matrix.head_cons, Matrix.tail_cons, Bool.false_eq_true, ↓reduceIte,
    mul_zero, add_zero, one_mul, neg_one_mul]
  have h2eq : 1 / Real.sqrt 2 + 1 / Real.sqrt 2 = Real.sqrt 2 := by
    field_simp [ne_of_gt hsqrt_pos]; linarith
  have h2neg : -(1 / Real.sqrt 2) + -(1 / Real.sqrt 2) = -Real.sqrt 2 := by linarith
  have h0a : 1 / Real.sqrt 2 + -(1 / Real.sqrt 2) = 0 := by ring
  have h0b : -(1 / Real.sqrt 2) + 1 / Real.sqrt 2 = 0 := by ring
  rw [h2eq, h2neg, h0a, h0b, abs_neg, abs_of_pos hsqrt_pos, abs_zero]
  have hcube : Real.sqrt 2 ^ 3 = 2 * Real.sqrt 2 := by
    rw [pow_succ, hsq]
  rw [hcube]
  ring

theorem sharp_n3_deficit :
    M 3 - radCubicMoment sharpVec3 = 5 / (2 * Real.sqrt 3) - Real.sqrt 2 := by
  rw [M_three, sharp_n3_moment]

theorem sharp_n3_delta : Delta sharpVec3 = 1 / 6 := by
  unfold Delta
  simp only [Fin.sum_univ_three, sharpVec3, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.cons_val_two, Matrix.head_cons, Matrix.tail_cons]
  have hsq : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  rw [div_pow, one_pow, hsq]
  norm_num

/-- **Exact sharpness at `n = 3, a = (1, 1, 0) / √2`**:
`M₃ - E|a·ε|³ = (5√3 - 6√2) Δ₃(a)`. -/
theorem sharp_n3_equality :
    M 3 - radCubicMoment sharpVec3 = C_opt * Delta sharpVec3 := by
  rw [sharp_n3_deficit, sharp_n3_delta, C_opt]
  have h3 : Real.sqrt 3 ≠ 0 := ne_of_gt (Real.sqrt_pos.2 (by norm_num))
  have hsq : Real.sqrt 3 ^ 2 = 3 := Real.sq_sqrt (by norm_num)
  field_simp [h3]
  nlinarith [hsq]

theorem sharp_n3_ratio :
    (M 3 - radCubicMoment sharpVec3) / Delta sharpVec3 = 5 * Real.sqrt 3 - 6 * Real.sqrt 2 := by
  rw [sharp_n3_equality, mul_div_cancel_right₀ _ (by rw [sharp_n3_delta]; norm_num)]
  rfl

theorem secant_C_2_1 : (M 2 - M 1) / (1 / 1 - 1 / 2) = C_2_1 := by
  rw [M_two, M_one, C_2_1]; ring

theorem secant_C_4_2 : (M 4 - M 2) / (1 / 2 - 1 / 4) = C_4_2 := by
  rw [M_four, M_two, C_4_2]; ring

theorem secant_C_5_4 : (M 5 - M 4) / (1 / 4 - 1 / 5) = C_5_4 := by
  rw [M_five, M_four, C_5_4]; ring

/-! ### 2. Exact Integer-Squaring Constant Separation Inequalities -/

/-- `C = 5√3 - 6√2 < 7/40` via `4751² = 22572001 < 22579200 = 2 · 3360²`. -/
theorem C_opt_lt_seven_fortieths : C_opt < 7 / 40 := by
  unfold C_opt
  have h2 : 0 < Real.sqrt 2 := Real.sqrt_pos.2 (by norm_num)
  have h3 : 0 < Real.sqrt 3 := Real.sqrt_pos.2 (by norm_num)
  have hsq2 : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  have hsq3 : Real.sqrt 3 ^ 2 = 3 := Real.sq_sqrt (by norm_num)
  have hstep : (4751 : ℝ) < 3360 * Real.sqrt 2 := by
    nlinarith [h2, hsq2]
  have hstep2 : (5 * Real.sqrt 3) ^ 2 < (6 * Real.sqrt 2 + 7 / 40) ^ 2 := by
    nlinarith [hsq2, hsq3, hstep]
  nlinarith [h2, h3, hstep2]

/-- `C_{2,1} = 2(√2 - 1) > 7/40` via `8 · 40² = 12800 > 7569 = 87²`. -/
theorem C_2_1_gt_seven_fortieths : 7 / 40 < C_2_1 := by
  unfold C_2_1
  have h2 : 0 < Real.sqrt 2 := Real.sqrt_pos.2 (by norm_num)
  have hsq2 : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  nlinarith

/-- `C_{4,2} = 6 - 4√2 > 7/40` via `233² = 54289 > 51200 = 32 · 40²`. -/
theorem C_4_2_gt_seven_fortieths : 7 / 40 < C_4_2 := by
  unfold C_4_2
  have h2 : 0 < Real.sqrt 2 := Real.sqrt_pos.2 (by norm_num)
  have hsq2 : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  nlinarith

/-- `C_{5,4} = 27√5 / 2 - 30 > 7/40` via `5 · 540² = 1458000 > 1456849 = 1207²`. -/
theorem C_5_4_gt_seven_fortieths : 7 / 40 < C_5_4 := by
  unfold C_5_4
  have h5 : 0 < Real.sqrt 5 := Real.sqrt_pos.2 (by norm_num)
  have hsq5 : Real.sqrt 5 ^ 2 = 5 := Real.sq_sqrt (by norm_num)
  nlinarith

/-! ### 3. `π`-Bracketed Large-`n` / Gaussian Comparison Bounds -/

theorem four_div_pi_curv_gt :
    (0.16 : ℝ) < (4 / Real.pi) * (0.64 / 4 - 1 / 32) := by
  have hpi_pos : 0 < Real.pi := Real.pi_pos
  have hpi_lt : Real.pi < 3.1416 := Real.pi_lt_d4
  rw [div_mul_eq_mul_div, lt_div_iff₀ hpi_pos]
  linarith

theorem sqrt_eight_div_pi_bounds :
    (1.595 : ℝ) < Real.sqrt (8 / Real.pi) ∧ Real.sqrt (8 / Real.pi) < 1.596 := by
  have hpi_pos : 0 < Real.pi := Real.pi_pos
  have hpi_gt : (3.141592 : ℝ) < Real.pi := Real.pi_gt_d6
  have hpi_lt : Real.pi < 3.1416 := Real.pi_lt_d4
  have h1 : (1.595 : ℝ) ^ 2 < 8 / Real.pi := by
    rw [lt_div_iff₀ hpi_pos]; linarith
  have h2 : 8 / Real.pi < (1.596 : ℝ) ^ 2 := by
    rw [div_lt_iff₀ hpi_pos]; linarith
  have hsq_pos : 0 ≤ 8 / Real.pi := le_of_lt (div_pos (by norm_num) hpi_pos)
  constructor
  · rw [← Real.sqrt_sq (by norm_num : (0 : ℝ) ≤ 1.595)]
    exact Real.sqrt_lt_sqrt (by norm_num) h1
  · rw [← Real.sqrt_sq (by norm_num : (0 : ℝ) ≤ 1.596)]
    exact Real.sqrt_lt_sqrt hsq_pos h2

theorem sqrt_two_div_pi_gt : (0.797 : ℝ) < Real.sqrt (2 / Real.pi) := by
  have hpi_pos : 0 < Real.pi := Real.pi_pos
  have hpi_lt : Real.pi < 3.1416 := Real.pi_lt_d4
  have h1 : (0.797 : ℝ) ^ 2 < 2 / Real.pi := by
    rw [lt_div_iff₀ hpi_pos]; linarith
  rw [← Real.sqrt_sq (by norm_num : (0 : ℝ) ≤ 0.797)]
  exact Real.sqrt_lt_sqrt (by norm_num) h1

/-! ### 4. Fourier Majorant, Curvature `L₀`, and Far-Region Endpoint Reductions -/

/-- Exact telescoping polynomial identity for the Fourier recurrence of `|cos t|^s`. -/
theorem fourier_recurrence_telescope (u j : ℝ) :
    let P : ℝ → ℝ := fun k => (k ^ 3 + (u - 1) * k ^ 2 + u ^ 2) / 2
    (u + j + 1) * P j - (u - j) * P (j + 1) = (u + j + 1) * j ^ 3 := by
  intro P
  dsimp only [P]
  ring

/-- Pointwise weighted AM–GM Fourier majorant (Proposition 1):
`∏ cos(√x_i t) ≤ ∑_{x_i > 0} x_i |cos(√x_i t)|^{1/x_i}` for every `x ∈ Δ^{n-1}`. -/
theorem cos_prod_le_weighted_rpow {n : ℕ} (x : Fin n → ℝ)
    (hx0 : ∀ i, 0 ≤ x i) (hx1 : ∑ i : Fin n, x i = 1) (t : ℝ) :
    ∏ i : Fin n, Real.cos (Real.sqrt (x i) * t) ≤
      ∑ i ∈ Finset.univ.filter (fun i => 0 < x i),
        x i * |Real.cos (Real.sqrt (x i) * t)| ^ (1 / x i) := by
  let S := Finset.univ.filter (fun i : Fin n => 0 < x i)
  have hsumS : ∑ i ∈ S, x i = 1 := by
    rw [← hx1]
    apply Finset.sum_subset (Finset.filter_subset _ _)
    intro i _ hi
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, not_lt] at hi
    linarith [hx0 i]
  have hprod_le : ∏ i : Fin n, Real.cos (Real.sqrt (x i) * t) ≤
      ∏ i ∈ S, |Real.cos (Real.sqrt (x i) * t)| := by
    have h1 : ∏ i : Fin n, Real.cos (Real.sqrt (x i) * t) ≤
        |∏ i : Fin n, Real.cos (Real.sqrt (x i) * t)| := le_abs_self _
    rw [Finset.abs_prod] at h1
    refine le_trans h1 (le_of_eq ?_)
    symm
    apply Finset.prod_subset (Finset.filter_subset _ _)
    intro i _ hi
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, not_lt] at hi
    have hxi : x i = 0 := le_antisymm hi (hx0 i)
    simp [hxi]
  refine le_trans hprod_le ?_
  have hrpow : ∏ i ∈ S, |Real.cos (Real.sqrt (x i) * t)| =
      ∏ i ∈ S, (|Real.cos (Real.sqrt (x i) * t)| ^ (1 / x i)) ^ (x i) := by
    apply Finset.prod_congr rfl
    intro i hi
    have hxi : 0 < x i := (Finset.mem_filter.1 hi).2
    rw [← Real.rpow_mul (abs_nonneg _), one_div_mul_cancel (ne_of_gt hxi), Real.rpow_one]
  rw [hrpow]
  exact Real.geom_mean_le_arith_mean_weighted S x
    (fun i => |Real.cos (Real.sqrt (x i) * t)| ^ (1 / x i))
    (fun i _ => hx0 i) hsumS
    (fun i _ => Real.rpow_nonneg (abs_nonneg _) _)

/-- Explicit curvature lower bound `L₀(x) = 1.595 (1 - x/4) ((1 - x²)/4 - x/32)`. -/
noncomputable def L0 (x : ℝ) : ℝ :=
  1.595 * (1 - x / 4) * ((1 - x ^ 2) / 4 - x / 32)

lemma hasDerivAt_L0 (x : ℝ) :
    HasDerivAt L0 (1.595 * (3 * x ^ 2 / 16 - 31 * x / 64 - 3 / 32)) x := by
  have hpol : L0 = fun t => 1.595 * (t ^ 3 / 16 - 31 * t ^ 2 / 128 - 3 * t / 32 + 1 / 4) := by
    funext t; unfold L0; ring
  rw [hpol]
  have h1 : HasDerivAt (fun t : ℝ => t ^ 3 / 16 - 31 * t ^ 2 / 128 - 3 * t / 32 + 1 / 4)
      (3 * x ^ 2 / 16 - 31 * x / 64 - 3 / 32) x := by
    have ha := ((hasDerivAt_pow 3 x).div_const 16)
    have hb := (((hasDerivAt_pow 2 x).const_mul 31).div_const 128)
    have hc := (((hasDerivAt_id x).const_mul 3).div_const 32)
    convert ((ha.sub hb).sub hc).add (hasDerivAt_const x (1 / 4 : ℝ)) using 1
    ring
  simpa using h1.const_mul 1.595

lemma L0_deriv_neg {x : ℝ} (hx0 : 0 ≤ x) (hx1 : x ≤ 1 / 2) :
    1.595 * (3 * x ^ 2 / 16 - 31 * x / 64 - 3 / 32) < 0 := by
  nlinarith

lemma L0_second_deriv_neg {x : ℝ} (hx1 : x ≤ 1 / 2) :
    1.595 * (3 * x / 8 - 31 / 64) < 0 := by
  linarith

theorem L0_strictAntiOn : StrictAntiOn L0 (Set.Icc 0 (1 / 2)) := by
  apply strictAntiOn_of_deriv_neg (convex_Icc 0 (1 / 2))
  · exact fun x _ => (hasDerivAt_L0 x).continuousAt.continuousWithinAt
  · intro x hx
    rw [interior_Icc] at hx
    rw [(hasDerivAt_L0 x).deriv]
    exact L0_deriv_neg (le_of_lt hx.1) (le_of_lt hx.2)

/-- Tangent deficit constant `T₆ > 7/40` settling all even dimensions `n ≥ 6`. -/
theorem T6_gt_c : (5 * L0 (4 / 9) + 4 * 0.16) / 9 > 7 / 40 := by
  unfold L0; norm_num

/-- Tangent deficit constant `T₇ > 0.2219` for `n ≥ 7`. -/
theorem T7_gt_02219 : (5 * L0 (3 / 7) + 4 * 0.16) / 9 > 0.2219 := by
  unfold L0; norm_num

/-- Far odd region `D ≥ 6`: `(T₇ - 7/40) * 6 > 0.234`. -/
theorem far_odd_D_ge_6 : (((5 * L0 (3 / 7) + 4 * 0.16) / 9) - 7 / 40) * 6 > 0.234 := by
  unfold L0; norm_num

/-- Odd defect upper bound `δ_ub(n) = 1.596 (n / (8(n-1)) + 1 / (64(n-1)²))` from (6). -/
noncomputable def deltaUb (n : ℝ) : ℝ :=
  1.596 * (n / (8 * (n - 1)) + 1 / (64 * (n - 1) ^ 2))

theorem deltaUb_strictAnti {m n : ℝ} (hm : 1 < m) (hmn : m < n) :
    deltaUb n < deltaUb m := by
  unfold deltaUb
  have hm1 : 0 < m - 1 := by linarith
  have hn1 : 0 < n - 1 := by linarith
  have h1 : n / (8 * (n - 1)) < m / (8 * (m - 1)) := by
    rw [div_lt_div_iff₀ (by linarith) (by linarith)]
    nlinarith
  have h2 : 1 / (64 * (n - 1) ^ 2) < 1 / (64 * (m - 1) ^ 2) := by
    rw [div_lt_div_iff₀ (by positivity) (by positivity)]
    nlinarith
  linarith

theorem deltaUb_7 : deltaUb 7 ≤ 0.234 := by unfold deltaUb; norm_num
theorem deltaUb_9 : deltaUb 9 ≤ 0.225 := by unfold deltaUb; norm_num
theorem deltaUb_11 : deltaUb 11 ≤ 0.220 := by unfold deltaUb; norm_num

theorem deltaUb_ge_11 {n : ℝ} (hn : 11 ≤ n) : deltaUb n ≤ 0.220 := by
  rcases eq_or_lt_of_le hn with rfl | hlt
  · exact deltaUb_11
  · exact le_trans (le_of_lt (deltaUb_strictAnti (by norm_num) hlt)) deltaUb_11

/-- Far-odd endpoint certificate function `ψ_N(D) = D (L₀((1 + √D) / N) - 7/40)`. -/
noncomputable def psiFar (N D : ℝ) : ℝ :=
  D * (L0 ((1 + Real.sqrt D) / N) - 7 / 40)

/-- Second-derivative concavity expression of `ψ_N(D)` on `(0, 6]` is strictly negative:
`(3/2) L₀'(x) x' + D L₀''(x) (x')² < 0` whenever `x ∈ [0, 1/2]`, `x' > 0`, `D > 0`. -/
theorem psiFar_second_deriv_neg {D x x' : ℝ}
    (hD : 0 < D) (hx0 : 0 ≤ x) (hx1 : x ≤ 1 / 2) (hx' : 0 < x') :
    (3 / 2 : ℝ) * (1.595 * (3 * x ^ 2 / 16 - 31 * x / 64 - 3 / 32)) * x' +
      D * (1.595 * (3 * x / 8 - 31 / 64)) * x' ^ 2 < 0 := by
  have hd1 := L0_deriv_neg hx0 hx1
  have hd2 := L0_second_deriv_neg hx1
  have h1 : (3 / 2 : ℝ) * (1.595 * (3 * x ^ 2 / 16 - 31 * x / 64 - 3 / 32)) * x' < 0 := by
    nlinarith
  have h2 : D * (1.595 * (3 * x / 8 - 31 / 64)) * x' ^ 2 < 0 := by
    have : 0 < D * x' ^ 2 := mul_pos hD (sq_pos_of_pos hx')
    nlinarith
  linarith

/-- Exact verification of all 6 far-odd rational endpoints (`N ∈ {7, 9, 11}`, `D ∈ {H(N), 6}`). -/
theorem far_odd_six_endpoints :
    psiFar 7 1.75 > 0.234 ∧ psiFar 7 6 > 0.234 ∧
    psiFar 9 1.45 > 0.225 ∧ psiFar 9 6 > 0.225 ∧
    psiFar 11 1.24 > 0.220 ∧ psiFar 11 6 > 0.220 := by
  have haux : ∀ (N D u e : ℝ), 0 < N → 0 < D → 0 < u → D ≤ u ^ 2 →
      (1 + u) / N ≤ 1 / 2 → D * (L0 ((1 + u) / N) - 7 / 40) > e → psiFar N D > e := by
    intro N D u e hN hD hu hDu hu_le hval
    unfold psiFar
    have hsqrt_le : Real.sqrt D ≤ u := by
      rw [← Real.sqrt_sq (le_of_lt hu)]
      exact Real.sqrt_le_sqrt hDu
    have hx_le : (1 + Real.sqrt D) / N ≤ (1 + u) / N :=
      div_le_div_of_nonneg_right (by linarith) (le_of_lt hN)
    have hx0 : 0 ≤ (1 + Real.sqrt D) / N :=
      div_nonneg (by linarith [Real.sqrt_nonneg D]) (le_of_lt hN)
    have hL : L0 ((1 + u) / N) ≤ L0 ((1 + Real.sqrt D) / N) := by
      rcases eq_or_lt_of_le hx_le with heq | hlt
      · rw [heq]
      · exact le_of_lt (L0_strictAntiOn ⟨hx0, le_trans hx_le hu_le⟩ ⟨le_trans hx0 hx_le, hu_le⟩ hlt)
    nlinarith
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact haux 7 1.75 1.323 0.234 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by unfold L0; norm_num)
  · exact haux 7 6 2.45 0.234 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by unfold L0; norm_num)
  · exact haux 9 1.45 1.205 0.225 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by unfold L0; norm_num)
  · exact haux 9 6 2.45 0.225 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by unfold L0; norm_num)
  · exact haux 11 1.24 1.114 0.220 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by unfold L0; norm_num)
  · exact haux 11 6 2.45 0.220 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by unfold L0; norm_num)

/-! ### 5. Two-Coordinate Kernel `K` and Equalization-Path Monotonicity (Section 4) -/

/-- Two-coordinate derivative kernel `K(h₊, l, r) = (h₊ - r)₊² / h₊ - (l - r)₊² / l`. -/
noncomputable def K (hp l r : ℝ) : ℝ :=
  (max (hp - r) 0) ^ 2 / hp - (max (l - r) 0) ^ 2 / l

/-- Closed-form piecewise expansion of `K(h₊, l, r)` for `0 < l < h₊`. -/
lemma K_cases {hp l r : ℝ} (hl : 0 < l) (hlh : l < hp) :
    (r ≤ l → K hp l r = (hp - l) + (1 / hp - 1 / l) * r ^ 2) ∧
    (l ≤ r → r ≤ hp → K hp l r = hp - 2 * r + r ^ 2 / hp) ∧
    (hp ≤ r → K hp l r = 0) := by
  have hhp : 0 < hp := lt_trans hl hlh
  refine ⟨?_, ?_, ?_⟩
  · intro hrl
    unfold K
    rw [max_eq_left (by linarith : 0 ≤ hp - r), max_eq_left (by linarith : 0 ≤ l - r)]
    field_simp [ne_of_gt hhp, ne_of_gt hl]
    ring
  · intro hlr hrh
    unfold K
    rw [max_eq_left (by linarith : 0 ≤ hp - r), max_eq_right (by linarith : l - r ≤ 0)]
    field_simp [ne_of_gt hhp]
    ring
  · intro hhr
    unfold K
    rw [max_eq_right (by linarith : hp - r ≤ 0), max_eq_right (by linarith : l - r ≤ 0)]
    norm_num

theorem K_nonneg {hp l r : ℝ} (hl : 0 < l) (hlh : l < hp) (hr : 0 ≤ r) :
    0 ≤ K hp l r := by
  have hhp : 0 < hp := lt_trans hl hlh
  have ⟨h1, h2, h3⟩ := K_cases (r := r) hl hlh
  by_cases hrl : r ≤ l
  · rw [h1 hrl]
    have hfactor : (hp - l) + (1 / hp - 1 / l) * r ^ 2 =
        (hp - l) * (hp * l - r ^ 2) / (hp * l) := by
      field_simp [ne_of_gt hhp, ne_of_gt hl]; ring
    rw [hfactor]
    apply div_nonneg _ (le_of_lt (mul_pos hhp hl))
    have : r ^ 2 ≤ hp * l := by nlinarith
    nlinarith
  · push_neg at hrl
    by_cases hrh : r ≤ hp
    · rw [h2 (le_of_lt hrl) hrh]
      have : hp - 2 * r + r ^ 2 / hp = (hp - r) ^ 2 / hp := by
        field_simp [ne_of_gt hhp]; ring
      rw [this]
      exact div_nonneg (sq_nonneg _) (le_of_lt hhp)
    · push_neg at hrh
      rw [h3 (le_of_lt hrh)]

/-- Piecewise subgradient / derivative `m(v) = d/dv K(√v)` on `[0, ∞)`. -/
noncomputable def KsqrtDeriv (hp l v : ℝ) : ℝ :=
  if v ≤ l ^ 2 then 1 / hp - 1 / l
  else if v ≤ hp ^ 2 then 1 / hp - 1 / Real.sqrt v
  else 0

lemma KsqrtDeriv_nonpos {hp l v : ℝ} (hl : 0 < l) (hlh : l < hp) :
    KsqrtDeriv hp l v ≤ 0 := by
  have hhp : 0 < hp := lt_trans hl hlh
  unfold KsqrtDeriv
  split_ifs with h1 h2
  · rw [sub_nonpos, one_div_le_one_div hhp hl]; exact le_of_lt hlh
  · push_neg at h1
    have hsqv_le : Real.sqrt v ≤ hp := by
      rw [← Real.sqrt_sq (le_of_lt hhp)]; exact Real.sqrt_le_sqrt h2
    have hsqv_pos : 0 < Real.sqrt v := Real.sqrt_pos.2 (lt_trans (sq_pos_of_pos hl) h1)
    rw [sub_nonpos, one_div_le_one_div hhp hsqv_pos]; exact hsqv_le
  · exact le_rfl

/-- Elementary supporting-line inequality for `v ↦ K(√v)` on `[0, ∞)`:
`K(√v) ≥ K(√v₀) + KsqrtDeriv(v₀) * (v - v₀)` for all `v, v₀ ≥ 0`. -/
lemma K_sqrt_support {hp l v v₀ : ℝ} (hl : 0 < l) (hlh : l < hp) (hv : 0 ≤ v) (hv₀ : 0 ≤ v₀) :
    K hp l (Real.sqrt v₀) + KsqrtDeriv hp l v₀ * (v - v₀) ≤ K hp l (Real.sqrt v) := by
  have hhp : 0 < hp := lt_trans hl hlh
  set r := Real.sqrt v
  set r₀ := Real.sqrt v₀
  have hr : 0 ≤ r := Real.sqrt_nonneg v
  have hr₀ : 0 ≤ r₀ := Real.sqrt_nonneg v₀
  have hrv : r ^ 2 = v := Real.sq_sqrt hv
  have hr₀v : r₀ ^ 2 = v₀ := Real.sq_sqrt hv₀
  have hiff_l : ∀ t ≥ 0, (t ≤ l ^ 2 ↔ Real.sqrt t ≤ l) := by
    intro t ht; constructor
    · intro h; rw [← Real.sqrt_sq (le_of_lt hl)]; exact Real.sqrt_le_sqrt h
    · intro h; nlinarith [Real.sq_sqrt ht, Real.sqrt_nonneg t]
  have hiff_hp : ∀ t ≥ 0, (t ≤ hp ^ 2 ↔ Real.sqrt t ≤ hp) := by
    intro t ht; constructor
    · intro h; rw [← Real.sqrt_sq (le_of_lt hhp)]; exact Real.sqrt_le_sqrt h
    · intro h; nlinarith [Real.sq_sqrt ht, Real.sqrt_nonneg t]
  have ⟨c1, c2, c3⟩ := K_cases (r := r) hl hlh
  have ⟨d1, d2, d3⟩ := K_cases (r := r₀) hl hlh
  -- Universal lower bounds on K(r) in terms of the three branches
  have hK_ge_A : (hp - l) + (1 / hp - 1 / l) * r ^ 2 ≤ K hp l r := by
    by_cases hrl : r ≤ l
    · rw [c1 hrl]
    · push_neg at hrl
      by_cases hrh : r ≤ hp
      · rw [c2 (le_of_lt hrl) hrh]
        have hdiff : (hp - 2 * r + r ^ 2 / hp) - ((hp - l) + (1 / hp - 1 / l) * r ^ 2) =
            (r - l) ^ 2 / l := by field_simp [ne_of_gt hhp, ne_of_gt hl]; ring
        have hnn : 0 ≤ (r - l) ^ 2 / l := div_nonneg (sq_nonneg _) (le_of_lt hl)
        linarith
      · push_neg at hrh
        rw [c3 (le_of_lt hrh)]
        have hdiff : -((hp - l) + (1 / hp - 1 / l) * r ^ 2) =
            ((hp - l) * (r ^ 2 - hp * l)) / (hp * l) := by
          field_simp [ne_of_gt hhp, ne_of_gt hl]; ring
        have hnn : 0 ≤ ((hp - l) * (r ^ 2 - hp * l)) / (hp * l) := by
          apply div_nonneg _ (le_of_lt (mul_pos hhp hl))
          have : hp * l ≤ r ^ 2 := by nlinarith
          nlinarith
        linarith
  have hK_ge_0 : 0 ≤ K hp l r := K_nonneg hl hlh hr
  -- Split on the regime of v₀
  unfold KsqrtDeriv
  split_ifs with h0l h0h
  · -- Regime 1: v₀ ≤ l² (r₀ ≤ l)
    have hr0l : r₀ ≤ l := (hiff_l v₀ hv₀).1 h0l
    rw [d1 hr0l, ← hrv, ← hr₀v]
    linarith [hK_ge_A]
  · -- Regime 2: l² < v₀ ≤ hp² (l < r₀ ≤ hp)
    push_neg at h0l
    have hr0l : l < r₀ := by
      rw [← Real.sqrt_sq (le_of_lt hl)]; exact Real.sqrt_lt_sqrt (sq_nonneg l) h0l
    have hr0h : r₀ ≤ hp := (hiff_hp v₀ hv₀).1 h0h
    have hr0_pos : 0 < r₀ := lt_trans hl hr0l
    rw [d2 (le_of_lt hr0l) hr0h]
    change (hp - 2 * r₀ + r₀ ^ 2 / hp) + (1 / hp - 1 / r₀) * (v - v₀) ≤ K hp l r
    rw [← hrv, ← hr₀v]
    by_cases hrl : r ≤ l
    · rw [c1 hrl]
      have hdiff : ((hp - l) + (1 / hp - 1 / l) * r ^ 2) -
          ((hp - 2 * r₀ + r₀ ^ 2 / hp) + (1 / hp - 1 / r₀) * (r ^ 2 - r₀ ^ 2)) =
          (r₀ - l) * (l * r₀ - r ^ 2) / (l * r₀) := by
        field_simp [ne_of_gt hhp, ne_of_gt hl, ne_of_gt hr0_pos]; ring
      have hnn : 0 ≤ (r₀ - l) * (l * r₀ - r ^ 2) / (l * r₀) := by
        apply div_nonneg _ (le_of_lt (mul_pos hl hr0_pos))
        have : r ^ 2 ≤ l * r₀ := by nlinarith
        nlinarith
      linarith
    · push_neg at hrl
      by_cases hrh : r ≤ hp
      · rw [c2 (le_of_lt hrl) hrh]
        have hdiff : (hp - 2 * r + r ^ 2 / hp) -
            ((hp - 2 * r₀ + r₀ ^ 2 / hp) + (1 / hp - 1 / r₀) * (r ^ 2 - r₀ ^ 2)) =
            (r - r₀) ^ 2 / r₀ := by
          field_simp [ne_of_gt hhp, ne_of_gt hr0_pos]; ring
        have hnn : 0 ≤ (r - r₀) ^ 2 / r₀ := div_nonneg (sq_nonneg _) (le_of_lt hr0_pos)
        linarith
      · push_neg at hrh
        rw [c3 (le_of_lt hrh)]
        have hdiff : -((hp - 2 * r₀ + r₀ ^ 2 / hp) + (1 / hp - 1 / r₀) * (r ^ 2 - r₀ ^ 2)) =
            (hp - r₀) * (r ^ 2 - hp * r₀) / (hp * r₀) := by
          field_simp [ne_of_gt hhp, ne_of_gt hr0_pos]; ring
        have hnn : 0 ≤ (hp - r₀) * (r ^ 2 - hp * r₀) / (hp * r₀) := by
          apply div_nonneg _ (le_of_lt (mul_pos hhp hr0_pos))
          have : hp * r₀ ≤ r ^ 2 := by nlinarith
          nlinarith
        linarith
  · -- Regime 3: v₀ > hp² (r₀ > hp)
    push_neg at h0h
    have hr0h : hp < r₀ := by
      rw [← Real.sqrt_sq (le_of_lt hhp)]; exact Real.sqrt_lt_sqrt (sq_nonneg hp) h0h
    rw [d3 (le_of_lt hr0h), zero_mul, add_zero]
    exact hK_ge_0

/-- **Lemma 3(2a)**: `K` is antitone (nonincreasing) on `[0, ∞)`. -/
theorem K_antitoneOn {hp l : ℝ} (hl : 0 < l) (hlh : l < hp) :
    AntitoneOn (K hp l) (Set.Ici 0) := by
  intro r₁ hr₁ r₂ hr₂ hr
  have hr₁' : 0 ≤ r₁ := hr₁
  have hr₂' : 0 ≤ r₂ := hr₂
  have hsq : r₁ ^ 2 ≤ r₂ ^ 2 := by nlinarith
  have hsup := K_sqrt_support hl hlh (sq_nonneg r₁) (sq_nonneg r₂)
  rw [Real.sqrt_sq hr₁', Real.sqrt_sq hr₂'] at hsup
  have hder : KsqrtDeriv hp l (r₂ ^ 2) ≤ 0 := KsqrtDeriv_nonpos hl hlh
  nlinarith

/-- **Lemma 3(2b)**: `v ↦ K(√v)` is convex on `[0, ∞)`. -/
theorem K_sqrt_convexOn {hp l : ℝ} (hl : 0 < l) (hlh : l < hp) :
    ConvexOn ℝ (Set.Ici 0) (fun v => K hp l (Real.sqrt v)) := by
  refine ⟨convex_Ici 0, ?_⟩
  intro x hx y hy a b ha hb hab
  have hx0 : 0 ≤ x := hx
  have hy0 : 0 ≤ y := hy
  set v₀ := a * x + b * y
  have hv₀ : 0 ≤ v₀ := add_nonneg (mul_nonneg ha hx0) (mul_nonneg hb hy0)
  have hsx := K_sqrt_support hl hlh hx0 hv₀
  have hsy := K_sqrt_support hl hlh hy0 hv₀
  have hcomb : a * (K hp l (Real.sqrt v₀) + KsqrtDeriv hp l v₀ * (x - v₀)) +
      b * (K hp l (Real.sqrt v₀) + KsqrtDeriv hp l v₀ * (y - v₀)) = K hp l (Real.sqrt v₀) := by
    have : a * (x - v₀) + b * (y - v₀) = 0 := by dsimp only [v₀]; nlinarith
    calc a * (K hp l (Real.sqrt v₀) + KsqrtDeriv hp l v₀ * (x - v₀)) +
          b * (K hp l (Real.sqrt v₀) + KsqrtDeriv hp l v₀ * (y - v₀))
        = (a + b) * K hp l (Real.sqrt v₀) + KsqrtDeriv hp l v₀ * (a * (x - v₀) + b * (y - v₀)) := by ring
      _ = K hp l (Real.sqrt v₀) := by rw [hab, this]; ring
  simp only [smul_eq_mul]
  have h1 := mul_le_mul_of_nonneg_left hsx ha
  have h2 := mul_le_mul_of_nonneg_left hsy hb
  linarith

/-- Central binomial probability `P_{2j} = 2^{-2j} C(2j, j)`. -/
noncomputable def centralP (j : ℕ) : ℝ :=
  (Nat.choose (2 * j) j : ℝ) / (2 : ℝ) ^ (2 * j)

lemma centralP_pos (j : ℕ) : 0 < centralP j :=
  div_pos (Nat.cast_pos.2 (Nat.choose_pos (by omega))) (by positivity)

lemma centralP_step (j : ℕ) :
    centralP (j + 1) = centralP j * ((2 * j + 1 : ℝ) / (2 * j + 2)) := by
  unfold centralP
  have hch : (j + 1) * Nat.choose (2 * (j + 1)) (j + 1) =
      2 * (2 * j + 1) * Nat.choose (2 * j) j := by
    have h2 := Nat.succ_mul_choose_eq (2 * j) j
    have h3 : Nat.choose (2 * j + 1) j = Nat.choose (2 * j + 1) (j + 1) := by
      have := Nat.choose_symm (show j ≤ 2 * j + 1 by omega)
      simpa [show 2 * j + 1 - j = j + 1 by omega] using this.symm
    have h4 : Nat.choose (2 * (j + 1)) (j + 1) =
        Nat.choose (2 * j + 1) j + Nat.choose (2 * j + 1) (j + 1) := by
      rw [show 2 * (j + 1) = (2 * j + 1) + 1 by omega, Nat.choose_succ_succ]
    have h5 : (j + 1) * Nat.choose (2 * j + 1) (j + 1) = (2 * j + 1) * Nat.choose (2 * j) j := by
      simpa [Nat.succ_eq_add_one, mul_comm] using h2.symm
    rw [h4, h3]
    linarith [h5]
  have hchR : ((j : ℝ) + 1) * (Nat.choose (2 * (j + 1)) (j + 1) : ℝ) =
      2 * (2 * j + 1) * (Nat.choose (2 * j) j : ℝ) := by
    exact_mod_cast hch
  have hpow : (2 : ℝ) ^ (2 * (j + 1)) = 4 * (2 : ℝ) ^ (2 * j) := by
    rw [show 2 * (j + 1) = 2 * j + 2 by omega, pow_add]; ring
  rw [hpow]
  have hden : (2 * (j : ℝ) + 2) ≠ 0 := by positivity
  have hpow_ne : (2 : ℝ) ^ (2 * j) ≠ 0 := by positivity
  field_simp [hden, hpow_ne]
  linarith

/-- `P_{2j} √(2j + 1)` is strictly decreasing in `j` (Lemma 3(3)). -/
theorem centralP_sqrt_succ_strictAnti (j : ℕ) :
    centralP (j + 1) * Real.sqrt (2 * (j + 1) + 1) <
      centralP j * Real.sqrt (2 * j + 1) := by
  rw [centralP_step, mul_assoc]
  apply mul_lt_mul_of_pos_left _ (centralP_pos j)
  have hden : 0 < (2 * (j : ℝ) + 2) := by positivity
  rw [div_mul_eq_mul_div, div_lt_iff₀ hden]
  have hlhs_pos : 0 ≤ (2 * (j : ℝ) + 1) * Real.sqrt (2 * (j + 1) + 1) := by positivity
  have hrhs_pos : 0 ≤ Real.sqrt (2 * (j : ℝ) + 1) * (2 * j + 2) := by positivity
  have hsq : ((2 * (j : ℝ) + 1) * Real.sqrt (2 * (j + 1) + 1)) ^ 2 <
      (Real.sqrt (2 * (j : ℝ) + 1) * (2 * j + 2)) ^ 2 := by
    rw [mul_pow, mul_pow, Real.sq_sqrt (by positivity), Real.sq_sqrt (by positivity)]
    nlinarith
  nlinarith

/-- `P_{2j} √(2j)` is strictly increasing in `j ≥ 1` (Theorem 4 Step 2). -/
theorem centralP_sqrt_strictMono {j : ℕ} (hj : 1 ≤ j) :
    centralP j * Real.sqrt (2 * j) <
      centralP (j + 1) * Real.sqrt (2 * (j + 1)) := by
  rw [centralP_step, mul_assoc]
  apply mul_lt_mul_of_pos_left _ (centralP_pos j)
  have hden : 0 < (2 * (j : ℝ) + 2) := by positivity
  rw [div_mul_eq_mul_div, lt_div_iff₀ hden]
  have hjR : (1 : ℝ) ≤ j := Nat.one_le_cast.2 hj
  have hsq : (Real.sqrt (2 * (j : ℝ)) * (2 * j + 2)) ^ 2 <
      ((2 * (j : ℝ) + 1) * Real.sqrt (2 * (j + 1))) ^ 2 := by
    rw [mul_pow, mul_pow, Real.sq_sqrt (by positivity), Real.sq_sqrt (by positivity)]
    nlinarith
  have hlhs_pos : 0 ≤ Real.sqrt (2 * (j : ℝ)) * (2 * j + 2) := by positivity
  have hrhs_pos : 0 ≤ (2 * (j : ℝ) + 1) * Real.sqrt (2 * (j + 1)) := by positivity
  nlinarith

/-- `t ↦ t √(k + t²) / (1 + t)` is strictly increasing on `(0, 1)` for `k ≥ 2` (Theorem 4 Step 2). -/
theorem equalization_quotient_strictMono {k : ℝ} (hk : 2 ≤ k) :
    StrictMonoOn (fun t : ℝ => t * Real.sqrt (k + t ^ 2) / (1 + t)) (Set.Ioo 0 1) := by
  intro t₁ ⟨ht10, ht11⟩ t₂ ⟨ht20, ht21⟩ hlt
  have h1 : t₁ / (1 + t₁) < t₂ / (1 + t₂) := by
    rw [div_lt_div_iff₀ (by linarith) (by linarith)]
    nlinarith
  have h2 : Real.sqrt (k + t₁ ^ 2) < Real.sqrt (k + t₂ ^ 2) := by
    apply Real.sqrt_lt_sqrt (by positivity)
    nlinarith
  have hmul : (t₁ / (1 + t₁)) * Real.sqrt (k + t₁ ^ 2) <
      (t₂ / (1 + t₂)) * Real.sqrt (k + t₂ ^ 2) :=
    mul_lt_mul'' h1 h2 (div_nonneg (le_of_lt ht10) (by linarith)) (Real.sqrt_nonneg _)
  simpa [div_mul_eq_mul_div] using hmul

/-- The rational factor `r / (1 + r)²` is strictly decreasing for `r > 1`. -/
theorem secant_factor_strictAnti {r₁ r₂ : ℝ} (hr1 : 1 < r₁) (hlt : r₁ < r₂) :
    r₂ / (1 + r₂) ^ 2 < r₁ / (1 + r₁) ^ 2 := by
  have hd1 : 0 < (1 + r₁) ^ 2 := sq_pos_of_pos (by linarith)
  have hd2 : 0 < (1 + r₂) ^ 2 := sq_pos_of_pos (by linarith)
  rw [div_lt_div_iff₀ hd2 hd1]
  have hprod : 1 < r₁ * r₂ := by nlinarith
  have hdiff : r₁ * (1 + r₂) ^ 2 - r₂ * (1 + r₁) ^ 2 = (r₂ - r₁) * (r₁ * r₂ - 1) := by ring
  nlinarith

/-- Odd-to-even secant slope `S(j) = P_{2j} √(2j) · r / (1 + r)²` with `r = √(1 + 1/(2j))`. -/
noncomputable def oddEvenSecant (j : ℕ) : ℝ :=
  let k : ℝ := 2 * j
  let r : ℝ := Real.sqrt (1 + 1 / k)
  centralP j * Real.sqrt k * (r / (1 + r) ^ 2)

/-- Strict monotonicity of the odd-to-even secant slope for all even `k = 2j ≥ 2`. -/
theorem secant_slope_strictMono {j : ℕ} (hj : 1 ≤ j) :
    oddEvenSecant j < oddEvenSecant (j + 1) := by
  unfold oddEvenSecant
  have hjR : (1 : ℝ) ≤ j := Nat.one_le_cast.2 hj
  have hj0 : (0 : ℝ) < j := by linarith
  have h1 := centralP_sqrt_strictMono hj
  have hr_gt : 1 < Real.sqrt (1 + 1 / (2 * ((j + 1 : ℕ) : ℝ))) := by
    rw [← Real.sqrt_one]
    apply Real.sqrt_lt_sqrt (by norm_num)
    have : (0 : ℝ) < 1 / (2 * ((j + 1 : ℕ) : ℝ)) := by positivity
    linarith
  have hr_lt : Real.sqrt (1 + 1 / (2 * ((j + 1 : ℕ) : ℝ))) <
      Real.sqrt (1 + 1 / (2 * (j : ℝ))) := by
    apply Real.sqrt_lt_sqrt (by positivity)
    have : 1 / (2 * ((j + 1 : ℕ) : ℝ)) < 1 / (2 * (j : ℝ)) := by
      rw [one_div_lt_one_div (by positivity) (by positivity)]
      push_cast; linarith
    linarith
  have h2 := secant_factor_strictAnti hr_gt hr_lt
  have h1' : centralP j * Real.sqrt (2 * (j : ℝ)) <
      centralP (j + 1) * Real.sqrt (2 * ((j + 1 : ℕ) : ℝ)) := by
    simpa [Nat.cast_add, Nat.cast_one] using h1
  exact mul_lt_mul'' h1' h2
    (mul_nonneg (le_of_lt (centralP_pos j)) (Real.sqrt_nonneg _))
    (div_nonneg (Real.sqrt_nonneg _) (sq_nonneg _))

/-- At `k = 2` (`j = 1`), the odd-to-even secant slope equals `C = 5√3 - 6√2`. -/
theorem secant_slope_at_one : oddEvenSecant 1 = C_opt := by
  unfold oddEvenSecant centralP C_opt
  have h2 : 0 < Real.sqrt 2 := Real.sqrt_pos.2 (by norm_num)
  have h3 : 0 < Real.sqrt 3 := Real.sqrt_pos.2 (by norm_num)
  have hsq2 : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  have hsq3 : Real.sqrt 3 ^ 2 = 3 := Real.sq_sqrt (by norm_num)
  have hr : Real.sqrt (1 + 1 / (2 * (1 : ℝ))) = Real.sqrt 3 / Real.sqrt 2 := by
    rw [show (1 : ℝ) + 1 / (2 * 1) = 3 / 2 by norm_num, Real.sqrt_div (by norm_num)]
  have hch : Nat.choose (2 * 1) 1 = 2 := by decide
  simp only [Nat.cast_one, hch]
  rw [hr]
  have hden : (1 + Real.sqrt 3 / Real.sqrt 2) ^ 2 ≠ 0 := by positivity
  norm_num
  field_simp [ne_of_gt h2, ne_of_gt h3, hden]
  have h23 : Real.sqrt 2 ^ 3 = 2 * Real.sqrt 2 := by rw [pow_succ, hsq2]
  have h33 : Real.sqrt 3 ^ 3 = 3 * Real.sqrt 3 := by rw [pow_succ, hsq3]
  ring_nf
  rw [h23, h33, hsq2, hsq3]
  ring

end RademacherFourthMoment
