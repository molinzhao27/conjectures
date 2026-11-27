import Mathlib

/-!
# Lean-Verified Elementary Algebraic Inequalities Underlying the `Ω = 2` Wampler–Barato Analysis (arXiv:2110.14540)

We formalize in Lean 4 + Mathlib (zero `sorry`) the elementary algebraic inequalities and
hyperbolic parameter bridge underlying the `Ω = 2` skewness/kurtosis analysis of
Wampler & Barato (*J. Phys. A* 2022, arXiv:2110.14540):
1. **Unimodality & Endpoint Minimization of `skewSq` and `kurtU`**:
   - `skewSq_le_max`, `skewSq_at_opt`, `skewSq_ge_min_endpoints`
   - `kurtU_le_max`, `kurtU_ge_min_endpoints`
2. **Exact `Ω = 2` Upper-Bound Threshold Conditions & Concrete Rational Counterexample Instances**:
   - Algebraic threshold equivalence in `t = tanh(A/4) ∈ (0, 1)` (`refutation_table1_omega2_skewness`,
     `refutation_table1_omega2_kurtosis`), together with the hyperbolic identity
     `Real.cosh (A / 2) / Real.sinh (A / 2) = (1 + Real.tanh (A / 4) ^ 2) / (2 * Real.tanh (A / 4))`
     (`coth_half_eq_tanh_quarter`) and explicit rational witness instantiations at `t = 3/5`
     (`skewness_witness_t_3_5`) and `t = 13/20` (`kurtosis_witness_t_13_20`).
3. **Elementary Algebraic Positivity Step (`universal_maclaurin_positivity`)**:
   - Given `η > 1`, `e₄ ≥ 0`, and `4 * e₄ ≤ e₃`, proves `c * γ₃ - γ₄ ≥ η * (η² + 2) * e₄ ≥ 0`.

Note: The physics derivation of the FPT SCGF `ρ(s)`, the cumulant formulas `S(y₂)` and `K(y₂)`,
the achievable `Ω = 2` interval via AM–GM, the `Ω = 3` Hessian bifurcation, and the `Ω → ∞`
continuum limits are proved in prose in `solutions/stoch_thermo/solution.md`.
-/

namespace WamplerBaratoStochThermo

/-- Unscaled skewness kernel `f_S(c, y) = (3 c y - 1) / y^(3/2)`, where `S = √A · f_S(c, y)`.
To avoid fractional powers in algebraic inequalities, we work with `f_S(c, y)² = (3 c y - 1)² / y³`. -/
noncomputable def skewSq (c y : ℝ) : ℝ := (3 * c * y - 1) ^ 2 / y ^ 3

/-- Unscaled kurtosis kernel `f_K(c, u) = 3 c + (12 c² - 6) u - 3 c u²` in `u = 1/y`,
where `K = A · f_K(c, 1/y)`. -/
def kurtU (c u : ℝ) : ℝ := 3 * c + (12 * c ^ 2 - 6) * u - 3 * c * u ^ 2

/-- **Theorem 1 (Exact Unimodality & Global Upper Bound for `Ω = 2` Skewness)**:
For every `c > 1` and `y > 0` in the physical positive-skewness regime `1 ≤ 3 * c * y`
(note for `Ω = 2`, `c * y ≥ c * y_min = (1 + t²) / (4 t²) > 1/2 > 1/3`),
`skewSq c y ≤ 4 c³ = skewSq c (1 / c)`! -/
theorem skewSq_le_max {c y : ℝ} (_hc : 1 < c) (hy : 0 < y) (hphys : 1 ≤ 3 * c * y) :
    skewSq c y ≤ 4 * c ^ 3 := by
  unfold skewSq
  rw [div_le_iff₀ (pow_pos hy 3)]
  have hident : 4 * c ^ 3 * y ^ 3 - (3 * c * y - 1) ^ 2 = (c * y - 1) ^ 2 * (4 * c * y - 1) := by
    ring
  have hpos : 0 ≤ (c * y - 1) ^ 2 * (4 * c * y - 1) := by
    apply mul_nonneg (sq_nonneg _)
    linarith
  linarith

/-- `skewSq c (1 / c) = 4 * c^3` (attaining the global upper bound). -/
theorem skewSq_at_opt {c : ℝ} (hc : 0 < c) : skewSq c (1 / c) = 4 * c ^ 3 := by
  unfold skewSq
  have hne : c ≠ 0 := ne_of_gt hc
  field_simp [hne]
  ring

/-- **Theorem 2 (Omega = 2 Skewness Lower Bound — Endpoint Minimization)**:
For `c > 1` and `1 / (3 c) ≤ a ≤ y ≤ b` (note for `Ω = 2`, `c * y_min = (1 + t²) / (4 t²) > 1/2 > 1/3`),
`skewSq c y ≥ min (skewSq c a) (skewSq c b)`. -/
theorem skewSq_ge_min_endpoints {c a b y : ℝ} (hc : 1 < c)
    (ha3 : 1 ≤ 3 * c * a) (hay : a ≤ y) (hyb : y ≤ b) :
    min (skewSq c a) (skewSq c b) ≤ skewSq c y := by
  have hc_pos : 0 < c := by linarith
  have ha : 0 < a := by nlinarith
  have hy : 0 < y := lt_of_lt_of_le ha hay
  have hb : 0 < b := lt_of_lt_of_le hy hyb
  by_cases hcy : c * y ≤ 1
  · -- 1/(3c) ≤ a ≤ y ≤ 1/c: set p = c*a - 1/3, q = c*y - 1/3 ∈ [0, 2/3], p ≤ q!
    refine le_trans (min_le_left _ _) ?_
    unfold skewSq
    rw [div_le_div_iff₀ (pow_pos ha 3) (pow_pos hy 3)]
    set p : ℝ := c * a - 1 / 3
    set q : ℝ := c * y - 1 / 3
    have hp0 : 0 ≤ p := by dsimp [p]; linarith
    have hq0 : 0 ≤ q := by dsimp [q]; nlinarith
    have hpq : 0 ≤ q - p := by dsimp [p, q]; nlinarith
    have h2p : 0 ≤ 2 - 3 * p := by dsimp [p]; nlinarith
    have h2q : 0 ≤ 2 - 3 * q := by dsimp [q]; linarith
    have h4pq : 0 ≤ 4 - 9 * p * q := by nlinarith
    have hfactor :
        3 * c ^ 3 * ((3 * c * y - 1) ^ 2 * a ^ 3 - (3 * c * a - 1) ^ 2 * y ^ 3) =
        (q - p) * (1 / 2 * p * (2 - 3 * q) + 1 / 2 * q * (2 - 3 * p) + 3 * p * q * (4 - 9 * p * q)) := by
      dsimp [p, q]; ring
    have hpos : 0 ≤ 3 * c ^ 3 * ((3 * c * y - 1) ^ 2 * a ^ 3 - (3 * c * a - 1) ^ 2 * y ^ 3) := by
      rw [hfactor]
      apply mul_nonneg hpq
      have t1 : 0 ≤ 1 / 2 * p * (2 - 3 * q) := by positivity
      have t2 : 0 ≤ 1 / 2 * q * (2 - 3 * p) := by positivity
      have t3 : 0 ≤ 3 * p * q * (4 - 9 * p * q) := by positivity
      linarith
    have hc3 : 0 < 3 * c ^ 3 := by positivity
    nlinarith
  · -- 1/c < y ≤ b: show skewSq c b ≤ skewSq c y!
    refine le_trans (min_le_right _ _) ?_
    unfold skewSq
    rw [div_le_div_iff₀ (pow_pos hb 3) (pow_pos hy 3)]
    have hcy' : 1 < c * y := not_le.mp hcy
    -- Notice (3cy - 1)^2 b^3 - (3cb - 1)^2 y^3
    -- = (b - y) * [ 9 c^2 y^2 b^2 (b - y)? No: compute exact expansion! ]
    -- Let X = c y > 1, Y = c b ≥ X > 1.
    -- Need (3X - 1)^2 Y^3 - (3Y - 1)^2 X^3 ≥ 0:
    -- Factor: (Y - X) * [ (9 X^2 Y^2 - 6 X Y (X + Y) + (X^2 + X Y + Y^2)) ... ]
    -- Let's check exact identity:
    -- (3X - 1)^2 Y^3 - (3Y - 1)^2 X^3
    -- = (Y - X) * (9 X^2 Y^2 (Y - X)? No: (9 X^2 - 6X + 1) Y^3 - (9 Y^2 - 6Y + 1) X^3
    -- = 9 X^2 Y^2 (Y - X) - 6 X Y (Y^2 - X^2) + (Y^3 - X^3)
    -- = (Y - X) * [ 9 X^2 Y^2 - 6 X Y (Y + X) + (Y^2 + X Y + X^2) ]
    -- Wait: substitute X = 1 + u, Y = 1 + v (v ≥ u > 0):
    -- 9(1+u)^2(1+v)^2 - 6(1+u)(1+v)(2+u+v) + ((1+v)^2+(1+u)(1+v)+(1+u)^2)
    -- = 9 u^2 v^2 + 12 u v (u + v) + 3 (u^2 + 5 u v + v^2) + 3 (u + v) > 0!!
    have hX : 0 < c * y - 1 := by linarith
    have hY : 0 < c * b - 1 := by nlinarith
    have hYX : 0 ≤ c * b - c * y := by nlinarith
    have hfactor :
        c ^ 3 * ((3 * c * y - 1) ^ 2 * b ^ 3 - (3 * c * b - 1) ^ 2 * y ^ 3) =
        (c * b - c * y) *
          (9 * (c * y - 1) ^ 2 * (c * b - 1) ^ 2 +
           12 * (c * y - 1) * (c * b - 1) * ((c * y - 1) + (c * b - 1)) +
           (4 * (c * y - 1) ^ 2 + 13 * (c * y - 1) * (c * b - 1) + 4 * (c * b - 1) ^ 2) +
           3 * ((c * y - 1) + (c * b - 1))) := by
      ring
    have hpos : 0 ≤ c ^ 3 * ((3 * c * y - 1) ^ 2 * b ^ 3 - (3 * c * b - 1) ^ 2 * y ^ 3) := by
      rw [hfactor]
      apply mul_nonneg hYX
      nlinarith [sq_nonneg (c * y - 1), sq_nonneg (c * b - 1)]
    have hc3 : 0 < c ^ 3 := pow_pos (by linarith) 3
    nlinarith

/-- **Theorem 3 (Omega = 2 Kurtosis Exact Global Maximum & Endpoint Minimum)**:
In `u = 1/y`, `kurtU c u = 3c + (12c² - 6)u - 3c u²` is a strictly concave quadratic with
unique global maximum at `u_* = (2c² - 1)/c` and value `3(4c⁴ - 3c² + 1)/c`,
and is minimized on any closed interval `[u₁, u₂]` at one of the endpoints! -/
theorem kurtU_le_max {c u : ℝ} (hc : 0 < c) :
    kurtU c u ≤ 3 * (4 * c ^ 4 - 3 * c ^ 2 + 1) / c := by
  unfold kurtU
  have hne : c ≠ 0 := ne_of_gt hc
  rw [le_div_iff₀ hc]
  have hident : 3 * (4 * c ^ 4 - 3 * c ^ 2 + 1) - (3 * c + (12 * c ^ 2 - 6) * u - 3 * c * u ^ 2) * c =
      3 * (c * u - (2 * c ^ 2 - 1)) ^ 2 := by ring
  nlinarith [sq_nonneg (c * u - (2 * c ^ 2 - 1))]

theorem kurtU_ge_min_endpoints {c u₁ u₂ u : ℝ} (hc : 0 < c) (h1 : u₁ ≤ u) (h2 : u ≤ u₂) :
    min (kurtU c u₁) (kurtU c u₂) ≤ kurtU c u := by
  -- Concavity: on [u₁, u₂], kurtU(u) - min ≥ 0. Specifically:
  -- (u₂ - u) kurtU(u₁) + (u - u₁) kurtU(u₂) ≤ (u₂ - u₁) kurtU(u)
  by_cases heq : u₁ = u₂
  · have hu : u = u₁ := le_antisymm (heq ▸ h2) h1
    subst hu; exact min_le_left _ _
  · have hlt : 0 < u₂ - u₁ := sub_pos.mpr (lt_of_le_of_ne (le_trans h1 h2) heq)
    have hconc : (u₂ - u₁) * kurtU c u - ((u₂ - u) * kurtU c u₁ + (u - u₁) * kurtU c u₂) =
        3 * c * (u₂ - u₁) * (u - u₁) * (u₂ - u) := by
      unfold kurtU; ring
    have hnonneg : 0 ≤ 3 * c * (u₂ - u₁) * (u - u₁) * (u₂ - u) := by
      have h1' : 0 ≤ u - u₁ := sub_nonneg.mpr h1
      have h2' : 0 ≤ u₂ - u := sub_nonneg.mpr h2
      positivity
    have havg : (u₂ - u) * min (kurtU c u₁) (kurtU c u₂) + (u - u₁) * min (kurtU c u₁) (kurtU c u₂) ≤
        (u₂ - u) * kurtU c u₁ + (u - u₁) * kurtU c u₂ := by
      apply add_le_add
      · exact mul_le_mul_of_nonneg_left (min_le_left _ _) (sub_nonneg.mpr h2)
      · exact mul_le_mul_of_nonneg_left (min_le_right _ _) (sub_nonneg.mpr h1)
    have hlin : (u₂ - u) * min (kurtU c u₁) (kurtU c u₂) + (u - u₁) * min (kurtU c u₁) (kurtU c u₂) =
        (u₂ - u₁) * min (kurtU c u₁) (kurtU c u₂) := by ring
    nlinarith

/-- **Theorem 4 (Exact Refutation of Wampler–Barato Table 1 for `Ω = 2`, Skewness & Kurtosis)**:
Writing `t = tanh(A/4) ∈ (0, 1)`, we have `c = coth(A/2) = (1 + t²) / (2t)` and
`y_min = (1/2) coth(A/4) = 1 / (2t)`.
1. The exact interior skewness maximum `1/c` exceeds `y_min` iff `t² > 1/3`,
   and strictly beats `S_ARW(A, 2)` for every `t ∈ (1/√3, 1)` (i.e. `A > 2 ln(2 + √3) ≈ 2.6339 < 2.84`)!
   In particular, at `t = 3/5` (`A = 4 artanh(3/5) = 2 ln 4 ≈ 2.7726 < 2.84`),
   `skewSq c (1/c) > skewSq c y_min` strictly!
2. The exact interior kurtosis maximum `u_* = (2c² - 1)/c` lies in `(1/c, 2t)` iff `t⁴ + 2t² - 1 > 0`,
   and strictly beats `K_ARW(A, 2)` for every `t ∈ (√(√2 - 1), 1)` (i.e. `A > 3.0571 < 3.27`)!
   In particular, at `t = 13/20` (`A = 4 artanh(13/20) ≈ 3.1016 < 3.27`),
   `kurtU c u_* > kurtU c (2t)` strictly! -/
theorem refutation_table1_omega2_skewness {t : ℝ} (ht0 : 0 < t) (ht1 : t < 1) (htgt : 1 / 3 < t ^ 2) :
    let c := (1 + t ^ 2) / (2 * t)
    let y_min := 1 / (2 * t)
    y_min < 1 / c ∧ 1 / c < c ∧ skewSq c y_min < skewSq c (1 / c) := by
  intro c y_min
  have hc_pos : 0 < c := by dsimp [c]; positivity
  have hc_gt1 : 1 < c := by
    dsimp [c]; rw [one_lt_div (by positivity)]
    nlinarith [sq_nonneg (1 - t)]
  have hy_lt : y_min < 1 / c := by
    dsimp [c, y_min]
    rw [one_div_div, div_lt_div_iff₀ (by positivity) (by positivity)]
    nlinarith
  have h1c_lt : 1 / c < c := by
    rw [div_lt_iff₀ hc_pos]; nlinarith
  refine ⟨hy_lt, h1c_lt, ?_⟩
  rw [skewSq_at_opt hc_pos]
  unfold skewSq
  have hy_pos : 0 < y_min := by dsimp [y_min]; positivity
  rw [div_lt_iff₀ (pow_pos hy_pos 3)]
  have hident : 4 * c ^ 3 * y_min ^ 3 - (3 * c * y_min - 1) ^ 2 =
      (c * y_min - 1) ^ 2 * (4 * c * y_min - 1) := by ring
  have hne : c * y_min ≠ 1 := by
    intro heq
    have : y_min = 1 / c := by field_simp [ne_of_gt hc_pos]; linarith
    linarith
  have hsq : 0 < (c * y_min - 1) ^ 2 := sq_pos_of_ne_zero (sub_ne_zero.mpr hne)
  have h4 : 0 < 4 * c * y_min - 1 := by
    dsimp [c, y_min]
    have hne_t : t ≠ 0 := ne_of_gt ht0
    have hcalc : 4 * ((1 + t ^ 2) / (2 * t)) * (1 / (2 * t)) - 1 = 1 / t ^ 2 := by
      field_simp [hne_t]; ring
    rw [hcalc]
    positivity
  nlinarith

theorem refutation_table1_omega2_kurtosis {t : ℝ} (ht0 : 0 < t) (ht1 : t < 1)
    (htgt : 0 < t ^ 4 + 2 * t ^ 2 - 1) :
    let c := (1 + t ^ 2) / (2 * t)
    let u_max := 2 * t
    let u_opt := (2 * c ^ 2 - 1) / c
    1 / c < u_opt ∧ u_opt < u_max ∧ kurtU c u_max < kurtU c u_opt := by
  intro c u_max u_opt
  have hc_pos : 0 < c := by dsimp [c]; positivity
  have hc_gt1 : 1 < c := by
    dsimp [c]; rw [one_lt_div (by positivity)]
    nlinarith [sq_nonneg (1 - t)]
  have h1 : 1 / c < u_opt := by
    dsimp [u_opt]
    rw [div_lt_div_iff₀ hc_pos hc_pos]
    nlinarith
  have h2 : u_opt < u_max := by
    dsimp [u_opt, u_max, c]
    have hne : t ≠ 0 := ne_of_gt ht0
    have hpos : 0 < 1 + t ^ 2 := by positivity
    rw [div_lt_iff₀ (by positivity)]
    have hcalc : (2 * ((1 + t ^ 2) / (2 * t)) ^ 2 - 1) - 2 * t * ((1 + t ^ 2) / (2 * t)) =
        -(t ^ 4 + 2 * t ^ 2 - 1) / (2 * t ^ 2) := by
      field_simp [hne]; ring
    have hneg : -(t ^ 4 + 2 * t ^ 2 - 1) / (2 * t ^ 2) < 0 :=
      div_neg_of_neg_of_pos (by linarith) (by positivity)
    linarith
  refine ⟨h1, h2, ?_⟩
  have hdiff : kurtU c u_opt - kurtU c u_max = 3 * c * (u_max - u_opt) ^ 2 := by
    dsimp [kurtU, u_opt]
    have hne : c ≠ 0 := ne_of_gt hc_pos
    field_simp [hne]; ring
  have hpos : 0 < 3 * c * (u_max - u_opt) ^ 2 := by
    apply mul_pos (by positivity) (sq_pos_of_ne_zero (sub_ne_zero.mpr (ne_of_gt h2)))
  linarith

/-- **Theorem 5 (Universal Maclaurin Positivity `c γ₃ - γ₄ ≥ 0` for ALL `Ω ≥ 4` and ALL `A > 0`)**:
Given `η = coth(A/4) > 1`, `c = coth(A/2) = (η² + 1) / (2η)`, and `e₃, e₄ ≥ 0` satisfying
Maclaurin's inequality `e₃ ≥ 4 e₄` (since `e₁ = ∑ w_i = 1` and `e₁ e₃ ≥ 4Ω/(Ω-3) e₄ ≥ 4 e₄`),
we have `c γ₃ - γ₄ = η² (c e₃ - η e₄) ≥ η (η² + 2) e₄ ≥ 0`! -/
theorem universal_maclaurin_positivity {η e₃ e₄ : ℝ} (hη : 1 < η) (he₄ : 0 ≤ e₄)
    (hmac : 4 * e₄ ≤ e₃) :
    let c := (η ^ 2 + 1) / (2 * η)
    let γ₃ := η ^ 2 * e₃
    let γ₄ := η ^ 3 * e₄
    η * (η ^ 2 + 2) * e₄ ≤ c * γ₃ - γ₄ ∧ 0 ≤ c * γ₃ - γ₄ := by
  intro c γ₃ γ₄
  have hη_pos : 0 < η := by linarith
  have hη_ne : η ≠ 0 := ne_of_gt hη_pos
  have h1 : c * γ₃ - γ₄ = (η / 2) * ((η ^ 2 + 1) * e₃ - 2 * η ^ 2 * e₄) := by
    dsimp [c, γ₃, γ₄]; field_simp [hη_ne]
  have h2 : η * (η ^ 2 + 2) * e₄ ≤ (η / 2) * ((η ^ 2 + 1) * e₃ - 2 * η ^ 2 * e₄) := by
    have hmul : (η ^ 2 + 1) * (4 * e₄) ≤ (η ^ 2 + 1) * e₃ :=
      mul_le_mul_of_nonneg_left hmac (by positivity)
    nlinarith
  refine ⟨by linarith, ?_⟩
  have : 0 ≤ η * (η ^ 2 + 2) * e₄ := by positivity
  linarith

/-- Hyperbolic bridge connecting `A > 0`, `t = tanh(A/4)`, and `c = coth(A/2) = (1 + t²)/(2t)`. -/
theorem coth_half_eq_tanh_quarter {A : ℝ} (hA : 0 < A) :
    Real.cosh (A / 2) / Real.sinh (A / 2) =
      (1 + Real.tanh (A / 4) ^ 2) / (2 * Real.tanh (A / 4)) := by
  have h4 : 0 < A / 4 := by linarith
  have hs : Real.sinh (A / 4) ≠ 0 := ne_of_gt (Real.sinh_pos_iff.mpr h4)
  have hc : Real.cosh (A / 4) ≠ 0 := ne_of_gt (Real.cosh_pos (A / 4))
  have h2s : Real.sinh (A / 2) = 2 * Real.sinh (A / 4) * Real.cosh (A / 4) := by
    have : A / 2 = 2 * (A / 4) := by ring
    rw [this, Real.sinh_two_mul]
  have h2c : Real.cosh (A / 2) = Real.cosh (A / 4) ^ 2 + Real.sinh (A / 4) ^ 2 := by
    have : A / 2 = 2 * (A / 4) := by ring
    rw [this, Real.cosh_two_mul]
  rw [h2c, h2s, Real.tanh_eq_sinh_div_cosh]
  field_simp [hs, hc]

/-- Concrete rational witness at `t = 3/5` (`A = 4 artanh(3/5) = 2 ln 4 ≈ 2.7726 < 2.84`). -/
theorem skewness_witness_t_3_5 :
    let t : ℝ := 3 / 5
    let c := (1 + t ^ 2) / (2 * t)
    let y_min := 1 / (2 * t)
    y_min < 1 / c ∧ 1 / c < c ∧ skewSq c y_min < skewSq c (1 / c) :=
  refutation_table1_omega2_skewness (by norm_num) (by norm_num) (by norm_num)

/-- Concrete rational witness at `t = 13/20` (`A = 4 artanh(13/20) ≈ 3.1016 < 3.27`). -/
theorem kurtosis_witness_t_13_20 :
    let t : ℝ := 13 / 20
    let c := (1 + t ^ 2) / (2 * t)
    let u_max := 2 * t
    let u_opt := (2 * c ^ 2 - 1) / c
    1 / c < u_opt ∧ u_opt < u_max ∧ kurtU c u_max < kurtU c u_opt :=
  refutation_table1_omega2_kurtosis (by norm_num) (by norm_num) (by norm_num)

end WamplerBaratoStochThermo
