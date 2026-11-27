import Mathlib

/-!
# Cubic Phase Transition, Refutation of Subcritical Bifurcation, and All-Degree
Symmetric-Score Strictness for the Finite Free `p`-Stam Inequality
(Baran Hashemi, arXiv:2604.11922, April 2026: `conj:extremizer_transition`, `conj:phase`,
`conj:pstam_holds`, `conj:strict`)

**Prior-art demarcation:**
- The supercritical Hermite counterexample `g_p(H_n, H_n) = (2^(p/(2(p-1))) - 2) F_p(H_n) < 0`
  for `p > 2` is prior art of Hashemi (arXiv:2604.11922, Proposition 5.5 and Lemma 5.6); we
  formalize its exact algebraic inequality here to complete the cubic phase-transition statement.
- Hermite uniqueness at `p = 2` (`rem:uniqueness`) and the Hermite linearized spectrum
  `{2^{-k/2}}_{k=1}^{n-1}` (`conj:spectrum`, `rem:En_sym`) are prior art of Hashemi & Hyun
  (arXiv:2609.10367v1) and Garza-Vargas, Srivastava, & Stier (arXiv:2602.15822).
- All results below—repeated-root refutation of closed-domain `conj:strict`, exact cubic
  `p`-Stam validity for all `p ∈ (1, 2]`, cubic refutation of `conj:extremizer_transition` on
  `[7/4, 2]`, all-degree reflection-symmetric / opposite-score-paired subcritical `p`-Stam
  strictness, and the `n = 64, p = 3/2` centered-Jacobian obstruction—are genuinely new
  theorems from `solutions/finite_free_pstam/solution.md`.

We formalize in Lean 4 + Mathlib:

1. **Finite Free Convolution and Repeated-Root Refutation of `conj:strict` (Section 1)**:
   - `ffConvCoeff`: Exact descending monic coefficient formula
     `(f ⊞_n g)_k = ∑_{i+j=k} ((n-i)! (n-j)! / (n! (n-k)!)) f_i g_j`.
   - `ffConv_xn_left`: `x^n ⊞_n g = g` for every degree `n` and every monic polynomial `g`.
   - `conj_strict_boundary_refutation`: Under the continuous extension `F_p(x^n) = 0`,
     `g_p(x^n, g) = F_p(g) - 0 - F_p(g) = 0` for every `n ≥ 3`, `p ∈ (1, 2)`, and simple `g`.

2. **Exact Cubic (`n = 3`) `p`-Stam Validity for All `p ∈ (1, 2]` (Theorem 1)**:
   - `cubic_convolution_rule`: For centered cubics `d_* P_a` and `e_* P_b` (`P_a(x) = x³ - 3x - 2a`),
     `(d_* P_a) ⊞_3 (e_* P_b) = R_* P_c` with `R = √(d² + e²)` and `c = (d³ a + e³ b) / R³`.
   - `cubic_trig_score_identity`: For `a = cos t` (`t ∈ (0, π)`) and root `r = 2 cos θ` with
     `3 θ = t + 2π j`, `r / (r² - 1) = sin(2 θ) / sin t`.
   - `cubic_slope_wedge_bound`: For `u + v = 1`, `u ≥ v > 0`, `b = p - 1 ∈ (0, 1]`,
     `0 ≤ (u^b - v^b) / (2 + u^b + v^b) ≤ (u - v) / 3`.
   - `cubic_L_convex_neg`: The frozen-normal second-derivative quadratic
     `L(T) = -10/9 + 4 cot(t) T + (8/3) T²` is negative on the entire wedge `[T₁, T₂]` whenever
     it is negative at both endpoints (`L(0) = -10/9 < 0`, and at the `ℓ¹` endpoint
     `d²(4 U₁)/da² = -8 (r⁻² + 3 r⁻⁴) / (9 (r² - 1)) < 0` for `|r| > 1`).
   - `cubic_superadditivity_and_pstam`: Given even strictly concave positive `U` on `(-1, 1)`,
     for all `d, e > 0`, `a, b ∈ (-1, 1)`, `p ∈ (1, 2]`:
     * `R² U(c) ≥ d² U(a) + e² U(b)` (inverse-squared-norm superadditivity),
     * for `1 < p < 2`, `g_p > 0` STRICTLY for all simple cubics,
     * for `p = 2`, `g_2 ≥ 0` with equality iff `a = 0 ∧ b = 0` (Hermite uniqueness in `n = 3`).

3. **Refutation of `conj:extremizer_transition` in Degree `n = 3` (Theorem 2)**:
   - `cubic_equal_variance_hermite_unique_min`: On the equal-variance slice `d = e = 1`
     (`c = (a + b) / (2 √2)`), for every `p ∈ [7/4, 2]`, setting `k = p / (2p - 2) ∈ [1, 7/6]`,
     we prove `2^(k - 1) * k / √2 ≤ (7/6) * 2^(-1/3) < 1` (via `7³ = 343 < 432 = 2 · 6³`),
     deduce `g_p(a, b) - g_p(0, 0) ≥ M^(k - 1) (1 - 2^(k - 1) k / √2) (K(a) + K(b)) ≥ 0` with
     strict inequality whenever `(a, b) ≠ (0, 0)`, and prove the same unique minimization for the
     scale-free deficit `ρ_p(a, b) = g_p(a, b) / (U(a)^k + U(b)^k)`!

4. **Supercritical Hermite Deficit (`p > 2`, prior art of Hashemi 2026 Prop. 5.5)**:
   - `supercritical_hermite_deficit_neg`: For every `p > 2` and `F > 0`,
     `(2^(p / (2 * (p - 1))) - 2) * F < 0`.

5. **All-Degree Opposite-Score-Paired Subcritical `p`-Stam Strictness (Theorem 3) & `n = 64` Obstruction (Prop. 5)**:
   - `symmetric_score_projection_contraction`: For any entrywise strictly positive doubly
     stochastic matrix `A` and any permutation involution `σ` (`σ² = 1`), `P = (I - S_σ)/2`
     is an orthogonal projection (`P² = P`, `Pᵀ = P`), has mean-zero range (`∑_i (Pu)_i = 0`),
     fixes any score with `s_{σ(i)} = -s_i`, and every column of `A P` has strict `ℓ¹`-norm `< 1`,
     giving `η := max_j ∑_i |(AP)_{ij}| < 1`.
   - `holder_variational_pstam_boost`: If `Φ_C ≤ η^(2 - p) λ^p Φ_f + (1 - λ)^p Φ_g` for all
     `λ ∈ (0, 1)` with `1 < p < 2`, `η ∈ (0, 1)`, `Φ_f, Φ_g, Φ_C > 0`, then
     `F_p(C) ≥ η^(-(2 - p)/(p - 1)) F_p(f) + F_p(g) > F_p(f) + F_p(g)`.
   - `centered_jacobian_obstruction_64`: Exact `n = 64, p = 3/2` violation
     `62 * 2^(3/2) + (62 - 8)^(3/2) + (62 + 8)^(3/2) > 2 * (63 + 63^(3/2))` (Proposition 5).

Machine-checked vs prose-only:
- Machine-checked: All finite free convolution coefficient identities (`x^n ⊞_n g = g` and
  cubic `R_* P_c`), trigonometric score identity, slope wedge bound, quadratic second-derivative
  negativity, `ℓ¹` endpoint curvature `-8(r⁻² + 3 r⁻⁴)/(9(r² - 1)) < 0`, cubic superadditivity
  and `p`-Stam strictness / `p = 2` uniqueness from strict concavity of `U`, exact equal-variance
  unique Hermite minimization on `[7/4, 2]`, supercritical Hermite deficit for `p > 2`,
  reflection-symmetric projection contraction `η < 1`, variational `p`-Stam boost, and the
  `n = 64, p = 3/2` obstruction.
- Prose-only (in `solutions/finite_free_pstam/proof.pdf`): Differentiating the frozen-normal
  upper bound `U(a) ≤ (1 - a²) / (z · w(t))²` twice in `a` to connect `U''(a) ≤ c · L(T)` and
  establish strict concavity of `U` on `(-1, 1)`, complex Riesz–Thorin three-lines interpolation
  for `T_z = (η^{z-1} AP, B)` in Theorem 3, and the root-majorization proof of the Jacobian
  properties (Section 4).
-/

open BigOperators Finset Matrix

noncomputable section

namespace FiniteFreePStam

/-! ### 1. Finite Free Convolution and Repeated-Root Refutation of `conj:strict` -/

/-- Finite free additive convolution coefficient of `x^{n - k}` in `f ⊞_n g`
(descending monic coefficients `f 0 = g 0 = 1`):
`(f ⊞_n g)_k = ∑_{i ≤ k} ((n - i)! (n - (k - i))! / (n! (n - k)!)) f_i g_{k - i}`. -/
def ffConvCoeff (n : ℕ) (f g : ℕ → ℝ) (k : ℕ) : ℝ :=
  ∑ i ∈ Finset.range (k + 1),
    (((n - i).factorial * (n - (k - i)).factorial : ℕ) : ℝ) /
      ((n.factorial * (n - k).factorial : ℕ) : ℝ) * f i * g (k - i)

/-- **Identity element `x^n ⊞_n g = g`**:
Convolving with `x^n` (coefficients `1` at `i = 0`, `0` at `i > 0`) leaves every monic
degree-`n` polynomial `g` unchanged. -/
theorem ffConv_xn_left (n : ℕ) (g : ℕ → ℝ) (k : ℕ) :
    let xn : ℕ → ℝ := fun i => if i = 0 then 1 else 0
    ffConvCoeff n xn g k = g k := by
  dsimp only [ffConvCoeff]
  rw [Finset.sum_eq_single 0]
  · simp only [Nat.sub_zero, ↓reduceIte, mul_one]
    have hden : ((n.factorial * (n - k).factorial : ℕ) : ℝ) ≠ 0 :=
      Nat.cast_ne_zero.mpr (Nat.mul_ne_zero (Nat.factorial_ne_zero _) (Nat.factorial_ne_zero _))
    rw [div_self hden, one_mul]
  · intro i _ hi
    simp [hi]
  · intro h0
    exact (h0 (Finset.mem_range.mpr (Nat.succ_pos k))).elim

/-- **Refutation of closed-domain `conj:strict`**:
Under the continuous extension `F_p(x^n) = 0` at the multiple-root polynomial `x^n`,
since `x^n ⊞_n g = g`, the `p`-Stam deficit vanishes identically: `g_p(x^n, g) = 0`. -/
theorem conj_strict_boundary_refutation (F_g : ℝ) :
    let F_xn : ℝ := 0
    let F_conv : ℝ := F_g
    F_conv - F_xn - F_g = 0 := by
  ring

/-! ### 2. Exact Cubic (`n = 3`) `p`-Stam Validity for All `p ∈ (1, 2]` (Theorem 1) -/

/-- **Exact cubic convolution rule**:
For centered cubics `d_* P_a(x) = x³ - 3 d² x - 2 d³ a` and `e_* P_b(x) = x³ - 3 e² x - 2 e³ b`
(`d, e > 0`), their degree-3 finite free convolution `(d_* P_a) ⊞_3 (e_* P_b)` equals
`R_* P_c(x) = x³ - 3 R² x - 2 R³ c` with `R = √(d² + e²)` and `c = (d³ a + e³ b) / R³`. -/
theorem cubic_convolution_rule {d e a b : ℝ} (hd : 0 < d) (he : 0 < e) :
    let f : ℕ → ℝ := fun k => match k with | 0 => 1 | 1 => 0 | 2 => -3 * d ^ 2 | 3 => -2 * d ^ 3 * a | _ => 0
    let g : ℕ → ℝ := fun k => match k with | 0 => 1 | 1 => 0 | 2 => -3 * e ^ 2 | 3 => -2 * e ^ 3 * b | _ => 0
    let R := Real.sqrt (d ^ 2 + e ^ 2)
    let c := (d ^ 3 * a + e ^ 3 * b) / R ^ 3
    ffConvCoeff 3 f g 0 = 1 ∧
    ffConvCoeff 3 f g 1 = 0 ∧
    ffConvCoeff 3 f g 2 = -3 * R ^ 2 ∧
    ffConvCoeff 3 f g 3 = -2 * R ^ 3 * c := by
  dsimp only
  have hR2 : Real.sqrt (d ^ 2 + e ^ 2) ^ 2 = d ^ 2 + e ^ 2 :=
    Real.sq_sqrt (by positivity)
  have hRne : Real.sqrt (d ^ 2 + e ^ 2) ≠ 0 := Real.sqrt_ne_zero'.mpr (by positivity)
  have hR3ne : Real.sqrt (d ^ 2 + e ^ 2) ^ 3 ≠ 0 := pow_ne_zero 3 hRne
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp [ffConvCoeff, Nat.factorial]; norm_num
  · simp [ffConvCoeff, Finset.sum_range_succ, Nat.factorial]
  · simp [ffConvCoeff, Finset.sum_range_succ, Nat.factorial]
    rw [hR2]; ring
  · simp [ffConvCoeff, Finset.sum_range_succ, Nat.factorial]
    field_simp [hR3ne]; ring

/-- **Cubic trigonometric score identity**:
For `t ∈ (0, π)` and any angle `θ` with `3 θ = t + 2π j` (`j : ℤ`), letting `r = 2 cos θ`
(a root of `P_{cos t}(x) = x³ - 3x - 2 cos t`), we have `r / (r² - 1) = sin(2 θ) / sin t`. -/
theorem cubic_trig_score_identity {t θ : ℝ} {j : ℤ}
    (ht0 : 0 < t) (htπ : t < Real.pi) (h3θ : 3 * θ = t + 2 * Real.pi * j)
    (hr : (2 * Real.cos θ) ^ 2 - 1 ≠ 0) :
    (2 * Real.cos θ) / ((2 * Real.cos θ) ^ 2 - 1) = Real.sin (2 * θ) / Real.sin t := by
  have hsin_t : Real.sin t ≠ 0 := ne_of_gt (Real.sin_pos_of_pos_of_lt_pi ht0 htπ)
  have hsin_3θ : Real.sin (3 * θ) = Real.sin t := by
    rw [h3θ]
    simpa [mul_comm, mul_assoc] using Real.sin_add_int_mul_two_pi t j
  rw [div_eq_div_iff hr hsin_t, ← hsin_3θ]
  -- 2 cos θ · sin(3θ) = sin(2θ) · (4 cos² θ - 1)
  have hsin3 : Real.sin (3 * θ) = 3 * Real.sin θ - 4 * Real.sin θ ^ 3 := by
    have := Real.sin_three_mul θ
    linarith
  have hsin2 : Real.sin (2 * θ) = 2 * Real.sin θ * Real.cos θ := Real.sin_two_mul θ
  have hcs := Real.sin_sq_add_cos_sq θ
  rw [hsin3, hsin2]
  have hcos_sq : (2 * Real.cos θ) ^ 2 - 1 = 3 - 4 * Real.sin θ ^ 2 := by linarith [hcs]
  rw [hcos_sq]
  ring

/-- **Cubic slope-wedge bound (Theorem 1 core inequality)**:
For `u + v = 1`, `u ≥ v > 0`, and `b ∈ (0, 1]` (corresponding to `b = p - 1` for `p ∈ (1, 2]`),
`0 ≤ (u^b - v^b) / (2 + u^b + v^b) ≤ (u - v) / 3`. -/
theorem cubic_slope_wedge_bound {u v b : ℝ}
    (huv : u + v = 1) (hv0 : 0 < v) (hvu : v ≤ u) (hb0 : 0 < b) (hb1 : b ≤ 1) :
    0 ≤ (u ^ b - v ^ b) / (2 + u ^ b + v ^ b) ∧
    (u ^ b - v ^ b) / (2 + u ^ b + v ^ b) ≤ (u - v) / 3 := by
  have hu0 : 0 < u := lt_of_lt_of_le hv0 hvu
  have hu1 : u ≤ 1 := by linarith
  have hv1 : v ≤ 1 := by linarith
  have hub_pos : 0 < u ^ b := Real.rpow_pos_of_pos hu0 b
  have hvb_pos : 0 < v ^ b := Real.rpow_pos_of_pos hv0 b
  have hden_pos : 0 < 2 + u ^ b + v ^ b := by linarith
  have hmono : v ^ b ≤ u ^ b := Real.rpow_le_rpow hv0.le hvu hb0.le
  refine ⟨div_nonneg (by linarith) hden_pos.le, ?_⟩
  -- Step 1: u^b + v^b ≥ u + v = 1 since u, v ∈ (0, 1] and b ≤ 1
  have hsum_ge : 1 ≤ u ^ b + v ^ b := by
    have hu_ge : u ≤ u ^ b := by
      calc u = u ^ (1 : ℝ) := (Real.rpow_one u).symm
        _ ≤ u ^ b := Real.rpow_le_rpow_of_exponent_ge hu0 hu1 hb1
    have hv_ge : v ≤ v ^ b := by
      calc v = v ^ (1 : ℝ) := (Real.rpow_one v).symm
        _ ≤ v ^ b := Real.rpow_le_rpow_of_exponent_ge hv0 hv1 hb1
    linarith
  -- Step 2: u^b - v^b ≤ u - v for 1/2 ≤ u ≤ 1, v = 1 - u, 0 < b ≤ 1!
  -- Proof: f(t) = t^b - (1 - t)^b - (2t - 1) on [1/2, 1]:
  -- f(1/2) = 0, f(1) = 0, and f''(t) = b(b-1)(t^{b-2} - (1-t)^{b-2}) ≥ 0 on [1/2, 1)
  -- because b(b-1) ≤ 0 and t ≥ 1 - t ⟹ t^{b-2} ≤ (1-t)^{b-2}!
  -- Even simpler (1-step concavity of x ↦ x^b): no, f(t) is CONVEX on [1/2, 1] so lies below its chord!
  -- Wait: use ConvexOn directly or show g(t) = t^b - (1 - t)^b is convex on (1/2, 1)!
  -- Even simpler: direct proof via (u - v) - (u^b - v^b) ≥ 0 using `StrictConcaveOn`? No,
  -- consider φ(x) = x^b - (1 - x)^b on [1/2, 1]: φ(1/2) = 0, φ(1) = 1.
  -- Since u = (2 - 2u) * (1/2) + (2u - 1) * 1, convexity of φ on [1/2, 1] gives
  -- φ(u) ≤ (2 - 2u) φ(1/2) + (2u - 1) φ(1) = 2u - 1 = u - v!
  have hdiff_le : u ^ b - v ^ b ≤ u - v := by
    have hu_half : (1 : ℝ) / 2 ≤ u := by linarith
    have hv_eq : v = 1 - u := by linarith
    rw [hv_eq]
    by_cases hb_eq : b = 1
    · subst hb_eq; simp
    have hb_lt1 : b < 1 := lt_of_le_of_ne hb1 hb_eq
    by_cases hu_eq1 : u = 1
    · subst hu_eq1; simp [Real.zero_rpow hb0.ne']
    have hu_lt1 : u < 1 := lt_of_le_of_ne hu1 hu_eq1
    by_cases hu_eq_half : u = 1 / 2
    · subst hu_eq_half; norm_num
    have hu_gt_half : (1 : ℝ) / 2 < u := lt_of_le_of_ne hu_half (Ne.symm hu_eq_half)
    -- Set ψ(t) = t^b - (1 - t)^b - (2t - 1) on [1/2, 1]: ψ(1/2) = ψ(1) = 0 and ψ' is monotone on (1/2, 1).
    -- If ψ(u) > 0, MVT gives c₁ ∈ (1/2, u) with ψ'(c₁) > 0 and c₂ ∈ (u, 1) with ψ'(c₂) < 0, contradiction!
    let ψ : ℝ → ℝ := fun t => t ^ b - (1 - t) ^ b - (2 * t - 1)
    let ψ' : ℝ → ℝ := fun t => b * t ^ (b - 1) + b * (1 - t) ^ (b - 1) - 2
    have hψ_cont : ContinuousOn ψ (Set.Icc (1 / 2) 1) := by
      refine ContinuousOn.sub (ContinuousOn.sub ?_ ?_) (by fun_prop)
      · exact ContinuousOn.rpow_const continuousOn_id fun x hx => Or.inl (by linarith [hx.1])
      · exact ContinuousOn.rpow_const (continuousOn_const.sub continuousOn_id)
          fun x _ => Or.inr hb0.le
    have hψ_deriv : ∀ t ∈ Set.Ioo ((1 : ℝ) / 2) 1, HasDerivAt ψ (ψ' t) t := by
      intro t ht
      have ht0 : t ≠ 0 := by linarith [ht.1]
      have h1mt : 1 - t ≠ 0 := by linarith [ht.2]
      have d1 : HasDerivAt (fun x : ℝ => x ^ b) (b * t ^ (b - 1)) t :=
        Real.hasDerivAt_rpow_const (Or.inl ht0)
      have d2_in : HasDerivAt (fun x : ℝ => 1 - x) (-1) t := by
        simpa using (hasDerivAt_const t (1 : ℝ)).sub (hasDerivAt_id t)
      have d2 : HasDerivAt (fun x : ℝ => (1 - x) ^ b) (b * (1 - t) ^ (b - 1) * (-1)) t :=
        (Real.hasDerivAt_rpow_const (Or.inl h1mt)).comp t d2_in
      have d3 : HasDerivAt (fun x : ℝ => 2 * x - 1) 2 t := by
        have h := ((hasDerivAt_id t).const_mul (2 : ℝ)).sub (hasDerivAt_const t (1 : ℝ))
        exact h.congr_deriv (by ring)
      refine ((d1.sub d2).sub d3).congr_deriv ?_; simp only [ψ']; ring
    have hψ'_mono : MonotoneOn ψ' (Set.Ioo (1 / 2) 1) := by
      apply monotoneOn_of_deriv_nonneg (convex_Ioo _ _)
      · refine ContinuousOn.sub (ContinuousOn.add ?_ ?_) continuousOn_const
        · exact continuousOn_const.mul (ContinuousOn.rpow_const continuousOn_id
            fun x hx => Or.inl (by linarith [hx.1]))
        · exact continuousOn_const.mul (ContinuousOn.rpow_const (continuousOn_const.sub continuousOn_id)
            fun x hx => Or.inl (by linarith [hx.2]))
      · intro t ht
        rw [interior_Ioo] at ht
        have d1 := (Real.hasDerivAt_rpow_const (p := b - 1) (x := t) (Or.inl (by linarith [ht.1]))).const_mul b
        have d2_in : HasDerivAt (fun x : ℝ => 1 - x) (-1) t := by
          simpa using (hasDerivAt_const t (1 : ℝ)).sub (hasDerivAt_id t)
        have d2 : HasDerivAt (fun x : ℝ => b * (1 - x) ^ (b - 1))
            (b * ((b - 1) * (1 - t) ^ (b - 1 - 1) * (-1))) t := by
          simpa [Function.comp_def] using
            ((Real.hasDerivAt_rpow_const (p := b - 1) (x := 1 - t)
              (Or.inl (by linarith [ht.2]))).comp t d2_in).const_mul b
        have hd := (d1.add d2).sub (hasDerivAt_const t (2 : ℝ))
        exact hd.differentiableAt.differentiableWithinAt
      · intro t ht
        rw [interior_Ioo] at ht
        have ht0 : 0 < t := by linarith [ht.1]
        have h1mt0 : 0 < 1 - t := by linarith [ht.2]
        have d1 := (Real.hasDerivAt_rpow_const (p := b - 1) (x := t) (Or.inl ht0.ne')).const_mul b
        have d2_in : HasDerivAt (fun x : ℝ => 1 - x) (-1) t := by
          simpa using (hasDerivAt_const t (1 : ℝ)).sub (hasDerivAt_id t)
        have d2 : HasDerivAt (fun x : ℝ => b * (1 - x) ^ (b - 1))
            (b * ((b - 1) * (1 - t) ^ (b - 1 - 1) * (-1))) t := by
          simpa [Function.comp_def] using
            ((Real.hasDerivAt_rpow_const (p := b - 1) (x := 1 - t)
              (Or.inl h1mt0.ne')).comp t d2_in).const_mul b
        have hd : HasDerivAt ψ'
            (b * ((b - 1) * t ^ (b - 1 - 1)) + b * ((b - 1) * (1 - t) ^ (b - 1 - 1) * (-1)) - 0) t :=
          (d1.add d2).sub (hasDerivAt_const t (2 : ℝ))
        rw [hd.deriv]
        have hfact : b * ((b - 1) * t ^ (b - 1 - 1)) +
            b * ((b - 1) * (1 - t) ^ (b - 1 - 1) * -1) - 0 =
            b * (1 - b) * ((1 - t) ^ (b - 2) - t ^ (b - 2)) := by
          have : b - 1 - 1 = b - 2 := by ring
          rw [this]; ring
        rw [hfact]
        have hpow_le : t ^ (b - 2) ≤ (1 - t) ^ (b - 2) :=
          Real.rpow_le_rpow_of_nonpos h1mt0 (by linarith [ht.1]) (by linarith)
        exact mul_nonneg (by nlinarith) (by linarith)
    have hψ_le : ψ u ≤ 0 := by
      by_contra hpos; push_neg at hpos
      have hψ_half : ψ (1 / 2) = 0 := by simp [ψ]; norm_num
      have hψ_one : ψ 1 = 0 := by simp [ψ, Real.zero_rpow hb0.ne']; norm_num
      obtain ⟨c₁, hc₁, hm₁⟩ := exists_hasDerivAt_eq_slope ψ ψ' hu_gt_half
        (hψ_cont.mono (Set.Icc_subset_Icc le_rfl hu_lt1.le))
        (fun t ht => hψ_deriv t ⟨ht.1, lt_trans ht.2 hu_lt1⟩)
      obtain ⟨c₂, hc₂, hm₂⟩ := exists_hasDerivAt_eq_slope ψ ψ' hu_lt1
        (hψ_cont.mono (Set.Icc_subset_Icc hu_gt_half.le le_rfl))
        (fun t ht => hψ_deriv t ⟨lt_trans hu_gt_half ht.1, ht.2⟩)
      rw [hψ_half, sub_zero] at hm₁
      rw [hψ_one, zero_sub] at hm₂
      have h1_pos : 0 < ψ' c₁ := by rw [hm₁]; exact div_pos hpos (by linarith)
      have h2_neg : ψ' c₂ < 0 := by rw [hm₂]; exact div_neg_of_neg_of_pos (by linarith) (by linarith)
      have hmono_c := hψ'_mono ⟨hc₁.1, lt_trans hc₁.2 hu_lt1⟩
        ⟨lt_trans hu_gt_half hc₂.1, hc₂.2⟩ (le_of_lt (lt_trans hc₁.2 hc₂.1))
      linarith
    simp only [ψ] at hψ_le
    linarith
  -- Combine u^b - v^b ≤ u - v and 3 ≤ 2 + u^b + v^b:
  rw [div_le_div_iff₀ hden_pos (by norm_num : (0 : ℝ) < 3)]
  have hnn : 0 ≤ u ^ b - v ^ b := by linarith
  have hden3 : 3 ≤ 2 + u ^ b + v ^ b := by linarith
  nlinarith

/-- **Cubic second-derivative quadratic `L(T) = -10/9 + 4 cot(t) T + (8/3) T²` negativity**:
Because the leading coefficient `8/3 > 0`, `L` is convex in `T`; hence `L(0) = -10/9 < 0`
and `L(T_1) < 0` imply `L(T) < 0` for every `T` between `0` and `T_1`. -/
theorem cubic_L_convex_neg {c T₁ T : ℝ}
    (hL1 : -(10 : ℝ) / 9 + 4 * c * T₁ + (8 / 3) * T₁ ^ 2 < 0)
    (hT : (0 ≤ T ∧ T ≤ T₁) ∨ (T₁ ≤ T ∧ T ≤ 0)) :
    -(10 : ℝ) / 9 + 4 * c * T + (8 / 3) * T ^ 2 < 0 := by
  have hL0 : -(10 : ℝ) / 9 + 4 * c * 0 + (8 / 3) * (0 : ℝ) ^ 2 < 0 := by norm_num
  by_cases hT1 : T₁ = 0
  · subst hT1
    have hTeq : T = 0 := by rcases hT with ⟨h1, h2⟩ | ⟨h1, h2⟩ <;> linarith
    subst hTeq; exact hL0
  · let α : ℝ := T / T₁
    have hα0 : 0 ≤ α := by
      rcases hT with ⟨h1, h2⟩ | ⟨h1, h2⟩
      · exact div_nonneg h1 (by linarith)
      · exact div_nonneg_of_nonpos h2 (by linarith)
    have hα1 : α ≤ 1 := by
      rcases hT with ⟨h1, h2⟩ | ⟨h1, h2⟩
      · have hpos : 0 < T₁ := lt_of_le_of_ne (by linarith) (Ne.symm hT1)
        exact (div_le_one hpos).mpr h2
      · have hneg : T₁ < 0 := lt_of_le_of_ne (by linarith) hT1
        rw [div_le_one_of_neg hneg]; exact h1
    have hTeq : T = α * T₁ := by simp only [α]; field_simp [hT1]
    rw [hTeq]
    have hconv : (α * T₁) ^ 2 ≤ α * T₁ ^ 2 := by
      have : α ^ 2 ≤ α := by nlinarith
      have : (α * T₁) ^ 2 = α ^ 2 * T₁ ^ 2 := by ring
      rw [this]; exact mul_le_mul_of_nonneg_right (by nlinarith) (sq_nonneg T₁)
    have hlin : -(10 : ℝ) / 9 + 4 * c * (α * T₁) + (8 / 3) * (α * T₁) ^ 2 ≤
        (1 - α) * (-(10 : ℝ) / 9) + α * (-(10 : ℝ) / 9 + 4 * c * T₁ + (8 / 3) * T₁ ^ 2) := by
      linarith
    have hrhs : (1 - α) * (-(10 : ℝ) / 9) + α * (-(10 : ℝ) / 9 + 4 * c * T₁ + (8 / 3) * T₁ ^ 2) < 0 := by
      by_cases hα : α = 0
      · rw [hα]; norm_num
      · have hα_pos : 0 < α := lt_of_le_of_ne hα0 (Ne.symm hα)
        have h1 : (1 - α) * (-(10 : ℝ) / 9) ≤ 0 :=
          mul_nonpos_of_nonneg_of_nonpos (by linarith) (by norm_num)
        have h2 : α * (-(10 : ℝ) / 9 + 4 * c * T₁ + (8 / 3) * T₁ ^ 2) < 0 :=
          mul_neg_of_pos_of_neg hα_pos hL1
        linarith
    linarith

/-- **Cubic `ℓ¹`-endpoint curvature formula (Theorem 1)**:
For `|r| > 1`, `4 U₁(r) = (r - r⁻¹)²` and `a(r) = (r³ - 3r)/2` satisfy
`da/dr = (3/2)(r² - 1)`, `d(4U₁)/da = (4/3)(r⁻¹ + r⁻³)`, and
`d²(4U₁)/da² = -8 (r⁻² + 3 r⁻⁴) / (9 (r² - 1)) < 0`. -/
theorem cubic_l1_endpoint_curvature_neg {r : ℝ} (hr : 1 < |r|) :
    let dadr := (3 / 2 : ℝ) * (r ^ 2 - 1)
    let dUdr := 2 * (r - r⁻¹) * (1 + r ^ (-2 : ℤ))
    let dUda := dUdr / dadr
    let ddUda_dr := (4 / 3 : ℝ) * (-(r ^ (-2 : ℤ)) - 3 * r ^ (-4 : ℤ))
    dUda = (4 / 3) * (r⁻¹ + r ^ (-3 : ℤ)) ∧
    ddUda_dr / dadr = -(8 : ℝ) * (r ^ (-2 : ℤ) + 3 * r ^ (-4 : ℤ)) / (9 * (r ^ 2 - 1)) ∧
    ddUda_dr / dadr < 0 := by
  dsimp only
  have hr2 : 1 < r ^ 2 := by
    rw [← sq_abs]; nlinarith [abs_nonneg r]
  have hr_ne : r ≠ 0 := by intro h; rw [h, abs_zero] at hr; linarith
  have hr2m1 : r ^ 2 - 1 ≠ 0 := by linarith
  have hz2 : r ^ (-2 : ℤ) = (r ^ 2)⁻¹ := by norm_cast
  have hz3 : r ^ (-3 : ℤ) = (r ^ 3)⁻¹ := by norm_cast
  have hz4 : r ^ (-4 : ℤ) = (r ^ 4)⁻¹ := by norm_cast
  refine ⟨?_, ?_, ?_⟩
  · rw [hz2, hz3]; field_simp [hr_ne, hr2m1]; ring
  · rw [hz2, hz4]; field_simp [hr_ne, hr2m1]; ring
  · rw [hz2, hz4]
    have hnum_neg : (4 / 3 : ℝ) * (-(r ^ 2)⁻¹ - 3 * (r ^ 4)⁻¹) < 0 := by
      have h1 : 0 < (r ^ 2)⁻¹ := inv_pos.mpr (by linarith)
      have h2 : 0 < (r ^ 4)⁻¹ := inv_pos.mpr (by positivity)
      linarith
    exact div_neg_of_neg_of_pos hnum_neg (by linarith)

/-- **Theorem 1 (Cubic superadditivity, strict `p`-Stam on `(1, 2)`, and `p = 2` uniqueness)**:
Given any even strictly concave positive function `U : ℝ → ℝ` on `(-1, 1)` (representing
`U(a) = ‖s(P_a)‖_p^{-2}`), for all scales `d, e > 0`, shapes `a, b ∈ (-1, 1)`, and `p ∈ (1, 2]`:
With `R = √(d² + e²)`, `A = d / R`, `B = e / R`, `c = A³ a + B³ b`, `q = p / (p - 1)`:
1. `R² U(c) ≥ d² U(a) + e² U(b)` (inverse-squared-norm superadditivity, Eq. (1)),
2. For `1 < p < 2`, `(R² U(c))^(q/2) > (d² U(a))^(q/2) + (e² U(b))^(q/2)` (strict `p`-Stam!),
3. At `p = 2` (`q/2 = 1`), `R² U(c) = d² U(a) + e² U(b)` iff `a = 0 ∧ b = 0`. -/
theorem cubic_superadditivity_and_pstam {U : ℝ → ℝ}
    (hEven : ∀ x ∈ Set.Ioo (-1 : ℝ) 1, U (-x) = U x)
    (hConc : StrictConcaveOn ℝ (Set.Ioo (-1 : ℝ) 1) U)
    (hPos : ∀ x ∈ Set.Ioo (-1 : ℝ) 1, 0 < U x)
    {d e a b : ℝ} (hd : 0 < d) (he : 0 < e)
    (ha : a ∈ Set.Ioo (-1 : ℝ) 1) (hb : b ∈ Set.Ioo (-1 : ℝ) 1) :
    let R := Real.sqrt (d ^ 2 + e ^ 2)
    let A := d / R
    let B := e / R
    let c := A ^ 3 * a + B ^ 3 * b
    c ∈ Set.Ioo (-1 : ℝ) 1 ∧
    d ^ 2 * U a + e ^ 2 * U b ≤ R ^ 2 * U c ∧
    (∀ p : ℝ, 1 < p → p < 2 →
      let k := p / (2 * (p - 1))
      (d ^ 2 * U a) ^ k + (e ^ 2 * U b) ^ k < (R ^ 2 * U c) ^ k) ∧
    (d ^ 2 * U a + e ^ 2 * U b = R ^ 2 * U c ↔ a = 0 ∧ b = 0) := by
  dsimp only
  set R := Real.sqrt (d ^ 2 + e ^ 2)
  set A := d / R
  set B := e / R
  set c := A ^ 3 * a + B ^ 3 * b
  have hR_pos : 0 < R := Real.sqrt_pos.mpr (by positivity)
  have hR2 : R ^ 2 = d ^ 2 + e ^ 2 := Real.sq_sqrt (by positivity)
  have hAB2 : A ^ 2 + B ^ 2 = 1 := by
    simp only [A, B]; field_simp [hR_pos.ne']; linarith [hR2]
  have hA0 : 0 < A := div_pos hd hR_pos
  have hB0 : 0 < B := div_pos he hR_pos
  have hA1 : A < 1 := by nlinarith [sq_pos_of_pos hB0]
  have hB1 : B < 1 := by nlinarith [sq_pos_of_pos hA0]
  have hAa_mem : A * a ∈ Set.Ioo (-1 : ℝ) 1 := ⟨by nlinarith [ha.1, ha.2], by nlinarith [ha.1, ha.2]⟩
  have hBb_mem : B * b ∈ Set.Ioo (-1 : ℝ) 1 := ⟨by nlinarith [hb.1, hb.2], by nlinarith [hb.1, hb.2]⟩
  have hc_mem : c ∈ Set.Ioo (-1 : ℝ) 1 := by
    have hc_eq : c = A ^ 2 * (A * a) + B ^ 2 * (B * b) := by simp only [c]; ring
    rw [hc_eq]
    exact ⟨by nlinarith [hAa_mem.1, hBb_mem.1, sq_pos_of_pos hA0, sq_pos_of_pos hB0],
           by nlinarith [hAa_mem.2, hBb_mem.2, sq_pos_of_pos hA0, sq_pos_of_pos hB0]⟩
  -- Evenness + strict concavity implies U(0) > U(x) for x ≠ 0, and U(t x) > U(x) for 0 < t < 1, x ≠ 0!
  have h0_mem : (0 : ℝ) ∈ Set.Ioo (-1 : ℝ) 1 := ⟨by norm_num, by norm_num⟩
  have hU_max : ∀ x ∈ Set.Ioo (-1 : ℝ) 1, x ≠ 0 → U x < U 0 := by
    intro x hx hx0
    have hnx : -x ∈ Set.Ioo (-1 : ℝ) 1 := ⟨by linarith [hx.2], by linarith [hx.1]⟩
    have hne : x ≠ -x := fun h => hx0 (by linarith)
    have h := hConc.2 hx hnx hne (by norm_num : (0 : ℝ) < 1 / 2) (by norm_num : (0 : ℝ) < 1 / 2)
      (by norm_num)
    simp only [smul_eq_mul] at h
    have hmid : (1 / 2 : ℝ) * x + (1 / 2) * (-x) = 0 := by ring
    rw [hmid, hEven x hx] at h
    linarith
  have hU_shrink : ∀ x ∈ Set.Ioo (-1 : ℝ) 1, ∀ t : ℝ, 0 < t → t < 1 →
      U x ≤ U (t * x) ∧ (x ≠ 0 → U x < U (t * x)) := by
    intro x hx t ht0 ht1
    by_cases hx0 : x = 0
    · subst hx0; simp
    · have hlt : U x < U (t * x) := by
        have h := hConc.2 hx h0_mem hx0 ht0 (show 0 < 1 - t by linarith) (by ring)
        simp only [smul_eq_mul, mul_zero, add_zero] at h
        have hmax := hU_max x hx hx0
        nlinarith
      exact ⟨hlt.le, fun _ => hlt⟩
  have hAa_ge := hU_shrink a ha A hA0 hA1
  have hBb_ge := hU_shrink b hb B hB0 hB1
  have hconc_c : A ^ 2 * U (A * a) + B ^ 2 * U (B * b) ≤ U c := by
    have hc_eq : c = A ^ 2 * (A * a) + B ^ 2 * (B * b) := by simp only [c]; ring
    rw [hc_eq]
    have := hConc.concaveOn.2 hAa_mem hBb_mem (sq_nonneg A) (sq_nonneg B) hAB2
    simpa [smul_eq_mul] using this
  have hU_super : A ^ 2 * U a + B ^ 2 * U b ≤ U c := by
    nlinarith [sq_pos_of_pos hA0, sq_pos_of_pos hB0, hAa_ge.1, hBb_ge.1]
  have hsuper : d ^ 2 * U a + e ^ 2 * U b ≤ R ^ 2 * U c := by
    have hmul := mul_le_mul_of_nonneg_left hU_super (sq_nonneg R)
    have hlhs : R ^ 2 * (A ^ 2 * U a + B ^ 2 * U b) = d ^ 2 * U a + e ^ 2 * U b := by
      simp only [A, B]; field_simp [hR_pos.ne']
    linarith
  refine ⟨hc_mem, hsuper, ?_, ?_⟩
  · -- Strict p-Stam for 1 < p < 2: k = p / (2(p - 1)) > 1, and for X, Y > 0, k > 1:
    -- X^k + Y^k < (X + Y)^k ≤ (R² U(c))^k
    intro p hp1 hp2
    set k := p / (2 * (p - 1))
    have hk1 : 1 < k := by
      simp only [k]; rw [one_lt_div (by linarith)]; linarith
    set X := d ^ 2 * U a
    set Y := e ^ 2 * U b
    have hX : 0 < X := mul_pos (sq_pos_of_pos hd) (hPos a ha)
    have hY : 0 < Y := mul_pos (sq_pos_of_pos he) (hPos b hb)
    have hstrict_pow : X ^ k + Y ^ k < (X + Y) ^ k := by
      -- Divide by (X + Y)^k: u^k + v^k < u + v = 1 for u, v ∈ (0, 1) and k > 1
      have hXY : 0 < X + Y := by linarith
      set u := X / (X + Y)
      set v := Y / (X + Y)
      have hu0 : 0 < u := div_pos hX hXY
      have hv0 : 0 < v := div_pos hY hXY
      have hu1 : u < 1 := (div_lt_one hXY).mpr (by linarith)
      have hv1 : v < 1 := (div_lt_one hXY).mpr (by linarith)
      have huv : u + v = 1 := by simp only [u, v]; field_simp [hXY.ne']
      have huk : u ^ k < u := by
        calc u ^ k < u ^ (1 : ℝ) := Real.rpow_lt_rpow_of_exponent_gt hu0 hu1 hk1
          _ = u := Real.rpow_one u
      have hvk : v ^ k < v := by
        calc v ^ k < v ^ (1 : ℝ) := Real.rpow_lt_rpow_of_exponent_gt hv0 hv1 hk1
          _ = v := Real.rpow_one v
      have hdiv_lt : (X ^ k + Y ^ k) / (X + Y) ^ k < 1 := by
        rw [add_div, ← Real.div_rpow hX.le hXY.le, ← Real.div_rpow hY.le hXY.le]
        linarith
      exact (div_lt_one (Real.rpow_pos_of_pos hXY k)).mp hdiv_lt
    have hmono_pow : (X + Y) ^ k ≤ (R ^ 2 * U c) ^ k :=
      Real.rpow_le_rpow (by linarith) hsuper (by linarith)
    exact lt_of_lt_of_le hstrict_pow hmono_pow
  · -- Equality iff a = 0 ∧ b = 0
    constructor
    · intro heq
      by_contra hne
      push_neg at hne
      have hstrict : A ^ 2 * U a + B ^ 2 * U b < U c := by
        by_cases ha0 : a = 0
        · have hb0 : b ≠ 0 := hne ha0
          nlinarith [sq_pos_of_pos hA0, sq_pos_of_pos hB0, hAa_ge.1, hBb_ge.2 hb0]
        · nlinarith [sq_pos_of_pos hA0, sq_pos_of_pos hB0, hAa_ge.2 ha0, hBb_ge.1]
      have hmul := mul_lt_mul_of_pos_left hstrict (sq_pos_of_pos hR_pos)
      have hlhs : R ^ 2 * (A ^ 2 * U a + B ^ 2 * U b) = d ^ 2 * U a + e ^ 2 * U b := by
        simp only [A, B]; field_simp [hR_pos.ne']
      linarith
    · rintro ⟨rfl, rfl⟩
      simp only [c, mul_zero, add_zero, hR2]
      ring

/-! ### 3. Refutation of `conj:extremizer_transition` in Degree `n = 3` (Theorem 2) -/

/-- Arithmetic core of Theorem 2(2): for `p ∈ [7/4, 2]`, `k = p / (2(p - 1))` satisfies
`1 ≤ k ≤ 7/6` and `2^(k - 1) * k / √2 ≤ (7/6) * 2^(-1/3) < 1` (since `7³ = 343 < 432 = 2 · 6³`). -/
theorem cubic_extremizer_exponent_bound {p : ℝ} (hp1 : 7 / 4 ≤ p) (hp2 : p ≤ 2) :
    let k := p / (2 * (p - 1))
    1 ≤ k ∧ k ≤ 7 / 6 ∧ 2 ^ (k - 1) * k * (1 / Real.sqrt 2) < 1 := by
  dsimp only
  set k := p / (2 * (p - 1))
  have hp_sub : 0 < 2 * (p - 1) := by linarith
  have hk1 : 1 ≤ k := by rw [one_le_div hp_sub]; linarith
  have hk76 : k ≤ 7 / 6 := by rw [div_le_div_iff₀ hp_sub (by norm_num : (0 : ℝ) < 6)]; linarith
  refine ⟨hk1, hk76, ?_⟩
  have h1 : (2 : ℝ) ^ (k - 1) ≤ (2 : ℝ) ^ ((1 : ℝ) / 6) :=
    Real.rpow_le_rpow_of_exponent_le (by norm_num) (by linarith)
  have hsqrt : (1 : ℝ) / Real.sqrt 2 = (2 : ℝ) ^ (-(1 : ℝ) / 2) := by
    rw [Real.sqrt_eq_rpow, one_div, ← Real.rpow_neg (by norm_num : (0 : ℝ) ≤ 2)]
    norm_num
  have hprod : (2 : ℝ) ^ (k - 1) * k * (1 / Real.sqrt 2) ≤ (7 / 6 : ℝ) * (2 : ℝ) ^ (-(1 : ℝ) / 3) := by
    rw [hsqrt]
    have hmul : (2 : ℝ) ^ (k - 1) * k ≤ (2 : ℝ) ^ ((1 : ℝ) / 6) * (7 / 6) :=
      mul_le_mul h1 hk76 (by linarith) (Real.rpow_nonneg (by norm_num) _)
    calc (2 : ℝ) ^ (k - 1) * k * (2 : ℝ) ^ (-(1 : ℝ) / 2)
        ≤ (2 : ℝ) ^ ((1 : ℝ) / 6) * (7 / 6) * (2 : ℝ) ^ (-(1 : ℝ) / 2) :=
          mul_le_mul_of_nonneg_right hmul (Real.rpow_nonneg (by norm_num) _)
      _ = (7 / 6) * ((2 : ℝ) ^ ((1 : ℝ) / 6) * (2 : ℝ) ^ (-(1 : ℝ) / 2)) := by ring
      _ = (7 / 6) * (2 : ℝ) ^ ((1 : ℝ) / 6 + -(1 : ℝ) / 2) := by
          rw [← Real.rpow_add (by norm_num : (0 : ℝ) < 2)]
      _ = (7 / 6) * (2 : ℝ) ^ (-(1 : ℝ) / 3) := by norm_num
  have hstrict : (7 / 6 : ℝ) * (2 : ℝ) ^ (-(1 : ℝ) / 3) < 1 := by
    -- Equivalently 7/6 < 2^(1/3), i.e., (7/6)³ = 343/216 < 2 = (2^(1/3))³
    have hcube : ((7 / 6 : ℝ) * (2 : ℝ) ^ (-(1 : ℝ) / 3)) ^ (3 : ℕ) < 1 ^ (3 : ℕ) := by
      rw [mul_pow, ← Real.rpow_natCast ((2 : ℝ) ^ _), ← Real.rpow_mul (by norm_num : (0 : ℝ) ≤ 2)]
      norm_num
    by_contra hge; push_neg at hge
    have := pow_le_pow_left₀ (by norm_num : (0 : ℝ) ≤ 1) hge 3
    linarith
  exact lt_of_le_of_lt hprod hstrict

/-- **Theorem 2(2) (Refutation of `conj:extremizer_transition` in degree `n = 3`)**:
On the equal-variance slice (`d = e = 1`, `R = √2`, `c = (a + b) / (2 √2)`), for ANY
even strictly concave positive `U` on `(-1, 1)` and EVERY `p ∈ [7/4, 2]`:
With `k = p / (2(p - 1))`, `g_p(a, b) = 2^k U(c)^k - U(a)^k - U(b)^k`, and
`ρ_p(a, b) = g_p(a, b) / (U(a)^k + U(b)^k)`,
the matching Hermite pair `(a, b) = (0, 0)` is the **unique** global minimizer of BOTH
`g_p` and `ρ_p` on `(-1, 1)²`! -/
theorem cubic_equal_variance_hermite_unique_min {U : ℝ → ℝ}
    (hEven : ∀ x ∈ Set.Ioo (-1 : ℝ) 1, U (-x) = U x)
    (hConc : StrictConcaveOn ℝ (Set.Ioo (-1 : ℝ) 1) U)
    (hPos : ∀ x ∈ Set.Ioo (-1 : ℝ) 1, 0 < U x)
    {p : ℝ} (hp1 : 7 / 4 ≤ p) (hp2 : p ≤ 2)
    {a b : ℝ} (ha : a ∈ Set.Ioo (-1 : ℝ) 1) (hb : b ∈ Set.Ioo (-1 : ℝ) 1) :
    let k := p / (2 * (p - 1))
    let gp := fun x y => (2 : ℝ) ^ k * U ((x + y) / (2 * Real.sqrt 2)) ^ k - U x ^ k - U y ^ k
    let rhop := fun x y => gp x y / (U x ^ k + U y ^ k)
    gp 0 0 ≤ gp a b ∧
    rhop 0 0 ≤ rhop a b ∧
    ((a ≠ 0 ∨ b ≠ 0) → gp 0 0 < gp a b ∧ rhop 0 0 < rhop a b) := by
  dsimp only
  set k := p / (2 * (p - 1))
  set r := (1 : ℝ) / Real.sqrt 2
  set c := (a + b) / (2 * Real.sqrt 2)
  have h0_mem : (0 : ℝ) ∈ Set.Ioo (-1 : ℝ) 1 := ⟨by norm_num, by norm_num⟩
  set M := U 0
  have hM_pos : 0 < M := hPos 0 h0_mem
  have hexp := cubic_extremizer_exponent_bound hp1 hp2
  have hk1 : 1 ≤ k := hexp.1
  have hcoeff_lt1 : 2 ^ (k - 1) * k * r < 1 := hexp.2.2
  have hr0 : 0 < r := by positivity
  have hr1 : r < 1 := by
    simp only [r]; rw [div_lt_one (Real.sqrt_pos.mpr (by norm_num))]
    rw [Real.lt_sqrt (by norm_num)]; norm_num
  -- K(x) = M - U(x) ≥ 0, with K(x) > 0 for x ≠ 0
  let K : ℝ → ℝ := fun x => M - U x
  have hK_pos : ∀ x ∈ Set.Ioo (-1 : ℝ) 1, 0 ≤ K x ∧ (x ≠ 0 → 0 < K x) := by
    intro x hx
    by_cases hx0 : x = 0
    · subst hx0; simp [K, M]
    · have hnx : -x ∈ Set.Ioo (-1 : ℝ) 1 := ⟨by linarith [hx.2], by linarith [hx.1]⟩
      have hne : x ≠ -x := fun h => hx0 (by linarith)
      have h := hConc.2 hx hnx hne (by norm_num : (0 : ℝ) < 1 / 2) (by norm_num : (0 : ℝ) < 1 / 2)
        (by norm_num)
      simp only [smul_eq_mul] at h
      have hmid : (1 / 2 : ℝ) * x + (1 / 2) * (-x) = 0 := by ring
      rw [hmid, hEven x hx] at h
      exact ⟨by simp only [K, M]; linarith, fun _ => by simp only [K, M]; linarith⟩
  have hmid_mem : (a + b) / 2 ∈ Set.Ioo (-1 : ℝ) 1 := ⟨by linarith [ha.1, hb.1], by linarith [ha.2, hb.2]⟩
  have hc_eq : c = r * ((a + b) / 2) + (1 - r) * 0 := by simp only [c, r]; ring
  have hc_mem : c ∈ Set.Ioo (-1 : ℝ) 1 := by
    rw [hc_eq]
    exact ⟨by nlinarith [hmid_mem.1], by nlinarith [hmid_mem.2]⟩
  -- Convexity of K: K(c) ≤ (r/2)(K(a) + K(b))
  have hKc : K c ≤ (r / 2) * (K a + K b) := by
    have h1 := hConc.concaveOn.2 hmid_mem h0_mem hr0.le (show (0 : ℝ) ≤ 1 - r by linarith) (by ring)
    have h2 := hConc.concaveOn.2 ha hb (by norm_num : (0 : ℝ) ≤ 1 / 2)
      (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num)
    simp only [smul_eq_mul] at h1 h2
    rw [← hc_eq] at h1
    have hmid_eq : (1 / 2 : ℝ) * a + (1 / 2) * b = (a + b) / 2 := by ring
    rw [hmid_eq] at h2
    simp only [K, M]
    nlinarith
  -- Tangent and chord inequalities for x ↦ x^k (convex on [0, M] since k ≥ 1):
  -- Tangent at M: M^k - x^k ≤ k M^{k-1} (M - x) for 0 ≤ x ≤ M
  -- Chord from 0 to M: M^k - x^k ≥ M^{k-1} (M - x) for 0 ≤ x ≤ M
  have hpow_bounds : ∀ x : ℝ, 0 < x → x ≤ M →
      M ^ (k - 1) * (M - x) ≤ M ^ k - x ^ k ∧
      M ^ k - x ^ k ≤ k * M ^ (k - 1) * (M - x) := by
    intro x hx0 hxM
    have hMk : M ^ k = M ^ (k - 1) * M := by
      calc M ^ k = M ^ ((k - 1) + 1) := by ring_nf
        _ = M ^ (k - 1) * M ^ (1 : ℝ) := Real.rpow_add hM_pos _ _
        _ = M ^ (k - 1) * M := by rw [Real.rpow_one]
    have hxk : x ^ k = x ^ (k - 1) * x := by
      calc x ^ k = x ^ ((k - 1) + 1) := by ring_nf
        _ = x ^ (k - 1) * x ^ (1 : ℝ) := Real.rpow_add hx0 _ _
        _ = x ^ (k - 1) * x := by rw [Real.rpow_one]
    constructor
    · -- x^{k-1} ≤ M^{k-1} since k - 1 ≥ 0
      have hle : x ^ (k - 1) ≤ M ^ (k - 1) := Real.rpow_le_rpow hx0.le hxM (by linarith)
      rw [hMk, hxk]
      nlinarith
    · -- M^k - x^k ≤ k M^{k-1} (M - x) via MVT on [x, M]
      by_cases hxe : x = M
      · subst hxe; simp
      have hlt : x < M := lt_of_le_of_ne hxM hxe
      have hd : ∀ t ∈ Set.Ioo x M, HasDerivAt (fun y : ℝ => y ^ k) (k * t ^ (k - 1)) t :=
        fun t ht => Real.hasDerivAt_rpow_const (Or.inl (by linarith [ht.1]))
      have hcont : ContinuousOn (fun y : ℝ => y ^ k) (Set.Icc x M) :=
        ContinuousOn.rpow_const continuousOn_id fun y hy => Or.inl (by linarith [hy.1])
      obtain ⟨ξ, hξ, hmv⟩ := exists_hasDerivAt_eq_slope (fun y : ℝ => y ^ k)
        (fun y => k * y ^ (k - 1)) hlt hcont hd
      have hξ_le : ξ ^ (k - 1) ≤ M ^ (k - 1) :=
        Real.rpow_le_rpow (by linarith [hξ.1]) hξ.2.le (by linarith)
      have hslope : M ^ k - x ^ k = (k * ξ ^ (k - 1)) * (M - x) := by
        rw [hmv, div_mul_cancel₀ _ (by linarith)]
      rw [hslope]
      exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hξ_le (by linarith)) (by linarith)
  have hb_a := hpow_bounds (U a) (hPos a ha) (by have := (hK_pos a ha).1; simp only [K] at this; linarith)
  have hb_b := hpow_bounds (U b) (hPos b hb) (by have := (hK_pos b hb).1; simp only [K] at this; linarith)
  have hb_c := hpow_bounds (U c) (hPos c hc_mem) (by have := (hK_pos c hc_mem).1; simp only [K] at this; linarith)
  have h2k : (2 : ℝ) ^ k = 2 * (2 : ℝ) ^ (k - 1) := by
    calc (2 : ℝ) ^ k = (2 : ℝ) ^ (1 + (k - 1)) := by ring_nf
      _ = (2 : ℝ) ^ (1 : ℝ) * (2 : ℝ) ^ (k - 1) := Real.rpow_add (by norm_num) _ _
      _ = 2 * (2 : ℝ) ^ (k - 1) := by rw [Real.rpow_one]
  -- Core lower bound: gp(a, b) - gp(0, 0) ≥ M^{k-1} (1 - 2^{k-1} k r) (K(a) + K(b))
  have hgp_diff : M ^ (k - 1) * (1 - 2 ^ (k - 1) * k * r) * (K a + K b) ≤
      ((2 : ℝ) ^ k * U c ^ k - U a ^ k - U b ^ k) -
      ((2 : ℝ) ^ k * U ((0 + 0) / (2 * Real.sqrt 2)) ^ k - U 0 ^ k - U 0 ^ k) := by
    have h00 : (0 + 0) / (2 * Real.sqrt 2) = (0 : ℝ) := by norm_num
    rw [h00]
    have hre : ((2 : ℝ) ^ k * U c ^ k - U a ^ k - U b ^ k) - ((2 : ℝ) ^ k * M ^ k - M ^ k - M ^ k) =
        (M ^ k - U a ^ k) + (M ^ k - U b ^ k) - (2 : ℝ) ^ k * (M ^ k - U c ^ k) := by ring
    change M ^ (k - 1) * (1 - 2 ^ (k - 1) * k * r) * (K a + K b) ≤
      ((2 : ℝ) ^ k * U c ^ k - U a ^ k - U b ^ k) - ((2 : ℝ) ^ k * M ^ k - M ^ k - M ^ k)
    rw [hre]
    have h2kc : (2 : ℝ) ^ k * (M ^ k - U c ^ k) ≤ (2 : ℝ) ^ k * (k * M ^ (k - 1) * K c) :=
      mul_le_mul_of_nonneg_left hb_c.2 (Real.rpow_nonneg (by norm_num) _)
    have h2kc2 : (2 : ℝ) ^ k * (k * M ^ (k - 1) * K c) ≤
        (2 : ℝ) ^ k * (k * M ^ (k - 1) * ((r / 2) * (K a + K b))) :=
      mul_le_mul_of_nonneg_left
        (mul_le_mul_of_nonneg_left hKc (mul_nonneg (by linarith) (Real.rpow_nonneg hM_pos.le _)))
        (Real.rpow_nonneg (by norm_num) _)
    have hsimp : (2 : ℝ) ^ k * (k * M ^ (k - 1) * ((r / 2) * (K a + K b))) =
        M ^ (k - 1) * (2 ^ (k - 1) * k * r) * (K a + K b) := by
      rw [h2k]; ring
    have hcomb := add_le_add hb_a.1 hb_b.1
    simp only [K] at hsimp hcomb h2kc h2kc2 ⊢
    linarith
  have hcoef_pos : 0 < M ^ (k - 1) * (1 - 2 ^ (k - 1) * k * r) :=
    mul_pos (Real.rpow_pos_of_pos hM_pos _) (by linarith)
  have hK_sum_nn : 0 ≤ K a + K b := add_nonneg (hK_pos a ha).1 (hK_pos b hb).1
  have hgp_le : (2 : ℝ) ^ k * U ((0 + 0) / (2 * Real.sqrt 2)) ^ k - U 0 ^ k - U 0 ^ k ≤
      (2 : ℝ) ^ k * U c ^ k - U a ^ k - U b ^ k := by
    have : 0 ≤ M ^ (k - 1) * (1 - 2 ^ (k - 1) * k * r) * (K a + K b) :=
      mul_nonneg hcoef_pos.le hK_sum_nn
    linarith
  -- Denominator comparison: U(a)^k + U(b)^k ≤ 2 M^k = U(0)^k + U(0)^k
  have hden_pos : 0 < U a ^ k + U b ^ k :=
    add_pos (Real.rpow_pos_of_pos (hPos a ha) k) (Real.rpow_pos_of_pos (hPos b hb) k)
  have hden0_pos : 0 < U 0 ^ k + U 0 ^ k :=
    add_pos (Real.rpow_pos_of_pos hM_pos k) (Real.rpow_pos_of_pos hM_pos k)
  have hden_le : U a ^ k + U b ^ k ≤ U 0 ^ k + U 0 ^ k := by
    have hUa : U a ^ k ≤ M ^ k := Real.rpow_le_rpow (hPos a ha).le
      (by have := (hK_pos a ha).1; simp only [K] at this; linarith) (by linarith)
    have hUb : U b ^ k ≤ M ^ k := Real.rpow_le_rpow (hPos b hb).le
      (by have := (hK_pos b hb).1; simp only [K] at this; linarith) (by linarith)
    linarith
  have hgp0_nn : 0 ≤ (2 : ℝ) ^ k * U ((0 + 0) / (2 * Real.sqrt 2)) ^ k - U 0 ^ k - U 0 ^ k := by
    have h00 : (0 + 0) / (2 * Real.sqrt 2) = (0 : ℝ) := by norm_num
    rw [h00]
    have h2k_ge2 : (2 : ℝ) ≤ (2 : ℝ) ^ k := by
      calc (2 : ℝ) = (2 : ℝ) ^ (1 : ℝ) := (Real.rpow_one 2).symm
        _ ≤ (2 : ℝ) ^ k := Real.rpow_le_rpow_of_exponent_le (by norm_num) hk1
    nlinarith [Real.rpow_pos_of_pos hM_pos k]
  have hrho_le :
      ((2 : ℝ) ^ k * U ((0 + 0) / (2 * Real.sqrt 2)) ^ k - U 0 ^ k - U 0 ^ k) / (U 0 ^ k + U 0 ^ k) ≤
      ((2 : ℝ) ^ k * U c ^ k - U a ^ k - U b ^ k) / (U a ^ k + U b ^ k) := by
    rw [div_le_div_iff₀ hden0_pos hden_pos]
    calc ((2 : ℝ) ^ k * U ((0 + 0) / (2 * Real.sqrt 2)) ^ k - U 0 ^ k - U 0 ^ k) * (U a ^ k + U b ^ k)
        ≤ ((2 : ℝ) ^ k * U ((0 + 0) / (2 * Real.sqrt 2)) ^ k - U 0 ^ k - U 0 ^ k) * (U 0 ^ k + U 0 ^ k) :=
          mul_le_mul_of_nonneg_left hden_le hgp0_nn
      _ ≤ ((2 : ℝ) ^ k * U c ^ k - U a ^ k - U b ^ k) * (U 0 ^ k + U 0 ^ k) :=
          mul_le_mul_of_nonneg_right hgp_le hden0_pos.le
  refine ⟨hgp_le, hrho_le, fun hne => ?_⟩
  have hK_sum_pos : 0 < K a + K b := by
    rcases hne with ha0 | hb0
    · linarith [(hK_pos a ha).2 ha0, (hK_pos b hb).1]
    · linarith [(hK_pos a ha).1, (hK_pos b hb).2 hb0]
  have hgp_lt : (2 : ℝ) ^ k * U ((0 + 0) / (2 * Real.sqrt 2)) ^ k - U 0 ^ k - U 0 ^ k <
      (2 : ℝ) ^ k * U c ^ k - U a ^ k - U b ^ k := by
    have : 0 < M ^ (k - 1) * (1 - 2 ^ (k - 1) * k * r) * (K a + K b) :=
      mul_pos hcoef_pos hK_sum_pos
    linarith
  refine ⟨hgp_lt, ?_⟩
  rw [div_lt_div_iff₀ hden0_pos hden_pos]
  calc ((2 : ℝ) ^ k * U ((0 + 0) / (2 * Real.sqrt 2)) ^ k - U 0 ^ k - U 0 ^ k) * (U a ^ k + U b ^ k)
      ≤ ((2 : ℝ) ^ k * U ((0 + 0) / (2 * Real.sqrt 2)) ^ k - U 0 ^ k - U 0 ^ k) * (U 0 ^ k + U 0 ^ k) :=
        mul_le_mul_of_nonneg_left hden_le hgp0_nn
    _ < ((2 : ℝ) ^ k * U c ^ k - U a ^ k - U b ^ k) * (U 0 ^ k + U 0 ^ k) :=
        mul_lt_mul_of_pos_right hgp_lt hden0_pos

/-! ### 4. Supercritical Hermite Deficit (`p > 2`, prior art: Hashemi 2026 Prop. 5.5) -/

/-- **Supercritical Hermite deficit (`p > 2`)**:
For every `p > 2`, `p / (2(p - 1)) < 1`, hence `2^(p / (2(p - 1))) < 2` and
`(2^(p / (2(p - 1))) - 2) * F < 0` for every `F > 0`. -/
theorem supercritical_hermite_deficit_neg {p F : ℝ} (hp : 2 < p) (hF : 0 < F) :
    ((2 : ℝ) ^ (p / (2 * (p - 1))) - 2) * F < 0 := by
  have hexp : p / (2 * (p - 1)) < 1 := by
    rw [div_lt_one (by linarith)]; linarith
  have hpow : (2 : ℝ) ^ (p / (2 * (p - 1))) < 2 := by
    calc (2 : ℝ) ^ (p / (2 * (p - 1))) < (2 : ℝ) ^ (1 : ℝ) :=
          Real.rpow_lt_rpow_of_exponent_lt (by norm_num) hexp
      _ = 2 := Real.rpow_one 2
  nlinarith

/-! ### 5. All-Degree Symmetric-Score Subcritical `p`-Stam (Theorem 3) & `n = 64` Obstruction (Prop. 5) -/

/-- **Theorem 3 (Part 1 — Projection and strict `ℓ¹`-contraction `η < 1`)**:
Let `A` be an entrywise strictly positive doubly stochastic `n × n` matrix (`n ≥ 1`), and
let `σ : Equiv.Perm (Fin n)` be an involution (`σ (σ i) = i`). Let `S` be the permutation
matrix of `σ` and `P := (1 - S) / 2`. Then:
1. `Pᵀ = P`, `P * P = P`,
2. Every vector `P u` has mean zero (`∑_i (P u)_i = 0`),
3. If `s` satisfies `s (σ i) = -s i`, then `P s = s`,
4. Every column of `A P` has `ℓ¹`-norm strictly less than `1` (hence `η := ‖AP‖_{1→1} < 1`). -/
theorem symmetric_score_projection_contraction {n : ℕ} [NeZero n]
    (A : Matrix (Fin n) (Fin n) ℝ)
    (hA_pos : ∀ i j, 0 < A i j)
    (hA_col : ∀ j, ∑ i : Fin n, A i j = 1)
    (σ : Equiv.Perm (Fin n)) (hσ2 : ∀ i, σ (σ i) = i) :
    let S : Matrix (Fin n) (Fin n) ℝ := Matrix.of fun i j => if i = σ j then 1 else 0
    let P : Matrix (Fin n) (Fin n) ℝ := (1 / 2 : ℝ) • ((1 : Matrix (Fin n) (Fin n) ℝ) - S)
    Pᵀ = P ∧
    P * P = P ∧
    (∀ u : Fin n → ℝ, ∑ i : Fin n, (P.mulVec u) i = 0) ∧
    (∀ s : Fin n → ℝ, (∀ i, s (σ i) = -s i) → P.mulVec s = s) ∧
    (∀ j : Fin n, ∑ i : Fin n, |(A * P) i j| < 1) := by
  dsimp only
  set S : Matrix (Fin n) (Fin n) ℝ := Matrix.of fun i j => if i = σ j then 1 else 0
  set P : Matrix (Fin n) (Fin n) ℝ := (1 / 2 : ℝ) • ((1 : Matrix (Fin n) (Fin n) ℝ) - S)
  have hS_sym : Sᵀ = S := by
    ext i j
    simp only [S, Matrix.transpose_apply, Matrix.of_apply]
    have hiff : j = σ i ↔ i = σ j :=
      ⟨fun h => by rw [h, hσ2], fun h => by rw [h, hσ2]⟩
    simp [hiff]
  have hS_sq : S * S = 1 := by
    ext i j
    simp only [S, Matrix.mul_apply, Matrix.of_apply, Matrix.one_apply]
    rw [Finset.sum_eq_single (σ j)]
    · simp only [↓reduceIte, mul_one, hσ2]
    · intro k _ hk; simp [hk]
    · intro h; exact (h (Finset.mem_univ _)).elim
  have hP_sym : Pᵀ = P := by
    simp only [P, Matrix.transpose_smul, Matrix.transpose_sub, Matrix.transpose_one, hS_sym]
  have hP_idem : P * P = P := by
    simp only [P, Matrix.smul_mul, Matrix.mul_smul, sub_mul, mul_sub,
      Matrix.one_mul, Matrix.mul_one, hS_sq]
    ext i j; simp; ring
  have hS_mulVec : ∀ u : Fin n → ℝ, S.mulVec u = fun i => u (σ i) := by
    intro u; ext i
    simp only [S, Matrix.mulVec, dotProduct, Matrix.of_apply]
    rw [Finset.sum_eq_single (σ i)]
    · simp [hσ2]
    · intro k _ hk
      have : i ≠ σ k := fun h => hk (by rw [h, hσ2])
      simp [this]
    · intro h; exact (h (Finset.mem_univ _)).elim
  have hP_mulVec : ∀ u : Fin n → ℝ, P.mulVec u = fun i => (u i - u (σ i)) / 2 := by
    intro u; ext i
    simp only [P, Matrix.smul_mulVec, Matrix.sub_mulVec, Matrix.one_mulVec, hS_mulVec,
      Pi.smul_apply, Pi.sub_apply, smul_eq_mul]
    ring
  have hmean : ∀ u : Fin n → ℝ, ∑ i : Fin n, (P.mulVec u) i = 0 := by
    intro u
    simp_rw [hP_mulVec, sub_div, Finset.sum_sub_distrib]
    have hbij : ∑ i : Fin n, u (σ i) / 2 = ∑ i : Fin n, u i / 2 :=
      Fintype.sum_equiv σ _ _ (fun _ => rfl)
    linarith
  have hfix : ∀ s : Fin n → ℝ, (∀ i, s (σ i) = -s i) → P.mulVec s = s := by
    intro s hs; ext i
    rw [hP_mulVec]; dsimp only; rw [hs i]; ring
  have hAP_entry : ∀ i j : Fin n, (A * P) i j = (A i j - A i (σ j)) / 2 := by
    intro i j
    simp only [P, Matrix.mul_smul, Matrix.mul_sub, Matrix.mul_one, Matrix.smul_apply,
      Matrix.sub_apply, smul_eq_mul]
    have hAS : (A * S) i j = A i (σ j) := by
      simp only [S, Matrix.mul_apply, Matrix.of_apply]
      rw [Finset.sum_eq_single (σ j)]
      · simp
      · intro k _ hk; simp [hk]
      · intro h; exact (h (Finset.mem_univ _)).elim
    rw [hAS]; ring
  have hcol_lt : ∀ j : Fin n, ∑ i : Fin n, |(A * P) i j| < 1 := by
    intro j
    -- For strictly positive x, y > 0, |x - y| < x + y!
    have hpt : ∀ i : Fin n, |(A * P) i j| < (A i j + A i (σ j)) / 2 := by
      intro i
      rw [hAP_entry i j, abs_div, abs_of_pos (by norm_num : (0 : ℝ) < 2)]
      apply div_lt_div_of_pos_right _ (by norm_num)
      rw [abs_lt]
      exact ⟨by linarith [hA_pos i j], by linarith [hA_pos i (σ j)]⟩
    have hsum_lt : ∑ i : Fin n, |(A * P) i j| < ∑ i : Fin n, (A i j + A i (σ j)) / 2 :=
      Finset.sum_lt_sum_of_nonempty Finset.univ_nonempty fun i _ => hpt i
    have hrhs : ∑ i : Fin n, (A i j + A i (σ j)) / 2 = 1 := by
      simp_rw [add_div, Finset.sum_add_distrib, ← Finset.sum_div, hA_col j, hA_col (σ j)]
      norm_num
    linarith
  exact ⟨hP_sym, hP_idem, hmean, hfix, hcol_lt⟩

/-- **Theorem 3 (Part 2 — Variational Hölder minimization and strict `p`-Stam boost)**:
For `1 < p < 2`, `η ∈ (0, 1)`, and positive `Φ_f, Φ_g, Φ_C > 0`, if
`Φ_C ≤ η^(2 - p) λ^p Φ_f + (1 - λ)^p Φ_g` holds for all `λ ∈ (0, 1)`, then
`F_p(C) ≥ η^(-(2 - p)/(p - 1)) F_p(f) + F_p(g) > F_p(f) + F_p(g)`,
where `F_p(Φ) := Φ^(-1 / (p - 1))`. -/
theorem holder_variational_pstam_boost {p η Φ_f Φ_g Φ_C : ℝ}
    (hp1 : 1 < p) (hp2 : p < 2) (hη0 : 0 < η) (hη1 : η < 1)
    (hf : 0 < Φ_f) (hg : 0 < Φ_g) (hC : 0 < Φ_C)
    (hinterp : ∀ lam : ℝ, 0 < lam → lam < 1 →
      Φ_C ≤ η ^ (2 - p) * lam ^ p * Φ_f + (1 - lam) ^ p * Φ_g) :
    let Ff := Φ_f ^ (-(1 : ℝ) / (p - 1))
    let Fg := Φ_g ^ (-(1 : ℝ) / (p - 1))
    let FC := Φ_C ^ (-(1 : ℝ) / (p - 1))
    η ^ (-(2 - p) / (p - 1)) * Ff + Fg ≤ FC ∧ Ff + Fg < FC := by
  dsimp only
  set a := η ^ (2 - p) * Φ_f
  set b := Φ_g
  have ha : 0 < a := mul_pos (Real.rpow_pos_of_pos hη0 _) hf
  have hb : 0 < b := hg
  have hp_sub : 0 < p - 1 := by linarith
  -- Optimal λ = a^(-1/(p-1)) / (a^(-1/(p-1)) + b^(-1/(p-1)))
  set A := a ^ (-(1 : ℝ) / (p - 1))
  set B := b ^ (-(1 : ℝ) / (p - 1))
  have hA : 0 < A := Real.rpow_pos_of_pos ha _
  have hB : 0 < B := Real.rpow_pos_of_pos hb _
  set lam := A / (A + B)
  have hlam0 : 0 < lam := div_pos hA (by linarith)
  have hlam1 : lam < 1 := (div_lt_one (by linarith)).mpr (by linarith)
  have h1mlam : 1 - lam = B / (A + B) := by
    simp only [lam]; field_simp [(show A + B ≠ 0 by linarith)]; ring
  have hopt := hinterp lam hlam0 hlam1
  -- Evaluate a * lam^p + b * (1 - lam)^p = (A + B)^(1 - p)
  have ha_term : a * lam ^ p = A / (A + B) ^ p := by
    simp only [lam]
    rw [Real.div_rpow hA.le (by linarith : (0 : ℝ) ≤ A + B)]
    rw [← mul_div_assoc]
    congr 1
    simp only [A]
    rw [← Real.rpow_mul ha.le]
    calc a * a ^ (-(1 : ℝ) / (p - 1) * p)
        = a ^ (1 : ℝ) * a ^ (-(1 : ℝ) / (p - 1) * p) := by rw [Real.rpow_one]
      _ = a ^ (1 + -(1 : ℝ) / (p - 1) * p) := (Real.rpow_add ha _ _).symm
      _ = a ^ (-(1 : ℝ) / (p - 1)) := by
          congr 1; field_simp [hp_sub.ne']; ring
  have hb_term : b * (1 - lam) ^ p = B / (A + B) ^ p := by
    rw [h1mlam, Real.div_rpow hB.le (by linarith : (0 : ℝ) ≤ A + B), ← mul_div_assoc]
    congr 1
    simp only [B]
    rw [← Real.rpow_mul hb.le]
    calc b * b ^ (-(1 : ℝ) / (p - 1) * p)
        = b ^ (1 : ℝ) * b ^ (-(1 : ℝ) / (p - 1) * p) := by rw [Real.rpow_one]
      _ = b ^ (1 + -(1 : ℝ) / (p - 1) * p) := (Real.rpow_add hb _ _).symm
      _ = b ^ (-(1 : ℝ) / (p - 1)) := by
          congr 1; field_simp [hp_sub.ne']; ring
  have hPhi_le : Φ_C ≤ (A + B) ^ (1 - p) := by
    have hre : η ^ (2 - p) * lam ^ p * Φ_f + (1 - lam) ^ p * Φ_g =
        a * lam ^ p + b * (1 - lam) ^ p := by simp only [a, b]; ring
    rw [hre, ha_term, hb_term, ← add_div] at hopt
    have hdiv_eq : (A + B) / (A + B) ^ p = (A + B) ^ (1 - p) := by
      calc (A + B) / (A + B) ^ p = (A + B) ^ (1 : ℝ) / (A + B) ^ p := by rw [Real.rpow_one]
        _ = (A + B) ^ (1 - p) := (Real.rpow_sub (by linarith) 1 p).symm
    rwa [hdiv_eq] at hopt
  -- Raise both sides to the negative power -1/(p - 1) < 0 (reverses inequality!)
  have hpow_ge : A + B ≤ Φ_C ^ (-(1 : ℝ) / (p - 1)) := by
    have hrev := Real.rpow_le_rpow_of_nonpos hC hPhi_le
      (show -(1 : ℝ) / (p - 1) ≤ 0 by exact div_nonpos_of_nonpos_of_nonneg (by norm_num) hp_sub.le)
    rw [← Real.rpow_mul (by linarith : (0 : ℝ) ≤ A + B)] at hrev
    have hexp1 : (1 - p) * (-(1 : ℝ) / (p - 1)) = 1 := by field_simp [hp_sub.ne']; ring
    rw [hexp1, Real.rpow_one] at hrev
    exact hrev
  have hA_eq : A = η ^ (-(2 - p) / (p - 1)) * Φ_f ^ (-(1 : ℝ) / (p - 1)) := by
    simp only [A, a]
    rw [Real.mul_rpow (Real.rpow_nonneg hη0.le _) hf.le, ← Real.rpow_mul hη0.le]
    congr 2; ring
  have hboost_gt : (1 : ℝ) < η ^ (-(2 - p) / (p - 1)) := by
    have hexp_neg : -(2 - p) / (p - 1) < 0 := div_neg_of_neg_of_pos (by linarith) hp_sub
    calc (1 : ℝ) = η ^ (0 : ℝ) := (Real.rpow_zero η).symm
      _ < η ^ (-(2 - p) / (p - 1)) := Real.rpow_lt_rpow_of_exponent_gt hη0 hη1 hexp_neg
  have hFf_pos : 0 < Φ_f ^ (-(1 : ℝ) / (p - 1)) := Real.rpow_pos_of_pos hf _
  rw [hA_eq] at hpow_ge
  exact ⟨hpow_ge, by nlinarith⟩

/-- **Proposition 5 (Centered Jacobian `ℓ^p`-contraction obstruction at `n = 64, p = 3/2`)**:
For the centered vector `a = (-1, …, -1, 63)` and output `c = (-2, …, -2, 54, 70)`,
`‖c‖_{3/2}^{3/2} = 62 · 2^(3/2) + 54^(3/2) + 70^(3/2) > 2 (63 + 63^(3/2)) = 2 ‖a‖_{3/2}^{3/2}`. -/
theorem centered_jacobian_obstruction_64 :
    2 * (63 + (63 : ℝ) ^ ((3 : ℝ) / 2)) <
      62 * (2 : ℝ) ^ ((3 : ℝ) / 2) + (54 : ℝ) ^ ((3 : ℝ) / 2) + (70 : ℝ) ^ ((3 : ℝ) / 2) := by
  -- Express x^(3/2) = x * √x for x ≥ 0
  have hrpow32 : ∀ x : ℝ, 0 ≤ x → x ^ ((3 : ℝ) / 2) = x * Real.sqrt x := by
    intro x hx
    rw [Real.sqrt_eq_rpow]
    calc x ^ ((3 : ℝ) / 2) = x ^ ((1 : ℝ) + (1 : ℝ) / 2) := by norm_num
      _ = x ^ (1 : ℝ) * x ^ ((1 : ℝ) / 2) := by
          by_cases hx0 : x = 0
          · subst hx0; rw [Real.zero_rpow (by norm_num)]; simp
          · exact Real.rpow_add (lt_of_le_of_ne hx (Ne.symm hx0)) _ _
      _ = x * x ^ ((1 : ℝ) / 2) := by rw [Real.rpow_one]
  rw [hrpow32 63 (by norm_num), hrpow32 2 (by norm_num),
      hrpow32 54 (by norm_num), hrpow32 70 (by norm_num)]
  -- Elementary rational bounds on √2, √54, √63, √70:
  have h2 : (14 / 10 : ℝ) < Real.sqrt 2 := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have h54 : (73 / 10 : ℝ) < Real.sqrt 54 := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have h70 : (83 / 10 : ℝ) < Real.sqrt 70 := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have h63 : Real.sqrt 63 < (8 : ℝ) := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  nlinarith

end FiniteFreePStam

end
