import Mathlib

/-!
# Disproof of Fisher–Volume Scalar Curvature Quantization for Bitnets
(Carlos C. Rodríguez, AIP Conf. Proc. 735 (2004), `conj:main`; and arXiv:2603.10054v1, March 2026)

We formalize in Lean 4 + Mathlib the exact algebraic curvature formulas, exact Beta and
Dirichlet(1/2, 1/2, 1/2, 1/2) integral evaluations, and counterexamples settling Conjecture
`conj:main` and correcting the erroneous claims of arXiv:2603.10054v1:

1. **Multi-warped product & edge-base curvature algebra (Lemmas 1–2)**:
   - `multi_warped_curvature_identity`: Algebraic equivalence between the sectional-curvature
     sum `R₀ - 2 ∑ (Δf_j / f_j) - ∑_{i ≠ j} Γ(log f_i, log f_j)` and
     `R₀ - 2 Δ(log F) - Γ(log F, log F) - ∑ Γ(log f_j, log f_j)`.
   - `edge_leaf_curvature_eq`: Exact derivation of the `(k, l)`-leaf scalar curvature on the
     edge base `A → B` (`R₀ = 3/2`),
     `R = 3/2 + C_k(u) + C_l(v) - (k l / 2) * w (1 - 2a)(1 - 2b) / (u v)` with
     `C_k(u) = k² + 5k/2 - k(k+1)/(4u)`.
   - `fiveNodeTree_curvature_formula`: Specialization `(k, l) = (1, 2)` giving
     `R = 14 - 1/(2u) - 3/(2v) - w(1 - 2a)(1 - 2b)/(uv)`.

2. **Exact Beta integrals `J_s = ∫₀¹ (a(1-a))^s da` and Five-Node Tree Counterexample (Theorem 1)**:
   - `betaIntegral_half_int`: Using the `C¹` change of variables `a = (1 - cos θ)/2` on `(0, π)`,
     `∫ a in Set.Ioo 0 1, (a * (1 - a)) ^ ((K - 1 : ℝ) / 2) = (1 / 2^K) * ∫ θ in 0..π, sin θ ^ K`
     for every `K : ℕ`.
   - `sinPowIntegral_recurrence`: Exact Wallis recurrence on `0..π`,
     `∫ θ in 0..π, sin θ ^ (n + 2) = ((n + 1)/(n + 2)) * ∫ θ in 0..π, sin θ ^ n`.
   - `betaIntegral_neg_half`, `betaIntegral_half`, `betaIntegral_three_halves`:
     `J(-1/2) = π`, `J(1/2) = π/8`, `J(3/2) = 3π/128`.
   - `betaHalf_first_moment`, `betaHalf_product_moment`:
     Exact `Beta(1/2, 1/2)` moments `E[t] = 1/2` and `E[t(1 - t)] = 1/8` (hence `E[t²] = 3/8`).
   - `condExp_v_given_a`, `condExp_cross_given_a`: Exact conditional expectations over
     independent `t₀, t₁ ~ Beta(1/2, 1/2)`, yielding `E₀[v | a] = (1 + 2u)/8` and
     `E₀[w(1 - 2b) | a] = u(1 - 2a)/4`.
   - `fiveNodeTree_avgCurvature` and `fourteen_elevenths_not_half_int`:
     Exact Fisher-volume average `⟨R⟩ = 14 / 11 ∉ (1/2)ℤ` for the 5-node binary tree
     `A → B, A → C, B → D, B → E` (`d = 9`), disproving Rodríguez (2004) `conj:main` and
     Theorem 4.3 of arXiv:2603.10054v1.

3. **Exact Dirichlet(1/2, 1/2, 1/2, 1/2) moments, Six-Node Tree (Prop. 2), & 7-Node 1-Loop (Prop. 4)**:
   - `dirichletHalfMoment`: Exact rational moment functional `∏ (1/2)_{n_α} / (2)_{∑ n_α}`
     and polynomial expectation identities `E₀[uv] = 7/240`, `E₀[u] = E₀[v] = 1/6`,
     `E₀[w(1-2a)(1-2b)] = 1/120`, `E₀[u²] = 1/30`, `E₀[u²v] = 1/168`,
     `E₀[uw(1-2a)(1-2b)] = 1/840`.
   - `sixNodeTree_avgCurvature`, `twentyfive_fourteenths_not_half_int`:
     Exact `⟨R⟩ = 25 / 14 ∉ (1/2)ℤ` for the 6-node tree `(k, l) = (2, 2)`.
   - `sevenNodeOneLoop_avgCurvature`, `twentyfour_fifths_not_half_int`:
     Exact `⟨R⟩ = 24 / 5 ∉ (1/2)ℤ` for the 7-node chordless 4-cycle graph (`d = 16`).

4. **Shared-Root Colliders and Corrections to arXiv:2603.10054v1 (Proposition 3)**:
   - `sharedRoot_curvature_eq`: Exact root-level scalar curvature
     `S_K(r) = K(2K + 1)/2 - K(K - 1)/(4U)` (from `Δ(log U) = -1/(2U)` on a 1D root base).
   - `betaIntegral_ratio`: For all `K ≥ 2`, `J((K-3)/2) / J((K-1)/2) = 4K / (K - 1)`.
   - `sharedRootCollider_avgCurvature`: Exact `⟨R⟩ = n * K / 2` for `n` independent roots
     and `m` common sinks with `K = m * 2^(n - 1) ≥ 2` (and `⟨R⟩ = 3/2` for `K = 1`, `n = m = 1`),
     always lying in `(1/2)ℤ`.
   - `doubleCollider_D4_avgCurvature`: `D_4` (`n = 2, m = 2, K = 4`) has `⟨R⟩ = 4 ≠ 36/5`
     (refuting Proposition 5.2 of arXiv:2603.10054v1).
   - `fork_E3_avgCurvature`: `Ẽ_3` (`n = 1, m = 2, K = 2`) has `⟨R⟩ = 1 ≠ 5/2`.
   - `collapsingStar_C6_avgCurvature`: `C̃_6` (`n = 5, m = 1, K = 16`) has `⟨R⟩ = 40 > 0`
     (refuting Theorem 3.4's claimed sign inversion to `-272`).

Machine-checked vs prose-only:
- Machine-checked: All multi-warped algebraic identities, all exact `J_s` Beta integrals on
  `(0, 1)` via Lebesgue change of variables and Wallis recursion, conditional expectation
  algebra, exact rational Dirichlet(1/2, 1/2, 1/2, 1/2) moment computations, all non-half-integer
  disproofs (`14/11`, `25/14`, `24/5`), and all shared-root collider averages (`D_4 ↦ 4`,
  `Ẽ_3 ↦ 1`, `C̃_6 ↦ 40`).
- Prose-only (in `solutions/bitnet_ricci_quant/proof.pdf`): Koszul connection derivation of the
  Riemannian scalar curvature on warped products, spherical `S³(2)` Laplacian/gradient derivation,
  and the directed-diamond convergent binomial series (Section 4).
-/

open MeasureTheory BigOperators Finset

namespace BitnetRicciQuant

/-! ### 1. Multi-Warped Product and Edge-Base Curvature Algebra (Lemmas 1–2) -/

/-- **Lemma 1 (Algebraic core of multi-warped product scalar curvature)**:
Substituting `Δf_j / f_j = Δ(log f_j) + Γ(log f_j, log f_j)` and expanding
`Γ(∑ log f_j, ∑ log f_j) = ∑ Γ(log f_j, log f_j) + ∑_{i ≠ j} Γ(log f_i, log f_j)`
transforms the sectional-curvature sum into Eq. (1). -/
theorem multi_warped_curvature_identity {ι : Type*} [DecidableEq ι] (s : Finset ι)
    (R₀ : ℝ) (lapLog : ι → ℝ) (gam : ι → ι → ℝ) :
    R₀ - 2 * ∑ j ∈ s, (lapLog j + gam j j)
      - ∑ i ∈ s, ∑ j ∈ s.erase i, gam i j =
    R₀ - 2 * (∑ j ∈ s, lapLog j)
      - (∑ i ∈ s, ∑ j ∈ s, gam i j)
      - ∑ j ∈ s, gam j j := by
  have hsplit : ∑ i ∈ s, ∑ j ∈ s, gam i j =
      (∑ i ∈ s, gam i i) + ∑ i ∈ s, ∑ j ∈ s.erase i, gam i j := by
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun i hi => ?_
    exact (Finset.add_sum_erase s (gam i) hi).symm
  rw [Finset.sum_add_distrib, hsplit]
  ring

/-- Single-node leaf curvature contribution `C_k(u) = k² + 5k/2 - k(k+1)/(4u)`. -/
noncomputable def leafContribution (k : ℝ) (u : ℝ) : ℝ :=
  k ^ 2 + 5 * k / 2 - k * (k + 1) / (4 * u)

/-- Edge-base `(k, l)`-leaf scalar curvature formula Eq. (2) as a function of
`u = a(1-a)`, `v = b(1-b)`, `w = p₁₁ - ab`. -/
noncomputable def edgeCurvature (k l a b p₁₁ : ℝ) : ℝ :=
  let u := a * (1 - a)
  let v := b * (1 - b)
  let w := p₁₁ - a * b
  3 / 2 + leafContribution k u + leafContribution l v
    - (k * l / 2) * (w * (1 - 2 * a) * (1 - 2 * b)) / (u * v)

/-- **Lemma 2 (Algebraic assembly of `(k, l)`-leaf scalar curvature on `A → B`)**:
Substituting `R₀ = 3/2`, `Δ(log u) = Δ(log v) = -2`, `Γ(log u, log u) = 1/u - 4`,
`Γ(log v, log v) = 1/v - 4`, `Γ(log u, log v) = w(1 - 2a)(1 - 2b)/(uv)`, and the
leaf vertical gradient sums `(k/4)(1/u - 2)` and `(l/4)(1/v - 2)` into Eq. (1) yields Eq. (2). -/
theorem edge_leaf_curvature_eq (k l a b p₁₁ : ℝ)
    (ha0 : a ≠ 0) (ha1 : 1 - a ≠ 0) (hb0 : b ≠ 0) (hb1 : 1 - b ≠ 0) :
    let u := a * (1 - a)
    let v := b * (1 - b)
    let w := p₁₁ - a * b
    let lapLogF := (k / 2) * (-2) + (l / 2) * (-2)
    let gamLogU := (1 - 2 * a) ^ 2 / u
    let gamLogV := (1 - 2 * b) ^ 2 / v
    let gamLogUV := w * (1 - 2 * a) * (1 - 2 * b) / (u * v)
    let gamLogF := (k / 2) ^ 2 * gamLogU + (l / 2) ^ 2 * gamLogV + 2 * (k / 2) * (l / 2) * gamLogUV
    let vertA := (k / 4) * (u / a ^ 2 + u / (1 - a) ^ 2)
    let vertB := (l / 4) * (v / b ^ 2 + v / (1 - b) ^ 2)
    (3 / 2 : ℝ) - 2 * lapLogF - gamLogF - (vertA + vertB) = edgeCurvature k l a b p₁₁ := by
  dsimp only [edgeCurvature, leafContribution]
  have hu : a * (1 - a) ≠ 0 := mul_ne_zero ha0 ha1
  have hv : b * (1 - b) ≠ 0 := mul_ne_zero hb0 hb1
  field_simp [ha0, ha1, hb0, hb1, hu, hv]
  ring

/-- Exact specialization of Eq. (2) for the 5-node tree `(k, l) = (1, 2)`:
`R = 14 - 1/(2u) - 3/(2v) - w(1 - 2a)(1 - 2b)/(uv)`. -/
theorem fiveNodeTree_curvature_formula (a b p₁₁ : ℝ) :
    let u := a * (1 - a)
    let v := b * (1 - b)
    let w := p₁₁ - a * b
    edgeCurvature 1 2 a b p₁₁ =
      14 - 1 / (2 * u) - 3 / (2 * v) - w * (1 - 2 * a) * (1 - 2 * b) / (u * v) := by
  dsimp only [edgeCurvature, leafContribution]
  have h1 : (1 : ℝ) * (1 + 1) / (4 * (a * (1 - a))) = 1 / (2 * (a * (1 - a))) := by
    rw [show (4 : ℝ) * (a * (1 - a)) = 2 * (2 * (a * (1 - a))) by ring,
      show (1 : ℝ) * (1 + 1) = 2 * 1 by ring, mul_div_mul_left _ _ (by norm_num)]
  have h2 : (2 : ℝ) * (2 + 1) / (4 * (b * (1 - b))) = 3 / (2 * (b * (1 - b))) := by
    rw [show (4 : ℝ) * (b * (1 - b)) = 2 * (2 * (b * (1 - b))) by ring,
      show (2 : ℝ) * (2 + 1) = 2 * 3 by ring, mul_div_mul_left _ _ (by norm_num)]
  rw [h1, h2]
  ring

/-! ### 2. Exact Beta Integrals and Five-Node Tree Counterexample (Theorem 1) -/

/-- Exact Beta integral `J(s) = ∫_{a ∈ (0, 1)} (a (1 - a))^s da` (Lebesgue integral on `(0, 1)`). -/
noncomputable def betaIntegral (s : ℝ) : ℝ :=
  ∫ a in Set.Ioo (0 : ℝ) 1, (a * (1 - a)) ^ s

/-- Exact sine-power integral `W_n = ∫₀^π sin^n θ dθ`. -/
noncomputable def sinPowIntegral (n : ℕ) : ℝ :=
  ∫ θ in (0 : ℝ)..Real.pi, Real.sin θ ^ n

/-- Exact Wallis recurrence on `[0, π]`: `W_{n+2} = ((n + 1) / (n + 2)) * W_n`. -/
theorem sinPowIntegral_recurrence (n : ℕ) :
    sinPowIntegral (n + 2) = ((n : ℝ) + 1) / ((n : ℝ) + 2) * sinPowIntegral n := by
  unfold sinPowIntegral
  rw [integral_sin_pow n]
  simp [Real.sin_zero, Real.sin_pi]

theorem sinPowIntegral_zero : sinPowIntegral 0 = Real.pi := by
  unfold sinPowIntegral
  simp

theorem sinPowIntegral_one : sinPowIntegral 1 = 2 := by
  unfold sinPowIntegral
  simp [integral_sin]
  norm_num

theorem sinPowIntegral_two : sinPowIntegral 2 = Real.pi / 2 := by
  rw [sinPowIntegral_recurrence 0, sinPowIntegral_zero]
  ring

theorem sinPowIntegral_four : sinPowIntegral 4 = 3 * Real.pi / 8 := by
  rw [sinPowIntegral_recurrence 2, sinPowIntegral_two]
  ring

theorem sinPowIntegral_pos (n : ℕ) : 0 < sinPowIntegral n := by
  induction n using Nat.strong_induction_on with
  | h n ih =>
    match n with
    | 0 => rw [sinPowIntegral_zero]; exact Real.pi_pos
    | 1 => rw [sinPowIntegral_one]; norm_num
    | m + 2 =>
      rw [sinPowIntegral_recurrence m]
      have hpos : 0 < sinPowIntegral m := ih m (by omega)
      positivity

/-- Exact diffeomorphism reduction of `J((K - 1)/2)` to `(1 / 2^K) * ∫₀^π sin^K θ dθ`
for EVERY `K : ℕ` (including the improper case `K = 0`, `s = -1/2`), via the `C¹`
substitution `a = (1 - cos θ) / 2` mapping `(0, π)` bijectively onto `(0, 1)`. -/
theorem betaIntegral_half_int (K : ℕ) :
    betaIntegral (((K : ℝ) - 1) / 2) = (1 / (2 : ℝ) ^ K) * sinPowIntegral K := by
  unfold betaIntegral sinPowIntegral
  let g : ℝ → ℝ := fun θ => (1 - Real.cos θ) / 2
  let g' : ℝ → ℝ := fun θ => Real.sin θ / 2
  have hg_deriv : ∀ θ ∈ Set.Ioo (0 : ℝ) Real.pi,
      HasDerivWithinAt g (g' θ) (Set.Ioo 0 Real.pi) θ := by
    intro θ _
    have hd : HasDerivAt g (g' θ) θ := by
      have h1 : HasDerivAt (fun x => (1 : ℝ) - Real.cos x) (Real.sin θ) θ := by
        simpa using (hasDerivAt_const θ (1 : ℝ)).sub (Real.hasDerivAt_cos θ)
      simpa [g, g'] using h1.div_const 2
    exact hd.hasDerivWithinAt
  have hg_inj : Set.InjOn g (Set.Ioo 0 Real.pi) := by
    intro x hx y hy hxy
    have hcos : Real.cos x = Real.cos y := by
      dsimp [g] at hxy; linarith
    exact Real.injOn_cos (Set.Ioo_subset_Icc_self hx) (Set.Ioo_subset_Icc_self hy) hcos
  have hg_img : g '' Set.Ioo 0 Real.pi = Set.Ioo 0 1 := by
    ext a
    constructor
    · rintro ⟨θ, ⟨hθ0, hθπ⟩, rfl⟩
      have hcos_lt : Real.cos θ < 1 := by
        rw [← Real.cos_zero]
        exact Real.strictAntiOn_cos (Set.left_mem_Icc.mpr Real.pi_pos.le)
          (Set.Ioo_subset_Icc_self ⟨hθ0, hθπ⟩) hθ0
      have hcos_gt : -1 < Real.cos θ := by
        rw [← Real.cos_pi]
        exact Real.strictAntiOn_cos (Set.Ioo_subset_Icc_self ⟨hθ0, hθπ⟩)
          (Set.right_mem_Icc.mpr Real.pi_pos.le) hθπ
      dsimp [g]
      exact ⟨by linarith, by linarith⟩
    · rintro ⟨ha0, ha1⟩
      refine ⟨Real.arccos (1 - 2 * a), ?_, ?_⟩
      · constructor
        · exact Real.arccos_pos.mpr (by linarith)
        · exact lt_of_le_of_ne (Real.arccos_le_pi _) fun h => by
            have hc := congrArg Real.cos h
            rw [Real.cos_arccos (by linarith) (by linarith), Real.cos_pi] at hc
            linarith
      · dsimp [g]
        rw [Real.cos_arccos (by linarith) (by linarith)]
        ring
  rw [← hg_img]
  rw [MeasureTheory.integral_image_eq_integral_abs_deriv_smul measurableSet_Ioo hg_deriv hg_inj]
  have hcongr : ∫ θ in Set.Ioo (0 : ℝ) Real.pi, |g' θ| • (g θ * (1 - g θ)) ^ (((K : ℝ) - 1) / 2) =
      ∫ θ in Set.Ioo (0 : ℝ) Real.pi, (1 / (2 : ℝ) ^ K) * Real.sin θ ^ K := by
    refine MeasureTheory.setIntegral_congr_fun measurableSet_Ioo fun θ hθ => ?_
    have hsin : 0 < Real.sin θ := Real.sin_pos_of_pos_of_lt_pi hθ.1 hθ.2
    have hg'_pos : 0 < g' θ := by dsimp [g']; linarith
    rw [smul_eq_mul, abs_of_pos hg'_pos]
    have hprod : g θ * (1 - g θ) = (Real.sin θ / 2) ^ 2 := by
      dsimp [g]
      have hsq := Real.sin_sq θ
      nlinarith
    rw [hprod]
    have hhalf_pos : 0 < Real.sin θ / 2 := by linarith
    rw [← Real.rpow_natCast (Real.sin θ / 2) 2, ← Real.rpow_mul hhalf_pos.le]
    have hexp : ((2 : ℕ) : ℝ) * (((K : ℝ) - 1) / 2) = (K : ℝ) - 1 := by push_cast; ring
    rw [hexp]
    dsimp [g']
    calc (Real.sin θ / 2) * (Real.sin θ / 2) ^ ((K : ℝ) - 1)
        = (Real.sin θ / 2) ^ (1 : ℝ) * (Real.sin θ / 2) ^ ((K : ℝ) - 1) := by
          rw [Real.rpow_one]
      _ = (Real.sin θ / 2) ^ ((1 : ℝ) + ((K : ℝ) - 1)) := by
          rw [← Real.rpow_add hhalf_pos]
      _ = (Real.sin θ / 2) ^ (K : ℝ) := by ring_nf
      _ = (Real.sin θ / 2) ^ K := Real.rpow_natCast _ K
      _ = (1 / (2 : ℝ) ^ K) * Real.sin θ ^ K := by
          rw [div_pow]; ring
  rw [hcongr, MeasureTheory.integral_const_mul]
  congr 1
  rw [intervalIntegral.integral_of_le Real.pi_pos.le, MeasureTheory.integral_Ioc_eq_integral_Ioo]

/-- Exact evaluation `J(-1/2) = π`. -/
theorem betaIntegral_neg_half : betaIntegral (-(1 : ℝ) / 2) = Real.pi := by
  have h := betaIntegral_half_int 0
  have heq : (((0 : ℕ) : ℝ) - 1) / 2 = -(1 : ℝ) / 2 := by norm_num
  rw [heq] at h
  rw [h, sinPowIntegral_zero]
  norm_num

/-- Exact evaluation `J(1/2) = π / 8`. -/
theorem betaIntegral_half : betaIntegral ((1 : ℝ) / 2) = Real.pi / 8 := by
  have h := betaIntegral_half_int 2
  have heq : (((2 : ℕ) : ℝ) - 1) / 2 = (1 : ℝ) / 2 := by norm_num
  rw [heq] at h
  rw [h, sinPowIntegral_two]
  ring

/-- Exact evaluation `J(3/2) = 3π / 128`. -/
theorem betaIntegral_three_halves : betaIntegral ((3 : ℝ) / 2) = 3 * Real.pi / 128 := by
  have h := betaIntegral_half_int 4
  have heq : (((4 : ℕ) : ℝ) - 1) / 2 = (3 : ℝ) / 2 := by norm_num
  rw [heq] at h
  rw [h, sinPowIntegral_four]
  ring

/-- Exact first moment of `Beta(1/2, 1/2)`:
`∫₀¹ t · (π⁻¹ (t(1-t))⁻¹/²) dt = 1/2`. -/
theorem betaHalf_first_moment :
    (1 / Real.pi) * ∫ t in Set.Ioo (0 : ℝ) 1, t * (t * (1 - t)) ^ (-(1 : ℝ) / 2) = 1 / 2 := by
  have hsym : ∫ t in Set.Ioo (0 : ℝ) 1, (1 - t) * (t * (1 - t)) ^ (-(1 : ℝ) / 2) =
      ∫ t in Set.Ioo (0 : ℝ) 1, t * (t * (1 - t)) ^ (-(1 : ℝ) / 2) := by
    let g : ℝ → ℝ := fun t => 1 - t
    have hg_img : g '' Set.Ioo (0 : ℝ) 1 = Set.Ioo 0 1 := by
      ext x; constructor
      · rintro ⟨t, ⟨ht0, ht1⟩, rfl⟩; exact ⟨by dsimp [g]; linarith, by dsimp [g]; linarith⟩
      · rintro ⟨hx0, hx1⟩; exact ⟨1 - x, ⟨by linarith, by linarith⟩, by dsimp [g]; ring⟩
    have hg_deriv : ∀ t ∈ Set.Ioo (0 : ℝ) 1, HasDerivWithinAt g (-1) (Set.Ioo 0 1) t := fun t _ =>
      ((hasDerivAt_const t (1 : ℝ)).sub (hasDerivAt_id t)).hasDerivWithinAt.congr_deriv (by ring)
    have hg_inj : Set.InjOn g (Set.Ioo 0 1) := fun x _ y _ h => by dsimp [g] at h; linarith
    have h := MeasureTheory.integral_image_eq_integral_abs_deriv_smul
      measurableSet_Ioo hg_deriv hg_inj
      (fun u => u * (u * (1 - u)) ^ (-(1 : ℝ) / 2))
    rw [hg_img] at h
    rw [h]
    refine MeasureTheory.setIntegral_congr_fun measurableSet_Ioo fun t _ => ?_
    simp only [abs_neg, abs_one, one_smul, g]
    congr 2; ring
  have hint_tot : IntegrableOn (fun t : ℝ => (t * (1 - t)) ^ (-(1 : ℝ) / 2)) (Set.Ioo 0 1) := by
    by_contra hnot
    have h0 : ∫ t in Set.Ioo (0 : ℝ) 1, (t * (1 - t)) ^ (-(1 : ℝ) / 2) = 0 :=
      MeasureTheory.integral_undef hnot
    have hpi : ∫ t in Set.Ioo (0 : ℝ) 1, (t * (1 - t)) ^ (-(1 : ℝ) / 2) = Real.pi :=
      betaIntegral_neg_half
    exact Real.pi_ne_zero (hpi.symm.trans h0)
  have hint_t : IntegrableOn (fun t : ℝ => t * (t * (1 - t)) ^ (-(1 : ℝ) / 2)) (Set.Ioo 0 1) := by
    refine hint_tot.mono'
      (AEStronglyMeasurable.mul aestronglyMeasurable_id hint_tot.aestronglyMeasurable) ?_
    filter_upwards [ae_restrict_mem measurableSet_Ioo] with t ht
    rw [Real.norm_eq_abs, abs_mul, abs_of_pos ht.1]
    have hnn : 0 ≤ (t * (1 - t)) ^ (-(1 : ℝ) / 2) :=
      Real.rpow_nonneg (by nlinarith [ht.1, ht.2]) _
    rw [abs_of_nonneg hnn]
    nlinarith [ht.2]
  have hint_1mt : IntegrableOn (fun t : ℝ => (1 - t) * (t * (1 - t)) ^ (-(1 : ℝ) / 2))
      (Set.Ioo 0 1) := by
    refine hint_tot.mono' (((aestronglyMeasurable_const.sub aestronglyMeasurable_id)).mul
      hint_tot.aestronglyMeasurable) ?_
    filter_upwards [ae_restrict_mem measurableSet_Ioo] with t ht
    rw [Real.norm_eq_abs, abs_mul, abs_of_pos (by linarith [ht.2])]
    have hnn : 0 ≤ (t * (1 - t)) ^ (-(1 : ℝ) / 2) :=
      Real.rpow_nonneg (by nlinarith [ht.1, ht.2]) _
    rw [abs_of_nonneg hnn]
    nlinarith [ht.1]
  have hsum : (∫ t in Set.Ioo (0 : ℝ) 1, t * (t * (1 - t)) ^ (-(1 : ℝ) / 2)) +
      (∫ t in Set.Ioo (0 : ℝ) 1, (1 - t) * (t * (1 - t)) ^ (-(1 : ℝ) / 2)) = Real.pi := by
    rw [← MeasureTheory.integral_add hint_t hint_1mt]
    have heq : (fun t : ℝ => t * (t * (1 - t)) ^ (-(1 : ℝ) / 2) +
        (1 - t) * (t * (1 - t)) ^ (-(1 : ℝ) / 2)) =
        fun t => (t * (1 - t)) ^ (-(1 : ℝ) / 2) := by
      ext t; ring
    rw [heq]
    exact betaIntegral_neg_half
  have hpi : Real.pi ≠ 0 := Real.pi_ne_zero
  have hval : ∫ t in Set.Ioo (0 : ℝ) 1, t * (t * (1 - t)) ^ (-(1 : ℝ) / 2) = Real.pi / 2 := by
    linarith
  rw [hval]
  field_simp [hpi]

/-- Exact variance/second-moment identity for `Beta(1/2, 1/2)`:
`E[t(1 - t)] = J(1/2) / J(-1/2) = 1/8`, hence `E[t²] = E[t] - E[t(1-t)] = 3/8` and
`Var(t) = 1/8`. -/
theorem betaHalf_product_moment :
    betaIntegral ((1 : ℝ) / 2) / betaIntegral (-(1 : ℝ) / 2) = 1 / 8 := by
  rw [betaIntegral_half, betaIntegral_neg_half]
  have hpi : Real.pi ≠ 0 := Real.pi_ne_zero
  field_simp [hpi]

/-- Exact conditional expectation `E₀[v | a] = (1 + 2u)/8` from independent
`Beta(1/2, 1/2)` moments `E[t_i] = 1/2`, `E[t_i²] = 3/8`, `E[t₀ t₁] = 1/4`. -/
theorem condExp_v_given_a (a : ℝ) :
    let u := a * (1 - a)
    (1 - a) * (1 / 2 : ℝ) + a * (1 / 2)
      - (1 - a) ^ 2 * (3 / 8) - 2 * a * (1 - a) * (1 / 4) - a ^ 2 * (3 / 8) =
    (1 + 2 * u) / 8 := by
  ring

/-- Exact conditional expectation `E₀[w(1 - 2b) | a] = u(1 - 2a)/4` from independent
`Beta(1/2, 1/2)` moments `E[t_i] = 1/2`, `E[t_i²] = 3/8`, `E[t₀ t₁] = 1/4`. -/
theorem condExp_cross_given_a (a : ℝ) :
    let u := a * (1 - a)
    u * ((1 / 2 : ℝ) - 1 / 2 + 2 * (1 - a) * (3 / 8) - 2 * a * (3 / 8)
      + 2 * (2 * a - 1) * (1 / 4)) =
    u * (1 - 2 * a) / 4 := by
  ring

/-- **Theorem 1 (Five-node tree counterexample)**:
For the 5-node binary tree `A → B, A → C, B → D, B → E` (`(k, l) = (1, 2)`), the four exact
weighted integrals evaluate to `11π/512`, `5π/32`, `π/8`, `π/128`, yielding exact normalized
Fisher-volume average scalar curvature `⟨R⟩ = 14 / 11`. -/
theorem fiveNodeTree_avgCurvature :
    let I_F := (betaIntegral (1 / 2) + 2 * betaIntegral (3 / 2)) / 8
    let I_Fu := (betaIntegral (-1 / 2) + 2 * betaIntegral (1 / 2)) / 8
    let I_Fv := betaIntegral (1 / 2)
    let I_cross := (betaIntegral (1 / 2) - 4 * betaIntegral (3 / 2)) / 4
    I_F = 11 * Real.pi / 512 ∧
    I_Fu = 5 * Real.pi / 32 ∧
    I_Fv = Real.pi / 8 ∧
    I_cross = Real.pi / 128 ∧
    14 - ((1 / 2) * I_Fu + (3 / 2) * I_Fv + I_cross) / I_F = 14 / 11 := by
  dsimp only
  rw [betaIntegral_neg_half, betaIntegral_half, betaIntegral_three_halves]
  have hpi : Real.pi ≠ 0 := Real.pi_ne_zero
  refine ⟨by ring, by ring, by ring, by ring, ?_⟩
  field_simp [hpi]
  ring

/-- `14 / 11 ∉ (1/2)ℤ` (over `ℝ`), disproving Rodríguez (2004) `conj:main` and the tree
quantization theorem of arXiv:2603.10054v1. -/
theorem fourteen_elevenths_not_half_int : ¬ ∃ m : ℤ, (14 : ℝ) / 11 = (m : ℝ) / 2 := by
  rintro ⟨m, hm⟩
  have h1 : (28 : ℝ) = 11 * (m : ℝ) := by linarith
  have h2 : (28 : ℤ) = 11 * m := by exact_mod_cast h1
  omega

/-! ### 3. Exact Dirichlet(1/2, 1/2, 1/2, 1/2) Moments, 6-Node Tree (Prop. 2), & 7-Node 1-Loop (Prop. 4) -/

/-- Rational rising factorial `(x)_n = x (x + 1) ⋯ (x + n - 1)`. -/
def risingFact (x : ℚ) : ℕ → ℚ
  | 0 => 1
  | n + 1 => risingFact x n * (x + n)

/-- Exact `Dirichlet(1/2, 1/2, 1/2, 1/2)` monomial moment
`E₀[p₀₀^a p₀₁^b p₁₀^c p₁₁^d] = ∏ (1/2)_{n_α} / (2)_{∑ n_α}`. -/
def dirichletHalfMoment (n₀₀ n₀₁ n₁₀ n₁₁ : ℕ) : ℚ :=
  risingFact (1 / 2) n₀₀ * risingFact (1 / 2) n₀₁ *
    risingFact (1 / 2) n₁₀ * risingFact (1 / 2) n₁₁ /
    risingFact 2 (n₀₀ + n₀₁ + n₁₀ + n₁₁)

/-- Exact moment of a finite rational linear combination of 4-variable monomials under
`Dirichlet(1/2, 1/2, 1/2, 1/2)`. -/
def dirichletPolyExp (terms : List (ℚ × ℕ × ℕ × ℕ × ℕ)) : ℚ :=
  (terms.map fun (c, a, b, d, e) => c * dirichletHalfMoment a b d e).sum

/-- Polynomial multiplication on sparse 4-variable monomial lists (for exact algebraic
expansion of `u`, `v`, `w`, `1 - 2a`, `1 - 2b` in `(p₀₀, p₀₁, p₁₀, p₁₁)`). -/
def polyMul (p q : List (ℚ × ℕ × ℕ × ℕ × ℕ)) : List (ℚ × ℕ × ℕ × ℕ × ℕ) :=
  p.flatMap fun (c₁, a₁, b₁, d₁, e₁) =>
    q.map fun (c₂, a₂, b₂, d₂, e₂) => (c₁ * c₂, a₁ + a₂, b₁ + b₂, d₁ + d₂, e₁ + e₂)

def poly_a : List (ℚ × ℕ × ℕ × ℕ × ℕ) := [(1, 0, 0, 1, 0), (1, 0, 0, 0, 1)]
def poly_1ma : List (ℚ × ℕ × ℕ × ℕ × ℕ) := [(1, 1, 0, 0, 0), (1, 0, 1, 0, 0)]
def poly_b : List (ℚ × ℕ × ℕ × ℕ × ℕ) := [(1, 0, 1, 0, 0), (1, 0, 0, 0, 1)]
def poly_1mb : List (ℚ × ℕ × ℕ × ℕ × ℕ) := [(1, 1, 0, 0, 0), (1, 0, 0, 1, 0)]
def poly_u : List (ℚ × ℕ × ℕ × ℕ × ℕ) := polyMul poly_a poly_1ma
def poly_v : List (ℚ × ℕ × ℕ × ℕ × ℕ) := polyMul poly_b poly_1mb
def poly_w : List (ℚ × ℕ × ℕ × ℕ × ℕ) := [(1, 1, 0, 0, 1), (-1, 0, 1, 1, 0)]
def poly_1m2a : List (ℚ × ℕ × ℕ × ℕ × ℕ) :=
  [(1, 1, 0, 0, 0), (1, 0, 1, 0, 0), (-1, 0, 0, 1, 0), (-1, 0, 0, 0, 1)]
def poly_1m2b : List (ℚ × ℕ × ℕ × ℕ × ℕ) :=
  [(1, 1, 0, 0, 0), (-1, 0, 1, 0, 0), (1, 0, 0, 1, 0), (-1, 0, 0, 0, 1)]

/-- Algebraic verification that `w = p₁₁ - ab = p₀₀ p₁₁ - p₀₁ p₁₀` on the simplex
`p₀₀ + p₀₁ + p₁₀ + p₁₁ = 1`. -/
theorem cov_simplex_identity (p₀₀ p₀₁ p₁₀ p₁₁ : ℝ) (hsum : p₀₀ + p₀₁ + p₁₀ + p₁₁ = 1) :
    p₁₁ - (p₁₀ + p₁₁) * (p₀₁ + p₁₁) = p₀₀ * p₁₁ - p₀₁ * p₁₀ := by
  calc p₁₁ - (p₁₀ + p₁₁) * (p₀₁ + p₁₁)
      = p₁₁ * (p₀₀ + p₀₁ + p₁₀ + p₁₁) - (p₁₀ + p₁₁) * (p₀₁ + p₁₁) := by rw [hsum, mul_one]
    _ = p₀₀ * p₁₁ - p₀₁ * p₁₀ := by ring

/-- Odd double factorial `(2n - 1)!! = 1 · 3 · ⋯ · (2n - 1)`. -/
def oddDoubleFact : ℕ → ℕ
  | 0 => 1
  | n + 1 => oddDoubleFact n * (2 * n + 1)

lemma risingFact_half (n : ℕ) : risingFact (1 / 2) n = (oddDoubleFact n : ℚ) / (2 : ℚ) ^ n := by
  induction n with
  | zero => simp [risingFact, oddDoubleFact]
  | succ k ih =>
    simp only [risingFact, oddDoubleFact, ih, Nat.cast_mul, Nat.cast_add, Nat.cast_one,
      Nat.cast_ofNat, pow_succ]
    ring

lemma risingFact_two (n : ℕ) : risingFact 2 n = ((n + 1).factorial : ℚ) := by
  induction n with
  | zero => simp [risingFact]
  | succ k ih =>
    simp only [risingFact, ih, Nat.factorial_succ, Nat.cast_mul, Nat.cast_add, Nat.cast_one]
    ring

/-- Integer-scaled `Dirichlet(1/2, 1/2, 1/2, 1/2)` moment for degree `≤ 6`:
`322560 * E₀[p₀₀^a p₀₁^b p₁₀^c p₁₁^d]` (`322560 = 2⁶ · 7!`), exact integer value for fast kernel
verification. -/
lemma dirichletHalfMoment_eq (a b d e : ℕ) :
    dirichletHalfMoment a b d e =
      ((oddDoubleFact a * oddDoubleFact b * oddDoubleFact d * oddDoubleFact e : ℕ) : ℚ) /
        ((2 ^ (a + b + d + e) * (a + b + d + e + 1).factorial : ℕ) : ℚ) := by
  unfold dirichletHalfMoment
  rw [risingFact_half, risingFact_half, risingFact_half, risingFact_half, risingFact_two]
  push_cast
  rw [pow_add, pow_add, pow_add]
  ring

/-- Helper: rational evaluation of a sparse 4-variable polynomial moment via integer scaling by `2⁶·7! = 322560`. -/
def scaledDirichletExp (terms : List (ℤ × ℕ × ℕ × ℕ × ℕ)) : ℤ :=
  (terms.map fun (c, a, b, d, e) =>
    let N := a + b + d + e
    c * ((oddDoubleFact a * oddDoubleFact b * oddDoubleFact d * oddDoubleFact e : ℕ) : ℤ) *
      (((2 ^ (6 - N) * (Nat.factorial 7 / Nat.factorial (N + 1)) : ℕ) : ℤ))).sum

lemma dirichletPolyExp_eq_scaled (terms : List (ℤ × ℕ × ℕ × ℕ × ℕ))
    (hdeg : ∀ t ∈ terms, t.2.1 + t.2.2.1 + t.2.2.2.1 + t.2.2.2.2 ≤ 6) :
    dirichletPolyExp (terms.map fun (c, a, b, d, e) => ((c : ℚ), a, b, d, e)) =
      (scaledDirichletExp terms : ℚ) / 322560 := by
  induction terms with
  | nil => simp [dirichletPolyExp, scaledDirichletExp]
  | cons hd tl ih =>
    rcases hd with ⟨c, a, b, d, e⟩
    have hhd : a + b + d + e ≤ 6 := hdeg (c, a, b, d, e) (by simp)
    have htl : ∀ t ∈ tl, t.2.1 + t.2.2.1 + t.2.2.2.1 + t.2.2.2.2 ≤ 6 :=
      fun t ht => hdeg t (by simp [ht])
    simp only [dirichletPolyExp, scaledDirichletExp, List.map_cons, List.sum_cons,
      Int.cast_add] at ih ⊢
    rw [ih htl, add_div]
    congr 1
    rw [dirichletHalfMoment_eq]
    set N := a + b + d + e
    have hpow : 2 ^ N * 2 ^ (6 - N) = 64 := by
      rw [← pow_add, Nat.add_sub_of_le hhd]; norm_num
    have hfac_dvd : (N + 1).factorial ∣ (7 : ℕ).factorial :=
      Nat.factorial_dvd_factorial (by omega)
    have hfac : (N + 1).factorial * ((7 : ℕ).factorial / (N + 1).factorial) = 5040 := by
      rw [Nat.mul_div_cancel' hfac_dvd]; decide
    have hden_ne : ((2 ^ N * (N + 1).factorial : ℕ) : ℚ) ≠ 0 := by
      exact Nat.cast_ne_zero.mpr (Nat.mul_ne_zero (by positivity) (Nat.factorial_ne_zero _))
    have hprod : ((2 ^ N * (N + 1).factorial : ℕ) : ℚ) *
        ((2 ^ (6 - N) * ((7 : ℕ).factorial / (N + 1).factorial) : ℕ) : ℚ) = 322560 := by
      have hnat : (2 ^ N * (N + 1).factorial) * (2 ^ (6 - N) * ((7 : ℕ).factorial / (N + 1).factorial)) =
          322560 := by
        calc (2 ^ N * (N + 1).factorial) * (2 ^ (6 - N) * ((7 : ℕ).factorial / (N + 1).factorial))
            = (2 ^ N * 2 ^ (6 - N)) * ((N + 1).factorial * ((7 : ℕ).factorial / (N + 1).factorial)) := by ring
          _ = 64 * 5040 := by rw [hpow, hfac]
          _ = 322560 := by norm_num
      exact_mod_cast hnat
    rw [Int.cast_mul, Int.cast_mul, Int.cast_natCast, Int.cast_natCast]
    field_simp [hden_ne]
    rw [← hprod]
    ring

def polyMulZ (p q : List (ℤ × ℕ × ℕ × ℕ × ℕ)) : List (ℤ × ℕ × ℕ × ℕ × ℕ) :=
  p.flatMap fun (c₁, a₁, b₁, d₁, e₁) =>
    q.map fun (c₂, a₂, b₂, d₂, e₂) => (c₁ * c₂, a₁ + a₂, b₁ + b₂, d₁ + d₂, e₁ + e₂)

private def polyZ_a : List (ℤ × ℕ × ℕ × ℕ × ℕ) := [(1, 0, 0, 1, 0), (1, 0, 0, 0, 1)]
private def polyZ_1ma : List (ℤ × ℕ × ℕ × ℕ × ℕ) := [(1, 1, 0, 0, 0), (1, 0, 1, 0, 0)]
private def polyZ_b : List (ℤ × ℕ × ℕ × ℕ × ℕ) := [(1, 0, 1, 0, 0), (1, 0, 0, 0, 1)]
private def polyZ_1mb : List (ℤ × ℕ × ℕ × ℕ × ℕ) := [(1, 1, 0, 0, 0), (1, 0, 0, 1, 0)]
private def polyZ_u : List (ℤ × ℕ × ℕ × ℕ × ℕ) := polyMulZ polyZ_a polyZ_1ma
private def polyZ_v : List (ℤ × ℕ × ℕ × ℕ × ℕ) := polyMulZ polyZ_b polyZ_1mb
private def polyZ_w : List (ℤ × ℕ × ℕ × ℕ × ℕ) := [(1, 1, 0, 0, 1), (-1, 0, 1, 1, 0)]
private def polyZ_1m2a : List (ℤ × ℕ × ℕ × ℕ × ℕ) :=
  [(1, 1, 0, 0, 0), (1, 0, 1, 0, 0), (-1, 0, 0, 1, 0), (-1, 0, 0, 0, 1)]
private def polyZ_1m2b : List (ℤ × ℕ × ℕ × ℕ × ℕ) :=
  [(1, 1, 0, 0, 0), (-1, 0, 1, 0, 0), (1, 0, 0, 1, 0), (-1, 0, 0, 0, 1)]

lemma polyMul_map_cast (p q : List (ℤ × ℕ × ℕ × ℕ × ℕ)) :
    polyMul (p.map fun (c, a, b, d, e) => ((c : ℚ), a, b, d, e))
            (q.map fun (c, a, b, d, e) => ((c : ℚ), a, b, d, e)) =
    (polyMulZ p q).map fun (c, a, b, d, e) => ((c : ℚ), a, b, d, e) := by
  induction p with
  | nil => simp [polyMul, polyMulZ]
  | cons hd tl ih =>
    rcases hd with ⟨c₁, a₁, b₁, d₁, e₁⟩
    simp only [polyMul, polyMulZ, List.map_cons, List.flatMap_cons, List.map_append,
      List.map_map, Function.comp_def] at ih ⊢
    rw [ih]
    congr 1
    refine List.map_congr_left fun t _ => ?_
    rcases t with ⟨c₂, a₂, b₂, d₂, e₂⟩
    simp

/-- Exact `Dirichlet(1/2, 1/2, 1/2, 1/2)` moments needed for Propositions 2 and 4. -/
theorem dirichlet_edge_moments :
    dirichletPolyExp poly_u = 1 / 6 ∧
    dirichletPolyExp poly_v = 1 / 6 ∧
    dirichletPolyExp (polyMul poly_u poly_v) = 7 / 240 ∧
    dirichletPolyExp (polyMul poly_w (polyMul poly_1m2a poly_1m2b)) = 1 / 120 ∧
    dirichletPolyExp (polyMul poly_u poly_u) = 1 / 30 ∧
    dirichletPolyExp (polyMul (polyMul poly_u poly_u) poly_v) = 1 / 168 ∧
    dirichletPolyExp (polyMul poly_u (polyMul poly_w (polyMul poly_1m2a poly_1m2b))) = 1 / 840 := by
  have hu_eq : poly_u = polyZ_u.map fun (c, a, b, d, e) => ((c : ℚ), a, b, d, e) := by
    simpa [poly_u, polyZ_u, poly_a, poly_1ma, polyZ_a, polyZ_1ma] using polyMul_map_cast polyZ_a polyZ_1ma
  have hv_eq : poly_v = polyZ_v.map fun (c, a, b, d, e) => ((c : ℚ), a, b, d, e) := by
    simpa [poly_v, polyZ_v, poly_b, poly_1mb, polyZ_b, polyZ_1mb] using polyMul_map_cast polyZ_b polyZ_1mb
  have hw_eq : poly_w = polyZ_w.map fun (c, a, b, d, e) => ((c : ℚ), a, b, d, e) := by
    simp [poly_w, polyZ_w]
  have h1a_eq : poly_1m2a = polyZ_1m2a.map fun (c, a, b, d, e) => ((c : ℚ), a, b, d, e) := by
    simp [poly_1m2a, polyZ_1m2a]
  have h1b_eq : poly_1m2b = polyZ_1m2b.map fun (c, a, b, d, e) => ((c : ℚ), a, b, d, e) := by
    simp [poly_1m2b, polyZ_1m2b]
  rw [hu_eq, hv_eq, hw_eq, h1a_eq, h1b_eq]
  simp only [polyMul_map_cast]
  have hdeg_mul : ∀ (p q : List (ℤ × ℕ × ℕ × ℕ × ℕ)) (d₁ d₂ : ℕ),
      (∀ t ∈ p, t.2.1 + t.2.2.1 + t.2.2.2.1 + t.2.2.2.2 ≤ d₁) →
      (∀ t ∈ q, t.2.1 + t.2.2.1 + t.2.2.2.1 + t.2.2.2.2 ≤ d₂) →
      ∀ t ∈ polyMulZ p q, t.2.1 + t.2.2.1 + t.2.2.2.1 + t.2.2.2.2 ≤ d₁ + d₂ := by
    intro p q d₁ d₂ hp hq t ht
    simp only [polyMulZ, List.mem_flatMap, List.mem_map] at ht
    rcases ht with ⟨⟨c₁, a₁, b₁, x₁, e₁⟩, h1, ⟨c₂, a₂, b₂, x₂, e₂⟩, h2, rfl⟩
    have := hp _ h1; have := hq _ h2; dsimp only at *; omega
  have hdu : ∀ t ∈ polyZ_u, t.2.1 + t.2.2.1 + t.2.2.2.1 + t.2.2.2.2 ≤ 2 := by decide
  have hdv : ∀ t ∈ polyZ_v, t.2.1 + t.2.2.1 + t.2.2.2.1 + t.2.2.2.2 ≤ 2 := by decide
  have hdw : ∀ t ∈ polyZ_w, t.2.1 + t.2.2.1 + t.2.2.2.1 + t.2.2.2.2 ≤ 2 := by decide
  have hd1a : ∀ t ∈ polyZ_1m2a, t.2.1 + t.2.2.1 + t.2.2.2.1 + t.2.2.2.2 ≤ 1 := by decide
  have hd1b : ∀ t ∈ polyZ_1m2b, t.2.1 + t.2.2.1 + t.2.2.2.1 + t.2.2.2.2 ≤ 1 := by decide
  have hd_uv := hdeg_mul _ _ 2 2 hdu hdv
  have hd_1ab := hdeg_mul _ _ 1 1 hd1a hd1b
  have hd_w1ab := hdeg_mul _ _ 2 2 hdw hd_1ab
  have hd_uu := hdeg_mul _ _ 2 2 hdu hdu
  have hd_uuv := hdeg_mul _ _ 4 2 hd_uu hdv
  have hd_uw1ab := hdeg_mul _ _ 2 4 hdu hd_w1ab
  rw [dirichletPolyExp_eq_scaled _ (fun t ht => le_trans (hdu t ht) (by norm_num)),
      dirichletPolyExp_eq_scaled _ (fun t ht => le_trans (hdv t ht) (by norm_num)),
      dirichletPolyExp_eq_scaled _ (fun t ht => le_trans (hd_uv t ht) (by norm_num)),
      dirichletPolyExp_eq_scaled _ (fun t ht => le_trans (hd_w1ab t ht) (by norm_num)),
      dirichletPolyExp_eq_scaled _ (fun t ht => le_trans (hd_uu t ht) (by norm_num)),
      dirichletPolyExp_eq_scaled _ (fun t ht => le_trans (hd_uuv t ht) (by norm_num)),
      dirichletPolyExp_eq_scaled _ (fun t ht => le_trans (hd_uw1ab t ht) (by norm_num))]
  have hscaled_append : ∀ xs ys : List (ℤ × ℕ × ℕ × ℕ × ℕ),
      scaledDirichletExp (xs ++ ys) = scaledDirichletExp xs + scaledDirichletExp ys := by
    intro xs ys; simp [scaledDirichletExp, List.map_append, List.sum_append]
  have hscaled_mul : ∀ (p q : List (ℤ × ℕ × ℕ × ℕ × ℕ)),
      scaledDirichletExp (polyMulZ p q) =
        (p.map fun x => scaledDirichletExp (polyMulZ [x] q)).sum := by
    intro p q
    induction p with
    | nil => simp [polyMulZ, scaledDirichletExp]
    | cons hd tl ih =>
      have : polyMulZ (hd :: tl) q = polyMulZ [hd] q ++ polyMulZ tl q := by
        simp [polyMulZ]
      rw [this, hscaled_append, ih, List.map_cons, List.sum_cons]
  have h1 : scaledDirichletExp polyZ_u = 53760 := by decide
  have h2 : scaledDirichletExp polyZ_v = 53760 := by decide
  have h3 : scaledDirichletExp (polyMulZ polyZ_u polyZ_v) = 9408 := by decide
  have h4 : scaledDirichletExp (polyMulZ polyZ_w (polyMulZ polyZ_1m2a polyZ_1m2b)) = 2688 := by decide
  have h5 : scaledDirichletExp (polyMulZ polyZ_u polyZ_u) = 10752 := by decide
  have h6 : scaledDirichletExp (polyMulZ (polyMulZ polyZ_u polyZ_u) polyZ_v) = 1920 := by
    rw [hscaled_mul]; decide
  have h7 : scaledDirichletExp (polyMulZ polyZ_u (polyMulZ polyZ_w (polyMulZ polyZ_1m2a polyZ_1m2b))) = 384 := by
    rw [hscaled_mul]; decide
  rw [h1, h2, h3, h4, h5, h6, h7]
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩

/-- **Proposition 2 (Six-node tree `(k, l) = (2, 2)` counterexample)**:
`F = uv`, constant curvature term `3/2 + 9 + 9 = 39/2`, and exact average `⟨R⟩ = 25 / 14 ∉ (1/2)ℤ`. -/
theorem sixNodeTree_avgCurvature :
    let E_uv : ℚ := dirichletPolyExp (polyMul poly_u poly_v)
    let E_u : ℚ := dirichletPolyExp poly_u
    let E_v : ℚ := dirichletPolyExp poly_v
    let E_cross : ℚ := dirichletPolyExp (polyMul poly_w (polyMul poly_1m2a poly_1m2b))
    (39 / 2 : ℚ) - ((3 / 2) * E_v + (3 / 2) * E_u + 2 * E_cross) / E_uv = 25 / 14 := by
  dsimp only
  have hm := dirichlet_edge_moments
  rw [hm.1, hm.2.1, hm.2.2.1, hm.2.2.2.1]
  norm_num

theorem twentyfive_fourteenths_not_half_int : ¬ ∃ m : ℤ, (25 : ℝ) / 14 = (m : ℝ) / 2 := by
  rintro ⟨m, hm⟩
  have h1 : (50 : ℝ) = 14 * (m : ℝ) := by linarith
  have h2 : (50 : ℤ) = 14 * m := by exact_mod_cast h1
  omega

/-- **Proposition 4 (Seven-node 1-loop chordless 4-cycle counterexample)**:
On the edge `A → B` with `(k, l) = (4, 2)` and independent root `H` (`K = 4`, average `2`),
the edge numerator is `1/60`, denominator `E₀[u² v] = 1/168`, and exact average
`⟨R⟩ = 2 + (1/60)/(1/168) = 24 / 5 ∉ (1/2)ℤ`. -/
theorem sevenNodeOneLoop_avgCurvature :
    let E_u2v : ℚ := dirichletPolyExp (polyMul (polyMul poly_u poly_u) poly_v)
    let E_uv : ℚ := dirichletPolyExp (polyMul poly_u poly_v)
    let E_u2 : ℚ := dirichletPolyExp (polyMul poly_u poly_u)
    let E_ucross : ℚ :=
      dirichletPolyExp (polyMul poly_u (polyMul poly_w (polyMul poly_1m2a poly_1m2b)))
    let num_AB : ℚ := (3 / 2 + 26 + 9) * E_u2v - 5 * E_uv - (3 / 2) * E_u2 - 4 * E_ucross
    num_AB = 1 / 60 ∧ (2 : ℚ) + num_AB / E_u2v = 24 / 5 := by
  dsimp only
  have hm := dirichlet_edge_moments
  rw [hm.2.2.1, hm.2.2.2.2.1, hm.2.2.2.2.2.1, hm.2.2.2.2.2.2]
  constructor <;> norm_num

theorem twentyfour_fifths_not_half_int : ¬ ∃ m : ℤ, (24 : ℝ) / 5 = (m : ℝ) / 2 := by
  rintro ⟨m, hm⟩
  have h1 : (48 : ℝ) = 5 * (m : ℝ) := by linarith
  have h2 : (48 : ℤ) = 5 * m := by exact_mod_cast h1
  omega

/-! ### 4. Shared-Root Colliders and Refutation of arXiv:2603.10054v1 Claims (Proposition 3) -/

/-- Exact single-root scalar curvature `S_K(r) = K(2K + 1)/2 - K(K - 1)/(4U)` for a root
appearing `K` times per state in the sink warps. -/
noncomputable def rootCurvature (K : ℝ) (U : ℝ) : ℝ :=
  K * (2 * K + 1) / 2 - K * (K - 1) / (4 * U)

/-- Exact algebraic derivation of `S_K(r)` from Lemma 1 on a flat 1D root base (`R₀ = 0`),
where `Δ(log U) = U (log U)'' + (1/2)(1 - 2r)(log U)' = -1/(2U)`,
`Γ(log U, log U) = 1/U - 4`, and the `2K` leaf vertical terms sum to `(K/4)(1/U - 2)`. -/
theorem sharedRoot_curvature_eq (K U : ℝ) (hU : U ≠ 0) :
    -2 * (K / 2) * (-1 / (2 * U)) - (K ^ 2 / 4) * (1 / U - 4) - (K / 4) * (1 / U - 2) =
      rootCurvature K U := by
  unfold rootCurvature
  field_simp [hU]
  ring

/-- Exact Beta integral ratio `J((K - 3)/2) / J((K - 1)/2) = 4K / (K - 1)` for EVERY `K ≥ 2`,
giving `E[1/U] = 4K / (K - 1)` under the marginal density `∝ U^{(K-1)/2} dr`. -/
theorem betaIntegral_ratio (K : ℕ) (hK : 2 ≤ K) :
    betaIntegral (((K : ℝ) - 3) / 2) / betaIntegral (((K : ℝ) - 1) / 2) =
      4 * (K : ℝ) / ((K : ℝ) - 1) := by
  obtain ⟨m, rfl⟩ : ∃ m, K = m + 2 := ⟨K - 2, by omega⟩
  have h1 : (((m + 2 : ℕ) : ℝ) - 3) / 2 = (((m : ℕ) : ℝ) - 1) / 2 := by push_cast; ring
  rw [h1, betaIntegral_half_int m, betaIntegral_half_int (m + 2)]
  rw [sinPowIntegral_recurrence m]
  have hW : sinPowIntegral m ≠ 0 := ne_of_gt (sinPowIntegral_pos m)
  have hpow : (2 : ℝ) ^ m ≠ 0 := by positivity
  have hm1 : (m : ℝ) + 1 ≠ 0 := by positivity
  have hm2 : (m : ℝ) + 2 ≠ 0 := by positivity
  push_cast
  have hsub : (m : ℝ) + 2 - 1 = (m : ℝ) + 1 := by ring
  rw [hsub, pow_add]
  field_simp [hW, hpow, hm1, hm2]
  ring

/-- **Proposition 3 (Shared-root colliders)**:
For `n ≥ 1` independent roots and `m ≥ 1` common sinks (`K = m * 2^(n - 1)`):
- Whenever `K ≥ 2`, the exact normalized Fisher-volume average scalar curvature is
  `⟨R⟩ = n * K / 2 ∈ (1/2)ℤ`.
- For the degenerate case `K = 1` (`n = 1, m = 1`, the 2-node chain `K̃_2`), `S_1 ≡ 3/2`
  so `⟨R⟩ = 3/2 ∈ (1/2)ℤ`. -/
theorem sharedRootCollider_avgCurvature (n m : ℕ) (hm : 1 ≤ m) :
    let K : ℕ := m * 2 ^ (n - 1)
    let singleRootAvg : ℝ :=
      if K = 1 then 3 / 2
      else (K : ℝ) * (2 * K + 1) / 2 - ((K : ℝ) * ((K : ℝ) - 1) / 4) *
        (betaIntegral (((K : ℝ) - 3) / 2) / betaIntegral (((K : ℝ) - 1) / 2))
    (2 ≤ K → (n : ℝ) * singleRootAvg = (n : ℝ) * (K : ℝ) / 2) ∧
    ∃ z : ℤ, (n : ℝ) * singleRootAvg = (z : ℝ) / 2 := by
  dsimp only
  set K : ℕ := m * 2 ^ (n - 1)
  have hKpos : 1 ≤ K := Nat.mul_pos hm Nat.one_le_two_pow
  by_cases h1 : K = 1
  · constructor
    · intro h2; omega
    · refine ⟨(3 * n : ℕ), ?_⟩
      rw [if_pos h1]
      push_cast; ring
  · have hK2 : 2 ≤ K := by omega
    have hsingle : (if K = 1 then (3 : ℝ) / 2
        else (K : ℝ) * (2 * K + 1) / 2 - ((K : ℝ) * ((K : ℝ) - 1) / 4) *
          (betaIntegral (((K : ℝ) - 3) / 2) / betaIntegral (((K : ℝ) - 1) / 2))) = (K : ℝ) / 2 := by
      rw [if_neg h1, betaIntegral_ratio K hK2]
      have hKm1 : (K : ℝ) - 1 ≠ 0 := by
        have : (2 : ℝ) ≤ K := by exact_mod_cast hK2
        linarith
      field_simp [hKm1]
      ring
    rw [hsingle]
    refine ⟨fun _ => by ring, (n * K : ℕ), ?_⟩
    push_cast; ring

/-- **Correction 1 to arXiv:2603.10054v1 (Double collider `D_4`)**:
With `n = 2, m = 2` (`K = 4`), the true root volume exponent is `(K - 1)/2 = 3/2`,
giving `E[1/U_i] = 16/3` and `⟨R⟩_{D_4} = 4 ∈ (1/2)ℤ` (refuting Proposition 5.2's `36/5`). -/
theorem doubleCollider_D4_avgCurvature :
    let K : ℕ := 2 * 2 ^ (2 - 1)
    K = 4 ∧
    betaIntegral (((K : ℝ) - 3) / 2) / betaIntegral (((K : ℝ) - 1) / 2) = 16 / 3 ∧
    (2 : ℝ) * ((K : ℝ) * (2 * K + 1) / 2 - ((K : ℝ) * ((K : ℝ) - 1) / 4) *
      (betaIntegral (((K : ℝ) - 3) / 2) / betaIntegral (((K : ℝ) - 1) / 2))) = 4 ∧
    (4 : ℝ) ≠ 36 / 5 := by
  dsimp only
  have hK : 2 * 2 ^ (2 - 1) = 4 := by norm_num
  rw [hK]
  have hrat : betaIntegral (((4 : ℕ) : ℝ) - 3) / 2 = betaIntegral (((4 : ℕ) : ℝ) - 3) / 2 := rfl
  have hratio := betaIntegral_ratio 4 (by norm_num)
  have hval : 4 * ((4 : ℕ) : ℝ) / (((4 : ℕ) : ℝ) - 1) = 16 / 3 := by norm_num
  rw [hval] at hratio
  refine ⟨rfl, hratio, ?_, by norm_num⟩
  rw [hratio]
  norm_num

/-- **Correction 2 to arXiv:2603.10054v1 (Three-node fork `Ẽ_3` / chain `L̃_3`)**:
`n = 1, m = 2` (`K = 2`) has `⟨R⟩ = 1 ≠ 5/2`. -/
theorem fork_E3_avgCurvature :
    let K : ℕ := 2 * 2 ^ (1 - 1)
    K = 2 ∧
    (1 : ℝ) * ((K : ℝ) * (2 * K + 1) / 2 - ((K : ℝ) * ((K : ℝ) - 1) / 4) *
      (betaIntegral (((K : ℝ) - 3) / 2) / betaIntegral (((K : ℝ) - 1) / 2))) = 1 ∧
    (1 : ℝ) ≠ 5 / 2 := by
  dsimp only
  have hK : 2 * 2 ^ (1 - 1) = 2 := by norm_num
  rw [hK]
  have hratio := betaIntegral_ratio 2 (by norm_num)
  refine ⟨rfl, ?_, by norm_num⟩
  rw [hratio]
  norm_num

/-- **Correction 3 to arXiv:2603.10054v1 (Five-parent collapsing star `C̃_6`)**:
`n = 5, m = 1` (`K = 16`) has `⟨R⟩ = 40 > 0` (refuting Theorem 3.4's claimed `-272`). -/
theorem collapsingStar_C6_avgCurvature :
    let K : ℕ := 1 * 2 ^ (5 - 1)
    K = 16 ∧
    (5 : ℝ) * ((K : ℝ) * (2 * K + 1) / 2 - ((K : ℝ) * ((K : ℝ) - 1) / 4) *
      (betaIntegral (((K : ℝ) - 3) / 2) / betaIntegral (((K : ℝ) - 1) / 2))) = 40 ∧
    (0 : ℝ) < 40 := by
  dsimp only
  have hK : 1 * 2 ^ (5 - 1) = 16 := by norm_num
  rw [hK]
  have hratio := betaIntegral_ratio 16 (by norm_num)
  refine ⟨rfl, ?_, by norm_num⟩
  rw [hratio]
  norm_num

end BitnetRicciQuant
