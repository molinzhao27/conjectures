import Mathlib

/-!
# Exact Single-Round KL Contraction for the Gaussian Hypercube
(Conjecture `conj:singleround`, arXiv:2606.00703)

We formalize in Lean 4 + Mathlib (zero `sorry`, standard axioms only) the three analytic and
information-theoretic cores of the exact proof that `η_KL(P_Y, P_{V|Y}) = r(s)`:

1. **ODE supersolution inequality and comparison**:
   - `sinh_ge_self`: For every `u ≥ 0`, `Real.sinh u ≥ u` (hence `Real.sinh (2 * u) ≥ 2 * u`).
   - `tanh_div_ge_one_sub_tanh_sq`: For every `u > 0`, `Real.tanh u / u ≥ 1 - (Real.tanh u)^2`.
   - `ode_supersolution_ineq`: For `0 < r < 1`, `k = r / (1 - r)`, `h(t) = Real.tanh (r * t)`,
     and every `t > 0`, `h` has derivative `r * (1 - h(t)^2)` and satisfies
     `r * (1 - h(t)^2) ≥ 1 - h(t)^2 - h(t) / (k * t)`.
   - `ode_comparison_le_tanh`: Any `m` continuous on `[0, ∞)` with `m 0 = 0` and differentiable on
     `(0, ∞)` satisfying `m'(t) ≤ 1 - m(t)^2 - m(t) / (k * t)` obeys `m(t) ≤ Real.tanh (r * t)`
     for all `t ≥ 0`.
   - `mgf_bound_of_deriv_le_tanh`: Integrating `L'(t) ≤ Real.tanh (r * t)` from `0` yields
     `L(t) ≤ (1 / r) * Real.log (Real.cosh (r * t))` for all `t ≥ 0`.
2. **Legendre duality for fair-Bernoulli KL divergence**:
   - `bernKL_eq_klDiv`: `D(Bern((1 + b)/2) ‖ Bern(1/2)) = ((1 + b)/2) log(1 + b) + ((1 - b)/2) log(1 - b)`.
   - `phi_le_bernKL`, `phi_attains_bernKL`: For `-1 < b < 1`, `z ↦ z * b - log (cosh z)` attains
     its global maximum `bernKL b` at `z = (1 / 2) * log ((1 + b) / (1 - b))`.
   - `bernKL_eq_iSup`: `bernKL b = ⨆ z : ℝ, (z * b - Real.log (Real.cosh z))`.
   - `mgf_bound_implies_sdpi_scalar`: If `D_in ≥ t * b - (1 / r) * log (cosh (r * t))` for all `t`,
     then `bernKL b ≤ r * D_in`.
3. **Finite-alphabet KL SDPI product tensorization bound**:
   - `log_sum_ineq`: Discrete log-sum inequality for nonnegative `a_i` and positive `b_i`.
   - `kl_chain_rule`: Exact chain rule for discrete KL divergence against a product reference law.
   - `kl_sdpi_prod_two`: Two-factor product tensorization of the KL strong data processing
     coefficient `η` for arbitrary joint input distributions `Q` on `α₁ × α₂`.
   - `kl_sdpi_prod_tensorization`: By induction on `d : ℕ`, if a finite-alphabet kernel `K`
     satisfies `KL(Q K ‖ P K) ≤ η * KL(Q ‖ P)` for all `Q`, then the `d`-fold product kernel
     `K^{⊗d}` satisfies `KL(Q K^{⊗d} ‖ P^{⊗d} K^{⊗d}) ≤ η * KL(Q ‖ P^{⊗d})` for every joint
     distribution `Q` on `Fin d → α`.

Machine-checked vs prose-only:
- Machine-checked: The complete ODE supersolution inequality (`sinh(2u) ≥ 2u ⇒ h' ≥ 1 - h² - h/(kt)`),
  ODE comparison and MGF integration, exact Legendre duality `bernKL b = sup_z (z b - log cosh z)`,
  the MGF-to-KL scalar SDPI deduction, and full finite-alphabet `d`-fold KL SDPI product
  tensorization via the log-sum inequality and chain rule.
- Prose-only (in `solutions/bit_constrained_kl/results.md`): The Gaussian shift integration-by-parts
  identity (1), Chebyshev negative-covariance step yielding the differential inequality (3) for
  the continuous Gaussian posterior `W = tanh X`, extension of tensorization from finite alphabets
  to continuous Gaussian observations, and the `χ²`/Gauss–Hermite numerical checks.
-/

open BigOperators Finset

namespace BitConstrainedKL

/-! ### 1. ODE Supersolution Inequality and Comparison -/

/-- `sinh u ≥ u` for all `u ≥ 0`. -/
theorem sinh_ge_self {u : ℝ} (hu : 0 ≤ u) : u ≤ Real.sinh u := by
  let f : ℝ → ℝ := fun x => Real.sinh x - x
  have hf : ∀ x, HasDerivAt f (Real.cosh x - 1) x :=
    fun x => (Real.hasDerivAt_sinh x).sub (hasDerivAt_id x)
  have hmono : Monotone f := by
    refine monotone_of_deriv_nonneg (fun x => (hf x).differentiableAt) fun x => ?_
    rw [(hf x).deriv]
    linarith [Real.one_le_cosh x]
  have h0 := hmono hu
  simp only [f, Real.sinh_zero, sub_zero] at h0
  linarith

/-- Derivative of `Real.tanh`: `HasDerivAt Real.tanh (1 - (Real.tanh x)^2) x`. -/
lemma hasDerivAt_tanh (x : ℝ) : HasDerivAt Real.tanh (1 - (Real.tanh x) ^ 2) x := by
  have hcosh_ne : Real.cosh x ≠ 0 := ne_of_gt (Real.cosh_pos x)
  have hdiv := (Real.hasDerivAt_sinh x).div (Real.hasDerivAt_cosh x) hcosh_ne
  have heq : Real.tanh = fun s => Real.sinh s / Real.cosh s := by
    funext s; exact Real.tanh_eq_sinh_div_cosh s
  rw [heq]
  convert hdiv using 1
  have hsq := Real.cosh_sq_sub_sinh_sq x
  field_simp [hcosh_ne]

/-- From `sinh(2u) ≥ 2u`, deduce `tanh(u) / u ≥ 1 - tanh(u)^2` for all `u > 0`. -/
theorem tanh_div_ge_one_sub_tanh_sq {u : ℝ} (hu : 0 < u) :
    1 - (Real.tanh u) ^ 2 ≤ Real.tanh u / u := by
  have h2u : 2 * u ≤ Real.sinh (2 * u) := sinh_ge_self (by linarith)
  have hdouble : Real.sinh (2 * u) = 2 * Real.sinh u * Real.cosh u := Real.sinh_two_mul u
  have hsc : u ≤ Real.sinh u * Real.cosh u := by linarith
  have hcosh_pos : 0 < Real.cosh u := Real.cosh_pos u
  have hcosh_ne : Real.cosh u ≠ 0 := ne_of_gt hcosh_pos
  have hsech : 1 - (Real.tanh u) ^ 2 = 1 / (Real.cosh u) ^ 2 := by
    rw [Real.tanh_eq_sinh_div_cosh]
    have hsq := Real.cosh_sq_sub_sinh_sq u
    field_simp [hcosh_ne]
    linarith
  rw [hsech, Real.tanh_eq_sinh_div_cosh]
  have hden1 : 0 < (Real.cosh u) ^ 2 := sq_pos_of_pos hcosh_pos
  have hden2 : 0 < Real.cosh u * u := mul_pos hcosh_pos hu
  rw [div_div, div_le_div_iff₀ hden1 hden2]
  nlinarith

/-- **ODE supersolution inequality**: for `0 < r < 1`, `k = r / (1 - r)`, `h(t) = tanh(r * t)`,
and every `t > 0`, `HasDerivAt h (r * (1 - h(t)^2)) t` and
`r * (1 - h(t)^2) ≥ 1 - h(t)^2 - h(t) / (k * t)`. -/
theorem ode_supersolution_ineq {r t : ℝ} (hr0 : 0 < r) (hr1 : r < 1) (ht : 0 < t) :
    let k : ℝ := r / (1 - r)
    let h : ℝ → ℝ := fun s => Real.tanh (r * s)
    HasDerivAt h (r * (1 - (h t) ^ 2)) t ∧
    1 - (h t) ^ 2 - h t / (k * t) ≤ r * (1 - (h t) ^ 2) := by
  intro k h
  refine ⟨?_, ?_⟩
  · have hlin : HasDerivAt (fun s => r * s) r t := by
      simpa using (hasDerivAt_id t).const_mul r
    have hcomp := (hasDerivAt_tanh (r * t)).comp t hlin
    convert hcomp using 1
    ring
  · have hrt : 0 < r * t := mul_pos hr0 ht
    have htanh_ineq := tanh_div_ge_one_sub_tanh_sq hrt
    have h1r : 0 < 1 - r := by linarith
    have hmul := mul_le_mul_of_nonneg_left htanh_ineq (le_of_lt h1r)
    have hkt : h t / (k * t) = (1 - r) * (Real.tanh (r * t) / (r * t)) := by
      dsimp only [h, k]
      have hr_ne : r ≠ 0 := ne_of_gt hr0
      have h1r_ne : 1 - r ≠ 0 := ne_of_gt h1r
      have ht_ne : t ≠ 0 := ne_of_gt ht
      field_simp [hr_ne, h1r_ne, ht_ne]
    rw [hkt]
    dsimp only [h]
    linarith

/-- **ODE comparison theorem**: if `m` is continuous on `[0, ∞)` with `m 0 = 0`, differentiable
on `(0, ∞)`, and satisfies `m'(t) ≤ 1 - m(t)^2 - m(t) / (k * t)` for all `t > 0` (where
`k = r / (1 - r)` and `0 < r < 1`), then `m(t) ≤ tanh(r * t)` for all `t ≥ 0`. -/
theorem ode_comparison_le_tanh {r : ℝ} (hr0 : 0 < r) (hr1 : r < 1)
    {m : ℝ → ℝ} (hm_cont : ContinuousOn m (Set.Ici 0)) (hm0 : m 0 = 0)
    (hm_diff : ∀ t > 0, DifferentiableAt ℝ m t)
    (hm_ode : ∀ t > 0, deriv m t ≤ 1 - (m t) ^ 2 - m t / ((r / (1 - r)) * t)) :
    ∀ t ≥ 0, m t ≤ Real.tanh (r * t) := by
  intro t₀ ht₀
  by_contra hgt
  push_neg at hgt
  let k : ℝ := r / (1 - r)
  have hk : 0 < k := div_pos hr0 (by linarith)
  let h : ℝ → ℝ := fun t => Real.tanh (r * t)
  let g : ℝ → ℝ := fun t => m t - h t
  have hg0 : g 0 = 0 := by simp [g, h, hm0, Real.tanh_zero]
  have ht₀_pos : 0 < t₀ := by
    rcases eq_or_lt_of_le ht₀ with rfl | hpos
    · simp only [hm0, mul_zero, Real.tanh_zero, lt_self_iff_false] at hgt
    · exact hpos
  have hh_diff : ∀ t, DifferentiableAt ℝ h t := fun t =>
    ((hasDerivAt_tanh (r * t)).comp t (by simpa using (hasDerivAt_id t).const_mul r)).differentiableAt
  have hh_cont : Continuous h := continuous_iff_continuousAt.2 fun t => (hh_diff t).continuousAt
  have hg_cont : ContinuousOn g (Set.Icc 0 t₀) :=
    (hm_cont.mono (fun x hx => hx.1)).sub hh_cont.continuousOn
  let S : Set ℝ := {t ∈ Set.Icc 0 t₀ | g t ≤ 0}
  have hS_ne : S.Nonempty := ⟨0, ⟨le_refl 0, le_of_lt ht₀_pos⟩, by simp [hg0]⟩
  have hS_bdd : BddAbove S := ⟨t₀, fun x hx => hx.1.2⟩
  have hS_closed : IsClosed S := by
    have h1 : S = Set.Icc 0 t₀ ∩ (Set.Icc 0 t₀ ∩ g ⁻¹' Set.Iic 0) := by
      ext x; simp [S]
    have hclosed_pre : IsClosed {x ∈ Set.Icc 0 t₀ | g x ≤ 0} :=
      hg_cont.preimage_isClosed_of_isClosed isClosed_Icc isClosed_Iic
    exact hclosed_pre
  let t₁ : ℝ := sSup S
  have ht₁_mem : t₁ ∈ S := hS_closed.csSup_mem hS_ne hS_bdd
  have ht₁_lt : t₁ < t₀ := by
    rcases eq_or_lt_of_le ht₁_mem.1.2 with heq | hlt
    · have : g t₀ ≤ 0 := heq ▸ ht₁_mem.2
      dsimp only [g, h] at this; linarith
    · exact hlt
  have hg_diff_oo : DifferentiableOn ℝ g (Set.Ioo t₁ t₀) := fun t ht => by
    have ht_pos : 0 < t := lt_of_le_of_lt ht₁_mem.1.1 ht.1
    exact ((hm_diff t ht_pos).sub (hh_diff t)).differentiableWithinAt
  have hg_cont_cc : ContinuousOn g (Set.Icc t₁ t₀) :=
    hg_cont.mono fun x hx => ⟨le_trans ht₁_mem.1.1 hx.1, hx.2⟩
  obtain ⟨c, hc, hslope⟩ := exists_deriv_eq_slope g ht₁_lt hg_cont_cc hg_diff_oo
  have hc_pos : 0 < c := lt_of_le_of_lt ht₁_mem.1.1 hc.1
  have hgc_pos : 0 < g c := by
    by_contra hle
    push_neg at hle
    have hc_mem : c ∈ S := ⟨⟨le_of_lt hc_pos, le_of_lt hc.2⟩, hle⟩
    have := le_csSup hS_bdd hc_mem
    linarith [hc.1]
  have hslope_pos : 0 < deriv g c := by
    rw [hslope]
    have hg_t₀_pos : 0 < g t₀ := by dsimp only [g, h]; linarith
    exact div_pos (by linarith [ht₁_mem.2]) (by linarith)
  have hsuper := ode_supersolution_ineq hr0 hr1 hc_pos
  have hdg : deriv g c = deriv m c - r * (1 - (h c) ^ 2) := by
    have hm_d := (hm_diff c hc_pos).hasDerivAt
    exact (hm_d.sub hsuper.1).deriv
  have hmc_gt : h c < m c := by dsimp only [g] at hgc_pos; linarith
  have hhc_pos : 0 < h c := by
    dsimp only [h]
    rw [Real.tanh_eq_sinh_div_cosh]
    exact div_pos (Real.sinh_pos_iff.2 (mul_pos hr0 hc_pos)) (Real.cosh_pos _)
  have hdg_le : deriv g c ≤ -((m c + h c + 1 / (k * c)) * g c) := by
    have h1 := hm_ode c hc_pos
    have h2 := hsuper.2
    rw [hdg]
    have halg : (1 - (m c) ^ 2 - m c / (k * c)) - (1 - (h c) ^ 2 - h c / (k * c)) =
        -((m c + h c + 1 / (k * c)) * (m c - h c)) := by
      have hkc : k * c ≠ 0 := ne_of_gt (mul_pos hk hc_pos)
      field_simp [hkc]
      ring
    dsimp only [g, k] at halg ⊢
    linarith
  have hcoeff_pos : 0 < m c + h c + 1 / (k * c) := by
    have : 0 < 1 / (k * c) := div_pos one_pos (mul_pos hk hc_pos)
    linarith
  have hneg : deriv g c < 0 := by nlinarith
  linarith

/-- Integrating `L'(t) ≤ tanh(r * t)` from `0` yields `L(t) ≤ (1 / r) * log(cosh(r * t))`. -/
theorem mgf_bound_of_deriv_le_tanh {r : ℝ} (hr0 : 0 < r)
    {L : ℝ → ℝ} (hL0 : L 0 = 0)
    (hL_cont : ContinuousOn L (Set.Ici 0))
    (hL_diff : ∀ t > 0, DifferentiableAt ℝ L t)
    (hL_le : ∀ t > 0, deriv L t ≤ Real.tanh (r * t)) :
    ∀ t ≥ 0, L t ≤ (1 / r) * Real.log (Real.cosh (r * t)) := by
  let B : ℝ → ℝ := fun t => (1 / r) * Real.log (Real.cosh (r * t))
  have hB_deriv : ∀ t, HasDerivAt B (Real.tanh (r * t)) t := by
    intro t
    have hlin : HasDerivAt (fun s => r * s) r t := by
      simpa using (hasDerivAt_id t).const_mul r
    have hcosh : HasDerivAt (fun s => Real.cosh (r * s)) (Real.sinh (r * t) * r) t :=
      (Real.hasDerivAt_cosh (r * t)).comp t hlin
    have hlog := hcosh.log (ne_of_gt (Real.cosh_pos (r * t)))
    convert hlog.const_mul (1 / r) using 1
    rw [Real.tanh_eq_sinh_div_cosh]
    field_simp [ne_of_gt hr0, ne_of_gt (Real.cosh_pos (r * t))]
  let ψ : ℝ → ℝ := fun t => L t - B t
  have hψ_cont : ContinuousOn ψ (Set.Ici 0) :=
    hL_cont.sub (continuous_iff_continuousAt.2 fun t => (hB_deriv t).continuousAt).continuousOn
  have hψ_diff : DifferentiableOn ℝ ψ (interior (Set.Ici 0)) := fun t ht => by
    have ht_pos : 0 < t := by simpa [interior_Ici] using ht
    exact ((hL_diff t ht_pos).sub (hB_deriv t).differentiableAt).differentiableWithinAt
  have hψ_anti : AntitoneOn ψ (Set.Ici 0) := by
    refine antitoneOn_of_deriv_nonpos (convex_Ici 0) hψ_cont hψ_diff fun t ht => ?_
    have ht_pos : 0 < t := by simpa [interior_Ici] using ht
    have hd : deriv ψ t = deriv L t - Real.tanh (r * t) :=
      ((hL_diff t ht_pos).hasDerivAt.sub (hB_deriv t)).deriv
    rw [hd]
    linarith [hL_le t ht_pos]
  intro t ht
  have hle := hψ_anti Set.left_mem_Ici ht ht
  have hψ0 : ψ 0 = 0 := by simp [ψ, B, hL0, Real.cosh_zero]
  dsimp only [ψ, B] at hle hψ0
  linarith

/-! ### 2. Legendre Duality for Fair-Bernoulli KL Divergence -/

/-- Binary KL divergence `D(Bern((1 + b)/2) ‖ Bern(1/2))` in nats. -/
noncomputable def bernKL (b : ℝ) : ℝ :=
  ((1 + b) / 2) * Real.log (1 + b) + ((1 - b) / 2) * Real.log (1 - b)

/-- Explicit equivalence with the 2-point KL formula `p log(p/(1/2)) + (1-p) log((1-p)/(1/2))`. -/
theorem bernKL_eq_klDiv (b : ℝ) :
    ((1 + b) / 2) * Real.log (((1 + b) / 2) / (1 / 2)) +
      ((1 - b) / 2) * Real.log (((1 - b) / 2) / (1 / 2)) = bernKL b := by
  unfold bernKL
  have h1 : ((1 + b) / 2) / (1 / 2 : ℝ) = 1 + b := by ring
  have h2 : ((1 - b) / 2) / (1 / 2 : ℝ) = 1 - b := by ring
  rw [h1, h2]

/-- Legendre objective `φ(b, z) = z * b - log (cosh z)`. -/
noncomputable def legendreObj (b z : ℝ) : ℝ :=
  z * b - Real.log (Real.cosh z)

/-- Explicit maximizer `z*(b) = (1 / 2) * log ((1 + b) / (1 - b))` (`arctanh b`). -/
noncomputable def atanhVal (b : ℝ) : ℝ :=
  (1 / 2) * Real.log ((1 + b) / (1 - b))

lemma exp_atanhVal {b : ℝ} (hb1 : -1 < b) (hb2 : b < 1) :
    Real.exp (atanhVal b) = Real.sqrt ((1 + b) / (1 - b)) := by
  unfold atanhVal
  have hpos : 0 < (1 + b) / (1 - b) := div_pos (by linarith) (by linarith)
  rw [Real.sqrt_eq_rpow, Real.rpow_def_of_pos hpos]
  congr 1; ring

lemma exp_neg_atanhVal {b : ℝ} (hb1 : -1 < b) (hb2 : b < 1) :
    Real.exp (-atanhVal b) = Real.sqrt ((1 - b) / (1 + b)) := by
  rw [Real.exp_neg, exp_atanhVal hb1 hb2]
  have hp1 : 0 < 1 + b := by linarith
  have hp2 : 0 < 1 - b := by linarith
  rw [Real.sqrt_div (le_of_lt hp1), Real.sqrt_div (le_of_lt hp2), inv_div]

lemma cosh_atanhVal {b : ℝ} (hb1 : -1 < b) (hb2 : b < 1) :
    Real.cosh (atanhVal b) = 1 / Real.sqrt (1 - b ^ 2) := by
  rw [Real.cosh_eq, exp_atanhVal hb1 hb2, exp_neg_atanhVal hb1 hb2]
  have hp1 : 0 < 1 + b := by linarith
  have hp2 : 0 < 1 - b := by linarith
  rw [Real.sqrt_div (le_of_lt hp1), Real.sqrt_div (le_of_lt hp2)]
  have hs1 : Real.sqrt (1 + b) ≠ 0 := by positivity
  have hs2 : Real.sqrt (1 - b) ≠ 0 := by positivity
  have hprod : Real.sqrt (1 - b ^ 2) = Real.sqrt (1 - b) * Real.sqrt (1 + b) := by
    rw [← Real.sqrt_mul (le_of_lt hp2)]
    congr 1; ring
  rw [hprod]
  field_simp [hs1, hs2]
  rw [Real.sq_sqrt (le_of_lt hp1), Real.sq_sqrt (le_of_lt hp2)]
  ring

lemma tanh_atanhVal {b : ℝ} (hb1 : -1 < b) (hb2 : b < 1) :
    Real.tanh (atanhVal b) = b := by
  rw [Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq,
    exp_atanhVal hb1 hb2, exp_neg_atanhVal hb1 hb2]
  have hp1 : 0 < 1 + b := by linarith
  have hp2 : 0 < 1 - b := by linarith
  rw [Real.sqrt_div (le_of_lt hp1), Real.sqrt_div (le_of_lt hp2)]
  have hs1 : Real.sqrt (1 + b) ≠ 0 := by positivity
  have hs2 : Real.sqrt (1 - b) ≠ 0 := by positivity
  have hden : Real.sqrt (1 + b) / Real.sqrt (1 - b) + Real.sqrt (1 - b) / Real.sqrt (1 + b) ≠ 0 := by
    positivity
  field_simp [hs1, hs2, hden]
  rw [Real.sq_sqrt (le_of_lt hp1), Real.sq_sqrt (le_of_lt hp2)]
  ring

/-- At `z = atanhVal b`, the Legendre objective equals `bernKL b`. -/
theorem legendreObj_attains_bernKL {b : ℝ} (hb1 : -1 < b) (hb2 : b < 1) :
    legendreObj b (atanhVal b) = bernKL b := by
  unfold legendreObj bernKL
  rw [cosh_atanhVal hb1 hb2]
  unfold atanhVal
  have hp1 : 0 < 1 + b := by linarith
  have hp2 : 0 < 1 - b := by linarith
  have h1b2 : 0 < 1 - b ^ 2 := by nlinarith
  rw [Real.log_div (ne_of_gt hp1) (ne_of_gt hp2), Real.log_div one_ne_zero (by positivity),
    Real.log_one, zero_sub, Real.log_sqrt (le_of_lt h1b2)]
  have hfac : 1 - b ^ 2 = (1 + b) * (1 - b) := by ring
  rw [hfac, Real.log_mul (ne_of_gt hp1) (ne_of_gt hp2)]
  ring

/-- Global upper bound: `z * b - log (cosh z) ≤ bernKL b` for all `z : ℝ` and `-1 < b < 1`. -/
theorem legendreObj_le_bernKL {b : ℝ} (hb1 : -1 < b) (hb2 : b < 1) (z : ℝ) :
    legendreObj b z ≤ bernKL b := by
  rw [← legendreObj_attains_bernKL hb1 hb2]
  have hderiv : ∀ x, HasDerivAt (legendreObj b) (b - Real.tanh x) x := by
    intro x
    have h1 : HasDerivAt (fun t => t * b) b x := by
      simpa using (hasDerivAt_id x).mul_const b
    have h2 : HasDerivAt (fun t => Real.log (Real.cosh t)) (Real.tanh x) x := by
      have := (Real.hasDerivAt_cosh x).log (ne_of_gt (Real.cosh_pos x))
      rwa [← Real.tanh_eq_sinh_div_cosh] at this
    exact h1.sub h2
  have htanh_mono : StrictMono Real.tanh := by
    refine strictMono_of_deriv_pos fun x => ?_
    rw [(hasDerivAt_tanh x).deriv]
    have hcosh_ne : Real.cosh x ≠ 0 := ne_of_gt (Real.cosh_pos x)
    have hsech : 1 - (Real.tanh x) ^ 2 = 1 / (Real.cosh x) ^ 2 := by
      rw [Real.tanh_eq_sinh_div_cosh]
      have hsq := Real.cosh_sq_sub_sinh_sq x
      field_simp [hcosh_ne]; linarith
    rw [hsech]
    positivity
  rcases le_total z (atanhVal b) with hle | hge
  · have hmono : MonotoneOn (legendreObj b) (Set.Iic (atanhVal b)) := by
      refine monotoneOn_of_deriv_nonneg (convex_Iic _)
        (fun x _ => (hderiv x).continuousAt.continuousWithinAt)
        (fun x _ => (hderiv x).differentiableAt.differentiableWithinAt) fun x hx => ?_
      rw [(hderiv x).deriv]
      have hxle : x ≤ atanhVal b := le_of_lt (by simpa [interior_Iic] using hx)
      have := htanh_mono.monotone hxle
      rw [tanh_atanhVal hb1 hb2] at this
      linarith
    exact hmono hle (Set.right_mem_Iic) hle
  · have hanti : AntitoneOn (legendreObj b) (Set.Ici (atanhVal b)) := by
      refine antitoneOn_of_deriv_nonpos (convex_Ici _)
        (fun x _ => (hderiv x).continuousAt.continuousWithinAt)
        (fun x _ => (hderiv x).differentiableAt.differentiableWithinAt) fun x hx => ?_
      rw [(hderiv x).deriv]
      have hxge : atanhVal b ≤ x := le_of_lt (by simpa [interior_Ici] using hx)
      have := htanh_mono.monotone hxge
      rw [tanh_atanhVal hb1 hb2] at this
      linarith
    exact hanti Set.left_mem_Ici hge hge

/-- **Legendre duality identity**: for every `-1 < b < 1`,
`D(Bern((1 + b)/2) ‖ Bern(1/2)) = ⨆ z : ℝ, (z * b - log (cosh z))`. -/
theorem bernKL_eq_iSup {b : ℝ} (hb1 : -1 < b) (hb2 : b < 1) :
    bernKL b = ⨆ z : ℝ, (z * b - Real.log (Real.cosh z)) := by
  symm
  refine ciSup_eq_of_forall_le_of_forall_lt_exists_gt
    (fun z => legendreObj_le_bernKL hb1 hb2 z) ?_
  intro w hw
  refine ⟨atanhVal b, ?_⟩
  rw [show atanhVal b * b - Real.log (Real.cosh (atanhVal b)) = legendreObj b (atanhVal b) from rfl,
    legendreObj_attains_bernKL hb1 hb2]
  exact hw

/-- **From MGF bound to scalar KL SDPI**: if `0 < r` and `D_in ≥ t * b - (1 / r) * log(cosh(r * t))`
for all `t : ℝ`, then `bernKL b ≤ r * D_in` for all `-1 < b < 1`. -/
theorem mgf_bound_implies_sdpi_scalar {r b D_in : ℝ} (hr0 : 0 < r) (hb1 : -1 < b) (hb2 : b < 1)
    (hD : ∀ t : ℝ, t * b - (1 / r) * Real.log (Real.cosh (r * t)) ≤ D_in) :
    bernKL b ≤ r * D_in := by
  have ht := hD (atanhVal b / r)
  have hrt : r * (atanhVal b / r) = atanhVal b := mul_div_cancel₀ _ (ne_of_gt hr0)
  rw [hrt] at ht
  have hmul := mul_le_mul_of_nonneg_left ht (le_of_lt hr0)
  have hsimp : r * (atanhVal b / r * b - 1 / r * Real.log (Real.cosh (atanhVal b))) =
      legendreObj b (atanhVal b) := by
    unfold legendreObj
    field_simp [ne_of_gt hr0]
  rw [hsimp, legendreObj_attains_bernKL hb1 hb2] at hmul
  exact hmul

/-! ### 3. Finite-Alphabet KL SDPI Product Tensorization Bound -/

/-- Discrete KL divergence on a finite type `γ`. Since `Real.log 0 = 0` in Mathlib, zero-mass
terms automatically vanish (`0 * Real.log (0 / q x) = 0`). -/
noncomputable def klFin {γ : Type*} [Fintype γ] (p q : γ → ℝ) : ℝ :=
  ∑ x : γ, p x * Real.log (p x / q x)

/-- Pushforward of a distribution `p` through a Markov kernel `K : α → β → ℝ`. -/
def pushforward {α β : Type*} [Fintype α] (K : α → β → ℝ) (p : α → ℝ) (y : β) : ℝ :=
  ∑ x : α, p x * K x y

/-- A finite-alphabet probability distribution on `γ`. -/
def IsProb {γ : Type*} [Fintype γ] (p : γ → ℝ) : Prop :=
  (∀ x, 0 ≤ p x) ∧ ∑ x : γ, p x = 1

/-- A finite-alphabet Markov kernel from `α` to `β`. -/
def IsKernel {α β : Type*} [Fintype β] (K : α → β → ℝ) : Prop :=
  (∀ x y, 0 ≤ K x y) ∧ ∀ x, ∑ y : β, K x y = 1

/-- Strong data processing inequality (SDPI) with constant `η` at reference input law `P`. -/
def HasKLSDPI {α β : Type*} [Fintype α] [Fintype β]
    (P : α → ℝ) (K : α → β → ℝ) (η : ℝ) : Prop :=
  ∀ Q : α → ℝ, IsProb Q → klFin (pushforward K Q) (pushforward K P) ≤ η * klFin Q P

/-- **Discrete log-sum inequality**: for `a_i ≥ 0` and `b_i > 0` on a finite type `ι` with
`0 < ∑ b_i`, `(∑ a_i) * log((∑ a_i) / (∑ b_i)) ≤ ∑ a_i * log(a_i / b_i)`. -/
theorem log_sum_ineq {ι : Type*} [Fintype ι] (a b : ι → ℝ)
    (ha : ∀ i, 0 ≤ a i) (hb : ∀ i, 0 < b i) (hB : 0 < ∑ i, b i) :
    (∑ i, a i) * Real.log ((∑ i, a i) / (∑ i, b i)) ≤ ∑ i, a i * Real.log (a i / b i) := by
  set A := ∑ i, a i
  set B := ∑ i, b i
  by_cases hA : A = 0
  · have ha0 : ∀ i, a i = 0 := fun i =>
      le_antisymm (by
        have hle := Finset.single_le_sum (fun j _ => ha j) (Finset.mem_univ i)
        linarith) (ha i)
    simp [hA, ha0]
  · have hA_pos : 0 < A := lt_of_le_of_ne (Finset.sum_nonneg fun i _ => ha i) (Ne.symm hA)
    have hterm : ∀ i, a i * Real.log (A / B) - a i * Real.log (a i / b i) ≤ (A / B) * b i - a i := by
      intro i
      by_cases hai : a i = 0
      · simp only [hai, zero_mul, sub_zero]
        exact mul_nonneg (div_nonneg (le_of_lt hA_pos) (le_of_lt hB)) (le_of_lt (hb i))
      · have hai_pos : 0 < a i := lt_of_le_of_ne (ha i) (Ne.symm hai)
        have hsub : Real.log (A / B) - Real.log (a i / b i) = Real.log ((A * b i) / (B * a i)) := by
          rw [← Real.log_div (ne_of_gt (div_pos hA_pos hB)) (ne_of_gt (div_pos hai_pos (hb i)))]
          congr 1; field_simp [ne_of_gt hB, ne_of_gt (hb i), ne_of_gt hai_pos]
        have hlog := Real.log_le_sub_one_of_pos (div_pos (mul_pos hA_pos (hb i)) (mul_pos hB hai_pos))
        have hmul := mul_le_mul_of_nonneg_left hlog (le_of_lt hai_pos)
        have hrhs : a i * ((A * b i) / (B * a i) - 1) = (A / B) * b i - a i := by
          field_simp [ne_of_gt hB, ne_of_gt hai_pos]
        calc a i * Real.log (A / B) - a i * Real.log (a i / b i)
            = a i * (Real.log (A / B) - Real.log (a i / b i)) := by ring
          _ = a i * Real.log ((A * b i) / (B * a i)) := by rw [hsub]
          _ ≤ a i * ((A * b i) / (B * a i) - 1) := hmul
          _ = (A / B) * b i - a i := hrhs
    have hsum := Finset.sum_le_sum fun i (_ : i ∈ Finset.univ) => hterm i
    rw [Finset.sum_sub_distrib, Finset.sum_sub_distrib, ← Finset.sum_mul, ← Finset.mul_sum] at hsum
    have hcancel : (A / B) * B - A = 0 := by field_simp [ne_of_gt hB]; ring
    linarith

/-- Product distribution on `α₁ × α₂`. -/
def prod2 {α₁ α₂ : Type*} (P₁ : α₁ → ℝ) (P₂ : α₂ → ℝ) (x : α₁ × α₂) : ℝ :=
  P₁ x.1 * P₂ x.2

/-- Product Markov kernel on `(α₁ × α₂) → (β₁ × β₂) → ℝ`. -/
def kernel2 {α₁ α₂ β₁ β₂ : Type*} (K₁ : α₁ → β₁ → ℝ) (K₂ : α₂ → β₂ → ℝ)
    (x : α₁ × α₂) (y : β₁ × β₂) : ℝ :=
  K₁ x.1 y.1 * K₂ x.2 y.2

lemma pushforward_kernel2_prod2 {α₁ α₂ β₁ β₂ : Type*} [Fintype α₁] [Fintype α₂]
    (K₁ : α₁ → β₁ → ℝ) (K₂ : α₂ → β₂ → ℝ) (P₁ : α₁ → ℝ) (P₂ : α₂ → ℝ) :
    pushforward (kernel2 K₁ K₂) (prod2 P₁ P₂) = prod2 (pushforward K₁ P₁) (pushforward K₂ P₂) := by
  funext ⟨y₁, y₂⟩
  unfold pushforward kernel2 prod2
  simp only [Fintype.sum_prod_type]
  calc ∑ x₁, ∑ x₂, (P₁ x₁ * P₂ x₂) * (K₁ x₁ y₁ * K₂ x₂ y₂)
      = ∑ x₁, (P₁ x₁ * K₁ x₁ y₁) * (∑ x₂, P₂ x₂ * K₂ x₂ y₂) := by
          refine Finset.sum_congr rfl fun x₁ _ => ?_
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun x₂ _ => by ring
    _ = (∑ x₁, P₁ x₁ * K₁ x₁ y₁) * (∑ x₂, P₂ x₂ * K₂ x₂ y₂) := by rw [← Finset.sum_mul]

/-- Exact chain rule for discrete KL divergence against a product reference distribution `P₁ ⊗ P₂`. -/
theorem kl_chain_rule {α₁ α₂ : Type*} [Fintype α₁] [Fintype α₂]
    (Q : α₁ × α₂ → ℝ) (P₁ : α₁ → ℝ) (P₂ : α₂ → ℝ)
    (hQ_nonneg : ∀ x, 0 ≤ Q x) (hP₁_pos : ∀ x₁, 0 < P₁ x₁) (hP₂_pos : ∀ x₂, 0 < P₂ x₂) :
    let Q₁ : α₁ → ℝ := fun x₁ => ∑ x₂, Q (x₁, x₂)
    let Qcond : α₁ → α₂ → ℝ := fun x₁ x₂ => Q (x₁, x₂) / Q₁ x₁
    klFin Q (prod2 P₁ P₂) = klFin Q₁ P₁ + ∑ x₁, Q₁ x₁ * klFin (Qcond x₁) P₂ := by
  intro Q₁ Qcond
  unfold klFin prod2
  rw [Fintype.sum_prod_type]
  have hinner : ∀ x₁ : α₁,
      ∑ x₂ : α₂, Q (x₁, x₂) * Real.log (Q (x₁, x₂) / (P₁ x₁ * P₂ x₂)) =
      Q₁ x₁ * Real.log (Q₁ x₁ / P₁ x₁) + Q₁ x₁ * ∑ x₂ : α₂, Qcond x₁ x₂ * Real.log (Qcond x₁ x₂ / P₂ x₂) := by
    intro x₁
    by_cases hq1 : Q₁ x₁ = 0
    · have hq0 : ∀ x₂, Q (x₁, x₂) = 0 := fun x₂ =>
        le_antisymm (by
          have hle := Finset.single_le_sum (fun y _ => hQ_nonneg (x₁, y)) (Finset.mem_univ x₂)
          linarith) (hQ_nonneg (x₁, x₂))
      simp [hq1, hq0]
    · have hq1_pos : 0 < Q₁ x₁ :=
        lt_of_le_of_ne (Finset.sum_nonneg fun x₂ _ => hQ_nonneg (x₁, x₂)) (Ne.symm hq1)
      have hterm : ∀ x₂,
          Q (x₁, x₂) * Real.log (Q (x₁, x₂) / (P₁ x₁ * P₂ x₂)) =
          Q (x₁, x₂) * Real.log (Q₁ x₁ / P₁ x₁) +
          Q₁ x₁ * (Qcond x₁ x₂ * Real.log (Qcond x₁ x₂ / P₂ x₂)) := by
        intro x₂
        by_cases hq : Q (x₁, x₂) = 0
        · simp [Qcond, hq]
        · have hq_pos : 0 < Q (x₁, x₂) := lt_of_le_of_ne (hQ_nonneg (x₁, x₂)) (Ne.symm hq)
          have hqc_pos : 0 < Qcond x₁ x₂ := div_pos hq_pos hq1_pos
          have hsplit : Q (x₁, x₂) / (P₁ x₁ * P₂ x₂) = (Q₁ x₁ / P₁ x₁) * (Qcond x₁ x₂ / P₂ x₂) := by
            dsimp only [Qcond]; field_simp [ne_of_gt hq1_pos, ne_of_gt (hP₁_pos x₁), ne_of_gt (hP₂_pos x₂)]
          rw [hsplit, Real.log_mul (ne_of_gt (div_pos hq1_pos (hP₁_pos x₁)))
            (ne_of_gt (div_pos hqc_pos (hP₂_pos x₂)))]
          have hcancel : Q₁ x₁ * Qcond x₁ x₂ = Q (x₁, x₂) := mul_div_cancel₀ _ (ne_of_gt hq1_pos)
          calc Q (x₁, x₂) * (Real.log (Q₁ x₁ / P₁ x₁) + Real.log (Qcond x₁ x₂ / P₂ x₂))
              = Q (x₁, x₂) * Real.log (Q₁ x₁ / P₁ x₁) + Q (x₁, x₂) * Real.log (Qcond x₁ x₂ / P₂ x₂) := by ring
            _ = Q (x₁, x₂) * Real.log (Q₁ x₁ / P₁ x₁) +
                  Q₁ x₁ * (Qcond x₁ x₂ * Real.log (Qcond x₁ x₂ / P₂ x₂)) := by
                rw [← hcancel]; ring
      simp_rw [hterm, Finset.sum_add_distrib, ← Finset.sum_mul, ← Finset.mul_sum]
      rfl
  simp_rw [hinner, Finset.sum_add_distrib]

/-- **Two-factor KL SDPI product tensorization**: if `K₁` has KL SDPI `η` at `P₁` and `K₂` has
KL SDPI `η` at `P₂`, then `K₁ ⊗ K₂` has KL SDPI `η` at `P₁ ⊗ P₂` for every joint law `Q`. -/
theorem kl_sdpi_prod_two {α₁ α₂ β₁ β₂ : Type*}
    [Fintype α₁] [Fintype α₂] [Fintype β₁] [Fintype β₂]
    (P₁ : α₁ → ℝ) (P₂ : α₂ → ℝ) (K₁ : α₁ → β₁ → ℝ) (K₂ : α₂ → β₂ → ℝ) {η : ℝ}
    (hη : 0 ≤ η)
    (hP₁_pos : ∀ x₁, 0 < P₁ x₁) (hP₂_pos : ∀ x₂, 0 < P₂ x₂)
    (hK₁ : IsKernel K₁) (hK₂ : IsKernel K₂)
    (hPY₁_pos : ∀ y₁, 0 < pushforward K₁ P₁ y₁) (hPY₂_pos : ∀ y₂, 0 < pushforward K₂ P₂ y₂)
    (hSDPI₁ : HasKLSDPI P₁ K₁ η) (hSDPI₂ : HasKLSDPI P₂ K₂ η) :
    HasKLSDPI (prod2 P₁ P₂) (kernel2 K₁ K₂) η := by
  intro Q hQ
  rw [pushforward_kernel2_prod2]
  let Q₁ : α₁ → ℝ := fun x₁ => ∑ x₂, Q (x₁, x₂)
  let QcondX : α₁ → α₂ → ℝ := fun x₁ x₂ => Q (x₁, x₂) / Q₁ x₁
  let QY : β₁ × β₂ → ℝ := pushforward (kernel2 K₁ K₂) Q
  let QY₁ : β₁ → ℝ := fun y₁ => ∑ y₂, QY (y₁, y₂)
  let QYcond : β₁ → β₂ → ℝ := fun y₁ y₂ => QY (y₁, y₂) / QY₁ y₁
  have hQY_nonneg : ∀ y, 0 ≤ QY y := fun y =>
    Finset.sum_nonneg fun x _ => mul_nonneg (hQ.1 x) (mul_nonneg (hK₁.1 x.1 y.1) (hK₂.1 x.2 y.2))
  -- Identify marginal QY₁ = pushforward K₁ Q₁
  have hQY₁_eq : QY₁ = pushforward K₁ Q₁ := by
    funext y₁
    dsimp only [QY₁, QY, pushforward, kernel2, Q₁]
    simp only [Fintype.sum_prod_type]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun x₁ _ => ?_
    rw [Finset.sum_comm]
    calc ∑ x₂, ∑ y₂, Q (x₁, x₂) * (K₁ x₁ y₁ * K₂ x₂ y₂)
        = ∑ x₂, (Q (x₁, x₂) * K₁ x₁ y₁) * (∑ y₂, K₂ x₂ y₂) := by
            refine Finset.sum_congr rfl fun x₂ _ => ?_
            rw [Finset.mul_sum]; refine Finset.sum_congr rfl fun y₂ _ => by ring
      _ = ∑ x₂, Q (x₁, x₂) * K₁ x₁ y₁ := by simp_rw [hK₂.2, mul_one]
      _ = (∑ x₂, Q (x₁, x₂)) * K₁ x₁ y₁ := by rw [← Finset.sum_mul]
  -- Define Q_{X₂ | Y₁ = y₁}
  let QX2_given_Y1 : β₁ → α₂ → ℝ := fun y₁ x₂ => (∑ x₁, Q (x₁, x₂) * K₁ x₁ y₁) / QY₁ y₁
  -- Step 1: Chain rule on output
  have hchain_out := kl_chain_rule QY (pushforward K₁ P₁) (pushforward K₂ P₂) hQY_nonneg hPY₁_pos hPY₂_pos
  -- Step 2: Chain rule on input
  have hchain_in := kl_chain_rule Q P₁ P₂ hQ.1 hP₁_pos hP₂_pos
  -- Step 3: Q₁ is a probability distribution, so SDPI₁ applies to QY₁
  have hQ₁_prob : IsProb Q₁ := by
    refine ⟨fun x₁ => Finset.sum_nonneg fun x₂ _ => hQ.1 (x₁, x₂), ?_⟩
    simpa [Q₁, Fintype.sum_prod_type] using hQ.2
  have hbound₁ : klFin QY₁ (pushforward K₁ P₁) ≤ η * klFin Q₁ P₁ := by
    rw [hQY₁_eq]; exact hSDPI₁ Q₁ hQ₁_prob
  -- Step 4: For each y₁, QY₁ y₁ * klFin (QYcond y₁) (pushforward K₂ P₂) ≤ η * QY₁ y₁ * klFin (QX2_given_Y1 y₁) P₂
  have hbound₂_y₁ : ∀ y₁ : β₁,
      QY₁ y₁ * klFin (QYcond y₁) (pushforward K₂ P₂) ≤
      η * (QY₁ y₁ * klFin (QX2_given_Y1 y₁) P₂) := by
    intro y₁
    by_cases hy₁ : QY₁ y₁ = 0
    · simp [hy₁]
    · have hy₁_pos : 0 < QY₁ y₁ :=
        lt_of_le_of_ne (Finset.sum_nonneg fun y₂ _ => hQY_nonneg (y₁, y₂)) (Ne.symm hy₁)
      have hQX2_prob : IsProb (QX2_given_Y1 y₁) := by
        refine ⟨fun x₂ => div_nonneg (Finset.sum_nonneg fun x₁ _ => mul_nonneg (hQ.1 (x₁, x₂)) (hK₁.1 x₁ y₁)) (le_of_lt hy₁_pos), ?_⟩
        dsimp only [QX2_given_Y1]
        rw [← Finset.sum_div, Finset.sum_comm]
        have hnum : ∑ x₁, ∑ x₂, Q (x₁, x₂) * K₁ x₁ y₁ = QY₁ y₁ := by
          rw [hQY₁_eq]; dsimp only [pushforward, Q₁]
          refine Finset.sum_congr rfl fun x₁ _ => by rw [← Finset.sum_mul]
        rw [hnum, div_self (ne_of_gt hy₁_pos)]
      have hpush_eq : pushforward K₂ (QX2_given_Y1 y₁) = QYcond y₁ := by
        funext y₂
        dsimp only [pushforward, QX2_given_Y1, QYcond, QY, kernel2]
        simp only [Fintype.sum_prod_type]
        calc ∑ x₂, (∑ x₁, Q (x₁, x₂) * K₁ x₁ y₁) / QY₁ y₁ * K₂ x₂ y₂
            = (∑ x₂, ∑ x₁, Q (x₁, x₂) * (K₁ x₁ y₁ * K₂ x₂ y₂)) / QY₁ y₁ := by
                rw [Finset.sum_div]
                refine Finset.sum_congr rfl fun x₂ _ => ?_
                rw [div_mul_eq_mul_div, Finset.sum_mul]
                congr 1; refine Finset.sum_congr rfl fun x₁ _ => by ring
          _ = (∑ x₁, ∑ x₂, Q (x₁, x₂) * (K₁ x₁ y₁ * K₂ x₂ y₂)) / QY₁ y₁ := by
                rw [Finset.sum_comm]
      have hsdpi := hSDPI₂ (QX2_given_Y1 y₁) hQX2_prob
      rw [hpush_eq] at hsdpi
      have hmul := mul_le_mul_of_nonneg_left hsdpi (le_of_lt hy₁_pos)
      linarith
  -- Step 5: Convexity (log-sum inequality) gives
  -- ∑ y₁, QY₁ y₁ * klFin (QX2_given_Y1 y₁) P₂ ≤ ∑ x₁, Q₁ x₁ * klFin (QcondX x₁) P₂
  have hconvex_y₁ : ∀ y₁ : β₁,
      QY₁ y₁ * klFin (QX2_given_Y1 y₁) P₂ ≤
      ∑ x₁ : α₁, Q₁ x₁ * K₁ x₁ y₁ * klFin (QcondX x₁) P₂ := by
    intro y₁
    by_cases hy₁ : QY₁ y₁ = 0
    · rw [hy₁, zero_mul]
      refine Finset.sum_nonneg fun x₁ _ => ?_
      have hterm0 : Q₁ x₁ * K₁ x₁ y₁ = 0 := by
        have hsum0 : ∑ x : α₁, Q₁ x * K₁ x y₁ = 0 := by
          have := congrFun hQY₁_eq y₁; simpa [pushforward, hy₁] using this.symm
        exact le_antisymm (by
          have hle := Finset.single_le_sum (fun x _ => mul_nonneg (hQ₁_prob.1 x) (hK₁.1 x y₁)) (Finset.mem_univ x₁)
          linarith) (mul_nonneg (hQ₁_prob.1 x₁) (hK₁.1 x₁ y₁))
      simp [hterm0]
    · have hy₁_pos : 0 < QY₁ y₁ :=
        lt_of_le_of_ne (Finset.sum_nonneg fun y₂ _ => hQY_nonneg (y₁, y₂)) (Ne.symm hy₁)
      -- Restrict the log-sum inequality to the support S = {x₁ | 0 < Q₁ x₁ * K₁ x₁ y₁}
      let S : Finset α₁ := Finset.univ.filter (fun x₁ => 0 < Q₁ x₁ * K₁ x₁ y₁)
      have houtside : ∀ x₁, x₁ ∉ S → Q₁ x₁ * K₁ x₁ y₁ = 0 := by
        intro x₁ hx₁
        simp only [S, Finset.mem_filter, Finset.mem_univ, true_and, not_lt] at hx₁
        exact le_antisymm hx₁ (mul_nonneg (hQ₁_prob.1 x₁) (hK₁.1 x₁ y₁))
      have houtside_Q : ∀ x₁ x₂, x₁ ∉ S → Q (x₁, x₂) * K₁ x₁ y₁ = 0 := by
        intro x₁ x₂ hx₁
        have h0 := houtside x₁ hx₁
        rcases mul_eq_zero.1 h0 with hq1 | hk1
        · have hq : Q (x₁, x₂) = 0 := le_antisymm (by
            have hle := Finset.single_le_sum (fun y _ => hQ.1 (x₁, y)) (Finset.mem_univ x₂)
            linarith) (hQ.1 (x₁, x₂))
          simp [hq]
        · simp [hk1]
      have hsum_S : ∑ x₁ ∈ S, Q₁ x₁ * K₁ x₁ y₁ = QY₁ y₁ := by
        have := congrFun hQY₁_eq y₁
        rw [this]; dsimp only [pushforward]
        exact Finset.sum_subset (Finset.subset_univ S) fun x₁ _ hx₁ => houtside x₁ hx₁
      have hsum_num : ∀ x₂ : α₂, ∑ x₁ ∈ S, Q (x₁, x₂) * K₁ x₁ y₁ = ∑ x₁, Q (x₁, x₂) * K₁ x₁ y₁ := by
        intro x₂
        exact Finset.sum_subset (Finset.subset_univ S) fun x₁ _ hx₁ => houtside_Q x₁ x₂ hx₁
      have hper_x₂ : ∀ x₂ : α₂,
          QY₁ y₁ * (QX2_given_Y1 y₁ x₂ * Real.log (QX2_given_Y1 y₁ x₂ / P₂ x₂)) ≤
          ∑ x₁ : α₁, Q₁ x₁ * K₁ x₁ y₁ * (QcondX x₁ x₂ * Real.log (QcondX x₁ x₂ / P₂ x₂)) := by
        intro x₂
        let a : S → ℝ := fun x₁ => Q (x₁.1, x₂) * K₁ x₁.1 y₁
        let b : S → ℝ := fun x₁ => Q₁ x₁.1 * K₁ x₁.1 y₁ * P₂ x₂
        have ha : ∀ x₁ : S, 0 ≤ a x₁ := fun x₁ => mul_nonneg (hQ.1 (x₁.1, x₂)) (hK₁.1 x₁.1 y₁)
        have hb : ∀ x₁ : S, 0 < b x₁ := fun x₁ => by
          have hx₁ := (Finset.mem_filter.1 x₁.2).2
          exact mul_pos hx₁ (hP₂_pos x₂)
        have hB_eq : ∑ x₁ : S, b x₁ = QY₁ y₁ * P₂ x₂ := by
          dsimp only [b]
          rw [← Finset.sum_mul, Finset.sum_coe_sort S (fun x₁ => Q₁ x₁ * K₁ x₁ y₁), hsum_S]
        have hB_pos : 0 < ∑ x₁ : S, b x₁ := by
          rw [hB_eq]; exact mul_pos hy₁_pos (hP₂_pos x₂)
        have hA_eq : ∑ x₁ : S, a x₁ = ∑ x₁, Q (x₁, x₂) * K₁ x₁ y₁ := by
          dsimp only [a]
          rw [Finset.sum_coe_sort S (fun x₁ => Q (x₁, x₂) * K₁ x₁ y₁), hsum_num x₂]
        have hls := log_sum_ineq a b ha hb hB_pos
        rw [hA_eq, hB_eq] at hls
        have hlhs : QY₁ y₁ * (QX2_given_Y1 y₁ x₂ * Real.log (QX2_given_Y1 y₁ x₂ / P₂ x₂)) =
            (∑ x₁, Q (x₁, x₂) * K₁ x₁ y₁) * Real.log ((∑ x₁, Q (x₁, x₂) * K₁ x₁ y₁) / (QY₁ y₁ * P₂ x₂)) := by
          dsimp only [QX2_given_Y1]
          rw [← mul_assoc, mul_div_cancel₀ _ (ne_of_gt hy₁_pos), div_div]
        have hrhs : ∑ x₁ : S, a x₁ * Real.log (a x₁ / b x₁) =
            ∑ x₁ : α₁, Q₁ x₁ * K₁ x₁ y₁ * (QcondX x₁ x₂ * Real.log (QcondX x₁ x₂ / P₂ x₂)) := by
          rw [Finset.sum_coe_sort S (fun x₁ => (Q (x₁, x₂) * K₁ x₁ y₁) *
            Real.log ((Q (x₁, x₂) * K₁ x₁ y₁) / (Q₁ x₁ * K₁ x₁ y₁ * P₂ x₂)))]
          have hcongr : ∑ x₁ ∈ S, (Q (x₁, x₂) * K₁ x₁ y₁) *
                Real.log ((Q (x₁, x₂) * K₁ x₁ y₁) / (Q₁ x₁ * K₁ x₁ y₁ * P₂ x₂)) =
              ∑ x₁ ∈ S, Q₁ x₁ * K₁ x₁ y₁ * (QcondX x₁ x₂ * Real.log (QcondX x₁ x₂ / P₂ x₂)) := by
            refine Finset.sum_congr rfl fun x₁ hx₁ => ?_
            have hpos := (Finset.mem_filter.1 hx₁).2
            have hq1_ne : Q₁ x₁ ≠ 0 := fun h => by rw [h, zero_mul] at hpos; exact lt_irrefl 0 hpos
            have hk1_ne : K₁ x₁ y₁ ≠ 0 := fun h => by rw [h, mul_zero] at hpos; exact lt_irrefl 0 hpos
            have hfrac : (Q (x₁, x₂) * K₁ x₁ y₁) / (Q₁ x₁ * K₁ x₁ y₁ * P₂ x₂) = QcondX x₁ x₂ / P₂ x₂ := by
              dsimp only [QcondX]; field_simp [hq1_ne, hk1_ne, ne_of_gt (hP₂_pos x₂)]
            have hcoef : Q₁ x₁ * K₁ x₁ y₁ * QcondX x₁ x₂ = Q (x₁, x₂) * K₁ x₁ y₁ := by
              dsimp only [QcondX]; field_simp [hq1_ne]
            rw [hfrac, ← mul_assoc, hcoef]
          rw [hcongr]
          exact Finset.sum_subset (Finset.subset_univ S) fun x₁ _ hx₁ => by
            rw [houtside x₁ hx₁, zero_mul]
        rw [hlhs, ← hrhs]
        exact hls
      calc QY₁ y₁ * klFin (QX2_given_Y1 y₁) P₂
          = ∑ x₂ : α₂, QY₁ y₁ * (QX2_given_Y1 y₁ x₂ * Real.log (QX2_given_Y1 y₁ x₂ / P₂ x₂)) := by
              unfold klFin; rw [Finset.mul_sum]
        _ ≤ ∑ x₂ : α₂, ∑ x₁ : α₁, Q₁ x₁ * K₁ x₁ y₁ * (QcondX x₁ x₂ * Real.log (QcondX x₁ x₂ / P₂ x₂)) :=
              Finset.sum_le_sum fun x₂ _ => hper_x₂ x₂
        _ = ∑ x₁ : α₁, Q₁ x₁ * K₁ x₁ y₁ * klFin (QcondX x₁) P₂ := by
              rw [Finset.sum_comm]
              refine Finset.sum_congr rfl fun x₁ _ => by unfold klFin; rw [Finset.mul_sum]
  -- Combine all steps
  have hsum_y₁ : ∑ y₁ : β₁, QY₁ y₁ * klFin (QYcond y₁) (pushforward K₂ P₂) ≤
      η * ∑ x₁ : α₁, Q₁ x₁ * klFin (QcondX x₁) P₂ := by
    calc ∑ y₁ : β₁, QY₁ y₁ * klFin (QYcond y₁) (pushforward K₂ P₂)
        ≤ ∑ y₁ : β₁, η * (QY₁ y₁ * klFin (QX2_given_Y1 y₁) P₂) :=
            Finset.sum_le_sum fun y₁ _ => hbound₂_y₁ y₁
      _ = η * ∑ y₁ : β₁, QY₁ y₁ * klFin (QX2_given_Y1 y₁) P₂ := by rw [← Finset.mul_sum]
      _ ≤ η * ∑ y₁ : β₁, ∑ x₁ : α₁, Q₁ x₁ * K₁ x₁ y₁ * klFin (QcondX x₁) P₂ :=
            mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun y₁ _ => hconvex_y₁ y₁) hη
      _ = η * ∑ x₁ : α₁, Q₁ x₁ * klFin (QcondX x₁) P₂ := by
            congr 1
            rw [Finset.sum_comm]
            refine Finset.sum_congr rfl fun x₁ _ => ?_
            calc ∑ y₁ : β₁, Q₁ x₁ * K₁ x₁ y₁ * klFin (QcondX x₁) P₂
                = Q₁ x₁ * (∑ y₁ : β₁, K₁ x₁ y₁) * klFin (QcondX x₁) P₂ := by
                    rw [Finset.mul_sum, Finset.sum_mul]
              _ = Q₁ x₁ * klFin (QcondX x₁) P₂ := by rw [hK₁.2 x₁, mul_one]
  dsimp only [QY, QY₁, QYcond] at hchain_out hbound₁ hsum_y₁
  dsimp only [Q₁, QcondX] at hchain_in hbound₁ hsum_y₁
  rw [hchain_out, hchain_in, mul_add]
  linarith

/-- `d`-fold product distribution `P^{⊗d}` on `Fin d → α`. -/
def prodDist {α : Type*} (P : α → ℝ) (d : ℕ) (x : Fin d → α) : ℝ :=
  ∏ i : Fin d, P (x i)

/-- `d`-fold product Markov kernel `K^{⊗d}` on `(Fin d → α) → (Fin d → β) → ℝ`. -/
def prodKernel {α β : Type*} (K : α → β → ℝ) (d : ℕ) (x : Fin d → α) (y : Fin d → β) : ℝ :=
  ∏ i : Fin d, K (x i) (y i)

lemma prodDist_pos {α : Type*} {P : α → ℝ} (hP : ∀ x, 0 < P x) (d : ℕ) (x : Fin d → α) :
    0 < prodDist P d x :=
  Finset.prod_pos fun i _ => hP (x i)

lemma isKernel_prodKernel {α β : Type*} [Fintype β] {K : α → β → ℝ} (hK : IsKernel K) :
    ∀ d : ℕ, IsKernel (prodKernel K d)
  | 0 => ⟨fun _ _ => by simp [prodKernel], fun _ => by simp [prodKernel]⟩
  | d + 1 => by
      refine ⟨fun x y => Finset.prod_nonneg fun i _ => hK.1 (x i) (y i), fun x => ?_⟩
      let e : (β × (Fin d → β)) ≃ (Fin (d + 1) → β) :=
        { toFun := fun p => Fin.cons p.1 p.2
          invFun := fun y => (y 0, Fin.tail y)
          left_inv := fun ⟨y0, yt⟩ => by simp
          right_inv := fun y => Fin.cons_self_tail y }
      rw [← Equiv.sum_comp e]
      simp only [e, Equiv.coe_fn_mk, prodKernel, Fin.prod_univ_succ, Fin.cons_zero, Fin.cons_succ,
        Fintype.sum_prod_type]
      have hih := (isKernel_prodKernel hK d).2 (Fin.tail x)
      calc ∑ y0 : β, ∑ yt : Fin d → β, K (x 0) y0 * ∏ i : Fin d, K (x i.succ) (yt i)
          = (∑ y0 : β, K (x 0) y0) * (∑ yt : Fin d → β, prodKernel K d (Fin.tail x) yt) := by
              rw [Finset.sum_mul]
              refine Finset.sum_congr rfl fun y0 _ => by
                rw [Finset.mul_sum]; rfl
        _ = 1 * 1 := by rw [hK.2 (x 0), hih]
        _ = 1 := one_mul 1

lemma pushforward_prodKernel_prodDist {α β : Type*} [Fintype α]
    (K : α → β → ℝ) (P : α → ℝ) :
    ∀ d : ℕ, pushforward (prodKernel K d) (prodDist P d) = prodDist (pushforward K P) d
  | 0 => by
      funext y
      simp [pushforward, prodKernel, prodDist]
  | d + 1 => by
      funext y
      let e : (α × (Fin d → α)) ≃ (Fin (d + 1) → α) :=
        { toFun := fun p => Fin.cons p.1 p.2
          invFun := fun x => (x 0, Fin.tail x)
          left_inv := fun ⟨x0, xt⟩ => by simp
          right_inv := fun x => Fin.cons_self_tail x }
      unfold pushforward
      rw [← Equiv.sum_comp e]
      simp only [e, Equiv.coe_fn_mk, prodDist, prodKernel, Fin.prod_univ_succ, Fin.cons_zero,
        Fin.cons_succ, Fintype.sum_prod_type]
      have hih := congrFun (pushforward_prodKernel_prodDist K P d) (Fin.tail y)
      dsimp only [pushforward, prodDist, prodKernel, Fin.tail] at hih
      calc ∑ x0 : α, ∑ xt : Fin d → α,
            (P x0 * ∏ i : Fin d, P (xt i)) * (K x0 (y 0) * ∏ i : Fin d, K (xt i) (y i.succ))
          = (∑ x0 : α, P x0 * K x0 (y 0)) *
              (∑ xt : Fin d → α, (∏ i : Fin d, P (xt i)) * ∏ i : Fin d, K (xt i) (y i.succ)) := by
              rw [Finset.sum_mul]
              refine Finset.sum_congr rfl fun x0 _ => ?_
              rw [Finset.mul_sum]
              refine Finset.sum_congr rfl fun xt _ => by ring
        _ = (∑ x0 : α, P x0 * K x0 (y 0)) * ∏ i : Fin d, pushforward K P (y i.succ) := by
              rw [hih]; rfl

/-- **Finite-alphabet `d`-fold KL SDPI product tensorization**: if `K` has KL SDPI constant `η ≥ 0`
at positive input law `P` (with positive output marginal `P K`), then for every dimension `d : ℕ`
and every joint distribution `Q` on `Fin d → α`,
`KL(Q K^{⊗d} ‖ P^{⊗d} K^{⊗d}) ≤ η * KL(Q ‖ P^{⊗d})`. -/
theorem kl_sdpi_prod_tensorization {α β : Type*} [Fintype α] [Fintype β]
    (P : α → ℝ) (K : α → β → ℝ) {η : ℝ}
    (hη : 0 ≤ η) (hP_pos : ∀ x, 0 < P x) (hK : IsKernel K)
    (hPY_pos : ∀ y, 0 < pushforward K P y) (hSDPI : HasKLSDPI P K η) :
    ∀ d : ℕ, HasKLSDPI (prodDist P d) (prodKernel K d) η
  | 0 => by
      intro Q hQ
      have hQ0 : Q default = 1 := by
        simpa [Fintype.sum_unique] using hQ.2
      have hP0 : prodDist P 0 default = 1 := by simp [prodDist]
      have hpushQ : pushforward (prodKernel K 0) Q default = 1 := by
        simp [pushforward, prodKernel, hQ0]
      have hpushP : pushforward (prodKernel K 0) (prodDist P 0) default = 1 := by
        simp [pushforward, prodKernel, prodDist]
      simp [klFin, hQ0, hP0, hpushQ, hpushP]
  | d + 1 => by
      have hih := kl_sdpi_prod_tensorization P K hη hP_pos hK hPY_pos hSDPI d
      have hPYd_pos : ∀ y : Fin d → β, 0 < pushforward (prodKernel K d) (prodDist P d) y := by
        intro y
        rw [pushforward_prodKernel_prodDist]
        exact prodDist_pos hPY_pos d y
      have htwo := kl_sdpi_prod_two P (prodDist P d) K (prodKernel K d) hη
        hP_pos (prodDist_pos hP_pos d) hK (isKernel_prodKernel hK d) hPY_pos hPYd_pos hSDPI hih
      intro Q hQ
      let eα : (α × (Fin d → α)) ≃ (Fin (d + 1) → α) :=
        { toFun := fun p => Fin.cons p.1 p.2
          invFun := fun x => (x 0, Fin.tail x)
          left_inv := fun ⟨x0, xt⟩ => by simp
          right_inv := fun x => Fin.cons_self_tail x }
      let eβ : (β × (Fin d → β)) ≃ (Fin (d + 1) → β) :=
        { toFun := fun p => Fin.cons p.1 p.2
          invFun := fun y => (y 0, Fin.tail y)
          left_inv := fun ⟨y0, yt⟩ => by simp
          right_inv := fun y => Fin.cons_self_tail y }
      let Q' : α × (Fin d → α) → ℝ := fun p => Q (eα p)
      have hQ'_prob : IsProb Q' :=
        ⟨fun p => hQ.1 (eα p), by simpa [Q'] using (Equiv.sum_comp eα Q).trans hQ.2⟩
      have hbound := htwo Q' hQ'_prob
      have hkl_in : klFin Q' (prod2 P (prodDist P d)) = klFin Q (prodDist P (d + 1)) := by
        unfold klFin
        rw [← Equiv.sum_comp eα]
        refine Finset.sum_congr rfl fun p _ => ?_
        have hPeq : prod2 P (prodDist P d) p = prodDist P (d + 1) (eα p) := by
          simp [prod2, prodDist, eα, Fin.prod_univ_succ]
        rw [hPeq]
      have hpush_eq : ∀ q : β × (Fin d → β),
          pushforward (kernel2 K (prodKernel K d)) Q' q =
          pushforward (prodKernel K (d + 1)) Q (eβ q) := by
        intro q
        unfold pushforward
        rw [← Equiv.sum_comp eα]
        refine Finset.sum_congr rfl fun p _ => ?_
        have hKeq : kernel2 K (prodKernel K d) p q = prodKernel K (d + 1) (eα p) (eβ q) := by
          simp [kernel2, prodKernel, eα, eβ, Fin.prod_univ_succ]
        rw [hKeq]
      have hpushP_eq : ∀ q : β × (Fin d → β),
          pushforward (kernel2 K (prodKernel K d)) (prod2 P (prodDist P d)) q =
          pushforward (prodKernel K (d + 1)) (prodDist P (d + 1)) (eβ q) := by
        intro q
        rw [pushforward_kernel2_prod2, pushforward_prodKernel_prodDist,
          pushforward_prodKernel_prodDist]
        simp [prod2, prodDist, eβ, Fin.prod_univ_succ]
      have hkl_out : klFin (pushforward (kernel2 K (prodKernel K d)) Q')
            (pushforward (kernel2 K (prodKernel K d)) (prod2 P (prodDist P d))) =
          klFin (pushforward (prodKernel K (d + 1)) Q)
            (pushforward (prodKernel K (d + 1)) (prodDist P (d + 1))) := by
        unfold klFin
        rw [← Equiv.sum_comp eβ]
        refine Finset.sum_congr rfl fun q _ => by rw [hpush_eq q, hpushP_eq q]
      rwa [hkl_in, hkl_out] at hbound

end BitConstrainedKL
