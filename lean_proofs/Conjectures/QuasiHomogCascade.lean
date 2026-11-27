import Mathlib

/-!
# Refutation of the Cascading Minimization Conjecture for Quasi-Homogeneous Networks
(arXiv:2210.03820, ICLR 2023, Conjecture `conj:cascading`)

We formalize in Lean 4 + Mathlib the exact two-parameter counterexample refuting the
Cascading Minimization Conjecture of Daniel Kunin, Atsushi Yamamura, Chao Ma, and
Surya Ganguli, *The Asymmetric Maximum Margin Bias of Quasi-Homogeneous Neural Networks*
(ICLR 2023, arXiv:2210.03820), under both the intended gradient-descent sign and the
literal printed plus sign in Assumption A4:

1. **Model and unique quasi-homogeneous scaling rates (Theorem 1, Step 1)**:
   - `model x (a, b) = x₁ a b + x₂ a³` on training data `(e₁, +1), (-e₂, -1)` with
     signed margins `q₁(a, b) = a b` and `q₂(a, b) = a³`.
   - `quasi_homog_unique_rates`: `model x (ψ_α(a, b)) = e^α model x (a, b)` for all `x, a, b, α`
     if and only if `Λ = diag(1/3, 2/3)`.
2. **Primal feasibility obstruction (Proposition 1)**:
   - `cascade_infeasible_of_q1_nonpos`: If `q₁(θ̄) ≤ 0`, then for EVERY `α : ℝ` and EVERY
     cascade threshold `λ̃ : ℝ`, `ψ_α(θ̄)` violates `q₁ ≥ 1` and cannot be a feasible KKT point.
3. **Gradient-descent dynamics, cancellation invariant, `b/a → 1`, and normalized limit `(√3, 0)` (Theorem 1, Steps 2–4)**:
   - `expLoss_neg_grad`: The ODE `ȧ = (b e^{-ab} + 3a² e^{-a³})/2`, `ḃ = a e^{-ab}/2`
     is literally `-∇ L(a, b)` for `L(a, b) = (e^{-ab} + e^{-a³})/2`.
   - `descent_diff_sq_invariant`: Exact cancellation `d/dt (a² - b²) = 3 a³ e^{-a³} > 0`, giving
     `b(t)² ≤ a(t)² + b(0)² - a(0)²` for all `t ≥ 0`.
   - `descent_diverge_and_loss_zero`: `a(t) → +∞` and `L(a(t), b(t)) → 0`
     (verifying Assumption A5 for every positive start).
   - `descent_ratio_tendsto_one`: `D(t) = a(t)² - b(t)²` stays bounded above (proved via the
     elementary Lyapunov comparison `d/dt (D(t) + K / a(t)) ≤ 0`), yielding `b(t)/a(t) → 1`.
   - `descent_normalized_limit_and_refutation`: The `Λ`-normalized iterate
     `(â(t), b̂(t)) = (a(t) r(t), b(t) r(t)²)` with `â²/3 + 2 b̂²/3 = 1` converges to
     `θ̄ = (√3, 0)` (verifying A6), at which `q₁(ψ_α(√3, 0)) = 0 < 1` for every `α : ℝ`.
4. **Literal-A4 sign counterexample (Theorem 2)**:
   - `literal_A4_sign_refutation`: On `{(e₁, +1), (-e₁, -1)}` (`q₁ = q₂ = ab`, `L(a, b) = e^{-ab}`),
     the symmetric literal-sign flow `ȧ = ḃ = ṡ = -s e^{-s²}` from `(1, 1)` satisfies
     `L(0) = e⁻¹ < 1/2`, `e^{-t} ≤ s(t) ≤ e^{-t/e}` for all `t ≥ 0`, `s(t) → 0⁺`, and
     `Λ`-normalized limit `θ̄ = (0, √(3/2))`, where both margins vanish for every `α : ℝ`.

Machine-checked vs prose-only:
- Machine-checked: Uniqueness of `Λ = diag(1/3, 2/3)`, primal feasibility obstruction,
  gradient-flow verification, positivity, exact cancellation invariant `D' = 3a³e^{-a³}`,
  divergence `a → ∞`, `L → 0`, `b/a → 1`, `Λ`-normalized limit `(√3, 0)`, infeasibility of
  all rescalings `ψ_α(√3, 0)`, and the complete literal-A4 sign counterexample.
- Prose-only (in `solutions/quasi_homog_cascade/proof.pdf`): Standard Picard–Lindelöf local
  existence + finite-time non-explosion argument establishing existence of the global ODE
  solutions on `[0, ∞)`, and the numerical DOP853 verification table.
-/

open Filter Topology

namespace QuasiHomogCascade

/-! ### 1. Model, Quasi-Homogeneity, and Primal Feasibility Obstruction -/

/-- The smooth polynomial classifier `f(x; a, b) = x₁ a b + x₂ a³`. -/
def model (x θ : ℝ × ℝ) : ℝ :=
  x.1 * θ.1 * θ.2 + x.2 * θ.1 ^ 3

/-- Quasi-homogeneous scaling `ψ_α(a, b) = (e^{α λ₁} a, e^{α λ₂} b)`. -/
noncomputable def psi (lam : ℝ × ℝ) (α : ℝ) (θ : ℝ × ℝ) : ℝ × ℝ :=
  (Real.exp (α * lam.1) * θ.1, Real.exp (α * lam.2) * θ.2)

/-- Signed margins on `(e₁, +1)` and `(-e₂, -1)`. -/
def q1 (θ : ℝ × ℝ) : ℝ := θ.1 * θ.2
def q2 (θ : ℝ × ℝ) : ℝ := θ.1 ^ 3

lemma q1_eq_model (θ : ℝ × ℝ) : q1 θ = model (1, 0) θ := by
  unfold q1 model; ring

lemma q2_eq_model (θ : ℝ × ℝ) : q2 θ = (-1) * model (0, -1) θ := by
  unfold q2 model; ring

/-- **Theorem 1, Step 1**: `model` is quasi-homogeneous with diagonal rate matrix `Λ`
if and only if `Λ = (1/3, 2/3)`. -/
theorem quasi_homog_unique_rates (lam : ℝ × ℝ) :
    (∀ x θ α, model x (psi lam α θ) = Real.exp α * model x θ) ↔
      lam = (1 / 3, 2 / 3) := by
  constructor
  · intro h
    have h2 := h (0, 1) (1, 1) 1
    simp only [model, psi, zero_mul, one_mul, mul_one, one_pow, zero_add] at h2
    have he2 : Real.exp (3 * lam.1) = Real.exp 1 := by
      calc Real.exp (3 * lam.1) = Real.exp (lam.1 + lam.1 + lam.1) := by ring_nf
        _ = Real.exp lam.1 ^ 3 := by rw [Real.exp_add, Real.exp_add]; ring
        _ = Real.exp 1 := h2
    have hl1 : lam.1 = 1 / 3 := by
      have := Real.exp_injective he2
      linarith
    have h1 := h (1, 0) (1, 1) 1
    simp only [model, psi, one_mul, mul_one, zero_mul, add_zero] at h1
    have he1 : Real.exp (lam.1 + lam.2) = Real.exp 1 := by
      rw [Real.exp_add]; exact h1
    have hl2 : lam.2 = 2 / 3 := by
      have := Real.exp_injective he1
      linarith
    exact Prod.ext hl1 hl2
  · rintro rfl x θ α
    unfold model psi
    have hpow : (Real.exp (α * (1 / 3)) * θ.1) ^ 3 = Real.exp α * θ.1 ^ 3 := by
      have h3 : Real.exp (α * (1 / 3)) ^ 3 = Real.exp α := by
        calc Real.exp (α * (1 / 3)) ^ 3
            = Real.exp (α * (1 / 3) + α * (1 / 3) + α * (1 / 3)) := by
              rw [Real.exp_add, Real.exp_add]; ring
          _ = Real.exp α := by ring_nf
      calc (Real.exp (α * (1 / 3)) * θ.1) ^ 3
          = Real.exp (α * (1 / 3)) ^ 3 * θ.1 ^ 3 := by ring
        _ = Real.exp α * θ.1 ^ 3 := by rw [h3]
    have hprod : Real.exp (α * (1 / 3)) * Real.exp (α * (2 / 3)) = Real.exp α := by
      rw [← Real.exp_add]; ring_nf
    calc x.1 * (Real.exp (α * (1 / 3)) * θ.1) * (Real.exp (α * (2 / 3)) * θ.2) +
          x.2 * (Real.exp (α * (1 / 3)) * θ.1) ^ 3
        = x.1 * (Real.exp (α * (1 / 3)) * Real.exp (α * (2 / 3))) * θ.1 * θ.2 +
          x.2 * (Real.exp α * θ.1 ^ 3) := by rw [hpow]; ring
      _ = Real.exp α * (x.1 * θ.1 * θ.2 + x.2 * θ.1 ^ 3) := by rw [hprod]; ring

/-- Primal feasibility for the cascading minimization problem at threshold `λ̃`. -/
def CascadeFeasible (lam : ℝ × ℝ) (lamTilde : ℝ) (θ : ℝ × ℝ) : Prop :=
  1 ≤ q1 θ ∧ 1 ≤ q2 θ ∧ (lamTilde < lam.1 → θ.1 = 0) ∧ (lamTilde < lam.2 → θ.2 = 0)

/-- **Proposition 1 (Feasibility obstruction)**: If `q₁(θ̄) ≤ 0`, then NO finite rescaling
`ψ_α(θ̄)` is feasible (hence never a KKT point) for ANY cascade threshold `λ̃ ∈ ℝ`. -/
theorem cascade_infeasible_of_q1_nonpos {θBar : ℝ × ℝ} (hq : q1 θBar ≤ 0)
    (α lamTilde : ℝ) :
    ¬ CascadeFeasible (1 / 3, 2 / 3) lamTilde (psi (1 / 3, 2 / 3) α θBar) := by
  intro hfeas
  have hq1 := hfeas.1
  unfold q1 psi at hq1
  unfold q1 at hq
  have hexp : Real.exp (α * (1 / 3)) * Real.exp (α * (2 / 3)) = Real.exp α := by
    rw [← Real.exp_add]; ring_nf
  have hscaled : Real.exp (α * (1 / 3)) * θBar.1 * (Real.exp (α * (2 / 3)) * θBar.2) =
      Real.exp α * (θBar.1 * θBar.2) := by
    calc Real.exp (α * (1 / 3)) * θBar.1 * (Real.exp (α * (2 / 3)) * θBar.2)
        = (Real.exp (α * (1 / 3)) * Real.exp (α * (2 / 3))) * (θBar.1 * θBar.2) := by ring
      _ = Real.exp α * (θBar.1 * θBar.2) := by rw [hexp]
  rw [hscaled] at hq1
  have hnonpos : Real.exp α * (θBar.1 * θBar.2) ≤ 0 :=
    mul_nonpos_of_nonneg_of_nonpos (le_of_lt (Real.exp_pos α)) hq
  linarith

/-! ### 2. Gradient-Descent Flow Dynamics and Normalized Limit `(√3, 0)` (Theorem 1) -/

/-- Exponential loss `L(a, b) = (e^{-ab} + e^{-a³}) / 2`. -/
noncomputable def expLoss (a b : ℝ) : ℝ :=
  (Real.exp (-(a * b)) + Real.exp (-(a ^ 3))) / 2

/-- The descent vector field equals `-∇ L(a, b)`. -/
theorem expLoss_neg_grad (a b : ℝ) :
    HasDerivAt (fun x => expLoss x b) (-(b * Real.exp (-(a * b)) + 3 * a ^ 2 * Real.exp (-(a ^ 3))) / 2) a ∧
    HasDerivAt (fun y => expLoss a y) (-(a * Real.exp (-(a * b))) / 2) b := by
  constructor
  · have h1 : HasDerivAt (fun x : ℝ => -(x * b)) (-b) a := by
      have := ((hasDerivAt_id a).mul_const b).neg
      simpa using this
    have h2 : HasDerivAt (fun x : ℝ => -(x ^ 3)) (-(3 * a ^ 2)) a := by
      have := ((hasDerivAt_id a).pow 3).neg
      simpa using this
    have hsum := (h1.exp.add h2.exp).div_const 2
    convert hsum using 1
    ring
  · have h1 : HasDerivAt (fun y : ℝ => -(a * y)) (-a) b := by
      have := ((hasDerivAt_id b).const_mul a).neg
      simpa using this
    have hsum := (h1.exp.add_const (Real.exp (-(a ^ 3)))).div_const 2
    convert hsum using 1
    ring

/-- A global `C¹` gradient-descent trajectory `(a, b)` on `[0, ∞)`. -/
structure DescentTrajectory (a b : ℝ → ℝ) : Prop where
  a0_pos : 0 < a 0
  b0_pos : 0 < b 0
  ha : ∀ t ≥ 0, HasDerivAt a
    ((b t * Real.exp (-(a t * b t)) + 3 * (a t) ^ 2 * Real.exp (-(a t) ^ 3)) / 2) t
  hb : ∀ t ≥ 0, HasDerivAt b ((a t * Real.exp (-(a t * b t))) / 2) t

/-- Along any positive-start descent trajectory, `a(t) ≥ a(0) > 0` and `b(t) ≥ b(0) > 0`
for all `t ≥ 0`. -/
lemma descent_pos {a b : ℝ → ℝ} (h : DescentTrajectory a b) :
    ∀ t ≥ 0, a 0 ≤ a t ∧ b 0 ≤ b t := by
  have hcont_a : ContinuousOn a (Set.Ici 0) :=
    fun t ht => (h.ha t ht).continuousAt.continuousWithinAt
  have hcont_b : ContinuousOn b (Set.Ici 0) :=
    fun t ht => (h.hb t ht).continuousAt.continuousWithinAt
  have hpos : ∀ t ≥ 0, 0 < a t ∧ 0 < b t := by
    by_contra hnot
    push_neg at hnot
    let S : Set ℝ := {t : ℝ | 0 ≤ t ∧ (a t ≤ 0 ∨ b t ≤ 0)}
    have hSne : S.Nonempty := by
      obtain ⟨t, ht0, htneg⟩ := hnot
      by_cases hat : a t ≤ 0
      · exact ⟨t, ht0, Or.inl hat⟩
      · push_neg at hat
        exact ⟨t, ht0, Or.inr (htneg hat)⟩
    have hSbdd : BddBelow S := ⟨0, fun _ hx => hx.1⟩
    set t₀ := sInf S
    have ht0_nonneg : 0 ≤ t₀ := le_csInf hSne (fun _ hx => hx.1)
    have hlt_pos : ∀ t ∈ Set.Ico 0 t₀, 0 < a t ∧ 0 < b t := by
      intro t ⟨ht_ge, ht_lt⟩
      by_contra hle
      have htS : t ∈ S := by
        refine ⟨ht_ge, ?_⟩
        by_cases ha0 : a t ≤ 0
        · exact Or.inl ha0
        · push_neg at ha0; exact Or.inr (le_of_not_gt (fun hb0 => hle ⟨ha0, hb0⟩))
      exact not_lt_of_ge (csInf_le hSbdd htS) ht_lt
    have ha_mono : MonotoneOn a (Set.Icc 0 t₀) := by
      apply monotoneOn_of_deriv_nonneg (convex_Icc 0 t₀)
      · exact hcont_a.mono (fun x hx => hx.1)
      · intro x hx
        rw [interior_Icc] at hx
        exact (h.ha x (le_of_lt hx.1)).differentiableAt.differentiableWithinAt
      · intro x hx
        rw [interior_Icc] at hx
        rw [(h.ha x (le_of_lt hx.1)).deriv]
        have hp := hlt_pos x ⟨le_of_lt hx.1, hx.2⟩
        have h1 : 0 < b x * Real.exp (-(a x * b x)) := mul_pos hp.2 (Real.exp_pos _)
        have h2 : 0 ≤ 3 * a x ^ 2 * Real.exp (-(a x ^ 3)) :=
          mul_nonneg (by positivity) (le_of_lt (Real.exp_pos _))
        linarith
    have hb_mono : MonotoneOn b (Set.Icc 0 t₀) := by
      apply monotoneOn_of_deriv_nonneg (convex_Icc 0 t₀)
      · exact hcont_b.mono (fun x hx => hx.1)
      · intro x hx
        rw [interior_Icc] at hx
        exact (h.hb x (le_of_lt hx.1)).differentiableAt.differentiableWithinAt
      · intro x hx
        rw [interior_Icc] at hx
        rw [(h.hb x (le_of_lt hx.1)).deriv]
        have hp := hlt_pos x ⟨le_of_lt hx.1, hx.2⟩
        have h1 : 0 < a x * Real.exp (-(a x * b x)) := mul_pos hp.1 (Real.exp_pos _)
        linarith
    have ha_t0_pos : 0 < a t₀ :=
      lt_of_lt_of_le h.a0_pos (ha_mono ⟨le_rfl, ht0_nonneg⟩ ⟨ht0_nonneg, le_rfl⟩ ht0_nonneg)
    have hb_t0_pos : 0 < b t₀ :=
      lt_of_lt_of_le h.b0_pos (hb_mono ⟨le_rfl, ht0_nonneg⟩ ⟨ht0_nonneg, le_rfl⟩ ht0_nonneg)
    have heva := (h.ha t₀ ht0_nonneg).continuousAt.eventually (lt_mem_nhds ha_t0_pos)
    have hevb := (h.hb t₀ ht0_nonneg).continuousAt.eventually (lt_mem_nhds hb_t0_pos)
    rw [Metric.eventually_nhds_iff] at heva hevb
    obtain ⟨δa, hδa, hba⟩ := heva
    obtain ⟨δb, hδb, hbb⟩ := hevb
    set δ := min δa δb
    have hδ : 0 < δ := lt_min hδa hδb
    have hlt_ex : ∃ s ∈ S, s < t₀ + δ := exists_lt_of_csInf_lt hSne (by linarith)
    obtain ⟨s, hsS, hslt⟩ := hlt_ex
    have hs_ge : t₀ ≤ s := csInf_le hSbdd hsS
    have hdist_s : dist s t₀ < δ := by
      rw [Real.dist_eq, abs_of_nonneg (sub_nonneg.mpr hs_ge)]
      linarith
    have has : 0 < a s := hba (lt_of_lt_of_le hdist_s (min_le_left _ _))
    have hbs : 0 < b s := hbb (lt_of_lt_of_le hdist_s (min_le_right _ _))
    rcases hsS.2 with h1 | h2 <;> linarith
  have ha_mono : MonotoneOn a (Set.Ici 0) := by
    apply monotoneOn_of_deriv_nonneg (convex_Ici 0) hcont_a
    · intro x hx
      rw [interior_Ici] at hx
      exact (h.ha x (le_of_lt hx)).differentiableAt.differentiableWithinAt
    · intro x hx
      rw [interior_Ici] at hx
      rw [(h.ha x (le_of_lt hx)).deriv]
      have hp := hpos x (le_of_lt hx)
      have h1 : 0 < b x * Real.exp (-(a x * b x)) := mul_pos hp.2 (Real.exp_pos _)
      have h2 : 0 ≤ 3 * a x ^ 2 * Real.exp (-(a x ^ 3)) :=
        mul_nonneg (by positivity) (le_of_lt (Real.exp_pos _))
      linarith
  have hb_mono : MonotoneOn b (Set.Ici 0) := by
    apply monotoneOn_of_deriv_nonneg (convex_Ici 0) hcont_b
    · intro x hx
      rw [interior_Ici] at hx
      exact (h.hb x (le_of_lt hx)).differentiableAt.differentiableWithinAt
    · intro x hx
      rw [interior_Ici] at hx
      rw [(h.hb x (le_of_lt hx)).deriv]
      have hp := hpos x (le_of_lt hx)
      have h1 : 0 < a x * Real.exp (-(a x * b x)) := mul_pos hp.1 (Real.exp_pos _)
      linarith
  intro t ht
  exact ⟨ha_mono Set.left_mem_Ici ht ht, hb_mono Set.left_mem_Ici ht ht⟩

/-- **Exact cancellation invariant (eq. 3)**: `D(t) = a(t)² - b(t)²` satisfies
`D'(t) = 3 a(t)³ e^{-a(t)³} > 0`, whence `b(t)² ≤ a(t)² + b(0)² - a(0)²` for all `t ≥ 0`. -/
theorem descent_diff_sq_invariant {a b : ℝ → ℝ} (h : DescentTrajectory a b) :
    (∀ t ≥ 0, HasDerivAt (fun s => a s ^ 2 - b s ^ 2) (3 * a t ^ 3 * Real.exp (-(a t ^ 3))) t) ∧
    (∀ t ≥ 0, b t ^ 2 ≤ a t ^ 2 + b 0 ^ 2 - a 0 ^ 2) := by
  have hderiv : ∀ t ≥ 0, HasDerivAt (fun s => a s ^ 2 - b s ^ 2)
      (3 * a t ^ 3 * Real.exp (-(a t ^ 3))) t := by
    intro t ht
    have h1 := ((h.ha t ht).pow 2).sub ((h.hb t ht).pow 2)
    convert h1 using 1
    ring
  refine ⟨hderiv, ?_⟩
  have hmono : MonotoneOn (fun s => a s ^ 2 - b s ^ 2) (Set.Ici 0) := by
    apply monotoneOn_of_deriv_nonneg (convex_Ici 0)
    · intro t ht; exact (hderiv t ht).continuousAt.continuousWithinAt
    · intro t ht; rw [interior_Ici] at ht
      exact (hderiv t (le_of_lt ht)).differentiableAt.differentiableWithinAt
    · intro t ht; rw [interior_Ici] at ht
      rw [(hderiv t (le_of_lt ht)).deriv]
      have ha_pos : 0 < a t := lt_of_lt_of_le h.a0_pos (descent_pos h t (le_of_lt ht)).1
      exact mul_nonneg (by positivity) (le_of_lt (Real.exp_pos _))
  intro t ht
  have hle : a 0 ^ 2 - b 0 ^ 2 ≤ a t ^ 2 - b t ^ 2 := hmono Set.left_mem_Ici ht ht
  linarith

/-- Along any positive-start descent trajectory, `a(t) → +∞` and `expLoss (a t) (b t) → 0`. -/
theorem descent_diverge_and_loss_zero {a b : ℝ → ℝ} (h : DescentTrajectory a b) :
    Tendsto a atTop atTop ∧ Tendsto (fun t => expLoss (a t) (b t)) atTop (nhds 0) := by
  have hpos := descent_pos h
  have hinv := (descent_diff_sq_invariant h).2
  have ha_mono : MonotoneOn a (Set.Ici 0) := by
    apply monotoneOn_of_deriv_nonneg (convex_Ici 0)
    · intro t ht; exact (h.ha t ht).continuousAt.continuousWithinAt
    · intro t ht; rw [interior_Ici] at ht
      exact (h.ha t (le_of_lt ht)).differentiableAt.differentiableWithinAt
    · intro t ht; rw [interior_Ici] at ht
      rw [(h.ha t (le_of_lt ht)).deriv]
      have hp := hpos t (le_of_lt ht)
      have ha' : 0 < a t := lt_of_lt_of_le h.a0_pos hp.1
      have hb' : 0 < b t := lt_of_lt_of_le h.b0_pos hp.2
      have h1 : 0 < b t * Real.exp (-(a t * b t)) := mul_pos hb' (Real.exp_pos _)
      have h2 : 0 ≤ 3 * a t ^ 2 * Real.exp (-(a t ^ 3)) :=
        mul_nonneg (by positivity) (le_of_lt (Real.exp_pos _))
      linarith
  have ha_top : Tendsto a atTop atTop := by
    rw [tendsto_atTop_atTop]
    intro M
    by_contra hnot
    push_neg at hnot
    have ha_bdd : ∀ t ≥ 0, a t ≤ max M (a 0) := by
      intro t ht
      obtain ⟨s, hs, has⟩ := hnot t
      exact le_trans (ha_mono ht (le_trans ht hs) hs) (le_trans (le_of_lt has) (le_max_left _ _))
    set M₁ := max M (a 0)
    have hM1_pos : 0 < M₁ := lt_of_lt_of_le h.a0_pos (le_max_right _ _)
    set M₂ := Real.sqrt (M₁ ^ 2 + b 0 ^ 2)
    have hb_bdd : ∀ t ≥ 0, b t ≤ M₂ := by
      intro t ht
      have hbt_sq : b t ^ 2 ≤ M₁ ^ 2 + b 0 ^ 2 := by
        have h1 := hinv t ht
        have h2 := ha_bdd t ht
        have ha_nn : 0 ≤ a t := le_trans (le_of_lt h.a0_pos) (hpos t ht).1
        nlinarith [sq_nonneg (a 0)]
      have hbt_nn : 0 ≤ b t := le_trans (le_of_lt h.b0_pos) (hpos t ht).2
      rw [← Real.sqrt_sq hbt_nn]
      exact Real.sqrt_le_sqrt hbt_sq
    set c : ℝ := (b 0 / 2) * Real.exp (-(M₁ * M₂))
    have hc_pos : 0 < c := mul_pos (half_pos h.b0_pos) (Real.exp_pos _)
    have hlin_deriv : ∀ t ≥ 0, HasDerivAt (fun s => a s - c * s)
        ((b t * Real.exp (-(a t * b t)) + 3 * a t ^ 2 * Real.exp (-(a t ^ 3))) / 2 - c) t := by
      intro t ht
      have hd := (h.ha t ht).sub ((hasDerivAt_id t).const_mul c)
      convert hd using 1; ring
    have hlin_mono : MonotoneOn (fun t => a t - c * t) (Set.Ici 0) := by
      apply monotoneOn_of_deriv_nonneg (convex_Ici 0)
      · intro t ht; exact (hlin_deriv t ht).continuousAt.continuousWithinAt
      · intro t ht; rw [interior_Ici] at ht
        exact (hlin_deriv t (le_of_lt ht)).differentiableAt.differentiableWithinAt
      · intro t ht; rw [interior_Ici] at ht
        rw [(hlin_deriv t (le_of_lt ht)).deriv]
        have hp := hpos t (le_of_lt ht)
        have ha_le := ha_bdd t (le_of_lt ht)
        have hb_le := hb_bdd t (le_of_lt ht)
        have ha_nn : 0 ≤ a t := le_trans (le_of_lt h.a0_pos) hp.1
        have hb_nn : 0 ≤ b t := le_trans (le_of_lt h.b0_pos) hp.2
        have hab_le : a t * b t ≤ M₁ * M₂ := mul_le_mul ha_le hb_le hb_nn (le_of_lt hM1_pos)
        have hexp_ge : Real.exp (-(M₁ * M₂)) ≤ Real.exp (-(a t * b t)) :=
          Real.exp_le_exp.mpr (by linarith)
        have hterm1 : c ≤ b t * Real.exp (-(a t * b t)) / 2 := by
          dsimp [c]
          have h1 : b 0 * Real.exp (-(M₁ * M₂)) ≤ b t * Real.exp (-(a t * b t)) :=
            mul_le_mul hp.2 hexp_ge (le_of_lt (Real.exp_pos _)) hb_nn
          linarith
        have hterm2 : 0 ≤ 3 * a t ^ 2 * Real.exp (-(a t ^ 3)) / 2 :=
          div_nonneg (mul_nonneg (by positivity) (le_of_lt (Real.exp_pos _))) (by norm_num)
        linarith
    set T := (M₁ - a 0 + 1) / c
    have hT_nonneg : 0 ≤ T := div_nonneg (by linarith [le_max_right M (a 0)]) (le_of_lt hc_pos)
    have hge := hlin_mono Set.left_mem_Ici hT_nonneg hT_nonneg
    have haT := ha_bdd T hT_nonneg
    have hcT : c * T = M₁ - a 0 + 1 := mul_div_cancel₀ _ (ne_of_gt hc_pos)
    linarith
  refine ⟨ha_top, ?_⟩
  have hab_top : Tendsto (fun t => a t * b t) atTop atTop := by
    rw [tendsto_atTop_atTop] at ha_top ⊢
    intro M
    obtain ⟨T, hT⟩ := ha_top (max 0 (M / b 0))
    refine ⟨max T 0, fun t ht => ?_⟩
    have ht0 : 0 ≤ t := le_trans (le_max_right T 0) ht
    have hat : M / b 0 ≤ a t := le_trans (le_max_right _ _) (hT t (le_trans (le_max_left T 0) ht))
    have hbt : b 0 ≤ b t := (hpos t ht0).2
    have h1 : M ≤ a t * b 0 := by
      have := mul_le_mul_of_nonneg_right hat (le_of_lt h.b0_pos)
      rwa [div_mul_cancel₀ _ (ne_of_gt h.b0_pos)] at this
    have hat_nn : 0 ≤ a t := le_trans (le_of_lt h.a0_pos) (hpos t ht0).1
    exact le_trans h1 (mul_le_mul_of_nonneg_left hbt hat_nn)
  have ha3_top : Tendsto (fun t => a t ^ 3) atTop atTop := by
    have heq : (fun t => a t ^ 3) = fun t => (a t * a t) * a t := by ext t; ring
    rw [heq]
    exact (ha_top.atTop_mul_atTop₀ ha_top).atTop_mul_atTop₀ ha_top
  have h1 : Tendsto (fun t => Real.exp (-(a t * b t))) atTop (nhds 0) :=
    Real.tendsto_exp_atBot.comp (tendsto_neg_atTop_atBot.comp hab_top)
  have h2 : Tendsto (fun t => Real.exp (-(a t ^ 3))) atTop (nhds 0) :=
    Real.tendsto_exp_atBot.comp (tendsto_neg_atTop_atBot.comp ha3_top)
  have hsum := (h1.add h2).div_const 2
  simpa [expLoss] using hsum

/-- **Asymptotic ratio `b(t)/a(t) → 1`**: `D(t) = a(t)² - b(t)²` stays bounded above along the
flow, forcing `b(t)/a(t) → 1`. -/
theorem descent_ratio_tendsto_one {a b : ℝ → ℝ} (h : DescentTrajectory a b) :
    Tendsto (fun t => b t / a t) atTop (nhds 1) := by
  have hpos := descent_pos h
  have hinv := descent_diff_sq_invariant h
  have ha_top := (descent_diverge_and_loss_zero h).1
  obtain ⟨T₁, hT₁⟩ := (tendsto_atTop_atTop.mp ha_top) (max 4 (b 0))
  set T := max T₁ 0
  have hT_nonneg : 0 ≤ T := le_max_right _ _
  have hlarge : ∀ t ≥ T, 4 ≤ a t ∧ b 0 ≤ a t := fun t ht => by
    have hge := hT₁ t (le_trans (le_max_left _ _) ht)
    exact ⟨le_trans (le_max_left _ _) hge, le_trans (le_max_right _ _) hge⟩
  set K : ℝ := 720 / b 0
  have hK_pos : 0 < K := div_pos (by norm_num) h.b0_pos
  have hlyap_deriv : ∀ t ≥ T,
      HasDerivAt (fun s => (a s ^ 2 - b s ^ 2) + K / a s)
        (3 * a t ^ 3 * Real.exp (-(a t ^ 3)) -
          K * ((b t * Real.exp (-(a t * b t)) + 3 * a t ^ 2 * Real.exp (-(a t ^ 3))) / 2) / a t ^ 2) t := by
    intro t ht
    have ht0 : 0 ≤ t := le_trans hT_nonneg ht
    have ha_ne : a t ≠ 0 := ne_of_gt (lt_of_lt_of_le (by norm_num) (hlarge t ht).1)
    have hd1 := hinv.1 t ht0
    have hd2 : HasDerivAt (fun s => K / a s)
        (-(K * ((b t * Real.exp (-(a t * b t)) + 3 * a t ^ 2 * Real.exp (-(a t ^ 3))) / 2)) / a t ^ 2) t := by
      have hmul := ((h.ha t ht0).inv ha_ne).const_mul K
      have hfun : (fun y => K * a⁻¹ y) = (fun s => K / a s) := by
        funext s; exact (div_eq_mul_inv K (a s)).symm
      rw [hfun] at hmul
      convert hmul using 1
      ring
    have hadd := hd1.add hd2
    convert hadd using 1
    ring
  have hlyap_anti : AntitoneOn (fun s => (a s ^ 2 - b s ^ 2) + K / a s) (Set.Ici T) := by
    apply antitoneOn_of_deriv_nonpos (convex_Ici T)
    · intro t ht; exact (hlyap_deriv t ht).continuousAt.continuousWithinAt
    · intro t ht; rw [interior_Ici] at ht
      exact (hlyap_deriv t (le_of_lt ht)).differentiableAt.differentiableWithinAt
    · intro t ht; rw [interior_Ici] at ht
      have htT : T ≤ t := le_of_lt ht
      have ht0 : 0 ≤ t := le_trans hT_nonneg htT
      rw [(hlyap_deriv t htT).deriv]
      have ha4 : 4 ≤ a t := (hlarge t htT).1
      have hab0 : b 0 ≤ a t := (hlarge t htT).2
      have ha_pos : 0 < a t := by linarith
      have hb_pos : 0 < b t := lt_of_lt_of_le h.b0_pos (hpos t ht0).2
      have hb_le : b t ≤ 2 * a t := by
        have hb0_sq : b 0 ^ 2 ≤ a t ^ 2 := by nlinarith [h.b0_pos]
        have hsq : b t ^ 2 ≤ (2 * a t) ^ 2 := by
          have := hinv.2 t ht0
          nlinarith [sq_nonneg (a 0)]
        nlinarith
      have hexp_arg : a t * b t - a t ^ 3 ≤ -a t := by
        have h1 : a t * b t ≤ 2 * a t ^ 2 := by nlinarith
        have h2 : a t * (a t * (a t - 2) - 1) ≥ 0 := by nlinarith
        nlinarith
      have htaylor : a t ^ 5 / 120 ≤ Real.exp (a t) := by
        have hsum := Real.sum_le_exp_of_nonneg (le_of_lt ha_pos) 6
        have hterm : a t ^ 5 / (Nat.factorial 5 : ℝ) ≤
            ∑ i ∈ Finset.range 6, a t ^ i / (Nat.factorial i : ℝ) :=
          Finset.single_le_sum (f := fun i : ℕ => a t ^ i / (Nat.factorial i : ℝ))
            (fun i _ => by positivity) (Finset.mem_range.mpr (by norm_num))
        have hfact : (Nat.factorial 5 : ℝ) = 120 := by norm_num [Nat.factorial]
        rw [hfact] at hterm
        linarith
      have hkey : 3 * a t ^ 5 * Real.exp (-(a t ^ 3)) ≤ 360 * Real.exp (-(a t * b t)) := by
        have h1 : 3 * a t ^ 5 ≤ 360 * Real.exp (a t) := by linarith
        have h2 : Real.exp (a t) * Real.exp (-(a t ^ 3)) ≤ Real.exp (-(a t * b t)) := by
          rw [← Real.exp_add]
          exact Real.exp_le_exp.mpr (by linarith)
        have h3 : 3 * a t ^ 5 * Real.exp (-(a t ^ 3)) ≤
            360 * Real.exp (a t) * Real.exp (-(a t ^ 3)) :=
          mul_le_mul_of_nonneg_right h1 (le_of_lt (Real.exp_pos _))
        have h4 : 360 * Real.exp (a t) * Real.exp (-(a t ^ 3)) ≤ 360 * Real.exp (-(a t * b t)) := by
          have := mul_le_mul_of_nonneg_left h2 (show (0 : ℝ) ≤ 360 by norm_num)
          linarith
        linarith
      have hKb : K * b 0 / 2 = 360 := by
        dsimp [K]; field_simp [ne_of_gt h.b0_pos]; ring
      have hnum : 3 * a t ^ 5 * Real.exp (-(a t ^ 3)) ≤
          K * ((b t * Real.exp (-(a t * b t)) + 3 * a t ^ 2 * Real.exp (-(a t ^ 3))) / 2) := by
        have hbge : b 0 ≤ b t := (hpos t ht0).2
        have hpos2 : 0 ≤ 3 * a t ^ 2 * Real.exp (-(a t ^ 3)) :=
          mul_nonneg (by positivity) (le_of_lt (Real.exp_pos _))
        have h1 : K * (b 0 * Real.exp (-(a t * b t)) / 2) ≤
            K * ((b t * Real.exp (-(a t * b t)) + 3 * a t ^ 2 * Real.exp (-(a t ^ 3))) / 2) := by
          apply mul_le_mul_of_nonneg_left _ (le_of_lt hK_pos)
          have := mul_le_mul_of_nonneg_right hbge (le_of_lt (Real.exp_pos (-(a t * b t))))
          linarith
        have h2 : K * (b 0 * Real.exp (-(a t * b t)) / 2) = 360 * Real.exp (-(a t * b t)) := by
          calc K * (b 0 * Real.exp (-(a t * b t)) / 2)
              = (K * b 0 / 2) * Real.exp (-(a t * b t)) := by ring
            _ = 360 * Real.exp (-(a t * b t)) := by rw [hKb]
        linarith
      rw [sub_nonpos, le_div_iff₀ (sq_pos_of_pos ha_pos)]
      calc 3 * a t ^ 3 * Real.exp (-(a t ^ 3)) * a t ^ 2
          = 3 * a t ^ 5 * Real.exp (-(a t ^ 3)) := by ring
        _ ≤ K * ((b t * Real.exp (-(a t * b t)) + 3 * a t ^ 2 * Real.exp (-(a t ^ 3))) / 2) := hnum
  set C_hi := (a T ^ 2 - b T ^ 2) + K / a T
  set C_lo := a 0 ^ 2 - b 0 ^ 2
  have hD_bounds : ∀ t ≥ T, C_lo ≤ a t ^ 2 - b t ^ 2 ∧ a t ^ 2 - b t ^ 2 ≤ C_hi := by
    intro t ht
    have ht0 : 0 ≤ t := le_trans hT_nonneg ht
    have ha_pos : 0 < a t := lt_of_lt_of_le (by norm_num) (hlarge t ht).1
    refine ⟨by linarith [hinv.2 t ht0], ?_⟩
    have hanti := hlyap_anti Set.left_mem_Ici ht ht
    have hKpos : 0 < K / a t := div_pos hK_pos ha_pos
    linarith
  have ha2_top : Tendsto (fun t => a t ^ 2) atTop atTop := by
    have heq : (fun t => a t ^ 2) = fun t => a t * a t := by ext t; ring
    rw [heq]; exact ha_top.atTop_mul_atTop₀ ha_top
  have hD_div_zero : Tendsto (fun t => (a t ^ 2 - b t ^ 2) / a t ^ 2) atTop (nhds 0) := by
    refine squeeze_zero_norm' (a := fun t => max |C_lo| |C_hi| / a t ^ 2) ?_ ?_
    · filter_upwards [eventually_ge_atTop T] with t ht
      have ha_pos : 0 < a t := lt_of_lt_of_le (by norm_num) (hlarge t ht).1
      rw [Real.norm_eq_abs, abs_div, abs_of_pos (sq_pos_of_pos ha_pos)]
      apply div_le_div_of_nonneg_right _ (le_of_lt (sq_pos_of_pos ha_pos))
      have hb := hD_bounds t ht
      rw [abs_le]
      constructor
      · linarith [neg_abs_le C_lo, le_max_left |C_lo| |C_hi|]
      · linarith [le_abs_self C_hi, le_max_right |C_lo| |C_hi|]
    · exact tendsto_const_nhds.div_atTop ha2_top
  have hsq_one : Tendsto (fun t => (b t / a t) ^ 2) atTop (nhds 1) := by
    have hsub : Tendsto (fun t => 1 - (a t ^ 2 - b t ^ 2) / a t ^ 2) atTop (nhds (1 - 0)) :=
      tendsto_const_nhds.sub hD_div_zero
    rw [sub_zero] at hsub
    refine hsub.congr' ?_
    filter_upwards [eventually_ge_atTop T] with t ht
    have ha_ne : a t ≠ 0 := ne_of_gt (lt_of_lt_of_le (by norm_num) (hlarge t ht).1)
    field_simp [ha_ne]
    ring
  have hsqrt := (Real.continuous_sqrt.tendsto 1).comp hsq_one
  rw [Real.sqrt_one] at hsqrt
  refine hsqrt.congr' ?_
  filter_upwards [eventually_ge_atTop T] with t ht
  have ht0 : 0 ≤ t := le_trans hT_nonneg ht
  have ha_pos : 0 < a t := lt_of_lt_of_le (by norm_num) (hlarge t ht).1
  have hb_pos : 0 < b t := lt_of_lt_of_le h.b0_pos (hpos t ht0).2
  simp only [Function.comp_apply]
  exact Real.sqrt_sq (div_nonneg (le_of_lt hb_pos) (le_of_lt ha_pos))

/-- Explicit positive `Λ`-normalization factor `r(a, b) > 0` satisfying `(a r)² + 2(b r²)² = 3`. -/
noncomputable def normScale (a b : ℝ) : ℝ :=
  Real.sqrt ((-a ^ 2 + Real.sqrt (a ^ 4 + 24 * b ^ 2)) / (4 * b ^ 2))

lemma normScale_spec {a b : ℝ} (_ha : 0 < a) (hb : 0 < b) :
    0 < normScale a b ∧
    (a * normScale a b) ^ 2 + 2 * (b * normScale a b ^ 2) ^ 2 = 3 := by
  set disc := a ^ 4 + 24 * b ^ 2
  have hdisc_gt : a ^ 4 < disc := by dsimp [disc]; linarith [sq_pos_of_pos hb]
  have hsqrt_gt : a ^ 2 < Real.sqrt disc := by
    rw [← Real.sqrt_sq (sq_nonneg a)]
    have : (a ^ 2) ^ 2 = a ^ 4 := by ring
    rw [this]
    exact Real.sqrt_lt_sqrt (by positivity) hdisc_gt
  set u := (-a ^ 2 + Real.sqrt disc) / (4 * b ^ 2)
  have hu_pos : 0 < u := div_pos (by linarith) (by positivity)
  have hr_pos : 0 < normScale a b := Real.sqrt_pos.mpr hu_pos
  have hr_sq : normScale a b ^ 2 = u := Real.sq_sqrt (le_of_lt hu_pos)
  refine ⟨hr_pos, ?_⟩
  have hsq_disc : Real.sqrt disc ^ 2 = disc := Real.sq_sqrt (by positivity)
  have hb2_ne : (4 * b ^ 2 : ℝ) ≠ 0 := by positivity
  calc (a * normScale a b) ^ 2 + 2 * (b * normScale a b ^ 2) ^ 2
      = a ^ 2 * (normScale a b ^ 2) + 2 * b ^ 2 * (normScale a b ^ 2) ^ 2 := by ring
    _ = a ^ 2 * u + 2 * b ^ 2 * u ^ 2 := by rw [hr_sq]
    _ = 3 := by
        dsimp [u]
        field_simp [ne_of_gt hb]
        nlinarith [hsq_disc, show disc = a ^ 4 + 24 * b ^ 2 from rfl]

/-- **Theorem 1, Step 4 (`Λ`-normalized limit and complete refutation)**:
Along any positive-start descent trajectory, the `Λ`-normalized trajectory
`(â(t), b̂(t)) = (a(t) r(t), b(t) r(t)²)` converges to `(√3, 0)`, at which EVERY finite
rescaling `ψ_α(√3, 0)` violates primal feasibility (`q₁ = 0 < 1`) for every cascade
threshold `λ̃ ∈ ℝ`. -/
theorem descent_normalized_limit_and_refutation {a b : ℝ → ℝ} (h : DescentTrajectory a b) :
    Tendsto (fun t => a t * normScale (a t) (b t)) atTop (nhds (Real.sqrt 3)) ∧
    Tendsto (fun t => b t * normScale (a t) (b t) ^ 2) atTop (nhds 0) ∧
    (∀ α lamTilde : ℝ,
      ¬ CascadeFeasible (1 / 3, 2 / 3) lamTilde (psi (1 / 3, 2 / 3) α (Real.sqrt 3, 0))) := by
  have hpos := descent_pos h
  have ha_top := (descent_diverge_and_loss_zero h).1
  have hratio := descent_ratio_tendsto_one h
  have hbhat_zero : Tendsto (fun t => b t * normScale (a t) (b t) ^ 2) atTop (nhds 0) := by
    refine squeeze_zero' (g := fun t => 3 * (b t / a t) / a t) ?_ ?_ ?_
    · filter_upwards [eventually_ge_atTop (0 : ℝ)] with t ht
      have hp := hpos t ht
      have hb_pos : 0 < b t := lt_of_lt_of_le h.b0_pos hp.2
      positivity
    · filter_upwards [eventually_ge_atTop (0 : ℝ)] with t ht
      have hp := hpos t ht
      have ha_pos : 0 < a t := lt_of_lt_of_le h.a0_pos hp.1
      have hb_pos : 0 < b t := lt_of_lt_of_le h.b0_pos hp.2
      have hspec := normScale_spec ha_pos hb_pos
      have har_sq : (a t * normScale (a t) (b t)) ^ 2 ≤ 3 := by
        have : 0 ≤ 2 * (b t * normScale (a t) (b t) ^ 2) ^ 2 := by positivity
        linarith [hspec.2]
      have hr2_le : normScale (a t) (b t) ^ 2 ≤ 3 / a t ^ 2 := by
        rw [le_div_iff₀ (sq_pos_of_pos ha_pos)]
        nlinarith
      calc b t * normScale (a t) (b t) ^ 2
          ≤ b t * (3 / a t ^ 2) := mul_le_mul_of_nonneg_left hr2_le (le_of_lt hb_pos)
        _ = 3 * (b t / a t) / a t := by ring
    · have hnum : Tendsto (fun t => 3 * (b t / a t)) atTop (nhds 3) := by
        simpa using hratio.const_mul 3
      exact hnum.div_atTop ha_top
  have hahat_lim : Tendsto (fun t => a t * normScale (a t) (b t)) atTop (nhds (Real.sqrt 3)) := by
    have hsq : Tendsto (fun t => 3 - 2 * (b t * normScale (a t) (b t) ^ 2) ^ 2)
        atTop (nhds (3 - 2 * 0 ^ 2)) :=
      tendsto_const_nhds.sub ((hbhat_zero.pow 2).const_mul 2)
    have hsqrt := (Real.continuous_sqrt.tendsto 3).comp (by simpa using hsq)
    refine hsqrt.congr' ?_
    filter_upwards [eventually_ge_atTop (0 : ℝ)] with t ht
    have hp := hpos t ht
    have ha_pos : 0 < a t := lt_of_lt_of_le h.a0_pos hp.1
    have hb_pos : 0 < b t := lt_of_lt_of_le h.b0_pos hp.2
    have hspec := normScale_spec ha_pos hb_pos
    have h3 : 3 - 2 * (b t * normScale (a t) (b t) ^ 2) ^ 2 =
        (a t * normScale (a t) (b t)) ^ 2 := by linarith [hspec.2]
    simp only [Function.comp_apply]
    rw [h3, Real.sqrt_sq (mul_nonneg (le_of_lt ha_pos) (le_of_lt hspec.1))]
  refine ⟨hahat_lim, hbhat_zero, fun α lamTilde => ?_⟩
  exact cascade_infeasible_of_q1_nonpos (by simp [q1]) α lamTilde

/-! ### 3. Counterexample for the Literal Printed Sign in Assumption A4 (Theorem 2) -/

/-- Literal-sign symmetric trajectory `a(t) = b(t) = s(t)` satisfying `ṡ = -s e^{-s²}`, `s(0) = 1`. -/
structure LiteralSignTrajectory (s : ℝ → ℝ) : Prop where
  s0 : s 0 = 1
  hs : ∀ t ≥ 0, HasDerivAt s (-(s t * Real.exp (-(s t ^ 2)))) t

/-- **Theorem 2 (Literal A4 plus-sign refutation)**: Along the symmetric literal-sign flow
`ṡ = -s e^{-s²}` from `s(0) = 1` on `{(e₁, +1), (-e₁, -1)}` (`q₁ = q₂ = ab`, `L = e^{-ab}`):
1. `e^{-t} ≤ s(t) ≤ e^{-t / e}` for all `t ≥ 0` (so `s(t) → 0⁺`);
2. `L(s(0), s(0)) = e⁻¹ < 1/2` (verifying A5);
3. The `Λ`-normalized iterate converges to `(0, √(3/2))`, where both margins vanish for every `α`. -/
theorem literal_A4_sign_refutation {s : ℝ → ℝ} (h : LiteralSignTrajectory s) :
    (∀ t ≥ 0, Real.exp (-t) ≤ s t ∧ s t ≤ Real.exp (-t / Real.exp 1)) ∧
    Tendsto s atTop (nhds 0) ∧
    Real.exp (-(s 0 * s 0)) < 1 / 2 ∧
    Tendsto (fun t => s t * normScale (s t) (s t)) atTop (nhds 0) ∧
    Tendsto (fun t => s t * normScale (s t) (s t) ^ 2) atTop (nhds (Real.sqrt (3 / 2))) ∧
    (∀ α lamTilde : ℝ,
      ¬ CascadeFeasible (1 / 3, 2 / 3) lamTilde (psi (1 / 3, 2 / 3) α (0, Real.sqrt (3 / 2)))) := by
  have hg_deriv : ∀ t ≥ 0, HasDerivAt (fun u => s u * Real.exp u)
      (s t * Real.exp t * (1 - Real.exp (-(s t ^ 2)))) t := by
    intro t ht
    have hd := (h.hs t ht).mul (Real.hasDerivAt_exp t)
    convert hd using 1
    ring
  have hs_lower : ∀ t ≥ 0, Real.exp (-t) ≤ s t := by
    have hpos : ∀ t ≥ 0, 0 < s t := by
      by_contra hnot
      push_neg at hnot
      let S : Set ℝ := {t : ℝ | 0 ≤ t ∧ s t ≤ 0}
      have hSne : S.Nonempty := by obtain ⟨t, ht, hle⟩ := hnot; exact ⟨t, ht, hle⟩
      have hSbdd : BddBelow S := ⟨0, fun _ hx => hx.1⟩
      set t₀ := sInf S
      have ht0_nn : 0 ≤ t₀ := le_csInf hSne (fun _ hx => hx.1)
      have hlt_pos : ∀ t ∈ Set.Ico 0 t₀, 0 < s t := by
        intro t ⟨ht1, ht2⟩
        by_contra hle; push_neg at hle
        exact not_lt_of_ge (csInf_le hSbdd ⟨ht1, hle⟩) ht2
      have hmono_g : MonotoneOn (fun t => s t * Real.exp t) (Set.Icc 0 t₀) := by
        apply monotoneOn_of_deriv_nonneg (convex_Icc 0 t₀)
        · intro t ht; exact (hg_deriv t ht.1).continuousAt.continuousWithinAt
        · intro t ht; rw [interior_Icc] at ht
          exact (hg_deriv t (le_of_lt ht.1)).differentiableAt.differentiableWithinAt
        · intro t ht; rw [interior_Icc] at ht
          rw [(hg_deriv t (le_of_lt ht.1)).deriv]
          have hst : 0 < s t := hlt_pos t ⟨le_of_lt ht.1, ht.2⟩
          have hexp_le : Real.exp (-(s t ^ 2)) ≤ 1 := Real.exp_le_one_iff.mpr (by nlinarith)
          exact mul_nonneg (mul_nonneg (le_of_lt hst) (le_of_lt (Real.exp_pos t))) (by linarith)
      have hg0 : (1 : ℝ) ≤ s t₀ * Real.exp t₀ := by
        have hle := hmono_g ⟨le_rfl, ht0_nn⟩ ⟨ht0_nn, le_rfl⟩ ht0_nn
        simp [h.s0] at hle
        exact hle
      have hst0_pos : 0 < s t₀ := by
        have := Real.exp_pos t₀
        nlinarith
      have hev := (h.hs t₀ ht0_nn).continuousAt.eventually (lt_mem_nhds hst0_pos)
      rw [Metric.eventually_nhds_iff] at hev
      obtain ⟨δ, hδ, hball⟩ := hev
      have hlt_ex : ∃ u ∈ S, u < t₀ + δ := exists_lt_of_csInf_lt hSne (by linarith)
      obtain ⟨u, huS, hult⟩ := hlt_ex
      have hu_ge : t₀ ≤ u := csInf_le hSbdd huS
      have hdist : dist u t₀ < δ := by
        rw [Real.dist_eq, abs_of_nonneg (sub_nonneg.mpr hu_ge)]; linarith
      linarith [hball hdist, huS.2]
    have hmono_g : MonotoneOn (fun t => s t * Real.exp t) (Set.Ici 0) := by
      apply monotoneOn_of_deriv_nonneg (convex_Ici 0)
      · intro t ht; exact (hg_deriv t ht).continuousAt.continuousWithinAt
      · intro t ht; rw [interior_Ici] at ht
        exact (hg_deriv t (le_of_lt ht)).differentiableAt.differentiableWithinAt
      · intro t ht; rw [interior_Ici] at ht
        rw [(hg_deriv t (le_of_lt ht)).deriv]
        have hst : 0 < s t := hpos t (le_of_lt ht)
        have hexp_le : Real.exp (-(s t ^ 2)) ≤ 1 := Real.exp_le_one_iff.mpr (by nlinarith)
        exact mul_nonneg (mul_nonneg (le_of_lt hst) (le_of_lt (Real.exp_pos t))) (by linarith)
    intro t ht
    have hle := hmono_g Set.left_mem_Ici ht ht
    simp only [h.s0, Real.exp_zero, mul_one] at hle
    calc Real.exp (-t) = 1 * Real.exp (-t) := (one_mul _).symm
      _ ≤ (s t * Real.exp t) * Real.exp (-t) :=
        mul_le_mul_of_nonneg_right hle (le_of_lt (Real.exp_pos _))
      _ = s t := by rw [mul_assoc, ← Real.exp_add, add_neg_cancel, Real.exp_zero, mul_one]
  have hs_le_one : ∀ t ≥ 0, s t ≤ 1 := by
    have hanti : AntitoneOn s (Set.Ici 0) := by
      apply antitoneOn_of_deriv_nonpos (convex_Ici 0)
      · intro t ht; exact (h.hs t ht).continuousAt.continuousWithinAt
      · intro t ht; rw [interior_Ici] at ht
        exact (h.hs t (le_of_lt ht)).differentiableAt.differentiableWithinAt
      · intro t ht; rw [interior_Ici] at ht
        rw [(h.hs t (le_of_lt ht)).deriv]
        have hpos : 0 < s t := lt_of_lt_of_le (Real.exp_pos _) (hs_lower t (le_of_lt ht))
        nlinarith [Real.exp_pos (-(s t ^ 2))]
    intro t ht
    simpa [h.s0] using hanti Set.left_mem_Ici ht ht
  have hs_upper : ∀ t ≥ 0, s t ≤ Real.exp (-t / Real.exp 1) := by
    set c : ℝ := 1 / Real.exp 1
    have hgc_deriv : ∀ t ≥ 0, HasDerivAt (fun u => s u * Real.exp (c * u))
        (s t * Real.exp (c * t) * (c - Real.exp (-(s t ^ 2)))) t := by
      intro t ht
      have hd := (h.hs t ht).mul (((hasDerivAt_id t).const_mul c).exp)
      convert hd using 1
      simp only [id_eq]; ring
    have hanti : AntitoneOn (fun t => s t * Real.exp (c * t)) (Set.Ici 0) := by
      apply antitoneOn_of_deriv_nonpos (convex_Ici 0)
      · intro t ht; exact (hgc_deriv t ht).continuousAt.continuousWithinAt
      · intro t ht; rw [interior_Ici] at ht
        exact (hgc_deriv t (le_of_lt ht)).differentiableAt.differentiableWithinAt
      · intro t ht; rw [interior_Ici] at ht
        rw [(hgc_deriv t (le_of_lt ht)).deriv]
        have hst_pos : 0 < s t := lt_of_lt_of_le (Real.exp_pos _) (hs_lower t (le_of_lt ht))
        have hst_le1 : s t ≤ 1 := hs_le_one t (le_of_lt ht)
        have hexp_ge : c ≤ Real.exp (-(s t ^ 2)) := by
          dsimp [c]
          rw [one_div, ← Real.exp_neg]
          exact Real.exp_le_exp.mpr (by nlinarith)
        exact mul_nonpos_of_nonneg_of_nonpos
          (mul_nonneg (le_of_lt hst_pos) (le_of_lt (Real.exp_pos _))) (by linarith)
    intro t ht
    have hle := hanti Set.left_mem_Ici ht ht
    simp only [h.s0, mul_zero, Real.exp_zero, mul_one] at hle
    have harg : -t / Real.exp 1 = -(c * t) := by dsimp [c]; ring
    rw [harg]
    calc s t = (s t * Real.exp (c * t)) * Real.exp (-(c * t)) := by
          rw [mul_assoc, ← Real.exp_add, add_neg_cancel, Real.exp_zero, mul_one]
      _ ≤ 1 * Real.exp (-(c * t)) :=
          mul_le_mul_of_nonneg_right hle (le_of_lt (Real.exp_pos _))
      _ = Real.exp (-(c * t)) := one_mul _
  have hs_zero : Tendsto s atTop (nhds 0) := by
    refine squeeze_zero' (g := fun t => Real.exp (-t / Real.exp 1)) ?_ ?_ ?_
    · filter_upwards [eventually_ge_atTop (0 : ℝ)] with t ht
      exact le_of_lt (lt_of_lt_of_le (Real.exp_pos _) (hs_lower t ht))
    · filter_upwards [eventually_ge_atTop (0 : ℝ)] with t ht
      exact hs_upper t ht
    · have hlin : Tendsto (fun t : ℝ => -t / Real.exp 1) atTop atBot := by
        have heq : (fun t : ℝ => -t / Real.exp 1) = fun t => -(t * (1 / Real.exp 1)) := by
          ext t; ring
        rw [heq]
        exact tendsto_neg_atTop_atBot.comp (tendsto_id.atTop_mul_const (by positivity))
      exact Real.tendsto_exp_atBot.comp hlin
  have hL0 : Real.exp (-(s 0 * s 0)) < 1 / 2 := by
    rw [h.s0, mul_one, Real.exp_neg, inv_lt_comm₀ (Real.exp_pos 1) (by norm_num)]
    have h1 : (2 : ℝ) < Real.exp 1 := by
      have hsum := Real.sum_le_exp_of_nonneg zero_le_one 3
      simp [Finset.sum_range_succ, Nat.factorial] at hsum
      linarith
    linarith
  have hahat_zero : Tendsto (fun t => s t * normScale (s t) (s t)) atTop (nhds 0) := by
    have hsq_zero : Tendsto (fun t => (s t * normScale (s t) (s t)) ^ 2) atTop (nhds 0) := by
      refine squeeze_zero' (g := fun t => s t * Real.sqrt (3 / 2)) ?_ ?_ ?_
      · filter_upwards with t; exact sq_nonneg _
      · filter_upwards [eventually_ge_atTop (0 : ℝ)] with t ht
        have hs_pos : 0 < s t := lt_of_lt_of_le (Real.exp_pos _) (hs_lower t ht)
        have hspec := normScale_spec hs_pos hs_pos
        have hbhat_sq : (s t * normScale (s t) (s t) ^ 2) ^ 2 ≤ 3 / 2 := by
          have : 0 ≤ (s t * normScale (s t) (s t)) ^ 2 := sq_nonneg _
          linarith [hspec.2]
        have hbhat_le : s t * normScale (s t) (s t) ^ 2 ≤ Real.sqrt (3 / 2) := by
          rw [← Real.sqrt_sq (mul_nonneg (le_of_lt hs_pos) (sq_nonneg _))]
          exact Real.sqrt_le_sqrt hbhat_sq
        calc (s t * normScale (s t) (s t)) ^ 2
            = s t * (s t * normScale (s t) (s t) ^ 2) := by ring
          _ ≤ s t * Real.sqrt (3 / 2) := mul_le_mul_of_nonneg_left hbhat_le (le_of_lt hs_pos)
      · simpa using hs_zero.mul_const (Real.sqrt (3 / 2))
    have hsqrt := (Real.continuous_sqrt.tendsto 0).comp hsq_zero
    rw [Real.sqrt_zero] at hsqrt
    refine hsqrt.congr' ?_
    filter_upwards [eventually_ge_atTop (0 : ℝ)] with t ht
    have hs_pos : 0 < s t := lt_of_lt_of_le (Real.exp_pos _) (hs_lower t ht)
    have hspec := normScale_spec hs_pos hs_pos
    simp only [Function.comp_apply]
    exact Real.sqrt_sq (mul_nonneg (le_of_lt hs_pos) (le_of_lt hspec.1))
  have hbhat_lim : Tendsto (fun t => s t * normScale (s t) (s t) ^ 2) atTop (nhds (Real.sqrt (3 / 2))) := by
    have hsq : Tendsto (fun t => (3 - (s t * normScale (s t) (s t)) ^ 2) / 2)
        atTop (nhds ((3 - 0 ^ 2) / 2)) :=
      (tendsto_const_nhds.sub (hahat_zero.pow 2)).div_const 2
    have hsqrt := (Real.continuous_sqrt.tendsto (3 / 2)).comp (by simpa using hsq)
    refine hsqrt.congr' ?_
    filter_upwards [eventually_ge_atTop (0 : ℝ)] with t ht
    have hs_pos : 0 < s t := lt_of_lt_of_le (Real.exp_pos _) (hs_lower t ht)
    have hspec := normScale_spec hs_pos hs_pos
    have h3 : (3 - (s t * normScale (s t) (s t)) ^ 2) / 2 =
        (s t * normScale (s t) (s t) ^ 2) ^ 2 := by linarith [hspec.2]
    simp only [Function.comp_apply]
    rw [h3, Real.sqrt_sq (by positivity)]
  refine ⟨fun t ht => ⟨hs_lower t ht, hs_upper t ht⟩, hs_zero, hL0, hahat_zero, hbhat_lim,
    fun α lamTilde => cascade_infeasible_of_q1_nonpos (by simp [q1]) α lamTilde⟩

end QuasiHomogCascade
