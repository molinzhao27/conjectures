import Mathlib

/-!
# A 2×2 Counterexample to Moving-Reference I-Projection Selection
(Luis Leal, arXiv:2606.28308, Section 7, Conjecture 1 `conj:iproj`)

We formalize in Lean 4 + Mathlib (complete kernel-verified proofs, standard foundations only) the complete refutation
of Conjecture 1 (`conj:iproj`), including its uniform-reference / maximum-entropy specialization,
using the 2×2 zero-sum game `A = [[0, 2], [0, -1]]`:

1. **Exact Nash Polytope and I-Projection (`Section 1`)**:
   - `payoff_eq_matrix`: `xᵀ A y = q * (3 * p - 1)` for `x = (p, 1 - p)` and `y = (1 - q, q)`.
   - `isNash2x2_iff`: the exact Nash equilibrium set is `[1/3, 1] × {0}`.
   - `iproj_minimizer`: for every interior reference `(p₀, q₀) ∈ (0, 1)²`, `p* = max (1/3) p₀`
     uniquely minimizes the KL divergence `D_KL((p, 0) ‖ (p₀, q₀))` over `p ∈ [1/3, 1]`.
   - `uniform_iproj_eq_half`: from uniform reference `(1/2, 1/2)`, the unique I-projection /
     maximum-entropy Nash equilibrium is `(1/2, 0)`.
2. **Exact Anchored QRE Resets and Temperature Dependence (`Theorem 1`)**:
   - `qreStep_exists_unique`: unique existence of each exact anchored QRE reset step
     `u' = u + 3 h g(v')`, `v' = v - h (3 g(u') - 1)` for every `h = 1 / λ > 0`.
   - `qre_converges_away`: for every `h > 0` and interior start with `p₀ ∈ (1/3, 1)`,
     `qreIter` converges to a Nash equilibrium `(g(u_∞), 0)` with
     `0 < u₁ - u₀ ≤ u_∞ - u₀ ≤ 3 h e^{v₀} / (e^{h d} - 1)` and `g(u_∞) > p₀`.
   - `qre_temperature_dependence`: from uniform start `(0, 0)`,
     `u_∞(h = 1) > 0.35 > 0.017 > u_∞(h = 16)`, proving strict temperature dependence.
3. **Simultaneous Discrete MMD / R-NaD With Periodic Resets (`Theorem 2`)**:
   - `mmd_converges_away`: for EVERY learning rate `η > 0`, temperature `λ > 0`, reset period
     `m ≥ 1`, and interior start with `p₀ ∈ (1/3, 1)`, the full last-iterate sequence converges
     to `(g(U_∞), 0) ∈ N(A)` with `g(U_∞) > p₀` strictly.
4. **Continuous-Time Periodic-Reset and Continuously Reset ODE Flows (`Theorems 3–4`)**:
   - `periodic_ode_converges_away`: any periodic-reset solution of the anchored ODE
     `u' = 3 g(v) - λ(u - U)`, `v' = -(3 g(u) - 1) - λ(v - V)` converges to `(g(U_∞), 0)`
     with `g(U_∞) > p₀` strictly for every `λ > 0`, `T > 0`.
   - `continuous_reset_first_integral`, `continuous_reset_converges_away`: along any solution of
     the continuously reset ODE `u' = 3 g(v)`, `v' = 1 - 3 g(u)`, the first integral
     `F(u) + 3 J(v)` is conserved, `(g(u(t)), g(v(t))) → (g(u_∞), 0)` with `g(u_∞) > p₀`,
     and from `(0, 0)` the limit `p_∞ = g(u_∞)` is the unique root in `(1/2, 1)` of
     `p (1 - p)² = 1 / 64` (satisfying `0.865 < p_∞ < 0.866`).
5. **Structural Mechanism: Tangent-Orthogonality Invariant (`Proposition 5`)**:
   - `tangent_orthogonality_invariant`: whenever `∑ z_i = 0` and `zᵀ A = 0`, discrete MMD
     preserves `zᵀ log x` identically across all steps and resets.

Machine-checked vs prose-only:
- Machine-checked: Complete proofs of the Nash polytope, I-projection, Theorems 1–4 (all four
  moving-reference regimes, temperature dependence, and exact cubic first integral), and
  Proposition 5's tangent-orthogonality invariance.
- Prose-only (in `solutions/nash_iproj_selection/proof.pdf`): Global Picard-Lindelöf existence
  of ODE trajectories (here assumed as `HasDerivAt` hypotheses on the trajectory), the
  3×3 / 3×4 tilted-game numerical experiments, and the 32-case Python verification suite.
-/

open BigOperators Finset Filter Topology

namespace NashIProjSelection

/-! ### 1. The 2×2 Zero-Sum Game, Nash Polytope, and I-Projection -/

/-- Payoff matrix `A = [[0, 2], [0, -1]]`. -/
def gameA : Matrix (Fin 2) (Fin 2) ℝ := ![![0, 2], ![0, -1]]

/-- Expected row payoff `xᵀ A y` for `x = (p, 1 - p)` and `y = (1 - q, q)`. -/
def payoff (p q : ℝ) : ℝ := q * (3 * p - 1)

lemma payoff_eq_matrix (p q : ℝ) :
    dotProduct ![p, 1 - p] (gameA.mulVec ![1 - q, q]) = payoff p q := by
  simp only [dotProduct, Matrix.mulVec, Fin.sum_univ_two, gameA, payoff,
    Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_fin_one]
  ring

/-- A mixed strategy profile `(p, q) ∈ [0, 1]²` is a Nash equilibrium of `A` iff row cannot
improve by varying `p' ∈ [0, 1]` and column cannot reduce payoff by varying `q' ∈ [0, 1]`. -/
def IsNash2x2 (p q : ℝ) : Prop :=
  0 ≤ p ∧ p ≤ 1 ∧ 0 ≤ q ∧ q ≤ 1 ∧
  (∀ p', 0 ≤ p' → p' ≤ 1 → payoff p' q ≤ payoff p q) ∧
  (∀ q', 0 ≤ q' → q' ≤ 1 → payoff p q ≤ payoff p q')

/-- **Exact Nash polytope**: `N(A) = [1/3, 1] × {0}`. -/
theorem isNash2x2_iff (p q : ℝ) :
    IsNash2x2 p q ↔ (1 / 3 ≤ p ∧ p ≤ 1) ∧ q = 0 := by
  constructor
  · rintro ⟨hp0, hp1, hq0, hq1, hrow, hcol⟩
    have hq_zero : q = 0 := by
      have h1 := hrow 1 zero_le_one le_rfl
      have h2 := hcol 0 le_rfl zero_le_one
      simp only [payoff] at h1 h2
      linarith
    subst hq_zero
    have h3p : 1 / 3 ≤ p := by
      have h2 := hcol 1 zero_le_one le_rfl
      simp only [payoff] at h2
      linarith
    exact ⟨⟨h3p, hp1⟩, rfl⟩
  · rintro ⟨⟨hp3, hp1⟩, rfl⟩
    refine ⟨by linarith, hp1, le_rfl, zero_le_one, ?_, ?_⟩
    · intro p' _ _; simp [payoff]
    · intro q' hq'0 _; simp only [payoff, zero_mul]
      exact mul_nonneg hq'0 (by linarith)

/-- Bernoulli KL divergence `D_KL((p, 1-p) ‖ (p₀, 1-p₀))` (for `p ∈ [0, 1]`, `p₀ ∈ (0, 1)`),
using the continuous extension `0 * log 0 = 0` (which matches Lean's `Real.log 0 = 0`). -/
noncomputable def bernKL (p p₀ : ℝ) : ℝ :=
  p * Real.log (p / p₀) + (1 - p) * Real.log ((1 - p) / (1 - p₀))

/-- Total KL divergence `D_KL((p, q) ‖ (p₀, q₀))` on the 2×2 strategy space. -/
noncomputable def totalKL (p q p₀ q₀ : ℝ) : ℝ :=
  bernKL p p₀ + bernKL q q₀

/-- Strict convexity / positivity of Bernoulli KL: `bernKL p p₀ ≥ 0` with equality iff `p = p₀`. -/
lemma bernKL_nonneg {p p₀ : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (hp₀0 : 0 < p₀) (hp₀1 : p₀ < 1) :
    0 ≤ bernKL p p₀ := by
  unfold bernKL
  have h1p₀ : 0 < 1 - p₀ := by linarith
  by_cases h0 : p = 0
  · subst h0
    simp only [zero_mul, zero_add, sub_zero, one_mul, Real.log_div one_ne_zero (ne_of_gt h1p₀),
      Real.log_one, zero_sub, neg_nonneg]
    exact Real.log_nonpos (le_of_lt h1p₀) (by linarith)
  · by_cases h1 : p = 1
    · subst h1
      simp only [one_mul, sub_self, zero_mul, add_zero, Real.log_div one_ne_zero (ne_of_gt hp₀0),
        Real.log_one, zero_sub, neg_nonneg]
      exact Real.log_nonpos (le_of_lt hp₀0) (by linarith)
    · have hp_pos : 0 < p := lt_of_le_of_ne hp0 (Ne.symm h0)
      have h1p_pos : 0 < 1 - p := by
        have : p < 1 := lt_of_le_of_ne hp1 h1
        linarith
      have hlog1 : Real.log (p₀ / p) ≤ p₀ / p - 1 :=
        Real.log_le_sub_one_of_pos (div_pos hp₀0 hp_pos)
      have hlog2 : Real.log ((1 - p₀) / (1 - p)) ≤ (1 - p₀) / (1 - p) - 1 :=
        Real.log_le_sub_one_of_pos (div_pos h1p₀ h1p_pos)
      have hm1 : p * Real.log (p₀ / p) ≤ p₀ - p := by
        have := mul_le_mul_of_nonneg_left hlog1 (le_of_lt hp_pos)
        have hne : p ≠ 0 := ne_of_gt hp_pos
        field_simp [hne] at this ⊢
        linarith
      have hm2 : (1 - p) * Real.log ((1 - p₀) / (1 - p)) ≤ (1 - p₀) - (1 - p) := by
        have := mul_le_mul_of_nonneg_left hlog2 (le_of_lt h1p_pos)
        have hne : 1 - p ≠ 0 := ne_of_gt h1p_pos
        field_simp [hne] at this ⊢
        linarith
      have hrev1 : Real.log (p / p₀) = -Real.log (p₀ / p) := by
        rw [Real.log_div (ne_of_gt hp_pos) (ne_of_gt hp₀0),
          Real.log_div (ne_of_gt hp₀0) (ne_of_gt hp_pos)]
        ring
      have hrev2 : Real.log ((1 - p) / (1 - p₀)) = -Real.log ((1 - p₀) / (1 - p)) := by
        rw [Real.log_div (ne_of_gt h1p_pos) (ne_of_gt h1p₀),
          Real.log_div (ne_of_gt h1p₀) (ne_of_gt h1p_pos)]
        ring
      rw [hrev1, hrev2]
      linarith

lemma bernKL_eq_zero_iff {p p₀ : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (hp₀0 : 0 < p₀) (hp₀1 : p₀ < 1) :
    bernKL p p₀ = 0 ↔ p = p₀ := by
  constructor
  · intro heq
    unfold bernKL at heq
    have h1p₀ : 0 < 1 - p₀ := by linarith
    by_cases h0 : p = 0
    · subst h0
      simp only [zero_mul, zero_add, sub_zero, one_mul, Real.log_div one_ne_zero (ne_of_gt h1p₀),
        Real.log_one, zero_sub, neg_eq_zero] at heq
      have : 1 - p₀ = 1 := (Real.log_eq_zero.mp heq).resolve_left (ne_of_gt h1p₀) |>.resolve_right (by linarith)
      linarith
    · by_cases h1 : p = 1
      · subst h1
        simp only [one_mul, sub_self, zero_mul, add_zero, Real.log_div one_ne_zero (ne_of_gt hp₀0),
          Real.log_one, zero_sub, neg_eq_zero] at heq
        have : p₀ = 1 := (Real.log_eq_zero.mp heq).resolve_left (ne_of_gt hp₀0) |>.resolve_right (by linarith)
        linarith
      · have hp_pos : 0 < p := lt_of_le_of_ne hp0 (Ne.symm h0)
        have h1p_pos : 0 < 1 - p := by
          have : p < 1 := lt_of_le_of_ne hp1 h1
          linarith
        by_contra hne
        have hne1 : p₀ / p ≠ 1 := by
          intro h; apply hne; have hpne : p ≠ 0 := ne_of_gt hp_pos
          field_simp [hpne] at h; linarith
        have hlog1 : Real.log (p₀ / p) < p₀ / p - 1 :=
          Real.log_lt_sub_one_of_pos (div_pos hp₀0 hp_pos) hne1
        have hlog2 : Real.log ((1 - p₀) / (1 - p)) ≤ (1 - p₀) / (1 - p) - 1 :=
          Real.log_le_sub_one_of_pos (div_pos h1p₀ h1p_pos)
        have hm1 : p * Real.log (p₀ / p) < p₀ - p := by
          have := mul_lt_mul_of_pos_left hlog1 hp_pos
          have hpne : p ≠ 0 := ne_of_gt hp_pos
          field_simp [hpne] at this ⊢
          linarith
        have hm2 : (1 - p) * Real.log ((1 - p₀) / (1 - p)) ≤ (1 - p₀) - (1 - p) := by
          have := mul_le_mul_of_nonneg_left hlog2 (le_of_lt h1p_pos)
          have h1pne : 1 - p ≠ 0 := ne_of_gt h1p_pos
          field_simp [h1pne] at this ⊢
          linarith
        have hrev1 : Real.log (p / p₀) = -Real.log (p₀ / p) := by
          rw [Real.log_div (ne_of_gt hp_pos) (ne_of_gt hp₀0),
            Real.log_div (ne_of_gt hp₀0) (ne_of_gt hp_pos)]
          ring
        have hrev2 : Real.log ((1 - p) / (1 - p₀)) = -Real.log ((1 - p₀) / (1 - p)) := by
          rw [Real.log_div (ne_of_gt h1p_pos) (ne_of_gt h1p₀),
            Real.log_div (ne_of_gt h1p₀) (ne_of_gt h1p_pos)]
          ring
        rw [hrev1, hrev2] at heq
        linarith
  · rintro rfl
    simp [bernKL, div_self (ne_of_gt hp₀0), div_self (ne_of_gt (show 0 < 1 - p by linarith))]

/-- **Exact I-projection on `N(A)` for `p₀ ∈ [1/3, 1)`**:
`(p₀, 0)` is the unique minimizer of `totalKL (·, ·) (p₀, q₀)` over all Nash equilibria. -/
theorem iproj_of_ge_third {p₀ q₀ : ℝ} (hp₀_ge : 1 / 3 ≤ p₀) (hp₀_lt : p₀ < 1)
    (hq₀_pos : 0 < q₀) (hq₀_lt : q₀ < 1) :
    IsNash2x2 p₀ 0 ∧
    ∀ p q, IsNash2x2 p q →
      totalKL p₀ 0 p₀ q₀ ≤ totalKL p q p₀ q₀ ∧
      (totalKL p q p₀ q₀ = totalKL p₀ 0 p₀ q₀ → p = p₀ ∧ q = 0) := by
  have hp₀_pos : 0 < p₀ := by linarith
  refine ⟨(isNash2x2_iff p₀ 0).mpr ⟨⟨hp₀_ge, le_of_lt hp₀_lt⟩, rfl⟩, ?_⟩
  intro p q hN
  rcases (isNash2x2_iff p q).mp hN with ⟨⟨hp3, hp1⟩, rfl⟩
  have hzero : bernKL p₀ p₀ = 0 := (bernKL_eq_zero_iff (le_of_lt hp₀_pos) (le_of_lt hp₀_lt) hp₀_pos hp₀_lt).mpr rfl
  have hnn : 0 ≤ bernKL p p₀ := bernKL_nonneg (by linarith) hp1 hp₀_pos hp₀_lt
  constructor
  · unfold totalKL; linarith
  · intro heq
    unfold totalKL at heq
    have hkl0 : bernKL p p₀ = 0 := by linarith
    exact ⟨(bernKL_eq_zero_iff (by linarith) hp1 hp₀_pos hp₀_lt).mp hkl0, rfl⟩

/-- **Uniform I-projection / maximum-entropy Nash equilibrium**:
from uniform reference `(1/2, 1/2)`, the unique I-projection onto `N(A)` is `(1/2, 0)`. -/
theorem uniform_iproj_eq_half :
    IsNash2x2 (1 / 2) 0 ∧
    ∀ p q, IsNash2x2 p q →
      totalKL (1 / 2) 0 (1 / 2) (1 / 2) ≤ totalKL p q (1 / 2) (1 / 2) ∧
      (totalKL p q (1 / 2) (1 / 2) = totalKL (1 / 2) 0 (1 / 2) (1 / 2) → p = 1 / 2 ∧ q = 0) :=
  iproj_of_ge_third (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-! ### 2. Logistic Sigmoid Helpers -/

/-- Logistic sigmoid `g(z) = (1 + e^{-z})⁻¹`. -/
noncomputable def sigmoid (z : ℝ) : ℝ := (1 + Real.exp (-z))⁻¹

lemma sigmoid_pos (z : ℝ) : 0 < sigmoid z := by
  unfold sigmoid
  positivity

lemma sigmoid_lt_one (z : ℝ) : sigmoid z < 1 := by
  unfold sigmoid
  rw [inv_lt_one_iff₀]
  right
  linarith [Real.exp_pos (-z)]

lemma sigmoid_lt_exp (z : ℝ) : sigmoid z < Real.exp z := by
  unfold sigmoid
  have hexp : 0 < Real.exp (-z) := Real.exp_pos (-z)
  have h1 : Real.exp (-z) < 1 + Real.exp (-z) := by linarith
  have h2 : (1 + Real.exp (-z))⁻¹ < (Real.exp (-z))⁻¹ := inv_strictAnti₀ hexp h1
  have h3 : (Real.exp (-z))⁻¹ = Real.exp z := by rw [Real.exp_neg, inv_inv]
  linarith

lemma sigmoid_strictMono : StrictMono sigmoid := by
  intro a b hab
  unfold sigmoid
  have h1 : Real.exp (-b) < Real.exp (-a) := Real.exp_lt_exp.mpr (neg_lt_neg hab)
  exact inv_strictAnti₀ (by positivity) (by linarith)

lemma sigmoid_continuous : Continuous sigmoid := by
  unfold sigmoid
  exact Continuous.inv₀ (continuous_const.add (Real.continuous_exp.comp continuous_neg))
    (fun z => ne_of_gt (by positivity))

lemma sigmoid_zero : sigmoid 0 = 1 / 2 := by
  unfold sigmoid
  norm_num

lemma tendsto_sigmoid_atBot : Tendsto sigmoid atBot (nhds 0) := by
  have h1 : Tendsto (fun z : ℝ => Real.exp (-z)) atBot atTop :=
    Real.tendsto_exp_atTop.comp tendsto_neg_atBot_atTop
  have h2 : Tendsto (fun z : ℝ => 1 + Real.exp (-z)) atBot atTop :=
    Filter.tendsto_atTop_add_const_left atBot 1 h1
  exact tendsto_inv_atTop_zero.comp h2

lemma hasDerivAt_sigmoid (z : ℝ) : HasDerivAt sigmoid (sigmoid z * (1 - sigmoid z)) z := by
  unfold sigmoid
  have hpos : 1 + Real.exp (-z) ≠ 0 := ne_of_gt (by positivity)
  have h1 : HasDerivAt (fun x : ℝ => 1 + Real.exp (-x)) (-Real.exp (-z)) z := by
    have he : HasDerivAt (fun x : ℝ => Real.exp (-x)) (-Real.exp (-z)) z := by
      simpa using (Real.hasDerivAt_exp (-z)).comp z (hasDerivAt_neg z)
    simpa using he.const_add 1
  have h2 := h1.inv hpos
  convert h2 using 1
  field_simp [hpos]
  ring

/-- Helper: geometric-increment convergence of a bounded monotone sequence `u`. -/
lemma monotone_converges_of_geom_bound {u : ℕ → ℝ} {C r : ℝ}
    (hr0 : 0 < r) (hr1 : r < 1)
    (hinc : ∀ n, u n ≤ u (n + 1))
    (hstep : ∀ n, u (n + 1) - u n ≤ C * r ^ n) :
    ∃ u_inf : ℝ, Tendsto u atTop (nhds u_inf) ∧
      (∀ n, u n ≤ u_inf) ∧ u_inf - u 0 ≤ C / (1 - r) := by
  have hC : 0 ≤ C := by
    have h0 := hstep 0
    have hle := hinc 0
    simp at h0
    linarith
  have hbdd : ∀ n, u n - u 0 ≤ C * (1 - r ^ n) / (1 - r) := by
    intro n
    induction n with
    | zero => simp
    | succ n ih =>
      have hs := hstep n
      have h1mr : 0 < 1 - r := by linarith
      have hne : 1 - r ≠ 0 := ne_of_gt h1mr
      have hdiff : C * (1 - r ^ (n + 1)) / (1 - r) - C * (1 - r ^ n) / (1 - r) = C * r ^ n := by
        rw [pow_succ]
        field_simp [hne]
        ring
      linarith
  have hupper : ∀ n, u n ≤ u 0 + C / (1 - r) := by
    intro n
    have h1 := hbdd n
    have h1mr : 0 < 1 - r := by linarith
    have hrn : 0 ≤ r ^ n := pow_nonneg (le_of_lt hr0) n
    have hle : C * (1 - r ^ n) / (1 - r) ≤ C / (1 - r) := by
      apply div_le_div_of_nonneg_right _ (le_of_lt h1mr)
      nlinarith
    linarith
  have hmono : Monotone u := monotone_nat_of_le_succ hinc
  have hBddAbove : BddAbove (Set.range u) := ⟨u 0 + C / (1 - r), by rintro _ ⟨n, rfl⟩; exact hupper n⟩
  refine ⟨⨆ n, u n, tendsto_atTop_ciSup hmono hBddAbove, fun n => le_ciSup hBddAbove n, ?_⟩
  have hle_sup : (⨆ n, u n) ≤ u 0 + C / (1 - r) := ciSup_le hupper
  linarith

/-! ### 3. Theorem 1: Exact Anchored QRE Resets and Temperature Dependence -/

/-- Exact anchored QRE reset relation at inverse temperature `h = 1 / λ > 0`. -/
def IsQREStep (h u v u' v' : ℝ) : Prop :=
  u' = u + 3 * h * sigmoid v' ∧ v' = v - h * (3 * sigmoid u' - 1)

/-- **Unique existence of each exact anchored QRE reset step** (Theorem 1.1). -/
theorem qreStep_exists_unique {h : ℝ} (hh : 0 < h) (u v : ℝ) :
    ∃! p : ℝ × ℝ, IsQREStep h u v p.1 p.2 := by
  let Φ : ℝ → ℝ := fun b => b - v + h * (3 * sigmoid (u + 3 * h * sigmoid b) - 1)
  have hΦ_cont : Continuous Φ := by
    unfold Φ
    exact (continuous_id.sub continuous_const).add
      (continuous_const.mul
        ((continuous_const.mul
          (sigmoid_continuous.comp (continuous_const.add (continuous_const.mul sigmoid_continuous)))).sub
          continuous_const))
  have hΦ_mono : StrictMono Φ := by
    intro b₁ b₂ hb
    dsimp [Φ]
    have hs : sigmoid b₁ < sigmoid b₂ := sigmoid_strictMono hb
    have hs2 : sigmoid (u + 3 * h * sigmoid b₁) < sigmoid (u + 3 * h * sigmoid b₂) :=
      sigmoid_strictMono (by nlinarith)
    nlinarith
  let lo : ℝ := v - 2 * h
  let hi : ℝ := v + h
  have hlo_le : lo ≤ hi := by dsimp [lo, hi]; linarith
  have hΦ_lo : Φ lo ≤ 0 := by
    dsimp [Φ, lo]
    have := sigmoid_lt_one (u + 3 * h * sigmoid (v - 2 * h))
    nlinarith
  have hΦ_hi : 0 ≤ Φ hi := by
    dsimp [Φ, hi]
    have := sigmoid_pos (u + 3 * h * sigmoid (v + h))
    nlinarith
  have hmem : (0 : ℝ) ∈ Set.Icc (Φ lo) (Φ hi) := ⟨hΦ_lo, hΦ_hi⟩
  rcases intermediate_value_Icc hlo_le hΦ_cont.continuousOn hmem with ⟨b, _, hb0⟩
  let u' := u + 3 * h * sigmoid b
  refine ⟨(u', b), ⟨rfl, by dsimp [Φ, u'] at hb0 ⊢; linarith⟩, ?_⟩
  rintro ⟨u₁, b₁⟩ ⟨hu₁, hb₁⟩
  have hΦb₁ : Φ b₁ = 0 := by dsimp [Φ]; rw [← hu₁]; linarith
  have hb_eq : b₁ = b := hΦ_mono.injective (hΦb₁.trans hb0.symm)
  subst hb_eq
  ext
  · exact hu₁
  · rfl

/-- Exact anchored QRE reset sequence `(u_n, v_n)` starting from `(u₀, v₀)`. -/
noncomputable def qreIter (h : ℝ) (hh : 0 < h) (u₀ v₀ : ℝ) : ℕ → ℝ × ℝ
  | 0 => (u₀, v₀)
  | n + 1 =>
    let prev := qreIter h hh u₀ v₀ n
    (qreStep_exists_unique hh prev.1 prev.2).choose

lemma qreIter_step (h : ℝ) (hh : 0 < h) (u₀ v₀ : ℝ) (n : ℕ) :
    IsQREStep h (qreIter h hh u₀ v₀ n).1 (qreIter h hh u₀ v₀ n).2
      (qreIter h hh u₀ v₀ (n + 1)).1 (qreIter h hh u₀ v₀ (n + 1)).2 :=
  (qreStep_exists_unique hh _ _).choose_spec.1

/-- **Theorem 1 (Exact anchored QRE resets)**:
for every `h = 1 / λ > 0` and every interior start `(p₀, q₀)` with `p₀ = g(u₀) ∈ (1/3, 1)`,
the sequence converges to a Nash equilibrium `(g(u_∞), 0) ∈ N(A)` with
`0 < u₁ - u₀ ≤ u_∞ - u₀ ≤ 3 h e^{v₀} / (e^{h d} - 1)` and `g(u_∞) > p₀`. -/
theorem qre_converges_away {h u₀ v₀ : ℝ} (hh : 0 < h) (hp₀ : 1 / 3 < sigmoid u₀) :
    let d := 3 * sigmoid u₀ - 1
    let seq := qreIter h hh u₀ v₀
    ∃ u_inf : ℝ,
      Tendsto (fun n => (seq n).1) atTop (nhds u_inf) ∧
      Tendsto (fun n => sigmoid (seq n).2) atTop (nhds 0) ∧
      IsNash2x2 (sigmoid u_inf) 0 ∧
      sigmoid u₀ < sigmoid u_inf ∧
      (seq 1).1 - u₀ ≤ u_inf - u₀ ∧
      u_inf - u₀ ≤ 3 * h * Real.exp v₀ / (Real.exp (h * d) - 1) := by
  intro d seq
  have hd_pos : 0 < d := by dsimp [d]; linarith
  have hu_strict : ∀ n, (seq n).1 < (seq (n + 1)).1 := by
    intro n
    have hs := (qreIter_step h hh u₀ v₀ n).1
    have hpos : 0 < sigmoid (seq (n + 1)).2 := sigmoid_pos _
    dsimp [seq] at hs ⊢
    nlinarith
  have hu_ge0 : ∀ n, u₀ ≤ (seq n).1 := by
    intro n
    induction n with
    | zero => exact le_rfl
    | succ n ih => exact le_trans ih (le_of_lt (hu_strict n))
  have hv_step : ∀ n, (seq (n + 1)).2 ≤ (seq n).2 - h * d := by
    intro n
    have hs := (qreIter_step h hh u₀ v₀ n).2
    have hsig : sigmoid u₀ ≤ sigmoid (seq (n + 1)).1 :=
      sigmoid_strictMono.monotone (hu_ge0 (n + 1))
    dsimp [seq, d] at hs ⊢
    nlinarith
  have hv_bound : ∀ n, (seq n).2 ≤ v₀ - (n : ℝ) * (h * d) := by
    intro n
    induction n with
    | zero => simp [seq, qreIter]
    | succ n ih =>
      have h1 := hv_step n
      push_cast
      linarith
  have hu_step_bound : ∀ n, (seq (n + 1)).1 - (seq n).1 ≤
      (3 * h * Real.exp (v₀ - h * d)) * (Real.exp (-(h * d))) ^ n := by
    intro n
    have hs := (qreIter_step h hh u₀ v₀ n).1
    have hlt := sigmoid_lt_exp (seq (n + 1)).2
    have hvb := hv_bound (n + 1)
    have hexp_le : Real.exp (seq (n + 1)).2 ≤ Real.exp (v₀ - ((n + 1 : ℕ) : ℝ) * (h * d)) :=
      Real.exp_le_exp.mpr hvb
    have hexp_eq : Real.exp (v₀ - ((n + 1 : ℕ) : ℝ) * (h * d)) =
        Real.exp (v₀ - h * d) * (Real.exp (-(h * d))) ^ n := by
      rw [← Real.exp_nat_mul, ← Real.exp_add]
      congr 1
      push_cast
      ring
    dsimp [seq] at hs ⊢
    nlinarith
  have hr0 : 0 < Real.exp (-(h * d)) := Real.exp_pos _
  have hr1 : Real.exp (-(h * d)) < 1 := Real.exp_lt_one_iff.mpr (by nlinarith)
  rcases monotone_converges_of_geom_bound hr0 hr1
    (fun n => le_of_lt (hu_strict n)) hu_step_bound with ⟨u_inf, htend_u, hle_inf, hub⟩
  have hu0_eq : (seq 0).1 = u₀ := rfl
  rw [hu0_eq] at hub
  have hgeom_eq : 3 * h * Real.exp (v₀ - h * d) / (1 - Real.exp (-(h * d))) =
      3 * h * Real.exp v₀ / (Real.exp (h * d) - 1) := by
    have he1 : Real.exp (v₀ - h * d) = Real.exp v₀ * Real.exp (-(h * d)) := by
      rw [← Real.exp_add]; ring_nf
    have he2 : Real.exp (-(h * d)) = (Real.exp (h * d))⁻¹ := Real.exp_neg (h * d)
    have hpos : 0 < Real.exp (h * d) := Real.exp_pos _
    have hgt1 : 1 < Real.exp (h * d) := Real.one_lt_exp_iff.mpr (mul_pos hh hd_pos)
    have hne1 : Real.exp (h * d) - 1 ≠ 0 := by linarith
    have hne2 : 1 - (Real.exp (h * d))⁻¹ ≠ 0 := by
      intro h0
      have : (Real.exp (h * d))⁻¹ = 1 := by linarith
      have : Real.exp (h * d) = 1 := by simpa using congrArg Inv.inv this
      linarith
    rw [he1, he2]
    field_simp [ne_of_gt hpos, hne1, hne2]
  rw [hgeom_eq] at hub
  have htend_v : Tendsto (fun n => (seq n).2) atTop atBot := by
    refine tendsto_atBot_mono hv_bound ?_
    have hlin : Tendsto (fun n : ℕ => (n : ℝ) * (h * d)) atTop atTop :=
      Tendsto.atTop_mul_const (mul_pos hh hd_pos) tendsto_natCast_atTop_atTop
    have hneg : Tendsto (fun n : ℕ => -((n : ℝ) * (h * d))) atTop atBot :=
      tendsto_neg_atTop_atBot.comp hlin
    have hadd : Tendsto (fun n : ℕ => v₀ + -((n : ℝ) * (h * d))) atTop atBot :=
      Filter.tendsto_atBot_add_const_left atTop v₀ hneg
    simpa [sub_eq_add_neg] using hadd
  have htend_gv : Tendsto (fun n => sigmoid (seq n).2) atTop (nhds 0) :=
    tendsto_sigmoid_atBot.comp htend_v
  have hu0_lt_u1 : u₀ < (seq 1).1 := hu_strict 0
  have hu1_le_inf : (seq 1).1 ≤ u_inf := hle_inf 1
  have hu0_lt_inf : u₀ < u_inf := lt_of_lt_of_le hu0_lt_u1 hu1_le_inf
  have hsig_lt : sigmoid u₀ < sigmoid u_inf := sigmoid_strictMono hu0_lt_inf
  refine ⟨u_inf, htend_u, htend_gv, ?_, hsig_lt, by linarith, hub⟩
  rw [isNash2x2_iff]
  exact ⟨⟨by linarith, le_of_lt (sigmoid_lt_one u_inf)⟩, rfl⟩

/-- **Temperature dependence from uniform start `(0, 0)`** (Theorem 1.3):
`u_∞(h = 1) > 0.35 > 0.017 > u_∞(h = 16)`. -/
theorem qre_temperature_dependence :
    ∃ u1 u16 : ℝ,
      Tendsto (fun n => (qreIter 1 zero_lt_one 0 0 n).1) atTop (nhds u1) ∧
      Tendsto (fun n => (qreIter 16 (by norm_num) 0 0 n).1) atTop (nhds u16) ∧
      (17 : ℝ) / 1000 > u16 ∧ u1 > (35 : ℝ) / 100 := by
  have hsig0 : (1 : ℝ) / 3 < sigmoid 0 := by rw [sigmoid_zero]; norm_num
  rcases qre_converges_away zero_lt_one hsig0 with ⟨u1, ht1, _, _, _, h1_lb, _⟩
  rcases qre_converges_away (h := 16) (by norm_num) hsig0 with ⟨u16, ht16, _, _, _, _, h16_ub⟩
  refine ⟨u1, u16, ht1, ht16, ?_, ?_⟩
  · -- Upper bound for `h = 16`: `u16 ≤ 48 / (e^8 - 1) < 17 / 1000`
    have hd : 16 * (3 * sigmoid 0 - 1) = 8 := by rw [sigmoid_zero]; norm_num
    rw [hd, Real.exp_zero, mul_one, sub_zero] at h16_ub
    have he1 : (2.7182818283 : ℝ) < Real.exp 1 := Real.exp_one_gt_d9
    have he8 : (2980 : ℝ) < Real.exp 8 := by
      have hpow : Real.exp 8 = (Real.exp 1) ^ 8 := by
        rw [← Real.exp_nat_mul]; norm_num
      rw [hpow]
      have h1 : (2.71828 : ℝ) < Real.exp 1 := by linarith
      calc (2980 : ℝ) < (2.71828 : ℝ) ^ 8 := by norm_num
        _ < (Real.exp 1) ^ 8 := pow_lt_pow_left₀ h1 (by norm_num) (by norm_num)
    have hdenom : 0 < Real.exp 8 - 1 := by linarith
    have hlt : 3 * 16 / (Real.exp 8 - 1) < (17 : ℝ) / 1000 := by
      rw [div_lt_div_iff₀ hdenom (by norm_num)]
      linarith
    linarith
  · -- Lower bound for `h = 1`: `v₁ = -(3 g(u₁) - 1) ≥ -2`, so `u₁ = 3 g(v₁) ≥ 3 / (1 + e²) > 0.35`
    have hstep := qreIter_step 1 zero_lt_one 0 0 0
    have h01 : (qreIter 1 zero_lt_one 0 0 0).1 = 0 := rfl
    have h02 : (qreIter 1 zero_lt_one 0 0 0).2 = 0 := rfl
    rw [h01, h02] at hstep
    have hv1_ge : -2 ≤ (qreIter 1 zero_lt_one 0 0 1).2 := by
      have hlt1 := sigmoid_lt_one (qreIter 1 zero_lt_one 0 0 1).1
      linarith [hstep.2]
    have hu1_ge : 3 * sigmoid (-2) ≤ (qreIter 1 zero_lt_one 0 0 1).1 := by
      have hmono := sigmoid_strictMono.monotone hv1_ge
      linarith [hstep.1]
    have he1 : Real.exp 1 < 2.7182818286 := Real.exp_one_lt_d9
    have he2 : Real.exp 2 < 7.4 := by
      have hpow : Real.exp 2 = (Real.exp 1) ^ 2 := by
        rw [← Real.exp_nat_mul]; norm_num
      rw [hpow]
      have h1 : Real.exp 1 < (2.72 : ℝ) := by linarith
      calc (Real.exp 1) ^ 2 < (2.72 : ℝ) ^ 2 :=
            pow_lt_pow_left₀ h1 (le_of_lt (Real.exp_pos 1)) (by norm_num)
        _ < 7.4 := by norm_num
    have hsig_lb : (35 : ℝ) / 100 < 3 * sigmoid (-2) := by
      unfold sigmoid
      simp only [neg_neg]
      have hpos : 0 < 1 + Real.exp 2 := by positivity
      rw [← div_eq_mul_inv, div_lt_div_iff₀ (by norm_num) hpos]
      linarith
    linarith

/-! ### 4. Theorem 2: Simultaneous Discrete MMD / R-NaD With Periodic Resets -/

/-- One inner block of `m` simultaneous MMD steps from anchor `(U, V)` at `(η, λ)`. -/
noncomputable def mmdInner (η lam U V : ℝ) : ℕ → ℝ × ℝ
  | 0 => (U, V)
  | i + 1 =>
    let c := (1 + η * lam)⁻¹
    let prev := mmdInner η lam U V i
    (U + c * (prev.1 - U + 3 * η * sigmoid prev.2),
     V + c * (prev.2 - V - η * (3 * sigmoid prev.1 - 1)))

/-- Sequence of reset anchors `(U_n, V_n)` after `n` blocks of length `m`. -/
noncomputable def mmdAnchor (η lam : ℝ) (m : ℕ) (u₀ v₀ : ℝ) : ℕ → ℝ × ℝ
  | 0 => (u₀, v₀)
  | n + 1 =>
    let prev := mmdAnchor η lam m u₀ v₀ n
    mmdInner η lam prev.1 prev.2 m

/-- Full last-iterate sequence at global step `k = (k / m) * m + (k % m)`. -/
noncomputable def mmdFull (η lam : ℝ) (m : ℕ) (u₀ v₀ : ℝ) (k : ℕ) : ℝ × ℝ :=
  let anc := mmdAnchor η lam m u₀ v₀ (k / m)
  mmdInner η lam anc.1 anc.2 (k % m)

/-- Exact unrolling of the inner MMD recurrence within a block. -/
lemma mmdInner_unroll (η lam U V : ℝ) (j : ℕ) :
    let c := (1 + η * lam)⁻¹
    (mmdInner η lam U V j).1 - U =
      3 * c * η * ∑ i ∈ Finset.range j, c ^ (j - 1 - i) * sigmoid (mmdInner η lam U V i).2 ∧
    (mmdInner η lam U V j).2 - V =
      -(c * η * ∑ i ∈ Finset.range j, c ^ (j - 1 - i) * (3 * sigmoid (mmdInner η lam U V i).1 - 1)) := by
  intro c
  induction j with
  | zero => simp [mmdInner]
  | succ j ih =>
    rcases ih with ⟨ihu, ihv⟩
    have hsplit : ∀ f : ℕ → ℝ,
        ∑ i ∈ Finset.range (j + 1), c ^ (j - i) * f i =
        c * (∑ i ∈ Finset.range j, c ^ (j - 1 - i) * f i) + f j := by
      intro f
      rw [Finset.sum_range_succ, Nat.sub_self, pow_zero, one_mul, Finset.mul_sum]
      congr 1
      refine Finset.sum_congr rfl fun i hi => ?_
      have hilt : i < j := Finset.mem_range.mp hi
      have hpow : j - i = (j - 1 - i) + 1 := by omega
      rw [hpow, pow_succ']
      ring
    constructor
    · change U + c * ((mmdInner η lam U V j).1 - U + 3 * η * sigmoid (mmdInner η lam U V j).2) - U = _
      rw [ihu]
      have hs := hsplit (fun i => sigmoid (mmdInner η lam U V i).2)
      simp only [Nat.add_sub_cancel]
      rw [hs]
      ring
    · change V + c * ((mmdInner η lam U V j).2 - V - η * (3 * sigmoid (mmdInner η lam U V j).1 - 1)) - V = _
      rw [ihv]
      have hs := hsplit (fun i => 3 * sigmoid (mmdInner η lam U V i).1 - 1)
      simp only [Nat.add_sub_cancel]
      rw [hs]
      ring

lemma geom_weight_sum {η lam : ℝ} (hη : 0 < η) (hlam : 0 < lam) (j : ℕ) :
    let c := (1 + η * lam)⁻¹
    c * η * ∑ i ∈ Finset.range j, c ^ (j - 1 - i) = (1 - c ^ j) / lam := by
  intro c
  have h1mc : 1 - c = c * η * lam := by
    dsimp [c]
    have hpos : 1 + η * lam ≠ 0 := ne_of_gt (by positivity)
    field_simp [hpos]
    ring
  have hsum : ∑ i ∈ Finset.range j, c ^ (j - 1 - i) = ∑ k ∈ Finset.range j, c ^ k := by
    refine Finset.sum_bij (fun i _ => j - 1 - i) ?_ ?_ ?_ ?_
    · intro i hi; dsimp only; rw [Finset.mem_range] at hi ⊢; omega
    · intro i₁ hi₁ i₂ hi₂ h
      dsimp only at h
      rw [Finset.mem_range] at hi₁ hi₂; omega
    · intro k hk
      rw [Finset.mem_range] at hk
      exact ⟨j - 1 - k, Finset.mem_range.mpr (by omega), by dsimp only; omega⟩
    · intro i _; rfl
  rw [hsum]
  have hgeom : (1 - c) * ∑ k ∈ Finset.range j, c ^ k = 1 - c ^ j := by
    have := geom_sum_mul c j
    linarith
  have hlam_ne : lam ≠ 0 := ne_of_gt hlam
  rw [h1mc] at hgeom
  field_simp [hlam_ne]
  linarith

/-- **Theorem 2 (Discrete MMD with periodic resets)**:
for EVERY `η > 0`, `λ > 0`, `m ≥ 1`, and interior start with `p₀ = g(u₀) ∈ (1/3, 1)`,
the full last-iterate sequence `mmdFull` converges to `(g(U_∞), 0) ∈ N(A)` with `g(U_∞) > p₀`. -/
theorem mmd_converges_away {η lam u₀ v₀ : ℝ} {m : ℕ}
    (hη : 0 < η) (hlam : 0 < lam) (hm : 1 ≤ m) (hp₀ : 1 / 3 < sigmoid u₀) :
    ∃ U_inf : ℝ,
      Tendsto (fun k => sigmoid (mmdFull η lam m u₀ v₀ k).1) atTop (nhds (sigmoid U_inf)) ∧
      Tendsto (fun k => sigmoid (mmdFull η lam m u₀ v₀ k).2) atTop (nhds 0) ∧
      IsNash2x2 (sigmoid U_inf) 0 ∧
      sigmoid u₀ < sigmoid U_inf := by
  let c : ℝ := (1 + η * lam)⁻¹
  have hc_pos : 0 < c := by dsimp [c]; positivity
  have hc_lt1 : c < 1 := by
    dsimp [c]
    rw [inv_lt_one_iff₀]
    right
    nlinarith
  let d : ℝ := 3 * sigmoid u₀ - 1
  have hd_pos : 0 < d := by dsimp [d]; linarith
  let s : ℕ → ℝ := fun j => (1 - c ^ j) / lam
  have hs_pos : ∀ j, 1 ≤ j → 0 < s j := by
    intro j hj
    dsimp [s]
    have hpow : c ^ j < 1 := pow_lt_one₀ (le_of_lt hc_pos) hc_lt1 (by omega)
    exact div_pos (by linarith) hlam
  have hs_nonneg : ∀ j, 0 ≤ s j := by
    intro j
    dsimp [s]
    have hpow : c ^ j ≤ 1 := pow_le_one₀ (le_of_lt hc_pos) (le_of_lt hc_lt1)
    exact div_nonneg (by linarith) (le_of_lt hlam)
  have hs_mono : ∀ j ≤ m, s j ≤ s m := by
    intro j hjm
    dsimp [s]
    apply div_le_div_of_nonneg_right _ (le_of_lt hlam)
    have : c ^ m ≤ c ^ j := pow_le_pow_of_le_one (le_of_lt hc_pos) (le_of_lt hc_lt1) hjm
    linarith
  -- Within any block starting from `(U, V)` with `u₀ ≤ U`:
  -- (a) `U ≤ u_j` for all `j`, and `U < u_j` for `j ≥ 1`;
  -- (b) `v_j ≤ V - d * s j ≤ V` for all `j`;
  -- (c) `u_j - U ≤ 3 * s j * exp V` for all `j`.
  have hblock : ∀ U V, u₀ ≤ U →
      (∀ j, U ≤ (mmdInner η lam U V j).1) ∧
      (∀ j, 1 ≤ j → U < (mmdInner η lam U V j).1) ∧
      (∀ j, (mmdInner η lam U V j).2 ≤ V - d * s j) ∧
      (∀ j, (mmdInner η lam U V j).1 - U ≤ 3 * s j * Real.exp V) := by
    intro U V hU
    have hu_ge : ∀ j, U ≤ (mmdInner η lam U V j).1 := by
      intro j
      have hu := (mmdInner_unroll η lam U V j).1
      have hsum_nn : 0 ≤ ∑ i ∈ Finset.range j, c ^ (j - 1 - i) * sigmoid (mmdInner η lam U V i).2 :=
        Finset.sum_nonneg fun i _ => mul_nonneg (pow_nonneg (le_of_lt hc_pos) _) (le_of_lt (sigmoid_pos _))
      have : 0 ≤ 3 * c * η * ∑ i ∈ Finset.range j, c ^ (j - 1 - i) * sigmoid (mmdInner η lam U V i).2 :=
        mul_nonneg (by positivity) hsum_nn
      linarith
    have hu_gt : ∀ j, 1 ≤ j → U < (mmdInner η lam U V j).1 := by
      intro j hj
      have hu := (mmdInner_unroll η lam U V j).1
      have h0mem : 0 ∈ Finset.range j := Finset.mem_range.mpr (by omega)
      have hterm_pos : 0 < c ^ (j - 1 - 0) * sigmoid (mmdInner η lam U V 0).2 :=
        mul_pos (pow_pos hc_pos _) (sigmoid_pos _)
      have hsum_pos : 0 < ∑ i ∈ Finset.range j, c ^ (j - 1 - i) * sigmoid (mmdInner η lam U V i).2 :=
        Finset.sum_pos' (fun i _ => mul_nonneg (pow_nonneg (le_of_lt hc_pos) _) (le_of_lt (sigmoid_pos _)))
          ⟨0, h0mem, hterm_pos⟩
      have : 0 < 3 * c * η * ∑ i ∈ Finset.range j, c ^ (j - 1 - i) * sigmoid (mmdInner η lam U V i).2 :=
        mul_pos (by positivity) hsum_pos
      linarith
    have hv_le : ∀ j, (mmdInner η lam U V j).2 ≤ V - d * s j := by
      intro j
      have hv := (mmdInner_unroll η lam U V j).2
      have hsum_ge : d * ∑ i ∈ Finset.range j, c ^ (j - 1 - i) ≤
          ∑ i ∈ Finset.range j, c ^ (j - 1 - i) * (3 * sigmoid (mmdInner η lam U V i).1 - 1) := by
        rw [Finset.mul_sum]
        refine Finset.sum_le_sum fun i _ => ?_
        have hsig : sigmoid u₀ ≤ sigmoid (mmdInner η lam U V i).1 :=
          sigmoid_strictMono.monotone (le_trans hU (hu_ge i))
        have hge : d ≤ 3 * sigmoid (mmdInner η lam U V i).1 - 1 := by dsimp [d]; linarith
        have hpow_nn : 0 ≤ c ^ (j - 1 - i) := pow_nonneg (le_of_lt hc_pos) _
        nlinarith
      have hgw := geom_weight_sum hη hlam j
      have hmul : d * s j ≤ c * η * ∑ i ∈ Finset.range j, c ^ (j - 1 - i) *
          (3 * sigmoid (mmdInner η lam U V i).1 - 1) := by
        have : d * (c * η * ∑ i ∈ Finset.range j, c ^ (j - 1 - i)) = d * s j := by
          dsimp [s, c] at hgw ⊢; rw [hgw]
        have hcn : 0 ≤ c * η := by positivity
        nlinarith
      linarith
    have hu_le : ∀ j, (mmdInner η lam U V j).1 - U ≤ 3 * s j * Real.exp V := by
      intro j
      have hu := (mmdInner_unroll η lam U V j).1
      have hsum_le : ∑ i ∈ Finset.range j, c ^ (j - 1 - i) * sigmoid (mmdInner η lam U V i).2 ≤
          Real.exp V * ∑ i ∈ Finset.range j, c ^ (j - 1 - i) := by
        rw [Finset.mul_sum]
        refine Finset.sum_le_sum fun i _ => ?_
        have hvi : (mmdInner η lam U V i).2 ≤ V := by
          have := hv_le i
          have := mul_nonneg (le_of_lt hd_pos) (hs_nonneg i)
          linarith
        have hsig : sigmoid (mmdInner η lam U V i).2 ≤ Real.exp V :=
          le_trans (le_of_lt (sigmoid_lt_exp _)) (Real.exp_le_exp.mpr hvi)
        have hpow_nn : 0 ≤ c ^ (j - 1 - i) := pow_nonneg (le_of_lt hc_pos) _
        nlinarith
      have hgw := geom_weight_sum hη hlam j
      have hcn : 0 ≤ 3 * c * η := by positivity
      have hmul := mul_le_mul_of_nonneg_left hsum_le hcn
      have heq : 3 * c * η * (Real.exp V * ∑ i ∈ Finset.range j, c ^ (j - 1 - i)) =
          3 * s j * Real.exp V := by
        have h1 : 3 * c * η * (Real.exp V * ∑ i ∈ Finset.range j, c ^ (j - 1 - i)) =
            3 * (c * η * ∑ i ∈ Finset.range j, c ^ (j - 1 - i)) * Real.exp V := by ring
        rw [h1]
        dsimp [s, c] at hgw ⊢
        rw [hgw]
      linarith
    exact ⟨hu_ge, hu_gt, hv_le, hu_le⟩
  -- Anchor induction: `U_n` strictly increasing, `V_n ≤ v₀ - n * d * s m`
  have hU_ge0 : ∀ n, u₀ ≤ (mmdAnchor η lam m u₀ v₀ n).1 := by
    intro n
    induction n with
    | zero => exact le_rfl
    | succ n ih => exact le_trans ih ((hblock _ _ ih).1 m)
  have hU_strict : ∀ n, (mmdAnchor η lam m u₀ v₀ n).1 < (mmdAnchor η lam m u₀ v₀ (n + 1)).1 := by
    intro n
    exact (hblock _ _ (hU_ge0 n)).2.1 m hm
  have hV_bound : ∀ n, (mmdAnchor η lam m u₀ v₀ n).2 ≤ v₀ - (n : ℝ) * (d * s m) := by
    intro n
    induction n with
    | zero => simp [mmdAnchor]
    | succ n ih =>
      have h1 := (hblock (mmdAnchor η lam m u₀ v₀ n).1 (mmdAnchor η lam m u₀ v₀ n).2 (hU_ge0 n)).2.2.1 m
      change (mmdInner η lam (mmdAnchor η lam m u₀ v₀ n).1 (mmdAnchor η lam m u₀ v₀ n).2 m).2 ≤ _
      push_cast
      linarith
  have hU_step : ∀ n, (mmdAnchor η lam m u₀ v₀ (n + 1)).1 - (mmdAnchor η lam m u₀ v₀ n).1 ≤
      (3 * s m * Real.exp v₀) * (Real.exp (-(d * s m))) ^ n := by
    intro n
    have h1 := (hblock (mmdAnchor η lam m u₀ v₀ n).1 (mmdAnchor η lam m u₀ v₀ n).2 (hU_ge0 n)).2.2.2 m
    have h2 : Real.exp (mmdAnchor η lam m u₀ v₀ n).2 ≤ Real.exp (v₀ - (n : ℝ) * (d * s m)) :=
      Real.exp_le_exp.mpr (hV_bound n)
    have heq : Real.exp (v₀ - (n : ℝ) * (d * s m)) = Real.exp v₀ * (Real.exp (-(d * s m))) ^ n := by
      rw [← Real.exp_nat_mul, ← Real.exp_add]
      congr 1; ring
    have hsm_pos : 0 < s m := hs_pos m hm
    change (mmdInner η lam (mmdAnchor η lam m u₀ v₀ n).1 (mmdAnchor η lam m u₀ v₀ n).2 m).1 -
      (mmdAnchor η lam m u₀ v₀ n).1 ≤ _
    nlinarith
  have hsm_pos : 0 < s m := hs_pos m hm
  have hr0 : 0 < Real.exp (-(d * s m)) := Real.exp_pos _
  have hr1 : Real.exp (-(d * s m)) < 1 := Real.exp_lt_one_iff.mpr (by nlinarith)
  rcases monotone_converges_of_geom_bound hr0 hr1
    (fun n => le_of_lt (hU_strict n)) hU_step with ⟨U_inf, htend_U, hle_inf, _⟩
  have hU0_lt_inf : u₀ < U_inf := lt_of_lt_of_le (hU_strict 0) (hle_inf 1)
  have hdiv_tendsto : Tendsto (fun k : ℕ => k / m) atTop atTop := by
    rw [tendsto_atTop_atTop]
    intro N
    refine ⟨N * m, fun k hk => ?_⟩
    exact (Nat.le_div_iff_mul_le (by omega)).mpr hk
  have htend_V : Tendsto (fun n => (mmdAnchor η lam m u₀ v₀ n).2) atTop atBot := by
    refine tendsto_atBot_mono hV_bound ?_
    have hlin : Tendsto (fun n : ℕ => (n : ℝ) * (d * s m)) atTop atTop :=
      Tendsto.atTop_mul_const (mul_pos hd_pos hsm_pos) tendsto_natCast_atTop_atTop
    have hneg : Tendsto (fun n : ℕ => -((n : ℝ) * (d * s m))) atTop atBot :=
      tendsto_neg_atTop_atBot.comp hlin
    simpa [sub_eq_add_neg] using Filter.tendsto_atBot_add_const_left atTop v₀ hneg
  -- Full last-iterate convergence for `u_k` (squeezed between `U_{k/m}` and `U_{k/m} + 3 s_m exp(V_{k/m})`)
  have htend_u_full : Tendsto (fun k => (mmdFull η lam m u₀ v₀ k).1) atTop (nhds U_inf) := by
    have hlo : Tendsto (fun k => (mmdAnchor η lam m u₀ v₀ (k / m)).1) atTop (nhds U_inf) :=
      htend_U.comp hdiv_tendsto
    have hexp0 : Tendsto (fun k => 3 * s m * Real.exp (mmdAnchor η lam m u₀ v₀ (k / m)).2) atTop (nhds 0) := by
      have h1 : Tendsto (fun n => Real.exp (mmdAnchor η lam m u₀ v₀ n).2) atTop (nhds 0) :=
        Real.tendsto_exp_atBot.comp htend_V
      simpa using (h1.comp hdiv_tendsto).const_mul (3 * s m)
    have hhi : Tendsto (fun k => (mmdAnchor η lam m u₀ v₀ (k / m)).1 +
        3 * s m * Real.exp (mmdAnchor η lam m u₀ v₀ (k / m)).2) atTop (nhds U_inf) := by
      simpa using hlo.add hexp0
    refine tendsto_of_tendsto_of_tendsto_of_le_of_le hlo hhi ?_ ?_
    · intro k
      exact (hblock _ _ (hU_ge0 (k / m))).1 (k % m)
    · intro k
      have h1 := (hblock (mmdAnchor η lam m u₀ v₀ (k / m)).1
        (mmdAnchor η lam m u₀ v₀ (k / m)).2 (hU_ge0 (k / m))).2.2.2 (k % m)
      have hmod_le : k % m ≤ m := le_of_lt (Nat.mod_lt k (by omega))
      have hs_le : s (k % m) ≤ s m := hs_mono (k % m) hmod_le
      have hexp_pos : 0 < Real.exp (mmdAnchor η lam m u₀ v₀ (k / m)).2 := Real.exp_pos _
      change (mmdInner η lam (mmdAnchor η lam m u₀ v₀ (k / m)).1
        (mmdAnchor η lam m u₀ v₀ (k / m)).2 (k % m)).1 ≤ _
      nlinarith
  -- Full last-iterate convergence for `v_k ≤ V_{k/m} → -∞`
  have htend_v_full : Tendsto (fun k => (mmdFull η lam m u₀ v₀ k).2) atTop atBot := by
    refine tendsto_atBot_mono (fun k => ?_) (htend_V.comp hdiv_tendsto)
    have h1 := (hblock (mmdAnchor η lam m u₀ v₀ (k / m)).1
      (mmdAnchor η lam m u₀ v₀ (k / m)).2 (hU_ge0 (k / m))).2.2.1 (k % m)
    have hnn := mul_nonneg (le_of_lt hd_pos) (hs_nonneg (k % m))
    change (mmdInner η lam (mmdAnchor η lam m u₀ v₀ (k / m)).1
      (mmdAnchor η lam m u₀ v₀ (k / m)).2 (k % m)).2 ≤
      (mmdAnchor η lam m u₀ v₀ (k / m)).2
    linarith
  refine ⟨U_inf, (sigmoid_continuous.tendsto U_inf).comp htend_u_full,
    tendsto_sigmoid_atBot.comp htend_v_full, ?_, sigmoid_strictMono hU0_lt_inf⟩
  rw [isNash2x2_iff]
  exact ⟨⟨by linarith [sigmoid_strictMono hU0_lt_inf], le_of_lt (sigmoid_lt_one U_inf)⟩, rfl⟩

/-! ### 5. Theorem 3: Continuous-Time Flow With Periodic Resets -/

/-- A single reset block `[0, T]` of the anchored continuous-time ODE (2):
`u' = 3 g(v) - λ(u - U)`, `v' = -(3 g(u) - 1) - λ(v - V)` starting from `(U, V)`. -/
def IsAnchoredODEBlock (lam U V T : ℝ) (u v : ℝ → ℝ) : Prop :=
  u 0 = U ∧ v 0 = V ∧
  ∀ t ∈ Set.Icc 0 T,
    HasDerivAt u (3 * sigmoid (v t) - lam * (u t - U)) t ∧
    HasDerivAt v (-(3 * sigmoid (u t) - 1) - lam * (v t - V)) t

/-- Exact block estimates via integrating factor `e^{λ t}` (no integration library needed!). -/
lemma anchoredODEBlock_bounds {lam U V T u₀ : ℝ} {u v : ℝ → ℝ}
    (hlam : 0 < lam) (hT : 0 < T) (hp₀ : 1 / 3 < sigmoid u₀) (hU : u₀ ≤ U)
    (hode : IsAnchoredODEBlock lam U V T u v) :
    let d := 3 * sigmoid u₀ - 1
    let s := (1 - Real.exp (-(lam * T))) / lam
    (∀ t ∈ Set.Icc 0 T, U ≤ u t) ∧
    U < u T ∧
    (∀ t ∈ Set.Icc 0 T, v t ≤ V - d * ((1 - Real.exp (-(lam * t))) / lam)) ∧
    v T ≤ V - d * s ∧
    u T - U ≤ 3 * s * Real.exp V := by
  intro d s
  have hd_pos : 0 < d := by dsimp [d]; linarith
  -- Integrating factor: `φ(t) = e^{λ t} (u(t) - U)` has `φ'(t) = 3 e^{λ t} g(v(t)) > 0`.
  let φ : ℝ → ℝ := fun t => Real.exp (lam * t) * (u t - U)
  have hφ_deriv : ∀ t ∈ Set.Icc 0 T, HasDerivAt φ (3 * Real.exp (lam * t) * sigmoid (v t)) t := by
    intro t ht
    have he : HasDerivAt (fun x => Real.exp (lam * x)) (Real.exp (lam * t) * lam) t := by
      have := (Real.hasDerivAt_exp (lam * t)).comp t ((hasDerivAt_id t).const_mul lam)
      simpa [mul_comm] using this
    have hu : HasDerivAt (fun x => u x - U) (3 * sigmoid (v t) - lam * (u t - U)) t :=
      (hode.2.2 t ht).1.sub_const U
    convert he.mul hu using 1
    ring
  have hφ_mono : StrictMonoOn φ (Set.Icc 0 T) := by
    refine strictMonoOn_of_deriv_pos (convex_Icc 0 T) ?_ ?_
    · exact fun t ht => (hφ_deriv t ht).continuousAt.continuousWithinAt
    · intro t ht
      have ht' : t ∈ Set.Icc 0 T := interior_subset ht
      rw [(hφ_deriv t ht').deriv]
      have := sigmoid_pos (v t)
      positivity
  have hφ0 : φ 0 = 0 := by simp [φ, hode.1]
  have hu_ge : ∀ t ∈ Set.Icc 0 T, U ≤ u t := by
    intro t ht
    rcases eq_or_lt_of_le ht.1 with rfl | ht_pos
    · rw [hode.1]
    · have hlt : φ 0 < φ t := hφ_mono ⟨le_rfl, le_of_lt hT⟩ ht ht_pos
      rw [hφ0] at hlt
      have hexp : 0 < Real.exp (lam * t) := Real.exp_pos _
      have : 0 < u t - U := by nlinarith
      linarith
  have huT_gt : U < u T := by
    have hlt : φ 0 < φ T := hφ_mono ⟨le_rfl, le_of_lt hT⟩ ⟨le_of_lt hT, le_rfl⟩ hT
    rw [hφ0] at hlt
    have hexp : 0 < Real.exp (lam * T) := Real.exp_pos _
    nlinarith
  -- Second integrating factor: `ψ(t) = e^{λ t}(v(t) - V) + (d/λ)(e^{λ t} - 1)` has `ψ'(t) ≤ 0`.
  let ψ : ℝ → ℝ := fun t => Real.exp (lam * t) * (v t - V) + (d / lam) * (Real.exp (lam * t) - 1)
  have hψ_deriv : ∀ t ∈ Set.Icc 0 T,
      HasDerivAt ψ (Real.exp (lam * t) * (d - (3 * sigmoid (u t) - 1))) t := by
    intro t ht
    have he : HasDerivAt (fun x => Real.exp (lam * x)) (Real.exp (lam * t) * lam) t := by
      have := (Real.hasDerivAt_exp (lam * t)).comp t ((hasDerivAt_id t).const_mul lam)
      simpa [mul_comm] using this
    have hv : HasDerivAt (fun x => v x - V) (-(3 * sigmoid (u t) - 1) - lam * (v t - V)) t :=
      (hode.2.2 t ht).2.sub_const V
    have h2 : HasDerivAt (fun x => (d / lam) * (Real.exp (lam * x) - 1))
        ((d / lam) * (Real.exp (lam * t) * lam)) t :=
      (he.sub_const 1).const_mul (d / lam)
    convert (he.mul hv).add h2 using 1
    have hlam_ne : lam ≠ 0 := ne_of_gt hlam
    field_simp [hlam_ne]
    ring
  have hψ_anti : AntitoneOn ψ (Set.Icc 0 T) := by
    refine antitoneOn_of_deriv_nonpos (convex_Icc 0 T) ?_ ?_ ?_
    · exact fun t ht => (hψ_deriv t ht).continuousAt.continuousWithinAt
    · exact fun t ht => (hψ_deriv t (interior_subset ht)).differentiableAt.differentiableWithinAt
    · intro t ht
      have ht' : t ∈ Set.Icc 0 T := interior_subset ht
      rw [(hψ_deriv t ht').deriv]
      have hsig : sigmoid u₀ ≤ sigmoid (u t) := sigmoid_strictMono.monotone (le_trans hU (hu_ge t ht'))
      have hle : d - (3 * sigmoid (u t) - 1) ≤ 0 := by dsimp [d]; linarith
      exact mul_nonpos_of_nonneg_of_nonpos (le_of_lt (Real.exp_pos _)) hle
  have hψ0 : ψ 0 = 0 := by simp [ψ, hode.2.1]
  have hv_le : ∀ t ∈ Set.Icc 0 T, v t ≤ V - d * ((1 - Real.exp (-(lam * t))) / lam) := by
    intro t ht
    have hle : ψ t ≤ ψ 0 := hψ_anti ⟨le_rfl, le_of_lt hT⟩ ht ht.1
    rw [hψ0] at hle
    have hexp_pos : 0 < Real.exp (lam * t) := Real.exp_pos _
    have hexp_ne : Real.exp (lam * t) ≠ 0 := ne_of_gt hexp_pos
    have hlam_ne : lam ≠ 0 := ne_of_gt hlam
    have hinv : Real.exp (-(lam * t)) = (Real.exp (lam * t))⁻¹ := Real.exp_neg (lam * t)
    dsimp [ψ] at hle
    have hmul : Real.exp (lam * t) * (v t - (V - d * ((1 - Real.exp (-(lam * t))) / lam))) ≤ 0 := by
      rw [hinv]
      have heq : Real.exp (lam * t) * (v t - (V - d * ((1 - (Real.exp (lam * t))⁻¹) / lam))) =
          Real.exp (lam * t) * (v t - V) + d / lam * (Real.exp (lam * t) - 1) := by
        field_simp [hexp_ne, hlam_ne]
        ring
      linarith
    have : v t - (V - d * ((1 - Real.exp (-(lam * t))) / lam)) ≤ 0 :=
      nonpos_of_mul_nonpos_right hmul hexp_pos
    linarith
  -- Third integrating factor: `χ(t) = φ(t) - (3 e^V / λ) (e^{λ t} - 1)` has `χ'(t) ≤ 0`.
  let χ : ℝ → ℝ := fun t => φ t - (3 * Real.exp V / lam) * (Real.exp (lam * t) - 1)
  have hχ_deriv : ∀ t ∈ Set.Icc 0 T,
      HasDerivAt χ (3 * Real.exp (lam * t) * (sigmoid (v t) - Real.exp V)) t := by
    intro t ht
    have he : HasDerivAt (fun x => Real.exp (lam * x)) (Real.exp (lam * t) * lam) t := by
      have := (Real.hasDerivAt_exp (lam * t)).comp t ((hasDerivAt_id t).const_mul lam)
      simpa [mul_comm] using this
    have h2 : HasDerivAt (fun x => (3 * Real.exp V / lam) * (Real.exp (lam * x) - 1))
        ((3 * Real.exp V / lam) * (Real.exp (lam * t) * lam)) t :=
      (he.sub_const 1).const_mul (3 * Real.exp V / lam)
    convert (hφ_deriv t ht).sub h2 using 1
    have hlam_ne : lam ≠ 0 := ne_of_gt hlam
    field_simp [hlam_ne]
  have hχ_anti : AntitoneOn χ (Set.Icc 0 T) := by
    refine antitoneOn_of_deriv_nonpos (convex_Icc 0 T) ?_ ?_ ?_
    · exact fun t ht => (hχ_deriv t ht).continuousAt.continuousWithinAt
    · exact fun t ht => (hχ_deriv t (interior_subset ht)).differentiableAt.differentiableWithinAt
    · intro t ht
      have ht' : t ∈ Set.Icc 0 T := interior_subset ht
      rw [(hχ_deriv t ht').deriv]
      have hvt : v t ≤ V := by
        have h1 := hv_le t ht'
        have hexp1 : Real.exp (-(lam * t)) ≤ 1 := Real.exp_le_one_iff.mpr (by nlinarith [ht'.1])
        have hnn : 0 ≤ d * ((1 - Real.exp (-(lam * t))) / lam) :=
          mul_nonneg (le_of_lt hd_pos) (div_nonneg (by linarith) (le_of_lt hlam))
        linarith
      have hsig : sigmoid (v t) ≤ Real.exp V :=
        le_trans (le_of_lt (sigmoid_lt_exp _)) (Real.exp_le_exp.mpr hvt)
      exact mul_nonpos_of_nonneg_of_nonpos (by positivity) (by linarith)
  have hχ0 : χ 0 = 0 := by simp [χ, hφ0]
  have huT_ub : u T - U ≤ 3 * s * Real.exp V := by
    have hle : χ T ≤ χ 0 := hχ_anti ⟨le_rfl, le_of_lt hT⟩ ⟨le_of_lt hT, le_rfl⟩ (le_of_lt hT)
    rw [hχ0] at hle
    have hexp_pos : 0 < Real.exp (lam * T) := Real.exp_pos _
    have hexp_ne : Real.exp (lam * T) ≠ 0 := ne_of_gt hexp_pos
    have hlam_ne : lam ≠ 0 := ne_of_gt hlam
    have hinv : Real.exp (-(lam * T)) = (Real.exp (lam * T))⁻¹ := Real.exp_neg (lam * T)
    dsimp [χ, φ, s] at hle ⊢
    have hmul : Real.exp (lam * T) * (u T - U - 3 * ((1 - Real.exp (-(lam * T))) / lam) * Real.exp V) ≤ 0 := by
      rw [hinv]
      have heq : Real.exp (lam * T) * (u T - U - 3 * ((1 - (Real.exp (lam * T))⁻¹) / lam) * Real.exp V) =
          Real.exp (lam * T) * (u T - U) - 3 * Real.exp V / lam * (Real.exp (lam * T) - 1) := by
        field_simp [hexp_ne, hlam_ne]
      linarith
    have : u T - U - 3 * ((1 - Real.exp (-(lam * T))) / lam) * Real.exp V ≤ 0 :=
      nonpos_of_mul_nonpos_right hmul hexp_pos
    linarith
  exact ⟨hu_ge, huT_gt, hv_le, hv_le T ⟨le_of_lt hT, le_rfl⟩, huT_ub⟩

/-- **Theorem 3 (Continuous-time flow with periodic resets)**:
for every `λ > 0`, `T > 0`, and interior start with `p₀ = g(u₀) ∈ (1/3, 1)`, any sequence of
reset blocks solving (2) converges to a Nash equilibrium `(g(U_∞), 0)` with `g(U_∞) > p₀`. -/
theorem periodic_ode_converges_away {lam T u₀ v₀ : ℝ}
    (hlam : 0 < lam) (hT : 0 < T) (hp₀ : 1 / 3 < sigmoid u₀)
    (U V : ℕ → ℝ) (u_blk v_blk : ℕ → ℝ → ℝ)
    (hinit : U 0 = u₀ ∧ V 0 = v₀)
    (hblocks : ∀ n, IsAnchoredODEBlock lam (U n) (V n) T (u_blk n) (v_blk n))
    (hreset : ∀ n, U (n + 1) = u_blk n T ∧ V (n + 1) = v_blk n T) :
    ∃ U_inf : ℝ,
      Tendsto U atTop (nhds U_inf) ∧
      Tendsto (fun n => sigmoid (V n)) atTop (nhds 0) ∧
      IsNash2x2 (sigmoid U_inf) 0 ∧
      sigmoid u₀ < sigmoid U_inf := by
  let d : ℝ := 3 * sigmoid u₀ - 1
  have hd_pos : 0 < d := by dsimp [d]; linarith
  let s : ℝ := (1 - Real.exp (-(lam * T))) / lam
  have hs_pos : 0 < s := by
    dsimp [s]
    have hlt : Real.exp (-(lam * T)) < 1 := Real.exp_lt_one_iff.mpr (by nlinarith)
    exact div_pos (by linarith) hlam
  have hU_ge0 : ∀ n, u₀ ≤ U n := by
    intro n
    induction n with
    | zero => rw [hinit.1]
    | succ n ih =>
      rw [(hreset n).1]
      exact le_trans ih (le_of_lt (anchoredODEBlock_bounds hlam hT hp₀ ih (hblocks n)).2.1)
  have hU_strict : ∀ n, U n < U (n + 1) := by
    intro n
    rw [(hreset n).1]
    exact (anchoredODEBlock_bounds hlam hT hp₀ (hU_ge0 n) (hblocks n)).2.1
  have hV_bound : ∀ n, V n ≤ v₀ - (n : ℝ) * (d * s) := by
    intro n
    induction n with
    | zero => simp [hinit.2]
    | succ n ih =>
      rw [(hreset n).2]
      have h1 := (anchoredODEBlock_bounds hlam hT hp₀ (hU_ge0 n) (hblocks n)).2.2.2.1
      push_cast
      linarith
  have hU_step : ∀ n, U (n + 1) - U n ≤ (3 * s * Real.exp v₀) * (Real.exp (-(d * s))) ^ n := by
    intro n
    rw [(hreset n).1]
    have h1 := (anchoredODEBlock_bounds hlam hT hp₀ (hU_ge0 n) (hblocks n)).2.2.2.2
    have h2 : Real.exp (V n) ≤ Real.exp (v₀ - (n : ℝ) * (d * s)) := Real.exp_le_exp.mpr (hV_bound n)
    have heq : Real.exp (v₀ - (n : ℝ) * (d * s)) = Real.exp v₀ * (Real.exp (-(d * s))) ^ n := by
      rw [← Real.exp_nat_mul, ← Real.exp_add]; congr 1; ring
    nlinarith
  have hr0 : 0 < Real.exp (-(d * s)) := Real.exp_pos _
  have hr1 : Real.exp (-(d * s)) < 1 := Real.exp_lt_one_iff.mpr (by nlinarith)
  rcases monotone_converges_of_geom_bound hr0 hr1
    (fun n => le_of_lt (hU_strict n)) hU_step with ⟨U_inf, htend_U, hle_inf, _⟩
  have hU0_lt : u₀ < U_inf := by
    rw [← hinit.1]
    exact lt_of_lt_of_le (hU_strict 0) (hle_inf 1)
  have htend_V : Tendsto V atTop atBot := by
    refine tendsto_atBot_mono hV_bound ?_
    have hlin : Tendsto (fun n : ℕ => (n : ℝ) * (d * s)) atTop atTop :=
      Tendsto.atTop_mul_const (mul_pos hd_pos hs_pos) tendsto_natCast_atTop_atTop
    have hneg : Tendsto (fun n : ℕ => -((n : ℝ) * (d * s))) atTop atBot :=
      tendsto_neg_atTop_atBot.comp hlin
    simpa [sub_eq_add_neg] using Filter.tendsto_atBot_add_const_left atTop v₀ hneg
  refine ⟨U_inf, htend_U, tendsto_sigmoid_atBot.comp htend_V, ?_, sigmoid_strictMono hU0_lt⟩
  rw [isNash2x2_iff]
  exact ⟨⟨by linarith [sigmoid_strictMono hU0_lt], le_of_lt (sigmoid_lt_one U_inf)⟩, rfl⟩

/-! ### 6. Theorem 4: Continuously Reset ODE Flow and Exact First Integral -/

/-- First-integral components `F(u) = -u + 3 log(1 + e^u)` and `J(v) = log(1 + e^v)`. -/
noncomputable def firstIntF (u : ℝ) : ℝ := -u + 3 * Real.log (1 + Real.exp u)
noncomputable def firstIntJ (v : ℝ) : ℝ := Real.log (1 + Real.exp v)

lemma hasDerivAt_firstIntJ (v : ℝ) : HasDerivAt firstIntJ (sigmoid v) v := by
  unfold firstIntJ sigmoid
  have hpos : 0 < 1 + Real.exp v := by positivity
  have h1 : HasDerivAt (fun x => 1 + Real.exp x) (Real.exp v) v := by
    simpa using (Real.hasDerivAt_exp v).const_add 1
  have h2 := h1.log (ne_of_gt hpos)
  convert h2 using 1
  have hepos : 0 < Real.exp v := Real.exp_pos v
  have hene : Real.exp v ≠ 0 := ne_of_gt hepos
  rw [Real.exp_neg]
  field_simp [ne_of_gt hpos, hene]
  ring

lemma hasDerivAt_firstIntF (u : ℝ) : HasDerivAt firstIntF (3 * sigmoid u - 1) u := by
  unfold firstIntF
  have h1 := (hasDerivAt_neg u).add ((hasDerivAt_firstIntJ u).const_mul 3)
  convert h1 using 1
  ring

/-- **Exact first integral conservation** along the continuously reset ODE
`u' = 3 g(v)`, `v' = 1 - 3 g(u)`: `F(u(t)) + 3 J(v(t)) = F(u(0)) + 3 J(v(0))` for all `t ≥ 0`. -/
theorem continuous_reset_first_integral {u v : ℝ → ℝ}
    (hode : ∀ t ≥ 0, HasDerivAt u (3 * sigmoid (v t)) t ∧ HasDerivAt v (1 - 3 * sigmoid (u t)) t) :
    ∀ t ≥ 0, firstIntF (u t) + 3 * firstIntJ (v t) = firstIntF (u 0) + 3 * firstIntJ (v 0) := by
  let I : ℝ → ℝ := fun t => firstIntF (u t) + 3 * firstIntJ (v t)
  have hI_deriv : ∀ t ≥ 0, HasDerivAt I 0 t := by
    intro t ht
    have h1 := (hasDerivAt_firstIntF (u t)).comp t (hode t ht).1
    have h2 := ((hasDerivAt_firstIntJ (v t)).comp t (hode t ht).2).const_mul 3
    convert h1.add h2 using 1
    ring
  intro t ht
  have hconst : ∀ x ∈ Set.Icc 0 t, I x = I 0 := by
    have hmono1 : MonotoneOn I (Set.Icc 0 t) := by
      refine monotoneOn_of_deriv_nonneg (convex_Icc 0 t) ?_ ?_ ?_
      · exact fun x hx => (hI_deriv x hx.1).continuousAt.continuousWithinAt
      · exact fun x hx => (hI_deriv x (interior_subset hx).1).differentiableAt.differentiableWithinAt
      · intro x hx; rw [(hI_deriv x (interior_subset hx).1).deriv]
    have hanti1 : AntitoneOn I (Set.Icc 0 t) := by
      refine antitoneOn_of_deriv_nonpos (convex_Icc 0 t) ?_ ?_ ?_
      · exact fun x hx => (hI_deriv x hx.1).continuousAt.continuousWithinAt
      · exact fun x hx => (hI_deriv x (interior_subset hx).1).differentiableAt.differentiableWithinAt
      · intro x hx; rw [(hI_deriv x (interior_subset hx).1).deriv]
    intro x hx
    exact le_antisymm (hanti1 ⟨le_rfl, ht⟩ hx hx.1) (hmono1 ⟨le_rfl, ht⟩ hx hx.1)
  exact hconst t ⟨ht, le_rfl⟩

/-- Algebraic identity: `exp(F(u)) = (g(u) * (1 - g(u))²)⁻¹`. -/
lemma exp_firstIntF_eq (u : ℝ) :
    Real.exp (firstIntF u) = (sigmoid u * (1 - sigmoid u) ^ 2)⁻¹ := by
  unfold firstIntF sigmoid
  have hepos : 0 < Real.exp u := Real.exp_pos u
  have h1pos : 0 < 1 + Real.exp u := by positivity
  have h1negpos : 0 < 1 + Real.exp (-u) := by positivity
  rw [Real.exp_add, Real.exp_neg, show (3 : ℝ) * Real.log (1 + Real.exp u) =
    ((3 : ℕ) : ℝ) * Real.log (1 + Real.exp u) by norm_num, Real.exp_nat_mul, Real.exp_log h1pos]
  have hene : Real.exp u ≠ 0 := ne_of_gt hepos
  have h1ne : 1 + Real.exp u ≠ 0 := ne_of_gt h1pos
  have h1nne : 1 + (Real.exp u)⁻¹ ≠ 0 := by
    rw [← Real.exp_neg]; exact ne_of_gt h1negpos
  field_simp [hene, h1ne, h1nne]
  ring

/-- **Theorem 4 (Continuously reset ODE flow)**:
from any start with `p₀ = g(u(0)) ∈ (1/3, 1)`, `(g(u(t)), g(v(t))) → (g(u_∞), 0) ∈ N(A)` with
`g(u_∞) > p₀`; and from uniform start `(u(0), v(0)) = (0, 0)`, `p_∞ = g(u_∞)` is the unique
root in `(1/2, 1)` of `p * (1 - p)² = 1 / 64` (and satisfies `0.865 < p_∞ < 0.866`). -/
theorem continuous_reset_converges_away {u v : ℝ → ℝ}
    (hp₀ : 1 / 3 < sigmoid (u 0))
    (hode : ∀ t ≥ 0, HasDerivAt u (3 * sigmoid (v t)) t ∧ HasDerivAt v (1 - 3 * sigmoid (u t)) t) :
    ∃ u_inf : ℝ,
      Tendsto u atTop (nhds u_inf) ∧
      Tendsto (fun t => sigmoid (v t)) atTop (nhds 0) ∧
      IsNash2x2 (sigmoid u_inf) 0 ∧
      sigmoid (u 0) < sigmoid u_inf ∧
      (u 0 = 0 ∧ v 0 = 0 →
        sigmoid u_inf * (1 - sigmoid u_inf) ^ 2 = 1 / 64 ∧
        (865 : ℝ) / 1000 < sigmoid u_inf ∧ sigmoid u_inf < (866 : ℝ) / 1000) := by
  let d : ℝ := 3 * sigmoid (u 0) - 1
  have hd_pos : 0 < d := by dsimp [d]; linarith
  -- `u` is strictly increasing on `[0, ∞)`
  have hu_strict : StrictMonoOn u (Set.Ici 0) := by
    refine strictMonoOn_of_deriv_pos (convex_Ici 0) ?_ ?_
    · exact fun t ht => (hode t ht).1.continuousAt.continuousWithinAt
    · intro t ht
      have ht0 : 0 ≤ t := interior_subset ht
      rw [(hode t ht0).1.deriv]
      linarith [sigmoid_pos (v t)]
  -- `v(t) ≤ v(0) - d * t` for all `t ≥ 0`
  have hv_bound : ∀ t ≥ 0, v t ≤ v 0 - d * t := by
    intro t ht
    let f : ℝ → ℝ := fun x => v x + d * x
    have hf_anti : AntitoneOn f (Set.Icc 0 t) := by
      refine antitoneOn_of_deriv_nonpos (convex_Icc 0 t) ?_ ?_ ?_
      · exact fun x hx => ((hode x hx.1).2.add ((hasDerivAt_id x).const_mul d)).continuousAt.continuousWithinAt
      · exact fun x hx => ((hode x (interior_subset hx).1).2.add ((hasDerivAt_id x).const_mul d)).differentiableAt.differentiableWithinAt
      · intro x hx
        have hx0 : 0 ≤ x := (interior_subset hx).1
        have hderiv : HasDerivAt f (1 - 3 * sigmoid (u x) + d) x := by
          simpa using (hode x hx0).2.add ((hasDerivAt_id x).const_mul d)
        rw [hderiv.deriv]
        have hu_ge : u 0 ≤ u x := hu_strict.monotoneOn (Set.left_mem_Ici) hx0 hx0
        have hsig := sigmoid_strictMono.monotone hu_ge
        dsimp [d]
        linarith
    have := hf_anti ⟨le_rfl, ht⟩ ⟨ht, le_rfl⟩ ht
    dsimp [f] at this
    linarith
  -- `u(t) ≤ u(0) + (3 / d) * exp(v(0))` for all `t ≥ 0`
  have hu_upper : ∀ t ≥ 0, u t ≤ u 0 + (3 / d) * Real.exp (v 0) := by
    intro t ht
    let g : ℝ → ℝ := fun x => u x + (3 / d) * Real.exp (v 0 - d * x)
    have hg_deriv : ∀ x ≥ 0, HasDerivAt g (3 * (sigmoid (v x) - Real.exp (v 0 - d * x))) x := by
      intro x hx
      have he : HasDerivAt (fun y => Real.exp (v 0 - d * y))
          (Real.exp (v 0 - d * x) * (-d)) x := by
        have hlin : HasDerivAt (fun y => v 0 - d * y) (-d) x := by
          simpa using (hasDerivAt_const x (v 0)).sub ((hasDerivAt_id x).const_mul d)
        exact (Real.hasDerivAt_exp (v 0 - d * x)).comp x hlin
      convert (hode x hx).1.add (he.const_mul (3 / d)) using 1
      have hdne : d ≠ 0 := ne_of_gt hd_pos
      field_simp [hdne]
      ring
    have hg_anti : AntitoneOn g (Set.Icc 0 t) := by
      refine antitoneOn_of_deriv_nonpos (convex_Icc 0 t) ?_ ?_ ?_
      · exact fun x hx => (hg_deriv x hx.1).continuousAt.continuousWithinAt
      · exact fun x hx => (hg_deriv x (interior_subset hx).1).differentiableAt.differentiableWithinAt
      · intro x hx
        have hx0 : 0 ≤ x := (interior_subset hx).1
        rw [(hg_deriv x hx0).deriv]
        have h1 : sigmoid (v x) ≤ Real.exp (v 0 - d * x) :=
          le_trans (le_of_lt (sigmoid_lt_exp _)) (Real.exp_le_exp.mpr (hv_bound x hx0))
        linarith
    have := hg_anti ⟨le_rfl, ht⟩ ⟨ht, le_rfl⟩ ht
    dsimp [g] at this
    have hpos : 0 < (3 / d) * Real.exp (v 0 - d * t) := by positivity
    simp only [mul_zero, sub_zero] at this
    linarith
  -- Existence of `u_inf = sSup (u '' Set.Ici 0)` and convergence `u(t) → u_inf`
  let u_inf : ℝ := sSup (u '' Set.Ici 0)
  have hBdd : BddAbove (u '' Set.Ici 0) :=
    ⟨u 0 + (3 / d) * Real.exp (v 0), by rintro _ ⟨t, ht, rfl⟩; exact hu_upper t ht⟩
  have hNonempty : (u '' Set.Ici 0).Nonempty := ⟨u 0, 0, Set.left_mem_Ici, rfl⟩
  have htend_u : Tendsto u atTop (nhds u_inf) := by
    rw [Metric.tendsto_atTop]
    intro ε hε
    have hlt : u_inf - ε < u_inf := by linarith
    rcases exists_lt_of_lt_csSup hNonempty hlt with ⟨_, ⟨T, hT0, rfl⟩, hTu⟩
    refine ⟨max T 0, fun t ht => ?_⟩
    have htT : T ≤ t := le_trans (le_max_left _ _) ht
    have ht0 : 0 ≤ t := le_trans (le_max_right _ _) ht
    have hle1 : u T ≤ u t := hu_strict.monotoneOn hT0 ht0 htT
    have hle2 : u t ≤ u_inf := le_csSup hBdd ⟨t, ht0, rfl⟩
    rw [Real.dist_eq, abs_lt]
    constructor <;> linarith
  have htend_v : Tendsto v atTop atBot := by
    rw [tendsto_atBot]
    intro M
    rw [eventually_atTop]
    refine ⟨max 0 ((v 0 - M) / d), fun t ht => ?_⟩
    have ht0 : 0 ≤ t := le_trans (le_max_left _ _) ht
    have htM : (v 0 - M) / d ≤ t := le_trans (le_max_right _ _) ht
    have hmul : v 0 - M ≤ d * t := by
      have := (div_le_iff₀' hd_pos).mp htM
      linarith
    linarith [hv_bound t ht0]
  have hu0_lt_inf : u 0 < u_inf := by
    have h1 : u 0 < u 1 := hu_strict Set.left_mem_Ici (Set.mem_Ici.mpr zero_le_one) zero_lt_one
    have h2 : u 1 ≤ u_inf := le_csSup hBdd ⟨1, Set.mem_Ici.mpr zero_le_one, rfl⟩
    linarith
  have hsig_lt : sigmoid (u 0) < sigmoid u_inf := sigmoid_strictMono hu0_lt_inf
  refine ⟨u_inf, htend_u, tendsto_sigmoid_atBot.comp htend_v, ?_, hsig_lt, ?_⟩
  · rw [isNash2x2_iff]
    exact ⟨⟨by linarith, le_of_lt (sigmoid_lt_one u_inf)⟩, rfl⟩
  · rintro ⟨hu0, hv0⟩
    -- Take `t → ∞` in `firstIntF (u t) + 3 * firstIntJ (v t) = 6 * log 2`
    have hconst : ∀ t ≥ 0, firstIntF (u t) + 3 * firstIntJ (v t) = 6 * Real.log 2 := by
      intro t ht
      have h1 := continuous_reset_first_integral hode t ht
      rw [h1, hu0, hv0]
      simp only [firstIntF, firstIntJ, neg_zero, zero_add, Real.exp_zero]
      norm_num
      ring
    have hF_cont : Continuous firstIntF := by
      unfold firstIntF
      exact continuous_neg.add (continuous_const.mul
        ((continuous_const.add Real.continuous_exp).log (fun x => ne_of_gt (by positivity))))
    have htend_F : Tendsto (fun t => firstIntF (u t)) atTop (nhds (firstIntF u_inf)) :=
      (hF_cont.tendsto u_inf).comp htend_u
    have htend_J : Tendsto (fun t => firstIntJ (v t)) atTop (nhds 0) := by
      have he0 : Tendsto (fun t => Real.exp (v t)) atTop (nhds 0) :=
        Real.tendsto_exp_atBot.comp htend_v
      have h1 : Tendsto (fun t => 1 + Real.exp (v t)) atTop (nhds 1) := by
        simpa using he0.const_add 1
      simpa [firstIntJ, Real.log_one] using (Real.continuousAt_log one_ne_zero).tendsto.comp h1
    have htend_sum : Tendsto (fun t => firstIntF (u t) + 3 * firstIntJ (v t)) atTop
        (nhds (firstIntF u_inf)) := by
      simpa using htend_F.add (htend_J.const_mul 3)
    have htend_const : Tendsto (fun t => firstIntF (u t) + 3 * firstIntJ (v t)) atTop
        (nhds (6 * Real.log 2)) :=
      tendsto_const_nhds.congr' (by filter_upwards [eventually_ge_atTop (0 : ℝ)] with t ht; exact (hconst t ht).symm)
    have hFinf : firstIntF u_inf = 6 * Real.log 2 := tendsto_nhds_unique htend_sum htend_const
    have hexp_eq : Real.exp (firstIntF u_inf) = 64 := by
      rw [hFinf, show (6 : ℝ) * Real.log 2 = ((6 : ℕ) : ℝ) * Real.log 2 by norm_num,
        Real.exp_nat_mul, Real.exp_log (by norm_num)]
      norm_num
    rw [exp_firstIntF_eq] at hexp_eq
    have hcubic : sigmoid u_inf * (1 - sigmoid u_inf) ^ 2 = 1 / 64 := by
      have := congrArg Inv.inv hexp_eq
      simpa [inv_inv] using this
    refine ⟨hcubic, ?_, ?_⟩
    · -- On `[1/3, 1]`, `p ↦ p * (1 - p)^2` is strictly decreasing:
      -- `f(0.865) = 0.865 * 0.135^2 = 0.015764625 > 1/64 = 0.015625`
      by_contra hle
      push_neg at hle
      have hp_ge : (1 : ℝ) / 2 < sigmoid u_inf := by rw [← sigmoid_zero, ← hu0]; exact hsig_lt
      have hdiff : sigmoid u_inf * (1 - sigmoid u_inf) ^ 2 -
          (865 / 1000 : ℝ) * (1 - 865 / 1000) ^ 2 ≥ 0 := by
        -- Factor `(p - a) * (...)`: for `1/2 ≤ p ≤ a < 1`, `f(p) - f(a) = (a - p) * (3 p a ...)`?
        -- Actually `p(1-p)^2 - a(1-a)^2 = (a - p) * (p^2 + p a + a^2 - 2p - 2a + 1)`
        -- Wait: check `f'(x) = (1 - x)(1 - 3x) < 0` on `(1/3, 1)`!
        have hfac : sigmoid u_inf * (1 - sigmoid u_inf) ^ 2 -
            (865 / 1000 : ℝ) * (1 - 865 / 1000) ^ 2 =
            ((865 / 1000 : ℝ) - sigmoid u_inf) *
              (2 * (sigmoid u_inf + 865 / 1000) - 1 -
                (sigmoid u_inf ^ 2 + sigmoid u_inf * (865 / 1000) + (865 / 1000) ^ 2)) := by ring
        have hfactor_pos : 0 ≤ 2 * (sigmoid u_inf + 865 / 1000) - 1 -
            (sigmoid u_inf ^ 2 + sigmoid u_inf * (865 / 1000) + (865 / 1000) ^ 2) := by
          -- Rewrite as `(1 - p)(2p - ...)`, or simply `nlinarith` with `p > 1/2, p ≤ 0.865`
          nlinarith [sq_nonneg (sigmoid u_inf - 1 / 2)]
        rw [hfac]
        exact mul_nonneg (by linarith) hfactor_pos
      have hnum : (1 : ℝ) / 64 < (865 / 1000 : ℝ) * (1 - 865 / 1000) ^ 2 := by norm_num
      linarith
    · -- `f(0.866) = 0.866 * 0.134^2 = 0.015549896 < 1/64 = 0.015625`
      by_contra hge
      push_neg at hge
      have hp_lt1 : sigmoid u_inf < 1 := sigmoid_lt_one u_inf
      have hdiff : (866 / 1000 : ℝ) * (1 - 866 / 1000) ^ 2 -
          sigmoid u_inf * (1 - sigmoid u_inf) ^ 2 ≥ 0 := by
        have hfac : (866 / 1000 : ℝ) * (1 - 866 / 1000) ^ 2 -
            sigmoid u_inf * (1 - sigmoid u_inf) ^ 2 =
            (sigmoid u_inf - 866 / 1000) *
              (2 * (sigmoid u_inf + 866 / 1000) - 1 -
                (sigmoid u_inf ^ 2 + sigmoid u_inf * (866 / 1000) + (866 / 1000) ^ 2)) := by ring
        have hfactor_pos : 0 ≤ 2 * (sigmoid u_inf + 866 / 1000) - 1 -
            (sigmoid u_inf ^ 2 + sigmoid u_inf * (866 / 1000) + (866 / 1000) ^ 2) := by
          nlinarith [sq_nonneg (1 - sigmoid u_inf)]
        rw [hfac]
        exact mul_nonneg (by linarith) hfactor_pos
      have hnum : (866 / 1000 : ℝ) * (1 - 866 / 1000) ^ 2 < (1 : ℝ) / 64 := by norm_num
      linarith

/-- Uniqueness of the cubic root `p(1 - p)² = 1/64` on `(1/3, 1)`. -/
theorem cubic_root_unique {p₁ p₂ : ℝ} (h₁ : 1 / 3 < p₁) (h₁' : p₁ < 1)
    (h₂ : 1 / 3 < p₂) (h₂' : p₂ < 1)
    (heq : p₁ * (1 - p₁) ^ 2 = p₂ * (1 - p₂) ^ 2) : p₁ = p₂ := by
  have hanti : StrictAntiOn (fun p : ℝ => p * (1 - p) ^ 2) (Set.Ioo (1 / 3) 1) := by
    refine strictAntiOn_of_deriv_neg (convex_Ioo (1 / 3) 1) ?_ ?_
    · exact (continuous_id.mul ((continuous_const.sub continuous_id).pow 2)).continuousOn
    · intro x hx
      rw [interior_Ioo] at hx
      have hd : HasDerivAt (fun p : ℝ => p * (1 - p) ^ 2) ((1 - x) * (1 - 3 * x)) x := by
        have h1 : HasDerivAt (fun p : ℝ => (1 - p) ^ 2) (2 * (1 - x) * (-1)) x := by
          simpa using ((hasDerivAt_const x (1 : ℝ)).sub (hasDerivAt_id x)).pow 2
        convert (hasDerivAt_id x).mul h1 using 1
        simp only [id_eq]
        ring
      rw [hd.deriv]
      have hx1 : 0 < 1 - x := by linarith [hx.2]
      have hx2 : 1 - 3 * x < 0 := by linarith [hx.1]
      exact mul_neg_of_pos_of_neg hx1 hx2
  exact hanti.injOn ⟨h₁, h₁'⟩ ⟨h₂, h₂'⟩ heq

/-! ### 7. Proposition 5: Tangent-Orthogonality Invariant of Discrete MMD -/

/-- **Proposition 5 (Tangent-orthogonality invariant)**:
if `zᵀ 1 = 0` and `zᵀ A = 0`, then any discrete MMD log-policy step
`logX' - X = c • (logX - X + η • (A *ᵥ y)) + γ • 1` preserves `zᵀ logX = zᵀ X` identically. -/
theorem tangent_orthogonality_invariant {m n : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ) (z : Fin m → ℝ)
    (hz1 : ∑ i : Fin m, z i = 0)
    (hzA : ∀ j : Fin n, ∑ i : Fin m, z i * A i j = 0)
    (c η γ : ℝ) (X logX logX' : Fin m → ℝ) (y : Fin n → ℝ)
    (hstep : ∀ i : Fin m, logX' i - X i = c * (logX i - X i + η * (A.mulVec y i)) + γ) :
    ∑ i : Fin m, z i * (logX' i - X i) = c * ∑ i : Fin m, z i * (logX i - X i) := by
  have h1 : ∑ i : Fin m, z i * (logX' i - X i) =
      c * (∑ i : Fin m, z i * (logX i - X i)) +
      c * η * (∑ i : Fin m, z i * (A.mulVec y i)) +
      γ * (∑ i : Fin m, z i) := by
    simp_rw [hstep]
    rw [Finset.mul_sum, Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib,
      ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    ring
  have h2 : ∑ i : Fin m, z i * (A.mulVec y i) = 0 := by
    simp only [Matrix.mulVec, dotProduct, Finset.mul_sum]
    rw [Finset.sum_comm]
    refine Finset.sum_eq_zero fun j _ => ?_
    have : ∑ i : Fin m, z i * (A i j * y j) = (∑ i : Fin m, z i * A i j) * y j := by
      rw [Finset.sum_mul]
      refine Finset.sum_congr rfl fun i _ => by ring
    rw [this, hzA j, zero_mul]
  rw [h1, h2, hz1]
  ring

end NashIProjSelection
