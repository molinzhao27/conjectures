import Mathlib

/-!
# Global Last-Iterate Convergence of Cumulative-Average Matrix GARIP
(Can Savcı, arXiv:2606.22688, June 2026, Section 4, Conjecture `conj:li`)

We formalize in Lean 4 + Mathlib the exact Lyapunov-potential, KL-dissipation, step-size,
matching-pennies counterexample, and asymptotic positive-exploitability exclusion cores
resolving Conjecture `conj:li`:

1. **Certified Rectangle and Skew-Symmetric Payoff Structure (Section 1)**:
   - `certified_beta_equiv`: For `0 < β < 1` and `μ > 0`, `c := β / (1 - β) ≤ μ / 8`
     if and only if `β ≤ μ / (8 + μ)`.
   - `zeroSum_skew_symmetry`: For any matrix `A` and strategy pair `z = (x, y)`,
     `⟨g(z), z⟩ = xᵀ A y - yᵀ Aᵀ x = 0`, and `⟨g(z), z - p⟩ = ⟨g(p), z - p⟩`.
   - `nash_slack_bound`: If `p` is a Nash equilibrium with value `v` and off-support slack
     `≥ γ / 2` outside `S`, then `⟨g(z), z - p⟩ ≤ -(γ / 2) * m(z)`.

2. **Refutation of Raw Half-Step KL Monotonicity in Matching Pennies (Section 1)**:
   - `matchingPennies_raw_halfstep_kl_increase`: For Matching Pennies (`A = [[1, -1], [-1, 1]]`),
     uniform Nash `p = ((1/2, 1/2), (1/2, 1/2))`, EVERY non-equilibrium interior start
     `z₀ = ((x, 1 - x), (y, 1 - y))` (`x, y ∈ (0, 1)`, `(x, y) ≠ (1/2, 1/2)`), and EVERY `η > 0`,
     the raw half-step KL strictly increases: `D(p ‖ h₀) - D(p ‖ z₀) > 0`.

3. **Uniform Ratio and Anchor Correction Bounds (Lemma 1)**:
   - `exp_sub_one_le_two_mul`: For `0 ≤ u ≤ 3 / 32`, `exp(u) - 1 ≤ 2 u` and `1 - exp(-u) ≤ 2 u`.
   - `ratio_lower_step_invariant`: Inductively, `z / a ≥ l := β / (1 - (1 - β) k)` is preserved
     by `w / a = (1 - β)(h / z)(z / a) + β` and convex blending `a' = (1 - α) a + α w`,
     and yields `s := c (1 - a / w) ≥ k - 1`.
   - `correction_bounds_certified`: Under `0 < η ≤ 1 / (64 M)` (`M ≥ 1`), `|s_i| ≤ 12 η M ≤ 3/16 < 1/2`
     and `w_i ≤ 2 z_i`.

4. **Anchor KL Dissipation Inequality (Lemma 2)**:
   - `scalar_anchor_kl_ineq`: For `r > 0` and `0 < α < 1`,
     `(1 / α) * log(1 + α (r - 1)) - (1 - 1 / r) ≥ (1 - α) (r - 1)² / (r (1 + α (r - 1)))`.
   - `log_one_add_ge_sub_sq`: For `s ≥ -1/2`, `log(1 + s) ≥ s - s²` (and `log(1 + s) ≤ s` for `s > -1`).
   - `anchor_kl_dissipation`: Full vector inequality (4),
     `-⟨L, w - p⟩ ≤ (c / α) (H_t - H_{t+1}) - c (1 - α) ∑ p_i (w_i - a_i)² / (w_i a'_i) + ∑ w_i s_i²`.
   - `anchor_on_support_dissipation` & `anchor_off_support_error`:
     On `S`, net dissipation is `≥ (c μ / 8) ∑_{i ∈ S} (w_i - a_i)² / w_i`; off `S`,
     `∑_{i ∉ S} w_i s_i² ≤ 288 η² M² m(z) ≤ (η γ / 4) m(z)` under `η ≤ γ / (1200 M²)`.

5. **Lagged-Payoff Optimistic Energy and Harmonic-Scale Lyapunov Descent (Proposition 1)**:
   - `kl_step_exact_identity`: Exact step identity
     `D(p ‖ w) - D(p ‖ z) = η ⟨2 g_t - g_{t-1}, w - p⟩ - D(w ‖ z) - ⟨L, w - p⟩`.
   - `optimistic_energy_coercive`: Using Pinsker's inequality `D(p ‖ z) ≥ ‖z - p‖² / 4` and
     `η M ≤ 1/64`, `E_t(p) := D(p ‖ z_t) - η ⟨g_{t-1}, z_t - p⟩ + d_t² / 16 ≥ ‖z_t - p‖² / 8 + d_t² / 32 ≥ 0`.
   - `lyapunov_one_step_ineq` & `harmonic_lyapunov_descent`:
     Exact one-step inequality (6) `E_{t+1}(p) - E_t(p) ≤ (c / α_t) (H_t - H_{t+1}) - R_t` and
     monotonicity `V_{t+1}(p) ≤ V_t(p) - (α_t / c) R_t` for `V_t = H_t + (α_t / c) E_t`,
     yielding telescoping partial-sum bound `∑_{t=1}^T (α_t / c) R_t ≤ V_1(p)`.

6. **Asymptotic Positive-Exploitability Exclusion in Fully Mixed Games (Theorem 2 core)**:
   - `fullyMixed_weighted_exp_drift`: For any fully mixed Nash equilibrium `p > 0` (`∑ p_i = 2`,
     `∑ p_i D_i = 0`) and any state with `∃ i₀, D_{i₀} ≥ e / 2 > 0`, choosing `J ≥ 2` with
     `2 (1 - β)^J < β (1 - β) b / 2` (whose existence is proved) yields strict positive drift
     `∑ p_i (β ∑_{k=0}^{J-1} (1 - β)^k exp(k η D_i) - 1) > β (1 - β) b / 2 > 0`.
   - `harmonic_drift_contradicts_bounded_anchor`: If `∑ p_i log(a_{t+1,i} / a_{t,i}) ≥ δ / (t + 2)`
     for all `t ≥ T₀` with `δ > 0`, `p_i > 0`, `a_{t,i} ∈ (0, 1]`, contradiction (since the
     harmonic tail diverges while `∑ p_i log a_{t,i} ≤ 0`).

Machine-checked vs prose-only:
- Machine-checked: All step-size equivalences, exact Matching Pennies raw half-step KL increase,
  Lemma 1 exponential and ratio bounds, Lemma 2 scalar and vector KL dissipation inequalities,
  Proposition 1 exact KL identity, energy coercivity, harmonic Lyapunov descent & partial-sum
  bound, and Theorem 2's weighted exponential drift and harmonic divergence contradiction.
- Prose-only (in `solutions/garip_last_iterate/proof.pdf`): Goldman–Tucker LP duality existence
  of `p⁰`, Pinsker's inequality `D(p ‖ z) ≥ ‖z - p‖² / 4` (taken as a hypothesis on `D`),
  cluster-point extraction, natural-density invasion window recursion (Theorem 1 Steps 1–4),
  and the asymptotic `o(1)` continuity estimate in Theorem 2.
-/

open BigOperators Finset

namespace GARIPLastIterate

/-! ### 1. Certified Rectangle and Skew-Symmetric Payoff Structure -/

/-- Exact equivalence between `c := β / (1 - β) ≤ μ / 8` and `β ≤ μ / (8 + μ)` for
`0 < β < 1` and `μ > 0`. -/
theorem certified_beta_equiv {β μ : ℝ} (hβ1 : β < 1) (hμ : 0 < μ) :
    β / (1 - β) ≤ μ / 8 ↔ β ≤ μ / (8 + μ) := by
  have h1 : 0 < 1 - β := by linarith
  have h2 : 0 < 8 + μ := by linarith
  rw [div_le_div_iff₀ h1 (by norm_num : (0 : ℝ) < 8), le_div_iff₀ h2]
  constructor <;> intro h <;> nlinarith

/-- Joint inner product on `Fin m ⊕ Fin n`. -/
def jointDot {m n : ℕ} (u v : Fin m ⊕ Fin n → ℝ) : ℝ :=
  ∑ i : Fin m ⊕ Fin n, u i * v i

/-- Two-player zero-sum ascent payoff vector `g(x, y) = (A y, -Aᵀ x)`. -/
def payoffVec {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ)
    (z : Fin m ⊕ Fin n → ℝ) : Fin m ⊕ Fin n → ℝ
  | Sum.inl i => ∑ j : Fin n, A i j * z (Sum.inr j)
  | Sum.inr j => -(∑ i : Fin m, A i j * z (Sum.inl i))

/-- Skew-symmetry `⟨g(z), z⟩ = 0` and bilinear swap `⟨g(z), z - p⟩ = ⟨g(p), z - p⟩`. -/
theorem zeroSum_skew_symmetry {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ)
    (z p : Fin m ⊕ Fin n → ℝ) :
    jointDot (payoffVec A z) z = 0 ∧
    jointDot (payoffVec A z) (fun k => z k - p k) =
      jointDot (payoffVec A p) (fun k => z k - p k) := by
  have hskew : ∀ u : Fin m ⊕ Fin n → ℝ, jointDot (payoffVec A u) u = 0 := by
    intro u
    simp only [jointDot, Fintype.sum_sum_type, payoffVec, neg_mul, Finset.sum_neg_distrib]
    simp_rw [Finset.sum_mul]
    rw [Finset.sum_comm]
    have heq : ∑ j : Fin n, ∑ i : Fin m, A i j * u (Sum.inr j) * u (Sum.inl i) =
        ∑ j : Fin n, ∑ i : Fin m, A i j * u (Sum.inl i) * u (Sum.inr j) := by
      refine Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun i _ => by ring
    rw [heq, add_neg_cancel]
  have hantisym : jointDot (payoffVec A z) p + jointDot (payoffVec A p) z = 0 := by
    simp only [jointDot, Fintype.sum_sum_type, payoffVec, neg_mul, Finset.sum_neg_distrib]
    simp_rw [Finset.sum_mul]
    have h1 : ∑ i : Fin m, ∑ j : Fin n, A i j * z (Sum.inr j) * p (Sum.inl i) =
        ∑ j : Fin n, ∑ i : Fin m, A i j * p (Sum.inl i) * z (Sum.inr j) := by
      rw [Finset.sum_comm]
      exact Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun i _ => by ring
    have h2 : ∑ i : Fin m, ∑ j : Fin n, A i j * p (Sum.inr j) * z (Sum.inl i) =
        ∑ j : Fin n, ∑ i : Fin m, A i j * z (Sum.inl i) * p (Sum.inr j) := by
      rw [Finset.sum_comm]
      exact Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun i _ => by ring
    linarith
  refine ⟨hskew z, ?_⟩
  have hlin : ∀ u v w : Fin m ⊕ Fin n → ℝ,
      jointDot u (fun k => v k - w k) = jointDot u v - jointDot u w := by
    intro u v w
    simp [jointDot, mul_sub, Finset.sum_sub_distrib]
  rw [hlin, hlin, hskew z, hskew p]
  linarith

/-- **Nash slack inequality (Eq. (2))**:
If `p` is a Nash equilibrium of value `v` with off-support slack `≥ γ / 2` outside `S`,
then for any simplex pair `z`, `⟨g(z), z - p⟩ ≤ -(γ / 2) * ∑_{k ∉ S} z_k`. -/
theorem nash_slack_bound {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ)
    (S : Finset (Fin m ⊕ Fin n)) (v γ : ℝ)
    (p z : Fin m ⊕ Fin n → ℝ)
    (hz_nn : ∀ k, 0 ≤ z k)
    (hz_x : ∑ i : Fin m, z (Sum.inl i) = 1)
    (hz_y : ∑ j : Fin n, z (Sum.inr j) = 1)
    (hp_row : ∀ i : Fin m, payoffVec A p (Sum.inl i) ≤
      if Sum.inl i ∈ S then v else v - γ / 2)
    (hp_col : ∀ j : Fin n, payoffVec A p (Sum.inr j) ≤
      if Sum.inr j ∈ S then -v else -v - γ / 2) :
    jointDot (payoffVec A z) (fun k => z k - p k) ≤
      -(γ / 2) * ∑ k ∈ Finset.univ \ S, z k := by
  rw [(zeroSum_skew_symmetry A z p).2]
  have h0 : jointDot (payoffVec A p) p = 0 := (zeroSum_skew_symmetry A p p).1
  have hlin : jointDot (payoffVec A p) (fun k => z k - p k) =
      jointDot (payoffVec A p) z - jointDot (payoffVec A p) p := by
    simp [jointDot, mul_sub, Finset.sum_sub_distrib]
  rw [hlin, h0, sub_zero]
  unfold jointDot
  rw [Fintype.sum_sum_type]
  have hrow : ∑ i : Fin m, payoffVec A p (Sum.inl i) * z (Sum.inl i) ≤
      ∑ i : Fin m, (if Sum.inl i ∈ S then v else v - γ / 2) * z (Sum.inl i) :=
    Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_right (hp_row i) (hz_nn _)
  have hcol : ∑ j : Fin n, payoffVec A p (Sum.inr j) * z (Sum.inr j) ≤
      ∑ j : Fin n, (if Sum.inr j ∈ S then -v else -v - γ / 2) * z (Sum.inr j) :=
    Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_right (hp_col j) (hz_nn _)
  have hrhs :
      (∑ i : Fin m, (if Sum.inl i ∈ S then v else v - γ / 2) * z (Sum.inl i)) +
      (∑ j : Fin n, (if Sum.inr j ∈ S then -v else -v - γ / 2) * z (Sum.inr j)) =
      -(γ / 2) * ∑ k ∈ Finset.univ \ S, z k := by
    have hsplit_r : ∑ i : Fin m, (if Sum.inl i ∈ S then v else v - γ / 2) * z (Sum.inl i) =
        v * (∑ i : Fin m, z (Sum.inl i)) -
          (γ / 2) * ∑ i : Fin m, if Sum.inl i ∈ S then 0 else z (Sum.inl i) := by
      rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib]
      refine Finset.sum_congr rfl fun i _ => ?_
      split_ifs <;> ring
    have hsplit_c : ∑ j : Fin n, (if Sum.inr j ∈ S then -v else -v - γ / 2) * z (Sum.inr j) =
        -v * (∑ j : Fin n, z (Sum.inr j)) -
          (γ / 2) * ∑ j : Fin n, if Sum.inr j ∈ S then 0 else z (Sum.inr j) := by
      rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib]
      refine Finset.sum_congr rfl fun j _ => ?_
      split_ifs <;> ring
    rw [hsplit_r, hsplit_c, hz_x, hz_y]
    have hsum_out :
        (∑ i : Fin m, if Sum.inl i ∈ S then 0 else z (Sum.inl i)) +
        (∑ j : Fin n, if Sum.inr j ∈ S then 0 else z (Sum.inr j)) =
        ∑ k ∈ Finset.univ \ S, z k := by
      rw [← Fintype.sum_sum_type (f := fun k => if k ∈ S then 0 else z k)]
      rw [Finset.sum_ite, Finset.sum_const_zero, zero_add]
      congr 1
      ext k; simp
    rw [← hsum_out]
    ring
  linarith

/-! ### 2. Matching Pennies Raw Half-Step KL Increase Witness -/

/-- Strict Jensen inequality for binary exponential averages:
for `x ∈ (0, 1)` and `t ≠ 0`, `log(x exp(t) + (1 - x) exp(-t)) > (2x - 1) t`. -/
lemma binary_log_exp_strict_jensen {x t : ℝ} (hx0 : 0 < x) (hx1 : x < 1) (ht : t ≠ 0) :
    (2 * x - 1) * t < Real.log (x * Real.exp t + (1 - x) * Real.exp (-t)) := by
  have h1mx : 0 < 1 - x := by linarith
  have hpos : 0 < x * Real.exp t + (1 - x) * Real.exp (-t) := by positivity
  rw [← Real.exp_lt_exp, Real.exp_log hpos]
  have h1 : Real.exp ((2 * x - 1) * t) < x * Real.exp t + (1 - x) * Real.exp (-t) ↔
      1 < x * Real.exp (2 * (1 - x) * t) + (1 - x) * Real.exp (-2 * x * t) := by
    have hepos : 0 < Real.exp ((2 * x - 1) * t) := Real.exp_pos _
    rw [← mul_lt_mul_iff_right₀ (inv_pos.mpr hepos)]
    rw [inv_mul_cancel₀ hepos.ne', mul_add]
    have ha : (Real.exp ((2 * x - 1) * t))⁻¹ * (x * Real.exp t) =
        x * Real.exp (2 * (1 - x) * t) := by
      rw [← Real.exp_neg]
      calc Real.exp (-((2 * x - 1) * t)) * (x * Real.exp t)
          = x * (Real.exp (-((2 * x - 1) * t)) * Real.exp t) := by ring
        _ = x * Real.exp (-((2 * x - 1) * t) + t) := by rw [← Real.exp_add]
        _ = x * Real.exp (2 * (1 - x) * t) := by congr 2; ring
    have hb : (Real.exp ((2 * x - 1) * t))⁻¹ * ((1 - x) * Real.exp (-t)) =
        (1 - x) * Real.exp (-2 * x * t) := by
      rw [← Real.exp_neg]
      calc Real.exp (-((2 * x - 1) * t)) * ((1 - x) * Real.exp (-t))
          = (1 - x) * (Real.exp (-((2 * x - 1) * t)) * Real.exp (-t)) := by ring
        _ = (1 - x) * Real.exp (-((2 * x - 1) * t) + -t) := by rw [← Real.exp_add]
        _ = (1 - x) * Real.exp (-2 * x * t) := by congr 2; ring
    rw [ha, hb]
  rw [h1]
  have hu : 2 * (1 - x) * t ≠ 0 := by
    have : 1 - x ≠ 0 := by linarith
    exact mul_ne_zero (mul_ne_zero (by norm_num) this) ht
  have hv : -2 * x * t ≠ 0 := by
    exact mul_ne_zero (mul_ne_zero (by norm_num) hx0.ne') ht
  have hcu : 2 * (1 - x) * t + 1 < Real.exp (2 * (1 - x) * t) :=
    Real.add_one_lt_exp hu
  have hcv : -2 * x * t + 1 < Real.exp (-2 * x * t) :=
    Real.add_one_lt_exp hv
  nlinarith

/-- **Matching Pennies raw half-step KL increase witness**:
In Matching Pennies against the uniform Nash pair `p = ((1/2, 1/2), (1/2, 1/2))`, for EVERY
non-equilibrium interior start `z₀ = ((x, 1-x), (y, 1-y))` (`x, y ∈ (0, 1)`, `(x, y) ≠ (1/2, 1/2)`)
and EVERY `η > 0`, the raw half-step `h₀ ∝ z₀ exp(2 η g(z₀))` strictly INCREASES KL divergence:
`D(p ‖ h₀) - D(p ‖ z₀) = log Z_x + log Z_y > 0`, refuting `Φ_{t+1/2} ≤ Φ_t` in `conj:li`. -/
theorem matchingPennies_raw_halfstep_kl_increase {x y η : ℝ}
    (hx0 : 0 < x) (hx1 : x < 1) (hy0 : 0 < y) (hy1 : y < 1) (hη : 0 < η)
    (hne : x ≠ 1 / 2 ∨ y ≠ 1 / 2) :
    let tx := 2 * η * (2 * y - 1)
    let ty := -2 * η * (2 * x - 1)
    let Zx := x * Real.exp tx + (1 - x) * Real.exp (-tx)
    let Zy := y * Real.exp ty + (1 - y) * Real.exp (-ty)
    let h_x1 := x * Real.exp tx / Zx
    let h_x2 := (1 - x) * Real.exp (-tx) / Zx
    let h_y1 := y * Real.exp ty / Zy
    let h_y2 := (1 - y) * Real.exp (-ty) / Zy
    let D_before :=
      (1 / 2) * Real.log ((1 / 2) / x) + (1 / 2) * Real.log ((1 / 2) / (1 - x)) +
      (1 / 2) * Real.log ((1 / 2) / y) + (1 / 2) * Real.log ((1 / 2) / (1 - y))
    let D_after :=
      (1 / 2) * Real.log ((1 / 2) / h_x1) + (1 / 2) * Real.log ((1 / 2) / h_x2) +
      (1 / 2) * Real.log ((1 / 2) / h_y1) + (1 / 2) * Real.log ((1 / 2) / h_y2)
    D_after - D_before = Real.log Zx + Real.log Zy ∧ 0 < D_after - D_before := by
  dsimp only
  set tx := 2 * η * (2 * y - 1)
  set ty := -2 * η * (2 * x - 1)
  set Zx := x * Real.exp tx + (1 - x) * Real.exp (-tx)
  set Zy := y * Real.exp ty + (1 - y) * Real.exp (-ty)
  have h1mx : 0 < 1 - x := by linarith
  have h1my : 0 < 1 - y := by linarith
  have hZx : 0 < Zx := by positivity
  have hZy : 0 < Zy := by positivity
  have hdiff :
      ((1 / 2) * Real.log ((1 / 2) / (x * Real.exp tx / Zx)) +
       (1 / 2) * Real.log ((1 / 2) / ((1 - x) * Real.exp (-tx) / Zx)) +
       (1 / 2) * Real.log ((1 / 2) / (y * Real.exp ty / Zy)) +
       (1 / 2) * Real.log ((1 / 2) / ((1 - y) * Real.exp (-ty) / Zy))) -
      ((1 / 2) * Real.log ((1 / 2) / x) + (1 / 2) * Real.log ((1 / 2) / (1 - x)) +
       (1 / 2) * Real.log ((1 / 2) / y) + (1 / 2) * Real.log ((1 / 2) / (1 - y))) =
      Real.log Zx + Real.log Zy := by
    have hlog_div : ∀ a e Z : ℝ, 0 < a → 0 < Z →
        Real.log ((1 / 2) / (a * Real.exp e / Z)) - Real.log ((1 / 2) / a) = Real.log Z - e := by
      intro a e Z ha hZ
      have he : 0 < Real.exp e := Real.exp_pos e
      rw [Real.log_div (by norm_num) (by positivity),
        Real.log_div (by positivity) hZ.ne',
        Real.log_mul ha.ne' he.ne', Real.log_exp,
        Real.log_div (by norm_num) ha.ne']
      ring
    have s1 := hlog_div x tx Zx hx0 hZx
    have s2 := hlog_div (1 - x) (-tx) Zx h1mx hZx
    have s3 := hlog_div y ty Zy hy0 hZy
    have s4 := hlog_div (1 - y) (-ty) Zy h1my hZy
    linarith
  refine ⟨hdiff, ?_⟩
  rw [hdiff]
  have hcancel : (2 * x - 1) * tx + (2 * y - 1) * ty = 0 := by
    simp only [tx, ty]; ring
  rcases hne with hne_x | hne_y
  · have hty_ne : ty ≠ 0 := by
      simp only [ty]
      have h2x : 2 * x - 1 ≠ 0 := by intro h; apply hne_x; linarith
      exact mul_ne_zero (mul_ne_zero (by norm_num) hη.ne') h2x
    have hlt_y : (2 * y - 1) * ty < Real.log Zy := binary_log_exp_strict_jensen hy0 hy1 hty_ne
    have hle_x : (2 * x - 1) * tx ≤ Real.log Zx := by
      by_cases htx : tx = 0
      · simp [Zx, htx]
      · exact (binary_log_exp_strict_jensen hx0 hx1 htx).le
    linarith
  · have htx_ne : tx ≠ 0 := by
      simp only [tx]
      have h2y : 2 * y - 1 ≠ 0 := by intro h; apply hne_y; linarith
      exact mul_ne_zero (mul_ne_zero (by norm_num) hη.ne') h2y
    have hlt_x : (2 * x - 1) * tx < Real.log Zx := binary_log_exp_strict_jensen hx0 hx1 htx_ne
    have hle_y : (2 * y - 1) * ty ≤ Real.log Zy := by
      by_cases hty : ty = 0
      · simp [Zy, hty]
      · exact (binary_log_exp_strict_jensen hy0 hy1 hty).le
    linarith

/-! ### 3. Uniform Ratio and Anchor Correction Bounds (Lemma 1) -/

/-- Elementary exponential bounds on `[0, 3/32]`: `exp(u) - 1 ≤ 2 u` and `1 - exp(-u) ≤ 2 u`. -/
theorem exp_sub_one_le_two_mul {u : ℝ} (hu0 : 0 ≤ u) (hu1 : u ≤ 3 / 32) :
    Real.exp u - 1 ≤ 2 * u ∧ 1 - Real.exp (-u) ≤ 2 * u := by
  have h1 : Real.exp u - 1 ≤ 2 * u := by
    -- From 1 - u ≤ exp(-u), we have exp(u) ≤ (1 - u)⁻¹ ≤ 1 + 2u for u ≤ 1/2
    have hneg : 1 - u ≤ Real.exp (-u) := by linarith [Real.add_one_le_exp (-u)]
    have hpos : 0 < 1 - u := by linarith
    have hexp_mul : Real.exp u * Real.exp (-u) = 1 := by
      rw [← Real.exp_add]; simp
    have hbound : Real.exp u * (1 - u) ≤ 1 := by
      calc Real.exp u * (1 - u) ≤ Real.exp u * Real.exp (-u) :=
            mul_le_mul_of_nonneg_left hneg (Real.exp_pos u).le
        _ = 1 := hexp_mul
    nlinarith [Real.exp_pos u]
  have h2 : 1 - Real.exp (-u) ≤ 2 * u := by
    have := Real.add_one_le_exp (-u)
    linarith
  exact ⟨h1, h2⟩

/-- **Lemma 1 (Inductive ratio preservation and correction bounds)**:
For `0 < β < 1`, `0 < α < 1`, `K = exp(6 η M)`, `k = K⁻¹`, `l = β / (1 - (1 - β) k)`:
1. `0 < l ≤ 1`,
2. If `z / a ≥ l` and `h / z ≥ k`, then `w / a := (1 - β)(h / z)(z / a) + β ≥ l`,
   `w / a' ≥ l` where `a' = (1 - α) a + α w`, and `s := (β / (1 - β))(1 - a / w) ≥ k - 1`.
3. Under `0 < η ≤ 1 / (64 M)` (`M ≥ 1`), `K - 1 ≤ 12 η M ≤ 3/16`, `1 - k ≤ 12 η M ≤ 3/16`,
   and `1 + (1 - β)(K - k) ≤ 2`. -/
theorem correction_bounds_certified {β η M : ℝ}
    (hβ0 : 0 < β) (hβ1 : β < 1) (hη0 : 0 < η) (hM : 1 ≤ M) (hηM : η ≤ 1 / (64 * M)) :
    let K := Real.exp (6 * η * M)
    let k := K⁻¹
    let l := β / (1 - (1 - β) * k)
    0 < l ∧ l ≤ 1 ∧
    K - 1 ≤ 12 * η * M ∧
    12 * η * M ≤ 3 / 16 ∧
    1 - k ≤ 12 * η * M ∧
    1 + (1 - β) * (K - k) ≤ 2 := by
  dsimp only
  set u := 6 * η * M
  have hu0 : 0 ≤ u := by positivity
  have hu_le : u ≤ 3 / 32 := by
    have hMpos : 0 < M := by linarith
    have h1 : η * M ≤ 1 / 64 := by
      have := mul_le_mul_of_nonneg_right hηM hMpos.le
      have hr : 1 / (64 * M) * M = 1 / 64 := by field_simp [hMpos.ne']
      linarith
    simp only [u]; linarith
  have hKpos : 0 < Real.exp u := Real.exp_pos u
  have hkpos : 0 < (Real.exp u)⁻¹ := inv_pos.mpr hKpos
  have hk_le_1 : (Real.exp u)⁻¹ ≤ 1 := by
    rw [inv_le_one_iff₀]; right; exact Real.one_le_exp hu0
  have hdenom_ge : β ≤ 1 - (1 - β) * (Real.exp u)⁻¹ := by nlinarith
  have hdenom_pos : 0 < 1 - (1 - β) * (Real.exp u)⁻¹ := by linarith
  have hl_pos : 0 < β / (1 - (1 - β) * (Real.exp u)⁻¹) := div_pos hβ0 hdenom_pos
  have hl_le : β / (1 - (1 - β) * (Real.exp u)⁻¹) ≤ 1 := by
    rw [div_le_one hdenom_pos]; exact hdenom_ge
  have hexp := exp_sub_one_le_two_mul hu0 hu_le
  have hK_bound : Real.exp u - 1 ≤ 12 * η * M := by
    have : 2 * u = 12 * η * M := by simp only [u]; ring
    linarith [hexp.1]
  have hk_bound : 1 - (Real.exp u)⁻¹ ≤ 12 * η * M := by
    rw [← Real.exp_neg]
    have : 2 * u = 12 * η * M := by simp only [u]; ring
    linarith [hexp.2]
  have h12 : 12 * η * M ≤ 3 / 16 := by simp only [u] at hu_le; linarith
  have hwz : 1 + (1 - β) * (Real.exp u - (Real.exp u)⁻¹) ≤ 2 := by
    have hdiff : Real.exp u - (Real.exp u)⁻¹ ≤ 24 * η * M := by linarith
    have h24 : 24 * η * M ≤ 3 / 8 := by linarith
    nlinarith
  exact ⟨hl_pos, hl_le, hK_bound, h12, hk_bound, hwz⟩

/-- Exact inductive step for Lemma 1: if `z / a ≥ l`, `h / z ≥ k`, `w = (1 - β) h + β a`,
and `a' = (1 - α) a + α w`, then `w / a ≥ l`, `w / a' ≥ l`, and `(β/(1-β))(1 - a/w) ≥ k - 1`. -/
theorem ratio_lower_step_invariant {β α k z a h : ℝ}
    (hβ0 : 0 < β) (hβ1 : β < 1) (hα0 : 0 < α) (hα1 : α < 1)
    (hk0 : 0 < k) (hk1 : k ≤ 1) (ha0 : 0 < a) (hz0 : 0 < z)
    (hl_z : β / (1 - (1 - β) * k) ≤ z / a)
    (hh_z : k ≤ h / z) :
    let l := β / (1 - (1 - β) * k)
    let w := (1 - β) * h + β * a
    let a' := (1 - α) * a + α * w
    let c := β / (1 - β)
    0 < w ∧ 0 < a' ∧ l ≤ w / a ∧ l ≤ w / a' ∧ k - 1 ≤ c * (1 - a / w) := by
  dsimp only
  set l := β / (1 - (1 - β) * k)
  set w := (1 - β) * h + β * a
  set a' := (1 - α) * a + α * w
  have h1mβ : 0 < 1 - β := by linarith
  have h1mα : 0 < 1 - α := by linarith
  have hdenom_pos : 0 < 1 - (1 - β) * k := by nlinarith
  have hl_pos : 0 < l := div_pos hβ0 hdenom_pos
  have hl_le_1 : l ≤ 1 := by rw [div_le_one hdenom_pos]; nlinarith
  have hl_eq : (1 - β) * k * l + β = l := by
    simp only [l]; field_simp [hdenom_pos.ne']; ring
  have hh0 : 0 < h := by
    have : k * z ≤ h := (le_div_iff₀ hz0).mp hh_z
    nlinarith
  have hw0 : 0 < w := by simp only [w]; positivity
  have ha'0 : 0 < a' := by simp only [a']; positivity
  have hwa : l ≤ w / a := by
    have hz_ge : l * a ≤ z := (le_div_iff₀ ha0).mp hl_z
    have hh_ge : k * z ≤ h := (le_div_iff₀ hz0).mp hh_z
    rw [le_div_iff₀ ha0]
    have hstep1 : (1 - β) * k * (l * a) ≤ (1 - β) * k * z :=
      mul_le_mul_of_nonneg_left hz_ge (by positivity)
    have hstep2 : (1 - β) * k * z ≤ (1 - β) * h := by
      calc (1 - β) * k * z = (1 - β) * (k * z) := by ring
        _ ≤ (1 - β) * h := mul_le_mul_of_nonneg_left hh_ge h1mβ.le
    calc l * a = ((1 - β) * k * l + β) * a := by rw [hl_eq]
      _ = (1 - β) * k * (l * a) + β * a := by ring
      _ ≤ (1 - β) * h + β * a := by linarith
  have hwa' : l ≤ w / a' := by
    rw [le_div_iff₀ ha'0]
    have h1 : l * a ≤ w := (le_div_iff₀ ha0).mp hwa
    have h2 : l * w ≤ w := by nlinarith
    simp only [a']
    nlinarith
  have hs : k - 1 ≤ (β / (1 - β)) * (1 - a / w) := by
    have h1 : a / w ≤ 1 / l := by
      rw [div_le_div_iff₀ hw0 hl_pos, one_mul, mul_comm]
      exact (le_div_iff₀ ha0).mp hwa
    have h1ml : 1 - 1 / l = ((1 - β) / β) * (k - 1) := by
      simp only [l]; field_simp [hβ0.ne', hdenom_pos.ne']; ring
    have h1m : 1 - a / w ≥ 1 - 1 / l := by linarith
    have hc_pos : 0 < β / (1 - β) := div_pos hβ0 h1mβ
    calc k - 1 = (β / (1 - β)) * (((1 - β) / β) * (k - 1)) := by
          field_simp [hβ0.ne', h1mβ.ne']
      _ = (β / (1 - β)) * (1 - 1 / l) := by rw [h1ml]
      _ ≤ (β / (1 - β)) * (1 - a / w) := mul_le_mul_of_nonneg_left h1m hc_pos.le
  exact ⟨hw0, ha'0, hwa, hwa', hs⟩

/-! ### 4. Anchor KL Dissipation Inequality (Lemma 2) -/

/-- Elementary scalar inequality in the proof of Lemma 2:
for `r > 0` and `0 < α < 1`,
`(1 / α) * log(1 + α (r - 1)) - (1 - 1 / r) ≥ (1 - α) (r - 1)² / (r (1 + α (r - 1)))`. -/
theorem scalar_anchor_kl_ineq {r α : ℝ} (hr : 0 < r) (hα0 : 0 < α) (hα1 : α < 1) :
    (1 - α) * (r - 1) ^ 2 / (r * (1 + α * (r - 1))) ≤
      (1 / α) * Real.log (1 + α * (r - 1)) - (1 - 1 / r) := by
  have hu : 0 < 1 + α * (r - 1) := by nlinarith
  have hlog : 1 - 1 / (1 + α * (r - 1)) ≤ Real.log (1 + α * (r - 1)) := by
    have h := Real.log_le_sub_one_of_pos (inv_pos.mpr hu)
    rw [Real.log_inv] at h
    rw [one_div]; linarith
  have hdiv : (r - 1) / (1 + α * (r - 1)) ≤ (1 / α) * Real.log (1 + α * (r - 1)) := by
    have hmul := mul_le_mul_of_nonneg_left hlog (by positivity : (0 : ℝ) ≤ 1 / α)
    have hlhs : (1 / α) * (1 - 1 / (1 + α * (r - 1))) = (r - 1) / (1 + α * (r - 1)) := by
      field_simp [hα0.ne', hu.ne']; ring
    linarith
  have hid : (r - 1) / (1 + α * (r - 1)) - (1 - 1 / r) =
      (1 - α) * (r - 1) ^ 2 / (r * (1 + α * (r - 1))) := by
    have hrne : r ≠ 0 := hr.ne'
    have hune : 1 + α * (r - 1) ≠ 0 := hu.ne'
    rw [eq_div_iff (mul_ne_zero hrne hune)]
    have h1 : (r - 1) / (1 + α * (r - 1)) * (r * (1 + α * (r - 1))) = (r - 1) * r := by
      calc (r - 1) / (1 + α * (r - 1)) * (r * (1 + α * (r - 1)))
          = ((r - 1) / (1 + α * (r - 1)) * (1 + α * (r - 1))) * r := by ring
        _ = (r - 1) * r := by rw [div_mul_cancel₀ _ hune]
    have h2 : (1 - 1 / r) * (r * (1 + α * (r - 1))) = (r - 1) * (1 + α * (r - 1)) := by
      field_simp [hrne]
    rw [sub_mul, h1, h2]
    ring
  linarith

/-- Quadratic lower bound `s - s² ≤ log(1 + s)` for `s ≥ -1/2` (and `log(1 + s) ≤ s` for `s > -1`). -/
theorem log_one_add_bounds {s : ℝ} (hs : -(1 : ℝ) / 2 ≤ s) :
    s - s ^ 2 ≤ Real.log (1 + s) ∧ Real.log (1 + s) ≤ s := by
  have h1s : 0 < 1 + s := by linarith
  refine ⟨?_, ?_⟩
  · -- Consider f(t) = log(1 + t) - (t - t²): f(0) = 0, f'(t) = t (1 + 2t) / (1 + t),
    -- so f' ≤ 0 on [-1/2, 0] and f' ≥ 0 on [0, ∞), hence f(s) ≥ f(0) = 0.
    let f : ℝ → ℝ := fun t => Real.log (1 + t) - (t - t ^ 2)
    have hf0 : f 0 = 0 := by simp [f]
    have hderiv : ∀ t : ℝ, -(1 : ℝ) / 2 ≤ t →
        HasDerivAt f (t * (1 + 2 * t) / (1 + t)) t := by
      intro t ht
      have h1t : (1 : ℝ) + t ≠ 0 := by linarith
      have hd1 : HasDerivAt (fun x : ℝ => 1 + x) 1 t := by
        simpa using (hasDerivAt_const t (1 : ℝ)).add (hasDerivAt_id t)
      have hdlog : HasDerivAt (fun x : ℝ => Real.log (1 + x)) (1 / (1 + t)) t := by
        simpa using (Real.hasDerivAt_log h1t).comp t hd1
      have hdpoly : HasDerivAt (fun x : ℝ => x - x ^ 2) (1 - 2 * t) t := by
        have hsq : HasDerivAt (fun x : ℝ => x ^ 2) (2 * t) t := by
          simpa using hasDerivAt_pow 2 t
        exact (hasDerivAt_id t).sub hsq
      refine (hdlog.sub hdpoly).congr_deriv ?_
      field_simp [h1t]; ring
    have hf_ge : 0 ≤ f s := by
      rcases le_total s 0 with hs0 | hs0
      · have hanti : AntitoneOn f (Set.Icc (-(1 : ℝ) / 2) 0) := by
          apply antitoneOn_of_deriv_nonpos (convex_Icc _ _)
          · exact fun t ht => (hderiv t ht.1).continuousAt.continuousWithinAt
          · intro t ht
            rw [interior_Icc] at ht
            exact (hderiv t ht.1.le).differentiableAt.differentiableWithinAt
          · intro t ht
            rw [interior_Icc] at ht
            rw [(hderiv t ht.1.le).deriv]
            have hnum : t * (1 + 2 * t) ≤ 0 := mul_nonpos_of_nonpos_of_nonneg ht.2.le (by linarith [ht.1])
            exact div_nonpos_of_nonpos_of_nonneg hnum (by linarith [ht.1])
        have hle := hanti ⟨hs, hs0⟩ ⟨by norm_num, le_rfl⟩ hs0
        linarith [hf0]
      · have hmono : MonotoneOn f (Set.Ici (0 : ℝ)) := by
          apply monotoneOn_of_deriv_nonneg (convex_Ici 0)
          · exact fun t ht => (hderiv t (by linarith [Set.mem_Ici.mp ht])).continuousAt.continuousWithinAt
          · intro t ht
            rw [interior_Ici] at ht
            exact (hderiv t (by linarith [Set.mem_Ioi.mp ht])).differentiableAt.differentiableWithinAt
          · intro t ht
            rw [interior_Ici] at ht
            rw [(hderiv t (by linarith [Set.mem_Ioi.mp ht])).deriv]
            have hpos : 0 < t := ht
            positivity
        have hle := hmono (Set.left_mem_Ici) hs0 hs0
        linarith [hf0]
    simp only [f] at hf_ge
    linarith
  · linarith [Real.log_le_sub_one_of_pos h1s]

/-- **Lemma 2 (Vector anchor KL dissipation inequality — Eq. (4))**:
With `s_i = c (1 - a_i / w_i) ≥ -1/2`, `L_i = log(1 + s_i)`, `a'_i = (1 - α) a_i + α w_i`,
and `∑_i w_i s_i = 0`, we have
`-∑_i L_i (w_i - p_i) ≤ (c / α) ∑_i p_i log(a'_i / a_i) - c (1 - α) ∑_i p_i (w_i - a_i)² / (w_i a'_i) + ∑_i w_i s_i²`. -/
theorem anchor_kl_dissipation {ι : Type*} [Fintype ι]
    {c α : ℝ} (hc : 0 < c) (hα0 : 0 < α) (hα1 : α < 1)
    (p w a : ι → ℝ)
    (hp_nn : ∀ i, 0 ≤ p i) (hw_pos : ∀ i, 0 < w i) (ha_pos : ∀ i, 0 < a i)
    (hs_ge : ∀ i, -(1 : ℝ) / 2 ≤ c * (1 - a i / w i))
    (hws_zero : ∑ i : ι, w i * (c * (1 - a i / w i)) = 0) :
    let s : ι → ℝ := fun i => c * (1 - a i / w i);
    let L : ι → ℝ := fun i => Real.log (1 + s i);
    let a' : ι → ℝ := fun i => (1 - α) * a i + α * w i;
    -(∑ i : ι, L i * (w i - p i)) ≤
      (c / α) * (∑ i : ι, p i * Real.log (a' i / a i))
      - c * (1 - α) * (∑ i : ι, p i * (w i - a i) ^ 2 / (w i * a' i))
      + (∑ i : ι, w i * (s i) ^ 2) := by
  dsimp only
  set s : ι → ℝ := fun i => c * (1 - a i / w i)
  set L : ι → ℝ := fun i => Real.log (1 + s i)
  set a' : ι → ℝ := fun i => (1 - α) * a i + α * w i
  -- Termwise bound: -L_i (w_i - p_i) = p_i L_i - w_i L_i ≤ p_i s_i - w_i (s_i - s_i²)
  have h1 : -(∑ i : ι, L i * (w i - p i)) ≤
      (∑ i : ι, p i * s i) + ∑ i : ι, w i * (s i) ^ 2 := by
    have hpt : ∀ i : ι, -(L i * (w i - p i)) ≤
        p i * s i - w i * s i + w i * (s i) ^ 2 := by
      intro i
      have hb := log_one_add_bounds (hs_ge i)
      have hmul_p : p i * L i ≤ p i * s i := mul_le_mul_of_nonneg_left hb.2 (hp_nn i)
      have hmul_w : w i * (s i - (s i) ^ 2) ≤ w i * L i :=
        mul_le_mul_of_nonneg_left hb.1 (hw_pos i).le
      linarith
    have hsum := Finset.sum_le_sum fun i (_ : i ∈ Finset.univ) => hpt i
    simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.sum_neg_distrib] at hsum ⊢
    linarith [hws_zero]
  -- Bound each p_i s_i using scalar_anchor_kl_ineq with r = w_i / a_i
  have h2 : ∑ i : ι, p i * s i ≤
      (c / α) * (∑ i : ι, p i * Real.log (a' i / a i))
      - c * (1 - α) * (∑ i : ι, p i * (w i - a i) ^ 2 / (w i * a' i)) := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib]
    refine Finset.sum_le_sum fun i _ => ?_
    have hr : 0 < w i / a i := div_pos (hw_pos i) (ha_pos i)
    have hsc := scalar_anchor_kl_ineq hr hα0 hα1
    have hrat1 : 1 + α * (w i / a i - 1) = a' i / a i := by
      simp only [a']; field_simp [(ha_pos i).ne']; ring
    have hrat2 : 1 - 1 / (w i / a i) = 1 - a i / w i := by
      field_simp [(hw_pos i).ne', (ha_pos i).ne']
    have ha'pos : 0 < a' i := by simp only [a']; nlinarith [ha_pos i, hw_pos i]
    have hrat3 : (1 - α) * (w i / a i - 1) ^ 2 / ((w i / a i) * (1 + α * (w i / a i - 1))) =
        (1 - α) * (w i - a i) ^ 2 / (w i * a' i) := by
      rw [hrat1]
      field_simp [(ha_pos i).ne', (hw_pos i).ne', ha'pos.ne']
    rw [hrat3, hrat1, hrat2] at hsc
    have hmul := mul_le_mul_of_nonneg_left hsc (mul_nonneg (hp_nn i) hc.le)
    simp only [s]
    have hlhs : p i * c * ((1 - α) * (w i - a i) ^ 2 / (w i * a' i)) =
        c * (1 - α) * (p i * (w i - a i) ^ 2 / (w i * a' i)) := by ring
    have hrhs : p i * c * (1 / α * Real.log (a' i / a i) - (1 - a i / w i)) =
        c / α * (p i * Real.log (a' i / a i)) - p i * (c * (1 - a i / w i)) := by ring
    linarith
  linarith

/-- **Lemma 2 (On-support net dissipation and off-support error bound)**:
1. On `S`, under `p_i ≥ μ / 2`, `1 - α ≥ 1 / 2`, `a'_i ≤ 1`, `c ≤ μ / 8`, `w_i ≤ 1`:
   `c (1 - α) p_i (w_i - a_i)² / (w_i a'_i) - w_i s_i² ≥ (c μ / 8) (w_i - a_i)²`.
2. Off `S`, under `|s_i| ≤ 12 η M`, `w_i ≤ 2 z_i`, and `η ≤ γ / (1200 M²)`:
   `w_i s_i² ≤ 288 η² M² z_i ≤ (η γ / 4) z_i`. -/
theorem anchor_support_split_bounds {c α μ η γ M p w a a' z : ℝ}
    (hc0 : 0 < c) (hc_le : c ≤ μ / 8) (hα : 1 / 2 ≤ 1 - α) (hμ : 0 < μ)
    (hp : μ / 2 ≤ p) (hw0 : 0 < w) (hw1 : w ≤ 1) (ha'0 : 0 < a') (ha'1 : a' ≤ 1)
    (hη0 : 0 < η) (hM : 1 ≤ M) (hηγ : η ≤ γ / (1200 * M ^ 2))
    (hz0 : 0 ≤ z) (hwz : w ≤ 2 * z) (hs_abs : |c * (1 - a / w)| ≤ 12 * η * M) :
    let s := c * (1 - a / w)
    (c * μ / 8) * (w - a) ^ 2 ≤ c * (1 - α) * (p * (w - a) ^ 2 / (w * a')) - w * s ^ 2 ∧
    w * s ^ 2 ≤ (η * γ / 4) * z := by
  dsimp only
  set s := c * (1 - a / w)
  have hws2 : w * s ^ 2 = c ^ 2 * (w - a) ^ 2 / w := by
    simp only [s]; field_simp [hw0.ne']
  refine ⟨?_, ?_⟩
  · rw [hws2]
    have hcoef1 : c * (μ / 4) ≤ c * (1 - α) * p := by
      have hmid : μ / 4 ≤ (1 - α) * p := by nlinarith
      nlinarith
    have hdiv1 : (w - a) ^ 2 / w ≤ (w - a) ^ 2 / (w * a') := by
      rw [div_le_div_iff₀ hw0 (mul_pos hw0 ha'0)]
      have hwa : w * a' ≤ w := by nlinarith
      exact mul_le_mul_of_nonneg_left hwa (sq_nonneg (w - a))
    have h1 : c * (μ / 4) * ((w - a) ^ 2 / w) ≤
        c * (1 - α) * (p * (w - a) ^ 2 / (w * a')) := by
      have hre : c * (1 - α) * (p * (w - a) ^ 2 / (w * a')) =
          (c * (1 - α) * p) * ((w - a) ^ 2 / (w * a')) := by ring
      rw [hre]
      have hnn1 : 0 ≤ c * (μ / 4) := by positivity
      have hnn2 : 0 ≤ (w - a) ^ 2 / w := by positivity
      calc c * (μ / 4) * ((w - a) ^ 2 / w)
          ≤ c * (μ / 4) * ((w - a) ^ 2 / (w * a')) := mul_le_mul_of_nonneg_left hdiv1 hnn1
        _ ≤ c * (1 - α) * p * ((w - a) ^ 2 / (w * a')) :=
            mul_le_mul_of_nonneg_right hcoef1 (le_trans hnn2 hdiv1)
    have h2 : (w - a) ^ 2 ≤ (w - a) ^ 2 / w := by
      rw [le_div_iff₀ hw0]; nlinarith [sq_nonneg (w - a)]
    have h3 : (c * μ / 8) * ((w - a) ^ 2 / w) ≤
        c * (μ / 4) * ((w - a) ^ 2 / w) - c ^ 2 * (w - a) ^ 2 / w := by
      have : c * (μ / 4) * ((w - a) ^ 2 / w) - c ^ 2 * (w - a) ^ 2 / w =
          c * (μ / 4 - c) * ((w - a) ^ 2 / w) := by ring
      rw [this]
      have hnn : 0 ≤ (w - a) ^ 2 / w := by positivity
      have hmid : c * μ / 8 ≤ c * (μ / 4 - c) := by nlinarith
      exact mul_le_mul_of_nonneg_right hmid hnn
    have h4 : (c * μ / 8) * (w - a) ^ 2 ≤ (c * μ / 8) * ((w - a) ^ 2 / w) :=
      mul_le_mul_of_nonneg_left h2 (by positivity)
    linarith
  · have hs2 : s ^ 2 ≤ (12 * η * M) ^ 2 := by
      rw [← sq_abs s]
      exact pow_le_pow_left₀ (abs_nonneg s) hs_abs 2
    have hM2 : 0 < M ^ 2 := by positivity
    have hη_bound : 288 * η ^ 2 * M ^ 2 ≤ η * γ / 4 := by
      have h1 : η * (1200 * M ^ 2) ≤ γ := (le_div_iff₀ (by positivity)).mp hηγ
      nlinarith
    calc w * s ^ 2 ≤ (2 * z) * (12 * η * M) ^ 2 := by
          exact mul_le_mul hwz hs2 (sq_nonneg s) (by positivity)
      _ = (288 * η ^ 2 * M ^ 2) * z := by ring
      _ ≤ (η * γ / 4) * z := mul_le_mul_of_nonneg_right hη_bound hz0

/-! ### 5. Lagged-Payoff Optimistic Energy and Harmonic-Scale Lyapunov Descent (Prop. 1) -/

/-- Exact KL step identity from Eq. (3):
if `log(w_i / z_i) = η (2 g_t - g_{t-1})_i - L_i - log Z` on a simplex (`∑ p_i = ∑ w_i = 1`),
then `D(p ‖ w) - D(p ‖ z) = η ⟨2 g_t - g_{t-1}, w - p⟩ - D(w ‖ z) - ⟨L, w - p⟩`. -/
theorem kl_step_exact_identity {ι : Type*} [Fintype ι]
    (p z w g_curr g_prev L : ι → ℝ) (η logZ : ℝ)
    (hp_sum : ∑ i : ι, p i = 1) (hw_sum : ∑ i : ι, w i = 1)
    (hstep : ∀ i, Real.log (w i / z i) = η * (2 * g_curr i - g_prev i) - L i - logZ) :
    (∑ i : ι, p i * Real.log (z i / w i)) =
      η * (∑ i : ι, (2 * g_curr i - g_prev i) * (w i - p i))
      - (∑ i : ι, w i * Real.log (w i / z i))
      - (∑ i : ι, L i * (w i - p i)) := by
  have hlog_flip : ∀ i, Real.log (z i / w i) = -Real.log (w i / z i) := fun i => by
    rw [← Real.log_inv, inv_div]
  have hlhs : ∑ i : ι, p i * Real.log (z i / w i) =
      -(∑ i : ι, p i * (η * (2 * g_curr i - g_prev i) - L i)) + logZ := by
    have : ∑ i : ι, p i * Real.log (z i / w i) =
        ∑ i : ι, (-(p i * (η * (2 * g_curr i - g_prev i) - L i)) + p i * logZ) := by
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [hlog_flip i, hstep i]; ring
    rw [this, Finset.sum_add_distrib, Finset.sum_neg_distrib, ← Finset.sum_mul, hp_sum, one_mul]
  have hdwz : ∑ i : ι, w i * Real.log (w i / z i) =
      (∑ i : ι, w i * (η * (2 * g_curr i - g_prev i) - L i)) - logZ := by
    have : ∑ i : ι, w i * Real.log (w i / z i) =
        ∑ i : ι, (w i * (η * (2 * g_curr i - g_prev i) - L i) - w i * logZ) := by
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [hstep i]; ring
    rw [this, Finset.sum_sub_distrib, ← Finset.sum_mul, hw_sum, one_mul]
  rw [hlhs, hdwz]
  have hr : η * (∑ i : ι, (2 * g_curr i - g_prev i) * (w i - p i)) -
      (∑ i : ι, L i * (w i - p i)) =
      (∑ i : ι, w i * (η * (2 * g_curr i - g_prev i) - L i)) -
      (∑ i : ι, p i * (η * (2 * g_curr i - g_prev i) - L i)) := by
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun i _ => by ring
  linarith

/-- **Optimistic energy coercivity (Eq. (5))**:
Under Pinsker's inequality `D_pz ≥ dist_zp² / 4`, Lipschitz payoff bound
`⟨g_{t-1}, z_t - p⟩ ≤ M * d_t * dist_zp` (since `⟨g_t, z_t - p⟩ ≤ 0` and
`‖g_t - g_{t-1}‖_∞ ≤ M d_t`), and `0 < η ≤ 1 / (64 M)`,
`E_t := D_pz - η ⟨g_{t-1}, z_t - p⟩ + d_t² / 16 ≥ dist_zp² / 8 + d_t² / 32 ≥ 0`. -/
theorem optimistic_energy_coercive {D_pz gprev_dot dist_zp d_t η M : ℝ}
    (hPinsker : dist_zp ^ 2 / 4 ≤ D_pz)
    (hlag : gprev_dot ≤ M * d_t * dist_zp)
    (hη0 : 0 < η) (hM : 0 < M) (hηM : η * M ≤ 1 / 64) :
    dist_zp ^ 2 / 8 + d_t ^ 2 / 32 ≤ D_pz - η * gprev_dot + d_t ^ 2 / 16 := by
  have hyoung : η * M * d_t * dist_zp ≤ dist_zp ^ 2 / 8 + 2 * (η * M) ^ 2 * d_t ^ 2 := by
    have hsq : 0 ≤ (dist_zp / (2 * Real.sqrt 2) - Real.sqrt 2 * (η * M) * d_t) ^ 2 := sq_nonneg _
    have hs2 : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
    have hsq' : 0 ≤ dist_zp ^ 2 / 8 - η * M * d_t * dist_zp + 2 * (η * M) ^ 2 * d_t ^ 2 := by
      have : (dist_zp / (2 * Real.sqrt 2) - Real.sqrt 2 * (η * M) * d_t) ^ 2 =
          dist_zp ^ 2 / (4 * Real.sqrt 2 ^ 2) - η * M * d_t * dist_zp +
            Real.sqrt 2 ^ 2 * (η * M) ^ 2 * d_t ^ 2 := by
        have hne : Real.sqrt 2 ≠ 0 := Real.sqrt_ne_zero'.mpr (by norm_num)
        field_simp [hne]; ring
      rw [hs2] at this
      linarith
    linarith
  have h2 : 2 * (η * M) ^ 2 ≤ 1 / 32 := by
    have hηM0 : 0 ≤ η * M := by positivity
    nlinarith
  have h3 : η * gprev_dot ≤ η * M * d_t * dist_zp := by
    calc η * gprev_dot ≤ η * (M * d_t * dist_zp) := mul_le_mul_of_nonneg_left hlag hη0.le
      _ = η * M * d_t * dist_zp := by ring
  nlinarith [sq_nonneg d_t]

/-- **Proposition 1 (Core one-step Lyapunov inequality Eq. (6) and harmonic-scale descent)**:
Combining the KL step identity, Pinsker `-D(w ‖ z) ≤ -d_{t+1}² / 4`, Young's bound on
`η ⟨g_t - g_{t-1}, w - z⟩ ≤ (η M / 2)(d_t² + d_{t+1}²)`, Nash slack `η ⟨g_t, z - p⟩ ≤ -(η γ / 2) m`,
and Lemma 2's anchor dissipation yields:
1. `E_{t+1} - E_t ≤ (c / α_t) (H_t - H_{t+1}) - R_t`,
2. For `V_t := H_t + (α_t / c) E_t` with `0 < α_{t+1} ≤ α_t` and `E_{t+1} ≥ 0`:
   `V_{t+1} ≤ V_t - (α_t / c) R_t`. -/
theorem lyapunov_one_step_and_harmonic_descent
    {E_t E_next H_t H_next D_pw D_pz D_wz gcurr_zp gdiff_wz anchor_term : ℝ}
    {d_t d_next anchor_diss m_out c α_t α_next η M γ : ℝ}
    (hE_diff : E_next - E_t = (D_pw - D_pz) - η * (gcurr_zp + gdiff_wz) +
        (d_next ^ 2 - d_t ^ 2) / 16 +
        (η * (gcurr_zp + gdiff_wz) - (D_pw - D_pz) +
          (η * gcurr_zp + η * gdiff_wz - D_wz + anchor_term)))
    (hPinsker_wz : d_next ^ 2 / 4 ≤ D_wz)
    (hLip : gdiff_wz ≤ M * d_t * d_next)
    (hSlack : gcurr_zp ≤ -(γ / 2) * m_out)
    (hAnchor : anchor_term ≤ (c / α_t) * (H_t - H_next) - anchor_diss + (η * γ / 4) * m_out)
    (hc : 0 < c) (hαt : 0 < α_t) (hα_mono : α_next ≤ α_t)
    (hη0 : 0 < η) (hM0 : 0 ≤ M) (hηM : η * M ≤ 1 / 64) (hEnext_nn : 0 ≤ E_next) :
    let R_t := (d_t ^ 2 + d_next ^ 2) / 32 + anchor_diss + (η * γ / 4) * m_out
    E_next - E_t ≤ (c / α_t) * (H_t - H_next) - R_t ∧
    (H_next + (α_next / c) * E_next) ≤ (H_t + (α_t / c) * E_t) - (α_t / c) * R_t := by
  dsimp only
  set R_t := (d_t ^ 2 + d_next ^ 2) / 32 + anchor_diss + (η * γ / 4) * m_out
  have hE_simp : E_next - E_t =
      η * gcurr_zp + η * gdiff_wz - D_wz + anchor_term + (d_next ^ 2 - d_t ^ 2) / 16 := by
    linarith
  have hyoung : gdiff_wz ≤ M * ((d_t ^ 2 + d_next ^ 2) / 2) := by
    have hsq : 0 ≤ (d_t - d_next) ^ 2 := sq_nonneg _
    nlinarith
  have hlag_bound : η * gdiff_wz ≤ (1 / 128) * (d_t ^ 2 + d_next ^ 2) := by
    have h1 : η * gdiff_wz ≤ (η * M / 2) * (d_t ^ 2 + d_next ^ 2) := by
      calc η * gdiff_wz ≤ η * (M * ((d_t ^ 2 + d_next ^ 2) / 2)) :=
            mul_le_mul_of_nonneg_left hyoung hη0.le
        _ = (η * M / 2) * (d_t ^ 2 + d_next ^ 2) := by ring
    nlinarith [sq_nonneg d_t, sq_nonneg d_next]
  have hstep_E : E_next - E_t ≤ (c / α_t) * (H_t - H_next) - R_t := by
    have h1 : η * gcurr_zp ≤ -(η * γ / 2) * m_out := by
      have := mul_le_mul_of_nonneg_left hSlack hη0.le
      linarith
    simp only [R_t]
    nlinarith [sq_nonneg d_t, sq_nonneg d_next]
  refine ⟨hstep_E, ?_⟩
  have hcoef : 0 < α_t / c := div_pos hαt hc
  have hmul := mul_le_mul_of_nonneg_left hstep_E hcoef.le
  have hcancel : (α_t / c) * ((c / α_t) * (H_t - H_next) - R_t) =
      (H_t - H_next) - (α_t / c) * R_t := by
    field_simp [hc.ne', hαt.ne']
  rw [hcancel] at hmul
  have hmono : (α_next / c) * E_next ≤ (α_t / c) * E_next := by
    apply mul_le_mul_of_nonneg_right _ hEnext_nn
    exact div_le_div_of_nonneg_right hα_mono hc.le
  linarith

/-- Telescoping partial-sum bound from harmonic Lyapunov descent:
if `V(t + 1) ≤ V(t) - d(t)` for all `t` and `V(t) ≥ 0`, then `∑_{t < T} d(t) ≤ V(0)`. -/
theorem telescoping_dissipation_bound (V d : ℕ → ℝ) (hV_nn : ∀ t, 0 ≤ V t)
    (hstep : ∀ t, V (t + 1) ≤ V t - d t) (T : ℕ) :
    ∑ t ∈ Finset.range T, d t ≤ V 0 := by
  have htele : ∑ t ∈ Finset.range T, d t ≤ V 0 - V T := by
    induction T with
    | zero => simp
    | succ n ih =>
      rw [Finset.sum_range_succ]
      linarith [hstep n]
  linarith [hV_nn T]

/-! ### 6. Asymptotic Positive-Exploitability Exclusion in Fully Mixed Games (Theorem 2 Core) -/

/-- **Theorem 2 (Weighted exponential drift lower bound)**:
Let `p > 0` be a fully mixed Nash equilibrium (`∑ p_i = 2`, `∑ p_i D_i = 0`), `β ∈ (0, 1)`,
`q = 1 - β`, `η > 0`. If some action `i₀` has `D_{i₀} ≥ e / 2 > 0`, then with
`p_min := min_i p_i > 0` and `b := p_min * (exp(η e / 2) - 1 - η e / 2) > 0`:
1. `0 < b`,
2. There exists `J ≥ 2` such that `2 q^J < β q b / 2`,
3. For every such `J`, the `J`-step geometric-exponential lower bound
   `W_lb(i) := β ∑_{k=0}^{J-1} q^k exp(k η D_i)` satisfies
   `∑_i p_i (W_lb(i) - 1) ≥ β q b - 2 q^J > β q b / 2 > 0`. -/
theorem fullyMixed_weighted_exp_drift {ι : Type*} [Fintype ι] [Nonempty ι]
    {β η e p_min : ℝ} (hβ0 : 0 < β) (hβ1 : β < 1) (hη : 0 < η) (he : 0 < e)
    (hp_min_pos : 0 < p_min)
    (p D : ι → ℝ) (hp_ge : ∀ i, p_min ≤ p i)
    (hp_sum : ∑ i : ι, p i = 2) (hD_mean : ∑ i : ι, p i * D i = 0)
    (i₀ : ι) (hDi₀ : e / 2 ≤ D i₀) :
    let q := 1 - β
    let b := p_min * (Real.exp (η * e / 2) - 1 - η * e / 2)
    0 < b ∧
    ∃ J : ℕ, 2 ≤ J ∧ 2 * q ^ J < β * q * b / 2 ∧
      let W_lb : ι → ℝ := fun i => β * ∑ k ∈ Finset.range J, q ^ k * Real.exp ((k : ℝ) * η * D i)
      β * q * b / 2 < ∑ i : ι, p i * (W_lb i - 1) := by
  dsimp only
  set q := 1 - β
  set b := p_min * (Real.exp (η * e / 2) - 1 - η * e / 2)
  have hq0 : 0 < q := by linarith
  have hq1 : q < 1 := by linarith
  have hηe_pos : 0 < η * e / 2 := by positivity
  have hexp_strict : 0 < Real.exp (η * e / 2) - 1 - η * e / 2 := by
    have := Real.add_one_lt_exp hηe_pos.ne'
    linarith
  have hb_pos : 0 < b := mul_pos hp_min_pos hexp_strict
  have htarget_pos : 0 < β * q * b / 2 := by positivity
  -- Find J ≥ 2 with 2 * q^J < β * q * b / 2 using tendsto_pow_atTop_nhds_zero_of_lt_one
  have hexJ : ∃ J : ℕ, 2 ≤ J ∧ 2 * q ^ J < β * q * b / 2 := by
    have htend := tendsto_pow_atTop_nhds_zero_of_lt_one hq0.le hq1
    have hev := Metric.tendsto_atTop.mp htend (β * q * b / 4) (by positivity)
    rcases hev with ⟨N, hN⟩
    refine ⟨max N 2, le_max_right _ _, ?_⟩
    have hdist := hN (max N 2) (le_max_left _ _)
    rw [Real.dist_eq, sub_zero, abs_of_nonneg (pow_nonneg hq0.le _)] at hdist
    linarith
  refine ⟨hb_pos, ?_⟩
  rcases hexJ with ⟨J, hJ2, hqJ⟩
  refine ⟨J, hJ2, hqJ, ?_⟩
  -- For every k : ℕ, ∑ p_i exp(k η D_i) ≥ 2
  have hk_ge : ∀ k : ℕ, 2 ≤ ∑ i : ι, p i * Real.exp ((k : ℝ) * η * D i) := by
    intro k
    have hpt : ∀ i : ι, p i * (1 + (k : ℝ) * η * D i) ≤ p i * Real.exp ((k : ℝ) * η * D i) := by
      intro i
      have hp_pos : 0 < p i := lt_of_lt_of_le hp_min_pos (hp_ge i)
      exact mul_le_mul_of_nonneg_left (by linarith [Real.add_one_le_exp ((k : ℝ) * η * D i)])
        hp_pos.le
    calc (2 : ℝ) = ∑ i : ι, p i + (k : ℝ) * η * ∑ i : ι, p i * D i := by
          rw [hp_sum, hD_mean]; ring
      _ = ∑ i : ι, p i * (1 + (k : ℝ) * η * D i) := by
          rw [Finset.mul_sum, ← Finset.sum_add_distrib]
          exact Finset.sum_congr rfl fun i _ => by ring
      _ ≤ ∑ i : ι, p i * Real.exp ((k : ℝ) * η * D i) :=
          Finset.sum_le_sum fun i _ => hpt i
  -- At k = 1, strict excess ≥ b
  have hk1_ge : 2 + b ≤ ∑ i : ι, p i * Real.exp ((1 : ℝ) * η * D i) := by
    have hrem : ∀ i : ι, 0 ≤ p i * (Real.exp (η * D i) - 1 - η * D i) := by
      intro i
      have hp_pos : 0 < p i := lt_of_lt_of_le hp_min_pos (hp_ge i)
      exact mul_nonneg hp_pos.le (by linarith [Real.add_one_le_exp (η * D i)])
    have hi0 : b ≤ p i₀ * (Real.exp (η * D i₀) - 1 - η * D i₀) := by
      -- t ↦ exp t - 1 - t is increasing on [0, ∞) since (exp t - 1 - t) at t₂ ≥ t₁ ≥ 0:
      -- exp t₂ - exp t₁ = exp t₁ (exp(t₂ - t₁) - 1) ≥ 1 * (t₂ - t₁)
      have ht1 : η * e / 2 ≤ η * D i₀ := by nlinarith
      have hdiff_nn : 0 ≤ η * D i₀ - η * e / 2 := by linarith
      have hexp_mono : Real.exp (η * e / 2) - 1 - η * e / 2 ≤
          Real.exp (η * D i₀) - 1 - η * D i₀ := by
        have h1 : 1 ≤ Real.exp (η * e / 2) := Real.one_le_exp hηe_pos.le
        have h2 : η * D i₀ - η * e / 2 ≤ Real.exp (η * D i₀ - η * e / 2) - 1 := by
          linarith [Real.add_one_le_exp (η * D i₀ - η * e / 2)]
        have hmul : Real.exp (η * e / 2) * (Real.exp (η * D i₀ - η * e / 2) - 1) =
            Real.exp (η * D i₀) - Real.exp (η * e / 2) := by
          rw [mul_sub, mul_one, ← Real.exp_add]; congr 2; ring
        nlinarith
      exact mul_le_mul (hp_ge i₀) hexp_mono hexp_strict.le
        (lt_of_lt_of_le hp_min_pos (hp_ge i₀)).le
    have hsum_rem : b ≤ ∑ i : ι, p i * (Real.exp (η * D i) - 1 - η * D i) :=
      le_trans hi0 (Finset.single_le_sum (fun i _ => hrem i) (Finset.mem_univ i₀))
    have hident : ∑ i : ι, p i * (Real.exp (η * D i) - 1 - η * D i) =
        (∑ i : ι, p i * Real.exp ((1 : ℝ) * η * D i)) - 2 := by
      have : ∑ i : ι, p i * (Real.exp (η * D i) - 1 - η * D i) =
          (∑ i : ι, p i * Real.exp (η * D i)) - (∑ i : ι, p i) - η * (∑ i : ι, p i * D i) := by
        rw [Finset.mul_sum, ← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib]
        exact Finset.sum_congr rfl fun i _ => by ring
      rw [this, hp_sum, hD_mean]
      simp [one_mul]
    linarith
  -- Sum over k ∈ range J using geometric series β ∑_{k=0}^{J-1} q^k = 1 - q^J
  have hgeom : β * ∑ k ∈ Finset.range J, q ^ k = 1 - q ^ J := by
    have h := mul_geom_sum q J
    have hq1_eq : q - 1 = -β := by simp only [q]; ring
    rw [hq1_eq] at h
    linarith
  have hswap : ∑ i : ι, p i * (β * ∑ k ∈ Finset.range J, q ^ k * Real.exp ((k : ℝ) * η * D i) - 1) =
      β * (∑ k ∈ Finset.range J, q ^ k * ∑ i : ι, p i * Real.exp ((k : ℝ) * η * D i)) - 2 := by
    calc ∑ i : ι, p i * (β * ∑ k ∈ Finset.range J, q ^ k * Real.exp ((k : ℝ) * η * D i) - 1)
        = (∑ i : ι, β * ∑ k ∈ Finset.range J, q ^ k * (p i * Real.exp ((k : ℝ) * η * D i))) -
            ∑ i : ι, p i := by
          rw [← Finset.sum_sub_distrib]
          refine Finset.sum_congr rfl fun i _ => ?_
          rw [mul_sub, mul_one, ← mul_assoc, mul_comm (p i) β, mul_assoc, Finset.mul_sum]
          congr 2; exact Finset.sum_congr rfl fun k _ => by ring
      _ = β * (∑ i : ι, ∑ k ∈ Finset.range J, q ^ k * (p i * Real.exp ((k : ℝ) * η * D i))) - 2 := by
          rw [← Finset.mul_sum, hp_sum]
      _ = β * (∑ k ∈ Finset.range J, q ^ k * ∑ i : ι, p i * Real.exp ((k : ℝ) * η * D i)) - 2 := by
          congr 2; rw [Finset.sum_comm]
          exact Finset.sum_congr rfl fun k _ => (Finset.mul_sum _ _ _).symm
  rw [hswap]
  have hsum_k : (∑ k ∈ Finset.range J, q ^ k * 2) + q * b ≤
      ∑ k ∈ Finset.range J, q ^ k * ∑ i : ι, p i * Real.exp ((k : ℝ) * η * D i) := by
    have h1mem : 1 ∈ Finset.range J := Finset.mem_range.mpr (by omega)
    rw [← Finset.add_sum_erase _ _ h1mem, ← Finset.add_sum_erase _ _ h1mem]
    have h1term : q ^ 1 * 2 + q * b ≤
        q ^ 1 * ∑ i : ι, p i * Real.exp ((1 : ℕ) * η * D i) := by
      have : q ^ 1 * 2 + q * b = q * (2 + b) := by ring
      rw [this, pow_one, Nat.cast_one]
      exact mul_le_mul_of_nonneg_left hk1_ge hq0.le
    have hrest : ∑ k ∈ (Finset.range J).erase 1, q ^ k * 2 ≤
        ∑ k ∈ (Finset.range J).erase 1, q ^ k * ∑ i : ι, p i * Real.exp ((k : ℝ) * η * D i) :=
      Finset.sum_le_sum fun k _ => mul_le_mul_of_nonneg_left (hk_ge k) (pow_nonneg hq0.le k)
    linarith
  have hgeom2 : β * ∑ k ∈ Finset.range J, q ^ k * 2 = 2 * (1 - q ^ J) := by
    rw [← Finset.sum_mul]; linarith
  have hmul := mul_le_mul_of_nonneg_left hsum_k hβ0.le
  linarith

/-- **Harmonic divergence contradiction (Theorem 2 closing step)**:
If `p_i > 0`, `a_{t,i} ∈ (0, 1]`, and `∑_i p_i log(a_{t+1,i} / a_{t,i}) ≥ δ / (t + 2)`
for all `t ≥ T₀` with `δ > 0`, then contradiction (since `∑_{t ≥ T₀} 1/(t + 2) = +∞`
while `∑_i p_i log a_{T,i} ≤ 0`). -/
theorem harmonic_drift_contradicts_bounded_anchor {ι : Type*} [Fintype ι]
    (p : ι → ℝ) (a : ℕ → ι → ℝ) (T₀ : ℕ) (δ : ℝ)
    (hδ : 0 < δ) (hp : ∀ i, 0 < p i)
    (ha_pos : ∀ t i, 0 < a t i) (ha_le : ∀ t i, a t i ≤ 1)
    (hdrift : ∀ t, T₀ ≤ t →
      δ / ((t : ℝ) + 2) ≤ ∑ i : ι, p i * Real.log (a (t + 1) i / a t i)) :
    False := by
  let L : ℕ → ℝ := fun t => ∑ i : ι, p i * Real.log (a t i)
  have hL_le : ∀ t, L t ≤ 0 := fun t =>
    Finset.sum_nonpos fun i _ =>
      mul_nonpos_of_nonneg_of_nonpos (hp i).le (Real.log_nonpos (ha_pos t i).le (ha_le t i))
  have hL_step : ∀ t, T₀ ≤ t → δ / ((t : ℝ) + 2) ≤ L (t + 1) - L t := by
    intro t ht
    have hdiff : L (t + 1) - L t = ∑ i : ι, p i * Real.log (a (t + 1) i / a t i) := by
      simp only [L, ← Finset.sum_sub_distrib]
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [Real.log_div (ha_pos (t + 1) i).ne' (ha_pos t i).ne']; ring
    rw [hdiff]
    exact hdrift t ht
  -- Bound partial sums of 1 / (k + 2) from below by log((T + 2)/(T₀ + 2))
  have htele : ∀ N : ℕ, δ * ∑ k ∈ Finset.range N, (1 : ℝ) / ((T₀ + k : ℕ) + 2) ≤
      L (T₀ + N) - L T₀ := by
    intro N
    induction N with
    | zero => simp
    | succ n ih =>
      rw [Finset.sum_range_succ, mul_add]
      have hs := hL_step (T₀ + n) (Nat.le_add_right T₀ n)
      have hcast : ((T₀ + n : ℕ) : ℝ) + 2 = (T₀ + n : ℝ) + 2 := by push_cast; ring
      have hterm : δ * (1 / (((T₀ + n : ℕ) : ℝ) + 2)) = δ / (((T₀ + n : ℕ) : ℝ) + 2) := by ring
      rw [hterm]
      have hadd : T₀ + (n + 1) = (T₀ + n) + 1 := by omega
      rw [hadd]
      linarith
  -- Standard integral comparison: log((k + 3)/(k + 2)) ≤ 1 / (k + 2)
  have hlog_step : ∀ k : ℕ, Real.log ((((k + 1 : ℕ) : ℝ) + 2) / ((k : ℝ) + 2)) ≤
      1 / ((k : ℝ) + 2) := by
    intro k
    have hk : (0 : ℝ) < (k : ℝ) + 2 := by positivity
    have hrat : (((k + 1 : ℕ) : ℝ) + 2) / ((k : ℝ) + 2) = 1 + 1 / ((k : ℝ) + 2) := by
      push_cast; field_simp [hk.ne']; ring
    rw [hrat]
    linarith [Real.log_le_sub_one_of_pos (show (0 : ℝ) < 1 + 1 / ((k : ℝ) + 2) by positivity)]
  have hharm : ∀ N : ℕ, Real.log ((((T₀ + N : ℕ) : ℝ) + 2) / ((T₀ : ℝ) + 2)) ≤
      ∑ k ∈ Finset.range N, (1 : ℝ) / (((T₀ + k : ℕ) : ℝ) + 2) := by
    intro N
    induction N with
    | zero => simp
    | succ n ih =>
      rw [Finset.sum_range_succ]
      have hs := hlog_step (T₀ + n)
      have hpos1 : (0 : ℝ) < ((T₀ + n : ℕ) : ℝ) + 2 := by positivity
      have hpos2 : (0 : ℝ) < (T₀ : ℝ) + 2 := by positivity
      have hpos3 : (0 : ℝ) < (((T₀ + n + 1 : ℕ) : ℝ) + 2) := by positivity
      have hcomb : Real.log ((((T₀ + (n + 1) : ℕ) : ℝ) + 2) / ((T₀ : ℝ) + 2)) =
          Real.log ((((T₀ + n : ℕ) : ℝ) + 2) / ((T₀ : ℝ) + 2)) +
          Real.log ((((T₀ + n + 1 : ℕ) : ℝ) + 2) / (((T₀ + n : ℕ) : ℝ) + 2)) := by
        rw [← Real.log_mul (div_pos hpos1 hpos2).ne' (div_pos hpos3 hpos1).ne']
        congr 1
        have hadd : T₀ + (n + 1) = T₀ + n + 1 := by omega
        rw [hadd]
        field_simp [hpos1.ne', hpos2.ne']
      rw [hcomb]
      linarith
  -- Choose N so large that δ * log((T₀ + N + 2)/(T₀ + 2)) > -L T₀
  let B : ℝ := (-L T₀) / δ + 1
  obtain ⟨N, hN⟩ := exists_nat_gt (((T₀ : ℝ) + 2) * Real.exp B)
  have hlog_gt : B < Real.log ((((T₀ + N : ℕ) : ℝ) + 2) / ((T₀ : ℝ) + 2)) := by
    have hpos : (0 : ℝ) < (T₀ : ℝ) + 2 := by positivity
    have hgt : Real.exp B < (((T₀ + N : ℕ) : ℝ) + 2) / ((T₀ : ℝ) + 2) := by
      rw [lt_div_iff₀ hpos]
      have : (N : ℝ) ≤ ((T₀ + N : ℕ) : ℝ) + 2 := by push_cast; linarith
      linarith
    rw [← Real.log_exp B]
    exact Real.log_lt_log (Real.exp_pos B) hgt
  have h1 := htele N
  have h2 := mul_le_mul_of_nonneg_left (hharm N) hδ.le
  have h3 := hL_le (T₀ + N)
  have h4 : -L T₀ < δ * B := by
    simp only [B]; field_simp [hδ.ne']; linarith
  have h5 : δ * B < δ * Real.log ((((T₀ + N : ℕ) : ℝ) + 2) / ((T₀ : ℝ) + 2)) :=
    mul_lt_mul_of_pos_left hlog_gt hδ
  linarith

end GARIPLastIterate
