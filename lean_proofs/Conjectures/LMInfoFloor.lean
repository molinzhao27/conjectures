import Mathlib

/-!
# Sharp Information Floor for a Softmax Readout
(Bruno Abrahao, *Neural Collapse Is Forbidden: Information Floors in Language Models*,
arXiv:2607.09487)

We formalize in Lean 4 + Mathlib (zero `sorry`, standard axioms only) the complete finite-support
proof of the appendix conjecture `R^2 * D_k ≥ 2 * I_k` (and the sharper row-diameter inequality
`L^2 * D_k ≥ 8 * I_k`), together with the explicit two-context sharpness witnesses for `m = 2`
and every `m > 2`:

1. **Popoviciu variance bound**:
   - `variance_eq_centered_sub`: For any finite probability distribution `p` and random variable `x`,
     `variance p x + (expectation p x - c)^2 = ∑ i, p i * (x i - c)^2`.
   - `popoviciu_variance_bound`: If `∀ i, a ≤ x i ∧ x i ≤ b`, then
     `variance p x ≤ (b - a)^2 / 4`.
2. **Softmax log-partition derivatives, Bregman remainder, and information floor**:
   - `expected_kl_decomposition`, `mutual_info_le_expected_kl_ref`: Marginal optimality
     `∑ i, π i * KL(q_i ‖ r) = I_k + KL(q̄ ‖ r) ≥ I_k`.
   - `logZ_line_hasDerivAt`, `logZ_line_hasDerivAt_second`: Along `h + t • v`, the first and
     second derivatives of the log-partition function `A(h + t • v)` are the softmax expectation
     and softmax variance of `δ_c · v`.
   - `klDiv_softmax_eq_bregman`, `klDiv_softmax_le_width_sq`: The Bregman remainder satisfies
     `KL(q_h ‖ q_μ) ≤ (b - a)^2 / 8` whenever `δ_c · (μ - h) ∈ [a, b]`.
   - `klDiv_softmax_le_diameter`, `info_floor_diameter`, `info_floor_radius`:
     For every finite context law `(π, H)` with mean `μ` and dispersion `D_k = ∑ π_i ‖H_i - μ‖²`,
     `8 * I_k ≤ L^2 * D_k` (row diameter `L`) and `2 * I_k ≤ R^2 * D_k` (row radius `R`).
3. **Explicit centered two-context sharpness witnesses**:
   - `sharpness_m2_exact`, `sharpness_m2_refutes_larger_const`: For `m = 2`, centered rows `±1`
     and equiprobable contexts `±t` give `D_k = t^2` and `I_k = t * tanh t - log(cosh t)`, which
     refutes any uniform constant `C > 2`.
   - `sharpness_general_m_exact`, `sharpness_general_m_refutes_larger_const`: For every `m > 2`
     and `0 < t < 1/2`, the explicit 2D centered rows `(±√(1 - t^2), t)` and `(0, -2t/(m - 2))`
     with contexts `(±t, t⁻²)` have max norm `≤ 1`, dispersion `D_k = t^2`, exact information
     `I_k = p_t * (a * t * tanh (a * t) - log (cosh (a * t)))`, and refute any constant `C > 2`.

Machine-checked vs prose-only:
- Machine-checked: All discrete-law inequalities above, exact derivatives of the finite-category
  log-partition function, the Bregman upper bound, `L^2 D_k ≥ 8 I_k`, `R^2 D_k ≥ 2 I_k`, and both
  `m = 2` and general `m > 2` sharpness families (including refutation of every `C > 2`).
- Prose-only (in `solutions/lm_info_floor/results.md`): Extension of the outer context expectation
  from arbitrary finite support to general square-integrable probability measures via dominated
  convergence / monotone approximation, and numerical differential-evolution checks.
-/

open BigOperators Finset

namespace LMInfoFloor

/-! ### 1. Finite Expectations, Variance, and Popoviciu's Inequality -/

/-- Expectation of a real-valued function on `Fin m` under weights `p`. -/
def expectation {m : ℕ} (p x : Fin m → ℝ) : ℝ :=
  ∑ c : Fin m, p c * x c

/-- Variance of a real-valued function on `Fin m` under weights `p`. -/
def variance {m : ℕ} (p x : Fin m → ℝ) : ℝ :=
  ∑ c : Fin m, p c * (x c - expectation p x) ^ 2

/-- Parallel-axis identity: mean squared deviation around `c` decomposes as variance plus bias². -/
theorem variance_add_sq_eq {m : ℕ} (p x : Fin m → ℝ) (hp_sum : ∑ i, p i = 1) (c : ℝ) :
    variance p x + (expectation p x - c) ^ 2 = ∑ i : Fin m, p i * (x i - c) ^ 2 := by
  unfold variance expectation
  set μ := ∑ i : Fin m, p i * x i
  have hcross : ∑ i : Fin m, p i * (x i - μ) = 0 := by
    calc ∑ i : Fin m, p i * (x i - μ)
        = (∑ i, p i * x i) - (∑ i, p i) * μ := by
          simp_rw [mul_sub, sum_sub_distrib, sum_mul]
      _ = μ - 1 * μ := by rw [hp_sum]
      _ = 0 := by ring
  calc (∑ i, p i * (x i - μ) ^ 2) + (μ - c) ^ 2
      = (∑ i, p i * (x i - μ) ^ 2) + 2 * (μ - c) * (∑ i, p i * (x i - μ)) +
          (∑ i, p i) * (μ - c) ^ 2 := by rw [hcross, hp_sum]; ring
    _ = ∑ i, (p i * (x i - μ) ^ 2 + 2 * (μ - c) * (p i * (x i - μ)) + p i * (μ - c) ^ 2) := by
          simp_rw [sum_add_distrib, ← mul_sum, ← sum_mul]
    _ = ∑ i, p i * (x i - c) ^ 2 := by
          refine Finset.sum_congr rfl fun i _ => ?_
          ring

/-- Variance is bounded by the mean squared deviation around any center `c`. -/
theorem variance_le_centered_sq {m : ℕ} (p x : Fin m → ℝ) (hp_sum : ∑ i, p i = 1) (c : ℝ) :
    variance p x ≤ ∑ i : Fin m, p i * (x i - c) ^ 2 := by
  rw [← variance_add_sq_eq p x hp_sum c]
  linarith [sq_nonneg (expectation p x - c)]

/-- **Popoviciu's variance inequality**: any random variable taking values in `[a, b]` has variance
at most `(b - a)^2 / 4`. -/
theorem popoviciu_variance_bound {m : ℕ} (p x : Fin m → ℝ)
    (hp_nonneg : ∀ i, 0 ≤ p i) (hp_sum : ∑ i, p i = 1)
    {a b : ℝ} (hbounds : ∀ i, a ≤ x i ∧ x i ≤ b) :
    variance p x ≤ (b - a) ^ 2 / 4 := by
  let mid : ℝ := (a + b) / 2
  have hsq : ∀ i, (x i - mid) ^ 2 ≤ (b - a) ^ 2 / 4 := by
    intro i
    have h1 : -(b - a) / 2 ≤ x i - mid := by dsimp only [mid]; linarith [(hbounds i).1]
    have h2 : x i - mid ≤ (b - a) / 2 := by dsimp only [mid]; linarith [(hbounds i).2]
    have h3 : (x i - mid) ^ 2 ≤ ((b - a) / 2) ^ 2 := by nlinarith
    linarith
  calc variance p x
      ≤ ∑ i, p i * (x i - mid) ^ 2 := variance_le_centered_sq p x hp_sum mid
    _ ≤ ∑ i, p i * ((b - a) ^ 2 / 4) := by
          refine Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (hsq i) (hp_nonneg i)
    _ = (∑ i, p i) * ((b - a) ^ 2 / 4) := by rw [← sum_mul]
    _ = (b - a) ^ 2 / 4 := by rw [hp_sum, one_mul]

/-! ### 2. Euclidean Helpers, Softmax, and KL Divergence -/

/-- Euclidean dot product on `Fin d → ℝ`. -/
def dot {d : ℕ} (u v : Fin d → ℝ) : ℝ :=
  ∑ i : Fin d, u i * v i

/-- Squared Euclidean norm on `Fin d → ℝ`. -/
def normSq {d : ℕ} (v : Fin d → ℝ) : ℝ :=
  dot v v

lemma normSq_nonneg {d : ℕ} (v : Fin d → ℝ) : 0 ≤ normSq v := by
  unfold normSq dot
  exact Finset.sum_nonneg fun i _ => mul_self_nonneg (v i)

lemma dot_sub_left {d : ℕ} (u₁ u₂ v : Fin d → ℝ) :
    dot (fun i => u₁ i - u₂ i) v = dot u₁ v - dot u₂ v := by
  unfold dot
  simp_rw [sub_mul, sum_sub_distrib]

lemma dot_neg_right {d : ℕ} (u v : Fin d → ℝ) :
    dot u (fun i => -v i) = -dot u v := by
  unfold dot
  simp_rw [mul_neg, sum_neg_distrib]

/-- Cauchy–Schwarz inequality in squared form: `(u · v)^2 ≤ ‖u‖² ‖v‖²`. -/
theorem cauchy_schwarz_sq {d : ℕ} (u v : Fin d → ℝ) :
    (dot u v) ^ 2 ≤ normSq u * normSq v := by
  unfold dot normSq
  have h := Finset.sum_mul_sq_le_sq_mul_sq (Finset.univ : Finset (Fin d)) u v
  simpa [sq] using h

/-- Triangle inequality in squared form: `‖u - v‖² ≤ (2 R)²` when `‖u‖² ≤ R²` and `‖v‖² ≤ R²`. -/
theorem normSq_sub_le_four_sq {d : ℕ} (u v : Fin d → ℝ) {R : ℝ}
    (hu : normSq u ≤ R ^ 2) (hv : normSq v ≤ R ^ 2) :
    normSq (fun i => u i - v i) ≤ (2 * R) ^ 2 := by
  have hexp : normSq (fun i => u i - v i) = normSq u + normSq v - 2 * dot u v := by
    unfold normSq dot
    calc ∑ i, (u i - v i) * (u i - v i)
        = ∑ i, (u i * u i + v i * v i - 2 * (u i * v i)) := by
            refine Finset.sum_congr rfl fun i _ => ?_; ring
      _ = (∑ i, u i * u i) + (∑ i, v i * v i) - 2 * (∑ i, u i * v i) := by
            simp_rw [sum_sub_distrib, sum_add_distrib, ← mul_sum]
  have hcs : (dot u v) ^ 2 ≤ R ^ 4 := by
    calc (dot u v) ^ 2 ≤ normSq u * normSq v := cauchy_schwarz_sq u v
      _ ≤ R ^ 2 * R ^ 2 := mul_le_mul hu hv (normSq_nonneg v) (sq_nonneg R)
      _ = R ^ 4 := by ring
  have hneg : -dot u v ≤ R ^ 2 := by nlinarith [sq_nonneg R]
  rw [hexp]
  linarith

/-- Partition function `Z(h) = ∑_c exp(δ_c · h)`. -/
noncomputable def partZ {m d : ℕ} (δ : Fin m → Fin d → ℝ) (h : Fin d → ℝ) : ℝ :=
  ∑ c : Fin m, Real.exp (dot (δ c) h)

/-- Log-partition function `A(h) = log ∑_c exp(δ_c · h)`. -/
noncomputable def logZ {m d : ℕ} (δ : Fin m → Fin d → ℝ) (h : Fin d → ℝ) : ℝ :=
  Real.log (partZ δ h)

/-- Softmax distribution `q_h(c) = exp(δ_c · h) / ∑_j exp(δ_j · h)`. -/
noncomputable def softmax {m d : ℕ} (δ : Fin m → Fin d → ℝ) (h : Fin d → ℝ) (c : Fin m) : ℝ :=
  Real.exp (dot (δ c) h) / partZ δ h

lemma partZ_pos {m d : ℕ} [NeZero m] (δ : Fin m → Fin d → ℝ) (h : Fin d → ℝ) :
    0 < partZ δ h := by
  unfold partZ
  exact Finset.sum_pos (fun c _ => Real.exp_pos _) Finset.univ_nonempty

lemma softmax_pos {m d : ℕ} [NeZero m] (δ : Fin m → Fin d → ℝ) (h : Fin d → ℝ) (c : Fin m) :
    0 < softmax δ h c :=
  div_pos (Real.exp_pos _) (partZ_pos δ h)

lemma softmax_nonneg {m d : ℕ} [NeZero m] (δ : Fin m → Fin d → ℝ) (h : Fin d → ℝ) (c : Fin m) :
    0 ≤ softmax δ h c :=
  le_of_lt (softmax_pos δ h c)

lemma softmax_sum_one {m d : ℕ} [NeZero m] (δ : Fin m → Fin d → ℝ) (h : Fin d → ℝ) :
    ∑ c : Fin m, softmax δ h c = 1 := by
  unfold softmax
  rw [← Finset.sum_div]
  exact div_self (ne_of_gt (partZ_pos δ h))

/-- Kullback–Leibler divergence on `Fin m` in nats. -/
noncomputable def klDiv {m : ℕ} (q r : Fin m → ℝ) : ℝ :=
  ∑ c : Fin m, q c * Real.log (q c / r c)

/-- Nonnegativity of KL divergence for strictly positive probability vectors. -/
theorem klDiv_nonneg {m : ℕ} (q r : Fin m → ℝ)
    (hq_pos : ∀ c, 0 < q c) (hr_pos : ∀ c, 0 < r c)
    (hq_sum : ∑ c, q c = 1) (hr_sum : ∑ c, r c = 1) :
    0 ≤ klDiv q r := by
  unfold klDiv
  have hterm : ∀ c, q c - r c ≤ q c * Real.log (q c / r c) := by
    intro c
    have hlog : Real.log (r c / q c) ≤ r c / q c - 1 :=
      Real.log_le_sub_one_of_pos (div_pos (hr_pos c) (hq_pos c))
    have hinv : Real.log (q c / r c) = -Real.log (r c / q c) := by
      rw [← Real.log_inv, inv_div]
    rw [hinv]
    have hmul := mul_le_mul_of_nonneg_left hlog (le_of_lt (hq_pos c))
    have hqne : q c ≠ 0 := ne_of_gt (hq_pos c)
    have hsimp : q c * (r c / q c - 1) = r c - q c := by field_simp
    linarith
  calc (0 : ℝ) = (∑ c, q c) - (∑ c, r c) := by rw [hq_sum, hr_sum]; ring
    _ = ∑ c, (q c - r c) := by rw [← sum_sub_distrib]
    _ ≤ ∑ c, q c * Real.log (q c / r c) := Finset.sum_le_sum fun c _ => hterm c

/-- Marginal optimality identity: for any finite context law `π`, conditionals `q_i`, marginal
`q̄ c = ∑ i, π i * q i c`, and positive reference `r`,
`∑ i, π i * KL(q_i ‖ r) = (∑ i, π i * KL(q_i ‖ q̄)) + KL(q̄ ‖ r)`. -/
theorem expected_kl_decomposition {N m : ℕ} (π : Fin N → ℝ)
    (q : Fin N → Fin m → ℝ) (r : Fin m → ℝ)
    (hπ_nonneg : ∀ i, 0 ≤ π i) (hπ_sum : ∑ i, π i = 1)
    (hq_pos : ∀ i c, 0 < q i c) (hr_pos : ∀ c, 0 < r c) :
    let qbar : Fin m → ℝ := fun c => ∑ i, π i * q i c
    ∑ i, π i * klDiv (q i) r = (∑ i, π i * klDiv (q i) qbar) + klDiv qbar r := by
  intro qbar
  have hqbar_pos : ∀ c, 0 < qbar c := by
    intro c
    have hne : ∃ i, 0 < π i := by
      by_contra h
      push_neg at h
      have h0 : ∑ i, π i ≤ 0 := Finset.sum_nonpos fun i _ => h i
      linarith
    rcases hne with ⟨i₀, hi₀⟩
    exact Finset.sum_pos' (fun i _ => mul_nonneg (hπ_nonneg i) (le_of_lt (hq_pos i c)))
      ⟨i₀, Finset.mem_univ i₀, mul_pos hi₀ (hq_pos i₀ c)⟩
  unfold klDiv
  have hsplit : ∀ i c,
      Real.log (q i c / r c) = Real.log (q i c / qbar c) + Real.log (qbar c / r c) := by
    intro i c
    rw [← Real.log_mul (ne_of_gt (div_pos (hq_pos i c) (hqbar_pos c)))
      (ne_of_gt (div_pos (hqbar_pos c) (hr_pos c)))]
    congr 1
    field_simp [ne_of_gt (hqbar_pos c), ne_of_gt (hr_pos c)]
  calc ∑ i, π i * ∑ c, q i c * Real.log (q i c / r c)
      = ∑ i, π i * ∑ c, (q i c * Real.log (q i c / qbar c) +
          q i c * Real.log (qbar c / r c)) := by
          refine Finset.sum_congr rfl fun i _ => congrArg (π i * ·) ?_
          refine Finset.sum_congr rfl fun c _ => by rw [hsplit i c, mul_add]
    _ = (∑ i, π i * ∑ c, q i c * Real.log (q i c / qbar c)) +
          ∑ i, ∑ c, (π i * q i c) * Real.log (qbar c / r c) := by
          simp_rw [sum_add_distrib, mul_add, sum_add_distrib, mul_sum, mul_assoc]
    _ = (∑ i, π i * ∑ c, q i c * Real.log (q i c / qbar c)) +
          ∑ c, qbar c * Real.log (qbar c / r c) := by
          congr 1
          rw [Finset.sum_comm]
          refine Finset.sum_congr rfl fun c _ => by rw [← Finset.sum_mul]

/-- The expected KL to any fixed reference distribution `r` upper-bounds the mutual information
`I_k = ∑ i, π i * KL(q_i ‖ q̄)`. -/
theorem mutual_info_le_expected_kl_ref {N m : ℕ} (π : Fin N → ℝ)
    (q : Fin N → Fin m → ℝ) (r : Fin m → ℝ)
    (hπ_nonneg : ∀ i, 0 ≤ π i) (hπ_sum : ∑ i, π i = 1)
    (hq_pos : ∀ i c, 0 < q i c) (hq_sum : ∀ i, ∑ c, q i c = 1)
    (hr_pos : ∀ c, 0 < r c) (hr_sum : ∑ c, r c = 1) :
    let qbar : Fin m → ℝ := fun c => ∑ i, π i * q i c
    ∑ i, π i * klDiv (q i) qbar ≤ ∑ i, π i * klDiv (q i) r := by
  intro qbar
  have hdecomp := expected_kl_decomposition π q r hπ_nonneg hπ_sum hq_pos hr_pos
  have hqbar_pos : ∀ c, 0 < qbar c := by
    intro c
    have hne : ∃ i, 0 < π i := by
      by_contra h; push_neg at h
      linarith [Finset.sum_nonpos (s := Finset.univ) fun i _ => h i]
    rcases hne with ⟨i₀, hi₀⟩
    exact Finset.sum_pos' (fun i _ => mul_nonneg (hπ_nonneg i) (le_of_lt (hq_pos i c)))
      ⟨i₀, Finset.mem_univ i₀, mul_pos hi₀ (hq_pos i₀ c)⟩
  have hqbar_sum : ∑ c, qbar c = 1 := by
    calc ∑ c, ∑ i, π i * q i c
        = ∑ i, ∑ c, π i * q i c := Finset.sum_comm
      _ = ∑ i, π i * ∑ c, q i c := by simp_rw [← mul_sum]
      _ = ∑ i, π i * 1 := by simp_rw [hq_sum]
      _ = 1 := by simp [hπ_sum]
  have hkl : 0 ≤ klDiv qbar r := klDiv_nonneg qbar r hqbar_pos hr_pos hqbar_sum hr_sum
  linarith

/-! ### 3. Log-Partition Derivatives and Bregman Remainder Inequality -/

/-- Directional line `h + t • v`. -/
def linePt {d : ℕ} (h v : Fin d → ℝ) (t : ℝ) : Fin d → ℝ :=
  fun i => h i + t * v i

lemma dot_linePt {d : ℕ} (u h v : Fin d → ℝ) (t : ℝ) :
    dot u (linePt h v t) = dot u h + t * dot u v := by
  unfold dot linePt
  calc ∑ i, u i * (h i + t * v i)
      = ∑ i, (u i * h i + t * (u i * v i)) := by
          refine Finset.sum_congr rfl fun i _ => ?_; ring
    _ = (∑ i, u i * h i) + t * (∑ i, u i * v i) := by
          rw [sum_add_distrib, ← mul_sum]

/-- First derivative of `t ↦ A(h + t v)` is the softmax expectation of `δ_c · v`. -/
theorem logZ_line_hasDerivAt {m d : ℕ} [NeZero m]
    (δ : Fin m → Fin d → ℝ) (h v : Fin d → ℝ) (t : ℝ) :
    HasDerivAt (fun s => logZ δ (linePt h v s))
      (expectation (softmax δ (linePt h v t)) (fun c => dot (δ c) v)) t := by
  have hexp_deriv : ∀ c : Fin m,
      HasDerivAt (fun s => Real.exp (dot (δ c) (linePt h v s)))
        (Real.exp (dot (δ c) (linePt h v t)) * dot (δ c) v) t := by
    intro c
    have hlin : HasDerivAt (fun s => dot (δ c) h + s * dot (δ c) v) (dot (δ c) v) t := by
      simpa using (hasDerivAt_id t).mul_const (dot (δ c) v) |>.const_add (dot (δ c) h)
    have hcomp := (Real.hasDerivAt_exp (dot (δ c) h + t * dot (δ c) v)).comp t hlin
    simpa [dot_linePt] using hcomp
  have hZ_deriv : HasDerivAt (fun s => partZ δ (linePt h v s))
      (∑ c : Fin m, Real.exp (dot (δ c) (linePt h v t)) * dot (δ c) v) t := by
    simpa [partZ] using HasDerivAt.fun_sum fun c (_ : c ∈ Finset.univ) => hexp_deriv c
  have hlog := hZ_deriv.log (ne_of_gt (partZ_pos δ (linePt h v t)))
  have heq : (∑ c : Fin m, Real.exp (dot (δ c) (linePt h v t)) * dot (δ c) v) /
      partZ δ (linePt h v t) =
      expectation (softmax δ (linePt h v t)) (fun c => dot (δ c) v) := by
    unfold expectation softmax
    rw [Finset.sum_div]
    refine Finset.sum_congr rfl fun c _ => by ring
  rw [heq] at hlog
  exact hlog

/-- Second derivative of `t ↦ A(h + t v)` is the softmax variance of `δ_c · v`. -/
theorem logZ_line_hasDerivAt_second {m d : ℕ} [NeZero m]
    (δ : Fin m → Fin d → ℝ) (h v : Fin d → ℝ) (t : ℝ) :
    HasDerivAt (fun s => expectation (softmax δ (linePt h v s)) (fun c => dot (δ c) v))
      (variance (softmax δ (linePt h v t)) (fun c => dot (δ c) v)) t := by
  let a : Fin m → ℝ := fun c => dot (δ c) v
  let E : Fin m → ℝ → ℝ := fun c s => Real.exp (dot (δ c) (linePt h v s))
  let N : ℝ → ℝ := fun s => ∑ c, E c s * a c
  let Z : ℝ → ℝ := fun s => partZ δ (linePt h v s)
  have hE : ∀ c, HasDerivAt (E c) (E c t * a c) t := by
    intro c
    have hlin : HasDerivAt (fun s => dot (δ c) h + s * a c) (a c) t := by
      simpa [a] using (hasDerivAt_id t).mul_const (a c) |>.const_add (dot (δ c) h)
    have hcomp := (Real.hasDerivAt_exp (dot (δ c) h + t * a c)).comp t hlin
    simpa [E, dot_linePt, a] using hcomp
  have hN : HasDerivAt N (∑ c, E c t * (a c) ^ 2) t := by
    have hsum : HasDerivAt (fun s => ∑ c : Fin m, E c s * a c)
        (∑ c : Fin m, (E c t * a c) * a c) t :=
      HasDerivAt.fun_sum fun c (_ : c ∈ Finset.univ) => (hE c).mul_const (a c)
    convert hsum using 1
    refine Finset.sum_congr rfl fun c _ => by ring
  have hZ : HasDerivAt Z (N t) t := by
    simpa [Z, N, partZ, E] using HasDerivAt.fun_sum fun c (_ : c ∈ Finset.univ) => hE c
  have hZpos : Z t ≠ 0 := ne_of_gt (partZ_pos δ (linePt h v t))
  have hdiv := hN.fun_div hZ hZpos
  have hfun_eq : (fun s => N s / Z s) =
      (fun s => expectation (softmax δ (linePt h v s)) a) := by
    funext s
    unfold expectation softmax
    simp only [N, Z, E]
    rw [Finset.sum_div]
    refine Finset.sum_congr rfl fun c _ => by ring
  rw [hfun_eq] at hdiv
  convert hdiv using 1
  have hsum1 : ∑ c : Fin m, softmax δ (linePt h v t) c = 1 :=
    softmax_sum_one δ (linePt h v t)
  have hvar := variance_add_sq_eq (softmax δ (linePt h v t)) a hsum1 0
  have hμ : expectation (softmax δ (linePt h v t)) a = N t / Z t :=
    (congrFun hfun_eq t).symm
  have hsec : ∑ c : Fin m, softmax δ (linePt h v t) c * (a c) ^ 2 =
      (∑ c, E c t * (a c) ^ 2) / Z t := by
    unfold softmax
    rw [Finset.sum_div]
    refine Finset.sum_congr rfl fun c _ => by simp only [E, Z]; ring
  have hvar' : variance (softmax δ (linePt h v t)) a =
      (∑ c, E c t * (a c) ^ 2) / Z t - (N t / Z t) ^ 2 := by
    have : ∑ c : Fin m, softmax δ (linePt h v t) c * (a c - 0) ^ 2 =
        ∑ c : Fin m, softmax δ (linePt h v t) c * (a c) ^ 2 := by simp
    rw [this, hsec, hμ] at hvar
    linarith
  rw [hvar']
  field_simp [hZpos]

/-- Exact Bregman identity: `KL(q_h ‖ q_μ) = A(μ) - A(h) - E_{q_h}[δ · (μ - h)]`. -/
theorem klDiv_softmax_eq_bregman {m d : ℕ} [NeZero m]
    (δ : Fin m → Fin d → ℝ) (h μ : Fin d → ℝ) :
    let v : Fin d → ℝ := fun i => μ i - h i
    klDiv (softmax δ h) (softmax δ μ) =
      logZ δ μ - logZ δ h - expectation (softmax δ h) (fun c => dot (δ c) v) := by
  intro v
  have hZh : 0 < partZ δ h := partZ_pos δ h
  have hZμ : 0 < partZ δ μ := partZ_pos δ μ
  have hsum1 : ∑ c : Fin m, softmax δ h c = 1 := softmax_sum_one δ h
  have hlog_ratio : ∀ c, Real.log (softmax δ h c / softmax δ μ c) =
      logZ δ μ - logZ δ h - dot (δ c) v := by
    intro c
    unfold softmax logZ
    have hratio : (Real.exp (dot (δ c) h) / partZ δ h) / (Real.exp (dot (δ c) μ) / partZ δ μ) =
        Real.exp (dot (δ c) h - dot (δ c) μ) * (partZ δ μ / partZ δ h) := by
      rw [Real.exp_sub]
      field_simp [ne_of_gt hZh, ne_of_gt hZμ]
    rw [hratio, Real.log_mul (ne_of_gt (Real.exp_pos _)) (ne_of_gt (div_pos hZμ hZh)),
      Real.log_exp, Real.log_div (ne_of_gt hZμ) (ne_of_gt hZh)]
    have hdot : dot (δ c) v = dot (δ c) μ - dot (δ c) h := by
      unfold dot v
      simp_rw [mul_sub, sum_sub_distrib]
    linarith
  unfold klDiv expectation
  calc ∑ c, softmax δ h c * Real.log (softmax δ h c / softmax δ μ c)
      = ∑ c, softmax δ h c * (logZ δ μ - logZ δ h - dot (δ c) v) := by
          refine Finset.sum_congr rfl fun c _ => by rw [hlog_ratio c]
    _ = (∑ c, softmax δ h c) * (logZ δ μ - logZ δ h) -
          ∑ c, softmax δ h c * dot (δ c) v := by
          simp_rw [mul_sub, sum_sub_distrib, ← sum_mul]
    _ = logZ δ μ - logZ δ h - ∑ c, softmax δ h c * dot (δ c) v := by rw [hsum1, one_mul]

/-- **Bregman remainder bound via Popoviciu**: if `δ_c · (μ - h) ∈ [a, b]` for all `c`, then
`KL(q_h ‖ q_μ) ≤ (b - a)^2 / 8`. -/
theorem klDiv_softmax_le_width_sq {m d : ℕ} [NeZero m]
    (δ : Fin m → Fin d → ℝ) (h μ : Fin d → ℝ) {a b : ℝ}
    (hbounds : ∀ c, a ≤ dot (δ c) (fun i => μ i - h i) ∧
      dot (δ c) (fun i => μ i - h i) ≤ b) :
    klDiv (softmax δ h) (softmax δ μ) ≤ (b - a) ^ 2 / 8 := by
  let v : Fin d → ℝ := fun i => μ i - h i
  let f : ℝ → ℝ := fun t => logZ δ (linePt h v t)
  let f' : ℝ → ℝ := fun t => expectation (softmax δ (linePt h v t)) (fun c => dot (δ c) v)
  let M : ℝ := (b - a) ^ 2 / 4
  let g' : ℝ → ℝ := fun t => f' t - f' 0 - M * t
  let g : ℝ → ℝ := fun t => f t - f 0 - f' 0 * t - (M / 2) * t ^ 2
  have hg'_deriv : ∀ t, HasDerivAt g' (variance (softmax δ (linePt h v t)) (fun c => dot (δ c) v) - M) t := by
    intro t
    have h1 := logZ_line_hasDerivAt_second δ h v t
    have h2 : HasDerivAt (fun s => M * s) M t := by
      simpa using (hasDerivAt_id t).const_mul M
    simpa [g', f'] using (h1.sub_const (f' 0)).sub h2
  have hg'_antitone : Antitone g' := by
    refine antitone_of_deriv_nonpos (fun t => (hg'_deriv t).differentiableAt) fun t => ?_
    rw [(hg'_deriv t).deriv]
    have hpop := popoviciu_variance_bound (softmax δ (linePt h v t)) (fun c => dot (δ c) v)
      (softmax_nonneg δ (linePt h v t)) (softmax_sum_one δ (linePt h v t)) hbounds
    dsimp only [M]
    linarith
  have hg'_nonpos : ∀ t ∈ Set.Icc (0 : ℝ) 1, g' t ≤ 0 := by
    intro t ht
    have hle := hg'_antitone ht.1
    have hg'0 : g' 0 = 0 := by simp [g']
    linarith
  have hg_deriv : ∀ t, HasDerivAt g (g' t) t := by
    intro t
    have h1 := logZ_line_hasDerivAt δ h v t
    have h2 : HasDerivAt (fun s => f' 0 * s) (f' 0) t := by
      simpa using (hasDerivAt_id t).const_mul (f' 0)
    have h3 : HasDerivAt (fun s => (M / 2) * s ^ 2) (M * t) t := by
      have hsq : HasDerivAt (fun s : ℝ => s ^ 2) (2 * t) t := by
        simpa using hasDerivAt_pow 2 t
      convert hsq.const_mul (M / 2) using 1
      ring
    simpa [g, g', f, f'] using ((h1.sub_const (f 0)).sub h2).sub h3
  have hg_antitone : AntitoneOn g (Set.Icc (0 : ℝ) 1) := by
    refine antitoneOn_of_deriv_nonpos (convex_Icc 0 1)
      (fun t _ => (hg_deriv t).continuousAt.continuousWithinAt)
      (fun t _ => (hg_deriv t).differentiableAt.differentiableWithinAt) fun t ht => ?_
    rw [(hg_deriv t).deriv]
    exact hg'_nonpos t (interior_subset ht)
  have hg1_le : g 1 ≤ g 0 := hg_antitone ⟨le_refl 0, zero_le_one⟩ ⟨zero_le_one, le_refl 1⟩ zero_le_one
  have hg0 : g 0 = 0 := by simp [g]
  have hf0 : f 0 = logZ δ h := by
    have hpt : linePt h v 0 = h := by funext i; simp [linePt]
    simp [f, hpt]
  have hf1 : f 1 = logZ δ μ := by
    have hpt : linePt h v 1 = μ := by funext i; simp [linePt, v]
    simp [f, hpt]
  have hf'0 : f' 0 = expectation (softmax δ h) (fun c => dot (δ c) v) := by
    have hpt : linePt h v 0 = h := by funext i; simp [linePt]
    simp [f', hpt]
  rw [klDiv_softmax_eq_bregman δ h μ]
  have : f 1 - f 0 - f' 0 ≤ M / 2 := by
    have h1 : g 1 = f 1 - f 0 - f' 0 - M / 2 := by simp [g]
    linarith
  rw [hf1, hf0, hf'0] at this
  have hM2 : M / 2 = (b - a) ^ 2 / 8 := by simp [M]; ring
  linarith

/-- Pointwise diameter bound: `KL(q_h ‖ q_μ) ≤ (L^2 / 8) * ‖h - μ‖²`. -/
theorem klDiv_softmax_le_diameter {m d : ℕ} [NeZero m]
    (δ : Fin m → Fin d → ℝ) (h μ : Fin d → ℝ) {L : ℝ} (hL_nonneg : 0 ≤ L)
    (hdiam : ∀ c₁ c₂, normSq (fun i => δ c₁ i - δ c₂ i) ≤ L ^ 2) :
    klDiv (softmax δ h) (softmax δ μ) ≤ (L ^ 2 / 8) * normSq (fun i => h i - μ i) := by
  let v : Fin d → ℝ := fun i => μ i - h i
  let s : Fin m → ℝ := fun c => dot (δ c) v
  obtain ⟨cmin, _, hcmin⟩ := Finset.exists_min_image Finset.univ s Finset.univ_nonempty
  obtain ⟨cmax, _, hcmax⟩ := Finset.exists_max_image Finset.univ s Finset.univ_nonempty
  have hbounds : ∀ c, s cmin ≤ s c ∧ s c ≤ s cmax :=
    fun c => ⟨hcmin c (Finset.mem_univ c), hcmax c (Finset.mem_univ c)⟩
  have hkl := klDiv_softmax_le_width_sq δ h μ hbounds
  have hdiff_sq : (s cmax - s cmin) ^ 2 ≤ L ^ 2 * normSq v := by
    have hdot : s cmax - s cmin = dot (fun i => δ cmax i - δ cmin i) v := by
      simp [s, dot_sub_left]
    rw [hdot]
    calc (dot (fun i => δ cmax i - δ cmin i) v) ^ 2
        ≤ normSq (fun i => δ cmax i - δ cmin i) * normSq v := cauchy_schwarz_sq _ v
      _ ≤ L ^ 2 * normSq v := mul_le_mul_of_nonneg_right (hdiam cmax cmin) (normSq_nonneg v)
  have hnorm_sym : normSq v = normSq (fun i => h i - μ i) := by
    unfold normSq dot v
    refine Finset.sum_congr rfl fun i _ => by ring
  rw [hnorm_sym] at hdiff_sq
  have _ := hL_nonneg
  linarith

/-- **Sharper information floor (row diameter `L`)**: for every finite context law `(π, H)` with
mean `μ`, dispersion `D_k = ∑_i π_i ‖H_i - μ‖²`, and mutual information `I_k`,
`8 * I_k ≤ L^2 * D_k`. -/
theorem info_floor_diameter {N m d : ℕ} [NeZero m]
    (δ : Fin m → Fin d → ℝ) (π : Fin N → ℝ) (H : Fin N → Fin d → ℝ)
    (hπ_nonneg : ∀ i, 0 ≤ π i) (hπ_sum : ∑ i, π i = 1)
    {L : ℝ} (hL_nonneg : 0 ≤ L)
    (hdiam : ∀ c₁ c₂, normSq (fun j => δ c₁ j - δ c₂ j) ≤ L ^ 2) :
    let μ : Fin d → ℝ := fun j => ∑ i, π i * H i j
    let q : Fin N → Fin m → ℝ := fun i => softmax δ (H i)
    let qbar : Fin m → ℝ := fun c => ∑ i, π i * q i c
    let I_k : ℝ := ∑ i, π i * klDiv (q i) qbar
    let D_k : ℝ := ∑ i, π i * normSq (fun j => H i j - μ j)
    8 * I_k ≤ L ^ 2 * D_k := by
  intro μ q qbar I_k D_k
  have h1 : I_k ≤ ∑ i, π i * klDiv (q i) (softmax δ μ) :=
    mutual_info_le_expected_kl_ref π q (softmax δ μ) hπ_nonneg hπ_sum
      (fun i c => softmax_pos δ (H i) c) (fun i => softmax_sum_one δ (H i))
      (softmax_pos δ μ) (softmax_sum_one δ μ)
  have h2 : ∑ i, π i * klDiv (q i) (softmax δ μ) ≤
      ∑ i, π i * ((L ^ 2 / 8) * normSq (fun j => H i j - μ j)) := by
    refine Finset.sum_le_sum fun i _ =>
      mul_le_mul_of_nonneg_left (klDiv_softmax_le_diameter δ (H i) μ hL_nonneg hdiam) (hπ_nonneg i)
  have h3 : ∑ i, π i * ((L ^ 2 / 8) * normSq (fun j => H i j - μ j)) = (L ^ 2 / 8) * D_k := by
    dsimp only [D_k]
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun i _ => by ring
  linarith

/-- **Main appendix conjecture (`R^2 * D_k ≥ 2 * I_k`)**: if every residual row satisfies
`‖δ_c‖² ≤ R^2` (with `0 ≤ R`), then `2 * I_k ≤ R^2 * D_k`. -/
theorem info_floor_radius {N m d : ℕ} [NeZero m]
    (δ : Fin m → Fin d → ℝ) (π : Fin N → ℝ) (H : Fin N → Fin d → ℝ)
    (hπ_nonneg : ∀ i, 0 ≤ π i) (hπ_sum : ∑ i, π i = 1)
    {R : ℝ} (hR_nonneg : 0 ≤ R)
    (hradius : ∀ c, normSq (δ c) ≤ R ^ 2) :
    let μ : Fin d → ℝ := fun j => ∑ i, π i * H i j
    let q : Fin N → Fin m → ℝ := fun i => softmax δ (H i)
    let qbar : Fin m → ℝ := fun c => ∑ i, π i * q i c
    let I_k : ℝ := ∑ i, π i * klDiv (q i) qbar
    let D_k : ℝ := ∑ i, π i * normSq (fun j => H i j - μ j)
    2 * I_k ≤ R ^ 2 * D_k := by
  have hdiam : ∀ c₁ c₂, normSq (fun j => δ c₁ j - δ c₂ j) ≤ (2 * R) ^ 2 :=
    fun c₁ c₂ => normSq_sub_le_four_sq (δ c₁) (δ c₂) (hradius c₁) (hradius c₂)
  have h := info_floor_diameter δ π H hπ_nonneg hπ_sum (by linarith) hdiam
  dsimp only at h ⊢
  linarith

/-! ### 4. Explicit Centered Two-Context Sharpness Witnesses (`m = 2` and `m > 2`) -/

/-- The binary sharpness information function `g(x) = x tanh x - log cosh x`. -/
noncomputable def binaryInfo (x : ℝ) : ℝ :=
  x * Real.tanh x - Real.log (Real.cosh x)

lemma tanh_le_self_of_nonneg {x : ℝ} (hx : 0 ≤ x) : Real.tanh x ≤ x := by
  let f : ℝ → ℝ := fun t => t - Real.tanh t
  have hf : ∀ t, HasDerivAt f (1 - (1 / (Real.cosh t) ^ 2)) t := by
    intro t
    have hcosh_ne : Real.cosh t ≠ 0 := ne_of_gt (Real.cosh_pos t)
    have htanh : HasDerivAt Real.tanh (1 / (Real.cosh t) ^ 2) t := by
      have hdiv := (Real.hasDerivAt_sinh t).div (Real.hasDerivAt_cosh t) hcosh_ne
      have heq : Real.tanh = fun s => Real.sinh s / Real.cosh s := by
        funext s; exact Real.tanh_eq_sinh_div_cosh s
      rw [heq]
      convert hdiv using 1
      have hsq := Real.cosh_sq_sub_sinh_sq t
      field_simp [hcosh_ne]
      linarith
    exact (hasDerivAt_id t).sub htanh
  have hmono : Monotone f := by
    refine monotone_of_deriv_nonneg (fun t => (hf t).differentiableAt) fun t => ?_
    rw [(hf t).deriv]
    have h1 : 1 ≤ Real.cosh t := Real.one_le_cosh t
    have h2 : 1 ≤ (Real.cosh t) ^ 2 := by nlinarith
    have h3 : 1 / (Real.cosh t) ^ 2 ≤ 1 := (div_le_one (by positivity)).2 h2
    linarith
  have h0 : f 0 ≤ f x := hmono hx
  simp only [f, Real.tanh_zero, sub_zero] at h0
  linarith

lemma binaryInfo_hasDerivAt (x : ℝ) :
    HasDerivAt binaryInfo (x / (Real.cosh x) ^ 2) x := by
  have hcosh_pos : 0 < Real.cosh x := Real.cosh_pos x
  have hcosh_ne : Real.cosh x ≠ 0 := ne_of_gt hcosh_pos
  have htanh : HasDerivAt Real.tanh (1 / (Real.cosh x) ^ 2) x := by
    have hdiv := (Real.hasDerivAt_sinh x).div (Real.hasDerivAt_cosh x) hcosh_ne
    have heq : Real.tanh = fun s => Real.sinh s / Real.cosh s := by
      funext s; exact Real.tanh_eq_sinh_div_cosh s
    rw [heq]
    convert hdiv using 1
    have hsq := Real.cosh_sq_sub_sinh_sq x
    field_simp [hcosh_ne]
    linarith
  have hprod : HasDerivAt (fun s => s * Real.tanh s)
      (1 * Real.tanh x + x * (1 / (Real.cosh x) ^ 2)) x :=
    (hasDerivAt_id x).mul htanh
  have hlog : HasDerivAt (fun s => Real.log (Real.cosh s))
      (Real.sinh x / Real.cosh x) x :=
    (Real.hasDerivAt_cosh x).log hcosh_ne
  convert hprod.sub hlog using 1
  rw [Real.tanh_eq_sinh_div_cosh]
  ring

/-- Quadratic lower bound `g(x) ≥ x^2 / 2 - x^4 / 4` for all `x ≥ 0`. -/
theorem binaryInfo_lower_bound {x : ℝ} (hx : 0 ≤ x) :
    x ^ 2 / 2 - x ^ 4 / 4 ≤ binaryInfo x := by
  let ψ : ℝ → ℝ := fun t => binaryInfo t - (t ^ 2 / 2 - t ^ 4 / 4)
  have hψ : ∀ t, HasDerivAt ψ (t / (Real.cosh t) ^ 2 - (t - t ^ 3)) t := by
    intro t
    have hpoly : HasDerivAt (fun s : ℝ => s ^ 2 / 2 - s ^ 4 / 4) (t - t ^ 3) t := by
      have h2 : HasDerivAt (fun s : ℝ => s ^ 2 / 2) t t := by
        convert (hasDerivAt_pow 2 t).div_const 2 using 1; ring
      have h4 : HasDerivAt (fun s : ℝ => s ^ 4 / 4) (t ^ 3) t := by
        convert (hasDerivAt_pow 4 t).div_const 4 using 1; ring
      exact h2.sub h4
    exact (binaryInfo_hasDerivAt t).sub hpoly
  have hmono : MonotoneOn ψ (Set.Ici 0) := by
    refine monotoneOn_of_deriv_nonneg (convex_Ici 0)
      (fun t _ => (hψ t).continuousAt.continuousWithinAt)
      (fun t _ => (hψ t).differentiableAt.differentiableWithinAt) fun t ht => ?_
    have ht0 : 0 ≤ t := le_of_lt (by simpa [interior_Ici] using ht)
    rw [(hψ t).deriv]
    have hcosh_ne : Real.cosh t ≠ 0 := ne_of_gt (Real.cosh_pos t)
    have hsech : 1 / (Real.cosh t) ^ 2 = 1 - (Real.tanh t) ^ 2 := by
      rw [Real.tanh_eq_sinh_div_cosh]
      have hsq := Real.cosh_sq_sub_sinh_sq t
      field_simp [hcosh_ne]
      linarith
    have htanh_nonneg : 0 ≤ Real.tanh t := by
      rw [Real.tanh_eq_sinh_div_cosh]
      exact div_nonneg (Real.sinh_nonneg_iff.2 ht0) (le_of_lt (Real.cosh_pos t))
    have htanh_le : Real.tanh t ≤ t := tanh_le_self_of_nonneg ht0
    have hsq_le : (Real.tanh t) ^ 2 ≤ t ^ 2 := by nlinarith
    have : t * (1 - (Real.tanh t) ^ 2) ≥ t * (1 - t ^ 2) := by nlinarith
    have : t / (Real.cosh t) ^ 2 = t * (1 - (Real.tanh t) ^ 2) := by
      rw [← hsech]; ring
    linarith
  have h0 : ψ 0 ≤ ψ x := hmono (Set.left_mem_Ici) hx hx
  have hψ0 : ψ 0 = 0 := by
    simp [ψ, binaryInfo, Real.tanh_zero, Real.cosh_zero]
  rw [hψ0] at h0
  dsimp only [ψ] at h0
  linarith

/-- Explicit `m = 2` sharpness witness: rows `+1, -1` and equiprobable contexts `+t, -t` are
centered, have radius `1`, dispersion `D_k = t^2`, and mutual information `I_k = binaryInfo t`. -/
theorem sharpness_m2_exact (t : ℝ) :
    let δ : Fin 2 → Fin 1 → ℝ := ![fun _ => 1, fun _ => -1]
    let π : Fin 2 → ℝ := ![1 / 2, 1 / 2]
    let H : Fin 2 → Fin 1 → ℝ := ![fun _ => t, fun _ => -t]
    let μ : Fin 1 → ℝ := fun j => ∑ i, π i * H i j
    let q : Fin 2 → Fin 2 → ℝ := fun i => softmax δ (H i)
    let qbar : Fin 2 → ℝ := fun c => ∑ i, π i * q i c
    let I_k : ℝ := ∑ i, π i * klDiv (q i) qbar
    let D_k : ℝ := ∑ i, π i * normSq (fun j => H i j - μ j)
    (∀ j, ∑ c, δ c j = 0) ∧
    (∀ c, normSq (δ c) = 1) ∧
    D_k = t ^ 2 ∧
    I_k = binaryInfo t := by
  intro δ π H μ q qbar I_k D_k
  have hcent : ∀ j, ∑ c, δ c j = 0 := by
    intro j; simp [Fin.sum_univ_two, δ]
  have hnorm : ∀ c, normSq (δ c) = 1 := by
    intro c; fin_cases c <;> simp [normSq, dot, δ]
  have hμ : μ = fun _ => 0 := by
    funext j; simp [μ, Fin.sum_univ_two, π, H]
  have hD : D_k = t ^ 2 := by
    simp [D_k, hμ, Fin.sum_univ_two, π, H, normSq, dot]; ring
  have hZ0 : partZ δ (H 0) = 2 * Real.cosh t := by
    simp [partZ, Fin.sum_univ_two, δ, H, dot, Real.cosh_eq]; ring
  have hZ1 : partZ δ (H 1) = 2 * Real.cosh t := by
    simp [partZ, Fin.sum_univ_two, δ, H, dot, Real.cosh_eq]; ring
  have hcosh_pos : 0 < Real.cosh t := Real.cosh_pos t
  have hcosh_ne : Real.cosh t ≠ 0 := ne_of_gt hcosh_pos
  have hq00 : q 0 0 = Real.exp t / (2 * Real.cosh t) := by
    simp only [q, softmax, hZ0]; simp [δ, H, dot]
  have hq01 : q 0 1 = Real.exp (-t) / (2 * Real.cosh t) := by
    simp only [q, softmax, hZ0]; simp [δ, H, dot]
  have hq10 : q 1 0 = Real.exp (-t) / (2 * Real.cosh t) := by
    simp only [q, softmax, hZ1]; simp [δ, H, dot]
  have hq11 : q 1 1 = Real.exp t / (2 * Real.cosh t) := by
    simp only [q, softmax, hZ1]; simp [δ, H, dot]
  have hqbar0 : qbar 0 = 1 / 2 := by
    simp only [qbar, Fin.sum_univ_two, π, hq00, hq10, Matrix.cons_val_zero, Matrix.cons_val_one]
    rw [Real.cosh_eq]; field_simp [hcosh_ne]
  have hqbar1 : qbar 1 = 1 / 2 := by
    simp only [qbar, Fin.sum_univ_two, π, hq01, hq11, Matrix.cons_val_zero, Matrix.cons_val_one]
    rw [Real.cosh_eq]; field_simp [hcosh_ne]; ring
  have hkl0 : klDiv (q 0) qbar = binaryInfo t := by
    simp only [klDiv, Fin.sum_univ_two, hq00, hq01, hqbar0, hqbar1]
    have hr1 : Real.exp t / (2 * Real.cosh t) / (1 / 2) = Real.exp t / Real.cosh t := by
      field_simp [hcosh_ne]
    have hr2 : Real.exp (-t) / (2 * Real.cosh t) / (1 / 2) = Real.exp (-t) / Real.cosh t := by
      field_simp [hcosh_ne]
    rw [hr1, hr2, Real.log_div (ne_of_gt (Real.exp_pos t)) hcosh_ne,
      Real.log_div (ne_of_gt (Real.exp_pos (-t))) hcosh_ne, Real.log_exp, Real.log_exp]
    unfold binaryInfo
    rw [Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq]
    have hsum_ne : Real.exp t + Real.exp (-t) ≠ 0 := by positivity
    field_simp [hsum_ne]
    ring
  have hkl1 : klDiv (q 1) qbar = binaryInfo t := by
    simp only [klDiv, Fin.sum_univ_two, hq10, hq11, hqbar0, hqbar1]
    have hr1 : Real.exp (-t) / (2 * Real.cosh t) / (1 / 2) = Real.exp (-t) / Real.cosh t := by
      field_simp [hcosh_ne]
    have hr2 : Real.exp t / (2 * Real.cosh t) / (1 / 2) = Real.exp t / Real.cosh t := by
      field_simp [hcosh_ne]
    rw [hr1, hr2, Real.log_div (ne_of_gt (Real.exp_pos (-t))) hcosh_ne,
      Real.log_div (ne_of_gt (Real.exp_pos t)) hcosh_ne, Real.log_exp, Real.log_exp]
    unfold binaryInfo
    rw [Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq]
    have hsum_ne : Real.exp t + Real.exp (-t) ≠ 0 := by positivity
    field_simp [hsum_ne]
    ring
  have hI : I_k = binaryInfo t := by
    simp only [I_k, Fin.sum_univ_two, π, Matrix.cons_val_zero, Matrix.cons_val_one, hkl0, hkl1]
    ring
  exact ⟨hcent, hnorm, hD, hI⟩

/-- Sharpness for `m = 2`: any uniform constant `C > 2` in `R^2 D_k ≥ C I_k` fails for small `t`. -/
theorem sharpness_m2_refutes_larger_const {C : ℝ} (hC : 2 < C) :
    ∃ t : ℝ, 0 < t ∧ t ^ 2 < C * binaryInfo t := by
  let ε : ℝ := (C - 2) / C
  have hε_pos : 0 < ε := div_pos (by linarith) (by linarith)
  obtain ⟨t, ht_pos, ht_sq⟩ : ∃ t : ℝ, 0 < t ∧ t ^ 2 < ε :=
    ⟨Real.sqrt (ε / 2), Real.sqrt_pos.2 (by linarith), by
      rw [Real.sq_sqrt (by linarith)]; linarith⟩
  refine ⟨t, ht_pos, ?_⟩
  have hlb := binaryInfo_lower_bound (le_of_lt ht_pos)
  have ht2_pos : 0 < t ^ 2 := sq_pos_of_pos ht_pos
  have h1 : t ^ 2 / 2 - t ^ 4 / 4 = (t ^ 2 / 2) * (1 - t ^ 2 / 2) := by ring
  have h2 : (1 : ℝ) - t ^ 2 / 2 > 1 - ε := by linarith
  have h3 : (1 : ℝ) - ε = 2 / C := by dsimp only [ε]; field_simp [show C ≠ 0 by linarith]; ring
  have h4 : (2 / C) * (t ^ 2 / 2) < binaryInfo t := by nlinarith
  have h5 : C * ((2 / C) * (t ^ 2 / 2)) = t ^ 2 := by
    field_simp [show C ≠ 0 by linarith]
  nlinarith

/-- Explicit 2D centered head for arbitrary `m > 2` and `0 < t < 1/2`. -/
noncomputable def sharpHead (m : ℕ) (t : ℝ) (c : Fin m) : Fin 2 → ℝ :=
  if c.val = 0 then ![Real.sqrt (1 - t ^ 2), t]
  else if c.val = 1 then ![-Real.sqrt (1 - t ^ 2), t]
  else ![0, -2 * t / ((m : ℝ) - 2)]

/-- First-pair probability mass `p_t` in the general `m > 2` sharpness construction. -/
noncomputable def pairMass (m : ℕ) (t : ℝ) : ℝ :=
  let a := Real.sqrt (1 - t ^ 2)
  (2 * Real.exp (1 / t) * Real.cosh (a * t)) /
    (2 * Real.exp (1 / t) * Real.cosh (a * t) +
      ((m : ℝ) - 2) * Real.exp (-2 / (((m : ℝ) - 2) * t)))

/-- **General `m > 2` exact sharpness construction**: the 2D head `sharpHead m t` with contexts
`(±t, t⁻²)` is centered, has max row norm `≤ 1`, dispersion `D_k = t^2`, and exact mutual
information `I_k = pairMass m t * binaryInfo (√(1 - t^2) * t)`. -/
theorem sharpness_general_m_exact {m : ℕ} (hm : 2 < m) {t : ℝ} (ht0 : 0 < t) (ht1 : t < 1 / 2) :
    haveI : NeZero m := ⟨by omega⟩
    let δ := sharpHead m t
    let π : Fin 2 → ℝ := ![1 / 2, 1 / 2]
    let H : Fin 2 → Fin 2 → ℝ := ![![t, t⁻¹ ^ 2], ![-t, t⁻¹ ^ 2]]
    let μ : Fin 2 → ℝ := fun j => ∑ i, π i * H i j
    let q : Fin 2 → Fin m → ℝ := fun i => softmax δ (H i)
    let qbar : Fin m → ℝ := fun c => ∑ i, π i * q i c
    let I_k : ℝ := ∑ i, π i * klDiv (q i) qbar
    let D_k : ℝ := ∑ i, π i * normSq (fun j => H i j - μ j)
    (∀ j, ∑ c, δ c j = 0) ∧
    (∀ c, normSq (δ c) ≤ 1) ∧
    D_k = t ^ 2 ∧
    I_k = pairMass m t * binaryInfo (Real.sqrt (1 - t ^ 2) * t) := by
  haveI : NeZero m := ⟨by omega⟩
  intro δ π H μ q qbar I_k D_k
  let c0 : Fin m := ⟨0, by omega⟩
  let c1 : Fin m := ⟨1, by omega⟩
  have hc01 : c0 ≠ c1 := by
    intro h; have := congrArg Fin.val h; dsimp [c0, c1] at this; omega
  let rest : Finset (Fin m) := Finset.univ.erase c0 |>.erase c1
  have hcard_rest : rest.card = m - 2 := by
    have hmem : c1 ∈ Finset.univ.erase c0 := Finset.mem_erase.2 ⟨hc01.symm, Finset.mem_univ c1⟩
    rw [Finset.card_erase_of_mem hmem, Finset.card_erase_of_mem (Finset.mem_univ c0),
      Finset.card_fin]
    omega
  have hsum_split : ∀ f : Fin m → ℝ, ∑ c : Fin m, f c = f c0 + f c1 + ∑ c ∈ rest, f c := by
    intro f
    have hmem : c1 ∈ Finset.univ.erase c0 := Finset.mem_erase.2 ⟨hc01.symm, Finset.mem_univ c1⟩
    rw [← Finset.add_sum_erase _ _ (Finset.mem_univ c0),
      ← Finset.add_sum_erase _ _ hmem]
    ring
  have hrest_val : ∀ c ∈ rest, c.val ≠ 0 ∧ c.val ≠ 1 := by
    intro c hc
    have h1 := Finset.mem_erase.1 hc
    have h2 := Finset.mem_erase.1 h1.2
    exact ⟨fun hv => h2.1 (Fin.ext hv), fun hv => h1.1 (Fin.ext hv)⟩
  have hm2_pos : (0 : ℝ) < (m : ℝ) - 2 := by
    have : (2 : ℝ) < (m : ℝ) := by exact_mod_cast hm
    linarith
  have hm2_cast : ((m - 2 : ℕ) : ℝ) = (m : ℝ) - 2 := Nat.cast_sub (by omega)
  have hcent : ∀ j, ∑ c, δ c j = 0 := by
    intro j
    rw [hsum_split]
    fin_cases j <;> dsimp only
    · have h0 : δ c0 0 = Real.sqrt (1 - t ^ 2) := by simp [δ, sharpHead, c0]
      have h1 : δ c1 0 = -Real.sqrt (1 - t ^ 2) := by simp [δ, sharpHead, c1]
      have hr : ∑ c ∈ rest, δ c 0 = 0 := Finset.sum_eq_zero fun c hc => by
        rcases hrest_val c hc with ⟨hn0, hn1⟩
        simp [δ, sharpHead, hn0, hn1]
      change δ c0 0 + δ c1 0 + ∑ c ∈ rest, δ c 0 = 0
      rw [h0, h1, hr]; ring
    · have h0 : δ c0 1 = t := by simp [δ, sharpHead, c0]
      have h1 : δ c1 1 = t := by simp [δ, sharpHead, c1]
      have hr : ∑ c ∈ rest, δ c 1 = ((m : ℝ) - 2) * (-2 * t / ((m : ℝ) - 2)) := by
        calc ∑ c ∈ rest, δ c 1
            = ∑ _c ∈ rest, (-2 * t / ((m : ℝ) - 2)) := by
                refine Finset.sum_congr rfl fun c hc => ?_
                rcases hrest_val c hc with ⟨hn0, hn1⟩
                simp [δ, sharpHead, hn0, hn1]
          _ = (rest.card : ℝ) * (-2 * t / ((m : ℝ) - 2)) := by simp
          _ = ((m : ℝ) - 2) * (-2 * t / ((m : ℝ) - 2)) := by rw [hcard_rest, hm2_cast]
      change δ c0 1 + δ c1 1 + ∑ c ∈ rest, δ c 1 = 0
      rw [h0, h1, hr]
      field_simp [ne_of_gt hm2_pos]
      ring
  have hnorm : ∀ c, normSq (δ c) ≤ 1 := by
    intro c
    have h1t : 0 ≤ 1 - t ^ 2 := by nlinarith
    by_cases h0 : c.val = 0
    · simp only [normSq, dot, Fin.sum_univ_two, δ, sharpHead, h0, ↓reduceIte,
        Matrix.cons_val_zero, Matrix.cons_val_one]
      rw [Real.mul_self_sqrt h1t]; linarith
    · by_cases h1 : c.val = 1
      · simp only [normSq, dot, Fin.sum_univ_two, δ, sharpHead, h1, Nat.one_ne_zero, ↓reduceIte,
          Matrix.cons_val_zero, Matrix.cons_val_one, neg_mul_neg]
        rw [Real.mul_self_sqrt h1t]; linarith
      · simp only [normSq, dot, Fin.sum_univ_two, δ, sharpHead, h0, h1, ↓reduceIte,
          Matrix.cons_val_zero, Matrix.cons_val_one, mul_zero, zero_add]
        have hm2_ge1 : (1 : ℝ) ≤ (m : ℝ) - 2 := by
          have : (3 : ℝ) ≤ (m : ℝ) := by exact_mod_cast (by omega : 3 ≤ m)
          linarith
        have habs : 2 * t / ((m : ℝ) - 2) ≤ 1 := by
          rw [div_le_one hm2_pos]; linarith
        have hnonneg : 0 ≤ 2 * t / ((m : ℝ) - 2) := div_nonneg (by linarith) (le_of_lt hm2_pos)
        have : (-2 * t / ((m : ℝ) - 2)) * (-2 * t / ((m : ℝ) - 2)) =
            (2 * t / ((m : ℝ) - 2)) ^ 2 := by ring
        rw [this]
        nlinarith
  have hμ : μ = ![0, t⁻¹ ^ 2] := by
    funext j
    fin_cases j
    · simp [μ, Fin.sum_univ_two, π, H]
    · simp [μ, Fin.sum_univ_two, π, H]; ring
  have hD : D_k = t ^ 2 := by
    simp [D_k, hμ, Fin.sum_univ_two, π, H, normSq, dot]; ring
  -- Compute logits and partition functions at both contexts
  let a : ℝ := Real.sqrt (1 - t ^ 2)
  have ht_ne : t ≠ 0 := ne_of_gt ht0
  have hdot00 : dot (δ c0) (H 0) = a * t + 1 / t := by
    simp only [dot, Fin.sum_univ_two, δ, sharpHead, c0, H, ↓reduceIte,
      Matrix.cons_val_zero, Matrix.cons_val_one, a]
    field_simp [ht_ne]
  have hdot10 : dot (δ c1) (H 0) = -(a * t) + 1 / t := by
    simp only [dot, Fin.sum_univ_two, δ, sharpHead, c1, H, Nat.one_ne_zero, ↓reduceIte,
      Matrix.cons_val_zero, Matrix.cons_val_one, a]
    field_simp [ht_ne]
  have hdot01 : dot (δ c0) (H 1) = -(a * t) + 1 / t := by
    simp only [dot, Fin.sum_univ_two, δ, sharpHead, c0, H, ↓reduceIte,
      Matrix.cons_val_zero, Matrix.cons_val_one, a]
    field_simp [ht_ne]
  have hdot11 : dot (δ c1) (H 1) = a * t + 1 / t := by
    simp only [dot, Fin.sum_univ_two, δ, sharpHead, c1, H, Nat.one_ne_zero, ↓reduceIte,
      Matrix.cons_val_zero, Matrix.cons_val_one, a]
    field_simp [ht_ne]
  have hdot_rest : ∀ c ∈ rest, ∀ i : Fin 2,
      dot (δ c) (H i) = -2 / (((m : ℝ) - 2) * t) := by
    intro c hc i
    rcases hrest_val c hc with ⟨hn0, hn1⟩
    fin_cases i
    · change dot (δ c) (H 0) = -2 / (((m : ℝ) - 2) * t)
      simp only [dot, Fin.sum_univ_two, δ, sharpHead, hn0, hn1, H, ↓reduceIte,
        Matrix.cons_val_zero, Matrix.cons_val_one, zero_mul, zero_add]
      field_simp [ht_ne, ne_of_gt hm2_pos]
    · change dot (δ c) (H 1) = -2 / (((m : ℝ) - 2) * t)
      simp only [dot, Fin.sum_univ_two, δ, sharpHead, hn0, hn1, H, ↓reduceIte,
        Matrix.cons_val_zero, Matrix.cons_val_one, zero_mul, zero_add]
      field_simp [ht_ne, ne_of_gt hm2_pos]
  let Zval : ℝ := 2 * Real.exp (1 / t) * Real.cosh (a * t) +
    ((m : ℝ) - 2) * Real.exp (-2 / (((m : ℝ) - 2) * t))
  have hZval_pos : 0 < Zval := by
    dsimp only [Zval]
    have := Real.exp_pos (1 / t)
    have := Real.cosh_pos (a * t)
    have := Real.exp_pos (-2 / (((m : ℝ) - 2) * t))
    nlinarith
  have hZ0 : partZ δ (H 0) = Zval := by
    unfold partZ
    rw [hsum_split, hdot00, hdot10]
    have hrest_sum : ∑ c ∈ rest, Real.exp (dot (δ c) (H 0)) =
        ((m : ℝ) - 2) * Real.exp (-2 / (((m : ℝ) - 2) * t)) := by
      calc ∑ c ∈ rest, Real.exp (dot (δ c) (H 0))
          = ∑ _c ∈ rest, Real.exp (-2 / (((m : ℝ) - 2) * t)) :=
              Finset.sum_congr rfl fun c hc => by rw [hdot_rest c hc 0]
        _ = (rest.card : ℝ) * Real.exp (-2 / (((m : ℝ) - 2) * t)) := by simp
        _ = ((m : ℝ) - 2) * Real.exp (-2 / (((m : ℝ) - 2) * t)) := by rw [hcard_rest, hm2_cast]
    dsimp only [Zval]
    rw [hrest_sum, Real.exp_add, Real.exp_add, Real.cosh_eq]
    ring
  have hZ1 : partZ δ (H 1) = Zval := by
    unfold partZ
    rw [hsum_split, hdot01, hdot11]
    have hrest_sum : ∑ c ∈ rest, Real.exp (dot (δ c) (H 1)) =
        ((m : ℝ) - 2) * Real.exp (-2 / (((m : ℝ) - 2) * t)) := by
      calc ∑ c ∈ rest, Real.exp (dot (δ c) (H 1))
          = ∑ _c ∈ rest, Real.exp (-2 / (((m : ℝ) - 2) * t)) :=
              Finset.sum_congr rfl fun c hc => by rw [hdot_rest c hc 1]
        _ = (rest.card : ℝ) * Real.exp (-2 / (((m : ℝ) - 2) * t)) := by simp
        _ = ((m : ℝ) - 2) * Real.exp (-2 / (((m : ℝ) - 2) * t)) := by rw [hcard_rest, hm2_cast]
    dsimp only [Zval]
    rw [hrest_sum, Real.exp_add, Real.exp_add, Real.cosh_eq]
    ring
  -- Evaluate q and qbar
  have hq_rest_eq : ∀ c ∈ rest, q 0 c = q 1 c := by
    intro c hc
    simp only [q, softmax, hZ0, hZ1, hdot_rest c hc 0, hdot_rest c hc 1]
  have hqbar_rest : ∀ c ∈ rest, qbar c = q 0 c := by
    intro c hc
    simp only [qbar, Fin.sum_univ_two, π, Matrix.cons_val_zero, Matrix.cons_val_one,
      ← hq_rest_eq c hc]
    ring
  have hq00 : q 0 c0 = Real.exp (a * t) * Real.exp (1 / t) / Zval := by
    simp only [q, softmax, hZ0, hdot00, Real.exp_add]
  have hq01 : q 0 c1 = Real.exp (-(a * t)) * Real.exp (1 / t) / Zval := by
    simp only [q, softmax, hZ0, hdot10, Real.exp_add]
  have hq10 : q 1 c0 = Real.exp (-(a * t)) * Real.exp (1 / t) / Zval := by
    simp only [q, softmax, hZ1, hdot01, Real.exp_add]
  have hq11 : q 1 c1 = Real.exp (a * t) * Real.exp (1 / t) / Zval := by
    simp only [q, softmax, hZ1, hdot11, Real.exp_add]
  have hqbar0 : qbar c0 = Real.exp (1 / t) * Real.cosh (a * t) / Zval := by
    simp only [qbar, Fin.sum_univ_two, π, Matrix.cons_val_zero, Matrix.cons_val_one,
      hq00, hq10, Real.cosh_eq]
    ring
  have hqbar1 : qbar c1 = Real.exp (1 / t) * Real.cosh (a * t) / Zval := by
    simp only [qbar, Fin.sum_univ_two, π, Matrix.cons_val_zero, Matrix.cons_val_one,
      hq01, hq11, Real.cosh_eq]
    ring
  have hcosh_ne : Real.cosh (a * t) ≠ 0 := ne_of_gt (Real.cosh_pos _)
  have hexp_ne : Real.exp (1 / t) ≠ 0 := ne_of_gt (Real.exp_pos _)
  have hZval_ne : Zval ≠ 0 := ne_of_gt hZval_pos
  have hkl_rest_zero : ∀ i : Fin 2, ∑ c ∈ rest, q i c * Real.log (q i c / qbar c) = 0 := by
    intro i
    refine Finset.sum_eq_zero fun c hc => ?_
    have hqeq : q i c = qbar c := by
      rw [hqbar_rest c hc]
      fin_cases i
      · rfl
      · exact (hq_rest_eq c hc).symm
    have hqpos : 0 < q i c := softmax_pos δ (H i) c
    rw [hqeq, div_self (ne_of_gt (hqeq ▸ hqpos)), Real.log_one, mul_zero]
  have hkl0 : klDiv (q 0) qbar = pairMass m t * binaryInfo (a * t) := by
    unfold klDiv
    rw [hsum_split, hkl_rest_zero 0, add_zero, hq00, hq01, hqbar0, hqbar1]
    have hr1 : (Real.exp (a * t) * Real.exp (1 / t) / Zval) /
        (Real.exp (1 / t) * Real.cosh (a * t) / Zval) = Real.exp (a * t) / Real.cosh (a * t) := by
      field_simp [hZval_ne, hexp_ne, hcosh_ne]
    have hr2 : (Real.exp (-(a * t)) * Real.exp (1 / t) / Zval) /
        (Real.exp (1 / t) * Real.cosh (a * t) / Zval) =
        Real.exp (-(a * t)) / Real.cosh (a * t) := by
      field_simp [hZval_ne, hexp_ne, hcosh_ne]
    rw [hr1, hr2, Real.log_div (ne_of_gt (Real.exp_pos _)) hcosh_ne,
      Real.log_div (ne_of_gt (Real.exp_pos _)) hcosh_ne, Real.log_exp, Real.log_exp]
    unfold pairMass binaryInfo
    simp only [a, Zval]
    rw [Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq]
    have hsum_ne : Real.exp (Real.sqrt (1 - t ^ 2) * t) +
        Real.exp (-(Real.sqrt (1 - t ^ 2) * t)) ≠ 0 := by positivity
    field_simp [hZval_ne, hsum_ne]
    ring
  have hkl1 : klDiv (q 1) qbar = pairMass m t * binaryInfo (a * t) := by
    unfold klDiv
    rw [hsum_split, hkl_rest_zero 1, add_zero, hq10, hq11, hqbar0, hqbar1]
    have hr1 : (Real.exp (-(a * t)) * Real.exp (1 / t) / Zval) /
        (Real.exp (1 / t) * Real.cosh (a * t) / Zval) =
        Real.exp (-(a * t)) / Real.cosh (a * t) := by
      field_simp [hZval_ne, hexp_ne, hcosh_ne]
    have hr2 : (Real.exp (a * t) * Real.exp (1 / t) / Zval) /
        (Real.exp (1 / t) * Real.cosh (a * t) / Zval) = Real.exp (a * t) / Real.cosh (a * t) := by
      field_simp [hZval_ne, hexp_ne, hcosh_ne]
    rw [hr1, hr2, Real.log_div (ne_of_gt (Real.exp_pos _)) hcosh_ne,
      Real.log_div (ne_of_gt (Real.exp_pos _)) hcosh_ne, Real.log_exp, Real.log_exp]
    unfold pairMass binaryInfo
    simp only [a, Zval]
    rw [Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq]
    have hsum_ne : Real.exp (Real.sqrt (1 - t ^ 2) * t) +
        Real.exp (-(Real.sqrt (1 - t ^ 2) * t)) ≠ 0 := by positivity
    field_simp [hZval_ne, hsum_ne]
    ring
  have hI : I_k = pairMass m t * binaryInfo (a * t) := by
    simp only [I_k, Fin.sum_univ_two, π, Matrix.cons_val_zero, Matrix.cons_val_one, hkl0, hkl1]
    ring
  exact ⟨hcent, hnorm, hD, hI⟩

/-- Lower bound on `pairMass m t`: for `m > 2` and `0 < t < 1/2`,
`pairMass m t ≥ 1 / (1 + (m - 2) * t)`. -/
lemma pairMass_lower_bound {m : ℕ} (hm : 2 < m) {t : ℝ} (ht0 : 0 < t) :
    1 / (1 + ((m : ℝ) - 2) * t) ≤ pairMass m t := by
  have hm2_pos : (0 : ℝ) < (m : ℝ) - 2 := by
    have : (2 : ℝ) < (m : ℝ) := by exact_mod_cast hm
    linarith
  let a := Real.sqrt (1 - t ^ 2)
  have hcosh1 : 1 ≤ Real.cosh (a * t) := Real.one_le_cosh (a * t)
  have hexp_ge : 1 / t ≤ Real.exp (1 / t) := by
    have := Real.add_one_le_exp (1 / t)
    linarith
  have hexp_neg_le : Real.exp (-2 / (((m : ℝ) - 2) * t)) ≤ 1 := by
    rw [Real.exp_le_one_iff]
    exact div_nonpos_of_nonpos_of_nonneg (by norm_num) (mul_nonneg (le_of_lt hm2_pos) (le_of_lt ht0))
  have hnum_ge : 2 / t ≤ 2 * Real.exp (1 / t) * Real.cosh (a * t) := by
    have h1 : 2 / t = 2 * (1 / t) := by ring
    have h2 : 2 * (1 / t) ≤ 2 * Real.exp (1 / t) := by linarith
    have h3 : 2 * Real.exp (1 / t) * 1 ≤ 2 * Real.exp (1 / t) * Real.cosh (a * t) :=
      mul_le_mul_of_nonneg_left hcosh1 (by positivity)
    linarith
  have htail_le : ((m : ℝ) - 2) * Real.exp (-2 / (((m : ℝ) - 2) * t)) ≤ (m : ℝ) - 2 := by
    nlinarith
  have hden_pos : 0 < 2 * Real.exp (1 / t) * Real.cosh (a * t) +
      ((m : ℝ) - 2) * Real.exp (-2 / (((m : ℝ) - 2) * t)) := by
    have := Real.exp_pos (1 / t)
    have := Real.cosh_pos (a * t)
    have := Real.exp_pos (-2 / (((m : ℝ) - 2) * t))
    nlinarith
  have h1mt_pos : 0 < 1 + ((m : ℝ) - 2) * t := by positivity
  unfold pairMass
  dsimp only
  rw [div_le_div_iff₀ h1mt_pos hden_pos]
  have hprod : ((m : ℝ) - 2) ≤ ((m : ℝ) - 2) * t * (2 * Real.exp (1 / t) * Real.cosh (a * t)) := by
    have ht_ne : t ≠ 0 := ne_of_gt ht0
    have hmul : ((m : ℝ) - 2) * t * (2 / t) = 2 * ((m : ℝ) - 2) := by field_simp [ht_ne]
    have hle := mul_le_mul_of_nonneg_left hnum_ge (by positivity : 0 ≤ ((m : ℝ) - 2) * t)
    linarith
  linarith

/-- **Sharpness for every `m > 2`**: any uniform constant `C > 2` in `R^2 D_k ≥ C I_k` fails for
sufficiently small `t ∈ (0, 1/2)`. -/
theorem sharpness_general_m_refutes_larger_const {m : ℕ} (hm : 2 < m) {C : ℝ} (hC : 2 < C) :
    ∃ t : ℝ, 0 < t ∧ t < 1 / 2 ∧
      t ^ 2 < C * (pairMass m t * binaryInfo (Real.sqrt (1 - t ^ 2) * t)) := by
  have hm2_pos : (0 : ℝ) < (m : ℝ) - 2 := by
    have : (2 : ℝ) < (m : ℝ) := by exact_mod_cast hm
    linarith
  set K : ℝ := (m : ℝ) - 2 + 2
  have hK_pos : 0 < K := by dsimp only [K]; linarith
  set t : ℝ := min (1 / 4) ((C - 2) / (4 * C * K))
  have ht0 : 0 < t := lt_min (by norm_num) (div_pos (by linarith) (by positivity))
  have ht14 : t ≤ 1 / 4 := min_le_left _ _
  have ht_small : t ≤ (C - 2) / (4 * C * K) := min_le_right _ _
  have ht12 : t < 1 / 2 := by linarith
  refine ⟨t, ht0, ht12, ?_⟩
  set a : ℝ := Real.sqrt (1 - t ^ 2)
  have h1t : 0 ≤ 1 - t ^ 2 := by nlinarith
  have hat_nonneg : 0 ≤ a * t := mul_nonneg (Real.sqrt_nonneg _) (le_of_lt ht0)
  have hbin := binaryInfo_lower_bound hat_nonneg
  have ha2 : a ^ 2 = 1 - t ^ 2 := Real.sq_sqrt h1t
  have hbin_simp : (a * t) ^ 2 / 2 - (a * t) ^ 4 / 4 =
      (t ^ 2 / 2) * ((1 - t ^ 2) - (1 - t ^ 2) ^ 2 * t ^ 2 / 2) := by
    have h1 : (a * t) ^ 2 = a ^ 2 * t ^ 2 := by ring
    have h2 : (a * t) ^ 4 = (a ^ 2) ^ 2 * t ^ 4 := by ring
    rw [h1, h2, ha2]; ring
  have hpoly_lb : 1 - 2 * t ≤ (1 - t ^ 2) - (1 - t ^ 2) ^ 2 * t ^ 2 / 2 := by
    have hsq1 : (1 - t ^ 2) ^ 2 ≤ 1 := by nlinarith
    have hsq2 : t ^ 2 ≤ t / 4 := by nlinarith
    have hterm : (1 - t ^ 2) ^ 2 * t ^ 2 ≤ t ^ 2 := by nlinarith [sq_nonneg t]
    linarith
  have hbin_lb : (t ^ 2 / 2) * (1 - 2 * t) ≤ binaryInfo (a * t) := by
    rw [hbin_simp] at hbin
    have hmul := mul_le_mul_of_nonneg_left hpoly_lb (by positivity : 0 ≤ t ^ 2 / 2)
    linarith
  have hpm := pairMass_lower_bound hm ht0
  have h1mt_pos : 0 < 1 + ((m : ℝ) - 2) * t := by positivity
  have hpm_pos : 0 < 1 / (1 + ((m : ℝ) - 2) * t) := div_pos one_pos h1mt_pos
  have hbin_pos : 0 < (t ^ 2 / 2) * (1 - 2 * t) := mul_pos (by positivity) (by linarith)
  have hprod_lb : (1 / (1 + ((m : ℝ) - 2) * t)) * ((t ^ 2 / 2) * (1 - 2 * t)) ≤
      pairMass m t * binaryInfo (a * t) :=
    mul_le_mul hpm hbin_lb (le_of_lt hbin_pos) (le_trans (le_of_lt hpm_pos) hpm)
  have hC_ne : C ≠ 0 := by linarith
  have hK_ne : K ≠ 0 := ne_of_gt hK_pos
  have hKt : K * t ≤ (C - 2) / (4 * C) := by
    calc K * t ≤ K * ((C - 2) / (4 * C * K)) := mul_le_mul_of_nonneg_left ht_small (le_of_lt hK_pos)
      _ = (C - 2) / (4 * C) := by field_simp [hK_ne, hC_ne]
  have hCKt : C * (K * t) ≤ (C - 2) / 4 := by
    calc C * (K * t) ≤ C * ((C - 2) / (4 * C)) := mul_le_mul_of_nonneg_left hKt (by linarith)
      _ = (C - 2) / 4 := by field_simp [hC_ne]
  have hkey : 2 * (1 + ((m : ℝ) - 2) * t) < C * (1 - 2 * t) := by
    have hm_le : 2 * ((m : ℝ) - 2) * t ≤ C * ((m : ℝ) - 2) * t :=
      mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right (le_of_lt hC) (le_of_lt hm2_pos))
        (le_of_lt ht0)
    have hsum_le : 2 * ((m : ℝ) - 2) * t + 2 * C * t ≤ C * (K * t) := by
      have hKeq : C * (K * t) = C * ((m : ℝ) - 2) * t + 2 * C * t := by dsimp only [K]; ring
      linarith
    linarith
  have hratio : 1 < C * ((1 / (1 + ((m : ℝ) - 2) * t)) * ((1 - 2 * t) / 2)) := by
    have heq : C * ((1 / (1 + ((m : ℝ) - 2) * t)) * ((1 - 2 * t) / 2)) =
        (C * (1 - 2 * t)) / (2 * (1 + ((m : ℝ) - 2) * t)) := by
      field_simp [ne_of_gt h1mt_pos]
    rw [heq, one_lt_div (by positivity)]
    exact hkey
  have hfinal : t ^ 2 < C * ((1 / (1 + ((m : ℝ) - 2) * t)) * ((t ^ 2 / 2) * (1 - 2 * t))) := by
    have hmul := mul_lt_mul_of_pos_left hratio (sq_pos_of_pos ht0)
    have hrhs : t ^ 2 * (C * ((1 / (1 + ((m : ℝ) - 2) * t)) * ((1 - 2 * t) / 2))) =
        C * ((1 / (1 + ((m : ℝ) - 2) * t)) * ((t ^ 2 / 2) * (1 - 2 * t))) := by ring
    linarith
  have hC_lb : C * ((1 / (1 + ((m : ℝ) - 2) * t)) * ((t ^ 2 / 2) * (1 - 2 * t))) ≤
      C * (pairMass m t * binaryInfo (a * t)) :=
    mul_le_mul_of_nonneg_left hprod_lb (by linarith)
  exact lt_of_lt_of_le hfinal hC_lb

end LMInfoFloor
