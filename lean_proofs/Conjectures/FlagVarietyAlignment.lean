import Mathlib

/-!
# Refutation of Conjecture 6.2 (Static and Dynamical Head-Subspace Incidence)
(Xiao, Sui, and Ruan, arXiv:2605.09861, Section 6, Conjecture 6.2 `conj:static_head_incidence`)

We formalize in Lean 4 + Mathlib (complete proofs, standard axioms only) the mathematical cores of
all four results refuting both the static and dynamical claims of Conjecture 6.2:

1. **Proposition 1 (Exact Frobenius projector overlap identity)**:
   - `projector_overlap_identity`: For any `h ≥ 2` orthogonal rank-`r` projectors
     `P_1, ..., P_h` in `ℝ^d` (`P_iᵀ = P_i`, `P_i² = P_i`, `tr(P_i) = r`) with aggregate
     `S = ∑_i P_i` and mean pairwise overlap `R = (1 / (h(h-1)r)) ∑_{i≠j} tr(P_i P_j)`,
     `h(h-1)r (R - r/d) = ‖S - (hr/d) I_d‖_F² - hr (1 - r/d)`.
   - `overlap2_bounds`, `overlap2_orth_zero`, `overlap2_collinear_one`: Exact 2D rank-1 overlap
     `R(q₀, q₁) = ⟨q₀, q₁⟩² / (‖q₀‖² ‖q₁‖²) ∈ [0, 1]`, equal to `0 < 1/2` for orthogonal heads
     and `1 > 1/2` for collinear heads.
2. **Theorem 2 (Static orthogonal task: `R = 0 < 1/2` at every global minimizer + strict local
   convexity certificate)**:
   - `two_token_softmax_eq_tanh`: Exact identity relating 2-token `(m, -m)` softmax attention to
     `o (vᵀ m) tanh((qᵀ x)(kᵀ m))`.
   - `orth_obj_lower_bound` & `orth_minimizer_attains`: AM–GM + `tanh`-monotonicity lower bound
     `F(θ) ≥ m(w₀) + m(w₁)` attained exactly by the dedicated orthogonal configuration
     `q_j = k_j = √b_j e_j`, `v_j = o_j = √a_j e_j` with `R = 0 < 1/2`.
   - `orth_every_global_min_R_zero`: For `λ = 1/100` and `w₀, w₁ ≥ 1` (covering both isotropic
     `(1, 1)` and anisotropic `(2, 1)`), EVERY global minimizer of `F` has nonzero mutually
     orthogonal query vectors (`q₀ ⊥ q₁`), forcing `R = 0 < 1/2`.
   - `stationarityCubic_min_root_pos_deriv` & `reduced_cost_hasDerivAt`: For `δ = λ / w ∈ (0, 1/4)`,
     the reduced cost derivative is `λ · p(s) / s³` (`s = tanh b`), where
     `p(s) = 2s³ - δs² - s + δ` has a root `s* ∈ (1/2, 1)` with strictly positive derivative
     `p'(s*) > 0`, certifying strict positive definiteness of the reduced `(a, b)` Hessian and
     an open attracting basin.
3. **Proposition 3 (Shared query direction on identical anisotropic inputs gives `R = 1 > 1/2`)**:
   - `shared_every_global_min_R_one`: Keeping `w = (2, 1)` and `λ = 1/100` fixed and changing
     only the targets to `y_{lj} = 1_{l=0} e_j`, EVERY global minimizer has both `q₀, q₁` nonzero
     and collinear with `e₀`, forcing `R = 1 > 1/2`.
4. **Proposition 4 (Dynamic alignment drift: exact `dR/dt|_{t=0} < 0` under gradient flow)**:
   - `tilted_gradient_exact` & `dynamic_overlap_deriv_neg`: Starting at the isotropic orthogonal
     minimizer with `q₀` tilted to `(c, t)` (`c = √b₁ > 0, t > 0`), exact `HasDerivAt` gradients
     of the full 16-parameter objective yield `q̇_{00}(0) = 0`, `q̇_1(0) = 0`,
     `ṫ(0) = -λt - w₁ a₁² c tanh(ct) sech²(ct) < 0`, `R(0) = t² / (c² + t²)`, and
     `Ṙ(0) = (2 t c² / (c² + t²)²) ṫ(0) < 0`. At `t = 1.1 c`, `R(0) = 121/221 > 1/2` starts
     strictly above the Haar baseline while strictly decreasing.

Machine-checked vs prose-only:
- Machine-checked: Complete Proposition 1 projector identity, complete Theorem 2 global optimality
  (`R = 0` at every global minimizer) and stationarity-cubic strict curvature certificate,
  complete Proposition 3 (`R = 1` at every global minimizer), and complete Proposition 4 exact
  `HasDerivAt` negative initial drift `Ṙ(0) < 0` from `R(0) > 1/2`.
- Prose-only (in `solutions/flag_variety_alignment/proof.pdf`): Haar integration `E[R] = r/d`,
  assembly of the full 16×16 block-diagonal Hessian into a global ODE basin radius, and
  numerical DOP853/L-BFGS-B trajectory verification.
-/

open Matrix BigOperators Finset

namespace FlagVarietyAlignment

/-! ### 1. Proposition 1: Exact Frobenius Projector Identity & 2D Overlap -/

/-- Squared Frobenius norm `∑_{i,j} A_{ij}²` of a `d × d` real matrix. -/
def frobSq {d : ℕ} (A : Matrix (Fin d) (Fin d) ℝ) : ℝ :=
  ∑ i : Fin d, ∑ j : Fin d, (A i j) ^ 2

lemma frobSq_eq_trace_mul_transpose {d : ℕ} (A : Matrix (Fin d) (Fin d) ℝ) :
    frobSq A = Matrix.trace (A * Aᵀ) := by
  unfold frobSq
  simp only [Matrix.trace, Matrix.diag_apply]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Matrix.mul_apply]
  refine Finset.sum_congr rfl fun j _ => ?_
  simp [sq]

/-- **Proposition 1(2) (Exact Frobenius projector overlap identity)**:
For `h ≥ 2` symmetric idempotent rank-`r` projectors `P : Fin h → Matrix (Fin d) (Fin d) ℝ`
(`P_iᵀ = P_i`, `P_i * P_i = P_i`, `trace (P_i) = r`) with aggregate `S = ∑_i P_i` and
`R = (∑_{i ≠ j} trace (P_i * P_j)) / (h * (h - 1) * r)`,
`h(h-1)r (R - r/d) = frobSq (S - (hr/d) • 1) - hr (1 - r/d)`. -/
theorem projector_overlap_identity {d h : ℕ} (hd : 1 ≤ d) (hh : 2 ≤ h) {r : ℝ} (hr : r ≠ 0)
    (P : Fin h → Matrix (Fin d) (Fin d) ℝ)
    (hsym : ∀ i, (P i)ᵀ = P i)
    (hidem : ∀ i, P i * P i = P i)
    (htr : ∀ i, Matrix.trace (P i) = r) :
    let S : Matrix (Fin d) (Fin d) ℝ := ∑ i : Fin h, P i
    let offSum : ℝ := ∑ i : Fin h, ∑ j ∈ Finset.univ.erase i, Matrix.trace (P i * P j)
    let R : ℝ := offSum / ((h : ℝ) * ((h : ℝ) - 1) * r)
    (h : ℝ) * ((h : ℝ) - 1) * r * (R - r / (d : ℝ)) =
      frobSq (S - (((h : ℝ) * r) / (d : ℝ)) • (1 : Matrix (Fin d) (Fin d) ℝ)) -
      (h : ℝ) * r * (1 - r / (d : ℝ)) := by
  intro S offSum R
  have hd0 : (d : ℝ) ≠ 0 := by positivity
  have hh0 : (h : ℝ) ≠ 0 := by positivity
  have hh1 : (h : ℝ) - 1 ≠ 0 := by
    have : (2 : ℝ) ≤ (h : ℝ) := by exact_mod_cast hh
    linarith
  have hS_sym : Sᵀ = S := by
    dsimp only [S]
    rw [Matrix.transpose_sum]
    exact Finset.sum_congr rfl fun i _ => hsym i
  have htrS : Matrix.trace S = (h : ℝ) * r := by
    dsimp only [S]
    rw [Matrix.trace_sum]
    simp [htr]
  have htrS2 : Matrix.trace (S * S) = (h : ℝ) * r + offSum := by
    dsimp only [S, offSum]
    rw [Finset.sum_mul, Matrix.trace_sum]
    have hsplit : ∀ i : Fin h,
        Matrix.trace (P i * ∑ j : Fin h, P j) =
          r + ∑ j ∈ Finset.univ.erase i, Matrix.trace (P i * P j) := by
      intro i
      rw [Finset.mul_sum, Matrix.trace_sum, ← Finset.add_sum_erase _ _ (Finset.mem_univ i),
        hidem i, htr i]
    simp_rw [hsplit, Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
      nsmul_eq_mul]
  set c : ℝ := ((h : ℝ) * r) / (d : ℝ)
  have hM_sym : (S - c • (1 : Matrix (Fin d) (Fin d) ℝ))ᵀ = S - c • 1 := by
    rw [Matrix.transpose_sub, Matrix.transpose_smul, Matrix.transpose_one, hS_sym]
  have hfrob : frobSq (S - c • (1 : Matrix (Fin d) (Fin d) ℝ)) =
      Matrix.trace (S * S) - 2 * c * Matrix.trace S + c ^ 2 * (d : ℝ) := by
    rw [frobSq_eq_trace_mul_transpose, hM_sym]
    have hexp : (S - c • (1 : Matrix (Fin d) (Fin d) ℝ)) * (S - c • 1) =
        S * S - (2 * c) • S + (c ^ 2) • (1 : Matrix (Fin d) (Fin d) ℝ) := by
      simp only [Matrix.sub_mul, Matrix.mul_sub, Matrix.smul_mul, Matrix.mul_smul,
        Matrix.one_mul, Matrix.mul_one]
      ext i j
      simp only [Matrix.sub_apply, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul]
      ring
    rw [hexp, Matrix.trace_add, Matrix.trace_sub, Matrix.trace_smul, Matrix.trace_smul,
      Matrix.trace_one, Fintype.card_fin]
    simp only [smul_eq_mul]
  have hR_mul : (h : ℝ) * ((h : ℝ) - 1) * r * R = offSum := by
    dsimp only [R]
    exact mul_div_cancel₀ offSum (mul_ne_zero (mul_ne_zero hh0 hh1) hr)
  rw [hfrob, htrS2, htrS]
  have hleft : (h : ℝ) * ((h : ℝ) - 1) * r * (R - r / (d : ℝ)) =
      offSum - (h : ℝ) * ((h : ℝ) - 1) * r * (r / (d : ℝ)) := by
    calc (h : ℝ) * ((h : ℝ) - 1) * r * (R - r / (d : ℝ))
        = (h : ℝ) * ((h : ℝ) - 1) * r * R - (h : ℝ) * ((h : ℝ) - 1) * r * (r / (d : ℝ)) := by ring
      _ = offSum - (h : ℝ) * ((h : ℝ) - 1) * r * (r / (d : ℝ)) := by rw [hR_mul]
  rw [hleft]
  dsimp only [c]
  field_simp [hd0]
  ring

/-- Mean pairwise Frobenius overlap `R = ⟨q₀, q₁⟩² / (‖q₀‖² ‖q₁‖²)` for two rank-1 query heads
in `ℝ²` (`d = h = 2, r = 1`, Haar baseline `r/d = 1/2`). -/
noncomputable def overlap2 (q₀ q₁ : Fin 2 → ℝ) : ℝ :=
  (q₀ 0 * q₁ 0 + q₀ 1 * q₁ 1) ^ 2 /
    ((q₀ 0 ^ 2 + q₀ 1 ^ 2) * (q₁ 0 ^ 2 + q₁ 1 ^ 2))

lemma overlap2_bounds (q₀ q₁ : Fin 2 → ℝ) :
    0 ≤ overlap2 q₀ q₁ ∧ overlap2 q₀ q₁ ≤ 1 := by
  unfold overlap2
  have hden_nn : 0 ≤ (q₀ 0 ^ 2 + q₀ 1 ^ 2) * (q₁ 0 ^ 2 + q₁ 1 ^ 2) := by positivity
  constructor
  · exact div_nonneg (sq_nonneg _) hden_nn
  · by_cases h0 : (q₀ 0 ^ 2 + q₀ 1 ^ 2) * (q₁ 0 ^ 2 + q₁ 1 ^ 2) = 0
    · rw [h0, div_zero]; norm_num
    · have hpos : 0 < (q₀ 0 ^ 2 + q₀ 1 ^ 2) * (q₁ 0 ^ 2 + q₁ 1 ^ 2) :=
        lt_of_le_of_ne hden_nn (Ne.symm h0)
      rw [div_le_one hpos]
      nlinarith [sq_nonneg (q₀ 0 * q₁ 1 - q₀ 1 * q₁ 0)]

lemma overlap2_orth_zero {q₀ q₁ : Fin 2 → ℝ} (horth : q₀ 0 * q₁ 0 + q₀ 1 * q₁ 1 = 0) :
    overlap2 q₀ q₁ = 0 := by
  simp [overlap2, horth]

lemma overlap2_collinear_e0 {c₀ c₁ : ℝ} (h0 : c₀ ≠ 0) (h1 : c₁ ≠ 0) :
    overlap2 ![c₀, 0] ![c₁, 0] = 1 := by
  unfold overlap2
  simp only [Matrix.cons_val_zero, Matrix.cons_val_one, mul_zero, add_zero,
    zero_pow (by norm_num : 2 ≠ 0)]
  have hne : c₀ ^ 2 * c₁ ^ 2 ≠ 0 := mul_ne_zero (pow_ne_zero 2 h0) (pow_ne_zero 2 h1)
  rw [mul_pow, div_self hne]

/-! ### 2. Two-Head Softmax Attention Layer and Static Objective (Theorem 2) -/

/-- Exact 2-token softmax identity: attending to `(m, -m)` with logits `(s, -s)` produces
`v * tanh s`. -/
theorem two_token_softmax_eq_tanh (s v : ℝ) :
    (Real.exp s * v + Real.exp (-s) * (-v)) / (Real.exp s + Real.exp (-s)) =
      v * Real.tanh s := by
  rw [Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq]
  have hpos : 0 < Real.exp s + Real.exp (-s) := by positivity
  field_simp [ne_of_gt hpos]
  ring

/-- The 16 trainable parameters of the two-head `d = h = 2, r = d_v = 1` attention layer. -/
structure AttnParams where
  q : Fin 2 → Fin 2 → ℝ
  k : Fin 2 → Fin 2 → ℝ
  v : Fin 2 → Fin 2 → ℝ
  o : Fin 2 → Fin 2 → ℝ

/-- Standard basis vector `e_j ∈ ℝ²`. -/
def basis2 (j : Fin 2) : Fin 2 → ℝ := fun c => if c = j then 1 else 0

/-- Exact layer output `f(x, m)_c = ∑_{i=0}^1 o_{i,c} (v_iᵀ m) tanh((q_iᵀ x)(k_iᵀ m))`. -/
noncomputable def attnLayer (θ : AttnParams) (x m : Fin 2 → ℝ) (c : Fin 2) : ℝ :=
  ∑ i : Fin 2,
    θ.o i c * (θ.v i 0 * m 0 + θ.v i 1 * m 1) *
      Real.tanh ((θ.q i 0 * x 0 + θ.q i 1 * x 1) * (θ.k i 0 * m 0 + θ.k i 1 * m 1))

lemma attnLayer_basis (θ : AttnParams) (l j c : Fin 2) :
    attnLayer θ (basis2 l) (basis2 j) c =
      ∑ i : Fin 2, θ.o i c * θ.v i j * Real.tanh (θ.q i l * θ.k i j) := by
  unfold attnLayer
  refine Finset.sum_congr rfl fun i _ => ?_
  fin_cases l <;> fin_cases j <;> simp [basis2]

/-- Squared-error objective with `L₂` weight decay `λ` on the four basis input pairs `(e_l, e_j)`. -/
noncomputable def objF (w : Fin 2 → ℝ) (y : Fin 2 → Fin 2 → Fin 2 → ℝ) (lam : ℝ)
    (θ : AttnParams) : ℝ :=
  (1 / 2 : ℝ) * ∑ l : Fin 2, w l * ∑ j : Fin 2, ∑ c : Fin 2,
      (attnLayer θ (basis2 l) (basis2 j) c - y l j c) ^ 2 +
  (lam / 2) * ∑ i : Fin 2, ∑ j : Fin 2,
      (θ.q i j ^ 2 + θ.k i j ^ 2 + θ.v i j ^ 2 + θ.o i j ^ 2)

/-- Orthogonal task targets `y_{lj} = 1_{l = j} e_j`. -/
def yOrth (l j c : Fin 2) : ℝ :=
  if l = j ∧ c = j then 1 else 0

/-- Shared-query task targets `y_{lj} = 1_{l = 0} e_j`. -/
def yShared (l j c : Fin 2) : ℝ :=
  if l = 0 ∧ c = j then 1 else 0

/-- Reduced 1D scalar cost `φ_w(a, b) = (w/2)(1 - a tanh b)² + λ(a + b)` for `a, b ≥ 0`. -/
noncomputable def scalarCost (wj lam a b : ℝ) : ℝ :=
  (wj / 2) * (1 - a * Real.tanh b) ^ 2 + lam * (a + b)

/-- `Real.tanh` is strictly increasing on `ℝ` (hence on `[0, ∞)`). -/
lemma strictMono_tanh : StrictMono Real.tanh := by
  intro x y hxy
  rw [Real.tanh_eq_sinh_div_cosh, Real.tanh_eq_sinh_div_cosh]
  have hcx : 0 < Real.cosh x := Real.cosh_pos x
  have hcy : 0 < Real.cosh y := Real.cosh_pos y
  rw [div_lt_div_iff₀ hcx hcy]
  have hdiff : Real.sinh y * Real.cosh x - Real.sinh x * Real.cosh y = Real.sinh (y - x) := by
    rw [Real.sinh_sub]; ring
  have hpos : 0 < Real.sinh (y - x) := Real.sinh_pos_iff.mpr (sub_pos.mpr hxy)
  linarith

lemma tanh_nonneg_of_nonneg {x : ℝ} (hx : 0 ≤ x) : 0 ≤ Real.tanh x := by
  rw [← Real.tanh_zero]
  exact strictMono_tanh.monotone hx

lemma tanh_pos_of_pos {x : ℝ} (hx : 0 < x) : 0 < Real.tanh x := by
  rw [← Real.tanh_zero]
  exact strictMono_tanh hx

lemma abs_tanh_eq (x : ℝ) : |Real.tanh x| = Real.tanh |x| := by
  rcases le_total 0 x with h | h
  · rw [abs_of_nonneg h, abs_of_nonneg (tanh_nonneg_of_nonneg h)]
  · rw [abs_of_nonpos h, Real.tanh_neg, abs_of_nonpos (by
      have := tanh_nonneg_of_nonneg (neg_nonneg.mpr h)
      rw [Real.tanh_neg] at this; linarith)]

/-- Explicit numerical lower bound `1 / 2 < Real.tanh 1` (used to show the trial `b = 1`
strictly beats zero activity `a = b = 0` at `λ = 1/100`). -/
lemma half_lt_tanh_one : (1 / 2 : ℝ) < Real.tanh 1 := by
  rw [Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq]
  have hpos : 0 < Real.exp 1 + Real.exp (-1) := by positivity
  have hinv : Real.exp 1 * Real.exp (-1) = 1 := by
    rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
  have he : (2 : ℝ) < Real.exp 1 := by
    have h1 := Real.add_one_lt_exp (by norm_num : (1 : ℝ) ≠ 0)
    -- Better: exp 1 ≥ 1 + 1 + 1/2 = 5/2 > 2 via exp(1/2) ≥ 3/2
    have hhalf : (3 / 2 : ℝ) ≤ Real.exp (1 / 2) := by
      linarith [Real.add_one_le_exp (1 / 2 : ℝ)]
    have hsq : Real.exp (1 / 2) * Real.exp (1 / 2) = Real.exp 1 := by
      rw [← Real.exp_add]; norm_num
    nlinarith
  have hsq_gt : (3 : ℝ) < Real.exp 1 * Real.exp 1 := by nlinarith
  have hlt : 3 * Real.exp (-1) < Real.exp 1 := by
    have hmul : 3 * Real.exp (-1) * Real.exp 1 = 3 := by
      calc 3 * Real.exp (-1) * Real.exp 1 = 3 * (Real.exp 1 * Real.exp (-1)) := by ring
        _ = 3 := by rw [hinv, mul_one]
    have hexp_pos : 0 < Real.exp 1 := Real.exp_pos 1
    nlinarith
  rw [div_div_div_cancel_right₀ (by norm_num : (2 : ℝ) ≠ 0), div_lt_div_iff₀ (by norm_num) hpos]
  linarith

/-- Trial bound at `λ = 1/100`: `scalarCost wj (1/100) (Real.tanh 1)⁻¹ 1 < 3 / 100`. -/
lemma trial_scalarCost_lt (wj : ℝ) :
    scalarCost wj (1 / 100) (Real.tanh 1)⁻¹ 1 < 3 / 100 := by
  unfold scalarCost
  have ht : (1 / 2 : ℝ) < Real.tanh 1 := half_lt_tanh_one
  have ht_pos : 0 < Real.tanh 1 := by linarith
  rw [inv_mul_cancel₀ (ne_of_gt ht_pos), sub_self, zero_pow (by norm_num : 2 ≠ 0), mul_zero,
      zero_add]
  have hinv_lt : (Real.tanh 1)⁻¹ < 2 := by
    rw [inv_lt_comm₀ ht_pos (by norm_num : (0 : ℝ) < 2)]
    linarith
  linarith

/-- Helper: key per-coordinate inequality from the proof of Theorem 2.
For coordinate `j`, letting `A_j = ∑_i |o_{ij} v_{ij}|` and `B_j = ∑_i |q_{ij} k_{ij}|`,
the retained scalar prediction `z_j = attnLayer θ (basis2 j) (basis2 j) j` satisfies
`|z_j| ≤ A_j * tanh B_j`, and there exist `a_j, b_j ≥ 0` with `a_j ≤ A_j`, `b_j = B_j`, and
`a_j * tanh b_j = |z_j|`, yielding `scalarCost (w j) lam a_j b_j ≤` the coordinate-`j` contribution
to `objF`. -/
lemma orth_coord_bound (wj lam : ℝ) (hw : 0 ≤ wj) (hlam : 0 ≤ lam)
    (q0 k0 v0 o0 q1 k1 v1 o1 : ℝ) :
    let z := o0 * v0 * Real.tanh (q0 * k0) + o1 * v1 * Real.tanh (q1 * k1)
    let A := |o0 * v0| + |o1 * v1|
    let B := |q0 * k0| + |q1 * k1|
    ∃ a b : ℝ, 0 ≤ a ∧ 0 ≤ b ∧ a ≤ A ∧ b = B ∧
      scalarCost wj lam a b ≤
        (wj / 2) * (z - 1) ^ 2 +
        (lam / 2) * (q0 ^ 2 + k0 ^ 2 + v0 ^ 2 + o0 ^ 2 + q1 ^ 2 + k1 ^ 2 + v1 ^ 2 + o1 ^ 2) := by
  intro z A B
  have hB_nn : 0 ≤ B := add_nonneg (abs_nonneg _) (abs_nonneg _)
  have hA_nn : 0 ≤ A := add_nonneg (abs_nonneg _) (abs_nonneg _)
  -- AM-GM on the 4 pairs of squares
  have hamgm : 2 * (A + B) ≤
      q0 ^ 2 + k0 ^ 2 + v0 ^ 2 + o0 ^ 2 + q1 ^ 2 + k1 ^ 2 + v1 ^ 2 + o1 ^ 2 := by
    have h1 : 2 * |o0 * v0| ≤ o0 ^ 2 + v0 ^ 2 := by
      rw [abs_mul]; nlinarith [sq_nonneg (|o0| - |v0|), sq_abs o0, sq_abs v0]
    have h2 : 2 * |o1 * v1| ≤ o1 ^ 2 + v1 ^ 2 := by
      rw [abs_mul]; nlinarith [sq_nonneg (|o1| - |v1|), sq_abs o1, sq_abs v1]
    have h3 : 2 * |q0 * k0| ≤ q0 ^ 2 + k0 ^ 2 := by
      rw [abs_mul]; nlinarith [sq_nonneg (|q0| - |k0|), sq_abs q0, sq_abs k0]
    have h4 : 2 * |q1 * k1| ≤ q1 ^ 2 + k1 ^ 2 := by
      rw [abs_mul]; nlinarith [sq_nonneg (|q1| - |k1|), sq_abs q1, sq_abs k1]
    linarith
  have hz_le : |z| ≤ A * Real.tanh B := by
    have htri : |z| ≤ |o0 * v0| * Real.tanh |q0 * k0| + |o1 * v1| * Real.tanh |q1 * k1| := by
      calc |z| ≤ |o0 * v0 * Real.tanh (q0 * k0)| + |o1 * v1 * Real.tanh (q1 * k1)| :=
            abs_add_le _ _
        _ = |o0 * v0| * |Real.tanh (q0 * k0)| + |o1 * v1| * |Real.tanh (q1 * k1)| := by
            rw [abs_mul (o0 * v0), abs_mul (o1 * v1)]
        _ = |o0 * v0| * Real.tanh |q0 * k0| + |o1 * v1| * Real.tanh |q1 * k1| := by
            rw [abs_tanh_eq, abs_tanh_eq]
    have ht0 : Real.tanh |q0 * k0| ≤ Real.tanh B :=
      strictMono_tanh.monotone (by dsimp only [B]; linarith [abs_nonneg (q1 * k1)])
    have ht1 : Real.tanh |q1 * k1| ≤ Real.tanh B :=
      strictMono_tanh.monotone (by dsimp only [B]; linarith [abs_nonneg (q0 * k0)])
    nlinarith [abs_nonneg (o0 * v0), abs_nonneg (o1 * v1)]
  have hsq_z : (1 - |z|) ^ 2 ≤ (z - 1) ^ 2 := by
    have hle : z ≤ |z| := le_abs_self z
    have hsq : |z| ^ 2 = z ^ 2 := sq_abs z
    nlinarith
  by_cases hB0 : B = 0
  · refine ⟨0, 0, le_refl 0, le_refl 0, hA_nn, hB0.symm, ?_⟩
    have hz0 : z = 0 := by
      rw [hB0, Real.tanh_zero, mul_zero] at hz_le
      exact abs_nonpos_iff.mp hz_le
    unfold scalarCost
    rw [hz0]
    simp only [zero_mul, sub_zero, add_zero, mul_zero, one_pow, zero_sub, neg_sq]
    nlinarith [hamgm]
  · have hBpos : 0 < B := lt_of_le_of_ne hB_nn (Ne.symm hB0)
    have htpos : 0 < Real.tanh B := tanh_pos_of_pos hBpos
    refine ⟨|z| / Real.tanh B, B, div_nonneg (abs_nonneg _) (le_of_lt htpos), hB_nn,
      (div_le_iff₀ htpos).mpr hz_le, rfl, ?_⟩
    unfold scalarCost
    rw [div_mul_cancel₀ _ (ne_of_gt htpos)]
    have ha_le_A : |z| / Real.tanh B ≤ A := (div_le_iff₀ htpos).mpr hz_le
    nlinarith [hsq_z, hamgm, ha_le_A]

/-- **Theorem 2(1a) — Global lower bound `F(θ) ≥ m(w₀) + m(w₁)`**:
If `(a₀, b₀)` and `(a₁, b₁)` minimize `scalarCost (w 0) lam` and `scalarCost (w 1) lam` over
`[0, ∞)²`, then for EVERY 16-parameter tuple `θ`, `objF w yOrth lam θ ≥ scalarCost (w 0) lam a₀ b₀ + scalarCost (w 1) lam a₁ b₁`. -/
theorem orth_obj_lower_bound {w : Fin 2 → ℝ} (hw0 : 0 ≤ w 0) (hw1 : 0 ≤ w 1)
    {lam m0 m1 : ℝ} (hlam : 0 ≤ lam)
    (hm0 : ∀ a b : ℝ, 0 ≤ a → 0 ≤ b → m0 ≤ scalarCost (w 0) lam a b)
    (hm1 : ∀ a b : ℝ, 0 ≤ a → 0 ≤ b → m1 ≤ scalarCost (w 1) lam a b)
    (θ : AttnParams) :
    m0 + m1 ≤ objF w yOrth lam θ := by
  obtain ⟨a₀, b₀, ha₀, hb₀, _, _, hbd0⟩ := orth_coord_bound (w 0) lam hw0 hlam
    (θ.q 0 0) (θ.k 0 0) (θ.v 0 0) (θ.o 0 0) (θ.q 1 0) (θ.k 1 0) (θ.v 1 0) (θ.o 1 0)
  obtain ⟨a₁, b₁, ha₁, hb₁, _, _, hbd1⟩ := orth_coord_bound (w 1) lam hw1 hlam
    (θ.q 0 1) (θ.k 0 1) (θ.v 0 1) (θ.o 0 1) (θ.q 1 1) (θ.k 1 1) (θ.v 1 1) (θ.o 1 1)
  have hm0_le := hm0 a₀ b₀ ha₀ hb₀
  have hm1_le := hm1 a₁ b₁ ha₁ hb₁
  unfold objF
  simp only [Fin.sum_univ_two, attnLayer_basis, yOrth]
  simp only [show (0 : Fin 2) ≠ 1 by decide, show (1 : Fin 2) ≠ 0 by decide, and_self, and_false,
    false_and, ↓reduceIte, sub_zero, mul_add]
  simp only [show ∀ u v : ℝ, (1 / 2 : ℝ) * (u * v) = (u / 2) * v from fun u v => by ring]
  have hw02 : 0 ≤ w 0 / 2 := by linarith
  have hw12 : 0 ≤ w 1 / 2 := by linarith
  have hs1 := mul_nonneg hw02 (sq_nonneg (θ.o 0 1 * θ.v 0 0 * Real.tanh (θ.q 0 0 * θ.k 0 0) +
      θ.o 1 1 * θ.v 1 0 * Real.tanh (θ.q 1 0 * θ.k 1 0)))
  have hs2 := mul_nonneg hw02 (sq_nonneg (θ.o 0 0 * θ.v 0 1 * Real.tanh (θ.q 0 0 * θ.k 0 1) +
      θ.o 1 0 * θ.v 1 1 * Real.tanh (θ.q 1 0 * θ.k 1 1)))
  have hs3 := mul_nonneg hw02 (sq_nonneg (θ.o 0 1 * θ.v 0 1 * Real.tanh (θ.q 0 0 * θ.k 0 1) +
      θ.o 1 1 * θ.v 1 1 * Real.tanh (θ.q 1 0 * θ.k 1 1)))
  have hs4 := mul_nonneg hw12 (sq_nonneg (θ.o 0 0 * θ.v 0 0 * Real.tanh (θ.q 0 1 * θ.k 0 0) +
      θ.o 1 0 * θ.v 1 0 * Real.tanh (θ.q 1 1 * θ.k 1 0)))
  have hs5 := mul_nonneg hw12 (sq_nonneg (θ.o 0 1 * θ.v 0 0 * Real.tanh (θ.q 0 1 * θ.k 0 0) +
      θ.o 1 1 * θ.v 1 0 * Real.tanh (θ.q 1 1 * θ.k 1 0)))
  have hs6 := mul_nonneg hw12 (sq_nonneg (θ.o 0 0 * θ.v 0 1 * Real.tanh (θ.q 0 1 * θ.k 0 1) +
      θ.o 1 0 * θ.v 1 1 * Real.tanh (θ.q 1 1 * θ.k 1 1)))
  linarith

/-- Dedicated orthogonal configuration `q_j = k_j = √b_j e_j`, `v_j = o_j = √a_j e_j`. -/
noncomputable def orthParams (a₀ b₀ a₁ b₁ : ℝ) : AttnParams where
  q := ![![Real.sqrt b₀, 0], ![0, Real.sqrt b₁]]
  k := ![![Real.sqrt b₀, 0], ![0, Real.sqrt b₁]]
  v := ![![Real.sqrt a₀, 0], ![0, Real.sqrt a₁]]
  o := ![![Real.sqrt a₀, 0], ![0, Real.sqrt a₁]]

/-- **Theorem 2(1b) — Exact attainment with `R = 0 < 1/2`**:
For any `a₀, b₀, a₁, b₁ > 0`, `orthParams a₀ b₀ a₁ b₁` achieves
`objF w yOrth lam = scalarCost (w 0) lam a₀ b₀ + scalarCost (w 1) lam a₁ b₁` and has nonzero
orthogonal query vectors with `overlap2 = 0 < 1 / 2`. -/
theorem orth_minimizer_attains (w : Fin 2 → ℝ) (lam : ℝ)
    {a₀ b₀ a₁ b₁ : ℝ} (ha₀ : 0 < a₀) (hb₀ : 0 < b₀) (ha₁ : 0 < a₁) (hb₁ : 0 < b₁) :
    let θ := orthParams a₀ b₀ a₁ b₁
    objF w yOrth lam θ = scalarCost (w 0) lam a₀ b₀ + scalarCost (w 1) lam a₁ b₁ ∧
    θ.q 0 ≠ 0 ∧ θ.q 1 ≠ 0 ∧
    overlap2 (θ.q 0) (θ.q 1) = 0 := by
  intro θ
  have ha0s : Real.sqrt a₀ * Real.sqrt a₀ = a₀ := Real.mul_self_sqrt (le_of_lt ha₀)
  have hb0s : Real.sqrt b₀ * Real.sqrt b₀ = b₀ := Real.mul_self_sqrt (le_of_lt hb₀)
  have ha1s : Real.sqrt a₁ * Real.sqrt a₁ = a₁ := Real.mul_self_sqrt (le_of_lt ha₁)
  have hb1s : Real.sqrt b₁ * Real.sqrt b₁ = b₁ := Real.mul_self_sqrt (le_of_lt hb₁)
  have ha0q : (Real.sqrt a₀) ^ 2 = a₀ := Real.sq_sqrt (le_of_lt ha₀)
  have hb0q : (Real.sqrt b₀) ^ 2 = b₀ := Real.sq_sqrt (le_of_lt hb₀)
  have ha1q : (Real.sqrt a₁) ^ 2 = a₁ := Real.sq_sqrt (le_of_lt ha₁)
  have hb1q : (Real.sqrt b₁) ^ 2 = b₁ := Real.sq_sqrt (le_of_lt hb₁)
  refine ⟨?_, ?_, ?_, ?_⟩
  · unfold objF scalarCost
    simp only [Fin.sum_univ_two, attnLayer_basis, yOrth, θ, orthParams,
      Matrix.cons_val_zero, Matrix.cons_val_one, show (0 : Fin 2) ≠ 1 by decide,
      show (1 : Fin 2) ≠ 0 by decide, and_self, and_false, false_and, ↓reduceIte,
      mul_zero, zero_mul, Real.tanh_zero, add_zero, zero_add, sub_zero,
      zero_pow (by norm_num : 2 ≠ 0), ha0s, hb0s, ha1s, hb1s, ha0q, hb0q, ha1q, hb1q]
    ring
  · intro h
    have h0 := congrFun h 0
    simp only [θ, orthParams, Matrix.cons_val_zero, Pi.zero_apply] at h0
    have : 0 < Real.sqrt b₀ := Real.sqrt_pos.mpr hb₀
    linarith
  · intro h
    have h1 := congrFun h 1
    simp only [θ, orthParams, Matrix.cons_val_zero, Matrix.cons_val_one, Pi.zero_apply] at h1
    have : 0 < Real.sqrt b₁ := Real.sqrt_pos.mpr hb₁
    linarith
  · apply overlap2_orth_zero
    simp [θ, orthParams]

/-- Helper lemma for Theorem 2(1c): equality in a coordinate forces a single head to carry that
coordinate with all four entries nonzero while the other head vanishes identically on it. -/
private lemma orth_coord_spec (wj m q0 k0 v0 o0 q1 k1 v1 o1 : ℝ)
    (hwj : 1 ≤ wj) (hm_lt : m < 3 / 100)
    (hm_min : ∀ a b : ℝ, 0 ≤ a → 0 ≤ b → m ≤ scalarCost wj (1 / 100) a b)
    (hCeq : (wj / 2) * (o0 * v0 * Real.tanh (q0 * k0) + o1 * v1 * Real.tanh (q1 * k1) - 1) ^ 2 +
      ((1 / 100 : ℝ) / 2) * (q0 ^ 2 + k0 ^ 2 + v0 ^ 2 + o0 ^ 2 + q1 ^ 2 + k1 ^ 2 + v1 ^ 2 + o1 ^ 2) = m) :
    (q0 ≠ 0 ∧ k0 ≠ 0 ∧ v0 ≠ 0 ∧ o0 ≠ 0 ∧ q1 = 0 ∧ k1 = 0 ∧ v1 = 0 ∧ o1 = 0) ∨
    (q1 ≠ 0 ∧ k1 ≠ 0 ∧ v1 ≠ 0 ∧ o1 ≠ 0 ∧ q0 = 0 ∧ k0 = 0 ∧ v0 = 0 ∧ o0 = 0) := by
  set A := |o0 * v0| + |o1 * v1|
  set B := |q0 * k0| + |q1 * k1|
  set z := o0 * v0 * Real.tanh (q0 * k0) + o1 * v1 * Real.tanh (q1 * k1)
  have hA_nn : 0 ≤ A := add_nonneg (abs_nonneg _) (abs_nonneg _)
  have hB_nn : 0 ≤ B := add_nonneg (abs_nonneg _) (abs_nonneg _)
  have hz_ne : z ≠ 0 := by
    intro hz0
    have hCeq' : (wj / 2) * (z - 1) ^ 2 +
        ((1 / 100 : ℝ) / 2) * (q0 ^ 2 + k0 ^ 2 + v0 ^ 2 + o0 ^ 2 + q1 ^ 2 + k1 ^ 2 + v1 ^ 2 + o1 ^ 2) = m :=
      hCeq
    rw [hz0] at hCeq'
    simp only [zero_sub, neg_sq, one_pow, mul_one] at hCeq'
    have : wj / 2 ≤ m := by
      linarith [sq_nonneg q0, sq_nonneg k0, sq_nonneg v0, sq_nonneg o0,
                sq_nonneg q1, sq_nonneg k1, sq_nonneg v1, sq_nonneg o1]
    linarith
  have hB_pos : 0 < B := by
    by_contra hle
    have hB0 : B = 0 := le_antisymm (not_lt.mp hle) hB_nn
    have h00 : q0 * k0 = 0 := abs_eq_zero.mp (by linarith [abs_nonneg (q0 * k0), abs_nonneg (q1 * k1)])
    have h11 : q1 * k1 = 0 := abs_eq_zero.mp (by linarith [abs_nonneg (q0 * k0), abs_nonneg (q1 * k1)])
    exact hz_ne (by simp [z, h00, h11])
  have htB_pos : 0 < Real.tanh B := tanh_pos_of_pos hB_pos
  have hqk0_le_B : |q0 * k0| ≤ B := le_add_of_nonneg_right (abs_nonneg _)
  have hqk1_le_B : |q1 * k1| ≤ B := le_add_of_nonneg_left (abs_nonneg _)
  have hz_le : |z| ≤ A * Real.tanh B := by
    calc |z| ≤ |o0 * v0 * Real.tanh (q0 * k0)| + |o1 * v1 * Real.tanh (q1 * k1)| := abs_add_le _ _
      _ = |o0 * v0| * Real.tanh |q0 * k0| + |o1 * v1| * Real.tanh |q1 * k1| := by
          rw [abs_mul (o0 * v0), abs_mul (o1 * v1), abs_tanh_eq, abs_tanh_eq]
      _ ≤ |o0 * v0| * Real.tanh B + |o1 * v1| * Real.tanh B :=
          add_le_add (mul_le_mul_of_nonneg_left (strictMono_tanh.monotone hqk0_le_B) (abs_nonneg _))
            (mul_le_mul_of_nonneg_left (strictMono_tanh.monotone hqk1_le_B) (abs_nonneg _))
      _ = A * Real.tanh B := by ring
  have ha_le_A : |z| / Real.tanh B ≤ A := (div_le_iff₀ htB_pos).mpr hz_le
  have hsc_ge := hm_min (|z| / Real.tanh B) B (div_nonneg (abs_nonneg _) (le_of_lt htB_pos)) hB_nn
  unfold scalarCost at hsc_ge
  rw [div_mul_cancel₀ _ (ne_of_gt htB_pos)] at hsc_ge
  have hamgm2 : ∀ x y : ℝ, 2 * |x * y| ≤ x ^ 2 + y ^ 2 := fun x y => by
    rw [abs_mul]
    have hsq := sq_nonneg (|x| - |y|)
    have hid : (|x| - |y|) ^ 2 = |x| ^ 2 - 2 * (|x| * |y|) + |y| ^ 2 := by ring
    linarith only [hsq, hid, sq_abs x, sq_abs y]
  have h1 : 2 * |o0 * v0| ≤ o0 ^ 2 + v0 ^ 2 := hamgm2 o0 v0
  have h2 : 2 * |o1 * v1| ≤ o1 ^ 2 + v1 ^ 2 := hamgm2 o1 v1
  have h3 : 2 * |q0 * k0| ≤ q0 ^ 2 + k0 ^ 2 := hamgm2 q0 k0
  have h4 : 2 * |q1 * k1| ≤ q1 ^ 2 + k1 ^ 2 := hamgm2 q1 k1
  have hsq_z : (1 - |z|) ^ 2 ≤ (z - 1) ^ 2 := by
    have hid : (z - 1) ^ 2 - (1 - |z|) ^ 2 = (z ^ 2 - |z| ^ 2) + 2 * (|z| - z) := by ring
    linarith only [hid, le_abs_self z, sq_abs z]
  have hsq_zw : (wj / 2) * (1 - |z|) ^ 2 ≤ (wj / 2) * (z - 1) ^ 2 :=
    mul_le_mul_of_nonneg_left hsq_z (by linarith)
  have hCeq_z : (wj / 2) * (z - 1) ^ 2 +
      ((1 / 100 : ℝ) / 2) * (q0 ^ 2 + k0 ^ 2 + v0 ^ 2 + o0 ^ 2 + q1 ^ 2 + k1 ^ 2 + v1 ^ 2 + o1 ^ 2) = m :=
    hCeq
  have hA_eq : |z| / Real.tanh B = A := by linarith only [ha_le_A, hsc_ge, h1, h2, h3, h4, hsq_zw, hCeq_z]
  have hz_eq : |z| = A * Real.tanh B := (div_eq_iff (ne_of_gt htB_pos)).mp hA_eq
  have heq1 : 2 * |o0 * v0| = o0 ^ 2 + v0 ^ 2 := by linarith only [ha_le_A, hsc_ge, h1, h2, h3, h4, hsq_zw, hCeq_z]
  have heq2 : 2 * |o1 * v1| = o1 ^ 2 + v1 ^ 2 := by linarith only [ha_le_A, hsc_ge, h1, h2, h3, h4, hsq_zw, hCeq_z]
  have heq3 : 2 * |q0 * k0| = q0 ^ 2 + k0 ^ 2 := by linarith only [ha_le_A, hsc_ge, h1, h2, h3, h4, hsq_zw, hCeq_z]
  have heq4 : 2 * |q1 * k1| = q1 ^ 2 + k1 ^ 2 := by linarith only [ha_le_A, hsc_ge, h1, h2, h3, h4, hsq_zw, hCeq_z]
  have htri_eq : |o0 * v0| * Real.tanh |q0 * k0| + |o1 * v1| * Real.tanh |q1 * k1| = A * Real.tanh B := by
    have htri : |z| ≤ |o0 * v0| * Real.tanh |q0 * k0| + |o1 * v1| * Real.tanh |q1 * k1| := by
      calc |z| ≤ |o0 * v0 * Real.tanh (q0 * k0)| + |o1 * v1 * Real.tanh (q1 * k1)| := abs_add_le _ _
        _ = |o0 * v0| * Real.tanh |q0 * k0| + |o1 * v1| * Real.tanh |q1 * k1| := by
            rw [abs_mul (o0 * v0), abs_mul (o1 * v1), abs_tanh_eq, abs_tanh_eq]
    have ht0 := mul_le_mul_of_nonneg_left (strictMono_tanh.monotone hqk0_le_B) (abs_nonneg (o0 * v0))
    have ht1 := mul_le_mul_of_nonneg_left (strictMono_tanh.monotone hqk1_le_B) (abs_nonneg (o1 * v1))
    linarith only [hz_eq, htri, ht0, ht1]
  have hA_pos : 0 < A := by
    have hzp : 0 < |z| := abs_pos.mpr hz_ne
    by_contra hle
    push_neg at hle
    have hnp : A * Real.tanh B ≤ 0 := mul_nonpos_of_nonpos_of_nonneg hle (le_of_lt htB_pos)
    linarith only [hzp, hz_eq, hnp]
  by_cases hov0 : o0 * v0 ≠ 0
  · left
    have hov0_pos : 0 < |o0 * v0| := abs_pos.mpr hov0
    have ht0_eq : Real.tanh |q0 * k0| = Real.tanh B := by
      have ht0_mul := mul_le_mul_of_nonneg_left (strictMono_tanh.monotone hqk0_le_B) (abs_nonneg (o0 * v0))
      have ht1_mul := mul_le_mul_of_nonneg_left (strictMono_tanh.monotone hqk1_le_B) (abs_nonneg (o1 * v1))
      have hmul_eq : |o0 * v0| * Real.tanh |q0 * k0| = |o0 * v0| * Real.tanh B := by
        linarith only [htri_eq, ht0_mul, ht1_mul]
      exact mul_left_cancel₀ (ne_of_gt hov0_pos) hmul_eq
    have hqk0_eq : |q0 * k0| = B := strictMono_tanh.injective ht0_eq
    have hqk1_0 : q1 * k1 = 0 := abs_eq_zero.mp (by dsimp only [B] at hqk0_eq; linarith)
    have hq1_0 : q1 = 0 := by
      rw [hqk1_0, abs_zero, mul_zero] at heq4
      exact sq_eq_zero_iff.mp (by linarith only [heq4, sq_nonneg q1, sq_nonneg k1])
    have hk1_0 : k1 = 0 := by
      rw [hqk1_0, abs_zero, mul_zero] at heq4
      exact sq_eq_zero_iff.mp (by linarith only [heq4, sq_nonneg q1, sq_nonneg k1])
    have hov1_0 : o1 * v1 = 0 := by
      rw [hqk1_0, abs_zero, Real.tanh_zero, mul_zero, add_zero, ht0_eq] at htri_eq
      have : |o1 * v1| * Real.tanh B = 0 := by dsimp only [A] at htri_eq; linarith
      exact abs_eq_zero.mp ((mul_eq_zero.mp this).resolve_right (ne_of_gt htB_pos))
    have ho1_0 : o1 = 0 := by
      rw [hov1_0, abs_zero, mul_zero] at heq2
      exact sq_eq_zero_iff.mp (by linarith only [heq2, sq_nonneg o1, sq_nonneg v1])
    have hv1_0 : v1 = 0 := by
      rw [hov1_0, abs_zero, mul_zero] at heq2
      exact sq_eq_zero_iff.mp (by linarith only [heq2, sq_nonneg o1, sq_nonneg v1])
    have hqk0_ne : q0 * k0 ≠ 0 := abs_pos.mp (by linarith)
    exact ⟨left_ne_zero_of_mul hqk0_ne, right_ne_zero_of_mul hqk0_ne,
           right_ne_zero_of_mul hov0, left_ne_zero_of_mul hov0,
           hq1_0, hk1_0, hv1_0, ho1_0⟩
  · right
    push_neg at hov0
    have hov1 : o1 * v1 ≠ 0 := by
      intro h; dsimp only [A] at hA_pos; rw [hov0, h, abs_zero, add_zero] at hA_pos; linarith
    have hov1_pos : 0 < |o1 * v1| := abs_pos.mpr hov1
    have ht1_eq : Real.tanh |q1 * k1| = Real.tanh B := by
      rw [hov0, abs_zero, zero_mul, zero_add] at htri_eq
      dsimp only [A] at htri_eq
      rw [hov0, abs_zero, zero_add] at htri_eq
      exact mul_left_cancel₀ (ne_of_gt hov1_pos) htri_eq
    have hqk1_eq : |q1 * k1| = B := strictMono_tanh.injective ht1_eq
    have hqk0_0 : q0 * k0 = 0 := abs_eq_zero.mp (by dsimp only [B] at hqk1_eq; linarith)
    have hq0_0 : q0 = 0 := by
      rw [hqk0_0, abs_zero, mul_zero] at heq3
      exact sq_eq_zero_iff.mp (by linarith only [heq3, sq_nonneg q0, sq_nonneg k0])
    have hk0_0 : k0 = 0 := by
      rw [hqk0_0, abs_zero, mul_zero] at heq3
      exact sq_eq_zero_iff.mp (by linarith only [heq3, sq_nonneg q0, sq_nonneg k0])
    have ho0_0 : o0 = 0 := by
      rw [hov0, abs_zero, mul_zero] at heq1
      exact sq_eq_zero_iff.mp (by linarith only [heq1, sq_nonneg o0, sq_nonneg v0])
    have hv0_0 : v0 = 0 := by
      rw [hov0, abs_zero, mul_zero] at heq1
      exact sq_eq_zero_iff.mp (by linarith only [heq1, sq_nonneg o0, sq_nonneg v0])
    have hqk1_ne : q1 * k1 ≠ 0 := abs_pos.mp (by linarith)
    exact ⟨left_ne_zero_of_mul hqk1_ne, right_ne_zero_of_mul hqk1_ne,
           right_ne_zero_of_mul hov1, left_ne_zero_of_mul hov1,
           hq0_0, hk0_0, hv0_0, ho0_0⟩

/-- **Theorem 2(1c) — Every global minimizer has `q₀, q₁` nonzero and mutually orthogonal (`R = 0`)**:
For `λ = 1/100`, `w 0 ≥ 1`, `w 1 ≥ 1`, and global scalar minima `m₀, m₁` (strictly `< 3/100` by
`trial_scalarCost_lt`), ANY `θ` attaining `objF w yOrth (1/100) θ = m₀ + m₁` has both query
vectors nonzero and orthogonal, hence `overlap2 (θ.q 0) (θ.q 1) = 0 < 1 / 2`. -/
theorem orth_every_global_min_R_zero {w : Fin 2 → ℝ} (hw0 : 1 ≤ w 0) (hw1 : 1 ≤ w 1)
    {m0 m1 : ℝ}
    (hm0_le : ∀ a b : ℝ, 0 ≤ a → 0 ≤ b → m0 ≤ scalarCost (w 0) (1 / 100) a b)
    (hm1_le : ∀ a b : ℝ, 0 ≤ a → 0 ≤ b → m1 ≤ scalarCost (w 1) (1 / 100) a b)
    (θ : AttnParams) (hmin : objF w yOrth (1 / 100) θ = m0 + m1) :
    θ.q 0 ≠ 0 ∧ θ.q 1 ≠ 0 ∧
    θ.q 0 0 * θ.q 1 0 + θ.q 0 1 * θ.q 1 1 = 0 ∧
    overlap2 (θ.q 0) (θ.q 1) = 0 := by
  have hm0_lt : m0 < 3 / 100 :=
    lt_of_le_of_lt (hm0_le (Real.tanh 1)⁻¹ 1 (inv_nonneg.mpr (by linarith [half_lt_tanh_one]))
      (by norm_num)) (trial_scalarCost_lt (w 0))
  have hm1_lt : m1 < 3 / 100 :=
    lt_of_le_of_lt (hm1_le (Real.tanh 1)⁻¹ 1 (inv_nonneg.mpr (by linarith [half_lt_tanh_one]))
      (by norm_num)) (trial_scalarCost_lt (w 1))
  -- Apply orth_coord_bound on coordinates 0 and 1
  obtain ⟨a₀, b₀, ha₀, hb₀, ha₀_le, hb₀_eq, hbd0⟩ := orth_coord_bound (w 0) (1 / 100) (by linarith)
    (by norm_num) (θ.q 0 0) (θ.k 0 0) (θ.v 0 0) (θ.o 0 0) (θ.q 1 0) (θ.k 1 0) (θ.v 1 0) (θ.o 1 0)
  obtain ⟨a₁, b₁, ha₁, hb₁, ha₁_le, hb₁_eq, hbd1⟩ := orth_coord_bound (w 1) (1 / 100) (by linarith)
    (by norm_num) (θ.q 0 1) (θ.k 0 1) (θ.v 0 1) (θ.o 0 1) (θ.q 1 1) (θ.k 1 1) (θ.v 1 1) (θ.o 1 1)
  have hm0_ge := hm0_le a₀ b₀ ha₀ hb₀
  have hm1_ge := hm1_le a₁ b₁ ha₁ hb₁
  -- Expand objF to isolate the two coordinate contributions plus the 6 discarded squared residuals
  set z0 := θ.o 0 0 * θ.v 0 0 * Real.tanh (θ.q 0 0 * θ.k 0 0) +
            θ.o 1 0 * θ.v 1 0 * Real.tanh (θ.q 1 0 * θ.k 1 0)
  set z1 := θ.o 0 1 * θ.v 0 1 * Real.tanh (θ.q 0 1 * θ.k 0 1) +
            θ.o 1 1 * θ.v 1 1 * Real.tanh (θ.q 1 1 * θ.k 1 1)
  set r001 := θ.o 0 1 * θ.v 0 0 * Real.tanh (θ.q 0 0 * θ.k 0 0) +
              θ.o 1 1 * θ.v 1 0 * Real.tanh (θ.q 1 0 * θ.k 1 0)
  set C0 := (w 0 / 2) * (z0 - 1) ^ 2 +
      ((1 / 100 : ℝ) / 2) * (θ.q 0 0 ^ 2 + θ.k 0 0 ^ 2 + θ.v 0 0 ^ 2 + θ.o 0 0 ^ 2 +
                             θ.q 1 0 ^ 2 + θ.k 1 0 ^ 2 + θ.v 1 0 ^ 2 + θ.o 1 0 ^ 2)
  set C1 := (w 1 / 2) * (z1 - 1) ^ 2 +
      ((1 / 100 : ℝ) / 2) * (θ.q 0 1 ^ 2 + θ.k 0 1 ^ 2 + θ.v 0 1 ^ 2 + θ.o 0 1 ^ 2 +
                             θ.q 1 1 ^ 2 + θ.k 1 1 ^ 2 + θ.v 1 1 ^ 2 + θ.o 1 1 ^ 2)
  have hobj_ge : C0 + C1 + (w 0 / 2) * r001 ^ 2 ≤ objF w yOrth (1 / 100) θ := by
    unfold objF
    simp only [Fin.sum_univ_two, attnLayer_basis, yOrth, show (0 : Fin 2) ≠ 1 by decide,
      show (1 : Fin 2) ≠ 0 by decide, and_self, and_false, false_and, ↓reduceIte, sub_zero,
      mul_add]
    simp only [show ∀ u v : ℝ, (1 / 2 : ℝ) * (u * v) = (u / 2) * v from fun u v => by ring]
    have hw02 : 0 ≤ w 0 / 2 := by linarith
    have hw12 : 0 ≤ w 1 / 2 := by linarith
    have hs2 := mul_nonneg hw02 (sq_nonneg (θ.o 0 0 * θ.v 0 1 * Real.tanh (θ.q 0 0 * θ.k 0 1) +
        θ.o 1 0 * θ.v 1 1 * Real.tanh (θ.q 1 0 * θ.k 1 1)))
    have hs3 := mul_nonneg hw02 (sq_nonneg (θ.o 0 1 * θ.v 0 1 * Real.tanh (θ.q 0 0 * θ.k 0 1) +
        θ.o 1 1 * θ.v 1 1 * Real.tanh (θ.q 1 0 * θ.k 1 1)))
    have hs4 := mul_nonneg hw12 (sq_nonneg (θ.o 0 0 * θ.v 0 0 * Real.tanh (θ.q 0 1 * θ.k 0 0) +
        θ.o 1 0 * θ.v 1 0 * Real.tanh (θ.q 1 1 * θ.k 1 0)))
    have hs5 := mul_nonneg hw12 (sq_nonneg (θ.o 0 1 * θ.v 0 0 * Real.tanh (θ.q 0 1 * θ.k 0 0) +
        θ.o 1 1 * θ.v 1 0 * Real.tanh (θ.q 1 1 * θ.k 1 0)))
    have hs6 := mul_nonneg hw12 (sq_nonneg (θ.o 0 0 * θ.v 0 1 * Real.tanh (θ.q 0 1 * θ.k 0 1) +
        θ.o 1 0 * θ.v 1 1 * Real.tanh (θ.q 1 1 * θ.k 1 1)))
    dsimp only [C0, C1, z0, z1, r001]
    simp only [mul_add]
    linarith
  -- Force equality in both coordinates and r001 = 0
  have hC0_ge : m0 ≤ C0 := le_trans hm0_ge hbd0
  have hC1_ge : m1 ≤ C1 := le_trans hm1_ge hbd1
  have hr001_nn : 0 ≤ (w 0 / 2) * r001 ^ 2 := mul_nonneg (by linarith) (sq_nonneg r001)
  have hC0_eq : C0 = m0 := by linarith only [hC0_ge, hC1_ge, hr001_nn, hobj_ge, hmin]
  have hC1_eq : C1 = m1 := by linarith only [hC0_ge, hC1_ge, hr001_nn, hobj_ge, hmin]
  have hr001 : r001 = 0 := by
    have hmul0 : (w 0 / 2) * r001 ^ 2 = 0 := by
      linarith only [hC0_ge, hC1_ge, hr001_nn, hobj_ge, hmin]
    have hw0_pos : (w 0 / 2) ≠ 0 := by linarith
    exact sq_eq_zero_iff.mp ((mul_eq_zero.mp hmul0).resolve_left hw0_pos)
  have hspec0 := orth_coord_spec (w 0) m0 (θ.q 0 0) (θ.k 0 0) (θ.v 0 0) (θ.o 0 0)
    (θ.q 1 0) (θ.k 1 0) (θ.v 1 0) (θ.o 1 0) hw0 hm0_lt hm0_le hC0_eq
  have hspec1 := orth_coord_spec (w 1) m1 (θ.q 0 1) (θ.k 0 1) (θ.v 0 1) (θ.o 0 1)
    (θ.q 1 1) (θ.k 1 1) (θ.v 1 1) (θ.o 1 1) hw1 hm1_lt hm1_le hC1_eq
  -- Eliminate the two same-head cases using r001 = 0
  rcases hspec0 with ⟨hq00, hk00, hv00, _, hq10, hk10, hv10, _⟩ | ⟨hq10, hk10, hv10, _, hq00, hk00, hv00, _⟩
  · rcases hspec1 with ⟨_, _, _, ho01, _, _, _, ho11⟩ | ⟨hq11, _, _, _, hq01, _, _, _⟩
    · -- Head 0 carries both coordinates → r001 = o01 * v00 * tanh(q00 * k00) ≠ 0, contradiction!
      exfalso
      dsimp only [r001] at hr001
      rw [ho11, zero_mul, zero_mul, add_zero] at hr001
      have htne : Real.tanh (θ.q 0 0 * θ.k 0 0) ≠ 0 := by
        intro ht
        have := strictMono_tanh.injective (ht.trans Real.tanh_zero.symm)
        exact mul_ne_zero hq00 hk00 this
      exact mul_ne_zero (mul_ne_zero ho01 hv00) htne hr001
    · -- Head 0 carries coord 0, Head 1 carries coord 1!
      have horth : θ.q 0 0 * θ.q 1 0 + θ.q 0 1 * θ.q 1 1 = 0 := by rw [hq10, hq01]; ring
      exact ⟨fun h => hq00 (congrFun h 0), fun h => hq11 (congrFun h 1),
             horth, overlap2_orth_zero horth⟩
  · rcases hspec1 with ⟨hq01, _, _, _, hq11, _, _, _⟩ | ⟨_, _, _, ho11, _, _, _, ho01⟩
    · -- Head 1 carries coord 0, Head 0 carries coord 1!
      have horth : θ.q 0 0 * θ.q 1 0 + θ.q 0 1 * θ.q 1 1 = 0 := by rw [hq00, hq11]; ring
      exact ⟨fun h => hq01 (congrFun h 1), fun h => hq10 (congrFun h 0),
             horth, overlap2_orth_zero horth⟩
    · -- Head 1 carries both coordinates → r001 = o11 * v10 * tanh(q10 * k10) ≠ 0, contradiction!
      exfalso
      dsimp only [r001] at hr001
      rw [ho01, zero_mul, zero_mul, zero_add] at hr001
      have htne : Real.tanh (θ.q 1 0 * θ.k 1 0) ≠ 0 := by
        intro ht
        have := strictMono_tanh.injective (ht.trans Real.tanh_zero.symm)
        exact mul_ne_zero hq10 hk10 this
      exact mul_ne_zero (mul_ne_zero ho11 hv10) htne hr001

/-! ### 2(b). Theorem 2(2): Stationarity Cubic & Strict Local Convexity Certificate -/

/-- Stationarity cubic `p(s) = 2s³ - δs² - s + δ` for the reduced 1D objective (`s = tanh b`). -/
def stationarityCubic (δ s : ℝ) : ℝ :=
  2 * s ^ 3 - δ * s ^ 2 - s + δ

/-- Derivative `p'(s) = 6s² - 2δs - 1` of the stationarity cubic. -/
def stationarityCubicDeriv (δ s : ℝ) : ℝ :=
  6 * s ^ 2 - 2 * δ * s - 1

/-- **Theorem 2(2a) — Existence of the active root in `(1/2, 1)` with strictly positive curvature**:
For every `0 < δ < 1 / 4` (covering `δ = 1/100` and `δ = 1/200`), there exists `s* ∈ (1/2, 1)`
with `stationarityCubic δ s* = 0`, and EVERY root `s ∈ (1/2, 1)` satisfies
`stationarityCubicDeriv δ s > 0` (strictly positive reduced second derivative in `b`). -/
theorem stationarityCubic_min_root_pos_deriv {δ : ℝ} (hδ0 : 0 < δ) (hδ1 : δ < 1 / 4) :
    (∃ s ∈ Set.Ioo (1 / 2 : ℝ) 1, stationarityCubic δ s = 0) ∧
    ∀ s ∈ Set.Ioo (1 / 2 : ℝ) 1, 0 < stationarityCubicDeriv δ s := by
  constructor
  · have hcont : ContinuousOn (stationarityCubic δ) (Set.Icc (1 / 2 : ℝ) 1) := by
      unfold stationarityCubic; fun_prop
    have hlo : stationarityCubic δ (1 / 2) < 0 := by
      unfold stationarityCubic; nlinarith
    have hhi : 0 < stationarityCubic δ 1 := by
      unfold stationarityCubic; nlinarith
    have h0mem : (0 : ℝ) ∈ Set.Icc (stationarityCubic δ (1 / 2)) (stationarityCubic δ 1) :=
      ⟨le_of_lt hlo, le_of_lt hhi⟩
    obtain ⟨s, hsIcc, hseq⟩ := intermediate_value_Icc (by norm_num : (1 / 2 : ℝ) ≤ 1) hcont h0mem
    refine ⟨s, ⟨?_, ?_⟩, hseq⟩
    · exact lt_of_le_of_ne hsIcc.1 (fun h => by rw [← h] at hseq; linarith)
    · exact lt_of_le_of_ne hsIcc.2 (fun h => by rw [h] at hseq; linarith)
  · intro s ⟨hs1, hs2⟩
    unfold stationarityCubicDeriv
    nlinarith

/-- Continuity of `Real.tanh`. -/
lemma continuous_tanh : Continuous Real.tanh := by
  have hfn : Real.tanh = fun t => Real.sinh t / Real.cosh t := funext Real.tanh_eq_sinh_div_cosh
  rw [hfn]
  exact Real.continuous_sinh.div Real.continuous_cosh (fun t => ne_of_gt (Real.cosh_pos t))

/-- Derivative of `Real.tanh`: `HasDerivAt Real.tanh (1 - (Real.tanh x)²) x`. -/
lemma hasDerivAt_tanh (x : ℝ) : HasDerivAt Real.tanh (1 - (Real.tanh x) ^ 2) x := by
  have hfn : Real.tanh = fun t => Real.sinh t / Real.cosh t := funext Real.tanh_eq_sinh_div_cosh
  rw [hfn]
  have hch : Real.cosh x ≠ 0 := ne_of_gt (Real.cosh_pos x)
  have hd := (Real.hasDerivAt_sinh x).div (Real.hasDerivAt_cosh x) hch
  convert hd using 1
  field_simp [hch]

/-- **Theorem 2(2b) — Exact derivative of the reduced cost `b ↦ scalarCost w lam (a(b)) b`**:
With `δ = lam / w` and `a(b) = (tanh b - δ) / (tanh b)²`, the derivative with respect to `b`
equals `lam * stationarityCubic δ (tanh b) / (tanh b)³`. -/
theorem reduced_cost_hasDerivAt {w lam b : ℝ} (hw : 0 < w) (hb : 0 < b) :
    let δ := lam / w
    let red : ℝ → ℝ := fun t => scalarCost w lam ((Real.tanh t - δ) / (Real.tanh t) ^ 2) t
    HasDerivAt red (lam * stationarityCubic δ (Real.tanh b) / (Real.tanh b) ^ 3) b := by
  intro δ red
  have hs_pos : 0 < Real.tanh b := tanh_pos_of_pos hb
  have hs_ne : Real.tanh b ≠ 0 := ne_of_gt hs_pos
  have hrew : ∀ᶠ t in nhds b, red t =
      lam * (1 / Real.tanh t - δ / (2 * Real.tanh t ^ 2) + t) := by
    filter_upwards [continuous_tanh.continuousAt.eventually (Ioi_mem_nhds hs_pos)] with t ht
    have htn : Real.tanh t ≠ 0 := ne_of_gt ht
    dsimp only [red, scalarCost, δ]
    field_simp [ne_of_gt hw, htn]
    ring
  have hbase : HasDerivAt (fun t => lam * (1 / Real.tanh t - δ / (2 * Real.tanh t ^ 2) + t))
      (lam * stationarityCubic δ (Real.tanh b) / (Real.tanh b) ^ 3) b := by
    have ht := hasDerivAt_tanh b
    have h1 := (hasDerivAt_const b (1 : ℝ)).div ht hs_ne
    have hden2 : HasDerivAt (fun t => 2 * (Real.tanh t) ^ 2)
        (2 * (2 * Real.tanh b ^ (2 - 1) * (1 - Real.tanh b ^ 2))) b :=
      (ht.pow 2).const_mul 2
    have h2 := (hasDerivAt_const b δ).div hden2 (mul_ne_zero (by norm_num) (pow_ne_zero 2 hs_ne))
    have hsum := ((h1.sub h2).add (hasDerivAt_id' b)).const_mul lam
    convert hsum using 1
    unfold stationarityCubic
    field_simp [hs_ne]
    ring
  exact hbase.congr_of_eventuallyEq hrew

/-! ### 3. Proposition 3: Shared Query Direction (`R = 1 > 1/2`) on Identical Anisotropic Inputs -/

/-- **Proposition 3 (Shared query direction on identical anisotropic inputs)**:
Keep `w = ![2, 1]` and `λ = 1/100`, with shared-query targets `yShared l j c = 1_{l=0 ∧ c=j}`.
1. The explicit shared-query configuration `q₀ = q₁ = e₀`, `k_j = e_j`, `v_j = o_j = (√(tanh 1))⁻¹ e_j`
   achieves `objF < 1`;
2. Every global minimizer `θ` of `objF w yShared (1/100)` has `q₀, q₁` nonzero and collinear with
   `e₀` (`q₀ 1 = 0, q₁ 1 = 0, q₀ 0 ≠ 0, q₁ 0 ≠ 0`), forcing `overlap2 (θ.q 0) (θ.q 1) = 1 > 1 / 2`. -/
theorem shared_every_global_min_R_one
    (θ : AttnParams)
    (hmin : ∀ θ' : AttnParams, objF ![2, 1] yShared (1 / 100) θ ≤ objF ![2, 1] yShared (1 / 100) θ') :
    θ.q 0 0 ≠ 0 ∧ θ.q 0 1 = 0 ∧ θ.q 1 0 ≠ 0 ∧ θ.q 1 1 = 0 ∧
    overlap2 (θ.q 0) (θ.q 1) = 1 := by
  have ht1 : (1 / 2 : ℝ) < Real.tanh 1 := half_lt_tanh_one
  have ht1_pos : 0 < Real.tanh 1 := by linarith
  -- Construct witness θ_w with objF < 1
  let r := Real.sqrt (Real.tanh 1)⁻¹
  let θw : AttnParams :=
    { q := ![![1, 0], ![1, 0]]
      k := ![![1, 0], ![0, 1]]
      v := ![![r, 0], ![0, r]]
      o := ![![r, 0], ![0, r]] }
  have hr2 : r ^ 2 = (Real.tanh 1)⁻¹ := Real.sq_sqrt (inv_nonneg.mpr (le_of_lt ht1_pos))
  have hr_mul : r * r = (Real.tanh 1)⁻¹ := Real.mul_self_sqrt (inv_nonneg.mpr (le_of_lt ht1_pos))
  have hθw_val : objF ![2, 1] yShared (1 / 100) θw = (2 / 100) * (1 + (Real.tanh 1)⁻¹) := by
    unfold objF
    simp only [Fin.sum_univ_two, attnLayer_basis, yShared, θw,
      Matrix.cons_val_zero, Matrix.cons_val_one, show (0 : Fin 2) ≠ 1 by decide,
      show (1 : Fin 2) ≠ 0 by decide, and_self, and_false, false_and, ↓reduceIte,
      mul_zero, zero_mul, Real.tanh_zero, add_zero, zero_add, mul_one,
      one_pow, zero_pow (by norm_num : 2 ≠ 0), hr_mul, hr2, inv_mul_cancel₀ (ne_of_gt ht1_pos),
      sub_self]
    ring
  have hθw_lt1 : objF ![2, 1] yShared (1 / 100) θw < 1 := by
    rw [hθw_val]
    have hinv_lt : (Real.tanh 1)⁻¹ < 2 := by
      rw [inv_lt_comm₀ ht1_pos (by norm_num : (0 : ℝ) < 2)]; linarith
    linarith
  have hθ_lt1 : objF ![2, 1] yShared (1 / 100) θ < 1 := lt_of_le_of_lt (hmin θw) hθw_lt1
  -- Step 1: Zeroing q 0 1 and q 1 1 preserves l = 0 predictions, zeros l = 1 residuals, and
  -- strictly decreases weight decay unless q 0 1 = 0 and q 1 1 = 0.
  let θ0 : AttnParams :=
    { q := fun i j => if j = 0 then θ.q i 0 else 0
      k := θ.k, v := θ.v, o := θ.o }
  have hcmp := hmin θ0
  have hdiff : objF ![2, 1] yShared (1 / 100) θ0 +
      ((1 / 100 : ℝ) / 2) * (θ.q 0 1 ^ 2 + θ.q 1 1 ^ 2) ≤ objF ![2, 1] yShared (1 / 100) θ := by
    unfold objF
    simp only [Fin.sum_univ_two, attnLayer_basis, yShared, θ0,
      Matrix.cons_val_zero, Matrix.cons_val_one, show (0 : Fin 2) ≠ 1 by decide,
      show (1 : Fin 2) ≠ 0 by decide, and_self, and_false, false_and, ↓reduceIte,
      mul_zero, zero_mul, Real.tanh_zero, add_zero, zero_add, sub_zero,
      zero_pow (by norm_num : 2 ≠ 0)]
    linarith [sq_nonneg (θ.o 0 0 * θ.v 0 0 * Real.tanh (θ.q 0 1 * θ.k 0 0) +
                  θ.o 1 0 * θ.v 1 0 * Real.tanh (θ.q 1 1 * θ.k 1 0)),
              sq_nonneg (θ.o 0 1 * θ.v 0 0 * Real.tanh (θ.q 0 1 * θ.k 0 0) +
                  θ.o 1 1 * θ.v 1 0 * Real.tanh (θ.q 1 1 * θ.k 1 0)),
              sq_nonneg (θ.o 0 0 * θ.v 0 1 * Real.tanh (θ.q 0 1 * θ.k 0 1) +
                  θ.o 1 0 * θ.v 1 1 * Real.tanh (θ.q 1 1 * θ.k 1 1)),
              sq_nonneg (θ.o 0 1 * θ.v 0 1 * Real.tanh (θ.q 0 1 * θ.k 0 1) +
                  θ.o 1 1 * θ.v 1 1 * Real.tanh (θ.q 1 1 * θ.k 1 1))]
  have hq01 : θ.q 0 1 = 0 := by nlinarith [sq_nonneg (θ.q 0 1), sq_nonneg (θ.q 1 1)]
  have hq11 : θ.q 1 1 = 0 := by nlinarith [sq_nonneg (θ.q 0 1), sq_nonneg (θ.q 1 1)]
  -- Step 2: 2D rank-1 approximation lower bound: for any u ∈ ℝ² and scalars α, β,
  -- (α u₀ - 1)² + (α u₁)² + (β u₀)² + (β u₁ - 1)² ≥ 1!
  have hrank1_lb : ∀ u0 u1 α β : ℝ,
      1 ≤ (α * u0 - 1) ^ 2 + (α * u1) ^ 2 + (β * u0) ^ 2 + (β * u1 - 1) ^ 2 := by
    intro u0 u1 α β
    by_cases hu : u0 ^ 2 + u1 ^ 2 = 0
    · have hu0 : u0 = 0 := by nlinarith [sq_nonneg u0, sq_nonneg u1]
      have hu1 : u1 = 0 := by nlinarith [sq_nonneg u0, sq_nonneg u1]
      rw [hu0, hu1]; norm_num
    · have hupos : 0 < u0 ^ 2 + u1 ^ 2 := lt_of_le_of_ne (by positivity) (Ne.symm hu)
      -- Multiply by (u0² + u1²) and write as a sum of squares!
      have hid : (u0 ^ 2 + u1 ^ 2) *
          ((α * u0 - 1) ^ 2 + (α * u1) ^ 2 + (β * u0) ^ 2 + (β * u1 - 1) ^ 2 - 1) =
          (α * (u0 ^ 2 + u1 ^ 2) - u0) ^ 2 + (β * (u0 ^ 2 + u1 ^ 2) - u1) ^ 2 := by ring
      nlinarith [sq_nonneg (α * (u0 ^ 2 + u1 ^ 2) - u0),
                 sq_nonneg (β * (u0 ^ 2 + u1 ^ 2) - u1)]
  have hq00 : θ.q 0 0 ≠ 0 := by
    intro h0
    have hlb : 1 ≤ objF ![2, 1] yShared (1 / 100) θ := by
      unfold objF
      simp only [Fin.sum_univ_two, attnLayer_basis, yShared, h0, zero_mul, Real.tanh_zero,
        mul_zero, zero_add, Matrix.cons_val_zero, Matrix.cons_val_one,
        show (0 : Fin 2) ≠ 1 by decide, show (1 : Fin 2) ≠ 0 by decide, and_self, and_false,
        false_and, ↓reduceIte, sub_zero]
      have hr1 := hrank1_lb (θ.o 1 0) (θ.o 1 1)
        (θ.v 1 0 * Real.tanh (θ.q 1 0 * θ.k 1 0))
        (θ.v 1 1 * Real.tanh (θ.q 1 0 * θ.k 1 1))
      have hcomm :
          (θ.o 1 0 * θ.v 1 0 * Real.tanh (θ.q 1 0 * θ.k 1 0) - 1) ^ 2 +
          (θ.o 1 1 * θ.v 1 0 * Real.tanh (θ.q 1 0 * θ.k 1 0)) ^ 2 +
          ((θ.o 1 0 * θ.v 1 1 * Real.tanh (θ.q 1 0 * θ.k 1 1)) ^ 2 +
          (θ.o 1 1 * θ.v 1 1 * Real.tanh (θ.q 1 0 * θ.k 1 1) - 1) ^ 2) =
          (θ.v 1 0 * Real.tanh (θ.q 1 0 * θ.k 1 0) * θ.o 1 0 - 1) ^ 2 +
          (θ.v 1 0 * Real.tanh (θ.q 1 0 * θ.k 1 0) * θ.o 1 1) ^ 2 +
          (θ.v 1 1 * Real.tanh (θ.q 1 0 * θ.k 1 1) * θ.o 1 0) ^ 2 +
          (θ.v 1 1 * Real.tanh (θ.q 1 0 * θ.k 1 1) * θ.o 1 1 - 1) ^ 2 := by ring
      linarith [sq_nonneg (θ.o 0 0 * θ.v 0 0 * Real.tanh (θ.q 0 1 * θ.k 0 0) +
                    θ.o 1 0 * θ.v 1 0 * Real.tanh (θ.q 1 1 * θ.k 1 0)),
                sq_nonneg (θ.o 0 1 * θ.v 0 0 * Real.tanh (θ.q 0 1 * θ.k 0 0) +
                    θ.o 1 1 * θ.v 1 0 * Real.tanh (θ.q 1 1 * θ.k 1 0)),
                sq_nonneg (θ.o 0 0 * θ.v 0 1 * Real.tanh (θ.q 0 1 * θ.k 0 1) +
                    θ.o 1 0 * θ.v 1 1 * Real.tanh (θ.q 1 1 * θ.k 1 1)),
                sq_nonneg (θ.o 0 1 * θ.v 0 1 * Real.tanh (θ.q 0 1 * θ.k 0 1) +
                    θ.o 1 1 * θ.v 1 1 * Real.tanh (θ.q 1 1 * θ.k 1 1)),
                sq_nonneg (θ.k 0 0), sq_nonneg (θ.v 0 0), sq_nonneg (θ.o 0 0),
                sq_nonneg (θ.q 0 1), sq_nonneg (θ.k 0 1), sq_nonneg (θ.v 0 1), sq_nonneg (θ.o 0 1),
                sq_nonneg (θ.q 1 0), sq_nonneg (θ.k 1 0), sq_nonneg (θ.v 1 0), sq_nonneg (θ.o 1 0),
                sq_nonneg (θ.q 1 1), sq_nonneg (θ.k 1 1), sq_nonneg (θ.v 1 1), sq_nonneg (θ.o 1 1)]
    linarith
  have hq10 : θ.q 1 0 ≠ 0 := by
    intro h0
    have hlb : 1 ≤ objF ![2, 1] yShared (1 / 100) θ := by
      unfold objF
      simp only [Fin.sum_univ_two, attnLayer_basis, yShared, h0, zero_mul, Real.tanh_zero,
        mul_zero, add_zero, Matrix.cons_val_zero, Matrix.cons_val_one,
        show (0 : Fin 2) ≠ 1 by decide, show (1 : Fin 2) ≠ 0 by decide, and_self, and_false,
        false_and, ↓reduceIte, sub_zero]
      have hr1 := hrank1_lb (θ.o 0 0) (θ.o 0 1)
        (θ.v 0 0 * Real.tanh (θ.q 0 0 * θ.k 0 0))
        (θ.v 0 1 * Real.tanh (θ.q 0 0 * θ.k 0 1))
      have hcomm :
          (θ.o 0 0 * θ.v 0 0 * Real.tanh (θ.q 0 0 * θ.k 0 0) - 1) ^ 2 +
          (θ.o 0 1 * θ.v 0 0 * Real.tanh (θ.q 0 0 * θ.k 0 0)) ^ 2 +
          ((θ.o 0 0 * θ.v 0 1 * Real.tanh (θ.q 0 0 * θ.k 0 1)) ^ 2 +
          (θ.o 0 1 * θ.v 0 1 * Real.tanh (θ.q 0 0 * θ.k 0 1) - 1) ^ 2) =
          (θ.v 0 0 * Real.tanh (θ.q 0 0 * θ.k 0 0) * θ.o 0 0 - 1) ^ 2 +
          (θ.v 0 0 * Real.tanh (θ.q 0 0 * θ.k 0 0) * θ.o 0 1) ^ 2 +
          (θ.v 0 1 * Real.tanh (θ.q 0 0 * θ.k 0 1) * θ.o 0 0) ^ 2 +
          (θ.v 0 1 * Real.tanh (θ.q 0 0 * θ.k 0 1) * θ.o 0 1 - 1) ^ 2 := by ring
      linarith [sq_nonneg (θ.o 0 0 * θ.v 0 0 * Real.tanh (θ.q 0 1 * θ.k 0 0) +
                    θ.o 1 0 * θ.v 1 0 * Real.tanh (θ.q 1 1 * θ.k 1 0)),
                sq_nonneg (θ.o 0 1 * θ.v 0 0 * Real.tanh (θ.q 0 1 * θ.k 0 0) +
                    θ.o 1 1 * θ.v 1 0 * Real.tanh (θ.q 1 1 * θ.k 1 0)),
                sq_nonneg (θ.o 0 0 * θ.v 0 1 * Real.tanh (θ.q 0 1 * θ.k 0 1) +
                    θ.o 1 0 * θ.v 1 1 * Real.tanh (θ.q 1 1 * θ.k 1 1)),
                sq_nonneg (θ.o 0 1 * θ.v 0 1 * Real.tanh (θ.q 0 1 * θ.k 0 1) +
                    θ.o 1 1 * θ.v 1 1 * Real.tanh (θ.q 1 1 * θ.k 1 1)),
                sq_nonneg (θ.q 0 0), sq_nonneg (θ.k 0 0), sq_nonneg (θ.v 0 0), sq_nonneg (θ.o 0 0),
                sq_nonneg (θ.q 0 1), sq_nonneg (θ.k 0 1), sq_nonneg (θ.v 0 1), sq_nonneg (θ.o 0 1),
                sq_nonneg (θ.k 1 0), sq_nonneg (θ.v 1 0), sq_nonneg (θ.o 1 0),
                sq_nonneg (θ.q 1 1), sq_nonneg (θ.k 1 1), sq_nonneg (θ.v 1 1), sq_nonneg (θ.o 1 1)]
    linarith
  have hover : overlap2 (θ.q 0) (θ.q 1) = 1 := by
    have h0vec : θ.q 0 = ![θ.q 0 0, 0] := by ext j; fin_cases j <;> simp [hq01]
    have h1vec : θ.q 1 = ![θ.q 1 0, 0] := by ext j; fin_cases j <;> simp [hq11]
    rw [h0vec, h1vec]
    exact overlap2_collinear_e0 hq00 hq10
  exact ⟨hq00, hq01, hq10, hq11, hover⟩

/-! ### 4. Proposition 4: Negative Initial Dynamical Drift `dR/dt|_{t=0} < 0` -/

/-- Tilted initialization around the isotropic orthogonal minimizer (`c = √b₁`, `a₁ > 0`),
parameterized by the four query entries `(q00, q01, q10, q11)` while keeping `k, v, o` at the
minimizer. -/
noncomputable def tiltedParams (a₁ c q00 q01 q10 q11 : ℝ) : AttnParams where
  q := ![![q00, q01], ![q10, q11]]
  k := ![![c, 0], ![0, c]]
  v := ![![Real.sqrt a₁, 0], ![0, Real.sqrt a₁]]
  o := ![![Real.sqrt a₁, 0], ![0, Real.sqrt a₁]]

/-- **Proposition 4(a) — Exact partial derivatives of the full 16-parameter objective at the tilted
configuration `(q00 = c, q01 = t, q10 = 0, q11 = c)`**:
Assuming stationarity of `(a₁, c²)` (`w₀ (a₁ tanh(c²) - 1) a₁ (1 - tanh²(c²)) + lam = 0`):
- `∂F/∂q00 = 0`, `∂F/∂q10 = 0`, `∂F/∂q11 = 0`, and
- `∂F/∂q01 = lam * t + w₁ * a₁² * c * tanh(c t) * (1 - tanh²(c t)) > 0`. -/
theorem tilted_gradient_exact {w0 w1 lam a₁ c t : ℝ}
    (hw1 : 0 < w1) (hlam : 0 < lam) (ha₁ : 0 < a₁) (hc : 0 < c) (ht : 0 < t)
    (hstat0 : w0 * (a₁ * Real.tanh (c ^ 2) - 1) * a₁ * (1 - Real.tanh (c ^ 2) ^ 2) + lam = 0)
    (hstat1 : w1 * (a₁ * Real.tanh (c ^ 2) - 1) * a₁ * (1 - Real.tanh (c ^ 2) ^ 2) + lam = 0) :
    let Fq := fun q00 q01 q10 q11 => objF ![w0, w1] yOrth lam (tiltedParams a₁ c q00 q01 q10 q11)
    let g01 := lam * t + w1 * a₁ ^ 2 * c * Real.tanh (c * t) * (1 - Real.tanh (c * t) ^ 2)
    HasDerivAt (fun x => Fq x t 0 c) 0 c ∧
    HasDerivAt (fun x => Fq c x 0 c) g01 t ∧
    HasDerivAt (fun x => Fq c t x c) 0 0 ∧
    HasDerivAt (fun x => Fq c t 0 x) 0 c ∧
    0 < g01 := by
  intro Fq g01
  have ha_mul : Real.sqrt a₁ * Real.sqrt a₁ = a₁ := Real.mul_self_sqrt (le_of_lt ha₁)
  have ha_sq : (Real.sqrt a₁) ^ 2 = a₁ := Real.sq_sqrt (le_of_lt ha₁)
  -- Explicit closed form of Fq q00 q01 q10 q11
  have hFq : ∀ q00 q01 q10 q11 : ℝ,
      Fq q00 q01 q10 q11 =
        (w0 / 2) * ((a₁ * Real.tanh (q00 * c) - 1) ^ 2 + (a₁ * Real.tanh (q10 * c)) ^ 2) +
        (w1 / 2) * ((a₁ * Real.tanh (q01 * c)) ^ 2 + (a₁ * Real.tanh (q11 * c) - 1) ^ 2) +
        (lam / 2) * (q00 ^ 2 + q01 ^ 2 + q10 ^ 2 + q11 ^ 2 + 2 * c ^ 2 + 4 * a₁) := by
    intro q00 q01 q10 q11
    dsimp only [Fq]
    unfold objF
    simp only [Fin.sum_univ_two, attnLayer_basis, yOrth, tiltedParams,
      Matrix.cons_val_zero, Matrix.cons_val_one, show (0 : Fin 2) ≠ 1 by decide,
      show (1 : Fin 2) ≠ 0 by decide, and_self, and_false, false_and, ↓reduceIte,
      mul_zero, zero_mul, Real.tanh_zero, add_zero, zero_add, sub_zero,
      zero_pow (by norm_num : 2 ≠ 0), ha_mul, ha_sq]
    ring
  have hd_tanh_c : ∀ x : ℝ, HasDerivAt (fun u => Real.tanh (u * c)) ((1 - Real.tanh (x * c) ^ 2) * c) x := by
    intro x
    have hlin : HasDerivAt (fun u => u * c) c x := by simpa using (hasDerivAt_id' x).mul_const c
    exact (hasDerivAt_tanh (x * c)).comp x hlin
  have hd_quad : ∀ x₀ : ℝ, HasDerivAt (fun x : ℝ => x ^ 2) (2 * x₀) x₀ := fun x₀ => by
    simpa using (hasDerivAt_id' x₀).pow 2
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · -- ∂F/∂q00 at c
    simp_rw [hFq]
    have hd1 : HasDerivAt (fun x => (a₁ * Real.tanh (x * c) - 1) ^ 2)
        (2 * (a₁ * Real.tanh (c * c) - 1) * (a₁ * ((1 - Real.tanh (c * c) ^ 2) * c))) c := by
      have h := (((hd_tanh_c c).const_mul a₁).sub_const 1).pow 2
      simpa [mul_assoc] using h
    have htot :=
      (((hd1.add_const ((a₁ * Real.tanh (0 * c)) ^ 2)).const_mul (w0 / 2)).add_const
        ((w1 / 2) * ((a₁ * Real.tanh (t * c)) ^ 2 + (a₁ * Real.tanh (c * c) - 1) ^ 2))).add
      (((((hd_quad c).add_const (t ^ 2)).add_const ((0 : ℝ) ^ 2)).add_const (c ^ 2)
        |>.add_const (2 * c ^ 2) |>.add_const (4 * a₁)).const_mul (lam / 2))
    convert htot using 1
    have hcc : c * c = c ^ 2 := (sq c).symm
    rw [hcc]
    linear_combination -c * hstat0
  · -- ∂F/∂q01 at t
    simp_rw [hFq]
    have hd1 : HasDerivAt (fun x => (a₁ * Real.tanh (x * c)) ^ 2)
        (2 * (a₁ * Real.tanh (t * c)) * (a₁ * ((1 - Real.tanh (t * c) ^ 2) * c))) t := by
      have h := ((hd_tanh_c t).const_mul a₁).pow 2
      simpa [mul_assoc] using h
    have htot :=
      (((hd1.add_const ((a₁ * Real.tanh (c * c) - 1) ^ 2)).const_mul (w1 / 2)).const_add
        ((w0 / 2) * ((a₁ * Real.tanh (c * c) - 1) ^ 2 + (a₁ * Real.tanh (0 * c)) ^ 2))).add
      (((((hd_quad t).const_add (c ^ 2)).add_const ((0 : ℝ) ^ 2)).add_const (c ^ 2)
        |>.add_const (2 * c ^ 2) |>.add_const (4 * a₁)).const_mul (lam / 2))
    convert htot using 1
    dsimp only [g01]
    rw [mul_comm t c]
    ring
  · -- ∂F/∂q10 at 0
    simp_rw [hFq]
    have hd1 : HasDerivAt (fun x => (a₁ * Real.tanh (x * c)) ^ 2)
        (2 * (a₁ * Real.tanh (0 * c)) * (a₁ * ((1 - Real.tanh (0 * c) ^ 2) * c))) 0 := by
      have h := ((hd_tanh_c 0).const_mul a₁).pow 2
      simpa [mul_assoc] using h
    have htot :=
      (((hd1.const_add ((a₁ * Real.tanh (c * c) - 1) ^ 2)).const_mul (w0 / 2)).add_const
        ((w1 / 2) * ((a₁ * Real.tanh (t * c)) ^ 2 + (a₁ * Real.tanh (c * c) - 1) ^ 2))).add
      (((((hd_quad 0).const_add (c ^ 2 + t ^ 2)).add_const (c ^ 2)).add_const (2 * c ^ 2)
        |>.add_const (4 * a₁)).const_mul (lam / 2))
    convert htot using 1
    simp
  · -- ∂F/∂q11 at c
    simp_rw [hFq]
    have hd1 : HasDerivAt (fun x => (a₁ * Real.tanh (x * c) - 1) ^ 2)
        (2 * (a₁ * Real.tanh (c * c) - 1) * (a₁ * ((1 - Real.tanh (c * c) ^ 2) * c))) c := by
      have h := (((hd_tanh_c c).const_mul a₁).sub_const 1).pow 2
      simpa [mul_assoc] using h
    have htot :=
      (((hd1.const_add ((a₁ * Real.tanh (t * c)) ^ 2)).const_mul (w1 / 2)).const_add
        ((w0 / 2) * ((a₁ * Real.tanh (c * c) - 1) ^ 2 + (a₁ * Real.tanh (0 * c)) ^ 2))).add
      (((((hd_quad c).const_add (c ^ 2 + t ^ 2 + 0 ^ 2)).add_const (2 * c ^ 2)).add_const (4 * a₁)).const_mul (lam / 2))
    convert htot using 1
    have hcc : c * c = c ^ 2 := (sq c).symm
    rw [hcc]
    linear_combination -c * hstat1
  · -- g01 > 0
    dsimp only [g01]
    have hct_pos : 0 < c * t := mul_pos hc ht
    have hth_pos : 0 < Real.tanh (c * t) := tanh_pos_of_pos hct_pos
    have hsech2_pos : 0 < 1 - Real.tanh (c * t) ^ 2 := by
      have hlt1 : Real.tanh (c * t) < 1 := by
        rw [Real.tanh_eq_sinh_div_cosh, div_lt_one (Real.cosh_pos _), Real.sinh_eq, Real.cosh_eq]
        linarith [Real.exp_pos (-(c * t))]
      nlinarith
    positivity

/-- **Proposition 4(b) — Exact negative initial derivative `dR/dt|_{s=0} < 0` along gradient flow**:
For any differentiable trajectory `(q00(s), q01(s), q10(s), q11(s))` starting at `(c, t, 0, c)`
with Euclidean gradient-flow velocities `(0, ṫ, 0, 0)` where `ṫ < 0` (given by
`tilted_gradient_exact`):
1. `R(0) = t² / (c² + t²)`;
2. `HasDerivAt (fun s => overlap2 ![q00 s, q01 s] ![q10 s, q11 s]) ((2 * t * c² / (c² + t²)²) * ṫ) 0`;
3. `(2 * t * c² / (c² + t²)²) * ṫ < 0`;
4. At `t = (11 / 10) * c`, `R(0) = 121 / 221 > 1 / 2` (strictly above the Haar baseline). -/
theorem dynamic_overlap_deriv_neg {c t tdot : ℝ} (hc : 0 < c) (ht : 0 < t) (htdot : tdot < 0)
    (q00 q01 q10 q11 : ℝ → ℝ)
    (h00 : q00 0 = c) (h01 : q01 0 = t) (h10 : q10 0 = 0) (h11 : q11 0 = c)
    (hd00 : HasDerivAt q00 0 0) (hd01 : HasDerivAt q01 tdot 0)
    (hd10 : HasDerivAt q10 0 0) (hd11 : HasDerivAt q11 0 0) :
    let Rtraj : ℝ → ℝ := fun s => overlap2 ![q00 s, q01 s] ![q10 s, q11 s]
    Rtraj 0 = t ^ 2 / (c ^ 2 + t ^ 2) ∧
    HasDerivAt Rtraj ((2 * t * c ^ 2 / (c ^ 2 + t ^ 2) ^ 2) * tdot) 0 ∧
    (2 * t * c ^ 2 / (c ^ 2 + t ^ 2) ^ 2) * tdot < 0 ∧
    (11 / 10 * c) ^ 2 / (c ^ 2 + (11 / 10 * c) ^ 2) = 121 / 221 ∧
    (1 / 2 : ℝ) < 121 / 221 := by
  intro Rtraj
  have hc2 : 0 < c ^ 2 := sq_pos_of_pos hc
  have hct2 : 0 < c ^ 2 + t ^ 2 := by positivity
  have hnum_d : HasDerivAt (fun s => (q00 s * q10 s + q01 s * q11 s) ^ 2)
      (2 * (t * c) * (tdot * c)) 0 := by
    have h1 := (hd00.mul hd10).add (hd01.mul hd11)
    have h2 := h1.pow 2
    simp only [Pi.add_apply, Pi.mul_apply, h00, h01, h10, h11] at h2
    convert h2 using 1
    ring
  have hden_d : HasDerivAt (fun s => (q00 s ^ 2 + q01 s ^ 2) * (q10 s ^ 2 + q11 s ^ 2))
      ((2 * t * tdot) * c ^ 2) 0 := by
    have hA : HasDerivAt (fun s => q00 s ^ 2 + q01 s ^ 2) (2 * t * tdot) 0 := by
      have h := (hd00.pow 2).add (hd01.pow 2)
      simp only [h00, h01] at h
      convert h using 1; ring
    have hB : HasDerivAt (fun s => q10 s ^ 2 + q11 s ^ 2) 0 0 := by
      have h := (hd10.pow 2).add (hd11.pow 2)
      simp only [h10, h11] at h
      convert h using 1; ring
    have hmul := hA.mul hB
    simp only [h00, h01, h10, h11] at hmul
    convert hmul using 1
    ring
  have hden0 : ((q00 0 ^ 2 + q01 0 ^ 2) * (q10 0 ^ 2 + q11 0 ^ 2)) ≠ 0 := by
    rw [h00, h01, h10, h11]
    have : 0 < (c ^ 2 + t ^ 2) * (0 ^ 2 + c ^ 2) := by positivity
    exact ne_of_gt this
  refine ⟨?_, ?_, ?_, ?_, by norm_num⟩
  · dsimp only [Rtraj, overlap2]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, h00, h01, h10, h11]
    field_simp [ne_of_gt hc2, ne_of_gt hct2]
    ring
  · have hq := hnum_d.div hden_d hden0
    have hfun : Rtraj = fun s =>
        (q00 s * q10 s + q01 s * q11 s) ^ 2 /
        ((q00 s ^ 2 + q01 s ^ 2) * (q10 s ^ 2 + q11 s ^ 2)) := by
      ext s; simp [Rtraj, overlap2]
    rw [hfun]
    convert hq using 1
    rw [h00, h01, h10, h11]
    field_simp [ne_of_gt hc2, ne_of_gt hct2]
    ring
  · have hcoeff : 0 < 2 * t * c ^ 2 / (c ^ 2 + t ^ 2) ^ 2 := by positivity
    exact mul_neg_of_pos_of_neg hcoeff htdot
  · field_simp [ne_of_gt hc2]
    ring

end FlagVarietyAlignment
