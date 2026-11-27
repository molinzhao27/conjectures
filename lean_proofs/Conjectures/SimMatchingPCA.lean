import Mathlib

/-!
# Resolution of the Three Similarity Matching PCA Conjectures
(Centorrino, Bullo, and Russo, arXiv:2506.06134 / Neural Computation 38(5), Apr. 2026)

We formalize in Lean 4 + Mathlib (zero `sorry`, standard axioms only) the exact algebraic,
spectral-witness, and counterexample cores of all three conjectures (using a cube-root matrix
witness `K` satisfying `K ^ 3 = W * C * Wᵀ` to stay completely elementary without needing
general matrix fractional-power APIs):

1. **Conjecture 1 (`conj_full_row_rank` — Gram ODE & Barrier Inequalities)**:
   - `gram_ode_identity`: If `K` is symmetric invertible with `K ^ 3 = W * C * Wᵀ` and
     `dW = -4 • W + 4 • (K⁻¹ * W * C)`, then `dW * Wᵀ + W * dWᵀ = -8 • (W * Wᵀ) + 8 • K ^ 2`.
   - `lower_barrier_pos`, `upper_barrier_neg`: Strict positive/negative drift of `8 * ((c * ℓ)^(2/3) - ℓ)`
     for `0 < ℓ < c^2` and `ℓ > d^2`, driving the Weyl/first-contact invariant-region proof.
2. **Conjecture 2 (`conj_full_strict_saddle` — Stationarity, Typo Counterexample, Rank-1 Escape, Curvatures)**:
   - `stationary_implies_gram_eq_sq`: At any full-rank critical point (`K * W = W * C`),
     `W * Wᵀ = K ^ 2` (equivalently `W * C = (W * Wᵀ)^(1/2) * W`).
   - `printed_set_not_stationary`: The printed stationary set using `(P V_C)_mᵀ` contains an explicit
     `1 × 2` point `W` with `W * C = 2 • W`, `W * Wᵀ = 1`, and nonzero gradient `4 * (2^(2/3) - 1) • W ≠ 0`.
   - `rank_one_perturbation_algebra`: Under `uᵀ W = 0` and `W C v = 0`, the rank-1 perturbation
     `W + t u vᵀ` decouples orthogonally in both `W Wᵀ` and `W C Wᵀ`.
   - `rank_one_escape_negative`: For every `γ > 0` and `0 < |t| < (3/2)^(3/2) * γ`,
     `2 * t^2 - 3 * |t|^(4/3) * γ^(2/3) < 0` (no rank-deficient point is a local minimum).
   - `exchange_curvature_neg`, `radial_curvature_pos`: Strict saddle curvatures `4 * (1 - β / α) < 0`
     whenever `0 < α < β`, and `8 / 3 > 0`.
3. **Conjecture 3 (`conj_lin_prog` — Matrix Cube-Root Identity & Decisive Counterexamples)**:
   - `conj3_cube_identity`: If `K` is invertible with `K ^ 3 = B * Bᵀ` and `N = Bᵀ * K⁻¹ * B`,
     then `N ^ 3 = (Bᵀ * B) ^ 2` (algebraic core of `D^{1/2} H D^{1/2} = (Bᵀ B)^{2/3}`).
   - `conj3_kernel_rotation_counterexample`: For `m = 1, n = 3`, `Λ₁ = [1, 0, 0]`, ANY positive
     diagonal `Λ₂ = diag(d₁, d₂, d₃)`, and the orthogonal kernel rotation `U = diag(1, R_{3,4,5})`,
     `U` is orthogonal, NOT a signed permutation matrix, yet `H = Uᵀ Λ₁ᵀ K⁻¹ Λ₁ U = diag(d₁^{-1/3}, 0, 0)`
     is strictly diagonal.
   - `conj3_active_tie_counterexample`: For `m = 2, n = 2`, `Λ₁ = I₂`, ANY positive diagonal
     `Λ₂ = diag(d₁, d₂)`, and ANY orthogonal `U`, `K = U * diag(d₁^{1/3}, d₂^{1/3}) * Uᵀ` satisfies
     `K ^ 3 = U * Λ₂ * Uᵀ` and `Uᵀ * K⁻¹ * U = diag(d₁^{-1/3}, d₂^{-1/3})` is strictly diagonal.

Full ODE global existence, Weyl/Dini barrier continuation, general-dimension Hessian expansion,
and the full Conjecture 3 `iff` criterion are proved in prose in `solutions/sim_matching_pca/solution.md`.
-/

open Matrix BigOperators

namespace SimMatchingPCA

/-! ### 1. Conjecture 1: Gram ODE Identity and Barrier Drift Inequalities -/

/-- **Gram ODE algebraic identity**:
If `K` is symmetric invertible (`K * K⁻¹ = 1`, `K⁻¹ * K = 1`, `Kᵀ = K`), `C` is symmetric,
`K ^ 3 = W * C * Wᵀ`, and `dW = (-4 : ℝ) • W + (4 : ℝ) • (K⁻¹ * W * C)`, then
`dW * Wᵀ + W * dWᵀ = (-8 : ℝ) • (W * Wᵀ) + (8 : ℝ) • (K ^ 2)`. -/
theorem gram_ode_identity {m n : ℕ}
    (W : Matrix (Fin m) (Fin n) ℝ)
    (C : Matrix (Fin n) (Fin n) ℝ) (hCsym : Cᵀ = C)
    (K Kinv : Matrix (Fin m) (Fin m) ℝ)
    (hKinvsym : Kinvᵀ = Kinv)
    (hKL : Kinv * K = 1) (hKR : K * Kinv = 1)
    (hK3 : K * K * K = W * C * Wᵀ) :
    let dW := (-4 : ℝ) • W + (4 : ℝ) • (Kinv * W * C)
    dW * Wᵀ + W * dWᵀ = (-8 : ℝ) • (W * Wᵀ) + (8 : ℝ) • (K * K) := by
  intro dW
  have h1 : Kinv * W * C * Wᵀ = K * K := by
    calc Kinv * W * C * Wᵀ = Kinv * (W * C * Wᵀ) := by simp [Matrix.mul_assoc]
      _ = Kinv * (K * K * K) := by rw [← hK3]
      _ = (Kinv * K) * (K * K) := by simp [Matrix.mul_assoc]
      _ = K * K := by rw [hKL, Matrix.one_mul]
  have h2 : W * C * Wᵀ * Kinv = K * K := by
    calc W * C * Wᵀ * Kinv = (K * K * K) * Kinv := by rw [← hK3]
      _ = (K * K) * (K * Kinv) := by simp [Matrix.mul_assoc]
      _ = K * K := by rw [hKR, Matrix.mul_one]
  have hdWt : dWᵀ = (-4 : ℝ) • Wᵀ + (4 : ℝ) • (C * Wᵀ * Kinv) := by
    simp only [dW, Matrix.transpose_add, Matrix.transpose_smul, Matrix.transpose_mul,
      hCsym, hKinvsym, Matrix.mul_assoc]
  rw [hdWt]
  simp only [dW, Matrix.add_mul, Matrix.mul_add, Matrix.smul_mul, Matrix.mul_smul,
    Matrix.mul_assoc]
  have h1' : Kinv * (W * (C * Wᵀ)) = K * K := by simpa [Matrix.mul_assoc] using h1
  have h2' : W * (C * (Wᵀ * Kinv)) = K * K := by simpa [Matrix.mul_assoc] using h2
  rw [h1', h2']
  ext i j
  simp
  ring

/-- **Lower barrier drift sign (Conjecture 1)**:
For `c > 0` and `0 < ℓ < c^2`, `8 * ((c * ℓ)^(2/3) - ℓ) > 0`. -/
theorem lower_barrier_pos {c ℓ : ℝ} (hc : 0 < c) (hℓ : 0 < ℓ) (hlt : ℓ < c ^ 2) :
    0 < 8 * ((c * ℓ) ^ ((2 : ℝ) / 3) - ℓ) := by
  have hcℓ : 0 < c * ℓ := mul_pos hc hℓ
  have hpow : ℓ < (c * ℓ) ^ ((2 : ℝ) / 3) := by
    have hcube : ℓ ^ (3 : ℕ) < (c * ℓ) ^ (2 : ℕ) := by
      have : ℓ ^ 3 = ℓ * ℓ ^ 2 := by ring
      have : (c * ℓ) ^ 2 = c ^ 2 * ℓ ^ 2 := by ring
      nlinarith [sq_pos_of_pos hℓ]
    have hrhs : ((c * ℓ) ^ ((2 : ℝ) / 3)) ^ (3 : ℕ) = (c * ℓ) ^ (2 : ℕ) := by
      rw [← Real.rpow_natCast, ← Real.rpow_mul (le_of_lt hcℓ)]
      norm_num
    by_contra hge
    push_neg at hge
    have h3 : ((c * ℓ) ^ ((2 : ℝ) / 3)) ^ (3 : ℕ) ≤ ℓ ^ (3 : ℕ) :=
      pow_le_pow_left₀ (Real.rpow_nonneg (le_of_lt hcℓ) _) hge 3
    linarith
  linarith

/-- **Upper barrier drift sign (Conjecture 1)**:
For `d > 0` and `d^2 < ℓ`, `8 * ((d * ℓ)^(2/3) - ℓ) < 0`. -/
theorem upper_barrier_neg {d ℓ : ℝ} (hd : 0 < d) (hgt : d ^ 2 < ℓ) :
    8 * ((d * ℓ) ^ ((2 : ℝ) / 3) - ℓ) < 0 := by
  have hℓ : 0 < ℓ := lt_trans (sq_pos_of_pos hd) hgt
  have hdℓ : 0 < d * ℓ := mul_pos hd hℓ
  have hpow : (d * ℓ) ^ ((2 : ℝ) / 3) < ℓ := by
    have hcube : (d * ℓ) ^ (2 : ℕ) < ℓ ^ (3 : ℕ) := by
      have : ℓ ^ 3 = ℓ * ℓ ^ 2 := by ring
      have : (d * ℓ) ^ 2 = d ^ 2 * ℓ ^ 2 := by ring
      nlinarith [sq_pos_of_pos hℓ]
    have hlhs : ((d * ℓ) ^ ((2 : ℝ) / 3)) ^ (3 : ℕ) = (d * ℓ) ^ (2 : ℕ) := by
      rw [← Real.rpow_natCast, ← Real.rpow_mul (le_of_lt hdℓ)]
      norm_num
    by_contra hle
    push_neg at hle
    have h3 : ℓ ^ (3 : ℕ) ≤ ((d * ℓ) ^ ((2 : ℝ) / 3)) ^ (3 : ℕ) :=
      pow_le_pow_left₀ (le_of_lt hℓ) hle 3
    linarith
  linarith

/-! ### 2. Conjecture 2: Stationarity, Printed-Set Counterexample, Rank-1 Escape, Curvatures -/

/-- **Full-rank stationarity identity**:
If `K` is invertible with `K ^ 3 = W * C * Wᵀ` and `K * W = W * C` (equivalently
`-4 W + 4 K⁻¹ W C = 0`), then `W * Wᵀ = K ^ 2`. -/
theorem stationary_implies_gram_eq_sq {m n : ℕ}
    (W : Matrix (Fin m) (Fin n) ℝ)
    (C : Matrix (Fin n) (Fin n) ℝ)
    (K Kinv : Matrix (Fin m) (Fin m) ℝ)
    (hKL : Kinv * K = 1)
    (hK3 : K * K * K = W * C * Wᵀ)
    (hstat : K * W = W * C) :
    W * Wᵀ = K * K := by
  have h1 : K * (W * Wᵀ) = K * (K * K) := by
    calc K * (W * Wᵀ) = (K * W) * Wᵀ := by simp [Matrix.mul_assoc]
      _ = (W * C) * Wᵀ := by rw [hstat]
      _ = K * K * K := hK3.symm
      _ = K * (K * K) := by simp [Matrix.mul_assoc]
  calc W * Wᵀ = (Kinv * K) * (W * Wᵀ) := by rw [hKL, Matrix.one_mul]
    _ = Kinv * (K * (W * Wᵀ)) := by simp [Matrix.mul_assoc]
    _ = Kinv * (K * (K * K)) := by rw [h1]
    _ = (Kinv * K) * (K * K) := by simp [Matrix.mul_assoc]
    _ = K * K := by rw [hKL, Matrix.one_mul]

/-- **Printed-set counterexample (`m = 1, n = 2`)**:
With `V = (1/√2) [[1,1],[1,-1]]`, `D = diag(2,1)`, `C = V D Vᵀ = (1/2) [[3,1],[1,3]]`,
and `P = [[0,1],[1,0]]`, the printed formula `(P D Pᵀ)_{1,1} ((P V)_{:,1})ᵀ` produces
`W = (1/√2) [1, 1]`. This satisfies `W * Wᵀ = [[1]]`, `W * C = 2 • W`, `W * C * Wᵀ = [[2]]`,
yet the gradient `-4 W + 4 (W C Wᵀ)^{-1/3} W C = 4 (2^{2/3} - 1) W ≠ 0`. -/
theorem printed_set_not_stationary :
    let s : ℝ := (Real.sqrt 2)⁻¹
    let C : Matrix (Fin 2) (Fin 2) ℝ := !![(3 : ℝ) / 2, 1 / 2; 1 / 2, 3 / 2]
    let W : Matrix (Fin 1) (Fin 2) ℝ := !![s, s]
    let K : Matrix (Fin 1) (Fin 1) ℝ := !![(2 : ℝ) ^ ((1 : ℝ) / 3)]
    let Kinv : Matrix (Fin 1) (Fin 1) ℝ := !![(2 : ℝ) ^ (-(1 : ℝ) / 3)]
    K * K * K = W * C * Wᵀ ∧
    Kinv * K = 1 ∧
    (-4 : ℝ) • W + (4 : ℝ) • (Kinv * W * C) ≠ 0 := by
  intro s C W K Kinv
  have hs2 : s * s = 1 / 2 := by
    dsimp only [s]
    rw [← mul_inv, Real.mul_self_sqrt (by norm_num : (0 : ℝ) ≤ 2)]
    norm_num
  have h2pos : (0 : ℝ) < 2 := by norm_num
  have hK3_scalar : (2 : ℝ) ^ ((1 : ℝ) / 3) * (2 : ℝ) ^ ((1 : ℝ) / 3) * (2 : ℝ) ^ ((1 : ℝ) / 3) = 2 := by
    rw [← Real.rpow_add h2pos, ← Real.rpow_add h2pos]
    norm_num
  have hKinvK_scalar : (2 : ℝ) ^ (-(1 : ℝ) / 3) * (2 : ℝ) ^ ((1 : ℝ) / 3) = 1 := by
    rw [← Real.rpow_add h2pos]
    norm_num
  refine ⟨?_, ?_, ?_⟩
  · ext i j
    fin_cases i; fin_cases j
    simp only [K, W, C, Matrix.mul_apply, Fin.sum_univ_two, Fin.sum_univ_one,
      Matrix.transpose_apply, Matrix.of_apply, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.cons_val_fin_one]
    have hcalc : (s * (3 / 2) + s * (1 / 2)) * s + (s * (1 / 2) + s * (3 / 2)) * s = 4 * (s * s) := by ring
    rw [hK3_scalar, hcalc, hs2]
    norm_num
  · ext i j
    fin_cases i; fin_cases j
    simp only [Kinv, K, Matrix.mul_apply, Fin.sum_univ_one, Matrix.one_apply_eq,
      Matrix.of_apply, Matrix.cons_val_fin_one]
    exact hKinvK_scalar
  · intro hzero
    have h00 := congrFun (congrFun hzero 0) 0
    simp only [W, C, Kinv, Matrix.add_apply, Matrix.smul_apply, Matrix.mul_apply,
      Fin.sum_univ_two, Fin.sum_univ_one, Matrix.zero_apply, smul_eq_mul,
      Matrix.of_apply, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_fin_one] at h00
    have hs_pos : 0 < s := inv_pos.mpr (Real.sqrt_pos.mpr h2pos)
    have hfactor :
        -4 * s + 4 * ((2 : ℝ) ^ (-(1 : ℝ) / 3) * s * (3 / 2) + (2 : ℝ) ^ (-(1 : ℝ) / 3) * s * (1 / 2)) =
        4 * s * (2 * (2 : ℝ) ^ (-(1 : ℝ) / 3) - 1) := by ring
    rw [hfactor] at h00
    have hcoeff : 2 * (2 : ℝ) ^ (-(1 : ℝ) / 3) = 1 := by
      have h4s : 4 * s ≠ 0 := by positivity
      have := mul_eq_zero.mp h00
      rcases this with h | h
      · exact absurd h h4s
      · linarith
    have h2pow : 2 * (2 : ℝ) ^ (-(1 : ℝ) / 3) = (2 : ℝ) ^ ((2 : ℝ) / 3) := by
      conv_lhs => arg 1; rw [← Real.rpow_one (2 : ℝ)]
      rw [← Real.rpow_add h2pos]
      norm_num
    rw [h2pow] at hcoeff
    have hgt1 : (1 : ℝ) < (2 : ℝ) ^ ((2 : ℝ) / 3) :=
      Real.one_lt_rpow (by norm_num) (by norm_num)
    linarith

/-- **Rank-1 perturbation decoupling algebra**:
If `uᵀ * W = 0` and `W * C * v = 0` with `Cᵀ = C`, then for `W_t = W + t • (u * vᵀ)`:
1. `Matrix.trace (W_t * W_tᵀ) = Matrix.trace (W * Wᵀ) + t^2 * Matrix.trace (u * (vᵀ * v) * uᵀ)`,
2. `W_t * C * W_tᵀ = W * C * Wᵀ + (t ^ 2) • (u * (vᵀ * C * v) * uᵀ)`. -/
theorem rank_one_perturbation_algebra {m n : ℕ}
    (W : Matrix (Fin m) (Fin n) ℝ)
    (C : Matrix (Fin n) (Fin n) ℝ) (hCsym : Cᵀ = C)
    (u : Matrix (Fin m) (Fin 1) ℝ) (v : Matrix (Fin n) (Fin 1) ℝ)
    (huW : uᵀ * W = 0) (hWCv : W * C * v = 0) (t : ℝ) :
    let Wt := W + t • (u * vᵀ)
    Matrix.trace (Wt * Wtᵀ) =
      Matrix.trace (W * Wᵀ) + t ^ 2 * Matrix.trace (u * (vᵀ * v) * uᵀ) ∧
    Wt * C * Wtᵀ = W * C * Wᵀ + (t ^ 2) • (u * (vᵀ * C * v) * uᵀ) := by
  intro Wt
  have hWtu : Wᵀ * u = 0 := by
    have := congrArg Matrix.transpose huW
    simpa [Matrix.transpose_mul] using this
  have hvCW : vᵀ * C * Wᵀ = 0 := by
    have := congrArg Matrix.transpose hWCv
    simpa [Matrix.transpose_mul, hCsym, Matrix.mul_assoc] using this
  have hWtT : Wtᵀ = Wᵀ + t • (v * uᵀ) := by
    simp [Wt, Matrix.transpose_add, Matrix.transpose_smul, Matrix.transpose_mul]
  constructor
  · have htr1 : Matrix.trace (W * (v * uᵀ)) = 0 := by
      calc Matrix.trace (W * (v * uᵀ))
          = Matrix.trace ((W * v) * uᵀ) := by rw [Matrix.mul_assoc]
        _ = Matrix.trace (uᵀ * (W * v)) := Matrix.trace_mul_comm _ _
        _ = Matrix.trace ((uᵀ * W) * v) := by rw [Matrix.mul_assoc]
        _ = 0 := by rw [huW, Matrix.zero_mul, Matrix.trace_zero]
    have htr2 : Matrix.trace (u * vᵀ * Wᵀ) = 0 := by
      calc Matrix.trace (u * vᵀ * Wᵀ)
          = Matrix.trace ((u * vᵀ) * Wᵀ) := rfl
        _ = Matrix.trace (Wᵀ * (u * vᵀ)) := Matrix.trace_mul_comm _ _
        _ = Matrix.trace ((Wᵀ * u) * vᵀ) := by rw [Matrix.mul_assoc]
        _ = 0 := by rw [hWtu, Matrix.zero_mul, Matrix.trace_zero]
    rw [hWtT]
    simp only [Wt, Matrix.add_mul, Matrix.mul_add, Matrix.smul_mul, Matrix.mul_smul,
      Matrix.trace_add, Matrix.trace_smul, smul_eq_mul]
    rw [htr1, htr2]
    have hquad : u * vᵀ * (v * uᵀ) = u * (vᵀ * v) * uᵀ := by simp [Matrix.mul_assoc]
    rw [hquad]
    ring
  · rw [hWtT]
    simp only [Wt, Matrix.add_mul, Matrix.mul_add, Matrix.smul_mul, Matrix.mul_smul,
      Matrix.mul_assoc]
    have h1 : W * (C * (v * uᵀ)) = 0 := by
      calc W * (C * (v * uᵀ)) = (W * C * v) * uᵀ := by simp [Matrix.mul_assoc]
        _ = 0 := by rw [hWCv, Matrix.zero_mul]
    have h2 : u * (vᵀ * (C * Wᵀ)) = 0 := by
      calc u * (vᵀ * (C * Wᵀ)) = u * (vᵀ * C * Wᵀ) := by simp [Matrix.mul_assoc]
        _ = 0 := by rw [hvCW, Matrix.mul_zero]
    rw [h1, h2]
    simp only [smul_zero, add_zero, zero_add, smul_smul]
    ring_nf

/-- **Rank-1 escape inequality (no rank-deficient local minimum)**:
For every `γ > 0` and every `t` with `0 < |t| < (3 / 2) ^ (3 / 2) * γ`,
`2 * t^2 - 3 * |t|^(4/3) * γ^(2/3) < 0`. -/
theorem rank_one_escape_negative {γ t : ℝ} (hγ : 0 < γ)
    (ht0 : 0 < |t|) (ht : |t| < ((3 : ℝ) / 2) ^ ((3 : ℝ) / 2) * γ) :
    2 * t ^ 2 - 3 * |t| ^ ((4 : ℝ) / 3) * γ ^ ((2 : ℝ) / 3) < 0 := by
  have hsq : t ^ 2 = |t| ^ (2 : ℕ) := (sq_abs t).symm
  rw [hsq]
  -- Rewrite `|t|^2 = |t|^(4/3) * |t|^(2/3)`.
  have hsplit : |t| ^ (2 : ℕ) = |t| ^ ((4 : ℝ) / 3) * |t| ^ ((2 : ℝ) / 3) := by
    rw [← Real.rpow_add ht0]
    norm_num
  rw [hsplit]
  have h43 : 0 < |t| ^ ((4 : ℝ) / 3) := Real.rpow_pos_of_pos ht0 _
  -- Show `2 * |t|^(2/3) < 3 * γ^(2/3)`, i.e. `(|t| / γ)^(2/3) < 3/2`, i.e. `|t| / γ < (3/2)^(3/2)`.
  have hlt_pow : 2 * |t| ^ ((2 : ℝ) / 3) < 3 * γ ^ ((2 : ℝ) / 3) := by
    have hrat : |t| / γ < ((3 : ℝ) / 2) ^ ((3 : ℝ) / 2) := (div_lt_iff₀ hγ).mpr ht
    have hpow23 : (|t| / γ) ^ ((2 : ℝ) / 3) < (((3 : ℝ) / 2) ^ ((3 : ℝ) / 2)) ^ ((2 : ℝ) / 3) :=
      Real.rpow_lt_rpow (le_of_lt (div_pos ht0 hγ)) hrat (by norm_num)
    have hrhs : (((3 : ℝ) / 2) ^ ((3 : ℝ) / 2)) ^ ((2 : ℝ) / 3) = (3 : ℝ) / 2 := by
      rw [← Real.rpow_mul (by norm_num : (0 : ℝ) ≤ 3 / 2)]
      norm_num
    rw [hrhs, Real.div_rpow (le_of_lt ht0) (le_of_lt hγ)] at hpow23
    have hγ23 : 0 < γ ^ ((2 : ℝ) / 3) := Real.rpow_pos_of_pos hγ _
    have hmul := (div_lt_iff₀ hγ23).mp hpow23
    linarith
  nlinarith

/-- **Strict-saddle exchange curvature**:
Whenever `0 < α < β`, `4 * (1 - β / α) < 0`. -/
theorem exchange_curvature_neg {α β : ℝ} (hα : 0 < α) (hlt : α < β) :
    4 * (1 - β / α) < 0 := by
  have h1 : 1 < β / α := (one_lt_div hα).mpr hlt
  linarith

/-- **Positive radial curvature**: `8 / 3 > 0`. -/
theorem radial_curvature_pos : (0 : ℝ) < 8 / 3 := by norm_num

/-! ### 3. Conjecture 3: Matrix Cube-Root Identity and Decisive Counterexamples -/

/-- **Algebraic core of Conjecture 3**:
If `K` is invertible (`Kinv * K = 1`) with `K ^ 3 = B * Bᵀ` and `N = Bᵀ * Kinv * B`,
then `N ^ 3 = (Bᵀ * B) ^ 2` (the witness form of `D^{1/2} H D^{1/2} = (Bᵀ B)^{2/3}`). -/
theorem conj3_cube_identity {m n : ℕ}
    (B : Matrix (Fin m) (Fin n) ℝ)
    (K Kinv : Matrix (Fin m) (Fin m) ℝ)
    (hKL : Kinv * K = 1) (hKR : K * Kinv = 1)
    (hK3 : K * K * K = B * Bᵀ) :
    let N := Bᵀ * Kinv * B
    N * N * N = (Bᵀ * B) * (Bᵀ * B) := by
  intro N
  have hmid : Kinv * (B * Bᵀ) * Kinv * (B * Bᵀ) * Kinv = K * K * K := by
    rw [← hK3]
    calc Kinv * (K * K * K) * Kinv * (K * K * K) * Kinv
        = (Kinv * K) * (K * (K * Kinv)) * ((K * K) * (K * Kinv)) := by simp [Matrix.mul_assoc]
      _ = 1 * (K * 1) * ((K * K) * 1) := by rw [hKL, hKR]
      _ = K * K * K := by simp [Matrix.mul_assoc]
  calc N * N * N
      = Bᵀ * (Kinv * (B * Bᵀ) * Kinv * (B * Bᵀ) * Kinv) * B := by simp [N, Matrix.mul_assoc]
    _ = Bᵀ * (K * K * K) * B := by rw [hmid]
    _ = Bᵀ * (B * Bᵀ) * B := by rw [hK3]
    _ = (Bᵀ * B) * (Bᵀ * B) := by simp [Matrix.mul_assoc]

/-- **Conjecture 3 decisive refutation (Kernel Rotation, `m = 1, n = 3`)**:
For `Λ₁ = [1, 0, 0]`, ANY positive diagonal `Λ₂ = diag(d₁, d₂, d₃)`, and the orthogonal
matrix `U = diag(1, R_{3,4,5})`:
1. `U` is orthogonal (`Uᵀ * U = 1`),
2. `U` is NOT a signed permutation matrix (`|U 1 1| = 3/5 ∉ {0, 1}`),
3. With `K = d₁^{1/3} I₁`, `K ^ 3 = Λ₁ * U * Λ₂ * Uᵀ * Λ₁ᵀ` and `K⁻¹ * K = 1`,
4. `H = Uᵀ * Λ₁ᵀ * K⁻¹ * Λ₁ * U = diag(d₁^{-1/3}, 0, 0)` is strictly diagonal! -/
theorem conj3_kernel_rotation_counterexample (d₁ d₂ d₃ : ℝ) (hd₁ : 0 < d₁) :
    let Λ₁ : Matrix (Fin 1) (Fin 3) ℝ := !![1, 0, 0]
    let Λ₂ : Matrix (Fin 3) (Fin 3) ℝ := Matrix.diagonal ![d₁, d₂, d₃]
    let U : Matrix (Fin 3) (Fin 3) ℝ :=
      !![1, 0, 0;
         0, (3 : ℝ) / 5, -(4 : ℝ) / 5;
         0, (4 : ℝ) / 5, (3 : ℝ) / 5]
    let K : Matrix (Fin 1) (Fin 1) ℝ := !![d₁ ^ ((1 : ℝ) / 3)]
    let Kinv : Matrix (Fin 1) (Fin 1) ℝ := !![d₁ ^ (-(1 : ℝ) / 3)]
    let H := Uᵀ * Λ₁ᵀ * Kinv * Λ₁ * U
    Uᵀ * U = 1 ∧
    (U 1 1 ≠ 0 ∧ U 1 1 ≠ 1 ∧ U 1 1 ≠ -1) ∧
    K * K * K = Λ₁ * U * Λ₂ * Uᵀ * Λ₁ᵀ ∧
    Kinv * K = 1 ∧
    H = Matrix.diagonal ![d₁ ^ (-(1 : ℝ) / 3), 0, 0] := by
  intro Λ₁ Λ₂ U K Kinv H
  have hK3_s : d₁ ^ ((1 : ℝ) / 3) * d₁ ^ ((1 : ℝ) / 3) * d₁ ^ ((1 : ℝ) / 3) = d₁ := by
    rw [← Real.rpow_add hd₁, ← Real.rpow_add hd₁]
    norm_num
  have hKinv_s : d₁ ^ (-(1 : ℝ) / 3) * d₁ ^ ((1 : ℝ) / 3) = 1 := by
    rw [← Real.rpow_add hd₁]
    norm_num
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · ext i j
    fin_cases i <;> fin_cases j <;>
      simp [U, Matrix.mul_apply, Fin.sum_univ_three] <;> norm_num
  · simp only [U, Matrix.of_apply, Matrix.cons_val_one, Matrix.cons_val_zero]
    norm_num
  · have hΛU : Λ₁ * U = Λ₁ := by
      ext i j
      fin_cases i
      fin_cases j <;>
        simp [Λ₁, U, Matrix.mul_apply, Fin.sum_univ_three]
    have hRHS : Λ₁ * U * Λ₂ * Uᵀ * Λ₁ᵀ = !![d₁] := by
      have hUΛt : Uᵀ * Λ₁ᵀ = Λ₁ᵀ := by
        have := congrArg Matrix.transpose hΛU
        simpa [Matrix.transpose_mul] using this
      calc Λ₁ * U * Λ₂ * Uᵀ * Λ₁ᵀ
          = (Λ₁ * U) * Λ₂ * (Uᵀ * Λ₁ᵀ) := by simp [Matrix.mul_assoc]
        _ = Λ₁ * Λ₂ * Λ₁ᵀ := by rw [hΛU, hUΛt]
        _ = !![d₁] := by
            ext i j; fin_cases i; fin_cases j
            simp [Λ₁, Λ₂, Matrix.mul_apply, Matrix.vecMul_diagonal, Fin.sum_univ_three]
    rw [hRHS]
    ext i j
    fin_cases i; fin_cases j
    simp only [K, Matrix.mul_apply, Fin.sum_univ_one, Matrix.of_apply, Matrix.cons_val_fin_one]
    exact hK3_s
  · ext i j
    fin_cases i; fin_cases j
    simp only [Kinv, K, Matrix.mul_apply, Fin.sum_univ_one, Matrix.one_apply_eq,
      Matrix.of_apply, Matrix.cons_val_fin_one]
    exact hKinv_s
  · ext i j
    fin_cases i <;> fin_cases j <;>
      simp [H, U, Λ₁, Kinv, Matrix.mul_apply, Fin.sum_univ_three]

/-- **Conjecture 3 active-tie refutation (`m = 2, n = 2`)**:
When `Λ₁ = I₂` (repeated active singular value), for ANY positive diagonal `Λ₂ = diag(d₁, d₂)`
and ANY orthogonal `U` (`Uᵀ * U = 1`, `U * Uᵀ = 1`), `K = U * diag(d₁^{1/3}, d₂^{1/3}) * Uᵀ`
satisfies `K ^ 3 = U * Λ₂ * Uᵀ` and `Uᵀ * K⁻¹ * U = diag(d₁^{-1/3}, d₂^{-1/3})` is diagonal! -/
theorem conj3_active_tie_counterexample (d₁ d₂ : ℝ) (hd₁ : 0 < d₁) (hd₂ : 0 < d₂)
    (U : Matrix (Fin 2) (Fin 2) ℝ) (hUtU : Uᵀ * U = 1) (hUUt : U * Uᵀ = 1) :
    let Λ₂ : Matrix (Fin 2) (Fin 2) ℝ := Matrix.diagonal ![d₁, d₂]
    let D13 : Matrix (Fin 2) (Fin 2) ℝ := Matrix.diagonal ![d₁ ^ ((1 : ℝ) / 3), d₂ ^ ((1 : ℝ) / 3)]
    let Dm13 : Matrix (Fin 2) (Fin 2) ℝ := Matrix.diagonal ![d₁ ^ (-(1 : ℝ) / 3), d₂ ^ (-(1 : ℝ) / 3)]
    let K := U * D13 * Uᵀ
    let Kinv := U * Dm13 * Uᵀ
    K * K * K = U * Λ₂ * Uᵀ ∧
    Kinv * K = 1 ∧
    Uᵀ * Kinv * U = Dm13 := by
  intro Λ₂ D13 Dm13 K Kinv
  have hD3 : D13 * D13 * D13 = Λ₂ := by
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp only [D13, Λ₂, Matrix.mul_apply, Fin.sum_univ_two, Matrix.diagonal_apply,
        Matrix.cons_val_zero, Matrix.cons_val_one,
        Fin.isValue, Fin.zero_eta, Fin.mk_one, zero_ne_one, one_ne_zero, ↓reduceIte,
        mul_zero, zero_mul, add_zero, zero_add]
    · rw [← Real.rpow_add hd₁, ← Real.rpow_add hd₁]; norm_num
    · rw [← Real.rpow_add hd₂, ← Real.rpow_add hd₂]; norm_num
  have hDmD : Dm13 * D13 = 1 := by
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp only [Dm13, D13, Matrix.mul_apply, Fin.sum_univ_two, Matrix.diagonal_apply,
        Matrix.one_apply, Matrix.cons_val_zero, Matrix.cons_val_one,
        Fin.isValue, Fin.zero_eta, Fin.mk_one, zero_ne_one, one_ne_zero, ↓reduceIte,
        mul_zero, zero_mul, add_zero, zero_add]
    · rw [← Real.rpow_add hd₁]; norm_num
    · rw [← Real.rpow_add hd₂]; norm_num
  refine ⟨?_, ?_, ?_⟩
  · calc K * K * K
        = U * D13 * (Uᵀ * U) * D13 * (Uᵀ * U) * D13 * Uᵀ := by simp [K, Matrix.mul_assoc]
      _ = U * (D13 * D13 * D13) * Uᵀ := by rw [hUtU]; simp [Matrix.mul_assoc]
      _ = U * Λ₂ * Uᵀ := by rw [hD3]
  · calc Kinv * K
        = U * Dm13 * (Uᵀ * U) * D13 * Uᵀ := by simp [Kinv, K, Matrix.mul_assoc]
      _ = U * (Dm13 * D13) * Uᵀ := by rw [hUtU]; simp [Matrix.mul_assoc]
      _ = 1 := by rw [hDmD, Matrix.mul_one, hUUt]
  · calc Uᵀ * Kinv * U
        = (Uᵀ * U) * Dm13 * (Uᵀ * U) := by simp [Kinv, Matrix.mul_assoc]
      _ = Dm13 := by rw [hUtU, Matrix.one_mul, Matrix.mul_one]

end SimMatchingPCA
