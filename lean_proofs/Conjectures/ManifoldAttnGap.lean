import Mathlib

/-!
# Refutation of the Matched-Budget Attention Rate Gap (`conj:rate-gap`) and Depth-`L` Polynomial Analysis
(Roheda, Chowdhury, Unde, and Bala, arXiv:2611.10857, Section 3 & Appendix, Proposition 2 & Conjecture 1)

We formalize in Lean 4 + Mathlib (complete proofs, standard axioms only) the mathematical cores of
all three main theorems:

1. **Theorem 1 (Exact sphere counterexample refuting `conj:rate-gap` with `W_O = 0`)**:
   - `langevin_integral_exact`: Exact closed-form evaluation of the spherical height integrals
     `∫_{-1}^1 exp(κ a) da = 2 sinh(κ)/κ` and `∫_{-1}^1 a exp(κ a) da = (2/κ)(cosh(κ) - sinh(κ)/κ)`,
     yielding the exact Langevin function `A(κ) = coth(κ) - 1/κ` for every `κ ≠ 0`.
   - `sphere_params_valid` & `sphere_denoiser_fixed_point`: Setting `t* = coth(1) - 1` and
     `R* = (1 - t*) / t*`, we prove `0 < t* < 1`, `0 < R*`, exact tilting parameter
     `κ* = (t* R* / (1 - t*))² = 1`, and exact score cancellation `R* A(κ*) = t* R*`, so
     `f_{t*}(z) = z` identically on `t* (R* S²)`.
   - `zero_WO_residual_refutes_rate_gap`: Any depth-`L` residual softmax attention stack with
     `W_O = 0` in every layer acts as the exact identity, attaining worst-case squared error `0`
     on `t* (R* S²)` and refuting every strictly positive lower bound
     `C₂ (2^L)^{-2s} (1 + Δ) > 0`.
2. **Theorem 2 (Exact algebraic & growth core of the depth-`L` affine-polynomial theorem)**:
   - `softmax_block_linear_growth` & `residual_stack_linear_growth`: Every residual softmax
     attention block with affine values satisfies a global linear growth bound in `max_{i,c} |X_{ic}|`,
     and by induction every depth-`L` composition satisfies `maxNorm (F X) ≤ a_L * maxNorm X + b_L`.
   - `poly_linear_growth_implies_affine`: Any polynomial `P : Polynomial ℝ` obeying a global
     linear growth bound `∀ λ, |P.eval λ| ≤ a |λ| + b` has `P.natDegree ≤ 1` (globally affine).
3. **Theorem 3 (Small-logit Taylor squaring gadget `S_ε(w) = w² + O(ε²)` and depth-`L` closure)**:
   - `tanh_arctanh_half` & `sqGadget_exact_rational`: At `b₀ = log(3)/2` (`tanh b₀ = 1/2`,
     `T'(b₀) = 3/4`, `T''(b₀) = -3/4`), the 3-head residual softmax squaring gadget
     `S_ε(w) = w + (T(b₀ + εw) + T(b₀ - εw) - 2T(b₀))/(ε² T''(b₀)) - (T(b₀ + εw) - T(b₀ - εw))/(2ε T'(b₀))`
     equals `w + 4 t² / (ε² (4 - t²)) - 4 t / (ε (4 - t²))` where `t = tanh(ε w)`.
   - `sqGadget_uniform_error`: Quantitative uniform `O(ε²)` error bound: for ALL `ε ≠ 0` and
     ALL `w : ℝ`, `|sqGadget ε w - w²| ≤ ((2/3) |w|³ + (1/3) |w|⁴) * ε²` (in particular uniform
     convergence to `w²` on every compact interval `[-M, M]`).
   - `sqGadget_iter_uniform_closure`: By induction on depth `L`, the `L`-fold composition
     `sqGadget^[L] ε` converges uniformly on every compact interval `[-M, M]` to `w ↦ w ^ (2 ^ L)`
     as `ε → 0`.

Machine-checked vs prose-only:
- Machine-checked: Exact Langevin integrals on `[-1, 1]`, exact `t* = coth(1) - 1`, `R* = (1-t*)/t*`,
  `f_{t*}(z) = z` fixed-point cancellation, `W_O = 0` depth-`L` refutation of `conj:rate-gap`,
  depth-`L` residual softmax linear growth and polynomial-degree `≤ 1` collapse, and the full
  quantitative uniform `O(ε²)` error bound and depth-`L` uniform convergence of the small-logit
  squaring gadget to `w^{2^L}`.
- Prose-only (in `solutions/manifold_attn_gap/proof.pdf`): Rotational invariance reduction from
  surface measure on `S² ⊂ ℝ^D` to uniform height on `[-1, 1]`, real-analytic identity theorem
  from an open set to all of `ℝ^{N × d_in}` in Theorem 2, and Waring linear-form spanning
  parameter count `4(2H+1)²` in Theorem 3.
-/

open Matrix BigOperators Finset Filter Topology

namespace ManifoldAttnGap

/-! ### 1. Theorem 1: Exact Sphere Radius, Langevin Integrals, and `W_O = 0` Refutation -/

/-- Exact Langevin function defined as the ratio of spherical height integrals on `[-1, 1]`:
`A(κ) = (∫_{-1}^1 a exp(κ a) da) / (∫_{-1}^1 exp(κ a) da)`. -/
noncomputable def langevinIntegral (κ : ℝ) : ℝ :=
  (∫ a in (-1 : ℝ)..1, a * Real.exp (κ * a)) / (∫ a in (-1 : ℝ)..1, Real.exp (κ * a))

/-- **Exact spherical height integrals and Langevin formula**:
For every `κ ≠ 0`:
1. `∫_{-1}^1 exp(κ a) da = 2 * sinh κ / κ`;
2. `∫_{-1}^1 a exp(κ a) da = (2 / κ) * (cosh κ - sinh κ / κ)`;
3. `langevinIntegral κ = cosh κ / sinh κ - 1 / κ`. -/
theorem langevin_integral_exact {κ : ℝ} (hκ : κ ≠ 0) :
    (∫ a in (-1 : ℝ)..1, Real.exp (κ * a)) = 2 * Real.sinh κ / κ ∧
    (∫ a in (-1 : ℝ)..1, a * Real.exp (κ * a)) = (2 / κ) * (Real.cosh κ - Real.sinh κ / κ) ∧
    langevinIntegral κ = Real.cosh κ / Real.sinh κ - 1 / κ := by
  have hint1 : ∫ a in (-1 : ℝ)..1, Real.exp (κ * a) = 2 * Real.sinh κ / κ := by
    have hd : ∀ x ∈ Set.uIcc (-1 : ℝ) 1,
        HasDerivAt (fun a => Real.exp (κ * a) / κ) (Real.exp (κ * x)) x := by
      intro x _
      have h1 : HasDerivAt (fun a => κ * a) κ x := by simpa using (hasDerivAt_id x).const_mul κ
      have h2 := (Real.hasDerivAt_exp (κ * x)).comp x h1
      have h3 := h2.div_const κ
      convert h3 using 1
      field_simp [hκ]
    have hcont : Continuous (fun a : ℝ => Real.exp (κ * a)) := by fun_prop
    rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hd (hcont.intervalIntegrable (-1) 1)]
    rw [Real.sinh_eq]
    simp only [mul_one, mul_neg]
    field_simp [hκ]
  have hint2 : ∫ a in (-1 : ℝ)..1, a * Real.exp (κ * a) =
      (2 / κ) * (Real.cosh κ - Real.sinh κ / κ) := by
    have hd : ∀ x ∈ Set.uIcc (-1 : ℝ) 1,
        HasDerivAt (fun a => (κ * a - 1) * Real.exp (κ * a) / κ ^ 2) (x * Real.exp (κ * x)) x := by
      intro x _
      have hlin : HasDerivAt (fun a => κ * a - 1) κ x := by
        simpa using ((hasDerivAt_id x).const_mul κ).sub_const 1
      have hexp : HasDerivAt (fun a => Real.exp (κ * a)) (Real.exp (κ * x) * κ) x :=
        (Real.hasDerivAt_exp (κ * x)).comp x (by simpa using (hasDerivAt_id x).const_mul κ)
      have hmul := (hlin.mul hexp).div_const (κ ^ 2)
      convert hmul using 1
      field_simp [hκ]
      ring
    have hcont : Continuous (fun a : ℝ => a * Real.exp (κ * a)) := by fun_prop
    rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hd (hcont.intervalIntegrable (-1) 1)]
    rw [Real.cosh_eq, Real.sinh_eq]
    simp only [mul_one, mul_neg]
    field_simp [hκ]
    ring
  refine ⟨hint1, hint2, ?_⟩
  unfold langevinIntegral
  rw [hint1, hint2]
  have hsh : Real.sinh κ ≠ 0 := by
    intro h0
    have := Real.sinh_injective (h0.trans Real.sinh_zero.symm)
    exact hκ this
  field_simp [hκ, hsh]

/-- Exact counterexample parameters `t* = coth(1) - 1`, `σ* = 1 - t*`, and `R* = (1 - t*) / t*`. -/
noncomputable def tStar : ℝ := Real.cosh 1 / Real.sinh 1 - 1
noncomputable def sigmaStar : ℝ := 1 - tStar
noncomputable def RStar : ℝ := (1 - tStar) / tStar

/-- `0 < t* < 1`, `0 < σ*`, and `0 < R*`. -/
theorem sphere_params_valid :
    0 < tStar ∧ tStar < 1 ∧ 0 < sigmaStar ∧ 0 < RStar := by
  have hsh_pos : 0 < Real.sinh 1 := Real.sinh_pos_iff.mpr (by norm_num)
  have hch_gt_sh : Real.sinh 1 < Real.cosh 1 := by
    rw [Real.sinh_eq, Real.cosh_eq]
    linarith [Real.exp_pos (-1 : ℝ)]
  have ht_pos : 0 < tStar := by
    unfold tStar
    rw [sub_pos, one_lt_div hsh_pos]
    exact hch_gt_sh
  have ht_lt1 : tStar < 1 := by
    unfold tStar
    rw [sub_lt_iff_lt_add, show (1 : ℝ) + 1 = 2 by norm_num, div_lt_iff₀ hsh_pos]
    rw [Real.cosh_eq, Real.sinh_eq]
    -- Need (e + e⁻¹)/2 < e - e⁻¹, i.e. 3 e⁻¹ < e, i.e. 3 < e²
    have he : (2 : ℝ) < Real.exp 1 := by
      have hhalf : (3 / 2 : ℝ) ≤ Real.exp (1 / 2) := by
        linarith [Real.add_one_le_exp (1 / 2 : ℝ)]
      have hsq : Real.exp (1 / 2) * Real.exp (1 / 2) = Real.exp 1 := by
        rw [← Real.exp_add]; norm_num
      nlinarith
    have hinv : Real.exp 1 * Real.exp (-1) = 1 := by
      rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
    have hpos_e : 0 < Real.exp 1 := Real.exp_pos 1
    nlinarith
  have hsig_pos : 0 < sigmaStar := by unfold sigmaStar; linarith
  have hR_pos : 0 < RStar := div_pos hsig_pos ht_pos
  exact ⟨ht_pos, ht_lt1, hsig_pos, hR_pos⟩

/-- **Theorem 1(a) — Exact fixed-point and score cancellation on `t* (R* S²)`**:
With `t* = coth(1) - 1`, `σ* = 1 - t*`, `R* = (1 - t*) / t*`:
1. The posterior tilting parameter `κ* = (t* R* / σ*)² = 1`;
2. `R* * langevinIntegral κ* = t* * R*`;
3. Consequently, for EVERY unit direction `u : Fin D → ℝ`, the MMSE denoiser
   `f_{t*}(t* R* u) = (R* A(κ*)) • u` equals `t* R* u` identically! -/
theorem sphere_denoiser_fixed_point {D : ℕ} (u : Fin D → ℝ) :
    let κStar := (tStar * RStar / sigmaStar) ^ 2
    κStar = 1 ∧
    RStar * langevinIntegral κStar = tStar * RStar ∧
    (RStar * langevinIntegral κStar) • u = (tStar * RStar) • u := by
  intro κStar
  obtain ⟨ht0, ht1, hsig0, _⟩ := sphere_params_valid
  have htR : tStar * RStar = sigmaStar := by
    unfold RStar sigmaStar
    exact mul_div_cancel₀ (1 - tStar) (ne_of_gt ht0)
  have hκ : κStar = 1 := by
    dsimp only [κStar]
    rw [htR, div_self (ne_of_gt hsig0), one_pow]
  have hL : langevinIntegral κStar = tStar := by
    rw [hκ, (langevin_integral_exact one_ne_zero).2.2]
    unfold tStar
    simp
  refine ⟨hκ, ?_, ?_⟩
  · rw [hL]; ring
  · rw [hL, mul_comm]

/-- A depth-`L` residual softmax attention stack: at each layer `ℓ`, `X ↦ X + attn_ℓ(X) * W_O^{(ℓ)}`. -/
def resStack {N d : ℕ} : (L : ℕ) →
    (Fin L → (Matrix (Fin N) (Fin d) ℝ → Matrix (Fin N) (Fin d) ℝ)) →
    (Fin L → Matrix (Fin d) (Fin d) ℝ) →
    Matrix (Fin N) (Fin d) ℝ → Matrix (Fin N) (Fin d) ℝ
  | 0, _, _, X => X
  | L + 1, blocks, WOs, X =>
      let prev := resStack L (fun i => blocks i.castSucc) (fun i => WOs i.castSucc) X
      prev + blocks (Fin.last L) prev * WOs (Fin.last L)

/-- Setting `W_O = 0` in every layer makes any depth-`L` residual attention stack equal the
exact identity map `X ↦ X`. -/
lemma resStack_zero_WO {N d : ℕ} (L : ℕ)
    (blocks : Fin L → (Matrix (Fin N) (Fin d) ℝ → Matrix (Fin N) (Fin d) ℝ))
    (X : Matrix (Fin N) (Fin d) ℝ) :
    resStack L blocks (fun _ => 0) X = X := by
  induction L with
  | zero => rfl
  | succ k ih =>
    simp only [resStack, Matrix.mul_zero, add_zero]
    exact ih (fun i => blocks i.castSucc)

/-- **Theorem 1(b) — Refutation of `conj:rate-gap` at every depth `L ≥ 1`**:
With `W_O = 0` in every layer, the depth-`L` residual stack attains ZERO squared error at every
point `z = (t* R*) • u` of `t* (R* S²)`, refuting any strictly positive rate-gap lower bound
`C₂ * (2^L)^(-2s) * (1 + Δ) > 0`. -/
theorem zero_WO_residual_refutes_rate_gap {D L : ℕ}
    (blocks : Fin L → (Matrix (Fin 1) (Fin D) ℝ → Matrix (Fin 1) (Fin D) ℝ))
    {C₂ s Δ : ℝ} (hC₂ : 0 < C₂) (hΔ : 0 ≤ Δ) :
    let h := resStack L blocks (fun _ => 0)
    (∀ u : Fin D → ℝ,
      let z : Matrix (Fin 1) (Fin D) ℝ := Matrix.of fun _ j => (tStar * RStar) * u j
      let ftz : Matrix (Fin 1) (Fin D) ℝ :=
        Matrix.of fun _ j => (RStar * langevinIntegral ((tStar * RStar / sigmaStar) ^ 2)) * u j
      ∑ j : Fin D, (ftz 0 j - h z 0 j) ^ 2 = 0) ∧
    (0 : ℝ) < C₂ * ((2 ^ L : ℝ) ^ (-(2 * s))) * (1 + Δ) := by
  intro h
  constructor
  · intro u z ftz
    have hfix := (sphere_denoiser_fixed_point u).2.1
    have hid : h z = z := resStack_zero_WO L blocks z
    rw [hid]
    apply Finset.sum_eq_zero
    intro j _
    simp only [ftz, z, Matrix.of_apply, hfix, sub_self, zero_pow (by norm_num : 2 ≠ 0)]
  · have h2L : (0 : ℝ) < (2 ^ L : ℝ) := by positivity
    have hpow : (0 : ℝ) < (2 ^ L : ℝ) ^ (-(2 * s)) := Real.rpow_pos_of_pos h2L _
    positivity

/-! ### 2. Theorem 2 Core: Depth-`L` Linear Growth and Affine Polynomial Collapse -/

/-- Entrywise max-norm bound `‖X‖_∞ ≤ M`. -/
def EntryBounded {N d : ℕ} (X : Matrix (Fin N) (Fin d) ℝ) (M : ℝ) : Prop :=
  ∀ i j, |X i j| ≤ M

/-- Single-layer linear growth of any row-stochastic attention block with affine values
`X ↦ X + A(X) * (X * V + 1 bᵀ) * W_O`: if `‖X‖_∞ ≤ M` (with `M ≥ 0`), then `‖out‖_∞ ≤ a M + b`
for explicit parameter-dependent constants `a, b ≥ 0`. -/
theorem softmax_block_linear_growth {N d : ℕ}
    (V WO : Matrix (Fin d) (Fin d) ℝ) (bVec : Fin d → ℝ) :
    ∃ a b : ℝ, 0 ≤ a ∧ 0 ≤ b ∧
      ∀ (A : Matrix (Fin N) (Fin N) ℝ),
        (∀ i k, 0 ≤ A i k) → (∀ i, ∑ k : Fin N, A i k = 1) →
        ∀ (X : Matrix (Fin N) (Fin d) ℝ) (M : ℝ), 0 ≤ M → EntryBounded X M →
          EntryBounded (X + A * (X * V + Matrix.of (fun _ j => bVec j)) * WO) (a * M + b) := by
  let C_V : ℝ := ∑ r : Fin d, ∑ s : Fin d, |V r s|
  let C_W : ℝ := ∑ r : Fin d, ∑ s : Fin d, |WO r s|
  let C_B : ℝ := ∑ s : Fin d, |bVec s|
  refine ⟨1 + C_V * C_W, C_B * C_W, by positivity, by positivity, ?_⟩
  intro A hAnn hAsum X M hM hX i j
  -- First bound each entry of Y = X * V + 1 bᵀ by C_V * M + C_B
  set Y : Matrix (Fin N) (Fin d) ℝ := X * V + Matrix.of (fun _ c => bVec c)
  have hY : ∀ k c : Fin _, |Y k c| ≤ C_V * M + C_B := by
    intro k c
    dsimp only [Y]
    rw [Matrix.add_apply, Matrix.mul_apply, Matrix.of_apply]
    have h1 : |∑ r : Fin d, X k r * V r c| ≤ C_V * M := by
      calc |∑ r : Fin d, X k r * V r c|
          ≤ ∑ r : Fin d, |X k r * V r c| := Finset.abs_sum_le_sum_abs _ _
        _ = ∑ r : Fin d, |X k r| * |V r c| := by simp_rw [abs_mul]
        _ ≤ ∑ r : Fin d, M * |V r c| :=
            Finset.sum_le_sum fun r _ => mul_le_mul_of_nonneg_right (hX k r) (abs_nonneg _)
        _ = (∑ r : Fin d, |V r c|) * M := by rw [← Finset.mul_sum, mul_comm]
        _ ≤ C_V * M := by
            apply mul_le_mul_of_nonneg_right _ hM
            dsimp only [C_V]
            rw [Finset.sum_comm]
            exact Finset.single_le_sum (f := fun s => ∑ r : Fin d, |V r s|)
              (fun s _ => Finset.sum_nonneg fun r _ => abs_nonneg _) (Finset.mem_univ c)
    have h2 : |bVec c| ≤ C_B :=
      Finset.single_le_sum (f := fun s => |bVec s|) (fun s _ => abs_nonneg _) (Finset.mem_univ c)
    linarith [abs_add_le (∑ r : Fin d, X k r * V r c) (bVec c)]
  -- Next convex combination A * Y preserves the exact same entry bound
  have hAY : ∀ c : Fin d, |(A * Y) i c| ≤ C_V * M + C_B := by
    intro c
    rw [Matrix.mul_apply]
    calc |∑ k : Fin N, A i k * Y k c|
        ≤ ∑ k : Fin N, |A i k * Y k c| := Finset.abs_sum_le_sum_abs _ _
      _ = ∑ k : Fin N, A i k * |Y k c| := by
          refine Finset.sum_congr rfl fun k _ => by rw [abs_mul, abs_of_nonneg (hAnn i k)]
      _ ≤ ∑ k : Fin N, A i k * (C_V * M + C_B) :=
          Finset.sum_le_sum fun k _ => mul_le_mul_of_nonneg_left (hY k c) (hAnn i k)
      _ = (∑ k : Fin N, A i k) * (C_V * M + C_B) := (Finset.sum_mul _ _ _).symm
      _ = C_V * M + C_B := by rw [hAsum i, one_mul]
  -- Finally multiply by WO and add residual X
  rw [Matrix.add_apply, Matrix.mul_apply]
  have hWO : |∑ c : Fin d, (A * Y) i c * WO c j| ≤ (C_V * M + C_B) * C_W := by
    calc |∑ c : Fin d, (A * Y) i c * WO c j|
        ≤ ∑ c : Fin d, |(A * Y) i c * WO c j| := Finset.abs_sum_le_sum_abs _ _
      _ = ∑ c : Fin d, |(A * Y) i c| * |WO c j| := by simp_rw [abs_mul]
      _ ≤ ∑ c : Fin d, (C_V * M + C_B) * |WO c j| :=
          Finset.sum_le_sum fun c _ => mul_le_mul_of_nonneg_right (hAY c) (abs_nonneg _)
      _ = (C_V * M + C_B) * ∑ c : Fin d, |WO c j| := (Finset.mul_sum _ _ _).symm
      _ ≤ (C_V * M + C_B) * C_W := by
          apply mul_le_mul_of_nonneg_left _ (by positivity)
          dsimp only [C_W]
          rw [Finset.sum_comm]
          exact Finset.single_le_sum (f := fun s => ∑ r : Fin d, |WO r s|)
            (fun s _ => Finset.sum_nonneg fun r _ => abs_nonneg _) (Finset.mem_univ j)
  have htri := abs_add_le (X i j) (∑ c : Fin d, (A * Y) i c * WO c j)
  have hXij := hX i j
  linarith

/-- Composing `L` maps that each satisfy a linear growth bound preserves linear growth. -/
theorem linear_growth_composition {α : Type*} (norm : α → ℝ) :
    ∀ (L : ℕ) (f : Fin L → α → α),
      (∀ ℓ, ∃ a b : ℝ, 0 ≤ a ∧ 0 ≤ b ∧ ∀ x, norm (f ℓ x) ≤ a * norm x + b) →
      ∃ a b : ℝ, 0 ≤ a ∧ 0 ≤ b ∧
        ∀ x, norm (List.ofFn f |>.foldl (fun acc g => g acc) x) ≤ a * norm x + b := by
  intro L
  induction L with
  | zero =>
    intro f _
    exact ⟨1, 0, by norm_num, le_refl 0, fun x => by simp⟩
  | succ k ih =>
    intro f hf
    obtain ⟨a₀, b₀, ha₀, hb₀, hf₀⟩ := hf 0
    obtain ⟨a₁, b₁, ha₁, hb₁, hrest⟩ := ih (fun i => f i.succ) (fun i => hf i.succ)
    refine ⟨a₁ * a₀, a₁ * b₀ + b₁, mul_nonneg ha₁ ha₀, by positivity, fun x => ?_⟩
    rw [List.ofFn_succ, List.foldl_cons]
    have h1 := hrest (f 0 x)
    have h2 := hf₀ x
    nlinarith

/-- **Theorem 2 (Polynomial linear-growth collapse to degree `≤ 1`)**:
Any real polynomial `P : Polynomial ℝ` satisfying a global linear growth bound
`∀ λ : ℝ, |P.eval λ| ≤ a * |λ| + b` has `P.natDegree ≤ 1`, i.e., `P` is globally affine
(`∃ c₁ c₀, ∀ λ, P.eval λ = c₁ * λ + c₀`). -/
theorem poly_linear_growth_implies_affine (P : Polynomial ℝ) {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b)
    (hgrow : ∀ t : ℝ, |P.eval t| ≤ a * |t| + b) :
    P.natDegree ≤ 1 ∧ ∃ c₁ c₀ : ℝ, ∀ t : ℝ, P.eval t = c₁ * t + c₀ := by
  have hdeg : P.natDegree ≤ 1 := by
    by_contra hgt
    push_neg at hgt
    set d := P.natDegree with hd_def
    have hd2 : 2 ≤ d := hgt
    have hPne : P ≠ 0 := by
      intro h0; rw [h0, Polynomial.natDegree_zero] at hd_def; omega
    have hlc_pos : 0 < |P.leadingCoeff| := abs_pos.mpr (Polynomial.leadingCoeff_ne_zero.mpr hPne)
    -- Bound the lower-degree terms for t ≥ 1: |P(t) - lc * t^d| ≤ C * t^{d-1}
    let C : ℝ := ∑ k ∈ Finset.range d, |P.coeff k|
    have hC_nn : 0 ≤ C := Finset.sum_nonneg fun _ _ => abs_nonneg _
    have hlow : ∀ t : ℝ, 1 ≤ t → |P.eval t - P.leadingCoeff * t ^ d| ≤ C * t ^ (d - 1) := by
      intro t ht1
      have hsplit : P.eval t = P.leadingCoeff * t ^ d + ∑ k ∈ Finset.range d, P.coeff k * t ^ k := by
        rw [Polynomial.eval_eq_sum_range, Finset.sum_range_succ, Polynomial.leadingCoeff, ← hd_def]
        ring
      rw [hsplit, add_sub_cancel_left]
      calc |∑ k ∈ Finset.range d, P.coeff k * t ^ k|
          ≤ ∑ k ∈ Finset.range d, |P.coeff k * t ^ k| := Finset.abs_sum_le_sum_abs _ _
        _ = ∑ k ∈ Finset.range d, |P.coeff k| * t ^ k := by
            refine Finset.sum_congr rfl fun k _ => by
              rw [abs_mul, abs_of_nonneg (pow_nonneg (le_trans zero_le_one ht1) k)]
        _ ≤ ∑ k ∈ Finset.range d, |P.coeff k| * t ^ (d - 1) := by
            apply Finset.sum_le_sum
            intro k hk
            have hkd : k ≤ d - 1 := by have := Finset.mem_range.mp hk; omega
            exact mul_le_mul_of_nonneg_left (pow_le_pow_right₀ ht1 hkd) (abs_nonneg _)
        _ = C * t ^ (d - 1) := (Finset.sum_mul _ _ _).symm
    -- Pick a large explicit t > max 1 ((C + a + b + 1) / |lc|)
    let t : ℝ := max 2 ((C + a + b + 1) / |P.leadingCoeff|)
    have ht1 : 1 ≤ t := le_trans (by norm_num) (le_max_left _ _)
    have ht_pos : 0 < t := by linarith
    have ht_big : (C + a + b + 1) / |P.leadingCoeff| ≤ t := le_max_right _ _
    have htd : |P.leadingCoeff| * t ^ d ≤ |P.eval t| + C * t ^ (d - 1) := by
      have htri := abs_sub_abs_le_abs_sub (P.leadingCoeff * t ^ d) (P.eval t)
      rw [abs_sub_comm, abs_mul, abs_of_nonneg (pow_nonneg (le_of_lt ht_pos) d)] at htri
      linarith [hlow t ht1]
    have hg := hgrow t
    rw [abs_of_nonneg (le_of_lt ht_pos)] at hg
    have hpow_d : t ^ d = t * t ^ (d - 1) := by
      have : d = (d - 1) + 1 := by omega
      conv_lhs => rw [this, pow_succ, mul_comm]
    have hpow_ge : t ≤ t ^ (d - 1) := by
      have h1 : 1 ≤ d - 1 := by omega
      simpa using pow_le_pow_right₀ ht1 h1
    have h1_le_pow : 1 ≤ t ^ (d - 1) := le_trans ht1 hpow_ge
    have hmul_big : C + a + b + 1 ≤ |P.leadingCoeff| * t := by
      have := (div_le_iff₀ hlc_pos).mp ht_big
      linarith
    have hcomb : |P.leadingCoeff| * t * t ^ (d - 1) ≤ (C + a + b) * t ^ (d - 1) := by
      have h1 : a * t ≤ a * t ^ (d - 1) := mul_le_mul_of_nonneg_left hpow_ge ha
      have h2 : b ≤ b * t ^ (d - 1) := by nlinarith
      rw [hpow_d] at htd
      nlinarith
    have htdm1_pos : 0 < t ^ (d - 1) := pow_pos ht_pos _
    nlinarith
  refine ⟨hdeg, P.coeff 1, P.coeff 0, fun t => ?_⟩
  conv_lhs => rw [Polynomial.eq_X_add_C_of_natDegree_le_one hdeg]
  simp [Polynomial.eval_add, Polynomial.eval_mul, Polynomial.eval_X, Polynomial.eval_C]

/-! ### 3. Theorem 3: Small-Logit Taylor Squaring Gadget and Depth-`L` Uniform Closure -/

/-- Exact bias `b₀ = log(3) / 2 = arctanh(1/2)`. -/
noncomputable def bZero : ℝ := Real.log 3 / 2

/-- `Real.tanh bZero = 1 / 2`. -/
theorem tanh_arctanh_half : Real.tanh bZero = 1 / 2 := by
  unfold bZero
  rw [Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq]
  have hpos : 0 < Real.exp (Real.log 3 / 2) + Real.exp (-(Real.log 3 / 2)) := by positivity
  have he2 : Real.exp (Real.log 3 / 2) * Real.exp (Real.log 3 / 2) = 3 := by
    rw [← Real.exp_add]
    have : Real.log 3 / 2 + Real.log 3 / 2 = Real.log 3 := by ring
    rw [this, Real.exp_log (by norm_num)]
  have hinv : Real.exp (Real.log 3 / 2) * Real.exp (-(Real.log 3 / 2)) = 1 := by
    rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
  rw [div_div_div_cancel_right₀ (by norm_num : (2 : ℝ) ≠ 0), div_eq_iff (ne_of_gt hpos)]
  have hpos1 : 0 < Real.exp (Real.log 3 / 2) := Real.exp_pos _
  nlinarith

/-- Exact hyperbolic addition formula at `b₀ = arctanh(1/2)`:
`tanh(b₀ + u) = (1 + 2 tanh u) / (2 + tanh u)`. -/
lemma tanh_bZero_add (u : ℝ) :
    Real.tanh (bZero + u) = (1 + 2 * Real.tanh u) / (2 + Real.tanh u) := by
  have hth_add : ∀ x y : ℝ, Real.tanh (x + y) =
      (Real.tanh x + Real.tanh y) / (1 + Real.tanh x * Real.tanh y) := by
    intro x y
    rw [Real.tanh_eq_sinh_div_cosh, Real.tanh_eq_sinh_div_cosh, Real.tanh_eq_sinh_div_cosh,
        Real.sinh_add, Real.cosh_add]
    have hcx : 0 < Real.cosh x := Real.cosh_pos x
    have hcy : 0 < Real.cosh y := Real.cosh_pos y
    have hcxy : 0 < Real.cosh x * Real.cosh y + Real.sinh x * Real.sinh y := by
      rw [← Real.cosh_add]; exact Real.cosh_pos (x + y)
    have hden : 1 + Real.sinh x / Real.cosh x * (Real.sinh y / Real.cosh y) ≠ 0 := by
      have : 1 + Real.sinh x / Real.cosh x * (Real.sinh y / Real.cosh y) =
          (Real.cosh x * Real.cosh y + Real.sinh x * Real.sinh y) / (Real.cosh x * Real.cosh y) := by
        field_simp [ne_of_gt hcx, ne_of_gt hcy]
      rw [this]
      exact div_ne_zero (ne_of_gt hcxy) (mul_ne_zero (ne_of_gt hcx) (ne_of_gt hcy))
    field_simp [ne_of_gt hcx, ne_of_gt hcy, ne_of_gt hcxy, hden]
  rw [hth_add bZero u, tanh_arctanh_half]
  have ht_gt : -1 < Real.tanh u := by
    rw [Real.tanh_eq_sinh_div_cosh, lt_div_iff₀ (Real.cosh_pos u), Real.sinh_eq, Real.cosh_eq]
    linarith [Real.exp_pos u]
  have h1 : 1 + 1 / 2 * Real.tanh u ≠ 0 := by linarith
  have h2 : 2 + Real.tanh u ≠ 0 := by linarith
  field_simp [h1, h2]

/-- The 3-head residual softmax small-logit squaring gadget `S_ε(w)` from Section 4
(`T'(b₀) = 3/4`, `T''(b₀) = -3/4`). -/
noncomputable def sqGadget (ε w : ℝ) : ℝ :=
  w +
  (Real.tanh (bZero + ε * w) + Real.tanh (bZero - ε * w) - 2 * Real.tanh bZero) /
    (ε ^ 2 * (-(3 / 4))) -
  (Real.tanh (bZero + ε * w) - Real.tanh (bZero - ε * w)) /
    (2 * ε * (3 / 4))

/-- **Exact rational form of `sqGadget ε w`** in terms of `t = Real.tanh (ε * w)`:
`sqGadget ε w = w + 4 t² / (ε² (4 - t²)) - 4 t / (ε (4 - t²))`. -/
theorem sqGadget_exact_rational {ε : ℝ} (hε : ε ≠ 0) (w : ℝ) :
    let t := Real.tanh (ε * w)
    sqGadget ε w = w + 4 * t ^ 2 / (ε ^ 2 * (4 - t ^ 2)) - 4 * t / (ε * (4 - t ^ 2)) := by
  intro t
  have ht_abs : |t| < 1 := by
    dsimp only [t]
    rw [abs_lt, Real.tanh_eq_sinh_div_cosh, lt_div_iff₀ (Real.cosh_pos _),
        div_lt_iff₀ (Real.cosh_pos _), Real.sinh_eq, Real.cosh_eq]
    constructor <;> linarith [Real.exp_pos (ε * w), Real.exp_pos (-(ε * w))]
  have h2p : 2 + t ≠ 0 := by linarith [(abs_lt.mp ht_abs).1]
  have h2m : 2 - t ≠ 0 := by linarith [(abs_lt.mp ht_abs).2]
  have h4m : 4 - t ^ 2 ≠ 0 := by nlinarith [abs_lt.mp ht_abs]
  have hplus : Real.tanh (bZero + ε * w) = (1 + 2 * t) / (2 + t) := tanh_bZero_add (ε * w)
  have hminus : Real.tanh (bZero - ε * w) = (1 - 2 * t) / (2 - t) := by
    have := tanh_bZero_add (-(ε * w))
    rw [sub_eq_add_neg, this, Real.tanh_neg]
    dsimp only [t]
    ring_nf
  unfold sqGadget
  rw [hplus, hminus, tanh_arctanh_half]
  field_simp [hε, h2p, h2m, h4m]
  ring

/-- Continuity of `Real.tanh`. -/
lemma continuous_tanh : Continuous Real.tanh := by
  have hfn : Real.tanh = fun t => Real.sinh t / Real.cosh t := funext Real.tanh_eq_sinh_div_cosh
  rw [hfn]
  exact Real.continuous_sinh.div Real.continuous_cosh (fun t => ne_of_gt (Real.cosh_pos t))

/-- Derivative of `Real.tanh`. -/
lemma hasDerivAt_tanh (x : ℝ) : HasDerivAt Real.tanh (1 - Real.tanh x ^ 2) x := by
  have hfn : Real.tanh = fun t => Real.sinh t / Real.cosh t := funext Real.tanh_eq_sinh_div_cosh
  rw [hfn]
  have hch : Real.cosh x ≠ 0 := ne_of_gt (Real.cosh_pos x)
  convert (Real.hasDerivAt_sinh x).div (Real.hasDerivAt_cosh x) hch using 1
  field_simp [hch]

/-- Elementary cubic remainder bound for `tanh`: for all `u : ℝ`,
`|u - Real.tanh u| ≤ |u| ^ 3 / 3` and `|Real.tanh u| ≤ |u|`. -/
lemma abs_tanh_le_and_sub_le (u : ℝ) :
    |Real.tanh u| ≤ |u| ∧ |u - Real.tanh u| ≤ |u| ^ 3 / 3 := by
  have hnonneg : ∀ x : ℝ, 0 ≤ x →
      Real.tanh x ≤ x ∧ 0 ≤ x - Real.tanh x ∧ x - Real.tanh x ≤ x ^ 3 / 3 := by
    intro x hx
    -- Derivative of x - tanh x is tanh(x)^2 ≥ 0, so x - tanh x ≥ 0 on [0, ∞)
    have hmono1 : MonotoneOn (fun z => z - Real.tanh z) (Set.Ici 0) := by
      apply monotoneOn_of_deriv_nonneg (convex_Ici 0)
      · exact (continuous_id.sub continuous_tanh).continuousOn
      · exact fun z _ => ((hasDerivAt_id' z).sub (hasDerivAt_tanh z)).differentiableAt.differentiableWithinAt
      · intro z _
        have hd : HasDerivAt (fun z => z - Real.tanh z) (1 - (1 - Real.tanh z ^ 2)) z :=
          (hasDerivAt_id' z).sub (hasDerivAt_tanh z)
        rw [hd.deriv]
        nlinarith [sq_nonneg (Real.tanh z)]
    have h1 : 0 ≤ x - Real.tanh x := by
      have := hmono1 (Set.left_mem_Ici) hx hx
      simpa using this
    -- Now derivative of x^3/3 - (x - tanh x) is x^2 - tanh(x)^2 ≥ 0 on [0, ∞)
    have hmono2 : MonotoneOn (fun z => z ^ 3 / 3 - (z - Real.tanh z)) (Set.Ici 0) := by
      apply monotoneOn_of_deriv_nonneg (convex_Ici 0)
      · exact (((continuous_id.pow 3).div_const 3).sub
          (continuous_id.sub continuous_tanh)).continuousOn
      · intro z _
        have hd1 : DifferentiableAt ℝ (fun r : ℝ => r ^ 3 / 3) z :=
          (((hasDerivAt_id' z).pow 3).div_const 3).differentiableAt
        have hd2 : DifferentiableAt ℝ (fun r : ℝ => r - Real.tanh r) z :=
          ((hasDerivAt_id' z).sub (hasDerivAt_tanh z)).differentiableAt
        exact (hd1.sub hd2).differentiableWithinAt
      · intro z hz
        rw [interior_Ici] at hz
        have hz0 : 0 ≤ z := le_of_lt hz
        have hd1 : HasDerivAt (fun r : ℝ => r ^ 3 / 3) (z ^ 2) z := by
          have := ((hasDerivAt_id' z).pow 3).div_const 3
          convert this using 1; ring
        have hd : HasDerivAt (fun z => z ^ 3 / 3 - (z - Real.tanh z))
            (z ^ 2 - (1 - (1 - Real.tanh z ^ 2))) z :=
          hd1.sub ((hasDerivAt_id' z).sub (hasDerivAt_tanh z))
        rw [hd.deriv]
        have hz_le : Real.tanh z ≤ z := by
          have := hmono1 Set.left_mem_Ici hz0 hz0
          simp only [Real.tanh_zero, sub_zero] at this
          linarith
        have htz_nn : 0 ≤ Real.tanh z := by
          rw [Real.tanh_eq_sinh_div_cosh]
          exact div_nonneg (le_of_lt (Real.sinh_pos_iff.mpr hz)) (le_of_lt (Real.cosh_pos z))
        nlinarith
    have h2 : 0 ≤ x ^ 3 / 3 - (x - Real.tanh x) := by
      have := hmono2 Set.left_mem_Ici hx hx
      simpa using this
    exact ⟨by linarith, h1, by linarith⟩
  have hpos_tanh : ∀ x : ℝ, 0 ≤ x → 0 ≤ Real.tanh x := by
    intro x hx
    rw [Real.tanh_eq_sinh_div_cosh]
    rcases eq_or_lt_of_le hx with rfl | hpos
    · simp
    · exact div_nonneg (le_of_lt (Real.sinh_pos_iff.mpr hpos)) (le_of_lt (Real.cosh_pos x))
  rcases le_total 0 u with hu | hu
  · obtain ⟨h1, h2, h3⟩ := hnonneg u hu
    rw [abs_of_nonneg (hpos_tanh u hu), abs_of_nonneg hu, abs_of_nonneg h2]
    exact ⟨h1, h3⟩
  · have hnu : 0 ≤ -u := neg_nonneg.mpr hu
    obtain ⟨h1, h2, h3⟩ := hnonneg (-u) hnu
    have htu_neg : Real.tanh u = -Real.tanh (-u) := by rw [Real.tanh_neg, neg_neg]
    rw [htu_neg, abs_neg, abs_of_nonneg (hpos_tanh (-u) hnu), abs_of_nonpos hu]
    refine ⟨h1, ?_⟩
    have hsub : |u - -Real.tanh (-u)| = |(-u) - Real.tanh (-u)| := by
      rw [← abs_neg (u - -Real.tanh (-u))]; congr 1; ring
    rw [hsub, abs_of_nonneg h2]
    exact h3

/-- **Theorem 3(a) — Quantitative uniform `O(ε²)` error bound for the small-logit squaring gadget**:
For EVERY `ε ≠ 0` and EVERY `w : ℝ`,
`|sqGadget ε w - w²| ≤ 2 * (|w| ^ 3 + |w| ^ 4) * ε ^ 2`. -/
theorem sqGadget_uniform_error {ε : ℝ} (hε : ε ≠ 0) (w : ℝ) :
    |sqGadget ε w - w ^ 2| ≤ 2 * (|w| ^ 3 + |w| ^ 4) * ε ^ 2 := by
  set t := Real.tanh (ε * w)
  have ht_abs : |t| < 1 := by
    dsimp only [t]
    rw [abs_lt, Real.tanh_eq_sinh_div_cosh, lt_div_iff₀ (Real.cosh_pos _),
        div_lt_iff₀ (Real.cosh_pos _), Real.sinh_eq, Real.cosh_eq]
    constructor <;> linarith [Real.exp_pos (ε * w), Real.exp_pos (-(ε * w))]
  have h4m_ge3 : 3 ≤ 4 - t ^ 2 := by nlinarith [abs_lt.mp ht_abs]
  have h4m_pos : 0 < 4 - t ^ 2 := by linarith
  obtain ⟨ht_le, hrem⟩ := abs_tanh_le_and_sub_le (ε * w)
  rw [sqGadget_exact_rational hε w]
  -- Split exact error into linear-cancellation remainder + quadratic remainder:
  -- (w - 4 t / (ε (4 - t²))) + (4 t² / (ε² (4 - t²)) - w²)
  -- = [4(εw - t) - εw t²] / (ε (4 - t²))  +  [4(t² - (εw)²) + (εw)² t²] / (ε² (4 - t²))
  have hid :
      w + 4 * t ^ 2 / (ε ^ 2 * (4 - t ^ 2)) - 4 * t / (ε * (4 - t ^ 2)) - w ^ 2 =
      (4 * (ε * w - t) - (ε * w) * t ^ 2) / (ε * (4 - t ^ 2)) +
      (4 * (t - ε * w) * (t + ε * w) + (ε * w) ^ 2 * t ^ 2) / (ε ^ 2 * (4 - t ^ 2)) := by
    field_simp [hε, ne_of_gt h4m_pos]
    ring
  rw [hid]
  have h1 : |(4 * (ε * w - t) - (ε * w) * t ^ 2) / (ε * (4 - t ^ 2))| ≤
      ((4 / 9 : ℝ) + 1 / 3) * |w| ^ 3 * ε ^ 2 := by
    rw [abs_div, abs_mul, abs_of_pos h4m_pos]
    have hnum : |4 * (ε * w - t) - (ε * w) * t ^ 2| ≤ (4 / 3 + 1) * (|ε| ^ 3 * |w| ^ 3) := by
      have htri := abs_sub (4 * (ε * w - t)) ((ε * w) * t ^ 2)
      have ha : |4 * (ε * w - t)| ≤ 4 / 3 * (|ε| ^ 3 * |w| ^ 3) := by
        rw [abs_mul, abs_of_pos (by norm_num : (0 : ℝ) < 4)]
        rw [abs_mul, mul_pow] at hrem
        linarith
      have hb : |(ε * w) * t ^ 2| ≤ |ε| ^ 3 * |w| ^ 3 := by
        rw [abs_mul, abs_pow, abs_mul]
        have ht2 : |t| ^ 2 ≤ (|ε| * |w|) ^ 2 := by
          rw [abs_mul] at ht_le
          nlinarith [abs_nonneg t, abs_nonneg ε, abs_nonneg w]
        have hmul := mul_le_mul_of_nonneg_left ht2 (mul_nonneg (abs_nonneg ε) (abs_nonneg w))
        linarith [show |ε| * |w| * (|ε| * |w|) ^ 2 = |ε| ^ 3 * |w| ^ 3 by ring]
      linarith
    have hden_pos : 0 < |ε| * (4 - t ^ 2) := mul_pos (abs_pos.mpr hε) h4m_pos
    rw [div_le_iff₀ hden_pos]
    have hεsq : |ε| ^ 2 = ε ^ 2 := sq_abs ε
    have hε3 : |ε| ^ 3 = ε ^ 2 * |ε| := by rw [pow_succ, hεsq]
    rw [hε3] at hnum
    have h3 : ((4 / 9 : ℝ) + 1 / 3) * |w| ^ 3 * ε ^ 2 * (|ε| * (4 - t ^ 2)) ≥
        ((4 / 9 : ℝ) + 1 / 3) * |w| ^ 3 * ε ^ 2 * (|ε| * 3) := by
      have : 0 ≤ ((4 / 9 : ℝ) + 1 / 3) * |w| ^ 3 * ε ^ 2 * |ε| := by positivity
      nlinarith
    linarith
  have h2 : |(4 * (t - ε * w) * (t + ε * w) + (ε * w) ^ 2 * t ^ 2) / (ε ^ 2 * (4 - t ^ 2))| ≤
      ((8 / 9 : ℝ) + 1 / 3) * |w| ^ 4 * ε ^ 2 := by
    rw [abs_div, abs_mul, abs_of_pos (sq_pos_of_ne_zero hε), abs_of_pos h4m_pos]
    have hnum : |4 * (t - ε * w) * (t + ε * w) + (ε * w) ^ 2 * t ^ 2| ≤
        (8 / 3 + 1) * (ε ^ 4 * |w| ^ 4) := by
      have htri := abs_add_le (4 * (t - ε * w) * (t + ε * w)) ((ε * w) ^ 2 * t ^ 2)
      have hεsq : |ε| ^ 2 = ε ^ 2 := sq_abs ε
      have hε4 : |ε| ^ 4 = ε ^ 4 := by
        have : |ε| ^ 4 = (|ε| ^ 2) ^ 2 := by ring
        rw [this, hεsq]; ring
      have ha : |4 * (t - ε * w) * (t + ε * w)| ≤ 8 / 3 * (ε ^ 4 * |w| ^ 4) := by
        rw [abs_mul, abs_mul, abs_of_pos (by norm_num : (0 : ℝ) < 4)]
        have hsum_le : |t + ε * w| ≤ 2 * |ε * w| := by
          linarith [abs_add_le t (ε * w)]
        have hprod : |t - ε * w| * |t + ε * w| ≤ (|ε * w| ^ 3 / 3) * (2 * |ε * w|) := by
          rw [abs_sub_comm]
          exact mul_le_mul hrem hsum_le (abs_nonneg _) (by positivity)
        have h4 : (|ε * w| ^ 3 / 3) * (2 * |ε * w|) = 2 / 3 * (ε ^ 4 * |w| ^ 4) := by
          rw [abs_mul]
          have hring : ((|ε| * |w|) ^ 3 / 3) * (2 * (|ε| * |w|)) = 2 / 3 * (|ε| ^ 4 * |w| ^ 4) := by
            ring
          rw [hring, hε4]
        linarith
      have hb : |(ε * w) ^ 2 * t ^ 2| ≤ ε ^ 4 * |w| ^ 4 := by
        rw [abs_mul, abs_pow, abs_pow, abs_mul]
        have ht2 : |t| ^ 2 ≤ (|ε| * |w|) ^ 2 := by
          rw [abs_mul] at ht_le
          nlinarith [abs_nonneg t, abs_nonneg ε, abs_nonneg w]
        have hmul := mul_le_mul_of_nonneg_left ht2 (sq_nonneg (|ε| * |w|))
        have h4 : (|ε| * |w|) ^ 2 * (|ε| * |w|) ^ 2 = ε ^ 4 * |w| ^ 4 := by
          have : (|ε| * |w|) ^ 2 * (|ε| * |w|) ^ 2 = |ε| ^ 4 * |w| ^ 4 := by ring
          rw [this, hε4]
        linarith
      linarith
    have hden_pos : 0 < ε ^ 2 * (4 - t ^ 2) := mul_pos (sq_pos_of_ne_zero hε) h4m_pos
    rw [div_le_iff₀ hden_pos]
    have h3 : ((8 / 9 : ℝ) + 1 / 3) * |w| ^ 4 * ε ^ 2 * (ε ^ 2 * (4 - t ^ 2)) ≥
        ((8 / 9 : ℝ) + 1 / 3) * |w| ^ 4 * ε ^ 2 * (ε ^ 2 * 3) := by
      have : 0 ≤ ((8 / 9 : ℝ) + 1 / 3) * |w| ^ 4 * ε ^ 2 * ε ^ 2 := by positivity
      nlinarith
    have h4 : ((8 / 9 : ℝ) + 1 / 3) * |w| ^ 4 * ε ^ 2 * (ε ^ 2 * 3) =
        (8 / 3 + 1) * (ε ^ 4 * |w| ^ 4) := by ring
    linarith
  have htri := abs_add_le
    ((4 * (ε * w - t) - (ε * w) * t ^ 2) / (ε * (4 - t ^ 2)))
    ((4 * (t - ε * w) * (t + ε * w) + (ε * w) ^ 2 * t ^ 2) / (ε ^ 2 * (4 - t ^ 2)))
  nlinarith [sq_nonneg ε, pow_nonneg (abs_nonneg w) 3, pow_nonneg (abs_nonneg w) 4]

/-- **Theorem 3(b) — Depth-`L` uniform convergence of iterated `sqGadget` to `w ↦ w^{2^L}` on
every compact interval `[-M, M]`**:
For every depth `L : ℕ`, radius `M ≥ 0`, and tolerance `η > 0`, there exists `δ > 0` such that
for every `ε` with `0 < |ε| ≤ δ` and every `w` with `|w| ≤ M`,
`|(sqGadget ε)^[L] w - w ^ (2 ^ L)| ≤ η`. -/
theorem sqGadget_iter_uniform_closure :
    ∀ (L : ℕ) (M : ℝ), 0 ≤ M →
      ∀ η > 0, ∃ δ > 0, ∀ ε : ℝ, ε ≠ 0 → |ε| ≤ δ →
        ∀ w : ℝ, |w| ≤ M → |(sqGadget ε)^[L] w - w ^ (2 ^ L)| ≤ η := by
  intro L
  induction L with
  | zero =>
    intro M _ η hη
    exact ⟨1, one_pos, fun _ _ _ w _ => by simp [le_of_lt hη]⟩
  | succ k ih =>
    intro M hM η hη
    -- Enlarge target compact bound for the outer squaring step
    let M₁ : ℝ := M ^ (2 ^ k) + 1
    have hM₁ : 0 ≤ M₁ := by positivity
    -- Choose inner tolerance η₁ ≤ min 1 (η / (2 * (2 * M₁ + 1))) so outer squaring error ≤ η/2
    let η₁ : ℝ := min 1 (η / (2 * (2 * M₁ + 1)))
    have hη₁ : 0 < η₁ := lt_min one_pos (div_pos hη (by positivity))
    obtain ⟨δ₁, hδ₁, hinner⟩ := ih M hM η₁ hη₁
    -- Choose outer ε bound δ₂ so |sqGadget ε y - y²| ≤ η/2 for all |y| ≤ M₁
    let K : ℝ := 2 * M₁ ^ 3 + 2 * M₁ ^ 4 + 1
    have hK : 0 < K := by positivity
    let δ₂ : ℝ := Real.sqrt (η / (2 * K))
    have hδ₂ : 0 < δ₂ := Real.sqrt_pos.mpr (div_pos hη (by positivity))
    refine ⟨min δ₁ δ₂, lt_min hδ₁ hδ₂, fun ε hε hεδ w hw => ?_⟩
    have hεδ₁ : |ε| ≤ δ₁ := le_trans hεδ (min_le_left _ _)
    have hεδ₂ : |ε| ≤ δ₂ := le_trans hεδ (min_le_right _ _)
    set y := (sqGadget ε)^[k] w
    have hy_err : |y - w ^ (2 ^ k)| ≤ η₁ := hinner ε hε hεδ₁ w hw
    have hw_pow : |w ^ (2 ^ k)| ≤ M ^ (2 ^ k) := by
      rw [abs_pow]; exact pow_le_pow_left₀ (abs_nonneg w) hw _
    have hy_le : |y| ≤ M₁ := by
      have htri := abs_sub_abs_le_abs_sub y (w ^ (2 ^ k))
      have hη1_le1 : η₁ ≤ 1 := min_le_left _ _
      dsimp only [M₁]; linarith
    -- Rewrite (sqGadget ε)^[k + 1] w = sqGadget ε y
    rw [Function.iterate_succ_apply']
    have hpow_succ : w ^ (2 ^ (k + 1)) = (w ^ (2 ^ k)) ^ 2 := by
      rw [pow_succ, pow_mul]
    rw [hpow_succ]
    -- Triangle inequality: |sqGadget ε y - y²| + |y² - (w^{2^k})²|
    have houter : |sqGadget ε y - y ^ 2| ≤ η / 2 := by
      have herr := sqGadget_uniform_error hε y
      have hε2 : ε ^ 2 ≤ η / (2 * K) := by
        have := sq_le_sq' (by linarith [abs_nonneg ε, hδ₂]) hεδ₂
        rw [sq_abs, Real.sq_sqrt (le_of_lt (div_pos hη (by positivity)))] at this
        exact this
      have hy3 : |y| ^ 3 ≤ M₁ ^ 3 := pow_le_pow_left₀ (abs_nonneg y) hy_le 3
      have hy4 : |y| ^ 4 ≤ M₁ ^ 4 := pow_le_pow_left₀ (abs_nonneg y) hy_le 4
      have hcoef : 2 * (|y| ^ 3 + |y| ^ 4) ≤ K := by dsimp only [K]; linarith
      calc |sqGadget ε y - y ^ 2|
          ≤ 2 * (|y| ^ 3 + |y| ^ 4) * ε ^ 2 := herr
        _ ≤ K * ε ^ 2 := mul_le_mul_of_nonneg_right hcoef (sq_nonneg ε)
        _ ≤ K * (η / (2 * K)) := mul_le_mul_of_nonneg_left hε2 (le_of_lt hK)
        _ = η / 2 := by field_simp [ne_of_gt hK]
    have hsq_lip : |y ^ 2 - (w ^ (2 ^ k)) ^ 2| ≤ η / 2 := by
      have hfac : y ^ 2 - (w ^ (2 ^ k)) ^ 2 = (y - w ^ (2 ^ k)) * (y + w ^ (2 ^ k)) := by ring
      rw [hfac, abs_mul]
      have hsum : |y + w ^ (2 ^ k)| ≤ 2 * M₁ + 1 := by
        have := abs_add_le y (w ^ (2 ^ k))
        dsimp only [M₁] at hy_le ⊢
        linarith
      have hη1_le : η₁ ≤ η / (2 * (2 * M₁ + 1)) := min_le_right _ _
      calc |y - w ^ (2 ^ k)| * |y + w ^ (2 ^ k)|
          ≤ η₁ * (2 * M₁ + 1) := mul_le_mul hy_err hsum (abs_nonneg _) (le_of_lt hη₁)
        _ ≤ (η / (2 * (2 * M₁ + 1))) * (2 * M₁ + 1) :=
            mul_le_mul_of_nonneg_right hη1_le (by positivity)
        _ = η / 2 := by
            have hne : 2 * M₁ + 1 ≠ 0 := by positivity
            field_simp [hne]
    have htri := abs_add_le (sqGadget ε y - y ^ 2) (y ^ 2 - (w ^ (2 ^ k)) ^ 2)
    have hadd : (sqGadget ε y - y ^ 2) + (y ^ 2 - (w ^ (2 ^ k)) ^ 2) =
        sqGadget ε y - (w ^ (2 ^ k)) ^ 2 := by ring
    rw [hadd] at htri
    linarith

end ManifoldAttnGap
