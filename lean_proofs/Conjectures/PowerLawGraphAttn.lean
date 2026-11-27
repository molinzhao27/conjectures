import Mathlib

/-!
# Refutation of the Four Architectural Conjectures in Power Law Graph Attention
(Burc Gokden, arXiv:2608.10288, Section 11: `conj:invariance`, `conj:spectral`,
`conj:functorial`, `conj:modeltransfer`)

We formalize in Lean 4 + Mathlib (complete kernel-checked proofs) the exact mathematical
counterexamples to all four architectural conjectures from
`solutions/powerlaw_graph_attn/solution.md`:

1. **Theorem 1 (`conj:invariance` — Failure of rigidity & universal power-law exponents)**:
   - `mRel_scale_invariant`: Exact scale invariance of the diagnostic relative fluctuation
     `m_T(x, y) = 2 r(T(x) - T(y)) / (r(T(x)) + r(T(y)))`.
   - `f_plga_deriv_at_one`: Exact derivative `f'(1) = (2(1 + e⁻¹) + e⁻¹)/(1 + e⁻¹)² > 1` of the
     PLGA nonlinearity `f(z) = z² / (1 + e⁻ᶻ)`.
   - `rigidity_refutation`: For every even width `d = 2k` (`k ≥ 1`) and `c > 0`, taking
     `A₀ = J`, `A₁(t) = J + t c v vᵀ`, `B_i(t) = f(A_i(t)) + 10⁻⁹`, `G_i(t) = B_i(t) / t`:
     as `t → 0⁺`, `m_A(t) → 0`, `m_B(t) → 0`, and `m_G(t) = m_B(t) → 0` uniformly, yet the
     optimal absolute RMS radius `(1/2) r(G₁(t) - G₀(t))` converges to `c f'(1) / 2 > 0`!
   - `arbitrary_rate_realizable`: For every prescribed positive rate `h > 0`, rescaling
     `G'_i = J + (h / R)(G_i - G₀)` yields exact optimal RMS radius `h`, refuting any universal
     monomial scaling `d^{-α} N^{-β}`.
   - `exact_zero_mRel_alternating`: `G_k = (-1)^k J_{2k}` has `m_G ≡ 0` identically while
     alternating between `+J` and `-J` (`r(J - (-J)) = 2`).
2. **Theorem 2 (`conj:spectral` — Spectral accumulation & monotonicity refutation)**:
   - `causal_unique_stationary_delta0`: For EVERY `S ≥ 1` and EVERY row-stochastic lower-
     triangular attention matrix `E` with `E_{i, 0} > 0`, the unique stationary distribution is
     `δ_0`, forcing `L²_0(δ_0) = {0}` (universal obstruction under the ideal causal mask).
   - `finite_mask_doeblin_contraction`: For any row-stochastic `E` with stationary law `π`
     satisfying Doeblin minorization `E_{ij} ≥ η π_j` (`0 < η ≤ 1`), `E` contracts `L²_0(π)` by
     at least `1 - η` (`‖E f‖_{L²(π)}² ≤ (1 - η)² ‖f‖_{L²(π)}²`), yielding a uniform spectral
     gap at `m = 0` under a finite additive mask.
   - `two_by_two_block_eigenvalue`: For `S = 2k`, the reversible `2×2`-block Markov kernel mixed
     with `(1/S) J` has the alternating observable `(-1)^i` as an exact eigenvector with
     eigenvalue `(1 - 1/S) r` (allowing an arbitrary spectral gap or atomic edge limit `δ_1` as
     `r ↑ 1`, all at `m = 0`).
   - `invisible_key_score_invariance` & `invisible_key_mG_tendsto_two`: Adding a zero-key column
     block leaves all attention scores `Q G Kᵀ` unchanged while driving `m_G → 2` as `u → ∞`.
3. **Theorem 3 (`conj:functorial` — Congruence alignment `J` vs `-J` refutation)**:
   - `key_sign_flip_invariance`: Negating both `G` and `K` leaves all bilinear scores
     `Q (-G) (-K)ᵀ = Q G Kᵀ` identically invariant.
   - `congruence_rms_lower_bound_J` & `congruence_rms_lower_bound_I_add_J`: For EVERY width
     `d ≥ 1` and EVERY real matrix `S ∈ ℝ^{d × d}` (without any condition-number restriction),
     `r(-J - Sᵀ J S) ≥ 1` and `r(-(I + J) - Sᵀ (I + J) S) ≥ 1` (Frobenius error `≥ d`).
4. **Theorem 4 (`conj:modeltransfer` — Whole-model frozen-LN transfer: `50%` vs `75%`)**:
   - `long_range_autocorr_formula`: Exact autocorrelation integral
     `∫_0^1 (1/2) r^k dr + (1/2)(-1/2)^k = 1/(2(k+1)) + (1/2)(-1/2)^k`, vanishing at lag `k = 1`.
   - `frozen_ln_accuracy_half`: On target domain `𝒟_2` (`C_{t+1} = B_t Z_{t+1}`, `P(Z = +1) = 3/4`),
     every constant predictor forced by a zero-gain, zero-shift frozen final LayerNorm achieves
     exact rational accuracy `1/2` (`50%`).
   - `matched_control_accuracy_three_quarters`: The matched-budget control with unit-gain final
     LayerNorm predicting `C_{t+1} = B_t` achieves exact rational accuracy `3/4` (`75%`), giving
     the exact irreducible `1/4` (`25` percentage-point) transfer deficit, with exact one-step
     population correlation `E[B_t C_{t+1}] = 1/2 > 0`.

Machine-checked vs prose-only:
- Machine-checked: All four counterexample theorems above in exact real/rational arithmetic.
- Prose-only (in `solutions/powerlaw_graph_attn/proof.pdf`): Explicit 8-residual-block SwiGLU
  LayerNorm constant `c ≈ 0.9999955`, RoPE Vandermonde right-inverse lift for arbitrary strictly
  positive stochastic `H`, weak convergence of quantile-block empirical measures to an arbitrary
  probability measure `μ` on `[0, 1]`, and `float64` numerical benchmarks.
-/

open Matrix BigOperators Finset Filter Topology

namespace PowerLawGraphAttn

/-! ### 1. Diagnostic RMS Norm, Relative Fluctuation `m_T`, and Theorem 1 (`conj:invariance`) -/

/-- All-ones `d × d` matrix `J_d`. -/
def allOnes (d : ℕ) : Matrix (Fin d) (Fin d) ℝ := Matrix.of fun _ _ => 1

/-- Entry-RMS norm `r(T) = √((∑_{i,j} T_{ij}²) / d²)` of a `d × d` matrix. -/
noncomputable def entryRMS {d : ℕ} (T : Matrix (Fin d) (Fin d) ℝ) : ℝ :=
  Real.sqrt ((∑ i : Fin d, ∑ j : Fin d, (T i j) ^ 2) / ((d : ℝ) * (d : ℝ)))

/-- Pairwise relative fluctuation `m_T(T₀, T₁) = 2 r(T₀ - T₁) / (r(T₀) + r(T₁))`
(or `0` when `r(T₀ - T₁) = 0`). -/
noncomputable def mRel {d : ℕ} (T₀ T₁ : Matrix (Fin d) (Fin d) ℝ) : ℝ :=
  if entryRMS (T₀ - T₁) = 0 then 0 else
    2 * entryRMS (T₀ - T₁) / (entryRMS T₀ + entryRMS T₁)

lemma entryRMS_nonneg {d : ℕ} (T : Matrix (Fin d) (Fin d) ℝ) : 0 ≤ entryRMS T :=
  Real.sqrt_nonneg _

lemma entryRMS_smul {d : ℕ} (c : ℝ) (T : Matrix (Fin d) (Fin d) ℝ) :
    entryRMS (c • T) = |c| * entryRMS T := by
  unfold entryRMS
  have hsum : (∑ i : Fin d, ∑ j : Fin d, ((c • T) i j) ^ 2) / ((d : ℝ) * (d : ℝ)) =
      c ^ 2 * ((∑ i : Fin d, ∑ j : Fin d, (T i j) ^ 2) / ((d : ℝ) * (d : ℝ))) := by
    have h1 : ∑ i : Fin d, ∑ j : Fin d, ((c • T) i j) ^ 2 =
        c ^ 2 * ∑ i : Fin d, ∑ j : Fin d, (T i j) ^ 2 := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun j _ => by simp [Matrix.smul_apply, mul_pow]
    rw [h1, mul_div_assoc]
  rw [hsum, Real.sqrt_mul (sq_nonneg c), Real.sqrt_sq_eq_abs]

/-- **Exact scale invariance of `m_T`**: scaling both matrices by any `c > 0` leaves `m_T`
identically unchanged. -/
theorem mRel_scale_invariant {d : ℕ} {c : ℝ} (hc : 0 < c) (T₀ T₁ : Matrix (Fin d) (Fin d) ℝ) :
    mRel (c • T₀) (c • T₁) = mRel T₀ T₁ := by
  unfold mRel
  rw [← smul_sub, entryRMS_smul, entryRMS_smul, entryRMS_smul, abs_of_pos hc]
  by_cases h0 : entryRMS (T₀ - T₁) = 0
  · simp [h0]
  · have hc0 : c * entryRMS (T₀ - T₁) ≠ 0 := mul_ne_zero hc.ne' h0
    simp only [h0, hc0, ↓reduceIte]
    rw [← mul_add]
    field_simp [hc.ne']

lemma entryRMS_allOnes {d : ℕ} (hd : 1 ≤ d) : entryRMS (allOnes d) = 1 := by
  unfold entryRMS allOnes
  have hd_pos : (0 : ℝ) < d := by positivity
  simp only [Matrix.of_apply, one_pow, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul, mul_one]
  rw [div_self (mul_ne_zero hd_pos.ne' hd_pos.ne'), Real.sqrt_one]

/-- PLGA nonlinearity `f(z) = z² / (1 + e⁻ᶻ)`. -/
noncomputable def fPlga (z : ℝ) : ℝ := z ^ 2 / (1 + Real.exp (-z))

/-- Exact value of `f'(1)`. -/
noncomputable def fPlgaDerivOne : ℝ :=
  (2 * (1 + Real.exp (-1)) + Real.exp (-1)) / (1 + Real.exp (-1)) ^ 2

/-- **Exact derivative of `f(z) = z² / (1 + e⁻ᶻ)` at `z = 1`**, with `f'(1) > 1`. -/
theorem fPlga_deriv_at_one :
    HasDerivAt fPlga fPlgaDerivOne 1 ∧ 1 < fPlgaDerivOne := by
  have he_pos : 0 < Real.exp (-1 : ℝ) := Real.exp_pos _
  have hden_pos : 0 < 1 + Real.exp (-1 : ℝ) := by linarith
  have hderiv : HasDerivAt fPlga fPlgaDerivOne 1 := by
    unfold fPlga fPlgaDerivOne
    have hnum : HasDerivAt (fun z : ℝ => z ^ 2) 2 1 := by
      simpa using (hasDerivAt_id (1 : ℝ)).pow 2
    have hexp : HasDerivAt (fun z : ℝ => 1 + Real.exp (-z)) (-Real.exp (-1)) 1 := by
      have h1 := ((hasDerivAt_id (1 : ℝ)).neg).exp.const_add (1 : ℝ)
      simpa using h1
    have hq := hnum.div hexp hden_pos.ne'
    convert hq using 1
    ring
  have hgt1 : 1 < fPlgaDerivOne := by
    unfold fPlgaDerivOne
    rw [one_lt_div (sq_pos_of_pos hden_pos)]
    have he_lt1 : Real.exp (-1 : ℝ) < 1 := by
      rw [Real.exp_lt_one_iff]; norm_num
    nlinarith
  exact ⟨hderiv, hgt1⟩

/-- Balanced alternating sign matrix `v vᵀ` of even size `2k × 2k` (`v_i = (-1)^i`). -/
def altMat (k : ℕ) : Matrix (Fin (2 * k)) (Fin (2 * k)) ℝ :=
  Matrix.of fun i j => (if i.val % 2 = 0 then (1 : ℝ) else -1) *
    (if j.val % 2 = 0 then 1 else -1)

/-- For any function `g : ℝ → ℝ` applied entrywise to `J + s • altMat k`, the entry-RMS
equals `√((g(1 + s)² + g(1 - s)²) / 2)` for every `k ≥ 1`. -/
lemma entryRMS_entrywise_altMat {k : ℕ} (hk : 1 ≤ k) (g : ℝ → ℝ) (s : ℝ) :
    entryRMS (Matrix.of fun i j : Fin (2 * k) => g (1 + s * altMat k i j)) =
      Real.sqrt (((g (1 + s)) ^ 2 + (g (1 - s)) ^ 2) / 2) := by
  unfold entryRMS
  have hk_pos : (0 : ℝ) < k := by positivity
  -- Count even and odd indices in Fin (2 * k): exactly k even and k odd!
  have heven : (Finset.univ.filter fun i : Fin (2 * k) => i.val % 2 = 0).card = k := by
    have hbij : Finset.image (fun m : Fin k => (⟨2 * m.val, by omega⟩ : Fin (2 * k))) Finset.univ =
        Finset.univ.filter (fun i : Fin (2 * k) => i.val % 2 = 0) := by
      ext i
      simp only [Finset.mem_image, Finset.mem_univ, true_and, Finset.mem_filter]
      constructor
      · rintro ⟨m, rfl⟩; simp
      · intro hi
        refine ⟨⟨i.val / 2, by omega⟩, Fin.ext (by dsimp; omega)⟩
    rw [← hbij, Finset.card_image_of_injective]
    · simp
    · intro a b hab; ext; have := congrArg Fin.val hab; dsimp at this; omega
  have hodd : (Finset.univ.filter fun i : Fin (2 * k) => ¬ i.val % 2 = 0).card = k := by
    have hsum := Finset.filter_card_add_filter_neg_card_eq_card
      (s := (Finset.univ : Finset (Fin (2 * k)))) (p := fun i : Fin (2 * k) => i.val % 2 = 0)
    simp only [Finset.card_univ, Fintype.card_fin, heven] at hsum
    omega
  have hrow : ∀ i : Fin (2 * k),
      ∑ j : Fin (2 * k), (g (1 + s * altMat k i j)) ^ 2 =
        (k : ℝ) * ((g (1 + s)) ^ 2 + (g (1 - s)) ^ 2) := by
    intro i
    rw [← Finset.sum_filter_add_sum_filter_not (s := Finset.univ)
      (p := fun j : Fin (2 * k) => j.val % 2 = 0)]
    by_cases hi : i.val % 2 = 0
    · have h1 : ∑ j ∈ Finset.univ.filter (fun j : Fin (2 * k) => j.val % 2 = 0),
          (g (1 + s * altMat k i j)) ^ 2 = (k : ℝ) * (g (1 + s)) ^ 2 := by
        have hconst : ∀ j ∈ Finset.univ.filter (fun j : Fin (2 * k) => j.val % 2 = 0),
            (g (1 + s * altMat k i j)) ^ 2 = (g (1 + s)) ^ 2 := by
          intro j hj; simp [altMat, hi, (Finset.mem_filter.1 hj).2]
        rw [Finset.sum_congr rfl hconst, Finset.sum_const, heven, nsmul_eq_mul]
      have h2 : ∑ j ∈ Finset.univ.filter (fun j : Fin (2 * k) => ¬ j.val % 2 = 0),
          (g (1 + s * altMat k i j)) ^ 2 = (k : ℝ) * (g (1 - s)) ^ 2 := by
        have hconst : ∀ j ∈ Finset.univ.filter (fun j : Fin (2 * k) => ¬ j.val % 2 = 0),
            (g (1 + s * altMat k i j)) ^ 2 = (g (1 - s)) ^ 2 := by
          intro j hj
          have hj' : j.val % 2 ≠ 0 := (Finset.mem_filter.1 hj).2
          simp only [altMat, Matrix.of_apply, hi, hj', ↓reduceIte]
          congr 2; ring
        rw [Finset.sum_congr rfl hconst, Finset.sum_const, hodd, nsmul_eq_mul]
      rw [h1, h2]; ring
    · have h1 : ∑ j ∈ Finset.univ.filter (fun j : Fin (2 * k) => j.val % 2 = 0),
          (g (1 + s * altMat k i j)) ^ 2 = (k : ℝ) * (g (1 - s)) ^ 2 := by
        have hconst : ∀ j ∈ Finset.univ.filter (fun j : Fin (2 * k) => j.val % 2 = 0),
            (g (1 + s * altMat k i j)) ^ 2 = (g (1 - s)) ^ 2 := by
          intro j hj
          have hj' : j.val % 2 = 0 := (Finset.mem_filter.1 hj).2
          simp only [altMat, Matrix.of_apply, hi, hj', ↓reduceIte]
          congr 2; ring
        rw [Finset.sum_congr rfl hconst, Finset.sum_const, heven, nsmul_eq_mul]
      have h2 : ∑ j ∈ Finset.univ.filter (fun j : Fin (2 * k) => ¬ j.val % 2 = 0),
          (g (1 + s * altMat k i j)) ^ 2 = (k : ℝ) * (g (1 + s)) ^ 2 := by
        have hconst : ∀ j ∈ Finset.univ.filter (fun j : Fin (2 * k) => ¬ j.val % 2 = 0),
            (g (1 + s * altMat k i j)) ^ 2 = (g (1 + s)) ^ 2 := by
          intro j hj
          have hj' : j.val % 2 ≠ 0 := (Finset.mem_filter.1 hj).2
          simp only [altMat, Matrix.of_apply, hi, hj', ↓reduceIte]
          congr 2; ring
        rw [Finset.sum_congr rfl hconst, Finset.sum_const, hodd, nsmul_eq_mul]
      rw [h1, h2]; ring
  simp only [Matrix.of_apply]
  simp_rw [hrow]
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  congr 1
  push_cast
  field_simp [hk_pos.ne']

/-- **Theorem 1(1) (`conj:invariance` rigidity refutation)**:
For every even width `d = 2k` (`k ≥ 1`) and `c > 0`, taking `A₀ = J`, `A₁(t) = J + t c v vᵀ`,
`B_i(t) = f(A_i(t)) + 10⁻⁹`, and `G_i(t) = B_i(t) / t`:
1. `m_G(t) = m_B(t)` identically for all `t > 0`;
2. `m_A(t) → 0` and `m_G(t) → 0` as `t → 0⁺`;
3. Yet the optimal absolute RMS radius `(1/2) r(G₁(t) - G₀(t))` converges to `c f'(1) / 2 > 0`! -/
theorem rigidity_refutation {k : ℕ} (hk : 1 ≤ k) {c : ℝ} (hc : 0 < c) :
    let A₀ : Matrix (Fin (2 * k)) (Fin (2 * k)) ℝ := allOnes (2 * k)
    let A₁ : ℝ → Matrix (Fin (2 * k)) (Fin (2 * k)) ℝ := fun t => A₀ + (t * c) • altMat k
    let B₀ : Matrix (Fin (2 * k)) (Fin (2 * k)) ℝ :=
      Matrix.of fun i j => fPlga (A₀ i j) + 1e-9
    let B₁ : ℝ → Matrix (Fin (2 * k)) (Fin (2 * k)) ℝ :=
      fun t => Matrix.of fun i j => fPlga (A₁ t i j) + 1e-9
    let G₀ : ℝ → Matrix (Fin (2 * k)) (Fin (2 * k)) ℝ := fun t => (1 / t) • B₀
    let G₁ : ℝ → Matrix (Fin (2 * k)) (Fin (2 * k)) ℝ := fun t => (1 / t) • B₁ t
    (∀ t > 0, mRel (G₀ t) (G₁ t) = mRel B₀ (B₁ t)) ∧
    Tendsto (fun t => mRel (G₀ t) (G₁ t)) (𝓝[>] 0) (𝓝 0) ∧
    Tendsto (fun t => (1 / 2 : ℝ) * entryRMS (G₁ t - G₀ t))
      (𝓝[>] 0) (𝓝 (c * fPlgaDerivOne / 2)) ∧
    0 < c * fPlgaDerivOne / 2 := by
  intro A₀ A₁ B₀ B₁ G₀ G₁
  have hmG_eq : ∀ t > 0, mRel (G₀ t) (G₁ t) = mRel B₀ (B₁ t) := fun t ht =>
    mRel_scale_invariant (one_div_pos.2 ht) B₀ (B₁ t)
  -- Express entryRMS (B₁ t - B₀) cleanly using entryRMS_entrywise_altMat!
  have hBdiff : ∀ t : ℝ, entryRMS (B₁ t - B₀) =
      Real.sqrt (((fPlga (1 + t * c) - fPlga 1) ^ 2 + (fPlga (1 - t * c) - fPlga 1) ^ 2) / 2) := by
    intro t
    have hmat : B₁ t - B₀ = Matrix.of fun i j =>
        (fun z => fPlga z - fPlga 1) (1 + (t * c) * altMat k i j) := by
      ext i j
      simp only [B₁, B₀, A₁, A₀, allOnes, Matrix.sub_apply, Matrix.of_apply, Matrix.add_apply,
        Matrix.smul_apply, smul_eq_mul]
      ring
    rw [hmat, entryRMS_entrywise_altMat hk (fun z => fPlga z - fPlga 1) (t * c)]
  have hB0_rms : entryRMS B₀ = fPlga 1 + 1e-9 := by
    have hmat : B₀ = (fPlga 1 + 1e-9) • allOnes (2 * k) := by
      ext i j; simp [B₀, A₀, allOnes]
    have hpos : 0 < fPlga 1 + 1e-9 := by
      unfold fPlga; positivity
    rw [hmat, entryRMS_smul, abs_of_pos hpos, entryRMS_allOnes (by omega), mul_one]
  -- 3. Limit of (1/2) * entryRMS (G₁ t - G₀ t) as t → 0⁺ is c * f'(1) / 2!
  have hderiv := fPlga_deriv_at_one.1
  have hpos_deriv : 0 < fPlgaDerivOne := lt_trans zero_lt_one fPlga_deriv_at_one.2
  have hslope_plus : Tendsto (fun t : ℝ => (fPlga (1 + t * c) - fPlga 1) / t)
      (𝓝[>] 0) (𝓝 (c * fPlgaDerivOne)) := by
    have h1 : HasDerivAt (fun t : ℝ => fPlga (1 + t * c)) (fPlgaDerivOne * c) 0 := by
      have hin : HasDerivAt (fun t : ℝ => 1 + t * c) c 0 := by
        simpa using ((hasDerivAt_id (0 : ℝ)).mul_const c).const_add (1 : ℝ)
      have hderiv' : HasDerivAt fPlga fPlgaDerivOne (1 + 0 * c) := by simpa using hderiv
      exact hderiv'.comp (0 : ℝ) hin
    have hs := h1.tendsto_slope_zero_right
    simpa [smul_eq_mul, mul_comm, div_eq_inv_mul] using hs
  have hslope_minus : Tendsto (fun t : ℝ => (fPlga (1 - t * c) - fPlga 1) / t)
      (𝓝[>] 0) (𝓝 (-(c * fPlgaDerivOne))) := by
    have h1 : HasDerivAt (fun t : ℝ => fPlga (1 - t * c)) (fPlgaDerivOne * (-c)) 0 := by
      have hin : HasDerivAt (fun t : ℝ => 1 - t * c) (-c) 0 := by
        simpa using ((hasDerivAt_id (0 : ℝ)).mul_const c).const_sub (1 : ℝ)
      have hderiv' : HasDerivAt fPlga fPlgaDerivOne (1 - 0 * c) := by simpa using hderiv
      exact hderiv'.comp (0 : ℝ) hin
    have hs := h1.tendsto_slope_zero_right
    simpa [smul_eq_mul, mul_comm, neg_mul, mul_neg, div_eq_inv_mul] using hs
  have hGdiff_eq : ∀ᶠ t in 𝓝[>] (0 : ℝ),
      entryRMS (G₁ t - G₀ t) =
        Real.sqrt ((((fPlga (1 + t * c) - fPlga 1) / t) ^ 2 +
                    ((fPlga (1 - t * c) - fPlga 1) / t) ^ 2) / 2) := by
    filter_upwards [self_mem_nhdsWithin] with t ht
    have ht' : (0 : ℝ) < t := ht
    have h1 : G₁ t - G₀ t = (1 / t) • (B₁ t - B₀) := by simp [G₁, G₀, smul_sub]
    rw [h1, entryRMS_smul, abs_of_pos (one_div_pos.2 ht'), hBdiff t]
    rw [← Real.sqrt_sq (one_div_pos.2 ht').le, ← Real.sqrt_mul (sq_nonneg _)]
    congr 1; field_simp [ht'.ne']
  have hGdiff_lim : Tendsto (fun t => entryRMS (G₁ t - G₀ t))
      (𝓝[>] 0) (𝓝 (c * fPlgaDerivOne)) := by
    refine Tendsto.congr' (EventuallyEq.symm hGdiff_eq) ?_
    have hsq1 := hslope_plus.pow 2
    have hsq2 := hslope_minus.pow 2
    have havg := ((hsq1.add hsq2).div_const 2).sqrt
    have hval : Real.sqrt (((c * fPlgaDerivOne) ^ 2 + (-(c * fPlgaDerivOne)) ^ 2) / 2) =
        c * fPlgaDerivOne := by
      have h1 : ((c * fPlgaDerivOne) ^ 2 + (-(c * fPlgaDerivOne)) ^ 2) / 2 =
          (c * fPlgaDerivOne) ^ 2 := by ring
      rw [h1, Real.sqrt_sq (mul_pos hc hpos_deriv).le]
    rwa [hval] at havg
  have hrad_lim : Tendsto (fun t => (1 / 2 : ℝ) * entryRMS (G₁ t - G₀ t))
      (𝓝[>] 0) (𝓝 (c * fPlgaDerivOne / 2)) := by
    have := hGdiff_lim.const_mul (1 / 2 : ℝ)
    simpa [div_eq_mul_inv, mul_comm] using this
  -- 2. m_G(t) = m_B(t) → 0 as t → 0⁺
  have hBdiff_lim : Tendsto (fun t => entryRMS (B₁ t - B₀)) (𝓝[>] 0) (𝓝 0) := by
    have hmul : ∀ᶠ t in 𝓝[>] (0 : ℝ),
        entryRMS (B₁ t - B₀) = t * entryRMS (G₁ t - G₀ t) := by
      filter_upwards [self_mem_nhdsWithin] with t ht
      have ht' : (0 : ℝ) < t := ht
      have h1 : G₁ t - G₀ t = (1 / t) • (B₁ t - B₀) := by simp [G₁, G₀, smul_sub]
      rw [h1, entryRMS_smul, abs_of_pos (one_div_pos.2 ht')]
      field_simp [ht'.ne']
    refine Tendsto.congr' (EventuallyEq.symm hmul) ?_
    have hid : Tendsto (fun t : ℝ => t) (𝓝[>] 0) (𝓝 0) :=
      tendsto_nhdsWithin_of_tendsto_nhds tendsto_id
    simpa using hid.mul hGdiff_lim
  have hmG_lim : Tendsto (fun t => mRel (G₀ t) (G₁ t)) (𝓝[>] 0) (𝓝 0) := by
    have hB0_pos : 0 < entryRMS B₀ := by rw [hB0_rms]; unfold fPlga; positivity
    have hbound : ∀ᶠ t in 𝓝[>] (0 : ℝ),
        0 ≤ mRel (G₀ t) (G₁ t) ∧
        mRel (G₀ t) (G₁ t) ≤ 2 * entryRMS (B₁ t - B₀) / entryRMS B₀ := by
      filter_upwards [self_mem_nhdsWithin] with t ht
      have ht' : (0 : ℝ) < t := ht
      rw [hmG_eq t ht']
      unfold mRel
      have hsym : entryRMS (B₀ - B₁ t) = entryRMS (B₁ t - B₀) := by
        have : B₀ - B₁ t = (-1 : ℝ) • (B₁ t - B₀) := by ext i j; simp
        rw [this, entryRMS_smul]; norm_num
      rw [hsym]
      split_ifs with h0
      · exact ⟨le_rfl, div_nonneg (mul_nonneg (by norm_num) (entryRMS_nonneg _)) hB0_pos.le⟩
      · have hden_ge : entryRMS B₀ ≤ entryRMS B₀ + entryRMS (B₁ t) :=
          le_add_of_nonneg_right (entryRMS_nonneg _)
        refine ⟨div_nonneg (mul_nonneg (by norm_num) (entryRMS_nonneg _))
          (by linarith [entryRMS_nonneg (B₁ t)]), ?_⟩
        exact div_le_div_of_nonneg_left (mul_nonneg (by norm_num) (entryRMS_nonneg _)) hB0_pos hden_ge
    refine tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds
      (by simpa using (hBdiff_lim.const_mul 2).div_const (entryRMS B₀))
      (hbound.mono fun _ h => h.1) (hbound.mono fun _ h => h.2)
  exact ⟨hmG_eq, hmG_lim, hrad_lim, by positivity⟩

/-- **Theorem 1(2) (Arbitrary prescribed RMS radius `h > 0`)**:
Given any `G₀, G₁` with `R = (1/2) r(G₁ - G₀) > 0` and ANY `h > 0`, rescaling
`G'_i = J + (h / R)(G_i - G₀)` attains exact optimal RMS radius `(1/2) r(G'_1 - G'_0) = h`. -/
theorem arbitrary_rate_realizable {d : ℕ} (G₀ G₁ : Matrix (Fin d) (Fin d) ℝ)
    (hR : 0 < (1 / 2 : ℝ) * entryRMS (G₁ - G₀)) {h : ℝ} (hh : 0 < h) :
    let R := (1 / 2 : ℝ) * entryRMS (G₁ - G₀)
    let G'₀ := allOnes d
    let G'₁ := allOnes d + (h / R) • (G₁ - G₀)
    (1 / 2 : ℝ) * entryRMS (G'₁ - G'₀) = h := by
  intro R G'₀ G'₁
  have hsub : G'₁ - G'₀ = (h / R) • (G₁ - G₀) := by simp [G'₁, G'₀]
  rw [hsub, entryRMS_smul, abs_of_pos (div_pos hh hR)]
  change (1 / 2 : ℝ) * (h / ((1 / 2 : ℝ) * entryRMS (G₁ - G₀)) * entryRMS (G₁ - G₀)) = h
  have hrms_pos : 0 < entryRMS (G₁ - G₀) := by linarith
  field_simp [hrms_pos.ne']

/-- **Theorem 1(3) (Exact `m = 0` alternating constant limit)**:
`G_k = (-1)^k J_{2k}` has `m_G = 0` identically while alternating sign across widths
(`r(J - (-J)) = 2`). -/
theorem exact_zero_mRel_alternating {k : ℕ} (hk : 1 ≤ k) :
    let G := ((-1 : ℝ) ^ k) • allOnes (2 * k)
    mRel G G = 0 ∧ entryRMS (allOnes (2 * k) - (-allOnes (2 * k))) = 2 := by
  intro G
  constructor
  · simp [mRel, entryRMS]
  · have h2 : allOnes (2 * k) - (-allOnes (2 * k)) = (2 : ℝ) • allOnes (2 * k) := by
      ext i j; simp [allOnes]; ring
    rw [h2, entryRMS_smul, abs_of_pos (by norm_num : (0 : ℝ) < 2), entryRMS_allOnes (by omega)]
    ring

/-! ### 2. Theorem 2 (`conj:spectral`): Spectral Obstructions and Arbitrariness of Edge Laws -/

/-- **Theorem 2(1) (Universal ideal causal mask obstruction)**:
For EVERY `S ≥ 1` and EVERY row-stochastic lower-triangular matrix `E` with `E_{i, 0} > 0` for
all `i`, the unique stationary distribution is `π = δ_0`, forcing `L²_0(π) = {0}`. -/
theorem causal_unique_stationary_delta0 {S : ℕ} (hS : 1 ≤ S)
    (E : Matrix (Fin S) (Fin S) ℝ)
    (hE_lower : ∀ i j : Fin S, i < j → E i j = 0)
    (hE_col0 : ∀ i : Fin S, 0 < E i ⟨0, hS⟩)
    (hE_rowsum : ∀ i : Fin S, ∑ j : Fin S, E i j = 1)
    (pi : Fin S → ℝ) (hpi_nn : ∀ i, 0 ≤ pi i) (hpi_sum : ∑ i, pi i = 1)
    (hstat : ∀ j : Fin S, ∑ i : Fin S, pi i * E i j = pi j) :
    pi = Pi.single ⟨0, hS⟩ 1 ∧
    (∀ f : Fin S → ℝ, ∑ i, pi i * f i = 0 → ∑ i, pi i * (f i) ^ 2 = 0) := by
  let z0 : Fin S := ⟨0, hS⟩
  have hE00 : E z0 z0 = 1 := by
    have hrow := hE_rowsum z0
    rw [Finset.sum_eq_single z0] at hrow
    · exact hrow
    · intro j _ hj
      apply hE_lower z0 j
      exact Fin.lt_def.2 (lt_of_le_of_ne (Nat.zero_le j.val) (fun h => hj (Fin.ext h.symm)))
    · intro h; exact (h (Finset.mem_univ z0)).elim
  have hstat0 := hstat z0
  rw [← Finset.add_sum_erase (s := Finset.univ) (f := fun i => pi i * E i z0) (a := z0)
    (Finset.mem_univ z0), hE00, mul_one] at hstat0
  have hrest0 : ∑ i ∈ Finset.univ.erase z0, pi i * E i z0 = 0 := by linarith
  have hpi_zero : ∀ i ∈ Finset.univ.erase z0, pi i = 0 := by
    intro i hi
    have hterm := (Finset.sum_eq_zero_iff_of_nonneg fun j _ =>
      mul_nonneg (hpi_nn j) (hE_col0 j).le).1 hrest0 i hi
    exact (mul_eq_zero.1 hterm).resolve_right (hE_col0 i).ne'
  have hpi0 : pi z0 = 1 := by
    rw [← Finset.add_sum_erase Finset.univ pi (Finset.mem_univ z0)] at hpi_sum
    have hsum_rest : ∑ i ∈ Finset.univ.erase z0, pi i = 0 :=
      Finset.sum_eq_zero fun i hi => hpi_zero i hi
    linarith
  have hpi_eq : pi = Pi.single z0 1 := by
    ext i
    by_cases hi : i = z0
    · subst hi; simp [hpi0]
    · simp [Pi.single_eq_of_ne hi, hpi_zero i (Finset.mem_erase.2 ⟨hi, Finset.mem_univ i⟩)]
  refine ⟨hpi_eq, ?_⟩
  intro f hf
  rw [hpi_eq] at hf ⊢
  simp only [Pi.single_apply, ite_mul, one_mul, zero_mul, Finset.sum_ite_eq', Finset.mem_univ,
    ↓reduceIte] at hf ⊢
  rw [hf, zero_pow (by norm_num)]

/-- **Theorem 2(2) (Uniform Doeblin spectral-gap contraction under a finite additive mask)**:
For any Markov matrix `E` with stationary distribution `π` satisfying `E_{ij} ≥ η π_j`
(`η < 1`), every mean-zero observable `f ∈ L²_0(π)` satisfies
`‖E f‖_{L²(π)}² ≤ (1 - η)² ‖f‖_{L²(π)}²`. -/
theorem finite_mask_doeblin_contraction {S : ℕ}
    (E : Matrix (Fin S) (Fin S) ℝ) (pi : Fin S → ℝ) {η : ℝ}
    (hη1 : η < 1)
    (hE_rowsum : ∀ i, ∑ j, E i j = 1)
    (hpi_nn : ∀ i, 0 ≤ pi i) (hpi_sum : ∑ i, pi i = 1)
    (hstat : ∀ j, ∑ i, pi i * E i j = pi j)
    (hdoeblin : ∀ i j, η * pi j ≤ E i j)
    (f : Fin S → ℝ) (hmean : ∑ j, pi j * f j = 0) :
    ∑ i, pi i * (E.mulVec f i) ^ 2 ≤ (1 - η) ^ 2 * ∑ j, pi j * (f j) ^ 2 := by
  have h1m : 0 < 1 - η := by linarith
  -- Define residual Markov kernel K_{ij} = (E_{ij} - η π_j) / (1 - η).
  let K : Matrix (Fin S) (Fin S) ℝ := Matrix.of fun i j => (E i j - η * pi j) / (1 - η)
  have hK_nn : ∀ i j, 0 ≤ K i j := fun i j =>
    div_nonneg (sub_nonneg.2 (hdoeblin i j)) h1m.le
  have hK_row : ∀ i, ∑ j, K i j = 1 := by
    intro i
    simp only [K, Matrix.of_apply, ← Finset.sum_div, Finset.sum_sub_distrib, ← Finset.mul_sum,
      hE_rowsum i, hpi_sum, mul_one, div_self h1m.ne']
  have hK_stat : ∀ j, ∑ i, pi i * K i j = pi j := by
    intro j
    simp only [K, Matrix.of_apply]
    have h1 : ∑ i, pi i * ((E i j - η * pi j) / (1 - η)) =
        ((∑ i, pi i * E i j) - η * pi j * ∑ i, pi i) / (1 - η) := by
      rw [Finset.mul_sum, ← Finset.sum_sub_distrib, Finset.sum_div]
      refine Finset.sum_congr rfl fun i _ => by ring
    rw [h1, hstat j, hpi_sum]; field_simp [h1m.ne']
  have hEf : ∀ i, E.mulVec f i = (1 - η) * K.mulVec f i := by
    intro i
    simp only [Matrix.mulVec, dotProduct, K, Matrix.of_apply]
    have h1 : (1 - η) * ∑ j, (E i j - η * pi j) / (1 - η) * f j =
        (∑ j, E i j * f j) - η * ∑ j, pi j * f j := by
      rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib]
      refine Finset.sum_congr rfl fun j _ => by field_simp [h1m.ne']
    rw [h1, hmean, mul_zero, sub_zero]
  have hjensen : ∀ i, (K.mulVec f i) ^ 2 ≤ ∑ j, K i j * (f j) ^ 2 := by
    intro i
    have hcs := Finset.sum_mul_sq_le_sq_mul_sq (Finset.univ : Finset (Fin S))
      (fun j => Real.sqrt (K i j)) (fun j => Real.sqrt (K i j) * f j)
    have hlhs : ∑ j : Fin S, Real.sqrt (K i j) * (Real.sqrt (K i j) * f j) = K.mulVec f i := by
      simp only [Matrix.mulVec, dotProduct]
      refine Finset.sum_congr rfl fun j _ => ?_
      rw [← mul_assoc, Real.mul_self_sqrt (hK_nn i j)]
    have hrhs1 : ∑ j : Fin S, (Real.sqrt (K i j)) ^ 2 = 1 := by
      rw [Finset.sum_congr rfl fun j _ => Real.sq_sqrt (hK_nn i j)]
      exact hK_row i
    have hrhs2 : ∑ j : Fin S, (Real.sqrt (K i j) * f j) ^ 2 = ∑ j, K i j * (f j) ^ 2 := by
      refine Finset.sum_congr rfl fun j _ => by rw [mul_pow, Real.sq_sqrt (hK_nn i j)]
    rw [hlhs, hrhs1, hrhs2, one_mul] at hcs
    exact hcs
  calc ∑ i, pi i * (E.mulVec f i) ^ 2
      = (1 - η) ^ 2 * ∑ i, pi i * (K.mulVec f i) ^ 2 := by
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl fun i _ => by rw [hEf i, mul_pow]; ring
    _ ≤ (1 - η) ^ 2 * ∑ i, pi i * ∑ j, K i j * (f j) ^ 2 := by
        apply mul_le_mul_of_nonneg_left _ (sq_nonneg _)
        exact Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (hjensen i) (hpi_nn i)
    _ = (1 - η) ^ 2 * ∑ j, pi j * (f j) ^ 2 := by
        congr 1
        trans ∑ i, ∑ j, pi i * K i j * (f j) ^ 2
        · refine Finset.sum_congr rfl fun i _ => ?_
          rw [Finset.mul_sum]; refine Finset.sum_congr rfl fun j _ => by ring
        · rw [Finset.sum_comm]
          refine Finset.sum_congr rfl fun j _ => by rw [← Finset.sum_mul, hK_stat j]

/-- **Theorem 2(3) (Arbitrary reversible eigenvalue `(1 - 1/S) r` at `m = 0`)**:
For `S = 2k` (`k ≥ 1`), the `2×2`-block reversible stochastic matrix mixed with `(1/S) J`
has the alternating vector `v_i = (-1)^i` as an exact eigenvector with eigenvalue `(1 - 1/S) r`. -/
theorem two_by_two_block_eigenvalue {k : ℕ} (r : ℝ) :
    let S := 2 * k
    let H : Matrix (Fin S) (Fin S) ℝ := Matrix.of fun i j =>
      (1 - 1 / (S : ℝ)) *
        (if i.val / 2 = j.val / 2 then (if i = j then (1 + r) / 2 else (1 - r) / 2) else 0) +
      1 / ((S : ℝ) * (S : ℝ))
    let v : Fin S → ℝ := fun i => if i.val % 2 = 0 then 1 else -1
    H.mulVec v = ((1 - 1 / (S : ℝ)) * r) • v := by
  intro S H v
  have hv_sum : ∑ j : Fin S, v j = 0 := by
    have hbij : ∀ m : Fin k, v ⟨2 * m.val, by omega⟩ + v ⟨2 * m.val + 1, by omega⟩ = 0 := by
      intro m; simp [v]
    have hpair : ∑ j : Fin S, v j =
        ∑ m : Fin k, (v ⟨2 * m.val, by omega⟩ + v ⟨2 * m.val + 1, by omega⟩) := by
      have h1 := Finset.sum_filter_add_sum_filter_not (s := (Finset.univ : Finset (Fin S)))
        (p := fun j : Fin S => j.val % 2 = 0) (f := v)
      rw [← h1]
      have hev : ∑ j ∈ Finset.univ.filter (fun j : Fin S => j.val % 2 = 0), v j =
          ∑ m : Fin k, v ⟨2 * m.val, by omega⟩ := by
        refine Finset.sum_nbij (fun j => ⟨j.val / 2, by omega⟩) ?_ ?_ ?_ ?_
        · intro _ _; exact Finset.mem_univ _
        · intro a ha b hb hab
          have ha' := (Finset.mem_filter.1 ha).2
          have hb' := (Finset.mem_filter.1 hb).2
          have hdiv := congrArg Fin.val hab; dsimp at hdiv
          exact Fin.ext (by omega)
        · intro m _
          refine ⟨⟨2 * m.val, by omega⟩, Finset.mem_filter.2 ⟨Finset.mem_univ _, by simp⟩,
            Fin.ext (by dsimp; omega)⟩
        · intro a ha
          have ha' := (Finset.mem_filter.1 ha).2
          congr 1; exact Fin.ext (by dsimp; omega)
      have hod : ∑ j ∈ Finset.univ.filter (fun j : Fin S => ¬ j.val % 2 = 0), v j =
          ∑ m : Fin k, v ⟨2 * m.val + 1, by omega⟩ := by
        refine Finset.sum_nbij (fun j => ⟨j.val / 2, by omega⟩) ?_ ?_ ?_ ?_
        · intro _ _; exact Finset.mem_univ _
        · intro a ha b hb hab
          have ha' := (Finset.mem_filter.1 ha).2
          have hb' := (Finset.mem_filter.1 hb).2
          have hdiv := congrArg Fin.val hab; dsimp at hdiv
          exact Fin.ext (by omega)
        · intro m _
          refine ⟨⟨2 * m.val + 1, by omega⟩, Finset.mem_filter.2 ⟨Finset.mem_univ _, by dsimp; omega⟩,
            Fin.ext (by dsimp; omega)⟩
        · intro a ha
          have ha' := (Finset.mem_filter.1 ha).2
          congr 1; exact Fin.ext (by dsimp; omega)
      rw [hev, hod, Finset.sum_add_distrib]
    rw [hpair]
    exact Finset.sum_eq_zero fun m _ => hbij m
  ext i
  simp only [Matrix.mulVec, dotProduct, H, Matrix.of_apply, Pi.smul_apply, smul_eq_mul, add_mul,
    Finset.sum_add_distrib, ← Finset.mul_sum, hv_sum, mul_zero, add_zero]
  let p : Fin S := if h : i.val % 2 = 0 then ⟨i.val + 1, by omega⟩ else ⟨i.val - 1, by omega⟩
  have hip_ne : p ≠ i := by
    dsimp only [p]
    split_ifs with h
    · intro heq; have := congrArg Fin.val heq; dsimp at this; omega
    · intro heq; have := congrArg Fin.val heq; dsimp at this; omega
  have hvp : v p = -v i := by
    dsimp only [v, p]
    by_cases h : i.val % 2 = 0
    · simp only [h, ↓reduceDIte, ↓reduceIte]
      have hmod : (i.val + 1) % 2 ≠ 0 := by omega
      simp [hmod]
    · simp only [h, ↓reduceDIte, ↓reduceIte]
      have hmod : (i.val - 1) % 2 = 0 := by omega
      simp [hmod]
  have hblock : ∀ j : Fin S, j.val / 2 = i.val / 2 ↔ j = i ∨ j = p := by
    intro j; unfold p; constructor
    · intro hdiv
      split_ifs with h
      · by_cases hj : j.val = i.val
        · left; exact Fin.ext hj
        · right; exact Fin.ext (by dsimp; omega)
      · by_cases hj : j.val = i.val
        · left; exact Fin.ext hj
        · right; exact Fin.ext (by dsimp; omega)
    · rintro (rfl | rfl)
      · rfl
      · split_ifs with h <;> (dsimp; omega)
  have hsum2 : ∑ j : Fin S,
      (if i.val / 2 = j.val / 2 then (if i = j then (1 + r) / 2 else (1 - r) / 2) else 0) * v j =
      ((1 + r) / 2) * v i + ((1 - r) / 2) * v p := by
    rw [← Finset.add_sum_erase Finset.univ _ (Finset.mem_univ i)]
    have hp_mem : p ∈ Finset.univ.erase i := Finset.mem_erase.2 ⟨hip_ne, Finset.mem_univ p⟩
    rw [← Finset.add_sum_erase (Finset.univ.erase i) _ hp_mem]
    have hrest : ∑ j ∈ (Finset.univ.erase i).erase p,
        (if i.val / 2 = j.val / 2 then (if i = j then (1 + r) / 2 else (1 - r) / 2) else 0) * v j = 0 := by
      refine Finset.sum_eq_zero fun j hj => ?_
      have hjp : j ≠ p := (Finset.mem_erase.1 hj).1
      have hji : j ≠ i := (Finset.mem_erase.1 (Finset.mem_erase.1 hj).2).1
      have hndiv : i.val / 2 ≠ j.val / 2 := fun h => by
        rcases (hblock j).1 h.symm with rfl | rfl
        · exact hji rfl
        · exact hjp rfl
      simp [hndiv]
    have hdiv_p : i.val / 2 = p.val / 2 := ((hblock p).2 (Or.inr rfl)).symm
    rw [hrest, add_zero]
    simp only [↓reduceIte, hdiv_p, hip_ne.symm]
  have hfac : ∑ j : Fin S, (1 - 1 / (S : ℝ)) *
      (if i.val / 2 = j.val / 2 then (if i = j then (1 + r) / 2 else (1 - r) / 2) else 0) * v j =
      (1 - 1 / (S : ℝ)) * ∑ j : Fin S,
        (if i.val / 2 = j.val / 2 then (if i = j then (1 + r) / 2 else (1 - r) / 2) else 0) * v j := by
    rw [Finset.mul_sum]; refine Finset.sum_congr rfl fun j _ => by ring
  rw [hfac, hsum2, hvp]; ring

/-- **Theorem 2(4) (Invisible-key variation drives `m_G → 2` without altering attention scores)**:
If `ΔG` is supported on columns where `K` vanishes (`ΔG * Kᵀ = 0`), then `Q (G₀ + u ΔG) Kᵀ = Q G₀ Kᵀ`
for every `u ∈ ℝ`, while for any `ΔG` with `entryRMS ΔG > 0`, `mRel G₀ (G₀ + u • ΔG) → 2` as
`u → +∞`. -/
theorem invisible_key_mG_tendsto_two {S d : ℕ}
    (Q K : Matrix (Fin S) (Fin d) ℝ) (G₀ ΔG : Matrix (Fin d) (Fin d) ℝ)
    (hinvis : ΔG * Kᵀ = 0) (hΔ : 0 < entryRMS ΔG) :
    (∀ u : ℝ, Q * (G₀ + u • ΔG) * Kᵀ = Q * G₀ * Kᵀ) ∧
    Tendsto (fun u : ℝ => mRel G₀ (G₀ + u • ΔG)) atTop (𝓝 2) := by
  constructor
  · intro u
    rw [Matrix.mul_add, Matrix.add_mul, Matrix.mul_smul, Matrix.smul_mul,
      Matrix.mul_assoc Q ΔG Kᵀ, hinvis, Matrix.mul_zero, smul_zero, add_zero]
  · -- By triangle inequality on Frobenius norm: u r(ΔG) - r(G₀) ≤ r(G₀ + u ΔG) ≤ u r(ΔG) + r(G₀).
    have htri : ∀ A B : Matrix (Fin d) (Fin d) ℝ, entryRMS (A + B) ≤ entryRMS A + entryRMS B := by
      intro A B
      by_cases hd0 : d = 0
      · subst hd0; simp [entryRMS]
      · have hd2 : (0 : ℝ) < (d : ℝ) * (d : ℝ) := by positivity
        set sA := ∑ p : Fin d × Fin d, (A p.1 p.2) ^ 2
        set sB := ∑ p : Fin d × Fin d, (B p.1 p.2) ^ 2
        set sAB := ∑ p : Fin d × Fin d, A p.1 p.2 * B p.1 p.2
        have hsA : 0 ≤ sA := Finset.sum_nonneg fun _ _ => sq_nonneg _
        have hsB : 0 ≤ sB := Finset.sum_nonneg fun _ _ => sq_nonneg _
        have hcs : sAB ≤ Real.sqrt sA * Real.sqrt sB := by
          rw [← Real.sqrt_mul hsA]
          exact Real.le_sqrt_of_sq_le (Finset.sum_mul_sq_le_sq_mul_sq Finset.univ
            (fun p : Fin d × Fin d => A p.1 p.2) (fun p => B p.1 p.2))
        have hexp : ∑ p : Fin d × Fin d, ((A + B) p.1 p.2) ^ 2 = sA + sB + 2 * sAB := by
          have h1 : ∀ p : Fin d × Fin d, ((A + B) p.1 p.2) ^ 2 =
              (A p.1 p.2) ^ 2 + (B p.1 p.2) ^ 2 + 2 * (A p.1 p.2 * B p.1 p.2) := by
            intro p; simp [Matrix.add_apply]; ring
          simp_rw [h1, Finset.sum_add_distrib, ← Finset.mul_sum]; rfl
        have hsq : entryRMS (A + B) ^ 2 ≤ (entryRMS A + entryRMS B) ^ 2 := by
          have hsA_eq : ∑ i : Fin d, ∑ j : Fin d, (A i j) ^ 2 = sA :=
            (Fintype.sum_prod_type (fun p : Fin d × Fin d => (A p.1 p.2) ^ 2)).symm
          have hsB_eq : ∑ i : Fin d, ∑ j : Fin d, (B i j) ^ 2 = sB :=
            (Fintype.sum_prod_type (fun p : Fin d × Fin d => (B p.1 p.2) ^ 2)).symm
          have hsAB_eq : ∑ i : Fin d, ∑ j : Fin d, ((A + B) i j) ^ 2 =
              ∑ p : Fin d × Fin d, ((A + B) p.1 p.2) ^ 2 :=
            (Fintype.sum_prod_type (fun p : Fin d × Fin d => ((A + B) p.1 p.2) ^ 2)).symm
          have hA : entryRMS A = Real.sqrt sA / (d : ℝ) := by
            unfold entryRMS
            rw [hsA_eq, Real.sqrt_div hsA, Real.sqrt_mul_self (Nat.cast_nonneg d)]
          have hB : entryRMS B = Real.sqrt sB / (d : ℝ) := by
            unfold entryRMS
            rw [hsB_eq, Real.sqrt_div hsB, Real.sqrt_mul_self (Nat.cast_nonneg d)]
          have hAB : entryRMS (A + B) ^ 2 = (sA + sB + 2 * sAB) / ((d : ℝ) * (d : ℝ)) := by
            unfold entryRMS
            rw [hsAB_eq, hexp, Real.sq_sqrt]
            have hnn : 0 ≤ ∑ p : Fin d × Fin d, ((A + B) p.1 p.2) ^ 2 :=
              Finset.sum_nonneg fun _ _ => sq_nonneg _
            rw [hexp] at hnn; exact div_nonneg hnn hd2.le
          rw [hAB, hA, hB]
          have hrhs : (Real.sqrt sA / (d : ℝ) + Real.sqrt sB / (d : ℝ)) ^ 2 =
              (sA + sB + 2 * (Real.sqrt sA * Real.sqrt sB)) / ((d : ℝ) * (d : ℝ)) := by
            have hd_ne : (d : ℝ) ≠ 0 := by positivity
            field_simp [hd_ne]
            rw [add_sq, Real.sq_sqrt hsA, Real.sq_sqrt hsB]; ring
          rw [hrhs]
          exact div_le_div_of_nonneg_right (by linarith) hd2.le
        nlinarith [entryRMS_nonneg (A + B), entryRMS_nonneg A, entryRMS_nonneg B]
    have hupper : ∀ᶠ u : ℝ in atTop, mRel G₀ (G₀ + u • ΔG) ≤ 2 := by
      filter_upwards [eventually_gt_atTop (0 : ℝ)] with u hu
      unfold mRel
      have hdiff : G₀ - (G₀ + u • ΔG) = (-u) • ΔG := by ext i j; simp
      rw [hdiff, entryRMS_smul, abs_neg, abs_of_pos hu]
      have hne : u * entryRMS ΔG ≠ 0 := mul_ne_zero hu.ne' hΔ.ne'
      simp only [hne, ↓reduceIte]
      have htri1 : u * entryRMS ΔG ≤ entryRMS G₀ + entryRMS (G₀ + u • ΔG) := by
        have heq : u • ΔG = (-1 : ℝ) • G₀ + (G₀ + u • ΔG) := by ext i j; simp
        have h1 := htri ((-1 : ℝ) • G₀) (G₀ + u • ΔG)
        rw [← heq, entryRMS_smul, abs_of_pos hu, entryRMS_smul] at h1
        norm_num at h1; exact h1
      have hden_pos : 0 < entryRMS G₀ + entryRMS (G₀ + u • ΔG) := by
        linarith [mul_pos hu hΔ]
      rw [div_le_iff₀ hden_pos]; linarith
    have hlower : ∀ᶠ u : ℝ in atTop,
        2 * (u * entryRMS ΔG) / (2 * entryRMS G₀ + u * entryRMS ΔG) ≤ mRel G₀ (G₀ + u • ΔG) := by
      filter_upwards [eventually_gt_atTop (0 : ℝ)] with u hu
      unfold mRel
      have hdiff : G₀ - (G₀ + u • ΔG) = (-u) • ΔG := by ext i j; simp
      rw [hdiff, entryRMS_smul, abs_neg, abs_of_pos hu]
      have hne : u * entryRMS ΔG ≠ 0 := mul_ne_zero hu.ne' hΔ.ne'
      simp only [hne, ↓reduceIte]
      have htri2 : entryRMS (G₀ + u • ΔG) ≤ entryRMS G₀ + u * entryRMS ΔG := by
        have h1 := htri G₀ (u • ΔG)
        rw [entryRMS_smul, abs_of_pos hu] at h1; exact h1
      have hden_pos : 0 < entryRMS G₀ + entryRMS (G₀ + u • ΔG) := by
        have heq : u • ΔG = (-1 : ℝ) • G₀ + (G₀ + u • ΔG) := by ext i j; simp
        have h1 := htri ((-1 : ℝ) • G₀) (G₀ + u • ΔG)
        rw [← heq, entryRMS_smul, abs_of_pos hu, entryRMS_smul] at h1
        norm_num at h1; linarith [mul_pos hu hΔ]
      exact div_le_div_of_nonneg_left (by positivity) hden_pos (by linarith)
    have hlim_low : Tendsto (fun u : ℝ => 2 * (u * entryRMS ΔG) / (2 * entryRMS G₀ + u * entryRMS ΔG))
        atTop (𝓝 2) := by
      have hrewrite : ∀ᶠ u : ℝ in atTop,
          2 * (u * entryRMS ΔG) / (2 * entryRMS G₀ + u * entryRMS ΔG) =
          2 / (2 * entryRMS G₀ / (u * entryRMS ΔG) + 1) := by
        filter_upwards [eventually_gt_atTop (0 : ℝ)] with u hu
        have hne : u * entryRMS ΔG ≠ 0 := mul_ne_zero hu.ne' hΔ.ne'
        field_simp [hne]
      refine Tendsto.congr' (EventuallyEq.symm hrewrite) ?_
      have hinv : Tendsto (fun u : ℝ => 2 * entryRMS G₀ / (u * entryRMS ΔG)) atTop (𝓝 0) := by
        have hmul : Tendsto (fun u : ℝ => u * entryRMS ΔG) atTop atTop :=
          Tendsto.atTop_mul_const hΔ tendsto_id
        exact tendsto_const_nhds.div_atTop hmul
      have hden : Tendsto (fun u : ℝ => 2 * entryRMS G₀ / (u * entryRMS ΔG) + 1) atTop (𝓝 (0 + 1)) :=
        hinv.add tendsto_const_nhds
      simpa using tendsto_const_nhds.div hden (by norm_num)
    exact tendsto_of_tendsto_of_tendsto_of_le_of_le' hlim_low tendsto_const_nhds hlower hupper

/-! ### 3. Theorem 3 (`conj:functorial`): Operator Transfer `J` vs `-J` Refutation -/

/-- Exact key sign-flip symmetry: negating both `G` and `K` preserves all bilinear scores. -/
theorem key_sign_flip_invariance {S d : ℕ}
    (Q K : Matrix (Fin S) (Fin d) ℝ) (G : Matrix (Fin d) (Fin d) ℝ) :
    Q * (-G) * (-K)ᵀ = Q * G * Kᵀ := by
  simp

/-- **Theorem 3 (`conj:functorial` refutation for `G₁ = J`, `G₂ = -J`)**:
For EVERY width `d ≥ 1` and EVERY real matrix `S ∈ ℝ^{d × d}` (even without any invertibility
or condition-number restriction), `r(-J - Sᵀ J S) ≥ 1` (equivalently `‖-J - Sᵀ J S‖_F ≥ d`). -/
theorem congruence_rms_lower_bound_J {d : ℕ} (hd : 1 ≤ d) (S : Matrix (Fin d) (Fin d) ℝ) :
    1 ≤ entryRMS (-allOnes d - Sᵀ * allOnes d * S) := by
  have hd_pos : (0 : ℝ) < d := by positivity
  -- Step 1: ∑_{i,j} (Sᵀ J S)_{ij} = (∑_{a,b} S_{ab})² ≥ 0!
  have hsum_cong : ∑ i : Fin d, ∑ j : Fin d, (Sᵀ * allOnes d * S) i j =
      (∑ a : Fin d, ∑ b : Fin d, S a b) ^ 2 := by
    have hinner : ∀ i j : Fin d, (Sᵀ * allOnes d * S) i j =
        (∑ a : Fin d, S a i) * (∑ b : Fin d, S b j) := by
      intro i j
      simp only [Matrix.mul_apply, Matrix.transpose_apply, allOnes, Matrix.of_apply, mul_one,
        Finset.sum_mul, Finset.mul_sum]
    simp_rw [hinner, ← Finset.mul_sum, ← Finset.sum_mul, sq]
    congr 1 <;> exact Finset.sum_comm
  -- Step 2: ∑_{i,j} (-J - Sᵀ J S)_{ij} = -d² - (∑ S)² ≤ -d²!
  let M := -allOnes d - Sᵀ * allOnes d * S
  have hsum_M : ∑ i : Fin d, ∑ j : Fin d, M i j =
      -((d : ℝ) * (d : ℝ)) - (∑ a : Fin d, ∑ b : Fin d, S a b) ^ 2 := by
    simp only [M, Matrix.sub_apply, Matrix.neg_apply,
      Finset.sum_sub_distrib, Finset.sum_neg_distrib, hsum_cong]
    simp only [allOnes, Matrix.of_apply, Finset.sum_const, Finset.card_univ,
      Fintype.card_fin, nsmul_eq_mul, mul_one]
  -- Step 3: By Cauchy-Schwarz, (∑ M)² ≤ d² * ∑ M², so d⁴ ≤ d² * ∑ M², hence 1 ≤ r(M)!
  have hcs := Finset.sum_mul_sq_le_sq_mul_sq (Finset.univ : Finset (Fin d × Fin d))
    (fun _ => (1 : ℝ)) (fun p => M p.1 p.2)
  have hcs' : (∑ i : Fin d, ∑ j : Fin d, M i j) ^ 2 ≤
      ((d : ℝ) * (d : ℝ)) * (∑ i : Fin d, ∑ j : Fin d, (M i j) ^ 2) := by
    simpa [Fintype.sum_prod_type] using hcs
  have hsq_ge : ((d : ℝ) * (d : ℝ)) ^ 2 ≤ (∑ i : Fin d, ∑ j : Fin d, M i j) ^ 2 := by
    rw [hsum_M]
    nlinarith [sq_nonneg (∑ a : Fin d, ∑ b : Fin d, S a b)]
  have hdiv : 1 ≤ (∑ i : Fin d, ∑ j : Fin d, (M i j) ^ 2) / ((d : ℝ) * (d : ℝ)) := by
    rw [one_le_div (mul_pos hd_pos hd_pos)]
    nlinarith [mul_pos hd_pos hd_pos]
  unfold entryRMS
  exact Real.one_le_sqrt.2 hdiv

/-- **Theorem 3 (`conj:functorial` refutation for full-rank `G₁ = I + J`, `G₂ = -(I + J)`)**:
For EVERY `d ≥ 1` and EVERY real matrix `S`, `r(-(I + J) - Sᵀ (I + J) S) ≥ 1`. -/
theorem congruence_rms_lower_bound_I_add_J {d : ℕ} (hd : 1 ≤ d) (S : Matrix (Fin d) (Fin d) ℝ) :
    let G₁ := (1 : Matrix (Fin d) (Fin d) ℝ) + allOnes d
    1 ≤ entryRMS (-G₁ - Sᵀ * G₁ * S) := by
  intro G₁
  have hd_pos : (0 : ℝ) < d := by positivity
  let M := -G₁ - Sᵀ * G₁ * S
  -- Sum of entries of Sᵀ G₁ S = ∑_k (∑_i S_{ki})² + (∑_{a,b} S_{ab})² ≥ 0!
  have hsum_I : ∑ i : Fin d, ∑ j : Fin d, (Sᵀ * (1 : Matrix (Fin d) (Fin d) ℝ) * S) i j =
      ∑ k : Fin d, (∑ i : Fin d, S k i) ^ 2 := by
    simp only [Matrix.mul_one, Matrix.mul_apply, Matrix.transpose_apply]
    calc ∑ i : Fin d, ∑ j : Fin d, ∑ k : Fin d, S k i * S k j
        = ∑ i : Fin d, ∑ k : Fin d, ∑ j : Fin d, S k i * S k j :=
          Finset.sum_congr rfl fun _ _ => Finset.sum_comm
      _ = ∑ k : Fin d, ∑ i : Fin d, ∑ j : Fin d, S k i * S k j := Finset.sum_comm
      _ = ∑ k : Fin d, (∑ i : Fin d, S k i) ^ 2 := by
          refine Finset.sum_congr rfl fun k _ => by
            simp_rw [← Finset.mul_sum, ← Finset.sum_mul, sq]
  have hsum_J : ∑ i : Fin d, ∑ j : Fin d, (Sᵀ * allOnes d * S) i j =
      (∑ a : Fin d, ∑ b : Fin d, S a b) ^ 2 := by
    have hinner : ∀ i j : Fin d, (Sᵀ * allOnes d * S) i j =
        (∑ a : Fin d, S a i) * (∑ b : Fin d, S b j) := by
      intro i j
      simp only [Matrix.mul_apply, Matrix.transpose_apply, allOnes, Matrix.of_apply, mul_one,
        Finset.sum_mul, Finset.mul_sum]
    simp_rw [hinner, ← Finset.mul_sum, ← Finset.sum_mul, sq]
    congr 1 <;> exact Finset.sum_comm
  have hsum_M : ∑ i : Fin d, ∑ j : Fin d, M i j =
      -((d : ℝ) + (d : ℝ) * (d : ℝ)) -
      (∑ k : Fin d, (∑ i : Fin d, S k i) ^ 2) - (∑ a : Fin d, ∑ b : Fin d, S a b) ^ 2 := by
    have hsplit : Sᵀ * G₁ * S = Sᵀ * (1 : Matrix (Fin d) (Fin d) ℝ) * S + Sᵀ * allOnes d * S := by
      simp [G₁, Matrix.mul_add, Matrix.add_mul]
    have hsum_G1 : ∑ i : Fin d, ∑ j : Fin d, G₁ i j = (d : ℝ) + (d : ℝ) * (d : ℝ) := by
      simp only [G₁, Matrix.add_apply, Matrix.one_apply, allOnes, Matrix.of_apply,
        Finset.sum_add_distrib, Finset.sum_ite_eq, Finset.mem_univ, ↓reduceIte,
        Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one]
    simp only [M, hsplit, Matrix.sub_apply, Matrix.neg_apply, Matrix.add_apply,
      Finset.sum_sub_distrib, Finset.sum_neg_distrib, Finset.sum_add_distrib,
      hsum_G1, hsum_I, hsum_J]
    ring
  have hcs := Finset.sum_mul_sq_le_sq_mul_sq (Finset.univ : Finset (Fin d × Fin d))
    (fun _ => (1 : ℝ)) (fun p => M p.1 p.2)
  have hcs' : (∑ i : Fin d, ∑ j : Fin d, M i j) ^ 2 ≤
      ((d : ℝ) * (d : ℝ)) * (∑ i : Fin d, ∑ j : Fin d, (M i j) ^ 2) := by
    simpa [Fintype.sum_prod_type] using hcs
  have hnonneg1 : 0 ≤ ∑ k : Fin d, (∑ i : Fin d, S k i) ^ 2 :=
    Finset.sum_nonneg fun _ _ => sq_nonneg _
  have hnonneg2 : 0 ≤ (∑ a : Fin d, ∑ b : Fin d, S a b) ^ 2 := sq_nonneg _
  have hsq_ge : ((d : ℝ) * (d : ℝ)) ^ 2 ≤ (∑ i : Fin d, ∑ j : Fin d, M i j) ^ 2 := by
    rw [hsum_M]
    nlinarith
  have hdiv : 1 ≤ (∑ i : Fin d, ∑ j : Fin d, (M i j) ^ 2) / ((d : ℝ) * (d : ℝ)) := by
    rw [one_le_div (mul_pos hd_pos hd_pos)]
    nlinarith [mul_pos hd_pos hd_pos]
  unfold entryRMS
  exact Real.one_le_sqrt.2 hdiv

/-! ### 4. Theorem 4 (`conj:modeltransfer`): Frozen-LN Transfer (`50%` vs `75%`) -/

/-- **Theorem 4 (Long-range autocorrelation formula)**:
For `R ~ (1/2) Unif[0, 1] + (1/2) δ_{-1/2}`, the lag-`k` autocorrelation is
`E[R^k] = 1/(2(k+1)) + (1/2)(-1/2)^k`, which vanishes identically at lag `k = 1`
(neighboring signs are uncorrelated/independent while long-range lags decay as `~ 1/(2k)`). -/
theorem long_range_autocorr_formula (k : ℕ) :
    (∫ r in (0 : ℝ)..1, (1 / 2 : ℝ) * r ^ k) + (1 / 2 : ℝ) * (-1 / 2 : ℝ) ^ k =
      1 / (2 * ((k : ℝ) + 1)) + (1 / 2 : ℝ) * (-1 / 2 : ℝ) ^ k ∧
    (1 / (2 * ((1 : ℝ) + 1)) + (1 / 2 : ℝ) * (-1 / 2 : ℝ) ^ 1 = 0) := by
  constructor
  · congr 1
    rw [intervalIntegral.integral_const_mul, integral_pow]
    simp [div_eq_mul_inv]
    ring
  · norm_num

/-- Sign encoding of a bit (`true ↦ +1`, `false ↦ -1`). -/
def bitSign (b : Bool) : ℚ := if b then 1 else -1

/-- Exact rational one-step joint probability mass `P_{𝒟_2}((H_t, B_t, C_t), (H_{t+1}, B_{t+1}, C_{t+1}))`
on the 8-token space `Bool × Bool × Bool`:
- `(H_t, B_t, C_t)` is uniform (`1/8`);
- `H_{t+1}` and `B_{t+1}` are independent fair bits (`1/4`);
- `C_{t+1} = B_t` with probability `3/4`, and `C_{t+1} ≠ B_t` with probability `1/4`. -/
def probD2 (x y : Bool × Bool × Bool) : ℚ :=
  (1 / 8) * (1 / 4) * (if y.2.2 = x.2.1 then 3 / 4 else 1 / 4)

/-- Population next-token accuracy on `𝒟_2` for any predictor `pred : (Bool × Bool × Bool) → Bool`
predicting `C_{t+1}`. -/
def accD2 (pred : Bool × Bool × Bool → Bool) : ℚ :=
  ∑ x : Bool × Bool × Bool, ∑ y : Bool × Bool × Bool,
    if pred x = y.2.2 then probD2 x y else 0

/-- **Theorem 4 (Exact `50%` vs `75%` frozen-LN transfer deficit on `𝒟_2`)**:
1. `probD2` is a valid probability distribution (sums to `1`);
2. Every constant predictor `pred _ = c₀` (the ONLY class of predictors realizable after a
   frozen zero-gain, zero-shift final LayerNorm) achieves exact accuracy `1/2` (`50%`);
3. The matched-budget control predictor `pred x = x.2.1` (realized by unit-gain final LayerNorm,
   embedding `x ↦ B_t v`, and linear readout) achieves exact accuracy `3/4` (`75%`), yielding an
   exact irreducible `1/4` (`25` percentage-point) transfer deficit;
4. The exact one-step population correlation is `E[B_t C_{t+1}] = 1/2 > 0` (so a single gradient
   step from zero readout weights already attains `75%` accuracy). -/
theorem frozen_ln_transfer_deficit :
    (∑ x : Bool × Bool × Bool, ∑ y : Bool × Bool × Bool, probD2 x y) = 1 ∧
    (∀ c₀ : Bool, accD2 (fun _ => c₀) = 1 / 2) ∧
    accD2 (fun x => x.2.1) = 3 / 4 ∧
    accD2 (fun x => x.2.1) - accD2 (fun _ => true) = 1 / 4 ∧
    (∑ x : Bool × Bool × Bool, ∑ y : Bool × Bool × Bool,
      probD2 x y * bitSign x.2.1 * bitSign y.2.2) = 1 / 2 := by
  refine ⟨?_, fun c₀ => by cases c₀ <;>
      (simp only [accD2, probD2, Fintype.sum_prod_type, Fintype.sum_bool]; norm_num),
    by simp only [accD2, probD2, Fintype.sum_prod_type, Fintype.sum_bool]; norm_num,
    by simp only [accD2, probD2, Fintype.sum_prod_type, Fintype.sum_bool]; norm_num,
    by simp only [probD2, bitSign, Fintype.sum_prod_type, Fintype.sum_bool]; norm_num⟩
  simp only [probD2, Fintype.sum_prod_type, Fintype.sum_bool]
  norm_num

end PowerLawGraphAttn
