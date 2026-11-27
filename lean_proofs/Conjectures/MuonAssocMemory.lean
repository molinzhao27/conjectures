import Mathlib

/-!
# Partial Resolution of Exact Muon Rates and Refutation of Literal Newton Matching
(arXiv:2603.26554, Section 4 Conjecture)

We formalize in Lean 4 + Mathlib (completely verified, standard foundations only) BOTH the exact
sparse-minibatch Muon step-rate identities (including `λ_t ≡ 0`) AND the literal-Newton
refutations from `solutions/muon_assoc_memory/solution.md`:

1. **Exact Sparse-Minibatch Muon Step-Rate Identities (Lemma 1 & Theorem 2)**:
   - `hLambda`: Exact stabilized spectral map `h_λ(s) = s / √(s² + λ²)` for `λ > 0` and exact
     polar sign `h_0(s) = sign(s)` at `λ = 0`.
   - `hLambda_pos_le_one` and `hLambda_first_step_lower_bound`: For `D > 0`, `h_0(D) = 1`, and
     for every `λ > 0`, `h_λ(D) ≥ (1 / √2) min(1, D / λ)`, yielding eq. (103)'s uniform
     diagonal increment bound across all schedules `0 ≤ λ_t ≤ C √d / (B (log d)^6)`.
   - `exact_gradient_factorization`: Exact softmax gradient factorization
     `q_i (u_i - (P_{ii} u_i + (1 - P_{ii}) b_i)) = q_i (1 - P_{ii}) (u_i - b_i)`.
   - `softmax_prob_lower_bound`: Logit-margin to confidence bound
     `P_{ii} ≥ 1 / (1 + (N - 1) exp (-Δ)) ≥ 1 - (N - 1) exp (-Δ)`.
   - `probe_contraction_bound`: Uniform bilinear contraction error bound (eq. (97)).
   - `amgm_unseen_recovery_upper_bound`: AM–GM bound `∏_{i=1}^k a_i ≤ (1 / k)^k` when
     `∑ a_i ≤ 1` (Theorem 2(3) matching batch-exponent lower bound).
2. **Explicit 4-Item Shear Counterexample: Newton (`4/4`) vs. Muon (`2/4`) for ALL `λ ≥ 0`
   (Theorem 3)**:
   - `shearA`, `vShear`, `uShear`, `G_shear`, `SigmaU_shear`, `Mv_shear`, `D_newton`:
     Exact `d = 2, N = 4` shear setup with `A = [[1, 2], [0, 1]]`, `v = (e₁, -e₁, e₂, -e₂)`,
     `u_i = A v_i`, and `a = p₁ + p₂ > b = p₃ + p₄ > 0`.
   - `newton_equation_exact`: Exact `W₀ = 0` Newton equation `Σ_u D_N M_v = G`.
   - `newton_recovers_all_four`: One Newton step with ANY `η_N > 0` strictly recovers **4/4** items.
   - `S_inv_sqrt`, `M_muon`, `muon_is_exact_spectral_map`: Exact closed-form `2 × 2`
     inverse square root proving `S_inv_sqrtᵀ = S_inv_sqrt`, `0 < trace S_inv_sqrt`,
     `0 < det S_inv_sqrt`, and `S_inv_sqrt * S_inv_sqrt * (Gᵀ G + λ² I₂) = I₂`, certifying that
     `M_muon = G S_inv_sqrt` is the exact stabilized spectral update `h_λ(G)` for EVERY `λ ≥ 0`
     (including `λ = 0`).
   - `muon_recovers_only_two_of_four` and `zipf4_shear_separation`: For EVERY `λ ≥ 0` and EVERY
     `η > 0`, one Muon step strictly fails on items `±e₁` (wrong output `±u₃` beats `±u₁`) and
     strictly recovers items `±e₂` — **exactly 2/4** items, including at exact Zipf `p_i ∝ i⁻²`.
   - `muon_newton_noncollinear`: `G M_λᵀ` is symmetric whereas `G D_Nᵀ` is nonsymmetric, proving
     `D_N` and `h_λ(G)` are never collinear for any `λ ≥ 0`.
3. **Exact Scalar 3-Item 2-Cycle Oscillation Witness (Theorem 4) & Polar Step-Norm Obstruction
   (Theorem 5)**:
   - `mu3`, `L3_deriv`, `hasDerivAt_L3_deriv`, `L3_deriv_strictMono`: Exact derivative
     `L'(w) = (e^w + 2 e^{2w}) / (1 + e^w + e^{2w}) - 17/49` is strictly increasing.
   - `L3_deriv_zero_pos`, `L3_deriv_neg_one_pos`, `L3_deriv_neg_two_neg`, `L3_unique_minimizer`:
     Unique global minimizer `w_* ∈ (-2, -1)` with `L'(0) > 0`, `L'(-1) > 0`, `L'(-2) < 0`.
   - `polarStep_two_cycle`: Exact unit-step polar Muon (`λ = 0, η = 1`) enters the persistent
     2-cycle `polarStep^[n] 0 = if Odd n then -1 else -2` for all `n ≥ 1`, never converging.
   - `stabilized_muon_deriv_at_opt`: Exact local linearization derivative `1 - η L''(w_*) / λ`
     for fixed-`λ > 0` stabilized Muon.
   - `polar_step_norm_obstruction`: Triangle-inequality obstruction `η - ‖W - W_*‖ ≤ ‖W_+ - W_*‖`.

Machine-checked vs prose-only:
- Machine-checked: All sparse-minibatch step-rate, gradient-factorization, softmax-margin,
  probe-contraction, and AM–GM identities; the complete 4-item shear proof (including exact
  certification of `h_λ(G)` as the PD inverse square root for all `λ ≥ 0`); non-collinearity;
  the complete 3-item scalar 2-cycle oscillation proof and stabilized linearization; and the
  polar step-norm obstruction.
- Prose-only (in `solutions/muon_assoc_memory/proof.pdf`): Gaussian concentration of dictionary
  coherence `δ = C_M √(log d / d)`, complex resolvent Neumann series integration (Lemma 1),
  high-probability Zipf coupon-collector coverage (Theorem 2), and Wendel's theorem coercivity
  for `N / d² → ∞` (Theorem 5).
-/

open BigOperators Finset Matrix

namespace MuonAssocMemory

/-! ### 1. Exact Sparse-Minibatch Muon Step-Rate Identities (Lemma 1 & Theorem 2) -/

/-- Stabilized spectral scalar map `h_λ(s)`:
at `λ = 0`, the exact polar sign `1` (`s > 0`), `-1` (`s < 0`), `0` (`s = 0`);
at `λ ≠ 0`, `s / √(s² + λ²)`. -/
noncomputable def hLambda (lam s : ℝ) : ℝ :=
  if lam = 0 then (if 0 < s then 1 else if s < 0 then -1 else 0)
  else s / Real.sqrt (s ^ 2 + lam ^ 2)

theorem hLambda_zero_of_pos {s : ℝ} (hs : 0 < s) : hLambda 0 s = 1 := by
  simp [hLambda, hs]

theorem hLambda_zero_of_neg {s : ℝ} (hs : s < 0) : hLambda 0 s = -1 := by
  have hnot : ¬ (0 < s) := by linarith
  simp [hLambda, hnot, hs]

theorem hLambda_pos_bounds {lam D : ℝ} (hD : 0 < D) :
    0 < hLambda lam D ∧ hLambda lam D ≤ 1 := by
  unfold hLambda
  split_ifs with hlam
  · exact ⟨by norm_num, le_rfl⟩
  · have hden_pos : 0 < Real.sqrt (D ^ 2 + lam ^ 2) := Real.sqrt_pos.mpr (by positivity)
    refine ⟨div_pos hD hden_pos, ?_⟩
    rw [div_le_one hden_pos]
    calc D = Real.sqrt (D ^ 2) := (Real.sqrt_sq (le_of_lt hD)).symm
      _ ≤ Real.sqrt (D ^ 2 + lam ^ 2) := Real.sqrt_le_sqrt (by linarith [sq_nonneg lam])

/-- **Uniform First-Appearance Diagonal Increment Lower Bound** (eq. (103) of `solution.md`):
For any `D > 0` and `λ > 0`, `h_λ(D) ≥ (1 / √2) * min 1 (D / λ)` (while at `λ = 0`, `h_0(D) = 1`).
At `D ≥ 1 / (2 B)`, this immediately gives `h_λ(D) ≥ (1 / (2 √2)) * min 1 (1 / (B λ))`. -/
theorem hLambda_first_step_lower_bound {lam D : ℝ} (hlam : 0 < lam) (hD : 0 < D) :
    (1 / Real.sqrt 2) * min 1 (D / lam) ≤ hLambda lam D := by
  unfold hLambda
  rw [if_neg (ne_of_gt hlam)]
  have hden_pos : 0 < Real.sqrt (D ^ 2 + lam ^ 2) := Real.sqrt_pos.mpr (by positivity)
  have hsqrt2_pos : (0 : ℝ) < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
  by_cases hle : lam ≤ D
  · have hmin : min 1 (D / lam) = 1 := min_eq_left ((one_le_div hlam).mpr hle)
    rw [hmin, mul_one, div_le_div_iff₀ hsqrt2_pos hden_pos, one_mul]
    have hsq : D ^ 2 + lam ^ 2 ≤ 2 * D ^ 2 := by nlinarith
    calc Real.sqrt (D ^ 2 + lam ^ 2)
        ≤ Real.sqrt (2 * D ^ 2) := Real.sqrt_le_sqrt hsq
      _ = Real.sqrt 2 * Real.sqrt (D ^ 2) := Real.sqrt_mul (by norm_num) _
      _ = Real.sqrt 2 * D := by rw [Real.sqrt_sq (le_of_lt hD)]
      _ = D * Real.sqrt 2 := mul_comm _ _
  · push_neg at hle
    have hmin : min 1 (D / lam) = D / lam := min_eq_right (le_of_lt ((div_lt_one hlam).mpr hle))
    rw [hmin]
    have hsq : D ^ 2 + lam ^ 2 ≤ 2 * lam ^ 2 := by nlinarith
    have hden_ub : Real.sqrt (D ^ 2 + lam ^ 2) ≤ Real.sqrt 2 * lam := by
      calc Real.sqrt (D ^ 2 + lam ^ 2)
          ≤ Real.sqrt (2 * lam ^ 2) := Real.sqrt_le_sqrt hsq
        _ = Real.sqrt 2 * Real.sqrt (lam ^ 2) := Real.sqrt_mul (by norm_num) _
        _ = Real.sqrt 2 * lam := by rw [Real.sqrt_sq (le_of_lt hlam)]
    have h1 : (1 / Real.sqrt 2) * (D / lam) = D / (Real.sqrt 2 * lam) := by ring
    rw [h1]
    exact div_le_div_of_nonneg_left (le_of_lt hD) hden_pos hden_ub

/-- **Exact Softmax Gradient Factorization** (Section 2.2):
Writing `μ_i = P_{ii} u_i + (1 - P_{ii}) b_i`, the exact negative gradient term satisfies
`q_i (u_i - μ_i) = (q_i (1 - P_{ii})) (u_i - b_i)`. -/
theorem exact_gradient_factorization {d : ℕ} (q Pii : ℝ) (u b : Fin d → ℝ) :
    q • (u - (Pii • u + (1 - Pii) • b)) = (q * (1 - Pii)) • (u - b) := by
  ext r
  simp only [Pi.smul_apply, Pi.sub_apply, Pi.add_apply, smul_eq_mul]
  ring

/-- **Softmax Confidence Lower Bound from Logit Margin** (Theorem 2(1)):
If `z_{ii} - z_{ki} ≥ Δ` for every `k ≠ i` on `Fin N`, then
`P_{ii} = exp(z_{ii}) / ∑_k exp(z_{ki}) ≥ 1 / (1 + (N - 1) exp(-Δ)) ≥ 1 - (N - 1) exp(-Δ)`. -/
theorem softmax_prob_lower_bound {N : ℕ} (i : Fin N) (z : Fin N → ℝ) {Δ : ℝ}
    (hmargin : ∀ k : Fin N, k ≠ i → Δ ≤ z i - z k) :
    let Pii := Real.exp (z i) / ∑ k : Fin N, Real.exp (z k)
    1 - ((N - 1 : ℕ) : ℝ) * Real.exp (-Δ) ≤ Pii := by
  intro Pii
  have hsum_pos : 0 < ∑ k : Fin N, Real.exp (z k) :=
    Finset.sum_pos (fun k _ => Real.exp_pos (z k)) ⟨i, Finset.mem_univ i⟩
  have herase : ∑ k ∈ Finset.univ.erase i, Real.exp (z k) ≤
      ((N - 1 : ℕ) : ℝ) * (Real.exp (z i) * Real.exp (-Δ)) := by
    have hterm : ∀ k ∈ Finset.univ.erase i,
        Real.exp (z k) ≤ Real.exp (z i) * Real.exp (-Δ) := by
      intro k hk
      have hne : k ≠ i := (Finset.mem_erase.mp hk).1
      rw [← Real.exp_add]
      exact Real.exp_le_exp.mpr (by linarith [hmargin k hne])
    have h1 := Finset.sum_le_sum hterm
    rw [Finset.sum_const, Finset.card_erase_of_mem (Finset.mem_univ i), Finset.card_univ,
      Fintype.card_fin, nsmul_eq_mul] at h1
    exact h1
  have hsum_le : ∑ k : Fin N, Real.exp (z k) ≤
      Real.exp (z i) * (1 + ((N - 1 : ℕ) : ℝ) * Real.exp (-Δ)) := by
    rw [← Finset.add_sum_erase Finset.univ (fun k => Real.exp (z k)) (Finset.mem_univ i)]
    linarith
  have h1 : 1 / (1 + ((N - 1 : ℕ) : ℝ) * Real.exp (-Δ)) ≤ Pii := by
    have hden : 0 < 1 + ((N - 1 : ℕ) : ℝ) * Real.exp (-Δ) := by positivity
    dsimp [Pii]
    rw [div_le_div_iff₀ hden hsum_pos, one_mul]
    linarith
  have h2 : 1 - ((N - 1 : ℕ) : ℝ) * Real.exp (-Δ) ≤
      1 / (1 + ((N - 1 : ℕ) : ℝ) * Real.exp (-Δ)) := by
    have hden : 0 < 1 + ((N - 1 : ℕ) : ℝ) * Real.exp (-Δ) := by positivity
    rw [le_div_iff₀ hden]
    nlinarith [sq_nonneg (((N - 1 : ℕ) : ℝ) * Real.exp (-Δ))]
  linarith

/-- **Bilinear Probe Contraction Error Bound** (eq. (97) of `solution.md`):
If probe vectors `p, q : Fin m → ℝ` satisfy `|p_r - δ_{r, k}| ≤ δ` and `|q_r - δ_{r, j}| ≤ δ`,
diagonal weights satisfy `|d_r| ≤ 1`, and perturbation matrix `E` satisfies `|E_{rs}| ≤ C δ`,
then the quadratic contraction `pᵀ (diag(d) + E) q` differs from `δ_{k, j} d_j` by at most
`(2 + m δ) δ + C δ (1 + m δ)²`. -/
theorem probe_contraction_bound {m : ℕ} (p q d : Fin m → ℝ) (E : Matrix (Fin m) (Fin m) ℝ)
    (eK eJ : Fin m → ℝ) {δ C : ℝ} (hδ : 0 ≤ δ) (hC : 0 ≤ C)
    (hp : ∀ r, |p r - eK r| ≤ δ) (hq : ∀ r, |q r - eJ r| ≤ δ)
    (heK_l1 : ∑ r : Fin m, |eK r| ≤ 1) (heJ_l1 : ∑ r : Fin m, |eJ r| ≤ 1)
    (hd : ∀ r, |d r| ≤ 1) (hE : ∀ r s, |E r s| ≤ C * δ) :
    |(∑ r : Fin m, p r * d r * q r + ∑ r : Fin m, ∑ s : Fin m, p r * E r s * q s) -
      ∑ r : Fin m, eK r * d r * eJ r| ≤
    (2 + (m : ℝ) * δ) * δ + C * δ * (1 + (m : ℝ) * δ) ^ 2 := by
  have hp_l1 : ∑ r : Fin m, |p r| ≤ 1 + (m : ℝ) * δ := by
    have h1 : ∑ r : Fin m, |p r| ≤ ∑ r : Fin m, (|eK r| + δ) := by
      refine Finset.sum_le_sum fun r _ => ?_
      calc |p r| = |eK r + (p r - eK r)| := by ring_nf
        _ ≤ |eK r| + |p r - eK r| := abs_add_le _ _
        _ ≤ |eK r| + δ := by linarith [hp r]
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
      nsmul_eq_mul] at h1
    linarith
  have hq_l1 : ∑ r : Fin m, |q r| ≤ 1 + (m : ℝ) * δ := by
    have h1 : ∑ r : Fin m, |q r| ≤ ∑ r : Fin m, (|eJ r| + δ) := by
      refine Finset.sum_le_sum fun r _ => ?_
      calc |q r| = |eJ r + (q r - eJ r)| := by ring_nf
        _ ≤ |eJ r| + |q r - eJ r| := abs_add_le _ _
        _ ≤ |eJ r| + δ := by linarith [hq r]
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
      nsmul_eq_mul] at h1
    linarith
  -- Bound diagonal difference `|∑ (p r * d r * q r - eK r * d r * eJ r)|`.
  have hdiag : |∑ r : Fin m, p r * d r * q r - ∑ r : Fin m, eK r * d r * eJ r| ≤
      (2 + (m : ℝ) * δ) * δ := by
    rw [← Finset.sum_sub_distrib]
    calc |∑ r : Fin m, (p r * d r * q r - eK r * d r * eJ r)|
        ≤ ∑ r : Fin m, |p r * d r * q r - eK r * d r * eJ r| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ r : Fin m, (|eK r| * δ + |eJ r| * δ + δ ^ 2) := by
          refine Finset.sum_le_sum fun r _ => ?_
          have hid : p r * d r * q r - eK r * d r * eJ r =
              d r * (eK r * (q r - eJ r) + (p r - eK r) * eJ r + (p r - eK r) * (q r - eJ r)) := by
            ring
          rw [hid, abs_mul]
          have htri : |eK r * (q r - eJ r) + (p r - eK r) * eJ r + (p r - eK r) * (q r - eJ r)| ≤
              |eK r| * δ + |eJ r| * δ + δ ^ 2 := by
            have h1 := abs_add_le (eK r * (q r - eJ r) + (p r - eK r) * eJ r)
              ((p r - eK r) * (q r - eJ r))
            have h2 := abs_add_le (eK r * (q r - eJ r)) ((p r - eK r) * eJ r)
            rw [abs_mul] at h1
            rw [abs_mul, abs_mul] at h2
            have ha : |eK r| * |q r - eJ r| ≤ |eK r| * δ :=
              mul_le_mul_of_nonneg_left (hq r) (abs_nonneg _)
            have hb : |p r - eK r| * |eJ r| ≤ δ * |eJ r| :=
              mul_le_mul_of_nonneg_right (hp r) (abs_nonneg _)
            have hc : |p r - eK r| * |q r - eJ r| ≤ δ * δ :=
              mul_le_mul (hp r) (hq r) (abs_nonneg _) hδ
            linarith [show δ * δ = δ ^ 2 by ring]
          have hnonneg : 0 ≤ |eK r| * δ + |eJ r| * δ + δ ^ 2 := by positivity
          calc |d r| * |eK r * (q r - eJ r) + (p r - eK r) * eJ r + (p r - eK r) * (q r - eJ r)|
              ≤ 1 * (|eK r| * δ + |eJ r| * δ + δ ^ 2) :=
                mul_le_mul (hd r) htri (abs_nonneg _) zero_le_one
            _ = |eK r| * δ + |eJ r| * δ + δ ^ 2 := one_mul _
      _ = (∑ r : Fin m, |eK r|) * δ + (∑ r : Fin m, |eJ r|) * δ + (m : ℝ) * δ ^ 2 := by
          simp only [Finset.sum_add_distrib, ← Finset.sum_mul, Finset.sum_const, Finset.card_univ,
            Fintype.card_fin, nsmul_eq_mul]
      _ ≤ 1 * δ + 1 * δ + (m : ℝ) * δ ^ 2 := by
          nlinarith
      _ = (2 + (m : ℝ) * δ) * δ := by ring
  -- Bound `E` term `|∑ r, ∑ s, p r * E r s * q s| ≤ C δ (1 + m δ)²`.
  have hEterm : |∑ r : Fin m, ∑ s : Fin m, p r * E r s * q s| ≤
      C * δ * (1 + (m : ℝ) * δ) ^ 2 := by
    calc |∑ r : Fin m, ∑ s : Fin m, p r * E r s * q s|
        ≤ ∑ r : Fin m, |∑ s : Fin m, p r * E r s * q s| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ r : Fin m, ∑ s : Fin m, |p r * E r s * q s| :=
          Finset.sum_le_sum fun r _ => Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ r : Fin m, ∑ s : Fin m, (C * δ) * (|p r| * |q s|) := by
          refine Finset.sum_le_sum fun r _ => Finset.sum_le_sum fun s _ => ?_
          rw [abs_mul, abs_mul]
          have h1 := mul_le_mul_of_nonneg_left (hE r s) (abs_nonneg (p r))
          have h2 := mul_le_mul_of_nonneg_right h1 (abs_nonneg (q s))
          linarith
      _ = (C * δ) * ((∑ r : Fin m, |p r|) * (∑ s : Fin m, |q s|)) := by
          simp only [← Finset.mul_sum, Finset.sum_mul]
      _ ≤ (C * δ) * ((1 + (m : ℝ) * δ) * (1 + (m : ℝ) * δ)) := by
          refine mul_le_mul_of_nonneg_left ?_ (mul_nonneg hC hδ)
          exact mul_le_mul hp_l1 hq_l1 (Finset.sum_nonneg fun _ _ => abs_nonneg _) (by positivity)
      _ = C * δ * (1 + (m : ℝ) * δ) ^ 2 := by ring
  have hrew : (∑ r : Fin m, p r * d r * q r + ∑ r : Fin m, ∑ s : Fin m, p r * E r s * q s) -
      ∑ r : Fin m, eK r * d r * eJ r =
      (∑ r : Fin m, p r * d r * q r - ∑ r : Fin m, eK r * d r * eJ r) +
      (∑ r : Fin m, ∑ s : Fin m, p r * E r s * q s) := by ring
  rw [hrew]
  exact le_trans (abs_add_le _ _) (add_le_add hdiag hEterm)

/-- **AM–GM Strict Recovery Probability Upper Bound** (Theorem 2(3)):
If `k ≥ 1` unseen items have nonnegative normal-cone probabilities `a_i` summing to at most `1`,
then the joint probability `∏_i a_i` is at most `(1 / k)^k`. -/
theorem amgm_unseen_recovery_upper_bound {k : ℕ} (hk : 1 ≤ k) (a : Fin k → ℝ)
    (ha : ∀ i, 0 ≤ a i) (hsum : ∑ i : Fin k, a i ≤ 1) :
    ∏ i : Fin k, a i ≤ (1 / (k : ℝ)) ^ k := by
  have hk_pos : (0 : ℝ) < (k : ℝ) := by exact_mod_cast (by omega : 0 < k)
  have hk_ne : (k : ℝ) ≠ 0 := ne_of_gt hk_pos
  have hamgm := Real.geom_mean_le_arith_mean_weighted Finset.univ
    (fun _ : Fin k => (1 / (k : ℝ))) a
    (fun _ _ => by positivity)
    (by simp [hk_ne])
    (fun i _ => ha i)
  have hrhs : ∑ i : Fin k, (1 / (k : ℝ)) * a i ≤ 1 / (k : ℝ) := by
    rw [← Finset.mul_sum]
    have := mul_le_mul_of_nonneg_left hsum (by positivity : (0 : ℝ) ≤ 1 / (k : ℝ))
    simpa using this
  have hgm : ∏ i : Fin k, a i ^ (1 / (k : ℝ)) ≤ 1 / (k : ℝ) := le_trans hamgm hrhs
  have hpow := pow_le_pow_left₀
    (Finset.prod_nonneg fun i _ => Real.rpow_nonneg (ha i) _) hgm k
  rw [← Finset.prod_pow] at hpow
  have hsimp : ∀ i : Fin k, (a i ^ (1 / (k : ℝ))) ^ k = a i := by
    intro i
    rw [← Real.rpow_natCast, ← Real.rpow_mul (ha i), one_div_mul_cancel hk_ne, Real.rpow_one]
  simp_rw [hsimp] at hpow
  exact hpow

/-! ### 2. Explicit 4-Item Shear Matrix Separating Muon (`2/4`) vs. Newton (`4/4`) (Theorem 3) -/

/-- The `2 × 2` shear matrix `A = [[1, 2], [0, 1]]`. -/
def shearA : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; 0, 1]

/-- Four input embeddings `(e₁, -e₁, e₂, -e₂)` in `ℝ²`. -/
def vShear : Fin 4 → Fin 2 → ℝ
  | 0 => ![1, 0]
  | 1 => ![-1, 0]
  | 2 => ![0, 1]
  | 3 => ![0, -1]

/-- Sheared output embeddings `u_i = A v_i` in `ℝ²`. -/
def uShear (i : Fin 4) : Fin 2 → ℝ := shearA *ᵥ vShear i

/-- Score `z_{ki}(W) = u_kᵀ W v_i` in the 4-item shear instance. -/
def shearScore (W : Matrix (Fin 2) (Fin 2) ℝ) (k i : Fin 4) : ℝ :=
  dotProduct (uShear k) (W *ᵥ vShear i)

/-- Population negative gradient `G = A diag(a, b) = [[a, 2b], [0, b]]` at `W₀ = 0`,
where `a = p₁ + p₂` and `b = p₃ + p₄`. -/
def G_shear (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![a, 2 * b; 0, b]

/-- Output covariance `Σ_u = (1/2) A Aᵀ = (1/2) [[5, 2], [2, 1]]`. -/
noncomputable def SigmaU_shear : Matrix (Fin 2) (Fin 2) ℝ := ((1 : ℝ) / 2) • (shearA * shearAᵀ)

/-- Input second moment `M_v = diag(a, b)`. -/
def Mv_shear (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![a, 0; 0, b]

/-- Exact Newton direction `D_N = 2 A⁻ᵀ = [[2, 0], [-4, 2]]`. -/
def D_newton : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; -4, 2]

/-- Verify that `G_shear` equals the exact `W₀ = 0` population negative gradient
`∑_i p_i u_i v_iᵀ` (since `ū = 0` by sign symmetry). -/
theorem G_shear_eq_exact_grad (p₁ p₂ p₃ p₄ : ℝ) :
    p₁ • Matrix.vecMulVec (uShear 0) (vShear 0) +
    p₂ • Matrix.vecMulVec (uShear 1) (vShear 1) +
    p₃ • Matrix.vecMulVec (uShear 2) (vShear 2) +
    p₄ • Matrix.vecMulVec (uShear 3) (vShear 3) =
    G_shear (p₁ + p₂) (p₃ + p₄) := by
  have hu0 : uShear 0 = ![1, 0] := by ext i; fin_cases i <;> simp [uShear, shearA, vShear, Matrix.mulVec, dotProduct, Fin.sum_univ_two]
  have hu1 : uShear 1 = ![-1, 0] := by ext i; fin_cases i <;> simp [uShear, shearA, vShear, Matrix.mulVec, dotProduct, Fin.sum_univ_two]
  have hu2 : uShear 2 = ![2, 1] := by ext i; fin_cases i <;> simp [uShear, shearA, vShear, Matrix.mulVec, dotProduct, Fin.sum_univ_two]
  have hu3 : uShear 3 = ![-2, -1] := by ext i; fin_cases i <;> simp [uShear, shearA, vShear, Matrix.mulVec, dotProduct, Fin.sum_univ_two]
  rw [hu0, hu1, hu2, hu3]
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp only [G_shear, vShear, Matrix.add_apply, Matrix.smul_apply, Matrix.vecMulVec_apply,
      smul_eq_mul, Fin.zero_eta, Fin.mk_one, Fin.isValue, Matrix.of_apply, Matrix.cons_val',
      Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_fin_one] <;> ring

/-- Exact `W₀ = 0` Newton equation `Σ_u D_N M_v = G`. -/
theorem newton_equation_exact (a b : ℝ) :
    SigmaU_shear * D_newton * Mv_shear a b = G_shear a b := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp only [SigmaU_shear, shearA, D_newton, Mv_shear, G_shear, Matrix.mul_apply,
      Matrix.smul_apply, Matrix.transpose_apply, smul_eq_mul, Fin.sum_univ_two, Fin.isValue,
      Fin.zero_eta, Fin.mk_one, Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero,
      Matrix.cons_val_one, Matrix.cons_val_fin_one] <;> ring

/-- **Theorem 3(1), Newton side**: From `W₀ = 0`, one Newton step with ANY step size `η_N > 0`
strictly recovers **4 / 4** items on the 4-item shear instance! -/
theorem newton_recovers_all_four {η : ℝ} (hη : 0 < η) :
    ∀ i k : Fin 4, k ≠ i →
      shearScore (η • D_newton) k i < shearScore (η • D_newton) i i := by
  have hsc : ∀ k i : Fin 4, shearScore (η • D_newton) k i = η * shearScore D_newton k i := by
    intro k i
    unfold shearScore
    rw [Matrix.smul_mulVec, dotProduct_smul, smul_eq_mul]
  intro i k hne
  rw [hsc, hsc]
  refine mul_lt_mul_of_pos_left ?_ hη
  fin_cases i <;> fin_cases k
  all_goals first
    | exact absurd rfl hne
    | (simp only [shearScore, uShear, vShear, shearA, D_newton, Matrix.mulVec, dotProduct,
        Fin.sum_univ_two, Fin.isValue, Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero,
        Matrix.cons_val_one, Matrix.cons_val_fin_one]; norm_num)

/-- `S = Gᵀ G + λ² I₂` for the 4-item shear matrix `G = [[a, 2b], [0, b]]`. -/
def S_shear (a b lam : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  (G_shear a b)ᵀ * G_shear a b + (lam ^ 2) • (1 : Matrix (Fin 2) (Fin 2) ℝ)

/-- `r = √(det S) = √(a² b² + λ² (a² + 5 b²) + λ⁴)`. -/
noncomputable def rDet (a b lam : ℝ) : ℝ :=
  Real.sqrt (a ^ 2 * b ^ 2 + lam ^ 2 * (a ^ 2 + 5 * b ^ 2) + lam ^ 4)

/-- Closed-form `2 × 2` inverse square root `S⁻¹/² = ((tr S + r) I₂ - S) / (r √(tr S + 2 r))`. -/
noncomputable def S_inv_sqrt (a b lam : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let trS := a ^ 2 + 5 * b ^ 2 + 2 * lam ^ 2
  let r := rDet a b lam
  ((r * Real.sqrt (trS + 2 * r))⁻¹) • ((trS + r) • (1 : Matrix (Fin 2) (Fin 2) ℝ) - S_shear a b lam)

/-- Exact stabilized Muon update `h_λ(G) = G S⁻¹/²` for every `λ ≥ 0` (including `λ = 0`). -/
noncomputable def M_muon (a b lam : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  G_shear a b * S_inv_sqrt a b lam

lemma rDet_ge_ab {a b lam : ℝ} (ha : 0 < a) (hb : 0 < b) :
    a * b ≤ rDet a b lam := by
  unfold rDet
  calc a * b = Real.sqrt ((a * b) ^ 2) := (Real.sqrt_sq (by positivity)).symm
    _ ≤ Real.sqrt (a ^ 2 * b ^ 2 + lam ^ 2 * (a ^ 2 + 5 * b ^ 2) + lam ^ 4) := by
        apply Real.sqrt_le_sqrt
        have : 0 ≤ lam ^ 2 * (a ^ 2 + 5 * b ^ 2) + lam ^ 4 := by positivity
        nlinarith

lemma rDet_sq {a b lam : ℝ} :
    (rDet a b lam) ^ 2 = a ^ 2 * b ^ 2 + lam ^ 2 * (a ^ 2 + 5 * b ^ 2) + lam ^ 4 :=
  Real.sq_sqrt (by positivity)

private lemma S_shear_entries (a b lam : ℝ) :
    S_shear a b lam 0 0 = a ^ 2 + lam ^ 2 ∧
    S_shear a b lam 0 1 = 2 * a * b ∧
    S_shear a b lam 1 0 = 2 * a * b ∧
    S_shear a b lam 1 1 = 5 * b ^ 2 + lam ^ 2 := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;>
    simp [S_shear, G_shear, Matrix.mul_apply, Fin.sum_univ_two] <;> ring

/-- **Exact Certification of `S_inv_sqrt` as the Positive-Definite Inverse Square Root of
`Gᵀ G + λ² I₂`** for EVERY `a, b > 0` and EVERY `λ ≥ 0` (including `λ = 0`!):
1. Symmetric: `S_inv_sqrtᵀ = S_inv_sqrt`.
2. Positive definite: `trace > 0` and `det > 0`.
3. Inverse square root: `S_inv_sqrt * S_inv_sqrt * (Gᵀ G + λ² I₂) = I₂`. -/
theorem muon_is_exact_spectral_map {a b lam : ℝ} (ha : 0 < a) (hb : 0 < b) :
    (S_inv_sqrt a b lam)ᵀ = S_inv_sqrt a b lam ∧
    0 < Matrix.trace (S_inv_sqrt a b lam) ∧
    0 < (S_inv_sqrt a b lam).det ∧
    S_inv_sqrt a b lam * S_inv_sqrt a b lam * S_shear a b lam = 1 := by
  set trS : ℝ := a ^ 2 + 5 * b ^ 2 + 2 * lam ^ 2
  set r : ℝ := rDet a b lam
  set c : ℝ := (r * Real.sqrt (trS + 2 * r))⁻¹
  have hr_pos : 0 < r := lt_of_lt_of_le (mul_pos ha hb) (rDet_ge_ab ha hb)
  have htr_pos : 0 < trS := by positivity
  have hden_pos : 0 < r * Real.sqrt (trS + 2 * r) :=
    mul_pos hr_pos (Real.sqrt_pos.mpr (by linarith))
  have hc_pos : 0 < c := inv_pos.mpr hden_pos
  have hr2 : r ^ 2 = a ^ 2 * b ^ 2 + lam ^ 2 * (a ^ 2 + 5 * b ^ 2) + lam ^ 4 := rDet_sq
  have hS := S_shear_entries a b lam
  have hinv00 : S_inv_sqrt a b lam 0 0 = c * (5 * b ^ 2 + lam ^ 2 + r) := by
    simp only [S_inv_sqrt, Matrix.smul_apply, Matrix.sub_apply, Matrix.one_apply, smul_eq_mul,
      ↓reduceIte, mul_one, hS.1]
    change c * (trS + r - (a ^ 2 + lam ^ 2)) = c * (5 * b ^ 2 + lam ^ 2 + r)
    dsimp [trS]; ring
  have hinv01 : S_inv_sqrt a b lam 0 1 = c * (-(2 * a * b)) := by
    simp only [S_inv_sqrt, Matrix.smul_apply, Matrix.sub_apply, Matrix.one_apply, smul_eq_mul,
      Fin.isValue, zero_ne_one, ↓reduceIte, mul_zero, zero_sub, hS.2.1]
    rfl
  have hinv10 : S_inv_sqrt a b lam 1 0 = c * (-(2 * a * b)) := by
    simp only [S_inv_sqrt, Matrix.smul_apply, Matrix.sub_apply, Matrix.one_apply, smul_eq_mul,
      Fin.isValue, one_ne_zero, ↓reduceIte, mul_zero, zero_sub, hS.2.2.1]
    rfl
  have hinv11 : S_inv_sqrt a b lam 1 1 = c * (a ^ 2 + lam ^ 2 + r) := by
    simp only [S_inv_sqrt, Matrix.smul_apply, Matrix.sub_apply, Matrix.one_apply, smul_eq_mul,
      ↓reduceIte, mul_one, hS.2.2.2]
    change c * (trS + r - (5 * b ^ 2 + lam ^ 2)) = c * (a ^ 2 + lam ^ 2 + r)
    dsimp [trS]; ring
  refine ⟨?_, ?_, ?_, ?_⟩
  · ext i j
    fin_cases i <;> fin_cases j <;> simp [hinv01, hinv10]
  · simp only [Matrix.trace, Matrix.diag, Fin.sum_univ_two, hinv00, hinv11]
    have hsum_pos : 0 < (5 * b ^ 2 + lam ^ 2 + r) + (a ^ 2 + lam ^ 2 + r) := by positivity
    nlinarith
  · rw [Matrix.det_fin_two, hinv00, hinv01, hinv10, hinv11]
    have hdet_num : (5 * b ^ 2 + lam ^ 2 + r) * (a ^ 2 + lam ^ 2 + r) -
        (-(2 * a * b)) * (-(2 * a * b)) = r * (trS + 2 * r) := by
      dsimp [trS]; nlinarith [hr2]
    have hfact : c * (5 * b ^ 2 + lam ^ 2 + r) * (c * (a ^ 2 + lam ^ 2 + r)) -
        c * -(2 * a * b) * (c * -(2 * a * b)) =
        c ^ 2 * ((5 * b ^ 2 + lam ^ 2 + r) * (a ^ 2 + lam ^ 2 + r) -
          (-(2 * a * b)) * (-(2 * a * b))) := by ring
    rw [hfact, hdet_num]
    exact mul_pos (sq_pos_of_pos hc_pos) (mul_pos hr_pos (by linarith))
  · have hc2 : c ^ 2 = (r ^ 2 * (trS + 2 * r))⁻¹ := by
      dsimp [c]
      rw [inv_pow, mul_pow, Real.sq_sqrt (by linarith)]
    have hden2_ne : r ^ 2 * (trS + 2 * r) ≠ 0 :=
      ne_of_gt (mul_pos (sq_pos_of_pos hr_pos) (by linarith))
    ext i j
    fin_cases i <;> fin_cases j
    · simp only [Fin.zero_eta, Matrix.mul_apply, Fin.sum_univ_two, Fin.isValue, Matrix.one_apply,
        ↓reduceIte, hinv00, hinv01, hinv10, hinv11, hS.1, hS.2.2.1]
      have hcalc :
        ((c * (5 * b ^ 2 + lam ^ 2 + r) * (c * (5 * b ^ 2 + lam ^ 2 + r)) +
            c * -(2 * a * b) * (c * -(2 * a * b))) * (a ^ 2 + lam ^ 2) +
          (c * (5 * b ^ 2 + lam ^ 2 + r) * (c * -(2 * a * b)) +
            c * -(2 * a * b) * (c * (a ^ 2 + lam ^ 2 + r))) * (2 * a * b)) =
        c ^ 2 * (((5 * b ^ 2 + lam ^ 2 + r) ^ 2 + 4 * a ^ 2 * b ^ 2) * (a ^ 2 + lam ^ 2) -
          4 * a ^ 2 * b ^ 2 * (trS + 2 * r)) := by dsimp [trS]; ring
      rw [hcalc, hc2, inv_mul_eq_div, div_eq_one_iff_eq hden2_ne]
      dsimp [trS]
      linear_combination (-(5 * b ^ 2 + lam ^ 2 + 2 * r)) * hr2
    · simp only [Fin.zero_eta, Fin.mk_one, Matrix.mul_apply, Fin.sum_univ_two, Fin.isValue,
        Matrix.one_apply, zero_ne_one, ↓reduceIte, hinv00, hinv01, hinv10, hinv11, hS.2.1, hS.2.2.2]
      have hcalc :
        ((c * (5 * b ^ 2 + lam ^ 2 + r) * (c * (5 * b ^ 2 + lam ^ 2 + r)) +
            c * -(2 * a * b) * (c * -(2 * a * b))) * (2 * a * b) +
          (c * (5 * b ^ 2 + lam ^ 2 + r) * (c * -(2 * a * b)) +
            c * -(2 * a * b) * (c * (a ^ 2 + lam ^ 2 + r))) * (5 * b ^ 2 + lam ^ 2)) =
        c ^ 2 * (2 * a * b) *
          (((5 * b ^ 2 + lam ^ 2 + r) ^ 2 + 4 * a ^ 2 * b ^ 2) -
            (trS + 2 * r) * (5 * b ^ 2 + lam ^ 2)) := by dsimp [trS]; ring
      rw [hcalc]
      have hzero : ((5 * b ^ 2 + lam ^ 2 + r) ^ 2 + 4 * a ^ 2 * b ^ 2) -
          (trS + 2 * r) * (5 * b ^ 2 + lam ^ 2) = 0 := by dsimp [trS]; linear_combination hr2
      rw [hzero, mul_zero]
    · simp only [Fin.zero_eta, Fin.mk_one, Matrix.mul_apply, Fin.sum_univ_two, Fin.isValue,
        Matrix.one_apply, one_ne_zero, ↓reduceIte, hinv00, hinv01, hinv10, hinv11, hS.1, hS.2.2.1]
      have hcalc :
        ((c * -(2 * a * b) * (c * (5 * b ^ 2 + lam ^ 2 + r)) +
            c * (a ^ 2 + lam ^ 2 + r) * (c * -(2 * a * b))) * (a ^ 2 + lam ^ 2) +
          (c * -(2 * a * b) * (c * -(2 * a * b)) +
            c * (a ^ 2 + lam ^ 2 + r) * (c * (a ^ 2 + lam ^ 2 + r))) * (2 * a * b)) =
        c ^ 2 * (2 * a * b) *
          (((a ^ 2 + lam ^ 2 + r) ^ 2 + 4 * a ^ 2 * b ^ 2) -
            (trS + 2 * r) * (a ^ 2 + lam ^ 2)) := by dsimp [trS]; ring
      rw [hcalc]
      have hzero : ((a ^ 2 + lam ^ 2 + r) ^ 2 + 4 * a ^ 2 * b ^ 2) -
          (trS + 2 * r) * (a ^ 2 + lam ^ 2) = 0 := by dsimp [trS]; linear_combination hr2
      rw [hzero, mul_zero]
    · simp only [Fin.mk_one, Matrix.mul_apply, Fin.sum_univ_two, Fin.isValue, Matrix.one_apply,
        ↓reduceIte, hinv00, hinv01, hinv10, hinv11, hS.2.1, hS.2.2.2]
      have hcalc :
        ((c * -(2 * a * b) * (c * (5 * b ^ 2 + lam ^ 2 + r)) +
            c * (a ^ 2 + lam ^ 2 + r) * (c * -(2 * a * b))) * (2 * a * b) +
          (c * -(2 * a * b) * (c * -(2 * a * b)) +
            c * (a ^ 2 + lam ^ 2 + r) * (c * (a ^ 2 + lam ^ 2 + r))) * (5 * b ^ 2 + lam ^ 2)) =
        c ^ 2 * (((a ^ 2 + lam ^ 2 + r) ^ 2 + 4 * a ^ 2 * b ^ 2) * (5 * b ^ 2 + lam ^ 2) -
          4 * a ^ 2 * b ^ 2 * (trS + 2 * r)) := by dsimp [trS]; ring
      rw [hcalc, hc2, inv_mul_eq_div, div_eq_one_iff_eq hden2_ne]
      dsimp [trS]
      linear_combination (-(a ^ 2 + lam ^ 2 + 2 * r)) * hr2

/-- **Theorem 3(1), Muon side**: For EVERY `a > b > 0`, EVERY `λ ≥ 0` (including `λ = 0`),
and EVERY step size `η > 0`, one Muon step `η • h_λ(G)` on the 4-item shear instance:
- **strictly FAILS** on `v₁ = e₁` (`u₃` strictly beats `u₁`) and on `v₂ = -e₁` (`u₄` strictly beats `u₂`),
- **strictly SUCCEEDS** on `v₃ = e₂` and `v₄ = -e₂`
— recovering **exactly 2 / 4** items! -/
theorem muon_recovers_only_two_of_four {a b lam η : ℝ}
    (hb : 0 < b) (hab : b < a) (hη : 0 < η) :
    shearScore (η • M_muon a b lam) 0 0 < shearScore (η • M_muon a b lam) 2 0 ∧
    shearScore (η • M_muon a b lam) 1 1 < shearScore (η • M_muon a b lam) 3 1 ∧
    (∀ k : Fin 4, k ≠ 2 →
      shearScore (η • M_muon a b lam) k 2 < shearScore (η • M_muon a b lam) 2 2) ∧
    (∀ k : Fin 4, k ≠ 3 →
      shearScore (η • M_muon a b lam) k 3 < shearScore (η • M_muon a b lam) 3 3) := by
  have ha : 0 < a := lt_trans hb hab
  set trS : ℝ := a ^ 2 + 5 * b ^ 2 + 2 * lam ^ 2
  set r : ℝ := rDet a b lam
  have hr_ge : a * b ≤ r := rDet_ge_ab ha hb
  have hr_pos : 0 < r := lt_of_lt_of_le (mul_pos ha hb) hr_ge
  have htr_pos : 0 < trS := by positivity
  have hc_pos : 0 < η * (r * Real.sqrt (trS + 2 * r))⁻¹ :=
    mul_pos hη (inv_pos.mpr (mul_pos hr_pos (Real.sqrt_pos.mpr (by linarith))))
  -- Compute the 4 entries of `M_muon` up to the common positive factor `c = (r √(trS + 2r))⁻¹`.
  have hsc : ∀ k i : Fin 4,
      shearScore (η • M_muon a b lam) k i =
        (η * (r * Real.sqrt (trS + 2 * r))⁻¹) *
          dotProduct (uShear k)
            ((G_shear a b * ((trS + r) • (1 : Matrix (Fin 2) (Fin 2) ℝ) - S_shear a b lam)) *ᵥ vShear i) := by
    intro k i
    unfold shearScore M_muon S_inv_sqrt
    rw [Matrix.mul_smul, smul_smul, Matrix.smul_mulVec, dotProduct_smul, smul_eq_mul]
  -- Key margin: `a * (lam^2 + r - b^2) > 0` because `r ≥ a b > b^2`.
  have hfail_margin : 0 < a * (lam ^ 2 + r - b ^ 2) := by
    have hgt : b ^ 2 < r := by nlinarith
    have : 0 < lam ^ 2 + r - b ^ 2 := by nlinarith [sq_nonneg lam]
    exact mul_pos ha this
  have hcol2_1 : 0 < 2 * b * (lam ^ 2 + r) := by positivity
  have hcol2_2 : 0 < b * (a ^ 2 + lam ^ 2 + r) := by positivity
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hsc 0 0, hsc 2 0]
    refine mul_lt_mul_of_pos_left ?_ hc_pos
    simp only [uShear, vShear, shearA, G_shear, S_shear, Matrix.mulVec, dotProduct, Matrix.mul_apply,
      Matrix.sub_apply, Matrix.add_apply, Matrix.smul_apply, Matrix.transpose_apply, Matrix.one_apply,
      smul_eq_mul, Fin.sum_univ_two, Fin.isValue, Matrix.of_apply, Matrix.cons_val',
      Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_fin_one, ↓reduceIte, zero_ne_one,
      one_ne_zero]
    dsimp [trS]
    linarith
  · rw [hsc 1 1, hsc 3 1]
    refine mul_lt_mul_of_pos_left ?_ hc_pos
    simp only [uShear, vShear, shearA, G_shear, S_shear, Matrix.mulVec, dotProduct, Matrix.mul_apply,
      Matrix.sub_apply, Matrix.add_apply, Matrix.smul_apply, Matrix.transpose_apply, Matrix.one_apply,
      smul_eq_mul, Fin.sum_univ_two, Fin.isValue, Matrix.of_apply, Matrix.cons_val',
      Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_fin_one, ↓reduceIte, zero_ne_one,
      one_ne_zero]
    dsimp [trS]
    linarith
  · intro k hk
    rw [hsc k 2, hsc 2 2]
    refine mul_lt_mul_of_pos_left ?_ hc_pos
    fin_cases k
    all_goals first
      | exact absurd rfl hk
      | (simp only [uShear, vShear, shearA, G_shear, S_shear, Matrix.mulVec, dotProduct, Matrix.mul_apply,
          Matrix.sub_apply, Matrix.add_apply, Matrix.smul_apply, Matrix.transpose_apply, Matrix.one_apply,
          smul_eq_mul, Fin.sum_univ_two, Fin.isValue, Matrix.of_apply, Matrix.cons_val',
          Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_fin_one, ↓reduceIte, zero_ne_one,
          one_ne_zero]; dsimp [trS]; linarith)
  · intro k hk
    rw [hsc k 3, hsc 3 3]
    refine mul_lt_mul_of_pos_left ?_ hc_pos
    fin_cases k
    all_goals first
      | exact absurd rfl hk
      | (simp only [uShear, vShear, shearA, G_shear, S_shear, Matrix.mulVec, dotProduct, Matrix.mul_apply,
          Matrix.sub_apply, Matrix.add_apply, Matrix.smul_apply, Matrix.transpose_apply, Matrix.one_apply,
          smul_eq_mul, Fin.sum_univ_two, Fin.isValue, Matrix.of_apply, Matrix.cons_val',
          Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_fin_one, ↓reduceIte, zero_ne_one,
          one_ne_zero]; dsimp [trS]; linarith)

/-- Exact instantiation at Zipf probabilities `p_i ∝ i⁻²` (`p = (144, 36, 16, 9) / 205`):
`a = 180 / 205 > 25 / 205 = b > 0`, separating Newton (`4/4`) from Muon (`2/4`) for all `λ ≥ 0`. -/
theorem zipf4_shear_separation {lam ηN ηM : ℝ} (hηN : 0 < ηN) (hηM : 0 < ηM) :
    (∀ i k : Fin 4, k ≠ i →
      shearScore (ηN • D_newton) k i < shearScore (ηN • D_newton) i i) ∧
    shearScore (ηM • M_muon (180 / 205) (25 / 205) lam) 0 0 <
      shearScore (ηM • M_muon (180 / 205) (25 / 205) lam) 2 0 ∧
    shearScore (ηM • M_muon (180 / 205) (25 / 205) lam) 1 1 <
      shearScore (ηM • M_muon (180 / 205) (25 / 205) lam) 3 1 := by
  have hM := muon_recovers_only_two_of_four (a := 180 / 205) (b := 25 / 205) (lam := lam)
    (by norm_num) (by norm_num) hηM
  exact ⟨newton_recovers_all_four hηN, hM.1, hM.2.1⟩

/-- **Non-Collinearity Witness** (Theorem 3(3)):
For every `a > b > 0` and every `λ ≥ 0`, `G M_λᵀ` is symmetric whereas `G D_Nᵀ` is nonsymmetric
(its `(0, 1)` entry is `4 (b - a) ≠ 0` while its `(1, 0)` entry is `0`), so `D_N` is NEVER a
scalar multiple of `M_λ`. -/
theorem muon_newton_noncollinear {a b lam : ℝ} (ha : 0 < a) (hb : 0 < b) (hab : b < a) :
    (G_shear a b * (M_muon a b lam)ᵀ)ᵀ = G_shear a b * (M_muon a b lam)ᵀ ∧
    ¬ ∃ α : ℝ, D_newton = α • M_muon a b lam := by
  have hsymS : (S_inv_sqrt a b lam)ᵀ = S_inv_sqrt a b lam := (muon_is_exact_spectral_map ha hb).1
  have hsymM : (G_shear a b * (M_muon a b lam)ᵀ)ᵀ = G_shear a b * (M_muon a b lam)ᵀ := by
    unfold M_muon
    rw [Matrix.transpose_mul, Matrix.transpose_mul, Matrix.transpose_mul, hsymS,
      Matrix.transpose_transpose, Matrix.mul_assoc, hsymS]
  refine ⟨hsymM, ?_⟩
  rintro ⟨α, hcol⟩
  have hsymN : (G_shear a b * D_newtonᵀ)ᵀ = G_shear a b * D_newtonᵀ := by
    rw [hcol, Matrix.transpose_smul, Matrix.mul_smul, Matrix.transpose_smul, hsymM]
  have hentry := congrArg (fun M : Matrix (Fin 2) (Fin 2) ℝ => M 0 1) hsymN
  simp only [Matrix.transpose_apply, Matrix.mul_apply, G_shear, D_newton, Fin.sum_univ_two,
    Fin.isValue, Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.cons_val_fin_one] at hentry
  linarith

/-! ### 3. Exact Scalar 3-Item 2-Cycle Oscillation Witness (Theorem 4) & Polar Obstruction -/

/-- Softmax mean output `μ(w) = (e^w + 2 e^{2w}) / (1 + e^w + e^{2w})` on the 3-item scalar
memory `u = (0, 1, 2), v = (1, 1, 1)`. -/
noncomputable def mu3 (w : ℝ) : ℝ :=
  (Real.exp w + 2 * Real.exp (2 * w)) / (1 + Real.exp w + Real.exp (2 * w))

/-- Exact derivative `L'(w) = μ(w) - 17 / 49` for Zipf weights `p = (36, 9, 4) / 49`. -/
noncomputable def L3_deriv (w : ℝ) : ℝ := mu3 w - 17 / 49

/-- Exact second derivative `L''(w) = (e^w + 4 e^{2w} + e^{3w}) / (1 + e^w + e^{2w})² > 0`. -/
noncomputable def L3_deriv2 (w : ℝ) : ℝ :=
  (Real.exp w + 4 * Real.exp (2 * w) + Real.exp (3 * w)) /
    (1 + Real.exp w + Real.exp (2 * w)) ^ 2

theorem hasDerivAt_L3_deriv (w : ℝ) : HasDerivAt L3_deriv (L3_deriv2 w) w := by
  unfold L3_deriv mu3 L3_deriv2
  have hden_pos : 0 < 1 + Real.exp w + Real.exp (2 * w) := by positivity
  have hden_ne : 1 + Real.exp w + Real.exp (2 * w) ≠ 0 := ne_of_gt hden_pos
  have he1 : HasDerivAt (fun x : ℝ => Real.exp x) (Real.exp w) w := Real.hasDerivAt_exp w
  have he2 : HasDerivAt (fun x : ℝ => Real.exp (2 * x)) (Real.exp (2 * w) * 2) w := by
    have hlin : HasDerivAt (fun x : ℝ => 2 * x) 2 w := by
      simpa using (hasDerivAt_id w).const_mul 2
    exact (Real.hasDerivAt_exp (2 * w)).comp w hlin
  have hnum : HasDerivAt (fun x : ℝ => Real.exp x + 2 * Real.exp (2 * x))
      (Real.exp w + 2 * (Real.exp (2 * w) * 2)) w := he1.add (he2.const_mul 2)
  have hden : HasDerivAt (fun x : ℝ => 1 + Real.exp x + Real.exp (2 * x))
      (Real.exp w + Real.exp (2 * w) * 2) w := ((he1.const_add 1).add he2)
  have hq := (hnum.div hden hden_ne).sub_const (17 / 49 : ℝ)
  convert hq using 1
  have h3 : Real.exp (3 * w) = Real.exp w * Real.exp (2 * w) := by
    rw [← Real.exp_add]; congr 1; ring
  have h22 : Real.exp (2 * w) * Real.exp (2 * w) = Real.exp (2 * w) ^ 2 := by ring
  have h11 : Real.exp w * Real.exp w = Real.exp (2 * w) := by
    rw [← Real.exp_add]; congr 1; ring
  rw [h3]
  field_simp [hden_ne]
  nlinarith [h11]

theorem L3_deriv2_pos (w : ℝ) : 0 < L3_deriv2 w := by
  unfold L3_deriv2
  positivity

theorem L3_deriv_strictMono : StrictMono L3_deriv :=
  strictMono_of_deriv_pos fun w => by rw [(hasDerivAt_L3_deriv w).deriv]; exact L3_deriv2_pos w

private lemma exp_one_crude_bounds : (27 / 10 : ℝ) < Real.exp 1 ∧ Real.exp 1 < 272 / 100 := by
  have hb := Real.exp_bound (x := (1 : ℝ)) (by norm_num) (n := 6) (by norm_num)
  simp only [Finset.sum_range_succ, Finset.range_zero, Finset.sum_empty, Nat.factorial,
    Nat.succ_eq_add_one] at hb
  norm_num at hb
  rcases abs_le.mp hb with ⟨hl, hr⟩
  exact ⟨by linarith, by linarith⟩

theorem L3_deriv_zero_pos : 0 < L3_deriv 0 := by
  unfold L3_deriv mu3
  norm_num

theorem L3_deriv_neg_one_pos : 0 < L3_deriv (-1) := by
  unfold L3_deriv mu3
  have ⟨he_lo, he_hi⟩ := exp_one_crude_bounds
  have he_pos : 0 < Real.exp 1 := Real.exp_pos 1
  have h1 : Real.exp (-1) = (Real.exp 1)⁻¹ := Real.exp_neg 1
  have h2 : Real.exp (2 * (-1)) = ((Real.exp 1) ^ 2)⁻¹ := by
    rw [show (2 : ℝ) * (-1) = -((2 : ℕ) * (1 : ℝ)) by norm_num, Real.exp_neg, Real.exp_nat_mul]
  rw [h1, h2]
  have hden : 0 < 1 + (Real.exp 1)⁻¹ + ((Real.exp 1) ^ 2)⁻¹ := by positivity
  rw [sub_pos, div_lt_div_iff₀ (by norm_num) hden]
  have hmul : (17 : ℝ) * (Real.exp 1) ^ 2 - 32 * Real.exp 1 - 81 < 0 := by nlinarith
  have he2_ne : (Real.exp 1) ^ 2 ≠ 0 := ne_of_gt (sq_pos_of_pos he_pos)
  have hclear : (17 * (1 + (Real.exp 1)⁻¹ + ((Real.exp 1) ^ 2)⁻¹) -
      ((Real.exp 1)⁻¹ + 2 * ((Real.exp 1) ^ 2)⁻¹) * 49) * (Real.exp 1) ^ 2 =
      17 * (Real.exp 1) ^ 2 - 32 * Real.exp 1 - 81 := by
    field_simp [ne_of_gt he_pos, he2_ne]; ring
  nlinarith [sq_pos_of_pos he_pos]

theorem L3_deriv_neg_two_neg : L3_deriv (-2) < 0 := by
  unfold L3_deriv mu3
  have ⟨he_lo, _⟩ := exp_one_crude_bounds
  have he_pos : 0 < Real.exp 1 := Real.exp_pos 1
  have h1 : Real.exp (-2) = ((Real.exp 1) ^ 2)⁻¹ := by
    rw [show (-2 : ℝ) = -((2 : ℕ) * (1 : ℝ)) by norm_num, Real.exp_neg, Real.exp_nat_mul]
  have h2 : Real.exp (2 * (-2)) = ((Real.exp 1) ^ 4)⁻¹ := by
    rw [show (2 : ℝ) * (-2) = -((4 : ℕ) * (1 : ℝ)) by norm_num, Real.exp_neg, Real.exp_nat_mul]
  rw [h1, h2]
  have hden : 0 < 1 + ((Real.exp 1) ^ 2)⁻¹ + ((Real.exp 1) ^ 4)⁻¹ := by positivity
  rw [sub_neg, div_lt_div_iff₀ hden (by norm_num)]
  have he2 : (729 / 100 : ℝ) < (Real.exp 1) ^ 2 := by nlinarith
  have hpoly : 0 < 17 * ((Real.exp 1) ^ 2) ^ 2 - 32 * (Real.exp 1) ^ 2 - 81 := by nlinarith
  have he4_ne : (Real.exp 1) ^ 4 ≠ 0 := ne_of_gt (pow_pos he_pos 4)
  have hclear : (17 * (1 + ((Real.exp 1) ^ 2)⁻¹ + ((Real.exp 1) ^ 4)⁻¹) -
      (((Real.exp 1) ^ 2)⁻¹ + 2 * ((Real.exp 1) ^ 4)⁻¹) * 49) * (Real.exp 1) ^ 4 =
      17 * ((Real.exp 1) ^ 2) ^ 2 - 32 * (Real.exp 1) ^ 2 - 81 := by
    field_simp [he4_ne]; ring
  nlinarith [pow_pos he_pos 4]

/-- **Unique Global Minimizer `w_* ∈ (-2, -1)`** on the 3-item scalar memory (Theorem 4). -/
theorem L3_unique_minimizer :
    ∃! w_star : ℝ, L3_deriv w_star = 0 ∧ -2 < w_star ∧ w_star < -1 := by
  have hcont : ContinuousOn L3_deriv (Set.Icc (-2) (-1)) :=
    fun w _ => (hasDerivAt_L3_deriv w).continuousAt.continuousWithinAt
  have h0_mem : (0 : ℝ) ∈ Set.Icc (L3_deriv (-2)) (L3_deriv (-1)) :=
    ⟨le_of_lt L3_deriv_neg_two_neg, le_of_lt L3_deriv_neg_one_pos⟩
  obtain ⟨w_star, hw_mem, hw_zero⟩ :=
    intermediate_value_Icc (by norm_num : (-2 : ℝ) ≤ -1) hcont h0_mem
  have hgt : -2 < w_star := lt_of_le_of_ne hw_mem.1 fun h => by
    subst h; linarith [L3_deriv_neg_two_neg]
  have hlt : w_star < -1 := lt_of_le_of_ne hw_mem.2 fun h => by
    subst h; linarith [L3_deriv_neg_one_pos]
  refine ⟨w_star, ⟨hw_zero, hgt, hlt⟩, ?_⟩
  intro w' ⟨hw'_zero, _, _⟩
  exact L3_deriv_strictMono.injective (hw'_zero.trans hw_zero.symm)

/-- Unit-step polar Muon update (`λ = 0, η = 1`) on the 3-item scalar memory:
`w_{t+1} = w_t - h_0(L'(w_t))`. -/
noncomputable def polarStep (w : ℝ) : ℝ :=
  w - hLambda 0 (L3_deriv w)

/-- **Theorem 4(1) Exact Scalar 2-Cycle Oscillation Witness**:
Starting from `w₀ = 0`, unit-step polar Muon (`λ = 0, η = 1`) enters the persistent 2-cycle
`-1, -2, -1, -2, ...` for all `n ≥ 1`, never converging to `w_* ∈ (-2, -1)`. -/
theorem polarStep_two_cycle :
    polarStep 0 = -1 ∧ polarStep (-1) = -2 ∧ polarStep (-2) = -1 ∧
    ∀ n : ℕ, 1 ≤ n → polarStep^[n] 0 = if Odd n then (-1 : ℝ) else (-2 : ℝ) := by
  have h0 : polarStep 0 = -1 := by
    unfold polarStep
    rw [hLambda_zero_of_pos L3_deriv_zero_pos]; norm_num
  have h1 : polarStep (-1) = -2 := by
    unfold polarStep
    rw [hLambda_zero_of_pos L3_deriv_neg_one_pos]; norm_num
  have h2 : polarStep (-2) = -1 := by
    unfold polarStep
    rw [hLambda_zero_of_neg L3_deriv_neg_two_neg]; norm_num
  refine ⟨h0, h1, h2, ?_⟩
  intro n hn
  induction n with
  | zero => omega
  | succ k ih =>
    cases k with
    | zero =>
      simp [h0]
    | succ j =>
      rw [Function.iterate_succ_apply']
      have hj := ih (by omega)
      rw [hj]
      by_cases hodd : Odd (j + 1)
      · have hnot : ¬ Odd (j + 2) := by
          rcases hodd with ⟨m, hm⟩
          intro ⟨m', hm'⟩
          omega
        simp [hodd, hnot, h1]
      · have hodd2 : Odd (j + 2) := by
          have heven : Even (j + 1) := Nat.not_odd_iff_even.mp hodd
          rcases heven with ⟨m, hm⟩
          exact ⟨m, by omega⟩
        simp [hodd, hodd2, h2]

/-- **Theorem 4(3) Stabilized Muon Local Linearization**:
For fixed `λ > 0` and step size `η`, the stabilized Muon update `w ↦ w - η h_λ(L'(w))`
has exact derivative `1 - η L''(w_*) / λ` at the minimizer `w_*` (where `L'(w_*) = 0`). -/
theorem stabilized_muon_deriv_at_opt {lam η w_star : ℝ} (hlam : 0 < lam)
    (hopt : L3_deriv w_star = 0) :
    HasDerivAt (fun w => w - η * hLambda lam (L3_deriv w))
      (1 - η * L3_deriv2 w_star / lam) w_star := by
  have hlam_ne : lam ≠ 0 := ne_of_gt hlam
  -- Derivative of `s ↦ s / √(s² + λ²)` at `s = 0` is `1 / λ`.
  have hh : HasDerivAt (hLambda lam) (1 / lam) 0 := by
    have heq : hLambda lam = fun s => s / Real.sqrt (s ^ 2 + lam ^ 2) := by
      ext s; simp [hLambda, hlam_ne]
    rw [heq]
    have hsq0 : (0 : ℝ) ^ 2 + lam ^ 2 = lam ^ 2 := by ring
    have hsqrt0 : Real.sqrt ((0 : ℝ) ^ 2 + lam ^ 2) = lam := by
      rw [hsq0, Real.sqrt_sq (le_of_lt hlam)]
    have hinner : HasDerivAt (fun s : ℝ => s ^ 2 + lam ^ 2) 0 0 := by
      simpa using ((hasDerivAt_pow 2 (0 : ℝ)).add_const (lam ^ 2))
    have hden : HasDerivAt (fun s : ℝ => Real.sqrt (s ^ 2 + lam ^ 2)) 0 0 := by
      have h := hinner.sqrt (by positivity : (0 : ℝ) ^ 2 + lam ^ 2 ≠ 0)
      simpa using h
    have hquot := (hasDerivAt_id (0 : ℝ)).div hden (by rw [hsqrt0]; exact hlam_ne)
    rw [hsqrt0] at hquot
    convert hquot using 1
    simp only [id_eq, zero_mul, sub_zero, one_mul]
    field_simp [hlam_ne]
  have hh' : HasDerivAt (hLambda lam) (1 / lam) (L3_deriv w_star) := by rwa [hopt]
  have hcomp : HasDerivAt (fun w => hLambda lam (L3_deriv w)) ((1 / lam) * L3_deriv2 w_star) w_star := by
    simpa [Function.comp_def] using hh'.comp w_star (hasDerivAt_L3_deriv w_star)
  have hsub := (hasDerivAt_id w_star).sub (hcomp.const_mul η)
  convert hsub using 1
  ring

/-- **Theorem 5 Polar Step-Norm Non-Convergence Obstruction**:
Whenever a fixed-step polar update satisfies `‖W_+ - W‖ ≥ η` at `W ≠ W_*`, the triangle
inequality forces `η - ‖W - W_*‖ ≤ ‖W_+ - W_*‖`, preventing local convergence to `W_*`. -/
theorem polar_step_norm_obstruction {E : Type*} [SeminormedAddCommGroup E]
    (W W_plus W_star : E) {η : ℝ} (hstep : η ≤ ‖W_plus - W‖) :
    η - ‖W - W_star‖ ≤ ‖W_plus - W_star‖ := by
  have htri : ‖W_plus - W‖ ≤ ‖W_plus - W_star‖ + ‖W_star - W‖ := by
    calc ‖W_plus - W‖ = ‖(W_plus - W_star) + (W_star - W)‖ := by abel_nf
      _ ≤ ‖W_plus - W_star‖ + ‖W_star - W‖ := norm_add_le _ _
  rw [norm_sub_rev W_star W] at htri
  linarith

end MuonAssocMemory
