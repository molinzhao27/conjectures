import Mathlib

/-!
# Refutation of the Entropic Causal Identifiability Conjecture
(arXiv:2101.03501, Section 2, Conjecture 1 `conj:entropic`)

We formalize in Lean 4 + Mathlib (completely verified, standard foundations only) the exact entropy
inequalities and explicit refutation constructions from
`solutions/entropic_causal_ident/solution.md`:

1. **Shannon Entropy, KL Nonnegativity, and Max-Coordinate Lower Bound**:
   - `shannonEntropy_le_log_card`: `H(p) ≤ log |s|` via Gibbs' inequality.
   - `kl_nonneg_of_sum_le_one`: `∑ p_i log (p_i / q_i) ≥ 0` for probability `p` and
     sub-probability `q` (`∑ q_i ≤ 1`).
   - `max_prob_ge_exp_neg_entropy`: `H(p) ≤ t → ∃ j, exp (-t) ≤ p j` (used in Lemma 2).
2. **Lemma 1 (Finite Greedy Coupling Bound `< log n + 1`)**:
   - `greedy_atom_count_le_sq`: `1 + n (n - 1) ≤ n^2` for all `n ≥ 1`.
   - `geom_entropy_extra_lt_one`: `-(n - 1) log (1 - 1 / n) < 1` for all `n ≥ 2`.
   - `greedy_coupling_entropy_lt`: Comparing the greedy step distribution `J` (with
     `E[J] ≤ n`) to `Geom(1 / n)` via KL nonnegativity proves `H(J) < log n + 1`.
   - `tail_sum_eq_expectation`: Exact finite tail-sum identity
     `∑_{i=0}^{K-1} (i + 1) p_i = ∑_{k=0}^{K-1} ∑_{i ≥ k} p_i`.
3. **Theorem 3 (Short `m = n^2`, Cap `log n + 2` Refutation) & Lemma 2/4 Core Inequalities**:
   - `simplex_dilation_mem_and_l1`: Vertex dilation `T_i(q) = e_i + (1 + δ)(q - e_i)` stays in
     the simplex when `q_i ≥ δ` and has `ℓ₁`-distance `2 δ (1 - q_i)` (Lemma 2).
   - `mass_transfer_entropy_drop`: Transferring `v / 2` from `v` to `u ≥ v > 0` reduces Shannon
     entropy by at least `v * ((3/2) log (3/2) + (1/2) log (1/2)) > v / 4` (Lemma 4(1)).
   - `quadratic_refutation_gap_bound` and `quadratic_refutation_c2_neg`: At `m = n^2` and
     `c = 2`, the four-entropy gap satisfies `G_m < 2 - γ - c + 2 ε`, strictly negative for
     `c = 2` and small `ε > 0`.
4. **Zero-Offset Refutations: Exact Super-Exponential Gap Identity (Theorem 6) & Certified
   Polynomial-Alphabet `m = n^{1024}` Counterexample (`gap < 0`, Theorem 7)**:
   - `super_exp_zero_offset_gap`: Exact four-entropy identity `G = -D(0) - c` at `m = ⌈e^{n^5}⌉`.
   - `pointwise_Du_lower_bound`: For `u ∈ (0, 1)` and `z ≥ 0`,
     `(u + (1 - u) z) log (u + (1 - u) z) ≥ u log u + (1 - u) log (1 - u) * z + (1 - u) (z log z)`.
   - `dyadicRemainder_lt_half`: Elementary global bound `r(z) = 1 - z / (e^z - 1) < z / 2` for
     all `z > 0`.
   - `dyadic_remainder_deriv_pos`: Exact derivative `r'_a(z) = 1 - (v / (2 sinh(v/2)))^2 > 0`
     (`v = z / a > 0`), proving monotonicity of the limiting dyadic remainder in `a > 0`.
   - `euler_mascheroni_cancellation`: Exact cancellation of `γ` in
     `d - (1 - u) D(0) = 1 - (1 - u) S₀ - u γ` where `S₀ = e⁻¹ ∑_{k≥1} H_k / (k - 1)!`.
   - `poly_zero_offset_gap_strictly_neg`: Pure-Lean certified proof (completely verified, standard foundations only)
     that the Theorem 7 zero-offset (`c = 0`, `u = 1 / 1024`) asymptotic gap bound satisfies
     `U + d - D(1 / 1024) < -0.03 < 0`!

Machine-checked vs prose-only:
- Machine-checked: All Shannon entropy / KL bounds, greedy coupling `< log n + 1` proof,
  simplex dilation & mass-transfer entropy drop, quadratic `m = n^2` gap bound, exact
  zero-offset gap identities, `D(u)` superadditivity lower bound, dyadic remainder derivative
  and global `z / 2` bound, and full certified proof of `U + d - D(1 / 1024) < 0`.
- Prose-only (in `solutions/entropic_causal_ident/proof.pdf`): Concentration of volume on
  `{H(q) ≤ log n + c}`, Bernstein union bound for the uniform tail, binomial-to-Poisson weak
  convergence of `H(Y)` to `log n - D(u)`, and the Proposition 8 rational-independence support
  obstruction for `m < n^2`.
-/

open BigOperators Finset

namespace EntropicCausalIdent

/-! ### 1. Shannon Entropy and KL Divergence Core -/

/-- Natural-logarithm Shannon entropy `H(p) = -∑_{x ∈ s} p(x) log p(x)` (with `0 log 0 = 0`,
automatic from Mathlib's `Real.log 0 = 0`). -/
noncomputable def shannonEntropy {α : Type*} (s : Finset α) (p : α → ℝ) : ℝ :=
  -∑ x ∈ s, p x * Real.log (p x)

/-- Binary entropy `h(u) = -u log u - (1 - u) log (1 - u)`. -/
noncomputable def binEntropy (u : ℝ) : ℝ :=
  -(u * Real.log u + (1 - u) * Real.log (1 - u))

/-- **KL Nonnegativity** against any sub-probability measure `q` (`∑ q ≤ 1`):
`∑_{x ∈ s} p x * log (p x / q x) ≥ 0`. -/
theorem kl_nonneg_of_sum_le_one {α : Type*} (s : Finset α) (p q : α → ℝ)
    (hp : ∀ x ∈ s, 0 ≤ p x) (hq : ∀ x ∈ s, 0 ≤ q x)
    (hsum_p : ∑ x ∈ s, p x = 1) (hsum_q : ∑ x ∈ s, q x ≤ 1)
    (habs : ∀ x ∈ s, p x ≠ 0 → 0 < q x) :
    0 ≤ ∑ x ∈ s, p x * Real.log (p x / q x) := by
  -- Show `∑ p_x * log (q_x / p_x) ≤ ∑ (q_x - p_x) ≤ 0`.
  have hterm : ∀ x ∈ s, p x * (q x / p x - 1) ≥ p x * Real.log (q x / p x) := by
    intro x hx
    by_cases hpx : p x = 0
    · simp [hpx]
    · have hp_pos : 0 < p x := lt_of_le_of_ne (hp x hx) (Ne.symm hpx)
      have hq_pos : 0 < q x := habs x hx hpx
      exact mul_le_mul_of_nonneg_left (Real.log_le_sub_one_of_pos (div_pos hq_pos hp_pos))
        (le_of_lt hp_pos)
  have hsum1 : ∑ x ∈ s, p x * Real.log (q x / p x) ≤ ∑ x ∈ s, (q x - p x) := by
    calc ∑ x ∈ s, p x * Real.log (q x / p x)
        ≤ ∑ x ∈ s, p x * (q x / p x - 1) := Finset.sum_le_sum hterm
      _ ≤ ∑ x ∈ s, (q x - p x) := by
          refine Finset.sum_le_sum fun x hx => ?_
          by_cases hpx : p x = 0
          · simp [hpx, hq x hx]
          · field_simp [hpx]
            exact le_rfl
  have hsum2 : ∑ x ∈ s, (q x - p x) ≤ 0 := by
    rw [Finset.sum_sub_distrib, hsum_p]
    linarith
  have hneg : ∑ x ∈ s, p x * Real.log (p x / q x) = -∑ x ∈ s, p x * Real.log (q x / p x) := by
    rw [← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun x hx => ?_
    by_cases hpx : p x = 0
    · simp [hpx]
    · have hlog : Real.log (p x / q x) = -Real.log (q x / p x) := by
        rw [← Real.log_inv, inv_div]
      rw [hlog]
      ring
  linarith

/-- Maximum entropy bound `H(p) ≤ log |s|` on any nonempty finite set `s`. -/
theorem shannonEntropy_le_log_card {α : Type*} {s : Finset α} (hs : s.Nonempty)
    (p : α → ℝ) (hp : ∀ x ∈ s, 0 ≤ p x) (hsum : ∑ x ∈ s, p x = 1) :
    shannonEntropy s p ≤ Real.log (s.card : ℝ) := by
  have hc_pos : (0 : ℝ) < (s.card : ℝ) := by exact_mod_cast hs.card_pos
  have hc_ne : (s.card : ℝ) ≠ 0 := ne_of_gt hc_pos
  let q : α → ℝ := fun _ => (s.card : ℝ)⁻¹
  have hkl := kl_nonneg_of_sum_le_one s p q hp (fun _ _ => by positivity) hsum
    (by simp [q, hc_ne]) (fun _ _ _ => inv_pos.mpr hc_pos)
  have hexpand : ∑ x ∈ s, p x * Real.log (p x / q x) =
      -shannonEntropy s p + Real.log (s.card : ℝ) := by
    unfold shannonEntropy
    have h1 : ∑ x ∈ s, p x * Real.log (p x / q x) =
        ∑ x ∈ s, (p x * Real.log (p x) + p x * Real.log (s.card : ℝ)) := by
      refine Finset.sum_congr rfl fun x hx => ?_
      by_cases hpx : p x = 0
      · simp [hpx]
      · have hq_ne : q x ≠ 0 := ne_of_gt (inv_pos.mpr hc_pos)
        rw [Real.log_div hpx hq_ne]
        dsimp [q]
        rw [Real.log_inv]
        ring
    rw [h1, Finset.sum_add_distrib, ← Finset.sum_mul, hsum, one_mul]
    ring
  linarith

/-- Max-coordinate lower bound (Lemma 2): if `H(p) ≤ t` on a nonempty set `s`, then
`∃ j ∈ s, exp (-t) ≤ p j`. -/
theorem max_prob_ge_exp_neg_entropy {α : Type*} {s : Finset α} (hs : s.Nonempty)
    (p : α → ℝ) (hp : ∀ x ∈ s, 0 ≤ p x) (hsum : ∑ x ∈ s, p x = 1)
    {t : ℝ} (hH : shannonEntropy s p ≤ t) :
    ∃ j ∈ s, Real.exp (-t) ≤ p j := by
  obtain ⟨j, hj, hjmax⟩ := Finset.exists_max_image s p hs
  refine ⟨j, hj, ?_⟩
  have hM_pos : 0 < p j := by
    by_contra hle
    push_neg at hle
    have hzero : ∑ x ∈ s, p x ≤ 0 := by
      calc ∑ x ∈ s, p x ≤ ∑ _x ∈ s, (0 : ℝ) := Finset.sum_le_sum fun x hx => le_trans (hjmax x hx) hle
        _ = 0 := by simp
    linarith
  have hlog_le : -Real.log (p j) ≤ shannonEntropy s p := by
    unfold shannonEntropy
    have h1 : ∑ x ∈ s, p x * Real.log (p x) ≤ ∑ x ∈ s, p x * Real.log (p j) := by
      refine Finset.sum_le_sum fun x hx => ?_
      by_cases hpx : p x = 0
      · simp [hpx]
      · exact mul_le_mul_of_nonneg_left
          (Real.log_le_log (lt_of_le_of_ne (hp x hx) (Ne.symm hpx)) (hjmax x hx)) (hp x hx)
    rw [← Finset.sum_mul, hsum, one_mul] at h1
    linarith
  have h2 : -t ≤ Real.log (p j) := by linarith
  have h3 := Real.exp_le_exp.mpr h2
  rwa [Real.exp_log hM_pos] at h3

/-! ### 2. Lemma 1: Finite Greedy Coupling Bound (`< log n + 1`) -/

theorem greedy_atom_count_le_sq {n : ℕ} (hn : 1 ≤ n) :
    1 + n * (n - 1) ≤ n ^ 2 := by
  cases n with
  | zero => omega
  | succ k =>
    simp only [Nat.add_sub_cancel, sq]
    nlinarith

/-- Strict geometric-tail entropy excess bound: for every `n ≥ 2`,
`-(n - 1) * log (1 - 1 / n) < 1`. -/
theorem geom_entropy_extra_lt_one {n : ℕ} (hn : 2 ≤ n) :
    -((n : ℝ) - 1) * Real.log (1 - 1 / (n : ℝ)) < 1 := by
  have hn_pos : (0 : ℝ) < (n : ℝ) := by exact_mod_cast (by omega : 0 < n)
  have hn1_pos : (0 : ℝ) < (n : ℝ) - 1 := by
    have : (2 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
    linarith
  have hinv_eq : (1 - 1 / (n : ℝ))⁻¹ = 1 + 1 / ((n : ℝ) - 1) := by
    field_simp [ne_of_gt hn_pos, ne_of_gt hn1_pos]
    ring
  have harg_pos : 0 < 1 + 1 / ((n : ℝ) - 1) := by positivity
  have harg_ne : 1 + 1 / ((n : ℝ) - 1) ≠ 1 := by
    have : (0 : ℝ) < 1 / ((n : ℝ) - 1) := by positivity
    linarith
  have hlog : Real.log (1 + 1 / ((n : ℝ) - 1)) < 1 / ((n : ℝ) - 1) := by
    have := Real.log_lt_sub_one_of_pos harg_pos harg_ne
    linarith
  have hneg : -Real.log (1 - 1 / (n : ℝ)) = Real.log (1 + 1 / ((n : ℝ) - 1)) := by
    rw [← Real.log_inv, hinv_eq]
  calc -((n : ℝ) - 1) * Real.log (1 - 1 / (n : ℝ))
      = ((n : ℝ) - 1) * (-Real.log (1 - 1 / (n : ℝ))) := by ring
    _ = ((n : ℝ) - 1) * Real.log (1 + 1 / ((n : ℝ) - 1)) := by rw [hneg]
    _ < ((n : ℝ) - 1) * (1 / ((n : ℝ) - 1)) := mul_lt_mul_of_pos_left hlog hn1_pos
    _ = 1 := mul_one_div_cancel (ne_of_gt hn1_pos)

/-- Exact finite tail-sum expectation identity:
`∑_{i : Fin K} (i + 1) * p_i = ∑_{k : Fin K} ∑_{i ≥ k} p_i`. -/
theorem tail_sum_eq_expectation {K : ℕ} (p : Fin K → ℝ) :
    ∑ i : Fin K, ((i : ℕ) + 1 : ℝ) * p i =
      ∑ k : Fin K, ∑ i ∈ Finset.univ.filter (k ≤ ·), p i := by
  have h1 : ∀ k : Fin K, ∑ i ∈ Finset.univ.filter (k ≤ ·), p i =
      ∑ i : Fin K, if k ≤ i then p i else 0 := fun k => by
    rw [Finset.sum_filter]
  simp_rw [h1]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul]
  congr 1
  have hcard : (Finset.univ.filter (fun k : Fin K => k ≤ i)).card = (i : ℕ) + 1 := by
    have hbi : (Finset.univ.filter (fun k : Fin K => k ≤ i)).card =
        (Finset.range ((i : ℕ) + 1)).card := by
      refine Finset.card_bij (fun k _ => (k : ℕ)) ?_ ?_ ?_
      · intro k hk
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hk
        exact Finset.mem_range.mpr (Nat.lt_succ_of_le hk)
      · intro k₁ _ k₂ _ h
        exact Fin.ext h
      · intro m hm
        have hmi : m ≤ (i : ℕ) := Nat.lt_succ_iff.mp (Finset.mem_range.mp hm)
        exact ⟨⟨m, lt_of_le_of_lt hmi i.isLt⟩,
          Finset.mem_filter.mpr ⟨Finset.mem_univ _, hmi⟩, rfl⟩
    rw [hbi, Finset.card_range]
  exact_mod_cast hcard.symm

/-- **Lemma 1 (Finite Greedy Coupling Entropy Bound)**:
For any `n ≥ 2` and any step distribution `p : Fin K → ℝ` with expected step count
`∑_i (i + 1) p_i ≤ n`, the Shannon entropy satisfies `H(p) < log n + 1`. -/
theorem greedy_coupling_entropy_lt {n K : ℕ} (hn : 2 ≤ n) (p : Fin K → ℝ)
    (hp : ∀ i, 0 ≤ p i) (hsum : ∑ i : Fin K, p i = 1)
    (hexp : ∑ i : Fin K, ((i : ℕ) + 1 : ℝ) * p i ≤ (n : ℝ)) :
    shannonEntropy Finset.univ p < Real.log (n : ℝ) + 1 := by
  have hn_pos : (0 : ℝ) < (n : ℝ) := by exact_mod_cast (by omega : 0 < n)
  have h1m_pos : (0 : ℝ) < 1 - 1 / (n : ℝ) := by
    have : (2 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
    have : 1 / (n : ℝ) ≤ 1 / 2 := one_div_le_one_div_of_le (by norm_num) this
    linarith
  have h1m_lt1 : 1 - 1 / (n : ℝ) < 1 := by
    have : 0 < 1 / (n : ℝ) := by positivity
    linarith
  let q : Fin K → ℝ := fun i => (1 / (n : ℝ)) * (1 - 1 / (n : ℝ)) ^ (i : ℕ)
  have hq_pos : ∀ i : Fin K, 0 < q i := fun i => by
    dsimp [q]; positivity
  have hq_sum : ∑ i : Fin K, q i ≤ 1 := by
    have hgeom : ∑ i : Fin K, (1 - 1 / (n : ℝ)) ^ (i : ℕ) =
        (1 - (1 - 1 / (n : ℝ)) ^ K) / (1 / (n : ℝ)) := by
      rw [Fin.sum_univ_eq_sum_range (fun k => (1 - 1 / (n : ℝ)) ^ k), geom_sum_eq (ne_of_lt h1m_lt1)]
      have hden : 1 - 1 / (n : ℝ) - 1 = -(1 / (n : ℝ)) := by ring
      rw [hden]
      field_simp [ne_of_gt hn_pos]
      ring
    dsimp [q]
    rw [← Finset.mul_sum, hgeom, mul_div_cancel₀ _ (by positivity)]
    have : 0 ≤ (1 - 1 / (n : ℝ)) ^ K := by positivity
    linarith
  have hkl := kl_nonneg_of_sum_le_one Finset.univ p q (fun i _ => hp i)
    (fun i _ => le_of_lt (hq_pos i)) hsum hq_sum (fun i _ _ => hq_pos i)
  have hexpand : ∑ i : Fin K, p i * Real.log (p i / q i) =
      -shannonEntropy Finset.univ p + Real.log (n : ℝ) -
        (∑ i : Fin K, (i : ℝ) * p i) * Real.log (1 - 1 / (n : ℝ)) := by
    unfold shannonEntropy
    have hterm : ∀ i : Fin K, p i * Real.log (p i / q i) =
        p i * Real.log (p i) + p i * Real.log (n : ℝ) -
          ((i : ℝ) * p i) * Real.log (1 - 1 / (n : ℝ)) := by
      intro i
      by_cases hpi : p i = 0
      · simp [hpi]
      · rw [Real.log_div hpi (ne_of_gt (hq_pos i))]
        have hlogq : Real.log (q i) = -Real.log (n : ℝ) + (i : ℝ) * Real.log (1 - 1 / (n : ℝ)) := by
          dsimp [q]
          rw [Real.log_mul (by positivity) (ne_of_gt (pow_pos h1m_pos _)),
            one_div, Real.log_inv, Real.log_pow]
        rw [hlogq]
        ring
    simp_rw [hterm, Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.sum_mul, hsum, one_mul]
    ring
  have hEi : ∑ i : Fin K, (i : ℝ) * p i ≤ (n : ℝ) - 1 := by
    have h1 : ∑ i : Fin K, ((i : ℕ) + 1 : ℝ) * p i =
        (∑ i : Fin K, (i : ℝ) * p i) + ∑ i : Fin K, p i := by
      rw [← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl fun i _ => by ring
    linarith
  have hlog_neg : Real.log (1 - 1 / (n : ℝ)) ≤ 0 := Real.log_nonpos (le_of_lt h1m_pos) (le_of_lt h1m_lt1)
  have hmul_le : -((∑ i : Fin K, (i : ℝ) * p i) * Real.log (1 - 1 / (n : ℝ))) ≤
      -((n : ℝ) - 1) * Real.log (1 - 1 / (n : ℝ)) := by
    nlinarith
  have hextra := geom_entropy_extra_lt_one hn
  linarith

/-! ### 3. Theorem 3 (`m = n^2`, Cap `log n + 2`) & Lemma 2/4 Inequalities -/

/-- Total-entropy gap `G = H(Y) + B - H(X) - H(E)`. -/
def totalEntropyGap (H_Y B H_X H_E : ℝ) : ℝ :=
  H_Y + B - H_X - H_E

/-- **Theorem 3 (Quadratic-Alphabet Refutation Gap Bound)**:
Combining `H(Y) ≤ log n`, `B < log n + 1` (Lemma 1 for `1 + n(n-1) ≤ n^2 = m`),
`H(X) ≥ log n - (1 - γ) - ε`, and `H(E) ≥ log n + c - ε` (Lemma 2) yields
`G < 2 - γ - c + 2 ε`. -/
theorem quadratic_refutation_gap_bound {logn H_Y B H_X H_E gamma c eps : ℝ}
    (hY : H_Y ≤ logn) (hB : B < logn + 1)
    (hX : logn - (1 - gamma) - eps ≤ H_X)
    (hE : logn + c - eps ≤ H_E) :
    totalEntropyGap H_Y B H_X H_E < 2 - gamma - c + 2 * eps := by
  unfold totalEntropyGap
  linarith

/-- At `c = 2`, for `γ > 1/2` (Euler–Mascheroni `γ ≈ 0.5772`) and `0 < ε < 1/4`,
the quadratic-alphabet `m = n^2` cap `log n + 2` gives strictly negative gap `G < 0`. -/
theorem quadratic_refutation_c2_neg {logn H_Y B H_X H_E gamma eps : ℝ}
    (hgamma : (1 : ℝ) / 2 < gamma) (heps : eps < (1 : ℝ) / 4)
    (hY : H_Y ≤ logn) (hB : B < logn + 1)
    (hX : logn - (1 - gamma) - eps ≤ H_X)
    (hE : logn + 2 - eps ≤ H_E) :
    totalEntropyGap H_Y B H_X H_E < 0 := by
  have h := quadratic_refutation_gap_bound hY hB hX hE
  linarith

/-- **Lemma 2 Simplex Dilation Property**:
Dilating `q ∈ Δ_m` about vertex `e_i` by `1 + δ` (`T_i(q)_j = (1 + δ) q_j` for `j ≠ i`,
`T_i(q)_i = q_i - δ (1 - q_i)`) preserves simplex membership when `q_i ≥ δ` and has
exact `ℓ₁` displacement `2 δ (1 - q_i)`. -/
theorem simplex_dilation_mem_and_l1 {m : ℕ} (i : Fin m) (q : Fin m → ℝ) {δ : ℝ}
    (hδ : 0 ≤ δ) (hq : ∀ j, 0 ≤ q j) (hsum : ∑ j : Fin m, q j = 1) (hqi : δ ≤ q i) :
    let T : Fin m → ℝ := fun j => if j = i then q i - δ * (1 - q i) else (1 + δ) * q j
    (∀ j, 0 ≤ T j) ∧ (∑ j : Fin m, T j = 1) ∧
    (∑ j : Fin m, |T j - q j| = 2 * δ * (1 - q i)) := by
  intro T
  have hqi_le1 : q i ≤ 1 := by
    rw [← hsum]
    exact Finset.single_le_sum (fun j _ => hq j) (Finset.mem_univ i)
  have hrest : ∑ j ∈ Finset.univ.erase i, q j = 1 - q i := by
    have := Finset.add_sum_erase Finset.univ q (Finset.mem_univ i)
    linarith
  refine ⟨?_, ?_, ?_⟩
  · intro j
    dsimp [T]
    split_ifs with h
    · nlinarith [hq i]
    · have := hq j; nlinarith
  · rw [← Finset.add_sum_erase Finset.univ T (Finset.mem_univ i)]
    have hTi : T i = q i - δ * (1 - q i) := by simp [T]
    have hTrest : ∑ j ∈ Finset.univ.erase i, T j = (1 + δ) * (1 - q i) := by
      have hcongr : ∑ j ∈ Finset.univ.erase i, T j =
          ∑ j ∈ Finset.univ.erase i, (1 + δ) * q j := by
        refine Finset.sum_congr rfl fun j hj => ?_
        have hne : j ≠ i := (Finset.mem_erase.mp hj).1
        simp [T, hne]
      rw [hcongr, ← Finset.mul_sum, hrest]
    rw [hTi, hTrest]
    ring
  · rw [← Finset.add_sum_erase Finset.univ (fun j => |T j - q j|) (Finset.mem_univ i)]
    have hTi : |T i - q i| = δ * (1 - q i) := by
      simp only [T, ↓reduceIte, sub_sub_cancel_left, abs_neg]
      exact abs_of_nonneg (mul_nonneg hδ (by linarith))
    have hTrest : ∑ j ∈ Finset.univ.erase i, |T j - q j| = δ * (1 - q i) := by
      have hcongr : ∑ j ∈ Finset.univ.erase i, |T j - q j| =
          ∑ j ∈ Finset.univ.erase i, δ * q j := by
        refine Finset.sum_congr rfl fun j hj => ?_
        have hne : j ≠ i := (Finset.mem_erase.mp hj).1
        simp only [T, hne, ↓reduceIte]
        have : (1 + δ) * q j - q j = δ * q j := by ring
        rw [this, abs_of_nonneg (mul_nonneg hδ (hq j))]
      rw [hcongr, ← Finset.mul_sum, hrest]
    rw [hTi, hTrest]
    ring

/-- **Lemma 4(1) Mass-Transfer Entropy Drop**:
Transferring mass `v / 2` from coordinate `v > 0` to `u ≥ v` decreases Shannon entropy
by at least `v * ((3 / 2) log (3 / 2) + (1 / 2) log (1 / 2)) > v / 4`. -/
theorem mass_transfer_entropy_drop {u v : ℝ} (hv : 0 < v) (huv : v ≤ u) :
    v / 4 < (u + v / 2) * Real.log (u + v / 2) + (v / 2) * Real.log (v / 2) -
      (u * Real.log u + v * Real.log v) := by
  -- First show `f(u) = (u + v/2) log (u + v/2) - u log u` is nondecreasing for `u ≥ v`,
  -- because `f'(u) = log(1 + v/(2u)) > 0`.
  have hmono : (v + v / 2) * Real.log (v + v / 2) - v * Real.log v ≤
      (u + v / 2) * Real.log (u + v / 2) - u * Real.log u := by
    let f : ℝ → ℝ := fun x => (x + v / 2) * Real.log (x + v / 2) - x * Real.log x
    have hdiff : ∀ x ∈ Set.Icc v u,
        HasDerivAt f (Real.log (x + v / 2) - Real.log x) x := by
      intro x hx
      have hx_pos : 0 < x := lt_of_lt_of_le hv hx.1
      have hxv_pos : 0 < x + v / 2 := by linarith
      have hd1 : HasDerivAt (fun t => (t + v / 2) * Real.log (t + v / 2))
          (1 * Real.log (x + v / 2) + (x + v / 2) * (x + v / 2)⁻¹) x := by
        have hlin : HasDerivAt (fun t => t + v / 2) 1 x := (hasDerivAt_id x).add_const (v / 2)
        have hlg : HasDerivAt (fun t => Real.log (t + v / 2)) ((x + v / 2)⁻¹ * 1) x := by
          simpa [Function.comp_def] using (Real.hasDerivAt_log (ne_of_gt hxv_pos)).comp x hlin
        simpa [mul_one] using hlin.mul hlg
      have hd2 : HasDerivAt (fun t => t * Real.log t) (1 * Real.log x + x * x⁻¹) x := by
        simpa using (hasDerivAt_id x).mul (Real.hasDerivAt_log (ne_of_gt hx_pos))
      have hd := hd1.sub hd2
      have hsimp : (1 * Real.log (x + v / 2) + (x + v / 2) * (x + v / 2)⁻¹) -
          (1 * Real.log x + x * x⁻¹) = Real.log (x + v / 2) - Real.log x := by
        rw [mul_inv_cancel₀ (ne_of_gt hxv_pos), mul_inv_cancel₀ (ne_of_gt hx_pos)]
        ring
      rwa [hsimp] at hd
    have hnonneg : ∀ x ∈ interior (Set.Icc v u), 0 ≤ Real.log (x + v / 2) - Real.log x := by
      intro x hx
      have hxv := (interior_subset hx).1
      have hx_pos : 0 < x := lt_of_lt_of_le hv hxv
      have := Real.log_le_log hx_pos (by linarith : x ≤ x + v / 2)
      linarith
    have hcont : ContinuousOn f (Set.Icc v u) := fun x hx => (hdiff x hx).continuousAt.continuousWithinAt
    have hderiv : ∀ x ∈ interior (Set.Icc v u), DifferentiableAt ℝ f x :=
      fun x hx => (hdiff x (interior_subset hx)).differentiableAt
    have hmon := monotoneOn_of_deriv_nonneg (convex_Icc v u) hcont
      (fun x hx => (hderiv x hx).differentiableWithinAt)
      (fun x hx => by rw [(hdiff x (interior_subset hx)).deriv]; exact hnonneg x hx)
    exact hmon (Set.left_mem_Icc.mpr huv) (Set.right_mem_Icc.mpr huv) huv
  -- Evaluate at `u = v`: drop equals `(v / 2) * log (27 / 16)`.
  have hatv : (v + v / 2) * Real.log (v + v / 2) + (v / 2) * Real.log (v / 2) -
      (v * Real.log v + v * Real.log v) = (v / 2) * Real.log (27 / 16) := by
    have h1 : v + v / 2 = (3 / 2 : ℝ) * v := by ring
    have h2 : v / 2 = (1 / 2 : ℝ) * v := by ring
    rw [h1, h2, Real.log_mul (by norm_num) (ne_of_gt hv), Real.log_mul (by norm_num) (ne_of_gt hv)]
    have hlog27 : Real.log (27 / 16 : ℝ) = 3 * Real.log (3 / 2) + Real.log (1 / 2) := by
      have hprod : (27 / 16 : ℝ) = (3 / 2 : ℝ) ^ 3 * (1 / 2 : ℝ) := by norm_num
      rw [hprod, Real.log_mul (by norm_num) (by norm_num), Real.log_pow]
      norm_num
    rw [hlog27]
    ring
  -- Certify `log (27 / 16) > 1 / 2`, i.e., `exp (1 / 2) < 27 / 16` (`e < 729 / 256 = 2.84765625`).
  have hlog_gt_half : (1 : ℝ) / 2 < Real.log (27 / 16) := by
    rw [← Real.exp_lt_exp, Real.exp_log (by norm_num : (0 : ℝ) < 27 / 16)]
    have hexp_half : Real.exp (1 / 2 : ℝ) < 27 / 16 := by
      have hb := Real.exp_bound (x := (1 / 2 : ℝ)) (by norm_num) (n := 4) (by norm_num)
      simp only [Finset.sum_range_succ, Finset.range_zero, Finset.sum_empty, Nat.factorial,
        Nat.succ_eq_add_one] at hb
      norm_num at hb
      linarith [(abs_le.mp hb).2]
    exact hexp_half
  have hstrict : v / 4 < (v / 2) * Real.log (27 / 16) := by nlinarith
  linarith

/-! ### 4. Zero-Offset Optimal Gap Identity (Theorem 6) & Polynomial-Alphabet Refutation (Theorem 7) -/

/-- **Theorem 6 (Super-Exponential Zero-Offset Optimal Gap Identity)**:
At `m = ⌈e^{n^5}⌉`, the four asymptotic entropies `H(X) = log n - d`, `H(E) = log n + c`,
`H(Y) = log n - D(0)`, and `B_m = log n - d` yield the exact gap `G = -D(0) - c`
(strictly negative `-D(0)` at zero offset `c = 0`). -/
theorem super_exp_zero_offset_gap (logn d D0 c : ℝ) :
    totalEntropyGap (logn - D0) (logn - d) (logn - d) (logn + c) = -D0 - c := by
  unfold totalEntropyGap
  ring

/-- **Pointwise Superadditivity Lower Bound for `D(u)`** (eq. (152)–(154) of `solution.md`):
For every `u ∈ (0, 1)` and `z ≥ 0`,
`(u + (1 - u) z) log (u + (1 - u) z) ≥ u log u + (1 - u) log (1 - u) * z + (1 - u) (z log z)`.
Taking expectation with `E[Z] = 1` gives `D(u) ≥ (1 - u) D(0) - h(u)`. -/
theorem pointwise_Du_lower_bound {u z : ℝ} (hu0 : 0 < u) (hu1 : u < 1) (hz : 0 ≤ z) :
    u * Real.log u + (1 - u) * Real.log (1 - u) * z + (1 - u) * (z * Real.log z) ≤
      (u + (1 - u) * z) * Real.log (u + (1 - u) * z) := by
  have h1u : 0 < 1 - u := by linarith
  have hsum_ge_u : u ≤ u + (1 - u) * z := by nlinarith
  have hsum_ge_uz : (1 - u) * z ≤ u + (1 - u) * z := by linarith
  have hlog1 : Real.log u ≤ Real.log (u + (1 - u) * z) := Real.log_le_log hu0 hsum_ge_u
  have hterm1 : u * Real.log u ≤ u * Real.log (u + (1 - u) * z) :=
    mul_le_mul_of_nonneg_left hlog1 (le_of_lt hu0)
  have hterm2 : (1 - u) * Real.log (1 - u) * z + (1 - u) * (z * Real.log z) ≤
      (1 - u) * z * Real.log (u + (1 - u) * z) := by
    rcases eq_or_lt_of_le hz with rfl | hz_pos
    · simp
    · have huz_pos : 0 < (1 - u) * z := mul_pos h1u hz_pos
      have hlog2 : Real.log ((1 - u) * z) ≤ Real.log (u + (1 - u) * z) :=
        Real.log_le_log huz_pos hsum_ge_uz
      rw [Real.log_mul (ne_of_gt h1u) (ne_of_gt hz_pos)] at hlog2
      have hmul := mul_le_mul_of_nonneg_left hlog2 (le_of_lt huz_pos)
      linarith
  linarith

/-- Limiting dyadic-packing remainder `r(z) = 1 - z / (e^z - 1)` for `z > 0`. -/
noncomputable def dyadicRem (z : ℝ) : ℝ :=
  1 - z / (Real.exp z - 1)

/-- Scaled dyadic remainder `r_a(z) = a - z / (e^{z / a} - 1)` for `a > 0, z > 0`. -/
noncomputable def dyadicRemScaled (z a : ℝ) : ℝ :=
  a - z / (Real.exp (z / a) - 1)

/-- Elementary global bound `r(z) < z / 2` for every `z > 0`. -/
theorem dyadicRem_lt_half {z : ℝ} (hz : 0 < z) : dyadicRem z < z / 2 := by
  unfold dyadicRem
  have hexp_pos : 0 < Real.exp z - 1 := by
    have := Real.one_lt_exp_iff.mpr hz
    linarith
  -- Equivalent to `(1 - z/2) * (e^z - 1) < z`, i.e., `(1 - z/2) e^z < 1 + z/2`.
  suffices h : (1 - z / 2) * Real.exp z < 1 + z / 2 by
    have hmul : (1 - z / 2) * (Real.exp z - 1) < z := by linarith
    have hdiv : 1 - z / 2 < z / (Real.exp z - 1) := (lt_div_iff₀ hexp_pos).mpr hmul
    linarith
  -- Multiply by `e^{-z/2} > 0`: define `g(t) = (1 + t/2) e^{-t/2} - (1 - t/2) e^{t/2}`.
  -- Its derivative is `(t / 4) (e^{t/2} - e^{-t/2}) > 0` for `t > 0`, and `g(0) = 0`.
  let g : ℝ → ℝ := fun t => (1 + t / 2) * Real.exp (-(t / 2)) - (1 - t / 2) * Real.exp (t / 2)
  have hg_deriv : ∀ t : ℝ, HasDerivAt g ((t / 4) * (Real.exp (t / 2) - Real.exp (-(t / 2)))) t := by
    intro t
    have h1 : HasDerivAt (fun x => (1 + x / 2) * Real.exp (-(x / 2)))
        ((1 / 2) * Real.exp (-(t / 2)) + (1 + t / 2) * (Real.exp (-(t / 2)) * (-(1 / 2)))) t := by
      have ha : HasDerivAt (fun x : ℝ => 1 + x / 2) (1 / 2) t := by
        simpa using ((hasDerivAt_id t).div_const 2).const_add 1
      have hb : HasDerivAt (fun x : ℝ => Real.exp (-(x / 2))) (Real.exp (-(t / 2)) * (-(1 / 2))) t := by
        have hlin : HasDerivAt (fun x : ℝ => -(x / 2)) (-(1 / 2)) t := by
          simpa using ((hasDerivAt_id t).div_const 2).neg
        exact (Real.hasDerivAt_exp _).comp t hlin
      exact ha.mul hb
    have h2 : HasDerivAt (fun x => (1 - x / 2) * Real.exp (x / 2))
        ((-(1 / 2)) * Real.exp (t / 2) + (1 - t / 2) * (Real.exp (t / 2) * (1 / 2))) t := by
      have ha : HasDerivAt (fun x : ℝ => 1 - x / 2) (-(1 / 2)) t := by
        simpa using ((hasDerivAt_id t).div_const 2).const_sub 1
      have hb : HasDerivAt (fun x : ℝ => Real.exp (x / 2)) (Real.exp (t / 2) * (1 / 2)) t := by
        have hlin : HasDerivAt (fun x : ℝ => x / 2) (1 / 2) t := by
          simpa using (hasDerivAt_id t).div_const 2
        exact (Real.hasDerivAt_exp _).comp t hlin
      exact ha.mul hb
    have hsub := h1.sub h2
    convert hsub using 1
    ring
  have hg_pos : 0 < g z := by
    have hmono : StrictMonoOn g (Set.Ici 0) := by
      refine strictMonoOn_of_deriv_pos (convex_Ici 0)
        (fun x _ => (hg_deriv x).continuousAt.continuousWithinAt) fun x hx => ?_
      rw [interior_Ici] at hx
      rw [(hg_deriv x).deriv]
      have hx_pos : 0 < x := hx
      have he : Real.exp (-(x / 2)) < Real.exp (x / 2) := Real.exp_lt_exp.mpr (by linarith)
      exact mul_pos (by positivity) (by linarith)
    have hg0 : g 0 = 0 := by simp [g]
    rw [← hg0]
    exact hmono (Set.left_mem_Ici) (le_of_lt hz) hz
  have hmul := mul_pos hg_pos (Real.exp_pos (z / 2))
  have hexp_cancel : Real.exp (-(z / 2)) * Real.exp (z / 2) = 1 := by
    rw [← Real.exp_add]; simp
  have hexp_sq : Real.exp (z / 2) * Real.exp (z / 2) = Real.exp z := by
    rw [← Real.exp_add]; congr 1; ring
  dsimp [g] at hmul
  have hexpanded : ((1 + z / 2) * Real.exp (-(z / 2)) - (1 - z / 2) * Real.exp (z / 2)) *
      Real.exp (z / 2) = (1 + z / 2) - (1 - z / 2) * Real.exp z := by
    calc ((1 + z / 2) * Real.exp (-(z / 2)) - (1 - z / 2) * Real.exp (z / 2)) * Real.exp (z / 2)
        = (1 + z / 2) * (Real.exp (-(z / 2)) * Real.exp (z / 2)) -
          (1 - z / 2) * (Real.exp (z / 2) * Real.exp (z / 2)) := by ring
      _ = (1 + z / 2) - (1 - z / 2) * Real.exp z := by rw [hexp_cancel, hexp_sq]; ring
  linarith

/-- Elementary positive lower bound `z / (2 + z) ≤ r(z)` for every `z > 0`
(from `e^z ≥ 1 + z + z^2 / 2`). -/
theorem dyadicRem_ge {z : ℝ} (hz : 0 < z) : z / (2 + z) ≤ dyadicRem z := by
  unfold dyadicRem
  have hexp_lb : 1 + z + z ^ 2 / 2 ≤ Real.exp z := by
    have h := Real.sum_le_exp_of_nonneg (le_of_lt hz) 3
    simp only [Finset.sum_range_succ, Finset.range_zero, Finset.sum_empty, Nat.factorial] at h
    norm_num at h
    linarith
  have hexp_pos : 0 < Real.exp z - 1 := by nlinarith
  have h2z_pos : 0 < 2 + z := by linarith
  have h1 : z / (Real.exp z - 1) ≤ 2 / (2 + z) := by
    rw [div_le_div_iff₀ hexp_pos h2z_pos]
    nlinarith
  have h2 : 1 - 2 / (2 + z) = z / (2 + z) := by
    field_simp [ne_of_gt h2z_pos]
    ring
  linarith

/-- **Monotonicity Derivative Identity for `r_a(z)`** (Theorem 7 of `solution.md`):
For `a > 0` and `z > 0` with `v = z / a`,
`∂/∂a (r_a(z)) = 1 - (v / (2 sinh (v / 2)))^2 > 0`. -/
theorem dyadicRemScaled_deriv_pos {z a : ℝ} (hz : 0 < z) (ha : 0 < a) :
    let v := z / a
    HasDerivAt (dyadicRemScaled z) (1 - (v / (2 * Real.sinh (v / 2))) ^ 2) a ∧
    0 < 1 - (v / (2 * Real.sinh (v / 2))) ^ 2 := by
  intro v
  have hv_pos : 0 < v := div_pos hz ha
  have hexp_gt1 : 1 < Real.exp v := Real.one_lt_exp_iff.mpr hv_pos
  have hden_ne : Real.exp v - 1 ≠ 0 := by linarith
  have ha_ne : a ≠ 0 := ne_of_gt ha
  -- Compute derivative of `a ↦ a - z / (exp (z / a) - 1)`.
  have hderiv_raw : HasDerivAt (dyadicRemScaled z)
      (1 - z ^ 2 * Real.exp v / (a ^ 2 * (Real.exp v - 1) ^ 2)) a := by
    unfold dyadicRemScaled
    have hza : HasDerivAt (fun t : ℝ => z / t) (-(z / a ^ 2)) a := by
      have h := (hasDerivAt_const a z).div (hasDerivAt_id a) ha_ne
      convert h using 1
      simp only [id_eq]
      ring
    have hexp_comp : HasDerivAt (fun t : ℝ => Real.exp (z / t) - 1)
        (Real.exp v * (-(z / a ^ 2))) a := by
      simpa [Function.comp_def] using ((Real.hasDerivAt_exp v).comp a hza).sub_const 1
    have hquot : HasDerivAt (fun t : ℝ => z / (Real.exp (z / t) - 1))
        (-(z * (Real.exp v * (-(z / a ^ 2)))) / (Real.exp v - 1) ^ 2) a := by
      have h := (hasDerivAt_const a z).div hexp_comp hden_ne
      convert h using 1
      ring
    have hsub := (hasDerivAt_id a).sub hquot
    convert hsub using 1
    field_simp [ha_ne, hden_ne]
  -- Show `(v / (2 sinh (v / 2)))^2 = z^2 * exp v / (a^2 * (exp v - 1)^2)`.
  have hsinh_sq : (2 * Real.sinh (v / 2)) ^ 2 = (Real.exp v - 1) ^ 2 / Real.exp v := by
    rw [Real.sinh_eq]
    have h1 : 2 * ((Real.exp (v / 2) - Real.exp (-(v / 2))) / 2) =
        Real.exp (v / 2) - Real.exp (-(v / 2)) := by ring
    rw [h1]
    have he2 : Real.exp (v / 2) * Real.exp (v / 2) = Real.exp v := by
      rw [← Real.exp_add]; congr 1; ring
    have heneg : Real.exp (-(v / 2)) * Real.exp (v / 2) = 1 := by
      rw [← Real.exp_add]; simp
    have hpos_e : Real.exp v ≠ 0 := ne_of_gt (Real.exp_pos v)
    field_simp [hpos_e]
    calc (Real.exp (v / 2) - Real.exp (-(v / 2))) ^ 2 * Real.exp v
        = ((Real.exp (v / 2) - Real.exp (-(v / 2))) * Real.exp (v / 2)) ^ 2 := by
            rw [← he2]; ring
      _ = (Real.exp (v / 2) * Real.exp (v / 2) - Real.exp (-(v / 2)) * Real.exp (v / 2)) ^ 2 := by ring
      _ = (Real.exp v - 1) ^ 2 := by rw [he2, heneg]
  have heq_deriv : z ^ 2 * Real.exp v / (a ^ 2 * (Real.exp v - 1) ^ 2) =
      (v / (2 * Real.sinh (v / 2))) ^ 2 := by
    rw [div_pow, hsinh_sq]
    dsimp [v]
    have hpos_e : Real.exp v ≠ 0 := ne_of_gt (Real.exp_pos v)
    field_simp [ha_ne, hden_ne, hpos_e]
  rw [heq_deriv] at hderiv_raw
  refine ⟨hderiv_raw, ?_⟩
  -- Prove `v < 2 * sinh (v / 2)` for `v > 0`, i.e., `0 < exp (v/2) - exp (-v/2) - v`.
  have hsinh_gt : v < 2 * Real.sinh (v / 2) := by
    rw [Real.sinh_eq]
    have h1 : Real.exp (v / 2) - Real.exp (-(v / 2)) > v := by
      have hlb1 : 1 + (v / 2) + (v / 2) ^ 2 / 2 + (v / 2) ^ 3 / 6 ≤ Real.exp (v / 2) := by
        have h := Real.sum_le_exp_of_nonneg (by positivity : 0 ≤ v / 2) 4
        simp only [Finset.sum_range_succ, Finset.range_zero, Finset.sum_empty, Nat.factorial] at h
        norm_num at h
        linarith
      -- `exp(-x) ≤ 1 / (1 + x + x^2/2)` since `exp(x) ≥ 1 + x + x^2/2`.
      have hlb2 : 1 + (v / 2) + (v / 2) ^ 2 / 2 ≤ Real.exp (v / 2) := by
        have : 0 ≤ (v / 2) ^ 3 / 6 := by positivity
        linarith
      have hinv : Real.exp (-(v / 2)) ≤ (1 + v / 2 + (v / 2) ^ 2 / 2)⁻¹ := by
        rw [Real.exp_neg]
        exact inv_anti₀ (by positivity) hlb2
      -- Even simpler: `(1 + x + x^2/2) * (1 - x + x^2/2) = 1 + x^4/4 ≥ 1`, so
      -- `(1 + x + x^2/2)⁻¹ ≤ 1 - x + x^2/2`!
      have hinv2 : (1 + v / 2 + (v / 2) ^ 2 / 2)⁻¹ ≤ 1 - v / 2 + (v / 2) ^ 2 / 2 := by
        rw [inv_le_iff_one_le_mul₀ (by positivity)]
        nlinarith [sq_nonneg ((v / 2) ^ 2)]
      have hcube : 0 < (v / 2) ^ 3 / 6 := by positivity
      linarith
    linarith
  have hratio_lt1 : v / (2 * Real.sinh (v / 2)) < 1 :=
    (div_lt_one (lt_trans hv_pos hsinh_gt)).mpr hsinh_gt
  have hratio_nn : 0 ≤ v / (2 * Real.sinh (v / 2)) := by
    exact div_nonneg (le_of_lt hv_pos) (le_of_lt (lt_trans hv_pos hsinh_gt))
  nlinarith

/-- **Exact Euler–Mascheroni Cancellation Identity** (Theorem 7):
Writing `d = 1 - γ` and `D(0) = S₀ - γ` (where `S₀ = e⁻¹ ∑_{k ≥ 1} H_k / (k - 1)!`),
`d - (1 - u) D(0) = 1 - (1 - u) S₀ - u γ`, so `γ` cancels completely at `u = 0` and only
contributes the nonpositive term `-u γ ≤ 0` for `u ∈ [0, 1], γ ≥ 0`. -/
theorem euler_mascheroni_cancellation (S₀ gamma u : ℝ) :
    (1 - gamma) - (1 - u) * (S₀ - gamma) = 1 - (1 - u) * S₀ - u * gamma := by
  ring

/-- The Theorem 7 dyadic-packing constant `U` over scales `j = -12, ..., 5`
(rewritten identically via summation by parts):
`U = -∑_{j=-12}^5 (r_{j+1} - r_j) log (2^j) + r_{-12} (1 - log r_{-12})`,
where `r_6 = 1` and `r_j = dyadicRem (2^j)` for `j ≤ 5`. -/
noncomputable def polyDyadicU : ℝ :=
  -(∑ j ∈ Finset.Icc (-12 : ℤ) 5,
    ((if j = 5 then (1 : ℝ) else dyadicRem ((2 : ℝ) ^ (j + 1))) - dyadicRem ((2 : ℝ) ^ j)) *
      Real.log ((2 : ℝ) ^ j)) +
  dyadicRem ((2 : ℝ) ^ (-12 : ℤ)) * (1 - Real.log (dyadicRem ((2 : ℝ) ^ (-12 : ℤ))))

/-- Exact finite partial sum `S₀^{(6)} = e⁻¹ ∑_{k=1}^6 H_k / (k - 1)!`, a strict lower bound
for `D(0) + γ = e⁻¹ ∑_{k=1}^∞ H_k / (k - 1)!` (all omitted terms are strictly positive). -/
noncomputable def S0Partial6 : ℝ :=
  Real.exp (-1) * (1 + (3 / 2 : ℝ) + 11 / 12 + 25 / 72 + 137 / 1440 + 147 / 7200)

/-- Elementary certified bounds `2.7182 < e < 2.7183` from Mathlib's `Real.exp_bound`. -/
private lemma exp_one_bounds : (27182 / 10000 : ℝ) < Real.exp 1 ∧ Real.exp 1 < 27183 / 10000 := by
  have hb := Real.exp_bound (x := (1 : ℝ)) (by norm_num) (n := 8) (by norm_num)
  simp only [Finset.sum_range_succ, Finset.range_zero, Finset.sum_empty, Nat.factorial,
    Nat.succ_eq_add_one] at hb
  norm_num at hb
  rcases abs_le.mp hb with ⟨hl, hr⟩
  exact ⟨by linarith, by linarith⟩

/-- Elementary certified bound `log 2 < 694 / 1000` (from `e^{0.694} > 2`). -/
private lemma log_two_lt : Real.log 2 < 694 / 1000 := by
  rw [Real.log_lt_iff_lt_exp (by norm_num)]
  have h := Real.sum_le_exp_of_nonneg (show (0 : ℝ) ≤ 694 / 1000 by norm_num) 6
  simp only [Finset.sum_range_succ, Finset.range_zero, Finset.sum_empty, Nat.factorial] at h
  norm_num at h
  linarith

/-- Certified upper bound `polyDyadicU < 377 / 1000` (Theorem 7). -/
private lemma polyDyadicU_lt : polyDyadicU < 377 / 1000 := by
  have ⟨_, he_ub⟩ := exp_one_bounds
  have hlog2 := log_two_lt
  have hlog2_pos : 0 < Real.log 2 := Real.log_pos (by norm_num)
  let f : ℤ → ℝ := fun j => if j = 6 then (1 : ℝ) else dyadicRem ((2 : ℝ) ^ j)
  have hf_eq : ∀ j ∈ Finset.Icc (-12 : ℤ) 5,
      (if j = 5 then (1 : ℝ) else dyadicRem ((2 : ℝ) ^ (j + 1))) = f (j + 1) := by
    intro j _
    dsimp [f]
    by_cases h5 : j = 5
    · subst h5; simp
    · have h6 : j + 1 ≠ 6 := by omega
      simp [h5, h6]
  have hU_eq : polyDyadicU =
      Real.log 2 * (-5 - 12 * dyadicRem (1 / 4096) +
        ∑ j ∈ Finset.Icc (-11 : ℤ) 5, dyadicRem ((2 : ℝ) ^ j)) +
      dyadicRem (1 / 4096) * (1 - Real.log (dyadicRem (1 / 4096))) := by
    unfold polyDyadicU
    have h2m12 : (2 : ℝ) ^ (-12 : ℤ) = 1 / 4096 := by norm_num
    rw [h2m12]
    congr 1
    have hrewrite :
        ∑ j ∈ Finset.Icc (-12 : ℤ) 5,
          ((if j = 5 then (1 : ℝ) else dyadicRem ((2 : ℝ) ^ (j + 1))) - dyadicRem ((2 : ℝ) ^ j)) *
            Real.log ((2 : ℝ) ^ j) =
        Real.log 2 * ∑ j ∈ Finset.Icc (-12 : ℤ) 5, (f (j + 1) - f j) * (j : ℝ) := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun j hj => ?_
      have hj6 : j ≠ 6 := by have := (Finset.mem_Icc.mp hj).2; omega
      rw [hf_eq j hj, Real.log_zpow]
      dsimp [f]
      rw [if_neg hj6]
      ring
    rw [hrewrite]
    have htelescope : ∑ j ∈ Finset.Icc (-12 : ℤ) 5, (f (j + 1) - f j) * (j : ℝ) =
        5 * f 6 + 12 * f (-12) - ∑ j ∈ Finset.Icc (-11 : ℤ) 5, f j := by
      have hsplit : ∑ j ∈ Finset.Icc (-12 : ℤ) 5, (f (j + 1) - f j) * (j : ℝ) =
          (∑ j ∈ Finset.Icc (-12 : ℤ) 5, f (j + 1) * (j : ℝ)) -
          (∑ j ∈ Finset.Icc (-12 : ℤ) 5, f j * (j : ℝ)) := by
        rw [← Finset.sum_sub_distrib]
        refine Finset.sum_congr rfl fun j _ => by ring
      rw [hsplit]
      have hshift : ∑ j ∈ Finset.Icc (-12 : ℤ) 5, f (j + 1) * (j : ℝ) =
          ∑ k ∈ Finset.Icc (-11 : ℤ) 6, f k * ((k : ℝ) - 1) := by
        refine Finset.sum_nbij (· + 1) ?_ ?_ ?_ ?_
        · intro j hj; simp only [Finset.mem_Icc] at hj ⊢; omega
        · intro a _ b _ h; exact add_right_cancel h
        · intro k hk
          simp only [Finset.mem_coe, Finset.mem_Icc] at hk
          refine ⟨k - 1, Finset.mem_coe.mpr (Finset.mem_Icc.mpr ⟨by omega, by omega⟩), ?_⟩
          dsimp only; omega
        · intro j _
          push_cast; ring
      rw [hshift]
      have hIcc6 : Finset.Icc (-11 : ℤ) 6 = insert 6 (Finset.Icc (-11 : ℤ) 5) := by
        ext k; simp only [Finset.mem_Icc, Finset.mem_insert]; omega
      have hIcc12 : Finset.Icc (-12 : ℤ) 5 = insert (-12) (Finset.Icc (-11 : ℤ) 5) := by
        ext k; simp only [Finset.mem_Icc, Finset.mem_insert]; omega
      rw [hIcc6, Finset.sum_insert (by simp), hIcc12, Finset.sum_insert (by simp)]
      have hdiff : (∑ k ∈ Finset.Icc (-11 : ℤ) 5, f k * ((k : ℝ) - 1)) -
          (∑ k ∈ Finset.Icc (-11 : ℤ) 5, f k * (k : ℝ)) =
          -(∑ k ∈ Finset.Icc (-11 : ℤ) 5, f k) := by
        rw [← Finset.sum_sub_distrib, ← Finset.sum_neg_distrib]
        refine Finset.sum_congr rfl fun k _ => by ring
      push_cast
      linarith
    have hfsum : ∑ j ∈ Finset.Icc (-11 : ℤ) 5, f j =
        ∑ j ∈ Finset.Icc (-11 : ℤ) 5, dyadicRem ((2 : ℝ) ^ j) := by
      refine Finset.sum_congr rfl fun j hj => ?_
      have hj6 : j ≠ 6 := by have := (Finset.mem_Icc.mp hj).2; omega
      simp [f, hj6]
    have hf6 : f 6 = 1 := by simp [f]
    have hf12 : f (-12) = dyadicRem (1 / 4096) := by
      simp only [f, show (-12 : ℤ) ≠ 6 by omega, ↓reduceIte, h2m12]
    rw [htelescope, hfsum, hf6, hf12]
    ring
  have hr_neg_sum : ∑ j ∈ Finset.Icc (-11 : ℤ) (-1), dyadicRem ((2 : ℝ) ^ j) < 1 / 2 := by
    have h1 : ∑ j ∈ Finset.Icc (-11 : ℤ) (-1), dyadicRem ((2 : ℝ) ^ j) <
        ∑ j ∈ Finset.Icc (-11 : ℤ) (-1), ((2 : ℝ) ^ j) / 2 :=
      Finset.sum_lt_sum_of_nonempty (⟨-1, Finset.mem_Icc.mpr ⟨by omega, le_rfl⟩⟩) fun j _ =>
        dyadicRem_lt_half (by positivity)
    have h2 : ∑ j ∈ Finset.Icc (-11 : ℤ) (-1), ((2 : ℝ) ^ j) / 2 < 1 / 2 := by
      have hgeom : ∑ j ∈ Finset.Icc (-11 : ℤ) (-1), ((2 : ℝ) ^ j) / 2 =
          ∑ k ∈ Finset.range 11, ((1 / 4096 : ℝ) * (2 : ℝ) ^ k) := by
        refine Finset.sum_nbij (fun j => (j + 11).toNat) ?_ ?_ ?_ ?_
        · intro j hj; simp only [Finset.mem_Icc, Finset.mem_range] at hj ⊢; omega
        · intro a ha b hb h
          simp only [Finset.mem_coe, Finset.mem_Icc] at ha hb
          dsimp only at h
          omega
        · intro k hk
          simp only [Finset.mem_coe, Finset.mem_range] at hk
          refine ⟨(k : ℤ) - 11, Finset.mem_coe.mpr (Finset.mem_Icc.mpr ⟨by omega, by omega⟩), ?_⟩
          dsimp only; omega
        · intro j hj
          simp only [Finset.mem_Icc] at hj
          have hk : (((j + 11).toNat : ℤ)) = j + 11 := Int.toNat_of_nonneg (by omega)
          rw [← zpow_natCast, hk, zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
          norm_num
          ring
      rw [hgeom, ← Finset.mul_sum, geom_sum_eq (by norm_num : (2 : ℝ) ≠ 1)]
      norm_num
    linarith
  have hr0 : dyadicRem 1 < 42 / 100 := by
    unfold dyadicRem
    have hden1 : (0 : ℝ) < Real.exp 1 - 1 := by
      have := Real.one_lt_exp_iff.mpr (show (0 : ℝ) < 1 by norm_num); linarith
    have hdiv : (1 : ℝ) / (27183 / 10000 - 1) < 1 / (Real.exp 1 - 1) :=
      one_div_lt_one_div_of_lt hden1 (by linarith)
    rw [one_div] at hdiv ⊢
    norm_num at hdiv
    linarith
  have hr1 : dyadicRem 2 < 69 / 100 := by
    unfold dyadicRem
    have he2_ub : Real.exp 2 < (27183 / 10000 : ℝ) ^ 2 := by
      rw [show (2 : ℝ) = (2 : ℕ) * 1 by norm_num, Real.exp_nat_mul]
      exact pow_lt_pow_left₀ he_ub (le_of_lt (Real.exp_pos 1)) (by norm_num)
    have hden : (0 : ℝ) < Real.exp 2 - 1 := by
      have := Real.one_lt_exp_iff.mpr (show (0 : ℝ) < 2 by norm_num); linarith
    have hdiv : (2 : ℝ) / ((27183 / 10000 : ℝ) ^ 2 - 1) < 2 / (Real.exp 2 - 1) :=
      div_lt_div_of_pos_left (by norm_num) hden (by linarith)
    norm_num at hdiv
    linarith
  have hr2 : dyadicRem 4 < 93 / 100 := by
    unfold dyadicRem
    have he4_ub : Real.exp 4 < (27183 / 10000 : ℝ) ^ 4 := by
      rw [show (4 : ℝ) = (4 : ℕ) * 1 by norm_num, Real.exp_nat_mul]
      exact pow_lt_pow_left₀ he_ub (le_of_lt (Real.exp_pos 1)) (by norm_num)
    have hden : (0 : ℝ) < Real.exp 4 - 1 := by
      have := Real.one_lt_exp_iff.mpr (show (0 : ℝ) < 4 by norm_num); linarith
    have hdiv : (4 : ℝ) / ((27183 / 10000 : ℝ) ^ 4 - 1) < 4 / (Real.exp 4 - 1) :=
      div_lt_div_of_pos_left (by norm_num) hden (by linarith)
    norm_num at hdiv
    linarith
  have hr_lt1 : ∀ z : ℝ, 0 < z → dyadicRem z < 1 := by
    intro z hz
    unfold dyadicRem
    have hexp : 0 < Real.exp z - 1 := by have := Real.one_lt_exp_iff.mpr hz; linarith
    have : 0 < z / (Real.exp z - 1) := div_pos hz hexp
    linarith
  have hsum_all : ∑ j ∈ Finset.Icc (-11 : ℤ) 5, dyadicRem ((2 : ℝ) ^ j) < 554 / 100 := by
    have hsplit : Finset.Icc (-11 : ℤ) 5 =
        Finset.Icc (-11 : ℤ) (-1) ∪ Finset.Icc (0 : ℤ) 5 := by
      ext k; simp only [Finset.mem_Icc, Finset.mem_union]; omega
    have hdisj : Disjoint (Finset.Icc (-11 : ℤ) (-1)) (Finset.Icc (0 : ℤ) 5) := by
      rw [Finset.disjoint_left]; intro k hk1 hk2
      simp only [Finset.mem_Icc] at hk1 hk2; omega
    have hpos_sum : ∑ j ∈ Finset.Icc (0 : ℤ) 5, dyadicRem ((2 : ℝ) ^ j) =
        dyadicRem 1 + dyadicRem 2 + dyadicRem 4 + dyadicRem 8 + dyadicRem 16 + dyadicRem 32 := by
      have h05 : Finset.Icc (0 : ℤ) 5 = {0, 1, 2, 3, 4, 5} := by
        ext k; simp only [Finset.mem_Icc, Finset.mem_insert, Finset.mem_singleton]; omega
      rw [h05]
      simp only [Finset.sum_insert, Finset.mem_insert, Finset.mem_singleton, Finset.sum_singleton,
        Int.reduceEq, or_self, not_false_eq_true]
      norm_num
      ring
    rw [hsplit, Finset.sum_union hdisj, hpos_sum]
    have hr3 := hr_lt1 8 (by norm_num)
    have hr4 := hr_lt1 16 (by norm_num)
    have hr5 := hr_lt1 32 (by norm_num)
    linarith
  set r12 : ℝ := dyadicRem (1 / 4096)
  have hr12_ub : r12 < (1 : ℝ) / 8192 := by
    have h := dyadicRem_lt_half (z := (1 / 4096 : ℝ)) (by norm_num)
    norm_num at h
    exact h
  have hr12_lb : (1 : ℝ) / 16384 < r12 := by
    have h := dyadicRem_ge (z := (1 / 4096 : ℝ)) (by norm_num)
    norm_num at h
    linarith
  have hr12_pos : 0 < r12 := by linarith
  have hlog_r12 : -14 * Real.log 2 < Real.log r12 := by
    have h := Real.log_lt_log (by norm_num : (0 : ℝ) < 1 / 16384) hr12_lb
    have h16 : Real.log ((1 : ℝ) / 16384) = -14 * Real.log 2 := by
      have : (1 : ℝ) / 16384 = ((2 : ℝ) ^ (14 : ℕ))⁻¹ := by norm_num
      rw [this, Real.log_inv, Real.log_pow]
      push_cast; ring
    linarith
  have hres_ub : r12 * (1 - Real.log r12) < 14 / 10000 := by
    have h1 : 1 - Real.log r12 < 1 + 14 * Real.log 2 := by linarith
    have h2 : 1 + 14 * Real.log 2 < 11 := by linarith
    have h3 : r12 * (1 - Real.log r12) < (1 / 8192 : ℝ) * 11 := by nlinarith
    norm_num at h3 ⊢
    linarith
  rw [hU_eq]
  have hbr : -5 - 12 * r12 + ∑ j ∈ Finset.Icc (-11 : ℤ) 5, dyadicRem ((2 : ℝ) ^ j) < 54 / 100 := by
    linarith
  have hmul : Real.log 2 * (-5 - 12 * r12 + ∑ j ∈ Finset.Icc (-11 : ℤ) 5, dyadicRem ((2 : ℝ) ^ j)) <
      (694 / 1000 : ℝ) * (54 / 100) := by
    by_cases hneg : -5 - 12 * r12 + ∑ j ∈ Finset.Icc (-11 : ℤ) 5, dyadicRem ((2 : ℝ) ^ j) ≤ 0
    · have : Real.log 2 * (-5 - 12 * r12 + ∑ j ∈ Finset.Icc (-11 : ℤ) 5, dyadicRem ((2 : ℝ) ^ j)) ≤ 0 :=
        mul_nonpos_of_nonneg_of_nonpos (le_of_lt hlog2_pos) hneg
      norm_num
      linarith
    · push_neg at hneg
      nlinarith
  norm_num at hmul ⊢
  linarith

/-- **Theorem 7 Certified Zero-Offset Polynomial-Alphabet Counterexample (`m = n^{1024}, c = 0`)**:
For any `γ ≥ 0` and any `S₀ ≥ S0Partial6` (matching `D(0) = S₀ - γ`) and `D_u ≥ (1 - u) D(0) - h(u)`
at `u = 1 / 1024`, the asymptotic zero-offset total-entropy gap upper bound satisfies
`polyDyadicU + (1 - γ) - D_u < -3 / 100 < 0`! -/
theorem poly_zero_offset_gap_strictly_neg {gamma S₀ D_u : ℝ}
    (hgamma : 0 ≤ gamma) (hS₀ : S0Partial6 ≤ S₀)
    (hDu : (1 - (1 : ℝ) / 1024) * (S₀ - gamma) - binEntropy (1 / 1024) ≤ D_u) :
    polyDyadicU + (1 - gamma) - D_u < -(3 / 100 : ℝ) := by
  have ⟨_, he_ub⟩ := exp_one_bounds
  have hlog2 := log_two_lt
  have hS06_lb : (14271 / 10000 : ℝ) < S0Partial6 := by
    unfold S0Partial6
    rw [Real.exp_neg]
    have hinv : (27183 / 10000 : ℝ)⁻¹ < (Real.exp 1)⁻¹ :=
      inv_strictAnti₀ (Real.exp_pos 1) he_ub
    have hcoeff : (0 : ℝ) < 1 + (3 / 2 : ℝ) + 11 / 12 + 25 / 72 + 137 / 1440 + 147 / 7200 := by
      norm_num
    have hmul := mul_lt_mul_of_pos_right hinv hcoeff
    norm_num at hmul ⊢
    linarith
  have hbin_ub : binEntropy (1 / 1024) < 8 / 1024 := by
    unfold binEntropy
    have h1 : Real.log ((1 : ℝ) / 1024) = -10 * Real.log 2 := by
      have : (1 : ℝ) / 1024 = ((2 : ℝ) ^ (10 : ℕ))⁻¹ := by norm_num
      rw [this, Real.log_inv, Real.log_pow]
      push_cast; ring
    have h2 : -((1023 : ℝ) / 1024 * Real.log (1023 / 1024)) ≤ 1 / 1024 := by
      have hlog : Real.log (1024 / 1023 : ℝ) ≤ 1024 / 1023 - 1 :=
        Real.log_le_sub_one_of_pos (by norm_num)
      have hinv : -Real.log (1023 / 1024 : ℝ) = Real.log (1024 / 1023 : ℝ) := by
        rw [← Real.log_inv]; norm_num
      have hmul : (1023 : ℝ) / 1024 * (-Real.log (1023 / 1024 : ℝ)) ≤
          (1023 : ℝ) / 1024 * (1024 / 1023 - 1) := by
        rw [hinv]
        exact mul_le_mul_of_nonneg_left hlog (by norm_num)
      norm_num at hmul
      linarith
    have h1m : (1 : ℝ) - 1 / 1024 = 1023 / 1024 := by norm_num
    rw [h1, h1m]
    linarith
  have hU_ub : polyDyadicU < 377 / 1000 := polyDyadicU_lt
  have hcancel := euler_mascheroni_cancellation S₀ gamma (1 / 1024)
  have hS0_term : (1 - (1 : ℝ) / 1024) * (14271 / 10000) < (1 - (1 : ℝ) / 1024) * S₀ :=
    mul_lt_mul_of_pos_left (lt_of_lt_of_le hS06_lb hS₀) (by norm_num)
  have hgamma_term : 0 ≤ (1 : ℝ) / 1024 * gamma := by positivity
  norm_num at hS0_term ⊢
  linarith

end EntropicCausalIdent

