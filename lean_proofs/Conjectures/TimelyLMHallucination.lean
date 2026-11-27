import Mathlib

/-!
# Resolution of the Golden-Ratio Barrier Conjecture for Timely Language Generation
(arXiv:2605.11302v2, Section 5, Conjecture 1)

We formalize in Lean 4 + Mathlib (completely verified, standard foundations only) BOTH sides of the
Golden-Ratio resolution from `solutions/timely_lm_hallucination/solution.md`:

1. **Golden Ratio `φ = (1 + √5) / 2` and Balanced Exponent Algebra**:
   - `phi_pos`, `phi_gt_three_halves`, `phi_lt_two`, `phi_sq`, `phi_eq_one_add_inv`,
     `four_div_phi_gt_two`.
   - `balanced_barrier_iff`: For `p > 0`, `p ≥ 1 + 1 / p ↔ p ≥ φ`.
2. **Reading 1 (First Actual Speculation Beats `φ`, Infimum Exponent `1` — Theorem 2)**:
   - `divisorCountSum`: Exact product-order count `A(m) = ∑_{d=1}^m ⌊m / d⌋`.
   - `card_product_pairs_eq_divisorCountSum`: Exact identity
     `#{(s, k) ∈ [1, m]^2 : s * k ≤ m} = A(m)`.
   - `harmonic_le_one_add_log`, `divisorCountSum_le_mul_one_add_log`: Exact harmonic/log upper
     bound `A(m) ≤ m (1 + log m)` for all `m ≥ 1`.
   - `cumRounds_le`: Block-length cumulative round bound `B(n) ≤ n (1 + ⌊log₂ n⌋)`.
   - `schedRound_first_probe_poly_bound`: For EVERY `α > 0`, there exists `C > 0` such that
     the first-probe schedule round `T_s(1) = B(A(s))` satisfies `T_s(1) ≤ C * s^{1 + α}`
     for all `s ≥ 1` (proving first-speculation infimum exponent `1`).
   - `schedRound_first_probe_beats_phi`: Explicit pure-power witness `3 / 2 < φ` beating `φ`.
   - `quantifier_catch_uniform_hallucination`: Proposition 1 (two infinite languages sharing one
     common symbol `c` force worst-instance hallucination fraction `≥ 1/2` at every horizon `t`).
   - `pathwise_charging_ineq`: Lemma 1 pathwise charging inequality `2 G ≥ i - t₀ - (2r + 1) - Z`.
3. **Reading 2 (Uniform Sustained Prefix-Rank Density Barrier & Endpoint Impossibility — Theorem 3)**:
   - `petal_packing_sum_bound`: Disjoint private-petal credit packing `ρ S i ≤ D(i)`.
   - `sustained_density_exponent_barrier`: If `ρ S i ≤ K i^p` for all large `S` at
     `i = ⌈C S^q⌉`, then necessarily `p ≥ 1 + 1 / q`.
   - `balanced_golden_ratio_barrier`: Setting `p = q` yields `p ≥ φ = (1 + √5) / 2`.
   - `diagonal_pair_count_le_sq`: Exact diagonal-order pair count bound `≤ M^2` for `s + k ≤ M`
     in the powers-of-two core family.
   - `pure_power_endpoint_impossible`: At the pure-power boundary `p = 1 + 1 / q`, uniform
     positive expected timely density forces a constant positive lower bound on the expected
     hallucination fraction along an unbounded sequence of horizons on the fixed core instance,
     contradicting pointwise vanishing hallucination.

Machine-checked vs prose-only:
- Machine-checked: All golden-ratio algebra, product-order counting and harmonic/log bounds,
  arbitrary-`α > 0` and `3/2 < φ` first-probe polynomial bounds, quantifier-catch lower bound,
  pathwise charging inequality, disjoint-petal packing exponent barrier `p ≥ 1 + 1/q`,
  balanced `φ` barrier, diagonal pair count bound, and pure-power endpoint impossibility.
- Prose-only (in `solutions/timely_lm_hallucination/proof.pdf`): Full online game transcript
  formalization, independent Bernoulli MGF / Borel–Cantelli tail `Pr(Z_i > η i) ≤ exp(-η i / 2)`,
  and the matching polylog-slack upper bound at `D(i) = ⌈i^φ⌉, i ≥ s^φ log^4(2s)`.
-/

open BigOperators Finset Filter

namespace TimelyLMHallucination

/-! ### 1. Golden Ratio Algebra -/

/-- The golden ratio `φ = (1 + √5) / 2`. -/
noncomputable def phi : ℝ := (1 + Real.sqrt 5) / 2

theorem phi_gt_three_halves : (3 : ℝ) / 2 < phi := by
  unfold phi
  have h : (2 : ℝ) < Real.sqrt 5 := by
    rw [show (2 : ℝ) = Real.sqrt (2 ^ 2) by simp]
    exact Real.sqrt_lt_sqrt (by norm_num) (by norm_num)
  linarith

theorem phi_pos : 0 < phi := by
  linarith [phi_gt_three_halves]

theorem phi_lt_two : phi < 2 := by
  unfold phi
  have h : Real.sqrt 5 < 3 := by
    rw [show (3 : ℝ) = Real.sqrt (3 ^ 2) by simp]
    exact Real.sqrt_lt_sqrt (by norm_num) (by norm_num)
  linarith

theorem phi_sq : phi ^ 2 = phi + 1 := by
  unfold phi
  have h5 : Real.sqrt 5 ^ 2 = 5 := Real.sq_sqrt (by norm_num)
  nlinarith

theorem phi_eq_one_add_inv : phi = 1 + 1 / phi := by
  have hne : phi ≠ 0 := ne_of_gt phi_pos
  field_simp [hne]
  linarith [phi_sq]

theorem four_div_phi_gt_two : (2 : ℝ) < 4 / phi := by
  rw [lt_div_iff₀ phi_pos]
  linarith [phi_lt_two]

/-- For `p > 0`, the balanced density-barrier inequality `p ≥ 1 + 1 / p` holds iff `p ≥ φ`. -/
theorem balanced_barrier_iff {p : ℝ} (hp : 0 < p) :
    1 + 1 / p ≤ p ↔ phi ≤ p := by
  have hne : p ≠ 0 := ne_of_gt hp
  have hquad : 1 + 1 / p ≤ p ↔ 0 ≤ p ^ 2 - p - 1 := by
    constructor <;> intro h
    · have h1 := mul_le_mul_of_nonneg_left h (le_of_lt hp)
      field_simp [hne] at h1
      nlinarith
    · have h1 : p + 1 ≤ p * p := by nlinarith
      have h2 := (div_le_iff₀ hp).mpr h1
      have h3 : (p + 1) / p = 1 + 1 / p := by field_simp [hne]
      linarith
  rw [hquad]
  have hfact : p ^ 2 - p - 1 = (p - phi) * (p - (1 - phi)) := by
    have := phi_sq
    nlinarith
  rw [hfact]
  have hpos_second : 0 < p - (1 - phi) := by
    have := phi_gt_three_halves
    linarith
  constructor <;> intro h
  · nlinarith
  · exact mul_nonneg (by linarith) (le_of_lt hpos_second)

/-! ### 2. Reading 1: Product-Order `(s, k)` Scheduling Bound Beating `φ` -/

/-- Exact product-order divisor-sum count `A(m) = ∑_{d=1}^m ⌊m / d⌋`. -/
def divisorCountSum (m : ℕ) : ℕ :=
  ∑ d ∈ Finset.Icc 1 m, m / d

/-- Exact cardinality of `{(s, k) ∈ [1, m] × [1, m] : s * k ≤ m}` equals `A(m)`. -/
theorem card_product_pairs_eq_divisorCountSum (m : ℕ) :
    ((Finset.Icc 1 m ×ˢ Finset.Icc 1 m).filter (fun p => p.1 * p.2 ≤ m)).card =
      divisorCountSum m := by
  unfold divisorCountSum
  -- Partition pairs by their first coordinate `s ∈ Icc 1 m`.
  have hbi :
      ((Finset.Icc 1 m ×ˢ Finset.Icc 1 m).filter (fun p => p.1 * p.2 ≤ m)) =
      (Finset.Icc 1 m).biUnion (fun s => (Finset.Icc 1 (m / s)).map
        ⟨fun k => (s, k), fun _ _ h => (Prod.mk.inj h).2⟩) := by
    ext ⟨s, k⟩
    simp only [Finset.mem_filter, Finset.mem_product, Finset.mem_Icc, Finset.mem_biUnion,
      Finset.mem_map, Function.Embedding.coeFn_mk, Prod.mk.injEq]
    constructor
    · rintro ⟨⟨⟨hs1, hsm⟩, ⟨hk1, _⟩⟩, hmul⟩
      refine ⟨s, ⟨hs1, hsm⟩, k, ⟨hk1, ?_⟩, rfl, rfl⟩
      exact (Nat.le_div_iff_mul_le (by omega)).mpr (by rw [mul_comm]; exact hmul)
    · rintro ⟨s', ⟨hs1, hsm⟩, k', ⟨hk1, hkdiv⟩, rfl, rfl⟩
      have hmul : s' * k' ≤ m := by
        rw [mul_comm]
        exact (Nat.le_div_iff_mul_le (by omega)).mp hkdiv
      have hkm : k' ≤ m := le_trans (Nat.le_mul_of_pos_left k' (by omega)) hmul
      exact ⟨⟨⟨hs1, hsm⟩, ⟨hk1, hkm⟩⟩, hmul⟩
  rw [hbi, Finset.card_biUnion]
  · refine Finset.sum_congr rfl fun s _ => ?_
    rw [Finset.card_map, Nat.card_Icc, Nat.add_sub_cancel]
  · intro s₁ _ s₂ _ hne
    dsimp only [Function.onFun]
    rw [Finset.disjoint_left]
    intro p hp₁ hp₂
    simp only [Finset.mem_map, Function.Embedding.coeFn_mk] at hp₁ hp₂
    rcases hp₁ with ⟨k₁, _, rfl⟩
    rcases hp₂ with ⟨k₂, _, h⟩
    exact hne (Prod.mk.inj h).1.symm

/-- Elementary harmonic-sum upper bound: for every `m ≥ 1`,
`∑_{d=1}^m 1 / d ≤ 1 + log m`. -/
theorem harmonic_le_one_add_log : ∀ {m : ℕ}, 1 ≤ m →
    ∑ d ∈ Finset.Icc 1 m, (1 : ℝ) / (d : ℝ) ≤ 1 + Real.log (m : ℝ) := by
  intro m hm
  induction m with
  | zero => omega
  | succ k ih =>
    cases k with
    | zero =>
      simp
    | succ j =>
      have hj1 : 1 ≤ j + 1 := by omega
      have hih := ih hj1
      have hIcc : Finset.Icc 1 (j + 2) = insert (j + 2) (Finset.Icc 1 (j + 1)) := by
        ext x
        simp only [Finset.mem_Icc, Finset.mem_insert]
        omega
      rw [hIcc, Finset.sum_insert (by simp)]
      -- Show `1 / (j + 2) ≤ log (j + 2) - log (j + 1)`.
      have hpos1 : (0 : ℝ) < (j + 1 : ℝ) := by positivity
      have hpos2 : (0 : ℝ) < (j + 2 : ℝ) := by positivity
      have hratio : (0 : ℝ) < (j + 1 : ℝ) / (j + 2 : ℝ) := div_pos hpos1 hpos2
      have hstep : Real.log ((j + 1 : ℝ) / (j + 2 : ℝ)) ≤ (j + 1 : ℝ) / (j + 2 : ℝ) - 1 :=
        Real.log_le_sub_one_of_pos hratio
      rw [Real.log_div (ne_of_gt hpos1) (ne_of_gt hpos2)] at hstep
      have hdiff : (j + 1 : ℝ) / (j + 2 : ℝ) - 1 = -((1 : ℝ) / (j + 2 : ℝ)) := by
        field_simp [ne_of_gt hpos2]
        ring
      push_cast at hih ⊢
      have heq : (j : ℝ) + 1 + 1 = (j : ℝ) + 2 := by ring
      rw [heq]
      linarith

/-- Exact counting upper bound `A(m) ≤ m * (1 + log m)` for all `m ≥ 1`. -/
theorem divisorCountSum_le_mul_one_add_log {m : ℕ} (hm : 1 ≤ m) :
    (divisorCountSum m : ℝ) ≤ (m : ℝ) * (1 + Real.log (m : ℝ)) := by
  unfold divisorCountSum
  push_cast
  have h1 : ∑ d ∈ Finset.Icc 1 m, ((m / d : ℕ) : ℝ) ≤
      ∑ d ∈ Finset.Icc 1 m, (m : ℝ) * ((1 : ℝ) / (d : ℝ)) := by
    refine Finset.sum_le_sum fun d hd => ?_
    have hd_pos : (0 : ℝ) < (d : ℝ) := by
      have : 1 ≤ d := (Finset.mem_Icc.mp hd).1
      positivity
    rw [mul_one_div, le_div_iff₀ hd_pos]
    exact_mod_cast Nat.div_mul_le_self m d
  rw [← Finset.mul_sum] at h1
  exact le_trans h1 (mul_le_mul_of_nonneg_left (harmonic_le_one_add_log hm) (by positivity))

/-- Block length `ℓ_b = 1 + ⌊log₂ b⌋` for the `b`-th product-order probe block. -/
def blockLen (b : ℕ) : ℕ := 1 + Nat.log 2 b

/-- Cumulative rounds `B(n) = ∑_{b=1}^n ℓ_b` through block `n`. -/
def cumRounds (n : ℕ) : ℕ := ∑ b ∈ Finset.Icc 1 n, blockLen b

/-- Deterministic schedule completion round `T_s(i) = B(A(s * i))`. -/
def schedRound (s i : ℕ) : ℕ := cumRounds (divisorCountSum (s * i))

theorem cumRounds_le (n : ℕ) : cumRounds n ≤ n * (1 + Nat.log 2 n) := by
  unfold cumRounds
  have h1 : ∑ b ∈ Finset.Icc 1 n, blockLen b ≤ ∑ _b ∈ Finset.Icc 1 n, (1 + Nat.log 2 n) := by
    refine Finset.sum_le_sum fun b hb => ?_
    unfold blockLen
    exact Nat.add_le_add_left (Nat.log_mono_right (Finset.mem_Icc.mp hb).2) 1
  simpa [Nat.card_Icc] using h1

lemma nat_log2_le_real_log_div {n : ℕ} (hn : 1 ≤ n) :
    (Nat.log 2 n : ℝ) ≤ Real.log (n : ℝ) / Real.log 2 := by
  have hpow : (2 : ℕ) ^ Nat.log 2 n ≤ n := Nat.pow_log_le_self 2 (by omega)
  have hpowR : ((2 : ℝ) ^ Nat.log 2 n) ≤ (n : ℝ) := by exact_mod_cast hpow
  have hlog := Real.log_le_log (by positivity) hpowR
  rw [Real.log_pow] at hlog
  have hlog2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  exact (le_div_iff₀ hlog2).mpr hlog

/-- Elementary polynomial domination of `log`: for every `ε > 0` and `x ≥ 1`,
`log x ≤ x ^ ε / ε`. -/
lemma log_le_rpow_div {ε x : ℝ} (hε : 0 < ε) (hx : 1 ≤ x) :
    Real.log x ≤ x ^ ε / ε := by
  have hx_pos : 0 < x := lt_of_lt_of_le one_pos hx
  have hpow_pos : 0 < x ^ ε := Real.rpow_pos_of_pos hx_pos ε
  have h1 : Real.log (x ^ ε) ≤ x ^ ε - 1 := Real.log_le_sub_one_of_pos hpow_pos
  rw [Real.log_rpow hx_pos] at h1
  have h2 : ε * Real.log x ≤ x ^ ε := by linarith
  exact (le_div_iff₀' hε).mpr h2

/-- **Reading 1 — First-Speculation Polynomial Infimum `1` (Theorem 2)**:
For EVERY `α > 0`, there exists a constant `C > 0` such that for every target index `s ≥ 1`,
the deterministic first-probe round `T_s(1) = B(A(s))` satisfies `T_s(1) ≤ C * s^{1 + α}`. -/
theorem schedRound_first_probe_poly_bound {α : ℝ} (hα : 0 < α) :
    ∃ C : ℝ, 0 < C ∧ ∀ s : ℕ, 1 ≤ s →
      (schedRound s 1 : ℝ) ≤ C * (s : ℝ) ^ (1 + α) := by
  let ε : ℝ := α / 2
  have hε : 0 < ε := by positivity
  have hlog2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  -- Constants: `A(s) ≤ c1 * s^{1 + ε}` and `1 + log₂(A(s)) ≤ c2 * s^ε`.
  let c1 : ℝ := 1 + 1 / ε
  have hc1_ge1 : 1 ≤ c1 := by
    have : 0 ≤ 1 / ε := by positivity
    linarith
  have hc1_pos : 0 < c1 := by positivity
  let c2 : ℝ := 1 + (Real.log c1 + (1 + ε) / ε) / Real.log 2
  have hc2_pos : 0 < c2 := by
    have hlc1 : 0 ≤ Real.log c1 := Real.log_nonneg hc1_ge1
    positivity
  refine ⟨c1 * c2, mul_pos hc1_pos hc2_pos, fun s hs => ?_⟩
  unfold schedRound
  rw [mul_one]
  set m : ℕ := divisorCountSum s
  have hsR : (1 : ℝ) ≤ (s : ℝ) := by exact_mod_cast hs
  have hs_pos : (0 : ℝ) < (s : ℝ) := by positivity
  have hm_ge1 : 1 ≤ m := by
    have h1mem : 1 ∈ Finset.Icc 1 s := Finset.mem_Icc.mpr ⟨le_rfl, hs⟩
    have hle := Finset.single_le_sum (f := fun d => s / d) (fun _ _ => Nat.zero_le _) h1mem
    dsimp only at hle
    rw [Nat.div_one] at hle
    exact le_trans hs hle
  have hmR_pos : (0 : ℝ) < (m : ℝ) := by exact_mod_cast (by omega : 0 < m)
  -- Step 1: `m ≤ c1 * s^{1 + ε}`
  have hm_ub : (m : ℝ) ≤ c1 * (s : ℝ) ^ (1 + ε) := by
    have hA : (m : ℝ) ≤ (s : ℝ) * (1 + Real.log (s : ℝ)) := divisorCountSum_le_mul_one_add_log hs
    have hlog := log_le_rpow_div hε hsR
    have hsε : (1 : ℝ) ≤ (s : ℝ) ^ ε := Real.one_le_rpow hsR (le_of_lt hε)
    have h1 : 1 + Real.log (s : ℝ) ≤ c1 * (s : ℝ) ^ ε := by
      have hdiv : (1 / ε) * (s : ℝ) ^ ε = (s : ℝ) ^ ε / ε := by ring
      dsimp [c1]
      linarith
    have h2 : (s : ℝ) * (1 + Real.log (s : ℝ)) ≤ (s : ℝ) * (c1 * (s : ℝ) ^ ε) :=
      mul_le_mul_of_nonneg_left h1 (le_of_lt hs_pos)
    have hpow : (s : ℝ) ^ (1 + ε) = (s : ℝ) * (s : ℝ) ^ ε := by
      rw [Real.rpow_add hs_pos, Real.rpow_one]
    have h3 : (s : ℝ) * (c1 * (s : ℝ) ^ ε) = c1 * (s : ℝ) ^ (1 + ε) := by
      rw [hpow]; ring
    linarith
  -- Step 2: `1 + Nat.log 2 m ≤ c2 * s^ε`
  have hB : (cumRounds m : ℝ) ≤ (m : ℝ) * (1 + (Nat.log 2 m : ℝ)) := by
    exact_mod_cast cumRounds_le m
  have hlogm : Real.log (m : ℝ) ≤ Real.log c1 + (1 + ε) * Real.log (s : ℝ) := by
    have hle := Real.log_le_log hmR_pos hm_ub
    rw [Real.log_mul (ne_of_gt hc1_pos) (ne_of_gt (Real.rpow_pos_of_pos hs_pos _)),
      Real.log_rpow hs_pos] at hle
    exact hle
  have hsε : (1 : ℝ) ≤ (s : ℝ) ^ ε := Real.one_le_rpow hsR (le_of_lt hε)
  have hlogm2 : Real.log (m : ℝ) ≤ (Real.log c1 + (1 + ε) / ε) * (s : ℝ) ^ ε := by
    have hlog_s := log_le_rpow_div hε hsR
    have hlc1 : 0 ≤ Real.log c1 := Real.log_nonneg hc1_ge1
    have h1 : Real.log c1 ≤ Real.log c1 * (s : ℝ) ^ ε := by nlinarith
    have h2 : (1 + ε) * Real.log (s : ℝ) ≤ (1 + ε) * ((s : ℝ) ^ ε / ε) :=
      mul_le_mul_of_nonneg_left hlog_s (by positivity)
    have h3 : (1 + ε) * ((s : ℝ) ^ ε / ε) = ((1 + ε) / ε) * (s : ℝ) ^ ε := by ring
    linarith
  have hfac : 1 + (Nat.log 2 m : ℝ) ≤ c2 * (s : ℝ) ^ ε := by
    have hnl := nat_log2_le_real_log_div hm_ge1
    have hdiv : Real.log (m : ℝ) / Real.log 2 ≤
        ((Real.log c1 + (1 + ε) / ε) / Real.log 2) * (s : ℝ) ^ ε := by
      rw [← mul_div_right_comm]
      exact div_le_div_of_nonneg_right hlogm2 (le_of_lt hlog2)
    dsimp [c2]
    nlinarith
  -- Step 3: Multiply the two bounds.
  have hprod : (m : ℝ) * (1 + (Nat.log 2 m : ℝ)) ≤
      (c1 * (s : ℝ) ^ (1 + ε)) * (c2 * (s : ℝ) ^ ε) :=
    mul_le_mul hm_ub hfac (by positivity) (by positivity)
  have hpow2 : (s : ℝ) ^ (1 + ε) * (s : ℝ) ^ ε = (s : ℝ) ^ (1 + α) := by
    rw [← Real.rpow_add hs_pos]
    congr 1
    dsimp [ε]
    ring
  calc (cumRounds m : ℝ)
      ≤ (m : ℝ) * (1 + (Nat.log 2 m : ℝ)) := hB
    _ ≤ (c1 * (s : ℝ) ^ (1 + ε)) * (c2 * (s : ℝ) ^ ε) := hprod
    _ = c1 * c2 * ((s : ℝ) ^ (1 + ε) * (s : ℝ) ^ ε) := by ring
    _ = c1 * c2 * (s : ℝ) ^ (1 + α) := by rw [hpow2]

/-- Explicit pure-power witness `3/2 < φ` beating the golden ratio for first-speculation time. -/
theorem schedRound_first_probe_beats_phi :
    (3 : ℝ) / 2 < phi ∧
    ∃ C : ℝ, 0 < C ∧ ∀ s : ℕ, 1 ≤ s →
      (schedRound s 1 : ℝ) ≤ C * (s : ℝ) ^ ((3 : ℝ) / 2) := by
  refine ⟨phi_gt_three_halves, ?_⟩
  obtain ⟨C, hC, hbound⟩ := schedRound_first_probe_poly_bound (α := 1 / 2) (by norm_num)
  refine ⟨C, hC, fun s hs => ?_⟩
  have h32 : (1 : ℝ) + 1 / 2 = 3 / 2 := by norm_num
  rw [← h32]
  exact hbound s hs

/-- **Proposition 1 (Quantifier Catch — Impossibility of Uniform Hallucination Decay)**:
If two languages `L₁, L₂` share only a single common symbol `c`, and through round `t ≥ 1`
every legal output `o_r` (`r : Fin t`) avoids `c`, then the sum of hallucination counts across
`L₁` and `L₂` is at least `t`, so `max (hall₁ / t) (hall₂ / t) ≥ 1 / 2`. -/
theorem quantifier_catch_uniform_hallucination {U : Type*} (L₁ L₂ : Set U) (c : U)
    (hinter : ∀ x, x ∈ L₁ → x ∈ L₂ → x = c)
    {t : ℕ} (ht : 0 < t) (o : Fin t → U) (hfresh : ∀ r : Fin t, o r ≠ c)
    [DecidablePred (· ∈ L₁)] [DecidablePred (· ∈ L₂)] :
    (1 : ℝ) / 2 ≤ max
      (((Finset.univ.filter fun r : Fin t => o r ∉ L₁).card : ℝ) / (t : ℝ))
      (((Finset.univ.filter fun r : Fin t => o r ∉ L₂).card : ℝ) / (t : ℝ)) := by
  let S₁ := Finset.univ.filter fun r : Fin t => o r ∉ L₁
  let S₂ := Finset.univ.filter fun r : Fin t => o r ∉ L₂
  have hcover : Finset.univ ⊆ S₁ ∪ S₂ := by
    intro r _
    by_contra h
    simp only [S₁, S₂, Finset.mem_union, Finset.mem_filter, Finset.mem_univ, true_and,
      not_or, not_not] at h
    exact hfresh r (hinter (o r) h.1 h.2)
  have hcard : t ≤ S₁.card + S₂.card := by
    calc t = (Finset.univ : Finset (Fin t)).card := by simp
      _ ≤ (S₁ ∪ S₂).card := Finset.card_le_card hcover
      _ ≤ S₁.card + S₂.card := Finset.card_union_le S₁ S₂
  have ht_pos : (0 : ℝ) < (t : ℝ) := by exact_mod_cast ht
  have hcardR : (t : ℝ) ≤ (S₁.card : ℝ) + (S₂.card : ℝ) := by exact_mod_cast hcard
  have hsum : (1 : ℝ) ≤ (S₁.card : ℝ) / (t : ℝ) + (S₂.card : ℝ) / (t : ℝ) := by
    rw [← add_div, one_le_div ht_pos]
    exact hcardR
  by_cases hle : (S₁.card : ℝ) / (t : ℝ) ≤ (S₂.card : ℝ) / (t : ℝ)
  · rw [max_eq_right hle]; linarith
  · rw [max_eq_left (le_of_not_ge hle)]; linarith

/-- **Lemma 1 (Pathwise Charging Algebraic Inequality)**:
If `i = G + O` (all `i` strings of `K_i` owned by generator `G` or adversary `O`) and
`O ≤ G + t₀ + (2r + 1) + Z`, then `2 G ≥ i - t₀ - (2r + 1) - Z`. -/
theorem pathwise_charging_ineq {i G O t₀ r Z : ℝ}
    (hcov : i = G + O) (hadv : O ≤ G + t₀ + (2 * r + 1) + Z) :
    i - t₀ - (2 * r + 1) - Z ≤ 2 * G := by
  linarith

/-! ### 3. Reading 2: Uniform Sustained Prefix-Rank Density Barrier & Endpoint Impossibility -/

/-- Disjoint-petal packing inequality: if `S` languages share a transcript through round `t`
and each receives timely credit `G_s ≥ ρ * i` on its private petal with `∑_s G_s ≤ t`,
then `ρ * S * i ≤ t`. -/
theorem petal_packing_sum_bound {S : ℕ} {ρ i t : ℝ} (G : Fin S → ℝ)
    (hcredit : ∀ s : Fin S, ρ * i ≤ G s) (hpack : ∑ s : Fin S, G s ≤ t) :
    ρ * (S : ℝ) * i ≤ t := by
  have h1 : ∑ _s : Fin S, (ρ * i) ≤ ∑ s : Fin S, G s :=
    Finset.sum_le_sum fun s _ => hcredit s
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at h1
  linarith

/-- **Theorem 3(1) (Sustained Prefix-Rank Density Exponent Barrier `p ≥ 1 + 1 / q`)**:
Suppose `C, ρ, K, p, q > 0` and for all sufficiently large `S : ℕ`, taking `i = ⌈C * S^q⌉`,
the packing bound `ρ * S * i ≤ K * i^p` holds. Then necessarily `p ≥ 1 + 1 / q`. -/
theorem sustained_density_exponent_barrier {C ρ K p q : ℝ}
    (hC : 0 < C) (hρ : 0 < ρ) (hK : 0 < K) (hq : 0 < q)
    (hpack : ∀ᶠ S : ℕ in atTop,
      let i : ℝ := (Nat.ceil (C * (S : ℝ) ^ q) : ℝ)
      ρ * (S : ℝ) * i ≤ K * i ^ p) :
    1 + 1 / q ≤ p := by
  by_contra hlt
  push_neg at hlt
  -- Put `δ := 1 + 1/q - p > 0`, so `q * (1 + 1/q - p) = 1 + q - q * p > 0`.
  set β : ℝ := 1 + q - q * p
  have hβ : 0 < β := by
    have hmul := mul_lt_mul_of_pos_left hlt hq
    have hq_ne : q ≠ 0 := ne_of_gt hq
    have : q * (1 + 1 / q) = q + 1 := by field_simp [hq_ne]
    linarith
  -- For large `S`, `i ≥ C * S^q` and `i ≤ 2 * C * S^q`, giving `ρ * C * S^{1+q} ≤ K * (2C)^p * S^{qp}`.
  have hlarge : ∀ᶠ S : ℕ in atTop,
      (1 : ℝ) ≤ (S : ℝ) ∧
      (1 : ℝ) ≤ C * (S : ℝ) ^ q ∧
      K * max (C ^ p) ((2 * C) ^ p) / (ρ * C) < (S : ℝ) ^ β := by
    have h1 : ∀ᶠ S : ℕ in atTop, (1 : ℝ) ≤ (S : ℝ) :=
      eventually_atTop.mpr ⟨1, fun S hS => by exact_mod_cast hS⟩
    have h2 : ∀ᶠ S : ℕ in atTop, (1 : ℝ) ≤ C * (S : ℝ) ^ q := by
      have ht := (tendsto_rpow_atTop hq).comp tendsto_natCast_atTop_atTop
      have ht2 := Filter.Tendsto.const_mul_atTop hC ht
      exact ht2.eventually (eventually_ge_atTop 1)
    have h3 : ∀ᶠ S : ℕ in atTop, K * max (C ^ p) ((2 * C) ^ p) / (ρ * C) < (S : ℝ) ^ β := by
      have ht := (tendsto_rpow_atTop hβ).comp tendsto_natCast_atTop_atTop
      exact ht.eventually (eventually_gt_atTop _)
    exact h1.and (h2.and h3)
  obtain ⟨S, hpackS, hS1, hCS, hSβ⟩ := (hpack.and hlarge).exists
  have hS_pos : (0 : ℝ) < (S : ℝ) := lt_of_lt_of_le one_pos hS1
  set x : ℝ := C * (S : ℝ) ^ q
  set i : ℝ := (Nat.ceil x : ℝ)
  have hx_pos : 0 < x := lt_of_lt_of_le one_pos hCS
  have hi_ge : x ≤ i := Nat.le_ceil x
  have hi_le : i ≤ 2 * x := by
    have hlt1 : i < x + 1 := Nat.ceil_lt_add_one (le_of_lt hx_pos)
    linarith
  have hi_pos : 0 < i := lt_of_lt_of_le hx_pos hi_ge
  -- Bound `i ^ p ≤ max (C^p) ((2C)^p) * S^{q*p}`.
  have hip_le : i ^ p ≤ max (C ^ p) ((2 * C) ^ p) * (S : ℝ) ^ (q * p) := by
    have hpow_eq : ∀ c : ℝ, 0 < c → (c * (S : ℝ) ^ q) ^ p = c ^ p * (S : ℝ) ^ (q * p) := by
      intro c hc
      rw [Real.mul_rpow (le_of_lt hc) (le_of_lt (Real.rpow_pos_of_pos hS_pos q)),
        ← Real.rpow_mul (le_of_lt hS_pos)]
    by_cases hp0 : 0 ≤ p
    · have h1 : i ^ p ≤ (2 * x) ^ p := Real.rpow_le_rpow (le_of_lt hi_pos) hi_le hp0
      have h2x : 2 * x = (2 * C) * (S : ℝ) ^ q := by dsimp [x]; ring
      rw [h2x, hpow_eq (2 * C) (by positivity)] at h1
      exact le_trans h1 (mul_le_mul_of_nonneg_right (le_max_right _ _) (by positivity))
    · have h1 : i ^ p ≤ x ^ p := Real.rpow_le_rpow_of_nonpos hx_pos hi_ge (le_of_not_ge hp0)
      rw [show x = C * (S : ℝ) ^ q from rfl, hpow_eq C hC] at h1
      exact le_trans h1 (mul_le_mul_of_nonneg_right (le_max_left _ _) (by positivity))
  -- Combine with `ρ * S * x ≤ ρ * S * i ≤ K * i^p`.
  have hlower : ρ * C * (S : ℝ) ^ (1 + q) ≤ ρ * (S : ℝ) * i := by
    have hpow : (S : ℝ) ^ (1 + q) = (S : ℝ) * (S : ℝ) ^ q := by
      rw [Real.rpow_add hS_pos, Real.rpow_one]
    have h1 : ρ * (S : ℝ) * x ≤ ρ * (S : ℝ) * i :=
      mul_le_mul_of_nonneg_left hi_ge (by positivity)
    dsimp [x] at h1
    linarith [show ρ * C * (S : ℝ) ^ (1 + q) = ρ * (S : ℝ) * (C * (S : ℝ) ^ q) by rw [hpow]; ring]
  have hupper : ρ * C * (S : ℝ) ^ (1 + q) ≤
      K * max (C ^ p) ((2 * C) ^ p) * (S : ℝ) ^ (q * p) := by
    have hKmul := mul_le_mul_of_nonneg_left hip_le (le_of_lt hK)
    linarith
  have hSqp_pos : 0 < (S : ℝ) ^ (q * p) := Real.rpow_pos_of_pos hS_pos (q * p)
  have hsplit : (S : ℝ) ^ (1 + q) = (S : ℝ) ^ β * (S : ℝ) ^ (q * p) := by
    rw [← Real.rpow_add hS_pos]
    congr 1
    dsimp [β]
    ring
  rw [hsplit] at hupper
  have hcancel : ρ * C * (S : ℝ) ^ β ≤ K * max (C ^ p) ((2 * C) ^ p) := by
    nlinarith [hSqp_pos]
  have hcontra : (S : ℝ) ^ β ≤ K * max (C ^ p) ((2 * C) ^ p) / (ρ * C) :=
    (le_div_iff₀ (mul_pos hρ hC)).mpr (by linarith)
  linarith

/-- **Golden-Ratio Barrier for Balanced Objective `p = q`** (Theorem 3(1)):
If `ρ S i ≤ K i^p` for all large `S` at `i = ⌈C S^p⌉` (`p > 0`), then `p ≥ φ = (1 + √5) / 2`. -/
theorem balanced_golden_ratio_barrier {C ρ K p : ℝ}
    (hC : 0 < C) (hρ : 0 < ρ) (hK : 0 < K) (hp : 0 < p)
    (hpack : ∀ᶠ S : ℕ in atTop,
      let i : ℝ := (Nat.ceil (C * (S : ℝ) ^ p) : ℝ)
      ρ * (S : ℝ) * i ≤ K * i ^ p) :
    phi ≤ p :=
  (balanced_barrier_iff hp).mp (sustained_density_exponent_barrier hC hρ hK hp hpack)

/-- Exact diagonal-order pair count bound (powers-of-two core family, Theorem 3(3)):
at most `M^2` pairs `(s, k) ∈ [1, M]^2` satisfy `s + k ≤ M`. -/
theorem diagonal_pair_count_le_sq (M : ℕ) :
    ((Finset.Icc 1 M ×ˢ Finset.Icc 1 M).filter (fun p => p.1 + p.2 ≤ M)).card ≤ M ^ 2 := by
  calc ((Finset.Icc 1 M ×ˢ Finset.Icc 1 M).filter (fun p => p.1 + p.2 ≤ M)).card
      ≤ (Finset.Icc 1 M ×ˢ Finset.Icc 1 M).card := Finset.card_le_card (Finset.filter_subset _ _)
    _ = M ^ 2 := by simp [Finset.card_product, Nat.card_Icc, sq]

/-- **Theorem 3(3) (Impossibility of the Pure-Power Endpoint `p = 1 + 1 / q`)**:
On the boundary `p = 1 + 1/q`, if `D(i) ≤ K * i^{1 + 1/q}` and along `i_S = ⌈C (S + 1)^q⌉`
the expected hallucination count on the fixed core instance satisfies
`H(D(i_S)) ≥ S * (ρ * i_S - c₀ * log (2 + S + i_S))` for all large `S`, then the
hallucination fraction `H(t) / t` stays bounded below by a strictly positive constant
along `t_S = D(i_S)`, contradicting pointwise vanishing hallucination `H(t) / t → 0`. -/
theorem pure_power_endpoint_impossible {C ρ K q c₀ : ℝ}
    (hC : 0 < C) (hρ : 0 < ρ) (hK : 0 < K) (hq : 0 < q) (hc₀ : 0 ≤ c₀)
    (D : ℕ → ℕ) (H : ℕ → ℝ)
    (hD_pos : ∀ i, 1 ≤ i → 1 ≤ D i)
    (hD_ub : ∀ i, 1 ≤ i → (D i : ℝ) ≤ K * (i : ℝ) ^ (1 + 1 / q))
    (hD_top : Tendsto D atTop atTop)
    (hH_lb : ∀ᶠ S : ℕ in atTop,
      let i : ℕ := Nat.ceil (C * ((S + 1 : ℕ) : ℝ) ^ q)
      (S : ℝ) * (ρ * (i : ℝ) - c₀ * Real.log (2 + (S : ℝ) + (i : ℝ))) ≤ H (D i)) :
    ¬ Tendsto (fun t : ℕ => H t / (t : ℝ)) atTop (nhds 0) := by
  intro htend
  -- Positive asymptotic lower bound constant `δ`.
  let δ : ℝ := ρ / (8 * K * (2 * C) ^ (1 / q))
  have hδ : 0 < δ := by positivity
  -- Along `S → ∞`, `i_S = ⌈C (S+1)^q⌉ → ∞`, so `D(i_S) → ∞`, hence eventually `H(D(i_S)) / D(i_S) < δ`.
  let idx : ℕ → ℕ := fun S => Nat.ceil (C * ((S + 1 : ℕ) : ℝ) ^ q)
  have hidx_top : Tendsto idx atTop atTop := by
    rw [tendsto_atTop_atTop]
    intro b
    have ht : Tendsto (fun S : ℕ => C * ((S + 1 : ℕ) : ℝ) ^ q) atTop atTop := by
      have h1 : Tendsto (fun S : ℕ => ((S + 1 : ℕ) : ℝ)) atTop atTop :=
        tendsto_natCast_atTop_atTop.comp (tendsto_add_atTop_nat 1)
      exact Filter.Tendsto.const_mul_atTop hC ((tendsto_rpow_atTop hq).comp h1)
    obtain ⟨N, hN⟩ := (tendsto_atTop.mp ht (b : ℝ)).exists_forall_of_atTop
    refine ⟨N, fun S hS => ?_⟩
    exact_mod_cast le_trans (hN S hS) (Nat.le_ceil _)
  have hsmall : ∀ᶠ S : ℕ in atTop, H (D (idx S)) / (D (idx S) : ℝ) < δ := by
    have hcomp := htend.comp (hD_top.comp hidx_top)
    exact hcomp.eventually (Iio_mem_nhds hδ)
  -- Show `c₀ * log (2 + S + i_S) ≤ (ρ / 2) * i_S` and `S ≥ (S + 1) / 2` eventually.
  have hgood : ∀ᶠ S : ℕ in atTop,
      1 ≤ S ∧
      (1 : ℝ) ≤ C * ((S + 1 : ℕ) : ℝ) ^ q ∧
      c₀ * Real.log (2 + (S : ℝ) + (idx S : ℝ)) ≤ (ρ / 2) * (idx S : ℝ) := by
    have h1 : ∀ᶠ S : ℕ in atTop, 1 ≤ S := eventually_atTop.mpr ⟨1, fun _ h => h⟩
    have h2 : ∀ᶠ S : ℕ in atTop, (1 : ℝ) ≤ C * ((S + 1 : ℕ) : ℝ) ^ q := by
      have ht : Tendsto (fun S : ℕ => C * ((S + 1 : ℕ) : ℝ) ^ q) atTop atTop :=
        Filter.Tendsto.const_mul_atTop hC ((tendsto_rpow_atTop hq).comp
          (tendsto_natCast_atTop_atTop.comp (tendsto_add_atTop_nat 1)))
      exact ht.eventually (eventually_ge_atTop 1)
    -- Use `log (2 + S + i) ≤ (2 + S + i)^ε / ε` with `ε = min (1/2) (q / 2)`!
    -- Even simpler: `log x ≤ (x^α) / α` where `α = min (1/2) (q / (2 * (1 + q)))`.
    let α : ℝ := min (1 / 2) (q / (2 * (1 + q)))
    have hα_pos : 0 < α := by positivity
    have hα_lt_half : α ≤ 1 / 2 := min_le_left _ _
    have hα_q : α * max 1 q < q := by
      have hα2 : α ≤ q / (2 * (1 + q)) := min_le_right _ _
      have hm : max 1 q ≤ 1 + q := max_le (by linarith) (by linarith)
      have h1 : α * max 1 q ≤ (q / (2 * (1 + q))) * (1 + q) :=
        mul_le_mul hα2 hm (by positivity) (by positivity)
      have h2 : (q / (2 * (1 + q))) * (1 + q) = q / 2 := by
        have : 1 + q ≠ 0 := by linarith
        field_simp [this]
      linarith
    have h3 : ∀ᶠ S : ℕ in atTop,
        c₀ * ((2 + (1 : ℝ) + 2 * C) ^ α / α) * ((S + 1 : ℕ) : ℝ) ^ (α * max 1 q) ≤
          (ρ / 2) * C * ((S + 1 : ℕ) : ℝ) ^ q := by
      have hdiff : 0 < q - α * max 1 q := sub_pos.mpr hα_q
      have ht : Tendsto (fun S : ℕ => ((S + 1 : ℕ) : ℝ) ^ (q - α * max 1 q)) atTop atTop :=
        (tendsto_rpow_atTop hdiff).comp
          (tendsto_natCast_atTop_atTop.comp (tendsto_add_atTop_nat 1))
      filter_upwards [ht.eventually (eventually_ge_atTop
        (c₀ * ((2 + (1 : ℝ) + 2 * C) ^ α / α) / ((ρ / 2) * C)))] with S hS
      have hSp : (0 : ℝ) < ((S + 1 : ℕ) : ℝ) := by positivity
      have hsplit : ((S + 1 : ℕ) : ℝ) ^ q =
          ((S + 1 : ℕ) : ℝ) ^ (q - α * max 1 q) * ((S + 1 : ℕ) : ℝ) ^ (α * max 1 q) := by
        rw [← Real.rpow_add hSp]; congr 1; ring
      rw [hsplit]
      have hmul := (div_le_iff₀ (by positivity : (0 : ℝ) < (ρ / 2) * C)).mp hS
      have hpow_nn : 0 ≤ ((S + 1 : ℕ) : ℝ) ^ (α * max 1 q) := by positivity
      nlinarith
    filter_upwards [h1, h2, h3] with S hS1 hCS h3S
    refine ⟨hS1, hCS, ?_⟩
    set x : ℝ := C * ((S + 1 : ℕ) : ℝ) ^ q
    have hi_ge : x ≤ (idx S : ℝ) := Nat.le_ceil x
    have hi_le : (idx S : ℝ) ≤ 2 * x := by
      have := Nat.ceil_lt_add_one (show 0 ≤ x by linarith)
      linarith
    have hS1R : (1 : ℝ) ≤ ((S + 1 : ℕ) : ℝ) := by exact_mod_cast (by omega : 1 ≤ S + 1)
    have harg_ge1 : (1 : ℝ) ≤ 2 + (S : ℝ) + (idx S : ℝ) := by
      have h1 : (0 : ℝ) ≤ (S : ℝ) := by positivity
      have h2 : (0 : ℝ) ≤ (idx S : ℝ) := by positivity
      linarith
    have harg_ub : 2 + (S : ℝ) + (idx S : ℝ) ≤
        (2 + 1 + 2 * C) * ((S + 1 : ℕ) : ℝ) ^ max 1 q := by
      have hM1 : (1 : ℝ) ≤ ((S + 1 : ℕ) : ℝ) ^ max 1 q :=
        Real.one_le_rpow hS1R (le_trans zero_le_one (le_max_left 1 q))
      have hM2 : (S : ℝ) ≤ ((S + 1 : ℕ) : ℝ) ^ max 1 q := by
        calc (S : ℝ) ≤ ((S + 1 : ℕ) : ℝ) := by exact_mod_cast Nat.le_succ S
          _ = ((S + 1 : ℕ) : ℝ) ^ (1 : ℝ) := (Real.rpow_one _).symm
          _ ≤ ((S + 1 : ℕ) : ℝ) ^ max 1 q := Real.rpow_le_rpow_of_exponent_le hS1R (le_max_left 1 q)
      have hM3 : ((S + 1 : ℕ) : ℝ) ^ q ≤ ((S + 1 : ℕ) : ℝ) ^ max 1 q :=
        Real.rpow_le_rpow_of_exponent_le hS1R (le_max_right 1 q)
      dsimp [x] at hi_le
      nlinarith
    have hlog1 : Real.log (2 + (S : ℝ) + (idx S : ℝ)) ≤
        (2 + (S : ℝ) + (idx S : ℝ)) ^ α / α := log_le_rpow_div hα_pos harg_ge1
    have hlog2 : (2 + (S : ℝ) + (idx S : ℝ)) ^ α ≤
        (2 + 1 + 2 * C) ^ α * ((S + 1 : ℕ) : ℝ) ^ (α * max 1 q) := by
      have hpow := Real.rpow_le_rpow (by positivity) harg_ub (le_of_lt hα_pos)
      rw [Real.mul_rpow (by linarith) (by positivity),
        ← Real.rpow_mul (by positivity), mul_comm (max 1 q) α] at hpow
      exact hpow
    have hlog3 : c₀ * Real.log (2 + (S : ℝ) + (idx S : ℝ)) ≤
        c₀ * ((2 + 1 + 2 * C) ^ α / α) * ((S + 1 : ℕ) : ℝ) ^ (α * max 1 q) := by
      have hdiv : (2 + (S : ℝ) + (idx S : ℝ)) ^ α / α ≤
          ((2 + 1 + 2 * C) ^ α / α) * ((S + 1 : ℕ) : ℝ) ^ (α * max 1 q) := by
        rw [div_mul_eq_mul_div]
        exact div_le_div_of_nonneg_right hlog2 (le_of_lt hα_pos)
      have := mul_le_mul_of_nonneg_left (le_trans hlog1 hdiv) hc₀
      linarith
    have hfinal : (ρ / 2) * C * ((S + 1 : ℕ) : ℝ) ^ q ≤ (ρ / 2) * (idx S : ℝ) := by
      dsimp [x] at hi_ge
      nlinarith
    linarith
  obtain ⟨S, hH_S, hsmallS, hS1, hCS, hlogS⟩ := (hH_lb.and (hsmall.and hgood)).exists
  set i : ℕ := idx S
  have hi_ge1 : 1 ≤ i := by
    have : (1 : ℝ) ≤ (i : ℝ) := le_trans hCS (Nat.le_ceil _)
    exact_mod_cast this
  have hD_ge1 : 1 ≤ D i := hD_pos i hi_ge1
  have hD_posR : (0 : ℝ) < (D i : ℝ) := by exact_mod_cast (by omega : 0 < D i)
  have hi_le : (i : ℝ) ≤ 2 * C * ((S + 1 : ℕ) : ℝ) ^ q := by
    have := Nat.ceil_lt_add_one (show 0 ≤ C * ((S + 1 : ℕ) : ℝ) ^ q by linarith)
    linarith
  have hi_posR : (0 : ℝ) < (i : ℝ) := by exact_mod_cast (by omega : 0 < i)
  -- Upper bound `D(i) ≤ K * i * (2C)^{1/q} * (S + 1)`.
  have hD_bound : (D i : ℝ) ≤ K * (2 * C) ^ (1 / q) * ((S + 1 : ℕ) : ℝ) * (i : ℝ) := by
    have h1 := hD_ub i hi_ge1
    rw [Real.rpow_add hi_posR, Real.rpow_one] at h1
    have h2 : (i : ℝ) ^ (1 / q) ≤ (2 * C) ^ (1 / q) * ((S + 1 : ℕ) : ℝ) := by
      have hpow := Real.rpow_le_rpow (le_of_lt hi_posR) hi_le (by positivity : (0 : ℝ) ≤ 1 / q)
      rw [Real.mul_rpow (by positivity) (by positivity),
        ← Real.rpow_mul (by positivity),
        mul_one_div_cancel (ne_of_gt hq), Real.rpow_one] at hpow
      exact hpow
    have h3 : K * ((i : ℝ) * (i : ℝ) ^ (1 / q)) ≤
        K * ((i : ℝ) * ((2 * C) ^ (1 / q) * ((S + 1 : ℕ) : ℝ))) :=
      mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left h2 (le_of_lt hi_posR)) (le_of_lt hK)
    linarith
  -- Lower bound `H(D(i)) ≥ (ρ / 4) * (S + 1) * i`.
  have hS_half : ((S + 1 : ℕ) : ℝ) / 2 ≤ (S : ℝ) := by
    have : (1 : ℝ) ≤ (S : ℝ) := by exact_mod_cast hS1
    push_cast
    linarith
  have hH_lower : (ρ / 4) * ((S + 1 : ℕ) : ℝ) * (i : ℝ) ≤ H (D i) := by
    have h1 : (S : ℝ) * ((ρ / 2) * (i : ℝ)) ≤ H (D i) := by
      have : (ρ / 2) * (i : ℝ) ≤ ρ * (i : ℝ) - c₀ * Real.log (2 + (S : ℝ) + (i : ℝ)) := by linarith
      exact le_trans (mul_le_mul_of_nonneg_left this (by positivity)) hH_S
    have h2 : (((S + 1 : ℕ) : ℝ) / 2) * ((ρ / 2) * (i : ℝ)) ≤ (S : ℝ) * ((ρ / 2) * (i : ℝ)) :=
      mul_le_mul_of_nonneg_right hS_half (by positivity)
    linarith
  -- Hence `H(D(i)) / D(i) ≥ ρ / (4 K (2C)^{1/q}) = 2 δ > δ`, contradiction!
  have hfrac_ge : δ ≤ H (D i) / (D i : ℝ) := by
    rw [le_div_iff₀ hD_posR]
    have h1 : δ * (D i : ℝ) ≤ δ * (K * (2 * C) ^ (1 / q) * ((S + 1 : ℕ) : ℝ) * (i : ℝ)) :=
      mul_le_mul_of_nonneg_left hD_bound (le_of_lt hδ)
    have h2 : δ * (K * (2 * C) ^ (1 / q) * ((S + 1 : ℕ) : ℝ) * (i : ℝ)) =
        (ρ / 8) * ((S + 1 : ℕ) : ℝ) * (i : ℝ) := by
      dsimp [δ]
      have hK_ne : K ≠ 0 := ne_of_gt hK
      have h2C_ne : (2 * C) ^ (1 / q) ≠ 0 := ne_of_gt (Real.rpow_pos_of_pos (by positivity) _)
      field_simp [hK_ne, h2C_ne]
    have h3 : (ρ / 8) * ((S + 1 : ℕ) : ℝ) * (i : ℝ) ≤ (ρ / 4) * ((S + 1 : ℕ) : ℝ) * (i : ℝ) := by
      nlinarith [hi_posR, show (0 : ℝ) < ((S + 1 : ℕ) : ℝ) by positivity]
    linarith
  linarith

end TimelyLMHallucination
