import Mathlib

/-!
# Counterexamples to Three Coded Block Retrieval Conjectures
(Daniella Bar-Lev, arXiv:2603.17154, Conjectures `conj:hyperbola`, `conj:monotone`, `conj:limiting`)

We formalize in Lean 4 + Mathlib (complete kernel-verified proofs, standard foundations only) the exact disproofs of
all three conjectures:

1. **Exact Coupon-Collector / Subspace-Recovery Expected Wait Framework**:
   - `expectedWait`: exact rational expected number of i.i.d. uniform draws from a finite index
     set `α` until a monotone recovery predicate `Rec : Finset α → Prop` holds.
   - `expectedWait_bellman`, `expectedWait_unique`: exact Bellman recursion and uniqueness.
   - `tailProb_hasSum_expectedWait`: exact probabilistic equivalence `∑' t, P(τ > t) = E[τ]`
     connecting the rational Bellman recursion to the tail-sum expectation over i.i.d. uniform draws.
   - `expectedWait_mono`, `expectedWait_ge_one_add_frac`: monotonicity and first-step lower bound.
   - `expectedWait_repeat`: exact `m`-fold block repetition invariance for every `m ≥ 1`.
2. **Refutation of `conj:hyperbola` at `(s₁, s₂) = (2, 1)` over `𝔽₁₁` (`n = 22`)**:
   - `G22`: `11` copies of `z = e₃` plus `v_t = (1, t, t²)` for `t ∈ ZMod 11`.
   - `G22_three_ord_li`, `G22_two_ord_special_li`, `G22_full_rank`: exact Vandermonde and
     projection linear independence showing every 3 distinct types span `(ZMod 11)³`.
   - `G22_rec_F1_iff`, `G22_rec_F2_iff`: exact recovery characterizations for `F₁ = ⟨e₁, e₂⟩`
     and `F₂ = ⟨e₃⟩`.
   - `G22_E1`, `G22_E2`, `refute_conj_hyperbola`: exact values `(E₁, E₂) = (937/210, 25/14)`
     and `2/E₁ + 1/E₂ = 23618/23425 > 1`.
3. **Refutation of `conj:limiting` for every repetition factor `m ≥ 1`**:
   - `refute_conj_limiting`: the `m`-fold concatenation `G22_rep m` of length `22 * m` has full
     rank `3` and achieves the exact same `(937/210, 25/14)` with `2/E₁ + 1/E₂ = 23618/23425 > 1`.
4. **Refutation of `conj:monotone` over EVERY field `K` (including every finite field `𝔽_q`)**:
   - `G9_full_rank`, `G9_E1`, `G9_E2`: the length-9 matrix `[e₁, e₂, e₃⁷]` has rank 3 and achieves
     `(E₁, E₂) = (27/2, 9/7)` over every field `K`.
   - `refute_conj_monotone`: every rank-3 length-10 matrix `G' : Fin 10 → (Fin 3 → K)` over any
     field `K` with `E₂(G') ≤ 9/7` must satisfy `E₁(G') ≥ 15 > 27/2`.

Machine-checked vs prose-only:
- Machine-checked: All three conjecture refutations (Theorems 1 and 2 of
  `solutions/coded_block_retrieval/solution.md`), including the full Vandermonde/projection linear
  independence, exact recovery characterizations, Bellman uniqueness, tail-sum expectation
  identity, `m`-fold repetition invariance, and field-independent length 9 → 10 obstruction.
- Prose-only (in `solutions/coded_block_retrieval/proof.pdf`): The projectively distinct `(2, 2)`
  length-40 witness over `𝔽₈₃` (Theorem 3) and the 81-matrix Python cross-check suite.
-/

open BigOperators Finset

namespace CodedBlockRetrieval

/-! ### 1. Exact Expected Wait Time Under Uniform Sampling With Replacement -/

/-- Expected remaining uniform draws from `α` until a monotone recovery predicate `Rec` holds,
starting from the already-seen subset `S : Finset α`. -/
noncomputable def expectedWait {α : Type*} [Fintype α] [DecidableEq α]
    (Rec : Finset α → Prop) [DecidablePred Rec] (hUniv : Rec Finset.univ)
    (S : Finset α) : ℚ :=
  if Rec S then 0
  else
    ((Fintype.card α : ℚ) +
      ∑ a ∈ (Finset.univ \ S).attach,
        expectedWait Rec hUniv (insert a.1 S)) / ((Finset.univ \ S).card : ℚ)
termination_by (Finset.univ \ S).card
decreasing_by
  have ha : a.1 ∈ Finset.univ \ S := a.2
  have hsub : Finset.univ \ insert a.1 S = (Finset.univ \ S).erase a.1 := by
    ext x
    simp only [Finset.mem_sdiff, Finset.mem_univ, true_and, Finset.mem_insert,
      Finset.mem_erase, not_or]
  rw [hsub]
  exact Finset.card_erase_lt_of_mem ha

lemma expectedWait_of_rec {α : Type*} [Fintype α] [DecidableEq α]
    (Rec : Finset α → Prop) [DecidablePred Rec] (hUniv : Rec Finset.univ)
    {S : Finset α} (hS : Rec S) :
    expectedWait Rec hUniv S = 0 := by
  rw [expectedWait]
  simp [hS]

lemma expectedWait_unfold {α : Type*} [Fintype α] [DecidableEq α]
    (Rec : Finset α → Prop) [DecidablePred Rec] (hUniv : Rec Finset.univ)
    {S : Finset α} (hS : ¬ Rec S) :
    expectedWait Rec hUniv S =
      ((Fintype.card α : ℚ) +
        ∑ a ∈ Finset.univ \ S, expectedWait Rec hUniv (insert a S)) /
      ((Finset.univ \ S).card : ℚ) := by
  rw [expectedWait]
  simp only [hS, ↓reduceIte]
  congr 2
  exact Finset.sum_attach (Finset.univ \ S) (fun a => expectedWait Rec hUniv (insert a S))

/-- Exact Bellman equation for `expectedWait`: when `¬ Rec S`,
`W(S) = 1 + (1 / |α|) * ∑_{a : α} W(insert a S)`. -/
theorem expectedWait_bellman {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α]
    (Rec : Finset α → Prop) [DecidablePred Rec] (hUniv : Rec Finset.univ)
    {S : Finset α} (hS : ¬ Rec S) :
    expectedWait Rec hUniv S =
      1 + (1 / (Fintype.card α : ℚ)) *
        ∑ a : α, expectedWait Rec hUniv (insert a S) := by
  have hrem_pos_nat : 0 < (Finset.univ \ S).card := by
    rw [Finset.card_pos, Finset.sdiff_nonempty]
    intro hsub
    have hEq : S = Finset.univ := Finset.eq_univ_iff_forall.mpr (fun a => hsub (Finset.mem_univ a))
    exact hS (hEq ▸ hUniv)
  have hrem_ne : ((Finset.univ \ S).card : ℚ) ≠ 0 :=
    Nat.cast_ne_zero.mpr (Nat.ne_of_gt hrem_pos_nat)
  have hcard_ne : (Fintype.card α : ℚ) ≠ 0 :=
    Nat.cast_ne_zero.mpr Fintype.card_ne_zero
  have hcard_split : (S.card : ℚ) + ((Finset.univ \ S).card : ℚ) = (Fintype.card α : ℚ) := by
    have hnat : S.card + (Finset.univ \ S).card = Fintype.card α := by
      have := Finset.card_sdiff_add_card_eq_card (Finset.subset_univ S)
      rw [Finset.card_univ] at this
      omega
    exact_mod_cast hnat
  have hsum_split : ∑ a : α, expectedWait Rec hUniv (insert a S) =
      (S.card : ℚ) * expectedWait Rec hUniv S +
      ∑ a ∈ Finset.univ \ S, expectedWait Rec hUniv (insert a S) := by
    have hdisj : Disjoint S (Finset.univ \ S) := Finset.disjoint_sdiff
    have hunion : S ∪ (Finset.univ \ S) = Finset.univ :=
      Finset.union_sdiff_of_subset (Finset.subset_univ S)
    conv_lhs => rw [← hunion]
    rw [Finset.sum_union hdisj]
    congr 1
    have hcongr : ∑ a ∈ S, expectedWait Rec hUniv (insert a S) =
        ∑ _a ∈ S, expectedWait Rec hUniv S :=
      Finset.sum_congr rfl fun a ha => by rw [Finset.insert_eq_of_mem ha]
    rw [hcongr, Finset.sum_const, nsmul_eq_mul]
  have hunf := expectedWait_unfold Rec hUniv hS
  have hmul : expectedWait Rec hUniv S * ((Finset.univ \ S).card : ℚ) =
      (Fintype.card α : ℚ) + ∑ a ∈ Finset.univ \ S, expectedWait Rec hUniv (insert a S) := by
    rw [hunf, div_mul_cancel₀ _ hrem_ne]
  rw [hsum_split]
  have hkey : expectedWait Rec hUniv S * (Fintype.card α : ℚ) =
      (Fintype.card α : ℚ) +
        ((S.card : ℚ) * expectedWait Rec hUniv S +
          ∑ a ∈ Finset.univ \ S, expectedWait Rec hUniv (insert a S)) := by
    calc expectedWait Rec hUniv S * (Fintype.card α : ℚ)
        = expectedWait Rec hUniv S * ((S.card : ℚ) + ((Finset.univ \ S).card : ℚ)) := by
          rw [hcard_split]
      _ = (S.card : ℚ) * expectedWait Rec hUniv S +
          expectedWait Rec hUniv S * ((Finset.univ \ S).card : ℚ) := by ring
      _ = (Fintype.card α : ℚ) +
          ((S.card : ℚ) * expectedWait Rec hUniv S +
            ∑ a ∈ Finset.univ \ S, expectedWait Rec hUniv (insert a S)) := by linarith [hmul]
  field_simp [hcard_ne]
  linarith

/-- Uniqueness of solutions to the Bellman equation on `Finset α`. -/
theorem expectedWait_unique {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α]
    (Rec : Finset α → Prop) [DecidablePred Rec] (hUniv : Rec Finset.univ)
    (W : Finset α → ℚ)
    (hZero : ∀ S, Rec S → W S = 0)
    (hStep : ∀ S, ¬ Rec S →
      W S = 1 + (1 / (Fintype.card α : ℚ)) * ∑ a : α, W (insert a S)) :
    ∀ S, W S = expectedWait Rec hUniv S := by
  intro S
  induction hN : (Finset.univ \ S).card using Nat.strong_induction_on generalizing S with
  | _ n ih =>
    by_cases hRec : Rec S
    · rw [hZero S hRec, expectedWait_of_rec Rec hUniv hRec]
    · have hrem_pos_nat : 0 < (Finset.univ \ S).card := by
        rw [Finset.card_pos, Finset.sdiff_nonempty]
        intro hsub
        exact hRec ((Finset.eq_univ_iff_forall.mpr fun a => hsub (Finset.mem_univ a)) ▸ hUniv)
      have hrem_ne : ((Finset.univ \ S).card : ℚ) ≠ 0 :=
        Nat.cast_ne_zero.mpr (Nat.ne_of_gt hrem_pos_nat)
      have hcard_ne : (Fintype.card α : ℚ) ≠ 0 :=
        Nat.cast_ne_zero.mpr Fintype.card_ne_zero
      have hcard_split : (S.card : ℚ) + ((Finset.univ \ S).card : ℚ) = (Fintype.card α : ℚ) := by
        have hnat : S.card + (Finset.univ \ S).card = Fintype.card α := by
          have := Finset.card_sdiff_add_card_eq_card (Finset.subset_univ S)
          rw [Finset.card_univ] at this
          omega
        exact_mod_cast hnat
      have hEq_sdiff : ∀ a ∈ Finset.univ \ S,
          W (insert a S) = expectedWait Rec hUniv (insert a S) := by
        intro a ha
        have hlt : (Finset.univ \ insert a S).card < n := by
          rw [← hN]
          have hsub : Finset.univ \ insert a S = (Finset.univ \ S).erase a := by
            ext x
            simp only [Finset.mem_sdiff, Finset.mem_univ, true_and, Finset.mem_insert,
              Finset.mem_erase, not_or]
          rw [hsub]
          exact Finset.card_erase_lt_of_mem ha
        exact ih _ hlt (insert a S) rfl
      have hW_step := hStep S hRec
      have hE_step := expectedWait_bellman Rec hUniv hRec
      have hsum_diff : ∀ f : Finset α → ℚ,
          ∑ a : α, f (insert a S) =
          (S.card : ℚ) * f S + ∑ a ∈ Finset.univ \ S, f (insert a S) := by
        intro f
        have hdisj : Disjoint S (Finset.univ \ S) := Finset.disjoint_sdiff
        have hunion : S ∪ (Finset.univ \ S) = Finset.univ :=
          Finset.union_sdiff_of_subset (Finset.subset_univ S)
        conv_lhs => rw [← hunion]
        rw [Finset.sum_union hdisj]
        congr 1
        have : ∑ a ∈ S, f (insert a S) = ∑ _a ∈ S, f S :=
          Finset.sum_congr rfl fun a ha => by rw [Finset.insert_eq_of_mem ha]
        rw [this, Finset.sum_const, nsmul_eq_mul]
      rw [hsum_diff W] at hW_step
      rw [hsum_diff (expectedWait Rec hUniv)] at hE_step
      have hsum_eq : ∑ a ∈ Finset.univ \ S, W (insert a S) =
          ∑ a ∈ Finset.univ \ S, expectedWait Rec hUniv (insert a S) :=
        Finset.sum_congr rfl hEq_sdiff
      rw [hsum_eq] at hW_step
      have hdiff : (W S - expectedWait Rec hUniv S) * ((Finset.univ \ S).card : ℚ) = 0 := by
        have h1 : (W S - expectedWait Rec hUniv S) * (Fintype.card α : ℚ) =
            (S.card : ℚ) * (W S - expectedWait Rec hUniv S) := by
          have hW' : W S * (Fintype.card α : ℚ) = (Fintype.card α : ℚ) +
              ((S.card : ℚ) * W S + ∑ a ∈ Finset.univ \ S, expectedWait Rec hUniv (insert a S)) := by
            field_simp [hcard_ne] at hW_step
            linarith
          have hE' : expectedWait Rec hUniv S * (Fintype.card α : ℚ) = (Fintype.card α : ℚ) +
              ((S.card : ℚ) * expectedWait Rec hUniv S +
                ∑ a ∈ Finset.univ \ S, expectedWait Rec hUniv (insert a S)) := by
            field_simp [hcard_ne] at hE_step
            linarith
          linarith
        have h2 : (W S - expectedWait Rec hUniv S) * ((S.card : ℚ) + ((Finset.univ \ S).card : ℚ)) =
            (S.card : ℚ) * (W S - expectedWait Rec hUniv S) := by
          rw [hcard_split]
          exact h1
        linarith
      exact sub_eq_zero.mp ((mul_eq_zero.mp hdiff).resolve_right hrem_ne)

/-- Nonnegativity of `expectedWait`. -/
lemma expectedWait_nonneg {α : Type*} [Fintype α] [DecidableEq α]
    (Rec : Finset α → Prop) [DecidablePred Rec] (hUniv : Rec Finset.univ) :
    ∀ S, 0 ≤ expectedWait Rec hUniv S := by
  intro S
  induction hN : (Finset.univ \ S).card using Nat.strong_induction_on generalizing S with
  | _ n ih =>
    by_cases hRec : Rec S
    · rw [expectedWait_of_rec Rec hUniv hRec]
    · rw [expectedWait_unfold Rec hUniv hRec]
      apply div_nonneg
      · apply add_nonneg (Nat.cast_nonneg _)
        refine Finset.sum_nonneg fun a ha => ?_
        have hlt : (Finset.univ \ insert a S).card < n := by
          rw [← hN]
          have hsub : Finset.univ \ insert a S = (Finset.univ \ S).erase a := by
            ext x
            simp only [Finset.mem_sdiff, Finset.mem_univ, true_and, Finset.mem_insert,
              Finset.mem_erase, not_or]
          rw [hsub]
          exact Finset.card_erase_lt_of_mem ha
        exact ih _ hlt (insert a S) rfl
      · exact Nat.cast_nonneg _

/-- Any unrecovered state has expected remaining wait at least `1`. -/
lemma expectedWait_ge_one {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α]
    (Rec : Finset α → Prop) [DecidablePred Rec] (hUniv : Rec Finset.univ)
    {S : Finset α} (hS : ¬ Rec S) :
    1 ≤ expectedWait Rec hUniv S := by
  rw [expectedWait_bellman Rec hUniv hS]
  have hsum : 0 ≤ ∑ a : α, expectedWait Rec hUniv (insert a S) :=
    Finset.sum_nonneg fun a _ => expectedWait_nonneg Rec hUniv _
  have hpos : (0 : ℚ) ≤ 1 / (Fintype.card α : ℚ) := by positivity
  linarith [mul_nonneg hpos hsum]

/-- Monotonicity: an easier recovery condition (`Rec₁ → Rec₂`) yields a smaller expected wait. -/
theorem expectedWait_mono {α : Type*} [Fintype α] [DecidableEq α]
    (Rec₁ Rec₂ : Finset α → Prop) [DecidablePred Rec₁] [DecidablePred Rec₂]
    (hUniv₁ : Rec₁ Finset.univ) (hUniv₂ : Rec₂ Finset.univ)
    (himp : ∀ S, Rec₁ S → Rec₂ S) :
    ∀ S, expectedWait Rec₂ hUniv₂ S ≤ expectedWait Rec₁ hUniv₁ S := by
  intro S
  induction hN : (Finset.univ \ S).card using Nat.strong_induction_on generalizing S with
  | _ n ih =>
    by_cases h₂ : Rec₂ S
    · rw [expectedWait_of_rec Rec₂ hUniv₂ h₂]
      exact expectedWait_nonneg Rec₁ hUniv₁ S
    · have h₁ : ¬ Rec₁ S := fun h => h₂ (himp S h)
      rw [expectedWait_unfold Rec₂ hUniv₂ h₂, expectedWait_unfold Rec₁ hUniv₁ h₁]
      apply div_le_div_of_nonneg_right _ (Nat.cast_nonneg _)
      apply add_le_add_left
      refine Finset.sum_le_sum fun a ha => ?_
      have hlt : (Finset.univ \ insert a S).card < n := by
        rw [← hN]
        have hsub : Finset.univ \ insert a S = (Finset.univ \ S).erase a := by
          ext x
          simp only [Finset.mem_sdiff, Finset.mem_univ, true_and, Finset.mem_insert,
            Finset.mem_erase, not_or]
        rw [hsub]
        exact Finset.card_erase_lt_of_mem ha
      exact ih _ hlt (insert a S) rfl

/-- Exact tail probability `P(τ > t | S)` for i.i.d. uniform draws from `α`:
the fraction of sequences `w : Fin t → α` that fail to recover `Rec`. -/
noncomputable def tailProb {α : Type*} [Fintype α] [DecidableEq α]
    (Rec : Finset α → Prop) [DecidablePred Rec] (S : Finset α) : ℕ → ℝ
  | 0 => if Rec S then 0 else 1
  | t + 1 =>
    if Rec S then 0
    else (1 / (Fintype.card α : ℝ)) * ∑ a : α, tailProb Rec (insert a S) t

/-- Exact probabilistic identity: `E[τ] = ∑_{t=0}^∞ P(τ > t) = expectedWait`. -/
theorem tailProb_hasSum_expectedWait {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α]
    (Rec : Finset α → Prop) [DecidablePred Rec] (hUniv : Rec Finset.univ) :
    ∀ S, HasSum (tailProb Rec S) (expectedWait Rec hUniv S : ℝ) := by
  intro S
  induction hN : (Finset.univ \ S).card using Nat.strong_induction_on generalizing S with
  | _ n ih =>
    by_cases hRec : Rec S
    · rw [expectedWait_of_rec Rec hUniv hRec, Rat.cast_zero]
      have hzero : tailProb Rec S = fun _ => 0 := by
        funext t
        cases t <;> simp [tailProb, hRec]
      rw [hzero]
      exact hasSum_zero
    · have hrem_pos_nat : 0 < (Finset.univ \ S).card := by
        rw [Finset.card_pos, Finset.sdiff_nonempty]
        intro hsub
        exact hRec ((Finset.eq_univ_iff_forall.mpr fun a => hsub (Finset.mem_univ a)) ▸ hUniv)
      have hcard_pos : (0 : ℝ) < (Fintype.card α : ℝ) := Nat.cast_pos.mpr Fintype.card_pos
      have hcard_split : (S.card : ℝ) + ((Finset.univ \ S).card : ℝ) = (Fintype.card α : ℝ) := by
        have hnat : S.card + (Finset.univ \ S).card = Fintype.card α := by
          have := Finset.card_sdiff_add_card_eq_card (Finset.subset_univ S)
          rw [Finset.card_univ] at this
          omega
        exact_mod_cast hnat
      have hp_lt_one : (S.card : ℝ) / (Fintype.card α : ℝ) < 1 := by
        rw [div_lt_one hcard_pos]
        have : (0 : ℝ) < ((Finset.univ \ S).card : ℝ) := Nat.cast_pos.mpr hrem_pos_nat
        linarith
      have hp_nonneg : 0 ≤ (S.card : ℝ) / (Fintype.card α : ℝ) := by positivity
      -- For every `a ∈ univ \ S`, IH gives `HasSum (tailProb Rec (insert a S)) (W (insert a S))`
      have hIH : ∀ a ∈ Finset.univ \ S,
          HasSum (tailProb Rec (insert a S)) (expectedWait Rec hUniv (insert a S) : ℝ) := by
        intro a ha
        have hlt : (Finset.univ \ insert a S).card < n := by
          rw [← hN]
          have hsub : Finset.univ \ insert a S = (Finset.univ \ S).erase a := by
            ext x
            simp only [Finset.mem_sdiff, Finset.mem_univ, true_and, Finset.mem_insert,
              Finset.mem_erase, not_or]
          rw [hsub]
          exact Finset.card_erase_lt_of_mem ha
        exact ih _ hlt (insert a S) rfl
      -- Define `g t := (1 / |α|) * ∑_{a ∉ S} tailProb Rec (insert a S) t`
      let p : ℝ := (S.card : ℝ) / (Fintype.card α : ℝ)
      let g : ℕ → ℝ := fun t =>
        (1 / (Fintype.card α : ℝ)) * ∑ a ∈ Finset.univ \ S, tailProb Rec (insert a S) t
      have hg_hasSum : HasSum g ((1 / (Fintype.card α : ℝ)) *
          ∑ a ∈ Finset.univ \ S, (expectedWait Rec hUniv (insert a S) : ℝ)) := by
        exact (hasSum_sum hIH).mul_left (1 / (Fintype.card α : ℝ))
      -- Show `tailProb Rec S (t + 1) = p * tailProb Rec S t + g t`
      have hrec : ∀ t, tailProb Rec S (t + 1) = p * tailProb Rec S t + g t := by
        intro t
        simp only [tailProb, hRec, ↓reduceIte]
        have hdisj : Disjoint S (Finset.univ \ S) := Finset.disjoint_sdiff
        have hunion : S ∪ (Finset.univ \ S) = Finset.univ :=
          Finset.union_sdiff_of_subset (Finset.subset_univ S)
        conv_lhs => rw [← hunion]
        rw [Finset.sum_union hdisj]
        have hSsum : ∑ a ∈ S, tailProb Rec (insert a S) t = (S.card : ℝ) * tailProb Rec S t := by
          have : ∑ a ∈ S, tailProb Rec (insert a S) t = ∑ _a ∈ S, tailProb Rec S t :=
            Finset.sum_congr rfl fun a ha => by rw [Finset.insert_eq_of_mem ha]
          rw [this, Finset.sum_const, nsmul_eq_mul]
        rw [hSsum, mul_add]
        simp only [p, g]
        ring
      -- Prove summability via partial sums: `(1 - p) * ∑_{t < N} f t + f N = 1 + ∑_{t < N} g t`
      have hnonneg : ∀ t, 0 ≤ tailProb Rec S t := by
        intro t
        have hgen : ∀ (m : ℕ) (T : Finset α), 0 ≤ tailProb Rec T m := by
          intro m
          induction m with
          | zero => intro T; dsimp [tailProb]; split_ifs <;> norm_num
          | succ m ihm =>
            intro T
            dsimp [tailProb]
            split_ifs
            · exact le_rfl
            · exact mul_nonneg (by positivity) (Finset.sum_nonneg fun a _ => ihm _)
        exact hgen t S
      have hpartial : ∀ N : ℕ,
          (1 - p) * ∑ t ∈ Finset.range N, tailProb Rec S t + tailProb Rec S N =
          1 + ∑ t ∈ Finset.range N, g t := by
        intro N
        induction N with
        | zero => simp [tailProb, hRec]
        | succ N ihN =>
          rw [Finset.sum_range_succ, Finset.sum_range_succ, hrec N]
          linarith
      have h1mp_pos : 0 < 1 - p := by linarith
      have hg_nonneg : ∀ t, 0 ≤ g t := by
        intro t
        refine mul_nonneg (by positivity) (Finset.sum_nonneg fun a _ => ?_)
        have hgen : ∀ (m : ℕ) (T : Finset α), 0 ≤ tailProb Rec T m := by
          intro m
          induction m with
          | zero => intro T; dsimp [tailProb]; split_ifs <;> norm_num
          | succ m ihm =>
            intro T; dsimp [tailProb]; split_ifs
            · exact le_rfl
            · exact mul_nonneg (by positivity) (Finset.sum_nonneg fun a _ => ihm _)
        exact hgen t _
      have hsum_bdd : ∀ N : ℕ,
          ∑ t ∈ Finset.range N, tailProb Rec S t ≤
            (1 + ((1 / (Fintype.card α : ℝ)) *
              ∑ a ∈ Finset.univ \ S, (expectedWait Rec hUniv (insert a S) : ℝ))) / (1 - p) := by
        intro N
        have hpN := hpartial N
        have hg_le : ∑ t ∈ Finset.range N, g t ≤
            (1 / (Fintype.card α : ℝ)) *
              ∑ a ∈ Finset.univ \ S, (expectedWait Rec hUniv (insert a S) : ℝ) :=
          (hg_hasSum.summable.sum_le_tsum (Finset.range N) (fun t _ => hg_nonneg t)).trans_eq
            hg_hasSum.tsum_eq
        rw [le_div_iff₀' h1mp_pos]
        linarith [hnonneg N]
      have hsummable : Summable (tailProb Rec S) :=
        summable_of_sum_range_le hnonneg hsum_bdd
      have htsum_eq : ∑' t, tailProb Rec S t = (expectedWait Rec hUniv S : ℝ) := by
        have hshift : HasSum (fun t => tailProb Rec S (t + 1)) (∑' t, tailProb Rec S t - 1) := by
          have h0 : tailProb Rec S 0 = 1 := by simp [tailProb, hRec]
          have h1 := (hasSum_nat_add_iff' 1).mpr hsummable.hasSum
          simpa [Finset.sum_range_one, h0] using h1
        have hrhs : HasSum (fun t => p * tailProb Rec S t + g t)
            (p * (∑' t, tailProb Rec S t) + (1 / (Fintype.card α : ℝ)) *
              ∑ a ∈ Finset.univ \ S, (expectedWait Rec hUniv (insert a S) : ℝ)) :=
          (hsummable.hasSum.mul_left p).add hg_hasSum
        have heq_fn : (fun t => tailProb Rec S (t + 1)) = fun t => p * tailProb Rec S t + g t :=
          funext hrec
        rw [heq_fn] at hshift
        have hval := hshift.unique hrhs
        have h1mp_eq : 1 - p = ((Finset.univ \ S).card : ℝ) / (Fintype.card α : ℝ) := by
          simp only [p]
          have hne : (Fintype.card α : ℝ) ≠ 0 := ne_of_gt hcard_pos
          field_simp [hne]
          linarith [hcard_split]
        have hunf : (expectedWait Rec hUniv S : ℝ) =
            ((Fintype.card α : ℝ) +
              ∑ a ∈ Finset.univ \ S, (expectedWait Rec hUniv (insert a S) : ℝ)) /
            ((Finset.univ \ S).card : ℝ) := by
          rw [expectedWait_unfold Rec hUniv hRec]
          push_cast
          rfl
        rw [hunf]
        have hrem_ne : ((Finset.univ \ S).card : ℝ) ≠ 0 :=
          Nat.cast_ne_zero.mpr (Nat.ne_of_gt hrem_pos_nat)
        have hcard_ne : (Fintype.card α : ℝ) ≠ 0 := ne_of_gt hcard_pos
        have hlin : (1 - p) * ∑' t, tailProb Rec S t =
            1 + (1 / (Fintype.card α : ℝ)) *
              ∑ a ∈ Finset.univ \ S, (expectedWait Rec hUniv (insert a S) : ℝ) := by
          linarith
        rw [h1mp_eq] at hlin
        field_simp [hrem_ne, hcard_ne] at hlin ⊢
        linarith
      rw [← htsum_eq]
      exact hsummable.hasSum

/-- **Exact `m`-fold block repetition invariance** (Proposition 10 of arXiv:2603.17154):
repeating every column `m` times (`m ≥ 1`) preserves the exact expected retrieval time. -/
theorem expectedWait_repeat {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α]
    (m : ℕ) (hm : 1 ≤ m)
    (Rec : Finset α → Prop) [DecidablePred Rec] (hUniv : Rec Finset.univ) :
    haveI : Nonempty (Fin m) := ⟨⟨0, hm⟩⟩
    have hUnivRep : Rec ((Finset.univ : Finset (Fin m × α)).image Prod.snd) := by
      have : (Finset.univ : Finset (Fin m × α)).image Prod.snd = Finset.univ := by
        ext a
        simp only [Finset.mem_image, Finset.mem_univ, true_and, iff_true]
        exact ⟨(⟨0, hm⟩, a), rfl⟩
      rw [this]
      exact hUniv
    expectedWait (fun S : Finset (Fin m × α) => Rec (S.image Prod.snd)) hUnivRep ∅ =
      expectedWait Rec hUniv ∅ := by
  haveI : Nonempty (Fin m) := ⟨⟨0, hm⟩⟩
  have hUnivRep : Rec ((Finset.univ : Finset (Fin m × α)).image Prod.snd) := by
    have : (Finset.univ : Finset (Fin m × α)).image Prod.snd = Finset.univ := by
      ext a
      simp only [Finset.mem_image, Finset.mem_univ, true_and, iff_true]
      exact ⟨(⟨0, hm⟩, a), rfl⟩
    rw [this]
    exact hUniv
  have hEq := expectedWait_unique
    (fun S : Finset (Fin m × α) => Rec (S.image Prod.snd)) hUnivRep
    (fun S => expectedWait Rec hUniv (S.image Prod.snd))
    (fun S hS => expectedWait_of_rec Rec hUniv hS)
    (fun S hS => by
      dsimp only
      rw [expectedWait_bellman Rec hUniv hS]
      rw [Fintype.card_prod, Fintype.card_fin]
      rw [Fintype.sum_prod_type]
      have hinner : ∀ r : Fin m,
          ∑ a : α, expectedWait Rec hUniv ((insert (r, a) S).image Prod.snd) =
          ∑ a : α, expectedWait Rec hUniv (insert a (S.image Prod.snd)) := by
        intro r
        refine Finset.sum_congr rfl fun a _ => ?_
        rw [Finset.image_insert]
      simp_rw [hinner]
      rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      have hm_ne : (m : ℚ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
      have ha_ne : (Fintype.card α : ℚ) ≠ 0 := Nat.cast_ne_zero.mpr Fintype.card_ne_zero
      push_cast
      field_simp [hm_ne, ha_ne])
    ∅
  simpa using hEq.symm

/-! ### 2. Subspace Recovery Model for Coded Block Retrieval -/

/-- Standard basis vector `e_i` in `Fin k → K`. -/
def stdBasis {K : Type*} [Zero K] [One K] {k : ℕ} (i : Fin k) : Fin k → K :=
  Pi.single i 1

/-- Span of the columns of `G` indexed by `S`. -/
def spanCols {K : Type*} [Field K] {n k : ℕ} (G : Fin n → (Fin k → K))
    (S : Finset (Fin n)) : Submodule K (Fin k → K) :=
  Submodule.span K (G '' (S : Set (Fin n)))

/-- File subspaces `F₁ = ⟨e₀, e₁⟩` and `F₂ = ⟨e₂⟩` for partition `(s₁, s₂) = (2, 1)`. -/
def fileF1 (K : Type*) [Field K] : Submodule K (Fin 3 → K) :=
  Submodule.span K {stdBasis (K := K) (0 : Fin 3), stdBasis (K := K) (1 : Fin 3)}

def fileF2 (K : Type*) [Field K] : Submodule K (Fin 3 → K) :=
  Submodule.span K {stdBasis (K := K) (2 : Fin 3)}

lemma mem_fileF1_iff {K : Type*} [Field K] (v : Fin 3 → K) :
    v ∈ fileF1 K ↔ v 2 = 0 := by
  constructor
  · intro hv
    refine Submodule.span_induction ?_ ?_ ?_ ?_ hv
    · rintro x (rfl | rfl) <;> simp [stdBasis]
    · simp
    · intro x y _ _ hx hy; simp [hx, hy]
    · intro c x _ hx; simp [hx]
  · intro hv2
    have hEq : v = v 0 • stdBasis (0 : Fin 3) + v 1 • stdBasis (1 : Fin 3) := by
      ext i
      fin_cases i <;> simp [stdBasis, hv2]
    rw [hEq]
    exact Submodule.add_mem _
      (Submodule.smul_mem _ _ (Submodule.subset_span (by simp)))
      (Submodule.smul_mem _ _ (Submodule.subset_span (by simp)))

lemma fileF1_le_iff {K : Type*} [Field K] (V : Submodule K (Fin 3 → K)) :
    fileF1 K ≤ V ↔ stdBasis (K := K) (0 : Fin 3) ∈ V ∧ stdBasis (K := K) (1 : Fin 3) ∈ V := by
  rw [fileF1, Submodule.span_le]
  simp [Set.insert_subset_iff]

lemma fileF2_le_iff {K : Type*} [Field K] (V : Submodule K (Fin 3 → K)) :
    fileF2 K ≤ V ↔ stdBasis (K := K) (2 : Fin 3) ∈ V := by
  rw [fileF2, Submodule.span_le]
  simp

/-! ### 3. Refutation of `conj:hyperbola` and `conj:limiting` at `(2, 1)` over `𝔽₁₁` -/

instance : Fact (Nat.Prime 11) := ⟨by decide⟩

/-- Ordinary column `v_t = (1, t, t²)` over `𝔽₁₁`. -/
def ordCol (t : ZMod 11) : Fin 3 → ZMod 11 := ![1, t, t ^ 2]

/-- Special column `z = e₃ = (0, 0, 1)` over `𝔽₁₁`. -/
def specCol : Fin 3 → ZMod 11 := stdBasis (2 : Fin 3)

/-- The length-22 generator matrix `G22 ∈ 𝔽₁₁^{3 × 22}`:
indices `0..10` are `11` copies of `z = e₃`, and indices `11..21` are `v_0, ..., v_10`. -/
def G22 (i : Fin 22) : Fin 3 → ZMod 11 :=
  if i.1 < 11 then specCol else ordCol ((i.1 - 11 : ℕ) : ZMod 11)

/-- Any three distinct ordinary columns `v_{t₁}, v_{t₂}, v_{t₃}` span `𝔽₁₁³` (Vandermonde). -/
lemma span_three_ord {t₁ t₂ t₃ : ZMod 11} (h12 : t₁ ≠ t₂) (h13 : t₁ ≠ t₃) (h23 : t₂ ≠ t₃) :
    Submodule.span (ZMod 11) {ordCol t₁, ordCol t₂, ordCol t₃} = ⊤ := by
  rw [eq_top_iff]
  intro w _
  let D : ZMod 11 := (t₂ - t₁) * (t₃ - t₁) * (t₃ - t₂)
  have hD : D ≠ 0 := by
    simp only [D]
    exact mul_ne_zero (mul_ne_zero (sub_ne_zero.mpr h12.symm) (sub_ne_zero.mpr h13.symm))
      (sub_ne_zero.mpr h23.symm)
  -- Explicit Lagrange interpolation solving `c₁ • ordCol t₁ + c₂ • ordCol t₂ + c₃ • ordCol t₃ = w`:
  let c₁ : ZMod 11 := (w 0 * (t₂ * t₃) - w 1 * (t₂ + t₃) + w 2) / ((t₁ - t₂) * (t₁ - t₃))
  let c₂ : ZMod 11 := (w 0 * (t₁ * t₃) - w 1 * (t₁ + t₃) + w 2) / ((t₂ - t₁) * (t₂ - t₃))
  let c₃ : ZMod 11 := (w 0 * (t₁ * t₂) - w 1 * (t₁ + t₂) + w 2) / ((t₃ - t₁) * (t₃ - t₂))
  have h12' : t₁ - t₂ ≠ 0 := sub_ne_zero.mpr h12
  have h13' : t₁ - t₃ ≠ 0 := sub_ne_zero.mpr h13
  have h21' : t₂ - t₁ ≠ 0 := sub_ne_zero.mpr h12.symm
  have h23' : t₂ - t₃ ≠ 0 := sub_ne_zero.mpr h23
  have h31' : t₃ - t₁ ≠ 0 := sub_ne_zero.mpr h13.symm
  have h32' : t₃ - t₂ ≠ 0 := sub_ne_zero.mpr h23.symm
  have hEq : w = c₁ • ordCol t₁ + c₂ • ordCol t₂ + c₃ • ordCol t₃ := by
    ext i
    fin_cases i
    · change w 0 = c₁ * 1 + c₂ * 1 + c₃ * 1
      dsimp only [c₁, c₂, c₃]
      field_simp [h12', h13', h21', h23', h31', h32']
      ring
    · change w 1 = c₁ * t₁ + c₂ * t₂ + c₃ * t₃
      dsimp only [c₁, c₂, c₃]
      field_simp [h12', h13', h21', h23', h31', h32']
      ring
    · change w 2 = c₁ * t₁ ^ 2 + c₂ * t₂ ^ 2 + c₃ * t₃ ^ 2
      dsimp only [c₁, c₂, c₃]
      field_simp [h12', h13', h21', h23', h31', h32']
      ring
  rw [hEq]
  exact Submodule.add_mem _
    (Submodule.add_mem _
      (Submodule.smul_mem _ _ (Submodule.subset_span (by simp)))
      (Submodule.smul_mem _ _ (Submodule.subset_span (by simp))))
    (Submodule.smul_mem _ _ (Submodule.subset_span (by simp)))

/-- Any two distinct ordinary columns `v_{t₁}, v_{t₂}` together with `z = e₃` span `𝔽₁₁³`. -/
lemma span_two_ord_spec {t₁ t₂ : ZMod 11} (h12 : t₁ ≠ t₂) :
    Submodule.span (ZMod 11) {ordCol t₁, ordCol t₂, specCol} = ⊤ := by
  rw [eq_top_iff]
  intro w _
  have h21' : t₂ - t₁ ≠ 0 := sub_ne_zero.mpr h12.symm
  let c₁ : ZMod 11 := (t₂ * w 0 - w 1) / (t₂ - t₁)
  let c₂ : ZMod 11 := (w 1 - t₁ * w 0) / (t₂ - t₁)
  let c₃ : ZMod 11 := w 2 - c₁ * t₁ ^ 2 - c₂ * t₂ ^ 2
  have hEq : w = c₁ • ordCol t₁ + c₂ • ordCol t₂ + c₃ • specCol := by
    ext i
    fin_cases i
    · change w 0 = c₁ * 1 + c₂ * 1 + c₃ * 0
      dsimp only [c₁, c₂, c₃]
      field_simp [h21']
      ring
    · change w 1 = c₁ * t₁ + c₂ * t₂ + c₃ * 0
      dsimp only [c₁, c₂, c₃]
      field_simp [h21']
      ring
    · change w 2 = c₁ * t₁ ^ 2 + c₂ * t₂ ^ 2 + c₃ * 1
      dsimp only [c₁, c₂, c₃]
      field_simp [h21']
      ring
  rw [hEq]
  exact Submodule.add_mem _
    (Submodule.add_mem _
      (Submodule.smul_mem _ _ (Submodule.subset_span (by simp)))
      (Submodule.smul_mem _ _ (Submodule.subset_span (by simp))))
    (Submodule.smul_mem _ _ (Submodule.subset_span (by simp)))

/-- `G22` has full rank `3`. -/
theorem G22_full_rank : spanCols G22 Finset.univ = ⊤ := by
  rw [eq_top_iff, ← span_two_ord_spec (t₁ := 0) (t₂ := 1) (by decide)]
  apply Submodule.span_mono
  intro x hx
  simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at hx
  rcases hx with rfl | rfl | rfl
  · exact ⟨⟨11, by decide⟩, Finset.mem_univ _, by simp [G22]⟩
  · exact ⟨⟨12, by decide⟩, Finset.mem_univ _, by simp [G22]⟩
  · exact ⟨⟨0, by decide⟩, Finset.mem_univ _, by simp [G22]⟩

/-- Number of ordinary indices (`i ≥ 11`) in `S : Finset (Fin 22)`. -/
def ordCount (S : Finset (Fin 22)) : ℕ :=
  (S.filter (fun i : Fin 22 => 11 ≤ i.1)).card

/-- Whether `S : Finset (Fin 22)` contains at least one special index (`i < 11`). -/
def hasSpecial (S : Finset (Fin 22)) : Bool :=
  decide (∃ i ∈ S, i.1 < 11)

lemma hasSpecial_eq_true_iff (S : Finset (Fin 22)) :
    hasSpecial S = true ↔ ∃ i ∈ S, i.1 < 11 := by
  simp [hasSpecial]

lemma hasSpecial_eq_false_iff (S : Finset (Fin 22)) :
    hasSpecial S = false ↔ ∀ i ∈ S, 11 ≤ i.1 := by
  simp [hasSpecial]

lemma ordCol_val_inj {i j : Fin 22} (hi : 11 ≤ i.1) (hj : 11 ≤ j.1)
    (h : ((i.1 - 11 : ℕ) : ZMod 11) = ((j.1 - 11 : ℕ) : ZMod 11)) : i = j := by
  have hi' : i.1 - 11 < 11 := by omega
  have hj' : j.1 - 11 < 11 := by omega
  have hval : (i.1 - 11) % 11 = (j.1 - 11) % 11 := by
    have := congrArg ZMod.val h
    simpa [ZMod.val_natCast] using this
  rw [Nat.mod_eq_of_lt hi', Nat.mod_eq_of_lt hj'] at hval
  ext
  omega

/-- No span of at most two ordinary columns contains `e₃` (`F₂`) or both `e₁, e₂` (`F₁`). -/
lemma span_two_ord_not_F1_F2 (t₁ t₂ : ZMod 11) :
    ¬ (fileF1 (ZMod 11) ≤ Submodule.span (ZMod 11) {ordCol t₁, ordCol t₂}) ∧
    ¬ (fileF2 (ZMod 11) ≤ Submodule.span (ZMod 11) {ordCol t₁, ordCol t₂}) := by
  have hmem : ∀ v ∈ Submodule.span (ZMod 11) {ordCol t₁, ordCol t₂},
      ∃ a b : ZMod 11, v = a • ordCol t₁ + b • ordCol t₂ := by
    intro v hv
    rcases Submodule.mem_span_pair.mp hv with ⟨a, b, hab⟩
    exact ⟨a, b, hab.symm⟩
  constructor
  · intro hle
    rw [fileF1_le_iff] at hle
    rcases hmem _ hle.1 with ⟨a₁, b₁, h₁⟩
    rcases hmem _ hle.2 with ⟨a₂, b₂, h₂⟩
    have e10 : a₁ + b₁ = 1 := by simpa [ordCol, stdBasis] using (congr_fun h₁ 0).symm
    have e11 : a₁ * t₁ + b₁ * t₂ = 0 := by simpa [ordCol, stdBasis] using (congr_fun h₁ 1).symm
    have e12 : a₁ * t₁ ^ 2 + b₁ * t₂ ^ 2 = 0 := by simpa [ordCol, stdBasis] using (congr_fun h₁ 2).symm
    have e20 : a₂ + b₂ = 0 := by simpa [ordCol, stdBasis] using (congr_fun h₂ 0).symm
    have e21 : a₂ * t₁ + b₂ * t₂ = 1 := by simpa [ordCol, stdBasis] using (congr_fun h₂ 1).symm
    have e22 : a₂ * t₁ ^ 2 + b₂ * t₂ ^ 2 = 0 := by simpa [ordCol, stdBasis] using (congr_fun h₂ 2).symm
    -- From `a₂ = -b₂`, `b₂ * (t₂ - t₁) = 1` and `b₂ * (t₂^2 - t₁^2) = 0`, so `t₂ + t₁ = 0`.
    -- From `a₁ = 1 - b₁`, `t₁ + b₁ * (t₂ - t₁) = 0`, and `t₁^2 + b₁ * (t₂^2 - t₁^2) = 0`,
    -- giving `t₁^2 = 0` hence `t₁ = 0`, then `t₂ = 0`, contradicting `b₂ * (t₂ - t₁) = 1`!
    have hdiff : t₂ - t₁ ≠ 0 := by
      intro h0
      have : a₂ * t₁ + b₂ * t₂ = (a₂ + b₂) * t₁ + b₂ * (t₂ - t₁) := by ring
      rw [e21, e20, h0] at this
      simp at this
    have hsum0 : t₁ + t₂ = 0 := by
      have h1 : b₂ * (t₂ - t₁) = 1 := by linear_combination e21 - t₁ * e20
      have h2 : b₂ * (t₂ - t₁) * (t₁ + t₂) = 0 := by linear_combination e22 - t₁ ^ 2 * e20
      rw [h1, one_mul] at h2
      exact h2
    have ht1 : t₁ = 0 := by
      have h3 : t₁ * t₂ = 0 := by
        linear_combination (t₁ + t₂) * e11 - e12 - t₁ * t₂ * e10
      -- Wait: `(t₁ + t₂) * (a₁ t₁ + b₁ t₂) - (a₁ t₁² + b₁ t₂²) = t₁ t₂ (a₁ + b₁) = t₁ t₂`!
      have h4 : -t₁ ^ 2 = 0 := by
        have ht2 : t₂ = -t₁ := eq_neg_of_add_eq_zero_right hsum0
        rw [ht2] at h3
        linear_combination h3
      have h5 : t₁ ^ 2 = 0 := by linear_combination -h4
      exact pow_eq_zero h5
    have ht2 : t₂ = 0 := by rw [ht1, zero_add] at hsum0; exact hsum0
    exact hdiff (by rw [ht1, ht2, sub_self])
  · intro hle
    rw [fileF2_le_iff] at hle
    rcases hmem _ hle with ⟨a, b, h⟩
    have e0 : a + b = 0 := by simpa [ordCol, stdBasis] using (congr_fun h 0).symm
    have e1 : a * t₁ + b * t₂ = 0 := by simpa [ordCol, stdBasis] using (congr_fun h 1).symm
    have e2 : a * t₁ ^ 2 + b * t₂ ^ 2 = 1 := by simpa [ordCol, stdBasis] using (congr_fun h 2).symm
    have h1 : b * (t₂ - t₁) = 0 := by linear_combination e1 - t₁ * e0
    have h2 : b * (t₂ - t₁) * (t₁ + t₂) = 1 := by linear_combination e2 - t₁ ^ 2 * e0
    rw [h1, zero_mul] at h2
    exact zero_ne_one h2

/-- No span of `z = e₃` and at most one ordinary column contains `F₁ = ⟨e₁, e₂⟩`. -/
lemma span_one_ord_spec_not_F1 (t : ZMod 11) :
    ¬ (fileF1 (ZMod 11) ≤ Submodule.span (ZMod 11) {ordCol t, specCol}) := by
  intro hle
  rw [fileF1_le_iff] at hle
  rcases Submodule.mem_span_pair.mp hle.1 with ⟨a₁, b₁, h₁⟩
  rcases Submodule.mem_span_pair.mp hle.2 with ⟨a₂, b₂, h₂⟩
  have e20 : a₂ = 0 := by simpa [ordCol, specCol, stdBasis] using congr_fun h₂ 0
  have e21 : a₂ * t = 1 := by simpa [ordCol, specCol, stdBasis] using congr_fun h₂ 1
  rw [e20, zero_mul] at e21
  exact zero_ne_one e21

lemma finset_card_le_two {α : Type*} [DecidableEq α] [Nonempty α] {S : Finset α} (h : S.card ≤ 2) :
    ∃ a b : α, S ⊆ {a, b} := by
  by_cases h0 : S.card = 0
  · rw [Finset.card_eq_zero.mp h0]
    exact ⟨Classical.arbitrary α, Classical.arbitrary α, Finset.empty_subset _⟩
  · have hne : S.Nonempty := Finset.card_pos.mp (Nat.pos_of_ne_zero h0)
    rcases hne with ⟨a, ha⟩
    have herase : (S.erase a).card ≤ 1 := by
      have := Finset.card_erase_of_mem ha
      omega
    rcases Finset.card_le_one_iff_subset_singleton.mp herase with ⟨b, hb⟩
    refine ⟨a, b, fun x hx => ?_⟩
    by_cases hxa : x = a
    · simp [hxa]
    · have : x ∈ S.erase a := Finset.mem_erase.mpr ⟨hxa, hx⟩
      have hxb : x = b := Finset.mem_singleton.mp (hb this)
      simp [hxb]

/-- Exact recovery characterization for `F₁` on `G22`:
`F₁ ≤ spanCols G22 S ↔ ordCount S + (if hasSpecial S then 1 else 0) ≥ 3`. -/
theorem G22_rec_F1_iff (S : Finset (Fin 22)) :
    fileF1 (ZMod 11) ≤ spanCols G22 S ↔
      3 ≤ ordCount S + (if hasSpecial S then 1 else 0) := by
  let O := S.filter (fun i : Fin 22 => 11 ≤ i.1)
  have hO_card : O.card = ordCount S := rfl
  constructor
  · intro hle
    by_contra hlt
    push_neg at hlt
    cases hsp : hasSpecial S
    · -- No special column, `O.card ≤ 2`
      rw [hsp] at hlt
      simp only [Bool.false_eq_true, ↓reduceIte, add_zero] at hlt
      have hO_eq : O = S := Finset.filter_true_of_mem (hasSpecial_eq_false_iff S |>.mp hsp)
      -- Embed `G22 '' S` into `{ordCol t₁, ordCol t₂}` for some `t₁, t₂`
      have hle2 : ∃ t₁ t₂ : ZMod 11,
          spanCols G22 S ≤ Submodule.span (ZMod 11) {ordCol t₁, ordCol t₂} := by
        have hcard : S.card ≤ 2 := by rw [← hO_eq, hO_card]; omega
        rcases finset_card_le_two hcard with ⟨a, b, hsub⟩
        refine ⟨((a.1 - 11 : ℕ) : ZMod 11), ((b.1 - 11 : ℕ) : ZMod 11), ?_⟩
        apply Submodule.span_mono
        rintro _ ⟨i, hi, rfl⟩
        have hi11 : 11 ≤ i.1 := (hasSpecial_eq_false_iff S).mp hsp i hi
        have hnotlt : ¬ (i.1 < 11) := by omega
        simp only [G22, hnotlt, ↓reduceIte]
        have hiab : i = a ∨ i = b := by simpa using hsub hi
        rcases hiab with rfl | rfl <;> simp
      rcases hle2 with ⟨t₁, t₂, hsub⟩
      exact (span_two_ord_not_F1_F2 t₁ t₂).1 (hle.trans hsub)
    · -- Has special column, `O.card ≤ 1`
      rw [hsp] at hlt
      simp only [↓reduceIte] at hlt
      have hcard : O.card ≤ 1 := by omega
      rcases Finset.card_le_one_iff_subset_singleton.mp hcard with ⟨a, hsub⟩
      have hsub_span : spanCols G22 S ≤
          Submodule.span (ZMod 11) {ordCol ((a.1 - 11 : ℕ) : ZMod 11), specCol} := by
        apply Submodule.span_mono
        rintro _ ⟨i, hi, rfl⟩
        by_cases hilt : i.1 < 11
        · simp [G22, hilt]
        · have hiO : i ∈ O := Finset.mem_filter.mpr ⟨hi, by omega⟩
          have hia : i = a := Finset.mem_singleton.mp (hsub hiO)
          subst hia
          simp [G22, hilt]
      exact span_one_ord_spec_not_F1 _ (hle.trans hsub_span)
  · intro hge
    -- Show `spanCols G22 S = ⊤`, hence `fileF1 ≤ ⊤`
    have htop : spanCols G22 S = ⊤ := by
      cases hsp : hasSpecial S
      · rw [hsp] at hge
        simp only [Bool.false_eq_true, ↓reduceIte, add_zero] at hge
        have h3 : 2 < O.card := by omega
        rcases Finset.two_lt_card.mp h3 with ⟨i₁, hi₁, i₂, hi₂, i₃, hi₃, h12, h13, h23⟩
        have hi₁' := Finset.mem_filter.mp hi₁
        have hi₂' := Finset.mem_filter.mp hi₂
        have hi₃' := Finset.mem_filter.mp hi₃
        have ht12 : ((i₁.1 - 11 : ℕ) : ZMod 11) ≠ ((i₂.1 - 11 : ℕ) : ZMod 11) := fun h =>
          h12 (ordCol_val_inj hi₁'.2 hi₂'.2 h)
        have ht13 : ((i₁.1 - 11 : ℕ) : ZMod 11) ≠ ((i₃.1 - 11 : ℕ) : ZMod 11) := fun h =>
          h13 (ordCol_val_inj hi₁'.2 hi₃'.2 h)
        have ht23 : ((i₂.1 - 11 : ℕ) : ZMod 11) ≠ ((i₃.1 - 11 : ℕ) : ZMod 11) := fun h =>
          h23 (ordCol_val_inj hi₂'.2 hi₃'.2 h)
        rw [eq_top_iff, ← span_three_ord ht12 ht13 ht23]
        apply Submodule.span_mono
        intro x hx
        simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at hx
        rcases hx with rfl | rfl | rfl
        · exact ⟨i₁, hi₁'.1, by simp [G22, show ¬ i₁.1 < 11 by omega]⟩
        · exact ⟨i₂, hi₂'.1, by simp [G22, show ¬ i₂.1 < 11 by omega]⟩
        · exact ⟨i₃, hi₃'.1, by simp [G22, show ¬ i₃.1 < 11 by omega]⟩
      · rw [hsp] at hge
        simp only [↓reduceIte] at hge
        have h2 : 1 < O.card := by omega
        rcases Finset.one_lt_card.mp h2 with ⟨i₁, hi₁, i₂, hi₂, h12⟩
        rcases (hasSpecial_eq_true_iff S).mp hsp with ⟨j, hj, hjlt⟩
        have hi₁' := Finset.mem_filter.mp hi₁
        have hi₂' := Finset.mem_filter.mp hi₂
        have ht12 : ((i₁.1 - 11 : ℕ) : ZMod 11) ≠ ((i₂.1 - 11 : ℕ) : ZMod 11) := fun h =>
          h12 (ordCol_val_inj hi₁'.2 hi₂'.2 h)
        rw [eq_top_iff, ← span_two_ord_spec ht12]
        apply Submodule.span_mono
        intro x hx
        simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at hx
        rcases hx with rfl | rfl | rfl
        · exact ⟨i₁, hi₁'.1, by simp [G22, show ¬ i₁.1 < 11 by omega]⟩
        · exact ⟨i₂, hi₂'.1, by simp [G22, show ¬ i₂.1 < 11 by omega]⟩
        · exact ⟨j, hj, by simp [G22, hjlt]⟩
    rw [htop]
    exact le_top

/-- Exact recovery characterization for `F₂` on `G22`:
`F₂ ≤ spanCols G22 S ↔ hasSpecial S = true ∨ 3 ≤ ordCount S`. -/
theorem G22_rec_F2_iff (S : Finset (Fin 22)) :
    fileF2 (ZMod 11) ≤ spanCols G22 S ↔ hasSpecial S = true ∨ 3 ≤ ordCount S := by
  let O := S.filter (fun i : Fin 22 => 11 ≤ i.1)
  constructor
  · intro hle
    by_contra hnor
    push_neg at hnor
    have hsp : hasSpecial S = false := by
      cases h : hasSpecial S
      · rfl
      · exact absurd h hnor.1
    have hO_eq : O = S := Finset.filter_true_of_mem ((hasSpecial_eq_false_iff S).mp hsp)
    have hcard : S.card ≤ 2 := by rw [← hO_eq]; change ordCount S ≤ 2; omega
    rcases finset_card_le_two hcard with ⟨a, b, hsub⟩
    have hsub_span : spanCols G22 S ≤
        Submodule.span (ZMod 11)
          {ordCol ((a.1 - 11 : ℕ) : ZMod 11), ordCol ((b.1 - 11 : ℕ) : ZMod 11)} := by
      apply Submodule.span_mono
      rintro _ ⟨i, hi, rfl⟩
      have hi11 : 11 ≤ i.1 := (hasSpecial_eq_false_iff S).mp hsp i hi
      simp only [G22, show ¬ i.1 < 11 by omega, ↓reduceIte]
      have hiab : i = a ∨ i = b := by simpa using hsub hi
      rcases hiab with rfl | rfl <;> simp
    exact (span_two_ord_not_F1_F2 _ _).2 (hle.trans hsub_span)
  · rintro (hsp | h3)
    · rcases (hasSpecial_eq_true_iff S).mp hsp with ⟨j, hj, hjlt⟩
      rw [fileF2_le_iff]
      exact Submodule.subset_span ⟨j, hj, by simp [G22, hjlt, specCol]⟩
    · have hcard : 2 < O.card := by change 2 < ordCount S; omega
      rcases Finset.two_lt_card.mp hcard with ⟨i₁, hi₁, i₂, hi₂, i₃, hi₃, h12, h13, h23⟩
      have hi₁' := Finset.mem_filter.mp hi₁
      have hi₂' := Finset.mem_filter.mp hi₂
      have hi₃' := Finset.mem_filter.mp hi₃
      have ht12 : ((i₁.1 - 11 : ℕ) : ZMod 11) ≠ ((i₂.1 - 11 : ℕ) : ZMod 11) := fun h =>
        h12 (ordCol_val_inj hi₁'.2 hi₂'.2 h)
      have ht13 : ((i₁.1 - 11 : ℕ) : ZMod 11) ≠ ((i₃.1 - 11 : ℕ) : ZMod 11) := fun h =>
        h13 (ordCol_val_inj hi₁'.2 hi₃'.2 h)
      have ht23 : ((i₂.1 - 11 : ℕ) : ZMod 11) ≠ ((i₃.1 - 11 : ℕ) : ZMod 11) := fun h =>
        h23 (ordCol_val_inj hi₂'.2 hi₃'.2 h)
      have htop : spanCols G22 S = ⊤ := by
        rw [eq_top_iff, ← span_three_ord ht12 ht13 ht23]
        apply Submodule.span_mono
        intro x hx
        simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at hx
        rcases hx with rfl | rfl | rfl
        · exact ⟨i₁, hi₁'.1, by simp [G22, show ¬ i₁.1 < 11 by omega]⟩
        · exact ⟨i₂, hi₂'.1, by simp [G22, show ¬ i₂.1 < 11 by omega]⟩
        · exact ⟨i₃, hi₃'.1, by simp [G22, show ¬ i₃.1 < 11 by omega]⟩
      rw [htop]
      exact le_top

/-- Exact closed-form DP table for `W₁(j, h)` and `W₂(j, h)` on `G22`. -/
def W1Table : ℕ → Bool → ℚ
  | 0, false => 937 / 210
  | 1, false => 286 / 105
  | 2, false => 11 / 10
  | 0, true  => 21 / 5
  | 1, true  => 11 / 5
  | _, _     => 0

def W2Table : ℕ → Bool → ℚ
  | 0, false => 25 / 14
  | 1, false => 11 / 7
  | 2, false => 11 / 10
  | _, _     => 0

/-- Helper splitting a sum over `Fin 22` by whether `i` is special (`i < 11`),
an already-seen ordinary index, or an unseen ordinary index. -/
lemma sum_fin22_state_transitions (S : Finset (Fin 22)) (f : ℕ → Bool → ℚ) :
    ∑ i : Fin 22, f (ordCount (insert i S)) (hasSpecial (insert i S)) =
      (11 : ℚ) * f (ordCount S) true +
      (ordCount S : ℚ) * f (ordCount S) (hasSpecial S) +
      ((11 - ordCount S : ℕ) : ℚ) * f (ordCount S + 1) (hasSpecial S) := by
  let Spec : Finset (Fin 22) := Finset.univ.filter (fun i : Fin 22 => i.1 < 11)
  let OrdSeen : Finset (Fin 22) := S.filter (fun i : Fin 22 => 11 ≤ i.1)
  let OrdNew : Finset (Fin 22) := (Finset.univ \ S).filter (fun i : Fin 22 => 11 ≤ i.1)
  have hSpec_card : Spec.card = 11 := by decide
  have hOrdAll_card : (Finset.univ.filter (fun i : Fin 22 => 11 ≤ i.1)).card = 11 := by decide
  have hOrdNew_card : OrdNew.card = 11 - ordCount S := by
    have hdisj : Disjoint OrdSeen OrdNew := by
      rw [Finset.disjoint_left]
      intro i hi₁ hi₂
      exact (Finset.mem_sdiff.mp (Finset.mem_filter.mp hi₂).1).2 (Finset.mem_filter.mp hi₁).1
    have hunion : OrdSeen ∪ OrdNew = Finset.univ.filter (fun i : Fin 22 => 11 ≤ i.1) := by
      ext i
      simp only [OrdSeen, OrdNew, Finset.mem_union, Finset.mem_filter, Finset.mem_sdiff,
        Finset.mem_univ, true_and]
      tauto
    have hsum : OrdSeen.card + OrdNew.card = 11 := by
      rw [← Finset.card_union_of_disjoint hdisj, hunion, hOrdAll_card]
    change OrdNew.card = 11 - OrdSeen.card
    omega
  have hpart : Spec ∪ (OrdSeen ∪ OrdNew) = Finset.univ := by
    ext i
    simp only [Spec, OrdSeen, OrdNew, Finset.mem_union, Finset.mem_filter, Finset.mem_sdiff,
      Finset.mem_univ, true_and, iff_true]
    by_cases hlt : i.1 < 11
    · left; exact hlt
    · right
      by_cases hi : i ∈ S
      · left; exact ⟨hi, by omega⟩
      · right; exact ⟨hi, by omega⟩
  have hd1 : Disjoint Spec (OrdSeen ∪ OrdNew) := by
    rw [Finset.disjoint_left]
    intro i hi₁ hi₂
    have hlt := (Finset.mem_filter.mp hi₁).2
    rcases Finset.mem_union.mp hi₂ with h | h
    · have := (Finset.mem_filter.mp h).2; omega
    · have := (Finset.mem_filter.mp h).2; omega
  have hd2 : Disjoint OrdSeen OrdNew := by
    rw [Finset.disjoint_left]
    intro i hi₁ hi₂
    exact (Finset.mem_sdiff.mp (Finset.mem_filter.mp hi₂).1).2 (Finset.mem_filter.mp hi₁).1
  rw [← hpart, Finset.sum_union hd1, Finset.sum_union hd2, ← add_assoc]
  congr 1
  · congr 1
    · have hcongr : ∑ i ∈ Spec, f (ordCount (insert i S)) (hasSpecial (insert i S)) =
          ∑ _i ∈ Spec, f (ordCount S) true := by
        refine Finset.sum_congr rfl fun i hi => ?_
        have hilt : i.1 < 11 := (Finset.mem_filter.mp hi).2
        have hord : ordCount (insert i S) = ordCount S := by
          unfold ordCount
          rw [Finset.filter_insert]
          simp [show ¬ (11 ≤ i.1) by omega]
        have hsp : hasSpecial (insert i S) = true :=
          (hasSpecial_eq_true_iff _).mpr ⟨i, Finset.mem_insert_self _ _, hilt⟩
        rw [hord, hsp]
      rw [hcongr, Finset.sum_const, hSpec_card, nsmul_eq_mul]
      norm_cast
    · have hcongr : ∑ i ∈ OrdSeen, f (ordCount (insert i S)) (hasSpecial (insert i S)) =
          ∑ _i ∈ OrdSeen, f (ordCount S) (hasSpecial S) := by
        refine Finset.sum_congr rfl fun i hi => ?_
        rw [Finset.insert_eq_of_mem (Finset.mem_filter.mp hi).1]
      rw [hcongr, Finset.sum_const, nsmul_eq_mul]
      rfl
  · have hcongr : ∑ i ∈ OrdNew, f (ordCount (insert i S)) (hasSpecial (insert i S)) =
        ∑ _i ∈ OrdNew, f (ordCount S + 1) (hasSpecial S) := by
      refine Finset.sum_congr rfl fun i hi => ?_
      have hi' := Finset.mem_filter.mp hi
      have hnotS : i ∉ S := (Finset.mem_sdiff.mp hi'.1).2
      have hge11 : 11 ≤ i.1 := hi'.2
      have hord : ordCount (insert i S) = ordCount S + 1 := by
        unfold ordCount
        rw [Finset.filter_insert, if_pos hge11, Finset.card_insert_of_notMem]
        intro hmem
        exact hnotS (Finset.mem_filter.mp hmem).1
      have hsp : hasSpecial (insert i S) = hasSpecial S := by
        unfold hasSpecial
        apply decide_eq_decide.mpr
        simp only [Finset.mem_insert]
        constructor
        · rintro ⟨x, rfl | hx, hxlt⟩
          · omega
          · exact ⟨x, hx, hxlt⟩
        · rintro ⟨x, hx, hxlt⟩
          exact ⟨x, Or.inr hx, hxlt⟩
      rw [hord, hsp]
    rw [hcongr, Finset.sum_const, hOrdNew_card, nsmul_eq_mul]

/-- Exact value `E₁(G22) = 937 / 210`. -/
theorem G22_E1 :
    haveI : DecidablePred (fun S : Finset (Fin 22) => fileF1 (ZMod 11) ≤ spanCols G22 S) :=
      Classical.decPred _
    expectedWait (fun S => fileF1 (ZMod 11) ≤ spanCols G22 S)
      (by dsimp only; rw [G22_full_rank]; exact le_top) ∅ = 937 / 210 := by
  classical
  have hUniv : (fun S : Finset (Fin 22) => fileF1 (ZMod 11) ≤ spanCols G22 S) Finset.univ := by
    dsimp only; rw [G22_full_rank]; exact le_top
  have hEq := expectedWait_unique
    (fun S : Finset (Fin 22) => fileF1 (ZMod 11) ≤ spanCols G22 S) hUniv
    (fun S => W1Table (ordCount S) (hasSpecial S))
    (fun S hS => by
      dsimp only at hS ⊢
      rw [G22_rec_F1_iff] at hS
      cases hsp : hasSpecial S <;> rw [hsp] at hS
      · simp only [Bool.false_eq_true, ↓reduceIte, add_zero] at hS
        rcases Nat.exists_eq_add_of_le hS with ⟨d, hd⟩
        rw [show ordCount S = d + 3 by omega]
        simp [W1Table]
      · simp only [↓reduceIte] at hS
        rcases Nat.exists_eq_add_of_le hS with ⟨d, hd⟩
        rw [show ordCount S = d + 2 by omega]
        simp [W1Table])
    (fun S hS => by
      dsimp only at hS ⊢
      rw [G22_rec_F1_iff] at hS
      rw [Fintype.card_fin, sum_fin22_state_transitions S W1Table]
      cases hsp : hasSpecial S <;> rw [hsp] at hS
      · simp only [Bool.false_eq_true, ↓reduceIte, add_zero] at hS
        have hlt : ordCount S < 3 := by omega
        interval_cases h : ordCount S <;> norm_num [W1Table]
      · simp only [↓reduceIte] at hS
        have hlt : ordCount S < 2 := by omega
        interval_cases h : ordCount S <;> norm_num [W1Table])
    ∅
  have h0 : ordCount ∅ = 0 := rfl
  have hs0 : hasSpecial ∅ = false := rfl
  rw [← hEq, h0, hs0]
  rfl

/-- Exact value `E₂(G22) = 25 / 14`. -/
theorem G22_E2 :
    haveI : DecidablePred (fun S : Finset (Fin 22) => fileF2 (ZMod 11) ≤ spanCols G22 S) :=
      Classical.decPred _
    expectedWait (fun S => fileF2 (ZMod 11) ≤ spanCols G22 S)
      (by dsimp only; rw [G22_full_rank]; exact le_top) ∅ = 25 / 14 := by
  classical
  have hUniv : (fun S : Finset (Fin 22) => fileF2 (ZMod 11) ≤ spanCols G22 S) Finset.univ := by
    dsimp only; rw [G22_full_rank]; exact le_top
  have hEq := expectedWait_unique
    (fun S : Finset (Fin 22) => fileF2 (ZMod 11) ≤ spanCols G22 S) hUniv
    (fun S => W2Table (ordCount S) (hasSpecial S))
    (fun S hS => by
      dsimp only at hS ⊢
      rw [G22_rec_F2_iff] at hS
      rcases hS with hsp | h3
      · rw [hsp]; cases ordCount S <;> simp [W2Table]
      · rcases Nat.exists_eq_add_of_le h3 with ⟨d, hd⟩
        rw [show ordCount S = d + 3 by omega]
        cases hasSpecial S <;> simp [W2Table])
    (fun S hS => by
      dsimp only at hS ⊢
      rw [G22_rec_F2_iff] at hS
      push_neg at hS
      have hsp : hasSpecial S = false := by
        cases h : hasSpecial S
        · rfl
        · exact absurd h hS.1
      rw [Fintype.card_fin, sum_fin22_state_transitions S W2Table, hsp]
      have hlt : ordCount S < 3 := hS.2
      interval_cases h : ordCount S <;> norm_num [W2Table])
    ∅
  have h0 : ordCount ∅ = 0 := rfl
  have hs0 : hasSpecial ∅ = false := rfl
  rw [← hEq, h0, hs0]
  rfl

/-- **Refutation of `conj:hyperbola`**:
at `(s₁, s₂) = (2, 1)` over `𝔽₁₁`, `G22` achieves `2 / E₁ + 1 / E₂ = 23618 / 23425 > 1`. -/
theorem refute_conj_hyperbola :
    haveI : DecidablePred (fun S : Finset (Fin 22) => fileF1 (ZMod 11) ≤ spanCols G22 S) :=
      Classical.decPred _
    haveI : DecidablePred (fun S : Finset (Fin 22) => fileF2 (ZMod 11) ≤ spanCols G22 S) :=
      Classical.decPred _
    let E₁ := expectedWait (fun S => fileF1 (ZMod 11) ≤ spanCols G22 S)
      (by dsimp only; rw [G22_full_rank]; exact le_top) ∅
    let E₂ := expectedWait (fun S => fileF2 (ZMod 11) ≤ spanCols G22 S)
      (by dsimp only; rw [G22_full_rank]; exact le_top) ∅
    2 / E₁ + 1 / E₂ = 23618 / 23425 ∧ (1 : ℚ) < 2 / E₁ + 1 / E₂ := by
  classical
  dsimp only
  rw [G22_E1, G22_E2]
  constructor <;> norm_num

/-- **Refutation of `conj:limiting`**:
for every integer `m ≥ 1`, the `m`-fold repetition of `G22` on `Fin m × Fin 22` (length `22m`)
achieves the exact same pair `(937/210, 25/14)` with `2/E₁ + 1/E₂ = 23618/23425 > 1`. -/
theorem refute_conj_limiting (m : ℕ) (hm : 1 ≤ m) :
    haveI : DecidablePred (fun S : Finset (Fin m × Fin 22) =>
      fileF1 (ZMod 11) ≤ Submodule.span (ZMod 11) ((fun p : Fin m × Fin 22 => G22 p.2) '' (S : Set _))) :=
      Classical.decPred _
    haveI : DecidablePred (fun S : Finset (Fin m × Fin 22) =>
      fileF2 (ZMod 11) ≤ Submodule.span (ZMod 11) ((fun p : Fin m × Fin 22 => G22 p.2) '' (S : Set _))) :=
      Classical.decPred _
    have hUniv : Submodule.span (ZMod 11)
        ((fun p : Fin m × Fin 22 => G22 p.2) '' ((Finset.univ : Finset (Fin m × Fin 22)) : Set _)) = ⊤ := by
      have himg : (fun p : Fin m × Fin 22 => G22 p.2) '' ((Finset.univ : Finset (Fin m × Fin 22)) : Set _) =
          G22 '' ((Finset.univ : Finset (Fin 22)) : Set _) := by
        ext x
        simp only [Set.mem_image, Finset.coe_univ, Set.mem_univ, true_and]
        exact ⟨fun ⟨p, hp⟩ => ⟨p.2, hp⟩, fun ⟨j, hj⟩ => ⟨(⟨0, hm⟩, j), hj⟩⟩
      rw [himg]
      exact G22_full_rank
    let E₁ := expectedWait
      (fun S : Finset (Fin m × Fin 22) =>
        fileF1 (ZMod 11) ≤ Submodule.span (ZMod 11) ((fun p : Fin m × Fin 22 => G22 p.2) '' (S : Set _)))
      (by dsimp only; rw [hUniv]; exact le_top) ∅
    let E₂ := expectedWait
      (fun S : Finset (Fin m × Fin 22) =>
        fileF2 (ZMod 11) ≤ Submodule.span (ZMod 11) ((fun p : Fin m × Fin 22 => G22 p.2) '' (S : Set _)))
      (by dsimp only; rw [hUniv]; exact le_top) ∅
    E₁ = 937 / 210 ∧ E₂ = 25 / 14 ∧ (1 : ℚ) < 2 / E₁ + 1 / E₂ := by
  classical
  have hspan_eq : ∀ S : Finset (Fin m × Fin 22),
      Submodule.span (ZMod 11) ((fun p : Fin m × Fin 22 => G22 p.2) '' (S : Set _)) =
      spanCols G22 (S.image Prod.snd) := by
    intro S
    unfold spanCols
    congr 1
    ext x
    simp only [Set.mem_image, Finset.mem_coe, Finset.mem_image]
    constructor
    · rintro ⟨p, hp, rfl⟩; exact ⟨p.2, ⟨p, hp, rfl⟩, rfl⟩
    · rintro ⟨j, ⟨p, hp, rfl⟩, rfl⟩; exact ⟨p, hp, rfl⟩
  have h1 : ∀ hU₁ hU₂,
      expectedWait
        (fun S : Finset (Fin m × Fin 22) =>
          fileF1 (ZMod 11) ≤ Submodule.span (ZMod 11) ((fun p : Fin m × Fin 22 => G22 p.2) '' (S : Set _)))
        hU₁ ∅ =
      expectedWait (fun S : Finset (Fin 22) => fileF1 (ZMod 11) ≤ spanCols G22 S) hU₂ ∅ := by
    intro hU₁ hU₂
    haveI : Nonempty (Fin m × Fin 22) := ⟨(⟨0, hm⟩, ⟨0, by decide⟩)⟩
    have hEq := expectedWait_repeat m hm
      (fun S : Finset (Fin 22) => fileF1 (ZMod 11) ≤ spanCols G22 S) hU₂
    have hUrep : (fun S : Finset (Fin m × Fin 22) =>
        fileF1 (ZMod 11) ≤ spanCols G22 (S.image Prod.snd)) Finset.univ := by
      dsimp only; rw [← hspan_eq]; exact hU₁
    have huniq := expectedWait_unique
      (fun S : Finset (Fin m × Fin 22) =>
        fileF1 (ZMod 11) ≤ Submodule.span (ZMod 11) ((fun p : Fin m × Fin 22 => G22 p.2) '' (S : Set _)))
      hU₁
      (expectedWait (fun S : Finset (Fin m × Fin 22) =>
        fileF1 (ZMod 11) ≤ spanCols G22 (S.image Prod.snd)) hUrep)
      (fun S hS => expectedWait_of_rec _ hUrep (by dsimp only at hS ⊢; rwa [← hspan_eq]))
      (fun S hS => expectedWait_bellman _ hUrep (by dsimp only at hS ⊢; rwa [← hspan_eq]))
      ∅
    rw [← huniq]
    exact hEq
  have h2 : ∀ hU₁ hU₂,
      expectedWait
        (fun S : Finset (Fin m × Fin 22) =>
          fileF2 (ZMod 11) ≤ Submodule.span (ZMod 11) ((fun p : Fin m × Fin 22 => G22 p.2) '' (S : Set _)))
        hU₁ ∅ =
      expectedWait (fun S : Finset (Fin 22) => fileF2 (ZMod 11) ≤ spanCols G22 S) hU₂ ∅ := by
    intro hU₁ hU₂
    haveI : Nonempty (Fin m × Fin 22) := ⟨(⟨0, hm⟩, ⟨0, by decide⟩)⟩
    have hEq := expectedWait_repeat m hm
      (fun S : Finset (Fin 22) => fileF2 (ZMod 11) ≤ spanCols G22 S) hU₂
    have hUrep : (fun S : Finset (Fin m × Fin 22) =>
        fileF2 (ZMod 11) ≤ spanCols G22 (S.image Prod.snd)) Finset.univ := by
      dsimp only; rw [← hspan_eq]; exact hU₁
    have huniq := expectedWait_unique
      (fun S : Finset (Fin m × Fin 22) =>
        fileF2 (ZMod 11) ≤ Submodule.span (ZMod 11) ((fun p : Fin m × Fin 22 => G22 p.2) '' (S : Set _)))
      hU₁
      (expectedWait (fun S : Finset (Fin m × Fin 22) =>
        fileF2 (ZMod 11) ≤ spanCols G22 (S.image Prod.snd)) hUrep)
      (fun S hS => expectedWait_of_rec _ hUrep (by dsimp only at hS ⊢; rwa [← hspan_eq]))
      (fun S hS => expectedWait_bellman _ hUrep (by dsimp only at hS ⊢; rwa [← hspan_eq]))
      ∅
    rw [← huniq]
    exact hEq
  dsimp only
  rw [h1 _ (by dsimp only; rw [G22_full_rank]; exact le_top),
    h2 _ (by dsimp only; rw [G22_full_rank]; exact le_top),
    G22_E1, G22_E2]
  exact ⟨rfl, rfl, by norm_num⟩

/-! ### 4. Refutation of `conj:monotone` Over Every Field `K` -/

/-- Exact expected wait to cover a 2-element target set `{a, b}` (`a ≠ b`) in `Fin n`: `3n / 2`. -/
theorem expectedWait_pair {n : ℕ} [NeZero n] {a b : Fin n} (hab : a ≠ b) :
    expectedWait (fun S : Finset (Fin n) => a ∈ S ∧ b ∈ S)
      ⟨Finset.mem_univ a, Finset.mem_univ b⟩ ∅ = 3 * (n : ℚ) / 2 := by
  haveI : Nonempty (Fin n) := ⟨a⟩
  let W : Finset (Fin n) → ℚ := fun S =>
    if a ∈ S ∧ b ∈ S then 0
    else if a ∈ S ∨ b ∈ S then (n : ℚ)
    else 3 * (n : ℚ) / 2
  have hEq := expectedWait_unique
    (fun S : Finset (Fin n) => a ∈ S ∧ b ∈ S) ⟨Finset.mem_univ a, Finset.mem_univ b⟩ W
    (fun S hS => by simp [W, hS])
    (fun S hS => by
      have hn : (n : ℚ) ≠ 0 := Nat.cast_ne_zero.mpr (NeZero.ne n)
      by_cases hor : a ∈ S ∨ b ∈ S
      · -- Exactly one of `a, b` is missing
        have hW : W S = (n : ℚ) := by simp [W, hS, hor]
        rw [hW, Fintype.card_fin]
        rcases hor with ha | hb
        · have hb : b ∉ S := fun h => hS ⟨ha, h⟩
          have hsum : ∑ i : Fin n, W (insert i S) = ((n - 1 : ℕ) : ℚ) * (n : ℚ) := by
            have hcongr : ∑ i : Fin n, W (insert i S) =
                ∑ i : Fin n, if i = b then (0 : ℚ) else (n : ℚ) := by
              refine Finset.sum_congr rfl fun i _ => ?_
              by_cases hib : i = b
              · subst hib; simp [W, ha]
              · have hnb : b ∉ insert i S := by simp [Ne.symm hib, hb]
                have hna : a ∈ insert i S := Finset.mem_insert_of_mem ha
                simp [W, hib, hnb, hna]
            rw [hcongr, Finset.sum_ite, Finset.sum_const_zero, zero_add, Finset.sum_const,
              nsmul_eq_mul]
            have hcard : (Finset.univ.filter (fun i : Fin n => ¬ i = b)).card = n - 1 := by
              have : Finset.univ.filter (fun i : Fin n => ¬ i = b) = Finset.univ.erase b := by
                ext x; simp
              rw [this, Finset.card_erase_of_mem (Finset.mem_univ b), Finset.card_univ,
                Fintype.card_fin]
            rw [hcard]
          rw [hsum]
          have h1 : 1 ≤ n := Nat.pos_of_ne_zero (NeZero.ne n)
          rw [Nat.cast_sub h1, Nat.cast_one]
          field_simp [hn]
          ring
        · have ha : a ∉ S := fun h => hS ⟨h, hb⟩
          have hsum : ∑ i : Fin n, W (insert i S) = ((n - 1 : ℕ) : ℚ) * (n : ℚ) := by
            have hcongr : ∑ i : Fin n, W (insert i S) =
                ∑ i : Fin n, if i = a then (0 : ℚ) else (n : ℚ) := by
              refine Finset.sum_congr rfl fun i _ => ?_
              by_cases hia : i = a
              · subst hia; simp [W, hb]
              · have hna : a ∉ insert i S := by simp [Ne.symm hia, ha]
                have hnb : b ∈ insert i S := Finset.mem_insert_of_mem hb
                simp [W, hia, hna, hnb]
            rw [hcongr, Finset.sum_ite, Finset.sum_const_zero, zero_add, Finset.sum_const,
              nsmul_eq_mul]
            have hcard : (Finset.univ.filter (fun i : Fin n => ¬ i = a)).card = n - 1 := by
              have : Finset.univ.filter (fun i : Fin n => ¬ i = a) = Finset.univ.erase a := by
                ext x; simp
              rw [this, Finset.card_erase_of_mem (Finset.mem_univ a), Finset.card_univ,
                Fintype.card_fin]
            rw [hcard]
          rw [hsum]
          have h1 : 1 ≤ n := Nat.pos_of_ne_zero (NeZero.ne n)
          rw [Nat.cast_sub h1, Nat.cast_one]
          field_simp [hn]
          ring
      · -- Neither `a` nor `b` is in `S`
        push_neg at hor
        have hW : W S = 3 * (n : ℚ) / 2 := by simp [W, hor.1, hor.2]
        rw [hW, Fintype.card_fin]
        have hsum : ∑ i : Fin n, W (insert i S) =
            2 * (n : ℚ) + ((n - 2 : ℕ) : ℚ) * (3 * (n : ℚ) / 2) := by
          have hcongr : ∑ i : Fin n, W (insert i S) =
              ∑ i : Fin n, if i = a ∨ i = b then (n : ℚ) else 3 * (n : ℚ) / 2 := by
            refine Finset.sum_congr rfl fun i _ => ?_
            by_cases hia : i = a
            · subst hia; simp [W, hor.2, hab.symm]
            · by_cases hib : i = b
              · subst hib; simp [W, hor.1, hab]
              · simp [W, hia, hib, hor.1, hor.2, Ne.symm hia, Ne.symm hib]
          rw [hcongr, Finset.sum_ite, Finset.sum_const, Finset.sum_const, nsmul_eq_mul, nsmul_eq_mul]
          have hpair : Finset.univ.filter (fun i : Fin n => i = a ∨ i = b) = {a, b} := by
            ext x; simp
          have hpair_card : ({a, b} : Finset (Fin n)).card = 2 := Finset.card_pair hab
          have hrest : (Finset.univ.filter (fun i : Fin n => ¬ (i = a ∨ i = b))).card = n - 2 := by
            have hEq' : Finset.univ.filter (fun i : Fin n => ¬ (i = a ∨ i = b)) =
                Finset.univ \ {a, b} := by
              ext x; simp
            rw [hEq', Finset.card_sdiff, Finset.inter_univ, Finset.card_univ, Fintype.card_fin,
              hpair_card]
          rw [hpair, hpair_card, hrest]
          norm_num
        rw [hsum]
        have h2 : 2 ≤ n := by
          have hlt : 1 < Fintype.card (Fin n) := Fintype.one_lt_card_iff.mpr ⟨a, b, hab⟩
          rw [Fintype.card_fin] at hlt
          omega
        rw [Nat.cast_sub h2, Nat.cast_ofNat]
        field_simp [hn]
        ring)
    ∅
  rw [← hEq]
  simp [W]

/-- Exact expected wait to hit any of `k` good indices in `Fin n`: `n / k`. -/
theorem expectedWait_hit {n : ℕ} [NeZero n] (Good : Finset (Fin n)) (hG : Good.Nonempty) :
    expectedWait (fun S : Finset (Fin n) => (S ∩ Good).Nonempty)
      (by simpa using hG) ∅ = (n : ℚ) / (Good.card : ℚ) := by
  haveI : Nonempty (Fin n) := ⟨hG.choose⟩
  have hG_pos : (0 : ℚ) < (Good.card : ℚ) := Nat.cast_pos.mpr (Finset.card_pos.mpr hG)
  have hG_ne : (Good.card : ℚ) ≠ 0 := ne_of_gt hG_pos
  have hn_ne : (n : ℚ) ≠ 0 := Nat.cast_ne_zero.mpr (NeZero.ne n)
  let W : Finset (Fin n) → ℚ := fun S =>
    if (S ∩ Good).Nonempty then 0 else (n : ℚ) / (Good.card : ℚ)
  have hEq := expectedWait_unique
    (fun S : Finset (Fin n) => (S ∩ Good).Nonempty) (by simpa using hG) W
    (fun S hS => by simp [W, hS])
    (fun S hS => by
      rw [Fintype.card_fin]
      simp only [W, hS, ↓reduceIte]
      have hcongr : ∑ i : Fin n, W (insert i S) =
          ∑ i : Fin n, if i ∈ Good then (0 : ℚ) else (n : ℚ) / (Good.card : ℚ) := by
        refine Finset.sum_congr rfl fun i _ => ?_
        by_cases hi : i ∈ Good
        · have hne : ((insert i S) ∩ Good).Nonempty :=
            ⟨i, Finset.mem_inter.mpr ⟨Finset.mem_insert_self _ _, hi⟩⟩
          simp only [W, hi, hne, ↓reduceIte]
        · have hempty : ¬ ((insert i S) ∩ Good).Nonempty := by
            rintro ⟨x, hx⟩
            rcases Finset.mem_inter.mp hx with ⟨hx1, hx2⟩
            rcases Finset.mem_insert.mp hx1 with rfl | hxS
            · exact hi hx2
            · exact hS ⟨x, Finset.mem_inter.mpr ⟨hxS, hx2⟩⟩
          simp only [W, hi, hempty, ↓reduceIte]
      rw [hcongr, Finset.sum_ite, Finset.sum_const_zero, zero_add, Finset.sum_const, nsmul_eq_mul]
      have hcard : (Finset.univ.filter (fun i : Fin n => i ∉ Good)).card = n - Good.card := by
        have : Finset.univ.filter (fun i : Fin n => i ∉ Good) = Finset.univ \ Good := by
          ext x; simp
        rw [this, Finset.card_sdiff, Finset.inter_univ, Finset.card_univ, Fintype.card_fin]
      rw [hcard]
      have hle : Good.card ≤ n := by
        have := Finset.card_le_univ Good
        rwa [Fintype.card_fin] at this
      rw [Nat.cast_sub hle]
      field_simp [hn_ne, hG_ne]
      ring)
    ∅
  rw [← hEq]
  simp [W]

/-- The length-9 counterexample matrix `G9 = [e₁, e₂, e₃⁷]` over ANY field `K`. -/
def G9 (K : Type*) [Field K] (i : Fin 9) : Fin 3 → K :=
  if i = 0 then stdBasis (0 : Fin 3)
  else if i = 1 then stdBasis (1 : Fin 3)
  else stdBasis (2 : Fin 3)

theorem G9_full_rank (K : Type*) [Field K] : spanCols (G9 K) Finset.univ = ⊤ := by
  rw [eq_top_iff]
  intro w _
  have hEq : w = w 0 • stdBasis (0 : Fin 3) + w 1 • stdBasis (1 : Fin 3) +
      w 2 • stdBasis (2 : Fin 3) := by
    ext i
    fin_cases i <;> simp [stdBasis]
  rw [hEq]
  have h0 : stdBasis (K := K) (0 : Fin 3) ∈ spanCols (G9 K) Finset.univ :=
    Submodule.subset_span ⟨0, Finset.mem_univ _, by simp [G9]⟩
  have h1 : stdBasis (K := K) (1 : Fin 3) ∈ spanCols (G9 K) Finset.univ :=
    Submodule.subset_span ⟨1, Finset.mem_univ _, by simp [G9]⟩
  have h2 : stdBasis (K := K) (2 : Fin 3) ∈ spanCols (G9 K) Finset.univ :=
    Submodule.subset_span ⟨2, Finset.mem_univ _, by simp [G9]⟩
  exact Submodule.add_mem _ (Submodule.add_mem _ (Submodule.smul_mem _ _ h0)
    (Submodule.smul_mem _ _ h1)) (Submodule.smul_mem _ _ h2)

lemma G9_rec_F1_iff (K : Type*) [Field K] (S : Finset (Fin 9)) :
    fileF1 K ≤ spanCols (G9 K) S ↔ (0 : Fin 9) ∈ S ∧ (1 : Fin 9) ∈ S := by
  constructor
  · intro hle
    rw [fileF1_le_iff] at hle
    constructor
    · by_contra h0
      have hcoord0 : ∀ v ∈ spanCols (G9 K) S, v 0 = 0 := by
        intro v hv
        refine Submodule.span_induction ?_ ?_ ?_ ?_ hv
        · rintro _ ⟨i, hi, rfl⟩
          have hne0 : i ≠ 0 := fun h => h0 (h ▸ hi)
          by_cases h1 : i = 1 <;> simp [G9, hne0, h1, stdBasis]
        · simp
        · intro x y _ _ hx hy; simp [hx, hy]
        · intro c x _ hx; simp [hx]
      have := hcoord0 _ hle.1
      simp [stdBasis] at this
    · by_contra h1
      have hcoord1 : ∀ v ∈ spanCols (G9 K) S, v 1 = 0 := by
        intro v hv
        refine Submodule.span_induction ?_ ?_ ?_ ?_ hv
        · rintro _ ⟨i, hi, rfl⟩
          have hne1 : i ≠ 1 := fun h => h1 (h ▸ hi)
          by_cases h0 : i = 0 <;> simp [G9, h0, hne1, stdBasis]
        · simp
        · intro x y _ _ hx hy; simp [hx, hy]
        · intro c x _ hx; simp [hx]
      have := hcoord1 _ hle.2
      simp [stdBasis] at this
  · rintro ⟨h0, h1⟩
    rw [fileF1_le_iff]
    exact ⟨Submodule.subset_span ⟨0, h0, by simp [G9]⟩,
      Submodule.subset_span ⟨1, h1, by simp [G9]⟩⟩

lemma G9_rec_F2_iff (K : Type*) [Field K] (S : Finset (Fin 9)) :
    fileF2 K ≤ spanCols (G9 K) S ↔
      (S ∩ (Finset.univ.filter (fun i : Fin 9 => i ≠ 0 ∧ i ≠ 1))).Nonempty := by
  constructor
  · intro hle
    rw [fileF2_le_iff] at hle
    by_contra hempty
    have hcoord2 : ∀ v ∈ spanCols (G9 K) S, v 2 = 0 := by
      intro v hv
      refine Submodule.span_induction ?_ ?_ ?_ ?_ hv
      · rintro _ ⟨i, hi, rfl⟩
        by_cases h0 : i = 0
        · simp [G9, h0, stdBasis]
        · by_cases h1 : i = 1
          · simp [G9, h1, stdBasis]
          · exact absurd ⟨i, Finset.mem_inter.mpr ⟨hi,
              Finset.mem_filter.mpr ⟨Finset.mem_univ i, h0, h1⟩⟩⟩ hempty
      · simp
      · intro x y _ _ hx hy; simp [hx, hy]
      · intro c x _ hx; simp [hx]
    have := hcoord2 _ hle
    simp [stdBasis] at this
  · rintro ⟨i, hi⟩
    rcases Finset.mem_inter.mp hi with ⟨hiS, hiF⟩
    rcases (Finset.mem_filter.mp hiF).2 with ⟨h0, h1⟩
    rw [fileF2_le_iff]
    exact Submodule.subset_span ⟨i, hiS, by simp [G9, h0, h1]⟩

/-- Exact expectations for `G9`: `E₁(G9) = 27 / 2` and `E₂(G9) = 9 / 7` over EVERY field `K`. -/
theorem G9_expectations (K : Type*) [Field K] :
    haveI : DecidablePred (fun S : Finset (Fin 9) => fileF1 K ≤ spanCols (G9 K) S) :=
      Classical.decPred _
    haveI : DecidablePred (fun S : Finset (Fin 9) => fileF2 K ≤ spanCols (G9 K) S) :=
      Classical.decPred _
    expectedWait (fun S => fileF1 K ≤ spanCols (G9 K) S)
      (by dsimp only; rw [G9_full_rank K]; exact le_top) ∅ = 27 / 2 ∧
    expectedWait (fun S => fileF2 K ≤ spanCols (G9 K) S)
      (by dsimp only; rw [G9_full_rank K]; exact le_top) ∅ = 9 / 7 := by
  classical
  constructor
  · have hEq := expectedWait_unique
      (fun S : Finset (Fin 9) => fileF1 K ≤ spanCols (G9 K) S)
      (by dsimp only; rw [G9_full_rank K]; exact le_top)
      (expectedWait (fun S : Finset (Fin 9) => (0 : Fin 9) ∈ S ∧ (1 : Fin 9) ∈ S)
        ⟨Finset.mem_univ _, Finset.mem_univ _⟩)
      (fun S hS => expectedWait_of_rec _ _ ((G9_rec_F1_iff K S).mp hS))
      (fun S hS => expectedWait_bellman _ _ (fun h => hS ((G9_rec_F1_iff K S).mpr h)))
      ∅
    rw [← hEq, expectedWait_pair (by decide : (0 : Fin 9) ≠ 1)]
    norm_num
  · let Good : Finset (Fin 9) := Finset.univ.filter (fun i : Fin 9 => i ≠ 0 ∧ i ≠ 1)
    have hGood_card : Good.card = 7 := by decide
    have hGood_ne : Good.Nonempty := Finset.card_pos.mp (by omega)
    have hEq := expectedWait_unique
      (fun S : Finset (Fin 9) => fileF2 K ≤ spanCols (G9 K) S)
      (by dsimp only; rw [G9_full_rank K]; exact le_top)
      (expectedWait (fun S : Finset (Fin 9) => (S ∩ Good).Nonempty) (by simpa using hGood_ne))
      (fun S hS => expectedWait_of_rec _ _ ((G9_rec_F2_iff K S).mp hS))
      (fun S hS => expectedWait_bellman _ _ (fun h => hS ((G9_rec_F2_iff K S).mpr h)))
      ∅
    rw [← hEq, expectedWait_hit Good hGood_ne, hGood_card]
    norm_num

/-- Projection `π : K³ → K²` onto the first two coordinates (isomorphic to `K³ / F₂`). -/
def proj12 (K : Type*) [Field K] : (Fin 3 → K) →ₗ[K] (Fin 2 → K) where
  toFun v := ![v 0, v 1]
  map_add' x y := by ext i; fin_cases i <;> simp
  map_smul' c x := by ext i; fin_cases i <;> simp

lemma proj12_surjective (K : Type*) [Field K] : Function.Surjective (proj12 K) := by
  intro y
  refine ⟨![y 0, y 1, 0], ?_⟩
  ext i
  fin_cases i <;> simp [proj12]

lemma proj12_eq_zero_of_fileF2_le_span_singleton {K : Type*} [Field K] {v : Fin 3 → K}
    (hv : fileF2 K ≤ Submodule.span K ({v} : Set (Fin 3 → K))) :
    proj12 K v = 0 := by
  rw [fileF2_le_iff, Submodule.mem_span_singleton] at hv
  rcases hv with ⟨c, hc⟩
  have hc_ne : c ≠ 0 := by
    intro h0
    rw [h0, zero_smul] at hc
    have := congr_fun hc 2
    simp [stdBasis] at this
  ext i
  fin_cases i
  · have h0 := congr_fun hc 0
    simp only [Pi.smul_apply, smul_eq_mul, stdBasis, Pi.single_apply, Fin.reduceEq, ↓reduceIte] at h0
    have hv0 : v 0 = 0 := (mul_eq_zero.mp h0).resolve_left hc_ne
    simp [proj12, hv0]
  · have h1 := congr_fun hc 1
    simp only [Pi.smul_apply, smul_eq_mul, stdBasis, Pi.single_apply, Fin.reduceEq, ↓reduceIte] at h1
    have hv1 : v 1 = 0 := (mul_eq_zero.mp h1).resolve_left hc_ne
    simp [proj12, hv1]

/-- **Refutation of `conj:monotone` over EVERY field `K`**:
for ANY full-rank length-10 matrix `G' : Fin 10 → (Fin 3 → K)` with `E₂(G') ≤ 9 / 7`,
we necessarily have `E₁(G') ≥ 15 > 27 / 2 = E₁(G9)`. -/
theorem refute_conj_monotone (K : Type*) [Field K]
    (G' : Fin 10 → (Fin 3 → K)) (hRank : spanCols G' Finset.univ = ⊤) :
    haveI : DecidablePred (fun S : Finset (Fin 10) => fileF1 K ≤ spanCols G' S) :=
      Classical.decPred _
    haveI : DecidablePred (fun S : Finset (Fin 10) => fileF2 K ≤ spanCols G' S) :=
      Classical.decPred _
    expectedWait (fun S => fileF2 K ≤ spanCols G' S)
      (by dsimp only; rw [hRank]; exact le_top) ∅ ≤ 9 / 7 →
    (15 : ℚ) ≤ expectedWait (fun S => fileF1 K ≤ spanCols G' S)
      (by dsimp only; rw [hRank]; exact le_top) ∅ := by
  classical
  intro hE2
  have hU1 : (fun S : Finset (Fin 10) => fileF1 K ≤ spanCols G' S) Finset.univ := by
    dsimp only; rw [hRank]; exact le_top
  have hU2 : (fun S : Finset (Fin 10) => fileF2 K ≤ spanCols G' S) Finset.univ := by
    dsimp only; rw [hRank]; exact le_top
  let bad : Finset (Fin 10) := Finset.univ.filter (fun i => ¬ (fileF2 K ≤ spanCols G' {i}))
  -- Step 1: `E₂(G') ≥ 1 + bad.card / 10`, hence `bad.card ≤ 2`.
  have hnot_empty : ¬ (fileF2 K ≤ spanCols G' ∅) := by
    intro hle
    rw [fileF2_le_iff] at hle
    have hbot : spanCols G' ∅ = ⊥ := by simp [spanCols]
    rw [hbot, Submodule.mem_bot] at hle
    have := congr_fun hle 2
    simp [stdBasis] at this
  have hE2_lb : 1 + (bad.card : ℚ) / 10 ≤
      expectedWait (fun S => fileF2 K ≤ spanCols G' S) hU2 ∅ := by
    rw [expectedWait_bellman _ hU2 hnot_empty, Fintype.card_fin]
    have hsum_lb : (bad.card : ℚ) ≤
        ∑ i : Fin 10, expectedWait (fun S => fileF2 K ≤ spanCols G' S) hU2 (insert i ∅) := by
      calc (bad.card : ℚ)
          = ∑ _i ∈ bad, (1 : ℚ) := by simp
        _ ≤ ∑ i ∈ bad, expectedWait (fun S => fileF2 K ≤ spanCols G' S) hU2 {i} := by
            refine Finset.sum_le_sum fun i hi => ?_
            exact expectedWait_ge_one _ hU2 (Finset.mem_filter.mp hi).2
        _ ≤ ∑ i : Fin 10, expectedWait (fun S => fileF2 K ≤ spanCols G' S) hU2 {i} :=
            Finset.sum_le_univ_sum_of_nonneg fun i => expectedWait_nonneg _ hU2 _
    have hcast : ((10 : ℕ) : ℚ) = 10 := by norm_num
    rw [hcast]
    linarith only [hsum_lb]
  have hbad_le2 : bad.card ≤ 2 := by
    have hq : (bad.card : ℚ) ≤ 20 / 7 := by linarith only [hE2_lb, hE2]
    have hnat : (bad.card : ℚ) < 3 := by linarith only [hq]
    have hnat' : bad.card < 3 := by exact_mod_cast hnat
    omega
  -- Step 2: every `i ∉ bad` has `proj12 K (G' i) = 0`.
  have hproj_zero : ∀ i : Fin 10, i ∉ bad → proj12 K (G' i) = 0 := by
    intro i hi
    have hrec : fileF2 K ≤ spanCols G' {i} := by
      by_contra hnot
      exact hi (Finset.mem_filter.mpr ⟨Finset.mem_univ i, hnot⟩)
    apply proj12_eq_zero_of_fileF2_le_span_singleton
    simpa [spanCols] using hrec
  -- Helper: any subset `S` whose `proj12` image spans `⊤ : Submodule K (Fin 2 → K)` satisfies `2 ≤ (S ∩ bad).card`
  have hspan_proj_card : ∀ S : Finset (Fin 10),
      Submodule.map (proj12 K) (spanCols G' S) = ⊤ → 2 ≤ (S ∩ bad).card := by
    intro S hmap
    have hspan_sub : (⊤ : Submodule K (Fin 2 → K)) ≤
        Submodule.span K ((fun i => proj12 K (G' i)) '' ((S ∩ bad : Finset (Fin 10)) : Set (Fin 10))) := by
      rw [← hmap, spanCols, Submodule.map_span, Submodule.span_le]
      rintro _ ⟨_, ⟨i, hi, rfl⟩, rfl⟩
      by_cases hib : i ∈ bad
      · exact Submodule.subset_span ⟨i, Finset.mem_inter.mpr ⟨hi, hib⟩, rfl⟩
      · rw [hproj_zero i hib]
        exact Submodule.zero_mem _
    have htop_eq : Submodule.span K ((fun i => proj12 K (G' i)) '' ((S ∩ bad : Finset (Fin 10)) : Set (Fin 10))) = ⊤ :=
      top_le_iff.mp hspan_sub
    have hfr : Module.finrank K
        (Submodule.span K ((fun i => proj12 K (G' i)) '' ((S ∩ bad : Finset (Fin 10)) : Set (Fin 10)))) = 2 := by
      rw [htop_eq, finrank_top, Module.finrank_fin_fun]
    have hle_card : Module.finrank K
        (Submodule.span K ((fun i => proj12 K (G' i)) '' ((S ∩ bad : Finset (Fin 10)) : Set (Fin 10)))) ≤
        ((S ∩ bad).image (fun i => proj12 K (G' i))).card := by
      have himg : ((fun i => proj12 K (G' i)) '' ((S ∩ bad : Finset (Fin 10)) : Set (Fin 10))) =
          (((S ∩ bad).image (fun i => proj12 K (G' i)) : Finset (Fin 2 → K)) : Set (Fin 2 → K)) := by
        simp [Finset.coe_image]
      rw [himg]
      simpa using finrank_span_le_card (((S ∩ bad).image (fun i => proj12 K (G' i)) : Finset (Fin 2 → K)) : Set (Fin 2 → K))
    have hle_inter : ((S ∩ bad).image (fun i => proj12 K (G' i))).card ≤ (S ∩ bad).card :=
      Finset.card_image_le
    omega
  -- Step 3: `bad.card = 2`, so `bad = {a, b}` with `a ≠ b`.
  have hbad_ge2 : 2 ≤ bad.card := by
    have hmap_univ : Submodule.map (proj12 K) (spanCols G' Finset.univ) = ⊤ := by
      rw [hRank, Submodule.map_top, LinearMap.range_eq_top.mpr (proj12_surjective K)]
    have := hspan_proj_card Finset.univ hmap_univ
    simpa using this
  have hbad_eq2 : bad.card = 2 := by omega
  rcases Finset.card_eq_two.mp hbad_eq2 with ⟨a, b, hab, hbad_pair⟩
  -- Step 4: Every `F₁`-recovering subset `S` must contain both `a` and `b`.
  have himp : ∀ S : Finset (Fin 10), fileF1 K ≤ spanCols G' S → a ∈ S ∧ b ∈ S := by
    intro S hle
    have hmapF1 : Submodule.map (proj12 K) (fileF1 K) = ⊤ := by
      rw [eq_top_iff]
      intro y _
      refine ⟨![y 0, y 1, 0], ?_, ?_⟩
      · exact (mem_fileF1_iff (![y 0, y 1, 0] : Fin 3 → K)).mpr rfl
      · ext i; fin_cases i <;> simp [proj12]
    have hmapS : Submodule.map (proj12 K) (spanCols G' S) = ⊤ :=
      top_le_iff.mp (hmapF1 ▸ Submodule.map_mono hle)
    have hcard_inter : 2 ≤ (S ∩ bad).card := hspan_proj_card S hmapS
    have hsub_eq : S ∩ bad = bad :=
      Finset.eq_of_subset_of_card_le Finset.inter_subset_right (by omega)
    have hbad_sub : bad ⊆ S := Finset.inter_eq_right.mp hsub_eq
    rw [hbad_pair] at hbad_sub
    exact ⟨hbad_sub (by simp), hbad_sub (by simp)⟩
  -- Step 5: Monotonicity + `expectedWait_pair` gives `E₁(G') ≥ 15`.
  have hmono := expectedWait_mono
    (fun S : Finset (Fin 10) => fileF1 K ≤ spanCols G' S)
    (fun S : Finset (Fin 10) => a ∈ S ∧ b ∈ S)
    hU1 ⟨Finset.mem_univ a, Finset.mem_univ b⟩ himp ∅
  rw [expectedWait_pair hab] at hmono
  have h15 : (3 * (10 : ℚ) / 2 : ℚ) = 15 := by norm_num
  simpa [h15] using hmono

end CodedBlockRetrieval
