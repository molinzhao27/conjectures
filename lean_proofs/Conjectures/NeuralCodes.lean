import Mathlib

/-!
# Formalization of Core Results on Geraci–Kunin–Seceleanu (arXiv:2603.10837)
"Adjoints of Morphisms of Neural Codes"

We formalize in Lean 4 + Mathlib (zero `sorry`):
1. Exact definitions of combinatorial neural codes / binary matrices `C : Matrix (Fin m) (Fin n) Bool`,
   Boolean matrix multiplication `boolMul`, Boolean rank `brank`, trunks, redundancy, simple vs
   non-simple redundancy, and intersection-completeness.
2. **Minimal Clean Counterexample (`refutation_clean_3x4`)**: Explicit `3 × 4` clean matrix `C_min`
   with explicit reduced submatrix `rho_min = J₃ - I₃`, columns 0–2 non-redundant, column 3 redundant
   and not simply redundant, and `brank(rho_min) = brank(C_min) = 3` (whereas `3 + 1 = 4`).
3. **Exponential-Gap Infinite Family (`exponential_gap_family`, Theorem 2.3)**: For EVERY `m ≥ 2`,
   explicit `(m + 1) × (2 * m)` base matrix `B_exp m` and `(m + 1) × (2^m + m - 1)` extension
   `C_exp m` with no trivial columns, all `2 * m` base columns non-redundant, all `2^m - m - 1`
   appended columns non-simply redundant, and `brank(B_exp m) = brank(C_exp m) = m + 1` (while
   `brank(B_exp m) + ℓ = (m + 1) + (2^m - m - 1) = 2^m`).
4. **General All-1s Column Theorem (`brank_append_allOnes`, Proposition 2.5)**: For EVERY `m × n`
   binary matrix `M`, appending an all-1s column increases `brank` by `1` if `M` has an all-zero
   row and leaves `brank` unchanged if `M` has no all-zero row — refuting Section 7.3 line 1112
   in full generality.
5. **All-1s Column Counterexample (`refutation_all1_reverse`)**:
   For `C_all1 = [[0,1],[1,1]]` and its reduced submatrix `rho_all1 = [[0],[1]]`, column 1 is simply
   redundant, yet `brank(C_all1) = 2 > 1 = brank(rho_all1)`.
6. **Duplicated Non-Simple Redundancy (`refutation_dup_reverse`)**:
   Explicit `4 × 4` matrix `C_dup` and submatrix `rho_dup`, where columns 2 and 3 are each simply
   redundant to the other, with `brank(C_dup) = 3 > 2 = brank(rho_dup)`. (Note: as flagged in review,
   this violates the conjecture under the simultaneous definition of `ℓ`, not under sequential
   deduplication-first counting.)
7. **Lower-Triangular Boolean Rank (`brank_lower_triangular`)**: Every `u × u` lower-triangular
   Boolean matrix with `1`s on the diagonal has `brank = u`.
8. **Positive Theorem for ALL Intersection-Complete Codes (`brank_intersection_complete`)**:
   For EVERY intersection-complete binary matrix `C : Matrix (Fin m) (Fin n) Bool`,
   `brank(C)` equals the number of distinct nonzero columns of `C`!
9. **Reducedness Verification for Question 2 (`question2_sharpness`)**:
   For EVERY `n ≥ 3`, the explicit `n`-neuron code `C_q2 n` and the explicit `(2n - 2)`-neuron code
   `D_q2 n` (representing the covering code after deleting the two trivial neurons) are both reduced.

Note: The general reduction operator `ρ`, Proposition 2.4, Theorem 3.1, minimality of `3 × 4`,
and the formal link from `def:excessivecovering` to `D_q2 n` are proved in prose in
`solutions/neural_codes/solution.md`.
-/

open Finset

namespace NeuralCodesBMF

/-- Boolean matrix multiplication `(V ⊙ H)_{i,j} = ∃ t, V i t = true ∧ H t j = true`. -/
def boolMul {m r n : ℕ} (V : Matrix (Fin m) (Fin r) Bool)
    (H : Matrix (Fin r) (Fin n) Bool) : Matrix (Fin m) (Fin n) Bool :=
  fun i j => decide (∃ t : Fin r, V i t = true ∧ H t j = true)

/-- `HasBoolRank C r` holds when `C = V ⊙ H` for some `V : m × r`, `H : r × n`. -/
def HasBoolRank {m n : ℕ} (C : Matrix (Fin m) (Fin n) Bool) (r : ℕ) : Prop :=
  ∃ (V : Matrix (Fin m) (Fin r) Bool) (H : Matrix (Fin r) (Fin n) Bool), boolMul V H = C

/-- Every `m × n` Boolean matrix has a rank-`m` factorization `I_m ⊙ C = C`. -/
lemma hasBoolRank_rows {m n : ℕ} (C : Matrix (Fin m) (Fin n) Bool) : HasBoolRank C m := by
  refine ⟨fun i t => decide (i = t), C, ?_⟩
  ext i j
  simp only [boolMul, decide_eq_true_eq]
  cases hC : C i j
  · simp [hC]
  · simp [hC]

/-- Every `m × n` Boolean matrix has a rank-`n` factorization `C ⊙ I_n = C`. -/
lemma hasBoolRank_cols {m n : ℕ} (C : Matrix (Fin m) (Fin n) Bool) : HasBoolRank C n := by
  refine ⟨C, fun t j => decide (t = j), ?_⟩
  ext i j
  simp only [boolMul, decide_eq_true_eq]
  cases hC : C i j
  · simp [hC]
  · simp [hC]

/-- Exact Boolean rank `brank C := Nat.find ...`, the minimum `r` such that `HasBoolRank C r`. -/
noncomputable def brank {m n : ℕ} (C : Matrix (Fin m) (Fin n) Bool) : ℕ := by
  classical
  exact Nat.find ⟨m, hasBoolRank_rows C⟩

lemma brank_le {m n : ℕ} {C : Matrix (Fin m) (Fin n) Bool} {r : ℕ}
    (h : HasBoolRank C r) : brank C ≤ r := by
  classical
  unfold brank
  exact Nat.find_min' _ h

lemma le_brank {m n : ℕ} {C : Matrix (Fin m) (Fin n) Bool} {r : ℕ}
    (h : ∀ k, HasBoolRank C k → r ≤ k) : r ≤ brank C := by
  classical
  unfold brank
  exact h _ (Nat.find_spec ⟨m, hasBoolRank_rows C⟩)

lemma brank_eq {m n : ℕ} {C : Matrix (Fin m) (Fin n) Bool} {r : ℕ}
    (h1 : HasBoolRank C r) (h2 : ∀ k, HasBoolRank C k → r ≤ k) : brank C = r :=
  le_antisymm (brank_le h1) (le_brank h2)

lemma hasBoolRank_brank {m n : ℕ} (C : Matrix (Fin m) (Fin n) Bool) :
    HasBoolRank C (brank C) := by
  classical
  unfold brank
  exact Nat.find_spec ⟨m, hasBoolRank_rows C⟩

/-- Deleting/reindexing columns cannot increase Boolean rank. -/
lemma hasBoolRank_reindex_cols {m n₁ n₂ r : ℕ} {C : Matrix (Fin m) (Fin n₁) Bool}
    (f : Fin n₂ → Fin n₁) (h : HasBoolRank C r) :
    HasBoolRank (fun i j => C i (f j)) r := by
  rcases h with ⟨V, H, hVH⟩
  refine ⟨V, fun t j => H t (f j), ?_⟩
  ext i j
  have := congrFun (congrFun hVH i) (f j)
  simpa [boolMul] using this

/-- Extracting a submatrix on rows `rf` and columns `cf` cannot increase Boolean rank. -/
lemma hasBoolRank_submatrix {m₁ n₁ m₂ n₂ r : ℕ} {C : Matrix (Fin m₁) (Fin n₁) Bool}
    (rf : Fin m₂ → Fin m₁) (cf : Fin n₂ → Fin n₁) (h : HasBoolRank C r) :
    HasBoolRank (fun i j => C (rf i) (cf j)) r := by
  rcases h with ⟨V, H, hVH⟩
  refine ⟨fun i t => V (rf i) t, fun t j => H t (cf j), ?_⟩
  ext i j
  have := congrFun (congrFun hVH (rf i)) (cf j)
  simpa [boolMul] using this

/-! ### Exact Redundancy Definitions from arXiv:2603.10837 -/

/-- Trunk of a subset `σ : Finset (Fin n)` of neurons in `C`: rows where all `j ∈ σ` are `true`. -/
def trunk {m n : ℕ} (C : Matrix (Fin m) (Fin n) Bool) (σ : Finset (Fin n)) : Finset (Fin m) :=
  Finset.univ.filter fun i => ∀ j ∈ σ, C i j = true

/-- Neuron `j` is trivial iff its column is all `false` (`trunk C {j} = ∅`). -/
abbrev IsTrivial {m n : ℕ} (C : Matrix (Fin m) (Fin n) Bool) (j : Fin n) : Prop :=
  trunk C {j} = ∅

/-- Neuron `j` is redundant to `σ ⊆ [n] \ {j}` iff `trunk C {j} = trunk C σ`. -/
abbrev IsRedundantTo {m n : ℕ} (C : Matrix (Fin m) (Fin n) Bool) (j : Fin n)
    (σ : Finset (Fin n)) : Prop :=
  j ∉ σ ∧ trunk C {j} = trunk C σ

/-- Neuron `j` is redundant iff it is redundant to some `σ ⊆ [n] \ {j}`. -/
def IsRedundant {m n : ℕ} (C : Matrix (Fin m) (Fin n) Bool) (j : Fin n) : Prop :=
  ∃ σ : Finset (Fin n), IsRedundantTo C j σ

/-- Neuron `j` is simply redundant iff it is redundant to `∅` (all-1s) or a singleton `{k}` (`k ≠ j`). -/
def IsSimplyRedundant {m n : ℕ} (C : Matrix (Fin m) (Fin n) Bool) (j : Fin n) : Prop :=
  IsRedundantTo C j ∅ ∨ ∃ k : Fin n, IsRedundantTo C j {k}

/-- Neuron `j` is non-simply redundant iff it is redundant and not simply redundant. -/
def IsNonSimplyRedundant {m n : ℕ} (C : Matrix (Fin m) (Fin n) Bool) (j : Fin n) : Prop :=
  IsRedundant C j ∧ ¬ IsSimplyRedundant C j

/-- A code matrix `C` is reduced iff it has no trivial and no redundant neurons. -/
def IsReduced {m n : ℕ} (C : Matrix (Fin m) (Fin n) Bool) : Prop :=
  (∀ j, ¬ IsTrivial C j) ∧ (∀ j, ¬ IsRedundant C j)

/-- A code matrix `C` is intersection-complete iff the entrywise AND of any two rows is a row. -/
def IsIntersectionComplete {m n : ℕ} (C : Matrix (Fin m) (Fin n) Bool) : Prop :=
  ∀ i₁ i₂ : Fin m, ∃ i₃ : Fin m, ∀ j : Fin n, C i₃ j = (C i₁ j && C i₂ j)

/-! ### Lemma 3.3: Every `u × u` Lower-Triangular Matrix with 1s on Diagonal has `brank = u` -/

theorem brank_lower_triangular {u : ℕ} (M : Matrix (Fin u) (Fin u) Bool)
    (hdiag : ∀ i : Fin u, M i i = true)
    (hzero : ∀ i j : Fin u, i < j → M i j = false) :
    brank M = u := by
  refine brank_eq (hasBoolRank_rows M) ?_
  intro r ⟨V, H, hVH⟩
  -- For each i, pick f i : Fin r with V i (f i) = true and H (f i) i = true.
  have hex : ∀ i : Fin u, ∃ t : Fin r, V i t = true ∧ H t i = true := by
    intro i
    have hi := congrFun (congrFun hVH i) i
    rw [hdiag i] at hi
    simpa [boolMul] using hi
  choose f hfV hfH using hex
  -- Show f : Fin u → Fin r is injective!
  have hinj : Function.Injective f := by
    intro i₁ i₂ heq
    by_contra hne
    rcases lt_or_gt_of_ne hne with hlt | hgt
    · -- i₁ < i₂, but V i₁ (f i₁) = true and H (f i₁) i₂ = H (f i₂) i₂ = true ⇒ M i₁ i₂ = true!
      have hm := congrFun (congrFun hVH i₁) i₂
      rw [hzero i₁ i₂ hlt] at hm
      have htrue : boolMul V H i₁ i₂ = true := by
        simp only [boolMul, decide_eq_true_eq]
        exact ⟨f i₁, hfV i₁, heq ▸ hfH i₂⟩
      rw [htrue] at hm; cases hm
    · have hm := congrFun (congrFun hVH i₂) i₁
      rw [hzero i₂ i₁ hgt] at hm
      have htrue : boolMul V H i₂ i₁ = true := by
        simp only [boolMul, decide_eq_true_eq]
        exact ⟨f i₂, hfV i₂, heq.symm ▸ hfH i₁⟩
      rw [htrue] at hm; cases hm
  have hcard := Fintype.card_le_of_injective f hinj
  simpa using hcard

/-! ### Theorem 2.1: Provably Minimal `3 × 4` Clean Counterexample -/

/-- The reduced `3 × 3` matrix `J₃ - I₃`. -/
def rho_min : Matrix (Fin 3) (Fin 3) Bool :=
  fun i j => decide (i ≠ j)

/-- The `3 × 4` clean counterexample `C_min`: columns `0, 1, 2` are `J₃ - I₃`,
column `3` is `col 0 ∧ col 1 = (0, 0, 1)ᵀ`. -/
def C_min : Matrix (Fin 3) (Fin 4) Bool :=
  ![ ![false, true, true, false],
     ![true, false, true, false],
     ![true, true, false, true] ]

lemma brank_rho_min : brank rho_min = 3 := by
  refine brank_eq (hasBoolRank_rows rho_min) ?_
  intro r ⟨V, H, hVH⟩
  by_contra hlt
  push_neg at hlt
  -- r ≤ 2: every row of rho_min is determined by V i : Fin r → Bool
  -- Actually even simpler: for r ≤ 2, Fin r has ≤ 2 elements, so consider the 3 rows of V!
  -- Each row i of rho_min has 2 true entries and 1 false entry (at i), so V i cannot be all-false,
  -- and no two rows of V can satisfy V i₁ ≤ V i₂ (since row i₁ ≰ row i₂ for i₁ ≠ i₂).
  have hnotle : ∀ i₁ i₂ : Fin 3, i₁ ≠ i₂ → ∃ t : Fin r, V i₁ t = true ∧ V i₂ t = false := by
    intro i₁ i₂ hne
    -- rho_min i₁ i₂ = true, rho_min i₂ i₂ = false
    have h1 : boolMul V H i₁ i₂ = true := by
      rw [hVH]; simp [rho_min, hne]
    have h2 : boolMul V H i₂ i₂ = false := by
      rw [hVH]; simp [rho_min]
    unfold boolMul at h1 h2
    rw [decide_eq_true_eq] at h1
    rcases h1 with ⟨t, ht1, htH⟩
    refine ⟨t, ht1, ?_⟩
    cases hvt2 : V i₂ t
    · rfl
    · exfalso
      rw [decide_eq_false_iff_not] at h2
      exact h2 ⟨t, hvt2, htH⟩
  -- Now r ≤ 2 implies the subsets S_i = {t | V i t = true} of Fin r form an antichain of size 3!
  let S : Fin 3 → Finset (Fin r) := fun i => Finset.univ.filter (fun t => V i t = true)
  have hanti : ∀ i₁ i₂ : Fin 3, i₁ ≠ i₂ → ¬ (S i₁ ⊆ S i₂) := by
    intro i₁ i₂ hne hsub
    obtain ⟨t, ht1, ht2⟩ := hnotle i₁ i₂ hne
    have : t ∈ S i₂ := hsub (by simp [S, ht1])
    simp [S, ht2] at this
  -- Every S i is nonempty (otherwise S i ⊆ S j) and proper (otherwise S j ⊆ S i), so card (S i) ∈ [1, r-1].
  -- Since r ≤ 2, card (S i) = 1 for all i : Fin 3, giving 3 distinct singletons in Fin r, impossible!
  have hcard1 : ∀ i : Fin 3, (S i).card = 1 := by
    intro i
    -- Pick j ≠ i
    let j : Fin 3 := if i = 0 then 1 else 0
    have hij : i ≠ j := by fin_cases i <;> decide
    have hne_emp : (S i).Nonempty := by
      by_contra he
      rw [Finset.not_nonempty_iff_eq_empty] at he
      exact hanti i j hij (by rw [he]; exact Finset.empty_subset _)
    have hne_univ : S i ≠ Finset.univ := by
      intro hu
      exact hanti j i hij.symm (by rw [hu]; exact Finset.subset_univ _)
    have h1 : 1 ≤ (S i).card := Finset.one_le_card.mpr hne_emp
    have h2 : (S i).card < r := by
      have := Finset.card_lt_card (Finset.ssubset_univ_iff.mpr hne_univ)
      simpa using this
    omega
  -- Extract the unique element of S i; gives injection Fin 3 → Fin r, contradicting r < 3!
  have hexu : ∀ i : Fin 3, ∃ t : Fin r, S i = {t} := fun i => Finset.card_eq_one.mp (hcard1 i)
  choose g hg using hexu
  have hginj : Function.Injective g := by
    intro i₁ i₂ heq
    by_contra hne
    exact hanti i₁ i₂ hne (by rw [hg i₁, hg i₂, heq])
  have := Fintype.card_le_of_injective g hginj
  simp at this
  omega

/-- **Theorem 2.1 (Refutation of Section 7.3 Conjecture, `brank(C) < brank(ρ(C)) + ℓ`)**:
`C_min` is a clean `3 × 4` binary matrix with:
- no trivial neurons,
- columns `0, 1, 2` non-redundant (forming `ρ(C_min) = rho_min`),
- column `3` is the unique redundant neuron and is NOT simply redundant (`k = ℓ = 1`),
- `brank(rho_min) = 3` and `brank(C_min) = 3 < 3 + 1`. -/
theorem refutation_clean_3x4 :
    (∀ j : Fin 4, ¬ IsTrivial C_min j) ∧
    (∀ j : Fin 3, ¬ IsRedundant C_min j.castSucc) ∧
    IsNonSimplyRedundant C_min 3 ∧
    brank rho_min = 3 ∧
    brank C_min = 3 := by
  refine ⟨?_, ?_, ?_, brank_rho_min, ?_⟩
  · intro j; fin_cases j <;> decide
  · intro j; fin_cases j <;>
    · rintro ⟨σ, hnot, heq⟩
      revert σ hnot heq
      decide
  · refine ⟨⟨{0, 1}, by decide⟩, ?_⟩
    rintro (h0 | ⟨k, hk⟩)
    · revert h0; decide
    · revert k hk; decide
  · refine brank_eq (hasBoolRank_rows C_min) ?_
    intro r hr
    have hsub : HasBoolRank rho_min r := by
      have h := hasBoolRank_reindex_cols (fun j : Fin 3 => j.castSucc) hr
      have heq : (fun i (j : Fin 3) => C_min i j.castSucc) = rho_min := by decide
      rwa [heq] at h
    have := brank_le hsub
    rw [brank_rho_min] at this
    exact this

/-! ### Theorem 2.3: Exponential-Gap Infinite Family (`brank(B_exp m) = brank(C_exp m) = m + 1`) -/

/-- Subsets of `Fin m` of cardinality at least `2`. -/
def BigSubsets (m : ℕ) : Finset (Finset (Fin m)) :=
  Finset.univ.filter (fun S : Finset (Fin m) => 2 ≤ S.card)

lemma card_bigSubsets_add (m : ℕ) : (BigSubsets m).card + m + 1 = 2 ^ m := by
  let A₀ : Finset (Finset (Fin m)) := {∅}
  let A₁ : Finset (Finset (Fin m)) := Finset.univ.image (fun a : Fin m => ({a} : Finset (Fin m)))
  let A₂ : Finset (Finset (Fin m)) := BigSubsets m
  have hdisj01 : Disjoint A₀ A₁ := by
    rw [Finset.disjoint_left]
    intro S hS0 hS1
    rw [Finset.mem_singleton] at hS0
    rcases Finset.mem_image.mp hS1 with ⟨a, _, ha⟩
    rw [hS0] at ha
    exact Finset.notMem_empty a (ha ▸ Finset.mem_singleton_self a)
  have hdisj012 : Disjoint (A₀ ∪ A₁) A₂ := by
    rw [Finset.disjoint_left]
    intro S hS01 hS2
    have h2 : 2 ≤ S.card := (Finset.mem_filter.mp hS2).2
    rcases Finset.mem_union.mp hS01 with hS0 | hS1
    · rw [Finset.mem_singleton] at hS0
      subst hS0
      simp at h2
    · rcases Finset.mem_image.mp hS1 with ⟨a, _, rfl⟩
      simp at h2
  have hunion : A₀ ∪ A₁ ∪ A₂ = Finset.univ := by
    ext S
    simp only [Finset.mem_univ, iff_true, Finset.mem_union]
    by_cases h0 : S.card = 0
    · left; left
      exact Finset.mem_singleton.mpr (Finset.card_eq_zero.mp h0)
    · by_cases h1 : S.card = 1
      · left; right
        obtain ⟨a, ha⟩ := Finset.card_eq_one.mp h1
        exact Finset.mem_image.mpr ⟨a, Finset.mem_univ a, ha.symm⟩
      · right
        exact Finset.mem_filter.mpr ⟨Finset.mem_univ S, by omega⟩
  have hA0 : A₀.card = 1 := Finset.card_singleton ∅
  have hA1 : A₁.card = m := by
    rw [Finset.card_image_of_injective _ Finset.singleton_injective, Finset.card_univ, Fintype.card_fin]
  have huniv : (Finset.univ : Finset (Finset (Fin m))).card = 2 ^ m := by
    rw [Finset.card_univ, Fintype.card_finset, Fintype.card_fin]
  have htot : (A₀ ∪ A₁ ∪ A₂).card = A₀.card + A₁.card + A₂.card := by
    rw [Finset.card_union_of_disjoint hdisj012, Finset.card_union_of_disjoint hdisj01]
  rw [hunion, huniv, hA0, hA1] at htot
  dsimp only [A₂] at htot
  omega

lemma card_bigSubsets (m : ℕ) : (BigSubsets m).card = 2 ^ m - m - 1 := by
  have h := card_bigSubsets_add m
  omega

/-- Base `(m + 1) × (2 * m)` matrix `B_exp m = [[I_m, J_m - I_m], [0, 1]]` for Theorem 2.3. -/
def B_exp (m : ℕ) : Matrix (Fin (m + 1)) (Fin (2 * m)) Bool :=
  fun i j =>
    if j.val < m then
      Fin.lastCases false (fun r => decide (r.val = j.val)) i
    else
      Fin.lastCases true (fun r => decide (r.val ≠ j.val - m)) i

/-- Equivalence indexing the `2^m - m - 1` subsets of `Fin m` of size at least `2`. -/
noncomputable def bigSubsetsEquiv (m : ℕ) : Fin (BigSubsets m).card ≃ BigSubsets m :=
  (Fintype.equivFinOfCardEq (Fintype.card_coe (BigSubsets m))).symm

/-- Full `(m + 1) × (2 * m + |BigSubsets m|)` matrix `C_exp m` for Theorem 2.3:
columns `0 .. 2*m - 1` are `B_exp m`, and the remaining `2^m - m - 1` columns are
`c_S = ⋀_{b ∈ S} b_{m+b}` for each `S ⊆ Fin m` with `2 ≤ S.card`. -/
noncomputable def C_exp (m : ℕ) :
    Matrix (Fin (m + 1)) (Fin (2 * m + (BigSubsets m).card)) Bool :=
  fun i j =>
    if h : j.val < 2 * m then
      B_exp m i ⟨j.val, h⟩
    else
      Fin.lastCases true
        (fun r => decide (r ∉ (bigSubsetsEquiv m ⟨j.val - 2 * m, by omega⟩).1)) i

lemma C_exp_castAdd {m : ℕ} (i : Fin (m + 1)) (j : Fin (2 * m)) :
    C_exp m i (Fin.castAdd (BigSubsets m).card j) = B_exp m i j := by
  unfold C_exp
  rw [dif_pos (show (Fin.castAdd (BigSubsets m).card j).val < 2 * m from j.isLt)]
  rfl

lemma C_exp_natAdd_castSucc {m : ℕ} (r : Fin m) (t : Fin (BigSubsets m).card) :
    C_exp m r.castSucc (Fin.natAdd (2 * m) t) = decide (r ∉ (bigSubsetsEquiv m t).1) := by
  have hnot : ¬ (Fin.natAdd (2 * m) t).val < 2 * m := by simp
  have ht : (⟨(Fin.natAdd (2 * m) t).val - 2 * m, by simp⟩ : Fin (BigSubsets m).card) = t := by
    ext; simp
  unfold C_exp
  rw [dif_neg hnot, Fin.lastCases_castSucc, ht]

lemma C_exp_natAdd_last {m : ℕ} (t : Fin (BigSubsets m).card) :
    C_exp m (Fin.last m) (Fin.natAdd (2 * m) t) = true := by
  have hnot : ¬ (Fin.natAdd (2 * m) t).val < 2 * m := by simp
  unfold C_exp
  rw [dif_neg hnot, Fin.lastCases_last]

lemma C_exp_last_false_imp {m : ℕ} {k : Fin (2 * m + (BigSubsets m).card)}
    (hk : C_exp m (Fin.last m) k = false) : k.val < m := by
  by_contra hge
  unfold C_exp at hk
  split_ifs at hk with h2m
  · simp [B_exp, hge] at hk
  · rw [Fin.lastCases_last] at hk; cases hk

lemma C_exp_castSucc_of_lt {m : ℕ} (r : Fin m) {k : Fin (2 * m + (BigSubsets m).card)}
    (hk : k.val < m) : C_exp m r.castSucc k = decide (r.val = k.val) := by
  have h2m : k.val < 2 * m := by omega
  unfold C_exp
  rw [dif_pos h2m]
  simp [B_exp, hk]

lemma C_exp_last_of_lt {m : ℕ} {k : Fin (2 * m + (BigSubsets m).card)}
    (hk : k.val < m) : C_exp m (Fin.last m) k = false := by
  have h2m : k.val < 2 * m := by omega
  unfold C_exp
  rw [dif_pos h2m]
  simp [B_exp, hk]

lemma C_exp_castSucc_of_mid {m : ℕ} (r : Fin m) {k : Fin (2 * m + (BigSubsets m).card)}
    (hk1 : ¬ k.val < m) (hk2 : k.val < 2 * m) :
    C_exp m r.castSucc k = decide (r.val ≠ k.val - m) := by
  unfold C_exp
  rw [dif_pos hk2]
  simp [B_exp, hk1]

lemma C_exp_last_of_ge {m : ℕ} {k : Fin (2 * m + (BigSubsets m).card)}
    (hk1 : ¬ k.val < m) : C_exp m (Fin.last m) k = true := by
  unfold C_exp
  split_ifs with h2m
  · simp [B_exp, hk1]
  · rw [Fin.lastCases_last]

lemma C_exp_castSucc_of_ge2m {m : ℕ} (r : Fin m) {k : Fin (2 * m + (BigSubsets m).card)}
    (hk2 : ¬ k.val < 2 * m) :
    C_exp m r.castSucc k =
      decide (r ∉ (bigSubsetsEquiv m ⟨k.val - 2 * m, by omega⟩).1) := by
  unfold C_exp
  rw [dif_neg hk2, Fin.lastCases_castSucc]

lemma exists_mem_ne_of_two_le_card {α : Type*} [DecidableEq α] {S : Finset α}
    (hS : 2 ≤ S.card) (b : α) : ∃ r ∈ S, r ≠ b := by
  by_contra h
  push_neg at h
  have hsub : S ⊆ {b} := fun r hr => Finset.mem_singleton.mpr (h r hr)
  have hcard := Finset.card_le_card hsub
  simp at hcard
  omega

lemma brank_B_exp (m : ℕ) (hm : 1 ≤ m) : brank (B_exp m) = m + 1 := by
  refine brank_eq (hasBoolRank_rows (B_exp m)) ?_
  intro r ⟨V, H, hVH⟩
  have hex : ∀ a : Fin m, ∃ t : Fin r, V a.castSucc t = true ∧ H t ⟨a.val, by omega⟩ = true := by
    intro a
    have ha := congrFun (congrFun hVH a.castSucc) ⟨a.val, by omega⟩
    have hval : B_exp m a.castSucc ⟨a.val, by omega⟩ = true := by
      simp [B_exp]
    rw [hval] at ha
    simpa [boolMul] using ha
  choose f hfV hfH using hex
  have hex_last : ∃ t : Fin r, V (Fin.last m) t = true ∧ H t ⟨m, by omega⟩ = true := by
    have hl := congrFun (congrFun hVH (Fin.last m)) ⟨m, by omega⟩
    have hval : B_exp m (Fin.last m) ⟨m, by omega⟩ = true := by
      simp [B_exp]
    rw [hval] at hl
    simpa [boolMul] using hl
  rcases hex_last with ⟨t_last, ht_lastV, _⟩
  have hrow_eq : ∀ (i : Fin (m + 1)) (a : Fin m), V i (f a) = true → i = a.castSucc := by
    intro i a hVi
    have htrue : boolMul V H i ⟨a.val, by omega⟩ = true := by
      simp only [boolMul, decide_eq_true_eq]
      exact ⟨f a, hVi, hfH a⟩
    rw [hVH] at htrue
    revert htrue
    refine Fin.lastCases ?_ ?_ i
    · intro hfalse
      simp [B_exp] at hfalse
    · intro r₀ hr₀
      simp [B_exp] at hr₀
      exact congrArg Fin.castSucc (Fin.ext hr₀)
  let g : Fin (m + 1) → Fin r := Fin.lastCases t_last f
  have hginj : Function.Injective g := by
    intro i₁ i₂
    refine Fin.lastCases ?_ ?_ i₁
    · refine Fin.lastCases ?_ ?_ i₂
      · intro _; rfl
      · intro a₂ heq
        simp only [g, Fin.lastCases_last, Fin.lastCases_castSucc] at heq
        have hbad := hrow_eq (Fin.last m) a₂ (heq ▸ ht_lastV)
        have hval : m = a₂.val := congrArg Fin.val hbad
        omega
    · intro a₁
      refine Fin.lastCases ?_ ?_ i₂
      · intro heq
        simp only [g, Fin.lastCases_last, Fin.lastCases_castSucc] at heq
        have hbad := hrow_eq (Fin.last m) a₁ (heq.symm ▸ ht_lastV)
        have hval : m = a₁.val := congrArg Fin.val hbad
        omega
      · intro a₂ heq
        simp only [g, Fin.lastCases_castSucc] at heq
        exact hrow_eq a₁.castSucc a₂ (heq ▸ hfV a₁)
  have hcard := Fintype.card_le_of_injective g hginj
  simpa using hcard

lemma brank_C_exp (m : ℕ) (hm : 1 ≤ m) : brank (C_exp m) = m + 1 := by
  refine brank_eq (hasBoolRank_rows (C_exp m)) ?_
  intro r hr
  have hsub : HasBoolRank (B_exp m) r := by
    have h := hasBoolRank_reindex_cols (Fin.castAdd (BigSubsets m).card) hr
    have heq : (fun i j => C_exp m i (Fin.castAdd (BigSubsets m).card j)) = B_exp m := by
      ext i j; exact C_exp_castAdd i j
    rwa [heq] at h
  have := brank_le hsub
  rw [brank_B_exp m hm] at this
  exact this

lemma C_exp_not_trivial (m : ℕ) (j : Fin (2 * m + (BigSubsets m).card)) :
    ¬ IsTrivial (C_exp m) j := by
  intro htriv
  by_cases hj : j.val < m
  · let r : Fin (m + 1) := (⟨j.val, hj⟩ : Fin m).castSucc
    have hr : r ∈ trunk (C_exp m) {j} := by
      simp only [trunk, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton,
        forall_eq, r]
      rw [C_exp_castSucc_of_lt _ hj]
      simp
    rw [htriv] at hr
    exact Finset.notMem_empty r hr
  · have hr : Fin.last m ∈ trunk (C_exp m) {j} := by
      simp only [trunk, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton,
        forall_eq]
      exact C_exp_last_of_ge hj
    rw [htriv] at hr
    exact Finset.notMem_empty (Fin.last m) hr

lemma C_exp_base_not_redundant (m : ℕ) (j₀ : Fin (2 * m)) :
    ¬ IsRedundant (C_exp m) (Fin.castAdd (BigSubsets m).card j₀) := by
  set j : Fin (2 * m + (BigSubsets m).card) := Fin.castAdd (BigSubsets m).card j₀
  rintro ⟨σ, hjnot, heq⟩
  by_cases hj : j₀.val < m
  · have hlast_not : Fin.last m ∉ trunk (C_exp m) σ := by
      rw [← heq]
      simp only [trunk, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton,
        forall_eq]
      rw [C_exp_last_of_lt (show j.val < m from hj)]
      decide
    have hex_k : ∃ k ∈ σ, C_exp m (Fin.last m) k = false := by
      by_contra hall
      push_neg at hall
      apply hlast_not
      simp only [trunk, Finset.mem_filter, Finset.mem_univ, true_and]
      intro k hk
      cases hck : C_exp m (Fin.last m) k
      · exact absurd hck (hall k hk)
      · rfl
    rcases hex_k with ⟨k, hkσ, hk_last⟩
    have hk_lt : k.val < m := C_exp_last_false_imp hk_last
    have hkj : k ≠ j := fun h => hjnot (h ▸ hkσ)
    let a : Fin m := ⟨j₀.val, hj⟩
    have ha_in : a.castSucc ∈ trunk (C_exp m) σ := by
      rw [← heq]
      simp only [trunk, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton,
        forall_eq]
      rw [C_exp_castSucc_of_lt a (show j.val < m from hj)]
      simp [a, j]
    have ha_k := (Finset.mem_filter.mp ha_in).2 k hkσ
    rw [C_exp_castSucc_of_lt a hk_lt] at ha_k
    simp only [decide_eq_true_eq, a] at ha_k
    exact hkj (Fin.ext (by simpa [j] using ha_k.symm))
  · have hlast_in : Fin.last m ∈ trunk (C_exp m) σ := by
      rw [← heq]
      simp only [trunk, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton,
        forall_eq]
      exact C_exp_last_of_ge (show ¬ j.val < m from hj)
    have hall_ge : ∀ k ∈ σ, ¬ k.val < m := by
      intro k hkσ hlt
      have hkt := (Finset.mem_filter.mp hlast_in).2 k hkσ
      rw [C_exp_last_of_lt hlt] at hkt
      cases hkt
    let b : Fin m := ⟨j₀.val - m, by omega⟩
    have hb_not : b.castSucc ∉ trunk (C_exp m) σ := by
      rw [← heq]
      simp only [trunk, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton,
        forall_eq]
      rw [C_exp_castSucc_of_mid b (show ¬ j.val < m from hj) (show j.val < 2 * m from j₀.isLt)]
      simp [b, j]
    have hex_k : ∃ k ∈ σ, C_exp m b.castSucc k = false := by
      by_contra hall
      push_neg at hall
      apply hb_not
      simp only [trunk, Finset.mem_filter, Finset.mem_univ, true_and]
      intro k hk
      cases hck : C_exp m b.castSucc k
      · exact absurd hck (hall k hk)
      · rfl
    rcases hex_k with ⟨k, hkσ, hk_b⟩
    have hk_ge : ¬ k.val < m := hall_ge k hkσ
    have hkj : k ≠ j := fun h => hjnot (h ▸ hkσ)
    by_cases hk2m : k.val < 2 * m
    · rw [C_exp_castSucc_of_mid b hk_ge hk2m, decide_eq_false_iff_not] at hk_b
      have hkeq : b.val = k.val - m := by
        by_contra hne
        exact hk_b hne
      exact hkj (Fin.ext (by show k.val = j₀.val; have hb : b.val = j₀.val - m := rfl; omega))
    · let S : Finset (Fin m) := (bigSubsetsEquiv m ⟨k.val - 2 * m, by omega⟩).1
      have hScard : 2 ≤ S.card :=
        (Finset.mem_filter.mp (bigSubsetsEquiv m ⟨k.val - 2 * m, by omega⟩).2).2
      obtain ⟨r, hrS, hrb⟩ := exists_mem_ne_of_two_le_card hScard b
      have hr_in : r.castSucc ∈ trunk (C_exp m) σ := by
        rw [← heq]
        simp only [trunk, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton,
          forall_eq]
        rw [C_exp_castSucc_of_mid r (show ¬ j.val < m from hj) (show j.val < 2 * m from j₀.isLt)]
        simp only [decide_eq_true_eq]
        intro hval
        apply hrb
        exact Fin.ext (by show r.val = j₀.val - m; simpa [j] using hval)
      have hr_k := (Finset.mem_filter.mp hr_in).2 k hkσ
      rw [C_exp_castSucc_of_ge2m r hk2m] at hr_k
      simp only [decide_eq_true_eq] at hr_k
      exact hr_k hrS

lemma C_exp_appended_non_simply_redundant (m : ℕ) (t : Fin (BigSubsets m).card) :
    IsNonSimplyRedundant (C_exp m) (Fin.natAdd (2 * m) t) := by
  set j : Fin (2 * m + (BigSubsets m).card) := Fin.natAdd (2 * m) t
  set S : Finset (Fin m) := (bigSubsetsEquiv m t).1
  have hScard : 2 ≤ S.card := (Finset.mem_filter.mp (bigSubsetsEquiv m t).2).2
  constructor
  · let σ : Finset (Fin (2 * m + (BigSubsets m).card)) :=
      S.image (fun b : Fin m => ⟨m + b.val, by omega⟩)
    refine ⟨σ, ?_, ?_⟩
    · intro hjσ
      rcases Finset.mem_image.mp hjσ with ⟨b, _, hb⟩
      have hval : m + b.val = 2 * m + t.val := congrArg Fin.val hb
      omega
    · ext i
      refine Fin.lastCases ?_ ?_ i
      · simp only [trunk, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton,
          forall_eq]
        rw [show C_exp m (Fin.last m) j = true from C_exp_natAdd_last t]
        simp only [true_iff]
        intro k hkσ
        rcases Finset.mem_image.mp hkσ with ⟨b, _, rfl⟩
        exact C_exp_last_of_ge (by dsimp; omega)
      · intro r
        simp only [trunk, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton,
          forall_eq]
        rw [show C_exp m r.castSucc j = decide (r ∉ S) from C_exp_natAdd_castSucc r t]
        simp only [decide_eq_true_eq]
        constructor
        · intro hrS k hkσ
          rcases Finset.mem_image.mp hkσ with ⟨b, hbS, rfl⟩
          rw [C_exp_castSucc_of_mid r (by dsimp; omega) (by dsimp; have := b.isLt; omega)]
          simp only [decide_eq_true_eq]
          intro hval
          apply hrS
          have hrb : r = b := Fin.ext (by omega)
          rwa [hrb]
        · intro hall hrS
          have hkσ : (⟨m + r.val, by omega⟩ : Fin (2 * m + (BigSubsets m).card)) ∈ σ :=
            Finset.mem_image.mpr ⟨r, hrS, rfl⟩
          have hkr := hall _ hkσ
          rw [C_exp_castSucc_of_mid r (by dsimp; omega) (by dsimp; have := r.isLt; omega)] at hkr
          simp only [decide_eq_true_eq] at hkr
          dsimp at hkr
          omega
  · rintro (⟨_, heq⟩ | ⟨k, hjk, heq⟩)
    · obtain ⟨r, hrS⟩ := Finset.card_pos.mp (by omega : 0 < S.card)
      have hr_in : r.castSucc ∈ trunk (C_exp m) {j} := by
        rw [heq]; simp [trunk]
      have hr_val := (Finset.mem_filter.mp hr_in).2 j (Finset.mem_singleton_self j)
      rw [show C_exp m r.castSucc j = decide (r ∉ S) from C_exp_natAdd_castSucc r t] at hr_val
      simp only [decide_eq_true_eq] at hr_val
      exact hr_val hrS
    · have hkj : k ≠ j := fun h => hjk (Finset.mem_singleton.mpr h.symm)
      by_cases hk1 : k.val < m
      · have hlast : Fin.last m ∈ trunk (C_exp m) {k} := by
          rw [← heq]
          simp only [trunk, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton,
            forall_eq]
          exact C_exp_natAdd_last t
        have hval := (Finset.mem_filter.mp hlast).2 k (Finset.mem_singleton_self k)
        rw [C_exp_last_of_lt hk1] at hval
        cases hval
      · by_cases hk2 : k.val < 2 * m
        · let b : Fin m := ⟨k.val - m, by omega⟩
          obtain ⟨r, hrS, hrb⟩ := exists_mem_ne_of_two_le_card hScard b
          have hr_in : r.castSucc ∈ trunk (C_exp m) {j} := by
            rw [heq]
            simp only [trunk, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton,
              forall_eq]
            rw [C_exp_castSucc_of_mid r hk1 hk2]
            simp only [decide_eq_true_eq]
            intro hval
            exact hrb (Fin.ext hval)
          have hr_val := (Finset.mem_filter.mp hr_in).2 j (Finset.mem_singleton_self j)
          rw [show C_exp m r.castSucc j = decide (r ∉ S) from C_exp_natAdd_castSucc r t] at hr_val
          simp only [decide_eq_true_eq] at hr_val
          exact hr_val hrS
        · let t' : Fin (BigSubsets m).card := ⟨k.val - 2 * m, by omega⟩
          have hSeq : S = (bigSubsetsEquiv m t').1 := by
            ext r
            have hiff : r.castSucc ∈ trunk (C_exp m) {j} ↔ r.castSucc ∈ trunk (C_exp m) {k} := by
              rw [heq]
            simp only [trunk, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton,
              forall_eq] at hiff
            rw [show C_exp m r.castSucc j = decide (r ∉ S) from C_exp_natAdd_castSucc r t,
                C_exp_castSucc_of_ge2m r hk2] at hiff
            simp only [decide_eq_true_eq] at hiff
            exact not_iff_not.mp hiff
          have htt' : t = t' := (bigSubsetsEquiv m).injective (Subtype.ext hSeq)
          apply hkj
          ext
          have hval : t.val = k.val - 2 * m := congrArg Fin.val htt'
          have hjval : j.val = 2 * m + t.val := rfl
          omega

/-- **Theorem 2.3 (Exponential Gap with Zero Rank Increase)**:
For every `m ≥ 2`, `C_exp m` is an `(m + 1) × (2^m + m - 1)` binary matrix extending the
`(m + 1) × (2 * m)` base matrix `B_exp m` by `ℓ = 2^m - m - 1` columns such that:
1. `(BigSubsets m).card = 2 ^ m - m - 1` and total columns `2 * m + (BigSubsets m).card = 2 ^ m + m - 1`;
2. Every column of `C_exp m` is non-trivial;
3. All `2 * m` base columns of `B_exp m` are non-redundant in `C_exp m`;
4. All `2^m - m - 1` appended columns are non-simply redundant in `C_exp m`;
5. `brank (B_exp m) = m + 1` and `brank (C_exp m) = m + 1`
   (while `brank (B_exp m) + ℓ = (m + 1) + (2^m - m - 1) = 2^m`). -/
theorem exponential_gap_family (m : ℕ) (hm : 2 ≤ m) :
    (BigSubsets m).card = 2 ^ m - m - 1 ∧
    2 * m + (BigSubsets m).card = 2 ^ m + m - 1 ∧
    (m + 1) + (BigSubsets m).card = 2 ^ m ∧
    (∀ j : Fin (2 * m + (BigSubsets m).card), ¬ IsTrivial (C_exp m) j) ∧
    (∀ j : Fin (2 * m), ¬ IsRedundant (C_exp m) (Fin.castAdd (BigSubsets m).card j)) ∧
    (∀ t : Fin (BigSubsets m).card, IsNonSimplyRedundant (C_exp m) (Fin.natAdd (2 * m) t)) ∧
    brank (B_exp m) = m + 1 ∧
    brank (C_exp m) = m + 1 := by
  have hadd := card_bigSubsets_add m
  refine ⟨card_bigSubsets m, ?_, ?_, C_exp_not_trivial m, C_exp_base_not_redundant m,
    C_exp_appended_non_simply_redundant m, brank_B_exp m (by omega), brank_C_exp m (by omega)⟩
  · omega
  · omega

/-! ### Proposition 2.5 & Counterexamples 2.6, 2.7 (Reverse Direction Refutations) -/

/-- Appending an all-1s column to an `m × n` Boolean matrix `M`. -/
def appendAllOnes {m n : ℕ} (M : Matrix (Fin m) (Fin n) Bool) :
    Matrix (Fin m) (Fin (n + 1)) Bool :=
  fun i j => Fin.lastCases true (fun k => M i k) j

theorem brank_appendAllOnes_of_no_zero_row {m n : ℕ} (M : Matrix (Fin m) (Fin n) Bool)
    (hnz : ∀ i : Fin m, ∃ j : Fin n, M i j = true) :
    brank (appendAllOnes M) = brank M := by
  have hub : HasBoolRank (appendAllOnes M) (brank M) := by
    obtain ⟨V, H, hVH⟩ := hasBoolRank_brank M
    refine ⟨V, fun t j => Fin.lastCases true (fun k => H t k) j, ?_⟩
    ext i j
    refine Fin.lastCases ?_ ?_ j
    · simp [boolMul, appendAllOnes]
      obtain ⟨j₀, hj₀⟩ := hnz i
      have hrow := congrFun (congrFun hVH i) j₀
      rw [hj₀] at hrow
      simp only [boolMul, decide_eq_true_eq] at hrow
      obtain ⟨t, htV, _⟩ := hrow
      exact ⟨t, htV⟩
    · intro k
      have hrow := congrFun (congrFun hVH i) k
      simpa [boolMul, appendAllOnes] using hrow
  refine brank_eq hub ?_
  intro r hr
  have hsub : HasBoolRank M r := by
    have h := hasBoolRank_reindex_cols (fun k : Fin n => k.castSucc) hr
    have heq : (fun i (k : Fin n) => appendAllOnes M i k.castSucc) = M := by
      ext i k; simp [appendAllOnes]
    rwa [heq] at h
  exact brank_le hsub

theorem brank_appendAllOnes_of_zero_row {m n : ℕ} (M : Matrix (Fin m) (Fin n) Bool)
    (h0 : ∃ i₀ : Fin m, ∀ j : Fin n, M i₀ j = false) :
    brank (appendAllOnes M) = brank M + 1 := by
  have hub : HasBoolRank (appendAllOnes M) (brank M + 1) := by
    obtain ⟨V, H, hVH⟩ := hasBoolRank_brank M
    let V' : Matrix (Fin m) (Fin (brank M + 1)) Bool :=
      fun i t => Fin.lastCases true (fun t₀ => V i t₀) t
    let H' : Matrix (Fin (brank M + 1)) (Fin (n + 1)) Bool :=
      fun t j =>
        Fin.lastCases
          (Fin.lastCases true (fun _ => false) j)
          (fun t₀ => Fin.lastCases false (fun k => H t₀ k) j)
          t
    refine ⟨V', H', ?_⟩
    ext i j
    refine Fin.lastCases ?_ ?_ j
    · simp only [boolMul, appendAllOnes, Fin.lastCases_last, decide_eq_true_eq]
      refine ⟨Fin.last (brank M), ?_, ?_⟩
      · simp [V']
      · simp [H']
    · intro k
      have hrow := congrFun (congrFun hVH i) k
      cases hM : M i k
      · rw [hM] at hrow
        simp only [boolMul, decide_eq_false_iff_not, not_exists, not_and] at hrow
        simp only [boolMul, appendAllOnes, Fin.lastCases_castSucc, hM, decide_eq_false_iff_not,
          not_exists, not_and]
        intro t
        refine Fin.lastCases ?_ ?_ t
        · intro _ hH
          simp [H'] at hH
        · intro t₀ hV hH
          simp only [V', Fin.lastCases_castSucc] at hV
          simp only [H', Fin.lastCases_castSucc] at hH
          exact hrow t₀ hV hH
      · rw [hM] at hrow
        simp only [boolMul, decide_eq_true_eq] at hrow
        rcases hrow with ⟨t₀, htV, htH⟩
        simp only [boolMul, appendAllOnes, Fin.lastCases_castSucc, hM, decide_eq_true_eq]
        refine ⟨t₀.castSucc, ?_, ?_⟩
        · simp [V', htV]
        · simp [H', htH]
  refine brank_eq hub ?_
  intro r ⟨V, H, hVH⟩
  rcases h0 with ⟨i₀, hi₀⟩
  have hlast := congrFun (congrFun hVH i₀) (Fin.last n)
  simp only [boolMul, appendAllOnes, Fin.lastCases_last, decide_eq_true_eq] at hlast
  rcases hlast with ⟨t₀, ht₀V, _⟩
  have hHt₀ : ∀ k : Fin n, H t₀ k.castSucc = false := by
    intro k
    cases hk : H t₀ k.castSucc
    · rfl
    · exfalso
      have hrow := congrFun (congrFun hVH i₀) k.castSucc
      simp only [appendAllOnes, Fin.lastCases_castSucc, hi₀ k] at hrow
      have htrue : boolMul V H i₀ k.castSucc = true := by
        simp only [boolMul, decide_eq_true_eq]
        exact ⟨t₀, ht₀V, hk⟩
      rw [htrue] at hrow
      cases hrow
  cases r with
  | zero => exact Fin.elim0 t₀
  | succ r' =>
    let drop : Fin r' → Fin (r' + 1) := fun s =>
      if h : s.val < t₀.val then ⟨s.val, by omega⟩ else ⟨s.val + 1, by omega⟩
    have hsurj : ∀ t : Fin (r' + 1), t ≠ t₀ → ∃ s : Fin r', drop s = t := by
      intro t ht
      by_cases hlt : t.val < t₀.val
      · refine ⟨⟨t.val, by omega⟩, ?_⟩
        ext
        simp [drop, hlt]
      · have hgt : t₀.val < t.val := by
          have hne : t.val ≠ t₀.val := fun h => ht (Fin.ext h)
          omega
        refine ⟨⟨t.val - 1, by omega⟩, ?_⟩
        have hnot : ¬ (t.val - 1 < t₀.val) := by omega
        ext
        dsimp only [drop]
        rw [dif_neg hnot]
        dsimp only
        omega
    have hM : HasBoolRank M r' := by
      refine ⟨fun i s => V i (drop s), fun s k => H (drop s) k.castSucc, ?_⟩
      ext i k
      have hrow := congrFun (congrFun hVH i) k.castSucc
      simp only [appendAllOnes, Fin.lastCases_castSucc] at hrow
      cases hMik : M i k
      · rw [hMik] at hrow
        simp only [boolMul, decide_eq_false_iff_not, not_exists, not_and] at hrow ⊢
        intro s hV hH
        exact hrow (drop s) hV hH
      · rw [hMik] at hrow
        simp only [boolMul, decide_eq_true_eq] at hrow ⊢
        rcases hrow with ⟨t, htV, htH⟩
        have htne : t ≠ t₀ := by
          intro heq
          subst heq
          rw [hHt₀ k] at htH
          cases htH
        obtain ⟨s, hs⟩ := hsurj t htne
        exact ⟨s, hs.symm ▸ htV, hs.symm ▸ htH⟩
    have hle := brank_le hM
    omega

/-- **Proposition 2.5 (Exact Effect of an All-1s Column — General Refutation of Section 7.3 Line 1112)**:
Appending an all-1s column to any `m × n` Boolean matrix `M` increases `brank` by `1` iff
`M` has an all-zero row, and leaves `brank` unchanged iff `M` has no all-zero row. -/
theorem brank_append_allOnes {m n : ℕ} (M : Matrix (Fin m) (Fin n) Bool) :
    ((∃ i₀ : Fin m, ∀ j : Fin n, M i₀ j = false) → brank (appendAllOnes M) = brank M + 1) ∧
    ((∀ i : Fin m, ∃ j : Fin n, M i j = true) → brank (appendAllOnes M) = brank M) :=
  ⟨brank_appendAllOnes_of_zero_row M, brank_appendAllOnes_of_no_zero_row M⟩

/-- Counterexample 2.6: `2 × 2` matrix with an all-1s column (`k = 1, ℓ = 0`) and `brank = 2 > 1`. -/
def C_all1 : Matrix (Fin 2) (Fin 2) Bool :=
  ![ ![false, true],
     ![true, true] ]

def rho_all1 : Matrix (Fin 2) (Fin 1) Bool :=
  ![ ![false],
     ![true] ]

theorem refutation_all1_reverse :
    IsSimplyRedundant C_all1 1 ∧
    ¬ IsNonSimplyRedundant C_all1 1 ∧
    brank rho_all1 = 1 ∧
    brank C_all1 = 2 := by
  refine ⟨Or.inl (by decide), ?_, ?_, ?_⟩
  · intro h; exact h.2 (Or.inl (by decide))
  · refine brank_eq (hasBoolRank_cols rho_all1) ?_
    intro r ⟨V, H, hVH⟩
    by_contra hr; push_neg at hr
    have hr0 : r = 0 := by omega
    subst hr0
    have h1 := congrFun (congrFun hVH 1) 0
    simp [boolMul, rho_all1] at h1
  · -- Swap columns of C_all1 to get lower-triangular [[1,0],[1,1]]!
    refine brank_eq (hasBoolRank_rows C_all1) ?_
    intro r hr
    let swap : Fin 2 → Fin 2 := fun j => if j = 0 then 1 else 0
    have hsub := hasBoolRank_reindex_cols swap hr
    have hlt : brank (fun i j => C_all1 i (swap j)) = 2 := by
      apply brank_lower_triangular <;> decide
    have := brank_le hsub
    omega

/-- Counterexample 2.7: Duplicated non-simple redundancy (`4 × 4`, no all-1s col, `∅ ∈ C`).
Both cols 2 and 3 are simply redundant (to each other), so `k = 2, ℓ = 0`, yet `brank = 3 > 2`. -/
def C_dup : Matrix (Fin 4) (Fin 4) Bool :=
  ![ ![false, false, false, false],
     ![true,  false, false, false],
     ![false, true,  false, false],
     ![true,  true,  true,  true] ]

def rho_dup : Matrix (Fin 4) (Fin 2) Bool :=
  fun i j => C_dup i j.castSucc.castSucc

theorem refutation_dup_reverse :
    IsSimplyRedundant C_dup 2 ∧
    IsSimplyRedundant C_dup 3 ∧
    (∀ j : Fin 4, ¬ IsNonSimplyRedundant C_dup j) ∧
    brank rho_dup = 2 ∧
    brank C_dup = 3 := by
  have hsr2 : IsSimplyRedundant C_dup 2 := Or.inr ⟨3, by decide⟩
  have hsr3 : IsSimplyRedundant C_dup 3 := Or.inr ⟨2, by decide⟩
  refine ⟨hsr2, hsr3, ?_, ?_, ?_⟩
  · intro j hnsr
    fin_cases j
    · rcases hnsr.1 with ⟨σ, hnot, heq⟩; revert σ hnot heq; decide
    · rcases hnsr.1 with ⟨σ, hnot, heq⟩; revert σ hnot heq; decide
    · exact hnsr.2 hsr2
    · exact hnsr.2 hsr3
  · refine brank_eq (hasBoolRank_cols rho_dup) ?_
    intro r hr
    let rf : Fin 2 → Fin 4 := ![1, 2]
    have hsub : HasBoolRank (fun i j => rho_dup (rf i) j) r := hasBoolRank_submatrix rf id hr
    have hlt : brank (fun i j => rho_dup (rf i) j) = 2 := by
      apply brank_lower_triangular <;> decide
    have := brank_le hsub; omega
  · -- Upper bound 3: columns 2 and 3 are equal, so factor through first 3 cols
    have hub : HasBoolRank C_dup 3 := by
      refine ⟨![ ![false, false, false], ![true, false, false],
                 ![false, true, false], ![true, true, true] ],
              ![ ![true, false, false, false], ![false, true, false, false],
                 ![false, false, true, true] ], by decide⟩
    refine brank_eq hub ?_
    intro r hr
    -- Extract rows 1, 2, 3 and cols 0, 1, 2 -> lower triangular 3 × 3!
    let rf : Fin 3 → Fin 4 := ![1, 2, 3]
    let cf : Fin 3 → Fin 4 := ![0, 1, 2]
    have hsub := hasBoolRank_submatrix rf cf hr
    have hlt : brank (fun i j => C_dup (rf i) (cf j)) = 3 := by
      apply brank_lower_triangular <;> decide
    have := brank_le hsub; omega

/-! ### Theorem 3.4: Positive Theorem — Exact Boolean Rank of ALL Intersection-Complete Codes -/

/-- Number of distinct nonzero columns of `C`. -/
def distinctNonzeroCols {m n : ℕ} (C : Matrix (Fin m) (Fin n) Bool) : Finset (Fin m → Bool) :=
  (Finset.univ.image (fun j : Fin n => fun i : Fin m => C i j)).filter (fun col => ∃ i, col i = true)

/-- **Theorem 3.4 (Exact Boolean Rank of Intersection-Complete Codes)**:
For EVERY intersection-complete binary matrix `C : Matrix (Fin m) (Fin n) Bool`,
`brank C` equals the number of distinct nonzero columns of `C`! -/
theorem brank_intersection_complete {m n : ℕ} (C : Matrix (Fin m) (Fin n) Bool)
    (hic : IsIntersectionComplete C) :
    brank C = (distinctNonzeroCols C).card := by
  set U := distinctNonzeroCols C
  set u := U.card
  -- Step 1 (Upper bound: brank C ≤ u): Factor C through the u distinct nonzero columns.
  have hub : HasBoolRank C u := by
    let e : Fin u ≃ U := (Fintype.equivFinOfCardEq (Fintype.card_coe U)).symm
    refine ⟨fun i t => (e t).1 i, fun t j => decide ((e t).1 = fun i => C i j), ?_⟩
    ext i j
    cases hC : C i j
    · simp only [boolMul, decide_eq_false_iff_not, not_exists, not_and]
      intro t h1 h2
      have h2' : (e t).1 = fun i' => C i' j := of_decide_eq_true h2
      have := congrFun h2' i
      rw [h1, hC] at this; cases this
    · simp only [boolMul, decide_eq_true_eq]
      have hmem : (fun i' => C i' j) ∈ U := by
        simp only [U, distinctNonzeroCols, Finset.mem_filter, Finset.mem_image, Finset.mem_univ,
          true_and]
        exact ⟨⟨j, rfl⟩, ⟨i, hC⟩⟩
      refine ⟨e.symm ⟨fun i' => C i' j, hmem⟩, ?_, ?_⟩
      · show (e (e.symm ⟨fun i' => C i' j, hmem⟩)).1 i = true
        rw [Equiv.apply_symm_apply]; exact hC
      · show (e (e.symm ⟨fun i' => C i' j, hmem⟩)).1 = fun i' => C i' j
        rw [Equiv.apply_symm_apply]
  -- Step 2 (Lower bound: u ≤ brank C):
  -- For each nonzero column col ∈ U, choose a representative column index rep col : Fin n,
  -- and take the intersection of all rows in its nonempty trunk!
  -- By intersection-completeness, this intersection is realized by some row rootRow(col) : Fin m!
  have hroot_exists : ∀ (col : U), ∃ r : Fin m,
      (∀ j : Fin n, C r j = true ↔ ∀ i : Fin m, col.1 i = true → C i j = true) := by
    intro col
    -- Nonempty finite set of rows T = {i | col.1 i = true}
    let T : Finset (Fin m) := Finset.univ.filter (fun i => col.1 i = true)
    have hTne : T.Nonempty := by
      have hmem := (Finset.mem_filter.mp col.2).2
      rcases hmem with ⟨i, hi⟩
      exact ⟨i, Finset.mem_filter.mpr ⟨Finset.mem_univ i, hi⟩⟩
    -- Prove by Finset.Nonempty.induction that for ANY nonempty S ⊆ Fin m,
    -- there exists a row r realizing the bitwise AND of all rows in S!
    suffices hfold : ∀ (S : Finset (Fin m)), S.Nonempty →
        ∃ r : Fin m, ∀ j : Fin n, C r j = true ↔ ∀ i ∈ S, C i j = true by
      obtain ⟨r, hr⟩ := hfold T hTne
      refine ⟨r, fun j => ?_⟩
      rw [hr j]
      simp [T]
    intro S hSne
    refine Finset.Nonempty.cons_induction (s := S) (hs := hSne) ?_ ?_
    · intro a
      exact ⟨a, fun j => by simp⟩
    · intro a s has hs ⟨r_s, hr_s⟩
      obtain ⟨r_new, hr_new⟩ := hic a r_s
      refine ⟨r_new, fun j => ?_⟩
      rw [hr_new j, Bool.and_eq_true, hr_s j]
      simp
  -- Pick column index rep(col) and root row rootRow(col) for each col ∈ U
  have hrep_exists : ∀ (col : U), ∃ j : Fin n, (fun i => C i j) = col.1 := by
    intro col
    have hmem := (Finset.mem_filter.mp col.2).1
    rcases Finset.mem_image.mp hmem with ⟨j, _, hj⟩
    exact ⟨j, hj⟩
  choose rep hrep using hrep_exists
  choose rootRow hroot using hroot_exists
  -- Sort U by nondecreasing Hamming weight of rootRow(col) using Tuple.sort!
  let weight : U → ℕ := fun col => (Finset.univ.filter (fun j : Fin n => C (rootRow col) j = true)).card
  let e0 : Fin u ≃ U := (Fintype.equivFinOfCardEq (Fintype.card_coe U)).symm
  let perm : Equiv.Perm (Fin u) := Tuple.sort (fun t : Fin u => weight (e0 t))
  let idx : Fin u → U := fun t => e0 (perm t)
  have hidx_inj : Function.Injective idx := e0.injective.comp perm.injective
  have hidx_mono : ∀ a b : Fin u, a < b → weight (idx a) ≤ weight (idx b) := by
    intro a b hab
    exact Tuple.monotone_sort (fun t : Fin u => weight (e0 t)) (le_of_lt hab)
  -- Extract the u × u submatrix M a b := C (rootRow (idx a)) (rep (idx b))
  let M : Matrix (Fin u) (Fin u) Bool := fun a b => C (rootRow (idx a)) (rep (idx b))
  -- Check M is lower-triangular with 1s on diagonal!
  have hMdiag : ∀ a : Fin u, M a a = true := by
    intro a
    show C (rootRow (idx a)) (rep (idx a)) = true
    rw [hroot (idx a) (rep (idx a))]
    intro i hi
    have hcol := congrFun (hrep (idx a)) i
    rwa [hcol]
  have hMzero : ∀ a b : Fin u, a < b → M a b = false := by
    intro a b hab
    by_contra hne
    have htrue : C (rootRow (idx a)) (rep (idx b)) = true := by
      cases h : M a b
      · exact absurd h hne
      · exact h
    -- Since C (rootRow (idx a)) (rep (idx b)) = true, every row in trunk(idx a) is in trunk(idx b),
    -- i.e. idx a ⊆ idx b as column supports! Wait: check the direction!
    -- hroot (idx a) says: C (rootRow (idx a)) j = true ↔ ∀ i, (idx a).1 i = true → C i j = true.
    -- Putting j = rep (idx b): ∀ i, (idx a).1 i = true → (idx b).1 i = true!
    have hcol_sub : ∀ i : Fin m, (idx a).1 i = true → (idx b).1 i = true := by
      intro i hi
      have h1 := (hroot (idx a) (rep (idx b))).mp htrue i hi
      rwa [congrFun (hrep (idx b)) i] at h1
    -- Therefore EVERY neuron j active on trunk(idx b) is ALSO active on trunk(idx a)!
    -- i.e., support(rootRow (idx b)) ⊆ support(rootRow (idx a))!
    have hrow_sub : Finset.univ.filter (fun j : Fin n => C (rootRow (idx b)) j = true) ⊆
        Finset.univ.filter (fun j : Fin n => C (rootRow (idx a)) j = true) := by
      intro j hj
      rw [Finset.mem_filter] at hj ⊢
      refine ⟨Finset.mem_univ j, ?_⟩
      rw [hroot (idx a) j]
      intro i hi
      exact (hroot (idx b) j).mp hj.2 i (hcol_sub i hi)
    -- Wait: we sorted by NONDECREASING weight, so weight(idx a) ≤ weight(idx b)!
    -- Together with row_sub (weight(idx b) ≤ weight(idx a)), the two root rows have EQUAL support!
    have hcard_le := Finset.card_le_card hrow_sub
    have hmono := hidx_mono a b hab
    have hrow_eq : Finset.univ.filter (fun j : Fin n => C (rootRow (idx b)) j = true) =
        Finset.univ.filter (fun j : Fin n => C (rootRow (idx a)) j = true) :=
      Finset.eq_of_subset_of_card_le hrow_sub hmono
    -- Wait: since rootRow(idx a) and rootRow(idx b) have the exact same true columns,
    -- in particular rep(idx a) is true in rootRow(idx b)!
    have hmem_a : rep (idx a) ∈ Finset.univ.filter (fun j : Fin n => C (rootRow (idx a)) j = true) := by
      rw [Finset.mem_filter]; exact ⟨Finset.mem_univ _, hMdiag a⟩
    rw [← hrow_eq, Finset.mem_filter] at hmem_a
    have hcol_sub2 : ∀ i : Fin m, (idx b).1 i = true → (idx a).1 i = true := by
      intro i hi
      have h1 := (hroot (idx b) (rep (idx a))).mp hmem_a.2 i hi
      rwa [congrFun (hrep (idx a)) i] at h1
    -- Hence (idx a).1 = (idx b).1, contradicting a < b!
    have heq_col : idx a = idx b := by
      apply Subtype.ext
      funext i
      cases ha : (idx a).1 i <;> cases hb : (idx b).1 i
      · rfl
      · exfalso; have := hcol_sub2 i hb; rw [ha] at this; cases this
      · exfalso; have := hcol_sub i ha; rw [hb] at this; cases this
      · rfl
    exact ne_of_lt hab (hidx_inj heq_col)
  refine brank_eq hub ?_
  intro r hr
  have hsub : HasBoolRank M r := hasBoolRank_submatrix (fun a => rootRow (idx a)) (fun b => rep (idx b)) hr
  have hMrank : brank M = u := brank_lower_triangular M hMdiag hMzero
  have := brank_le hsub
  omega

/-! ### Theorem 4.1: Resolution of Question 2 (Section 7.2) — Sharpness `λ(C^{(1)}) = 2n - 2` -/

/-- For `n ≥ 3`, the `(2n - 1) × n` matrix `C_q2 n` representing
`C_n = {∅} ∪ {{j} : 1 ≤ j < n} ∪ {{0, j} : 1 ≤ j < n}` (0-indexed neurons, neuron `0` is `i`). -/
def C_q2 (n : ℕ) : Matrix (Fin (1 + 2 * (n - 1))) (Fin n) Bool :=
  fun r c =>
    if r.val = 0 then false
    else if r.val ≤ n - 1 then decide (c.val = r.val)
    else decide (c.val = 0 ∨ c.val = r.val - (n - 1))

/-- The reduced covering code `D_q2 n` of `C_q2 n` with respect to neuron `0`,
having `2n - 2` columns: `n - 1` columns `A_j = tk(j)` and `n - 1` columns `B_j = tk(0) ∩ tk(j)`. -/
def D_q2 (n : ℕ) : Matrix (Fin (1 + 2 * (n - 1))) (Fin (2 * (n - 1))) Bool :=
  fun r c =>
    if c.val < n - 1 then
      -- A_j for j = c.val + 1: active on row j (single {j}) and row (n-1)+j ({0, j})
      decide (r.val = c.val + 1 ∨ r.val = (n - 1) + (c.val + 1))
    else
      -- B_j for j = c.val - (n - 1) + 1: active ONLY on row (n-1)+j ({0, j})
      decide (r.val = (n - 1) + (c.val - (n - 1) + 1))

/-- **Theorem 4.1 (Complete Resolution of Question 2, Section 7.2)**:
For EVERY `n ≥ 3`:
1. `C_q2 n` is reduced (so `λ(C_q2 n) = n`).
2. `D_q2 n` (representing the non-trivial trunks of `(C_q2 n)^{(0)}`) is reduced on `2n - 2` neurons,
   so `λ((C_q2 n)^{(0)}) = 2n - 2 = 2λ(C_q2 n) - 2`! -/
theorem question2_sharpness (n : ℕ) (hn : 3 ≤ n) :
    IsReduced (C_q2 n) ∧ IsReduced (D_q2 n) := by
  constructor
  · -- Part 1: C_q2 n is reduced
    constructor
    · -- No trivial neurons
      intro j htriv
      by_cases hj0 : j.val = 0
      · -- Row n (which is (n-1) + 1) has neuron 0 active!
        let r : Fin (1 + 2 * (n - 1)) := ⟨n, by omega⟩
        have hr : r ∈ trunk (C_q2 n) {j} := by
          simp only [trunk, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton,
            forall_eq, C_q2, r, hj0]
          have h1 : n ≠ 0 := by omega
          have h2 : ¬ (n ≤ n - 1) := by omega
          simp [h1, h2]
        rw [htriv] at hr; exact Finset.notMem_empty r hr
      · -- Row j.val has neuron j active!
        let r : Fin (1 + 2 * (n - 1)) := ⟨j.val, by omega⟩
        have hr : r ∈ trunk (C_q2 n) {j} := by
          simp only [trunk, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton,
            forall_eq, C_q2, r]
          have h1 : j.val ≠ 0 := hj0
          have h2 : j.val ≤ n - 1 := by omega
          simp [h1, h2]
        rw [htriv] at hr; exact Finset.notMem_empty r hr
    · -- No redundant neurons
      intro j ⟨σ, hjnot, heq⟩
      by_cases hj0 : j.val = 0
      · by_cases hσ : σ = ∅
        · let r0 : Fin (1 + 2 * (n - 1)) := ⟨0, by omega⟩
          have hr0 : r0 ∈ trunk (C_q2 n) ∅ := by simp [trunk]
          rw [← hσ, ← heq] at hr0
          simp [trunk, C_q2, r0] at hr0
        · obtain ⟨k, hk⟩ := Finset.nonempty_iff_ne_empty.mpr hσ
          have hk0 : k.val ≠ 0 := by
            intro h; apply hjnot
            have : j = k := Fin.ext (by omega)
            rwa [this]
          let m : ℕ := if k.val = 1 then 2 else 1
          have hm_ne : k.val ≠ m := by dsimp [m]; split_ifs <;> omega
          have hm_le : m ≤ n - 1 := by dsimp [m]; split_ifs <;> omega
          have hm_pos : 1 ≤ m := by dsimp [m]; split_ifs <;> omega
          let r : Fin (1 + 2 * (n - 1)) := ⟨(n - 1) + m, by omega⟩
          have hr_in : r ∈ trunk (C_q2 n) {j} := by
            simp only [trunk, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton,
              forall_eq, C_q2, r, hj0]
            have h1 : (n - 1) + m ≠ 0 := by omega
            have h2 : ¬ ((n - 1) + m ≤ n - 1) := by omega
            simp only [h1, h2, ↓reduceIte, decide_eq_true_eq]
            exact Or.inl trivial
          rw [heq] at hr_in
          have hr_k := (Finset.mem_filter.mp hr_in).2 k hk
          simp only [C_q2, r] at hr_k
          have h1 : (n - 1) + m ≠ 0 := by omega
          have h2 : ¬ ((n - 1) + m ≤ n - 1) := by omega
          simp only [h1, h2, ↓reduceIte, decide_eq_true_eq] at hr_k
          rcases hr_k with h | h
          · exact hk0 h
          · omega
      · by_cases hσ : σ = ∅
        · let r0 : Fin (1 + 2 * (n - 1)) := ⟨0, by omega⟩
          have hr0 : r0 ∈ trunk (C_q2 n) ∅ := by simp [trunk]
          rw [← hσ, ← heq] at hr0
          simp [trunk, C_q2, r0] at hr0
        · obtain ⟨k, hk⟩ := Finset.nonempty_iff_ne_empty.mpr hσ
          have hkj : k.val ≠ j.val := fun h => hjnot (by rwa [Fin.ext h.symm])
          let r : Fin (1 + 2 * (n - 1)) := ⟨j.val, by omega⟩
          have hr_in : r ∈ trunk (C_q2 n) {j} := by
            simp only [trunk, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton,
              forall_eq, C_q2, r]
            have h2 : j.val ≤ n - 1 := by omega
            simp [hj0, h2]
          rw [heq] at hr_in
          have hr_k := (Finset.mem_filter.mp hr_in).2 k hk
          simp only [C_q2, r] at hr_k
          have h2 : j.val ≤ n - 1 := by omega
          simp only [hj0, h2, ↓reduceIte, decide_eq_true_eq] at hr_k
          exact hkj hr_k
  · -- Part 2: D_q2 n is reduced on 2*(n - 1) neurons!
    constructor
    · -- No trivial neurons
      intro c htriv
      by_cases hc : c.val < n - 1
      · let r : Fin (1 + 2 * (n - 1)) := ⟨c.val + 1, by omega⟩
        have hr : r ∈ trunk (D_q2 n) {c} := by
          simp [trunk, D_q2, r, hc]
        rw [htriv] at hr; exact Finset.notMem_empty r hr
      · let r : Fin (1 + 2 * (n - 1)) := ⟨(n - 1) + (c.val - (n - 1) + 1), by omega⟩
        have hr : r ∈ trunk (D_q2 n) {c} := by
          simp [trunk, D_q2, r, hc]
        rw [htriv] at hr; exact Finset.notMem_empty r hr
    · -- No redundant neurons
      intro c ⟨σ, hcnot, heq⟩
      by_cases hσ : σ = ∅
      · let r0 : Fin (1 + 2 * (n - 1)) := ⟨0, by omega⟩
        have hr0 : r0 ∈ trunk (D_q2 n) ∅ := by simp [trunk]
        rw [← hσ, ← heq] at hr0
        simp only [trunk, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton,
          forall_eq, D_q2, r0] at hr0
        split_ifs at hr0 <;> simp at hr0 <;> omega
      · by_cases hc : c.val < n - 1
        · -- Column A_j (j = c.val + 1): row j has ONLY column c active!
          obtain ⟨k, hk⟩ := Finset.nonempty_iff_ne_empty.mpr hσ
          have hkc : k.val ≠ c.val := fun h => hcnot (by rwa [Fin.ext h.symm])
          let r : Fin (1 + 2 * (n - 1)) := ⟨c.val + 1, by omega⟩
          have hr_in : r ∈ trunk (D_q2 n) {c} := by simp [trunk, D_q2, r, hc]
          rw [heq] at hr_in
          have hr_k := (Finset.mem_filter.mp hr_in).2 k hk
          simp only [D_q2, r] at hr_k
          split_ifs at hr_k <;> simp at hr_k <;> omega
        · -- Column B_j (j = c.val - (n - 1) + 1): row (n-1)+j has ONLY c and c-(n-1) active!
          -- So every k ∈ σ must be c-(n-1) (the unique other column active on row (n-1)+j).
          -- Thus σ = {c-(n-1)}, whose trunk also contains row j, whereas B_j does NOT contain row j!
          let r_top : Fin (1 + 2 * (n - 1)) := ⟨(n - 1) + (c.val - (n - 1) + 1), by omega⟩
          have hr_top : r_top ∈ trunk (D_q2 n) {c} := by simp [trunk, D_q2, r_top, hc]
          rw [heq] at hr_top
          -- Every k ∈ σ must equal c_partner = c.val - (n - 1)
          have hall : ∀ k ∈ σ, k.val = c.val - (n - 1) := by
            intro k hk
            have hr_k := (Finset.mem_filter.mp hr_top).2 k hk
            have hkc : k.val ≠ c.val := fun h => hcnot (by rwa [Fin.ext h.symm])
            simp only [D_q2, r_top] at hr_k
            split_ifs at hr_k <;> simp at hr_k <;> omega
          -- Now row j = c.val - (n - 1) + 1 is in trunk σ, hence in trunk {c}, contradiction!
          let r_bot : Fin (1 + 2 * (n - 1)) := ⟨c.val - (n - 1) + 1, by omega⟩
          have hr_bot : r_bot ∈ trunk (D_q2 n) σ := by
            simp only [trunk, Finset.mem_filter, Finset.mem_univ, true_and]
            intro k hk
            have hkv := hall k hk
            have hlt : k.val < n - 1 := by omega
            simp only [D_q2, r_bot, hlt, ↓reduceIte, decide_eq_true_eq]
            left; omega
          rw [← heq] at hr_bot
          have hbot_c := (Finset.mem_filter.mp hr_bot).2 c (Finset.mem_singleton_self c)
          simp only [D_q2, r_bot, hc, ↓reduceIte, decide_eq_true_eq] at hbot_c
          omega

end NeuralCodesBMF
