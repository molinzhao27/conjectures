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
3. **All-1s Column Counterexample (`refutation_all1_reverse`)**:
   For `C_all1 = [[0,1],[1,1]]` and its reduced submatrix `rho_all1 = [[0],[1]]`, column 1 is simply
   redundant, yet `brank(C_all1) = 2 > 1 = brank(rho_all1)`.
4. **Duplicated Non-Simple Redundancy (`refutation_dup_reverse`)**:
   Explicit `4 × 4` matrix `C_dup` and submatrix `rho_dup`, where columns 2 and 3 are each simply
   redundant to the other, with `brank(C_dup) = 3 > 2 = brank(rho_dup)`. (Note: as flagged in review,
   this violates the conjecture under the simultaneous definition of `ℓ`, not under sequential
   deduplication-first counting.)
5. **Lower-Triangular Boolean Rank (`brank_lower_triangular`)**: Every `u × u` lower-triangular
   Boolean matrix with `1`s on the diagonal has `brank = u`.
6. **Positive Theorem for ALL Intersection-Complete Codes (`brank_intersection_complete`)**:
   For EVERY intersection-complete binary matrix `C : Matrix (Fin m) (Fin n) Bool`,
   `brank(C)` equals the number of distinct nonzero columns of `C`!
7. **Reducedness Verification for Question 2 (`question2_sharpness`)**:
   For EVERY `n ≥ 3`, the explicit `n`-neuron code `C_q2 n` and the explicit `(2n - 2)`-neuron code
   `D_q2 n` (representing the covering code after deleting the two trivial neurons) are both reduced.

Note: The exponential-gap family `C_exp`, general reduction operator `ρ`, Proposition 2.4–2.5,
Theorem 3.1, minimality of `3 × 4`, and the formal link from `def:excessivecovering` to `D_q2 n`
are proved in prose in `solutions/neural_codes/solution.md`.
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
    have h1 := congrFun (congrFun hVH i₁) i₂
    have h2 := congrFun (congrFun hVH i₂) i₂
    simp only [rho_min, decide_eq_true_eq, decide_eq_false_iff_not, ne_eq, not_true_eq_false,
      boolMul] at h1 h2
    have h1' : ∃ t : Fin r, V i₁ t = true ∧ H t i₂ = true := by
      have : decide (i₁ ≠ i₂) = true := decide_eq_true hne
      rw [this] at h1
      exact of_decide_eq_true h1
    rcases h1' with ⟨t, ht1, htH⟩
    refine ⟨t, ht1, ?_⟩
    cases hvt2 : V i₂ t
    · rfl
    · exfalso
      have hfalse : decide (∃ t : Fin r, V i₂ t = true ∧ H t i₂ = true) = true :=
        decide_eq_true ⟨t, hvt2, htH⟩
      rw [hfalse] at h2
      cases h2
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

/-! ### Proposition 2.5 & Counterexamples 2.6, 2.7 (Reverse Direction Refutations) -/

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
      apply brank_lower_triangular
      · intro i; fin_cases i <;> decide
      · intro i j hij; fin_cases i <;> fin_cases j <;> simp at hij <;> decide
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
