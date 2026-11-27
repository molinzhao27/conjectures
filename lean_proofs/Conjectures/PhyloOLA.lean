import Mathlib

/-!
# Formalization of the Combinatorial Core of the OLA–SPR Bound (arXiv:2503.10169 / Bull. Math. Biol. 2026)

Richman, Zhang, Matsen (arXiv:2503.10169, Conjecture 11; published version *Bull. Math. Biol.* 88:58,
2026, Theorem 6) study the OLA distance `d_OLA(T, T')` after a single rSPR move.

We formalize in Lean 4 + Mathlib (zero `sorry`) the **combinatorial core** underlying our sharper
deterministic worst-case bound `d_OLA(T, T') ≤ |B| - 1 ≤ 2 * ht_r(T) - 2`:
- Inductive type `BinTree` of rooted binary leaf-labeled trees on `ℕ` (note: `internalSubtrees`
  collects subtrees as values, which is unambiguous under `IsValid` since distinct positions have
  distinct leaf sets).
- Exact definitions of `leaves`, `height` (`ht_r`, from the topmost bifurcating node; paper `ht_ρ = ht_r + 1`),
  `cladeFounder` (`CF`), `cladeSplitter` (`CS`), `restrict` (`T|_i`), `olaEntry` (`a_i(T)`), and `olaDist`.
- **Lemma 1 (`olaEntry_subtree_locality`)**: If an internal subtree `S` appears inside both `T` and `T'`,
  then at index `i = CS(S)`, `olaEntry T i = olaEntry T' i`.
- **Lemma 2 (`olaDist_le_upward_closed`)**: If `B` is a nonempty upward-closed set of internal subtrees
  of `T` such that every internal subtree of `T` not in `B` occurs as a subtree of `T'`, then
  `olaDist n T T' ≤ |B| - 1`.
- **Combinatorial rSPR Bound (`deterministic_spr_bound`)**: Taking `B = ancestors(p) ∪ ancestors(v_top)`
  and assuming the preservation hypothesis `hkeep` (proved for every rSPR move in prose in
  `solutions/phylo_ola/solution.md`), proves `olaDist n T T' ≤ |B| - 1 ≤ 2 * height(T) - 2`.

Note: The rSPR move definition itself, the reduction from rSPR to `hkeep`, child-swap invariance of
`olaEntry`, exact sharpness for `2h ≤ n ≤ 2^h`, and the caterpillar lower bound are proved in prose in
`solutions/phylo_ola/solution.md`.
-/

open Finset

namespace PhyloOLA

/-- Rooted binary tree with leaf labels in `ℕ`. -/
inductive BinTree where
  | leaf : ℕ → BinTree
  | node : BinTree → BinTree → BinTree
  deriving DecidableEq, Repr

namespace BinTree

/-- Set of leaf labels appearing in `t`. -/
def leaves : BinTree → Finset ℕ
  | leaf l => {l}
  | node l r => leaves l ∪ leaves r

/-- Every `BinTree` has nonempty leaf set. -/
lemma leaves_nonempty : ∀ t : BinTree, (leaves t).Nonempty
  | leaf l => ⟨l, Finset.mem_singleton_self l⟩
  | node l _ => (leaves_nonempty l).mono Finset.subset_union_left

/-- Height `ht_r(t)` = max edges from root of `t` to a leaf. -/
def height : BinTree → ℕ
  | leaf _ => 0
  | node l r => 1 + max (height l) (height r)

/-- Clade-founder `CF(t) = min(leaves t)`. -/
def cf (t : BinTree) : ℕ :=
  (leaves t).min' (leaves_nonempty t)

lemma cf_mem_leaves (t : BinTree) : cf t ∈ leaves t :=
  Finset.min'_mem _ _

lemma cf_le_of_mem {t : BinTree} {x : ℕ} (hx : x ∈ leaves t) : cf t ≤ x :=
  Finset.min'_le _ _ hx

lemma cf_node (l r : BinTree) : cf (node l r) = min (cf l) (cf r) := by
  apply le_antisymm
  · rw [le_min_iff]
    exact ⟨cf_le_of_mem (Finset.mem_union_left _ (cf_mem_leaves l)),
           cf_le_of_mem (Finset.mem_union_right _ (cf_mem_leaves r))⟩
  · unfold cf
    rw [Finset.le_min'_iff]
    intro x hx
    rcases Finset.mem_union.mp hx with hl | hr
    · exact le_trans (min_le_left _ _) (cf_le_of_mem hl)
    · exact le_trans (min_le_right _ _) (cf_le_of_mem hr)

/-- Clade-splitter `CS(node l r) = max(CF(l), CF(r))`. -/
def cs : BinTree → ℕ
  | leaf _ => 0
  | node l r => max (cf l) (cf r)

/-- Canonical label of a subtree: `+(l : ℤ)` for a leaf, `-(cs t : ℤ)` for an internal node. -/
def canonLabel : BinTree → ℤ
  | leaf l => (l : ℤ)
  | node l r => -((cs (node l r) : ℕ) : ℤ)

/-- `IsValid t`: disjoint leaf sets at every internal node (each leaf label occurs at most once). -/
def IsValid : BinTree → Prop
  | leaf _ => True
  | node l r => Disjoint (leaves l) (leaves r) ∧ IsValid l ∧ IsValid r

/-- `IsPhylogenetic n t`: valid tree with exact leaf set `{0, 1, ..., n-1}`. -/
def IsPhylogenetic (n : ℕ) (t : BinTree) : Prop :=
  IsValid t ∧ leaves t = Finset.range n

/-- Restriction `restrict A t`: induced rooted binary tree on leaf subset `A`
(returns `none` if `leaves t` is disjoint from `A`). -/
def restrict (A : Finset ℕ) : BinTree → Option BinTree
  | leaf l => if l ∈ A then some (leaf l) else none
  | node l r =>
    match restrict A l, restrict A r with
    | none, none => none
    | some l', none => some l'
    | none, some r' => some r'
    | some l', some r' => some (node l' r')

lemma restrict_isSome_iff (A : Finset ℕ) :
    ∀ t : BinTree, (restrict A t).isSome ↔ (leaves t ∩ A).Nonempty
  | leaf l => by
    simp only [restrict, leaves]
    by_cases hl : l ∈ A
    · simp [hl]
    · simp [hl]
  | node l r => by
    simp only [restrict, leaves]
    have ihl := restrict_isSome_iff A l
    have ihr := restrict_isSome_iff A r
    cases hrl : restrict A l <;> cases hrr : restrict A r <;>
      simp_all [Finset.union_inter_distrib_right]

lemma restrict_leaves (A : Finset ℕ) :
    ∀ {t t' : BinTree}, restrict A t = some t' → leaves t' = leaves t ∩ A
  | leaf l, t', h => by
    simp only [restrict] at h
    split_ifs at h with hl
    · cases h; simp [leaves, hl]
  | node l r, t', h => by
    simp only [restrict] at h
    cases hrl : restrict A l with
    | none =>
      rw [hrl] at h
      cases hrr : restrict A r with
      | none => rw [hrr] at h; cases h
      | some r' =>
        rw [hrr] at h; cases h
        have hl0 : leaves l ∩ A = ∅ := by
          rw [← Finset.not_nonempty_iff_eq_empty, ← restrict_isSome_iff A l, hrl]; simp
        rw [restrict_leaves A hrr, leaves, Finset.union_inter_distrib_right, hl0, Finset.empty_union]
    | some l' =>
      rw [hrl] at h
      cases hrr : restrict A r with
      | none =>
        rw [hrr] at h; cases h
        have hr0 : leaves r ∩ A = ∅ := by
          rw [← Finset.not_nonempty_iff_eq_empty, ← restrict_isSome_iff A r, hrr]; simp
        rw [restrict_leaves A hrl, leaves, Finset.union_inter_distrib_right, hr0, Finset.union_empty]
      | some r' =>
        rw [hrr] at h; cases h
        rw [leaves, restrict_leaves A hrl, restrict_leaves A hrr, leaves,
          Finset.union_inter_distrib_right]

/-- Stability of `cf` under restriction to `range (k + 1)` when `cf t ≤ k`. -/
lemma cf_restrict_range {k : ℕ} {t t' : BinTree}
    (hres : restrict (Finset.range (k + 1)) t = some t') :
    cf t' = cf t := by
  have hlt' : leaves t' = leaves t ∩ Finset.range (k + 1) := restrict_leaves _ hres
  have hne : (leaves t ∩ Finset.range (k + 1)).Nonempty := by
    rw [← hlt']; exact leaves_nonempty t'
  rcases hne with ⟨x, hx⟩
  have hxin : x ∈ leaves t ∧ x < k + 1 := by simpa using hx
  have hcf_lt : cf t < k + 1 := lt_of_le_of_lt (cf_le_of_mem hxin.1) hxin.2
  have hcf_mem' : cf t ∈ leaves t' := by
    rw [hlt']; exact Finset.mem_inter.mpr ⟨cf_mem_leaves t, Finset.mem_range.mpr hcf_lt⟩
  apply le_antisymm
  · exact cf_le_of_mem hcf_mem'
  · have hm : cf t' ∈ leaves t := by
      have := cf_mem_leaves t'
      rw [hlt'] at this
      exact (Finset.mem_inter.mp this).1
    exact cf_le_of_mem hm

/-- Sister canonical label of leaf `i` in a tree `t` (returns `none` if `i` is not a child of an internal node). -/
def sisterOf (i : ℕ) : BinTree → Option ℤ
  | leaf _ => none
  | node l r =>
    if l = leaf i then some (canonLabel r)
    else if r = leaf i then some (canonLabel l)
    else match sisterOf i l with
         | some z => some z
         | none => sisterOf i r

/-- The `i`-th OLA entry `a_i(t)` (for `i ≥ 1`): canonical label of the sister of leaf `i` in `t|_i`. -/
def olaEntry (t : BinTree) (i : ℕ) : Option ℤ :=
  match restrict (Finset.range (i + 1)) t with
  | none => none
  | some ti => sisterOf i ti

/-- Exact Hamming OLA distance on coordinates `i ∈ {1, ..., n - 1}`. -/
def olaDist (n : ℕ) (t₁ t₂ : BinTree) : ℕ :=
  ((Finset.Ico 1 n).filter (fun i => olaEntry t₁ i ≠ olaEntry t₂ i)).card

/-- Subtree relation: `s` is a subtree of `t` (possibly equal to `t`). -/
inductive Subtree : BinTree → BinTree → Prop
  | refl (t : BinTree) : Subtree t t
  | left {s l r : BinTree} : Subtree s l → Subtree s (node l r)
  | right {s l r : BinTree} : Subtree s r → Subtree s (node l r)

lemma subtree_leaves_subset {s t : BinTree} (h : Subtree s t) : leaves s ⊆ leaves t := by
  induction h with
  | refl => exact Finset.Subset.refl _
  | left _ ih => exact ih.trans Finset.subset_union_left
  | right _ ih => exact ih.trans Finset.subset_union_right

lemma subtree_valid {s t : BinTree} (h : Subtree s t) (hv : IsValid t) : IsValid s := by
  induction h with
  | refl => exact hv
  | left _ ih => exact ih hv.2.1
  | right _ ih => exact ih hv.2.2

/-- If `i ∈ leaves s` and `s` is a valid subtree of a valid tree `t`, then `sisterOf i` on `t`
agrees with `sisterOf i` on `s` whenever `sisterOf i s` is `some z`. -/
lemma sisterOf_of_subtree {i : ℕ} {s t : BinTree} {z : ℤ}
    (hsub : Subtree s t) (hv : IsValid t) (hs : sisterOf i s = some z) :
    sisterOf i t = some z := by
  induction hsub with
  | refl => exact hs
  | @left l r hsub_l ih =>
    have hvl := hv.2.1
    have hdisj := hv.1
    have hi_l : i ∈ leaves l := by
      have hi_s : i ∈ leaves s := by
        clear ih hsub_l hvl hdisj hv
        induction s with
        | leaf x => simp [sisterOf] at hs
        | node a b iha ihb =>
          simp only [sisterOf] at hs
          split_ifs at hs with ha hb
          · subst ha; simp [leaves]
          · subst hb; simp [leaves]
          · cases h1 : sisterOf i a with
            | some w =>
              rw [h1] at hs; cases hs
              exact Finset.mem_union_left _ (iha h1)
            | none =>
              rw [h1] at hs
              exact Finset.mem_union_right _ (ihb hs)
      exact subtree_leaves_subset hsub_l hi_s
    have hne_l : l ≠ leaf i := by
      intro hl
      subst hl
      have : s = leaf i := by
        cases hsub_l with
        | refl => rfl
      subst this
      simp [sisterOf] at hs
    have hne_r : r ≠ leaf i := by
      intro hr
      subst hr
      have hi_r : i ∈ leaves (leaf i) := by simp [leaves]
      exact Finset.disjoint_left.mp hdisj hi_l hi_r
    simp only [sisterOf, hne_l, hne_r, ↓reduceIte, ih hvl]
  | @right l r hsub_r ih =>
    have hvr := hv.2.2
    have hdisj := hv.1
    have hi_r : i ∈ leaves r := by
      have hi_s : i ∈ leaves s := by
        clear ih hsub_r hvr hdisj hv
        induction s with
        | leaf x => simp [sisterOf] at hs
        | node a b iha ihb =>
          simp only [sisterOf] at hs
          split_ifs at hs with ha hb
          · subst ha; simp [leaves]
          · subst hb; simp [leaves]
          · cases h1 : sisterOf i a with
            | some w =>
              rw [h1] at hs; cases hs
              exact Finset.mem_union_left _ (iha h1)
            | none =>
              rw [h1] at hs
              exact Finset.mem_union_right _ (ihb hs)
      exact subtree_leaves_subset hsub_r hi_s
    have hnot_l : i ∉ leaves l := Finset.disjoint_right.mp hdisj hi_r
    have hne_l : l ≠ leaf i := by
      intro hl; subst hl; exact hnot_l (by simp [leaves])
    have hne_r : r ≠ leaf i := by
      intro hr
      subst hr
      have : s = leaf i := by
        cases hsub_r with
        | refl => rfl
      subst this
      simp [sisterOf] at hs
    have hnone_l : sisterOf i l = none := by
      clear ih hsub_r hvr hdisj hv hs hne_l hne_r hi_r
      induction l with
      | leaf x => rfl
      | node a b iha ihb =>
        have ha : i ∉ leaves a := fun h => hnot_l (Finset.mem_union_left _ h)
        have hb : i ∉ leaves b := fun h => hnot_l (Finset.mem_union_right _ h)
        have hnea : a ≠ leaf i := fun h => ha (by subst h; simp [leaves])
        have hneb : b ≠ leaf i := fun h => hb (by subst h; simp [leaves])
        simp [sisterOf, hnea, hneb, iha ha, ihb hb]
    simp only [sisterOf, hne_l, hne_r, ↓reduceIte, hnone_l, ih hvr]

/-- Restriction preserves validity. -/
lemma restrict_valid (A : Finset ℕ) :
    ∀ {t t' : BinTree}, IsValid t → restrict A t = some t' → IsValid t'
  | leaf l, t', _, h => by
    simp only [restrict] at h
    split_ifs at h with hl
    · cases h; trivial
  | node l r, t', hv, h => by
    simp only [restrict] at h
    cases hrl : restrict A l with
    | none =>
      rw [hrl] at h
      cases hrr : restrict A r with
      | none => rw [hrr] at h; cases h
      | some r' => rw [hrr] at h; cases h; exact restrict_valid A hv.2.2 hrr
    | some l' =>
      rw [hrl] at h
      cases hrr : restrict A r with
      | none => rw [hrr] at h; cases h; exact restrict_valid A hv.2.1 hrl
      | some r' =>
        rw [hrr] at h; cases h
        refine ⟨?_, restrict_valid A hv.2.1 hrl, restrict_valid A hv.2.2 hrr⟩
        rw [restrict_leaves A hrl, restrict_leaves A hrr]
        exact hv.1.mono Finset.inter_subset_left Finset.inter_subset_left

/-- Restriction respects the subtree relation. -/
lemma restrict_subtree (A : Finset ℕ) {s t s' : BinTree}
    (hsub : Subtree s t) (hs : restrict A s = some s') :
    ∃ t' : BinTree, restrict A t = some t' ∧ Subtree s' t' := by
  induction hsub with
  | refl => exact ⟨s', hs, Subtree.refl s'⟩
  | @left l r _ ih =>
    obtain ⟨l', hl', hsub'⟩ := ih
    cases hrr : restrict A r with
    | none => exact ⟨l', by simp [restrict, hl', hrr], hsub'⟩
    | some r' => exact ⟨node l' r', by simp [restrict, hl', hrr], Subtree.left hsub'⟩
  | @right l r _ ih =>
    obtain ⟨r', hr', hsub'⟩ := ih
    cases hrl : restrict A l with
    | none => exact ⟨r', by simp [restrict, hrl, hr'], hsub'⟩
    | some l' => exact ⟨node l' r', by simp [restrict, hrl, hr'], Subtree.right hsub'⟩

/-- **Lemma 1 (Subtree Locality of OLA Entries)**:
If an internal subtree `w = node c₁ c₂` occurs as a subtree of a valid tree `t`,
then `olaEntry t (cs w)` depends ONLY on `w` itself (not on the surrounding tree `t`)!
Specifically, `olaEntry t (cs w) = olaEntry w (cs w)`. -/
theorem olaEntry_subtree_locality {c₁ c₂ t : BinTree}
    (hsub : Subtree (node c₁ c₂) t) (hv : IsValid t) :
    olaEntry t (cs (node c₁ c₂)) = olaEntry (node c₁ c₂) (cs (node c₁ c₂)) := by
  set w := node c₁ c₂
  set i := cs w
  set A := Finset.range (i + 1)
  have hvw : IsValid w := subtree_valid hsub hv
  have hdisj : Disjoint (leaves c₁) (leaves c₂) := hvw.1
  have hcf_ne : cf c₁ ≠ cf c₂ := fun heq =>
    Finset.disjoint_left.mp hdisj (cf_mem_leaves c₁) (heq ▸ cf_mem_leaves c₂)
  -- Both c₁ and c₂ intersect A = range (i + 1) because cf c₁ ≤ i and cf c₂ ≤ i!
  have h1ne : (leaves c₁ ∩ A).Nonempty :=
    ⟨cf c₁, Finset.mem_inter.mpr ⟨cf_mem_leaves c₁, Finset.mem_range.mpr (Nat.lt_succ_of_le (le_max_left _ _))⟩⟩
  have h2ne : (leaves c₂ ∩ A).Nonempty :=
    ⟨cf c₂, Finset.mem_inter.mpr ⟨cf_mem_leaves c₂, Finset.mem_range.mpr (Nat.lt_succ_of_le (le_max_right _ _))⟩⟩
  obtain ⟨c₁', hc₁'⟩ := Option.isSome_iff_exists.mp ((restrict_isSome_iff A c₁).mpr h1ne)
  obtain ⟨c₂', hc₂'⟩ := Option.isSome_iff_exists.mp ((restrict_isSome_iff A c₂).mpr h2ne)
  have hwres : restrict A w = some (node c₁' c₂') := by simp [w, restrict, hc₁', hc₂']
  -- One of c₁', c₂' has leaves = {i}, hence equals leaf i!
  have hleaf_of_leaves_singleton : ∀ u : BinTree, IsValid u → leaves u = {i} → u = leaf i := by
    intro u hvu hlu
    cases u with
    | leaf x =>
      have : x = i := Finset.singleton_injective hlu
      rw [this]
    | node a b =>
      exfalso
      obtain ⟨xa, hxa⟩ := leaves_nonempty a
      obtain ⟨xb, hxb⟩ := leaves_nonempty b
      have hxa' : xa = i := Finset.mem_singleton.mp (hlu ▸ Finset.mem_union_left _ hxa)
      have hxb' : xb = i := Finset.mem_singleton.mp (hlu ▸ Finset.mem_union_right _ hxb)
      exact Finset.disjoint_left.mp hvu.1 (hxa' ▸ hxa) (hxb' ▸ hxb)
  have hsis_some : ∃ z : ℤ, sisterOf i (node c₁' c₂') = some z := by
    rcases lt_or_gt_of_ne hcf_ne with hlt | hgt
    · -- cf c₁ < cf c₂ = i, so leaves c₂' = {i} ⇒ c₂' = leaf i
      have hi_eq : cf c₂ = i := (max_eq_right (le_of_lt hlt)).symm
      have hl2 : leaves c₂' = {i} := by
        rw [restrict_leaves A hc₂']
        ext x
        simp only [Finset.mem_inter, Finset.mem_range, Finset.mem_singleton, A]
        constructor
        · rintro ⟨hx2, hxlt⟩
          have hge : i ≤ x := hi_eq ▸ cf_le_of_mem hx2
          omega
        · rintro rfl
          exact ⟨hi_eq ▸ cf_mem_leaves c₂, Nat.lt_succ_self i⟩
      have hc2_leaf : c₂' = leaf i := hleaf_of_leaves_singleton c₂' (restrict_valid A hvw.2.2 hc₂') hl2
      have hc1_ne : c₁' ≠ leaf i := by
        intro h1l
        have hcf1 : cf c₁' = i := by rw [h1l]; rfl
        rw [cf_restrict_range hc₁'] at hcf1
        omega
      exact ⟨canonLabel c₁', by simp [sisterOf, hc1_ne, hc2_leaf]⟩
    · -- cf c₂ < cf c₁ = i, so leaves c₁' = {i} ⇒ c₁' = leaf i
      have hi_eq : cf c₁ = i := (max_eq_left (le_of_lt hgt)).symm
      have hl1 : leaves c₁' = {i} := by
        rw [restrict_leaves A hc₁']
        ext x
        simp only [Finset.mem_inter, Finset.mem_range, Finset.mem_singleton, A]
        constructor
        · rintro ⟨hx1, hxlt⟩
          have hge : i ≤ x := hi_eq ▸ cf_le_of_mem hx1
          omega
        · rintro rfl
          exact ⟨hi_eq ▸ cf_mem_leaves c₁, Nat.lt_succ_self i⟩
      have hc1_leaf : c₁' = leaf i := hleaf_of_leaves_singleton c₁' (restrict_valid A hvw.2.1 hc₁') hl1
      exact ⟨canonLabel c₂', by simp [sisterOf, hc1_leaf]⟩
  obtain ⟨z, hz⟩ := hsis_some
  obtain ⟨t', ht', hsub'⟩ := restrict_subtree A hsub hwres
  have hvt' : IsValid t' := restrict_valid A hv ht'
  have hsis_t : sisterOf i t' = some z := sisterOf_of_subtree hsub' hvt' hz
  simp only [olaEntry, A] at *
  rw [ht', hwres]
  simpa using hsis_t.trans hz.symm

/-- Corollary of Lemma 1: if an internal subtree `w = node c₁ c₂` occurs in BOTH valid trees `t₁` and `t₂`,
then `olaEntry t₁ (cs w) = olaEntry t₂ (cs w)`. -/
theorem olaEntry_eq_of_common_subtree {c₁ c₂ t₁ t₂ : BinTree}
    (h1 : Subtree (node c₁ c₂) t₁) (h2 : Subtree (node c₁ c₂) t₂)
    (hv1 : IsValid t₁) (hv2 : IsValid t₂) :
    olaEntry t₁ (cs (node c₁ c₂)) = olaEntry t₂ (cs (node c₁ c₂)) := by
  rw [olaEntry_subtree_locality h1 hv1, olaEntry_subtree_locality h2 hv2]

/-! ### Exact Bijection of Clade-Splitters and Lemma 2 / Theorem 3 -/

/-- List of all internal subtrees of `t`. -/
def internalSubtrees : BinTree → Finset BinTree
  | leaf _ => ∅
  | node l r => insert (node l r) (internalSubtrees l ∪ internalSubtrees r)

lemma mem_internalSubtrees_iff {w t : BinTree} :
    w ∈ internalSubtrees t ↔ Subtree w t ∧ ∃ c₁ c₂, w = node c₁ c₂ := by
  induction t with
  | leaf x =>
    simp only [internalSubtrees, Finset.notMem_empty, false_iff, not_and]
    intro hsub
    cases hsub
    rintro ⟨_, _, h⟩; cases h
  | node l r ihl ihr =>
    simp only [internalSubtrees, Finset.mem_insert, Finset.mem_union, ihl, ihr]
    constructor
    · rintro (rfl | ⟨h, hc⟩ | ⟨h, hc⟩)
      · exact ⟨Subtree.refl _, ⟨l, r, rfl⟩⟩
      · exact ⟨Subtree.left h, hc⟩
      · exact ⟨Subtree.right h, hc⟩
    · rintro ⟨hsub, c₁, c₂, rfl⟩
      cases hsub with
      | refl => exact Or.inl rfl
      | left h => exact Or.inr (Or.inl ⟨h, c₁, c₂, rfl⟩)
      | right h => exact Or.inr (Or.inr ⟨h, c₁, c₂, rfl⟩)

/-- Every `i ∈ {1, ..., n - 1}` is the clade-splitter of some internal subtree of `t`
when `IsPhylogenetic n t`. -/
lemma exists_internal_with_cs {n : ℕ} {t : BinTree} (hp : IsPhylogenetic n t)
    {i : ℕ} (hi1 : 1 ≤ i) (hin : i < n) :
    ∃ w ∈ internalSubtrees t, cs w = i := by
  -- Induction on any valid subtree `u` that contains both `i` and some leaf `< i`:
  suffices h : ∀ u : BinTree, IsValid u → i ∈ leaves u → (∃ x ∈ leaves u, x < i) →
      ∃ w ∈ internalSubtrees u, cs w = i by
    apply h t hp.1
    · rw [hp.2]; exact Finset.mem_range.mpr hin
    · refine ⟨0, ?_, hi1⟩
      rw [hp.2]; exact Finset.mem_range.mpr (by omega)
  intro u hvu hi_u hlt_u
  induction u with
  | leaf y =>
    rcases hlt_u with ⟨x, hx_u, hxi⟩
    have hy1 : i = y := Finset.mem_singleton.mp hi_u
    have hy2 : x = y := Finset.mem_singleton.mp hx_u
    omega
  | node l r ihl ihr =>
    rcases hlt_u with ⟨x, hx_u, hxi⟩
    rcases Finset.mem_union.mp hi_u with hi_l | hi_r
    · by_cases hx_l : ∃ y ∈ leaves l, y < i
      · obtain ⟨w, hw, hcs⟩ := ihl hvu.2.1 hi_l hx_l
        exact ⟨w, Finset.mem_insert_of_mem (Finset.mem_union_left _ hw), hcs⟩
      · push_neg at hx_l
        have hcf_l : cf l = i := le_antisymm (cf_le_of_mem hi_l) (hx_l (cf l) (cf_mem_leaves l))
        have hx_r : x ∈ leaves r := by
          rcases Finset.mem_union.mp hx_u with h | h
          · exfalso; have := hx_l x h; omega
          · exact h
        have hcf_r : cf r < i := lt_of_le_of_lt (cf_le_of_mem hx_r) hxi
        refine ⟨node l r, Finset.mem_insert_self _ _, ?_⟩
        simp [cs, hcf_l, le_of_lt hcf_r]
    · by_cases hx_r : ∃ y ∈ leaves r, y < i
      · obtain ⟨w, hw, hcs⟩ := ihr hvu.2.2 hi_r hx_r
        exact ⟨w, Finset.mem_insert_of_mem (Finset.mem_union_right _ hw), hcs⟩
      · push_neg at hx_r
        have hcf_r : cf r = i := le_antisymm (cf_le_of_mem hi_r) (hx_r (cf r) (cf_mem_leaves r))
        have hx_l : x ∈ leaves l := by
          rcases Finset.mem_union.mp hx_u with h | h
          · exact h
          · exfalso; have := hx_r x h; omega
        have hcf_l : cf l < i := lt_of_le_of_lt (cf_le_of_mem hx_l) hxi
        refine ⟨node l r, Finset.mem_insert_self _ _, ?_⟩
        simp [cs, hcf_r, le_of_lt hcf_l]

/-- **Lemma 2 (Upward-Closed Change Set Bound, with the `-1` saving)**:
Let `t₁, t₂` be valid phylogenetic trees on `{0, ..., n - 1}`.
Suppose `B ⊆ internalSubtrees t₁` is nonempty and upward-closed
(`w ∈ B` and `Subtree w v` with `v ∈ internalSubtrees t₁` implies `v ∈ B`),
and every `w ∈ internalSubtrees t₁ \ B` is a subtree of `t₂`.
Then `olaDist n t₁ t₂ ≤ B.card - 1`! -/
theorem olaDist_le_upward_closed {n : ℕ} {t₁ t₂ : BinTree}
    (hp1 : IsPhylogenetic n t₁) (hp2 : IsPhylogenetic n t₂)
    (B : Finset BinTree) (hBsub : B ⊆ internalSubtrees t₁) (hBne : B.Nonempty)
    (hBup : ∀ w ∈ B, ∀ v ∈ internalSubtrees t₁, Subtree w v → v ∈ B)
    (hkeep : ∀ w ∈ internalSubtrees t₁, w ∉ B → Subtree w t₂) :
    olaDist n t₁ t₂ ≤ B.card - 1 := by
  let J : Finset ℕ := B.image cs
  have hJne : J.Nonempty := hBne.image cs
  set jstar := J.min' hJne
  -- Step A: Every i ∈ Ico 1 n with olaEntry t₁ i ≠ olaEntry t₂ i must lie in J!
  have hsubJ : (Finset.Ico 1 n).filter (fun i => olaEntry t₁ i ≠ olaEntry t₂ i) ⊆ J := by
    intro i hi
    rw [Finset.mem_filter, Finset.mem_Ico] at hi
    obtain ⟨⟨hi1, hin⟩, hne⟩ := hi
    obtain ⟨w, hw_mem, hw_cs⟩ := exists_internal_with_cs hp1 hi1 hin
    by_cases hwB : w ∈ B
    · exact Finset.mem_image.mpr ⟨w, hwB, hw_cs⟩
    · exfalso
      obtain ⟨hsub1, c₁, c₂, rfl⟩ := mem_internalSubtrees_iff.mp hw_mem
      have hsub2 : Subtree (node c₁ c₂) t₂ := hkeep (node c₁ c₂) hw_mem hwB
      have heq := olaEntry_eq_of_common_subtree hsub1 hsub2 hp1.1 hp2.1
      rw [hw_cs] at heq
      exact hne heq
  -- Step B: Prove jstar ∉ changed set (i.e. olaEntry t₁ jstar = olaEntry t₂ jstar)!
  have hjstar_eq : jstar ∈ Finset.Ico 1 n → olaEntry t₁ jstar = olaEntry t₂ jstar := by
    intro hjrange
    rw [Finset.mem_Ico] at hjrange
    by_cases hj1 : jstar = 1
    · -- For i = 1: leaves(t|_1) = {0, 1}, so t|_1 is node (leaf 0) (leaf 1) or node (leaf 1) (leaf 0),
      -- and in both cases sisterOf 1 (t|_1) = some 0!
      have h_at1 : ∀ t : BinTree, IsPhylogenetic n t → olaEntry t 1 = some 0 := by
        intro t hpt
        have h01_ne : (leaves t ∩ Finset.range 2).Nonempty := by
          refine ⟨0, Finset.mem_inter.mpr ⟨?_, by decide⟩⟩
          rw [hpt.2]; exact Finset.mem_range.mpr (by omega)
        obtain ⟨t', ht'⟩ := Option.isSome_iff_exists.mp ((restrict_isSome_iff (Finset.range 2) t).mpr h01_ne)
        have hlt' : leaves t' = {0, 1} := by
          rw [restrict_leaves _ ht', hpt.2]
          ext x; simp only [Finset.mem_inter, Finset.mem_range, Finset.mem_insert, Finset.mem_singleton]
          omega
        have hvt' := restrict_valid _ hpt.1 ht'
        have hsis : sisterOf 1 t' = some 0 := by
          cases t' with
          | leaf x =>
            have h0 : 0 ∈ ({x} : Finset ℕ) := by rw [← leaves, hlt']; simp
            have h1 : 1 ∈ ({x} : Finset ℕ) := by rw [← leaves, hlt']; simp
            simp at h0 h1; omega
          | node a b =>
            have hdisj := hvt'.1
            have ha_ne := leaves_nonempty a
            have hb_ne := leaves_nonempty b
            have hsub_a : leaves a ⊆ {0, 1} := hlt' ▸ Finset.subset_union_left
            have hsub_b : leaves b ⊆ {0, 1} := hlt' ▸ Finset.subset_union_right
            -- Each of leaves a, leaves b has card = 1 (since disjoint, nonempty, union = {0, 1})
            have hca : (leaves a).card = 1 := by
              have h1 := Finset.one_le_card.mpr ha_ne
              have h2 := Finset.one_le_card.mpr hb_ne
              have hu : (leaves a ∪ leaves b).card = (leaves a).card + (leaves b).card :=
                Finset.card_union_of_disjoint hdisj
              have htot : (leaves a ∪ leaves b).card = 2 := by rw [← leaves, hlt']; decide
              omega
            have hcb : (leaves b).card = 1 := by
              have h1 := Finset.one_le_card.mpr ha_ne
              have h2 := Finset.one_le_card.mpr hb_ne
              have hu : (leaves a ∪ leaves b).card = (leaves a).card + (leaves b).card :=
                Finset.card_union_of_disjoint hdisj
              have htot : (leaves a ∪ leaves b).card = 2 := by rw [← leaves, hlt']; decide
              omega
            have hleaf_of_card1 : ∀ u : BinTree, IsValid u → (leaves u).card = 1 →
                ∃ x, u = leaf x ∧ leaves u = {x} := by
              intro u hvu hcu
              cases u with
              | leaf x => exact ⟨x, rfl, rfl⟩
              | node p q =>
                exfalso
                have := Finset.one_le_card.mpr (leaves_nonempty p)
                have := Finset.one_le_card.mpr (leaves_nonempty q)
                have := Finset.card_union_of_disjoint hvu.1
                simp [leaves] at hcu; omega
            obtain ⟨xa, rfl, hxa⟩ := hleaf_of_card1 a hvt'.2.1 hca
            obtain ⟨xb, rfl, hxb⟩ := hleaf_of_card1 b hvt'.2.2 hcb
            have hxa_mem : xa ∈ ({0, 1} : Finset ℕ) := hsub_a (by simp [leaves])
            have hxb_mem : xb ∈ ({0, 1} : Finset ℕ) := hsub_b (by simp [leaves])
            have hne_ab : xa ≠ xb := by
              intro heq
              subst heq
              exact Finset.disjoint_left.mp hdisj (by simp [leaves] : xa ∈ leaves (leaf xa))
                (by simp [leaves] : xa ∈ leaves (leaf xa))
            simp only [Finset.mem_insert, Finset.mem_singleton] at hxa_mem hxb_mem
            rcases hxa_mem with rfl | rfl <;> rcases hxb_mem with rfl | rfl
            · exact absurd rfl hne_ab
            · simp [sisterOf, canonLabel]
            · simp [sisterOf, canonLabel]
            · exact absurd rfl hne_ab
        simp [olaEntry, ht', hsis]
      rw [hj1, h_at1 t₁ hp1, h_at1 t₂ hp2]
    · -- jstar ≥ 2: find vstar = lca_{t₁}(range jstar) ∈ internalSubtrees t₁!
      -- By induction on u containing range jstar, there is a minimal subtree vstar of t₁
      -- containing range jstar, so BOTH children of vstar intersect range jstar!
      have hex_vstar : ∃ v ∈ internalSubtrees t₁,
          Finset.range jstar ⊆ leaves v ∧ cs v < jstar := by
        suffices h : ∀ u : BinTree, IsValid u → Finset.range jstar ⊆ leaves u →
            ∃ v ∈ internalSubtrees u, Finset.range jstar ⊆ leaves v ∧ cs v < jstar by
          apply h t₁ hp1.1
          rw [hp1.2]
          exact Finset.range_mono (le_of_lt hjrange.2)
        intro u hvu hsub_u
        induction u with
        | leaf x =>
          exfalso
          have h0 : 0 ∈ leaves (leaf x) := hsub_u (Finset.mem_range.mpr (by omega))
          have h1 : 1 ∈ leaves (leaf x) := hsub_u (Finset.mem_range.mpr (by omega))
          simp [leaves] at h0 h1; omega
        | node l r ihl ihr =>
          by_cases hl : Finset.range jstar ⊆ leaves l
          · obtain ⟨v, hv, hsubv, hcsv⟩ := ihl hvu.2.1 hl
            exact ⟨v, Finset.mem_insert_of_mem (Finset.mem_union_left _ hv), hsubv, hcsv⟩
          · by_cases hr : Finset.range jstar ⊆ leaves r
            · obtain ⟨v, hv, hsubv, hcsv⟩ := ihr hvu.2.2 hr
              exact ⟨v, Finset.mem_insert_of_mem (Finset.mem_union_right _ hv), hsubv, hcsv⟩
            · -- Neither l nor r contains all of range jstar, so BOTH intersect range jstar!
              rw [Finset.not_subset] at hl hr
              obtain ⟨x, hx_rng, hx_notl⟩ := hl
              obtain ⟨y, hy_rng, hy_notr⟩ := hr
              have hx_r : x ∈ leaves r := by
                rcases Finset.mem_union.mp (hsub_u hx_rng) with h | h
                · exact absurd h hx_notl
                · exact h
              have hy_l : y ∈ leaves l := by
                rcases Finset.mem_union.mp (hsub_u hy_rng) with h | h
                · exact h
                · exact absurd h hy_notr
              have hcf_l : cf l < jstar := lt_of_le_of_lt (cf_le_of_mem hy_l) (Finset.mem_range.mp hy_rng)
              have hcf_r : cf r < jstar := lt_of_le_of_lt (cf_le_of_mem hx_r) (Finset.mem_range.mp hx_rng)
              exact ⟨node l r, Finset.mem_insert_self _ _, hsub_u, max_lt hcf_l hcf_r⟩
      obtain ⟨vstar, hvstar_mem, hvstar_rng, hvstar_cs⟩ := hex_vstar
      -- Since cs vstar < jstar = min J, vstar ∉ B!
      have hvstar_notB : vstar ∉ B := by
        intro hmem
        have hle : jstar ≤ cs vstar := Finset.min'_le J (cs vstar) (Finset.mem_image_of_mem cs hmem)
        omega
      -- Therefore vstar is a subtree of t₂ as well!
      have hvstar_sub1 : Subtree vstar t₁ := (mem_internalSubtrees_iff.mp hvstar_mem).1
      have hvstar_sub2 : Subtree vstar t₂ := hkeep vstar hvstar_mem hvstar_notB
      -- Claim: jstar ∉ leaves vstar!
      have hjstar_not_vstar : jstar ∉ leaves vstar := by
        intro hj_in
        -- Pick wstar ∈ B with cs wstar = jstar
        obtain ⟨wstar, hwstar_B, hwstar_cs⟩ := Finset.mem_image.mp (Finset.min'_mem J hJne)
        have hwstar_mem := hBsub hwstar_B
        obtain ⟨hwstar_sub, c₁, c₂, hwstar_eq⟩ := mem_internalSubtrees_iff.mp hwstar_mem
        -- jstar ∈ leaves wstar (since cs wstar = max (cf c₁) (cf c₂) = jstar)
        have hj_in_w : jstar ∈ leaves wstar := by
          rw [hwstar_eq, leaves]
          have hmax : max (cf c₁) (cf c₂) = jstar := by simpa [hwstar_eq, cs] using hwstar_cs
          rcases le_total (cf c₁) (cf c₂) with hle | hle
          · rw [max_eq_right hle] at hmax
            exact Finset.mem_union_right _ (hmax ▸ cf_mem_leaves c₂)
          · rw [max_eq_left hle] at hmax
            exact Finset.mem_union_left _ (hmax ▸ cf_mem_leaves c₁)
        -- Two subtrees of a valid tree sharing a leaf jstar must be comparable (one is a subtree of the other)!
        have hcomp : ∀ u : BinTree, IsValid u → ∀ a b : BinTree,
            Subtree a u → Subtree b u → (leaves a ∩ leaves b).Nonempty →
            Subtree a b ∨ Subtree b a := by
          intro u hvu
          induction u with
          | leaf x =>
            intro a b ha hb _
            cases ha; cases hb; exact Or.inl (Subtree.refl _)
          | node l r ihl ihr =>
            intro a b ha hb hne
            cases ha with
            | refl => exact Or.inr hb
            | left hal =>
              cases hb with
              | refl => exact Or.inl (Subtree.left hal)
              | left hbl => exact ihl hvu.2.1 a b hal hbl hne
              | right hbr =>
                exfalso
                rcases hne with ⟨z, hz⟩
                rw [Finset.mem_inter] at hz
                exact Finset.disjoint_left.mp hvu.1
                  (subtree_leaves_subset hal hz.1) (subtree_leaves_subset hbr hz.2)
            | right har =>
              cases hb with
              | refl => exact Or.inl (Subtree.right har)
              | left hbl =>
                exfalso
                rcases hne with ⟨z, hz⟩
                rw [Finset.mem_inter] at hz
                exact Finset.disjoint_left.mp hvu.1
                  (subtree_leaves_subset hbl hz.2) (subtree_leaves_subset har hz.1)
              | right hbr => exact ihr hvu.2.2 a b har hbr hne
        rcases hcomp t₁ hp1.1 wstar vstar hwstar_sub hvstar_sub1
            ⟨jstar, Finset.mem_inter.mpr ⟨hj_in_w, hj_in⟩⟩ with hw_in_v | hv_in_w
        · -- wstar is a subtree of vstar ⇒ by upward closure, vstar ∈ B, contradiction!
          exact hvstar_notB (hBup wstar hwstar_B vstar hvstar_mem hw_in_v)
        · -- vstar is a subtree of wstar: cannot equal wstar (cs vstar < jstar = cs wstar),
          -- so vstar is a subtree of c₁ or c₂, which then contains range (jstar + 1),
          -- forcing the OTHER child of wstar to have cf > jstar ⇒ cs wstar > jstar!
          have hne_vw : vstar ≠ wstar := fun h => by rw [h, hwstar_cs] at hvstar_cs; omega
          rw [hwstar_eq] at hv_in_w hne_vw
          have hvw_valid := subtree_valid hwstar_sub hp1.1
          rw [hwstar_eq] at hvw_valid
          have hsub_child : Subtree vstar c₁ ∨ Subtree vstar c₂ := by
            cases hv_in_w with
            | refl => exact absurd rfl hne_vw
            | left h => exact Or.inl h
            | right h => exact Or.inr h
          have hrng_j1 : Finset.range (jstar + 1) ⊆ leaves vstar := by
            intro x hx
            rw [Finset.mem_range] at hx
            by_cases hxj : x = jstar
            · rwa [hxj]
            · exact hvstar_rng (Finset.mem_range.mpr (by omega))
          rcases hsub_child with h1 | h2
          · -- range (jstar + 1) ⊆ leaves c₁, so leaves c₂ is disjoint from range (jstar + 1) ⇒ cf c₂ > jstar!
            have hcf2_gt : jstar < cf c₂ := by
              by_contra hle; push_neg at hle
              have hmem1 : cf c₂ ∈ leaves c₁ :=
                subtree_leaves_subset h1 (hrng_j1 (Finset.mem_range.mpr (Nat.lt_succ_of_le hle)))
              exact Finset.disjoint_left.mp hvw_valid.1 hmem1 (cf_mem_leaves c₂)
            have : jstar < cs wstar := by
              rw [hwstar_eq, cs]; exact lt_of_lt_of_le hcf2_gt (le_max_right _ _)
            omega
          · have hcf1_gt : jstar < cf c₁ := by
              by_contra hle; push_neg at hle
              have hmem2 : cf c₁ ∈ leaves c₂ :=
                subtree_leaves_subset h2 (hrng_j1 (Finset.mem_range.mpr (Nat.lt_succ_of_le hle)))
              exact Finset.disjoint_left.mp hvw_valid.1 (cf_mem_leaves c₁) hmem2
            have : jstar < cs wstar := by
              rw [hwstar_eq, cs]; exact lt_of_lt_of_le hcf1_gt (le_max_left _ _)
            omega
      -- Now for ANY phylogenetic tree t on {0..n-1} containing vstar as a subtree:
      -- Since leaves vstar ∩ range (jstar + 1) = range jstar, restrict (range (jstar+1)) t
      -- has top children vstar' (with canonLabel = canonLabel vstar) and leaf jstar!
      -- Therefore olaEntry t jstar = some (canonLabel vstar)!
      have h_ola_vstar : ∀ t : BinTree, IsPhylogenetic n t → Subtree vstar t →
          olaEntry t jstar = some (canonLabel vstar) := by
        intro t hpt hv_sub
        set A := Finset.range (jstar + 1)
        have hv_inter : leaves vstar ∩ A = Finset.range jstar := by
          ext x
          simp only [Finset.mem_inter, Finset.mem_range, A]
          constructor
          · rintro ⟨hx_v, hx_lt⟩
            by_cases hxj : x = jstar
            · subst hxj; exact absurd hx_v hjstar_not_vstar
            · omega
          · intro hx_lt
            exact ⟨hvstar_rng (Finset.mem_range.mpr hx_lt), by omega⟩
        have hv_ne : (leaves vstar ∩ A).Nonempty := by
          rw [hv_inter]; exact ⟨0, Finset.mem_range.mpr (by omega)⟩
        obtain ⟨v', hv'⟩ := Option.isSome_iff_exists.mp ((restrict_isSome_iff A vstar).mpr hv_ne)
        have hv'_leaves : leaves v' = Finset.range jstar := by
          rw [restrict_leaves A hv', hv_inter]
        -- Also canonLabel v' = canonLabel vstar!
        have hv'_canon : canonLabel v' = canonLabel vstar := by
          obtain ⟨_, c₁, c₂, rfl⟩ := mem_internalSubtrees_iff.mp hvstar_mem
          -- Both c₁ and c₂ intersect range jstar (since cs (node c₁ c₂) < jstar)!
          have h1ne : (leaves c₁ ∩ A).Nonempty :=
            ⟨cf c₁, Finset.mem_inter.mpr ⟨cf_mem_leaves c₁,
              Finset.mem_range.mpr (by have := le_max_left (cf c₁) (cf c₂); simp [cs] at hvstar_cs; omega)⟩⟩
          have h2ne : (leaves c₂ ∩ A).Nonempty :=
            ⟨cf c₂, Finset.mem_inter.mpr ⟨cf_mem_leaves c₂,
              Finset.mem_range.mpr (by have := le_max_right (cf c₁) (cf c₂); simp [cs] at hvstar_cs; omega)⟩⟩
          obtain ⟨c₁', hc₁'⟩ := Option.isSome_iff_exists.mp ((restrict_isSome_iff A c₁).mpr h1ne)
          obtain ⟨c₂', hc₂'⟩ := Option.isSome_iff_exists.mp ((restrict_isSome_iff A c₂).mpr h2ne)
          have hveq : v' = node c₁' c₂' := by
            simp [restrict, hc₁', hc₂'] at hv'; exact hv'.symm
          rw [hveq, canonLabel, canonLabel, cs, cs, cf_restrict_range hc₁', cf_restrict_range hc₂']
        obtain ⟨t', ht', hv'_sub⟩ := restrict_subtree A hv_sub hv'
        have ht'_leaves : leaves t' = Finset.range (jstar + 1) := by
          rw [restrict_leaves A ht', hpt.2]
          ext x; simp only [Finset.mem_inter, Finset.mem_range, A]; omega
        have hvt' := restrict_valid A hpt.1 ht'
        -- Since leaves v' = range jstar and leaves t' = range (jstar + 1) = range jstar ∪ {jstar},
        -- t' MUST be `node v' (leaf jstar)` or `node (leaf jstar) v'`!
        have htop : t' = node v' (leaf jstar) ∨ t' = node (leaf jstar) v' := by
          cases ht'_eq : t' with
          | leaf x =>
            exfalso
            have h0 : 0 ∈ ({x} : Finset ℕ) := by rw [← leaves, ← ht'_eq, ht'_leaves]; simp
            have hj : jstar ∈ ({x} : Finset ℕ) := by rw [← leaves, ← ht'_eq, ht'_leaves]; simp
            simp at h0 hj; omega
          | node a b =>
            rw [ht'_eq] at hv'_sub ht'_leaves hvt'
            have hne_v : v' ≠ node a b := by
              intro heq
              have hj : jstar ∈ leaves v' := by rw [heq, ht'_leaves]; simp
              rw [hv'_leaves] at hj; simp at hj
            have hleaf_j : ∀ u : BinTree, IsValid u → leaves u = {jstar} → u = leaf jstar := by
              intro u hvu hlu
              cases u with
              | leaf x => rw [Finset.singleton_injective hlu]
              | node p q =>
                exfalso
                obtain ⟨xp, hxp⟩ := leaves_nonempty p
                obtain ⟨xq, hxq⟩ := leaves_nonempty q
                have h1 : xp = jstar := Finset.mem_singleton.mp (hlu ▸ Finset.mem_union_left _ hxp)
                have h2 : xq = jstar := Finset.mem_singleton.mp (hlu ▸ Finset.mem_union_right _ hxq)
                exact Finset.disjoint_left.mp hvu.1 (h1 ▸ hxp) (h2 ▸ hxq)
            have heq_of_same_leaves : ∀ s u : BinTree, IsValid u → Subtree s u →
                leaves s = leaves u → s = u := by
              intro s u hvu hsu hlu
              cases hsu with
              | refl => rfl
              | @left p q hsp =>
                exfalso
                obtain ⟨xq, hxq⟩ := leaves_nonempty q
                have hxq_s : xq ∈ leaves s := by rw [hlu, leaves]; exact Finset.mem_union_right _ hxq
                exact Finset.disjoint_left.mp hvu.1 (subtree_leaves_subset hsp hxq_s) hxq
              | @right p q hsq =>
                exfalso
                obtain ⟨xp, hxp⟩ := leaves_nonempty p
                have hxp_s : xp ∈ leaves s := by rw [hlu, leaves]; exact Finset.mem_union_left _ hxp
                exact Finset.disjoint_left.mp hvu.1 hxp (subtree_leaves_subset hsq hxp_s)
            cases hv'_sub with
            | refl => exact absurd rfl hne_v
            | left hsub_a =>
              -- range jstar = leaves v' ⊆ leaves a, and leaves b is nonempty and disjoint from leaves a!
              -- Since leaves a ∪ leaves b = range (jstar + 1) = range jstar ∪ {jstar},
              -- leaves b MUST be {jstar} and leaves a MUST be range jstar = leaves v'!
              have hlb : leaves b = {jstar} := by
                ext x
                simp only [Finset.mem_singleton]
                constructor
                · intro hxb
                  have hxr : x < jstar + 1 := Finset.mem_range.mp (ht'_leaves ▸ Finset.mem_union_right _ hxb)
                  by_contra hne
                  have hxa : x ∈ leaves a := subtree_leaves_subset hsub_a (hv'_leaves ▸ Finset.mem_range.mpr (by omega))
                  exact Finset.disjoint_left.mp hvt'.1 hxa hxb
                · rintro rfl
                  obtain ⟨xb, hxb⟩ := leaves_nonempty b
                  have hxr : xb < jstar + 1 := Finset.mem_range.mp (ht'_leaves ▸ Finset.mem_union_right _ hxb)
                  by_cases hxbj : xb = jstar
                  · rwa [← hxbj]
                  · exfalso
                    have hxa : xb ∈ leaves a := subtree_leaves_subset hsub_a (hv'_leaves ▸ Finset.mem_range.mpr (by omega))
                    exact Finset.disjoint_left.mp hvt'.1 hxa hxb
              have hla : leaves v' = leaves a := by
                ext x
                constructor
                · exact fun hx => subtree_leaves_subset hsub_a hx
                · intro hxa
                  have hxr : x < jstar + 1 := Finset.mem_range.mp (ht'_leaves ▸ Finset.mem_union_left _ hxa)
                  have hne : x ≠ jstar := fun heq =>
                    Finset.disjoint_left.mp hvt'.1 hxa (hlb ▸ Finset.mem_singleton.mpr heq)
                  rw [hv'_leaves]; exact Finset.mem_range.mpr (by omega)
              left
              rw [heq_of_same_leaves v' a hvt'.2.1 hsub_a hla, hleaf_j b hvt'.2.2 hlb]
            | right hsub_b =>
              have hla : leaves a = {jstar} := by
                ext x
                simp only [Finset.mem_singleton]
                constructor
                · intro hxa
                  have hxr : x < jstar + 1 := Finset.mem_range.mp (ht'_leaves ▸ Finset.mem_union_left _ hxa)
                  by_contra hne
                  have hxb : x ∈ leaves b := subtree_leaves_subset hsub_b (hv'_leaves ▸ Finset.mem_range.mpr (by omega))
                  exact Finset.disjoint_left.mp hvt'.1 hxa hxb
                · rintro rfl
                  obtain ⟨xa, hxa⟩ := leaves_nonempty a
                  have hxr : xa < jstar + 1 := Finset.mem_range.mp (ht'_leaves ▸ Finset.mem_union_left _ hxa)
                  by_cases hxaj : xa = jstar
                  · rwa [← hxaj]
                  · exfalso
                    have hxb : xa ∈ leaves b := subtree_leaves_subset hsub_b (hv'_leaves ▸ Finset.mem_range.mpr (by omega))
                    exact Finset.disjoint_left.mp hvt'.1 hxa hxb
              have hlb : leaves v' = leaves b := by
                ext x
                constructor
                · exact fun hx => subtree_leaves_subset hsub_b hx
                · intro hxb
                  have hxr : x < jstar + 1 := Finset.mem_range.mp (ht'_leaves ▸ Finset.mem_union_right _ hxb)
                  have hne : x ≠ jstar := fun heq =>
                    Finset.disjoint_left.mp hvt'.1 (hla ▸ Finset.mem_singleton.mpr heq) hxb
                  rw [hv'_leaves]; exact Finset.mem_range.mpr (by omega)
              right
              rw [heq_of_same_leaves v' b hvt'.2.2 hsub_b hlb, hleaf_j a hvt'.2.1 hla]
        have hv'_ne_leaf : v' ≠ leaf jstar := by
          intro heq
          have : 0 ∈ leaves v' := by rw [hv'_leaves]; exact Finset.mem_range.mpr (by omega)
          rw [heq, leaves] at this; simp at this; omega
        simp only [olaEntry, A, ht']
        rcases htop with h | h
        · rw [h]; simp [sisterOf, hv'_ne_leaf, hv'_canon]
        · rw [h]; simp [sisterOf, hv'_canon]
      rw [h_ola_vstar t₁ hp1 hvstar_sub1, h_ola_vstar t₂ hp2 hvstar_sub2]
  -- Step C: Therefore changed set ⊆ J.erase jstar, which has cardinality |J| - 1 ≤ |B| - 1!
  have hsub_erase : (Finset.Ico 1 n).filter (fun i => olaEntry t₁ i ≠ olaEntry t₂ i) ⊆
      J.erase jstar := by
    intro i hi
    rw [Finset.mem_erase]
    refine ⟨?_, hsubJ hi⟩
    intro heq
    subst heq
    rw [Finset.mem_filter] at hi
    exact hi.2 (hjstar_eq hi.1)
  calc olaDist n t₁ t₂
      ≤ (J.erase jstar).card := Finset.card_le_card hsub_erase
    _ = J.card - 1 := Finset.card_erase_of_mem (Finset.min'_mem J hJne)
    _ ≤ B.card - 1 := Nat.sub_le_sub_right Finset.card_image_le 1

/-! ### Theorem 3: Height Bound on Ancestor Paths (`|A(p) ∪ A(v_top)| - 1 ≤ 2*height(T) - 2`) -/

section Ancestors
open Classical

/-- Set of internal ancestors of a subtree `x` inside `t` (empty if `x` is not a subtree of `t`). -/
noncomputable def ancestors (t x : BinTree) : Finset BinTree :=
  (internalSubtrees t).filter (fun v => Subtree x v)

lemma ancestors_subset_internal (t x : BinTree) : ancestors t x ⊆ internalSubtrees t :=
  Finset.filter_subset _ _

lemma ancestors_upward_closed (t x : BinTree) :
    ∀ w ∈ ancestors t x, ∀ v ∈ internalSubtrees t, Subtree w v → v ∈ ancestors t x := by
  intro w hw v hv hwv
  rw [ancestors, Finset.mem_filter] at hw ⊢
  refine ⟨hv, ?_⟩
  have htrans : ∀ {a b c : BinTree}, Subtree a b → Subtree b c → Subtree a c := by
    intro a b c hab hbc
    induction hbc with
    | refl => exact hab
    | left _ ih => exact Subtree.left ih
    | right _ ih => exact Subtree.right ih
  exact htrans hw.2 hwv

lemma card_ancestors_le_height : ∀ (t x : BinTree), IsValid t →
    (ancestors t x).card ≤ height t := by
  intro t x hvt
  induction t with
  | leaf l => simp [ancestors, internalSubtrees, height]
  | node l r ihl ihr =>
    have hcard_disj :
        ((internalSubtrees l).filter (Subtree x)).card +
        ((internalSubtrees r).filter (Subtree x)).card ≤ max (height l) (height r) := by
      by_cases hl : ∃ w ∈ internalSubtrees l, Subtree x w
      · obtain ⟨w0, hw0, hyw0⟩ := hl
        have hrempty : (internalSubtrees r).filter (Subtree x) = ∅ := by
          rw [Finset.filter_eq_empty_iff]
          intro w hw hyw
          obtain ⟨z, hz⟩ := leaves_nonempty x
          have hzl : z ∈ leaves l :=
            subtree_leaves_subset (mem_internalSubtrees_iff.mp hw0).1 (subtree_leaves_subset hyw0 hz)
          have hzr : z ∈ leaves r :=
            subtree_leaves_subset (mem_internalSubtrees_iff.mp hw).1 (subtree_leaves_subset hyw hz)
          exact Finset.disjoint_left.mp hvt.1 hzl hzr
        rw [hrempty, Finset.card_empty, add_zero]
        exact le_trans (ihl hvt.2.1) (le_max_left _ _)
      · push_neg at hl
        have hlempty : (internalSubtrees l).filter (Subtree x) = ∅ := by
          rw [Finset.filter_eq_empty_iff]; exact hl
        rw [hlempty, Finset.card_empty, zero_add]
        exact le_trans (ihr hvt.2.2) (le_max_right _ _)
    have hsub : ancestors (node l r) x ⊆
        insert (node l r) (((internalSubtrees l).filter (Subtree x)) ∪
                           ((internalSubtrees r).filter (Subtree x))) := by
      intro v hv
      simp only [ancestors, internalSubtrees, Finset.mem_filter, Finset.mem_insert,
        Finset.mem_union] at hv ⊢
      rcases hv with ⟨rfl | hl | hr, hxv⟩
      · exact Or.inl rfl
      · exact Or.inr (Or.inl ⟨hl, hxv⟩)
      · exact Or.inr (Or.inr ⟨hr, hxv⟩)
    calc (ancestors (node l r) x).card
        ≤ (insert (node l r) (((internalSubtrees l).filter (Subtree x)) ∪
                              ((internalSubtrees r).filter (Subtree x)))).card :=
          Finset.card_le_card hsub
      _ ≤ 1 + (((internalSubtrees l).filter (Subtree x)) ∪
               ((internalSubtrees r).filter (Subtree x))).card := by
          have := Finset.card_insert_le (node l r)
            (((internalSubtrees l).filter (Subtree x)) ∪ ((internalSubtrees r).filter (Subtree x)))
          omega
      _ ≤ 1 + (((internalSubtrees l).filter (Subtree x)).card +
               ((internalSubtrees r).filter (Subtree x)).card) := by
          have := Finset.card_union_le
            ((internalSubtrees l).filter (Subtree x)) ((internalSubtrees r).filter (Subtree x))
          omega
      _ ≤ 1 + max (height l) (height r) := Nat.add_le_add_left hcard_disj 1

/-- Helper: optional ancestor set (empty when regrafting at the root edge). -/
noncomputable def optAncestors (t : BinTree) : Option BinTree → Finset BinTree
  | none => ∅
  | some v => ancestors t v

/-- **Theorem 3 (Sharp Deterministic rSPR Bound; Proves Conjecture 11 `conj:spr-move-bound`)**:
Let `t₁, t₂` be valid phylogenetic trees on `{0, ..., n - 1}`. If every internal subtree of `t₁`
outside `B = ancestors t₁ p ∪ optAncestors t₁ v_opt` (for internal `p` and either `v_opt = some v`
an internal node of `t₁`, or the root case `v_opt = none`) is preserved as a subtree of `t₂`, then:
- `olaDist n t₁ t₂ ≤ |B| - 1`, and
- `olaDist n t₁ t₂ ≤ 2 * height t₁ - 2`! -/
theorem deterministic_spr_bound {n : ℕ} {t₁ t₂ p : BinTree}
    (hp1 : IsPhylogenetic n t₁) (hp2 : IsPhylogenetic n t₂)
    (hp_int : p ∈ internalSubtrees t₁)
    (v_opt : Option BinTree)
    (hv_int : ∀ v, v_opt = some v → v ∈ internalSubtrees t₁)
    (hkeep : ∀ w ∈ internalSubtrees t₁,
      w ∉ (ancestors t₁ p ∪ optAncestors t₁ v_opt) → Subtree w t₂) :
    let B := ancestors t₁ p ∪ optAncestors t₁ v_opt
    olaDist n t₁ t₂ ≤ B.card - 1 ∧ olaDist n t₁ t₂ ≤ 2 * height t₁ - 2 := by
  intro B
  have hBsub : B ⊆ internalSubtrees t₁ := by
    refine Finset.union_subset (ancestors_subset_internal t₁ p) ?_
    cases v_opt with
    | none => exact Finset.empty_subset _
    | some v => exact ancestors_subset_internal t₁ v
  have hp_in_anc : p ∈ ancestors t₁ p :=
    Finset.mem_filter.mpr ⟨hp_int, Subtree.refl p⟩
  have hBne : B.Nonempty := ⟨p, Finset.mem_union_left _ hp_in_anc⟩
  have hBup : ∀ w ∈ B, ∀ v ∈ internalSubtrees t₁, Subtree w v → v ∈ B := by
    intro w hw v hv hwv
    rcases Finset.mem_union.mp hw with h1 | h2
    · exact Finset.mem_union_left _ (ancestors_upward_closed t₁ p w h1 v hv hwv)
    · refine Finset.mem_union_right _ ?_
      cases v_opt with
      | none => exact absurd h2 (Finset.notMem_empty w)
      | some u => exact ancestors_upward_closed t₁ u w h2 v hv hwv
  have h1 : olaDist n t₁ t₂ ≤ B.card - 1 :=
    olaDist_le_upward_closed hp1 hp2 B hBsub hBne hBup hkeep
  refine ⟨h1, ?_⟩
  have ht1_int : t₁ ∈ internalSubtrees t₁ := by
    cases t₁ with
    | leaf x => simp [internalSubtrees] at hp_int
    | node l r => exact Finset.mem_insert_self _ _
  have ht1_anc_p : t₁ ∈ ancestors t₁ p :=
    Finset.mem_filter.mpr ⟨ht1_int, (mem_internalSubtrees_iff.mp hp_int).1⟩
  have hcp : (ancestors t₁ p).card ≤ height t₁ := card_ancestors_le_height t₁ p hp1.1
  have hBcard : B.card ≤ 2 * height t₁ - 1 := by
    cases v_opt with
    | none =>
      have : B = ancestors t₁ p := by simp [B, optAncestors]
      rw [this]
      have hpos : 1 ≤ height t₁ := by
        have := Finset.card_pos.mpr ⟨p, hp_in_anc⟩
        omega
      omega
    | some v =>
      have hv_mem := hv_int v rfl
      have ht1_anc_v : t₁ ∈ ancestors t₁ v :=
        Finset.mem_filter.mpr ⟨ht1_int, (mem_internalSubtrees_iff.mp hv_mem).1⟩
      have hcv : (ancestors t₁ v).card ≤ height t₁ := card_ancestors_le_height t₁ v hp1.1
      have hinter_pos : 1 ≤ (ancestors t₁ p ∩ ancestors t₁ v).card :=
        Finset.one_le_card.mpr ⟨t₁, Finset.mem_inter.mpr ⟨ht1_anc_p, ht1_anc_v⟩⟩
      have hu := Finset.card_union_add_card_inter (ancestors t₁ p) (ancestors t₁ v)
      have hBeq : B = ancestors t₁ p ∪ ancestors t₁ v := rfl
      rw [hBeq]
      omega
  omega

end Ancestors

end BinTree
end PhyloOLA
