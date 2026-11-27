import Mathlib

/-!
# Unbounded Separation Between `UB` and `ILB` for Temporal Feedback Graphs
(Gatmiry and Schneider, COLT 2024 Conjecture 12 / arXiv:2407.00571 Conjecture 1 `conj:tight`)

We formalize in Lean 4 + Mathlib (complete kernel-verified proofs, standard foundations only) the complete disproof of
Conjecture `conj:tight`, proving that no universal constant `γ` bounds `UB(S) / ILB(S)` across
temporal feedback graphs:

1. **Lexicographic triangle powers `S_k` on `Fin k → ZMod 3`** (`lexEdge`):
   - `lexEdge_irrefl`, `lexEdge_asymm`, `lexEdge_total`: `S_k` is a tournament for every `k`.
   - `lexEdge_add_right`: coordinatewise translation is a vertex-transitive automorphism group.
   - `lexEdge_indeg_eq`, `lexEdge_outdeg_eq`: every vertex has exact indegree and outdegree
     `d_k = (3^k - 1) / 2` (for every `k ≥ 1`).
2. **Exact maximum directed order length `a_k = 2^k`**:
   - `order_length_le_two_pow`: every directed order (transitive subtournament) in `S_k` has
     length at most `2^k`.
   - `exists_order_two_pow`: there exists a directed order in `S_k` of exact length `2^k`,
     which is necessarily maximal (`exists_maximal_order_two_pow`).
3. **Exact `UB` and `ILB` program values**:
   - `ubCost_ge`: every feasible `UB` assignment on `S_k` has cost at least `3^k / √(2^k)`.
   - `ub_exact`: the translation-orbit assignment on maximal orders achieves `3^k / √(2^k)`,
     giving `IsLeast (ubValues (lexEdge k)) (3^k / √(2^k))` and `UB(S_k) = 3^k / √(2^k)`.
   - `indepSet_card_eq_one`: in any tournament, every nonempty independent set is a singleton.
   - `ilbObj_le`: every feasible `ILB` assignment on `S_k` (`k ≥ 1`) has objective at most
     `3^k / √((3^k - 1) / 2)`.
   - `ilb_exact`: the uniform singleton assignment `z_{s} = 1 / d_k` achieves
     `3^k / √((3^k - 1) / 2)`, giving `IsGreatest (ilbValues (lexEdge k)) (3^k / √((3^k - 1)/2))`
     and `ILB(S_k) = 3^k / √((3^k - 1) / 2)`.
4. **Refutation of `conj:tight`**:
   - `ub_div_ilb_eq`: `UB(S_k) / ILB(S_k) = √((3^k - 1) / 2^(k + 1))` for all `k ≥ 1`.
   - `refute_conj_tight`: for every `γ : ℝ`, there exists `k ≥ 1` with
     `UB(S_k) / ILB(S_k) > γ`.

Machine-checked vs prose-only:
- Machine-checked: The entire disproof of Conjecture `conj:tight` (Lemma 1 and Theorem 1 of
  `solutions/temporal_feedback_regret/solution.md`), including full tournament regularity,
  exact maximum order length `2^k`, exact `UB` and `ILB` optimization values, exact ratio
  formula, and unboundedness.
- Prose-only (in `solutions/temporal_feedback_regret/proof.pdf`): General vertex-transitive
  Proposition 2 for non-tournaments and the numerical verification suite on 246 circulant graphs.
-/

open BigOperators Finset

namespace TemporalFeedbackRegret

/-! ### 1. General Temporal Feedback Graph Definitions and Convex Programs -/

/-- A directed order (transitive subtournament sequence) in a feedback graph `edge`
(`s → t` means `s ∈ S_t`): distinct vertices with `t_i → t_j` for all `i < j`. -/
def IsOrder {V : Type*} (edge : V → V → Prop) (C : List V) : Prop :=
  C.Nodup ∧ C.Pairwise edge

/-- An order `C` is maximal if no strictly longer order contains `C` as a subsequence. -/
def IsMaximalOrder {V : Type*} (edge : V → V → Prop) (C : List V) : Prop :=
  IsOrder edge C ∧ ∀ C', IsOrder edge C' → C.Sublist C' → C'.length = C.length

/-- A feasible `UB` assignment supported on a finite collection `supp` of valid orders. -/
structure UBFeasible {V : Type*} [DecidableEq V] (edge : V → V → Prop) where
  supp : Finset (List V)
  valid : ∀ C ∈ supp, IsOrder edge C
  w : List V → V → ℝ
  nonneg : ∀ C t, 0 ≤ w C t
  supported : ∀ C t, t ∉ C → w C t = 0
  cover : ∀ t : V, ∑ C ∈ supp, w C t = 1

/-- Objective value `∑_{C} ‖λ_C‖_2` of a feasible `UB` assignment. -/
noncomputable def ubCost {V : Type*} [DecidableEq V] {edge : V → V → Prop}
    (u : UBFeasible edge) : ℝ :=
  ∑ C ∈ u.supp, Real.sqrt (∑ t ∈ C.toFinset, (u.w C t) ^ 2)

/-- Set of achievable `UB` objective values over all feasible assignments. -/
def ubValues {V : Type*} [DecidableEq V] (edge : V → V → Prop) : Set ℝ :=
  { val | ∃ u : UBFeasible edge, ubCost u = val }

/-- Set of achievable `UB` objective values restricted to maximal orders only. -/
def ubMaximalValues {V : Type*} [DecidableEq V] (edge : V → V → Prop) : Set ℝ :=
  { val | ∃ u : UBFeasible edge, (∀ C ∈ u.supp, IsMaximalOrder edge C) ∧ ubCost u = val }

/-- Infimum value `UB(S)` of the upper-bound program. -/
noncomputable def UB {V : Type*} [DecidableEq V] (edge : V → V → Prop) : ℝ :=
  sInf (ubValues edge)

/-- Nonempty independent sets of the underlying undirected graph of `edge`. -/
def IndepSet {V : Type*} (edge : V → V → Prop) :=
  { I : Finset V // I.Nonempty ∧ ∀ u ∈ I, ∀ v ∈ I, ¬ edge u v }

instance instFintypeIndepSet {V : Type*} [Fintype V] [DecidableEq V]
    (edge : V → V → Prop) [DecidableRel edge] : Fintype (IndepSet edge) := by
  unfold IndepSet
  infer_instance

/-- A feasible `ILB` assignment `z_I ≥ 0` with `∑_{I : I ∩ S_t ≠ ∅} z_I ≤ 1` for every `t`. -/
structure ILBFeasible {V : Type*} [Fintype V] [DecidableEq V]
    (edge : V → V → Prop) [DecidableRel edge] where
  z : IndepSet edge → ℝ
  nonneg : ∀ I, 0 ≤ z I
  constr : ∀ t : V,
    ∑ I : IndepSet edge, (if ∃ s ∈ I.1, edge s t then z I else 0) ≤ 1

/-- Objective value `∑_t √(∑_{I ∋ t} z_I)` of a feasible `ILB` assignment. -/
noncomputable def ilbObj {V : Type*} [Fintype V] [DecidableEq V]
    {edge : V → V → Prop} [DecidableRel edge] (u : ILBFeasible edge) : ℝ :=
  ∑ t : V, Real.sqrt (∑ I : IndepSet edge, if t ∈ I.1 then u.z I else 0)

/-- Set of achievable `ILB` objective values. -/
def ilbValues {V : Type*} [Fintype V] [DecidableEq V]
    (edge : V → V → Prop) [DecidableRel edge] : Set ℝ :=
  { val | ∃ u : ILBFeasible edge, ilbObj u = val }

/-- Supremum value `ILB(S)` of the independent-set lower-bound program. -/
noncomputable def ILB {V : Type*} [Fintype V] [DecidableEq V]
    (edge : V → V → Prop) [DecidableRel edge] : ℝ :=
  sSup (ilbValues edge)

/-! ### 2. The Lexicographic Triangle Power `S_k` on `Fin k → ZMod 3` -/

/-- Vertex set `V_k = (ℤ / 3ℤ)^k` of size `3^k`. -/
abbrev Vert (k : ℕ) := Fin k → ZMod 3

/-- Directed edge relation `x → y` in the `k`-fold lexicographic power of `C_3`:
at the first coordinate `i` where `x i ≠ y i`, we have `y i - x i = 1` in `ZMod 3`. -/
def lexEdge (k : ℕ) (x y : Vert k) : Prop :=
  ∃ i : Fin k, (∀ j : Fin k, j < i → x j = y j) ∧ y i - x i = 1

instance instDecidableRelLexEdge (k : ℕ) : DecidableRel (lexEdge k) := by
  intro x y
  unfold lexEdge
  infer_instance

lemma zmod3_cases (a : ZMod 3) : a = 0 ∨ a = 1 ∨ a = 2 := by
  fin_cases a <;> simp

lemma lexEdge_irrefl (k : ℕ) (x : Vert k) : ¬ lexEdge k x x := by
  rintro ⟨i, _, hi⟩
  simp at hi

lemma lexEdge_asymm (k : ℕ) {x y : Vert k} (hxy : lexEdge k x y) : ¬ lexEdge k y x := by
  rintro ⟨j, hj_eq, hj⟩
  rcases hxy with ⟨i, hi_eq, hi⟩
  rcases lt_trichotomy i j with hij | rfl | hji
  · have hEq : y i = x i := hj_eq i hij
    rw [hEq, sub_self] at hi
    exact zero_ne_one hi
  · have hsum : (y i - x i) + (x i - y i) = 0 := by ring
    rw [hi, hj] at hsum
    exact absurd hsum (by decide)
  · have hEq : x j = y j := hi_eq j hji
    rw [hEq, sub_self] at hj
    exact zero_ne_one hj

lemma lexEdge_total (k : ℕ) {x y : Vert k} (hne : x ≠ y) :
    lexEdge k x y ∨ lexEdge k y x := by
  have hex : ∃ i : Fin k, x i ≠ y i := by
    by_contra hall
    push_neg at hall
    exact hne (funext hall)
  classical
  let S : Finset (Fin k) := Finset.univ.filter (fun i => x i ≠ y i)
  have hS : S.Nonempty := by
    rcases hex with ⟨i, hi⟩
    exact ⟨i, Finset.mem_filter.mpr ⟨Finset.mem_univ i, hi⟩⟩
  let i := S.min' hS
  have hi_mem : i ∈ S := Finset.min'_mem S hS
  have hi_ne : x i ≠ y i := (Finset.mem_filter.mp hi_mem).2
  have hlt : ∀ j : Fin k, j < i → x j = y j := by
    intro j hj
    by_contra hjne
    have hj_mem : j ∈ S := Finset.mem_filter.mpr ⟨Finset.mem_univ j, hjne⟩
    exact not_le_of_gt hj (Finset.min'_le S j hj_mem)
  have hsub_ne : y i - x i ≠ 0 := sub_ne_zero.mpr (Ne.symm hi_ne)
  rcases zmod3_cases (y i - x i) with h0 | h1 | h2
  · exact absurd h0 hsub_ne
  · exact Or.inl ⟨i, hlt, h1⟩
  · right
    refine ⟨i, fun j hj => (hlt j hj).symm, ?_⟩
    have : x i - y i = -(y i - x i) := by ring
    rw [this, h2]
    decide

lemma lexEdge_add_right (k : ℕ) (c x y : Vert k) :
    lexEdge k (x + c) (y + c) ↔ lexEdge k x y := by
  constructor <;> rintro ⟨i, heq, hsub⟩ <;> refine ⟨i, ?_, ?_⟩
  · intro j hj
    have h := heq j hj
    simpa [Pi.add_apply] using h
  · simpa [Pi.add_apply] using hsub
  · intro j hj
    simp [Pi.add_apply, heq j hj]
  · simp [Pi.add_apply]
    exact hsub

lemma lexEdge_neg_iff (k : ℕ) (x y : Vert k) :
    lexEdge k (-x) (-y) ↔ lexEdge k y x := by
  constructor <;> rintro ⟨i, heq, hsub⟩ <;> refine ⟨i, ?_, ?_⟩
  · intro j hj
    have h := heq j hj
    simpa [Pi.neg_apply] using h.symm
  · have h : -y i - -x i = x i - y i := by ring
    simpa [Pi.neg_apply, h] using hsub
  · intro j hj
    simp [Pi.neg_apply, (heq j hj).symm]
  · simp [Pi.neg_apply]
    have h : -y i + x i = x i - y i := by ring
    rw [h]
    exact hsub

lemma card_vert (k : ℕ) : Fintype.card (Vert k) = 3 ^ k := by
  simp [Vert]

/-- Exact indegree `(3^k - 1) / 2` of every vertex in `S_k`. -/
theorem lexEdge_indeg_eq (k : ℕ) (t : Vert k) :
    (Finset.univ.filter (fun s : Vert k => lexEdge k s t)).card = (3 ^ k - 1) / 2 := by
  -- Translate to `t = 0`
  have htrans : (Finset.univ.filter (fun s : Vert k => lexEdge k s t)).card =
      (Finset.univ.filter (fun s : Vert k => lexEdge k s 0)).card := by
    refine Finset.card_bij (fun s _ => s - t) ?_ ?_ ?_
    · intro s hs
      rw [Finset.mem_filter] at hs ⊢
      refine ⟨Finset.mem_univ _, ?_⟩
      have := (lexEdge_add_right k (-t) s t).mpr hs.2
      simpa [sub_eq_add_neg] using this
    · intro s₁ _ s₂ _ h
      exact sub_left_inj.mp h
    · intro s hs
      rw [Finset.mem_filter] at hs
      refine ⟨s + t, ?_, by simp⟩
      rw [Finset.mem_filter]
      refine ⟨Finset.mem_univ _, ?_⟩
      have := (lexEdge_add_right k t s 0).mpr hs.2
      simpa using this
  rw [htrans]
  let A := Finset.univ.filter (fun s : Vert k => lexEdge k s 0)
  let B := Finset.univ.filter (fun s : Vert k => lexEdge k 0 s)
  have hcard_eq : A.card = B.card := by
    refine Finset.card_bij (fun s _ => -s) ?_ ?_ ?_
    · intro s hs
      rw [Finset.mem_filter] at hs ⊢
      refine ⟨Finset.mem_univ _, ?_⟩
      simpa using (lexEdge_neg_iff k 0 s).mpr hs.2
    · intro s₁ _ s₂ _ h
      exact neg_inj.mp h
    · intro s hs
      rw [Finset.mem_filter] at hs
      refine ⟨-s, ?_, neg_neg s⟩
      rw [Finset.mem_filter]
      exact ⟨Finset.mem_univ _, by simpa using (lexEdge_neg_iff k s 0).mpr hs.2⟩
  have hdisj : Disjoint A B := by
    rw [Finset.disjoint_left]
    intro s hsA hsB
    exact lexEdge_asymm k (Finset.mem_filter.mp hsA).2 (Finset.mem_filter.mp hsB).2
  have hunion : A ∪ B = Finset.univ.erase (0 : Vert k) := by
    ext s
    simp only [A, B, Finset.mem_union, Finset.mem_filter, Finset.mem_univ, true_and,
      Finset.mem_erase]
    constructor
    · rintro (h | h)
      · exact ⟨fun h0 => lexEdge_irrefl k 0 (h0 ▸ h), trivial⟩
      · exact ⟨fun h0 => lexEdge_irrefl k 0 (h0 ▸ h), trivial⟩
    · rintro ⟨hne, _⟩
      exact lexEdge_total k hne
  have hsum : A.card + B.card = 3 ^ k - 1 := by
    rw [← Finset.card_union_of_disjoint hdisj, hunion,
      Finset.card_erase_of_mem (Finset.mem_univ _), Finset.card_univ, card_vert]
  change A.card = (3 ^ k - 1) / 2
  omega

/-- Exact outdegree `(3^k - 1) / 2` of every vertex in `S_k`. -/
theorem lexEdge_outdeg_eq (k : ℕ) (s : Vert k) :
    (Finset.univ.filter (fun t : Vert k => lexEdge k s t)).card = (3 ^ k - 1) / 2 := by
  have hbij : (Finset.univ.filter (fun t : Vert k => lexEdge k s t)).card =
      (Finset.univ.filter (fun t : Vert k => lexEdge k t (-s))).card := by
    refine Finset.card_bij (fun t _ => -t) ?_ ?_ ?_
    · intro t ht
      rw [Finset.mem_filter] at ht ⊢
      exact ⟨Finset.mem_univ _, (lexEdge_neg_iff k t s).mpr ht.2⟩
    · intro t₁ _ t₂ _ h
      exact neg_inj.mp h
    · intro t ht
      rw [Finset.mem_filter] at ht
      refine ⟨-t, ?_, neg_neg t⟩
      rw [Finset.mem_filter]
      exact ⟨Finset.mem_univ _, by simpa using (lexEdge_neg_iff k (-t) s).mp (by simpa using ht.2)⟩
  rw [hbij, lexEdge_indeg_eq]

lemma two_mul_deg (k : ℕ) : 2 * ((3 ^ k - 1) / 2) = 3 ^ k - 1 := by
  have hodd : Odd (3 ^ k) := Odd.pow (by decide : Odd 3)
  rcases hodd with ⟨m, hm⟩
  omega

/-! ### 3. Maximum Directed Order Length `a_k = 2^k` -/

lemma lexEdge_cons_eq {k : ℕ} (a : ZMod 3) (x y : Vert k) :
    lexEdge (k + 1) (Fin.cons a x) (Fin.cons a y) ↔ lexEdge k x y := by
  constructor
  · rintro ⟨i, heq, hsub⟩
    cases i using Fin.cases with
    | zero =>
      simp at hsub
    | succ i' =>
      refine ⟨i', ?_, ?_⟩
      · intro j hj
        have := heq j.succ (Fin.succ_lt_succ_iff.mpr hj)
        simpa using this
      · simpa using hsub
  · rintro ⟨i, heq, hsub⟩
    refine ⟨i.succ, ?_, ?_⟩
    · intro j hj
      cases j using Fin.cases with
      | zero => simp
      | succ j' =>
        simpa using heq j' (Fin.succ_lt_succ_iff.mp hj)
    · simpa using hsub

lemma lexEdge_head_step {k : ℕ} {x y : Vert (k + 1)} (h : y 0 - x 0 = 1) :
    lexEdge (k + 1) x y := by
  refine ⟨0, ?_, h⟩
  intro j hj
  exact absurd hj (Fin.not_lt_zero j)

lemma lexEdge_tail_of_head_eq {k : ℕ} {x y : Vert (k + 1)}
    (h0 : x 0 = y 0) (hxy : lexEdge (k + 1) x y) :
    lexEdge k (Fin.tail x) (Fin.tail y) := by
  have hx : x = Fin.cons (x 0) (Fin.tail x) := (Fin.cons_self_tail x).symm
  have hy : y = Fin.cons (x 0) (Fin.tail y) := by rw [h0]; exact (Fin.cons_self_tail y).symm
  rw [hx, hy, lexEdge_cons_eq] at hxy
  exact hxy

/-- No directed order in `S_{k+1}` can intersect all three first-coordinate blocks `0, 1, 2`. -/
lemma order_misses_block {k : ℕ} {C : List (Vert (k + 1))} (hC : IsOrder (lexEdge (k + 1)) C) :
    ∃ r : ZMod 3, ∀ x ∈ C, x 0 ≠ r := by
  by_contra hall
  push_neg at hall
  rcases hall 0 with ⟨u, hu, hu0⟩
  rcases hall 1 with ⟨v, hv, hv0⟩
  rcases hall 2 with ⟨w, hw, hw0⟩
  have huv : lexEdge (k + 1) u v := lexEdge_head_step (by rw [hu0, hv0]; decide)
  have hvw : lexEdge (k + 1) v w := lexEdge_head_step (by rw [hv0, hw0]; decide)
  have hwu : lexEdge (k + 1) w u := lexEdge_head_step (by rw [hw0, hu0]; decide)
  -- In a `Pairwise (lexEdge (k+1))` list, any two distinct elements are ordered by their index;
  -- specifically, for `a, b ∈ C` with `a ≠ b`, either `a` precedes `b` (`lexEdge a b`) or vice versa.
  -- Consider the indices of `u, v, w` in `C`:
  have idx_edge : ∀ a ∈ C, ∀ b ∈ C, C.idxOf a < C.idxOf b → lexEdge (k + 1) a b := by
    intro a ha b hb hlt
    have hpw := List.pairwise_iff_getElem.mp hC.2
    have hia : C.idxOf a < C.length := List.idxOf_lt_length_iff.mpr ha
    have hib : C.idxOf b < C.length := List.idxOf_lt_length_iff.mpr hb
    have := hpw (C.idxOf a) (C.idxOf b) hia hib hlt
    simpa [List.getElem_idxOf] using this
  have hne_uv : u ≠ v := fun h => by rw [h, hv0] at hu0; exact absurd hu0 (by decide)
  have hne_vw : v ≠ w := fun h => by rw [h, hw0] at hv0; exact absurd hv0 (by decide)
  have hne_wu : w ≠ u := fun h => by rw [h, hu0] at hw0; exact absurd hw0 (by decide)
  have hiu_ne_iv : C.idxOf u ≠ C.idxOf v := fun h =>
    hne_uv ((List.idxOf_inj hu hv).mp h)
  have hiv_ne_iw : C.idxOf v ≠ C.idxOf w := fun h =>
    hne_vw ((List.idxOf_inj hv hw).mp h)
  have hiw_ne_iu : C.idxOf w ≠ C.idxOf u := fun h =>
    hne_wu ((List.idxOf_inj hw hu).mp h)
  -- Since `u → v`, we cannot have `idx v < idx u`, so `idx u < idx v`. Similarly `idx v < idx w` and `idx w < idx u`!
  have h1 : C.idxOf u < C.idxOf v := by
    rcases lt_or_gt_of_ne hiu_ne_iv with h | h
    · exact h
    · exact absurd (idx_edge v hv u hu h) (lexEdge_asymm (k + 1) huv)
  have h2 : C.idxOf v < C.idxOf w := by
    rcases lt_or_gt_of_ne hiv_ne_iw with h | h
    · exact h
    · exact absurd (idx_edge w hw v hv h) (lexEdge_asymm (k + 1) hvw)
  have h3 : C.idxOf w < C.idxOf u := by
    rcases lt_or_gt_of_ne hiw_ne_iu with h | h
    · exact h
    · exact absurd (idx_edge u hu w hw h) (lexEdge_asymm (k + 1) hwu)
  omega

/-- Every directed order in `S_k` has length at most `2^k`. -/
theorem order_length_le_two_pow :
    ∀ (k : ℕ) (C : List (Vert k)), IsOrder (lexEdge k) C → C.length ≤ 2 ^ k
  | 0, C, hC => by
    have hcard : C.toFinset.card ≤ Fintype.card (Vert 0) := Finset.card_le_univ _
    rw [List.toFinset_card_of_nodup hC.1, card_vert] at hcard
    simpa using hcard
  | k + 1, C, hC => by
    have hblock : ∀ r : ZMod 3, (C.filter (fun x => x 0 == r)).length ≤ 2 ^ k := by
      intro r
      let Cr := C.filter (fun x => x 0 == r)
      have hCr_sub : Cr.Sublist C := List.filter_sublist
      have hCr_order : IsOrder (lexEdge (k + 1)) Cr :=
        ⟨hC.1.sublist hCr_sub, hC.2.sublist hCr_sub⟩
      have hCr_head : ∀ x ∈ Cr, x 0 = r := by
        intro x hx
        exact beq_iff_eq.mp (List.mem_filter.mp hx).2
      have htail_order : IsOrder (lexEdge k) (Cr.map Fin.tail) := by
        constructor
        · rw [List.nodup_map_iff_inj_on hCr_order.1]
          intro x hx y hy hxy
          rw [← Fin.cons_self_tail x, ← Fin.cons_self_tail y, hCr_head x hx, hCr_head y hy, hxy]
        · rw [List.pairwise_map]
          exact hCr_order.2.imp_of_mem fun {x y} hx hy hxy =>
            lexEdge_tail_of_head_eq ((hCr_head x hx).trans (hCr_head y hy).symm) hxy
      have hle := order_length_le_two_pow k (Cr.map Fin.tail) htail_order
      simpa using hle
    have hpart : ∀ L : List (Vert (k + 1)),
        L.length = (L.filter (fun x => x 0 == (0 : ZMod 3))).length +
          (L.filter (fun x => x 0 == (1 : ZMod 3))).length +
          (L.filter (fun x => x 0 == (2 : ZMod 3))).length := by
      intro L
      induction L with
      | nil => simp
      | cons a t ih =>
        rcases zmod3_cases (a 0) with ha | ha | ha <;>
          · simp only [List.filter_cons, ha, List.length_cons, beq_self_eq_true, ↓reduceIte]
            have h01 : ((0 : ZMod 3) == 1) = false := by decide
            have h02 : ((0 : ZMod 3) == 2) = false := by decide
            have h10 : ((1 : ZMod 3) == 0) = false := by decide
            have h12 : ((1 : ZMod 3) == 2) = false := by decide
            have h20 : ((2 : ZMod 3) == 0) = false := by decide
            have h21 : ((2 : ZMod 3) == 1) = false := by decide
            simp only [h01, h02, h10, h12, h20, h21, Bool.false_eq_true, ↓reduceIte]
            omega
    rcases order_misses_block hC with ⟨r, hr⟩
    have hzero : (C.filter (fun x => x 0 == r)).length = 0 := by
      rw [List.length_eq_zero_iff, List.filter_eq_nil_iff]
      intro x hx
      simp [hr x hx]
    have h0 := hblock 0
    have h1 := hblock 1
    have h2 := hblock 2
    have hp := hpart C
    rcases zmod3_cases r with rfl | rfl | rfl <;>
      · rw [hzero] at hp
        have : 2 ^ (k + 1) = 2 ^ k + 2 ^ k := by ring
        omega

/-- Explicit canonical maximum order in `S_k` of exact length `2^k`. -/
def maxOrder : (k : ℕ) → List (Vert k)
  | 0 => [Fin.elim0]
  | k + 1 =>
    (maxOrder k).map (Fin.cons (0 : ZMod 3)) ++ (maxOrder k).map (Fin.cons (1 : ZMod 3))

lemma maxOrder_length (k : ℕ) : (maxOrder k).length = 2 ^ k := by
  induction k with
  | zero => rfl
  | succ k ih =>
    simp [maxOrder, ih, pow_succ]
    ring

lemma maxOrder_isOrder (k : ℕ) : IsOrder (lexEdge k) (maxOrder k) := by
  induction k with
  | zero =>
    exact ⟨by simp [maxOrder], by simp [maxOrder]⟩
  | succ k ih =>
    simp only [maxOrder]
    have hinj : ∀ a : ZMod 3, Function.Injective (fun x : Vert k => Fin.cons (α := fun _ => ZMod 3) a x) := by
      intro a x y h
      ext i
      have := congr_fun h i.succ
      simpa using this
    constructor
    · rw [List.nodup_append]
      refine ⟨ih.1.map (hinj 0), ih.1.map (hinj 1), ?_⟩
      intro x hx y hy hxy
      rcases List.mem_map.mp hx with ⟨x', _, rfl⟩
      rcases List.mem_map.mp hy with ⟨y', _, rfl⟩
      have h0 := congr_fun hxy 0
      simp at h0
    · rw [List.pairwise_append]
      refine ⟨?_, ?_, ?_⟩
      · rw [List.pairwise_map]
        exact ih.2.imp fun h => (lexEdge_cons_eq 0 _ _).mpr h
      · rw [List.pairwise_map]
        exact ih.2.imp fun h => (lexEdge_cons_eq 1 _ _).mpr h
      · intro x hx y hy
        rcases List.mem_map.mp hx with ⟨x', _, rfl⟩
        rcases List.mem_map.mp hy with ⟨y', _, rfl⟩
        exact lexEdge_head_step (by simp)

/-- `maxOrder k` is a maximal directed order of length `2^k`. -/
theorem maxOrder_isMaximal (k : ℕ) : IsMaximalOrder (lexEdge k) (maxOrder k) := by
  refine ⟨maxOrder_isOrder k, fun C' hC' hsub => ?_⟩
  have hle : C'.length ≤ 2 ^ k := order_length_le_two_pow k C' hC'
  have hge : 2 ^ k ≤ C'.length := by
    rw [← maxOrder_length k]
    exact hsub.length_le
  rw [maxOrder_length k]
  omega

/-- Existence of a maximal directed order of exact length `2^k`. -/
theorem exists_maximal_order_two_pow (k : ℕ) :
    ∃ C : List (Vert k), IsMaximalOrder (lexEdge k) C ∧ C.length = 2 ^ k :=
  ⟨maxOrder k, maxOrder_isMaximal k, maxOrder_length k⟩

/-! ### 4. Exact Value of `UB(S_k) = 3^k / √(2^k)` -/

/-- Cauchy-Schwarz helper: `(∑_{t ∈ S} f t) ≤ √(|S|) * √(∑_{t ∈ S} (f t)^2)`. -/
lemma sum_le_sqrt_card_mul_l2 {α : Type*} (S : Finset α) (f : α → ℝ) :
    ∑ t ∈ S, f t ≤ Real.sqrt (S.card : ℝ) * Real.sqrt (∑ t ∈ S, (f t) ^ 2) := by
  have hcs := Finset.sum_mul_sq_le_sq_mul_sq S (fun _ => (1 : ℝ)) f
  simp only [one_mul, one_pow, Finset.sum_const, nsmul_eq_mul, mul_one] at hcs
  have hnonneg : 0 ≤ (S.card : ℝ) * ∑ t ∈ S, (f t) ^ 2 :=
    mul_nonneg (Nat.cast_nonneg _) (Finset.sum_nonneg fun _ _ => sq_nonneg _)
  have hle : ∑ t ∈ S, f t ≤ Real.sqrt ((S.card : ℝ) * ∑ t ∈ S, (f t) ^ 2) := by
    calc ∑ t ∈ S, f t ≤ |∑ t ∈ S, f t| := le_abs_self _
      _ = Real.sqrt ((∑ t ∈ S, f t) ^ 2) := (Real.sqrt_sq_eq_abs _).symm
      _ ≤ Real.sqrt ((S.card : ℝ) * ∑ t ∈ S, (f t) ^ 2) := Real.sqrt_le_sqrt hcs
  rw [Real.sqrt_mul (Nat.cast_nonneg _)] at hle
  exact hle

/-- Universal lower bound on `UB(S_k)`: every feasible assignment costs at least `3^k / √(2^k)`. -/
theorem ubCost_ge (k : ℕ) (u : UBFeasible (lexEdge k)) :
    (3 : ℝ) ^ k / Real.sqrt ((2 : ℝ) ^ k) ≤ ubCost u := by
  have hsqrt_pos : 0 < Real.sqrt ((2 : ℝ) ^ k) := Real.sqrt_pos.mpr (by positivity)
  rw [div_le_iff₀ hsqrt_pos]
  have hsumT : (3 : ℝ) ^ k = ∑ t : Vert k, ∑ C ∈ u.supp, u.w C t := by
    simp only [u.cover, Finset.sum_const, Finset.card_univ, card_vert, nsmul_eq_mul, mul_one]
    norm_cast
  rw [hsumT, Finset.sum_comm]
  unfold ubCost
  rw [Finset.sum_mul]
  refine Finset.sum_le_sum fun C hC => ?_
  have hsub : ∑ t : Vert k, u.w C t = ∑ t ∈ C.toFinset, u.w C t := by
    symm
    refine Finset.sum_subset (Finset.subset_univ _) ?_
    intro t _ ht
    exact u.supported C t (by simpa using ht)
  rw [hsub]
  have hord := u.valid C hC
  have hcard_le : (C.toFinset.card : ℝ) ≤ (2 : ℝ) ^ k := by
    rw [List.toFinset_card_of_nodup hord.1]
    exact_mod_cast order_length_le_two_pow k C hord
  calc ∑ t ∈ C.toFinset, u.w C t
      ≤ Real.sqrt (C.toFinset.card : ℝ) * Real.sqrt (∑ t ∈ C.toFinset, (u.w C t) ^ 2) :=
        sum_le_sqrt_card_mul_l2 C.toFinset (u.w C)
    _ ≤ Real.sqrt ((2 : ℝ) ^ k) * Real.sqrt (∑ t ∈ C.toFinset, (u.w C t) ^ 2) :=
        mul_le_mul_of_nonneg_right (Real.sqrt_le_sqrt hcard_le) (Real.sqrt_nonneg _)
    _ = Real.sqrt (∑ t ∈ C.toFinset, (u.w C t) ^ 2) * Real.sqrt ((2 : ℝ) ^ k) := mul_comm _ _

/-- Translate a list of vertices by `c : Vert k`. -/
def shiftOrder {k : ℕ} (c : Vert k) (C : List (Vert k)) : List (Vert k) :=
  C.map (· + c)

lemma shiftOrder_isMaximal {k : ℕ} (c : Vert k) {C : List (Vert k)}
    (hC : IsMaximalOrder (lexEdge k) C) :
    IsMaximalOrder (lexEdge k) (shiftOrder c C) := by
  have hord : IsOrder (lexEdge k) (shiftOrder c C) := by
    constructor
    · exact hC.1.1.map (add_left_injective c)
    · rw [shiftOrder, List.pairwise_map]
      exact hC.1.2.imp fun h => (lexEdge_add_right k c _ _).mpr h
  refine ⟨hord, fun C' hC' hsub => ?_⟩
  have hsub' : C.Sublist (C'.map (· - c)) := by
    have h1 := hsub.map (· - c)
    have h2 : (shiftOrder c C).map (· - c) = C := by
      simp [shiftOrder, List.map_map, Function.comp_def]
    simpa [h2] using h1
  have hord' : IsOrder (lexEdge k) (C'.map (· - c)) := by
    constructor
    · exact hC'.1.map (fun x y h => sub_left_inj.mp h)
    · rw [List.pairwise_map]
      exact hC'.2.imp fun h => by
        have := (lexEdge_add_right k (-c) _ _).mpr h
        simpa [sub_eq_add_neg] using this
  have hlen := hC.2 (C'.map (· - c)) hord' hsub'
  simpa [shiftOrder] using hlen

/-- All translates `shiftOrder c (maxOrder k)` are distinct (`c ↦ shiftOrder c C` is injective
whenever `C` is nonempty), giving an orbit of exact size `3^k` where each vertex appears `2^k`
times. -/
lemma shiftOrder_injective {k : ℕ} {C : List (Vert k)} (hC : C ≠ []) :
    Function.Injective (fun c : Vert k => shiftOrder c C) := by
  intro c₁ c₂ h
  cases C with
  | nil => exact absurd rfl hC
  | cons a t =>
    have h0 : a + c₁ = a + c₂ := by
      simpa [shiftOrder] using congrArg List.head? h
    exact add_left_cancel h0

/-- Exact attainment of `UB(S_k) = 3^k / √(2^k)` via the translation orbit of `maxOrder k`. -/
theorem ub_achieved (k : ℕ) :
    ∃ u : UBFeasible (lexEdge k),
      (∀ C ∈ u.supp, IsMaximalOrder (lexEdge k) C) ∧
      ubCost u = (3 : ℝ) ^ k / Real.sqrt ((2 : ℝ) ^ k) := by
  let C₀ := maxOrder k
  have hC₀_ne : C₀ ≠ [] := by
    intro h
    have hlen := maxOrder_length k
    change C₀.length = 2 ^ k at hlen
    rw [h, List.length_nil] at hlen
    exact absurd hlen.symm (by positivity)
  have hinj : Function.Injective (fun c : Vert k => shiftOrder c C₀) :=
    shiftOrder_injective hC₀_ne
  let supp : Finset (List (Vert k)) := Finset.univ.image (fun c : Vert k => shiftOrder c C₀)
  let r : ℝ := (2 : ℝ) ^ k
  have hr_pos : 0 < r := by positivity
  let w : List (Vert k) → Vert k → ℝ := fun C t => if t ∈ C then 1 / r else 0
  have hvalid : ∀ C ∈ supp, IsMaximalOrder (lexEdge k) C := by
    intro C hC
    rcases Finset.mem_image.mp hC with ⟨c, _, rfl⟩
    exact shiftOrder_isMaximal c (maxOrder_isMaximal k)
  have hcover : ∀ t : Vert k, ∑ C ∈ supp, w C t = 1 := by
    intro t
    rw [Finset.sum_image (fun a _ b _ hab => hinj hab)]
    have hcount : (Finset.univ.filter (fun c : Vert k => t ∈ shiftOrder c C₀)).card = 2 ^ k := by
      have hbij : (Finset.univ.filter (fun c : Vert k => t ∈ shiftOrder c C₀)).card =
          C₀.toFinset.card := by
        refine Finset.card_bij (fun c _ => t - c) ?_ ?_ ?_
        · intro c hc
          rw [Finset.mem_filter] at hc
          rcases List.mem_map.mp hc.2 with ⟨x, hx, hxt⟩
          dsimp only
          have : t - c = x := by rw [← hxt]; simp
          rw [this]
          exact List.mem_toFinset.mpr hx
        · intro c₁ _ c₂ _ h
          exact sub_right_inj.mp h
        · intro x hx
          refine ⟨t - x, ?_, by simp⟩
          rw [Finset.mem_filter]
          refine ⟨Finset.mem_univ _, ?_⟩
          exact List.mem_map.mpr ⟨x, List.mem_toFinset.mp hx, by abel⟩
      rw [hbij, List.toFinset_card_of_nodup (maxOrder_isOrder k).1, maxOrder_length k]
    simp only [w, Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const, nsmul_eq_mul,
      hcount]
    simp only [r, Nat.cast_pow, Nat.cast_ofNat]
    field_simp
  let u : UBFeasible (lexEdge k) :=
    { supp := supp
      valid := fun C hC => (hvalid C hC).1
      w := w
      nonneg := fun C t => by
        dsimp [w]
        split_ifs <;> positivity
      supported := fun C t ht => by
        simp [w, ht]
      cover := hcover }
  refine ⟨u, hvalid, ?_⟩
  unfold ubCost
  simp only [u]
  rw [Finset.sum_image (fun a _ b _ hab => hinj hab)]
  have hterm : ∀ c : Vert k,
      Real.sqrt (∑ t ∈ (shiftOrder c C₀).toFinset, (w (shiftOrder c C₀) t) ^ 2) =
        1 / Real.sqrt ((2 : ℝ) ^ k) := by
    intro c
    have hlen : (shiftOrder c C₀).toFinset.card = 2 ^ k := by
      rw [List.toFinset_card_of_nodup (hvalid _ (Finset.mem_image_of_mem _ (Finset.mem_univ c))).1.1]
      simp [shiftOrder, C₀, maxOrder_length]
    have hsum_sq : ∑ t ∈ (shiftOrder c C₀).toFinset, (w (shiftOrder c C₀) t) ^ 2 =
        1 / (2 : ℝ) ^ k := by
      have hcongr : ∑ t ∈ (shiftOrder c C₀).toFinset, (w (shiftOrder c C₀) t) ^ 2 =
          ∑ _t ∈ (shiftOrder c C₀).toFinset, (1 / r) ^ 2 := by
        refine Finset.sum_congr rfl fun t ht => ?_
        have hmem : t ∈ shiftOrder c C₀ := List.mem_toFinset.mp ht
        simp [w, hmem]
      rw [hcongr, Finset.sum_const, hlen, nsmul_eq_mul]
      simp only [r, Nat.cast_pow, Nat.cast_ofNat]
      field_simp
    rw [hsum_sq, Real.sqrt_div zero_le_one, Real.sqrt_one]
  simp_rw [hterm]
  rw [Finset.sum_const, Finset.card_univ, card_vert, nsmul_eq_mul]
  push_cast
  ring

/-- `UB(S_k) = 3^k / √(2^k)` is the exact minimum of the upper-bound program. -/
theorem ub_isLeast (k : ℕ) :
    IsLeast (ubValues (lexEdge k)) ((3 : ℝ) ^ k / Real.sqrt ((2 : ℝ) ^ k)) := by
  refine ⟨?_, ?_⟩
  · rcases ub_achieved k with ⟨u, _, hu⟩
    exact ⟨u, hu⟩
  · rintro val ⟨u, rfl⟩
    exact ubCost_ge k u

theorem UB_eq (k : ℕ) : UB (lexEdge k) = (3 : ℝ) ^ k / Real.sqrt ((2 : ℝ) ^ k) :=
  (ub_isLeast k).csInf_eq

/-! ### 5. Exact Value of `ILB(S_k) = 3^k / √((3^k - 1) / 2)` -/

/-- In the tournament `S_k`, every nonempty independent set is a singleton `{s}`. -/
theorem indepSet_card_eq_one (k : ℕ) (I : IndepSet (lexEdge k)) : I.1.card = 1 := by
  rcases I.2.1 with ⟨a, ha⟩
  have hsub : I.1 = {a} := by
    ext b
    simp only [Finset.mem_singleton]
    constructor
    · intro hb
      by_contra hne
      rcases lexEdge_total k hne with h | h
      · exact I.2.2 b hb a ha h
      · exact I.2.2 a ha b hb h
    · rintro rfl
      exact ha
  rw [hsub, Finset.card_singleton]

/-- Equivalence between `Vert k` and `IndepSet (lexEdge k)` via singletons `{s}`. -/
noncomputable def vertEquivIndepSet (k : ℕ) : Vert k ≃ IndepSet (lexEdge k) where
  toFun s := ⟨{s}, Finset.singleton_nonempty s, fun u hu v hv => by
    rw [Finset.mem_singleton.mp hu, Finset.mem_singleton.mp hv]
    exact lexEdge_irrefl k s⟩
  invFun I := (Finset.card_eq_one.mp (indepSet_card_eq_one k I)).choose
  left_inv s := by
    let I : IndepSet (lexEdge k) := ⟨{s}, Finset.singleton_nonempty s, fun u hu v hv => by
      rw [Finset.mem_singleton.mp hu, Finset.mem_singleton.mp hv]
      exact lexEdge_irrefl k s⟩
    have h : I.1 = {(Finset.card_eq_one.mp (indepSet_card_eq_one k I)).choose} :=
      (Finset.card_eq_one.mp (indepSet_card_eq_one k I)).choose_spec
    exact (Finset.singleton_injective h).symm
  right_inv I := by
    apply Subtype.ext
    exact (Finset.card_eq_one.mp (indepSet_card_eq_one k I)).choose_spec.symm

/-- Universal upper bound on `ILB(S_k)` for every `k ≥ 1`. -/
theorem ilbObj_le (k : ℕ) (hk : 1 ≤ k) (u : ILBFeasible (lexEdge k)) :
    ilbObj u ≤ (3 : ℝ) ^ k / Real.sqrt (((3 ^ k - 1) / 2 : ℕ) : ℝ) := by
  let d : ℕ := (3 ^ k - 1) / 2
  have hd_pos_nat : 0 < d := by
    have h3 : 3 ≤ 3 ^ k := by
      calc 3 = 3 ^ 1 := by norm_num
        _ ≤ 3 ^ k := Nat.pow_le_pow_right (by norm_num) hk
    dsimp [d]
    omega
  have hd_pos : (0 : ℝ) < (d : ℝ) := Nat.cast_pos.mpr hd_pos_nat
  let z' : Vert k → ℝ := fun s => u.z (vertEquivIndepSet k s)
  have hz'_nonneg : ∀ s, 0 ≤ z' s := fun s => u.nonneg _
  have hobj_eq : ilbObj u = ∑ t : Vert k, Real.sqrt (z' t) := by
    unfold ilbObj
    refine Finset.sum_congr rfl fun t _ => ?_
    congr 1
    rw [← (vertEquivIndepSet k).sum_comp]
    have hif : ∀ s : Vert k,
        (if t ∈ (vertEquivIndepSet k s).1 then u.z (vertEquivIndepSet k s) else 0) =
        (if s = t then z' s else 0) := by
      intro s
      simp [vertEquivIndepSet, z', eq_comm]
    simp_rw [hif]
    simp
  -- Sum all `3^k` constraints: `d * ∑ s, z' s ≤ 3^k`
  have hsum_constr : (d : ℝ) * ∑ s : Vert k, z' s ≤ (3 : ℝ) ^ k := by
    have h1 : ∑ t : Vert k, ∑ s : Vert k, (if lexEdge k s t then z' s else 0) ≤ (3 : ℝ) ^ k := by
      calc ∑ t : Vert k, ∑ s : Vert k, (if lexEdge k s t then z' s else 0)
          = ∑ t : Vert k, ∑ I : IndepSet (lexEdge k),
              (if ∃ s ∈ I.1, lexEdge k s t then u.z I else 0) := by
            refine Finset.sum_congr rfl fun t _ => ?_
            rw [← (vertEquivIndepSet k).sum_comp]
            refine Finset.sum_congr rfl fun s _ => ?_
            simp [vertEquivIndepSet, z']
        _ ≤ ∑ _t : Vert k, (1 : ℝ) := Finset.sum_le_sum fun t _ => u.constr t
        _ = (3 : ℝ) ^ k := by
            rw [Finset.sum_const, Finset.card_univ, card_vert, nsmul_eq_mul, mul_one]
            norm_cast
    rw [Finset.sum_comm] at h1
    have h2 : ∑ s : Vert k, ∑ t : Vert k, (if lexEdge k s t then z' s else 0) =
        (d : ℝ) * ∑ s : Vert k, z' s := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun s _ => ?_
      rw [Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const,
        lexEdge_outdeg_eq k s, nsmul_eq_mul]
    rw [← h2]
    exact h1
  have hsum_le : ∑ s : Vert k, z' s ≤ (3 : ℝ) ^ k / (d : ℝ) :=
    (le_div_iff₀' hd_pos).mpr hsum_constr
  rw [hobj_eq]
  have hcs := sum_le_sqrt_card_mul_l2 (Finset.univ : Finset (Vert k)) (fun s => Real.sqrt (z' s))
  have hsq : ∑ s : Vert k, (Real.sqrt (z' s)) ^ 2 = ∑ s : Vert k, z' s :=
    Finset.sum_congr rfl fun s _ => Real.sq_sqrt (hz'_nonneg s)
  rw [hsq, Finset.card_univ, card_vert] at hcs
  calc ∑ t : Vert k, Real.sqrt (z' t)
      ≤ Real.sqrt ((3 ^ k : ℕ) : ℝ) * Real.sqrt (∑ s : Vert k, z' s) := hcs
    _ ≤ Real.sqrt ((3 : ℝ) ^ k) * Real.sqrt ((3 : ℝ) ^ k / (d : ℝ)) := by
        rw [Nat.cast_pow, Nat.cast_ofNat]
        exact mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt hsum_le) (Real.sqrt_nonneg _)
    _ = (3 : ℝ) ^ k / Real.sqrt (d : ℝ) := by
        rw [Real.sqrt_div (by positivity), ← mul_div_assoc, Real.mul_self_sqrt (by positivity)]

/-- Exact attainment of `ILB(S_k) = 3^k / √((3^k - 1)/2)` via uniform singleton weights `1 / d_k`. -/
theorem ilb_achieved (k : ℕ) (hk : 1 ≤ k) :
    ∃ u : ILBFeasible (lexEdge k),
      ilbObj u = (3 : ℝ) ^ k / Real.sqrt (((3 ^ k - 1) / 2 : ℕ) : ℝ) := by
  let d : ℕ := (3 ^ k - 1) / 2
  have hd_pos_nat : 0 < d := by
    have h3 : 3 ≤ 3 ^ k := by
      calc 3 = 3 ^ 1 := by norm_num
        _ ≤ 3 ^ k := Nat.pow_le_pow_right (by norm_num) hk
    dsimp [d]
    omega
  have hd_pos : (0 : ℝ) < (d : ℝ) := Nat.cast_pos.mpr hd_pos_nat
  let u : ILBFeasible (lexEdge k) :=
    { z := fun _ => 1 / (d : ℝ)
      nonneg := fun _ => by positivity
      constr := fun t => by
        rw [← (vertEquivIndepSet k).sum_comp]
        have hif : ∀ s : Vert k,
            (if ∃ x ∈ (vertEquivIndepSet k s).1, lexEdge k x t then (1 : ℝ) / (d : ℝ) else 0) =
            (if lexEdge k s t then (1 : ℝ) / (d : ℝ) else 0) := by
          intro s
          simp [vertEquivIndepSet]
        simp_rw [hif]
        rw [Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const,
          lexEdge_indeg_eq k t, nsmul_eq_mul]
        have hne : (d : ℝ) ≠ 0 := ne_of_gt hd_pos
        change (d : ℝ) * (1 / (d : ℝ)) ≤ 1
        rw [mul_one_div, div_self hne] }
  refine ⟨u, ?_⟩
  unfold ilbObj
  have hterm : ∀ t : Vert k,
      Real.sqrt (∑ I : IndepSet (lexEdge k), if t ∈ I.1 then u.z I else 0) =
        1 / Real.sqrt (d : ℝ) := by
    intro t
    rw [← (vertEquivIndepSet k).sum_comp]
    have hif : ∀ s : Vert k,
        (if t ∈ (vertEquivIndepSet k s).1 then u.z (vertEquivIndepSet k s) else 0) =
        (if s = t then (1 : ℝ) / (d : ℝ) else 0) := by
      intro s
      simp [vertEquivIndepSet, u, eq_comm]
    simp_rw [hif]
    simp
  simp_rw [hterm]
  rw [Finset.sum_const, Finset.card_univ, card_vert, nsmul_eq_mul]
  push_cast
  ring

/-- `ILB(S_k) = 3^k / √((3^k - 1)/2)` is the exact maximum of the independent-set program. -/
theorem ilb_isGreatest (k : ℕ) (hk : 1 ≤ k) :
    IsGreatest (ilbValues (lexEdge k)) ((3 : ℝ) ^ k / Real.sqrt (((3 ^ k - 1) / 2 : ℕ) : ℝ)) := by
  refine ⟨?_, ?_⟩
  · rcases ilb_achieved k hk with ⟨u, hu⟩
    exact ⟨u, hu⟩
  · rintro val ⟨u, rfl⟩
    exact ilbObj_le k hk u

theorem ILB_eq (k : ℕ) (hk : 1 ≤ k) :
    ILB (lexEdge k) = (3 : ℝ) ^ k / Real.sqrt (((3 ^ k - 1) / 2 : ℕ) : ℝ) :=
  (ilb_isGreatest k hk).csSup_eq

/-! ### 6. Exact Ratio Formula and Unbounded Separation Refuting `conj:tight` -/

/-- Exact separation ratio `UB(S_k) / ILB(S_k) = √((3^k - 1) / 2^(k + 1))` for all `k ≥ 1`. -/
theorem ub_div_ilb_eq (k : ℕ) (hk : 1 ≤ k) :
    UB (lexEdge k) / ILB (lexEdge k) = Real.sqrt (((3 : ℝ) ^ k - 1) / (2 : ℝ) ^ (k + 1)) := by
  rw [UB_eq k, ILB_eq k hk]
  have h3_pos : (0 : ℝ) < (3 : ℝ) ^ k := by positivity
  have h2_pos : (0 : ℝ) < (2 : ℝ) ^ k := by positivity
  have hd_cast : (((3 ^ k - 1) / 2 : ℕ) : ℝ) = ((3 : ℝ) ^ k - 1) / 2 := by
    have hmul := two_mul_deg k
    have h1 : (1 : ℕ) ≤ 3 ^ k := Nat.one_le_pow k 3 (by norm_num)
    have hcast : (2 : ℝ) * (((3 ^ k - 1) / 2 : ℕ) : ℝ) = (3 : ℝ) ^ k - 1 := by
      have h' : ((2 * ((3 ^ k - 1) / 2) : ℕ) : ℝ) = ((3 ^ k - 1 : ℕ) : ℝ) := by rw [hmul]
      rw [Nat.cast_mul, Nat.cast_ofNat, Nat.cast_sub h1, Nat.cast_pow, Nat.cast_ofNat,
        Nat.cast_one] at h'
      exact h'
    linarith
  rw [hd_cast]
  have hnum_nonneg : 0 ≤ ((3 : ℝ) ^ k - 1) / 2 := by
    have h1 : (1 : ℝ) ≤ (3 : ℝ) ^ k := one_le_pow₀ (by norm_num)
    linarith
  have hdiv : ((3 : ℝ) ^ k / Real.sqrt ((2 : ℝ) ^ k)) /
      ((3 : ℝ) ^ k / Real.sqrt (((3 : ℝ) ^ k - 1) / 2)) =
      Real.sqrt (((3 : ℝ) ^ k - 1) / 2) / Real.sqrt ((2 : ℝ) ^ k) := by
    have h3ne : (3 : ℝ) ^ k ≠ 0 := ne_of_gt h3_pos
    field_simp
  rw [hdiv, ← Real.sqrt_div hnum_nonneg]
  congr 1
  rw [pow_succ]
  ring

/-- **Refutation of COLT 2024 Conjecture 12 / `conj:tight`**:
for every universal constant `γ : ℝ`, there exists `k ≥ 1` such that
`UB(S_k) / ILB(S_k) > γ`. -/
theorem refute_conj_tight (γ : ℝ) :
    ∃ k : ℕ, 1 ≤ k ∧ γ < UB (lexEdge k) / ILB (lexEdge k) := by
  let M : ℝ := max γ 0
  -- Choose `k ≥ 1` so that `(3/2)^k > 4 * (M^2 + 1)`.
  have hgt : (1 : ℝ) < 3 / 2 := by norm_num
  obtain ⟨k₀, hk₀⟩ := exists_pow_lt_of_lt_one (by positivity : (0 : ℝ) < (4 * (M ^ 2 + 1))⁻¹)
    (by norm_num : (2 : ℝ) / 3 < 1)
  have hk₀' : 4 * (M ^ 2 + 1) < ((3 : ℝ) / 2) ^ k₀ := by
    have hinv : ((2 : ℝ) / 3) ^ k₀ = (((3 : ℝ) / 2) ^ k₀)⁻¹ := by
      rw [← inv_pow]
      congr 1
      norm_num
    rw [hinv, inv_lt_inv₀ (by positivity) (by positivity)] at hk₀
    exact hk₀
  let k := max k₀ 1
  have hk1 : 1 ≤ k := le_max_right _ _
  have hk_pow : 4 * (M ^ 2 + 1) < ((3 : ℝ) / 2) ^ k := by
    calc 4 * (M ^ 2 + 1) < ((3 : ℝ) / 2) ^ k₀ := hk₀'
      _ ≤ ((3 : ℝ) / 2) ^ k := pow_le_pow_right₀ (by norm_num) (le_max_left _ _)
  refine ⟨k, hk1, ?_⟩
  rw [ub_div_ilb_eq k hk1]
  have h3ge2 : (2 : ℝ) ≤ (3 : ℝ) ^ k := by
    calc (2 : ℝ) ≤ 3 := by norm_num
      _ = (3 : ℝ) ^ 1 := by norm_num
      _ ≤ (3 : ℝ) ^ k := pow_le_pow_right₀ (by norm_num) hk1
  have hlower : M ^ 2 < ((3 : ℝ) ^ k - 1) / (2 : ℝ) ^ (k + 1) := by
    have hhalf : (3 : ℝ) ^ k / 2 ≤ (3 : ℝ) ^ k - 1 := by linarith
    have h2pos : (0 : ℝ) < (2 : ℝ) ^ (k + 1) := by positivity
    have hstep : ((3 : ℝ) / 2) ^ k / 4 ≤ ((3 : ℝ) ^ k - 1) / (2 : ℝ) ^ (k + 1) := by
      rw [div_pow, div_div, pow_succ]
      have hdenom : (2 : ℝ) ^ k * 4 = (2 : ℝ) ^ k * 2 * 2 := by ring
      rw [hdenom]
      have : (3 : ℝ) ^ k / ((2 : ℝ) ^ k * 2 * 2) = ((3 : ℝ) ^ k / 2) / ((2 : ℝ) ^ k * 2) := by ring
      rw [this]
      exact div_le_div_of_nonneg_right hhalf (by positivity)
    linarith
  calc γ ≤ M := le_max_left γ 0
    _ = Real.sqrt (M ^ 2) := (Real.sqrt_sq (le_max_right γ 0)).symm
    _ < Real.sqrt (((3 : ℝ) ^ k - 1) / (2 : ℝ) ^ (k + 1)) :=
      Real.sqrt_lt_sqrt (sq_nonneg M) hlower

end TemporalFeedbackRegret
