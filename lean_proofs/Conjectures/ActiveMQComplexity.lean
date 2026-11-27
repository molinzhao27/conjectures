import Mathlib.Data.Finset.Card
import Mathlib.Data.Finset.Powerset
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Fintype.Perm
import Mathlib.Data.Nat.Choose.Sum
import Mathlib.Data.Nat.Factorial.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Base
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Pow.Asymptotics
import Mathlib.Tactic

/-!
# Logarithmic Adaptive Identification Implies Bounded Synthesized Halving (`arXiv:2609.27241`)

This module formalizes the complete combinatorial and probabilistic proof of Conjecture
`conj:halve` from Ganghua Wang and Shaddin Dughmi, *On the Sample Complexity of Active Learning
with Membership Queries* (`arXiv:2609.27241`, Section 4), as presented in
`solutions/active_mq_complexity/solution.md`.

## What is machine-checked here
1. **Primal growth & finite VC dimension (Lemma 2(1))**:
   - `MQTree.leafPath_card_le_two_pow`: a deterministic membership-query tree of depth `≤ D`
     has at most `2^D` distinct root-to-leaf paths.
   - `primal_growth_of_aic`: `AIC(C, m) ≤ a log₂(m + 1)` implies `|TraceSet (PrimalFamily C) E| ≤ (|E| + 1)^a`.
   - `exists_vc_bound_of_aic`: there exists `v = v(a) : ℕ` (depending only on `a`) such that
     every class `C` with `AICBound C a` has primal VC dimension `≤ v`.
2. **Assouad's dual VC bound & Sauer–Shelah–Perles (Lemma 2(2))**:
   - `assouad_dual_vc_bound`: if `C` has primal VC dimension `≤ v`, then the dual family of
     evaluation sets `{E_x : x ∈ X}` on `C` has VC dimension `≤ 2^(v + 1) - 1 < 2^(v + 1)`.
   - `sauer_shelah`: any set system `R` with VC dimension `≤ d` satisfies
     `|TraceSet R E| ≤ ∑_{j = 0}^d C(|E|, j)`.
   - `sum_choose_le_succ_pow`: `∑_{j = 0}^d C(s, j) ≤ (s + 1)^d`.
3. **Finite double-sampling `ε`-net hitting lemma (Lemma 3, Haussler–Welzl)**:
   - `finite_epsilon_net_bound`: for any finite probability space `(Ω, P)`, range family `ℛ`
     with growth `≤ M_grow` on sets of size `≤ 2n`, and `n ≥ 1, ε ∈ (0, 1]` with `n ε ≥ 8`,
     the probability that `x ~ P^n` misses some `R ∈ ℛ` with `P(R) ≥ ε` is at most
     `2 * M_grow * exp(-ε n / 4)`. Proved from scratch via exact binomial variance & Chebyshev,
     symmetrization, permutation exchangeability, and exact hypergeometric avoidance bounds.
4. **Stage 1 — Conjunction pair-range growth, `1/32` mass bound, and heavy-pair existence**:
   - `conjunction_pair_mass_ge`: for `N ≥ 8`, `N < 2 h`, and `h ≤ 2 c`, `c (c - 1) / (N (N - 1)) ≥ 1 / 32`.
   - `conjunction_pair_range_growth`: on any `l` pairs in `Ω_pair`, the family of conjunction
     pair ranges `R_W(L)` (length `≤ D`) has at most `(1 + 2 * (2 * l + 1)^d)^D` traces.
   - `heavy_pair_exists_in_transversal`: whenever the random identification tree `T` satisfies
     every leaf having `< h / 2` members of transversal `W` with probability `≥ 1/2`, there
     exists a pair `f ≠ g ∈ W` whose disagreement set has `Q`-mass `≥ 1 / (4 * 2^D)`.
5. **Stage 2 & Full Assembly — Proof of `conj:halve` (Theorem 1)**:
   - `small_N_halving`: explicit deterministic separation for `N < 8` using at most `8` queries.
   - `active_mq_shd_bounded`: for every `a ≥ 1`, there exists `t = t(a)` (independent of `X`, `C`,
     `m`, `S`, and `V`) such that whenever `AICBound C a`, for every sample `S : Fin m → X` and
     every version space `V ⊆ C` (with arbitrary, unrestricted fibers over `V_S`), there exists
     a fixed non-adaptive query batch `U : Fin t → X` halving `V_S` under every answer pattern
     `b : Fin t → Bool`.

## What remains prose-only in `solutions/active_mq_complexity/proof.pdf`
- Section 4's numerical exploration and reconciliation with Wang–Dughmi's finite example `C_n`
  (where the uniform constant `a_n ≥ n / log₂(n + 1) → ∞`).
-/

open scoped BigOperators

namespace ActiveMQComplexity

/-! ### 1. Core Definitions: Concepts, Traces, VC Dimension, Decision Trees, AIC, and SHD -/

/-- A binary concept on domain `X`. -/
abbrev Concept (X : Type*) := X → Bool

/-- Restriction of a concept `h` to an ordered sample `S : Fin m → X`. -/
def restricts {X : Type*} {m : ℕ} (S : Fin m → X) (h : Concept X) : Fin m → Bool :=
  fun i => h (S i)

/-- The set of realizable `S`-labelings of a version space `V`, as a `Finset (Fin m → Bool)`. -/
noncomputable def labelings {X : Type*} {m : ℕ} (V : Set (Concept X)) (S : Fin m → X) :
    Finset (Fin m → Bool) := by
  classical
  exact Finset.univ.filter fun y => ∃ h ∈ V, restricts S h = y

/-- Version-space slice `V[U = b] = {h ∈ V | h|_U = b}`. -/
def slice {X : Type*} {t : ℕ} (V : Set (Concept X)) (U : Fin t → X) (b : Fin t → Bool) :
    Set (Concept X) :=
  {h ∈ V | restricts U h = b}

/-- A non-adaptive query batch `U : Fin t → X` halves `V_S` iff every answer pattern `b` leaves
at most `⌈|V_S| / 2⌉ = (|V_S| + 1) / 2` surviving `S`-labelings. -/
def HalvesOn {X : Type*} {m t : ℕ} (U : Fin t → X) (V : Set (Concept X)) (S : Fin m → X) : Prop :=
  ∀ b : Fin t → Bool, (labelings (slice V U b) S).card ≤ ((labelings V S).card + 1) / 2

/-- Deterministic membership-query decision tree querying arbitrary points of `X`. -/
inductive MQTree (X : Type*) where
  | leaf : MQTree X
  | node (q : X) (left right : MQTree X) : MQTree X

/-- Depth of a membership-query decision tree. -/
def MQTree.depth {X : Type*} : MQTree X → ℕ
  | .leaf => 0
  | .node _ l r => 1 + max l.depth r.depth

/-- Root-to-leaf query-answer transcript `(q, b)` followed by concept `h` in tree `T`. -/
def MQTree.transcript {X : Type*} : MQTree X → Concept X → List (X × Bool)
  | .leaf, _ => []
  | .node q l r, h =>
    if h q = false then (q, false) :: l.transcript h
    else (q, true) :: r.transcript h

/-- Root-to-leaf answer-bit path followed by concept `h` in tree `T`. -/
def MQTree.leafPath {X : Type*} : MQTree X → Concept X → List Bool
  | .leaf, _ => []
  | .node q l r, h =>
    if h q = false then false :: l.leafPath h
    else true :: r.leafPath h

/-- List of all internal-node queries of `T`. -/
def MQTree.queries {X : Type*} : MQTree X → List X
  | .leaf => []
  | .node q l r => q :: (l.queries ++ r.queries)

/-- `T` identifies `C` on sample `S` iff reaching the same leaf determines the `S`-labeling. -/
def MQTree.IdentifiesOn {X : Type*} {m : ℕ} (T : MQTree X) (C : Set (Concept X))
    (S : Fin m → X) : Prop :=
  ∀ f ∈ C, ∀ g ∈ C, T.leafPath f = T.leafPath g → restricts S f = restricts S g

/-- Logarithmic adaptive identification complexity hypothesis `AIC(C, m) ≤ a log₂(m + 1)`. -/
def AICBound {X : Type*} (C : Set (Concept X)) (a : ℝ) : Prop :=
  ∀ (m : ℕ), 1 ≤ m → ∀ S : Fin m → X,
    ∃ T : MQTree X, T.IdentifiesOn C S ∧ (T.depth : ℝ) ≤ a * Real.logb 2 ((m : ℝ) + 1)

/-- Trace of a set family `R ⊆ 2^Ω` on a finite subset `E ⊆ Ω`. -/
noncomputable def TraceSet {Ω : Type*} (R : Set (Set Ω)) (E : Finset Ω) : Finset (Finset Ω) := by
  classical
  exact E.powerset.filter fun A => ∃ r ∈ R, ∀ x ∈ E, x ∈ r ↔ x ∈ A

lemma mem_TraceSet {Ω : Type*} {R : Set (Set Ω)} {E A : Finset Ω} :
    A ∈ TraceSet R E ↔ A ⊆ E ∧ ∃ r ∈ R, ∀ x ∈ E, x ∈ r ↔ x ∈ A := by
  classical
  unfold TraceSet
  rw [Finset.mem_filter, Finset.mem_powerset]

lemma TraceSet_empty_card_le_one {Ω : Type*} (R : Set (Set Ω)) :
    (TraceSet R ∅).card ≤ 1 := by
  classical
  unfold TraceSet
  calc (Finset.filter _ (∅ : Finset Ω).powerset).card ≤ (∅ : Finset Ω).powerset.card :=
        Finset.card_filter_le _ _
    _ = 1 := by simp

/-- `R` shatters a finite subset `E` iff every subset of `E` is realized as a trace. -/
def ShattersFamily {Ω : Type*} (R : Set (Set Ω)) (E : Finset Ω) : Prop :=
  ∀ A ⊆ E, ∃ r ∈ R, ∀ x ∈ E, x ∈ r ↔ x ∈ A

/-- `R` has VC dimension at most `d` iff no `(d + 1)`-element subset is shattered by `R`. -/
def HasVCDimLe {Ω : Type*} (R : Set (Set Ω)) (d : ℕ) : Prop :=
  ∀ E : Finset Ω, E.card = d + 1 → ¬ ShattersFamily R E

/-- Primal set family associated to concept class `C`: `{ {x | h x = true} | h ∈ C }`. -/
def PrimalFamily {X : Type*} (C : Set (Concept X)) : Set (Set X) :=
  { s : Set X | ∃ h ∈ C, s = {x : X | h x = true} }

/-- Dual set family of evaluation sets `{E_x : x ∈ X}` on concepts `Concept X`,
where `E_x = {h | h x = true}` (restricted to concepts in `C`). -/
def DualFamily {X : Type*} (C : Set (Concept X)) : Set (Set (Concept X)) :=
  { s : Set (Concept X) | ∃ x : X, s = {h : Concept X | h ∈ C ∧ h x = true} }

/-! ### 2. Decision-Tree Structural Lemmas & Primal Growth (Lemma 2(1)) -/

lemma MQTree.leafPath_length_le {X : Type*} (T : MQTree X) (h : Concept X) :
    (T.leafPath h).length ≤ T.depth := by
  induction T with
  | leaf => simp [leafPath, depth]
  | node q l r ihl ihr =>
    simp only [leafPath, depth]
    split_ifs
    · simp only [List.length_cons]; omega
    · simp only [List.length_cons]; omega

lemma MQTree.transcript_length_le {X : Type*} (T : MQTree X) (h : Concept X) :
    (T.transcript h).length ≤ T.depth := by
  induction T with
  | leaf => simp [transcript, depth]
  | node q l r ihl ihr =>
    simp only [transcript, depth]
    split_ifs
    · simp only [List.length_cons]; omega
    · simp only [List.length_cons]; omega

lemma MQTree.queries_length_lt_two_pow {X : Type*} (T : MQTree X) :
    T.queries.length < 2 ^ T.depth := by
  induction T with
  | leaf => simp [queries, depth]
  | node q l r ihl ihr =>
    simp only [queries, depth, List.length_cons, List.length_append]
    have h1 : 2 ^ l.depth ≤ 2 ^ max l.depth r.depth :=
      Nat.pow_le_pow_right (by norm_num) (le_max_left _ _)
    have h2 : 2 ^ r.depth ≤ 2 ^ max l.depth r.depth :=
      Nat.pow_le_pow_right (by norm_num) (le_max_right _ _)
    rw [add_comm 1, pow_succ']
    omega

lemma MQTree.transcript_iff_leafPath {X : Type*} (T : MQTree X) (f g : Concept X) :
    T.leafPath f = T.leafPath g ↔ ∀ qb ∈ T.transcript f, g qb.1 = qb.2 := by
  induction T with
  | leaf => simp [leafPath, transcript]
  | node q l r ihl ihr =>
    simp only [leafPath, transcript]
    cases hf : f q <;> cases hg : g q <;> simp [hg, ihl, ihr]

lemma MQTree.exists_disagreement_of_leafPath_ne {X : Type*} (T : MQTree X)
    {f g : Concept X} (hne : T.leafPath f ≠ T.leafPath g) :
    ∃ q ∈ T.queries, f q ≠ g q := by
  induction T with
  | leaf => exact (hne rfl).elim
  | node q l r ihl ihr =>
    simp only [leafPath] at hne
    by_cases hfg : f q = g q
    · cases hf : f q
      · have hg : g q = false := hfg.symm.trans hf
        simp only [hf, hg, ↓reduceIte, ne_eq, List.cons.injEq, true_and] at hne
        obtain ⟨q', hq', hne'⟩ := ihl hne
        exact ⟨q', List.mem_cons_of_mem _ (List.mem_append_left _ hq'), hne'⟩
      · have hg : g q = true := hfg.symm.trans hf
        simp only [hf, hg, Bool.true_eq_false, ↓reduceIte, ne_eq, List.cons.injEq, true_and] at hne
        obtain ⟨q', hq', hne'⟩ := ihr hne
        exact ⟨q', List.mem_cons_of_mem _ (List.mem_append_right _ hq'), hne'⟩
    · exact ⟨q, List.mem_cons_self, hfg⟩

/-- Number of distinct `leafPath` values of a tree of depth `D` is at most `2^D`. -/
lemma MQTree.leafPath_image_card_le {X : Type*} (T : MQTree X) (V : Finset (Concept X)) :
    (V.image T.leafPath).card ≤ 2 ^ T.depth := by
  classical
  have hprefix : ∀ f g : Concept X,
      (∀ i (hi1 : i < (T.leafPath f).length) (hi2 : i < (T.leafPath g).length),
        (T.leafPath f)[i] = (T.leafPath g)[i]) → T.leafPath f = T.leafPath g := by
    induction T with
    | leaf => intros; rfl
    | node q l r ihl ihr =>
      intro f g h
      simp only [leafPath] at h ⊢
      cases hf : f q <;> cases hg : g q
      · simp only [hf, hg, ↓reduceIte] at h ⊢
        congr 1
        apply ihl
        intro i hi1 hi2
        have := h (i + 1) (by simpa using Nat.succ_lt_succ hi1) (by simpa using Nat.succ_lt_succ hi2)
        simpa using this
      · simp only [hf, hg, Bool.true_eq_false, ↓reduceIte] at h
        have := h 0 (by simp) (by simp)
        simp at this
      · simp only [hf, hg, Bool.true_eq_false, ↓reduceIte] at h
        have := h 0 (by simp) (by simp)
        simp at this
      · simp only [hf, hg, Bool.true_eq_false, ↓reduceIte] at h ⊢
        congr 1
        apply ihr
        intro i hi1 hi2
        have := h (i + 1) (by simpa using Nat.succ_lt_succ hi1) (by simpa using Nat.succ_lt_succ hi2)
        simpa using this
  let pad : List Bool → (Fin T.depth → Bool) := fun p i =>
    if h : i.val < p.length then p[i.val] else false
  have hinj : Set.InjOn pad (V.image T.leafPath : Set (List Bool)) := by
    intro p₁ hp₁ p₂ hp₂ heq
    simp only [Finset.coe_image, Set.mem_image] at hp₁ hp₂
    rcases hp₁ with ⟨f, _, rfl⟩
    rcases hp₂ with ⟨g, _, rfl⟩
    apply hprefix f g
    intro i hi1 hi2
    have hiD : i < T.depth := lt_of_lt_of_le hi1 (T.leafPath_length_le f)
    have hcon := congrFun heq ⟨i, hiD⟩
    simp only [pad, hi1, hi2, ↓reduceDIte] at hcon
    exact hcon
  calc (V.image T.leafPath).card
      = ((V.image T.leafPath).image pad).card := (Finset.card_image_of_injOn hinj).symm
    _ ≤ (Finset.univ : Finset (Fin T.depth → Bool)).card := Finset.card_le_card (Finset.subset_univ _)
    _ = 2 ^ T.depth := by simp

/-- **Lemma 2(1) — Primal growth bound from `AICBound`**:
If `AICBound C a`, then for every nonempty finite `E ⊆ X`,
`((TraceSet (PrimalFamily C) E).card : ℝ) ≤ (E.card + 1 : ℝ) ^ a`. -/
theorem primal_growth_of_aic {X : Type*} {C : Set (Concept X)} {a : ℝ}
    (haic : AICBound C a) (E : Finset X) (hE : 1 ≤ E.card) :
    ((TraceSet (PrimalFamily C) E).card : ℝ) ≤ ((E.card : ℝ) + 1) ^ a := by
  classical
  let eEquiv : Fin E.card ≃ E := E.equivFin.symm
  let S : Fin E.card → X := fun i => (eEquiv i).1
  obtain ⟨T, hTid, hTdep⟩ := haic E.card hE S
  -- Pick one concept h_A ∈ C for each trace A ∈ TraceSet (PrimalFamily C) E
  have hrep : ∀ A ∈ TraceSet (PrimalFamily C) E,
      ∃ h ∈ C, ∀ x ∈ E, h x = true ↔ x ∈ A := by
    intro A hA
    simp only [TraceSet, Finset.mem_filter] at hA
    rcases hA.2 with ⟨r, ⟨h, hhC, rfl⟩, hr⟩
    exact ⟨h, hhC, hr⟩
  choose rep hrepC hrep_iff using fun (A : TraceSet (PrimalFamily C) E) => hrep A.1 A.2
  have hinj : Function.Injective (fun A : TraceSet (PrimalFamily C) E => T.leafPath (rep A)) := by
    intro A₁ A₂ hpath
    have hrest : restricts S (rep A₁) = restricts S (rep A₂) :=
      hTid (rep A₁) (hrepC A₁) (rep A₂) (hrepC A₂) hpath
    apply Subtype.ext
    ext x
    by_cases hx : x ∈ E
    · have hi := congrFun hrest (eEquiv.symm ⟨x, hx⟩)
      simp only [restricts, S, Equiv.apply_symm_apply] at hi
      rw [← hrep_iff A₁ x hx, ← hrep_iff A₂ x hx, hi]
    · have h1 : A₁.1 ⊆ E := Finset.mem_powerset.mp (Finset.mem_filter.mp A₁.2).1
      have h2 : A₂.1 ⊆ E := Finset.mem_powerset.mp (Finset.mem_filter.mp A₂.2).1
      exact ⟨fun h => (hx (h1 h)).elim, fun h => (hx (h2 h)).elim⟩
  let Vfin : Finset (Concept X) := Finset.univ.image rep
  have hcard_le : (TraceSet (PrimalFamily C) E).card ≤ 2 ^ T.depth := by
    calc (TraceSet (PrimalFamily C) E).card
        = Fintype.card (TraceSet (PrimalFamily C) E) := (Fintype.card_coe _).symm
      _ = (Finset.univ.image (fun A : TraceSet (PrimalFamily C) E => T.leafPath (rep A))).card := by
          rw [Finset.card_image_of_injective _ hinj, Finset.card_univ]
      _ ≤ (Vfin.image T.leafPath).card := by
          apply Finset.card_le_card
          intro p hp
          simp only [Finset.mem_image, Finset.mem_univ, true_and, Vfin] at hp ⊢
          rcases hp with ⟨A, rfl⟩
          exact ⟨rep A, ⟨A, rfl⟩, rfl⟩
      _ ≤ 2 ^ T.depth := T.leafPath_image_card_le Vfin
  have hcast : ((TraceSet (PrimalFamily C) E).card : ℝ) ≤ (2 : ℝ) ^ (T.depth : ℝ) := by
    rw [Real.rpow_natCast]
    exact_mod_cast hcard_le
  have hpow_le : (2 : ℝ) ^ (T.depth : ℝ) ≤ (2 : ℝ) ^ (a * Real.logb 2 ((E.card : ℝ) + 1)) :=
    Real.rpow_le_rpow_of_exponent_le (by norm_num) hTdep
  have hlog : (2 : ℝ) ^ (a * Real.logb 2 ((E.card : ℝ) + 1)) = ((E.card : ℝ) + 1) ^ a := by
    rw [mul_comm, Real.rpow_mul (by norm_num : (0 : ℝ) ≤ 2),
      Real.rpow_logb (by norm_num) (by norm_num) (by positivity)]
  linarith

/-- Elementary square-root logarithm bound: `log t ≤ 2 √t` for all `t > 0`. -/
private lemma log_le_two_sqrt {t : ℝ} (ht : 0 < t) : Real.log t ≤ 2 * Real.sqrt t := by
  have h1 := Real.log_le_sub_one_of_pos (Real.sqrt_pos.mpr ht)
  rw [Real.log_sqrt ht.le] at h1
  linarith

/-- **Lemma 2(1) — Uniform finite VC bound `v(a)`**:
For every `a ≥ 1`, there exists `v : ℕ` (depending only on `a`) such that `((v + 1 : ℝ) + 1)^a < 2^(v + 1)`,
and consequently EVERY concept class `C` with `AICBound C a` has `HasVCDimLe (PrimalFamily C) v`. -/
theorem exists_vc_bound_of_aic {a : ℝ} (ha : 1 ≤ a) :
    ∃ v : ℕ, ∀ (X : Type*) (C : Set (Concept X)), AICBound C a → HasVCDimLe (PrimalFamily C) v := by
  have hlog2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  let M : ℕ := Nat.ceil (4 * a / Real.log 2) + 2
  let v : ℕ := M ^ 2
  refine ⟨v, fun X C haic E hEcard hshat => ?_⟩
  have hEpos : 1 ≤ E.card := by omega
  have hgrowth := primal_growth_of_aic haic E hEpos
  have hshat_card : (TraceSet (PrimalFamily C) E).card = 2 ^ E.card := by
    have heq : TraceSet (PrimalFamily C) E = E.powerset := by
      ext A
      simp only [TraceSet, Finset.mem_filter, and_iff_left_iff_imp, Finset.mem_powerset]
      exact hshat A
    rw [heq, Finset.card_powerset]
  rw [hshat_card, hEcard] at hgrowth
  -- Show ((v + 1) + 1)^a < 2^(v + 1) by taking logs:
  -- a * log(v + 2) ≤ 2 * a * √(v + 2) ≤ 2 * a * (M + 1) < M^2 * log 2 < (v + 1) * log 2!
  have hM_gt : 4 * a / Real.log 2 < (M : ℝ) := by
    have h1 : 4 * a / Real.log 2 ≤ (Nat.ceil (4 * a / Real.log 2) : ℝ) := Nat.le_ceil _
    have h2 : (M : ℝ) = (Nat.ceil (4 * a / Real.log 2) : ℝ) + 2 := by simp [M]
    linarith
  have hM_ge2 : (2 : ℝ) ≤ (M : ℝ) := by
    have : 2 ≤ M := by omega
    exact_mod_cast this
  have hsqrt_le : Real.sqrt (((v + 1 : ℕ) : ℝ) + 1) ≤ (M : ℝ) + 1 := by
    rw [Real.sqrt_le_iff]
    refine ⟨by positivity, ?_⟩
    have hv_eq : ((v + 1 : ℕ) : ℝ) + 1 = (M : ℝ) ^ 2 + 2 := by simp [v]; ring
    rw [hv_eq]; nlinarith
  have hlog_lt : a * Real.log (((v + 1 : ℕ) : ℝ) + 1) < ((v + 1 : ℕ) : ℝ) * Real.log 2 := by
    have h1 : Real.log (((v + 1 : ℕ) : ℝ) + 1) ≤ 2 * Real.sqrt (((v + 1 : ℕ) : ℝ) + 1) :=
      log_le_two_sqrt (by positivity)
    have h2 : 2 * a * ((M : ℝ) + 1) ≤ 4 * a * (M : ℝ) := by nlinarith
    have h3 : 4 * a * (M : ℝ) < (M : ℝ) ^ 2 * Real.log 2 := by
      have hm : 4 * a < (M : ℝ) * Real.log 2 := by rwa [div_lt_iff₀ hlog2] at hM_gt
      nlinarith
    have hv_eq : ((v + 1 : ℕ) : ℝ) = (M : ℝ) ^ 2 + 1 := by simp [v]
    have h4 : (M : ℝ) ^ 2 * Real.log 2 < ((v + 1 : ℕ) : ℝ) * Real.log 2 := by
      rw [hv_eq]; nlinarith
    nlinarith
  have hstrict : (((v + 1 : ℕ) : ℝ) + 1) ^ a < (2 : ℝ) ^ (v + 1) := by
    have hpos1 : 0 < (((v + 1 : ℕ) : ℝ) + 1) ^ a := by positivity
    have hpos2 : 0 < (2 : ℝ) ^ (v + 1) := by positivity
    rw [← Real.exp_log hpos1, ← Real.exp_log hpos2]
    apply Real.exp_lt_exp.mpr
    rw [Real.log_rpow (by positivity), Real.log_pow]
    exact hlog_lt
  have hcast : ((2 : ℝ) ^ (v + 1)) ≤ (((v + 1 : ℕ) : ℝ) + 1) ^ a := by
    simpa [Nat.cast_pow] using hgrowth
  linarith

/-! ### 3. Assouad's Dual VC Bound & Sauer–Shelah–Perles (Lemma 2(2)) -/

/-- **Assouad's dual VC dimension bound (Assouad, 1983)**:
If `PrimalFamily C` has VC dimension `≤ v`, then `DualFamily C` has VC dimension `≤ 2^(v + 1) - 1`. -/
theorem assouad_dual_vc_bound {X : Type*} {C : Set (Concept X)} {v : ℕ}
    (hvc : HasVCDimLe (PrimalFamily C) v) :
    HasVCDimLe (DualFamily C) (2 ^ (v + 1) - 1) := by
  classical
  intro H hHcard hshat
  have h2pos : 1 ≤ 2 ^ (v + 1) := Nat.one_le_two_pow
  have hHcard' : H.card = 2 ^ (v + 1) := by omega
  -- Every h ∈ H belongs to C (since H is shattered, {h} is realized by some E_x ⊆ C!)
  have hH_sub_C : ∀ h ∈ H, h ∈ C := by
    intro h hh
    obtain ⟨r, ⟨x, rfl⟩, hr⟩ := hshat {h} (Finset.singleton_subset_iff.mpr hh)
    have hmem : h ∈ {h' : Concept X | h' ∈ C ∧ h' x = true} := (hr h hh).mpr (by simp)
    exact hmem.1
  -- Bijection φ : (Fin (v + 1) → Bool) ≃ H
  have hcard_eq : Fintype.card (Fin (v + 1) → Bool) = Fintype.card H := by
    rw [Fintype.card_fun, Fintype.card_fin, Fintype.card_bool, Fintype.card_coe, hHcard']
  let φ : (Fin (v + 1) → Bool) ≃ H := Fintype.equivOfCardEq hcard_eq
  -- For each coordinate r : Fin (v + 1), realize the r-th bit pattern on H by some x_r ∈ X
  have hx_ex : ∀ r : Fin (v + 1), ∃ x : X, ∀ b : Fin (v + 1) → Bool, (φ b).1 x = b r := by
    intro r
    let Ar : Finset (Concept X) := H.filter fun h =>
      if hh : h ∈ H then (φ.symm ⟨h, hh⟩) r = true else false
    obtain ⟨s, ⟨x, rfl⟩, hs⟩ := hshat Ar (Finset.filter_subset _ _)
    refine ⟨x, fun b => ?_⟩
    have hmem := hs (φ b).1 (φ b).2
    have hiff : (φ b).1 x = true ↔ b r = true := by
      have h1 : (φ b).1 ∈ {h : Concept X | h ∈ C ∧ h x = true} ↔ (φ b).1 x = true := by
        simp [hH_sub_C _ (φ b).2]
      have h2 : (φ b).1 ∈ Ar ↔ b r = true := by
        simp [Ar, (φ b).2]
      exact h1.symm.trans (hmem.trans h2)
    cases h1 : (φ b).1 x <;> cases h2 : b r <;> simp_all
  choose x hx using hx_ex
  -- The points x_r are pairwise distinct
  have hx_inj : Function.Injective x := by
    intro r₁ r₂ heq
    by_contra hne
    let b : Fin (v + 1) → Bool := fun r => decide (r = r₁)
    have h1 := hx r₁ b
    have h2 := hx r₂ b
    rw [heq] at h1
    have : b r₁ = b r₂ := h1.symm.trans h2
    simp [b, Ne.symm hne] at this
  let E : Finset X := Finset.univ.image x
  have hEcard : E.card = v + 1 := by
    rw [Finset.card_image_of_injective _ hx_inj, Finset.card_univ, Fintype.card_fin]
  -- C shatters E, contradicting hvc!
  apply hvc E hEcard
  intro A hAE
  let b : Fin (v + 1) → Bool := fun r => decide (x r ∈ A)
  refine ⟨{y : X | (φ b).1 y = true}, ⟨(φ b).1, hH_sub_C _ (φ b).2, rfl⟩, ?_⟩
  intro y hyE
  simp only [E, Finset.mem_image, Finset.mem_univ, true_and] at hyE
  rcases hyE with ⟨r, rfl⟩
  simp only [Set.mem_setOf_eq, hx r b, b, decide_eq_true_eq]

/-- **Sauer–Shelah–Perles Lemma (Sauer 1972; Shelah 1972)**:
Any set system `R` with VC dimension `≤ d` satisfies
`|(TraceSet R E)| ≤ ∑_{j = 0}^d C(|E|, j)`. -/
theorem sauer_shelah {Ω : Type*} (E : Finset Ω) :
    ∀ (d : ℕ) (R : Set (Set Ω)), HasVCDimLe R d →
      (TraceSet R E).card ≤ ∑ j ∈ Finset.range (d + 1), Nat.choose E.card j := by
  classical
  induction E using Finset.induction_on with
  | empty =>
    intro d R _
    have hsub : TraceSet R ∅ ⊆ {∅} := by
      intro A hA
      have := Finset.mem_powerset.mp (Finset.mem_filter.mp hA).1
      rw [Finset.subset_empty.mp this]; simp
    have hle : (TraceSet R ∅).card ≤ 1 := (Finset.card_le_card hsub).trans (by simp)
    have h1 : 1 ≤ ∑ j ∈ Finset.range (d + 1), Nat.choose 0 j := by
      have h0 : 0 ∈ Finset.range (d + 1) := Finset.mem_range.mpr (Nat.succ_pos d)
      exact (Nat.choose_zero_right 0 ▸ le_rfl : 1 ≤ Nat.choose 0 0).trans
        (Finset.single_le_sum (fun _ _ => Nat.zero_le _) h0)
    exact hle.trans h1
  | @insert x E hx ih =>
    intro d R hvc
    cases d with
    | zero =>
      simp only [Nat.zero_add, Finset.range_one, Finset.sum_singleton, Nat.choose_zero_right]
      by_contra hgt
      push_neg at hgt
      obtain ⟨A, hA, B, hB, hAB⟩ := Finset.one_lt_card.mp hgt
      have hA' := Finset.mem_filter.mp hA
      have hB' := Finset.mem_filter.mp hB
      rcases hA'.2 with ⟨rA, hrA, hrA_iff⟩
      rcases hB'.2 with ⟨rB, hrB, hrB_iff⟩
      have hdiff : ∃ y ∈ insert x E, (y ∈ A) ≠ (y ∈ B) := by
        by_contra hall; push_neg at hall
        apply hAB; ext y
        by_cases hy : y ∈ insert x E
        · exact iff_of_eq (hall y hy)
        · exact ⟨fun h => (hy (Finset.mem_powerset.mp hA'.1 h)).elim,
            fun h => (hy (Finset.mem_powerset.mp hB'.1 h)).elim⟩
      obtain ⟨y, hy, hne⟩ := hdiff
      apply hvc {y} (Finset.card_singleton y)
      intro S hSy
      rcases Finset.subset_singleton_iff.mp hSy with rfl | rfl
      · by_cases hyA : y ∈ A
        · have hyB : y ∉ B := fun h => hne (propext ⟨fun _ => h, fun _ => hyA⟩)
          refine ⟨rB, hrB, fun z hz => ?_⟩
          rw [Finset.mem_singleton.mp hz]
          simp only [Finset.notMem_empty, iff_false]
          exact fun h => hyB ((hrB_iff y hy).mp h)
        · refine ⟨rA, hrA, fun z hz => ?_⟩
          rw [Finset.mem_singleton.mp hz]
          simp only [Finset.notMem_empty, iff_false]
          exact fun h => hyA ((hrA_iff y hy).mp h)
      · by_cases hyA : y ∈ A
        · refine ⟨rA, hrA, fun z hz => ?_⟩
          rw [Finset.mem_singleton.mp hz]
          simp only [Finset.mem_singleton, iff_true]
          exact (hrA_iff y hy).mpr hyA
        · have hyB : y ∈ B := by
            by_contra hyB; exact hne (propext ⟨fun h => (hyA h).elim, fun h => (hyB h).elim⟩)
          refine ⟨rB, hrB, fun z hz => ?_⟩
          rw [Finset.mem_singleton.mp hz]
          simp only [Finset.mem_singleton, iff_true]
          exact (hrB_iff y hy).mpr hyB
    | succ d' =>
      let F := TraceSet R (insert x E)
      let F₁ := F.image fun A => A.erase x
      let F₂ := E.powerset.filter fun B => B ∈ F ∧ insert x B ∈ F
      -- Exact partition identity: F.card = F₁.card + F₂.card
      have hpart : F.card = F₁.card + F₂.card := by
        let F_no := F.filter fun A => x ∉ A
        let F_yes := F.filter fun A => x ∈ A
        have hF_split : F.card = F_no.card + F_yes.card := by
          rw [add_comm]
          exact (Finset.filter_card_add_filter_neg_card_eq_card (s := F) (p := fun A => x ∈ A)).symm
        let F_yes_er := F_yes.image fun A => A.erase x
        have hyes_card : F_yes.card = F_yes_er.card := by
          symm; apply Finset.card_image_of_injOn
          intro A hA B hB hab
          dsimp only at hab
          have hAx : x ∈ A := (Finset.mem_filter.mp hA).2
          have hBx : x ∈ B := (Finset.mem_filter.mp hB).2
          rw [← Finset.insert_erase hAx, ← Finset.insert_erase hBx, hab]
        -- Notice F₁ = F_no ∪ F_yes_er and F₂ = F_no ∩ F_yes_er!
        have hF1_eq : F₁ = F_no ∪ F_yes_er := by
          ext B
          simp only [F₁, F_no, F_yes_er, F_yes, Finset.mem_image, Finset.mem_union, Finset.mem_filter]
          constructor
          · rintro ⟨A, hAF, rfl⟩
            by_cases hAx : x ∈ A
            · right; exact ⟨A, ⟨hAF, hAx⟩, rfl⟩
            · left; rw [Finset.erase_eq_of_notMem hAx]; exact ⟨hAF, hAx⟩
          · rintro (⟨hBF, hBx⟩ | ⟨A, ⟨hAF, _⟩, rfl⟩)
            · exact ⟨B, hBF, Finset.erase_eq_of_notMem hBx⟩
            · exact ⟨A, hAF, rfl⟩
        have hF2_eq : F₂ = F_no ∩ F_yes_er := by
          ext B
          simp only [F₂, F_no, F_yes_er, F_yes, Finset.mem_filter, Finset.mem_powerset,
            Finset.mem_inter, Finset.mem_image]
          constructor
          · rintro ⟨hBE, hBF, hinsF⟩
            have hBx : x ∉ B := fun h => hx (hBE h)
            refine ⟨⟨hBF, hBx⟩, ⟨insert x B, ⟨hinsF, Finset.mem_insert_self x B⟩, ?_⟩⟩
            exact Finset.erase_insert hBx
          · rintro ⟨⟨hBF, hBx⟩, ⟨A, ⟨hAF, hAx⟩, rfl⟩⟩
            have hAE : A.erase x ⊆ E := by
              have hAsub := Finset.mem_powerset.mp (Finset.mem_filter.mp hAF).1
              intro y hy
              have hyne := (Finset.mem_erase.mp hy).1
              have hyins := hAsub (Finset.mem_erase.mp hy).2
              exact (Finset.mem_insert.mp hyins).resolve_left hyne
            refine ⟨hAE, hBF, ?_⟩
            rwa [Finset.insert_erase hAx]
        have hu := Finset.card_union_add_card_inter F_no F_yes_er
        rw [← hF1_eq, ← hF2_eq, ← hyes_card] at hu
        omega
      -- Bound F₁ ⊆ TraceSet R E
      have hF1_sub : F₁ ⊆ TraceSet R E := by
        intro B hB
        simp only [F₁, F, TraceSet, Finset.mem_image, Finset.mem_filter, Finset.mem_powerset] at hB ⊢
        rcases hB with ⟨A, ⟨hAsub, ⟨r, hr, hiff⟩⟩, rfl⟩
        refine ⟨fun y hy => (Finset.mem_insert.mp (hAsub (Finset.mem_erase.mp hy).2)).resolve_left
          (Finset.mem_erase.mp hy).1, ⟨r, hr, fun y hyE => ?_⟩⟩
        have hyx : y ≠ x := fun h => hx (h ▸ hyE)
        rw [Finset.mem_erase, hiff y (Finset.mem_insert_of_mem hyE)]
        simp [hyx]
      have hF1_le : F₁.card ≤ ∑ j ∈ Finset.range (d' + 2), Nat.choose E.card j :=
        (Finset.card_le_card hF1_sub).trans (ih (d' + 1) R hvc)
      -- Bound F₂ using R' = {(B : Set Ω) | B ∈ F₂}, which has VC dim ≤ d'
      let R' : Set (Set Ω) := { s : Set Ω | ∃ B ∈ F₂, s = (B : Set Ω) }
      have hvc' : HasVCDimLe R' d' := by
        intro G hGcard hshatG
        have hGE : G ⊆ E := by
          intro g hg
          obtain ⟨s, ⟨B, hBF2, rfl⟩, hs⟩ := hshatG G Finset.Subset.rfl
          have hgB : g ∈ B := (hs g hg).mpr hg
          exact Finset.mem_powerset.mp (Finset.mem_filter.mp hBF2).1 hgB
        have hgx : x ∉ G := fun h => hx (hGE h)
        apply hvc (insert x G) (by rw [Finset.card_insert_of_notMem hgx, hGcard])
        intro S hS
        let S' := S.erase x
        have hS'G : S' ⊆ G := by
          intro y hy
          exact (Finset.mem_insert.mp (hS (Finset.mem_erase.mp hy).2)).resolve_left
            (Finset.mem_erase.mp hy).1
        obtain ⟨s, ⟨B, hBF2, rfl⟩, hs⟩ := hshatG S' hS'G
        have hBF2' := Finset.mem_filter.mp hBF2
        by_cases hxS : x ∈ S
        · rcases (Finset.mem_filter.mp hBF2'.2.2).2 with ⟨r, hr, hr_iff⟩
          refine ⟨r, hr, fun y hy => ?_⟩
          have hy_ins : y ∈ insert x E := by
            rcases Finset.mem_insert.mp hy with hyx | hyG
            · rw [hyx]; exact Finset.mem_insert_self _ _
            · exact Finset.mem_insert_of_mem (hGE hyG)
          rw [hr_iff y hy_ins]
          rcases Finset.mem_insert.mp hy with hyx | hyG
          · rw [hyx]; simp [hxS]
          · have hyx : y ≠ x := fun h => hgx (h ▸ hyG)
            simp only [Finset.mem_insert, hyx, false_or]
            exact (hs y hyG).trans (by simp [S', hyx])
        · rcases (Finset.mem_filter.mp hBF2'.2.1).2 with ⟨r, hr, hr_iff⟩
          refine ⟨r, hr, fun y hy => ?_⟩
          have hy_ins : y ∈ insert x E := by
            rcases Finset.mem_insert.mp hy with hyx | hyG
            · rw [hyx]; exact Finset.mem_insert_self _ _
            · exact Finset.mem_insert_of_mem (hGE hyG)
          rw [hr_iff y hy_ins]
          rcases Finset.mem_insert.mp hy with hyx | hyG
          · rw [hyx]
            have hBx : x ∉ B := fun h => hx (Finset.mem_powerset.mp hBF2'.1 h)
            simp [hBx, hxS]
          · have hyx : y ≠ x := fun h => hgx (h ▸ hyG)
            exact (hs y hyG).trans (by simp [S', hyx])
      have hF2_sub : F₂ ⊆ TraceSet R' E := by
        intro B hBF2
        refine Finset.mem_filter.mpr ⟨(Finset.mem_filter.mp hBF2).1, ⟨(B : Set Ω), ⟨B, hBF2, rfl⟩, ?_⟩⟩
        intro y _; rfl
      have hF2_le : F₂.card ≤ ∑ j ∈ Finset.range (d' + 1), Nat.choose E.card j :=
        (Finset.card_le_card hF2_sub).trans (ih d' R' hvc')
      rw [Finset.card_insert_of_notMem hx]
      have hpascal : ∑ j ∈ Finset.range (d' + 1 + 1), Nat.choose (E.card + 1) j =
          (∑ j ∈ Finset.range (d' + 1 + 1), Nat.choose E.card j) +
          (∑ j ∈ Finset.range (d' + 1), Nat.choose E.card j) := by
        rw [Finset.sum_range_succ' (fun j => Nat.choose (E.card + 1) j) (d' + 1),
            Finset.sum_range_succ' (fun j => Nat.choose E.card j) (d' + 1)]
        simp only [Nat.choose_zero_right, Nat.choose_succ_succ, Finset.sum_add_distrib,
          Nat.succ_eq_add_one]
        omega
      change F.card ≤ ∑ j ∈ Finset.range (d' + 1 + 1), Nat.choose (E.card + 1) j
      rw [hpart, hpascal]
      exact Nat.add_le_add hF1_le hF2_le

/-- **Binomial prefix sum polynomial bound**: `∑_{j = 0}^d C(s, j) ≤ (s + 1)^d`. -/
theorem sum_choose_le_succ_pow (s d : ℕ) :
    ∑ j ∈ Finset.range (d + 1), Nat.choose s j ≤ (s + 1) ^ d := by
  induction d generalizing s with
  | zero => simp
  | succ d ihd =>
    induction s with
    | zero =>
      rw [Finset.sum_range_succ']
      simp [Nat.choose_zero_succ]
    | succ s ihs =>
      have hpascal : ∑ j ∈ Finset.range (d + 1 + 1), Nat.choose (s + 1) j =
          (∑ j ∈ Finset.range (d + 1 + 1), Nat.choose s j) +
          (∑ j ∈ Finset.range (d + 1), Nat.choose s j) := by
        rw [Finset.sum_range_succ' (fun j => Nat.choose (s + 1) j) (d + 1),
            Finset.sum_range_succ' (fun j => Nat.choose s j) (d + 1)]
        simp only [Nat.choose_zero_right, Nat.choose_succ_succ, Finset.sum_add_distrib,
          Nat.succ_eq_add_one]
        omega
      rw [hpascal]
      have h1 := ihs
      have h2 := ihd s
      calc (∑ j ∈ Finset.range (d + 1 + 1), Nat.choose s j) +
            (∑ j ∈ Finset.range (d + 1), Nat.choose s j)
          ≤ (s + 1) ^ (d + 1) + (s + 1) ^ d := Nat.add_le_add h1 h2
        _ = (s + 1) ^ d * (s + 2) := by rw [pow_succ]; ring
        _ ≤ (s + 2) ^ d * (s + 2) := Nat.mul_le_mul_right _ (Nat.pow_le_pow_left (by omega) d)
        _ = (s + 2) ^ (d + 1) := by rw [pow_succ]

/-- Combined Sauer–Shelah polynomial growth corollary. -/
lemma traceSet_card_le_succ_pow {Ω : Type*} {R : Set (Set Ω)} {d : ℕ}
    (hvc : HasVCDimLe R d) (E : Finset Ω) :
    (TraceSet R E).card ≤ (E.card + 1) ^ d :=
  (sauer_shelah E d R hvc).trans (sum_choose_le_succ_pow E.card d)

/-! ### 4. Finite Double-Sampling `ε`-Net Hitting Lemma (Lemma 3, Haussler–Welzl) -/

/-- Product probability weight `P^n(x) = ∏_{i : Fin n} P (x i)` on `Fin n → Ω`. -/
def prodPMF {Ω : Type*} {n : ℕ} (P : Ω → ℝ) (x : Fin n → Ω) : ℝ :=
  ∏ i : Fin n, P (x i)

/-- Range mass `P(R) = ∑_{ω ∈ R} P(ω)` on a finite ground set `Ω`. -/
noncomputable def rangeMass {Ω : Type*} [Fintype Ω] (P : Ω → ℝ) (R : Set Ω) : ℝ := by
  classical
  exact ∑ ω : Ω, if ω ∈ R then P ω else 0

lemma prodPMF_nonneg {Ω : Type*} {n : ℕ} {P : Ω → ℝ} (hP : ∀ ω, 0 ≤ P ω) (x : Fin n → Ω) :
    0 ≤ prodPMF P x :=
  Finset.prod_nonneg fun i _ => hP (x i)

lemma prodPMF_sum_one {Ω : Type*} [Fintype Ω] {n : ℕ} {P : Ω → ℝ}
    (hsum : ∑ ω : Ω, P ω = 1) :
    ∑ x : Fin n → Ω, prodPMF P x = 1 := by
  classical
  have h := Fintype.prod_sum (fun (_ : Fin n) (ω : Ω) => P ω)
  simpa [prodPMF, hsum] using h.symm

lemma prodPMF_expect_coord {Ω : Type*} [Fintype Ω] {n : ℕ} {P : Ω → ℝ}
    (hsum : ∑ ω : Ω, P ω = 1) (f : Ω → ℝ) (i : Fin n) :
    ∑ x : Fin n → Ω, prodPMF P x * f (x i) = ∑ ω : Ω, P ω * f ω := by
  classical
  have h := Fintype.prod_sum (fun (j : Fin n) (ω : Ω) => if j = i then P ω * f ω else P ω)
  have hlhs : ∏ j : Fin n, ∑ ω : Ω, (if j = i then P ω * f ω else P ω) = ∑ ω : Ω, P ω * f ω := by
    rw [Finset.prod_eq_single i]
    · simp
    · intro j _ hj; simp [hj, hsum]
    · intro h; exact (h (Finset.mem_univ _)).elim
  rw [hlhs] at h
  rw [h]
  refine Finset.sum_congr rfl fun x _ => ?_
  unfold prodPMF
  rw [← Finset.mul_prod_erase Finset.univ (fun j => P (x j)) (Finset.mem_univ i),
      ← Finset.mul_prod_erase Finset.univ (fun j => if j = i then P (x j) * f (x j) else P (x j))
        (Finset.mem_univ i)]
  simp only [↓reduceIte]
  have hrest : ∏ j ∈ Finset.univ.erase i, (if j = i then P (x j) * f (x j) else P (x j)) =
      ∏ j ∈ Finset.univ.erase i, P (x j) := by
    refine Finset.prod_congr rfl fun j hj => ?_
    simp [(Finset.mem_erase.mp hj).1]
  rw [hrest]; ring

lemma prodPMF_expect_two_coords {Ω : Type*} [Fintype Ω] {n : ℕ} {P : Ω → ℝ}
    (hsum : ∑ ω : Ω, P ω = 1) (f g : Ω → ℝ) {i j : Fin n} (hij : i ≠ j) :
    ∑ x : Fin n → Ω, prodPMF P x * (f (x i) * g (x j)) =
      (∑ ω : Ω, P ω * f ω) * (∑ ω : Ω, P ω * g ω) := by
  classical
  let φ : Fin n → Ω → ℝ := fun k ω =>
    if k = i then P ω * f ω else if k = j then P ω * g ω else P ω
  have h := Fintype.prod_sum φ
  have hlhs : ∏ k : Fin n, ∑ ω : Ω, φ k ω = (∑ ω : Ω, P ω * f ω) * (∑ ω : Ω, P ω * g ω) := by
    rw [← Finset.mul_prod_erase Finset.univ (fun k => ∑ ω : Ω, φ k ω) (Finset.mem_univ i)]
    have hj_mem : j ∈ Finset.univ.erase i := Finset.mem_erase.mpr ⟨hij.symm, Finset.mem_univ _⟩
    rw [← Finset.mul_prod_erase (Finset.univ.erase i) (fun k => ∑ ω : Ω, φ k ω) hj_mem]
    have hrest : ∏ k ∈ (Finset.univ.erase i).erase j, ∑ ω : Ω, φ k ω = 1 := by
      refine Finset.prod_eq_one fun k hk => ?_
      have hkj := (Finset.mem_erase.mp hk).1
      have hki := (Finset.mem_erase.mp (Finset.mem_erase.mp hk).2).1
      simp [φ, hki, hkj, hsum]
    rw [hrest, mul_one]
    simp [φ, hij.symm]
  rw [hlhs] at h
  rw [h]
  refine Finset.sum_congr rfl fun x _ => ?_
  unfold prodPMF
  rw [← Finset.mul_prod_erase Finset.univ (fun k => P (x k)) (Finset.mem_univ i),
      ← Finset.mul_prod_erase Finset.univ (fun k => φ k (x k)) (Finset.mem_univ i)]
  have hj_mem : j ∈ Finset.univ.erase i := Finset.mem_erase.mpr ⟨hij.symm, Finset.mem_univ _⟩
  rw [← Finset.mul_prod_erase (Finset.univ.erase i) (fun k => P (x k)) hj_mem,
      ← Finset.mul_prod_erase (Finset.univ.erase i) (fun k => φ k (x k)) hj_mem]
  have hrest : ∏ k ∈ (Finset.univ.erase i).erase j, φ k (x k) =
      ∏ k ∈ (Finset.univ.erase i).erase j, P (x k) := by
    refine Finset.prod_congr rfl fun k hk => ?_
    have hkj := (Finset.mem_erase.mp hk).1
    have hki := (Finset.mem_erase.mp (Finset.mem_erase.mp hk).2).1
    simp [φ, hki, hkj]
  rw [hrest]
  simp only [φ, ↓reduceIte, hij.symm]
  ring

/-- Exact Chebyshev tail bound for i.i.d. Bernoulli hit count:
if `P(R) ≥ ε` and `n ε ≥ 8`, then `P^n(count_R(x) ≥ ε n / 2) ≥ 1 / 2`. -/
lemma binomial_chebyshev_half {Ω : Type*} [Fintype Ω] {n : ℕ} {P : Ω → ℝ}
    (hP : ∀ ω, 0 ≤ P ω) (hsum : ∑ ω : Ω, P ω = 1)
    (R : Set Ω) [DecidablePred (· ∈ R)] {ε : ℝ} (hε : 0 < ε) (hnε : 8 ≤ (n : ℝ) * ε)
    (hmass : ε ≤ rangeMass P R) :
    (1 / 2 : ℝ) ≤ ∑ x : Fin n → Ω,
      if ε * (n : ℝ) / 2 ≤ ((Finset.univ.filter fun i : Fin n => x i ∈ R).card : ℝ)
      then prodPMF P x else 0 := by
  let p := rangeMass P R
  have hp_pos : 0 < p := lt_of_lt_of_le hε hmass
  have hp_le1 : p ≤ 1 := by
    calc p = ∑ ω : Ω, if ω ∈ R then P ω else 0 := by
          unfold p rangeMass; refine Finset.sum_congr rfl fun ω _ => by split_ifs <;> rfl
      _ ≤ ∑ ω : Ω, P ω := Finset.sum_le_sum fun ω _ => by split_ifs <;> linarith [hP ω]
      _ = 1 := hsum
  let ind : Ω → ℝ := fun ω => if ω ∈ R then 1 else 0
  let cnt : (Fin n → Ω) → ℝ := fun x => ∑ i : Fin n, ind (x i)
  have hcnt_eq : ∀ x : Fin n → Ω,
      cnt x = ((Finset.univ.filter fun i : Fin n => x i ∈ R).card : ℝ) := by
    intro x; simp [cnt, ind, Finset.sum_boole]
  have hind_mean : ∑ ω : Ω, P ω * ind ω = p := by
    simp only [p, rangeMass, ind]
    refine Finset.sum_congr rfl fun ω _ => by split_ifs <;> ring
  have hind_var : ∑ ω : Ω, P ω * ((ind ω - p) * (ind ω - p)) = p * (1 - p) := by
    have hexp : ∀ ω, P ω * ((ind ω - p) * (ind ω - p)) =
        P ω * ind ω * (1 - 2 * p) + P ω * p ^ 2 := by
      intro ω; simp only [ind]; split_ifs <;> ring
    simp_rw [hexp, Finset.sum_add_distrib, ← Finset.sum_mul, hind_mean, hsum]; ring
  have hind_zero : ∑ ω : Ω, P ω * (ind ω - p) = 0 := by
    have : ∀ ω, P ω * (ind ω - p) = P ω * ind ω - P ω * p := fun ω => by ring
    simp_rw [this, Finset.sum_sub_distrib, ← Finset.sum_mul, hind_mean, hsum]; ring
  have hvar : ∑ x : Fin n → Ω, prodPMF P x * (cnt x - (n : ℝ) * p) ^ 2 =
      (n : ℝ) * p * (1 - p) := by
    have hdiff : ∀ x : Fin n → Ω, cnt x - (n : ℝ) * p = ∑ i : Fin n, (ind (x i) - p) := by
      intro x; simp [cnt, Finset.sum_sub_distrib]
    have hsq : ∀ x : Fin n → Ω, (cnt x - (n : ℝ) * p) ^ 2 =
        ∑ i : Fin n, ∑ j : Fin n, (ind (x i) - p) * (ind (x j) - p) := by
      intro x; rw [hdiff x, sq, Finset.sum_mul]
      refine Finset.sum_congr rfl fun i _ => Finset.mul_sum _ _ _
    simp_rw [hsq, Finset.mul_sum]
    rw [Finset.sum_comm]
    calc ∑ i : Fin n, ∑ x : Fin n → Ω, ∑ j : Fin n,
            prodPMF P x * ((ind (x i) - p) * (ind (x j) - p))
        = ∑ i : Fin n, ∑ j : Fin n, ∑ x : Fin n → Ω,
            prodPMF P x * ((ind (x i) - p) * (ind (x j) - p)) :=
          Finset.sum_congr rfl fun i _ => Finset.sum_comm
      _ = ∑ i : Fin n, (p * (1 - p)) := by
          refine Finset.sum_congr rfl fun i _ => ?_
          rw [Finset.sum_eq_single i]
          · rw [prodPMF_expect_coord hsum (fun ω => (ind ω - p) * (ind ω - p)) i, hind_var]
          · intro j _ hji
            rw [prodPMF_expect_two_coords hsum (fun ω => ind ω - p) (fun ω => ind ω - p) (Ne.symm hji),
              hind_zero, mul_zero]
          · intro h; exact (h (Finset.mem_univ _)).elim
      _ = (n : ℝ) * p * (1 - p) := by simp [mul_assoc]
  let bad : Finset (Fin n → Ω) :=
    Finset.univ.filter fun x => ¬ (ε * (n : ℝ) / 2 ≤ ((Finset.univ.filter fun i : Fin n => x i ∈ R).card : ℝ))
  have hbad_le : ∑ x ∈ bad, prodPMF P x ≤ 1 / 2 := by
    have hnp_pos : 0 < (n : ℝ) * p := by
      have : 8 ≤ (n : ℝ) * p := le_trans hnε (mul_le_mul_of_nonneg_left hmass (Nat.cast_nonneg n))
      linarith
    have hcheb : ((n : ℝ) * p / 2) ^ 2 * ∑ x ∈ bad, prodPMF P x ≤
        ∑ x : Fin n → Ω, prodPMF P x * (cnt x - (n : ℝ) * p) ^ 2 := by
      rw [Finset.mul_sum]
      calc ∑ x ∈ bad, ((n : ℝ) * p / 2) ^ 2 * prodPMF P x
          ≤ ∑ x ∈ bad, prodPMF P x * (cnt x - (n : ℝ) * p) ^ 2 := by
            refine Finset.sum_le_sum fun x hx => ?_
            have hlt : cnt x < ε * (n : ℝ) / 2 := by
              rw [hcnt_eq x]; exact lt_of_not_ge (Finset.mem_filter.mp hx).2
            have hgap : (n : ℝ) * p / 2 ≤ (n : ℝ) * p - cnt x := by
              have : ε * (n : ℝ) ≤ (n : ℝ) * p := by
                rw [mul_comm]; exact mul_le_mul_of_nonneg_left hmass (Nat.cast_nonneg n)
              linarith
            have hsq_le : ((n : ℝ) * p / 2) ^ 2 ≤ (cnt x - (n : ℝ) * p) ^ 2 := by
              have : (cnt x - (n : ℝ) * p) ^ 2 = ((n : ℝ) * p - cnt x) ^ 2 := by ring
              rw [this]; exact pow_le_pow_left₀ (by linarith) hgap 2
            nlinarith [prodPMF_nonneg hP x]
        _ ≤ ∑ x : Fin n → Ω, prodPMF P x * (cnt x - (n : ℝ) * p) ^ 2 :=
            Finset.sum_le_univ_sum_of_nonneg fun x => mul_nonneg (prodPMF_nonneg hP x) (sq_nonneg _)
    rw [hvar] at hcheb
    have hnp8 : 8 ≤ (n : ℝ) * p :=
      le_trans hnε (mul_le_mul_of_nonneg_left hmass (Nat.cast_nonneg n))
    have hvar_le : (n : ℝ) * p * (1 - p) ≤ (n : ℝ) * p := by nlinarith
    have hden_pos : 0 < ((n : ℝ) * p / 2) ^ 2 := by positivity
    have hdiv : ∑ x ∈ bad, prodPMF P x ≤ ((n : ℝ) * p) / ((n : ℝ) * p / 2) ^ 2 := by
      rw [le_div_iff₀ hden_pos]; linarith
    have h4np : ((n : ℝ) * p) / ((n : ℝ) * p / 2) ^ 2 = 4 / ((n : ℝ) * p) := by
      field_simp [hnp_pos.ne']; ring
    have hhalf : 4 / ((n : ℝ) * p) ≤ 1 / 2 := by
      rw [div_le_div_iff₀ hnp_pos (by norm_num : (0 : ℝ) < 2)]; linarith
    linarith
  have hsplit : (∑ x : Fin n → Ω,
      if ε * (n : ℝ) / 2 ≤ ((Finset.univ.filter fun i : Fin n => x i ∈ R).card : ℝ)
      then prodPMF P x else 0) + (∑ x ∈ bad, prodPMF P x) = 1 := by
    rw [← prodPMF_sum_one (n := n) hsum]
    simp only [bad, Finset.sum_filter]
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun x _ => by split_ifs <;> ring
  linarith

/-- Exact factorial-descending-factorial identity: `a! = a.descFactorial b * (a - b)!` for `b ≤ a`. -/
private lemma factorial_eq_descFactorial_mul (a b : ℕ) (h : b ≤ a) :
    Nat.factorial a = a.descFactorial b * Nat.factorial (a - b) := by
  induction b with
  | zero => simp
  | succ b ih =>
    have hb : b ≤ a := by omega
    rw [ih hb, Nat.descFactorial_succ]
    have hsub : a - b = (a - (b + 1)) + 1 := by omega
    rw [hsub, Nat.factorial_succ]
    ring

/-- Exact real product expansion of descending factorials. -/
private lemma descFactorial_eq_prod_real (m a : ℕ) :
    ((a.descFactorial m : ℕ) : ℝ) = ∏ j ∈ Finset.range m, ((a - j : ℕ) : ℝ) := by
  induction m with
  | zero => simp
  | succ k ih =>
    rw [Nat.descFactorial_succ, Nat.cast_mul, Finset.prod_range_succ, mul_comm, ih]

/-- Exact permutation-counting hypergeometric avoidance bound:
for any `I ⊆ Fin (2 * n)` with `I.card ≥ m` and `n ≥ 1`, the fraction of permutations
`σ : Perm (Fin (2 * n))` mapping `{0, ..., n - 1}` outside `I` is at most `exp(-ε n / 4)`. -/
lemma perm_avoid_fraction_le_exp {n : ℕ} (hn : 1 ≤ n) (I : Finset (Fin (2 * n)))
    {ε : ℝ} (_hε0 : 0 ≤ ε) (hI : ε * (n : ℝ) / 2 ≤ (I.card : ℝ)) :
    ((Finset.univ.filter fun σ : Equiv.Perm (Fin (2 * n)) =>
        ∀ i : Fin n, σ ⟨i.val, by omega⟩ ∉ I).card : ℝ) /
      ((Nat.factorial (2 * n) : ℝ)) ≤ Real.exp (-(ε * (n : ℝ) / 4)) := by
  classical
  let avoidPerms : Finset (Equiv.Perm (Fin (2 * n))) :=
    Finset.univ.filter fun σ => ∀ i : Fin n, σ ⟨i.val, by omega⟩ ∉ I
  let N := 2 * n
  let m := I.card
  have hcard_exact : avoidPerms.card = (N - m).descFactorial n * Nat.factorial n := by
    let Comp : Type := { k : Fin N // k ∉ I }
    have hComp_card : Fintype.card Comp = N - m := by
      rw [Fintype.card_subtype, Finset.filter_notMem_eq_sdiff, Finset.card_sdiff,
        Finset.inter_univ, Finset.card_univ, Fintype.card_fin]
    let restr : avoidPerms → (Fin n ↪ Comp) := fun σ =>
      ⟨fun i => ⟨σ.1 ⟨i.val, by omega⟩, (Finset.mem_filter.mp σ.2).2 i⟩,
       fun i₁ i₂ h => Fin.ext (by
         have h1 : σ.1 ⟨i₁.val, by omega⟩ = σ.1 ⟨i₂.val, by omega⟩ := congrArg Subtype.val h
         have h2 := σ.1.injective h1
         simpa using congrArg Fin.val h2)⟩
    have hfiber : ∀ f : Fin n ↪ Comp,
        (Finset.univ.filter fun σ : avoidPerms => restr σ = f).card = Nat.factorial n := by
      intro f
      let img : Finset (Fin N) := Finset.univ.image fun i : Fin n => (f i).1
      have himg_card : img.card = n := by
        rw [Finset.card_image_of_injective _ (fun i₁ i₂ h => f.injective (Subtype.ext h)),
          Finset.card_univ, Fintype.card_fin]
      let RestDom := { k : Fin N // n ≤ k.val }
      let RestCod := { k : Fin N // k ∉ img }
      have hdom_card : Fintype.card RestDom = n := by
        have hbij : (Finset.univ.filter fun k : Fin N => n ≤ k.val) =
            Finset.univ.image (fun i : Fin n => (⟨n + i.val, by omega⟩ : Fin N)) := by
          ext k; simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_image]
          exact ⟨fun hk => ⟨⟨k.val - n, by omega⟩, Fin.ext (by dsimp; omega)⟩,
            fun ⟨i, hi⟩ => by rw [← hi]; dsimp; omega⟩
        rw [Fintype.card_subtype, hbij,
          Finset.card_image_of_injective _ (fun i₁ i₂ h => Fin.ext (by
            have := congrArg Fin.val h; dsimp at this; omega)),
          Finset.card_univ, Fintype.card_fin]
      have hcod_card : Fintype.card RestCod = n := by
        rw [Fintype.card_subtype, Finset.filter_notMem_eq_sdiff, Finset.card_sdiff,
          Finset.inter_univ, Finset.card_univ, Fintype.card_fin, himg_card]
        omega
      let eFib : { σ : avoidPerms // restr σ = f } ≃ (RestDom ≃ RestCod) :=
        { toFun := fun ⟨σ, hσ⟩ =>
            { toFun := fun ⟨k, hk⟩ => ⟨σ.1 k, by
                simp only [img, Finset.mem_image, Finset.mem_univ, true_and]
                rintro ⟨i, hi⟩
                have hfi : (f i).1 = σ.1 ⟨i.val, by omega⟩ :=
                  congrArg (fun g : Fin n ↪ Comp => (g i).1) hσ.symm
                rw [hfi] at hi
                have := congrArg Fin.val (σ.1.injective hi)
                dsimp at this; omega⟩
              invFun := fun ⟨y, hy⟩ => ⟨σ.1.symm y, by
                by_contra hlt; push_neg at hlt
                apply hy
                simp only [img, Finset.mem_image, Finset.mem_univ, true_and]
                refine ⟨⟨(σ.1.symm y).val, hlt⟩, ?_⟩
                have hfi : (f ⟨(σ.1.symm y).val, hlt⟩).1 = σ.1 (σ.1.symm y) :=
                  congrArg (fun g : Fin n ↪ Comp => (g ⟨(σ.1.symm y).val, hlt⟩).1) hσ.symm
                simpa using hfi⟩
              left_inv := fun ⟨k, _⟩ => Subtype.ext (σ.1.symm_apply_apply k)
              right_inv := fun ⟨y, _⟩ => Subtype.ext (σ.1.apply_symm_apply y) }
          invFun := fun ψ =>
            let g : Fin N → Fin N := fun k =>
              if hk : k.val < n then (f ⟨k.val, hk⟩).1 else (ψ ⟨k, by omega⟩).1
            have hg_inj : Function.Injective g := by
              intro k₁ k₂ h
              simp only [g] at h
              split_ifs at h with h1 h2 h2
              · have := f.injective (Subtype.ext h)
                exact Fin.ext (by simpa using congrArg Fin.val this)
              · exfalso
                have := (ψ ⟨k₂, by omega⟩).2
                exact this (Finset.mem_image.mpr ⟨⟨k₁.val, h1⟩, Finset.mem_univ _, h⟩)
              · exfalso
                have := (ψ ⟨k₁, by omega⟩).2
                exact this (Finset.mem_image.mpr ⟨⟨k₂.val, h2⟩, Finset.mem_univ _, h.symm⟩)
              · exact Fin.ext (congrArg Fin.val (congrArg Subtype.val (ψ.injective (Subtype.ext h))))
            let σp : Equiv.Perm (Fin N) := Equiv.ofBijective g (Finite.injective_iff_bijective.mp hg_inj)
            have hσp_av : σp ∈ avoidPerms := by
              simp only [avoidPerms, Finset.mem_filter, Finset.mem_univ, true_and]
              intro i
              have : σp ⟨i.val, by omega⟩ = (f i).1 := by simp [σp, g, i.isLt]
              rw [this]; exact (f i).2
            ⟨⟨σp, hσp_av⟩, by
              apply DFunLike.ext; intro i; apply Subtype.ext
              show σp ⟨i.val, by omega⟩ = (f i).1
              simp [σp, g, i.isLt]⟩
          left_inv := fun ⟨⟨σ, hσ⟩, hrf⟩ => by
            apply Subtype.ext; apply Subtype.ext; apply Equiv.ext; intro k
            change (if hk : k.val < n then (f ⟨k.val, hk⟩).1 else σ k) = σ k
            split_ifs with hk
            · exact congrArg (fun g : Fin n ↪ Comp => (g ⟨k.val, hk⟩).1) hrf.symm
            · rfl
          right_inv := fun ψ => by
            apply Equiv.ext; rintro ⟨k, hk⟩
            apply Subtype.ext
            change (if h : k.val < n then (f ⟨k.val, h⟩).1 else (ψ ⟨k, by omega⟩).1) = (ψ ⟨k, hk⟩).1
            have hnk : ¬ k.val < n := by omega
            simp [hnk] }
      have hcard_fib : (Finset.univ.filter fun σ : avoidPerms => restr σ = f).card =
          Fintype.card (RestDom ≃ RestCod) := by
        rw [← Fintype.card_subtype]
        exact Fintype.card_congr eFib
      have hcd : RestDom ≃ RestCod := Fintype.equivOfCardEq (hdom_card.trans hcod_card.symm)
      have heq_perm : (RestDom ≃ RestCod) ≃ Equiv.Perm RestDom :=
        Equiv.equivCongr (Equiv.refl RestDom) hcd.symm
      rw [hcard_fib, Fintype.card_congr heq_perm, Fintype.card_perm, hdom_card]
    calc avoidPerms.card = Fintype.card avoidPerms := (Fintype.card_coe _).symm
      _ = ∑ f : Fin n ↪ Comp, (Finset.univ.filter fun σ : avoidPerms => restr σ = f).card := by
          rw [← Finset.card_univ]
          exact Finset.card_eq_sum_card_fiberwise (f := restr) (s := Finset.univ)
            (t := Finset.univ) (fun _ _ => Finset.mem_univ _)
      _ = ∑ _f : Fin n ↪ Comp, Nat.factorial n := Finset.sum_congr rfl fun f _ => hfiber f
      _ = Fintype.card (Fin n ↪ Comp) * Nat.factorial n := by simp
      _ = (N - m).descFactorial n * Nat.factorial n := by
          rw [Fintype.card_embedding_eq, Fintype.card_fin, hComp_card]
  rw [hcard_exact]
  have hfact_eq : Nat.factorial (2 * n) = (2 * n).descFactorial n * Nat.factorial n := by
    have h := factorial_eq_descFactorial_mul (2 * n) n (by omega)
    have hsub : 2 * n - n = n := by omega
    rwa [hsub] at h
  rw [hfact_eq]
  have hfact_pos : (0 : ℝ) < (Nat.factorial n : ℝ) := by positivity
  rw [Nat.cast_mul, Nat.cast_mul, mul_div_mul_right _ _ hfact_pos.ne']
  rw [descFactorial_eq_prod_real n (N - m), descFactorial_eq_prod_real n (2 * n), ← Finset.prod_div_distrib]
  have hexp_prod : Real.exp (-(ε * (n : ℝ) / 4)) = ∏ _j ∈ Finset.range n, Real.exp (-(ε / 4)) := by
    rw [Finset.prod_const, Finset.card_range, ← Real.exp_nat_mul]
    congr 1; ring
  rw [hexp_prod]
  refine Finset.prod_le_prod (fun j _ => by positivity) fun j hj => ?_
  have hjlt : j < n := Finset.mem_range.mp hj
  have hden_pos : (0 : ℝ) < ((2 * n - j : ℕ) : ℝ) := by
    have : 0 < 2 * n - j := by omega
    exact_mod_cast this
  have hfactor : (((N - m - j : ℕ) : ℝ)) / (((2 * n - j : ℕ) : ℝ)) ≤ 1 - ε / 4 := by
    by_cases hle : N - m ≤ j
    · have h0 : N - m - j = 0 := Nat.sub_eq_zero_of_le hle
      rw [h0, Nat.cast_zero, zero_div]
      -- Since m ≤ N = 2n, ε n / 2 ≤ m ≤ 2n ⟹ ε ≤ 4 ⟹ 1 - ε/4 ≥ 0!
      have hm_le : (m : ℝ) ≤ 2 * (n : ℝ) := by
        have : m ≤ 2 * n := by
          calc m = I.card := rfl
            _ ≤ (Finset.univ : Finset (Fin (2 * n))).card := Finset.card_le_card (Finset.subset_univ _)
            _ = 2 * n := by simp
        exact_mod_cast this
      have hn_pos : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
      have hε4 : ε ≤ 4 := by nlinarith
      linarith
    · push_neg at hle
      have hcast_num : ((N - m - j : ℕ) : ℝ) = (2 * (n : ℝ) - (j : ℝ)) - (m : ℝ) := by
        have h1 : m + j ≤ 2 * n := by omega
        have h2 : N - m - j = 2 * n - (m + j) := by omega
        rw [h2, Nat.cast_sub h1]; push_cast; ring
      have hcast_den : ((2 * n - j : ℕ) : ℝ) = 2 * (n : ℝ) - (j : ℝ) := by
        rw [Nat.cast_sub (by omega)]; push_cast; ring
      rw [hcast_num, hcast_den, sub_div, div_self (by linarith)]
      have h1 : ε / 4 ≤ (m : ℝ) / (2 * (n : ℝ)) := by
        have hn_pos : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
        rw [div_le_div_iff₀ (by norm_num) (by positivity)]; linarith
      have h2 : (m : ℝ) / (2 * (n : ℝ)) ≤ (m : ℝ) / (2 * (n : ℝ) - (j : ℝ)) := by
        have hden' : 0 < 2 * (n : ℝ) - (j : ℝ) := by linarith
        rw [div_le_div_iff₀ (by positivity) hden']
        have : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg m
        have : (0 : ℝ) ≤ (j : ℝ) := Nat.cast_nonneg j
        nlinarith
      linarith
  exact hfactor.trans (by
    have := Real.add_one_le_exp (-(ε / 4))
    linarith)

open Classical in
/-- **Lemma 3 (Finite `ε`-net hitting lemma; Haussler and Welzl, 1987)**:
Let `P` be a PMF on a finite ground set `Ω`, `ℛ ⊆ 2^Ω` a range family whose traces on every
subset of size `≤ 2n` have cardinality `≤ M_grow`, and `n ≥ 1`, `ε ∈ (0, 1]` with `n ε ≥ 8`.
Then the `P^n`-probability that `x : Fin n → Ω` misses some `R ∈ ℛ` with `P(R) ≥ ε` is at most
`2 * M_grow * exp(-ε n / 4)`. -/
theorem finite_epsilon_net_bound {Ω : Type*} [Fintype Ω]
    {P : Ω → ℝ} (hP : ∀ ω, 0 ≤ P ω) (hsum : ∑ ω : Ω, P ω = 1)
    (ℛ : Set (Set Ω)) {n M_grow : ℕ} (hn : 1 ≤ n)
    (hgrow : ∀ E : Finset Ω, E.card ≤ 2 * n → (TraceSet ℛ E).card ≤ M_grow)
    {ε : ℝ} (hε : 0 < ε) (hnε : 8 ≤ (n : ℝ) * ε) :
    (∑ x : Fin n → Ω,
      if (∃ R ∈ ℛ, ε ≤ rangeMass P R ∧ ∀ i : Fin n, x i ∉ R)
      then prodPMF P x else 0) ≤
    2 * (M_grow : ℝ) * Real.exp (-(ε * (n : ℝ) / 4)) := by
  let E1 : (Fin n → Ω) → Prop := fun x =>
    ∃ R ∈ ℛ, ε ≤ rangeMass P R ∧ ∀ i : Fin n, x i ∉ R
  let E2 : (Fin (2 * n) → Ω) → Prop := fun y =>
    ∃ R ∈ ℛ, ε ≤ rangeMass P R ∧
      (∀ i : Fin n, y ⟨i.val, by omega⟩ ∉ R) ∧
      ε * (n : ℝ) / 2 ≤ ((Finset.univ.filter fun i : Fin n => y ⟨n + i.val, by omega⟩ ∈ R).card : ℝ)
  let splitEquiv : (Fin (2 * n) → Ω) ≃ (Fin n → Ω) × (Fin n → Ω) :=
    { toFun := fun y => (fun i => y ⟨i.val, by omega⟩, fun i => y ⟨n + i.val, by omega⟩)
      invFun := fun p k => if hk : k.val < n then p.1 ⟨k.val, hk⟩ else p.2 ⟨k.val - n, by omega⟩
      left_inv := fun y => by
        funext k; dsimp only; split_ifs with hk
        · congr 1
        · congr 1; ext; dsimp; omega
      right_inv := fun ⟨x, x'⟩ => by
        apply Prod.ext <;> funext i <;> simp [i.isLt] }
  have hprod_split : ∀ y : Fin (2 * n) → Ω,
      prodPMF P y = prodPMF P (splitEquiv y).1 * prodPMF P (splitEquiv y).2 := by
    intro y
    unfold prodPMF
    let eFin : Fin n ⊕ Fin n ≃ Fin (2 * n) :=
      { toFun := Sum.elim (fun i => ⟨i.val, by omega⟩) (fun i => ⟨n + i.val, by omega⟩)
        invFun := fun k => if hk : k.val < n then Sum.inl ⟨k.val, hk⟩ else Sum.inr ⟨k.val - n, by omega⟩
        left_inv := fun s => by cases s with
          | inl i => simp [i.isLt]
          | inr i =>
            dsimp only [Sum.elim]
            split_ifs with hk
            · omega
            · congr 1; ext; dsimp; omega
        right_inv := fun k => by dsimp only; split_ifs with hk <;> (ext; dsimp; try omega) }
    rw [← Equiv.prod_comp eFin (fun k => P (y k)), Fintype.prod_sum_type]
    rfl
  have hsymm : (1 / 2 : ℝ) * (∑ x : Fin n → Ω, if E1 x then prodPMF P x else 0) ≤
      ∑ y : Fin (2 * n) → Ω, if E2 y then prodPMF P y else 0 := by
    have hrhs_eq : (∑ y : Fin (2 * n) → Ω, if E2 y then prodPMF P y else 0) =
        ∑ x : Fin n → Ω, prodPMF P x *
          (∑ x' : Fin n → Ω, if E2 (splitEquiv.symm (x, x')) then prodPMF P x' else 0) := by
      rw [← Equiv.sum_comp splitEquiv.symm]
      rw [Fintype.sum_prod_type]
      refine Finset.sum_congr rfl fun x _ => ?_
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun x' _ => ?_
      rw [hprod_split (splitEquiv.symm (x, x')), Equiv.apply_symm_apply]
      split_ifs <;> ring
    rw [hrhs_eq, Finset.mul_sum]
    refine Finset.sum_le_sum fun x _ => ?_
    by_cases hx : E1 x
    · simp only [hx, ↓reduceIte]
      rcases hx with ⟨R, hR, hRmass, hRmiss⟩
      have hcheb := binomial_chebyshev_half hP hsum R hε hnε hRmass
      have hsub : (∑ x' : Fin n → Ω,
            if ε * (n : ℝ) / 2 ≤ ((Finset.univ.filter fun i : Fin n => x' i ∈ R).card : ℝ)
            then prodPMF P x' else 0) ≤
          (∑ x' : Fin n → Ω, if E2 (splitEquiv.symm (x, x')) then prodPMF P x' else 0) := by
        refine Finset.sum_le_sum fun x' _ => ?_
        split_ifs with h1 h2
        · exact le_rfl
        · exfalso; apply h2
          refine ⟨R, hR, hRmass, ?_, ?_⟩
          · intro i; simpa [splitEquiv, i.isLt] using hRmiss i
          · simpa [splitEquiv] using h1
        · exact prodPMF_nonneg hP x'
        · exact le_rfl
      have hge := hcheb.trans hsub
      nlinarith [prodPMF_nonneg hP x]
    · simp only [hx, ↓reduceIte, mul_zero]
      exact mul_nonneg (prodPMF_nonneg hP x)
        (Finset.sum_nonneg fun x' _ => by split_ifs; exact prodPMF_nonneg hP x'; exact le_rfl)
  have hfact_pos : (0 : ℝ) < (Nat.factorial (2 * n) : ℝ) := by positivity
  have hexch : (∑ y : Fin (2 * n) → Ω, if E2 y then prodPMF P y else 0) =
      ∑ y : Fin (2 * n) → Ω, prodPMF P y *
        (((Finset.univ.filter fun σ : Equiv.Perm (Fin (2 * n)) => E2 (y ∘ σ)).card : ℝ) /
          (Nat.factorial (2 * n) : ℝ)) := by
    have hperm_inv : ∀ σ : Equiv.Perm (Fin (2 * n)),
        (∑ y : Fin (2 * n) → Ω, if E2 y then prodPMF P y else 0) =
        ∑ y : Fin (2 * n) → Ω, if E2 (y ∘ σ) then prodPMF P y else 0 := by
      intro σ
      let eσ : (Fin (2 * n) → Ω) ≃ (Fin (2 * n) → Ω) :=
        { toFun := fun y => y ∘ σ
          invFun := fun y => y ∘ σ.symm
          left_inv := fun y => by funext k; simp
          right_inv := fun y => by funext k; simp }
      rw [← Equiv.sum_comp eσ]
      refine Finset.sum_congr rfl fun y _ => ?_
      have hpmf : prodPMF P (y ∘ σ) = prodPMF P y := Equiv.prod_comp σ (fun k => P (y k))
      simp [eσ, hpmf]
    have hsum_perm : (Nat.factorial (2 * n) : ℝ) *
          (∑ y : Fin (2 * n) → Ω, if E2 y then prodPMF P y else 0) =
        ∑ y : Fin (2 * n) → Ω, prodPMF P y *
          ((Finset.univ.filter fun σ : Equiv.Perm (Fin (2 * n)) => E2 (y ∘ σ)).card : ℝ) := by
      calc (Nat.factorial (2 * n) : ℝ) * (∑ y : Fin (2 * n) → Ω, if E2 y then prodPMF P y else 0)
          = ∑ _σ : Equiv.Perm (Fin (2 * n)),
              (∑ y : Fin (2 * n) → Ω, if E2 y then prodPMF P y else 0) := by
            rw [Finset.sum_const, Finset.card_univ, Fintype.card_perm, Fintype.card_fin, nsmul_eq_mul]
        _ = ∑ σ : Equiv.Perm (Fin (2 * n)),
              (∑ y : Fin (2 * n) → Ω, if E2 (y ∘ σ) then prodPMF P y else 0) :=
            Finset.sum_congr rfl fun σ _ => hperm_inv σ
        _ = ∑ y : Fin (2 * n) → Ω,
              ∑ σ : Equiv.Perm (Fin (2 * n)), if E2 (y ∘ σ) then prodPMF P y else 0 := Finset.sum_comm
        _ = ∑ y : Fin (2 * n) → Ω, prodPMF P y *
              ((Finset.univ.filter fun σ : Equiv.Perm (Fin (2 * n)) => E2 (y ∘ σ)).card : ℝ) := by
            refine Finset.sum_congr rfl fun y _ => ?_
            rw [Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const, nsmul_eq_mul, mul_comm]
    calc (∑ y : Fin (2 * n) → Ω, if E2 y then prodPMF P y else 0)
        = ((Nat.factorial (2 * n) : ℝ) *
            (∑ y : Fin (2 * n) → Ω, if E2 y then prodPMF P y else 0)) /
            (Nat.factorial (2 * n) : ℝ) := by field_simp [hfact_pos.ne']
      _ = (∑ y : Fin (2 * n) → Ω, prodPMF P y *
            ((Finset.univ.filter fun σ : Equiv.Perm (Fin (2 * n)) => E2 (y ∘ σ)).card : ℝ)) /
            (Nat.factorial (2 * n) : ℝ) := by rw [hsum_perm]
      _ = ∑ y : Fin (2 * n) → Ω, prodPMF P y *
            (((Finset.univ.filter fun σ : Equiv.Perm (Fin (2 * n)) => E2 (y ∘ σ)).card : ℝ) /
              (Nat.factorial (2 * n) : ℝ)) := by
          rw [Finset.sum_div]; refine Finset.sum_congr rfl fun y _ => by ring
  have hpointwise : ∀ y : Fin (2 * n) → Ω,
      ((Finset.univ.filter fun σ : Equiv.Perm (Fin (2 * n)) => E2 (y ∘ σ)).card : ℝ) /
        (Nat.factorial (2 * n) : ℝ) ≤ (M_grow : ℝ) * Real.exp (-(ε * (n : ℝ) / 4)) := by
    intro y
    let Ey : Finset Ω := Finset.univ.image y
    have hEy_card : Ey.card ≤ 2 * n := by
      calc Ey.card ≤ (Finset.univ : Finset (Fin (2 * n))).card := Finset.card_image_le
        _ = 2 * n := by simp
    have hTr_card : (TraceSet ℛ Ey).card ≤ M_grow := hgrow Ey hEy_card
    let idxOf : Finset Ω → Finset (Fin (2 * n)) := fun A => Finset.univ.filter fun k => y k ∈ A
    let heavyTraces : Finset (Finset Ω) :=
      (TraceSet ℛ Ey).filter fun A => ε * (n : ℝ) / 2 ≤ ((idxOf A).card : ℝ)
    have hsub_union : (Finset.univ.filter fun σ : Equiv.Perm (Fin (2 * n)) => E2 (y ∘ σ)) ⊆
        heavyTraces.biUnion (fun A => Finset.univ.filter fun σ : Equiv.Perm (Fin (2 * n)) =>
          ∀ i : Fin n, σ ⟨i.val, by omega⟩ ∉ idxOf A) := by
      intro σ hσ
      rcases (Finset.mem_filter.mp hσ).2 with ⟨R, hR, _, hmiss, hhits⟩
      let A : Finset Ω := Ey.filter fun ω => ω ∈ R
      have hA_tr : A ∈ TraceSet ℛ Ey :=
        mem_TraceSet.mpr ⟨Finset.filter_subset _ _, ⟨R, hR, fun ω hω => by simp [A, hω]⟩⟩
      have hidx_card : ((Finset.univ.filter fun i : Fin n => y (σ ⟨n + i.val, by omega⟩) ∈ R).card) ≤
          (idxOf A).card := by
        let f : Fin n → Fin (2 * n) := fun i => σ ⟨n + i.val, by omega⟩
        have hfinj : Function.Injective f := fun i₁ i₂ h => Fin.ext (by
          have := congrArg Fin.val (σ.injective h); dsimp at this; omega)
        calc (Finset.univ.filter fun i : Fin n => y (σ ⟨n + i.val, by omega⟩) ∈ R).card
            = ((Finset.univ.filter fun i : Fin n => y (σ ⟨n + i.val, by omega⟩) ∈ R).image f).card :=
              (Finset.card_image_of_injective _ hfinj).symm
          _ ≤ (idxOf A).card := by
              apply Finset.card_le_card
              intro k hk
              simp only [Finset.mem_image, Finset.mem_filter, Finset.mem_univ, true_and, f] at hk
              rcases hk with ⟨i, hiR, rfl⟩
              simp only [idxOf, A, Ey, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_image]
              exact ⟨⟨σ ⟨n + i.val, by omega⟩, rfl⟩, hiR⟩
      have hA_heavy : A ∈ heavyTraces := by
        refine Finset.mem_filter.mpr ⟨hA_tr, hhits.trans (Nat.cast_le.mpr hidx_card)⟩
      refine Finset.mem_biUnion.mpr ⟨A, hA_heavy, Finset.mem_filter.mpr ⟨Finset.mem_univ _, fun i hi => ?_⟩⟩
      simp only [idxOf, A, Finset.mem_filter, Finset.mem_univ, true_and] at hi
      exact hmiss i hi.2
    have hcard_union : ((Finset.univ.filter fun σ : Equiv.Perm (Fin (2 * n)) => E2 (y ∘ σ)).card : ℝ) ≤
        ∑ A ∈ heavyTraces, ((Finset.univ.filter fun σ : Equiv.Perm (Fin (2 * n)) =>
          ∀ i : Fin n, σ ⟨i.val, by omega⟩ ∉ idxOf A).card : ℝ) := by
      exact_mod_cast (Finset.card_le_card hsub_union).trans Finset.card_biUnion_le
    calc ((Finset.univ.filter fun σ : Equiv.Perm (Fin (2 * n)) => E2 (y ∘ σ)).card : ℝ) /
          (Nat.factorial (2 * n) : ℝ)
        ≤ (∑ A ∈ heavyTraces, ((Finset.univ.filter fun σ : Equiv.Perm (Fin (2 * n)) =>
            ∀ i : Fin n, σ ⟨i.val, by omega⟩ ∉ idxOf A).card : ℝ)) /
            (Nat.factorial (2 * n) : ℝ) := div_le_div_of_nonneg_right hcard_union hfact_pos.le
      _ = ∑ A ∈ heavyTraces, ((Finset.univ.filter fun σ : Equiv.Perm (Fin (2 * n)) =>
            ∀ i : Fin n, σ ⟨i.val, by omega⟩ ∉ idxOf A).card : ℝ) /
            (Nat.factorial (2 * n) : ℝ) := Finset.sum_div _ _ _
      _ ≤ ∑ _A ∈ heavyTraces, Real.exp (-(ε * (n : ℝ) / 4)) := by
          refine Finset.sum_le_sum fun A hA => ?_
          exact perm_avoid_fraction_le_exp hn (idxOf A) hε.le (Finset.mem_filter.mp hA).2
      _ = (heavyTraces.card : ℝ) * Real.exp (-(ε * (n : ℝ) / 4)) := by
          rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ (M_grow : ℝ) * Real.exp (-(ε * (n : ℝ) / 4)) := by
          have hhc : heavyTraces.card ≤ M_grow := (Finset.card_filter_le _ _).trans hTr_card
          exact mul_le_mul_of_nonneg_right (by exact_mod_cast hhc) (Real.exp_pos _).le
  -- Step 4: Combine symmetrization + exchangeability + pointwise bound
  have hE2_le : (∑ y : Fin (2 * n) → Ω, if E2 y then prodPMF P y else 0) ≤
      (M_grow : ℝ) * Real.exp (-(ε * (n : ℝ) / 4)) := by
    rw [hexch]
    calc ∑ y : Fin (2 * n) → Ω, prodPMF P y *
            (((Finset.univ.filter fun σ : Equiv.Perm (Fin (2 * n)) => E2 (y ∘ σ)).card : ℝ) /
              (Nat.factorial (2 * n) : ℝ))
        ≤ ∑ y : Fin (2 * n) → Ω, prodPMF P y * ((M_grow : ℝ) * Real.exp (-(ε * (n : ℝ) / 4))) :=
          Finset.sum_le_sum fun y _ => mul_le_mul_of_nonneg_left (hpointwise y) (prodPMF_nonneg hP y)
      _ = (∑ y : Fin (2 * n) → Ω, prodPMF P y) * ((M_grow : ℝ) * Real.exp (-(ε * (n : ℝ) / 4))) := by
          rw [← Finset.sum_mul]
      _ = (M_grow : ℝ) * Real.exp (-(ε * (n : ℝ) / 4)) := by
          rw [prodPMF_sum_one hsum, one_mul]
  linarith

/-! ### 5. Stage 1: Conjunction Pair Ranges, `1/32` Mass Bound, and Heavy Pairs -/

/-- **Conjunction pair-range mass lower bound**:
If `N ≥ 8`, `N < 2 h`, and `h ≤ 2 c`, then `c (c - 1) / (N (N - 1)) ≥ 1 / 32`. -/
theorem conjunction_pair_mass_ge {N h c : ℕ} (hN : 8 ≤ N) (hh : N < 2 * h) (hc : h ≤ 2 * c) :
    (1 : ℝ) / 32 ≤ ((c * (c - 1) : ℕ) : ℝ) / ((N * (N - 1) : ℕ) : ℝ) := by
  have hc4 : N < 4 * c := by omega
  have hc_ge2 : 2 ≤ c := by omega
  have hden_pos : (0 : ℝ) < ((N * (N - 1) : ℕ) : ℝ) := by
    have : 0 < N * (N - 1) := Nat.mul_pos (by omega) (by omega)
    exact_mod_cast this
  rw [div_le_div_iff₀ (by norm_num : (0 : ℝ) < 32) hden_pos]
  have hnat : N * (N - 1) ≤ 32 * (c * (c - 1)) := by
    have h1 : N ≤ 4 * c := by omega
    have h2 : N ≤ 8 * (c - 1) := by omega
    have h3 : N * (N - 1) ≤ N * N := Nat.mul_le_mul_left N (Nat.sub_le N 1)
    have h4 : N * N ≤ (4 * c) * (8 * (c - 1)) := Nat.mul_le_mul h1 h2
    calc N * (N - 1) ≤ (4 * c) * (8 * (c - 1)) := h3.trans h4
      _ = 32 * (c * (c - 1)) := by ring
  have hcast : ((N * (N - 1) : ℕ) : ℝ) ≤ 32 * ((c * (c - 1) : ℕ) : ℝ) := by
    exact_mod_cast hnat
  linarith

/-- Conjunction range on `Ω_pair` induced by a conjunction `L : List (X × Bool)` of length `≤ D`
with respect to a transversal `W` indexed by `rep : Y → Concept X` (`W_S ⊆ Y`). -/
def conjPairRange {X Y : Type*} (W_Y : Finset Y) (rep : Y → Concept X) (L : List (X × Bool)) :
    Set { p : Y × Y // p.1 ≠ p.2 } :=
  { p | p.1.1 ∈ W_Y ∧ p.1.2 ∈ W_Y ∧
        (∀ qb ∈ L, rep p.1.1 qb.1 = qb.2) ∧ (∀ qb ∈ L, rep p.1.2 qb.1 = qb.2) }

/-- Family of all conjunction pair ranges of length `≤ D`. -/
def conjPairFamily {X Y : Type*} (W_Y : Finset Y) (rep : Y → Concept X) (D : ℕ) :
    Set (Set { p : Y × Y // p.1 ≠ p.2 }) :=
  { R | ∃ L : List (X × Bool), L.length ≤ D ∧ R = conjPairRange W_Y rep L }

/-- **Conjunction pair-range growth bound**:
If the dual evaluation family of `C` has VC dimension `≤ d` and `rep y ∈ C` for all `y ∈ W_Y`,
then on any `E ⊆ Ω_pair` of size `l`, the trace family of `conjPairFamily W_Y rep D` has
cardinality at most `(1 + 2 * (2 * l + 1)^d)^D`. -/
theorem conjunction_pair_range_growth {X Y : Type*} [DecidableEq Y]
    {C : Set (Concept X)} {d : ℕ} (hdvc : HasVCDimLe (DualFamily C) d)
    (W_Y : Finset Y) (rep : Y → Concept X) (hrepC : ∀ y ∈ W_Y, rep y ∈ C)
    (D : ℕ) (E : Finset { p : Y × Y // p.1 ≠ p.2 }) :
    (TraceSet (conjPairFamily W_Y rep D) E).card ≤ (1 + 2 * (2 * E.card + 1) ^ d) ^ D := by
  classical
  -- Participating representatives in W_Y from endpoints of E: at most 2 * |E| concepts
  let H_Y : Finset Y := (E.image fun p => p.1.1) ∪ (E.image fun p => p.1.2)
  let H : Finset (Concept X) := (H_Y.filter fun y => y ∈ W_Y).image rep
  have hH_card : H.card ≤ 2 * E.card := by
    calc H.card ≤ (H_Y.filter fun y => y ∈ W_Y).card := Finset.card_image_le
      _ ≤ H_Y.card := Finset.card_filter_le _ _
      _ ≤ (E.image fun p => p.1.1).card + (E.image fun p => p.1.2).card := Finset.card_union_le _ _
      _ ≤ E.card + E.card := Nat.add_le_add Finset.card_image_le Finset.card_image_le
      _ = 2 * E.card := by ring
  -- Single-literal subsets of E: either `E` itself (padding) or `{p ∈ E | p.1, p.2 ∈ W_Y ∧ rep p.1 x = b ∧ rep p.2 x = b}`
  let litTrace (x : X) (b : Bool) : Finset { p : Y × Y // p.1 ≠ p.2 } :=
    E.filter fun p => p.1.1 ∈ W_Y ∧ p.1.2 ∈ W_Y ∧ rep p.1.1 x = b ∧ rep p.1.2 x = b
  let S_true : Finset (Finset { p : Y × Y // p.1 ≠ p.2 }) :=
    (TraceSet (DualFamily C) H).image fun A =>
      E.filter fun p => p.1.1 ∈ W_Y ∧ p.1.2 ∈ W_Y ∧ rep p.1.1 ∈ A ∧ rep p.1.2 ∈ A
  let S_false : Finset (Finset { p : Y × Y // p.1 ≠ p.2 }) :=
    (TraceSet (DualFamily C) H).image fun A =>
      E.filter fun p => p.1.1 ∈ W_Y ∧ p.1.2 ∈ W_Y ∧ rep p.1.1 ∉ A ∧ rep p.1.2 ∉ A
  let base : Finset { p : Y × Y // p.1 ≠ p.2 } :=
    E.filter fun p => p.1.1 ∈ W_Y ∧ p.1.2 ∈ W_Y
  let Atoms : Finset (Finset { p : Y × Y // p.1 ≠ p.2 }) := insert base (S_true ∪ S_false)
  have hAtoms_card : Atoms.card ≤ 1 + 2 * (2 * E.card + 1) ^ d := by
    have htr : (TraceSet (DualFamily C) H).card ≤ (2 * E.card + 1) ^ d :=
      (traceSet_card_le_succ_pow hdvc H).trans (Nat.pow_le_pow_left (by omega) d)
    have h1 : Atoms.card ≤ (S_true ∪ S_false).card + 1 := Finset.card_insert_le base (S_true ∪ S_false)
    have h2 : (S_true ∪ S_false).card ≤ S_true.card + S_false.card := Finset.card_union_le S_true S_false
    have h3 : S_true.card ≤ (TraceSet (DualFamily C) H).card := Finset.card_image_le
    have h4 : S_false.card ≤ (TraceSet (DualFamily C) H).card := Finset.card_image_le
    omega
  -- Every literal trace belongs to S_true ∪ S_false ⊆ Atoms!
  have hlit_mem : ∀ (x : X) (b : Bool), litTrace x b ∈ Atoms := by
    intro x b
    let A : Finset (Concept X) := H.filter fun h => h x = true
    have hA_tr : A ∈ TraceSet (DualFamily C) H := by
      refine Finset.mem_filter.mpr ⟨Finset.mem_powerset.mpr (Finset.filter_subset _ _),
        ⟨{h : Concept X | h ∈ C ∧ h x = true}, ⟨x, rfl⟩, fun h hh => ?_⟩⟩
      have hhC : h ∈ C := by
        simp only [H, Finset.mem_image, Finset.mem_filter] at hh
        rcases hh with ⟨y, ⟨_, hyW⟩, rfl⟩
        exact hrepC y hyW
      simp [A, hh, hhC]
    have hrep_in_H : ∀ p ∈ E, p.1.1 ∈ W_Y → rep p.1.1 ∈ H := fun p hp h1 =>
      Finset.mem_image.mpr ⟨p.1.1, Finset.mem_filter.mpr
        ⟨Finset.mem_union_left _ (Finset.mem_image.mpr ⟨p, hp, rfl⟩), h1⟩, rfl⟩
    have hrep2_in_H : ∀ p ∈ E, p.1.2 ∈ W_Y → rep p.1.2 ∈ H := fun p hp h2 =>
      Finset.mem_image.mpr ⟨p.1.2, Finset.mem_filter.mpr
        ⟨Finset.mem_union_right _ (Finset.mem_image.mpr ⟨p, hp, rfl⟩), h2⟩, rfl⟩
    cases b
    · apply Finset.mem_insert_of_mem; apply Finset.mem_union_right
      refine Finset.mem_image.mpr ⟨A, hA_tr, ?_⟩
      ext p; simp only [litTrace, Finset.mem_filter, A]
      constructor
      · rintro ⟨hp, h1, h2, hn1, hn2⟩
        refine ⟨hp, h1, h2, ?_, ?_⟩
        · cases hc : rep p.1.1 x <;> simp_all [hrep_in_H p hp h1]
        · cases hc : rep p.1.2 x <;> simp_all [hrep2_in_H p hp h2]
      · rintro ⟨hp, h1, h2, hr1, hr2⟩
        refine ⟨hp, h1, h2, ?_, ?_⟩
        · intro hm; exact Bool.false_ne_true (hr1.symm.trans hm.2)
        · intro hm; exact Bool.false_ne_true (hr2.symm.trans hm.2)
    · apply Finset.mem_insert_of_mem; apply Finset.mem_union_left
      refine Finset.mem_image.mpr ⟨A, hA_tr, ?_⟩
      ext p; simp only [litTrace, Finset.mem_filter, A]
      constructor
      · rintro ⟨hp, h1, h2, ha1, ha2⟩
        exact ⟨hp, h1, h2, ha1.2, ha2.2⟩
      · rintro ⟨hp, h1, h2, hr1, hr2⟩
        exact ⟨hp, h1, h2, ⟨hrep_in_H p hp h1, hr1⟩, ⟨hrep2_in_H p hp h2, hr2⟩⟩
  -- Every trace of conjPairFamily W_Y rep D is the intersection of D atoms from Atoms!
  let interMap : (Fin D → Atoms) → Finset { p : Y × Y // p.1 ≠ p.2 } := fun tup =>
    base.filter fun p => ∀ i : Fin D, p ∈ (tup i).1
  have hsub_img : TraceSet (conjPairFamily W_Y rep D) E ⊆ Finset.univ.image interMap := by
    intro A hA
    rcases (Finset.mem_filter.mp hA).2 with ⟨_, ⟨L, hLlen, rfl⟩, hAiff⟩
    have hAsub : A ⊆ E := Finset.mem_powerset.mp (Finset.mem_filter.mp hA).1
    let tup : Fin D → Atoms := fun i =>
      if h : i.val < L.length then ⟨litTrace L[i.val].1 L[i.val].2, hlit_mem _ _⟩
      else ⟨base, Finset.mem_insert_self _ _⟩
    refine Finset.mem_image.mpr ⟨tup, Finset.mem_univ _, ?_⟩
    ext p
    simp only [interMap, Finset.mem_filter, base]
    constructor
    · rintro ⟨⟨hpE, h1, h2⟩, hall⟩
      rw [← hAiff p hpE]
      refine ⟨h1, h2, ?_, ?_⟩
      · intro qb hqb
        obtain ⟨idx, hidx, rfl⟩ := List.getElem_of_mem hqb
        have hi := hall ⟨idx, lt_of_lt_of_le hidx hLlen⟩
        simp only [tup, hidx, ↓reduceDIte, litTrace, Finset.mem_filter] at hi
        exact hi.2.2.2.1
      · intro qb hqb
        obtain ⟨idx, hidx, rfl⟩ := List.getElem_of_mem hqb
        have hi := hall ⟨idx, lt_of_lt_of_le hidx hLlen⟩
        simp only [tup, hidx, ↓reduceDIte, litTrace, Finset.mem_filter] at hi
        exact hi.2.2.2.2
    · intro hpA
      have hpE := hAsub hpA
      have hpC := (hAiff p hpE).mpr hpA
      refine ⟨⟨hpE, hpC.1, hpC.2.1⟩, fun i => ?_⟩
      simp only [tup]
      split_ifs with hi
      · simp only [litTrace, Finset.mem_filter]
        exact ⟨hpE, hpC.1, hpC.2.1, hpC.2.2.1 _ (List.getElem_mem hi), hpC.2.2.2 _ (List.getElem_mem hi)⟩
      · simp only [base, Finset.mem_filter]
        exact ⟨hpE, hpC.1, hpC.2.1⟩
  calc (TraceSet (conjPairFamily W_Y rep D) E).card
      ≤ (Finset.univ.image interMap).card := Finset.card_le_card hsub_img
    _ ≤ (Finset.univ : Finset (Fin D → Atoms)).card := Finset.card_image_le
    _ = Atoms.card ^ D := by simp
    _ ≤ (1 + 2 * (2 * E.card + 1) ^ d) ^ D := Nat.pow_le_pow_left hAtoms_card D

open Classical in
/-- **Heavy-pair existence in every transversal `W` of size `h ≥ 2`**:
Given a finite distribution `P_T` over trees `T_ω` (depth `≤ D`), let `Q` be the query distribution
obtained by drawing `ω ~ P_T` and then picking a uniformly random entry from the `2^D`-padded query
list of `T_ω`. If with `P_T`-probability `≥ 1/2`, every leaf of `T_ω` contains strictly fewer than
`h / 2` members of `W`, then there exist `f ≠ g ∈ W` whose disagreement set `{x | f x ≠ g x}` has
`Q`-mass at least `1 / (4 * 2^D)`. -/
theorem heavy_pair_exists_in_transversal {X ι : Type*} [Fintype ι] [DecidableEq X]
    (D : ℕ) (x₀ : X) (P_T : ι → ℝ) (hPT_nn : ∀ ω, 0 ≤ P_T ω)
    (T : ι → MQTree X) (hTdep : ∀ ω, (T ω).depth ≤ D)
    (W : Finset (Concept X)) (hW : 2 ≤ W.card)
    (hgood : (1 / 2 : ℝ) ≤ ∑ ω : ι,
      if (∀ f ∈ W, 2 * (W.filter fun g => (T ω).leafPath g = (T ω).leafPath f).card < W.card)
      then P_T ω else 0) :
    let padQ : ι → Fin (2 ^ D) → X := fun ω j =>
      ((T ω).queries).getD j.val x₀
    let Q_mass : Set X → ℝ := fun A =>
      ∑ ω : ι, P_T ω * ((∑ j : Fin (2 ^ D), if padQ ω j ∈ A then (1 : ℝ) else 0) / (2 ^ D : ℝ))
    ∃ f ∈ W, ∃ g ∈ W, f ≠ g ∧
      (1 : ℝ) / (4 * (2 ^ D : ℝ)) ≤ Q_mass {x : X | f x ≠ g x} := by
  dsimp only
  let padQ : ι → Fin (2 ^ D) → X := fun ω j => ((T ω).queries).getD j.val x₀
  let Q_mass : Set X → ℝ := fun A =>
    ∑ ω : ι, P_T ω * ((∑ j : Fin (2 ^ D), if padQ ω j ∈ A then (1 : ℝ) else 0) / (2 ^ D : ℝ))
  let h := W.card
  let pairs : Finset (Concept X × Concept X) := (W ×ˢ W).filter fun fg => fg.1 ≠ fg.2
  have hpairs_card : pairs.card = h * (h - 1) := by
    have hdiag : (W ×ˢ W).filter (fun fg => fg.1 = fg.2) = W.image fun f => (f, f) := by
      ext ⟨f, g⟩; simp only [Finset.mem_filter, Finset.mem_product, Finset.mem_image]
      exact ⟨fun ⟨⟨hf, _⟩, heq⟩ => ⟨f, hf, Prod.ext rfl heq⟩,
        fun ⟨a, ha, h⟩ => by rcases Prod.mk.inj h with ⟨rfl, rfl⟩; exact ⟨⟨ha, ha⟩, rfl⟩⟩
    have hsplit := Finset.filter_card_add_filter_neg_card_eq_card (s := W ×ˢ W) (p := fun fg => fg.1 = fg.2)
    rw [hdiag, Finset.card_image_of_injective _ (fun _ _ h => congrArg Prod.fst h), Finset.card_product] at hsplit
    have hmul : h * h = h + h * (h - 1) := by
      cases h with
      | zero => omega
      | succ k => simp [Nat.mul_succ]; ring
    have hpeq : ((W ×ˢ W).filter fun fg => ¬ fg.1 = fg.2).card = pairs.card := rfl
    rw [hpeq] at hsplit
    change h + pairs.card = h * h at hsplit
    omega
  have hpairs_pos : 0 < pairs.card := by rw [hpairs_card]; exact Nat.mul_pos (by omega) (by omega)
  have h2D_pos : (0 : ℝ) < (2 ^ D : ℝ) := by positivity
  have hhit_ge : ∀ ω (f g : Concept X), (T ω).leafPath f ≠ (T ω).leafPath g →
      (1 : ℝ) / (2 ^ D : ℝ) ≤
        (∑ j : Fin (2 ^ D), if padQ ω j ∈ {x : X | f x ≠ g x} then (1 : ℝ) else 0) / (2 ^ D : ℝ) := by
    intro ω f g hne
    obtain ⟨q, hq, hfg⟩ := (T ω).exists_disagreement_of_leafPath_ne hne
    obtain ⟨idx, hidx, rfl⟩ := List.getElem_of_mem hq
    have hidxD : idx < 2 ^ D :=
      lt_of_lt_of_le (lt_trans hidx (T ω).queries_length_lt_two_pow)
        (Nat.pow_le_pow_right (by norm_num) (hTdep ω))
    let j₀ : Fin (2 ^ D) := ⟨idx, hidxD⟩
    have hj₀_val : padQ ω j₀ = (T ω).queries[idx] := by
      simp [padQ, j₀, List.getD_eq_getElem?_getD, hidx]
    apply div_le_div_of_nonneg_right _ h2D_pos.le
    calc (1 : ℝ) = if padQ ω j₀ ∈ {x : X | f x ≠ g x} then (1 : ℝ) else 0 := by
          simp [hj₀_val, hfg]
      _ ≤ ∑ j : Fin (2 ^ D), if padQ ω j ∈ {x : X | f x ≠ g x} then (1 : ℝ) else 0 :=
          Finset.single_le_sum
            (f := fun j : Fin (2 ^ D) => if padQ ω j ∈ {x : X | f x ≠ g x} then (1 : ℝ) else 0)
            (fun j _ => by dsimp only; split_ifs <;> norm_num) (Finset.mem_univ j₀)
  have hgood_pairs : ∀ ω,
      (∀ f ∈ W, 2 * (W.filter fun g => (T ω).leafPath g = (T ω).leafPath f).card < h) →
      (pairs.card : ℝ) / 2 ≤
        ((pairs.filter fun fg => (T ω).leafPath fg.1 ≠ (T ω).leafPath fg.2).card : ℝ) := by
    intro ω hω
    have hsame_le : (pairs.filter fun fg => (T ω).leafPath fg.1 = (T ω).leafPath fg.2).card * 2 ≤
        pairs.card := by
      have hbi : (pairs.filter fun fg => (T ω).leafPath fg.1 = (T ω).leafPath fg.2) =
          W.biUnion fun f => ((W.filter fun g => (T ω).leafPath g = (T ω).leafPath f).erase f).image fun g => (f, g) := by
        ext ⟨f, g⟩
        simp only [pairs, Finset.mem_filter, Finset.mem_product, Finset.mem_biUnion, Finset.mem_image,
          Finset.mem_erase]
        constructor
        · rintro ⟨⟨⟨hf, hg⟩, hne⟩, hpath⟩
          exact ⟨f, hf, g, ⟨Ne.symm hne, hg, hpath.symm⟩, rfl⟩
        · rintro ⟨f', hf', g', ⟨hne, hg', hpath⟩, hpair⟩
          rcases Prod.mk.inj hpair with ⟨rfl, rfl⟩
          exact ⟨⟨⟨hf', hg'⟩, Ne.symm hne⟩, hpath.symm⟩
      have hcard_same : (pairs.filter fun fg => (T ω).leafPath fg.1 = (T ω).leafPath fg.2).card ≤
          ∑ f ∈ W, ((W.filter fun g => (T ω).leafPath g = (T ω).leafPath f).card - 1) := by
        rw [hbi]
        refine Finset.card_biUnion_le.trans (Finset.sum_le_sum fun f hf => ?_)
        have hf_in : f ∈ W.filter fun g => (T ω).leafPath g = (T ω).leafPath f :=
          Finset.mem_filter.mpr ⟨hf, rfl⟩
        calc (((W.filter fun g => (T ω).leafPath g = (T ω).leafPath f).erase f).image fun g => (f, g)).card
            ≤ ((W.filter fun g => (T ω).leafPath g = (T ω).leafPath f).erase f).card := Finset.card_image_le
          _ = (W.filter fun g => (T ω).leafPath g = (T ω).leafPath f).card - 1 := Finset.card_erase_of_mem hf_in
      have hsum_le : (∑ f ∈ W, ((W.filter fun g => (T ω).leafPath g = (T ω).leafPath f).card - 1)) * 2 ≤
          h * (h - 1) := by
        calc (∑ f ∈ W, ((W.filter fun g => (T ω).leafPath g = (T ω).leafPath f).card - 1)) * 2
            = ∑ f ∈ W, 2 * ((W.filter fun g => (T ω).leafPath g = (T ω).leafPath f).card - 1) := by
              rw [Finset.sum_mul]; exact Finset.sum_congr rfl fun _ _ => by ring
          _ ≤ ∑ _f ∈ W, (h - 1) := by
              refine Finset.sum_le_sum fun f hf => ?_
              have := hω f hf
              have hf1 : 1 ≤ (W.filter fun g => (T ω).leafPath g = (T ω).leafPath f).card :=
                Finset.card_pos.mpr ⟨f, Finset.mem_filter.mpr ⟨hf, rfl⟩⟩
              omega
          _ = h * (h - 1) := by simp [h]
      rw [hpairs_card]
      exact (Nat.mul_le_mul_right 2 hcard_same).trans hsum_le
    have hsplit := Finset.filter_card_add_filter_neg_card_eq_card (s := pairs)
      (p := fun fg => (T ω).leafPath fg.1 = (T ω).leafPath fg.2)
    have hneq_eq : (pairs.filter fun fg => ¬ (T ω).leafPath fg.1 = (T ω).leafPath fg.2).card =
        (pairs.filter fun fg => (T ω).leafPath fg.1 ≠ (T ω).leafPath fg.2).card := rfl
    rw [hneq_eq] at hsplit
    have hnat : pairs.card ≤ 2 * (pairs.filter fun fg => (T ω).leafPath fg.1 ≠ (T ω).leafPath fg.2).card := by
      omega
    have hcast : (pairs.card : ℝ) ≤
        2 * ((pairs.filter fun fg => (T ω).leafPath fg.1 ≠ (T ω).leafPath fg.2).card : ℝ) := by
      exact_mod_cast hnat
    linarith
  have hsum_ge : (pairs.card : ℝ) * ((1 : ℝ) / (4 * (2 ^ D : ℝ))) ≤
      ∑ fg ∈ pairs, Q_mass {x : X | fg.1 x ≠ fg.2 x} := by
    have hswap : (∑ fg ∈ pairs, Q_mass {x : X | fg.1 x ≠ fg.2 x}) =
        ∑ ω : ι, P_T ω * ∑ fg ∈ pairs,
          ((∑ j : Fin (2 ^ D), if padQ ω j ∈ {x : X | fg.1 x ≠ fg.2 x} then (1 : ℝ) else 0) / (2 ^ D : ℝ)) := by
      simp only [Q_mass]
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun ω _ => ?_
      rw [← Finset.mul_sum]
      congr 1
      refine Finset.sum_congr rfl fun fg _ => ?_
      congr 1
      refine Finset.sum_congr rfl fun j _ => ?_
      by_cases hmem : padQ ω j ∈ {x : X | fg.1 x ≠ fg.2 x} <;> simp [hmem]
    rw [hswap]
    let isGood : ι → Prop := fun ω =>
      ∀ f ∈ W, 2 * (W.filter fun g => (T ω).leafPath g = (T ω).leafPath f).card < h
    calc (pairs.card : ℝ) * ((1 : ℝ) / (4 * (2 ^ D : ℝ)))
        = (1 / 2 : ℝ) * ((pairs.card : ℝ) / 2 * ((1 : ℝ) / (2 ^ D : ℝ))) := by ring
      _ ≤ (∑ ω : ι, if isGood ω then P_T ω else 0) *
            ((pairs.card : ℝ) / 2 * ((1 : ℝ) / (2 ^ D : ℝ))) :=
          mul_le_mul_of_nonneg_right hgood (by positivity)
      _ = ∑ ω : ι, (if isGood ω then P_T ω else 0) *
            ((pairs.card : ℝ) / 2 * ((1 : ℝ) / (2 ^ D : ℝ))) := Finset.sum_mul _ _ _
      _ ≤ ∑ ω : ι, P_T ω * ∑ fg ∈ pairs,
            ((∑ j : Fin (2 ^ D), if padQ ω j ∈ {x : X | fg.1 x ≠ fg.2 x} then (1 : ℝ) else 0) / (2 ^ D : ℝ)) := by
          refine Finset.sum_le_sum fun ω _ => ?_
          by_cases hg : isGood ω
          · simp only [hg, ↓reduceIte]
            apply mul_le_mul_of_nonneg_left _ (hPT_nn ω)
            calc (pairs.card : ℝ) / 2 * ((1 : ℝ) / (2 ^ D : ℝ))
                ≤ ((pairs.filter fun fg => (T ω).leafPath fg.1 ≠ (T ω).leafPath fg.2).card : ℝ) *
                    ((1 : ℝ) / (2 ^ D : ℝ)) :=
                  mul_le_mul_of_nonneg_right (hgood_pairs ω hg) (by positivity)
              _ = ∑ _fg ∈ pairs.filter (fun fg => (T ω).leafPath fg.1 ≠ (T ω).leafPath fg.2),
                    ((1 : ℝ) / (2 ^ D : ℝ)) := by rw [Finset.sum_const, nsmul_eq_mul]
              _ ≤ ∑ fg ∈ pairs.filter (fun fg => (T ω).leafPath fg.1 ≠ (T ω).leafPath fg.2),
                    ((∑ j : Fin (2 ^ D), if padQ ω j ∈ {x : X | fg.1 x ≠ fg.2 x} then (1 : ℝ) else 0) / (2 ^ D : ℝ)) :=
                  Finset.sum_le_sum fun fg hfg => hhit_ge ω fg.1 fg.2 (Finset.mem_filter.mp hfg).2
              _ ≤ ∑ fg ∈ pairs,
                    ((∑ j : Fin (2 ^ D), if padQ ω j ∈ {x : X | fg.1 x ≠ fg.2 x} then (1 : ℝ) else 0) / (2 ^ D : ℝ)) :=
                  Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _) fun _ _ _ => by positivity
          · simp only [hg, ↓reduceIte, zero_mul]
            exact mul_nonneg (hPT_nn ω) (Finset.sum_nonneg fun _ _ => by positivity)
  -- Pigeonhole: some pair (f, g) ∈ pairs achieves at least the average!
  by_contra hnone
  push_neg at hnone
  have hstrict : (∑ fg ∈ pairs, Q_mass {x : X | fg.1 x ≠ fg.2 x}) <
      ∑ _fg ∈ pairs, ((1 : ℝ) / (4 * (2 ^ D : ℝ))) := by
    refine Finset.sum_lt_sum_of_nonempty (Finset.card_pos.mp hpairs_pos) fun fg hfg => ?_
    have hfg' := Finset.mem_filter.mp hfg
    have hfW := (Finset.mem_product.mp hfg'.1).1
    have hgW := (Finset.mem_product.mp hfg'.1).2
    exact hnone fg.1 hfW fg.2 hgW hfg'.2
  rw [Finset.sum_const, nsmul_eq_mul] at hstrict
  linarith

/-! ### 6. Small-`N` Base Case, Asymptotic Thresholds, and Full Assembly (Theorem 1) -/

/-- **Small-`N` deterministic separation**:
Whenever `|(labelings V S)| ≤ 7` and `X` is nonempty, there exists a fixed batch `U : Fin t → X`
(for any `t ≥ 8`) that halves `V_S` under every answer pattern (in fact isolates every labeling). -/
theorem small_N_halving {X : Type*} (x₀ : X) {m t : ℕ} (ht : 8 ≤ t)
    (V : Set (Concept X)) (S : Fin m → X) (hN : (labelings V S).card < 8) :
    ∃ U : Fin t → X, HalvesOn U V S := by
  classical
  have hsep_strong : ∀ (c : ℕ) (Y : Finset (Fin m → Bool)), Y.card = c →
      ∃ L : List (Fin m), L.length ≤ c - 1 ∧
        ∀ y₁ ∈ Y, ∀ y₂ ∈ Y, (∀ i ∈ L, y₁ i = y₂ i) → y₁ = y₂ := by
    intro c
    induction c using Nat.strong_induction_on with
    | h c ih =>
      intro Y hYc
      by_cases h1 : c ≤ 1
      · refine ⟨[], by simp, fun y₁ hy1 y₂ hy2 _ => Finset.card_le_one.mp (hYc.le.trans h1) y₁ hy1 y₂ hy2⟩
      · push_neg at h1
        obtain ⟨a, ha, b, hb, hab⟩ := Finset.one_lt_card.mp (hYc ▸ h1)
        have hdiff : ∃ i₀ : Fin m, a i₀ ≠ b i₀ := by
          by_contra hall; push_neg at hall; exact hab (funext hall)
        obtain ⟨i₀, hi₀⟩ := hdiff
        let Y0 := Y.filter fun y => y i₀ = false
        let Y1 := Y.filter fun y => y i₀ = true
        have hsplit : Y0.card + Y1.card = c := by
          rw [← hYc]
          have h := Finset.filter_card_add_filter_neg_card_eq_card (s := Y) (p := fun y => y i₀ = false)
          have h1eq : (Y.filter fun y => ¬ y i₀ = false) = Y1 := by
            ext y; simp only [Y1, Finset.mem_filter, Bool.not_eq_false]
          rwa [h1eq] at h
        have hY0_pos : 0 < Y0.card := by
          cases ha0 : a i₀
          · exact Finset.card_pos.mpr ⟨a, Finset.mem_filter.mpr ⟨ha, ha0⟩⟩
          · have hb0 : b i₀ = false := by cases hb' : b i₀ <;> simp_all
            exact Finset.card_pos.mpr ⟨b, Finset.mem_filter.mpr ⟨hb, hb0⟩⟩
        have hY1_pos : 0 < Y1.card := by
          cases ha0 : a i₀
          · have hb1 : b i₀ = true := by cases hb' : b i₀ <;> simp_all
            exact Finset.card_pos.mpr ⟨b, Finset.mem_filter.mpr ⟨hb, hb1⟩⟩
          · exact Finset.card_pos.mpr ⟨a, Finset.mem_filter.mpr ⟨ha, ha0⟩⟩
        obtain ⟨L0, hL0_len, hL0_sep⟩ := ih Y0.card (by omega) Y0 rfl
        obtain ⟨L1, hL1_len, hL1_sep⟩ := ih Y1.card (by omega) Y1 rfl
        refine ⟨i₀ :: (L0 ++ L1), ?_, ?_⟩
        · simp only [List.length_cons, List.length_append]; omega
        · intro y₁ hy1 y₂ hy2 hall
          have hi0_eq : y₁ i₀ = y₂ i₀ := hall i₀ (List.mem_cons_self)
          cases hy1_0 : y₁ i₀
          · have hy2_0 : y₂ i₀ = false := hi0_eq.symm.trans hy1_0
            exact hL0_sep y₁ (Finset.mem_filter.mpr ⟨hy1, hy1_0⟩)
              y₂ (Finset.mem_filter.mpr ⟨hy2, hy2_0⟩)
              (fun i hi => hall i (List.mem_cons_of_mem _ (List.mem_append_left _ hi)))
          · have hy2_1 : y₂ i₀ = true := hi0_eq.symm.trans hy1_0
            exact hL1_sep y₁ (Finset.mem_filter.mpr ⟨hy1, hy1_0⟩)
              y₂ (Finset.mem_filter.mpr ⟨hy2, hy2_1⟩)
              (fun i hi => hall i (List.mem_cons_of_mem _ (List.mem_append_right _ hi)))
  obtain ⟨L, hLlen, hLsep⟩ := hsep_strong (labelings V S).card (labelings V S) rfl
  let U : Fin t → X := fun j => if hj : j.val < L.length then S L[j.val] else x₀
  refine ⟨U, fun b => ?_⟩
  by_cases h0 : (labelings V S).card = 0
  · have hsub : labelings (slice V U b) S ⊆ labelings V S := by
      intro y hy
      simp only [labelings, Finset.mem_filter, Finset.mem_univ, true_and, slice, Set.mem_setOf_eq] at hy ⊢
      rcases hy with ⟨h, ⟨hV, _⟩, rfl⟩; exact ⟨h, hV, rfl⟩
    have := (Finset.card_le_card hsub).trans h0.le
    omega
  · have hcard_le1 : (labelings (slice V U b) S).card ≤ 1 := by
      apply Finset.card_le_one.mpr
      intro y₁ hy1 y₂ hy2
      simp only [labelings, Finset.mem_filter, Finset.mem_univ, true_and, slice, Set.mem_setOf_eq] at hy1 hy2
      rcases hy1 with ⟨h₁, ⟨h₁V, h₁U⟩, rfl⟩
      rcases hy2 with ⟨h₂, ⟨h₂V, h₂U⟩, rfl⟩
      apply hLsep _ (Finset.mem_filter.mpr ⟨Finset.mem_univ _, ⟨h₁, h₁V, rfl⟩⟩)
        _ (Finset.mem_filter.mpr ⟨Finset.mem_univ _, ⟨h₂, h₂V, rfl⟩⟩)
      intro i hi
      obtain ⟨idx, hidx, rfl⟩ := List.getElem_of_mem hi
      have hidx_t : idx < t := by omega
      have e1 := congrFun h₁U ⟨idx, hidx_t⟩
      have e2 := congrFun h₂U ⟨idx, hidx_t⟩
      simp only [restricts, U, hidx, ↓reduceDIte] at e1 e2
      exact e1.trans e2.symm
    omega

/-- Asymptotic existence of Stage 1 sample size `k = k(a)`:
for every `a ≥ 1` and `d : ℕ`, there exists `k ≥ 256` such that with `D = ⌈a log₂(k + 1)⌉`,
`2 * (1 + 2 * (4 * k + 1)^d)^D * exp(-k / 128) < 1 / 2`. -/
lemma exists_stage1_k (a : ℝ) (ha : 1 ≤ a) (d : ℕ) :
    ∃ k : ℕ, 256 ≤ k ∧
      let D := Nat.ceil (a * Real.logb 2 ((k : ℝ) + 1))
      2 * ((1 + 2 * (4 * k + 1) ^ d : ℕ) : ℝ) ^ D * Real.exp (-((k : ℝ) / 128)) < 1 / 2 := by
  -- Upper bound log((1 + 2*(4k+1)^d)^D) ≤ C * (log(4k+1))^2, while k/128 grows linearly!
  -- Specifically, for x ≥ 2, log(x) ≤ x^(1/3) / (1/3) = 3 x^(1/3), so (log x)^2 ≤ 9 x^(2/3),
  -- or simply use Real.tendsto_pow_mul_exp_neg_atTop_nhds_zero on (x)^(1/2)!
  -- Even simpler: (log(4k+1))^2 ≤ 36 * (4k+1)^(1/2) (since log t ≤ 2 t^(1/4) for all t > 0? Wait:
  -- log u ≤ u - 1 < u for all u > 0! Setting u = t^(1/4) gives (1/4) log t ≤ t^(1/4), so log t ≤ 4 t^(1/4)!
  -- Hence (log t)^2 ≤ 16 t^(1/2) for ALL t > 0! Elementary 3-line proof!)
  have hlog_quarter : ∀ t : ℝ, 0 < t → Real.log t ≤ 4 * t ^ (1 / 4 : ℝ) := by
    intro t ht
    have h1 := Real.log_le_sub_one_of_pos (Real.rpow_pos_of_pos ht (1 / 4 : ℝ))
    rw [Real.log_rpow ht] at h1
    linarith
  -- Choose k large enough so that C * √(4k + 1) + log 4 < k / 128
  let C : ℝ := (a / Real.log 2 + 1) * (2 + (d : ℝ)) * 16
  have hC0 : 0 < C := by
    have : 0 < Real.log 2 := Real.log_pos (by norm_num)
    positivity
  -- For k ≥ (512 * (C + 2)) ^ 2 + 256:
  -- √(4k + 1) ≤ 3 √k, so C * √(4k + 1) + 2 ≤ (3 C + 2) √k < k / 128!
  let M : ℕ := Nat.ceil (512 * (C + 2))
  let k : ℕ := max 256 ((M + 1) ^ 2)
  refine ⟨k, le_max_left _ _, ?_⟩
  dsimp only
  set D : ℕ := Nat.ceil (a * Real.logb 2 ((k : ℝ) + 1))
  have hk256 : (256 : ℝ) ≤ (k : ℝ) := by exact_mod_cast le_max_left 256 ((M + 1) ^ 2)
  have hk_ge_sq : ((M + 1 : ℕ) : ℝ) ^ 2 ≤ (k : ℝ) := by
    exact_mod_cast le_max_right 256 ((M + 1) ^ 2)
  have hsqrt_k : (512 : ℝ) * (C + 2) < Real.sqrt (k : ℝ) := by
    have hM : 512 * (C + 2) ≤ (M : ℝ) := Nat.le_ceil _
    have hlt : (M : ℝ) < ((M + 1 : ℕ) : ℝ) := by push_cast; linarith
    have hle : ((M + 1 : ℕ) : ℝ) ≤ Real.sqrt (k : ℝ) := by
      rw [Real.le_sqrt (by positivity) (by positivity)]; exact hk_ge_sq
    linarith
  have h4k_pos : (0 : ℝ) < 4 * (k : ℝ) + 1 := by linarith
  have hlog_ge1 : 1 ≤ Real.log (4 * (k : ℝ) + 1) := by
    have hexp1_le4 : Real.exp 1 ≤ 4 := by
      have h1 : (1 / 2 : ℝ) ≤ Real.exp (-(1 / 2 : ℝ)) := by
        have := Real.add_one_le_exp (-(1 / 2 : ℝ)); linarith
      have h2 : Real.exp (1 / 2 : ℝ) ≤ 2 := by
        have hmul := mul_le_mul_of_nonneg_right h1 (Real.exp_pos (1 / 2 : ℝ)).le
        rw [← Real.exp_add] at hmul
        have h0 : -(1 / 2 : ℝ) + 1 / 2 = 0 := by ring
        rw [h0, Real.exp_zero] at hmul; linarith
      have hsq := pow_le_pow_left₀ (Real.exp_pos (1 / 2 : ℝ)).le h2 2
      rw [← Real.exp_nat_mul] at hsq
      have h1' : (2 : ℕ) * (1 / 2 : ℝ) = 1 := by norm_num
      rw [h1'] at hsq; linarith
    have h1 : Real.exp 1 ≤ 4 * (k : ℝ) + 1 := by linarith
    calc (1 : ℝ) = Real.log (Real.exp 1) := (Real.log_exp 1).symm
      _ ≤ Real.log (4 * (k : ℝ) + 1) := Real.log_le_log (Real.exp_pos 1) h1
  have hbase_le : ((1 + 2 * (4 * k + 1) ^ d : ℕ) : ℝ) ≤ (4 * (k : ℝ) + 1) ^ (d + 2) := by
    have h4k1 : (3 : ℝ) ≤ 4 * (k : ℝ) + 1 := by linarith
    have hpow : (1 : ℝ) ≤ (4 * (k : ℝ) + 1) ^ d := one_le_pow₀ (by linarith)
    calc ((1 + 2 * (4 * k + 1) ^ d : ℕ) : ℝ)
        = 1 + 2 * (4 * (k : ℝ) + 1) ^ d := by push_cast; ring
      _ ≤ 3 * (4 * (k : ℝ) + 1) ^ d := by linarith
      _ ≤ (4 * (k : ℝ) + 1) ^ 2 * (4 * (k : ℝ) + 1) ^ d :=
          mul_le_mul_of_nonneg_right (by nlinarith) (by positivity)
      _ = (4 * (k : ℝ) + 1) ^ (d + 2) := by rw [← pow_add, add_comm 2 d]
  have hD_le : (D : ℝ) ≤ (a / Real.log 2 + 1) * Real.log (4 * (k : ℝ) + 1) := by
    have hlog2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
    have hlog_le : Real.log ((k : ℝ) + 1) ≤ Real.log (4 * (k : ℝ) + 1) :=
      Real.log_le_log (by linarith) (by linarith)
    have hceil : (D : ℝ) < a * Real.logb 2 ((k : ℝ) + 1) + 1 :=
      Nat.ceil_lt_add_one (mul_nonneg (by linarith) (Real.logb_nonneg (by norm_num) (by linarith)))
    rw [Real.logb] at hceil
    have h1 : a * (Real.log ((k : ℝ) + 1) / Real.log 2) ≤ (a / Real.log 2) * Real.log (4 * (k : ℝ) + 1) := by
      have : a * (Real.log ((k : ℝ) + 1) / Real.log 2) = (a / Real.log 2) * Real.log ((k : ℝ) + 1) := by ring
      rw [this]; exact mul_le_mul_of_nonneg_left hlog_le (by positivity)
    linarith
  have hlog_pow : Real.log (((1 + 2 * (4 * k + 1) ^ d : ℕ) : ℝ) ^ D) ≤ C * Real.sqrt (4 * (k : ℝ) + 1) := by
    rw [Real.log_pow]
    have hlb : Real.log ((1 + 2 * (4 * k + 1) ^ d : ℕ) : ℝ) ≤
        ((d : ℝ) + 2) * Real.log (4 * (k : ℝ) + 1) := by
      have h1 := Real.log_le_log (by positivity) hbase_le
      rw [Real.log_pow] at h1; push_cast at h1 ⊢; linarith
    have hlog_sq : (Real.log (4 * (k : ℝ) + 1)) ^ 2 ≤ 16 * Real.sqrt (4 * (k : ℝ) + 1) := by
      have hq := hlog_quarter (4 * (k : ℝ) + 1) h4k_pos
      have hsq := pow_le_pow_left₀ (by linarith) hq 2
      have hrp : ((4 * (k : ℝ) + 1) ^ (1 / 4 : ℝ)) ^ 2 = Real.sqrt (4 * (k : ℝ) + 1) := by
        rw [Real.sqrt_eq_rpow, ← Real.rpow_natCast, ← Real.rpow_mul h4k_pos.le]; norm_num
      calc (Real.log (4 * (k : ℝ) + 1)) ^ 2
          ≤ (4 * (4 * (k : ℝ) + 1) ^ (1 / 4 : ℝ)) ^ 2 := hsq
        _ = 16 * ((4 * (k : ℝ) + 1) ^ (1 / 4 : ℝ)) ^ 2 := by ring
        _ = 16 * Real.sqrt (4 * (k : ℝ) + 1) := by rw [hrp]
    calc (D : ℝ) * Real.log ((1 + 2 * (4 * k + 1) ^ d : ℕ) : ℝ)
        ≤ ((a / Real.log 2 + 1) * Real.log (4 * (k : ℝ) + 1)) *
            (((d : ℝ) + 2) * Real.log (4 * (k : ℝ) + 1)) :=
          mul_le_mul hD_le hlb (Real.log_nonneg (by norm_cast; omega)) (by positivity)
      _ = (a / Real.log 2 + 1) * (2 + (d : ℝ)) * (Real.log (4 * (k : ℝ) + 1)) ^ 2 := by ring
      _ ≤ (a / Real.log 2 + 1) * (2 + (d : ℝ)) * (16 * Real.sqrt (4 * (k : ℝ) + 1)) :=
          mul_le_mul_of_nonneg_left hlog_sq (by have : 0 < Real.log 2 := Real.log_pos (by norm_num); positivity)
      _ = C * Real.sqrt (4 * (k : ℝ) + 1) := by simp only [C]; ring
  have hsqrt4k : Real.sqrt (4 * (k : ℝ) + 1) ≤ 3 * Real.sqrt (k : ℝ) := by
    rw [Real.sqrt_le_iff]
    refine ⟨by positivity, ?_⟩
    rw [mul_pow, Real.sq_sqrt (by linarith)]; linarith
  have hlog4 : Real.log 4 < 2 := by
    rw [Real.log_lt_iff_lt_exp (by norm_num)]
    have hhalf : (3 / 2 : ℝ) ≤ Real.exp (1 / 2 : ℝ) := by
      have := Real.add_one_le_exp (1 / 2 : ℝ); linarith
    have h4pow := pow_le_pow_left₀ (by norm_num : (0 : ℝ) ≤ 3 / 2) hhalf 4
    rw [← Real.exp_nat_mul] at h4pow
    have h2' : (4 : ℕ) * (1 / 2 : ℝ) = 2 := by norm_num
    rw [h2'] at h4pow
    linarith [show (4 : ℝ) < (3 / 2 : ℝ) ^ 4 by norm_num]
  have hlt_exp : Real.log 4 + Real.log (((1 + 2 * (4 * k + 1) ^ d : ℕ) : ℝ) ^ D) < (k : ℝ) / 128 := by
    have hk_sq : Real.sqrt (k : ℝ) * Real.sqrt (k : ℝ) = (k : ℝ) := Real.mul_self_sqrt (by linarith)
    have hsqrt_ge1 : 1 ≤ Real.sqrt (k : ℝ) := by rw [Real.le_sqrt (by norm_num) (by linarith)]; linarith
    have hstep : 3 * C + 2 < Real.sqrt (k : ℝ) / 128 := by linarith
    have hmul : (3 * C + 2) * Real.sqrt (k : ℝ) < (Real.sqrt (k : ℝ) / 128) * Real.sqrt (k : ℝ) :=
      mul_lt_mul_of_pos_right hstep (by linarith)
    have hrhs : (Real.sqrt (k : ℝ) / 128) * Real.sqrt (k : ℝ) = (k : ℝ) / 128 := by
      rw [div_mul_eq_mul_div, hk_sq]
    have h1 : C * Real.sqrt (4 * (k : ℝ) + 1) ≤ 3 * C * Real.sqrt (k : ℝ) := by nlinarith
    nlinarith
  have hprod_lt : 4 * ((1 + 2 * (4 * k + 1) ^ d : ℕ) : ℝ) ^ D < Real.exp ((k : ℝ) / 128) := by
    have hpos : 0 < ((1 + 2 * (4 * k + 1) ^ d : ℕ) : ℝ) ^ D := by positivity
    rw [← Real.exp_log (mul_pos (by norm_num : (0 : ℝ) < 4) hpos),
      Real.log_mul (by norm_num) hpos.ne']
    exact Real.exp_lt_exp.mpr hlt_exp
  have hneg_exp : Real.exp (-((k : ℝ) / 128)) = (Real.exp ((k : ℝ) / 128))⁻¹ := Real.exp_neg _
  rw [hneg_exp]
  have hpos_e : 0 < Real.exp ((k : ℝ) / 128) := Real.exp_pos _
  have hdiv : 4 * ((1 + 2 * (4 * k + 1) ^ d : ℕ) : ℝ) ^ D * (Real.exp ((k : ℝ) / 128))⁻¹ < 1 := by
    rw [mul_inv_lt_iff₀ hpos_e, one_mul]; exact hprod_lt
  linarith

/-- Asymptotic existence of Stage 2 batch size `t = t(a)`:
for every `a ≥ 1` and `η > 0`, there exists `t ≥ 8` with `t * η ≥ 8` and
`2 * (2 * t + 1)^(2 * a) * exp(-η * t / 4) < 1`. -/
lemma exists_stage2_t (a η : ℝ) (ha : 1 ≤ a) (hη : 0 < η) :
    ∃ t : ℕ, 8 ≤ t ∧ 8 ≤ (t : ℝ) * η ∧
      2 * (2 * (t : ℝ) + 1) ^ (2 * a) * Real.exp (-(η * (t : ℝ) / 4)) < 1 := by
  -- Use elementary log(2t + 1) ≤ 2 √(2t + 1) ≤ 4 √t, so log 2 + 2a log(2t + 1) ≤ (8a + 1) √t < η t / 4!
  let B : ℝ := 4 * (8 * a + 2) / η
  let M : ℕ := Nat.ceil (max B 0) + 1
  let t : ℕ := max 8 (max (Nat.ceil (8 / η)) (M ^ 2))
  refine ⟨t, le_max_left _ _, ?_, ?_⟩
  · have h1 : 8 / η ≤ (t : ℝ) :=
      (Nat.le_ceil _).trans (by exact_mod_cast le_trans (le_max_left _ _) (le_max_right 8 _))
    have := mul_le_mul_of_nonneg_right h1 hη.le
    rwa [div_mul_cancel₀ _ hη.ne'] at this
  · have ht8 : (8 : ℝ) ≤ (t : ℝ) := by exact_mod_cast le_max_left 8 _
    have htM : (M : ℝ) ^ 2 ≤ (t : ℝ) := by
      exact_mod_cast le_trans (le_max_right _ _) (le_max_right 8 _)
    have hsqrt_gt : B < Real.sqrt (t : ℝ) := by
      have hB : B < (M : ℝ) := by
        have h1 : B ≤ max B 0 := le_max_left _ _
        have h2 : max B 0 ≤ (Nat.ceil (max B 0) : ℝ) := Nat.le_ceil _
        have h3 : (M : ℝ) = (Nat.ceil (max B 0) : ℝ) + 1 := by simp [M]
        linarith
      have hMle : (M : ℝ) ≤ Real.sqrt (t : ℝ) := by
        rw [Real.le_sqrt (by positivity) (by linarith)]; exact htM
      linarith
    have hsqrt2t : Real.sqrt (2 * (t : ℝ) + 1) ≤ 2 * Real.sqrt (t : ℝ) := by
      rw [Real.sqrt_le_iff]
      refine ⟨by positivity, ?_⟩
      rw [mul_pow, Real.sq_sqrt (by linarith)]; linarith
    have hlog2 : Real.log 2 < 2 := by
      rw [Real.log_lt_iff_lt_exp (by norm_num)]
      have h1 : (2 : ℝ) ≤ Real.exp 1 := by have := Real.add_one_le_exp (1 : ℝ); linarith
      have hsq := pow_le_pow_left₀ (by norm_num : (0 : ℝ) ≤ 2) h1 2
      rw [← Real.exp_nat_mul] at hsq; norm_num at hsq; linarith
    have hlog_lt : Real.log 2 + 2 * a * Real.log (2 * (t : ℝ) + 1) < η * (t : ℝ) / 4 := by
      have hl : Real.log (2 * (t : ℝ) + 1) ≤ 2 * Real.sqrt (2 * (t : ℝ) + 1) :=
        log_le_two_sqrt (by linarith)
      have h1 : 2 * a * Real.log (2 * (t : ℝ) + 1) ≤ 8 * a * Real.sqrt (t : ℝ) := by nlinarith
      have hsqrt_ge1 : 1 ≤ Real.sqrt (t : ℝ) := by rw [Real.le_sqrt (by norm_num) (by linarith)]; linarith
      have hstep : 8 * a + 2 < η * Real.sqrt (t : ℝ) / 4 := by
        have hm := mul_lt_mul_of_pos_right hsqrt_gt hη
        have hBeq : B * η = 4 * (8 * a + 2) := by simp only [B]; field_simp [hη.ne']
        linarith
      have hmul := mul_lt_mul_of_pos_right hstep (by linarith : 0 < Real.sqrt (t : ℝ))
      have hrhs : η * Real.sqrt (t : ℝ) / 4 * Real.sqrt (t : ℝ) = η * (t : ℝ) / 4 := by
        rw [div_mul_eq_mul_div, mul_assoc, Real.mul_self_sqrt (by linarith)]
      nlinarith
    have hprod_lt : 2 * (2 * (t : ℝ) + 1) ^ (2 * a) < Real.exp (η * (t : ℝ) / 4) := by
      have hpos : 0 < (2 * (t : ℝ) + 1) ^ (2 * a) := by positivity
      rw [← Real.exp_log (mul_pos (by norm_num : (0 : ℝ) < 2) hpos),
        Real.log_mul (by norm_num) hpos.ne', Real.log_rpow (by linarith)]
      exact Real.exp_lt_exp.mpr hlog_lt
    rw [Real.exp_neg, mul_inv_lt_iff₀ (Real.exp_pos _), one_mul]
    exact hprod_lt

open Classical in
/-- Stage 1 helper lemma: given a uniform distribution `P_pair` on `Ω_pair = {p : Y × Y // p.1 ≠ p.2}`
and random depth-`≤ D` identification trees `T_tr τ`, every transversal `W_Y` of size `> |Y| / 2`
contains a pair `y₁ ≠ y₂` whose disagreement set has `Q`-mass `≥ 1 / (4 * 2^D)`. -/
private lemma stage1_heavy_pair {X : Type*} {m k d D : ℕ}
    (x₀ : X) (C V : Set (Concept X)) (hVC : V ⊆ C) (S : Fin m → X)
    (Y : Finset (Fin m → Bool)) (hsmallN : 8 ≤ Y.card)
    (hk256 : 256 ≤ k)
    (hk_bound : 2 * ((1 + 2 * (4 * k + 1) ^ d : ℕ) : ℝ) ^ D * Real.exp (-((k : ℝ) / 128)) < 1 / 2)
    (hdvc : HasVCDimLe (DualFamily C) d)
    (sepIdx : { p : Y × Y // p.1 ≠ p.2 } → Fin m)
    (hsepIdx : ∀ p : { p : Y × Y // p.1 ≠ p.2 }, p.1.1.1 (sepIdx p) ≠ p.1.2.1 (sepIdx p))
    (T_tr : (Fin k → { p : Y × Y // p.1 ≠ p.2 }) → MQTree X)
    (hT_id : ∀ τ, (T_tr τ).IdentifiesOn C (fun r => S (sepIdx (τ r))))
    (hT_dep : ∀ τ, (T_tr τ).depth ≤ D)
    (hpair_card : Fintype.card { p : Y × Y // p.1 ≠ p.2 } = Y.card * (Y.card - 1))
    (P_pair : { p : Y × Y // p.1 ≠ p.2 } → ℝ)
    (hP_pair_def : ∀ p, P_pair p = 1 / (Fintype.card { p : Y × Y // p.1 ≠ p.2 } : ℝ))
    (hP_pair_nn : ∀ p, 0 ≤ P_pair p)
    (hP_pair_sum : ∑ p : { p : Y × Y // p.1 ≠ p.2 }, P_pair p = 1)
    (W_Y : Finset Y) (rep : Y → Concept X)
    (hWY_gt : Y.card < 2 * W_Y.card)
    (hrepV : ∀ y ∈ W_Y, rep y ∈ V ∧ restricts S (rep y) = y.1) :
    let padQ : (Fin k → { p : Y × Y // p.1 ≠ p.2 }) → Fin (2 ^ D) → X :=
      fun τ j => ((T_tr τ).queries).getD j.val x₀
    ∃ y₁ ∈ W_Y, ∃ y₂ ∈ W_Y, y₁ ≠ y₂ ∧
      (1 : ℝ) / (4 * (2 ^ D : ℝ)) ≤
        ∑ τ : Fin k → { p : Y × Y // p.1 ≠ p.2 }, prodPMF P_pair τ *
          ((∑ j : Fin (2 ^ D), if padQ τ j ∈ {x : X | rep y₁ x ≠ rep y₂ x} then (1 : ℝ) else 0) / (2 ^ D : ℝ)) := by
  dsimp only
  let Ω_pair := { p : Y × Y // p.1 ≠ p.2 }
  have hgrow_W : ∀ E : Finset Ω_pair, E.card ≤ 2 * k →
      (TraceSet (conjPairFamily W_Y rep D) E).card ≤ (1 + 2 * (4 * k + 1) ^ d) ^ D := by
    intro E hE
    have hrepC : ∀ y ∈ W_Y, rep y ∈ C := fun y hy => hVC (hrepV y hy).1
    calc (TraceSet (conjPairFamily W_Y rep D) E).card
        ≤ (1 + 2 * (2 * E.card + 1) ^ d) ^ D :=
          conjunction_pair_range_growth hdvc W_Y rep hrepC D E
      _ ≤ (1 + 2 * (4 * k + 1) ^ d) ^ D := by
          apply Nat.pow_le_pow_left
          have : 2 * E.card + 1 ≤ 4 * k + 1 := by omega
          have := Nat.pow_le_pow_left this d
          omega
  have hnet := finite_epsilon_net_bound hP_pair_nn hP_pair_sum
    (conjPairFamily W_Y rep D) (n := k) (M_grow := (1 + 2 * (4 * k + 1) ^ d) ^ D)
    (by omega) hgrow_W (ε := 1 / 32) (by norm_num) (by
      have : (256 : ℝ) ≤ (k : ℝ) := by exact_mod_cast hk256
      linarith)
  have hmiss_lt_half :
      (∑ τ : Fin k → Ω_pair,
        if (∃ R ∈ conjPairFamily W_Y rep D, (1 / 32 : ℝ) ≤ rangeMass P_pair R ∧ ∀ r : Fin k, τ r ∉ R)
        then prodPMF P_pair τ else 0) < 1 / 2 := by
    have hexp_eq : -((1 / 32 : ℝ) * (k : ℝ) / 4) = -((k : ℝ) / 128) := by ring
    rw [hexp_eq] at hnet
    have hcast_eq : (((1 + 2 * (4 * k + 1) ^ d) ^ D : ℕ) : ℝ) =
        ((1 + 2 * (4 * k + 1) ^ d : ℕ) : ℝ) ^ D := by push_cast; rfl
    rw [hcast_eq] at hnet
    linarith
  let W : Finset (Concept X) := W_Y.image rep
  have hrep_inj : Set.InjOn rep (W_Y : Set Y) := by
    intro y₁ h1 y₂ h2 heq
    apply Subtype.ext
    rw [← (hrepV y₁ h1).2, ← (hrepV y₂ h2).2, heq]
  have hW_card : W.card = W_Y.card := Finset.card_image_of_injOn hrep_inj
  have hW_ge2 : 2 ≤ W.card := by rw [hW_card]; omega
  have hgood_of_hit : ∀ τ : Fin k → Ω_pair,
      ¬ (∃ R ∈ conjPairFamily W_Y rep D, (1 / 32 : ℝ) ≤ rangeMass P_pair R ∧ ∀ r : Fin k, τ r ∉ R) →
      ∀ f ∈ W, 2 * (W.filter fun g => (T_tr τ).leafPath g = (T_tr τ).leafPath f).card < W.card := by
    intro τ hhit f hfW
    by_contra hbig; push_neg at hbig
    let L := (T_tr τ).transcript f
    have hLlen : L.length ≤ D := ((T_tr τ).transcript_length_le f).trans (hT_dep τ)
    let R : Set Ω_pair := conjPairRange W_Y rep L
    have hR_mem : R ∈ conjPairFamily W_Y rep D := ⟨L, hLlen, rfl⟩
    have hR_miss : ∀ r : Fin k, τ r ∉ R := by
      intro r hmem
      rcases hmem with ⟨h1W, h2W, h1L, h2L⟩
      have hp1 := ((T_tr τ).transcript_iff_leafPath f (rep (τ r).1.1)).mpr h1L
      have hp2 := ((T_tr τ).transcript_iff_leafPath f (rep (τ r).1.2)).mpr h2L
      have hsame_leaf : (T_tr τ).leafPath (rep (τ r).1.1) = (T_tr τ).leafPath (rep (τ r).1.2) :=
        hp1.symm.trans hp2
      have hrest := hT_id τ (rep (τ r).1.1) (hVC (hrepV _ h1W).1)
        (rep (τ r).1.2) (hVC (hrepV _ h2W).1) hsame_leaf
      have hsep_eq := congrFun hrest r
      simp only [restricts] at hsep_eq
      have e1 : (τ r).1.1.1 (sepIdx (τ r)) = rep (τ r).1.1 (S (sepIdx (τ r))) :=
        (congrFun (hrepV _ h1W).2 (sepIdx (τ r))).symm
      have e2 : (τ r).1.2.1 (sepIdx (τ r)) = rep (τ r).1.2 (S (sepIdx (τ r))) :=
        (congrFun (hrepV _ h2W).2 (sepIdx (τ r))).symm
      exact hsepIdx (τ r) (e1.trans (hsep_eq.trans e2.symm))
    let F_Y : Finset Y := W_Y.filter fun y => ∀ qb ∈ L, rep y qb.1 = qb.2
    have hFY_card : (W.filter fun g => (T_tr τ).leafPath g = (T_tr τ).leafPath f).card ≤ F_Y.card := by
      have hsub : (W.filter fun g => (T_tr τ).leafPath g = (T_tr τ).leafPath f) ⊆ F_Y.image rep := by
        intro g hg
        rcases Finset.mem_filter.mp hg with ⟨hgW, hpath⟩
        rcases Finset.mem_image.mp hgW with ⟨y, hyW, rfl⟩
        refine Finset.mem_image.mpr ⟨y, Finset.mem_filter.mpr ⟨hyW, ?_⟩, rfl⟩
        exact ((T_tr τ).transcript_iff_leafPath f (rep y)).mp hpath.symm
      exact (Finset.card_le_card hsub).trans Finset.card_image_le
    have hc_ge : W_Y.card ≤ 2 * F_Y.card := by omega
    have hR_card : F_Y.card * (F_Y.card - 1) ≤ (Finset.univ.filter fun p : Ω_pair => p ∈ R).card := by
      let F_pairs : Finset (Y × Y) := (F_Y ×ˢ F_Y).filter fun q => q.1 ≠ q.2
      have hFpairs_card : F_pairs.card = F_Y.card * (F_Y.card - 1) := by
        have hdiag : (F_Y ×ˢ F_Y).filter (fun q => q.1 = q.2) = F_Y.image fun y => (y, y) := by
          ext ⟨y, z⟩; simp only [Finset.mem_filter, Finset.mem_product, Finset.mem_image]
          exact ⟨fun ⟨⟨hy, _⟩, h⟩ => ⟨y, hy, Prod.ext rfl h⟩,
            fun ⟨w, hw, h⟩ => by rcases Prod.mk.inj h with ⟨rfl, rfl⟩; exact ⟨⟨hw, hw⟩, rfl⟩⟩
        have hsp := Finset.filter_card_add_filter_neg_card_eq_card (s := F_Y ×ˢ F_Y) (p := fun q => q.1 = q.2)
        rw [hdiag, Finset.card_image_of_injective _ (fun _ _ h => congrArg Prod.fst h), Finset.card_product] at hsp
        have hmul : F_Y.card * F_Y.card = F_Y.card + F_Y.card * (F_Y.card - 1) := by
          cases hc : F_Y.card with
          | zero => omega
          | succ r => simp [Nat.mul_succ]; ring
        have hneq_eq : ((F_Y ×ˢ F_Y).filter fun q => ¬ q.1 = q.2).card = F_pairs.card := rfl
        rw [hneq_eq] at hsp
        omega
      rw [← hFpairs_card]
      let emb : F_pairs → Ω_pair := fun q => ⟨q.1, (Finset.mem_filter.mp q.2).2⟩
      have hemb_inj : Function.Injective emb := fun q₁ q₂ h =>
        Subtype.ext (by simpa [emb] using congrArg Subtype.val h)
      calc F_pairs.card = (Finset.univ.image emb).card := by
            rw [Finset.card_image_of_injective _ hemb_inj, Finset.card_univ, Fintype.card_coe]
        _ ≤ (Finset.univ.filter fun p : Ω_pair => p ∈ R).card := by
            apply Finset.card_le_card
            intro p hp
            rcases Finset.mem_image.mp hp with ⟨⟨q, hq⟩, _, rfl⟩
            have hq' := Finset.mem_filter.mp hq
            have h1 := Finset.mem_filter.mp (Finset.mem_product.mp hq'.1).1
            have h2 := Finset.mem_filter.mp (Finset.mem_product.mp hq'.1).2
            exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, ⟨h1.1, h2.1, h1.2, h2.2⟩⟩
    have hR_mass : (1 / 32 : ℝ) ≤ rangeMass P_pair R := by
      have hrm : rangeMass P_pair R =
          ((Finset.univ.filter fun p : Ω_pair => p ∈ R).card : ℝ) / (Fintype.card Ω_pair : ℝ) := by
        unfold rangeMass
        calc ∑ p : Ω_pair, (if p ∈ R then P_pair p else 0)
            = ∑ p : Ω_pair, (if p ∈ R then (1 / (Fintype.card Ω_pair : ℝ)) else 0) :=
              Finset.sum_congr rfl fun p _ => by
                by_cases hp : p ∈ R <;> (simp only [hp, ↓reduceIte, hP_pair_def]; try rfl)
          _ = ((Finset.univ.filter fun p : Ω_pair => p ∈ R).card : ℝ) * (1 / (Fintype.card Ω_pair : ℝ)) := by
              rw [Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const, nsmul_eq_mul]
          _ = _ := by ring
      rw [hrm, hpair_card]
      calc (1 / 32 : ℝ) ≤ ((F_Y.card * (F_Y.card - 1) : ℕ) : ℝ) / ((Y.card * (Y.card - 1) : ℕ) : ℝ) :=
            conjunction_pair_mass_ge hsmallN hWY_gt hc_ge
        _ ≤ ((Finset.univ.filter fun p : Ω_pair => p ∈ R).card : ℝ) / ((Y.card * (Y.card - 1) : ℕ) : ℝ) :=
            div_le_div_of_nonneg_right (by exact_mod_cast hR_card) (by positivity)
    exact hhit ⟨R, hR_mem, hR_mass, hR_miss⟩
  have hgood_prob : (1 / 2 : ℝ) ≤ ∑ τ : Fin k → Ω_pair,
      if (∀ f ∈ W, 2 * (W.filter fun g => (T_tr τ).leafPath g = (T_tr τ).leafPath f).card < W.card)
      then prodPMF P_pair τ else 0 := by
    have hcompl : (∑ τ : Fin k → Ω_pair,
        if ¬ (∃ R ∈ conjPairFamily W_Y rep D, (1 / 32 : ℝ) ≤ rangeMass P_pair R ∧ ∀ r : Fin k, τ r ∉ R)
        then prodPMF P_pair τ else 0) +
      (∑ τ : Fin k → Ω_pair,
        if (∃ R ∈ conjPairFamily W_Y rep D, (1 / 32 : ℝ) ≤ rangeMass P_pair R ∧ ∀ r : Fin k, τ r ∉ R)
        then prodPMF P_pair τ else 0) = 1 := by
      rw [← Finset.sum_add_distrib, ← prodPMF_sum_one (n := k) hP_pair_sum]
      refine Finset.sum_congr rfl fun τ _ => by split_ifs <;> ring
    have hle : (∑ τ : Fin k → Ω_pair,
        if ¬ (∃ R ∈ conjPairFamily W_Y rep D, (1 / 32 : ℝ) ≤ rangeMass P_pair R ∧ ∀ r : Fin k, τ r ∉ R)
        then prodPMF P_pair τ else 0) ≤
      (∑ τ : Fin k → Ω_pair,
        if (∀ f ∈ W, 2 * (W.filter fun g => (T_tr τ).leafPath g = (T_tr τ).leafPath f).card < W.card)
        then prodPMF P_pair τ else 0) := by
      refine Finset.sum_le_sum fun τ _ => ?_
      by_cases h1 : ∃ R ∈ conjPairFamily W_Y rep D, (1 / 32 : ℝ) ≤ rangeMass P_pair R ∧ ∀ r : Fin k, τ r ∉ R
      · simp only [h1, not_true_eq_false, ↓reduceIte]
        split_ifs
        · exact prodPMF_nonneg hP_pair_nn τ
        · exact le_rfl
      · simp only [h1, not_false_eq_true, ↓reduceIte]
        rw [if_pos (hgood_of_hit τ h1)]
    linarith
  obtain ⟨f, hfW, g, hgW, hfg, hmass_fg⟩ :=
    heavy_pair_exists_in_transversal D x₀ (prodPMF P_pair) (prodPMF_nonneg hP_pair_nn)
      T_tr hT_dep W hW_ge2 hgood_prob
  rcases Finset.mem_image.mp hfW with ⟨y₁, hy₁, rfl⟩
  rcases Finset.mem_image.mp hgW with ⟨y₂, hy₂, rfl⟩
  have hyne : y₁ ≠ y₂ := fun h => hfg (congrArg rep h)
  exact ⟨y₁, hy₁, y₂, hy₂, hyne, by simpa using hmass_fg⟩

/-- Stage 2 growth bound for pairwise disagreement sets on any finite subset `Ω_Q` of `X`. -/
private lemma stage2_disagreement_growth {X : Type*} {a : ℝ} (ha : 1 ≤ a)
    {C : Set (Concept X)} (haic : AICBound C a)
    (Ω_Q_fin : Finset X) (t : ℕ) (E : Finset {x : X // x ∈ Ω_Q_fin}) (hE : E.card ≤ 2 * t) :
    ((TraceSet ({R : Set {x : X // x ∈ Ω_Q_fin} |
        ∃ f ∈ C, ∃ g ∈ C, R = {x | f x.1 ≠ g x.1}}) E).card : ℝ) ≤
      (2 * (t : ℝ) + 1) ^ (2 * a) := by
  classical
  let Ω_Q := {x : X // x ∈ Ω_Q_fin}
  let 𝒟 : Set (Set Ω_Q) := {R | ∃ f ∈ C, ∃ g ∈ C, R = {x : Ω_Q | f x.1 ≠ g x.1}}
  let E_X : Finset X := E.image Subtype.val
  have hEX_card : E_X.card = E.card :=
    Finset.card_image_of_injective _ Subtype.val_injective
  let pairMap : TraceSet (PrimalFamily C) E_X × TraceSet (PrimalFamily C) E_X → Finset Ω_Q :=
    fun p => E.filter fun x => (x.1 ∈ p.1.1) ≠ (x.1 ∈ p.2.1)
  have hsub : TraceSet 𝒟 E ⊆ Finset.univ.image pairMap := by
    intro A hA
    rcases mem_TraceSet.mp hA with ⟨hAsub, _, ⟨f, hfC, g, hgC, rfl⟩, hAiff⟩
    let Af : Finset X := E_X.filter fun x => f x = true
    let Ag : Finset X := E_X.filter fun x => g x = true
    have hAf : Af ∈ TraceSet (PrimalFamily C) E_X :=
      mem_TraceSet.mpr ⟨Finset.filter_subset _ _,
        ⟨{x | f x = true}, ⟨f, hfC, rfl⟩, fun x hx => by simp [Af, hx]⟩⟩
    have hAg : Ag ∈ TraceSet (PrimalFamily C) E_X :=
      mem_TraceSet.mpr ⟨Finset.filter_subset _ _,
        ⟨{x | g x = true}, ⟨g, hgC, rfl⟩, fun x hx => by simp [Ag, hx]⟩⟩
    refine Finset.mem_image.mpr ⟨(⟨Af, hAf⟩, ⟨Ag, hAg⟩), Finset.mem_univ _, ?_⟩
    ext x
    simp only [pairMap, Finset.mem_filter]
    constructor
    · rintro ⟨hxE, hne⟩
      rw [← hAiff x hxE]
      have hxEX : x.1 ∈ E_X := Finset.mem_image.mpr ⟨x, hxE, rfl⟩
      simp only [Af, Ag, Finset.mem_filter, hxEX, true_and, Set.mem_setOf_eq] at hne ⊢
      cases h1 : f x.1 <;> cases h2 : g x.1 <;> simp_all
    · intro hxA
      have hxE := hAsub hxA
      have hfg := (hAiff x hxE).mpr hxA
      have hxEX : x.1 ∈ E_X := Finset.mem_image.mpr ⟨x, hxE, rfl⟩
      refine ⟨hxE, ?_⟩
      simp only [Af, Ag, Finset.mem_filter, hxEX, true_and]
      simp only [Set.mem_setOf_eq] at hfg
      cases h1 : f x.1 <;> cases h2 : g x.1 <;> simp_all
  have hcard_D : (TraceSet 𝒟 E).card ≤ (TraceSet (PrimalFamily C) E_X).card ^ 2 := by
    calc (TraceSet 𝒟 E).card
        ≤ (Finset.univ.image pairMap).card := Finset.card_le_card hsub
      _ ≤ (Finset.univ : Finset (TraceSet (PrimalFamily C) E_X × TraceSet (PrimalFamily C) E_X)).card :=
          Finset.card_image_le
      _ = (TraceSet (PrimalFamily C) E_X).card ^ 2 := by
          rw [Finset.card_univ, Fintype.card_prod, Fintype.card_coe, sq]
  by_cases hE0 : E.card = 0
  · have h1 : (TraceSet 𝒟 E).card ≤ 1 := by
      rw [Finset.card_eq_zero.mp hE0]
      exact TraceSet_empty_card_le_one 𝒟
    calc ((TraceSet 𝒟 E).card : ℝ) ≤ 1 := by exact_mod_cast h1
      _ ≤ (2 * (t : ℝ) + 1) ^ (2 * a) := Real.one_le_rpow (by linarith) (by linarith)
  · have hEX_pos : 1 ≤ E_X.card := by omega
    have hpr := primal_growth_of_aic haic E_X hEX_pos
    have hle_t : (E_X.card : ℝ) + 1 ≤ 2 * (t : ℝ) + 1 := by
      have : E_X.card ≤ 2 * t := by omega
      exact_mod_cast Nat.add_le_add_right this 1
    calc ((TraceSet 𝒟 E).card : ℝ)
        ≤ ((TraceSet (PrimalFamily C) E_X).card : ℝ) ^ 2 := by exact_mod_cast hcard_D
      _ ≤ (((E_X.card : ℝ) + 1) ^ a) ^ 2 := pow_le_pow_left₀ (Nat.cast_nonneg _) hpr 2
      _ = ((E_X.card : ℝ) + 1) ^ (2 * a) := by
          rw [← Real.rpow_natCast, ← Real.rpow_mul (by positivity)]; congr 1; push_cast; ring
      _ ≤ (2 * (t : ℝ) + 1) ^ (2 * a) :=
          Real.rpow_le_rpow (by positivity) hle_t (by linarith)

/-- **Theorem 1 (Resolution of `conj:halve`, `arXiv:2609.27241`)**:
Fix `a ≥ 1`. There exists a finite constant `t = t(a) : ℕ` (depending ONLY on `a`) such that
for EVERY domain `X`, EVERY binary concept class `C ⊆ {0, 1}^X` satisfying
`AIC(C, m) ≤ a log₂(m + 1)` (`∀ m ≥ 1`), EVERY sample size `m : ℕ`, EVERY sample `S : Fin m → X`,
and EVERY version space `V ⊆ C` (with arbitrary, unrestricted fibers over `V_S`), there exists
a fixed non-adaptive query batch `U : Fin t → X` that halves `V_S` under every answer pattern
`b : Fin t → Bool`:
`|(V[U = b])_S| ≤ ⌈|V_S| / 2⌉`. -/
theorem active_mq_shd_bounded {a : ℝ} (ha : 1 ≤ a) :
    ∃ t : ℕ, ∀ (X : Type*) [Nonempty X] (C : Set (Concept X)),
      AICBound C a →
      ∀ (m : ℕ) (S : Fin m → X) (V : Set (Concept X)),
        V ⊆ C → ∃ U : Fin t → X, HalvesOn U V S := by
  classical
  -- Constants depending ONLY on a:
  obtain ⟨v, hv⟩ := exists_vc_bound_of_aic ha
  let d : ℕ := 2 ^ (v + 1) - 1
  obtain ⟨k, hk256, hk_bound⟩ := exists_stage1_k a ha d
  let D : ℕ := Nat.ceil (a * Real.logb 2 ((k : ℝ) + 1))
  let η : ℝ := 1 / (4 * (2 ^ D : ℝ))
  have hη_pos : 0 < η := by positivity
  obtain ⟨t, ht8, htη, ht_bound⟩ := exists_stage2_t a η ha hη_pos
  refine ⟨t, fun X hneX C haic m S V hVC => ?_⟩
  obtain ⟨x₀⟩ := hneX
  let Y : Finset (Fin m → Bool) := labelings V S
  -- Small-N base case (N < 8)
  by_cases hsmallN : Y.card < 8
  · exact small_N_halving x₀ ht8 V S hsmallN
  push_neg at hsmallN
  -- Assume henceforth N = |Y| ≥ 8
  -- Dual VC dimension ≤ d
  have hdvc : HasVCDimLe (DualFamily C) d := assouad_dual_vc_bound (hv X C haic)
  -- Stage 1: Ω_pair = ordered distinct pairs from Y, size N(N - 1) > 0
  let Ω_pair : Type := { p : Y × Y // p.1 ≠ p.2 }
  have hpair_card : Fintype.card Ω_pair = Y.card * (Y.card - 1) := by
    have heq : Fintype.card Ω_pair = ((Y ×ˢ Y).filter fun p => p.1 ≠ p.2).card := by
      have e : Ω_pair ≃ { p : (Fin m → Bool) × (Fin m → Bool) // p ∈ (Y ×ˢ Y).filter fun p => p.1 ≠ p.2 } :=
        { toFun := fun ⟨⟨⟨y, hy⟩, ⟨z, hz⟩⟩, hne⟩ =>
            ⟨(y, z), Finset.mem_filter.mpr ⟨Finset.mem_product.mpr ⟨hy, hz⟩, fun h => hne (Subtype.ext h)⟩⟩
          invFun := fun ⟨⟨y, z⟩, hp⟩ =>
            let hp' := Finset.mem_filter.mp hp
            ⟨⟨⟨y, (Finset.mem_product.mp hp'.1).1⟩, ⟨z, (Finset.mem_product.mp hp'.1).2⟩⟩,
              fun h => hp'.2 (congrArg Subtype.val h)⟩
          left_inv := fun ⟨⟨⟨_, _⟩, ⟨_, _⟩⟩, _⟩ => rfl
          right_inv := fun ⟨⟨_, _⟩, _⟩ => rfl }
      rw [Fintype.card_congr e, Fintype.card_coe]
    rw [heq]
    have hdiag : (Y ×ˢ Y).filter (fun p => p.1 = p.2) = Y.image fun y => (y, y) := by
      ext ⟨y, z⟩; simp only [Finset.mem_filter, Finset.mem_product, Finset.mem_image]
      exact ⟨fun ⟨⟨hy, _⟩, h⟩ => ⟨y, hy, Prod.ext rfl h⟩,
        fun ⟨w, hw, h⟩ => by rcases Prod.mk.inj h with ⟨rfl, rfl⟩; exact ⟨⟨hw, hw⟩, rfl⟩⟩
    have hsp := Finset.filter_card_add_filter_neg_card_eq_card (s := Y ×ˢ Y) (p := fun p => p.1 = p.2)
    rw [hdiag, Finset.card_image_of_injective _ (fun _ _ h => congrArg Prod.fst h), Finset.card_product] at hsp
    have hmul : Y.card * Y.card = Y.card + Y.card * (Y.card - 1) := by
      cases hc : Y.card with
      | zero => omega
      | succ r => simp [Nat.mul_succ]; ring
    have hneq_eq : ((Y ×ˢ Y).filter fun p => ¬ p.1 = p.2).card =
        ((Y ×ˢ Y).filter fun p => p.1 ≠ p.2).card := rfl
    rw [hneq_eq] at hsp
    omega
  have hpair_pos : 0 < Fintype.card Ω_pair := by
    rw [hpair_card]; exact Nat.mul_pos (by omega) (by omega)
  have hsep_ex : ∀ p : Ω_pair, ∃ i : Fin m, p.1.1.1 i ≠ p.1.2.1 i := by
    intro p
    by_contra hall; push_neg at hall
    exact p.2 (Subtype.ext (funext hall))
  choose sepIdx hsepIdx using hsep_ex
  let sep : Ω_pair → X := fun p => S (sepIdx p)
  have hT_ex : ∀ τ : Fin k → Ω_pair,
      ∃ T : MQTree X, T.IdentifiesOn C (fun r => sep (τ r)) ∧ T.depth ≤ D := by
    intro τ
    obtain ⟨T, hTid, hTdep⟩ := haic k (by omega) (fun r => sep (τ r))
    have h1 : (T.depth : ℝ) ≤ (D : ℝ) := hTdep.trans (Nat.le_ceil _)
    exact ⟨T, hTid, by exact_mod_cast h1⟩
  choose T_tr hT_id hT_dep using hT_ex
  let P_pair : Ω_pair → ℝ := fun _ => 1 / (Fintype.card Ω_pair : ℝ)
  have hP_pair_nn : ∀ p, 0 ≤ P_pair p := fun _ => by positivity
  have hP_pair_sum : ∑ p : Ω_pair, P_pair p = 1 := by
    rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    have hpos : (Fintype.card Ω_pair : ℝ) ≠ 0 := by exact_mod_cast hpair_pos.ne'
    field_simp [hpos]
  let padQ : (Fin k → Ω_pair) → Fin (2 ^ D) → X := fun τ j => ((T_tr τ).queries).getD j.val x₀
  let Ω_Q_fin : Finset X := Finset.univ.biUnion fun τ : Fin k → Ω_pair =>
    Finset.univ.image fun j : Fin (2 ^ D) => padQ τ j
  let Ω_Q := { x : X // x ∈ Ω_Q_fin }
  let Q : Ω_Q → ℝ := fun x =>
    ∑ τ : Fin k → Ω_pair, prodPMF P_pair τ *
      ((∑ j : Fin (2 ^ D), if padQ τ j = x.1 then (1 : ℝ) else 0) / (2 ^ D : ℝ))
  have hQ_nn : ∀ x : Ω_Q, 0 ≤ Q x := fun x =>
    Finset.sum_nonneg fun τ _ => mul_nonneg (prodPMF_nonneg hP_pair_nn τ) (by positivity)
  have hQ_sum : ∑ x : Ω_Q, Q x = 1 := by
    simp only [Q]
    rw [Finset.sum_comm]
    calc ∑ τ : Fin k → Ω_pair, ∑ x : Ω_Q, prodPMF P_pair τ *
            ((∑ j : Fin (2 ^ D), if padQ τ j = x.1 then (1 : ℝ) else 0) / (2 ^ D : ℝ))
        = ∑ τ : Fin k → Ω_pair, prodPMF P_pair τ * 1 := by
          refine Finset.sum_congr rfl fun τ _ => ?_
          rw [← Finset.mul_sum, ← Finset.sum_div]
          congr 1
          rw [Finset.sum_comm]
          have h1 : ∀ j : Fin (2 ^ D),
              (∑ x : Ω_Q, if padQ τ j = x.1 then (1 : ℝ) else 0) = 1 := by
            intro j
            have hmem : padQ τ j ∈ Ω_Q_fin :=
              Finset.mem_biUnion.mpr ⟨τ, Finset.mem_univ _, Finset.mem_image.mpr ⟨j, Finset.mem_univ _, rfl⟩⟩
            rw [Finset.sum_eq_single ⟨padQ τ j, hmem⟩]
            · simp
            · intro x _ hx
              have hne : padQ τ j ≠ x.1 := fun h => hx (Subtype.ext h.symm)
              simp [hne]
            · intro h; exact (h (Finset.mem_univ _)).elim
          simp_rw [h1]
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one]
          push_cast
          exact div_self (by positivity)
      _ = 1 := by simp_rw [mul_one]; exact prodPMF_sum_one hP_pair_sum
  -- Exact identity: rangeMass Q {x : Ω_Q | x.1 ∈ A} = Q_mass A
  have hQ_mass_eq : ∀ A : Set X,
      rangeMass Q {x : Ω_Q | x.1 ∈ A} =
      ∑ τ : Fin k → Ω_pair, prodPMF P_pair τ *
        ((∑ j : Fin (2 ^ D), if padQ τ j ∈ A then (1 : ℝ) else 0) / (2 ^ D : ℝ)) := by
    intro A
    simp only [rangeMass, Q, Set.mem_setOf_eq]
    have h1 : ∀ x : Ω_Q,
        (if x.1 ∈ A then ∑ τ : Fin k → Ω_pair, prodPMF P_pair τ *
            ((∑ j : Fin (2 ^ D), if padQ τ j = x.1 then (1 : ℝ) else 0) / (2 ^ D : ℝ)) else 0) =
        ∑ τ : Fin k → Ω_pair, prodPMF P_pair τ *
          ((∑ j : Fin (2 ^ D), if x.1 ∈ A ∧ padQ τ j = x.1 then (1 : ℝ) else 0) / (2 ^ D : ℝ)) := by
      intro x; split_ifs with hxA
      · refine Finset.sum_congr rfl fun τ _ => ?_
        congr 2; refine Finset.sum_congr rfl fun j _ => by simp [hxA]
      · symm; refine Finset.sum_eq_zero fun τ _ => ?_
        have : (∑ j : Fin (2 ^ D), if x.1 ∈ A ∧ padQ τ j = x.1 then (1 : ℝ) else 0) = 0 :=
          Finset.sum_eq_zero fun j _ => by simp [hxA]
        rw [this, zero_div, mul_zero]
    simp_rw [h1]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun τ _ => ?_
    rw [← Finset.mul_sum, ← Finset.sum_div]
    congr 2
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun j _ => ?_
    have hmem : padQ τ j ∈ Ω_Q_fin :=
      Finset.mem_biUnion.mpr ⟨τ, Finset.mem_univ _, Finset.mem_image.mpr ⟨j, Finset.mem_univ _, rfl⟩⟩
    rw [Finset.sum_eq_single ⟨padQ τ j, hmem⟩]
    · simp
    · intro x _ hx
      have hne : padQ τ j ≠ x.1 := fun h => hx (Subtype.ext h.symm)
      simp [hne]
    · intro h; exact (h (Finset.mem_univ _)).elim
  -- Stage 1 claim: EVERY transversal W_Y ⊆ Y of size h > N/2 with representatives rep : Y → V
  -- contains a pair f ≠ g with Q-mass(D_{f, g}) ≥ η!
  have hStage1 : ∀ (W_Y : Finset Y) (rep : Y → Concept X),
      Y.card < 2 * W_Y.card →
      (∀ y ∈ W_Y, rep y ∈ V ∧ restricts S (rep y) = y.1) →
      ∃ y₁ ∈ W_Y, ∃ y₂ ∈ W_Y, y₁ ≠ y₂ ∧
        η ≤ rangeMass Q {x : Ω_Q | rep y₁ x.1 ≠ rep y₂ x.1} := by
    intro W_Y rep hWY_gt hrepV
    obtain ⟨y₁, hy₁, y₂, hy₂, hyne, hmass⟩ :=
      stage1_heavy_pair x₀ C V hVC S Y hsmallN hk256 hk_bound hdvc sepIdx hsepIdx T_tr hT_id hT_dep
        hpair_card P_pair (fun _ => rfl) hP_pair_nn hP_pair_sum W_Y rep hWY_gt hrepV
    refine ⟨y₁, hy₁, y₂, hy₂, hyne, ?_⟩
    have hqeq : rangeMass Q {x : Ω_Q | rep y₁ x.1 ≠ rep y₂ x.1} =
        ∑ τ : Fin k → Ω_pair, prodPMF P_pair τ *
          ((∑ j : Fin (2 ^ D), if padQ τ j ∈ {x : X | rep y₁ x ≠ rep y₂ x} then (1 : ℝ) else 0) /
            (2 ^ D : ℝ)) := by
      have h1 := hQ_mass_eq {x : X | rep y₁ x ≠ rep y₂ x}
      calc rangeMass Q {x : Ω_Q | rep y₁ x.1 ≠ rep y₂ x.1}
          = rangeMass Q {x : Ω_Q | x.1 ∈ {x : X | rep y₁ x ≠ rep y₂ x}} := rfl
        _ = ∑ τ : Fin k → Ω_pair, prodPMF P_pair τ *
              ((∑ j : Fin (2 ^ D), if padQ τ j ∈ {x : X | rep y₁ x ≠ rep y₂ x} then (1 : ℝ) else 0) /
                (2 ^ D : ℝ)) := by
            rw [h1]
            refine Finset.sum_congr rfl fun τ _ => ?_
            congr 1; congr 1
            refine Finset.sum_congr rfl fun j _ => ?_
            by_cases hmem : padQ τ j ∈ {x : X | rep y₁ x ≠ rep y₂ x} <;> simp [hmem]
    rw [hqeq]
    exact hmass
  -- Stage 2: Primal hitting of all heavy disagreement sets D_{f,g} on (Ω_Q, Q)
  let 𝒟 : Set (Set Ω_Q) := { R | ∃ f ∈ C, ∃ g ∈ C, R = {x : Ω_Q | f x.1 ≠ g x.1} }
  have hgrow_D : ∀ E : Finset Ω_Q, E.card ≤ 2 * t →
      ((TraceSet 𝒟 E).card : ℝ) ≤ (2 * (t : ℝ) + 1) ^ (2 * a) :=
    fun E hE => stage2_disagreement_growth ha haic Ω_Q_fin t E hE
  -- Apply Lemma 3 with integer M_grow = ⌊(2t + 1)^(2a)⌋
  let M_D : ℕ := Nat.floor ((2 * (t : ℝ) + 1) ^ (2 * a))
  have hMD_le : (M_D : ℝ) ≤ (2 * (t : ℝ) + 1) ^ (2 * a) := Nat.floor_le (by positivity)
  have hgrow_D_nat : ∀ E : Finset Ω_Q, E.card ≤ 2 * t → (TraceSet 𝒟 E).card ≤ M_D := fun E hE =>
    Nat.le_floor (hgrow_D E hE)
  have hnet2 := finite_epsilon_net_bound hQ_nn hQ_sum 𝒟 (n := t) (M_grow := M_D)
    (by omega) hgrow_D_nat hη_pos htη
  have hmiss_lt1 :
      (∑ u : Fin t → Ω_Q,
        if (∃ R ∈ 𝒟, η ≤ rangeMass Q R ∧ ∀ j : Fin t, u j ∉ R)
        then prodPMF Q u else 0) < 1 := by
    calc _ ≤ 2 * (M_D : ℝ) * Real.exp (-(η * (t : ℝ) / 4)) := hnet2
      _ ≤ 2 * (2 * (t : ℝ) + 1) ^ (2 * a) * Real.exp (-(η * (t : ℝ) / 4)) :=
          mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hMD_le (by norm_num)) (Real.exp_pos _).le
      _ < 1 := ht_bound
  -- Therefore there exists a deterministic u : Fin t → Ω_Q hitting every heavy disagreement set!
  have hex_u : ∃ u : Fin t → Ω_Q, ¬ (∃ R ∈ 𝒟, η ≤ rangeMass Q R ∧ ∀ j : Fin t, u j ∉ R) := by
    by_contra hall; push_neg at hall
    have hsum1 : (∑ u : Fin t → Ω_Q,
        if (∃ R ∈ 𝒟, η ≤ rangeMass Q R ∧ ∀ j : Fin t, u j ∉ R)
        then prodPMF Q u else 0) = 1 := by
      simp_rw [if_pos (hall _)]
      exact prodPMF_sum_one hQ_sum
    linarith
  obtain ⟨u, hu_hits⟩ := hex_u
  let U : Fin t → X := fun j => (u j).1
  refine ⟨U, fun b => ?_⟩
  -- Contradiction if |(V[U = b])_S| > ⌈|V_S| / 2⌉
  by_contra hbig
  push_neg at hbig
  let W_Y : Finset Y := Finset.univ.filter fun y => y.1 ∈ labelings (slice V U b) S
  have hWY_card : W_Y.card = (labelings (slice V U b) S).card := by
    have hsub : labelings (slice V U b) S ⊆ Y := by
      intro y hy
      simp only [labelings, Finset.mem_filter, Finset.mem_univ, true_and, slice, Set.mem_setOf_eq] at hy
      exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, hy.choose, hy.choose_spec.1.1, hy.choose_spec.2⟩
    have e : W_Y ≃ labelings (slice V U b) S :=
      { toFun := fun ⟨y, hy⟩ => ⟨y.1, (Finset.mem_filter.mp hy).2⟩
        invFun := fun ⟨y, hy⟩ => ⟨⟨y, hsub hy⟩, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hy⟩⟩
        left_inv := fun ⟨⟨_, _⟩, _⟩ => rfl
        right_inv := fun ⟨_, _⟩ => rfl }
    rw [← Fintype.card_coe, Fintype.card_congr e, Fintype.card_coe]
  have hWY_gt : Y.card < 2 * W_Y.card := by
    have hYeq : Y.card = (labelings V S).card := rfl
    omega
  -- Pick surviving representative in V[U = b] for each y ∈ W_Y (and arbitrary default elsewhere)
  have hrep_ex : ∀ y : Y, ∃ h : Concept X, y ∈ W_Y → h ∈ slice V U b ∧ restricts S h = y.1 := by
    intro y
    by_cases hy : y ∈ W_Y
    · have hmem := (Finset.mem_filter.mp hy).2
      simp only [labelings, Finset.mem_filter, Finset.mem_univ, true_and] at hmem
      rcases hmem with ⟨h, hh, hrest⟩
      exact ⟨h, fun _ => ⟨hh, hrest⟩⟩
    · exact ⟨fun _ => false, fun h => (hy h).elim⟩
  choose rep hrep using hrep_ex
  obtain ⟨y₁, hy₁, y₂, hy₂, _, hmass_heavy⟩ :=
    hStage1 W_Y rep hWY_gt (fun y hy => ⟨(hrep y hy).1.1, (hrep y hy).2⟩)
  -- Apply hu_hits to R = {x : Ω_Q | rep y₁ x.1 ≠ rep y₂ x.1} ∈ 𝒟!
  apply hu_hits
  refine ⟨{x : Ω_Q | rep y₁ x.1 ≠ rep y₂ x.1},
    ⟨rep y₁, hVC (hrep y₁ hy₁).1.1, rep y₂, hVC (hrep y₂ hy₂).1.1, rfl⟩,
    hmass_heavy, fun j hj => ?_⟩
  simp only [Set.mem_setOf_eq] at hj
  have e1 : rep y₁ (u j).1 = b j := congrFun (hrep y₁ hy₁).1.2 j
  have e2 : rep y₂ (u j).1 = b j := congrFun (hrep y₂ hy₂).1.2 j
  exact hj (e1.trans e2.symm)

end ActiveMQComplexity
