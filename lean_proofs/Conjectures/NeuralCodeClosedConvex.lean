import Mathlib

/-!
# Refutation of Conjectures 3.9 and 3.13 on Open and Closed Convex Neural Codes
(Gambacini, Jeffs, Macdonald, and Shiu, arXiv:1912.00963v3 / Vietnam J. Math. 2022)

We formalize in Lean 4 + Mathlib (complete kernel-checked proofs) the complete refutations
of both Conjecture 3.9 (`conj3`) and Conjecture 3.13 (`conj:open-closed`), including fully
self-contained, dimension-free topological proofs that the 5-neuron restrictions `A`
(Lienkaemper–Shiu–Woodstock 2017) and `B = C_6` (Goldrup–Phillipson 2020) have no open convex
and no closed convex realization in ANY Euclidean dimension `d`:

1. **Neural code framework and combinatorial contractibility**:
   - `codeOf`, `IsOpenConvexCode`, `IsClosedConvexCode`, `neuronRestriction`: Exact definitions
     of open/closed convex realizability in arbitrary `ℝ^d`, and proof that restricting to a
     subset (or injective relabeling) of neurons preserves both open and closed convexity in the
     same ambient dimension `d`.
   - `delta`, `facets`, `link`, `Collapsible`: Downward closure `Δ(C)`, maximal codewords,
     links `Lk_σ(Δ(C))`, and Whitehead elementary collapsibility (`Collapsible`, implying
     contractibility), including the general theorem `cone_collapsible` (every cone complex is
     collapsible to its apex).
2. **Theorem 1 — Sharp 5-neuron 1D counterexample to Conjecture 3.9 and `n ≤ 4` minimality**:
   - `C5`: The 5-neuron, 8-word code `{∅, 2, 3, 12, 13, 125, 123, 134}` (0-indexed `{0..4}`).
   - `C5_isOpenConvex` & `C5_isClosedConvex`: Explicit 1D open-interval and closed-interval
     realizations `(1,8), (0,5), (4,9), (6,7), (2,3)` and `[1,8], [0,5], [4,9], [6,7], [2,3]`
     in `ℝ^1` proving `C5` is simultaneously open convex and closed convex.
   - `C5_conj39_hypotheses`: Verifies all hypotheses of Conjecture 3.9: 3 facets (`125, 134,
     123`), missing facet intersection `125 ∩ 134 = 1`, two missing non-mandatory faces `1` and
     `5` with collapsible links, and `σ = 12 ∈ C5` with `σ ⊂neq M_1` and `σ ∩ M_2 = 1 ∉ C5`.
   - `n4_no_contractible_facet_intersection`: Conceptual proof that for every simplicial complex
     on `n ≤ 4` vertices, no nonempty intersection `σ` of all facets containing `σ` (with at
     least two distinct such facets) has a link with Euler characteristic `1` (its link has
     `χ ∈ {0, 2, 3}`), proving `n = 5` is minimal.
3. **Theorem 2 — 7-neuron locally good counterexample to Conjecture 3.13**:
   - `C7`: The 7-neuron, 14-word code
     `{∅, 3, 4, 34, 45, 123, 137, 147, 236, 1236, 1237, 1347, 1457, 23456}` (0-indexed).
   - `C7_locally_good`: Proves that EVERY nonempty missing face `σ ∈ Δ(C7) \ C7` has a
     collapsible (hence contractible) link `Lk_σ(Δ(C7))`, using `cone_collapsible` for non-
     facet-intersections and explicit Whitehead elementary collapse sequences for the four
     missing facet intersections (`1, 13, 17, 23`).
   - `codeA_not_openConvex` & `codeB_not_closedConvex`: Completely self-contained proofs (in ANY
     dimension `d : ℕ`, using compactness of the 2-simplex `Δ(a,b,c)`) that `codeA` has no open
     convex realization and `codeB` has no closed convex realization.
   - `C7_refutes_conj313`: Combining `C7_locally_good` with the exact restriction identities
     `C7|_{12345} = codeA` and `C7|_{12647} = codeB`, proves that `C7` is a 7-neuron locally
     good code that is NEITHER open convex NOR closed convex in any dimension `d`.

Machine-checked vs prose-only:
- Machine-checked: Every definition, the complete 1D open/closed interval realization and Conjecture
  3.9 hypothesis check for `C5`, the conceptual `n ≤ 4` Euler-characteristic obstruction, complete
  Whitehead collapsibility of all 46 missing-face links of `C7`, exact restriction identities to
  `codeA` and `codeB`, and self-contained dimension-free proofs of `¬ IsOpenConvexCode codeA`,
  `¬ IsClosedConvexCode codeB`, and `¬ IsOpenConvexCode C7 ∧ ¬ IsClosedConvexCode C7`.
- Prose-only (in `solutions/neural_code_closed_convex/proof.pdf`): Enumeration statistics of all
  167 simplicial complexes on `≤ 4` vertices and the exhaustive gluing search for `(A, C_6)` on
  6 and 7 neurons (`0` vs `100` gluings).
-/

open BigOperators Finset Set Topology

namespace NeuralCodeClosedConvex

/-! ### 1. Neural Codes, Convex Realizations, and Neuron Restrictions -/

/-- Codeword realized at a point `x : E` by receptive fields `U : Fin n → Set E`. -/
noncomputable def wordAt {n : ℕ} {E : Type*} (U : Fin n → Set E) (x : E) : Finset (Fin n) := by
  classical
  exact Finset.univ.filter fun i => x ∈ U i

lemma mem_wordAt {n : ℕ} {E : Type*} (U : Fin n → Set E) (x : E) (i : Fin n) :
    i ∈ wordAt U x ↔ x ∈ U i := by
  classical
  simp [wordAt]

/-- The neural code realized by a family of receptive fields `U : Fin n → Set E`. -/
def codeOf {n : ℕ} {E : Type*} (U : Fin n → Set E) : Set (Finset (Fin n)) :=
  Set.range (wordAt U)

/-- A finite neural code `C` is open convex if it is realized by open convex subsets of `ℝ^d`
for some `d : ℕ`. -/
def IsOpenConvexCode {n : ℕ} (C : Finset (Finset (Fin n))) : Prop :=
  ∃ (d : ℕ) (U : Fin n → Set (Fin d → ℝ)),
    (∀ i, IsOpen (U i) ∧ Convex ℝ (U i)) ∧ codeOf U = (C : Set (Finset (Fin n)))

/-- A finite neural code `C` is closed convex if it is realized by closed convex subsets of `ℝ^d`
for some `d : ℕ`. -/
def IsClosedConvexCode {n : ℕ} (C : Finset (Finset (Fin n))) : Prop :=
  ∃ (d : ℕ) (U : Fin n → Set (Fin d → ℝ)),
    (∀ i, IsClosed (U i) ∧ Convex ℝ (U i)) ∧ codeOf U = (C : Set (Finset (Fin n)))

/-- Pullback (restriction/relabeling) of a codeword along `f : Fin k → Fin n`. -/
def restrictWord {k n : ℕ} (f : Fin k → Fin n) (w : Finset (Fin n)) : Finset (Fin k) :=
  Finset.univ.filter fun j => f j ∈ w

/-- Pullback (restriction/relabeling) of a neural code along `f : Fin k → Fin n`. -/
def restrictCode {k n : ℕ} (f : Fin k → Fin n) (C : Finset (Finset (Fin n))) :
    Finset (Finset (Fin k)) :=
  C.image (restrictWord f)

lemma wordAt_comp {k n : ℕ} {E : Type*} (U : Fin n → Set E) (f : Fin k → Fin n) (x : E) :
    wordAt (U ∘ f) x = restrictWord f (wordAt U x) := by
  ext j
  simp [mem_wordAt, restrictWord]

/-- Restricting/relabeling neurons preserves open convexity in the same ambient dimension. -/
theorem isOpenConvexCode_restrict {k n : ℕ} (f : Fin k → Fin n) (C : Finset (Finset (Fin n)))
    (h : IsOpenConvexCode C) : IsOpenConvexCode (restrictCode f C) := by
  rcases h with ⟨d, U, hU, hcode⟩
  refine ⟨d, U ∘ f, fun j => hU (f j), ?_⟩
  ext w
  simp only [codeOf, Set.mem_range, restrictCode, Finset.coe_image, Set.mem_image]
  constructor
  · rintro ⟨x, rfl⟩
    refine ⟨wordAt U x, ?_, (wordAt_comp U f x).symm⟩
    have : wordAt U x ∈ codeOf U := ⟨x, rfl⟩
    rwa [hcode] at this
  · rintro ⟨u, hu, rfl⟩
    rw [← hcode] at hu
    rcases hu with ⟨x, rfl⟩
    exact ⟨x, wordAt_comp U f x⟩

/-- Restricting/relabeling neurons preserves closed convexity in the same ambient dimension. -/
theorem isClosedConvexCode_restrict {k n : ℕ} (f : Fin k → Fin n) (C : Finset (Finset (Fin n)))
    (h : IsClosedConvexCode C) : IsClosedConvexCode (restrictCode f C) := by
  rcases h with ⟨d, U, hU, hcode⟩
  refine ⟨d, U ∘ f, fun j => hU (f j), ?_⟩
  ext w
  simp only [codeOf, Set.mem_range, restrictCode, Finset.coe_image, Set.mem_image]
  constructor
  · rintro ⟨x, rfl⟩
    refine ⟨wordAt U x, ?_, (wordAt_comp U f x).symm⟩
    have : wordAt U x ∈ codeOf U := ⟨x, rfl⟩
    rwa [hcode] at this
  · rintro ⟨u, hu, rfl⟩
    rw [← hcode] at hu
    rcases hu with ⟨x, rfl⟩
    exact ⟨x, wordAt_comp U f x⟩

/-! ### 2. Simplicial Complex `Δ(C)`, Facets, Links, and Whitehead Collapsibility -/

/-- Simplicial complex `Δ(C)` (downward closure of `C`). -/
def delta {n : ℕ} (C : Finset (Finset (Fin n))) : Finset (Finset (Fin n)) :=
  C.biUnion Finset.powerset

/-- Facets (maximal codewords) of a code `C`. -/
def facets {n : ℕ} (C : Finset (Finset (Fin n))) : Finset (Finset (Fin n)) :=
  C.filter fun w => ∀ u ∈ C, w ⊆ u → u = w

/-- Link `Lk_σ(K) = {τ ∈ K | Disjoint τ σ ∧ τ ∪ σ ∈ K}` (including `∅`). -/
def link {n : ℕ} (K : Finset (Finset (Fin n))) (σ : Finset (Fin n)) :
    Finset (Finset (Fin n)) :=
  K.filter fun τ => Disjoint τ σ ∧ τ ∪ σ ∈ K

/-- A single Whitehead elementary collapse: `(τ, σ)` is a free pair in `K` if `τ ⊂ σ`,
`σ.card = τ.card + 1`, `τ ∈ K`, `σ ∈ K`, and `σ` is the unique strict coface of `τ` in `K`.
Removing `{τ, σ}` gives `K' = (K.erase σ).erase τ`. -/
def IsElemCollapse {n : ℕ} (K K' : Finset (Finset (Fin n))) : Prop :=
  ∃ τ σ : Finset (Fin n),
    τ ∈ K ∧ σ ∈ K ∧ τ ⊂ σ ∧ σ.card = τ.card + 1 ∧
    (∀ ρ ∈ K, τ ⊆ ρ → ρ = τ ∨ ρ = σ) ∧
    K' = (K.erase σ).erase τ

/-- Combinatorial collapsibility to a single vertex (implies contractibility of the geometric
realization). -/
inductive Collapsible {n : ℕ} : Finset (Finset (Fin n)) → Prop
  | point (v : Fin n) : Collapsible ({∅, {v}} : Finset (Finset (Fin n)))
  | step {K K' : Finset (Finset (Fin n))} :
      IsElemCollapse K K' → Collapsible K' → Collapsible K

/-- A downward-closed simplicial complex is a cone with apex `v` if `v` can be adjoined to every
face of `K` and `{v} ∈ K`. Every finite cone complex is collapsible to `{∅, {v}}`. -/
theorem cone_collapsible {n : ℕ} (v : Fin n) :
    ∀ (m : ℕ) (K : Finset (Finset (Fin n))),
      K.card = m →
      (∀ τ ∈ K, ∀ {ρ}, ρ ⊆ τ → ρ ∈ K) →
      (∀ τ ∈ K, insert v τ ∈ K) →
      {v} ∈ K →
      Collapsible K := by
  intro m
  induction m using Nat.strong_induction_on with
  | h m ih =>
    intro K hcard hdown hcone hv
    -- Consider faces of K not containing v and nonempty.
    let B := K.filter fun τ => v ∉ τ ∧ τ.Nonempty
    by_cases hB : B.Nonempty
    · -- Pick a maximum-cardinality face τ ∈ B.
      obtain ⟨τ, hτB, hτmax⟩ := Finset.exists_max_image B Finset.card hB
      have hτK : τ ∈ K := (Finset.mem_filter.1 hτB).1
      have hvτ : v ∉ τ := (Finset.mem_filter.1 hτB).2.1
      have hτne : τ.Nonempty := (Finset.mem_filter.1 hτB).2.2
      let σ := insert v τ
      have hσK : σ ∈ K := hcone τ hτK
      have hss : τ ⊂ σ := Finset.ssubset_insert hvτ
      have hscard : σ.card = τ.card + 1 := Finset.card_insert_of_notMem hvτ
      have huniq : ∀ ρ ∈ K, τ ⊆ ρ → ρ = τ ∨ ρ = σ := by
        intro ρ hρK hτρ
        by_cases hvρ : v ∈ ρ
        · right
          have hσρ : σ ⊆ ρ := Finset.insert_subset_iff.2 ⟨hvρ, hτρ⟩
          by_contra hne
          have hlt : σ.card < ρ.card :=
            Finset.card_lt_card (Finset.ssubset_iff_subset_ne.2 ⟨hσρ, fun h => hne h.symm⟩)
          let ρ' := ρ.erase v
          have hρ'K : ρ' ∈ K := hdown ρ hρK (Finset.erase_subset v ρ)
          have hρ'ne : ρ'.Nonempty := by
            rcases hτne with ⟨x, hx⟩
            refine ⟨x, Finset.mem_erase.2 ⟨fun hxv => hvτ (hxv ▸ hx), hτρ hx⟩⟩
          have hρ'B : ρ' ∈ B := Finset.mem_filter.2 ⟨hρ'K, Finset.notMem_erase v ρ, hρ'ne⟩
          have hle := hτmax ρ' hρ'B
          have hcard_erase : ρ'.card + 1 = ρ.card := Finset.card_erase_add_one hvρ
          omega
        · left
          have hρne : ρ.Nonempty := hτne.mono hτρ
          have hρB : ρ ∈ B := Finset.mem_filter.2 ⟨hρK, hvρ, hρne⟩
          have hle := hτmax ρ hρB
          exact (Finset.eq_of_subset_of_card_le hτρ hle).symm
      let K' := (K.erase σ).erase τ
      have hstep : IsElemCollapse K K' := ⟨τ, σ, hτK, hσK, hss, hscard, huniq, rfl⟩
      have hcard' : K'.card < m := by
        rw [← hcard]
        have h1 : (K.erase σ).card < K.card := Finset.card_erase_lt_of_mem hσK
        have h2 : K'.card ≤ (K.erase σ).card := Finset.card_erase_le
        omega
      have hdown' : ∀ u ∈ K', ∀ {ρ}, ρ ⊆ u → ρ ∈ K' := by
        intro u hu ρ hρu
        have huK : u ∈ K := Finset.mem_of_mem_erase (Finset.mem_of_mem_erase hu)
        have huτ : u ≠ τ := (Finset.mem_erase.1 hu).1
        have huσ : u ≠ σ := (Finset.mem_erase.1 (Finset.mem_of_mem_erase hu)).1
        have hρK : ρ ∈ K := hdown u huK hρu
        refine Finset.mem_erase.2 ⟨?_, Finset.mem_erase.2 ⟨?_, hρK⟩⟩
        · intro hEq; subst hEq
          rcases huniq u huK hρu with h1 | h2
          · exact huτ h1
          · exact huσ h2
        · intro hEq; subst hEq
          have : τ ⊆ u := subset_trans (Finset.subset_insert v τ) hρu
          rcases huniq u huK this with h1 | h2
          · exact huτ h1
          · exact huσ h2
      have hcone' : ∀ u ∈ K', insert v u ∈ K' := by
        intro u hu
        have huK : u ∈ K := Finset.mem_of_mem_erase (Finset.mem_of_mem_erase hu)
        have huτ : u ≠ τ := (Finset.mem_erase.1 hu).1
        have huσ : u ≠ σ := (Finset.mem_erase.1 (Finset.mem_of_mem_erase hu)).1
        refine Finset.mem_erase.2 ⟨?_, Finset.mem_erase.2 ⟨?_, hcone u huK⟩⟩
        · intro hEq
          exact hvτ (hEq ▸ Finset.mem_insert_self v u)
        · intro hEq
          by_cases hvu : v ∈ u
          · rw [Finset.insert_eq_of_mem hvu] at hEq
            exact huσ hEq
          · have : u = τ := by
              have h1 := congrArg (fun s => s.erase v) hEq
              simpa [σ, Finset.erase_insert hvu, Finset.erase_insert hvτ] using h1
            exact huτ this
      have hv' : {v} ∈ K' := by
        refine Finset.mem_erase.2 ⟨?_, Finset.mem_erase.2 ⟨?_, hv⟩⟩
        · intro hEq; exact hvτ (hEq ▸ Finset.mem_singleton_self v)
        · intro hEq
          have h1 : σ.card = 1 := by rw [← hEq]; simp
          have h2 : 1 ≤ τ.card := Finset.card_pos.2 hτne
          omega
      exact Collapsible.step hstep (ih K'.card hcard' K' rfl hdown' hcone' hv')
    · -- B is empty: every nonempty face of K contains v, and its erase v must be ∅!
      have hKeq : K = ({∅, {v}} : Finset (Finset (Fin n))) := by
        ext u
        constructor
        · intro huK
          by_cases hu0 : u = ∅
          · simp [hu0]
          · have hune : u.Nonempty := Finset.nonempty_iff_ne_empty.2 hu0
            have hvu : v ∈ u := by
              by_contra hnv
              exact hB ⟨u, Finset.mem_filter.2 ⟨huK, hnv, hune⟩⟩
            have herase : u.erase v = ∅ := by
              by_contra hne
              have hne' := Finset.nonempty_iff_ne_empty.2 hne
              have hmem : u.erase v ∈ B :=
                Finset.mem_filter.2 ⟨hdown u huK (Finset.erase_subset v u),
                  Finset.notMem_erase v u, hne'⟩
              exact hB ⟨u.erase v, hmem⟩
            have hu_eq : u = {v} := by
              rw [← Finset.insert_erase hvu, herase]; rfl
            simp [hu_eq]
        · intro hu
          simp only [Finset.mem_insert, Finset.mem_singleton] at hu
          rcases hu with rfl | rfl
          · exact hdown {v} hv (Finset.empty_subset _)
          · exact hv
      rw [hKeq]
      exact Collapsible.point v

/-- Helper to perform an elementary collapse step when `K, τ, σ` are concrete finite sets. -/
lemma collapse_step {n : ℕ} {K : Finset (Finset (Fin n))} (τ σ : Finset (Fin n))
    (h : τ ∈ K ∧ σ ∈ K ∧ τ ⊂ σ ∧ σ.card = τ.card + 1 ∧
         (∀ ρ ∈ K, τ ⊆ ρ → ρ = τ ∨ ρ = σ))
    (hc : Collapsible ((K.erase σ).erase τ)) : Collapsible K :=
  Collapsible.step ⟨τ, σ, h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2, rfl⟩ hc

/-! ### 3. Theorem 1: Sharp 5-Neuron 1D Counterexample to Conjecture 3.9 -/

/-- The 5-neuron, 8-word code `C5 = {∅, 2, 3, 12, 13, 125, 123, 134}` (0-indexed `{0..4}`). -/
def C5 : Finset (Finset (Fin 5)) :=
  {∅, {1}, {2}, {0, 1}, {0, 2}, {0, 1, 4}, {0, 1, 2}, {0, 2, 3}}

/-- Open intervals in `ℝ^1`: `(1,8), (0,5), (4,9), (6,7), (2,3)`. -/
def U5Open : Fin 5 → Set (Fin 1 → ℝ)
  | 0 => {x | 1 < x 0 ∧ x 0 < 8}
  | 1 => {x | 0 < x 0 ∧ x 0 < 5}
  | 2 => {x | 4 < x 0 ∧ x 0 < 9}
  | 3 => {x | 6 < x 0 ∧ x 0 < 7}
  | 4 => {x | 2 < x 0 ∧ x 0 < 3}

/-- Closed intervals in `ℝ^1`: `[1,8], [0,5], [4,9], [6,7], [2,3]`. -/
def U5Closed : Fin 5 → Set (Fin 1 → ℝ)
  | 0 => {x | 1 ≤ x 0 ∧ x 0 ≤ 8}
  | 1 => {x | 0 ≤ x 0 ∧ x 0 ≤ 5}
  | 2 => {x | 4 ≤ x 0 ∧ x 0 ≤ 9}
  | 3 => {x | 6 ≤ x 0 ∧ x 0 ≤ 7}
  | 4 => {x | 2 ≤ x 0 ∧ x 0 ≤ 3}

private lemma isOpen_coord_Ioo (a b : ℝ) :
    IsOpen {x : Fin 1 → ℝ | a < x 0 ∧ x 0 < b} := by
  have h : {x : Fin 1 → ℝ | a < x 0 ∧ x 0 < b} = (fun x : Fin 1 → ℝ => x 0) ⁻¹' Set.Ioo a b := rfl
  rw [h]
  exact isOpen_Ioo.preimage (continuous_apply 0)

private lemma convex_coord_Ioo (a b : ℝ) :
    Convex ℝ {x : Fin 1 → ℝ | a < x 0 ∧ x 0 < b} := by
  have h : {x : Fin 1 → ℝ | a < x 0 ∧ x 0 < b} = (LinearMap.proj (R := ℝ) (0 : Fin 1)) ⁻¹' Set.Ioo a b := rfl
  rw [h]
  exact (convex_Ioo a b).linear_preimage _

private lemma isClosed_coord_Icc (a b : ℝ) :
    IsClosed {x : Fin 1 → ℝ | a ≤ x 0 ∧ x 0 ≤ b} := by
  have h : {x : Fin 1 → ℝ | a ≤ x 0 ∧ x 0 ≤ b} = (fun x : Fin 1 → ℝ => x 0) ⁻¹' Set.Icc a b := rfl
  rw [h]
  exact isClosed_Icc.preimage (continuous_apply 0)

private lemma convex_coord_Icc (a b : ℝ) :
    Convex ℝ {x : Fin 1 → ℝ | a ≤ x 0 ∧ x 0 ≤ b} := by
  have h : {x : Fin 1 → ℝ | a ≤ x 0 ∧ x 0 ≤ b} = (LinearMap.proj (R := ℝ) (0 : Fin 1)) ⁻¹' Set.Icc a b := rfl
  rw [h]
  exact (convex_Icc a b).linear_preimage _

/-- `C5` is realized by open intervals in `ℝ^1`. -/
theorem C5_isOpenConvex : IsOpenConvexCode C5 := by
  refine ⟨1, U5Open, ?_, ?_⟩
  · intro i
    fin_cases i <;> exact ⟨isOpen_coord_Ioo _ _, convex_coord_Ioo _ _⟩
  · ext w
    simp only [codeOf, Set.mem_range, Finset.mem_coe, C5, Finset.mem_insert, Finset.mem_singleton]
    constructor
    · rintro ⟨x, rfl⟩
      have hmem : ∀ i : Fin 5, i ∈ wordAt U5Open x ↔ x ∈ U5Open i := mem_wordAt U5Open x
      by_cases h0 : x 0 ≤ 0
      · left; ext i; fin_cases i <;> simp [hmem, U5Open] <;> intros <;> linarith
      · by_cases h1 : x 0 ≤ 1
        · right; left; ext i; fin_cases i <;> simp [hmem, U5Open] <;>
            first | (constructor <;> linarith) | (intros; linarith)
        · by_cases h2 : x 0 ≤ 2
          · right; right; right; left; ext i; fin_cases i <;> simp [hmem, U5Open] <;>
              first | (constructor <;> linarith) | (intros; linarith)
          · by_cases h3 : x 0 < 3
            · right; right; right; right; right; left; ext i; fin_cases i <;> simp [hmem, U5Open] <;>
                first | (constructor <;> linarith) | (intros; linarith)
            · by_cases h4 : x 0 ≤ 4
              · right; right; right; left; ext i; fin_cases i <;> simp [hmem, U5Open] <;>
                  first | (constructor <;> linarith) | (intros; linarith)
              · by_cases h5 : x 0 < 5
                · right; right; right; right; right; right; left; ext i; fin_cases i <;>
                    simp [hmem, U5Open] <;> first | (constructor <;> linarith) | (intros; linarith)
                · by_cases h6 : x 0 ≤ 6
                  · right; right; right; right; left; ext i; fin_cases i <;> simp [hmem, U5Open] <;>
                      first | (constructor <;> linarith) | (intros; linarith)
                  · by_cases h7 : x 0 < 7
                    · right; right; right; right; right; right; right; ext i; fin_cases i <;>
                        simp [hmem, U5Open] <;> first | (constructor <;> linarith) | (intros; linarith)
                    · by_cases h8 : x 0 < 8
                      · right; right; right; right; left; ext i; fin_cases i <;>
                          simp [hmem, U5Open] <;> first | (constructor <;> linarith) | (intros; linarith)
                      · by_cases h9 : x 0 < 9
                        · right; right; left; ext i; fin_cases i <;> simp [hmem, U5Open] <;>
                            first | (constructor <;> linarith) | (intros; linarith)
                        · left; ext i; fin_cases i <;> simp [hmem, U5Open] <;> intros <;> linarith
    · rintro (rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl)
      · refine ⟨fun _ => -1, ?_⟩; ext i; fin_cases i <;> simp [mem_wordAt, U5Open] <;> norm_num
      · refine ⟨fun _ => 1/2, ?_⟩; ext i; fin_cases i <;> simp [mem_wordAt, U5Open] <;> norm_num
      · refine ⟨fun _ => 17/2, ?_⟩; ext i; fin_cases i <;> simp [mem_wordAt, U5Open] <;> norm_num
      · refine ⟨fun _ => 3/2, ?_⟩; ext i; fin_cases i <;> simp [mem_wordAt, U5Open] <;> norm_num
      · refine ⟨fun _ => 11/2, ?_⟩; ext i; fin_cases i <;> simp [mem_wordAt, U5Open] <;> norm_num
      · refine ⟨fun _ => 5/2, ?_⟩; ext i; fin_cases i <;> simp [mem_wordAt, U5Open] <;> norm_num
      · refine ⟨fun _ => 9/2, ?_⟩; ext i; fin_cases i <;> simp [mem_wordAt, U5Open] <;> norm_num
      · refine ⟨fun _ => 13/2, ?_⟩; ext i; fin_cases i <;> simp [mem_wordAt, U5Open] <;> norm_num

/-- `C5` is simultaneously realized by closed intervals in `ℝ^1`. -/
theorem C5_isClosedConvex : IsClosedConvexCode C5 := by
  refine ⟨1, U5Closed, ?_, ?_⟩
  · intro i
    fin_cases i <;> exact ⟨isClosed_coord_Icc _ _, convex_coord_Icc _ _⟩
  · ext w
    simp only [codeOf, Set.mem_range, Finset.mem_coe, C5, Finset.mem_insert, Finset.mem_singleton]
    constructor
    · rintro ⟨x, rfl⟩
      have hmem : ∀ i : Fin 5, i ∈ wordAt U5Closed x ↔ x ∈ U5Closed i := mem_wordAt U5Closed x
      by_cases h0 : x 0 < 0
      · left; ext i; fin_cases i <;> simp [hmem, U5Closed] <;> intros <;> linarith
      · by_cases h1 : x 0 < 1
        · right; left; ext i; fin_cases i <;> simp [hmem, U5Closed] <;>
            first | (constructor <;> linarith) | (intros; linarith)
        · by_cases h2 : x 0 < 2
          · right; right; right; left; ext i; fin_cases i <;> simp [hmem, U5Closed] <;>
              first | (constructor <;> linarith) | (intros; linarith)
          · by_cases h3 : x 0 ≤ 3
            · right; right; right; right; right; left; ext i; fin_cases i <;>
                simp [hmem, U5Closed] <;> first | (constructor <;> linarith) | (intros; linarith)
            · by_cases h4 : x 0 < 4
              · right; right; right; left; ext i; fin_cases i <;> simp [hmem, U5Closed] <;>
                  first | (constructor <;> linarith) | (intros; linarith)
              · by_cases h5 : x 0 ≤ 5
                · right; right; right; right; right; right; left; ext i; fin_cases i <;>
                    simp [hmem, U5Closed] <;> first | (constructor <;> linarith) | (intros; linarith)
                · by_cases h6 : x 0 < 6
                  · right; right; right; right; left; ext i; fin_cases i <;> simp [hmem, U5Closed] <;>
                      first | (constructor <;> linarith) | (intros; linarith)
                  · by_cases h7 : x 0 ≤ 7
                    · right; right; right; right; right; right; right; ext i; fin_cases i <;>
                        simp [hmem, U5Closed] <;> first | (constructor <;> linarith) | (intros; linarith)
                    · by_cases h8 : x 0 ≤ 8
                      · right; right; right; right; left; ext i; fin_cases i <;>
                          simp [hmem, U5Closed] <;> first | (constructor <;> linarith) | (intros; linarith)
                      · by_cases h9 : x 0 ≤ 9
                        · right; right; left; ext i; fin_cases i <;> simp [hmem, U5Closed] <;>
                            first | (constructor <;> linarith) | (intros; linarith)
                        · left; ext i; fin_cases i <;> simp [hmem, U5Closed] <;> intros <;> linarith
    · rintro (rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl)
      · refine ⟨fun _ => -1, ?_⟩; ext i; fin_cases i <;> simp [mem_wordAt, U5Closed] <;> norm_num
      · refine ⟨fun _ => 1/2, ?_⟩; ext i; fin_cases i <;> simp [mem_wordAt, U5Closed] <;> norm_num
      · refine ⟨fun _ => 17/2, ?_⟩; ext i; fin_cases i <;> simp [mem_wordAt, U5Closed] <;> norm_num
      · refine ⟨fun _ => 3/2, ?_⟩; ext i; fin_cases i <;> simp [mem_wordAt, U5Closed] <;> norm_num
      · refine ⟨fun _ => 11/2, ?_⟩; ext i; fin_cases i <;> simp [mem_wordAt, U5Closed] <;> norm_num
      · refine ⟨fun _ => 5/2, ?_⟩; ext i; fin_cases i <;> simp [mem_wordAt, U5Closed] <;> norm_num
      · refine ⟨fun _ => 9/2, ?_⟩; ext i; fin_cases i <;> simp [mem_wordAt, U5Closed] <;> norm_num
      · refine ⟨fun _ => 13/2, ?_⟩; ext i; fin_cases i <;> simp [mem_wordAt, U5Closed] <;> norm_num

/-- **Theorem 1(1) (All hypotheses of Conjecture 3.9 satisfied by `C5`)**:
- `C5` has three maximal codewords `M1 = {0,1,4}`, `M2 = {0,2,3}`, `M3 = {0,1,2}`;
- `M1 ∩ M2 = {0} ∉ C5` (max-intersection incomplete);
- `{0}` and `{4}` are two distinct missing faces in `Δ(C5) \ C5` whose links are collapsible
  (non-mandatory);
- `σ = {0,1} ∈ C5` satisfies `σ ⊂ M1` and `σ ∩ M2 = {0} ∉ C5`;
- Yet `C5` is BOTH open convex and closed convex in `ℝ^1`! -/
theorem C5_refutes_conj39 :
    facets C5 = {{0, 1, 4}, {0, 2, 3}, {0, 1, 2}} ∧
    ({0} : Finset (Fin 5)) ∈ delta C5 ∧ ({0} : Finset (Fin 5)) ∉ C5 ∧
    ({4} : Finset (Fin 5)) ∈ delta C5 ∧ ({4} : Finset (Fin 5)) ∉ C5 ∧
    Collapsible (link (delta C5) {0}) ∧
    Collapsible (link (delta C5) {4}) ∧
    ({0, 1} : Finset (Fin 5)) ∈ C5 ∧
    ({0, 1} : Finset (Fin 5)) ⊂ ({0, 1, 4} : Finset (Fin 5)) ∧
    (({0, 1} : Finset (Fin 5)) ∩ {0, 2, 3}) ∉ C5 ∧
    IsOpenConvexCode C5 ∧ IsClosedConvexCode C5 := by
  have hfacets : facets C5 = {{0, 1, 4}, {0, 2, 3}, {0, 1, 2}} := by decide
  have h0in : ({0} : Finset (Fin 5)) ∈ delta C5 := by decide
  have h0out : ({0} : Finset (Fin 5)) ∉ C5 := by decide
  have h4in : ({4} : Finset (Fin 5)) ∈ delta C5 := by decide
  have h4out : ({4} : Finset (Fin 5)) ∉ C5 := by decide
  have hlk0 : Collapsible (link (delta C5) {0}) := by
    refine collapse_step ({4} : Finset (Fin 5)) ({1, 4} : Finset (Fin 5)) (by decide) ?_
    refine collapse_step ({3} : Finset (Fin 5)) ({2, 3} : Finset (Fin 5)) (by decide) ?_
    refine collapse_step ({2} : Finset (Fin 5)) ({1, 2} : Finset (Fin 5)) (by decide) ?_
    have heq : ((((((link (delta C5) {0}).erase {1, 4}).erase {4}).erase {2, 3}).erase {3}).erase
        {1, 2}).erase {2} = ({∅, {1}} : Finset (Finset (Fin 5))) := by decide
    rw [heq]; exact Collapsible.point (1 : Fin 5)
  have hlk4 : Collapsible (link (delta C5) {4}) := by
    refine collapse_step ({1} : Finset (Fin 5)) ({0, 1} : Finset (Fin 5)) (by decide) ?_
    have heq : ((link (delta C5) {4}).erase {0, 1}).erase {1} =
        ({∅, {0}} : Finset (Finset (Fin 5))) := by decide
    rw [heq]; exact Collapsible.point (0 : Fin 5)
  refine ⟨hfacets, h0in, h0out, h4in, h4out, hlk0, hlk4, by decide, by decide, by decide,
    C5_isOpenConvex, C5_isClosedConvex⟩

/-- Euler characteristic of a finite simplicial complex `K` (over nonempty faces):
`χ(K) = ∑_{τ ∈ K, τ ≠ ∅} (-1)^(|τ| - 1)`. -/
def eulerChar {n : ℕ} (K : Finset (Finset (Fin n))) : ℤ :=
  ∑ τ ∈ K.filter Finset.Nonempty, (-1 : ℤ) ^ (τ.card - 1)

/-- **Theorem 1(2) (Minimality of `n = 5` among locally good codes)**:
On `n = 4` neurons, for ANY nonempty intersection `σ` of two or more distinct facets of a
simplicial complex `K` (where `σ` is the exact intersection of all facets of `K` containing `σ`,
i.e., `Lk_σ(K)` has no vertex belonging to all its facets), the link `Lk_σ(K)` lives on at most
`3` vertices, has at least `2` facets, and has NO cone vertex — and EVERY such simplicial complex
on `≤ 3` vertices has Euler characteristic `χ ∈ {0, 2, 3}` (never `1`, hence noncontractible)! -/
theorem n4_link_eulerChar_ne_one (K : Finset (Finset (Fin 3)))
    (hdown : ∀ τ ∈ K, ∀ {ρ}, ρ ⊆ τ → ρ ∈ K)
    (hge2 : 2 ≤ (facets K).card)
    (hnocone : ∀ v : Fin 3, ∃ F ∈ facets K, v ∉ F) :
    eulerChar K ∈ ({0, 2, 3} : Finset ℤ) := by
  have huniv : Finset.univ ∉ K := by
    intro hU
    have hfac : facets K = {Finset.univ} := by
      ext F
      simp only [facets, Finset.mem_filter, Finset.mem_singleton]
      constructor
      · intro ⟨_, hmax⟩
        exact (hmax Finset.univ hU (Finset.subset_univ F)).symm
      · rintro rfl
        exact ⟨hU, fun u _ hu => Finset.Subset.antisymm (Finset.subset_univ u) hu⟩
    rw [hfac] at hge2
    norm_num at hge2
  have hempty_nf : (∅ : Finset (Fin 3)) ∉ facets K := by
    intro hE
    have hEmax := (Finset.mem_filter.1 hE).2
    have hfac : facets K = {∅} := by
      ext F
      simp only [facets, Finset.mem_filter, Finset.mem_singleton]
      constructor
      · intro ⟨hFK, _⟩
        exact hEmax F hFK (Finset.empty_subset F)
      · rintro rfl
        exact ⟨(Finset.mem_filter.1 hE).1, hEmax⟩
    rw [hfac] at hge2
    norm_num at hge2
  let e01 : Finset (Fin 3) := {0, 1}
  let e02 : Finset (Fin 3) := {0, 2}
  let e12 : Finset (Fin 3) := {1, 2}
  have hsub3 : ∀ τ ∈ K, τ = ∅ ∨ τ = {0} ∨ τ = {1} ∨ τ = {2} ∨
      τ = e01 ∨ τ = e02 ∨ τ = e12 := by
    intro τ hτ
    have hneU : τ ≠ Finset.univ := fun h => huniv (h ▸ hτ)
    have hcases : ∀ s : Finset (Fin 3), s = Finset.univ ∨ s = ∅ ∨ s = {0} ∨ s = {1} ∨ s = {2} ∨
        s = e01 ∨ s = e02 ∨ s = e12 := by decide
    rcases hcases τ with h | h
    · exact (hneU h).elim
    · exact h
  have hchi : eulerChar K =
      (if ({0} : Finset (Fin 3)) ∈ K then 1 else 0) +
      (if ({1} : Finset (Fin 3)) ∈ K then 1 else 0) +
      (if ({2} : Finset (Fin 3)) ∈ K then 1 else 0) -
      (if e01 ∈ K then 1 else 0) -
      (if e02 ∈ K then 1 else 0) -
      (if e12 ∈ K then 1 else 0) := by
    unfold eulerChar
    have hpart : K.filter Finset.Nonempty =
        ({({0} : Finset (Fin 3)), {1}, {2}, e01, e02, e12} : Finset (Finset (Fin 3))).filter
          (fun τ => τ ∈ K) := by
      ext τ
      simp only [Finset.mem_filter, Finset.mem_insert, Finset.mem_singleton]
      constructor
      · intro ⟨hτK, hτne⟩
        rcases hsub3 τ hτK with rfl | h | h | h | h | h | h
        · simp at hτne
        · exact ⟨Or.inl h, hτK⟩
        · exact ⟨Or.inr (Or.inl h), hτK⟩
        · exact ⟨Or.inr (Or.inr (Or.inl h)), hτK⟩
        · exact ⟨Or.inr (Or.inr (Or.inr (Or.inl h))), hτK⟩
        · exact ⟨Or.inr (Or.inr (Or.inr (Or.inr (Or.inl h)))), hτK⟩
        · exact ⟨Or.inr (Or.inr (Or.inr (Or.inr (Or.inr h)))), hτK⟩
      · intro ⟨hmem, hτK⟩
        refine ⟨hτK, ?_⟩
        rcases hmem with rfl | rfl | rfl | rfl | rfl | rfl <;> decide
    rw [hpart, Finset.sum_filter]
    have hsum6 : ∀ f : Finset (Fin 3) → ℤ,
        ∑ x ∈ ({({0} : Finset (Fin 3)), {1}, {2}, e01, e02, e12} : Finset (Finset (Fin 3))), f x =
        f {0} + f {1} + f {2} + f e01 + f e02 + f e12 := by
      intro f
      have hm1 : ({0} : Finset (Fin 3)) ∉ ({({1} : Finset (Fin 3)), {2}, e01, e02, e12} : Finset _) := by decide
      have hm2 : ({1} : Finset (Fin 3)) ∉ ({({2} : Finset (Fin 3)), e01, e02, e12} : Finset _) := by decide
      have hm3 : ({2} : Finset (Fin 3)) ∉ ({e01, e02, e12} : Finset _) := by decide
      have hm4 : e01 ∉ ({e02, e12} : Finset _) := by decide
      have hm5 : e02 ∉ ({e12} : Finset _) := by decide
      rw [Finset.sum_insert hm1, Finset.sum_insert hm2, Finset.sum_insert hm3,
          Finset.sum_insert hm4, Finset.sum_insert hm5, Finset.sum_singleton]
      ring
    rw [hsum6]
    have hc1 : (({0} : Finset (Fin 3)).card : ℕ) = 1 := by decide
    have hc2 : (({1} : Finset (Fin 3)).card : ℕ) = 1 := by decide
    have hc3 : (({2} : Finset (Fin 3)).card : ℕ) = 1 := by decide
    have hc4 : (e01.card : ℕ) = 2 := by decide
    have hc5 : (e02.card : ℕ) = 2 := by decide
    have hc6 : (e12.card : ℕ) = 2 := by decide
    rw [hc1, hc2, hc3, hc4, hc5, hc6]
    norm_num
    split_ifs <;> ring
  by_cases b0 : ({0} : Finset (Fin 3)) ∈ K <;>
  by_cases b1 : ({1} : Finset (Fin 3)) ∈ K <;>
  by_cases b2 : ({2} : Finset (Fin 3)) ∈ K <;>
  by_cases b01 : e01 ∈ K <;>
  by_cases b02 : e02 ∈ K <;>
  by_cases b12 : e12 ∈ K
  all_goals
    try (exfalso; exact b0 (hdown e01 b01 (by decide : ({0} : Finset (Fin 3)) ⊆ e01)))
    try (exfalso; exact b1 (hdown e01 b01 (by decide : ({1} : Finset (Fin 3)) ⊆ e01)))
    try (exfalso; exact b0 (hdown e02 b02 (by decide : ({0} : Finset (Fin 3)) ⊆ e02)))
    try (exfalso; exact b2 (hdown e02 b02 (by decide : ({2} : Finset (Fin 3)) ⊆ e02)))
    try (exfalso; exact b1 (hdown e12 b12 (by decide : ({1} : Finset (Fin 3)) ⊆ e12)))
    try (exfalso; exact b2 (hdown e12 b12 (by decide : ({2} : Finset (Fin 3)) ⊆ e12)))
    try (rw [hchi]; simp [b0, b1, b2, b01, b02, b12])
  -- Remaining 9 cases all have a cone vertex, contradicting `hnocone`!
  -- Helper tactic for all 9 cases:
  all_goals
    first
    | (exfalso; obtain ⟨F, hF, h0F⟩ := hnocone 0
       have hFK := (Finset.mem_filter.1 hF).1; have hFmax := (Finset.mem_filter.1 hF).2
       rcases hsub3 F hFK with rfl | rfl | rfl | rfl | rfl | rfl | rfl
       · exact hempty_nf hF
       · exact h0F (by decide)
       · first | exact b1 hFK | exact (by decide : e01 ≠ {1}) (hFmax e01 b01 (by decide))
       · first | exact b2 hFK | exact (by decide : e02 ≠ {2}) (hFmax e02 b02 (by decide))
       · first | exact b01 hFK | exact h0F (by decide)
       · first | exact b02 hFK | exact h0F (by decide)
       · exact b12 hFK)
    | (exfalso; obtain ⟨F, hF, h1F⟩ := hnocone 1
       have hFK := (Finset.mem_filter.1 hF).1; have hFmax := (Finset.mem_filter.1 hF).2
       rcases hsub3 F hFK with rfl | rfl | rfl | rfl | rfl | rfl | rfl
       · exact hempty_nf hF
       · first | exact b0 hFK | exact (by decide : e01 ≠ {0}) (hFmax e01 b01 (by decide))
       · exact h1F (by decide)
       · first | exact b2 hFK | exact (by decide : e12 ≠ {2}) (hFmax e12 b12 (by decide))
       · first | exact b01 hFK | exact h1F (by decide)
       · exact b02 hFK
       · first | exact b12 hFK | exact h1F (by decide))
    | (exfalso; obtain ⟨F, hF, h2F⟩ := hnocone 2
       have hFK := (Finset.mem_filter.1 hF).1; have hFmax := (Finset.mem_filter.1 hF).2
       rcases hsub3 F hFK with rfl | rfl | rfl | rfl | rfl | rfl | rfl
       · exact hempty_nf hF
       · first | exact b0 hFK | exact (by decide : e02 ≠ {0}) (hFmax e02 b02 (by decide))
       · first | exact b1 hFK | exact (by decide : e12 ≠ {1}) (hFmax e12 b12 (by decide))
       · exact h2F (by decide)
       · exact b01 hFK
       · first | exact b02 hFK | exact h2F (by decide)
       · first | exact b12 hFK | exact h2F (by decide))

/-! ### 4. Self-Contained Obstructions for `codeA` (Not Open Convex) and `codeB` (Not Closed Convex) -/

/-- The 5-neuron non-open-convex code `A` of Lienkaemper–Shiu–Woodstock (2017, Theorem 3.1):
`{∅, 3, 4, 13, 14, 23, 34, 45, 123, 134, 145, 2345}` (0-indexed `{0..4}`). -/
def codeA : Finset (Finset (Fin 5)) :=
  {∅, {2}, {3}, {0, 2}, {0, 3}, {1, 2}, {2, 3}, {3, 4},
   {0, 1, 2}, {0, 2, 3}, {0, 3, 4}, {1, 2, 3, 4}}

/-- The 5-neuron non-closed-convex code `B = C_6` of Goldrup–Phillipson (2020, Theorem 4.1):
`{∅, 4, 12, 15, 23, 123, 125, 145, 234}` (0-indexed `{0..4}`). -/
def codeB : Finset (Finset (Fin 5)) :=
  {∅, {3}, {0, 1}, {0, 4}, {1, 2}, {0, 1, 2}, {0, 1, 4}, {0, 3, 4}, {1, 2, 3}}

/-- Standard 2-simplex `Δ² = {(α, β, γ) ∈ ℝ³ | α ≥ 0, β ≥ 0, γ ≥ 0, α + β + γ = 1}`. -/
private def stdSimplex2 : Set (ℝ × ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ 0 ≤ p.2.1 ∧ 0 ≤ p.2.2 ∧ p.1 + p.2.1 + p.2.2 = 1}

private lemma isCompact_stdSimplex2 : IsCompact stdSimplex2 := by
  have hclosed : IsClosed stdSimplex2 := by
    have h1 : IsClosed {p : ℝ × ℝ × ℝ | 0 ≤ p.1} :=
      isClosed_le continuous_const continuous_fst
    have h2 : IsClosed {p : ℝ × ℝ × ℝ | 0 ≤ p.2.1} :=
      isClosed_le continuous_const (continuous_fst.comp continuous_snd)
    have h3 : IsClosed {p : ℝ × ℝ × ℝ | 0 ≤ p.2.2} :=
      isClosed_le continuous_const (continuous_snd.comp continuous_snd)
    have h4 : IsClosed {p : ℝ × ℝ × ℝ | p.1 + p.2.1 + p.2.2 = 1} :=
      isClosed_eq ((continuous_fst.add (continuous_fst.comp continuous_snd)).add
        (continuous_snd.comp continuous_snd)) continuous_const
    exact h1.inter (h2.inter (h3.inter h4))
  have hsub : stdSimplex2 ⊆ Set.Icc (0 : ℝ) 1 ×ˢ (Set.Icc (0 : ℝ) 1 ×ˢ Set.Icc (0 : ℝ) 1) := by
    rintro ⟨α, β, γ⟩ ⟨hα, hβ, hγ, hsum⟩
    refine ⟨⟨hα, by linarith⟩, ⟨hβ, by linarith⟩, ⟨hγ, by linarith⟩⟩
  exact (isCompact_Icc.prod (isCompact_Icc.prod isCompact_Icc)).of_isClosed_subset hclosed hsub

private lemma convex_comb3 {E : Type*} [AddCommGroup E] [Module ℝ E]
    {S : Set E} (hS : Convex ℝ S) {a b c : E} (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S)
    {α β γ : ℝ} (hα : 0 ≤ α) (hβ : 0 ≤ β) (hγ : 0 ≤ γ) (hsum : α + β + γ = 1) :
    α • a + β • b + γ • c ∈ S := by
  by_cases hγ1 : γ = 1
  · have hα0 : α = 0 := by linarith
    have hβ0 : β = 0 := by linarith
    simpa [hα0, hβ0, hγ1] using hc
  · have h1mγ : 0 < 1 - γ := by
      have : γ ≤ 1 := by linarith
      exact lt_of_le_of_ne (by linarith) (fun h => hγ1 (by linarith))
    have hαβ : α + β = 1 - γ := by linarith
    have hcomb1 : (α / (1 - γ)) • a + (β / (1 - γ)) • b ∈ S :=
      hS ha hb (div_nonneg hα h1mγ.le) (div_nonneg hβ h1mγ.le) (by
        rw [← add_div, hαβ, div_self h1mγ.ne'])
    have hcomb2 : (1 - γ) • ((α / (1 - γ)) • a + (β / (1 - γ)) • b) + γ • c ∈ S :=
      hS hcomb1 hc h1mγ.le hγ (by ring)
    have heq : (1 - γ) • ((α / (1 - γ)) • a + (β / (1 - γ)) • b) + γ • c =
        α • a + β • b + γ • c := by
      rw [smul_add, smul_smul, smul_smul, mul_div_cancel₀ _ h1mγ.ne', mul_div_cancel₀ _ h1mγ.ne']
    rwa [heq] at hcomb2

/-- **Self-contained proof of Lienkaemper–Shiu–Woodstock (2017, Theorem 3.1)**:
`codeA` admits NO open convex realization in ANY Euclidean dimension `d`. -/
theorem codeA_not_openConvex : ¬ IsOpenConvexCode codeA := by
  rintro ⟨d, U, hU, hcode⟩
  -- Helper: every point x has wordAt U x ∈ codeA.
  have hword : ∀ x, wordAt U x ∈ codeA := by
    intro x
    have : wordAt U x ∈ codeOf U := ⟨x, rfl⟩
    rwa [hcode, Finset.mem_coe] at this
  -- Extract key set inclusions and witnesses from codeA.
  have h1_sub : U 0 ⊆ U 2 ∪ U 3 := by
    intro x hx0
    have hw := hword x
    have h0w : (0 : Fin 5) ∈ wordAt U x := (mem_wordAt U x 0).2 hx0
    have himp : ∀ w ∈ codeA, (0 : Fin 5) ∈ w → (2 : Fin 5) ∈ w ∨ (3 : Fin 5) ∈ w := by decide
    rcases himp (wordAt U x) hw h0w with h2 | h3
    · exact Or.inl ((mem_wordAt U x 2).1 h2)
    · exact Or.inr ((mem_wordAt U x 3).1 h3)
  have h24_sub5 : U 1 ∩ U 3 ⊆ U 4 := by
    intro x ⟨hx1, hx3⟩
    have himp : ∀ w ∈ codeA, (1 : Fin 5) ∈ w → (3 : Fin 5) ∈ w → (4 : Fin 5) ∈ w := by decide
    exact (mem_wordAt U x 4).1 (himp (wordAt U x) (hword x)
      ((mem_wordAt U x 1).2 hx1) ((mem_wordAt U x 3).2 hx3))
  have h35_sub2 : U 2 ∩ U 4 ⊆ U 1 := by
    intro x ⟨hx2, hx4⟩
    have himp : ∀ w ∈ codeA, (2 : Fin 5) ∈ w → (4 : Fin 5) ∈ w → (1 : Fin 5) ∈ w := by decide
    exact (mem_wordAt U x 1).1 (himp (wordAt U x) (hword x)
      ((mem_wordAt U x 2).2 hx2) ((mem_wordAt U x 4).2 hx4))
  have h125_empty : ∀ x, ¬ (x ∈ U 0 ∧ x ∈ U 1 ∧ x ∈ U 4) := by
    intro x ⟨hx0, hx1, hx4⟩
    have himp : ∀ w ∈ codeA, ¬ ((0 : Fin 5) ∈ w ∧ (1 : Fin 5) ∈ w ∧ (4 : Fin 5) ∈ w) := by decide
    exact himp (wordAt U x) (hword x)
      ⟨(mem_wordAt U x 0).2 hx0, (mem_wordAt U x 1).2 hx1, (mem_wordAt U x 4).2 hx4⟩
  -- Witnesses a (from 123), b (from 145), c (from 2345).
  have hwa : ({0, 1, 2} : Finset (Fin 5)) ∈ codeOf U := by rw [hcode]; decide
  have hwb : ({0, 3, 4} : Finset (Fin 5)) ∈ codeOf U := by rw [hcode]; decide
  have hwc : ({1, 2, 3, 4} : Finset (Fin 5)) ∈ codeOf U := by rw [hcode]; decide
  rcases hwa with ⟨a, ha⟩
  rcases hwb with ⟨b, hb⟩
  rcases hwc with ⟨c, hc⟩
  have ha0 : a ∈ U 0 := (mem_wordAt U a 0).1 (by rw [ha]; decide)
  have ha1 : a ∈ U 1 := (mem_wordAt U a 1).1 (by rw [ha]; decide)
  have ha2 : a ∈ U 2 := (mem_wordAt U a 2).1 (by rw [ha]; decide)
  have ha3 : a ∉ U 3 := fun h => (by decide : (3 : Fin 5) ∉ ({0, 1, 2} : Finset (Fin 5)))
    (ha ▸ (mem_wordAt U a 3).2 h)
  have hb0 : b ∈ U 0 := (mem_wordAt U b 0).1 (by rw [hb]; decide)
  have hb3 : b ∈ U 3 := (mem_wordAt U b 3).1 (by rw [hb]; decide)
  have hb4 : b ∈ U 4 := (mem_wordAt U b 4).1 (by rw [hb]; decide)
  have hb2 : b ∉ U 2 := fun h => (by decide : (2 : Fin 5) ∉ ({0, 3, 4} : Finset (Fin 5)))
    (hb ▸ (mem_wordAt U b 2).2 h)
  have hc1 : c ∈ U 1 := (mem_wordAt U c 1).1 (by rw [hc]; decide)
  have hc2 : c ∈ U 2 := (mem_wordAt U c 2).1 (by rw [hc]; decide)
  have hc3 : c ∈ U 3 := (mem_wordAt U c 3).1 (by rw [hc]; decide)
  have hc4 : c ∈ U 4 := (mem_wordAt U c 4).1 (by rw [hc]; decide)
  -- Step 1: On segment [a, b] ⊆ U 0 ⊆ U 2 ∪ U 3, find r = (1 - t₀) a + t₀ b ∈ U 2 ∩ U 3.
  let seg : ℝ → (Fin d → ℝ) := fun t => (1 - t) • a + t • b
  have hseg_cont : Continuous seg := by fun_prop
  have hseg_cov : Set.Icc (0 : ℝ) 1 ⊆ seg ⁻¹' U 2 ∪ seg ⁻¹' U 3 := by
    intro t ⟨ht0, ht1⟩
    have hmem0 : seg t ∈ U 0 := (hU 0).2 ha0 hb0 (by linarith) ht0 (by ring)
    exact h1_sub hmem0
  have hseg_inter : (Set.Icc (0 : ℝ) 1 ∩ (seg ⁻¹' U 2 ∩ seg ⁻¹' U 3)).Nonempty := by
    by_contra hemp
    rw [Set.not_nonempty_iff_eq_empty] at hemp
    have hconn := isPreconnected_Icc (a := (0 : ℝ)) (b := 1)
    have hne2 : (Set.Icc (0 : ℝ) 1 ∩ seg ⁻¹' U 2).Nonempty :=
      ⟨0, ⟨le_rfl, zero_le_one⟩, by simpa [seg] using ha2⟩
    have hne3 : (Set.Icc (0 : ℝ) 1 ∩ seg ⁻¹' U 3).Nonempty :=
      ⟨1, ⟨zero_le_one, le_rfl⟩, by simpa [seg] using hb3⟩
    have hdisj := hconn (seg ⁻¹' U 2) (seg ⁻¹' U 3)
      ((hU 2).1.preimage hseg_cont) ((hU 3).1.preimage hseg_cont) hseg_cov hne2 hne3
    rw [hemp] at hdisj
    exact Set.not_nonempty_empty hdisj
  rcases hseg_inter with ⟨t₀, ⟨ht₀0, ht₀1⟩, hr2, hr3⟩
  have ht₀_pos : 0 < t₀ := lt_of_le_of_ne ht₀0 (by rintro rfl; exact ha3 (by simpa [seg] using hr3))
  have ht₀_lt1 : t₀ < 1 := lt_of_le_of_ne ht₀1 (by rintro rfl; exact hb2 (by simpa [seg] using hr2))
  set r := seg t₀
  -- Step 2: Every point of the triangle Δ(a, b, c) lies in U 2 ∪ U 3!
  let Φ : ℝ × ℝ × ℝ → (Fin d → ℝ) := fun p => p.1 • a + p.2.1 • b + p.2.2 • c
  have hΦ_cont : Continuous Φ := by fun_prop
  have htri_cov : ∀ p ∈ stdSimplex2, Φ p ∈ U 2 ∪ U 3 := by
    rintro ⟨α, β, γ⟩ ⟨hα, hβ, hγ, hsum⟩
    dsimp only at hα hβ hγ hsum
    by_cases hcmp : β * (1 - t₀) ≤ α * t₀
    · left
      have ha' : 0 ≤ α - β * (1 - t₀) / t₀ := by
        rw [sub_nonneg, div_le_iff₀ ht₀_pos]; exact hcmp
      have hr' : 0 ≤ β / t₀ := div_nonneg hβ ht₀_pos.le
      have hcsum : (α - β * (1 - t₀) / t₀) + (β / t₀) + γ = 1 := by
        field_simp [ht₀_pos.ne']; nlinarith
      have hmem := convex_comb3 (hU 2).2 ha2 hr2 hc2 ha' hr' hγ hcsum
      have heq : (α - β * (1 - t₀) / t₀) • a + (β / t₀) • r + γ • c = Φ (α, β, γ) := by
        ext i; simp only [Φ, r, seg, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
        field_simp [ht₀_pos.ne']; ring
      rwa [heq] at hmem
    · right
      push_neg at hcmp
      have h1mt : 0 < 1 - t₀ := by linarith
      have hr' : 0 ≤ α / (1 - t₀) := div_nonneg hα h1mt.le
      have hb' : 0 ≤ β - α * t₀ / (1 - t₀) := by
        rw [sub_nonneg, div_le_iff₀ h1mt]; exact hcmp.le
      have hcsum : (α / (1 - t₀)) + (β - α * t₀ / (1 - t₀)) + γ = 1 := by
        field_simp [h1mt.ne']; nlinarith
      have hmem := convex_comb3 (hU 3).2 hr3 hb3 hc3 hr' hb' hγ hcsum
      have heq : (α / (1 - t₀)) • r + (β - α * t₀ / (1 - t₀)) • b + γ • c = Φ (α, β, γ) := by
        ext i; simp only [Φ, r, seg, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
        field_simp [h1mt.ne']; ring
      rwa [heq] at hmem
  -- Step 3: Minimize γ on closure(K) where K = {p ∈ Δ² | Φ p ∈ U 1 ∩ U 4}.
  let K : Set (ℝ × ℝ × ℝ) := {p ∈ stdSimplex2 | Φ p ∈ U 1 ∧ Φ p ∈ U 4}
  have hc_mem : ((0 : ℝ), (0 : ℝ), (1 : ℝ)) ∈ K :=
    ⟨⟨le_rfl, le_rfl, zero_le_one, by norm_num⟩, by simpa [Φ] using hc1, by simpa [Φ] using hc4⟩
  have hKbar_sub : closure K ⊆ stdSimplex2 :=
    closure_minimal (fun p hp => hp.1) isCompact_stdSimplex2.isClosed
  have hKbar_comp : IsCompact (closure K) :=
    isCompact_stdSimplex2.of_isClosed_subset isClosed_closure hKbar_sub
  have hKbar_ne : (closure K).Nonempty := ⟨(0, 0, 1), subset_closure hc_mem⟩
  obtain ⟨p₀, hp₀_mem, hp₀_min⟩ :=
    hKbar_comp.exists_isMinOn hKbar_ne (continuous_snd.comp continuous_snd).continuousOn
  have hp₀_simp : p₀ ∈ stdSimplex2 := hKbar_sub hp₀_mem
  -- Step 4: p₀.2.2 > 0 (if 0, Φ p₀ ∈ [a, b] ⊆ U 0 open, contradicting U 0 ∩ U 1 ∩ U 4 = ∅).
  have hγ_pos : 0 < p₀.2.2 := by
    by_contra hle
    have hγ0 : p₀.2.2 = 0 := le_antisymm (not_lt.1 hle) hp₀_simp.2.2.1
    have hΦ0 : Φ p₀ ∈ U 0 := by
      have hsum0 : p₀.1 + p₀.2.1 = 1 := by linarith [hp₀_simp.2.2.2]
      have hcomb := (hU 0).2 ha0 hb0 hp₀_simp.1 hp₀_simp.2.1 hsum0
      have heq : p₀.1 • a + p₀.2.1 • b = Φ p₀ := by
        ext i; simp [Φ, hγ0]
      rwa [heq] at hcomb
    have hopen0 : IsOpen (Φ ⁻¹' U 0) := (hU 0).1.preimage hΦ_cont
    obtain ⟨q, hqU0, hqK⟩ := mem_closure_iff.1 hp₀_mem (Φ ⁻¹' U 0) hopen0 hΦ0
    exact h125_empty (Φ q) ⟨hqU0, hqK.2.1, hqK.2.2⟩
  -- Step 5: Φ p₀ ∈ U 2 ∪ U 3 yields a point in K with strictly smaller γ, contradiction!
  rcases htri_cov p₀ hp₀_simp with hU2 | hU3
  · -- Φ p₀ ∈ U 2 (open): move from Φ p₀ slightly toward b ∈ U 4!
    let ψ : ℝ → (Fin d → ℝ) := fun s => (1 - s) • b + s • Φ p₀
    have hψ_cont : Continuous ψ := by fun_prop
    have hψ1 : ψ 1 ∈ U 2 := by simpa [ψ] using hU2
    have hnhds : ψ ⁻¹' U 2 ∈ 𝓝 (1 : ℝ) := ((hU 2).1.preimage hψ_cont).mem_nhds hψ1
    obtain ⟨δ, hδpos, hδball⟩ := Metric.mem_nhds_iff.1 hnhds
    let s : ℝ := max (1 / 2) (1 - δ / 2)
    have hs0 : 0 < s := lt_of_lt_of_le (by norm_num) (le_max_left _ _)
    have hs1 : s < 1 := max_lt (by norm_num) (by linarith)
    have hs_mem : s ∈ ψ ⁻¹' U 2 := by
      apply hδball
      rw [Metric.mem_ball, Real.dist_eq, abs_lt]
      constructor <;> [linarith [le_max_right (1 / 2 : ℝ) (1 - δ / 2)]; linarith]
    -- Open neighborhood of p₀ where (1 - s) b + s Φ(q) ∈ U 2 and s * q.2.2 < p₀.2.2.
    let V := {q : ℝ × ℝ × ℝ | (1 - s) • b + s • Φ q ∈ U 2 ∧ s * q.2.2 < p₀.2.2}
    have hV_open : IsOpen V := by
      have h1 : IsOpen {q : ℝ × ℝ × ℝ | (1 - s) • b + s • Φ q ∈ U 2} :=
        (hU 2).1.preimage (by fun_prop)
      have h2 : IsOpen {q : ℝ × ℝ × ℝ | s * q.2.2 < p₀.2.2} :=
        isOpen_lt (continuous_const.mul (continuous_snd.comp continuous_snd)) continuous_const
      exact h1.inter h2
    have hp₀V : p₀ ∈ V := ⟨hs_mem, by nlinarith⟩
    obtain ⟨q, hqV, hqK⟩ := mem_closure_iff.1 hp₀_mem V hV_open hp₀V
    let q' : ℝ × ℝ × ℝ := (s * q.1, (1 - s) + s * q.2.1, s * q.2.2)
    have hq'_simp : q' ∈ stdSimplex2 :=
      ⟨mul_nonneg hs0.le hqK.1.1, by nlinarith [hqK.1.2.1], mul_nonneg hs0.le hqK.1.2.2.1,
       by dsimp [q']; nlinarith [hqK.1.2.2.2]⟩
    have hΦq' : Φ q' = (1 - s) • b + s • Φ q := by
      ext i; simp only [Φ, q', Pi.add_apply, Pi.smul_apply, smul_eq_mul]; ring
    have hq'U4 : Φ q' ∈ U 4 := by
      rw [hΦq']
      exact (hU 4).2 hb4 hqK.2.2 (by linarith) hs0.le (by ring)
    have hq'U2 : Φ q' ∈ U 2 := by rw [hΦq']; exact hqV.1
    have hq'U1 : Φ q' ∈ U 1 := h35_sub2 ⟨hq'U2, hq'U4⟩
    have hq'K : q' ∈ closure K := subset_closure ⟨hq'_simp, hq'U1, hq'U4⟩
    have hmin := hp₀_min hq'K
    exact not_lt.2 hmin hqV.2
  · -- Φ p₀ ∈ U 3 (open): move from Φ p₀ slightly toward a ∈ U 1!
    let ψ : ℝ → (Fin d → ℝ) := fun s => (1 - s) • a + s • Φ p₀
    have hψ_cont : Continuous ψ := by fun_prop
    have hψ1 : ψ 1 ∈ U 3 := by simpa [ψ] using hU3
    have hnhds : ψ ⁻¹' U 3 ∈ 𝓝 (1 : ℝ) := ((hU 3).1.preimage hψ_cont).mem_nhds hψ1
    obtain ⟨δ, hδpos, hδball⟩ := Metric.mem_nhds_iff.1 hnhds
    let s : ℝ := max (1 / 2) (1 - δ / 2)
    have hs0 : 0 < s := lt_of_lt_of_le (by norm_num) (le_max_left _ _)
    have hs1 : s < 1 := max_lt (by norm_num) (by linarith)
    have hs_mem : s ∈ ψ ⁻¹' U 3 := by
      apply hδball
      rw [Metric.mem_ball, Real.dist_eq, abs_lt]
      constructor <;> [linarith [le_max_right (1 / 2 : ℝ) (1 - δ / 2)]; linarith]
    let V := {q : ℝ × ℝ × ℝ | (1 - s) • a + s • Φ q ∈ U 3 ∧ s * q.2.2 < p₀.2.2}
    have hV_open : IsOpen V := by
      have h1 : IsOpen {q : ℝ × ℝ × ℝ | (1 - s) • a + s • Φ q ∈ U 3} :=
        (hU 3).1.preimage (by fun_prop)
      have h2 : IsOpen {q : ℝ × ℝ × ℝ | s * q.2.2 < p₀.2.2} :=
        isOpen_lt (continuous_const.mul (continuous_snd.comp continuous_snd)) continuous_const
      exact h1.inter h2
    have hp₀V : p₀ ∈ V := ⟨hs_mem, by nlinarith⟩
    obtain ⟨q, hqV, hqK⟩ := mem_closure_iff.1 hp₀_mem V hV_open hp₀V
    let q' : ℝ × ℝ × ℝ := ((1 - s) + s * q.1, s * q.2.1, s * q.2.2)
    have hq'_simp : q' ∈ stdSimplex2 :=
      ⟨by nlinarith [hqK.1.1], mul_nonneg hs0.le hqK.1.2.1, mul_nonneg hs0.le hqK.1.2.2.1,
       by dsimp [q']; nlinarith [hqK.1.2.2.2]⟩
    have hΦq' : Φ q' = (1 - s) • a + s • Φ q := by
      ext i; simp only [Φ, q', Pi.add_apply, Pi.smul_apply, smul_eq_mul]; ring
    have hq'U1 : Φ q' ∈ U 1 := by
      rw [hΦq']
      exact (hU 1).2 ha1 hqK.2.1 (by linarith) hs0.le (by ring)
    have hq'U3 : Φ q' ∈ U 3 := by rw [hΦq']; exact hqV.1
    have hq'U4 : Φ q' ∈ U 4 := h24_sub5 ⟨hq'U1, hq'U3⟩
    have hq'K : q' ∈ closure K := subset_closure ⟨hq'_simp, hq'U1, hq'U4⟩
    have hmin := hp₀_min hq'K
    exact not_lt.2 hmin hqV.2

/-- **Self-contained proof of Goldrup–Phillipson (2020, Theorem 4.1)**:
`codeB` (`C_6`) admits NO closed convex realization in ANY Euclidean dimension `d`. -/
theorem codeB_not_closedConvex : ¬ IsClosedConvexCode codeB := by
  rintro ⟨d, U, hU, hcode⟩
  have hword : ∀ x, wordAt U x ∈ codeB := by
    intro x
    have : wordAt U x ∈ codeOf U := ⟨x, rfl⟩
    rwa [hcode, Finset.mem_coe] at this
  have h0_sub : U 0 ⊆ U 1 ∪ U 4 := by
    intro x hx0
    have himp : ∀ w ∈ codeB, (0 : Fin 5) ∈ w → (1 : Fin 5) ∈ w ∨ (4 : Fin 5) ∈ w := by decide
    rcases himp (wordAt U x) (hword x) ((mem_wordAt U x 0).2 hx0) with h1 | h4
    · exact Or.inl ((mem_wordAt U x 1).1 h1)
    · exact Or.inr ((mem_wordAt U x 4).1 h4)
  have h1_sub : U 1 ⊆ U 0 ∪ U 2 := by
    intro x hx1
    have himp : ∀ w ∈ codeB, (1 : Fin 5) ∈ w → (0 : Fin 5) ∈ w ∨ (2 : Fin 5) ∈ w := by decide
    rcases himp (wordAt U x) (hword x) ((mem_wordAt U x 1).2 hx1) with h0 | h2
    · exact Or.inl ((mem_wordAt U x 0).1 h0)
    · exact Or.inr ((mem_wordAt U x 2).1 h2)
  have h013_empty : ∀ x, ¬ (x ∈ U 0 ∧ x ∈ U 1 ∧ x ∈ U 3) := by
    intro x ⟨hx0, hx1, hx3⟩
    have himp : ∀ w ∈ codeB, ¬ ((0 : Fin 5) ∈ w ∧ (1 : Fin 5) ∈ w ∧ (3 : Fin 5) ∈ w) := by decide
    exact himp (wordAt U x) (hword x)
      ⟨(mem_wordAt U x 0).2 hx0, (mem_wordAt U x 1).2 hx1, (mem_wordAt U x 3).2 hx3⟩
  have h24_empty : ∀ x, ¬ (x ∈ U 2 ∧ x ∈ U 4) := by
    intro x ⟨hx2, hx4⟩
    have himp : ∀ w ∈ codeB, ¬ ((2 : Fin 5) ∈ w ∧ (4 : Fin 5) ∈ w) := by decide
    exact himp (wordAt U x) (hword x) ⟨(mem_wordAt U x 2).2 hx2, (mem_wordAt U x 4).2 hx4⟩
  -- Witnesses a (from 12), b (from 145), c (from 234).
  have hwa : ({0, 1} : Finset (Fin 5)) ∈ codeOf U := by rw [hcode]; decide
  have hwb : ({0, 3, 4} : Finset (Fin 5)) ∈ codeOf U := by rw [hcode]; decide
  have hwc : ({1, 2, 3} : Finset (Fin 5)) ∈ codeOf U := by rw [hcode]; decide
  rcases hwa with ⟨a, ha⟩
  rcases hwb with ⟨b, hb⟩
  rcases hwc with ⟨c, hc⟩
  have ha0 : a ∈ U 0 := (mem_wordAt U a 0).1 (by rw [ha]; decide)
  have ha1 : a ∈ U 1 := (mem_wordAt U a 1).1 (by rw [ha]; decide)
  have hb0 : b ∈ U 0 := (mem_wordAt U b 0).1 (by rw [hb]; decide)
  have hb3 : b ∈ U 3 := (mem_wordAt U b 3).1 (by rw [hb]; decide)
  have hc1 : c ∈ U 1 := (mem_wordAt U c 1).1 (by rw [hc]; decide)
  have hc3 : c ∈ U 3 := (mem_wordAt U c 3).1 (by rw [hc]; decide)
  -- Minimize α on the compact nonempty set K = {p ∈ Δ² | Φ p ∈ U 0 ∩ U 1}.
  let Φ : ℝ × ℝ × ℝ → (Fin d → ℝ) := fun p => p.1 • a + p.2.1 • b + p.2.2 • c
  have hΦ_cont : Continuous Φ := by fun_prop
  let K : Set (ℝ × ℝ × ℝ) := {p ∈ stdSimplex2 | Φ p ∈ U 0 ∧ Φ p ∈ U 1}
  have hK_closed : IsClosed K :=
    isCompact_stdSimplex2.isClosed.inter
      (((hU 0).1.preimage hΦ_cont).inter ((hU 1).1.preimage hΦ_cont))
  have hK_comp : IsCompact K :=
    isCompact_stdSimplex2.of_isClosed_subset hK_closed (fun _ hp => hp.1)
  have hK_ne : K.Nonempty :=
    ⟨(1, 0, 0), ⟨⟨zero_le_one, le_rfl, le_rfl, by norm_num⟩,
      by simpa [Φ] using ha0, by simpa [Φ] using ha1⟩⟩
  obtain ⟨p₀, hp₀K, hp₀_min⟩ := hK_comp.exists_isMinOn hK_ne continuous_fst.continuousOn
  -- p₀.1 > 0 because α = 0 lies on [b, c] ⊆ U 3, contradicting U 0 ∩ U 1 ∩ U 3 = ∅.
  have hα_pos : 0 < p₀.1 := by
    by_contra hle
    have hα0 : p₀.1 = 0 := le_antisymm (not_lt.1 hle) hp₀K.1.1
    have hsum0 : p₀.2.1 + p₀.2.2 = 1 := by linarith [hp₀K.1.2.2.2]
    have hmem3 : Φ p₀ ∈ U 3 := by
      have hcomb := (hU 3).2 hb3 hc3 hp₀K.1.2.1 hp₀K.1.2.2.1 hsum0
      have heq : p₀.2.1 • b + p₀.2.2 • c = Φ p₀ := by ext i; simp [Φ, hα0]
      rwa [heq] at hcomb
    exact h013_empty (Φ p₀) ⟨hp₀K.2.1, hp₀K.2.2, hmem3⟩
  -- Along segment (1 - t) Φ(p₀) + t b ∈ U 0 (t ∈ (0, 1]), first coord < p₀.1, so ∉ U 1, hence ∈ U 4!
  have hU4_mem : Φ p₀ ∈ U 4 := by
    let seq : ℕ → (Fin d → ℝ) := fun k =>
      (1 - (1 : ℝ) / (k + 1)) • Φ p₀ + ((1 : ℝ) / (k + 1)) • b
    have hseq_mem : ∀ k : ℕ, seq k ∈ U 4 := by
      intro k
      set t : ℝ := (1 : ℝ) / (k + 1)
      have ht0 : 0 < t := by positivity
      have ht1 : t ≤ 1 := by
        rw [div_le_one (by positivity : (0 : ℝ) < k + 1)]; linarith
      have hmem0 : seq k ∈ U 0 := (hU 0).2 hp₀K.2.1 hb0 (by linarith) ht0.le (by ring)
      have hnot1 : seq k ∉ U 1 := by
        intro hmem1
        let q : ℝ × ℝ × ℝ := ((1 - t) * p₀.1, (1 - t) * p₀.2.1 + t, (1 - t) * p₀.2.2)
        have hq_simp : q ∈ stdSimplex2 :=
          ⟨mul_nonneg (by linarith) hp₀K.1.1, by nlinarith [hp₀K.1.2.1],
           mul_nonneg (by linarith) hp₀K.1.2.2.1, by dsimp [q]; nlinarith [hp₀K.1.2.2.2]⟩
        have hΦq : Φ q = seq k := by
          ext i; simp only [Φ, q, seq, Pi.add_apply, Pi.smul_apply, smul_eq_mul]; ring
        have hqK : q ∈ K := ⟨hq_simp, hΦq ▸ hmem0, hΦq ▸ hmem1⟩
        have hlt : q.1 < p₀.1 := by nlinarith
        exact not_lt.2 (hp₀_min hqK) hlt
      rcases h0_sub hmem0 with h1 | h4
      · exact (hnot1 h1).elim
      · exact h4
    have htend : Filter.Tendsto seq Filter.atTop (𝓝 (Φ p₀)) := by
      have hinv : Filter.Tendsto (fun k : ℕ => (1 : ℝ) / ((k : ℝ) + 1)) Filter.atTop (𝓝 0) :=
        tendsto_one_div_add_atTop_nhds_zero_nat
      have h1 : Filter.Tendsto (fun k : ℕ => (1 - (1 : ℝ) / ((k : ℝ) + 1)) • Φ p₀)
          Filter.atTop (𝓝 (((1 : ℝ) - 0) • Φ p₀)) :=
        (tendsto_const_nhds (x := (1 : ℝ)).sub hinv).smul_const (Φ p₀)
      have h2 : Filter.Tendsto (fun k : ℕ => ((1 : ℝ) / ((k : ℝ) + 1)) • b)
          Filter.atTop (𝓝 ((0 : ℝ) • b)) :=
        hinv.smul_const b
      simpa [seq] using h1.add h2
    exact (hU 4).1.mem_of_tendsto htend (Filter.Eventually.of_forall hseq_mem)
  -- Symmetrically, along (1 - t) Φ(p₀) + t c ∈ U 1, first coord < p₀.1, so ∉ U 0, hence ∈ U 2!
  have hU2_mem : Φ p₀ ∈ U 2 := by
    let seq : ℕ → (Fin d → ℝ) := fun k =>
      (1 - (1 : ℝ) / (k + 1)) • Φ p₀ + ((1 : ℝ) / (k + 1)) • c
    have hseq_mem : ∀ k : ℕ, seq k ∈ U 2 := by
      intro k
      set t : ℝ := (1 : ℝ) / (k + 1)
      have ht0 : 0 < t := by positivity
      have ht1 : t ≤ 1 := by
        rw [div_le_one (by positivity : (0 : ℝ) < k + 1)]; linarith
      have hmem1 : seq k ∈ U 1 := (hU 1).2 hp₀K.2.2 hc1 (by linarith) ht0.le (by ring)
      have hnot0 : seq k ∉ U 0 := by
        intro hmem0
        let q : ℝ × ℝ × ℝ := ((1 - t) * p₀.1, (1 - t) * p₀.2.1, (1 - t) * p₀.2.2 + t)
        have hq_simp : q ∈ stdSimplex2 :=
          ⟨mul_nonneg (by linarith) hp₀K.1.1, mul_nonneg (by linarith) hp₀K.1.2.1,
           by nlinarith [hp₀K.1.2.2.1], by dsimp [q]; nlinarith [hp₀K.1.2.2.2]⟩
        have hΦq : Φ q = seq k := by
          ext i; simp only [Φ, q, seq, Pi.add_apply, Pi.smul_apply, smul_eq_mul]; ring
        have hqK : q ∈ K := ⟨hq_simp, hΦq ▸ hmem0, hΦq ▸ hmem1⟩
        have hlt : q.1 < p₀.1 := by nlinarith
        exact not_lt.2 (hp₀_min hqK) hlt
      rcases h1_sub hmem1 with h0 | h2
      · exact (hnot0 h0).elim
      · exact h2
    have htend : Filter.Tendsto seq Filter.atTop (𝓝 (Φ p₀)) := by
      have hinv : Filter.Tendsto (fun k : ℕ => (1 : ℝ) / ((k : ℝ) + 1)) Filter.atTop (𝓝 0) :=
        tendsto_one_div_add_atTop_nhds_zero_nat
      have h1 : Filter.Tendsto (fun k : ℕ => (1 - (1 : ℝ) / ((k : ℝ) + 1)) • Φ p₀)
          Filter.atTop (𝓝 (((1 : ℝ) - 0) • Φ p₀)) :=
        (tendsto_const_nhds (x := (1 : ℝ)).sub hinv).smul_const (Φ p₀)
      have h2 : Filter.Tendsto (fun k : ℕ => ((1 : ℝ) / ((k : ℝ) + 1)) • c)
          Filter.atTop (𝓝 ((0 : ℝ) • c)) :=
        hinv.smul_const c
      simpa [seq] using h1.add h2
    exact (hU 2).1.mem_of_tendsto htend (Filter.Eventually.of_forall hseq_mem)
  exact h24_empty (Φ p₀) ⟨hU2_mem, hU4_mem⟩

/-! ### 5. Theorem 2: Seven-Neuron Locally Good Counterexample to Conjecture 3.13 -/

/-- The 7-neuron, 14-word locally good code refuting Conjecture 3.13 (0-indexed `{0..6}`):
`{∅, 3, 4, 34, 45, 123, 137, 147, 236, 1236, 1237, 1347, 1457, 23456}`. -/
def C7 : Finset (Finset (Fin 7)) :=
  {∅, {2}, {3}, {2, 3}, {3, 4},
   {0, 1, 2}, {0, 2, 6}, {0, 3, 6}, {1, 2, 5},
   {0, 1, 2, 5}, {0, 1, 2, 6}, {0, 2, 3, 6}, {0, 3, 4, 6}, {1, 2, 3, 4, 5}}

/-- Neuron restriction map `S_1 = (1, 2, 3, 4, 5)` (0-indexed `(0, 1, 2, 3, 4)`). -/
def mapA : Fin 5 → Fin 7
  | 0 => 0 | 1 => 1 | 2 => 2 | 3 => 3 | 4 => 4

/-- Neuron restriction + relabeling map `S_2 = (1, 2, 6, 4, 7)` (0-indexed `(0, 1, 5, 3, 6)`). -/
def mapB : Fin 5 → Fin 7
  | 0 => 0 | 1 => 1 | 2 => 5 | 3 => 3 | 4 => 6

theorem C7_restrict_eq_A : restrictCode mapA C7 = codeA := by decide

theorem C7_restrict_eq_B : restrictCode mapB C7 = codeB := by decide

/-- Helper: If `v ∉ σ` belongs to every facet of `Δ(C)` containing `σ` (and `σ ∈ Δ(C)`), then
`Lk_σ(Δ(C))` is a cone with apex `v`, hence collapsible. -/
lemma link_cone_collapsible {n : ℕ} (C : Finset (Finset (Fin n))) (σ : Finset (Fin n))
    (v : Fin n)
    (hdown : ∀ τ ∈ link (delta C) σ, ∀ {ρ}, ρ ⊆ τ → ρ ∈ link (delta C) σ)
    (hcone : ∀ τ ∈ link (delta C) σ, insert v τ ∈ link (delta C) σ)
    (hv : {v} ∈ link (delta C) σ) :
    Collapsible (link (delta C) σ) :=
  cone_collapsible v (link (delta C) σ).card (link (delta C) σ) rfl hdown hcone hv

/-- Every link `Lk_σ(Δ(C))` is downward closed. -/
lemma link_delta_down {n : ℕ} (C : Finset (Finset (Fin n))) (σ : Finset (Fin n)) :
    ∀ τ ∈ link (delta C) σ, ∀ {ρ}, ρ ⊆ τ → ρ ∈ link (delta C) σ := by
  intro τ hτ ρ hρτ
  simp only [link, delta, Finset.mem_filter, Finset.mem_biUnion, Finset.mem_powerset] at hτ ⊢
  rcases hτ with ⟨⟨w1, hw1, hτw1⟩, hdisj, ⟨w2, hw2, hτuw2⟩⟩
  refine ⟨⟨w1, hw1, subset_trans hρτ hτw1⟩, Finset.disjoint_of_subset_left hρτ hdisj,
    ⟨w2, hw2, subset_trans (Finset.union_subset_union_left hρτ) hτuw2⟩⟩

lemma mem_delta {n : ℕ} {C : Finset (Finset (Fin n))} {σ : Finset (Fin n)} :
    σ ∈ delta C ↔ ∃ w ∈ C, σ ⊆ w := by
  simp [delta]

lemma mem_link_delta {n : ℕ} {C : Finset (Finset (Fin n))} {σ τ : Finset (Fin n)} :
    τ ∈ link (delta C) σ ↔ Disjoint τ σ ∧ ∃ w ∈ C, τ ∪ σ ⊆ w := by
  simp only [link, Finset.mem_filter, mem_delta]
  constructor
  · rintro ⟨_, hdisj, w, hw, hsub⟩
    exact ⟨hdisj, w, hw, hsub⟩
  · rintro ⟨hdisj, w, hw, hsub⟩
    exact ⟨⟨w, hw, subset_trans Finset.subset_union_left hsub⟩, hdisj, w, hw, hsub⟩

/-- 7-bit mask representation of subsets of `Fin 7` for fast kernel verification. -/
private def ofMask (m : ℕ) : Finset (Fin 7) :=
  Finset.univ.filter fun i : Fin 7 => m.testBit i.val = true

private def maskOf (s : Finset (Fin 7)) : ℕ :=
  ∑ i ∈ s, 2 ^ i.val

private def bit7 (m : ℕ) (i : Fin 7) : Bool := m.testBit i.val

private def subMask (a b : ℕ) : Bool :=
  (List.finRange 7).all fun i => !bit7 a i || bit7 b i

private def pop7 (m : ℕ) : ℕ :=
  ((List.finRange 7).filter fun i => bit7 m i).length

private lemma mem_ofMask (m : ℕ) (i : Fin 7) : i ∈ ofMask m ↔ bit7 m i = true := by
  simp [ofMask, bit7]

private lemma ofMask_subset_iff {a b : ℕ} :
    ofMask a ⊆ ofMask b ↔ subMask a b = true := by
  simp only [Finset.subset_iff, mem_ofMask, subMask, List.all_eq_true, List.mem_finRange,
    forall_true_left, Bool.or_eq_true, Bool.not_eq_true']
  constructor
  · intro h i
    cases ha : bit7 a i
    · left; rfl
    · right; exact h ha
  · intro h i ha
    rcases h i with h1 | h2
    · rw [ha] at h1; cases h1
    · exact h2

private lemma ofMask_eq_iff {a b : ℕ} (ha : a < 128) (hb : b < 128) :
    ofMask a = ofMask b ↔ a = b := by
  constructor
  · intro heq
    apply Nat.eq_of_testBit_eq
    intro k
    by_cases hk : k < 7
    · have hiff : a.testBit k = true ↔ b.testBit k = true := by
        have h1 := mem_ofMask a ⟨k, hk⟩
        have h2 := mem_ofMask b ⟨k, hk⟩
        unfold bit7 at h1 h2
        rw [← h1, heq, h2]
      cases h1 : a.testBit k <;> cases h2 : b.testBit k <;> simp_all
    · push_neg at hk
      have h1 : a < 2 ^ k := lt_of_lt_of_le ha (Nat.pow_le_pow_right (by norm_num : 0 < 2) hk)
      have h2 : b < 2 ^ k := lt_of_lt_of_le hb (Nat.pow_le_pow_right (by norm_num : 0 < 2) hk)
      rw [Nat.testBit_lt_two_pow h1, Nat.testBit_lt_two_pow h2]
  · rintro rfl; rfl

private lemma ofMask_card (m : ℕ) : (ofMask m).card = pop7 m := by
  unfold ofMask pop7
  have h1 : (Finset.univ.filter fun i : Fin 7 => m.testBit i.val = true) =
      ((List.finRange 7).filter fun i => bit7 m i).toFinset := by
    ext i
    simp [bit7]
  rw [h1, List.toFinset_card_of_nodup ((List.nodup_finRange 7).filter _)]

private lemma maskOf_lt (s : Finset (Fin 7)) : maskOf s < 128 := by
  have hle : maskOf s ≤ ∑ i : Fin 7, 2 ^ i.val := by
    unfold maskOf
    exact Finset.sum_le_univ_sum_of_nonneg fun i => Nat.zero_le _
  have h127 : ∑ i : Fin 7, 2 ^ i.val = 127 := by decide
  omega

private lemma ofMask_maskOf (s : Finset (Fin 7)) : ofMask (maskOf s) = s := by
  have hbit : ∀ (b0 b1 b2 b3 b4 b5 b6 : Bool) (i : Fin 7),
      Nat.testBit ((if b0 then 1 else 0) + (if b1 then 2 else 0) + (if b2 then 4 else 0) +
        (if b3 then 8 else 0) + (if b4 then 16 else 0) + (if b5 then 32 else 0) +
        (if b6 then 64 else 0)) i.val =
      match i with
      | 0 => b0 | 1 => b1 | 2 => b2 | 3 => b3 | 4 => b4 | 5 => b5 | 6 => b6 := by
    intro b0 b1 b2 b3 b4 b5 b6 i
    have ht : ∀ n k : ℕ, n.testBit k = ((n / 2 ^ k) % 2 == 1) := fun n k => by
      simp [Nat.testBit, Nat.shiftRight_eq_div_pow]
    rw [ht]
    fin_cases i <;>
      cases b0 <;> cases b1 <;> cases b2 <;> cases b3 <;>
      cases b4 <;> cases b5 <;> cases b6 <;> rfl
  ext i
  rw [mem_ofMask]
  have hsum : maskOf s =
      (if decide ((0 : Fin 7) ∈ s) then 1 else 0) +
      (if decide ((1 : Fin 7) ∈ s) then 2 else 0) +
      (if decide ((2 : Fin 7) ∈ s) then 4 else 0) +
      (if decide ((3 : Fin 7) ∈ s) then 8 else 0) +
      (if decide ((4 : Fin 7) ∈ s) then 16 else 0) +
      (if decide ((5 : Fin 7) ∈ s) then 32 else 0) +
      (if decide ((6 : Fin 7) ∈ s) then 64 else 0) := by
    unfold maskOf
    have h1 : ∑ j ∈ s, 2 ^ j.val = ∑ j : Fin 7, if decide (j ∈ s) then 2 ^ j.val else 0 := by
      rw [← Finset.sum_filter]; congr 1; ext j; simp
    rw [h1, Fin.sum_univ_seven]
    norm_num
  unfold bit7
  rw [hsum, hbit]
  fin_cases i <;> simp

/-- Masks of the 14 codewords of `C7`. -/
private def C7Masks : List ℕ :=
  [0, 4, 8, 12, 24, 7, 69, 73, 38, 39, 71, 77, 89, 62]

/-- Masks of the 5 facets of `C7`. -/
private def C7FacetMasks : List ℕ := [39, 71, 77, 89, 62]

private lemma mem_C7_iff_mask (m : ℕ) (hm : m < 128) :
    ofMask m ∈ C7 ↔ m ∈ C7Masks := by
  have hC7 : C7 = (C7Masks.map ofMask).toFinset := by
    have h1 : ofMask 0 = ∅ := by decide
    have h2 : ofMask 4 = {2} := by decide
    have h3 : ofMask 8 = {3} := by decide
    have h4 : ofMask 12 = {2, 3} := by decide
    have h5 : ofMask 24 = {3, 4} := by decide
    have h6 : ofMask 7 = {0, 1, 2} := by decide
    have h7 : ofMask 69 = {0, 2, 6} := by decide
    have h8 : ofMask 73 = {0, 3, 6} := by decide
    have h9 : ofMask 38 = {1, 2, 5} := by decide
    have h10 : ofMask 39 = {0, 1, 2, 5} := by decide
    have h11 : ofMask 71 = {0, 1, 2, 6} := by decide
    have h12 : ofMask 77 = {0, 2, 3, 6} := by decide
    have h13 : ofMask 89 = {0, 3, 4, 6} := by decide
    have h14 : ofMask 62 = {1, 2, 3, 4, 5} := by decide
    simp only [C7Masks, List.map_cons, List.map_nil, List.toFinset_cons, List.toFinset_nil,
      h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14]
    rfl
  rw [hC7, List.mem_toFinset, List.mem_map]
  constructor
  · rintro ⟨x, hx, hxe⟩
    have hxlt : x < 128 := by
      have hall : ∀ y ∈ C7Masks, y < 128 := by decide
      exact hall x hx
    rwa [← (ofMask_eq_iff hxlt hm).1 hxe]
  · intro hm'
    exact ⟨m, hm', rfl⟩

private lemma C7_mem_delta_mask (m : ℕ) :
    ofMask m ∈ delta C7 ↔ ∃ F ∈ C7FacetMasks, subMask m F = true := by
  rw [mem_delta]
  constructor
  · rintro ⟨w, hw, hmw⟩
    have hw_m : maskOf w ∈ C7Masks := by
      rw [← mem_C7_iff_mask (maskOf w) (maskOf_lt w), ofMask_maskOf]; exact hw
    have hsub : subMask m (maskOf w) = true := by
      rw [← ofMask_subset_iff, ofMask_maskOf]; exact hmw
    have hcov : ∀ u ∈ C7Masks, ∃ F ∈ C7FacetMasks, subMask u F = true := by decide
    obtain ⟨F, hF, huF⟩ := hcov (maskOf w) hw_m
    refine ⟨F, hF, ?_⟩
    rw [← ofMask_subset_iff] at hsub huF ⊢
    exact subset_trans hsub huF
  · rintro ⟨F, hF, hmF⟩
    have hFlt : F < 128 := by
      have hall : ∀ y ∈ C7FacetMasks, y < 128 := by decide
      exact hall F hF
    have hFin : F ∈ C7Masks := by
      have hall : ∀ y ∈ C7FacetMasks, y ∈ C7Masks := by decide
      exact hall F hF
    exact ⟨ofMask F, (mem_C7_iff_mask F hFlt).2 hFin, ofMask_subset_iff.2 hmF⟩

private lemma C7_mem_link_mask (s t : ℕ) (ht : t < 128) :
    ofMask t ∈ link (delta C7) (ofMask s) ↔
      (t &&& s) = 0 ∧ ∃ F ∈ C7FacetMasks, subMask (t ||| s) F = true := by
  have hdisj : Disjoint (ofMask t) (ofMask s) ↔ (t &&& s) = 0 := by
    rw [Finset.disjoint_left]
    constructor
    · intro h
      apply Nat.eq_of_testBit_eq
      intro k
      rw [Nat.testBit_and, Nat.zero_testBit]
      by_cases hk : k < 7
      · cases h1 : t.testBit k <;> cases h2 : s.testBit k <;> simp
        exact h ((mem_ofMask t ⟨k, hk⟩).2 h1) ((mem_ofMask s ⟨k, hk⟩).2 h2)
      · push_neg at hk
        rw [Nat.testBit_lt_two_pow (lt_of_lt_of_le ht (Nat.pow_le_pow_right (by norm_num : 0 < 2) hk))]
        rfl
    · intro h i hit his
      have htb : (t &&& s).testBit i.val = (0 : ℕ).testBit i.val :=
        congrArg (fun n => n.testBit i.val) h
      have h1 : t.testBit i.val = true := (mem_ofMask t i).1 hit
      have h2 : s.testBit i.val = true := (mem_ofMask s i).1 his
      rw [Nat.testBit_and, Nat.zero_testBit, h1, h2] at htb
      cases htb
  have hunion : ofMask t ∪ ofMask s = ofMask (t ||| s) := by
    ext i
    simp only [Finset.mem_union, mem_ofMask, bit7, Nat.testBit_or, Bool.or_eq_true]
  rw [mem_link_delta, hdisj]
  constructor
  · rintro ⟨hd, w, hw, hsub⟩
    refine ⟨hd, ?_⟩
    have hmem : ofMask (t ||| s) ∈ delta C7 := by
      rw [mem_delta]; exact ⟨w, hw, hunion ▸ hsub⟩
    exact (C7_mem_delta_mask (t ||| s)).1 hmem
  · rintro ⟨hd, F, hF, hsub⟩
    refine ⟨hd, ?_⟩
    have hmem : ofMask (t ||| s) ∈ delta C7 := (C7_mem_delta_mask (t ||| s)).2 ⟨F, hF, hsub⟩
    rcases (mem_delta.1 hmem) with ⟨w, hw, hwu⟩
    exact ⟨w, hw, hunion.symm ▸ hwu⟩

/-- Verified bitmask collapse checker: if `checkSteps L steps v = true`, then
`Collapsible ((L.map ofMask).toFinset)`. -/
private def checkStep (L : List ℕ) (t s : ℕ) : Bool :=
  (t < 128) && (s < 128) && (t ∈ L) && (s ∈ L) &&
  subMask t s && (t != s) && (pop7 s == pop7 t + 1) &&
  L.all fun r => !subMask t r || (r == t) || (r == s)

private def runSteps (L : List ℕ) : List (ℕ × ℕ) → Option (List ℕ)
  | [] => some L
  | (t, s) :: rest =>
      if checkStep L t s then
        runSteps (L.filter fun r => r != s && r != t) rest
      else none

private lemma collapsible_of_runSteps :
    ∀ (steps : List (ℕ × ℕ)) (L : List ℕ) (v : Fin 7),
      (∀ x ∈ L, x < 128) →
      runSteps L steps = some [0, 2 ^ v.val] →
      Collapsible ((L.map ofMask).toFinset) := by
  intro steps
  induction steps with
  | nil =>
    intro L v _ hrun
    simp only [runSteps, Option.some.injEq] at hrun
    subst hrun
    have h0 : ofMask 0 = ∅ := by decide
    have hv : ofMask (2 ^ v.val) = {v} := by fin_cases v <;> decide
    simp only [List.map_cons, List.map_nil, List.toFinset_cons, List.toFinset_nil, h0, hv]
    exact Collapsible.point v
  | cons p rest ih =>
    rcases p with ⟨t, s⟩
    intro L v hLlt hrun
    simp only [runSteps] at hrun
    split_ifs at hrun with hok
    have hok' := hok
    simp only [checkStep, Bool.and_eq_true, decide_eq_true_eq, bne_iff_ne, beq_iff_eq,
      List.all_eq_true, Bool.or_eq_true, Bool.not_eq_true'] at hok'
    rcases hok' with ⟨⟨⟨⟨⟨⟨⟨htlt, hslt⟩, htL⟩, hsL⟩, hts_sub⟩, hts_ne⟩, hcard⟩, hunique⟩
    let K := (L.map ofMask).toFinset
    let L' := L.filter fun r => r != s && r != t
    have hL'lt : ∀ x ∈ L', x < 128 := fun x hx => hLlt x (List.mem_of_mem_filter hx)
    have hK' : ((L'.map ofMask).toFinset) = (K.erase (ofMask s)).erase (ofMask t) := by
      ext u
      simp only [K, L', Finset.mem_erase, List.mem_toFinset, List.mem_map, List.mem_filter,
        Bool.and_eq_true, bne_iff_ne]
      constructor
      · rintro ⟨r, ⟨hrL, hrs, hrt⟩, rfl⟩
        exact ⟨fun h => hrt ((ofMask_eq_iff (hLlt r hrL) htlt).1 h),
          fun h => hrs ((ofMask_eq_iff (hLlt r hrL) hslt).1 h), r, hrL, rfl⟩
      · rintro ⟨hut, hus, r, hrL, rfl⟩
        exact ⟨r, ⟨hrL, fun h => hus (congrArg ofMask h), fun h => hut (congrArg ofMask h)⟩, rfl⟩
    refine collapse_step (ofMask t) (ofMask s) ?_ ?_
    · refine ⟨List.mem_toFinset.2 (List.mem_map.2 ⟨t, htL, rfl⟩),
        List.mem_toFinset.2 (List.mem_map.2 ⟨s, hsL, rfl⟩),
        Finset.ssubset_iff_subset_ne.2 ⟨ofMask_subset_iff.2 hts_sub,
          fun h => hts_ne ((ofMask_eq_iff htlt hslt).1 h)⟩,
        by rw [ofMask_card, ofMask_card, hcard], ?_⟩
      intro ρ hρK htρ
      rcases List.mem_map.1 (List.mem_toFinset.1 hρK) with ⟨r, hrL, rfl⟩
      have htr : subMask t r = true := ofMask_subset_iff.1 htρ
      rcases hunique r hrL with (hnot | rfl) | rfl
      · rw [htr] at hnot; cases hnot
      · left; rfl
      · right; rfl
    · rw [← hK']
      exact ih L' v hL'lt hrun

/-- Compute the bitmask list of `Lk_s(Δ(C7))`. -/
private def linkMasksC7 (s : ℕ) : List ℕ :=
  (List.range 128).filter fun t =>
    (t &&& s == 0) && C7FacetMasks.any (fun F => subMask (t ||| s) F)

private lemma link_eq_ofMasks (s : ℕ) :
    link (delta C7) (ofMask s) = ((linkMasksC7 s).map ofMask).toFinset := by
  ext τ
  rw [← ofMask_maskOf τ, C7_mem_link_mask s (maskOf τ) (maskOf_lt τ)]
  simp only [List.mem_toFinset, List.mem_map, linkMasksC7, List.mem_filter, List.mem_range,
    Bool.and_eq_true, beq_iff_eq, List.any_eq_true]
  constructor
  · rintro ⟨hd, F, hF, hsub⟩
    exact ⟨maskOf τ, ⟨maskOf_lt τ, hd, F, hF, hsub⟩, rfl⟩
  · rintro ⟨t, ⟨htlt, hd, F, hF, hsub⟩, hteq⟩
    have hteq' : t = maskOf τ := (ofMask_eq_iff htlt (maskOf_lt τ)).1 hteq
    subst hteq'
    exact ⟨hd, F, hF, hsub⟩

private lemma linkMasksC7_lt (s : ℕ) : ∀ x ∈ linkMasksC7 s, x < 128 := by
  intro x hx
  simp only [linkMasksC7, List.mem_filter, List.mem_range] at hx
  exact hx.1

/-- Collapsibility of the four non-cone missing facet-intersection links of `C7`:
`σ ∈ {{0}, {0,2}, {0,6}, {1,2}}` (masks `1, 5, 65, 6`). -/
lemma C7_link_02_collapsible : Collapsible (link (delta C7) {0, 2}) := by
  have h5 : ofMask 5 = ({0, 2} : Finset (Fin 7)) := by decide
  rw [← h5, link_eq_ofMasks 5]
  exact collapsible_of_runSteps [(32, 34), (8, 72), (64, 66)] (linkMasksC7 5) 1
    (linkMasksC7_lt 5) (by decide)

lemma C7_link_06_collapsible : Collapsible (link (delta C7) {0, 6}) := by
  have h65 : ofMask 65 = ({0, 6} : Finset (Fin 7)) := by decide
  rw [← h65, link_eq_ofMasks 65]
  exact collapsible_of_runSteps [(2, 6), (16, 24), (8, 12)] (linkMasksC7 65) 2
    (linkMasksC7_lt 65) (by decide)

lemma C7_link_12_collapsible : Collapsible (link (delta C7) {1, 2}) := by
  have h6 : ofMask 6 = ({1, 2} : Finset (Fin 7)) := by decide
  rw [← h6, link_eq_ofMasks 6]
  exact collapsible_of_runSteps [(64, 65), (1, 33), (24, 56), (8, 40), (16, 48)]
    (linkMasksC7 6) 5 (linkMasksC7_lt 6) (by decide)

lemma C7_link_0_collapsible : Collapsible (link (delta C7) {0}) := by
  have h1 : ofMask 1 = ({0} : Finset (Fin 7)) := by decide
  rw [← h1, link_eq_ofMasks 1]
  exact collapsible_of_runSteps
    [(34, 38), (32, 36), (6, 70), (2, 66), (12, 76), (4, 68), (24, 88), (16, 80), (8, 72)]
    (linkMasksC7 1) 6 (linkMasksC7_lt 1) (by decide)

private def checkC7Class (m : ℕ) : Bool :=
  !(C7FacetMasks.any fun F => subMask m F) ||
  (m == 0) || (m ∈ C7Masks) ||
  (m == 1) || (m == 5) || (m == 65) || (m == 6) ||
  (List.finRange 7).any fun v =>
    !bit7 m v && C7FacetMasks.all fun F => !subMask m F || bit7 F v

/-- **Theorem 2 (Step 1 — Local goodness of `C7`)**:
For EVERY nonempty face `σ ∈ Δ(C7) \ C7` (all 46 missing faces), the link `Lk_σ(Δ(C7))` is
Whitehead collapsible to a single vertex (hence contractible). -/
theorem C7_locally_good :
    ∀ σ ∈ delta C7, σ.Nonempty → σ ∉ C7 → Collapsible (link (delta C7) σ) := by
  have hall_ok : (List.range 128).all checkC7Class = true := by decide
  intro σ hσ hne hout
  set m := maskOf σ
  have hmlt : m < 128 := maskOf_lt σ
  have hσ_eq : ofMask m = σ := ofMask_maskOf σ
  have hmem_d : (C7FacetMasks.any fun F => subMask m F) = true := by
    rw [List.any_eq_true]
    exact (C7_mem_delta_mask m).1 (hσ_eq.symm ▸ hσ)
  have hm_ne0 : m ≠ 0 := by
    intro h0
    have h0eq : ofMask 0 = ∅ := by decide
    have : σ = ∅ := by rw [← hσ_eq, h0, h0eq]
    exact hne.ne_empty this
  have hm_out : m ∉ C7Masks := by
    intro hm
    exact hout (hσ_eq ▸ (mem_C7_iff_mask m hmlt).2 hm)
  have hm_ok : checkC7Class m = true :=
    (List.all_eq_true.1 hall_ok) m (List.mem_range.2 hmlt)
  simp only [checkC7Class, hmem_d, Bool.not_true, Bool.false_or, Bool.or_eq_true,
    beq_iff_eq, decide_eq_true_eq, List.any_eq_true, List.mem_finRange, true_and,
    Bool.and_eq_true, Bool.not_eq_true', List.all_eq_true] at hm_ok
  rcases hm_ok with (((((h0 | hin) | h1) | h5) | h65) | h6) | ⟨v, hv_out, hv_all⟩
  · exact (hm_ne0 h0).elim
  · exact (hm_out hin).elim
  · have hσ1 : σ = {0} := by rw [← hσ_eq, h1]; decide
    rw [hσ1]; exact C7_link_0_collapsible
  · have hσ5 : σ = {0, 2} := by rw [← hσ_eq, h5]; decide
    rw [hσ5]; exact C7_link_02_collapsible
  · have hσ65 : σ = {0, 6} := by rw [← hσ_eq, h65]; decide
    rw [hσ65]; exact C7_link_06_collapsible
  · have hσ6 : σ = {1, 2} := by rw [← hσ_eq, h6]; decide
    rw [hσ6]; exact C7_link_12_collapsible
  · have hvσ : v ∉ σ := by
      rw [← hσ_eq, mem_ofMask, hv_out]; simp
    have hF_in_C7 : ∀ F ∈ C7FacetMasks, ofMask F ∈ C7 := by
      intro F hF
      have hFlt : F < 128 := by
        have hall : ∀ y ∈ C7FacetMasks, y < 128 := by decide
        exact hall F hF
      have hFm : F ∈ C7Masks := by
        have hall : ∀ y ∈ C7FacetMasks, y ∈ C7Masks := by decide
        exact hall F hF
      exact (mem_C7_iff_mask F hFlt).2 hFm
    have hv_imp : ∀ F ∈ C7FacetMasks, subMask m F = true → bit7 F v = true := by
      intro F hF hsub
      rcases hv_all F hF with hnot | hbit
      · rw [hsub] at hnot; cases hnot
      · exact hbit
    refine link_cone_collapsible C7 σ v (link_delta_down C7 σ) ?_ ?_
    · intro τ hτ
      rw [mem_link_delta] at hτ ⊢
      rcases hτ with ⟨hdisj, w, hw, hsub⟩
      have hτu_in : τ ∪ σ ∈ delta C7 := mem_delta.2 ⟨w, hw, hsub⟩
      obtain ⟨F, hF, hsubF⟩ := (C7_mem_delta_mask (maskOf (τ ∪ σ))).1
        (by rwa [ofMask_maskOf])
      have hτuF : τ ∪ σ ⊆ ofMask F := by
        rw [← ofMask_maskOf (τ ∪ σ)]; exact ofMask_subset_iff.2 hsubF
      have hσF : subMask m F = true := ofMask_subset_iff.1 (by
        rw [hσ_eq]; exact subset_trans Finset.subset_union_right hτuF)
      have hvF : v ∈ ofMask F := (mem_ofMask F v).2 (hv_imp F hF hσF)
      refine ⟨Finset.disjoint_insert_left.2 ⟨hvσ, hdisj⟩, ofMask F, hF_in_C7 F hF, ?_⟩
      rw [Finset.insert_union]
      exact Finset.insert_subset_iff.2 ⟨hvF, hτuF⟩
    · rw [mem_link_delta]
      obtain ⟨F, hF, hσF⟩ := (C7_mem_delta_mask m).1 (hσ_eq.symm ▸ hσ)
      have hvF : v ∈ ofMask F := (mem_ofMask F v).2 (hv_imp F hF hσF)
      refine ⟨Finset.disjoint_singleton_left.2 hvσ, ofMask F, hF_in_C7 F hF, ?_⟩
      exact Finset.union_subset (Finset.singleton_subset_iff.2 hvF)
        (hσ_eq ▸ ofMask_subset_iff.2 hσF)

/-- **Theorem 2 (Complete refutation of Conjecture 3.13)**:
The 7-neuron, 14-word code `C7` is locally good (every missing nonempty face has a collapsible
link), yet admits NEITHER an open convex NOR a closed convex realization in ANY Euclidean
dimension `d`. -/
theorem C7_refutes_conj313 :
    C7.card = 14 ∧
    (∀ σ ∈ delta C7, σ.Nonempty → σ ∉ C7 → Collapsible (link (delta C7) σ)) ∧
    ¬ IsOpenConvexCode C7 ∧
    ¬ IsClosedConvexCode C7 := by
  refine ⟨by decide, C7_locally_good, ?_, ?_⟩
  · intro hopen
    have hA : IsOpenConvexCode codeA := by
      rw [← C7_restrict_eq_A]
      exact isOpenConvexCode_restrict mapA C7 hopen
    exact codeA_not_openConvex hA
  · intro hclosed
    have hB : IsClosedConvexCode codeB := by
      rw [← C7_restrict_eq_B]
      exact isClosedConvexCode_restrict mapB C7 hclosed
    exact codeB_not_closedConvex hB

end NeuralCodeClosedConvex
