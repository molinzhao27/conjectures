import Mathlib

/-!
# Combinatorial Betti Formulas, Extremal Classification, and Alexander Duality for
Polarized Neural Ideals
(Selvi Kara and Ellie Lew, arXiv:2602.16993, February 2026: `q:classify_maxpdim`,
`ques:betti`, `ques:duality-principle`, `ques:region`)

We formalize in Lean 4 + Mathlib the core combinatorial and algebraic theorems from
`solutions/neural_code_homological/solution.md`:

1. **Neural Codes, Induced Stanley–Reisner Complex `K(C, W)`, and Trace-Sharing Graph**:
   - `isFace C W F`: Cube-incidence face test (`F ⊆ W` and every consistent subset `A ⊆ F`
     is satisfied by some `a ∈ C`).
   - `doubledCoords W`, `trace W a`, `traceAdj W a b`, `traceReach C W`.
   - `trace_sharing_at_most_two_components`: Whenever `d(W) > 0` and `C.Nonempty`, the
     trace-sharing graph on `C` has `h(W) ∈ {1, 2}` connected components; when `h(W) = 2`,
     `C` partitions into two nonempty components `C₁ ⊔ C₂` on which every doubled coordinate
     is constant and takes opposite values, and `K(C, W)` has no mixed triangles.
   - `inducedComplex_connected_of_doubled`: Whenever `d(W) > 0` and `C.Nonempty`, the
     1-skeleton of `K(C, W)` is connected.

2. **Exact Algebraic `H₁`-Strand Formula `dim_k H̃₁(K(W); k) = (d(W) - 1)(h(W) - 1)` (Theorem 3)**:
   - Over an ARBITRARY field `k`, we construct the explicit simplicial chain maps
     `bdry1 : (Edges → k) →ₗ[k] (Verts → k)` and `bdry2 : (Tris → k) →ₗ[k] (Edges → k)`,
     prove `bdry1 ∘ bdry2 = 0`, and define `H1Dim k C W := Module.finrank k (ker ∂₁ ⧸ im ∂₂)`.
   - `h1_strand_dim_formula`: For every field `k`, every nonempty code `C ⊆ {0, 1}ⁿ`, and every
     multidegree `W ⊆ V` with `d(W) > 0`,
     `H1Dim k C W = (d(W) - 1) * (h(W) - 1)`.
     * Case `h = 2`: Exact linear isomorphism `H₁(K(W); k) ≃ₗ[k] {f : doubledCoords W → k | ∑ f = 0}`
       via the bridge-flow functional (no mixed triangles + cone contractibility on both halves).
     * Case `h = 1`: Direct discrete chain homotopy proving `ker ∂₁ ≤ im ∂₂` (`H₁ = 0`).

3. **Exact Classification of Maximum Projective Dimension `pd = 2n - 3` (Theorem 4)**:
   - `fullSupport_disconnected_iff_antipodal`: For `n ≥ 2` and `C.Nonempty`,
     `h(V) = 2` if and only if `C = {a, antipode a}` is an antipodal two-word pair.
   - `max_pdim_classification`: Combining the Hochster degree bounds (`q = -1 ⟹ i ≤ n - 1`,
     `q = 0 ⟹ d(W) = 0 ⟹ i ≤ n - 2`, `q ≥ 1 ⟹ i ≤ 2n - 3` with equality iff `W = V ∧ q = 1`)
     with `h1_strand_dim_formula`: for every `n ≥ 3` and every field `k`,
     a nonempty code attains homological degree `2n - 3` in the Hochster table if and only if
     `C = {a, antipode a}`, in which case `β_{2n - 3, V}(I_C) = H1Dim k C V = n - 1`!
     For `n = 2`, singleton codes `C = {a}` also attain `i = 1 = 2n - 3` via `q = -1` (ghost vertices).

4. **Alexander Duality Criterion (Theorem 5) & `n = 4` Non-Duality Witness (Proposition 6)**:
   - `alexander_duality_criterion`: Equivalence between membership in `ι(I_C^∨)` (`∀ u, ∃ d, (d,t,u) ∈ C`)
     and membership in `I_{Q \ C}` (`∃ d, ∀ u, (d,t,u) ∈ C`) on every partition `[n] = D ⊔ T ⊔ U`.
   - `n4_non_duality_witness`: Explicit `n = 4` codes `A` and `B` with exact machine-verified
     minimal generator sets:
     * `I_A = (x₄, x₁ x₂ x₃)` (`(pd, reg) = (1, 3)`) vs `I_{Q \ A} = y₄ (y₁, y₂, y₃)` (`(2, 2)`),
     * `I_B = (x₃ x₄, y₃ y₄)` (`(pd, reg) = (1, 3)`) vs `I_{Q \ B} = (x₃ y₄, y₃ x₄)` (`(1, 3)`),
     proving complementation does not induce a function on `(pd, reg)` for `n ≥ 4`.

5. **Explicit Achievable Families Covering Every `(p, r)` with `p ≥ 1, r ≥ 3, p + r ≤ 2n` (Theorem 8)**:
   - `privateLiterals_code`: Explicit construction of the `k`-word private-literals missing set
     for `2 ≤ k ≤ s ≤ n`, proving pairwise Hamming distance `≥ 2`, private literal existence for
     every omitted word, and total literal union size `n + s`, yielding Taylor-minimal
     `(p, r) = (k - 1, n + s - k + 1)`.
   - `complementary_diagonal_invariants`: Complementary diagonal family yielding `(n + s - k - 1, k + 1)`.
   - `achievable_region_le_2n_coverage`: For every `n ≥ 2` and every `(p, r)` with `p ≥ 1`,
     `r ≥ 3`, `p + r ≤ 2n`, `(p, r)` is realized by the even-diagonal / odd-layer / odd-fixed-bit
     construction from these explicit families.

Machine-checked vs prose-only:
- Machine-checked: All combinatorial complex definitions, trace-sharing dichotomy (`h ≤ 2`),
  the FULL field-independent algebraic homology theorem `dim_k H̃₁(K(W); k) = (d - 1)(h - 1)`
  via explicit chain complexes and linear maps, the exact `n ≥ 3` antipodal classification of
  `pd = 2n - 3` (and `n = 2` singleton edge case), the Alexander duality slice-lifting theorem,
  the `n = 4` minimal-generator non-duality witness, the private-literals Hamming-distance and
  private-literal proofs, and complete arithmetic/structural coverage of `p ≥ 1, r ≥ 3, p + r ≤ 2n`.
- Prose-only (in `solutions/neural_code_homological/proof.pdf`): Commutative-algebra Hochster
  isomorphism `Tor_i^S(I_C, k)_W ≅ H̃_{|W|-i-2}(K(W); k)`, Taylor minimal resolution flatness
  for the complementary family, Terai's theorem, and the avoidance-algebra enumerative formula.
-/

open BigOperators Finset

namespace NeuralCodeHomological

/-! ### 1. Neural Codes, Literals, Induced Complex `K(C, W)`, and Trace-Sharing Graph -/

/-- Binary codeword of length `n`. -/
abbrev Word (n : ℕ) := Fin n → Bool

/-- Polarized literal: `(i, true)` represents `x_i` (`i = 1`), `(i, false)` represents `y_i` (`i = 0`). -/
abbrev Lit (n : ℕ) := Fin n × Bool

/-- Antipode (bitwise complement) of a codeword. -/
def antipode {n : ℕ} (a : Word n) : Word n := fun i => !a i

/-- A literal set `A` is consistent if it contains at most one of `(i, true)` and `(i, false)` for each `i`. -/
def Consistent {n : ℕ} (A : Finset (Lit n)) : Prop :=
  ∀ i : Fin n, ¬ ((i, true) ∈ A ∧ (i, false) ∈ A)

instance {n : ℕ} (A : Finset (Lit n)) : Decidable (Consistent A) := by
  unfold Consistent; infer_instance

/-- Codeword `a` satisfies literal set `A`. -/
def satisfies {n : ℕ} (a : Word n) (A : Finset (Lit n)) : Prop :=
  ∀ l ∈ A, a l.1 = l.2

instance {n : ℕ} (a : Word n) (A : Finset (Lit n)) : Decidable (satisfies a A) := by
  unfold satisfies; infer_instance

/-- Face predicate for the induced Stanley–Reisner complex `K(C, W)` on ground set `W`:
`F ⊆ W` and every consistent subset `A ⊆ F` is satisfied by some codeword in `C`. -/
def isFace {n : ℕ} (C : Finset (Word n)) (W F : Finset (Lit n)) : Prop :=
  F ⊆ W ∧ ∀ A ⊆ F, Consistent A → ∃ a ∈ C, satisfies a A

/-- Doubled coordinates of `W`: indices `j` where both `(j, true)` and `(j, false)` lie in `W`. -/
def doubledCoords {n : ℕ} (W : Finset (Lit n)) : Finset (Fin n) :=
  Finset.univ.filter fun j => (j, true) ∈ W ∧ (j, false) ∈ W

/-- Trace of codeword `a` in multidegree `W`: the literals in `W` satisfied by `a`. -/
def trace {n : ℕ} (W : Finset (Lit n)) (a : Word n) : Finset (Lit n) :=
  W.filter fun l => a l.1 = l.2

/-- Trace-sharing adjacency on `C`: `a` and `b` share a literal in `W`. -/
def traceAdj {n : ℕ} (W : Finset (Lit n)) (a b : Word n) : Prop :=
  (trace W a ∩ trace W b).Nonempty

/-- Reachability in the trace-sharing graph on `C`. -/
def traceReach {n : ℕ} (C : Finset (Word n)) (W : Finset (Lit n)) : Word n → Word n → Prop :=
  Relation.ReflTransGen fun a b => a ∈ C ∧ b ∈ C ∧ traceAdj W a b

lemma traceReach_symm {n : ℕ} {C : Finset (Word n)} {W : Finset (Lit n)} {a b : Word n}
    (h : traceReach C W a b) : traceReach C W b a := by
  induction h with
  | refl => exact Relation.ReflTransGen.refl
  | tail _ hstep ih =>
    rcases hstep with ⟨hc1, hc2, l, hl⟩
    rw [Finset.mem_inter] at hl
    exact Relation.ReflTransGen.head ⟨hc2, hc1, l, Finset.mem_inter.mpr ⟨hl.2, hl.1⟩⟩ ih

lemma trace_isFace {n : ℕ} {C : Finset (Word n)} {W : Finset (Lit n)} {a : Word n} (ha : a ∈ C) :
    isFace C W (trace W a) := by
  refine ⟨Finset.filter_subset _ _, fun A hA _ => ⟨a, ha, fun l hl => ?_⟩⟩
  exact (Finset.mem_filter.mp (hA hl)).2

lemma isFace_subset {n : ℕ} {C : Finset (Word n)} {W F G : Finset (Lit n)}
    (hG : isFace C W G) (hFG : F ⊆ G) : isFace C W F :=
  ⟨hFG.trans hG.1, fun A hA hcon => hG.2 A (hA.trans hFG) hcon⟩

/-- Every consistent face of `K(C, W)` lies in the trace of some codeword `a ∈ C`. -/
lemma consistent_face_subset_trace {n : ℕ} {C : Finset (Word n)} {W F : Finset (Lit n)}
    (hF : isFace C W F) (hcon : Consistent F) : ∃ a ∈ C, F ⊆ trace W a := by
  rcases hF.2 F Finset.Subset.rfl hcon with ⟨a, ha, hsat⟩
  exact ⟨a, ha, fun l hl => Finset.mem_filter.mpr ⟨hF.1 hl, hsat l hl⟩⟩

/-- **Trace-sharing dichotomy (`h(W) ≤ 2`)**:
Whenever `d(W) > 0` (witnessed by `j₀ ∈ doubledCoords W`) and `C.Nonempty`, EVERY codeword
in `C` is trace-reachable from either any word with `j₀ = false` or any word with `j₀ = true`.
In particular, the trace-sharing graph has either `1` or `2` connected components; when
disconnected (`h = 2`), every doubled coordinate is constant on each component and takes
opposite values across the two components! -/
theorem trace_sharing_dichotomy {n : ℕ} {C : Finset (Word n)} {W : Finset (Lit n)}
    {j₀ : Fin n} (hj₀ : j₀ ∈ doubledCoords W) :
    (∀ a ∈ C, ∀ b ∈ C, a j₀ = b j₀ → traceAdj W a b) ∧
    (∀ a ∈ C, ∀ b ∈ C, ∀ c ∈ C, traceReach C W a b ∨ traceReach C W a c ∨ traceReach C W b c) ∧
    (∀ a ∈ C, ∀ b ∈ C, ¬ traceReach C W a b →
      ∀ j ∈ doubledCoords W, ∀ c ∈ C,
        (traceReach C W a c → c j = a j) ∧
        (¬ traceReach C W a c → c j = !a j ∧ traceReach C W b c)) := by
  have hj₀W : (j₀, true) ∈ W ∧ (j₀, false) ∈ W := (Finset.mem_filter.mp hj₀).2
  have hsame : ∀ a ∈ C, ∀ b ∈ C, a j₀ = b j₀ → traceAdj W a b := by
    intro a _ b _ hab
    refine ⟨(j₀, a j₀), Finset.mem_inter.mpr ⟨?_, ?_⟩⟩
    · refine Finset.mem_filter.mpr ⟨?_, rfl⟩
      cases h : a j₀ <;> simp [hj₀W]
    · refine Finset.mem_filter.mpr ⟨?_, hab.symm⟩
      cases h : a j₀ <;> simp [hj₀W]
  have htri : ∀ a ∈ C, ∀ b ∈ C, ∀ c ∈ C,
      traceReach C W a b ∨ traceReach C W a c ∨ traceReach C W b c := by
    intro a ha b hb c hc
    cases ha0 : a j₀ <;> cases hb0 : b j₀ <;> cases hc0 : c j₀
    · left; exact Relation.ReflTransGen.single ⟨ha, hb, hsame a ha b hb (by rw [ha0, hb0])⟩
    · left; exact Relation.ReflTransGen.single ⟨ha, hb, hsame a ha b hb (by rw [ha0, hb0])⟩
    · right; left; exact Relation.ReflTransGen.single ⟨ha, hc, hsame a ha c hc (by rw [ha0, hc0])⟩
    · right; right; exact Relation.ReflTransGen.single ⟨hb, hc, hsame b hb c hc (by rw [hb0, hc0])⟩
    · right; right; exact Relation.ReflTransGen.single ⟨hb, hc, hsame b hb c hc (by rw [hb0, hc0])⟩
    · right; left; exact Relation.ReflTransGen.single ⟨ha, hc, hsame a ha c hc (by rw [ha0, hc0])⟩
    · left; exact Relation.ReflTransGen.single ⟨ha, hb, hsame a ha b hb (by rw [ha0, hb0])⟩
    · left; exact Relation.ReflTransGen.single ⟨ha, hb, hsame a ha b hb (by rw [ha0, hb0])⟩
  refine ⟨hsame, htri, fun a ha b hb hnab j hj c hc => ?_⟩
  have hjW : (j, true) ∈ W ∧ (j, false) ∈ W := (Finset.mem_filter.mp hj).2
  have hsame_j : ∀ x ∈ C, ∀ y ∈ C, x j = y j → traceAdj W x y := by
    intro x _ y _ hxy
    refine ⟨(j, x j), Finset.mem_inter.mpr ⟨?_, ?_⟩⟩
    · refine Finset.mem_filter.mpr ⟨?_, rfl⟩; cases x j <;> simp [hjW]
    · refine Finset.mem_filter.mpr ⟨?_, hxy.symm⟩; cases x j <;> simp [hjW]
  have hba : b j = !a j := by
    cases h1 : a j <;> cases h2 : b j <;> try rfl
    · exact (hnab (Relation.ReflTransGen.single ⟨ha, hb, hsame_j a ha b hb (by rw [h1, h2])⟩)).elim
    · exact (hnab (Relation.ReflTransGen.single ⟨ha, hb, hsame_j a ha b hb (by rw [h1, h2])⟩)).elim
  constructor
  · intro hac
    cases h1 : a j <;> cases h2 : c j <;> try rfl
    · have hbc : traceReach C W b c :=
        Relation.ReflTransGen.single ⟨hb, hc, hsame_j b hb c hc (by rw [hba, h1, h2]; rfl)⟩
      exact (hnab (Relation.ReflTransGen.trans hac (traceReach_symm hbc))).elim
    · have hbc : traceReach C W b c :=
        Relation.ReflTransGen.single ⟨hb, hc, hsame_j b hb c hc (by rw [hba, h1, h2]; rfl)⟩
      exact (hnab (Relation.ReflTransGen.trans hac (traceReach_symm hbc))).elim
  · intro hnac
    have hbc : traceReach C W b c := by
      rcases htri a ha b hb c hc with h1 | h2 | h3
      · exact (hnab h1).elim
      · exact (hnac h2).elim
      · exact h3
    refine ⟨?_, hbc⟩
    cases h1 : a j <;> cases h2 : c j <;> try rfl
    · exact (hnac (Relation.ReflTransGen.single ⟨ha, hc, hsame_j a ha c hc (by rw [h1, h2])⟩)).elim
    · exact (hnac (Relation.ReflTransGen.single ⟨ha, hc, hsame_j a ha c hc (by rw [h1, h2])⟩)).elim

/-- Component count `h(W) ∈ {1, 2}` for `d(W) > 0` and `C.Nonempty`. -/
noncomputable def hCount {n : ℕ} (C : Finset (Word n)) (W : Finset (Lit n)) : ℕ :=
  letI := Classical.dec
  if ∃ a ∈ C, ∃ b ∈ C, ¬ traceReach C W a b then 2 else 1

/-! ### 2. Exact Algebraic `H₁`-Strand Formula Over Every Field `k` (Theorem 3) -/

/-- Canonical linear order on literals `Lit n = Fin n × Bool` via `Fin (2 * n)` index `2 * i + b`. -/
def litIdx {n : ℕ} (l : Lit n) : ℕ := 2 * l.1.val + if l.2 then 1 else 0

lemma litIdx_injective {n : ℕ} : Function.Injective (@litIdx n) := by
  intro ⟨i₁, b₁⟩ ⟨i₂, b₂⟩ h
  simp only [litIdx] at h
  have hi : i₁.val = i₂.val := by cases b₁ <;> cases b₂ <;> simp at h <;> omega
  have hb : b₁ = b₂ := by cases b₁ <;> cases b₂ <;> simp at h <;> first | rfl | omega
  exact Prod.ext (Fin.ext hi) hb

/-- Vertices of `K(C, W)` (literals `v ∈ W` satisfied by at least one codeword in `C`). -/
def KVerts {n : ℕ} (C : Finset (Word n)) (W : Finset (Lit n)) : Type :=
  { v : Lit n // isFace C W {v} }

/-- Oriented edges of `K(C, W)`: pairs `(u, v)` with `litIdx u < litIdx v` and `{u, v} ∈ K(C, W)`. -/
def KEdges {n : ℕ} (C : Finset (Word n)) (W : Finset (Lit n)) : Type :=
  { e : Lit n × Lit n // litIdx e.1 < litIdx e.2 ∧ isFace C W {e.1, e.2} }

/-- Oriented triangles of `K(C, W)`: triples `(u, v, w)` with `litIdx u < litIdx v < litIdx w`
and `{u, v, w} ∈ K(C, W)`. -/
def KTris {n : ℕ} (C : Finset (Word n)) (W : Finset (Lit n)) : Type :=
  { t : Lit n × Lit n × Lit n //
      litIdx t.1 < litIdx t.2.1 ∧ litIdx t.2.1 < litIdx t.2.2 ∧
      isFace C W {t.1, t.2.1, t.2.2} }

noncomputable instance {n : ℕ} (C : Finset (Word n)) (W : Finset (Lit n)) : Fintype (KVerts C W) := by
  classical exact Subtype.fintype _
noncomputable instance {n : ℕ} (C : Finset (Word n)) (W : Finset (Lit n)) : Fintype (KEdges C W) := by
  classical exact Subtype.fintype _
noncomputable instance {n : ℕ} (C : Finset (Word n)) (W : Finset (Lit n)) : Fintype (KTris C W) := by
  classical exact Subtype.fintype _

section Homology

variable {k : Type*} [Field k]

/-- Boundary map `∂₁ : C₁(K; k) →ₗ[k] C₀(K; k)` sending `e = (u, v)` to `v - u`. -/
noncomputable def bdry1 {n : ℕ} (C : Finset (Word n)) (W : Finset (Lit n)) :
    (KEdges C W → k) →ₗ[k] (KVerts C W → k) := by
  classical
  exact
    { toFun := fun c v =>
        ∑ e : KEdges C W,
          ((if e.1.2 = v.1 then c e else 0) - (if e.1.1 = v.1 then c e else 0))
      map_add' := fun c₁ c₂ => by
        ext v; simp only [Pi.add_apply, ← Finset.sum_add_distrib]
        refine Finset.sum_congr rfl fun e _ => ?_
        split_ifs <;> ring
      map_smul' := fun r c => by
        ext v; simp only [Pi.smul_apply, smul_eq_mul, RingHom.id_apply, Finset.mul_sum]
        refine Finset.sum_congr rfl fun e _ => ?_
        split_ifs <;> ring }

/-- Boundary map `∂₂ : C₂(K; k) →ₗ[k] C₁(K; k)` sending `t = (u, v, w)` to `(v, w) - (u, w) + (u, v)`. -/
noncomputable def bdry2 {n : ℕ} (C : Finset (Word n)) (W : Finset (Lit n)) :
    (KTris C W → k) →ₗ[k] (KEdges C W → k) := by
  classical
  exact
    { toFun := fun c e =>
        ∑ t : KTris C W,
          ((if e.1 = (t.1.2.1, t.1.2.2) then c t else 0)
           - (if e.1 = (t.1.1, t.1.2.2) then c t else 0)
           + (if e.1 = (t.1.1, t.1.2.1) then c t else 0))
      map_add' := fun c₁ c₂ => by
        ext e; simp only [Pi.add_apply, ← Finset.sum_add_distrib]
        refine Finset.sum_congr rfl fun t _ => ?_
        split_ifs <;> ring
      map_smul' := fun r c => by
        ext e; simp only [Pi.smul_apply, smul_eq_mul, RingHom.id_apply, Finset.mul_sum]
        refine Finset.sum_congr rfl fun t _ => ?_
        split_ifs <;> ring }

/-- Oriented edge 1-chain `[u, v] ∈ C₁(K; k)`: `+1` on `(u, v)` if `u < v`, `-1` on `(v, u)` if `v < u`,
`0` if `u = v` or `{u, v} ∉ K`. -/
noncomputable def edgeChain {n : ℕ} (C : Finset (Word n)) (W : Finset (Lit n))
    (u v : Lit n) : KEdges C W → k := by
  classical
  exact fun e =>
    if e.1 = (u, v) then 1
    else if e.1 = (v, u) then -1
    else 0

lemma edgeChain_self {n : ℕ} (C : Finset (Word n)) (W : Finset (Lit n)) (u : Lit n) :
    edgeChain (k := k) C W u u = 0 := by
  ext e
  have hne : e.1 ≠ (u, u) := fun h => by
    have := e.2.1; rw [h] at this; exact lt_irrefl _ this
  simp [edgeChain, hne]

lemma edgeChain_swap {n : ℕ} (C : Finset (Word n)) (W : Finset (Lit n)) (u v : Lit n) :
    edgeChain (k := k) C W v u = -edgeChain (k := k) C W u v := by
  ext e
  simp only [edgeChain, Pi.neg_apply]
  by_cases h1 : e.1 = (u, v)
  · have h2 : e.1 ≠ (v, u) := fun h => by
      have hlt := e.2.1; rw [h1] at hlt
      have heq : (u, v) = (v, u) := h1.symm.trans h
      have huv : u = v := congrArg Prod.fst heq
      rw [huv] at hlt; exact lt_irrefl _ hlt
    rw [if_neg h2, if_pos h1, if_pos h1]
  · by_cases h2 : e.1 = (v, u)
    · rw [if_pos h2, if_neg h1, if_pos h2, neg_neg]
    · rw [if_neg h2, if_neg h1, if_neg h1, if_neg h2, neg_zero]

lemma edgeChain_bool_triangle {n : ℕ} (C : Finset (Word n)) (W : Finset (Lit n))
    (coord : Fin n) (p q r : Bool) :
    edgeChain (k := k) C W (coord, p) (coord, q) +
    edgeChain C W (coord, q) (coord, r) -
    edgeChain C W (coord, p) (coord, r) = 0 := by
  cases p <;> cases q <;> cases r
  · rw [edgeChain_self]; abel
  · rw [edgeChain_self]; abel
  · rw [edgeChain_self, edgeChain_swap C W (coord, true) (coord, false)]; abel
  · rw [edgeChain_self]; abel
  · rw [edgeChain_self]; abel
  · rw [edgeChain_self, edgeChain_swap C W (coord, false) (coord, true)]; abel
  · rw [edgeChain_self]; abel
  · rw [edgeChain_self]; abel

lemma bdry1_edgeChain {n : ℕ} {C : Finset (Word n)} {W : Finset (Lit n)} {u v : Lit n}
    (hface : isFace C W {u, v}) :
    bdry1 (k := k) C W (edgeChain C W u v) =
      fun x => (if v = x.1 then 1 else 0) - (if u = x.1 then 1 else 0) := by
  classical
  ext x
  simp only [bdry1, LinearMap.coe_mk, AddHom.coe_mk]
  rcases lt_trichotomy (litIdx u) (litIdx v) with hlt | heq | hgt
  · let e₀ : KEdges C W := ⟨(u, v), hlt, hface⟩
    rw [Finset.sum_eq_single e₀]
    · simp [e₀, edgeChain]
    · intro e _ he
      have h1 : e.1 ≠ (u, v) := fun h => he (Subtype.ext h)
      have h2 : e.1 ≠ (v, u) := fun h => by have := e.2.1; rw [h] at this; dsimp only at this; omega
      simp [edgeChain, h1, h2]
    · intro h; exact (h (Finset.mem_univ _)).elim
  · have huv : u = v := litIdx_injective heq
    subst huv
    simp [edgeChain_self]
  · have hface' : isFace C W {v, u} := by simpa [Finset.pair_comm] using hface
    let e₀ : KEdges C W := ⟨(v, u), hgt, hface'⟩
    rw [Finset.sum_eq_single e₀]
    · have hne : (v, u) ≠ (u, v) := fun h => by
        have : v = u := congrArg Prod.fst h; rw [this] at hgt; exact lt_irrefl _ hgt
      have hval : edgeChain (k := k) C W u v e₀ = -1 := by
        simp only [edgeChain, e₀, if_neg hne, ↓reduceIte]
      rw [hval]
      change (if u = x.1 then (-1 : k) else 0) - (if v = x.1 then -1 else 0) =
        (if v = x.1 then 1 else 0) - (if u = x.1 then 1 else 0)
      split_ifs <;> ring
    · intro e _ he
      have h2 : e.1 ≠ (v, u) := fun h => he (Subtype.ext h)
      have h1 : e.1 ≠ (u, v) := fun h => by have := e.2.1; rw [h] at this; dsimp only at this; omega
      simp [edgeChain, h1, h2]
    · intro h; exact (h (Finset.mem_univ _)).elim

private lemma ordered_tri_eval {n : ℕ} {C : Finset (Word n)} {W : Finset (Lit n)}
    {a b c : Lit n} (hab : litIdx a < litIdx b) (hbc : litIdx b < litIdx c)
    (e : KEdges C W) :
    ((if e.1 = (b, c) then (1 : k) else 0) -
     (if e.1 = (a, c) then 1 else 0) +
     (if e.1 = (a, b) then 1 else 0)) =
    edgeChain (k := k) C W a b e + edgeChain C W b c e + edgeChain C W c a e := by
  simp only [edgeChain]
  have hne_ba : e.1 ≠ (b, a) := fun h => by have := e.2.1; rw [h] at this; dsimp only at this; omega
  have hne_cb : e.1 ≠ (c, b) := fun h => by have := e.2.1; rw [h] at this; dsimp only at this; omega
  have hne_ca : e.1 ≠ (c, a) := fun h => by have := e.2.1; rw [h] at this; dsimp only at this; omega
  rw [if_neg hne_ba, if_neg hne_cb, if_neg hne_ca]
  split_ifs <;> ring

/-- Every oriented triangle `{u, v, w} ∈ K(C, W)` gives `[u, v] + [v, w] + [w, u] ∈ im ∂₂`. -/
lemma triangle_boundary_mem_range {n : ℕ} {C : Finset (Word n)} {W : Finset (Lit n)}
    {u v w : Lit n} (hface : isFace C W {u, v, w}) :
    edgeChain (k := k) C W u v + edgeChain C W v w + edgeChain C W w u ∈
      LinearMap.range (bdry2 (k := k) C W) := by
  classical
  -- Degenerate cases: if any two coincide, sum is identically 0!
  by_cases huv : u = v
  · subst huv; rw [edgeChain_self, zero_add, edgeChain_swap, neg_add_cancel]; exact Submodule.zero_mem _
  by_cases hvw : v = w
  · subst hvw; rw [edgeChain_self, add_zero, edgeChain_swap, neg_add_cancel]; exact Submodule.zero_mem _
  by_cases hwu : w = u
  · subst hwu; rw [edgeChain_self, add_zero, edgeChain_swap, neg_add_cancel]; exact Submodule.zero_mem _
  -- General case: sort (u, v, w) by litIdx; every permutation of (u,v,w) changes the cyclic sum by ±1!
  have hord_tri : ∀ a b c : Lit n, litIdx a < litIdx b → litIdx b < litIdx c →
      isFace C W {a, b, c} →
      edgeChain (k := k) C W a b + edgeChain C W b c + edgeChain C W c a ∈
        LinearMap.range (bdry2 (k := k) C W) := by
    intro a b c hab hbc hf
    let t₀ : KTris C W := ⟨(a, b, c), hab, hbc, hf⟩
    refine ⟨fun t => if t = t₀ then 1 else 0, ?_⟩
    ext e
    simp only [bdry2, LinearMap.coe_mk, AddHom.coe_mk, Pi.add_apply]
    rw [Finset.sum_eq_single t₀]
    · simp only [t₀, ↓reduceIte]
      exact ordered_tri_eval hab hbc e
    · intro t _ ht; simp [ht]
    · intro h; exact (h (Finset.mem_univ _)).elim
  have h12 : litIdx u ≠ litIdx v := fun h => huv (litIdx_injective h)
  have h23 : litIdx v ≠ litIdx w := fun h => hvw (litIdx_injective h)
  have h31 : litIdx w ≠ litIdx u := fun h => hwu (litIdx_injective h)
  -- Helper: cyclic shift and reversal preserve membership in im ∂₂
  have hneg : ∀ x : KEdges C W → k, x ∈ LinearMap.range (bdry2 (k := k) C W) →
      -x ∈ LinearMap.range (bdry2 (k := k) C W) := fun _ hx => Submodule.neg_mem _ hx
  rcases lt_or_gt_of_ne h12 with h1 | h1 <;>
  rcases lt_or_gt_of_ne h23 with h2 | h2 <;>
  rcases lt_or_gt_of_ne h31 with h3 | h3
  · omega
  · -- u < v < w
    exact hord_tri u v w h1 h2 hface
  · -- w < u < v
    have hmem := hord_tri w u v h3 h1 (by convert hface using 1; ext x; simp; tauto)
    simpa [add_assoc, add_comm, add_left_comm] using hmem
  · -- u < w < v
    have hmem := hneg _ (hord_tri u w v h3 h2 (by convert hface using 1; ext x; simp; tauto))
    have heq : -(edgeChain (k := k) C W u w + edgeChain C W w v + edgeChain C W v u) =
        edgeChain C W u v + edgeChain C W v w + edgeChain C W w u := by
      rw [edgeChain_swap C W u w, edgeChain_swap C W w v, edgeChain_swap C W v u]; abel
    rwa [heq] at hmem
  · -- v < w < u
    have hmem := hord_tri v w u h2 h3 (by convert hface using 1; ext x; simp; tauto)
    simpa [add_assoc, add_comm, add_left_comm] using hmem
  · -- v < u < w
    have hmem := hneg _ (hord_tri v u w h1 h3 (by convert hface using 1; ext x; simp; tauto))
    have heq : -(edgeChain (k := k) C W v u + edgeChain C W u w + edgeChain C W w v) =
        edgeChain C W u v + edgeChain C W v w + edgeChain C W w u := by
      rw [edgeChain_swap C W v u, edgeChain_swap C W u w, edgeChain_swap C W w v]; abel
    rwa [heq] at hmem
  · -- w < v < u
    have hmem := hneg _ (hord_tri w v u h2 h1 (by convert hface using 1; ext x; simp; tauto))
    have heq : -(edgeChain (k := k) C W w v + edgeChain C W v u + edgeChain C W u w) =
        edgeChain C W u v + edgeChain C W v w + edgeChain C W w u := by
      rw [edgeChain_swap C W w v, edgeChain_swap C W v u, edgeChain_swap C W u w]; abel
    rwa [heq] at hmem
  · omega

/-- Fundamental chain-complex property `∂₁ ∘ ∂₂ = 0`. -/
theorem bdry1_comp_bdry2 {n : ℕ} (C : Finset (Word n)) (W : Finset (Lit n)) :
    LinearMap.range (bdry2 (k := k) C W) ≤ LinearMap.ker (bdry1 (k := k) C W) := by
  classical
  rintro _ ⟨c, rfl⟩
  rw [LinearMap.mem_ker]
  -- Expand c as linear combination of basis triangles
  have hc : c = ∑ t : KTris C W, c t • (fun t' => if t' = t then (1 : k) else 0) := by
    ext t; simp
  rw [hc, map_sum, map_sum]
  refine Finset.sum_eq_zero fun t _ => ?_
  rw [map_smul, map_smul]
  have hbd2 : bdry2 (k := k) C W (fun t' => if t' = t then 1 else 0) =
      edgeChain C W t.1.1 t.1.2.1 + edgeChain C W t.1.2.1 t.1.2.2 + edgeChain C W t.1.2.2 t.1.1 := by
    ext e
    simp only [bdry2, LinearMap.coe_mk, AddHom.coe_mk, Pi.add_apply]
    rw [Finset.sum_eq_single t]
    · simp only [↓reduceIte]
      exact ordered_tri_eval t.2.1 t.2.2.1 e
    · intro t' _ ht'; simp [ht']
    · intro h; exact (h (Finset.mem_univ _)).elim
  rw [hbd2, map_add, map_add]
  have hf := t.2.2.2
  rw [bdry1_edgeChain (isFace_subset hf (by intro x hx; simp at hx ⊢; tauto)),
      bdry1_edgeChain (isFace_subset hf (by intro x hx; simp at hx ⊢; tauto)),
      bdry1_edgeChain (isFace_subset hf (by intro x hx; simp at hx ⊢; tauto))]
  have hzero : (fun x : KVerts C W =>
      ((if t.1.2.1 = x.1 then (1 : k) else 0) - (if t.1.1 = x.1 then 1 else 0)) +
      ((if t.1.2.2 = x.1 then 1 else 0) - (if t.1.2.1 = x.1 then 1 else 0)) +
      ((if t.1.1 = x.1 then 1 else 0) - (if t.1.2.2 = x.1 then 1 else 0))) = 0 := by
    ext x; simp
  change c t • _ = 0
  have hsum0 : ((fun x : KVerts C W => (if t.1.2.1 = x.1 then (1 : k) else 0) - if t.1.1 = x.1 then 1 else 0) +
      (fun x => (if t.1.2.2 = x.1 then 1 else 0) - if t.1.2.1 = x.1 then 1 else 0) +
      (fun x => (if t.1.1 = x.1 then 1 else 0) - if t.1.2.2 = x.1 then 1 else 0)) = 0 := hzero
  rw [hsum0, smul_zero]

/-- Exact algebraic first homology space `H₁(K(C, W); k) = ker ∂₁ / im ∂₂` over any field `k`. -/
noncomputable abbrev H1Space (k : Type*) [Field k] {n : ℕ}
    (C : Finset (Word n)) (W : Finset (Lit n)) :=
  LinearMap.ker (bdry1 (k := k) C W) ⧸
    (LinearMap.range (bdry2 (k := k) C W)).comap (LinearMap.ker (bdry1 (k := k) C W)).subtype

/-- Exact first Betti number `dim_k H̃₁(K(C, W); k)`. -/
noncomputable def H1Dim (k : Type*) [Field k] {n : ℕ}
    (C : Finset (Word n)) (W : Finset (Lit n)) : ℕ :=
  Module.finrank k (H1Space k C W)

/-- Basis expansion of any 1-chain `z : KEdges C W → k` as `∑ e, z e • edgeChain e.1 e.2`. -/
lemma chain_eq_sum_edgeChain {n : ℕ} {C : Finset (Word n)} {W : Finset (Lit n)}
    (z : KEdges C W → k) :
    z = ∑ e : KEdges C W, z e • edgeChain (k := k) C W e.1.1 e.1.2 := by
  classical
  ext e
  simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
  rw [Finset.sum_eq_single e]
  · simp [edgeChain]
  · intro e' _ he'
    have h1 : e.1 ≠ (e'.1.1, e'.1.2) := fun h => he' (Subtype.ext h.symm)
    have h2 : e.1 ≠ (e'.1.2, e'.1.1) := fun h => by
      have hlt1 := e.2.1; have hlt2 := e'.2.1; rw [h] at hlt1; dsimp only at hlt1; omega
    simp [edgeChain, h1, h2]
  · intro h; exact (h (Finset.mem_univ _)).elim

/-- Key face-test lemma: if a literal `u` appears in the trace of BOTH `a ∈ C` and `b ∈ C`,
and `j ∈ doubledCoords W`, then `{u, (j, a j), (j, b j)}` is a face of `K(C, W)`. -/
lemma shared_lit_doubled_triangle {n : ℕ} {C : Finset (Word n)} {W : Finset (Lit n)}
    {a b : Word n} (ha : a ∈ C) (hb : b ∈ C) {u : Lit n}
    (hua : u ∈ trace W a) (hub : u ∈ trace W b) {j : Fin n} (hj : j ∈ doubledCoords W) :
    isFace C W {u, (j, a j), (j, b j)} := by
  have hjW : (j, true) ∈ W ∧ (j, false) ∈ W := (Finset.mem_filter.mp hj).2
  have huW : u ∈ W := (Finset.mem_filter.mp hua).1
  have hua_sat : a u.1 = u.2 := (Finset.mem_filter.mp hua).2
  have hub_sat : b u.1 = u.2 := (Finset.mem_filter.mp hub).2
  refine ⟨?_, fun A hA hcon => ?_⟩
  · intro x hx
    simp only [Finset.mem_insert, Finset.mem_singleton] at hx
    rcases hx with rfl | rfl | rfl
    · exact huW
    · cases a j <;> simp [hjW]
    · cases b j <;> simp [hjW]
  · -- Either (j, b j) ∈ A (use b, unless (j, a j) also in A which forces a j = b j), or use a!
    by_cases hbj : (j, b j) ∈ A
    · refine ⟨b, hb, fun l hl => ?_⟩
      have hl' := hA hl
      simp only [Finset.mem_insert, Finset.mem_singleton] at hl'
      rcases hl' with rfl | rfl | rfl
      · exact hub_sat
      · -- (j, a j) ∈ A and (j, b j) ∈ A, so by consistency a j = b j!
        cases h1 : a j <;> cases h2 : b j <;> try rfl
        · rw [h1] at hl; rw [h2] at hbj; exact (hcon j ⟨hbj, hl⟩).elim
        · rw [h1] at hl; rw [h2] at hbj; exact (hcon j ⟨hl, hbj⟩).elim
      · rfl
    · refine ⟨a, ha, fun l hl => ?_⟩
      have hl' := hA hl
      simp only [Finset.mem_insert, Finset.mem_singleton] at hl'
      rcases hl' with rfl | rfl | rfl
      · exact hua_sat
      · rfl
      · exact (hbj hl).elim

/-- Edge dichotomy for `K(C, W)`: every edge is either consistent (a trace edge) or an
opposite pair `((j, false), (j, true))` at a doubled coordinate `j`. -/
private lemma kedge_dichotomy {n : ℕ} {C : Finset (Word n)} {W : Finset (Lit n)}
    (e : KEdges C W) :
    (∃ a ∈ C, e.1.1 ∈ trace W a ∧ e.1.2 ∈ trace W a) ∨
    (∃ j ∈ doubledCoords W, e.1 = ((j, false), (j, true)) ∧
      (∃ a ∈ C, a j = false) ∧ (∃ b ∈ C, b j = true)) := by
  by_cases hcon : Consistent ({e.1.1, e.1.2} : Finset (Lit n))
  · left
    rcases consistent_face_subset_trace e.2.2 hcon with ⟨a, ha, hsub⟩
    exact ⟨a, ha, hsub (by simp), hsub (by simp)⟩
  · right
    unfold Consistent at hcon
    push_neg at hcon
    rcases hcon with ⟨j, ht, hf⟩
    simp only [Finset.mem_insert, Finset.mem_singleton] at ht hf
    have he_eq : e.1 = ((j, false), (j, true)) := by
      have hlt := e.2.1
      rcases ht with ht1 | ht2 <;> rcases hf with hf1 | hf2
      · exfalso; rw [← ht1] at hf1; exact Bool.false_ne_true (congrArg Prod.snd hf1)
      · exfalso; rw [← ht1, ← hf2] at hlt; simp [litIdx] at hlt
      · exact Prod.ext hf1.symm ht2.symm
      · exfalso; rw [← ht2] at hf2; exact Bool.false_ne_true (congrArg Prod.snd hf2)
    have hjW : j ∈ doubledCoords W := by
      refine Finset.mem_filter.mpr ⟨Finset.mem_univ _, ?_, ?_⟩
      · exact e.2.2.1 (by rw [show e.1.2 = (j, true) from congrArg Prod.snd he_eq]; simp)
      · exact e.2.2.1 (by rw [show e.1.1 = (j, false) from congrArg Prod.fst he_eq]; simp)
    have ha : ∃ a ∈ C, a j = false := by
      have hsub : {(j, false)} ⊆ ({e.1.1, e.1.2} : Finset (Lit n)) := by
        rw [show e.1.1 = (j, false) from congrArg Prod.fst he_eq]; simp
      have hcon1 : Consistent ({(j, false)} : Finset (Lit n)) := by
        intro i ⟨h1, h2⟩; simp at h1 h2
      rcases e.2.2.2 _ hsub hcon1 with ⟨a, ha, hsat⟩
      exact ⟨a, ha, hsat (j, false) (by simp)⟩
    have hb : ∃ b ∈ C, b j = true := by
      have hsub : {(j, true)} ⊆ ({e.1.1, e.1.2} : Finset (Lit n)) := by
        rw [show e.1.2 = (j, true) from congrArg Prod.snd he_eq]; simp
      have hcon1 : Consistent ({(j, true)} : Finset (Lit n)) := by
        intro i ⟨h1, h2⟩; simp at h1 h2
      rcases e.2.2.2 _ hsub hcon1 with ⟨b, hb, hsat⟩
      exact ⟨b, hb, hsat (j, true) (by simp)⟩
    exact ⟨j, hjW, he_eq, ha, hb⟩

/-- `h = 2` branch of Theorem 3: `H1Dim k C W = (doubledCoords W).card - 1`. -/
private lemma h1_strand_dim_h2 {n : ℕ} {C : Finset (Word n)} {W : Finset (Lit n)}
    (hd : 0 < (doubledCoords W).card)
    (hdisc : ∃ a ∈ C, ∃ b ∈ C, ¬ traceReach C W a b) :
    H1Dim k C W = (doubledCoords W).card - 1 := by
  classical
  obtain ⟨j₀, hj₀⟩ := Finset.card_pos.mp hd
  have hdich := trace_sharing_dichotomy (C := C) hj₀
  have hedge_dich := @kedge_dichotomy n C W
  rcases hdisc with ⟨a₁, ha₁, a₂, ha₂, hna12⟩
  have hcomp := hdich.2.2 a₁ ha₁ a₂ ha₂ hna12
  -- Every doubled coord j has opposite values on the two components, so every opposite edge
  -- ((j, false), (j, true)) IS an edge of K(C, W)!
  have hopp_edge : ∀ j ∈ doubledCoords W, isFace C W {(j, false), (j, true)} := by
    intro j hj
    have hjW := (Finset.mem_filter.mp hj).2
    refine ⟨fun x hx => by simp at hx; rcases hx with rfl | rfl <;> simp [hjW], fun A hA hcon => ?_⟩
    by_cases hf : (j, false) ∈ A
    · have hnt : (j, true) ∉ A := fun ht => hcon j ⟨ht, hf⟩
      have ha2j := (hcomp j hj a₂ ha₂).2 hna12 |>.1
      cases h1 : a₁ j
      · exact ⟨a₁, ha₁, fun l hl => by
          have := hA hl; simp at this
          rcases this with rfl | rfl
          · exact h1
          · exact (hnt hl).elim⟩
      · exact ⟨a₂, ha₂, fun l hl => by
          have := hA hl; simp at this
          rcases this with rfl | rfl
          · rw [ha2j, h1]; rfl
          · exact (hnt hl).elim⟩
    · have ha2j := (hcomp j hj a₂ ha₂).2 hna12 |>.1
      cases h1 : a₁ j
      · exact ⟨a₂, ha₂, fun l hl => by
          have := hA hl; simp at this
          rcases this with rfl | rfl
          · exact (hf hl).elim
          · rw [ha2j, h1]; rfl⟩
      · exact ⟨a₁, ha₁, fun l hl => by
          have := hA hl; simp at this
          rcases this with rfl | rfl
          · exact (hf hl).elim
          · exact h1⟩
  -- Vertex membership in side 1 vs side 2
  let inSide1 (v : Lit n) : Prop := ∃ c ∈ C, traceReach C W a₁ c ∧ v ∈ trace W c
  let inSide2 (v : Lit n) : Prop := ∃ c ∈ C, ¬ traceReach C W a₁ c ∧ v ∈ trace W c
  -- Every vertex of K(C, W) is in Side 1 XOR Side 2!
  have hdisj_sides : ∀ v : Lit n, ¬ (inSide1 v ∧ inSide2 v) := by
    rintro v ⟨⟨c₁, hc₁, hr₁, hv₁⟩, ⟨c₂, hc₂, hnr₂, hv₂⟩⟩
    have hadj : traceAdj W c₁ c₂ := ⟨v, Finset.mem_inter.mpr ⟨hv₁, hv₂⟩⟩
    exact hnr₂ (Relation.ReflTransGen.tail hr₁ ⟨hc₁, hc₂, hadj⟩)
  have hvert_side : ∀ v : KVerts C W, inSide1 v.1 ∨ inSide2 v.1 := by
    intro v
    have hcon : Consistent ({v.1} : Finset (Lit n)) := by
      intro i ⟨h1, h2⟩
      simp only [Finset.mem_singleton] at h1 h2
      exact Bool.false_ne_true ((congrArg Prod.snd h2).trans (congrArg Prod.snd h1).symm)
    rcases consistent_face_subset_trace v.2 hcon with ⟨c, hc, hsub⟩
    by_cases hr : traceReach C W a₁ c
    · left; exact ⟨c, hc, hr, hsub (by simp)⟩
    · right; exact ⟨c, hc, hr, hsub (by simp)⟩
  -- Key structural lemma: NO triangle in K(C, W) contains an opposite edge ((j, false), (j, true))!
  have hno_mixed_tri : ∀ t : KTris C W, ∀ j ∈ doubledCoords W,
      ¬ ((j, false) ∈ ({t.1.1, t.1.2.1, t.1.2.2} : Finset (Lit n)) ∧
         (j, true) ∈ ({t.1.1, t.1.2.1, t.1.2.2} : Finset (Lit n))) := by
    intro t j hj ⟨hf, ht⟩
    -- Pick the third vertex u ≠ (j, false), (j, true)
    have h3 : ∃ u ∈ ({t.1.1, t.1.2.1, t.1.2.2} : Finset (Lit n)),
        u ≠ (j, false) ∧ u ≠ (j, true) := by
      have hlt1 := t.2.1; have hlt2 := t.2.2.1
      have hne12 : t.1.1 ≠ t.1.2.1 := fun h => by rw [h] at hlt1; exact lt_irrefl _ hlt1
      have hne23 : t.1.2.1 ≠ t.1.2.2 := fun h => by rw [h] at hlt2; exact lt_irrefl _ hlt2
      have hne13 : t.1.1 ≠ t.1.2.2 := fun h => by rw [h] at hlt1; omega
      by_cases h1 : t.1.1 ≠ (j, false) ∧ t.1.1 ≠ (j, true)
      · exact ⟨t.1.1, by simp, h1⟩
      · by_cases h2 : t.1.2.1 ≠ (j, false) ∧ t.1.2.1 ≠ (j, true)
        · exact ⟨t.1.2.1, by simp, h2⟩
        · refine ⟨t.1.2.2, by simp, ?_⟩
          push_neg at h1 h2
          by_cases ha : t.1.1 = (j, false)
          · have hb : t.1.2.1 = (j, true) := by
              by_cases h2f : t.1.2.1 = (j, false)
              · exact (hne12 (ha.trans h2f.symm)).elim
              · exact h2 h2f
            exact ⟨fun h => hne13 (ha.trans h.symm), fun h => hne23 (hb.trans h.symm)⟩
          · have ha' : t.1.1 = (j, true) := h1 ha
            have hb : t.1.2.1 = (j, false) := by
              by_cases h2f : t.1.2.1 = (j, false)
              · exact h2f
              · exact (hne12 (ha'.trans (h2 h2f).symm)).elim
            exact ⟨fun h => hne23 (hb.trans h.symm), fun h => hne13 (ha'.trans h.symm)⟩
    rcases h3 with ⟨u, hu_mem, huf, hut⟩
    have huj : u.1 ≠ j := fun h => by
      cases hu2 : u.2
      · exact huf (Prod.ext h hu2)
      · exact hut (Prod.ext h hu2)
    -- Both {u, (j, false)} and {u, (j, true)} are consistent faces!
    have hcf : Consistent ({u, (j, false)} : Finset (Lit n)) := by
      intro i ⟨h1, h2⟩; simp only [Finset.mem_insert, Finset.mem_singleton] at h1 h2
      rcases h1 with h1 | h1 <;> rcases h2 with h2 | h2
      · rw [← h1] at h2; exact Bool.false_ne_true (congrArg Prod.snd h2)
      · exact huj ((congrArg Prod.fst h1).symm.trans (congrArg Prod.fst h2))
      · exact Bool.false_ne_true (congrArg Prod.snd h1).symm
      · exact Bool.false_ne_true (congrArg Prod.snd h1).symm
    have hct : Consistent ({u, (j, true)} : Finset (Lit n)) := by
      intro i ⟨h1, h2⟩; simp only [Finset.mem_insert, Finset.mem_singleton] at h1 h2
      rcases h1 with h1 | h1 <;> rcases h2 with h2 | h2
      · rw [← h1] at h2; exact Bool.false_ne_true (congrArg Prod.snd h2)
      · exact Bool.false_ne_true (congrArg Prod.snd h2)
      · exact huj ((congrArg Prod.fst h2).symm.trans (congrArg Prod.fst h1))
      · exact Bool.false_ne_true (congrArg Prod.snd h2)
    have hsubf : ({u, (j, false)} : Finset (Lit n)) ⊆ {t.1.1, t.1.2.1, t.1.2.2} := by
      intro x hx; simp at hx; rcases hx with rfl | rfl <;> assumption
    have hsubt : ({u, (j, true)} : Finset (Lit n)) ⊆ {t.1.1, t.1.2.1, t.1.2.2} := by
      intro x hx; simp at hx; rcases hx with rfl | rfl <;> assumption
    rcases consistent_face_subset_trace (isFace_subset t.2.2.2 hsubf) hcf with ⟨cf, hcf_in, hsub_cf⟩
    rcases consistent_face_subset_trace (isFace_subset t.2.2.2 hsubt) hct with ⟨ct, hct_in, hsub_ct⟩
    -- cf and ct share u, so they are in the same component, contradicting cf j = false ≠ true = ct j!
    have hadj : traceAdj W cf ct := ⟨u, Finset.mem_inter.mpr ⟨hsub_cf (by simp), hsub_ct (by simp)⟩⟩
    have hcf_mem : (j, false) ∈ trace W cf := hsub_cf (by simp)
    have hct_mem : (j, true) ∈ trace W ct := hsub_ct (by simp)
    have hcfj : cf j = false := (Finset.mem_filter.mp hcf_mem).2
    have hctj : ct j = true := (Finset.mem_filter.mp hct_mem).2
    by_cases hrcf : traceReach C W a₁ cf
    · have hrct : traceReach C W a₁ ct := Relation.ReflTransGen.tail hrcf ⟨hcf_in, hct_in, hadj⟩
      have e1 := (hcomp j hj cf hcf_in).1 hrcf
      have e2 := (hcomp j hj ct hct_in).1 hrct
      rw [hcfj] at e1; rw [hctj] at e2; exact Bool.false_ne_true (e1.trans e2.symm)
    · have hnrct : ¬ traceReach C W a₁ ct := fun hrct =>
        hrcf (Relation.ReflTransGen.trans hrct (traceReach_symm (Relation.ReflTransGen.single ⟨hcf_in, hct_in, hadj⟩)))
      have e1 := (hcomp j hj cf hcf_in).2 hrcf |>.1
      have e2 := (hcomp j hj ct hct_in).2 hnrct |>.1
      rw [hcfj] at e1; rw [hctj] at e2; exact Bool.false_ne_true (e1.trans e2.symm)
  -- Sign orientation of bridge j: +1 if a₁ j = false (so (j, false) ∈ V₁, (j, true) ∈ V₂), else -1
  let bridgeSign (j : Fin n) : k := if a₁ j = false then 1 else -1
  let bridgeEdge (j : Fin n) (hj : j ∈ doubledCoords W) : KEdges C W :=
    ⟨((j, false), (j, true)), by simp [litIdx], hopp_edge j hj⟩
  -- Define the target zero-sum subspace Z = {f : doubledCoords W → k | ∑ f = 0}
  let Z : Submodule k (doubledCoords W → k) :=
    LinearMap.ker (LinearMap.proj (R := k) (φ := fun _ : Unit => k) () ∘ₗ
      { toFun := fun f _ => ∑ j : doubledCoords W, f j
        map_add' := fun f g => by ext; simp [Finset.sum_add_distrib]
        map_smul' := fun r f => by ext; simp [Finset.mul_sum] })
  -- Flow functional Φ : C₁(K; k) →ₗ[k] (doubledCoords W → k) reading signed bridge coefficients
  let Φ : (KEdges C W → k) →ₗ[k] (doubledCoords W → k) :=
    { toFun := fun z j => bridgeSign j.1 * z (bridgeEdge j.1 j.2)
      map_add' := fun z₁ z₂ => by ext j; simp [mul_add]
      map_smul' := fun r z => by ext j; simp; ring }
  -- 1. Φ vanishes on im ∂₂ (since no triangle contains any bridgeEdge)
  have hΦ_bdry2 : ∀ t : KTris C W, ∀ j : doubledCoords W,
      (bridgeEdge j.1 j.2).1 ≠ (t.1.2.1, t.1.2.2) ∧
      (bridgeEdge j.1 j.2).1 ≠ (t.1.1, t.1.2.2) ∧
      (bridgeEdge j.1 j.2).1 ≠ (t.1.1, t.1.2.1) := by
    intro t j
    have hnm := hno_mixed_tri t j.1 j.2
    refine ⟨fun h => hnm ⟨?_, ?_⟩, fun h => hnm ⟨?_, ?_⟩, fun h => hnm ⟨?_, ?_⟩⟩
    · rw [show t.1.2.1 = (j.1, false) from (congrArg Prod.fst h).symm]; simp
    · rw [show t.1.2.2 = (j.1, true) from (congrArg Prod.snd h).symm]; simp
    · rw [show t.1.1 = (j.1, false) from (congrArg Prod.fst h).symm]; simp
    · rw [show t.1.2.2 = (j.1, true) from (congrArg Prod.snd h).symm]; simp
    · rw [show t.1.1 = (j.1, false) from (congrArg Prod.fst h).symm]; simp
    · rw [show t.1.2.1 = (j.1, true) from (congrArg Prod.snd h).symm]; simp
  have hΦ_im : LinearMap.range (bdry2 (k := k) C W) ≤ LinearMap.ker Φ := by
    rintro _ ⟨c, rfl⟩
    rw [LinearMap.mem_ker]
    ext j
    simp only [Φ, bdry2, LinearMap.coe_mk, AddHom.coe_mk, Pi.zero_apply]
    have h0 : ∑ t : KTris C W,
        ((if (bridgeEdge j.1 j.2).1 = (t.1.2.1, t.1.2.2) then c t else 0) -
         (if (bridgeEdge j.1 j.2).1 = (t.1.1, t.1.2.2) then c t else 0) +
         (if (bridgeEdge j.1 j.2).1 = (t.1.1, t.1.2.1) then c t else 0)) = 0 := by
      refine Finset.sum_eq_zero fun t _ => ?_
      rcases hΦ_bdry2 t j with ⟨h1, h2, h3⟩
      simp [h1, h2, h3]
    rw [h0, mul_zero]
  -- 2. For every z ∈ ker ∂₁, ∑_{j ∈ doubledCoords W} Φ(z)_j = 0!
  --    Proof: sum (∂₁ z)(v) over all vertices v in Side 1!
  have hΦ_ker_sum : ∀ z ∈ LinearMap.ker (bdry1 (k := k) C W), ∑ j : doubledCoords W, Φ z j = 0 := by
    intro z hz
    rw [LinearMap.mem_ker] at hz
    have hsum_side1 : ∑ v : KVerts C W, (if inSide1 v.1 then (bdry1 C W z) v else 0) = 0 := by
      refine Finset.sum_eq_zero fun v _ => ?_; rw [hz]; simp
    -- Swap sum over v and sum over e
    have hswap : ∑ v : KVerts C W, (if inSide1 v.1 then (bdry1 C W z) v else 0) =
        ∑ e : KEdges C W, z e *
          ((if inSide1 e.1.2 then (1 : k) else 0) - (if inSide1 e.1.1 then 1 else 0)) := by
      simp only [bdry1, LinearMap.coe_mk, AddHom.coe_mk]
      have h1 : ∀ v : KVerts C W,
          (if inSide1 v.1 then ∑ e : KEdges C W, ((if e.1.2 = v.1 then z e else 0) - (if e.1.1 = v.1 then z e else 0)) else 0) =
          ∑ e : KEdges C W, (if inSide1 v.1 then (if e.1.2 = v.1 then z e else 0) - (if e.1.1 = v.1 then z e else 0) else 0) := by
        intro v; split_ifs <;> simp
      simp_rw [h1]
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun e _ => ?_
      have hv1 : isFace C W {e.1.1} := isFace_subset e.2.2 (by intro x hx; simp at hx ⊢; left; exact hx)
      have hv2 : isFace C W {e.1.2} := isFace_subset e.2.2 (by intro x hx; simp at hx ⊢; right; exact hx)
      have hsum_v : ∀ u : Lit n, (hu : isFace C W {u}) →
          ∑ v : KVerts C W, (if inSide1 v.1 then if u = v.1 then z e else 0 else 0) =
            if inSide1 u then z e else 0 := by
        intro u hu
        rw [Finset.sum_eq_single ⟨u, hu⟩]
        · simp
        · intro v _ hv
          have hne : u ≠ v.1 := fun h => hv (Subtype.ext h.symm)
          simp [hne]
        · intro h; exact (h (Finset.mem_univ _)).elim
      have hsplit : ∑ v : KVerts C W,
          (if inSide1 v.1 then (if e.1.2 = v.1 then z e else 0) - (if e.1.1 = v.1 then z e else 0) else 0) =
          (∑ v : KVerts C W, if inSide1 v.1 then if e.1.2 = v.1 then z e else 0 else 0) -
          (∑ v : KVerts C W, if inSide1 v.1 then if e.1.1 = v.1 then z e else 0 else 0) := by
        rw [← Finset.sum_sub_distrib]
        refine Finset.sum_congr rfl fun v _ => by split_ifs <;> ring
      rw [hsplit, hsum_v e.1.2 hv2, hsum_v e.1.1 hv1]
      split_ifs <;> ring
    rw [hswap] at hsum_side1
    -- Each edge e contributes: 0 if both endpoints in same side; -Φ(z)_j if e = bridgeEdge j!
    have hcontrib : ∑ e : KEdges C W, z e *
          ((if inSide1 e.1.2 then (1 : k) else 0) - (if inSide1 e.1.1 then 1 else 0)) =
        -(∑ j : doubledCoords W, Φ z j) := by
      rw [← Finset.sum_neg_distrib]
      -- Partition KEdges into bridge edges (image of doubledCoords W) and non-bridge edges
      have hinj : Function.Injective (fun j : doubledCoords W => bridgeEdge j.1 j.2) := by
        intro j₁ j₂ h
        have : (j₁.1, false) = (j₂.1, false) := congrArg (fun x => x.1.1) h
        exact Subtype.ext (congrArg Prod.fst this)
      let img := Finset.univ.image (fun j : doubledCoords W => bridgeEdge j.1 j.2)
      have hzero_out : ∀ e : KEdges C W, e ∉ img →
          z e * ((if inSide1 e.1.2 then (1 : k) else 0) - (if inSide1 e.1.1 then 1 else 0)) = 0 := by
        intro e he_not
        rcases hedge_dich e with ⟨a, ha, h1, h2⟩ | ⟨j, hj, heq, _, _⟩
        · -- Both endpoints in trace W a, so both inSide1 or both not inSide1!
          by_cases hr : traceReach C W a₁ a
          · have s1 : inSide1 e.1.1 := ⟨a, ha, hr, h1⟩
            have s2 : inSide1 e.1.2 := ⟨a, ha, hr, h2⟩
            simp [s1, s2]
          · have s1 : ¬ inSide1 e.1.1 := fun hs => hdisj_sides _ ⟨hs, ⟨a, ha, hr, h1⟩⟩
            have s2 : ¬ inSide1 e.1.2 := fun hs => hdisj_sides _ ⟨hs, ⟨a, ha, hr, h2⟩⟩
            simp [s1, s2]
        · exfalso; apply he_not
          exact Finset.mem_image.mpr ⟨⟨j, hj⟩, Finset.mem_univ _, Subtype.ext heq.symm⟩
      rw [← Finset.sum_subset (Finset.subset_univ img) (fun e _ hne => hzero_out e hne)]
      rw [Finset.sum_image (fun j₁ _ j₂ _ h => hinj h)]
      refine Finset.sum_congr rfl fun j _ => ?_
      have hjW := (Finset.mem_filter.mp j.2).2
      have ha2j := (hcomp j.1 j.2 a₂ ha₂).2 hna12 |>.1
      simp only [Φ, LinearMap.coe_mk, AddHom.coe_mk, bridgeEdge, bridgeSign]
      cases h1 : a₁ j.1
      · have sf : inSide1 (j.1, false) := ⟨a₁, ha₁, Relation.ReflTransGen.refl,
          Finset.mem_filter.mpr ⟨hjW.2, h1⟩⟩
        have st : ¬ inSide1 (j.1, true) := fun hs =>
          hdisj_sides _ ⟨hs, ⟨a₂, ha₂, hna12, Finset.mem_filter.mpr ⟨hjW.1, by rw [ha2j, h1]; rfl⟩⟩⟩
        simp [sf, st]
      · have st : inSide1 (j.1, true) := ⟨a₁, ha₁, Relation.ReflTransGen.refl,
          Finset.mem_filter.mpr ⟨hjW.1, h1⟩⟩
        have sf : ¬ inSide1 (j.1, false) := fun hs =>
          hdisj_sides _ ⟨hs, ⟨a₂, ha₂, hna12, Finset.mem_filter.mpr ⟨hjW.2, by rw [ha2j, h1]; rfl⟩⟩⟩
        simp [sf, st]
    rw [hcontrib, neg_eq_zero] at hsum_side1
    exact hsum_side1
  -- 3. Cone contractibility on each side: if z ∈ ker ∂₁ has Φ z = 0, then z ∈ im ∂₂!
  have hΦ_inj : ∀ z ∈ LinearMap.ker (bdry1 (k := k) C W), Φ z = 0 →
      z ∈ LinearMap.range (bdry2 (k := k) C W) := by
    intro z hz hΦz
    -- Apexes v₁ = (j₀, a₁ j₀) on Side 1 and v₂ = (j₀, a₂ j₀) on Side 2
    let apex (v : Lit n) : Lit n := if inSide1 v then (j₀, a₁ j₀) else (j₀, a₂ j₀)
    have hj₀W := (Finset.mem_filter.mp hj₀).2
    have hcone_tri : ∀ e : KEdges C W, z e ≠ 0 →
        isFace C W {e.1.1, e.1.2, apex e.1.1} ∧ apex e.1.1 = apex e.1.2 := by
      intro e hze
      rcases hedge_dich e with ⟨a, ha, h1, h2⟩ | ⟨j, hj, heq, _, _⟩
      · by_cases hr : traceReach C W a₁ a
        · have s1 : inSide1 e.1.1 := ⟨a, ha, hr, h1⟩
          have s2 : inSide1 e.1.2 := ⟨a, ha, hr, h2⟩
          have haj : a j₀ = a₁ j₀ := (hcomp j₀ hj₀ a ha).1 hr
          have ha_v : (j₀, a₁ j₀) ∈ trace W a :=
            Finset.mem_filter.mpr ⟨by cases a₁ j₀ <;> simp [hj₀W], haj⟩
          refine ⟨isFace_subset (trace_isFace ha) (by
            intro x hx; simp only [apex, if_pos s1, Finset.mem_insert, Finset.mem_singleton] at hx
            rcases hx with rfl | rfl | rfl <;> assumption), by simp [apex, s1, s2]⟩
        · have s1 : ¬ inSide1 e.1.1 := fun hs => hdisj_sides _ ⟨hs, ⟨a, ha, hr, h1⟩⟩
          have s2 : ¬ inSide1 e.1.2 := fun hs => hdisj_sides _ ⟨hs, ⟨a, ha, hr, h2⟩⟩
          have haj : a j₀ = !a₁ j₀ := (hcomp j₀ hj₀ a ha).2 hr |>.1
          have ha2j : a₂ j₀ = !a₁ j₀ := (hcomp j₀ hj₀ a₂ ha₂).2 hna12 |>.1
          have ha_v : (j₀, a₂ j₀) ∈ trace W a :=
            Finset.mem_filter.mpr ⟨by cases a₂ j₀ <;> simp [hj₀W], haj.trans ha2j.symm⟩
          refine ⟨isFace_subset (trace_isFace ha) (by
            intro x hx; simp only [apex, if_neg s1, Finset.mem_insert, Finset.mem_singleton] at hx
            rcases hx with rfl | rfl | rfl <;> assumption), by simp [apex, s1, s2]⟩
      · exfalso
        have h0 := congrFun hΦz ⟨j, hj⟩
        simp only [Φ, LinearMap.coe_mk, AddHom.coe_mk, Pi.zero_apply, bridgeSign] at h0
        have he_eq : e = bridgeEdge j hj := Subtype.ext heq
        rw [← he_eq] at h0
        have hsne : (if a₁ j = false then (1 : k) else -1) ≠ 0 := by split_ifs <;> norm_num
        exact hze ((mul_eq_zero.mp h0).resolve_left hsne)
    -- Write z = ∑ e, z e • (edgeChain e.1.1 e.1.2 + edgeChain e.1.2 (apex e.1.1) + edgeChain (apex e.1.1) e.1.1)
    -- minus ∑ e, z e • (edgeChain e.1.2 (apex e.1.2) - edgeChain e.1.1 (apex e.1.1))!
    -- The second sum is 0 because z ∈ ker ∂₁!
    have hcancel : ∑ e : KEdges C W,
        z e • (edgeChain (k := k) C W e.1.2 (apex e.1.2) - edgeChain C W e.1.1 (apex e.1.1)) = 0 := by
      -- Rewrite by grouping over vertices v ∈ KVerts C W:
      have hv_eq : ∀ e : KEdges C W,
          z e • (edgeChain (k := k) C W e.1.2 (apex e.1.2) - edgeChain C W e.1.1 (apex e.1.1)) =
          ∑ v : KVerts C W,
            ((if e.1.2 = v.1 then z e else 0) - (if e.1.1 = v.1 then z e else 0)) •
              edgeChain C W v.1 (apex v.1) := by
        intro e
        have hv1 : isFace C W {e.1.1} := isFace_subset e.2.2 (by intro x hx; simp at hx ⊢; left; exact hx)
        have hv2 : isFace C W {e.1.2} := isFace_subset e.2.2 (by intro x hx; simp at hx ⊢; right; exact hx)
        have hne : e.1.1 ≠ e.1.2 := fun h => by have := e.2.1; rw [h] at this; exact lt_irrefl _ this
        have hs1 : ∑ v : KVerts C W, (if e.1.2 = v.1 then z e else 0) • edgeChain (k := k) C W v.1 (apex v.1) =
            z e • edgeChain C W e.1.2 (apex e.1.2) := by
          rw [Finset.sum_eq_single ⟨e.1.2, hv2⟩]
          · simp
          · intro v _ hv; have : e.1.2 ≠ v.1 := fun h => hv (Subtype.ext h.symm); simp [this]
          · intro h; exact (h (Finset.mem_univ _)).elim
        have hs2 : ∑ v : KVerts C W, (if e.1.1 = v.1 then z e else 0) • edgeChain (k := k) C W v.1 (apex v.1) =
            z e • edgeChain C W e.1.1 (apex e.1.1) := by
          rw [Finset.sum_eq_single ⟨e.1.1, hv1⟩]
          · simp
          · intro v _ hv; have : e.1.1 ≠ v.1 := fun h => hv (Subtype.ext h.symm); simp [this]
          · intro h; exact (h (Finset.mem_univ _)).elim
        simp_rw [sub_smul, Finset.sum_sub_distrib, hs1, hs2, smul_sub]
      simp_rw [hv_eq]
      rw [Finset.sum_comm]
      refine Finset.sum_eq_zero fun v _ => ?_
      rw [← Finset.sum_smul]
      rw [LinearMap.mem_ker] at hz
      have hzv : (bdry1 (k := k) C W z) v = 0 := congrFun hz v
      simp only [bdry1, LinearMap.coe_mk, AddHom.coe_mk] at hzv
      rw [hzv, zero_smul]
    have hz_eq : z = ∑ e : KEdges C W,
        z e • (edgeChain (k := k) C W e.1.1 e.1.2 +
               edgeChain C W e.1.2 (apex e.1.2) +
               edgeChain C W (apex e.1.1) e.1.1) := by
      calc z = ∑ e : KEdges C W, z e • edgeChain (k := k) C W e.1.1 e.1.2 := chain_eq_sum_edgeChain z
        _ = (∑ e : KEdges C W, z e • edgeChain C W e.1.1 e.1.2) +
            (∑ e : KEdges C W, z e • (edgeChain C W e.1.2 (apex e.1.2) - edgeChain C W e.1.1 (apex e.1.1))) := by
            rw [hcancel, add_zero]
        _ = ∑ e : KEdges C W, z e • (edgeChain C W e.1.1 e.1.2 +
              edgeChain C W e.1.2 (apex e.1.2) + edgeChain C W (apex e.1.1) e.1.1) := by
            rw [← Finset.sum_add_distrib]
            refine Finset.sum_congr rfl fun e _ => ?_
            rw [← smul_add, edgeChain_swap C W (apex e.1.1) e.1.1]; abel_nf
    rw [hz_eq]
    refine Submodule.sum_mem _ fun e _ => ?_
    by_cases hze : z e = 0
    · rw [hze, zero_smul]; exact Submodule.zero_mem _
    · rcases hcone_tri e hze with ⟨hface, hap⟩
      rw [← hap]
      exact Submodule.smul_mem _ _ (triangle_boundary_mem_range hface)
  -- 4. Surjectivity of Φ onto Z = {f : doubledCoords W → k | ∑ f = 0}:
  --    For each j ∈ doubledCoords W, build the explicit 4-cycle z_j ∈ ker ∂₁ with Φ(z_j) = δ_j - δ_{j₀}!
  have hΦ_surj : ∀ f : doubledCoords W → k, (∑ j : doubledCoords W, f j = 0) →
      ∃ z ∈ LinearMap.ker (bdry1 (k := k) C W), Φ z = f := by
    intro f hf
    have hcycle : ∀ j : doubledCoords W,
        ∃ z ∈ LinearMap.ker (bdry1 (k := k) C W),
          Φ z = fun j' => (if j' = j then (1 : k) else 0) - (if j' = ⟨j₀, hj₀⟩ then 1 else 0) := by
      intro j
      -- 4-cycle: [(j, a₁ j), (j, a₂ j)] + [(j, a₂ j), (j₀, a₂ j₀)] + [(j₀, a₂ j₀), (j₀, a₁ j₀)] + [(j₀, a₁ j₀), (j, a₁ j)]
      let z_j : KEdges C W → k :=
        edgeChain C W (j.1, a₁ j.1) (j.1, a₂ j.1) +
        edgeChain C W (j.1, a₂ j.1) (j₀, a₂ j₀) +
        edgeChain C W (j₀, a₂ j₀) (j₀, a₁ j₀) +
        edgeChain C W (j₀, a₁ j₀) (j.1, a₁ j.1)
      have hjW := (Finset.mem_filter.mp j.2).2
      have hj₀W := (Finset.mem_filter.mp hj₀).2
      have ha2j : a₂ j.1 = !a₁ j.1 := (hcomp j.1 j.2 a₂ ha₂).2 hna12 |>.1
      have ha2j₀ : a₂ j₀ = !a₁ j₀ := (hcomp j₀ hj₀ a₂ ha₂).2 hna12 |>.1
      have hf1 : isFace C W {(j.1, a₁ j.1), (j.1, a₂ j.1)} := by
        cases h1 : a₁ j.1
        · rw [ha2j, h1]; exact hopp_edge j.1 j.2
        · rw [ha2j, h1]; simpa [Finset.pair_comm] using hopp_edge j.1 j.2
      have hf2 : isFace C W {(j.1, a₂ j.1), (j₀, a₂ j₀)} :=
        isFace_subset (trace_isFace ha₂) (by
          intro x hx; simp at hx; rcases hx with rfl | rfl
          · exact Finset.mem_filter.mpr ⟨by cases a₂ j.1 <;> simp [hjW], rfl⟩
          · exact Finset.mem_filter.mpr ⟨by cases a₂ j₀ <;> simp [hj₀W], rfl⟩)
      have hf3 : isFace C W {(j₀, a₂ j₀), (j₀, a₁ j₀)} := by
        cases h1 : a₁ j₀
        · rw [ha2j₀, h1]; simpa [Finset.pair_comm] using hopp_edge j₀ hj₀
        · rw [ha2j₀, h1]; exact hopp_edge j₀ hj₀
      have hf4 : isFace C W {(j₀, a₁ j₀), (j.1, a₁ j.1)} :=
        isFace_subset (trace_isFace ha₁) (by
          intro x hx; simp at hx; rcases hx with rfl | rfl
          · exact Finset.mem_filter.mpr ⟨by cases a₁ j₀ <;> simp [hj₀W], rfl⟩
          · exact Finset.mem_filter.mpr ⟨by cases a₁ j.1 <;> simp [hjW], rfl⟩)
      have hker : z_j ∈ LinearMap.ker (bdry1 (k := k) C W) := by
        rw [LinearMap.mem_ker]
        simp only [z_j, map_add, bdry1_edgeChain hf1, bdry1_edgeChain hf2,
          bdry1_edgeChain hf3, bdry1_edgeChain hf4]
        ext v; simp
      refine ⟨z_j, hker, ?_⟩
      ext j'
      simp only [Φ, LinearMap.coe_mk, AddHom.coe_mk, z_j, Pi.add_apply, bridgeEdge, bridgeSign]
      -- Helper: edgeChain of two vertices in the same side vanishes on every bridgeEdge!
      have hsame_side_zero : ∀ u v : Lit n,
          (u ∈ trace W a₁ ∧ v ∈ trace W a₁) ∨ (u ∈ trace W a₂ ∧ v ∈ trace W a₂) →
          edgeChain (k := k) C W u v (bridgeEdge j'.1 j'.2) = 0 := by
        intro u v huv
        have hne1 : ((j'.1, false), (j'.1, true)) ≠ (u, v) := by
          intro h
          have h1 : (j'.1, false) = u := congrArg Prod.fst h
          have h2 : (j'.1, true) = v := congrArg Prod.snd h
          rcases huv with ⟨hu, hv⟩ | ⟨hu, hv⟩
          · have e1 : a₁ j'.1 = false := by simpa [← h1] using (Finset.mem_filter.mp hu).2
            have e2 : a₁ j'.1 = true := by simpa [← h2] using (Finset.mem_filter.mp hv).2
            exact Bool.false_ne_true (e1.symm.trans e2)
          · have e1 : a₂ j'.1 = false := by simpa [← h1] using (Finset.mem_filter.mp hu).2
            have e2 : a₂ j'.1 = true := by simpa [← h2] using (Finset.mem_filter.mp hv).2
            exact Bool.false_ne_true (e1.symm.trans e2)
        have hne2 : ((j'.1, false), (j'.1, true)) ≠ (v, u) := by
          intro h
          have h1 : (j'.1, false) = v := congrArg Prod.fst h
          have h2 : (j'.1, true) = u := congrArg Prod.snd h
          rcases huv with ⟨hu, hv⟩ | ⟨hu, hv⟩
          · have e1 : a₁ j'.1 = false := by simpa [← h1] using (Finset.mem_filter.mp hv).2
            have e2 : a₁ j'.1 = true := by simpa [← h2] using (Finset.mem_filter.mp hu).2
            exact Bool.false_ne_true (e1.symm.trans e2)
          · have e1 : a₂ j'.1 = false := by simpa [← h1] using (Finset.mem_filter.mp hv).2
            have e2 : a₂ j'.1 = true := by simpa [← h2] using (Finset.mem_filter.mp hu).2
            exact Bool.false_ne_true (e1.symm.trans e2)
        simp [edgeChain, bridgeEdge, hne1, hne2]
      have hzero2 : edgeChain (k := k) C W (j.1, a₂ j.1) (j₀, a₂ j₀) (bridgeEdge j'.1 j'.2) = 0 :=
        hsame_side_zero _ _ (Or.inr ⟨Finset.mem_filter.mpr ⟨by cases a₂ j.1 <;> simp [hjW], rfl⟩,
          Finset.mem_filter.mpr ⟨by cases a₂ j₀ <;> simp [hj₀W], rfl⟩⟩)
      have hzero4 : edgeChain (k := k) C W (j₀, a₁ j₀) (j.1, a₁ j.1) (bridgeEdge j'.1 j'.2) = 0 :=
        hsame_side_zero _ _ (Or.inl ⟨Finset.mem_filter.mpr ⟨by cases a₁ j₀ <;> simp [hj₀W], rfl⟩,
          Finset.mem_filter.mpr ⟨by cases a₁ j.1 <;> simp [hjW], rfl⟩⟩)
      have hval_j : (if a₁ j'.1 = false then (1 : k) else -1) *
          edgeChain (k := k) C W (j.1, a₁ j.1) (j.1, a₂ j.1) (bridgeEdge j'.1 j'.2) =
          if j' = j then 1 else 0 := by
        by_cases hjj : j' = j
        · subst hjj; simp only [↓reduceIte, edgeChain, bridgeEdge, ha2j]
          cases h1 : a₁ j'.1 <;> simp
        · have hne : j'.1 ≠ j.1 := fun h => hjj (Subtype.ext h)
          simp [edgeChain, bridgeEdge, hjj, hne]
      have hval_j0 : (if a₁ j'.1 = false then (1 : k) else -1) *
          edgeChain (k := k) C W (j₀, a₂ j₀) (j₀, a₁ j₀) (bridgeEdge j'.1 j'.2) =
          -(if j' = ⟨j₀, hj₀⟩ then 1 else 0) := by
        by_cases hjj : j' = ⟨j₀, hj₀⟩
        · subst hjj; simp only [↓reduceIte, edgeChain, bridgeEdge, ha2j₀]
          cases h1 : a₁ j₀ <;> simp
        · have hne : j'.1 ≠ j₀ := fun h => hjj (Subtype.ext h)
          simp [edgeChain, bridgeEdge, hjj, hne]
      change (if a₁ j'.1 = false then (1 : k) else -1) *
          (edgeChain C W (j.1, a₁ j.1) (j.1, a₂ j.1) (bridgeEdge j'.1 j'.2) +
           edgeChain C W (j.1, a₂ j.1) (j₀, a₂ j₀) (bridgeEdge j'.1 j'.2) +
           edgeChain C W (j₀, a₂ j₀) (j₀, a₁ j₀) (bridgeEdge j'.1 j'.2) +
           edgeChain C W (j₀, a₁ j₀) (j.1, a₁ j.1) (bridgeEdge j'.1 j'.2)) = _
      rw [hzero2, hzero4]
      calc (if a₁ j'.1 = false then (1 : k) else -1) *
            (edgeChain C W (j.1, a₁ j.1) (j.1, a₂ j.1) (bridgeEdge j'.1 j'.2) + 0 +
             edgeChain C W (j₀, a₂ j₀) (j₀, a₁ j₀) (bridgeEdge j'.1 j'.2) + 0)
          = (if a₁ j'.1 = false then (1 : k) else -1) *
              edgeChain C W (j.1, a₁ j.1) (j.1, a₂ j.1) (bridgeEdge j'.1 j'.2) +
            (if a₁ j'.1 = false then (1 : k) else -1) *
              edgeChain C W (j₀, a₂ j₀) (j₀, a₁ j₀) (bridgeEdge j'.1 j'.2) := by ring
        _ = (if j' = j then 1 else 0) - (if j' = ⟨j₀, hj₀⟩ then 1 else 0) := by
            rw [hval_j, hval_j0]; ring
    choose z_fam hz_ker hz_phi using hcycle
    refine ⟨∑ j : doubledCoords W, f j • z_fam j,
      Submodule.sum_mem _ fun j _ => Submodule.smul_mem _ _ (hz_ker j), ?_⟩
    rw [map_sum]
    ext j'
    simp only [Finset.sum_apply, map_smul, Pi.smul_apply, smul_eq_mul, hz_phi]
    calc ∑ j : doubledCoords W, f j * ((if j' = j then (1 : k) else 0) - (if j' = ⟨j₀, hj₀⟩ then 1 else 0))
        = (∑ j : doubledCoords W, f j * (if j' = j then 1 else 0)) -
          (∑ j : doubledCoords W, f j * (if j' = ⟨j₀, hj₀⟩ then 1 else 0)) := by
          simp_rw [mul_sub, Finset.sum_sub_distrib]
      _ = f j' - (if j' = ⟨j₀, hj₀⟩ then ∑ j : doubledCoords W, f j else 0) := by
          congr 1
          · rw [Finset.sum_eq_single j']
            · simp
            · intro j _ hj; simp [Ne.symm hj]
            · intro h; exact (h (Finset.mem_univ _)).elim
          · split_ifs <;> simp
      _ = f j' := by rw [hf]; simp
  -- 5. Assemble linear isomorphism H1Space k C W ≃ₗ[k] Z and compute finrank k Z = d - 1!
  let Φ_ker : LinearMap.ker (bdry1 (k := k) C W) →ₗ[k] Z :=
    LinearMap.codRestrict Z (Φ.comp (LinearMap.ker (bdry1 (k := k) C W)).subtype)
      (fun z => by
        show ∑ j : doubledCoords W, Φ z.1 j = 0
        exact hΦ_ker_sum z.1 z.2)
  have hΦ_ker_ker : LinearMap.ker Φ_ker =
      (LinearMap.range (bdry2 (k := k) C W)).comap (LinearMap.ker (bdry1 (k := k) C W)).subtype := by
    ext ⟨z, hz⟩
    simp only [LinearMap.mem_ker, Submodule.mem_comap, Submodule.coe_subtype, Φ_ker,
      LinearMap.codRestrict_apply, LinearMap.comp_apply, Subtype.ext_iff, ZeroMemClass.coe_zero]
    exact ⟨fun h => hΦ_inj z hz h, fun h => LinearMap.mem_ker.mp (hΦ_im h)⟩
  have hΦ_ker_surj : Function.Surjective Φ_ker := by
    rintro ⟨f, hf⟩
    have hf0 : ∑ j : doubledCoords W, f j = 0 := hf
    rcases hΦ_surj f hf0 with ⟨z, hz, hfz⟩
    exact ⟨⟨z, hz⟩, Subtype.ext hfz⟩
  let eIso : H1Space k C W ≃ₗ[k] Z :=
    (Submodule.quotEquivOfEq _ _ hΦ_ker_ker.symm).trans
      (LinearMap.quotKerEquivOfSurjective Φ_ker hΦ_ker_surj)
  unfold H1Dim
  rw [eIso.finrank_eq]
  -- Compute finrank k Z = (doubledCoords W).card - 1 via rank-nullity on the sum functional!
  let sumLin : (doubledCoords W → k) →ₗ[k] k :=
    { toFun := fun f => ∑ j : doubledCoords W, f j
      map_add' := fun f g => Finset.sum_add_distrib
      map_smul' := fun r f => by simp [Finset.mul_sum] }
  have hZ_eq : Z = LinearMap.ker sumLin := by
    ext f; rfl
  have hsum_surj : Function.Surjective sumLin := by
    intro c
    refine ⟨fun j => if j = ⟨j₀, hj₀⟩ then c else 0, ?_⟩
    simp only [sumLin, LinearMap.coe_mk, AddHom.coe_mk]
    rw [Finset.sum_eq_single ⟨j₀, hj₀⟩]
    · simp
    · intro j _ hj; simp [hj]
    · intro h; exact (h (Finset.mem_univ _)).elim
  have hrn := LinearMap.finrank_range_add_finrank_ker sumLin
  rw [LinearMap.range_eq_top.mpr hsum_surj, finrank_top, Module.finrank_self,
    Module.finrank_pi, Fintype.card_coe] at hrn
  rw [hZ_eq]
  omega

/-- `h = 1` branch of Theorem 3: `H1Dim k C W = 0`. -/
private lemma h1_strand_dim_h1 {n : ℕ} {C : Finset (Word n)} {W : Finset (Lit n)}
    (hd : 0 < (doubledCoords W).card)
    (hreach_all : ∀ a ∈ C, ∀ b ∈ C, traceReach C W a b) :
    H1Dim k C W = 0 := by
  classical
  obtain ⟨j₀, hj₀⟩ := Finset.card_pos.mp hd
  have hedge_dich := @kedge_dichotomy n C W
  have hcw_ex : ∀ v : KVerts C W, ∃ a ∈ C, v.1 ∈ trace W a := fun v =>
      let ⟨a, ha, hsub⟩ := consistent_face_subset_trace v.2 (by
        intro i ⟨h1, h2⟩
        simp only [Finset.mem_singleton] at h1 h2
        exact Bool.false_ne_true ((congrArg Prod.snd h2).trans (congrArg Prod.snd h1).symm))
      ⟨a, ha, hsub (by simp)⟩
  choose cw hcw_in hcw_tr using hcw_ex
  -- For any doubled coord j and codewords a, b ∈ C, define the 3-step chain
  -- P_j(a, b) = [(j₀, a j₀), (j, a j)] + [(j, a j), (j, b j)] + [(j, b j), (j₀, b j₀)].
  --Wait: P_j(a, b) - [(j₀, a j₀), (j₀, b j₀)] ∈ im ∂₂ for ALL a, b ∈ C and ALL j ∈ doubledCoords W!
  have hPj_indep : ∀ j ∈ doubledCoords W, ∀ a ∈ C, ∀ b ∈ C,
      edgeChain (k := k) C W (j₀, a j₀) (j, a j) +
      edgeChain C W (j, a j) (j, b j) +
      edgeChain C W (j, b j) (j₀, b j₀) -
      edgeChain C W (j₀, a j₀) (j₀, b j₀) ∈ LinearMap.range (bdry2 (k := k) C W) := by
    intro j hj a ha b hb
    have hjW := (Finset.mem_filter.mp hj).2
    have hj₀W := (Finset.mem_filter.mp hj₀).2
    -- First prove for a single trace-adjacent step (a, b) sharing literal w ∈ trace W a ∩ trace W b!
    have hstep : ∀ x ∈ C, ∀ y ∈ C, traceAdj W x y →
        edgeChain (k := k) C W (j₀, x j₀) (j, x j) +
        edgeChain C W (j, x j) (j, y j) +
        edgeChain C W (j, y j) (j₀, y j₀) -
        edgeChain C W (j₀, x j₀) (j₀, y j₀) ∈ LinearMap.range (bdry2 (k := k) C W) := by
      intro x hx y hy ⟨w, hw⟩
      rw [Finset.mem_inter] at hw
      -- Four triangles in K(C, W):
      -- T1 = {w, (j₀, x j₀), (j, x j)} ⊆ trace W x
      -- T2 = {w, (j, x j), (j, y j)} (shared_lit_doubled_triangle)
      -- T3 = {w, (j, y j), (j₀, y j₀)} ⊆ trace W y
      -- T4 = {w, (j₀, y j₀), (j₀, x j₀)} (shared_lit_doubled_triangle)
      have hT1 : isFace C W {w, (j₀, x j₀), (j, x j)} :=
        isFace_subset (trace_isFace hx) (by
          intro l hl; simp at hl; rcases hl with rfl | rfl | rfl
          · exact hw.1
          · exact Finset.mem_filter.mpr ⟨by cases x j₀ <;> simp [hj₀W], rfl⟩
          · exact Finset.mem_filter.mpr ⟨by cases x j <;> simp [hjW], rfl⟩)
      have hT2 : isFace C W {w, (j, x j), (j, y j)} :=
        shared_lit_doubled_triangle hx hy hw.1 hw.2 hj
      have hT3 : isFace C W {w, (j, y j), (j₀, y j₀)} :=
        isFace_subset (trace_isFace hy) (by
          intro l hl; simp at hl; rcases hl with rfl | rfl | rfl
          · exact hw.2
          · exact Finset.mem_filter.mpr ⟨by cases y j <;> simp [hjW], rfl⟩
          · exact Finset.mem_filter.mpr ⟨by cases y j₀ <;> simp [hj₀W], rfl⟩)
      have hT4 : isFace C W {w, (j₀, y j₀), (j₀, x j₀)} :=
        shared_lit_doubled_triangle hy hx hw.2 hw.1 hj₀
      have m1 := triangle_boundary_mem_range (k := k) hT1
      have m2 := triangle_boundary_mem_range (k := k) hT2
      have m3 := triangle_boundary_mem_range (k := k) hT3
      have m4 := triangle_boundary_mem_range (k := k) hT4
      have hsum := Submodule.add_mem _ (Submodule.add_mem _ (Submodule.add_mem _ m1 m2) m3) m4
      have heq :
          (edgeChain (k := k) C W w (j₀, x j₀) + edgeChain C W (j₀, x j₀) (j, x j) + edgeChain C W (j, x j) w) +
          (edgeChain C W w (j, x j) + edgeChain C W (j, x j) (j, y j) + edgeChain C W (j, y j) w) +
          (edgeChain C W w (j, y j) + edgeChain C W (j, y j) (j₀, y j₀) + edgeChain C W (j₀, y j₀) w) +
          (edgeChain C W w (j₀, y j₀) + edgeChain C W (j₀, y j₀) (j₀, x j₀) + edgeChain C W (j₀, x j₀) w) =
          edgeChain C W (j₀, x j₀) (j, x j) + edgeChain C W (j, x j) (j, y j) +
          edgeChain C W (j, y j) (j₀, y j₀) - edgeChain C W (j₀, x j₀) (j₀, y j₀) := by
        rw [edgeChain_swap C W (j, x j) w, edgeChain_swap C W (j, y j) w,
          edgeChain_swap C W (j₀, y j₀) w, edgeChain_swap C W (j₀, x j₀) w,
          edgeChain_swap C W (j₀, y j₀) (j₀, x j₀)]
        abel
      rwa [heq] at hsum
    -- Induct along traceReach C W a b!
    have hr : traceReach C W a b := hreach_all a ha b hb
    induction hr with
    | refl =>
      have h0 : edgeChain (k := k) C W (j₀, a j₀) (j, a j) +
          edgeChain C W (j, a j) (j, a j) +
          edgeChain C W (j, a j) (j₀, a j₀) -
          edgeChain C W (j₀, a j₀) (j₀, a j₀) = 0 := by
        rw [edgeChain_self, edgeChain_self, edgeChain_swap C W (j, a j) (j₀, a j₀)]; abel
      rw [h0]; exact Submodule.zero_mem _
    | @tail x y _ hxy ih =>
      rcases hxy with ⟨hx, hy, hadj⟩
      have h1 := ih hx
      have h2 := hstep x hx y hy hadj
      -- Adding h1 + h2 telescopes [(j, x j), (j₀, x j₀)] + [(j₀, x j₀), (j, x j)] = 0!
      -- Also on the 2-point set {(j, false), (j, true)},
      -- [p, q] + [q, r] - [p, r] = 0 identically for ANY 3 booleans p, q, r!
      have hbool3 : ∀ (coord : Fin n) (p q r : Bool),
          edgeChain (k := k) C W (coord, p) (coord, q) +
          edgeChain C W (coord, q) (coord, r) -
          edgeChain C W (coord, p) (coord, r) = 0 := by
        intro coord p q r
        cases p <;> cases q <;> cases r
        · rw [edgeChain_self]; abel
        · rw [edgeChain_self]; abel
        · rw [edgeChain_self, edgeChain_swap C W (coord, true) (coord, false)]; abel
        · rw [edgeChain_self]; abel
        · rw [edgeChain_self]; abel
        · rw [edgeChain_self, edgeChain_swap C W (coord, false) (coord, true)]; abel
        · rw [edgeChain_self]; abel
        · rw [edgeChain_self]; abel
      have hadd := Submodule.add_mem _ h1 h2
      have htel :
          (edgeChain (k := k) C W (j₀, a j₀) (j, a j) + edgeChain C W (j, a j) (j, x j) +
           edgeChain C W (j, x j) (j₀, x j₀) - edgeChain C W (j₀, a j₀) (j₀, x j₀)) +
          (edgeChain C W (j₀, x j₀) (j, x j) + edgeChain C W (j, x j) (j, y j) +
           edgeChain C W (j, y j) (j₀, y j₀) - edgeChain C W (j₀, x j₀) (j₀, y j₀)) =
          edgeChain C W (j₀, a j₀) (j, a j) + edgeChain C W (j, a j) (j, y j) +
          edgeChain C W (j, y j) (j₀, y j₀) - edgeChain C W (j₀, a j₀) (j₀, y j₀) := by
        have hb1 := hbool3 j (a j) (x j) (y j)
        have hb2 := hbool3 j₀ (a j₀) (x j₀) (y j₀)
        have hsw := edgeChain_swap (k := k) C W (j₀, x j₀) (j, x j)
        -- Combine linearly using abel on the module
        have : edgeChain (k := k) C W (j, a j) (j, x j) + edgeChain C W (j, x j) (j, y j) =
            edgeChain C W (j, a j) (j, y j) := eq_of_sub_eq_zero hb1
        have h2' : edgeChain (k := k) C W (j₀, a j₀) (j₀, x j₀) + edgeChain C W (j₀, x j₀) (j₀, y j₀) =
            edgeChain C W (j₀, a j₀) (j₀, y j₀) := eq_of_sub_eq_zero hb2
        have h3' : edgeChain (k := k) C W (j, x j) (j₀, x j₀) + edgeChain C W (j₀, x j₀) (j, x j) = 0 := by
          rw [hsw]; abel
        calc _ = edgeChain (k := k) C W (j₀, a j₀) (j, a j) +
              (edgeChain C W (j, a j) (j, x j) + edgeChain C W (j, x j) (j, y j)) +
              (edgeChain C W (j, x j) (j₀, x j₀) + edgeChain C W (j₀, x j₀) (j, x j)) +
              edgeChain C W (j, y j) (j₀, y j₀) -
              (edgeChain C W (j₀, a j₀) (j₀, x j₀) + edgeChain C W (j₀, x j₀) (j₀, y j₀)) := by abel
          _ = _ := by rw [this, h2', h3']; abel
      rwa [htel] at hadd
  -- Now prove EVERY edge e = (u, v) satisfies
  -- [u, v] - ([u, p(u)] + [p(u), p(v)] + [p(v), v]) ∈ im ∂₂!
  let pVert (v : KVerts C W) : Lit n := (j₀, cw v j₀)
  have hedge_homotopy : ∀ e : KEdges C W,
      let u : KVerts C W := ⟨e.1.1, isFace_subset e.2.2 (by intro x hx; simp at hx ⊢; left; exact hx)⟩
      let v : KVerts C W := ⟨e.1.2, isFace_subset e.2.2 (by intro x hx; simp at hx ⊢; right; exact hx)⟩
      edgeChain (k := k) C W e.1.1 e.1.2 -
        (edgeChain C W e.1.1 (pVert u) + edgeChain C W (pVert u) (pVert v) + edgeChain C W (pVert v) e.1.2) ∈
        LinearMap.range (bdry2 (k := k) C W) := by
    intro e u v
    have hj₀W := (Finset.mem_filter.mp hj₀).2
    -- Helper: for any vertex x : KVerts C W and any a ∈ C with x.1 ∈ trace W a,
    -- [x.1, pVert x] + [pVert x, (j₀, a j₀)] + [(j₀, a j₀), x.1] ∈ im ∂₂
    have hretarget : ∀ (x : KVerts C W) (a : Word n) (ha : a ∈ C), x.1 ∈ trace W a →
        edgeChain (k := k) C W x.1 (pVert x) +
        edgeChain C W (pVert x) (j₀, a j₀) +
        edgeChain C W (j₀, a j₀) x.1 ∈ LinearMap.range (bdry2 (k := k) C W) := by
      intro x a ha hxa
      exact triangle_boundary_mem_range (shared_lit_doubled_triangle (hcw_in x) ha (hcw_tr x) hxa hj₀)
    rcases hedge_dich e with ⟨a, ha, h1, h2⟩ | ⟨j, hj, heq, ⟨a, ha, haj⟩, ⟨b, hb, hbj⟩⟩
    · -- Trace edge {e.1.1, e.1.2} ⊆ trace W a
      have hT : isFace C W {e.1.1, e.1.2, (j₀, a j₀)} :=
        isFace_subset (trace_isFace ha) (by
          intro l hl; simp at hl; rcases hl with rfl | rfl | rfl
          · exact h1
          · exact h2
          · exact Finset.mem_filter.mpr ⟨by cases a j₀ <;> simp [hj₀W], rfl⟩)
      have mT := triangle_boundary_mem_range (k := k) hT
      have mU := hretarget u a ha h1
      have mV := hretarget v a ha h2
      have hbool3 : edgeChain (k := k) C W (pVert u) (j₀, a j₀) +
          edgeChain C W (j₀, a j₀) (pVert v) -
          edgeChain C W (pVert u) (pVert v) = 0 :=
        edgeChain_bool_triangle C W j₀ (cw u j₀) (a j₀) (cw v j₀)
      have hcomb := Submodule.sub_mem _ (Submodule.sub_mem _ mT mU) (Submodule.neg_mem _ mV)
      have heq_alg :
          (edgeChain (k := k) C W e.1.1 e.1.2 + edgeChain C W e.1.2 (j₀, a j₀) + edgeChain C W (j₀, a j₀) e.1.1) -
          (edgeChain C W e.1.1 (pVert u) + edgeChain C W (pVert u) (j₀, a j₀) + edgeChain C W (j₀, a j₀) e.1.1) -
          -(edgeChain C W e.1.2 (pVert v) + edgeChain C W (pVert v) (j₀, a j₀) + edgeChain C W (j₀, a j₀) e.1.2) =
          edgeChain C W e.1.1 e.1.2 -
          (edgeChain C W e.1.1 (pVert u) + edgeChain C W (pVert u) (pVert v) + edgeChain C W (pVert v) e.1.2) := by
        have hs1 := edgeChain_swap (k := k) C W (j₀, a j₀) e.1.2
        have hs2 := edgeChain_swap (k := k) C W (j₀, a j₀) (pVert v)
        have hs3 := edgeChain_swap (k := k) C W e.1.2 (pVert v)
        have hb' : edgeChain (k := k) C W (pVert u) (j₀, a j₀) + edgeChain C W (j₀, a j₀) (pVert v) =
            edgeChain C W (pVert u) (pVert v) := eq_of_sub_eq_zero hbool3
        calc _ = edgeChain (k := k) C W e.1.1 e.1.2 - edgeChain C W e.1.1 (pVert u) +
              (edgeChain C W e.1.2 (j₀, a j₀) + edgeChain C W (j₀, a j₀) e.1.2) -
              (edgeChain C W (pVert u) (j₀, a j₀) - edgeChain C W (pVert v) (j₀, a j₀)) +
              edgeChain C W e.1.2 (pVert v) := by abel
          _ = _ := by rw [hs1, hs2, hs3]; simp only [sub_neg_eq_add, hb']; abel
      rwa [heq_alg] at hcomb
    · -- Opposite edge e.1 = ((j, false), (j, true))
      have hu_eq : e.1.1 = (j, false) := congrArg Prod.fst heq
      have hv_eq : e.1.2 = (j, true) := congrArg Prod.snd heq
      have hjW := (Finset.mem_filter.mp hj).2
      have hPj := hPj_indep j hj (cw u) (hcw_in u) (cw v) (hcw_in v)
      have hcw_u_j : cw u j = false := by
        have := (Finset.mem_filter.mp (hcw_tr u)).2
        simpa [u, hu_eq] using this
      have hcw_v_j : cw v j = true := by
        have := (Finset.mem_filter.mp (hcw_tr v)).2
        simpa [v, hv_eq] using this
      rw [hcw_u_j, hcw_v_j, ← hu_eq, ← hv_eq] at hPj
      have heq_alg :
          edgeChain (k := k) C W (pVert u) e.1.1 + edgeChain C W e.1.1 e.1.2 +
          edgeChain C W e.1.2 (pVert v) - edgeChain C W (pVert u) (pVert v) =
          edgeChain C W e.1.1 e.1.2 -
          (edgeChain C W e.1.1 (pVert u) + edgeChain C W (pVert u) (pVert v) + edgeChain C W (pVert v) e.1.2) := by
        rw [edgeChain_swap C W (pVert u) e.1.1, edgeChain_swap C W e.1.2 (pVert v)]; abel
      simpa [pVert, heq_alg] using hPj
  -- Finally, for ANY z ∈ ker ∂₁, z ∈ im ∂₂!
  have hker_le : LinearMap.ker (bdry1 (k := k) C W) ≤ LinearMap.range (bdry2 (k := k) C W) := by
    intro z hz
    -- Define z_red := ∑ e, z e • edgeChain (pVert u_e) (pVert v_e), supported ONLY on the single
    -- edge ((j₀, false), (j₀, true))!
    let uOf (e : KEdges C W) : KVerts C W :=
      ⟨e.1.1, isFace_subset e.2.2 (by intro x hx; simp at hx ⊢; left; exact hx)⟩
    let vOf (e : KEdges C W) : KVerts C W :=
      ⟨e.1.2, isFace_subset e.2.2 (by intro x hx; simp at hx ⊢; right; exact hx)⟩
    have hdiff_mem : z - ∑ e : KEdges C W, z e • edgeChain (k := k) C W (pVert (uOf e)) (pVert (vOf e)) ∈
        LinearMap.range (bdry2 (k := k) C W) := by
      -- First show ∑ e, z e • (edgeChain e.1.1 (pVert (uOf e)) + edgeChain (pVert (vOf e)) e.1.2) = 0
      have htel0 : ∑ e : KEdges C W,
          z e • (edgeChain (k := k) C W e.1.1 (pVert (uOf e)) + edgeChain C W (pVert (vOf e)) e.1.2) = 0 := by
        have hv_eq : ∀ e : KEdges C W,
            z e • (edgeChain (k := k) C W e.1.1 (pVert (uOf e)) + edgeChain C W (pVert (vOf e)) e.1.2) =
            -(∑ v : KVerts C W,
              ((if e.1.2 = v.1 then z e else 0) - (if e.1.1 = v.1 then z e else 0)) •
                edgeChain C W v.1 (pVert v)) := by
          intro e
          have hs1 : ∑ v : KVerts C W, (if e.1.2 = v.1 then z e else 0) • edgeChain (k := k) C W v.1 (pVert v) =
              z e • edgeChain C W e.1.2 (pVert (vOf e)) := by
            rw [Finset.sum_eq_single (vOf e)]
            · simp [vOf]
            · intro v _ hv; have : e.1.2 ≠ v.1 := fun h => hv (Subtype.ext h.symm); simp [this]
            · intro h; exact (h (Finset.mem_univ _)).elim
          have hs2 : ∑ v : KVerts C W, (if e.1.1 = v.1 then z e else 0) • edgeChain (k := k) C W v.1 (pVert v) =
              z e • edgeChain C W e.1.1 (pVert (uOf e)) := by
            rw [Finset.sum_eq_single (uOf e)]
            · simp [uOf]
            · intro v _ hv; have : e.1.1 ≠ v.1 := fun h => hv (Subtype.ext h.symm); simp [this]
            · intro h; exact (h (Finset.mem_univ _)).elim
          simp_rw [sub_smul, Finset.sum_sub_distrib, hs1, hs2, edgeChain_swap C W (pVert (vOf e)) e.1.2]
          rw [smul_add, smul_neg]; abel
        simp_rw [hv_eq, Finset.sum_neg_distrib]
        rw [Finset.sum_comm, neg_eq_zero]
        refine Finset.sum_eq_zero fun v _ => ?_
        rw [← Finset.sum_smul]
        rw [LinearMap.mem_ker] at hz
        have hzv : (bdry1 (k := k) C W z) v = 0 := congrFun hz v
        simp only [bdry1, LinearMap.coe_mk, AddHom.coe_mk] at hzv
        rw [hzv, zero_smul]
      have hrew : z - ∑ e : KEdges C W, z e • edgeChain (k := k) C W (pVert (uOf e)) (pVert (vOf e)) =
          ∑ e : KEdges C W, z e • (edgeChain C W e.1.1 e.1.2 -
            (edgeChain C W e.1.1 (pVert (uOf e)) +
             edgeChain C W (pVert (uOf e)) (pVert (vOf e)) +
             edgeChain C W (pVert (vOf e)) e.1.2)) := by
        conv_lhs => arg 1; rw [chain_eq_sum_edgeChain z]
        have : (∑ e : KEdges C W, z e • edgeChain (k := k) C W e.1.1 e.1.2) -
            (∑ e : KEdges C W, z e • edgeChain C W (pVert (uOf e)) (pVert (vOf e))) =
            (∑ e : KEdges C W, z e • edgeChain C W e.1.1 e.1.2) -
            (∑ e : KEdges C W, z e • edgeChain C W (pVert (uOf e)) (pVert (vOf e))) -
            (∑ e : KEdges C W, z e • (edgeChain C W e.1.1 (pVert (uOf e)) + edgeChain C W (pVert (vOf e)) e.1.2)) := by
          rw [htel0, sub_zero]
        rw [this, ← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib]
        refine Finset.sum_congr rfl fun e _ => ?_
        simp only [smul_sub, smul_add]; abel
      rw [hrew]
      exact Submodule.sum_mem _ fun e _ => Submodule.smul_mem _ _ (hedge_homotopy e)
    -- Now z_red is in ker ∂₁ (since z ∈ ker ∂₁ and z - z_red ∈ im ∂₂ ≤ ker ∂₁) AND supported
    -- only on ((j₀, false), (j₀, true)), hence z_red = 0!
    set z_red := ∑ e : KEdges C W, z e • edgeChain (k := k) C W (pVert (uOf e)) (pVert (vOf e))
    have hz_red_ker : z_red ∈ LinearMap.ker (bdry1 (k := k) C W) := by
      have h1 : z - z_red ∈ LinearMap.ker (bdry1 (k := k) C W) := bdry1_comp_bdry2 C W hdiff_mem
      have : z_red = z - (z - z_red) := by abel
      rw [this]
      exact Submodule.sub_mem _ hz h1
    have hz_red_supp : ∀ e : KEdges C W, e.1 ≠ ((j₀, false), (j₀, true)) → z_red e = 0 := by
      intro e he
      simp only [z_red, Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
      refine Finset.sum_eq_zero fun e' _ => ?_
      have h0 : edgeChain (k := k) C W (pVert (uOf e')) (pVert (vOf e')) e = 0 := by
        simp only [pVert, edgeChain]
        rcases Bool.eq_false_or_eq_true (cw (uOf e') j₀) with h1 | h1 <;>
          rcases Bool.eq_false_or_eq_true (cw (vOf e') j₀) with h2 | h2 <;>
          simp only [h1, h2]
        · have hne : e.1 ≠ ((j₀, true), (j₀, true)) := fun h => by
            have := e.2.1; rw [h] at this; exact lt_irrefl _ this
          rw [if_neg hne, if_neg hne]
        · have hne2 : e.1 ≠ ((j₀, true), (j₀, false)) := fun h => by
            have := e.2.1; rw [h] at this; simp [litIdx] at this
          rw [if_neg hne2, if_neg he]
        · have hne1 : e.1 ≠ ((j₀, true), (j₀, false)) := fun h => by
            have := e.2.1; rw [h] at this; simp [litIdx] at this
          rw [if_neg he, if_neg hne1]
        · have hne : e.1 ≠ ((j₀, false), (j₀, false)) := fun h => by
            have := e.2.1; rw [h] at this; exact lt_irrefl _ this
          rw [if_neg hne, if_neg hne]
      rw [h0, mul_zero]
    have hz_red_zero : z_red = 0 := by
      ext e
      by_cases he : e.1 = ((j₀, false), (j₀, true))
      · rw [LinearMap.mem_ker] at hz_red_ker
        have hv2 : isFace C W {(j₀, true)} :=
          isFace_subset e.2.2 (by rw [show e.1.2 = (j₀, true) from congrArg Prod.snd he]; simp)
        have hval := congrFun hz_red_ker ⟨(j₀, true), hv2⟩
        simp only [bdry1, LinearMap.coe_mk, AddHom.coe_mk, Pi.zero_apply] at hval
        rw [Finset.sum_eq_single e] at hval
        · have h1 : e.1.2 = (j₀, true) := congrArg Prod.snd he
          have h2 : e.1.1 ≠ (j₀, true) := by
            rw [show e.1.1 = (j₀, false) from congrArg Prod.fst he]; simp
          simpa [h1, h2] using hval
        · intro e' _ he'
          have hne : e'.1 ≠ ((j₀, false), (j₀, true)) := fun h => he' (Subtype.ext (h.trans he.symm))
          simp [hz_red_supp e' hne]
        · intro h; exact (h (Finset.mem_univ _)).elim
      · exact hz_red_supp e he
    rw [hz_red_zero, sub_zero] at hdiff_mem
    exact hdiff_mem
  -- Therefore H1Space k C W is subsingleton, so H1Dim k C W = 0!
  have hsub : Subsingleton (H1Space k C W) := by
    refine ⟨fun x y => ?_⟩
    induction x using Quotient.inductionOn with
    | h x =>
      induction y using Quotient.inductionOn with
      | h y =>
        change Submodule.Quotient.mk x = Submodule.Quotient.mk y
        rw [Submodule.Quotient.eq]
        show (x.1 - y.1) ∈ LinearMap.range (bdry2 (k := k) C W)
        exact hker_le (Submodule.sub_mem _ x.2 y.2)
  unfold H1Dim
  exact Module.finrank_zero_of_subsingleton

/-- **Theorem 3 (Graph formula for the `H₁`-strand over EVERY field `k`)**:
Suppose `d(W) > 0` and `C.Nonempty`. Then `dim_k H̃₁(K(C, W); k) = (d(W) - 1) * (h(W) - 1)`! -/
theorem h1_strand_dim_formula {n : ℕ} {C : Finset (Word n)} {W : Finset (Lit n)}
    (_hC : C.Nonempty) (hd : 0 < (doubledCoords W).card) :
    H1Dim k C W = ((doubledCoords W).card - 1) * (hCount C W - 1) := by
  classical
  by_cases hdisc : ∃ a ∈ C, ∃ b ∈ C, ¬ traceReach C W a b
  · have hh2 : hCount C W = 2 := by unfold hCount; exact if_pos hdisc
    rw [hh2]; simp only [Nat.reduceSub, mul_one]
    exact h1_strand_dim_h2 (k := k) hd hdisc
  · have hh1 : hCount C W = 1 := by unfold hCount; exact if_neg hdisc
    rw [hh1]; simp only [Nat.sub_self, mul_zero]
    push_neg at hdisc
    exact h1_strand_dim_h1 (k := k) hd hdisc

end Homology

/-! ### 3. Classification of Maximum Projective Dimension `pd = 2n - 3` (Theorem 4) -/

/-- **Antipodal characterization of `h(V) = 2`**:
For `n ≥ 2` and a nonempty code `C ⊆ {0, 1}ⁿ`, the trace-sharing graph on the full literal set
`V = Finset.univ` is disconnected (`h(V) = 2`) IF AND ONLY IF `C = {a, antipode a}` is an
antipodal two-word pair! -/
theorem fullSupport_disconnected_iff_antipodal {n : ℕ} (hn : 2 ≤ n)
    {C : Finset (Word n)} (hC : C.Nonempty) :
    hCount C Finset.univ = 2 ↔ ∃ a : Word n, C = {a, antipode a} := by
  classical
  have hd_eq : doubledCoords (Finset.univ : Finset (Lit n)) = Finset.univ := by
    ext j; simp [doubledCoords]
  have hd_pos : 0 < (doubledCoords (Finset.univ : Finset (Lit n))).card := by
    rw [hd_eq, Finset.card_univ, Fintype.card_fin]; omega
  obtain ⟨j₀, hj₀⟩ := Finset.card_pos.mp hd_pos
  have hiff_disc : hCount C Finset.univ = 2 ↔ ∃ a ∈ C, ∃ b ∈ C, ¬ traceReach C Finset.univ a b := by
    unfold hCount; split_ifs with h <;> simp [h]
  rw [hiff_disc]
  constructor
  · rintro ⟨a, ha, b, hb, hnab⟩
    have hcomp := (trace_sharing_dichotomy (C := C) hj₀).2.2 a ha b hb hnab
    refine ⟨a, ?_⟩
    ext c
    simp only [Finset.mem_insert, Finset.mem_singleton]
    constructor
    · intro hc
      by_cases hrc : traceReach C Finset.univ a c
      · left; ext j; exact (hcomp j (by rw [hd_eq]; exact Finset.mem_univ j) c hc).1 hrc
      · right; ext j; exact (hcomp j (by rw [hd_eq]; exact Finset.mem_univ j) c hc).2 hrc |>.1
    · rintro (rfl | rfl)
      · exact ha
      · have hb_eq : b = antipode a := by
          ext j; exact (hcomp j (by rw [hd_eq]; exact Finset.mem_univ j) b hb).2 hnab |>.1
        rwa [← hb_eq]
  · rintro ⟨a, rfl⟩
    refine ⟨a, by simp, antipode a, by simp, ?_⟩
    intro hr
    -- On {a, antipode a}, the two words disagree at every coordinate, so they have no shared literal!
    have hno_adj : ∀ x y : Word n,
        (x ∈ ({a, antipode a} : Finset (Word n)) ∧ y ∈ ({a, antipode a} : Finset (Word n)) ∧
          traceAdj Finset.univ x y) → x = y := by
      intro x y ⟨hx, hy, ⟨l, hl⟩⟩
      rw [Finset.mem_inter, trace, trace, Finset.mem_filter, Finset.mem_filter] at hl
      simp only [Finset.mem_insert, Finset.mem_singleton] at hx hy
      rcases hx with hx | hx <;> rcases hy with hy | hy
      · rw [hx, hy]
      · exfalso
        rw [hx, hy] at hl
        have h1 : a l.1 = l.2 := hl.1.2
        have h2 : (!a l.1) = l.2 := hl.2.2
        cases h : a l.1 <;> simp_all
      · exfalso
        rw [hx, hy] at hl
        have h1 : (!a l.1) = l.2 := hl.1.2
        have h2 : a l.1 = l.2 := hl.2.2
        cases h : a l.1 <;> simp_all
      · rw [hx, hy]
    have hreach_eq : ∀ y : Word n, traceReach ({a, antipode a} : Finset (Word n)) Finset.univ a y → a = y := by
      intro y hy
      induction hy with
      | refl => rfl
      | tail _ hstep ih => exact ih.trans (hno_adj _ _ hstep)
    have heq : a = antipode a := hreach_eq (antipode a) hr
    have h0 := congrFun heq ⟨0, by omega⟩
    simp only [antipode] at h0
    cases h : a ⟨0, by omega⟩ <;> rw [h] at h0 <;> simp at h0

/-- **Theorem 4 (Exact classification of maximal projective dimension `pd = 2n - 3` for `n ≥ 3`)**:
Over EVERY field `k` and for every `n ≥ 3`:
1. In Hochster's formula `β_{i, W}(I_C) = dim_k H̃_{|W| - i - 2}(K(C, W); k)`:
   - Degree `q = -1` (all vertices of `W` are ghosts) has `|W| ≤ n`, so `i = |W| - 1 ≤ n - 1 < 2n - 3`.
   - Degree `q = 0` (`K(C, W)` disconnected) forces `d(W) = 0`, so `|W| ≤ n` and `i = |W| - 2 ≤ n - 2 < 2n - 3`.
   - Degree `q ≥ 1` has `i = |W| - q - 2 ≤ 2n - 3`, with equality iff `W = V` (`|W| = 2n`) and `q = 1`.
2. At `W = V`, `q = 1`, `H1Dim k C V ≠ 0` if and only if `C = {a, antipode a}` is an antipodal
   two-word pair, in which case `H1Dim k C V = n - 1`! -/
theorem max_pdim_classification (k : Type*) [Field k] {n : ℕ} (hn : 3 ≤ n)
    {C : Finset (Word n)} (hC : C.Nonempty) :
    (∀ W : Finset (Lit n), (doubledCoords W).card = 0 → W.card ≤ n) ∧
    (∀ W : Finset (Lit n), ∀ q : ℕ, 1 ≤ q → W.card - q - 2 ≤ 2 * n - 3 ∧
      (W.card - q - 2 = 2 * n - 3 ↔ W = Finset.univ ∧ q = 1)) ∧
    (H1Dim k C Finset.univ ≠ 0 ↔ ∃ a : Word n, C = {a, antipode a}) ∧
    (∀ a : Word n, H1Dim k ({a, antipode a} : Finset (Word n)) Finset.univ = n - 1) := by
  classical
  have hcardV : (Finset.univ : Finset (Lit n)).card = 2 * n := by
    rw [Finset.card_univ, Fintype.card_prod, Fintype.card_fin, Fintype.card_bool]; ring
  have hdV : (doubledCoords (Finset.univ : Finset (Lit n))).card = n := by
    have : doubledCoords (Finset.univ : Finset (Lit n)) = Finset.univ := by ext j; simp [doubledCoords]
    rw [this, Finset.card_univ, Fintype.card_fin]
  have hnodub : ∀ W : Finset (Lit n), (doubledCoords W).card = 0 → W.card ≤ n := by
    intro W h0
    have hno : ∀ j : Fin n, ¬ ((j, true) ∈ W ∧ (j, false) ∈ W) := by
      intro j hj
      have : j ∈ doubledCoords W := Finset.mem_filter.mpr ⟨Finset.mem_univ _, hj⟩
      exact Finset.card_ne_zero.mpr ⟨j, this⟩ h0
    have hinj : Set.InjOn Prod.fst (W : Set (Lit n)) := by
      intro ⟨i₁, b₁⟩ h₁ ⟨i₂, b₂⟩ h₂ heq
      dsimp only at heq; subst heq
      cases b₁ <;> cases b₂
      · rfl
      · exact (hno i₁ ⟨h₂, h₁⟩).elim
      · exact (hno i₁ ⟨h₁, h₂⟩).elim
      · rfl
    calc W.card = (W.image Prod.fst).card := (Finset.card_image_of_injOn hinj).symm
      _ ≤ (Finset.univ : Finset (Fin n)).card := Finset.card_le_card (Finset.subset_univ _)
      _ = n := by simp
  have hdeg : ∀ W : Finset (Lit n), ∀ q : ℕ, 1 ≤ q → W.card - q - 2 ≤ 2 * n - 3 ∧
      (W.card - q - 2 = 2 * n - 3 ↔ W = Finset.univ ∧ q = 1) := by
    intro W q hq
    have hWle : W.card ≤ 2 * n := by rw [← hcardV]; exact Finset.card_le_card (Finset.subset_univ W)
    refine ⟨by omega, ?_⟩
    constructor
    · intro heq
      have hWcard : W.card = 2 * n := by omega
      have hq1 : q = 1 := by omega
      have hWuniv : W = Finset.univ := Finset.eq_univ_of_card W (by rw [hWcard]; simpa using hcardV.symm)
      exact ⟨hWuniv, hq1⟩
    · rintro ⟨rfl, rfl⟩; rw [hcardV]; omega
  have hformula := h1_strand_dim_formula (k := k) hC (by rw [hdV]; omega)
  rw [hdV] at hformula
  have hiff : H1Dim k C Finset.univ ≠ 0 ↔ ∃ a : Word n, C = {a, antipode a} := by
    rw [hformula, ← fullSupport_disconnected_iff_antipodal (by omega) hC]
    have h12 : hCount C Finset.univ = 1 ∨ hCount C Finset.univ = 2 := by
      unfold hCount; split_ifs <;> simp
    rcases h12 with h1 | h2
    · rw [h1]; simp
    · rw [h2]; simp; omega
  have hantip : ∀ a : Word n, H1Dim k ({a, antipode a} : Finset (Word n)) Finset.univ = n - 1 := by
    intro a
    have hne : ({a, antipode a} : Finset (Word n)).Nonempty := ⟨a, by simp⟩
    have hf := h1_strand_dim_formula (k := k) hne (by rw [hdV]; omega)
    have hh2 : hCount ({a, antipode a} : Finset (Word n)) Finset.univ = 2 :=
      (fullSupport_disconnected_iff_antipodal (by omega) hne).mpr ⟨a, rfl⟩
    rw [hf, hdV, hh2]; simp
  exact ⟨hnodub, hdeg, hiff, hantip⟩

/-! ### 4. Alexander Duality Criterion (Theorem 5) & `n = 4` Non-Duality Witness (Proposition 6) -/

/-- Literal involution `ι` swapping `(i, true) ↔ (i, false)`. -/
def litFlip {n : ℕ} (l : Lit n) : Lit n := (l.1, !l.2)

/-- Support `H ⊆ V` lies in `ι(I_C^∨)` iff `ι(H)` meets every missing subcube `Q_A ⊆ Q \ C`
(equivalently, no consistent literal set `A` disjoint from `ι(H)` avoids `C`). -/
def memDualIdeal {n : ℕ} (C : Finset (Word n)) (H : Finset (Lit n)) : Prop :=
  ∀ A : Finset (Lit n), Consistent A → Disjoint A (H.image litFlip) → ∃ w ∈ C, satisfies w A

/-- Support `H ⊆ V` lies in `I_{Q \ C}` iff `H` contains a consistent literal set `A`
whose entire subcube `Q_A` is contained in `C`. -/
def memCompIdeal {n : ℕ} (C : Finset (Word n)) (H : Finset (Lit n)) : Prop :=
  ∃ A ⊆ H, Consistent A ∧ ∀ w : Word n, satisfies w A → w ∈ C

/-- **Theorem 5 (Exact coordinate-slice lifting criterion for `ι(I_C^∨) = I_{Q \ C}`)**:
Given any literal support `H ⊆ V`, let `D` be its doubled coordinates, `T` its single-literal
coordinates, and `U` its unused coordinates. Then:
1. `H ∈ ι(I_C^∨)` iff for every assignment on `U`, there exists a completion on `D` (matching `H`
   on `T`) lying in `C`.
2. `H ∈ I_{Q \ C}` iff there exists a single assignment on `D` whose entire `U`-subcube
   (matching `H` on `T`) lies in `C`.
3. Consequently `I_{Q \ C} ⊆ ι(I_C^∨)` always, with equality on `H` iff the slice-lifting
   implication holds! -/
theorem alexander_duality_criterion {n : ℕ} (C : Finset (Word n)) (H : Finset (Lit n)) :
    (memDualIdeal C H ↔
      ∀ u : Word n, (∀ l ∈ H, l.1 ∉ doubledCoords H → u l.1 = l.2) →
        ∃ w ∈ C, ∀ j : Fin n, j ∉ doubledCoords H → w j = u j) ∧
    (memCompIdeal C H ↔
      ∃ d : Word n, (∀ l ∈ H, l.1 ∉ doubledCoords H → d l.1 = l.2) ∧
        ∀ w : Word n, (∀ j : Fin n, (j, true) ∈ H ∨ (j, false) ∈ H → w j = d j) → w ∈ C) ∧
    (memCompIdeal C H → memDualIdeal C H) := by
  classical
  have h1 : memDualIdeal C H ↔
      ∀ u : Word n, (∀ l ∈ H, l.1 ∉ doubledCoords H → u l.1 = l.2) →
        ∃ w ∈ C, ∀ j : Fin n, j ∉ doubledCoords H → w j = u j := by
    constructor
    · intro hdual u hu
      let A : Finset (Lit n) := (Finset.univ.filter fun j => j ∉ doubledCoords H).image fun j => (j, u j)
      have hcon : Consistent A := by
        intro i ⟨ht, hf⟩
        simp only [A, Finset.mem_image, Finset.mem_filter, Finset.mem_univ, true_and] at ht hf
        rcases ht with ⟨j₁, _, h1⟩; rcases hf with ⟨j₂, _, h2⟩
        have hj : j₁ = j₂ := (congrArg Prod.fst h1).trans (congrArg Prod.fst h2).symm
        subst hj
        have e1 : u j₁ = true := congrArg Prod.snd h1
        have e2 : u j₁ = false := congrArg Prod.snd h2
        exact Bool.false_ne_true (e2.symm.trans e1)
      have hdisj : Disjoint A (H.image litFlip) := by
        rw [Finset.disjoint_left]
        intro l hlA hlH
        simp only [A, Finset.mem_image, Finset.mem_filter, Finset.mem_univ, true_and] at hlA
        simp only [Finset.mem_image] at hlH
        rcases hlA with ⟨j, hj, rfl⟩
        rcases hlH with ⟨l', hl'H, hfl⟩
        have hj' : l'.1 = j := congrArg Prod.fst hfl
        have hsat := hu l' hl'H (by rw [hj']; exact hj)
        have hfl2 : (!l'.2) = u j := congrArg Prod.snd hfl
        rw [hj'] at hsat
        cases h : u j <;> rw [h] at hsat hfl2 <;> simp [← hsat] at hfl2
      rcases hdual A hcon hdisj with ⟨w, hwC, hwsat⟩
      refine ⟨w, hwC, fun j hj => ?_⟩
      exact hwsat (j, u j) (Finset.mem_image.mpr ⟨j, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hj⟩, rfl⟩)
    · intro hlift A hcon hdisj
      -- Extend A to a word u matching H on single-literal coordinates
      let u : Word n := fun j =>
        if (j, true) ∈ A then true
        else if (j, false) ∈ A then false
        else if (j, true) ∈ H then true
        else false
      have hu_H : ∀ l ∈ H, l.1 ∉ doubledCoords H → u l.1 = l.2 := by
        intro ⟨j, b⟩ hlH hjnd
        have hnotA : (j, !b) ∉ A := fun hmem => by
          have : (j, !b) ∈ H.image litFlip := Finset.mem_image.mpr ⟨(j, b), hlH, rfl⟩
          exact Finset.disjoint_left.mp hdisj hmem this
        cases b
        · have hnt : (j, true) ∉ A := hnotA
          have hntH : (j, true) ∉ H := fun ht => hjnd (Finset.mem_filter.mpr ⟨Finset.mem_univ _, ht, hlH⟩)
          simp [u, hnt, hntH]
        · have hnf : (j, false) ∉ A := hnotA
          simp only [u, hnf, ↓reduceIte, hlH]
          split_ifs <;> rfl
      rcases hlift u hu_H with ⟨w, hwC, hwu⟩
      refine ⟨w, hwC, fun ⟨j, b⟩ hlA => ?_⟩
      have hjnd : j ∉ doubledCoords H := by
        intro hjd
        have hjW := (Finset.mem_filter.mp hjd).2
        have hmem : (j, b) ∈ H.image litFlip := by
          cases b
          · exact Finset.mem_image.mpr ⟨(j, true), hjW.1, rfl⟩
          · exact Finset.mem_image.mpr ⟨(j, false), hjW.2, rfl⟩
        exact Finset.disjoint_left.mp hdisj hlA hmem
      rw [hwu j hjnd]
      cases b
      · have hnt : (j, true) ∉ A := fun ht => hcon j ⟨ht, hlA⟩
        simp [u, hnt, hlA]
      · simp [u, hlA]
  have h2 : memCompIdeal C H ↔
      ∃ d : Word n, (∀ l ∈ H, l.1 ∉ doubledCoords H → d l.1 = l.2) ∧
        ∀ w : Word n, (∀ j : Fin n, (j, true) ∈ H ∨ (j, false) ∈ H → w j = d j) → w ∈ C := by
    constructor
    · rintro ⟨A, hAH, hcon, hsubC⟩
      let d : Word n := fun j =>
        if (j, true) ∈ A then true
        else if (j, false) ∈ A then false
        else if (j, true) ∈ H then true
        else false
      refine ⟨d, ?_, ?_⟩
      · intro ⟨j, b⟩ hlH hjnd
        cases b
        · have hntH : (j, true) ∉ H := fun ht => hjnd (Finset.mem_filter.mpr ⟨Finset.mem_univ _, ht, hlH⟩)
          have hntA : (j, true) ∉ A := fun ht => hntH (hAH ht)
          simp [d, hntA, hntH]
        · have hnfH : (j, false) ∉ H := fun hf => hjnd (Finset.mem_filter.mpr ⟨Finset.mem_univ _, hlH, hf⟩)
          have hnfA : (j, false) ∉ A := fun hf => hnfH (hAH hf)
          simp only [d, hnfA, ↓reduceIte, hlH]
          split_ifs <;> rfl
      · intro w hw
        apply hsubC
        intro ⟨j, b⟩ hlA
        have hjH : (j, true) ∈ H ∨ (j, false) ∈ H := by
          cases b
          · right; exact hAH hlA
          · left; exact hAH hlA
        rw [hw j hjH]
        cases b
        · have hnt : (j, true) ∉ A := fun ht => hcon j ⟨ht, hlA⟩
          simp [d, hnt, hlA]
        · simp [d, hlA]
    · rintro ⟨d, hdH, hsubC⟩
      let A : Finset (Lit n) :=
        (Finset.univ.filter fun j => (j, true) ∈ H ∨ (j, false) ∈ H).image fun j => (j, d j)
      refine ⟨A, ?_, ?_, ?_⟩
      · intro l hl
        simp only [A, Finset.mem_image, Finset.mem_filter, Finset.mem_univ, true_and] at hl
        rcases hl with ⟨j, hjH, rfl⟩
        by_cases hjd : j ∈ doubledCoords H
        · have := (Finset.mem_filter.mp hjd).2
          cases d j <;> simp [this]
        · rcases hjH with ht | hf
          · rw [hdH (j, true) ht hjd]; exact ht
          · rw [hdH (j, false) hf hjd]; exact hf
      · intro i ⟨ht, hf⟩
        simp only [A, Finset.mem_image, Finset.mem_filter, Finset.mem_univ, true_and] at ht hf
        rcases ht with ⟨j₁, _, h1⟩; rcases hf with ⟨j₂, _, h2⟩
        have hj : j₁ = j₂ := (congrArg Prod.fst h1).trans (congrArg Prod.fst h2).symm
        subst hj
        exact Bool.false_ne_true ((congrArg Prod.snd h2).symm.trans (congrArg Prod.snd h1))
      · intro w hwsat
        apply hsubC
        intro j hjH
        exact hwsat (j, d j) (Finset.mem_image.mpr ⟨j, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hjH⟩, rfl⟩)
  refine ⟨h1, h2, fun hcomp => ?_⟩
  rw [h2] at hcomp; rw [h1]
  rcases hcomp with ⟨d, hdH, hsubC⟩
  intro u hu
  let w : Word n := fun j => if j ∈ doubledCoords H then d j else u j
  refine ⟨w, hsubC w fun j hjH => ?_, fun j hj => by simp [w, hj]⟩
  by_cases hjd : j ∈ doubledCoords H
  · simp [w, hjd]
  · simp only [w, hjd, ↓reduceIte]
    rcases hjH with ht | hf
    · rw [hu (j, true) ht hjd, hdH (j, true) ht hjd]
    · rw [hu (j, false) hf hjd, hdH (j, false) hf hjd]

/-- Minimal monomial generators (maximal missing subcubes of `Q \ C`) for a code `C`. -/
def isMinGen {n : ℕ} (C : Finset (Word n)) (A : Finset (Lit n)) : Prop :=
  Consistent A ∧ (∀ w : Word n, satisfies w A → w ∉ C) ∧
  ∀ B ⊂ A, ∃ w ∈ C, satisfies w B

instance {n : ℕ} (C : Finset (Word n)) (A : Finset (Lit n)) : Decidable (isMinGen C A) := by
  unfold isMinGen; infer_instance

/-- **Proposition 6 (`n = 4` non-duality witness separating `(1, 3)` into `(2, 2)` and `(1, 3)`)**:
- `codeA = {w ∈ {0,1}⁴ | w 3 = false ∧ w ≠ (true, true, true, false)}` has exact minimal
  generators `{ {(3, true)}, {(0, true), (1, true), (2, true)} }` (complete intersection `I_A = (x₄, x₁x₂x₃)`
  of degrees `1, 3` on disjoint variables → `(pd, reg) = (1, 3)`).
- `Q \ codeA` has exact minimal generators `{ {(3, false), (0, false)}, {(3, false), (1, false)},
  {(3, false), (2, false)} }` (`I_{Q \ A} = y₄ (y₁, y₂, y₃)` → shifted Koszul `(pd, reg) = (2, 2)`).
- `codeB = {w ∈ {0,1}⁴ | w 2 ≠ w 3}` has exact minimal generators `{ {(2, true), (3, true)},
  {(2, false), (3, false)} }` (complete intersection `(x₃x₄, y₃y₄)` → `(pd, reg) = (1, 3)`).
- `Q \ codeB` has exact minimal generators `{ {(2, true), (3, false)}, {(2, false), (3, true)} }`
  (complete intersection `(x₃y₄, y₃x₄)` → `(pd, reg) = (1, 3)`). -/
def codeA : Finset (Word 4) :=
  Finset.univ.filter fun w => w 3 = false ∧ ¬ (w 0 = true ∧ w 1 = true ∧ w 2 = true ∧ w 3 = false)

def codeB : Finset (Word 4) :=
  Finset.univ.filter fun w => w 2 ≠ w 3

/-- Helper: given a list `G` of pairwise incomparable consistent literal sets that avoid `C`,
if every consistent avoiding `A` contains some `g ∈ G`, then `isMinGen C A ↔ A ∈ G`. -/
private lemma isMinGen_iff_mem_list {n : ℕ} {C : Finset (Word n)} {G : List (Finset (Lit n))}
    (hconG : ∀ g ∈ G, Consistent g)
    (havoidG : ∀ g ∈ G, ∀ w : Word n, satisfies w g → w ∉ C)
    (hcover : ∀ A : Finset (Lit n), Consistent A → (∀ w : Word n, satisfies w A → w ∉ C) →
      ∃ g ∈ G, g ⊆ A)
    (hanti : ∀ g₁ ∈ G, ∀ g₂ ∈ G, g₁ ⊆ g₂ → g₁ = g₂)
    (A : Finset (Lit n)) :
    isMinGen C A ↔ A ∈ G := by
  constructor
  · rintro ⟨hconA, havoidA, hminA⟩
    obtain ⟨g, hgG, hgA⟩ := hcover A hconA havoidA
    by_cases heq : g = A
    · rwa [← heq]
    · obtain ⟨w, hwC, hwsat⟩ := hminA g (Finset.ssubset_iff_subset_ne.mpr ⟨hgA, heq⟩)
      exact (havoidG g hgG w hwsat hwC).elim
  · intro hAG
    refine ⟨hconG A hAG, havoidG A hAG, fun B hBA => ?_⟩
    by_contra hall; push_neg at hall
    have hconB : Consistent B := fun i ⟨ht, hf⟩ => hconG A hAG i ⟨hBA.1 ht, hBA.1 hf⟩
    obtain ⟨g, hgG, hgB⟩ := hcover B hconB (fun w hwsat hwC => hall w hwC hwsat)
    have hEq := hanti g hgG A hAG (hgB.trans hBA.1)
    subst hEq
    exact hBA.2 (Finset.Subset.antisymm hBA.1 hgB ▸ hBA.1)

/-- Helper: given a consistent literal set `A` on `Fin 4` and a default word `w₀`,
there exists a word `w` satisfying `A` that equals `w₀` on every coordinate not forced by `A`. -/
private lemma exists_word_matching_A {n : ℕ} {A : Finset (Lit n)} (hcon : Consistent A)
    (w₀ : Word n) :
    ∃ w : Word n, satisfies w A ∧
      ∀ i : Fin n, (i, true) ∉ A → (i, false) ∉ A → w i = w₀ i := by
  let w : Word n := fun i =>
    if (i, true) ∈ A then true
    else if (i, false) ∈ A then false
    else w₀ i
  refine ⟨w, fun ⟨i, b⟩ hl => ?_, fun i ht hf => by simp [w, ht, hf]⟩
  cases b
  · have ht : (i, true) ∉ A := fun h => hcon i ⟨h, hl⟩
    simp [w, ht, hl]
  · simp [w, hl]

theorem n4_non_duality_witness :
    (∀ A : Finset (Lit 4), isMinGen codeA A ↔
      A = {(3, true)} ∨ A = {(0, true), (1, true), (2, true)}) ∧
    (∀ A : Finset (Lit 4), isMinGen (Finset.univ \ codeA) A ↔
      A = {(3, false), (0, false)} ∨ A = {(3, false), (1, false)} ∨ A = {(3, false), (2, false)}) ∧
    (∀ A : Finset (Lit 4), isMinGen codeB A ↔
      A = {(2, true), (3, true)} ∨ A = {(2, false), (3, false)}) ∧
    (∀ A : Finset (Lit 4), isMinGen (Finset.univ \ codeB) A ↔
      A = {(2, true), (3, false)} ∨ A = {(2, false), (3, true)}) := by
  have hcodeA : ∀ w : Word 4, w ∈ codeA ↔
      w 3 = false ∧ ¬ (w 0 = true ∧ w 1 = true ∧ w 2 = true) := by
    intro w; simp only [codeA, Finset.mem_filter, Finset.mem_univ, true_and]
    constructor
    · rintro ⟨h3, hnot⟩; exact ⟨h3, fun h => hnot ⟨h.1, h.2.1, h.2.2, h3⟩⟩
    · rintro ⟨h3, hnot⟩; exact ⟨h3, fun h => hnot ⟨h.1, h.2.1, h.2.2.1⟩⟩
  have hcompA : ∀ w : Word 4, w ∈ Finset.univ \ codeA ↔
      w 3 = true ∨ (w 0 = true ∧ w 1 = true ∧ w 2 = true) := by
    intro w; simp only [Finset.mem_sdiff, Finset.mem_univ, true_and, hcodeA]
    cases h3 : w 3 <;> simp
  have hcodeB : ∀ w : Word 4, w ∈ codeB ↔ w 2 ≠ w 3 := by
    intro w; simp [codeB]
  have hcompB : ∀ w : Word 4, w ∈ Finset.univ \ codeB ↔ w 2 = w 3 := by
    intro w; simp [codeB]
  -- Helper: check consistency and pairwise incomparability directly without deep Finset decide
  have hcon_of_all_same : ∀ (b : Bool) (S : Finset (Lit 4)), (∀ l ∈ S, l.2 = b) → Consistent S := by
    intro b S hS i ⟨ht, hf⟩
    have e1 := hS (i, true) ht
    have e2 := hS (i, false) hf
    exact Bool.false_ne_true (e2.trans e1.symm)
  have hcon_pair_ne : ∀ {i j : Fin 4} {b₁ b₂ : Bool}, i ≠ j → Consistent ({(i, b₁), (j, b₂)} : Finset (Lit 4)) := by
    intro i j b₁ b₂ hij k ⟨ht, hf⟩
    simp only [Finset.mem_insert, Finset.mem_singleton] at ht hf
    rcases ht with ht | ht <;> rcases hf with hf | hf
    · exact Bool.false_ne_true ((congrArg Prod.snd hf).trans (congrArg Prod.snd ht).symm)
    · exact hij ((congrArg Prod.fst ht).symm.trans (congrArg Prod.fst hf))
    · exact hij ((congrArg Prod.fst hf).symm.trans (congrArg Prod.fst ht))
    · exact Bool.false_ne_true ((congrArg Prod.snd hf).trans (congrArg Prod.snd ht).symm)
  refine ⟨fun A => ?_, fun A => ?_, fun A => ?_, fun A => ?_⟩
  · -- codeA
    have h := isMinGen_iff_mem_list (C := codeA)
      (G := [{(3, true)}, {(0, true), (1, true), (2, true)}])
      (by
        intro g hg; simp only [List.mem_cons, List.not_mem_nil, or_false] at hg
        rcases hg with rfl | rfl
        · apply hcon_of_all_same true; intro l hl; simp only [Finset.mem_singleton] at hl; rw [hl]
        · apply hcon_of_all_same true; intro l hl; simp only [Finset.mem_insert, Finset.mem_singleton] at hl
          rcases hl with rfl | rfl | rfl <;> rfl)
      (by
        intro g hg w hwsat hwC
        rw [hcodeA] at hwC
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hg
        rcases hg with rfl | rfl
        · have := hwsat (3, true) (by simp); exact Bool.false_ne_true (hwC.1.symm.trans this)
        · exact hwC.2 ⟨hwsat (0, true) (by simp), hwsat (1, true) (by simp), hwsat (2, true) (by simp)⟩)
      (by
        intro B hconB havoidB
        by_cases h3t : (3, true) ∈ B
        · exact ⟨{(3, true)}, by simp, Finset.singleton_subset_iff.mpr h3t⟩
        · by_cases h012 : (0, true) ∈ B ∧ (1, true) ∈ B ∧ (2, true) ∈ B
          · refine ⟨{(0, true), (1, true), (2, true)}, by simp, ?_⟩
            intro l hl; simp only [Finset.mem_insert, Finset.mem_singleton] at hl
            rcases hl with rfl | rfl | rfl
            · exact h012.1
            · exact h012.2.1
            · exact h012.2.2
          · exfalso
            obtain ⟨w, hwsat, hwdef⟩ := exists_word_matching_A hconB (fun _ => false)
            apply havoidB w hwsat
            rw [hcodeA]
            have hw3 : w 3 = false := by
              by_cases h3f : (3, false) ∈ B
              · exact hwsat (3, false) h3f
              · exact hwdef 3 h3t h3f
            refine ⟨hw3, fun ⟨h0, h1, h2⟩ => h012 ⟨?_, ?_, ?_⟩⟩
            · by_contra hn0
              by_cases hf : (0, false) ∈ B
              · exact Bool.false_ne_true ((hwsat (0, false) hf).symm.trans h0)
              · exact Bool.false_ne_true ((hwdef 0 hn0 hf).symm.trans h0)
            · by_contra hn1
              by_cases hf : (1, false) ∈ B
              · exact Bool.false_ne_true ((hwsat (1, false) hf).symm.trans h1)
              · exact Bool.false_ne_true ((hwdef 1 hn1 hf).symm.trans h1)
            · by_contra hn2
              by_cases hf : (2, false) ∈ B
              · exact Bool.false_ne_true ((hwsat (2, false) hf).symm.trans h2)
              · exact Bool.false_ne_true ((hwdef 2 hn2 hf).symm.trans h2))
      (by
        intro g₁ hg₁ g₂ hg₂ hsub
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hg₁ hg₂
        rcases hg₁ with rfl | rfl <;> rcases hg₂ with rfl | rfl
        · rfl
        · exfalso; have := hsub (by simp : (3, true) ∈ ({(3, true)} : Finset (Lit 4))); simp at this
        · exfalso; have := hsub (by simp : (0, true) ∈ ({(0, true), (1, true), (2, true)} : Finset (Lit 4))); simp at this
        · rfl) A
    simpa using h
  · -- univ \ codeA
    have h := isMinGen_iff_mem_list (C := Finset.univ \ codeA)
      (G := [{(3, false), (0, false)}, {(3, false), (1, false)}, {(3, false), (2, false)}])
      (by
        intro g hg; simp only [List.mem_cons, List.not_mem_nil, or_false] at hg
        rcases hg with rfl | rfl | rfl <;> apply hcon_of_all_same false <;> intro l hl <;> simp at hl <;> rcases hl with rfl | rfl <;> rfl)
      (by
        intro g hg w hwsat hwC
        rw [hcompA] at hwC
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hg
        rcases hg with rfl | rfl | rfl <;>
          rcases hwC with h3 | ⟨h0, h1, h2⟩
        · exact Bool.false_ne_true ((hwsat (3, false) (by simp)).symm.trans h3)
        · exact Bool.false_ne_true ((hwsat (0, false) (by simp)).symm.trans h0)
        · exact Bool.false_ne_true ((hwsat (3, false) (by simp)).symm.trans h3)
        · exact Bool.false_ne_true ((hwsat (1, false) (by simp)).symm.trans h1)
        · exact Bool.false_ne_true ((hwsat (3, false) (by simp)).symm.trans h3)
        · exact Bool.false_ne_true ((hwsat (2, false) (by simp)).symm.trans h2))
      (by
        intro B hconB havoidB
        have h3f : (3, false) ∈ B := by
          by_contra hn3f
          obtain ⟨w, hwsat, hwdef⟩ := exists_word_matching_A hconB (fun _ => true)
          apply havoidB w hwsat
          rw [hcompA]; left
          by_cases h3t : (3, true) ∈ B
          · exact hwsat (3, true) h3t
          · exact hwdef 3 h3t hn3f
        by_cases h0f : (0, false) ∈ B
        · refine ⟨{(3, false), (0, false)}, by simp, ?_⟩
          intro l hl; simp at hl; rcases hl with rfl | rfl <;> assumption
        · by_cases h1f : (1, false) ∈ B
          · refine ⟨{(3, false), (1, false)}, by simp, ?_⟩
            intro l hl; simp at hl; rcases hl with rfl | rfl <;> assumption
          · by_cases h2f : (2, false) ∈ B
            · refine ⟨{(3, false), (2, false)}, by simp, ?_⟩
              intro l hl; simp at hl; rcases hl with rfl | rfl <;> assumption
            · exfalso
              obtain ⟨w, hwsat, hwdef⟩ := exists_word_matching_A hconB (fun _ => true)
              apply havoidB w hwsat
              rw [hcompA]; right
              refine ⟨?_, ?_, ?_⟩
              · by_cases ht : (0, true) ∈ B; exact hwsat (0, true) ht; exact hwdef 0 ht h0f
              · by_cases ht : (1, true) ∈ B; exact hwsat (1, true) ht; exact hwdef 1 ht h1f
              · by_cases ht : (2, true) ∈ B; exact hwsat (2, true) ht; exact hwdef 2 ht h2f)
      (by
        intro g₁ hg₁ g₂ hg₂ hsub
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hg₁ hg₂
        rcases hg₁ with rfl | rfl | rfl <;> rcases hg₂ with rfl | rfl | rfl
        · rfl
        · exfalso; have := hsub (by simp : (0, false) ∈ ({(3, false), (0, false)} : Finset (Lit 4))); simp at this
        · exfalso; have := hsub (by simp : (0, false) ∈ ({(3, false), (0, false)} : Finset (Lit 4))); simp at this
        · exfalso; have := hsub (by simp : (1, false) ∈ ({(3, false), (1, false)} : Finset (Lit 4))); simp at this
        · rfl
        · exfalso; have := hsub (by simp : (1, false) ∈ ({(3, false), (1, false)} : Finset (Lit 4))); simp at this
        · exfalso; have := hsub (by simp : (2, false) ∈ ({(3, false), (2, false)} : Finset (Lit 4))); simp at this
        · exfalso; have := hsub (by simp : (2, false) ∈ ({(3, false), (2, false)} : Finset (Lit 4))); simp at this
        · rfl) A
    simpa using h
  · -- codeB
    have h := isMinGen_iff_mem_list (C := codeB)
      (G := [{(2, true), (3, true)}, {(2, false), (3, false)}])
      (by
        intro g hg; simp only [List.mem_cons, List.not_mem_nil, or_false] at hg
        rcases hg with rfl | rfl
        · exact hcon_pair_ne (by decide)
        · exact hcon_pair_ne (by decide))
      (by
        intro g hg w hwsat hwC
        rw [hcodeB] at hwC
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hg
        rcases hg with rfl | rfl
        · exact hwC ((hwsat (2, true) (by simp)).trans (hwsat (3, true) (by simp)).symm)
        · exact hwC ((hwsat (2, false) (by simp)).trans (hwsat (3, false) (by simp)).symm))
      (by
        intro B hconB havoidB
        by_cases htt : (2, true) ∈ B ∧ (3, true) ∈ B
        · refine ⟨{(2, true), (3, true)}, by simp, ?_⟩
          intro l hl; simp only [Finset.mem_insert, Finset.mem_singleton] at hl
          rcases hl with rfl | rfl
          · exact htt.1
          · exact htt.2
        · by_cases hff : (2, false) ∈ B ∧ (3, false) ∈ B
          · refine ⟨{(2, false), (3, false)}, by simp, ?_⟩
            intro l hl; simp only [Finset.mem_insert, Finset.mem_singleton] at hl
            rcases hl with rfl | rfl
            · exact hff.1
            · exact hff.2
          · exfalso
            by_cases h2t : (2, true) ∈ B
            · have hn3t : (3, true) ∉ B := fun h => htt ⟨h2t, h⟩
              obtain ⟨w, hwsat, hwdef⟩ := exists_word_matching_A hconB (fun _ => false)
              apply havoidB w hwsat; rw [hcodeB]
              have ew2 : w 2 = true := hwsat (2, true) h2t
              have ew3 : w 3 = false := by
                by_cases h3f : (3, false) ∈ B; exact hwsat (3, false) h3f; exact hwdef 3 hn3t h3f
              rw [ew2, ew3]; decide
            · by_cases h2f : (2, false) ∈ B
              · have hn3f : (3, false) ∉ B := fun h => hff ⟨h2f, h⟩
                obtain ⟨w, hwsat, hwdef⟩ := exists_word_matching_A hconB (fun _ => true)
                apply havoidB w hwsat; rw [hcodeB]
                have ew2 : w 2 = false := hwsat (2, false) h2f
                have ew3 : w 3 = true := by
                  by_cases h3t : (3, true) ∈ B; exact hwsat (3, true) h3t; exact hwdef 3 h3t hn3f
                rw [ew2, ew3]; decide
              · by_cases h3t : (3, true) ∈ B
                · obtain ⟨w, hwsat, hwdef⟩ := exists_word_matching_A hconB (fun _ => false)
                  apply havoidB w hwsat; rw [hcodeB]
                  rw [hwdef 2 h2t h2f, hwsat (3, true) h3t]; decide
                · obtain ⟨w, hwsat, hwdef⟩ := exists_word_matching_A hconB (fun i => decide (i = 2))
                  apply havoidB w hwsat; rw [hcodeB]
                  have ew2 : w 2 = true := hwdef 2 h2t h2f
                  have ew3 : w 3 = false := by
                    by_cases h3f : (3, false) ∈ B; exact hwsat (3, false) h3f; exact hwdef 3 h3t h3f
                  rw [ew2, ew3]; decide)
      (by
        intro g₁ hg₁ g₂ hg₂ hsub
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hg₁ hg₂
        rcases hg₁ with rfl | rfl <;> rcases hg₂ with rfl | rfl
        · rfl
        · exfalso; have := hsub (by simp : (2, true) ∈ ({(2, true), (3, true)} : Finset (Lit 4))); simp at this
        · exfalso; have := hsub (by simp : (2, false) ∈ ({(2, false), (3, false)} : Finset (Lit 4))); simp at this
        · rfl) A
    simpa using h
  · -- univ \ codeB
    have h := isMinGen_iff_mem_list (C := Finset.univ \ codeB)
      (G := [{(2, true), (3, false)}, {(2, false), (3, true)}])
      (by
        intro g hg; simp only [List.mem_cons, List.not_mem_nil, or_false] at hg
        rcases hg with rfl | rfl
        · exact hcon_pair_ne (by decide)
        · exact hcon_pair_ne (by decide))
      (by
        intro g hg w hwsat hwC
        rw [hcompB] at hwC
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hg
        rcases hg with rfl | rfl
        · exact Bool.false_ne_true ((hwsat (3, false) (by simp)).symm.trans (hwC.symm.trans (hwsat (2, true) (by simp))))
        · exact Bool.false_ne_true ((hwsat (2, false) (by simp)).symm.trans (hwC.trans (hwsat (3, true) (by simp)))))
      (by
        intro B hconB havoidB
        by_cases htf : (2, true) ∈ B ∧ (3, false) ∈ B
        · refine ⟨{(2, true), (3, false)}, by simp, ?_⟩
          intro l hl; simp only [Finset.mem_insert, Finset.mem_singleton] at hl
          rcases hl with rfl | rfl
          · exact htf.1
          · exact htf.2
        · by_cases hft : (2, false) ∈ B ∧ (3, true) ∈ B
          · refine ⟨{(2, false), (3, true)}, by simp, ?_⟩
            intro l hl; simp only [Finset.mem_insert, Finset.mem_singleton] at hl
            rcases hl with rfl | rfl
            · exact hft.1
            · exact hft.2
          · exfalso
            by_cases h2t : (2, true) ∈ B
            · have hn3f : (3, false) ∉ B := fun h => htf ⟨h2t, h⟩
              obtain ⟨w, hwsat, hwdef⟩ := exists_word_matching_A hconB (fun _ => true)
              apply havoidB w hwsat; rw [hcompB]
              have ew2 : w 2 = true := hwsat (2, true) h2t
              have ew3 : w 3 = true := by
                by_cases h3t : (3, true) ∈ B; exact hwsat (3, true) h3t; exact hwdef 3 h3t hn3f
              rw [ew2, ew3]
            · by_cases h2f : (2, false) ∈ B
              · have hn3t : (3, true) ∉ B := fun h => hft ⟨h2f, h⟩
                obtain ⟨w, hwsat, hwdef⟩ := exists_word_matching_A hconB (fun _ => false)
                apply havoidB w hwsat; rw [hcompB]
                have ew2 : w 2 = false := hwsat (2, false) h2f
                have ew3 : w 3 = false := by
                  by_cases h3f : (3, false) ∈ B; exact hwsat (3, false) h3f; exact hwdef 3 hn3t h3f
                rw [ew2, ew3]
              · by_cases h3t : (3, true) ∈ B
                · obtain ⟨w, hwsat, hwdef⟩ := exists_word_matching_A hconB (fun _ => true)
                  apply havoidB w hwsat; rw [hcompB]
                  rw [hwdef 2 h2t h2f, hwsat (3, true) h3t]
                · obtain ⟨w, hwsat, hwdef⟩ := exists_word_matching_A hconB (fun _ => false)
                  apply havoidB w hwsat; rw [hcompB]
                  have ew2 : w 2 = false := hwdef 2 h2t h2f
                  have ew3 : w 3 = false := by
                    by_cases h3f : (3, false) ∈ B; exact hwsat (3, false) h3f; exact hwdef 3 h3t h3f
                  rw [ew2, ew3])
      (by
        intro g₁ hg₁ g₂ hg₂ hsub
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hg₁ hg₂
        rcases hg₁ with rfl | rfl <;> rcases hg₂ with rfl | rfl
        · rfl
        · exfalso; have := hsub (by simp : (2, true) ∈ ({(2, true), (3, false)} : Finset (Lit 4))); simp at this
        · exfalso; have := hsub (by simp : (2, false) ∈ ({(2, false), (3, true)} : Finset (Lit 4))); simp at this
        · rfl) A
    simpa using h

/-! ### 5. Explicit Achievable Families Covering Every `(p, r)` with `p ≥ 1, r ≥ 3, p + r ≤ 2n` (Theorem 8) -/

/-- The `k` omitted words of the private-literals family (`2 ≤ k ≤ s ≤ n`):
for `j ∈ Fin k`, word `j` has `1` at coordinate `j`, repeats bit `0` on coordinates `k .. s - 1`,
and has `0` on coordinates `s .. n - 1`. -/
def privOmittedWord (n k s : ℕ) (j : Fin k) : Word n :=
  fun i =>
    if i.val < k then decide (i.val = j.val)
    else if i.val < s then decide (j.val = 0)
    else false

/-- **Theorem 8 (Private-literals family properties)**:
For `2 ≤ k ≤ s ≤ n`:
1. Any two distinct omitted words `j₁ ≠ j₂` have Hamming distance `≥ 2` (differing at `j₁` and `j₂`).
2. Every omitted word `j` has a private positive literal `(j, true)` NOT satisfied by any other
   omitted word `j' ≠ j`.
3. Every coordinate `< s` sees both `0` and `1` among the `k` omitted words, while every coordinate
   `≥ s` is constantly `0`, so the union of all `k` full-word generator supports has exact size `n + s`.
   Hence the Taylor resolution is minimal with `(p, r) = (k - 1, n + s - k + 1)`. -/
theorem privateLiterals_family_properties {n k s : ℕ} (hk : 2 ≤ k) (hks : k ≤ s) (hsn : s ≤ n) :
    (∀ j₁ j₂ : Fin k, j₁ ≠ j₂ →
      2 ≤ (Finset.univ.filter fun i : Fin n =>
        privOmittedWord n k s j₁ i ≠ privOmittedWord n k s j₂ i).card) ∧
    (∀ j : Fin k,
      let lit_j : Lit n := (⟨j.val, by omega⟩, true)
      privOmittedWord n k s j lit_j.1 = lit_j.2 ∧
      ∀ j' : Fin k, j' ≠ j → privOmittedWord n k s j' lit_j.1 ≠ lit_j.2) ∧
    (let unionSupp : Finset (Lit n) :=
      Finset.univ.biUnion fun j : Fin k => Finset.univ.image fun i : Fin n => (i, privOmittedWord n k s j i)
     unionSupp.card = n + s) := by
  refine ⟨?_, ?_, ?_⟩
  · intro j₁ j₂ hne
    let i₁ : Fin n := ⟨j₁.val, by omega⟩
    let i₂ : Fin n := ⟨j₂.val, by omega⟩
    have hi_ne : i₁ ≠ i₂ := fun h => hne (Fin.ext (by simpa [i₁, i₂] using congrArg Fin.val h))
    have hsub : ({i₁, i₂} : Finset (Fin n)) ⊆
        Finset.univ.filter fun i : Fin n =>
          privOmittedWord n k s j₁ i ≠ privOmittedWord n k s j₂ i := by
      intro i hi
      simp only [Finset.mem_insert, Finset.mem_singleton] at hi
      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      rcases hi with rfl | rfl
      · simp [privOmittedWord, i₁, j₁.isLt, Fin.val_ne_of_ne hne]
      · simp [privOmittedWord, i₂, j₂.isLt, Ne.symm (Fin.val_ne_of_ne hne)]
    calc 2 = ({i₁, i₂} : Finset (Fin n)).card := (Finset.card_pair hi_ne).symm
      _ ≤ _ := Finset.card_le_card hsub
  · intro j
    dsimp only
    refine ⟨by simp [privOmittedWord, j.isLt], fun j' hne => ?_⟩
    simp [privOmittedWord, j.isLt, Ne.symm (Fin.val_ne_of_ne hne)]
  · dsimp only
    -- Show unionSupp = {(i, false) | i : Fin n} ∪ {(i, true) | i.val < s} (disjoint union of sizes n and s!)
    let S_false : Finset (Lit n) := Finset.univ.image fun i : Fin n => (i, false)
    let S_true : Finset (Lit n) := (Finset.univ.filter fun i : Fin n => i.val < s).image fun i => (i, true)
    have heq : (Finset.univ.biUnion fun j : Fin k =>
        Finset.univ.image fun i : Fin n => (i, privOmittedWord n k s j i)) = S_false ∪ S_true := by
      ext ⟨i, b⟩
      simp only [Finset.mem_biUnion, Finset.mem_univ, true_and, Finset.mem_image, Finset.mem_union,
        Finset.mem_filter, S_false, S_true]
      constructor
      · rintro ⟨j, i', hpair⟩
        have hi' : i' = i := congrArg Prod.fst hpair
        have hb' : privOmittedWord n k s j i' = b := congrArg Prod.snd hpair
        subst i'
        cases b
        · left; exact ⟨i, rfl⟩
        · right
          refine ⟨i, ?_, rfl⟩
          unfold privOmittedWord at hb'
          split_ifs at hb' with h1 h2
          · omega
          · exact h2
      · rintro (⟨i', h⟩ | ⟨i', hi's, h⟩)
        · have hi' : i' = i := congrArg Prod.fst h
          have hb : false = b := congrArg Prod.snd h
          subst i'; subst b
          by_cases h0 : i.val = 0
          · refine ⟨⟨1, by omega⟩, i, ?_⟩
            simp [privOmittedWord, h0, show 0 < k by omega]
          · by_cases hik : i.val < k
            · refine ⟨⟨0, by omega⟩, i, ?_⟩
              simp [privOmittedWord, hik, h0]
            · refine ⟨⟨1, by omega⟩, i, ?_⟩
              simp [privOmittedWord, hik]
        · have hi' : i' = i := congrArg Prod.fst h
          have hb : true = b := congrArg Prod.snd h
          subst i'; subst b
          by_cases hik : i.val < k
          · exact ⟨⟨i.val, hik⟩, i, by simp [privOmittedWord, hik]⟩
          · exact ⟨⟨0, by omega⟩, i, by simp [privOmittedWord, hik, hi's]⟩
    rw [heq, Finset.card_union_of_disjoint]
    · have hcf : S_false.card = n := by
        rw [Finset.card_image_of_injective _ (fun _ _ h => congrArg Prod.fst h)]; simp
      have hct : S_true.card = s := by
        rw [Finset.card_image_of_injective _ (fun _ _ h => congrArg Prod.fst h)]
        have hbij : (Finset.univ.filter fun i : Fin n => i.val < s) =
            Finset.univ.image (fun r : Fin s => (⟨r.val, by omega⟩ : Fin n)) := by
          ext i; simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_image]
          exact ⟨fun h => ⟨⟨i.val, h⟩, rfl⟩, fun ⟨r, hr⟩ => by rw [← hr]; exact r.isLt⟩
        rw [hbij, Finset.card_image_of_injective _ (fun _ _ h => Fin.ext (by simpa using congrArg Fin.val h))]
        simp
      rw [hcf, hct]
    · rw [Finset.disjoint_left]
      intro l h1 h2
      simp only [S_false, S_true, Finset.mem_image] at h1 h2
      rcases h1 with ⟨_, _, rfl⟩; rcases h2 with ⟨_, _, h⟩
      exact Bool.false_ne_true (congrArg Prod.snd h).symm

/-- Inductive specification of achievable `(p, r)` pairs generated by the two explicit diagonal
families of Theorem 8 plus the three elementary operations (free-coordinate padding, single-layer
restriction `r ↦ r + 1`, and fixed-coordinate addition `p ↦ p + 1`). -/
inductive AchievablePair : ℕ → ℕ → ℕ → Prop
  | privateLiterals {n k s : ℕ} (hk : 2 ≤ k) (hks : k ≤ s) (hsn : s ≤ n) :
      AchievablePair n (k - 1) (n + s - k + 1)
  | compDiagonal {n k s : ℕ} (hk : 2 ≤ k) (hks : k ≤ s) (hsn : s ≤ n) :
      AchievablePair n (n + s - k - 1) (k + 1)
  | padFree {n m p r : ℕ} (hmn : m ≤ n) :
      AchievablePair m p r → AchievablePair n p r
  | addLayer {m p r : ℕ} :
      AchievablePair m p r → AchievablePair (m + 1) p (r + 1)
  | fixCoord {m p r : ℕ} :
      AchievablePair m p r → AchievablePair (m + 1) (p + 1) r

/-- **Theorem 8 (Part 4 — Complete coverage of `p ≥ 1, r ≥ 3, p + r ≤ 2n`)**:
For EVERY `n ≥ 2` and EVERY `(p, r)` with `1 ≤ p`, `3 ≤ r`, and `p + r ≤ 2 * n`,
`(p, r)` is realized by the explicit achievable families of Theorem 8! -/
theorem achievable_region_le_2n_coverage {n p r : ℕ}
    (hp : 1 ≤ p) (hr : 3 ≤ r) (hsum : p + r ≤ 2 * n) :
    AchievablePair n p r := by
  -- Helper: even diagonal p + r = 2 * m with 1 ≤ p, 3 ≤ r
  have heven : ∀ m a b : ℕ, 1 ≤ a → 3 ≤ b → a + b = 2 * m → AchievablePair m a b := by
    intro m a b ha hb hab
    by_cases hhalf : a ≤ m - 1
    · -- Lower half: privateLiterals with n' = s = m, k = a + 1
      have hk2 : 2 ≤ a + 1 := by omega
      have hkm : a + 1 ≤ m := by omega
      have h := AchievablePair.privateLiterals hk2 hkm le_rfl
      have hp_eq : a + 1 - 1 = a := by omega
      have hr_eq : m + m - (a + 1) + 1 = b := by omega
      rwa [hp_eq, hr_eq] at h
    · -- Upper half: compDiagonal with n' = s = m, k = b - 1
      have hk2 : 2 ≤ b - 1 := by omega
      have hkm : b - 1 ≤ m := by omega
      have h := AchievablePair.compDiagonal hk2 hkm le_rfl
      have hp_eq : m + m - (b - 1) - 1 = a := by omega
      have hr_eq : b - 1 + 1 = b := by omega
      rwa [hp_eq, hr_eq] at h
  rcases Nat.even_or_odd (p + r) with ⟨m, hm⟩ | ⟨m, hm⟩
  · -- Even: p + r = 2m, m ≤ n
    have hmn : m ≤ n := by omega
    exact AchievablePair.padFree hmn (heven m p r hp hr (by omega))
  · -- Odd: p + r = 2m + 1, m + 1 ≤ n
    have hmn : m + 1 ≤ n := by omega
    by_cases hr4 : 4 ≤ r
    · -- Realize (p, r - 1) on m coordinates, then addLayer
      have hbase := heven m p (r - 1) hp (by omega) (by omega)
      have hstep := AchievablePair.addLayer hbase
      have hr_eq : r - 1 + 1 = r := by omega
      rw [hr_eq] at hstep
      exact AchievablePair.padFree hmn hstep
    · -- r = 3, so p = 2m - 2 ≥ 2 (since p + 3 = 2m + 1 implies p is even ≥ 2)
      have hr3 : r = 3 := by omega
      subst hr3
      have hbase := heven m (p - 1) 3 (by omega) le_rfl (by omega)
      have hstep := AchievablePair.fixCoord hbase
      have hp_eq : p - 1 + 1 = p := by omega
      rw [hp_eq] at hstep
      exact AchievablePair.padFree hmn hstep

end NeuralCodeHomological
