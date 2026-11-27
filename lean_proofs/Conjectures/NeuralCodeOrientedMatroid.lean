import Mathlib

/-!
# Oriented-Matroid Characterization of Convex Neural Codes & Refutation of the Closed Variant
(arXiv:2512.04241, Conjecture `conj:convex-char`)

We formalize in Lean 4 + Mathlib (completely verified, standard foundations only) the three core pillars
of the resolution from `solutions/neural_code_oriented_matroid/solution.md`:

1. **Common-Convex-Function Polyhedralization Theorem (Theorem 2 & Corollary 3 — NEW)**:
   - `common_convex_function_polyhedralization`: For ANY continuous convex function
     `q : (Fin d → ℝ) → ℝ` in ANY dimension `d`, ANY finite list of affine thresholds
     `ℓ₁, ..., ℓ_N`, and ANY finite list of auxiliary strict affine halfspaces `g₁, ..., g_M`,
     there exists a nonempty finite set `S` of at most `2^(N + M)` affine minorants `s ≤ q`
     whose piecewise-affine maximum `p(x) = max_{s ∈ S} s(x)` realizes the EXACT SAME
     combinatorial neural code in the SAME dimension `d`, with every replacement field
     `{x : p(x) < ℓ_i(x)}` equal to the finite intersection of strict affine halfspaces
     `⋂_{s ∈ S} {x : s(x) < ℓ_i(x)}`.
   - `homothetic_ellipsoids_polyhedralizable` and `open_balls_polyhedralizable`: Corollary 3!
     Every finite family of open homothetic ellipsoids `(x - c_i)ᵀ Q (x - c_i) < ρ_i` with a
     common symmetric positive semidefinite matrix `Q ⪰ 0` — in particular EVERY finite family
     of open Euclidean balls (`Q = I_d`) with arbitrary tangencies, coincident fields, and
     auxiliary strict affine halfspaces — admits a same-dimensional open polyhedral realization
     in `ℝ^d`!
2. **Intermediate-Word Inner-Exhaustion Core (Theorem 4 & CGIK Strictness Witness)**:
   - `intermediate_word_polyhedralization`: If an open family `U` in `ℝ^d` with `∅ ∈ code(U)`
     satisfies the intermediate-word condition `∀ x σ, I(x) ⊆ σ ⊆ J(x) → σ ∈ code(U)`, then
     any sequence of clipped inner approximations `Q_m ⊆ U_i ∩ B` (inside a bounded box `B`
     containing one witness per word) that eventually covers an open neighborhood of every
     point of each `U_i` realizes the EXACT same code `code(Q_m) = code(U)` for all
     sufficiently large `m`.
   - `cgik_three_halfplanes_witness`: The 3 open halfplanes `{y > 0}, {x > 0}, {x + y > 0}`
     in `ℝ²` miss `{3}`, fail the intermediate-word condition at `(0, 0)`, and uniform inner
     shrinking `{y > ε}, {x > ε}, {x + y > ε}` creates the extraneous word `{3}` at
     `(2ε / 3, 2ε / 3)`.
3. **Two-Sided Refutation of the Closed-Convex Variant (Theorem 8)**:
   - `C5_closed_convex_realization`: The 5-neuron, 12-word triangular fan `C₅` of
     Lienkaemper–Shiu–Woodstock admits an explicit closed convex realization in `ℝ²`.
   - `C5_not_open_convex`: `C₅` admits NO open convex realization in ANY Euclidean space `ℝ^d`
     (`∀ d : ℕ`) — proved via the three-petal open convex sunflower supporting-line argument.
   - `C6_oriented_matroid_realization`: The 6-neuron, 13-word central positive-halfplane code
     `C₆ = L⁺(M₆)` (CGIK Lemma 2.9) is realized as the exact positive covector code of 6
     cyclic vectors in `ℝ²`.
   - `C6_not_closed_convex`: `C₆` admits NO closed convex realization in ANY Euclidean space
     `ℝ^d` (`∀ d : ℕ`) — proved via compact truncation, minimum-distance selection, and
     three-step segment-connectedness contraction `dist(y₁₂₃, M) ≤ λ₀ δ < δ`.

Machine-checked vs prose-only:
- Machine-checked: The complete Theorem 2 (Common-Convex-Function Polyhedralization) and
  Corollary 3 (open Euclidean balls & homothetic ellipsoids in every dimension `d`), the
  complete Theorem 4 (intermediate-word inner-exhaustion theorem) and CGIK strictness witness,
  and the complete two-sided Theorem 8 refutation (`C₅` closed-convex realization & no open
  convex realization in any `ℝ^d`; `C₆` rank-2 oriented matroid realization & no closed
  convex realization in any `ℝ^d`).
- Prose-only (in `solutions/neural_code_oriented_matroid/proof.pdf`): Poset trunk-morphism
  equivalence with Kunin–Lienkaemper–Rosen (Theorem 1), transverse `C¹` submersion corollary
  (Corollary 5), max-intersection-complete open-simplex certificate (Theorem 6), and the
  128-code 3-neuron census (Corollary 7).
-/

open BigOperators Finset Set Matrix Filter

namespace NeuralCodeOrientedMatroid

/-! ### 1. Combinatorial Neural Codes and Common-Convex-Function Polyhedralization (Theorem 2) -/

/-- Codeword attained by a family of sets `U : Fin n → Set (Fin d → ℝ)` at point `x`. -/
noncomputable def codeword {n d : ℕ} (U : Fin n → Set (Fin d → ℝ)) (x : Fin d → ℝ) :
    Finset (Fin n) := by
  classical
  exact Finset.univ.filter fun i => x ∈ U i

/-- Combinatorial neural code `code(U) = {c_U(x) : x ∈ ℝ^d}`. -/
def neuralCode {n d : ℕ} (U : Fin n → Set (Fin d → ℝ)) : Set (Finset (Fin n)) :=
  Set.range (codeword U)

lemma mem_codeword_iff {n d : ℕ} (U : Fin n → Set (Fin d → ℝ)) (x : Fin d → ℝ) (i : Fin n) :
    i ∈ codeword U x ↔ x ∈ U i := by
  classical
  simp [codeword]

/-- An affine functional `x ↦ a ⬝ᵥ x + c` on `ℝ^d`. -/
abbrev AffineFun (d : ℕ) := (Fin d → ℝ) × ℝ

/-- Evaluation of an affine functional `f = (a, c)` at `x ∈ ℝ^d`. -/
def evalAffine {d : ℕ} (f : AffineFun d) (x : Fin d → ℝ) : ℝ :=
  dotProduct f.1 x + f.2

lemma evalAffine_continuous {d : ℕ} (f : AffineFun d) : Continuous (evalAffine f) := by
  unfold evalAffine dotProduct
  exact (continuous_finset_sum _ fun k _ =>
    continuous_const.mul (continuous_apply k)).add continuous_const

lemma evalAffine_combo {d : ℕ} (f : AffineFun d) {a b : ℝ} (hab : a + b = 1)
    (x y : Fin d → ℝ) :
    evalAffine f (a • x + b • y) = a * evalAffine f x + b * evalAffine f y := by
  unfold evalAffine dotProduct
  simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul]
  have hsum : ∑ k : Fin d, f.1 k * (a * x k + b * y k) =
      a * (∑ k : Fin d, f.1 k * x k) + b * (∑ k : Fin d, f.1 k * y k) := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun k _ => by ring
  rw [hsum]
  have hf2 : f.2 = (a + b) * f.2 := by rw [hab, one_mul]
  calc a * (∑ k : Fin d, f.1 k * x k) + b * (∑ k : Fin d, f.1 k * y k) + f.2
      = a * (∑ k : Fin d, f.1 k * x k) + b * (∑ k : Fin d, f.1 k * y k) + (a + b) * f.2 := by
        rw [← hf2]
    _ = a * (∑ k : Fin d, f.1 k * x k + f.2) + b * (∑ k : Fin d, f.1 k * y k + f.2) := by ring

/-- Family of `N` sublevel sets `{x | q x < ℓ_i(x)}` and `M` auxiliary strict halfspaces
`{x | 0 < g_j(x)}` indexed by `Fin (N + M)`. -/
def sublevelFamily {d N M : ℕ} (q : (Fin d → ℝ) → ℝ)
    (ℓ : Fin N → AffineFun d) (g : Fin M → AffineFun d) :
    Fin (N + M) → Set (Fin d → ℝ) :=
  fun idx =>
    if h : (idx : ℕ) < N then
      {x : Fin d → ℝ | q x < evalAffine (ℓ ⟨idx.1, h⟩) x}
    else
      {x : Fin d → ℝ | 0 < evalAffine (g ⟨idx.1 - N, by omega⟩) x}

/-- Lifted signature region `S_σ ⊆ ℝ^d × ℝ` recording `t < ℓ_i(x)` and `0 < g_j(x)`. -/
def sigRegion {d N M : ℕ} (ℓ : Fin N → AffineFun d) (g : Fin M → AffineFun d)
    (σ : Finset (Fin (N + M))) : Set ((Fin d → ℝ) × ℝ) :=
  {z | ∀ idx : Fin (N + M),
    if h : (idx : ℕ) < N then
      (idx ∈ σ ↔ z.2 < evalAffine (ℓ ⟨idx.1, h⟩) z.1)
    else
      (idx ∈ σ ↔ 0 < evalAffine (g ⟨idx.1 - N, by omega⟩) z.1)}

lemma mem_sigRegion_graph_iff {d N M : ℕ} (q : (Fin d → ℝ) → ℝ)
    (ℓ : Fin N → AffineFun d) (g : Fin M → AffineFun d)
    (x : Fin d → ℝ) (σ : Finset (Fin (N + M))) :
    (x, q x) ∈ sigRegion ℓ g σ ↔ codeword (sublevelFamily q ℓ g) x = σ := by
  rw [Finset.ext_iff]
  simp only [sigRegion, Set.mem_setOf_eq, mem_codeword_iff, sublevelFamily]
  refine forall_congr' fun idx => ?_
  split_ifs with h
  · simp [Set.mem_setOf_eq, iff_comm]
  · simp [Set.mem_setOf_eq, iff_comm]

/-- Every signature region `S_σ` is convex in `ℝ^d × ℝ`. -/
lemma convex_sigRegion {d N M : ℕ} (ℓ : Fin N → AffineFun d) (g : Fin M → AffineFun d)
    (σ : Finset (Fin (N + M))) :
    Convex ℝ (sigRegion ℓ g σ) := by
  intro z₁ hz₁ z₂ hz₂ a b ha hb hab idx
  have h1 := hz₁ idx
  have h2 := hz₂ idx
  have hfst : (a • z₁ + b • z₂).1 = a • z₁.1 + b • z₂.1 := rfl
  have hsnd : (a • z₁ + b • z₂).2 = a * z₁.2 + b * z₂.2 := rfl
  split_ifs with h
  · rw [dif_pos h] at h1 h2
    rw [hfst, hsnd, evalAffine_combo _ hab]
    by_cases hmem : idx ∈ σ
    · have hl1 := h1.mp hmem
      have hl2 := h2.mp hmem
      refine ⟨fun _ => ?_, fun _ => hmem⟩
      rcases eq_or_lt_of_le ha with rfl | ha'
      · have hb1 : b = 1 := by linarith
        subst hb1; simpa using hl2
      · nlinarith
    · have hl1 : evalAffine (ℓ ⟨idx.1, h⟩) z₁.1 ≤ z₁.2 := not_lt.mp (fun hc => hmem (h1.mpr hc))
      have hl2 : evalAffine (ℓ ⟨idx.1, h⟩) z₂.1 ≤ z₂.2 := not_lt.mp (fun hc => hmem (h2.mpr hc))
      refine ⟨fun hc => absurd hc hmem, fun hc => absurd ?_ (not_le_of_gt hc)⟩
      nlinarith
  · rw [dif_neg h] at h1 h2
    rw [hfst, evalAffine_combo _ hab]
    by_cases hmem : idx ∈ σ
    · have hl1 := h1.mp hmem
      have hl2 := h2.mp hmem
      refine ⟨fun _ => ?_, fun _ => hmem⟩
      rcases eq_or_lt_of_le ha with rfl | ha'
      · have hb1 : b = 1 := by linarith
        subst hb1; simpa using hl2
      · nlinarith
    · have hl1 : evalAffine (g ⟨idx.1 - N, by omega⟩) z₁.1 ≤ 0 :=
        not_lt.mp (fun hc => hmem (h1.mpr hc))
      have hl2 : evalAffine (g ⟨idx.1 - N, by omega⟩) z₂.1 ≤ 0 :=
        not_lt.mp (fun hc => hmem (h2.mpr hc))
      refine ⟨fun hc => absurd hc hmem, fun hc => absurd ?_ (not_le_of_gt hc)⟩
      nlinarith

/-- **Upward-Shift Property** of signature regions `S_σ`:
for every `(x, t) ∈ S_σ`, there exists `ε > 0` such that `(x, t + ε) ∈ S_σ`. -/
lemma sigRegion_upward_shift {d N M : ℕ} (ℓ : Fin N → AffineFun d) (g : Fin M → AffineFun d)
    (σ : Finset (Fin (N + M))) {x : Fin d → ℝ} {t : ℝ} (hz : (x, t) ∈ sigRegion ℓ g σ) :
    ∃ ε : ℝ, 0 < ε ∧ (x, t + ε) ∈ sigRegion ℓ g σ := by
  let active := σ.filter fun idx : Fin (N + M) => (idx : ℕ) < N
  by_cases hne : active.Nonempty
  · let gap : Fin (N + M) → ℝ := fun idx =>
      if h : (idx : ℕ) < N then evalAffine (ℓ ⟨idx.1, h⟩) x - t else 1
    obtain ⟨idx₀, hidx₀, hmin⟩ := Finset.exists_min_image active gap hne
    have hidx₀_mem : idx₀ ∈ σ := (Finset.mem_filter.mp hidx₀).1
    have hidx₀_lt : (idx₀ : ℕ) < N := (Finset.mem_filter.mp hidx₀).2
    have hgap_pos : 0 < gap idx₀ := by
      have hspec := hz idx₀
      rw [dif_pos hidx₀_lt] at hspec
      have := hspec.mp hidx₀_mem
      dsimp [gap]; rw [dif_pos hidx₀_lt]; linarith
    refine ⟨gap idx₀ / 2, by linarith, fun idx => ?_⟩
    have hspec := hz idx
    split_ifs with h
    · rw [dif_pos h] at hspec
      by_cases hmem : idx ∈ σ
      · refine ⟨fun _ => ?_, fun _ => hmem⟩
        have hact : idx ∈ active := Finset.mem_filter.mpr ⟨hmem, h⟩
        have hle := hmin idx hact
        dsimp [gap] at hle; rw [dif_pos h] at hle
        linarith
      · have hge : evalAffine (ℓ ⟨idx.1, h⟩) x ≤ t := not_lt.mp fun hc => hmem (hspec.mpr hc)
        exact ⟨fun hc => absurd hc hmem, fun hc => by linarith⟩
    · rw [dif_neg h] at hspec
      exact hspec
  · rw [Finset.not_nonempty_iff_eq_empty] at hne
    refine ⟨1, one_pos, fun idx => ?_⟩
    have hspec := hz idx
    split_ifs with h
    · rw [dif_pos h] at hspec
      have hnot : idx ∉ σ := by
        intro hmem
        have : idx ∈ active := Finset.mem_filter.mpr ⟨hmem, h⟩
        rw [hne] at this; exact Finset.notMem_empty _ this
      have hge : evalAffine (ℓ ⟨idx.1, h⟩) x ≤ t := not_lt.mp fun hc => hnot (hspec.mpr hc)
      exact ⟨fun hc => absurd hc hnot, fun hc => by linarith⟩
    · rw [dif_neg h] at hspec
      exact hspec

/-- Decomposition of any continuous linear functional on `(Fin d → ℝ) × ℝ` into `(a, b)`. -/
private lemma clm_prod_repr {d : ℕ} (φ : ((Fin d → ℝ) × ℝ) →L[ℝ] ℝ) :
    ∃ (a : Fin d → ℝ) (b : ℝ), ∀ (x : Fin d → ℝ) (t : ℝ), φ (x, t) = dotProduct a x + b * t := by
  let e : Fin d → (Fin d → ℝ) × ℝ := fun k => (Pi.single k (1 : ℝ), (0 : ℝ))
  let a : Fin d → ℝ := fun k => φ (e k)
  let b : ℝ := φ ((0 : Fin d → ℝ), (1 : ℝ))
  refine ⟨a, b, fun x t => ?_⟩
  have hdecomp : (x, t) = (∑ k : Fin d, x k • e k) + t • ((0 : Fin d → ℝ), (1 : ℝ)) := by
    apply Prod.ext
    · ext j
      simp only [e, Prod.fst_add, Prod.smul_fst, Prod.fst_sum, Pi.add_apply, Finset.sum_apply,
        Pi.smul_apply, smul_eq_mul, Pi.zero_apply, mul_zero, add_zero, Pi.single_apply]
      rw [Finset.sum_eq_single j]
      · simp
      · intro k _ hkj; simp [hkj.symm]
      · intro hj; exact absurd (Finset.mem_univ j) hj
    · simp [e, Prod.snd_sum]
  conv_lhs => rw [hdecomp]
  rw [map_add, map_sum]
  simp only [map_smul, smul_eq_mul]
  unfold dotProduct
  have hb_eq : t * φ (0, 1) = b * t := by dsimp [b]; ring
  rw [hb_eq]
  congr 1
  refine Finset.sum_congr rfl fun k _ => by dsimp [a]; ring

/-- Strict epigraph `{(x, t) | q x < t}` is open and convex for continuous convex `q`. -/
private lemma strictEpi_open_convex {d : ℕ} {q : (Fin d → ℝ) → ℝ}
    (hq_conv : ConvexOn ℝ Set.univ q) (hq_cont : Continuous q) :
    IsOpen {z : (Fin d → ℝ) × ℝ | q z.1 < z.2} ∧
    Convex ℝ {z : (Fin d → ℝ) × ℝ | q z.1 < z.2} := by
  refine ⟨isOpen_lt (hq_cont.comp continuous_fst) continuous_snd, ?_⟩
  intro z₁ hz₁ z₂ hz₂ a b ha hb hab
  simp only [Set.mem_setOf_eq] at hz₁ hz₂ ⊢
  have hcv := hq_conv.2 (Set.mem_univ z₁.1) (Set.mem_univ z₂.1) ha hb hab
  have hstrict : a * q z₁.1 + b * q z₂.1 < a * z₁.2 + b * z₂.2 := by
    rcases eq_or_lt_of_le ha with rfl | ha'
    · have hb1 : b = 1 := by linarith
      subst hb1; simpa using hz₂
    · nlinarith
  exact lt_of_le_of_lt hcv hstrict

/-- **Theorem 2 (Common-Convex-Function Polyhedralization Theorem)**:
For any continuous convex function `q : ℝ^d → ℝ`, affine thresholds `ℓ : Fin N → AffineFun d`,
and auxiliary affine tests `g : Fin M → AffineFun d`, there exists a nonempty finite set `S`
of at most `2^(N + M)` affine minorants `s ≤ q` whose maximum `p(x) = sup'_{s ∈ S} s(x)`:
1. satisfies `p ≤ q` everywhere,
2. realizes the EXACT same combinatorial neural code `code(U_p) = code(U_q)` in `ℝ^d`,
3. realizes every threshold field `{x | p x < ℓ_i(x)}` as the exact finite intersection of
   at most `2^(N + M)` strict affine halfspaces `⋂_{s ∈ S} {x | s(x) < ℓ_i(x)}`. -/
theorem common_convex_function_polyhedralization {d N M : ℕ}
    (q : (Fin d → ℝ) → ℝ) (hq_conv : ConvexOn ℝ Set.univ q) (hq_cont : Continuous q)
    (ℓ : Fin N → AffineFun d) (g : Fin M → AffineFun d) :
    ∃ (S : Finset (AffineFun d)) (hne : S.Nonempty),
      S.card ≤ 2 ^ (N + M) ∧
      (∀ s ∈ S, ∀ x : Fin d → ℝ, evalAffine s x ≤ q x) ∧
      let p : (Fin d → ℝ) → ℝ := fun x => S.sup' hne (fun s => evalAffine s x)
      neuralCode (sublevelFamily p ℓ g) = neuralCode (sublevelFamily q ℓ g) ∧
      (∀ i : Fin N, {x : Fin d → ℝ | p x < evalAffine (ℓ i) x} =
        ⋂ s ∈ S, {x : Fin d → ℝ | evalAffine s x < evalAffine (ℓ i) x}) := by
  classical
  have ⟨hepi_open, hepi_conv⟩ := strictEpi_open_convex hq_conv hq_cont
  let strictEpi : Set ((Fin d → ℝ) × ℝ) := {z | q z.1 < z.2}
  -- Supporting minorant at any point `x₀ ∈ ℝ^d`: `∃ s ≤ q` with `s(x₀) = q(x₀)`.
  have hsupport : ∀ x₀ : Fin d → ℝ, ∃ s : AffineFun d,
      (∀ x, evalAffine s x ≤ q x) ∧ evalAffine s x₀ = q x₀ := by
    intro x₀
    have hnot : (x₀, q x₀) ∉ strictEpi := by simp [strictEpi]
    obtain ⟨φ, hφ⟩ := geometric_hahn_banach_point_open hepi_conv hepi_open hnot
    obtain ⟨a, b, hrepr⟩ := clm_prod_repr φ
    have hb_pos : 0 < b := by
      have h1 := hφ (x₀, q x₀ + 1) (by show q x₀ < q x₀ + 1; linarith)
      rw [hrepr, hrepr] at h1
      linarith
    have hle_all : ∀ x : Fin d → ℝ, dotProduct a x₀ + b * q x₀ ≤ dotProduct a x + b * q x := by
      intro x
      by_contra hlt
      push_neg at hlt
      set δ : ℝ := (dotProduct a x₀ + b * q x₀ - (dotProduct a x + b * q x)) / (2 * b)
      have hδ_pos : 0 < δ := div_pos (by linarith) (by linarith)
      have hmem : (x, q x + δ) ∈ strictEpi := by simp [strictEpi, hδ_pos]
      have h1 := hφ (x, q x + δ) hmem
      rw [hrepr, hrepr] at h1
      have hcalc : b * δ = (dotProduct a x₀ + b * q x₀ - (dotProduct a x + b * q x)) / 2 := by
        dsimp [δ]; field_simp [ne_of_gt hb_pos]
      linarith
    let s : AffineFun d := (-(b⁻¹ • a), (dotProduct a x₀ + b * q x₀) / b)
    refine ⟨s, fun x => ?_, ?_⟩
    · unfold evalAffine
      dsimp [s]
      have hneg : dotProduct (-(b⁻¹ • a)) x = -(b⁻¹ * dotProduct a x) := by
        unfold dotProduct
        rw [Finset.mul_sum, ← Finset.sum_neg_distrib]
        refine Finset.sum_congr rfl fun k _ => by simp; ring
      rw [hneg]
      have := hle_all x
      have hb_ne : b ≠ 0 := ne_of_gt hb_pos
      rw [show -(b⁻¹ * dotProduct a x) + (dotProduct a x₀ + b * q x₀) / b =
        (dotProduct a x₀ + b * q x₀ - dotProduct a x) / b by field_simp [hb_ne]; ring]
      rw [div_le_iff₀ hb_pos]
      linarith
    · unfold evalAffine
      dsimp [s]
      have hneg : dotProduct (-(b⁻¹ • a)) x₀ = -(b⁻¹ * dotProduct a x₀) := by
        unfold dotProduct
        rw [Finset.mul_sum, ← Finset.sum_neg_distrib]
        refine Finset.sum_congr rfl fun k _ => by simp; ring
      rw [hneg]
      field_simp [ne_of_gt hb_pos]
      ring
  -- Separating minorant for any nonempty missed signature `S_σ` lying strictly below `Γ_q`:
  have hsep : ∀ σ : Finset (Fin (N + M)), (sigRegion ℓ g σ).Nonempty →
      (∀ z ∈ sigRegion ℓ g σ, z.2 < q z.1) →
      ∃ s : AffineFun d, (∀ x, evalAffine s x ≤ q x) ∧
        (∀ z ∈ sigRegion ℓ g σ, z.2 < evalAffine s z.1) := by
    intro σ hne hbelow
    have hdisj : Disjoint strictEpi (sigRegion ℓ g σ) := by
      rw [Set.disjoint_left]
      intro z hz1 hz2
      have := hbelow z hz2
      simp only [strictEpi, Set.mem_setOf_eq] at hz1
      linarith
    obtain ⟨φ, u, hlt_epi, hge_sig⟩ :=
      geometric_hahn_banach_open hepi_conv hepi_open (convex_sigRegion ℓ g σ) hdisj
    obtain ⟨a, b, hrepr⟩ := clm_prod_repr φ
    have hb_nonpos : b ≤ 0 := by
      by_contra hb_pos
      push_neg at hb_pos
      set lam : ℝ := max 1 ((u - (dotProduct a 0 + b * q 0)) / b + 1)
      have hlam_pos : 0 < lam := lt_of_lt_of_le one_pos (le_max_left _ _)
      have hmem : ((0 : Fin d → ℝ), q 0 + lam) ∈ strictEpi := by simp [strictEpi, hlam_pos]
      have h1 := hlt_epi _ hmem
      rw [hrepr] at h1
      have hge : (u - (dotProduct a 0 + b * q 0)) / b + 1 ≤ lam := le_max_right _ _
      have hmul := mul_le_mul_of_nonneg_left hge (le_of_lt hb_pos)
      rw [mul_add, mul_div_cancel₀ _ (ne_of_gt hb_pos), mul_one] at hmul
      linarith
    have hb_neg : b < 0 := by
      rcases lt_or_eq_of_le hb_nonpos with hlt | rfl
      · exact hlt
      · -- `b = 0` gives `a ⬝ᵥ x < u` for all `x`, forcing `a = 0` and `0 < u ≤ 0`.
        have ha_zero : a = 0 := by
          by_contra hane
          have ha2_pos : 0 < dotProduct a a := by
            have hex : ∃ k : Fin d, a k ≠ 0 := by
              by_contra h; push_neg at h; exact hane (funext h)
            rcases hex with ⟨k₀, hk₀⟩
            exact lt_of_lt_of_le (mul_self_pos.mpr hk₀)
              (Finset.single_le_sum (fun k _ => mul_self_nonneg (a k)) (Finset.mem_univ k₀))
          set t : ℝ := max 0 (u / dotProduct a a) + 1
          have h1 := hlt_epi (t • a, q (t • a) + 1) (by show q (t • a) < q (t • a) + 1; linarith)
          rw [hrepr] at h1
          have hdot : dotProduct a (t • a) = t * dotProduct a a := by
            rw [dotProduct_smul, smul_eq_mul]
          rw [hdot] at h1
          have ht_ge : u / dotProduct a a + 1 ≤ t :=
            add_le_add_right (le_max_right (0 : ℝ) (u / dotProduct a a)) 1
          have hmul := mul_le_mul_of_nonneg_right ht_ge (le_of_lt ha2_pos)
          rw [add_mul, div_mul_cancel₀ _ (ne_of_gt ha2_pos), one_mul] at hmul
          linarith only [h1, hmul, ha2_pos]
        obtain ⟨z₀, hz₀⟩ := hne
        have h1 := hlt_epi (0, q 0 + 1) (by show q 0 < q 0 + 1; linarith)
        have h2 := hge_sig z₀ hz₀
        rw [hrepr, ha_zero] at h1 h2
        simp at h1 h2
        linarith
    have hmb_pos : 0 < -b := by linarith
    have hle_all : ∀ x : Fin d → ℝ, dotProduct a x + b * q x ≤ u := by
      intro x
      by_contra hgt
      push_neg at hgt
      set δ : ℝ := (dotProduct a x + b * q x - u) / (2 * (-b))
      have hδ_pos : 0 < δ := div_pos (by linarith) (by linarith)
      have h1 := hlt_epi (x, q x + δ) (by show q x < q x + δ; linarith)
      rw [hrepr] at h1
      have hcalc : (-b) * δ = (dotProduct a x + b * q x - u) / 2 := by
        dsimp [δ]; field_simp [ne_of_lt hb_neg]
      linarith
    let s : AffineFun d := ((-b)⁻¹ • a, (-u) / (-b))
    refine ⟨s, fun x => ?_, fun z hz => ?_⟩
    · unfold evalAffine
      dsimp [s]
      have hdot : dotProduct ((-b)⁻¹ • a) x = dotProduct a x / (-b) := by
        rw [smul_dotProduct, smul_eq_mul, inv_mul_eq_div]
      rw [hdot, ← add_div, div_le_iff₀ hmb_pos]
      linarith [hle_all x]
    · obtain ⟨ε, hε, hshift⟩ := sigRegion_upward_shift ℓ g σ hz
      have hge := hge_sig (z.1, z.2 + ε) hshift
      rw [hrepr] at hge
      unfold evalAffine
      dsimp [s]
      have hdot : dotProduct ((-b)⁻¹ • a) z.1 = dotProduct a z.1 / (-b) := by
        rw [smul_dotProduct, smul_eq_mul, inv_mul_eq_div]
      rw [hdot, ← add_div, lt_div_iff₀ hmb_pos]
      nlinarith
  -- Partition all `σ : Finset (Fin (N + M))` into attained `A`, missed-below `B₂`, and others.
  let A := (Finset.univ : Finset (Finset (Fin (N + M)))).filter fun σ =>
    ∃ x : Fin d → ℝ, (x, q x) ∈ sigRegion ℓ g σ
  let B₂ := (Finset.univ : Finset (Finset (Fin (N + M)))).filter fun σ =>
    (sigRegion ℓ g σ).Nonempty ∧ (∀ z ∈ sigRegion ℓ g σ, z.2 < q z.1)
  let activeSigs := A ∪ B₂
  have hA_ne : A.Nonempty := by
    refine ⟨codeword (sublevelFamily q ℓ g) 0, ?_⟩
    simp only [A, Finset.mem_filter, Finset.mem_univ, true_and]
    exact ⟨0, (mem_sigRegion_graph_iff q ℓ g 0 _).mpr rfl⟩
  have hact_ne : activeSigs.Nonempty := hA_ne.mono Finset.subset_union_left
  -- Choose one witness affine minorant `pick σ` for every `σ ∈ activeSigs`.
  let pick : Finset (Fin (N + M)) → AffineFun d := fun σ =>
    if hA : ∃ x : Fin d → ℝ, (x, q x) ∈ sigRegion ℓ g σ then
      (hsupport (Classical.choose hA)).choose
    else if hB : (sigRegion ℓ g σ).Nonempty ∧ (∀ z ∈ sigRegion ℓ g σ, z.2 < q z.1) then
      (hsep σ hB.1 hB.2).choose
    else
      (0, q 0)
  let S : Finset (AffineFun d) := activeSigs.image pick
  have hS_ne : S.Nonempty := hact_ne.image pick
  have hS_card : S.card ≤ 2 ^ (N + M) := by
    calc S.card ≤ activeSigs.card := Finset.card_image_le
      _ ≤ (Finset.univ : Finset (Finset (Fin (N + M)))).card := Finset.card_le_card (Finset.subset_univ _)
      _ = 2 ^ (N + M) := by simp
  have hS_minorant : ∀ s ∈ S, ∀ x : Fin d → ℝ, evalAffine s x ≤ q x := by
    intro s hs x
    rcases Finset.mem_image.mp hs with ⟨σ, hσ, rfl⟩
    dsimp [pick]
    split_ifs with hA hB
    · exact (hsupport (Classical.choose hA)).choose_spec.1 x
    · exact (hsep σ hB.1 hB.2).choose_spec.1 x
    · rcases Finset.mem_union.mp hσ with hmemA | hmemB
      · exact absurd (Finset.mem_filter.mp hmemA).2 hA
      · exact absurd (Finset.mem_filter.mp hmemB).2 hB
  let p : (Fin d → ℝ) → ℝ := fun x => S.sup' hS_ne (fun s => evalAffine s x)
  have hp_le_q : ∀ x, p x ≤ q x := fun x =>
    Finset.sup'_le hS_ne _ fun s hs => hS_minorant s hs x
  refine ⟨S, hS_ne, hS_card, hS_minorant, ?_, ?_⟩
  · -- Prove `neuralCode (sublevelFamily p ℓ g) = neuralCode (sublevelFamily q ℓ g)`.
    ext σ
    constructor
    · -- Any `σ ∈ code(U_p)` must be in `code(U_q)`.
      rintro ⟨x, rfl⟩
      set τ := codeword (sublevelFamily p ℓ g) x
      have hxp : (x, p x) ∈ sigRegion ℓ g τ := (mem_sigRegion_graph_iff p ℓ g x τ).mpr rfl
      by_contra hmiss
      -- Since `τ` is missed by `Γ_q`, `z.2 - q z.1` never vanishes on the convex set `S_τ`,
      -- so by IVT it has constant strict sign on `S_τ`.
      have hnozero : ∀ z ∈ sigRegion ℓ g τ, z.2 ≠ q z.1 := by
        intro z hz heq
        apply hmiss
        have hz' : (z.1, q z.1) ∈ sigRegion ℓ g τ := by rwa [← heq]
        exact ⟨z.1, (mem_sigRegion_graph_iff q ℓ g z.1 τ).mp hz'⟩
      have hstrict_lt : p x < q x := lt_of_le_of_ne (hp_le_q x) (hnozero (x, p x) hxp)
      -- Therefore EVERY `z ∈ S_τ` satisfies `z.2 < q z.1` (by IVT along the segment!).
      have hall_below : ∀ z ∈ sigRegion ℓ g τ, z.2 < q z.1 := by
        intro z hz
        by_contra hge
        push_neg at hge
        have hgt : q z.1 < z.2 := lt_of_le_of_ne hge (Ne.symm (hnozero z hz))
        let γ : ℝ → ℝ := fun r =>
          ((1 - r) * p x + r * z.2) - q ((1 - r) • x + r • z.1)
        have hγ_cont : ContinuousOn γ (Set.Icc 0 1) := by
          have hlin : Continuous (fun r : ℝ => (1 - r) • x + r • z.1) :=
            ((continuous_const.sub continuous_id).smul continuous_const).add
              (continuous_id.smul continuous_const)
          exact (((continuous_const.sub continuous_id).mul continuous_const).add
            (continuous_id.mul continuous_const)).sub (hq_cont.comp hlin) |>.continuousOn
        have h0_mem : (0 : ℝ) ∈ Set.Icc (γ 0) (γ 1) := by
          constructor
          · dsimp [γ]; simp; linarith
          · dsimp [γ]; simp; linarith
        obtain ⟨r, hr, hr0⟩ := intermediate_value_Icc (by norm_num : (0 : ℝ) ≤ 1) hγ_cont h0_mem
        have hseg : ((1 - r) • x + r • z.1, (1 - r) * p x + r * z.2) ∈ sigRegion ℓ g τ := by
          have := convex_sigRegion ℓ g τ hxp hz (a := 1 - r) (b := r)
            (by linarith [hr.2]) hr.1 (by ring)
          exact this
        have heq := hnozero _ hseg
        dsimp [γ] at hr0
        exact heq (by linarith)
      -- Thus `τ ∈ B₂ ⊆ activeSigs`, so `pick τ ∈ S` gives `p x < evalAffine (pick τ) x ≤ p x`!
      have hτ_B2 : τ ∈ B₂ := Finset.mem_filter.mpr ⟨Finset.mem_univ _, ⟨(x, p x), hxp⟩, hall_below⟩
      have hτ_notA : ¬ ∃ y : Fin d → ℝ, (y, q y) ∈ sigRegion ℓ g τ := by
        rintro ⟨y, hy⟩
        exact hmiss ⟨y, (mem_sigRegion_graph_iff q ℓ g y τ).mp hy⟩
      have hpick_mem : pick τ ∈ S := Finset.mem_image_of_mem pick (Finset.mem_union_right A hτ_B2)
      have hpick_gt : p x < evalAffine (pick τ) x := by
        dsimp [pick]
        rw [dif_neg hτ_notA, dif_pos ⟨⟨(x, p x), hxp⟩, hall_below⟩]
        exact (hsep τ ⟨(x, p x), hxp⟩ hall_below).choose_spec.2 (x, p x) hxp
      have hpick_le : evalAffine (pick τ) x ≤ p x :=
        Finset.le_sup' (f := fun s => evalAffine s x) hpick_mem
      linarith
    · -- Any `σ ∈ code(U_q)` is attained by `p` at the witness `x_σ = Classical.choose hA`.
      rintro ⟨x, rfl⟩
      set τ := codeword (sublevelFamily q ℓ g) x
      have hA_ex : ∃ y : Fin d → ℝ, (y, q y) ∈ sigRegion ℓ g τ :=
        ⟨x, (mem_sigRegion_graph_iff q ℓ g x τ).mpr rfl⟩
      have hτ_A : τ ∈ A := Finset.mem_filter.mpr ⟨Finset.mem_univ _, hA_ex⟩
      set x₀ := Classical.choose hA_ex
      have hx₀_sig : (x₀, q x₀) ∈ sigRegion ℓ g τ := Classical.choose_spec hA_ex
      have hpick_mem : pick τ ∈ S := Finset.mem_image_of_mem pick (Finset.mem_union_left B₂ hτ_A)
      have hpick_eq : evalAffine (pick τ) x₀ = q x₀ := by
        dsimp [pick]; rw [dif_pos hA_ex]
        exact (hsupport x₀).choose_spec.2
      have hpx₀ : p x₀ = q x₀ :=
        le_antisymm (hp_le_q x₀) (hpick_eq ▸ Finset.le_sup' (f := fun s => evalAffine s x₀) hpick_mem)
      refine ⟨x₀, (mem_sigRegion_graph_iff p ℓ g x₀ τ).mp ?_⟩
      rwa [hpx₀]
  · -- Each replacement field `{p < ℓ_i}` is the exact intersection `⋂_{s ∈ S} {s < ℓ_i}`.
    intro i
    ext x
    simp only [Set.mem_setOf_eq, Set.mem_iInter, Finset.sup'_lt_iff]

/-- **Corollary 3 (Open Homothetic Ellipsoids in Every Dimension `d`)**:
For any symmetric positive semidefinite matrix `Q ∈ ℝ^{d × d}` (`Qᵀ = Q` and `vᵀ Q v ≥ 0`),
centers `c₁, ..., c_N ∈ ℝ^d`, thresholds `ρ₁, ..., ρ_N ∈ ℝ`, and auxiliary strict affine
halfspaces `g₁, ..., g_M`, the open homothetic ellipsoid family
`U_i = {x : (x - c_i)ᵀ Q (x - c_i) < ρ_i}` admits an EXACT same-dimensional open polyhedral
realization in `ℝ^d` (with at most `2^(N + M)` strict affine halfspaces per field)! -/
theorem homothetic_ellipsoids_polyhedralizable {d N M : ℕ}
    (Q : Matrix (Fin d) (Fin d) ℝ) (hQsym : Qᵀ = Q)
    (hQpsd : ∀ v : Fin d → ℝ, 0 ≤ dotProduct v (Q *ᵥ v))
    (c : Fin N → Fin d → ℝ) (ρ : Fin N → ℝ) (g : Fin M → AffineFun d) :
    let U_ell : Fin (N + M) → Set (Fin d → ℝ) := fun idx =>
      if h : (idx : ℕ) < N then
        {x : Fin d → ℝ | dotProduct (x - c ⟨idx.1, h⟩) (Q *ᵥ (x - c ⟨idx.1, h⟩)) < ρ ⟨idx.1, h⟩}
      else
        {x : Fin d → ℝ | 0 < evalAffine (g ⟨idx.1 - N, by omega⟩) x}
    ∃ (S : Finset (AffineFun d)) (hne : S.Nonempty) (ℓ : Fin N → AffineFun d),
      S.card ≤ 2 ^ (N + M) ∧
      let p : (Fin d → ℝ) → ℝ := fun x => S.sup' hne (fun s => evalAffine s x)
      neuralCode (sublevelFamily p ℓ g) = neuralCode U_ell := by
  intro U_ell
  let q : (Fin d → ℝ) → ℝ := fun x => dotProduct x (Q *ᵥ x)
  -- Define `ℓ_i(x) = 2 (Q c_i) ⬝ᵥ x + (ρ_i - c_iᵀ Q c_i)`.
  let ℓ : Fin N → AffineFun d := fun i =>
    ((2 : ℝ) • (Q *ᵥ c i), ρ i - dotProduct (c i) (Q *ᵥ c i))
  have hQ_dot_comm : ∀ x y : Fin d → ℝ, dotProduct x (Q *ᵥ y) = dotProduct y (Q *ᵥ x) := by
    intro x y
    rw [Matrix.dotProduct_mulVec, ← Matrix.mulVec_transpose, hQsym, dotProduct_comm]
  -- Quadratic form `q(x) = xᵀ Q x` is continuous and convex on `ℝ^d`.
  have hq_cont : Continuous q := by
    unfold q dotProduct Matrix.mulVec
    exact continuous_finset_sum _ fun i _ =>
      (continuous_apply i).mul (continuous_finset_sum _ fun j _ =>
        continuous_const.mul (continuous_apply j))
  have hq_conv : ConvexOn ℝ Set.univ q := by
    refine ⟨convex_univ, fun x _ y _ a b ha hb hab => ?_⟩
    dsimp [q]
    have hexp : dotProduct (a • x + b • y) (Q *ᵥ (a • x + b • y)) =
        a * dotProduct x (Q *ᵥ x) + b * dotProduct y (Q *ᵥ y) -
        a * b * dotProduct (x - y) (Q *ᵥ (x - y)) := by
      rw [Matrix.mulVec_add, Matrix.mulVec_smul, Matrix.mulVec_smul,
        add_dotProduct, dotProduct_add, dotProduct_add,
        smul_dotProduct, smul_dotProduct, smul_dotProduct, smul_dotProduct,
        dotProduct_smul, dotProduct_smul, dotProduct_smul, dotProduct_smul,
        Matrix.mulVec_sub, sub_dotProduct, dotProduct_sub, dotProduct_sub]
      simp only [smul_eq_mul, hQ_dot_comm y x]
      have hb_eq : b = 1 - a := by linarith
      subst hb_eq
      ring
    have hpsd := hQpsd (x - y)
    have hnn : 0 ≤ a * b * dotProduct (x - y) (Q *ᵥ (x - y)) := mul_nonneg (mul_nonneg ha hb) hpsd
    change dotProduct (a • x + b • y) (Q *ᵥ (a • x + b • y)) ≤
      a * dotProduct x (Q *ᵥ x) + b * dotProduct y (Q *ᵥ y)
    linarith
  -- Show `sublevelFamily q ℓ g = U_ell` identically!
  have hfam_eq : sublevelFamily q ℓ g = U_ell := by
    funext idx
    simp only [sublevelFamily, U_ell]
    split_ifs with h
    · ext x
      simp only [Set.mem_setOf_eq, q, ℓ, evalAffine]
      have hquad : dotProduct (x - c ⟨idx.1, h⟩) (Q *ᵥ (x - c ⟨idx.1, h⟩)) =
          dotProduct x (Q *ᵥ x) - 2 * dotProduct (Q *ᵥ c ⟨idx.1, h⟩) x +
            dotProduct (c ⟨idx.1, h⟩) (Q *ᵥ c ⟨idx.1, h⟩) := by
        rw [Matrix.mulVec_sub, sub_dotProduct, dotProduct_sub, dotProduct_sub,
          hQ_dot_comm (c ⟨idx.1, h⟩) x, dotProduct_comm (Q *ᵥ c ⟨idx.1, h⟩) x]
        ring
      have hsmul : dotProduct ((2 : ℝ) • (Q *ᵥ c ⟨idx.1, h⟩)) x =
          2 * dotProduct (Q *ᵥ c ⟨idx.1, h⟩) x := by
        rw [smul_dotProduct, smul_eq_mul]
      rw [hquad, hsmul]
      constructor <;> intro hlt <;> linarith
    · rfl
  obtain ⟨S, hne, hcard, _, hcode, _⟩ :=
    common_convex_function_polyhedralization q hq_conv hq_cont ℓ g
  rw [hfam_eq] at hcode
  exact ⟨S, hne, ℓ, hcard, hcode⟩

/-- **Corollary 3 (Open Euclidean Balls in Every Dimension `d`)**:
Specializing `Q = I_d` realizes every finite family of open Euclidean balls
`{x : ∑_k (x_k - c_{i,k})² < ρ_i}` (plus auxiliary strict affine halfspaces) by open
polyhedra in the SAME dimension `d`. -/
theorem open_balls_polyhedralizable {d N M : ℕ}
    (c : Fin N → Fin d → ℝ) (ρ : Fin N → ℝ) (g : Fin M → AffineFun d) :
    let U_ball : Fin (N + M) → Set (Fin d → ℝ) := fun idx =>
      if h : (idx : ℕ) < N then
        {x : Fin d → ℝ | dotProduct (x - c ⟨idx.1, h⟩) (x - c ⟨idx.1, h⟩) < ρ ⟨idx.1, h⟩}
      else
        {x : Fin d → ℝ | 0 < evalAffine (g ⟨idx.1 - N, by omega⟩) x}
    ∃ (S : Finset (AffineFun d)) (hne : S.Nonempty) (ℓ : Fin N → AffineFun d),
      S.card ≤ 2 ^ (N + M) ∧
      let p : (Fin d → ℝ) → ℝ := fun x => S.sup' hne (fun s => evalAffine s x)
      neuralCode (sublevelFamily p ℓ g) = neuralCode U_ball := by
  have h := homothetic_ellipsoids_polyhedralizable (1 : Matrix (Fin d) (Fin d) ℝ)
    Matrix.transpose_one
    (fun v => by rw [Matrix.one_mulVec]; exact Finset.sum_nonneg fun k _ => mul_self_nonneg (v k))
    c ρ g
  simpa only [Matrix.one_mulVec] using h

/-! ### 2. Intermediate-Word Inner-Exhaustion Core (Theorem 4) & CGIK Witness -/

/-- Inner word `I(x) = {i | x ∈ U_i}` and outer closure word `J(x) = {i | x ∈ closure (U_i)}`. -/
noncomputable def innerWord {n d : ℕ} (U : Fin n → Set (Fin d → ℝ)) (x : Fin d → ℝ) :
    Finset (Fin n) := codeword U x

noncomputable def outerWord {n d : ℕ} (U : Fin n → Set (Fin d → ℝ)) (x : Fin d → ℝ) :
    Finset (Fin n) := by
  classical
  exact Finset.univ.filter fun i => x ∈ closure (U i)

/-- **Theorem 4 (Intermediate-Word Inner-Exhaustion Core)**:
Suppose `U : Fin n → Set (Fin d → ℝ)` has `∅ ∈ code(U)` and satisfies the intermediate-word
condition `∀ x σ, I(x) ⊆ σ ⊆ J(x) → σ ∈ code(U)`. Let `B` be a bounded set with compact
closure containing one witness per word of `code(U)`, and let `Q : ℕ → Fin n → Set (Fin d → ℝ)`
be inner approximations (`Q m i ⊆ U i ∩ B`) that eventually cover an open neighborhood of
every point of `U i` inside `B`. Then for all sufficiently large `m`, `code(Q m) = code(U)`. -/
theorem intermediate_word_polyhedralization {n d : ℕ}
    (U : Fin n → Set (Fin d → ℝ)) (B : Set (Fin d → ℝ))
    (hempty : (∅ : Finset (Fin n)) ∈ neuralCode U)
    (hB_comp : IsCompact (closure B))
    (hB_wit : ∀ τ ∈ neuralCode U, ∃ w ∈ B, codeword U w = τ)
    (hinter : ∀ (x : Fin d → ℝ) (σ : Finset (Fin n)),
      innerWord U x ⊆ σ → σ ⊆ outerWord U x → σ ∈ neuralCode U)
    (Q : ℕ → Fin n → Set (Fin d → ℝ))
    (hQ_sub : ∀ m i, Q m i ⊆ U i ∩ B)
    (hQ_exhaust : ∀ (i : Fin n) (x : Fin d → ℝ), x ∈ U i →
      ∃ V : Set (Fin d → ℝ), IsOpen V ∧ x ∈ V ∧ ∃ m₀ : ℕ, ∀ m ≥ m₀, V ∩ B ⊆ Q m i) :
    ∃ m₀ : ℕ, ∀ m ≥ m₀, neuralCode (Q m) = neuralCode U := by
  classical
  -- Step 1: Eventually `code(U) ⊆ code(Q m)` (finitely many codewords, each witnessed in `B`).
  have hsub_ev : ∃ m₁ : ℕ, ∀ m ≥ m₁, neuralCode U ⊆ neuralCode (Q m) := by
    have hwit_ev : ∀ τ : Finset (Fin n),
        ∃ m_τ : ℕ, τ ∈ neuralCode U → ∀ m ≥ m_τ, τ ∈ neuralCode (Q m) := by
      intro τ
      by_cases hτ : τ ∈ neuralCode U
      · obtain ⟨w, hwB, hw_eq⟩ := hB_wit τ hτ
        have hex : ∀ i : Fin n, ∃ m_i : ℕ, i ∈ τ → ∀ m ≥ m_i, w ∈ Q m i := by
          intro i
          by_cases hi : i ∈ τ
          · have hwi : w ∈ U i := by rw [← mem_codeword_iff, hw_eq]; exact hi
            obtain ⟨V, _, hwV, m_i, hm_i⟩ := hQ_exhaust i w hwi
            exact ⟨m_i, fun _ m hm => hm_i m hm ⟨hwV, hwB⟩⟩
          · exact ⟨0, fun hc => absurd hc hi⟩
        choose m_i hm_i using hex
        refine ⟨Finset.univ.sup m_i, fun _ m hm => ⟨w, ?_⟩⟩
        ext i
        rw [mem_codeword_iff]
        constructor
        · intro hwQ
          have hwU : w ∈ U i := (hQ_sub m i hwQ).1
          rw [← hw_eq, mem_codeword_iff]; exact hwU
        · intro hi
          exact hm_i i hi m (le_trans (Finset.le_sup (f := m_i) (Finset.mem_univ i)) hm)
      · exact ⟨0, fun hc => absurd hc hτ⟩
    choose m_τ hm_τ using hwit_ev
    refine ⟨Finset.univ.sup m_τ, fun m hm τ hτ => ?_⟩
    exact hm_τ τ hτ m (le_trans (Finset.le_sup (f := m_τ) (Finset.mem_univ τ)) hm)
  -- Step 2: Eventually `code(Q m) ⊆ code(U)` by compactness of `closure B` and intermediate-word!
  have hsup_ev : ∃ m₂ : ℕ, ∀ m ≥ m₂, neuralCode (Q m) ⊆ neuralCode U := by
    by_contra hnot
    push_neg at hnot
    -- For every `k : ℕ`, choose `m k ≥ k` and `x k` with `codeword (Q (m k)) (x k) ∉ code(U)`.
    have hbad : ∀ k : ℕ, ∃ m ≥ k, ∃ x : Fin d → ℝ, codeword (Q m) x ∉ neuralCode U := by
      intro k
      obtain ⟨m, hmk, hnot_sub⟩ := hnot k
      rw [Set.not_subset] at hnot_sub
      obtain ⟨σ, ⟨x, rfl⟩, hσ_not⟩ := hnot_sub
      exact ⟨m, hmk, x, hσ_not⟩
    choose m hm_ge x hx_bad using hbad
    -- Each `x k` must lie in `B` (otherwise `codeword (Q (m k)) (x k) = ∅ ∈ code(U)`).
    have hxB : ∀ k : ℕ, x k ∈ B := by
      intro k
      by_contra hnotB
      have hemp : codeword (Q (m k)) (x k) = ∅ := by
        ext i
        simp only [mem_codeword_iff, Finset.notMem_empty, iff_false]
        intro hmem
        exact hnotB (hQ_sub (m k) i hmem).2
      exact hx_bad k (hemp ▸ hempty)
    -- Pass to a subsequence along which `x (φ k) → x_* ∈ closure B`.
    have hx_cl : ∀ k : ℕ, x k ∈ closure B := fun k => subset_closure (hxB k)
    obtain ⟨x_star, _, φ, hφ_mono, hφ_tend⟩ := hB_comp.tendsto_subseq hx_cl
    -- By pigeonhole over the finite set `Finset (Fin n)`, some word `σ` occurs infinitely often!
    let wordAt : ℕ → Finset (Fin n) := fun k => codeword (Q (m (φ k))) (x (φ k))
    have hinf : ∃ σ : Finset (Fin n), ∀ N : ℕ, ∃ k ≥ N, wordAt k = σ := by
      by_contra hfin
      push_neg at hfin
      choose N hN using hfin
      let Nmax := (Finset.univ : Finset (Finset (Fin n))).sup N
      have hge : N (wordAt Nmax) ≤ Nmax := Finset.le_sup (f := N) (Finset.mem_univ _)
      exact hN (wordAt Nmax) Nmax hge rfl
    obtain ⟨σ, hσ_inf⟩ := hinf
    have hσ_not : σ ∉ neuralCode U := by
      obtain ⟨k, _, hk⟩ := hσ_inf 0
      rw [← hk]; exact hx_bad (φ k)
    -- Show `innerWord U x_star ⊆ σ ⊆ outerWord U x_star`, contradicting `hinter`!
    have hσ_le_outer : σ ⊆ outerWord U x_star := by
      intro i hi
      simp only [outerWord, Finset.mem_filter, Finset.mem_univ, true_and]
      rw [mem_closure_iff_nhds]
      intro W hW
      obtain ⟨K, hK⟩ := Filter.eventually_atTop.mp (hφ_tend.eventually hW)
      obtain ⟨k, hkK, hkσ⟩ := hσ_inf K
      refine ⟨x (φ k), hK k hkK, ?_⟩
      have hmemQ : x (φ k) ∈ Q (m (φ k)) i := by
        rw [← mem_codeword_iff]; dsimp [wordAt] at hkσ; rw [hkσ]; exact hi
      exact (hQ_sub (m (φ k)) i hmemQ).1
    have hinner_le_σ : innerWord U x_star ⊆ σ := by
      intro i hi
      have hxi : x_star ∈ U i := (mem_codeword_iff U x_star i).mp hi
      obtain ⟨V, hV_open, hxV, m_i, hm_i⟩ := hQ_exhaust i x_star hxi
      obtain ⟨K₁, hK₁⟩ := Filter.eventually_atTop.mp (hφ_tend.eventually (hV_open.mem_nhds hxV))
      obtain ⟨k, hk, hkσ⟩ := hσ_inf (max K₁ m_i)
      have hkK₁ : K₁ ≤ k := le_trans (le_max_left _ _) hk
      have hkm : m_i ≤ m (φ k) := by
        have h1 : m_i ≤ k := le_trans (le_max_right _ _) hk
        have h2 : k ≤ φ k := hφ_mono.id_le k
        exact le_trans (le_trans h1 h2) (hm_ge (φ k))
      have hmemQ : x (φ k) ∈ Q (m (φ k)) i := hm_i (m (φ k)) hkm ⟨hK₁ k hkK₁, hxB (φ k)⟩
      rw [← hkσ]
      exact (mem_codeword_iff (Q (m (φ k))) (x (φ k)) i).mpr hmemQ
    exact hσ_not (hinter x_star σ hinner_le_σ hσ_le_outer)
  obtain ⟨m₁, hm₁⟩ := hsub_ev
  obtain ⟨m₂, hm₂⟩ := hsup_ev
  exact ⟨max m₁ m₂, fun m hm =>
    Set.Subset.antisymm (hm₂ m (le_trans (le_max_right _ _) hm))
      (hm₁ m (le_trans (le_max_left _ _) hm))⟩

/-- **CGIK Non-Degeneracy vs. Intermediate-Word Strictness Witness** (Remark of Section 3):
For the three open halfplanes `U₀ = {y > 0}`, `U₁ = {x > 0}`, `U₂ = {x + y > 0}` in `ℝ²`:
1. The singleton word `{2}` is NOT in `code(U)`.
2. At `0 = (0, 0)`, `I(0) = ∅ ⊆ {2} ⊆ {0, 1, 2} = J(0)`, so the intermediate-word condition fails.
3. For every `ε > 0`, uniform inner shrinking `{y > ε}, {x > ε}, {x + y > ε}` creates the
   extraneous word `{2}` at `(2 ε / 3, 2 ε / 3)`. -/
theorem cgik_three_halfplanes_witness :
    let U : Fin 3 → Set (Fin 2 → ℝ) := fun i => match i with
      | 0 => {z | 0 < z 1}
      | 1 => {z | 0 < z 0}
      | 2 => {z | 0 < z 0 + z 1}
    ({(2 : Fin 3)} : Finset (Fin 3)) ∉ neuralCode U ∧
    innerWord U 0 = ∅ ∧
    ({(2 : Fin 3)} : Finset (Fin 3)) ⊆ outerWord U 0 ∧
    ∀ ε : ℝ, 0 < ε →
      let Uε : Fin 3 → Set (Fin 2 → ℝ) := fun i => match i with
        | 0 => {z | ε < z 1}
        | 1 => {z | ε < z 0}
        | 2 => {z | ε < z 0 + z 1}
      codeword Uε ![2 * ε / 3, 2 * ε / 3] = {(2 : Fin 3)} := by
  intro U
  refine ⟨?_, ?_, ?_, ?_⟩
  · rintro ⟨z, hz⟩
    have h2 : z 0 + z 1 > 0 := by
      have := (mem_codeword_iff U z 2).mp (by rw [hz]; simp)
      simpa [U] using this
    have h0 : ¬ (0 < z 1) := by
      intro hc
      have : (0 : Fin 3) ∈ codeword U z := (mem_codeword_iff U z 0).mpr hc
      rw [hz] at this; simp at this
    have h1 : ¬ (0 < z 0) := by
      intro hc
      have : (1 : Fin 3) ∈ codeword U z := (mem_codeword_iff U z 1).mpr hc
      rw [hz] at this; simp at this
    linarith
  · ext i
    fin_cases i <;> simp [innerWord, mem_codeword_iff, U]
  · intro i hi
    simp only [Finset.mem_singleton] at hi
    subst hi
    simp only [outerWord, Finset.mem_filter, Finset.mem_univ, true_and]
    rw [Metric.mem_closure_iff]
    intro ε hε
    refine ⟨fun _ => ε / 2, ?_, ?_⟩
    · simp only [U, Set.mem_setOf_eq]; linarith
    · rw [dist_pi_lt_iff hε]
      intro k
      simp only [Pi.zero_apply, Real.dist_eq, zero_sub, abs_neg, abs_of_pos (by linarith : 0 < ε / 2)]
      linarith
  · intro ε hε Uε
    ext i
    rw [mem_codeword_iff]
    fin_cases i
    · simp only [Uε, Set.mem_setOf_eq, Finset.mem_singleton, Fin.isValue, Matrix.cons_val_one,
        Matrix.cons_val_fin_one]
      exact ⟨fun h => absurd (by linarith : False) id, fun h => absurd h (by decide)⟩
    · simp only [Uε, Set.mem_setOf_eq, Finset.mem_singleton, Fin.isValue, Matrix.cons_val_zero]
      exact ⟨fun h => absurd (by linarith : False) id, fun h => absurd h (by decide)⟩
    · simp only [Uε, Set.mem_setOf_eq, Finset.mem_singleton, Fin.isValue, Matrix.cons_val_zero,
        Matrix.cons_val_one, Matrix.cons_val_fin_one]
      exact ⟨fun _ => rfl, fun _ => by linarith⟩

/-! ### 3. Two-Sided Refutation of the Closed-Convex Variant (Theorem 8) -/

/-- The 5-neuron, 12-word Lienkaemper–Shiu–Woodstock closed triangular fan code `C₅`
(0-indexed: neurons `0,1,2,3,4` correspond to `1,2,3,4,5` in `solution.md`). -/
def codeC5 : Finset (Finset (Fin 5)) :=
  {∅, {1, 2, 3, 4}, {0, 1, 2}, {0, 2, 3}, {0, 3, 4}, {0, 2}, {0, 3},
    {1, 2}, {2, 3}, {3, 4}, {2}, {3}}

/-- Explicit closed convex realization of `C₅` in `ℝ²` (Theorem 8(1)):
`U₀ = [A, C]`, `U₁ = [O, A]`, `U₂ = conv(O, A, B)`, `U₃ = conv(O, B, C)`, `U₄ = [O, C]`
with `O = (0,0), A = (-1,1), B = (0,1), C = (1,1)`. -/
def C5Sets : Fin 5 → Set (Fin 2 → ℝ)
  | 0 => {z | z 1 = 1 ∧ -1 ≤ z 0 ∧ z 0 ≤ 1}
  | 1 => {z | z 0 = -z 1 ∧ 0 ≤ z 1 ∧ z 1 ≤ 1}
  | 2 => {z | 0 ≤ z 1 ∧ z 1 ≤ 1 ∧ -z 1 ≤ z 0 ∧ z 0 ≤ 0}
  | 3 => {z | 0 ≤ z 1 ∧ z 1 ≤ 1 ∧ 0 ≤ z 0 ∧ z 0 ≤ z 1}
  | 4 => {z | z 0 = z 1 ∧ 0 ≤ z 1 ∧ z 1 ≤ 1}

/-- Helper for `C5_closed_convex_realization`: every point of `ℝ²` produces a word in `codeC5`. -/
private lemma C5Sets_codeword_mem (z : Fin 2 → ℝ) : codeword C5Sets z ∈ codeC5 := by
  set x := z 0
  set y := z 1
  have hcw : ∀ S : Finset (Fin 5),
      ((0 : Fin 5) ∈ S ↔ y = 1 ∧ -1 ≤ x ∧ x ≤ 1) →
      ((1 : Fin 5) ∈ S ↔ x = -y ∧ 0 ≤ y ∧ y ≤ 1) →
      ((2 : Fin 5) ∈ S ↔ 0 ≤ y ∧ y ≤ 1 ∧ -y ≤ x ∧ x ≤ 0) →
      ((3 : Fin 5) ∈ S ↔ 0 ≤ y ∧ y ≤ 1 ∧ 0 ≤ x ∧ x ≤ y) →
      ((4 : Fin 5) ∈ S ↔ x = y ∧ 0 ≤ y ∧ y ≤ 1) →
      codeword C5Sets z = S := by
    intro S h0 h1 h2 h3 h4
    ext i
    rw [mem_codeword_iff]
    fin_cases i
    · exact h0.symm
    · exact h1.symm
    · exact h2.symm
    · exact h3.symm
    · exact h4.symm
  have hY : ∀ {i : Fin 5} {S : Finset (Fin 5)} {P : Prop}, i ∈ S → P → (i ∈ S ↔ P) :=
    fun hi hP => ⟨fun _ => hP, fun _ => hi⟩
  have hN : ∀ {i : Fin 5} {S : Finset (Fin 5)} {P : Prop}, i ∉ S → ¬ P → (i ∈ S ↔ P) :=
    fun hi hP => ⟨fun h => absurd h hi, fun h => absurd h hP⟩
  by_cases hy_out : y < 0 ∨ 1 < y
  · have heq : codeword C5Sets z = ∅ := hcw ∅
      (hN (by decide) (by rintro ⟨h1, _, _⟩; rcases hy_out with h | h <;> linarith))
      (hN (by decide) (by rintro ⟨_, h1, h2⟩; rcases hy_out with h | h <;> linarith))
      (hN (by decide) (by rintro ⟨h1, h2, _, _⟩; rcases hy_out with h | h <;> linarith))
      (hN (by decide) (by rintro ⟨h1, h2, _, _⟩; rcases hy_out with h | h <;> linarith))
      (hN (by decide) (by rintro ⟨_, h1, h2⟩; rcases hy_out with h | h <;> linarith))
    rw [heq]; decide
  · push_neg at hy_out
    rcases eq_or_lt_of_le hy_out.1 with hy0 | hy_pos
    · by_cases hx0 : x = 0
      · have heq : codeword C5Sets z = {1, 2, 3, 4} := hcw {1, 2, 3, 4}
          (hN (by decide) (by rintro ⟨h1, _, _⟩; linarith))
          (hY (by decide) ⟨by linarith, by linarith, by linarith⟩)
          (hY (by decide) ⟨by linarith, by linarith, by linarith, by linarith⟩)
          (hY (by decide) ⟨by linarith, by linarith, by linarith, by linarith⟩)
          (hY (by decide) ⟨by linarith, by linarith, by linarith⟩)
        rw [heq]; decide
      · have heq : codeword C5Sets z = ∅ := hcw ∅
          (hN (by decide) (by rintro ⟨h1, _, _⟩; linarith))
          (hN (by decide) (by rintro ⟨h1, _, _⟩; exact hx0 (by linarith)))
          (hN (by decide) (by rintro ⟨_, _, h1, h2⟩; exact hx0 (by linarith)))
          (hN (by decide) (by rintro ⟨_, _, h1, h2⟩; exact hx0 (by linarith)))
          (hN (by decide) (by rintro ⟨h1, _, _⟩; exact hx0 (by linarith)))
        rw [heq]; decide
    · rcases eq_or_lt_of_le hy_out.2 with hy1 | hy_lt1
      · by_cases hx_out : x < -1 ∨ 1 < x
        · have heq : codeword C5Sets z = ∅ := hcw ∅
            (hN (by decide) (by rintro ⟨_, h1, h2⟩; rcases hx_out with h | h <;> linarith))
            (hN (by decide) (by rintro ⟨h1, _, _⟩; rcases hx_out with h | h <;> linarith))
            (hN (by decide) (by rintro ⟨_, _, h1, h2⟩; rcases hx_out with h | h <;> linarith))
            (hN (by decide) (by rintro ⟨_, _, h1, h2⟩; rcases hx_out with h | h <;> linarith))
            (hN (by decide) (by rintro ⟨h1, _, _⟩; rcases hx_out with h | h <;> linarith))
          rw [heq]; decide
        · push_neg at hx_out
          rcases eq_or_lt_of_le hx_out.1 with hxm1 | hxm1_lt
          · have heq : codeword C5Sets z = {0, 1, 2} := hcw {0, 1, 2}
              (hY (by decide) ⟨by linarith, by linarith, by linarith⟩)
              (hY (by decide) ⟨by linarith, by linarith, by linarith⟩)
              (hY (by decide) ⟨by linarith, by linarith, by linarith, by linarith⟩)
              (hN (by decide) (by rintro ⟨_, _, h1, _⟩; linarith))
              (hN (by decide) (by rintro ⟨h1, _, _⟩; linarith))
            rw [heq]; decide
          · rcases lt_trichotomy x 0 with hx_neg | hx0 | hx_pos
            · have heq : codeword C5Sets z = {0, 2} := hcw {0, 2}
                (hY (by decide) ⟨by linarith, by linarith, by linarith⟩)
                (hN (by decide) (by rintro ⟨h1, _, _⟩; linarith))
                (hY (by decide) ⟨by linarith, by linarith, by linarith, by linarith⟩)
                (hN (by decide) (by rintro ⟨_, _, h1, _⟩; linarith))
                (hN (by decide) (by rintro ⟨h1, _, _⟩; linarith))
              rw [heq]; decide
            · have heq : codeword C5Sets z = {0, 2, 3} := hcw {0, 2, 3}
                (hY (by decide) ⟨by linarith, by linarith, by linarith⟩)
                (hN (by decide) (by rintro ⟨h1, _, _⟩; linarith))
                (hY (by decide) ⟨by linarith, by linarith, by linarith, by linarith⟩)
                (hY (by decide) ⟨by linarith, by linarith, by linarith, by linarith⟩)
                (hN (by decide) (by rintro ⟨h1, _, _⟩; linarith))
              rw [heq]; decide
            · rcases eq_or_lt_of_le hx_out.2 with hx1 | hx1_lt
              · have heq : codeword C5Sets z = {0, 3, 4} := hcw {0, 3, 4}
                  (hY (by decide) ⟨by linarith, by linarith, by linarith⟩)
                  (hN (by decide) (by rintro ⟨h1, _, _⟩; linarith))
                  (hN (by decide) (by rintro ⟨_, _, _, h2⟩; linarith))
                  (hY (by decide) ⟨by linarith, by linarith, by linarith, by linarith⟩)
                  (hY (by decide) ⟨by linarith, by linarith, by linarith⟩)
                rw [heq]; decide
              · have heq : codeword C5Sets z = {0, 3} := hcw {0, 3}
                  (hY (by decide) ⟨by linarith, by linarith, by linarith⟩)
                  (hN (by decide) (by rintro ⟨h1, _, _⟩; linarith))
                  (hN (by decide) (by rintro ⟨_, _, _, h2⟩; linarith))
                  (hY (by decide) ⟨by linarith, by linarith, by linarith, by linarith⟩)
                  (hN (by decide) (by rintro ⟨h1, _, _⟩; linarith))
                rw [heq]; decide
      · by_cases hx_out : x < -y ∨ y < x
        · have heq : codeword C5Sets z = ∅ := hcw ∅
            (hN (by decide) (by rintro ⟨h1, _, _⟩; linarith))
            (hN (by decide) (by rintro ⟨h1, _, _⟩; rcases hx_out with h | h <;> linarith))
            (hN (by decide) (by rintro ⟨_, _, h1, h2⟩; rcases hx_out with h | h <;> linarith))
            (hN (by decide) (by rintro ⟨_, _, h1, h2⟩; rcases hx_out with h | h <;> linarith))
            (hN (by decide) (by rintro ⟨h1, _, _⟩; rcases hx_out with h | h <;> linarith))
          rw [heq]; decide
        · push_neg at hx_out
          rcases eq_or_lt_of_le hx_out.1 with hxny | hxny_lt
          · have heq : codeword C5Sets z = {1, 2} := hcw {1, 2}
              (hN (by decide) (by rintro ⟨h1, _, _⟩; linarith))
              (hY (by decide) ⟨by linarith, by linarith, by linarith⟩)
              (hY (by decide) ⟨by linarith, by linarith, by linarith, by linarith⟩)
              (hN (by decide) (by rintro ⟨_, _, h1, _⟩; linarith))
              (hN (by decide) (by rintro ⟨h1, _, _⟩; linarith))
            rw [heq]; decide
          · rcases lt_trichotomy x 0 with hx_neg | hx0 | hx_pos
            · have heq : codeword C5Sets z = {2} := hcw {2}
                (hN (by decide) (by rintro ⟨h1, _, _⟩; linarith))
                (hN (by decide) (by rintro ⟨h1, _, _⟩; linarith))
                (hY (by decide) ⟨by linarith, by linarith, by linarith, by linarith⟩)
                (hN (by decide) (by rintro ⟨_, _, h1, _⟩; linarith))
                (hN (by decide) (by rintro ⟨h1, _, _⟩; linarith))
              rw [heq]; decide
            · have heq : codeword C5Sets z = {2, 3} := hcw {2, 3}
                (hN (by decide) (by rintro ⟨h1, _, _⟩; linarith))
                (hN (by decide) (by rintro ⟨h1, _, _⟩; linarith))
                (hY (by decide) ⟨by linarith, by linarith, by linarith, by linarith⟩)
                (hY (by decide) ⟨by linarith, by linarith, by linarith, by linarith⟩)
                (hN (by decide) (by rintro ⟨h1, _, _⟩; linarith))
              rw [heq]; decide
            · rcases eq_or_lt_of_le hx_out.2 with hxy | hxy_lt
              · have heq : codeword C5Sets z = {3, 4} := hcw {3, 4}
                  (hN (by decide) (by rintro ⟨h1, _, _⟩; linarith))
                  (hN (by decide) (by rintro ⟨h1, _, _⟩; linarith))
                  (hN (by decide) (by rintro ⟨_, _, _, h2⟩; linarith))
                  (hY (by decide) ⟨by linarith, by linarith, by linarith, by linarith⟩)
                  (hY (by decide) ⟨by linarith, by linarith, by linarith⟩)
                rw [heq]; decide
              · have heq : codeword C5Sets z = {3} := hcw {3}
                  (hN (by decide) (by rintro ⟨h1, _, _⟩; linarith))
                  (hN (by decide) (by rintro ⟨h1, _, _⟩; linarith))
                  (hN (by decide) (by rintro ⟨_, _, _, h2⟩; linarith))
                  (hY (by decide) ⟨by linarith, by linarith, by linarith, by linarith⟩)
                  (hN (by decide) (by rintro ⟨h1, _, _⟩; linarith))
                rw [heq]; decide

/-- Helper for `C5_closed_convex_realization`: all 12 words of `codeC5` are attained by `C5Sets`. -/
private lemma C5Sets_witnesses {σ : Finset (Fin 5)} (hσ : σ ∈ codeC5) :
    σ ∈ neuralCode C5Sets := by
  have hpt : ∀ (a b : ℝ) (S : Finset (Fin 5)),
      ((0 : Fin 5) ∈ S ↔ b = 1 ∧ -1 ≤ a ∧ a ≤ 1) →
      ((1 : Fin 5) ∈ S ↔ a = -b ∧ 0 ≤ b ∧ b ≤ 1) →
      ((2 : Fin 5) ∈ S ↔ 0 ≤ b ∧ b ≤ 1 ∧ -b ≤ a ∧ a ≤ 0) →
      ((3 : Fin 5) ∈ S ↔ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ a ∧ a ≤ b) →
      ((4 : Fin 5) ∈ S ↔ a = b ∧ 0 ≤ b ∧ b ≤ 1) →
      S ∈ neuralCode C5Sets := by
    intro a b S h0 h1 h2 h3 h4
    refine ⟨![a, b], ?_⟩
    ext i; rw [mem_codeword_iff]
    fin_cases i
    · exact h0.symm
    · exact h1.symm
    · exact h2.symm
    · exact h3.symm
    · exact h4.symm
  have hY : ∀ {i : Fin 5} {S : Finset (Fin 5)} {P : Prop}, i ∈ S → P → (i ∈ S ↔ P) :=
    fun hi hP => ⟨fun _ => hP, fun _ => hi⟩
  have hN : ∀ {i : Fin 5} {S : Finset (Fin 5)} {P : Prop}, i ∉ S → ¬ P → (i ∈ S ↔ P) :=
    fun hi hP => ⟨fun h => absurd h hi, fun h => absurd h hP⟩
  simp only [codeC5, Finset.mem_insert, Finset.mem_singleton] at hσ
  rcases hσ with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact hpt 0 (-1) ∅ (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num))
      (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num))
  · exact hpt 0 0 {1, 2, 3, 4} (hN (by decide) (by norm_num)) (hY (by decide) (by norm_num))
      (hY (by decide) (by norm_num)) (hY (by decide) (by norm_num)) (hY (by decide) (by norm_num))
  · exact hpt (-1) 1 {0, 1, 2} (hY (by decide) (by norm_num)) (hY (by decide) (by norm_num))
      (hY (by decide) (by norm_num)) (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num))
  · exact hpt 0 1 {0, 2, 3} (hY (by decide) (by norm_num)) (hN (by decide) (by norm_num))
      (hY (by decide) (by norm_num)) (hY (by decide) (by norm_num)) (hN (by decide) (by norm_num))
  · exact hpt 1 1 {0, 3, 4} (hY (by decide) (by norm_num)) (hN (by decide) (by norm_num))
      (hN (by decide) (by norm_num)) (hY (by decide) (by norm_num)) (hY (by decide) (by norm_num))
  · exact hpt (-(1/2)) 1 {0, 2} (hY (by decide) (by norm_num)) (hN (by decide) (by norm_num))
      (hY (by decide) (by norm_num)) (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num))
  · exact hpt (1/2) 1 {0, 3} (hY (by decide) (by norm_num)) (hN (by decide) (by norm_num))
      (hN (by decide) (by norm_num)) (hY (by decide) (by norm_num)) (hN (by decide) (by norm_num))
  · exact hpt (-(1/2)) (1/2) {1, 2} (hN (by decide) (by norm_num)) (hY (by decide) (by norm_num))
      (hY (by decide) (by norm_num)) (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num))
  · exact hpt 0 (1/2) {2, 3} (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num))
      (hY (by decide) (by norm_num)) (hY (by decide) (by norm_num)) (hN (by decide) (by norm_num))
  · exact hpt (1/2) (1/2) {3, 4} (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num))
      (hN (by decide) (by norm_num)) (hY (by decide) (by norm_num)) (hY (by decide) (by norm_num))
  · exact hpt (-(1/4)) (1/2) {2} (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num))
      (hY (by decide) (by norm_num)) (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num))
  · exact hpt (1/4) (1/2) {3} (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num))
      (hN (by decide) (by norm_num)) (hY (by decide) (by norm_num)) (hN (by decide) (by norm_num))

/-- **Theorem 8(1a)**: `C₅` admits a closed convex realization in `ℝ²`. -/
theorem C5_closed_convex_realization :
    (∀ i : Fin 5, IsClosed (C5Sets i)) ∧
    (∀ i : Fin 5, Convex ℝ (C5Sets i)) ∧
    neuralCode C5Sets = (codeC5 : Set (Finset (Fin 5))) := by
  have hc0 : Continuous (fun z : Fin 2 → ℝ => z 0) := continuous_apply 0
  have hc1 : Continuous (fun z : Fin 2 → ℝ => z 1) := continuous_apply 1
  refine ⟨?_, ?_, ?_⟩
  · intro i
    fin_cases i
    · exact (isClosed_eq hc1 continuous_const).inter
        ((isClosed_le continuous_const hc0).inter (isClosed_le hc0 continuous_const))
    · exact (isClosed_eq hc0 hc1.neg).inter
        ((isClosed_le continuous_const hc1).inter (isClosed_le hc1 continuous_const))
    · exact (isClosed_le continuous_const hc1).inter ((isClosed_le hc1 continuous_const).inter
        ((isClosed_le hc1.neg hc0).inter (isClosed_le hc0 continuous_const)))
    · exact (isClosed_le continuous_const hc1).inter ((isClosed_le hc1 continuous_const).inter
        ((isClosed_le continuous_const hc0).inter (isClosed_le hc0 hc1)))
    · exact (isClosed_eq hc0 hc1).inter
        ((isClosed_le continuous_const hc1).inter (isClosed_le hc1 continuous_const))
  · intro i
    fin_cases i <;> intro x hx y hy a b ha hb hab <;>
      simp only [C5Sets, Set.mem_setOf_eq, Pi.add_apply, Pi.smul_apply, smul_eq_mul] at hx hy ⊢
    · exact ⟨by nlinarith, by nlinarith, by nlinarith⟩
    · exact ⟨by nlinarith, by nlinarith, by nlinarith⟩
    · exact ⟨by nlinarith, by nlinarith, by nlinarith, by nlinarith⟩
    · exact ⟨by nlinarith, by nlinarith, by nlinarith, by nlinarith⟩
    · exact ⟨by nlinarith, by nlinarith, by nlinarith⟩
  · ext σ
    simp only [Finset.mem_coe]
    exact ⟨fun ⟨z, hz⟩ => hz ▸ C5Sets_codeword_mem z, C5Sets_witnesses⟩

/-- Helper lemma: open sets covering a closed segment `[a, b]` must overlap on `(a, b)`. -/
private lemma open_segment_overlap {d : ℕ} {A B : Set (Fin d → ℝ)}
    (hA : IsOpen A) (hB : IsOpen B) {p q : Fin d → ℝ}
    (hcov : ∀ t ∈ Set.Icc (0 : ℝ) 1, (1 - t) • p + t • q ∈ A ∪ B)
    (hpA : p ∈ A) (hpB : p ∉ B) (hqB : q ∈ B) (hqA : q ∉ A) :
    ∃ t ∈ Set.Ioo (0 : ℝ) 1, (1 - t) • p + t • q ∈ A ∩ B := by
  let γ : ℝ → Fin d → ℝ := fun t => (1 - t) • p + t • q
  have hγ : Continuous γ :=
    ((continuous_const.sub continuous_id).smul continuous_const).add
      (continuous_id.smul continuous_const)
  have hdisj_contra : ¬ Disjoint (Set.Icc (0 : ℝ) 1 ∩ γ ⁻¹' A) (Set.Icc (0 : ℝ) 1 ∩ γ ⁻¹' B) := by
    intro hdisj
    have hpre := (isPreconnected_Icc (a := (0 : ℝ)) (b := 1)) (γ ⁻¹' A) (γ ⁻¹' B)
      (hA.preimage hγ) (hB.preimage hγ)
      (fun t ht => hcov t ht)
      ⟨0, Set.left_mem_Icc.mpr zero_le_one, by simpa [γ] using hpA⟩
      ⟨1, Set.right_mem_Icc.mpr zero_le_one, by simpa [γ] using hqB⟩
    obtain ⟨t, ht, htA, htB⟩ := hpre
    exact Set.disjoint_left.mp hdisj ⟨ht, htA⟩ ⟨ht, htB⟩
  rw [Set.not_disjoint_iff] at hdisj_contra
  obtain ⟨t, ⟨ht, htA⟩, ⟨_, htB⟩⟩ := hdisj_contra
  have ht0 : 0 < t := lt_of_le_of_ne ht.1 fun h => by subst h; simp [γ] at htB; exact hpB htB
  have ht1 : t < 1 := lt_of_le_of_ne ht.2 fun h => by subst h; simp [γ] at htA; exact hqA htA
  exact ⟨t, ⟨ht0, ht1⟩, htA, htB⟩

/-- Helper lemma: closed sets covering a closed segment `[a, b]` must overlap on `(a, b)`. -/
private lemma closed_segment_overlap {d : ℕ} {A B : Set (Fin d → ℝ)}
    (hA : IsClosed A) (hB : IsClosed B) {p q : Fin d → ℝ}
    (hcov : ∀ t ∈ Set.Icc (0 : ℝ) 1, (1 - t) • p + t • q ∈ A ∪ B)
    (hpB : p ∉ B) (hqA : q ∉ A) :
    ∃ t ∈ Set.Ioo (0 : ℝ) 1, (1 - t) • p + t • q ∈ A ∩ B := by
  let γ : ℝ → Fin d → ℝ := fun t => (1 - t) • p + t • q
  have hγ : Continuous γ :=
    ((continuous_const.sub continuous_id).smul continuous_const).add
      (continuous_id.smul continuous_const)
  have hdisj_contra : ¬ Disjoint (Set.Icc (0 : ℝ) 1 ∩ γ ⁻¹' A) (Set.Icc (0 : ℝ) 1 ∩ γ ⁻¹' B) := by
    intro hdisj
    have hpre := isPreconnected_iff_subset_of_disjoint_closed.mp
      (isPreconnected_Icc (a := (0 : ℝ)) (b := 1)) (γ ⁻¹' A) (γ ⁻¹' B)
      (hA.preimage hγ) (hB.preimage hγ)
      (fun t ht => hcov t ht)
      (by
        ext t
        simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false]
        rintro ⟨ht, htA, htB⟩
        exact Set.disjoint_left.mp hdisj ⟨ht, htA⟩ ⟨ht, htB⟩)
    rcases hpre with hsubA | hsubB
    · have h1 := hsubA (Set.right_mem_Icc.mpr zero_le_one)
      simp [γ] at h1; exact hqA h1
    · have h0 := hsubB (Set.left_mem_Icc.mpr zero_le_one)
      simp [γ] at h0; exact hpB h0
  rw [Set.not_disjoint_iff] at hdisj_contra
  obtain ⟨t, ⟨ht, htA⟩, ⟨_, htB⟩⟩ := hdisj_contra
  have ht0 : 0 < t := lt_of_le_of_ne ht.1 fun h => by subst h; simp [γ] at htB; exact hpB htB
  have ht1 : t < 1 := lt_of_le_of_ne ht.2 fun h => by subst h; simp [γ] at htA; exact hqA htA
  exact ⟨t, ⟨ht0, ht1⟩, htA, htB⟩

/-- **2D Open Convex Sunflower Lemma** (Lienkaemper–Shiu–Woodstock Lemma 3.2):
If three open convex sets `B₁, B₂, B₃ ⊆ ℝ²` have pairwise intersections `B₁ ∩ B₂ = B₂ ∩ B₃ = Z`
with `Z ⊆ {y > 0}`, `![0, 1] ∈ Z`, `![a₁, 0] ∈ B₁` (`a₁ < 0`), `![0, 0] ∈ B₂`, and `![1, 0] ∈ B₃`,
then a supporting line to `Z` at the lowest point `(0, y_*)` of `Z` on the `y`-axis gives a
contradiction. -/
private lemma open_sunflower_2d_impossible {B₁ B₂ B₃ Z : Set (Fin 2 → ℝ)} {a₁ : ℝ}
    (ha1 : a₁ < 0)
    (hB1_o : IsOpen B₁) (hB1_c : Convex ℝ B₁)
    (hB2_o : IsOpen B₂) (hB2_c : Convex ℝ B₂)
    (hB3_o : IsOpen B₃) (hB3_c : Convex ℝ B₃)
    (hZ_o : IsOpen Z) (hZ_c : Convex ℝ Z)
    (h12 : B₁ ∩ B₂ = Z) (h23 : B₂ ∩ B₃ = Z)
    (hZ_pos : ∀ q ∈ Z, 0 < q 1)
    (hz0 : (![0, 1] : Fin 2 → ℝ) ∈ Z)
    (hp1 : (![a₁, 0] : Fin 2 → ℝ) ∈ B₁)
    (hp2 : (![0, 0] : Fin 2 → ℝ) ∈ B₂)
    (hp3 : (![1, 0] : Fin 2 → ℝ) ∈ B₃) : False := by
  let Yset : Set ℝ := {y : ℝ | (![0, y] : Fin 2 → ℝ) ∈ Z}
  have hY_ne : Yset.Nonempty := ⟨1, hz0⟩
  have hY_bdd : BddBelow Yset := ⟨0, fun y hy => le_of_lt (hZ_pos ![0, y] hy)⟩
  set y_star : ℝ := sInf Yset
  have hy_nonneg : 0 ≤ y_star := le_csInf hY_ne fun y hy => le_of_lt (hZ_pos ![0, y] hy)
  have hy_lt1 : y_star < 1 := by
    obtain ⟨ε, hε, hball⟩ := Metric.isOpen_iff.mp hZ_o ![0, 1] hz0
    set d0 : ℝ := min (1 / 2) (ε / 2)
    have hd0_pos : 0 < d0 := lt_min (by norm_num) (by linarith)
    have hd0_le : d0 ≤ ε / 2 := min_le_right _ _
    have hmem : (![0, 1 - d0] : Fin 2 → ℝ) ∈ Z := by
      apply hball
      rw [Metric.mem_ball, dist_pi_lt_iff hε]
      refine Fin.forall_fin_two.mpr ⟨?_, ?_⟩
      · simpa using hε
      · simp only [Matrix.cons_val_one, Matrix.cons_val_fin_one, Real.dist_eq]
        rw [show 1 - d0 - 1 = -d0 by ring, abs_neg, abs_of_pos hd0_pos]
        linarith
    have hle : y_star ≤ 1 - d0 := csInf_le hY_bdd hmem
    linarith
  let w : Fin 2 → ℝ := ![0, y_star]
  have hw_notZ : w ∉ Z := by
    intro hwZ
    obtain ⟨ε, hε, hball⟩ := Metric.isOpen_iff.mp hZ_o w hwZ
    have hmem : (![0, y_star - ε / 2] : Fin 2 → ℝ) ∈ Z := by
      apply hball
      rw [Metric.mem_ball, dist_pi_lt_iff hε]
      refine Fin.forall_fin_two.mpr ⟨?_, ?_⟩
      · simpa [w] using hε
      · simp only [w, Matrix.cons_val_one, Matrix.cons_val_fin_one, Real.dist_eq]
        rw [show y_star - ε / 2 - y_star = -(ε / 2) by ring, abs_neg, abs_of_pos (by linarith)]
        linarith
    have hle : y_star ≤ y_star - ε / 2 := csInf_le hY_bdd hmem
    linarith
  have hz0_B2 : (![0, 1] : Fin 2 → ℝ) ∈ B₂ := (h23 ▸ hz0).1
  have hw_B2 : w ∈ B₂ := by
    have hcomb : w = (1 - y_star) • (![0, 0] : Fin 2 → ℝ) + y_star • (![0, 1] : Fin 2 → ℝ) := by
      ext i; fin_cases i <;> simp [w]
    rw [hcomb]
    exact hB2_c hp2 hz0_B2 (by linarith) hy_nonneg (by ring)
  have hz_delta : ∀ δ : ℝ, 0 < δ → δ < 1 - y_star → (![0, y_star + δ] : Fin 2 → ℝ) ∈ Z := by
    intro δ hδ hδ1
    obtain ⟨y₁, hy₁, hy₁_lt⟩ := exists_lt_of_csInf_lt hY_ne (show y_star < y_star + δ by linarith)
    have hden : 0 < 1 - y₁ := by linarith
    set t := (y_star + δ - y₁) / (1 - y₁)
    have ht0 : 0 ≤ t := div_nonneg (by linarith) (le_of_lt hden)
    have ht1 : t ≤ 1 := (div_le_one hden).mpr (by linarith)
    have hcomb : (![0, y_star + δ] : Fin 2 → ℝ) =
        (1 - t) • (![0, y₁] : Fin 2 → ℝ) + t • (![0, 1] : Fin 2 → ℝ) := by
      ext i; fin_cases i
      · simp
      · simp only [Fin.mk_one, Pi.add_apply, Pi.smul_apply, smul_eq_mul, Matrix.cons_val_one,
          Matrix.cons_val_fin_one, mul_one, t]
        field_simp [ne_of_gt hden]; ring
    rw [hcomb]
    exact hZ_c hy₁ hz0 (by linarith) ht0 (by ring)
  obtain ⟨ψ, hψ⟩ := geometric_hahn_banach_point_open hZ_c hZ_o hw_notZ
  let α : ℝ := ψ ![1, 0]
  let β : ℝ := ψ ![0, 1]
  have hψ_eval : ∀ q : Fin 2 → ℝ, ψ q = α * q 0 + β * q 1 := by
    intro q
    have hq : q = q 0 • (![1, 0] : Fin 2 → ℝ) + q 1 • (![0, 1] : Fin 2 → ℝ) := by
      ext i; fin_cases i <;> simp
    conv_lhs => rw [hq]
    rw [map_add, map_smul, map_smul, smul_eq_mul, smul_eq_mul]
    dsimp [α, β]; ring
  have hβ_pos : 0 < β := by
    have h1 := hψ ![0, 1] hz0
    rw [hψ_eval, hψ_eval] at h1
    simp only [w, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_fin_one,
      mul_zero, zero_add, mul_one] at h1
    nlinarith
  have hbranch : ∀ (a_end : ℝ) (B_end : Set (Fin 2 → ℝ)),
      IsOpen B_end → Convex ℝ B_end → B₂ ∩ B_end = Z →
      (![a_end, 0] : Fin 2 → ℝ) ∈ B_end → α * a_end ≤ 0 → False := by
    intro a_end B_end hBe_o hBe_c hBe_inter hpend hαa
    obtain ⟨r, hr_pos, hr_ball⟩ := Metric.isOpen_iff.mp hB2_o w hw_B2
    obtain ⟨η, hη_pos, hη_ball⟩ := Metric.isOpen_iff.mp hBe_o ![a_end, 0] hpend
    let p_end' : Fin 2 → ℝ := ![a_end, -(η / 2)]
    have hp'_mem : p_end' ∈ B_end := by
      apply hη_ball
      rw [Metric.mem_ball, dist_pi_lt_iff hη_pos]
      refine Fin.forall_fin_two.mpr ⟨?_, ?_⟩
      · simpa [p_end'] using hη_pos
      · simp only [p_end', Matrix.cons_val_one, Matrix.cons_val_fin_one, Real.dist_eq, sub_zero,
          abs_neg, abs_of_pos (by linarith : 0 < η / 2)]
        linarith
    let D : ℝ := max (|a_end|) (y_star + η / 2) + 1
    have hD_pos : 0 < D := by
      have : 0 ≤ max (|a_end|) (y_star + η / 2) := le_trans (abs_nonneg a_end) (le_max_left _ _)
      linarith
    let t : ℝ := min (1 / 2) (r / (4 * D))
    have ht_pos : 0 < t := lt_min (by norm_num) (by positivity)
    have ht_lt1 : t < 1 := lt_of_le_of_lt (min_le_left _ _) (by norm_num)
    let δ : ℝ := min ((1 - y_star) / 2) (min (r / 4) ((t / (1 - t)) * (η / 4)))
    have hδ_pos : 0 < δ := lt_min (half_pos (sub_pos.mpr hy_lt1))
      (lt_min (div_pos hr_pos (by norm_num))
        (mul_pos (div_pos ht_pos (sub_pos.mpr ht_lt1)) (div_pos hη_pos (by norm_num))))
    have hδ_lt : δ < 1 - y_star := lt_of_le_of_lt (min_le_left _ _) (half_lt_self (sub_pos.mpr hy_lt1))
    let z_d : Fin 2 → ℝ := ![0, y_star + δ]
    have hzd_Z : z_d ∈ Z := hz_delta δ hδ_pos hδ_lt
    have hzd_Be : z_d ∈ B_end := (hBe_inter ▸ hzd_Z).2
    let u : Fin 2 → ℝ := (1 - t) • z_d + t • p_end'
    have hu_Be : u ∈ B_end := hBe_c hzd_Be hp'_mem (by linarith) (le_of_lt ht_pos) (by ring)
    have hu_B2 : u ∈ B₂ := by
      apply hr_ball
      rw [Metric.mem_ball, dist_pi_lt_iff hr_pos]
      refine Fin.forall_fin_two.mpr ⟨?_, ?_⟩
      · simp only [u, z_d, p_end', w, Pi.add_apply, Pi.smul_apply, smul_eq_mul,
          Matrix.cons_val_zero, mul_zero, zero_add, Real.dist_eq, sub_zero, abs_mul,
          abs_of_pos ht_pos]
        have ht_le : t ≤ r / (4 * D) := min_le_right _ _
        have ha_le : |a_end| ≤ D := le_trans (le_max_left _ _) (le_add_of_nonneg_right zero_le_one)
        have h1 : t * |a_end| ≤ (r / (4 * D)) * D := mul_le_mul ht_le ha_le (abs_nonneg _) (by positivity)
        have h2 : (r / (4 * D)) * D = r / 4 := by field_simp [ne_of_gt hD_pos]
        linarith only [h1, h2, hr_pos]
      · simp only [u, z_d, p_end', w, Pi.add_apply, Pi.smul_apply, smul_eq_mul,
          Matrix.cons_val_one, Matrix.cons_val_fin_one, Real.dist_eq]
        have hrew : (1 - t) * (y_star + δ) + t * -(η / 2) - y_star =
            (1 - t) * δ - t * (y_star + η / 2) := by ring
        rw [hrew]
        have htri : |(1 - t) * δ - t * (y_star + η / 2)| ≤
            |(1 - t) * δ| + |t * (y_star + η / 2)| := abs_sub _ _
        have h1 : |(1 - t) * δ| ≤ r / 4 := by
          rw [abs_of_pos (mul_pos (sub_pos.mpr ht_lt1) hδ_pos)]
          have hδr : δ ≤ r / 4 := le_trans (min_le_right _ _) (min_le_left _ _)
          nlinarith only [hδr, ht_pos, hδ_pos]
        have h2 : |t * (y_star + η / 2)| ≤ r / 4 := by
          have hyη_pos : 0 < y_star + η / 2 := add_pos_of_nonneg_of_pos hy_nonneg (half_pos hη_pos)
          rw [abs_of_pos (mul_pos ht_pos hyη_pos)]
          have ht_le : t ≤ r / (4 * D) := min_le_right _ _
          have hy_le : y_star + η / 2 ≤ D := le_trans (le_max_right _ _) (le_add_of_nonneg_right zero_le_one)
          have hm := mul_le_mul ht_le hy_le (le_of_lt hyη_pos) (by positivity)
          have hc : (r / (4 * D)) * D = r / 4 := by field_simp [ne_of_gt hD_pos]
          linarith only [hm, hc]
        linarith only [htri, h1, h2, hr_pos]
    have hu_Z : u ∈ Z := hBe_inter ▸ ⟨hu_B2, hu_Be⟩
    have hψ_u := hψ u hu_Z
    rw [hψ_eval u, hψ_eval w] at hψ_u
    simp only [u, z_d, p_end', w, Pi.add_apply, Pi.smul_apply, smul_eq_mul,
      Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_fin_one, mul_zero, zero_add] at hψ_u
    have hδ_mul : (1 - t) * δ ≤ t * (η / 4) := by
      have hδ_t : δ ≤ (t / (1 - t)) * (η / 4) := le_trans (min_le_right _ _) (min_le_right _ _)
      have h1 := mul_le_mul_of_nonneg_left hδ_t (sub_nonneg.mpr (le_of_lt ht_lt1))
      rw [← mul_assoc, mul_div_cancel₀ _ (sub_ne_zero.mpr (ne_of_gt ht_lt1))] at h1
      exact h1
    have hαt : α * (t * a_end) ≤ 0 := by nlinarith only [ht_pos, hαa]
    have hβ_part : (1 - t) * (y_star + δ) + t * -(η / 2) - y_star ≤ -(t * (η / 4)) := by
      nlinarith only [hδ_mul, ht_pos, hy_nonneg]
    have hneg_val : α * (t * a_end) + β * ((1 - t) * (y_star + δ) + t * -(η / 2)) < β * y_star := by
      nlinarith only [hαt, hβ_part, hβ_pos, mul_pos ht_pos hη_pos]
    linarith only [hψ_u, hneg_val]
  by_cases hα : α ≤ 0
  · exact hbranch 1 B₃ hB3_o hB3_c h23 hp3 (by linarith)
  · push_neg at hα
    exact hbranch a₁ B₁ hB1_o hB1_c (by rw [Set.inter_comm, h12]) hp1 (by nlinarith)

/-- **Theorem 8(1b)**: `C₅` admits NO open convex realization in ANY Euclidean space `ℝ^d`! -/
theorem C5_not_open_convex {d : ℕ} (U : Fin 5 → Set (Fin d → ℝ))
    (hopen : ∀ i, IsOpen (U i)) (hconv : ∀ i, Convex ℝ (U i)) :
    neuralCode U ≠ (codeC5 : Set (Finset (Fin 5))) := by
  intro hcode
  have hmem : ∀ σ : Finset (Fin 5), σ ∈ codeC5 ↔ ∃ x : Fin d → ℝ, codeword U x = σ := by
    intro σ
    have := Set.ext_iff.mp hcode σ
    simpa [neuralCode] using this.symm
  -- Combinatorial properties of `C₅`:
  have hU0_sub : U 0 ⊆ U 2 ∪ U 3 := by
    intro x hx
    have hcx : codeword U x ∈ codeC5 := (hmem _).mpr ⟨x, rfl⟩
    have h0 : (0 : Fin 5) ∈ codeword U x := (mem_codeword_iff U x 0).mpr hx
    simp only [codeC5, Finset.mem_insert, Finset.mem_singleton] at hcx
    rcases hcx with h | h | h | h | h | h | h | h | h | h | h | h <;> rw [h] at h0
    all_goals first
      | (exfalso; revert h0; decide)
      | (left; exact (mem_codeword_iff U x 2).mp (by rw [h]; decide))
      | (right; exact (mem_codeword_iff U x 3).mp (by rw [h]; decide))
  let A₁ := U 1 ∩ U 2
  let A₂ := U 2 ∩ U 3
  let A₃ := U 3 ∩ U 4
  let Z := U 1 ∩ U 2 ∩ U 3 ∩ U 4
  have hA12 : A₁ ∩ A₂ = Z := by
    ext x
    simp only [A₁, A₂, Z, Set.mem_inter_iff]
    constructor
    · rintro ⟨⟨h1, h2⟩, ⟨_, h3⟩⟩
      have hcx : codeword U x ∈ codeC5 := (hmem _).mpr ⟨x, rfl⟩
      have hm1 : (1 : Fin 5) ∈ codeword U x := (mem_codeword_iff U x 1).mpr h1
      have hm2 : (2 : Fin 5) ∈ codeword U x := (mem_codeword_iff U x 2).mpr h2
      have hm3 : (3 : Fin 5) ∈ codeword U x := (mem_codeword_iff U x 3).mpr h3
      have h4 : (4 : Fin 5) ∈ codeword U x := by
        simp only [codeC5, Finset.mem_insert, Finset.mem_singleton] at hcx
        rcases hcx with h | h | h | h | h | h | h | h | h | h | h | h <;> rw [h] at hm1 hm2 hm3 ⊢
        all_goals first | decide | (exfalso; revert hm1 hm2 hm3; decide)
      exact ⟨⟨⟨h1, h2⟩, h3⟩, (mem_codeword_iff U x 4).mp h4⟩
    · rintro ⟨⟨⟨h1, h2⟩, h3⟩, _⟩; exact ⟨⟨h1, h2⟩, ⟨h2, h3⟩⟩
  have hA23 : A₂ ∩ A₃ = Z := by
    ext x
    simp only [A₂, A₃, Z, Set.mem_inter_iff]
    constructor
    · rintro ⟨⟨h2, h3⟩, ⟨_, h4⟩⟩
      have hcx : codeword U x ∈ codeC5 := (hmem _).mpr ⟨x, rfl⟩
      have hm2 : (2 : Fin 5) ∈ codeword U x := (mem_codeword_iff U x 2).mpr h2
      have hm3 : (3 : Fin 5) ∈ codeword U x := (mem_codeword_iff U x 3).mpr h3
      have hm4 : (4 : Fin 5) ∈ codeword U x := (mem_codeword_iff U x 4).mpr h4
      have h1 : (1 : Fin 5) ∈ codeword U x := by
        simp only [codeC5, Finset.mem_insert, Finset.mem_singleton] at hcx
        rcases hcx with h | h | h | h | h | h | h | h | h | h | h | h <;> rw [h] at hm2 hm3 hm4 ⊢
        all_goals first | decide | (exfalso; revert hm2 hm3 hm4; decide)
      exact ⟨⟨⟨(mem_codeword_iff U x 1).mp h1, h2⟩, h3⟩, h4⟩
    · rintro ⟨⟨⟨_, h2⟩, h3⟩, h4⟩; exact ⟨⟨h2, h3⟩, ⟨h3, h4⟩⟩
  have hU0_Z : Disjoint (U 0) Z := by
    rw [Set.disjoint_left]
    intro x h0 ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩
    have hcx : codeword U x ∈ codeC5 := (hmem _).mpr ⟨x, rfl⟩
    have hm0 : (0 : Fin 5) ∈ codeword U x := (mem_codeword_iff U x 0).mpr h0
    have hm1 : (1 : Fin 5) ∈ codeword U x := (mem_codeword_iff U x 1).mpr h1
    have hm4 : (4 : Fin 5) ∈ codeword U x := (mem_codeword_iff U x 4).mpr h4
    simp only [codeC5, Finset.mem_insert, Finset.mem_singleton] at hcx
    rcases hcx with h | h | h | h | h | h | h | h | h | h | h | h <;> rw [h] at hm0 hm1 hm4
    all_goals (revert hm0 hm1 hm4; decide)
  obtain ⟨p₁, hp₁⟩ := (hmem {0, 1, 2}).mp (by decide)
  obtain ⟨p₃, hp₃⟩ := (hmem {0, 3, 4}).mp (by decide)
  obtain ⟨z₀, hz₀⟩ := (hmem {1, 2, 3, 4}).mp (by decide)
  have hp1_0 : p₁ ∈ U 0 := (mem_codeword_iff U p₁ 0).mp (by rw [hp₁]; decide)
  have hp1_1 : p₁ ∈ U 1 := (mem_codeword_iff U p₁ 1).mp (by rw [hp₁]; decide)
  have hp1_2 : p₁ ∈ U 2 := (mem_codeword_iff U p₁ 2).mp (by rw [hp₁]; decide)
  have hp1_n3 : p₁ ∉ U 3 := fun h => by
    have := (mem_codeword_iff U p₁ 3).mpr h; rw [hp₁] at this; revert this; decide
  have hp3_0 : p₃ ∈ U 0 := (mem_codeword_iff U p₃ 0).mp (by rw [hp₃]; decide)
  have hp3_3 : p₃ ∈ U 3 := (mem_codeword_iff U p₃ 3).mp (by rw [hp₃]; decide)
  have hp3_4 : p₃ ∈ U 4 := (mem_codeword_iff U p₃ 4).mp (by rw [hp₃]; decide)
  have hp3_n2 : p₃ ∉ U 2 := fun h => by
    have := (mem_codeword_iff U p₃ 2).mpr h; rw [hp₃] at this; revert this; decide
  have hz0_Z : z₀ ∈ Z := ⟨⟨⟨(mem_codeword_iff U z₀ 1).mp (by rw [hz₀]; decide),
    (mem_codeword_iff U z₀ 2).mp (by rw [hz₀]; decide)⟩,
    (mem_codeword_iff U z₀ 3).mp (by rw [hz₀]; decide)⟩,
    (mem_codeword_iff U z₀ 4).mp (by rw [hz₀]; decide)⟩
  obtain ⟨s, hs, hp2_A2⟩ := open_segment_overlap (hopen 2) (hopen 3)
    (fun t ht => hU0_sub (hconv 0 hp1_0 hp3_0 (by linarith [ht.2]) ht.1 (by ring)))
    hp1_2 hp1_n3 hp3_3 hp3_n2
  set p₂ : Fin d → ℝ := (1 - s) • p₁ + s • p₃
  let Φ : (Fin 2 → ℝ) → (Fin d → ℝ) := fun q => p₂ + q 0 • (p₃ - p₂) + q 1 • (z₀ - p₂)
  have hΦ_cont : Continuous Φ :=
    (continuous_const.add ((continuous_apply 0).smul continuous_const)).add
      ((continuous_apply 1).smul continuous_const)
  have hΦ_affine : ∀ (q₁ q₂ : Fin 2 → ℝ) (a b : ℝ), a + b = 1 →
      Φ (a • q₁ + b • q₂) = a • Φ q₁ + b • Φ q₂ := by
    intro q₁ q₂ a b hab
    have hb : b = 1 - a := by linarith
    subst hb
    ext k
    simp only [Φ, Pi.add_apply, Pi.smul_apply, Pi.sub_apply, smul_eq_mul]
    ring
  have hΦ_pre_conv : ∀ {S : Set (Fin d → ℝ)}, Convex ℝ S → Convex ℝ (Φ ⁻¹' S) := by
    intro S hS q₁ hq₁ q₂ hq₂ a b ha hb hab
    simp only [Set.mem_preimage] at hq₁ hq₂ ⊢
    rw [hΦ_affine q₁ q₂ a b hab]
    exact hS hq₁ hq₂ ha hb hab
  let a₁ : ℝ := -(s / (1 - s))
  have ha1_neg : a₁ < 0 := neg_lt_zero.mpr (div_pos hs.1 (by linarith [hs.2]))
  have hΦ_p1 : Φ ![a₁, 0] = p₁ := by
    ext k
    simp only [Φ, p₂, a₁, Pi.add_apply, Pi.smul_apply, Pi.sub_apply, smul_eq_mul,
      Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_fin_one, zero_mul, add_zero]
    have h1s : 1 - s ≠ 0 := by linarith [hs.2]
    field_simp [h1s]; ring
  have hΦ_p2 : Φ ![0, 0] = p₂ := by ext k; simp [Φ]
  have hΦ_p3 : Φ ![1, 0] = p₃ := by ext k; simp [Φ]
  have hΦ_z0 : Φ ![0, 1] = z₀ := by ext k; simp [Φ]
  have hline_disj : ∀ x : ℝ, Φ ![x, 0] ∉ Z := by
    intro x hxZ
    by_cases hmid : a₁ ≤ x ∧ x ≤ 1
    · have hden : 0 < 1 - a₁ := by linarith
      set t := (x - a₁) / (1 - a₁)
      have ht0 : 0 ≤ t := div_nonneg (by linarith) (le_of_lt hden)
      have ht1 : t ≤ 1 := (div_le_one hden).mpr (by linarith)
      have hcomb : Φ ![x, 0] = (1 - t) • p₁ + t • p₃ := by
        rw [← hΦ_p1, ← hΦ_p3, ← hΦ_affine ![a₁, 0] ![1, 0] (1 - t) t (by ring)]
        congr 1; ext i; fin_cases i
        · simp only [Fin.zero_eta, Pi.add_apply, Pi.smul_apply, smul_eq_mul, Matrix.cons_val_zero,
            mul_one, t]; field_simp [ne_of_gt hden]; ring
        · simp
      rw [hcomb] at hxZ
      exact Set.disjoint_left.mp hU0_Z (hconv 0 hp1_0 hp3_0 (by linarith) ht0 (by ring)) hxZ
    · by_cases hlt : x < a₁
      · set t := a₁ / x
        have ht0 : 0 < t := div_pos_of_neg_of_neg ha1_neg (by linarith)
        have ht1 : t < 1 := by rw [div_lt_one_of_neg (by linarith)]; exact hlt
        have hcomb : p₁ = t • Φ ![x, 0] + (1 - t) • p₂ := by
          rw [← hΦ_p1, ← hΦ_p2, ← hΦ_affine ![x, 0] ![0, 0] t (1 - t) (by ring)]
          congr 1; ext i; fin_cases i
          · simp only [Fin.zero_eta, Pi.add_apply, Pi.smul_apply, smul_eq_mul, Matrix.cons_val_zero,
              mul_zero, add_zero, t]; field_simp [show x ≠ 0 by linarith]
          · simp
        exact hp1_n3 (hcomb ▸ hconv 3 hxZ.1.2 hp2_A2.2 (le_of_lt ht0) (by linarith) (by ring))
      · have hgt : 1 < x := by
          rcases not_and_or.mp hmid with h | h
          · exact absurd (not_lt.mp hlt) h
          · exact not_le.mp h
        set t := 1 / x
        have ht0 : 0 < t := div_pos one_pos (by linarith)
        have ht1 : t < 1 := (div_lt_one (by linarith)).mpr hgt
        have hcomb : p₃ = (1 - t) • p₂ + t • Φ ![x, 0] := by
          rw [← hΦ_p3, ← hΦ_p2, ← hΦ_affine ![0, 0] ![x, 0] (1 - t) t (by ring)]
          congr 1; ext i; fin_cases i
          · simp only [Fin.zero_eta, Pi.add_apply, Pi.smul_apply, smul_eq_mul, Matrix.cons_val_zero,
              mul_zero, zero_add, t]; field_simp [show x ≠ 0 by linarith]
          · simp
        exact hp3_n2 (hcomb ▸ hconv 2 hp2_A2.1 hxZ.1.1.2 (by linarith) (le_of_lt ht0) (by ring))
  have hZ_pl_conv : Convex ℝ (Φ ⁻¹' Z) :=
    hΦ_pre_conv ((((hconv 1).inter (hconv 2)).inter (hconv 3)).inter (hconv 4))
  have hZ_pos_y : ∀ q ∈ Φ ⁻¹' Z, 0 < q 1 := by
    intro q hq
    by_contra hle
    push_neg at hle
    rcases eq_or_lt_of_le hle with h0 | hneg
    · have hqeq : q = ![q 0, 0] := by ext i; fin_cases i <;> simp [h0]
      exact hline_disj (q 0) (hqeq ▸ hq)
    · set t := (-q 1) / (1 - q 1)
      have hden : 0 < 1 - q 1 := by linarith
      have ht0 : 0 ≤ t := div_nonneg (by linarith) (le_of_lt hden)
      have ht1 : t ≤ 1 := (div_le_one hden).mpr (by linarith)
      have hcross : (1 - t) • q + t • (![0, 1] : Fin 2 → ℝ) ∈ Φ ⁻¹' Z :=
        hZ_pl_conv hq (show Φ ![0, 1] ∈ Z by rw [hΦ_z0]; exact hz0_Z)
          (a := 1 - t) (b := t) (by linarith) ht0 (by ring)
      have hy0 : ((1 - t) • q + t • (![0, 1] : Fin 2 → ℝ)) 1 = 0 := by
        simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, Matrix.cons_val_one,
          Matrix.cons_val_fin_one, mul_one, t]
        field_simp [ne_of_gt hden]; ring
      have hqeq : (1 - t) • q + t • (![0, 1] : Fin 2 → ℝ) =
          ![((1 - t) • q + t • (![0, 1] : Fin 2 → ℝ)) 0, 0] := by
        ext i; fin_cases i
        · rfl
        · exact hy0
      exact hline_disj _ (hqeq ▸ hcross)
  exact open_sunflower_2d_impossible ha1_neg
    (((hopen 1).inter (hopen 2)).preimage hΦ_cont) (hΦ_pre_conv ((hconv 1).inter (hconv 2)))
    (((hopen 2).inter (hopen 3)).preimage hΦ_cont) (hΦ_pre_conv ((hconv 2).inter (hconv 3)))
    (((hopen 3).inter (hopen 4)).preimage hΦ_cont) (hΦ_pre_conv ((hconv 3).inter (hconv 4)))
    (((((hopen 1).inter (hopen 2)).inter (hopen 3)).inter (hopen 4)).preimage hΦ_cont) hZ_pl_conv
    (by rw [← Set.preimage_inter, hA12]) (by rw [← Set.preimage_inter, hA23])
    hZ_pos_y (by show Φ ![0, 1] ∈ Z; rw [hΦ_z0]; exact hz0_Z)
    (by show Φ ![a₁, 0] ∈ A₁; rw [hΦ_p1]; exact ⟨hp1_1, hp1_2⟩)
    (by show Φ ![0, 0] ∈ A₂; rw [hΦ_p2]; exact hp2_A2)
    (by show Φ ![1, 0] ∈ A₃; rw [hΦ_p3]; exact ⟨hp3_3, hp3_4⟩)

/-- The 6-neuron, 13-word central positive-halfplane code `C₆ = L⁺(M₆)` (CGIK Lemma 2.9,
0-indexed: neurons `0..5` correspond to `1..6` in `solution.md`). -/
def codeC6 : Finset (Finset (Fin 6)) :=
  {∅, {0, 1, 2}, {1, 2, 3}, {2, 3, 4}, {3, 4, 5}, {4, 5, 0}, {5, 0, 1},
    {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 0}}

/-- Six cyclic normal vectors in `ℝ²` representing the rank-2 oriented matroid `M₆`. -/
def matroidM6 : Fin 6 → Fin 2 → ℝ
  | 0 => ![1, 0]
  | 1 => ![1, 1]
  | 2 => ![0, 1]
  | 3 => ![-1, 0]
  | 4 => ![-1, -1]
  | 5 => ![0, -1]

/-- Open positive halfplanes `{z ∈ ℝ² : v_i ⬝ᵥ z > 0}` defining `L⁺(M₆)`. -/
def C6Halfplanes : Fin 6 → Set (Fin 2 → ℝ) :=
  fun i => {z | 0 < dotProduct (matroidM6 i) z}

/-- **Theorem 8(2a)**: `C₆ = L⁺(M₆)` is the exact positive covector code of the rank-2
representable oriented matroid `M₆`. -/
theorem C6_oriented_matroid_realization :
    neuralCode C6Halfplanes = (codeC6 : Set (Finset (Fin 6))) := by
  ext σ
  simp only [neuralCode, Set.mem_range, Finset.mem_coe]
  constructor
  · rintro ⟨z, rfl⟩
    set x := z 0; set y := z 1
    have hdot : ∀ i : Fin 6,
        (i ∈ codeword C6Halfplanes z ↔ match i with
          | 0 => 0 < x
          | 1 => 0 < x + y
          | 2 => 0 < y
          | 3 => x < 0
          | 4 => x + y < 0
          | 5 => y < 0) := by
      intro i
      rw [mem_codeword_iff]
      fin_cases i <;>
        (simp [C6Halfplanes, matroidM6, dotProduct, Fin.sum_univ_two, x, y]
         try exact ⟨fun h => by linarith, fun h => by linarith⟩)
    have hcw : ∀ S : Finset (Fin 6),
        (∀ i : Fin 6, i ∈ S ↔ i ∈ codeword C6Halfplanes z) → codeword C6Halfplanes z = S :=
      fun S hS => Finset.ext fun i => (hS i).symm
    -- Exhaustively split on the signs of `x`, `y`, `x + y`.
    rcases lt_trichotomy x 0 with hx_neg | hx0 | hx_pos <;>
    rcases lt_trichotomy y 0 with hy_neg | hy0 | hy_pos
    · -- x < 0, y < 0 => {3, 4, 5}
      have : codeword C6Halfplanes z = {3, 4, 5} := hcw {3, 4, 5} fun i => by
        rw [hdot i]; fin_cases i <;> simp <;> linarith
      rw [this]; decide
    · -- x < 0, y = 0 => {3, 4}
      have : codeword C6Halfplanes z = {3, 4} := hcw {3, 4} fun i => by
        rw [hdot i]; fin_cases i <;> simp <;> linarith
      rw [this]; decide
    · -- x < 0, y > 0: split on `x + y`.
      rcases lt_trichotomy (x + y) 0 with hxy_neg | hxy0 | hxy_pos
      · have : codeword C6Halfplanes z = {2, 3, 4} := hcw {2, 3, 4} fun i => by
          rw [hdot i]; fin_cases i <;> simp <;> linarith
        rw [this]; decide
      · have : codeword C6Halfplanes z = {2, 3} := hcw {2, 3} fun i => by
          rw [hdot i]; fin_cases i <;> simp <;> linarith
        rw [this]; decide
      · have : codeword C6Halfplanes z = {1, 2, 3} := hcw {1, 2, 3} fun i => by
          rw [hdot i]; fin_cases i <;> simp <;> linarith
        rw [this]; decide
    · -- x = 0, y < 0 => {4, 5}
      have : codeword C6Halfplanes z = {4, 5} := hcw {4, 5} fun i => by
        rw [hdot i]; fin_cases i <;> simp <;> linarith
      rw [this]; decide
    · -- x = 0, y = 0 => ∅
      have : codeword C6Halfplanes z = ∅ := hcw ∅ fun i => by
        rw [hdot i]; fin_cases i <;> simp <;> linarith
      rw [this]; decide
    · -- x = 0, y > 0 => {1, 2}
      have : codeword C6Halfplanes z = {1, 2} := hcw {1, 2} fun i => by
        rw [hdot i]; fin_cases i <;> simp <;> linarith
      rw [this]; decide
    · -- x > 0, y < 0: split on `x + y`.
      rcases lt_trichotomy (x + y) 0 with hxy_neg | hxy0 | hxy_pos
      · have : codeword C6Halfplanes z = {4, 5, 0} := hcw {4, 5, 0} fun i => by
          rw [hdot i]; fin_cases i <;> simp <;> linarith
        rw [this]; decide
      · have : codeword C6Halfplanes z = {5, 0} := hcw {5, 0} fun i => by
          rw [hdot i]; fin_cases i <;> simp <;> linarith
        rw [this]; decide
      · have : codeword C6Halfplanes z = {5, 0, 1} := hcw {5, 0, 1} fun i => by
          rw [hdot i]; fin_cases i <;> simp <;> linarith
        rw [this]; decide
    · -- x > 0, y = 0 => {0, 1}
      have : codeword C6Halfplanes z = {0, 1} := hcw {0, 1} fun i => by
        rw [hdot i]; fin_cases i <;> simp <;> linarith
      rw [this]; decide
    · -- x > 0, y > 0 => {0, 1, 2}
      have : codeword C6Halfplanes z = {0, 1, 2} := hcw {0, 1, 2} fun i => by
        rw [hdot i]; fin_cases i <;> simp <;> linarith
      rw [this]; decide
  · intro hσ
    have hpt : ∀ (a b : ℝ) (S : Finset (Fin 6)),
        ((0 : Fin 6) ∈ S ↔ 0 < a) →
        ((1 : Fin 6) ∈ S ↔ 0 < a + b) →
        ((2 : Fin 6) ∈ S ↔ 0 < b) →
        ((3 : Fin 6) ∈ S ↔ a < 0) →
        ((4 : Fin 6) ∈ S ↔ a + b < 0) →
        ((5 : Fin 6) ∈ S ↔ b < 0) →
        ∃ z : Fin 2 → ℝ, codeword C6Halfplanes z = S := by
      intro a b S h0 h1 h2 h3 h4 h5
      refine ⟨![a, b], ?_⟩
      ext i; rw [mem_codeword_iff]
      fin_cases i
      · simp [C6Halfplanes, matroidM6, dotProduct, Fin.sum_univ_two]; exact h0.symm
      · simp [C6Halfplanes, matroidM6, dotProduct, Fin.sum_univ_two]; exact h1.symm
      · simp [C6Halfplanes, matroidM6, dotProduct, Fin.sum_univ_two]; exact h2.symm
      · simp [C6Halfplanes, matroidM6, dotProduct, Fin.sum_univ_two]; exact h3.symm
      · simp [C6Halfplanes, matroidM6, dotProduct, Fin.sum_univ_two]
        rw [h4]; exact ⟨fun h => by linarith, fun h => by linarith⟩
      · simp [C6Halfplanes, matroidM6, dotProduct, Fin.sum_univ_two]; exact h5.symm
    have hY : ∀ {i : Fin 6} {S : Finset (Fin 6)} {P : Prop}, i ∈ S → P → (i ∈ S ↔ P) :=
      fun hi hP => ⟨fun _ => hP, fun _ => hi⟩
    have hN : ∀ {i : Fin 6} {S : Finset (Fin 6)} {P : Prop}, i ∉ S → ¬ P → (i ∈ S ↔ P) :=
      fun hi hP => ⟨fun h => absurd h hi, fun h => absurd h hP⟩
    simp only [codeC6, Finset.mem_insert, Finset.mem_singleton] at hσ
    rcases hσ with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · exact hpt 0 0 ∅ (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num))
        (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num))
    · exact hpt 1 1 {0, 1, 2} (hY (by decide) (by norm_num)) (hY (by decide) (by norm_num)) (hY (by decide) (by norm_num))
        (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num))
    · exact hpt (-1) 2 {1, 2, 3} (hN (by decide) (by norm_num)) (hY (by decide) (by norm_num)) (hY (by decide) (by norm_num))
        (hY (by decide) (by norm_num)) (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num))
    · exact hpt (-2) 1 {2, 3, 4} (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num)) (hY (by decide) (by norm_num))
        (hY (by decide) (by norm_num)) (hY (by decide) (by norm_num)) (hN (by decide) (by norm_num))
    · exact hpt (-1) (-1) {3, 4, 5} (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num))
        (hY (by decide) (by norm_num)) (hY (by decide) (by norm_num)) (hY (by decide) (by norm_num))
    · exact hpt 1 (-2) {4, 5, 0} (hY (by decide) (by norm_num)) (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num))
        (hN (by decide) (by norm_num)) (hY (by decide) (by norm_num)) (hY (by decide) (by norm_num))
    · exact hpt 2 (-1) {5, 0, 1} (hY (by decide) (by norm_num)) (hY (by decide) (by norm_num)) (hN (by decide) (by norm_num))
        (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num)) (hY (by decide) (by norm_num))
    · exact hpt 1 0 {0, 1} (hY (by decide) (by norm_num)) (hY (by decide) (by norm_num)) (hN (by decide) (by norm_num))
        (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num))
    · exact hpt 0 1 {1, 2} (hN (by decide) (by norm_num)) (hY (by decide) (by norm_num)) (hY (by decide) (by norm_num))
        (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num))
    · exact hpt (-1) 1 {2, 3} (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num)) (hY (by decide) (by norm_num))
        (hY (by decide) (by norm_num)) (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num))
    · exact hpt (-1) 0 {3, 4} (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num))
        (hY (by decide) (by norm_num)) (hY (by decide) (by norm_num)) (hN (by decide) (by norm_num))
    · exact hpt 0 (-1) {4, 5} (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num))
        (hN (by decide) (by norm_num)) (hY (by decide) (by norm_num)) (hY (by decide) (by norm_num))
    · exact hpt 1 (-1) {5, 0} (hY (by decide) (by norm_num)) (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num))
        (hN (by decide) (by norm_num)) (hN (by decide) (by norm_num)) (hY (by decide) (by norm_num))

private lemma codeC6_props :
    (∀ σ ∈ codeC6, ¬ ((0 : Fin 6) ∈ σ ∧ (3 : Fin 6) ∈ σ)) ∧
    (∀ σ ∈ codeC6, ¬ ((1 : Fin 6) ∈ σ ∧ (4 : Fin 6) ∈ σ)) ∧
    (∀ σ ∈ codeC6, ¬ ((2 : Fin 6) ∈ σ ∧ (5 : Fin 6) ∈ σ)) ∧
    (∀ σ ∈ codeC6, (0 : Fin 6) ∈ σ → (1 : Fin 6) ∈ σ ∨ (5 : Fin 6) ∈ σ) ∧
    (∀ σ ∈ codeC6, (1 : Fin 6) ∈ σ → (0 : Fin 6) ∈ σ ∨ (2 : Fin 6) ∈ σ) ∧
    (∀ σ ∈ codeC6, (2 : Fin 6) ∈ σ → (1 : Fin 6) ∈ σ ∨ (3 : Fin 6) ∈ σ) := by
  simp [codeC6]

/-- Elementary contraction step for `C6_not_closed_convex`: given `α, β, γ ∈ (0, 1)` and
`tq ∈ [0, 1]`, the nested convex combination `y012 = (1 - γ) • y₁ + γ • y₃` is strictly closer
to the segment `[x234, x450]` than `x012` is to `(1 - tq) • x234 + tq • x450`. -/
private lemma C6_contraction_step {d : ℕ} (x012 x234 x450 : Fin d → ℝ)
    {α β γ tq : ℝ} (hα : α ∈ Set.Ioo (0 : ℝ) 1) (hβ : β ∈ Set.Ioo (0 : ℝ) 1)
    (hγ : γ ∈ Set.Ioo (0 : ℝ) 1) (htq : tq ∈ Set.Icc (0 : ℝ) 1)
    (hδ_pos : 0 < ‖x012 - ((1 - tq) • x234 + tq • x450)‖) :
    let y₁ := (1 - α) • x012 + α • x450
    let y₃ := (1 - β) • x012 + β • x234
    let y012 := (1 - γ) • y₁ + γ • y₃
    ∃ tr ∈ Set.Icc (0 : ℝ) 1,
      ‖y012 - ((1 - tr) • x234 + tr • x450)‖ < ‖x012 - ((1 - tq) • x234 + tq • x450)‖ := by
  intro y₁ y₃ y012
  set lam1 : ℝ := (1 - γ) * α
  set lam2 : ℝ := γ * β
  set lam0 : ℝ := (1 - γ) * (1 - α) + γ * (1 - β)
  have hlam1_pos : 0 < lam1 := mul_pos (sub_pos.mpr hγ.2) hα.1
  have hlam2_pos : 0 < lam2 := mul_pos hγ.1 hβ.1
  have hlam0_pos : 0 < lam0 :=
    add_pos (mul_pos (sub_pos.mpr hγ.2) (sub_pos.mpr hα.2)) (mul_pos hγ.1 (sub_pos.mpr hβ.2))
  have hlam_sum : lam0 + lam1 + lam2 = 1 := by dsimp [lam0, lam1, lam2]; ring
  have hlam0_lt1 : lam0 < 1 := by linarith
  have hden : lam1 + lam2 = 1 - lam0 := by linarith
  have hden_pos : 0 < lam1 + lam2 := add_pos hlam1_pos hlam2_pos
  have hden_ne : lam1 + lam2 ≠ 0 := ne_of_gt hden_pos
  let tp : ℝ := lam1 / (lam1 + lam2)
  have htp0 : 0 ≤ tp := div_nonneg (le_of_lt hlam1_pos) (le_of_lt hden_pos)
  have htp1 : tp ≤ 1 := (div_le_one hden_pos).mpr (by linarith)
  let tr : ℝ := lam0 * tq + (1 - lam0) * tp
  have htr_mem : tr ∈ Set.Icc (0 : ℝ) 1 := by
    refine ⟨add_nonneg (mul_nonneg (le_of_lt hlam0_pos) htq.1)
        (mul_nonneg (sub_nonneg.mpr (le_of_lt hlam0_lt1)) htp0), ?_⟩
    have h1 : lam0 * tq ≤ lam0 * 1 := mul_le_mul_of_nonneg_left htq.2 (le_of_lt hlam0_pos)
    have h2 : (1 - lam0) * tp ≤ (1 - lam0) * 1 :=
      mul_le_mul_of_nonneg_left htp1 (sub_nonneg.mpr (le_of_lt hlam0_lt1))
    linarith
  refine ⟨tr, htr_mem, ?_⟩
  have h1_tp : (1 - lam0) * (1 - tp) = lam2 := by
    rw [← hden]; dsimp [tp]; field_simp [hden_ne]; ring
  have h_tp : (1 - lam0) * tp = lam1 := by
    rw [← hden]; dsimp [tp]; field_simp [hden_ne]
  have h1_tr : 1 - tr = lam0 * (1 - tq) + lam2 := by
    have h0 : 1 - tr = lam0 * (1 - tq) + (1 - lam0) * (1 - tp) := by dsimp [tr]; ring
    rw [h0, h1_tp]
  have h_tr : tr = lam0 * tq + lam1 := by
    have h0 : tr = lam0 * tq + (1 - lam0) * tp := rfl
    rw [h0, h_tp]
  have hdiff_eq : y012 - ((1 - tr) • x234 + tr • x450) =
      lam0 • (x012 - ((1 - tq) • x234 + tq • x450)) := by
    ext k
    have hy : y012 k = lam0 * x012 k + lam1 * x450 k + lam2 * x234 k := by
      simp only [y012, y₁, y₃, lam0, lam1, lam2, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
      ring
    simp only [Pi.sub_apply, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    rw [hy, h1_tr, h_tr]
    ring
  rw [hdiff_eq, norm_smul, Real.norm_eq_abs, abs_of_pos hlam0_pos]
  exact mul_lt_of_lt_one_left hδ_pos hlam0_lt1

/-- Compactification and distance-minimizer helper for `C6_not_closed_convex`. -/
private lemma C6_min_dist_witness {d : ℕ} {K₀ : Set (Fin d → ℝ)} (hK₀_cl : IsClosed K₀)
    {w012 x234 x450 : Fin d → ℝ} (hw012 : w012 ∈ K₀) :
    ∃ (R : ℝ) (x012 : Fin d → ℝ) (tq : ℝ),
      x234 ∈ Metric.closedBall (0 : Fin d → ℝ) R ∧
      x450 ∈ Metric.closedBall (0 : Fin d → ℝ) R ∧
      x012 ∈ K₀ ∧ x012 ∈ Metric.closedBall (0 : Fin d → ℝ) R ∧
      tq ∈ Set.Icc (0 : ℝ) 1 ∧
      ∀ a ∈ K₀, a ∈ Metric.closedBall (0 : Fin d → ℝ) R → ∀ tr ∈ Set.Icc (0 : ℝ) 1,
        ‖x012 - ((1 - tq) • x234 + tq • x450)‖ ≤ ‖a - ((1 - tr) • x234 + tr • x450)‖ := by
  let R : ℝ := max ‖w012‖ (max ‖x234‖ ‖x450‖) + 1
  let B : Set (Fin d → ℝ) := Metric.closedBall 0 R
  have hw012_B : w012 ∈ B := by
    rw [Metric.mem_closedBall, dist_zero_right]
    exact le_trans (le_max_left _ _) (le_add_of_nonneg_right zero_le_one)
  have hx234_B : x234 ∈ B := by
    rw [Metric.mem_closedBall, dist_zero_right]
    exact le_trans (le_trans (le_max_left _ _) (le_max_right _ _)) (le_add_of_nonneg_right zero_le_one)
  have hx450_B : x450 ∈ B := by
    rw [Metric.mem_closedBall, dist_zero_right]
    exact le_trans (le_trans (le_max_right _ _) (le_max_right _ _)) (le_add_of_nonneg_right zero_le_one)
  let K : Set (Fin d → ℝ) := K₀ ∩ B
  have hK_comp : IsCompact K :=
    (isCompact_closedBall (0 : Fin d → ℝ) R).of_isClosed_subset
      (hK₀_cl.inter Metric.isClosed_closedBall) Set.inter_subset_right
  have hK_ne : K.Nonempty := ⟨w012, hw012, hw012_B⟩
  have hcont_f : Continuous (fun p : (Fin d → ℝ) × ℝ => ‖p.1 - ((1 - p.2) • x234 + p.2 • x450)‖) :=
    (continuous_fst.sub
      (((continuous_const.sub continuous_snd).smul continuous_const).add
        (continuous_snd.smul continuous_const))).norm
  obtain ⟨⟨x012, tq⟩, ⟨⟨hx012_K, hx012_B⟩, htq⟩, hmin⟩ :=
    (hK_comp.prod isCompact_Icc).exists_isMinOn
      (hK_ne.prod ⟨(0 : ℝ), Set.left_mem_Icc.mpr zero_le_one⟩) hcont_f.continuousOn
  exact ⟨R, x012, tq, hx234_B, hx450_B, hx012_K, hx012_B, htq,
    fun a haK haB tr htr => hmin (show (a, tr) ∈ K ×ˢ Set.Icc (0 : ℝ) 1 from ⟨⟨haK, haB⟩, htr⟩)⟩

/-- **Theorem 8(2b)**: `C₆ = L⁺(M₆)` admits NO closed convex realization in ANY Euclidean
space `ℝ^d`! (CGIK Lemma 2.9 compact-distance-contraction proof.) -/
theorem C6_not_closed_convex {d : ℕ} (U : Fin 6 → Set (Fin d → ℝ))
    (hclosed : ∀ i, IsClosed (U i)) (hconv : ∀ i, Convex ℝ (U i)) :
    neuralCode U ≠ (codeC6 : Set (Finset (Fin 6))) := by
  intro hcode
  have hmem : ∀ σ : Finset (Fin 6), σ ∈ codeC6 ↔ ∃ x : Fin d → ℝ, codeword U x = σ := by
    intro σ
    have := Set.ext_iff.mp hcode σ
    simpa [neuralCode] using this.symm
  have ⟨hd03, hd14, hd25, hc0, hc1, hc2⟩ := codeC6_props
  have hdisj_opp : ∀ x : Fin d → ℝ,
      ¬ (x ∈ U 0 ∧ x ∈ U 3) ∧ ¬ (x ∈ U 1 ∧ x ∈ U 4) ∧ ¬ (x ∈ U 2 ∧ x ∈ U 5) := by
    intro x
    have hcx : codeword U x ∈ codeC6 := (hmem _).mpr ⟨x, rfl⟩
    exact ⟨fun ⟨ha, hb⟩ => hd03 _ hcx ⟨(mem_codeword_iff U x 0).mpr ha, (mem_codeword_iff U x 3).mpr hb⟩,
      fun ⟨ha, hb⟩ => hd14 _ hcx ⟨(mem_codeword_iff U x 1).mpr ha, (mem_codeword_iff U x 4).mpr hb⟩,
      fun ⟨ha, hb⟩ => hd25 _ hcx ⟨(mem_codeword_iff U x 2).mpr ha, (mem_codeword_iff U x 5).mpr hb⟩⟩
  have hU0_cov : U 0 ⊆ U 1 ∪ U 5 := fun x hx =>
    (hc0 _ ((hmem _).mpr ⟨x, rfl⟩) ((mem_codeword_iff U x 0).mpr hx)).imp
      (mem_codeword_iff U x 1).mp (mem_codeword_iff U x 5).mp
  have hU1_cov : U 1 ⊆ U 0 ∪ U 2 := fun x hx =>
    (hc1 _ ((hmem _).mpr ⟨x, rfl⟩) ((mem_codeword_iff U x 1).mpr hx)).imp
      (mem_codeword_iff U x 0).mp (mem_codeword_iff U x 2).mp
  have hU2_cov : U 2 ⊆ U 1 ∪ U 3 := fun x hx =>
    (hc2 _ ((hmem _).mpr ⟨x, rfl⟩) ((mem_codeword_iff U x 2).mpr hx)).imp
      (mem_codeword_iff U x 1).mp (mem_codeword_iff U x 3).mp
  -- Pick witnesses of the maximal words `{0, 1, 2}`, `{2, 3, 4}`, `{4, 5, 0}`.
  obtain ⟨w012, hw012⟩ := (hmem {0, 1, 2}).mp (by simp [codeC6])
  obtain ⟨x234, hx234⟩ := (hmem {2, 3, 4}).mp (by simp [codeC6])
  obtain ⟨x450, hx450⟩ := (hmem {4, 5, 0}).mp (by simp [codeC6])
  have hw012_K0 : w012 ∈ U 0 ∩ U 1 ∩ U 2 :=
    ⟨⟨(mem_codeword_iff U w012 0).mp (by simp [hw012]),
      (mem_codeword_iff U w012 1).mp (by simp [hw012])⟩,
      (mem_codeword_iff U w012 2).mp (by simp [hw012])⟩
  obtain ⟨R, x012, tq, hx234_B, hx450_B, hx012_K0, hx012_B, htq, hmin⟩ :=
    C6_min_dist_witness (((hclosed 0).inter (hclosed 1)).inter (hclosed 2)) hw012_K0
  have hB_conv : Convex ℝ (Metric.closedBall (0 : Fin d → ℝ) R) := convex_closedBall 0 R
  have hqM_U4 : (1 - tq) • x234 + tq • x450 ∈ U 4 :=
    hconv 4 ((mem_codeword_iff U x234 4).mp (by simp [hx234]))
      ((mem_codeword_iff U x450 4).mp (by simp [hx450])) (sub_nonneg.mpr htq.2) htq.1 (by ring)
  have hδ_pos : 0 < ‖x012 - ((1 - tq) • x234 + tq • x450)‖ := by
    refine norm_pos_iff.mpr (sub_ne_zero.mpr fun heq => ?_)
    exact (hdisj_opp x012).2.1 ⟨hx012_K0.1.2, heq ▸ hqM_U4⟩
  -- Step 1: Segment `[x012, x450] ⊆ U 0 ⊆ U 1 ∪ U 5` gives `y₁ ∈ (x012, x450) ∩ U 0 ∩ U 1 ∩ U 5`.
  have hx450_0 : x450 ∈ U 0 := (mem_codeword_iff U x450 0).mp (by simp [hx450])
  have hx450_n1 : x450 ∉ U 1 := fun h =>
    (hdisj_opp x450).2.1 ⟨h, (mem_codeword_iff U x450 4).mp (by simp [hx450])⟩
  have hx012_n5 : x012 ∉ U 5 := fun h => (hdisj_opp x012).2.2 ⟨hx012_K0.2, h⟩
  obtain ⟨α, hα, hy1_15⟩ := closed_segment_overlap (hclosed 1) (hclosed 5)
    (fun t ht => hU0_cov (hconv 0 hx012_K0.1.1 hx450_0 (sub_nonneg.mpr ht.2) ht.1 (by ring)))
    hx012_n5 hx450_n1
  set y₁ : Fin d → ℝ := (1 - α) • x012 + α • x450
  have hy1_1 : y₁ ∈ U 1 := hy1_15.1
  have hy1_n2 : y₁ ∉ U 2 := fun h => (hdisj_opp y₁).2.2 ⟨h, hy1_15.2⟩
  have hy1_B : y₁ ∈ Metric.closedBall (0 : Fin d → ℝ) R :=
    hB_conv hx012_B hx450_B (sub_nonneg.mpr (le_of_lt hα.2)) (le_of_lt hα.1) (by ring)
  -- Step 2: Segment `[x012, x234] ⊆ U 2 ⊆ U 1 ∪ U 3` gives `y₃ ∈ (x012, x234) ∩ U 1 ∩ U 2 ∩ U 3`.
  have hx234_2 : x234 ∈ U 2 := (mem_codeword_iff U x234 2).mp (by simp [hx234])
  have hx234_n1 : x234 ∉ U 1 := fun h =>
    (hdisj_opp x234).2.1 ⟨h, (mem_codeword_iff U x234 4).mp (by simp [hx234])⟩
  have hx012_n3 : x012 ∉ U 3 := fun h => (hdisj_opp x012).1 ⟨hx012_K0.1.1, h⟩
  obtain ⟨β, hβ, hy3_13⟩ := closed_segment_overlap (hclosed 1) (hclosed 3)
    (fun t ht => hU2_cov (hconv 2 hx012_K0.2 hx234_2 (sub_nonneg.mpr ht.2) ht.1 (by ring)))
    hx012_n3 hx234_n1
  set y₃ : Fin d → ℝ := (1 - β) • x012 + β • x234
  have hy3_1 : y₃ ∈ U 1 := hy3_13.1
  have hy3_n0 : y₃ ∉ U 0 := fun h => (hdisj_opp y₃).1 ⟨h, hy3_13.2⟩
  have hy3_B : y₃ ∈ Metric.closedBall (0 : Fin d → ℝ) R :=
    hB_conv hx012_B hx234_B (sub_nonneg.mpr (le_of_lt hβ.2)) (le_of_lt hβ.1) (by ring)
  -- Step 3: Segment `[y₁, y₃] ⊆ U 1 ⊆ U 0 ∪ U 2` gives `y012 ∈ (y₁, y₃) ∩ K₀ ∩ B`!
  obtain ⟨γ, hγ, hy012_02⟩ := closed_segment_overlap (hclosed 0) (hclosed 2)
    (fun t ht => hU1_cov (hconv 1 hy1_1 hy3_1 (sub_nonneg.mpr ht.2) ht.1 (by ring)))
    hy1_n2 hy3_n0
  set y012 : Fin d → ℝ := (1 - γ) • y₁ + γ • y₃
  have hy012_K0 : y012 ∈ U 0 ∩ U 1 ∩ U 2 :=
    ⟨⟨hy012_02.1, hconv 1 hy1_1 hy3_1 (sub_nonneg.mpr (le_of_lt hγ.2)) (le_of_lt hγ.1) (by ring)⟩,
      hy012_02.2⟩
  have hy012_B : y012 ∈ Metric.closedBall (0 : Fin d → ℝ) R :=
    hB_conv hy1_B hy3_B (sub_nonneg.mpr (le_of_lt hγ.2)) (le_of_lt hγ.1) (by ring)
  -- Contraction via `C6_contraction_step`:
  obtain ⟨tr, htr_mem, hnorm_lt⟩ := C6_contraction_step x012 x234 x450 hα hβ hγ htq hδ_pos
  exact not_lt_of_ge (hmin y012 hy012_K0 hy012_B tr htr_mem) hnorm_lt

end NeuralCodeOrientedMatroid
