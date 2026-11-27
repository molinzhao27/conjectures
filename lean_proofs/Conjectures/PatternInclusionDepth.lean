import Mathlib

/-!
# Partial Resolution of the Monotone-Measure Conjecture for Pattern Inclusion Depth
(Wei Luo, COLT 2005 / arXiv:2605.30389)

We formalize in Lean 4 + Mathlib (complete kernel-checked proofs, standard foundations only) the main structural
and linear-algebraic theorems from `solutions/pattern_inclusion_depth/solution.md`:

1. **Erasing analogue disproof over every nonempty alphabet (Proposition 6(1))**:
   - `erasing_measure_eq`: `patXX = XX` and `patABC = ABC` both have Luo measure
     `μ(p) = 2 * len(p) - varnum(p) = 3`.
   - `erasing_inclusion_strict`: Over every nonempty alphabet `Σ`, `L_E(XX) ⊂neq L_E(ABC)`.
   - `unary_nonerasing_counterexample`: Over `Unit` (`|Σ| = 1`), `L_NE(ABCD) ⊂neq L_NE(XXY)`
     while `μ(ABCD) = 4 = μ(XXY)` (Proposition 6(4)).

2. **Basic structural constraints and binary letter-surjection descent (Lemma 1)**:
   - `len_le_of_langSubsetNE`: `L_Σ(p) ⊆ L_Σ(q) → len q ≤ len p`.
   - `termOcc_le_of_langSubsetNE`, `termCount_le_of_langSubsetNE`: Over any alphabet with at least
     two distinct letters, `L_Σ(p) ⊆ L_Σ(q)` implies `#_a(q) ≤ #_a(p)` and `t(q) ≤ t(p)`.
   - `binary_surjection_descent`, `binary_surjection_strict_of_len_gt`: Any surjection
     `π : Σ → Bool` (with a section lift) preserves non-erasing inclusion, lengths, variable
     counts, terminal counts, and strictness whenever `len q < len p`.

3. **Morphic-inclusion monotonicity and multiplicity-excess bound (Propositions 1 and 2)**:
   - `morphic_measure_mono`, `morphic_measure_strict_of_len_gt`: For any non-erasing pattern
     morphism `h` with `p = h(q)`, `μ(q) + (len p - len q) ≤ μ(p)`.
   - `min_varOcc_le_of_langSubsetNE`: Over any finite alphabet `|Σ| ≥ 2`, if every variable of `q`
     occurs at least `f` times (`f ≥ 1`) and `L_Σ(p) ⊆ L_Σ(q)`, then every variable of `p` occurs
     at least `f` times.
   - `multiplicity_excess_ineq`: `f * varnum p ≤ f * varnum q + (len p - len q) + excess q f`.
   - `strict_mono_of_small_excess`, `strict_mono_of_const_mult`: Whenever `len q < len p` and
     `excess q f < (2 * f - 1) * (len p - len q)` (in particular for all constant-multiplicity `q`),
     `μ(q) < μ(p)`.

4. **Rado's linear independent representatives and all-lengths gap-one theorem (Lemma 2 & Theorem 2)**:
   - `rado_linear_independent_representatives`: Finite-dimensional linear-matroid form of Rado's
     (1942) theorem on independent transversals.
   - `gap_one_theorem`: Over every alphabet with at least two distinct letters, if `L_Σ(p) ⊆ L_Σ(q)`
     and `len p = len q + 1`, then `varnum p ≤ varnum q + 1`, `defect q ≤ defect p`, and
     `μ(q) < μ(p)` (with no morphic assumption).

The exhaustive finite binary bitmask exclusion for `n(p) ≤ 12` (Theorem 3), the 810-case defect-two
bitmask verification (Proposition 4), and the upper semicomputability discussion (Proposition 5(3))
are documented in `solutions/pattern_inclusion_depth/solution.md` and `proof.pdf`.
-/

open Finset BigOperators

namespace PatternInclusionDepth

/-! ### 1. Core Definitions: Patterns, Parameters, and Pattern Languages -/

/-- A pattern token is either a terminal (`Sum.inl a`) or a variable (`Sum.inr x`). -/
abbrev Token (α V : Type*) := Sum α V

/-- A pattern is a list of tokens. -/
abbrev Pattern (α V : Type*) := List (Token α V)

/-- Extract the variable from a token, if any. -/
def tokenToVar? {α V : Type*} : Token α V → Option V
  | .inl _ => none
  | .inr x => some x

/-- The finite set of distinct variables occurring in a pattern `p`. -/
def vars {α V : Type*} [DecidableEq V] (p : Pattern α V) : Finset V :=
  (p.filterMap tokenToVar?).toFinset

/-- Number of distinct variables occurring in `p`. -/
def varnum {α V : Type*} [DecidableEq V] (p : Pattern α V) : ℕ :=
  (vars p).card

/-- Number of terminal positions in `p`. -/
def termCount {α V : Type*} (p : Pattern α V) : ℕ :=
  p.countP Sum.isLeft

/-- Number of occurrences of terminal `a` in `p`. -/
def termOcc {α V : Type*} [DecidableEq α] (a : α) (p : Pattern α V) : ℕ :=
  p.countP fun t => match t with | .inl b => decide (b = a) | .inr _ => false

/-- Number of occurrences of variable `x` in `p`. -/
def varOcc {α V : Type*} [DecidableEq V] (x : V) (p : Pattern α V) : ℕ :=
  p.countP fun t => match t with | .inl _ => false | .inr y => decide (y = x)

/-- The defect `c(p) = |p| - varnum(p)`. -/
def defect {α V : Type*} [DecidableEq V] (p : Pattern α V) : ℕ :=
  p.length - varnum p

/-- Luo's measure `μ(p) = 2|p| - varnum(p)`. -/
def measure {α V : Type*} [DecidableEq V] (p : Pattern α V) : ℕ :=
  2 * p.length - varnum p

/-- Apply a substitution `σ : V → List Σ` to a pattern `p`. -/
def applySub {α V : Type*} (σ : V → List α) (p : Pattern α V) : List α :=
  p.flatMap fun
    | .inl a => [a]
    | .inr x => σ x

/-- Apply a pattern morphism `h : V → Pattern Σ W` to a pattern `p`. -/
def applyMorph {α V W : Type*} (h : V → Pattern α W) (p : Pattern α V) : Pattern α W :=
  p.flatMap fun
    | .inl a => [.inl a]
    | .inr x => h x

/-- Coordinatewise terminal map `π : Σ → Τ` on a pattern, leaving variables unchanged. -/
def mapTerm {α β V : Type*} (π : α → β) (p : Pattern α V) : Pattern β V :=
  p.map (Sum.map π id)

/-- Non-erasing pattern language membership `w ∈ L_Σ(p)`. -/
def InLangNE {α V : Type*} (p : Pattern α V) (w : List α) : Prop :=
  ∃ σ : V → List α, (∀ x, σ x ≠ []) ∧ applySub σ p = w

/-- Erasing pattern language membership `w ∈ L_{Σ,E}(p)`. -/
def InLangE {α V : Type*} (p : Pattern α V) (w : List α) : Prop :=
  ∃ σ : V → List α, applySub σ p = w

/-- Non-erasing pattern language inclusion `L_Σ(p) ⊆ L_Σ(q)`. -/
def LangSubsetNE {α V₁ V₂ : Type*} (p : Pattern α V₁) (q : Pattern α V₂) : Prop :=
  ∀ w, InLangNE p w → InLangNE q w

/-- Erasing pattern language inclusion `L_{Σ,E}(p) ⊆ L_{Σ,E}(q)`. -/
def LangSubsetE {α V₁ V₂ : Type*} (p : Pattern α V₁) (q : Pattern α V₂) : Prop :=
  ∀ w, InLangE p w → InLangE q w

/-! ### Basic Combinatorial Lemmas on Patterns -/

lemma mem_vars_iff {α V : Type*} [DecidableEq V] {p : Pattern α V} {x : V} :
    x ∈ vars p ↔ Sum.inr x ∈ p := by
  simp only [vars, List.mem_toFinset, List.mem_filterMap, tokenToVar?]
  constructor
  · rintro ⟨t, ht, htx⟩
    cases t with
    | inl a => simp at htx
    | inr y =>
      simp only [Option.some.injEq] at htx
      simpa [htx] using ht
  · intro hx
    exact ⟨.inr x, hx, rfl⟩

lemma varOcc_pos_iff {α V : Type*} [DecidableEq V] {p : Pattern α V} {x : V} :
    0 < varOcc x p ↔ x ∈ vars p := by
  rw [varOcc, List.countP_pos_iff, mem_vars_iff]
  constructor
  · rintro ⟨t, ht, h⟩
    cases t with
    | inl a => simp at h
    | inr y =>
      simp only [decide_eq_true_eq] at h
      simpa [h] using ht
  · intro hx
    exact ⟨.inr x, hx, by simp⟩

lemma varOcc_eq_zero_of_not_mem {α V : Type*} [DecidableEq V] {p : Pattern α V} {x : V}
    (hx : x ∉ vars p) : varOcc x p = 0 := by
  have h := mt varOcc_pos_iff.mp hx
  omega

lemma vars_nil {α V : Type*} [DecidableEq V] : vars ([] : Pattern α V) = ∅ := by
  simp [vars]

lemma vars_cons_inl {α V : Type*} [DecidableEq V] (a : α) (p : Pattern α V) :
    vars (.inl a :: p) = vars p := by
  ext x
  simp [mem_vars_iff]

lemma vars_cons_inr {α V : Type*} [DecidableEq V] (y : V) (p : Pattern α V) :
    vars (.inr y :: p) = insert y (vars p) := by
  ext x
  simp [mem_vars_iff, eq_comm]

lemma vars_append {α V : Type*} [DecidableEq V] (p₁ p₂ : Pattern α V) :
    vars (p₁ ++ p₂) = vars p₁ ∪ vars p₂ := by
  ext x
  simp [mem_vars_iff]

lemma varnum_le_length {α V : Type*} [DecidableEq V] (p : Pattern α V) :
    varnum p ≤ p.length := by
  induction p with
  | nil => simp [varnum, vars_nil]
  | cons t ps ih =>
    cases t with
    | inl a =>
      simp only [varnum, vars_cons_inl, List.length_cons] at ih ⊢
      omega
    | inr x =>
      simp only [varnum, vars_cons_inr, List.length_cons] at ih ⊢
      have hcard : (insert x (vars ps)).card ≤ (vars ps).card + 1 := Finset.card_insert_le _ _
      exact le_trans hcard (Nat.add_le_add_right ih 1)

lemma length_eq_termCount_add_sum_varOcc {α V : Type*} [DecidableEq V] (p : Pattern α V) :
    p.length = termCount p + ∑ x ∈ vars p, varOcc x p := by
  induction p with
  | nil => simp [termCount, vars_nil]
  | cons t ps ih =>
    cases t with
    | inl a =>
      simp only [List.length_cons, termCount, List.countP_cons, Sum.isLeft_inl,
        vars_cons_inl, ite_true] at ih ⊢
      have hsum : ∑ x ∈ vars ps, varOcc x (.inl a :: ps) = ∑ x ∈ vars ps, varOcc x ps := by
        refine Finset.sum_congr rfl fun x _ => ?_
        simp [varOcc]
      rw [hsum]
      omega
    | inr y =>
      simp only [List.length_cons, termCount, List.countP_cons, Sum.isLeft_inr,
        Bool.false_eq_true, ite_false, Nat.add_zero, vars_cons_inr] at ih ⊢
      have hsplit : ∑ x ∈ insert y (vars ps), varOcc x (.inr y :: ps) =
          1 + ∑ x ∈ vars ps, varOcc x ps := by
        by_cases hy : y ∈ vars ps
        · rw [Finset.insert_eq_of_mem hy]
          have hstep : ∑ x ∈ vars ps, varOcc x (.inr y :: ps) =
              ∑ x ∈ vars ps, (varOcc x ps + if y = x then 1 else 0) := by
            refine Finset.sum_congr rfl fun x _ => ?_
            by_cases h : y = x <;> simp [varOcc, h]
          rw [hstep, Finset.sum_add_distrib]
          simp [hy]
          omega
        · rw [Finset.sum_insert hy]
          have hy0 : varOcc y (.inr y :: ps) = 1 := by
            have := varOcc_eq_zero_of_not_mem hy
            simp [varOcc] at this ⊢
            exact this
          have hrest : ∑ x ∈ vars ps, varOcc x (.inr y :: ps) = ∑ x ∈ vars ps, varOcc x ps := by
            refine Finset.sum_congr rfl fun x hx => ?_
            have hne : y ≠ x := fun h => hy (h ▸ hx)
            simp [varOcc, hne]
          rw [hy0, hrest]
      rw [hsplit]
      omega

lemma applySub_length {α V : Type*} [DecidableEq V] (σ : V → List α) (p : Pattern α V) :
    (applySub σ p).length = termCount p + ∑ x ∈ vars p, varOcc x p * (σ x).length := by
  induction p with
  | nil => simp [applySub, termCount, vars_nil]
  | cons t ps ih =>
    cases t with
    | inl a =>
      simp only [applySub, List.flatMap_cons, List.length_append, List.length_singleton,
        termCount, List.countP_cons, Sum.isLeft_inl, ite_true, vars_cons_inl] at ih ⊢
      have hsum : ∑ x ∈ vars ps, varOcc x (.inl a :: ps) * (σ x).length =
          ∑ x ∈ vars ps, varOcc x ps * (σ x).length := by
        refine Finset.sum_congr rfl fun x _ => ?_
        simp [varOcc]
      rw [hsum]
      omega
    | inr y =>
      simp only [applySub, List.flatMap_cons, List.length_append, termCount, List.countP_cons,
        Sum.isLeft_inr, Bool.false_eq_true, ite_false, Nat.add_zero, vars_cons_inr] at ih ⊢
      have hsplit : ∑ x ∈ insert y (vars ps), varOcc x (.inr y :: ps) * (σ x).length =
          (σ y).length + ∑ x ∈ vars ps, varOcc x ps * (σ x).length := by
        by_cases hy : y ∈ vars ps
        · rw [Finset.insert_eq_of_mem hy]
          have hstep : ∑ x ∈ vars ps, varOcc x (.inr y :: ps) * (σ x).length =
              ∑ x ∈ vars ps, (varOcc x ps * (σ x).length + if y = x then (σ x).length else 0) := by
            refine Finset.sum_congr rfl fun x _ => ?_
            by_cases h : y = x <;> simp [varOcc, h, Nat.add_mul]
          rw [hstep, Finset.sum_add_distrib]
          simp [hy]
          omega
        · rw [Finset.sum_insert hy]
          have hy1 : varOcc y (.inr y :: ps) = 1 := by
            have := varOcc_eq_zero_of_not_mem hy
            simp [varOcc] at this ⊢
            exact this
          have hrest : ∑ x ∈ vars ps, varOcc x (.inr y :: ps) * (σ x).length =
              ∑ x ∈ vars ps, varOcc x ps * (σ x).length := by
            refine Finset.sum_congr rfl fun x hx => ?_
            have hne : y ≠ x := fun h => hy (h ▸ hx)
            simp [varOcc, hne]
          rw [hy1, one_mul, hrest]
      rw [hsplit]
      omega

lemma applySub_length_ge {α V : Type*} {σ : V → List α} (hσ : ∀ x, σ x ≠ [])
    (p : Pattern α V) : p.length ≤ (applySub σ p).length := by
  induction p with
  | nil => simp [applySub]
  | cons t ps ih =>
    cases t with
    | inl a =>
      simp only [applySub, List.flatMap_cons, List.length_append, List.length_cons] at ih ⊢
      omega
    | inr x =>
      simp only [applySub, List.flatMap_cons, List.length_append, List.length_cons] at ih ⊢
      have hx : 1 ≤ (σ x).length := List.length_pos_iff.mpr (hσ x)
      omega

lemma applySub_single_length {α V : Type*} (a : α) (p : Pattern α V) :
    (applySub (fun _ => [a]) p).length = p.length := by
  induction p with
  | nil => simp [applySub]
  | cons t ps ih =>
    have hstep : (applySub (fun _ => [a]) (t :: ps)).length =
        (applySub (fun _ => [a]) ps).length + 1 := by
      cases t <;> simp [applySub]
    simp only [hstep, ih, List.length_cons]

lemma applySub_append {α V : Type*} (σ : V → List α) (p₁ p₂ : Pattern α V) :
    applySub σ (p₁ ++ p₂) = applySub σ p₁ ++ applySub σ p₂ := by
  simp [applySub]

lemma applySub_congr {α V : Type*} [DecidableEq V] {σ₁ σ₂ : V → List α} {p : Pattern α V}
    (h : ∀ x ∈ vars p, σ₁ x = σ₂ x) : applySub σ₁ p = applySub σ₂ p := by
  induction p with
  | nil => simp [applySub]
  | cons t ps ih =>
    cases t with
    | inl a =>
      simp only [applySub, List.flatMap_cons]
      congr 1
      exact ih fun x hx => h x (by simpa [vars_cons_inl] using hx)
    | inr y =>
      simp only [applySub, List.flatMap_cons]
      rw [h y (by simp [vars_cons_inr])]
      congr 1
      exact ih fun x hx => h x (by simp [vars_cons_inr, hx])

lemma len_le_of_langSubsetNE {α V₁ V₂ : Type*} [Nonempty α]
    {p : Pattern α V₁} {q : Pattern α V₂} (hsub : LangSubsetNE p q) :
    q.length ≤ p.length := by
  obtain ⟨a⟩ := ‹Nonempty α›
  have hp : InLangNE p (applySub (fun _ => [a]) p) :=
    ⟨fun _ => [a], fun _ => List.cons_ne_nil _ _, rfl⟩
  obtain ⟨τ, hτ, hw⟩ := hsub _ hp
  have h1 := applySub_length_ge hτ q
  rw [hw, applySub_single_length] at h1
  exact h1

lemma count_le_applySub {α V : Type*} [DecidableEq α] (a : α) (σ : V → List α)
    (p : Pattern α V) : termOcc a p ≤ (applySub σ p).count a := by
  induction p with
  | nil => simp [termOcc, applySub]
  | cons t ps ih =>
    cases t with
    | inl b =>
      simp only [termOcc, List.countP_cons, applySub, List.flatMap_cons, List.count_append,
        List.count_singleton] at ih ⊢
      by_cases h : b = a <;> simp [h] <;> omega
    | inr x =>
      simp only [termOcc, List.countP_cons, applySub, List.flatMap_cons, List.count_append,
        Bool.false_eq_true, ite_false, Nat.add_zero] at ih ⊢
      omega

lemma count_applySub_const_ne {α V : Type*} [DecidableEq α] {a b : α} (hne : b ≠ a)
    (p : Pattern α V) : (applySub (fun _ => [b]) p).count a = termOcc a p := by
  induction p with
  | nil => simp [termOcc, applySub]
  | cons t ps ih =>
    cases t with
    | inl c =>
      simp only [termOcc, List.countP_cons, applySub, List.flatMap_cons, List.count_append,
        List.count_singleton] at ih ⊢
      by_cases h : c = a <;> simp [h] <;> omega
    | inr x =>
      simp only [termOcc, List.countP_cons, applySub, List.flatMap_cons, List.count_append,
        List.count_singleton, Bool.false_eq_true, ite_false, Nat.add_zero] at ih ⊢
      simp [hne, ih]

/-- **Lemma 1(2) (Terminal occurrence lower bound)**: Over any alphabet with at least two
distinct letters, `L_α(p) ⊆ L_α(q)` implies `#_a(q) ≤ #_a(p)` for every terminal `a ∈ α`. -/
theorem termOcc_le_of_langSubsetNE {α V₁ V₂ : Type*} [DecidableEq α]
    (h2 : ∀ a : α, ∃ b : α, b ≠ a)
    {p : Pattern α V₁} {q : Pattern α V₂} (hsub : LangSubsetNE p q) (a : α) :
    termOcc a q ≤ termOcc a p := by
  obtain ⟨b, hne⟩ := h2 a
  have hp : InLangNE p (applySub (fun _ => [b]) p) :=
    ⟨fun _ => [b], fun _ => List.cons_ne_nil _ _, rfl⟩
  obtain ⟨τ, _, hw⟩ := hsub _ hp
  have h1 := count_le_applySub a τ q
  rw [hw, count_applySub_const_ne hne p] at h1
  exact h1

lemma sum_termOcc_eq_termCount {α V : Type*} [Fintype α] [DecidableEq α] (p : Pattern α V) :
    ∑ a : α, termOcc a p = termCount p := by
  induction p with
  | nil => simp [termOcc, termCount]
  | cons t ps ih =>
    cases t with
    | inl c =>
      simp only [termOcc, List.countP_cons, termCount, Sum.isLeft_inl, ite_true] at ih ⊢
      have hstep : ∑ a : α, (List.countP
            (fun t : Token α V => match t with | .inl b => decide (b = a) | .inr _ => false) ps +
            if decide (c = a) = true then 1 else 0) =
          (∑ a : α, List.countP
            (fun t : Token α V => match t with | .inl b => decide (b = a) | .inr _ => false) ps) + 1 := by
        rw [Finset.sum_add_distrib]
        simp
      rw [hstep, ih]
    | inr x =>
      simp only [termOcc, List.countP_cons, termCount, Sum.isLeft_inr, Bool.false_eq_true,
        ite_false, Nat.add_zero] at ih ⊢
      exact ih

/-- **Lemma 1(2) (Total terminal count lower bound)**: Over any finite alphabet with `|Σ| ≥ 2`,
`L_α(p) ⊆ L_α(q)` implies `t(q) ≤ t(p)`. -/
theorem termCount_le_of_langSubsetNE {α V₁ V₂ : Type*} [Fintype α] [DecidableEq α]
    (hcard : 2 ≤ Fintype.card α)
    {p : Pattern α V₁} {q : Pattern α V₂} (hsub : LangSubsetNE p q) :
    termCount q ≤ termCount p := by
  have h2 : ∀ a : α, ∃ b : α, b ≠ a := by
    intro a
    by_contra hall
    push_neg at hall
    have hle : Fintype.card α ≤ 1 := by
      have : Subsingleton α := ⟨fun x y => (hall x).trans (hall y).symm⟩
      exact Fintype.card_le_one_iff_subsingleton.mpr this
    omega
  rw [← sum_termOcc_eq_termCount q, ← sum_termOcc_eq_termCount p]
  exact Finset.sum_le_sum fun a _ => termOcc_le_of_langSubsetNE h2 hsub a

/-! ### 2. Erasing Analogue Disproof & Unary Control (Proposition 6) -/

/-- Pattern `XX` over any alphabet `Σ`. -/
def patXX (α : Type*) : Pattern α ℕ := [.inr 0, .inr 0]

/-- Pattern `ABC` over any alphabet `Σ`. -/
def patABC (α : Type*) : Pattern α ℕ := [.inr 0, .inr 1, .inr 2]

/-- **Proposition 6(1) (Measure equality)**: `μ(XX) = 3 = μ(ABC)`. -/
theorem erasing_measure_eq (α : Type*) :
    measure (patXX α) = 3 ∧ measure (patABC α) = 3 := by
  constructor
  · have hvars : vars (patXX α) = {0} := by
      ext x; simp [patXX, mem_vars_iff]
    have hlen : (patXX α).length = 2 := rfl
    rw [measure, varnum, hvars, hlen, Finset.card_singleton]
  · have hvars : vars (patABC α) = {0, 1, 2} := by
      ext x; simp [patABC, mem_vars_iff]
    have hlen : (patABC α).length = 3 := rfl
    rw [measure, varnum, hvars, hlen]
    decide

/-- **Proposition 6(1) (Erasing analogue disproof)**: Over every nonempty alphabet `Σ`,
`L_{α,E}(XX) ⊂neq L_{α,E}(ABC)` while `μ(XX) = 3 = μ(ABC)`. -/
theorem erasing_inclusion_strict (α : Type*) [Nonempty α] :
    LangSubsetE (patXX α) (patABC α) ∧ ¬ LangSubsetE (patABC α) (patXX α) := by
  constructor
  · rintro w ⟨σ, rfl⟩
    refine ⟨fun n => match n with | 0 => σ 0 | 1 => σ 0 | _ => [], ?_⟩
    simp [patXX, patABC, applySub]
  · intro hsub
    obtain ⟨a⟩ := ‹Nonempty α›
    have habc : InLangE (patABC α) [a] := by
      refine ⟨fun n => match n with | 0 => [a] | _ => [], ?_⟩
      simp [patABC, applySub]
    obtain ⟨σ, hw⟩ := hsub [a] habc
    simp only [patXX, applySub, List.flatMap_cons, List.flatMap_nil, List.append_nil] at hw
    have hlen := congrArg List.length hw
    simp only [List.length_append, List.length_singleton] at hlen
    omega

/-- **Proposition 6(4) (Unary non-erasing counterexample)**: Over a 1-letter alphabet `Unit`,
`L_{Unit}(ABCD) ⊂neq L_{Unit}(XXY)` while `μ(ABCD) = 4 = μ(XXY)`. -/
theorem unary_nonerasing_counterexample :
    let p : Pattern Unit ℕ := [.inr 0, .inr 1, .inr 2, .inr 3]
    let q : Pattern Unit ℕ := [.inr 0, .inr 0, .inr 1]
    measure p = 4 ∧ measure q = 4 ∧ LangSubsetNE p q ∧ ¬ LangSubsetNE q p := by
  dsimp only
  have hpv : vars ([.inr 0, .inr 1, .inr 2, .inr 3] : Pattern Unit ℕ) = {0, 1, 2, 3} := by
    ext x; simp [mem_vars_iff]
  have hqv : vars ([.inr 0, .inr 0, .inr 1] : Pattern Unit ℕ) = {0, 1} := by
    ext x; simp [mem_vars_iff]
  refine ⟨by simp [measure, varnum, hpv], by simp [measure, varnum, hqv], ?_, ?_⟩
  · rintro w ⟨σ, hσ, rfl⟩
    have hlen : 4 ≤ (applySub σ ([.inr 0, .inr 1, .inr 2, .inr 3] : Pattern Unit ℕ)).length :=
      applySub_length_ge hσ ([.inr 0, .inr 1, .inr 2, .inr 3] : Pattern Unit ℕ)
    set k := (applySub σ [.inr 0, .inr 1, .inr 2, .inr 3]).length
    have heq : applySub σ [.inr 0, .inr 1, .inr 2, .inr 3] = List.replicate k () := by
      apply List.ext_getElem (by simp [k])
      intro i _ _
      simp
    refine ⟨fun n => match n with | 0 => [()] | _ => List.replicate (k - 2) (), ?_, ?_⟩
    · intro n
      cases n with
      | zero => exact List.cons_ne_nil _ _
      | succ m =>
        simp only [ne_eq, List.replicate_eq_nil_iff]
        omega
    · rw [heq]
      simp only [applySub, List.flatMap_cons, List.flatMap_nil, List.append_nil]
      apply List.ext_getElem
      · simp; omega
      · intro i _ _; simp
  · intro hsub
    have hq : InLangNE ([.inr 0, .inr 0, .inr 1] : Pattern Unit ℕ) [(), (), ()] :=
      ⟨fun _ => [()], fun _ => List.cons_ne_nil _ _, by simp [applySub]⟩
    obtain ⟨σ, hσ, hw⟩ := hsub _ hq
    have hlen := applySub_length_ge hσ ([.inr 0, .inr 1, .inr 2, .inr 3] : Pattern Unit ℕ)
    rw [hw] at hlen
    simp at hlen

/-! ### 3. Binary Letter-Surjection Alphabet Reduction (Lemma 1(3)) -/

lemma mapTerm_length {α β V : Type*} (π : α → β) (p : Pattern α V) :
    (mapTerm π p).length = p.length := by
  simp [mapTerm]

lemma mapTerm_vars {α β V : Type*} [DecidableEq V] (π : α → β) (p : Pattern α V) :
    vars (mapTerm π p) = vars p := by
  ext x
  simp only [mem_vars_iff, mapTerm, List.mem_map]
  constructor
  · rintro ⟨t, ht, htx⟩
    cases t with
    | inl a => simp at htx
    | inr y =>
      simp only [Sum.map_inr, id_eq, Sum.inr.injEq] at htx
      simpa [htx] using ht
  · intro hx
    exact ⟨.inr x, hx, rfl⟩

lemma mapTerm_varnum {α β V : Type*} [DecidableEq V] (π : α → β) (p : Pattern α V) :
    varnum (mapTerm π p) = varnum p := by
  simp [varnum, mapTerm_vars]

lemma mapTerm_termCount {α β V : Type*} (π : α → β) (p : Pattern α V) :
    termCount (mapTerm π p) = termCount p := by
  induction p with
  | nil => simp [termCount, mapTerm]
  | cons t ps ih =>
    cases t <;>
    · simp only [termCount, mapTerm, List.map_cons, Sum.map_inl, Sum.map_inr, id_eq,
        List.countP_cons, Sum.isLeft_inl, Sum.isLeft_inr] at ih ⊢
      omega

lemma mapTerm_measure {α β V : Type*} [DecidableEq V] (π : α → β) (p : Pattern α V) :
    measure (mapTerm π p) = measure p := by
  simp [measure, mapTerm_length, mapTerm_varnum]

lemma applySub_mapTerm {α β V : Type*} (π : α → β) (σ : V → List α) (p : Pattern α V) :
    (applySub σ p).map π = applySub (fun x => (σ x).map π) (mapTerm π p) := by
  induction p with
  | nil => simp [applySub, mapTerm]
  | cons t ps ih =>
    cases t <;>
    · simp only [applySub, mapTerm, List.map_cons, Sum.map_inl, Sum.map_inr, id_eq,
        List.flatMap_cons, List.map_append, List.map_nil] at ih ⊢
      rw [ih]

/-- **Lemma 1(3) (Binary surjection descent)**: For any surjection `π : Σ → Τ` (in particular
`β = Bool`), `L_α(p) ⊆ L_α(q)` implies `L_β(π p) ⊆ L_β(π q)`. -/
theorem binary_surjection_descent {α β V₁ V₂ : Type*} (π : α → β) (hπ : Function.Surjective π)
    {p : Pattern α V₁} {q : Pattern α V₂} (hsub : LangSubsetNE p q) :
    LangSubsetNE (mapTerm π p) (mapTerm π q) := by
  classical
  let ι : β → α := Function.surjInv hπ
  have hπι : ∀ b, π (ι b) = b := Function.surjInv_eq hπ
  rintro w ⟨σ, hσ, rfl⟩
  let σ' : V₁ → List α := fun x => (σ x).map ι
  have hσ' : ∀ x, σ' x ≠ [] := fun x => by
    simp only [σ', ne_eq, List.map_eq_nil_iff]
    exact hσ x
  obtain ⟨τ, hτ, hw⟩ := hsub (applySub σ' p) ⟨σ', hσ', rfl⟩
  refine ⟨fun x => (τ x).map π, fun x => by simp [hτ x], ?_⟩
  rw [← applySub_mapTerm π τ q, hw, applySub_mapTerm π σ' p]
  have hcomp : ∀ x, (σ' x).map π = σ x := fun x => by
    simp [σ', List.map_map, Function.comp_def, hπι]
  simp only [hcomp]

/-- **Lemma 1(3) (Strictness preservation for positive length gap)**: Whenever `len q < len p`,
`L_β(π p) ⊂neq L_β(π q)` remains strict. -/
theorem binary_surjection_strict_of_len_gt {α β V₁ V₂ : Type*} [Nonempty β]
    (π : α → β) {p : Pattern α V₁} {q : Pattern α V₂} (hgt : q.length < p.length) :
    ¬ LangSubsetNE (mapTerm π q) (mapTerm π p) := by
  intro hrev
  have hle := len_le_of_langSubsetNE hrev
  rw [mapTerm_length, mapTerm_length] at hle
  omega

/-! ### 4. Morphic-Inclusion Monotonicity (Proposition 1) -/

lemma applyMorph_length {α V W : Type*} [DecidableEq V]
    (h : V → Pattern α W) (q : Pattern α V) :
    (applyMorph h q).length = termCount q + ∑ x ∈ vars q, varOcc x q * (h x).length := by
  induction q with
  | nil => simp [applyMorph, termCount, vars_nil]
  | cons t qs ih =>
    cases t with
    | inl a =>
      simp only [applyMorph, List.flatMap_cons, List.length_append, List.length_singleton,
        termCount, List.countP_cons, Sum.isLeft_inl, ite_true, vars_cons_inl]
      have hsum : ∑ x ∈ vars qs, varOcc x (.inl a :: qs) * (h x).length =
          ∑ x ∈ vars qs, varOcc x qs * (h x).length := by
        refine Finset.sum_congr rfl fun x _ => ?_
        simp [varOcc]
      rw [hsum]
      simp only [applyMorph, termCount] at ih
      omega
    | inr y =>
      simp only [applyMorph, List.flatMap_cons, List.length_append, termCount, List.countP_cons,
        Sum.isLeft_inr, Bool.false_eq_true, ite_false, Nat.add_zero, vars_cons_inr]
      have hsplit : ∑ x ∈ insert y (vars qs), varOcc x (.inr y :: qs) * (h x).length =
          (h y).length + ∑ x ∈ vars qs, varOcc x qs * (h x).length := by
        by_cases hy : y ∈ vars qs
        · rw [Finset.insert_eq_of_mem hy]
          have hstep : ∑ x ∈ vars qs, varOcc x (.inr y :: qs) * (h x).length =
              ∑ x ∈ vars qs, (varOcc x qs * (h x).length + if y = x then (h x).length else 0) := by
            refine Finset.sum_congr rfl fun x _ => ?_
            by_cases heq : y = x <;> simp [varOcc, heq, Nat.add_mul]
          rw [hstep, Finset.sum_add_distrib]
          simp [hy]
          omega
        · rw [Finset.sum_insert hy]
          have hy1 : varOcc y (.inr y :: qs) = 1 := by
            have := varOcc_eq_zero_of_not_mem hy
            simp [varOcc] at this ⊢
            exact this
          have hrest : ∑ x ∈ vars qs, varOcc x (.inr y :: qs) * (h x).length =
              ∑ x ∈ vars qs, varOcc x qs * (h x).length := by
            refine Finset.sum_congr rfl fun x hx => ?_
            have hne : y ≠ x := fun heq => hy (heq ▸ hx)
            simp [varOcc, hne]
          rw [hy1, one_mul, hrest]
      rw [hsplit]
      simp only [applyMorph, termCount] at ih
      omega

lemma vars_applyMorph_subset {α V W : Type*} [DecidableEq V] [DecidableEq W]
    (h : V → Pattern α W) (q : Pattern α V) :
    vars (applyMorph h q) ⊆ (vars q).biUnion (fun x => vars (h x)) := by
  intro w hw
  rw [mem_vars_iff, applyMorph, List.mem_flatMap] at hw
  obtain ⟨t, ht, htw⟩ := hw
  cases t with
  | inl a => simp at htw
  | inr x =>
    rw [Finset.mem_biUnion]
    exact ⟨x, mem_vars_iff.mpr ht, mem_vars_iff.mpr htw⟩

/-- **Proposition 1 (Morphic-inclusion monotonicity)**: If `p = applyMorph h q` for a
non-erasing morphism `h` (`∀ x ∈ vars q, 1 ≤ (h x).length`), then `q.length ≤ p.length` and
`μ(q) + (p.length - q.length) ≤ μ(p)`. -/
theorem morphic_measure_mono {α V W : Type*} [DecidableEq V] [DecidableEq W]
    (h : V → Pattern α W) {p : Pattern α W} {q : Pattern α V}
    (hne : ∀ x ∈ vars q, 1 ≤ (h x).length) (hp : p = applyMorph h q) :
    q.length ≤ p.length ∧ measure q + (p.length - q.length) ≤ measure p := by
  subst hp
  have hlen_q := length_eq_termCount_add_sum_varOcc q
  have hlen_p := applyMorph_length h q
  have hsum_le : ∑ x ∈ vars q, varOcc x q ≤ ∑ x ∈ vars q, varOcc x q * (h x).length := by
    refine Finset.sum_le_sum fun x hx => ?_
    have := hne x hx
    exact Nat.le_mul_of_pos_right _ (by omega)
  have hlen_le : q.length ≤ (applyMorph h q).length := by omega
  have hvar_bound : varnum (applyMorph h q) ≤ ∑ x ∈ vars q, (h x).length := by
    calc varnum (applyMorph h q)
        ≤ ((vars q).biUnion (fun x => vars (h x))).card :=
          Finset.card_le_card (vars_applyMorph_subset h q)
      _ ≤ ∑ x ∈ vars q, (vars (h x)).card := Finset.card_biUnion_le
      _ ≤ ∑ x ∈ vars q, (h x).length := Finset.sum_le_sum fun x _ => varnum_le_length (h x)
  have hdiff_sum : ∑ x ∈ vars q, ((h x).length - 1) ≤
      ∑ x ∈ vars q, varOcc x q * ((h x).length - 1) := by
    refine Finset.sum_le_sum fun x hx => ?_
    have hpos : 1 ≤ varOcc x q := varOcc_pos_iff.mpr hx
    exact Nat.le_mul_of_pos_left _ hpos
  have hsum_split1 : ∑ x ∈ vars q, (h x).length =
      varnum q + ∑ x ∈ vars q, ((h x).length - 1) := by
    have : ∑ x ∈ vars q, (h x).length = ∑ x ∈ vars q, (1 + ((h x).length - 1)) := by
      refine Finset.sum_congr rfl fun x hx => ?_
      have := hne x hx
      omega
    rw [this, Finset.sum_add_distrib, Finset.sum_const, smul_eq_mul, mul_one, varnum]
  have hsum_split2 : ∑ x ∈ vars q, varOcc x q * (h x).length =
      (∑ x ∈ vars q, varOcc x q) + ∑ x ∈ vars q, varOcc x q * ((h x).length - 1) := by
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun x hx => ?_
    have := hne x hx
    have hmul : varOcc x q * (1 + ((h x).length - 1)) =
        varOcc x q + varOcc x q * ((h x).length - 1) := by ring
    have heq : 1 + ((h x).length - 1) = (h x).length := by omega
    conv_lhs => rw [← heq]
    exact hmul
  have hvp_le : varnum (applyMorph h q) ≤ varnum q + ((applyMorph h q).length - q.length) := by
    omega
  have hvq_le := varnum_le_length q
  have hvp_len := varnum_le_length (applyMorph h q)
  refine ⟨hlen_le, ?_⟩
  simp only [measure]
  omega

/-- Strict morphic monotonicity whenever `len q < len p`. -/
theorem morphic_measure_strict_of_len_gt {α V W : Type*} [DecidableEq V] [DecidableEq W]
    (h : V → Pattern α W) {p : Pattern α W} {q : Pattern α V}
    (hne : ∀ x ∈ vars q, 1 ≤ (h x).length) (hp : p = applyMorph h q)
    (hgt : q.length < p.length) : measure q < measure p := by
  have ⟨_, hmono⟩ := morphic_measure_mono h hne hp
  omega

/-! ### 5. Multiplicity-Excess Inequality (Proposition 2) -/

/-- Total multiplicity excess `E(q) = |q| - t(q) - f * varnum(q)`. -/
def excess {α V : Type*} [DecidableEq V] (q : Pattern α V) (f : ℕ) : ℕ :=
  q.length - termCount q - f * varnum q

/-- First-occurrence decomposition of a pattern containing variable `y`. -/
lemma split_first_occ {α V : Type*} [DecidableEq V] {p : Pattern α V} {y : V}
    (hy : y ∈ vars p) :
    ∃ p₁ p₂ : Pattern α V, p = p₁ ++ [.inr y] ++ p₂ ∧ y ∉ vars p₁ := by
  induction p with
  | nil => simp [vars_nil] at hy
  | cons t ps ih =>
    cases t with
    | inl a =>
      rw [vars_cons_inl] at hy
      obtain ⟨p₁, p₂, heq, hnot⟩ := ih hy
      exact ⟨.inl a :: p₁, p₂, by simp [heq], by simpa [vars_cons_inl] using hnot⟩
    | inr x =>
      by_cases hxy : x = y
      · subst hxy
        exact ⟨[], ps, by simp, by simp [vars_nil]⟩
      · rw [vars_cons_inr, Finset.mem_insert] at hy
        have hy' : y ∈ vars ps := hy.resolve_left (Ne.symm hxy)
        obtain ⟨p₁, p₂, heq, hnot⟩ := ih hy'
        refine ⟨.inr x :: p₁, p₂, by simp [heq], ?_⟩
        simp [vars_cons_inr, Ne.symm hxy, hnot]

/-- Exponential growth eventually beats any polynomial `((A * t + B)^m) < 2^(t - C)`. -/
lemma exists_pow_lt_two_pow (m A B C : ℕ) :
    ∃ t : ℕ, C + 1 ≤ t ∧ (A * t + B) ^ m < 2 ^ (t - C) := by
  have htend : Filter.Tendsto (fun n : ℕ => ((A * (n + C + 1) + B : ℝ) ^ m) / (2 : ℝ) ^ n)
      Filter.atTop (nhds 0) := by
    have hpoly : ∀ (D : ℝ), Filter.Tendsto (fun n : ℕ => ((n : ℝ) + D) ^ m / (2 : ℝ) ^ n)
        Filter.atTop (nhds 0) := by
      intro D
      have h1 : Filter.Tendsto (fun n : ℕ => ((n : ℝ) ^ m) / (2 : ℝ) ^ n)
          Filter.atTop (nhds 0) := by
        simpa [Real.rpow_natCast] using
          tendsto_pow_const_div_const_pow_of_one_lt m (by norm_num : (1 : ℝ) < 2)
      have h2 : Filter.Tendsto (fun n : ℕ => ((n : ℝ) + D) ^ m / (2 : ℝ) ^ n)
          Filter.atTop (nhds 0) := by
        -- Compare `(n + D)^m ≤ (2 * n)^m = 2^m * n^m` for `n ≥ |D|`.
        refine squeeze_zero_norm' ?_ (by simpa using h1.const_mul ((2 : ℝ) ^ m))
        filter_upwards [Filter.eventually_ge_atTop (Nat.ceil |D| + 1)] with n hn
        have hnD : |D| ≤ (n : ℝ) := by
          have : (Nat.ceil |D| : ℝ) ≤ (n : ℝ) := by exact_mod_cast (by omega : Nat.ceil |D| ≤ n)
          exact le_trans (Nat.le_ceil _) this
        have hpos2 : (0 : ℝ) < (2 : ℝ) ^ n := by positivity
        rw [Real.norm_eq_abs, abs_div, abs_of_pos hpos2, abs_pow]
        have habs : |(n : ℝ) + D| ≤ 2 * (n : ℝ) := by
          have htr := abs_add_le (n : ℝ) D
          have hn0 : (0 : ℝ) ≤ n := Nat.cast_nonneg n
          rw [abs_of_nonneg hn0] at htr
          linarith
        calc |(n : ℝ) + D| ^ m / (2 : ℝ) ^ n
            ≤ (2 * (n : ℝ)) ^ m / (2 : ℝ) ^ n := by
              gcongr
          _ = (2 : ℝ) ^ m * ((n : ℝ) ^ m / (2 : ℝ) ^ n) := by
              rw [mul_pow, mul_div_assoc]
      exact h2
    by_cases hA : A = 0
    · simp only [hA, Nat.cast_zero, zero_mul, zero_add]
      have hconst : Filter.Tendsto (fun n : ℕ => (B : ℝ) ^ m * ((1 / 2 : ℝ) ^ n))
          Filter.atTop (nhds ((B : ℝ) ^ m * 0)) :=
        (tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num) (by norm_num)).const_mul _
      simp only [mul_zero] at hconst
      refine hconst.congr fun n => ?_
      rw [one_div, inv_pow, div_eq_mul_inv]
    · have hApos : (0 : ℝ) < A := by exact_mod_cast Nat.pos_of_ne_zero hA
      have hscaled := (hpoly ((C + 1 : ℝ) + (B : ℝ) / (A : ℝ))).const_mul ((A : ℝ) ^ m)
      simp only [mul_zero] at hscaled
      refine hscaled.congr fun n => ?_
      have hEq : (A : ℝ) * ((n : ℝ) + ((C + 1 : ℝ) + (B : ℝ) / (A : ℝ))) =
          (A : ℝ) * ((n : ℝ) + (C : ℝ) + 1) + (B : ℝ) := by
        field_simp [ne_of_gt hApos]
        ring
      rw [← mul_div_assoc, ← mul_pow, hEq]
  have hev := (tendsto_order.mp htend).2 1 (by norm_num)
  rw [Filter.eventually_atTop] at hev
  obtain ⟨n, hn⟩ := hev
  have hlt := hn n (le_refl n)
  have hpos : (0 : ℝ) < (2 : ℝ) ^ n := by positivity
  rw [div_lt_one hpos] at hlt
  refine ⟨n + C + 1, by omega, ?_⟩
  have hsub : n + C + 1 - C = n + 1 := by omega
  rw [hsub]
  have hnat : (A * (n + C + 1) + B) ^ m < 2 ^ n := by
    have hlt' : ((A * (n + C + 1) + B : ℕ) : ℝ) ^ m < (2 : ℝ) ^ n := by
      push_cast
      simpa [add_assoc] using hlt
    exact_mod_cast hlt'
  have hle2 : 2 ^ n ≤ 2 ^ (n + 1) := Nat.pow_le_pow_right (by omega) (by omega)
  omega

/-- Helper to split a buffer list `buf` into slices of lengths specified by `lens`. -/
def splitByLens {α : Type*} : List ℕ → List α → List (List α)
  | [], _ => []
  | k :: ks, buf => buf.take k :: splitByLens ks (buf.drop k)

lemma splitByLens_flatMap_append {α : Type*} (ls : List (List α)) (tail : List α) :
    splitByLens (ls.map List.length) (ls.flatten ++ tail) = ls := by
  induction ls with
  | nil => simp [splitByLens]
  | cons l ls ih =>
    simp only [List.map_cons, List.flatten_cons, List.append_assoc, splitByLens]
    rw [List.take_left, List.drop_left, ih]

/-- **Proposition 2 (Variable multiplicity lower bound)**: Over any finite alphabet `|Σ| ≥ 2`,
if `L_α(p) ⊆ L_α(q)`, `q` contains at least one variable, and every variable of `q` occurs
at least `f ≥ 1` times, then every variable of `p` occurs at least `f` times. -/
theorem min_varOcc_le_of_langSubsetNE {α V₁ V₂ : Type*}
    [Fintype α] [DecidableEq α] [DecidableEq V₁] [DecidableEq V₂]
    (hcard : 2 ≤ Fintype.card α)
    {p : Pattern α V₁} {q : Pattern α V₂} (hsub : LangSubsetNE p q)
    {f : ℕ} (hf : 1 ≤ f) (hq_mult : ∀ x ∈ vars q, f ≤ varOcc x q) :
    ∀ y ∈ vars p, f ≤ varOcc y p := by
  classical
  intro y hy
  by_contra hlt
  push_neg at hlt
  set e := varOcc y p
  set s := Fintype.card α
  set C := p.length - e
  set m := varnum q
  have ha0_exists : Nonempty α := Fintype.card_pos_iff.mp (by omega)
  obtain ⟨a₀⟩ := ha0_exists
  obtain ⟨p₁, p₂, hp_split, hy_not_p₁⟩ := split_first_occ hy
  -- For any k ≥ 1, counting words of length N = e * k + C gives s^k ≤ (N + 1)^m * s^(N / f).
  have hcount : ∀ k : ℕ, 1 ≤ k →
      s ^ k ≤ (e * k + C + 1) ^ m * s ^ ((e * k + C) / f) := by
    intro k hk
    set N := e * k + C
    set M := N / f
    -- Build the set of length-N words in L_NE(p) and L_NE(q) as subsets of `Fin N → Σ`.
    let S_p : Finset (Fin N → α) :=
      Finset.univ.filter (fun w => InLangNE p (List.ofFn w))
    let S_q : Finset (Fin N → α) :=
      Finset.univ.filter (fun w => InLangNE q (List.ofFn w))
    have hSp_sub_Sq : S_p ⊆ S_q := by
      intro w hw
      simp only [S_p, S_q, Finset.mem_filter, Finset.mem_univ, true_and] at hw ⊢
      exact hsub _ hw
    -- Lower bound: `s^k ≤ S_p.card` via injection from `Fin k → Σ`.
    have hlower : s ^ k ≤ S_p.card := by
      let subOf (u : Fin k → α) : V₁ → List α :=
        Function.update (fun _ => [a₀]) y (List.ofFn u)
      have hlen_sub : ∀ u : Fin k → α, (applySub (subOf u) p).length = N := by
        intro u
        rw [applySub_length]
        have hlen_p := length_eq_termCount_add_sum_varOcc p
        have hsum_eq : ∑ x ∈ vars p, varOcc x p * (subOf u x).length =
            e * k + (∑ x ∈ vars p, varOcc x p - e) := by
          have h1 : ∑ x ∈ vars p, varOcc x p * (subOf u x).length =
              varOcc y p * (subOf u y).length +
              ∑ x ∈ (vars p).erase y, varOcc x p * (subOf u x).length := by
            rw [← Finset.add_sum_erase _ _ hy]
          have h2 : ∑ x ∈ (vars p).erase y, varOcc x p * (subOf u x).length =
              ∑ x ∈ (vars p).erase y, varOcc x p := by
            refine Finset.sum_congr rfl fun x hx => ?_
            have hne : x ≠ y := (Finset.mem_erase.mp hx).1
            simp [subOf, hne]
          have h3 : ∑ x ∈ vars p, varOcc x p = varOcc y p + ∑ x ∈ (vars p).erase y, varOcc x p :=
            (Finset.add_sum_erase _ _ hy).symm
          rw [h1, h2]
          simp only [subOf, Function.update_self, List.length_ofFn, e]
          omega
        have he_le_sum : e ≤ ∑ x ∈ vars p, varOcc x p :=
          Finset.single_le_sum (f := fun x => varOcc x p) (fun _ _ => Nat.zero_le _) hy
        rw [hsum_eq]
        omega
      let emb (u : Fin k → α) : Fin N → α :=
        fun i => (applySub (subOf u) p).get (i.cast (hlen_sub u).symm)
      have hemb_ofFn : ∀ u, List.ofFn (emb u) = applySub (subOf u) p := by
        intro u
        apply List.ext_getElem
        · simp [hlen_sub u]
        · intro i h1 h2
          simp [emb]
      have hemb_mem : ∀ u, emb u ∈ S_p := by
        intro u
        simp only [S_p, Finset.mem_filter, Finset.mem_univ, true_and, hemb_ofFn]
        refine ⟨subOf u, fun x => ?_, rfl⟩
        by_cases hxy : x = y
        · subst hxy
          simp only [subOf, Function.update_self, ne_eq, List.ofFn_eq_nil_iff]
          omega
        · simp [subOf, hxy]
      have hemb_inj : Function.Injective emb := by
        intro u₁ u₂ heq
        have hlist : applySub (subOf u₁) p = applySub (subOf u₂) p := by
          rw [← hemb_ofFn u₁, ← hemb_ofFn u₂, heq]
        rw [hp_split, applySub_append, applySub_append, applySub_append, applySub_append] at hlist
        have hp1_eq : applySub (subOf u₁) p₁ = applySub (subOf u₂) p₁ := by
          refine applySub_congr fun x hx => ?_
          have hne : x ≠ y := fun h => hy_not_p₁ (h ▸ hx)
          simp [subOf, hne]
        rw [hp1_eq, List.append_assoc, List.append_assoc] at hlist
        have hdrop := List.append_cancel_left hlist
        simp only [applySub, List.flatMap_cons, List.flatMap_nil, List.append_nil,
          subOf, Function.update_self] at hdrop
        have htake := congrArg (List.take k) hdrop
        rw [List.take_left' (by simp), List.take_left' (by simp)] at htake
        exact List.ofFn_injective htake
      calc s ^ k = Fintype.card (Fin k → α) := by simp [s]
        _ = (Finset.univ.image emb).card := (Finset.card_image_of_injective _ hemb_inj).symm
        _ ≤ S_p.card := Finset.card_le_card (Finset.image_subset_iff.mpr fun u _ => hemb_mem u)
    -- Upper bound: `S_q.card ≤ (N + 1)^m * s^M` by surjection from `(Fin m → Fin (N + 1)) × (Fin M → Σ)`.
    have hupper : S_q.card ≤ (N + 1) ^ m * s ^ M := by
      let Lq : List V₂ := (vars q).toList
      have hLq_len : Lq.length = m := by simp [Lq, m, varnum]
      have hLq_nodup : Lq.Nodup := Finset.nodup_toList (vars q)
      let decode : (Fin m → Fin (N + 1)) × (Fin M → α) → Fin N → α := fun ⟨lens, buf⟩ =>
        let slices := splitByLens (List.ofFn (fun i => (lens i).val)) (List.ofFn buf)
        let assoc := Lq.zip slices
        let σ : V₂ → List α := fun x =>
          match assoc.find? (fun pr => decide (pr.1 = x)) with
          | some pr => pr.2
          | none => []
        let w := applySub σ q
        fun i => w.getD i.val a₀
      have hcover : S_q ⊆ Finset.univ.image decode := by
        intro w hw
        simp only [S_q, Finset.mem_filter, Finset.mem_univ, true_and] at hw
        obtain ⟨σ, _, hσw⟩ := hw
        have hlen_w : (applySub σ q).length = N := by rw [hσw, List.length_ofFn]
        have hsum_mult : f * ∑ x ∈ vars q, (σ x).length ≤ N := by
          calc f * ∑ x ∈ vars q, (σ x).length
              = ∑ x ∈ vars q, f * (σ x).length := Finset.mul_sum _ _ _
            _ ≤ ∑ x ∈ vars q, varOcc x q * (σ x).length :=
                Finset.sum_le_sum fun x hx => Nat.mul_le_mul_right _ (hq_mult x hx)
            _ ≤ termCount q + ∑ x ∈ vars q, varOcc x q * (σ x).length := Nat.le_add_left _ _
            _ = (applySub σ q).length := (applySub_length σ q).symm
            _ = N := hlen_w
        have hsum_le_M : ∑ x ∈ vars q, (σ x).length ≤ M := by
          exact (Nat.le_div_iff_mul_le (by omega : 0 < f)).mpr (by rw [mul_comm]; exact hsum_mult)
        have hLq_sum : (Lq.map σ).flatten.length = ∑ x ∈ vars q, (σ x).length := by
          rw [List.length_flatten, List.map_map]
          have h1 : (Lq.map (List.length ∘ σ)).sum = ∑ x ∈ Lq.toFinset, (σ x).length :=
            (List.sum_toFinset (List.length ∘ σ) hLq_nodup).symm
          rw [h1]
          have hto : Lq.toFinset = vars q := by
            ext x; simp [Lq]
          rw [hto]
        have hflat_le_M : (Lq.map σ).flatten.length ≤ M := by omega
        let lens : Fin m → Fin (N + 1) := fun i =>
          ⟨(σ (Lq.get (i.cast hLq_len.symm))).length, by
            have hmem : Lq.get (i.cast hLq_len.symm) ∈ vars q := by
              rw [← Finset.mem_toList]
              exact List.get_mem _ _
            have hle : (σ (Lq.get (i.cast hLq_len.symm))).length ≤ ∑ x ∈ vars q, (σ x).length :=
              Finset.single_le_sum (f := fun x => (σ x).length) (fun _ _ => Nat.zero_le _) hmem
            have hMN : M ≤ N := Nat.div_le_self N f
            omega⟩
        let tail : List α := List.replicate (M - (Lq.map σ).flatten.length) a₀
        let padded : List α := (Lq.map σ).flatten ++ tail
        have hpadded_len : padded.length = M := by
          simp only [padded, tail, List.length_append, List.length_replicate]
          omega
        let buf : Fin M → α := fun i => padded.get (i.cast hpadded_len.symm)
        have hbuf_ofFn : List.ofFn buf = padded := by
          apply List.ext_getElem
          · simp [hpadded_len]
          · intro i h1 h2; simp [buf]
        have hlens_eq : List.ofFn (fun i => (lens i).val) = (Lq.map σ).map List.length := by
          apply List.ext_getElem
          · simp [hLq_len]
          · intro i h1 h2
            simp [lens]
        have hslices : splitByLens (List.ofFn (fun i => (lens i).val)) (List.ofFn buf) =
            Lq.map σ := by
          rw [hlens_eq, hbuf_ofFn]
          exact splitByLens_flatMap_append (Lq.map σ) tail
        rw [Finset.mem_image]
        refine ⟨(lens, buf), Finset.mem_univ _, ?_⟩
        -- Show `decode (lens, buf) = w`.
        have hσ_eq : ∀ x ∈ vars q,
            (match (Lq.zip (Lq.map σ)).find? (fun pr => decide (pr.1 = x)) with
            | some pr => pr.2
            | none => []) = σ x := by
          intro x hx
          have hxLq : x ∈ Lq := Finset.mem_toList.mpr hx
          have hpair : (x, σ x) ∈ Lq.zip (Lq.map σ) := by
            obtain ⟨idx, hidx, hget⟩ := List.getElem_of_mem hxLq
            rw [List.mem_iff_getElem]
            refine ⟨idx, by simp [hidx], ?_⟩
            simp [hget]
          have hsome : ∃ pr, (Lq.zip (Lq.map σ)).find? (fun pr => decide (pr.1 = x)) = some pr := by
            rw [← Option.isSome_iff_exists, List.find?_isSome]
            exact ⟨(x, σ x), hpair, by simp⟩
          obtain ⟨pr, hpr⟩ := hsome
          rw [hpr]
          have hmem_pr := List.mem_of_find?_eq_some hpr
          have hpred := List.find?_some hpr
          simp only [decide_eq_true_eq] at hpred
          obtain ⟨idx, hidx, hget⟩ := List.getElem_of_mem hmem_pr
          have hidx' : idx < Lq.length := by simp at hidx; exact hidx
          simp only [List.getElem_zip, List.getElem_map] at hget
          have h1 : Lq[idx]'hidx' = pr.1 := congrArg Prod.fst hget
          have h2 : σ (Lq[idx]'hidx') = pr.2 := congrArg Prod.snd hget
          dsimp only
          rw [← h2, h1, hpred]
        have happly : applySub (fun x =>
            match (Lq.zip (Lq.map σ)).find? (fun pr => decide (pr.1 = x)) with
            | some pr => pr.2
            | none => []) q = List.ofFn w := by
          rw [applySub_congr hσ_eq, hσw]
        funext i
        simp only [decode, hslices, happly]
        simp
      calc S_q.card ≤ (Finset.univ.image decode).card := Finset.card_le_card hcover
        _ ≤ (Finset.univ : Finset ((Fin m → Fin (N + 1)) × (Fin M → α))).card :=
            Finset.card_image_le
        _ = (N + 1) ^ m * s ^ M := by simp [s]
    exact le_trans hlower (le_trans (Finset.card_le_card hSp_sub_Sq) hupper)
  -- Choose `k = f * t` with `t` large enough to contradict exponential vs polynomial growth.
  obtain ⟨t, htC, hcontra⟩ := exists_pow_lt_two_pow m (e * f) (C + 1) C
  have hk_pos : 1 ≤ f * t := by
    have : 1 ≤ t := by omega
    exact Nat.mul_le_mul hf this
  have hc := hcount (f * t) hk_pos
  have hdiv : (e * (f * t) + C) / f = e * t + C / f := by
    have hmul : e * (f * t) + C = C + f * (e * t) := by ring
    rw [hmul, Nat.add_mul_div_left _ _ (by omega : 0 < f), add_comm]
  rw [hdiv] at hc
  have hle_exp : e * t + C / f ≤ f * t := by
    have h1 : e + 1 ≤ f := hlt
    have h2 : C / f ≤ C := Nat.div_le_self C f
    have h3 : (e + 1) * t ≤ f * t := Nat.mul_le_mul_right t h1
    nlinarith
  have hsub_ge : t - C ≤ f * t - (e * t + C / f) := by
    have h1 : e + 1 ≤ f := hlt
    have h2 : C / f ≤ C := Nat.div_le_self C f
    have h3 : (e + 1) * t ≤ f * t := Nat.mul_le_mul_right t h1
    have h4 : (e + 1) * t = e * t + t := by ring
    omega
  have hs_pos : 0 < s ^ (e * t + C / f) := Nat.pow_pos (by omega)
  have hcancel : s ^ (f * t - (e * t + C / f)) ≤ (e * (f * t) + C + 1) ^ m := by
    have hpow_mul : s ^ (f * t - (e * t + C / f)) * s ^ (e * t + C / f) = s ^ (f * t) := by
      rw [← pow_add, Nat.sub_add_cancel hle_exp]
    rw [← hpow_mul] at hc
    exact Nat.le_of_mul_le_mul_right hc hs_pos
  have h2_le : 2 ^ (t - C) ≤ s ^ (f * t - (e * t + C / f)) := by
    calc 2 ^ (t - C) ≤ 2 ^ (f * t - (e * t + C / f)) :=
          Nat.pow_le_pow_right (by omega) hsub_ge
      _ ≤ s ^ (f * t - (e * t + C / f)) :=
          Nat.pow_le_pow_left hcard _
  have hEq : e * (f * t) + C + 1 = e * f * t + (C + 1) := by ring
  rw [hEq] at hcancel
  have hlt_self : (e * f * t + (C + 1)) ^ m < (e * f * t + (C + 1)) ^ m :=
    lt_of_lt_of_le hcontra (le_trans h2_le hcancel)
  exact lt_irrefl _ hlt_self

/-- **Proposition 2 (Multiplicity-excess inequality)**: Under the hypotheses of Proposition 2,
`f * varnum(p) ≤ f * varnum(q) + (|p| - |q|) + E(q)`. -/
theorem multiplicity_excess_ineq {α V₁ V₂ : Type*}
    [Fintype α] [DecidableEq α] [DecidableEq V₁] [DecidableEq V₂]
    (hcard : 2 ≤ Fintype.card α)
    {p : Pattern α V₁} {q : Pattern α V₂} (hsub : LangSubsetNE p q)
    {f : ℕ} (hf : 1 ≤ f) (hq_mult : ∀ x ∈ vars q, f ≤ varOcc x q) :
    f * varnum p ≤ f * varnum q + (p.length - q.length) + excess q f := by
  have hp_mult := min_varOcc_le_of_langSubsetNE hcard hsub hf hq_mult
  have ht_le := termCount_le_of_langSubsetNE hcard hsub
  have hlen_p := length_eq_termCount_add_sum_varOcc p
  have hlen_q := length_eq_termCount_add_sum_varOcc q
  have hsum_p : f * varnum p ≤ ∑ y ∈ vars p, varOcc y p := by
    calc f * varnum p = ∑ _y ∈ vars p, f := by simp [varnum, mul_comm]
      _ ≤ ∑ y ∈ vars p, varOcc y p := Finset.sum_le_sum fun y hy => hp_mult y hy
  have hsum_q : f * varnum q ≤ ∑ x ∈ vars q, varOcc x q := by
    calc f * varnum q = ∑ _x ∈ vars q, f := by simp [varnum, mul_comm]
      _ ≤ ∑ x ∈ vars q, varOcc x q := Finset.sum_le_sum fun x hx => hq_mult x hx
  have hlen_le : q.length ≤ p.length := by
    haveI : Nonempty α := Fintype.card_pos_iff.mp (by omega)
    exact len_le_of_langSubsetNE hsub
  simp only [excess]
  omega

/-- **Proposition 2 (Strict monotonicity under small multiplicity excess)**: If `|q| < |p|` and
`E(q) < (2f - 1)(|p| - |q|)`, then `μ(q) < μ(p)`. -/
theorem strict_mono_of_small_excess {α V₁ V₂ : Type*}
    [Fintype α] [DecidableEq α] [DecidableEq V₁] [DecidableEq V₂]
    (hcard : 2 ≤ Fintype.card α)
    {p : Pattern α V₁} {q : Pattern α V₂} (hsub : LangSubsetNE p q)
    {f : ℕ} (hf : 1 ≤ f) (hq_mult : ∀ x ∈ vars q, f ≤ varOcc x q)
    (hgt : q.length < p.length)
    (hex : excess q f < (2 * f - 1) * (p.length - q.length)) :
    measure q < measure p := by
  have hineq := multiplicity_excess_ineq hcard hsub hf hq_mult
  have hvp_le := varnum_le_length p
  have hvq_le := varnum_le_length q
  set d := p.length - q.length
  have hd : 1 ≤ d := by omega
  have hmul : f * varnum p < f * (varnum q + 2 * d) := by
    have h1 : d + (2 * f - 1) * d = 2 * f * d := by
      have : 1 + (2 * f - 1) = 2 * f := by omega
      calc d + (2 * f - 1) * d = (1 + (2 * f - 1)) * d := by ring
        _ = 2 * f * d := by rw [this]
    have h2 : f * (varnum q + 2 * d) = f * varnum q + 2 * f * d := by ring
    omega
  have hv_lt : varnum p < varnum q + 2 * d := Nat.lt_of_mul_lt_mul_left hmul
  simp only [measure]
  omega

/-- **Proposition 2 (Constant-multiplicity corollary)**: Whenever every variable of `q` has the
same multiplicity `f ≥ 1` and `|q| < |p|`, `L_α(p) ⊆ L_α(q)` implies `μ(q) < μ(p)`. -/
theorem strict_mono_of_const_mult {α V₁ V₂ : Type*}
    [Fintype α] [DecidableEq α] [DecidableEq V₁] [DecidableEq V₂]
    (hcard : 2 ≤ Fintype.card α)
    {p : Pattern α V₁} {q : Pattern α V₂} (hsub : LangSubsetNE p q)
    {f : ℕ} (hf : 1 ≤ f) (hq_const : ∀ x ∈ vars q, varOcc x q = f)
    (hgt : q.length < p.length) :
    measure q < measure p := by
  have hex0 : excess q f = 0 := by
    have hlen_q := length_eq_termCount_add_sum_varOcc q
    have hsum : ∑ x ∈ vars q, varOcc x q = f * varnum q := by
      calc ∑ x ∈ vars q, varOcc x q = ∑ _x ∈ vars q, f := Finset.sum_congr rfl hq_const
        _ = f * varnum q := by simp [varnum, mul_comm]
    simp only [excess]
    omega
  refine strict_mono_of_small_excess hcard hsub hf (fun x hx => (hq_const x hx).ge) hgt ?_
  rw [hex0]
  exact Nat.mul_pos (by omega) (by omega)

/-! ### 6. Rado's Linear Independent Representatives Theorem (Lemma 2) -/

section Rado

variable {K V : Type*} [Field K] [AddCommGroup V] [Module K V] [FiniteDimensional K V]

private lemma finrank_map_mkQ_add_le (W U : Submodule K V) :
    Module.finrank K U ≤ Module.finrank K (U.map W.mkQ) + Module.finrank K W := by
  let f : U →ₗ[K] V ⧸ W := W.mkQ.comp U.subtype
  have hrange : LinearMap.range f = U.map W.mkQ := by
    ext z
    simp [f, Submodule.mem_map]
  have hker_le : Module.finrank K (LinearMap.ker f) ≤ Module.finrank K W := by
    let g : LinearMap.ker f →ₗ[K] W :=
      { toFun := fun x => ⟨x.1.1, by
          have hx := x.2
          simp only [LinearMap.mem_ker, f, LinearMap.comp_apply, Submodule.coe_subtype,
            Submodule.mkQ_apply, Submodule.Quotient.mk_eq_zero] at hx
          exact hx⟩
        map_add' := fun _ _ => rfl
        map_smul' := fun _ _ => rfl }
    have hginj : Function.Injective g := by
      intro x y hxy
      exact Subtype.ext (Subtype.ext (congrArg (Subtype.val : W → V) hxy))
    exact LinearMap.finrank_le_finrank_of_injective hginj
  have hrn := LinearMap.finrank_range_add_finrank_ker f
  rw [hrange] at hrn
  omega

private lemma finrank_map_mkQ_of_le {W U : Submodule K V} (hWU : W ≤ U) :
    Module.finrank K (U.map W.mkQ) + Module.finrank K W = Module.finrank K U := by
  let f : U →ₗ[K] V ⧸ W := W.mkQ.comp U.subtype
  have hrange : LinearMap.range f = U.map W.mkQ := by
    ext z
    simp [f, Submodule.mem_map]
  have hker_eq : Module.finrank K (LinearMap.ker f) = Module.finrank K W := by
    let e : LinearMap.ker f ≃ₗ[K] W :=
      { toFun := fun x => ⟨x.1.1, by
          have hx := x.2
          simp only [LinearMap.mem_ker, f, LinearMap.comp_apply, Submodule.coe_subtype,
            Submodule.mkQ_apply, Submodule.Quotient.mk_eq_zero] at hx
          exact hx⟩
        invFun := fun w => ⟨⟨w.1, hWU w.2⟩, by
          simp [f, Submodule.Quotient.mk_eq_zero, w.2]⟩
        left_inv := fun x => by ext; rfl
        right_inv := fun w => by ext; rfl
        map_add' := fun _ _ => rfl
        map_smul' := fun _ _ => rfl }
    exact e.finrank_eq
  have hrn := LinearMap.finrank_range_add_finrank_ker f
  rw [hrange, hker_eq] at hrn
  exact hrn

/-- Helper induction for Lemma 2 over an arbitrary finite index set `A : Finset ι`. -/
private lemma rado_aux {ι : Type*} [DecidableEq ι] :
    ∀ n : ℕ, ∀ {V : Type*} [AddCommGroup V] [Module K V] [FiniteDimensional K V]
      (A : Finset ι) (_hcard : A.card = n) (S : ι → Finset V),
      (∀ I : Finset ι, I ⊆ A → I.Nonempty →
        I.card ≤ Module.finrank K (Submodule.span K (⋃ i ∈ I, (S i : Set V)))) →
      ∃ u : A → V, (∀ i : A, u i ∈ S i.1) ∧ LinearIndependent K u := by
  intro n
  induction n using Nat.strong_induction_on with
  | h n ih =>
    intro V _ _ _ A hcard S hRado
    classical
    by_cases hA0 : n = 0
    · subst hA0
      have hAe : A = ∅ := Finset.card_eq_zero.mp hcard
      subst hAe
      exact ⟨fun i => (Finset.notMem_empty i.1 i.2).elim,
        fun i => (Finset.notMem_empty i.1 i.2).elim,
        linearIndependent_empty_type⟩
    · -- Check if there is a tight nonempty proper subset `J ⊂ A`.
      by_cases htight : ∃ J : Finset ι, J ⊂ A ∧ J.Nonempty ∧
          Module.finrank K (Submodule.span K (⋃ i ∈ J, (S i : Set V))) = J.card
      · obtain ⟨J, hJss, hJne, hJtight⟩ := htight
        set W : Submodule K V := Submodule.span K (⋃ i ∈ J, (S i : Set V))
        have hJlt : J.card < n := by
          rw [← hcard]
          exact Finset.card_lt_card hJss
        -- Apply IH to `J`.
        obtain ⟨uJ, huJ_mem, huJ_li⟩ :=
          ih J.card hJlt J rfl S (fun I hIJ hIne => hRado I (hIJ.trans hJss.1) hIne)
        -- The span of `uJ` equals `W` since `uJ` is linearly independent of size `finrank W`.
        have hspan_uJ : Submodule.span K (Set.range uJ) = W := by
          have hle : Submodule.span K (Set.range uJ) ≤ W := by
            rw [Submodule.span_le]
            rintro v ⟨i, rfl⟩
            exact Submodule.subset_span (Set.mem_biUnion i.2 (huJ_mem i))
          have hfr : Module.finrank K (Submodule.span K (Set.range uJ)) = Module.finrank K W := by
            rw [finrank_span_eq_card huJ_li, Fintype.card_coe, hJtight]
          exact Submodule.eq_of_le_of_finrank_eq hle hfr
        -- Pass to the quotient `V ⧸ W` for `A \ J`.
        let R := A \ J
        have hRlt : R.card < n := by
          have hJpos : 0 < J.card := hJne.card_pos
          have hRc : R.card = A.card - J.card := by
            rw [Finset.card_sdiff, Finset.inter_eq_left.mpr hJss.1]
          omega
        let S' : ι → Finset (V ⧸ W) := fun i => (S i).image W.mkQ
        have hRado' : ∀ I : Finset ι, I ⊆ R → I.Nonempty →
            I.card ≤ Module.finrank K (Submodule.span K (⋃ i ∈ I, (S' i : Set (V ⧸ W)))) := by
          intro I hIR hIne
          have hdisj : Disjoint I J := Finset.disjoint_of_subset_left hIR disjoint_sdiff_self_left
          have hIUJ_sub : I ∪ J ⊆ A := Finset.union_subset (hIR.trans Finset.sdiff_subset) hJss.1
          have hIUJ_ne : (I ∪ J).Nonempty := hIne.mono Finset.subset_union_left
          have hbound := hRado (I ∪ J) hIUJ_sub hIUJ_ne
          rw [Finset.card_union_of_disjoint hdisj] at hbound
          set U := Submodule.span K (⋃ i ∈ I ∪ J, (S i : Set V))
          have hWU : W ≤ U := Submodule.span_mono (Set.biUnion_subset_biUnion_left
            (fun i hi => Finset.mem_union_right I hi))
          have hmap_eq : U.map W.mkQ = Submodule.span K (⋃ i ∈ I, (S' i : Set (V ⧸ W))) := by
            rw [Submodule.map_span]
            apply le_antisymm
            · rw [Submodule.span_le]
              rintro _ ⟨v, hv, rfl⟩
              rw [Set.mem_iUnion₂] at hv
              obtain ⟨i, hi, hvi⟩ := hv
              rcases Finset.mem_union.mp hi with hiI | hiJ
              · exact Submodule.subset_span (Set.mem_biUnion hiI
                  (Finset.mem_coe.mpr (Finset.mem_image_of_mem _ hvi)))
              · have hvW : v ∈ W := Submodule.subset_span (Set.mem_biUnion hiJ hvi)
                have hzero : W.mkQ v = 0 := (Submodule.Quotient.mk_eq_zero W).mpr hvW
                rw [hzero]
                exact Submodule.zero_mem _
            · rw [Submodule.span_le]
              rintro _ hmem
              rw [Set.mem_iUnion₂] at hmem
              obtain ⟨i, hiI, hvi⟩ := hmem
              rw [Finset.mem_coe, Finset.mem_image] at hvi
              obtain ⟨v, hvSi, rfl⟩ := hvi
              exact Submodule.subset_span ⟨v, Set.mem_biUnion (Finset.mem_union_left J hiI) hvSi, rfl⟩
          have hadd := finrank_map_mkQ_of_le hWU
          rw [hmap_eq, hJtight] at hadd
          omega
        obtain ⟨uR_q, huR_q_mem, huR_q_li⟩ := ih R.card hRlt R rfl S' hRado'
        -- Lift each `uR_q i` back to `uR i ∈ S i`.
        have hlift : ∀ i : R, ∃ v ∈ S i.1, W.mkQ v = uR_q i := fun i =>
          Finset.mem_image.mp (huR_q_mem i)
        choose uR huR_mem huR_mkQ using hlift
        -- Combine `uJ` and `uR` on `A`.
        let u : A → V := fun i =>
          if h : i.1 ∈ J then uJ ⟨i.1, h⟩ else uR ⟨i.1, Finset.mem_sdiff.mpr ⟨i.2, h⟩⟩
        refine ⟨u, fun i => ?_, ?_⟩
        · dsimp only [u]
          split_ifs with h
          · exact huJ_mem ⟨i.1, h⟩
          · exact huR_mem ⟨i.1, Finset.mem_sdiff.mpr ⟨i.2, h⟩⟩
        · rw [Fintype.linearIndependent_iff]
          intro g hg
          -- Apply `W.mkQ` to `∑ i : A, g i • u i = 0`.
          have hq0 : ∑ i : R, g ⟨i.1, (Finset.mem_sdiff.mp i.2).1⟩ • uR_q i = 0 := by
            have hmk := congrArg W.mkQ hg
            rw [map_sum, map_zero] at hmk
            have hsplit_sum : ∑ i : A, W.mkQ (g i • u i) =
                ∑ i : R, g ⟨i.1, (Finset.mem_sdiff.mp i.2).1⟩ • uR_q i := by
              rw [Finset.univ_eq_attach]
              have hpart : A.attach =
                  (A.attach.filter (fun i => i.1 ∈ J)) ∪ (A.attach.filter (fun i => i.1 ∉ J)) := by
                ext i; simp; tauto
              rw [hpart, Finset.sum_union (Finset.disjoint_filter.mpr (fun _ _ h1 h2 => h2 h1))]
              have hJzero : ∑ i ∈ A.attach.filter (fun i => i.1 ∈ J), W.mkQ (g i • u i) = 0 := by
                refine Finset.sum_eq_zero fun i hi => ?_
                have hiJ : i.1 ∈ J := (Finset.mem_filter.mp hi).2
                rw [map_smul]
                have huW : u i ∈ W := by
                  simp only [u, hiJ, dite_true]
                  rw [← hspan_uJ]
                  exact Submodule.subset_span ⟨⟨i.1, hiJ⟩, rfl⟩
                rw [Submodule.mkQ_apply, (Submodule.Quotient.mk_eq_zero W).mpr huW, smul_zero]
              rw [hJzero, zero_add]
              refine Finset.sum_bij (fun i hi => ⟨i.1, Finset.mem_sdiff.mpr ⟨i.2, (Finset.mem_filter.mp hi).2⟩⟩)
                (fun _ _ => Finset.mem_univ _)
                (fun i₁ _ i₂ _ heq => Subtype.ext (by simpa using congrArg Subtype.val heq))
                (fun r _ => ⟨⟨r.1, (Finset.mem_sdiff.mp r.2).1⟩,
                  Finset.mem_filter.mpr ⟨Finset.mem_attach _ _, (Finset.mem_sdiff.mp r.2).2⟩, rfl⟩)
                (fun i hi => ?_)
              have hiR : i.1 ∉ J := (Finset.mem_filter.mp hi).2
              simp only [u, hiR, dite_false, map_smul, huR_mkQ]
            rw [← hsplit_sum]
            exact hmk
          have hgR : ∀ i : R, g ⟨i.1, (Finset.mem_sdiff.mp i.2).1⟩ = 0 :=
            Fintype.linearIndependent_iff.mp huR_q_li _ hq0
          -- Now the sum reduces to `J`, where `uJ` is linearly independent.
          have hJ0 : ∑ j : J, g ⟨j.1, hJss.1 j.2⟩ • uJ j = 0 := by
            rw [← hg]
            symm
            rw [Finset.univ_eq_attach]
            have hpart : A.attach =
                (A.attach.filter (fun i => i.1 ∈ J)) ∪ (A.attach.filter (fun i => i.1 ∉ J)) := by
              ext i; simp; tauto
            rw [hpart, Finset.sum_union (Finset.disjoint_filter.mpr (fun _ _ h1 h2 => h2 h1))]
            have hRzero : ∑ i ∈ A.attach.filter (fun i => i.1 ∉ J), g i • u i = 0 := by
              refine Finset.sum_eq_zero fun i hi => ?_
              have hiR : i.1 ∉ J := (Finset.mem_filter.mp hi).2
              have h0 := hgR ⟨i.1, Finset.mem_sdiff.mpr ⟨i.2, hiR⟩⟩
              simp [h0]
            rw [hRzero, add_zero]
            refine Finset.sum_bij (fun i hi => ⟨i.1, (Finset.mem_filter.mp hi).2⟩)
              (fun _ _ => Finset.mem_univ _)
              (fun i₁ _ i₂ _ heq => Subtype.ext (by simpa using congrArg Subtype.val heq))
              (fun j _ => ⟨⟨j.1, hJss.1 j.2⟩,
                Finset.mem_filter.mpr ⟨Finset.mem_attach _ _, j.2⟩, rfl⟩)
              (fun i hi => ?_)
            have hiJ : i.1 ∈ J := (Finset.mem_filter.mp hi).2
            simp [u, hiJ]
          have hgJ : ∀ j : J, g ⟨j.1, hJss.1 j.2⟩ = 0 :=
            Fintype.linearIndependent_iff.mp huJ_li _ hJ0
          intro i
          by_cases hiJ : i.1 ∈ J
          · exact hgJ ⟨i.1, hiJ⟩
          · exact hgR ⟨i.1, Finset.mem_sdiff.mpr ⟨i.2, hiJ⟩⟩
      · -- No nonempty proper subset of `A` is tight!
        push_neg at htight
        have hAne : A.Nonempty := Finset.card_pos.mp (by omega)
        obtain ⟨a, haA⟩ := hAne
        -- `S a` contains a nonzero vector `v₀`.
        have hSa_dim : 1 ≤ Module.finrank K (Submodule.span K (⋃ i ∈ ({a} : Finset ι), (S i : Set V))) :=
          hRado {a} (Finset.singleton_subset_iff.mpr haA) (Finset.singleton_nonempty a)
        have hv0_ex : ∃ v₀ ∈ S a, v₀ ≠ 0 := by
          by_contra hzero
          push_neg at hzero
          have hsp0 : Submodule.span K (⋃ i ∈ ({a} : Finset ι), (S i : Set V)) = ⊥ := by
            rw [Submodule.span_eq_bot]
            intro v hv
            simp only [Set.mem_iUnion, Finset.mem_singleton, exists_prop, Finset.mem_coe] at hv
            obtain ⟨i, rfl, hvi⟩ := hv
            exact hzero v hvi
          rw [hsp0, finrank_bot] at hSa_dim
          omega
        obtain ⟨v₀, hv0_mem, hv0_ne⟩ := hv0_ex
        let W : Submodule K V := K ∙ v₀
        have hW_dim : Module.finrank K W = 1 := finrank_span_singleton hv0_ne
        let R := A.erase a
        have hRlt : R.card < n := by
          rw [Finset.card_erase_of_mem haA, hcard]
          omega
        let S' : ι → Finset (V ⧸ W) := fun i => (S i).image W.mkQ
        have hRado' : ∀ I : Finset ι, I ⊆ R → I.Nonempty →
            I.card ≤ Module.finrank K (Submodule.span K (⋃ i ∈ I, (S' i : Set (V ⧸ W)))) := by
          intro I hIR hIne
          have hIss : I ⊂ A := by
            rw [Finset.ssubset_iff_subset_ne]
            refine ⟨hIR.trans (Finset.erase_subset a A), fun heq => ?_⟩
            exact (Finset.notMem_erase a A) (hIR (heq ▸ haA))
          have hle := hRado I hIss.1 hIne
          have hne := htight I hIss hIne
          have hstrict : I.card + 1 ≤ Module.finrank K (Submodule.span K (⋃ i ∈ I, (S i : Set V))) := by
            omega
          set U := Submodule.span K (⋃ i ∈ I, (S i : Set V))
          have hmap_eq : U.map W.mkQ = Submodule.span K (⋃ i ∈ I, (S' i : Set (V ⧸ W))) := by
            rw [Submodule.map_span]
            congr 1
            ext z
            simp only [Set.mem_image, Set.mem_iUnion, Finset.mem_coe, exists_prop, S', Finset.mem_image]
            constructor
            · rintro ⟨v, ⟨i, hi, hv⟩, rfl⟩
              exact ⟨i, hi, v, hv, rfl⟩
            · rintro ⟨i, hi, v, hv, rfl⟩
              exact ⟨v, ⟨i, hi, hv⟩, rfl⟩
          have hdrop := finrank_map_mkQ_add_le W U
          rw [hmap_eq, hW_dim] at hdrop
          omega
        obtain ⟨uR_q, huR_q_mem, huR_q_li⟩ := ih R.card hRlt R rfl S' hRado'
        have hlift : ∀ i : R, ∃ v ∈ S i.1, W.mkQ v = uR_q i := fun i =>
          Finset.mem_image.mp (huR_q_mem i)
        choose uR huR_mem huR_mkQ using hlift
        let u : A → V := fun i =>
          if h : i.1 = a then v₀ else uR ⟨i.1, Finset.mem_erase.mpr ⟨h, i.2⟩⟩
        refine ⟨u, fun i => ?_, ?_⟩
        · dsimp only [u]
          split_ifs with h
          · rw [h]; exact hv0_mem
          · exact huR_mem ⟨i.1, Finset.mem_erase.mpr ⟨h, i.2⟩⟩
        · rw [Fintype.linearIndependent_iff]
          intro g hg
          have hq0 : ∑ i : R, g ⟨i.1, Finset.mem_of_mem_erase i.2⟩ • uR_q i = 0 := by
            have hmk := congrArg W.mkQ hg
            rw [map_sum, map_zero] at hmk
            have hsplit : ∑ i : A, W.mkQ (g i • u i) =
                ∑ i : R, g ⟨i.1, Finset.mem_of_mem_erase i.2⟩ • uR_q i := by
              rw [Finset.univ_eq_attach]
              have hpart : A.attach =
                  insert ⟨a, haA⟩ (A.attach.erase ⟨a, haA⟩) :=
                (Finset.insert_erase (Finset.mem_attach A ⟨a, haA⟩)).symm
              rw [hpart, Finset.sum_insert (Finset.notMem_erase _ _)]
              have ha0 : W.mkQ (g ⟨a, haA⟩ • u ⟨a, haA⟩) = 0 := by
                simp only [u, dite_true, map_smul, Submodule.mkQ_apply]
                have hvW : v₀ ∈ W := Submodule.mem_span_singleton_self v₀
                rw [(Submodule.Quotient.mk_eq_zero W).mpr hvW, smul_zero]
              rw [ha0, zero_add]
              refine Finset.sum_bij
                (fun i hi => ⟨i.1, Finset.mem_erase.mpr ⟨fun h => (Finset.mem_erase.mp hi).1 (Subtype.ext h), i.2⟩⟩)
                (fun _ _ => Finset.mem_univ _)
                (fun i₁ _ i₂ _ heq => Subtype.ext (by simpa using congrArg Subtype.val heq))
                (fun r _ => ⟨⟨r.1, Finset.mem_of_mem_erase r.2⟩,
                  Finset.mem_erase.mpr ⟨fun h => (Finset.mem_erase.mp r.2).1 (congrArg Subtype.val h),
                    Finset.mem_attach _ _⟩, rfl⟩)
                (fun i hi => ?_)
              have hne : i.1 ≠ a := fun h => (Finset.mem_erase.mp hi).1 (Subtype.ext h)
              simp only [u, hne, dite_false, map_smul, huR_mkQ]
            rw [← hsplit]
            exact hmk
          have hgR : ∀ i : R, g ⟨i.1, Finset.mem_of_mem_erase i.2⟩ = 0 :=
            Fintype.linearIndependent_iff.mp huR_q_li _ hq0
          have hga : g ⟨a, haA⟩ = 0 := by
            have honly : g ⟨a, haA⟩ • v₀ = 0 := by
              rw [← hg]
              symm
              rw [Finset.sum_eq_single ⟨a, haA⟩]
              · simp [u]
              · intro i _ hne
                have hne' : i.1 ≠ a := fun h => hne (Subtype.ext h)
                have h0 := hgR ⟨i.1, Finset.mem_erase.mpr ⟨hne', i.2⟩⟩
                simp [h0]
              · intro h; exact (h (Finset.mem_univ _)).elim
            exact (smul_eq_zero.mp honly).resolve_right hv0_ne
          intro i
          by_cases hi : i.1 = a
          · have : i = ⟨a, haA⟩ := Subtype.ext hi
            rw [this]; exact hga
          · exact hgR ⟨i.1, Finset.mem_erase.mpr ⟨hi, i.2⟩⟩

/-- **Lemma 2 (Linear independent representatives; Rado, 1942)**:
Let `S : Fin R → Finset V` be finite subsets of a finite-dimensional vector space `V` over `K`.
If `|I| ≤ finrank (span (⋃ i ∈ I, S i))` for every nonempty `I ⊆ Fin R`, then there exist
linearly independent vectors `u : Fin R → V` with `u i ∈ S i` for all `i`. -/
theorem rado_linear_independent_representatives {R : ℕ} (S : Fin R → Finset V)
    (hRado : ∀ I : Finset (Fin R), I.Nonempty →
      I.card ≤ Module.finrank K (Submodule.span K (⋃ i ∈ I, (S i : Set V)))) :
    ∃ u : Fin R → V, (∀ i, u i ∈ S i) ∧ LinearIndependent K u := by
  obtain ⟨u, hmem, hli⟩ :=
    rado_aux (K := K) R Finset.univ (by simp) S (fun I _ hIne => hRado I hIne)
  let e : Fin R ≃ (Finset.univ : Finset (Fin R)) :=
    { toFun := fun i => ⟨i, Finset.mem_univ i⟩
      invFun := fun i => i.1
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  exact ⟨u ∘ e, fun i => hmem (e i), hli.comp e e.injective⟩

end Rado

/-! ### 7. All-Lengths Gap-One Theorem (Theorem 2) -/

section GapOne

/-- Standard coordinate projection functional on `Fin n → ZMod 2`. -/
private def coordProj (n : ℕ) (k : Fin n) : Module.Dual (ZMod 2) (Fin n → ZMod 2) :=
  LinearMap.proj k

/-- Index shift inserting a free coordinate right after position `b : Fin m`. -/
private def shiftIdx {m : ℕ} (b j : Fin m) : Fin (m + 1) :=
  if j.val ≤ b.val then ⟨j.val, by omega⟩ else ⟨j.val + 1, by omega⟩

private lemma shiftIdx_injective {m : ℕ} (b : Fin m) : Function.Injective (shiftIdx b) := by
  intro j₁ j₂ h
  simp only [shiftIdx] at h
  split_ifs at h with h1 h2 h2
  · exact Fin.ext (by simpa using congrArg Fin.val h)
  · have := congrArg Fin.val h; dsimp at this; omega
  · have := congrArg Fin.val h; dsimp at this; omega
  · have := congrArg Fin.val h; dsimp at this; exact Fin.ext (by omega)

private lemma shiftIdx_ne_inserted {m : ℕ} (b j : Fin m) :
    shiftIdx b j ≠ ⟨b.val + 1, by omega⟩ := by
  simp only [shiftIdx]
  split_ifs with h
  · intro heq
    have := congrArg Fin.val heq; dsimp at this; omega
  · intro heq
    have := congrArg Fin.val heq; dsimp at this; omega

/-- Position `j` in `q` holds a singleton variable (occurring exactly once in `q`). -/
private def isSingletonPos {V : Type*} [DecidableEq V] (q : Pattern (ZMod 2) V)
    (j : Fin q.length) : Prop :=
  ∃ v : V, q[j.val] = .inr v ∧ varOcc v q = 1

private instance {V : Type*} [DecidableEq V] (q : Pattern (ZMod 2) V) (j : Fin q.length) :
    Decidable (isSingletonPos q j) := by
  unfold isSingletonPos
  cases h : q[j.val] with
  | inl a =>
    exact isFalse (by rintro ⟨v, hv, _⟩; simp at hv)
  | inr v =>
    if hv : varOcc v q = 1 then
      exact isTrue ⟨v, rfl, hv⟩
    else
      exact isFalse (by rintro ⟨w, hw, h1⟩; simp at hw; subst hw; exact hv h1)

/-- Position `b` is the right endpoint of a maximal consecutive run of singleton variables in `q`. -/
private def isRunEnd {V : Type*} [DecidableEq V] (q : Pattern (ZMod 2) V)
    (b : Fin q.length) : Prop :=
  isSingletonPos q b ∧ ∀ h : b.val + 1 < q.length, ¬ isSingletonPos q ⟨b.val + 1, h⟩

private instance {V : Type*} [DecidableEq V] (q : Pattern (ZMod 2) V) (b : Fin q.length) :
    Decidable (isRunEnd q b) := by
  unfold isRunEnd
  infer_instance

/-- Every singleton position `k` lies in a consecutive run of singletons ending at some run end `b ≥ k`. -/
private lemma exists_runEnd_ge {V : Type*} [DecidableEq V] (q : Pattern (ZMod 2) V)
    (k : Fin q.length) (hk : isSingletonPos q k) :
    ∃ b : Fin q.length, k.val ≤ b.val ∧ isRunEnd q b ∧
      ∀ j : Fin q.length, k.val ≤ j.val → j.val ≤ b.val → isSingletonPos q j := by
  -- Strong downward induction on `q.length - 1 - k.val`.
  suffices ∀ d : ℕ, ∀ k : Fin q.length, q.length - 1 - k.val = d → isSingletonPos q k →
      ∃ b : Fin q.length, k.val ≤ b.val ∧ isRunEnd q b ∧
        ∀ j : Fin q.length, k.val ≤ j.val → j.val ≤ b.val → isSingletonPos q j by
    exact this (q.length - 1 - k.val) k rfl hk
  intro d
  induction d with
  | zero =>
    intro k hkd hsk
    refine ⟨k, le_refl _, ⟨hsk, fun h => by omega⟩, fun j h1 h2 => ?_⟩
    have : j = k := Fin.ext (by omega)
    rw [this]; exact hsk
  | succ d ih =>
    intro k hkd hsk
    have hnext : k.val + 1 < q.length := by omega
    by_cases hsn : isSingletonPos q ⟨k.val + 1, hnext⟩
    · obtain ⟨b, hkb, hbe, hall⟩ := ih ⟨k.val + 1, hnext⟩ (by dsimp; omega) hsn
      refine ⟨b, by dsimp at hkb; omega, hbe, fun j h1 h2 => ?_⟩
      by_cases hjk : j.val = k.val
      · have : j = k := Fin.ext hjk
        rw [this]; exact hsk
      · exact hall j (by dsimp; omega) h2
    · refine ⟨k, le_refl _, ⟨hsk, fun _ => hsn⟩, fun j h1 h2 => ?_⟩
      have : j = k := Fin.ext (by omega)
      rw [this]; exact hsk

/-- An affine equation `(φ, β)` satisfied by `x` means `φ x = β`. -/
private abbrev AffEq (n : ℕ) := Module.Dual (ZMod 2) (Fin n → ZMod 2) × ZMod 2

/-- The defining affine equations of the expanded subspace `Q_b` for run end `b : Fin q.length`. -/
private noncomputable def eqsOfRunEnd {V : Type*} [DecidableEq V] (q : Pattern (ZMod 2) V)
    (b : Fin q.length) : Finset (AffEq (q.length + 1)) :=
  let termEqs : Finset (AffEq (q.length + 1)) :=
    Finset.univ.biUnion fun j : Fin q.length =>
      match q[j.val] with
      | .inl a => {(coordProj (q.length + 1) (shiftIdx b j), a)}
      | .inr _ => ∅
  let varEqs : Finset (AffEq (q.length + 1)) :=
    (Finset.univ ×ˢ Finset.univ).biUnion fun pr : Fin q.length × Fin q.length =>
      if pr.1 ≠ pr.2 then
        match q[pr.1.val], q[pr.2.val] with
        | .inr v₁, .inr v₂ =>
          if v₁ = v₂ then
            {(coordProj (q.length + 1) (shiftIdx b pr.1) -
              coordProj (q.length + 1) (shiftIdx b pr.2), 0)}
          else ∅
        | _, _ => ∅
      else ∅
  termEqs ∪ varEqs

/-- The linear parts of `eqsOfRunEnd q b`. -/
private noncomputable def linOfRunEnd {V : Type*} [DecidableEq V] (q : Pattern (ZMod 2) V)
    (b : Fin q.length) : Finset (Module.Dual (ZMod 2) (Fin (q.length + 1) → ZMod 2)) :=
  (eqsOfRunEnd q b).image Prod.fst

/-- Helper: length of `applySub σ p` when every variable of `p` maps to a length-1 list. -/
private lemma applySub_length_of_all_one {V : Type*} [DecidableEq V]
    (σ : V → List (ZMod 2)) (p : Pattern (ZMod 2) V) (h1 : ∀ x ∈ vars p, (σ x).length = 1) :
    (applySub σ p).length = p.length := by
  rw [applySub_length, length_eq_termCount_add_sum_varOcc]
  congr 1
  refine Finset.sum_congr rfl fun x hx => ?_
  rw [h1 x hx, mul_one]

/-- Helper: elementwise description of `applySub σ p` when every variable of `p` maps to length `1`. -/
private lemma applySub_getElem_of_all_one {V : Type*} [DecidableEq V]
    (σ : V → List (ZMod 2)) (p : Pattern (ZMod 2) V)
    (hσ_ne : ∀ x, σ x ≠ []) (h1 : ∀ x ∈ vars p, (σ x).length = 1) :
    (applySub σ p).length = p.length ∧
    (∀ (j : ℕ) (hj : j < p.length),
      (applySub σ p)[j]'(by rw [applySub_length_of_all_one σ p h1]; exact hj) =
        match p[j] with
        | .inl a => a
        | .inr v => (σ v)[0]'(List.length_pos_iff.mpr (hσ_ne v))) := by
  refine ⟨applySub_length_of_all_one σ p h1, ?_⟩
  induction p with
  | nil => intro j hj; simp at hj
  | cons t ps ih =>
    intro j hj
    cases t with
    | inl a =>
      have h1' : ∀ x ∈ vars ps, (σ x).length = 1 := fun x hx => h1 x (by simpa [vars_cons_inl] using hx)
      cases j with
      | zero => simp [applySub]
      | succ j' =>
        simp only [applySub, List.flatMap_cons, List.cons_append, List.nil_append,
          List.getElem_cons_succ]
        exact ih h1' j' (by simpa using hj)
    | inr y =>
      have hy1 : (σ y).length = 1 := h1 y (by simp [vars_cons_inr])
      obtain ⟨c, hc⟩ := List.length_eq_one_iff.mp hy1
      have h1' : ∀ x ∈ vars ps, (σ x).length = 1 := fun x hx => h1 x (by simp [vars_cons_inr, hx])
      cases j with
      | zero => simp [applySub, hc]
      | succ j' =>
        simp only [applySub, List.flatMap_cons, hc, List.cons_append, List.nil_append,
          List.getElem_cons_succ]
        exact ih h1' j' (by simpa using hj)

private lemma varOcc_eq_card_filter {α V : Type*} [DecidableEq α] [DecidableEq V]
    (v : V) (q : Pattern α V) :
    varOcc v q = (Finset.univ.filter (fun j : Fin q.length => q[j.val] = .inr v)).card := by
  rw [varOcc, List.countP_eq_length_filter]
  have hof : q = (List.finRange q.length).map (fun j => q[j.val]) := by
    apply List.ext_getElem
    · simp
    · intro i h1 h2; simp
  conv_lhs => rw [hof]
  rw [List.filter_map, List.length_map]
  dsimp only [Function.comp_def]
  have hnd : ((List.finRange q.length).filter (fun j : Fin q.length =>
      match q[j.val] with | .inl _ => false | .inr y => decide (y = v))).Nodup :=
    (List.nodup_finRange q.length).filter _
  rw [← List.toFinset_card_of_nodup hnd]
  congr 1
  ext j
  simp only [List.mem_toFinset, List.mem_filter, List.mem_finRange, true_and,
    Finset.mem_filter, Finset.mem_univ]
  cases q[j.val] <;> simp

/-- For a length-`(q.length + 1)` word in `L_{ZMod 2}(q)`, some run end `b` has all its defining
equations satisfied by the word. -/
private lemma inLangNE_gap_one_satisfies_some_runEnd {V : Type*} [DecidableEq V]
    (q : Pattern (ZMod 2) V) (x : Fin (q.length + 1) → ZMod 2)
    (hx : InLangNE q (List.ofFn x)) :
    ∃ b : Fin q.length, isRunEnd q b ∧ ∀ eq ∈ eqsOfRunEnd q b, eq.1 x = eq.2 := by
  obtain ⟨σ, hσ_ne, hσ_eq⟩ := hx
  have hlen_w : (applySub σ q).length = q.length + 1 := by rw [hσ_eq, List.length_ofFn]
  have hlen_q := length_eq_termCount_add_sum_varOcc q
  have hlen_sub := applySub_length σ q
  -- Compute excess length: `∑_{v ∈ vars q} varOcc v q * ((σ v).length - 1) = 1`.
  have hsum_excess : ∑ v ∈ vars q, varOcc v q * ((σ v).length - 1) = 1 := by
    have hsplit : ∑ v ∈ vars q, varOcc v q * (σ v).length =
        (∑ v ∈ vars q, varOcc v q) + ∑ v ∈ vars q, varOcc v q * ((σ v).length - 1) := by
      rw [← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl fun v _ => ?_
      have hpos : 1 ≤ (σ v).length := List.length_pos_iff.mpr (hσ_ne v)
      have heq : 1 + ((σ v).length - 1) = (σ v).length := by omega
      calc varOcc v q * (σ v).length = varOcc v q * (1 + ((σ v).length - 1)) := by rw [heq]
        _ = varOcc v q + varOcc v q * ((σ v).length - 1) := by ring
    omega
  -- Therefore there is a unique variable `v₀ ∈ vars q` with `varOcc v₀ q = 1`, `(σ v₀).length = 2`,
  -- and every other variable `v ∈ vars q` has `(σ v).length = 1`.
  have hv0_ex : ∃ v₀ ∈ vars q, varOcc v₀ q = 1 ∧ (σ v₀).length = 2 ∧
      ∀ v ∈ vars q, v ≠ v₀ → (σ v).length = 1 := by
    obtain ⟨v₀, hv₀, hpos⟩ : ∃ v₀ ∈ vars q, 0 < varOcc v₀ q * ((σ v₀).length - 1) := by
      by_contra hall
      push_neg at hall
      have h0 : ∑ v ∈ vars q, varOcc v q * ((σ v).length - 1) = 0 :=
        Finset.sum_eq_zero fun v hv => Nat.eq_zero_of_le_zero (hall v hv)
      omega
    have hterm_le : varOcc v₀ q * ((σ v₀).length - 1) ≤ 1 :=
      le_trans
        (Finset.single_le_sum (f := fun v => varOcc v q * ((σ v).length - 1))
          (fun _ _ => Nat.zero_le _) hv₀)
        (le_of_eq hsum_excess)
    have hterm_eq : varOcc v₀ q * ((σ v₀).length - 1) = 1 := by omega
    have hocc1 : varOcc v₀ q = 1 := (mul_eq_one.mp hterm_eq).1
    have hlen2 : (σ v₀).length = 2 := by
      have := (mul_eq_one.mp hterm_eq).2
      omega
    refine ⟨v₀, hv₀, hocc1, hlen2, fun v hv hne => ?_⟩
    have hpair_le : varOcc v₀ q * ((σ v₀).length - 1) + varOcc v q * ((σ v).length - 1) ≤ 1 := by
      have hsub2 : ({v₀, v} : Finset V) ⊆ vars q := by
        intro z hz; simp only [Finset.mem_insert, Finset.mem_singleton] at hz
        rcases hz with rfl | rfl <;> assumption
      have hle := Finset.sum_le_sum_of_subset_of_nonneg (f := fun z => varOcc z q * ((σ z).length - 1))
        hsub2 (fun _ _ _ => Nat.zero_le _)
      rw [Finset.sum_pair (Ne.symm hne)] at hle
      exact le_trans hle (le_of_eq hsum_excess)
    have hv_occ : 1 ≤ varOcc v q := varOcc_pos_iff.mpr hv
    have hv_len : 1 ≤ (σ v).length := List.length_pos_iff.mpr (hσ_ne v)
    have hmul0 : varOcc v q * ((σ v).length - 1) = 0 := by omega
    have hsub0 : (σ v).length - 1 = 0 := (Nat.mul_eq_zero.mp hmul0).resolve_left (by omega)
    omega
  obtain ⟨v₀, hv₀, hocc1, hlen2, hother1⟩ := hv0_ex
  -- Find the unique position `k : Fin q.length` of `v₀` in `q`.
  obtain ⟨kval, hkval, hq_k⟩ := List.getElem_of_mem (mem_vars_iff.mp hv₀)
  let k : Fin q.length := ⟨kval, hkval⟩
  have hsk : isSingletonPos q k := ⟨v₀, hq_k, hocc1⟩
  obtain ⟨b, hkb, hbe, hrun⟩ := exists_runEnd_ge q k hsk
  -- Split `q` at `k`: `q = q₁ ++ [.inr v₀] ++ q₂`, with `q₁.length = k.val` and `v₀ ∉ vars q₁`, `v₀ ∉ vars q₂`.
  let q₁ := q.take k.val
  let q₂ := q.drop (k.val + 1)
  have hq_split : q = q₁ ++ [.inr v₀] ++ q₂ := by
    have h1 := List.take_append_drop k.val q
    have hdrop : q.drop k.val = q[k.val] :: q.drop (k.val + 1) := List.drop_eq_getElem_cons hkval
    rw [hdrop, hq_k] at h1
    simpa [q₁, q₂, List.append_assoc] using h1.symm
  have hq1_len : q₁.length = k.val := by simp [q₁]
  have hq2_len : q₂.length = q.length - (k.val + 1) := by simp [q₂]
  have hv0_not_q1q2 : v₀ ∉ vars q₁ ∧ v₀ ∉ vars q₂ := by
    have hocc_split : varOcc v₀ q = varOcc v₀ q₁ + 1 + varOcc v₀ q₂ := by
      conv_lhs => rw [hq_split]
      simp only [varOcc, List.countP_append, List.countP_cons, List.countP_nil, decide_true,
        ite_true]
    have h1 : varOcc v₀ q₁ = 0 := by omega
    have h2 : varOcc v₀ q₂ = 0 := by omega
    exact ⟨fun h => by have := varOcc_pos_iff.mpr h; omega,
           fun h => by have := varOcc_pos_iff.mpr h; omega⟩
  have hq1_one : ∀ v ∈ vars q₁, (σ v).length = 1 := fun v hv =>
    hother1 v (by rw [hq_split, vars_append, vars_append]; simp [hv])
      (fun heq => hv0_not_q1q2.1 (heq ▸ hv))
  have hq2_one : ∀ v ∈ vars q₂, (σ v).length = 1 := fun v hv =>
    hother1 v (by rw [hq_split, vars_append, vars_append]; simp [hv])
      (fun heq => hv0_not_q1q2.2 (heq ▸ hv))
  have h1_info := applySub_getElem_of_all_one σ q₁ hσ_ne hq1_one
  have h2_info := applySub_getElem_of_all_one σ q₂ hσ_ne hq2_one
  -- Determine `x (shiftIdx k j)` for every `j ≠ k`!
  have hx_shift_k : ∀ j : Fin q.length, j ≠ k →
      x (shiftIdx k j) =
        match q[j.val] with
        | .inl a => a
        | .inr v => (σ v)[0]'(List.length_pos_iff.mpr (hσ_ne v)) := by
    intro j hjk
    have hget_x : x (shiftIdx k j) = (applySub σ q)[(shiftIdx k j).val]'(by rw [hlen_w]; exact (shiftIdx k j).2) := by
      have := congrArg (fun l => l[(shiftIdx k j).val]?) hσ_eq
      simp only [List.getElem?_ofFn, (shiftIdx k j).2, dite_true,
        List.getElem?_eq_getElem (by rw [hlen_w]; exact (shiftIdx k j).2), Option.some.injEq] at this
      exact this.symm
    rw [hget_x]
    have hw_split : applySub σ q = applySub σ q₁ ++ σ v₀ ++ applySub σ q₂ := by
      conv_lhs => rw [hq_split]
      simp [applySub, List.append_assoc]
    by_cases hlt : j.val < k.val
    · have hsh : (shiftIdx k j).val = j.val := by simp [shiftIdx, Nat.le_of_lt hlt]
      have hq1_get : q₁[j.val]'(by omega) = q[j.val] := by simp [q₁]
      have hl1 : (shiftIdx k j).val < (applySub σ q₁).length := by omega
      have hstep : (applySub σ q)[(shiftIdx k j).val]'(by omega) =
          (applySub σ q₁)[j.val]'(by omega) := by
        simp only [hw_split, hsh, List.append_assoc]
        rw [List.getElem_append_left (by omega)]
      rw [hstep, h1_info.2 j.val (by omega)]
      simp only [hq1_get]
    · have hgt : k.val < j.val := by
        have : j.val ≠ k.val := fun h => hjk (Fin.ext h)
        omega
      have hsh : (shiftIdx k j).val = j.val + 1 := by
        simp [shiftIdx, show ¬ (j.val ≤ k.val) by omega]
      have hq2_get : q₂[j.val - (k.val + 1)]'(by omega) = q[j.val] := by
        simp only [q₂, List.getElem_drop]
        congr 1; omega
      have hstep : (applySub σ q)[(shiftIdx k j).val]'(by omega) =
          (applySub σ q₂)[j.val - (k.val + 1)]'(by omega) := by
        simp only [hw_split, hsh]
        rw [List.getElem_append_right (by simp [h1_info.1, hlen2]; omega)]
        congr 1
        simp [h1_info.1, hlen2]; omega
      rw [hstep, h2_info.2 (j.val - (k.val + 1)) (by omega)]
      simp only [hq2_get]
  -- Key observation: for every NON-singleton position `j`, `shiftIdx b j = shiftIdx k j` and `j ≠ k`!
  have hshift_eq : ∀ j : Fin q.length, ¬ isSingletonPos q j →
      j ≠ k ∧ shiftIdx b j = shiftIdx k j := by
    intro j hns
    have hjk : j ≠ k := fun h => hns (h ▸ hsk)
    refine ⟨hjk, ?_⟩
    by_cases hlt : j.val < k.val
    · have h1 : j.val ≤ b.val := by omega
      have h2 : j.val ≤ k.val := by omega
      simp [shiftIdx, h1, h2]
    · have hgt : b.val < j.val := by
        by_contra hle
        push_neg at hle
        exact hns (hrun j (by omega) hle)
      simp [shiftIdx, show ¬ (j.val ≤ b.val) by omega, show ¬ (j.val ≤ k.val) by omega]
  refine ⟨b, hbe, ?_⟩
  intro eq heq
  simp only [eqsOfRunEnd, Finset.mem_union, Finset.mem_biUnion, Finset.mem_univ, true_and] at heq
  rcases heq with ⟨j, hj⟩ | ⟨pr, hpr⟩
  · cases hqj : q[j.val] with
    | inl a =>
      simp only [hqj, Finset.mem_singleton] at hj
      subst hj
      have hns : ¬ isSingletonPos q j := by rintro ⟨v, hv, _⟩; simp [hqj] at hv
      obtain ⟨hjk, hsh⟩ := hshift_eq j hns
      simp only [coordProj, LinearMap.proj_apply, hsh, hx_shift_k j hjk, hqj]
    | inr v =>
      simp [hqj] at hj
  · split_ifs at hpr with hne
    · cases hq1 : q[pr.1.val] with
      | inl a => simp [hq1] at hpr
      | inr v₁ =>
        cases hq2 : q[pr.2.val] with
        | inl a => simp [hq1, hq2] at hpr
        | inr v₂ =>
          simp only [hq1, hq2] at hpr
          split_ifs at hpr with hv
          · subst hv
            simp only [Finset.mem_singleton] at hpr
            obtain ⟨_, rfl⟩ := hpr
            -- Since `pr.1 ≠ pr.2` both carry `v₁`, `varOcc v₁ q ≥ 2`, so neither is a singleton!
            have hocc2 : 2 ≤ varOcc v₁ q := by
              rw [varOcc_eq_card_filter]
              have hcard_le : ({pr.1, pr.2} : Finset (Fin q.length)).card ≤
                  (Finset.univ.filter (fun j : Fin q.length => q[j.val] = .inr v₁)).card := by
                refine Finset.card_le_card fun j hj => ?_
                simp only [Finset.mem_insert, Finset.mem_singleton] at hj
                simp only [Finset.mem_filter, Finset.mem_univ, true_and]
                rcases hj with rfl | rfl <;> simp [hq1, hq2]
              rwa [Finset.card_pair hne] at hcard_le
            have hns1 : ¬ isSingletonPos q pr.1 := by
              rintro ⟨w, hw, h1⟩; rw [hq1] at hw; cases hw; omega
            have hns2 : ¬ isSingletonPos q pr.2 := by
              rintro ⟨w, hw, h1⟩; rw [hq2] at hw; cases hw; omega
            obtain ⟨hjk1, hsh1⟩ := hshift_eq pr.1 hns1
            obtain ⟨hjk2, hsh2⟩ := hshift_eq pr.2 hns2
            simp only [LinearMap.sub_apply, coordProj, LinearMap.proj_apply,
              hsh1, hsh2, hx_shift_k pr.1 hjk1, hx_shift_k pr.2 hjk2, hq1, hq2, sub_self]
          · simp at hpr
    · simp at hpr

/-- General finite-dimensional annihilator rank formula: for any finite set `S` of linear
functionals on a finite-dimensional space `E`, `finrank (span S) + finrank (⋂ φ ∈ S, ker φ) = finrank E`. -/
private lemma finrank_span_dual_add_ker {K E : Type} [Field K] [AddCommGroup E] [Module K E]
    [FiniteDimensional K E] (S : Finset (Module.Dual K E)) :
    Module.finrank K (Submodule.span K (S : Set (Module.Dual K E))) +
    Module.finrank K (⨅ φ ∈ S, LinearMap.ker φ : Submodule K E) = Module.finrank K E := by
  let U : Submodule K (Module.Dual K E) := Submodule.span K (S : Set (Module.Dual K E))
  change Module.finrank K U + Module.finrank K (⨅ φ ∈ S, LinearMap.ker φ : Submodule K E) =
    Module.finrank K E
  haveI : Module.Free K U := Module.Free.of_divisionRing K U
  haveI : FiniteDimensional K U := FiniteDimensional.span_of_finite K S.finite_toSet
  let b := Module.finBasis K U
  let ev : E →ₗ[K] (Fin (Module.finrank K U) → K) :=
    { toFun := fun x i => (b i).1 x
      map_add' := fun x y => by funext i; exact (b i).1.map_add x y
      map_smul' := fun c x => by funext i; exact (b i).1.map_smul c x }
  have hker : LinearMap.ker ev = (⨅ φ ∈ S, LinearMap.ker φ : Submodule K E) := by
    ext x
    simp only [LinearMap.mem_ker, Submodule.mem_iInf]
    constructor
    · intro hx φ hφ
      have hbi : ∀ i, (b i).1 x = 0 := fun i => congrFun hx i
      have hrepr := (b.sum_repr ⟨φ, Submodule.subset_span hφ⟩).symm
      have hval : φ = ∑ i, b.repr ⟨φ, Submodule.subset_span hφ⟩ i • (b i).1 := by
        have h := congrArg Subtype.val hrepr
        simp only [AddSubmonoidClass.coe_finset_sum, SetLike.val_smul] at h
        exact h
      rw [hval, LinearMap.sum_apply]
      exact Finset.sum_eq_zero fun i _ => by simp [hbi i]
    · intro hx
      funext i
      have hmem : (b i).1 ∈ U := (b i).2
      refine Submodule.span_induction (p := fun ψ _ => ψ x = 0) ?_ ?_ ?_ ?_ hmem
      · intro ψ hψ; exact hx ψ hψ
      · simp
      · intro ψ₁ ψ₂ _ _ h1 h2; simp [h1, h2]
      · intro c ψ _ h1; simp [h1]
  have hsurj : Function.Surjective ev := by
    obtain ⟨V', hcompl⟩ := Submodule.exists_isCompl U
    let e := Module.evalEquiv K E
    intro c
    let f : U →ₗ[K] K := b.constr K c
    let F : Module.Dual K (Module.Dual K E) := f.comp (U.linearProjOfIsCompl V' hcompl)
    refine ⟨e.symm F, ?_⟩
    funext i
    have h1 : ev (e.symm F) i = (e (e.symm F)) (b i).1 := rfl
    rw [h1, e.apply_symm_apply]
    change f ((U.linearProjOfIsCompl V' hcompl) (b i).1) = c i
    have hproj : (U.linearProjOfIsCompl V' hcompl) (b i).1 = b i :=
      Submodule.linearProjOfIsCompl_apply_left hcompl (b i)
    rw [hproj]
    exact b.constr_basis K c i
  have hrn := LinearMap.finrank_range_add_finrank_ker ev
  rw [LinearMap.range_eq_top.mpr hsurj, finrank_top, Module.finrank_pi_fintype, hker] at hrn
  simp only [Module.finrank_self, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    smul_eq_mul, mul_one] at hrn
  exact hrn

/-- Rank-nullity bound on a submodule: `finrank U ≤ finrank (U.map res) + finrank (ker res)`. -/
private lemma finrank_le_finrank_map_add_ker {K E F : Type} [Field K]
    [AddCommGroup E] [Module K E] [FiniteDimensional K E]
    [AddCommGroup F] [Module K F] [FiniteDimensional K F]
    (res : E →ₗ[K] F) (U : Submodule K E) :
    Module.finrank K U ≤ Module.finrank K (U.map res) + Module.finrank K (LinearMap.ker res) := by
  let f : U →ₗ[K] F := res.comp U.subtype
  have hfrange : LinearMap.range f = U.map res := by
    ext z
    simp only [f, LinearMap.mem_range, LinearMap.comp_apply, Submodule.coe_subtype,
      Submodule.mem_map, Subtype.exists, exists_prop]
  have hfker : Module.finrank K (LinearMap.ker f) ≤ Module.finrank K (LinearMap.ker res) := by
    let g : LinearMap.ker f →ₗ[K] LinearMap.ker res :=
      { toFun := fun x => ⟨x.1.1, x.2⟩
        map_add' := fun _ _ => rfl
        map_smul' := fun _ _ => rfl }
    have hg_inj : Function.Injective g := fun x y h =>
      Subtype.ext (Subtype.ext (congrArg (fun z : LinearMap.ker res => z.1) h))
    exact LinearMap.finrank_le_finrank_of_injective hg_inj
  have hrn := LinearMap.finrank_range_add_finrank_ker f
  rw [hfrange] at hrn
  omega

/-- Surjectivity of evaluation along linearly independent dual functionals. -/
private lemma surjective_eval_of_linearIndependent {K E ι : Type} [Field K] [AddCommGroup E]
    [Module K E] [FiniteDimensional K E] [Fintype ι] [DecidableEq ι]
    (u : ι → Module.Dual K E) (hu_li : LinearIndependent K u) :
    Function.Surjective (fun (v : E) (b : ι) => u b v) := by
  classical
  let evalMap : E →ₗ[K] (ι → K) :=
    { toFun := fun v b => u b v
      map_add' := fun v₁ v₂ => by funext b; exact (u b).map_add v₁ v₂
      map_smul' := fun c v => by funext b; exact (u b).map_smul c v }
  let S_u : Finset (Module.Dual K E) := Finset.univ.image u
  have hker_eq : LinearMap.ker evalMap = (⨅ φ ∈ S_u, LinearMap.ker φ : Submodule K E) := by
    ext v
    simp only [LinearMap.mem_ker, Submodule.mem_iInf, S_u, Finset.mem_image, Finset.mem_univ,
      true_and, forall_exists_index, forall_apply_eq_imp_iff, evalMap, LinearMap.coe_mk,
      AddHom.coe_mk, funext_iff, Pi.zero_apply]
  have hspan_dim : Module.finrank K (Submodule.span K (S_u : Set (Module.Dual K E))) =
      Fintype.card ι := by
    have hset : (S_u : Set (Module.Dual K E)) = Set.range u := by
      ext φ; simp [S_u]
    rw [hset, finrank_span_eq_card hu_li]
  have hdual := finrank_span_dual_add_ker S_u
  rw [hspan_dim, ← hker_eq] at hdual
  have hrn := LinearMap.finrank_range_add_finrank_ker evalMap
  have hrange_dim : Module.finrank K (LinearMap.range evalMap) =
      Module.finrank K (ι → K) := by
    rw [Module.finrank_pi_fintype]
    simp only [Module.finrank_self, Finset.sum_const, Finset.card_univ, smul_eq_mul, mul_one]
    omega
  change Function.Surjective evalMap
  rw [← LinearMap.range_eq_top]
  exact Submodule.eq_top_of_finrank_eq hrange_dim

/-- Dimension of the direction space of a pattern's shortest-word affine subspace equals `varnum`. -/
private lemma finrank_ker_runEnd {V : Type*} [DecidableEq V] (q : Pattern (ZMod 2) V)
    (b : Fin q.length) :
    Module.finrank (ZMod 2)
      (Submodule.span (ZMod 2) (linOfRunEnd q b : Set (Module.Dual (ZMod 2) (Fin (q.length + 1) → ZMod 2)))) =
      defect q := by
  have h1 := finrank_span_dual_add_ker (linOfRunEnd q b)
  rw [Module.finrank_pi_fintype] at h1
  simp only [Module.finrank_self, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    smul_eq_mul, mul_one] at h1
  set W : Submodule (ZMod 2) (Fin (q.length + 1) → ZMod 2) :=
    ⨅ φ ∈ linOfRunEnd q b, LinearMap.ker φ
  have hmemW : ∀ x : Fin (q.length + 1) → ZMod 2, x ∈ W ↔
      (∀ (j : Fin q.length) (a : ZMod 2), q[j.val] = .inl a → x (shiftIdx b j) = 0) ∧
      (∀ (j₁ j₂ : Fin q.length) (v : V), q[j₁.val] = .inr v → q[j₂.val] = .inr v →
        x (shiftIdx b j₁) = x (shiftIdx b j₂)) := by
    intro x
    simp only [W, Submodule.mem_iInf, LinearMap.mem_ker, linOfRunEnd, Finset.mem_image]
    constructor
    · intro hx
      refine ⟨fun j a hqj => ?_, fun j₁ j₂ v hq1 hq2 => ?_⟩
      · have hmem : (coordProj (q.length + 1) (shiftIdx b j), a) ∈ eqsOfRunEnd q b := by
          simp only [eqsOfRunEnd, Finset.mem_union, Finset.mem_biUnion, Finset.mem_univ, true_and]
          left; exact ⟨j, by simp [hqj]⟩
        exact hx _ ⟨_, hmem, rfl⟩
      · by_cases heq : j₁ = j₂
        · rw [heq]
        · have hmem : (coordProj (q.length + 1) (shiftIdx b j₁) -
              coordProj (q.length + 1) (shiftIdx b j₂), 0) ∈ eqsOfRunEnd q b := by
            simp only [eqsOfRunEnd, Finset.mem_union, Finset.mem_biUnion, Finset.mem_univ, true_and]
            right; exact ⟨(j₁, j₂), by simp [heq, hq1, hq2]⟩
          have hsub0 := hx _ ⟨_, hmem, rfl⟩
          exact sub_eq_zero.mp hsub0
    · rintro ⟨hterm, hvar⟩ φ ⟨eq, heq, rfl⟩
      simp only [eqsOfRunEnd, Finset.mem_union, Finset.mem_biUnion, Finset.mem_univ, true_and] at heq
      rcases heq with ⟨j, hj⟩ | ⟨pr, hpr⟩
      · cases hqj : q[j.val] with
        | inl a =>
          simp only [hqj, Finset.mem_singleton] at hj
          subst hj
          exact hterm j a hqj
        | inr v => simp [hqj] at hj
      · split_ifs at hpr with hne
        · cases hq1 : q[pr.1.val] with
          | inl a => simp [hq1] at hpr
          | inr v₁ =>
            cases hq2 : q[pr.2.val] with
            | inl a => simp [hq1, hq2] at hpr
            | inr v₂ =>
              simp only [hq1, hq2] at hpr
              split_ifs at hpr with hv
              · subst hv
                simp only [Finset.mem_singleton] at hpr
                obtain ⟨_, rfl⟩ := hpr
                simp only [LinearMap.sub_apply, coordProj, LinearMap.proj_apply]
                exact sub_eq_zero.mpr (hvar pr.1 pr.2 v₁ hq1 hq2)
              · simp at hpr
        · simp at hpr
  -- Build a linear equivalence `(vars q → ZMod 2) × ZMod 2 ≃ₗ[ZMod 2] W`.
  let unshift (k : Fin (q.length + 1)) : Option (Fin q.length) :=
    if h1 : k.val ≤ b.val then some ⟨k.val, by omega⟩
    else if h2 : k.val = b.val + 1 then none
    else some ⟨k.val - 1, by omega⟩
  have hunshift_shift : ∀ j : Fin q.length, unshift (shiftIdx b j) = some j := by
    intro j
    dsimp only [shiftIdx]
    by_cases hj : j.val ≤ b.val
    · simp only [hj, ite_true, unshift, dite_true]
    · simp only [hj, ite_false, unshift]
      have h1 : ¬ (j.val + 1 ≤ b.val) := by omega
      have h2 : ¬ (j.val + 1 = b.val + 1) := by omega
      simp only [h1, h2, dite_false, Option.some.injEq]
      exact Fin.ext (Nat.add_sub_cancel j.val 1)
  have hunshift_ins : unshift ⟨b.val + 1, by omega⟩ = none := by
    simp [unshift]
  have hshift_unshift : ∀ k : Fin (q.length + 1), ∀ j : Fin q.length,
      unshift k = some j → shiftIdx b j = k := by
    intro k j hkj
    dsimp only [unshift] at hkj
    by_cases h1 : k.val ≤ b.val
    · simp only [h1, dite_true, Option.some.injEq] at hkj
      subst hkj
      simp [shiftIdx, h1]
    · simp only [h1, dite_false] at hkj
      by_cases h2 : k.val = b.val + 1
      · simp [h2] at hkj
      · simp only [h2, dite_false, Option.some.injEq] at hkj
        subst hkj
        simp only [shiftIdx]
        have hgt : ¬ (k.val - 1 ≤ b.val) := by omega
        simp only [hgt, ite_false]
        exact Fin.ext (by dsimp; omega)
  let evalTok (f : vars q → ZMod 2) : Token (ZMod 2) V → ZMod 2
    | .inl _ => 0
    | .inr v => if hv : v ∈ vars q then f ⟨v, hv⟩ else 0
  let Φ : ((vars q → ZMod 2) × ZMod 2) →ₗ[ZMod 2] W :=
    { toFun := fun ⟨f, c⟩ =>
        ⟨fun k =>
          match unshift k with
          | none => c
          | some j => evalTok f q[j.val],
        (hmemW _).mpr ⟨
          fun j a hqj => by simp [hunshift_shift, hqj, evalTok],
          fun j₁ j₂ v hq1 hq2 => by simp [hunshift_shift, hq1, hq2, evalTok]⟩⟩
      map_add' := fun ⟨f₁, c₁⟩ ⟨f₂, c₂⟩ => by
        apply Subtype.ext; funext k
        dsimp
        cases unshift k with
        | none => rfl
        | some j =>
          cases hqj : q[j.val] with
          | inl a => simp [evalTok, hqj]
          | inr v => simp only [evalTok, hqj]; split_ifs <;> simp
      map_smul' := fun r ⟨f, c⟩ => by
        apply Subtype.ext; funext k
        dsimp
        cases unshift k with
        | none => rfl
        | some j =>
          cases hqj : q[j.val] with
          | inl a => simp [evalTok, hqj]
          | inr v => simp only [evalTok, hqj]; split_ifs <;> simp }
  have hΦ_bij : Function.Bijective Φ := by
    constructor
    · intro ⟨f₁, c₁⟩ ⟨f₂, c₂⟩ heq
      have hval := congrArg Subtype.val heq
      have hc : c₁ = c₂ := by
        have := congrFun hval ⟨b.val + 1, by omega⟩
        simpa [Φ, hunshift_ins] using this
      have hf : f₁ = f₂ := by
        funext ⟨v, hv⟩
        obtain ⟨jval, hjval, hqj⟩ := List.getElem_of_mem (mem_vars_iff.mp hv)
        have := congrFun hval (shiftIdx b ⟨jval, hjval⟩)
        simpa [Φ, hunshift_shift, hqj, evalTok, hv] using this
      exact Prod.ext hf hc
    · intro ⟨x, hx⟩
      have hx' := (hmemW x).mp hx
      let f : vars q → ZMod 2 := fun ⟨v, hv⟩ =>
        let idx := Classical.choose (List.getElem_of_mem (mem_vars_iff.mp hv))
        let hidx := Classical.choose_spec (List.getElem_of_mem (mem_vars_iff.mp hv))
        x (shiftIdx b ⟨idx, hidx.1⟩)
      let c : ZMod 2 := x ⟨b.val + 1, by omega⟩
      refine ⟨(f, c), Subtype.ext ?_⟩
      funext k
      dsimp [Φ]
      cases hk : unshift k with
      | none =>
        have hkval : k = ⟨b.val + 1, by omega⟩ := by
          dsimp only [unshift] at hk
          by_cases h1 : k.val ≤ b.val
          · simp [h1] at hk
          · simp only [h1, dite_false] at hk
            by_cases h2 : k.val = b.val + 1
            · exact Fin.ext h2
            · simp [h2] at hk
        rw [hkval]
      | some j =>
        have hsh := hshift_unshift k j hk
        cases hqj : q[j.val] with
        | inl a =>
          rw [← hsh]
          simp only [evalTok, hqj]
          exact (hx'.1 j a hqj).symm
        | inr v =>
          have hv : v ∈ vars q := mem_vars_iff.mpr (by rw [← hqj]; exact List.getElem_mem j.2)
          simp only [evalTok, hqj, hv, dite_true, f]
          set hspec := Classical.choose_spec (List.getElem_of_mem (mem_vars_iff.mp hv))
          rw [← hsh]
          exact hx'.2 ⟨Classical.choose _, hspec.1⟩ j v hspec.2 hqj
  have hW_dim : Module.finrank (ZMod 2) W = varnum q + 1 := by
    have heq := (LinearEquiv.ofBijective Φ hΦ_bij).finrank_eq
    rw [← heq, Module.finrank_prod, Module.finrank_pi_fintype]
    simp [varnum]
  have hvq := varnum_le_length q
  rw [hW_dim] at h1
  simp only [defect]
  omega

/-- Key rank lower bound in Theorem 2: for every nonempty subset `I` of run ends of `q`,
`defect q + I.card - 1 ≤ finrank (span (⋃ b ∈ I, linOfRunEnd q b))`. -/
private lemma rank_lower_bound_runEnds {V : Type*} [DecidableEq V] (q : Pattern (ZMod 2) V) :
    ∀ n : ℕ, ∀ I : Finset (Fin q.length), I.card = n → I.Nonempty →
      (∀ b ∈ I, isRunEnd q b) →
      defect q + I.card - 1 ≤
        Module.finrank (ZMod 2)
          (Submodule.span (ZMod 2)
            (⋃ b ∈ I, (linOfRunEnd q b : Set (Module.Dual (ZMod 2) (Fin (q.length + 1) → ZMod 2))))) := by
  intro n
  induction n with
  | zero =>
    intro I hcard hne _
    exact (hne.ne_empty (Finset.card_eq_zero.mp hcard)).elim
  | succ m ih =>
    intro I hcard hne hrun
    by_cases hm : m = 0
    · subst hm
      obtain ⟨b, rfl⟩ := Finset.card_eq_one.mp hcard
      have hunion : (⋃ b₁ ∈ ({b} : Finset (Fin q.length)),
          (linOfRunEnd q b₁ : Set (Module.Dual (ZMod 2) (Fin (q.length + 1) → ZMod 2)))) =
          (linOfRunEnd q b : Set (Module.Dual (ZMod 2) (Fin (q.length + 1) → ZMod 2))) := by
        ext φ; simp
      simp only [Finset.card_singleton, Nat.add_sub_cancel]
      rw [hunion, finrank_ker_runEnd q b]
    · -- Let `b_max` be the largest element of `I`, and `b_prev` the largest of `I.erase b_max`.
      let b_max := I.max' hne
      have hb_max_mem : b_max ∈ I := Finset.max'_mem I hne
      let I' := I.erase b_max
      have hI'_card : I'.card = m := by
        rw [Finset.card_erase_of_mem hb_max_mem, hcard]; omega
      have hI'_ne : I'.Nonempty := Finset.card_pos.mp (by omega)
      let b_prev := I'.max' hI'_ne
      have hb_prev_mem : b_prev ∈ I' := Finset.max'_mem I' hI'_ne
      have hb_prev_lt : b_prev < b_max := by
        have hle := Finset.le_max' I b_prev (Finset.mem_of_mem_erase hb_prev_mem)
        have hne' : b_prev ≠ b_max := (Finset.mem_erase.mp hb_prev_mem).1
        exact lt_of_le_of_ne hle hne'
      have hh_lt : b_prev.val + 1 < q.length := by
        have := b_max.2
        have : b_prev.val < b_max.val := hb_prev_lt
        omega
      let h : Fin q.length := ⟨b_prev.val + 1, hh_lt⟩
      let h_n : Fin (q.length + 1) := ⟨b_prev.val + 1, by omega⟩
      -- Position `h` is NOT a singleton (since `b_prev` is a run end).
      have hh_not_sing : ¬ isSingletonPos q h :=
        (hrun b_prev (Finset.mem_of_mem_erase hb_prev_mem)).2 hh_lt
      -- Every functional in `linOfRunEnd q b` for `b ∈ I'` vanishes on `Pi.single h_n 1`!
      have hvanish : ∀ b ∈ I', ∀ φ ∈ linOfRunEnd q b, φ (Pi.single h_n (1 : ZMod 2)) = 0 := by
        intro b hb φ hφ
        have hb_le : b ≤ b_prev := Finset.le_max' I' b hb
        have hnot_eq : ∀ j : Fin q.length, ¬ isSingletonPos q j → shiftIdx b j ≠ h_n := by
          intro j hns heq
          have hval := congrArg Fin.val heq
          simp only [shiftIdx, h_n] at hval
          split_ifs at hval with hjb
          · dsimp at hval
            have : b.val ≤ b_prev.val := hb_le
            omega
          · dsimp at hval
            have hj_eq : j = b_prev := Fin.ext (by omega)
            rw [hj_eq] at hns
            exact hns (hrun b_prev (Finset.mem_of_mem_erase hb_prev_mem)).1
        rw [linOfRunEnd, Finset.mem_image] at hφ
        obtain ⟨eq, heq, rfl⟩ := hφ
        simp only [eqsOfRunEnd, Finset.mem_union, Finset.mem_biUnion, Finset.mem_univ, true_and] at heq
        rcases heq with ⟨j, hj⟩ | ⟨pr, hpr⟩
        · cases hqj : q[j.val] with
          | inl a =>
            simp only [hqj, Finset.mem_singleton] at hj
            subst hj
            have hns : ¬ isSingletonPos q j := by rintro ⟨v, hv, _⟩; simp [hqj] at hv
            simp only [coordProj, LinearMap.proj_apply]
            exact Pi.single_eq_of_ne (hnot_eq j hns) _
          | inr v => simp [hqj] at hj
        · split_ifs at hpr with hne_pr
          · cases hq1 : q[pr.1.val] with
            | inl a => simp [hq1] at hpr
            | inr v₁ =>
              cases hq2 : q[pr.2.val] with
              | inl a => simp [hq1, hq2] at hpr
              | inr v₂ =>
                simp only [hq1, hq2] at hpr
                split_ifs at hpr with hv
                · subst hv
                  simp only [Finset.mem_singleton] at hpr
                  obtain ⟨_, rfl⟩ := hpr
                  -- Both `pr.1` and `pr.2` are non-singletons because `pr.1 ≠ pr.2` have the same variable `v₁`.
                  have hocc2 : 2 ≤ varOcc v₁ q := by
                    rw [varOcc_eq_card_filter]
                    have hle : ({pr.1, pr.2} : Finset (Fin q.length)).card ≤
                        (Finset.univ.filter (fun j : Fin q.length => q[j.val] = .inr v₁)).card := by
                      refine Finset.card_le_card fun j hj => ?_
                      simp only [Finset.mem_insert, Finset.mem_singleton] at hj
                      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
                      rcases hj with rfl | rfl <;> simp [hq1, hq2]
                    rwa [Finset.card_pair hne_pr] at hle
                  have hns1 : ¬ isSingletonPos q pr.1 := by
                    rintro ⟨w, hw, h1⟩; rw [hq1] at hw; cases hw; omega
                  have hns2 : ¬ isSingletonPos q pr.2 := by
                    rintro ⟨w, hw, h1⟩; rw [hq2] at hw; cases hw; omega
                  simp only [LinearMap.sub_apply, coordProj, LinearMap.proj_apply,
                    Pi.single_eq_of_ne (hnot_eq pr.1 hns1),
                    Pi.single_eq_of_ne (hnot_eq pr.2 hns2), sub_self]
                · simp at hpr
          · simp at hpr
      -- Conversely, `linOfRunEnd q b_max` contains a functional evaluating to `1` on `Pi.single h_n 1`!
      have hex_one : ∃ φ ∈ linOfRunEnd q b_max, φ (Pi.single h_n (1 : ZMod 2)) = 1 := by
        have hh_le_bmax : h.val ≤ b_max.val := by
          have : b_prev.val < b_max.val := hb_prev_lt
          dsimp [h]; omega
        have hsh_h : shiftIdx b_max h = h_n := by
          simp only [shiftIdx, hh_le_bmax, ite_true]
          rfl
        cases hqh : q[h.val] with
        | inl a =>
          refine ⟨coordProj (q.length + 1) h_n, ?_, by simp [coordProj]⟩
          rw [linOfRunEnd, Finset.mem_image]
          refine ⟨(coordProj (q.length + 1) h_n, a), ?_, rfl⟩
          simp only [eqsOfRunEnd, Finset.mem_union, Finset.mem_biUnion, Finset.mem_univ, true_and]
          left; exact ⟨h, by simp [hqh, hsh_h]⟩
        | inr v =>
          -- Since `h` is not a singleton, `varOcc v q ≠ 1`, and `0 < varOcc v q`, so `∃ h' ≠ h` with `q[h'] = .inr v`.
          have hv_mem : v ∈ vars q := mem_vars_iff.mpr (by rw [← hqh]; exact List.getElem_mem h.2)
          have hocc_ge2 : 2 ≤ varOcc v q := by
            have h1 := varOcc_pos_iff.mpr hv_mem
            have hne1 : varOcc v q ≠ 1 := fun h1' => hh_not_sing ⟨v, hqh, h1'⟩
            omega
          have hex_other : ∃ h' : Fin q.length, h' ≠ h ∧ q[h'.val] = .inr v := by
            by_contra hall
            push_neg at hall
            have hsub1 : Finset.univ.filter (fun j : Fin q.length => q[j.val] = .inr v) ⊆ {h} := by
              intro j hj
              simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hj
              simp only [Finset.mem_singleton]
              by_contra hne'
              exact hall j hne' hj
            have hle1 : varOcc v q ≤ 1 := by
              rw [varOcc_eq_card_filter]
              exact le_trans (Finset.card_le_card hsub1) (by simp)
            omega
          obtain ⟨h', hh'ne, hqh'⟩ := hex_other
          have hsh_ne : shiftIdx b_max h' ≠ h_n := by
            rw [← hsh_h]
            exact fun heq => hh'ne (shiftIdx_injective b_max heq)
          refine ⟨coordProj (q.length + 1) h_n - coordProj (q.length + 1) (shiftIdx b_max h'),
            ?_, ?_⟩
          · rw [linOfRunEnd, Finset.mem_image]
            refine ⟨(coordProj (q.length + 1) h_n - coordProj (q.length + 1) (shiftIdx b_max h'), 0),
              ?_, rfl⟩
            simp only [eqsOfRunEnd, Finset.mem_union, Finset.mem_biUnion, Finset.mem_univ, true_and]
            right
            exact ⟨(h, h'), by simp [Ne.symm hh'ne, hqh, hqh', hsh_h]⟩
          · simp only [LinearMap.sub_apply, coordProj, LinearMap.proj_apply,
              Pi.single_eq_same, Pi.single_eq_of_ne hsh_ne, sub_zero]
      -- Thus `span I' < span I`, so `finrank (span I') + 1 ≤ finrank (span I)`.
      set U' := Submodule.span (ZMod 2)
        (⋃ b ∈ I', (linOfRunEnd q b : Set (Module.Dual (ZMod 2) (Fin (q.length + 1) → ZMod 2))))
      set U := Submodule.span (ZMod 2)
        (⋃ b ∈ I, (linOfRunEnd q b : Set (Module.Dual (ZMod 2) (Fin (q.length + 1) → ZMod 2))))
      have hU'_le : U' ≤ U := Submodule.span_mono (Set.biUnion_subset_biUnion_left
        (fun b hb => Finset.mem_of_mem_erase hb))
      obtain ⟨φ₀, hφ₀_mem, hφ₀_one⟩ := hex_one
      have hφ₀_in_U : φ₀ ∈ U := Submodule.subset_span (Set.mem_biUnion hb_max_mem hφ₀_mem)
      have hφ₀_not_U' : φ₀ ∉ U' := by
        intro hmem
        have h0 : φ₀ (Pi.single h_n (1 : ZMod 2)) = 0 := by
          refine Submodule.span_induction (p := fun ψ _ => ψ (Pi.single h_n (1 : ZMod 2)) = 0)
            ?_ (by simp) (fun _ _ _ _ h1 h2 => by simp [h1, h2])
            (fun c _ _ h1 => by simp [h1]) hmem
          intro ψ hψ
          rw [Set.mem_iUnion₂] at hψ
          obtain ⟨b, hb, hψb⟩ := hψ
          exact hvanish b hb ψ hψb
        rw [hφ₀_one] at h0
        exact one_ne_zero h0
      have hlt_dim : Module.finrank (ZMod 2) U' < Module.finrank (ZMod 2) U :=
        Submodule.finrank_lt_finrank_of_lt (lt_of_le_of_ne hU'_le fun heq => hφ₀_not_U' (heq ▸ hφ₀_in_U))
      have hIH := ih I' hI'_card hI'_ne (fun b hb => hrun b (Finset.mem_of_mem_erase hb))
      rw [hI'_card] at hIH
      change defect q + m - 1 ≤ Module.finrank (ZMod 2) U' at hIH
      change defect q + I.card - 1 ≤ Module.finrank (ZMod 2) U
      rw [hcard]
      omega

/-- Direction space of a pattern's shortest-word affine subspace over `ZMod 2`. -/
private def patDirSpace {V₁ : Type*} (p : Pattern (ZMod 2) V₁) {n : ℕ} (hlen : p.length = n) :
    Submodule (ZMod 2) (Fin n → ZMod 2) :=
  { carrier := {v | (∀ (j : Fin n) (a : ZMod 2), p[j.val]'(by omega) = .inl a → v j = 0) ∧
      (∀ (j₁ j₂ : Fin n) (w : V₁), p[j₁.val]'(by omega) = .inr w → p[j₂.val]'(by omega) = .inr w →
        v j₁ = v j₂)}
    add_mem' := fun ha hb => ⟨fun j a h => by simp [ha.1 j a h, hb.1 j a h],
      fun j₁ j₂ w h1 h2 => by simp [ha.2 j₁ j₂ w h1 h2, hb.2 j₁ j₂ w h1 h2]⟩
    zero_mem' := ⟨fun _ _ _ => rfl, fun _ _ _ _ _ => rfl⟩
    smul_mem' := fun c _ ha => ⟨fun j a h => by simp [ha.1 j a h],
      fun j₁ j₂ w h1 h2 => by simp [ha.2 j₁ j₂ w h1 h2]⟩ }

private lemma finrank_patDirSpace {V₁ : Type*} [DecidableEq V₁]
    (p : Pattern (ZMod 2) V₁) {n : ℕ} (hlen : p.length = n) :
    Module.finrank (ZMod 2) (patDirSpace p hlen) = varnum p := by
  classical
  let evalTok (f : vars p → ZMod 2) : Token (ZMod 2) V₁ → ZMod 2
    | .inl _ => 0
    | .inr w => if hw : w ∈ vars p then f ⟨w, hw⟩ else 0
  let Ψ : (vars p → ZMod 2) →ₗ[ZMod 2] patDirSpace p hlen :=
    { toFun := fun f =>
        ⟨fun j => evalTok f (p[j.val]'(by omega)),
        ⟨fun j a hpj => by simp [evalTok, hpj],
         fun j₁ j₂ w h1 h2 => by simp [evalTok, h1, h2]⟩⟩
      map_add' := fun f₁ f₂ => by
        apply Subtype.ext; funext j; dsimp
        cases hpj : p[j.val]'(by omega) with
        | inl a => simp [evalTok]
        | inr w => simp only [evalTok]; split_ifs <;> simp
      map_smul' := fun c f => by
        apply Subtype.ext; funext j; dsimp
        cases hpj : p[j.val]'(by omega) with
        | inl a => simp [evalTok]
        | inr w => simp only [evalTok]; split_ifs <;> simp }
  have hΨ_bij : Function.Bijective Ψ := by
    constructor
    · intro f₁ f₂ heq
      have hval := congrArg Subtype.val heq
      funext ⟨w, hw⟩
      obtain ⟨j, hj, hpj⟩ := List.getElem_of_mem (mem_vars_iff.mp hw)
      have := congrFun hval ⟨j, by omega⟩
      simpa [Ψ, evalTok, hpj, hw] using this
    · intro ⟨v, hv⟩
      refine ⟨fun ⟨w, hw⟩ =>
        let hEx := List.getElem_of_mem (mem_vars_iff.mp hw)
        v ⟨Classical.choose hEx, by have := (Classical.choose_spec hEx).1; omega⟩, Subtype.ext ?_⟩
      funext j
      dsimp [Ψ]
      cases hpj : p[j.val]'(by omega) with
      | inl a => simp only [evalTok]; exact (hv.1 j a hpj).symm
      | inr w =>
        have hw : w ∈ vars p := mem_vars_iff.mpr (by rw [← hpj]; exact List.getElem_mem (by omega))
        simp only [evalTok, hw, dite_true]
        set hEx := List.getElem_of_mem (mem_vars_iff.mp hw)
        have hidx_lt : Classical.choose hEx < p.length := (Classical.choose_spec hEx).1
        exact hv.2 ⟨Classical.choose hEx, by omega⟩ j w (Classical.choose_spec hEx).2 hpj
  rw [← (LinearEquiv.ofBijective Ψ hΨ_bij).finrank_eq, Module.finrank_pi_fintype]
  simp [varnum]

/-- Core gap-one theorem over the binary alphabet `ZMod 2`. -/
private theorem gap_one_binary {V₁ V₂ : Type*} [DecidableEq V₁] [DecidableEq V₂]
    {p : Pattern (ZMod 2) V₁} {q : Pattern (ZMod 2) V₂}
    (hsub : LangSubsetNE p q) (hlen : p.length = q.length + 1) :
    defect q ≤ defect p := by
  classical
  set n := q.length + 1
  -- Define the base point `x₀ : Fin n → ZMod 2` (substituting `[0]` for every variable of `p`).
  let x₀ : Fin n → ZMod 2 := fun i =>
    match p[i.val]'(by omega) with
    | .inl a => a
    | .inr _ => 0
  let V_P : Submodule (ZMod 2) (Fin n → ZMod 2) := patDirSpace p hlen
  -- Every `x₀ + v` with `v ∈ V_P` belongs to `L_{ZMod 2}(p)`.
  have hx0_add_mem : ∀ v ∈ V_P, InLangNE p (List.ofFn (x₀ + v)) := by
    intro v hv
    let σ : V₁ → List (ZMod 2) := fun w =>
      if hw : w ∈ vars p then
        let hEx := List.getElem_of_mem (mem_vars_iff.mp hw)
        [v ⟨Classical.choose hEx, by have := (Classical.choose_spec hEx).1; omega⟩]
      else [0]
    have hσ_ne : ∀ w, σ w ≠ [] := by
      intro w; by_cases hw : w ∈ vars p <;> simp [σ, hw]
    refine ⟨σ, hσ_ne, ?_⟩
    have h1 := applySub_getElem_of_all_one σ p hσ_ne (fun w _ => by by_cases hw : w ∈ vars p <;> simp [σ, hw])
    apply List.ext_getElem
    · simp only [List.length_ofFn, h1.1, hlen, n]
    · intro j hj1 hj2
      rw [h1.2 j (by omega)]
      simp only [List.getElem_ofFn, Pi.add_apply, x₀]
      cases hpj : p[j]'(by omega) with
      | inl a =>
        have hv0 := hv.1 ⟨j, by omega⟩ a hpj
        simp [hv0]
      | inr w =>
        have hw : w ∈ vars p := mem_vars_iff.mpr (by rw [← hpj]; exact List.getElem_mem (by omega))
        simp only [σ, hw, dite_true, List.getElem_cons_zero, zero_add]
        set hEx := List.getElem_of_mem (mem_vars_iff.mp hw)
        have hidx_lt : Classical.choose hEx < p.length := (Classical.choose_spec hEx).1
        exact hv.2 ⟨Classical.choose hEx, by omega⟩ ⟨j, by omega⟩ w (Classical.choose_spec hEx).2 hpj
  have hVP_dim : Module.finrank (ZMod 2) V_P = varnum p := finrank_patDirSpace p hlen
  -- Suppose for contradiction that `defect p < defect q`.
  by_contra hlt
  push_neg at hlt
  -- Restriction map `res : Dual (Fin n → ZMod 2) →ₗ Dual V_P`.
  let res : Module.Dual (ZMod 2) (Fin n → ZMod 2) →ₗ[ZMod 2] Module.Dual (ZMod 2) V_P :=
    V_P.subtype.dualMap
  have hres_surj : Function.Surjective res :=
    LinearMap.dualMap_surjective_iff.mpr Subtype.val_injective
  have hker_res_dim : Module.finrank (ZMod 2) (LinearMap.ker res) = defect p := by
    have hrn := LinearMap.finrank_range_add_finrank_ker res
    rw [LinearMap.range_eq_top.mpr hres_surj, finrank_top, Subspace.dual_finrank_eq,
      Subspace.dual_finrank_eq, hVP_dim, Module.finrank_pi_fintype] at hrn
    simp only [Module.finrank_self, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
      smul_eq_mul, mul_one] at hrn
    simp only [defect]
    omega
  -- Collect all run ends of `q`.
  let runEnds : Finset (Fin q.length) := Finset.univ.filter (isRunEnd q)
  let Sbar : Fin q.length → Finset (Module.Dual (ZMod 2) V_P) :=
    fun b => (linOfRunEnd q b).image res
  -- Verify Rado's condition for `Sbar` on `runEnds`!
  have hRado : ∀ I : Finset (Fin q.length), I ⊆ runEnds → I.Nonempty →
      I.card ≤ Module.finrank (ZMod 2)
        (Submodule.span (ZMod 2) (⋃ b ∈ I, (Sbar b : Set (Module.Dual (ZMod 2) V_P)))) := by
    intro I hIsub hIne
    have hrun : ∀ b ∈ I, isRunEnd q b := fun b hb => (Finset.mem_filter.mp (hIsub hb)).2
    have hrank := rank_lower_bound_runEnds q I.card I rfl hIne hrun
    set U := Submodule.span (ZMod 2)
      (⋃ b ∈ I, (linOfRunEnd q b : Set (Module.Dual (ZMod 2) (Fin (q.length + 1) → ZMod 2))))
    have hmap_eq : U.map res = Submodule.span (ZMod 2)
        (⋃ b ∈ I, (Sbar b : Set (Module.Dual (ZMod 2) V_P))) := by
      rw [Submodule.map_span]
      congr 1
      ext ψ
      simp only [Set.mem_image, Set.mem_iUnion, Finset.mem_coe, exists_prop, Sbar, Finset.mem_image]
      exact ⟨fun ⟨x, ⟨i, hi, hx⟩, h⟩ => ⟨i, hi, x, hx, h⟩,
             fun ⟨i, hi, x, hx, h⟩ => ⟨x, ⟨i, hi, hx⟩, h⟩⟩
    -- `finrank U ≤ finrank (U.map res) + finrank (ker res)`.
    have hle_map := finrank_le_finrank_map_add_ker res U
    rw [hmap_eq] at hle_map
    have hcomb : defect q + I.card - 1 ≤
        Module.finrank (ZMod 2) (Submodule.span (ZMod 2)
          (⋃ b ∈ I, (Sbar b : Set (Module.Dual (ZMod 2) V_P)))) +
        Module.finrank (ZMod 2) (LinearMap.ker res) :=
      le_trans hrank hle_map
    omega
  -- Apply Rado's theorem to `runEnds`!
  obtain ⟨u, hu_mem, hu_li⟩ :=
    rado_aux (K := ZMod 2) runEnds.card runEnds rfl Sbar hRado
  -- Lift each `u b ∈ Sbar b` to an affine equation `(φ b, β b) ∈ eqsOfRunEnd q b.1` with `res (φ b) = u b`.
  have hlift : ∀ b : runEnds, ∃ eq ∈ eqsOfRunEnd q b.1, res eq.1 = u b := by
    intro b
    have h1 := hu_mem b
    simp only [Sbar, linOfRunEnd, Finset.mem_image] at h1
    obtain ⟨φ, ⟨eq, heq, rfl⟩, hres⟩ := h1
    exact ⟨eq, heq, hres⟩
  choose eqSel heqSel_mem heqSel_res using hlift
  -- Linear independence of `u : runEnds → Dual V_P` implies surjectivity of evaluation `V_P → (runEnds → ZMod 2)`.
  have heval_surj : Function.Surjective (fun (v : V_P) (b : runEnds) => u b v) :=
    surjective_eval_of_linearIndependent u hu_li
  -- Choose `v ∈ V_P` such that `u b v = (eqSel b).2 + 1 - (eqSel b).1 x₀` for every `b : runEnds`.
  obtain ⟨v, hv⟩ := heval_surj (fun b => (eqSel b).2 + 1 - (eqSel b).1 x₀)
  -- Then `x := x₀ + v.1` lies in `L(p) ⊆ L(q)`, yet violates `(eqSel b)` for every `b ∈ runEnds`!
  have hx_in_p : InLangNE p (List.ofFn (x₀ + v.1)) := hx0_add_mem v.1 v.2
  have hx_in_q : InLangNE q (List.ofFn (x₀ + v.1)) := hsub _ hx_in_p
  obtain ⟨b, hbe, hsat⟩ := inLangNE_gap_one_satisfies_some_runEnd q (x₀ + v.1) hx_in_q
  have hb_mem : b ∈ runEnds := Finset.mem_filter.mpr ⟨Finset.mem_univ b, hbe⟩
  let b' : runEnds := ⟨b, hb_mem⟩
  have h1 : (eqSel b').1 (x₀ + v.1) = (eqSel b').2 := hsat (eqSel b') (heqSel_mem b')
  have h2 : (eqSel b').1 (x₀ + v.1) = (eqSel b').2 + 1 := by
    rw [map_add]
    have hres_v : (eqSel b').1 v.1 = u b' v := by
      rw [← heqSel_res b']
      rfl
    rw [hres_v]
    have hvb : u b' v = (eqSel b').2 + 1 - (eqSel b').1 x₀ := congrFun hv b'
    rw [hvb]
    ring
  rw [h1] at h2
  have h01 : (0 : ZMod 2) = 1 := calc
    (0 : ZMod 2) = (eqSel b').2 - (eqSel b').2 := (sub_self _).symm
    _ = ((eqSel b').2 + 1) - (eqSel b').2 := by rw [← h2]
    _ = 1 := add_sub_cancel_left _ _
  exact zero_ne_one h01

/-- **Theorem 2 (All-lengths gap-one theorem)**: Over every alphabet `Σ` containing at least two
distinct letters, if `L_α(p) ⊆ L_α(q)` and `|p| = |q| + 1`, then
`varnum(p) ≤ varnum(q) + 1`, `defect(q) ≤ defect(p)`, and `μ(q) < μ(p)`. -/
theorem gap_one_theorem {α V₁ V₂ : Type*} [DecidableEq α] [DecidableEq V₁] [DecidableEq V₂]
    (h2 : ∃ a b : α, a ≠ b)
    {p : Pattern α V₁} {q : Pattern α V₂}
    (hsub : LangSubsetNE p q) (hlen : p.length = q.length + 1) :
    varnum p ≤ varnum q + 1 ∧ defect q ≤ defect p ∧ measure q < measure p := by
  obtain ⟨a, b, hab⟩ := h2
  let π : α → ZMod 2 := fun c => if c = a then 0 else 1
  have hπ_surj : Function.Surjective π := by
    intro y
    fin_cases y
    · exact ⟨a, by simp [π]⟩
    · exact ⟨b, by simp [π, Ne.symm hab]⟩
  have hsub2 : LangSubsetNE (mapTerm π p) (mapTerm π q) :=
    binary_surjection_descent π hπ_surj hsub
  have hlen2 : (mapTerm π p).length = (mapTerm π q).length + 1 := by
    rw [mapTerm_length, mapTerm_length, hlen]
  have hdef := gap_one_binary hsub2 hlen2
  simp only [defect, mapTerm_length, mapTerm_varnum] at hdef
  have hvp_le := varnum_le_length p
  have hvq_le := varnum_le_length q
  refine ⟨by omega, by simp only [defect]; omega, by simp only [measure]; omega⟩

end GapOne

end PatternInclusionDepth
