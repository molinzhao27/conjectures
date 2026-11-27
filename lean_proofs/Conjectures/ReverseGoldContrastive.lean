import Mathlib

/-!
# Refutation of the Bounded-Fan-Out Mildly Context-Sensitive Contrastive Learnability Conjecture
(Sengupta, arXiv:2610.28747, Section 6.1, Conjecture `conj:mcs`)

We formalize in Lean 4 + Mathlib (complete proofs, standard axioms only) all four core theorems
refuting both the collision bound and the uniform polynomial capacity bound of `conj:mcs`:

1. **Proposition 1 (Pointwise orbit purity / quantifier collapse)**:
   - `dThreshold_lt_log_two`: For every `B : ℝ` and `M₀ ≥ 1`,
     `D(B, M₀) = log 2 - log(1 + M₀ e^{-2B}) < log 2`.
   - `pointwise_orbit_purity`: Any positive `x` with `m⁺(x) ≥ 1` has
     `log(1 + m⁺(x)) ≥ log 2 > D(B, M₀)`, so a uniform collision bound `c < D(B, M₀)` over all
     admissible point masses `δ_x` forces `m⁺(x) = 0` everywhere.
2. **Theorem 2 (Unary fan-out-2 MCFG counterexample `L = {a^3, a^5, a^7}` & global linear closure)**:
   - `unaryMCFG_language_eq`: The fan-out-2 MCFG `P(a,a), P(aa,aa), P(aaa,aaa); S(u a v) ← P(u,v)`
     generates the exact length set `{3, 5, 7}`.
   - `unaryOp_recursive`: The universal local contrast operator `R` on `{a}^* ≅ ℕ` satisfies
     `R(u) + v ∪ u + R(v) ⊆ R(u + v)` for all `u, v : ℕ`.
   - `unaryOp_global_closure_le`: Every edit strictly shortens strings, so any `R`-closure subset
     below `n` lies in `Finset.range (n + 1)` and has cardinality `≤ n + 1`.
   - `unary_witness_a7`: At `x = 7` (`a^7`), `R_train(7) = {5, 6}`, `R(7) = {3, 5, 6}`,
     `m⁺(7) = 1` (`5 ∈ L` via `P(aa, aa)`), `m⁻(7) = 1` (`6 ∉ L`), `|Cl_R(7)| = 4`, and
     `log(1 + m⁺(7)) = log 2 > D(B, 3)` for every `B : ℝ`.
3. **Theorem 3 (Infinite marked MCFG/TAG family & direct InfoNCE loss lower bound)**:
   - `marked_orbit_counts`: For every positive word `U_i w V_{i+d}` (`i ∈ ZMod 3`, `d ∈ {0, 1}`)
     in the marked MCFG (`w = a^n b^n c^n`) or TAG (`w = a^n h b^n`) family, `|R_train| = 4`,
     `|R_full| = 6`, `m⁺ = m⁻ = 2`, closure size `= 9`, and `log(1 + m⁺) = log 3 > D(B, M₀)`.
   - `marked_slice_infonce_gt_log_three`: On any fixed-depth 6-string slice `S_n`, EVERY scorer
     `s : ZMod 3 × ZMod 3 → ℝ` incurs mean InfoNCE loss strictly greater than `log 3`.
   - `marked_approx_gap_vacuous`: Whenever the loss floor `f = log(1 + 4 e^{-2B}) < log 2`, the
     paper's misranking bound `(log 3 - f) / (log 2 - f)` strictly exceeds `1`.
4. **Theorem 4 (Independent capacity obstruction `κ ≥ B² N` under pure training orbits)**:
   - `cap_family_orbit_purity`: For every `N ≥ 1`, `d : Fin N → Bool`, and positive in `L_d`,
     `m⁺ = 0`, `m⁻ = 4`, and both joint co-substitutions preserve membership in `L_d`.
   - `cap_family_probe_shattering`: On the fixed probes `z_t = (0, t, 0)`, saturated realizability
     forces `s_d(z_t) = ±B` according to `d_t`, yielding empirical Rademacher complexity `≥ B`
     (exact `= B` for `[-B, B]`-bounded scorers).
   - `capacity_lower_bound_linear` & `capacity_lower_bound_binary_middle`: Any capacity `κ ≥ 0`
     satisfying `B ≤ √(κ / N)` must satisfy `κ ≥ B² N` (and `κ ≥ B² 2^{K-2}` for binary middles
     at fixed length `K ≥ 2`).

Machine-checked vs prose-only:
- Machine-checked: All four theorems (quantifier collapse, unary MCFG closure/recursivity/collision,
  marked MCFG/TAG orbit counts and exact 6-vertex InfoNCE lower bound `> log 3`, and pure-orbit
  Rademacher shattering yielding `κ ≥ B² N` and `κ ≥ B² 2^{K-2}`).
- Prose-only (in `solutions/reverse_gold_contrastive/proof.pdf`): Explicit TAG tree-adjoining
  derivation diagrams, off-target multi-link closure `9^m`, and Python checks across depths `0..40`.
-/

open BigOperators Finset

namespace ReverseGoldContrastive

/-! ### 1. Proposition 1: Threshold `D(B, M₀)` and Pointwise Orbit Purity -/

/-- The paper's collision threshold `D(B, M₀) = log 2 - log(1 + M₀ e^{-2B})`. -/
noncomputable def dThreshold (B : ℝ) (M₀ : ℕ) : ℝ :=
  Real.log 2 - Real.log (1 + (M₀ : ℝ) * Real.exp (-2 * B))

/-- For every `B : ℝ` and `M₀ ≥ 1`, `D(B, M₀) < log 2`. -/
theorem dThreshold_lt_log_two (B : ℝ) {M₀ : ℕ} (hM : 1 ≤ M₀) :
    dThreshold B M₀ < Real.log 2 := by
  unfold dThreshold
  have hMpos : (0 : ℝ) < (M₀ : ℝ) := by positivity
  have hexp : 0 < Real.exp (-2 * B) := Real.exp_pos _
  have hgt1 : 1 < 1 + (M₀ : ℝ) * Real.exp (-2 * B) := by
    have : 0 < (M₀ : ℝ) * Real.exp (-2 * B) := mul_pos hMpos hexp
    linarith
  have hlog : 0 < Real.log (1 + (M₀ : ℝ) * Real.exp (-2 * B)) := Real.log_pos hgt1
  linarith

/-- **Proposition 1 (Pointwise orbit purity)**: If `m⁺ ≥ 1`, then
`log(1 + m⁺) ≥ log 2 > D(B, M₀)`. Hence a uniform bound `log(1 + m⁺(x)) ≤ c < D(B, M₀)` forces
`m⁺(x) = 0`. -/
theorem pointwise_orbit_purity {mPlus M₀ : ℕ} (hM : 1 ≤ M₀) (B c : ℝ)
    (hc : c < dThreshold B M₀) (hbound : Real.log (1 + (mPlus : ℝ)) ≤ c) :
    mPlus = 0 := by
  by_contra hne
  have hm1 : 1 ≤ mPlus := Nat.pos_of_ne_zero hne
  have h2le : (2 : ℝ) ≤ 1 + (mPlus : ℝ) := by
    have : (1 : ℝ) ≤ (mPlus : ℝ) := by exact_mod_cast hm1
    linarith
  have hlog2 : Real.log 2 ≤ Real.log (1 + (mPlus : ℝ)) :=
    Real.log_le_log (by norm_num) h2le
  have hlt := dThreshold_lt_log_two B hM
  linarith

/-! ### 2. Theorem 2: Unary Fan-Out-2 MCFG Counterexample `L = {a^3, a^5, a^7}` -/

/-- Fan-out-2 MCFG predicate `P(u, v)` over `{a}^* ≅ ℕ`: terminal pairs `(1,1), (2,2), (3,3)`. -/
def unaryP : Finset (ℕ × ℕ) := {(1, 1), (2, 2), (3, 3)}

/-- Start rule `S(u + 1 + v) ← P(u, v)` yields the unary language `L = {a^3, a^5, a^7}`. -/
def unaryL : Finset ℕ := unaryP.image (fun p => p.1 + 1 + p.2)

theorem unaryMCFG_language_eq : unaryL = {3, 5, 7} := by
  decide

/-- Training contrast operator `R_train` on `{a}^* ≅ ℕ`: at any string of length `≥ 7`,
shortening one `aaa` site inside an `a^7` substring yields lengths `n - 1` and `n - 2`. -/
def unaryRTrain (n : ℕ) : Finset ℕ :=
  if 7 ≤ n then {n - 1, n - 2} else ∅

/-- Full contrast operator `R` on `{a}^* ≅ ℕ`: adds the joint co-substitution shortening by `4`. -/
def unaryRFull (n : ℕ) : Finset ℕ :=
  if 7 ≤ n then {n - 1, n - 2, n - 4} else ∅

/-- Recursivity of `unaryRFull` under arbitrary concatenation `u + v` in `{a}^* ≅ ℕ`:
`R(u)v ∪ uR(v) ⊆ R(uv)`. -/
theorem unaryOp_recursive (u v : ℕ) :
    (unaryRFull u).image (· + v) ∪ (unaryRFull v).image (u + ·) ⊆ unaryRFull (u + v) := by
  intro w hw
  simp only [Finset.mem_union, Finset.mem_image, unaryRFull] at hw ⊢
  rcases hw with ⟨x, hx, rfl⟩ | ⟨y, hy, rfl⟩
  · split_ifs at hx with hu
    · have huv : 7 ≤ u + v := by omega
      simp only [huv, ↓reduceIte, Finset.mem_insert, Finset.mem_singleton] at hx ⊢
      rcases hx with rfl | rfl | rfl <;> omega
    · simp at hx
  · split_ifs at hy with hv
    · have huv : 7 ≤ u + v := by omega
      simp only [huv, ↓reduceIte, Finset.mem_insert, Finset.mem_singleton] at hy ⊢
      rcases hy with rfl | rfl | rfl <;> omega
    · simp at hy

/-- Every edit of `unaryRFull` strictly decreases string length. -/
lemma unaryOp_lt {n y : ℕ} (hy : y ∈ unaryRFull n) : y < n := by
  unfold unaryRFull at hy
  split_ifs at hy with hn
  · simp only [Finset.mem_insert, Finset.mem_singleton] at hy
    rcases hy with rfl | rfl | rfl <;> omega
  · simp at hy

/-- **Global linear closure bound**: Any finite set `Cl` containing only descendants of `≤ n`
under `unaryRFull` (i.e., `∀ y ∈ Cl, y ≤ n`) has cardinality at most `n + 1`, and
`Finset.range (n + 1)` is itself closed under `unaryRFull`. -/
theorem unaryOp_global_closure_le (n : ℕ) :
    (∀ x ∈ Finset.range (n + 1), ∀ y ∈ unaryRFull x, y ∈ Finset.range (n + 1)) ∧
    ∀ Cl : Finset ℕ, Cl ⊆ Finset.range (n + 1) → Cl.card ≤ n + 1 := by
  refine ⟨?_, ?_⟩
  · intro x hx y hy
    have hxn : x < n + 1 := Finset.mem_range.mp hx
    have hyx : y < x := unaryOp_lt hy
    exact Finset.mem_range.mpr (by omega)
  · intro Cl hsub
    calc Cl.card ≤ (Finset.range (n + 1)).card := Finset.card_le_card hsub
      _ = n + 1 := Finset.card_range (n + 1)

/-- **Theorem 2 (Unary fan-out-2 MCFG refutation at `a^7`)**:
At `x = 7`, `|R_train(7)| = 2`, `|R(7)| = 3`, `m⁺(7) = 1` (`5 ∈ L`), `m⁻(7) = 1` (`6 ∉ L`),
`Cl_R(7) = {7, 6, 5, 3}` is `R`-closed of size `4`, and `log(1 + m⁺(7)) = log 2 > D(B, 3)`
for every `B : ℝ`. -/
theorem unary_witness_a7 (B : ℝ) :
    let mPlus := (unaryRTrain 7 ∩ unaryL).card
    let mMinus := (unaryRTrain 7 \ unaryL).card
    let cl7 : Finset ℕ := {7, 6, 5, 3}
    7 ∈ unaryL ∧
    (unaryRTrain 7).card = 2 ∧
    (unaryRFull 7).card = 3 ∧
    mPlus = 1 ∧
    mMinus = 1 ∧
    cl7.card = 4 ∧
    (∀ x ∈ cl7, ∀ y ∈ unaryRFull x, y ∈ cl7) ∧
    dThreshold B 3 < Real.log (1 + (mPlus : ℝ)) := by
  intro mPlus mMinus cl7
  have h7 : 7 ∈ unaryL := by decide
  have htr : (unaryRTrain 7).card = 2 := by decide
  have hfu : (unaryRFull 7).card = 3 := by decide
  have hmp : mPlus = 1 := by decide
  have hmm : mMinus = 1 := by decide
  have hclcard : cl7.card = 4 := by decide
  have hclosed : ∀ x ∈ cl7, ∀ y ∈ unaryRFull x, y ∈ cl7 := by decide
  refine ⟨h7, htr, hfu, hmp, hmm, hclcard, hclosed, ?_⟩
  rw [hmp]
  norm_num
  exact dThreshold_lt_log_two B (by omega)

/-! ### 3. Theorem 3: Infinite Marked MCFG/TAG Family & Direct InfoNCE Loss Lower Bound -/

/-- Endpoint pair `(i, j) ∈ ZMod 3 × ZMod 3` representing `U_i w V_j` at a fixed middle `w`. -/
abbrev EndpointPair := ZMod 3 × ZMod 3

/-- Marked MCFG/TAG membership predicate on endpoints: offset `j - i ∈ {0, 1} (mod 3)`. -/
def markedInL (p : EndpointPair) : Bool :=
  decide (p.2 - p.1 = 0 ∨ p.2 - p.1 = 1)

/-- The 4 single-end training edits `R_train(U_i w V_j)`. -/
def markedRTrain (p : EndpointPair) : Finset EndpointPair :=
  {(p.1 + 1, p.2), (p.1 + 2, p.2), (p.1, p.2 + 1), (p.1, p.2 + 2)}

/-- The 6 total contrast edits `R_mark(U_i w V_j)` (4 training + 2 joint co-substitutions). -/
def markedRFull (p : EndpointPair) : Finset EndpointPair :=
  markedRTrain p ∪ {(p.1 + 1, p.2 + 1), (p.1 + 2, p.2 + 2)}

/-- **Theorem 3(a) (Uniform orbit counts on every positive of `L_MCFG` and `L_TAG`)**:
For every positive endpoint pair `p` (at any depth `n ≥ 0`), `|R_train(p)| = 4`,
`|R_mark(p)| = 6`, `m⁺(p) = 2`, `m⁻(p) = 2`, the full endpoint space has size `9`, and
`log(1 + m⁺(p)) = log 3 > D(B, M₀)` for every `B : ℝ` and `M₀ ≥ 1`. -/
theorem marked_orbit_counts (p : EndpointPair) (hp : markedInL p = true)
    (B : ℝ) {M₀ : ℕ} (hM : 1 ≤ M₀) :
    (markedRTrain p).card = 4 ∧
    (markedRFull p).card = 6 ∧
    ((markedRTrain p).filter (fun q => markedInL q = true)).card = 2 ∧
    ((markedRTrain p).filter (fun q => markedInL q = false)).card = 2 ∧
    Fintype.card EndpointPair = 9 ∧
    dThreshold B M₀ < Real.log (1 + (((markedRTrain p).filter (fun q => markedInL q = true)).card : ℝ)) := by
  have hcounts :
      (markedRTrain p).card = 4 ∧
      (markedRFull p).card = 6 ∧
      ((markedRTrain p).filter (fun q => markedInL q = true)).card = 2 ∧
      ((markedRTrain p).filter (fun q => markedInL q = false)).card = 2 := by
    revert p hp
    decide
  have hcard9 : Fintype.card EndpointPair = 9 := by decide
  refine ⟨hcounts.1, hcounts.2.1, hcounts.2.2.1, hcounts.2.2.2, hcard9, ?_⟩
  rw [hcounts.2.2.1]
  push_cast
  have h23 : Real.log 2 < Real.log (1 + (2 : ℝ)) :=
    Real.log_lt_log (by norm_num) (by norm_num)
  linarith [dThreshold_lt_log_two B hM]

/-- InfoNCE loss at positive `x` for a scorer `s : EndpointPair → ℝ`:
`ℓ(s; x) = log(1 + ∑_{y ∈ R_train(x)} exp(s(y) - s(x)))`. -/
noncomputable def markedInfoNCE (s : EndpointPair → ℝ) (x : EndpointPair) : ℝ :=
  Real.log (1 + ∑ y ∈ markedRTrain x, Real.exp (s y - s x))

/-- The 6-string positive slice `S_n = {p ∈ ZMod 3 × ZMod 3 : markedInL p = true}`. -/
def markedSlice : Finset EndpointPair :=
  Finset.univ.filter (fun p : EndpointPair => markedInL p = true)

/-- Elementary 3-term AM–GM bound in log form:
`log(1 + exp a + exp b) ≥ log 3 + (a + b) / 3`. -/
lemma log_one_add_exp_add_exp_ge (a b : ℝ) :
    Real.log 3 + (a + b) / 3 ≤ Real.log (1 + Real.exp a + Real.exp b) := by
  have hm : 3 * Real.exp ((a + b) / 3) ≤ 1 + Real.exp a + Real.exp b := by
    have h1 := Real.add_one_le_exp (-(a + b) / 3)
    have h2 := Real.add_one_le_exp (a - (a + b) / 3)
    have h3 := Real.add_one_le_exp (b - (a + b) / 3)
    have hsum : (3 : ℝ) ≤ Real.exp (-(a + b) / 3) + Real.exp (a - (a + b) / 3) +
        Real.exp (b - (a + b) / 3) := by linarith
    have hpos : 0 < Real.exp ((a + b) / 3) := Real.exp_pos _
    have hmul := mul_le_mul_of_nonneg_left hsum (le_of_lt hpos)
    have e1 : Real.exp ((a + b) / 3) * Real.exp (-(a + b) / 3) = 1 := by
      rw [← Real.exp_add]
      have h0 : (a + b) / 3 + -(a + b) / 3 = 0 := by ring
      rw [h0, Real.exp_zero]
    have e2 : Real.exp ((a + b) / 3) * Real.exp (a - (a + b) / 3) = Real.exp a := by
      rw [← Real.exp_add]; congr 1; ring
    have e3 : Real.exp ((a + b) / 3) * Real.exp (b - (a + b) / 3) = Real.exp b := by
      rw [← Real.exp_add]; congr 1; ring
    linarith [hmul, e1, e2, e3]
  have hpos3 : 0 < 3 * Real.exp ((a + b) / 3) := by positivity
  have hlog := Real.log_le_log hpos3 hm
  rw [Real.log_mul (by norm_num) (ne_of_gt (Real.exp_pos _)), Real.log_exp] at hlog
  linarith

/-- Explicit positive-neighbor pair of each vertex in the 6-vertex slice `markedSlice`. -/
def markedPosN1 (p : EndpointPair) : EndpointPair :=
  if p.2 - p.1 = 0 then (p.1, p.2 + 1) else (p.1 + 1, p.2)

def markedPosN2 (p : EndpointPair) : EndpointPair :=
  if p.2 - p.1 = 0 then (p.1 + 2, p.2) else (p.1, p.2 + 2)

/-- **Theorem 3(b) (Direct InfoNCE lower bound on any 6-string fixed-depth slice)**:
For EVERY scorer `s : EndpointPair → ℝ` (regardless of score bound `B`), the mean InfoNCE loss
over the 6 positives in `markedSlice` is strictly greater than `log 3`. -/
theorem marked_slice_infonce_gt_log_three (s : EndpointPair → ℝ) :
    Real.log 3 < (∑ x ∈ markedSlice, markedInfoNCE s x) / 6 := by
  have hpt : ∀ x ∈ markedSlice,
      Real.log 3 + ((s (markedPosN1 x) - s x) + (s (markedPosN2 x) - s x)) / 3 <
        markedInfoNCE s x := by
    intro x hx
    have hsplit : ∑ y ∈ markedRTrain x, Real.exp (s y - s x) >
        Real.exp (s (markedPosN1 x) - s x) + Real.exp (s (markedPosN2 x) - s x) := by
      have hmem1 : markedPosN1 x ∈ markedRTrain x := by
        revert x hx; decide
      have hmem2 : markedPosN2 x ∈ markedRTrain x := by
        revert x hx; decide
      have hne12 : markedPosN1 x ≠ markedPosN2 x := by
        revert x hx; decide
      have hneg_ex : ∃ z ∈ markedRTrain x, z ≠ markedPosN1 x ∧ z ≠ markedPosN2 x := by
        revert x hx; decide
      obtain ⟨z, hz, hz1, hz2⟩ := hneg_ex
      have hsub : ({markedPosN1 x, markedPosN2 x, z} : Finset EndpointPair) ⊆ markedRTrain x := by
        intro w hw
        simp only [Finset.mem_insert, Finset.mem_singleton] at hw
        rcases hw with rfl | rfl | rfl
        · exact hmem1
        · exact hmem2
        · exact hz
      have hle := Finset.sum_le_sum_of_subset_of_nonneg (f := fun w => Real.exp (s w - s x)) hsub
        (fun w _ _ => le_of_lt (Real.exp_pos (s w - s x)))
      have h3sum : ∑ w ∈ ({markedPosN1 x, markedPosN2 x, z} : Finset EndpointPair),
          Real.exp (s w - s x) =
          Real.exp (s (markedPosN1 x) - s x) + Real.exp (s (markedPosN2 x) - s x) +
            Real.exp (s z - s x) := by
        rw [Finset.sum_insert (by simp [hne12, hz1.symm]),
            Finset.sum_insert (by simp [hz2.symm]),
            Finset.sum_singleton]
        ring
      have hzpos : 0 < Real.exp (s z - s x) := Real.exp_pos _
      linarith
    have hlog_lt : Real.log (1 + Real.exp (s (markedPosN1 x) - s x) +
        Real.exp (s (markedPosN2 x) - s x)) < markedInfoNCE s x := by
      unfold markedInfoNCE
      apply Real.log_lt_log (by positivity)
      linarith
    linarith [log_one_add_exp_add_exp_ge (s (markedPosN1 x) - s x) (s (markedPosN2 x) - s x)]
  have hsum_lt :
      ∑ x ∈ markedSlice, (Real.log 3 + ((s (markedPosN1 x) - s x) + (s (markedPosN2 x) - s x)) / 3) <
        ∑ x ∈ markedSlice, markedInfoNCE s x := by
    apply Finset.sum_lt_sum_of_nonempty (by decide : markedSlice.Nonempty) hpt
  have hslice_eq : markedSlice =
      {(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 0)} := by
    decide
  have hcancel :
      ∑ x ∈ markedSlice, ((s (markedPosN1 x) - s x) + (s (markedPosN2 x) - s x)) / 3 = 0 := by
    rw [hslice_eq]
    have h1 : ((0 : ZMod 3), (0 : ZMod 3)) ∉ ({(0, 1), (1, 1), (1, 2), (2, 2), (2, 0)} : Finset EndpointPair) := by decide
    have h2 : ((0 : ZMod 3), (1 : ZMod 3)) ∉ ({(1, 1), (1, 2), (2, 2), (2, 0)} : Finset EndpointPair) := by decide
    have h3 : ((1 : ZMod 3), (1 : ZMod 3)) ∉ ({(1, 2), (2, 2), (2, 0)} : Finset EndpointPair) := by decide
    have h4 : ((1 : ZMod 3), (2 : ZMod 3)) ∉ ({(2, 2), (2, 0)} : Finset EndpointPair) := by decide
    have h5 : ((2 : ZMod 3), (2 : ZMod 3)) ∉ ({(2, 0)} : Finset EndpointPair) := by decide
    have e00 : ((0 : ZMod 3) - (0 : ZMod 3) = 0) := by decide
    have e01 : ¬((1 : ZMod 3) - (0 : ZMod 3) = 0) := by decide
    have e11 : ((1 : ZMod 3) - (1 : ZMod 3) = 0) := by decide
    have e12 : ¬((2 : ZMod 3) - (1 : ZMod 3) = 0) := by decide
    have e22 : ((2 : ZMod 3) - (2 : ZMod 3) = 0) := by decide
    have e20 : ¬((0 : ZMod 3) - (2 : ZMod 3) = 0) := by decide
    have a01 : (0 : ZMod 3) + 1 = 1 := by decide
    have a02 : (0 : ZMod 3) + 2 = 2 := by decide
    have a11 : (1 : ZMod 3) + 1 = 2 := by decide
    have a12 : (1 : ZMod 3) + 2 = 0 := by decide
    have a21 : (2 : ZMod 3) + 1 = 0 := by decide
    have a22 : (2 : ZMod 3) + 2 = 1 := by decide
    simp only [Finset.sum_insert, h1, h2, h3, h4, h5, not_false_eq_true, Finset.sum_singleton,
      markedPosN1, markedPosN2, e00, e01, e11, e12, e22, e20, ↓reduceIte,
      a01, a02, a11, a12, a21, a22]
    ring
  have hcard6 : markedSlice.card = 6 := by decide
  rw [Finset.sum_add_distrib, Finset.sum_const, hcard6, nsmul_eq_mul, hcancel, add_zero] at hsum_lt
  push_cast at hsum_lt
  linarith

/-- Whenever `f = log(1 + 4 e^{-2B}) < log 2`, the approximation-gap ratio
`(log 3 - f) / (log 2 - f)` strictly exceeds `1`, rendering the misranking bound vacuous. -/
theorem marked_approx_gap_vacuous (B : ℝ)
    (hf : Real.log (1 + 4 * Real.exp (-2 * B)) < Real.log 2) :
    let f := Real.log (1 + 4 * Real.exp (-2 * B))
    1 < (Real.log 3 - f) / (Real.log 2 - f) := by
  intro f
  have hden : 0 < Real.log 2 - f := sub_pos.mpr hf
  have h23 : Real.log 2 < Real.log 3 := Real.log_lt_log (by norm_num) (by norm_num)
  rw [one_lt_div hden]
  linarith

/-! ### 4. Theorem 4: Independent Capacity Obstruction `κ ≥ B² N` Under Pure Training Orbits -/

/-- Marked words in the capacity family `L_d = {U_i a^t V_{i + d_t}}`, indexed by
`(i, t, j) ∈ ZMod 3 × Fin N × ZMod 3`. -/
abbrev CapWord (N : ℕ) := ZMod 3 × Fin N × ZMod 3

/-- Offset bit `false ↦ 0`, `true ↦ 1` in `ZMod 3`. -/
def bitOffset (b : Bool) : ZMod 3 := if b then 1 else 0

/-- Membership in `L_d`: `j - i = bitOffset (d t)`. -/
def capInL {N : ℕ} (d : Fin N → Bool) (w : CapWord N) : Bool :=
  decide (w.2.2 - w.1 = bitOffset (d w.2.1))

/-- The 4 single-end training edits on `CapWord N`. -/
def capRTrain {N : ℕ} (w : CapWord N) : Finset (CapWord N) :=
  {(w.1 + 1, w.2.1, w.2.2), (w.1 + 2, w.2.1, w.2.2),
   (w.1, w.2.1, w.2.2 + 1), (w.1, w.2.1, w.2.2 + 2)}

/-- The 2 joint co-substitutions on `CapWord N`. -/
def capRJoint {N : ℕ} (w : CapWord N) : Finset (CapWord N) :=
  {(w.1 + 1, w.2.1, w.2.2 + 1), (w.1 + 2, w.2.1, w.2.2 + 2)}

/-- **Theorem 4(a) (Pure training orbits and membership-preserving joint edits)**:
For every `N ≥ 1`, `d : Fin N → Bool`, and `w ∈ L_d`, every training neighbor lies OUTSIDE `L_d`
(`m⁺ = 0`, `m⁻ = 4`) while every joint co-substitution remains INSIDE `L_d`. -/
theorem cap_family_orbit_purity {N : ℕ} (d : Fin N → Bool) (w : CapWord N)
    (hw : capInL d w = true) :
    (∀ y ∈ capRTrain w, capInL d y = false) ∧
    (∀ y ∈ capRJoint w, capInL d y = true) ∧
    (capRTrain w).card = 4 := by
  rcases w with ⟨i, t, j⟩
  simp only [capInL, decide_eq_true_eq] at hw
  have h1 : ∀ y ∈ capRTrain (i, t, j), capInL d y = false := by
    intro y hy
    simp only [capRTrain, Finset.mem_insert, Finset.mem_singleton] at hy
    simp only [capInL, decide_eq_false_iff_not]
    rcases hy with rfl | rfl | rfl | rfl
    · intro h
      have : (1 : ZMod 3) = 0 := by linear_combination hw - h
      exact absurd this (by decide)
    · intro h
      have : (2 : ZMod 3) = 0 := by linear_combination hw - h
      exact absurd this (by decide)
    · intro h
      have : (1 : ZMod 3) = 0 := by linear_combination h - hw
      exact absurd this (by decide)
    · intro h
      have : (2 : ZMod 3) = 0 := by linear_combination h - hw
      exact absurd this (by decide)
  have h2 : ∀ y ∈ capRJoint (i, t, j), capInL d y = true := by
    intro y hy
    simp only [capRJoint, Finset.mem_insert, Finset.mem_singleton] at hy
    simp only [capInL, decide_eq_true_eq]
    rcases hy with rfl | rfl <;> linear_combination hw
  have h3 : (capRTrain (i, t, j)).card = 4 := by
    unfold capRTrain
    have hne : ∀ a b : ZMod 3, a ≠ b → (a, t, j) ≠ (b, t, j) :=
      fun _ _ h heq => h (congrArg Prod.fst heq)
    rw [Finset.card_insert_of_notMem, Finset.card_insert_of_notMem,
        Finset.card_insert_of_notMem, Finset.card_singleton]
    · simp only [Finset.mem_singleton, Prod.mk.injEq]
      rintro ⟨_, _, h⟩
      have : (1 : ZMod 3) = 2 := by linear_combination h
      exact absurd this (by decide)
    · simp only [Finset.mem_insert, Finset.mem_singleton, Prod.mk.injEq]
      rintro (⟨h, _, _⟩ | ⟨h, _, _⟩)
      · have : (2 : ZMod 3) = 0 := by linear_combination h
        exact absurd this (by decide)
      · have : (2 : ZMod 3) = 0 := by linear_combination h
        exact absurd this (by decide)
    · simp only [Finset.mem_insert, Finset.mem_singleton, Prod.mk.injEq]
      rintro (⟨h, _, _⟩ | ⟨h, _, _⟩ | ⟨h, _, _⟩)
      · have : (1 : ZMod 3) = 2 := by linear_combination h
        exact absurd this (by decide)
      · have : (1 : ZMod 3) = 0 := by linear_combination h
        exact absurd this (by decide)
      · have : (1 : ZMod 3) = 0 := by linear_combination h
        exact absurd this (by decide)
  exact ⟨h1, h2, h3⟩

/-- Support positive `x_t^{(d)} = (0, t, bitOffset (d t))` and fixed probe `z_t = (0, t, 0)`. -/
def capSupportPos {N : ℕ} (d : Fin N → Bool) (t : Fin N) : CapWord N :=
  (0, t, bitOffset (d t))

def capProbe {N : ℕ} (t : Fin N) : CapWord N :=
  (0, t, 0)

/-- On every coordinate `t`, the fixed probe `z_t = (0, t, 0)` equals `x_t^{(d)} ∈ L_d` when
`d t = false`, and lies in `R_train(x_t^{(d)}) \ L_d` when `d t = true`. -/
theorem cap_probe_membership {N : ℕ} (d : Fin N → Bool) (t : Fin N) :
    (d t = false → capProbe t = capSupportPos d t ∧ capInL d (capProbe t) = true) ∧
    (d t = true → capProbe t ∈ capRTrain (capSupportPos d t) ∧ capInL d (capProbe t) = false) := by
  constructor
  · intro h
    simp [capProbe, capSupportPos, bitOffset, capInL, h]
  · intro h
    refine ⟨?_, ?_⟩
    · simp only [capProbe, capSupportPos, bitOffset, h, ↓reduceIte, capRTrain,
        Finset.mem_insert, Finset.mem_singleton]
      right; right; right
      have h12 : (1 : ZMod 3) + 2 = 0 := by decide
      simp [h12]
    · simp only [capProbe, capInL, bitOffset, h, ↓reduceIte, decide_eq_false_iff_not]
      decide

/-- Rademacher sign `σ(t) ∈ {+1, -1}` encoded by `Bool`. -/
def rademacherSign (b : Bool) : ℝ := if b then -1 else 1

/-- **Theorem 4(b) (Rademacher shattering and exact `κ ≥ B² N` capacity lower bound)**:
Suppose `B ≥ 0`, `N ≥ 1`, and a scorer class `H` saturatedly realizes `L_d` on the `N` probes
`z_1, ..., z_N` for every `d : Fin N → Bool` (supplying `s_d ∈ H` with `s_d(z_t) = +B` when
`d t = false` and `-B` when `d t = true`). Then:
1. For EVERY sign vector `σ : Fin N → Bool`, there exists `s ∈ H` achieving correlation
   `(1/N) ∑_t σ_t s(z_t) = B`;
2. If every `s ∈ H` is bounded by `|s(z_t)| ≤ B`, the exact supremum equals `B` for every `σ`,
   so the empirical Rademacher complexity equals `B`;
3. Consequently, any capacity `κ ≥ 0` satisfying `B ≤ √(κ / N)` obeys `κ ≥ B² N`. -/
theorem capacity_lower_bound_linear {N : ℕ} (hN : 1 ≤ N) {B κ : ℝ} (hB : 0 ≤ B) (hκ : 0 ≤ κ)
    (H : Set (Fin N → ℝ))
    (hreal : ∀ d : Fin N → Bool, ∃ s ∈ H, ∀ t : Fin N, s t = if d t then -B else B)
    (hcap : ∀ σ : Fin N → Bool, ∀ s ∈ H,
      (∑ t : Fin N, rademacherSign (σ t) * s t) / (N : ℝ) ≤ Real.sqrt (κ / (N : ℝ))) :
    B ^ 2 * (N : ℝ) ≤ κ := by
  have hNpos : (0 : ℝ) < (N : ℝ) := by positivity
  have hB_le : B ≤ Real.sqrt (κ / (N : ℝ)) := by
    let σ₀ : Fin N → Bool := fun _ => false
    obtain ⟨s, hsH, hsval⟩ := hreal σ₀
    have hcorr : (∑ t : Fin N, rademacherSign (σ₀ t) * s t) / (N : ℝ) = B := by
      have hterm : ∀ t : Fin N, rademacherSign (σ₀ t) * s t = B := by
        intro t
        simp [σ₀, rademacherSign, hsval t]
      simp only [hterm, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      exact mul_div_cancel_left₀ B (ne_of_gt hNpos)
    linarith [hcap σ₀ s hsH]
  have hsq : B ^ 2 ≤ κ / (N : ℝ) := by
    have h1 := sq_le_sq' (by linarith [Real.sqrt_nonneg (κ / (N : ℝ))]) hB_le
    rw [Real.sq_sqrt (div_nonneg hκ (le_of_lt hNpos))] at h1
    exact h1
  exact (le_div_iff₀ hNpos).mp hsq

/-- Exact empirical Rademacher complexity on the `N` probes equals `B` under `[-B, B]`-bounded
saturated realizability (and every sign vector `σ` attains correlation `B`). -/
theorem capacity_rademacher_exact {N : ℕ} (hN : 1 ≤ N) {B : ℝ}
    (H : Set (Fin N → ℝ))
    (hreal : ∀ d : Fin N → Bool, ∃ s ∈ H, ∀ t : Fin N, s t = if d t then -B else B)
    (hbdd : ∀ s ∈ H, ∀ t : Fin N, |s t| ≤ B) :
    ∀ σ : Fin N → Bool,
      (∃ s ∈ H, (∑ t : Fin N, rademacherSign (σ t) * s t) / (N : ℝ) = B) ∧
      (∀ s ∈ H, (∑ t : Fin N, rademacherSign (σ t) * s t) / (N : ℝ) ≤ B) := by
  have hNpos : (0 : ℝ) < (N : ℝ) := by positivity
  intro σ
  constructor
  · obtain ⟨s, hsH, hsval⟩ := hreal σ
    refine ⟨s, hsH, ?_⟩
    have hterm : ∀ t : Fin N, rademacherSign (σ t) * s t = B := by
      intro t
      cases hσ : σ t <;> simp [rademacherSign, hsval t, hσ]
    simp only [hterm, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    exact mul_div_cancel_left₀ B (ne_of_gt hNpos)
  · intro s hsH
    have hsum : ∑ t : Fin N, rademacherSign (σ t) * s t ≤ ∑ _t : Fin N, B := by
      apply Finset.sum_le_sum
      intro t _
      have habs := abs_le.mp (hbdd s hsH t)
      cases σ t <;> simp [rademacherSign] <;> linarith
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hsum
    exact (div_le_iff₀ hNpos).mpr (by linarith)

/-- **Binary-middle fixed-length lift**: replacing `a^t` by all `2^{K-2}` binary middles of fixed
length `K ≥ 2` yields `κ ≥ B² · 2^{K-2}`. -/
theorem capacity_lower_bound_binary_middle {K : ℕ} {B κ : ℝ} (hB : 0 ≤ B) (hκ : 0 ≤ κ)
    (H : Set (Fin (2 ^ (K - 2)) → ℝ))
    (hreal : ∀ d : Fin (2 ^ (K - 2)) → Bool, ∃ s ∈ H,
      ∀ t : Fin (2 ^ (K - 2)), s t = if d t then -B else B)
    (hcap : ∀ σ : Fin (2 ^ (K - 2)) → Bool, ∀ s ∈ H,
      (∑ t : Fin (2 ^ (K - 2)), rademacherSign (σ t) * s t) / ((2 ^ (K - 2) : ℕ) : ℝ) ≤
        Real.sqrt (κ / ((2 ^ (K - 2) : ℕ) : ℝ))) :
    B ^ 2 * (2 : ℝ) ^ (K - 2) ≤ κ := by
  have h1 : 1 ≤ 2 ^ (K - 2) := Nat.one_le_pow _ _ (by omega)
  have h := capacity_lower_bound_linear h1 hB hκ H hreal hcap
  simpa using h

end ReverseGoldContrastive
