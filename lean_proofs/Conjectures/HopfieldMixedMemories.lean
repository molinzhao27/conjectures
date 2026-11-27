import Mathlib

/-!
# Formalization of Core Counterexamples to Conjecture 1.4 (`1.conj.2`) in Gayrard (arXiv:2504.04879)

We formalize in Lean 4 + Mathlib (zero `sorry`) the finite-dimensional Chow-parameter and
multiplicity-obstruction core of our refutation of the "only if" direction of Conjecture 1.4
(`1.conj.2`) in Véronique Gayrard, *"Mixed memories in Hopfield networks"*
(arXiv:2504.04879, `Mix-ALL_v2.tex`):

1. **Exact Chow-parameter map `chowParamQ`** on `Fin n → ℚ` and strict tie-freeness with explicit
   margin `HasMargin w c`.
2. **Classical Hopfield (`F(x) = x²/2`, `F'(x) = x`) counterexample at `n = 5`**:
   `m5 = (5/8, 3/8, 3/8, 1/8, 1/8)` is strictly tie-free with margin `1/8` (`m5_tie_free`),
   is an exact fixed point of the Chow-parameter map (`m5_chow_fixed`), and does not belong to
   Gayrard's family `M_all` (`m5_not_in_M_all`).
   (Note: this classical `n = 5` witness and the even-multiplicity obstruction were also posted
   independently in October 2026 at `the-omega-institute/trureturing#13571`.)
3. **Dense Hopfield (`F(x) = x^p / p`, `p = 3, 4`) counterexample at `n = 7`**:
   `m7 = (11/32, 11/32, 11/32, 9/32, 9/32, 9/32, 9/32)` is strictly tie-free (with certified
   positive margins `≥ 1/1024` for `p = 3` and `≥ 1/32768` for `p = 4`; exact margins `39/1024`
   and `127/32768`) and an exact Chow fixed point for both `p = 3` (`m7_p3_tie_free`,
   `m7_p3_chow_fixed`) and `p = 4` (`m7_p4_tie_free`, `m7_p4_chow_fixed`), and does not belong to
   `M_all` (`m7_not_in_M_all`).
4. **Lemma B (Even-Multiplicity Obstruction for `M_all`)**:
   Formalizing allowable compositions `(n₁, …, n_ℓ)` (even `n₁, …, n_{ℓ-1} ≥ 2`, odd `n_ℓ ≥ 1`),
   `alphaVal s = 2^(-(s-1)) * C(s-1, ⌊(s-1)/2⌋)`, and cumulative products `γ_k`, we prove that
   for any vector in `M_all`, every distinct positive magnitude strictly above the minimum
   positive magnitude appears with **even** multiplicity (`M_all_multiplicity_even_above_min`).
5. **Parseval Identity & Universal Lifting Inequality**:
   For any Boolean function `g : (Fin n → Bool) → ℚ` with values in `{-1, 1}`, its degree-1
   Fourier/Chow coefficients satisfy `∑ i, (Phi_i)^2 ≤ 1` (`chow_parseval`), hence for any
   `p ≥ 3` and `m̃ ∈ (0, 1)^n`, `∑ i, (m̃ i)^(p - 1) ≤ 1 < 2` (`dense_lift_sum_lt_two`).

(Note: the SLLN bridge from tie-free Chow fixed points to Definition `1.def1.2`, Section 4 local
minimality, the general `2k`-block lifting theorem, the `(2a+1, 2b)` dense family for `p ≥ 5`,
and the general smooth `F` counterexamples are established in prose in `solution.md` and verified
computationally in `verify_hopfield.py`, not formalized in this Lean file.)
-/

open Finset BigOperators

namespace HopfieldMixedMemories

/-- Map `Bool` to `{-1, 1}` in `ℚ`. -/
def bSign (b : Bool) : ℚ := if b then 1 else -1

/-- Sign function `{-1, 0, 1}` on `ℚ`. -/
def qSign (x : ℚ) : ℚ := if 0 < x then 1 else if x < 0 then -1 else 0

/-- Inner product `∑ j, bSign (r j) * w j`. -/
def rDot {n : ℕ} (r : Fin n → Bool) (w : Fin n → ℚ) : ℚ :=
  ∑ j : Fin n, bSign (r j) * w j

/-- Exact Chow parameter `Φ_i(w) = 2^{-n} ∑_{r ∈ {-1,1}^n} r_i sign(r · w)`. -/
def chowParamQ {n : ℕ} (w : Fin n → ℚ) (i : Fin n) : ℚ :=
  (∑ r : Fin n → Bool, bSign (r i) * qSign (rDot r w)) / (2 ^ n : ℚ)

/-- Strict tie-freeness with explicit margin `c > 0`. -/
def HasMargin {n : ℕ} (w : Fin n → ℚ) (c : ℚ) : Prop :=
  0 < c ∧ ∀ r : Fin n → Bool, c ≤ |rDot r w|

/-- Computable recursive sum over `Fin n → Bool` (avoids `Multiset.Pi` quotient in `decide`). -/
def sumBool : (n : ℕ) → ((Fin n → Bool) → ℚ) → ℚ
  | 0, f => f Fin.elim0
  | n + 1, f =>
      sumBool n (fun r => f (Fin.cons true r)) + sumBool n (fun r => f (Fin.cons false r))

lemma sumBool_eq_sum : ∀ (n : ℕ) (f : (Fin n → Bool) → ℚ),
    sumBool n f = ∑ r : Fin n → Bool, f r
  | 0, f => by
    simp [sumBool]
    exact congrArg f (Subsingleton.elim _ _)
  | n + 1, f => by
    simp only [sumBool, sumBool_eq_sum n]
    let e : (Bool × (Fin n → Bool)) ≃ (Fin (n + 1) → Bool) :=
      { toFun := fun p => Fin.cons p.1 p.2
        invFun := fun r => (r 0, fun i => r i.succ)
        left_inv := fun ⟨b, r⟩ => by ext <;> simp
        right_inv := fun r => by ext i; refine Fin.cases ?_ ?_ i <;> simp }
    rw [← Equiv.sum_comp e, Fintype.sum_prod_type]
    simp [e]

/-- Computable recursive universal quantifier over `Fin n → Bool`. -/
def allBool : (n : ℕ) → ((Fin n → Bool) → Bool) → Bool
  | 0, p => p Fin.elim0
  | n + 1, p =>
      allBool n (fun r => p (Fin.cons true r)) && allBool n (fun r => p (Fin.cons false r))

lemma allBool_eq_true_iff : ∀ (n : ℕ) (p : (Fin n → Bool) → Bool),
    allBool n p = true ↔ ∀ r : Fin n → Bool, p r = true
  | 0, p => by
    constructor
    · intro h r; rwa [Subsingleton.elim r Fin.elim0]
    · intro h; exact h Fin.elim0
  | n + 1, p => by
    simp only [allBool, Bool.and_eq_true, allBool_eq_true_iff n]
    constructor
    · rintro ⟨ht, hf⟩ r
      have hr : r = Fin.cons (r 0) (fun i => r i.succ) := by
        ext i; refine Fin.cases ?_ ?_ i <;> simp
      rw [hr]
      cases r 0
      · exact hf _
      · exact ht _
    · intro h
      exact ⟨fun r => h _, fun r => h _⟩

/-- Pure-integer helpers for fast kernel evaluation of finite Chow sums without `Rat` overhead. -/
def bSignZ (b : Bool) : ℤ := if b then 1 else -1
def zSign (x : ℤ) : ℤ := if 0 < x then 1 else if x < 0 then -1 else 0

def rDotZ : {n : ℕ} → (Fin n → Bool) → (Fin n → ℤ) → ℤ
  | 0, _, _ => 0
  | _ + 1, r, w => bSignZ (r 0) * w 0 + rDotZ (fun i => r i.succ) (fun i => w i.succ)

def sumBoolZ : (n : ℕ) → ((Fin n → Bool) → ℤ) → ℤ
  | 0, f => f Fin.elim0
  | n + 1, f =>
      sumBoolZ n (fun r => f (Fin.cons true r)) + sumBoolZ n (fun r => f (Fin.cons false r))

lemma sumBoolZ_cast : ∀ (n : ℕ) (f : (Fin n → Bool) → ℤ),
    (sumBoolZ n f : ℚ) = sumBool n (fun r => (f r : ℚ))
  | 0, _ => rfl
  | n + 1, f => by
    simp only [sumBoolZ, sumBool, Int.cast_add, sumBoolZ_cast n]

lemma rDotZ_eq_rDot_mul {n : ℕ} (D : ℚ) (_hD : D ≠ 0) (wZ : Fin n → ℤ) (wQ : Fin n → ℚ)
    (hw : ∀ i, wQ i = (wZ i : ℚ) / D) (r : Fin n → Bool) :
    rDot r wQ = (rDotZ r wZ : ℚ) / D := by
  induction n with
  | zero => simp [rDot, rDotZ]
  | succ m ih =>
    simp only [rDot, rDotZ, Fin.sum_univ_succ, Int.cast_add, Int.cast_mul, add_div]
    have h0 : bSign (r 0) * wQ 0 = (bSignZ (r 0) : ℚ) * (wZ 0 : ℚ) / D := by
      rw [hw 0]
      cases r 0 <;> (simp [bSign, bSignZ]; try ring)
    have htail := ih (fun i => wZ i.succ) (fun i => wQ i.succ) (fun i => hw i.succ) (fun i => r i.succ)
    unfold rDot at htail
    rw [h0, htail]

lemma qSign_div_pos (x : ℤ) {D : ℚ} (hD : 0 < D) :
    qSign ((x : ℚ) / D) = (zSign x : ℚ) := by
  unfold qSign zSign
  have hpos : (0 : ℚ) < (x : ℚ) / D ↔ (0 : ℤ) < x := by
    rw [div_pos_iff_of_pos_right hD]; exact_mod_cast Iff.rfl
  have hneg : (x : ℚ) / D < 0 ↔ x < (0 : ℤ) := by
    rw [div_neg_iff]; constructor
    · rintro (⟨_, h⟩ | ⟨h, _⟩); · linarith
      · exact_mod_cast h
    · intro h; exact Or.inr ⟨by exact_mod_cast h, hD⟩
  by_cases h1 : (0 : ℤ) < x
  · rw [if_pos (hpos.mpr h1), if_pos h1]; norm_num
  · rw [if_neg (mt hpos.mp h1), if_neg h1]
    by_cases h2 : x < 0
    · rw [if_pos (hneg.mpr h2), if_pos h2]; norm_num
    · rw [if_neg (mt hneg.mp h2), if_neg h2]; norm_num

/-! ### Part 1: Exact Verification of Minimal Counterexamples (`n = 5` Classical, `n = 7` Dense) -/

/-- Minimal classical Hopfield (`p = 2`) counterexample at `n = 5`: `(5/8, 3/8, 3/8, 1/8, 1/8)`. -/
def m5 : Fin 5 → ℚ := ![5 / 8, 3 / 8, 3 / 8, 1 / 8, 1 / 8]

/-- Minimal dense Hopfield (`p = 3, 4`) counterexample at `n = 7`: `(11/32)^3, (9/32)^4`. -/
def m7 : Fin 7 → ℚ := ![11 / 32, 11 / 32, 11 / 32, 9 / 32, 9 / 32, 9 / 32, 9 / 32]

def m5Z (i : Fin 5) : ℤ := match i.val with | 0 => 5 | 1 => 3 | 2 => 3 | _ => 1
def m7p3Z (i : Fin 7) : ℤ := if i.val < 3 then 121 else 81
def m7p4Z (i : Fin 7) : ℤ := if i.val < 3 then 1331 else 729

lemma m5_eq_div (i : Fin 5) : m5 i = (m5Z i : ℚ) / 8 := by
  fin_cases i <;> norm_num [m5, m5Z]

lemma m7p3_eq_div (i : Fin 7) : (m7 i) ^ 2 = (m7p3Z i : ℚ) / 1024 := by
  fin_cases i <;> norm_num [m7, m7p3Z]

lemma m7p4_eq_div (i : Fin 7) : (m7 i) ^ 3 = (m7p4Z i : ℚ) / 32768 := by
  fin_cases i <;> norm_num [m7, m7p4Z]

theorem m5_tie_free : HasMargin m5 (1 / 8) := by
  refine ⟨by norm_num, ?_⟩
  have h : allBool 5 (fun r => decide (1 ≤ |rDotZ r m5Z|)) = true := by decide
  rw [allBool_eq_true_iff] at h
  intro r
  have hr : (1 : ℤ) ≤ |rDotZ r m5Z| := of_decide_eq_true (h r)
  rw [rDotZ_eq_rDot_mul 8 (by norm_num) m5Z m5 m5_eq_div r, abs_div, abs_of_pos (by norm_num : (0 : ℚ) < 8)]
  have hrQ : (1 : ℚ) ≤ |(rDotZ r m5Z : ℚ)| := by exact_mod_cast hr
  linarith

theorem m5_chow_fixed : ∀ i : Fin 5, chowParamQ m5 i = m5 i := by
  have hZ : ∀ i : Fin 5, sumBoolZ 5 (fun r => bSignZ (r i) * zSign (rDotZ r m5Z)) = 4 * m5Z i := by
    decide
  intro i
  unfold chowParamQ
  rw [← sumBool_eq_sum]
  have hcongr : sumBool 5 (fun r => bSign (r i) * qSign (rDot r m5)) =
      (sumBoolZ 5 (fun r => bSignZ (r i) * zSign (rDotZ r m5Z)) : ℚ) := by
    rw [sumBoolZ_cast, sumBool_eq_sum, sumBool_eq_sum]
    refine Finset.sum_congr rfl fun r _ => ?_
    rw [rDotZ_eq_rDot_mul 8 (by norm_num) m5Z m5 m5_eq_div r, qSign_div_pos _ (by norm_num)]
    cases r i <;> simp [bSign, bSignZ]
  rw [hcongr, hZ i, m5_eq_div i]
  push_cast; ring

theorem m5_sum_lt_two : ∑ i : Fin 5, |m5 i| < 2 := by
  simp [Fin.sum_univ_succ, m5]
  norm_num

theorem m7_p3_tie_free : HasMargin (fun i => (m7 i) ^ 2) (1 / 1024) := by
  refine ⟨by norm_num, ?_⟩
  have h : allBool 7 (fun r => decide (1 ≤ |rDotZ r m7p3Z|)) = true := by decide
  rw [allBool_eq_true_iff] at h
  intro r
  have hr : (1 : ℤ) ≤ |rDotZ r m7p3Z| := of_decide_eq_true (h r)
  rw [rDotZ_eq_rDot_mul 1024 (by norm_num) m7p3Z (fun i => (m7 i) ^ 2) m7p3_eq_div r, abs_div,
    abs_of_pos (by norm_num : (0 : ℚ) < 1024)]
  have hrQ : (1 : ℚ) ≤ |(rDotZ r m7p3Z : ℚ)| := by exact_mod_cast hr
  linarith

theorem m7_p3_chow_fixed : ∀ i : Fin 7, chowParamQ (fun j => (m7 j) ^ 2) i = m7 i := by
  have hZ : ∀ i : Fin 7,
      sumBoolZ 7 (fun r => bSignZ (r i) * zSign (rDotZ r m7p3Z)) =
        if i.val < 3 then 44 else 36 := by
    decide
  intro i
  unfold chowParamQ
  rw [← sumBool_eq_sum]
  have hcongr : sumBool 7 (fun r => bSign (r i) * qSign (rDot r (fun j => (m7 j) ^ 2))) =
      (sumBoolZ 7 (fun r => bSignZ (r i) * zSign (rDotZ r m7p3Z)) : ℚ) := by
    rw [sumBoolZ_cast, sumBool_eq_sum, sumBool_eq_sum]
    refine Finset.sum_congr rfl fun r _ => ?_
    rw [rDotZ_eq_rDot_mul 1024 (by norm_num) m7p3Z (fun j => (m7 j) ^ 2) m7p3_eq_div r,
      qSign_div_pos _ (by norm_num)]
    cases r i <;> simp [bSign, bSignZ]
  rw [hcongr, hZ i]
  fin_cases i <;> norm_num [m7]

theorem m7_p4_tie_free : HasMargin (fun i => (m7 i) ^ 3) (1 / 32768) := by
  refine ⟨by norm_num, ?_⟩
  have h : allBool 7 (fun r => decide (1 ≤ |rDotZ r m7p4Z|)) = true := by decide
  rw [allBool_eq_true_iff] at h
  intro r
  have hr : (1 : ℤ) ≤ |rDotZ r m7p4Z| := of_decide_eq_true (h r)
  rw [rDotZ_eq_rDot_mul 32768 (by norm_num) m7p4Z (fun i => (m7 i) ^ 3) m7p4_eq_div r, abs_div,
    abs_of_pos (by norm_num : (0 : ℚ) < 32768)]
  have hrQ : (1 : ℚ) ≤ |(rDotZ r m7p4Z : ℚ)| := by exact_mod_cast hr
  linarith

theorem m7_p4_chow_fixed : ∀ i : Fin 7, chowParamQ (fun j => (m7 j) ^ 3) i = m7 i := by
  have hZ : ∀ i : Fin 7,
      sumBoolZ 7 (fun r => bSignZ (r i) * zSign (rDotZ r m7p4Z)) =
        if i.val < 3 then 44 else 36 := by
    decide
  intro i
  unfold chowParamQ
  rw [← sumBool_eq_sum]
  have hcongr : sumBool 7 (fun r => bSign (r i) * qSign (rDot r (fun j => (m7 j) ^ 3))) =
      (sumBoolZ 7 (fun r => bSignZ (r i) * zSign (rDotZ r m7p4Z)) : ℚ) := by
    rw [sumBoolZ_cast, sumBool_eq_sum, sumBool_eq_sum]
    refine Finset.sum_congr rfl fun r _ => ?_
    rw [rDotZ_eq_rDot_mul 32768 (by norm_num) m7p4Z (fun j => (m7 j) ^ 3) m7p4_eq_div r,
      qSign_div_pos _ (by norm_num)]
    cases r i <;> simp [bSign, bSignZ]
  rw [hcongr, hZ i]
  fin_cases i <;> norm_num [m7]

/-! ### Part 2: Gayrard's `α^(s)`, Allowable Compositions, and Multiplicity Invariant (Lemma B) -/

/-- Gayrard's Eq. (1.2.1.4): `α^(s) = 2^{-(s-1)} * C(s-1, ⌊(s-1)/2⌋)` for `s ≥ 1`. -/
def alphaVal (s : ℕ) : ℚ :=
  (Nat.choose (s - 1) ((s - 1) / 2) : ℚ) / (2 ^ (s - 1) : ℚ)

lemma alphaVal_pos (s : ℕ) : 0 < alphaVal s := by
  unfold alphaVal
  apply div_pos
  · exact_mod_cast Nat.choose_pos (Nat.div_le_self (s - 1) 2)
  · positivity

lemma alphaVal_one : alphaVal 1 = 1 := by
  norm_num [alphaVal]

/-- Central binomial ratio: `C(2k, k) ≤ 2^(2k - 1)` for all `k ≥ 1`, hence `α^(2k) ≤ 1/2 < 1`
and `α^(2k+1) ≤ 1/2 < 1`. -/
lemma centralBinom_le_half_pow (k : ℕ) (hk : 1 ≤ k) :
    2 * Nat.choose (2 * k) k ≤ 4 ^ k := by
  induction k with
  | zero => omega
  | succ m ih =>
    by_cases hm : m = 0
    · subst hm; norm_num
    · have hm1 : 1 ≤ m := Nat.one_le_iff_ne_zero.mpr hm
      have hrec := ih hm1
      have hcb : Nat.choose (2 * (m + 1)) (m + 1) * (m + 1) =
          Nat.choose (2 * m) m * (2 * (2 * m + 1)) := by
        have h1 : 2 * (m + 1) = 2 * m + 1 + 1 := by omega
        rw [h1]
        have ha := Nat.succ_mul_choose_eq (2 * m + 1) m
        have h3 : Nat.choose (2 * m + 1) m * (m + 1) = Nat.choose (2 * m) m * (2 * m + 1) := by
          have hsym := Nat.choose_symm (show m ≤ 2 * m + 1 by omega)
          have hsub : 2 * m + 1 - m = m + 1 := by omega
          rw [hsub] at hsym
          rw [← hsym]
          have h4 := Nat.succ_mul_choose_eq (2 * m) m
          rw [mul_comm (2 * m + 1)] at h4
          exact h4.symm
        have h5 : (2 * m + 1 + 1) * Nat.choose (2 * m + 1) m =
            Nat.choose (2 * m + 1 + 1) (m + 1) * (m + 1) := ha
        have h6 : Nat.choose (2 * m + 1 + 1) (m + 1) * (m + 1) * (m + 1) =
            Nat.choose (2 * m) m * (2 * (2 * m + 1)) * (m + 1) := by
          calc Nat.choose (2 * m + 1 + 1) (m + 1) * (m + 1) * (m + 1)
              = (2 * m + 1 + 1) * (Nat.choose (2 * m + 1) m * (m + 1)) := by rw [← h5]; ring
            _ = (2 * m + 1 + 1) * (Nat.choose (2 * m) m * (2 * m + 1)) := by rw [h3]
            _ = Nat.choose (2 * m) m * (2 * (2 * m + 1)) * (m + 1) := by ring
        exact Nat.eq_of_mul_eq_mul_right (Nat.succ_pos m) h6
      have hmul : 2 * Nat.choose (2 * (m + 1)) (m + 1) * (m + 1) ≤ 4 ^ (m + 1) * (m + 1) := by
        calc 2 * Nat.choose (2 * (m + 1)) (m + 1) * (m + 1)
            = 2 * (Nat.choose (2 * (m + 1)) (m + 1) * (m + 1)) := by ring
          _ = 2 * (Nat.choose (2 * m) m * (2 * (2 * m + 1))) := by rw [hcb]
          _ = (2 * Nat.choose (2 * m) m) * (2 * (2 * m + 1)) := by ring
          _ ≤ 4 ^ m * (2 * (2 * m + 1)) := Nat.mul_le_mul_right _ hrec
          _ ≤ 4 ^ m * (4 * (m + 1)) := Nat.mul_le_mul_left _ (by omega)
          _ = 4 ^ (m + 1) * (m + 1) := by ring
      exact Nat.le_of_mul_le_mul_right hmul (Nat.succ_pos m)

lemma alphaVal_lt_one_of_ge_two {s : ℕ} (hs : 2 ≤ s) : alphaVal s < 1 := by
  unfold alphaVal
  rw [div_lt_one (by positivity)]
  -- Show C(s-1, (s-1)/2) < 2^(s-1) for all s ≥ 2
  have hlt : Nat.choose (s - 1) ((s - 1) / 2) < 2 ^ (s - 1) := by
    have hsum := Nat.sum_range_choose (s - 1)
    have hmem : (s - 1) / 2 ∈ Finset.range (s - 1 + 1) :=
      Finset.mem_range.mpr (by omega)
    have h0mem : (if (s - 1) / 2 = 0 then 1 else 0) ∈ Finset.range (s - 1 + 1) := by
      split_ifs <;> exact Finset.mem_range.mpr (by omega)
    have hne : (if (s - 1) / 2 = 0 then 1 else 0) ≠ (s - 1) / 2 := by
      split_ifs with h <;> omega
    have hpair : ({(s - 1) / 2, if (s - 1) / 2 = 0 then 1 else 0} : Finset ℕ) ⊆
        Finset.range (s - 1 + 1) := by
      intro x hx
      simp only [Finset.mem_insert, Finset.mem_singleton] at hx
      rcases hx with rfl | rfl <;> assumption
    have hle := Finset.sum_le_sum_of_subset hpair (f := fun j => Nat.choose (s - 1) j)
    rw [hsum, Finset.sum_pair hne.symm] at hle
    have hpos : 0 < Nat.choose (s - 1) (if (s - 1) / 2 = 0 then 1 else 0) := by
      apply Nat.choose_pos
      split_ifs <;> omega
    omega
  exact_mod_cast hlt

/-- An allowable `ℓ`-composition of an odd integer `n`: a list of positive block sizes
`(n₁, …, n_{ℓ-1}, n_ℓ)` where every prefix entry `n_k` is even (`≥ 2`) and the final entry
`n_ℓ` is odd (`≥ 1`). -/
structure AllowableComp where
  prefixBlocks : List ℕ
  prefix_even : ∀ b ∈ prefixBlocks, 2 ≤ b ∧ Even b
  lastBlock : ℕ
  last_odd : Odd lastBlock

/-- The multiset of positive magnitudes `(γ_k repeated n_k times)` produced by an allowable
composition, represented as a list of `(value, multiplicity)` pairs before possible last-block
merging when `lastBlock = 1`. -/
def compBlocksAux (prod : ℚ) : List ℕ → ℕ → List (ℚ × ℕ)
  | [], last => [(prod * alphaVal last, last)]
  | b :: bs, last =>
      let prod' := prod * alphaVal b
      (prod', b) :: compBlocksAux prod' bs last

/-- The full list of `(value, multiplicity)` blocks for an `AllowableComp`. -/
def compBlocks (c : AllowableComp) : List (ℚ × ℕ) :=
  compBlocksAux 1 c.prefixBlocks c.lastBlock

/-- Total count of entries with magnitude `v` in a block list. -/
def blockMult (L : List (ℚ × ℕ)) (v : ℚ) : ℕ :=
  (L.filter (fun p => p.1 == v)).foldr (fun p acc => p.2 + acc) 0

/-- Minimum value appearing in a nonempty block list. -/
def minBlockVal : List (ℚ × ℕ) → ℚ
  | [] => 0
  | [p] => p.1
  | _ :: q :: rest => minBlockVal (q :: rest)

lemma compBlocksAux_strict_decreasing (prod : ℚ) (hprod : 0 < prod)
    (bs : List ℕ) (hbs : ∀ b ∈ bs, 2 ≤ b) (last : ℕ) (hlast : 1 ≤ last) :
    ∀ p ∈ compBlocksAux prod bs last, minBlockVal (compBlocksAux prod bs last) ≤ p.1 := by
  induction bs generalizing prod with
  | nil =>
    intro p hp
    simp [compBlocksAux, minBlockVal] at hp ⊢
    rw [hp]
  | cons b bs ih =>
    have hb2 : 2 ≤ b := hbs b (by simp)
    have hbs' : ∀ x ∈ bs, 2 ≤ x := fun x hx => hbs x (by simp [hx])
    have hprod' : 0 < prod * alphaVal b := mul_pos hprod (alphaVal_pos b)
    have hne : compBlocksAux (prod * alphaVal b) bs last ≠ [] := by
      cases bs <;> simp [compBlocksAux]
    have hmin_eq : minBlockVal (compBlocksAux prod (b :: bs) last) =
        minBlockVal (compBlocksAux (prod * alphaVal b) bs last) := by
      cases bs <;> simp [compBlocksAux, minBlockVal]
    rw [hmin_eq]
    intro p hp
    simp only [compBlocksAux, List.mem_cons] at hp
    rcases hp with rfl | hp'
    · -- Need minBlockVal (compBlocksAux prod' bs last) ≤ prod'
      -- True because the head of compBlocksAux prod' bs last is ≤ prod' (since alphaVal ≤ 1)!
      cases bs with
      | nil =>
        simp only [compBlocksAux, minBlockVal]
        by_cases hl : last = 1
        · subst hl; simp [alphaVal_one]
        · have hl2 : 2 ≤ last := by omega
          have hlt := alphaVal_lt_one_of_ge_two hl2
          nlinarith
      | cons b2 bs2 =>
        have hhead_mem : (prod * alphaVal b * alphaVal b2, b2) ∈
            compBlocksAux (prod * alphaVal b) (b2 :: bs2) last := by
          simp [compBlocksAux]
        have hle := ih (prod * alphaVal b) hprod' hbs' _ hhead_mem
        have hb2_2 : 2 ≤ b2 := hbs' b2 (by simp)
        have hlt := alphaVal_lt_one_of_ge_two hb2_2
        nlinarith
    · exact ih (prod * alphaVal b) hprod' hbs' p hp'

/-- **Lemma B (Multiplicity Invariant of Gayrard's `M_n^{all}`)**:
For EVERY allowable composition `c` and EVERY value `v > minBlockVal (compBlocks c)`,
the total multiplicity `blockMult (compBlocks c) v` is **EVEN**! -/
theorem M_all_multiplicity_even_above_min (c : AllowableComp) (v : ℚ)
    (hv : minBlockVal (compBlocks c) < v) :
    Even (blockMult (compBlocks c) v) := by
  -- Every block in prefixBlocks has even multiplicity, and the last block has value = minBlockVal!
  suffices h : ∀ (prod : ℚ) (bs : List ℕ) (last : ℕ),
      (∀ b ∈ bs, 2 ≤ b ∧ Even b) →
      minBlockVal (compBlocksAux prod bs last) < v →
      Even (blockMult (compBlocksAux prod bs last) v) by
    exact h 1 c.prefixBlocks c.lastBlock c.prefix_even hv
  intro prod bs last hbs hv'
  induction bs generalizing prod with
  | nil =>
    simp only [compBlocksAux, minBlockVal] at hv'
    have hne : (prod * alphaVal last == v) = false := by
      simp [ne_of_lt hv']
    simp [compBlocksAux, blockMult, hne]
  | cons b bs ih =>
    have hb : 2 ≤ b ∧ Even b := hbs b (by simp)
    have hbs' : ∀ x ∈ bs, 2 ≤ x ∧ Even x := fun x hx => hbs x (by simp [hx])
    have hmin_eq : minBlockVal (compBlocksAux prod (b :: bs) last) =
        minBlockVal (compBlocksAux (prod * alphaVal b) bs last) := by
      cases bs <;> simp [compBlocksAux, minBlockVal]
    rw [hmin_eq] at hv'
    have htail := ih (prod * alphaVal b) hbs' hv'
    simp only [compBlocksAux, blockMult, List.filter_cons]
    split_ifs
    · simp only [List.foldr_cons]
      exact hb.2.add htail
    · exact htail

/-- Membership of a vector `m : Fin n → ℚ` in Gayrard's `M_n^{all}` requires some allowable
composition `c` whose block multiplicities match `|{i | |m i| = v}|` for every `v > 0`. -/
def InGayrardMAll {n : ℕ} (m : Fin n → ℚ) : Prop :=
  ∃ c : AllowableComp,
    ∀ v : ℚ, 0 < v →
      (Finset.univ.filter (fun i : Fin n => |m i| = v)).card = blockMult (compBlocks c) v

/-- **Non-membership Corollary for `m5`**: `m5 = (5/8, 3/8, 3/8, 1/8, 1/8) ∉ M_5^{all}`! -/
theorem m5_not_in_M_all : ¬ InGayrardMAll m5 := by
  rintro ⟨c, hc⟩
  have hcard18 : (Finset.univ.filter (fun i : Fin 5 => |m5 i| = 1 / 8)).card = 2 := by
    rw [Finset.card_filter, Fin.sum_univ_succ, Fin.sum_univ_succ, Fin.sum_univ_succ,
      Fin.sum_univ_succ, Fin.sum_univ_one]
    simp only [m5, Matrix.cons_val_zero, Matrix.cons_val_succ]
    norm_num
  have hcard58 : (Finset.univ.filter (fun i : Fin 5 => |m5 i| = 5 / 8)).card = 1 := by
    rw [Finset.card_filter, Fin.sum_univ_succ, Fin.sum_univ_succ, Fin.sum_univ_succ,
      Fin.sum_univ_succ, Fin.sum_univ_one]
    simp only [m5, Matrix.cons_val_zero, Matrix.cons_val_succ]
    norm_num
  have hbm18 : blockMult (compBlocks c) (1 / 8) = 2 := by
    rw [← hc (1 / 8) (by norm_num), hcard18]
  have hbm58 : blockMult (compBlocks c) (5 / 8) = 1 := by
    rw [← hc (5 / 8) (by norm_num), hcard58]
  have hmem18 : ∃ p ∈ compBlocks c, p.1 = 1 / 8 := by
    by_contra hnone
    push_neg at hnone
    have hfilt : (compBlocks c).filter (fun p => p.1 == (1 / 8 : ℚ)) = [] := by
      rw [List.filter_eq_nil_iff]
      intro p hp
      rw [Bool.not_eq_true, beq_eq_false_iff_ne]
      exact hnone p hp
    unfold blockMult at hbm18
    rw [hfilt] at hbm18
    simp at hbm18
  obtain ⟨p, hp_mem, hp_val⟩ := hmem18
  have hmin_le : minBlockVal (compBlocks c) ≤ 1 / 8 := by
    rw [← hp_val]
    exact compBlocksAux_strict_decreasing 1 (by norm_num) c.prefixBlocks
      (fun b hb => (c.prefix_even b hb).1) c.lastBlock
      (by obtain ⟨k, hk⟩ := c.last_odd; omega) p hp_mem
  have heven := M_all_multiplicity_even_above_min c (5 / 8) (by linarith)
  rw [hbm58] at heven
  exact Nat.not_even_one heven

/-- **Non-membership Corollary for `m7`**: `m7 = (11/32^3, 9/32^4) ∉ M_7^{all}`! -/
theorem m7_not_in_M_all : ¬ InGayrardMAll m7 := by
  rintro ⟨c, hc⟩
  have hcard9 : (Finset.univ.filter (fun i : Fin 7 => |m7 i| = 9 / 32)).card = 4 := by
    rw [Finset.card_filter, Fin.sum_univ_succ, Fin.sum_univ_succ, Fin.sum_univ_succ,
      Fin.sum_univ_succ, Fin.sum_univ_succ, Fin.sum_univ_succ, Fin.sum_univ_one]
    simp only [m7, Matrix.cons_val_zero, Matrix.cons_val_succ]
    norm_num
  have hcard11 : (Finset.univ.filter (fun i : Fin 7 => |m7 i| = 11 / 32)).card = 3 := by
    rw [Finset.card_filter, Fin.sum_univ_succ, Fin.sum_univ_succ, Fin.sum_univ_succ,
      Fin.sum_univ_succ, Fin.sum_univ_succ, Fin.sum_univ_succ, Fin.sum_univ_one]
    simp only [m7, Matrix.cons_val_zero, Matrix.cons_val_succ]
    norm_num
  have hbm9 : blockMult (compBlocks c) (9 / 32) = 4 := by
    rw [← hc (9 / 32) (by norm_num), hcard9]
  have hbm11 : blockMult (compBlocks c) (11 / 32) = 3 := by
    rw [← hc (11 / 32) (by norm_num), hcard11]
  have hmem9 : ∃ p ∈ compBlocks c, p.1 = 9 / 32 := by
    by_contra hnone
    push_neg at hnone
    have hfilt : (compBlocks c).filter (fun p => p.1 == (9 / 32 : ℚ)) = [] := by
      rw [List.filter_eq_nil_iff]
      intro p hp
      rw [Bool.not_eq_true, beq_eq_false_iff_ne]
      exact hnone p hp
    unfold blockMult at hbm9
    rw [hfilt] at hbm9
    simp at hbm9
  obtain ⟨p, hp_mem, hp_val⟩ := hmem9
  have hmin_le : minBlockVal (compBlocks c) ≤ 9 / 32 := by
    rw [← hp_val]
    exact compBlocksAux_strict_decreasing 1 (by norm_num) c.prefixBlocks
      (fun b hb => (c.prefix_even b hb).1) c.lastBlock
      (by obtain ⟨k, hk⟩ := c.last_odd; omega) p hp_mem
  have heven := M_all_multiplicity_even_above_min c (11 / 32) (by linarith)
  rw [hbm11] at heven
  obtain ⟨t, ht⟩ := heven
  omega

/-! ### Part 3: Parseval's Identity for Chow Parameters (Universal Dense-Hopfield Lifting Bound) -/

/-- Orthogonality of distinct Rademacher coordinates on `Fin n → Bool`. -/
lemma sum_bSign_mul_bSign {n : ℕ} (i j : Fin n) :
    ∑ r : Fin n → Bool, bSign (r i) * bSign (r j) = if i = j then (2 ^ n : ℚ) else 0 := by
  by_cases hij : i = j
  · subst hij
    have h1 : ∀ r : Fin n → Bool, bSign (r i) * bSign (r i) = 1 := fun r => by
      cases r i <;> norm_num [bSign]
    simp [h1, Fintype.card_bool, Fintype.card_fin]
  · simp only [hij, ↓reduceIte]
    -- Flip coordinate i: bijection r ↦ Function.update r i (! r i) negates the term!
    let flip : (Fin n → Bool) ≃ (Fin n → Bool) :=
      { toFun := fun r => Function.update r i (!(r i))
        invFun := fun r => Function.update r i (!(r i))
        left_inv := fun r => by ext k; by_cases hk : k = i <;> simp [Function.update, hk]
        right_inv := fun r => by ext k; by_cases hk : k = i <;> simp [Function.update, hk] }
    have hneg : ∀ r : Fin n → Bool,
        bSign (flip r i) * bSign (flip r j) = -(bSign (r i) * bSign (r j)) := by
      intro r
      have h1 : flip r i = !(r i) := by simp [flip]
      have h2 : flip r j = r j := by simp [flip, Ne.symm hij]
      rw [h1, h2]
      cases r i <;> cases r j <;> norm_num [bSign]
    have hsum : ∑ r : Fin n → Bool, bSign (r i) * bSign (r j) =
        -(∑ r : Fin n → Bool, bSign (r i) * bSign (r j)) := by
      conv_lhs => rw [← Equiv.sum_comp flip]
      simp [hneg, Finset.sum_neg_distrib]
    linarith

/-- **Parseval / Bessel Inequality for Degree-1 Fourier (Chow) Coefficients**:
For any function `g : (Fin n → Bool) → ℚ` with `∀ r, (g r)^2 ≤ 1`, its Chow coefficients
`c_i = 2^{-n} ∑_r r_i g(r)` satisfy `∑ i, (c_i)^2 ≤ 1`. -/
theorem chow_parseval {n : ℕ} (g : (Fin n → Bool) → ℚ) (hg : ∀ r, (g r) ^ 2 ≤ 1) :
    ∑ i : Fin n, ((∑ r : Fin n → Bool, bSign (r i) * g r) / (2 ^ n : ℚ)) ^ 2 ≤ 1 := by
  set c : Fin n → ℚ := fun i => (∑ r : Fin n → Bool, bSign (r i) * g r) / (2 ^ n : ℚ)
  have h2n_pos : (0 : ℚ) < (2 ^ n : ℚ) := by positivity
  -- Nonnegativity of L² residual: ∑_r (g r - ∑_i c_i r_i)^2 ≥ 0
  have hnn : 0 ≤ ∑ r : Fin n → Bool, (g r - ∑ i : Fin n, c i * bSign (r i)) ^ 2 :=
    Finset.sum_nonneg fun r _ => sq_nonneg _
  have hexp : ∑ r : Fin n → Bool, (g r - ∑ i : Fin n, c i * bSign (r i)) ^ 2 =
      (∑ r : Fin n → Bool, (g r) ^ 2) - (2 ^ n : ℚ) * (∑ i : Fin n, (c i) ^ 2) := by
    have h1 : ∀ r : Fin n → Bool,
        (g r - ∑ i : Fin n, c i * bSign (r i)) ^ 2 =
        (g r) ^ 2 - 2 * g r * (∑ i : Fin n, c i * bSign (r i)) +
          (∑ i : Fin n, c i * bSign (r i)) ^ 2 := fun r => by ring
    simp_rw [h1, Finset.sum_add_distrib, Finset.sum_sub_distrib]
    -- Cross term: ∑_r 2 g r ∑_i c_i r_i = 2 * 2^n * ∑_i c_i^2
    have hcross : ∑ r : Fin n → Bool, 2 * g r * (∑ i : Fin n, c i * bSign (r i)) =
        2 * (2 ^ n : ℚ) * (∑ i : Fin n, (c i) ^ 2) := by
      calc ∑ r : Fin n → Bool, 2 * g r * (∑ i : Fin n, c i * bSign (r i))
          = ∑ r : Fin n → Bool, ∑ i : Fin n, 2 * c i * (bSign (r i) * g r) := by
            refine Finset.sum_congr rfl fun r _ => ?_
            rw [Finset.mul_sum]
            refine Finset.sum_congr rfl fun i _ => by ring
        _ = ∑ i : Fin n, ∑ r : Fin n → Bool, 2 * c i * (bSign (r i) * g r) :=
            Finset.sum_comm
        _ = ∑ i : Fin n, 2 * c i * (∑ r : Fin n → Bool, bSign (r i) * g r) := by
            refine Finset.sum_congr rfl fun i _ => by rw [← Finset.mul_sum]
        _ = ∑ i : Fin n, 2 * (2 ^ n : ℚ) * (c i) ^ 2 := by
            refine Finset.sum_congr rfl fun i _ => ?_
            have hci : ∑ r : Fin n → Bool, bSign (r i) * g r = (2 ^ n : ℚ) * c i := by
              simp only [c]; field_simp
            rw [hci]; ring
        _ = 2 * (2 ^ n : ℚ) * (∑ i : Fin n, (c i) ^ 2) := by rw [← Finset.mul_sum]
    -- Quadratic term: ∑_r (∑_i c_i r_i)^2 = 2^n * ∑_i c_i^2
    have hquad : ∑ r : Fin n → Bool, (∑ i : Fin n, c i * bSign (r i)) ^ 2 =
        (2 ^ n : ℚ) * (∑ i : Fin n, (c i) ^ 2) := by
      calc ∑ r : Fin n → Bool, (∑ i : Fin n, c i * bSign (r i)) ^ 2
          = ∑ r : Fin n → Bool, ∑ i : Fin n, ∑ j : Fin n,
              c i * c j * (bSign (r i) * bSign (r j)) := by
            refine Finset.sum_congr rfl fun r _ => ?_
            rw [sq, Finset.sum_mul_sum]
            refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring
        _ = ∑ i : Fin n, ∑ j : Fin n, c i * c j *
              (∑ r : Fin n → Bool, bSign (r i) * bSign (r j)) := by
            rw [Finset.sum_comm]
            refine Finset.sum_congr rfl fun i _ => ?_
            rw [Finset.sum_comm]
            refine Finset.sum_congr rfl fun j _ => by rw [← Finset.mul_sum]
        _ = ∑ i : Fin n, (2 ^ n : ℚ) * (c i) ^ 2 := by
            refine Finset.sum_congr rfl fun i _ => ?_
            simp_rw [sum_bSign_mul_bSign]
            simp [sq, mul_comm]
        _ = (2 ^ n : ℚ) * (∑ i : Fin n, (c i) ^ 2) := by rw [← Finset.mul_sum]
    rw [hcross, hquad]; ring
  have hg_sum : ∑ r : Fin n → Bool, (g r) ^ 2 ≤ (2 ^ n : ℚ) := by
    calc ∑ r : Fin n → Bool, (g r) ^ 2
        ≤ ∑ _r : Fin n → Bool, (1 : ℚ) := Finset.sum_le_sum fun r _ => hg r
      _ = (2 ^ n : ℚ) := by simp [Fintype.card_bool, Fintype.card_fin]
  have hmul : (2 ^ n : ℚ) * (∑ i : Fin n, (c i) ^ 2) ≤ (2 ^ n : ℚ) * 1 := by linarith
  exact le_of_mul_le_mul_left hmul h2n_pos

/-- **Universal Dense-Hopfield Lifting Bound (Step 2 of Theorem 2)**:
For any `p ≥ 3` and any Chow fixed point `m̃ ∈ [0, 1]^n` of `F'(x) = x^(p-1)`,
`∑ i, (m̃ i)^(p - 1) ≤ 1 < 2`, enabling lifting to all odd `n + 2k`. -/
theorem dense_lift_sum_lt_two {n p : ℕ} (hp : 3 ≤ p) (m : Fin n → ℚ)
    (hm_range : ∀ i, 0 ≤ m i ∧ m i ≤ 1)
    (hfixed : ∀ i, chowParamQ (fun j => (m j) ^ (p - 1)) i = m i) :
    ∑ i : Fin n, (m i) ^ (p - 1) < 2 := by
  have hparseval : ∑ i : Fin n, (m i) ^ 2 ≤ 1 := by
    have h := chow_parseval (fun r => qSign (rDot r (fun j => (m j) ^ (p - 1)))) (fun r => by
      dsimp only; unfold qSign; split_ifs <;> norm_num)
    refine le_trans ?_ h
    refine Finset.sum_le_sum fun i _ => ?_
    rw [← hfixed i]; rfl
  have hle : ∑ i : Fin n, (m i) ^ (p - 1) ≤ ∑ i : Fin n, (m i) ^ 2 := by
    refine Finset.sum_le_sum fun i _ => ?_
    exact pow_le_pow_of_le_one (hm_range i).1 (hm_range i).2 (by omega)
  linarith

end HopfieldMixedMemories
