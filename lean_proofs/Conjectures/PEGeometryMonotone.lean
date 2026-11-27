import Mathlib

/-!
# Counterexample to Positional Monotonicity
(G. Cirrincione, *On the Geometry of Positional Encodings in Transformers*, arXiv:2604.05217)

We formalize in Lean 4 + Mathlib (zero `sorry`, standard axioms only) the decisive refutation of
Conjecture `conj:monotone`, the Appendix A `A1` PSD impossibility, and the raw-stress minimum:

1. **Corpus & exact squared Hellinger distances**:
   - `sqrtProb`, `hellingerSq`: Translates of `(1, 2, 3) / √14` on vocabulary `Fin 5` (and `Fin (n + 2)`).
   - `hellingerSq_lag0`, `hellingerSq_lag1`, `hellingerSq_lag2`, `hellingerSq_lag3`:
     Exact rational values `0`, `6/7`, `11/7`, `2`, satisfying `0 < 6/7 < 11/7 < 2`.
2. **Tag marginals `(1, 9, 4) / 14`**:
   - `tag_marginals`: With binary tags `g = (1, 0, 0, 1, 0, 1, 1)`, the positive-label marginals
     at positions `0, 1, 2` (1-indexed `1, 2, 3`) are `(1/14, 9/14, 4/14)`, giving `r₁ < r₃ < r₂`.
3. **Unique BCE global minimizer and strict ordering `p₁* < p₃* < p₂*`**:
   - `hasDerivAt_F`, `F'_strictMono`, `F'_pStar_eq_zero`, `pStar_unique_global_min`:
     For every `0 < r < 1`, the binary cross-entropy loss `F r x` has unique global minimizer
     `pStar r = log zStar r`.
   - `pStar_strictMono`: `pStar` is strictly increasing on `(0, 1)`.
   - `counterexample_strict_order`: Setting `p₁ = pStar (1/14)`, `p₂ = pStar (9/14)`,
     `p₃ = pStar (4/14)`, we have `p₁ < p₃ < p₂`, hence `|p₁ - p₂| > |p₁ - p₃|` even though
     `d_H²(μ₁, μ₂) = 6/7 < 11/7 = d_H²(μ₁, μ₃)`.
4. **Appendix A `A1` PSD impossibility**:
   - `appendix_A1_psd_impossible`: Whenever `α_{ab} = f(d(a, b))` with `f` strictly increasing,
     `d(a, a) = 0`, `d(a, b) = d(b, a)`, and `d(i, j) > 0`, the quadratic form at `e_i - e_j` is
     `2 (f(0) - f(d(i, j))) < 0`, so `α` cannot be positive semidefinite (let alone positive definite).
5. **Raw-stress global optimum `u = v = 2/3`**:
   - `rawStress_sum_of_squares`, `rawStress_unique_global_min`:
     `((u - 1)^2 + (v - 1)^2 + (u + v - 1)^2) / 3` has unique global minimum `1/9` at `u = v = 2/3`.
   - `rawStress_order_reversal`: Placing `p₁ = 0, p₃ = 2/3, p₂ = 4/3` attains minimum stress `1/9`
     while reversing `|p₁ - p₂| = 4/3 > 2/3 = |p₁ - p₃|`.

Machine-checked vs prose-only:
- Machine-checked: All exact Hellinger distances `0 < 6/7 < 11/7 < 2`, tag marginals `(1, 9, 4)/14`,
  the exact BCE derivative, strict monotonicity and unique global optimality of `pStar`, the
  strict reversal `p₁* < p₃* < p₂*` and `|p₁* - p₂*| > |p₁* - p₃*|`, Appendix A `A1` PSD
  impossibility, and the unique raw-stress minimum at `u = v = 2/3`.
- Prose-only (in `solutions/pe_geometry_monotone/results.md`): Verification of the active
  attention perturbation (`ε = 10⁻⁴`) and swap margin `2β/196` over full sequence enumerations,
  higher-dimension transverse penalty extension, and decimal approximations.
-/

open BigOperators Finset

namespace PEGeometryMonotone

/-! ### 1. Corpus and Exact Squared Hellinger Distances -/

/-- Square-root probability vector for position `i`: `(1, 2, 3) / √14` supported at `i, i+1, i+2`. -/
noncomputable def sqrtProb (i v : ℕ) : ℝ :=
  if v = i then 1 / Real.sqrt 14
  else if v = i + 1 then 2 / Real.sqrt 14
  else if v = i + 2 then 3 / Real.sqrt 14
  else 0

/-- Squared Hellinger distance `∑_{v < V} (√μ_i(v) - √μ_j(v))^2` (unnormalized chord distance). -/
noncomputable def hellingerSq (i j V : ℕ) : ℝ :=
  ∑ v ∈ Finset.range V, (sqrtProb i v - sqrtProb j v) ^ 2

/-- Probability normalization: `∑_{v < V} (sqrtProb i v)^2 = 1` whenever `i + 3 ≤ V`. -/
theorem prob_sum_one {i V : ℕ} (hV : i + 3 ≤ V) :
    ∑ v ∈ Finset.range V, (sqrtProb i v) ^ 2 = 1 := by
  have hsub : ∀ v ∈ Finset.range V, v ∉ ({i, i + 1, i + 2} : Finset ℕ) →
      (sqrtProb i v) ^ 2 = 0 := by
    intro v _ hv
    simp only [Finset.mem_insert, Finset.mem_singleton, not_or] at hv
    simp [sqrtProb, hv.1, hv.2.1, hv.2.2]
  have hfin : ({i, i + 1, i + 2} : Finset ℕ) ⊆ Finset.range V := by
    intro v hv
    simp only [Finset.mem_insert, Finset.mem_singleton] at hv
    rw [Finset.mem_range]
    omega
  rw [← Finset.sum_subset hfin hsub]
  have h1 : i ∉ ({i + 1, i + 2} : Finset ℕ) := by simp
  have h2 : i + 1 ∉ ({i + 2} : Finset ℕ) := by simp
  rw [Finset.sum_insert h1, Finset.sum_insert h2, Finset.sum_singleton]
  have hne1 : i + 1 ≠ i := by omega
  have hne2 : i + 2 ≠ i := by omega
  have hne3 : i + 2 ≠ i + 1 := by omega
  simp only [sqrtProb, ↓reduceIte, hne1, hne2, hne3]
  have hsq14 : (Real.sqrt 14) ^ 2 = 14 := Real.sq_sqrt (by norm_num)
  rw [div_pow, div_pow, div_pow, hsq14]
  norm_num

/-- Lag 0 squared Hellinger distance is `0`. -/
theorem hellingerSq_lag0 (i V : ℕ) : hellingerSq i i V = 0 := by
  simp [hellingerSq]

/-- Lag 1 squared Hellinger distance is `6 / 7` whenever `i + 4 ≤ V`. -/
theorem hellingerSq_lag1 {i V : ℕ} (hV : i + 4 ≤ V) :
    hellingerSq i (i + 1) V = 6 / 7 := by
  unfold hellingerSq
  let S : Finset ℕ := {i, i + 1, i + 2, i + 3}
  have hfin : S ⊆ Finset.range V := by
    intro v hv
    simp only [S, Finset.mem_insert, Finset.mem_singleton] at hv
    rw [Finset.mem_range]; omega
  have hzero : ∀ v ∈ Finset.range V, v ∉ S →
      (sqrtProb i v - sqrtProb (i + 1) v) ^ 2 = 0 := by
    intro v _ hv
    simp only [S, Finset.mem_insert, Finset.mem_singleton, not_or] at hv
    rcases hv with ⟨h0, h1, h2, h3⟩
    have h3' : v ≠ i + 1 + 2 := by omega
    simp [sqrtProb, h0, h1, h2, h3']
  rw [← Finset.sum_subset hfin hzero]
  have hm0 : i ∉ ({i + 1, i + 2, i + 3} : Finset ℕ) := by simp
  have hm1 : i + 1 ∉ ({i + 2, i + 3} : Finset ℕ) := by simp
  have hm2 : i + 2 ∉ ({i + 3} : Finset ℕ) := by simp
  dsimp only [S]
  rw [Finset.sum_insert hm0, Finset.sum_insert hm1, Finset.sum_insert hm2, Finset.sum_singleton]
  have e0 : sqrtProb i i = 1 / Real.sqrt 14 := by simp [sqrtProb]
  have e1 : sqrtProb (i + 1) i = 0 := by
    have a2 : i ≠ i + 1 + 1 := by omega
    have a3 : i ≠ i + 1 + 2 := by omega
    simp [sqrtProb, a2, a3]
  have e2 : sqrtProb i (i + 1) = 2 / Real.sqrt 14 := by simp [sqrtProb]
  have e3 : sqrtProb (i + 1) (i + 1) = 1 / Real.sqrt 14 := by simp [sqrtProb]
  have e4 : sqrtProb i (i + 2) = 3 / Real.sqrt 14 := by simp [sqrtProb]
  have e5 : sqrtProb (i + 1) (i + 2) = 2 / Real.sqrt 14 := by
    have a2 : i + 2 = i + 1 + 1 := by omega
    simp [sqrtProb, a2]
  have e6 : sqrtProb i (i + 3) = 0 := by simp [sqrtProb]
  have e7 : sqrtProb (i + 1) (i + 3) = 3 / Real.sqrt 14 := by
    have a3 : i + 3 = i + 1 + 2 := by omega
    simp [sqrtProb, a3]
  rw [e0, e1, e2, e3, e4, e5, e6, e7]
  have hsq14 : (Real.sqrt 14) ^ 2 = 14 := Real.sq_sqrt (by norm_num)
  have hne : Real.sqrt 14 ≠ 0 := by positivity
  field_simp [hne]
  rw [hsq14]
  norm_num

/-- Lag 2 squared Hellinger distance is `11 / 7` whenever `i + 5 ≤ V`. -/
theorem hellingerSq_lag2 {i V : ℕ} (hV : i + 5 ≤ V) :
    hellingerSq i (i + 2) V = 11 / 7 := by
  unfold hellingerSq
  let S : Finset ℕ := {i, i + 1, i + 2, i + 3, i + 4}
  have hfin : S ⊆ Finset.range V := by
    intro v hv
    simp only [S, Finset.mem_insert, Finset.mem_singleton] at hv
    rw [Finset.mem_range]; omega
  have hzero : ∀ v ∈ Finset.range V, v ∉ S →
      (sqrtProb i v - sqrtProb (i + 2) v) ^ 2 = 0 := by
    intro v _ hv
    simp only [S, Finset.mem_insert, Finset.mem_singleton, not_or] at hv
    rcases hv with ⟨h0, h1, h2, h3, h4⟩
    have h3' : v ≠ i + 2 + 1 := by omega
    have h4' : v ≠ i + 2 + 2 := by omega
    simp [sqrtProb, h0, h1, h2, h3', h4']
  rw [← Finset.sum_subset hfin hzero]
  have hm0 : i ∉ ({i + 1, i + 2, i + 3, i + 4} : Finset ℕ) := by simp
  have hm1 : i + 1 ∉ ({i + 2, i + 3, i + 4} : Finset ℕ) := by simp
  have hm2 : i + 2 ∉ ({i + 3, i + 4} : Finset ℕ) := by simp
  have hm3 : i + 3 ∉ ({i + 4} : Finset ℕ) := by simp
  dsimp only [S]
  rw [Finset.sum_insert hm0, Finset.sum_insert hm1, Finset.sum_insert hm2,
    Finset.sum_insert hm3, Finset.sum_singleton]
  have e0 : sqrtProb i i = 1 / Real.sqrt 14 := by simp [sqrtProb]
  have e1 : sqrtProb (i + 2) i = 0 := by
    have a2 : i ≠ i + 2 + 1 := by omega
    have a3 : i ≠ i + 2 + 2 := by omega
    simp [sqrtProb, a2, a3]
  have e2 : sqrtProb i (i + 1) = 2 / Real.sqrt 14 := by simp [sqrtProb]
  have e3 : sqrtProb (i + 2) (i + 1) = 0 := by simp [sqrtProb]
  have e4 : sqrtProb i (i + 2) = 3 / Real.sqrt 14 := by simp [sqrtProb]
  have e5 : sqrtProb (i + 2) (i + 2) = 1 / Real.sqrt 14 := by simp [sqrtProb]
  have e6 : sqrtProb i (i + 3) = 0 := by simp [sqrtProb]
  have e7 : sqrtProb (i + 2) (i + 3) = 2 / Real.sqrt 14 := by
    have a2 : i + 3 = i + 2 + 1 := by omega
    simp [sqrtProb, a2]
  have e8 : sqrtProb i (i + 4) = 0 := by simp [sqrtProb]
  have e9 : sqrtProb (i + 2) (i + 4) = 3 / Real.sqrt 14 := by
    have a3 : i + 4 = i + 2 + 2 := by omega
    simp [sqrtProb, a3]
  rw [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9]
  have hsq14 : (Real.sqrt 14) ^ 2 = 14 := Real.sq_sqrt (by norm_num)
  have hne : Real.sqrt 14 ≠ 0 := by positivity
  field_simp [hne]
  rw [hsq14]
  norm_num

/-- Lag `≥ 3` squared Hellinger distance is `2` whenever `j + 3 ≤ V` and `i + 3 ≤ j`. -/
theorem hellingerSq_lag_ge3 {i j V : ℕ} (hij : i + 3 ≤ j) (hV : j + 3 ≤ V) :
    hellingerSq i j V = 2 := by
  unfold hellingerSq
  have hprod_zero : ∀ v, sqrtProb i v * sqrtProb j v = 0 := by
    intro v
    by_cases hv : v ≤ i + 2
    · have a1 : v ≠ j := by omega
      have a2 : v ≠ j + 1 := by omega
      have a3 : v ≠ j + 2 := by omega
      simp [sqrtProb, a1, a2, a3]
    · have a1 : v ≠ i := by omega
      have a2 : v ≠ i + 1 := by omega
      have a3 : v ≠ i + 2 := by omega
      simp [sqrtProb, a1, a2, a3]
  calc ∑ v ∈ Finset.range V, (sqrtProb i v - sqrtProb j v) ^ 2
      = ∑ v ∈ Finset.range V, ((sqrtProb i v) ^ 2 + (sqrtProb j v) ^ 2) := by
          refine Finset.sum_congr rfl fun v _ => ?_
          have hp := hprod_zero v
          linarith [show (sqrtProb i v - sqrtProb j v) ^ 2 =
            (sqrtProb i v) ^ 2 + (sqrtProb j v) ^ 2 - 2 * (sqrtProb i v * sqrtProb j v) by ring]
    _ = (∑ v ∈ Finset.range V, (sqrtProb i v) ^ 2) +
          (∑ v ∈ Finset.range V, (sqrtProb j v) ^ 2) := Finset.sum_add_distrib
    _ = 1 + 1 := by rw [prob_sum_one (by omega), prob_sum_one hV]
    _ = 2 := by norm_num

/-- Strict Hellinger lag chain: `0 < 6/7 < 11/7 < 2`. -/
theorem hellingerSq_strict_chain :
    (0 : ℝ) < 6 / 7 ∧ (6 : ℝ) / 7 < 11 / 7 ∧ (11 : ℝ) / 7 < 2 := by
  norm_num

/-! ### 2. Binary Tag Marginals `(1, 9, 4) / 14` -/

/-- Binary tags `g = (1, 0, 0, 1, 0, 1, 1)` on vocabulary `0..6`. -/
def tagQ : ℕ → ℚ
  | 0 => 1
  | 1 => 0
  | 2 => 0
  | 3 => 1
  | 4 => 0
  | 5 => 1
  | 6 => 1
  | _ => 0

/-- Positive-label probability `r_i = Pr(Y_i = 1)` at position `i`. -/
def posProb (i : ℕ) : ℚ :=
  (1 / 14) * tagQ i + (4 / 14) * tagQ (i + 1) + (9 / 14) * tagQ (i + 2)

/-- Exact tag marginals `(1, 9, 4, 10, 13) / 14` and strict non-monotonicity `r₁ < r₃ < r₂`. -/
theorem tag_marginals :
    posProb 0 = 1 / 14 ∧ posProb 1 = 9 / 14 ∧ posProb 2 = 4 / 14 ∧
    posProb 3 = 10 / 14 ∧ posProb 4 = 13 / 14 ∧
    posProb 0 < posProb 2 ∧ posProb 2 < posProb 1 := by
  unfold posProb tagQ
  norm_num

/-! ### 3. Exact BCE Global Minimizer and Strict Ordering `p₁* < p₃* < p₂*` -/

/-- Single-position expected binary cross-entropy loss with `β = log 2`. -/
noncomputable def F (r x : ℝ) : ℝ :=
  (1 - r) * Real.log (1 + Real.exp x) + r * Real.log (1 + Real.exp (-x - Real.log 2))

/-- First derivative of `F r` at `x`. -/
noncomputable def F' (r x : ℝ) : ℝ :=
  (1 - r) * (Real.exp x / (1 + Real.exp x)) - r / (1 + 2 * Real.exp x)

lemma exp_neg_sub_log2 (x : ℝ) :
    Real.exp (-x - Real.log 2) = 1 / (2 * Real.exp x) := by
  rw [sub_eq_add_neg, Real.exp_add, Real.exp_neg, Real.exp_neg, Real.exp_log (by norm_num : (0 : ℝ) < 2)]
  field_simp [ne_of_gt (Real.exp_pos x)]

/-- Exact derivative of `F r`. -/
theorem hasDerivAt_F (r x : ℝ) : HasDerivAt (F r) (F' r x) x := by
  have hexp_pos : 0 < Real.exp x := Real.exp_pos x
  have h1_pos : 0 < 1 + Real.exp x := by linarith
  have h2_pos : 0 < 1 + Real.exp (-x - Real.log 2) := by positivity
  have hd1 : HasDerivAt (fun t => Real.log (1 + Real.exp t))
      (Real.exp x / (1 + Real.exp x)) x := by
    have hinner : HasDerivAt (fun t => 1 + Real.exp t) (Real.exp x) x :=
      (Real.hasDerivAt_exp x).const_add 1
    exact hinner.log (ne_of_gt h1_pos)
  have hd2 : HasDerivAt (fun t => Real.log (1 + Real.exp (-t - Real.log 2)))
      (-Real.exp (-x - Real.log 2) / (1 + Real.exp (-x - Real.log 2))) x := by
    have hlin : HasDerivAt (fun t => -t - Real.log 2) (-1) x :=
      (hasDerivAt_id x).neg.sub_const (Real.log 2)
    have hexp : HasDerivAt (fun t => Real.exp (-t - Real.log 2))
        (Real.exp (-x - Real.log 2) * (-1)) x := by
      simpa [Function.comp_def] using (Real.hasDerivAt_exp (-x - Real.log 2)).comp x hlin
    have hinner : HasDerivAt (fun t => 1 + Real.exp (-t - Real.log 2))
        (Real.exp (-x - Real.log 2) * (-1)) x := hexp.const_add 1
    convert hinner.log (ne_of_gt h2_pos) using 1
    ring
  have hcomb := (hd1.const_mul (1 - r)).add (hd2.const_mul r)
  convert hcomb using 1
  unfold F'
  rw [exp_neg_sub_log2]
  have hne1 : 2 * Real.exp x ≠ 0 := by positivity
  have hne2 : 1 + 2 * Real.exp x ≠ 0 := by positivity
  field_simp [hne1, hne2]
  ring

/-- Closed-form positive root `z*(r)` of `2(1 - r) z^2 + (1 - 2r) z - r = 0`. -/
noncomputable def zStar (r : ℝ) : ℝ :=
  (2 * r - 1 + Real.sqrt ((2 * r - 1) ^ 2 + 8 * r * (1 - r))) / (4 * (1 - r))

/-- Unique global minimizer `p*(r) = log z*(r)`. -/
noncomputable def pStar (r : ℝ) : ℝ :=
  Real.log (zStar r)

lemma zStar_pos {r : ℝ} (hr0 : 0 < r) (hr1 : r < 1) : 0 < zStar r := by
  unfold zStar
  have hden : 0 < 4 * (1 - r) := by linarith
  apply div_pos _ hden
  have h8 : 0 < 8 * r * (1 - r) := by nlinarith
  have hlt : |2 * r - 1| < Real.sqrt ((2 * r - 1) ^ 2 + 8 * r * (1 - r)) := by
    rw [← Real.sqrt_sq_eq_abs]
    exact Real.sqrt_lt_sqrt (sq_nonneg _) (by linarith)
  have hneg : -(2 * r - 1) ≤ |2 * r - 1| := neg_le_abs _
  linarith

lemma zStar_quad {r : ℝ} (hr0 : 0 < r) (hr1 : r < 1) :
    2 * (1 - r) * (zStar r) ^ 2 + (1 - 2 * r) * zStar r - r = 0 := by
  unfold zStar
  have h1r : 1 - r ≠ 0 := by linarith
  have hden : 4 * (1 - r) ≠ 0 := by linarith
  have hdisc_nonneg : 0 ≤ (2 * r - 1) ^ 2 + 8 * r * (1 - r) := by nlinarith
  set s := Real.sqrt ((2 * r - 1) ^ 2 + 8 * r * (1 - r))
  have hsq : s ^ 2 = (2 * r - 1) ^ 2 + 8 * r * (1 - r) := Real.sq_sqrt hdisc_nonneg
  have hnum : 2 * (1 - r) * (2 * r - 1 + s) ^ 2 +
      (1 - 2 * r) * (2 * r - 1 + s) * (4 * (1 - r)) - r * (4 * (1 - r)) ^ 2 = 0 := by
    have hfac : 2 * (1 - r) * (2 * r - 1 + s) ^ 2 +
        (1 - 2 * r) * (2 * r - 1 + s) * (4 * (1 - r)) - r * (4 * (1 - r)) ^ 2 =
        2 * (1 - r) * (s ^ 2 - ((2 * r - 1) ^ 2 + 8 * r * (1 - r))) := by ring
    rw [hfac, hsq]; ring
  calc 2 * (1 - r) * ((2 * r - 1 + s) / (4 * (1 - r))) ^ 2 +
        (1 - 2 * r) * ((2 * r - 1 + s) / (4 * (1 - r))) - r
      = (2 * (1 - r) * (2 * r - 1 + s) ^ 2 +
          (1 - 2 * r) * (2 * r - 1 + s) * (4 * (1 - r)) - r * (4 * (1 - r)) ^ 2) /
          (4 * (1 - r)) ^ 2 := by field_simp [hden]
    _ = 0 := by rw [hnum, zero_div]

/-- First-order optimality: `F' r (pStar r) = 0`. -/
theorem F'_pStar_eq_zero {r : ℝ} (hr0 : 0 < r) (hr1 : r < 1) :
    F' r (pStar r) = 0 := by
  unfold F' pStar
  have hz : 0 < zStar r := zStar_pos hr0 hr1
  rw [Real.exp_log hz]
  have h1 : 1 + zStar r ≠ 0 := by linarith
  have h2 : 1 + 2 * zStar r ≠ 0 := by linarith
  field_simp [h1, h2]
  have hq := zStar_quad hr0 hr1
  linarith

/-- Strict convexity: `F' r` is strictly increasing on `ℝ` for every `0 < r < 1`. -/
theorem F'_strictMono {r : ℝ} (hr0 : 0 < r) (hr1 : r < 1) : StrictMono (F' r) := by
  intro x y hxy
  unfold F'
  have hex : Real.exp x < Real.exp y := Real.exp_lt_exp.2 hxy
  have hxpos : 0 < Real.exp x := Real.exp_pos x
  have hypos : 0 < Real.exp y := Real.exp_pos y
  have hterm1 : Real.exp x / (1 + Real.exp x) < Real.exp y / (1 + Real.exp y) := by
    rw [div_lt_div_iff₀ (by linarith) (by linarith)]
    nlinarith
  have hterm2 : r / (1 + 2 * Real.exp y) < r / (1 + 2 * Real.exp x) := by
    rw [div_lt_div_iff₀ (by linarith) (by linarith)]
    nlinarith
  have h1 : (1 - r) * (Real.exp x / (1 + Real.exp x)) <
      (1 - r) * (Real.exp y / (1 + Real.exp y)) :=
    mul_lt_mul_of_pos_left hterm1 (by linarith)
  linarith

/-- **Unique global minimizer**: for `0 < r < 1` and any `x ≠ pStar r`, `F r (pStar r) < F r x`. -/
theorem pStar_unique_global_min {r : ℝ} (hr0 : 0 < r) (hr1 : r < 1) {x : ℝ}
    (hne : x ≠ pStar r) :
    F r (pStar r) < F r x := by
  have hzero := F'_pStar_eq_zero hr0 hr1
  have hmono := F'_strictMono hr0 hr1
  rcases lt_or_gt_of_ne hne with hlt | hgt
  · -- x < pStar r: by MVT, F(pStar r) - F(x) = F'(c) * (pStar r - x) < 0
    obtain ⟨c, hc, hslope⟩ := exists_deriv_eq_slope (F r) hlt
      (fun t _ => (hasDerivAt_F r t).continuousAt.continuousWithinAt)
      (fun t _ => (hasDerivAt_F r t).differentiableAt.differentiableWithinAt)
    rw [(hasDerivAt_F r c).deriv] at hslope
    have hFc_neg : F' r c < 0 := by
      rw [← hzero]; exact hmono hc.2
    have hdiff : F r (pStar r) - F r x = F' r c * (pStar r - x) := by
      have hne' : pStar r - x ≠ 0 := by linarith
      field_simp [hne'] at hslope
      linarith
    have : F' r c * (pStar r - x) < 0 := mul_neg_of_neg_of_pos hFc_neg (by linarith)
    linarith
  · -- pStar r < x: by MVT, F(x) - F(pStar r) = F'(c) * (x - pStar r) > 0
    obtain ⟨c, hc, hslope⟩ := exists_deriv_eq_slope (F r) hgt
      (fun t _ => (hasDerivAt_F r t).continuousAt.continuousWithinAt)
      (fun t _ => (hasDerivAt_F r t).differentiableAt.differentiableWithinAt)
    rw [(hasDerivAt_F r c).deriv] at hslope
    have hFc_pos : 0 < F' r c := by
      rw [← hzero]; exact hmono hc.1
    have hdiff : F r x - F r (pStar r) = F' r c * (x - pStar r) := by
      have hne' : x - pStar r ≠ 0 := by linarith
      field_simp [hne'] at hslope
      linarith
    have : 0 < F' r c * (x - pStar r) := mul_pos hFc_pos (by linarith)
    linarith

/-- Strict monotonicity of the minimizer `r ↦ pStar r` on `(0, 1)`. -/
theorem pStar_strictMono {r₁ r₂ : ℝ} (hr₁0 : 0 < r₁) (hr₁₂ : r₁ < r₂) (hr₂1 : r₂ < 1) :
    pStar r₁ < pStar r₂ := by
  have h1 : F' r₁ (pStar r₁) = 0 := F'_pStar_eq_zero hr₁0 (lt_trans hr₁₂ hr₂1)
  have h2 : F' r₂ (pStar r₂) = 0 := F'_pStar_eq_zero (lt_trans hr₁0 hr₁₂) hr₂1
  have hcmp : F' r₂ (pStar r₁) < F' r₁ (pStar r₁) := by
    unfold F'
    have hexp : 0 < Real.exp (pStar r₁) := Real.exp_pos _
    have hpos : 0 < Real.exp (pStar r₁) / (1 + Real.exp (pStar r₁)) +
        1 / (1 + 2 * Real.exp (pStar r₁)) := by positivity
    have hdiff : F' r₁ (pStar r₁) - F' r₂ (pStar r₁) =
        (r₂ - r₁) * (Real.exp (pStar r₁) / (1 + Real.exp (pStar r₁)) +
          1 / (1 + 2 * Real.exp (pStar r₁))) := by
      unfold F'; ring
    have : 0 < (r₂ - r₁) * (Real.exp (pStar r₁) / (1 + Real.exp (pStar r₁)) +
        1 / (1 + 2 * Real.exp (pStar r₁))) := mul_pos (by linarith) hpos
    unfold F' at hdiff
    linarith
  have hlt : F' r₂ (pStar r₁) < F' r₂ (pStar r₂) := by linarith
  by_contra hge
  push_neg at hge
  have hmono := F'_strictMono (lt_trans hr₁0 hr₁₂) hr₂1
  have : F' r₂ (pStar r₂) ≤ F' r₂ (pStar r₁) := hmono.monotone hge
  linarith

/-- **Decisive refutation of `conj:monotone`**: with tag marginals `(1, 9, 4) / 14`, the unique
global minimizers satisfy `p₁* < p₃* < p₂*`, hence `|p₁* - p₂*| > |p₁* - p₃*|`, reversing the
strict Hellinger distance ordering `d_H²(μ₁, μ₂) = 6/7 < 11/7 = d_H²(μ₁, μ₃)`. -/
theorem counterexample_strict_order :
    let p₁ := pStar (1 / 14)
    let p₂ := pStar (9 / 14)
    let p₃ := pStar (4 / 14)
    p₁ < p₃ ∧ p₃ < p₂ ∧ |p₁ - p₃| < |p₁ - p₂| ∧
    hellingerSq 0 1 5 < hellingerSq 0 2 5 := by
  intro p₁ p₂ p₃
  have h13 : p₁ < p₃ := pStar_strictMono (by norm_num) (by norm_num) (by norm_num)
  have h32 : p₃ < p₂ := pStar_strictMono (by norm_num) (by norm_num) (by norm_num)
  have habs : |p₁ - p₃| < |p₁ - p₂| := by
    rw [abs_of_neg (by linarith), abs_of_neg (by linarith)]
    linarith
  have hhell : hellingerSq 0 1 5 < hellingerSq 0 2 5 := by
    rw [hellingerSq_lag1 (by omega), hellingerSq_lag2 (by omega)]
    norm_num
  exact ⟨h13, h32, habs, hhell⟩

/-! ### 4. Appendix A `A1` Positive-Semidefinite Impossibility -/

/-- **Appendix A `A1` impossibility**: if `α_{ab} = f(d(a, b))` with `f` strictly increasing,
`d(a, a) = 0`, `d(a, b) = d(b, a)`, and `d(i, j) > 0`, then `(e_i - e_j)ᵀ α (e_i - e_j) < 0`. -/
theorem appendix_A1_psd_impossible {n : ℕ}
    (d : Fin n → Fin n → ℝ) (hd_diag : ∀ a, d a a = 0) (hd_sym : ∀ a b, d a b = d b a)
    (f : ℝ → ℝ) (hf : StrictMono f)
    (α : Matrix (Fin n) (Fin n) ℝ) (hα : ∀ a b, α a b = f (d a b))
    {i j : Fin n} (hij : 0 < d i j) :
    let v : Fin n → ℝ := fun a => (if a = i then 1 else 0) - (if a = j then 1 else 0)
    dotProduct v (α.mulVec v) = 2 * (f 0 - f (d i j)) ∧
    dotProduct v (α.mulVec v) < 0 := by
  intro v
  have hne : i ≠ j := by
    intro h; subst h; linarith [hd_diag i]
  have hmul : ∀ a, (α.mulVec v) a = α a i - α a j := by
    intro a
    unfold Matrix.mulVec dotProduct v
    simp only [mul_sub, Finset.sum_sub_distrib, mul_ite, mul_one, mul_zero,
      Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte]
  have hquad : dotProduct v (α.mulVec v) = α i i - α i j - α j i + α j j := by
    unfold dotProduct
    simp_rw [hmul]
    unfold v
    simp only [sub_mul, ite_mul, one_mul, zero_mul, Finset.sum_sub_distrib,
      Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte]
    ring
  have heq : dotProduct v (α.mulVec v) = 2 * (f 0 - f (d i j)) := by
    rw [hquad, hα i i, hα i j, hα j i, hα j j, hd_diag i, hd_diag j, hd_sym j i]
    ring
  have hlt : f 0 < f (d i j) := hf hij
  exact ⟨heq, by linarith⟩

/-! ### 5. Raw-Stress Global Optimum `u = v = 2/3` -/

/-- Normalized 1D raw stress for three equidistant points with consecutive gaps `u, v`. -/
noncomputable def rawStress (u v : ℝ) : ℝ :=
  ((u - 1) ^ 2 + (v - 1) ^ 2 + (u + v - 1) ^ 2) / 3

/-- Exact sum-of-squares decomposition centered at `(u, v) = (2/3, 2/3)`. -/
theorem rawStress_sum_of_squares (u v : ℝ) :
    rawStress u v = 1 / 9 + ((u - 2 / 3) ^ 2 + (v - 2 / 3) ^ 2 + (u + v - 4 / 3) ^ 2) / 3 := by
  unfold rawStress
  ring

/-- **Unique global minimum of raw stress**: minimum value is `1/9`, attained uniquely at
`u = 2/3` and `v = 2/3`. -/
theorem rawStress_unique_global_min :
    rawStress (2 / 3) (2 / 3) = 1 / 9 ∧
    (∀ u v : ℝ, 1 / 9 ≤ rawStress u v) ∧
    (∀ u v : ℝ, rawStress u v = 1 / 9 ↔ u = 2 / 3 ∧ v = 2 / 3) := by
  refine ⟨by unfold rawStress; norm_num, ?_, ?_⟩
  · intro u v
    rw [rawStress_sum_of_squares]
    have h1 := sq_nonneg (u - 2 / 3)
    have h2 := sq_nonneg (v - 2 / 3)
    have h3 := sq_nonneg (u + v - 4 / 3)
    linarith
  · intro u v
    rw [rawStress_sum_of_squares]
    constructor
    · intro h
      have h1 := sq_nonneg (u - 2 / 3)
      have h2 := sq_nonneg (v - 2 / 3)
      have h3 := sq_nonneg (u + v - 4 / 3)
      have hsum : (u - 2 / 3) ^ 2 + (v - 2 / 3) ^ 2 + (u + v - 4 / 3) ^ 2 = 0 := by linarith
      have hu : (u - 2 / 3) ^ 2 = 0 := by linarith
      have hv : (v - 2 / 3) ^ 2 = 0 := by linarith
      exact ⟨by nlinarith, by nlinarith⟩
    · rintro ⟨rfl, rfl⟩
      norm_num

/-- Order-reversing global raw-stress minimizer: placing `p₁ = 0, p₃ = 2/3, p₂ = 4/3` attains the
global minimum `1/9` while `|p₁ - p₂| = 4/3 > 2/3 = |p₁ - p₃|`. -/
theorem rawStress_order_reversal :
    let p₁ : ℝ := 0
    let p₃ : ℝ := 2 / 3
    let p₂ : ℝ := 4 / 3
    rawStress (p₃ - p₁) (p₂ - p₃) = 1 / 9 ∧ |p₁ - p₃| < |p₁ - p₂| := by
  unfold rawStress
  norm_num

end PEGeometryMonotone
