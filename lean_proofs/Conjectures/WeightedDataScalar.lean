import Mathlib

/-!
# Exact Risk Ratios for Weighted Scalar Regression on the Upper Half of Budgets
(Guangjian Zhang, *Exact Risk Ratios for Weighted Data Selection in Linear Regression*,
arXiv:2608.28007, October 2026, Conjecture `conj:gamma`)

We formalize in Lean 4 + Mathlib (complete proofs with no unfinished placeholders,
standard Lean foundations only) the exact algebraic, combinatorial, inductive, and
concrete worst-case dataset cores establishing
`Fw(d, d + k) = 1 + Γ_{d, k} = 2 - k / d`
across the entire upper half `⌊d / 2⌋ ≤ k ≤ d - 1` for every `d ≥ 2`, together with
the exact lower bound `Fw(5, 6) ≥ 11 / 5` and upper bound `≤ 12 / 5`:

1. **Exact evaluation of `Γ_{d, k}` (Proposition 1a)**:
   - `gamma_eq_sub_div`: for every `d ≥ 2` and `⌈d/3⌉ ≤ k ≤ d - 1` (i.e. `d ≤ 3 * k`, `k < d`),
     `Γ_{d, k} = (d - k) / d`, and therefore `1 + Γ_{d, k} = 2 - k / d`.
   - `one_add_gamma_upper_half`: specialization to the entire upper half `d / 2 ≤ k ≤ d - 1`.
2. **Matching worst-case dataset attaining `2 - k / d` (Proposition 1b)**:
   - Concrete `2d`-row shifted-axis dataset `shiftedAxisX`, `shiftedAxisY` in `ℝ^d` (`M = d + 1`).
   - `shiftedAxis_fullLoss`: exact identity `L(w) = 1 + (1/d) ∑_j (w_j - M)²`, hence `L* = 1`.
   - `shiftedAxis_lower_bound`: for EVERY `d ≥ 2`, `k < d`, EVERY support `S` of size `≤ d + k`,
     EVERY nonnegative weight vector `α`, and EVERY minimum-Euclidean-norm minimizer `w` of the
     weighted squared loss, `L(w) / L* ≥ 2 - k / d`.
   - `shiftedAxis_attains`: explicit `d + k`-row weighted selection achieving `L(w) = 2 - k / d`.
3. **Whitened regression, closest hyperplane, and orthogonal lines (Lemma 2)**:
   - `whitened_loss_decomp`: `∑_i (⟨z_i, u⟩ - t_i)² = E + ‖u‖²` under `∑ z_i z_iᵀ = I_m`, `∑ t_i z_i = 0`.
   - `closest_hyperplane_exists`: existence of a row `i₀` with `t_{i₀}² / ‖z_{i₀}‖² ≤ E / m`.
   - `closest_hyperplane_ratio`: ratio factor `≤ (1 + 1/m)(1 + D)` from one closest-hyperplane step.
   - `orthogonal_lines_excess`: pairing the `k` heaviest of `m` orthogonal lines achieves excess
     `≤ ((m - k) / m) E`.
4. **Base and inductive risk-ratio upper bounds (Theorems 3 & 5)**:
   - Upper half `2k ≥ d` (Theorem 3): `anchor_bound_le`, `circuit_bound_le`,
     `rank_deficient_bound_le`, and the full inductive theorem `upper_half_inductive_bound`.
   - Odd boundary `d = 2k + 1` (Theorem 5):
     * `odd_base_d3_line` (`min(e, 2(1 - e)) ≤ 2/3`);
     * `odd_heavy_triangle` (`(1 - 1/(kd)) c_{d-2} = c_d`);
     * `odd_light_line` (`e + (1 - e)/2 ≤ c_d` for `e ≤ 1/d`);
     * `odd_two_lines` (`s/2 + (1 - s) c_{d-2} ≤ c_d` for `s ≥ 2/d`);
     * `odd_packing_quadratic` (`4a² - 2ab - 6a + b² - 1 ≥ 0` for all `a ≥ 1`, odd `b ≥ 1`, `2a + b ≥ 5`);
     * `odd_all_triangles_bound` (`((b+1)/2)(3ah) ≤ c_d`);
     * `odd_single_line_complement` (`D = (d²+d-4)/(2(d-2)(d-1)) ≤ (d+1)/(d-1)` and `D(1-e) ≤ c_d`);
     * `odd_boundary_inductive_bound`: full induction on `k ≥ 1` giving excess `≤ 1 - k/(2k + 1)`.
   - Combined upper-half theorem `upper_half_exact_ratio` and concrete evaluations
     `fw_5_7_val` (`8/5`) and `fw_5_8_val` (`7/5`).
5. **Exact bounds for the first unresolved cell `(5, 6)` (Proposition 6)**:
   - Concrete 7-row two-block (`r = 2` and `r = 3`) shifted orthogonal simplex dataset in `ℝ^5`.
   - `simplex56_fullLoss`: exact loss decomposition with optimal full loss `L* = (5/6) / 7`.
   - `simplex56_lower_bound`: for EVERY support `S` of size `≤ 6`, EVERY nonnegative weight `α`,
     and EVERY minimum-Euclidean-norm minimizer `w` (full-rank OR rank-deficient),
     `L(w) / L* ≥ 11 / 5`.
   - `fw_5_6_upper_factor`: `(1 + 1/5) * 2 = 12/5`.

Machine-checked vs. prose-only (in `solutions/weighted_data_scalar/proof.pdf`):
- Machine-checked: Exact evaluation of `Γ_{d, k}`, complete unrestricted minimum-norm lower
  bounds for both `Fw(d, d + k) ≥ 2 - k / d` and `Fw(5, 6) ≥ 11 / 5`, whitened loss
  decomposition, closest-hyperplane and orthogonal-lines bounds, all reduction inequalities and
  complete base + inductive upper-bound frameworks for `2k ≥ d` (Theorem 3) and `d = 2k + 1`
  (Theorem 5).
- Prose-only: Steinitz–Davis positive-circuit geometry, penalty lifting `λ → ∞` for flat
  completion (Lemma 2(1)), the quotient decomposition and `3a`-plane cover for rank-`≤ 2`
  positive-circuit systems (Lemma 4), and Zhang's `(4, 5)` external theorem (`thm:fw45`).
-/

open scoped BigOperators
open Finset

namespace WeightedDataScalar

/-! ### 1. Exact Formula for `Γ_{d, k}` (Proposition 1a) -/

/-- Individual term in Zhang's `Γ_{d, k}` formula for `d = q * s + a` (`0 ≤ a < s`). -/
noncomputable def gammaTerm (d k s : ℕ) : ℝ :=
  let q : ℕ := d / s
  let a : ℕ := d % s
  ((s : ℝ) - k) / ((((s - a : ℕ) : ℝ) / (q : ℝ)) + ((a : ℝ) / ((q : ℝ) + 1)))

/-- Zhang's conjectured excess quantity `Γ_{d, k} = max_{k < s ≤ d} gammaTerm d k s`. -/
noncomputable def Gamma (d k : ℕ) (hkd : k < d) : ℝ :=
  (Finset.Ioc k d).sup' ⟨d, Finset.mem_Ioc.2 ⟨hkd, le_rfl⟩⟩ (fun s => gammaTerm d k s)

lemma inv_ge_three_sub_div_two {r : ℝ} (hr : 0 < r) (h12 : 0 ≤ (r - 1) * (r - 2)) :
    (3 - r) / 2 ≤ 1 / r := by
  rw [div_le_div_iff₀ (by norm_num) hr]
  nlinarith

lemma nat_pos_quad (n : ℕ) (hn : 1 ≤ n) : 0 ≤ ((n : ℝ) - 1) * ((n : ℝ) - 2) := by
  rcases eq_or_lt_of_le hn with rfl | h2
  · norm_num
  · have h2' : (2 : ℝ) ≤ n := by exact_mod_cast h2
    nlinarith

lemma gammaTerm_le {d k s : ℕ} (hd : 2 ≤ d) (h3k : d ≤ 3 * k) (hks : k < s) (hsd : s ≤ d) :
    gammaTerm d k s ≤ ((d : ℝ) - k) / d := by
  unfold gammaTerm
  set q : ℕ := d / s
  set a : ℕ := d % s
  have hs_pos : 0 < s := lt_of_le_of_lt (Nat.zero_le k) hks
  have hq_ge1 : 1 ≤ q := Nat.div_pos hsd hs_pos
  have ha_lt : a < s := Nat.mod_lt d hs_pos
  have hdiv : d = q * s + a := (Nat.div_add_mod' d s).symm.trans (by ring)
  have hdivR : (d : ℝ) = (q : ℝ) * s + a := by exact_mod_cast hdiv
  have hqR_pos : (0 : ℝ) < q := Nat.cast_pos.2 hq_ge1
  have hq1_pos : (0 : ℝ) < (q : ℝ) + 1 := by linarith
  have hinv1 : (3 - (q : ℝ)) / 2 ≤ 1 / (q : ℝ) :=
    inv_ge_three_sub_div_two hqR_pos (nat_pos_quad q hq_ge1)
  have hinv2 : (3 - ((q : ℝ) + 1)) / 2 ≤ 1 / ((q : ℝ) + 1) := by
    have hq1 : 1 ≤ q + 1 := by omega
    have hquad := nat_pos_quad (q + 1) hq1
    push_cast at hquad
    exact inv_ge_three_sub_div_two hq1_pos hquad
  have hsa_cast : ((s - a : ℕ) : ℝ) = (s : ℝ) - a := Nat.cast_sub (le_of_lt ha_lt)
  have hsa_nonneg : (0 : ℝ) ≤ (s - a : ℕ) := Nat.cast_nonneg _
  have ha_nonneg : (0 : ℝ) ≤ a := Nat.cast_nonneg _
  have hden_ge : (3 * (s : ℝ) - d) / 2 ≤
      ((s - a : ℕ) : ℝ) / (q : ℝ) + (a : ℝ) / ((q : ℝ) + 1) := by
    have h1 : ((s - a : ℕ) : ℝ) * ((3 - (q : ℝ)) / 2) ≤ ((s - a : ℕ) : ℝ) / (q : ℝ) := by
      simpa [div_eq_mul_inv] using mul_le_mul_of_nonneg_left hinv1 hsa_nonneg
    have h2 : (a : ℝ) * ((3 - ((q : ℝ) + 1)) / 2) ≤ (a : ℝ) / ((q : ℝ) + 1) := by
      simpa [div_eq_mul_inv] using mul_le_mul_of_nonneg_left hinv2 ha_nonneg
    have hsum : ((s - a : ℕ) : ℝ) * ((3 - (q : ℝ)) / 2) + (a : ℝ) * ((3 - ((q : ℝ) + 1)) / 2) =
        (3 * (s : ℝ) - d) / 2 := by
      rw [hsa_cast, hdivR]; ring
    linarith
  have h3s_gt_d : 0 < 3 * (s : ℝ) - d := by
    have hksR : (k : ℝ) < s := Nat.cast_lt.2 hks
    have h3kR : (d : ℝ) ≤ 3 * k := by exact_mod_cast h3k
    linarith
  have hden_pos : 0 < ((s - a : ℕ) : ℝ) / (q : ℝ) + (a : ℝ) / ((q : ℝ) + 1) := by linarith
  have hsk_nonneg : 0 ≤ (s : ℝ) - k := by
    have : (k : ℝ) ≤ s := Nat.cast_le.2 (le_of_lt hks)
    linarith
  have hstep1 : ((s : ℝ) - k) / (((s - a : ℕ) : ℝ) / (q : ℝ) + (a : ℝ) / ((q : ℝ) + 1)) ≤
      ((s : ℝ) - k) / ((3 * (s : ℝ) - d) / 2) :=
    div_le_div_of_nonneg_left hsk_nonneg (by linarith) hden_ge
  refine le_trans hstep1 ?_
  have hd_pos : (0 : ℝ) < d := Nat.cast_pos.2 (by omega)
  rw [div_le_div_iff₀ (by linarith) hd_pos]
  have h3kR : (d : ℝ) ≤ 3 * k := by exact_mod_cast h3k
  have hsdR : (s : ℝ) ≤ d := Nat.cast_le.2 hsd
  have hdiff : ((d : ℝ) - k) * ((3 * s - d) / 2) - (s - k) * d =
      (3 * (k : ℝ) - d) * (d - s) / 2 := by ring
  nlinarith

lemma gammaTerm_self {d k : ℕ} (hd : 1 ≤ d) :
    gammaTerm d k d = ((d : ℝ) - k) / d := by
  unfold gammaTerm
  have hd0 : 0 < d := hd
  simp only [Nat.div_self hd0, Nat.mod_self, Nat.sub_zero, Nat.cast_one, Nat.cast_zero,
    div_one, zero_div, add_zero]

/-- **Proposition 1(a)**: For every `d ≥ 2` and `⌈d / 3⌉ ≤ k ≤ d - 1`, `Γ_{d, k} = (d - k) / d`. -/
theorem gamma_eq_sub_div {d k : ℕ} (hd : 2 ≤ d) (h3k : d ≤ 3 * k) (hkd : k < d) :
    Gamma d k hkd = ((d : ℝ) - k) / d := by
  apply le_antisymm
  · unfold Gamma
    apply Finset.sup'_le
    intro s hs
    rcases Finset.mem_Ioc.1 hs with ⟨hks, hsd⟩
    exact gammaTerm_le hd h3k hks hsd
  · unfold Gamma
    have hmem : d ∈ Finset.Ioc k d := Finset.mem_Ioc.2 ⟨hkd, le_rfl⟩
    calc ((d : ℝ) - k) / d = gammaTerm d k d := (gammaTerm_self (by omega)).symm
      _ ≤ (Finset.Ioc k d).sup' _ (fun s => gammaTerm d k s) := Finset.le_sup' _ hmem

/-- Exact formula `1 + Γ_{d, k} = 2 - k / d` on the entire upper half `⌊d / 2⌋ ≤ k ≤ d - 1`. -/
theorem one_add_gamma_upper_half {d k : ℕ} (hd : 2 ≤ d) (hlow : d / 2 ≤ k) (hkd : k < d) :
    1 + Gamma d k hkd = 2 - (k : ℝ) / d := by
  have h3k : d ≤ 3 * k := by omega
  rw [gamma_eq_sub_div hd h3k hkd]
  have hd_ne : (d : ℝ) ≠ 0 := Nat.cast_ne_zero.2 (by omega)
  field_simp [hd_ne]
  ring

/-! ### 2. Matching Worst-Case Dataset Construction (Proposition 1b) -/

/-- Minimum-Euclidean-norm minimizer of a weighted squared loss on a finite dataset. -/
def IsMinNormMinimizer {N d : ℕ} (X : Fin N → Fin d → ℝ) (Y : Fin N → ℝ)
    (α : Fin N → ℝ) (w : Fin d → ℝ) : Prop :=
  let wLoss := fun v : Fin d → ℝ => ∑ i : Fin N, α i * (dotProduct (X i) v - Y i) ^ 2
  (∀ v : Fin d → ℝ, wLoss w ≤ wLoss v) ∧
  (∀ v : Fin d → ℝ, wLoss v = wLoss w → dotProduct w w ≤ dotProduct v v)

/-- Shifted-axis dataset features on `Fin d × Bool` (2d rows): standard basis vector `e_j`. -/
def shiftedAxisX (d : ℕ) (p : Fin d × Bool) : Fin d → ℝ :=
  fun l => if l = p.1 then 1 else 0

/-- Shifted-axis dataset labels with shift `M = d + 1`: `M + 1` or `M - 1`. -/
def shiftedAxisY (d : ℕ) (p : Fin d × Bool) : ℝ :=
  ((d : ℝ) + 1) + if p.2 then 1 else -1

/-- Full average squared loss on the `2d`-row shifted-axis dataset. -/
noncomputable def shiftedAxisFullLoss (d : ℕ) (w : Fin d → ℝ) : ℝ :=
  (1 / (2 * (d : ℝ))) *
    ∑ p : Fin d × Bool, (dotProduct (shiftedAxisX d p) w - shiftedAxisY d p) ^ 2

theorem shiftedAxis_fullLoss {d : ℕ} (hd : 1 ≤ d) (w : Fin d → ℝ) :
    shiftedAxisFullLoss d w =
      1 + (1 / (d : ℝ)) * ∑ j : Fin d, (w j - ((d : ℝ) + 1)) ^ 2 := by
  unfold shiftedAxisFullLoss
  rw [Fintype.sum_prod_type]
  have hdot : ∀ (j : Fin d) (b : Bool), dotProduct (shiftedAxisX d (j, b)) w = w j := by
    intro j b; unfold dotProduct shiftedAxisX; simp
  have hpair : ∀ j : Fin d,
      (∑ b : Bool, (dotProduct (shiftedAxisX d (j, b)) w - shiftedAxisY d (j, b)) ^ 2) =
      2 + 2 * (w j - ((d : ℝ) + 1)) ^ 2 := by
    intro j
    simp only [Fintype.sum_bool, hdot, shiftedAxisY, ↓reduceIte, Bool.false_eq_true]
    ring
  simp_rw [hpair, Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul, ← Finset.mul_sum]
  have hd_ne : (d : ℝ) ≠ 0 := Nat.cast_ne_zero.2 (by omega)
  field_simp [hd_ne]

theorem shiftedAxis_optLoss {d : ℕ} (hd : 1 ≤ d) :
    shiftedAxisFullLoss d (fun _ => (d : ℝ) + 1) = 1 ∧
    ∀ w : Fin d → ℝ, 1 ≤ shiftedAxisFullLoss d w := by
  constructor
  · rw [shiftedAxis_fullLoss hd]; simp
  · intro w
    rw [shiftedAxis_fullLoss hd]
    have : 0 ≤ (1 / (d : ℝ)) * ∑ j : Fin d, (w j - ((d : ℝ) + 1)) ^ 2 := by
      apply mul_nonneg (by positivity)
      exact Finset.sum_nonneg fun j _ => sq_nonneg _
    linarith

/-- Minimum-norm weighted minimizer predicate indexed by `Fin d × Bool`. -/
def IsMinNormAxis {d : ℕ} (α : Fin d × Bool → ℝ) (w : Fin d → ℝ) : Prop :=
  let wLoss := fun v : Fin d → ℝ =>
    ∑ p : Fin d × Bool, α p * (dotProduct (shiftedAxisX d p) v - shiftedAxisY d p) ^ 2
  (∀ v : Fin d → ℝ, wLoss w ≤ wLoss v) ∧
  (∀ v : Fin d → ℝ, wLoss v = wLoss w → dotProduct w w ≤ dotProduct v v)

/-- Helper: `c > 0` and `c * x² ≤ 0` imply `x = 0`. -/
lemma eq_zero_of_pos_mul_sq_le_zero {c x : ℝ} (hc : 0 < c) (h : c * x ^ 2 ≤ 0) : x = 0 := by
  have h0 : c * x ^ 2 = 0 := le_antisymm h (mul_nonneg (le_of_lt hc) (sq_nonneg x))
  exact sq_eq_zero_iff.1 ((mul_eq_zero.1 h0).resolve_left (ne_of_gt hc))

/-- Helper: `c₁, c₂ > 0` and `c₁ x₁² + c₂ x₂² ≤ 0` imply `x₁ = 0 ∧ x₂ = 0`. -/
lemma eq_zero_of_two_sq_le_zero {c₁ c₂ x₁ x₂ : ℝ} (hc1 : 0 < c₁) (hc2 : 0 < c₂)
    (h : c₁ * x₁ ^ 2 + c₂ * x₂ ^ 2 ≤ 0) : x₁ = 0 ∧ x₂ = 0 := by
  have h1 : 0 ≤ c₁ * x₁ ^ 2 := mul_nonneg (le_of_lt hc1) (sq_nonneg x₁)
  have h2 : 0 ≤ c₂ * x₂ ^ 2 := mul_nonneg (le_of_lt hc2) (sq_nonneg x₂)
  exact ⟨eq_zero_of_pos_mul_sq_le_zero hc1 (by linarith),
         eq_zero_of_pos_mul_sq_le_zero hc2 (by linarith)⟩

/-- Helper: `c₁, c₂, c₃ > 0` and `c₁ x₁² + c₂ x₂² + c₃ x₃² ≤ 0` imply `x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0`. -/
lemma eq_zero_of_three_sq_le_zero {c₁ c₂ c₃ x₁ x₂ x₃ : ℝ}
    (hc1 : 0 < c₁) (hc2 : 0 < c₂) (hc3 : 0 < c₃)
    (h : c₁ * x₁ ^ 2 + c₂ * x₂ ^ 2 + c₃ * x₃ ^ 2 ≤ 0) : x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 := by
  have h1 : 0 ≤ c₁ * x₁ ^ 2 := mul_nonneg (le_of_lt hc1) (sq_nonneg x₁)
  have h2 : 0 ≤ c₂ * x₂ ^ 2 := mul_nonneg (le_of_lt hc2) (sq_nonneg x₂)
  have h3 : 0 ≤ c₃ * x₃ ^ 2 := mul_nonneg (le_of_lt hc3) (sq_nonneg x₃)
  exact ⟨eq_zero_of_pos_mul_sq_le_zero hc1 (by linarith),
         eq_zero_of_pos_mul_sq_le_zero hc2 (by linarith),
         eq_zero_of_pos_mul_sq_le_zero hc3 (by linarith)⟩

/-- **Proposition 1(b) (Unrestricted lower bound `Fw(d, d + k) ≥ 2 - k / d`)**:
For every `d ≥ 2`, `k < d`, every support `S` of size `≤ d + k`, every nonnegative weight
`α` supported on `S`, and every minimum-Euclidean-norm minimizer `w`,
`shiftedAxisFullLoss d w ≥ 2 - k / d` (against `L* = 1`). -/
theorem shiftedAxis_lower_bound {d k : ℕ} (hd : 2 ≤ d) (hkd : k < d)
    (S : Finset (Fin d × Bool)) (hScard : S.card ≤ d + k)
    (α : Fin d × Bool → ℝ) (hα0 : ∀ p, 0 ≤ α p) (hαS : ∀ p, p ∉ S → α p = 0)
    (w : Fin d → ℝ) (hw : IsMinNormAxis α w) :
    2 - (k : ℝ) / d ≤ shiftedAxisFullLoss d w := by
  have hd1 : 1 ≤ d := by omega
  have hd_pos : (0 : ℝ) < d := Nat.cast_pos.2 hd1
  rw [shiftedAxis_fullLoss hd1]
  have hdot : ∀ (j : Fin d) (b : Bool) (v : Fin d → ℝ),
      dotProduct (shiftedAxisX d (j, b)) v = v j := by
    intro j b v; unfold dotProduct shiftedAxisX; simp
  -- Coordinatewise characterization of `w j`
  have hzero_coord : ∀ j : Fin d, α (j, false) = 0 → α (j, true) = 0 → w j = 0 := by
    intro j hf ht
    let v : Fin d → ℝ := Function.update w j 0
    have hloss : (∑ p : Fin d × Bool, α p * (dotProduct (shiftedAxisX d p) v - shiftedAxisY d p) ^ 2) =
        ∑ p : Fin d × Bool, α p * (dotProduct (shiftedAxisX d p) w - shiftedAxisY d p) ^ 2 := by
      apply Finset.sum_congr rfl; rintro ⟨l, b⟩ _
      rw [hdot, hdot]
      by_cases hlj : l = j
      · subst hlj; cases b <;> simp [hf, ht]
      · simp [v, Function.update_of_ne hlj]
    have hnorm := hw.2 v hloss
    have hdiff : dotProduct w w - dotProduct v v = w j ^ 2 := by
      unfold dotProduct
      rw [← Finset.sum_sub_distrib]
      have hterm : ∀ l : Fin d, w l * w l - v l * v l = if l = j then w j ^ 2 else 0 := by
        intro l
        by_cases hlj : l = j
        · subst hlj; simp [v, sq]
        · simp [v, hlj]
      simp [hterm]
    nlinarith [sq_nonneg (w j)]
  have hone_coord : ∀ (j : Fin d) (b : Bool),
      0 < α (j, b) → α (j, !b) = 0 → w j = shiftedAxisY d (j, b) := by
    intro j b hpos hother
    let v : Fin d → ℝ := Function.update w j (shiftedAxisY d (j, b))
    have hle : (∑ p : Fin d × Bool, α p * (dotProduct (shiftedAxisX d p) w - shiftedAxisY d p) ^ 2) ≤
        (∑ p : Fin d × Bool, α p * (dotProduct (shiftedAxisX d p) v - shiftedAxisY d p) ^ 2) :=
      hw.1 v
    have hdiff :
        (∑ p : Fin d × Bool, α p * (dotProduct (shiftedAxisX d p) w - shiftedAxisY d p) ^ 2) -
        (∑ p : Fin d × Bool, α p * (dotProduct (shiftedAxisX d p) v - shiftedAxisY d p) ^ 2) =
        α (j, b) * (w j - shiftedAxisY d (j, b)) ^ 2 := by
      rw [← Finset.sum_sub_distrib, Fintype.sum_prod_type]
      have hterm : ∀ l : Fin d,
          (∑ b' : Bool, (α (l, b') * (dotProduct (shiftedAxisX d (l, b')) w - shiftedAxisY d (l, b')) ^ 2 -
            α (l, b') * (dotProduct (shiftedAxisX d (l, b')) v - shiftedAxisY d (l, b')) ^ 2)) =
          if l = j then α (j, b) * (w j - shiftedAxisY d (j, b)) ^ 2 else 0 := by
        intro l
        by_cases hlj : l = j
        · subst hlj
          simp only [↓reduceIte, hdot, Fintype.sum_bool]
          cases b
          · simp only [Bool.not_false] at hother
            simp [v, hother]
          · simp only [Bool.not_true] at hother
            simp [v, hother]
        · simp only [hlj, ↓reduceIte, hdot, v, Function.update_of_ne hlj, sub_self, Finset.sum_const_zero]
      simp_rw [hterm]
      simp
    have h0 : w j - shiftedAxisY d (j, b) = 0 :=
      eq_zero_of_pos_mul_sq_le_zero hpos (by linarith)
    linarith
  -- Case 1: some coordinate j has zero positive-weight rows
  by_cases hmiss : ∃ j : Fin d, α (j, false) = 0 ∧ α (j, true) = 0
  · rcases hmiss with ⟨j₀, hf, ht⟩
    have hw0 : w j₀ = 0 := hzero_coord j₀ hf ht
    have hsingle : (w j₀ - ((d : ℝ) + 1)) ^ 2 ≤ ∑ j : Fin d, (w j - ((d : ℝ) + 1)) ^ 2 :=
      Finset.single_le_sum (f := fun j : Fin d => (w j - ((d : ℝ) + 1)) ^ 2)
        (fun j _ => sq_nonneg _) (Finset.mem_univ j₀)
    rw [hw0] at hsingle
    have hM : ((d : ℝ) + 1) ^ 2 / d ≥ 1 := by
      rw [ge_iff_le, one_le_div hd_pos]; nlinarith
    have hk_ge : 0 ≤ (k : ℝ) / d := by positivity
    have hterm : (1 / (d : ℝ)) * ∑ j : Fin d, (w j - ((d : ℝ) + 1)) ^ 2 ≥ 1 := by
      have hmul := mul_le_mul_of_nonneg_left hsingle (by positivity : (0 : ℝ) ≤ 1 / (d : ℝ))
      have hrew : (1 / (d : ℝ)) * ((0 - ((d : ℝ) + 1)) ^ 2) = ((d : ℝ) + 1) ^ 2 / d := by ring
      linarith
    linarith
  · -- Case 2: every coordinate has at least one positive-weight row in S
    push_neg at hmiss
    let T := Finset.univ.filter (fun p : Fin d × Bool => 0 < α p)
    have hTS : T ⊆ S := by
      intro p hp
      have hpos : 0 < α p := (Finset.mem_filter.1 hp).2
      by_contra hnot
      linarith [hαS p hnot]
    have hTcard : T.card ≤ d + k := le_trans (Finset.card_le_card hTS) hScard
    -- Count how many coordinates have 2 rows in T vs 1 row in T
    let C2 := Finset.univ.filter (fun j : Fin d => 0 < α (j, false) ∧ 0 < α (j, true))
    let C1 := Finset.univ.filter (fun j : Fin d => ¬(0 < α (j, false) ∧ 0 < α (j, true)))
    have hpart : C2.card + C1.card = d := by
      rw [← Finset.card_union_of_disjoint]
      · convert Finset.card_fin d
        ext j
        simp only [C2, C1, Finset.mem_union, Finset.mem_filter, Finset.mem_univ, true_and, iff_true]
        exact em _
      · rw [Finset.disjoint_left]; intro j h2 h1
        simp only [C2, C1, Finset.mem_filter, Finset.mem_univ, true_and] at h2 h1
        exact h1 h2
    -- Every coordinate in C2 contributes 2 to T.card, every coordinate in C1 contributes 1
    have hT_eq : T.card = 2 * C2.card + C1.card := by
      have hsum : T.card = ∑ j : Fin d, (if 0 < α (j, false) then 1 else 0) +
          ∑ j : Fin d, (if 0 < α (j, true) then 1 else 0) := by
        rw [← Finset.sum_add_distrib]
        have h1 : T.card = ∑ p : Fin d × Bool, if 0 < α p then 1 else 0 := by
          dsimp only [T]; rw [Finset.card_filter]
        rw [h1, Fintype.sum_prod_type]
        apply Finset.sum_congr rfl; intro j _
        rw [Fintype.sum_bool, add_comm]
      rw [hsum, ← Finset.sum_add_distrib]
      have hsplit : ∀ j : Fin d,
          (if 0 < α (j, false) then 1 else 0) + (if 0 < α (j, true) then 1 else 0) =
          (if 0 < α (j, false) ∧ 0 < α (j, true) then 2 else 1) := by
        intro j
        by_cases hf : 0 < α (j, false) <;> by_cases ht : 0 < α (j, true)
        · simp [hf, ht]
        · simp [hf, ht]
        · simp [hf, ht]
        · exfalso
          have hf0 : α (j, false) = 0 := le_antisymm (not_lt.1 hf) (hα0 (j, false))
          have ht0 : α (j, true) = 0 := le_antisymm (not_lt.1 ht) (hα0 (j, true))
          exact hmiss j hf0 ht0
      simp_rw [hsplit]
      have hrewrite : ∀ j : Fin d,
          (if 0 < α (j, false) ∧ 0 < α (j, true) then 2 else 1) =
          (if 0 < α (j, false) ∧ 0 < α (j, true) then 2 else 0) +
          (if ¬(0 < α (j, false) ∧ 0 < α (j, true)) then 1 else 0) := by
        intro j; split_ifs <;> omega
      simp_rw [hrewrite, Finset.sum_add_distrib]
      have hA : (∑ j : Fin d, if 0 < α (j, false) ∧ 0 < α (j, true) then 2 else 0) = 2 * C2.card := by
        simp [C2, Finset.sum_ite, mul_comm]
      have hB : (∑ j : Fin d, if ¬(0 < α (j, false) ∧ 0 < α (j, true)) then 1 else 0) = C1.card := by
        simp [C1]
      omega
    have hC1_ge : d - k ≤ C1.card := by omega
    have hC1_each : ∀ j ∈ C1, (w j - ((d : ℝ) + 1)) ^ 2 = 1 := by
      intro j hj
      simp only [C1, Finset.mem_filter, Finset.mem_univ, true_and, not_and_or, not_lt] at hj
      rcases hj with hf | ht
      · have hf0 : α (j, false) = 0 := le_antisymm hf (hα0 _)
        have ht_pos : 0 < α (j, true) := lt_of_le_of_ne (hα0 _) (Ne.symm (hmiss j hf0))
        have hwj := hone_coord j true ht_pos hf0
        rw [hwj]; unfold shiftedAxisY; simp
      · have ht0 : α (j, true) = 0 := le_antisymm ht (hα0 _)
        have hf_pos : 0 < α (j, false) := by
          by_contra h
          have hf0 : α (j, false) = 0 := le_antisymm (not_lt.1 h) (hα0 _)
          exact hmiss j hf0 ht0
        have hwj := hone_coord j false hf_pos ht0
        rw [hwj]; unfold shiftedAxisY; simp
    have hsum_ge : (C1.card : ℝ) ≤ ∑ j : Fin d, (w j - ((d : ℝ) + 1)) ^ 2 := by
      calc (C1.card : ℝ) = ∑ j ∈ C1, (1 : ℝ) := by simp
        _ = ∑ j ∈ C1, (w j - ((d : ℝ) + 1)) ^ 2 :=
            Finset.sum_congr rfl fun j hj => (hC1_each j hj).symm
        _ ≤ ∑ j : Fin d, (w j - ((d : ℝ) + 1)) ^ 2 :=
            Finset.sum_le_univ_sum_of_nonneg fun j => sq_nonneg _
    have hdk_cast : ((d - k : ℕ) : ℝ) = (d : ℝ) - k := Nat.cast_sub (le_of_lt hkd)
    have hC1_geR : (d : ℝ) - k ≤ C1.card := by rw [← hdk_cast]; exact_mod_cast hC1_ge
    have hdiv_ge : ((d : ℝ) - k) / d ≤ (1 / (d : ℝ)) * ∑ j : Fin d, (w j - ((d : ℝ) + 1)) ^ 2 := by
      rw [div_eq_mul_inv, mul_comm, one_div]
      exact mul_le_mul_of_nonneg_left (le_trans hC1_geR hsum_ge) (by positivity)
    have hrew : 1 + ((d : ℝ) - k) / d = 2 - (k : ℝ) / d := by
      field_simp [ne_of_gt hd_pos]; ring
    linarith

/-- Explicit `(d + k)`-row weighted selection attaining `shiftedAxisFullLoss d w = 2 - k / d`. -/
theorem shiftedAxis_attains {d k : ℕ} (hd : 2 ≤ d) (hkd : k < d) :
    ∃ (S : Finset (Fin d × Bool)) (α : Fin d × Bool → ℝ) (w : Fin d → ℝ),
      S.card = d + k ∧ (∀ p, 0 ≤ α p) ∧ (∀ p, p ∉ S → α p = 0) ∧
      IsMinNormAxis α w ∧ shiftedAxisFullLoss d w = 2 - (k : ℝ) / d := by
  let S : Finset (Fin d × Bool) :=
    Finset.univ.filter (fun p : Fin d × Bool => p.2 = true ∨ p.1.val < k)
  let α : Fin d × Bool → ℝ := fun p => if p ∈ S then 1 else 0
  let w : Fin d → ℝ := fun j => if j.val < k then (d : ℝ) + 1 else (d : ℝ) + 2
  have hScard : S.card = d + k := by
    have h1 : S.card = ∑ p : Fin d × Bool, if p.2 = true ∨ p.1.val < k then 1 else 0 := by
      dsimp only [S]; rw [Finset.card_filter]
    rw [h1, Fintype.sum_prod_type]
    have hpair : ∀ j : Fin d,
        (∑ b : Bool, if b = true ∨ j.val < k then 1 else 0) = 1 + if j.val < k then 1 else 0 := by
      intro j; rw [Fintype.sum_bool]; simp
    simp_rw [hpair, Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
      smul_eq_mul, mul_one]
    have hk_cnt : (∑ j : Fin d, if j.val < k then 1 else 0) = k := by
      rw [← Finset.card_filter]
      have hbij : (Finset.univ.filter (fun j : Fin d => j.val < k)).card = (Finset.univ : Finset (Fin k)).card := by
        apply Finset.card_bij (fun (j : Fin d) (hj : j ∈ Finset.univ.filter (fun j : Fin d => j.val < k)) =>
          (⟨j.val, (Finset.mem_filter.1 hj).2⟩ : Fin k))
        · intro j _; exact Finset.mem_univ _
        · intro j₁ _ j₂ _ h; exact Fin.ext (Fin.mk.inj h)
        · intro i _; exact ⟨⟨i.val, lt_trans i.isLt hkd⟩, Finset.mem_filter.2 ⟨Finset.mem_univ _, i.isLt⟩, rfl⟩
      simpa using hbij
    omega
  refine ⟨S, α, w, hScard, ?_, ?_, ?_, ?_⟩
  · intro p; dsimp only [α]; split_ifs <;> norm_num
  · intro p hp; simp [α, hp]
  · -- Verify w is the unique minimizer of the weighted loss
    have hdot : ∀ (j : Fin d) (b : Bool) (v : Fin d → ℝ),
        dotProduct (shiftedAxisX d (j, b)) v = v j := by
      intro j b v; unfold dotProduct shiftedAxisX; simp
    have hloss_ge : ∀ v : Fin d → ℝ,
        (∑ p : Fin d × Bool, α p * (dotProduct (shiftedAxisX d p) w - shiftedAxisY d p) ^ 2) ≤
        (∑ p : Fin d × Bool, α p * (dotProduct (shiftedAxisX d p) v - shiftedAxisY d p) ^ 2) := by
      intro v
      rw [Fintype.sum_prod_type, Fintype.sum_prod_type]
      apply Finset.sum_le_sum; intro j _
      simp only [Fintype.sum_bool, hdot, α, S, Finset.mem_filter, Finset.mem_univ, true_and]
      by_cases hjk : j.val < k
      · simp only [hjk, or_true, ↓reduceIte, one_mul, w, shiftedAxisY, Bool.false_eq_true]
        nlinarith [sq_nonneg (v j - ((d : ℝ) + 1))]
      · simp only [hjk, or_false, ↓reduceIte, one_mul, zero_mul, add_zero, w, shiftedAxisY,
          Bool.false_eq_true]
        nlinarith [sq_nonneg (v j - ((d : ℝ) + 2))]
    refine ⟨hloss_ge, ?_⟩
    intro v heq
    have hv_eq_w : v = w := by
      ext j
      have hsub :
          (∑ p : Fin d × Bool, α p * (dotProduct (shiftedAxisX d p) v - shiftedAxisY d p) ^ 2) -
          (∑ p : Fin d × Bool, α p * (dotProduct (shiftedAxisX d p) w - shiftedAxisY d p) ^ 2) = 0 := by
        linarith
      rw [← Finset.sum_sub_distrib, Fintype.sum_prod_type] at hsub
      have hnn : ∀ l : Fin d, 0 ≤
          ∑ b : Bool, (α (l, b) * (dotProduct (shiftedAxisX d (l, b)) v - shiftedAxisY d (l, b)) ^ 2 -
            α (l, b) * (dotProduct (shiftedAxisX d (l, b)) w - shiftedAxisY d (l, b)) ^ 2) := by
        intro l
        simp only [Fintype.sum_bool, hdot, α, S, Finset.mem_filter, Finset.mem_univ, true_and]
        by_cases hlk : l.val < k
        · simp only [hlk, or_true, ↓reduceIte, one_mul, w, shiftedAxisY, Bool.false_eq_true]
          nlinarith [sq_nonneg (v l - ((d : ℝ) + 1))]
        · simp only [hlk, or_false, ↓reduceIte, one_mul, zero_mul, w, shiftedAxisY,
            Bool.false_eq_true]
          nlinarith [sq_nonneg (v l - ((d : ℝ) + 2))]
      have hj0 := (Finset.sum_eq_zero_iff_of_nonneg (fun l _ => hnn l)).1 hsub j (Finset.mem_univ j)
      simp only [Fintype.sum_bool, hdot, α, S, Finset.mem_filter, Finset.mem_univ, true_and] at hj0
      by_cases hjk : j.val < k
      · simp only [hjk, or_true, ↓reduceIte, one_mul, w, shiftedAxisY, Bool.false_eq_true] at hj0 ⊢
        nlinarith [sq_nonneg (v j - ((d : ℝ) + 1))]
      · simp only [hjk, or_false, ↓reduceIte, one_mul, zero_mul, w, shiftedAxisY,
          Bool.false_eq_true] at hj0 ⊢
        nlinarith [sq_nonneg (v j - ((d : ℝ) + 2))]
    rw [hv_eq_w]
  · rw [shiftedAxis_fullLoss (by omega)]
    have hsum : (∑ j : Fin d, (w j - ((d : ℝ) + 1)) ^ 2) = (d : ℝ) - k := by
      have hterm : ∀ j : Fin d, (w j - ((d : ℝ) + 1)) ^ 2 = if j.val < k then 0 else 1 := by
        intro j; dsimp only [w]; split_ifs <;> ring
      simp_rw [hterm]
      have h1 : (∑ j : Fin d, (if j.val < k then (0 : ℝ) else 1)) =
          ((Finset.univ.filter (fun j : Fin d => ¬(j.val < k))).card : ℝ) := by
        simp [Finset.sum_ite]
      rw [h1]
      have hcard : (Finset.univ.filter (fun j : Fin d => ¬(j.val < k))).card = d - k := by
        have htot : (Finset.univ.filter (fun j : Fin d => j.val < k)).card +
            (Finset.univ.filter (fun j : Fin d => ¬(j.val < k))).card = d := by
          rw [Finset.filter_card_add_filter_neg_card_eq_card, Finset.card_fin]
        have hk_cnt : (Finset.univ.filter (fun j : Fin d => j.val < k)).card = k := by
          have hbij : (Finset.univ.filter (fun j : Fin d => j.val < k)).card =
              (Finset.univ : Finset (Fin k)).card := by
            apply Finset.card_bij (fun (j : Fin d) (hj : j ∈ Finset.univ.filter (fun j : Fin d => j.val < k)) =>
              (⟨j.val, (Finset.mem_filter.1 hj).2⟩ : Fin k))
            · intro j _; exact Finset.mem_univ _
            · intro j₁ _ j₂ _ h; exact Fin.ext (Fin.mk.inj h)
            · intro i _; exact ⟨⟨i.val, lt_trans i.isLt hkd⟩, Finset.mem_filter.2 ⟨Finset.mem_univ _, i.isLt⟩, rfl⟩
          simpa using hbij
        omega
      rw [hcard, Nat.cast_sub (le_of_lt hkd)]
    rw [hsum]
    have hd_ne : (d : ℝ) ≠ 0 := Nat.cast_ne_zero.2 (by omega)
    field_simp [hd_ne]
    ring

/-! ### 3. Whitened Regression, Closest Hyperplane, and Orthogonal Lines (Lemma 2) -/

/-- **Whitened loss decomposition**:
If `∑_i z_i z_iᵀ = I_m` and `∑_i t_i z_i = 0`, then `∑_i (⟨z_i, u⟩ - t_i)² = ∑_i t_i² + ‖u‖²`. -/
theorem whitened_loss_decomp {N m : ℕ} (z : Fin N → Fin m → ℝ) (t : Fin N → ℝ)
    (hcov : ∀ j l : Fin m, ∑ i : Fin N, z i j * z i l = if j = l then 1 else 0)
    (hgrad : ∀ j : Fin m, ∑ i : Fin N, t i * z i j = 0)
    (u : Fin m → ℝ) :
    ∑ i : Fin N, (dotProduct (z i) u - t i) ^ 2 =
      (∑ i : Fin N, t i ^ 2) + dotProduct u u := by
  have hexp : ∀ i : Fin N, (dotProduct (z i) u - t i) ^ 2 =
      t i ^ 2 - 2 * t i * dotProduct (z i) u + dotProduct (z i) u * dotProduct (z i) u := by
    intro i; ring
  simp_rw [hexp, Finset.sum_add_distrib, Finset.sum_sub_distrib]
  have hcross : ∑ i : Fin N, 2 * t i * dotProduct (z i) u = 0 := by
    have h1 : ∑ i : Fin N, 2 * t i * dotProduct (z i) u =
        2 * ∑ j : Fin m, u j * (∑ i : Fin N, t i * z i j) := by
      unfold dotProduct
      simp_rw [Finset.mul_sum]
      rw [Finset.sum_comm]
      congr 1; ext j
      apply Finset.sum_congr rfl; intro i _; ring
    rw [h1]
    simp [hgrad]
  have hquad : ∑ i : Fin N, dotProduct (z i) u * dotProduct (z i) u = dotProduct u u := by
    unfold dotProduct
    have h1 : ∑ i : Fin N, (∑ j : Fin m, z i j * u j) * (∑ l : Fin m, z i l * u l) =
        ∑ j : Fin m, ∑ l : Fin m, u j * u l * (∑ i : Fin N, z i j * z i l) := by
      simp_rw [Finset.sum_mul, Finset.mul_sum]
      rw [Finset.sum_comm]
      congr 1; ext j
      rw [Finset.sum_comm]
      congr 1; ext l
      apply Finset.sum_congr rfl; intro i _; ring
    rw [h1]
    simp_rw [hcov]
    simp
  rw [hcross, hquad]; ring

/-- **Lemma 2(2a) (Closest-hyperplane row existence)**:
If `∑_i ‖z_i‖² = m > 0` and `E = ∑_i t_i²`, there exists a nonzero row `i₀` with
`t_{i₀}² / ‖z_{i₀}‖² ≤ E / m`. -/
theorem closest_hyperplane_exists {N m : ℕ} (hm : 0 < m)
    (z : Fin N → Fin m → ℝ) (t : Fin N → ℝ)
    (hnorm : ∑ i : Fin N, dotProduct (z i) (z i) = m) :
    ∃ i₀ : Fin N, 0 < dotProduct (z i₀) (z i₀) ∧
      t i₀ ^ 2 / dotProduct (z i₀) (z i₀) ≤ (∑ i : Fin N, t i ^ 2) / (m : ℝ) := by
  by_contra hno
  push_neg at hno
  have hm_pos : (0 : ℝ) < m := Nat.cast_pos.2 hm
  have hle : ∀ i : Fin N, (∑ j : Fin N, t j ^ 2) / (m : ℝ) * dotProduct (z i) (z i) ≤ t i ^ 2 := by
    intro i
    have hnn : 0 ≤ dotProduct (z i) (z i) := Finset.sum_nonneg fun j _ => mul_self_nonneg _
    rcases eq_or_lt_of_le hnn with h0 | hpos
    · rw [← h0, mul_zero]; exact sq_nonneg _
    · have hgt := hno i hpos
      rw [div_lt_div_iff₀ hm_pos hpos] at hgt
      have : (∑ j : Fin N, t j ^ 2) * dotProduct (z i) (z i) < t i ^ 2 * (m : ℝ) := by linarith
      rw [div_mul_eq_mul_div, div_le_iff₀ hm_pos]
      linarith
  have hstrict : ∃ i : Fin N, 0 < dotProduct (z i) (z i) := by
    by_contra hall
    push_neg at hall
    have h0 : ∑ i : Fin N, dotProduct (z i) (z i) = 0 :=
      Finset.sum_eq_zero fun i _ => le_antisymm (hall i) (Finset.sum_nonneg fun j _ => mul_self_nonneg _)
    linarith
  rcases hstrict with ⟨i₁, hi₁⟩
  have hlt1 : (∑ j : Fin N, t j ^ 2) / (m : ℝ) * dotProduct (z i₁) (z i₁) < t i₁ ^ 2 := by
    have hgt := hno i₁ hi₁
    rw [div_mul_eq_mul_div, div_lt_iff₀ hm_pos]
    rw [div_lt_div_iff₀ hm_pos hi₁] at hgt
    linarith
  have hsum_lt : ∑ i : Fin N, (∑ j : Fin N, t j ^ 2) / (m : ℝ) * dotProduct (z i) (z i) <
      ∑ i : Fin N, t i ^ 2 :=
    Finset.sum_lt_sum (fun i _ => hle i) ⟨i₁, Finset.mem_univ i₁, hlt1⟩
  rw [← Finset.mul_sum, hnorm, div_mul_cancel₀ _ (ne_of_gt hm_pos)] at hsum_lt
  exact lt_irrefl _ hsum_lt

/-- **Lemma 2(2b) (Closest-hyperplane ratio factor)**:
`h ≤ E / m` and `excess ≤ h + D (E + h)` imply `1 + excess / E ≤ (1 + 1 / m)(1 + D)`. -/
theorem closest_hyperplane_ratio {m E h D excess : ℝ}
    (hm : 0 < m) (hE : 0 < E) (hD : 0 ≤ D)
    (hh : h ≤ E / m) (hex : excess ≤ h + D * (E + h)) :
    1 + excess / E ≤ (1 + 1 / m) * (1 + D) := by
  have h1 : excess ≤ E / m + D * (E + E / m) := by nlinarith
  have h2 : excess / E ≤ (1 + 1 / m) * (1 + D) - 1 := by
    rw [div_le_iff₀ hE]
    have hrew : ((1 + 1 / m) * (1 + D) - 1) * E = E / m + D * (E + E / m) := by
      field_simp [ne_of_gt hm]; ring
    linarith
  linarith

/-- **Lemma 2(3) (Orthogonal-lines averaging bound)**:
Given `m ≥ 1` nonnegative line energies `e : Fin m → ℝ` summing to `E` and `k ≤ m`,
there exists a subset `K` of size `k` (the `k` heaviest lines) whose complement has energy
at most `((m - k) / m) * E`. -/
theorem orthogonal_lines_excess {m k : ℕ} (hm : 1 ≤ m) (hkm : k ≤ m) (e : Fin m → ℝ) :
    ∃ K : Finset (Fin m), K.card = k ∧
      ∑ j ∈ Finset.univ \ K, e j ≤ (((m - k : ℕ) : ℝ) / m) * ∑ j : Fin m, e j := by
  -- Averaging over all k-subsets K of Fin m via double counting!
  let F := (Finset.univ : Finset (Fin m)).powersetCard k
  have hFne : F.Nonempty := by
    rw [Finset.powersetCard_nonempty]
    simpa using hkm
  have hFcard : F.card = Nat.choose m k := by
    simp [F, Finset.card_powersetCard]
  have hsum_avg : ∑ K ∈ F, ∑ j ∈ Finset.univ \ K, e j =
      (Nat.choose (m - 1) k : ℝ) * ∑ j : Fin m, e j := by
    have h1 : ∑ K ∈ F, ∑ j ∈ Finset.univ \ K, e j =
        ∑ K ∈ F, ∑ j : Fin m, if j ∉ K then e j else 0 := by
      apply Finset.sum_congr rfl; intro K _
      rw [← Finset.sum_filter]
      congr 1; ext j; simp
    rw [h1, Finset.sum_comm, Finset.mul_sum]
    apply Finset.sum_congr rfl; intro j _
    rw [← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul]
    congr 1
    -- Count k-subsets of Fin m avoiding j: equals C(m - 1, k)
    have hcnt : (F.filter (fun K => j ∉ K)).card = Nat.choose (m - 1) k := by
      have heq : F.filter (fun K => j ∉ K) = (Finset.univ.erase j).powersetCard k := by
        ext K
        simp only [F, Finset.mem_filter, Finset.mem_powersetCard, Finset.subset_univ, true_and]
        constructor
        · rintro ⟨hcard, hnj⟩
          refine ⟨?_, hcard⟩
          intro x hx; exact Finset.mem_erase.2 ⟨fun hxj => hnj (hxj ▸ hx), Finset.mem_univ x⟩
        · rintro ⟨hsub, hcard⟩
          exact ⟨hcard, fun hjK => (Finset.mem_erase.1 (hsub hjK)).1 rfl⟩
      rw [heq, Finset.card_powersetCard, Finset.card_erase_of_mem (Finset.mem_univ j),
        Finset.card_fin]
    exact_mod_cast hcnt
  have hbinom_id : m * Nat.choose (m - 1) k = (m - k) * Nat.choose m k := by
    rcases eq_or_lt_of_le hkm with rfl | hlt
    · simp [Nat.choose_eq_zero_of_lt (show k - 1 < k by omega)]
    · have hsym1 : Nat.choose (m - 1) k = Nat.choose (m - 1) (m - 1 - k) := by
        rw [Nat.choose_symm (by omega)]
      have hsym2 : Nat.choose m k = Nat.choose m (m - k) := by
        rw [Nat.choose_symm hkm]
      have hsucc : m - 1 - k + 1 = m - k := by omega
      have hms : m - 1 + 1 = m := by omega
      rw [hsym1, hsym2]
      have h2 := Nat.succ_mul_choose_eq (m - 1) (m - 1 - k)
      simp only [Nat.succ_eq_add_one] at h2
      rw [hms, hsucc] at h2
      linarith
  have havg_eq : (∑ K ∈ F, ∑ j ∈ Finset.univ \ K, e j) =
      (F.card : ℝ) * ((((m - k : ℕ) : ℝ) / m) * ∑ j : Fin m, e j) := by
    rw [hsum_avg, hFcard]
    have hbinR : (m : ℝ) * (Nat.choose (m - 1) k : ℝ) =
        ((m - k : ℕ) : ℝ) * (Nat.choose m k : ℝ) := by exact_mod_cast hbinom_id
    have hm_ne : (m : ℝ) ≠ 0 := Nat.cast_ne_zero.2 (by omega)
    field_simp [hm_ne]
    calc (Nat.choose (m - 1) k : ℝ) * (∑ j : Fin m, e j) * (m : ℝ)
        = ((m : ℝ) * (Nat.choose (m - 1) k : ℝ)) * (∑ j : Fin m, e j) := by ring
      _ = (((m - k : ℕ) : ℝ) * (Nat.choose m k : ℝ)) * (∑ j : Fin m, e j) := by rw [hbinR]
      _ = (∑ j : Fin m, e j) * (Nat.choose m k : ℝ) * ((m - k : ℕ) : ℝ) := by ring
  -- By pigeonhole, some K ∈ F attains ≤ average
  by_contra hno
  push_neg at hno
  have hgt : ∀ K ∈ F, (((m - k : ℕ) : ℝ) / m) * ∑ j : Fin m, e j < ∑ j ∈ Finset.univ \ K, e j := by
    intro K hK
    exact hno K (Finset.mem_powersetCard.1 hK).2
  have hsum_gt : ∑ _K ∈ F, ((((m - k : ℕ) : ℝ) / m) * ∑ j : Fin m, e j) <
      ∑ K ∈ F, ∑ j ∈ Finset.univ \ K, e j :=
    Finset.sum_lt_sum_of_nonempty hFne hgt
  rw [Finset.sum_const, nsmul_eq_mul] at hsum_gt
  linarith

/-! ### 4. Base and Inductive Risk-Ratio Upper Bounds (Theorems 3 & 5) -/

/-- Anchor reduction inequality: `1 - k / (d - 1) ≤ 1 - k / d`. -/
theorem anchor_bound_le {d k : ℕ} (hd : 2 ≤ d) :
    1 - (k : ℝ) / ((d - 1 : ℕ) : ℝ) ≤ 1 - (k : ℝ) / d := by
  have hd1 : (0 : ℝ) < (d - 1 : ℕ) := Nat.cast_pos.2 (by omega)
  have hle : ((d - 1 : ℕ) : ℝ) ≤ d := by exact_mod_cast Nat.sub_le d 1
  have hdiv : (k : ℝ) / d ≤ (k : ℝ) / ((d - 1 : ℕ) : ℝ) :=
    div_le_div_of_nonneg_left (Nat.cast_nonneg k) hd1 hle
  linarith

/-- Rank-`r` (`r ≥ 2`) positive-circuit reduction on the upper half `2k ≥ d`:
preserves `2(k - 1) ≥ d - r` and improves `1 - (k - 1)/(d - r) ≤ 1 - k/d`. -/
theorem circuit_bound_le {d k r : ℕ} (hd : 2 ≤ d) (h2k : d ≤ 2 * k) (hr : 2 ≤ r) (hrd : r < d) :
    d - r ≤ 2 * (k - 1) ∧
    1 - ((k - 1 : ℕ) : ℝ) / ((d - r : ℕ) : ℝ) ≤ 1 - (k : ℝ) / d := by
  constructor
  · omega
  · have hdr_pos : (0 : ℝ) < (d - r : ℕ) := Nat.cast_pos.2 (by omega)
    have hd_pos : (0 : ℝ) < d := Nat.cast_pos.2 (by omega)
    have hk1 : 1 ≤ k := by omega
    have hdr_cast : ((d - r : ℕ) : ℝ) = (d : ℝ) - r := Nat.cast_sub (le_of_lt hrd)
    have hk1_cast : ((k - 1 : ℕ) : ℝ) = (k : ℝ) - 1 := by simpa using Nat.cast_sub hk1
    rw [sub_le_sub_iff_left, div_le_div_iff₀ hd_pos hdr_pos, hdr_cast, hk1_cast]
    have h2kR : (d : ℝ) ≤ 2 * k := by exact_mod_cast h2k
    have hrR : (2 : ℝ) ≤ r := by exact_mod_cast hr
    nlinarith

/-- Rank-deficient feature-span reduction (`1 ≤ m < d`, `k' = d + k - m`):
`1 - k' / m ≤ 1 - k / d`. -/
theorem rank_deficient_bound_le {d k m : ℕ} (hm : 1 ≤ m) (hmd : m ≤ d) (hkd : k ≤ d) :
    1 - ((d + k - m : ℕ) : ℝ) / m ≤ 1 - (k : ℝ) / d := by
  have hm_pos : (0 : ℝ) < m := Nat.cast_pos.2 hm
  have hd_pos : (0 : ℝ) < d := Nat.cast_pos.2 (by omega)
  rw [sub_le_sub_iff_left, div_le_div_iff₀ hd_pos hm_pos]
  have hcast : ((d + k - m : ℕ) : ℝ) = (d : ℝ) + k - m := by
    rw [Nat.cast_sub (by omega), Nat.cast_add]
  rw [hcast]
  have hmdR : (m : ℝ) ≤ d := Nat.cast_le.2 hmd
  have hkdR : (k : ℝ) ≤ d := Nat.cast_le.2 hkd
  nlinarith

/-- Abstract reduction system for the upper-half excess coefficient `c(d, k)` encoding
Lemma 2(1)–(4): exact recovery (`k ≥ d`), anchor reduction, all rank-1 lines, or rank `r ≥ 2`
positive-circuit completion. -/
structure UpperHalfReduction (excess : ℕ → ℕ → ℝ) : Prop where
  exact_recovery : ∀ d k, d ≤ k → excess d k ≤ 0
  step : ∀ d k, 1 ≤ d → k < d →
    (d ≥ 2 ∧ excess d k ≤ excess (d - 1) k) ∨
    excess d k ≤ 1 - (k : ℝ) / d ∨
    (∃ r, 2 ≤ r ∧ r ≤ d ∧ 1 ≤ k ∧ excess d k ≤ excess (d - r) (k - 1))

/-- **Theorem 3 (Inductive upper bound for `2k ≥ d`)**:
Every system obeying the Lemma 2 reductions satisfies `excess d k ≤ 1 - k / d` for all `d ≥ 1`
and `2k ≥ d`. -/
theorem upper_half_inductive_bound {excess : ℕ → ℕ → ℝ} (hsys : UpperHalfReduction excess) :
    ∀ d k, 1 ≤ d → d ≤ 2 * k → excess d k ≤ max 0 (1 - (k : ℝ) / d) := by
  intro d
  induction d using Nat.strong_induction_on with
  | h d ih =>
    intro k hd h2k
    by_cases hkd : d ≤ k
    · exact le_trans (hsys.exact_recovery d k hkd) (le_max_left _ _)
    · push_neg at hkd
      have hmax : max (0 : ℝ) (1 - (k : ℝ) / d) = 1 - (k : ℝ) / d := by
        apply max_eq_right
        have hd_pos : (0 : ℝ) < d := Nat.cast_pos.2 hd
        rw [sub_nonneg, div_le_one hd_pos]
        exact Nat.cast_le.2 (le_of_lt hkd)
      rw [hmax]
      rcases hsys.step d k hd hkd with ⟨hd2, hanch⟩ | hlines | ⟨r, hr2, hrd, hk1, hcirc⟩
      · have ih1 := ih (d - 1) (by omega) k (by omega) (by omega)
        by_cases hkd1 : d - 1 ≤ k
        · have h0 : excess (d - 1) k ≤ 0 := hsys.exact_recovery (d - 1) k hkd1
          have hnn : 0 ≤ 1 - (k : ℝ) / d := by
            rw [sub_nonneg, div_le_one (Nat.cast_pos.2 hd)]; exact Nat.cast_le.2 (le_of_lt hkd)
          linarith
        · push_neg at hkd1
          have hnn1 : 0 ≤ 1 - (k : ℝ) / ((d - 1 : ℕ) : ℝ) := by
            rw [sub_nonneg, div_le_one (Nat.cast_pos.2 (by omega))]
            exact Nat.cast_le.2 (le_of_lt hkd1)
          rw [max_eq_right hnn1] at ih1
          exact le_trans hanch (le_trans ih1 (anchor_bound_le hd2))
      · exact hlines
      · by_cases hrd_eq : r = d
        · have h0 : excess 0 (k - 1) ≤ 0 := hsys.exact_recovery 0 (k - 1) (Nat.zero_le _)
          have hnn : 0 ≤ 1 - (k : ℝ) / d := by
            rw [sub_nonneg, div_le_one (Nat.cast_pos.2 hd)]; exact Nat.cast_le.2 (le_of_lt hkd)
          rw [hrd_eq, Nat.sub_self] at hcirc
          linarith
        · have hrd_lt : r < d := lt_of_le_of_ne hrd hrd_eq
          have hd2 : 2 ≤ d := by omega
          have ⟨h2k', hbound⟩ := circuit_bound_le hd2 h2k hr2 hrd_lt
          have ih_r := ih (d - r) (by omega) (k - 1) (by omega) h2k'
          have hnn : 0 ≤ 1 - (k : ℝ) / d := by
            rw [sub_nonneg, div_le_one (Nat.cast_pos.2 hd)]; exact Nat.cast_le.2 (le_of_lt hkd)
          have hmax_le : max (0 : ℝ) (1 - ((k - 1 : ℕ) : ℝ) / ((d - r : ℕ) : ℝ)) ≤ 1 - (k : ℝ) / d :=
            max_le hnn hbound
          exact le_trans hcirc (le_trans ih_r hmax_le)

/-! ### 5. Odd Boundary `d = 2k + 1` Reductions and Induction (Theorem 5) -/

/-- Target odd-boundary excess coefficient `c_d = (d + 1) / (2d)`. -/
noncomputable def oddC (d : ℕ) : ℝ := ((d : ℝ) + 1) / (2 * (d : ℝ))

lemma oddC_eq_one_sub {k : ℕ} :
    oddC (2 * k + 1) = 1 - (k : ℝ) / (2 * k + 1) := by
  unfold oddC
  push_cast
  have hne : (2 * (k : ℝ) + 1) ≠ 0 := by positivity
  field_simp [hne]
  ring

/-- Base `d = 3`: `min(e, 2(1 - e)) ≤ 2/3`. -/
theorem odd_base_d3_line (e : ℝ) : min e (2 * (1 - e)) ≤ 2 / 3 := by
  by_cases h : e ≤ 2 / 3
  · exact le_trans (min_le_left _ _) h
  · exact le_trans (min_le_right _ _) (by linarith)

/-- Step 1 (Heavy triangle plane): `(1 - 1/(kd)) c_{d - 2} = c_d` for `d = 2k + 1 ≥ 5`. -/
theorem odd_heavy_triangle {k : ℕ} (hk : 2 ≤ k) :
    let d := 2 * k + 1
    let h : ℝ := 1 / ((k : ℝ) * d)
    (1 - h) * oddC (d - 2) = oddC d := by
  intro d h
  unfold oddC
  have hd2_cast : ((d - 2 : ℕ) : ℝ) = 2 * (k : ℝ) - 1 := by
    have : d - 2 = 2 * k - 1 := by omega
    rw [this, Nat.cast_sub (by omega)]
    push_cast; ring
  have hd_cast : (d : ℝ) = 2 * (k : ℝ) + 1 := by simp [d]
  rw [hd2_cast, hd_cast]
  have hkR : (2 : ℝ) ≤ k := by exact_mod_cast hk
  have hk_ne : (k : ℝ) ≠ 0 := Nat.cast_ne_zero.2 (by omega)
  have h1 : 2 * (k : ℝ) - 1 ≠ 0 := by linarith
  have h2 : 2 * (k : ℝ) + 1 ≠ 0 := by linarith
  dsimp only [h]
  rw [hd_cast]
  have hL : (1 - 1 / ((k : ℝ) * (2 * (k : ℝ) + 1))) * ((2 * (k : ℝ) - 1 + 1) / (2 * (2 * (k : ℝ) - 1))) =
      ((k : ℝ) * (2 * (k : ℝ) + 1) - 1) * (2 * (k : ℝ)) /
        (((k : ℝ) * (2 * (k : ℝ) + 1)) * (2 * (2 * (k : ℝ) - 1))) := by
    field_simp [hk_ne, h1, h2]
    ring
  rw [hL, div_eq_div_iff (by positivity) (by positivity)]
  ring

/-- Step 2 (Light single original line): `e ≤ 1/d → e + (1 - e)/2 ≤ c_d`. -/
theorem odd_light_line {d : ℕ} (hd : 1 ≤ d) {e : ℝ} (he : e ≤ 1 / (d : ℝ)) :
    e + (1 - e) / 2 ≤ oddC d := by
  have hd_ne : (d : ℝ) ≠ 0 := Nat.cast_ne_zero.2 (by omega)
  have hrew : oddC d = 1 / 2 + 1 / (2 * (d : ℝ)) := by
    unfold oddC; field_simp [hd_ne]
  rw [hrew]
  have hhalf : e / 2 ≤ 1 / (2 * (d : ℝ)) := by
    have : (1 / (d : ℝ)) / 2 = 1 / (2 * (d : ℝ)) := by ring
    linarith
  linarith

/-- Step 3 (Two single original lines): `s ≥ 2/d → s/2 + (1 - s) c_{d - 2} ≤ c_d` for `d = 2k + 1 ≥ 5`. -/
theorem odd_two_lines {k : ℕ} (hk : 2 ≤ k) {s : ℝ} (hs : 2 / ((2 * k + 1 : ℕ) : ℝ) ≤ s) :
    s / 2 + (1 - s) * oddC (2 * k - 1) ≤ oddC (2 * k + 1) := by
  unfold oddC
  have hkR : (2 : ℝ) ≤ k := by exact_mod_cast hk
  have hcast1 : ((2 * k - 1 : ℕ) : ℝ) = 2 * (k : ℝ) - 1 := by
    rw [Nat.cast_sub (by omega)]; push_cast; ring
  have hcast2 : ((2 * k + 1 : ℕ) : ℝ) = 2 * (k : ℝ) + 1 := by push_cast; ring
  rw [hcast2] at hs
  rw [hcast1, hcast2]
  have hden1 : 0 < 2 * (k : ℝ) - 1 := by linarith
  have hden2 : 0 < 2 * (k : ℝ) + 1 := by linarith
  have hhalf : (1 / 2 : ℝ) < (2 * (k : ℝ) - 1 + 1) / (2 * (2 * (k : ℝ) - 1)) := by
    rw [div_lt_div_iff₀ (by norm_num) (by linarith)]; linarith
  have hdiff :
      ((2 * (k : ℝ) + 1 + 1) / (2 * (2 * (k : ℝ) + 1))) -
      (s / 2 + (1 - s) * ((2 * (k : ℝ) - 1 + 1) / (2 * (2 * (k : ℝ) - 1)))) =
      (s - 2 / (2 * (k : ℝ) + 1)) * ((2 * (k : ℝ) - 1 + 1) / (2 * (2 * (k : ℝ) - 1)) - 1 / 2) := by
    field_simp [ne_of_gt hden1, ne_of_gt hden2]
    ring
  nlinarith

/-- Step 4 core quadratic inequality:
`4a² - 2ab - 6a + b² - 1 ≥ 0` for all integers `a ≥ 1`, odd `b ≥ 1`, with `2a + b ≥ 5`. -/
theorem odd_packing_quadratic {a b : ℕ} (ha : 1 ≤ a) (hb_odd : b % 2 = 1) (hd : 5 ≤ 2 * a + b) :
    (0 : ℝ) ≤ 4 * (a : ℝ) ^ 2 - 2 * a * b - 6 * a + b ^ 2 - 1 := by
  have haR : (1 : ℝ) ≤ a := by exact_mod_cast ha
  by_cases hb1 : b = 1
  · subst hb1
    have ha2 : (2 : ℝ) ≤ a := by exact_mod_cast (show 2 ≤ a by omega)
    norm_num; nlinarith
  · by_cases hb3 : b = 3
    · subst hb3
      rcases eq_or_lt_of_le ha with rfl | ha2
      · norm_num
      · have ha2R : (2 : ℝ) ≤ a := by exact_mod_cast ha2
        norm_num; nlinarith
    · have hb5 : (5 : ℝ) ≤ b := by exact_mod_cast (show 5 ≤ b by omega)
      have hsq : 4 * (4 * (a : ℝ) ^ 2 - 2 * a * b - 6 * a + b ^ 2 - 1) =
          (4 * (a : ℝ) - b - 3) ^ 2 + (3 * (b : ℝ) ^ 2 - 6 * b - 13) := by ring
      have hdisc : 0 < 3 * (b : ℝ) ^ 2 - 6 * b - 13 := by nlinarith
      nlinarith [sq_nonneg (4 * (a : ℝ) - b - 3)]

/-- Step 4 (Complete all `a` triangles):
`((b + 1) / 2) * (3 * a * h) ≤ c_d` for `d = 2a + b ≥ 5`, `a ≥ 1`, `b ≥ 1` odd, `k = a + (b - 1)/2`. -/
theorem odd_all_triangles_bound {a b : ℕ} (ha : 1 ≤ a) (hb_odd : b % 2 = 1) (hd5 : 5 ≤ 2 * a + b) :
    let d := 2 * a + b
    let k := a + (b - 1) / 2
    let h : ℝ := 1 / ((k : ℝ) * d)
    (((b : ℝ) + 1) / 2) * (3 * a * h) ≤ oddC d := by
  intro d k h
  have hd_pos : (0 : ℝ) < d := Nat.cast_pos.2 (by omega)
  have hk_eq : 2 * k + 1 = d := by omega
  have hk_pos : (0 : ℝ) < k := Nat.cast_pos.2 (by omega)
  have hkR : (k : ℝ) = (2 * (a : ℝ) + b - 1) / 2 := by
    have : 2 * (k : ℝ) + 1 = 2 * a + b := by exact_mod_cast hk_eq
    linarith
  have hdR : (d : ℝ) = 2 * a + b := by simp [d]
  unfold oddC
  dsimp only [h]
  rw [show (((b : ℝ) + 1) / 2) * (3 * a * (1 / ((k : ℝ) * d))) =
      3 * a * (b + 1) / (2 * k * d) by ring]
  rw [div_le_div_iff₀ (by positivity) (by positivity)]
  have hquad := odd_packing_quadratic ha hb_odd hd5
  have hdiff : ((d : ℝ) + 1) * (2 * k * d) - 3 * a * (b + 1) * (2 * d) =
      (d : ℝ) * (4 * (a : ℝ) ^ 2 - 2 * a * b - 6 * a + b ^ 2 - 1) := by
    rw [hkR, hdR]; ring
  nlinarith

/-- Step 5 (`b = 1` with single original line `e ≥ c_d`):
complement excess coefficient `D = (1 + 1/(d-1))(1 + c_{d-2}) - 1 ≤ (d + 1)/(d - 1)`
and `D (1 - e) ≤ c_d` for all odd `d = 2k + 1 ≥ 5`. -/
theorem odd_single_line_complement {k : ℕ} (hk : 2 ≤ k) {e : ℝ} (he : oddC (2 * k + 1) ≤ e) :
    let d := 2 * k + 1
    let D : ℝ := (1 + 1 / ((d - 1 : ℕ) : ℝ)) * (1 + oddC (d - 2)) - 1
    D = ((d : ℝ) ^ 2 + d - 4) / (2 * ((d - 2 : ℕ) : ℝ) * ((d - 1 : ℕ) : ℝ)) ∧
    D ≤ ((d : ℝ) + 1) / ((d - 1 : ℕ) : ℝ) ∧
    D * (1 - e) ≤ oddC d := by
  intro d D
  have hkR : (2 : ℝ) ≤ k := by exact_mod_cast hk
  have hd_cast : (d : ℝ) = 2 * (k : ℝ) + 1 := by simp [d]
  have hd1_cast : ((d - 1 : ℕ) : ℝ) = 2 * (k : ℝ) := by
    have : d - 1 = 2 * k := by omega
    rw [this]; push_cast; ring
  have hd2_cast : ((d - 2 : ℕ) : ℝ) = 2 * (k : ℝ) - 1 := by
    have : d - 2 = 2 * k - 1 := by omega
    rw [this, Nat.cast_sub (by omega)]; push_cast; ring
  have h1 : 0 < 2 * (k : ℝ) - 1 := by linarith
  have h2 : 0 < 2 * (k : ℝ) := by linarith
  have h3 : 0 < 2 * (k : ℝ) + 1 := by linarith
  have hD_eq : D = ((d : ℝ) ^ 2 + d - 4) / (2 * ((d - 2 : ℕ) : ℝ) * ((d - 1 : ℕ) : ℝ)) := by
    dsimp only [D]; unfold oddC
    rw [hd_cast, hd1_cast, hd2_cast]
    field_simp [ne_of_gt h1, ne_of_gt h2]
    ring
  have hD_le : D ≤ ((d : ℝ) + 1) / ((d - 1 : ℕ) : ℝ) := by
    rw [hD_eq, hd_cast, hd1_cast, hd2_cast]
    rw [div_le_div_iff₀ (by positivity) h2]
    nlinarith
  have hD_pos : 0 ≤ D := by
    rw [hD_eq, hd_cast, hd1_cast, hd2_cast]
    apply div_nonneg _ (by positivity)
    nlinarith
  have h1me_le : 1 - e ≤ 1 - oddC d := by linarith
  have hprod : D * (1 - e) ≤ oddC d := by
    have hstep1 : D * (1 - e) ≤ (((d : ℝ) + 1) / ((d - 1 : ℕ) : ℝ)) * (1 - oddC d) := by
      have h1mc_nn : 0 ≤ 1 - oddC d := by
        unfold oddC; rw [hd_cast, sub_nonneg, div_le_one (by linarith)]; linarith
      calc D * (1 - e) ≤ D * (1 - oddC d) := mul_le_mul_of_nonneg_left h1me_le hD_pos
        _ ≤ (((d : ℝ) + 1) / ((d - 1 : ℕ) : ℝ)) * (1 - oddC d) :=
            mul_le_mul_of_nonneg_right hD_le h1mc_nn
    have hstep2 : (((d : ℝ) + 1) / ((d - 1 : ℕ) : ℝ)) * (1 - oddC d) = oddC d := by
      unfold oddC; rw [hd_cast, hd1_cast]
      field_simp [ne_of_gt h2, ne_of_gt h3]
      ring
    linarith
  exact ⟨hD_eq, hD_le, hprod⟩

/-- Abstract reduction system for the odd boundary `d = 2k + 1` encoding the base `d = 3`
and the 5 inductive branches of Theorem 5. -/
structure OddBoundaryReduction (oddExcess : ℕ → ℝ) : Prop where
  base_d3 : (∃ e : ℝ, oddExcess 1 ≤ min e (2 * (1 - e))) ∨ oddExcess 1 ≤ 2 / 3
  step : ∀ k : ℕ, 2 ≤ k →
    let d : ℕ := 2 * k + 1
    let h : ℝ := 1 / ((k : ℝ) * (d : ℝ))
    -- Anchor / rank ≥ 3 circuit / all rank-1 lines
    oddExcess k ≤ oddC d ∨
    -- Branch 1: heavy triangle plane (h ≤ e ≤ 1)
    (∃ e : ℝ, h ≤ e ∧ e ≤ 1 ∧ oddExcess k ≤ (1 - e) * oddExcess (k - 1)) ∨
    -- Branch 2: light single original line (e ≤ 1/d)
    (∃ e : ℝ, e ≤ 1 / (d : ℝ) ∧ oddExcess k ≤ e + (1 - e) / 2) ∨
    -- Branch 3: two single original lines (s ≥ 2/d)
    (∃ s : ℝ, 2 / (d : ℝ) ≤ s ∧ s ≤ 1 ∧ oddExcess k ≤ s / 2 + (1 - s) * oddExcess (k - 1)) ∨
    -- Branch 4: complete all `a` triangles (b ≥ 1 odd, a ≥ 1, 2a + b = d)
    (∃ a b : ℕ, 1 ≤ a ∧ b % 2 = 1 ∧ 2 * a + b = d ∧
      oddExcess k ≤ (((b : ℝ) + 1) / 2) * (3 * (a : ℝ) * h)) ∨
    -- Branch 5: b = 1 with single original line of energy e
    (∃ e : ℝ, oddExcess k ≤ e ∧ e ≤ oddC d) ∨
    (∃ e : ℝ, oddC d ≤ e ∧ e ≤ 1 ∧
      oddExcess k ≤ ((1 + 1 / ((d - 1 : ℕ) : ℝ)) * (1 + oddExcess (k - 1)) - 1) * (1 - e))

/-- **Theorem 5 (Inductive upper bound for the odd boundary `d = 2k + 1`)**:
For every `k ≥ 1`, `oddExcess k ≤ oddC (2 * k + 1) = 1 - k / (2k + 1)`. -/
theorem odd_boundary_inductive_bound {oddExcess : ℕ → ℝ} (hsys : OddBoundaryReduction oddExcess) :
    ∀ k, 1 ≤ k → oddExcess k ≤ 1 - (k : ℝ) / (2 * k + 1) := by
  intro k hk
  rw [← oddC_eq_one_sub]
  induction k, hk using Nat.le_induction with
  | base =>
    have hc3 : oddC (2 * 1 + 1) = 2 / 3 := by unfold oddC; norm_num
    rw [hc3]
    rcases hsys.base_d3 with ⟨e, he⟩ | hle
    · exact le_trans he (odd_base_d3_line e)
    · exact hle
  | succ k hk ih =>
    have hk1 : 2 ≤ k + 1 := by omega
    have hprev : oddExcess ((k + 1) - 1) ≤ oddC (2 * (k + 1) + 1 - 2) := by
      simpa using ih
    rcases hsys.step (k + 1) hk1 with
      hd | ⟨e, he, he1, hle⟩ | ⟨e, he, hle⟩ | ⟨s, hs, hs1, hle⟩ |
      ⟨a, b, ha, hbo, hab, hle⟩ | ⟨e, hle, he⟩ | ⟨e, he, he1, hle⟩
    · exact hd
    · have h1 : 1 - e ≤ 1 - 1 / (((k + 1 : ℕ) : ℝ) * ((2 * (k + 1) + 1 : ℕ) : ℝ)) := by linarith
      have hc_nn : 0 ≤ oddC (2 * (k + 1) + 1 - 2) := by
        unfold oddC; positivity
      have h1e : 0 ≤ 1 - e := by linarith
      have h1e_le : (1 - e) * oddExcess ((k + 1) - 1) ≤
          (1 - 1 / (((k + 1 : ℕ) : ℝ) * ((2 * (k + 1) + 1 : ℕ) : ℝ))) * oddC (2 * (k + 1) + 1 - 2) := by
        calc (1 - e) * oddExcess ((k + 1) - 1)
            ≤ (1 - e) * oddC (2 * (k + 1) + 1 - 2) := mul_le_mul_of_nonneg_left hprev h1e
          _ ≤ (1 - 1 / (((k + 1 : ℕ) : ℝ) * ((2 * (k + 1) + 1 : ℕ) : ℝ))) * oddC (2 * (k + 1) + 1 - 2) :=
              mul_le_mul_of_nonneg_right h1 hc_nn
      have hht := odd_heavy_triangle hk1
      dsimp only at hht
      linarith
    · exact le_trans hle (odd_light_line (by omega) he)
    · have h1ms : 0 ≤ 1 - s := by linarith
      have hstep : s / 2 + (1 - s) * oddExcess ((k + 1) - 1) ≤
          s / 2 + (1 - s) * oddC (2 * (k + 1) - 1) := by
        have hprev' : oddExcess ((k + 1) - 1) ≤ oddC (2 * (k + 1) - 1) := by
          simpa [show 2 * (k + 1) + 1 - 2 = 2 * (k + 1) - 1 by omega] using hprev
        nlinarith
      exact le_trans hle (le_trans hstep (odd_two_lines hk1 hs))
    · have hd5 : 5 ≤ 2 * a + b := by omega
      have htri := odd_all_triangles_bound ha hbo hd5
      have hk_eq : a + (b - 1) / 2 = k + 1 := by omega
      rw [hab, hk_eq] at htri
      exact le_trans hle htri
    · exact le_trans hle he
    · have h1me : 0 ≤ 1 - e := by linarith
      have hfac : 0 ≤ 1 + 1 / (((2 * (k + 1) + 1 - 1 : ℕ) : ℝ)) := by positivity
      have hD_le : (1 + 1 / (((2 * (k + 1) + 1 - 1 : ℕ) : ℝ))) * (1 + oddExcess ((k + 1) - 1)) - 1 ≤
          (1 + 1 / (((2 * (k + 1) + 1 - 1 : ℕ) : ℝ))) * (1 + oddC (2 * (k + 1) + 1 - 2)) - 1 := by
        nlinarith
      have hcomp := (odd_single_line_complement hk1 he).2.2
      calc oddExcess (k + 1)
          ≤ ((1 + 1 / (((2 * (k + 1) + 1 - 1 : ℕ) : ℝ))) * (1 + oddExcess ((k + 1) - 1)) - 1) * (1 - e) := hle
        _ ≤ ((1 + 1 / (((2 * (k + 1) + 1 - 1 : ℕ) : ℝ))) * (1 + oddC (2 * (k + 1) + 1 - 2)) - 1) * (1 - e) :=
            mul_le_mul_of_nonneg_right hD_le h1me
        _ ≤ oddC (2 * (k + 1) + 1) := hcomp

/-- Concrete upper-half evaluations: `1 + Γ_{5, 2} = 8/5` (`Fw(5, 7)`) and
`1 + Γ_{5, 3} = 7/5` (`Fw(5, 8)`). -/
theorem fw_5_7_and_5_8_values :
    1 + Gamma 5 2 (by omega) = 8 / 5 ∧ 1 + Gamma 5 3 (by omega) = 7 / 5 := by
  rw [one_add_gamma_upper_half (by omega) (by omega) (by omega),
      one_add_gamma_upper_half (by omega) (by omega) (by omega)]
  norm_num

/-! ### 6. Exact Bounds for the First Unresolved Cell `(5, 6)` (Proposition 6) -/

/-- Two-block orthogonal simplex features in `ℝ^5`:
Block 1 (`i = 0, 1, 2`, rank 2 on coords `0, 1`), Block 2 (`i = 3, 4, 5, 6`, rank 3 on coords `2, 3, 4`). -/
def simplex56X : Fin 7 → Fin 5 → ℝ
  | 0, j => if j = 0 then 1 else 0
  | 1, j => if j = 1 then 1 else 0
  | 2, j => if j = 0 then -1 else if j = 1 then -1 else 0
  | 3, j => if j = 2 then 1 else 0
  | 4, j => if j = 3 then 1 else 0
  | 5, j => if j = 4 then 1 else 0
  | 6, j => if j = 2 then -1 else if j = 3 then -1 else if j = 4 then -1 else 0

/-- Shift vector `v = (10, 20, 10, 20, 30)` lying outside every proper row span. -/
def simplex56V : Fin 5 → ℝ
  | 0 => 10
  | 1 => 20
  | 2 => 10
  | 3 => 20
  | 4 => 30

/-- Block residuals `a₂ = 1/√6` (`6 a₂² = 1`) and `a₃ = 1/√12` (`12 a₃² = 1`). -/
noncomputable def simplex56Res : Fin 7 → ℝ
  | 0 => 1 / Real.sqrt 6
  | 1 => 1 / Real.sqrt 6
  | 2 => 1 / Real.sqrt 6
  | 3 => 1 / Real.sqrt 12
  | 4 => 1 / Real.sqrt 12
  | 5 => 1 / Real.sqrt 12
  | 6 => 1 / Real.sqrt 12

/-- Shifted labels `y_i = ⟨x_i, v⟩ + a_block`. -/
noncomputable def simplex56Y (i : Fin 7) : ℝ :=
  dotProduct (simplex56X i) simplex56V + simplex56Res i

/-- Unnormalized full squared loss `∑_{i=0}^6 (⟨x_i, w⟩ - y_i)²` on the 7-row dataset. -/
noncomputable def simplex56Loss (w : Fin 5 → ℝ) : ℝ :=
  ∑ i : Fin 7, (dotProduct (simplex56X i) w - simplex56Y i) ^ 2

@[simp] lemma dot_s56_0 (w : Fin 5 → ℝ) : dotProduct (simplex56X 0) w = w 0 := by
  simp [dotProduct, simplex56X]
@[simp] lemma dot_s56_1 (w : Fin 5 → ℝ) : dotProduct (simplex56X 1) w = w 1 := by
  simp [dotProduct, simplex56X]
@[simp] lemma dot_s56_2 (w : Fin 5 → ℝ) : dotProduct (simplex56X 2) w = -w 0 - w 1 := by
  simp [dotProduct, Fin.sum_univ_five, simplex56X]; ring
@[simp] lemma dot_s56_3 (w : Fin 5 → ℝ) : dotProduct (simplex56X 3) w = w 2 := by
  simp [dotProduct, simplex56X]
@[simp] lemma dot_s56_4 (w : Fin 5 → ℝ) : dotProduct (simplex56X 4) w = w 3 := by
  simp [dotProduct, simplex56X]
@[simp] lemma dot_s56_5 (w : Fin 5 → ℝ) : dotProduct (simplex56X 5) w = w 4 := by
  simp [dotProduct, simplex56X]
@[simp] lemma dot_s56_6 (w : Fin 5 → ℝ) : dotProduct (simplex56X 6) w = -w 2 - w 3 - w 4 := by
  simp [dotProduct, Fin.sum_univ_five, simplex56X]; ring

theorem simplex56_loss_decomp (w : Fin 5 → ℝ) :
    let u := fun j : Fin 5 => w j - simplex56V j
    simplex56Loss w =
      5 / 6 + (u 0 ^ 2 + u 1 ^ 2 + (u 0 + u 1) ^ 2) +
        (u 2 ^ 2 + u 3 ^ 2 + u 4 ^ 2 + (u 2 + u 3 + u 4) ^ 2) := by
  intro u
  unfold simplex56Loss
  simp only [Fin.sum_univ_seven, simplex56Y, simplex56Res, dot_s56_0, dot_s56_1, dot_s56_2,
    dot_s56_3, dot_s56_4, dot_s56_5, dot_s56_6, simplex56V, u]
  have hsq6 : (1 / Real.sqrt 6) ^ 2 = 1 / 6 := by
    rw [div_pow, one_pow, Real.sq_sqrt (by norm_num)]
  have hsq12 : (1 / Real.sqrt 12) ^ 2 = 1 / 12 := by
    rw [div_pow, one_pow, Real.sq_sqrt (by norm_num)]
  nlinarith [hsq6, hsq12]

theorem simplex56_optLoss :
    simplex56Loss simplex56V = 5 / 6 ∧ ∀ w : Fin 5 → ℝ, 5 / 6 ≤ simplex56Loss w := by
  constructor
  · rw [simplex56_loss_decomp]; simp
  · intro w
    rw [simplex56_loss_decomp]
    nlinarith [sq_nonneg (w 0 - simplex56V 0), sq_nonneg (w 1 - simplex56V 1),
      sq_nonneg (w 0 - simplex56V 0 + (w 1 - simplex56V 1)),
      sq_nonneg (w 2 - simplex56V 2), sq_nonneg (w 3 - simplex56V 3),
      sq_nonneg (w 4 - simplex56V 4),
      sq_nonneg (w 2 - simplex56V 2 + (w 3 - simplex56V 3) + (w 4 - simplex56V 4))]

/-- Helper: if `n : Fin 5 → ℝ` is nonzero and orthogonal to every row with `α i > 0`,
then any minimum-Euclidean-norm minimizer `w` satisfies `⟨w, n⟩ = 0`. -/
lemma minNorm_orth_of_perp {α : Fin 7 → ℝ} {w : Fin 5 → ℝ}
    (hw : IsMinNormMinimizer simplex56X simplex56Y α w)
    (n : Fin 5 → ℝ) (hn : 0 < dotProduct n n)
    (hperp : ∀ i : Fin 7, 0 < α i → dotProduct (simplex56X i) n = 0)
    (hα0 : ∀ i, 0 ≤ α i) :
    dotProduct w n = 0 := by
  let c : ℝ := dotProduct w n / dotProduct n n
  let v : Fin 5 → ℝ := fun j => w j - c * n j
  have hloss : (∑ i : Fin 7, α i * (dotProduct (simplex56X i) v - simplex56Y i) ^ 2) =
      ∑ i : Fin 7, α i * (dotProduct (simplex56X i) w - simplex56Y i) ^ 2 := by
    apply Finset.sum_congr rfl; intro i _
    rcases eq_or_lt_of_le (hα0 i) with h0 | hpos
    · rw [← h0]; simp
    · have hxv : dotProduct (simplex56X i) v = dotProduct (simplex56X i) w := by
        have hdot_sub : dotProduct (simplex56X i) v =
            dotProduct (simplex56X i) w - c * dotProduct (simplex56X i) n := by
          unfold dotProduct v
          simp only [mul_sub, Finset.sum_sub_distrib, Finset.mul_sum]
          congr 1; apply Finset.sum_congr rfl; intro j _; ring
        rw [hdot_sub, hperp i hpos, mul_zero, sub_zero]
      rw [hxv]
  have hnorm := hw.2 v hloss
  have hvv : dotProduct v v = dotProduct w w - (dotProduct w n) ^ 2 / dotProduct n n := by
    unfold dotProduct v
    have hexp : ∀ j : Fin 5, (w j - c * n j) * (w j - c * n j) =
        w j * w j - 2 * c * (w j * n j) + c ^ 2 * (n j * n j) := fun j => by ring
    simp_rw [hexp, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
    dsimp only [c]
    have hne : (∑ j : Fin 5, n j * n j) ≠ 0 := ne_of_gt hn
    unfold dotProduct
    field_simp [hne]
    ring
  rw [hvv] at hnorm
  have hsq0 : (dotProduct w n) ^ 2 / dotProduct n n ≤ 0 := by linarith
  have hsq_nn : 0 ≤ (dotProduct w n) ^ 2 / dotProduct n n :=
    div_nonneg (sq_nonneg _) (le_of_lt hn)
  have hsq_eq : (dotProduct w n) ^ 2 = 0 := by
    have : (dotProduct w n) ^ 2 / dotProduct n n = 0 := le_antisymm hsq0 hsq_nn
    exact (div_eq_zero_iff.1 this).resolve_right (ne_of_gt hn)
  exact sq_eq_zero_iff.1 hsq_eq

/-- Fast explicit 5-vector constructor avoiding `Matrix.vecCons` indexing at indices 3 and 4. -/
def mk5 (x0 x1 x2 x3 x4 : ℝ) : Fin 5 → ℝ
  | 0 => x0
  | 1 => x1
  | 2 => x2
  | 3 => x3
  | 4 => x4

lemma simplex56_block1_bound {α : Fin 7 → ℝ} {w : Fin 5 → ℝ}
    (hα0 : ∀ i, 0 ≤ α i)
    (hw : IsMinNormMinimizer simplex56X simplex56Y α w)
    (h012 : α 0 = 0 ∨ α 1 = 0 ∨ α 2 = 0) :
    let u := fun j : Fin 5 => w j - simplex56V j
    1 ≤ u 0 ^ 2 + u 1 ^ 2 + (u 0 + u 1) ^ 2 := by
  intro u
  have hsq6 : (1 / Real.sqrt 6) ^ 2 = 1 / 6 := by
    rw [div_pow, one_pow, Real.sq_sqrt (by norm_num)]
  have hw1 : ∀ v₀ : Fin 5 → ℝ,
      (∑ i : Fin 7, α i * (dotProduct (simplex56X i) w - simplex56Y i) ^ 2) ≤
      (∑ i : Fin 7, α i * (dotProduct (simplex56X i) v₀ - simplex56Y i) ^ 2) := hw.1
  have horth : ∀ (n : Fin 5 → ℝ), 0 < dotProduct n n →
      (∀ i : Fin 7, 0 < α i → dotProduct (simplex56X i) n = 0) →
      dotProduct w n = 0 := fun n hn hp => minNorm_orth_of_perp hw n hn hp hα0
  by_cases h0 : 0 < α 0 <;> by_cases h1 : 0 < α 1 <;> by_cases h2 : 0 < α 2
  · exfalso; rcases h012 with e | e | e <;> linarith
  · let a2 := 1 / Real.sqrt 6
    let v₀ := mk5 (10 + a2) (20 + a2) (w 2) (w 3) (w 4)
    have hle := hw1 v₀
    have h20 : α 2 = 0 := le_antisymm (not_lt.1 h2) (hα0 2)
    have hdiff :
        (∑ i : Fin 7, α i * (dotProduct (simplex56X i) w - simplex56Y i) ^ 2) -
        (∑ i : Fin 7, α i * (dotProduct (simplex56X i) v₀ - simplex56Y i) ^ 2) =
        α 0 * (u 0 - a2) ^ 2 + α 1 * (u 1 - a2) ^ 2 := by
      simp only [Fin.sum_univ_seven, h20, zero_mul, dot_s56_0, dot_s56_1, dot_s56_2,
        dot_s56_3, dot_s56_4, dot_s56_5, dot_s56_6, simplex56Y, simplex56V, simplex56Res,
        v₀, mk5, a2, u]
      ring
    have hz : u 0 - a2 = 0 ∧ u 1 - a2 = 0 := eq_zero_of_two_sq_le_zero h0 h1 (by linarith)
    have hu0 : u 0 = a2 := by linarith [hz.1]
    have hu1 : u 1 = a2 := by linarith [hz.2]
    rw [hu0, hu1]; dsimp only [a2]; nlinarith [hsq6]
  · let a2 := 1 / Real.sqrt 6
    let v₀ := mk5 (10 + a2) (20 - 2 * a2) (w 2) (w 3) (w 4)
    have hle := hw1 v₀
    have h10 : α 1 = 0 := le_antisymm (not_lt.1 h1) (hα0 1)
    have hdiff :
        (∑ i : Fin 7, α i * (dotProduct (simplex56X i) w - simplex56Y i) ^ 2) -
        (∑ i : Fin 7, α i * (dotProduct (simplex56X i) v₀ - simplex56Y i) ^ 2) =
        α 0 * (u 0 - a2) ^ 2 + α 2 * (-u 0 - u 1 - a2) ^ 2 := by
      simp only [Fin.sum_univ_seven, h10, zero_mul, dot_s56_0, dot_s56_1, dot_s56_2,
        dot_s56_3, dot_s56_4, dot_s56_5, dot_s56_6, simplex56Y, simplex56V, simplex56Res,
        v₀, mk5, a2, u]
      ring
    have hz : u 0 - a2 = 0 ∧ -u 0 - u 1 - a2 = 0 := eq_zero_of_two_sq_le_zero h0 h2 (by linarith)
    have hu0 : u 0 = a2 := by linarith [hz.1]
    have hu1 : u 1 = -2 * a2 := by linarith [hz.1, hz.2]
    rw [hu0, hu1]; dsimp only [a2]; nlinarith [hsq6]
  · have hp := horth (mk5 0 1 0 0 0) (by norm_num [dotProduct, Fin.sum_univ_five, mk5])
      (by intro i hi; fin_cases i <;> simp [dot_s56_0, dot_s56_1, dot_s56_2, dot_s56_3,
        dot_s56_4, dot_s56_5, dot_s56_6, mk5] at hi ⊢ <;> linarith)
    simp only [dotProduct, Fin.sum_univ_five, mk5, mul_zero, mul_one, add_zero, zero_add] at hp
    have hu1 : u 1 = -20 := by simp [u, simplex56V, hp]
    nlinarith [sq_nonneg (u 0), sq_nonneg (u 0 + u 1)]
  · let a2 := 1 / Real.sqrt 6
    let v₀ := mk5 (10 - 2 * a2) (20 + a2) (w 2) (w 3) (w 4)
    have hle := hw1 v₀
    have h00 : α 0 = 0 := le_antisymm (not_lt.1 h0) (hα0 0)
    have hdiff :
        (∑ i : Fin 7, α i * (dotProduct (simplex56X i) w - simplex56Y i) ^ 2) -
        (∑ i : Fin 7, α i * (dotProduct (simplex56X i) v₀ - simplex56Y i) ^ 2) =
        α 1 * (u 1 - a2) ^ 2 + α 2 * (-u 0 - u 1 - a2) ^ 2 := by
      simp only [Fin.sum_univ_seven, h00, zero_mul, dot_s56_0, dot_s56_1, dot_s56_2,
        dot_s56_3, dot_s56_4, dot_s56_5, dot_s56_6, simplex56Y, simplex56V, simplex56Res,
        v₀, mk5, a2, u]
      ring
    have hz : u 1 - a2 = 0 ∧ -u 0 - u 1 - a2 = 0 := eq_zero_of_two_sq_le_zero h1 h2 (by linarith)
    have hu1 : u 1 = a2 := by linarith [hz.1]
    have hu0 : u 0 = -2 * a2 := by linarith [hz.1, hz.2]
    rw [hu0, hu1]; dsimp only [a2]; nlinarith [hsq6]
  · have hp := horth (mk5 1 0 0 0 0) (by norm_num [dotProduct, Fin.sum_univ_five, mk5])
      (by intro i hi; fin_cases i <;> simp [dot_s56_0, dot_s56_1, dot_s56_2, dot_s56_3,
        dot_s56_4, dot_s56_5, dot_s56_6, mk5] at hi ⊢ <;> linarith)
    simp only [dotProduct, Fin.sum_univ_five, mk5, mul_zero, mul_one, add_zero] at hp
    have hu0 : u 0 = -10 := by simp [u, simplex56V, hp]
    nlinarith [sq_nonneg (u 1), sq_nonneg (u 0 + u 1)]
  · have hp := horth (mk5 1 (-1) 0 0 0) (by norm_num [dotProduct, Fin.sum_univ_five, mk5])
      (by intro i hi; fin_cases i <;> simp [dot_s56_0, dot_s56_1, dot_s56_2, dot_s56_3,
        dot_s56_4, dot_s56_5, dot_s56_6, mk5] at hi ⊢ <;> linarith)
    simp only [dotProduct, Fin.sum_univ_five, mk5, mul_zero, mul_one, mul_neg, add_zero] at hp
    have hd01 : u 0 - u 1 = 10 := by simp only [u, simplex56V]; linarith
    nlinarith [sq_nonneg (u 0 + u 1)]
  · have hp := horth (mk5 1 0 0 0 0) (by norm_num [dotProduct, Fin.sum_univ_five, mk5])
      (by intro i hi; fin_cases i <;> simp [dot_s56_0, dot_s56_1, dot_s56_2, dot_s56_3,
        dot_s56_4, dot_s56_5, dot_s56_6, mk5] at hi ⊢ <;> linarith)
    simp only [dotProduct, Fin.sum_univ_five, mk5, mul_zero, mul_one, add_zero] at hp
    have hu0 : u 0 = -10 := by simp [u, simplex56V, hp]
    nlinarith [sq_nonneg (u 1), sq_nonneg (u 0 + u 1)]

lemma simplex56_block2_pairs {α : Fin 7 → ℝ} {w : Fin 5 → ℝ}
    (hα0 : ∀ i, 0 ≤ α i)
    (hw : IsMinNormMinimizer simplex56X simplex56Y α w) :
    let u := fun j : Fin 5 => w j - simplex56V j
    ((α 3 = 0 ∧ α 6 = 0) ∨ (α 4 = 0 ∧ α 6 = 0) ∨ (α 5 = 0 ∧ α 6 = 0) ∨
     (α 3 = 0 ∧ α 4 = 0) ∨ (α 3 = 0 ∧ α 5 = 0) ∨ (α 4 = 0 ∧ α 5 = 0)) →
    1 ≤ u 2 ^ 2 + u 3 ^ 2 + u 4 ^ 2 + (u 2 + u 3 + u 4) ^ 2 := by
  intro u hpair
  have horth : ∀ (n : Fin 5 → ℝ), 0 < dotProduct n n →
      (∀ i : Fin 7, 0 < α i → dotProduct (simplex56X i) n = 0) →
      dotProduct w n = 0 := fun n hn hp => minNorm_orth_of_perp hw n hn hp hα0
  rcases hpair with ⟨h30, h60⟩ | ⟨h40, h60⟩ | ⟨h50, h60⟩ | ⟨h30, h40⟩ | ⟨h30, h50⟩ | ⟨h40, h50⟩
  · have hp := horth (mk5 0 0 1 0 0) (by norm_num [dotProduct, Fin.sum_univ_five, mk5])
      (by intro i hi; fin_cases i <;> simp [dot_s56_0, dot_s56_1, dot_s56_2, dot_s56_3,
        dot_s56_4, dot_s56_5, dot_s56_6, mk5] at hi ⊢ <;> linarith)
    simp only [dotProduct, Fin.sum_univ_five, mk5, mul_zero, mul_one, add_zero, zero_add] at hp
    have hu2 : u 2 = -10 := by simp [u, simplex56V, hp]
    nlinarith [sq_nonneg (u 3), sq_nonneg (u 4), sq_nonneg (u 2 + u 3 + u 4)]
  · have hp := horth (mk5 0 0 0 1 0) (by norm_num [dotProduct, Fin.sum_univ_five, mk5])
      (by intro i hi; fin_cases i <;> simp [dot_s56_0, dot_s56_1, dot_s56_2, dot_s56_3,
        dot_s56_4, dot_s56_5, dot_s56_6, mk5] at hi ⊢ <;> linarith)
    simp only [dotProduct, Fin.sum_univ_five, mk5, mul_zero, mul_one, add_zero, zero_add] at hp
    have hu3 : u 3 = -20 := by simp [u, simplex56V, hp]
    nlinarith [sq_nonneg (u 2), sq_nonneg (u 4), sq_nonneg (u 2 + u 3 + u 4)]
  · have hp := horth (mk5 0 0 0 0 1) (by norm_num [dotProduct, Fin.sum_univ_five, mk5])
      (by intro i hi; fin_cases i <;> simp [dot_s56_0, dot_s56_1, dot_s56_2, dot_s56_3,
        dot_s56_4, dot_s56_5, dot_s56_6, mk5] at hi ⊢ <;> linarith)
    simp only [dotProduct, Fin.sum_univ_five, mk5, mul_zero, mul_one, add_zero, zero_add] at hp
    have hu4 : u 4 = -30 := by simp [u, simplex56V, hp]
    nlinarith [sq_nonneg (u 2), sq_nonneg (u 3), sq_nonneg (u 2 + u 3 + u 4)]
  · have hp := horth (mk5 0 0 1 (-1) 0) (by norm_num [dotProduct, Fin.sum_univ_five, mk5])
      (by intro i hi; fin_cases i <;> simp [dot_s56_0, dot_s56_1, dot_s56_2, dot_s56_3,
        dot_s56_4, dot_s56_5, dot_s56_6, mk5] at hi ⊢ <;> linarith)
    simp only [dotProduct, Fin.sum_univ_five, mk5, mul_zero, mul_one, mul_neg, add_zero, zero_add] at hp
    have hd23 : u 2 - u 3 = 10 := by simp only [u, simplex56V]; linarith
    nlinarith [sq_nonneg (u 4), sq_nonneg (u 2 + u 3), sq_nonneg (u 2 + u 3 + u 4)]
  · have hp := horth (mk5 0 0 1 0 (-1)) (by norm_num [dotProduct, Fin.sum_univ_five, mk5])
      (by intro i hi; fin_cases i <;> simp [dot_s56_0, dot_s56_1, dot_s56_2, dot_s56_3,
        dot_s56_4, dot_s56_5, dot_s56_6, mk5] at hi ⊢ <;> linarith)
    simp only [dotProduct, Fin.sum_univ_five, mk5, mul_zero, mul_one, mul_neg, add_zero, zero_add] at hp
    have hd24 : u 2 - u 4 = 20 := by simp only [u, simplex56V]; linarith
    nlinarith [sq_nonneg (u 3), sq_nonneg (u 2 + u 4), sq_nonneg (u 2 + u 3 + u 4)]
  · have hp := horth (mk5 0 0 0 1 (-1)) (by norm_num [dotProduct, Fin.sum_univ_five, mk5])
      (by intro i hi; fin_cases i <;> simp [dot_s56_0, dot_s56_1, dot_s56_2, dot_s56_3,
        dot_s56_4, dot_s56_5, dot_s56_6, mk5] at hi ⊢ <;> linarith)
    simp only [dotProduct, Fin.sum_univ_five, mk5, mul_zero, mul_one, mul_neg, add_zero, zero_add] at hp
    have hd34 : u 3 - u 4 = 10 := by simp only [u, simplex56V]; linarith
    nlinarith [sq_nonneg (u 2), sq_nonneg (u 3 + u 4), sq_nonneg (u 2 + u 3 + u 4)]

lemma simplex56_block2_bound {α : Fin 7 → ℝ} {w : Fin 5 → ℝ}
    (hα0 : ∀ i, 0 ≤ α i)
    (hw : IsMinNormMinimizer simplex56X simplex56Y α w)
    (h3456 : α 3 = 0 ∨ α 4 = 0 ∨ α 5 = 0 ∨ α 6 = 0) :
    let u := fun j : Fin 5 => w j - simplex56V j
    1 ≤ u 2 ^ 2 + u 3 ^ 2 + u 4 ^ 2 + (u 2 + u 3 + u 4) ^ 2 := by
  intro u
  have hsq12 : (1 / Real.sqrt 12) ^ 2 = 1 / 12 := by
    rw [div_pow, one_pow, Real.sq_sqrt (by norm_num)]
  have hw1 : ∀ v₀ : Fin 5 → ℝ,
      (∑ i : Fin 7, α i * (dotProduct (simplex56X i) w - simplex56Y i) ^ 2) ≤
      (∑ i : Fin 7, α i * (dotProduct (simplex56X i) v₀ - simplex56Y i) ^ 2) := hw.1
  have hpairs := simplex56_block2_pairs hα0 hw
  by_cases h3 : 0 < α 3 <;> by_cases h4 : 0 < α 4 <;> by_cases h5 : 0 < α 5 <;> by_cases h6 : 0 < α 6
  · exfalso; rcases h3456 with e | e | e | e <;> linarith
  · let a3 := 1 / Real.sqrt 12
    let v₀ := mk5 (w 0) (w 1) (10 + a3) (20 + a3) (30 + a3)
    have hle := hw1 v₀
    have h60 : α 6 = 0 := le_antisymm (not_lt.1 h6) (hα0 6)
    have hdiff :
        (∑ i : Fin 7, α i * (dotProduct (simplex56X i) w - simplex56Y i) ^ 2) -
        (∑ i : Fin 7, α i * (dotProduct (simplex56X i) v₀ - simplex56Y i) ^ 2) =
        α 3 * (u 2 - a3) ^ 2 + α 4 * (u 3 - a3) ^ 2 + α 5 * (u 4 - a3) ^ 2 := by
      simp only [Fin.sum_univ_seven, h60, zero_mul, dot_s56_0, dot_s56_1, dot_s56_2,
        dot_s56_3, dot_s56_4, dot_s56_5, dot_s56_6, simplex56Y, simplex56V, simplex56Res,
        v₀, mk5, a3, u]
      ring
    have hz : u 2 - a3 = 0 ∧ u 3 - a3 = 0 ∧ u 4 - a3 = 0 :=
      eq_zero_of_three_sq_le_zero h3 h4 h5 (by linarith)
    have hu2 : u 2 = a3 := by linarith [hz.1]
    have hu3 : u 3 = a3 := by linarith [hz.2.1]
    have hu4 : u 4 = a3 := by linarith [hz.2.2]
    rw [hu2, hu3, hu4]; dsimp only [a3]; nlinarith [hsq12]
  · let a3 := 1 / Real.sqrt 12
    let v₀ := mk5 (w 0) (w 1) (10 + a3) (20 + a3) (30 - 3 * a3)
    have hle := hw1 v₀
    have h50 : α 5 = 0 := le_antisymm (not_lt.1 h5) (hα0 5)
    have hdiff :
        (∑ i : Fin 7, α i * (dotProduct (simplex56X i) w - simplex56Y i) ^ 2) -
        (∑ i : Fin 7, α i * (dotProduct (simplex56X i) v₀ - simplex56Y i) ^ 2) =
        α 3 * (u 2 - a3) ^ 2 + α 4 * (u 3 - a3) ^ 2 + α 6 * (-u 2 - u 3 - u 4 - a3) ^ 2 := by
      simp only [Fin.sum_univ_seven, h50, zero_mul, dot_s56_0, dot_s56_1, dot_s56_2,
        dot_s56_3, dot_s56_4, dot_s56_5, dot_s56_6, simplex56Y, simplex56V, simplex56Res,
        v₀, mk5, a3, u]
      ring
    have hz : u 2 - a3 = 0 ∧ u 3 - a3 = 0 ∧ -u 2 - u 3 - u 4 - a3 = 0 :=
      eq_zero_of_three_sq_le_zero h3 h4 h6 (by linarith)
    have hu2 : u 2 = a3 := by linarith [hz.1]
    have hu3 : u 3 = a3 := by linarith [hz.2.1]
    have hu4 : u 4 = -3 * a3 := by linarith [hz.1, hz.2.1, hz.2.2]
    rw [hu2, hu3, hu4]; dsimp only [a3]; nlinarith [hsq12]
  · exact hpairs (Or.inr (Or.inr (Or.inl ⟨le_antisymm (not_lt.1 h5) (hα0 5), le_antisymm (not_lt.1 h6) (hα0 6)⟩)))
  · let a3 := 1 / Real.sqrt 12
    let v₀ := mk5 (w 0) (w 1) (10 + a3) (20 - 3 * a3) (30 + a3)
    have hle := hw1 v₀
    have h40 : α 4 = 0 := le_antisymm (not_lt.1 h4) (hα0 4)
    have hdiff :
        (∑ i : Fin 7, α i * (dotProduct (simplex56X i) w - simplex56Y i) ^ 2) -
        (∑ i : Fin 7, α i * (dotProduct (simplex56X i) v₀ - simplex56Y i) ^ 2) =
        α 3 * (u 2 - a3) ^ 2 + α 5 * (u 4 - a3) ^ 2 + α 6 * (-u 2 - u 3 - u 4 - a3) ^ 2 := by
      simp only [Fin.sum_univ_seven, h40, zero_mul, dot_s56_0, dot_s56_1, dot_s56_2,
        dot_s56_3, dot_s56_4, dot_s56_5, dot_s56_6, simplex56Y, simplex56V, simplex56Res,
        v₀, mk5, a3, u]
      ring
    have hz : u 2 - a3 = 0 ∧ u 4 - a3 = 0 ∧ -u 2 - u 3 - u 4 - a3 = 0 :=
      eq_zero_of_three_sq_le_zero h3 h5 h6 (by linarith)
    have hu2 : u 2 = a3 := by linarith [hz.1]
    have hu4 : u 4 = a3 := by linarith [hz.2.1]
    have hu3 : u 3 = -3 * a3 := by linarith [hz.1, hz.2.1, hz.2.2]
    rw [hu2, hu3, hu4]; dsimp only [a3]; nlinarith [hsq12]
  · exact hpairs (Or.inr (Or.inl ⟨le_antisymm (not_lt.1 h4) (hα0 4), le_antisymm (not_lt.1 h6) (hα0 6)⟩))
  · exact hpairs (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨le_antisymm (not_lt.1 h4) (hα0 4), le_antisymm (not_lt.1 h5) (hα0 5)⟩)))))
  · exact hpairs (Or.inr (Or.inl ⟨le_antisymm (not_lt.1 h4) (hα0 4), le_antisymm (not_lt.1 h6) (hα0 6)⟩))
  · let a3 := 1 / Real.sqrt 12
    let v₀ := mk5 (w 0) (w 1) (10 - 3 * a3) (20 + a3) (30 + a3)
    have hle := hw1 v₀
    have h30 : α 3 = 0 := le_antisymm (not_lt.1 h3) (hα0 3)
    have hdiff :
        (∑ i : Fin 7, α i * (dotProduct (simplex56X i) w - simplex56Y i) ^ 2) -
        (∑ i : Fin 7, α i * (dotProduct (simplex56X i) v₀ - simplex56Y i) ^ 2) =
        α 4 * (u 3 - a3) ^ 2 + α 5 * (u 4 - a3) ^ 2 + α 6 * (-u 2 - u 3 - u 4 - a3) ^ 2 := by
      simp only [Fin.sum_univ_seven, h30, zero_mul, dot_s56_0, dot_s56_1, dot_s56_2,
        dot_s56_3, dot_s56_4, dot_s56_5, dot_s56_6, simplex56Y, simplex56V, simplex56Res,
        v₀, mk5, a3, u]
      ring
    have hz : u 3 - a3 = 0 ∧ u 4 - a3 = 0 ∧ -u 2 - u 3 - u 4 - a3 = 0 :=
      eq_zero_of_three_sq_le_zero h4 h5 h6 (by linarith)
    have hu3 : u 3 = a3 := by linarith [hz.1]
    have hu4 : u 4 = a3 := by linarith [hz.2.1]
    have hu2 : u 2 = -3 * a3 := by linarith [hz.1, hz.2.1, hz.2.2]
    rw [hu2, hu3, hu4]; dsimp only [a3]; nlinarith [hsq12]
  · exact hpairs (Or.inl ⟨le_antisymm (not_lt.1 h3) (hα0 3), le_antisymm (not_lt.1 h6) (hα0 6)⟩)
  · exact hpairs (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨le_antisymm (not_lt.1 h3) (hα0 3), le_antisymm (not_lt.1 h5) (hα0 5)⟩)))))
  · exact hpairs (Or.inl ⟨le_antisymm (not_lt.1 h3) (hα0 3), le_antisymm (not_lt.1 h6) (hα0 6)⟩)
  · exact hpairs (Or.inr (Or.inr (Or.inr (Or.inl ⟨le_antisymm (not_lt.1 h3) (hα0 3), le_antisymm (not_lt.1 h4) (hα0 4)⟩))))
  · exact hpairs (Or.inl ⟨le_antisymm (not_lt.1 h3) (hα0 3), le_antisymm (not_lt.1 h6) (hα0 6)⟩)
  · exact hpairs (Or.inr (Or.inr (Or.inr (Or.inl ⟨le_antisymm (not_lt.1 h3) (hα0 3), le_antisymm (not_lt.1 h4) (hα0 4)⟩))))
  · exact hpairs (Or.inl ⟨le_antisymm (not_lt.1 h3) (hα0 3), le_antisymm (not_lt.1 h6) (hα0 6)⟩)

/-- **Proposition 6 (Unrestricted lower bound `Fw(5, 6) ≥ 11 / 5`)**:
For EVERY subset `S ⊆ Fin 7` of size `≤ 6`, EVERY nonnegative weight `α` supported on `S`,
and EVERY minimum-Euclidean-norm minimizer `w` (full-rank OR rank-deficient),
`simplex56Loss w / (5 / 6) ≥ 11 / 5`. -/
theorem simplex56_lower_bound
    (S : Finset (Fin 7)) (hScard : S.card ≤ 6)
    (α : Fin 7 → ℝ) (hα0 : ∀ i, 0 ≤ α i) (hαS : ∀ i, i ∉ S → α i = 0)
    (w : Fin 5 → ℝ) (hw : IsMinNormMinimizer simplex56X simplex56Y α w) :
    11 / 5 ≤ simplex56Loss w / (5 / 6) := by
  have hgoal : 11 / 6 ≤ simplex56Loss w → 11 / 5 ≤ simplex56Loss w / (5 / 6) := by
    intro h; linarith
  apply hgoal
  rw [simplex56_loss_decomp]
  set u : Fin 5 → ℝ := fun j => w j - simplex56V j
  have hB1_nn : 0 ≤ u 0 ^ 2 + u 1 ^ 2 + (u 0 + u 1) ^ 2 := by positivity
  have hB2_nn : 0 ≤ u 2 ^ 2 + u 3 ^ 2 + u 4 ^ 2 + (u 2 + u 3 + u 4) ^ 2 := by positivity
  have hex0 : ∃ i : Fin 7, α i = 0 := by
    by_contra hall
    push_neg at hall
    have hsub : Finset.univ ⊆ S := by
      intro i _; by_contra hnot; exact hall i (hαS i hnot)
    have h7 := Finset.card_le_card hsub
    simp at h7
    omega
  by_cases h012 : α 0 = 0 ∨ α 1 = 0 ∨ α 2 = 0
  · linarith [simplex56_block1_bound hα0 hw h012]
  · have h3456 : α 3 = 0 ∨ α 4 = 0 ∨ α 5 = 0 ∨ α 6 = 0 := by
      rcases hex0 with ⟨i, hi⟩
      fin_cases i
      · exact absurd (Or.inl hi) h012
      · exact absurd (Or.inr (Or.inl hi)) h012
      · exact absurd (Or.inr (Or.inr hi)) h012
      · exact Or.inl hi
      · exact Or.inr (Or.inl hi)
      · exact Or.inr (Or.inr (Or.inl hi))
      · exact Or.inr (Or.inr (Or.inr hi))
    linarith [simplex56_block2_bound hα0 hw h3456]

/-- Proposition 6 upper-bound factor from one closest-hyperplane step plus `Fw(4, 5) = 2`. -/
theorem fw_5_6_upper_factor : ((1 : ℝ) + 1 / 5) * 2 = 12 / 5 := by norm_num

end WeightedDataScalar
