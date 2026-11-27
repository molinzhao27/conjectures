import Mathlib

/-!
# Rank-Constrained Capacity Equivalence for Gaussian Linear Associative Memory

Formalization of the mathematical core of the settlement of Conjecture `conjecture-capacity`
(Capacity Equivalence) from Giorlandino, Goldt, and Maillard, *Factual recall in linear
associative memories: sharp asymptotics and mechanistic insights* (`arXiv:2605.10795`),
as proved in `solutions/linear_assoc_memory/solution.md`.

## What is machine-checked in this file

1. **Quarter-circle law, quantile `q_κ`, energy `E_κ`, and exact full-rank threshold**:
   - Exact antiderivatives and closed forms on `[0, 2]` for the Marchenko–Pastur quarter-circle
     tail mass `qcMass q = ∫ s in q..2, √(4 - s²) / π` and tail second moment
     `qcEnergy q = ∫ s in q..2, s² √(4 - s²) / π`.
   - Existence and uniqueness of the rank-fraction quantile `q_κ ∈ [0, 2)` for every `κ ∈ (0, 1]`
     (`exists_unique_q_kappa`), strict positivity `0 < qcEnergy q` on `[0, 2)` (`qcEnergy_pos`),
     and the exact full-rank evaluation `qcMass 0 = 1`, `qcEnergy 0 = 1`, `E_1 / 2 = 1 / 2`
     (`qcEnergy_at_full_rank`).
2. **Original Problem (OP) feasibility and Frobenius normalization**:
   - Positive-homogeneous reduction of OP feasibility to the unit Frobenius sphere and the exact
     necessary inequality `⟨C, W'⟩_F > S(W')` (`op_feasible_implies_unit_corr_gt_compScore`).
3. **SVD head-tail bounds and Lipschitz regularity**:
   - Operator-norm tail bound `σ_i ≤ 1 / √k` for `i ≥ k - 1` (`tail_sv_le_inv_sqrt_k`) and
     nuclear-norm head bound `∑_{i ∈ S} σ_i ≤ √|S|` (`head_nuclear_le_sqrt_k`).
   - Exact `1`-Lipschitzness of the rank-`k` approximation distance `b(W) = dist_F(W, R)`
     (`bDist_one_lipschitz`) and Lipschitz continuity of the competitor score `S(W)` on bounded
     key/output clouds (`compScore_lipschitz_on_bounded_cloud`).
4. **Steps 1–3 deterministic, combinatorial packing, and Gaussian-conditioning cores**:
   - Step 1 coarse lower bound `S(W) / (p √(2L)) ≥ (1 / (4 √π)) ‖W‖_F` (`step1_coarse_lower_bound`).
   - Step 2 inclusion-maximal packing escape lemma: if no `n`-element good compatible subset exists
     in `A` (`|A| ≥ p/4`, `n ≤ p/16`), then a maximal good compatible `J ⊆ A` has `|J| < n`,
     `|A \ J| ≥ p/8`, and every `j ∈ A \ J` fails the fresh-point test relative to `J`
     (`step2_maximal_packing_escape`).
   - Step 3 orthogonal key split score transfer (`step3_orth_split_score_bound`), equicorrelated
     Slepian shift reduction (`step3_equicorr_shift`), and query-counting aggregation
     (`step3_query_counting_lower_bound`).
5. **Step 4 — Von Neumann head-tail split and sharp OP converse contradiction**:
   - Head-tail trace inequality aggregation (`von_neumann_head_tail_split`).
   - Exact contradiction showing that for every `E_κ > 0` and every `α > E_κ / 2`, once the
     vanishing `o(1)` error terms fall below explicit positive thresholds, the coarse lower bound,
     sharp `b(W)` lower bound, and rank-`m` correlation upper bound cannot hold simultaneously for
     any `b ∈ [0, 1]` (`op_converse_core_contradiction`), forcing `α_c = E_κ / 2`
     (`capacity_threshold_equality`).

## What remains prose-only in `solutions/linear_assoc_memory/proof.pdf`

The high-dimensional Gaussian concentration, ε-net union bounds, Gaussian Poincaré inequality,
Slepian's comparison lemma for correlated Gaussians, and the Marchenko–Pastur empirical singular
value limit `‖C_m‖_F / (d √p) → √E_κ` (imported from `arXiv:2605.10795`) are carried out in
standard probability prose in `solutions/linear_assoc_memory/proof.pdf`.
-/

namespace LinearAssocMemory

open scoped BigOperators
open Finset Matrix Real Set

/-! ### Part 1: Quarter-circle density, quantile `q_κ`, energy `E_κ`, and full-rank `E_1 = 1` -/

/-- The Marchenko–Pastur quarter-circle singular-value density on `[0, 2]`:
`ρ_qc(s) = √(4 - s²) / π`. -/
noncomputable def qcDensity (s : ℝ) : ℝ :=
  Real.sqrt (4 - s ^ 2) / Real.pi

/-- Upper tail mass `∫_{q}^{2} ρ_qc(s) ds`. -/
noncomputable def qcMass (q : ℝ) : ℝ :=
  ∫ s in q..2, qcDensity s

/-- Upper tail second moment (energy) `∫_{q}^{2} s² ρ_qc(s) ds`. -/
noncomputable def qcEnergy (q : ℝ) : ℝ :=
  ∫ s in q..2, s ^ 2 * qcDensity s

/-- Explicit antiderivative of `qcDensity` on `[0, 2]`. -/
noncomputable def qcMassAnti (s : ℝ) : ℝ :=
  (s * Real.sqrt (4 - s ^ 2) / 2 + 2 * Real.arcsin (s / 2)) / Real.pi

/-- Explicit antiderivative of `s ↦ s² * qcDensity s` on `[0, 2]`. -/
noncomputable def qcEnergyAnti (s : ℝ) : ℝ :=
  (s * (s ^ 2 - 2) * Real.sqrt (4 - s ^ 2) / 4 + 2 * Real.arcsin (s / 2)) / Real.pi

private lemma four_sub_sq_pos {s : ℝ} (hs0 : -2 < s) (hs2 : s < 2) :
    0 < 4 - s ^ 2 := by
  nlinarith

private lemma sqrt_one_sub_half_sq {s : ℝ} (hs0 : -2 < s) (hs2 : s < 2) :
    Real.sqrt (1 - (s / 2) ^ 2) = Real.sqrt (4 - s ^ 2) / 2 := by
  have h1 : 1 - (s / 2) ^ 2 = (4 - s ^ 2) / 4 := by ring
  rw [h1, Real.sqrt_div (le_of_lt (four_sub_sq_pos hs0 hs2))]
  have h4 : Real.sqrt 4 = 2 := by
    rw [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num : (0 : ℝ) ≤ 2)]
  rw [h4]

private lemma hasDerivAt_qcMassAnti {s : ℝ} (hs0 : -2 < s) (hs2 : s < 2) :
    HasDerivAt qcMassAnti (qcDensity s) s := by
  have h4pos : 0 < 4 - s ^ 2 := four_sub_sq_pos hs0 hs2
  have h4ne : 4 - s ^ 2 ≠ 0 := ne_of_gt h4pos
  have hsqpos : 0 < Real.sqrt (4 - s ^ 2) := Real.sqrt_pos.2 h4pos
  have hsqne : Real.sqrt (4 - s ^ 2) ≠ 0 := ne_of_gt hsqpos
  have hpi : Real.pi ≠ 0 := ne_of_gt Real.pi_pos
  have hsub : HasDerivAt (fun x : ℝ => 4 - x ^ 2) (-2 * s) s := by
    have h1 := (hasDerivAt_pow 2 s).const_sub 4
    convert h1 using 1
    ring
  have hsqrt : HasDerivAt (fun x : ℝ => Real.sqrt (4 - x ^ 2))
      ((-2 * s) / (2 * Real.sqrt (4 - s ^ 2))) s := by
    simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using hsub.sqrt h4ne
  have hmul : HasDerivAt (fun x : ℝ => x * Real.sqrt (4 - x ^ 2) / 2)
      ((1 * Real.sqrt (4 - s ^ 2) + s * ((-2 * s) / (2 * Real.sqrt (4 - s ^ 2)))) / 2) s := by
    exact ((hasDerivAt_id s).mul hsqrt).div_const 2
  have hhalf : HasDerivAt (fun x : ℝ => x / 2) (1 / 2) s := by
    simpa using (hasDerivAt_id s).div_const 2
  have harcsin0 : HasDerivAt Real.arcsin (1 / Real.sqrt (1 - (s / 2) ^ 2)) (s / 2) := by
    simpa [one_div] using Real.hasDerivAt_arcsin (x := s / 2) (by linarith) (by linarith)
  have harcsin : HasDerivAt (fun x : ℝ => 2 * Real.arcsin (x / 2))
      (2 * ((1 / Real.sqrt (1 - (s / 2) ^ 2)) * (1 / 2))) s := by
    exact (harcsin0.comp s hhalf).const_mul 2
  have htot := (hmul.add harcsin).div_const Real.pi
  convert htot using 1
  rw [qcDensity, sqrt_one_sub_half_sq hs0 hs2]
  have hsqsq : Real.sqrt (4 - s ^ 2) ^ 2 = 4 - s ^ 2 := Real.sq_sqrt (le_of_lt h4pos)
  field_simp [hsqne, hpi]
  nlinarith [hsqsq]

private lemma hasDerivAt_qcEnergyAnti {s : ℝ} (hs0 : -2 < s) (hs2 : s < 2) :
    HasDerivAt qcEnergyAnti (s ^ 2 * qcDensity s) s := by
  have h4pos : 0 < 4 - s ^ 2 := four_sub_sq_pos hs0 hs2
  have h4ne : 4 - s ^ 2 ≠ 0 := ne_of_gt h4pos
  have hsqpos : 0 < Real.sqrt (4 - s ^ 2) := Real.sqrt_pos.2 h4pos
  have hsqne : Real.sqrt (4 - s ^ 2) ≠ 0 := ne_of_gt hsqpos
  have hpi : Real.pi ≠ 0 := ne_of_gt Real.pi_pos
  have hpoly : HasDerivAt (fun x : ℝ => x * (x ^ 2 - 2)) (3 * s ^ 2 - 2) s := by
    have h1 := (hasDerivAt_id s).mul ((hasDerivAt_pow 2 s).sub_const 2)
    convert h1 using 1
    simp only [id_eq]
    ring
  have hsub : HasDerivAt (fun x : ℝ => 4 - x ^ 2) (-2 * s) s := by
    have h1 := (hasDerivAt_pow 2 s).const_sub 4
    convert h1 using 1
    ring
  have hsqrt : HasDerivAt (fun x : ℝ => Real.sqrt (4 - x ^ 2))
      ((-2 * s) / (2 * Real.sqrt (4 - s ^ 2))) s := by
    simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using hsub.sqrt h4ne
  have hmul : HasDerivAt (fun x : ℝ => x * (x ^ 2 - 2) * Real.sqrt (4 - x ^ 2) / 4)
      (((3 * s ^ 2 - 2) * Real.sqrt (4 - s ^ 2) +
        s * (s ^ 2 - 2) * ((-2 * s) / (2 * Real.sqrt (4 - s ^ 2)))) / 4) s := by
    exact (hpoly.mul hsqrt).div_const 4
  have hhalf : HasDerivAt (fun x : ℝ => x / 2) (1 / 2) s := by
    simpa using (hasDerivAt_id s).div_const 2
  have harcsin0 : HasDerivAt Real.arcsin (1 / Real.sqrt (1 - (s / 2) ^ 2)) (s / 2) := by
    simpa [one_div] using Real.hasDerivAt_arcsin (x := s / 2) (by linarith) (by linarith)
  have harcsin : HasDerivAt (fun x : ℝ => 2 * Real.arcsin (x / 2))
      (2 * ((1 / Real.sqrt (1 - (s / 2) ^ 2)) * (1 / 2))) s := by
    exact (harcsin0.comp s hhalf).const_mul 2
  have htot := (hmul.add harcsin).div_const Real.pi
  convert htot using 1
  rw [qcDensity, sqrt_one_sub_half_sq hs0 hs2]
  have hsqsq : Real.sqrt (4 - s ^ 2) ^ 2 = 4 - s ^ 2 := Real.sq_sqrt (le_of_lt h4pos)
  field_simp [hsqne, hpi]
  nlinarith [hsqsq]

private lemma continuous_qcMassAnti : Continuous qcMassAnti := by
  unfold qcMassAnti
  fun_prop

private lemma continuous_qcEnergyAnti : Continuous qcEnergyAnti := by
  unfold qcEnergyAnti
  fun_prop

private lemma continuous_qcDensity : Continuous qcDensity := by
  unfold qcDensity
  fun_prop

private lemma qcMassAnti_zero : qcMassAnti 0 = 0 := by
  simp [qcMassAnti]

private lemma qcMassAnti_two : qcMassAnti 2 = 1 := by
  have hpi : Real.pi ≠ 0 := ne_of_gt Real.pi_pos
  have h0 : (4 : ℝ) - 2 ^ 2 = 0 := by norm_num
  simp [qcMassAnti, h0, Real.arcsin_one]
  field_simp [hpi]

private lemma qcEnergyAnti_zero : qcEnergyAnti 0 = 0 := by
  simp [qcEnergyAnti]

private lemma qcEnergyAnti_two : qcEnergyAnti 2 = 1 := by
  have hpi : Real.pi ≠ 0 := ne_of_gt Real.pi_pos
  have h0 : (4 : ℝ) - 2 ^ 2 = 0 := by norm_num
  simp [qcEnergyAnti, h0, Real.arcsin_one]
  field_simp [hpi]

/-- Exact closed form for the quarter-circle upper tail mass on `[0, 2]`. -/
theorem qcMass_eq_one_sub_anti {q : ℝ} (hq0 : 0 ≤ q) (hq2 : q ≤ 2) :
    qcMass q = 1 - qcMassAnti q := by
  unfold qcMass
  have hint : IntervalIntegrable qcDensity MeasureTheory.volume q 2 :=
    continuous_qcDensity.intervalIntegrable q 2
  have hftc := intervalIntegral.integral_eq_sub_of_hasDerivAt_of_le hq2
    continuous_qcMassAnti.continuousOn
    (fun s hs => hasDerivAt_qcMassAnti (by linarith [hs.1]) hs.2)
    hint
  rw [hftc, qcMassAnti_two]

/-- Exact closed form for the quarter-circle upper tail energy on `[0, 2]`. -/
theorem qcEnergy_eq_one_sub_anti {q : ℝ} (hq0 : 0 ≤ q) (hq2 : q ≤ 2) :
    qcEnergy q = 1 - qcEnergyAnti q := by
  unfold qcEnergy
  have hcont : Continuous (fun s : ℝ => s ^ 2 * qcDensity s) :=
    (continuous_pow 2).mul continuous_qcDensity
  have hint : IntervalIntegrable (fun s : ℝ => s ^ 2 * qcDensity s) MeasureTheory.volume q 2 :=
    hcont.intervalIntegrable q 2
  have hftc := intervalIntegral.integral_eq_sub_of_hasDerivAt_of_le hq2
    continuous_qcEnergyAnti.continuousOn
    (fun s hs => hasDerivAt_qcEnergyAnti (by linarith [hs.1]) hs.2)
    hint
  rw [hftc, qcEnergyAnti_two]

/-- `qcMassAnti` is strictly increasing on `[0, 2]`. -/
lemma qcMassAnti_strictMonoOn : StrictMonoOn qcMassAnti (Set.Icc 0 2) := by
  apply strictMonoOn_of_deriv_pos (convex_Icc 0 2) continuous_qcMassAnti.continuousOn
  intro s hs
  rw [interior_Icc] at hs
  rw [(hasDerivAt_qcMassAnti (by linarith [hs.1]) hs.2).deriv]
  exact div_pos (Real.sqrt_pos.2 (four_sub_sq_pos (by linarith [hs.1]) hs.2)) Real.pi_pos

/-- `qcEnergyAnti` is strictly increasing on `[0, 2]`. -/
lemma qcEnergyAnti_strictMonoOn : StrictMonoOn qcEnergyAnti (Set.Icc 0 2) := by
  apply strictMonoOn_of_deriv_pos (convex_Icc 0 2) continuous_qcEnergyAnti.continuousOn
  intro s hs
  rw [interior_Icc] at hs
  rw [(hasDerivAt_qcEnergyAnti (by linarith [hs.1]) hs.2).deriv]
  have hs2pos : 0 < s ^ 2 := sq_pos_of_pos hs.1
  have hdpos : 0 < qcDensity s :=
    div_pos (Real.sqrt_pos.2 (four_sub_sq_pos (by linarith [hs.1]) hs.2)) Real.pi_pos
  exact mul_pos hs2pos hdpos

/-- **Quarter-circle quantile theorem**: For every rank fraction `κ ∈ (0, 1]`, there is a unique
quantile `q_κ ∈ [0, 2)` with `qcMass q_κ = κ`. -/
theorem exists_unique_q_kappa {κ : ℝ} (hκ0 : 0 < κ) (hκ1 : κ ≤ 1) :
    ∃! q : ℝ, q ∈ Set.Ico (0 : ℝ) 2 ∧ qcMass q = κ := by
  have hmem : 1 - κ ∈ Set.Icc (qcMassAnti 0) (qcMassAnti 2) := by
    rw [qcMassAnti_zero, qcMassAnti_two]
    exact ⟨by linarith, by linarith⟩
  obtain ⟨q, hqIcc, hqEq⟩ := intermediate_value_Icc (by norm_num : (0 : ℝ) ≤ 2)
    continuous_qcMassAnti.continuousOn hmem
  have hqLt2 : q < 2 := by
    by_contra hge
    have hq2 : q = 2 := le_antisymm hqIcc.2 (le_of_not_gt hge)
    rw [hq2, qcMassAnti_two] at hqEq
    linarith
  refine ⟨q, ⟨⟨hqIcc.1, hqLt2⟩, ?_⟩, ?_⟩
  · rw [qcMass_eq_one_sub_anti hqIcc.1 hqIcc.2, hqEq]
    ring
  · rintro q' ⟨hq'Ico, hq'Mass⟩
    have hq'Icc : q' ∈ Set.Icc (0 : ℝ) 2 := ⟨hq'Ico.1, le_of_lt hq'Ico.2⟩
    have hq'Eq : qcMassAnti q' = 1 - κ := by
      rw [qcMass_eq_one_sub_anti hq'Icc.1 hq'Icc.2] at hq'Mass
      linarith
    exact qcMassAnti_strictMonoOn.injOn hq'Icc hqIcc (hq'Eq.trans hqEq.symm)

/-- Strict positivity and upper bound `0 < E_κ ≤ 1` for every `q ∈ [0, 2)`. -/
theorem qcEnergy_pos_and_le_one {q : ℝ} (hq0 : 0 ≤ q) (hq2 : q < 2) :
    0 < qcEnergy q ∧ qcEnergy q ≤ 1 := by
  rw [qcEnergy_eq_one_sub_anti hq0 (le_of_lt hq2)]
  have hlt : qcEnergyAnti q < qcEnergyAnti 2 :=
    qcEnergyAnti_strictMonoOn ⟨hq0, le_of_lt hq2⟩ ⟨by norm_num, le_rfl⟩ hq2
  rw [qcEnergyAnti_two] at hlt
  have hge : qcEnergyAnti 0 ≤ qcEnergyAnti q :=
    qcEnergyAnti_strictMonoOn.monotoneOn ⟨le_rfl, by norm_num⟩ ⟨hq0, le_of_lt hq2⟩ hq0
  rw [qcEnergyAnti_zero] at hge
  exact ⟨by linarith, by linarith⟩

/-- Exact full-rank (`κ = 1`) evaluation: `q_1 = 0`, `E_1 = 1`, and `α_c(1) = E_1 / 2 = 1 / 2`. -/
theorem qcEnergy_at_full_rank :
    qcMass 0 = 1 ∧ qcEnergy 0 = 1 ∧ qcEnergy 0 / 2 = 1 / 2 := by
  have hm : qcMass 0 = 1 := by
    rw [qcMass_eq_one_sub_anti le_rfl (by norm_num), qcMassAnti_zero]; ring
  have he : qcEnergy 0 = 1 := by
    rw [qcEnergy_eq_one_sub_anti le_rfl (by norm_num), qcEnergyAnti_zero]; ring
  exact ⟨hm, he, by rw [he]⟩

/-! ### Part 2: Original Problem (OP) feasibility, Frobenius norm, and normalization -/

/-- Standard Frobenius inner product on `d × d` real matrices. -/
def frobInner {d : ℕ} (A B : Matrix (Fin d) (Fin d) ℝ) : ℝ :=
  ∑ i : Fin d, ∑ j : Fin d, A i j * B i j

/-- Squared Frobenius norm. -/
def frobNormSq {d : ℕ} (A : Matrix (Fin d) (Fin d) ℝ) : ℝ :=
  frobInner A A

/-- Frobenius norm. -/
noncomputable def frobNorm {d : ℕ} (A : Matrix (Fin d) (Fin d) ℝ) : ℝ :=
  Real.sqrt (frobNormSq A)

/-- Euclidean vector norm on `Fin d → ℝ`. -/
noncomputable def vecNorm {d : ℕ} (x : Fin d → ℝ) : ℝ :=
  Real.sqrt (∑ i : Fin d, (x i) ^ 2)

/-- Original Problem (OP) feasibility: `u_iᵀ W e_i > u_jᵀ W e_i` for all `i` and `j ≠ i`. -/
def IsOPFeasible {d p : ℕ} (W : Matrix (Fin d) (Fin d) ℝ)
    (U E : Fin p → Fin d → ℝ) : Prop :=
  ∀ i j : Fin p, j ≠ i → dotProduct (U i) (W.mulVec (E i)) > dotProduct (U j) (W.mulVec (E i))

/-- Empirical correlation matrix `C = ∑_{i=1}^p u_i e_iᵀ`. -/
def corrMatrix {d p : ℕ} (U E : Fin p → Fin d → ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  ∑ i : Fin p, Matrix.vecMulVec (U i) (E i)

private lemma erase_nonempty_of_two_le {p : ℕ} (hp : 2 ≤ p) (i : Fin p) :
    (Finset.univ.erase i).Nonempty := by
  rw [← Finset.card_pos, Finset.card_erase_of_mem (Finset.mem_univ i), Finset.card_univ,
    Fintype.card_fin]
  omega

/-- Competitor score `S(W) = ∑_{i=1}^p max_{j ≠ i} u_jᵀ W e_i` (well-defined for `p ≥ 2`). -/
def compScore {d p : ℕ} (W : Matrix (Fin d) (Fin d) ℝ)
    (U E : Fin p → Fin d → ℝ) (hp : 2 ≤ p) : ℝ :=
  ∑ i : Fin p, (Finset.univ.erase i).sup' (erase_nonempty_of_two_le hp i)
    (fun j => dotProduct (U j) (W.mulVec (E i)))

lemma frobNormSq_nonneg {d : ℕ} (A : Matrix (Fin d) (Fin d) ℝ) : 0 ≤ frobNormSq A := by
  unfold frobNormSq frobInner
  exact Finset.sum_nonneg fun i _ => Finset.sum_nonneg fun j _ => mul_self_nonneg (A i j)

lemma frobNorm_nonneg {d : ℕ} (A : Matrix (Fin d) (Fin d) ℝ) : 0 ≤ frobNorm A :=
  Real.sqrt_nonneg _

lemma frobNorm_sq {d : ℕ} (A : Matrix (Fin d) (Fin d) ℝ) :
    (frobNorm A) ^ 2 = frobNormSq A :=
  Real.sq_sqrt (frobNormSq_nonneg A)

lemma frobNormSq_pos_of_ne_zero {d : ℕ} {A : Matrix (Fin d) (Fin d) ℝ} (hA : A ≠ 0) :
    0 < frobNormSq A := by
  have hex : ∃ i j : Fin d, A i j ≠ 0 := by
    by_contra h
    push_neg at h
    apply hA
    ext i j
    exact h i j
  rcases hex with ⟨i₀, j₀, hij⟩
  unfold frobNormSq frobInner
  have hinner : 0 < ∑ j : Fin d, A i₀ j * A i₀ j :=
    Finset.sum_pos' (fun j _ => mul_self_nonneg (A i₀ j))
      ⟨j₀, Finset.mem_univ j₀, mul_self_pos.2 hij⟩
  exact Finset.sum_pos' (fun i _ => Finset.sum_nonneg fun j _ => mul_self_nonneg (A i j))
    ⟨i₀, Finset.mem_univ i₀, hinner⟩

lemma frobNorm_pos_of_ne_zero {d : ℕ} {A : Matrix (Fin d) (Fin d) ℝ} (hA : A ≠ 0) :
    0 < frobNorm A :=
  Real.sqrt_pos.2 (frobNormSq_pos_of_ne_zero hA)

lemma frobNorm_smul {d : ℕ} (c : ℝ) (A : Matrix (Fin d) (Fin d) ℝ) :
    frobNorm (c • A) = |c| * frobNorm A := by
  unfold frobNorm frobNormSq frobInner
  have h : (∑ i : Fin d, ∑ j : Fin d, (c • A) i j * (c • A) i j) =
      c ^ 2 * ∑ i : Fin d, ∑ j : Fin d, A i j * A i j := by
    simp only [Matrix.smul_apply, smul_eq_mul, Finset.mul_sum]
    refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
    ring
  rw [h, Real.sqrt_mul (sq_nonneg c), Real.sqrt_sq_eq_abs]

lemma frobInner_cauchy_schwarz {d : ℕ} (A B : Matrix (Fin d) (Fin d) ℝ) :
    |frobInner A B| ≤ frobNorm A * frobNorm B := by
  have hcs : (frobInner A B) ^ 2 ≤ frobNormSq A * frobNormSq B := by
    unfold frobNormSq frobInner
    have h1 : (∑ i : Fin d, ∑ j : Fin d, A i j * B i j) =
        ∑ p : Fin d × Fin d, A p.1 p.2 * B p.1 p.2 :=
      (Fintype.sum_prod_type (fun p : Fin d × Fin d => A p.1 p.2 * B p.1 p.2)).symm
    have h2 : (∑ i : Fin d, ∑ j : Fin d, A i j * A i j) =
        ∑ p : Fin d × Fin d, (A p.1 p.2) ^ 2 := by
      rw [Fintype.sum_prod_type]
      refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => (sq _).symm
    have h3 : (∑ i : Fin d, ∑ j : Fin d, B i j * B i j) =
        ∑ p : Fin d × Fin d, (B p.1 p.2) ^ 2 := by
      rw [Fintype.sum_prod_type]
      refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => (sq _).symm
    rw [h1, h2, h3]
    exact Finset.sum_mul_sq_le_sq_mul_sq Finset.univ
      (fun p : Fin d × Fin d => A p.1 p.2) (fun p : Fin d × Fin d => B p.1 p.2)
  have hsq : |frobInner A B| ^ 2 ≤ (frobNorm A * frobNorm B) ^ 2 := by
    rw [sq_abs, mul_pow, frobNorm_sq, frobNorm_sq]
    exact hcs
  have hprod_nn : 0 ≤ frobNorm A * frobNorm B := mul_nonneg (frobNorm_nonneg A) (frobNorm_nonneg B)
  nlinarith [abs_nonneg (frobInner A B), sq_nonneg (|frobInner A B| - frobNorm A * frobNorm B)]

lemma frobNorm_triangle {d : ℕ} (A B : Matrix (Fin d) (Fin d) ℝ) :
    frobNorm (A + B) ≤ frobNorm A + frobNorm B := by
  have hexp : frobNormSq (A + B) = frobNormSq A + 2 * frobInner A B + frobNormSq B := by
    unfold frobNormSq frobInner
    have hsum : ∀ i j, (A + B) i j * (A + B) i j = A i j * A i j + 2 * (A i j * B i j) + B i j * B i j := by
      intro i j; simp only [Matrix.add_apply]; ring
    simp_rw [hsum, Finset.sum_add_distrib, ← Finset.mul_sum]
  have hcs := le_trans (le_abs_self _) (frobInner_cauchy_schwarz A B)
  have hsq : (frobNorm (A + B)) ^ 2 ≤ (frobNorm A + frobNorm B) ^ 2 := by
    rw [frobNorm_sq, hexp]
    nlinarith [frobNorm_sq A, frobNorm_sq B, hcs]
  have hnn : 0 ≤ frobNorm A + frobNorm B := add_nonneg (frobNorm_nonneg A) (frobNorm_nonneg B)
  nlinarith [frobNorm_nonneg (A + B), sq_nonneg (frobNorm (A + B) - (frobNorm A + frobNorm B))]

lemma frobNorm_sub_comm {d : ℕ} (A B : Matrix (Fin d) (Fin d) ℝ) :
    frobNorm (A - B) = frobNorm (B - A) := by
  have h : A - B = (-1 : ℝ) • (B - A) := by ext i j; simp
  rw [h, frobNorm_smul]
  norm_num

lemma frobInner_corrMatrix {d p : ℕ} (U E : Fin p → Fin d → ℝ)
    (W : Matrix (Fin d) (Fin d) ℝ) :
    frobInner (corrMatrix U E) W = ∑ i : Fin p, dotProduct (U i) (W.mulVec (E i)) := by
  unfold frobInner corrMatrix
  simp only [Matrix.sum_apply, Matrix.vecMulVec_apply, Finset.sum_mul]
  calc ∑ a : Fin d, ∑ b : Fin d, ∑ i : Fin p, U i a * E i b * W a b
      = ∑ a : Fin d, ∑ i : Fin p, ∑ b : Fin d, U i a * E i b * W a b :=
        Finset.sum_congr rfl fun _ _ => Finset.sum_comm
    _ = ∑ i : Fin p, ∑ a : Fin d, ∑ b : Fin d, U i a * E i b * W a b := Finset.sum_comm
    _ = ∑ i : Fin p, dotProduct (U i) (W.mulVec (E i)) := by
        refine Finset.sum_congr rfl fun i _ => ?_
        simp only [dotProduct, Matrix.mulVec, Finset.mul_sum]
        refine Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ => ?_
        ring

lemma op_feasible_ne_zero {d p : ℕ} (hp : 2 ≤ p) {W : Matrix (Fin d) (Fin d) ℝ}
    {U E : Fin p → Fin d → ℝ} (hfeas : IsOPFeasible W U E) :
    W ≠ 0 := by
  intro h0
  subst h0
  have h0lt : 0 < p := by omega
  have h1lt : 1 < p := by omega
  have hne : (⟨1, h1lt⟩ : Fin p) ≠ ⟨0, h0lt⟩ := by intro h; cases h
  have hlt := hfeas ⟨0, h0lt⟩ ⟨1, h1lt⟩ hne
  simp at hlt

lemma op_feasible_smul {d p : ℕ} {c : ℝ} (hc : 0 < c) {W : Matrix (Fin d) (Fin d) ℝ}
    {U E : Fin p → Fin d → ℝ} (hfeas : IsOPFeasible W U E) :
    IsOPFeasible (c • W) U E := by
  intro i j hij
  have h1 : dotProduct (U i) ((c • W).mulVec (E i)) = c * dotProduct (U i) (W.mulVec (E i)) := by
    simp [Matrix.smul_mulVec, dotProduct_smul, smul_eq_mul]
  have h2 : dotProduct (U j) ((c • W).mulVec (E i)) = c * dotProduct (U j) (W.mulVec (E i)) := by
    simp [Matrix.smul_mulVec, dotProduct_smul, smul_eq_mul]
  rw [h1, h2]
  exact mul_lt_mul_of_pos_left (hfeas i j hij) hc

/-- **OP normalization & target-vs-competitor inequality**:
Every OP-feasible matrix `W` (for `p ≥ 2`) rescales positively to a unit-Frobenius feasible
matrix `W' = (‖W‖_F)⁻¹ • W` satisfying `⟨C, W'⟩_F > S(W')`. -/
theorem op_feasible_implies_unit_corr_gt_compScore {d p : ℕ} (hp : 2 ≤ p)
    {W : Matrix (Fin d) (Fin d) ℝ} {U E : Fin p → Fin d → ℝ}
    (hfeas : IsOPFeasible W U E) :
    let W' := (frobNorm W)⁻¹ • W
    frobNorm W' = 1 ∧ IsOPFeasible W' U E ∧
      frobInner (corrMatrix U E) W' > compScore W' U E hp := by
  intro W'
  have hWne : W ≠ 0 := op_feasible_ne_zero hp hfeas
  have hpos : 0 < frobNorm W := frobNorm_pos_of_ne_zero hWne
  have hnorm : frobNorm W' = 1 := by
    change frobNorm ((frobNorm W)⁻¹ • W) = 1
    rw [frobNorm_smul, abs_of_pos (inv_pos.2 hpos), inv_mul_cancel₀ (ne_of_gt hpos)]
  have hfeas' : IsOPFeasible W' U E := op_feasible_smul (inv_pos.2 hpos) hfeas
  refine ⟨hnorm, hfeas', ?_⟩
  rw [frobInner_corrMatrix, compScore]
  have hp_pos : (Finset.univ : Finset (Fin p)).Nonempty :=
    ⟨⟨0, by omega⟩, Finset.mem_univ _⟩
  refine Finset.sum_lt_sum_of_nonempty hp_pos fun i _ => ?_
  rw [Finset.sup'_lt_iff]
  intro j hj
  exact hfeas' i j (Finset.mem_erase.1 hj).1

/-! ### Part 3: SVD head-tail bounds and Lipschitz regularity -/

/-- **SVD tail operator-norm bound**: If `σ_0 ≥ σ_1 ≥ ... ≥ σ_{d-1} ≥ 0` satisfy `∑ σ_i² ≤ 1`
and `1 ≤ k`, then every tail singular value (`i ≥ k - 1`) satisfies `σ_i ≤ 1 / √k`. -/
theorem tail_sv_le_inv_sqrt_k {d : ℕ} {σ : Fin d → ℝ}
    (hanti : Antitone σ) (hnn : ∀ i, 0 ≤ σ i) (hsum : ∑ i : Fin d, (σ i) ^ 2 ≤ 1)
    {k : ℕ} (hk : 1 ≤ k) {i : Fin d} (hi : k - 1 ≤ i.val) :
    σ i ≤ 1 / Real.sqrt k := by
  have hk_pos : (0 : ℝ) < k := Nat.cast_pos.2 (by omega)
  have hsqrt_pos : 0 < Real.sqrt k := Real.sqrt_pos.2 hk_pos
  have hk_le_d : k ≤ d := by omega
  let S : Finset (Fin d) := Finset.univ.filter fun j => j.val < k
  have hScard : S.card = k := by
    have hbij : S.card = (Finset.range k).card := by
      refine Finset.card_bij (fun j _ => j.val) ?_ ?_ ?_
      · intro j hj; exact Finset.mem_range.2 (Finset.mem_filter.1 hj).2
      · intro j₁ _ j₂ _ heq; exact Fin.ext heq
      · intro m hm
        have hmk : m < k := Finset.mem_range.1 hm
        exact ⟨⟨m, lt_of_lt_of_le hmk hk_le_d⟩, Finset.mem_filter.2 ⟨Finset.mem_univ _, hmk⟩, rfl⟩
    simpa using hbij
  have hS_lb : (k : ℝ) * (σ i) ^ 2 ≤ ∑ j ∈ S, (σ j) ^ 2 := by
    have hconst : ∑ _j ∈ S, (σ i) ^ 2 = (k : ℝ) * (σ i) ^ 2 := by
      rw [Finset.sum_const, hScard, nsmul_eq_mul]
    rw [← hconst]
    refine Finset.sum_le_sum fun j hj => ?_
    have hjle : j ≤ i := by
      change j.val ≤ i.val
      have hjk : j.val < k := (Finset.mem_filter.1 hj).2
      omega
    have hσle : σ i ≤ σ j := hanti hjle
    exact pow_le_pow_left₀ (hnn i) hσle 2
  have hS_ub : ∑ j ∈ S, (σ j) ^ 2 ≤ 1 :=
    le_trans (Finset.sum_le_univ_sum_of_nonneg fun j => sq_nonneg (σ j)) hsum
  have hsq : (σ i) ^ 2 ≤ (1 / Real.sqrt k) ^ 2 := by
    rw [one_div, inv_pow, Real.sq_sqrt (le_of_lt hk_pos)]
    have h1 : (k : ℝ) * (σ i) ^ 2 ≤ 1 := le_trans hS_lb hS_ub
    rw [← one_div, le_div_iff₀ hk_pos]
    linarith
  have hnn_rhs : 0 ≤ 1 / Real.sqrt k := le_of_lt (div_pos one_pos hsqrt_pos)
  nlinarith [hnn i, sq_nonneg (σ i - 1 / Real.sqrt k)]

/-- **SVD head nuclear-norm bound**: For any subset `S` of singular values with `∑ σ_i² ≤ 1`,
`∑_{i ∈ S} σ_i ≤ √|S|`. -/
theorem head_nuclear_le_sqrt_k {d : ℕ} {σ : Fin d → ℝ}
    (hnn : ∀ i, 0 ≤ σ i) (hsum : ∑ i : Fin d, (σ i) ^ 2 ≤ 1) (S : Finset (Fin d)) :
    ∑ i ∈ S, σ i ≤ Real.sqrt S.card := by
  have hcs := Finset.sum_mul_sq_le_sq_mul_sq S (fun _ => (1 : ℝ)) σ
  simp only [one_mul, one_pow, Finset.sum_const, nsmul_eq_mul, mul_one] at hcs
  have hS_ub : ∑ i ∈ S, (σ i) ^ 2 ≤ 1 :=
    le_trans (Finset.sum_le_univ_sum_of_nonneg fun i => sq_nonneg (σ i)) hsum
  have hsq : (∑ i ∈ S, σ i) ^ 2 ≤ (Real.sqrt S.card) ^ 2 := by
    rw [Real.sq_sqrt (Nat.cast_nonneg S.card)]
    exact le_trans hcs (mul_le_of_le_one_right (Nat.cast_nonneg S.card) hS_ub)
  have hsum_nn : 0 ≤ ∑ i ∈ S, σ i := Finset.sum_nonneg fun i _ => hnn i
  nlinarith [Real.sqrt_nonneg S.card, sq_nonneg ((∑ i ∈ S, σ i) - Real.sqrt S.card)]

/-- Frobenius distance to an arbitrary nonempty set `R` (in particular `{X : rank X ≤ k}`). -/
noncomputable def bDist {d : ℕ} (R : Set (Matrix (Fin d) (Fin d) ℝ))
    (W : Matrix (Fin d) (Fin d) ℝ) : ℝ :=
  sInf ((fun X => frobNorm (W - X)) '' R)

/-- **`1`-Lipschitzness of `b(W) = dist_F(W, R)`**:
For any nonempty set `R`, `|bDist R W - bDist R V| ≤ ‖W - V‖_F`. -/
theorem bDist_one_lipschitz {d : ℕ} {R : Set (Matrix (Fin d) (Fin d) ℝ)} (hR : R.Nonempty)
    (W V : Matrix (Fin d) (Fin d) ℝ) :
    |bDist R W - bDist R V| ≤ frobNorm (W - V) := by
  have hbound : ∀ A B : Matrix (Fin d) (Fin d) ℝ,
      bDist R A ≤ bDist R B + frobNorm (A - B) := by
    intro A B
    have hne : ((fun X => frobNorm (B - X)) '' R).Nonempty := hR.image _
    have hbdd : BddBelow ((fun X => frobNorm (A - X)) '' R) :=
      ⟨0, by rintro _ ⟨X, _, rfl⟩; exact frobNorm_nonneg _⟩
    have hle : bDist R A - frobNorm (A - B) ≤ bDist R B := by
      refine le_csInf hne ?_
      rintro _ ⟨X, hX, rfl⟩
      have h1 : bDist R A ≤ frobNorm (A - X) :=
        csInf_le hbdd ⟨X, hX, rfl⟩
      have htri : frobNorm (A - X) ≤ frobNorm (A - B) + frobNorm (B - X) := by
        have heq : A - X = (A - B) + (B - X) := by abel
        rw [heq]
        exact frobNorm_triangle (A - B) (B - X)
      linarith
    linarith
  have h1 := hbound W V
  have h2 := hbound V W
  rw [frobNorm_sub_comm V W] at h2
  exact abs_le.2 ⟨by linarith, by linarith⟩

private lemma dot_mulVec_abs_le {d : ℕ} (u e : Fin d → ℝ)
    (M : Matrix (Fin d) (Fin d) ℝ) :
    |dotProduct u (M.mulVec e)| ≤ vecNorm u * vecNorm e * frobNorm M := by
  have hrew : dotProduct u (M.mulVec e) = frobInner (Matrix.vecMulVec u e) M := by
    unfold frobInner
    simp only [dotProduct, Matrix.mulVec, Matrix.vecMulVec_apply, Finset.mul_sum]
    refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
    ring
  rw [hrew]
  have hcs := frobInner_cauchy_schwarz (Matrix.vecMulVec u e) M
  have houter : frobNorm (Matrix.vecMulVec u e) = vecNorm u * vecNorm e := by
    unfold frobNorm frobNormSq frobInner vecNorm
    have hprod : (∑ i : Fin d, ∑ j : Fin d, (u i * e j) * (u i * e j)) =
        (∑ i : Fin d, (u i) ^ 2) * (∑ j : Fin d, (e j) ^ 2) := by
      rw [Finset.sum_mul]
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun j _ => ?_
      ring
    simp only [Matrix.vecMulVec_apply]
    rw [hprod, Real.sqrt_mul (Finset.sum_nonneg fun i _ => sq_nonneg (u i))]
  simpa [houter] using hcs

private lemma abs_sup'_sub_sup'_le {α : Type*} (s : Finset α) (hs : s.Nonempty)
    (f g : α → ℝ) {C : ℝ} (hfg : ∀ a ∈ s, |f a - g a| ≤ C) :
    |s.sup' hs f - s.sup' hs g| ≤ C := by
  have h1 : s.sup' hs f ≤ s.sup' hs g + C := by
    rw [Finset.sup'_le_iff]
    intro a ha
    have := (abs_le.1 (hfg a ha)).2
    have hga : g a ≤ s.sup' hs g := Finset.le_sup' g ha
    linarith
  have h2 : s.sup' hs g ≤ s.sup' hs f + C := by
    rw [Finset.sup'_le_iff]
    intro a ha
    have := (abs_le.1 (hfg a ha)).1
    have hfa : f a ≤ s.sup' hs f := Finset.le_sup' f ha
    linarith
  exact abs_le.2 ⟨by linarith, by linarith⟩

/-- **Lipschitz continuity of `S(W)` on bounded clouds**:
If `‖u_j‖ ≤ R_u` and `‖e_i‖ ≤ R_e` for all `i, j`, then
`|S(W) - S(V)| ≤ p · R_u · R_e · ‖W - V‖_F`.
(For `R_u = R_e = 2 √d`, this gives the paper's bound `4 p d ‖W - V‖_F`.) -/
theorem compScore_lipschitz_on_bounded_cloud {d p : ℕ} (hp : 2 ≤ p)
    (W V : Matrix (Fin d) (Fin d) ℝ) (U E : Fin p → Fin d → ℝ)
    {R_u R_e : ℝ} (hRu : ∀ j, vecNorm (U j) ≤ R_u) (hRe : ∀ i, vecNorm (E i) ≤ R_e) :
    |compScore W U E hp - compScore V U E hp| ≤
      (p : ℝ) * R_u * R_e * frobNorm (W - V) := by
  unfold compScore
  rw [← Finset.sum_sub_distrib]
  refine le_trans (Finset.abs_sum_le_sum_abs _ _) ?_
  have hterm : ∀ i : Fin p,
      |(Finset.univ.erase i).sup' (erase_nonempty_of_two_le hp i)
          (fun j => dotProduct (U j) (W.mulVec (E i))) -
       (Finset.univ.erase i).sup' (erase_nonempty_of_two_le hp i)
          (fun j => dotProduct (U j) (V.mulVec (E i)))| ≤
        R_u * R_e * frobNorm (W - V) := by
    intro i
    refine abs_sup'_sub_sup'_le _ _ _ _ fun j _ => ?_
    have hdiff : dotProduct (U j) (W.mulVec (E i)) - dotProduct (U j) (V.mulVec (E i)) =
        dotProduct (U j) ((W - V).mulVec (E i)) := by
      simp [Matrix.sub_mulVec, dotProduct_sub]
    rw [hdiff]
    have h1 := dot_mulVec_abs_le (U j) (E i) (W - V)
    have hprod : vecNorm (U j) * vecNorm (E i) ≤ R_u * R_e :=
      mul_le_mul (hRu j) (hRe i) (Real.sqrt_nonneg _)
        (le_trans (Real.sqrt_nonneg _) (hRu j))
    exact le_trans h1 (mul_le_mul_of_nonneg_right hprod (frobNorm_nonneg _))
  calc ∑ i : Fin p,
        |(Finset.univ.erase i).sup' (erase_nonempty_of_two_le hp i)
            (fun j => dotProduct (U j) (W.mulVec (E i))) -
         (Finset.univ.erase i).sup' (erase_nonempty_of_two_le hp i)
            (fun j => dotProduct (U j) (V.mulVec (E i)))|
      ≤ ∑ _i : Fin p, (R_u * R_e * frobNorm (W - V)) := Finset.sum_le_sum fun i _ => hterm i
    _ = (p : ℝ) * R_u * R_e * frobNorm (W - V) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
        ring

/-! ### Part 4: Steps 1–3 deterministic, packing, orthogonal-split, and equicorrelated cores -/

/-- **Step 1 coarse lower bound**:
On the high-probability Step 1 event (`max_{j ≠ i} u_jᵀ W e_i ≥ ‖W e_i‖ √L / 2` for every `i`,
and `p⁻¹ ∑ ‖W e_i‖ ≥ ‖W‖_F / √(2π)`), we have `S(W) / (p √(2L)) ≥ (1 / (4 √π)) ‖W‖_F`. -/
theorem step1_coarse_lower_bound {d p : ℕ} (hp : 2 ≤ p) {L : ℝ} (hL : 0 < L)
    (W : Matrix (Fin d) (Fin d) ℝ) (U E : Fin p → Fin d → ℝ)
    (hmax : ∀ i : Fin p,
      vecNorm (W.mulVec (E i)) * Real.sqrt L / 2 ≤
        (Finset.univ.erase i).sup' (erase_nonempty_of_two_le hp i)
          (fun j => dotProduct (U j) (W.mulVec (E i))))
    (hframe : frobNorm W / Real.sqrt (2 * Real.pi) ≤
      (∑ i : Fin p, vecNorm (W.mulVec (E i))) / p) :
    (1 / (4 * Real.sqrt Real.pi)) * frobNorm W ≤
      compScore W U E hp / (p * Real.sqrt (2 * L)) := by
  have hp_pos : (0 : ℝ) < p := Nat.cast_pos.2 (by omega)
  have hL_pos : 0 < Real.sqrt L := Real.sqrt_pos.2 hL
  have h2L_pos : 0 < Real.sqrt (2 * L) := Real.sqrt_pos.2 (by linarith)
  have h2pi_pos : 0 < Real.sqrt (2 * Real.pi) := Real.sqrt_pos.2 (by positivity)
  have hpi_pos : 0 < Real.sqrt Real.pi := Real.sqrt_pos.2 Real.pi_pos
  have h2_pos : (0 : ℝ) < Real.sqrt 2 := Real.sqrt_pos.2 (by norm_num)
  have hsum : (∑ i : Fin p, vecNorm (W.mulVec (E i))) * (Real.sqrt L / 2) ≤ compScore W U E hp := by
    unfold compScore
    rw [Finset.sum_mul]
    refine Finset.sum_le_sum fun i _ => ?_
    simpa [mul_div_assoc] using hmax i
  have hsum_norm : (p : ℝ) * (frobNorm W / Real.sqrt (2 * Real.pi)) ≤
      ∑ i : Fin p, vecNorm (W.mulVec (E i)) :=
    (le_div_iff₀ hp_pos).1 hframe |> fun h => by linarith
  have hchain : (p : ℝ) * (frobNorm W / Real.sqrt (2 * Real.pi)) * (Real.sqrt L / 2) ≤
      compScore W U E hp :=
    le_trans (mul_le_mul_of_nonneg_right hsum_norm (by positivity)) hsum
  rw [le_div_iff₀ (mul_pos hp_pos h2L_pos)]
  have hsqrt2L : Real.sqrt (2 * L) = Real.sqrt 2 * Real.sqrt L :=
    Real.sqrt_mul (by norm_num) L
  have hsqrt2pi : Real.sqrt (2 * Real.pi) = Real.sqrt 2 * Real.sqrt Real.pi :=
    Real.sqrt_mul (by norm_num) Real.pi
  have h2sq : (Real.sqrt 2) ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  have heq : (1 / (4 * Real.sqrt Real.pi)) * frobNorm W * (p * Real.sqrt (2 * L)) =
      (p : ℝ) * (frobNorm W / Real.sqrt (2 * Real.pi)) * (Real.sqrt L / 2) := by
    rw [hsqrt2L, hsqrt2pi]
    field_simp [ne_of_gt hpi_pos, ne_of_gt h2_pos]
    rw [h2sq]; ring
  linarith

/-- **Step 2 combinatorial maximal-packing escape lemma**:
Given `A ⊆ Fin p` with `|A| ≥ p/4` and `n ≤ p/16`, if no `n`-element subset of `A` consists of
`good` pairwise-`compat` indices, then there exists a good pairwise-compatible `J ⊆ A` with
`|J| < n`, `|A \ J| ≥ p/8`, and every `j ∈ A \ J` either fails `good j` or fails `compat j l`
for some `l ∈ J`. -/
theorem step2_maximal_packing_escape {p : ℕ} (A : Finset (Fin p)) (n : ℕ)
    (good : Fin p → Prop) (compat : Fin p → Fin p → Prop)
    (hcompat_symm : ∀ a b, compat a b → compat b a)
    (hA : p ≤ 4 * A.card) (hn : 16 * n ≤ p)
    (hno : ¬ ∃ I ⊆ A, I.card = n ∧ (∀ j ∈ I, good j) ∧
      (∀ j ∈ I, ∀ l ∈ I, j ≠ l → compat j l)) :
    ∃ J ⊆ A, J.card < n ∧ (∀ j ∈ J, good j) ∧
      (∀ j ∈ J, ∀ l ∈ J, j ≠ l → compat j l) ∧
      p ≤ 8 * (A \ J).card ∧
      (∀ j ∈ A \ J, ¬ good j ∨ ∃ l ∈ J, ¬ compat j l) := by
  classical
  let valid (I : Finset (Fin p)) : Prop :=
    I ⊆ A ∧ (∀ j ∈ I, good j) ∧ (∀ j ∈ I, ∀ l ∈ I, j ≠ l → compat j l)
  let V : Finset (Finset (Fin p)) := A.powerset.filter valid
  have hVne : V.Nonempty := by
    refine ⟨∅, Finset.mem_filter.2 ⟨Finset.mem_powerset.2 (Finset.empty_subset A), ?_⟩⟩
    exact ⟨Finset.empty_subset A, by simp, by simp⟩
  obtain ⟨J, hJV, hJmax⟩ := V.exists_max_image Finset.card hVne
  have hJval : valid J := (Finset.mem_filter.1 hJV).2
  rcases hJval with ⟨hJA, hJgood, hJcompat⟩
  have hJlt : J.card < n := by
    by_contra hge
    push_neg at hge
    obtain ⟨I, hIJ, hIcard⟩ := Finset.exists_subset_card_eq hge
    exact hno ⟨I, subset_trans hIJ hJA, hIcard,
      fun j hj => hJgood j (hIJ hj),
      fun j hj l hl hne => hJcompat j (hIJ hj) l (hIJ hl) hne⟩
  have hcard_sdiff : (A \ J).card = A.card - J.card := Finset.card_sdiff_of_subset hJA
  have hsdiff_lb : p ≤ 8 * (A \ J).card := by
    rw [hcard_sdiff]
    omega
  refine ⟨J, hJA, hJlt, hJgood, hJcompat, hsdiff_lb, ?_⟩
  intro j hj
  have hjA : j ∈ A := (Finset.mem_sdiff.1 hj).1
  have hjJ : j ∉ J := (Finset.mem_sdiff.1 hj).2
  by_contra hboth
  push_neg at hboth
  rcases hboth with ⟨hjgood, hjcompat⟩
  let J' := insert j J
  have hJ'V : J' ∈ V := by
    refine Finset.mem_filter.2 ⟨Finset.mem_powerset.2 (Finset.insert_subset hjA hJA), ?_⟩
    refine ⟨Finset.insert_subset hjA hJA, ?_, ?_⟩
    · intro x hx
      rcases Finset.mem_insert.1 hx with rfl | hxJ
      · exact hjgood
      · exact hJgood x hxJ
    · intro x hx y hy hxy
      rcases Finset.mem_insert.1 hx with hxj | hxJ
      · rw [hxj]
        rcases Finset.mem_insert.1 hy with hyj | hyJ
        · exact (hxy (hxj.trans hyj.symm)).elim
        · exact hjcompat y hyJ
      · rcases Finset.mem_insert.1 hy with hyj | hyJ
        · rw [hyj]
          exact hcompat_symm j x (hjcompat x hxJ)
        · exact hJcompat x hxJ y hyJ hxy
  have hle := hJmax J' hJ'V
  rw [Finset.card_insert_of_notMem hjJ] at hle
  omega

/-- **Step 3 orthogonal key-split score transfer**:
Under `V = H + T` with `T = b • B`, `e = e_∥ + e_⊥`, `H e_⊥ = 0`, `T e_∥ = 0`,
`uᵀ H e_∥ ≥ 0`, `‖Bᵀ u‖ ≥ 1 - η`, and `⟨Bᵀ u / ‖Bᵀ u‖, e_⊥⟩ ≥ t > 0`, we obtain
`uᵀ V e ≥ b (1 - η) t`. -/
theorem step3_orth_split_score_bound {d : ℕ}
    (V H T B : Matrix (Fin d) (Fin d) ℝ) (u e e_par e_perp : Fin d → ℝ)
    {b η t : ℝ} (hV : V = H + T) (he : e = e_par + e_perp)
    (hHperp : H.mulVec e_perp = 0) (hTpar : T.mulVec e_par = 0)
    (hT : T = b • B) (hb : 0 ≤ b) (hHnn : 0 ≤ dotProduct u (H.mulVec e_par))
    (hznorm : 1 - η ≤ vecNorm (Bᵀ.mulVec u)) (ht : 0 ≤ t)
    (hzpos : 0 < vecNorm (Bᵀ.mulVec u))
    (hX : t ≤ dotProduct ((vecNorm (Bᵀ.mulVec u))⁻¹ • (Bᵀ.mulVec u)) e_perp) :
    b * (1 - η) * t ≤ dotProduct u (V.mulVec e) := by
  have hdecomp : dotProduct u (V.mulVec e) =
      dotProduct u (H.mulVec e_par) + dotProduct u (T.mulVec e_perp) := by
    rw [hV, he, Matrix.add_mulVec, Matrix.mulVec_add, Matrix.mulVec_add, hHperp, hTpar,
      add_zero, zero_add, dotProduct_add]
  have hTdot : dotProduct u (T.mulVec e_perp) =
      b * vecNorm (Bᵀ.mulVec u) *
        dotProduct ((vecNorm (Bᵀ.mulVec u))⁻¹ • (Bᵀ.mulVec u)) e_perp := by
    rw [hT, Matrix.smul_mulVec, dotProduct_smul, smul_eq_mul, smul_dotProduct, smul_eq_mul]
    have hadj : dotProduct u (B.mulVec e_perp) = dotProduct (Bᵀ.mulVec u) e_perp := by
      simp [dotProduct, Matrix.mulVec, Finset.mul_sum, Finset.sum_mul]
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun i _ => ?_
      ring
    rw [hadj]
    have hnz : vecNorm (Bᵀ.mulVec u) ≠ 0 := ne_of_gt hzpos
    field_simp [hnz]
  rw [hdecomp, hTdot]
  have h1 : b * (1 - η) * t ≤ b * vecNorm (Bᵀ.mulVec u) * t := by
    have hmul := mul_le_mul_of_nonneg_left hznorm hb
    exact mul_le_mul_of_nonneg_right hmul ht
  have h2 : b * vecNorm (Bᵀ.mulVec u) * t ≤
      b * vecNorm (Bᵀ.mulVec u) *
        dotProduct ((vecNorm (Bᵀ.mulVec u))⁻¹ • (Bᵀ.mulVec u)) e_perp :=
    mul_le_mul_of_nonneg_left hX (mul_nonneg hb (le_of_lt hzpos))
  linarith

/-- **Step 3 equicorrelated Slepian shift reduction**:
If `Y_r = √ρ G_0 + √(1 - ρ) G_r`, `√ρ G_0 ≥ -(δ/4) √(2L)`, `1 - 3δ/4 ≤ (1 - δ/2) √(1 - ρ)`,
and `Y_r < (1 - δ) √(2L)`, then `G_r < (1 - δ/2) √(2L)`. -/
theorem step3_equicorr_shift {ρ δ L G0 Gr : ℝ}
    (hρ1 : ρ < 1) (hL : 0 < L)
    (hcond : 1 - 3 * δ / 4 ≤ (1 - δ / 2) * Real.sqrt (1 - ρ))
    (hshift : -(δ / 4) * Real.sqrt (2 * L) ≤ Real.sqrt ρ * G0)
    (hY : Real.sqrt ρ * G0 + Real.sqrt (1 - ρ) * Gr < (1 - δ) * Real.sqrt (2 * L)) :
    Gr < (1 - δ / 2) * Real.sqrt (2 * L) := by
  have hsq_pos : 0 < Real.sqrt (1 - ρ) := Real.sqrt_pos.2 (by linarith)
  have h2L_pos : 0 < Real.sqrt (2 * L) := Real.sqrt_pos.2 (by linarith)
  have h1 : Real.sqrt (1 - ρ) * Gr < (1 - 3 * δ / 4) * Real.sqrt (2 * L) := by linarith
  have h2 : (1 - 3 * δ / 4) * Real.sqrt (2 * L) ≤
      (1 - δ / 2) * Real.sqrt (1 - ρ) * Real.sqrt (2 * L) :=
    mul_le_mul_of_nonneg_right hcond (le_of_lt h2L_pos)
  have h3 : Real.sqrt (1 - ρ) * Gr < Real.sqrt (1 - ρ) * ((1 - δ / 2) * Real.sqrt (2 * L)) := by
    linarith
  exact lt_of_mul_lt_mul_left h3 (le_of_lt hsq_pos)

/-- **Step 3 query-counting aggregation**:
If all `p` query scores `s_i` are nonnegative and at least `(1 - δ) p` queries satisfy
`s_i ≥ b (1 - η) (1 - δ) √(2L)`, then `(∑ s_i) / (p √(2L)) ≥ (1 - δ)² (1 - η) b`. -/
theorem step3_query_counting_lower_bound {p : ℕ} (hp : 0 < p) {b η δ L : ℝ}
    (hb : 0 ≤ b) (hη : η ≤ 1) (hδ : δ ≤ 1) (hL : 0 < L)
    (s : Fin p → ℝ) (hs_nn : ∀ i, 0 ≤ s i)
    (Good : Finset (Fin p))
    (hcard : (1 - δ) * p ≤ (Good.card : ℝ))
    (hgood : ∀ i ∈ Good, b * (1 - η) * (1 - δ) * Real.sqrt (2 * L) ≤ s i) :
    (1 - δ) ^ 2 * (1 - η) * b ≤ (∑ i : Fin p, s i) / (p * Real.sqrt (2 * L)) := by
  have hp_pos : (0 : ℝ) < p := Nat.cast_pos.2 hp
  have h2L_pos : 0 < Real.sqrt (2 * L) := Real.sqrt_pos.2 (by linarith)
  have hthr_nn : 0 ≤ b * (1 - η) * (1 - δ) * Real.sqrt (2 * L) := by
    have h1 : 0 ≤ 1 - η := by linarith
    have h2 : 0 ≤ 1 - δ := by linarith
    positivity
  have hsum1 : ∑ i ∈ Good, s i ≤ ∑ i : Fin p, s i :=
    Finset.sum_le_univ_sum_of_nonneg hs_nn
  have hsum2 : (Good.card : ℝ) * (b * (1 - η) * (1 - δ) * Real.sqrt (2 * L)) ≤
      ∑ i ∈ Good, s i := by
    have hconst : ∑ _i ∈ Good, (b * (1 - η) * (1 - δ) * Real.sqrt (2 * L)) =
        (Good.card : ℝ) * (b * (1 - η) * (1 - δ) * Real.sqrt (2 * L)) := by
      rw [Finset.sum_const, nsmul_eq_mul]
    rw [← hconst]
    exact Finset.sum_le_sum hgood
  have hsum3 : (1 - δ) * p * (b * (1 - η) * (1 - δ) * Real.sqrt (2 * L)) ≤
      (Good.card : ℝ) * (b * (1 - η) * (1 - δ) * Real.sqrt (2 * L)) :=
    mul_le_mul_of_nonneg_right hcard hthr_nn
  rw [le_div_iff₀ (mul_pos hp_pos h2L_pos)]
  have heq : (1 - δ) ^ 2 * (1 - η) * b * (p * Real.sqrt (2 * L)) =
      (1 - δ) * p * (b * (1 - η) * (1 - δ) * Real.sqrt (2 * L)) := by ring
  linarith

/-! ### Part 5: Step 4 von Neumann split, OP converse contradiction, and capacity equivalence -/

/-- **Step 4 von Neumann head-tail split**:
Given `⟨C, W⟩_F = ⟨C, H⟩_F + ⟨C, T⟩_F`, `⟨C, H⟩_F ≤ ‖H‖_* ‖C‖_op`, `‖H‖_* ≤ √k`, and
`⟨C, T⟩_F ≤ ‖C_m‖_F · b`, we obtain the normalized correlation upper bound `(4)`. -/
theorem von_neumann_head_tail_split {d p k : ℕ} (hd : 0 < d) (hp : 0 < p)
    {C W H T : Matrix (Fin d) (Fin d) ℝ} {nucH opC normCm b : ℝ}
    (hsplit : frobInner C W = frobInner C H + frobInner C T)
    (hH_vn : frobInner C H ≤ nucH * opC)
    (hH_nuc : nucH ≤ Real.sqrt k)
    (hopC : 0 ≤ opC)
    (hT_vn : frobInner C T ≤ normCm * b) :
    frobInner C W / (d * Real.sqrt p) ≤
      (Real.sqrt k * opC) / (d * Real.sqrt p) + (normCm / (d * Real.sqrt p)) * b := by
  have hdenom : 0 < (d : ℝ) * Real.sqrt p :=
    mul_pos (Nat.cast_pos.2 hd) (Real.sqrt_pos.2 (Nat.cast_pos.2 hp))
  have hH_ub : frobInner C H ≤ Real.sqrt k * opC :=
    le_trans hH_vn (mul_le_mul_of_nonneg_right hH_nuc hopC)
  have htot : frobInner C W ≤ Real.sqrt k * opC + normCm * b := by linarith
  have hdiv := div_le_div_of_nonneg_right htot (le_of_lt hdenom)
  have hrhs : (Real.sqrt k * opC + normCm * b) / ((d : ℝ) * Real.sqrt p) =
      (Real.sqrt k * opC) / ((d : ℝ) * Real.sqrt p) +
        (normCm / ((d : ℝ) * Real.sqrt p)) * b := by ring
  linarith

/-- For every `E_κ > 0` and `α > E_κ / 2`, there exists `ε ∈ (0, 1)` such that
`(1 - ε) √(2 α) > √E_κ`. -/
theorem exists_eps_gap_above_half_energy {E_κ α : ℝ} (hE : 0 < E_κ) (hα : E_κ / 2 < α) :
    ∃ ε ∈ Set.Ioo (0 : ℝ) 1, Real.sqrt E_κ < (1 - ε) * Real.sqrt (2 * α) := by
  have h2α : E_κ < 2 * α := by linarith
  have hsq : Real.sqrt E_κ < Real.sqrt (2 * α) := Real.sqrt_lt_sqrt (le_of_lt hE) h2α
  have hsq_pos : 0 < Real.sqrt (2 * α) := Real.sqrt_pos.2 (by linarith)
  let ε : ℝ := (Real.sqrt (2 * α) - Real.sqrt E_κ) / (2 * Real.sqrt (2 * α))
  refine ⟨ε, ⟨?_, ?_⟩, ?_⟩
  · exact div_pos (by linarith) (by linarith)
  · rw [div_lt_one (by linarith)]
    have hE_pos : 0 < Real.sqrt E_κ := Real.sqrt_pos.2 hE
    linarith
  · have hcalc : (1 - ε) * Real.sqrt (2 * α) = (Real.sqrt (2 * α) + Real.sqrt E_κ) / 2 := by
      dsimp [ε]
      field_simp [ne_of_gt hsq_pos]
      ring
    rw [hcalc]
    linarith

/-- **Step 4 sharp OP converse contradiction**:
For any `E_κ > 0`, `α > E_κ / 2`, `c_0 > 0`, and `ε ∈ (0, 1)` with `(1 - ε) √(2α) > √E_κ`,
define `b_0 := c_0 √(2α) / (2 √E_κ) > 0` and `gap := (1 - ε) √(2α) - √E_κ > 0`.
Whenever the `o(1)` error terms satisfy `η₁ + η₃ ≤ c_0 √(2α) / 2` and `η₂ + η₃ < gap · b_0`,
no `b ∈ [0, 1]` and normalized scores `S_norm, C_norm` can simultaneously satisfy:
1. the coarse lower bound `c_0 √(2α) - η₁ ≤ S_norm`,
2. the sharp lower bound `(1 - ε) √(2α) b - η₂ ≤ S_norm`,
3. the rank-`m` correlation upper bound `C_norm ≤ √E_κ b + η₃`, and
4. OP feasibility `S_norm < C_norm`. -/
theorem op_converse_core_contradiction {E_κ α c_0 ε η₁ η₂ η₃ : ℝ}
    (hE : 0 < E_κ)
    (hgap : Real.sqrt E_κ < (1 - ε) * Real.sqrt (2 * α))
    (hη13 : η₁ + η₃ ≤ c_0 * Real.sqrt (2 * α) / 2)
    (hη23 : η₂ + η₃ <
      ((1 - ε) * Real.sqrt (2 * α) - Real.sqrt E_κ) *
        (c_0 * Real.sqrt (2 * α) / (2 * Real.sqrt E_κ))) :
    ¬ ∃ (b S_norm C_norm : ℝ),
      c_0 * Real.sqrt (2 * α) - η₁ ≤ S_norm ∧
      (1 - ε) * Real.sqrt (2 * α) * b - η₂ ≤ S_norm ∧
      C_norm ≤ Real.sqrt E_κ * b + η₃ ∧
      S_norm < C_norm := by
  rintro ⟨b, S_norm, C_norm, hcoarse, hsharp, hupper, hfeas⟩
  have hsqE : 0 < Real.sqrt E_κ := Real.sqrt_pos.2 hE
  set b0 : ℝ := c_0 * Real.sqrt (2 * α) / (2 * Real.sqrt E_κ)
  set gap : ℝ := (1 - ε) * Real.sqrt (2 * α) - Real.sqrt E_κ
  have hgap_pos : 0 < gap := by dsimp [gap]; linarith
  have hb_lb : b0 ≤ b := by
    have h1 : c_0 * Real.sqrt (2 * α) / 2 ≤ Real.sqrt E_κ * b := by linarith
    dsimp [b0]
    rw [div_le_iff₀ (by linarith : 0 < 2 * Real.sqrt E_κ)]
    linarith
  have hgap_mul : gap * b0 ≤ gap * b :=
    mul_le_mul_of_nonneg_left hb_lb (le_of_lt hgap_pos)
  have hdiff : C_norm - S_norm ≤ -gap * b + (η₂ + η₃) := by
    dsimp [gap]
    linarith
  linarith

/-- **Capacity threshold equivalence (`conjecture-capacity`)**:
For every fixed rank fraction `κ ∈ (0, 1]`, let `q_κ ∈ [0, 2)` be the unique quarter-circle
quantile (`qcMass q_κ = κ`) and `E_κ = qcEnergy q_κ`. Any sharp threshold `α_c` for which OP is
asymptotically feasible for all `α < E_κ / 2` and asymptotically infeasible for all `α > E_κ / 2`
satisfies `α_c = E_κ / 2` (and `α_c = 1 / 2` at full rank `κ = 1`). -/
theorem capacity_threshold_equality {κ : ℝ} (hκ0 : 0 < κ) (hκ1 : κ ≤ 1)
    {P : ℝ → Prop} {α_c : ℝ}
    (hach : ∀ α, α < (Classical.choose (exists_unique_q_kappa hκ0 hκ1) |> qcEnergy) / 2 → P α)
    (hconv : ∀ α, (Classical.choose (exists_unique_q_kappa hκ0 hκ1) |> qcEnergy) / 2 < α → ¬ P α)
    (hthr_lo : ∀ α, α < α_c → P α)
    (hthr_hi : ∀ α, α_c < α → ¬ P α) :
    α_c = qcEnergy (Classical.choose (exists_unique_q_kappa hκ0 hκ1)) / 2 := by
  set E_κ := qcEnergy (Classical.choose (exists_unique_q_kappa hκ0 hκ1))
  by_contra hne
  rcases lt_or_gt_of_ne hne with hlt | hgt
  · exact hthr_hi ((α_c + E_κ / 2) / 2) (by linarith) (hach ((α_c + E_κ / 2) / 2) (by linarith))
  · exact hconv ((α_c + E_κ / 2) / 2) (by linarith) (hthr_lo ((α_c + E_κ / 2) / 2) (by linarith))

end LinearAssocMemory
