import Mathlib

/-!
# Equilibria of Hierarchical and Random `p`-Adic Deep Neural Networks
(arXiv:2601.19070, Section 10, Conjectures `Conj_tree` and `Conj_random`)

We formalize in Lean 4 + Mathlib the mathematical core of the proof of `Conj_tree`
and the fixed-profile refutation of `Conj_random` from W. A. Zúñiga-Galindo,
*Critical Organization of `p`-Adic Deep Neural Networks* (arXiv:2601.19070):

1. **Branch geometry of `f(t) = artanh(t) - A t` and exact `3^n` / `C(n, k) 2^(n-k)` counts (`Conj_tree`)**:
   - `artanh`, `branchFn A t = artanh t - A t`, `foldPt A = √(1 - 1/A)`, `foldHeight A = -f(r)`.
   - `branchFn_hasDerivAt`, `branchFn_odd`, `branchFn_strictAntiOn_mid`,
     `branchFn_strictMonoOn_outer`, `foldHeight_pos`: Exact monotonicity and fold height `M_A > 0`.
   - `branchFn_zero_three_roots`: For `A > 1`, `f(t) = 0` has exact unique root in each of the
     three open branch intervals `(-1, -r)`, `(-r, r)`, `(r, 1)`.
   - `branchPattern_card`, `branchPattern_index_card`: Exact `3^n` branch patterns and exact
     binomial index count `C(n, k) * 2^(n - k)` for index `k` (number of middle-branch sites).
   - `decoupled_equilibria_card_eq_three_pow`: At `B = 0` (`β = 0`), the exact equilibrium count
     in `(-1, 1)^n` is `3^n`, with `C(n, k) * 2^(n - k)` equilibria having `k` unstable coordinates.
2. **Weighted Gershgorin bound, Schur-complement invertibility, and linear stability of all `2^n` outer equilibria**:
   - `weighted_gershgorin_posDef`: Every real symmetric matrix `M` admitting positive weights
     `w_i > 0` with `M_{ii} w_i > ∑_{j ≠ i} |M_{ij}| w_j` is strictly positive definite (`xᵀ M x > 0`).
   - `midpoint_deriv_identity`: Exact identity `f'((1 + r)/2) = A r + 1/(3 + r) > M_A > β`.
   - `outer_weighted_gershgorin_ineq`: For ANY equilibrium `u` of `f(u_i) = (B u)_i` with
     `B ≥ 0`, row sums `β < M_A`, and `O = {i | |u_i| > r}`, the weights `w_i = min(|u_i| - r, 1 - |u_i|)`
     satisfy `f'(|u_i|) w_i > ∑_{j ∈ O} B_{ij} w_j` for every `i ∈ O`.
   - `outer_hessian_posDef`: Therefore `H_{OO} = diag(f'(u_i))_{OO} - B_{OO}` is strictly positive definite.
   - `outer_jacobian_negDef_and_L2_contraction`: For every one of the `2^n` outer-only equilibria
     (`O = univ`), the symmetrized Jacobian `D^{1/2} J D^{-1/2} = -D^{1/2} H D^{1/2}` is strictly
     negative definite (`vᵀ (-D^{1/2} H D^{1/2}) v < 0` for all `v ≠ 0`), and on `(D^l)^⊥`
     the contraction bound `A * max_i (1 - u_i^2) < 1` places the complementary spectrum strictly
     inside `{Re z < 0}`.
3. **Refutation core for `Conj_random` (`0 < ρ(v) ≤ 1/64` with `≥ 3` equilibria)**:
   - `det_two_by_two_block`: `det ![![1, -x], ![-y, 1]] = 1 - x * y`.
   - `two_by_two_three_equilibria`: Whenever `x > 0, y > 0, x * y > 1` (so `det(I - M) < 0`),
     `h = M tanh h` admits at least three distinct equilibria `(0, 0)`, `(h₁*, h₂*)`, `(-h₁*, -h₂*)`
     with `h₁*, h₂* > 0` (proved constructively via IVT).
   - `parity_prob_formula`, `parity_prob_tendsto_half`: Exact odd-count probability
     `∑_{k odd} C(m, k) q^k (1 - q)^(m - k) = (1 - (1 - 2 q)^m) / 2`, converging to `1/2`
     whenever `m * q_m → ∞` and `0 ≤ q_m ≤ 1/2`.
   - `profile_mass_budget_le_one_div_64`: For any dyadic shell weights `a_k > 0` with
     `√(a_k) ≤ 2^{-(k+1)} / 64`, the total `L^1` profile mass `ρ(v)² = ∑' k, a_k` satisfies
     `0 < ρ(v)² ≤ 1/4096`, hence `0 < ρ(v) ≤ 1/64 < 1`.

Machine-checked vs prose-only:
- Machine-checked: Branch geometry of `artanh t - A t`, exact `3^n` and `C(n, k) 2^(n-k)`
  counts, weighted Gershgorin positive definiteness of `H_{OO}`, exact midpoint derivative
  identity, negative definiteness of the symmetrized Jacobian and `L^2` contraction bound
  `A ‖D‖ < 1` for all `2^n` outer equilibria, `2×2` constructive three-equilibria witness,
  exact parity probability formula and limit `1/2`, and exact profile mass budget `0 < ρ(v) ≤ 1/64`.
- Prose-only (in `solutions/padic_dnn_critical/proof.pdf`): Brouwer fixed-point existence and
  homotopy continuation across `b ∈ [0, 1]` for general `β > 0`, Sylvester inertia for mixed
  indices, `ε`-net Gaussian operator norm concentration for `B`, and Schur-complement sign
  transfer in Lemma 3.
-/

open BigOperators Finset Filter Topology

namespace PAdicDNNCritical

/-! ### 1. Branch Geometry of `f(t) = artanh(t) - A t` and Exact `3^n` / `C(n, k) 2^(n-k)` Counts -/

/-- Inverse hyperbolic tangent `artanh t = (1/2) log((1 + t)/(1 - t))` on `(-1, 1)`. -/
noncomputable def artanh (t : ℝ) : ℝ :=
  (1 / 2 : ℝ) * Real.log ((1 + t) / (1 - t))

/-- Branch function `f_A(t) = artanh t - A t`. -/
noncomputable def branchFn (A t : ℝ) : ℝ :=
  artanh t - A * t

/-- Derivative `f_A'(t) = 1 / (1 - t²) - A`. -/
noncomputable def branchFnDeriv (A t : ℝ) : ℝ :=
  1 / (1 - t ^ 2) - A

/-- Positive fold point `r_A = √(1 - 1/A) ∈ (0, 1)` for `A > 1`. -/
noncomputable def foldPt (A : ℝ) : ℝ :=
  Real.sqrt (1 - 1 / A)

/-- Fold height `M_A = A r_A - artanh r_A = -f_A(r_A)`. -/
noncomputable def foldHeight (A : ℝ) : ℝ :=
  -branchFn A (foldPt A)

lemma artanh_zero : artanh 0 = 0 := by simp [artanh]

lemma artanh_neg {t : ℝ} (ht1 : -1 < t) (ht2 : t < 1) : artanh (-t) = -artanh t := by
  unfold artanh
  have h1 : 0 < 1 - t := by linarith
  have h2 : 0 < 1 + t := by linarith
  have hinv : (1 + -t) / (1 - -t) = ((1 + t) / (1 - t))⁻¹ := by
    field_simp [ne_of_gt h1, ne_of_gt h2]; ring
  rw [hinv, Real.log_inv]
  ring

lemma branchFn_neg {A t : ℝ} (ht1 : -1 < t) (ht2 : t < 1) :
    branchFn A (-t) = -branchFn A t := by
  unfold branchFn
  rw [artanh_neg ht1 ht2]
  ring

lemma artanh_hasDerivAt {t : ℝ} (ht1 : -1 < t) (ht2 : t < 1) :
    HasDerivAt artanh (1 / (1 - t ^ 2)) t := by
  have h1 : 0 < 1 + t := by linarith
  have h2 : 0 < 1 - t := by linarith
  have hq : HasDerivAt (fun x : ℝ => (1 + x) / (1 - x)) (2 / (1 - t) ^ 2) t := by
    have ha : HasDerivAt (fun x : ℝ => 1 + x) 1 t := (hasDerivAt_id t).const_add 1
    have hb : HasDerivAt (fun x : ℝ => 1 - x) (-1) t := by
      have := ((hasDerivAt_id t).neg).const_add 1
      convert this using 1
    have hd := ha.div hb (ne_of_gt h2)
    convert hd using 1
    ring
  have hlog : HasDerivAt (fun x => Real.log ((1 + x) / (1 - x)))
      ((2 / (1 - t) ^ 2) / ((1 + t) / (1 - t))) t :=
    hq.log (ne_of_gt (div_pos h1 h2))
  have hmul := hlog.const_mul (1 / 2 : ℝ)
  convert hmul using 1
  have hne1 : 1 + t ≠ 0 := ne_of_gt h1
  have hne2 : 1 - t ≠ 0 := ne_of_gt h2
  have hne3 : 1 - t ^ 2 ≠ 0 := by
    rw [show 1 - t ^ 2 = (1 - t) * (1 + t) by ring]
    exact mul_ne_zero hne2 hne1
  field_simp [hne1, hne2, hne3]
  ring

lemma branchFn_hasDerivAt {A t : ℝ} (ht1 : -1 < t) (ht2 : t < 1) :
    HasDerivAt (branchFn A) (branchFnDeriv A t) t := by
  unfold branchFn branchFnDeriv
  have hlin : HasDerivAt (fun x : ℝ => A * x) A t := by
    simpa using (hasDerivAt_id t).const_mul A
  exact (artanh_hasDerivAt ht1 ht2).sub hlin

lemma foldPt_bounds {A : ℝ} (hA : 1 < A) :
    0 < foldPt A ∧ foldPt A < 1 ∧ 1 - foldPt A ^ 2 = 1 / A ∧ A = 1 / (1 - foldPt A ^ 2) := by
  have hpos : 0 < 1 - 1 / A := by
    rw [sub_pos, div_lt_one (by linarith)]; exact hA
  have hlt1 : 1 - 1 / A < 1 := by
    have : 0 < 1 / A := by positivity
    linarith
  have hr_pos : 0 < foldPt A := Real.sqrt_pos.mpr hpos
  have hr_lt1 : foldPt A < 1 := by
    calc foldPt A = Real.sqrt (1 - 1 / A) := rfl
      _ < Real.sqrt 1 := Real.sqrt_lt_sqrt (le_of_lt hpos) hlt1
      _ = 1 := Real.sqrt_one
  have hsq : foldPt A ^ 2 = 1 - 1 / A := Real.sq_sqrt (le_of_lt hpos)
  refine ⟨hr_pos, hr_lt1, by linarith, ?_⟩
  rw [hsq]
  have hAne : A ≠ 0 := by linarith
  field_simp [hAne]
  ring

lemma branchFnDeriv_neg_on_mid {A t : ℝ} (hA : 1 < A) (ht : |t| < foldPt A) :
    branchFnDeriv A t < 0 := by
  have hr := foldPt_bounds hA
  have htsq : t ^ 2 < foldPt A ^ 2 := by
    have h1 : |t| ^ 2 < foldPt A ^ 2 := by nlinarith [abs_nonneg t]
    rwa [sq_abs] at h1
  have hden : 1 / A < 1 - t ^ 2 := by linarith [hr.2.2.1]
  have hden_pos : 0 < 1 - t ^ 2 := lt_trans (by positivity) hden
  unfold branchFnDeriv
  rw [sub_neg, div_lt_iff₀ hden_pos]
  have hApos : 0 < A := by linarith
  have := mul_lt_mul_of_pos_left hden hApos
  rw [mul_one_div_cancel (ne_of_gt hApos)] at this
  linarith

lemma branchFnDeriv_pos_on_outer {A t : ℝ} (hA : 1 < A) (ht1 : foldPt A < |t|) (ht2 : |t| < 1) :
    0 < branchFnDeriv A t := by
  have hr := foldPt_bounds hA
  have htsq : foldPt A ^ 2 < t ^ 2 := by
    have h1 : foldPt A ^ 2 < |t| ^ 2 := by nlinarith [hr.1]
    rwa [sq_abs] at h1
  have htsq1 : t ^ 2 < 1 := by
    have h1 : |t| ^ 2 < 1 := by nlinarith [abs_nonneg t]
    rwa [sq_abs] at h1
  have hden_pos : 0 < 1 - t ^ 2 := sub_pos.mpr htsq1
  have hden_lt : 1 - t ^ 2 < 1 / A := by linarith [hr.2.2.1]
  unfold branchFnDeriv
  rw [sub_pos, lt_div_iff₀ hden_pos]
  have hApos : 0 < A := by linarith
  have := mul_lt_mul_of_pos_left hden_lt hApos
  rw [mul_one_div_cancel (ne_of_gt hApos)] at this
  linarith

/-- `f_A` is strictly decreasing on `[-r_A, r_A]` and strictly increasing on `[r_A, 1)`
and `(-1, -r_A]`, with strictly positive fold height `M_A = -f_A(r_A) > 0`. -/
theorem branchFn_monotonicity_and_foldHeight {A : ℝ} (hA : 1 < A) :
    StrictAntiOn (branchFn A) (Set.Icc (-foldPt A) (foldPt A)) ∧
    StrictMonoOn (branchFn A) (Set.Ico (foldPt A) 1) ∧
    StrictMonoOn (branchFn A) (Set.Ioc (-1) (-foldPt A)) ∧
    0 < foldHeight A := by
  have hr := foldPt_bounds hA
  have hanti : StrictAntiOn (branchFn A) (Set.Icc (-foldPt A) (foldPt A)) := by
    apply strictAntiOn_of_deriv_neg (convex_Icc _ _)
    · intro x hx
      exact (branchFn_hasDerivAt (by linarith [hx.1, hr.2.1]) (by linarith [hx.2, hr.2.1])).continuousAt.continuousWithinAt
    · intro x hx
      rw [interior_Icc] at hx
      rw [(branchFn_hasDerivAt (by linarith [hx.1, hr.2.1]) (by linarith [hx.2, hr.2.1])).deriv]
      exact branchFnDeriv_neg_on_mid hA (abs_lt.mpr hx)
  have hmono_pos : StrictMonoOn (branchFn A) (Set.Ico (foldPt A) 1) := by
    apply strictMonoOn_of_deriv_pos (convex_Ico _ _)
    · intro x hx
      exact (branchFn_hasDerivAt (by linarith [hx.1, hr.1]) hx.2).continuousAt.continuousWithinAt
    · intro x hx
      rw [interior_Ico] at hx
      rw [(branchFn_hasDerivAt (by linarith [hx.1, hr.1]) hx.2).deriv]
      exact branchFnDeriv_pos_on_outer hA (by rw [abs_of_pos (lt_trans hr.1 hx.1)]; exact hx.1)
        (by rw [abs_of_pos (lt_trans hr.1 hx.1)]; exact hx.2)
  have hmono_neg : StrictMonoOn (branchFn A) (Set.Ioc (-1) (-foldPt A)) := by
    apply strictMonoOn_of_deriv_pos (convex_Ioc _ _)
    · intro x hx
      exact (branchFn_hasDerivAt hx.1 (by linarith [hx.2, hr.1])).continuousAt.continuousWithinAt
    · intro x hx
      rw [interior_Ioc] at hx
      rw [(branchFn_hasDerivAt hx.1 (by linarith [hx.2, hr.1])).deriv]
      have hxneg : x < 0 := by linarith [hx.2, hr.1]
      exact branchFnDeriv_pos_on_outer hA (by rw [abs_of_neg hxneg]; linarith [hx.2])
        (by rw [abs_of_neg hxneg]; linarith [hx.1])
  have hMpos : 0 < foldHeight A := by
    unfold foldHeight
    have hlt := hanti ⟨by linarith [hr.1], by linarith [hr.1]⟩ ⟨by linarith [hr.1], le_rfl⟩ hr.1
    have h0 : branchFn A 0 = 0 := by simp [branchFn, artanh_zero]
    rw [h0] at hlt
    linarith
  exact ⟨hanti, hmono_pos, hmono_neg, hMpos⟩

/-- The three open branch labels `-, 0, +`. -/
inductive Branch : Type
  | neg
  | mid
  | pos
  deriving DecidableEq, Fintype

lemma card_branch : Fintype.card Branch = 3 := rfl

/-- Exact number of branch patterns is `3^n`. -/
theorem branchPattern_card (n : ℕ) : Fintype.card (Fin n → Branch) = 3 ^ n := by
  simp [card_branch]

/-- Exact number of branch patterns with instability index `k` (exact count of `Branch.mid`
coordinates) is `C(n, k) * 2^(n - k)`. -/
theorem branchPattern_index_card (n k : ℕ) :
    (Finset.univ.filter (fun σ : Fin n → Branch =>
      (Finset.univ.filter (fun i : Fin n => σ i = Branch.mid)).card = k)).card =
      n.choose k * 2 ^ (n - k) := by
  -- Biject with pairs (S, τ) where S : Finset (Fin n) has |S| = k and τ : {i // i ∉ S} → Bool
  have hfib : (Finset.univ.filter (fun σ : Fin n → Branch =>
      (Finset.univ.filter (fun i : Fin n => σ i = Branch.mid)).card = k)).card =
      ∑ S ∈ (Finset.univ : Finset (Fin n)).powersetCard k,
        (Finset.univ.filter (fun σ : Fin n → Branch =>
          Finset.univ.filter (fun i : Fin n => σ i = Branch.mid) = S)).card := by
    rw [← Finset.card_biUnion]
    · congr 1
      ext σ
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_biUnion,
        Finset.mem_powersetCard, Finset.subset_univ]
      constructor
      · intro h; exact ⟨Finset.univ.filter (fun i => σ i = Branch.mid), h, rfl⟩
      · rintro ⟨S, hScard, hSeq⟩; rw [hSeq]; exact hScard
    · intro S₁ _ S₂ _ hne
      change Disjoint _ _
      rw [Finset.disjoint_left]
      intro σ h1 h2
      exact hne ((Finset.mem_filter.mp h1).2.symm.trans (Finset.mem_filter.mp h2).2)
  rw [hfib]
  have hconst : ∀ S ∈ (Finset.univ : Finset (Fin n)).powersetCard k,
      (Finset.univ.filter (fun σ : Fin n → Branch =>
        Finset.univ.filter (fun i : Fin n => σ i = Branch.mid) = S)).card = 2 ^ (n - k) := by
    intro S hS
    have hScard : S.card = k := (Finset.mem_powersetCard.mp hS).2
    have hcomp : (Finset.univ \ S).card = n - k := by
      rw [Finset.card_sdiff_of_subset (Finset.subset_univ S), Finset.card_univ, Fintype.card_fin, hScard]
    rw [← hcomp]
    have hbij : (Finset.univ.filter (fun σ : Fin n → Branch =>
        Finset.univ.filter (fun i : Fin n => σ i = Branch.mid) = S)).card =
        (Finset.univ : Finset ({i : Fin n // i ∈ Finset.univ \ S} → Bool)).card := by
      refine Finset.card_bij
        (fun σ _ (i : {i : Fin n // i ∈ Finset.univ \ S}) => decide (σ i.1 = Branch.pos))
        (fun _ _ => Finset.mem_univ _) ?_ ?_
      · intro σ₁ h1 σ₂ h2 heq
        have hS1 := (Finset.mem_filter.mp h1).2
        have hS2 := (Finset.mem_filter.mp h2).2
        funext i
        by_cases hi : i ∈ S
        · have hm1 : σ₁ i = Branch.mid := by
            have : i ∈ Finset.univ.filter (fun j => σ₁ j = Branch.mid) := by rw [hS1]; exact hi
            exact (Finset.mem_filter.mp this).2
          have hm2 : σ₂ i = Branch.mid := by
            have : i ∈ Finset.univ.filter (fun j => σ₂ j = Branch.mid) := by rw [hS2]; exact hi
            exact (Finset.mem_filter.mp this).2
          rw [hm1, hm2]
        · have hi' : i ∈ Finset.univ \ S := Finset.mem_sdiff.mpr ⟨Finset.mem_univ _, hi⟩
          have hnm1 : σ₁ i ≠ Branch.mid := by
            intro hm
            have : i ∈ S := by rw [← hS1]; exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, hm⟩
            exact hi this
          have hnm2 : σ₂ i ≠ Branch.mid := by
            intro hm
            have : i ∈ S := by rw [← hS2]; exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, hm⟩
            exact hi this
          have hdec := congrFun heq ⟨i, hi'⟩
          dsimp only at hdec
          cases hσ1 : σ₁ i <;> cases hσ2 : σ₂ i <;> simp_all
      · intro τ _
        let σ : Fin n → Branch := fun i =>
          if hi : i ∈ S then Branch.mid
          else if τ ⟨i, Finset.mem_sdiff.mpr ⟨Finset.mem_univ _, hi⟩⟩ then Branch.pos else Branch.neg
        refine ⟨σ, Finset.mem_filter.mpr ⟨Finset.mem_univ _, ?_⟩, ?_⟩
        · ext i
          simp only [Finset.mem_filter, Finset.mem_univ, true_and, σ]
          by_cases hi : i ∈ S
          · simp [hi]
          · simp only [hi, ↓reduceDIte]
            cases τ ⟨i, Finset.mem_sdiff.mpr ⟨Finset.mem_univ _, hi⟩⟩ <;> simp
        · funext ⟨i, hi⟩
          have hni : i ∉ S := (Finset.mem_sdiff.mp hi).2
          simp only [σ, hni, ↓reduceDIte]
          cases hτ : τ ⟨i, hi⟩ <;> simp
    rw [hbij, Finset.card_univ, Fintype.card_fun, Fintype.card_bool, Fintype.card_coe]
  rw [Finset.sum_congr rfl hconst, Finset.sum_const, Finset.card_powersetCard,
    Finset.card_univ, Fintype.card_fin, smul_eq_mul]

/-- For `A > 1`, `f_A(t) = 0` has a unique root in `(r_A, 1)` (hence exactly 3 roots in `(-1, 1)`
by oddness and `f_A(0) = 0`). -/
theorem branchFn_three_roots {A : ℝ} (hA : 1 < A) :
    (∃! t, t ∈ Set.Ioo (foldPt A) 1 ∧ branchFn A t = 0) ∧
    (∃! t, t ∈ Set.Ioo (-foldPt A) (foldPt A) ∧ branchFn A t = 0) ∧
    (∃! t, t ∈ Set.Ioo (-1) (-foldPt A) ∧ branchFn A t = 0) := by
  have hr := foldPt_bounds hA
  have hmono := branchFn_monotonicity_and_foldHeight hA
  -- Positive root via IVT on [r_A, t₁] where t₁ = (e^(2A) - 1) / (e^(2A) + 1) ∈ (0, 1)
  have hpos_root : ∃! t, t ∈ Set.Ioo (foldPt A) 1 ∧ branchFn A t = 0 := by
    -- Choose t₁ close enough to 1 that artanh t₁ > A, namely t₁ = max (foldPt A) ((e^(2A) - 1)/(e^(2A) + 1))
    set E := Real.exp (2 * A)
    have hE_gt1 : 1 < E := Real.one_lt_exp_iff.mpr (by linarith)
    set t₀ := (E - 1) / (E + 1)
    have ht0_lt1 : t₀ < 1 := by
      dsimp [t₀]; rw [div_lt_one (by linarith)]; linarith
    have ht0_pos : 0 < t₀ := div_pos (by linarith) (by linarith)
    have hartanh_t0 : artanh t₀ = A := by
      unfold artanh
      have hratio : (1 + t₀) / (1 - t₀) = E := by
        dsimp [t₀]; field_simp [show E + 1 ≠ 0 by linarith]; ring
      rw [hratio, Real.log_exp]; ring
    have hft0_pos : 0 < branchFn A t₀ := by
      unfold branchFn; rw [hartanh_t0]; nlinarith
    set t₁ := max (foldPt A) t₀
    have ht1_lt1 : t₁ < 1 := max_lt hr.2.1 ht0_lt1
    have hfr_neg : branchFn A (foldPt A) < 0 := by
      have := hmono.2.2.2; unfold foldHeight at this; linarith
    have ht0_gt_r : foldPt A < t₀ := by
      by_contra hle; push_neg at hle
      have hle_val := hmono.1.antitoneOn ⟨by linarith [hr.1], by linarith [hr.1]⟩
        ⟨by linarith [hr.1], hle⟩ (le_of_lt ht0_pos)
      have h0 : branchFn A 0 = 0 := by simp [branchFn, artanh_zero]
      linarith
    have hcont : ContinuousOn (branchFn A) (Set.Icc (foldPt A) t₀) := fun x hx =>
      (branchFn_hasDerivAt (by linarith [hx.1, hr.1]) (by linarith [hx.2, ht0_lt1])).continuousAt.continuousWithinAt
    have hivt := intermediate_value_Icc (le_of_lt ht0_gt_r) hcont
    obtain ⟨t, ht_mem, hft⟩ := hivt ⟨le_of_lt hfr_neg, le_of_lt hft0_pos⟩
    have ht_gt : foldPt A < t := lt_of_le_of_ne ht_mem.1 (fun heq => by rw [← heq] at hft; linarith)
    refine ⟨t, ⟨⟨ht_gt, lt_of_le_of_lt ht_mem.2 ht0_lt1⟩, hft⟩, fun t' ⟨ht'_mem, hft'⟩ => ?_⟩
    exact hmono.2.1.injOn ⟨le_of_lt ht'_mem.1, ht'_mem.2⟩ ⟨le_of_lt ht_gt, lt_of_le_of_lt ht_mem.2 ht0_lt1⟩
      (hft'.trans hft.symm)
  refine ⟨hpos_root, ?_, ?_⟩
  · refine ⟨0, ⟨⟨by linarith [hr.1], hr.1⟩, by simp [branchFn, artanh_zero]⟩, fun t ⟨ht, hft⟩ => ?_⟩
    exact hmono.1.injOn ⟨le_of_lt ht.1, le_of_lt ht.2⟩ ⟨by linarith [hr.1], by linarith [hr.1]⟩
      (by simpa [branchFn, artanh_zero] using hft)
  · obtain ⟨t, ⟨ht_mem, hft⟩, huniq⟩ := hpos_root
    refine ⟨-t, ⟨⟨by linarith [ht_mem.2], by linarith [ht_mem.1]⟩, ?_⟩, fun t' ⟨ht'_mem, hft'⟩ => ?_⟩
    · rw [branchFn_neg (by linarith [hr.1, ht_mem.1]) ht_mem.2, hft, neg_zero]
    · have hneg : -t' = t := huniq (-t') ⟨⟨by linarith [ht'_mem.2], by linarith [ht'_mem.1]⟩,
        by rw [branchFn_neg ht'_mem.1 (by linarith [hr.1, ht'_mem.2]), hft', neg_zero]⟩
      linarith

/-! ### 2. Weighted Gershgorin Bound and Linear Stability of All `2^n` Outer Equilibria -/

/-- **Weighted Gershgorin positive definiteness**: If `M` is a real symmetric `n × n` matrix
and `w : Fin n → ℝ` satisfies `w i > 0` and `M i i * w i > ∑ j ≠ i, |M i j| * w j` for all `i`,
then `M` is strictly positive definite (`∀ x ≠ 0, ∑ i, ∑ j, x i * M i j * x j > 0`). -/
theorem weighted_gershgorin_posDef {n : ℕ} {M : Matrix (Fin n) (Fin n) ℝ}
    (hsym : M.IsSymm) {w : Fin n → ℝ} (hw : ∀ i, 0 < w i)
    (hdiag : ∀ i, ∑ j ∈ Finset.univ.erase i, |M i j| * w j < M i i * w i)
    {x : Fin n → ℝ} (hx : x ≠ 0) :
    0 < ∑ i : Fin n, ∑ j : Fin n, x i * M i j * x j := by
  let y : Fin n → ℝ := fun i => x i / w i
  have hxy : ∀ i, x i = w i * y i := fun i => (mul_div_cancel₀ (x i) (ne_of_gt (hw i))).symm
  have hquad : ∑ i : Fin n, ∑ j : Fin n, x i * M i j * x j =
      (∑ i : Fin n, M i i * w i ^ 2 * y i ^ 2) +
      (∑ i : Fin n, ∑ j ∈ Finset.univ.erase i, M i j * w i * w j * y i * y j) := by
    have hsplit : ∀ i : Fin n, ∑ j : Fin n, x i * M i j * x j =
        M i i * w i ^ 2 * y i ^ 2 +
        ∑ j ∈ Finset.univ.erase i, M i j * w i * w j * y i * y j := by
      intro i
      rw [← Finset.add_sum_erase _ _ (Finset.mem_univ i)]
      congr 1
      · rw [hxy i]; ring
      · refine Finset.sum_congr rfl fun j _ => ?_
        rw [hxy i, hxy j]; ring
    rw [Finset.sum_congr rfl (fun i _ => hsplit i), Finset.sum_add_distrib]
  -- Off-diagonal bound: M i j * w i * w j * y i * y j ≥ -|M i j| * w i * w j * (y i ^ 2 + y j ^ 2) / 2
  have hoff : -(∑ i : Fin n, ∑ j ∈ Finset.univ.erase i,
      |M i j| * w i * w j * (y i ^ 2 + y j ^ 2) / 2) ≤
      ∑ i : Fin n, ∑ j ∈ Finset.univ.erase i, M i j * w i * w j * y i * y j := by
    rw [← Finset.sum_neg_distrib]
    refine Finset.sum_le_sum fun i _ => ?_
    rw [← Finset.sum_neg_distrib]
    refine Finset.sum_le_sum fun j _ => ?_
    have hwi : 0 < w i := hw i
    have hwj : 0 < w j := hw j
    have habs : -|M i j * y i * y j| ≤ M i j * y i * y j := neg_abs_le _
    have ham : |y i * y j| ≤ (y i ^ 2 + y j ^ 2) / 2 := by
      rw [abs_mul]
      have hsq : 0 ≤ (|y i| - |y j|) ^ 2 := sq_nonneg _
      nlinarith [sq_abs (y i), sq_abs (y j)]
    have hmul : |M i j * y i * y j| ≤ |M i j| * ((y i ^ 2 + y j ^ 2) / 2) := by
      rw [mul_assoc, abs_mul]
      exact mul_le_mul_of_nonneg_left ham (abs_nonneg _)
    nlinarith [mul_pos hwi hwj]
  -- By symmetry of M, the two halves of (y i ^ 2 + y j ^ 2) sum equally
  have hsym_sum : ∑ i : Fin n, ∑ j ∈ Finset.univ.erase i,
      |M i j| * w i * w j * (y i ^ 2 + y j ^ 2) / 2 =
      ∑ i : Fin n, (∑ j ∈ Finset.univ.erase i, |M i j| * w j) * w i * y i ^ 2 := by
    have hsplit2 : ∀ i : Fin n,
        ∑ j ∈ Finset.univ.erase i, |M i j| * w i * w j * (y i ^ 2 + y j ^ 2) / 2 =
        (∑ j ∈ Finset.univ.erase i, |M i j| * w i * w j * y i ^ 2 / 2) +
        (∑ j ∈ Finset.univ.erase i, |M i j| * w i * w j * y j ^ 2 / 2) := by
      intro i
      rw [← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl fun j _ => by ring
    rw [Finset.sum_congr rfl (fun i _ => hsplit2 i), Finset.sum_add_distrib]
    have hswap : ∑ i : Fin n, ∑ j ∈ Finset.univ.erase i, |M i j| * w i * w j * y j ^ 2 / 2 =
        ∑ i : Fin n, ∑ j ∈ Finset.univ.erase i, |M i j| * w i * w j * y i ^ 2 / 2 := by
      have hrewrite : ∀ (f : Fin n → Fin n → ℝ),
          ∑ i : Fin n, ∑ j ∈ Finset.univ.erase i, f i j =
          ∑ i : Fin n, ∑ j : Fin n, if j = i then 0 else f i j := by
        intro f
        refine Finset.sum_congr rfl fun i _ => ?_
        rw [← Finset.add_sum_erase _ _ (Finset.mem_univ i), if_pos rfl, zero_add]
        refine Finset.sum_congr rfl fun j hj => by rw [if_neg (Finset.ne_of_mem_erase hj)]
      rw [hrewrite, hrewrite, Finset.sum_comm]
      refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
      by_cases hij : i = j
      · subst hij; simp
      · have hji : j ≠ i := Ne.symm hij
        rw [if_neg hij, if_neg hji]
        have hM : M j i = M i j := congrFun (congrFun hsym i) j
        rw [hM]; ring
    rw [hswap, ← two_mul, Finset.mul_sum]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [Finset.sum_mul, Finset.sum_mul, Finset.mul_sum]
    refine Finset.sum_congr rfl fun j _ => by ring
  have hlower : ∑ i : Fin n, (M i i * w i - ∑ j ∈ Finset.univ.erase i, |M i j| * w j) * w i * y i ^ 2 ≤
      ∑ i : Fin n, ∑ j : Fin n, x i * M i j * x j := by
    rw [hquad]
    have hcomb : ∑ i : Fin n, (M i i * w i - ∑ j ∈ Finset.univ.erase i, |M i j| * w j) * w i * y i ^ 2 =
        (∑ i : Fin n, M i i * w i ^ 2 * y i ^ 2) -
        (∑ i : Fin n, (∑ j ∈ Finset.univ.erase i, |M i j| * w j) * w i * y i ^ 2) := by
      rw [← Finset.sum_sub_distrib]
      refine Finset.sum_congr rfl fun i _ => by ring
    rw [hcomb, ← hsym_sum]
    linarith
  have hy_ne : ∃ i₀ : Fin n, y i₀ ≠ 0 := by
    by_contra hall
    push_neg at hall
    apply hx
    funext i
    rw [hxy i, hall i, mul_zero]; rfl
  obtain ⟨i₀, hi₀⟩ := hy_ne
  have hstrict : 0 < ∑ i : Fin n,
      (M i i * w i - ∑ j ∈ Finset.univ.erase i, |M i j| * w j) * w i * y i ^ 2 := by
    refine Finset.sum_pos' (fun i _ => ?_) ⟨i₀, Finset.mem_univ _, ?_⟩
    · exact mul_nonneg (mul_nonneg (sub_nonneg.mpr (le_of_lt (hdiag i))) (le_of_lt (hw i))) (sq_nonneg _)
    · exact mul_pos (mul_pos (sub_pos.mpr (hdiag i₀)) (hw i₀)) (sq_pos_of_ne_zero hi₀)
  linarith

/-- **Exact midpoint derivative identity (Step 2 of Theorem 1)**:
At `m = (1 + r)/2`, `f'(m) = A r + 1/(3 + r) > M_A`. -/
theorem midpoint_deriv_gt_foldHeight {A : ℝ} (hA : 1 < A) :
    let r := foldPt A
    let m := (1 + r) / 2
    branchFnDeriv A m = A * r + 1 / (3 + r) ∧
    foldHeight A < branchFnDeriv A m := by
  intro r m
  have hr := foldPt_bounds hA
  have h1r : 0 < 1 - r := sub_pos.mpr hr.2.1
  have h1pr : 0 < 1 + r := by linarith [hr.1]
  have h3r : 0 < 3 + r := by linarith [hr.1]
  have h1r2 : 1 - r ^ 2 ≠ 0 := by
    rw [show 1 - r ^ 2 = (1 - r) * (1 + r) by ring]
    exact mul_ne_zero (ne_of_gt h1r) (ne_of_gt h1pr)
  have hid : branchFnDeriv A m = A * r + 1 / (3 + r) := by
    unfold branchFnDeriv
    have hAr : A = 1 / (1 - r ^ 2) := hr.2.2.2
    rw [hAr]
    dsimp [m]
    have hden : 1 - ((1 + r) / 2) ^ 2 = (1 - r) * (3 + r) / 4 := by ring
    rw [hden]
    have hne1 : 1 - r ≠ 0 := ne_of_gt h1r
    have hne2 : 3 + r ≠ 0 := ne_of_gt h3r
    have hne3 : 1 + r ≠ 0 := ne_of_gt h1pr
    field_simp [hne1, hne2, hne3, h1r2]
    ring
  refine ⟨hid, ?_⟩
  rw [hid]
  unfold foldHeight branchFn
  -- foldHeight A = A * r - artanh r < A * r + 1/(3 + r) since artanh r > 0 and 1/(3 + r) > 0
  have hartanh_pos : 0 < artanh r := by
    unfold artanh
    have hgt1 : 1 < (1 + r) / (1 - r) := by
      rw [one_lt_div h1r]; linarith [hr.1]
    exact mul_pos (by norm_num) (Real.log_pos hgt1)
  have hpos3 : 0 < 1 / (3 + r) := div_pos one_pos h3r
  linarith

/-- **Step 2 & Step 3 (Weighted Gershgorin bound + Linear stability of all `2^n` outer equilibria)**:
Let `u : Fin n → ℝ` be an outer-only equilibrium (`r_A < |u_i| < 1` and `f_A(u_i) = (B u)_i`
for all `i`), where `B` is symmetric with `B_{ij} ≥ 0` and row sums `β < M_A`. Then:
1. For `w_i = min(|u_i| - r_A, 1 - |u_i|) > 0`, `f_A'(|u_i|) w_i > ∑_j B_{ij} w_j` for all `i`.
2. `H = diag(f_A'(u_i)) - B` is strictly positive definite.
3. The symmetrized Jacobian `-D^{1/2} H D^{1/2}` is strictly negative definite on `ℝ^n`.
4. The complementary `L^2` contraction factor satisfies `A * (1 - u_i^2) < 1` for every `i`. -/
theorem outer_equilibrium_linear_stability {n : ℕ} {A β : ℝ} (hA : 1 < A)
    (hβ : β < foldHeight A)
    {B : Matrix (Fin n) (Fin n) ℝ} (hBsym : B.IsSymm)
    (hBnn : ∀ i j, 0 ≤ B i j) (hBrow : ∀ i, ∑ j : Fin n, B i j = β)
    {u : Fin n → ℝ}
    (hu_outer : ∀ i, foldPt A < |u i| ∧ |u i| < 1)
    (hu_eq : ∀ i, branchFn A (u i) = ∑ j : Fin n, B i j * u j) :
    let H : Matrix (Fin n) (Fin n) ℝ := Matrix.diagonal (fun i => branchFnDeriv A (u i)) - B
    (∀ x : Fin n → ℝ, x ≠ 0 → 0 < ∑ i, ∑ j, x i * H i j * x j) ∧
    (∀ v : Fin n → ℝ, v ≠ 0 →
      ∑ i, ∑ j, v i * (-(Real.sqrt (1 - u i ^ 2) * H i j * Real.sqrt (1 - u j ^ 2))) * v j < 0) ∧
    (∀ i : Fin n, A * (1 - u i ^ 2) < 1) := by
  intro H
  have hr := foldPt_bounds hA
  have hmid := midpoint_deriv_gt_foldHeight hA
  set r := foldPt A
  set m := (1 + r) / 2
  set d := (1 - r) / 2
  let t : Fin n → ℝ := fun i => |u i|
  let w : Fin n → ℝ := fun i => min (t i - r) (1 - t i)
  have hw_pos : ∀ i, 0 < w i := fun i => lt_min (sub_pos.mpr (hu_outer i).1) (sub_pos.mpr (hu_outer i).2)
  have hw_le_d : ∀ i, w i ≤ d := fun i => by
    dsimp [w, d]
    by_cases htm : t i ≤ m
    · dsimp [m] at htm; exact le_trans (min_le_left _ _) (by linarith)
    · dsimp [m] at htm; exact le_trans (min_le_right _ _) (by linarith)
  -- Strict convexity of branchFn A on [r, 1): f'(t) (t - r) > f(t) - f(r) for r < t < 1
  have hconvex : ∀ x ∈ Set.Ioo r 1,
      branchFn A x - branchFn A r < branchFnDeriv A x * (x - r) := by
    intro x ⟨hxlo, hxhi⟩
    have hcont : ContinuousOn (branchFn A) (Set.Icc r x) := fun z hz =>
      (branchFn_hasDerivAt (by linarith [hz.1, hr.1]) (by linarith [hz.2, hxhi])).continuousAt.continuousWithinAt
    have hdiff : DifferentiableOn ℝ (branchFn A) (Set.Ioo r x) := fun z hz =>
      (branchFn_hasDerivAt (by linarith [hz.1, hr.1]) (by linarith [hz.2, hxhi])).differentiableAt.differentiableWithinAt
    obtain ⟨ξ, ⟨hξlo, hξhi⟩, hξ⟩ := exists_deriv_eq_slope (branchFn A) hxlo hcont hdiff
    rw [(branchFn_hasDerivAt (by linarith [hξlo, hr.1]) (by linarith [hξhi, hxhi])).deriv] at hξ
    have hderiv_lt : branchFnDeriv A ξ < branchFnDeriv A x := by
      unfold branchFnDeriv
      have hξpos : 0 < ξ := lt_trans hr.1 hξlo
      have hx2 : ξ ^ 2 < x ^ 2 := by nlinarith
      have h1x : 0 < 1 - x ^ 2 := by nlinarith
      have h1ξ : 0 < 1 - ξ ^ 2 := by nlinarith
      have : 1 / (1 - ξ ^ 2) < 1 / (1 - x ^ 2) :=
        one_div_lt_one_div_of_lt h1x (by linarith)
      linarith
    rw [eq_div_iff (ne_of_gt (sub_pos.mpr hxlo))] at hξ
    nlinarith
  -- Weighted Gershgorin inequality: f'(t_i) * w_i > ∑ j, B i j * w_j
  have hwg : ∀ i : Fin n, ∑ j : Fin n, B i j * w j < branchFnDeriv A (u i) * w i := by
    intro i
    have hderiv_eq : branchFnDeriv A (u i) = branchFnDeriv A (t i) := by
      unfold branchFnDeriv; dsimp [t]; rw [sq_abs]
    rw [hderiv_eq]
    by_cases htm : t i ≤ m
    · have hwi : w i = t i - r := min_eq_left (by dsimp [m] at htm; linarith)
      rw [hwi]
      have h1 := hconvex (t i) (hu_outer i)
      -- f(t_i) - f(r) = M_A + sgn(u_i) * (B u)_i
      have hft : branchFn A (t i) = if 0 ≤ u i then ∑ j, B i j * u j else -(∑ j, B i j * u j) := by
        dsimp [t]
        by_cases hui : 0 ≤ u i
        · rw [if_pos hui, abs_of_nonneg hui, hu_eq i]
        · push_neg at hui
          rw [if_neg (not_le.mpr hui), abs_of_neg hui,
              branchFn_neg (by linarith [(abs_lt.mp (hu_outer i).2).1]) (by linarith [hui]), hu_eq i]
      have hsum_ge : ∑ j : Fin n, B i j * w j ≤
          β + (if 0 ≤ u i then ∑ j, B i j * u j else -(∑ j, B i j * u j)) := by
        rw [← hBrow i]
        by_cases hui : 0 ≤ u i
        · rw [if_pos hui, ← Finset.sum_add_distrib]
          refine Finset.sum_le_sum fun j _ => ?_
          have h1u : w j ≤ 1 + u j := by
            have h1 : w j ≤ 1 - |u j| := min_le_right _ _
            have h2 : -u j ≤ |u j| := neg_le_abs (u j)
            linarith
          have := mul_le_mul_of_nonneg_left h1u (hBnn i j)
          linarith
        · rw [if_neg hui, ← sub_eq_add_neg, ← Finset.sum_sub_distrib]
          refine Finset.sum_le_sum fun j _ => ?_
          have h1u : w j ≤ 1 - u j := by
            have h1 : w j ≤ 1 - |u j| := min_le_right _ _
            have h2 : u j ≤ |u j| := le_abs_self (u j)
            linarith
          have := mul_le_mul_of_nonneg_left h1u (hBnn i j)
          linarith
      have hM : -branchFn A r = foldHeight A := rfl
      linarith
    · push_neg at htm
      have hwi : w i = 1 - t i := min_eq_right (by dsimp [m] at htm; linarith)
      rw [hwi]
      -- (1 - x) * f'(x) = 1/(1 + x) - A(1 - x) is strictly increasing on [m, 1)
      have hinc : (1 - m) * branchFnDeriv A m ≤ (1 - t i) * branchFnDeriv A (t i) := by
        have hform : ∀ x : ℝ, x ≠ 1 → x ≠ -1 → (1 - x) * branchFnDeriv A x = 1 / (1 + x) - A * (1 - x) := by
          intro x hx1 hx2
          unfold branchFnDeriv
          have hne : 1 - x ^ 2 ≠ 0 := by
            rw [show 1 - x ^ 2 = (1 - x) * (1 + x) by ring]
            exact mul_ne_zero (sub_ne_zero.mpr hx1.symm) (by intro h; apply hx2; linarith)
          field_simp [hne, show 1 + x ≠ 0 by intro h; apply hx2; linarith]
          ring
        have hm_lt1 : m < 1 := by dsimp [m]; linarith [hr.2.1]
        have hm_pos : 0 < m := by dsimp [m]; linarith [hr.1]
        have hti_lt1 : t i < 1 := (hu_outer i).2
        have hti_pos : 0 < t i := lt_trans hr.1 (hu_outer i).1
        rw [hform m (ne_of_lt hm_lt1) (by linarith), hform (t i) (ne_of_lt hti_lt1) (by linarith)]
        -- 1/(1 + m) - A(1 - m) ≤ 1/(1 + t i) - A(1 - t i) ↔ (t i - m) * (A - 1/((1+m)(1+t i))) ≥ 0
        have hdiff : (1 / (1 + t i) - A * (1 - t i)) - (1 / (1 + m) - A * (1 - m)) =
            (t i - m) * (A - 1 / ((1 + m) * (1 + t i))) := by
          field_simp [show (1 + m : ℝ) ≠ 0 by linarith, show (1 + t i : ℝ) ≠ 0 by linarith]
          ring
        have hprod_gt1 : 1 < (1 + m) * (1 + t i) := by nlinarith
        have hinv_lt1 : 1 / ((1 + m) * (1 + t i)) < 1 := by
          rw [div_lt_one (by linarith)]; exact hprod_gt1
        have hnonneg : 0 ≤ (t i - m) * (A - 1 / ((1 + m) * (1 + t i))) :=
          mul_nonneg (by linarith) (by linarith)
        linarith
      have hsum_le : ∑ j : Fin n, B i j * w j ≤ d * β := by
        calc ∑ j : Fin n, B i j * w j
            ≤ ∑ j : Fin n, B i j * d :=
              Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (hw_le_d j) (hBnn i j)
          _ = (∑ j : Fin n, B i j) * d := (Finset.sum_mul _ _ _).symm
          _ = d * β := by rw [hBrow i]; ring
      have hd_pos : 0 < d := by dsimp [d]; linarith [hr.2.1]
      have h1m : 1 - m = d := by dsimp [m, d]; ring
      rw [h1m] at hinc
      have hd_strict : d * β < d * branchFnDeriv A m :=
        mul_lt_mul_of_pos_left (lt_trans hβ hmid.2) hd_pos
      linarith
  -- Part 1: H is strictly positive definite via weighted_gershgorin_posDef
  have hHsym : H.IsSymm := by
    dsimp [H]
    exact (Matrix.isSymm_diagonal _).sub hBsym
  have hHpos : ∀ x : Fin n → ℝ, x ≠ 0 → 0 < ∑ i, ∑ j, x i * H i j * x j := by
    intro x hx
    apply weighted_gershgorin_posDef hHsym hw_pos _ hx
    intro i
    have hdiag_i : H i i = branchFnDeriv A (u i) - B i i := by simp [H]
    have hoff_i : ∑ j ∈ Finset.univ.erase i, |H i j| * w j =
        ∑ j ∈ Finset.univ.erase i, B i j * w j := by
      refine Finset.sum_congr rfl fun j hj => ?_
      have hne : j ≠ i := Finset.ne_of_mem_erase hj
      simp [H, Matrix.diagonal_apply_ne _ hne.symm, abs_of_nonneg (hBnn i j)]
    rw [hdiag_i, hoff_i]
    have hsplit_B := Finset.add_sum_erase Finset.univ (fun j => B i j * w j) (Finset.mem_univ i)
    linarith [hwg i]
  -- Part 2 & Part 3: Symmetrized Jacobian and L^2 contraction
  refine ⟨hHpos, ?_, ?_⟩
  · intro v hv
    let x : Fin n → ℝ := fun i => Real.sqrt (1 - u i ^ 2) * v i
    have hx_ne : x ≠ 0 := by
      intro hx0
      apply hv
      funext i
      have h1u : 0 < 1 - u i ^ 2 := by
        have hlt := (hu_outer i).2
        have : |u i| ^ 2 < 1 := by nlinarith [abs_nonneg (u i)]
        rw [sq_abs] at this; linarith
      have hsq_pos : 0 < Real.sqrt (1 - u i ^ 2) := Real.sqrt_pos.mpr h1u
      have hxi : Real.sqrt (1 - u i ^ 2) * v i = 0 := congrFun hx0 i
      exact (mul_eq_zero.mp hxi).resolve_left (ne_of_gt hsq_pos)
    have hpos := hHpos x hx_ne
    have hrew : ∑ i, ∑ j, v i * (-(Real.sqrt (1 - u i ^ 2) * H i j * Real.sqrt (1 - u j ^ 2))) * v j =
        -(∑ i, ∑ j, x i * H i j * x j) := by
      rw [← Finset.sum_neg_distrib]
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [← Finset.sum_neg_distrib]
      refine Finset.sum_congr rfl fun j _ => by dsimp [x]; ring
    linarith
  · intro i
    have hgt := (hu_outer i).1
    have hsq : foldPt A ^ 2 < u i ^ 2 := by
      have : foldPt A ^ 2 < |u i| ^ 2 := by nlinarith [hr.1]
      rwa [sq_abs] at this
    have hlt : 1 - u i ^ 2 < 1 / A := by linarith [hr.2.2.1]
    have hApos : 0 < A := by linarith
    have := mul_lt_mul_of_pos_left hlt hApos
    rwa [mul_one_div_cancel (ne_of_gt hApos)] at this

/-! ### 3. Refutation Core for `Conj_random` (`0 < ρ(v) ≤ 1/64` with `≥ 3` Equilibria) -/

/-- Exact determinant of a `2 × 2` pair block `I - S_r`. -/
theorem det_two_by_two_block (x y : ℝ) :
    Matrix.det ![![(1 : ℝ), -x], ![-y, 1]] = 1 - x * y := by
  simp [Matrix.det_fin_two]

/-- **Constructive 3-equilibria witness for `det(I - M) = 1 - x y < 0`**:
Whenever `x > 0`, `y > 0`, and `x * y > 1`, the `2 × 2` system
`h₁ = x tanh h₂`, `h₂ = y tanh h₁` has at least three distinct equilibria
`(0, 0)`, `(h₁*, h₂*)`, and `(-h₁*, -h₂*)` with `h₁* > 0, h₂* > 0`. -/
theorem two_by_two_three_equilibria {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hxy : 1 < x * y) :
    ∃ h₁ h₂ : ℝ, 0 < h₁ ∧ 0 < h₂ ∧
      (0 : ℝ) = x * Real.tanh 0 ∧ (0 : ℝ) = y * Real.tanh 0 ∧
      h₁ = x * Real.tanh h₂ ∧ h₂ = y * Real.tanh h₁ ∧
      -h₁ = x * Real.tanh (-h₂) ∧ -h₂ = y * Real.tanh (-h₁) := by
  let g : ℝ → ℝ := fun t => x * Real.tanh (y * Real.tanh t) - t
  have htanh_lt_one : ∀ z : ℝ, Real.tanh z < 1 := by
    intro z
    rw [Real.tanh_eq_sinh_div_cosh, div_lt_one (Real.cosh_pos z)]
    rw [Real.sinh_eq, Real.cosh_eq]
    linarith [Real.exp_pos (-z)]
  have htanh_pos : ∀ {z : ℝ}, 0 < z → 0 < Real.tanh z := by
    intro z hz
    rw [Real.tanh_eq_sinh_div_cosh]
    exact div_pos (Real.sinh_pos_iff.mpr hz) (Real.cosh_pos z)
  have htanh_deriv : ∀ z : ℝ, HasDerivAt Real.tanh (1 / Real.cosh z ^ 2) z := by
    intro z
    have heq : Real.tanh = fun w => Real.sinh w / Real.cosh w := funext Real.tanh_eq_sinh_div_cosh
    rw [heq]
    have hd := (Real.hasDerivAt_sinh z).div (Real.hasDerivAt_cosh z) (ne_of_gt (Real.cosh_pos z))
    convert hd using 1
    have hcs := Real.cosh_sq_sub_sinh_sq z
    field_simp [ne_of_gt (Real.cosh_pos z)]
    linarith
  have hg_deriv0 : HasDerivAt g (x * y - 1) 0 := by
    have h1 : HasDerivAt (fun t => y * Real.tanh t) (y * (1 / Real.cosh 0 ^ 2)) 0 :=
      (htanh_deriv 0).const_mul y
    have h2 : HasDerivAt (fun t => Real.tanh (y * Real.tanh t))
        ((1 / Real.cosh (y * Real.tanh 0) ^ 2) * (y * (1 / Real.cosh 0 ^ 2))) 0 := by
      have hc := (htanh_deriv (y * Real.tanh 0)).comp 0 h1
      simpa [Function.comp_def] using hc
    have h3 := (h2.const_mul x).sub (hasDerivAt_id (0 : ℝ))
    convert h3 using 1
    simp [Real.tanh_zero, Real.cosh_zero]
  have hslope := hg_deriv0.tendsto_slope_zero_right
  have hev := hslope.eventually (lt_mem_nhds (sub_pos.mpr hxy))
  have hev_lt : ∀ᶠ t in nhdsWithin (0 : ℝ) (Set.Ioi 0), t < x :=
    (gt_mem_nhds hx).filter_mono nhdsWithin_le_nhds
  have hself : ∀ᶠ t in nhdsWithin (0 : ℝ) (Set.Ioi 0), t ∈ Set.Ioi (0 : ℝ) :=
    self_mem_nhdsWithin
  obtain ⟨ε, ⟨hε_pos, hε_slope⟩, hε_lt⟩ := ((hself.and hev).and hev_lt).exists
  have hg0 : g 0 = 0 := by simp [g, Real.tanh_zero]
  have hgε_pos : 0 < g ε := by
    rw [Set.mem_Ioi] at hε_pos
    simp only [smul_eq_mul, zero_add, hg0, sub_zero] at hε_slope
    have h1 : 0 < ε⁻¹ * g ε := hε_slope
    have := mul_pos hε_pos h1
    rwa [mul_inv_cancel_left₀ (ne_of_gt hε_pos)] at this
  have hgx_neg : g x < 0 := by
    dsimp [g]
    have hlt := htanh_lt_one (y * Real.tanh x)
    nlinarith
  have hg_cont : ContinuousOn g (Set.Icc ε x) := by
    have htc : Continuous Real.tanh := by
      have : Real.tanh = fun w => Real.sinh w / Real.cosh w := funext Real.tanh_eq_sinh_div_cosh
      rw [this]
      exact Real.continuous_sinh.div Real.continuous_cosh (fun w => ne_of_gt (Real.cosh_pos w))
    exact ((continuous_const.mul (htc.comp (continuous_const.mul htc))).sub continuous_id).continuousOn
  have hivt := intermediate_value_Icc' (le_of_lt hε_lt) hg_cont
  obtain ⟨h₁, hh₁_mem, hgh₁⟩ := hivt ⟨le_of_lt hgx_neg, le_of_lt hgε_pos⟩
  have hh₁_pos : 0 < h₁ := lt_of_lt_of_le (Set.mem_Ioi.mp hε_pos) hh₁_mem.1
  set h₂ := y * Real.tanh h₁
  have hh₂_pos : 0 < h₂ := mul_pos hy (htanh_pos hh₁_pos)
  have heq1 : h₁ = x * Real.tanh h₂ := by
    dsimp [g] at hgh₁; linarith
  refine ⟨h₁, h₂, hh₁_pos, hh₂_pos, by simp [Real.tanh_zero], by simp [Real.tanh_zero],
    heq1, rfl, ?_, ?_⟩
  · rw [Real.tanh_neg, mul_neg, ← heq1]
  · rw [Real.tanh_neg, mul_neg]

/-- **Exact parity probability identity (eq. 3 of `solution.md`)**:
For `m` independent Bernoulli(`q`) pair flips, the probability of an odd count (so `det(I - S) < 0`)
equals `(1 - (1 - 2 q)^m) / 2`. -/
theorem parity_prob_formula (m : ℕ) (q : ℝ) :
    ∑ k ∈ (Finset.range (m + 1)).filter (fun k => k % 2 = 1),
        (m.choose k : ℝ) * q ^ k * (1 - q) ^ (m - k) =
      (1 - (1 - 2 * q) ^ m) / 2 := by
  have hplus : ∑ k ∈ Finset.range (m + 1), (m.choose k : ℝ) * q ^ k * (1 - q) ^ (m - k) = 1 := by
    calc ∑ k ∈ Finset.range (m + 1), (m.choose k : ℝ) * q ^ k * (1 - q) ^ (m - k)
        = ∑ k ∈ Finset.range (m + 1), q ^ k * (1 - q) ^ (m - k) * (m.choose k : ℝ) :=
          Finset.sum_congr rfl fun k _ => by ring
      _ = (q + (1 - q)) ^ m := (add_pow q (1 - q) m).symm
      _ = 1 := by ring
  have hminus : ∑ k ∈ Finset.range (m + 1),
      (m.choose k : ℝ) * (-q) ^ k * (1 - q) ^ (m - k) = (1 - 2 * q) ^ m := by
    calc ∑ k ∈ Finset.range (m + 1), (m.choose k : ℝ) * (-q) ^ k * (1 - q) ^ (m - k)
        = ∑ k ∈ Finset.range (m + 1), (-q) ^ k * (1 - q) ^ (m - k) * (m.choose k : ℝ) :=
          Finset.sum_congr rfl fun k _ => by ring
      _ = (-q + (1 - q)) ^ m := (add_pow (-q) (1 - q) m).symm
      _ = (1 - 2 * q) ^ m := by ring
  have hsub : 2 * ∑ k ∈ (Finset.range (m + 1)).filter (fun k => k % 2 = 1),
      (m.choose k : ℝ) * q ^ k * (1 - q) ^ (m - k) = 1 - (1 - 2 * q) ^ m := by
    calc 2 * ∑ k ∈ (Finset.range (m + 1)).filter (fun k => k % 2 = 1),
          (m.choose k : ℝ) * q ^ k * (1 - q) ^ (m - k)
        = ∑ k ∈ (Finset.range (m + 1)).filter (fun k => k % 2 = 1),
            ((m.choose k : ℝ) * q ^ k * (1 - q) ^ (m - k) -
             (m.choose k : ℝ) * (-q) ^ k * (1 - q) ^ (m - k)) := by
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun k hk => ?_
          have hodd : Odd k := Nat.odd_iff.mpr (Finset.mem_filter.mp hk).2
          rw [hodd.neg_pow]; ring
      _ = (∑ k ∈ (Finset.range (m + 1)).filter (fun k => k % 2 = 1),
            ((m.choose k : ℝ) * q ^ k * (1 - q) ^ (m - k) -
             (m.choose k : ℝ) * (-q) ^ k * (1 - q) ^ (m - k))) +
          (∑ k ∈ (Finset.range (m + 1)).filter (fun k => ¬ k % 2 = 1),
            ((m.choose k : ℝ) * q ^ k * (1 - q) ^ (m - k) -
             (m.choose k : ℝ) * (-q) ^ k * (1 - q) ^ (m - k))) := by
          have heven : ∑ k ∈ (Finset.range (m + 1)).filter (fun k => ¬ k % 2 = 1),
              ((m.choose k : ℝ) * q ^ k * (1 - q) ^ (m - k) -
               (m.choose k : ℝ) * (-q) ^ k * (1 - q) ^ (m - k)) = 0 := by
            refine Finset.sum_eq_zero fun k hk => ?_
            have hmod : k % 2 = 0 := by have := (Finset.mem_filter.mp hk).2; omega
            have hev : Even k := Nat.even_iff.mpr hmod
            rw [hev.neg_pow, sub_self]
          rw [heven, add_zero]
      _ = ∑ k ∈ Finset.range (m + 1),
            ((m.choose k : ℝ) * q ^ k * (1 - q) ^ (m - k) -
             (m.choose k : ℝ) * (-q) ^ k * (1 - q) ^ (m - k)) :=
          Finset.sum_filter_add_sum_filter_not _ _ _
      _ = (∑ k ∈ Finset.range (m + 1), (m.choose k : ℝ) * q ^ k * (1 - q) ^ (m - k)) -
          (∑ k ∈ Finset.range (m + 1), (m.choose k : ℝ) * (-q) ^ k * (1 - q) ^ (m - k)) :=
          Finset.sum_sub_distrib _ _
      _ = 1 - (1 - 2 * q) ^ m := by rw [hplus, hminus]
  linarith

/-- **Asymptotic parity limit `1/2`**: If `0 ≤ q m ≤ 1/2` and `m * q m → +∞`, then
`(1 - (1 - 2 q m)^m) / 2 → 1/2`. -/
theorem parity_prob_tendsto_half {q : ℕ → ℝ}
    (_hq0 : ∀ m, 0 ≤ q m) (hq1 : ∀ m, q m ≤ 1 / 2)
    (hmq : Tendsto (fun m : ℕ => (m : ℝ) * q m) atTop atTop) :
    Tendsto (fun m : ℕ => (1 - (1 - 2 * q m) ^ m) / 2) atTop (nhds (1 / 2)) := by
  have hpow_zero : Tendsto (fun m : ℕ => (1 - 2 * q m) ^ m) atTop (nhds 0) := by
    refine squeeze_zero (g := fun m : ℕ => Real.exp (-(2 * ((m : ℝ) * q m))))
      (fun m => pow_nonneg (by linarith [hq1 m]) m)
      (fun m => ?_) ?_
    · have h1 : 1 - 2 * q m ≤ Real.exp (-(2 * q m)) := by
        have := Real.add_one_le_exp (-(2 * q m)); linarith
      calc (1 - 2 * q m) ^ m ≤ (Real.exp (-(2 * q m))) ^ m :=
            pow_le_pow_left₀ (by linarith [hq1 m]) h1 m
        _ = Real.exp (-(2 * ((m : ℝ) * q m))) := by
            rw [← Real.exp_nat_mul]; congr 1; ring
    · exact Real.tendsto_exp_atBot.comp
        (tendsto_neg_atTop_atBot.comp (hmq.const_mul_atTop (by norm_num : (0 : ℝ) < 2)))
  have hlim : Tendsto (fun m : ℕ => (1 - (1 - 2 * q m) ^ m) / 2) atTop (nhds ((1 - 0) / 2)) :=
    ((tendsto_const_nhds (x := (1 : ℝ))).sub hpow_zero).div_const 2
  simpa using hlim

/-- **Exact `L^1` profile mass budget `0 < ρ(v) ≤ 1/64` (Theorem 4)**:
For any dyadic shell masses `a : ℕ → ℝ` with `0 < a k` and `√(a k) ≤ 2^{-(k+1)} / 64`,
both `∑' k, √(a k) ≤ 1/64` and `0 < ∑' k, a k ≤ 1/4096` (so `0 < ρ(v) ≤ 1/64 < 1`). -/
theorem profile_mass_budget_le_one_div_64 {a : ℕ → ℝ}
    (hpos : ∀ k, 0 < a k) (hbound : ∀ k, Real.sqrt (a k) ≤ (2 : ℝ)⁻¹ ^ (k + 1) / 64) :
    Summable a ∧
    0 < ∑' k, a k ∧
    ∑' k, a k ≤ 1 / 4096 ∧
    0 < Real.sqrt (∑' k, a k) ∧
    Real.sqrt (∑' k, a k) ≤ 1 / 64 := by
  have ha_le : ∀ k, a k ≤ ((4 : ℝ)⁻¹ ^ k) / (4 * 4096) := by
    intro k
    have hsq := hbound k
    have hsq2 : Real.sqrt (a k) ^ 2 ≤ ((2 : ℝ)⁻¹ ^ (k + 1) / 64) ^ 2 :=
      pow_le_pow_left₀ (Real.sqrt_nonneg _) hsq 2
    rw [Real.sq_sqrt (le_of_lt (hpos k))] at hsq2
    have hrhs : ((2 : ℝ)⁻¹ ^ (k + 1) / 64) ^ 2 = (4 : ℝ)⁻¹ ^ k / (4 * 4096) := by
      rw [pow_succ]
      have h24 : ((2 : ℝ)⁻¹ ^ k) ^ 2 = (4 : ℝ)⁻¹ ^ k := by
        rw [← pow_mul, mul_comm k 2, pow_mul]
        norm_num
      calc ((2 : ℝ)⁻¹ ^ k * (2 : ℝ)⁻¹ / 64) ^ 2
          = ((2 : ℝ)⁻¹ ^ k) ^ 2 * ((2 : ℝ)⁻¹ / 64) ^ 2 := by ring
        _ = (4 : ℝ)⁻¹ ^ k / (4 * 4096) := by rw [h24]; norm_num; ring
    linarith
  have hgeom_sum : Summable (fun k : ℕ => (4 : ℝ)⁻¹ ^ k / (4 * 4096)) :=
    (summable_geometric_of_lt_one (by norm_num) (by norm_num)).div_const _
  have ha_sum : Summable a :=
    Summable.of_nonneg_of_le (fun k => le_of_lt (hpos k)) ha_le hgeom_sum
  have htsum_pos : 0 < ∑' k, a k := by
    have hle : a 0 ≤ ∑' k, a k := ha_sum.le_tsum 0 (fun k _ => le_of_lt (hpos k))
    linarith [hpos 0]
  have htsum_le : ∑' k, a k ≤ 1 / 4096 := by
    have h1 := ha_sum.tsum_le_tsum ha_le hgeom_sum
    have h2 : ∑' k : ℕ, (4 : ℝ)⁻¹ ^ k / (4 * 4096) =
        (∑' k : ℕ, (4 : ℝ)⁻¹ ^ k) / (4 * 4096) := tsum_div_const
    rw [h2, tsum_geometric_of_lt_one (by norm_num) (by norm_num)] at h1
    norm_num at h1
    linarith
  have hsqrt_le : Real.sqrt (∑' k, a k) ≤ 1 / 64 := by
    calc Real.sqrt (∑' k, a k) ≤ Real.sqrt (1 / 4096) := Real.sqrt_le_sqrt htsum_le
      _ = 1 / 64 := by
        rw [show (1 / 4096 : ℝ) = (1 / 64) ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
  exact ⟨ha_sum, htsum_pos, htsum_le, Real.sqrt_pos.mpr htsum_pos, hsqrt_le⟩

end PAdicDNNCritical
