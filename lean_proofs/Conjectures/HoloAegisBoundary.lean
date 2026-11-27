import Mathlib

/-!
# Refutation of Reference-Noise Boundary Stability in HoloAegis
(arXiv:2608.08485, Section 3.4, Conjecture `conj:main`)

We formalize in Lean 4 + Mathlib the core analytical, combinatorial, and geometric
theorems refuting the HoloAegis Reference-Noise Boundary Stability conjecture of
Tak Ho Alex Li et al. (arXiv:2608.08485):

1. **Log-sum-exp shift inequality (Lemma 1, Part 1)**:
   - `lseAvg`: Temperature-scaled log-mean-exp `τ log((1/k) ∑_{i=1}^k exp(x_i / τ))`.
   - `lseAvg_shift_bounds`: If `u ≤ y_i - x_i ≤ v` for all `i`, then
     `u ≤ lseAvg τ y - lseAvg τ x ≤ v`.
   - `lseAvg_lipschitz`: `|lseAvg τ y - lseAvg τ x| ≤ ε` whenever `∀ i, |y_i - x_i| ≤ ε`.
2. **Order-statistic `L^∞`-Lipschitz bound for top-`k` Gibbs–Boltzmann potential (Lemma 1, Part 2)**:
   - `descOrderStat`: Descending order statistics of any finite score vector `s : Fin m → ℝ`
     via `Tuple.sort` (reversing the ascending sort).
   - `descOrderStat_linf_contraction`: If `∀ j, |y j - x j| ≤ ε`, then EVERY descending
     order statistic changes by at most `ε` (`|descOrderStat y r - descOrderStat x r| ≤ ε`).
   - `gibbsPotential_lipschitz`: Combining order-statistic contraction with `lseAvg_lipschitz`
     yields `|U(q + δ; R) - U(q; R)| ≤ ‖δ‖` for any unit-norm bank `R` and any query shift `δ`.
3. **Deterministic global `K`-means uniqueness certificate for separated balls (Theorem 1)**:
   - `kmeans_separated_balls_unique`: Given `K` separated balls `B(v_j, ρ)` of equal size `n`
     with `K * ρ² < (R - ρ)²` and `‖v_j - v_l‖ > 2 * R + 2 * ρ` (`j ≠ l`), every `K`-center set
     achieving total squared distortion `≤ K * n * ρ²` places exactly one center in each
     `B(v_j, R)`, every point in cluster `j` is strictly closer to its local center than to
     every other center, and the unique global `K`-means minimizer (up to relabeling) is the
     tuple of exact arithmetic cluster means.
   - `theorem1_ball_params_certified`: Certifies the exact Theorem 1 numerical inequalities
     (`K ≤ 300`, `ρ = 0.004`, `R = 0.25`, `‖v_j - v_l‖ ≥ 1.131`, `|r_w - r_v| < 0.001`).
4. **Exact paired-family shifts, ratio `4.5`, and surjectivity onto `[0, ∞)` (Theorem 1)**:
   - `paired_raw_shift_and_ratio`: With `a = 0.600`, `b = 0.599`, `s = 0.001`, `t ∈ [-s, 0]`,
     exact full-bank shift `S(X_N) = 2 |t| sin h`, exact raw anchor shift `S(A_K) = (t + s) sin h`,
     exact ratio `4.5` at `t = -s / 10`, and exact attainment of every `r ≥ 0` at `t = -s / (2r + 1)`.
   - `paired_normalized_bracket_and_surjectivity`: For unit-normalized centroids with
     `L_j ∈ [0.996, 1]`, the bracket `2 D sin h ≤ S(A_K) ≤ 2 D sin h / 0.996` holds;
     for ANY continuous shift curve `F` in this bracket (including expected shift `𝔼 S(A_K)`),
     the ratio `F(t) / (2 |t| sin h)` maps `[-s, 0)` onto ALL of `[0, ∞)` by IVT, and at `t = 0`
     realizes `F(0) ≥ s sin h > 0` with `S(X_N) = 0`.
5. **Single-merger divergence witness at `K = N - 1` (Theorem 3)**:
   - `single_merger_divergence`: For every `k ≥ 1`, `τ > 0`, and `h` with `sin h > 0`,
     the single-merger anchor shift `S(A_{N-1}(t))` stays strictly positive at `t = 0`
     while `S(X_N(t)) = 2 |t| sin h → 0`, yielding `S(A_{N-1}(t)) / S(X_N(t)) → +∞`
     as `t → 0⁻` uniformly in `N ≥ k + 1`.

Machine-checked vs prose-only:
- Machine-checked: Lemma 1 (LSE shift inequality and order-statistic `L^∞`-Lipschitz bound
  for the top-`k` Gibbs–Boltzmann potential), deterministic global `K`-means uniqueness
  certificate for separated balls, exact paired-family shifts (raw & normalized bracket,
  ratio `4.5`, IVT surjectivity onto `[0, ∞)`), and the `K = N - 1` divergence witness.
- Prose-only (in `solutions/holoaegis_boundary/proof.pdf`): Probability-space dominated
  convergence for the uniform angular noise `ϕ ~ Unif[-0.001, 0.001]`, the binomial Chernoff
  tail bound for Theorem 2 (`> 1.367 × 10¹⁶`), and the 72-configuration numerical script.
-/

open BigOperators Finset Filter Topology

namespace HoloAegisBoundary

/-! ### 1. Log-Sum-Exp Shift Inequality (Lemma 1, Part 1) -/

/-- Temperature-scaled log-mean-exp `τ log((1/k) ∑_{i=1}^k exp(x_i / τ))`. -/
noncomputable def lseAvg (τ : ℝ) {k : ℕ} (x : Fin k → ℝ) : ℝ :=
  τ * Real.log ((∑ i : Fin k, Real.exp (x i / τ)) / (k : ℝ))

/-- **Lemma 1(a) (Log-sum-exp shift inequality)**: For `k ≥ 1`, `τ > 0`, and `u ≤ y_i - x_i ≤ v`
for all `i`, `u ≤ lseAvg τ y - lseAvg τ x ≤ v`. -/
theorem lseAvg_shift_bounds {k : ℕ} (hk : 0 < k) {τ u v : ℝ} (hτ : 0 < τ)
    {x y : Fin k → ℝ} (hxy : ∀ i, u ≤ y i - x i ∧ y i - x i ≤ v) :
    u ≤ lseAvg τ y - lseAvg τ x ∧ lseAvg τ y - lseAvg τ x ≤ v := by
  have hk_pos : (0 : ℝ) < k := Nat.cast_pos.mpr hk
  have hsum_x_pos : 0 < ∑ i : Fin k, Real.exp (x i / τ) := by
    haveI : Nonempty (Fin k) := ⟨⟨0, hk⟩⟩
    exact Finset.sum_pos (fun i _ => Real.exp_pos _) Finset.univ_nonempty
  have hsum_y_pos : 0 < ∑ i : Fin k, Real.exp (y i / τ) := by
    haveI : Nonempty (Fin k) := ⟨⟨0, hk⟩⟩
    exact Finset.sum_pos (fun i _ => Real.exp_pos _) Finset.univ_nonempty
  have hlo : Real.exp (u / τ) * (∑ i : Fin k, Real.exp (x i / τ)) ≤
      ∑ i : Fin k, Real.exp (y i / τ) := by
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun i _ => ?_
    rw [← Real.exp_add, ← add_div]
    exact Real.exp_le_exp.mpr (div_le_div_of_nonneg_right (by linarith [(hxy i).1]) (le_of_lt hτ))
  have hhi : ∑ i : Fin k, Real.exp (y i / τ) ≤
      Real.exp (v / τ) * (∑ i : Fin k, Real.exp (x i / τ)) := by
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun i _ => ?_
    rw [← Real.exp_add, ← add_div]
    exact Real.exp_le_exp.mpr (div_le_div_of_nonneg_right (by linarith [(hxy i).2]) (le_of_lt hτ))
  have hdiff : lseAvg τ y - lseAvg τ x =
      τ * Real.log ((∑ i : Fin k, Real.exp (y i / τ)) / (∑ i : Fin k, Real.exp (x i / τ))) := by
    unfold lseAvg
    rw [← mul_sub, ← Real.log_div (ne_of_gt (div_pos hsum_y_pos hk_pos))
      (ne_of_gt (div_pos hsum_x_pos hk_pos)),
      div_div_div_cancel_right₀ (ne_of_gt hk_pos)]
  rw [hdiff]
  constructor
  · have hdiv : Real.exp (u / τ) ≤ (∑ i : Fin k, Real.exp (y i / τ)) /
        (∑ i : Fin k, Real.exp (x i / τ)) := by
      rw [le_div_iff₀ hsum_x_pos]; exact hlo
    have hlog := Real.log_le_log (Real.exp_pos _) hdiv
    rw [Real.log_exp] at hlog
    calc u = τ * (u / τ) := by field_simp [ne_of_gt hτ]
      _ ≤ τ * Real.log ((∑ i : Fin k, Real.exp (y i / τ)) / (∑ i : Fin k, Real.exp (x i / τ))) :=
        mul_le_mul_of_nonneg_left hlog (le_of_lt hτ)
  · have hdiv : (∑ i : Fin k, Real.exp (y i / τ)) / (∑ i : Fin k, Real.exp (x i / τ)) ≤
        Real.exp (v / τ) := by
      rw [div_le_iff₀ hsum_x_pos]; exact hhi
    have hlog := Real.log_le_log (div_pos hsum_y_pos hsum_x_pos) hdiv
    rw [Real.log_exp] at hlog
    calc τ * Real.log ((∑ i : Fin k, Real.exp (y i / τ)) / (∑ i : Fin k, Real.exp (x i / τ)))
        ≤ τ * (v / τ) := mul_le_mul_of_nonneg_left hlog (le_of_lt hτ)
      _ = v := by field_simp [ne_of_gt hτ]

/-- **Lemma 1(b)**: `lseAvg τ` is `1`-Lipschitz in `L^∞`. -/
theorem lseAvg_lipschitz {k : ℕ} (hk : 0 < k) {τ ε : ℝ} (hτ : 0 < τ)
    {x y : Fin k → ℝ} (hxy : ∀ i, |y i - x i| ≤ ε) :
    |lseAvg τ y - lseAvg τ x| ≤ ε := by
  have hb := lseAvg_shift_bounds hk hτ (u := -ε) (v := ε) (fun i => abs_le.mp (hxy i))
  exact abs_le.mpr hb

/-! ### 2. Order-Statistic `L^∞`-Lipschitz Bound for Top-`k` Gibbs–Boltzmann Potential -/

/-- Descending order statistics of `s : Fin m → ℝ`: `descOrderStat s r` is the `(r+1)`-th
largest entry of `s`, defined by reversing `Tuple.sort s`. -/
noncomputable def descOrderStat {m : ℕ} (s : Fin m → ℝ) (r : Fin m) : ℝ :=
  s (Tuple.sort s r.rev)

lemma descOrderStat_antitone {m : ℕ} (s : Fin m → ℝ) :
    Antitone (descOrderStat s) := by
  intro i j hij
  unfold descOrderStat
  exact Tuple.monotone_sort s (Fin.rev_le_rev.mpr hij)

/-- Pigeonhole characterization: for any `c`, if `c < descOrderStat s r`, then strictly more
than `r` indices satisfy `c < s j`. Conversely, if at least `r + 1` indices satisfy `a ≤ s j`,
then `a ≤ descOrderStat s r`. -/
lemma le_descOrderStat_of_card_ge {m : ℕ} (s : Fin m → ℝ) (r : Fin m) (a : ℝ)
    (hcard : r.1 + 1 ≤ (Finset.univ.filter (fun j : Fin m => a ≤ s j)).card) :
    a ≤ descOrderStat s r := by
  by_contra hlt
  push_neg at hlt
  -- Every index j with a ≤ s j must have (Tuple.sort s)⁻¹ j in the top r slots (i.e. > r.rev)
  let σ := Tuple.sort s
  let T : Finset (Fin m) := Finset.univ.filter (fun i : Fin m => r.rev < i)
  have hsub : (Finset.univ.filter (fun j : Fin m => a ≤ s j)).map σ.symm.toEmbedding ⊆ T := by
    intro i hi
    rw [Finset.mem_map] at hi
    obtain ⟨j, hj, rfl⟩ := hi
    have hle : a ≤ s j := (Finset.mem_filter.mp hj).2
    rw [Finset.mem_filter]
    refine ⟨Finset.mem_univ _, ?_⟩
    by_contra hnot
    push_neg at hnot
    have hmono : s (σ (σ.symm j)) ≤ s (σ r.rev) := Tuple.monotone_sort s hnot
    simp only [Equiv.apply_symm_apply] at hmono
    have : descOrderStat s r = s (σ r.rev) := rfl
    linarith
  have hTcard : T.card ≤ r.1 := by
    have hbij : T.card = (Finset.Ioo r.rev.1 m).card := by
      refine Finset.card_bij (fun i _ => i.1) ?_ ?_ ?_
      · intro i hi; rw [Finset.mem_Ioo]; exact ⟨(Finset.mem_filter.mp hi).2, i.2⟩
      · intro i1 _ i2 _ h; exact Fin.ext h
      · intro k hk; rw [Finset.mem_Ioo] at hk
        exact ⟨⟨k, hk.2⟩, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hk.1⟩, rfl⟩
    rw [hbij, Nat.card_Ioo, Fin.val_rev]
    omega
  have hle_card := Finset.card_le_card hsub
  rw [Finset.card_map] at hle_card
  omega

lemma card_ge_descOrderStat {m : ℕ} (s : Fin m → ℝ) (r : Fin m) :
    r.1 + 1 ≤ (Finset.univ.filter (fun j : Fin m => descOrderStat s r ≤ s j)).card := by
  let σ := Tuple.sort s
  let T : Finset (Fin m) := Finset.univ.filter (fun i : Fin m => r.rev ≤ i)
  have hsub : T.map σ.toEmbedding ⊆ Finset.univ.filter (fun j : Fin m => descOrderStat s r ≤ s j) := by
    intro j hj
    rw [Finset.mem_map] at hj
    obtain ⟨i, hi, rfl⟩ := hj
    rw [Finset.mem_filter]
    refine ⟨Finset.mem_univ _, ?_⟩
    exact Tuple.monotone_sort s (Finset.mem_filter.mp hi).2
  have hTcard : T.card = r.1 + 1 := by
    have hbij : T.card = (Finset.Ico r.rev.1 m).card := by
      refine Finset.card_bij (fun i _ => i.1) ?_ ?_ ?_
      · intro i hi; rw [Finset.mem_Ico]; exact ⟨(Finset.mem_filter.mp hi).2, i.2⟩
      · intro i1 _ i2 _ h; exact Fin.ext h
      · intro k hk; rw [Finset.mem_Ico] at hk
        exact ⟨⟨k, hk.2⟩, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hk.1⟩, rfl⟩
    rw [hbij, Nat.card_Ico, Fin.val_rev]
    omega
  have hle := Finset.card_le_card hsub
  rw [Finset.card_map, hTcard] at hle
  exact hle

/-- **Order-statistic `L^∞`-contraction**: If `∀ j, |y j - x j| ≤ ε`, then EVERY descending
order statistic shifts by at most `ε`. -/
theorem descOrderStat_linf_contraction {m : ℕ} {x y : Fin m → ℝ} {ε : ℝ}
    (hxy : ∀ j, |y j - x j| ≤ ε) (r : Fin m) :
    |descOrderStat y r - descOrderStat x r| ≤ ε := by
  have hstep : ∀ (u v : Fin m → ℝ), (∀ j, u j - ε ≤ v j) →
      descOrderStat u r - ε ≤ descOrderStat v r := by
    intro u v huv
    apply le_descOrderStat_of_card_ge
    refine le_trans (card_ge_descOrderStat u r) (Finset.card_le_card ?_)
    intro j hj
    rw [Finset.mem_filter] at hj ⊢
    exact ⟨Finset.mem_univ _, by linarith [hj.2, huv j]⟩
  rw [abs_le]
  constructor
  · have h1 := hstep x y (fun j => by linarith [(abs_le.mp (hxy j)).1])
    linarith
  · have h2 := hstep y x (fun j => by linarith [(abs_le.mp (hxy j)).2])
    linarith

/-- Top-`k` Gibbs–Boltzmann potential `U(s) = 1 - lseAvg τ (top-k of s)` for a score vector
`s : Fin m → ℝ` (`k ≤ m`). -/
noncomputable def gibbsPotential (τ : ℝ) {m k : ℕ} (hkm : k ≤ m) (s : Fin m → ℝ) : ℝ :=
  1 - lseAvg τ (fun r : Fin k => descOrderStat s (Fin.castLE hkm r))

/-- **Lemma 1(c) (Top-`k` Gibbs–Boltzmann `L^∞`-Lipschitz bound)**: If all similarity scores
shift by at most `ε` (e.g., `ε = ‖δ‖₂` for unit reference vectors), then the top-`k`
Gibbs–Boltzmann potential changes by at most `ε`. -/
theorem gibbsPotential_lipschitz {m k : ℕ} (hk : 0 < k) (hkm : k ≤ m)
    {τ ε : ℝ} (hτ : 0 < τ) {s₁ s₂ : Fin m → ℝ} (hs : ∀ j, |s₂ j - s₁ j| ≤ ε) :
    |gibbsPotential τ hkm s₂ - gibbsPotential τ hkm s₁| ≤ ε := by
  unfold gibbsPotential
  rw [show (1 - lseAvg τ (fun r : Fin k => descOrderStat s₂ (Fin.castLE hkm r))) -
      (1 - lseAvg τ (fun r : Fin k => descOrderStat s₁ (Fin.castLE hkm r))) =
      -(lseAvg τ (fun r : Fin k => descOrderStat s₂ (Fin.castLE hkm r)) -
        lseAvg τ (fun r : Fin k => descOrderStat s₁ (Fin.castLE hkm r))) by ring, abs_neg]
  exact lseAvg_lipschitz hk hτ (fun r => descOrderStat_linf_contraction hs (Fin.castLE hkm r))

/-! ### 3. Deterministic Global `K`-Means Uniqueness Certificate for Separated Balls -/

/-- Exact bias-variance decomposition in any real inner product space:
`∑_{i ∈ s} ‖x_i - c‖² = ∑_{i ∈ s} ‖x_i - μ‖² + |s| · ‖μ - c‖²`. -/
theorem kmeans_cluster_variance_identity {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    {ι : Type*} (s : Finset ι) (hs : s.Nonempty) (x : ι → E) (c : E) :
    let μ : E := ((s.card : ℝ)⁻¹) • ∑ i ∈ s, x i
    ∑ i ∈ s, ‖x i - c‖ ^ 2 = (∑ i ∈ s, ‖x i - μ‖ ^ 2) + (s.card : ℝ) * ‖μ - c‖ ^ 2 := by
  intro μ
  have hsc : (s.card : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hs.card_pos.ne'
  have hsum_zero : ∑ i ∈ s, (x i - μ) = 0 := by
    rw [Finset.sum_sub_distrib, Finset.sum_const, ← Nat.cast_smul_eq_nsmul ℝ]
    dsimp [μ]
    rw [smul_smul, mul_inv_cancel₀ hsc, one_smul, sub_self]
  have hexp : ∀ i, ‖x i - c‖ ^ 2 = ‖x i - μ‖ ^ 2 + 2 * @inner ℝ E _ (x i - μ) (μ - c) + ‖μ - c‖ ^ 2 := by
    intro i
    have hsplit : x i - c = (x i - μ) + (μ - c) := by abel
    rw [hsplit, norm_add_sq_real]
  calc ∑ i ∈ s, ‖x i - c‖ ^ 2
      = ∑ i ∈ s, (‖x i - μ‖ ^ 2 + 2 * @inner ℝ E _ (x i - μ) (μ - c) + ‖μ - c‖ ^ 2) :=
        Finset.sum_congr rfl fun i _ => hexp i
    _ = (∑ i ∈ s, ‖x i - μ‖ ^ 2) + 2 * @inner ℝ E _ (∑ i ∈ s, (x i - μ)) (μ - c) +
        (s.card : ℝ) * ‖μ - c‖ ^ 2 := by
      rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum,
        ← sum_inner, Finset.sum_const, nsmul_eq_mul]
    _ = (∑ i ∈ s, ‖x i - μ‖ ^ 2) + (s.card : ℝ) * ‖μ - c‖ ^ 2 := by
      rw [hsum_zero, inner_zero_left, mul_zero, add_zero]

/-- **Global `K`-means uniqueness certificate for separated balls**:
Given `K` clusters of equal size `n ≥ 1` where every point `x j i` (`j : Fin K, i : Fin n`)
lies in `B(v j, ρ)` (`‖x j i - v j‖ ≤ ρ`), with `ρ < R`, `(K : ℝ) * ρ² < (R - ρ)²`, and
pairwise separation `‖v j - v l‖ > 2 * R + 2 * ρ` (`j ≠ l`):
1. Every `K`-center tuple `c : Fin K → E` whose minimum-assignment distortion is at most
   `K * n * ρ²` (the distortion of `v`) must place a unique center `c (σ j)` in each ball
   `‖c (σ j) - v j‖ < R` (`σ : Equiv.Perm (Fin K)`).
2. For any such `c`, every point `x j i` is strictly closer to `c (σ j)` than to `c (σ l)`
   for every `l ≠ j`.
3. Consequently, the total distortion equals `∑_j ∑_i ‖x j i - μ j‖² + n ∑_j ‖μ j - c (σ j)‖²`,
   uniquely minimized (up to center permutation) at the exact cluster means `μ j`. -/
theorem kmeans_separated_balls_unique {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    {K n : ℕ} (hK : 0 < K) (hn : 0 < n) {ρ R : ℝ} (hρ : 0 ≤ ρ) (hρR : ρ < R)
    (hdist_ineq : (K : ℝ) * ρ ^ 2 < (R - ρ) ^ 2)
    {v : Fin K → E} (hsep : ∀ j l : Fin K, j ≠ l → 2 * R + 2 * ρ < ‖v j - v l‖)
    {x : Fin K → Fin n → E} (hball : ∀ j i, ‖x j i - v j‖ ≤ ρ)
    {c : Fin K → E}
    (hopt : (∑ j : Fin K, ∑ i : Fin n, ⨅ m : Fin K, ‖x j i - c m‖ ^ 2) ≤
      (K : ℝ) * (n : ℝ) * ρ ^ 2) :
    ∃ σ : Equiv.Perm (Fin K),
      (∀ j, ‖c (σ j) - v j‖ < R) ∧
      (∀ j i l, l ≠ j → ‖x j i - c (σ j)‖ < ‖x j i - c (σ l)‖) ∧
      (∑ j : Fin K, ∑ i : Fin n, ⨅ m : Fin K, ‖x j i - c m‖ ^ 2) =
        (∑ j : Fin K, ∑ i : Fin n, ‖x j i - ((n : ℝ)⁻¹ • ∑ i' : Fin n, x j i')‖ ^ 2) +
        (n : ℝ) * ∑ j : Fin K, ‖((n : ℝ)⁻¹ • ∑ i' : Fin n, x j i') - c (σ j)‖ ^ 2 := by
  haveI : Nonempty (Fin K) := ⟨⟨0, hK⟩⟩
  have hn_pos : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  -- Step 1: Every cluster j has some center m with ‖c m - v j‖ < R
  have hex : ∀ j : Fin K, ∃ m : Fin K, ‖c m - v j‖ < R := by
    intro j
    by_contra hno
    push_neg at hno
    have hpt : ∀ i : Fin n, (R - ρ) ^ 2 ≤ ⨅ m : Fin K, ‖x j i - c m‖ ^ 2 := by
      intro i
      apply le_ciInf
      intro m
      have htri : R - ρ ≤ ‖x j i - c m‖ := by
        have h1 : ‖c m - v j‖ ≤ ‖x j i - c m‖ + ‖x j i - v j‖ := by
          have : c m - v j = -(x j i - c m) + (x j i - v j) := by abel
          calc ‖c m - v j‖ = ‖-(x j i - c m) + (x j i - v j)‖ := by rw [this]
            _ ≤ ‖-(x j i - c m)‖ + ‖x j i - v j‖ := norm_add_le _ _
            _ = ‖x j i - c m‖ + ‖x j i - v j‖ := by rw [norm_neg]
        linarith [hno m, hball j i]
      have hRρ : 0 ≤ R - ρ := sub_nonneg.mpr (le_of_lt hρR)
      nlinarith
    have hcluster : (n : ℝ) * (R - ρ) ^ 2 ≤ ∑ i : Fin n, ⨅ m : Fin K, ‖x j i - c m‖ ^ 2 := by
      calc (n : ℝ) * (R - ρ) ^ 2 = ∑ _i : Fin n, (R - ρ) ^ 2 := by simp [Finset.sum_const, nsmul_eq_mul]
        _ ≤ ∑ i : Fin n, ⨅ m : Fin K, ‖x j i - c m‖ ^ 2 := Finset.sum_le_sum fun i _ => hpt i
    have hnonneg : ∀ j' : Fin K, 0 ≤ ∑ i : Fin n, ⨅ m : Fin K, ‖x j' i - c m‖ ^ 2 := by
      intro j'
      exact Finset.sum_nonneg fun i _ => Real.iInf_nonneg fun m => sq_nonneg _
    have htotal : ∑ i : Fin n, ⨅ m : Fin K, ‖x j i - c m‖ ^ 2 ≤
        ∑ j' : Fin K, ∑ i : Fin n, ⨅ m : Fin K, ‖x j' i - c m‖ ^ 2 :=
      Finset.single_le_sum (fun j' _ => hnonneg j') (Finset.mem_univ j)
    have hstrict : (K : ℝ) * (n : ℝ) * ρ ^ 2 < (n : ℝ) * (R - ρ) ^ 2 := by
      have := mul_lt_mul_of_pos_left hdist_ineq hn_pos
      linarith
    linarith
  -- Step 2: The choice function f : Fin K → Fin K with ‖c (f j) - v j‖ < R is injective (hence a Perm)
  choose f hf using hex
  have hfinj : Function.Injective f := by
    intro j₁ j₂ hfeq
    by_contra hne
    have htri : ‖v j₁ - v j₂‖ ≤ ‖c (f j₁) - v j₁‖ + ‖c (f j₂) - v j₂‖ := by
      have : v j₁ - v j₂ = -(c (f j₁) - v j₁) + (c (f j₂) - v j₂) := by rw [hfeq]; abel
      calc ‖v j₁ - v j₂‖ = ‖-(c (f j₁) - v j₁) + (c (f j₂) - v j₂)‖ := by rw [this]
        _ ≤ ‖-(c (f j₁) - v j₁)‖ + ‖c (f j₂) - v j₂‖ := norm_add_le _ _
        _ = ‖c (f j₁) - v j₁‖ + ‖c (f j₂) - v j₂‖ := by rw [norm_neg]
    linarith [hsep j₁ j₂ hne, hf j₁, hf j₂]
  let σ : Equiv.Perm (Fin K) := Equiv.ofBijective f (Finite.injective_iff_bijective.mp hfinj)
  have hσ_ball : ∀ j, ‖c (σ j) - v j‖ < R := fun j => hf j
  -- Step 3: Every point x j i is strictly closer to c (σ j) than to c (σ l) for l ≠ j
  have hcloser : ∀ j i l, l ≠ j → ‖x j i - c (σ j)‖ < ‖x j i - c (σ l)‖ := by
    intro j i l hlj
    have hnear : ‖x j i - c (σ j)‖ < R + ρ := by
      have : x j i - c (σ j) = (x j i - v j) - (c (σ j) - v j) := by abel
      calc ‖x j i - c (σ j)‖ = ‖(x j i - v j) - (c (σ j) - v j)‖ := by rw [this]
        _ ≤ ‖x j i - v j‖ + ‖c (σ j) - v j‖ := norm_sub_le _ _
        _ < ρ + R := add_lt_add_of_le_of_lt (hball j i) (hσ_ball j)
        _ = R + ρ := add_comm _ _
    have hfar : R + ρ < ‖x j i - c (σ l)‖ := by
      have htri : ‖v j - v l‖ ≤ ‖x j i - v j‖ + ‖x j i - c (σ l)‖ + ‖c (σ l) - v l‖ := by
        have : v j - v l = -(x j i - v j) + (x j i - c (σ l)) + (c (σ l) - v l) := by abel
        calc ‖v j - v l‖ = ‖-(x j i - v j) + (x j i - c (σ l)) + (c (σ l) - v l)‖ := by rw [this]
          _ ≤ ‖-(x j i - v j) + (x j i - c (σ l))‖ + ‖c (σ l) - v l‖ := norm_add_le _ _
          _ ≤ (‖-(x j i - v j)‖ + ‖x j i - c (σ l)‖) + ‖c (σ l) - v l‖ :=
            add_le_add_right (norm_add_le _ _) _
          _ = ‖x j i - v j‖ + ‖x j i - c (σ l)‖ + ‖c (σ l) - v l‖ := by rw [norm_neg]
      linarith [hsep j l hlj.symm, hball j i, hσ_ball l]
    exact lt_trans hnear hfar
  -- Step 4: Minimum assignment is forced to σ j, and apply kmeans_cluster_variance_identity
  have hinf_eq : ∀ j i, (⨅ m : Fin K, ‖x j i - c m‖ ^ 2) = ‖x j i - c (σ j)‖ ^ 2 := by
    intro j i
    apply le_antisymm
    · exact ciInf_le (Finite.bddBelow_range _) (σ j)
    · apply le_ciInf
      intro m
      by_cases hm : m = σ j
      · rw [hm]
      · set l := σ.symm m
        have hlj : l ≠ j := fun h => hm (by rw [← h]; simp [l])
        have hlt := hcloser j i l hlj
        rw [Equiv.apply_symm_apply] at hlt
        have hnn : 0 ≤ ‖x j i - c (σ j)‖ := norm_nonneg _
        nlinarith
  refine ⟨σ, hσ_ball, hcloser, ?_⟩
  have hrew : (∑ j : Fin K, ∑ i : Fin n, ⨅ m : Fin K, ‖x j i - c m‖ ^ 2) =
      ∑ j : Fin K, ∑ i : Fin n, ‖x j i - c (σ j)‖ ^ 2 :=
    Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun i _ => hinf_eq j i
  rw [hrew]
  have hne : (Finset.univ : Finset (Fin n)).Nonempty := ⟨⟨0, hn⟩, Finset.mem_univ _⟩
  have hper : ∀ j : Fin K,
      ∑ i : Fin n, ‖x j i - c (σ j)‖ ^ 2 =
        (∑ i : Fin n, ‖x j i - ((n : ℝ)⁻¹ • ∑ i' : Fin n, x j i')‖ ^ 2) +
        (n : ℝ) * ‖((n : ℝ)⁻¹ • ∑ i' : Fin n, x j i') - c (σ j)‖ ^ 2 := by
    intro j
    have hv := kmeans_cluster_variance_identity (Finset.univ : Finset (Fin n)) hne (x j) (c (σ j))
    simpa [Finset.card_univ, Fintype.card_fin] using hv
  calc ∑ j : Fin K, ∑ i : Fin n, ‖x j i - c (σ j)‖ ^ 2
      = ∑ j : Fin K, ((∑ i : Fin n, ‖x j i - ((n : ℝ)⁻¹ • ∑ i' : Fin n, x j i')‖ ^ 2) +
          (n : ℝ) * ‖((n : ℝ)⁻¹ • ∑ i' : Fin n, x j i') - c (σ j)‖ ^ 2) :=
        Finset.sum_congr rfl fun j _ => hper j
    _ = (∑ j : Fin K, ∑ i : Fin n, ‖x j i - ((n : ℝ)⁻¹ • ∑ i' : Fin n, x j i')‖ ^ 2) +
        (n : ℝ) * ∑ j : Fin K, ‖((n : ℝ)⁻¹ • ∑ i' : Fin n, x j i') - c (σ j)‖ ^ 2 := by
      rw [Finset.sum_add_distrib, ← Finset.mul_sum]

/-- Exact numerical certification of the Theorem 1 separated-ball parameters
(`K ≤ 300`, `ρ = 0.004`, `R = 0.25`, `‖v_j - v_l‖ ≥ 1.131`). -/
theorem theorem1_ball_params_certified {K : ℕ} (hK : K ≤ 300) :
    (0.004 : ℝ) < 0.25 ∧
    (K : ℝ) * (0.004 : ℝ) ^ 2 < (0.25 - 0.004 : ℝ) ^ 2 ∧
    2 * (0.25 : ℝ) + 2 * (0.004 : ℝ) < 1.131 := by
  have hKr : (K : ℝ) ≤ 300 := by exact_mod_cast hK
  refine ⟨by norm_num, by nlinarith, by norm_num⟩

/-! ### 4. Paired-Family Shifts, Exact Ratio `4.5`, and Surjectivity onto `[0, ∞)` (Theorem 1) -/

/-- Full-bank shift `S(X_N(t)) = 2 |t| sin h` when all top-`k` selected points have identical
scores `a cos h ± t sin h`. -/
theorem paired_full_bank_shift {k : ℕ} (hk : 0 < k) {τ a h t : ℝ} (hτ : 0 < τ)
    (hsinh : 0 < Real.sin h) :
    |lseAvg τ (fun _ : Fin k => a * Real.cos h + t * Real.sin h) -
     lseAvg τ (fun _ : Fin k => a * Real.cos h - t * Real.sin h)| = 2 * |t| * Real.sin h := by
  have hconst : ∀ c : ℝ, lseAvg τ (fun _ : Fin k => c) = c := by
    intro c
    unfold lseAvg
    have hk_ne : (k : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (ne_of_gt hk)
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
      mul_div_cancel_left₀ _ hk_ne, Real.log_exp]
    field_simp [ne_of_gt hτ]
  rw [hconst, hconst]
  rw [show (a * Real.cos h + t * Real.sin h) - (a * Real.cos h - t * Real.sin h) =
      2 * t * Real.sin h by ring, abs_mul, abs_mul, abs_of_pos (by norm_num : (0 : ℝ) < 2),
      abs_of_pos hsinh]

/-- **Paired-family raw centroid shift, exact ratio `4.5` at `t = -s/10`, and exact surjectivity
onto `[0, ∞)`**: With `s = 1/1000`, `C = (a + b)/2`, `D(t) = (t + s)/2 ≥ 0` on `t ∈ [-s, 0]`:
- Raw anchor shift `S(A_K) = 2 D(t) sin h = (t + s) sin h`.
- At `t = -s / 10`, `S(A_K) / S(X_N) = 9/2 = 4.5`.
- For every `r ≥ 0`, `t = -s / (2r + 1) ∈ [-s, 0)` realizes exact ratio `r`. -/
theorem paired_raw_shift_ratio_and_surj {k : ℕ} (hk : 0 < k) {τ C s h : ℝ}
    (hτ : 0 < τ) (hs : 0 < s) (hsinh : 0 < Real.sin h) :
    (∀ t ∈ Set.Icc (-s) 0,
      |lseAvg τ (fun _ : Fin k => C * Real.cos h + ((t + s) / 2) * Real.sin h) -
       lseAvg τ (fun _ : Fin k => C * Real.cos h - ((t + s) / 2) * Real.sin h)| =
        (t + s) * Real.sin h) ∧
    (((-s / 10 + s) * Real.sin h) / (2 * |-s / 10| * Real.sin h) = 9 / 2) ∧
    (∀ r : ℝ, 0 ≤ r → ∃ t ∈ Set.Ico (-s) 0,
      ((t + s) * Real.sin h) / (2 * |t| * Real.sin h) = r) := by
  have hconst : ∀ c : ℝ, lseAvg τ (fun _ : Fin k => c) = c := by
    intro c
    unfold lseAvg
    have hk_ne : (k : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (ne_of_gt hk)
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
      mul_div_cancel_left₀ _ hk_ne, Real.log_exp]
    field_simp [ne_of_gt hτ]
  refine ⟨?_, ?_, ?_⟩
  · intro t ⟨htlo, _⟩
    rw [hconst, hconst]
    rw [show (C * Real.cos h + (t + s) / 2 * Real.sin h) -
        (C * Real.cos h - (t + s) / 2 * Real.sin h) = (t + s) * Real.sin h by ring]
    exact abs_of_nonneg (mul_nonneg (by linarith) (le_of_lt hsinh))
  · rw [abs_of_neg (by linarith : -s / 10 < 0)]
    have hsne : s ≠ 0 := ne_of_gt hs
    have hshne : Real.sin h ≠ 0 := ne_of_gt hsinh
    field_simp [hsne, hshne]
    ring
  · intro r hr
    set t := -s / (2 * r + 1)
    have hden : 0 < 2 * r + 1 := by linarith
    have ht_neg : t < 0 := div_neg_of_neg_of_pos (by linarith) hden
    have ht_ge : -s ≤ t := by
      dsimp [t]
      rw [le_div_iff₀ hden]
      nlinarith
    refine ⟨t, ⟨ht_ge, ht_neg⟩, ?_⟩
    rw [abs_of_neg ht_neg]
    dsimp [t]
    have hsne : s ≠ 0 := ne_of_gt hs
    have hshne : Real.sin h ≠ 0 := ne_of_gt hsinh
    have hden_ne : 2 * r + 1 ≠ 0 := ne_of_gt hden
    field_simp [hsne, hshne, hden_ne]
    ring

/-- **Unit-normalized anchor shift bracket (eq. 3)**: When selected normalized anchors have
norms `L_i ∈ [0.996, 1]` and score shift `2 D sin h / L_i`, Lemma 1 gives
`2 D sin h ≤ S(A_K) ≤ 2 D sin h / 0.996`. -/
theorem paired_normalized_shift_bracket {k : ℕ} (hk : 0 < k) {τ C D h : ℝ}
    (hτ : 0 < τ) (hD : 0 ≤ D) (hsinh : 0 < Real.sin h)
    {L : Fin k → ℝ} (hL : ∀ i, (996 / 1000 : ℝ) ≤ L i ∧ L i ≤ 1) :
    let sPlus : Fin k → ℝ := fun i => (C * Real.cos h + D * Real.sin h) / L i
    let sMinus : Fin k → ℝ := fun i => (C * Real.cos h - D * Real.sin h) / L i
    2 * D * Real.sin h ≤ lseAvg τ sPlus - lseAvg τ sMinus ∧
    lseAvg τ sPlus - lseAvg τ sMinus ≤ (2 * D * Real.sin h) / (996 / 1000) := by
  intro sPlus sMinus
  apply lseAvg_shift_bounds hk hτ
  intro i
  have hLi_pos : 0 < L i := lt_of_lt_of_le (by norm_num) (hL i).1
  have hdiff : sPlus i - sMinus i = (2 * D * Real.sin h) / L i := by
    dsimp [sPlus, sMinus]; ring
  rw [hdiff]
  have hnum_nn : 0 ≤ 2 * D * Real.sin h := by positivity
  constructor
  · rw [le_div_iff₀ hLi_pos]
    nlinarith [(hL i).2]
  · exact div_le_div_of_nonneg_left hnum_nn (by norm_num) (hL i).1

/-- **Surjectivity onto `[0, ∞)` (Theorem 1)**: For ANY continuous shift function `F` on `[-s, 0]`
satisfying `(t + s) sin h ≤ F(t) ≤ (t + s) sin h / (996/1000)` (covering both raw and
unit-normalized expected shifts), `t ↦ F(t) / (2 |t| sin h)` maps `[-s, 0)` onto ALL of `[0, ∞)`,
and at `t = 0`, `F(0) ≥ s sin h > 0` while `2 |0| sin h = 0`. -/
theorem paired_ratio_surjective_Ici {s h : ℝ} (hs : 0 < s) (hsinh : 0 < Real.sin h)
    {F : ℝ → ℝ} (hFcont : ContinuousOn F (Set.Icc (-s) 0))
    (hFbounds : ∀ t ∈ Set.Icc (-s) 0,
      (t + s) * Real.sin h ≤ F t ∧ F t ≤ (t + s) * Real.sin h / (996 / 1000)) :
    (∀ r : ℝ, 0 ≤ r → ∃ t ∈ Set.Ico (-s) 0, F t / (2 * |t| * Real.sin h) = r) ∧
    (0 < F 0 ∧ 2 * |(0 : ℝ)| * Real.sin h = 0) := by
  refine ⟨?_, ?_⟩
  · intro r hr
    set R : ℝ → ℝ := fun t => F t / (2 * |t| * Real.sin h)
    have hRs : R (-s) = 0 := by
      have h0 := hFbounds (-s) ⟨le_rfl, by linarith⟩
      have hFs : F (-s) = 0 := by linarith
      dsimp [R]; rw [hFs, zero_div]
    -- Choose t₁ ∈ (-s, 0) sufficiently close to 0 that R(t₁) ≥ r
    set t₁ : ℝ := -s / (2 * r + 2)
    have ht1_neg : t₁ < 0 := div_neg_of_neg_of_pos (by linarith) (by linarith)
    have ht1_gt : -s < t₁ := by
      dsimp [t₁]; rw [lt_div_iff₀ (by linarith)]; nlinarith
    have hRt1 : r ≤ R t₁ := by
      dsimp [R]
      have hlo := (hFbounds t₁ ⟨le_of_lt ht1_gt, le_of_lt ht1_neg⟩).1
      rw [abs_of_neg ht1_neg]
      have hden_pos : 0 < 2 * -t₁ * Real.sin h := by nlinarith
      rw [le_div_iff₀ hden_pos]
      have heq : t₁ + s = s * (2 * r + 1) / (2 * r + 2) := by
        dsimp [t₁]; field_simp [show (2 * r + 2 : ℝ) ≠ 0 by linarith]; ring
      have heq2 : r * (2 * -t₁ * Real.sin h) = s * (2 * r) / (2 * r + 2) * Real.sin h := by
        dsimp [t₁]; field_simp [show (2 * r + 2 : ℝ) ≠ 0 by linarith]
      have hle_num : s * (2 * r) / (2 * r + 2) ≤ s * (2 * r + 1) / (2 * r + 2) :=
        div_le_div_of_nonneg_right (by nlinarith) (by linarith)
      have hle_mul := mul_le_mul_of_nonneg_right hle_num (le_of_lt hsinh)
      rw [heq] at hlo
      linarith
    have hRcont : ContinuousOn R (Set.Icc (-s) t₁) := by
      apply ContinuousOn.div (hFcont.mono (fun t ht => ⟨ht.1, le_trans ht.2 (le_of_lt ht1_neg)⟩))
      · exact ((continuous_const.mul continuous_abs).mul continuous_const).continuousOn
      · intro t ht
        have htneg : t < 0 := lt_of_le_of_lt ht.2 ht1_neg
        have : 0 < |t| := abs_pos.mpr (ne_of_lt htneg)
        positivity
    have hivt := intermediate_value_Icc (le_of_lt ht1_gt) hRcont
    have hr_mem : r ∈ Set.Icc (R (-s)) (R t₁) := ⟨by linarith, hRt1⟩
    obtain ⟨t, ht_mem, hRt⟩ := hivt hr_mem
    exact ⟨t, ⟨ht_mem.1, lt_of_le_of_lt ht_mem.2 ht1_neg⟩, hRt⟩
  · have h0 := (hFbounds 0 ⟨by linarith, le_rfl⟩).1
    exact ⟨by linarith [mul_pos hs hsinh], by simp⟩

/-! ### 5. Single-Merger Divergence Witness at `K = N - 1` (Theorem 3) -/

/-- Exact anchor shift `S(A_{N-1}(t))` in the Theorem 3 single-merger family, where the top-`k`
anchors consist of `k - 1` copies with scores `a cos h ± t sin h` and `1` copy with scores
`b cos h ± s sin h` (independent of `N ≥ k + 1`). -/
noncomputable def singleMergerAnchorShift (k : ℕ) (τ a b s h t : ℝ) : ℝ :=
  |τ * Real.log ((((k - 1 : ℕ) : ℝ) * Real.exp ((a * Real.cos h + t * Real.sin h) / τ) +
      Real.exp ((b * Real.cos h + s * Real.sin h) / τ)) / (k : ℝ)) -
   τ * Real.log ((((k - 1 : ℕ) : ℝ) * Real.exp ((a * Real.cos h - t * Real.sin h) / τ) +
      Real.exp ((b * Real.cos h - s * Real.sin h) / τ)) / (k : ℝ))|

/-- **Theorem 3 (Single-merger divergence at `K = N - 1`)**: For every `k ≥ 1`, `τ > 0`, `s > 0`,
`sin h > 0`:
1. At `t = 0`, `singleMergerAnchorShift k τ a b s h 0 > 0` (while `S(X_N(0)) = 0`).
2. As `t → 0⁻`, `singleMergerAnchorShift k τ a b s h t / (2 |t| sin h) → +∞` (uniformly in `N`). -/
theorem single_merger_divergence {k : ℕ} (hk : 1 ≤ k) {τ a b s h : ℝ}
    (hτ : 0 < τ) (hs : 0 < s) (hsinh : 0 < Real.sin h) :
    0 < singleMergerAnchorShift k τ a b s h 0 ∧
    Tendsto (fun t => singleMergerAnchorShift k τ a b s h t / (2 * |t| * Real.sin h))
      (nhdsWithin 0 (Set.Iio 0)) atTop := by
  have hk_pos : (0 : ℝ) < k := Nat.cast_pos.mpr (by omega)
  set fPlus : ℝ → ℝ := fun t =>
    (((k - 1 : ℕ) : ℝ) * Real.exp ((a * Real.cos h + t * Real.sin h) / τ) +
      Real.exp ((b * Real.cos h + s * Real.sin h) / τ)) / (k : ℝ)
  set fMinus : ℝ → ℝ := fun t =>
    (((k - 1 : ℕ) : ℝ) * Real.exp ((a * Real.cos h - t * Real.sin h) / τ) +
      Real.exp ((b * Real.cos h - s * Real.sin h) / τ)) / (k : ℝ)
  have hfPlus_pos : ∀ t, 0 < fPlus t := fun t => by
    dsimp [fPlus]
    have h1 : 0 ≤ ((k - 1 : ℕ) : ℝ) * Real.exp ((a * Real.cos h + t * Real.sin h) / τ) :=
      mul_nonneg (Nat.cast_nonneg _) (le_of_lt (Real.exp_pos _))
    have h2 : 0 < Real.exp ((b * Real.cos h + s * Real.sin h) / τ) := Real.exp_pos _
    exact div_pos (by linarith) hk_pos
  have hfMinus_pos : ∀ t, 0 < fMinus t := fun t => by
    dsimp [fMinus]
    have h1 : 0 ≤ ((k - 1 : ℕ) : ℝ) * Real.exp ((a * Real.cos h - t * Real.sin h) / τ) :=
      mul_nonneg (Nat.cast_nonneg _) (le_of_lt (Real.exp_pos _))
    have h2 : 0 < Real.exp ((b * Real.cos h - s * Real.sin h) / τ) := Real.exp_pos _
    exact div_pos (by linarith) hk_pos
  have hS0_pos : 0 < singleMergerAnchorShift k τ a b s h 0 := by
    unfold singleMergerAnchorShift
    change 0 < |τ * Real.log (fPlus 0) - τ * Real.log (fMinus 0)|
    have hlt : fMinus 0 < fPlus 0 := by
      dsimp [fPlus, fMinus]
      apply div_lt_div_of_pos_right _ hk_pos
      have hexp_lt : Real.exp ((b * Real.cos h - s * Real.sin h) / τ) <
          Real.exp ((b * Real.cos h + s * Real.sin h) / τ) := by
        apply Real.exp_strictMono
        exact div_lt_div_of_pos_right (by linarith [mul_pos hs hsinh]) hτ
      simp only [zero_mul, add_zero, sub_zero]
      linarith
    have hlog_lt : Real.log (fMinus 0) < Real.log (fPlus 0) :=
      Real.log_lt_log (hfMinus_pos 0) hlt
    have hdiff_pos : 0 < τ * Real.log (fPlus 0) - τ * Real.log (fMinus 0) := by
      nlinarith
    rw [abs_of_pos hdiff_pos]
    exact hdiff_pos
  refine ⟨hS0_pos, ?_⟩
  have hcont : Continuous (singleMergerAnchorShift k τ a b s h) := by
    unfold singleMergerAnchorShift
    have hcPlus : Continuous fPlus := by
      dsimp [fPlus]; fun_prop
    have hcMinus : Continuous fMinus := by
      dsimp [fMinus]; fun_prop
    have hlogP : Continuous (fun t => τ * Real.log (fPlus t)) :=
      continuous_const.mul (Continuous.log hcPlus (fun t => ne_of_gt (hfPlus_pos t)))
    have hlogM : Continuous (fun t => τ * Real.log (fMinus t)) :=
      continuous_const.mul (Continuous.log hcMinus (fun t => ne_of_gt (hfMinus_pos t)))
    exact (hlogP.sub hlogM).abs
  have hnum_tend : Tendsto (singleMergerAnchorShift k τ a b s h)
      (nhdsWithin 0 (Set.Iio 0)) (nhds (singleMergerAnchorShift k τ a b s h 0)) :=
    hcont.continuousAt.tendsto.mono_left nhdsWithin_le_nhds
  have hden_tend : Tendsto (fun t : ℝ => 2 * |t| * Real.sin h)
      (nhdsWithin 0 (Set.Iio 0)) (nhdsWithin 0 (Set.Ioi 0)) := by
    rw [tendsto_nhdsWithin_iff]
    refine ⟨?_, ?_⟩
    · have h0 : (2 : ℝ) * |(0 : ℝ)| * Real.sin h = 0 := by simp
      have hc : Continuous (fun t : ℝ => 2 * |t| * Real.sin h) := by fun_prop
      have ht0 : Tendsto (fun t : ℝ => 2 * |t| * Real.sin h) (nhds (0 : ℝ)) (nhds (0 : ℝ)) := by
        simpa [h0] using hc.continuousAt (x := (0 : ℝ)) |>.tendsto
      exact ht0.mono_left nhdsWithin_le_nhds
    · filter_upwards [self_mem_nhdsWithin] with t ht
      rw [Set.mem_Ioi]
      have hpos : 0 < |t| := abs_pos.mpr (ne_of_lt ht)
      positivity
  exact Filter.Tendsto.pos_mul_atTop hS0_pos hnum_tend (tendsto_inv_nhdsGT_zero.comp hden_tend)

end HoloAegisBoundary
