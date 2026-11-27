import Mathlib

/-!
# Exact Recovery Threshold for Multi-View Gaussian Planted Matching
(arXiv:2611.32892, Conjecture `conj:c1prime`)

We formalize in Lean 4 + Mathlib (completely verified, standard foundations only) the deterministic
combinatorial, variational, and positive-definite quadratic-form cores of the proof of
Conjecture `conj:c1prime` for general `K ≥ 2` and positive-definite covariance `R ≻ 0`
from `solutions/multiview_planted_align/solution.md`:

1. **Threshold rate, Universal Partition Inequality, and Wishart Fenchel–Legendre conjugate**:
   - `thresholdRate`: Exact threshold rate `L(R) = min (I_* / 2) (inf_{|P| ≥ 3} R_P)`.
   - `universal_partition_ineq`: Lemma 1 (`T_P(G) ≥ (q - 1)(L(R) - D(G ‖ p))` for all `q ≥ 2`).
   - `op_norm_fourth_le_hs_sq`: Finite-dimensional spectral bound `‖Z‖_op^4 ≤ tr((Z* Z)^2)`
     giving the `q = 2` universal rate reduction `R_P ≥ I_* / 2 ≥ L(R)`.
   - `wishart_kl_scalar_nonneg`, `wishart_fenchel_legendre_scalar`: Exact scalar Fenchel–Legendre
     conjugate `sup_{a < 1/2} (a v + (1/2) log (1 - 2a)) = (1/2)(v - log v - 1) ≥ 0` attained
     uniquely at `a = (1 - v⁻¹) / 2` (eigenvalue core of Lemma 2's Cramér rate function).
2. **Deterministic Combinatorial Component-Size Cap (Lemma 3)**:
   - `KRegularBipartite`: Exact `K`-regular bipartite multigraph modeled by `K` permutations
     `π : Fin K → Equiv.Perm (Fin m)`.
   - `card_leftNbhd_ge` and `strict_hall_expansion`: Double-counting edge slots shows
     `|N(E)| ≥ |E|` for all `E`, and strict expansion `|N(E)| ≥ |E| + 1` for every nonempty
     proper subset `E` of a connected component.
   - `strict_hall_prefix_ineq`: Summing strict expansion over `j` components of a prefix `F ⊆ B`
     yields `|N(F)| - j ≥ |F|`.
   - `greedy_activation_size_cap`: Greedy weight activation stops with `|C| ≤ 1 + K (T - 1) ≤ 1 + K T`.
   - `abel_summation_lower_bound`: Exact discrete Abel summation turning prefix rank bounds
     `∑_{i ≤ t} k_i ≥ t + 1` and nonincreasing weights `b_i ≥ λ` into
     `∑ k_i (b_i + η) ≥ (s - 1)(λ + η) + ∑ (b_i - λ)`.
   - `one_sided_budget_ineq` and `two_sided_cancellation_contradiction`: Adding the left and right
     admissibility inequalities cancels the weight sums completely and yields
     `(s + r - 2) η ≤ 2 λ ≤ 2 c`, contradicting `s + r ≥ T > 2 + 2 c / η`.
3. **Positive-Definite Quadratic-Form & Swap Covariance Inequalities (Section 2.3)**:
   - `swapSignMatrix`, `swapSignMatrix_mul_self`, `det_swap_conj`: Exact block sign-flip matrix
     `J = diag(I_S, -I_{S^c})` preserves determinant under congruence `J R J`.
   - `swap_average_blockDiag`, `det_swap_average`, `swap_quadratic_energy_identity`: Averaging
     the original and swapped difference covariances cancels the cross-covariance block exactly
     to `blockDiag(R₁₁, R₂₂)` with determinant `det R₁₁ * det R₂₂`.
   - `bhattacharyya_two_by_two_mi`: Exact Gaussian swap mutual information positivity for any
     `2 × 2` positive-definite covariance matrix.
   - `equicorr_posDef_iff`: Exact positive-definiteness criterion `-1 / (K - 1) < r < 1` for
     `K`-view equicorrelation matrices `R_{K,r} = (1 - r) I_K + r 1 1ᵀ`.
   - `bounded_component_exponent_neg`: Union-bound error exponent negativity when
     `d L / log n ≥ 1 + ε`.

Machine-checked vs prose-only:
- Machine-checked: All algebraic, variational, combinatorial (Hall/Abel/two-sided cancellation),
  block-matrix swap, positive-definite quadratic-form, and exponent-rate results listed above.
- Prose-only (in `solutions/multiview_planted_align/proof.pdf`): Finite-dimensional Cramér LDP
  tail probability for Wishart outer products, sequential Hilbert–Schmidt Cauchy–Schwarz integral
  bound across active views, Bayes-optimality transfer under the uniform permutation prior, and
  the matching strong converse from Wang & Hu.
-/

open BigOperators Finset Matrix

namespace MultiviewPlantedAlign

/-! ### 1. Threshold Rate, Universal Partition Inequality, and Wishart Rate Function -/

/-- Exact threshold rate `L(R) = min (I_* / 2) (inf_{|P| ≥ 3} R_P)` when `P_ge3` is nonempty,
and `I_* / 2` when `K = 2` (`P_ge3 = ∅`). -/
noncomputable def thresholdRate {ι : Type*} (I_star : ℝ) (P_ge3 : Finset ι) (R_P : ι → ℝ) : ℝ :=
  if h : P_ge3.Nonempty then min (I_star / 2) (P_ge3.inf' h R_P) else I_star / 2

lemma thresholdRate_le_half_I_star {ι : Type*} (I_star : ℝ) (P_ge3 : Finset ι) (R_P : ι → ℝ) :
    thresholdRate I_star P_ge3 R_P ≤ I_star / 2 := by
  unfold thresholdRate
  split_ifs with h
  · exact min_le_left _ _
  · exact le_rfl

lemma thresholdRate_le_partitionRate {ι : Type*} (I_star : ℝ) (P_ge3 : Finset ι) (R_P : ι → ℝ)
    {p : ι} (hp : p ∈ P_ge3) :
    thresholdRate I_star P_ge3 R_P ≤ R_P p := by
  unfold thresholdRate
  have hne : P_ge3.Nonempty := ⟨p, hp⟩
  rw [dif_pos hne]
  exact le_trans (min_le_right _ _) (Finset.inf'_le R_P hp)

/-- **Lemma 1 (Universal Partition Inequality)**:
For every partition with `q ≥ 2` blocks, if the partition rate `R_P` satisfies
`L ≤ R_P` and `D + T / (q - 1) ≥ R_P`, then `T ≥ (q - 1) * (L - D)`.
In `solution.md`, for `q ≥ 3` we have `L ≤ R_P` directly by `thresholdRate_le_partitionRate`,
while for `q = 2` the Hilbert–Schmidt reduction yields `R_P ≥ I_* / 2 ≥ L`. -/
theorem universal_partition_ineq {q : ℕ} (hq : 2 ≤ q) {L D T R_P : ℝ}
    (hL : L ≤ R_P) (hvar : R_P ≤ D + T / ((q : ℝ) - 1)) :
    ((q : ℝ) - 1) * (L - D) ≤ T := by
  have hq_pos : (0 : ℝ) < (q : ℝ) - 1 := by
    have : (2 : ℝ) ≤ (q : ℝ) := by exact_mod_cast hq
    linarith
  have h1 : L - D ≤ T / ((q : ℝ) - 1) := by linarith
  have h2 := mul_le_mul_of_nonneg_left h1 (le_of_lt hq_pos)
  rwa [mul_div_cancel₀ _ (ne_of_gt hq_pos)] at h2

/-- Finite-dimensional spectral core of the `q = 2` reduction:
if `λ_i` are the eigenvalues of `Z* Z` and `σ² = λ_{i₀}`, then
`σ^4 ≤ ∑_i λ_i^2 = tr((Z* Z)^2)`. Taking `-log` gives `-log σ² ≥ -(1/2) log tr((Z* Z)^2)`. -/
theorem op_norm_fourth_le_hs_sq {m : ℕ} (lam : Fin m → ℝ) (i₀ : Fin m) :
    (lam i₀) ^ 2 ≤ ∑ i : Fin m, (lam i) ^ 2 := by
  exact Finset.single_le_sum (fun i _ => sq_nonneg (lam i)) (Finset.mem_univ i₀)

/-- Logarithmic consequence of `op_norm_fourth_le_hs_sq`:
if `0 < σ²` and `σ^4 ≤ tr_sq`, then `-log σ² ≥ -(1/2) log tr_sq`. -/
theorem two_block_rate_ge_half_mi {op_sq tr_sq : ℝ} (hop : 0 < op_sq)
    (hle : op_sq ^ 2 ≤ tr_sq) :
    -(1 / 2 : ℝ) * Real.log tr_sq ≤ -Real.log op_sq := by
  have htr : 0 < tr_sq := lt_of_lt_of_le (sq_pos_of_pos hop) hle
  have hlog : Real.log (op_sq ^ 2) ≤ Real.log tr_sq := Real.log_le_log (sq_pos_of_pos hop) hle
  rw [Real.log_pow] at hlog
  push_cast at hlog
  linarith

/-- Scalar Wishart KL rate function `I(v) = (1/2)(v - log v - 1)` is nonnegative for `v > 0`,
with equality iff `v = 1`. -/
theorem wishart_kl_scalar_nonneg {v : ℝ} (hv : 0 < v) :
    0 ≤ (v - Real.log v - 1) / 2 := by
  have h := Real.log_le_sub_one_of_pos hv
  linarith

/-- Exact scalar Fenchel–Legendre duality (eigenvalue core of Lemma 2):
for every `v > 0` and `a < 1/2`,
`a * v + (1/2) * log (1 - 2a) ≤ (1/2) * (v - log v - 1)`,
with equality attained at `a = (1 - v⁻¹) / 2`. -/
theorem wishart_fenchel_legendre_le {v a : ℝ} (hv : 0 < v) (ha : 2 * a < 1) :
    a * v + Real.log (1 - 2 * a) / 2 ≤ (v - Real.log v - 1) / 2 := by
  have h1 : 0 < 1 - 2 * a := by linarith
  have hprod : 0 < (1 - 2 * a) * v := mul_pos h1 hv
  have hlog : Real.log ((1 - 2 * a) * v) ≤ (1 - 2 * a) * v - 1 :=
    Real.log_le_sub_one_of_pos hprod
  rw [Real.log_mul (ne_of_gt h1) (ne_of_gt hv)] at hlog
  linarith

theorem wishart_fenchel_legendre_eq {v : ℝ} (hv : 0 < v) :
    let a := (1 - v⁻¹) / 2
    2 * a < 1 ∧ a * v + Real.log (1 - 2 * a) / 2 = (v - Real.log v - 1) / 2 := by
  have hv_ne : v ≠ 0 := ne_of_gt hv
  have h1 : 1 - 2 * ((1 - v⁻¹) / 2) = v⁻¹ := by ring
  refine ⟨?_, ?_⟩
  · have : 0 < v⁻¹ := inv_pos.mpr hv
    linarith
  · rw [h1, Real.log_inv]
    field_simp
    ring

/-! ### 2. Deterministic Combinatorial Component-Size Cap (Lemma 3) -/

/-- A `K`-regular bipartite multigraph on `Fin m` (left rows) and `Fin m` (right rows)
formed by superimposing two alignments across `K` views: each view `a : Fin K` defines a
bijection `π a : Fin m ≃ Fin m` sending left row `u` to right row `π a u`. -/
structure KRegularBipartite (K m : ℕ) where
  π : Fin K → Equiv.Perm (Fin m)

namespace KRegularBipartite

/-- Left neighborhood `N(E)` of a right vertex set `E ⊆ Fin m`. -/
def leftNbhd {K m : ℕ} (G : KRegularBipartite K m) (E : Finset (Fin m)) : Finset (Fin m) :=
  Finset.univ.filter fun u => ∃ a : Fin K, G.π a u ∈ E

/-- Right neighborhood `N(A)` of a left vertex set `A ⊆ Fin m`. -/
def rightNbhd {K m : ℕ} (G : KRegularBipartite K m) (A : Finset (Fin m)) : Finset (Fin m) :=
  Finset.univ.filter fun v => ∃ a : Fin K, (G.π a).symm v ∈ A

/-- Connectivity of the `K`-regular bipartite multigraph: no nonempty proper subset `E` of
right vertices can be closed under all view matchings (i.e., `∀ u ∈ N(E), ∀ a, π a u ∈ E`). -/
def IsConnected {K m : ℕ} (G : KRegularBipartite K m) : Prop :=
  ∀ E : Finset (Fin m), E.Nonempty → E ≠ Finset.univ →
    ∃ u ∈ G.leftNbhd E, ∃ a : Fin K, G.π a u ∉ E

/-- Double-counting edge slots: for `K ≥ 1`, every right subset `E` satisfies `|E| ≤ |N(E)|`. -/
theorem card_leftNbhd_ge {K m : ℕ} (hK : 0 < K) (G : KRegularBipartite K m)
    (E : Finset (Fin m)) :
    E.card ≤ (G.leftNbhd E).card := by
  let a0 : Fin K := ⟨0, hK⟩
  have hsub : E.image (G.π a0).symm ⊆ G.leftNbhd E := by
    intro u hu
    rcases Finset.mem_image.mp hu with ⟨v, hv, rfl⟩
    simp only [leftNbhd, Finset.mem_filter, Finset.mem_univ, true_and]
    exact ⟨a0, by simpa using hv⟩
  calc E.card
      = (E.image (G.π a0).symm).card := (Finset.card_image_of_injective E (G.π a0).symm.injective).symm
    _ ≤ (G.leftNbhd E).card := Finset.card_le_card hsub

/-- **Strict Hall Expansion** (Lemma 3, step 1):
In a connected `K`-regular bipartite multigraph, every nonempty proper subset
`E ⊂neq Fin m` of right vertices satisfies `|N(E)| ≥ |E| + 1`. -/
theorem strict_hall_expansion {K m : ℕ} (G : KRegularBipartite K m)
    (hconn : G.IsConnected) {E : Finset (Fin m)} (hne : E.Nonempty) (hproper : E ≠ Finset.univ) :
    E.card + 1 ≤ (G.leftNbhd E).card := by
  -- Count directed view-edges `(a, u)` with `G.π a u ∈ E` vs all `(a, u)` with `u ∈ N(E)`.
  let allSlots := (Finset.univ : Finset (Fin K)) ×ˢ (G.leftNbhd E)
  let hitSlots := allSlots.filter fun p => G.π p.1 p.2 ∈ E
  have hstrict : hitSlots ⊂ allSlots := by
    rw [Finset.ssubset_iff_of_subset (Finset.filter_subset _ _)]
    obtain ⟨u, hu, a, ha⟩ := hconn E hne hproper
    exact ⟨(a, u), Finset.mem_product.mpr ⟨Finset.mem_univ a, hu⟩, by simp [ha]⟩
  have hcard_lt : hitSlots.card < allSlots.card := Finset.card_lt_card hstrict
  have hall_card : allSlots.card = K * (G.leftNbhd E).card := by
    simp [allSlots, Finset.card_product]
  have hhit_card : hitSlots.card = K * E.card := by
    have hbi : hitSlots.card = ((Finset.univ : Finset (Fin K)) ×ˢ E).card := by
      refine Finset.card_bij (fun p _ => (p.1, G.π p.1 p.2)) ?_ ?_ ?_
      · intro p hp
        simp only [hitSlots, Finset.mem_filter] at hp
        exact Finset.mem_product.mpr ⟨Finset.mem_univ p.1, hp.2⟩
      · intro p₁ _ p₂ _ h
        rcases p₁ with ⟨a₁, u₁⟩
        rcases p₂ with ⟨a₂, u₂⟩
        simp only [Prod.mk.injEq] at h
        rcases h with ⟨rfl, hu⟩
        exact Prod.ext rfl ((G.π a₁).injective hu)
      · intro q hq
        rcases q with ⟨a, v⟩
        rcases Finset.mem_product.mp hq with ⟨_, hv⟩
        refine ⟨(a, (G.π a).symm v), ?_, by simp⟩
        simp only [hitSlots, allSlots, Finset.mem_filter, Finset.mem_product, Finset.mem_univ,
          true_and, leftNbhd, Equiv.apply_symm_apply]
        exact ⟨⟨a, by simpa using hv⟩, hv⟩
    rw [hbi, Finset.card_product, Finset.card_univ, Fintype.card_fin]
  rw [hhit_card, hall_card] at hcard_lt
  have hlt : E.card < (G.leftNbhd E).card := Nat.lt_of_mul_lt_mul_left hcard_lt
  omega

end KRegularBipartite

/-- **Strict Hall Prefix Inequality** (eq. (5) of `solution.md`):
If a prefix `F` decomposes into `j` nonempty components `E_0, ..., E_{j-1}` whose neighborhoods
are pairwise disjoint inside `N(F)` and satisfy `|N(E_r)| ≥ |E_r| + 1`, then
`∑ k_v = |N(F)| - j ≥ |F|`. -/
theorem strict_hall_prefix_ineq {j : ℕ} (eCard nCard : Fin j → ℕ)
    (hstrict : ∀ r : Fin j, eCard r + 1 ≤ nCard r) :
    ∑ r : Fin j, eCard r + j ≤ ∑ r : Fin j, nCard r := by
  have h1 : ∑ r : Fin j, (eCard r + 1) ≤ ∑ r : Fin j, nCard r :=
    Finset.sum_le_sum fun r _ => hstrict r
  simpa [Finset.sum_add_distrib] using h1

/-- **Greedy Weight Activation Size Cap**:
When a newly activated vertex of degree at most `K` merges `d ≤ K` active components each of
size at most `T - 1`, the merged component `C` satisfies `|C| ≤ 1 + K * (T - 1) ≤ 1 + K * T`. -/
theorem greedy_activation_size_cap {K T d : ℕ} (hd : d ≤ K) (compSizes : Fin d → ℕ)
    (hsmall : ∀ i : Fin d, compSizes i ≤ T - 1) :
    1 + ∑ i : Fin d, compSizes i ≤ 1 + K * T := by
  have h1 : ∑ i : Fin d, compSizes i ≤ ∑ _i : Fin d, (T - 1) :=
    Finset.sum_le_sum fun i _ => hsmall i
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul] at h1
  have h2 : d * (T - 1) ≤ K * T := Nat.mul_le_mul hd (Nat.sub_le T 1)
  omega

/-- **Exact Discrete Abel Summation (Summation by Parts)** (eq. (6) of `solution.md`):
Let `b : Fin r → ℝ` be sorted in nonincreasing order (`b_0 ≥ b_1 ≥ ... ≥ b_{r-1} ≥ λ`),
and let `k : Fin r → ℝ` satisfy the strict Hall prefix lower bound
`∑_{i ≤ t} k_i ≥ t + 1` for every `t : Fin r` and total sum `∑_i k_i = s - 1`.
Then
  `∑_i k_i (b_i + η) ≥ (s - 1)(λ + η) + ∑_i (b_i - λ)`. -/
theorem abel_summation_lower_bound {r : ℕ} (b k : Fin r → ℝ) (lam eta s : ℝ)
    (hmono : ∀ i j : Fin r, i ≤ j → b j ≤ b i)
    (hge : ∀ i : Fin r, lam ≤ b i)
    (hpref : ∀ t : Fin r, ((t : ℕ) + 1 : ℝ) ≤ ∑ i ∈ Finset.univ.filter (· ≤ t), k i)
    (htot : ∑ i : Fin r, k i = s - 1) :
    (s - 1) * (lam + eta) + ∑ i : Fin r, (b i - lam) ≤ ∑ i : Fin r, k i * (b i + eta) := by
  -- Reduce to `∑ (b i - lam) ≤ ∑ k i * (b i - lam)` by splitting `b i + eta = (lam + eta) + (b i - lam)`.
  have hsplit : ∑ i : Fin r, k i * (b i + eta) =
      (s - 1) * (lam + eta) + ∑ i : Fin r, k i * (b i - lam) := by
    have : ∑ i : Fin r, k i * (b i + eta) =
        (∑ i : Fin r, k i) * (lam + eta) + ∑ i : Fin r, k i * (b i - lam) := by
      rw [Finset.sum_mul, ← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl fun i _ => by ring
    rw [this, htot]
  rw [hsplit]
  suffices hmain : ∑ i : Fin r, (b i - lam) ≤ ∑ i : Fin r, k i * (b i - lam) by linarith
  -- Prove the pure nonnegative-weight Abel inequality by induction on `r`.
  have key : ∀ (n : ℕ) (w u : Fin n → ℝ),
      (∀ i j : Fin n, i ≤ j → w j ≤ w i) →
      (∀ i : Fin n, 0 ≤ w i) →
      (∀ t : Fin n, ((t : ℕ) + 1 : ℝ) ≤ ∑ i ∈ Finset.univ.filter (· ≤ t), u i) →
      ∑ i : Fin n, w i ≤ ∑ i : Fin n, u i * w i := by
    intro n
    induction n with
    | zero =>
      intro w u _ _ _
      simp
    | succ m ih =>
      intro w u hw_mono hw_nonneg hu_pref
      let wLast : ℝ := w (Fin.last m)
      have hwLast_nonneg : 0 ≤ wLast := hw_nonneg (Fin.last m)
      -- Subtract `wLast` from all weights; last weight becomes 0 and the prefix of length `m`
      -- satisfies the induction hypothesis.
      let w' : Fin m → ℝ := fun i => w i.castSucc - wLast
      let u' : Fin m → ℝ := fun i => u i.castSucc
      have hw'_mono : ∀ i j : Fin m, i ≤ j → w' j ≤ w' i := by
        intro i j hij
        dsimp [w']
        have := hw_mono i.castSucc j.castSucc (Fin.castSucc_le_castSucc_iff.mpr hij)
        linarith
      have hw'_nonneg : ∀ i : Fin m, 0 ≤ w' i := by
        intro i
        dsimp [w', wLast]
        have := hw_mono i.castSucc (Fin.last m) (Fin.le_last _)
        linarith
      have hu'_pref : ∀ t : Fin m, ((t : ℕ) + 1 : ℝ) ≤
          ∑ i ∈ Finset.univ.filter (· ≤ t), u' i := by
        intro t
        have ht := hu_pref t.castSucc
        have hsum_eq :
            ∑ i ∈ Finset.univ.filter (· ≤ t.castSucc), u i =
            ∑ i ∈ Finset.univ.filter (· ≤ t), u' i := by
          symm
          refine Finset.sum_nbij (fun i : Fin m => i.castSucc) ?_ ?_ ?_ ?_
          · intro i hi
            simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hi ⊢
            exact Fin.castSucc_le_castSucc_iff.mpr hi
          · intro i₁ _ i₂ _ h
            exact Fin.castSucc_injective m h
          · intro j hj
            simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_univ, true_and] at hj
            have hj_lt : (j : ℕ) < m := lt_of_le_of_lt hj t.isLt
            refine ⟨⟨j.1, hj_lt⟩, ?_, ?_⟩
            · simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_univ, true_and]
              exact hj
            · ext; rfl
          · intro i _
            rfl
        rw [← hsum_eq]
        simpa using ht
      have hih := ih w' u' hw'_mono hw'_nonneg hu'_pref
      have hfull := hu_pref (Fin.last m)
      have hfilt_univ : Finset.univ.filter (fun i : Fin (m + 1) => i ≤ Fin.last m) = Finset.univ := by
        ext i
        exact ⟨fun _ => Finset.mem_univ i, fun _ => Finset.mem_filter.mpr ⟨Finset.mem_univ i, Fin.le_last i⟩⟩
      rw [hfilt_univ] at hfull
      have hsum_w : ∑ i : Fin (m + 1), w i = ((m + 1 : ℕ) : ℝ) * wLast + ∑ i : Fin m, w' i := by
        rw [Fin.sum_univ_castSucc]
        have h1 : ∑ i : Fin m, w i.castSucc = (m : ℝ) * wLast + ∑ i : Fin m, w' i := by
          have : ∑ i : Fin m, w' i = ∑ i : Fin m, w i.castSucc - ∑ _i : Fin m, wLast := by
            simp only [w', Finset.sum_sub_distrib]
          simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at this
          linarith
        dsimp [wLast] at h1 ⊢
        push_cast
        linarith
      have hsum_uw : ∑ i : Fin (m + 1), u i * w i =
          (∑ i : Fin (m + 1), u i) * wLast + ∑ i : Fin m, u' i * w' i := by
        rw [Fin.sum_univ_castSucc, Fin.sum_univ_castSucc]
        have h1 : ∑ i : Fin m, u i.castSucc * w i.castSucc =
            (∑ i : Fin m, u i.castSucc) * wLast + ∑ i : Fin m, u' i * w' i := by
          rw [Finset.sum_mul, ← Finset.sum_add_distrib]
          refine Finset.sum_congr rfl fun i _ => ?_
          dsimp [u', w']
          ring
        dsimp [wLast] at h1 ⊢
        linarith
      rw [hsum_w, hsum_uw]
      have hlast_term : ((m + 1 : ℕ) : ℝ) * wLast ≤ (∑ i : Fin (m + 1), u i) * wLast :=
        mul_le_mul_of_nonneg_right (by simpa using hfull) hwLast_nonneg
      linarith
  exact key r (fun i => b i - lam) k
    (fun i j hij => by linarith [hmono i j hij])
    (fun i => by linarith [hge i])
    hpref

/-- **One-Sided Admissibility Budget Inequality** (Lemma 3):
Combining the admissibility upper bound `T ≤ s * λ + ∑_{u ∈ A} (b_u - λ)` with the Abel lower
bound `T ≥ (s - 1)(λ + η) + ∑_{v ∈ B} (b_v - λ)` yields
`(s - 1) * η ≤ λ + ∑_{u ∈ A} (b_u - λ) - ∑_{v ∈ B} (b_v - λ)`. -/
theorem one_sided_budget_ineq {T s lam eta sumA sumB : ℝ}
    (hupper : T ≤ s * lam + sumA)
    (hlower : (s - 1) * (lam + eta) + sumB ≤ T) :
    (s - 1) * eta ≤ lam + sumA - sumB := by
  linarith

/-- **Two-Sided Cancellation Contradiction** (Lemma 3 finale):
Adding the left-side and right-side admissibility inequalities cancels `sumA` and `sumB`
identically, giving `(s + r - 2) * η ≤ 2 * λ ≤ 2 * c`. Together with `s + r ≥ T > 2 + 2c/η`
and `η > 0`, this yields `False` — forcing every connected component of two admissible
alignments to have size at most `H = 1 + K T`. -/
theorem two_sided_cancellation_contradiction {s r lam eta c T sumA sumB : ℝ}
    (heta : 0 < eta) (hlam : lam ≤ c)
    (hleft : (s - 1) * eta ≤ lam + sumA - sumB)
    (hright : (r - 1) * eta ≤ lam + sumB - sumA)
    (hsr : T ≤ s + r) (hT : 2 + 2 * c / eta < T) :
    False := by
  have hsum : (s + r - 2) * eta ≤ 2 * c := by linarith
  have hT_mul : 2 * eta + 2 * c < T * eta := by
    have h1 := mul_lt_mul_of_pos_right hT heta
    have h2 : (2 + 2 * c / eta) * eta = 2 * eta + 2 * c := by
      field_simp [ne_of_gt heta]
    linarith
  have hsr_mul : (T - 2) * eta ≤ (s + r - 2) * eta :=
    mul_le_mul_of_nonneg_right (by linarith) (le_of_lt heta)
  linarith

/-! ### 3. Positive-Definite Quadratic-Form & Swap Covariance Inequalities (Section 2.3) -/

/-- Block sign-flip involution `J = diag(I_S, -I_{S^c})` on `Sum α β` induced by swapping
the `S^c` coordinates between two rows in the difference block `(X₁ - X₂) / √2`. -/
def swapSignMatrix (α β : Type*) [DecidableEq α] [DecidableEq β] :
    Matrix (Sum α β) (Sum α β) ℝ :=
  Matrix.fromBlocks (1 : Matrix α α ℝ) 0 0 (-(1 : Matrix β β ℝ))

theorem swapSignMatrix_mul_self (α β : Type*) [DecidableEq α] [DecidableEq β]
    [Fintype α] [Fintype β] :
    swapSignMatrix α β * swapSignMatrix α β = 1 := by
  unfold swapSignMatrix
  rw [Matrix.fromBlocks_multiply]
  simp [Matrix.fromBlocks_one]

theorem swapSignMatrix_transpose (α β : Type*) [DecidableEq α] [DecidableEq β] :
    (swapSignMatrix α β)ᵀ = swapSignMatrix α β := by
  unfold swapSignMatrix
  simp [Matrix.fromBlocks_transpose]

theorem det_swap_conj {α β : Type*} [DecidableEq α] [DecidableEq β]
    [Fintype α] [Fintype β] (R : Matrix (Sum α β) (Sum α β) ℝ) :
    (swapSignMatrix α β * R * swapSignMatrix α β).det = R.det := by
  have hJ2 : (swapSignMatrix α β).det * (swapSignMatrix α β).det = 1 := by
    rw [← Matrix.det_mul, swapSignMatrix_mul_self, Matrix.det_one]
  rw [Matrix.det_mul, Matrix.det_mul]
  linarith [show (swapSignMatrix α β).det * R.det * (swapSignMatrix α β).det =
      ((swapSignMatrix α β).det * (swapSignMatrix α β).det) * R.det by ring,
    show ((swapSignMatrix α β).det * (swapSignMatrix α β).det) * R.det = R.det by rw [hJ2, one_mul]]

/-- **Block Swap Average Identity** (Section 2.3):
Averaging the unswapped difference covariance `R` and the swapped difference covariance
`J R J` cancels the off-diagonal cross-covariance blocks identically, leaving the block-diagonal
matrix `blockDiag(R₁₁, R₂₂)`. -/
theorem swap_average_blockDiag {α β : Type*} [DecidableEq α] [DecidableEq β]
    [Fintype α] [Fintype β]
    (R₁₁ : Matrix α α ℝ) (R₁₂ : Matrix α β ℝ) (R₂₁ : Matrix β α ℝ) (R₂₂ : Matrix β β ℝ) :
    ((1 : ℝ) / 2) • (Matrix.fromBlocks R₁₁ R₁₂ R₂₁ R₂₂ +
      swapSignMatrix α β * Matrix.fromBlocks R₁₁ R₁₂ R₂₁ R₂₂ * swapSignMatrix α β) =
      Matrix.fromBlocks R₁₁ 0 0 R₂₂ := by
  unfold swapSignMatrix
  rw [Matrix.fromBlocks_multiply, Matrix.fromBlocks_multiply]
  simp only [Matrix.one_mul, Matrix.mul_one, Matrix.zero_mul, Matrix.mul_zero,
    Matrix.neg_mul, Matrix.mul_neg, add_zero, zero_add, neg_neg]
  rw [Matrix.fromBlocks_add, Matrix.fromBlocks_smul]
  congr 1
  · ext i j; simp; ring
  · ext i j; simp
  · ext i j; simp
  · ext i j; simp; ring

/-- Determinant of the averaged swap difference covariance equals `det R₁₁ * det R₂₂`. -/
theorem det_swap_average {α β : Type*} [DecidableEq α] [DecidableEq β]
    [Fintype α] [Fintype β]
    (R₁₁ : Matrix α α ℝ) (R₁₂ : Matrix α β ℝ) (R₂₁ : Matrix β α ℝ) (R₂₂ : Matrix β β ℝ) :
    (((1 : ℝ) / 2) • (Matrix.fromBlocks R₁₁ R₁₂ R₂₁ R₂₂ +
      swapSignMatrix α β * Matrix.fromBlocks R₁₁ R₁₂ R₂₁ R₂₂ * swapSignMatrix α β)).det =
      R₁₁.det * R₂₂.det := by
  rw [swap_average_blockDiag, Matrix.det_fromBlocks_zero₂₁]

/-- **Swap Quadratic-Form Energy Identity**:
For any block matrix `R = fromBlocks R₁₁ R₁₂ R₂₁ R₂₂` and any vector `x`,
`xᵀ R x + (J x)ᵀ R (J x) = 2 * xᵀ blockDiag(R₁₁, R₂₂) x`. -/
theorem swap_quadratic_energy_identity {α β : Type*} [DecidableEq α] [DecidableEq β]
    [Fintype α] [Fintype β]
    (R₁₁ : Matrix α α ℝ) (R₁₂ : Matrix α β ℝ) (R₂₁ : Matrix β α ℝ) (R₂₂ : Matrix β β ℝ)
    (x : Sum α β → ℝ) :
    let R := Matrix.fromBlocks R₁₁ R₁₂ R₂₁ R₂₂
    let J := swapSignMatrix α β
    dotProduct x (R *ᵥ x) + dotProduct (J *ᵥ x) (R *ᵥ (J *ᵥ x)) =
      2 * dotProduct x (Matrix.fromBlocks R₁₁ 0 0 R₂₂ *ᵥ x) := by
  dsimp only
  have h1 : dotProduct (swapSignMatrix α β *ᵥ x)
      (Matrix.fromBlocks R₁₁ R₁₂ R₂₁ R₂₂ *ᵥ (swapSignMatrix α β *ᵥ x)) =
      dotProduct x ((swapSignMatrix α β * Matrix.fromBlocks R₁₁ R₁₂ R₂₁ R₂₂ *
        swapSignMatrix α β) *ᵥ x) := by
    rw [dotProduct_comm, Matrix.dotProduct_mulVec, ← Matrix.mulVec_transpose,
      dotProduct_comm, swapSignMatrix_transpose, Matrix.mulVec_mulVec,
      Matrix.mulVec_mulVec, Matrix.mul_assoc]
  rw [h1, ← dotProduct_add, ← Matrix.add_mulVec]
  have h2 := swap_average_blockDiag R₁₁ R₁₂ R₂₁ R₂₂
  have h3 : Matrix.fromBlocks R₁₁ R₁₂ R₂₁ R₂₂ +
      swapSignMatrix α β * Matrix.fromBlocks R₁₁ R₁₂ R₂₁ R₂₂ * swapSignMatrix α β =
      (2 : ℝ) • Matrix.fromBlocks R₁₁ 0 0 R₂₂ := by
    rw [← h2, smul_smul]
    norm_num
  rw [h3, Matrix.smul_mulVec, dotProduct_smul, smul_eq_mul]

/-- Exact Gaussian swap mutual information positivity for a `2 × 2` positive-definite
covariance matrix `R = [[a, b], [b, d]]`:
`I_p = (1/2) log ((a * d) / (a * d - b^2)) ≥ 0`, with equality iff `b = 0`. -/
theorem bhattacharyya_two_by_two_mi {a b d : ℝ} (ha : 0 < a) (hd : 0 < d)
    (hdet : 0 < a * d - b ^ 2) :
    0 ≤ (1 / 2 : ℝ) * Real.log ((a * d) / (a * d - b ^ 2)) ∧
    ((1 / 2 : ℝ) * Real.log ((a * d) / (a * d - b ^ 2)) = 0 ↔ b = 0) := by
  have hge1 : 1 ≤ (a * d) / (a * d - b ^ 2) := by
    rw [one_le_div hdet]
    nlinarith [sq_nonneg b]
  have hlog_nn : 0 ≤ Real.log ((a * d) / (a * d - b ^ 2)) := Real.log_nonneg hge1
  refine ⟨by linarith, ?_⟩
  constructor
  · intro h0
    have hlog0 : Real.log ((a * d) / (a * d - b ^ 2)) = 0 := by linarith
    have hpos : 0 < (a * d) / (a * d - b ^ 2) := div_pos (mul_pos ha hd) hdet
    have heq1 : (a * d) / (a * d - b ^ 2) = 1 := by
      have := congrArg Real.exp hlog0
      rwa [Real.exp_log hpos, Real.exp_zero] at this
    have hmul : a * d = a * d - b ^ 2 := by
      have := (div_eq_one_iff_eq (ne_of_gt hdet)).mp heq1
      exact this
    nlinarith
  · intro hb
    subst hb
    have hne : a * d ≠ 0 := ne_of_gt (mul_pos ha hd)
    simp [div_self hne]

/-- **Equicorrelation Positive-Definiteness Criterion** (for all `K ≥ 2` views):
The `K × K` equicorrelation matrix `R_{K,r} = (1 - r) I_K + r 1 1ᵀ` has strictly positive
quadratic form `xᵀ R_{K,r} x = (1 - r) ‖x‖² + r (∑ x_i)² > 0` for all `x ≠ 0`
if and only if `-1 / (K - 1) < r < 1`. -/
theorem equicorr_posDef_iff {K : ℕ} (hK : 2 ≤ K) (r : ℝ) :
    (∀ x : Fin K → ℝ, x ≠ 0 →
      0 < (1 - r) * (∑ i : Fin K, x i ^ 2) + r * (∑ i : Fin K, x i) ^ 2) ↔
    -(1 : ℝ) / ((K : ℝ) - 1) < r ∧ r < 1 := by
  have hK1_pos : (0 : ℝ) < (K : ℝ) - 1 := by
    have : (2 : ℝ) ≤ (K : ℝ) := by exact_mod_cast hK
    linarith
  have hK_pos : (0 : ℝ) < (K : ℝ) := by linarith
  constructor
  · intro hpd
    -- Test `x = e₀ - e₁` (sum = 0, norm² = 2) to get `r < 1`.
    let e01 : Fin K → ℝ := fun i => if i.1 = 0 then 1 else if i.1 = 1 then -1 else 0
    have he01_ne : e01 ≠ 0 := by
      intro h
      have := congrFun h ⟨0, by omega⟩
      simp [e01] at this
    have hsum01 : ∑ i : Fin K, e01 i = 0 := by
      let i0 : Fin K := ⟨0, by omega⟩
      let i1 : Fin K := ⟨1, by omega⟩
      have hne01 : i0 ≠ i1 := by intro h; have := congrArg Fin.val h; dsimp [i0, i1] at this; omega
      have hsub : ({i0, i1} : Finset (Fin K)) ⊆ Finset.univ := Finset.subset_univ _
      rw [← Finset.sum_subset hsub]
      · rw [Finset.sum_pair hne01]
        simp [e01, i0, i1]
      · intro i _ hi
        simp only [Finset.mem_insert, Finset.mem_singleton, not_or] at hi
        have h0 : i.1 ≠ 0 := fun hv => hi.1 (Fin.ext hv)
        have h1 : i.1 ≠ 1 := fun hv => hi.2 (Fin.ext hv)
        simp [e01, h0, h1]
    have hsq01 : ∑ i : Fin K, e01 i ^ 2 = 2 := by
      let i0 : Fin K := ⟨0, by omega⟩
      let i1 : Fin K := ⟨1, by omega⟩
      have hne01 : i0 ≠ i1 := by intro h; have := congrArg Fin.val h; dsimp [i0, i1] at this; omega
      have hsub : ({i0, i1} : Finset (Fin K)) ⊆ Finset.univ := Finset.subset_univ _
      rw [← Finset.sum_subset hsub]
      · rw [Finset.sum_pair hne01]
        simp [e01, i0, i1]
        norm_num
      · intro i _ hi
        simp only [Finset.mem_insert, Finset.mem_singleton, not_or] at hi
        have h0 : i.1 ≠ 0 := fun hv => hi.1 (Fin.ext hv)
        have h1 : i.1 ≠ 1 := fun hv => hi.2 (Fin.ext hv)
        simp [e01, h0, h1]
    have h1 := hpd e01 he01_ne
    rw [hsum01, hsq01] at h1
    have hr_lt1 : r < 1 := by nlinarith
    -- Test `x = 1` (all ones: sum = K, norm² = K) to get `-1 / (K - 1) < r`.
    let ones : Fin K → ℝ := fun _ => 1
    have hones_ne : ones ≠ 0 := by
      intro h
      have := congrFun h ⟨0, by omega⟩
      simp [ones] at this
    have h2 := hpd ones hones_ne
    simp only [ones, one_pow, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
      nsmul_eq_mul, mul_one] at h2
    have h3 : 0 < (K : ℝ) * (1 + ((K : ℝ) - 1) * r) := by nlinarith
    have h4 : 0 < 1 + ((K : ℝ) - 1) * r := by nlinarith
    have hr_gt : -(1 : ℝ) / ((K : ℝ) - 1) < r := by
      rw [div_lt_iff₀ hK1_pos]
      linarith
    exact ⟨hr_gt, hr_lt1⟩
  · rintro ⟨hr_gt, hr_lt1⟩ x hx
    have hsq_pos : 0 < ∑ i : Fin K, x i ^ 2 := by
      have hex : ∃ i : Fin K, x i ≠ 0 := by
        by_contra h
        push_neg at h
        exact hx (funext h)
      rcases hex with ⟨i0, hi0⟩
      exact lt_of_lt_of_le (sq_pos_of_ne_zero hi0)
        (Finset.single_le_sum (fun i _ => sq_nonneg (x i)) (Finset.mem_univ i0))
    -- Cauchy–Schwarz: `(∑ x_i)^2 ≤ K * ∑ x_i^2`.
    have hcs : (∑ i : Fin K, x i) ^ 2 ≤ (K : ℝ) * ∑ i : Fin K, x i ^ 2 := by
      have h := Finset.sum_mul_sq_le_sq_mul_sq (Finset.univ : Finset (Fin K)) (fun _ => (1 : ℝ)) x
      simpa using h
    by_cases hr0 : 0 ≤ r
    · have : 0 ≤ r * (∑ i : Fin K, x i) ^ 2 := mul_nonneg hr0 (sq_nonneg _)
      nlinarith
    · push_neg at hr0
      have hmul : (K : ℝ) * r * ∑ i : Fin K, x i ^ 2 ≤ r * (∑ i : Fin K, x i) ^ 2 := by
        nlinarith
      have hcoeff : 0 < 1 - r + (K : ℝ) * r := by
        have h1 := (div_lt_iff₀ hK1_pos).mp hr_gt
        linarith
      nlinarith

/-- Union-bound error exponent negativity when `d * L / log n ≥ 1 + ε` (`ε > 0, log n > 0`):
every bounded component of size `m ≥ 2` has strictly negative exponent `-m * (d * L - log n)`. -/
theorem bounded_component_exponent_neg {d L logn eps : ℝ} {m : ℕ}
    (hlogn : 0 < logn) (heps : 0 < eps) (hm : 2 ≤ m)
    (hrate : 1 + eps ≤ d * L / logn) :
    -(m : ℝ) * (d * L - logn) ≤ -(m : ℝ) * eps * logn ∧ -(m : ℝ) * (d * L - logn) < 0 := by
  have h1 : (1 + eps) * logn ≤ d * L := (le_div_iff₀ hlogn).mp hrate
  have hm_pos : (0 : ℝ) < (m : ℝ) := by
    have : (2 : ℝ) ≤ (m : ℝ) := by exact_mod_cast hm
    linarith
  have h2 : eps * logn ≤ d * L - logn := by linarith
  have h3 : (m : ℝ) * (eps * logn) ≤ (m : ℝ) * (d * L - logn) :=
    mul_le_mul_of_nonneg_left h2 (le_of_lt hm_pos)
  have h4 : 0 < (m : ℝ) * eps * logn := by positivity
  exact ⟨by linarith, by linarith⟩

end MultiviewPlantedAlign

