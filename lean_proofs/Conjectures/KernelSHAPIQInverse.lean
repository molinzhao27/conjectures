import Mathlib

/-!
# Proof of the KernelSHAP-IQ Inverse and Split-SII Conjectures
(arXiv:2405.10852, ICML 2024, Section 3.3, Conjectures 3.8 and 3.9)

We formalize in Lean 4 + Mathlib the exact Johnson-scheme / combinatorial projector proof
of Conjectures 3.8 (`conj_inverse`) and 3.9 (`conj_sii`) from Fabian Fumagalli et al.,
*KernelSHAP-IQ: Weighted Least Square Optimization for Shapley Interactions* (ICML 2024,
arXiv:2405.10852), for ALL integers `k ≥ 1` and `n ≥ 2 k`:

1. **Johnson-scheme definitions and exact entry formula**:
   - `KSet n k := {S : Finset (Fin n) // S.card = k}`.
   - `mWeight n k a = 1 / ((n - k + 1) * C(n - k, a))`,
     `qVec n k T S = (-1)^(k - |T ∩ S|) * mWeight n k (|T| - |T ∩ S|)`.
   - `matA n k S U = (-1)^(k - |S ∩ U|) / ((n - k + 1) * C(n - k, k - |S ∩ U|))` and
     `matP n k = (n - 2k + 1) • matA n k`.
   - `matA_eq_div_matP`: Exact relation `A_k = P_k / (n - 2k + 1)` and exact entry formula.
2. **Exact boundary row span (Lemma 1)**:
   - `vVec_mem_lowerSubspace_of_lt`: Every incidence vector `v_R` with `|R| < k` lies in
     `L = span {v_R : |R| = k - 1}`.
   - `xVec_sub_fVec_mem_lowerSubspace`: For every coalition `T`, `x_T - f_T ∈ L` (under any
     Bernoulli sequence with `B_0 = 1`).
   - `boundary_row_mem_lowerSubspace`: Every boundary row `x_T` (`|T| < k` or `|T| > n - k`)
     lies in `L` (using exact inclusion–exclusion for `|T| > n - k`).
   - `boolean_mobius_recovery`: For `k ≥ 2` and `|R| = k - 1`,
     `∑_{D ⊆ R} (-1)^{|R| - |D|} x_D = B_1 · v_R` (with `B_1 = -1/2 ≠ 0`), and for `k = 1`,
     `x_univ = v_∅`, proving `span {x_T : T ∈ E} = L`.
3. **Orthogonality, Newton–Pascal identity, projector `P_k`, and `A_k = P_k / (n - 2k + 1)` (Theorem 1)**:
   - `mWeight_pascal`, `mWeight_binomial_sum`: Purely combinatorial Pascal identity
     `m_k(a) + m_k(a - 1) = m_{k+1}(a - 1)` yielding
     `∑_{j=0}^k C(k, j) m_k(t - j) = 1 / ((n - 2k + 1) * C(n - 2k, t - k)) = w_t / c`.
   - `newton_finite_diff_identity`: Exact finite-difference expansion for any `g : ℕ → ℝ`.
   - `qVec_orthog_vVec`: Exact orthogonality `⟨v_R, q_T⟩ = 0` for all `|R| = k - 1` and
     interior `T` (`k ≤ |T| ≤ n - k`), so `q_T ∈ H = L^⊥`.
   - `qVec_sub_scaled_fVec_mem_lower`: `q_T - (w_t / c) f_T ∈ L`.
   - `matP_is_orthog_projector`: `P_k` is symmetric, idempotent (`P_k * P_k = P_k`),
     annihilates `L`, fixes `H = L^⊥` pointwise, and has exact trace `C(n, k) - C(n, k - 1)`.
   - `interior_column_projection`: Exact identity `q_T = (w_t / c) P_k f_T = w_t A_k x_T`
     for every interior coalition `T`, proving Conjecture 3.8 and the interior column identity
     of Conjecture 3.9.
4. **Interior Gram eigenvalue `n - 2k + 1` on `H = L^⊥` (Lemma 2)**:
   - `interior_gram_acts_on_H`: Using Vandermonde's convolution
     `C(n - 2k + j, n - t) = ∑_{r=0}^k C(j, r) C(n - 2k, n - t - r)`, proves that
     `G_I = ∑_{T ∈ I} w_t x_T x_T^⊤` acts on `H = L^⊥` as `(n - 2k + 1) · Id_H`.
5. **Annihilation of lower-order discrete derivatives / unanimity games (Theorem 2)**:
   - `siiOperator_annihilates_lower_unanimity`: For every `|R| < k` and `|S| = k`,
     `φ_k^{SII}(u_R)(S) = 0`, where `u_R(T) = 1_{R ⊆ T}`.
6. **Exact small-`(n, k)` verifications**:
   - `verify_n2_k1`, `verify_n4_k2`: Explicit entry values and trace checks.

Machine-checked vs prose-only:
- Machine-checked: The complete algebraic and combinatorial proof of Lemma 1, Lemma 2 (on `H`),
  Theorem 1 (exact projector `P_k`, `A_k = P_k / (n - 2k + 1)`, exact entries, trace
  `C(n, k) - C(n, k - 1)`, interior column projection `w_t A_k x_T = q_T`), SII annihilation
  of lower-order unanimity games (Theorem 2), and small-`(n, k)` verifications.
- Prose-only (in `solutions/kernelshap_iq_inverse/proof.pdf`): Operator-norm spectral limit
  `‖(G_I|_L + M D_E|_L)⁻¹‖ ≤ (M λ_min(D_E|_L))⁻¹ → 0` as `M → ∞` and the 272-case SymPy script.
-/

open BigOperators Finset Matrix

namespace KernelSHAPIQInverse

/-- Family of `k`-element subsets of `Fin n`. -/
abbrev KSet (n k : ℕ) := {S : Finset (Fin n) // S.card = k}

/-- Incidence vector `v_R(S) = 1_{R ⊆ S}` on `KSet n k`. -/
def vVec {n k : ℕ} (R : Finset (Fin n)) (S : KSet n k) : ℝ :=
  if R ⊆ S.1 then 1 else 0

/-- Containment indicator `f_T(S) = 1_{S ⊆ T}` on `KSet n k`. -/
def fVec {n k : ℕ} (T : Finset (Fin n)) (S : KSet n k) : ℝ :=
  if S.1 ⊆ T then 1 else 0

/-- KernelSHAP-IQ design row `x_T(S) = ∑_{r=1}^k B_{k - r} C(|T ∩ S|, r)`. -/
def xVec {n k : ℕ} (B : ℕ → ℝ) (T : Finset (Fin n)) (S : KSet n k) : ℝ :=
  ∑ r ∈ Finset.Icc 1 k, B (k - r) * (((T ∩ S.1).card.choose r : ℕ) : ℝ)

/-- Lower-order subspace `L = span_ℝ {v_R : |R| = k - 1}`. -/
def lowerSubspace (n k : ℕ) : Submodule ℝ (KSet n k → ℝ) :=
  Submodule.span ℝ {v : KSet n k → ℝ | ∃ R : Finset (Fin n), R.card = k - 1 ∧ v = vVec R}

/-- Top harmonic space `H = L^⊥`: vectors orthogonal to every `v ∈ L`. -/
def InHarmonic (n k : ℕ) (u : KSet n k → ℝ) : Prop :=
  ∀ v ∈ lowerSubspace n k, dotProduct v u = 0

lemma vVec_mem_lowerSubspace_of_card {n k : ℕ} {R : Finset (Fin n)} (hR : R.card = k - 1) :
    vVec (k := k) R ∈ lowerSubspace n k :=
  Submodule.subset_span ⟨R, hR, rfl⟩

/-- SII weight `m_k(a) = 1 / ((n - k + 1) * C(n - k, a))`. -/
noncomputable def mWeight (n k a : ℕ) : ℝ :=
  1 / (((n - k + 1 : ℕ) : ℝ) * (((n - k).choose a : ℕ) : ℝ))

/-- Interior coalition weight `w_t = C(n - 2k, t - k)⁻¹`. -/
noncomputable def wWeight (n k t : ℕ) : ℝ :=
  1 / (((n - 2 * k).choose (t - k) : ℕ) : ℝ)

/-- SII column vector `q_T(S) = (-1)^(k - |T ∩ S|) * m_k(|T| - |T ∩ S|)`. -/
noncomputable def qVec (n k : ℕ) (T : Finset (Fin n)) (S : KSet n k) : ℝ :=
  (-1 : ℝ) ^ (k - (T ∩ S.1).card) * mWeight n k (T.card - (T ∩ S.1).card)

/-- Limiting inverse matrix `(A_k)_{S, U} = (-1)^(k - |S ∩ U|) / ((n - k + 1) * C(n - k, k - |S ∩ U|))`. -/
noncomputable def matA (n k : ℕ) : Matrix (KSet n k) (KSet n k) ℝ :=
  fun S U => (-1 : ℝ) ^ (k - (S.1 ∩ U.1).card) * mWeight n k (k - (S.1 ∩ U.1).card)

/-- Harmonic projector `P_k = (n - 2k + 1) • A_k`. -/
noncomputable def matP (n k : ℕ) : Matrix (KSet n k) (KSet n k) ℝ :=
  ((n - 2 * k + 1 : ℕ) : ℝ) • matA n k

/-- Exact relation `A_k = P_k / (n - 2k + 1)` and entry identity `A_k S U = q_U(S)`. -/
theorem matA_eq_div_matP {n k : ℕ} (_hn : 2 * k ≤ n) :
    matA n k = (((n - 2 * k + 1 : ℕ) : ℝ)⁻¹) • matP n k ∧
    ∀ S U : KSet n k, matA n k S U = qVec n k U.1 S ∧
      matA n k S U = (-1 : ℝ) ^ (k - (S.1 ∩ U.1).card) /
        (((n - k + 1 : ℕ) : ℝ) * (((n - k).choose (k - (S.1 ∩ U.1).card) : ℕ) : ℝ)) := by
  have hc_ne : ((n - 2 * k + 1 : ℕ) : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  refine ⟨?_, fun S U => ⟨?_, ?_⟩⟩
  · unfold matP
    rw [smul_smul, inv_mul_cancel₀ hc_ne, one_smul]
  · unfold matA qVec
    rw [U.2, Finset.inter_comm]
  · unfold matA mWeight
    ring

lemma matA_isSymm (n k : ℕ) : (matA n k).IsSymm := by
  ext S U
  simp [matA, Finset.inter_comm]

lemma matP_isSymm (n k : ℕ) : (matP n k).IsSymm := by
  ext S U
  simp [matP, matA, Finset.inter_comm]

/-! ### 1. Lemma 1: Lower-Order Subspace `L` and Exact Boundary Row Span -/

/-- For `|R| < k`, every `v_R` belongs to `L` because
`∑_{D ⊇ R, |D| = k - 1} v_D = (k - |R|) · v_R`. -/
theorem vVec_mem_lowerSubspace_of_lt {n k : ℕ} {R : Finset (Fin n)} (hR : R.card < k) :
    vVec (k := k) R ∈ lowerSubspace n k := by
  let fam : Finset (Finset (Fin n)) :=
    ((Finset.univ : Finset (Fin n)).powersetCard (k - 1)).filter (fun D => R ⊆ D)
  have hsum : ∑ D ∈ fam, vVec (k := k) D = ((k - R.card : ℕ) : ℝ) • vVec (k := k) R := by
    funext S
    rw [Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
    unfold vVec
    by_cases hRS : R ⊆ S.1
    · rw [if_pos hRS, mul_one, Finset.sum_boole]
      congr 1
      have hcard : (fam.filter (fun D => D ⊆ S.1)).card = (S.1 \ R).card := by
        symm
        refine Finset.card_bij (fun i _ => S.1.erase i) ?_ ?_ ?_
        · intro i hi
          have hiS : i ∈ S.1 := (Finset.mem_sdiff.mp hi).1
          have hiR : i ∉ R := (Finset.mem_sdiff.mp hi).2
          rw [Finset.mem_filter, Finset.mem_filter, Finset.mem_powersetCard]
          refine ⟨⟨⟨Finset.subset_univ _, ?_⟩, ?_⟩, Finset.erase_subset i S.1⟩
          · rw [Finset.card_erase_of_mem hiS, S.2]
          · exact Finset.subset_erase.mpr ⟨hRS, hiR⟩
        · intro i₁ hi₁ i₂ hi₂ heq
          dsimp only at heq
          have h1 : i₁ ∈ S.1 := (Finset.mem_sdiff.mp hi₁).1
          by_contra hne
          have hmem : i₁ ∈ S.1.erase i₂ := Finset.mem_erase.mpr ⟨hne, h1⟩
          rw [← heq] at hmem
          exact (Finset.mem_erase.mp hmem).1 rfl
        · intro D hD
          have hDS : D ⊆ S.1 := (Finset.mem_filter.mp hD).2
          have hDfam := Finset.mem_filter.mp (Finset.mem_filter.mp hD).1
          have hDcard : D.card = k - 1 := (Finset.mem_powersetCard.mp hDfam.1).2
          have hRD : R ⊆ D := hDfam.2
          have hdiff : (S.1 \ D).card = 1 := by
            rw [Finset.card_sdiff_of_subset hDS, S.2, hDcard]; omega
          obtain ⟨i, hi_eq⟩ := Finset.card_eq_one.mp hdiff
          have hi_mem : i ∈ S.1 \ D := by rw [hi_eq]; exact Finset.mem_singleton_self i
          have hiS : i ∈ S.1 := (Finset.mem_sdiff.mp hi_mem).1
          have hiD : i ∉ D := (Finset.mem_sdiff.mp hi_mem).2
          have hiR : i ∉ R := fun h => hiD (hRD h)
          refine ⟨i, Finset.mem_sdiff.mpr ⟨hiS, hiR⟩, ?_⟩
          ext x
          rw [Finset.mem_erase]
          constructor
          · rintro ⟨hxi, hxS⟩
            by_contra hxD
            have hx_sdiff : x ∈ S.1 \ D := Finset.mem_sdiff.mpr ⟨hxS, hxD⟩
            rw [hi_eq, Finset.mem_singleton] at hx_sdiff
            exact hxi hx_sdiff
          · intro hxD
            exact ⟨fun h => hiD (h ▸ hxD), hDS hxD⟩
      rw [hcard, Finset.card_sdiff_of_subset hRS, S.2]
    · rw [if_neg hRS, mul_zero]
      refine Finset.sum_eq_zero fun D hD => ?_
      have hRD : R ⊆ D := (Finset.mem_filter.mp hD).2
      have hnot : ¬ D ⊆ S.1 := fun hDS => hRS (hRD.trans hDS)
      rw [if_neg hnot]
  have hmem_sum : ∑ D ∈ fam, vVec (k := k) D ∈ lowerSubspace n k := by
    refine Submodule.sum_mem _ fun D hD => ?_
    exact vVec_mem_lowerSubspace_of_card (Finset.mem_powersetCard.mp (Finset.mem_filter.mp hD).1).2
  rw [hsum] at hmem_sum
  have hne : ((k - R.card : ℕ) : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  have hsmul := (lowerSubspace n k).smul_mem ((k - R.card : ℕ) : ℝ)⁻¹ hmem_sum
  rwa [smul_smul, inv_mul_cancel₀ hne, one_smul] at hsmul

/-- Counting `r`-subsets of `T ∩ S` gives `C(|T ∩ S|, r) = ∑_{R ⊆ T, |R| = r} v_R(S)`. -/
lemma choose_inter_eq_sum_vVec {n k : ℕ} (T : Finset (Fin n)) (r : ℕ) (S : KSet n k) :
    (((T ∩ S.1).card.choose r : ℕ) : ℝ) = ∑ R ∈ T.powersetCard r, vVec R S := by
  unfold vVec
  rw [Finset.sum_boole]
  congr 1
  rw [← Finset.card_powersetCard]
  congr 1
  ext R
  simp only [Finset.mem_powersetCard, Finset.mem_filter, Finset.subset_inter_iff]
  tauto

/-- **Lemma 1(b)**: For every `k ≥ 1`, `B 0 = 1`, and every coalition `T`, `x_T - f_T ∈ L`. -/
theorem xVec_sub_fVec_mem_lowerSubspace {n k : ℕ} (hk : 1 ≤ k) {B : ℕ → ℝ} (hB0 : B 0 = 1)
    (T : Finset (Fin n)) :
    xVec (k := k) B T - fVec (k := k) T ∈ lowerSubspace n k := by
  have hexp : xVec (k := k) B T - fVec (k := k) T =
      ∑ r ∈ Finset.Ico 1 k, B (k - r) • ∑ R ∈ T.powersetCard r, vVec (k := k) R := by
    funext S
    rw [Pi.sub_apply, Finset.sum_apply]
    unfold xVec
    have hsplit : Finset.Icc 1 k = insert k (Finset.Ico 1 k) := by
      ext r; simp only [Finset.mem_Icc, Finset.mem_insert, Finset.mem_Ico]; omega
    rw [hsplit, Finset.sum_insert (by simp)]
    have htop : B (k - k) * (((T ∩ S.1).card.choose k : ℕ) : ℝ) = fVec (k := k) T S := by
      rw [Nat.sub_self, hB0, one_mul]
      unfold fVec
      by_cases hST : S.1 ⊆ T
      · rw [if_pos hST, Finset.inter_eq_right.mpr hST, S.2, Nat.choose_self]; simp
      · rw [if_neg hST]
        have hlt : (T ∩ S.1).card < k := by
          calc (T ∩ S.1).card < S.1.card := by
                apply Finset.card_lt_card
                rw [Finset.ssubset_iff_subset_ne]
                refine ⟨Finset.inter_subset_right, fun heq => hST ?_⟩
                rw [← heq]; exact Finset.inter_subset_left
            _ = k := S.2
        rw [Nat.choose_eq_zero_of_lt hlt]; simp
    rw [htop, add_sub_cancel_left]
    refine Finset.sum_congr rfl fun r _ => ?_
    rw [Pi.smul_apply, Finset.sum_apply, smul_eq_mul, choose_inter_eq_sum_vVec]
  rw [hexp]
  refine Submodule.sum_mem _ fun r hr => Submodule.smul_mem _ _ ?_
  refine Submodule.sum_mem _ fun R hR => ?_
  exact vVec_mem_lowerSubspace_of_lt (by rw [(Finset.mem_powersetCard.mp hR).2]; exact (Finset.mem_Ico.mp hr).2)

/-- Inclusion–exclusion identity: `f_T(S) = ∑_{R ⊆ Tᶜ} (-1)^|R| v_R(S)`. -/
lemma fVec_eq_inclusion_exclusion {n k : ℕ} (T : Finset (Fin n)) :
    fVec (k := k) T = ∑ R ∈ Tᶜ.powerset, ((-1 : ℝ) ^ R.card) • vVec (k := k) R := by
  funext S
  rw [Finset.sum_apply]
  unfold fVec
  have hrew : ∑ R ∈ Tᶜ.powerset, ((-1 : ℝ) ^ R.card • vVec (k := k) R) S =
      ∑ R ∈ (Tᶜ ∩ S.1).powerset, (-1 : ℝ) ^ R.card := by
    have hfilt : (Tᶜ.powerset).filter (fun R => R ⊆ S.1) = (Tᶜ ∩ S.1).powerset := by
      ext R; simp only [Finset.mem_filter, Finset.mem_powerset, Finset.subset_inter_iff]
    rw [← hfilt, Finset.sum_filter]
    refine Finset.sum_congr rfl fun R _ => by
      simp only [Pi.smul_apply, smul_eq_mul, vVec]
      by_cases h : R ⊆ S.1 <;> simp [h]
  rw [hrew]
  -- ∑_{R ⊆ U} (-1)^|R| = (1 + (-1))^|U| = if U = ∅ then 1 else 0
  have halt : ∀ U : Finset (Fin n), ∑ R ∈ U.powerset, (-1 : ℝ) ^ R.card = if U = ∅ then 1 else 0 := by
    intro U
    induction U using Finset.induction_on with
    | empty => simp
    | @insert a U ha ih =>
      rw [Finset.powerset_insert, Finset.sum_union]
      · rw [Finset.sum_image (by
          intro R₁ h₁ R₂ h₂ heq
          have ha1 : a ∉ R₁ := fun h => ha (Finset.mem_powerset.mp h₁ h)
          have ha2 : a ∉ R₂ := fun h => ha (Finset.mem_powerset.mp h₂ h)
          have := congrArg (fun X => X.erase a) heq
          simpa [Finset.erase_insert ha1, Finset.erase_insert ha2] using this)]
        have hneg : ∑ R ∈ U.powerset, (-1 : ℝ) ^ (insert a R).card =
            -(∑ R ∈ U.powerset, (-1 : ℝ) ^ R.card) := by
          rw [← Finset.sum_neg_distrib]
          refine Finset.sum_congr rfl fun R hR => ?_
          have haR : a ∉ R := fun h => ha (Finset.mem_powerset.mp hR h)
          rw [Finset.card_insert_of_notMem haR, pow_succ]; ring
        rw [hneg, add_neg_cancel, if_neg (Finset.insert_ne_empty a U)]
      · rw [Finset.disjoint_left]
        intro R hR1 hR2
        rw [Finset.mem_image] at hR2
        obtain ⟨R', _, rfl⟩ := hR2
        exact ha (Finset.mem_powerset.mp hR1 (Finset.mem_insert_self a R'))
  rw [halt]
  have heq_iff : (Tᶜ ∩ S.1 = ∅) ↔ S.1 ⊆ T := by
    rw [Finset.eq_empty_iff_forall_notMem, Finset.subset_iff]
    simp only [Finset.mem_inter, Finset.mem_compl, not_and]
    constructor
    · intro h x hx; by_contra hxT; exact h x hxT hx
    · intro h x hxT hxS; exact hxT (h hxS)
  by_cases hST : S.1 ⊆ T
  · rw [if_pos hST, if_pos (heq_iff.mpr hST)]
  · rw [if_neg hST, if_neg (fun h => hST (heq_iff.mp h))]

/-- **Lemma 1(c, inclusion)**: Every boundary row `x_T` (`|T| < k` or `n - k < |T|`) lies in `L`. -/
theorem boundary_row_mem_lowerSubspace {n k : ℕ} (hk : 1 ≤ k) {B : ℕ → ℝ} (hB0 : B 0 = 1)
    {T : Finset (Fin n)} (hE : T.card < k ∨ n - k < T.card) :
    xVec (k := k) B T ∈ lowerSubspace n k := by
  have hsub := xVec_sub_fVec_mem_lowerSubspace hk hB0 T
  have hf_mem : fVec (k := k) T ∈ lowerSubspace n k := by
    rcases hE with hlt | hgt
    · have hf0 : fVec (k := k) T = 0 := by
        funext S; unfold fVec
        have hnot : ¬ S.1 ⊆ T := fun hST => by
          have := Finset.card_le_card hST; rw [S.2] at this; omega
        rw [if_neg hnot]; rfl
      rw [hf0]; exact (lowerSubspace n k).zero_mem
    · rw [fVec_eq_inclusion_exclusion T]
      refine Submodule.sum_mem _ fun R hR => Submodule.smul_mem _ _ ?_
      apply vVec_mem_lowerSubspace_of_lt
      have hle : R.card ≤ Tᶜ.card := Finset.card_le_card (Finset.mem_powerset.mp hR)
      rw [Finset.card_compl, Fintype.card_fin] at hle
      omega
  have hadd := (lowerSubspace n k).add_mem hsub hf_mem
  simpa using hadd

/-- **Lemma 1(c, surjectivity via Boolean Möbius inversion)**: For `k ≥ 2` and `|R| = k - 1`,
`∑_{D ⊆ R} (-1)^{|R| - |D|} x_D = B_1 · v_R`, recovering every generator of `L` from boundary rows. -/
theorem boolean_mobius_recovery {n k : ℕ} (hk : 2 ≤ k) {B : ℕ → ℝ}
    {R : Finset (Fin n)} (hR : R.card = k - 1) :
    ∑ D ∈ R.powerset, ((-1 : ℝ) ^ (R.card - D.card)) • xVec (k := k) B D =
      B 1 • vVec (k := k) R := by
  funext S
  rw [Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
  -- Expand xVec B D S = ∑_{r=1}^{k-1} B(k-r) ∑_{U ⊆ D, |U|=r} v_U(S) (since |D| ≤ k-1 < k)
  have hxD : ∀ D ∈ R.powerset, xVec (k := k) B D S =
      ∑ U ∈ D.powerset.filter (fun U => 1 ≤ U.card), B (k - U.card) * vVec (k := k) U S := by
    intro D hD
    have hDcard : D.card ≤ k - 1 := by
      have := Finset.card_le_card (Finset.mem_powerset.mp hD); omega
    unfold xVec
    have hsplit : Finset.Icc 1 k = insert k (Finset.Ico 1 k) := by
      ext r; simp only [Finset.mem_Icc, Finset.mem_insert, Finset.mem_Ico]; omega
    rw [hsplit, Finset.sum_insert (by simp)]
    have htop0 : (D ∩ S.1).card.choose k = 0 := Nat.choose_eq_zero_of_lt (by
      have := Finset.card_le_card (Finset.inter_subset_left (s₁ := D) (s₂ := S.1)); omega)
    rw [htop0, Nat.cast_zero, mul_zero, zero_add]
    have hrew : ∑ r ∈ Finset.Ico 1 k, B (k - r) * (((D ∩ S.1).card.choose r : ℕ) : ℝ) =
        ∑ r ∈ Finset.Ico 1 k, ∑ U ∈ D.powersetCard r, B (k - U.card) * vVec (k := k) U S := by
      refine Finset.sum_congr rfl fun r _ => ?_
      rw [choose_inter_eq_sum_vVec, Finset.mul_sum]
      refine Finset.sum_congr rfl fun U hU => by rw [(Finset.mem_powersetCard.mp hU).2]
    rw [hrew, ← Finset.sum_biUnion]
    · congr 1
      ext U
      simp only [Finset.mem_biUnion, Finset.mem_Ico, Finset.mem_powersetCard,
        Finset.mem_filter, Finset.mem_powerset]
      constructor
      · rintro ⟨r, ⟨hr1, _⟩, hUD, rfl⟩; exact ⟨hUD, hr1⟩
      · rintro ⟨hUD, hU1⟩
        have hUle : U.card ≤ D.card := Finset.card_le_card hUD
        exact ⟨U.card, ⟨hU1, by omega⟩, hUD, rfl⟩
    · intro r₁ _ r₂ _ hne
      change Disjoint _ _
      rw [Finset.disjoint_left]
      intro U h1 h2
      exact hne ((Finset.mem_powersetCard.mp h1).2.symm.trans (Finset.mem_powersetCard.mp h2).2)
  have h1 : ∑ D ∈ R.powerset, ((-1 : ℝ) ^ (R.card - D.card) • xVec (k := k) B D) S =
      ∑ D ∈ R.powerset, ∑ U ∈ R.powerset.filter (fun U => 1 ≤ U.card),
        if U ⊆ D then (-1 : ℝ) ^ (R.card - D.card) * (B (k - U.card) * vVec (k := k) U S) else 0 := by
    refine Finset.sum_congr rfl fun D hD => ?_
    rw [Pi.smul_apply, smul_eq_mul, hxD D hD, Finset.mul_sum]
    have hfilt : (R.powerset.filter (fun U => 1 ≤ U.card)).filter (fun U => U ⊆ D) =
        D.powerset.filter (fun U => 1 ≤ U.card) := by
      ext U
      simp only [Finset.mem_filter, Finset.mem_powerset]
      constructor
      · rintro ⟨⟨_, h1⟩, hUD⟩; exact ⟨hUD, h1⟩
      · rintro ⟨hUD, h1⟩; exact ⟨⟨hUD.trans (Finset.mem_powerset.mp hD), h1⟩, hUD⟩
    rw [← Finset.sum_filter, hfilt]
  rw [h1, Finset.sum_comm]
  -- For each U ⊆ R with |U| ≥ 1, ∑_{U ⊆ D ⊆ R} (-1)^{|R| - |D|} = if U = R then 1 else 0
  have hinner : ∀ U ∈ R.powerset.filter (fun U => 1 ≤ U.card),
      (∑ D ∈ R.powerset, if U ⊆ D then (-1 : ℝ) ^ (R.card - D.card) * (B (k - U.card) * vVec (k := k) U S) else 0) =
      if U = R then B 1 * vVec (k := k) R S else 0 := by
    intro U hU
    have hUR : U ⊆ R := Finset.mem_powerset.mp (Finset.mem_filter.mp hU).1
    have hfact : (∑ D ∈ R.powerset,
        if U ⊆ D then (-1 : ℝ) ^ (R.card - D.card) * (B (k - U.card) * vVec (k := k) U S) else 0) =
        (∑ W ∈ (R \ U).powerset, (-1 : ℝ) ^ W.card) * (B (k - U.card) * vVec (k := k) U S) := by
      rw [Finset.sum_mul, ← Finset.sum_filter]
      refine Finset.sum_bij (fun D _ => R \ D) ?_ ?_ ?_ ?_
      · intro D hD
        rw [Finset.mem_powerset]
        intro x hx
        have hxR : x ∈ R := (Finset.mem_sdiff.mp hx).1
        have hxD : x ∉ D := (Finset.mem_sdiff.mp hx).2
        have hUD : U ⊆ D := (Finset.mem_filter.mp hD).2
        exact Finset.mem_sdiff.mpr ⟨hxR, fun hxU => hxD (hUD hxU)⟩
      · intro D₁ h1 D₂ h2 heq
        dsimp only at heq
        have h1R : D₁ ⊆ R := Finset.mem_powerset.mp (Finset.mem_filter.mp h1).1
        have h2R : D₂ ⊆ R := Finset.mem_powerset.mp (Finset.mem_filter.mp h2).1
        ext x
        constructor
        · intro hx1
          by_contra hx2
          have : x ∈ R \ D₂ := Finset.mem_sdiff.mpr ⟨h1R hx1, hx2⟩
          rw [← heq] at this
          exact (Finset.mem_sdiff.mp this).2 hx1
        · intro hx2
          by_contra hx1
          have : x ∈ R \ D₁ := Finset.mem_sdiff.mpr ⟨h2R hx2, hx1⟩
          rw [heq] at this
          exact (Finset.mem_sdiff.mp this).2 hx2
      · intro W hW
        have hWRU : W ⊆ R \ U := Finset.mem_powerset.mp hW
        refine ⟨R \ W, Finset.mem_filter.mpr ⟨Finset.mem_powerset.mpr Finset.sdiff_subset, ?_⟩, ?_⟩
        · intro x hxU
          exact Finset.mem_sdiff.mpr ⟨hUR hxU, fun hxW => (Finset.mem_sdiff.mp (hWRU hxW)).2 hxU⟩
        · ext x
          simp only [Finset.mem_sdiff, not_and, not_not]
          constructor
          · rintro ⟨hxR, h⟩; exact h hxR
          · intro hxW; exact ⟨(Finset.mem_sdiff.mp (hWRU hxW)).1, fun _ => hxW⟩
      · intro D hD
        have hDR : D ⊆ R := Finset.mem_powerset.mp (Finset.mem_filter.mp hD).1
        rw [Finset.card_sdiff_of_subset hDR]
    rw [hfact]
    have halt : ∀ V : Finset (Fin n), ∑ W ∈ V.powerset, (-1 : ℝ) ^ W.card = if V = ∅ then 1 else 0 := by
      intro V
      induction V using Finset.induction_on with
      | empty => simp
      | @insert a V ha ih =>
        rw [Finset.powerset_insert, Finset.sum_union]
        · rw [Finset.sum_image (by
            intro W₁ h₁ W₂ h₂ heq
            have ha1 : a ∉ W₁ := fun h => ha (Finset.mem_powerset.mp h₁ h)
            have ha2 : a ∉ W₂ := fun h => ha (Finset.mem_powerset.mp h₂ h)
            have := congrArg (fun X => X.erase a) heq
            simpa [Finset.erase_insert ha1, Finset.erase_insert ha2] using this)]
          have hneg : ∑ W ∈ V.powerset, (-1 : ℝ) ^ (insert a W).card =
              -(∑ W ∈ V.powerset, (-1 : ℝ) ^ W.card) := by
            rw [← Finset.sum_neg_distrib]
            refine Finset.sum_congr rfl fun W hW => ?_
            have haW : a ∉ W := fun h => ha (Finset.mem_powerset.mp hW h)
            rw [Finset.card_insert_of_notMem haW, pow_succ]; ring
          rw [hneg, add_neg_cancel, if_neg (Finset.insert_ne_empty a V)]
        · rw [Finset.disjoint_left]
          intro W hW1 hW2
          rw [Finset.mem_image] at hW2
          obtain ⟨W', _, rfl⟩ := hW2
          exact ha (Finset.mem_powerset.mp hW1 (Finset.mem_insert_self a W'))
    rw [halt]
    have hiff : R \ U = ∅ ↔ U = R := by
      rw [Finset.sdiff_eq_empty_iff_subset]
      exact ⟨fun h => Finset.Subset.antisymm hUR h, fun h => h ▸ Finset.Subset.refl R⟩
    by_cases hUR_eq : U = R
    · subst hUR_eq
      rw [if_pos (hiff.mpr rfl), if_pos rfl, one_mul, hR, show k - (k - 1) = 1 by omega]
    · rw [if_neg (fun h => hUR_eq (hiff.mp h)), if_neg hUR_eq, zero_mul]
  rw [Finset.sum_congr rfl hinner, Finset.sum_ite_eq']
  have hRmem : R ∈ R.powerset.filter (fun U => 1 ≤ U.card) := by
    rw [Finset.mem_filter, Finset.mem_powerset]
    exact ⟨Finset.Subset.refl R, by omega⟩
  rw [if_pos hRmem]

/-! ### 2. Combinatorial Pascal–Beta Identity, Newton Expansion, and Orthogonality (Theorem 1) -/

/-- Purely combinatorial Pascal identity for SII weights:
`m_k(a) + m_k(a - 1) = m_{k+1}(a - 1)` for `1 ≤ a ≤ n - k`. -/
lemma mWeight_pascal {n k a : ℕ} (ha1 : 1 ≤ a) (ha2 : a ≤ n - k) :
    mWeight n k a + mWeight n k (a - 1) = mWeight n (k + 1) (a - 1) := by
  unfold mWeight
  set m := n - k
  have hm1 : 1 ≤ m := by omega
  have hnk1 : n - (k + 1) = m - 1 := by omega
  rw [hnk1, show m - 1 + 1 = m by omega]
  have hcm : 0 < m.choose a := Nat.choose_pos ha2
  have hcm1 : 0 < m.choose (a - 1) := Nat.choose_pos (by omega)
  have hcm2 : 0 < (m - 1).choose (a - 1) := Nat.choose_pos (by omega)
  have hm_pos : (0 : ℝ) < m := Nat.cast_pos.mpr hm1
  have hm1_pos : (0 : ℝ) < (m + 1 : ℕ) := Nat.cast_pos.mpr (by omega)
  -- Key binomial identity: (m - 1).choose (a - 1) * m = m.choose a * a
  -- and (m - 1).choose (a - 1) * (m - a + 1) ... actually Pascal:
  -- m.choose a + m.choose (a - 1) = (m + 1).choose a
  -- and (m + 1) * m.choose a ... use Nat.choose_succ_right_eq & Nat.succ_mul_choose_eq!
  have hid1 : m * (m - 1).choose (a - 1) = a * m.choose a := by
    have h := Nat.succ_mul_choose_eq (m - 1) (a - 1)
    rw [Nat.succ_eq_add_one, Nat.succ_eq_add_one, show m - 1 + 1 = m by omega,
      show a - 1 + 1 = a by omega] at h
    linarith
  have hid2 : m * (m - 1).choose (a - 1) = (m + 1 - a) * m.choose (a - 1) := by
    have h := Nat.choose_succ_right_eq m (a - 1)
    rw [show a - 1 + 1 = a by omega] at h
    have h3 : m - (a - 1) = m + 1 - a := by omega
    rw [h3] at h
    linarith [hid1]
  have hid1R : (m : ℝ) * (((m - 1).choose (a - 1) : ℕ) : ℝ) = (a : ℝ) * ((m.choose a : ℕ) : ℝ) := by
    exact_mod_cast hid1
  have hid2R : (m : ℝ) * (((m - 1).choose (a - 1) : ℕ) : ℝ) =
      (((m + 1 - a : ℕ) : ℝ)) * ((m.choose (a - 1) : ℕ) : ℝ) := by
    exact_mod_cast hid2
  have hsum_coef : (a : ℝ) + ((m + 1 - a : ℕ) : ℝ) = ((m + 1 : ℕ) : ℝ) := by
    exact_mod_cast (show a + (m + 1 - a) = m + 1 by omega)
  have hne1 : ((m + 1 : ℕ) : ℝ) ≠ 0 := ne_of_gt hm1_pos
  have hne2 : ((m.choose a : ℕ) : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (ne_of_gt hcm)
  have hne3 : ((m.choose (a - 1) : ℕ) : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (ne_of_gt hcm1)
  have hne4 : (m : ℝ) ≠ 0 := ne_of_gt hm_pos
  have hne5 : (((m - 1).choose (a - 1) : ℕ) : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (ne_of_gt hcm2)
  field_simp [hne1, hne2, hne3, hne4, hne5]
  calc (((m.choose (a - 1) : ℕ) : ℝ) + ((m.choose a : ℕ) : ℝ)) * (m : ℝ) * (((m - 1).choose (a - 1) : ℕ) : ℝ)
      = ((m.choose (a - 1) : ℕ) : ℝ) * ((m : ℝ) * (((m - 1).choose (a - 1) : ℕ) : ℝ)) +
        ((m.choose a : ℕ) : ℝ) * ((m : ℝ) * (((m - 1).choose (a - 1) : ℕ) : ℝ)) := by ring
    _ = ((m.choose (a - 1) : ℕ) : ℝ) * ((a : ℝ) * ((m.choose a : ℕ) : ℝ)) +
        ((m.choose a : ℕ) : ℝ) * (((m + 1 - a : ℕ) : ℝ) * ((m.choose (a - 1) : ℕ) : ℝ)) := by
        congr 1
        · rw [hid1R]
        · rw [hid2R]
    _ = ((a : ℝ) + ((m + 1 - a : ℕ) : ℝ)) * ((m.choose a : ℕ) : ℝ) * ((m.choose (a - 1) : ℕ) : ℝ) := by ring
    _ = ((m + 1 : ℕ) : ℝ) * ((m.choose a : ℕ) : ℝ) * ((m.choose (a - 1) : ℕ) : ℝ) := by rw [hsum_coef]

/-- Inductive binomial collapse replacing the Beta integral:
`∑_{j=0}^r C(r, j) * m_k(t - j) = m_{k + r}(t - r)` for `r ≤ t` and `t ≤ n - k`. -/
theorem mWeight_binomial_sum {n k r t : ℕ} (hrt : r ≤ t) (ht : t ≤ n - k) :
    ∑ j ∈ Finset.range (r + 1), (r.choose j : ℝ) * mWeight n k (t - j) =
      mWeight n (k + r) (t - r) := by
  induction r generalizing k t with
  | zero => simp
  | succ r ih =>
    have hsplit : ∑ j ∈ Finset.range (r + 2), ((r + 1).choose j : ℝ) * mWeight n k (t - j) =
        (∑ j ∈ Finset.range (r + 1), (r.choose j : ℝ) * mWeight n k (t - j)) +
        (∑ j ∈ Finset.range (r + 1), (r.choose j : ℝ) * mWeight n k (t - (j + 1))) := by
      rw [Finset.sum_range_succ']
      simp only [Nat.choose_zero_right, Nat.cast_one, one_mul, Nat.sub_zero]
      have hpascal : ∑ j ∈ Finset.range (r + 1), ((r + 1).choose (j + 1) : ℝ) * mWeight n k (t - (j + 1)) =
          (∑ j ∈ Finset.range (r + 1), (r.choose (j + 1) : ℝ) * mWeight n k (t - (j + 1))) +
          (∑ j ∈ Finset.range (r + 1), (r.choose j : ℝ) * mWeight n k (t - (j + 1))) := by
        rw [← Finset.sum_add_distrib]
        refine Finset.sum_congr rfl fun j _ => ?_
        rw [Nat.choose_succ_succ, Nat.cast_add]; ring
      rw [hpascal]
      have hshift : (∑ j ∈ Finset.range (r + 1), (r.choose (j + 1) : ℝ) * mWeight n k (t - (j + 1))) +
          mWeight n k t =
          ∑ j ∈ Finset.range (r + 1), (r.choose j : ℝ) * mWeight n k (t - j) := by
        have h1 := Finset.sum_range_succ' (fun j => (r.choose j : ℝ) * mWeight n k (t - j)) r
        have h2 := Finset.sum_range_succ (fun j => (r.choose (j + 1) : ℝ) * mWeight n k (t - (j + 1))) r
        simp only [Nat.choose_zero_right, Nat.cast_one, one_mul, Nat.sub_zero,
          Nat.choose_succ_self, Nat.cast_zero, zero_mul, add_zero] at h1 h2
        linarith
      linarith
    rw [hsplit]
    have h1 := ih (k := k) (t := t) (by omega) ht
    have h2 : ∑ j ∈ Finset.range (r + 1), (r.choose j : ℝ) * mWeight n k (t - (j + 1)) =
        mWeight n (k + r) (t - 1 - r) := by
      have heq : ∑ j ∈ Finset.range (r + 1), (r.choose j : ℝ) * mWeight n k (t - (j + 1)) =
          ∑ j ∈ Finset.range (r + 1), (r.choose j : ℝ) * mWeight n k ((t - 1) - j) := by
        refine Finset.sum_congr rfl fun j _ => by congr 2; omega
      rw [heq]
      exact ih (k := k) (t := t - 1) (by omega) (by omega)
    rw [h1, h2]
    have hstep := mWeight_pascal (n := n) (k := k + r) (a := t - r) (by omega) (by omega)
    have heq1 : t - r - 1 = t - 1 - r := by omega
    have heq2 : k + r + 1 = k + (r + 1) := by omega
    have heq3 : t - (r + 1) = t - 1 - r := by omega
    rw [heq1, heq2] at hstep
    rw [heq3]
    exact hstep

/-- Exact Newton finite-difference expansion for any `g : ℕ → ℝ` and `j ≤ k`. -/
lemma newton_finite_diff_identity (g : ℕ → ℝ) {k j : ℕ} (hjk : j ≤ k) :
    g j = ∑ r ∈ Finset.range (k + 1),
      (∑ i ∈ Finset.range (r + 1), (-1 : ℝ) ^ (r - i) * (r.choose i : ℝ) * g i) * (j.choose r : ℝ) := by
  have h1 : ∑ r ∈ Finset.range (k + 1),
      (∑ i ∈ Finset.range (r + 1), (-1 : ℝ) ^ (r - i) * (r.choose i : ℝ) * g i) * (j.choose r : ℝ) =
      ∑ r ∈ Finset.range (k + 1), ∑ i ∈ Finset.range (k + 1),
        if i ≤ r then (-1 : ℝ) ^ (r - i) * (r.choose i : ℝ) * (j.choose r : ℝ) * g i else 0 := by
    refine Finset.sum_congr rfl fun r hr => ?_
    have hrk : r < k + 1 := Finset.mem_range.mp hr
    rw [Finset.sum_mul]
    have hfilt : (Finset.range (k + 1)).filter (fun i => i ≤ r) = Finset.range (r + 1) := by
      ext i; simp only [Finset.mem_filter, Finset.mem_range]; omega
    rw [← Finset.sum_filter, hfilt]
    refine Finset.sum_congr rfl fun i _ => by ring
  rw [h1, Finset.sum_comm]
  have hinner : ∀ i ∈ Finset.range (k + 1),
      (∑ r ∈ Finset.range (k + 1),
        if i ≤ r then (-1 : ℝ) ^ (r - i) * (r.choose i : ℝ) * (j.choose r : ℝ) * g i else 0) =
      if i = j then g j else 0 := by
    intro i hi
    by_cases hij : i ≤ j
    · have hsum_eq : (∑ r ∈ Finset.range (k + 1),
          if i ≤ r then (-1 : ℝ) ^ (r - i) * (r.choose i : ℝ) * (j.choose r : ℝ) * g i else 0) =
          (j.choose i : ℝ) * g i *
            ∑ m ∈ Finset.range (j - i + 1), (-1 : ℝ) ^ m * ((j - i).choose m : ℝ) := by
        rw [Finset.mul_sum]
        have hzero : ∀ r ∈ Finset.range (k + 1),
            (if i ≤ r then (-1 : ℝ) ^ (r - i) * (r.choose i : ℝ) * (j.choose r : ℝ) * g i else 0) =
            if i ≤ r ∧ r ≤ j then (-1 : ℝ) ^ (r - i) * (r.choose i : ℝ) * (j.choose r : ℝ) * g i else 0 := by
          intro r _
          by_cases h1 : i ≤ r
          · by_cases h2 : r ≤ j
            · simp [h1, h2]
            · simp [h1, h2, Nat.choose_eq_zero_of_lt (by omega : j < r)]
          · simp [h1]
        rw [Finset.sum_congr rfl hzero, ← Finset.sum_filter]
        refine Finset.sum_bij (fun r _ => r - i) ?_ ?_ ?_ ?_
        · intro r hr; dsimp only; rw [Finset.mem_range]; have := (Finset.mem_filter.mp hr).2; omega
        · intro r₁ h1 r₂ h2 heq
          dsimp only at heq
          have hr1 := (Finset.mem_filter.mp h1).2; have hr2 := (Finset.mem_filter.mp h2).2; omega
        · intro m hm
          rw [Finset.mem_range] at hm
          refine ⟨i + m, Finset.mem_filter.mpr ⟨Finset.mem_range.mpr (by omega), by omega⟩, ?_⟩
          dsimp only; omega
        · intro r hr
          dsimp only
          have hri := (Finset.mem_filter.mp hr).2
          have hch : j.choose r * r.choose i = j.choose i * (j - i).choose (r - i) := by
            have := Nat.choose_mul hri.2 hri.1
            linarith
          have hchR : (j.choose r : ℝ) * (r.choose i : ℝ) =
              (j.choose i : ℝ) * ((j - i).choose (r - i) : ℝ) := by exact_mod_cast hch
          calc (-1 : ℝ) ^ (r - i) * (r.choose i : ℝ) * (j.choose r : ℝ) * g i
              = (-1 : ℝ) ^ (r - i) * ((j.choose r : ℝ) * (r.choose i : ℝ)) * g i := by ring
            _ = (-1 : ℝ) ^ (r - i) * ((j.choose i : ℝ) * ((j - i).choose (r - i) : ℝ)) * g i := by rw [hchR]
            _ = (j.choose i : ℝ) * g i * ((-1 : ℝ) ^ (r - i) * ((j - i).choose (r - i) : ℝ)) := by ring
      rw [hsum_eq]
      have halt : ∀ d : ℕ, ∑ m ∈ Finset.range (d + 1), (-1 : ℝ) ^ m * (d.choose m : ℝ) =
          if d = 0 then 1 else 0 := by
        intro d
        cases d with
        | zero => simp
        | succ d =>
          have h := add_pow (-1 : ℝ) (1 : ℝ) (d + 1)
          simp only [neg_add_cancel, zero_pow (Nat.succ_ne_zero d), one_pow, mul_one] at h
          rw [if_neg (Nat.succ_ne_zero d), h]
      rw [halt]
      by_cases heq : i = j
      · subst heq; simp
      · have hne : j - i ≠ 0 := by omega
        rw [if_neg hne, if_neg heq, mul_zero]
    · rw [if_neg (by omega : i ≠ j)]
      refine Finset.sum_eq_zero fun r _ => ?_
      by_cases hri : i ≤ r
      · rw [if_pos hri, Nat.choose_eq_zero_of_lt (by omega : j < r)]; simp
      · rw [if_neg hri]
  rw [Finset.sum_congr rfl hinner, Finset.sum_ite_eq', if_pos (Finset.mem_range.mpr (by omega))]

/-- **Theorem 1, Part 1 (Orthogonality `q_T ∈ H = L^⊥`)**: For every `k ≥ 1`, `n ≥ 2k`,
interior coalition `T` (`k ≤ |T| ≤ n - k`), and `R` with `|R| = k - 1`, `⟨v_R, q_T⟩ = 0`. -/
theorem qVec_orthog_vVec {n k : ℕ} (hk : 1 ≤ k) (_hn : 2 * k ≤ n)
    {T : Finset (Fin n)} (hT1 : k ≤ T.card) (hT2 : T.card ≤ n - k)
    {R : Finset (Fin n)} (hR : R.card = k - 1) :
    dotProduct (vVec (k := k) R) (qVec n k T) = 0 := by
  unfold dotProduct vVec
  rw [show (∑ S : KSet n k, (if R ⊆ S.1 then (1 : ℝ) else 0) * qVec n k T S) =
      ∑ S ∈ Finset.univ.filter (fun S : KSet n k => R ⊆ S.1), qVec n k T S by
    rw [Finset.sum_filter]
    refine Finset.sum_congr rfl fun S _ => by by_cases h : R ⊆ S.1 <;> simp [h]]
  set b := (T ∩ R).card
  set a := T.card - b
  have hb_le_R : b ≤ k - 1 := by
    have := Finset.card_le_card (Finset.inter_subset_right (s₁ := T) (s₂ := R)); omega
  have hb_le_T : b ≤ T.card := Finset.card_le_card Finset.inter_subset_left
  have ha1 : 1 ≤ a := by omega
  have ha2 : a ≤ n - k := by omega
  let F := Finset.univ.filter (fun S : KSet n k => R ⊆ S.1)
  rw [← Finset.sum_filter_add_sum_filter_not F (fun S => (T ∩ S.1).card = b + 1)]
  have hpart1 : ∑ S ∈ F.filter (fun S => (T ∩ S.1).card = b + 1), qVec n k T S =
      (a : ℝ) * ((-1 : ℝ) ^ (k - (b + 1)) * mWeight n k (a - 1)) := by
    have hconst : ∀ S ∈ F.filter (fun S => (T ∩ S.1).card = b + 1),
        qVec n k T S = (-1 : ℝ) ^ (k - (b + 1)) * mWeight n k (a - 1) := by
      intro S hS
      have hcard := (Finset.mem_filter.mp hS).2
      unfold qVec; rw [hcard]
      have : T.card - (b + 1) = a - 1 := by omega
      rw [this]
    rw [Finset.sum_congr rfl hconst, Finset.sum_const, nsmul_eq_mul]
    congr 1
    have hbij : (F.filter (fun S => (T ∩ S.1).card = b + 1)).card = (T \ R).card := by
      symm
      refine Finset.card_bij
        (fun i hi => (⟨insert i R, by
          rw [Finset.card_insert_of_notMem (Finset.mem_sdiff.mp hi).2, hR]; omega⟩ : KSet n k))
        ?_ ?_ ?_
      · intro i hi
        have hiT : i ∈ T := (Finset.mem_sdiff.mp hi).1
        have hiR : i ∉ R := (Finset.mem_sdiff.mp hi).2
        rw [Finset.mem_filter, Finset.mem_filter]
        refine ⟨⟨Finset.mem_univ _, Finset.subset_insert i R⟩, ?_⟩
        have hint : T ∩ insert i R = insert i (T ∩ R) := by
          ext x; simp only [Finset.mem_inter, Finset.mem_insert]
          constructor
          · rintro ⟨hxT, rfl | hxR⟩; exact Or.inl rfl; exact Or.inr ⟨hxT, hxR⟩
          · rintro (rfl | ⟨hxT, hxR⟩); exact ⟨hiT, Or.inl rfl⟩; exact ⟨hxT, Or.inr hxR⟩
        rw [hint, Finset.card_insert_of_notMem (by simp [hiR])]
      · intro i₁ h1 i₂ h2 heq
        have h1R : i₁ ∉ R := (Finset.mem_sdiff.mp h1).2
        have heq1 : insert i₁ R = insert i₂ R := congrArg Subtype.val heq
        have hmem : i₁ ∈ insert i₂ R := by rw [← heq1]; exact Finset.mem_insert_self i₁ R
        exact (Finset.mem_insert.mp hmem).resolve_right h1R
      · intro S hS
        have hSF := (Finset.mem_filter.mp hS).1
        have hRS : R ⊆ S.1 := (Finset.mem_filter.mp hSF).2
        have hTS : (T ∩ S.1).card = b + 1 := (Finset.mem_filter.mp hS).2
        have hdiff : (S.1 \ R).card = 1 := by rw [Finset.card_sdiff_of_subset hRS, S.2, hR]; omega
        obtain ⟨i, hi_eq⟩ := Finset.card_eq_one.mp hdiff
        have hi_mem : i ∈ S.1 \ R := by rw [hi_eq]; exact Finset.mem_singleton_self i
        have hiR : i ∉ R := (Finset.mem_sdiff.mp hi_mem).2
        have hSeq : S.1 = insert i R := by
          calc S.1 = (S.1 \ R) ∪ R := (Finset.sdiff_union_of_subset hRS).symm
            _ = {i} ∪ R := by rw [hi_eq]
            _ = insert i R := Finset.singleton_union i R
        have hiT : i ∈ T := by
          by_contra hniT
          have hint : T ∩ S.1 = T ∩ R := by
            rw [hSeq]; ext x; simp only [Finset.mem_inter, Finset.mem_insert]
            constructor
            · rintro ⟨hxT, rfl | hxR⟩; exact (hniT hxT).elim; exact ⟨hxT, hxR⟩
            · rintro ⟨hxT, hxR⟩; exact ⟨hxT, Or.inr hxR⟩
          rw [hint] at hTS; omega
        exact ⟨i, Finset.mem_sdiff.mpr ⟨hiT, hiR⟩, Subtype.ext hSeq.symm⟩
    rw [hbij, Finset.card_sdiff, Finset.inter_comm]
  have hpart2 : ∑ S ∈ F.filter (fun S => ¬ (T ∩ S.1).card = b + 1), qVec n k T S =
      ((n - k + 1 - a : ℕ) : ℝ) * ((-1 : ℝ) ^ (k - b) * mWeight n k a) := by
    have hconst : ∀ S ∈ F.filter (fun S => ¬ (T ∩ S.1).card = b + 1),
        qVec n k T S = (-1 : ℝ) ^ (k - b) * mWeight n k a := by
      intro S hS
      have hRS : R ⊆ S.1 := (Finset.mem_filter.mp (Finset.mem_filter.mp hS).1).2
      have hne : (T ∩ S.1).card ≠ b + 1 := (Finset.mem_filter.mp hS).2
      have hdiff : (S.1 \ R).card = 1 := by rw [Finset.card_sdiff_of_subset hRS, S.2, hR]; omega
      obtain ⟨i, hi_eq⟩ := Finset.card_eq_one.mp hdiff
      have hi_mem : i ∈ S.1 \ R := by rw [hi_eq]; exact Finset.mem_singleton_self i
      have hiR : i ∉ R := (Finset.mem_sdiff.mp hi_mem).2
      have hSeq : S.1 = insert i R := by
        calc S.1 = (S.1 \ R) ∪ R := (Finset.sdiff_union_of_subset hRS).symm
          _ = {i} ∪ R := by rw [hi_eq]
          _ = insert i R := Finset.singleton_union i R
      have hniT : i ∉ T := by
        intro hiT
        apply hne
        have hint : T ∩ S.1 = insert i (T ∩ R) := by
          rw [hSeq]; ext x; simp only [Finset.mem_inter, Finset.mem_insert]
          constructor
          · rintro ⟨hxT, rfl | hxR⟩; exact Or.inl rfl; exact Or.inr ⟨hxT, hxR⟩
          · rintro (rfl | ⟨hxT, hxR⟩); exact ⟨hiT, Or.inl rfl⟩; exact ⟨hxT, Or.inr hxR⟩
        rw [hint, Finset.card_insert_of_notMem (by simp [hiR])]
      have hint : T ∩ S.1 = T ∩ R := by
        rw [hSeq]; ext x; simp only [Finset.mem_inter, Finset.mem_insert]
        constructor
        · rintro ⟨hxT, rfl | hxR⟩; exact (hniT hxT).elim; exact ⟨hxT, hxR⟩
        · rintro ⟨hxT, hxR⟩; exact ⟨hxT, Or.inr hxR⟩
      unfold qVec; rw [hint]
    rw [Finset.sum_congr rfl hconst, Finset.sum_const, nsmul_eq_mul]
    congr 1
    have hbij : (F.filter (fun S => ¬ (T ∩ S.1).card = b + 1)).card = (Finset.univ \ (T ∪ R)).card := by
      symm
      refine Finset.card_bij
        (fun i hi => (⟨insert i R, by
          have hiTR := (Finset.mem_sdiff.mp hi).2
          have hiR : i ∉ R := fun h => hiTR (Finset.mem_union_right T h)
          rw [Finset.card_insert_of_notMem hiR, hR]; omega⟩ : KSet n k))
        ?_ ?_ ?_
      · intro i hi
        have hiTR := (Finset.mem_sdiff.mp hi).2
        have hiT : i ∉ T := fun h => hiTR (Finset.mem_union_left R h)
        have hiR : i ∉ R := fun h => hiTR (Finset.mem_union_right T h)
        rw [Finset.mem_filter, Finset.mem_filter]
        refine ⟨⟨Finset.mem_univ _, Finset.subset_insert i R⟩, ?_⟩
        have hint : T ∩ insert i R = T ∩ R := by
          ext x; simp only [Finset.mem_inter, Finset.mem_insert]
          constructor
          · rintro ⟨hxT, rfl | hxR⟩; exact (hiT hxT).elim; exact ⟨hxT, hxR⟩
          · rintro ⟨hxT, hxR⟩; exact ⟨hxT, Or.inr hxR⟩
        rw [hint]; omega
      · intro i₁ h1 i₂ h2 heq
        have h1R : i₁ ∉ R := fun h => (Finset.mem_sdiff.mp h1).2 (Finset.mem_union_right T h)
        have heq1 : insert i₁ R = insert i₂ R := congrArg Subtype.val heq
        have hmem : i₁ ∈ insert i₂ R := by rw [← heq1]; exact Finset.mem_insert_self i₁ R
        exact (Finset.mem_insert.mp hmem).resolve_right h1R
      · intro S hS
        have hRS : R ⊆ S.1 := (Finset.mem_filter.mp (Finset.mem_filter.mp hS).1).2
        have hne : (T ∩ S.1).card ≠ b + 1 := (Finset.mem_filter.mp hS).2
        have hdiff : (S.1 \ R).card = 1 := by rw [Finset.card_sdiff_of_subset hRS, S.2, hR]; omega
        obtain ⟨i, hi_eq⟩ := Finset.card_eq_one.mp hdiff
        have hi_mem : i ∈ S.1 \ R := by rw [hi_eq]; exact Finset.mem_singleton_self i
        have hiR : i ∉ R := (Finset.mem_sdiff.mp hi_mem).2
        have hSeq : S.1 = insert i R := by
          calc S.1 = (S.1 \ R) ∪ R := (Finset.sdiff_union_of_subset hRS).symm
            _ = {i} ∪ R := by rw [hi_eq]
            _ = insert i R := Finset.singleton_union i R
        have hiT : i ∉ T := by
          intro h
          apply hne
          have hint : T ∩ S.1 = insert i (T ∩ R) := by
            rw [hSeq]; ext x; simp only [Finset.mem_inter, Finset.mem_insert]
            constructor
            · rintro ⟨hxT, rfl | hxR⟩; exact Or.inl rfl; exact Or.inr ⟨hxT, hxR⟩
            · rintro (rfl | ⟨hxT, hxR⟩); exact ⟨h, Or.inl rfl⟩; exact ⟨hxT, Or.inr hxR⟩
          rw [hint, Finset.card_insert_of_notMem (by simp [hiR])]
        refine ⟨i, Finset.mem_sdiff.mpr ⟨Finset.mem_univ _, ?_⟩, Subtype.ext hSeq.symm⟩
        simp [hiT, hiR]
    rw [hbij, Finset.card_sdiff_of_subset (Finset.subset_univ _), Finset.card_univ, Fintype.card_fin]
    have hunion : (T ∪ R).card + b = T.card + R.card := Finset.card_union_add_card_inter T R
    have : n - (T ∪ R).card = n - k + 1 - a := by omega
    rw [this]
  rw [hpart1, hpart2]
  -- Exact cancellation: a * mWeight n k (a - 1) = (n - k + 1 - a) * mWeight n k a
  have hpow : (-1 : ℝ) ^ (k - b) = -((-1 : ℝ) ^ (k - (b + 1))) := by
    rw [show k - b = (k - (b + 1)) + 1 by omega, pow_succ]; ring
  rw [hpow]
  have hcancel : (a : ℝ) * mWeight n k (a - 1) = ((n - k + 1 - a : ℕ) : ℝ) * mWeight n k a := by
    unfold mWeight
    have hch := Nat.choose_succ_right_eq (n - k) (a - 1)
    rw [show a - 1 + 1 = a by omega, show n - k - (a - 1) = n - k + 1 - a by omega] at hch
    have hchR : (((n - k).choose a : ℕ) : ℝ) * (a : ℝ) =
        (((n - k).choose (a - 1) : ℕ) : ℝ) * ((n - k + 1 - a : ℕ) : ℝ) := by exact_mod_cast hch
    have hne0 : ((n - k + 1 : ℕ) : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
    have hne1 : (((n - k).choose a : ℕ) : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.choose_pos ha2).ne'
    have hne2 : (((n - k).choose (a - 1) : ℕ) : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.choose_pos (by omega)).ne'
    field_simp [hne0, hne1, hne2]
    linarith
  calc (a : ℝ) * ((-1 : ℝ) ^ (k - (b + 1)) * mWeight n k (a - 1)) +
        ((n - k + 1 - a : ℕ) : ℝ) * (-((-1 : ℝ) ^ (k - (b + 1))) * mWeight n k a)
      = (-1 : ℝ) ^ (k - (b + 1)) * ((a : ℝ) * mWeight n k (a - 1) -
          ((n - k + 1 - a : ℕ) : ℝ) * mWeight n k a) := by ring
    _ = 0 := by rw [hcancel, sub_self, mul_zero]

lemma qVec_inHarmonic {n k : ℕ} (hk : 1 ≤ k) (hn : 2 * k ≤ n)
    {T : Finset (Fin n)} (hT1 : k ≤ T.card) (hT2 : T.card ≤ n - k) :
    InHarmonic n k (qVec n k T) := by
  intro v hv
  refine Submodule.span_induction (p := fun v _ => dotProduct v (qVec n k T) = 0) ?_ ?_ ?_ ?_ hv
  · rintro _ ⟨R, hR, rfl⟩
    exact qVec_orthog_vVec hk hn hT1 hT2 hR
  · simp
  · intro x y _ _ hx hy; rw [add_dotProduct, hx, hy, add_zero]
  · intro c x _ hx; rw [smul_dotProduct, hx, smul_zero]

/-- **Theorem 1, Part 2**: `q_T - (w_t / c) · f_T ∈ L` for every interior coalition `T`. -/
theorem qVec_sub_scaled_fVec_mem_lower {n k : ℕ} (_hk : 1 ≤ k) (_hn : 2 * k ≤ n)
    {T : Finset (Fin n)} (hT1 : k ≤ T.card) (hT2 : T.card ≤ n - k) :
    qVec n k T - (wWeight n k T.card / ((n - 2 * k + 1 : ℕ) : ℝ)) • fVec (k := k) T ∈
      lowerSubspace n k := by
  let g : ℕ → ℝ := fun j => (-1 : ℝ) ^ (k - j) * mWeight n k (T.card - j)
  let coef : ℕ → ℝ := fun r => ∑ i ∈ Finset.range (r + 1), (-1 : ℝ) ^ (r - i) * (r.choose i : ℝ) * g i
  have hcoef_k : coef k = wWeight n k T.card / ((n - 2 * k + 1 : ℕ) : ℝ) := by
    dsimp [coef, g]
    have hsign : ∀ i ∈ Finset.range (k + 1),
        (-1 : ℝ) ^ (k - i) * (k.choose i : ℝ) * ((-1 : ℝ) ^ (k - i) * mWeight n k (T.card - i)) =
        (k.choose i : ℝ) * mWeight n k (T.card - i) := by
      intro i _
      calc (-1 : ℝ) ^ (k - i) * (k.choose i : ℝ) * ((-1 : ℝ) ^ (k - i) * mWeight n k (T.card - i))
          = ((-1 : ℝ) * (-1 : ℝ)) ^ (k - i) * ((k.choose i : ℝ) * mWeight n k (T.card - i)) := by
            rw [mul_pow]; ring
        _ = (k.choose i : ℝ) * mWeight n k (T.card - i) := by norm_num
    rw [Finset.sum_congr rfl hsign, mWeight_binomial_sum hT1 hT2]
    unfold mWeight wWeight
    rw [show k + k = 2 * k by omega]
    ring
  have hrew : qVec n k T - (wWeight n k T.card / ((n - 2 * k + 1 : ℕ) : ℝ)) • fVec (k := k) T =
      ∑ r ∈ Finset.range k, coef r • ∑ R ∈ T.powersetCard r, vVec (k := k) R := by
    funext S
    rw [Pi.sub_apply, Pi.smul_apply, smul_eq_mul, Finset.sum_apply]
    have hj : (T ∩ S.1).card ≤ k :=
      calc (T ∩ S.1).card ≤ S.1.card := Finset.card_le_card Finset.inter_subset_right
        _ = k := S.2
    have hnewt := newton_finite_diff_identity g hj
    change g (T ∩ S.1).card - _ = _
    rw [hnewt, Finset.sum_range_succ]
    have htop : (wWeight n k T.card / ((n - 2 * k + 1 : ℕ) : ℝ)) * (((T ∩ S.1).card.choose k : ℕ) : ℝ) =
        (wWeight n k T.card / ((n - 2 * k + 1 : ℕ) : ℝ)) * fVec (k := k) T S := by
      congr 1
      unfold fVec
      by_cases hST : S.1 ⊆ T
      · rw [if_pos hST, Finset.inter_eq_right.mpr hST, S.2, Nat.choose_self]; simp
      · rw [if_neg hST]
        have hlt : (T ∩ S.1).card < k := by
          calc (T ∩ S.1).card < S.1.card := by
                apply Finset.card_lt_card
                rw [Finset.ssubset_iff_subset_ne]
                refine ⟨Finset.inter_subset_right, fun heq => hST ?_⟩
                rw [← heq]; exact Finset.inter_subset_left
            _ = k := S.2
        rw [Nat.choose_eq_zero_of_lt hlt]; simp
    have htop' : (∑ i ∈ Finset.range (k + 1), (-1 : ℝ) ^ (k - i) * (k.choose i : ℝ) * g i) *
        (((T ∩ S.1).card.choose k : ℕ) : ℝ) =
        (wWeight n k T.card / ((n - 2 * k + 1 : ℕ) : ℝ)) * fVec (k := k) T S := by
      change coef k * _ = _
      rw [hcoef_k]; exact htop
    rw [htop', add_sub_cancel_right]
    refine Finset.sum_congr rfl fun r _ => ?_
    rw [Pi.smul_apply, Finset.sum_apply, smul_eq_mul, choose_inter_eq_sum_vVec]
  rw [hrew]
  refine Submodule.sum_mem _ fun r hr => Submodule.smul_mem _ _ ?_
  refine Submodule.sum_mem _ fun R hR => ?_
  exact vVec_mem_lowerSubspace_of_lt (by rw [(Finset.mem_powersetCard.mp hR).2]; exact Finset.mem_range.mp hr)

/-- **Theorem 1, Part 3 (Exact orthogonal projector `P_k` and Conjecture 3.8)**:
For every `k ≥ 1` and `n ≥ 2k`, `P_k = (n - 2k + 1) • A_k` is the exact orthogonal projector
onto `H = L^⊥`:
1. `P_kᵀ = P_k`,
2. `P_k *ᵥ v = 0` for all `v ∈ L`,
3. `P_k *ᵥ u = u` for all `u ∈ H = L^⊥`,
4. `P_k * P_k = P_k`,
5. `tr(P_k) = C(n, k) - C(n, k - 1)`. -/
theorem matP_is_orthog_projector {n k : ℕ} (hk : 1 ≤ k) (hn : 2 * k ≤ n) :
    (matP n k).IsSymm ∧
    (∀ v ∈ lowerSubspace n k, (matP n k).mulVec v = 0) ∧
    (∀ u, InHarmonic n k u → (matP n k).mulVec u = u) ∧
    matP n k * matP n k = matP n k ∧
    Matrix.trace (matP n k) = ((n.choose k : ℕ) : ℝ) - ((n.choose (k - 1) : ℕ) : ℝ) := by
  have hc_ne : ((n - 2 * k + 1 : ℕ) : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  have hrow : ∀ U : KSet n k, (fun S => matP n k U S) = ((n - 2 * k + 1 : ℕ) : ℝ) • qVec n k U.1 := by
    intro U; funext S
    simp only [matP, matA, qVec, Matrix.smul_apply, smul_eq_mul, Pi.smul_apply, U.2]
  have hkill : ∀ v ∈ lowerSubspace n k, (matP n k).mulVec v = 0 := by
    intro v hv
    funext U
    change dotProduct (fun S => matP n k U S) v = 0
    rw [hrow U, smul_dotProduct, dotProduct_comm,
      qVec_inHarmonic hk hn (le_of_eq U.2.symm) (by rw [U.2]; omega) v hv, smul_zero]
  have hfix : ∀ u, InHarmonic n k u → (matP n k).mulVec u = u := by
    intro u hu
    funext U
    change dotProduct (fun S => matP n k U S) u = u U
    rw [hrow U, smul_dotProduct]
    have hq_sub := qVec_sub_scaled_fVec_mem_lower hk hn (T := U.1) (le_of_eq U.2.symm) (by rw [U.2]; omega)
    have hwk : wWeight n k U.1.card = 1 := by
      unfold wWeight; rw [U.2, Nat.sub_self, Nat.choose_zero_right]; simp
    rw [hwk] at hq_sub
    have horth := hu _ hq_sub
    rw [sub_dotProduct, smul_dotProduct] at horth
    have hdot_f : dotProduct (fVec (k := k) U.1) u = u U := by
      unfold dotProduct fVec
      rw [Finset.sum_eq_single U]
      · simp
      · intro S _ hne
        have hnot : ¬ S.1 ⊆ U.1 := fun hsub => hne (Subtype.ext (Finset.eq_of_subset_of_card_le hsub (by rw [S.2, U.2])))
        rw [if_neg hnot, zero_mul]
      · intro h; exact (h (Finset.mem_univ U)).elim
    rw [hdot_f] at horth
    have hdot_q : dotProduct (qVec n k U.1) u = (1 / ((n - 2 * k + 1 : ℕ) : ℝ)) * u U := by
      simp only [smul_eq_mul] at horth; linarith
    rw [hdot_q, smul_eq_mul]
    field_simp [hc_ne]
  have hidem : matP n k * matP n k = matP n k := by
    ext S U
    have hcol : InHarmonic n k (fun W => matP n k W U) := by
      have hsym : (fun W => matP n k W U) = ((n - 2 * k + 1 : ℕ) : ℝ) • qVec n k U.1 := by
        funext W
        have hWU : matP n k W U = matP n k U W := by
          have := congrFun (congrFun (matP_isSymm n k) U) W
          simpa [Matrix.transpose_apply] using this
        exact hWU.trans (congrFun (hrow U) W)
      rw [hsym]
      intro v hv
      rw [dotProduct_smul, qVec_inHarmonic hk hn (le_of_eq U.2.symm) (by rw [U.2]; omega) v hv, smul_zero]
    have := congrFun (hfix (fun W => matP n k W U) hcol) S
    simpa [Matrix.mul_apply, Matrix.mulVec, dotProduct] using this
  have htrace : Matrix.trace (matP n k) = ((n.choose k : ℕ) : ℝ) - ((n.choose (k - 1) : ℕ) : ℝ) := by
    unfold Matrix.trace Matrix.diag
    have hdiag : ∀ S : KSet n k, matP n k S S = ((n - 2 * k + 1 : ℕ) : ℝ) / ((n - k + 1 : ℕ) : ℝ) := by
      intro S
      unfold matP matA mWeight
      simp only [Matrix.smul_apply, smul_eq_mul, Finset.inter_self, S.2, Nat.sub_self,
        pow_zero, one_mul, Nat.choose_zero_right, Nat.cast_one, mul_one]
      ring
    rw [Finset.sum_congr rfl (fun S _ => hdiag S), Finset.sum_const, Finset.card_univ]
    have hcard : Fintype.card (KSet n k) = n.choose k := by
      rw [Fintype.card_subtype]
      have hfilt : (Finset.univ.filter (fun S : Finset (Fin n) => S.card = k)) =
          (Finset.univ : Finset (Fin n)).powersetCard k := by
        ext S; simp [Finset.mem_powersetCard]
      rw [hfilt, Finset.card_powersetCard, Finset.card_univ, Fintype.card_fin]
    rw [hcard, nsmul_eq_mul]
    -- C(n, k) * (n - 2k + 1) = (n - k + 1) * (C(n, k) - C(n, k - 1))
    -- i.e. k * C(n, k) = (n - k + 1) * C(n, k - 1)
    have hch := Nat.choose_succ_right_eq n (k - 1)
    rw [show k - 1 + 1 = k by omega, show n - (k - 1) = n - k + 1 by omega] at hch
    have hchR : ((n.choose k : ℕ) : ℝ) * (k : ℝ) =
        ((n.choose (k - 1) : ℕ) : ℝ) * ((n - k + 1 : ℕ) : ℝ) := by exact_mod_cast hch
    have hsub : ((n - 2 * k + 1 : ℕ) : ℝ) = ((n - k + 1 : ℕ) : ℝ) - (k : ℝ) := by
      have : (n - 2 * k + 1) + k = n - k + 1 := by omega
      have hR : ((n - 2 * k + 1 : ℕ) : ℝ) + (k : ℝ) = ((n - k + 1 : ℕ) : ℝ) := by exact_mod_cast this
      linarith
    have hne : ((n - k + 1 : ℕ) : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
    rw [hsub]
    field_simp [hne]
    linarith
  exact ⟨matP_isSymm n k, hkill, hfix, hidem, htrace⟩

/-- **Theorem 1, Part 4 (Interior column projection identity, eq. 3 of `solution.md`)**:
For every `k ≥ 1`, `n ≥ 2k`, `B 0 = 1`, and interior coalition `T` (`k ≤ |T| ≤ n - k`),
`q_T = (w_t / c) · P_k f_T = w_t · A_k x_T`. -/
theorem interior_column_projection {n k : ℕ} (hk : 1 ≤ k) (hn : 2 * k ≤ n)
    {B : ℕ → ℝ} (hB0 : B 0 = 1)
    {T : Finset (Fin n)} (hT1 : k ≤ T.card) (hT2 : T.card ≤ n - k) :
    qVec n k T = (wWeight n k T.card / ((n - 2 * k + 1 : ℕ) : ℝ)) • (matP n k).mulVec (fVec T) ∧
    qVec n k T = wWeight n k T.card • (matA n k).mulVec (xVec B T) := by
  have hproj := matP_is_orthog_projector hk hn
  have hqH := qVec_inHarmonic hk hn hT1 hT2
  have hqL := qVec_sub_scaled_fVec_mem_lower hk hn hT1 hT2
  have h1 : (matP n k).mulVec (qVec n k T - (wWeight n k T.card / ((n - 2 * k + 1 : ℕ) : ℝ)) • fVec T) = 0 :=
    hproj.2.1 _ hqL
  rw [Matrix.mulVec_sub, Matrix.mulVec_smul, hproj.2.2.1 _ hqH, sub_eq_zero] at h1
  refine ⟨h1, ?_⟩
  have hxL := xVec_sub_fVec_mem_lowerSubspace hk hB0 T
  have h2 : (matP n k).mulVec (xVec B T - fVec T) = 0 := hproj.2.1 _ hxL
  rw [Matrix.mulVec_sub, sub_eq_zero] at h2
  have hA : matA n k = (((n - 2 * k + 1 : ℕ) : ℝ)⁻¹) • matP n k := (matA_eq_div_matP hn).1
  rw [h1, hA, Matrix.smul_mulVec, h2, smul_smul, div_eq_mul_inv]

/-! ### 3. Lemma 2: Interior Gram Eigenvalue `n - 2k + 1` on `H = L^⊥` -/

/-- **Lemma 2 (Interior Gram action on `H`)**: For every `k ≥ 1`, `n ≥ 2k`, `B 0 = 1`,
and every `u ∈ H = L^⊥`, the interior weighted Gram matrix `G_I = ∑_{T ∈ I} w_{|T|} x_T x_T^⊤`
acts on `u` as `(n - 2k + 1) • u`. -/
theorem interior_gram_acts_on_H {n k : ℕ} (hk : 1 ≤ k) (hn : 2 * k ≤ n)
    {B : ℕ → ℝ} (hB0 : B 0 = 1)
    {u : KSet n k → ℝ} (hu : InHarmonic n k u) :
    (∑ T ∈ (Finset.univ : Finset (Finset (Fin n))).filter (fun T => k ≤ T.card ∧ T.card ≤ n - k),
      (wWeight n k T.card * dotProduct (xVec (k := k) B T) u) • xVec (k := k) B T) =
      ((n - 2 * k + 1 : ℕ) : ℝ) • u := by
  have hxf_dot : ∀ T : Finset (Fin n), dotProduct (xVec (k := k) B T) u = dotProduct (fVec (k := k) T) u := by
    intro T
    have h0 := hu _ (xVec_sub_fVec_mem_lowerSubspace hk hB0 T)
    rw [sub_dotProduct, sub_eq_zero] at h0
    exact h0
  have hCt_sum : ∀ (t : ℕ) (S U : KSet n k), k ≤ t → t ≤ n - k →
      ∑ T ∈ (Finset.univ : Finset (Fin n)).powersetCard t, fVec (k := k) T S * fVec (k := k) T U =
      (((n - 2 * k + (S.1 ∩ U.1).card).choose (n - t) : ℕ) : ℝ) := by
    intro t S U ht1 ht2
    unfold fVec
    have hprod : ∀ T : Finset (Fin n),
        (if S.1 ⊆ T then (1 : ℝ) else 0) * (if U.1 ⊆ T then (1 : ℝ) else 0) =
        if S.1 ∪ U.1 ⊆ T then 1 else 0 := by
      intro T
      by_cases h1 : S.1 ⊆ T <;> by_cases h2 : U.1 ⊆ T <;> simp [Finset.union_subset_iff, h1, h2]
    rw [Finset.sum_congr rfl (fun T _ => hprod T), Finset.sum_boole]
    congr 1
    have hunion : (S.1 ∪ U.1).card = 2 * k - (S.1 ∩ U.1).card := by
      have := Finset.card_union_add_card_inter S.1 U.1
      rw [S.2, U.2] at this; omega
    have hcomp_card : (Finset.univ \ (S.1 ∪ U.1)).card = n - 2 * k + (S.1 ∩ U.1).card := by
      rw [Finset.card_sdiff_of_subset (Finset.subset_univ _), Finset.card_univ, Fintype.card_fin]
      have hinter_le : (S.1 ∩ U.1).card ≤ k :=
        calc (S.1 ∩ U.1).card ≤ S.1.card := Finset.card_le_card Finset.inter_subset_left
          _ = k := S.2
      omega
    rw [← hcomp_card, ← Finset.card_powersetCard]
    refine Finset.card_bij (fun T _ => Finset.univ \ T) ?_ ?_ ?_
    · intro T hT
      have hTcard : T.card = t := (Finset.mem_powersetCard.mp (Finset.mem_filter.mp hT).1).2
      have hSUT : S.1 ∪ U.1 ⊆ T := (Finset.mem_filter.mp hT).2
      rw [Finset.mem_powersetCard]
      refine ⟨fun x hx => Finset.mem_sdiff.mpr ⟨Finset.mem_univ _, fun hxSU => (Finset.mem_sdiff.mp hx).2 (hSUT hxSU)⟩, ?_⟩
      rw [Finset.card_sdiff_of_subset (Finset.subset_univ _), Finset.card_univ, Fintype.card_fin, hTcard]
    · intro T₁ _ T₂ _ heq
      dsimp only at heq
      ext x
      have hx : (x ∈ Finset.univ \ T₁) ↔ (x ∈ Finset.univ \ T₂) := by rw [heq]
      simp at hx; exact not_iff_not.mp hx
    · intro W hW
      have hWsub : W ⊆ Finset.univ \ (S.1 ∪ U.1) := (Finset.mem_powersetCard.mp hW).1
      have hWcard : W.card = n - t := (Finset.mem_powersetCard.mp hW).2
      refine ⟨Finset.univ \ W, ?_, ?_⟩
      · rw [Finset.mem_filter, Finset.mem_powersetCard]
        refine ⟨⟨Finset.subset_univ _, ?_⟩, ?_⟩
        · rw [Finset.card_sdiff_of_subset (Finset.subset_univ _), Finset.card_univ, Fintype.card_fin, hWcard]; omega
        · intro x hxSU
          exact Finset.mem_sdiff.mpr ⟨Finset.mem_univ _, fun hxW => (Finset.mem_sdiff.mp (hWsub hxW)).2 hxSU⟩
      · ext x; simp
  have hlower_vanish : ∀ (t : ℕ) (R : Finset (Fin n)), k ≤ t → t ≤ n - k → R.card < k →
      ∑ T ∈ ((Finset.univ : Finset (Fin n)).powersetCard t).filter (fun T => R ⊆ T),
        dotProduct (fVec (k := k) T) u = 0 := by
    intro t R ht1 ht2 hRlt
    have hdot_eq : ∑ T ∈ ((Finset.univ : Finset (Fin n)).powersetCard t).filter (fun T => R ⊆ T),
        dotProduct (fVec (k := k) T) u =
        dotProduct (fun U : KSet n k =>
          (((n - k - R.card + (R ∩ U.1).card).choose (n - t) : ℕ) : ℝ)) u := by
      unfold dotProduct
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun U _ => ?_
      rw [← Finset.sum_mul]
      congr 1
      unfold fVec
      rw [Finset.sum_boole]
      congr 1
      have hcomp_card : (Finset.univ \ (R ∪ U.1)).card = n - k - R.card + (R ∩ U.1).card := by
        rw [Finset.card_sdiff_of_subset (Finset.subset_univ _), Finset.card_univ, Fintype.card_fin]
        have hun := Finset.card_union_add_card_inter R U.1
        have hinter : (R ∩ U.1).card ≤ R.card := Finset.card_le_card Finset.inter_subset_left
        rw [U.2] at hun; omega
      rw [← hcomp_card, ← Finset.card_powersetCard]
      refine Finset.card_bij (fun T _ => Finset.univ \ T) ?_ ?_ ?_
      · intro T hT
        have hUT : U.1 ⊆ T := (Finset.mem_filter.mp hT).2
        have hT1 := Finset.mem_filter.mp (Finset.mem_filter.mp hT).1
        have hRT : R ⊆ T := hT1.2
        have hTcard : T.card = t := (Finset.mem_powersetCard.mp hT1.1).2
        rw [Finset.mem_powersetCard]
        refine ⟨fun x hx => Finset.mem_sdiff.mpr ⟨Finset.mem_univ _, fun hxRU =>
          (Finset.mem_sdiff.mp hx).2 (Finset.union_subset hRT hUT hxRU)⟩, ?_⟩
        rw [Finset.card_sdiff_of_subset (Finset.subset_univ _), Finset.card_univ, Fintype.card_fin, hTcard]
      · intro T₁ _ T₂ _ heq
        dsimp only at heq
        ext x
        have hx : (x ∈ Finset.univ \ T₁) ↔ (x ∈ Finset.univ \ T₂) := by rw [heq]
        simp at hx; exact not_iff_not.mp hx
      · intro W hW
        have hWsub := (Finset.mem_powersetCard.mp hW).1
        have hWcard := (Finset.mem_powersetCard.mp hW).2
        refine ⟨Finset.univ \ W, ?_, by ext x; simp⟩
        rw [Finset.mem_filter, Finset.mem_filter, Finset.mem_powersetCard]
        refine ⟨⟨⟨Finset.subset_univ _, ?_⟩, ?_⟩, ?_⟩
        · rw [Finset.card_sdiff_of_subset (Finset.subset_univ _), Finset.card_univ, Fintype.card_fin, hWcard]; omega
        · intro x hxR; exact Finset.mem_sdiff.mpr ⟨Finset.mem_univ _, fun hxW =>
            (Finset.mem_sdiff.mp (hWsub hxW)).2 (Finset.mem_union_left U.1 hxR)⟩
        · intro x hxU; exact Finset.mem_sdiff.mpr ⟨Finset.mem_univ _, fun hxW =>
            (Finset.mem_sdiff.mp (hWsub hxW)).2 (Finset.mem_union_right R hxU)⟩
    rw [hdot_eq]
    have hmemL : (fun U : KSet n k => (((n - k - R.card + (R ∩ U.1).card).choose (n - t) : ℕ) : ℝ)) ∈
        lowerSubspace n k := by
      have hvand : (fun U : KSet n k => (((n - k - R.card + (R ∩ U.1).card).choose (n - t) : ℕ) : ℝ)) =
          ∑ j ∈ Finset.range (R.card + 1),
            (((n - k - R.card).choose (n - t - j) : ℕ) : ℝ) •
              ∑ Q ∈ R.powersetCard j, vVec (k := k) Q := by
        funext U
        rw [Finset.sum_apply]
        have hv := Nat.add_choose_eq (n - k - R.card) (R ∩ U.1).card (n - t)
        rw [hv, Nat.cast_sum]
        have hsum_anti : ∑ x ∈ Finset.antidiagonal (n - t),
            (((n - k - R.card).choose x.1 * (R ∩ U.1).card.choose x.2 : ℕ) : ℝ) =
            ∑ j ∈ Finset.range (R.card + 1),
              (((n - k - R.card).choose (n - t - j) : ℕ) : ℝ) * (((R ∩ U.1).card.choose j : ℕ) : ℝ) := by
          have hanti : ∑ x ∈ Finset.antidiagonal (n - t),
              (((n - k - R.card).choose x.1 * (R ∩ U.1).card.choose x.2 : ℕ) : ℝ) =
              ∑ j ∈ Finset.range (n - t + 1),
                (((n - k - R.card).choose (n - t - j) : ℕ) : ℝ) * (((R ∩ U.1).card.choose j : ℕ) : ℝ) := by
            rw [Finset.Nat.sum_antidiagonal_eq_sum_range_succ_mk]
            have hrev := Finset.sum_range_reflect
              (fun j => (((n - k - R.card).choose (n - t - j) : ℕ) : ℝ) * (((R ∩ U.1).card.choose j : ℕ) : ℝ))
              (n - t + 1)
            rw [← hrev]
            refine Finset.sum_congr rfl fun j hj => ?_
            have hjle : j ≤ n - t := Nat.lt_succ_iff.mp (Finset.mem_range.mp hj)
            simp only [Nat.cast_mul, Nat.add_sub_cancel]
            rw [show n - t - (n - t - j) = j by omega]
          rw [hanti]
          rw [← Finset.sum_subset (Finset.range_mono (by omega : R.card + 1 ≤ n - t + 1))]
          intro j _ hnot
          have hgt : R.card < j := by rw [Finset.mem_range] at hnot; omega
          have hlt_inter : (R ∩ U.1).card < j :=
            lt_of_le_of_lt (Finset.card_le_card Finset.inter_subset_left) hgt
          rw [Nat.choose_eq_zero_of_lt hlt_inter]; simp
        rw [hsum_anti]
        refine Finset.sum_congr rfl fun j _ => ?_
        rw [Pi.smul_apply, Finset.sum_apply, smul_eq_mul, choose_inter_eq_sum_vVec]
      rw [hvand]
      refine Submodule.sum_mem _ fun j hj => Submodule.smul_mem _ _ ?_
      refine Submodule.sum_mem _ fun Q hQ => ?_
      exact vVec_mem_lowerSubspace_of_lt (by
        rw [(Finset.mem_powersetCard.mp hQ).2]
        have := Finset.mem_range.mp hj; omega)
    exact hu _ hmemL
  have hGt : ∀ t : ℕ, k ≤ t → t ≤ n - k →
      (∑ T ∈ (Finset.univ : Finset (Fin n)).powersetCard t,
        dotProduct (xVec (k := k) B T) u • xVec (k := k) B T) =
      (((n - 2 * k).choose (t - k) : ℕ) : ℝ) • u := by
    intro t ht1 ht2
    have hstep1 : ∑ T ∈ (Finset.univ : Finset (Fin n)).powersetCard t,
        dotProduct (xVec (k := k) B T) u • xVec (k := k) B T =
        ∑ T ∈ (Finset.univ : Finset (Fin n)).powersetCard t,
          dotProduct (fVec (k := k) T) u • fVec (k := k) T := by
      have hdiff : ∑ T ∈ (Finset.univ : Finset (Fin n)).powersetCard t,
          dotProduct (fVec (k := k) T) u • (xVec (k := k) B T - fVec (k := k) T) = 0 := by
        funext S
        rw [Finset.sum_apply]
        have hxf : ∀ T : Finset (Fin n), xVec (k := k) B T S - fVec (k := k) T S =
            ∑ r ∈ Finset.Ico 1 k, B (k - r) * ∑ R ∈ T.powersetCard r, vVec (k := k) R S := by
          intro T
          unfold xVec
          have hsplit : Finset.Icc 1 k = insert k (Finset.Ico 1 k) := by
            ext r; simp only [Finset.mem_Icc, Finset.mem_insert, Finset.mem_Ico]; omega
          rw [hsplit, Finset.sum_insert (by simp)]
          have htop : B (k - k) * (((T ∩ S.1).card.choose k : ℕ) : ℝ) = fVec (k := k) T S := by
            rw [Nat.sub_self, hB0, one_mul]
            unfold fVec
            by_cases hST : S.1 ⊆ T
            · rw [if_pos hST, Finset.inter_eq_right.mpr hST, S.2, Nat.choose_self]; simp
            · rw [if_neg hST]
              have hlt : (T ∩ S.1).card < k := by
                calc (T ∩ S.1).card < S.1.card := by
                      apply Finset.card_lt_card
                      rw [Finset.ssubset_iff_subset_ne]
                      refine ⟨Finset.inter_subset_right, fun heq => hST (by rw [← heq]; exact Finset.inter_subset_left)⟩
                  _ = k := S.2
              rw [Nat.choose_eq_zero_of_lt hlt]; simp
          rw [htop, add_sub_cancel_left]
          refine Finset.sum_congr rfl fun r _ => by rw [choose_inter_eq_sum_vVec]
        simp_rw [Pi.smul_apply, Pi.sub_apply, smul_eq_mul, hxf]
        rw [Finset.sum_congr rfl (fun T _ => Finset.mul_sum _ _ _), Finset.sum_comm]
        refine Finset.sum_eq_zero fun r hr => ?_
        have hrew : ∑ T ∈ (Finset.univ : Finset (Fin n)).powersetCard t,
            dotProduct (fVec (k := k) T) u * (B (k - r) * ∑ R ∈ T.powersetCard r, vVec (k := k) R S) =
            B (k - r) * ∑ R ∈ (Finset.univ : Finset (Fin n)).powersetCard r,
              vVec (k := k) R S *
                ∑ T ∈ ((Finset.univ : Finset (Fin n)).powersetCard t).filter (fun T => R ⊆ T),
                  dotProduct (fVec (k := k) T) u := by
          rw [Finset.mul_sum]
          have h1 : ∑ T ∈ (Finset.univ : Finset (Fin n)).powersetCard t,
              dotProduct (fVec (k := k) T) u * (B (k - r) * ∑ R ∈ T.powersetCard r, vVec (k := k) R S) =
              ∑ T ∈ (Finset.univ : Finset (Fin n)).powersetCard t,
                ∑ R ∈ (Finset.univ : Finset (Fin n)).powersetCard r,
                  if R ⊆ T then B (k - r) * vVec (k := k) R S * dotProduct (fVec (k := k) T) u else 0 := by
            refine Finset.sum_congr rfl fun T _ => ?_
            rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_filter]
            have hfilt : ((Finset.univ : Finset (Fin n)).powersetCard r).filter (fun R => R ⊆ T) =
                T.powersetCard r := by
              ext R; simp [Finset.mem_powersetCard, and_comm]
            rw [hfilt]
            refine Finset.sum_congr rfl fun R _ => by ring
          rw [h1, Finset.sum_comm]
          refine Finset.sum_congr rfl fun R _ => ?_
          rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_filter]
          refine Finset.sum_congr rfl fun T _ => by ring
        rw [hrew]
        have hzero : ∑ R ∈ (Finset.univ : Finset (Fin n)).powersetCard r,
            vVec (k := k) R S *
              ∑ T ∈ ((Finset.univ : Finset (Fin n)).powersetCard t).filter (fun T => R ⊆ T),
                dotProduct (fVec (k := k) T) u = 0 := by
          refine Finset.sum_eq_zero fun R hR => ?_
          rw [hlower_vanish t R ht1 ht2 (by rw [(Finset.mem_powersetCard.mp hR).2]; exact (Finset.mem_Ico.mp hr).2),
            mul_zero]
        rw [hzero, mul_zero]
      have hsub : (∑ T ∈ (Finset.univ : Finset (Fin n)).powersetCard t,
          dotProduct (xVec (k := k) B T) u • xVec (k := k) B T) -
          (∑ T ∈ (Finset.univ : Finset (Fin n)).powersetCard t,
            dotProduct (fVec (k := k) T) u • fVec (k := k) T) = 0 := by
        rw [← Finset.sum_sub_distrib]
        calc ∑ T ∈ (Finset.univ : Finset (Fin n)).powersetCard t,
              (dotProduct (xVec (k := k) B T) u • xVec (k := k) B T -
               dotProduct (fVec (k := k) T) u • fVec (k := k) T)
            = ∑ T ∈ (Finset.univ : Finset (Fin n)).powersetCard t,
              dotProduct (fVec (k := k) T) u • (xVec (k := k) B T - fVec (k := k) T) := by
              refine Finset.sum_congr rfl fun T _ => by rw [hxf_dot T, smul_sub]
          _ = 0 := hdiff
      exact sub_eq_zero.mp hsub
    rw [hstep1]
    funext S
    rw [Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
    have hdot_C : ∑ T ∈ (Finset.univ : Finset (Fin n)).powersetCard t,
        (dotProduct (fVec (k := k) T) u • fVec (k := k) T) S =
        dotProduct (fun U : KSet n k => (((n - 2 * k + (S.1 ∩ U.1).card).choose (n - t) : ℕ) : ℝ)) u := by
      unfold dotProduct
      simp_rw [Pi.smul_apply, smul_eq_mul, Finset.sum_mul]
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun U _ => ?_
      rw [← hCt_sum t S U ht1 ht2, Finset.sum_mul]
      refine Finset.sum_congr rfl fun T _ => by ring
    rw [hdot_C]
    have hsym_choose : (n - 2 * k).choose (n - t - k) = (n - 2 * k).choose (t - k) := by
      have h := Nat.choose_symm (show t - k ≤ n - 2 * k by omega)
      rw [show n - 2 * k - (t - k) = n - t - k by omega] at h
      exact h
    have hmemL : (fun U : KSet n k => (((n - 2 * k + (S.1 ∩ U.1).card).choose (n - t) : ℕ) : ℝ)) -
        (((n - 2 * k).choose (t - k) : ℕ) : ℝ) • fVec (k := k) S.1 ∈ lowerSubspace n k := by
      have hexp : (fun U : KSet n k => (((n - 2 * k + (S.1 ∩ U.1).card).choose (n - t) : ℕ) : ℝ)) -
          (((n - 2 * k).choose (t - k) : ℕ) : ℝ) • fVec (k := k) S.1 =
          ∑ r ∈ Finset.range k,
            (((n - 2 * k).choose (n - t - r) : ℕ) : ℝ) •
              ∑ R ∈ S.1.powersetCard r, vVec (k := k) R := by
        funext U
        rw [Pi.sub_apply, Pi.smul_apply, smul_eq_mul, Finset.sum_apply]
        have hv := Nat.add_choose_eq (n - 2 * k) (S.1 ∩ U.1).card (n - t)
        rw [hv, Nat.cast_sum]
        have hsum_anti : ∑ x ∈ Finset.antidiagonal (n - t),
            (((n - 2 * k).choose x.1 * (S.1 ∩ U.1).card.choose x.2 : ℕ) : ℝ) =
            ∑ r ∈ Finset.range (k + 1),
              (((n - 2 * k).choose (n - t - r) : ℕ) : ℝ) * (((S.1 ∩ U.1).card.choose r : ℕ) : ℝ) := by
          have hanti : ∑ x ∈ Finset.antidiagonal (n - t),
              (((n - 2 * k).choose x.1 * (S.1 ∩ U.1).card.choose x.2 : ℕ) : ℝ) =
              ∑ r ∈ Finset.range (n - t + 1),
                (((n - 2 * k).choose (n - t - r) : ℕ) : ℝ) * (((S.1 ∩ U.1).card.choose r : ℕ) : ℝ) := by
            rw [Finset.Nat.sum_antidiagonal_eq_sum_range_succ_mk]
            have hrev := Finset.sum_range_reflect
              (fun r => (((n - 2 * k).choose (n - t - r) : ℕ) : ℝ) * (((S.1 ∩ U.1).card.choose r : ℕ) : ℝ))
              (n - t + 1)
            rw [← hrev]
            refine Finset.sum_congr rfl fun r hr => ?_
            have hrle : r ≤ n - t := Nat.lt_succ_iff.mp (Finset.mem_range.mp hr)
            simp only [Nat.cast_mul, Nat.add_sub_cancel]
            rw [show n - t - (n - t - r) = r by omega]
          rw [hanti]
          rw [← Finset.sum_subset (Finset.range_mono (by omega : k + 1 ≤ n - t + 1))]
          intro r _ hnot
          have hgt : k < r := by rw [Finset.mem_range] at hnot; omega
          have hlt_inter : (S.1 ∩ U.1).card < r :=
            lt_of_le_of_lt (calc (S.1 ∩ U.1).card ≤ S.1.card := Finset.card_le_card Finset.inter_subset_left
              _ = k := S.2) hgt
          rw [Nat.choose_eq_zero_of_lt hlt_inter]; simp
        rw [hsum_anti, Finset.sum_range_succ, hsym_choose]
        have htop : (((n - 2 * k).choose (t - k) : ℕ) : ℝ) * (((S.1 ∩ U.1).card.choose k : ℕ) : ℝ) =
            (((n - 2 * k).choose (t - k) : ℕ) : ℝ) * fVec (k := k) S.1 U := by
          congr 1
          unfold fVec
          by_cases hUS : U.1 ⊆ S.1
          · rw [if_pos hUS, Finset.inter_eq_right.mpr hUS, U.2, Nat.choose_self]; simp
          · rw [if_neg hUS]
            have hlt : (S.1 ∩ U.1).card < k := by
              calc (S.1 ∩ U.1).card < U.1.card := by
                    apply Finset.card_lt_card
                    rw [Finset.ssubset_iff_subset_ne]
                    refine ⟨Finset.inter_subset_right, fun heq => hUS (by rw [← heq]; exact Finset.inter_subset_left)⟩
                _ = k := U.2
            rw [Nat.choose_eq_zero_of_lt hlt]; simp
        rw [htop, add_sub_cancel_right]
        refine Finset.sum_congr rfl fun r _ => ?_
        rw [Pi.smul_apply, Finset.sum_apply, smul_eq_mul, choose_inter_eq_sum_vVec]
      rw [hexp]
      refine Submodule.sum_mem _ fun r hr => Submodule.smul_mem _ _ ?_
      refine Submodule.sum_mem _ fun R hR => ?_
      exact vVec_mem_lowerSubspace_of_lt (by rw [(Finset.mem_powersetCard.mp hR).2]; exact Finset.mem_range.mp hr)
    have h0 := hu _ hmemL
    rw [sub_dotProduct, smul_dotProduct, sub_eq_zero] at h0
    rw [h0, smul_eq_mul]
    congr 1
    unfold dotProduct fVec
    rw [Finset.sum_eq_single S]
    · simp
    · intro U _ hne
      have hnot : ¬ U.1 ⊆ S.1 := fun hsub => hne (Subtype.ext (Finset.eq_of_subset_of_card_le hsub (by rw [U.2, S.2])))
      rw [if_neg hnot, zero_mul]
    · intro h; exact (h (Finset.mem_univ S)).elim
  have hpart : (Finset.univ : Finset (Finset (Fin n))).filter (fun T => k ≤ T.card ∧ T.card ≤ n - k) =
      (Finset.Icc k (n - k)).biUnion (fun t => (Finset.univ : Finset (Fin n)).powersetCard t) := by
    ext T
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_biUnion,
      Finset.mem_Icc, Finset.mem_powersetCard, Finset.subset_univ]
    constructor
    · intro h; exact ⟨T.card, h, rfl⟩
    · rintro ⟨t, ht, rfl⟩; exact ht
  rw [hpart, Finset.sum_biUnion]
  · have hper_t : ∀ t ∈ Finset.Icc k (n - k),
        (∑ T ∈ (Finset.univ : Finset (Fin n)).powersetCard t,
          (wWeight n k T.card * dotProduct (xVec (k := k) B T) u) • xVec (k := k) B T) = u := by
      intro t ht
      have ht1 := (Finset.mem_Icc.mp ht).1
      have ht2 := (Finset.mem_Icc.mp ht).2
      have hfactor : (∑ T ∈ (Finset.univ : Finset (Fin n)).powersetCard t,
          (wWeight n k T.card * dotProduct (xVec (k := k) B T) u) • xVec (k := k) B T) =
          wWeight n k t • ∑ T ∈ (Finset.univ : Finset (Fin n)).powersetCard t,
            dotProduct (xVec (k := k) B T) u • xVec (k := k) B T := by
        rw [Finset.smul_sum]
        refine Finset.sum_congr rfl fun T hT => ?_
        rw [(Finset.mem_powersetCard.mp hT).2, MulAction.mul_smul]
      rw [hfactor, hGt t ht1 ht2, smul_smul]
      unfold wWeight
      have hpos : (((n - 2 * k).choose (t - k) : ℕ) : ℝ) ≠ 0 :=
        Nat.cast_ne_zero.mpr (Nat.choose_pos (by omega)).ne'
      rw [one_div_mul_cancel hpos, one_smul]
    rw [Finset.sum_congr rfl hper_t, Finset.sum_const, Nat.card_Icc, ← Nat.cast_smul_eq_nsmul ℝ]
    congr 1
    exact_mod_cast (show n - k + 1 - k = n - 2 * k + 1 by omega)
  · intro t₁ _ t₂ _ hne
    change Disjoint _ _
    rw [Finset.disjoint_left]
    intro T h1 h2
    exact hne ((Finset.mem_powersetCard.mp h1).2.symm.trans (Finset.mem_powersetCard.mp h2).2)

/-! ### 4. Annihilation of Lower-Order Discrete Derivatives / Unanimity Games (Theorem 2) -/

/-- Exact order-`k` Shapley Interaction Index operator `φ_k^{SII}(g)(S) = ∑_T q_T(S) g(T)`. -/
noncomputable def siiOperator (n k : ℕ) (g : Finset (Fin n) → ℝ) (S : KSet n k) : ℝ :=
  ∑ T : Finset (Fin n), qVec n k T S * g T

/-- **Theorem 2 (Annihilation of lower-order unanimity games)**: For every `R ⊆ Fin n` with
`|R| < k` and every `S ∈ KSet n k`, `φ_k^{SII}(u_R)(S) = 0`, where `u_R(T) = 1_{R ⊆ T}`. -/
theorem siiOperator_annihilates_lower_unanimity {n k : ℕ} {R : Finset (Fin n)} (hR : R.card < k)
    (S : KSet n k) :
    siiOperator n k (fun T => if R ⊆ T then 1 else 0) S = 0 := by
  have hnonempty : (S.1 \ R).Nonempty := by
    rw [Finset.nonempty_iff_ne_empty]
    intro hemp
    have hSR : S.1 ⊆ R := Finset.sdiff_eq_empty_iff_subset.mp hemp
    have := Finset.card_le_card hSR
    rw [S.2] at this; omega
  obtain ⟨i, hi⟩ := hnonempty
  have hiS : i ∈ S.1 := (Finset.mem_sdiff.mp hi).1
  have hiR : i ∉ R := (Finset.mem_sdiff.mp hi).2
  unfold siiOperator
  rw [← Finset.sum_filter_add_sum_filter_not Finset.univ (fun T : Finset (Fin n) => i ∈ T)]
  have hpair : ∑ T ∈ Finset.univ.filter (fun T : Finset (Fin n) => i ∈ T),
      qVec n k T S * (if R ⊆ T then (1 : ℝ) else 0) =
      -(∑ T ∈ Finset.univ.filter (fun T : Finset (Fin n) => ¬ i ∈ T),
        qVec n k T S * (if R ⊆ T then (1 : ℝ) else 0)) := by
    rw [← Finset.sum_neg_distrib]
    refine Finset.sum_bij (fun T _ => T.erase i) ?_ ?_ ?_ ?_
    · intro T _; rw [Finset.mem_filter]; exact ⟨Finset.mem_univ _, Finset.notMem_erase i T⟩
    · intro T₁ h1 T₂ h2 heq
      dsimp only at heq
      have h1i : i ∈ T₁ := (Finset.mem_filter.mp h1).2
      have h2i : i ∈ T₂ := (Finset.mem_filter.mp h2).2
      rw [← Finset.insert_erase h1i, ← Finset.insert_erase h2i, heq]
    · intro T hT
      have hni : i ∉ T := (Finset.mem_filter.mp hT).2
      exact ⟨insert i T, Finset.mem_filter.mpr ⟨Finset.mem_univ _, Finset.mem_insert_self i T⟩,
        Finset.erase_insert hni⟩
    · intro T hT
      dsimp only
      have hiT : i ∈ T := (Finset.mem_filter.mp hT).2
      have hiff : (R ⊆ T) ↔ (R ⊆ T.erase i) := by
        rw [Finset.subset_erase]; exact ⟨fun h => ⟨h, hiR⟩, fun h => h.1⟩
      have hint_eq : T ∩ S.1 = insert i ((T.erase i) ∩ S.1) := by
        ext x
        simp only [Finset.mem_inter, Finset.mem_insert, Finset.mem_erase]
        by_cases hxi : x = i
        · subst hxi; simp [hiT, hiS]
        · simp [hxi]
      have hni_inter : i ∉ (T.erase i) ∩ S.1 := by simp
      have hcard_inter : (T ∩ S.1).card = ((T.erase i) ∩ S.1).card + 1 := by
        rw [hint_eq, Finset.card_insert_of_notMem hni_inter]
      have hcard_T : T.card = (T.erase i).card + 1 := (Finset.card_erase_add_one hiT).symm
      have hle_k : ((T.erase i) ∩ S.1).card + 1 ≤ k := by
        calc ((T.erase i) ∩ S.1).card + 1 = (T ∩ S.1).card := hcard_inter.symm
          _ ≤ S.1.card := Finset.card_le_card Finset.inter_subset_right
          _ = k := S.2
      have hq_neg : qVec n k T S = -qVec n k (T.erase i) S := by
        unfold qVec
        rw [hcard_T, hcard_inter]
        have hsub1 : (T.erase i).card + 1 - (((T.erase i) ∩ S.1).card + 1) =
            (T.erase i).card - ((T.erase i) ∩ S.1).card := by omega
        have hsub2 : k - ((T.erase i) ∩ S.1).card = (k - (((T.erase i) ∩ S.1).card + 1)) + 1 := by omega
        rw [hsub1, hsub2, pow_succ]
        ring
      simp only [hiff, hq_neg]
      ring
  linarith

/-! ### 5. Exact Small-`(n, k)` Concrete Verifications -/

/-- Exact concrete verification at `(n, k) = (2, 1)`: `c = 1`, `A_1 = P_1`,
diagonal entry `1/2`, off-diagonal entry `-1/2`, and `tr(P_1) = 1 = C(2, 1) - C(2, 0)`. -/
theorem verify_n2_k1 :
    (∀ S : KSet 2 1, matA 2 1 S S = 1 / 2) ∧
    (∀ S U : KSet 2 1, S ≠ U → matA 2 1 S U = -1 / 2) ∧
    Matrix.trace (matP 2 1) = 1 := by
  refine ⟨?_, ?_, ?_⟩
  · intro S
    unfold matA mWeight
    simp [S.2]
  · intro S U hne
    have hinter : (S.1 ∩ U.1).card = 0 := by
      have hle : (S.1 ∩ U.1).card ≤ 1 :=
        calc (S.1 ∩ U.1).card ≤ S.1.card := Finset.card_le_card Finset.inter_subset_left
          _ = 1 := S.2
      by_contra h0
      have h1 : (S.1 ∩ U.1).card = 1 := by omega
      have hSU : S.1 = U.1 := by
        have h1' : S.1 ∩ U.1 = S.1 := Finset.eq_of_subset_of_card_le Finset.inter_subset_left (by rw [S.2, h1])
        have h2' : S.1 ∩ U.1 = U.1 := Finset.eq_of_subset_of_card_le Finset.inter_subset_right (by rw [U.2, h1])
        exact h1'.symm.trans h2'
      exact hne (Subtype.ext hSU)
    unfold matA mWeight
    rw [hinter]; norm_num
  · have htr := (matP_is_orthog_projector (n := 2) (k := 1) le_rfl (by omega)).2.2.2.2
    norm_num [Nat.choose] at htr ⊢
    exact htr

/-- Exact concrete verification at `(n, k) = (4, 2)`: `c = 1`, entries `1/3` (`h = 0`),
`-1/6` (`h = 1`), `1/3` (`h = 2`), and `tr(P_2) = 2 = C(4, 2) - C(4, 1)`. -/
theorem verify_n4_k2 :
    (∀ S U : KSet 4 2, (S.1 ∩ U.1).card = 2 → matA 4 2 S U = 1 / 3) ∧
    (∀ S U : KSet 4 2, (S.1 ∩ U.1).card = 1 → matA 4 2 S U = -1 / 6) ∧
    (∀ S U : KSet 4 2, (S.1 ∩ U.1).card = 0 → matA 4 2 S U = 1 / 3) ∧
    Matrix.trace (matP 4 2) = 2 := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro S U h; unfold matA mWeight; rw [h]; norm_num [Nat.choose]
  · intro S U h; unfold matA mWeight; rw [h]; norm_num [Nat.choose]
  · intro S U h; unfold matA mWeight; rw [h]; norm_num [Nat.choose]
  · have htr := (matP_is_orthog_projector (n := 4) (k := 2) (by omega) (by omega)).2.2.2.2
    norm_num [Nat.choose] at htr ⊢
    exact htr

end KernelSHAPIQInverse
