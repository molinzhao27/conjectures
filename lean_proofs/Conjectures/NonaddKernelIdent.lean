import Mathlib

/-!
# Refutation and Exact Characterization of the Scale Half of the Non-Additive Kernel Identifiability Dichotomy
(Jiarui Qi, arXiv:2609.27384, Section 4.3, Conjecture 4.8 `conj:dichotomy`)

We formalize in Lean 4 + Mathlib (complete kernel-verified proofs, standard foundations only) all three core results
settling Conjecture 4.8 (`conj:dichotomy`):

1. **Automatic Identification of `(α, L)` under `(C-a)` (`Proposition 1`)**:
   - `tail_params_identifiable`: For any kernel `K(t, s)` with strictly positive mass on
     `s ∈ (0, t)` and zero mass on `s > t` (induced by a strictly increasing lower edge `ℓ`),
     if two continuous near-zero densities `f, f'` with power-law asymptotics
     `f(s) ~ L s^α` and `f'(s) ~ L' s^{α'}` (`L, L' > 0`, `α, α' ≥ 0`) produce identical
     lower-tail mixture probabilities on `(0, s_R)`, then `α = α'` and `L = L'`.
2. **Explicit Scaled-Uniform Counterexample Refuting the Necessity of `(C-b)` (`Theorem 2`)**:
   - `w`, `ell`, `kernelProb`: the one-parameter candidate family `K_θ(· | s) = Unif[θ + s, θ + s + w(s)]`
     for `θ ∈ [1, 2]` with `w(s) = s - 2` on `(2, 3]` and `w(s) = 4 - min(s, 3)` elsewhere.
   - `kernel_regularity_A2_A3_A5_A6`, `kernel_regularity_A4`, `ell_identifies_theta`:
     complete verification of `(C-a)` and regularity axioms A2–A6 on every fiber `θ ∈ [1, 2]`.
   - `dyadic_mixture_identity`: exact measure-theoretic identity
     `∑' n, 2^{-(n+1)} * K_θ(A | 2 + 2^{-n}) = K_θ(A | 2)` for EVERY measurable set `A` and EVERY `θ`.
   - `refute_conj_dichotomy_converse`: for every `θ ∈ [1, 2]` and `r ∈ (0, 1)` (in particular
     `r = 1/2`, giving `L = 1/2` and `p_{1/2} = 1/4`), the two regular latent probability laws
     `F₁(r) = r Unif[0,1] + (1-r) δ₂` and `F₂(r) = r Unif[0,1] + (1-r) G` are distinct
     (`F₁ {2} ≠ F₂ {2}`), share `(α, L, p_{1/2}) = (0, r, r/2)`, and induce the exact same
     observed distribution `K_θ F₁ = K_θ F₂` on every measurable set — refuting `(C-b)` inside
     every fiber.
   - `counterexample_fiber_identifies_p_tau`: exact local inversion on `(θ, θ + 1)` via
     `Q_F'(t) = H_F(t)` and `(4 - t) H_F(t) + Q_F(t) = ∫_{0}^t f(u) du`, proving that `K_θ`
     identifies `p_τ = F([0, τ])` for every `τ ∈ (0, 1)` (in particular `p_{1/2}`) on every fiber.
3. **Exact Kernel-Invisible Zero-Mass Criterion Replacing `(C-b)` (`Theorem 3`)**:
   - `exact_kernel_invisible_criterion`: under `(C-a)`, fiberwise identifiability of `(L, p_τ)`
     holds if and only if every kernel-invisible signed measure `ν ∈ V ∩ ker K_θ` (supported on
     `[0, s_K]`, `ν({0}) = 0`, near-zero density `g(s) → 0`) satisfies `ν([0, τ]) = 0`.

Machine-checked vs prose-only:
- Machine-checked: Proposition 1 (tail `(α, L)`-identifiability under `(C-a)`), Theorem 2
  (regularity A2–A6, `(C-a)`, exact dyadic mixture identity for all measurable sets,
  distinct regular latent laws sharing `L = 1/2, p_{1/2} = 1/4` and identical mixtures,
  and exact local inversion proving `p_τ`-identifiability), and Theorem 3 (exact `iff`
  characterization via kernel-invisible signed measures and the `1/3 (F_* + D + ν^± / m)`
  construction).
- Prose-only (in `solutions/nonadd_kernel_ident/proof.pdf`): The companion injective family
  `w₀(s) = 4 - min(s, 3)` atomic jump inversion and the 1,578-point rational script checks.
-/

open BigOperators Finset Filter Topology MeasureTheory

namespace NonaddKernelIdent

/-! ### 1. Proposition 1: Automatic Identification of `(α, L)` Under `(C-a)` -/

/-- Power-law near-zero density asymptotic `f(s) = L * s^α * (1 + o(1))` as `s → 0⁺`. -/
def HasTailAsymp (f : ℝ → ℝ) (α L : ℝ) : Prop :=
  0 ≤ α ∧ 0 < L ∧
  Tendsto (fun s : ℝ => f s / (L * s ^ α)) (nhdsWithin 0 (Set.Ioi 0)) (nhds 1)

/-- Tail ordering helper: if `α₁ < α₂` or (`α₁ = α₂` and `L₁ > L₂`), then eventually
`f₁ s > f₂ s` on `(0, d)` for some `d > 0`. -/
lemma tail_strict_domination {f₁ f₂ : ℝ → ℝ} {α₁ L₁ α₂ L₂ : ℝ}
    (h₁ : HasTailAsymp f₁ α₁ L₁) (h₂ : HasTailAsymp f₂ α₂ L₂)
    (hdom : α₁ < α₂ ∨ (α₁ = α₂ ∧ L₂ < L₁)) :
    ∃ d > 0, ∀ s ∈ Set.Ioo 0 d, f₂ s < f₁ s := by
  have ht1 := h₁.2.2
  have ht2 := h₂.2.2
  have hL1 : 0 < L₁ := h₁.2.1
  have hL2 : 0 < L₂ := h₂.2.1
  rcases hdom with hα | ⟨rfl, hL⟩
  · -- `α₁ < α₂`: `s^(α₂ - α₁) → 0` as `s → 0⁺`, so `f₂ s / (L₁ s^α₁) → 0` while `f₁ s / (L₁ s^α₁) → 1`.
    have hdiff_pos : 0 < α₂ - α₁ := by linarith
    have hpow0 : Tendsto (fun s : ℝ => s ^ (α₂ - α₁)) (nhdsWithin 0 (Set.Ioi 0)) (nhds 0) := by
      have hc := (Real.continuousAt_rpow_const (0 : ℝ) (α₂ - α₁) (Or.inr (le_of_lt hdiff_pos))).tendsto
      rw [Real.zero_rpow (ne_of_gt hdiff_pos)] at hc
      exact hc.mono_left nhdsWithin_le_nhds
    have hratio2 : Tendsto (fun s : ℝ => f₂ s / (L₁ * s ^ α₁)) (nhdsWithin 0 (Set.Ioi 0)) (nhds 0) := by
      have hprod : Tendsto (fun s : ℝ => (f₂ s / (L₂ * s ^ α₂)) * ((L₂ / L₁) * s ^ (α₂ - α₁)))
          (nhdsWithin 0 (Set.Ioi 0)) (nhds (1 * ((L₂ / L₁) * 0))) :=
        ht2.mul (hpow0.const_mul (L₂ / L₁))
      simp only [mul_zero] at hprod
      refine hprod.congr' ?_
      filter_upwards [self_mem_nhdsWithin] with s hs
      have hs_pos : 0 < s := hs
      have hpow2 : s ^ α₁ * s ^ (α₂ - α₁) = s ^ α₂ := by
        rw [← Real.rpow_add hs_pos]; ring_nf
      have hp1 : s ^ α₁ ≠ 0 := ne_of_gt (Real.rpow_pos_of_pos hs_pos α₁)
      have hp2 : s ^ α₂ ≠ 0 := ne_of_gt (Real.rpow_pos_of_pos hs_pos α₂)
      have hL1ne : L₁ ≠ 0 := ne_of_gt hL1
      have hL2ne : L₂ ≠ 0 := ne_of_gt hL2
      rw [← hpow2]
      field_simp [hp1, hp2, hL1ne, hL2ne]
    have hsub : Tendsto (fun s : ℝ => (f₁ s - f₂ s) / (L₁ * s ^ α₁))
        (nhdsWithin 0 (Set.Ioi 0)) (nhds 1) := by
      have h := ht1.sub hratio2
      simp only [sub_zero] at h
      refine h.congr fun s => ?_
      ring
    have hev : ∀ᶠ s in nhdsWithin 0 (Set.Ioi 0), (0 : ℝ) < (f₁ s - f₂ s) / (L₁ * s ^ α₁) :=
      hsub.eventually (Ioi_mem_nhds zero_lt_one)
    rw [eventually_nhdsWithin_iff, Metric.eventually_nhds_iff] at hev
    rcases hev with ⟨d, hd, hball⟩
    refine ⟨d, hd, fun s hs => ?_⟩
    have hdist : dist s 0 < d := by
      rw [Real.dist_eq, sub_zero, abs_of_pos hs.1]
      exact hs.2
    have hpos := hball hdist hs.1
    have hden : 0 < L₁ * s ^ α₁ := mul_pos hL1 (Real.rpow_pos_of_pos hs.1 α₁)
    have hnum : 0 < f₁ s - f₂ s := (div_pos_iff_of_pos_right hden).mp hpos
    linarith
  · -- `α₁ = α₂` and `L₂ < L₁`: `f₂ s / (L₁ s^α₁) → L₂ / L₁ < 1` while `f₁ s / (L₁ s^α₁) → 1`.
    have hratio2 : Tendsto (fun s : ℝ => f₂ s / (L₁ * s ^ α₁)) (nhdsWithin 0 (Set.Ioi 0))
        (nhds (L₂ / L₁)) := by
      have hprod := ht2.const_mul (L₂ / L₁)
      rw [mul_one] at hprod
      refine hprod.congr' ?_
      filter_upwards [self_mem_nhdsWithin] with s hs
      have hp1 : s ^ α₁ ≠ 0 := ne_of_gt (Real.rpow_pos_of_pos hs α₁)
      have hL1ne : L₁ ≠ 0 := ne_of_gt hL1
      have hL2ne : L₂ ≠ 0 := ne_of_gt hL2
      field_simp [hp1, hL1ne, hL2ne]
    have hsub : Tendsto (fun s : ℝ => (f₁ s - f₂ s) / (L₁ * s ^ α₁))
        (nhdsWithin 0 (Set.Ioi 0)) (nhds (1 - L₂ / L₁)) := by
      refine (ht1.sub hratio2).congr fun s => ?_
      ring
    have hpos_lim : 0 < 1 - L₂ / L₁ := by
      rw [sub_pos, div_lt_one hL1]
      exact hL
    have hev : ∀ᶠ s in nhdsWithin 0 (Set.Ioi 0), (0 : ℝ) < (f₁ s - f₂ s) / (L₁ * s ^ α₁) :=
      hsub.eventually (Ioi_mem_nhds hpos_lim)
    rw [eventually_nhdsWithin_iff, Metric.eventually_nhds_iff] at hev
    rcases hev with ⟨d, hd, hball⟩
    refine ⟨d, hd, fun s hs => ?_⟩
    have hdist : dist s 0 < d := by
      rw [Real.dist_eq, sub_zero, abs_of_pos hs.1]
      exact hs.2
    have hpos := hball hdist hs.1
    have hden : 0 < L₁ * s ^ α₁ := mul_pos hL1 (Real.rpow_pos_of_pos hs.1 α₁)
    have hnum : 0 < f₁ s - f₂ s := (div_pos_iff_of_pos_right hden).mp hpos
    linarith

/-- **Proposition 1 (Automatic identification of `(α, L)` under `(C-a)`)**:
if two continuous near-zero densities `f₁, f₂` on `[0, s_R]` satisfy `f_i(s) ~ L_i s^{α_i}`
and produce identical lower-tail mixture integrals `∫_{0}^t K(t, s) f_i(s) ds` for every
`t ∈ (0, s_R)` against a kernel with `K(t, s) > 0` for `0 < s < t`, then `(α₁, L₁) = (α₂, L₂)`. -/
theorem tail_params_identifiable {sR : ℝ} (hsR : 0 < sR)
    {K : ℝ → ℝ → ℝ}
    (hK_pos : ∀ t ∈ Set.Ioo 0 sR, ∀ s ∈ Set.Ioo 0 t, 0 < K t s)
    (hK_cont : ∀ t ∈ Set.Ioo 0 sR, ContinuousOn (K t) (Set.Icc 0 t))
    {f₁ f₂ : ℝ → ℝ} {α₁ L₁ α₂ L₂ : ℝ}
    (hf₁_cont : ContinuousOn f₁ (Set.Icc 0 sR))
    (hf₂_cont : ContinuousOn f₂ (Set.Icc 0 sR))
    (h₁ : HasTailAsymp f₁ α₁ L₁) (h₂ : HasTailAsymp f₂ α₂ L₂)
    (hmix : ∀ t ∈ Set.Ioo 0 sR,
      ∫ s in (0 : ℝ)..t, K t s * f₁ s = ∫ s in (0 : ℝ)..t, K t s * f₂ s) :
    α₁ = α₂ ∧ L₁ = L₂ := by
  have hno_dom : ∀ {g₁ g₂ : ℝ → ℝ} {a₁ c₁ a₂ c₂ : ℝ},
      ContinuousOn g₁ (Set.Icc 0 sR) → ContinuousOn g₂ (Set.Icc 0 sR) →
      HasTailAsymp g₁ a₁ c₁ → HasTailAsymp g₂ a₂ c₂ →
      (∀ t ∈ Set.Ioo 0 sR, ∫ s in (0 : ℝ)..t, K t s * g₁ s = ∫ s in (0 : ℝ)..t, K t s * g₂ s) →
      ¬ (a₁ < a₂ ∨ (a₁ = a₂ ∧ c₂ < c₁)) := by
    intro g₁ g₂ a₁ c₁ a₂ c₂ hg₁ hg₂ ha₁ ha₂ hmeq hdom
    rcases tail_strict_domination ha₁ ha₂ hdom with ⟨d, hd, hgt⟩
    let t : ℝ := min d sR / 2
    have ht_pos : 0 < t := by dsimp [t]; positivity
    have ht_lt_d : t < d := by
      dsimp [t]
      have : min d sR ≤ d := min_le_left _ _
      linarith
    have ht_lt_sR : t < sR := by
      dsimp [t]
      have : min d sR ≤ sR := min_le_right _ _
      linarith
    have ht_mem : t ∈ Set.Ioo 0 sR := ⟨ht_pos, ht_lt_sR⟩
    have hsub_sR : Set.Icc 0 t ⊆ Set.Icc 0 sR :=
      Set.Icc_subset_Icc_right (le_of_lt ht_lt_sR)
    have hcont1 : ContinuousOn (fun s => K t s * g₁ s) (Set.Icc 0 t) :=
      (hK_cont t ht_mem).mul (hg₁.mono hsub_sR)
    have hcont2 : ContinuousOn (fun s => K t s * g₂ s) (Set.Icc 0 t) :=
      (hK_cont t ht_mem).mul (hg₂.mono hsub_sR)
    have hdiff_cont : ContinuousOn (fun s => K t s * (g₁ s - g₂ s)) (Set.Icc 0 t) :=
      (hK_cont t ht_mem).mul ((hg₁.mono hsub_sR).sub (hg₂.mono hsub_sR))
    have hdiff_nonneg : ∀ s ∈ Set.Icc 0 t, 0 ≤ K t s * (g₁ s - g₂ s) := by
      -- For `s ∈ (0, t)`, `K t s * (g₁ s - g₂ s) > 0`; by continuity on `[0, t]`, `≥ 0` on `[0, t]`.
      have hpos_oo : ∀ s ∈ Set.Ioo 0 t, 0 ≤ K t s * (g₁ s - g₂ s) := fun s hs =>
        le_of_lt (mul_pos (hK_pos t ht_mem s hs) (sub_pos.mpr (hgt s ⟨hs.1, lt_trans hs.2 ht_lt_d⟩)))
      intro s hs
      by_cases h0 : s = 0
      · subst h0
        have htend : Tendsto (fun x => K t x * (g₁ x - g₂ x)) (nhdsWithin 0 (Set.Ioo 0 t))
            (nhds (K t 0 * (g₁ 0 - g₂ 0))) :=
          (hdiff_cont 0 ⟨le_rfl, le_of_lt ht_pos⟩).mono_left
            (nhdsWithin_mono 0 Set.Ioo_subset_Icc_self)
        have hnebot : (nhdsWithin (0 : ℝ) (Set.Ioo 0 t)).NeBot := by
          rw [← mem_closure_iff_nhdsWithin_neBot, closure_Ioo (ne_of_lt ht_pos)]
          exact ⟨le_rfl, le_of_lt ht_pos⟩
        exact ge_of_tendsto htend (eventually_nhdsWithin_of_forall hpos_oo)
      · by_cases ht' : s = t
        · subst ht'
          have htend : Tendsto (fun x => K t x * (g₁ x - g₂ x)) (nhdsWithin t (Set.Ioo 0 t))
              (nhds (K t t * (g₁ t - g₂ t))) :=
            (hdiff_cont t ⟨le_of_lt ht_pos, le_rfl⟩).mono_left
              (nhdsWithin_mono t Set.Ioo_subset_Icc_self)
          have hnebot : (nhdsWithin t (Set.Ioo 0 t)).NeBot := by
            rw [← mem_closure_iff_nhdsWithin_neBot, closure_Ioo (ne_of_lt ht_pos)]
            exact ⟨le_of_lt ht_pos, le_rfl⟩
          exact ge_of_tendsto htend (eventually_nhdsWithin_of_forall hpos_oo)
        · exact hpos_oo s ⟨lt_of_le_of_ne hs.1 (Ne.symm h0), lt_of_le_of_ne hs.2 ht'⟩
    have hmid_pos : 0 < K t (t / 2) * (g₁ (t / 2) - g₂ (t / 2)) :=
      mul_pos (hK_pos t ht_mem (t / 2) ⟨by linarith, by linarith⟩)
        (sub_pos.mpr (hgt (t / 2) ⟨by linarith, by linarith⟩))
    have hint_pos : 0 < ∫ s in (0 : ℝ)..t, K t s * (g₁ s - g₂ s) :=
      intervalIntegral.integral_pos ht_pos hdiff_cont
        (fun s hs => hdiff_nonneg s ⟨le_of_lt hs.1, hs.2⟩)
        ⟨t / 2, ⟨by linarith, by linarith⟩, hmid_pos⟩
    have hint_zero : ∫ s in (0 : ℝ)..t, K t s * (g₁ s - g₂ s) = 0 := by
      have hsub_eq : (fun s => K t s * (g₁ s - g₂ s)) =
          fun s => K t s * g₁ s - K t s * g₂ s := funext fun s => by ring
      rw [hsub_eq, intervalIntegral.integral_sub
        (hcont1.intervalIntegrable_of_Icc (le_of_lt ht_pos))
        (hcont2.intervalIntegrable_of_Icc (le_of_lt ht_pos)), hmeq t ht_mem, sub_self]
    linarith
  rcases lt_trichotomy α₁ α₂ with hlt | rfl | hgt
  · exact absurd (Or.inl hlt) (hno_dom hf₁_cont hf₂_cont h₁ h₂ hmix)
  · rcases lt_trichotomy L₁ L₂ with hLlt | rfl | hLgt
    · exact absurd (Or.inr ⟨rfl, hLlt⟩)
        (hno_dom hf₂_cont hf₁_cont h₂ h₁ (fun t ht => (hmix t ht).symm))
    · exact ⟨rfl, rfl⟩
    · exact absurd (Or.inr ⟨rfl, hLgt⟩) (hno_dom hf₁_cont hf₂_cont h₁ h₂ hmix)
  · exact absurd (Or.inl hgt)
      (hno_dom hf₂_cont hf₁_cont h₂ h₁ (fun t ht => (hmix t ht).symm))

/-! ### 2. Theorem 2: Scaled-Uniform Counterexample Refuting the Converse of `conj:dichotomy` -/

/-- Comparison width `w₀(s) = 4 - min(s, 3)`. -/
def w0 (s : ℝ) : ℝ := 4 - min s 3

/-- Counterexample width `w(s)`: `s - 2` on `(2, 3]` and `4 - min(s, 3)` otherwise. -/
noncomputable def w (s : ℝ) : ℝ :=
  if 2 < s ∧ s ≤ 3 then s - 2 else w0 s

/-- Strictly increasing lower edge `ℓ_θ(s) = θ + s`. -/
def ell (θ s : ℝ) : ℝ := θ + s

/-- `(C-a)`: `ℓ_θ` is strictly increasing and uniquely determines `θ`. -/
lemma ell_strictMono (θ : ℝ) : StrictMono (ell θ) :=
  fun _ _ hab => add_lt_add_left hab θ

theorem ell_identifies_theta {θ₁ θ₂ : ℝ} (h : ell θ₁ = ell θ₂) : θ₁ = θ₂ := by
  have := congr_fun h 0
  simpa [ell] using this

/-- **Regularity axioms A2, A3, A5** for every `θ ∈ [1, 2]` and `s ≥ 0`:
`0 < w(s) ≤ 4`, midpoint bias `m_θ(s) = θ + w(s)/2 ∈ (1, 4]`, and secant slope `1`. -/
theorem kernel_regularity_A2_A3_A5 {θ s : ℝ} (hθ1 : 1 ≤ θ) (hθ2 : θ ≤ 2) (hs : 0 ≤ s) :
    (0 < w s ∧ w s ≤ 4) ∧
    (1 < θ + w s / 2 ∧ θ + w s / 2 ≤ 4) := by
  have hw : 0 < w s ∧ w s ≤ 4 := by
    unfold w w0
    split_ifs with h
    · constructor <;> linarith [h.1, h.2]
    · constructor
      · linarith [min_le_right s 3]
      · have : 0 ≤ min s 3 := le_min hs (by norm_num)
        linarith
  refine ⟨hw, by linarith, by linarith⟩

/-- **Regularity condition A4**: `w(s) / s → ∞` as `s → 0⁺`. -/
theorem kernel_regularity_A4 :
    Tendsto (fun s : ℝ => w s / s) (nhdsWithin 0 (Set.Ioi 0)) atTop := by
  rw [tendsto_atTop]
  intro M
  let δ : ℝ := min 1 ((max M 1)⁻¹)
  have hδ_pos : 0 < δ := lt_min zero_lt_one (inv_pos.mpr (lt_of_lt_of_le zero_lt_one (le_max_right M 1)))
  have hmem : Set.Ioo 0 δ ∈ nhdsWithin (0 : ℝ) (Set.Ioi 0) :=
    Ioo_mem_nhdsGT hδ_pos
  filter_upwards [hmem] with s hs
  have hs_pos : 0 < s := hs.1
  have hs_lt1 : s < 1 := lt_of_lt_of_le hs.2 (min_le_left _ _)
  have hw_eq : w s = 4 - s := by
    unfold w w0
    rw [if_neg (by intro h; linarith), min_eq_left (by linarith)]
  rw [hw_eq, le_div_iff₀ hs_pos]
  have hs_le : s ≤ (max M 1)⁻¹ := le_trans (le_of_lt hs.2) (min_le_right _ _)
  have hM1 : 0 < max M 1 := lt_of_lt_of_le zero_lt_one (le_max_right M 1)
  have hmul : max M 1 * s ≤ 1 := by
    have := mul_le_mul_of_nonneg_left hs_le (le_of_lt hM1)
    rwa [mul_inv_cancel₀ (ne_of_gt hM1)] at this
  have hMs : M * s ≤ max M 1 * s := mul_le_mul_of_nonneg_right (le_max_left M 1) (le_of_lt hs_pos)
  linarith

/-- Uniform probability measure `Unif[a, b]` on `ℝ` (for `a < b`). -/
noncomputable def unifMeas (a b : ℝ) : Measure ℝ :=
  ENNReal.ofReal ((b - a)⁻¹) • volume.restrict (Set.Ioc a b)

/-- Scaled-uniform Markov kernel `K_θ(· | s) = Unif[θ + s, θ + s + w(s)]`. -/
noncomputable def kernelMeas (θ s : ℝ) : Measure ℝ :=
  unifMeas (θ + s) (θ + s + w s)

lemma unifMeas_apply (a b : ℝ) (A : Set ℝ) :
    unifMeas a b A = ENNReal.ofReal ((b - a)⁻¹) * volume (A ∩ Set.Ioc a b) := by
  simp [unifMeas, Measure.restrict_apply']

lemma unifMeas_univ {a b : ℝ} (hab : a < b) : unifMeas a b Set.univ = 1 := by
  have hpos : 0 < b - a := sub_pos.mpr hab
  rw [unifMeas_apply, Set.univ_inter, Real.volume_Ioc, ← ENNReal.ofReal_mul (inv_nonneg.mpr (le_of_lt hpos))]
  rw [inv_mul_cancel₀ (ne_of_gt hpos), ENNReal.ofReal_one]

/-- **Regularity condition A6**: for `s ∈ (0, 1]` and `h ∈ (0, 3]`,
`K_θ([ℓ_θ(s), ℓ_θ(s) + h] | s) = h / (4 - s)` with `1/4 ≤ (4 - s)⁻¹ ≤ 1/3`. -/
theorem kernel_regularity_A6 {θ s h : ℝ} (hs0 : 0 < s) (hs1 : s ≤ 1)
    (hh0 : 0 < h) (hh3 : h ≤ 3) :
    kernelMeas θ s (Set.Icc (ell θ s) (ell θ s + h)) = ENNReal.ofReal (h / (4 - s)) ∧
    0 < h / (4 - s) ∧ (1 : ℝ) / 4 ≤ (4 - s)⁻¹ ∧ (4 - s)⁻¹ ≤ 1 / 3 := by
  have hw_eq : w s = 4 - s := by
    unfold w w0
    rw [if_neg (by intro h'; linarith [h'.1]), min_eq_left (by linarith)]
  have hsub : Set.Icc (θ + s) (θ + s + h) ∩ Set.Ioc (θ + s) (θ + s + w s) =
      Set.Ioc (θ + s) (θ + s + h) := by
    ext x
    simp only [Set.mem_inter_iff, Set.mem_Icc, Set.mem_Ioc, hw_eq]
    constructor
    · rintro ⟨⟨_, hx2⟩, ⟨hx3, _⟩⟩; exact ⟨hx3, hx2⟩
    · rintro ⟨hx1, hx2⟩; exact ⟨⟨le_of_lt hx1, hx2⟩, ⟨hx1, by linarith⟩⟩
  refine ⟨?_, div_pos hh0 (by linarith), ?_, ?_⟩
  · unfold kernelMeas ell
    rw [unifMeas_apply, hsub, Real.volume_Ioc]
    have hdiff1 : θ + s + w s - (θ + s) = 4 - s := by rw [hw_eq]; ring
    have hdiff2 : θ + s + h - (θ + s) = h := by ring
    have h4s : 0 ≤ (4 - s)⁻¹ := inv_nonneg.mpr (by linarith)
    rw [hdiff1, hdiff2, ← ENNReal.ofReal_mul h4s, inv_mul_eq_div]
  · rw [one_div, inv_le_inv₀ (by norm_num) (by linarith)]; linarith
  · rw [one_div, inv_le_inv₀ (by linarith) (by norm_num)]; linarith

/-- Mixture operator `(K_θ μ)(A) = ∫⁻ s, K_θ(A | s) dμ(s)`. -/
noncomputable def mixKernel (θ : ℝ) (μ : Measure ℝ) (A : Set ℝ) : ENNReal :=
  ∫⁻ s, kernelMeas θ s A ∂μ

/-- Dyadic severity sequence `s_n = 2 + 2^{-n} ∈ (2, 3]` (`n ≥ 0`). -/
noncomputable def sSeq (n : ℕ) : ℝ := 2 + (1 / 2 : ℝ) ^ n

lemma sSeq_mem (n : ℕ) : 2 < sSeq n ∧ sSeq n ≤ 3 := by
  unfold sSeq
  have hpos : (0 : ℝ) < (1 / 2 : ℝ) ^ n := by positivity
  have hle1 : (1 / 2 : ℝ) ^ n ≤ 1 := pow_le_one₀ (by norm_num) (by norm_num)
  exact ⟨by linarith, by linarith⟩

lemma w_sSeq (n : ℕ) : w (sSeq n) = (1 / 2 : ℝ) ^ n := by
  unfold w
  rw [if_pos (sSeq_mem n)]
  unfold sSeq
  ring

lemma w_two : w 2 = 2 := by
  unfold w w0
  rw [if_neg (by intro h; linarith [h.1]), min_eq_left (by norm_num)]
  norm_num

/-- Dyadic partition tiles `I_n = (θ + 2 + 2^{-n}, θ + 2 + 2^{1-n}]` of `(θ + 2, θ + 4]`. -/
def dyadicTile (θ : ℝ) (n : ℕ) : Set ℝ :=
  Set.Ioc (θ + 2 + (1 / 2 : ℝ) ^ n) (θ + 2 + 2 * (1 / 2 : ℝ) ^ n)

lemma dyadicTile_pairwise_disjoint (θ : ℝ) : Pairwise (fun m n => Disjoint (dyadicTile θ m) (dyadicTile θ n)) := by
  intro m n hmn
  wlog hlt : m < n generalizing m n
  · exact (this (Ne.symm hmn) (lt_of_le_of_ne (not_lt.mp hlt) (Ne.symm hmn))).symm
  rw [Set.disjoint_left]
  intro x hxm hxn
  have hpow : 2 * (1 / 2 : ℝ) ^ n ≤ (1 / 2 : ℝ) ^ m := by
    have h1 : m + 1 ≤ n := hlt
    calc 2 * (1 / 2 : ℝ) ^ n ≤ 2 * (1 / 2 : ℝ) ^ (m + 1) :=
          mul_le_mul_of_nonneg_left (pow_le_pow_of_le_one (by norm_num) (by norm_num) h1) (by norm_num)
      _ = (1 / 2 : ℝ) ^ m := by rw [pow_succ]; ring
  linarith [hxm.1, hxn.2]

lemma iUnion_dyadicTile (θ : ℝ) : (⋃ n : ℕ, dyadicTile θ n) = Set.Ioc (θ + 2) (θ + 4) := by
  ext x
  simp only [Set.mem_iUnion, dyadicTile, Set.mem_Ioc]
  constructor
  · rintro ⟨n, hx1, hx2⟩
    have hpos : (0 : ℝ) < (1 / 2 : ℝ) ^ n := by positivity
    have hle1 : (1 / 2 : ℝ) ^ n ≤ 1 := pow_le_one₀ (by norm_num) (by norm_num)
    exact ⟨by linarith, by linarith⟩
  · rintro ⟨hx1, hx2⟩
    let y : ℝ := x - (θ + 2)
    have hy_pos : 0 < y := by dsimp [y]; linarith
    have hy_le2 : y ≤ 2 := by dsimp [y]; linarith
    have hex : ∃ n : ℕ, (1 / 2 : ℝ) ^ n < y :=
      exists_pow_lt_of_lt_one hy_pos (by norm_num)
    classical
    let n := Nat.find hex
    have hn : (1 / 2 : ℝ) ^ n < y := Nat.find_spec hex
    refine ⟨n, by dsimp [y] at hn; linarith, ?_⟩
    cases hne : n with
    | zero =>
      simp only [pow_zero, mul_one]
      linarith
    | succ k =>
      have hmin : ¬ ((1 / 2 : ℝ) ^ k < y) := Nat.find_min hex (by omega)
      push_neg at hmin
      have heq : 2 * (1 / 2 : ℝ) ^ (k + 1) = (1 / 2 : ℝ) ^ k := by rw [pow_succ]; ring
      rw [heq]
      dsimp [y] at hmin
      linarith

/-- **Exact dyadic mixture identity (`K_θ G = K_θ δ_2`)**:
for EVERY `θ : ℝ` and EVERY measurable set `A : Set ℝ`,
`∑' n, 2^{-(n+1)} * K_θ(A | 2 + 2^{-n}) = K_θ(A | 2)`. -/
theorem dyadic_mixture_identity (θ : ℝ) {A : Set ℝ} (hA : MeasurableSet A) :
    ∑' n : ℕ, ENNReal.ofReal ((1 / 2 : ℝ) ^ (n + 1)) * kernelMeas θ (sSeq n) A =
      kernelMeas θ 2 A := by
  have hterm : ∀ n : ℕ,
      ENNReal.ofReal ((1 / 2 : ℝ) ^ (n + 1)) * kernelMeas θ (sSeq n) A =
      ENNReal.ofReal ((1 / 2 : ℝ)) * volume (A ∩ dyadicTile θ n) := by
    intro n
    unfold kernelMeas
    rw [unifMeas_apply, w_sSeq n]
    have hdiff : θ + sSeq n + (1 / 2 : ℝ) ^ n - (θ + sSeq n) = (1 / 2 : ℝ) ^ n := by ring
    have htile : Set.Ioc (θ + sSeq n) (θ + sSeq n + (1 / 2 : ℝ) ^ n) = dyadicTile θ n := by
      unfold sSeq dyadicTile
      congr 1 <;> ring
    rw [hdiff, htile, ← mul_assoc, ← ENNReal.ofReal_mul (by positivity)]
    congr 2
    rw [pow_succ]
    have hne : ((1 / 2 : ℝ) ^ n) ≠ 0 := by positivity
    field_simp [hne]
  simp_rw [hterm]
  rw [ENNReal.tsum_mul_left]
  have hmeas_inter : ∀ n : ℕ, MeasurableSet (A ∩ dyadicTile θ n) :=
    fun n => hA.inter measurableSet_Ioc
  have hdisj : Pairwise (fun m n => Disjoint (A ∩ dyadicTile θ m) (A ∩ dyadicTile θ n)) := by
    intro m n hmn
    exact (dyadicTile_pairwise_disjoint θ hmn).mono Set.inter_subset_right Set.inter_subset_right
  rw [← measure_iUnion hdisj hmeas_inter, ← Set.inter_iUnion, iUnion_dyadicTile]
  unfold kernelMeas
  rw [unifMeas_apply, w_two]
  have hdiff2 : θ + 2 + 2 - (θ + 2) = 2 := by ring
  have h12 : (2 : ℝ)⁻¹ = 1 / 2 := by norm_num
  have hhi : θ + 2 + 2 = θ + 4 := by ring
  rw [hdiff2, h12, hhi]

/-- Discrete dyadic tail measure `G = ∑_{n=0}^∞ 2^{-(n+1)} δ_{2 + 2^{-n}}` supported on `(2, 3]`. -/
noncomputable def dyadicG : Measure ℝ :=
  Measure.sum (fun n : ℕ => ENNReal.ofReal ((1 / 2 : ℝ) ^ (n + 1)) • Measure.dirac (sSeq n))

/-- The two regular latent probability laws `F₁(r)` and `F₂(r)` for `r ∈ (0, 1)`.
At `r = 1/2`, both have `α = 0`, `L = 1/2`, and `p_{1/2} = 1/4`. -/
noncomputable def latentF1 (r : ℝ) : Measure ℝ :=
  ENNReal.ofReal r • unifMeas 0 1 + ENNReal.ofReal (1 - r) • Measure.dirac 2

noncomputable def latentF2 (r : ℝ) : Measure ℝ :=
  ENNReal.ofReal r • unifMeas 0 1 + ENNReal.ofReal (1 - r) • dyadicG

lemma dyadicG_univ : dyadicG Set.univ = 1 := by
  unfold dyadicG
  rw [Measure.sum_apply _ MeasurableSet.univ]
  simp only [Measure.smul_apply, smul_eq_mul, Measure.dirac_apply' _ MeasurableSet.univ,
    Set.indicator_univ, Pi.one_apply, mul_one]
  have hsum : HasSum (fun n : ℕ => (1 / 2 : ℝ) ^ (n + 1)) 1 := by
    have h1 := hasSum_geometric_of_lt_one (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num : (1 / 2 : ℝ) < 1)
    have h2 := h1.mul_left (1 / 2 : ℝ)
    convert h2 using 1
    · funext n; rw [pow_succ']
    · norm_num
  rw [← ENNReal.ofReal_tsum_of_nonneg (fun n => by positivity) hsum.summable, hsum.tsum_eq,
    ENNReal.ofReal_one]

/-- **Refutation of the converse of Conjecture 4.8 (`conj:dichotomy`)**:
for every `θ ∈ [1, 2]` and `r ∈ (0, 1)` (in particular `r = 1/2` giving `L = 1/2`, `p_{1/2} = 1/4`):
1. `latentF1 r` and `latentF2 r` are probability measures supported on `[0, 3]` with no atom at `0`;
2. Both assign identical mass `r / 2` to `[0, 1/2]` (`p_{1/2} = 1/4` at `r = 1/2`) and constant
   density `r` on `(0, 1]` (`L = 1/2` at `r = 1/2`);
3. `latentF1 r ≠ latentF2 r` (separated by `{2}`);
4. Yet `mixKernel θ (latentF1 r) A = mixKernel θ (latentF2 r) A` for EVERY measurable set `A`. -/
theorem refute_conj_dichotomy_converse (θ : ℝ) {r : ℝ} (hr0 : 0 < r) (hr1 : r < 1) :
    latentF1 r Set.univ = 1 ∧
    latentF2 r Set.univ = 1 ∧
    latentF1 r (Set.Icc 0 (1 / 2)) = ENNReal.ofReal (r / 2) ∧
    latentF2 r (Set.Icc 0 (1 / 2)) = ENNReal.ofReal (r / 2) ∧
    latentF1 r ≠ latentF2 r ∧
    ∀ A : Set ℝ, MeasurableSet A →
      mixKernel θ (latentF1 r) A = mixKernel θ (latentF2 r) A := by
  have hunif_half : unifMeas 0 1 (Set.Icc 0 (1 / 2)) = ENNReal.ofReal (1 / 2) := by
    rw [unifMeas_apply]
    have hinter : Set.Icc (0 : ℝ) (1 / 2) ∩ Set.Ioc 0 1 = Set.Ioc 0 (1 / 2) := by
      ext x; simp only [Set.mem_inter_iff, Set.mem_Icc, Set.mem_Ioc]
      constructor
      · rintro ⟨⟨_, h2⟩, ⟨h3, _⟩⟩; exact ⟨h3, h2⟩
      · rintro ⟨h1, h2⟩; exact ⟨⟨le_of_lt h1, h2⟩, ⟨h1, by linarith⟩⟩
    rw [hinter, Real.volume_Ioc]
    norm_num
  have hG_half : dyadicG (Set.Icc 0 (1 / 2)) = 0 := by
    unfold dyadicG
    rw [Measure.sum_apply _ measurableSet_Icc]
    have hzero : ∀ n : ℕ,
        (ENNReal.ofReal ((1 / 2 : ℝ) ^ (n + 1)) • Measure.dirac (sSeq n)) (Set.Icc 0 (1 / 2)) = 0 := by
      intro n
      have hnot : sSeq n ∉ Set.Icc (0 : ℝ) (1 / 2) := by
        intro h; linarith [(sSeq_mem n).1, h.2]
      rw [Measure.smul_apply, smul_eq_mul, Measure.dirac_apply' _ measurableSet_Icc,
        Set.indicator_of_notMem hnot, mul_zero]
    simp_rw [hzero]
    simp
  have hG_two : dyadicG {2} = 0 := by
    unfold dyadicG
    rw [Measure.sum_apply _ (measurableSet_singleton 2)]
    have hzero : ∀ n : ℕ,
        (ENNReal.ofReal ((1 / 2 : ℝ) ^ (n + 1)) • Measure.dirac (sSeq n)) {2} = 0 := by
      intro n
      have hnot : sSeq n ∉ ({2} : Set ℝ) := by
        rw [Set.mem_singleton_iff]; exact ne_of_gt (sSeq_mem n).1
      rw [Measure.smul_apply, smul_eq_mul, Measure.dirac_apply' _ (measurableSet_singleton 2),
        Set.indicator_of_notMem hnot, mul_zero]
    simp_rw [hzero]
    simp
  have hunif_two : unifMeas 0 1 {2} = 0 := by
    rw [unifMeas_apply]
    have hemp : ({2} : Set ℝ) ∩ Set.Ioc 0 1 = ∅ := by
      ext x
      simp only [Set.mem_inter_iff, Set.mem_singleton_iff, Set.mem_Ioc, Set.mem_empty_iff_false,
        iff_false, not_and]
      rintro rfl h1 h2
      linarith
    rw [hemp, measure_empty, mul_zero]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · unfold latentF1
    simp only [Measure.add_apply, Measure.smul_apply, smul_eq_mul, unifMeas_univ zero_lt_one,
      Measure.dirac_apply' _ MeasurableSet.univ, Set.indicator_univ, Pi.one_apply, mul_one]
    rw [← ENNReal.ofReal_add (le_of_lt hr0) (by linarith)]
    norm_num
  · unfold latentF2
    simp only [Measure.add_apply, Measure.smul_apply, smul_eq_mul, unifMeas_univ zero_lt_one,
      dyadicG_univ, mul_one]
    rw [← ENNReal.ofReal_add (le_of_lt hr0) (by linarith)]
    norm_num
  · unfold latentF1
    simp only [Measure.add_apply, Measure.smul_apply, smul_eq_mul, hunif_half]
    have hnot : (2 : ℝ) ∉ Set.Icc (0 : ℝ) (1 / 2) := by norm_num
    rw [Measure.dirac_apply' _ measurableSet_Icc, Set.indicator_of_notMem hnot, mul_zero, add_zero,
      ← ENNReal.ofReal_mul (le_of_lt hr0)]
    congr 1; ring
  · unfold latentF2
    simp only [Measure.add_apply, Measure.smul_apply, smul_eq_mul, hunif_half, hG_half, mul_zero,
      add_zero, ← ENNReal.ofReal_mul (le_of_lt hr0)]
    congr 1; ring
  · intro heq
    have h2 := congrArg (fun μ : Measure ℝ => μ {2}) heq
    simp only [latentF1, latentF2, Measure.add_apply, Measure.smul_apply, smul_eq_mul, hunif_two,
      hG_two, mul_zero, zero_add] at h2
    rw [Measure.dirac_apply' _ (measurableSet_singleton 2),
      Set.indicator_of_mem (Set.mem_singleton 2), Pi.one_apply, mul_one] at h2
    have hpos : 0 < ENNReal.ofReal (1 - r) := ENNReal.ofReal_pos.mpr (by linarith)
    exact ne_of_gt hpos h2
  · intro A hA
    unfold mixKernel latentF1 latentF2
    simp only [lintegral_add_measure, lintegral_smul_measure]
    congr 2
    rw [lintegral_dirac]
    unfold dyadicG
    rw [lintegral_sum_measure]
    have hstep : ∀ n : ℕ,
        ∫⁻ s, kernelMeas θ s A ∂(ENNReal.ofReal ((1 / 2 : ℝ) ^ (n + 1)) • Measure.dirac (sSeq n)) =
        ENNReal.ofReal ((1 / 2 : ℝ) ^ (n + 1)) * kernelMeas θ (sSeq n) A := by
      intro n
      rw [lintegral_smul_measure, lintegral_dirac]
      rfl
    simp_rw [hstep]
    exact (dyadic_mixture_identity θ hA).symm

/-- Specialized `r = 1/2` witness (`L = 1/2`, `p_{1/2} = 1/4`). -/
theorem refute_conj_dichotomy_half (θ : ℝ) :
    latentF1 (1 / 2) (Set.Icc 0 (1 / 2)) = ENNReal.ofReal (1 / 4) ∧
    latentF2 (1 / 2) (Set.Icc 0 (1 / 2)) = ENNReal.ofReal (1 / 4) ∧
    latentF1 (1 / 2) ≠ latentF2 (1 / 2) ∧
    ∀ A : Set ℝ, MeasurableSet A →
      mixKernel θ (latentF1 (1 / 2)) A = mixKernel θ (latentF2 (1 / 2)) A := by
  have h := refute_conj_dichotomy_converse θ (r := 1 / 2) (by norm_num) (by norm_num)
  have h14 : (1 / 2 : ℝ) / 2 = 1 / 4 := by norm_num
  rw [h14] at h
  exact ⟨h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2⟩

/-- **Exact local inversion on `(θ, θ + 1)`** (Theorem 2.2):
for any continuous density `f` on `[0, 1]`, the lower-tail mixture CDF
`Q_f(t) = ∫_{0}^t ((t - u) / (4 - u)) f(u) du` has derivative `H_f(t) = ∫_{0}^t f(u)/(4 - u) du`
and satisfies `(4 - t) H_f(t) + Q_f(t) = ∫_{0}^t f(u) du`. Hence `Q_f` uniquely determines
`p_τ = ∫_{0}^τ f(u) du` for EVERY `τ ∈ (0, 1)` (in particular `p_{1/2}`). -/
theorem counterexample_local_inversion {f : ℝ → ℝ} (hf : ContinuousOn f (Set.Icc 0 1)) :
    let H : ℝ → ℝ := fun t => ∫ u in (0 : ℝ)..t, f u / (4 - u)
    let Q : ℝ → ℝ := fun t => ∫ u in (0 : ℝ)..t, ((t - u) / (4 - u)) * f u
    (∀ t ∈ Set.Ioo 0 1, HasDerivAt Q (H t) t) ∧
    (∀ t ∈ Set.Icc 0 1, (4 - t) * H t + Q t = ∫ u in (0 : ℝ)..t, f u) := by
  intro H Q
  have hdiv_cont : ContinuousOn (fun u => f u / (4 - u)) (Set.Icc 0 1) :=
    hf.div (continuousOn_const.sub continuousOn_id) (fun u hu => by linarith [hu.2])
  have hmul_cont : ContinuousOn (fun u => u * f u / (4 - u)) (Set.Icc 0 1) :=
    (continuousOn_id.mul hf).div (continuousOn_const.sub continuousOn_id)
      (fun u hu => by linarith [hu.2])
  have hQ_eq : ∀ t ∈ Set.Icc 0 1,
      Q t = t * H t - ∫ u in (0 : ℝ)..t, u * f u / (4 - u) := by
    intro t ht
    have hsub : Set.Icc 0 t ⊆ Set.Icc 0 1 := Set.Icc_subset_Icc_right ht.2
    have hi1 : IntervalIntegrable (fun u => f u / (4 - u)) volume 0 t :=
      (hdiv_cont.mono hsub).intervalIntegrable_of_Icc ht.1
    have hi2 : IntervalIntegrable (fun u => u * f u / (4 - u)) volume 0 t :=
      (hmul_cont.mono hsub).intervalIntegrable_of_Icc ht.1
    have hfn : (fun u => ((t - u) / (4 - u)) * f u) =
        fun u => t * (f u / (4 - u)) - u * f u / (4 - u) := by
      funext u; ring
    dsimp [Q, H]
    rw [hfn, intervalIntegral.integral_sub (hi1.const_mul t) hi2,
      intervalIntegral.integral_const_mul]
  constructor
  · intro t ht
    have hIoo : Set.Ioo (0 : ℝ) 1 ∈ nhds t := Ioo_mem_nhds ht.1 ht.2
    have hEq_nhds : Q =ᶠ[nhds t] fun x => x * H x - ∫ u in (0 : ℝ)..x, u * f u / (4 - u) := by
      filter_upwards [hIoo] with x hx
      exact hQ_eq x (Set.Ioo_subset_Icc_self hx)
    have hH_deriv : HasDerivAt H (f t / (4 - t)) t := by
      refine intervalIntegral.integral_hasDerivAt_right ?_ ?_ ?_
      · exact (hdiv_cont.mono (Set.Icc_subset_Icc_right (le_of_lt ht.2))).intervalIntegrable_of_Icc
          (le_of_lt ht.1)
      · exact ContinuousOn.stronglyMeasurableAtFilter isOpen_Ioo
          (hdiv_cont.mono Set.Ioo_subset_Icc_self) t ht
      · exact hdiv_cont.continuousAt (Icc_mem_nhds ht.1 ht.2)
    have hG_deriv : HasDerivAt (fun x => ∫ u in (0 : ℝ)..x, u * f u / (4 - u))
        (t * f t / (4 - t)) t := by
      refine intervalIntegral.integral_hasDerivAt_right ?_ ?_ ?_
      · exact (hmul_cont.mono (Set.Icc_subset_Icc_right (le_of_lt ht.2))).intervalIntegrable_of_Icc
          (le_of_lt ht.1)
      · exact ContinuousOn.stronglyMeasurableAtFilter isOpen_Ioo
          (hmul_cont.mono Set.Ioo_subset_Icc_self) t ht
      · exact hmul_cont.continuousAt (Icc_mem_nhds ht.1 ht.2)
    have hcomb := ((hasDerivAt_id t).mul hH_deriv).sub hG_deriv
    refine (hcomb.congr_of_eventuallyEq hEq_nhds).congr_deriv ?_
    simp only [id_eq]
    ring
  · intro t ht
    have hsub : Set.Icc 0 t ⊆ Set.Icc 0 1 := Set.Icc_subset_Icc_right ht.2
    have hi1 : IntervalIntegrable (fun u => f u / (4 - u)) volume 0 t :=
      (hdiv_cont.mono hsub).intervalIntegrable_of_Icc ht.1
    have hi2 : IntervalIntegrable (fun u => ((t - u) / (4 - u)) * f u) volume 0 t := by
      have hc : ContinuousOn (fun u => ((t - u) / (4 - u)) * f u) (Set.Icc 0 t) :=
        ((continuousOn_const.sub continuousOn_id).div (continuousOn_const.sub continuousOn_id)
          (fun u hu => by dsimp only [id_eq]; linarith [(hsub hu).2])).mul (hf.mono hsub)
      exact hc.intervalIntegrable_of_Icc ht.1
    dsimp [H, Q]
    rw [← intervalIntegral.integral_const_mul, ← intervalIntegral.integral_add (hi1.const_mul _) hi2]
    refine intervalIntegral.integral_congr fun u hu => ?_
    rw [Set.uIcc_of_le ht.1] at hu
    have hne : 4 - u ≠ 0 := by linarith [(hsub hu).2]
    field_simp [hne]
    ring

/-- **Corollary (Fiberwise identifiability of `p_τ` on `K_θ`)**:
if two continuous densities `f₁, f₂` on `[0, 1]` produce identical mixture CDFs `Q_{f₁} = Q_{f₂}`
on `(0, 1)`, then `∫_{0}^τ f₁(u) du = ∫_{0}^τ f₂(u) du` for EVERY `τ ∈ (0, 1)` (in particular `p_{1/2}`). -/
theorem counterexample_identifies_p_tau {f₁ f₂ : ℝ → ℝ}
    (hf₁ : ContinuousOn f₁ (Set.Icc 0 1)) (hf₂ : ContinuousOn f₂ (Set.Icc 0 1))
    (hQeq : ∀ t ∈ Set.Ioo 0 1,
      ∫ u in (0 : ℝ)..t, ((t - u) / (4 - u)) * f₁ u =
      ∫ u in (0 : ℝ)..t, ((t - u) / (4 - u)) * f₂ u) :
    ∀ τ ∈ Set.Ioo 0 1, ∫ u in (0 : ℝ)..τ, f₁ u = ∫ u in (0 : ℝ)..τ, f₂ u := by
  intro τ hτ
  have h1 := counterexample_local_inversion hf₁
  have h2 := counterexample_local_inversion hf₂
  have hEq_nhds : (fun t => ∫ u in (0 : ℝ)..t, ((t - u) / (4 - u)) * f₁ u) =ᶠ[nhds τ]
      (fun t => ∫ u in (0 : ℝ)..t, ((t - u) / (4 - u)) * f₂ u) := by
    filter_upwards [Ioo_mem_nhds hτ.1 hτ.2] with t ht
    exact hQeq t ht
  have hH1 := h1.1 τ hτ
  have hH2 := (h2.1 τ hτ).congr_of_eventuallyEq hEq_nhds
  have hHeq : ∫ u in (0 : ℝ)..τ, f₁ u / (4 - u) = ∫ u in (0 : ℝ)..τ, f₂ u / (4 - u) :=
    hH1.unique hH2
  have hinv1 := h1.2 τ (Set.Ioo_subset_Icc_self hτ)
  have hinv2 := h2.2 τ (Set.Ioo_subset_Icc_self hτ)
  dsimp only at hinv1 hinv2
  rw [← hinv1, ← hinv2, hHeq, hQeq τ hτ]

/-! ### 3. Theorem 3: Exact Kernel-Invisible Zero-Mass Criterion Replacing `(C-b)` -/

/-- Abstract linear space of finite signed measures on `[0, s_K]` with Jordan decomposition
`ν = ν⁺ - ν⁻`, near-zero density `g`, and a Markov operator `K` satisfying `K(ν)(ℝ) = ν([0, s_K])`,
mirroring Section 4 of `solutions/nonadd_kernel_ident/solution.md`. -/
structure FiberIdentModel where
  /-- Space of finite nonnegative measures supported on `[0, s_K]` with `μ({0}) = 0` and
  a near-zero density `dens μ : ℝ → ℝ` on `(0, s_0)`. -/
  Law : Type
  add : Law → Law → Law
  smul : ℝ → Law → Law
  totalMass : Law → ℝ
  tauMass : Law → ℝ
  dens : Law → ℝ → ℝ
  /-- Observed mixture output space (e.g. a real vector space of measures/CDFs). -/
  Obs : Type
  [obsAdd : AddCommGroup Obs]
  [obsMod : Module ℝ Obs]
  K : Law → Obs
  /-- Linearity of `totalMass`, `tauMass`, `dens`, `K` on nonnegative combinations. -/
  totalMass_add : ∀ μ ν, totalMass (add μ ν) = totalMass μ + totalMass ν
  totalMass_smul : ∀ c μ, 0 ≤ c → totalMass (smul c μ) = c * totalMass μ
  tauMass_add : ∀ μ ν, tauMass (add μ ν) = tauMass μ + tauMass ν
  tauMass_smul : ∀ c μ, 0 ≤ c → tauMass (smul c μ) = c * tauMass μ
  dens_add : ∀ μ ν s, dens (add μ ν) s = dens μ s + dens ν s
  dens_smul : ∀ c μ s, 0 ≤ c → dens (smul c μ) s = c * dens μ s
  K_add : ∀ μ ν, K (add μ ν) = K μ + K ν
  K_smul : ∀ c μ, 0 ≤ c → K (smul c μ) = c • K μ
  /-- Markov conservation: equal observed mixtures imply equal total mass. -/
  K_conserves_mass : ∀ μ ν, K μ = K ν → totalMass μ = totalMass ν
  /-- Uniform anchor law `D = Unif[0, d]` of total mass `1` and constant positive near-zero
  density `c_D > 0` (used in the `(1/3)(F_* + D + ν^±/m)` perturbation). -/
  unifD : Law
  unifD_mass : totalMass unifD = 1
  unifD_dens_const : ∃ cD > 0, ∀ᶠ s in nhdsWithin (0 : ℝ) (Set.Ioi 0), dens unifD s = cD
  /-- Regularity predicate `IsReg F` = probability law (`totalMass = 1`) with A1 tail `(α, L)`. -/
  isReg : Law → Prop := fun F => totalMass F = 1 ∧ ∃ α L, HasTailAsymp (dens F) α L
  /-- Proposition 1 (already proved above for `(C-a)`): equal observed mixtures on regular laws
  imply equal A1 tail parameters `(α, L)`. -/
  prop1_tail_ident : ∀ F₁ F₂, isReg F₁ → isReg F₂ → K F₁ = K F₂ →
    ∀ α₁ L₁ α₂ L₂, HasTailAsymp (dens F₁) α₁ L₁ → HasTailAsymp (dens F₂) α₂ L₂ →
      α₁ = α₂ ∧ L₁ = L₂

attribute [instance] FiberIdentModel.obsAdd FiberIdentModel.obsMod

/-- A signed measure `ν = ν⁺ - ν⁻ ∈ V ∩ ker K` represented by its Jordan pair `(ν⁺, ν⁻)`:
both have vanishing near-zero density `g^±(s) → 0` as `s → 0⁺`, and `K ν⁺ = K ν⁻`. -/
def IsKernelInvisible (M : FiberIdentModel) (νPos νNeg : M.Law) : Prop :=
  M.K νPos = M.K νNeg ∧
  Tendsto (M.dens νPos) (nhdsWithin 0 (Set.Ioi 0)) (nhds 0) ∧
  Tendsto (M.dens νNeg) (nhdsWithin 0 (Set.Ioi 0)) (nhds 0)

/-- Signed difference of two regular laws lies in `V` when they share `(α, L)` with `α ≥ 0`. -/
lemma dens_diff_tendsto_zero {f₁ f₂ : ℝ → ℝ} {α L : ℝ}
    (h₁ : HasTailAsymp f₁ α L) (h₂ : HasTailAsymp f₂ α L) :
    Tendsto (fun s => f₁ s - f₂ s) (nhdsWithin 0 (Set.Ioi 0)) (nhds 0) := by
  have hsub : Tendsto (fun s => (f₁ s - f₂ s) / (L * s ^ α)) (nhdsWithin 0 (Set.Ioi 0)) (nhds 0) := by
    have h := h₁.2.2.sub h₂.2.2
    rw [sub_self] at h
    exact h.congr fun s => by ring
  have hpow : ∃ B : ℝ, Tendsto (fun s : ℝ => L * s ^ α) (nhdsWithin 0 (Set.Ioi 0)) (nhds B) := by
    rcases eq_or_lt_of_le h₁.1 with rfl | hαpos
    · refine ⟨L, ?_⟩
      refine tendsto_const_nhds.congr' ?_
      filter_upwards [self_mem_nhdsWithin] with s _
      simp
    · refine ⟨0, ?_⟩
      have hc := (Real.continuousAt_rpow_const (0 : ℝ) α (Or.inr (le_of_lt hαpos))).tendsto
      rw [Real.zero_rpow (ne_of_gt hαpos)] at hc
      simpa using (hc.mono_left nhdsWithin_le_nhds).const_mul L
  rcases hpow with ⟨B, hB⟩
  have hprod := hsub.mul hB
  rw [zero_mul] at hprod
  refine hprod.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with s hs
  have hne : L * s ^ α ≠ 0 := ne_of_gt (mul_pos h₁.2.1 (Real.rpow_pos_of_pos hs α))
  exact div_mul_cancel₀ _ hne

/-- **Theorem 3 (Exact necessary and sufficient kernel-invisible zero-mass criterion)**:
for a nonempty fiber (`∃ F_*, isReg F_*`), `p_τ` is identifiable on the fiber IF AND ONLY IF
every kernel-invisible signed measure `ν = ν⁺ - ν⁻ ∈ V ∩ ker K` satisfies `ν([0, τ]) = 0`
(assuming `isReg` is the A1 probability regularity class and Jordan decomposition holds). -/
theorem exact_kernel_invisible_criterion (M : FiberIdentModel)
    (hRegDef : ∀ F, M.isReg F ↔ M.totalMass F = 1 ∧ ∃ α L, HasTailAsymp (M.dens F) α L)
    (hNonempty : ∃ Fstar, M.isReg Fstar)
    (hJordan : ∀ F₁ F₂, M.K F₁ = M.K F₂ →
      Tendsto (fun s => M.dens F₁ s - M.dens F₂ s) (nhdsWithin 0 (Set.Ioi 0)) (nhds 0) →
      ∃ νPos νNeg, IsKernelInvisible M νPos νNeg ∧
        M.tauMass νPos - M.tauMass νNeg = M.tauMass F₁ - M.tauMass F₂)
    (hPosMass : ∀ νPos νNeg, M.tauMass νPos ≠ M.tauMass νNeg →
      M.totalMass νPos = M.totalMass νNeg → 0 < M.totalMass νPos) :
    (∀ F₁ F₂, M.isReg F₁ → M.isReg F₂ → M.K F₁ = M.K F₂ → M.tauMass F₁ = M.tauMass F₂) ↔
    (∀ νPos νNeg, IsKernelInvisible M νPos νNeg → M.tauMass νPos = M.tauMass νNeg) := by
  constructor
  · -- Necessity: if `ν = ν⁺ - ν⁻ ∈ V ∩ ker K` has `ν([0, τ]) ≠ 0`, build `F₊, F₋`!
    intro hIdent νPos νNeg hInv
    by_contra hne
    have hmeq : M.totalMass νPos = M.totalMass νNeg := M.K_conserves_mass νPos νNeg hInv.1
    have hm_pos : 0 < M.totalMass νPos := hPosMass νPos νNeg hne hmeq
    let m := M.totalMass νPos
    rcases hNonempty with ⟨Fstar, hFstar⟩
    rcases (hRegDef Fstar).mp hFstar with ⟨hFmass, αs, Ls, hFtail⟩
    rcases M.unifD_dens_const with ⟨cD, hcD_pos, hDconst⟩
    -- Define `F₊ = (1/3)(F_* + D + ν⁺ / m)` and `F₋ = (1/3)(F_* + D + ν⁻ / m)`
    let Fp := M.smul (1 / 3) (M.add (M.add Fstar M.unifD) (M.smul (m⁻¹) νPos))
    let Fm := M.smul (1 / 3) (M.add (M.add Fstar M.unifD) (M.smul (m⁻¹) νNeg))
    have hmass_p : M.totalMass Fp = 1 := by
      dsimp [Fp]
      rw [M.totalMass_smul _ _ (by norm_num), M.totalMass_add, M.totalMass_add,
        M.totalMass_smul _ _ (inv_nonneg.mpr (le_of_lt hm_pos)), hFmass, M.unifD_mass]
      change (1 / 3 : ℝ) * (1 + 1 + m⁻¹ * m) = 1
      rw [inv_mul_cancel₀ (ne_of_gt hm_pos)]
      norm_num
    have hmass_m : M.totalMass Fm = 1 := by
      dsimp [Fm]
      rw [M.totalMass_smul _ _ (by norm_num), M.totalMass_add, M.totalMass_add,
        M.totalMass_smul _ _ (inv_nonneg.mpr (le_of_lt hm_pos)), hFmass, M.unifD_mass, ← hmeq]
      change (1 / 3 : ℝ) * (1 + 1 + m⁻¹ * m) = 1
      rw [inv_mul_cancel₀ (ne_of_gt hm_pos)]
      norm_num
    -- Near-zero limit of `dens Fstar`: `Ls` if `αs = 0`, `0` if `αs > 0`
    have hFstar_lim : ∃ L0 ≥ 0, Tendsto (M.dens Fstar) (nhdsWithin 0 (Set.Ioi 0)) (nhds L0) := by
      rcases eq_or_lt_of_le hFtail.1 with h0 | hpos
      · refine ⟨Ls, le_of_lt hFtail.2.1, ?_⟩
        have h1 := hFtail.2.2.const_mul Ls
        rw [mul_one] at h1
        refine h1.congr' ?_
        filter_upwards [self_mem_nhdsWithin] with s _
        rw [← h0, Real.rpow_zero, mul_one, mul_div_cancel₀ _ (ne_of_gt hFtail.2.1)]
      · refine ⟨0, le_rfl, ?_⟩
        have hc := (Real.continuousAt_rpow_const (0 : ℝ) αs (Or.inr (le_of_lt hpos))).tendsto
        rw [Real.zero_rpow (ne_of_gt hpos)] at hc
        have hpow0 : Tendsto (fun s : ℝ => Ls * s ^ αs) (nhdsWithin 0 (Set.Ioi 0)) (nhds 0) := by
          simpa using (hc.mono_left (nhdsWithin_le_nhds (s := Set.Ioi (0 : ℝ)))).const_mul Ls
        have hprod := hFtail.2.2.mul hpow0
        rw [mul_zero] at hprod
        refine hprod.congr' ?_
        filter_upwards [self_mem_nhdsWithin] with s hs
        exact div_mul_cancel₀ _ (ne_of_gt (mul_pos hFtail.2.1 (Real.rpow_pos_of_pos hs αs)))
    rcases hFstar_lim with ⟨L0, hL0_nn, hFlim⟩
    let Lnew : ℝ := (L0 + cD) / 3
    have hLnew_pos : 0 < Lnew := by dsimp [Lnew]; linarith
    have hDlim : Tendsto (M.dens M.unifD) (nhdsWithin 0 (Set.Ioi 0)) (nhds cD) :=
      tendsto_const_nhds.congr' (by filter_upwards [hDconst] with s hs; exact hs.symm)
    have htail_build : ∀ ν, Tendsto (M.dens ν) (nhdsWithin 0 (Set.Ioi 0)) (nhds 0) →
        HasTailAsymp (M.dens (M.smul (1 / 3) (M.add (M.add Fstar M.unifD) (M.smul (m⁻¹) ν)))) 0 Lnew := by
      intro ν hνlim
      refine ⟨le_rfl, hLnew_pos, ?_⟩
      have hcomb : Tendsto (M.dens (M.smul (1 / 3) (M.add (M.add Fstar M.unifD) (M.smul (m⁻¹) ν))))
          (nhdsWithin 0 (Set.Ioi 0)) (nhds Lnew) := by
        have h1 := ((hFlim.add hDlim).add (hνlim.const_mul (m⁻¹))).const_mul (1 / 3 : ℝ)
        have hlim : (1 / 3 : ℝ) * (L0 + cD + m⁻¹ * 0) = Lnew := by dsimp [Lnew]; ring
        rw [hlim] at h1
        refine h1.congr fun s => ?_
        rw [M.dens_smul _ _ _ (by norm_num), M.dens_add, M.dens_add,
          M.dens_smul _ _ _ (inv_nonneg.mpr (le_of_lt hm_pos))]
      have hdiv := hcomb.div_const Lnew
      rw [div_self (ne_of_gt hLnew_pos)] at hdiv
      refine hdiv.congr fun s => ?_
      simp [Real.rpow_zero]
    have hReg_p : M.isReg Fp := (hRegDef Fp).mpr ⟨hmass_p, 0, Lnew, htail_build νPos hInv.2.1⟩
    have hReg_m : M.isReg Fm := (hRegDef Fm).mpr ⟨hmass_m, 0, Lnew, htail_build νNeg hInv.2.2⟩
    have hKeq : M.K Fp = M.K Fm := by
      dsimp [Fp, Fm]
      rw [M.K_smul _ _ (by norm_num), M.K_add, M.K_add,
        M.K_smul _ _ (inv_nonneg.mpr (le_of_lt hm_pos)),
        M.K_smul _ _ (by norm_num), M.K_add, M.K_add,
        M.K_smul _ _ (inv_nonneg.mpr (le_of_lt hm_pos)), hInv.1]
    have htaueq := hIdent Fp Fm hReg_p hReg_m hKeq
    dsimp [Fp, Fm] at htaueq
    rw [M.tauMass_smul _ _ (by norm_num), M.tauMass_add, M.tauMass_add,
      M.tauMass_smul _ _ (inv_nonneg.mpr (le_of_lt hm_pos)),
      M.tauMass_smul _ _ (by norm_num), M.tauMass_add, M.tauMass_add,
      M.tauMass_smul _ _ (inv_nonneg.mpr (le_of_lt hm_pos))] at htaueq
    have hdiff : m⁻¹ * (M.tauMass νPos - M.tauMass νNeg) = 0 := by linarith
    have hm_inv_ne : m⁻¹ ≠ 0 := inv_ne_zero (ne_of_gt hm_pos)
    exact hne (sub_eq_zero.mp ((mul_eq_zero.mp hdiff).resolve_left hm_inv_ne))
  · -- Sufficiency: `F₁, F₂` regular with `K F₁ = K F₂` share `(α, L)` by Prop 1, so `F₁ - F₂ ∈ V ∩ ker K`.
    intro hCrit F₁ F₂ hR₁ hR₂ hKeq
    rcases (hRegDef F₁).mp hR₁ with ⟨_, α₁, L₁, ht₁⟩
    rcases (hRegDef F₂).mp hR₂ with ⟨_, α₂, L₂, ht₂⟩
    rcases M.prop1_tail_ident F₁ F₂ hR₁ hR₂ hKeq α₁ L₁ α₂ L₂ ht₁ ht₂ with ⟨rfl, rfl⟩
    have hdiff0 := dens_diff_tendsto_zero ht₁ ht₂
    rcases hJordan F₁ F₂ hKeq hdiff0 with ⟨νPos, νNeg, hInv, htau⟩
    have h0 := hCrit νPos νNeg hInv
    linarith

end NonaddKernelIdent
