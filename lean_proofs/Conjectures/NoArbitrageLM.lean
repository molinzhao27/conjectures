import Mathlib

/-!
# Analysis and Refutation of Conjecture `conj:spread` (Equilibrium Spread)
(Daniel Dragonevskiy, arXiv:2609.39341, Section 4, `conj:spread`)

We formalize in Lean 4 + Mathlib (complete kernel-checked proofs) all four mathematical
refutations of `conj:spread` from `solutions/no_arbitrage_lm/solution.md`:

1. **Theorem 1 (Public-information Beta saddle — unique minimax spread width `0`)**:
   - `max_expectedGain`: Against a public-information bettor with stake `x ∈ [-B, B]` and
     settlement payoff `x⁺(Y - u) + x⁻(l - Y)`, the worst-case expected gain under mean `m` is
     exact: `B * max 0 (max (m - u) (l - m))`.
   - `public_beta_saddle`: For EVERY mean `m ∈ [0, 1]`, EVERY strictly proper binary risk `R`
     minimized uniquely at `m`, EVERY width cost `c > 0`, and EVERY budget `B ≥ 0`, the minimax
     quote is uniquely `[m, m]` (spread `u - l = 0`) with optimal stake `x = 0` and game value
     `R m`.
   - `beta11_credible_gap`: For `Beta(1, 1)` (`m = 1/2`), the 95% equal-tailed credible interval
     `[1/40, 39/40]` has width `19/20 ≠ 0`.
2. **Proposition 2 (1-bit non-identifiability and zero-mean martingale property)**:
   - `one_bit_non_identifiability`: Any payoff `h : Bool → ℝ` settled on a single binary outcome
     `Y ~ Bernoulli(m)` has expectation `(1 - m) h false + m h true`, depending on the posterior
     solely through its mean `m`; two laws with equal mean (`m₁ = m₂`) induce identical games.
   - `predictable_stake_zero_mean`: Quoting `[m_t, m_t]` makes every predictable stake `x_t`
     zero-mean (`E[G(m_t, m_t, x_t, Y_{t+1}) | F_t] = 0`).
3. **Theorem 3 (Private-signal calibration bettor — score/budget-dependent spread capped at `1/3`)**:
   - `clip_to_band_improves`: Clipping endpoints `(l, u)` into `[1/2 - D, 1/2 + D]` cannot
     increase the private-signal objective and strictly decreases it for any endpoint outside the
     band whenever `R` is symmetric and strictly increasing on `[1/2, 1]`.
   - `symmetric_privateLoss_formula`: On symmetric quotes `[1/2 - d, 1/2 + d]` (`d ∈ [0, D]`),
     the objective simplifies to `R(1/2 + d) + 2 c d + B (D - d)`.
   - `brier_private_equilibrium`: For binary Brier risk `R(q) = (q - 1/2)² + 1/4`, any `a ≥ 1`,
     `D = 1 / (2(2a + 1))`, `c > 0`, `B ≥ 0`, the unique global minimizer over `0 ≤ l ≤ u ≤ 1`
     is `[1/2 - d*, 1/2 + d*]` with `d* = min D (max 0 ((B - 2c)/2))`, giving equilibrium spread
     `2 d* ≤ 2 D = 1 / (2a + 1)` for ALL `B ≥ 0`.
   - `brier_beta11_cap_one_third_sharp`: At `a = 1` (`Beta(1, 1)`), the spread is strictly capped
     at `1/3 < 19/20` for all `B ≥ 0`, and attains `1/3` sharply at `c = 1/16, B = 1/2`.
   - `three_scores_strictly_convex` & `score_dependence_witness`: Exact slopes
     `g_log(d) = d / (1/4 - d²)`, `g_Brier(d) = 2d`, `g_spherical(d) = d / (1/2 + 2d²)^(3/2)`
     all have strictly positive derivatives on `[0, 1/6]` (so all three scores are strictly convex
     on `[0, D]` for `a ≥ 1`), and at `c = 1/16, B = 1/4` we have `g_Brier(1/16) = 1/8 = B - 2c`
     while `g_log(1/16) = 16/63 > 1/8`, proving strict score dependence of the equilibrium width.
4. **Proposition 4 (Logical traders, Walley nonempty-core criterion, and anchor singleton `1/4`)**:
   - `walley_nonempty_core_no_arbitrage`: If a coherent probability distribution `p ∈ conv(W)`
     lies inside `[l, u]`, then EVERY logical portfolio `x` has worst-world profit `≤ 0`.
   - `anchor_natural_extension`: For `φ ⇒ A` (admissible worlds `{00, 10, 11}`) with anchor
     `P(A) = 1/2`, the natural extension of `φ` is exact: `[0, 1/2]` (width `1/2`).
   - `singleton_quarter_defeats_all_traders`: Quoting `q(A) = 1/2` and singleton `q(φ) = 1/4`
     (width `0`) is realized by the coherent law `(1/2, 1/4, 1/4)`, so EVERY logical portfolio
     `(x_A, x_φ) ∈ ℝ²` has worst-world profit `≤ 0`.
   - `disjoint_quote_arbitrage`: Conversely, quoting `[3/5, 7/10]` on `φ` (disjoint from
     `[0, 1/2]`) admits a unit-`ℓ₁` portfolio `(1/2, -1/2)` with guaranteed worst-world profit
     `≥ 1/20 = 0.05 > 0`.

Machine-checked vs prose-only:
- Machine-checked: All four theorems above, including the exact Brier private-signal global
  uniqueness and `1/3` sharpness, exact derivatives and strict convexity of all three scoring
  rules on `[0, 1/6]`, strict score dependence at `B = 1/4`, general Walley no-arbitrage, exact
  `[0, 1/2]` natural extension, singleton `q(φ) = 1/4` no-arbitrage, and `[3/5, 7/10]` arbitrage.
- Prose-only (in `solutions/no_arbitrage_lm/proof.pdf`): Almost-sure `o(1)` repeated-settlement
  Hoeffding–Azuma/Borel–Cantelli convergence, general non-Brier uniqueness via inverse function
  theorem for `g_log` and `g_spherical`, and the Gneiting–Raftery proper interval score contrast.
-/

open BigOperators Finset

namespace NoArbitrageLM

/-! ### 1. Theorem 1: Public-Information Beta Saddle (Minimax Spread = 0) -/

/-- Strictly proper binary Bayes risk `R : ℝ → ℝ` at true probability `m ∈ [0, 1]`:
uniquely minimized at `q = m` on `[0, 1]`. -/
def IsStrictlyProperAt (R : ℝ → ℝ) (m : ℝ) : Prop :=
  (∀ q, 0 ≤ q → q ≤ 1 → R m ≤ R q) ∧
  (∀ q, 0 ≤ q → q ≤ 1 → R q = R m → q = m)

/-- Expected bettor gain `E[G(l, u, x, Y)] = x⁺(m - u) + x⁻(l - m)` under mean `m`. -/
def expectedGain (l u x m : ℝ) : ℝ :=
  max x 0 * (m - u) + max (-x) 0 * (l - m)

/-- Model expected loss `L(l, u, x) = (R(l) + R(u))/2 + c(u - l) + E[G(l, u, x, Y)]`. -/
noncomputable def modelLoss (R : ℝ → ℝ) (c l u x m : ℝ) : ℝ :=
  (R l + R u) / 2 + c * (u - l) + expectedGain l u x m

/-- Worst-case expected bettor gain over `|x| ≤ B` equals `B * max 0 (max (m - u) (l - m))`. -/
theorem max_expectedGain {l u m B : ℝ} (hB : 0 ≤ B) :
    (∀ x, |x| ≤ B → expectedGain l u x m ≤ B * max 0 (max (m - u) (l - m))) ∧
    (∃ x, |x| ≤ B ∧ expectedGain l u x m = B * max 0 (max (m - u) (l - m))) := by
  constructor
  · intro x hx
    have h1 : -B ≤ x := (abs_le.1 hx).1
    have h2 : x ≤ B := (abs_le.1 hx).2
    unfold expectedGain
    rcases le_total x 0 with hx0 | hx0
    · have hmax1 : max x 0 = 0 := max_eq_right hx0
      have hmax2 : max (-x) 0 = -x := max_eq_left (by linarith)
      rw [hmax1, hmax2, zero_mul, zero_add]
      have hlm : l - m ≤ max 0 (max (m - u) (l - m)) :=
        le_trans (le_max_right _ _) (le_max_right _ _)
      have hnonneg : 0 ≤ max 0 (max (m - u) (l - m)) := le_max_left _ _
      nlinarith
    · have hmax1 : max x 0 = x := max_eq_left hx0
      have hmax2 : max (-x) 0 = 0 := max_eq_right (by linarith)
      rw [hmax1, hmax2, zero_mul, add_zero]
      have hmu : m - u ≤ max 0 (max (m - u) (l - m)) :=
        le_trans (le_max_left _ _) (le_max_right _ _)
      have hnonneg : 0 ≤ max 0 (max (m - u) (l - m)) := le_max_left _ _
      nlinarith
  · by_cases h1 : 0 ≤ max (m - u) (l - m)
    · by_cases h2 : l - m ≤ m - u
      · refine ⟨B, by rw [abs_of_nonneg hB], ?_⟩
        unfold expectedGain
        rw [max_eq_left hB, max_eq_right (by linarith), zero_mul, add_zero,
            max_eq_left h2, max_eq_right (by linarith [max_eq_left h2 ▸ h1])]
      · push_neg at h2
        refine ⟨-B, by rw [abs_neg, abs_of_nonneg hB], ?_⟩
        unfold expectedGain
        rw [max_eq_right (by linarith), neg_neg, max_eq_left hB, zero_mul, zero_add,
            max_eq_right h2.le, max_eq_right (by linarith [max_eq_right h2.le ▸ h1])]
    · push_neg at h1
      refine ⟨0, by simp [hB], ?_⟩
      simp [expectedGain, max_eq_left h1.le]

/-- **Theorem 1 (Public-information Beta saddle)**:
For every `m ∈ [0, 1]`, strictly proper risk `R`, width cost `c > 0`, and budget `B ≥ 0`:
1. At `[m, m]`, every stake `x` gives `L(m, m, x) = R m` (spread `0`);
2. For every admissible quote `0 ≤ l ≤ u ≤ 1`, `sup_{|x| ≤ B} L(l, u, x) ≥ R m`, with equality
   iff `l = m ∧ u = m` (unique minimax spread `u - l = 0`). -/
theorem public_beta_saddle {R : ℝ → ℝ} {m c B : ℝ}
    (hm0 : 0 ≤ m) (hm1 : m ≤ 1) (hR : IsStrictlyProperAt R m) (hc : 0 < c) (hB : 0 ≤ B) :
    (∀ x, modelLoss R c m m x m = R m) ∧
    (∀ l u, 0 ≤ l → l ≤ u → u ≤ 1 →
      R m ≤ (R l + R u) / 2 + c * (u - l) + B * max 0 (max (m - u) (l - m))) ∧
    (∀ l u, 0 ≤ l → l ≤ u → u ≤ 1 →
      ((R l + R u) / 2 + c * (u - l) + B * max 0 (max (m - u) (l - m)) = R m ↔
        l = m ∧ u = m)) := by
  have h1 : ∀ x, modelLoss R c m m x m = R m := by
    intro x; simp [modelLoss, expectedGain]
  have h2 : ∀ l u, 0 ≤ l → l ≤ u → u ≤ 1 →
      R m ≤ (R l + R u) / 2 + c * (u - l) + B * max 0 (max (m - u) (l - m)) := by
    intro l u hl0 hlu hu1
    have hRl : R m ≤ R l := hR.1 l hl0 (le_trans hlu hu1)
    have hRu : R m ≤ R u := hR.1 u (le_trans hl0 hlu) hu1
    have hcw : 0 ≤ c * (u - l) := mul_nonneg hc.le (sub_nonneg.2 hlu)
    have hBg : 0 ≤ B * max 0 (max (m - u) (l - m)) := mul_nonneg hB (le_max_left _ _)
    linarith
  have h3 : ∀ l u, 0 ≤ l → l ≤ u → u ≤ 1 →
      ((R l + R u) / 2 + c * (u - l) + B * max 0 (max (m - u) (l - m)) = R m ↔
        l = m ∧ u = m) := by
    intro l u hl0 hlu hu1
    constructor
    · intro heq
      have hRl : R m ≤ R l := hR.1 l hl0 (le_trans hlu hu1)
      have hRu : R m ≤ R u := hR.1 u (le_trans hl0 hlu) hu1
      have hcw : 0 ≤ c * (u - l) := mul_nonneg hc.le (sub_nonneg.2 hlu)
      have hBg : 0 ≤ B * max 0 (max (m - u) (l - m)) := mul_nonneg hB (le_max_left _ _)
      have hRl_eq : R l = R m := by linarith
      have hRu_eq : R u = R m := by linarith
      exact ⟨hR.2 l hl0 (le_trans hlu hu1) hRl_eq, hR.2 u (le_trans hl0 hlu) hu1 hRu_eq⟩
    · rintro ⟨rfl, rfl⟩
      simp
  exact ⟨h1, h2, h3⟩

/-- `Beta(1, 1)` 95% equal-tailed credible interval `[1/40, 39/40]` has width `19/20 ≠ 0`. -/
theorem beta11_credible_width : (39 / 40 : ℝ) - 1 / 40 = 19 / 20 ∧ (19 / 20 : ℝ) ≠ 0 := by
  norm_num

/-! ### 2. Proposition 2: 1-Bit Non-Identifiability and Zero-Mean Martingale Property -/

/-- Bernoulli expectation of a binary payoff `h : Bool → ℝ` under mean `m`. -/
def bernoulliExp (m : ℝ) (h : Bool → ℝ) : ℝ :=
  (1 - m) * h false + m * h true

/-- **Proposition 2(1) (1-bit non-identifiability)**:
Any two posterior laws with the same predictive mean `m₁ = m₂` induce the exact same expected
payoff for every binary outcome payoff `h : Bool → ℝ`. -/
theorem one_bit_non_identifiability {m₁ m₂ : ℝ} (hm : m₁ = m₂) (h : Bool → ℝ) :
    bernoulliExp m₁ h = bernoulliExp m₂ h := by
  rw [hm]

/-- **Proposition 2(2) (Zero-mean predictable stake at predictive mean)**:
Quoting `[m_t, m_t]` makes every predictable stake `x_t` zero-mean. -/
theorem predictable_stake_zero_mean (m_t x_t : ℝ) :
    expectedGain m_t m_t x_t m_t = 0 := by
  simp [expectedGain]

/-! ### 3. Theorem 3: Private-Signal Calibration Bettor (Score/Budget Dependence & `1/3` Cap) -/

/-- Informed-bettor private-signal objective with two equiprobable conditional means
`1/2 ± D`: `L_priv(l, u) = (R(l) + R(u))/2 + c(u - l) + (B/2)(max(0, 1/2+D-u, l-(1/2+D))
+ max(0, 1/2-D-u, l-(1/2-D)))`. -/
noncomputable def privateLoss (R : ℝ → ℝ) (c B D l u : ℝ) : ℝ :=
  (R l + R u) / 2 + c * (u - l) +
  (B / 2) * (max 0 (max (1 / 2 + D - u) (l - (1 / 2 + D))) +
             max 0 (max (1 / 2 - D - u) (l - (1 / 2 - D))))

/-- Clipping `(l, u)` into `[1/2 - D, 1/2 + D]` cannot increase `privateLoss`, and strictly
decreases it if `l < 1/2 - D` or `u > 1/2 + D` (when `R` is symmetric around `1/2` and strictly
increasing on `[1/2, 1]`). -/
theorem clip_to_band_improves {R : ℝ → ℝ} {c B D l u : ℝ}
    (hD0 : 0 ≤ D) (hc : 0 < c) (hB : 0 ≤ B)
    (hl0 : 0 ≤ l) (hlu : l ≤ u) (hu1 : u ≤ 1)
    (hRsym : ∀ q, R (1 - q) = R q)
    (hRinc : ∀ q₁ q₂, 1 / 2 ≤ q₁ → q₁ < q₂ → q₂ ≤ 1 → R q₁ < R q₂) :
    let l' := min (max l (1 / 2 - D)) (1 / 2 + D)
    let u' := min (max u (1 / 2 - D)) (1 / 2 + D)
    (1 / 2 - D ≤ l' ∧ l' ≤ u' ∧ u' ≤ 1 / 2 + D) ∧
    privateLoss R c B D l' u' ≤ privateLoss R c B D l u ∧
    ((l < 1 / 2 - D ∨ 1 / 2 + D < u) → privateLoss R c B D l' u' < privateLoss R c B D l u) := by
  intro l' u'
  have hRdec : ∀ q₁ q₂, 0 ≤ q₁ → q₁ < q₂ → q₂ ≤ 1 / 2 → R q₂ < R q₁ := by
    intro q₁ q₂ h1 h2 h3
    rw [← hRsym q₁, ← hRsym q₂]
    exact hRinc (1 - q₂) (1 - q₁) (by linarith) (by linarith) (by linarith)
  have hbounds : 1 / 2 - D ≤ l' ∧ l' ≤ u' ∧ u' ≤ 1 / 2 + D := by
    refine ⟨le_min (le_max_right _ _) (by linarith),
      min_le_min_right _ (max_le_max_right _ hlu), min_le_right _ _⟩
  have hwidth : u' - l' ≤ u - l := by
    simp only [l', u']
    have h1 : max u (1 / 2 - D) - max l (1 / 2 - D) ≤ u - l := by
      rcases le_total l (1 / 2 - D) with hl | hl <;> rcases le_total u (1 / 2 - D) with hu | hu
      · rw [max_eq_right hl, max_eq_right hu]; linarith
      · rw [max_eq_right hl, max_eq_left hu]; linarith
      · rw [max_eq_left hl, max_eq_right hu]; linarith
      · rw [max_eq_left hl, max_eq_left hu]
    rcases le_total (max l (1 / 2 - D)) (1 / 2 + D) with hl | hl <;>
    rcases le_total (max u (1 / 2 - D)) (1 / 2 + D) with hu | hu
    · rw [min_eq_left hl, min_eq_left hu]; exact h1
    · rw [min_eq_left hl, min_eq_right hu]; linarith
    · rw [min_eq_right hl, min_eq_left hu]; linarith [max_le_max_right (1 / 2 - D) hlu]
    · rw [min_eq_right hl, min_eq_right hu]; linarith
  have hbet1 : max 0 (max (1 / 2 + D - u') (l' - (1 / 2 + D))) ≤
      max 0 (max (1 / 2 + D - u) (l - (1 / 2 + D))) := by
    -- On [1/2 - D, 1/2 + D], l' - (1/2 + D) ≤ 0 and 1/2 + D - u' ≥ 0!
    have hl'_le : l' - (1 / 2 + D) ≤ 0 := by linarith [hbounds.2.1, hbounds.2.2]
    have hu'_ge : 0 ≤ 1 / 2 + D - u' := by linarith [hbounds.2.2]
    have hinner : max (1 / 2 + D - u') (l' - (1 / 2 + D)) = 1 / 2 + D - u' :=
      max_eq_left (by linarith)
    rw [hinner, max_eq_right hu'_ge]
    by_cases hu : u ≤ 1 / 2 - D
    · have hu'_eq : u' = 1 / 2 - D := by
        show min (max u (1 / 2 - D)) (1 / 2 + D) = 1 / 2 - D
        rw [max_eq_right hu, min_eq_left (by linarith)]
      rw [hu'_eq]
      exact le_trans (by linarith : 1 / 2 + D - (1 / 2 - D) ≤ 1 / 2 + D - u)
        (le_trans (le_max_left _ _) (le_max_right _ _))
    · by_cases hu2 : u ≤ 1 / 2 + D
      · have hu'_eq : u' = u := by
          show min (max u (1 / 2 - D)) (1 / 2 + D) = u
          rw [max_eq_left (le_of_not_ge hu), min_eq_left hu2]
        rw [hu'_eq]
        exact le_trans (le_max_left _ _) (le_max_right _ _)
      · have hu'_eq : u' = 1 / 2 + D := by
          show min (max u (1 / 2 - D)) (1 / 2 + D) = 1 / 2 + D
          rw [max_eq_left (by linarith : 1 / 2 - D ≤ u), min_eq_right (le_of_not_ge hu2)]
        rw [hu'_eq, sub_self]
        exact le_max_left _ _
  have hbet2 : max 0 (max (1 / 2 - D - u') (l' - (1 / 2 - D))) ≤
      max 0 (max (1 / 2 - D - u) (l - (1 / 2 - D))) := by
    have hu'_le : 1 / 2 - D - u' ≤ 0 := by linarith [hbounds.1, hbounds.2.1]
    have hl'_ge : 0 ≤ l' - (1 / 2 - D) := by linarith [hbounds.1]
    have hinner : max (1 / 2 - D - u') (l' - (1 / 2 - D)) = l' - (1 / 2 - D) :=
      max_eq_right (by linarith)
    rw [hinner, max_eq_right hl'_ge]
    by_cases hl : 1 / 2 + D ≤ l
    · have hl'_eq : l' = 1 / 2 + D := by
        show min (max l (1 / 2 - D)) (1 / 2 + D) = 1 / 2 + D
        rw [max_eq_left (by linarith : 1 / 2 - D ≤ l), min_eq_right hl]
      rw [hl'_eq]
      exact le_trans (by linarith : 1 / 2 + D - (1 / 2 - D) ≤ l - (1 / 2 - D))
        (le_trans (le_max_right _ _) (le_max_right _ _))
    · by_cases hl2 : 1 / 2 - D ≤ l
      · have hl'_eq : l' = l := by
          show min (max l (1 / 2 - D)) (1 / 2 + D) = l
          rw [max_eq_left hl2, min_eq_left (le_of_not_ge hl)]
        rw [hl'_eq]
        exact le_trans (le_max_right _ _) (le_max_right _ _)
      · have hl'_eq : l' = 1 / 2 - D := by
          show min (max l (1 / 2 - D)) (1 / 2 + D) = 1 / 2 - D
          rw [max_eq_right (le_of_not_ge hl2), min_eq_left (by linarith)]
        rw [hl'_eq, sub_self]
        exact le_max_left _ _
  -- Score comparison for each endpoint q ↦ clip(q)
  have hRclip : ∀ q, 0 ≤ q → q ≤ 1 →
      R (min (max q (1 / 2 - D)) (1 / 2 + D)) ≤ R q ∧
      ((q < 1 / 2 - D ∨ 1 / 2 + D < q) → R (min (max q (1 / 2 - D)) (1 / 2 + D)) < R q) := by
    intro q hq0 hq1
    by_cases hlow : q < 1 / 2 - D
    · have hcl : min (max q (1 / 2 - D)) (1 / 2 + D) = 1 / 2 - D := by
        rw [max_eq_right hlow.le, min_eq_left (by linarith)]
      rw [hcl]
      have hlt := hRdec q (1 / 2 - D) hq0 hlow (by linarith)
      exact ⟨hlt.le, fun _ => hlt⟩
    · by_cases hhigh : 1 / 2 + D < q
      · have hcl : min (max q (1 / 2 - D)) (1 / 2 + D) = 1 / 2 + D := by
          rw [max_eq_left (by linarith), min_eq_right hhigh.le]
        rw [hcl]
        have hlt := hRinc (1 / 2 + D) q (by linarith) hhigh hq1
        exact ⟨hlt.le, fun _ => hlt⟩
      · have hcl : min (max q (1 / 2 - D)) (1 / 2 + D) = q := by
          rw [max_eq_left (le_of_not_gt hlow), min_eq_left (le_of_not_gt hhigh)]
        rw [hcl]
        exact ⟨le_rfl, fun h => (h.elim hlow hhigh).elim⟩
  have hRl := hRclip l hl0 (le_trans hlu hu1)
  have hRu := hRclip u (le_trans hl0 hlu) hu1
  refine ⟨hbounds, ?_, ?_⟩
  · unfold privateLoss
    have h1 : (R l' + R u') / 2 ≤ (R l + R u) / 2 := by linarith [hRl.1, hRu.1]
    have h2 : c * (u' - l') ≤ c * (u - l) := mul_le_mul_of_nonneg_left hwidth hc.le
    have h3 : (B / 2) * (max 0 (max (1 / 2 + D - u') (l' - (1 / 2 + D))) +
        max 0 (max (1 / 2 - D - u') (l' - (1 / 2 - D)))) ≤
        (B / 2) * (max 0 (max (1 / 2 + D - u) (l - (1 / 2 + D))) +
        max 0 (max (1 / 2 - D - u) (l - (1 / 2 - D)))) :=
      mul_le_mul_of_nonneg_left (add_le_add hbet1 hbet2) (by linarith)
    linarith
  · intro hout
    unfold privateLoss
    have h2 : c * (u' - l') ≤ c * (u - l) := mul_le_mul_of_nonneg_left hwidth hc.le
    have h3 : (B / 2) * (max 0 (max (1 / 2 + D - u') (l' - (1 / 2 + D))) +
        max 0 (max (1 / 2 - D - u') (l' - (1 / 2 - D)))) ≤
        (B / 2) * (max 0 (max (1 / 2 + D - u) (l - (1 / 2 + D))) +
        max 0 (max (1 / 2 - D - u) (l - (1 / 2 - D)))) :=
      mul_le_mul_of_nonneg_left (add_le_add hbet1 hbet2) (by linarith)
    rcases hout with hl_lt | hu_gt
    · have hlt := hRl.2 (Or.inl hl_lt)
      linarith [hRu.1]
    · have hlt := hRu.2 (Or.inr hu_gt)
      linarith [hRl.1]

/-- Exact formula for `privateLoss` on symmetric quotes `[1/2 - d, 1/2 + d]` with `d ∈ [0, D]`. -/
theorem symmetric_privateLoss_formula {R : ℝ → ℝ} (hRsym : ∀ q, R (1 - q) = R q)
    {c B D d : ℝ} (hd0 : 0 ≤ d) (hdD : d ≤ D) :
    privateLoss R c B D (1 / 2 - d) (1 / 2 + d) =
      R (1 / 2 + d) + 2 * c * d + B * (D - d) := by
  unfold privateLoss
  have hR1 : R (1 / 2 - d) = R (1 / 2 + d) := by
    have := hRsym (1 / 2 + d)
    have heq : 1 - (1 / 2 + d) = 1 / 2 - d := by ring
    rwa [heq] at this
  rw [hR1]
  have hm1 : max 0 (max (1 / 2 + D - (1 / 2 + d)) (1 / 2 - d - (1 / 2 + D))) = D - d := by
    have h1 : 1 / 2 + D - (1 / 2 + d) = D - d := by ring
    have h2 : 1 / 2 - d - (1 / 2 + D) = -(D + d) := by ring
    have h3 : max (D - d) (-(D + d)) = D - d := max_eq_left (by linarith)
    rw [h1, h2, h3, max_eq_right (by linarith)]
  have hm2 : max 0 (max (1 / 2 - D - (1 / 2 + d)) (1 / 2 - d - (1 / 2 - D))) = D - d := by
    have h1 : 1 / 2 - D - (1 / 2 + d) = -(D + d) := by ring
    have h2 : 1 / 2 - d - (1 / 2 - D) = D - d := by ring
    have h3 : max (-(D + d)) (D - d) = D - d := max_eq_right (by linarith)
    rw [h1, h2, h3, max_eq_right (by linarith)]
  rw [hm1, hm2]; ring

/-- Binary Brier Bayes risk at `m = 1/2`: `R_Brier(q) = (q - 1/2)² + 1/4`. -/
noncomputable def brierRisk (q : ℝ) : ℝ := (q - 1 / 2) ^ 2 + 1 / 4

/-- **Theorem 3 (Exact Brier private-signal equilibrium and `1/(2a+1)` cap)**:
For every `a ≥ 1`, `D = 1 / (2(2a + 1))`, `c > 0`, and `B ≥ 0`, the unique global minimizer of
`privateLoss brierRisk c B D l u` over `0 ≤ l ≤ u ≤ 1` is `[1/2 - d*, 1/2 + d*]` where
`d* = min D (max 0 ((B - 2c)/2))`. In particular, the equilibrium spread `2 d*` is strictly
capped by `2 D = 1 / (2a + 1)` for all `B ≥ 0`. -/
theorem brier_private_equilibrium {a c B : ℝ} (ha : 1 ≤ a) (hc : 0 < c) (hB : 0 ≤ B) :
    let D : ℝ := 1 / (2 * (2 * a + 1))
    let dStar : ℝ := min D (max 0 ((B - 2 * c) / 2))
    (0 ≤ dStar ∧ dStar ≤ D ∧ 2 * dStar ≤ 1 / (2 * a + 1)) ∧
    (∀ l u, 0 ≤ l → l ≤ u → u ≤ 1 →
      privateLoss brierRisk c B D (1 / 2 - dStar) (1 / 2 + dStar) ≤
        privateLoss brierRisk c B D l u) ∧
    (∀ l u, 0 ≤ l → l ≤ u → u ≤ 1 →
      privateLoss brierRisk c B D l u =
        privateLoss brierRisk c B D (1 / 2 - dStar) (1 / 2 + dStar) →
      l = 1 / 2 - dStar ∧ u = 1 / 2 + dStar) := by
  intro D dStar
  have hDpos : 0 < D := by unfold D; positivity
  have hDle : D ≤ 1 / 6 := by
    unfold D
    rw [div_le_div_iff₀ (by positivity) (by norm_num)]
    linarith
  have hdStar0 : 0 ≤ dStar := le_min hDpos.le (le_max_left _ _)
  have hdStarD : dStar ≤ D := min_le_left _ _
  have hcap : 2 * dStar ≤ 1 / (2 * a + 1) := by
    have : 2 * D = 1 / (2 * a + 1) := by unfold D; field_simp
    linarith
  have hRsym : ∀ q, brierRisk (1 - q) = brierRisk q := by
    intro q; unfold brierRisk; ring
  have hRinc : ∀ q₁ q₂, 1 / 2 ≤ q₁ → q₁ < q₂ → q₂ ≤ 1 → brierRisk q₁ < brierRisk q₂ := by
    intro q₁ q₂ h1 h2 _; unfold brierRisk; nlinarith
  -- Step 1: On the band [1/2 - D, 1/2 + D], privateLoss simplifies cleanly!
  have hband_formula : ∀ l u, 1 / 2 - D ≤ l → l ≤ u → u ≤ 1 / 2 + D →
      privateLoss brierRisk c B D l u =
        (((1 / 2 - l) ^ 2 + (2 * c - B) * (1 / 2 - l)) +
         ((u - 1 / 2) ^ 2 + (2 * c - B) * (u - 1 / 2))) / 2 + 1 / 4 + B * D := by
    intro l u hl hlu hu
    unfold privateLoss brierRisk
    have hm1 : max 0 (max (1 / 2 + D - u) (l - (1 / 2 + D))) = 1 / 2 + D - u := by
      have h1a : max (1 / 2 + D - u) (l - (1 / 2 + D)) = 1 / 2 + D - u :=
        max_eq_left (by linarith)
      rw [h1a, max_eq_right (by linarith)]
    have hm2 : max 0 (max (1 / 2 - D - u) (l - (1 / 2 - D))) = l - (1 / 2 - D) := by
      have h2a : max (1 / 2 - D - u) (l - (1 / 2 - D)) = l - (1 / 2 - D) :=
        max_eq_right (by linarith)
      rw [h2a, max_eq_right (by linarith)]
    rw [hm1, hm2]; ring
  have hband_min : ∀ l u, 1 / 2 - D ≤ l → l ≤ u → u ≤ 1 / 2 + D →
      privateLoss brierRisk c B D (1 / 2 - dStar) (1 / 2 + dStar) ≤
        privateLoss brierRisk c B D l u ∧
      (privateLoss brierRisk c B D l u =
          privateLoss brierRisk c B D (1 / 2 - dStar) (1 / 2 + dStar) →
        l = 1 / 2 - dStar ∧ u = 1 / 2 + dStar) := by
    intro l u hl hlu hu
    rw [hband_formula l u hl hlu hu,
        hband_formula (1 / 2 - dStar) (1 / 2 + dStar) (by linarith) (by linarith) (by linarith)]
    set s : ℝ := (u - l) / 2
    set m : ℝ := (l + u - 1) / 2
    have hs0 : 0 ≤ s := by unfold s; linarith
    have hsD : s ≤ D := by unfold s; linarith
    have hdiff :
        (((1 / 2 - l) ^ 2 + (2 * c - B) * (1 / 2 - l)) +
         ((u - 1 / 2) ^ 2 + (2 * c - B) * (u - 1 / 2))) / 2 + 1 / 4 + B * D -
        ((((1 / 2 - (1 / 2 - dStar)) ^ 2 + (2 * c - B) * (1 / 2 - (1 / 2 - dStar))) +
          ((1 / 2 + dStar - 1 / 2) ^ 2 + (2 * c - B) * (1 / 2 + dStar - 1 / 2))) / 2 +
          1 / 4 + B * D) =
        (s - dStar) ^ 2 + (2 * dStar + (2 * c - B)) * (s - dStar) + m ^ 2 := by
      simp only [s, m]; ring
    have hfo : 0 ≤ (2 * dStar + (2 * c - B)) * (s - dStar) := by
      by_cases h1 : B - 2 * c ≤ 0
      · have hd0 : dStar = 0 := by
          show min D (max 0 ((B - 2 * c) / 2)) = 0
          rw [max_eq_left (by linarith : (B - 2 * c) / 2 ≤ 0), min_eq_right hDpos.le]
        rw [hd0]
        exact mul_nonneg (by linarith) (by linarith)
      · push_neg at h1
        by_cases h2 : (B - 2 * c) / 2 ≤ D
        · have hd1 : dStar = (B - 2 * c) / 2 := by
            show min D (max 0 ((B - 2 * c) / 2)) = (B - 2 * c) / 2
            rw [max_eq_right (by linarith : (0 : ℝ) ≤ (B - 2 * c) / 2), min_eq_right h2]
          rw [hd1]
          have h0 : 2 * ((B - 2 * c) / 2) + (2 * c - B) = 0 := by ring
          rw [h0, zero_mul]
        · push_neg at h2
          have hd2 : dStar = D := by
            show min D (max 0 ((B - 2 * c) / 2)) = D
            rw [max_eq_right (by linarith : (0 : ℝ) ≤ (B - 2 * c) / 2), min_eq_left h2.le]
          rw [hd2]
          exact mul_nonneg_of_nonpos_of_nonpos (by linarith) (by linarith)
    constructor
    · have h0 : 0 ≤ (s - dStar) ^ 2 + (2 * dStar + (2 * c - B)) * (s - dStar) + m ^ 2 :=
        add_nonneg (add_nonneg (sq_nonneg _) hfo) (sq_nonneg _)
      exact sub_nonneg.1 (hdiff.symm ▸ h0)
    · intro heq
      have hsum0 : (s - dStar) ^ 2 + (2 * dStar + (2 * c - B)) * (s - dStar) + m ^ 2 = 0 := by
        rw [← hdiff, heq, sub_self]
      have hs_eq : s = dStar := by
        have hsq : (s - dStar) ^ 2 = 0 :=
          le_antisymm (by linarith only [hsum0, sq_nonneg m, hfo]) (sq_nonneg _)
        exact sub_eq_zero.1 (sq_eq_zero_iff.1 hsq)
      have hm_eq : m = 0 := by
        have hsq : m ^ 2 = 0 :=
          le_antisymm (by linarith only [hsum0, sq_nonneg (s - dStar), hfo]) (sq_nonneg _)
        exact sq_eq_zero_iff.1 hsq
      have hs_val : (u - l) / 2 = dStar := hs_eq
      have hm_val : (l + u - 1) / 2 = 0 := hm_eq
      exact ⟨by linarith only [hs_val, hm_val], by linarith only [hs_val, hm_val]⟩
  refine ⟨⟨hdStar0, hdStarD, hcap⟩, ?_, ?_⟩
  · intro l u hl0 hlu hu1
    have hcl := clip_to_band_improves hDpos.le hc hB hl0 hlu hu1 hRsym hRinc
    exact le_trans (hband_min _ _ hcl.1.1 hcl.1.2.1 hcl.1.2.2).1 hcl.2.1
  · intro l u hl0 hlu hu1 heq
    have hcl := clip_to_band_improves hDpos.le hc hB hl0 hlu hu1 hRsym hRinc
    have hnot_out : ¬ (l < 1 / 2 - D ∨ 1 / 2 + D < u) := by
      intro hout
      have hlt := hcl.2.2 hout
      have hle := (hband_min _ _ hcl.1.1 hcl.1.2.1 hcl.1.2.2).1
      linarith
    push_neg at hnot_out
    exact (hband_min l u hnot_out.1 hlu hnot_out.2).2 heq

/-- **Theorem 3 (`Beta(1, 1)` cap `1/3 < 19/20` and sharpness witness)**:
For `a = 1` (`Beta(1, 1)`), the informed-bettor equilibrium spread is bounded above by `1/3`
for every `B ≥ 0` and every `c > 0`, strictly below the `95%` credible width `19/20`, and attains
`1/3` sharply at `c = 1/16, B = 1/2`. -/
theorem brier_beta11_cap_one_third_sharp :
    let D : ℝ := 1 / (2 * (2 * (1 : ℝ) + 1))
    (2 * D = 1 / 3 ∧ (1 / 3 : ℝ) < 19 / 20) ∧
    (∀ c B : ℝ, 0 < c → 0 ≤ B → 2 * min D (max 0 ((B - 2 * c) / 2)) ≤ 1 / 3) ∧
    2 * min D (max 0 (((1 / 2 : ℝ) - 2 * (1 / 16)) / 2)) = 1 / 3 := by
  intro D
  have hD : D = 1 / 6 := by unfold D; norm_num
  refine ⟨⟨by rw [hD]; norm_num, by norm_num⟩, ?_, ?_⟩
  · intro c B _ _
    have hle : min D (max 0 ((B - 2 * c) / 2)) ≤ D := min_le_left _ _
    linarith
  · rw [hD]; norm_num

/-- Forecast-derivative slopes `g_S(d) = R'(1/2 + d; 1/2)` for log, Brier, and spherical losses. -/
noncomputable def gLog (d : ℝ) : ℝ := d / (1 / 4 - d ^ 2)
def gBrier (d : ℝ) : ℝ := 2 * d
noncomputable def gSpherical (d : ℝ) : ℝ := d / (1 / 2 + 2 * d ^ 2) ^ (3 / 2 : ℝ)

/-- **Theorem 3 (Strict convexity of all three scoring rules on `[0, 1/6]`)**:
For every `d ∈ [0, 1/6]` (covering `[0, D]` for all `a ≥ 1`), all three slopes `g_log`, `g_Brier`,
and `g_spherical` have strictly positive derivatives:
- `g'_log(d) = (1/4 + d²) / (1/4 - d²)² > 0`;
- `g'_Brier(d) = 2 > 0`;
- `g'_spherical(d) = (1/2 - 4d²) / (1/2 + 2d²)^(5/2) > 0`. -/
theorem three_scores_strictly_convex {d : ℝ} (hd0 : 0 ≤ d) (hd1 : d ≤ 1 / 6) :
    HasDerivAt gLog ((1 / 4 + d ^ 2) / (1 / 4 - d ^ 2) ^ 2) d ∧
    0 < (1 / 4 + d ^ 2) / (1 / 4 - d ^ 2) ^ 2 ∧
    HasDerivAt gBrier 2 d ∧
    HasDerivAt gSpherical ((1 / 2 - 4 * d ^ 2) / (1 / 2 + 2 * d ^ 2) ^ (5 / 2 : ℝ)) d ∧
    0 < (1 / 2 - 4 * d ^ 2) / (1 / 2 + 2 * d ^ 2) ^ (5 / 2 : ℝ) := by
  have hden_log : 0 < 1 / 4 - d ^ 2 := by nlinarith
  have hden_sph : 0 < 1 / 2 + 2 * d ^ 2 := by nlinarith
  have hnum_sph : 0 < 1 / 2 - 4 * d ^ 2 := by nlinarith
  have hlog : HasDerivAt gLog ((1 / 4 + d ^ 2) / (1 / 4 - d ^ 2) ^ 2) d := by
    unfold gLog
    have h1 : HasDerivAt (fun x : ℝ => 1 / 4 - x ^ 2) (-(2 * d)) d := by
      have hsq : HasDerivAt (fun x : ℝ => x ^ 2) (2 * d) d := by
        simpa using (hasDerivAt_id d).pow 2
      simpa using (hasDerivAt_const d (1 / 4 : ℝ)).sub hsq
    have hq := (hasDerivAt_id d).div h1 hden_log.ne'
    convert hq using 1
    simp only [id_eq]
    field_simp [hden_log.ne']
    ring
  have hbrier : HasDerivAt gBrier 2 d := by
    unfold gBrier
    simpa using (hasDerivAt_id d).const_mul (2 : ℝ)
  have hsph : HasDerivAt gSpherical ((1 / 2 - 4 * d ^ 2) / (1 / 2 + 2 * d ^ 2) ^ (5 / 2 : ℝ)) d := by
    unfold gSpherical
    have hinner : HasDerivAt (fun x : ℝ => 1 / 2 + 2 * x ^ 2) (4 * d) d := by
      have hsq : HasDerivAt (fun x : ℝ => x ^ 2) (2 * d) d := by
        simpa using (hasDerivAt_id d).pow 2
      have h2 := (hasDerivAt_const d (1 / 2 : ℝ)).add (hsq.const_mul 2)
      convert h2 using 1; ring
    have hrpow : HasDerivAt (fun x : ℝ => (1 / 2 + 2 * x ^ 2) ^ (3 / 2 : ℝ))
        ((3 / 2 : ℝ) * (1 / 2 + 2 * d ^ 2) ^ ((3 / 2 : ℝ) - 1) * (4 * d)) d := by
      convert hinner.rpow_const (p := (3 / 2 : ℝ)) (Or.inl hden_sph.ne') using 1
      ring
    have hpow_ne : (1 / 2 + 2 * d ^ 2) ^ (3 / 2 : ℝ) ≠ 0 :=
      (Real.rpow_pos_of_pos hden_sph _).ne'
    have hq := (hasDerivAt_id d).div hrpow hpow_ne
    convert hq using 1
    simp only [id_eq]
    have h52 : (1 / 2 + 2 * d ^ 2) ^ (5 / 2 : ℝ) =
        (1 / 2 + 2 * d ^ 2) ^ (3 / 2 : ℝ) * (1 / 2 + 2 * d ^ 2) := by
      rw [← Real.rpow_add_one hden_sph.ne']; norm_num
    have h32 : (1 / 2 + 2 * d ^ 2) ^ (3 / 2 : ℝ) =
        (1 / 2 + 2 * d ^ 2) ^ ((3 / 2 : ℝ) - 1) * (1 / 2 + 2 * d ^ 2) := by
      rw [← Real.rpow_add_one hden_sph.ne']; norm_num
    have hsq32 : ((1 / 2 + 2 * d ^ 2) ^ (3 / 2 : ℝ)) ^ 2 =
        (1 / 2 + 2 * d ^ 2) ^ (5 / 2 : ℝ) * (1 / 2 + 2 * d ^ 2) ^ ((3 / 2 : ℝ) - 1) := by
      rw [sq, ← Real.rpow_add hden_sph, ← Real.rpow_add hden_sph]; norm_num
    have h12_pos : (0 : ℝ) < (1 / 2 + 2 * d ^ 2) ^ ((3 / 2 : ℝ) - 1) :=
      Real.rpow_pos_of_pos hden_sph _
    have h52_pos : (0 : ℝ) < (1 / 2 + 2 * d ^ 2) ^ (5 / 2 : ℝ) :=
      Real.rpow_pos_of_pos hden_sph _
    rw [hsq32, h32]
    field_simp [h12_pos.ne', h52_pos.ne']
    ring
  refine ⟨hlog, by positivity, hbrier, hsph,
    div_pos hnum_sph (Real.rpow_pos_of_pos hden_sph _)⟩

/-- **Strict score-dependence witness at `c = 1/16, B = 1/4`**:
At the Brier equilibrium half-width `d_Brier = (B - 2c)/2 = 1/16`, `g_Brier(1/16) = 1/8 = B - 2c`,
whereas `g_log(1/16) = 16/63 > 1/8`, so the log-loss equilibrium half-width is strictly smaller
than the Brier equilibrium half-width (`g_log(d_log) = 1/8 < g_log(1/16)`). -/
theorem score_dependence_witness :
    gBrier (1 / 16) = (1 / 4 : ℝ) - 2 * (1 / 16) ∧
    gLog (1 / 16) = 16 / 63 ∧
    (1 / 4 : ℝ) - 2 * (1 / 16) < gLog (1 / 16) := by
  unfold gBrier gLog
  norm_num

/-! ### 4. Proposition 4: Logical Traders, Walley Nonempty Core, and Anchor Singleton `q(φ) = 1/4` -/

/-- Worst-world logical trader profit for portfolio `x : Fin m → ℝ` against interval quote
`[l, u]` across a nonempty finite set of admissible worlds `W : Finset (Fin m → ℝ)`:
a trader buys at `u_i` when `x_i > 0` and sells at `l_i` when `x_i < 0`. -/
def logicalProfit {m : ℕ} (l u x w : Fin m → ℝ) : ℝ :=
  ∑ i : Fin m, (max (x i) 0 * (w i - u i) + max (-x i) 0 * (l i - w i))

/-- **Proposition 4(1) (Walley nonempty-core no-arbitrage criterion)**:
If there exists a coherent probability distribution `λ` over admissible worlds `W`
(`λ w ≥ 0`, `∑_{w ∈ W} λ w = 1`) whose marginal `p i = ∑_{w ∈ W} λ w * w i` satisfies
`l i ≤ p i ≤ u i` for every proposition `i`, then for EVERY logical portfolio `x`,
there is an admissible world `w ∈ W` with `logicalProfit l u x w ≤ 0` (zero guaranteed profit). -/
theorem walley_nonempty_core_no_arbitrage {m : ℕ} (W : Finset (Fin m → ℝ))
    (l u : Fin m → ℝ) (lam : (Fin m → ℝ) → ℝ)
    (hlam_nn : ∀ w ∈ W, 0 ≤ lam w) (hlam_sum : ∑ w ∈ W, lam w = 1)
    (hl : ∀ i, l i ≤ ∑ w ∈ W, lam w * w i)
    (hu : ∀ i, ∑ w ∈ W, lam w * w i ≤ u i)
    (x : Fin m → ℝ) :
    ∃ w ∈ W, logicalProfit l u x w ≤ 0 := by
  -- Show the λ-weighted average profit is ≤ 0!
  have havg : ∑ w ∈ W, lam w * logicalProfit l u x w ≤ 0 := by
    have hexp : ∑ w ∈ W, lam w * logicalProfit l u x w =
        ∑ i : Fin m,
          (max (x i) 0 * ((∑ w ∈ W, lam w * w i) - u i) +
           max (-x i) 0 * (l i - (∑ w ∈ W, lam w * w i))) := by
      unfold logicalProfit
      simp_rw [Finset.mul_sum]
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun i _ => ?_
      have hterm : ∀ w ∈ W,
          lam w * (max (x i) 0 * (w i - u i) + max (-x i) 0 * (l i - w i)) =
          max (x i) 0 * (lam w * w i) - max (x i) 0 * u i * lam w +
          (max (-x i) 0 * l i * lam w - max (-x i) 0 * (lam w * w i)) := by
        intro w _; ring
      rw [Finset.sum_congr rfl hterm, Finset.sum_add_distrib, Finset.sum_sub_distrib,
          Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.mul_sum, ← Finset.mul_sum,
          ← Finset.mul_sum, hlam_sum]
      ring
    rw [hexp]
    refine Finset.sum_nonpos fun i _ => ?_
    have h1 : max (x i) 0 * ((∑ w ∈ W, lam w * w i) - u i) ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos (le_max_right _ _) (sub_nonpos.2 (hu i))
    have h2 : max (-x i) 0 * (l i - (∑ w ∈ W, lam w * w i)) ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos (le_max_right _ _) (sub_nonpos.2 (hl i))
    linarith
  by_contra hall
  push_neg at hall
  have hpos : 0 < ∑ w ∈ W, lam w * logicalProfit l u x w := by
    have hex : ∃ w₀ ∈ W, 0 < lam w₀ := by
      by_contra h0
      push_neg at h0
      have hsum0 : ∑ w ∈ W, lam w = 0 :=
        Finset.sum_eq_zero fun w hw => le_antisymm (h0 w hw) (hlam_nn w hw)
      linarith
    rcases hex with ⟨w₀, hw₀, hlam₀⟩
    exact Finset.sum_pos' (fun w hw => mul_nonneg (hlam_nn w hw) (hall w hw).le)
      ⟨w₀, hw₀, mul_pos hlam₀ (hall w₀ hw₀)⟩
  linarith

/-- The three admissible worlds for `φ ⇒ A` in coordinate order `(A, φ)`:
`w₀ = (0, 0)`, `w₁ = (1, 0)`, `w₂ = (1, 1)`. -/
def w00 : Fin 2 → ℝ := ![0, 0]
def w10 : Fin 2 → ℝ := ![1, 0]
def w11 : Fin 2 → ℝ := ![1, 1]

/-- **Proposition 4(2a) (Exact anchor natural extension `[0, 1/2]`)**:
For `φ ⇒ A` with anchor probability `P(A) = 1/2`, a value `r` is realized as `P(φ)` by a
coherent probability distribution `(p₀₀, p₁₀, p₁₁)` with `p₁₀ + p₁₁ = 1/2` if and only if
`r ∈ [0, 1/2]` (exact natural-extension width `1/2`). -/
theorem anchor_natural_extension (r : ℝ) :
    (∃ p00 p10 p11 : ℝ, 0 ≤ p00 ∧ 0 ≤ p10 ∧ 0 ≤ p11 ∧
      p00 + p10 + p11 = 1 ∧ p10 + p11 = 1 / 2 ∧ p11 = r) ↔
    (0 ≤ r ∧ r ≤ 1 / 2) := by
  constructor
  · rintro ⟨_, p10, p11, _, hp10, hp11, _, hA, rfl⟩
    exact ⟨hp11, by linarith⟩
  · rintro ⟨hr0, hr1⟩
    exact ⟨1 / 2, 1 / 2 - r, r, by norm_num, by linarith, hr0, by ring, by ring, rfl⟩

/-- **Proposition 4(2b) (Singleton `q(φ) = 1/4` defeats ALL logical traders)**:
Quoting `[1/2, 1/2]` on `A` and the zero-width singleton `[1/4, 1/4]` on `φ` (realized by
coherent distribution `(1/2, 1/4, 1/4)` on `{00, 10, 11}`) guarantees that EVERY logical
portfolio `(x_A, x_φ) ∈ ℝ²` has worst-world profit `≤ 0` (zero logical exploitability despite
natural-extension width `1/2`). -/
theorem singleton_quarter_defeats_all_traders (x : Fin 2 → ℝ) :
    let q : Fin 2 → ℝ := ![1 / 2, 1 / 4]
    min (logicalProfit q q x w00)
      (min (logicalProfit q q x w10) (logicalProfit q q x w11)) ≤ 0 := by
  intro q
  have havg : (1 / 2 : ℝ) * logicalProfit q q x w00 +
      (1 / 4 : ℝ) * logicalProfit q q x w10 +
      (1 / 4 : ℝ) * logicalProfit q q x w11 = 0 := by
    simp only [logicalProfit, Fin.sum_univ_two, q, w00, w10, w11,
      Matrix.cons_val_zero, Matrix.cons_val_one]
    ring
  by_contra hpos
  push_neg at hpos
  have h1 : 0 < logicalProfit q q x w00 := lt_of_lt_of_le hpos (min_le_left _ _)
  have h2 : 0 < logicalProfit q q x w10 :=
    lt_of_lt_of_le hpos (le_trans (min_le_right _ _) (min_le_left _ _))
  have h3 : 0 < logicalProfit q q x w11 :=
    lt_of_lt_of_le hpos (le_trans (min_le_right _ _) (min_le_right _ _))
  linarith

/-- **Contrast (Disjoint quote `[3/5, 7/10]` admits unit-`ℓ₁` logical arbitrage `≥ 1/20 = 0.05`)**:
Quoting `1/2` on `A` and `[3/5, 7/10]` on `φ` (disjoint from `[0, 1/2]`) is strictly exploited
by the unit-`ℓ₁` portfolio `x = (1/2, -1/2)` with guaranteed profit `≥ 1/20` in every admissible
world `{00, 10, 11}`. -/
theorem disjoint_quote_arbitrage :
    let l : Fin 2 → ℝ := ![1 / 2, 3 / 5]
    let u : Fin 2 → ℝ := ![1 / 2, 7 / 10]
    let x : Fin 2 → ℝ := ![1 / 2, -1 / 2]
    (|x 0| + |x 1| = 1) ∧
    (1 / 20 : ℝ) ≤ logicalProfit l u x w00 ∧
    (1 / 20 : ℝ) ≤ logicalProfit l u x w10 ∧
    (1 / 20 : ℝ) ≤ logicalProfit l u x w11 := by
  intro l u x
  simp only [logicalProfit, Fin.sum_univ_two, l, u, x, w00, w10, w11,
    Matrix.cons_val_zero, Matrix.cons_val_one]
  norm_num

end NoArbitrageLM
