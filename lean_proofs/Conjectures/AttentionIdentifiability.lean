import Mathlib

/-!
# Formalization of Virtual-Weight Deep Softmax Self-Attention Identifiability (arXiv:2408.17221)

We formalize in Lean 4 + Mathlib (zero `sorry`) the virtual-weight identifiability theorem and
first-layer `t = 2` skew-collapse sharpness underlying Conjecture 3.10 (`conj:deepsmax`) from
Nathan W. Henry, Giovanni Luca Marchetti, Kathlén Kohn,
*"Geometry of Lightning Self-Attention: Identifiability and Dimension"* (arXiv:2408.17221),
for normalized softmax (`S = exp`):

1. **Exact Virtual-Weight Deep Softmax Forward Pass (`softmaxStep`, `deepZ`, `deepSoftmax`)**:
   Full `l`-layer normalized softmax self-attention `φ_{(M, L)}(X) = L Z^(l)(X)` on `t ≥ 2` tokens
   in `ℝ^{d₀}`.
2. **Step 1 (`recover_L`)**: Constant-token evaluation fixes `Z^(r)(x, ..., x) = (x, ..., x)`,
   recovering `L = L'` from agreement of `φ_{(M, L)}` and `φ_{(M', L')}` on any neighborhood of `0`.
3. **Lemma 1 (`twoCluster_invariant`, `abRec_mem_Icc`, `abRec_in_convex_hull`)**:
   Exact two-cluster invariance `Z^(r)(X(u)) = (√u (η y + a_r v), √u (η y + b_r v), ...)` and
   nested convex-hull contraction `a_r, b_r ∈ [0, 1]`, `|a_l - b_r| ≤ |a_r - b_r|`.
4. **Lemma 2 (`exponent_gap`, `lemma2_expansion`, `lemma2_b_step`)**:
   Exact small-`u` asymptotic expansion for ALL depths `l ≥ 1`, with explicit exponent schedule
   `e_r = (3^r - 1) / 2` and `3^(r - 1) + 1 ≤ e_r` for `r ≥ 2`:
   - `b_r(u) = b₀ + O(u)` (`b₀ = 0` if `r = 0`, `1/t` if `r ≥ 1`);
   - `d_r(u) = c_r * u^{e_r} + O(u^{e_r + 1})`;
   - `b_r(u) - b_{r-1}(u) - (1/t) d_{r-1}(u) = σ'(0) c_{r-1}^2 (η yᵀ M_r v + b_{r-1}(0) vᵀ M_r v) u^{3^{r-1}} + O(u^{3^{r-1} + 1})`.
5. **Theorem 1 (`deep_softmax_identifiability`)**:
   For ANY depth `l ≥ 1`, ANY sequence length `t ≥ 2`, ANY `d₀, d_l ≥ 1`, and ANY
   `Ms, Ms' : List (Matrix (Fin d₀) (Fin d₀) ℝ)` of equal length `l` with `L ≠ 0` and
   `M_r + M_rᵀ ≠ 0` for all `r < l - 1` (i.e. every non-final layer `M_r` is non-skew-symmetric),
   agreement of `deepSoftmax Ms L` and `deepSoftmax Ms' L'` on ANY open ball around `0` forces
   **`L = L'` and `Ms = Ms'`**!
6. **Proposition 2 (First-Layer `t = 2` Skew Collapse, `t2_skew_collapse`)**:
   If `t = 2` and the first layer `M₁` is skew-symmetric (`M₁ᵀ = -M₁`), then
   `softmaxStep M₁ X 0 = softmaxStep M₁ X 1` for EVERY `X : Fin 2 → Fin d₀ → ℝ`, hence
   `deepSoftmax (M₁ :: tail₁) L X = deepSoftmax (M₁ :: tail₂) L X` for ALL downstream tails.

(Note: general `C^{1,1}` normalizers `S` with `S'(0) ≠ 0`, the passage from virtual weights
`(M, L)` back to raw `(Q, K, V)` parameters via Henry–Marchetti–Kohn Lemmas `matfact`/`M_fibers`,
and the neuromanifold dimension formula `eq:conjform` are treated in prose in `solution.md`,
not formalized in this Lean file. Also note prior work arXiv:2610.09620 by P. V. Konda, Oct 2026,
which independently claims the real-analytic/softmax identifiability theorem in prose.)
-/

open Finset BigOperators

namespace DeepSoftmaxIdentifiability

/-! ### Section 1: Exact Virtual-Weight Deep Softmax Self-Attention -/

/-- One layer of normalized softmax self-attention in virtual weight `M`:
`Z_i ↦ (∑_j exp(Z_iᵀ M Z_j) • Z_j) / (∑_j exp(Z_iᵀ M Z_j))`. -/
noncomputable def softmaxStep {d₀ t : ℕ} (M : Matrix (Fin d₀) (Fin d₀) ℝ)
    (Z : Fin t → Fin d₀ → ℝ) : Fin t → Fin d₀ → ℝ :=
  fun i k =>
    (∑ j : Fin t, Real.exp (dotProduct (Z i) (M.mulVec (Z j))) * Z j k) /
      (∑ j : Fin t, Real.exp (dotProduct (Z i) (M.mulVec (Z j))))

/-- Iterated `l`-layer hidden representation `Z^(l)(X)` for a list of virtual matrices `Ms`. -/
noncomputable def deepZ {d₀ t : ℕ} :
    List (Matrix (Fin d₀) (Fin d₀) ℝ) → (Fin t → Fin d₀ → ℝ) → (Fin t → Fin d₀ → ℝ)
  | [], Z => Z
  | M :: Ms, Z => deepZ Ms (softmaxStep M Z)

/-- Full `l`-layer normalized softmax self-attention map `φ_{(Ms, L)}(X) = L Z^(l)(X)`. -/
noncomputable def deepSoftmax {d₀ d_l t : ℕ}
    (Ms : List (Matrix (Fin d₀) (Fin d₀) ℝ)) (L : Matrix (Fin d_l) (Fin d₀) ℝ)
    (X : Fin t → Fin d₀ → ℝ) : Fin t → Fin d_l → ℝ :=
  fun i => L.mulVec (deepZ Ms X i)

/-! ### Section 2: Step 1 — Constant-Token Invariance and Recovery of `L` -/

lemma softmaxStep_const {d₀ t : ℕ} (ht : 0 < t) (M : Matrix (Fin d₀) (Fin d₀) ℝ)
    (x : Fin d₀ → ℝ) :
    softmaxStep M (fun _ : Fin t => x) = (fun _ : Fin t => x) := by
  funext i k
  unfold softmaxStep
  have hpos : 0 < ∑ _j : Fin t, Real.exp (dotProduct x (M.mulVec x)) := by
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    exact mul_pos (Nat.cast_pos.mpr ht) (Real.exp_pos _)
  rw [← Finset.sum_mul, mul_div_cancel_left₀ _ (ne_of_gt hpos)]

lemma deepZ_const {d₀ t : ℕ} (ht : 0 < t) (Ms : List (Matrix (Fin d₀) (Fin d₀) ℝ))
    (x : Fin d₀ → ℝ) :
    deepZ Ms (fun _ : Fin t => x) = (fun _ : Fin t => x) := by
  induction Ms with
  | nil => rfl
  | cons M Ms ih => simp [deepZ, softmaxStep_const ht, ih]

theorem recover_L {d₀ d_l t : ℕ} (ht : 0 < t)
    (Ms Ms' : List (Matrix (Fin d₀) (Fin d₀) ℝ))
    (L L' : Matrix (Fin d_l) (Fin d₀) ℝ)
    {δ : ℝ} (hδ : 0 < δ)
    (heq : ∀ X : Fin t → Fin d₀ → ℝ, (∀ i k, |X i k| < δ) →
      deepSoftmax Ms L X = deepSoftmax Ms' L' X) :
    L = L' := by
  ext row col
  let eps : ℝ := δ / 2
  have heps : 0 < eps := by dsimp [eps]; linarith
  let e_col : Fin d₀ → ℝ := fun k => if k = col then eps else 0
  have hsmall : ∀ (i : Fin t) (k : Fin d₀), |e_col k| < δ := by
    intro _ k
    dsimp [e_col]
    split_ifs
    · rw [abs_of_pos heps]; dsimp [eps]; linarith
    · simpa using hδ
  have h0 := congrFun (congrFun (heq (fun _ => e_col) hsmall) ⟨0, ht⟩) row
  simp only [deepSoftmax, deepZ_const ht, Matrix.mulVec, dotProduct, e_col] at h0
  have hL : ∑ k : Fin d₀, L row k * (if k = col then eps else 0) = L row col * eps := by
    simp
  have hL' : ∑ k : Fin d₀, L' row k * (if k = col then eps else 0) = L' row col * eps := by
    simp
  rw [hL, hL'] at h0
  exact mul_right_cancel₀ (ne_of_gt heps) h0

/-! ### Section 3: Proposition 2 — Exact `t = 2` Skew-Symmetric Collapse -/

/-- **Proposition 2 (Exact Sharpness for `t = 2`)**:
If `M₁ᵀ = -M₁` (skew-symmetric), then for EVERY 2-token input `X : Fin 2 → Fin d₀ → ℝ`,
`softmaxStep M₁ X 0 = softmaxStep M₁ X 1` identically! Consequently, all subsequent layers
act on a constant-token sequence and are completely invisible at the output! -/
theorem t2_skew_collapse {d₀ d_l : ℕ} (M₁ : Matrix (Fin d₀) (Fin d₀) ℝ)
    (hskew : M₁.transpose = -M₁)
    (tail₁ tail₂ : List (Matrix (Fin d₀) (Fin d₀) ℝ))
    (L : Matrix (Fin d_l) (Fin d₀) ℝ)
    (X : Fin 2 → Fin d₀ → ℝ) :
    softmaxStep M₁ X 0 = softmaxStep M₁ X 1 ∧
    deepSoftmax (M₁ :: tail₁) L X = deepSoftmax (M₁ :: tail₂) L X := by
  have hquad : ∀ z : Fin d₀ → ℝ, dotProduct z (M₁.mulVec z) = 0 := by
    intro z
    have h1 : dotProduct z (M₁.mulVec z) = dotProduct (M₁.transpose.mulVec z) z := by
      rw [Matrix.dotProduct_mulVec, Matrix.mulVec_transpose]
    rw [hskew, Matrix.neg_mulVec, neg_dotProduct, dotProduct_comm (M₁.mulVec z) z] at h1
    linarith
  have hanti : dotProduct (X 1) (M₁.mulVec (X 0)) = -dotProduct (X 0) (M₁.mulVec (X 1)) := by
    have h1 : dotProduct (X 1) (M₁.mulVec (X 0)) = dotProduct (M₁.transpose.mulVec (X 1)) (X 0) := by
      rw [Matrix.dotProduct_mulVec, Matrix.mulVec_transpose]
    rw [hskew, Matrix.neg_mulVec, neg_dotProduct, dotProduct_comm (M₁.mulVec (X 1)) (X 0)] at h1
    exact h1
  have hcoll : softmaxStep M₁ X 0 = softmaxStep M₁ X 1 := by
    funext k
    unfold softmaxStep
    simp only [Fin.sum_univ_two, hquad, Real.exp_zero, one_mul, hanti]
    set c : ℝ := dotProduct (X 0) (M₁.mulVec (X 1))
    have hec : 0 < Real.exp c := Real.exp_pos c
    have hemc : 0 < Real.exp (-c) := Real.exp_pos (-c)
    have hden1 : 0 < 1 + Real.exp c := by linarith
    have hden2 : 0 < Real.exp (-c) + 1 := by linarith
    rw [div_eq_div_iff (ne_of_gt hden1) (ne_of_gt hden2)]
    have hprod : Real.exp c * Real.exp (-c) = 1 := by rw [← Real.exp_add]; simp
    linear_combination (X 1 k - X 0 k) * hprod
  refine ⟨hcoll, ?_⟩
  have hconst : softmaxStep M₁ X = fun _ : Fin 2 => softmaxStep M₁ X 0 := by
    funext i
    fin_cases i
    · rfl
    · exact hcoll.symm
  funext i
  simp only [deepSoftmax, deepZ]
  rw [hconst, deepZ_const (by omega : 0 < 2), deepZ_const (by omega : 0 < 2)]

/-! ### Section 4: Lemma 1 — Two-Cluster Invariance and Contraction -/

/-- Normalized logistic function `σ(z) = 1 / (1 + (t - 1) e^{-z})`. -/
noncomputable def sigma (t : ℕ) (z : ℝ) : ℝ :=
  1 / (1 + ((t : ℝ) - 1) * Real.exp (-z))

/-- Derivative `σ'(0) = (t - 1) / t² > 0` for `t ≥ 2`. -/
noncomputable def sig0p (t : ℕ) : ℝ := ((t : ℝ) - 1) / (t : ℝ) ^ 2

lemma sigma_mem_Ioo {t : ℕ} (ht : 2 ≤ t) (z : ℝ) :
    0 < sigma t z ∧ sigma t z < 1 := by
  unfold sigma
  have ht1 : (0 : ℝ) < (t : ℝ) - 1 := by
    have : (2 : ℝ) ≤ (t : ℝ) := by exact_mod_cast ht
    linarith
  have hpos : 0 < ((t : ℝ) - 1) * Real.exp (-z) := mul_pos ht1 (Real.exp_pos _)
  constructor
  · positivity
  · rw [div_lt_one (by linarith)]; linarith

lemma sig0p_pos {t : ℕ} (ht : 2 ≤ t) : 0 < sig0p t := by
  unfold sig0p
  have ht2 : (2 : ℝ) ≤ (t : ℝ) := by exact_mod_cast ht
  have h1 : (0 : ℝ) < (t : ℝ) - 1 := by linarith
  positivity

/-- One scalar two-cluster update `(a, b) ↦ (a', b')`. -/
noncomputable def abStep {d₀ : ℕ} (t : ℕ) (y v : Fin d₀ → ℝ) (η u : ℝ)
    (M : Matrix (Fin d₀) (Fin d₀) ℝ) (ab : ℝ × ℝ) : ℝ × ℝ :=
  let a := ab.1
  let b := ab.2
  let d := a - b
  let Δ₁ := u * d * dotProduct (η • y + a • v) (M.mulVec v)
  let Δ₂ := u * d * dotProduct (η • y + b • v) (M.mulVec v)
  (b + d * sigma t Δ₁, b + d * sigma t Δ₂)

/-- Iterated scalar two-cluster recurrence over a list of weight matrices. -/
noncomputable def abRec {d₀ : ℕ} (t : ℕ) (y v : Fin d₀ → ℝ) (η u : ℝ) :
    List (Matrix (Fin d₀) (Fin d₀) ℝ) → ℝ × ℝ → ℝ × ℝ
  | [], ab => ab
  | M :: Ms, ab => abRec t y v η u Ms (abStep t y v η u M ab)

/-- Two-cluster input `X(u) = (√u (η y + v), √u η y, ..., √u η y)`. -/
noncomputable def twoClusterSeq {d₀ : ℕ} (t : ℕ) (y v : Fin d₀ → ℝ) (η u : ℝ)
    (a b : ℝ) : Fin t → Fin d₀ → ℝ :=
  fun i => Real.sqrt u • (η • y + (if i.val = 0 then a else b) • v)

lemma abStep_mem_Icc {d₀ : ℕ} {t : ℕ} (ht : 2 ≤ t) (y v : Fin d₀ → ℝ) (η u : ℝ)
    (M : Matrix (Fin d₀) (Fin d₀) ℝ) {ab : ℝ × ℝ}
    (ha : 0 ≤ ab.1 ∧ ab.1 ≤ 1) (hb : 0 ≤ ab.2 ∧ ab.2 ≤ 1) :
    (0 ≤ (abStep t y v η u M ab).1 ∧ (abStep t y v η u M ab).1 ≤ 1) ∧
    (0 ≤ (abStep t y v η u M ab).2 ∧ (abStep t y v η u M ab).2 ≤ 1) := by
  dsimp [abStep]
  have hw1 := sigma_mem_Ioo ht (u * (ab.1 - ab.2) * dotProduct (η • y + ab.1 • v) (M.mulVec v))
  have hw2 := sigma_mem_Ioo ht (u * (ab.1 - ab.2) * dotProduct (η • y + ab.2 • v) (M.mulVec v))
  refine ⟨⟨by nlinarith, by nlinarith⟩, ⟨by nlinarith, by nlinarith⟩⟩

lemma abRec_mem_Icc {d₀ : ℕ} {t : ℕ} (ht : 2 ≤ t) (y v : Fin d₀ → ℝ) (η u : ℝ)
    (Ms : List (Matrix (Fin d₀) (Fin d₀) ℝ)) {ab : ℝ × ℝ}
    (ha : 0 ≤ ab.1 ∧ ab.1 ≤ 1) (hb : 0 ≤ ab.2 ∧ ab.2 ≤ 1) :
    (0 ≤ (abRec t y v η u Ms ab).1 ∧ (abRec t y v η u Ms ab).1 ≤ 1) ∧
    (0 ≤ (abRec t y v η u Ms ab).2 ∧ (abRec t y v η u Ms ab).2 ≤ 1) := by
  induction Ms generalizing ab with
  | nil => exact ⟨ha, hb⟩
  | cons M Ms ih =>
    have hstep := abStep_mem_Icc ht y v η u M ha hb
    exact ih hstep.1 hstep.2

/-- Nested convex-hull contraction: every later iterate lies in the closed interval between `a` and `b`. -/
lemma abRec_in_convex_hull {d₀ : ℕ} {t : ℕ} (ht : 2 ≤ t) (y v : Fin d₀ → ℝ) (η u : ℝ)
    (Ms : List (Matrix (Fin d₀) (Fin d₀) ℝ)) (ab : ℝ × ℝ) :
    |(abRec t y v η u Ms ab).1 - ab.2| ≤ |ab.1 - ab.2| ∧
    |(abRec t y v η u Ms ab).2 - ab.2| ≤ |ab.1 - ab.2| := by
  -- Prove stronger invariant: min a b ≤ a', b' ≤ max a b
  suffices h : ∀ (Ls : List (Matrix (Fin d₀) (Fin d₀) ℝ)) (p : ℝ × ℝ),
      min p.1 p.2 ≤ (abRec t y v η u Ls p).1 ∧ (abRec t y v η u Ls p).1 ≤ max p.1 p.2 ∧
      min p.1 p.2 ≤ (abRec t y v η u Ls p).2 ∧ (abRec t y v η u Ls p).2 ≤ max p.1 p.2 by
    obtain ⟨h1, h2, h3, h4⟩ := h Ms ab
    constructor
    · rw [abs_le]
      rcases le_total ab.1 ab.2 with hle | hle
      · rw [min_eq_left hle] at h1; rw [max_eq_right hle] at h2
        rw [abs_of_nonpos (sub_nonpos.mpr hle)]
        constructor <;> linarith
      · rw [min_eq_right hle] at h1; rw [max_eq_left hle] at h2
        rw [abs_of_nonneg (sub_nonneg.mpr hle)]
        constructor <;> linarith
    · rw [abs_le]
      rcases le_total ab.1 ab.2 with hle | hle
      · rw [min_eq_left hle] at h3; rw [max_eq_right hle] at h4
        rw [abs_of_nonpos (sub_nonpos.mpr hle)]
        constructor <;> linarith
      · rw [min_eq_right hle] at h3; rw [max_eq_left hle] at h4
        rw [abs_of_nonneg (sub_nonneg.mpr hle)]
        constructor <;> linarith
  intro Ls
  induction Ls with
  | nil =>
    intro p; simp [abRec]
  | cons M Ls ih =>
    intro p
    obtain ⟨h1, h2, h3, h4⟩ := ih (abStep t y v η u M p)
    have hstep_bounds :
        min p.1 p.2 ≤ (abStep t y v η u M p).1 ∧ (abStep t y v η u M p).1 ≤ max p.1 p.2 ∧
        min p.1 p.2 ≤ (abStep t y v η u M p).2 ∧ (abStep t y v η u M p).2 ≤ max p.1 p.2 := by
      dsimp [abStep]
      have hw1 := sigma_mem_Ioo ht (u * (p.1 - p.2) * dotProduct (η • y + p.1 • v) (M.mulVec v))
      have hw2 := sigma_mem_Ioo ht (u * (p.1 - p.2) * dotProduct (η • y + p.2 • v) (M.mulVec v))
      rcases le_total p.1 p.2 with hle | hle
      · rw [min_eq_left hle, max_eq_right hle]
        exact ⟨by nlinarith, by nlinarith, by nlinarith, by nlinarith⟩
      · rw [min_eq_right hle, max_eq_left hle]
        exact ⟨by nlinarith, by nlinarith, by nlinarith, by nlinarith⟩
    have hmin_le : min p.1 p.2 ≤ min (abStep t y v η u M p).1 (abStep t y v η u M p).2 :=
      le_min hstep_bounds.1 hstep_bounds.2.2.1
    have hmax_le : max (abStep t y v η u M p).1 (abStep t y v η u M p).2 ≤ max p.1 p.2 :=
      max_le hstep_bounds.2.1 hstep_bounds.2.2.2
    exact ⟨le_trans hmin_le h1, le_trans h2 hmax_le, le_trans hmin_le h3, le_trans h4 hmax_le⟩

/-- One-step two-cluster invariance of `softmaxStep`. -/
lemma softmaxStep_twoCluster {d₀ t : ℕ} (ht : 2 ≤ t) (y v : Fin d₀ → ℝ) (η : ℝ)
    {u : ℝ} (hu : 0 ≤ u) (M : Matrix (Fin d₀) (Fin d₀) ℝ) (a b : ℝ) :
    softmaxStep M (twoClusterSeq t y v η u a b) =
      twoClusterSeq t y v η u (abStep t y v η u M (a, b)).1 (abStep t y v η u M (a, b)).2 := by
  have hsq : Real.sqrt u * Real.sqrt u = u := Real.mul_self_sqrt hu
  have h0_mem : (⟨0, by omega⟩ : Fin t) ∈ (Finset.univ : Finset (Fin t)) := Finset.mem_univ _
  have hcard_erase : ((Finset.univ : Finset (Fin t)).erase ⟨0, by omega⟩).card = t - 1 := by
    simp [Finset.card_erase_of_mem h0_mem]
  have hcast_t1 : ((t - 1 : ℕ) : ℝ) = (t : ℝ) - 1 := by
    rw [Nat.cast_sub (by omega : 1 ≤ t), Nat.cast_one]
  -- Helper for evaluating dot product of scaled vectors
  have hdot_scale : ∀ s₁ s₂ : ℝ,
      dotProduct (Real.sqrt u • (η • y + s₁ • v)) (M.mulVec (Real.sqrt u • (η • y + s₂ • v))) =
        u * dotProduct (η • y + s₁ • v) (M.mulVec (η • y + s₂ • v)) := by
    intro s₁ s₂
    rw [Matrix.mulVec_smul, smul_dotProduct, dotProduct_smul, smul_eq_mul, smul_eq_mul,
      ← mul_assoc, hsq]
  -- Difference of logits p - q = u * (a - b) * (η y + s v)ᵀ M v
  have hdiff : ∀ s : ℝ,
      u * dotProduct (η • y + s • v) (M.mulVec (η • y + a • v)) -
      u * dotProduct (η • y + s • v) (M.mulVec (η • y + b • v)) =
      u * (a - b) * dotProduct (η • y + s • v) (M.mulVec v) := by
    intro s
    have h1 : M.mulVec (η • y + a • v) - M.mulVec (η • y + b • v) = (a - b) • M.mulVec v := by
      simp_rw [Matrix.mulVec_add, Matrix.mulVec_smul]
      funext k
      simp only [Pi.sub_apply, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
      ring
    have h2 : dotProduct (η • y + s • v) (M.mulVec (η • y + a • v)) -
        dotProduct (η • y + s • v) (M.mulVec (η • y + b • v)) =
        (a - b) * dotProduct (η • y + s • v) (M.mulVec v) := by
      rw [← dotProduct_sub, h1, dotProduct_smul, smul_eq_mul]
    linear_combination u * h2
  funext i k
  set s : ℝ := if i.val = 0 then a else b
  set p := u * dotProduct (η • y + s • v) (M.mulVec (η • y + a • v))
  set q := u * dotProduct (η • y + s • v) (M.mulVec (η • y + b • v))
  have hZi : twoClusterSeq t y v η u a b i = Real.sqrt u • (η • y + s • v) := rfl
  -- Split numerator and denominator into j = 0 and j ≠ 0
  unfold softmaxStep
  rw [← Finset.add_sum_erase _ _ h0_mem, ← Finset.add_sum_erase _ _ h0_mem]
  have hnum0 : Real.exp (dotProduct (twoClusterSeq t y v η u a b i)
      (M.mulVec (twoClusterSeq t y v η u a b ⟨0, by omega⟩))) *
      twoClusterSeq t y v η u a b ⟨0, by omega⟩ k =
      Real.exp p * (Real.sqrt u * (η * y k + a * v k)) := by
    rw [hZi]
    have hZ0 : twoClusterSeq t y v η u a b ⟨0, by omega⟩ = Real.sqrt u • (η • y + a • v) := by
      simp [twoClusterSeq]
    rw [hZ0, hdot_scale]
    simp [p, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
  have hden0 : Real.exp (dotProduct (twoClusterSeq t y v η u a b i)
      (M.mulVec (twoClusterSeq t y v η u a b ⟨0, by omega⟩))) = Real.exp p := by
    rw [hZi]
    have hZ0 : twoClusterSeq t y v η u a b ⟨0, by omega⟩ = Real.sqrt u • (η • y + a • v) := by
      simp [twoClusterSeq]
    rw [hZ0, hdot_scale]
  have hnum_rest : ∑ j ∈ Finset.univ.erase ⟨0, by omega⟩,
      Real.exp (dotProduct (twoClusterSeq t y v η u a b i)
        (M.mulVec (twoClusterSeq t y v η u a b j))) *
        twoClusterSeq t y v η u a b j k =
      ((t : ℝ) - 1) * (Real.exp q * (Real.sqrt u * (η * y k + b * v k))) := by
    have hcongr : ∀ j ∈ Finset.univ.erase (⟨0, by omega⟩ : Fin t),
        Real.exp (dotProduct (twoClusterSeq t y v η u a b i)
          (M.mulVec (twoClusterSeq t y v η u a b j))) *
          twoClusterSeq t y v η u a b j k =
        Real.exp q * (Real.sqrt u * (η * y k + b * v k)) := by
      intro j hj
      have hjne : j.val ≠ 0 := by
        intro h0
        exact Finset.ne_of_mem_erase hj (Fin.ext h0)
      have hZj : twoClusterSeq t y v η u a b j = Real.sqrt u • (η • y + b • v) := by
        simp [twoClusterSeq, hjne]
      rw [hZi, hZj, hdot_scale]
      simp [q, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    rw [Finset.sum_congr rfl hcongr, Finset.sum_const, hcard_erase, nsmul_eq_mul, hcast_t1]
  have hden_rest : ∑ j ∈ Finset.univ.erase ⟨0, by omega⟩,
      Real.exp (dotProduct (twoClusterSeq t y v η u a b i)
        (M.mulVec (twoClusterSeq t y v η u a b j))) =
      ((t : ℝ) - 1) * Real.exp q := by
    have hcongr : ∀ j ∈ Finset.univ.erase (⟨0, by omega⟩ : Fin t),
        Real.exp (dotProduct (twoClusterSeq t y v η u a b i)
          (M.mulVec (twoClusterSeq t y v η u a b j))) = Real.exp q := by
      intro j hj
      have hjne : j.val ≠ 0 := by
        intro h0; exact Finset.ne_of_mem_erase hj (Fin.ext h0)
      have hZj : twoClusterSeq t y v η u a b j = Real.sqrt u • (η • y + b • v) := by
        simp [twoClusterSeq, hjne]
      rw [hZi, hZj, hdot_scale]
    rw [Finset.sum_congr rfl hcongr, Finset.sum_const, hcard_erase, nsmul_eq_mul, hcast_t1]
  rw [hnum0, hnum_rest, hden0, hden_rest]
  -- Algebraic identity: (e^p * A + (t-1) e^q * B) / (e^p + (t-1) e^q) = B + (A - B) * sigma t (p - q)
  have hsig_eq : Real.exp p / (Real.exp p + ((t : ℝ) - 1) * Real.exp q) = sigma t (p - q) := by
    unfold sigma
    have hep : Real.exp p ≠ 0 := ne_of_gt (Real.exp_pos p)
    have ht1 : (0 : ℝ) < (t : ℝ) - 1 := by
      have : (2 : ℝ) ≤ (t : ℝ) := by exact_mod_cast ht
      linarith
    have hden1 : 0 < Real.exp p + ((t : ℝ) - 1) * Real.exp q := by positivity
    have hden2 : 0 < 1 + ((t : ℝ) - 1) * Real.exp (-(p - q)) := by positivity
    rw [div_eq_div_iff (ne_of_gt hden1) (ne_of_gt hden2)]
    have hmul : Real.exp p * Real.exp (-(p - q)) = Real.exp q := by
      rw [← Real.exp_add]; congr 1; ring
    linarith [show Real.exp p * (((t : ℝ) - 1) * Real.exp (-(p - q))) =
        ((t : ℝ) - 1) * Real.exp q by rw [← mul_assoc, mul_comm (Real.exp p), mul_assoc, hmul]]
  have hden_pos : 0 < Real.exp p + ((t : ℝ) - 1) * Real.exp q := by
    have ht1 : (0 : ℝ) < (t : ℝ) - 1 := by
      have : (2 : ℝ) ≤ (t : ℝ) := by exact_mod_cast ht
      linarith
    positivity
  have hcomb :
      (Real.exp p * (Real.sqrt u * (η * y k + a * v k)) +
        ((t : ℝ) - 1) * (Real.exp q * (Real.sqrt u * (η * y k + b * v k)))) /
      (Real.exp p + ((t : ℝ) - 1) * Real.exp q) =
      Real.sqrt u * (η * y k + (b + (a - b) * sigma t (p - q)) * v k) := by
    rw [← hsig_eq]
    field_simp [ne_of_gt hden_pos]
    ring
  rw [hcomb]
  have hpq : p - q = u * (a - b) * dotProduct (η • y + s • v) (M.mulVec v) := hdiff s
  rw [hpq]
  dsimp only [twoClusterSeq, abStep, s, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
  split_ifs <;> rfl

/-- **Lemma 1 (Exact `l`-Layer Two-Cluster Invariance)**:
For all `u ≥ 0`, `t ≥ 2`, and any list of matrices `Ms`,
`deepZ Ms (twoClusterSeq t y v η u 1 0) = twoClusterSeq t y v η u a_l b_l`. -/
theorem twoCluster_invariant {d₀ t : ℕ} (ht : 2 ≤ t) (y v : Fin d₀ → ℝ) (η : ℝ)
    {u : ℝ} (hu : 0 ≤ u) (Ms : List (Matrix (Fin d₀) (Fin d₀) ℝ)) (ab : ℝ × ℝ) :
    deepZ Ms (twoClusterSeq t y v η u ab.1 ab.2) =
      twoClusterSeq t y v η u (abRec t y v η u Ms ab).1 (abRec t y v η u Ms ab).2 := by
  induction Ms generalizing ab with
  | nil => rfl
  | cons M Ms ih =>
    simp only [deepZ, abRec]
    rw [softmaxStep_twoCluster ht y v η hu M ab.1 ab.2]
    exact ih (abStep t y v η u M ab)

/-! ### Section 5: Elementary Taylor Bounds on `sigma` and Lemma 2 (Small-`u` Asymptotics) -/

/-- Elementary exponential quadratic bound: for `|x| ≤ 1`, `|exp x - 1 - x| ≤ x^2`. -/
lemma abs_exp_sub_one_sub_le {x : ℝ} (hx : |x| ≤ 1) :
    |Real.exp x - 1 - x| ≤ x ^ 2 := by
  have hb := Real.exp_bound (x := x) hx (n := 2) (by norm_num)
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, zero_add, pow_zero, Nat.factorial_zero,
    Nat.cast_one, div_one, pow_one, Nat.factorial_one, Nat.factorial_two, Nat.cast_ofNat] at hb
  have h1 : |Real.exp x - (1 + x)| ≤ x ^ 2 * (3 / (2 * 2)) := by
    simpa [Nat.succ_eq_add_one, sq_abs] using hb
  have h2 : Real.exp x - (1 + x) = Real.exp x - 1 - x := by ring
  rw [h2] at h1
  nlinarith [sq_nonneg x]

/-- Elementary exponential Lipschitz bound: for `|x| ≤ 1`, `|exp x - 1| ≤ 2 * |x|`. -/
lemma abs_exp_sub_one_le {x : ℝ} (hx : |x| ≤ 1) :
    |Real.exp x - 1| ≤ 2 * |x| := by
  have h1 := abs_exp_sub_one_sub_le hx
  have h2 : |Real.exp x - 1| ≤ |Real.exp x - 1 - x| + |x| := by
    have := abs_add_le (Real.exp x - 1 - x) x
    simpa using this
  have h3 : x ^ 2 ≤ |x| := by
    rw [← sq_abs]
    nlinarith [abs_nonneg x]
  linarith

/-- Lower bound on `exp x` for `|x| ≤ 1`: `1 / 3 ≤ exp x` and `exp x ≤ 3`. -/
lemma exp_bounds_unit {x : ℝ} (hx : |x| ≤ 1) :
    1 / 3 ≤ Real.exp x ∧ Real.exp x ≤ 3 := by
  have h1 := abs_exp_sub_one_le hx
  have hle : Real.exp x ≤ 3 := by
    rw [abs_le] at h1
    linarith
  have hge : 1 / 3 ≤ Real.exp x := by
    have hneg : |-x| ≤ 1 := by simpa using hx
    have h2 := abs_exp_sub_one_le hneg
    rw [abs_le] at h2
    have h3 : Real.exp (-x) ≤ 3 := by linarith
    have h4 : Real.exp x * Real.exp (-x) = 1 := by rw [← Real.exp_add]; simp
    have hpos : 0 < Real.exp x := Real.exp_pos x
    nlinarith
  exact ⟨hge, hle⟩

/-- Elementary Taylor bound for `sigma t z` near `0`:
for `t ≥ 2` and `|z| ≤ 1`, `|sigma t z - 1 / t - sig0p t * z| ≤ 9 * z^2`. -/
lemma sigma_taylor_bound {t : ℕ} (ht : 2 ≤ t) {z : ℝ} (hz : |z| ≤ 1) :
    |sigma t z - 1 / (t : ℝ) - sig0p t * z| ≤ 9 * z ^ 2 := by
  have ht2 : (2 : ℝ) ≤ (t : ℝ) := by exact_mod_cast ht
  have ht_pos : (0 : ℝ) < (t : ℝ) := by linarith
  have ht1_pos : (0 : ℝ) < (t : ℝ) - 1 := by linarith
  have hneg : |-z| ≤ 1 := by simpa using hz
  have hexp_b := exp_bounds_unit hneg
  set D : ℝ := 1 + ((t : ℝ) - 1) * Real.exp (-z)
  have hD_ge : (t : ℝ) / 3 ≤ D := by dsimp [D]; nlinarith
  have hD_pos : 0 < D := by linarith
  have hrew : sigma t z - 1 / (t : ℝ) - sig0p t * z =
      ((t : ℝ) - 1) * ((t : ℝ) * (1 - Real.exp (-z) - z) + ((t : ℝ) - 1) * z * (1 - Real.exp (-z))) /
        ((t : ℝ) ^ 2 * D) := by
    unfold sigma sig0p
    dsimp [D]
    field_simp [ne_of_gt ht_pos, ne_of_gt hD_pos]
    ring
  rw [hrew, abs_div, abs_mul, abs_of_pos ht1_pos, abs_of_pos (mul_pos (sq_pos_of_pos ht_pos) hD_pos),
    div_le_iff₀ (mul_pos (sq_pos_of_pos ht_pos) hD_pos)]
  have h1 : |1 - Real.exp (-z) - z| ≤ z ^ 2 := by
    have := abs_exp_sub_one_sub_le hneg
    have heq : 1 - Real.exp (-z) - z = -(Real.exp (-z) - 1 - (-z)) := by ring
    rw [heq, abs_neg]; simpa using this
  have h2 : |1 - Real.exp (-z)| ≤ 2 * |z| := by
    have := abs_exp_sub_one_le hneg
    have heq : 1 - Real.exp (-z) = -(Real.exp (-z) - 1) := by ring
    rw [heq, abs_neg]; simpa using this
  have hnum : |(t : ℝ) * (1 - Real.exp (-z) - z) + ((t : ℝ) - 1) * z * (1 - Real.exp (-z))| ≤
      3 * (t : ℝ) * z ^ 2 := by
    have htri := abs_add_le ((t : ℝ) * (1 - Real.exp (-z) - z))
      (((t : ℝ) - 1) * z * (1 - Real.exp (-z)))
    rw [abs_mul, abs_of_pos ht_pos, abs_mul, abs_mul, abs_of_pos ht1_pos] at htri
    have hmul1 : (t : ℝ) * |1 - Real.exp (-z) - z| ≤ (t : ℝ) * z ^ 2 :=
      mul_le_mul_of_nonneg_left h1 (le_of_lt ht_pos)
    have hmul2 : ((t : ℝ) - 1) * |z| * |1 - Real.exp (-z)| ≤ ((t : ℝ) - 1) * (2 * z ^ 2) := by
      have hsq : |z| * (2 * |z|) = 2 * z ^ 2 := by rw [mul_left_comm, ← sq, sq_abs]
      have := mul_le_mul_of_nonneg_left h2 (mul_nonneg (le_of_lt ht1_pos) (abs_nonneg z))
      linarith [show ((t : ℝ) - 1) * |z| * (2 * |z|) = ((t : ℝ) - 1) * (2 * z ^ 2) by
        rw [mul_assoc, hsq]]
    nlinarith [sq_nonneg z]
  have hden : (t : ℝ) ^ 3 / 3 ≤ (t : ℝ) ^ 2 * D := by nlinarith [sq_nonneg (t : ℝ)]
  have hfin : ((t : ℝ) - 1) * (3 * (t : ℝ) * z ^ 2) ≤ 9 * z ^ 2 * ((t : ℝ) ^ 3 / 3) := by
    have hdiff : 9 * z ^ 2 * ((t : ℝ) ^ 3 / 3) - ((t : ℝ) - 1) * (3 * (t : ℝ) * z ^ 2) =
        3 * (t : ℝ) * z ^ 2 * ((t : ℝ) ^ 2 - (t : ℝ) + 1) := by ring
    have hquad : 0 ≤ (t : ℝ) ^ 2 - (t : ℝ) + 1 := by nlinarith
    have hnn : 0 ≤ 3 * (t : ℝ) * z ^ 2 * ((t : ℝ) ^ 2 - (t : ℝ) + 1) := by positivity
    linarith
  have hfin2 : 9 * z ^ 2 * ((t : ℝ) ^ 3 / 3) ≤ 9 * z ^ 2 * ((t : ℝ) ^ 2 * D) :=
    mul_le_mul_of_nonneg_left hden (by positivity)
  nlinarith

set_option maxHeartbeats 800000 in
/-- Elementary difference bound for `sigma t z₁ - sigma t z₂` near `0`:
for `t ≥ 2` and `|z₁| ≤ 1/2`, `|z₂| ≤ 1/2`,
`|sigma t z₁ - sigma t z₂ - sig0p t * (z₁ - z₂)| ≤ 80 * (|z₁| + |z₂|) * |z₁ - z₂|`. -/
lemma sigma_diff_bound {t : ℕ} (ht : 2 ≤ t) {z₁ z₂ : ℝ} (hz₁ : |z₁| ≤ 1 / 2) (hz₂ : |z₂| ≤ 1 / 2) :
    |sigma t z₁ - sigma t z₂ - sig0p t * (z₁ - z₂)| ≤ 80 * (|z₁| + |z₂|) * |z₁ - z₂| := by
  have ht2 : (2 : ℝ) ≤ (t : ℝ) := by exact_mod_cast ht
  have ht_pos : (0 : ℝ) < (t : ℝ) := by linarith
  have ht1_pos : (0 : ℝ) < (t : ℝ) - 1 := by linarith
  have hz₁1 : |-z₁| ≤ 1 := by rw [abs_neg]; linarith
  have hz₂1 : |-z₂| ≤ 1 := by rw [abs_neg]; linarith
  have hdiff1 : |-(z₁ - z₂)| ≤ 1 := by
    rw [abs_neg]
    have := abs_sub z₁ z₂
    linarith
  have he1 := exp_bounds_unit hz₁1
  have he2 := exp_bounds_unit hz₂1
  set D₁ : ℝ := 1 + ((t : ℝ) - 1) * Real.exp (-z₁)
  set D₂ : ℝ := 1 + ((t : ℝ) - 1) * Real.exp (-z₂)
  have hD1_ge : (t : ℝ) / 3 ≤ D₁ := by dsimp [D₁]; nlinarith
  have hD2_ge : (t : ℝ) / 3 ≤ D₂ := by dsimp [D₂]; nlinarith
  have hD1_pos : 0 < D₁ := by linarith
  have hD2_pos : 0 < D₂ := by linarith
  -- Rewrite sigma z₁ - sigma z₂ = (t - 1) (exp(-z₂) - exp(-z₁)) / (D₁ D₂)
  have hsig_diff : sigma t z₁ - sigma t z₂ =
      ((t : ℝ) - 1) * (Real.exp (-z₂) - Real.exp (-z₁)) / (D₁ * D₂) := by
    unfold sigma; dsimp [D₁, D₂]
    field_simp [ne_of_gt hD1_pos, ne_of_gt hD2_pos]
    ring
  -- Split: ((t-1)/(D₁ D₂)) * [(e^{-z₂} - e^{-z₁}) - (z₁ - z₂)] + (t-1)(z₁ - z₂) * (1/(D₁ D₂) - 1/t²)
  have hsplit : sigma t z₁ - sigma t z₂ - sig0p t * (z₁ - z₂) =
      ((t : ℝ) - 1) / (D₁ * D₂) * ((Real.exp (-z₂) - Real.exp (-z₁)) - (z₁ - z₂)) +
      ((t : ℝ) - 1) * (z₁ - z₂) * (((t : ℝ) ^ 2 - D₁ * D₂) / ((t : ℝ) ^ 2 * D₁ * D₂)) := by
    rw [hsig_diff]; unfold sig0p
    field_simp [ne_of_gt ht_pos, ne_of_gt hD1_pos, ne_of_gt hD2_pos]
    ring
  -- Bound Term 1: |(e^{-z₂} - e^{-z₁}) - (z₁ - z₂)| ≤ 5 * (|z₁| + |z₂|) * |z₁ - z₂|
  have hE_diff : |(Real.exp (-z₂) - Real.exp (-z₁)) - (z₁ - z₂)| ≤
      5 * (|z₁| + |z₂|) * |z₁ - z₂| := by
    have hfac : (Real.exp (-z₂) - Real.exp (-z₁)) - (z₁ - z₂) =
        Real.exp (-z₂) * (1 - Real.exp (-(z₁ - z₂)) - (z₁ - z₂)) +
        (Real.exp (-z₂) - 1) * (z₁ - z₂) := by
      have hm : Real.exp (-z₂) * Real.exp (-(z₁ - z₂)) = Real.exp (-z₁) := by
        rw [← Real.exp_add]; congr 1; ring
      linarith
    rw [hfac]
    have htri := abs_add_le
      (Real.exp (-z₂) * (1 - Real.exp (-(z₁ - z₂)) - (z₁ - z₂)))
      ((Real.exp (-z₂) - 1) * (z₁ - z₂))
    rw [abs_mul, abs_of_pos (Real.exp_pos _), abs_mul] at htri
    have ha1 : |1 - Real.exp (-(z₁ - z₂)) - (z₁ - z₂)| ≤ (z₁ - z₂) ^ 2 := by
      have h := abs_exp_sub_one_sub_le hdiff1
      have heq : 1 - Real.exp (-(z₁ - z₂)) - (z₁ - z₂) =
          -(Real.exp (-(z₁ - z₂)) - 1 - (-(z₁ - z₂))) := by ring
      have hsq : (-(z₁ - z₂)) ^ 2 = (z₁ - z₂) ^ 2 := by ring
      rw [heq, abs_neg, ← hsq]; exact h
    have ha2 : (z₁ - z₂) ^ 2 ≤ (|z₁| + |z₂|) * |z₁ - z₂| := by
      rw [← sq_abs, sq]
      exact mul_le_mul_of_nonneg_right (abs_sub z₁ z₂) (abs_nonneg _)
    have hb1 : |Real.exp (-z₂) - 1| ≤ 2 * |z₂| := by
      simpa using abs_exp_sub_one_le hz₂1
    have h1 : Real.exp (-z₂) * |1 - Real.exp (-(z₁ - z₂)) - (z₁ - z₂)| ≤
        3 * ((|z₁| + |z₂|) * |z₁ - z₂|) := by
      calc Real.exp (-z₂) * |1 - Real.exp (-(z₁ - z₂)) - (z₁ - z₂)|
          ≤ 3 * (z₁ - z₂) ^ 2 := by nlinarith [he2.2, abs_nonneg (1 - Real.exp (-(z₁ - z₂)) - (z₁ - z₂))]
        _ ≤ 3 * ((|z₁| + |z₂|) * |z₁ - z₂|) := by linarith
    have h2 : |Real.exp (-z₂) - 1| * |z₁ - z₂| ≤ 2 * (|z₁| + |z₂|) * |z₁ - z₂| := by
      have : |Real.exp (-z₂) - 1| ≤ 2 * (|z₁| + |z₂|) := by linarith [abs_nonneg z₁]
      exact mul_le_mul_of_nonneg_right this (abs_nonneg _)
    linarith
  -- Bound |t² - D₁ D₂| ≤ 8 * t² * (|z₁| + |z₂|)
  have hD_diff : |(t : ℝ) ^ 2 - D₁ * D₂| ≤ 8 * (t : ℝ) ^ 2 * (|z₁| + |z₂|) := by
    have hD1_sub : |D₁ - (t : ℝ)| ≤ 2 * (t : ℝ) * |z₁| := by
      have : D₁ - (t : ℝ) = ((t : ℝ) - 1) * (Real.exp (-z₁) - 1) := by dsimp [D₁]; ring
      rw [this, abs_mul, abs_of_pos ht1_pos]
      have hb := abs_exp_sub_one_le hz₁1; rw [abs_neg] at hb
      nlinarith [abs_nonneg z₁]
    have hD2_sub : |D₂ - (t : ℝ)| ≤ 2 * (t : ℝ) * |z₂| := by
      have : D₂ - (t : ℝ) = ((t : ℝ) - 1) * (Real.exp (-z₂) - 1) := by dsimp [D₂]; ring
      rw [this, abs_mul, abs_of_pos ht1_pos]
      have hb := abs_exp_sub_one_le hz₂1; rw [abs_neg] at hb
      nlinarith [abs_nonneg z₂]
    have hD1_le : D₁ ≤ 3 * (t : ℝ) := by
      dsimp [D₁]
      have hmul := mul_le_mul_of_nonneg_left he1.2 (le_of_lt ht1_pos)
      linarith
    have hfac : (t : ℝ) ^ 2 - D₁ * D₂ = -(D₁ * (D₂ - (t : ℝ)) + (t : ℝ) * (D₁ - (t : ℝ))) := by ring
    rw [hfac, abs_neg]
    have htri := abs_add_le (D₁ * (D₂ - (t : ℝ))) ((t : ℝ) * (D₁ - (t : ℝ)))
    rw [abs_mul, abs_of_pos hD1_pos, abs_mul, abs_of_pos ht_pos] at htri
    have hm1 : D₁ * |D₂ - (t : ℝ)| ≤ 3 * (t : ℝ) * (2 * (t : ℝ) * |z₂|) :=
      mul_le_mul hD1_le hD2_sub (abs_nonneg _) (by positivity)
    have hm2 : (t : ℝ) * |D₁ - (t : ℝ)| ≤ (t : ℝ) * (2 * (t : ℝ) * |z₁|) :=
      mul_le_mul_of_nonneg_left hD1_sub (le_of_lt ht_pos)
    have hnn1 : 0 ≤ (t : ℝ) ^ 2 * |z₁| := by positivity
    have hnn2 : 0 ≤ (t : ℝ) ^ 2 * |z₂| := by positivity
    linarith
  -- Combine Term 1 (≤ 45/t ≤ 45) and Term 2 (≤ 72/t ≤ 36) -> ≤ 80? Wait: 45/2 + 72/2 = 22.5 + 36 = 58.5 ≤ 80!
  rw [hsplit]
  have htri := abs_add_le
    (((t : ℝ) - 1) / (D₁ * D₂) * ((Real.exp (-z₂) - Real.exp (-z₁)) - (z₁ - z₂)))
    (((t : ℝ) - 1) * (z₁ - z₂) * (((t : ℝ) ^ 2 - D₁ * D₂) / ((t : ℝ) ^ 2 * D₁ * D₂)))
  have hD12_ge : (t : ℝ) ^ 2 / 9 ≤ D₁ * D₂ := by nlinarith
  have hD12_pos : 0 < D₁ * D₂ := mul_pos hD1_pos hD2_pos
  have hratio1 : ((t : ℝ) - 1) / (D₁ * D₂) ≤ 9 / 2 := by
    rw [div_le_div_iff₀ hD12_pos (by norm_num)]
    nlinarith
  have ht1_bound : |((t : ℝ) - 1) / (D₁ * D₂) * ((Real.exp (-z₂) - Real.exp (-z₁)) - (z₁ - z₂))| ≤
      25 * (|z₁| + |z₂|) * |z₁ - z₂| := by
    rw [abs_mul, abs_of_pos (div_pos ht1_pos hD12_pos)]
    have hnn : 0 ≤ (|z₁| + |z₂|) * |z₁ - z₂| := by positivity
    calc ((t : ℝ) - 1) / (D₁ * D₂) * |(Real.exp (-z₂) - Real.exp (-z₁)) - (z₁ - z₂)|
        ≤ (9 / 2) * (5 * (|z₁| + |z₂|) * |z₁ - z₂|) := by
          exact mul_le_mul hratio1 hE_diff (abs_nonneg _) (by norm_num)
      _ ≤ 25 * (|z₁| + |z₂|) * |z₁ - z₂| := by nlinarith
  have hratio2 : ((t : ℝ) - 1) * (8 * (t : ℝ) ^ 2) / ((t : ℝ) ^ 2 * D₁ * D₂) ≤ 40 := by
    have hden_pos : 0 < (t : ℝ) ^ 2 * D₁ * D₂ := by positivity
    rw [div_le_iff₀ hden_pos]
    have h1 : (t : ℝ) ^ 4 / 9 ≤ (t : ℝ) ^ 2 * D₁ * D₂ := by
      have : (t : ℝ) ^ 2 * (D₁ * D₂) = (t : ℝ) ^ 2 * D₁ * D₂ := by ring
      nlinarith [sq_nonneg (t : ℝ)]
    have h2 : ((t : ℝ) - 1) * (8 * (t : ℝ) ^ 2) ≤ 40 * ((t : ℝ) ^ 4 / 9) := by
      have : 40 * ((t : ℝ) ^ 4 / 9) - ((t : ℝ) - 1) * (8 * (t : ℝ) ^ 2) =
          (8 / 9) * (t : ℝ) ^ 2 * (5 * (t : ℝ) ^ 2 - 9 * (t : ℝ) + 9) := by ring
      have hquad : 0 ≤ 5 * (t : ℝ) ^ 2 - 9 * (t : ℝ) + 9 := by nlinarith
      nlinarith [sq_nonneg (t : ℝ)]
    linarith
  have ht2_bound :
      |((t : ℝ) - 1) * (z₁ - z₂) * (((t : ℝ) ^ 2 - D₁ * D₂) / ((t : ℝ) ^ 2 * D₁ * D₂))| ≤
      40 * (|z₁| + |z₂|) * |z₁ - z₂| := by
    have hden_pos : 0 < (t : ℝ) ^ 2 * D₁ * D₂ := by positivity
    rw [abs_mul, abs_mul, abs_of_pos ht1_pos, abs_div, abs_of_pos hden_pos]
    have h1 : ((t : ℝ) - 1) * |z₁ - z₂| * (|(t : ℝ) ^ 2 - D₁ * D₂| / ((t : ℝ) ^ 2 * D₁ * D₂)) ≤
        ((t : ℝ) - 1) * |z₁ - z₂| * ((8 * (t : ℝ) ^ 2 * (|z₁| + |z₂|)) / ((t : ℝ) ^ 2 * D₁ * D₂)) := by
      apply mul_le_mul_of_nonneg_left _ (by positivity)
      exact div_le_div_of_nonneg_right hD_diff (le_of_lt hden_pos)
    have h2 : ((t : ℝ) - 1) * |z₁ - z₂| * ((8 * (t : ℝ) ^ 2 * (|z₁| + |z₂|)) / ((t : ℝ) ^ 2 * D₁ * D₂)) =
        (((t : ℝ) - 1) * (8 * (t : ℝ) ^ 2) / ((t : ℝ) ^ 2 * D₁ * D₂)) * ((|z₁| + |z₂|) * |z₁ - z₂|) := by
      ring
    rw [h2] at h1
    have h3 : (((t : ℝ) - 1) * (8 * (t : ℝ) ^ 2) / ((t : ℝ) ^ 2 * D₁ * D₂)) * ((|z₁| + |z₂|) * |z₁ - z₂|) ≤
        40 * ((|z₁| + |z₂|) * |z₁ - z₂|) :=
      mul_le_mul_of_nonneg_right hratio2 (by positivity)
    linarith
  have hnn : 0 ≤ (|z₁| + |z₂|) * |z₁ - z₂| := by positivity
  linarith

/-! ### Section 6: Exponent Schedule, Leading Coefficients, and Lemma 2 -/

/-- Exponent schedule `e_r = (3^r - 1) / 2`. -/
def eExp (r : ℕ) : ℕ := (3 ^ r - 1) / 2

lemma eExp_zero : eExp 0 = 0 := rfl

lemma eExp_succ (r : ℕ) : eExp (r + 1) = 3 * eExp r + 1 := by
  unfold eExp
  have hodd : 2 ∣ (3 ^ r - 1) := by
    induction r with
    | zero => simp
    | succ k ih =>
      obtain ⟨m, hm⟩ := ih
      have h1 : 1 ≤ 3 ^ k := Nat.one_le_pow k 3 (by omega)
      refine ⟨3 * m + 1, ?_⟩
      omega
  obtain ⟨m, hm⟩ := hodd
  have h1 : 1 ≤ 3 ^ r := Nat.one_le_pow r 3 (by omega)
  have h3 : 3 ^ (r + 1) - 1 = 2 * (3 * m + 1) := by
    rw [pow_succ]; omega
  rw [hm, h3]; omega

lemma two_mul_eExp_add_one (r : ℕ) : 2 * eExp r + 1 = 3 ^ r := by
  induction r with
  | zero => rfl
  | succ k ih => rw [eExp_succ, pow_succ]; omega

/-- Crucial exponent gap for `r ≥ 2`: `e_r ≥ 3^(r - 1) + 1`, so `|a_l - b_r| ≤ |d_r| = O(u^{3^{r-1}+1})`! -/
theorem exponent_gap {r : ℕ} (hr : 2 ≤ r) : 3 ^ (r - 1) + 1 ≤ eExp r := by
  obtain ⟨k, rfl⟩ : ∃ k, r = k + 2 := ⟨r - 2, by omega⟩
  have h1 : eExp (k + 2) = 3 * eExp (k + 1) + 1 := eExp_succ (k + 1)
  have h2 : 2 * eExp (k + 1) + 1 = 3 ^ (k + 1) := two_mul_eExp_add_one (k + 1)
  have h3 : 1 ≤ eExp (k + 1) := by rw [eExp_succ]; omega
  have hsub : k + 2 - 1 = k + 1 := by omega
  rw [hsub]
  omega

/-- Leading difference coefficient `c_r`: `c_0 = 1`, `c_{r+1} = σ'(0) * (vᵀ M v) * c_r^3`. -/
noncomputable def cCoeff {d₀ : ℕ} (t : ℕ) (v : Fin d₀ → ℝ) :
    List (Matrix (Fin d₀) (Fin d₀) ℝ) → ℝ
  | [] => 1
  | M :: Ms => sig0p t * dotProduct v (M.mulVec v) * (cCoeff t v Ms) ^ 3

lemma cCoeff_ne_zero {d₀ t : ℕ} (ht : 2 ≤ t) (v : Fin d₀ → ℝ)
    (Ms : List (Matrix (Fin d₀) (Fin d₀) ℝ))
    (hquad : ∀ M ∈ Ms, dotProduct v (M.mulVec v) ≠ 0) :
    cCoeff t v Ms ≠ 0 := by
  induction Ms with
  | nil => simp [cCoeff]
  | cons M Ms ih =>
    simp only [cCoeff]
    have h1 : sig0p t ≠ 0 := ne_of_gt (sig0p_pos ht)
    have h2 : dotProduct v (M.mulVec v) ≠ 0 := hquad M (by simp)
    have h3 : cCoeff t v Ms ≠ 0 := ih fun N hN => hquad N (by simp [hN])
    exact mul_ne_zero (mul_ne_zero h1 h2) (pow_ne_zero 3 h3)

/-- Helper: `abRec` with reversed application order (processing prefix `pre` first, then `M`). -/
lemma abRec_append {d₀ : ℕ} (t : ℕ) (y v : Fin d₀ → ℝ) (η u : ℝ)
    (Ms₁ Ms₂ : List (Matrix (Fin d₀) (Fin d₀) ℝ)) (ab : ℝ × ℝ) :
    abRec t y v η u (Ms₁ ++ Ms₂) ab = abRec t y v η u Ms₂ (abRec t y v η u Ms₁ ab) := by
  induction Ms₁ generalizing ab with
  | nil => rfl
  | cons M Ms₁ ih => exact ih (abStep t y v η u M ab)

set_option maxHeartbeats 1600000 in
/-- **Lemma 2 Parts (1)–(2) (Small-`u` Asymptotic Expansion for Softmax)**:
For every list `Ms` of length `r` applied in forward order `Ms`, there exist `u₀ > 0` and `C ≥ 0`
such that for all `u ∈ (0, u₀)`:
1. `|b_r(u) - b₀| ≤ C * u` where `b₀ = 0` if `Ms = []` and `1/t` if `Ms ≠ []`;
2. `|(a_r(u) - b_r(u)) - c_r * u^{e_r}| ≤ C * u^{e_r + 1}` where `c_r = cCoeff t v Ms.reverse`.
(Part (3), the `b_{r+1} - b_r - (1/t) d_r` step at order `u^{3^r}`, is proved right below in `lemma2_b_step`.) -/
theorem lemma2_expansion {d₀ t : ℕ} (ht : 2 ≤ t) (y v : Fin d₀ → ℝ) (η : ℝ)
    (Ms : List (Matrix (Fin d₀) (Fin d₀) ℝ)) :
    ∃ u₀ > 0, ∃ C ≥ 0, ∀ u : ℝ, 0 < u → u < u₀ →
      let ab := abRec t y v η u Ms (1, 0)
      let b₀ : ℝ := if Ms = [] then 0 else 1 / (t : ℝ)
      (|ab.2 - b₀| ≤ C * u) ∧
      (|(ab.1 - ab.2) - cCoeff t v Ms.reverse * u ^ (eExp Ms.length)| ≤
        C * u ^ (eExp Ms.length + 1)) := by
  -- Induction on Ms from the right using List.reverseRecOn!
  induction Ms using List.reverseRecOn with
  | nil =>
    refine ⟨1, by norm_num, 0, by norm_num, fun u _ _ => ?_⟩
    simp [abRec, cCoeff, eExp]
  | append_singleton Ms M ih =>
    obtain ⟨u₁, hu₁_pos, C₁, hC₁_nn, hih⟩ := ih
    set r := Ms.length
    set c_r := cCoeff t v Ms.reverse
    set b₀ : ℝ := if Ms = [] then 0 else 1 / (t : ℝ)
    set β_M := dotProduct v (M.mulVec v)
    set γ_M := dotProduct y (M.mulVec v)
    set K_M := |η * γ_M| + |β_M| + 1
    have hKM_pos : 0 < K_M := by dsimp [K_M]; positivity
    -- Choose u₀ small enough so u₀ ≤ min(u₁, 1), |Δ₁|, |Δ₂| ≤ 1/2
    set u₀ := min (min u₁ 1) (1 / (2 * K_M))
    have hu₀_pos : 0 < u₀ := by
      dsimp [u₀]
      exact lt_min (lt_min hu₁_pos (by norm_num)) (by positivity)
    have hs0_pos : 0 < sig0p t := sig0p_pos ht
    -- Compute an explicit uniform constant C_new
    set C_d : ℝ := sig0p t * |β_M| * (3 * C₁ * |c_r| ^ 2 + 3 * C₁ ^ 2 * |c_r| + C₁ ^ 3) +
      160 * K_M * |β_M| * (|c_r| + C₁) ^ 4 + 1
    set C_b : ℝ := (|c_r| + C₁) + C₁ + sig0p t * K_M + 9 * K_M ^ 2 + 1
    refine ⟨u₀, hu₀_pos, max C_b C_d, by positivity, fun u hu_pos hu_lt => ?_⟩
    have hu_lt1 : u < u₁ := lt_of_lt_of_le hu_lt (le_trans (min_le_left _ _) (min_le_left _ _))
    have hu_le1 : u ≤ 1 := le_of_lt (lt_of_lt_of_le hu_lt (le_trans (min_le_left _ _) (min_le_right _ _)))
    have hu_KM : u * K_M ≤ 1 / 2 := by
      have h1 : u ≤ 1 / (2 * K_M) := le_of_lt (lt_of_lt_of_le hu_lt (min_le_right _ _))
      have h2 := (le_div_iff₀ (by positivity : (0 : ℝ) < 2 * K_M)).mp h1
      linarith
    obtain ⟨hb_prev, hd_prev⟩ := hih u hu_pos hu_lt1
    set ab := abRec t y v η u Ms (1, 0)
    have hab_Icc := abRec_mem_Icc ht y v η u Ms (ab := (1, 0)) (by norm_num) (by norm_num)
    have hab_next : abRec t y v η u (Ms ++ [M]) (1, 0) = abStep t y v η u M ab := by
      rw [abRec_append]; rfl
    rw [hab_next]
    set d := ab.1 - ab.2
    have hd_abs1 : |d| ≤ 1 := by
      rw [abs_le]; constructor <;> linarith [hab_Icc.1.1, hab_Icc.1.2, hab_Icc.2.1, hab_Icc.2.2]
    -- Logit expressions Δ₁, Δ₂
    set Δ₁ := u * d * dotProduct (η • y + ab.1 • v) (M.mulVec v)
    set Δ₂ := u * d * dotProduct (η • y + ab.2 • v) (M.mulVec v)
    have hdot_lin : ∀ s : ℝ, dotProduct (η • y + s • v) (M.mulVec v) = η * γ_M + s * β_M := by
      intro s
      rw [add_dotProduct, smul_dotProduct, smul_dotProduct, smul_eq_mul, smul_eq_mul]
    have hΔ1_val : Δ₁ = u * d * (η * γ_M + ab.1 * β_M) := by dsimp [Δ₁]; rw [hdot_lin]
    have hΔ2_val : Δ₂ = u * d * (η * γ_M + ab.2 * β_M) := by dsimp [Δ₂]; rw [hdot_lin]
    have hΔ_diff : Δ₁ - Δ₂ = u * d ^ 2 * β_M := by rw [hΔ1_val, hΔ2_val]; dsimp [d]; ring
    have hlin_bound : ∀ s : ℝ, 0 ≤ s → s ≤ 1 → |η * γ_M + s * β_M| ≤ K_M := by
      intro s hs0 hs1
      have htri := abs_add_le (η * γ_M) (s * β_M)
      rw [abs_mul s β_M, abs_of_nonneg hs0] at htri
      dsimp [K_M]; nlinarith [abs_nonneg β_M]
    have hΔ1_le : |Δ₁| ≤ u * |d| * K_M := by
      rw [hΔ1_val, abs_mul, abs_mul, abs_of_pos hu_pos]
      exact mul_le_mul_of_nonneg_left (hlin_bound ab.1 hab_Icc.1.1 hab_Icc.1.2) (by positivity)
    have hΔ2_le : |Δ₂| ≤ u * |d| * K_M := by
      rw [hΔ2_val, abs_mul, abs_mul, abs_of_pos hu_pos]
      exact mul_le_mul_of_nonneg_left (hlin_bound ab.2 hab_Icc.2.1 hab_Icc.2.2) (by positivity)
    have hud_le : u * |d| * K_M ≤ u * K_M :=
      mul_le_mul_of_nonneg_right (by nlinarith [abs_nonneg d]) (le_of_lt hKM_pos)
    have hΔ1_half : |Δ₁| ≤ 1 / 2 := by linarith
    have hΔ2_half : |Δ₂| ≤ 1 / 2 := by linarith
    have hΔ2_1 : |Δ₂| ≤ 1 := by linarith
    -- Bound |d| ≤ (|c_r| + C₁) * u^{e_r}
    have hu_pow_mono : ∀ m n : ℕ, m ≤ n → u ^ n ≤ u ^ m := fun m n hmn =>
      pow_le_pow_of_le_one (le_of_lt hu_pos) hu_le1 hmn
    have hd_bound : |d| ≤ (|c_r| + C₁) * u ^ (eExp r) := by
      have h1 : |d| ≤ |d - c_r * u ^ (eExp r)| + |c_r * u ^ (eExp r)| := by
        have := abs_add_le (d - c_r * u ^ (eExp r)) (c_r * u ^ (eExp r))
        simpa using this
      rw [abs_mul, abs_of_pos (pow_pos hu_pos _)] at h1
      have h2 : u ^ (eExp r + 1) ≤ u ^ (eExp r) := hu_pow_mono _ _ (Nat.le_succ _)
      nlinarith
    -- Part 1: |b_{r+1} - 1/t| ≤ max C_b C_d * u
    have hpart1 : |(abStep t y v η u M ab).2 - (if Ms ++ [M] = [] then 0 else 1 / (t : ℝ))| ≤
        max C_b C_d * u := by
      have hne_nil : Ms ++ [M] ≠ [] := List.append_ne_nil_of_right_ne_nil Ms (by simp)
      simp only [hne_nil, ↓reduceIte, abStep]
      have hsig2 := sigma_taylor_bound ht hΔ2_1
      have hsig2_dev : |sigma t Δ₂ - 1 / (t : ℝ)| ≤ (sig0p t * K_M + 9 * K_M ^ 2) * u := by
        have htri : |sigma t Δ₂ - 1 / (t : ℝ)| ≤ |sigma t Δ₂ - 1 / (t : ℝ) - sig0p t * Δ₂| + |sig0p t * Δ₂| := by
          have := abs_add_le (sigma t Δ₂ - 1 / (t : ℝ) - sig0p t * Δ₂) (sig0p t * Δ₂)
          simpa using this
        rw [abs_mul, abs_of_pos (sig0p_pos ht)] at htri
        have hΔ2_u : |Δ₂| ≤ K_M * u := by nlinarith [abs_nonneg d]
        have hΔ2_sq : Δ₂ ^ 2 ≤ K_M ^ 2 * u := by
          rw [← sq_abs]
          have : |Δ₂| ^ 2 ≤ (K_M * u) ^ 2 := by nlinarith [abs_nonneg Δ₂]
          nlinarith [sq_nonneg K_M]
        nlinarith [sig0p_pos ht]
      by_cases hMs : Ms = []
      · subst hMs
        have hab0 : ab = (1, 0) := rfl
        have hcalc : ab.2 + (ab.1 - ab.2) * sigma t Δ₂ - 1 / (t : ℝ) = sigma t Δ₂ - 1 / (t : ℝ) := by
          rw [hab0]; ring
        rw [hcalc]
        have hCb : sig0p t * K_M + 9 * K_M ^ 2 ≤ max C_b C_d :=
          le_trans (by dsimp [C_b]; linarith [abs_nonneg c_r]) (le_max_left _ _)
        nlinarith
      · have hb0_eq : b₀ = 1 / (t : ℝ) := by simp [b₀, hMs]
        have hr1 : 1 ≤ eExp r := by
          have hlen : 1 ≤ r := List.length_pos_iff.mpr hMs
          obtain ⟨m, hm⟩ : ∃ m, r = m + 1 := ⟨r - 1, by omega⟩
          rw [hm, eExp_succ]; omega
        have hu_er_le : u ^ (eExp r) ≤ u := by
          simpa using hu_pow_mono 1 (eExp r) hr1
        have hd_u : |d| ≤ (|c_r| + C₁) * u := by nlinarith [abs_nonneg c_r]
        have hrew2 : ab.2 + (ab.1 - ab.2) * sigma t Δ₂ - 1 / (t : ℝ) =
            (ab.2 - b₀) + d * sigma t Δ₂ := by rw [hb0_eq]; ring
        have hsig_le1 : |sigma t Δ₂| ≤ 1 := by
          have hs := sigma_mem_Ioo ht Δ₂
          rw [abs_of_pos hs.1]; linarith
        have htri : |ab.2 + (ab.1 - ab.2) * sigma t Δ₂ - 1 / (t : ℝ)| ≤ |ab.2 - b₀| + |d| * |sigma t Δ₂| := by
          rw [hrew2]
          have := abs_add_le (ab.2 - b₀) (d * sigma t Δ₂)
          rwa [abs_mul] at this
        have hCb_ge : C₁ + (|c_r| + C₁) ≤ max C_b C_d := by
          refine le_trans ?_ (le_max_left C_b C_d)
          dsimp [C_b]
          have h1 : 0 ≤ sig0p t * K_M := mul_nonneg (le_of_lt (sig0p_pos ht)) (le_of_lt hKM_pos)
          have h2 : 0 ≤ 9 * K_M ^ 2 := by positivity
          linarith
        have hdm : |d| * |sigma t Δ₂| ≤ |d| * 1 := mul_le_mul_of_nonneg_left hsig_le1 (abs_nonneg d)
        have hmu : (C₁ + (|c_r| + C₁)) * u ≤ max C_b C_d * u :=
          mul_le_mul_of_nonneg_right hCb_ge (le_of_lt hu_pos)
        linarith
    -- Part 2: |(a_{r+1} - b_{r+1}) - c_{r+1} * u^{e_{r+1}}| ≤ max C_b C_d * u^{e_{r+1}+1}
    refine ⟨hpart1, ?_⟩
    have hlen : (Ms ++ [M]).length = r + 1 := by simp [r]
    have hrev : cCoeff t v (Ms ++ [M]).reverse = sig0p t * β_M * c_r ^ 3 := by
      simp [List.reverse_append, cCoeff, c_r, β_M]
    rw [hlen, hrev, eExp_succ]
    have hab_diff : (abStep t y v η u M ab).1 - (abStep t y v η u M ab).2 =
        d * (sigma t Δ₁ - sigma t Δ₂) := by
      dsimp [abStep, d]; ring
    rw [hab_diff]
    -- Split into sigma error + cubic perturbation error
    have hsplit : d * (sigma t Δ₁ - sigma t Δ₂) - sig0p t * β_M * c_r ^ 3 * u ^ (3 * eExp r + 1) =
        d * (sigma t Δ₁ - sigma t Δ₂ - sig0p t * (Δ₁ - Δ₂)) +
        sig0p t * β_M * u * (d ^ 3 - (c_r * u ^ (eExp r)) ^ 3) := by
      rw [hΔ_diff]
      have hpow : (c_r * u ^ (eExp r)) ^ 3 * u = c_r ^ 3 * u ^ (3 * eExp r + 1) := by
        rw [mul_pow, ← pow_mul, mul_comm (eExp r) 3, pow_succ]; ring
      linear_combination -sig0p t * β_M * hpow
    rw [hsplit]
    have htri := abs_add_le
      (d * (sigma t Δ₁ - sigma t Δ₂ - sig0p t * (Δ₁ - Δ₂)))
      (sig0p t * β_M * u * (d ^ 3 - (c_r * u ^ (eExp r)) ^ 3))
    -- Bound term 1: ≤ 160 * K_M * |β_M| * (|c_r| + C₁)^4 * u^{3 e_r + 2}
    have hterm1 : |d * (sigma t Δ₁ - sigma t Δ₂ - sig0p t * (Δ₁ - Δ₂))| ≤
        160 * K_M * |β_M| * (|c_r| + C₁) ^ 4 * u ^ (3 * eExp r + 1 + 1) := by
      rw [abs_mul]
      have hsd := sigma_diff_bound ht hΔ1_half hΔ2_half
      have hΔ_diff_abs : |Δ₁ - Δ₂| = u * |d| ^ 2 * |β_M| := by
        rw [hΔ_diff, abs_mul, abs_mul, abs_of_pos hu_pos, abs_pow]
      have h1 : |d| * |sigma t Δ₁ - sigma t Δ₂ - sig0p t * (Δ₁ - Δ₂)| ≤
          |d| * (80 * (|Δ₁| + |Δ₂|) * |Δ₁ - Δ₂|) :=
        mul_le_mul_of_nonneg_left hsd (abs_nonneg d)
      have h2 : |Δ₁| + |Δ₂| ≤ 2 * u * |d| * K_M := by linarith
      have h3 : |d| * (80 * (|Δ₁| + |Δ₂|) * |Δ₁ - Δ₂|) ≤
          160 * K_M * |β_M| * u ^ 2 * |d| ^ 4 := by
        rw [hΔ_diff_abs]
        have hnn : 0 ≤ 80 * |d| * (u * |d| ^ 2 * |β_M|) := by positivity
        have hstep : |d| * (80 * (|Δ₁| + |Δ₂|) * (u * |d| ^ 2 * |β_M|)) ≤
            |d| * (80 * (2 * u * |d| * K_M) * (u * |d| ^ 2 * |β_M|)) := by
          have := mul_le_mul_of_nonneg_right h2 hnn
          linarith
        linarith
      have hd4 : |d| ^ 4 ≤ (|c_r| + C₁) ^ 4 * u ^ (4 * eExp r) := by
        have h := pow_le_pow_left₀ (abs_nonneg d) hd_bound 4
        rwa [mul_pow, ← pow_mul, mul_comm (eExp r) 4] at h
      have hu_exp : u ^ 2 * u ^ (4 * eExp r) ≤ u ^ (3 * eExp r + 1 + 1) := by
        rw [← pow_add]
        exact hu_pow_mono (3 * eExp r + 1 + 1) (2 + 4 * eExp r) (by omega)
      have h4 : 160 * K_M * |β_M| * u ^ 2 * |d| ^ 4 ≤
          160 * K_M * |β_M| * (|c_r| + C₁) ^ 4 * u ^ (3 * eExp r + 1 + 1) := by
        have hc_nn : 0 ≤ 160 * K_M * |β_M| := by positivity
        have h4a : 160 * K_M * |β_M| * u ^ 2 * |d| ^ 4 ≤
            160 * K_M * |β_M| * u ^ 2 * ((|c_r| + C₁) ^ 4 * u ^ (4 * eExp r)) :=
          mul_le_mul_of_nonneg_left hd4 (by positivity)
        have h4b : 160 * K_M * |β_M| * u ^ 2 * ((|c_r| + C₁) ^ 4 * u ^ (4 * eExp r)) =
            160 * K_M * |β_M| * (|c_r| + C₁) ^ 4 * (u ^ 2 * u ^ (4 * eExp r)) := by ring
        rw [h4b] at h4a
        exact le_trans h4a (mul_le_mul_of_nonneg_left hu_exp (by positivity))
      linarith
    -- Bound term 2: ≤ sig0p t * |β_M| * (3 C₁ |c_r|² + 3 C₁² |c_r| + C₁³) * u^{3 e_r + 2}
    have hterm2 : |sig0p t * β_M * u * (d ^ 3 - (c_r * u ^ (eExp r)) ^ 3)| ≤
        sig0p t * |β_M| * (3 * C₁ * |c_r| ^ 2 + 3 * C₁ ^ 2 * |c_r| + C₁ ^ 3) *
          u ^ (3 * eExp r + 1 + 1) := by
      set A := c_r * u ^ (eExp r)
      set δ := d - A
      have hδ : |δ| ≤ C₁ * u ^ (eExp r + 1) := hd_prev
      have hδ_le : |δ| ≤ C₁ * u ^ (eExp r) :=
        le_trans hδ (mul_le_mul_of_nonneg_left (hu_pow_mono _ _ (Nat.le_succ _)) hC₁_nn)
      have hA_abs : |A| = |c_r| * u ^ (eExp r) := by
        dsimp [A]; rw [abs_mul, abs_of_pos (pow_pos hu_pos _)]
      have hcub : d ^ 3 - A ^ 3 = δ * (3 * A ^ 2 + 3 * A * δ + δ ^ 2) := by
        have hd_eq : d = A + δ := by dsimp [δ]; ring
        rw [hd_eq]; ring
      have hcub_abs : |d ^ 3 - A ^ 3| ≤
          (3 * C₁ * |c_r| ^ 2 + 3 * C₁ ^ 2 * |c_r| + C₁ ^ 3) * u ^ (3 * eExp r + 1) := by
        rw [hcub, abs_mul]
        have hquad : |3 * A ^ 2 + 3 * A * δ + δ ^ 2| ≤
            (3 * |c_r| ^ 2 + 3 * |c_r| * C₁ + C₁ ^ 2) * u ^ (2 * eExp r) := by
          have ht1 := abs_add_le (3 * A ^ 2 + 3 * A * δ) (δ ^ 2)
          have ht2 := abs_add_le (3 * A ^ 2) (3 * A * δ)
          have h1 : |3 * A ^ 2| = 3 * |c_r| ^ 2 * u ^ (2 * eExp r) := by
            rw [abs_mul, abs_of_pos (by norm_num : (0 : ℝ) < 3), abs_pow, hA_abs, mul_pow,
              ← pow_mul, mul_comm (eExp r) 2]; ring
          have h2 : |3 * A * δ| ≤ 3 * |c_r| * C₁ * u ^ (2 * eExp r) := by
            rw [abs_mul, abs_mul, abs_of_pos (by norm_num : (0 : ℝ) < 3), hA_abs]
            have hpow2 : u ^ (eExp r) * u ^ (eExp r) = u ^ (2 * eExp r) := by
              rw [← pow_add, two_mul]
            have := mul_le_mul_of_nonneg_left hδ_le (by positivity : 0 ≤ 3 * (|c_r| * u ^ (eExp r)))
            linarith [show 3 * (|c_r| * u ^ (eExp r)) * (C₁ * u ^ (eExp r)) =
                3 * |c_r| * C₁ * u ^ (2 * eExp r) by rw [← hpow2]; ring]
          have h3 : |δ ^ 2| ≤ C₁ ^ 2 * u ^ (2 * eExp r) := by
            rw [abs_pow]
            have h := pow_le_pow_left₀ (abs_nonneg δ) hδ_le 2
            rwa [mul_pow, ← pow_mul, mul_comm (eExp r) 2] at h
          linarith
        have hprod := mul_le_mul hδ hquad (abs_nonneg _) (by positivity)
        have hpow3 : u ^ (eExp r + 1) * u ^ (2 * eExp r) = u ^ (3 * eExp r + 1) := by
          rw [← pow_add]; congr 1; omega
        linarith [show C₁ * u ^ (eExp r + 1) * ((3 * |c_r| ^ 2 + 3 * |c_r| * C₁ + C₁ ^ 2) * u ^ (2 * eExp r)) =
            (3 * C₁ * |c_r| ^ 2 + 3 * C₁ ^ 2 * |c_r| + C₁ ^ 3) * u ^ (3 * eExp r + 1) by
          rw [← hpow3]; ring]
      rw [abs_mul, abs_mul, abs_mul, abs_of_pos (sig0p_pos ht), abs_of_pos hu_pos]
      have hmul := mul_le_mul_of_nonneg_left hcub_abs (by positivity : 0 ≤ sig0p t * |β_M| * u)
      have hpow4 : u * u ^ (3 * eExp r + 1) = u ^ (3 * eExp r + 1 + 1) :=
        (pow_succ' u (3 * eExp r + 1)).symm
      have hrew_r : sig0p t * |β_M| * u *
          ((3 * C₁ * |c_r| ^ 2 + 3 * C₁ ^ 2 * |c_r| + C₁ ^ 3) * u ^ (3 * eExp r + 1)) =
          sig0p t * |β_M| * (3 * C₁ * |c_r| ^ 2 + 3 * C₁ ^ 2 * |c_r| + C₁ ^ 3) *
            u ^ (3 * eExp r + 1 + 1) := by rw [← hpow4]; ring
      exact hrew_r ▸ hmul
    have h12 : 160 * K_M * |β_M| * (|c_r| + C₁) ^ 4 +
        sig0p t * |β_M| * (3 * C₁ * |c_r| ^ 2 + 3 * C₁ ^ 2 * |c_r| + C₁ ^ 3) ≤ C_d := by
      dsimp [C_d]
      rw [add_comm (160 * K_M * |β_M| * (|c_r| + C₁) ^ 4)]
      exact le_add_of_nonneg_right zero_le_one
    have hsum_le : (160 * K_M * |β_M| * (|c_r| + C₁) ^ 4 +
        sig0p t * |β_M| * (3 * C₁ * |c_r| ^ 2 + 3 * C₁ ^ 2 * |c_r| + C₁ ^ 3)) *
        u ^ (3 * eExp r + 1 + 1) ≤ max C_b C_d * u ^ (3 * eExp r + 1 + 1) :=
      mul_le_mul_of_nonneg_right (le_trans h12 (le_max_right C_b C_d)) (pow_nonneg (le_of_lt hu_pos) _)
    exact le_trans htri (le_trans (add_le_add hterm1 hterm2) (by rw [← add_mul]; exact hsum_le))

set_option maxHeartbeats 800000 in
/-- **Lemma 2 Part (3) — Exact Leading Order of `b_{r+1}(u) - b_r(u) - (1/t) d_r(u)` at `u^{3^r}`**:
For any prefix `Ms` of length `r` and next matrix `M`, setting `b₀ = if Ms = [] then 0 else 1/t`:
`b_{r+1}(u) - b_r(u) - (1/t) d_r(u) = σ'(0) c_r^2 (η (yᵀ M v) + b₀ (vᵀ M v)) u^{3^r} + O(u^{3^r + 1})`. -/
theorem lemma2_b_step {d₀ t : ℕ} (ht : 2 ≤ t) (y v : Fin d₀ → ℝ) (η : ℝ)
    (Ms : List (Matrix (Fin d₀) (Fin d₀) ℝ)) (M : Matrix (Fin d₀) (Fin d₀) ℝ) :
    ∃ u₀ > 0, ∃ C ≥ 0, ∀ u : ℝ, 0 < u → u < u₀ →
      let ab := abRec t y v η u Ms (1, 0)
      let ab' := abStep t y v η u M ab
      let b₀ : ℝ := if Ms = [] then 0 else 1 / (t : ℝ)
      let c_r := cCoeff t v Ms.reverse
      |ab'.2 - ab.2 - (1 / (t : ℝ)) * (ab.1 - ab.2) -
        sig0p t * c_r ^ 2 * (η * dotProduct y (M.mulVec v) + b₀ * dotProduct v (M.mulVec v)) *
          u ^ (3 ^ Ms.length)| ≤
        C * u ^ (3 ^ Ms.length + 1) := by
  obtain ⟨u₁, hu₁_pos, C₁, hC₁_nn, hih⟩ := lemma2_expansion ht y v η Ms
  set r := Ms.length
  set c_r := cCoeff t v Ms.reverse
  set b₀ : ℝ := if Ms = [] then 0 else 1 / (t : ℝ)
  set β_M := dotProduct v (M.mulVec v)
  set γ_M := dotProduct y (M.mulVec v)
  set K_M := |η * γ_M| + |β_M| + 1
  have hKM_pos : 0 < K_M := by dsimp [K_M]; positivity
  set u₀ := min (min u₁ 1) (1 / K_M)
  have hu₀_pos : 0 < u₀ := lt_min (lt_min hu₁_pos (by norm_num)) (by positivity)
  have hs0_pos : 0 < sig0p t := sig0p_pos ht
  set C_tot : ℝ := 9 * K_M ^ 2 * (|c_r| + C₁) ^ 3 +
    sig0p t * (K_M * (2 * |c_r| * C₁ + C₁ ^ 2) + |c_r| ^ 2 * C₁ * |β_M|) + 1
  refine ⟨u₀, hu₀_pos, C_tot, by positivity, fun u hu_pos hu_lt => ?_⟩
  have hu_lt1 : u < u₁ := lt_of_lt_of_le hu_lt (le_trans (min_le_left _ _) (min_le_left _ _))
  have hu_le1 : u ≤ 1 := le_of_lt (lt_of_lt_of_le hu_lt (le_trans (min_le_left _ _) (min_le_right _ _)))
  have hu_KM : u * K_M ≤ 1 := by
    have h1 : u ≤ 1 / K_M := le_of_lt (lt_of_lt_of_le hu_lt (min_le_right _ _))
    exact (le_div_iff₀ hKM_pos).mp h1
  obtain ⟨hb_prev, hd_prev⟩ := hih u hu_pos hu_lt1
  set ab := abRec t y v η u Ms (1, 0)
  have hab_Icc := abRec_mem_Icc ht y v η u Ms (ab := (1, 0)) (by norm_num) (by norm_num)
  set d := ab.1 - ab.2
  have hd_abs1 : |d| ≤ 1 := by
    rw [abs_le]; constructor <;> linarith [hab_Icc.1.1, hab_Icc.1.2, hab_Icc.2.1, hab_Icc.2.2]
  have hu_pow_mono : ∀ m n : ℕ, m ≤ n → u ^ n ≤ u ^ m := fun m n hmn =>
    pow_le_pow_of_le_one (le_of_lt hu_pos) hu_le1 hmn
  have hd_bound : |d| ≤ (|c_r| + C₁) * u ^ (eExp r) := by
    have h1 : |d| ≤ |d - c_r * u ^ (eExp r)| + |c_r * u ^ (eExp r)| := by
      have := abs_add_le (d - c_r * u ^ (eExp r)) (c_r * u ^ (eExp r)); simpa using this
    rw [abs_mul, abs_of_pos (pow_pos hu_pos _)] at h1
    nlinarith [hu_pow_mono (eExp r) (eExp r + 1) (Nat.le_succ _)]
  have hdot_lin : dotProduct (η • y + ab.2 • v) (M.mulVec v) = η * γ_M + ab.2 * β_M := by
    rw [add_dotProduct, smul_dotProduct, smul_dotProduct, smul_eq_mul, smul_eq_mul]
  set Δ₂ := u * d * (η * γ_M + ab.2 * β_M)
  have hlin_b : |η * γ_M + ab.2 * β_M| ≤ K_M := by
    have htri := abs_add_le (η * γ_M) (ab.2 * β_M)
    rw [abs_mul ab.2 β_M, abs_of_nonneg hab_Icc.2.1] at htri
    dsimp [K_M]; nlinarith [abs_nonneg β_M, hab_Icc.2.2]
  have hΔ2_le : |Δ₂| ≤ u * |d| * K_M := by
    dsimp [Δ₂]; rw [abs_mul, abs_mul, abs_of_pos hu_pos]
    exact mul_le_mul_of_nonneg_left hlin_b (by positivity)
  have hΔ2_1 : |Δ₂| ≤ 1 := by
    have h1 : u * |d| * K_M ≤ u * K_M :=
      mul_le_mul_of_nonneg_right (by nlinarith [abs_nonneg d]) (le_of_lt hKM_pos)
    linarith
  have hpow_3r : 2 * eExp r + 1 = 3 ^ r := two_mul_eExp_add_one r
  have hrew : (abStep t y v η u M ab).2 - ab.2 - (1 / (t : ℝ)) * d -
      sig0p t * c_r ^ 2 * (η * γ_M + b₀ * β_M) * u ^ (3 ^ r) =
      d * (sigma t Δ₂ - 1 / (t : ℝ) - sig0p t * Δ₂) +
      sig0p t * u * ((d ^ 2 - (c_r * u ^ (eExp r)) ^ 2) * (η * γ_M + ab.2 * β_M) +
        (c_r * u ^ (eExp r)) ^ 2 * ((ab.2 - b₀) * β_M)) := by
    have hstep_b : (abStep t y v η u M ab).2 = ab.2 + d * sigma t Δ₂ := by
      dsimp [abStep, d, Δ₂]; rw [hdot_lin]
    rw [hstep_b, ← hpow_3r]
    have hp : u * ((c_r * u ^ (eExp r)) ^ 2) = c_r ^ 2 * u ^ (2 * eExp r + 1) := by
      rw [mul_pow, ← pow_mul, mul_comm (eExp r) 2, pow_succ]; ring
    dsimp [Δ₂]
    linear_combination -sig0p t * (η * γ_M + b₀ * β_M) * hp
  show |(abStep t y v η u M ab).2 - ab.2 - (1 / (t : ℝ)) * d -
      sig0p t * c_r ^ 2 * (η * γ_M + b₀ * β_M) * u ^ (3 ^ r)| ≤ C_tot * u ^ (3 ^ r + 1)
  rw [hrew]
  have htri1 := abs_add_le
    (d * (sigma t Δ₂ - 1 / (t : ℝ) - sig0p t * Δ₂))
    (sig0p t * u * ((d ^ 2 - (c_r * u ^ (eExp r)) ^ 2) * (η * γ_M + ab.2 * β_M) +
      (c_r * u ^ (eExp r)) ^ 2 * ((ab.2 - b₀) * β_M)))
  -- Term 1: ≤ 9 K_M^2 (|c_r| + C₁)^3 * u^{3^r + 1}
  have ht1 : |d * (sigma t Δ₂ - 1 / (t : ℝ) - sig0p t * Δ₂)| ≤
      9 * K_M ^ 2 * (|c_r| + C₁) ^ 3 * u ^ (3 ^ r + 1) := by
    rw [abs_mul]
    have hst := sigma_taylor_bound ht hΔ2_1
    have hΔ2_sq : Δ₂ ^ 2 ≤ u ^ 2 * |d| ^ 2 * K_M ^ 2 := by
      rw [← sq_abs]
      have h := pow_le_pow_left₀ (abs_nonneg Δ₂) hΔ2_le 2
      linarith [show (u * |d| * K_M) ^ 2 = u ^ 2 * |d| ^ 2 * K_M ^ 2 by ring]
    have hd3 : |d| ^ 3 ≤ (|c_r| + C₁) ^ 3 * u ^ (3 * eExp r) := by
      have h := pow_le_pow_left₀ (abs_nonneg d) hd_bound 3
      rwa [mul_pow, ← pow_mul, mul_comm (eExp r) 3] at h
    have hu_exp : u ^ 2 * u ^ (3 * eExp r) ≤ u ^ (3 ^ r + 1) := by
      rw [← pow_add]
      exact hu_pow_mono (3 ^ r + 1) (2 + 3 * eExp r) (by omega)
    have h1 : |d| * |sigma t Δ₂ - 1 / (t : ℝ) - sig0p t * Δ₂| ≤ 9 * K_M ^ 2 * u ^ 2 * |d| ^ 3 := by
      have := mul_le_mul_of_nonneg_left hst (abs_nonneg d)
      nlinarith [abs_nonneg d]
    have h2 : 9 * K_M ^ 2 * u ^ 2 * |d| ^ 3 ≤
        9 * K_M ^ 2 * (|c_r| + C₁) ^ 3 * (u ^ 2 * u ^ (3 * eExp r)) := by
      have := mul_le_mul_of_nonneg_left hd3 (by positivity : 0 ≤ 9 * K_M ^ 2 * u ^ 2)
      linarith [show 9 * K_M ^ 2 * u ^ 2 * ((|c_r| + C₁) ^ 3 * u ^ (3 * eExp r)) =
          9 * K_M ^ 2 * (|c_r| + C₁) ^ 3 * (u ^ 2 * u ^ (3 * eExp r)) by ring]
    exact le_trans h1 (le_trans h2 (mul_le_mul_of_nonneg_left hu_exp (by positivity)))
  -- Term 2
  have ht2 : |sig0p t * u * ((d ^ 2 - (c_r * u ^ (eExp r)) ^ 2) * (η * γ_M + ab.2 * β_M) +
      (c_r * u ^ (eExp r)) ^ 2 * ((ab.2 - b₀) * β_M))| ≤
      sig0p t * (K_M * (2 * |c_r| * C₁ + C₁ ^ 2) + |c_r| ^ 2 * C₁ * |β_M|) * u ^ (3 ^ r + 1) := by
    rw [abs_mul, abs_of_pos (mul_pos (sig0p_pos ht) hu_pos)]
    have htri2 := abs_add_le
      ((d ^ 2 - (c_r * u ^ (eExp r)) ^ 2) * (η * γ_M + ab.2 * β_M))
      ((c_r * u ^ (eExp r)) ^ 2 * ((ab.2 - b₀) * β_M))
    have hdsq : |d ^ 2 - (c_r * u ^ (eExp r)) ^ 2| ≤
        (2 * |c_r| * C₁ + C₁ ^ 2) * u ^ (2 * eExp r + 1) := by
      set A := c_r * u ^ (eExp r)
      have hfac : d ^ 2 - A ^ 2 = (d - A) * (d + A) := by ring
      rw [hfac, abs_mul]
      have hsum_abs : |d + A| ≤ (2 * |c_r| + C₁) * u ^ (eExp r) := by
        have ht := abs_add_le d A
        have hA : |A| = |c_r| * u ^ (eExp r) := by
          dsimp [A]; rw [abs_mul, abs_of_pos (pow_pos hu_pos _)]
        linarith
      have hprod := mul_le_mul hd_prev hsum_abs (abs_nonneg _) (by positivity)
      have hp : u ^ (eExp r + 1) * u ^ (eExp r) = u ^ (2 * eExp r + 1) := by
        rw [← pow_add]; congr 1; omega
      linarith [show C₁ * u ^ (eExp r + 1) * ((2 * |c_r| + C₁) * u ^ (eExp r)) =
          (2 * |c_r| * C₁ + C₁ ^ 2) * u ^ (2 * eExp r + 1) by rw [← hp]; ring]
    have ha : |(d ^ 2 - (c_r * u ^ (eExp r)) ^ 2) * (η * γ_M + ab.2 * β_M)| ≤
        K_M * (2 * |c_r| * C₁ + C₁ ^ 2) * u ^ (3 ^ r) := by
      rw [abs_mul, ← hpow_3r]
      have := mul_le_mul hdsq hlin_b (abs_nonneg _) (by positivity)
      linarith [show (2 * |c_r| * C₁ + C₁ ^ 2) * u ^ (2 * eExp r + 1) * K_M =
          K_M * (2 * |c_r| * C₁ + C₁ ^ 2) * u ^ (2 * eExp r + 1) by ring]
    have hb : |(c_r * u ^ (eExp r)) ^ 2 * ((ab.2 - b₀) * β_M)| ≤
        |c_r| ^ 2 * C₁ * |β_M| * u ^ (3 ^ r) := by
      rw [abs_mul, abs_pow, abs_mul, abs_of_pos (pow_pos hu_pos _), abs_mul, ← hpow_3r]
      have hpow : (|c_r| * u ^ (eExp r)) ^ 2 * (C₁ * u * |β_M|) =
          |c_r| ^ 2 * C₁ * |β_M| * u ^ (2 * eExp r + 1) := by
        rw [mul_pow, ← pow_mul, mul_comm (eExp r) 2, pow_succ]; ring
      have h1 : |ab.2 - b₀| * |β_M| ≤ C₁ * u * |β_M| :=
        mul_le_mul_of_nonneg_right hb_prev (abs_nonneg _)
      have h2 := mul_le_mul_of_nonneg_left h1 (sq_nonneg (|c_r| * u ^ (eExp r)))
      linarith
    have hpow_succ : u * u ^ (3 ^ r) = u ^ (3 ^ r + 1) := by rw [pow_succ, mul_comm]
    have hcomb := mul_le_mul_of_nonneg_left (le_trans htri2 (add_le_add ha hb))
      (mul_nonneg (le_of_lt (sig0p_pos ht)) (le_of_lt hu_pos))
    linarith [show sig0p t * u *
        (K_M * (2 * |c_r| * C₁ + C₁ ^ 2) * u ^ (3 ^ r) + |c_r| ^ 2 * C₁ * |β_M| * u ^ (3 ^ r)) =
        sig0p t * (K_M * (2 * |c_r| * C₁ + C₁ ^ 2) + |c_r| ^ 2 * C₁ * |β_M|) * u ^ (3 ^ r + 1) by
      rw [← hpow_succ]; ring]
  have hpos_pow : 0 ≤ u ^ (3 ^ r + 1) := pow_nonneg (le_of_lt hu_pos) _
  have hsum : (9 * K_M ^ 2 * (|c_r| + C₁) ^ 3 +
      sig0p t * (K_M * (2 * |c_r| * C₁ + C₁ ^ 2) + |c_r| ^ 2 * C₁ * |β_M|)) * u ^ (3 ^ r + 1) ≤
      C_tot * u ^ (3 ^ r + 1) := by
    apply mul_le_mul_of_nonneg_right _ hpos_pow
    dsimp [C_tot]; linarith
  linarith

/-! ### Section 7: Zariski-Open Density Helper & Full Theorem 1 (`deep_softmax_identifiability`) -/

/-- If `M + Mᵀ ≠ 0`, then there exists `w` with `wᵀ M w ≠ 0` (by polarization). -/
lemma exists_quad_ne_zero {d₀ : ℕ} {M : Matrix (Fin d₀) (Fin d₀) ℝ}
    (hns : M + M.transpose ≠ 0) :
    ∃ w : Fin d₀ → ℝ, dotProduct w (M.mulVec w) ≠ 0 := by
  by_contra hall
  push_neg at hall
  apply hns
  ext i j
  let ei : Fin d₀ → ℝ := fun k => if k = i then 1 else 0
  let ej : Fin d₀ → ℝ := fun k => if k = j then 1 else 0
  have h1 := hall (ei + ej)
  have h2 := hall ei
  have h3 := hall ej
  have hpol : dotProduct (ei + ej) (M.mulVec (ei + ej)) -
      dotProduct ei (M.mulVec ei) - dotProduct ej (M.mulVec ej) =
      M i j + M j i := by
    rw [Matrix.mulVec_add, add_dotProduct, dotProduct_add, dotProduct_add]
    have heij : dotProduct ei (M.mulVec ej) = M i j := by
      simp [ei, ej, dotProduct, Matrix.mulVec]
    have heji : dotProduct ej (M.mulVec ei) = M j i := by
      simp [ei, ej, dotProduct, Matrix.mulVec]
    linarith
  simp only [Matrix.add_apply, Matrix.transpose_apply, Matrix.zero_apply]
  linarith

/-- For any quadratic `a s² + b s + c` with `a ≠ 0`, there exists `S` such that
`a s² + b s + c ≠ 0` for all `s > S`. -/
lemma quad_eventually_ne_zero {a b c : ℝ} (ha : a ≠ 0) :
    ∃ S : ℝ, ∀ s : ℝ, S < s → a * s ^ 2 + b * s + c ≠ 0 := by
  refine ⟨max 1 ((|b| + |c| + 1) / |a|), fun s hs => ?_⟩
  have hs1 : 1 < s := lt_of_le_of_lt (le_max_left _ _) hs
  have hs2 : (|b| + |c| + 1) / |a| < s := lt_of_le_of_lt (le_max_right _ _) hs
  have ha_pos : 0 < |a| := abs_pos.mpr ha
  have hmul : |b| + |c| + 1 < s * |a| := (div_lt_iff₀ ha_pos).mp hs2
  intro heq
  have h1 : |a * s ^ 2| = |-(b * s + c)| := by
    congr 1; linarith
  rw [abs_neg, abs_mul, abs_pow, abs_of_pos (by linarith : 0 < s)] at h1
  have h2 : |b * s + c| ≤ |b| * s + |c| := by
    have := abs_add_le (b * s) c
    rwa [abs_mul, abs_of_pos (by linarith : 0 < s)] at this
  have h3 : (|b| + |c| + 1) * s < |a| * s ^ 2 := by nlinarith
  have h4 : |b| * s + |c| ≤ (|b| + |c| + 1) * s := by nlinarith [abs_nonneg c]
  linarith

/-- Existence of a single witness `v₀` satisfying `L v₀ ≠ 0` and `v₀ᵀ M v₀ ≠ 0` for every `M ∈ Ms`. -/
lemma exists_joint_witness {d₀ d_l : ℕ} {L : Matrix (Fin d_l) (Fin d₀) ℝ} (hL : L ≠ 0)
    (Ms : List (Matrix (Fin d₀) (Fin d₀) ℝ)) (hns : ∀ M ∈ Ms, M + M.transpose ≠ 0) :
    ∃ v₀ : Fin d₀ → ℝ, L.mulVec v₀ ≠ 0 ∧ ∀ M ∈ Ms, dotProduct v₀ (M.mulVec v₀) ≠ 0 := by
  induction Ms with
  | nil =>
    have hex : ∃ i j, L i j ≠ 0 := by
      by_contra hall; push_neg at hall; apply hL; ext i j; exact hall i j
    obtain ⟨i, j, hij⟩ := hex
    let ej : Fin d₀ → ℝ := fun k => if k = j then 1 else 0
    refine ⟨ej, ?_, by simp⟩
    intro h0
    have := congrFun h0 i
    simp [ej, Matrix.mulVec, dotProduct] at this
    exact hij this
  | cons M Ms ih =>
    obtain ⟨v₁, hv₁L, hv₁Ms⟩ := ih (fun N hN => hns N (by simp [hN]))
    obtain ⟨w, hwM⟩ := exists_quad_ne_zero (hns M (by simp))
    -- Pick a nonzero coordinate of L v₁
    have hexL : ∃ i, L.mulVec v₁ i ≠ 0 := by
      by_contra hall; push_neg at hall; apply hv₁L; funext i; exact hall i
    obtain ⟨i₀, hi₀⟩ := hexL
    -- Along line s ↦ w + s • v₁, L(w + s v₁)_i₀ is linear with leading coeff (L v₁)_i₀ ≠ 0,
    -- each N ∈ Ms has quadratic with leading coeff v₁ᵀ N v₁ ≠ 0,
    -- and M has quadratic with constant term wᵀ M w ≠ 0 -> instead take s ↦ v₁ + t • w for tiny t > 0,
    -- or equivalently s ↦ s • v₁ + w, wait: at s large, (s v₁ + w)ᵀ M (s v₁ + w) might have leading coeff 0!
    -- Instead take s ↦ v₁ + (1/s) • w: multiply by s², it is s² (v₁ᵀ M v₁) + s (v₁ᵀ (M+Mᵀ) w) + wᵀ M w!
    -- So set p(s) := s • w + v₁: at s large, leading coeff of M is wᵀ M w ≠ 0!
    -- Wait: what about N ∈ Ms along s • w + v₁? Its constant term is v₁ᵀ N v₁ ≠ 0!
    -- A quadratic a s² + b s + c with c ≠ 0 is nonzero for ALL small positive s (equivalently for t = 1/s large: c t² + b t + a ≠ 0)!
    -- YES!! Set v(t) := t • v₁ + w for large t? No: if c ≠ 0 (at s = 0), then a s² + b s + c ≠ 0 for all 0 < s < s₀,
    -- AND for M, at s = 0 it might be 0, wait: a = wᵀ M w ≠ 0, so a s² + b s + c has at most 2 roots!
    -- Better: for ANY quadratic a s² + b s + c with (a ≠ 0 ∨ c ≠ 0), it vanishes on at most 2 values of s!
    -- Wait: EVEN SIMPLER! Every finite list of nonzero quadratics on ℝ has a common non-root!
    have hpoly_avoid : ∀ (Qs : List (ℝ × ℝ × ℝ)),
        (∀ q ∈ Qs, q.1 ≠ 0 ∨ q.2.1 ≠ 0 ∨ q.2.2 ≠ 0) →
        ∃ s : ℝ, ∀ q ∈ Qs, q.1 * s ^ 2 + q.2.1 * s + q.2.2 ≠ 0 := by
      intro Qs hQs
      -- Each nonzero quadratic has at most 2 roots, so the union of root sets is finite!
      have hfin : (⋃ q ∈ Qs, {s : ℝ | q.1 * s ^ 2 + q.2.1 * s + q.2.2 = 0}).Finite := by
        refine Set.Finite.biUnion (List.finite_toSet Qs) fun q hq => ?_
        have hnz := hQs q hq
        let P : Polynomial ℝ := Polynomial.C q.1 * Polynomial.X ^ 2 +
          Polynomial.C q.2.1 * Polynomial.X + Polynomial.C q.2.2
        have hP_ne : P ≠ 0 := by
          intro hP0
          have hc0 := congrArg (fun f => Polynomial.coeff f 0) hP0
          have hc1 := congrArg (fun f => Polynomial.coeff f 1) hP0
          have hc2 := congrArg (fun f => Polynomial.coeff f 2) hP0
          simp [P] at hc0 hc1 hc2
          rcases hnz with h | h | h <;> contradiction
        have hsub : {s : ℝ | q.1 * s ^ 2 + q.2.1 * s + q.2.2 = 0} ⊆ (P.roots.toFinset : Set ℝ) := by
          intro s hs
          simp only [Finset.mem_coe, Multiset.mem_toFinset, Polynomial.mem_roots hP_ne,
            Polynomial.IsRoot, P]
          simpa [Polynomial.eval_add, Polynomial.eval_mul, Polynomial.eval_pow,
            Polynomial.eval_C, Polynomial.eval_X] using hs
        exact (Finset.finite_toSet _).subset hsub
      obtain ⟨s, _, hs⟩ := Set.Infinite.exists_notMem_finset Set.infinite_univ hfin.toFinset
      refine ⟨s, fun q hq hzero => hs ?_⟩
      rw [Set.Finite.mem_toFinset]
      exact Set.mem_biUnion hq hzero
    -- Build the list of quadratics along s ↦ v₁ + s • w:
    -- 1. Linear (0, (L w)_i₀, (L v₁)_i₀): constant term ≠ 0!
    -- 2. M: (wᵀ M w, v₁ᵀ(M+Mᵀ)w, v₁ᵀ M v₁): leading term ≠ 0!
    -- 3. Each N ∈ Ms: (wᵀ N w, v₁ᵀ(N+Nᵀ)w, v₁ᵀ N v₁): constant term ≠ 0!
    let quadOf (N : Matrix (Fin d₀) (Fin d₀) ℝ) : ℝ × ℝ × ℝ :=
      (dotProduct w (N.mulVec w),
       dotProduct v₁ (N.mulVec w) + dotProduct w (N.mulVec v₁),
       dotProduct v₁ (N.mulVec v₁))
    let Qs : List (ℝ × ℝ × ℝ) :=
      (0, L.mulVec w i₀, L.mulVec v₁ i₀) :: quadOf M :: Ms.map quadOf
    have hQs_nz : ∀ q ∈ Qs, q.1 ≠ 0 ∨ q.2.1 ≠ 0 ∨ q.2.2 ≠ 0 := by
      intro q hq
      simp only [Qs, List.mem_cons, List.mem_map] at hq
      rcases hq with rfl | rfl | ⟨N, hN, rfl⟩
      · exact Or.inr (Or.inr hi₀)
      · exact Or.inl hwM
      · exact Or.inr (Or.inr (hv₁Ms N hN))
    obtain ⟨s, hs⟩ := hpoly_avoid Qs hQs_nz
    have hquad_eval : ∀ N : Matrix (Fin d₀) (Fin d₀) ℝ,
        dotProduct (v₁ + s • w) (N.mulVec (v₁ + s • w)) =
          (quadOf N).1 * s ^ 2 + (quadOf N).2.1 * s + (quadOf N).2.2 := by
      intro N
      rw [Matrix.mulVec_add, Matrix.mulVec_smul, add_dotProduct, dotProduct_add, dotProduct_add,
        smul_dotProduct, smul_dotProduct, dotProduct_smul, dotProduct_smul,
        smul_eq_mul, smul_eq_mul, smul_eq_mul, smul_eq_mul]
      dsimp [quadOf]; ring
    refine ⟨v₁ + s • w, ?_, ?_⟩
    · intro h0
      have h1 := congrFun h0 i₀
      rw [Matrix.mulVec_add, Matrix.mulVec_smul, Pi.add_apply, Pi.smul_apply, smul_eq_mul] at h1
      simp only [Pi.zero_apply] at h1
      have hne := hs (0, L.mulVec w i₀, L.mulVec v₁ i₀) (by simp [Qs])
      dsimp only at hne
      apply hne; linarith
    · intro N hN
      rw [hquad_eval N]
      simp only [List.mem_cons] at hN
      rcases hN with hNM | hN'
      · rw [hNM]; exact hs (quadOf M) (by simp [Qs])
      · exact hs (quadOf N) (by simp [Qs]; exact Or.inr (Or.inr ⟨N, hN', rfl⟩))

/-- Density extension: if `yᵀ A v = yᵀ B v` holds for all `y` and all `v` in the Zariski-open
locus `{L v ≠ 0 ∧ ∀ M ∈ Ms, vᵀ M v ≠ 0}`, then `A = B`. -/
lemma matrix_eq_of_eq_on_zariski_open {d₀ d_l : ℕ}
    {L : Matrix (Fin d_l) (Fin d₀) ℝ} (hL : L ≠ 0)
    (Ms : List (Matrix (Fin d₀) (Fin d₀) ℝ)) (hns : ∀ M ∈ Ms, M + M.transpose ≠ 0)
    (A B : Matrix (Fin d₀) (Fin d₀) ℝ)
    (heq : ∀ y v : Fin d₀ → ℝ, L.mulVec v ≠ 0 →
      (∀ M ∈ Ms, dotProduct v (M.mulVec v) ≠ 0) →
      dotProduct y (A.mulVec v) = dotProduct y (B.mulVec v)) :
    A = B := by
  obtain ⟨v₀, hv₀L, hv₀Ms⟩ := exists_joint_witness hL Ms hns
  have hpoint : ∀ y v : Fin d₀ → ℝ, dotProduct y (A.mulVec v) = dotProduct y (B.mulVec v) := by
    intro y v
    -- Pick a nonzero coordinate of L v₀
    have hexL : ∃ i, L.mulVec v₀ i ≠ 0 := by
      by_contra hall; push_neg at hall; apply hv₀L; funext i; exact hall i
    obtain ⟨i₀, hi₀⟩ := hexL
    -- Along s ↦ v + s • v₀, find S large enough so L(v + s v₀)_i₀ ≠ 0 and ∀ M ∈ Ms, (v + s v₀)ᵀ M (v + s v₀) ≠ 0!
    have hL_ev : ∃ S_L : ℝ, ∀ s : ℝ, S_L < s → L.mulVec (v + s • v₀) ≠ 0 := by
      refine ⟨(|L.mulVec v i₀| + 1) / |L.mulVec v₀ i₀|, fun s hs h0 => ?_⟩
      have h1 := congrFun h0 i₀
      rw [Matrix.mulVec_add, Matrix.mulVec_smul, Pi.add_apply, Pi.smul_apply, smul_eq_mul] at h1
      have hpos : 0 < |L.mulVec v₀ i₀| := abs_pos.mpr hi₀
      have hs0 : 0 < s := lt_of_le_of_lt (by positivity) hs
      have hmul : |L.mulVec v i₀| + 1 < s * |L.mulVec v₀ i₀| := (div_lt_iff₀ hpos).mp hs
      have heq' : |s * L.mulVec v₀ i₀| = |-L.mulVec v i₀| := by
        congr 1; simp only [Pi.zero_apply] at h1; linarith
      rw [abs_neg, abs_mul, abs_of_pos hs0] at heq'
      linarith
    have hMs_ev : ∀ Ls : List (Matrix (Fin d₀) (Fin d₀) ℝ),
        (∀ M ∈ Ls, dotProduct v₀ (M.mulVec v₀) ≠ 0) →
        ∃ S : ℝ, ∀ s : ℝ, S < s → ∀ M ∈ Ls, dotProduct (v + s • v₀) (M.mulVec (v + s • v₀)) ≠ 0 := by
      intro Ls
      induction Ls with
      | nil => intro _; exact ⟨0, fun _ _ _ h => by simp at h⟩
      | cons M Ls ih =>
        intro hLs
        obtain ⟨S₁, hS₁⟩ := ih (fun N hN => hLs N (by simp [hN]))
        obtain ⟨S₂, hS₂⟩ := quad_eventually_ne_zero
          (a := dotProduct v₀ (M.mulVec v₀))
          (b := dotProduct v (M.mulVec v₀) + dotProduct v₀ (M.mulVec v))
          (c := dotProduct v (M.mulVec v))
          (hLs M (by simp))
        refine ⟨max S₁ S₂, fun s hs N hN => ?_⟩
        simp only [List.mem_cons] at hN
        rcases hN with hNM | hN'
        · rw [hNM]
          have hquad : dotProduct (v + s • v₀) (M.mulVec (v + s • v₀)) =
              dotProduct v₀ (M.mulVec v₀) * s ^ 2 +
              (dotProduct v (M.mulVec v₀) + dotProduct v₀ (M.mulVec v)) * s +
              dotProduct v (M.mulVec v) := by
            rw [Matrix.mulVec_add, Matrix.mulVec_smul, add_dotProduct, dotProduct_add,
              dotProduct_add, smul_dotProduct, smul_dotProduct, dotProduct_smul, dotProduct_smul,
              smul_eq_mul, smul_eq_mul, smul_eq_mul, smul_eq_mul]; ring
          rw [hquad]
          exact hS₂ s (lt_of_le_of_lt (le_max_right _ _) hs)
        · exact hS₁ s (lt_of_le_of_lt (le_max_left _ _) hs) N hN'
    obtain ⟨S_L, hSL⟩ := hL_ev
    obtain ⟨S_M, hSM⟩ := hMs_ev Ms hv₀Ms
    set s₁ := max S_L S_M + 1
    set s₂ := max S_L S_M + 2
    have he1 := heq y (v + s₁ • v₀)
      (hSL s₁ (by dsimp [s₁]; linarith [le_max_left S_L S_M]))
      (hSM s₁ (by dsimp [s₁]; linarith [le_max_right S_L S_M]))
    have he2 := heq y (v + s₂ • v₀)
      (hSL s₂ (by dsimp [s₂]; linarith [le_max_left S_L S_M]))
      (hSM s₂ (by dsimp [s₂]; linarith [le_max_right S_L S_M]))
    have hlin : ∀ s : ℝ, dotProduct y (A.mulVec (v + s • v₀)) = dotProduct y (B.mulVec (v + s • v₀)) →
        (dotProduct y (A.mulVec v) - dotProduct y (B.mulVec v)) +
        s * (dotProduct y (A.mulVec v₀) - dotProduct y (B.mulVec v₀)) = 0 := by
      intro s hs
      rw [Matrix.mulVec_add, Matrix.mulVec_smul, dotProduct_add, dotProduct_smul, smul_eq_mul,
        Matrix.mulVec_add, Matrix.mulVec_smul, dotProduct_add, dotProduct_smul, smul_eq_mul] at hs
      linarith
    have h1 := hlin s₁ he1
    have h2 := hlin s₂ he2
    have hdiff : s₂ - s₁ = 1 := by dsimp [s₁, s₂]; ring
    have hv0_eq : dotProduct y (A.mulVec v₀) - dotProduct y (B.mulVec v₀) = 0 := by
      linear_combination h2 - h1 - (dotProduct y (A.mulVec v₀) - dotProduct y (B.mulVec v₀)) * hdiff
    linear_combination h1 - s₁ * hv0_eq
  ext i j
  let ei : Fin d₀ → ℝ := fun k => if k = i then 1 else 0
  let ej : Fin d₀ → ℝ := fun k => if k = j then 1 else 0
  have h := hpoint ei ej
  simpa [ei, ej, dotProduct, Matrix.mulVec] using h

/-- Extract equal leading coefficients from an `O(u^{k+1})` difference on `(0, u₀)`. -/
lemma eq_of_bigO_succ {k : ℕ} {α α' : ℝ} {f f' : ℝ → ℝ}
    (heq : ∃ u₀ > 0, ∀ u, 0 < u → u < u₀ → f u = f' u)
    (hf : ∃ u₁ > 0, ∃ C₁ ≥ 0, ∀ u, 0 < u → u < u₁ → |f u - α * u ^ k| ≤ C₁ * u ^ (k + 1))
    (hf' : ∃ u₂ > 0, ∃ C₂ ≥ 0, ∀ u, 0 < u → u < u₂ → |f' u - α' * u ^ k| ≤ C₂ * u ^ (k + 1)) :
    α = α' := by
  obtain ⟨u₀, hu₀, h0⟩ := heq
  obtain ⟨u₁, hu₁, C₁, hC₁, h1⟩ := hf
  obtain ⟨u₂, hu₂, C₂, hC₂, h2⟩ := hf'
  by_contra hne
  have hpos : 0 < |α - α'| := abs_pos.mpr (sub_ne_zero.mpr hne)
  set C := C₁ + C₂ + 1
  have hC : 0 < C := by dsimp [C]; linarith
  set m := min (min u₀ (min u₁ u₂)) (|α - α'| / C)
  have hm_pos : 0 < m := lt_min (lt_min hu₀ (lt_min hu₁ hu₂)) (div_pos hpos hC)
  set u := m / 2
  have hu_pos : 0 < u := by dsimp [u]; linarith
  have hu_ltm : u < m := by dsimp [u]; linarith
  have hu_lt0 : u < u₀ :=
    lt_of_lt_of_le hu_ltm (le_trans (min_le_left _ _) (min_le_left _ _))
  have hu_lt1 : u < u₁ :=
    lt_of_lt_of_le hu_ltm (le_trans (min_le_left _ _) (le_trans (min_le_right _ _) (min_le_left _ _)))
  have hu_lt2 : u < u₂ :=
    lt_of_lt_of_le hu_ltm (le_trans (min_le_left _ _) (le_trans (min_le_right _ _) (min_le_right _ _)))
  have hu_ltC : u < |α - α'| / C :=
    lt_of_lt_of_le hu_ltm (min_le_right _ _)
  have hf_eq := h0 u hu_pos hu_lt0
  have hb1 := h1 u hu_pos hu_lt1
  have hb2 := h2 u hu_pos hu_lt2
  have hdiff : |(α - α') * u ^ k| ≤ (C₁ + C₂) * u ^ (k + 1) := by
    have hrew : (α - α') * u ^ k = (f' u - α' * u ^ k) - (f u - α * u ^ k) := by
      rw [hf_eq]; ring
    rw [hrew]
    have := abs_sub (f' u - α' * u ^ k) (f u - α * u ^ k)
    linarith
  rw [abs_mul, abs_of_pos (pow_pos hu_pos k), pow_succ] at hdiff
  have hcancel : |α - α'| ≤ (C₁ + C₂) * u := by
    nlinarith [pow_pos hu_pos k]
  have hlt : (C₁ + C₂) * u < |α - α'| := by
    have hCu : u * C < |α - α'| := (lt_div_iff₀ hC).mp hu_ltC
    dsimp [C] at hCu
    nlinarith
  linarith

set_option maxHeartbeats 1200000 in
/-- **Main Theorem 1 (Global Identifiability of Deep Softmax Self-Attention — Conjecture `conj:deepsmax`, arXiv:2408.17221)**:
For ANY depth `l ≥ 1`, ANY sequence length `t ≥ 2`, ANY dimensions `d₀, d_l ≥ 1`, and ANY
virtual weights `(Ms, L)` and `(Ms', L')` with `Ms.length = Ms'.length`, `L ≠ 0`, and
every non-final layer `M ∈ Ms.dropLast` non-skew-symmetric (`M + Mᵀ ≠ 0`):
if `deepSoftmax Ms L X = deepSoftmax Ms' L' X` for all `X` in some open cube around `0`,
then **`L = L'` and `Ms = Ms'`**! -/
theorem deep_softmax_identifiability {d₀ d_l t : ℕ} (ht : 2 ≤ t)
    (Ms Ms' : List (Matrix (Fin d₀) (Fin d₀) ℝ)) (hlen : Ms.length = Ms'.length)
    (L L' : Matrix (Fin d_l) (Fin d₀) ℝ)
    (hL : L ≠ 0)
    (hns : ∀ M ∈ Ms.dropLast, M + M.transpose ≠ 0)
    {δ : ℝ} (hδ : 0 < δ)
    (heq : ∀ X : Fin t → Fin d₀ → ℝ, (∀ i k, |X i k| < δ) →
      deepSoftmax Ms L X = deepSoftmax Ms' L' X) :
    L = L' ∧ Ms = Ms' := by
  have ht0 : 0 < t := by omega
  -- Step 1: Recover L = L'
  have hL_eq : L = L' := recover_L ht0 Ms Ms' L L' hδ heq
  refine ⟨hL_eq, ?_⟩
  subst hL_eq
  -- Helper: whenever L v ≠ 0, a_l(u) = a'_l(u) for all sufficiently small u > 0
  have ha_eq : ∀ (y v : Fin d₀ → ℝ) (η : ℝ), L.mulVec v ≠ 0 →
      ∃ u₀ > 0, ∀ u : ℝ, 0 < u → u < u₀ →
        (abRec t y v η u Ms (1, 0)).1 = (abRec t y v η u Ms' (1, 0)).1 := by
    intro y v η hLv
    -- Pick a coordinate i₀ where (L v)_i₀ ≠ 0
    have hex : ∃ i₀, L.mulVec v i₀ ≠ 0 := by
      by_contra hall; push_neg at hall; apply hLv; funext i; exact hall i
    obtain ⟨i₀, hi₀⟩ := hex
    -- Bound the entries of η y + v and η y to choose u₀ > 0 so twoClusterSeq lies in (-δ, δ)
    let M_norm : ℝ := (∑ k : Fin d₀, (|η * y k| + |v k|)) + 1
    have hM_pos : 0 < M_norm := by
      dsimp [M_norm]
      have : 0 ≤ ∑ k : Fin d₀, (|η * y k| + |v k|) :=
        Finset.sum_nonneg fun k _ => by positivity
      linarith
    refine ⟨(δ / (2 * M_norm)) ^ 2, by positivity, fun u hu_pos hu_lt => ?_⟩
    have hsqrt_lt : Real.sqrt u < δ / (2 * M_norm) := by
      have h1 : Real.sqrt u < Real.sqrt ((δ / (2 * M_norm)) ^ 2) := Real.sqrt_lt_sqrt (le_of_lt hu_pos) hu_lt
      rwa [Real.sqrt_sq (by positivity)] at h1
    have hsmall : ∀ (i : Fin t) (k : Fin d₀), |twoClusterSeq t y v η u 1 0 i k| < δ := by
      intro i k
      dsimp only [twoClusterSeq, Pi.smul_apply, Pi.add_apply, smul_eq_mul]
      rw [abs_mul, abs_of_nonneg (Real.sqrt_nonneg u)]
      have hcoeff : |η * y k + (if i.val = 0 then (1 : ℝ) else 0) * v k| ≤ M_norm := by
        have htri := abs_add_le (η * y k) ((if i.val = 0 then (1 : ℝ) else 0) * v k)
        have hterm : |(if i.val = 0 then (1 : ℝ) else 0) * v k| ≤ |v k| := by
          split_ifs <;> simp [abs_nonneg]
        have hsing : |η * y k| + |v k| ≤ ∑ j : Fin d₀, (|η * y j| + |v j|) :=
          Finset.single_le_sum (f := fun j : Fin d₀ => |η * y j| + |v j|)
            (fun j _ => by positivity) (Finset.mem_univ k)
        dsimp [M_norm]; linarith
      have hmul : Real.sqrt u * |η * y k + (if i.val = 0 then (1 : ℝ) else 0) * v k| ≤
          Real.sqrt u * M_norm := mul_le_mul_of_nonneg_left hcoeff (Real.sqrt_nonneg u)
      have hbound : Real.sqrt u * M_norm < δ / 2 := by
        have := (lt_div_iff₀ (by positivity : (0 : ℝ) < 2 * M_norm)).mp hsqrt_lt
        linarith
      linarith
    have hX := congrFun (congrFun (heq (twoClusterSeq t y v η u 1 0) hsmall) ⟨0, ht0⟩) i₀
    simp only [deepSoftmax] at hX
    rw [twoCluster_invariant ht y v η (le_of_lt hu_pos) Ms (1, 0),
        twoCluster_invariant ht y v η (le_of_lt hu_pos) Ms' (1, 0)] at hX
    have heval : ∀ a b : ℝ,
        L.mulVec (twoClusterSeq t y v η u a b ⟨0, ht0⟩) i₀ =
          Real.sqrt u * (η * L.mulVec y i₀ + a * L.mulVec v i₀) := by
      intro a b
      have hvec : twoClusterSeq t y v η u a b ⟨0, ht0⟩ = Real.sqrt u • (η • y + a • v) := by
        simp [twoClusterSeq]
      rw [hvec, Matrix.mulVec_smul, Matrix.mulVec_add, Matrix.mulVec_smul, Matrix.mulVec_smul]
      simp [Pi.smul_apply, Pi.add_apply, smul_eq_mul]
    rw [heval, heval] at hX
    have hsq_ne : Real.sqrt u ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hu_pos)
    have hcancel := mul_left_cancel₀ hsq_ne hX
    have hcancel2 : (abRec t y v η u Ms (1, 0)).1 * L.mulVec v i₀ =
        (abRec t y v η u Ms' (1, 0)).1 * L.mulVec v i₀ := by linarith
    exact mul_right_cancel₀ hi₀ hcancel2
  -- Step 4: Recover Ms[r] = Ms'[r] inductively for r = 0, 1, ..., Ms.length - 1!
  suffices hpref : ∀ r : ℕ, r ≤ Ms.length → Ms.take r = Ms'.take r by
    have h := hpref Ms.length le_rfl
    rw [List.take_length, hlen, List.take_length] at h
    exact h
  intro r
  induction r with
  | zero => intro _; simp
  | succ r ih =>
    intro hr_le
    have hprev : Ms.take r = Ms'.take r := ih (Nat.le_of_succ_le hr_le)
    have hr_lt : r < Ms.length := hr_le
    have hr_lt' : r < Ms'.length := hlen ▸ hr_lt
    -- Write Ms = pre ++ M :: suf, Ms' = pre ++ M' :: suf'
    set pre := Ms.take r
    set M := Ms[r]'hr_lt
    set M' := Ms'[r]'hr_lt'
    set suf := Ms.drop (r + 1)
    set suf' := Ms'.drop (r + 1)
    have hMs_split : Ms = pre ++ M :: suf := by
      have h1 := List.take_append_drop r Ms
      have h2 : Ms.drop r = M :: suf := List.drop_eq_getElem_cons hr_lt
      rw [h2] at h1
      exact h1.symm
    have hMs'_split : Ms' = pre ++ M' :: suf' := by
      have h1 := List.take_append_drop r Ms'
      rw [← hprev] at h1
      have h2 : Ms'.drop r = M' :: suf' := List.drop_eq_getElem_cons hr_lt'
      rw [h2] at h1
      exact h1.symm
    have hpre_len : pre.length = r := List.length_take_of_le (Nat.le_of_succ_le hr_le)
    have hpre_ns : ∀ N ∈ pre, N + N.transpose ≠ 0 := by
      intro N hN
      apply hns N
      have hsub : pre = (Ms.dropLast).take r := by
        dsimp [pre]
        rw [List.dropLast_eq_take, List.take_take, Nat.min_eq_left (by omega)]
      rw [hsub] at hN
      exact List.mem_of_mem_take hN
    -- Prove M = M' by showing yᵀ M v = yᵀ M' v on the Zariski-open locus for pre!
    have hM_eq : M = M' := by
      refine matrix_eq_of_eq_on_zariski_open hL pre hpre_ns M M' fun y v hLv hquad => ?_
      have hc_ne : cCoeff t v pre.reverse ≠ 0 :=
        cCoeff_ne_zero ht v pre.reverse (fun N hN => hquad N (List.mem_reverse.mp hN))
      -- Helper: for any η, compute the leading coefficient of residual R(u) at u^{3^r}
      have hextract : ∀ η : ℝ,
          sig0p t * (cCoeff t v pre.reverse) ^ 2 *
            (η * dotProduct y (M.mulVec v) + (if r = 0 then (if suf = [] then 1 else 1 / (t : ℝ)) else 1 / (t : ℝ)) * dotProduct v (M.mulVec v)) =
          sig0p t * (cCoeff t v pre.reverse) ^ 2 *
            (η * dotProduct y (M'.mulVec v) + (if r = 0 then (if suf = [] then 1 else 1 / (t : ℝ)) else 1 / (t : ℝ)) * dotProduct v (M'.mulVec v)) := by
        intro η
        have hlen_suf : suf.length = suf'.length := by
          dsimp [suf, suf']; simp [hlen]
        by_cases hr0 : r = 0
        · -- Base case r = 0: pre = []!
          have hpre_nil : pre = [] := List.eq_nil_of_length_eq_zero (by omega)
          have hMs_eq : Ms = M :: suf := by rw [hMs_split, hpre_nil]; rfl
          have hMs'_eq : Ms' = M' :: suf' := by rw [hMs'_split, hpre_nil]; rfl
          have hc1 : cCoeff t v pre.reverse = 1 := by rw [hpre_nil]; rfl
          by_cases hsuf : suf = []
          · -- l = 1 (single layer!): a_1(u) - 1/t = σ'(0) (η yᵀ M v + vᵀ M v) u + O(u²)
            have hsuf' : suf' = [] := List.eq_nil_of_length_eq_zero (by rw [← hlen_suf, hsuf]; rfl)
            have hexp1 : ∀ (A : Matrix (Fin d₀) (Fin d₀) ℝ),
                ∃ u₁ > 0, ∃ C₁ ≥ 0, ∀ u, 0 < u → u < u₁ →
                  |(abRec t y v η u [A] (1, 0)).1 - 1 / (t : ℝ) -
                    sig0p t * (η * dotProduct y (A.mulVec v) + dotProduct v (A.mulVec v)) * u ^ 1| ≤
                    C₁ * u ^ (1 + 1) := by
              intro A
              set K := |η * dotProduct y (A.mulVec v) + dotProduct v (A.mulVec v)| + 1
              refine ⟨1 / K, by positivity, 9 * K ^ 2, by positivity, fun u hu_pos hu_lt => ?_⟩
              have huK : u * K ≤ 1 := le_of_lt ((lt_div_iff₀ (by positivity : 0 < K)).mp hu_lt)
              have hΔ1 : |u * (η * dotProduct y (A.mulVec v) + dotProduct v (A.mulVec v))| ≤ 1 := by
                rw [abs_mul, abs_of_pos hu_pos]
                dsimp [K] at huK ⊢; nlinarith [abs_nonneg (η * dotProduct y (A.mulVec v) + dotProduct v (A.mulVec v))]
              have hval : (abRec t y v η u [A] (1, 0)).1 =
                  sigma t (u * (η * dotProduct y (A.mulVec v) + dotProduct v (A.mulVec v))) := by
                simp only [abRec, abStep, sub_zero, zero_add, one_mul, mul_one, one_smul]
                rw [add_dotProduct, smul_dotProduct, smul_eq_mul]
              rw [hval, pow_one]
              have hst := sigma_taylor_bound ht hΔ1
              have hrew : sigma t (u * (η * dotProduct y (A.mulVec v) + dotProduct v (A.mulVec v))) -
                  1 / (t : ℝ) - sig0p t * (η * dotProduct y (A.mulVec v) + dotProduct v (A.mulVec v)) * u =
                  sigma t (u * (η * dotProduct y (A.mulVec v) + dotProduct v (A.mulVec v))) -
                  1 / (t : ℝ) - sig0p t * (u * (η * dotProduct y (A.mulVec v) + dotProduct v (A.mulVec v))) := by ring
              rw [hrew]
              have hsq : (u * (η * dotProduct y (A.mulVec v) + dotProduct v (A.mulVec v))) ^ 2 ≤ K ^ 2 * u ^ 2 := by
                rw [mul_pow, mul_comm (u ^ 2)]
                apply mul_le_mul_of_nonneg_right _ (sq_nonneg u)
                rw [← sq_abs]
                dsimp [K]; nlinarith [abs_nonneg (η * dotProduct y (A.mulVec v) + dotProduct v (A.mulVec v))]
              nlinarith
            have hres := eq_of_bigO_succ (k := 1)
              (f := fun u => (abRec t y v η u Ms (1, 0)).1 - 1 / (t : ℝ))
              (f' := fun u => (abRec t y v η u Ms' (1, 0)).1 - 1 / (t : ℝ))
              (by
                obtain ⟨u₀, hu₀, h0⟩ := ha_eq y v η hLv
                exact ⟨u₀, hu₀, fun u h1 h2 => by dsimp only; rw [h0 u h1 h2]⟩)
              (by rw [hMs_eq, hsuf]; exact hexp1 M)
              (by rw [hMs'_eq, hsuf']; exact hexp1 M')
            simpa [hr0, hsuf, hc1] using hres
          · -- r = 0, suf ≠ [] (l ≥ 2!): a_l(u) - 1/t = σ'(0) (η yᵀ M v + (1/t) vᵀ M v) u + O(u²)
            obtain ⟨M2, suf2, hsuf_eq⟩ := List.exists_cons_of_ne_nil hsuf
            have hsuf'_ne : suf' ≠ [] := by intro h; rw [h] at hlen_suf; exact hsuf (List.eq_nil_of_length_eq_zero (by simpa using hlen_suf))
            obtain ⟨M2', suf2', hsuf'_eq⟩ := List.exists_cons_of_ne_nil hsuf'_ne
            have hexp_ge2 : ∀ (A B : Matrix (Fin d₀) (Fin d₀) ℝ) (rest : List (Matrix (Fin d₀) (Fin d₀) ℝ)),
                ∃ u₁ > 0, ∃ C₁ ≥ 0, ∀ u, 0 < u → u < u₁ →
                  |(abRec t y v η u (A :: B :: rest) (1, 0)).1 - 1 / (t : ℝ) -
                    sig0p t * (η * dotProduct y (A.mulVec v) + (1 / (t : ℝ)) * dotProduct v (A.mulVec v)) * u ^ 1| ≤
                    C₁ * u ^ (1 + 1) := by
              intro A B rest
              obtain ⟨u₁, hu₁, C₁, hC₁, h1⟩ := lemma2_expansion ht y v η [A]
              obtain ⟨u₂, hu₂, C₂, hC₂, h2⟩ := lemma2_expansion ht y v η [A, B]
              obtain ⟨u₃, hu₃, C₃, hC₃, h3⟩ := lemma2_b_step ht y v η [] A
              obtain ⟨u₄, hu₄, C₄, hC₄, h4⟩ := lemma2_b_step ht y v η [A] B
              set u₀ := min (min u₁ u₂) (min (min u₃ u₄) 1)
              have hu₀ : 0 < u₀ := lt_min (lt_min hu₁ hu₂) (lt_min (lt_min hu₃ hu₄) (by norm_num))
              set C_tot := C₃ + (1 / (t : ℝ)) * C₁ +
                (|sig0p t * (cCoeff t v [A].reverse) ^ 2 * (η * dotProduct y (B.mulVec v) + (1 / (t : ℝ)) * dotProduct v (B.mulVec v))| + C₄) +
                (|cCoeff t v [A, B].reverse| + C₂) + 1
              refine ⟨u₀, hu₀, C_tot, by positivity, fun u hu_pos hu_lt => ?_⟩
              have hu_le1 : u ≤ 1 := le_of_lt (lt_of_lt_of_le hu_lt (le_trans (min_le_right _ _) (min_le_right _ _)))
              have hu_lt1 : u < u₁ := lt_of_lt_of_le hu_lt (le_trans (min_le_left _ _) (min_le_left _ _))
              have hu_lt2 : u < u₂ := lt_of_lt_of_le hu_lt (le_trans (min_le_left _ _) (min_le_right _ _))
              have hu_lt3 : u < u₃ := lt_of_lt_of_le hu_lt (le_trans (min_le_right _ _) (le_trans (min_le_left _ _) (min_le_left _ _)))
              have hu_lt4 : u < u₄ := lt_of_lt_of_le hu_lt (le_trans (min_le_right _ _) (le_trans (min_le_left _ _) (min_le_right _ _)))
              -- Decompose: a_l - 1/t = (a_l - b_2) + (b_2 - b_1 - (1/t) d_1) + (b_1 - 1/t) + (1/t) d_1
              set ab1 := abRec t y v η u [A] (1, 0)
              set ab2 := abRec t y v η u [A, B] (1, 0)
              have hab_l : abRec t y v η u (A :: B :: rest) (1, 0) =
                  abRec t y v η u rest ab2 := by
                have : A :: B :: rest = [A, B] ++ rest := rfl
                rw [this, abRec_append]
              have hcont : |(abRec t y v η u (A :: B :: rest) (1, 0)).1 - ab2.2| ≤ |ab2.1 - ab2.2| := by
                rw [hab_l]; exact (abRec_in_convex_hull ht y v η u rest ab2).1
              have hd2_bound : |ab2.1 - ab2.2| ≤ (|cCoeff t v [A, B].reverse| + C₂) * u ^ 2 := by
                have h := (h2 u hu_pos hu_lt2).2
                have he2 : eExp ([A, B] : List _).length = 4 := by simp [eExp]
                rw [he2] at h
                change |ab2.1 - ab2.2 - cCoeff t v [A, B].reverse * u ^ 4| ≤ C₂ * u ^ 5 at h
                have htri : |ab2.1 - ab2.2| ≤ |(ab2.1 - ab2.2) - cCoeff t v [A, B].reverse * u ^ 4| +
                    |cCoeff t v [A, B].reverse| * u ^ 4 := by
                  have := abs_add_le ((ab2.1 - ab2.2) - cCoeff t v [A, B].reverse * u ^ 4)
                    (cCoeff t v [A, B].reverse * u ^ 4)
                  rwa [abs_mul, abs_of_pos (pow_pos hu_pos 4), sub_add_cancel] at this
                have hu4_le : u ^ 4 ≤ u ^ 2 := pow_le_pow_of_le_one (le_of_lt hu_pos) hu_le1 (by omega)
                have hu5_le : u ^ 5 ≤ u ^ 2 := pow_le_pow_of_le_one (le_of_lt hu_pos) hu_le1 (by omega)
                have hm1 : |cCoeff t v [A, B].reverse| * u ^ 4 ≤ |cCoeff t v [A, B].reverse| * u ^ 2 :=
                  mul_le_mul_of_nonneg_left hu4_le (abs_nonneg _)
                have hm2 : C₂ * u ^ 5 ≤ C₂ * u ^ 2 := mul_le_mul_of_nonneg_left hu5_le hC₂
                linarith
              have hb2_bound : |ab2.2 - ab1.2 - (1 / (t : ℝ)) * (ab1.1 - ab1.2)| ≤
                  (|sig0p t * (cCoeff t v [A].reverse) ^ 2 *
                    (η * dotProduct y (B.mulVec v) + (1 / (t : ℝ)) * dotProduct v (B.mulVec v))| + C₄) * u ^ 2 := by
                have h := h4 u hu_pos hu_lt4
                simp only [List.length_singleton, pow_one, List.cons_ne_self, ↓reduceIte] at h
                change |ab2.2 - ab1.2 - 1 / (t : ℝ) * (ab1.1 - ab1.2) -
                  sig0p t * cCoeff t v [A].reverse ^ 2 *
                    (η * dotProduct y (B.mulVec v) + 1 / (t : ℝ) * dotProduct v (B.mulVec v)) * u ^ 3| ≤
                  C₄ * u ^ 4 at h
                have htri : |ab2.2 - ab1.2 - 1 / (t : ℝ) * (ab1.1 - ab1.2)| ≤
                    |ab2.2 - ab1.2 - 1 / (t : ℝ) * (ab1.1 - ab1.2) -
                      sig0p t * cCoeff t v [A].reverse ^ 2 *
                        (η * dotProduct y (B.mulVec v) + 1 / (t : ℝ) * dotProduct v (B.mulVec v)) * u ^ 3| +
                    |sig0p t * cCoeff t v [A].reverse ^ 2 *
                      (η * dotProduct y (B.mulVec v) + 1 / (t : ℝ) * dotProduct v (B.mulVec v))| * u ^ 3 := by
                  have := abs_add_le
                    (ab2.2 - ab1.2 - 1 / (t : ℝ) * (ab1.1 - ab1.2) -
                      sig0p t * cCoeff t v [A].reverse ^ 2 *
                        (η * dotProduct y (B.mulVec v) + 1 / (t : ℝ) * dotProduct v (B.mulVec v)) * u ^ 3)
                    (sig0p t * cCoeff t v [A].reverse ^ 2 *
                      (η * dotProduct y (B.mulVec v) + 1 / (t : ℝ) * dotProduct v (B.mulVec v)) * u ^ 3)
                  rwa [abs_mul, abs_of_pos (pow_pos hu_pos 3), sub_add_cancel] at this
                have hu3_le : u ^ 3 ≤ u ^ 2 := pow_le_pow_of_le_one (le_of_lt hu_pos) hu_le1 (by omega)
                have hu4_le : u ^ 4 ≤ u ^ 2 := pow_le_pow_of_le_one (le_of_lt hu_pos) hu_le1 (by omega)
                have hm1 := mul_le_mul_of_nonneg_left hu3_le
                  (abs_nonneg (sig0p t * cCoeff t v [A].reverse ^ 2 *
                    (η * dotProduct y (B.mulVec v) + 1 / (t : ℝ) * dotProduct v (B.mulVec v))))
                have hm2 := mul_le_mul_of_nonneg_left hu4_le hC₄
                linarith
              have hb1_bound : |ab1.2 - 1 / (t : ℝ) - sig0p t * (η * dotProduct y (A.mulVec v)) * u| ≤
                  C₃ * u ^ 2 := by
                have h := h3 u hu_pos hu_lt3
                simp only [abRec, List.length_nil, pow_zero, List.reverse_nil, cCoeff, one_pow,
                  mul_one, ↓reduceIte, zero_mul, add_zero, sub_zero, pow_one] at h
                simpa [ab1, abRec, abStep, mul_one, zero_add] using h
              have hd1_bound : |(ab1.1 - ab1.2) - sig0p t * dotProduct v (A.mulVec v) * u| ≤ C₁ * u ^ 2 := by
                have h := (h1 u hu_pos hu_lt1).2
                have he1 : eExp ([A] : List _).length = 1 := by simp [eExp]
                rw [he1] at h
                simpa [ab1, cCoeff, pow_one] using h
              have hrew : (abRec t y v η u (A :: B :: rest) (1, 0)).1 - 1 / (t : ℝ) -
                  sig0p t * (η * dotProduct y (A.mulVec v) + (1 / (t : ℝ)) * dotProduct v (A.mulVec v)) * u ^ 1 =
                  ((abRec t y v η u (A :: B :: rest) (1, 0)).1 - ab2.2) +
                  (ab2.2 - ab1.2 - (1 / (t : ℝ)) * (ab1.1 - ab1.2)) +
                  (ab1.2 - 1 / (t : ℝ) - sig0p t * (η * dotProduct y (A.mulVec v)) * u) +
                  (1 / (t : ℝ)) * ((ab1.1 - ab1.2) - sig0p t * dotProduct v (A.mulVec v) * u) := by
                ring
              rw [hrew]
              have ht1 := abs_add_le
                (((abRec t y v η u (A :: B :: rest) (1, 0)).1 - ab2.2) +
                 (ab2.2 - ab1.2 - (1 / (t : ℝ)) * (ab1.1 - ab1.2)) +
                 (ab1.2 - 1 / (t : ℝ) - sig0p t * (η * dotProduct y (A.mulVec v)) * u))
                ((1 / (t : ℝ)) * ((ab1.1 - ab1.2) - sig0p t * dotProduct v (A.mulVec v) * u))
              have ht2 := abs_add_le
                (((abRec t y v η u (A :: B :: rest) (1, 0)).1 - ab2.2) +
                 (ab2.2 - ab1.2 - (1 / (t : ℝ)) * (ab1.1 - ab1.2)))
                (ab1.2 - 1 / (t : ℝ) - sig0p t * (η * dotProduct y (A.mulVec v)) * u)
              have ht3 := abs_add_le
                ((abRec t y v η u (A :: B :: rest) (1, 0)).1 - ab2.2)
                (ab2.2 - ab1.2 - (1 / (t : ℝ)) * (ab1.1 - ab1.2))
              have hinv_pos : (0 : ℝ) ≤ 1 / (t : ℝ) := by positivity
              rw [abs_mul, abs_of_nonneg hinv_pos] at ht1
              have hd1_scaled : (1 / (t : ℝ)) * |(ab1.1 - ab1.2) - sig0p t * dotProduct v (A.mulVec v) * u| ≤
                  (1 / (t : ℝ)) * C₁ * u ^ 2 := by
                have := mul_le_mul_of_nonneg_left hd1_bound hinv_pos
                linarith
              have hu2_nn : 0 ≤ u ^ 2 := sq_nonneg u
              have htot : (C₃ + (1 / (t : ℝ)) * C₁ +
                  (|sig0p t * (cCoeff t v [A].reverse) ^ 2 *
                    (η * dotProduct y (B.mulVec v) + (1 / (t : ℝ)) * dotProduct v (B.mulVec v))| + C₄) +
                  (|cCoeff t v [A, B].reverse| + C₂)) * u ^ 2 ≤ C_tot * u ^ (1 + 1) :=
                mul_le_mul_of_nonneg_right (by dsimp [C_tot]; linarith) hu2_nn
              linarith
            have hres := eq_of_bigO_succ (k := 1)
              (f := fun u => (abRec t y v η u Ms (1, 0)).1 - 1 / (t : ℝ))
              (f' := fun u => (abRec t y v η u Ms' (1, 0)).1 - 1 / (t : ℝ))
              (by
                obtain ⟨u₀, hu₀, h0⟩ := ha_eq y v η hLv
                exact ⟨u₀, hu₀, fun u h1 h2 => by dsimp only; rw [h0 u h1 h2]⟩)
              (by rw [hMs_eq, hsuf_eq]; exact hexp_ge2 M M2 suf2)
              (by rw [hMs'_eq, hsuf'_eq]; exact hexp_ge2 M' M2' suf2')
            simpa [hr0, hsuf, hc1] using hres
        · -- Inductive case r ≥ 1 (layer r + 1 ≥ 2): exponent gap e_{r+1} ≥ 3^r + 1!
          have hr1 : 1 ≤ r := Nat.one_le_iff_ne_zero.mpr hr0
          have hpre_ne : pre ≠ [] := by intro h; rw [h] at hpre_len; simp at hpre_len; omega
          have hexp_ind : ∀ (A : Matrix (Fin d₀) (Fin d₀) ℝ) (rest : List (Matrix (Fin d₀) (Fin d₀) ℝ)),
              ∃ u₁ > 0, ∃ C₁ ≥ 0, ∀ u, 0 < u → u < u₁ →
                let ab_p := abRec t y v η u pre (1, 0)
                |(abRec t y v η u (pre ++ A :: rest) (1, 0)).1 - ab_p.2 - (1 / (t : ℝ)) * (ab_p.1 - ab_p.2) -
                  sig0p t * (cCoeff t v pre.reverse) ^ 2 *
                    (η * dotProduct y (A.mulVec v) + (1 / (t : ℝ)) * dotProduct v (A.mulVec v)) * u ^ (3 ^ r)| ≤
                  C₁ * u ^ (3 ^ r + 1) := by
            intro A rest
            obtain ⟨u₁, hu₁, C₁, hC₁, h1⟩ := lemma2_expansion ht y v η (pre ++ [A])
            obtain ⟨u₂, hu₂, C₂, hC₂, h2⟩ := lemma2_b_step ht y v η pre A
            set u₀ := min (min u₁ u₂) 1
            have hu₀ : 0 < u₀ := lt_min (lt_min hu₁ hu₂) (by norm_num)
            set C_tot := C₂ + (|cCoeff t v (pre ++ [A]).reverse| + C₁) + 1
            refine ⟨u₀, hu₀, C_tot, by positivity, fun u hu_pos hu_lt => ?_⟩
            have hu_le1 : u ≤ 1 := le_of_lt (lt_of_lt_of_le hu_lt (min_le_right _ _))
            have hu_lt1 : u < u₁ := lt_of_lt_of_le hu_lt (le_trans (min_le_left _ _) (min_le_left _ _))
            have hu_lt2 : u < u₂ := lt_of_lt_of_le hu_lt (le_trans (min_le_left _ _) (min_le_right _ _))
            set ab_p := abRec t y v η u pre (1, 0)
            set ab_q := abRec t y v η u (pre ++ [A]) (1, 0)
            have hab_q_step : ab_q = abStep t y v η u A ab_p := by
              dsimp [ab_q, ab_p]; rw [abRec_append]; rfl
            have hab_l : abRec t y v η u (pre ++ A :: rest) (1, 0) =
                abRec t y v η u rest ab_q := by
              have hsplit : pre ++ A :: rest = (pre ++ [A]) ++ rest := by simp
              rw [hsplit, abRec_append]
            have hcont : |(abRec t y v η u (pre ++ A :: rest) (1, 0)).1 - ab_q.2| ≤ |ab_q.1 - ab_q.2| := by
              rw [hab_l]; exact (abRec_in_convex_hull ht y v η u rest ab_q).1
            -- Crucial exponent gap: e_{r+1} ≥ 3^r + 1!
            have hgap : 3 ^ r + 1 ≤ eExp (pre ++ [A]).length := by
              have hlen_pa : (pre ++ [A]).length = r + 1 := by simp [hpre_len]
              rw [hlen_pa]
              have := exponent_gap (r := r + 1) (by omega)
              simpa using this
            have hdq_bound : |ab_q.1 - ab_q.2| ≤
                (|cCoeff t v (pre ++ [A]).reverse| + C₁) * u ^ (3 ^ r + 1) := by
              have h := (h1 u hu_pos hu_lt1).2
              have htri : |ab_q.1 - ab_q.2| ≤
                  |(ab_q.1 - ab_q.2) - cCoeff t v (pre ++ [A]).reverse * u ^ (eExp (pre ++ [A]).length)| +
                  |cCoeff t v (pre ++ [A]).reverse| * u ^ (eExp (pre ++ [A]).length) := by
                have := abs_add_le
                  ((ab_q.1 - ab_q.2) - cCoeff t v (pre ++ [A]).reverse * u ^ (eExp (pre ++ [A]).length))
                  (cCoeff t v (pre ++ [A]).reverse * u ^ (eExp (pre ++ [A]).length))
                rwa [abs_mul, abs_of_pos (pow_pos hu_pos _), sub_add_cancel] at this
              have hp1 : u ^ (eExp (pre ++ [A]).length) ≤ u ^ (3 ^ r + 1) :=
                pow_le_pow_of_le_one (le_of_lt hu_pos) hu_le1 hgap
              have hp2 : u ^ (eExp (pre ++ [A]).length + 1) ≤ u ^ (3 ^ r + 1) :=
                pow_le_pow_of_le_one (le_of_lt hu_pos) hu_le1 (by omega)
              nlinarith [abs_nonneg (cCoeff t v (pre ++ [A]).reverse)]
            have hb_step : |ab_q.2 - ab_p.2 - (1 / (t : ℝ)) * (ab_p.1 - ab_p.2) -
                sig0p t * (cCoeff t v pre.reverse) ^ 2 *
                  (η * dotProduct y (A.mulVec v) + (1 / (t : ℝ)) * dotProduct v (A.mulVec v)) * u ^ (3 ^ r)| ≤
                C₂ * u ^ (3 ^ r + 1) := by
              have h := h2 u hu_pos hu_lt2
              simpa [hpre_ne, hpre_len, hab_q_step, ab_p] using h
            have hrew : (abRec t y v η u (pre ++ A :: rest) (1, 0)).1 - ab_p.2 - (1 / (t : ℝ)) * (ab_p.1 - ab_p.2) -
                sig0p t * (cCoeff t v pre.reverse) ^ 2 *
                  (η * dotProduct y (A.mulVec v) + (1 / (t : ℝ)) * dotProduct v (A.mulVec v)) * u ^ (3 ^ r) =
                ((abRec t y v η u (pre ++ A :: rest) (1, 0)).1 - ab_q.2) +
                (ab_q.2 - ab_p.2 - (1 / (t : ℝ)) * (ab_p.1 - ab_p.2) -
                  sig0p t * (cCoeff t v pre.reverse) ^ 2 *
                    (η * dotProduct y (A.mulVec v) + (1 / (t : ℝ)) * dotProduct v (A.mulVec v)) * u ^ (3 ^ r)) := by
              ring
            dsimp only
            rw [hrew]
            have htri := abs_add_le
              ((abRec t y v η u (pre ++ A :: rest) (1, 0)).1 - ab_q.2)
              (ab_q.2 - ab_p.2 - (1 / (t : ℝ)) * (ab_p.1 - ab_p.2) -
                sig0p t * (cCoeff t v pre.reverse) ^ 2 *
                  (η * dotProduct y (A.mulVec v) + (1 / (t : ℝ)) * dotProduct v (A.mulVec v)) * u ^ (3 ^ r))
            have hpow_nn : 0 ≤ u ^ (3 ^ r + 1) := pow_nonneg (le_of_lt hu_pos) _
            dsimp [C_tot]
            nlinarith
          have hres := eq_of_bigO_succ (k := 3 ^ r)
            (f := fun u =>
              let ab_p := abRec t y v η u pre (1, 0)
              (abRec t y v η u Ms (1, 0)).1 - ab_p.2 - (1 / (t : ℝ)) * (ab_p.1 - ab_p.2))
            (f' := fun u =>
              let ab_p := abRec t y v η u pre (1, 0)
              (abRec t y v η u Ms' (1, 0)).1 - ab_p.2 - (1 / (t : ℝ)) * (ab_p.1 - ab_p.2))
            (by
              obtain ⟨u₀, hu₀, h0⟩ := ha_eq y v η hLv
              exact ⟨u₀, hu₀, fun u h1 h2 => by dsimp only; rw [h0 u h1 h2]⟩)
            (by rw [hMs_split]; exact hexp_ind M suf)
            (by rw [hMs'_split]; exact hexp_ind M' suf')
          simpa [hr0] using hres
      -- Take η = 1 minus η = 0 and cancel σ'(0) * c_r² ≠ 0!
      have h1 := hextract 1
      have h0 := hextract 0
      have hsub : sig0p t * (cCoeff t v pre.reverse) ^ 2 * dotProduct y (M.mulVec v) =
          sig0p t * (cCoeff t v pre.reverse) ^ 2 * dotProduct y (M'.mulVec v) := by
        linarith
      have hcoef_ne : sig0p t * (cCoeff t v pre.reverse) ^ 2 ≠ 0 :=
        mul_ne_zero (ne_of_gt (sig0p_pos ht)) (pow_ne_zero 2 hc_ne)
      exact mul_left_cancel₀ hcoef_ne hsub
    -- Now Ms.take (r + 1) = pre ++ [M] = pre ++ [M'] = Ms'.take (r + 1)!
    rw [List.take_succ_eq_append_getElem hr_lt, List.take_succ_eq_append_getElem hr_lt']
    exact congrArg₂ (· ++ [·]) hprev hM_eq

end DeepSoftmaxIdentifiability

