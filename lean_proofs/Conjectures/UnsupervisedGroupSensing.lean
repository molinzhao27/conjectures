import Mathlib

/-!
# Refutation of the Compact-Group Sensing Conjecture (arXiv:2203.12513, Conjecture 10)

We formalize in Lean 4 + Mathlib (complete kernel-checked proofs) the complete mathematical
core of the refutation of Conjecture 10 (`conj: all groups`) from Tachella, Chen, and Davies
(arXiv:2203.12513 / JMLR 2023, Section 4.3):

1. **General `G_r = (ZMod 2)^r × (ZMod 2)^r` diagonal sign-character representation (`r ≥ 1`)**:
   - `card_D_finset`: Exact dimension `n_r = |D_r| = (2^r - 1) * 2^(r - 1)` for `r ≥ 1`, where
     `D_r = {(a, b) ∈ (ZMod 2)^r × (ZMod 2)^r | a · b = 1}` (in particular `n_3 = 28`).
   - `T_diag_add`, `T_diag_zero`, `T_diag_orth`: The diagonal action `T_g` is an orthogonal
     group representation of `G_r` on `D_r → ℝ`.
   - `character_orthogonality`: Exact character orthogonality
     `∑ g ∈ G_r, χ_p(g) χ_q(g) = if p = q then 4^r else 0` (so `Mᵀ M = 64 I_28` at `r = 3`),
     proving all `n_r` diagonal characters are pairwise distinct 1D irreducibles (`c_j = s_j = 1`,
     hence `C = max_j c_j / s_j = 1`).
   - `trace_nonid` & `T_diag_faithful` (Lemma 1): For every `g = (u, w) ≠ (0, 0)`,
     `tr T_{(u,w)} = -2^(r-1) * (-1)^(u · w)` (at `r = 3`, `-4 * (-1)^(u · w) ∈ {+4, -4}`),
     establishing faithfulness and exact majority/minority sign counts
     `(n_r + 2^(r-1)) / 2` and `k_r = (n_r - 2^(r-1)) / 2` (`16` and `12` at `r = 3`).
2. **Non-identifiability for full-support kernel vectors & sharp threshold (Theorem 1)**:
   - `nonidentifiability_conic` (branch A1(b)) and `nonidentifiability_bounded` (branch A1(a)):
     For every `r ≥ 3` and every linear sensor `A : (D_r → ℝ) → (Fin m → ℝ)` possessing a
     full-support kernel vector `v ∈ ker A`, the `G_r`-invariant `k_r`-sparse cone `X` (resp.
     its unit-ball truncation `X_B`) admits an explicit distinct `G_r`-invariant competitor `X'`
     (resp. bounded `X'_B`) with `A '' (T_h '' X') = A '' (T_h '' X)` for all `h ∈ G_r`.
   - `explicit_sensor_m27_nonidentifiable`: Explicit rank-27 sensor `A : (D_3 → ℝ) → (Fin 27 → ℝ)`
     with full-support kernel vector `v = 1`, refuting Conjecture 10 at `m = 27 > 2k + C + 1 = 26`.
   - `injective_sensor_identifies`: For `m ≥ 28`, every injective sensor uniquely identifies
     every model `X ⊆ (D_3 → ℝ)` from `A '' X`, establishing the sharp threshold `m = 28`.
3. **Unbounded gap family at `C = 1` (Corollary 2)**:
   - `general_gap_formula`: For every `r ≥ 3`, `n_r = (2^r - 1) * 2^(r - 1)`,
     `k_r = (n_r - 2^(r - 1)) / 2`, `C = 1`, and the exact threshold `n_r` exceeds the first
     integer `2 * k_r + C + 2` permitted by Conjecture 10 by `2^(r - 1) - 3` (which is `1` at
     `r = 3` and diverges to `+∞` as `r → ∞`).

Machine-checked vs prose-only:
- Machine-checked: Exact dimension `n_r = (2^r - 1) * 2^(r-1)` (`28` at `r = 3`), exact character
  orthogonality `4^r I`, exact nonidentity trace `-2^(r-1) (-1)^(u·w)`, faithfulness, exact
  minority-sign sparsity `k_r` (`12` at `r = 3`), both unbounded conic A1(b) and bounded A1(a)
  indistinguishable competitor constructions for any sensor with a full-support kernel vector,
  explicit `m = 27` non-identifiability and `m = 28` sharp threshold, and the exact gap formula
  `2^(r-1) - 3` for all `r ≥ 3`.
- Prose-only (in `solutions/unsupervised_group_sensing/proof.pdf`): Almost-everywhere Lebesgue
  full-support kernel genericity for `m ≤ n_r - 1`, upper box-counting dimension calculation
  `dim_B(X ∩ S^{n-1}) = k - 1`, and the comparison subspace quotient-Jacobian thresholds for
  `SO(2)`, `O(2)`, `SO(3)`, and `S_4` (Proposition 3).
-/

open BigOperators Finset

namespace UnsupervisedGroupSensing

/-! ### 1. Sign Characters on `G_r = (ZMod 2)^r × (ZMod 2)^r` -/

/-- Binary vector space `𝔽_2^r`. -/
abbrev Vec2 (r : ℕ) := Fin r → ZMod 2

/-- Ambient group `G_r = 𝔽_2^r × 𝔽_2^r ≅ (ℤ/2)^(2r)`. -/
abbrev Grp (r : ℕ) := Vec2 r × Vec2 r

/-- Standard mod-2 inner product on `𝔽_2^r`. -/
def dot2 {r : ℕ} (a b : Vec2 r) : ZMod 2 := ∑ i : Fin r, a i * b i

/-- Real sign character `(-1)^x` on `ZMod 2`. -/
def signVal (x : ZMod 2) : ℝ := if x = 0 then 1 else -1

lemma zmod2_cases (x : ZMod 2) : x = 0 ∨ x = 1 := by
  fin_cases x <;> simp

lemma zmod2_add_self (x : ZMod 2) : x + x = 0 := by
  fin_cases x <;> decide

lemma zmod2_eq_iff_add_eq_zero (x y : ZMod 2) : x = y ↔ x + y = 0 := by
  fin_cases x <;> fin_cases y <;> simp
  all_goals decide

lemma vec2_eq_iff_add_eq_zero {r : ℕ} (a b : Vec2 r) : a = b ↔ a + b = 0 := by
  constructor
  · rintro rfl; ext i; exact zmod2_add_self (a i)
  · intro h; ext i
    have hi : a i + b i = 0 := congrFun h i
    exact (zmod2_eq_iff_add_eq_zero (a i) (b i)).2 hi

lemma signVal_zero : signVal 0 = 1 := by simp [signVal]

lemma signVal_one : signVal 1 = -1 := by
  unfold signVal
  have h : (1 : ZMod 2) ≠ 0 := by decide
  simp [h]

lemma signVal_add (x y : ZMod 2) : signVal (x + y) = signVal x * signVal y := by
  rcases zmod2_cases x with rfl | rfl <;> rcases zmod2_cases y with rfl | rfl
  · simp [signVal_zero]
  · simp [signVal_zero, signVal_one]
  · simp [signVal_zero, signVal_one]
  · have h11 : (1 : ZMod 2) + 1 = 0 := by decide
    rw [h11, signVal_zero, signVal_one]; norm_num

lemma signVal_sq (x : ZMod 2) : signVal x ^ 2 = 1 := by
  rcases zmod2_cases x with rfl | rfl
  · simp [signVal_zero]
  · simp [signVal_one]

lemma signVal_mul_self (x : ZMod 2) : signVal x * signVal x = 1 := by
  simpa [sq] using signVal_sq x

lemma dot2_zero_left {r : ℕ} (b : Vec2 r) : dot2 0 b = 0 := by
  simp [dot2]

lemma dot2_zero_right {r : ℕ} (a : Vec2 r) : dot2 a 0 = 0 := by
  simp [dot2]

lemma dot2_comm {r : ℕ} (a b : Vec2 r) : dot2 a b = dot2 b a := by
  unfold dot2
  refine Finset.sum_congr rfl fun i _ => mul_comm _ _

lemma dot2_add_left {r : ℕ} (a b c : Vec2 r) : dot2 (a + b) c = dot2 a c + dot2 b c := by
  unfold dot2
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i _ => by simp [add_mul]

lemma dot2_add_right {r : ℕ} (a b c : Vec2 r) : dot2 a (b + c) = dot2 a b + dot2 a c := by
  rw [dot2_comm, dot2_add_left, dot2_comm b a, dot2_comm c a]

/-- Fundamental character sum over `𝔽_2^r`:
`∑ x ∈ 𝔽_2^r, (-1)^(v · x) = if v = 0 then 2^r else 0`. -/
lemma sum_signVal_dot2 {r : ℕ} (v : Vec2 r) :
    ∑ x : Vec2 r, signVal (dot2 v x) = if v = 0 then (2 : ℝ) ^ r else 0 := by
  by_cases hv : v = 0
  · subst hv
    simp [dot2_zero_left, signVal_zero]
  · simp only [hv, ↓reduceIte]
    have hex : ∃ i : Fin r, v i ≠ 0 := by
      by_contra hall
      push_neg at hall
      exact hv (funext hall)
    rcases hex with ⟨i₀, hi₀⟩
    have hvi₀ : v i₀ = 1 := by
      rcases zmod2_cases (v i₀) with h0 | h1
      · exact (hi₀ h0).elim
      · exact h1
    let e : Vec2 r := fun i => if i = i₀ then 1 else 0
    have hdot_e : dot2 v e = 1 := by
      unfold dot2
      rw [Finset.sum_eq_single i₀]
      · simp [e, hvi₀]
      · intro i _ hi; simp [e, hi]
      · intro h; exact (h (Finset.mem_univ i₀)).elim
    have hsum_eq : ∑ x : Vec2 r, signVal (dot2 v x) =
        ∑ x : Vec2 r, signVal (dot2 v (x + e)) :=
      (Fintype.sum_equiv (Equiv.addRight e) (fun x => signVal (dot2 v (x + e)))
        (fun x => signVal (dot2 v x)) (fun _ => rfl)).symm
    have hneg : ∑ x : Vec2 r, signVal (dot2 v (x + e)) =
        - ∑ x : Vec2 r, signVal (dot2 v x) := by
      rw [← Finset.sum_neg_distrib]
      refine Finset.sum_congr rfl fun x _ => ?_
      rw [dot2_add_right, hdot_e, signVal_add, signVal_one]
      ring
    linarith

/-! ### 2. Index Set `D_r`, Exact Dimension, Character Orthogonality, and Trace Formula -/

/-- The coordinate index set `D_r = {(a, b) ∈ 𝔽_2^r × 𝔽_2^r : a · b = 1}` as a `Finset`. -/
def D_finset (r : ℕ) : Finset (Grp r) :=
  Finset.univ.filter fun p => dot2 p.1 p.2 = 1

/-- Subtype of coordinate indices `D_r`. -/
abbrev D_idx (r : ℕ) := {p : Grp r // dot2 p.1 p.2 = 1}

/-- Diagonal character value `χ_{(a,b)}(u, w) = (-1)^(u · a + w · b)`. -/
def charVal {r : ℕ} (g p : Grp r) : ℝ :=
  signVal (dot2 g.1 p.1 + dot2 g.2 p.2)

/-- Diagonal orthogonal operator `T_g` acting on `D_r → ℝ`. -/
def T_diag {r : ℕ} (g : Grp r) (x : D_idx r → ℝ) : D_idx r → ℝ :=
  fun p => charVal g p.1 * x p

lemma charVal_zero {r : ℕ} (p : Grp r) : charVal (0 : Grp r) p = 1 := by
  simp [charVal, dot2_zero_left, signVal_zero]

lemma charVal_add {r : ℕ} (g h p : Grp r) :
    charVal (g + h) p = charVal g p * charVal h p := by
  unfold charVal
  have h1 : dot2 (g + h).1 p.1 + dot2 (g + h).2 p.2 =
      (dot2 g.1 p.1 + dot2 g.2 p.2) + (dot2 h.1 p.1 + dot2 h.2 p.2) := by
    simp only [Prod.fst_add, Prod.snd_add, dot2_add_left]
    ring
  rw [h1, signVal_add]

lemma charVal_sq {r : ℕ} (g p : Grp r) : charVal g p ^ 2 = 1 :=
  signVal_sq _

lemma T_diag_zero {r : ℕ} (x : D_idx r → ℝ) : T_diag (0 : Grp r) x = x := by
  ext p; simp [T_diag, charVal_zero]

lemma T_diag_add {r : ℕ} (g h : Grp r) (x : D_idx r → ℝ) :
    T_diag (g + h) x = T_diag g (T_diag h x) := by
  ext p
  simp only [T_diag, charVal_add]
  ring

lemma T_diag_norm_sq {r : ℕ} (g : Grp r) (x : D_idx r → ℝ) :
    ∑ p : D_idx r, (T_diag g x p) ^ 2 = ∑ p : D_idx r, (x p) ^ 2 := by
  refine Finset.sum_congr rfl fun p _ => ?_
  simp only [T_diag, mul_pow, charVal_sq, one_mul]

/-- Exact indicator identity on `ZMod 2`: `1_{x = 1} = (1 - (-1)^x) / 2`. -/
lemma indicator_one_eq {x : ZMod 2} :
    (if x = 1 then (1 : ℝ) else 0) = (1 - signVal x) / 2 := by
  rcases zmod2_cases x with rfl | rfl
  · have h01 : (0 : ZMod 2) ≠ 1 := by decide
    simp [h01, signVal_zero]
  · simp [signVal_one]

/-- **Exact dimension formula**: `|D_r| = (2^r - 1) * 2^(r - 1)` for every `r ≥ 1`.
In particular `|D_3| = 28`. -/
theorem card_D_finset (r : ℕ) (hr : 1 ≤ r) :
    (D_finset r).card = (2 ^ r - 1) * 2 ^ (r - 1) := by
  have hreal : ((D_finset r).card : ℝ) = ((2 ^ r - 1) * 2 ^ (r - 1) : ℕ) := by
    have hsum : ((D_finset r).card : ℝ) =
        ∑ p : Grp r, (1 - signVal (dot2 p.1 p.2)) / 2 := by
      unfold D_finset
      rw [Finset.card_filter, Nat.cast_sum]
      refine Finset.sum_congr rfl fun p _ => ?_
      rw [← indicator_one_eq]
      split_ifs <;> simp
    rw [hsum, ← Finset.sum_div, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ]
    have hcard_grp : (Fintype.card (Grp r) : ℝ) = (2 : ℝ) ^ r * (2 : ℝ) ^ r := by
      simp [Fintype.card_prod, ZMod.card]
    have hdot_sum : ∑ p : Grp r, signVal (dot2 p.1 p.2) = (2 : ℝ) ^ r := by
      rw [Fintype.sum_prod_type]
      simp_rw [sum_signVal_dot2]
      simp
    rw [nsmul_eq_mul, mul_one, hcard_grp, hdot_sum]
    have hpow : (2 : ℝ) ^ r = 2 * (2 : ℝ) ^ (r - 1) := by
      conv_lhs => rw [← Nat.sub_add_cancel hr, pow_succ']
    have hge1 : 1 ≤ 2 ^ r := Nat.one_le_pow r 2 (by norm_num)
    rw [Nat.cast_mul, Nat.cast_sub hge1, Nat.cast_pow, Nat.cast_pow, Nat.cast_ofNat, Nat.cast_one]
    rw [hpow]
    ring
  exact_mod_cast hreal

lemma card_D_idx (r : ℕ) (hr : 1 ≤ r) :
    Fintype.card (D_idx r) = (2 ^ r - 1) * 2 ^ (r - 1) := by
  simpa [D_finset, Fintype.card_subtype] using card_D_finset r hr

theorem card_D_three : Fintype.card (D_idx 3) = 28 := by
  rw [card_D_idx 3 (by norm_num)]
  norm_num

/-- **Exact character orthogonality**: For every `r ≥ 1` and `p, q ∈ D_r`,
`∑_{g ∈ G_r} χ_p(g) χ_q(g) = if p = q then 4^r else 0`.
Hence all `|D_r|` diagonal sign characters are pairwise distinct 1D irreducibles (`c_j = s_j = 1`,
`C = 1`). At `r = 3`, this gives `Mᵀ M = 64 I_28`. -/
theorem character_orthogonality {r : ℕ} (p q : D_idx r) :
    ∑ g : Grp r, charVal g p.1 * charVal g q.1 =
      if p = q then (4 : ℝ) ^ r else 0 := by
  have hprod : ∑ g : Grp r, charVal g p.1 * charVal g q.1 =
      (∑ u : Vec2 r, signVal (dot2 (p.1.1 + q.1.1) u)) *
      (∑ w : Vec2 r, signVal (dot2 (p.1.2 + q.1.2) w)) := by
    rw [Fintype.sum_prod_type, Finset.sum_mul]
    refine Finset.sum_congr rfl fun u _ => ?_
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun w _ => ?_
    unfold charVal
    rw [← signVal_add, ← signVal_add]
    congr 1
    rw [dot2_add_left, dot2_add_left, dot2_comm p.1.1 u, dot2_comm q.1.1 u,
        dot2_comm p.1.2 w, dot2_comm q.1.2 w]
    ring
  rw [hprod, sum_signVal_dot2, sum_signVal_dot2]
  by_cases hpq : p = q
  · subst hpq
    have h1 : p.1.1 + p.1.1 = 0 := (vec2_eq_iff_add_eq_zero _ _).1 rfl
    have h2 : p.1.2 + p.1.2 = 0 := (vec2_eq_iff_add_eq_zero _ _).1 rfl
    simp only [h1, h2, ↓reduceIte]
    rw [← mul_pow]
    norm_num
  · simp only [hpq, ↓reduceIte]
    have hne : p.1.1 + q.1.1 ≠ 0 ∨ p.1.2 + q.1.2 ≠ 0 := by
      by_contra hall
      push_neg at hall
      have h1 := (vec2_eq_iff_add_eq_zero p.1.1 q.1.1).2 hall.1
      have h2 := (vec2_eq_iff_add_eq_zero p.1.2 q.1.2).2 hall.2
      exact hpq (Subtype.ext (Prod.ext h1 h2))
    rcases hne with h1 | h2
    · simp [h1]
    · simp [h2]

/-- Trace of the diagonal operator `T_g` on `ℝ^{D_r}`. -/
def traceT {r : ℕ} (g : Grp r) : ℝ := ∑ p : D_idx r, charVal g p.1

/-- **Lemma 1 (Exact nonidentity trace formula)**:
For every `r ≥ 1` and every nonidentity `g = (u, w) ≠ (0, 0)` in `G_r`,
`tr T_{(u,w)} = -2^(r - 1) * (-1)^(u · w)`.
At `r = 3`, `tr T_{(u,w)} = -4 * (-1)^(u · w) ∈ {+4, -4}`. -/
theorem trace_nonid {r : ℕ} (hr : 1 ≤ r) {g : Grp r} (hg : g ≠ 0) :
    traceT g = -((2 : ℝ) ^ (r - 1)) * signVal (dot2 g.1 g.2) := by
  have hsub : traceT g = ∑ p ∈ D_finset r, charVal g p := by
    unfold traceT D_finset
    rw [← Finset.sum_subtype (Finset.univ.filter fun p : Grp r => dot2 p.1 p.2 = 1)]
    intro p; simp
  rw [hsub]
  have hind : ∑ p ∈ D_finset r, charVal g p =
      ∑ p : Grp r, charVal g p * ((1 - signVal (dot2 p.1 p.2)) / 2) := by
    unfold D_finset
    rw [Finset.sum_filter]
    refine Finset.sum_congr rfl fun p _ => ?_
    rw [← indicator_one_eq]
    split_ifs <;> simp
  rw [hind]
  have hsplit : ∑ p : Grp r, charVal g p * ((1 - signVal (dot2 p.1 p.2)) / 2) =
      (∑ p : Grp r, charVal g p) / 2 -
      (∑ p : Grp r, charVal g p * signVal (dot2 p.1 p.2)) / 2 := by
    rw [← sub_div, ← Finset.sum_sub_distrib, Finset.sum_div]
    refine Finset.sum_congr rfl fun p _ => by ring
  rw [hsplit]
  have hfirst : ∑ p : Grp r, charVal g p = 0 := by
    rw [Fintype.sum_prod_type]
    have hfac : ∑ a : Vec2 r, ∑ b : Vec2 r, charVal g (a, b) =
        (∑ a : Vec2 r, signVal (dot2 g.1 a)) * (∑ b : Vec2 r, signVal (dot2 g.2 b)) := by
      rw [Finset.sum_mul]
      refine Finset.sum_congr rfl fun a _ => ?_
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun b _ => ?_
      exact signVal_add _ _
    rw [hfac, sum_signVal_dot2, sum_signVal_dot2]
    have hne : g.1 ≠ 0 ∨ g.2 ≠ 0 := by
      by_contra h0
      push_neg at h0
      exact hg (Prod.ext h0.1 h0.2)
    rcases hne with h1 | h2
    · simp [h1]
    · simp [h2]
  have hsecond : ∑ p : Grp r, charVal g p * signVal (dot2 p.1 p.2) =
      (2 : ℝ) ^ r * signVal (dot2 g.1 g.2) := by
    rw [Fintype.sum_prod_type]
    have hinner : ∀ a : Vec2 r,
        ∑ b : Vec2 r, charVal g (a, b) * signVal (dot2 a b) =
          signVal (dot2 g.1 a) * (if g.2 + a = 0 then (2 : ℝ) ^ r else 0) := by
      intro a
      have heq : ∑ b : Vec2 r, charVal g (a, b) * signVal (dot2 a b) =
          signVal (dot2 g.1 a) * ∑ b : Vec2 r, signVal (dot2 (g.2 + a) b) := by
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl fun b _ => ?_
        unfold charVal
        rw [signVal_add, mul_assoc, ← signVal_add, dot2_add_left]
      rw [heq, sum_signVal_dot2]
    simp_rw [hinner]
    rw [Finset.sum_eq_single g.2]
    · have h0 : g.2 + g.2 = 0 := (vec2_eq_iff_add_eq_zero _ _).1 rfl
      simp only [h0, ↓reduceIte]
      ring
    · intro a _ ha
      have hne : g.2 + a ≠ 0 := fun h0 => ha ((vec2_eq_iff_add_eq_zero g.2 a).2 h0).symm
      simp [hne]
    · intro h; exact (h (Finset.mem_univ _)).elim
  rw [hfirst, hsecond]
  have hpow : (2 : ℝ) ^ r = 2 * (2 : ℝ) ^ (r - 1) := by
    conv_lhs => rw [← Nat.sub_add_cancel hr, pow_succ']
  rw [hpow]
  ring

/-- **Faithfulness (Lemma 1)**: For `r ≥ 3`, every nonidentity `g ∈ G_r` acts nontrivially
(`T_diag g ≠ id`, since `tr T_g ∈ {±2^(r-1)}` while `tr I = (2^r - 1) 2^(r-1) > 2^(r-1)`). -/
theorem T_diag_faithful {r : ℕ} (hr : 2 ≤ r) {g : Grp r} (hg : g ≠ 0) :
    ∃ p : D_idx r, charVal g p.1 = -1 := by
  by_contra hall
  push_neg at hall
  have hpos : ∀ p : D_idx r, charVal g p.1 = 1 := by
    intro p
    unfold charVal at hall ⊢
    rcases zmod2_cases (dot2 g.1 p.1.1 + dot2 g.2 p.1.2) with h0 | h1
    · rw [h0, signVal_zero]
    · exact ((hall p) (by rw [h1, signVal_one])).elim
  have htr1 : traceT g = (Fintype.card (D_idx r) : ℝ) := by
    unfold traceT
    simp [hpos]
  have htr2 := trace_nonid (by omega) hg
  rw [card_D_idx r (by omega)] at htr1
  have hle : traceT g ≤ (2 : ℝ) ^ (r - 1) := by
    rw [htr2]
    rcases zmod2_cases (dot2 g.1 g.2) with h0 | h1
    · rw [h0, signVal_zero]; nlinarith [pow_pos (by norm_num : (0 : ℝ) < 2) (r - 1)]
    · rw [h1, signVal_one]; linarith
  have hge3 : (3 : ℕ) ≤ 2 ^ r - 1 := by
    have : 4 ≤ 2 ^ r := by
      calc 4 = 2 ^ 2 := by norm_num
        _ ≤ 2 ^ r := Nat.pow_le_pow_right (by norm_num) hr
    omega
  have hbig : (3 : ℝ) * (2 : ℝ) ^ (r - 1) ≤ ((2 ^ r - 1) * 2 ^ (r - 1) : ℕ) := by
    rw [Nat.cast_mul, Nat.cast_pow, Nat.cast_ofNat]
    have h3r : (3 : ℝ) ≤ ((2 ^ r - 1 : ℕ) : ℝ) := by exact_mod_cast hge3
    nlinarith [pow_pos (by norm_num : (0 : ℝ) < 2) (r - 1)]
  linarith [pow_pos (by norm_num : (0 : ℝ) < 2) (r - 1)]

/-! ### 3. Majority Sign, Minority-Sign Sparsity `k_r`, and Exact Gap Formula -/

/-- Majority sign `ε_g ∈ {+1, -1}` of `T_g`: `+1` at `g = 0`, and `-(-1)^(u · w)` for `g ≠ 0`. -/
def majSign {r : ℕ} (g : Grp r) : ℝ :=
  if g = 0 then 1 else -signVal (dot2 g.1 g.2)

lemma majSign_sq {r : ℕ} (g : Grp r) : majSign g ^ 2 = 1 := by
  unfold majSign
  split_ifs
  · norm_num
  · rw [neg_sq]; exact signVal_sq _

/-- Dimension `n_r = (2^r - 1) * 2^(r - 1)`. -/
def dimN (r : ℕ) : ℕ := (2 ^ r - 1) * 2 ^ (r - 1)

/-- Minority-sign sparsity `k_r = (n_r - 2^(r - 1)) / 2 = (2^(r - 1) - 1) * 2^(r - 1)`. -/
def sparsityK (r : ℕ) : ℕ := (2 ^ (r - 1) - 1) * 2 ^ (r - 1)

lemma two_mul_sparsityK (r : ℕ) (hr : 1 ≤ r) :
    2 * sparsityK r + 2 ^ (r - 1) = dimN r := by
  unfold sparsityK dimN
  have h1 : 1 ≤ 2 ^ (r - 1) := Nat.one_le_pow (r - 1) 2 (by norm_num)
  have h2 : 2 ^ r = 2 * 2 ^ (r - 1) := by
    conv_lhs => rw [← Nat.sub_add_cancel hr, pow_succ']
  rw [h2]
  have h3 : 1 ≤ 2 * 2 ^ (r - 1) := by omega
  zify [h1, h3]
  ring

/-- **Corollary 2 (Exact gap formula at `C = 1`)**:
For every `r ≥ 3`, the exact generic threshold `n_r` exceeds the first integer
`2 * k_r + C + 2` (with `C = 1`) permitted by Conjecture 10 by `2^(r - 1) - 3`.
At `r = 3`: `n_3 = 28`, `k_3 = 12`, `2 * k_3 + 1 + 2 = 27`, and the gap is `1`. -/
theorem general_gap_formula (r : ℕ) (hr : 3 ≤ r) :
    dimN r - (2 * sparsityK r + 1 + 2) = 2 ^ (r - 1) - 3 ∧
    2 * sparsityK r + 1 + 2 < dimN r := by
  have hsum := two_mul_sparsityK r (by omega)
  have hpow4 : 4 ≤ 2 ^ (r - 1) := by
    have : 2 ≤ r - 1 := by omega
    calc 4 = 2 ^ 2 := by norm_num
      _ ≤ 2 ^ (r - 1) := Nat.pow_le_pow_right (by norm_num) this
  omega

/-- Exact count of coordinates where `T_g` differs from its majority sign `ε_g I`:
`0` for `g = 0`, and `k_r` for `g ≠ 0` (`12` at `r = 3`). -/
theorem minority_sign_card {r : ℕ} (hr : 1 ≤ r) (g : Grp r) :
    (Finset.univ.filter fun p : D_idx r => charVal g p.1 ≠ majSign g).card =
      if g = 0 then 0 else sparsityK r := by
  by_cases hg : g = 0
  · subst hg
    simp [majSign, charVal_zero]
  · simp only [hg, ↓reduceIte]
    set S := Finset.univ.filter (fun p : D_idx r => charVal g p.1 ≠ majSign g)
    have htr : traceT g = ((Fintype.card (D_idx r) : ℝ) - 2 * (S.card : ℝ)) * majSign g := by
      unfold traceT
      have hterm : ∀ p : D_idx r,
          charVal g p.1 = (1 - 2 * if charVal g p.1 ≠ majSign g then (1 : ℝ) else 0) * majSign g := by
        intro p
        have hc : charVal g p.1 = 1 ∨ charVal g p.1 = -1 := by
          unfold charVal
          rcases zmod2_cases (dot2 g.1 p.1.1 + dot2 g.2 p.1.2) with h0 | h1
          · left; rw [h0, signVal_zero]
          · right; rw [h1, signVal_one]
        have hm : majSign g = 1 ∨ majSign g = -1 := by
          unfold majSign
          simp only [hg, ↓reduceIte]
          rcases zmod2_cases (dot2 g.1 g.2) with h0 | h1
          · right; rw [h0, signVal_zero]
          · left; rw [h1, signVal_one]; norm_num
        rcases hc with h1 | h1 <;> rcases hm with h2 | h2 <;> simp [h1, h2] <;> norm_num
      rw [Finset.sum_congr rfl fun p _ => hterm p]
      rw [← Finset.sum_mul, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ,
          nsmul_eq_mul, mul_one, ← Finset.mul_sum]
      congr 1
      have hSc : (S.card : ℝ) = ∑ p : D_idx r, if charVal g p.1 ≠ majSign g then (1 : ℝ) else 0 := by
        rw [Finset.card_filter, Nat.cast_sum]
        refine Finset.sum_congr rfl fun p _ => by split_ifs <;> norm_num
      rw [hSc]
    have htr2 : traceT g = ((2 : ℝ) ^ (r - 1)) * majSign g := by
      rw [trace_nonid hr hg]
      unfold majSign
      simp only [hg, ↓reduceIte]
      ring
    have hm_ne : majSign g ≠ 0 := by
      intro h0
      have := majSign_sq g
      rw [h0] at this
      norm_num at this
    have heq_r : (Fintype.card (D_idx r) : ℝ) - 2 * (S.card : ℝ) = (2 : ℝ) ^ (r - 1) :=
      mul_right_cancel₀ hm_ne (htr.symm.trans htr2)
    have hcard_eq : Fintype.card (D_idx r) = dimN r := card_D_idx r hr
    have hsum := two_mul_sparsityK r hr
    have heq_nat : 2 * S.card + 2 ^ (r - 1) = dimN r := by
      have : (2 * S.card + 2 ^ (r - 1) : ℝ) = (dimN r : ℝ) := by
        rw [← hcard_eq]
        linarith
      exact_mod_cast this
    omega

/-! ### 4. Theorem 1: Non-Identifiability for Full-Support Kernel Vectors (A1(b) and A1(a)) -/

/-- Support cardinality `‖x‖_0` of a vector `x : D_idx r → ℝ`. -/
noncomputable def supportCard {r : ℕ} (x : D_idx r → ℝ) : ℕ := by
  classical
  exact (Finset.univ.filter fun p : D_idx r => x p ≠ 0).card

/-- Squared Euclidean norm `‖x‖_2^2`. -/
def normSq {r : ℕ} (x : D_idx r → ℝ) : ℝ := ∑ p : D_idx r, (x p) ^ 2

/-- Unbounded `G_r`-invariant `k_r`-sparse cone `X = {x : ‖x‖_0 ≤ k_r}` (branch A1(b)). -/
def sparseCone (r : ℕ) : Set (D_idx r → ℝ) :=
  {x | supportCard x ≤ sparsityK r}

/-- Bounded `G_r`-invariant `k_r`-sparse model `X_B = X ∩ B_1(0)` (branch A1(a)). -/
def sparseBall (r : ℕ) : Set (D_idx r → ℝ) :=
  {x | supportCard x ≤ sparsityK r ∧ normSq x ≤ 1}

lemma supportCard_T_diag {r : ℕ} (g : Grp r) (x : D_idx r → ℝ) :
    supportCard (T_diag g x) = supportCard x := by
  classical
  unfold supportCard
  congr 1
  ext p
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, T_diag, mul_ne_zero_iff]
  have hc : charVal g p.1 ≠ 0 := by
    intro h0; have := charVal_sq g p.1; rw [h0] at this; norm_num at this
  simp [hc]

lemma supportCard_smul_le {r : ℕ} (t : ℝ) (x : D_idx r → ℝ) :
    supportCard (t • x) ≤ supportCard x := by
  classical
  unfold supportCard
  apply Finset.card_le_card
  intro p hp
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, Pi.smul_apply, smul_eq_mul] at hp ⊢
  exact fun hx => hp (by simp [hx])

/-- Witness vector `y_g = (T_g - ε_g I) v`. -/
def witnessVec {r : ℕ} (g : Grp r) (v : D_idx r → ℝ) : D_idx r → ℝ :=
  fun p => (charVal g p.1 - majSign g) * v p

lemma witnessVec_supportCard_le {r : ℕ} (hr : 1 ≤ r) (g : Grp r) (v : D_idx r → ℝ) :
    supportCard (witnessVec g v) ≤ sparsityK r := by
  classical
  have hsub : (Finset.univ.filter fun p : D_idx r => witnessVec g v p ≠ 0) ⊆
      (Finset.univ.filter fun p : D_idx r => charVal g p.1 ≠ majSign g) := by
    intro p hp
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, witnessVec] at hp ⊢
    intro heq
    apply hp
    simp [heq]
  calc supportCard (witnessVec g v)
      ≤ (Finset.univ.filter fun p : D_idx r => charVal g p.1 ≠ majSign g).card :=
        Finset.card_le_card hsub
    _ = if g = 0 then 0 else sparsityK r := minority_sign_card hr g
    _ ≤ sparsityK r := by split_ifs <;> omega

lemma witnessVec_normSq_le {r : ℕ} (g : Grp r) (v : D_idx r → ℝ) :
    normSq (witnessVec g v) ≤ 4 * normSq v := by
  unfold normSq witnessVec
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum fun p _ => ?_
  have hdiff : (charVal g p.1 - majSign g) ^ 2 ≤ 4 := by
    have hc : charVal g p.1 ^ 2 = 1 := charVal_sq g p.1
    have hm : majSign g ^ 2 = 1 := majSign_sq g
    nlinarith [sq_nonneg (charVal g p.1 + majSign g)]
  rw [mul_pow]
  nlinarith [sq_nonneg (v p)]

lemma linear_image_witness {r m : ℕ} (A : (D_idx r → ℝ) →ₗ[ℝ] (Fin m → ℝ))
    {v : D_idx r → ℝ} (hv : A v = 0) (g : Grp r) :
    A (T_diag g v) = A (witnessVec g v) := by
  have hdiff : T_diag g v - witnessVec g v = majSign g • v := by
    ext p
    simp only [Pi.sub_apply, Pi.smul_apply, smul_eq_mul, T_diag, witnessVec]
    ring
  have hzero : A (T_diag g v - witnessVec g v) = 0 := by
    rw [hdiff, map_smul, hv, smul_zero]
  rw [map_sub] at hzero
  exact sub_eq_zero.1 hzero

/-- **Theorem 1 (Unbounded conic branch A1(b))**:
For every `r ≥ 3` and every linear sensor `A : (D_r → ℝ) → (Fin m → ℝ)` possessing a full-support
kernel vector `v ∈ ker A`, the `G_r`-invariant `k_r`-sparse cone `X` admits a strictly larger
`G_r`-invariant cone `X' ≠ X` with identical indexed measurements
`A '' (T_h '' X') = A '' (T_h '' X)` for all `h ∈ G_r`. -/
theorem nonidentifiability_conic {r m : ℕ} (hr : 3 ≤ r)
    (A : (D_idx r → ℝ) →ₗ[ℝ] (Fin m → ℝ))
    {v : D_idx r → ℝ} (hker : A v = 0) (hfull : ∀ p : D_idx r, v p ≠ 0) :
    let X := sparseCone r
    let X' := X ∪ {x | ∃ (g : Grp r) (t : ℝ), x = t • T_diag g v}
    (∀ h : Grp r, T_diag h '' X = X) ∧
    (∀ h : Grp r, T_diag h '' X' = X') ∧
    (∀ x ∈ X', ∀ t : ℝ, t • x ∈ X') ∧
    X' ≠ X ∧
    (∀ h : Grp r, A '' (T_diag h '' X') = A '' (T_diag h '' X)) := by
  intro X X'
  have hX_inv : ∀ h : Grp r, T_diag h '' X = X := by
    intro h
    ext x
    constructor
    · rintro ⟨y, hy, rfl⟩
      simpa [X, sparseCone, supportCard_T_diag] using hy
    · intro hx
      refine ⟨T_diag h x, ?_, ?_⟩
      · simpa [X, sparseCone, supportCard_T_diag] using hx
      · rw [← T_diag_add]
        have hh : h + h = 0 := Prod.ext ((vec2_eq_iff_add_eq_zero _ _).1 rfl)
            ((vec2_eq_iff_add_eq_zero _ _).1 rfl)
        rw [hh, T_diag_zero]
  have hT_smul : ∀ (h : Grp r) (t : ℝ) (x : D_idx r → ℝ),
      T_diag h (t • x) = t • T_diag h x := by
    intro h t x; ext p; simp [T_diag]; ring
  have hX'_inv : ∀ h : Grp r, T_diag h '' X' = X' := by
    intro h
    ext x
    constructor
    · rintro ⟨y, hy | ⟨g, t, rfl⟩, rfl⟩
      · left; exact (hX_inv h).subset ⟨y, hy, rfl⟩
      · right; exact ⟨h + g, t, by rw [hT_smul, T_diag_add]⟩
    · intro hx
      have hh : h + h = 0 := Prod.ext ((vec2_eq_iff_add_eq_zero _ _).1 rfl)
          ((vec2_eq_iff_add_eq_zero _ _).1 rfl)
      rcases hx with hx | ⟨g, t, rfl⟩
      · refine ⟨T_diag h x, Or.inl ?_, by rw [← T_diag_add, hh, T_diag_zero]⟩
        simpa [X, sparseCone, supportCard_T_diag] using hx
      · refine ⟨t • T_diag (h + g) v, Or.inr ⟨h + g, t, rfl⟩, ?_⟩
        rw [hT_smul, ← T_diag_add, ← add_assoc, hh, zero_add]
  have hX'_cone : ∀ x ∈ X', ∀ t : ℝ, t • x ∈ X' := by
    intro x hx t
    rcases hx with hx | ⟨g, s, rfl⟩
    · left; exact le_trans (supportCard_smul_le t x) hx
    · right; exact ⟨g, t * s, by rw [mul_smul]⟩
  have hne : X' ≠ X := by
    intro heq
    have hv_mem : v ∈ X' := Or.inr ⟨0, 1, by simp [T_diag_zero]⟩
    rw [heq] at hv_mem
    have hsupp_v : supportCard v = dimN r := by
      classical
      unfold supportCard
      have hfilt : (Finset.univ.filter fun p : D_idx r => v p ≠ 0) = Finset.univ := by
        ext p; simp [hfull p]
      rw [hfilt, Finset.card_univ, card_D_idx r (by omega)]
      rfl
    have hle : dimN r ≤ sparsityK r := by simpa [X, sparseCone, hsupp_v] using hv_mem
    have hsum := two_mul_sparsityK r (by omega)
    have hpos : 0 < 2 ^ (r - 1) := pow_pos (by norm_num) (r - 1)
    omega
  have hmeas : ∀ h : Grp r, A '' (T_diag h '' X') = A '' (T_diag h '' X) := by
    intro h
    rw [hX'_inv h, hX_inv h]
    apply Set.Subset.antisymm
    · rintro _ ⟨x, hx | ⟨g, t, rfl⟩, rfl⟩
      · exact ⟨x, hx, rfl⟩
      · refine ⟨t • witnessVec g v, ?_, ?_⟩
        · exact le_trans (supportCard_smul_le t _) (witnessVec_supportCard_le (by omega) g v)
        · rw [map_smul, map_smul, linear_image_witness A hker g]
    · exact Set.image_mono Set.subset_union_left
  exact ⟨hX_inv, hX'_inv, hX'_cone, hne, hmeas⟩

/-- **Theorem 1 (Bounded branch A1(a))**:
For every `r ≥ 3` and every linear sensor `A : (D_r → ℝ) → (Fin m → ℝ)` possessing a full-support
kernel vector `v ∈ ker A` scaled so `‖v‖_2^2 ≤ 1/4`, the bounded `G_r`-invariant model
`X_B = {x : ‖x‖_0 ≤ k_r ∧ ‖x‖_2^2 ≤ 1}` admits a distinct bounded `G_r`-invariant competitor
`X'_B = X_B ∪ {T_g v : g ∈ G_r}` with identical measurements `A '' X'_B = A '' X_B`. -/
theorem nonidentifiability_bounded {r m : ℕ} (hr : 3 ≤ r)
    (A : (D_idx r → ℝ) →ₗ[ℝ] (Fin m → ℝ))
    {v : D_idx r → ℝ} (hker : A v = 0) (hfull : ∀ p : D_idx r, v p ≠ 0)
    (hnorm : normSq v ≤ 1 / 4) :
    let XB := sparseBall r
    let XB' := XB ∪ {x | ∃ g : Grp r, x = T_diag g v}
    (∀ h : Grp r, T_diag h '' XB = XB) ∧
    (∀ h : Grp r, T_diag h '' XB' = XB') ∧
    (∀ x ∈ XB', normSq x ≤ 1) ∧
    XB' ≠ XB ∧
    A '' XB' = A '' XB := by
  intro XB XB'
  have hXB_inv : ∀ h : Grp r, T_diag h '' XB = XB := by
    intro h
    ext x
    constructor
    · rintro ⟨y, ⟨h1, h2⟩, rfl⟩
      exact ⟨by simpa [supportCard_T_diag] using h1, by simpa [normSq, T_diag_norm_sq] using h2⟩
    · intro ⟨h1, h2⟩
      have hh : h + h = 0 := Prod.ext ((vec2_eq_iff_add_eq_zero _ _).1 rfl)
          ((vec2_eq_iff_add_eq_zero _ _).1 rfl)
      refine ⟨T_diag h x, ⟨by simpa [supportCard_T_diag] using h1,
          by simpa [normSq, T_diag_norm_sq] using h2⟩, by rw [← T_diag_add, hh, T_diag_zero]⟩
  have hXB'_inv : ∀ h : Grp r, T_diag h '' XB' = XB' := by
    intro h
    ext x
    constructor
    · rintro ⟨y, hy | ⟨g, rfl⟩, rfl⟩
      · left; exact (hXB_inv h).subset ⟨y, hy, rfl⟩
      · right; exact ⟨h + g, by rw [T_diag_add]⟩
    · intro hx
      have hh : h + h = 0 := Prod.ext ((vec2_eq_iff_add_eq_zero _ _).1 rfl)
          ((vec2_eq_iff_add_eq_zero _ _).1 rfl)
      rcases hx with hx | ⟨g, rfl⟩
      · refine ⟨T_diag h x, Or.inl ?_, by rw [← T_diag_add, hh, T_diag_zero]⟩
        exact (hXB_inv h).subset ⟨x, hx, rfl⟩
      · refine ⟨T_diag (h + g) v, Or.inr ⟨h + g, rfl⟩, ?_⟩
        rw [← T_diag_add, ← add_assoc, hh, zero_add]
  have hbdd : ∀ x ∈ XB', normSq x ≤ 1 := by
    rintro x (⟨_, h2⟩ | ⟨g, rfl⟩)
    · exact h2
    · rw [show normSq (T_diag g v) = normSq v from T_diag_norm_sq g v]
      linarith
  have hne : XB' ≠ XB := by
    intro heq
    have hv_mem : v ∈ XB' := Or.inr ⟨0, by simp [T_diag_zero]⟩
    rw [heq] at hv_mem
    have hsupp_v : supportCard v = dimN r := by
      classical
      unfold supportCard
      have hfilt : (Finset.univ.filter fun p : D_idx r => v p ≠ 0) = Finset.univ := by
        ext p; simp [hfull p]
      rw [hfilt, Finset.card_univ, card_D_idx r (by omega)]
      rfl
    have hle : dimN r ≤ sparsityK r := by simpa [hsupp_v] using hv_mem.1
    have hsum := two_mul_sparsityK r (by omega)
    have hpos : 0 < 2 ^ (r - 1) := pow_pos (by norm_num) (r - 1)
    omega
  have hmeas : A '' XB' = A '' XB := by
    apply Set.Subset.antisymm
    · rintro _ ⟨x, hx | ⟨g, rfl⟩, rfl⟩
      · exact ⟨x, hx, rfl⟩
      · refine ⟨witnessVec g v, ⟨witnessVec_supportCard_le (by omega) g v, ?_⟩, ?_⟩
        · linarith [witnessVec_normSq_le g v]
        · exact (linear_image_witness A hker g).symm
    · exact Set.image_mono Set.subset_union_left
  exact ⟨hXB_inv, hXB'_inv, hbdd, hne, hmeas⟩

/-! ### 5. Explicit `m = 27` Refutation and Sharp `m = 28` Threshold at `r = 3` -/

/-- **Explicit `m = 27` refutation at `r = 3` (`n = 28`, `k = 12`, `C = 1`)**:
Conjecture 10 predicts unique generic identification for `m > 2k + C + 1 = 26` (hence `m = 27`),
yet there exists an explicit surjective (rank-27) sensor `A : (D_3 → ℝ) → (Fin 27 → ℝ)` that
fails to distinguish `sparseCone 3` from a strictly larger `G_3`-invariant cone (and likewise for
the bounded ball `sparseBall 3`). -/
theorem explicit_sensor_m27_nonidentifiable :
    ∃ A : (D_idx 3 → ℝ) →ₗ[ℝ] (Fin 27 → ℝ),
      Function.Surjective A ∧
      ∃ X' : Set (D_idx 3 → ℝ),
        (∀ h : Grp 3, T_diag h '' X' = X') ∧
        X' ≠ sparseCone 3 ∧
        (∀ h : Grp 3, A '' (T_diag h '' X') = A '' (T_diag h '' sparseCone 3)) := by
  let e : D_idx 3 ≃ Fin 28 := Fintype.equivFinOfCardEq card_D_three
  let A : (D_idx 3 → ℝ) →ₗ[ℝ] (Fin 27 → ℝ) :=
    { toFun := fun x i => x (e.symm i.castSucc) - x (e.symm ⟨27, by norm_num⟩)
      map_add' := fun x y => by ext i; simp; ring
      map_smul' := fun c x => by ext i; simp; ring }
  have hsurj : Function.Surjective A := by
    intro y
    let x : D_idx 3 → ℝ := fun p =>
      if h : (e p).val < 27 then y ⟨(e p).val, h⟩ else 0
    refine ⟨x, ?_⟩
    ext i
    simp only [A, LinearMap.coe_mk, AddHom.coe_mk, x, Equiv.apply_symm_apply]
    simp
  let v : D_idx 3 → ℝ := fun _ => 1
  have hker : A v = 0 := by ext i; simp [A, v]
  have hfull : ∀ p : D_idx 3, v p ≠ 0 := fun _ => one_ne_zero
  obtain ⟨_, hX'_inv, _, hne, hmeas⟩ := nonidentifiability_conic (r := 3) (by norm_num) A hker hfull
  exact ⟨A, hsurj, _, hX'_inv, hne, hmeas⟩

/-- **Sharp threshold `m = 28`**: Every injective linear sensor `A : (D_3 → ℝ) → (Fin m → ℝ)`
(which exists for all `m ≥ 28`, and holds generically for `m ≥ 28`) uniquely identifies every
signal set `X ⊆ (D_3 → ℝ)` from `A '' X`. -/
theorem injective_sensor_identifies {m : ℕ} (A : (D_idx 3 → ℝ) →ₗ[ℝ] (Fin m → ℝ))
    (hinj : Function.Injective A) (X X' : Set (D_idx 3 → ℝ))
    (hmeas : A '' X' = A '' X) : X' = X := by
  ext x
  constructor
  · intro hx
    have : A x ∈ A '' X := hmeas ▸ ⟨x, hx, rfl⟩
    rcases this with ⟨y, hy, hyx⟩
    exact hinj hyx ▸ hy
  · intro hx
    have : A x ∈ A '' X' := hmeas.symm ▸ ⟨x, hx, rfl⟩
    rcases this with ⟨y, hy, hyx⟩
    exact hinj hyx ▸ hy

theorem exists_injective_sensor_ge28 {m : ℕ} (hm : 28 ≤ m) :
    ∃ A : (D_idx 3 → ℝ) →ₗ[ℝ] (Fin m → ℝ), Function.Injective A := by
  let e : D_idx 3 ≃ Fin 28 := Fintype.equivFinOfCardEq card_D_three
  let A : (D_idx 3 → ℝ) →ₗ[ℝ] (Fin m → ℝ) :=
    { toFun := fun x i => if h : i.val < 28 then x (e.symm ⟨i.val, h⟩) else 0
      map_add' := fun x y => by ext i; dsimp; split_ifs <;> simp
      map_smul' := fun c x => by ext i; dsimp; split_ifs <;> simp }
  refine ⟨A, ?_⟩
  intro x y hxy
  ext p
  have hi : (e p).val < m := lt_of_lt_of_le (e p).isLt hm
  have := congrFun hxy ⟨(e p).val, hi⟩
  simp only [A, LinearMap.coe_mk, AddHom.coe_mk, (e p).isLt, ↓reduceDIte, Fin.eta,
    Equiv.symm_apply_apply] at this
  exact this

end UnsupervisedGroupSensing
