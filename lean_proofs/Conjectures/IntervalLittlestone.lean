import Mathlib

/-!
# Proof of Optimality of the Greedy Cutting Schedule for Randomized Littlestone Dimension of Intervals
(Miit Daga and Swarna Priya Ramu, arXiv:2611.28727, Section 7, Conjecture `conj:rl`)

We formalize in Lean 4 + Mathlib (complete kernel-verified proofs, standard foundations only) the COMPLETE proof of
Conjecture `conj:rl`, establishing both the optimality of the greedy canonical cutting schedule
for every finite multiset of interval blocks (`Theorem 1`) and the exact single-interval
recurrence `RL(Int_m) = R(m + 1)` for all `m ≥ 0` (`Theorem 2`):

1. **Dyadic Layer Decomposition (`Section 2`)**:
   - `lamTilde`: piecewise-linear interpolation `λ̃(x)` of `log₂ x` at powers of two.
   - `Q k x = max(0, x / 2^k - 1)` and `h k x = min(1, Q k x)`.
   - `h_eq_Q_sub`: exact identity `h_d(x) = Q_d(x) - 2 Q_{2d}(x)`.
   - `sum_h_eq_lamTilde`: `∑_{k < N} h_k(x) = λ̃(x)` for `x ≤ 2^N`.
   - `sum_Q_eq`: `∑_{k < N} Q_k(x) = 2(x - 1) - λ̃(x)` for `x ≤ 2^N`.
2. **Canonical Split, Canonical Forest, and Discounted Sums**:
   - `canonA`, `canonB`: canonical split `a = max(b, p - 2b)`, `a' = p - a` where
     `b = 2^(⌊log₂ p⌋ - 1)`.
   - `c_canon`: `c(a, a') = 1 + (λ̃(a) + λ̃(a')) / 2 = λ̃(p)`.
   - `sortDesc`, `D`: descending insertion sort and exact discounted sum `D(X) = ∑ 2^{-i} X_i↓`.
   - `greedy_layer_decomp`: `G(P) = ∑_{k < N} D(A_k(P))`.
   - `discRaw_le_D`, `rearrangement_cons`: placing any element first is bounded by the sorted
     discounted sum `(x + D(L)) / 2 ≤ D(x :: L)`.
3. **Packedness and Packed Majorization (`Lemmas 1–3`)**:
   - `IsPacked`: `0 ≤ x ≤ 1` for all entries and `topSum k A = min(k, A.sum)` for all `k`.
   - `single_rod_packed` (`Lemma 1`): for every `x ≥ 1` and `k ≥ 0`, `A_k([x])` is packed with
     exact total mass `Q k x`.
   - `per_layer_split_comparison` (`Lemma 2`): for every `p ≥ 2` and legal split `u + v = p`,
     `A_k([a, a'])` is packed with mass `M_k = Q k a + Q k a'`, and `A_k([u, v])` has mass
     `Q k u + Q k v ≥ M_k`.
   - `finite_abel_sum`, `topSum_append_le`, `packed_majorization` (`Lemma 3`): finite Abel
     summation and top-`k` union majorization proving `D(C ++ B) - D(C ++ A) ≤ E / 2`.
4. **Optimality of the Greedy Schedule (`Theorem 1`)**:
   - `bellmanV`: exact discounted rod-cutting Bellman recursion `V(P)` of `RL(J_S)`.
   - `greedy_optimal`: `V(P) = G(P)` for every finite list of positive rods `P`.
5. **Exact Single-Interval Recurrence (`Theorem 2`)**:
   - `R`: exact recurrence `R(1) = 0`, `R(p) = 1 - 2^{1 - p} + (R(⌊p/2⌋) + R(⌈p/2⌉)) / 2`.
   - `H_piecewise`: piecewise-linear dyadic discount curve `H(n) = 1 - 2^{1 - n}`.
   - `D_eq_H_of_packed`, `single_rod_eq_R`, `bellman_single_interval`:
     `G([p]) = R(p)` for all `p ≥ 1` and `V([m + 1]) = R(m + 1)` for all `m ≥ 0`.

Machine-checked vs prose-only:
- Machine-checked: The complete proof of Conjecture `conj:rl` (all lemmas, Theorem 1, and
  Theorem 2 of `solutions/interval_littlestone/solution.md`), purely finitary over `ℚ`.
- Prose-only (in `solutions/interval_littlestone/proof.pdf`): The reduction from the shattered
  binary query tree definition of `RL(J_S)` to the rod-cutting Bellman equation (1) (`eq:rlcut`
  of arXiv:2611.28727) and the 204,226-multiset exhaustive verification script.
-/

open BigOperators Finset

namespace IntervalLittlestone

/-! ### 1. Dyadic Layer Decomposition (`Section 2`) -/

/-- Piecewise-linear interpolation `λ̃(x)` of `log₂ x` at powers of two (for `x ≥ 1`). -/
def lamTilde (x : ℕ) : ℚ :=
  if x = 0 then 0
  else
    let t := Nat.log 2 x
    (t : ℚ) + ((x : ℚ) - (2 : ℚ) ^ t) / (2 : ℚ) ^ t

/-- Dyadic excess `Q_{2^k}(x) = max(0, x / 2^k - 1)`. -/
def Q (k x : ℕ) : ℚ :=
  max 0 ((x : ℚ) / (2 : ℚ) ^ k - 1)

/-- Dyadic scale reward `h_{2^k}(x) = min(1, Q_{2^k}(x))`. -/
def h (k x : ℕ) : ℚ :=
  min 1 (Q k x)

lemma Q_nonneg (k x : ℕ) : 0 ≤ Q k x := le_max_left _ _

lemma h_bounds (k x : ℕ) : 0 ≤ h k x ∧ h k x ≤ 1 :=
  ⟨le_min zero_le_one (Q_nonneg k x), min_le_left _ _⟩

lemma Q_eq_zero_of_le {k x : ℕ} (hx : x ≤ 2 ^ k) : Q k x = 0 := by
  unfold Q
  apply max_eq_left
  have h1 : (x : ℚ) ≤ (2 : ℚ) ^ k := by exact_mod_cast hx
  have hpos : (0 : ℚ) < (2 : ℚ) ^ k := by positivity
  rw [sub_nonpos, div_le_one hpos]
  exact h1

lemma h_eq_zero_of_le {k x : ℕ} (hx : x ≤ 2 ^ k) : h k x = 0 := by
  unfold h
  rw [Q_eq_zero_of_le hx]
  norm_num

lemma Q_eq_of_ge {k x : ℕ} (hx : 2 ^ k ≤ x) : Q k x = (x : ℚ) / (2 : ℚ) ^ k - 1 := by
  unfold Q
  apply max_eq_right
  have h1 : (2 : ℚ) ^ k ≤ (x : ℚ) := by exact_mod_cast hx
  have hpos : (0 : ℚ) < (2 : ℚ) ^ k := by positivity
  rw [sub_nonneg, one_le_div hpos]
  exact h1

lemma h_eq_one_of_ge {k x : ℕ} (hx : 2 ^ (k + 1) ≤ x) : h k x = 1 := by
  unfold h
  apply min_eq_left
  have hge : 2 ^ k ≤ x := le_trans (Nat.pow_le_pow_right (by norm_num) (Nat.le_succ k)) hx
  rw [Q_eq_of_ge hge]
  have h1 : (2 : ℚ) ^ (k + 1) ≤ (x : ℚ) := by exact_mod_cast hx
  have hpos : (0 : ℚ) < (2 : ℚ) ^ k := by positivity
  rw [le_sub_iff_add_le, le_div_iff₀ hpos]
  rw [pow_succ] at h1
  linarith

/-- Exact telescoping identity `h_k(x) = Q_k(x) - 2 Q_{k+1}(x)`. -/
theorem h_eq_Q_sub (k x : ℕ) : h k x = Q k x - 2 * Q (k + 1) x := by
  have hpos : (0 : ℚ) < (2 : ℚ) ^ k := by positivity
  by_cases h1 : x ≤ 2 ^ k
  · have h2 : x ≤ 2 ^ (k + 1) := le_trans h1 (Nat.pow_le_pow_right (by norm_num) (Nat.le_succ k))
    rw [h_eq_zero_of_le h1, Q_eq_zero_of_le h1, Q_eq_zero_of_le h2]
    ring
  · push_neg at h1
    by_cases h2 : x ≤ 2 ^ (k + 1)
    · rw [Q_eq_zero_of_le h2, mul_zero, sub_zero, Q_eq_of_ge (le_of_lt h1)]
      unfold h
      rw [Q_eq_of_ge (le_of_lt h1)]
      apply min_eq_right
      have hx2 : (x : ℚ) ≤ (2 : ℚ) ^ (k + 1) := by exact_mod_cast h2
      rw [sub_le_iff_le_add, div_le_iff₀ hpos]
      rw [pow_succ] at hx2
      linarith
    · push_neg at h2
      rw [h_eq_one_of_ge (le_of_lt h2), Q_eq_of_ge (le_of_lt h1), Q_eq_of_ge (le_of_lt h2)]
      rw [pow_succ]
      have hne : (2 : ℚ) ^ k ≠ 0 := ne_of_gt hpos
      field_simp [hne]
      ring

lemma Q_mono (k : ℕ) : Monotone (Q k) := by
  intro x y hxy
  unfold Q
  apply max_le_max le_rfl
  apply sub_le_sub_right
  exact div_le_div_of_nonneg_right (by exact_mod_cast hxy) (by positivity)

lemma h_mono (k : ℕ) : Monotone (h k) := by
  intro x y hxy
  unfold h
  exact min_le_min le_rfl (Q_mono k hxy)

/-- Exact layer decomposition of `λ̃(x)`: `∑_{k < N} h_k(x) = λ̃(x)` whenever `1 ≤ x ≤ 2^N`. -/
theorem sum_h_eq_lamTilde {x N : ℕ} (hx1 : 1 ≤ x) (hxN : x ≤ 2 ^ N) :
    ∑ k ∈ Finset.range N, h k x = lamTilde x := by
  let t := Nat.log 2 x
  have ht_le : 2 ^ t ≤ x := Nat.pow_log_le_self 2 (by omega)
  have ht_lt : x < 2 ^ (t + 1) := Nat.lt_pow_succ_log_self (by norm_num) x
  have hlam : lamTilde x = (t : ℚ) + ((x : ℚ) - (2 : ℚ) ^ t) / (2 : ℚ) ^ t := by
    unfold lamTilde
    rw [if_neg (by omega)]
  rw [hlam]
  by_cases hEq : x = 2 ^ N
  · -- `x = 2^N`, so `t = N`, all `k < N` satisfy `2^(k+1) ≤ 2^N = x`, giving `h k x = 1`.
    have htN : t = N := by
      dsimp [t]; rw [hEq, Nat.log_pow (by norm_num)]
    have hall : ∀ k ∈ Finset.range N, h k x = 1 := by
      intro k hk
      apply h_eq_one_of_ge
      rw [hEq]
      exact Nat.pow_le_pow_right (by norm_num) (Finset.mem_range.mp hk)
    rw [Finset.sum_congr rfl hall, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one,
      htN, hEq]
    push_cast
    simp
  · -- `x < 2^N`, so `t < N` (hence `t + 1 ≤ N`).
    have ht_lt_N : t < N := by
      by_contra hge
      push_neg at hge
      have : 2 ^ N ≤ 2 ^ t := Nat.pow_le_pow_right (by norm_num) hge
      omega
    have hsplit : Finset.range N = Finset.range t ∪ {t} ∪ (Finset.Ico (t + 1) N) := by
      ext k
      simp only [Finset.mem_range, Finset.mem_union, Finset.mem_singleton, Finset.mem_Ico]
      omega
    have hd1 : Disjoint (Finset.range t) {t} := by
      rw [Finset.disjoint_singleton_right, Finset.mem_range]; omega
    have hd2 : Disjoint (Finset.range t ∪ {t}) (Finset.Ico (t + 1) N) := by
      rw [Finset.disjoint_left]
      intro k hk1 hk2
      simp only [Finset.mem_union, Finset.mem_range, Finset.mem_singleton] at hk1
      simp only [Finset.mem_Ico] at hk2
      omega
    rw [hsplit, Finset.sum_union hd2, Finset.sum_union hd1, Finset.sum_singleton]
    have hlow : ∑ k ∈ Finset.range t, h k x = (t : ℚ) := by
      have hall : ∀ k ∈ Finset.range t, h k x = 1 := by
        intro k hk
        apply h_eq_one_of_ge
        exact le_trans (Nat.pow_le_pow_right (by norm_num) (Finset.mem_range.mp hk)) ht_le
      rw [Finset.sum_congr rfl hall, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one]
    have hmid : h t x = ((x : ℚ) - (2 : ℚ) ^ t) / (2 : ℚ) ^ t := by
      unfold h
      rw [Q_eq_of_ge ht_le]
      have hle1 : (x : ℚ) / (2 : ℚ) ^ t - 1 ≤ 1 := by
        have hpos : (0 : ℚ) < (2 : ℚ) ^ t := by positivity
        have hlt' : (x : ℚ) < (2 : ℚ) ^ (t + 1) := by exact_mod_cast ht_lt
        rw [sub_le_iff_le_add, div_le_iff₀ hpos]
        rw [pow_succ] at hlt'
        linarith
      rw [min_eq_right hle1]
      have hne : (2 : ℚ) ^ t ≠ 0 := by positivity
      field_simp [hne]
    have hhi : ∑ k ∈ Finset.Ico (t + 1) N, h k x = 0 := by
      refine Finset.sum_eq_zero fun k hk => h_eq_zero_of_le ?_
      have hk_ge : t + 1 ≤ k := (Finset.mem_Ico.mp hk).1
      exact le_trans (le_of_lt ht_lt) (Nat.pow_le_pow_right (by norm_num) hk_ge)
    rw [hlow, hmid, hhi, add_zero]

/-- Exact dyadic mass identity: `∑_{k < N} Q_k(x) = 2(x - 1) - λ̃(x)` whenever `1 ≤ x ≤ 2^N`. -/
theorem sum_Q_eq {x N : ℕ} (hx1 : 1 ≤ x) (hxN : x ≤ 2 ^ N) :
    ∑ k ∈ Finset.range N, Q k x = 2 * ((x : ℚ) - 1) - lamTilde x := by
  have htel : ∑ k ∈ Finset.range N, h k x =
      2 * Q 0 x - ∑ k ∈ Finset.range N, Q k x - 2 * Q N x := by
    have hsum : ∑ k ∈ Finset.range N, h k x =
        ∑ k ∈ Finset.range N, Q k x - 2 * ∑ k ∈ Finset.range N, Q (k + 1) x := by
      simp_rw [h_eq_Q_sub]
      rw [Finset.sum_sub_distrib, ← Finset.mul_sum]
    have hshift : ∑ k ∈ Finset.range N, Q (k + 1) x + Q 0 x =
        ∑ k ∈ Finset.range N, Q k x + Q N x := by
      have h1 := Finset.sum_range_succ' (fun k => Q k x) N
      have h2 := Finset.sum_range_succ (fun k => Q k x) N
      linarith
    linarith
  have hQ0 : Q 0 x = (x : ℚ) - 1 := by
    rw [Q_eq_of_ge (by simpa using hx1)]; simp
  have hQN : Q N x = 0 := Q_eq_zero_of_le hxN
  rw [sum_h_eq_lamTilde hx1 hxN, hQ0, hQN] at htel
  linarith

lemma lamTilde_mono : Monotone lamTilde := by
  intro x y hxy
  by_cases hx0 : x = 0
  · subst hx0
    have h0 : lamTilde 0 = 0 := rfl
    rw [h0]
    by_cases hy0 : y = 0
    · subst hy0; exact le_rfl
    · unfold lamTilde
      rw [if_neg hy0]
      have hle : (2 : ℚ) ^ Nat.log 2 y ≤ (y : ℚ) := by
        exact_mod_cast Nat.pow_log_le_self 2 hy0
      have h1 : (0 : ℚ) ≤ ((y : ℚ) - (2 : ℚ) ^ Nat.log 2 y) / (2 : ℚ) ^ Nat.log 2 y :=
        div_nonneg (by linarith) (by positivity)
      have h2 : (0 : ℚ) ≤ (Nat.log 2 y : ℚ) := Nat.cast_nonneg _
      linarith
  · have hx1 : 1 ≤ x := Nat.pos_of_ne_zero hx0
    have hy1 : 1 ≤ y := le_trans hx1 hxy
    let N := y
    have hyN : y ≤ 2 ^ N := le_of_lt (Nat.lt_two_pow_self)
    have hxN : x ≤ 2 ^ N := le_trans hxy hyN
    rw [← sum_h_eq_lamTilde hx1 hxN, ← sum_h_eq_lamTilde hy1 hyN]
    exact Finset.sum_le_sum fun k _ => h_mono k hxy

/-! ### 2. Canonical Split and Immediate Cut Reward -/

/-- Base power of two `b = 2^(⌊log₂ p⌋ - 1)` for `p ≥ 2`. -/
def canonBase (p : ℕ) : ℕ := 2 ^ (Nat.log 2 p - 1)

/-- Canonical split `(canonA p, canonB p)` of `p ≥ 2`. -/
def canonA (p : ℕ) : ℕ := max (canonBase p) (p - 2 * canonBase p)
def canonB (p : ℕ) : ℕ := p - canonA p

/-- Immediate cut reward `c(u, v) = 1 + (λ̃(u) + λ̃(v)) / 2`. -/
def cutReward (u v : ℕ) : ℚ := 1 + (lamTilde u + lamTilde v) / 2

lemma canon_bounds {p : ℕ} (hp : 2 ≤ p) :
    let b := canonBase p
    1 ≤ b ∧ 2 * b ≤ p ∧ p < 4 * b ∧
    canonA p + canonB p = p ∧
    b ≤ canonA p ∧ canonA p ≤ 2 * b ∧
    b ≤ canonB p ∧ canonB p ≤ 2 * b ∧
    canonA p < p ∧ canonB p < p ∧
    (canonA p = b ∨ canonB p = 2 * b) := by
  intro b
  have ht1 : 1 ≤ Nat.log 2 p := by
    rw [Nat.le_log_iff_pow_le (by norm_num) (by omega)]
    simpa using hp
  have hpow_le : 2 ^ Nat.log 2 p ≤ p := Nat.pow_log_le_self 2 (by omega)
  have hpow_lt : p < 2 ^ (Nat.log 2 p + 1) := Nat.lt_pow_succ_log_self (by norm_num) p
  have h2b : 2 * b = 2 ^ Nat.log 2 p := by
    dsimp [b, canonBase]
    have : Nat.log 2 p = (Nat.log 2 p - 1) + 1 := by omega
    conv_rhs => rw [this, pow_succ']
  have h4b : 4 * b = 2 ^ (Nat.log 2 p + 1) := by
    rw [pow_succ', ← h2b]; ring
  have hb1 : 1 ≤ b := Nat.one_le_pow _ 2 (by norm_num)
  have h2b_le : 2 * b ≤ p := h2b ▸ hpow_le
  have hp_lt4 : p < 4 * b := h4b ▸ hpow_lt
  have hA_def : canonA p = max b (p - 2 * b) := rfl
  have hB_def : canonB p = p - canonA p := rfl
  refine ⟨hb1, h2b_le, hp_lt4, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> omega

lemma lamTilde_on_segment {t x : ℕ} (h1 : 2 ^ t ≤ x) (h2 : x ≤ 2 ^ (t + 1)) :
    lamTilde x = (t : ℚ) + ((x : ℚ) - (2 : ℚ) ^ t) / (2 : ℚ) ^ t := by
  have hx0 : x ≠ 0 := by
    have := Nat.one_le_pow t 2 (by norm_num); omega
  rcases eq_or_lt_of_le h2 with rfl | hlt
  · unfold lamTilde
    rw [if_neg hx0, Nat.log_pow (by norm_num)]
    push_cast
    rw [pow_succ]
    have hne : (2 : ℚ) ^ t ≠ 0 := by positivity
    field_simp [hne]
    ring
  · have hlog : Nat.log 2 x = t :=
      le_antisymm (Nat.lt_succ_iff.mp ((Nat.log_lt_iff_lt_pow (by norm_num) hx0).mpr hlt))
        ((Nat.le_log_iff_pow_le (by norm_num) hx0).mpr h1)
    unfold lamTilde
    rw [if_neg hx0, hlog]

/-- Exact canonical split reward identity: `c(canonA p, canonB p) = λ̃(p)` for all `p ≥ 2`. -/
theorem cutReward_canon {p : ℕ} (hp : 2 ≤ p) :
    cutReward (canonA p) (canonB p) = lamTilde p := by
  have hb := canon_bounds hp
  let t := Nat.log 2 p - 1
  have ht1 : 1 ≤ Nat.log 2 p := by
    rw [Nat.le_log_iff_pow_le (by norm_num) (by omega)]
    simpa using hp
  have h2t : 2 ^ (t + 1) = 2 * canonBase p := by
    dsimp [t, canonBase]; rw [pow_succ']
  have hA := lamTilde_on_segment (t := t) hb.2.2.2.2.1 (h2t.symm ▸ hb.2.2.2.2.2.1)
  have hB := lamTilde_on_segment (t := t) hb.2.2.2.2.2.2.1 (h2t.symm ▸ hb.2.2.2.2.2.2.2.1)
  have hp_seg : lamTilde p = (t + 1 : ℚ) + ((p : ℚ) - (2 : ℚ) ^ (t + 1)) / (2 : ℚ) ^ (t + 1) := by
    unfold lamTilde
    rw [if_neg (by omega)]
    have hlog : Nat.log 2 p = t + 1 := by omega
    rw [hlog]
    push_cast
    rfl
  unfold cutReward
  rw [hA, hB, hp_seg]
  have hsum : (canonA p : ℚ) + (canonB p : ℚ) = (p : ℚ) := by
    exact_mod_cast hb.2.2.2.1
  have hne : (2 : ℚ) ^ t ≠ 0 := by positivity
  rw [pow_succ]
  field_simp [hne]
  linarith

/-! ### 3. Canonical Forest, Descending Insertion Sort, and Discounted Sums -/

/-- Internal nodes (measures `≥ 2`) of the canonical binary tree of a single rod `p`. -/
def canonNodes (p : ℕ) : List ℕ :=
  if h : 2 ≤ p then
    have : canonA p < p := (canon_bounds h).2.2.2.2.2.2.2.2.1
    have : canonB p < p := (canon_bounds h).2.2.2.2.2.2.2.2.2.1
    p :: (canonNodes (canonA p) ++ canonNodes (canonB p))
  else
    []
termination_by p

lemma canonNodes_of_lt_two {p : ℕ} (hp : p < 2) : canonNodes p = [] := by
  rw [canonNodes]; simp [show ¬ (2 ≤ p) by omega]

lemma canonNodes_of_ge_two {p : ℕ} (hp : 2 ≤ p) :
    canonNodes p = p :: (canonNodes (canonA p) ++ canonNodes (canonB p)) := by
  rw [canonNodes]; simp [hp]

lemma canonNodes_mem_le {p r : ℕ} (hr : r ∈ canonNodes p) : 2 ≤ r ∧ r ≤ p := by
  induction p using Nat.strong_induction_on with
  | _ p ih =>
    by_cases hp : 2 ≤ p
    · rw [canonNodes_of_ge_two hp] at hr
      simp only [List.mem_cons, List.mem_append] at hr
      have hb := canon_bounds hp
      rcases hr with rfl | hA | hB
      · exact ⟨hp, le_rfl⟩
      · have h' := ih (canonA p) hb.2.2.2.2.2.2.2.2.1 hA
        exact ⟨h'.1, le_trans h'.2 (le_of_lt hb.2.2.2.2.2.2.2.2.1)⟩
      · have h' := ih (canonB p) hb.2.2.2.2.2.2.2.2.2.1 hB
        exact ⟨h'.1, le_trans h'.2 (le_of_lt hb.2.2.2.2.2.2.2.2.2.1)⟩
    · rw [canonNodes_of_lt_two (by omega)] at hr
      simp at hr

/-- Internal nodes of the canonical forest of a rod list `P`. -/
def forestNodes (P : List ℕ) : List ℕ :=
  P.flatMap canonNodes

/-- Descending insertion into an already-descending list. -/
def insertDesc {α : Type*} [LinearOrder α] (x : α) : List α → List α
  | [] => [x]
  | y :: ys => if y ≤ x then x :: y :: ys else y :: insertDesc x ys

/-- Descending insertion sort on `List α`. -/
def sortDesc {α : Type*} [LinearOrder α] : List α → List α
  | [] => []
  | x :: xs => insertDesc x (sortDesc xs)

lemma insertDesc_perm {α : Type*} [LinearOrder α] (x : α) (L : List α) :
    (insertDesc x L).Perm (x :: L) := by
  induction L with
  | nil => simp [insertDesc]
  | cons y ys ih =>
    unfold insertDesc
    split_ifs
    · exact List.Perm.refl _
    · exact (ih.cons y).trans (List.Perm.swap x y ys)

lemma sortDesc_perm {α : Type*} [LinearOrder α] (L : List α) : (sortDesc L).Perm L := by
  induction L with
  | nil => exact List.Perm.refl _
  | cons x xs ih => exact (insertDesc_perm x _).trans (ih.cons x)

lemma insertDesc_pairwise {α : Type*} [LinearOrder α] (x : α) {L : List α}
    (hL : L.Pairwise (· ≥ ·)) : (insertDesc x L).Pairwise (· ≥ ·) := by
  induction L with
  | nil => simp [insertDesc]
  | cons y ys ih =>
    unfold insertDesc
    split_ifs with h
    · rw [List.pairwise_cons] at hL ⊢
      refine ⟨?_, List.pairwise_cons.mpr ⟨hL.1, hL.2⟩⟩
      intro z hz
      simp only [List.mem_cons] at hz
      rcases hz with rfl | hzys
      · exact h
      · exact le_trans (hL.1 z hzys) h
    · rw [List.pairwise_cons] at hL ⊢
      refine ⟨?_, ih hL.2⟩
      intro z hz
      have hz' : z ∈ x :: ys := (insertDesc_perm x ys).subset hz
      simp only [List.mem_cons] at hz'
      rcases hz' with rfl | hzys
      · exact le_of_lt (not_le.mp h)
      · exact hL.1 z hzys

lemma sortDesc_pairwise {α : Type*} [LinearOrder α] (L : List α) :
    (sortDesc L).Pairwise (· ≥ ·) := by
  induction L with
  | nil => simp [sortDesc]
  | cons x xs ih => exact insertDesc_pairwise x ih

lemma pairwise_ge_eq_of_perm {α : Type*} [LinearOrder α] :
    ∀ {L₁ L₂ : List α}, L₁.Perm L₂ → L₁.Pairwise (· ≥ ·) → L₂.Pairwise (· ≥ ·) → L₁ = L₂
  | [], L₂, hP, _, _ => (List.perm_nil.mp hP.symm).symm
  | a :: t₁, [], hP, _, _ => absurd hP.symm (by simp)
  | a :: t₁, b :: t₂, hP, h1, h2 => by
    rw [List.pairwise_cons] at h1 h2
    have hab : a ≤ b := by
      have ha : a ∈ b :: t₂ := hP.subset (List.mem_cons_self ..)
      simp only [List.mem_cons] at ha
      rcases ha with rfl | hat
      · exact le_rfl
      · exact h2.1 a hat
    have hba : b ≤ a := by
      have hb : b ∈ a :: t₁ := hP.symm.subset (List.mem_cons_self ..)
      simp only [List.mem_cons] at hb
      rcases hb with rfl | hbt
      · exact le_rfl
      · exact h1.1 b hbt
    have heq : a = b := le_antisymm hab hba
    subst heq
    congr 1
    exact pairwise_ge_eq_of_perm (List.Perm.cons_inv hP) h1.2 h2.2

lemma sortDesc_eq_self {α : Type*} [LinearOrder α] {L : List α} (hL : L.Pairwise (· ≥ ·)) :
    sortDesc L = L :=
  pairwise_ge_eq_of_perm (sortDesc_perm L) (sortDesc_pairwise L) hL

lemma sortDesc_eq_of_perm {α : Type*} [LinearOrder α] {L₁ L₂ : List α} (h : L₁.Perm L₂) :
    sortDesc L₁ = sortDesc L₂ :=
  pairwise_ge_eq_of_perm ((sortDesc_perm L₁).trans (h.trans (sortDesc_perm L₂).symm))
    (sortDesc_pairwise L₁) (sortDesc_pairwise L₂)

lemma sortDesc_map_mono {α β : Type*} [LinearOrder α] [LinearOrder β]
    {f : α → β} (hf : Monotone f) (L : List α) :
    sortDesc (L.map f) = (sortDesc L).map f := by
  have hperm : (L.map f).Perm ((sortDesc L).map f) := (sortDesc_perm L).symm.map f
  have hpw : ((sortDesc L).map f).Pairwise (· ≥ ·) := by
    rw [List.pairwise_map]
    exact (sortDesc_pairwise L).imp fun h => hf h
  rw [sortDesc_eq_of_perm hperm, sortDesc_eq_self hpw]

/-- Raw discounted sum `∑_{i=0}^{n-1} 2^{-(i+1)} L_i`. -/
def discRaw : List ℚ → ℚ
  | [] => 0
  | x :: xs => (x + discRaw xs) / 2

/-- Sorted discounted sum `D(X) = ∑_{i ≥ 1} 2^{-i} X_i↓`. -/
def D (X : List ℚ) : ℚ := discRaw (sortDesc X)

lemma D_perm {X Y : List ℚ} (h : X.Perm Y) : D X = D Y := by
  unfold D; rw [sortDesc_eq_of_perm h]

lemma discRaw_add (L : List ℕ) (f g : ℕ → ℚ) :
    discRaw (L.map (fun x => f x + g x)) = discRaw (L.map f) + discRaw (L.map g) := by
  induction L with
  | nil => simp [discRaw]
  | cons x xs ih => simp [discRaw, ih]; ring

lemma discRaw_sum (L : List ℕ) (N : ℕ) (F : ℕ → ℕ → ℚ) :
    discRaw (L.map (fun x => ∑ k ∈ Finset.range N, F k x)) =
      ∑ k ∈ Finset.range N, discRaw (L.map (F k)) := by
  induction N with
  | zero =>
    have h0 : ∀ M : List ℕ, discRaw (M.map (fun _ => (0 : ℚ))) = 0 := by
      intro M; induction M with
      | nil => rfl
      | cons _ _ ih => simp only [List.map_cons, discRaw, ih]; norm_num
    simpa using h0 L
  | succ N ih =>
    simp_rw [Finset.sum_range_succ]
    rw [discRaw_add, ih]

lemma discRaw_cons_le_insertDesc (x : ℚ) (L : List ℚ) :
    (x + discRaw L) / 2 ≤ discRaw (insertDesc x L) := by
  induction L with
  | nil => simp [insertDesc, discRaw]
  | cons y ys ih =>
    unfold insertDesc
    split_ifs with h
    · simp [discRaw]
    · have hlt : x < y := not_le.mp h
      simp only [discRaw]
      linarith

/-- Rearrangement inequality: placing `x` first is at most `D(x :: L)`. -/
theorem rearrangement_cons (x : ℚ) (L : List ℚ) : (x + D L) / 2 ≤ D (x :: L) :=
  discRaw_cons_le_insertDesc x (sortDesc L)

/-- Greedy schedule value `G(P) = D(forestNodes P |>.map lamTilde)`. -/
def greedyG (P : List ℕ) : ℚ := D ((forestNodes P).map lamTilde)

lemma perm_flatMap {α β : Type*} (f : α → List β) {L₁ L₂ : List α} (h : L₁.Perm L₂) :
    (L₁.flatMap f).Perm (L₂.flatMap f) := by
  induction h with
  | nil => exact List.Perm.refl _
  | cons a _ ih => exact ih.append_left (f a)
  | swap x y l =>
    simp only [List.flatMap_cons, ← List.append_assoc]
    exact List.Perm.append_right _ List.perm_append_comm
  | trans _ _ ih₁ ih₂ => exact ih₁.trans ih₂

lemma greedyG_perm {P₁ P₂ : List ℕ} (h : P₁.Perm P₂) : greedyG P₁ = greedyG P₂ := by
  unfold greedyG forestNodes
  exact D_perm ((perm_flatMap canonNodes h).map lamTilde)

/-- Exact layer decomposition of `G(P)` (equation (3) of `solution.md`):
`G(P) = ∑_{k < N} D((forestNodes P).map (h k))` whenever all rods in `P` satisfy `r ≤ 2^N`. -/
theorem greedy_layer_decomp {P : List ℕ} {N : ℕ} (hP : ∀ r ∈ P, r ≤ 2 ^ N) :
    greedyG P = ∑ k ∈ Finset.range N, D ((forestNodes P).map (h k)) := by
  unfold greedyG D
  rw [sortDesc_map_mono lamTilde_mono]
  have hcongr : (sortDesc (forestNodes P)).map lamTilde =
      (sortDesc (forestNodes P)).map (fun r => ∑ k ∈ Finset.range N, h k r) := by
    refine List.map_congr_left fun r hr => ?_
    have hr' : r ∈ forestNodes P := (sortDesc_perm _).subset hr
    rcases List.mem_flatMap.mp hr' with ⟨p, hp, hrp⟩
    have hbounds := canonNodes_mem_le hrp
    exact (sum_h_eq_lamTilde (by omega) (le_trans hbounds.2 (hP p hp))).symm
  rw [hcongr, discRaw_sum]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [sortDesc_map_mono (h_mono k)]

/-! ### 4. Top-`k` Sums, Finite Abel Summation, and Packed Majorization (`Lemmas 1–3`) -/

/-- Top-`k` sum `s_k(X)` of the `k` largest entries of `X`. -/
def topSum (k : ℕ) (X : List ℚ) : ℚ := ((sortDesc X).take k).sum

lemma topSum_perm {X Y : List ℚ} (h : X.Perm Y) (k : ℕ) : topSum k X = topSum k Y := by
  unfold topSum; rw [sortDesc_eq_of_perm h]

lemma topSum_zero (X : List ℚ) : topSum 0 X = 0 := by simp [topSum]

lemma topSum_ge_length {X : List ℚ} {k : ℕ} (hk : X.length ≤ k) : topSum k X = X.sum := by
  unfold topSum
  have hlen : (sortDesc X).length = X.length := (sortDesc_perm X).length_eq
  rw [List.take_of_length_le (by omega), (sortDesc_perm X).sum_eq]

lemma geom_quarter_sum (M : ℕ) :
    (1 : ℚ) / 4 + (∑ k ∈ Finset.range M, (1 : ℚ) / (2 : ℚ) ^ (k + 3)) +
      (1 : ℚ) / (2 : ℚ) ^ (M + 2) = 1 / 2 := by
  induction M with
  | zero => norm_num
  | succ m ihm =>
    rw [Finset.sum_range_succ]
    have h1 : (1 : ℚ) / (2 : ℚ) ^ (m + 3) + (1 : ℚ) / (2 : ℚ) ^ (m + 1 + 2) =
        (1 : ℚ) / (2 : ℚ) ^ (m + 2) := by
      have ha : m + 1 + 2 = m + 3 := by omega
      rw [ha, show m + 3 = (m + 2) + 1 by omega, pow_succ']
      ring
    linarith

/-- **Finite Abel summation**: for any list `X` of length `≤ M`,
`D(X) = (∑_{k=0}^{M-1} s_{k+1}(X) / 2^{k+2}) + s_M(X) / 2^{M+1}`. -/
theorem finite_abel_sum :
    ∀ (M : ℕ) (L : List ℚ), L.length ≤ M →
      discRaw L =
        (∑ k ∈ Finset.range M, ((L.take (k + 1)).sum) / (2 : ℚ) ^ (k + 2)) +
        ((L.take M).sum) / (2 : ℚ) ^ (M + 1)
  | 0, L, hL => by
    have hnil : L = [] := List.eq_nil_of_length_eq_zero (by omega)
    subst hnil; simp [discRaw]
  | M + 1, [], _ => by simp [discRaw]
  | M + 1, a :: L', hL => by
    have hL' : L'.length ≤ M := by simp at hL; omega
    have ih := finite_abel_sum M L' hL'
    simp only [discRaw]
    rw [ih, Finset.sum_range_succ']
    simp only [List.take_succ_cons, List.sum_cons, List.take_zero, List.sum_nil, add_zero]
    have hsplit_sum :
        ∑ k ∈ Finset.range M, (a + (L'.take (k + 1)).sum) / (2 : ℚ) ^ (k + 1 + 2) =
        a * (∑ k ∈ Finset.range M, (1 : ℚ) / (2 : ℚ) ^ (k + 3)) +
        (1 / 2 : ℚ) * ∑ k ∈ Finset.range M, (L'.take (k + 1)).sum / (2 : ℚ) ^ (k + 2) := by
      rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl fun k _ => ?_
      have hpow : (2 : ℚ) ^ (k + 1 + 2) = 2 * (2 : ℚ) ^ (k + 2) := by
        rw [show k + 1 + 2 = (k + 2) + 1 by omega, pow_succ']
      have hpow3 : (2 : ℚ) ^ (k + 3) = (2 : ℚ) ^ (k + 1 + 2) := by
        rw [show k + 3 = k + 1 + 2 by omega]
      rw [hpow3, hpow]
      ring
    have hgeom := geom_quarter_sum M
    rw [hsplit_sum]
    have hpow_last : (2 : ℚ) ^ (M + 1 + 1) = 2 * (2 : ℚ) ^ (M + 1) := by
      rw [show M + 1 + 1 = (M + 1) + 1 by omega, pow_succ']
    have hpow2 : (2 : ℚ) ^ (0 + 2) = 4 := by norm_num
    rw [hpow2, hpow_last]
    have htail : (a + (L'.take M).sum) / (2 * (2 : ℚ) ^ (M + 1)) =
        a * ((1 : ℚ) / (2 : ℚ) ^ (M + 2)) + (1 / 2 : ℚ) * ((L'.take M).sum / (2 : ℚ) ^ (M + 1)) := by
      rw [show M + 2 = (M + 1) + 1 by omega, pow_succ']
      ring
    rw [htail]
    have hcomb : a / 4 + (a * ∑ k ∈ Finset.range M, 1 / (2 : ℚ) ^ (k + 3)) +
        a * (1 / (2 : ℚ) ^ (M + 2)) = a / 2 := by
      calc a / 4 + (a * ∑ k ∈ Finset.range M, 1 / (2 : ℚ) ^ (k + 3)) + a * (1 / (2 : ℚ) ^ (M + 2))
          = a * (1 / 4 + (∑ k ∈ Finset.range M, 1 / (2 : ℚ) ^ (k + 3)) + 1 / (2 : ℚ) ^ (M + 2)) := by ring
        _ = a * (1 / 2) := by rw [hgeom]
        _ = a / 2 := by ring
    linarith

/-- Any sublist of length `≤ k` of a nonnegative list `X` has sum `≤ topSum k X`. -/
lemma sublist_sum_le_topSum {X S : List ℚ} (hX : ∀ x ∈ X, 0 ≤ x)
    (hsub : S.Sublist (sortDesc X)) (hlen : S.length ≤ k) :
    S.sum ≤ topSum k X := by
  -- Generalize to any descending list `L` with nonnegative entries
  have hgen : ∀ (L S : List ℚ) (k : ℕ),
      (∀ x ∈ L, 0 ≤ x) → L.Pairwise (· ≥ ·) → S.Sublist L → S.length ≤ k →
      S.sum ≤ (L.take k).sum := by
    intro L
    induction L with
    | nil =>
      intro S k _ _ hsub _
      rw [List.sublist_nil.mp hsub]; simp
    | cons a L' ih =>
      intro S k hnn hpw hsub hlen
      rw [List.pairwise_cons] at hpw
      have hnn' : ∀ x ∈ L', 0 ≤ x := fun x hx => hnn x (List.mem_cons_of_mem a hx)
      cases k with
      | zero =>
        rw [List.eq_nil_of_length_eq_zero (Nat.le_zero.mp hlen)]; simp
      | succ k' =>
        simp only [List.take_succ_cons, List.sum_cons]
        cases hsub with
        | cons _ hsub' =>
          -- `S` is a sublist of `L'` of length `≤ k' + 1`.
          cases S with
          | nil =>
            simp only [List.sum_nil]
            have h1 : 0 ≤ a := hnn a (List.mem_cons_self ..)
            have h2 : 0 ≤ (L'.take k').sum := List.sum_nonneg fun x hx => hnn' x (List.mem_of_mem_take hx)
            linarith
          | cons b S' =>
            simp only [List.length_cons] at hlen
            have hb_mem : b ∈ L' := hsub'.subset (List.mem_cons_self ..)
            have hba : b ≤ a := hpw.1 b hb_mem
            have hS'_sub : S'.Sublist L' := (List.sublist_cons_self b S').trans hsub'
            have hS'_le := ih S' k' hnn' hpw.2 hS'_sub (by omega)
            simp only [List.sum_cons]
            linarith
        | cons₂ _ hsub' =>
          rename_i S'
          simp only [List.length_cons] at hlen
          have hS'_le := ih S' k' hnn' hpw.2 hsub' (by omega)
          simp only [List.sum_cons]
          linarith
  exact hgen (sortDesc X) S k
    (fun x hx => hX x ((sortDesc_perm X).subset hx))
    (sortDesc_pairwise X) hsub hlen

/-- Dual characterization of `List.Subperm`: if `S <+~ B`, then some permutation of `S` is a sublist of `B`. -/
lemma exists_perm_sublist_of_subperm {α : Type*} :
    ∀ (B S : List α), S.Subperm B → ∃ S' : List α, S.Perm S' ∧ S'.Sublist B
  | [], S, hsp => by
    have hnil : S = [] := List.subperm_nil.mp hsp
    subst hnil
    exact ⟨[], List.Perm.refl _, List.nil_sublist _⟩
  | b :: B', S, hsp => by
    classical
    by_cases hb : b ∈ S
    · have hS_perm : S.Perm (b :: S.erase b) := List.perm_cons_erase hb
      have hsp' : (b :: S.erase b).Subperm (b :: B') := hS_perm.symm.subperm.trans hsp
      have herase : (S.erase b).Subperm B' := (List.subperm_cons b).mp hsp'
      rcases exists_perm_sublist_of_subperm B' (S.erase b) herase with ⟨S', hP, hsub⟩
      exact ⟨b :: S', hS_perm.trans (hP.cons b), hsub.cons_cons b⟩
    · have hsp' : S.Subperm B' := by
        rw [List.subperm_iff_count] at hsp ⊢
        intro x
        have hx := hsp x
        rw [List.count_cons] at hx
        by_cases hxb : x = b
        · subst hxb
          rw [List.count_eq_zero.mpr hb]; exact Nat.zero_le _
        · have hne : (b == x) = false := beq_false_of_ne (Ne.symm hxb)
          simpa [hne] using hx
      rcases exists_perm_sublist_of_subperm B' S hsp' with ⟨S', hP, hsub⟩
      exact ⟨S', hP, hsub.cons b⟩

/-- Splitting a sublist of a permutation of `A ++ B` into sublists of `A` and `B`. -/
lemma perm_sublist_split {α : Type*} {L A B S : List α}
    (hperm : L.Perm (A ++ B)) (hsub : S.Sublist L) :
    ∃ S₁ S₂ : List α, S.Perm (S₁ ++ S₂) ∧ S₁.Sublist A ∧ S₂.Sublist B := by
  have hgen : ∀ (S A B : List α), S.Subperm (A ++ B) →
      ∃ S₁ S₂ : List α, S.Perm (S₁ ++ S₂) ∧ S₁.Sublist A ∧ S₂.Sublist B := by
    intro S A
    induction A generalizing S with
    | nil =>
      intro B hsp
      rw [List.nil_append] at hsp
      rcases exists_perm_sublist_of_subperm B S hsp with ⟨S₂, hP, hsub₂⟩
      exact ⟨[], S₂, by simpa using hP, List.nil_sublist _, hsub₂⟩
    | cons a A' ih =>
      intro B hsp
      classical
      by_cases ha : a ∈ S
      · have hS_perm : S.Perm (a :: S.erase a) := List.perm_cons_erase ha
        have hsp' : (a :: S.erase a).Subperm (a :: (A' ++ B)) :=
          hS_perm.symm.subperm.trans hsp
        have herase : (S.erase a).Subperm (A' ++ B) := (List.subperm_cons a).mp hsp'
        rcases ih (S.erase a) B herase with ⟨S₁, S₂, hP, hsub₁, hsub₂⟩
        exact ⟨a :: S₁, S₂, hS_perm.trans (hP.cons a), hsub₁.cons_cons a, hsub₂⟩
      · have hsp' : S.Subperm (A' ++ B) := by
          rw [List.subperm_iff_count] at hsp ⊢
          intro x
          have hx := hsp x
          rw [List.cons_append, List.count_cons] at hx
          by_cases hxa : x = a
          · subst hxa
            rw [List.count_eq_zero.mpr ha]; exact Nat.zero_le _
          · have hne : (a == x) = false := beq_false_of_ne (Ne.symm hxa)
            simpa [hne] using hx
        rcases ih S B hsp' with ⟨S₁, S₂, hP, hsub₁, hsub₂⟩
        exact ⟨S₁, S₂, hP, hsub₁.cons a, hsub₂⟩
  exact hgen S A B (hsub.subperm.trans hperm.subperm)

/-- Top-`k` union upper bound: `∃ j ≤ k, topSum k (C ++ X) ≤ topSum j C + topSum (k - j) X`. -/
lemma topSum_append_upper {C X : List ℚ} (hC : ∀ x ∈ C, 0 ≤ x) (hX : ∀ x ∈ X, 0 ≤ x) (k : ℕ) :
    ∃ j ≤ k, topSum k (C ++ X) ≤ topSum j C + topSum (k - j) X := by
  let T := (sortDesc (C ++ X)).take k
  have hperm : (sortDesc (C ++ X)).Perm (sortDesc C ++ sortDesc X) :=
    (sortDesc_perm (C ++ X)).trans ((sortDesc_perm C).symm.append (sortDesc_perm X).symm)
  rcases perm_sublist_split hperm (List.take_sublist k (sortDesc (C ++ X))) with
    ⟨S₁, S₂, hTP, hsub₁, hsub₂⟩
  have hlen : S₁.length + S₂.length ≤ k := by
    have h1 : T.length = S₁.length + S₂.length := by
      have := hTP.length_eq; rwa [List.length_append] at this
    have h2 : T.length ≤ k := List.length_take_le k _
    omega
  refine ⟨S₁.length, by omega, ?_⟩
  have hsum_eq : topSum k (C ++ X) = S₁.sum + S₂.sum := by
    unfold topSum
    change T.sum = S₁.sum + S₂.sum
    simpa using hTP.sum_eq
  rw [hsum_eq]
  exact add_le_add
    (sublist_sum_le_topSum hC hsub₁ le_rfl)
    (sublist_sum_le_topSum hX hsub₂ (by omega))

/-- Top-`k` union lower bound: for any `j ≤ k`, `topSum j C + topSum (k - j) X ≤ topSum k (C ++ X)`. -/
lemma topSum_append_lower {C X : List ℚ} (hC : ∀ x ∈ C, 0 ≤ x) (hX : ∀ x ∈ X, 0 ≤ x)
    {j k : ℕ} (hjk : j ≤ k) :
    topSum j C + topSum (k - j) X ≤ topSum k (C ++ X) := by
  let S₁ := (sortDesc C).take j
  let S₂ := (sortDesc X).take (k - j)
  have hsp : (S₁ ++ S₂).Subperm (sortDesc (C ++ X)) := by
    have hsub : (S₁ ++ S₂).Sublist (sortDesc C ++ sortDesc X) :=
      (List.take_sublist j _).append (List.take_sublist (k - j) _)
    have hperm : (sortDesc C ++ sortDesc X).Perm (sortDesc (C ++ X)) :=
      ((sortDesc_perm C).append (sortDesc_perm X)).trans (sortDesc_perm (C ++ X)).symm
    exact hsub.subperm.trans hperm.subperm
  rcases exists_perm_sublist_of_subperm (sortDesc (C ++ X)) (S₁ ++ S₂) hsp with ⟨S, hP, hsub⟩
  have hlen : S.length ≤ k := by
    rw [← hP.length_eq, List.length_append]
    have h1 : S₁.length ≤ j := List.length_take_le j _
    have h2 : S₂.length ≤ k - j := List.length_take_le (k - j) _
    omega
  have hnn : ∀ x ∈ C ++ X, 0 ≤ x := by
    intro x hx; rcases List.mem_append.mp hx with h | h
    · exact hC x h
    · exact hX x h
  have hle := sublist_sum_le_topSum hnn hsub hlen
  have hsum : S.sum = topSum j C + topSum (k - j) X := by
    rw [← hP.sum_eq, List.sum_append]; rfl
  linarith

/-- Packed list characterization: entries in `[0, 1]` and `topSum k A = min(k, A.sum)` for all `k`. -/
def IsPacked (A : List ℚ) : Prop :=
  (∀ x ∈ A, 0 ≤ x ∧ x ≤ 1) ∧ ∀ k : ℕ, topSum k A = min (k : ℚ) A.sum

lemma isPacked_nil : IsPacked [] := by
  refine ⟨by simp, fun k => ?_⟩
  simp [topSum, sortDesc]

lemma isPacked_singleton {x : ℚ} (hx0 : 0 ≤ x) (hx1 : x ≤ 1) : IsPacked [x] := by
  refine ⟨by simp [hx0, hx1], fun k => ?_⟩
  cases k with
  | zero => simp [topSum, sortDesc, insertDesc, hx0]
  | succ k =>
    simp only [topSum, sortDesc, insertDesc, List.take_succ_cons, List.take_nil, List.sum_cons,
      List.sum_nil, add_zero]
    symm
    apply min_eq_right
    have : (1 : ℚ) ≤ ((k + 1 : ℕ) : ℚ) := by exact_mod_cast Nat.succ_pos k
    linarith

lemma isPacked_cons_zero {A : List ℚ} (hA : IsPacked A) : IsPacked (0 :: A) := by
  have htake : ∀ (L : List ℚ) (k : ℕ),
      (∀ x ∈ L, 0 ≤ x) → L.Pairwise (· ≥ ·) →
      ((insertDesc (0 : ℚ) L).take k).sum = (L.take k).sum := by
    intro L
    induction L with
    | nil =>
      intro k _ _
      cases k <;> simp [insertDesc]
    | cons y ys ih =>
      intro k hnn hpw
      rw [List.pairwise_cons] at hpw
      unfold insertDesc
      split_ifs with hy
      · have hy0 : y = 0 := le_antisymm hy (hnn y (List.mem_cons_self ..))
        have hys0 : ∀ z ∈ ys, z = 0 := fun z hz =>
          le_antisymm (hy0 ▸ hpw.1 z hz) (hnn z (List.mem_cons_of_mem _ hz))
        have hL0 : ∀ z ∈ (0 : ℚ) :: y :: ys, z = 0 := by
          intro z hz
          simp only [List.mem_cons] at hz
          rcases hz with rfl | rfl | h
          · rfl
          · exact hy0
          · exact hys0 z h
        have h1 : (((0 : ℚ) :: y :: ys).take k).sum = 0 :=
          List.sum_eq_zero fun z hz => hL0 z (List.mem_of_mem_take hz)
        have h2 : ((y :: ys).take k).sum = 0 :=
          List.sum_eq_zero fun z hz => hL0 z (List.mem_cons_of_mem _ (List.mem_of_mem_take hz))
        rw [h1, h2]
      · cases k with
        | zero => simp
        | succ k' =>
          simp only [List.take_succ_cons, List.sum_cons]
          congr 1
          exact ih k' (fun z hz => hnn z (List.mem_cons_of_mem _ hz)) hpw.2
  refine ⟨by
    intro x hx
    simp only [List.mem_cons] at hx
    rcases hx with rfl | hxA
    · exact ⟨le_rfl, zero_le_one⟩
    · exact hA.1 x hxA, fun k => ?_⟩
  simp only [topSum, sortDesc, List.sum_cons, zero_add]
  rw [htake (sortDesc A) k (fun x hx => (hA.1 x ((sortDesc_perm A).subset hx)).1)
    (sortDesc_pairwise A)]
  exact hA.2 k

lemma isPacked_cons_one {A : List ℚ} (hA : IsPacked A) : IsPacked (1 :: A) := by
  have hins : insertDesc (1 : ℚ) (sortDesc A) = 1 :: sortDesc A := by
    cases hL : sortDesc A with
    | nil => rfl
    | cons y ys =>
      have hy : y ∈ A := (sortDesc_perm A).subset (by rw [hL]; exact List.mem_cons_self ..)
      have hy1 : y ≤ 1 := (hA.1 y hy).2
      simp [insertDesc, hy1]
  refine ⟨by
    intro x hx
    simp only [List.mem_cons] at hx
    rcases hx with rfl | hxA
    · exact ⟨zero_le_one, le_rfl⟩
    · exact hA.1 x hxA, fun k => ?_⟩
  simp only [topSum, sortDesc, hins, List.sum_cons]
  cases k with
  | zero =>
    simp only [List.take_zero, List.sum_nil, Nat.cast_zero]
    symm
    apply min_eq_left
    have hnn : 0 ≤ A.sum := List.sum_nonneg fun x hx => (hA.1 x hx).1
    linarith
  | succ k' =>
    simp only [List.take_succ_cons, List.sum_cons, Nat.cast_add, Nat.cast_one]
    have hAk := hA.2 k'
    unfold topSum at hAk
    rw [hAk, add_comm (k' : ℚ) 1, add_min]

lemma list_sum_le_length_of_le_one (M : List ℚ) (hle : ∀ x ∈ M, x ≤ 1) :
    M.sum ≤ (M.length : ℚ) := by
  induction M with
  | nil => simp
  | cons a t ih =>
    have ha := hle a (List.mem_cons_self ..)
    have ht := ih (fun z hz => hle z (List.mem_cons_of_mem _ hz))
    simp only [List.sum_cons, List.length_cons, Nat.cast_add, Nat.cast_one]
    linarith

/-- If a packed list has integer mass `n : ℕ`, every entry is `0` or `1`. -/
lemma isPacked_entries_zero_or_one_of_nat_sum {A : List ℚ} (hA : IsPacked A) {n : ℕ}
    (hsum : A.sum = (n : ℚ)) : ∀ x ∈ A, x = 0 ∨ x = 1 := by
  let L := sortDesc A
  have hL_bounds : ∀ x ∈ L, 0 ≤ x ∧ x ≤ 1 := fun x hx => hA.1 x ((sortDesc_perm A).subset hx)
  have htake_sum : (L.take n).sum = (n : ℚ) := by
    have h := hA.2 n
    unfold topSum at h
    rw [hsum, min_self] at h
    exact h
  have htake_all_one : ∀ x ∈ L.take n, x = 1 := by
    have hgen : ∀ (M : List ℚ) (m : ℕ), M.length ≤ m → (∀ x ∈ M, x ≤ 1) →
        M.sum = (m : ℚ) → ∀ x ∈ M, x = 1 := by
      intro M
      induction M with
      | nil => simp
      | cons a t ih =>
        intro m hm hle hms
        cases m with
        | zero => simp at hm
        | succ m' =>
          have ht_le : t.length ≤ m' := by simp at hm; omega
          have ha1 : a ≤ 1 := hle a (List.mem_cons_self ..)
          have htsum_le : t.sum ≤ (m' : ℚ) := by
            have h1 := list_sum_le_length_of_le_one t (fun z hz => hle z (List.mem_cons_of_mem _ hz))
            have h2 : (t.length : ℚ) ≤ (m' : ℚ) := by exact_mod_cast ht_le
            linarith
          simp only [List.sum_cons, Nat.cast_add, Nat.cast_one] at hms
          have ha_eq : a = 1 := by linarith
          have ht_eq : t.sum = (m' : ℚ) := by linarith
          intro x hx
          simp only [List.mem_cons] at hx
          rcases hx with rfl | hxt
          · exact ha_eq
          · exact ih m' ht_le (fun z hz => hle z (List.mem_cons_of_mem _ hz)) ht_eq x hxt
    exact hgen (L.take n) n (List.length_take_le n L)
      (fun x hx => (hL_bounds x (List.mem_of_mem_take hx)).2) htake_sum
  have hdrop_sum : (L.drop n).sum = 0 := by
    have hsplit : L.sum = (L.take n).sum + (L.drop n).sum := by
      rw [← List.sum_append, List.take_append_drop]
    have hLsum : L.sum = (n : ℚ) := ((sortDesc_perm A).sum_eq).trans hsum
    linarith
  have hdrop_all_zero : ∀ x ∈ L.drop n, x = 0 := by
    have hgen : ∀ M : List ℚ, (∀ x ∈ M, 0 ≤ x) → M.sum = 0 → ∀ x ∈ M, x = 0 := by
      intro M
      induction M with
      | nil => simp
      | cons a t ih =>
        intro hnn hs
        have ha : 0 ≤ a := hnn a (List.mem_cons_self ..)
        have ht : 0 ≤ t.sum := List.sum_nonneg fun z hz => hnn z (List.mem_cons_of_mem _ hz)
        simp only [List.sum_cons] at hs
        have ha0 : a = 0 := by linarith
        have ht0 : t.sum = 0 := by linarith
        intro x hx
        simp only [List.mem_cons] at hx
        rcases hx with rfl | hxt
        · exact ha0
        · exact ih (fun z hz => hnn z (List.mem_cons_of_mem _ hz)) ht0 x hxt
    exact hgen (L.drop n) (fun x hx => (hL_bounds x (List.mem_of_mem_drop hx)).1) hdrop_sum
  intro x hx
  have hxL : x ∈ L.take n ++ L.drop n := by
    rw [List.take_append_drop]
    exact (sortDesc_perm A).symm.subset hx
  rcases List.mem_append.mp hxL with h1 | h2
  · exact Or.inr (htake_all_one x h1)
  · exact Or.inl (hdrop_all_zero x h2)

lemma isPacked_append_of_zero_one {A B : List ℚ} (hA : ∀ x ∈ A, x = 0 ∨ x = 1) (hB : IsPacked B) :
    IsPacked (A ++ B) := by
  induction A with
  | nil => simpa using hB
  | cons a t ih =>
    have iht := ih (fun x hx => hA x (List.mem_cons_of_mem _ hx))
    rcases hA a (List.mem_cons_self ..) with rfl | rfl
    · exact isPacked_cons_zero iht
    · exact isPacked_cons_one iht

lemma isPacked_perm {A B : List ℚ} (hP : A.Perm B) (hA : IsPacked A) : IsPacked B := by
  refine ⟨fun x hx => hA.1 x (hP.symm.subset hx), fun k => ?_⟩
  rw [topSum_perm hP.symm k, hA.2 k, hP.sum_eq]

lemma isPacked_append_of_nat_sum {A B : List ℚ} (hA : IsPacked A) {n : ℕ} (hsum : A.sum = (n : ℚ))
    (hB : IsPacked B) : IsPacked (A ++ B) :=
  isPacked_append_of_zero_one (isPacked_entries_zero_or_one_of_nat_sum hA hsum) hB

/-- **Lemma 1 (Single-rod packedness)**:
for every integer `x ≥ 1` and every dyadic scale `k ≥ 0`, the layer list
`A_k([x]) = (canonNodes x).map (h k)` is packed with exact total mass `Q k x`. -/
theorem single_rod_packed (k : ℕ) :
    ∀ x : ℕ, 1 ≤ x →
      IsPacked ((canonNodes x).map (h k)) ∧ ((canonNodes x).map (h k)).sum = Q k x := by
  intro x
  induction x using Nat.strong_induction_on with
  | _ x ih =>
    intro hx1
    by_cases hle : x ≤ 2 ^ k
    · -- All nodes in `canonNodes x` are `≤ x ≤ 2^k`, so every entry maps to `0`.
      have hall0 : ∀ y ∈ (canonNodes x).map (h k), y = 0 := by
        intro y hy
        rcases List.mem_map.mp hy with ⟨r, hr, rfl⟩
        exact h_eq_zero_of_le (le_trans (canonNodes_mem_le hr).2 hle)
      have hpack : IsPacked ((canonNodes x).map (h k)) := by
        have h := isPacked_append_of_zero_one (A := (canonNodes x).map (h k)) (B := [])
          (fun y hy => Or.inl (hall0 y hy)) isPacked_nil
        simpa using h
      refine ⟨hpack, ?_⟩
      rw [List.sum_eq_zero hall0, Q_eq_zero_of_le hle]
    · push_neg at hle
      have hx2 : 2 ≤ x := by
        have := Nat.one_le_pow k 2 (by norm_num); omega
      have hb := canon_bounds hx2
      rw [canonNodes_of_ge_two hx2, List.map_cons, List.map_append]
      by_cases hlt : x < 2 ^ (k + 1)
      · -- `2^k < x < 2^(k+1)`: `Nat.log 2 x = k`, so `2 * canonBase x = 2^k`, hence both children `≤ 2^k`!
        have hlog : Nat.log 2 x = k :=
          le_antisymm (Nat.lt_succ_iff.mp ((Nat.log_lt_iff_lt_pow (by norm_num) (by omega)).mpr hlt))
            ((Nat.le_log_iff_pow_le (by norm_num) (by omega)).mpr (le_of_lt hle))
        have h2b : 2 * canonBase x = 2 ^ k := by
          unfold canonBase
          rw [hlog]
          have hk1 : 1 ≤ k := by
            by_contra hk0
            have : k = 0 := by omega
            subst this; norm_num at hle hlt; omega
          have : k = (k - 1) + 1 := by omega
          conv_rhs => rw [this, pow_succ']
        have hAle : canonA x ≤ 2 ^ k := h2b ▸ hb.2.2.2.2.2.1
        have hBle : canonB x ≤ 2 ^ k := h2b ▸ hb.2.2.2.2.2.2.2.1
        have hchild0 : ∀ y ∈ (canonNodes (canonA x)).map (h k) ++ (canonNodes (canonB x)).map (h k),
            y = 0 := by
          intro y hy
          rcases List.mem_append.mp hy with h1 | h2
          · rcases List.mem_map.mp h1 with ⟨r, hr, rfl⟩
            exact h_eq_zero_of_le (le_trans (canonNodes_mem_le hr).2 hAle)
          · rcases List.mem_map.mp h2 with ⟨r, hr, rfl⟩
            exact h_eq_zero_of_le (le_trans (canonNodes_mem_le hr).2 hBle)
        have hroot_pack : IsPacked [h k x] := isPacked_singleton (h_bounds k x).1 (h_bounds k x).2
        have hperm : (h k x :: ((canonNodes (canonA x)).map (h k) ++ (canonNodes (canonB x)).map (h k))).Perm
            (((canonNodes (canonA x)).map (h k) ++ (canonNodes (canonB x)).map (h k)) ++ [h k x]) :=
          List.perm_append_singleton _ _ |>.symm
        have hpack : IsPacked (h k x :: ((canonNodes (canonA x)).map (h k) ++ (canonNodes (canonB x)).map (h k))) :=
          isPacked_perm hperm.symm (isPacked_append_of_zero_one (fun y hy => Or.inl (hchild0 y hy)) hroot_pack)
        refine ⟨hpack, ?_⟩
        rw [List.sum_cons, List.sum_eq_zero hchild0, add_zero]
        have hhq : h k x = Q k x := by
          rw [h_eq_Q_sub, Q_eq_zero_of_le (le_of_lt hlt)]; ring
        exact hhq
      · -- `2^(k+1) ≤ x`: `canonBase x ≥ 2^k`, both children `≥ 2^k`, one is divisible by `2^k`.
        push_neg at hlt
        have hlog_ge : k + 1 ≤ Nat.log 2 x :=
          (Nat.le_log_iff_pow_le (by norm_num) (by omega)).mpr hlt
        have hb_ge : 2 ^ k ≤ canonBase x :=
          Nat.pow_le_pow_right (by norm_num) (by omega)
        have hAge : 2 ^ k ≤ canonA x := le_trans hb_ge hb.2.2.2.2.1
        have hBge : 2 ^ k ≤ canonB x := le_trans hb_ge hb.2.2.2.2.2.2.1
        have hA1 : 1 ≤ canonA x := le_trans (Nat.one_le_pow k 2 (by norm_num)) hAge
        have hB1 : 1 ≤ canonB x := le_trans (Nat.one_le_pow k 2 (by norm_num)) hBge
        have ihA := ih (canonA x) hb.2.2.2.2.2.2.2.2.1 hA1
        have ihB := ih (canonB x) hb.2.2.2.2.2.2.2.2.2.1 hB1
        have hroot1 : h k x = 1 := h_eq_one_of_ge hlt
        have hsum_eq : 1 + Q k (canonA x) + Q k (canonB x) = Q k x := by
          rw [Q_eq_of_ge hAge, Q_eq_of_ge hBge,
            Q_eq_of_ge (le_trans (Nat.pow_le_pow_right (by norm_num) (Nat.le_succ k)) hlt)]
          have hab : (canonA x : ℚ) + (canonB x : ℚ) = (x : ℚ) := by exact_mod_cast hb.2.2.2.1
          have hne : (2 : ℚ) ^ k ≠ 0 := by positivity
          field_simp [hne]
          linarith
        -- One of `canonA x, canonB x` is a power of two `≥ 2^k`, hence its `Q k` is a `ℕ`.
        have hpow_nat : ∀ m : ℕ, k ≤ m → ∃ q : ℕ, Q k (2 ^ m) = (q : ℚ) := by
          intro m hkm
          have hge : 2 ^ k ≤ 2 ^ m := Nat.pow_le_pow_right (by norm_num) hkm
          rw [Q_eq_of_ge hge]
          refine ⟨2 ^ (m - k) - 1, ?_⟩
          have h1 : 1 ≤ 2 ^ (m - k) := Nat.one_le_pow _ 2 (by norm_num)
          rw [Nat.cast_sub h1, Nat.cast_pow, Nat.cast_ofNat, Nat.cast_one]
          have hm : m = (m - k) + k := by omega
          conv_lhs => rw [hm, pow_add]
          have hne : (2 : ℚ) ^ k ≠ 0 := by positivity
          field_simp [hne]
          push_cast
          ring
        have hchildren_pack : IsPacked ((canonNodes (canonA x)).map (h k) ++ (canonNodes (canonB x)).map (h k)) := by
          rcases hb.2.2.2.2.2.2.2.2.2.2 with hAeq | hBeq
          · rcases hpow_nat (Nat.log 2 x - 1) (by omega) with ⟨q, hq⟩
            have hAsum : ((canonNodes (canonA x)).map (h k)).sum = (q : ℚ) := by
              rw [ihA.2, hAeq]; exact hq
            exact isPacked_append_of_nat_sum ihA.1 hAsum ihB.1
          · rcases hpow_nat (Nat.log 2 x) (by omega) with ⟨q, hq⟩
            have h2b : 2 * canonBase x = 2 ^ Nat.log 2 x := by
              unfold canonBase
              have : Nat.log 2 x = (Nat.log 2 x - 1) + 1 := by omega
              conv_rhs => rw [this, pow_succ']
            have hBsum : ((canonNodes (canonB x)).map (h k)).sum = (q : ℚ) := by
              rw [ihB.2, hBeq, h2b]; exact hq
            exact isPacked_perm (List.perm_append_comm)
              (isPacked_append_of_nat_sum ihB.1 hBsum ihA.1)
        refine ⟨hroot1 ▸ isPacked_cons_one hchildren_pack, ?_⟩
        rw [List.sum_cons, List.sum_append, hroot1, ihA.2, ihB.2]
        linarith

/-- **Lemma 2 (Per-layer split comparison)**:
for `p ≥ 2`, canonical split `a + a' = p`, and ANY legal split `u + v = p` (`u, v ≥ 1`),
at every dyadic scale `k`:
1. `A_k([a, a'])` is packed with mass `M_k = Q k a + Q k a'`;
2. `A_k([u, v])` has mass `Q k u + Q k v ≥ M_k`. -/
theorem per_layer_split_comparison {p u v : ℕ} (hp : 2 ≤ p) (hu : 1 ≤ u) (hv : 1 ≤ v)
    (huv : u + v = p) (k : ℕ) :
    IsPacked ((forestNodes [canonA p, canonB p]).map (h k)) ∧
    ((forestNodes [canonA p, canonB p]).map (h k)).sum = Q k (canonA p) + Q k (canonB p) ∧
    ((forestNodes [u, v]).map (h k)).sum = Q k u + Q k v ∧
    Q k (canonA p) + Q k (canonB p) ≤ Q k u + Q k v := by
  have hb := canon_bounds hp
  have hA1 : 1 ≤ canonA p := le_trans hb.1 hb.2.2.2.2.1
  have hB1 : 1 ≤ canonB p := le_trans hb.1 hb.2.2.2.2.2.2.1
  have hPA := single_rod_packed k (canonA p) hA1
  have hPB := single_rod_packed k (canonB p) hB1
  have hPu := single_rod_packed k u hu
  have hPv := single_rod_packed k v hv
  have hfor_ab : (forestNodes [canonA p, canonB p]).map (h k) =
      (canonNodes (canonA p)).map (h k) ++ (canonNodes (canonB p)).map (h k) := by
    simp [forestNodes]
  have hfor_uv : (forestNodes [u, v]).map (h k) =
      (canonNodes u).map (h k) ++ (canonNodes v).map (h k) := by
    simp [forestNodes]
  have hsum_ab : ((forestNodes [canonA p, canonB p]).map (h k)).sum = Q k (canonA p) + Q k (canonB p) := by
    rw [hfor_ab, List.sum_append, hPA.2, hPB.2]
  have hsum_uv : ((forestNodes [u, v]).map (h k)).sum = Q k u + Q k v := by
    rw [hfor_uv, List.sum_append, hPu.2, hPv.2]
  by_cases hk : 2 ^ k ≤ canonBase p
  · -- `2^k ≤ b`: both `canonA p, canonB p ≥ 2^k`, and one is a power of 2 `≥ 2^k`.
    have hAge : 2 ^ k ≤ canonA p := le_trans hk hb.2.2.2.2.1
    have hBge : 2 ^ k ≤ canonB p := le_trans hk hb.2.2.2.2.2.2.1
    have ht1 : 1 ≤ Nat.log 2 p := by
      rw [Nat.le_log_iff_pow_le (by norm_num) (by omega)]; simpa using hp
    have hk_le_t : k ≤ Nat.log 2 p - 1 :=
      (Nat.pow_le_pow_iff_right (by norm_num : 1 < 2)).mp hk
    have hpow_nat : ∀ m : ℕ, k ≤ m → ∃ q : ℕ, Q k (2 ^ m) = (q : ℚ) := by
      intro m hkm
      rw [Q_eq_of_ge (Nat.pow_le_pow_right (by norm_num) hkm)]
      refine ⟨2 ^ (m - k) - 1, ?_⟩
      have h1 : 1 ≤ 2 ^ (m - k) := Nat.one_le_pow _ 2 (by norm_num)
      rw [Nat.cast_sub h1, Nat.cast_pow, Nat.cast_ofNat, Nat.cast_one]
      have hm : m = (m - k) + k := by omega
      conv_lhs => rw [hm, pow_add]
      have hne : (2 : ℚ) ^ k ≠ 0 := by positivity
      field_simp [hne]; push_cast; ring
    have hpack : IsPacked ((forestNodes [canonA p, canonB p]).map (h k)) := by
      rw [hfor_ab]
      rcases hb.2.2.2.2.2.2.2.2.2.2 with hAeq | hBeq
      · rcases hpow_nat (Nat.log 2 p - 1) hk_le_t with ⟨q, hq⟩
        exact isPacked_append_of_nat_sum hPA.1 (by rw [hPA.2, hAeq]; exact hq) hPB.1
      · rcases hpow_nat (Nat.log 2 p) (by omega) with ⟨q, hq⟩
        have h2b : 2 * canonBase p = 2 ^ Nat.log 2 p := by
          unfold canonBase
          have : Nat.log 2 p = (Nat.log 2 p - 1) + 1 := by omega
          conv_rhs => rw [this, pow_succ']
        exact isPacked_perm List.perm_append_comm
          (isPacked_append_of_nat_sum hPB.1 (by rw [hPB.2, hBeq, h2b]; exact hq) hPA.1)
    have hle : Q k (canonA p) + Q k (canonB p) ≤ Q k u + Q k v := by
      rw [Q_eq_of_ge hAge, Q_eq_of_ge hBge]
      have hqu : (u : ℚ) / (2 : ℚ) ^ k - 1 ≤ Q k u := le_max_right _ _
      have hqv : (v : ℚ) / (2 : ℚ) ^ k - 1 ≤ Q k v := le_max_right _ _
      have hsum1 : (canonA p : ℚ) + (canonB p : ℚ) = (u : ℚ) + (v : ℚ) := by
        exact_mod_cast hb.2.2.2.1.trans huv.symm
      have hne : (2 : ℚ) ^ k ≠ 0 := by positivity
      have hcomb : (canonA p : ℚ) / (2 : ℚ) ^ k - 1 + ((canonB p : ℚ) / (2 : ℚ) ^ k - 1) =
          (u : ℚ) / (2 : ℚ) ^ k - 1 + ((v : ℚ) / (2 : ℚ) ^ k - 1) := by
        field_simp [hne]; linarith
      linarith
    exact ⟨hpack, hsum_ab, hsum_uv, hle⟩
  · -- `2^k > b`, so since both are powers of two, `2 * b ≤ 2^k`, hence `canonA p, canonB p ≤ 2^k`.
    push_neg at hk
    have hk_gt : Nat.log 2 p - 1 < k :=
      (Nat.pow_lt_pow_iff_right (by norm_num : 1 < 2)).mp hk
    have h2b_le : 2 * canonBase p ≤ 2 ^ k := by
      unfold canonBase
      rw [← pow_succ']
      exact Nat.pow_le_pow_right (by norm_num) (by omega)
    have hAle : canonA p ≤ 2 ^ k := le_trans hb.2.2.2.2.2.1 h2b_le
    have hBle : canonB p ≤ 2 ^ k := le_trans hb.2.2.2.2.2.2.2.1 h2b_le
    have hQa0 : Q k (canonA p) = 0 := Q_eq_zero_of_le hAle
    have hQb0 : Q k (canonB p) = 0 := Q_eq_zero_of_le hBle
    have hall0 : ∀ y ∈ (forestNodes [canonA p, canonB p]).map (h k), y = 0 := by
      rw [hfor_ab]
      intro y hy
      rcases List.mem_append.mp hy with h1 | h2
      · rcases List.mem_map.mp h1 with ⟨r, hr, rfl⟩
        exact h_eq_zero_of_le (le_trans (canonNodes_mem_le hr).2 hAle)
      · rcases List.mem_map.mp h2 with ⟨r, hr, rfl⟩
        exact h_eq_zero_of_le (le_trans (canonNodes_mem_le hr).2 hBle)
    have hpack : IsPacked ((forestNodes [canonA p, canonB p]).map (h k)) := by
      have h := isPacked_append_of_zero_one (A := (forestNodes [canonA p, canonB p]).map (h k))
        (B := []) (fun y hy => Or.inl (hall0 y hy)) isPacked_nil
      simpa using h
    refine ⟨hpack, hsum_ab, hsum_uv, ?_⟩
    rw [hQa0, hQb0, add_zero]
    exact add_nonneg (Q_nonneg k u) (Q_nonneg k v)

/-- **Lemma 3 (Packed majorization under union and discounting)**:
if `A` is packed, `B` and `C` have entries in `[0, 1]`, and `B.sum = A.sum + E` with `0 ≤ E`,
then `D(C ++ B) - D(C ++ A) ≤ E / 2`. -/
theorem packed_majorization {A B C : List ℚ} {E : ℚ}
    (hA : IsPacked A)
    (hB : ∀ x ∈ B, 0 ≤ x ∧ x ≤ 1)
    (hC : ∀ x ∈ C, 0 ≤ x ∧ x ≤ 1)
    (hE0 : 0 ≤ E) (hsum : B.sum = A.sum + E) :
    D (C ++ B) - D (C ++ A) ≤ E / 2 := by
  -- Step 1: `topSum k B ≤ topSum k A + E` for every `k : ℕ`.
  have htop_BA : ∀ k : ℕ, topSum k B ≤ topSum k A + E := by
    intro k
    rw [hA.2 k]
    have hle_k : topSum k B ≤ (k : ℚ) := by
      unfold topSum
      have h1 : ∀ M : List ℚ, (∀ x ∈ M, x ≤ 1) → M.sum ≤ (M.length : ℚ) := by
        intro M
        induction M with
        | nil => simp
        | cons a t ih =>
          intro hle
          have ha := hle a (List.mem_cons_self ..)
          have ht := ih (fun z hz => hle z (List.mem_cons_of_mem _ hz))
          simp only [List.sum_cons, List.length_cons, Nat.cast_add, Nat.cast_one]
          linarith
      have hbound := h1 ((sortDesc B).take k)
        (fun x hx => (hB x ((sortDesc_perm B).subset (List.mem_of_mem_take hx))).2)
      have hlen : (((sortDesc B).take k).length : ℚ) ≤ (k : ℚ) := by
        exact_mod_cast List.length_take_le k (sortDesc B)
      linarith
    have hle_sum : topSum k B ≤ B.sum := by
      unfold topSum
      have hsplit : (sortDesc B).sum = ((sortDesc B).take k).sum + ((sortDesc B).drop k).sum := by
        rw [← List.sum_append, List.take_append_drop]
      have hdrop_nn : 0 ≤ ((sortDesc B).drop k).sum :=
        List.sum_nonneg fun x hx => (hB x ((sortDesc_perm B).subset (List.mem_of_mem_drop hx))).1
      rw [(sortDesc_perm B).sum_eq] at hsplit
      linarith
    rw [min_add]
    exact le_min (by linarith) (by linarith)
  -- Step 2: `topSum k (C ++ B) ≤ topSum k (C ++ A) + E` for every `k : ℕ`.
  have htop_union : ∀ k : ℕ, topSum k (C ++ B) ≤ topSum k (C ++ A) + E := by
    intro k
    rcases topSum_append_upper (fun x hx => (hC x hx).1) (fun x hx => (hB x hx).1) k with
      ⟨j, hjk, hup⟩
    have hlo := topSum_append_lower (fun x hx => (hC x hx).1) (fun x hx => (hA.1 x hx).1) hjk
    linarith [htop_BA (k - j)]
  -- Step 3: Finite Abel summation with `M = max (C ++ B).length (C ++ A).length`.
  let M := max (C ++ B).length (C ++ A).length
  have hlen1 : (sortDesc (C ++ B)).length ≤ M := by
    rw [(sortDesc_perm _).length_eq]; exact le_max_left _ _
  have hlen2 : (sortDesc (C ++ A)).length ≤ M := by
    rw [(sortDesc_perm _).length_eq]; exact le_max_right _ _
  unfold D
  rw [finite_abel_sum M (sortDesc (C ++ B)) hlen1, finite_abel_sum M (sortDesc (C ++ A)) hlen2]
  have hdiff_sum :
      (∑ k ∈ Finset.range M, topSum (k + 1) (C ++ B) / (2 : ℚ) ^ (k + 2)) -
      (∑ k ∈ Finset.range M, topSum (k + 1) (C ++ A) / (2 : ℚ) ^ (k + 2)) ≤
      E * ∑ k ∈ Finset.range M, (1 : ℚ) / (2 : ℚ) ^ (k + 2) := by
    rw [← Finset.sum_sub_distrib, Finset.mul_sum]
    refine Finset.sum_le_sum fun k _ => ?_
    have h1 := htop_union (k + 1)
    have hpos : (0 : ℚ) < (2 : ℚ) ^ (k + 2) := by positivity
    rw [← sub_div, mul_one_div]
    exact div_le_div_of_nonneg_right (by linarith) (le_of_lt hpos)
  have hdiff_tail :
      topSum M (C ++ B) / (2 : ℚ) ^ (M + 1) - topSum M (C ++ A) / (2 : ℚ) ^ (M + 1) ≤
      E * ((1 : ℚ) / (2 : ℚ) ^ (M + 1)) := by
    have h1 := htop_union M
    have hpos : (0 : ℚ) < (2 : ℚ) ^ (M + 1) := by positivity
    rw [← sub_div, mul_one_div]
    exact div_le_div_of_nonneg_right (by linarith) (le_of_lt hpos)
  have hcoeff : (∑ k ∈ Finset.range M, (1 : ℚ) / (2 : ℚ) ^ (k + 2)) +
      (1 : ℚ) / (2 : ℚ) ^ (M + 1) = 1 / 2 := by
    induction M with
    | zero => norm_num
    | succ m ihm =>
      rw [Finset.sum_range_succ]
      have hstep : (1 : ℚ) / (2 : ℚ) ^ (m + 2) + (1 : ℚ) / (2 : ℚ) ^ (m + 1 + 1) =
          (1 : ℚ) / (2 : ℚ) ^ (m + 1) := by
        have ha : m + 1 + 1 = m + 2 := by omega
        rw [ha, show m + 2 = (m + 1) + 1 by omega, pow_succ']
        ring
      linarith
  change ((∑ k ∈ Finset.range M, topSum (k + 1) (C ++ B) / (2 : ℚ) ^ (k + 2)) +
      topSum M (C ++ B) / (2 : ℚ) ^ (M + 1)) -
    ((∑ k ∈ Finset.range M, topSum (k + 1) (C ++ A) / (2 : ℚ) ^ (k + 2)) +
      topSum M (C ++ A) / (2 : ℚ) ^ (M + 1)) ≤ E / 2
  have htot : E * (∑ k ∈ Finset.range M, (1 : ℚ) / (2 : ℚ) ^ (k + 2)) +
      E * ((1 : ℚ) / (2 : ℚ) ^ (M + 1)) = E / 2 := by
    rw [← mul_add, hcoeff]; ring
  linarith

/-! ### 5. Theorem 1: Bellman Recursion and Optimality of Greedy (`V(P) = G(P)`) -/

/-- Rank `∑_{r ∈ P} (r - 1)` of a rod list; every legal cut decreases rank by `1`. -/
def rodRank (P : List ℕ) : ℕ := (P.map (fun r => r - 1)).sum

lemma perm_cons_eraseIdx {α : Type*} :
    ∀ (L : List α) (i : Fin L.length), L.Perm (L[i] :: L.eraseIdx i.1)
  | [], i => Fin.elim0 i
  | a :: t, ⟨0, _⟩ => List.Perm.refl _
  | a :: t, ⟨k + 1, hk⟩ => by
    have hk' : k < t.length := Nat.lt_of_succ_lt_succ hk
    have ih := perm_cons_eraseIdx t ⟨k, hk'⟩
    simp only [List.eraseIdx_cons_succ]
    exact (ih.cons a).trans (List.Perm.swap _ a _)

lemma rodRank_eraseIdx_add {P : List ℕ} {i : Fin P.length} {u : ℕ}
    (hu1 : 1 ≤ u) (hup : u < P[i]) :
    rodRank (u :: (P[i] - u) :: P.eraseIdx i.1) < rodRank P := by
  have hperm : P.Perm (P[i] :: P.eraseIdx i.1) := perm_cons_eraseIdx P i
  have hrank_eq : rodRank P = (P[i] - 1) + rodRank (P.eraseIdx i.1) := by
    unfold rodRank
    rw [(hperm.map (fun r => r - 1)).sum_eq, List.map_cons, List.sum_cons]
  have hnew : rodRank (u :: (P[i] - u) :: P.eraseIdx i.1) =
      (u - 1) + (P[i] - u - 1) + rodRank (P.eraseIdx i.1) := by
    simp [rodRank, add_assoc]
  omega

/-- Exact discounted rod-cutting Bellman value `V(P)` (equation (1) of `solution.md`,
`eq:rlcut` of arXiv:2611.28727). -/
def bellmanV (P : List ℕ) : ℚ :=
  ((List.finRange P.length).flatMap fun (i : Fin P.length) =>
    if hp : 2 ≤ P[i] then
      (List.range (P[i] - 1)).attach.map fun ⟨k, hk⟩ =>
        have hklt : k < P[i] - 1 := List.mem_range.mp hk
        have hu1 : 1 ≤ k + 1 := Nat.succ_le_succ (Nat.zero_le k)
        have hup : k + 1 < P[i] := by omega
        (cutReward (k + 1) (P[i] - (k + 1)) +
          bellmanV ((k + 1) :: (P[i] - (k + 1)) :: P.eraseIdx i.1)) / 2
    else
      []).foldr max 0
termination_by rodRank P
decreasing_by
  exact rodRank_eraseIdx_add hu1 hup

lemma foldr_max_nonneg (L : List ℚ) : 0 ≤ L.foldr max 0 := by
  induction L with
  | nil => exact le_rfl
  | cons a t ih => exact le_trans ih (le_max_right a _)

lemma le_foldr_max {L : List ℚ} {x : ℚ} (hx : x ∈ L) : x ≤ L.foldr max 0 := by
  induction L with
  | nil => simp at hx
  | cons a t ih =>
    simp only [List.mem_cons] at hx
    rcases hx with rfl | hxt
    · exact le_max_left _ _
    · exact le_trans (ih hxt) (le_max_right _ _)

lemma foldr_max_le {L : List ℚ} {B : ℚ} (hB0 : 0 ≤ B) (hL : ∀ x ∈ L, x ≤ B) :
    L.foldr max 0 ≤ B := by
  induction L with
  | nil => exact hB0
  | cons a t ih =>
    exact max_le (hL a (List.mem_cons_self ..)) (ih fun x hx => hL x (List.mem_cons_of_mem _ hx))

/-- Any legal first cut gives a lower bound on `bellmanV P`. -/
lemma cut_le_bellmanV (P : List ℕ) (i : Fin P.length) {u : ℕ}
    (hu1 : 1 ≤ u) (hup : u < P[i]) :
    (cutReward u (P[i] - u) + bellmanV (u :: (P[i] - u) :: P.eraseIdx i.1)) / 2 ≤ bellmanV P := by
  conv_rhs => rw [bellmanV]
  apply le_foldr_max
  rw [List.mem_flatMap]
  refine ⟨i, List.mem_finRange i, ?_⟩
  have hp : 2 ≤ P[i] := by omega
  simp only [hp, ↓reduceDIte, List.mem_map]
  refine ⟨⟨u - 1, List.mem_range.mpr (by omega)⟩, List.mem_attach _ _, ?_⟩
  have hsub : u - 1 + 1 = u := by omega
  simp only [hsub]

/-- Upper bound criterion for `bellmanV P`: if `0 ≤ B` and every legal first cut has value `≤ B`,
then `bellmanV P ≤ B`. -/
lemma bellmanV_le (P : List ℕ) {B : ℚ} (hB0 : 0 ≤ B)
    (hcuts : ∀ (i : Fin P.length) (u : ℕ), 1 ≤ u → u < P[i] →
      (cutReward u (P[i] - u) + bellmanV (u :: (P[i] - u) :: P.eraseIdx i.1)) / 2 ≤ B) :
    bellmanV P ≤ B := by
  rw [bellmanV]
  apply foldr_max_le hB0
  intro x hx
  rw [List.mem_flatMap] at hx
  rcases hx with ⟨i, _, hi⟩
  split_ifs at hi with hp
  · rcases List.mem_map.mp hi with ⟨⟨k, hk⟩, _, rfl⟩
    have hlt : k + 1 < P[i] := by rw [List.mem_range] at hk; omega
    exact hcuts i (k + 1) (Nat.succ_le_succ (Nat.zero_le k)) hlt
  · simp at hi

/-- **Theorem 1 (Optimality of the greedy canonical schedule — Conjecture `conj:rl`)**:
for every finite list `P` of positive rods (`∀ p ∈ P, 1 ≤ p`), `V(P) = G(P)`. -/
theorem greedy_optimal :
    ∀ (P : List ℕ), (∀ p ∈ P, 1 ≤ p) → bellmanV P = greedyG P := by
  intro P
  induction hR : rodRank P using Nat.strong_induction_on generalizing P with
  | _ n ih =>
    intro hP1
    have hG_nonneg : 0 ≤ greedyG P := by
      have hdisc_nn : ∀ L : List ℚ, (∀ x ∈ L, 0 ≤ x) → 0 ≤ discRaw L := by
        intro L
        induction L with
        | nil => simp [discRaw]
        | cons a t iht =>
          intro hnn
          simp only [discRaw]
          have ha := hnn a (List.mem_cons_self ..)
          have ht := iht (fun z hz => hnn z (List.mem_cons_of_mem _ hz))
          linarith
      apply hdisc_nn
      intro x hx
      rcases List.mem_map.mp ((sortDesc_perm _).subset hx) with ⟨r, hr, rfl⟩
      have h0 : lamTilde 0 ≤ lamTilde r := lamTilde_mono (Nat.zero_le r)
      simpa [lamTilde] using h0
    -- Direction 1: `bellmanV P ≤ greedyG P`
    have hle : bellmanV P ≤ greedyG P := by
      apply bellmanV_le P hG_nonneg
      intro i u hu1 hup
      let p := P[i]
      let v := p - u
      let T := P.eraseIdx i.1
      have hp2 : 2 ≤ p := by omega
      have hv1 : 1 ≤ v := by omega
      have huv : u + v = p := by omega
      have hP_perm : P.Perm (p :: T) := perm_cons_eraseIdx P i
      have hT1 : ∀ r ∈ T, 1 ≤ r := fun r hr =>
        hP1 r (hP_perm.symm.subset (List.mem_cons_of_mem _ hr))
      have hUV1 : ∀ r ∈ u :: v :: T, 1 ≤ r := by
        intro r hr
        simp only [List.mem_cons] at hr
        rcases hr with rfl | rfl | hrT
        · exact hu1
        · exact hv1
        · exact hT1 r hrT
      have hrank_lt : rodRank (u :: v :: T) < n := by
        rw [← hR]; exact rodRank_eraseIdx_add hu1 hup
      have hIH := ih (rodRank (u :: v :: T)) hrank_lt (u :: v :: T) rfl hUV1
      rw [hIH]
      -- Choose common dyadic horizon `N` larger than all rods in `P`
      let Mmax := P.foldr max 1
      let N := Mmax
      have hle_M : ∀ r ∈ P, r ≤ Mmax := by
        intro r hr
        have hgen : ∀ (L : List ℕ) (x : ℕ), x ∈ L → x ≤ L.foldr max 1 := by
          intro L x hx
          induction L with
          | nil => simp at hx
          | cons a t iht =>
            simp only [List.mem_cons] at hx
            rcases hx with rfl | hxt
            · exact le_max_left _ _
            · exact le_trans (iht hxt) (le_max_right _ _)
        exact hgen P r hr
      have hle_pow : ∀ r, r ≤ Mmax → r ≤ 2 ^ N := fun r hr =>
        le_trans hr (le_of_lt Nat.lt_two_pow_self)
      have hp_leM : p ≤ Mmax := hle_M p (List.getElem_mem i.2)
      have hT_le : ∀ r ∈ T, r ≤ 2 ^ N := fun r hr =>
        hle_pow r (hle_M r (hP_perm.symm.subset (List.mem_cons_of_mem _ hr)))
      have hUV_le : ∀ r ∈ u :: v :: T, r ≤ 2 ^ N := by
        intro r hr
        simp only [List.mem_cons] at hr
        rcases hr with rfl | rfl | hrT
        · exact hle_pow _ (by omega)
        · exact hle_pow _ (by omega)
        · exact hT_le r hrT
      have hb := canon_bounds hp2
      have hAB_le : ∀ r ∈ canonA p :: canonB p :: T, r ≤ 2 ^ N := by
        intro r hr
        simp only [List.mem_cons] at hr
        rcases hr with rfl | rfl | hrT
        · exact hle_pow (canonA p) (le_trans (le_of_lt hb.2.2.2.2.2.2.2.2.1) hp_leM)
        · exact hle_pow (canonB p) (le_trans (le_of_lt hb.2.2.2.2.2.2.2.2.2.1) hp_leM)
        · exact hT_le r hrT
      -- Apply Lemma 2 + Lemma 3 at every dyadic scale `k < N`
      have hlayer_bound : greedyG (u :: v :: T) - greedyG (canonA p :: canonB p :: T) ≤
          (1 / 2 : ℚ) * ∑ k ∈ Finset.range N, (Q k u + Q k v - (Q k (canonA p) + Q k (canonB p))) := by
        rw [greedy_layer_decomp hUV_le, greedy_layer_decomp hAB_le, ← Finset.sum_sub_distrib,
          Finset.mul_sum]
        refine Finset.sum_le_sum fun k _ => ?_
        have hL2 := per_layer_split_comparison hp2 hu1 hv1 huv k
        let C := (forestNodes T).map (h k)
        let A := (forestNodes [canonA p, canonB p]).map (h k)
        let B := (forestNodes [u, v]).map (h k)
        have hP_uv : ((forestNodes (u :: v :: T)).map (h k)).Perm (C ++ B) := by
          simp only [C, B, forestNodes, List.flatMap_cons, List.flatMap_nil, List.append_nil,
            List.map_append]
          rw [← List.append_assoc]
          exact List.perm_append_comm
        have hP_ab : ((forestNodes (canonA p :: canonB p :: T)).map (h k)).Perm (C ++ A) := by
          simp only [C, A, forestNodes, List.flatMap_cons, List.flatMap_nil, List.append_nil,
            List.map_append]
          rw [← List.append_assoc]
          exact List.perm_append_comm
        rw [D_perm hP_uv, D_perm hP_ab]
        have hL3 := packed_majorization (A := A) (B := B) (C := C) hL2.1
          (fun x hx => by rcases List.mem_map.mp hx with ⟨r, _, rfl⟩; exact h_bounds k r)
          (fun x hx => by rcases List.mem_map.mp hx with ⟨r, _, rfl⟩; exact h_bounds k r)
          (E := Q k u + Q k v - (Q k (canonA p) + Q k (canonB p)))
          (by linarith [hL2.2.2.2])
          (by dsimp only [A, B]; rw [hL2.2.1, hL2.2.2.1]; ring)
        linarith
      -- Equation (5): `(1/2) * ∑ E_k = c(a, a') - c(u, v) = λ̃(p) - c(u, v)`
      have hA1 : 1 ≤ canonA p := le_trans hb.1 hb.2.2.2.2.1
      have hB1 : 1 ≤ canonB p := le_trans hb.1 hb.2.2.2.2.2.2.1
      have hsum_E :
          (1 / 2 : ℚ) * ∑ k ∈ Finset.range N, (Q k u + Q k v - (Q k (canonA p) + Q k (canonB p))) =
          lamTilde p - cutReward u v := by
        have hsplit : ∑ k ∈ Finset.range N, (Q k u + Q k v - (Q k (canonA p) + Q k (canonB p))) =
            (∑ k ∈ Finset.range N, Q k u) + (∑ k ∈ Finset.range N, Q k v) -
            ((∑ k ∈ Finset.range N, Q k (canonA p)) + (∑ k ∈ Finset.range N, Q k (canonB p))) := by
          simp_rw [Finset.sum_sub_distrib, Finset.sum_add_distrib]
        rw [hsplit,
          sum_Q_eq hu1 (hle_pow u (by omega)),
          sum_Q_eq hv1 (hle_pow v (by omega)),
          sum_Q_eq hA1 (hle_pow (canonA p) (le_trans (le_of_lt hb.2.2.2.2.2.2.2.2.1) hp_leM)),
          sum_Q_eq hB1 (hle_pow (canonB p) (le_trans (le_of_lt hb.2.2.2.2.2.2.2.2.2.1) hp_leM))]
        have hsum_cast : (u : ℚ) + (v : ℚ) = (canonA p : ℚ) + (canonB p : ℚ) := by
          exact_mod_cast huv.trans hb.2.2.2.1.symm
        have hcc := cutReward_canon hp2
        unfold cutReward at hcc ⊢
        linarith
      -- Rearrangement: `(λ̃(p) + G(a :: a' :: T)) / 2 ≤ G(p :: T) = G(P)`
      have hrearr : (lamTilde p + greedyG (canonA p :: canonB p :: T)) / 2 ≤ greedyG P := by
        rw [greedyG_perm hP_perm]
        unfold greedyG
        have hnodes : forestNodes (p :: T) = p :: forestNodes (canonA p :: canonB p :: T) := by
          simp [forestNodes, canonNodes_of_ge_two hp2, List.append_assoc]
        rw [hnodes, List.map_cons]
        exact rearrangement_cons (lamTilde p) ((forestNodes (canonA p :: canonB p :: T)).map lamTilde)
      linarith
    -- Direction 2: `greedyG P ≤ bellmanV P`
    have hge : greedyG P ≤ bellmanV P := by
      by_cases hex : ∃ p ∈ P, 2 ≤ p
      · -- Pick an index `i : Fin P.length` maximizing `P[i]`.
        have hne : (Finset.univ : Finset (Fin P.length)).Nonempty := by
          rcases hex with ⟨p, hp, _⟩
          have hpos : 0 < P.length := List.length_pos_of_mem hp
          exact ⟨⟨0, hpos⟩, Finset.mem_univ _⟩
        rcases Finset.exists_max_image (Finset.univ : Finset (Fin P.length)) (fun i => P[i]) hne with
          ⟨i, _, himax⟩
        let p := P[i]
        have hp_max : ∀ r ∈ P, r ≤ p := by
          intro r hr
          rcases List.getElem_of_mem hr with ⟨k, hk, rfl⟩
          exact himax ⟨k, hk⟩ (Finset.mem_univ _)
        have hp2 : 2 ≤ p := by
          rcases hex with ⟨r, hr, hr2⟩
          exact le_trans hr2 (hp_max r hr)
        have hb := canon_bounds hp2
        let T := P.eraseIdx i.1
        have hP_perm : P.Perm (p :: T) := perm_cons_eraseIdx P i
        have hT1 : ∀ r ∈ T, 1 ≤ r := fun r hr =>
          hP1 r (hP_perm.symm.subset (List.mem_cons_of_mem _ hr))
        have hA1 : 1 ≤ canonA p := le_trans hb.1 hb.2.2.2.2.1
        have hB1 : 1 ≤ canonB p := le_trans hb.1 hb.2.2.2.2.2.2.1
        have hAB1 : ∀ r ∈ canonA p :: canonB p :: T, 1 ≤ r := by
          intro r hr
          simp only [List.mem_cons] at hr
          rcases hr with rfl | rfl | hrT
          · exact hA1
          · exact hB1
          · exact hT1 r hrT
        have hrank_lt : rodRank (canonA p :: canonB p :: T) < n := by
          rw [← hR]
          have hBeq : canonB p = p - canonA p := rfl
          rw [hBeq]
          exact rodRank_eraseIdx_add hA1 hb.2.2.2.2.2.2.2.2.1
        have hIH := ih (rodRank (canonA p :: canonB p :: T)) hrank_lt
          (canonA p :: canonB p :: T) rfl hAB1
        -- Because `p` is the largest rod in `P`, `lamTilde p` is the largest entry in `forestNodes P`.
        have hhead_max : ∀ r ∈ forestNodes (canonA p :: canonB p :: T), r ≤ p := by
          intro r hr
          rcases List.mem_flatMap.mp hr with ⟨q, hq, hrq⟩
          have hrle := (canonNodes_mem_le hrq).2
          simp only [List.mem_cons] at hq
          rcases hq with rfl | rfl | hqT
          · exact le_trans hrle (le_of_lt hb.2.2.2.2.2.2.2.2.1)
          · exact le_trans hrle (le_of_lt hb.2.2.2.2.2.2.2.2.2.1)
          · exact le_trans hrle (hp_max q (hP_perm.symm.subset (List.mem_cons_of_mem _ hqT)))
        have hG_step : greedyG P = (lamTilde p + greedyG (canonA p :: canonB p :: T)) / 2 := by
          rw [greedyG_perm hP_perm]
          unfold greedyG D
          have hnodes : forestNodes (p :: T) = p :: forestNodes (canonA p :: canonB p :: T) := by
            simp [forestNodes, canonNodes_of_ge_two hp2, List.append_assoc]
          rw [hnodes, List.map_cons]
          have hins : insertDesc (lamTilde p) (sortDesc ((forestNodes (canonA p :: canonB p :: T)).map lamTilde)) =
              lamTilde p :: sortDesc ((forestNodes (canonA p :: canonB p :: T)).map lamTilde) := by
            cases hL : sortDesc ((forestNodes (canonA p :: canonB p :: T)).map lamTilde) with
            | nil => rfl
            | cons y ys =>
              have hy : y ∈ (forestNodes (canonA p :: canonB p :: T)).map lamTilde :=
                (sortDesc_perm _).subset (by rw [hL]; exact List.mem_cons_self ..)
              rcases List.mem_map.mp hy with ⟨r, hr, rfl⟩
              have hle' : lamTilde r ≤ lamTilde p := lamTilde_mono (hhead_max r hr)
              simp [insertDesc, hle']
          simp only [sortDesc, hins, discRaw]
        have hcut := cut_le_bellmanV P i hA1 hb.2.2.2.2.2.2.2.2.1
        have hBeq : P[i] - canonA p = canonB p := rfl
        rw [hBeq, cutReward_canon hp2, hIH] at hcut
        linarith
      · -- Every rod in `P` is `< 2`, so `forestNodes P = []` and `greedyG P = 0 ≤ bellmanV P`.
        push_neg at hex
        have hfn : forestNodes P = [] := by
          unfold forestNodes
          rw [List.flatMap_eq_nil_iff]
          intro r hr
          exact canonNodes_of_lt_two (hex r hr)
        have hG0 : greedyG P = 0 := by simp [greedyG, hfn, D, sortDesc, discRaw]
        rw [hG0, bellmanV]
        exact foldr_max_nonneg _
    exact le_antisymm hle hge

/-! ### 6. Theorem 2: Exact Single-Interval Recurrence `RL(Int_m) = R(m + 1)` -/

/-- Exact single-interval recurrence `R(1) = 0`,
`R(p) = 1 - 2^{1 - p} + (R(⌊p/2⌋) + R(⌈p/2⌉)) / 2` for `p ≥ 2`. -/
def R : ℕ → ℚ
  | 0 => 0
  | 1 => 0
  | p + 2 =>
    1 - (1 / 2 : ℚ) ^ (p + 1) + (R ((p + 2) / 2) + R ((p + 3) / 2)) / 2
termination_by n => n

/-- Piecewise-linear discount curve `H(y)`: `0` on `y ≤ 1`, `1 - 2^{1 - n}` at integers `n ≥ 1`,
and linear on each unit interval `[n, n + 1]`. -/
def HCurve (y : ℚ) : ℚ :=
  if y ≤ 1 then 0
  else
    let n := ⌊y⌋₊
    let θ := y - (n : ℚ)
    (1 - (1 / 2 : ℚ) ^ (n - 1)) + θ * (1 / 2 : ℚ) ^ n

lemma HCurve_of_le_one {y : ℚ} (hy : y ≤ 1) : HCurve y = 0 := by
  unfold HCurve; rw [if_pos hy]

lemma HCurve_nat {n : ℕ} (hn : 1 ≤ n) : HCurve (n : ℚ) = 1 - (1 / 2 : ℚ) ^ (n - 1) := by
  rcases eq_or_lt_of_le hn with rfl | hgt
  · rw [HCurve_of_le_one (by norm_num)]; norm_num
  · unfold HCurve
    have hgt' : ¬ ((n : ℚ) ≤ 1) := by
      have : (1 : ℚ) < (n : ℚ) := by exact_mod_cast hgt
      linarith
    rw [if_neg hgt', Nat.floor_natCast]
    ring

/-- On any unit interval `[n, n + 1]` (`n ≥ 1`), `HCurve` is affine with slope `(1/2)^n`. -/
lemma HCurve_on_unit {n : ℕ} (hn : 1 ≤ n) {y : ℚ} (h1 : (n : ℚ) ≤ y) (h2 : y ≤ (n : ℚ) + 1) :
    HCurve y = (1 - (1 / 2 : ℚ) ^ (n - 1)) + (y - (n : ℚ)) * (1 / 2 : ℚ) ^ n := by
  rcases eq_or_lt_of_le h2 with rfl | hlt
  · have hn1 : 1 ≤ n + 1 := by omega
    have h := HCurve_nat hn1
    push_cast at h
    rw [h]
    have hn_eq : n = (n - 1) + 1 := by omega
    have hpow : (1 / 2 : ℚ) ^ n = (1 / 2 : ℚ) * (1 / 2 : ℚ) ^ (n - 1) := by
      conv_lhs => rw [hn_eq, pow_succ']
    rw [hpow]
    ring
  · by_cases hy1 : y ≤ 1
    · have hneq1 : n = 1 := by
        have : (n : ℚ) ≤ 1 := le_trans h1 hy1
        have : n ≤ 1 := by exact_mod_cast this
        omega
      subst hneq1
      have hy_eq : y = 1 := le_antisymm hy1 (by simpa using h1)
      rw [hy_eq, HCurve_of_le_one le_rfl]
      norm_num
    · unfold HCurve
      rw [if_neg hy1]
      have hfl : ⌊y⌋₊ = n := by
        rw [Nat.floor_eq_iff (by linarith [show (0 : ℚ) ≤ (n : ℚ) from Nat.cast_nonneg n])]
        exact ⟨h1, hlt⟩
      rw [hfl]

/-- For ANY packed list `A` of mass `max 0 (y - 1)` (`y ≥ 0`), `D(A) = HCurve y`. -/
theorem D_eq_HCurve_of_packed {A : List ℚ} {y : ℚ} (hy0 : 0 ≤ y)
    (hA : IsPacked A) (hsum : A.sum = max 0 (y - 1)) :
    D A = HCurve y := by
  by_cases hy1 : y ≤ 1
  · rw [HCurve_of_le_one hy1]
    have hsum0 : A.sum = 0 := by rw [hsum, max_eq_left (by linarith)]
    have hall0 : ∀ x ∈ sortDesc A, x = 0 := by
      have hgen : ∀ L : List ℚ, (∀ x ∈ L, 0 ≤ x) → L.sum = 0 → ∀ x ∈ L, x = 0 := by
        intro L; induction L with
        | nil => simp
        | cons a t ih =>
          intro hnn hs
          have ha := hnn a (List.mem_cons_self ..)
          have ht : 0 ≤ t.sum := List.sum_nonneg fun z hz => hnn z (List.mem_cons_of_mem _ hz)
          simp only [List.sum_cons] at hs
          intro x hx; simp only [List.mem_cons] at hx
          rcases hx with rfl | hxt
          · linarith
          · exact ih (fun z hz => hnn z (List.mem_cons_of_mem _ hz)) (by linarith) x hxt
      exact hgen (sortDesc A) (fun x hx => (hA.1 x ((sortDesc_perm A).subset hx)).1)
        ((sortDesc_perm A).sum_eq.trans hsum0)
    have hdisc0 : ∀ L : List ℚ, (∀ x ∈ L, x = 0) → discRaw L = 0 := by
      intro L; induction L with
      | nil => simp [discRaw]
      | cons a t ih =>
        intro h0
        simp only [discRaw, h0 a (List.mem_cons_self ..), ih (fun z hz => h0 z (List.mem_cons_of_mem _ hz))]
        norm_num
    exact hdisc0 (sortDesc A) hall0
  · push_neg at hy1
    let n := ⌊y⌋₊
    have hn1 : 1 ≤ n := Nat.le_floor (by exact_mod_cast le_of_lt hy1)
    have hny : (n : ℚ) ≤ y := Nat.floor_le hy0
    have hyn : y < (n : ℚ) + 1 := Nat.lt_floor_add_one y
    have hAsum : A.sum = y - 1 := by rw [hsum, max_eq_right (by linarith)]
    let B : List ℚ := List.replicate (n - 1) 1 ++ [y - (n : ℚ)]
    have hB_pack : IsPacked B := by
      refine isPacked_append_of_zero_one (fun x hx => Or.inr (List.eq_of_mem_replicate hx))
        (isPacked_singleton (by linarith) (by linarith))
    have hB_sum : B.sum = y - 1 := by
      simp only [B, List.sum_append, List.sum_replicate, nsmul_eq_mul, mul_one, List.sum_singleton]
      rw [Nat.cast_sub hn1, Nat.cast_one]
      ring
    have hDA_eq_DB : D A = D B := by
      let M' := max A.length B.length
      have h1 : (sortDesc A).length ≤ M' := by rw [(sortDesc_perm A).length_eq]; exact le_max_left _ _
      have h2 : (sortDesc B).length ≤ M' := by rw [(sortDesc_perm B).length_eq]; exact le_max_right _ _
      have htop_eq : ∀ k, topSum k A = topSum k B := fun k => by
        rw [hA.2 k, hB_pack.2 k, hAsum, hB_sum]
      unfold D
      rw [finite_abel_sum M' (sortDesc A) h1, finite_abel_sum M' (sortDesc B) h2]
      have htop_eq' : ∀ k, ((sortDesc A).take k).sum = ((sortDesc B).take k).sum := htop_eq
      simp_rw [htop_eq']
    rw [hDA_eq_DB]
    -- Compute `D B = HCurve y` directly!
    have hB_pw : B.Pairwise (· ≥ ·) := by
      dsimp [B]
      rw [List.pairwise_append]
      refine ⟨?_, by simp, ?_⟩
      · rw [List.pairwise_replicate]; exact Or.inr le_rfl
      · intro a ha b hb
        rw [List.eq_of_mem_replicate ha, List.mem_singleton.mp hb]
        linarith
    have hDB : D B = discRaw B := by unfold D; rw [sortDesc_eq_self hB_pw]
    rw [hDB]
    have hdisc_rep : ∀ (m : ℕ) (θ : ℚ),
        discRaw (List.replicate m 1 ++ [θ]) = (1 - (1 / 2 : ℚ) ^ m) + θ * (1 / 2 : ℚ) ^ (m + 1) := by
      intro m θ
      induction m with
      | zero => simp [discRaw]; ring
      | succ m ihm =>
        simp only [List.replicate_succ, List.cons_append, discRaw, ihm]
        rw [pow_succ', pow_succ']
        ring
    rw [hdisc_rep (n - 1) (y - (n : ℚ)), HCurve_on_unit hn1 hny (le_of_lt hyn)]
    have hsub : n - 1 + 1 = n := by omega
    rw [hsub]

/-- Midpoint affineness of `x ↦ HCurve (x / 2^j)` on `[⌊p/2⌋, ⌈p/2⌉]` for every `p ≥ 2, j ≥ 0`. -/
lemma HCurve_midpoint (p j : ℕ) :
    HCurve ((p : ℚ) / (2 : ℚ) ^ (j + 1)) =
      (HCurve (((p / 2 : ℕ) : ℚ) / (2 : ℚ) ^ j) +
       HCurve ((((p + 1) / 2 : ℕ) : ℚ) / (2 : ℚ) ^ j)) / 2 := by
  by_cases heven : p % 2 = 0
  · have h1 : (p + 1) / 2 = p / 2 := by omega
    have h2 : ((p / 2 : ℕ) : ℚ) / (2 : ℚ) ^ j = (p : ℚ) / (2 : ℚ) ^ (j + 1) := by
      have hp : p = 2 * (p / 2) := by omega
      conv_rhs => rw [hp, pow_succ']
      push_cast
      have hne : (2 : ℚ) ^ j ≠ 0 := by positivity
      field_simp [hne]
    rw [h1, h2]
    ring
  · -- `p = 2 * q + 1`, so `p / 2 = q`, `(p + 1) / 2 = q + 1`, and `q / 2^j`, `(q+1)/2^j` lie in the SAME unit interval `[m, m+1]`!
    let q := p / 2
    have hp : p = 2 * q + 1 := by omega
    have hq1 : (p + 1) / 2 = q + 1 := by omega
    let m := q / 2 ^ j
    have hm_le : m * 2 ^ j ≤ q := Nat.div_mul_le_self q (2 ^ j)
    have hq_lt : q < (m + 1) * 2 ^ j := by
      have h1 := Nat.lt_div_mul_add (a := q) (Nat.pow_pos (by norm_num : 0 < 2) (n := j))
      have h2 : (m + 1) * 2 ^ j = m * 2 ^ j + 2 ^ j := by ring
      dsimp [m] at h2 ⊢
      omega
    have hq1_le : q + 1 ≤ (m + 1) * 2 ^ j := by omega
    have h2pos : (0 : ℚ) < (2 : ℚ) ^ j := by positivity
    have hlo : (m : ℚ) ≤ (q : ℚ) / (2 : ℚ) ^ j := by
      rw [le_div_iff₀ h2pos]; exact_mod_cast hm_le
    have hhi : ((q + 1 : ℕ) : ℚ) / (2 : ℚ) ^ j ≤ (m : ℚ) + 1 := by
      rw [div_le_iff₀ h2pos]
      have : ((q + 1 : ℕ) : ℚ) ≤ ((m + 1 : ℕ) : ℚ) * (2 : ℚ) ^ j := by exact_mod_cast hq1_le
      push_cast at this ⊢; linarith
    have hmid_eq : (p : ℚ) / (2 : ℚ) ^ (j + 1) =
        ((q : ℚ) / (2 : ℚ) ^ j + ((q + 1 : ℕ) : ℚ) / (2 : ℚ) ^ j) / 2 := by
      rw [hp, pow_succ']
      push_cast
      have hne : (2 : ℚ) ^ j ≠ 0 := ne_of_gt h2pos
      field_simp [hne]
      ring
    rw [hq1, hmid_eq]
    by_cases hm0 : m = 0
    · -- All three points lie in `[0, 1]`, where `HCurve = 0`.
      rw [hm0] at hhi
      simp only [Nat.cast_zero, zero_add] at hhi
      have hlo1 : (q : ℚ) / (2 : ℚ) ^ j ≤ 1 := by
        have : (q : ℚ) ≤ ((q + 1 : ℕ) : ℚ) := by push_cast; linarith
        exact le_trans (div_le_div_of_nonneg_right this (le_of_lt h2pos)) hhi
      have hmid1 : ((q : ℚ) / (2 : ℚ) ^ j + ((q + 1 : ℕ) : ℚ) / (2 : ℚ) ^ j) / 2 ≤ 1 := by linarith
      rw [HCurve_of_le_one hmid1, HCurve_of_le_one hlo1, HCurve_of_le_one hhi]
      ring
    · -- `m ≥ 1`: all three points lie in `[m, m + 1]`, where `HCurve` is affine!
      have hm1 : 1 ≤ m := Nat.pos_of_ne_zero hm0
      have hlo_le_hi : (q : ℚ) / (2 : ℚ) ^ j ≤ ((q + 1 : ℕ) : ℚ) / (2 : ℚ) ^ j := by
        apply div_le_div_of_nonneg_right _ (le_of_lt h2pos)
        push_cast; linarith
      rw [HCurve_on_unit hm1 (by linarith) (by linarith),
          HCurve_on_unit hm1 hlo (le_trans hlo_le_hi hhi),
          HCurve_on_unit hm1 (le_trans hlo hlo_le_hi) hhi]
      ring

/-- **Theorem 2 (Single-interval exact formula)**:
for every integer `p ≥ 1`, `G([p]) = R(p)`. -/
theorem single_rod_eq_R : ∀ p : ℕ, 1 ≤ p → greedyG [p] = R p := by
  -- First express `greedyG [p] = ∑_{j < N} HCurve (p / 2^j)` whenever `p ≤ 2^N`.
  have hG_sum : ∀ (p N : ℕ), 1 ≤ p → p ≤ 2 ^ N →
      greedyG [p] = ∑ j ∈ Finset.range N, HCurve ((p : ℚ) / (2 : ℚ) ^ j) := by
    intro p N hp1 hpN
    rw [greedy_layer_decomp (N := N) (fun r hr => by rw [List.mem_singleton.mp hr]; exact hpN)]
    refine Finset.sum_congr rfl fun j _ => ?_
    have hfor : forestNodes [p] = canonNodes p := by simp [forestNodes]
    rw [hfor]
    have hL1 := single_rod_packed j p hp1
    exact D_eq_HCurve_of_packed (by positivity) hL1.1 hL1.2
  intro p
  induction p using Nat.strong_induction_on with
  | _ p ih =>
    intro hp1
    match p with
    | 0 => omega
    | 1 =>
      rw [hG_sum 1 1 le_rfl (by norm_num)]
      simp [R, HCurve_of_le_one le_rfl]
    | q + 2 =>
      let N := q + 2
      have hN : q + 2 ≤ 2 ^ N := le_of_lt Nat.lt_two_pow_self
      have hN_half1 : (q + 2) / 2 ≤ 2 ^ (N - 1) := by
        have h1 : q + 2 ≤ 2 * 2 ^ (N - 1) := by
          have : 2 * 2 ^ (N - 1) = 2 ^ N := by
            have hN1 : 1 ≤ N := by omega
            conv_rhs => rw [show N = (N - 1) + 1 by omega, pow_succ']
          omega
        omega
      have hN_half2 : (q + 3) / 2 ≤ 2 ^ (N - 1) := by
        have h1 : q + 3 ≤ 2 ^ N := by
          -- Actually take `N = q + 3` so `q + 3 ≤ 2^(N - 1) * 2 = 2^N`!
          -- Wait: `q + 2 < 2^(q + 2)` already implies `q + 3 ≤ 2^(q + 2) = 2^N`!
          exact Nat.lt_two_pow_self
        have h2 : 2 * 2 ^ (N - 1) = 2 ^ N := by
          conv_rhs => rw [show N = (N - 1) + 1 by omega, pow_succ']
        omega
      rw [hG_sum (q + 2) N (by omega) hN]
      have hN_eq : N = (N - 1) + 1 := by omega
      conv_lhs => rw [hN_eq, Finset.sum_range_succ']
      simp only [pow_zero, div_one]
      rw [HCurve_nat (by omega)]
      simp_rw [HCurve_midpoint (q + 2)]
      rw [← Finset.sum_div, Finset.sum_add_distrib, ← hG_sum ((q + 2) / 2) (N - 1) (by omega) hN_half1,
        ← hG_sum ((q + 3) / 2) (N - 1) (by omega) hN_half2]
      rw [ih ((q + 2) / 2) (by omega) (by omega), ih ((q + 3) / 2) (by omega) (by omega)]
      simp only [R]
      have hsub : q + 2 - 1 = q + 1 := by omega
      rw [hsub]
      ring

/-- **Main Corollary (`RL(Int_m) = R(m + 1)`)**:
for every integer `m ≥ 0`, the exact Bellman value of the single-interval class `Int_m`
(rod measure `m + 1`) equals `R(m + 1)`. -/
theorem bellman_single_interval (m : ℕ) : bellmanV [m + 1] = R (m + 1) := by
  rw [greedy_optimal [m + 1] (fun p hp => by rw [List.mem_singleton.mp hp]; omega)]
  exact single_rod_eq_R (m + 1) (Nat.succ_pos m)

end IntervalLittlestone

