import Mathlib

/-!
# Refutation of ENTP Conjecture 6.1 (`Count3`) and Exact One-Pass Streaming Bounds
(Ewer, Chae, Zeng, Kim, Lee, arXiv:2410.01600 / TMLR 2025)

We formalize in Lean 4 + Mathlib (zero `sorry`, standard axioms only):
1. **Exact `Count3` definition & histogram identity (`count3_eq_hist_sum`)**:
   For every `n ≥ 1` and `xs : Fin n → ℕ`, the ordered-pair modular count
   `count3 hn xs = |{(i, j) ∈ [n]² : (xs i + xs j + xs (n - 1)) % n = 0}| % n`
   equals `(∑ a < n, hist xs a * hist xs (partner n (xs last) a)) % n`.
2. **Exact correctness and resource bounds of the residue-block counterexample
   (`blockedCount3_eq_count3`, `sqrt_block_bounds`)**:
   For every `n ≥ 1`, every `xs : Fin n → ℕ`, and every block width `B ≥ 1`,
   partitioning residues into `⌈n / B⌉` blocks of width at most `B` computes `count3 hn xs`
   exactly using at most `B` histogram counters per pass and `⌈n / B⌉` passes (two linear scans
   per pass). Taking `B = Nat.sqrt n` yields `O(n^{3/2})` time and `O(√n)` words (`O(√n log n)`
   bits), refuting Conjecture 6.1 under read-only random access.
3. **Exact one-pass streaming lower bound via INDEX reduction
   (`indexInstance_bounds`, `index_reduction_correct`, `streaming_state_lower_bound`)**:
   For every `n ≥ 12`, setting `m = n / 12` yields an explicit family of prefixes of length
   `n - 1` indexed by `bits : Fin m → Bool` with entries in `{1, …, n - 1}` such that appending
   `x_n = n - (m + 1 + q)` yields `count3 = 2` if `bits q = true` and `0` otherwise.
   Consequently, any deterministic one-pass streaming algorithm for known length `n` must
   maintain at least `2 ^ (n / 12)` distinguishable states (at least `⌊n / 12⌋` bits).
4. **Exact one-pass streaming upper bound (`unaryCode_injective`)**:
   Any residue histogram of `n - 1` items in `n` buckets is injectively encoded by its
   `n - 1` strictly increasing separator positions inside `Fin (2 * n - 2)`, i.e. a `(2 * n - 2)`-bit
   binary string (matching the `Θ(n)`-bit bound up to a constant factor).

Note: The constant-word (`O(n² log log n / log n)` time, `O(1)` words) and near-linear
(`O(n log n)` time, `o(n)` bits) bit-packed RAM refinements are proved in prose in
`solutions/entp_count3/solution.md` and tested numerically in `solutions/entp_count3/explore.py`.
-/

open Finset BigOperators

namespace ENTPCount3

/-! ### 1. Exact `Count3` Definition and Histogram Representation -/

/-- The last index `n - 1` of `Fin n` for `n > 0`. -/
def lastIdx {n : ℕ} (hn : 0 < n) : Fin n := ⟨n - 1, Nat.sub_lt hn Nat.one_pos⟩

/-- Exact `Count3` definition matching Algorithm 1 of arXiv:2410.01600:
the number of ordered pairs `(i, j) ∈ [n]²` with `xs i + xs j + xs (n - 1) ≡ 0 (mod n)`,
reduced modulo `n`. -/
def count3 {n : ℕ} (hn : 0 < n) (xs : Fin n → ℕ) : ℕ :=
  ((Finset.univ ×ˢ Finset.univ).filter
    (fun p : Fin n × Fin n => (xs p.1 + xs p.2 + xs (lastIdx hn)) % n = 0)).card % n

/-- Multiplicity of residue `a` modulo `n` in `xs`. -/
def hist {n : ℕ} (xs : Fin n → ℕ) (a : ℕ) : ℕ :=
  (Finset.univ.filter (fun i : Fin n => xs i % n = a)).card

/-- Unique complementary residue `b ∈ [0, n)` satisfying `(a + b + t) % n = 0`. -/
def partner (n t a : ℕ) : ℕ :=
  (n - (a + t) % n) % n

lemma partner_lt {n : ℕ} (hn : 0 < n) (t a : ℕ) : partner n t a < n :=
  Nat.mod_lt _ hn

lemma partner_spec {n : ℕ} (hn : 0 < n) (t a b : ℕ) (hb : b < n) :
    (a + b + t) % n = 0 ↔ b = partner n t a := by
  unfold partner
  have hr : (a + t) % n < n := Nat.mod_lt _ hn
  have hmod : (a + b + t) % n = ((a + t) % n + b) % n := by
    have hre : a + b + t = (a + t) + b := by ring
    rw [hre, Nat.add_mod (a + t) b n, Nat.add_mod ((a + t) % n) b n, Nat.mod_mod]
  rw [hmod]
  by_cases hr0 : (a + t) % n = 0
  · rw [hr0, Nat.sub_zero, Nat.mod_self, zero_add, Nat.mod_eq_of_lt hb]
  · have hrpos : 0 < (a + t) % n := Nat.pos_of_ne_zero hr0
    have hsub : n - (a + t) % n < n := by omega
    rw [Nat.mod_eq_of_lt hsub]
    constructor
    · intro h0
      rcases Nat.dvd_of_mod_eq_zero h0 with ⟨k, hk⟩
      rcases k with _ | _ | k
      · simp at hk; omega
      · simp at hk; omega
      · have : 2 * n ≤ n * (k + 1 + 1) := by nlinarith
        omega
    · intro hbeq
      rw [hbeq]
      have hsum : (a + t) % n + (n - (a + t) % n) = n := by omega
      rw [hsum, Nat.mod_self]

lemma partner_mod_eq {n : ℕ} (t a : ℕ) : partner n t (a % n) = partner n t a := by
  unfold partner
  congr 2
  rw [Nat.add_mod, Nat.mod_mod, ← Nat.add_mod]

lemma partner_involutive {n : ℕ} (hn : 0 < n) (t a : ℕ) (ha : a < n) :
    partner n t (partner n t a) = a := by
  have hb := partner_lt hn t a
  have h1 : (a + partner n t a + t) % n = 0 := (partner_spec hn t a (partner n t a) hb).mpr rfl
  have h2 : (partner n t a + a + t) % n = 0 := by simpa [add_comm, add_left_comm] using h1
  exact ((partner_spec hn t (partner n t a) a ha).mp h2).symm

lemma triple_mod_eq (x y t n : ℕ) :
    (x + y + t) % n = (x % n + y % n + t) % n := by
  calc (x + y + t) % n
      = ((x + y) % n + t % n) % n := Nat.add_mod _ _ _
    _ = ((x % n + y % n) % n + t % n) % n := by rw [Nat.add_mod x y n]
    _ = (x % n + y % n + t) % n := (Nat.add_mod (x % n + y % n) t n).symm

/-- Exact count of matching ordered pairs equals the residue-grouped inner product. -/
lemma card_pairs_eq_hist_sum {n : ℕ} (hn : 0 < n) (xs : Fin n → ℕ) :
    ((Finset.univ ×ˢ Finset.univ).filter
      (fun p : Fin n × Fin n => (xs p.1 + xs p.2 + xs (lastIdx hn)) % n = 0)).card =
      ∑ a ∈ Finset.range n, hist xs a * hist xs (partner n (xs (lastIdx hn)) a) := by
  set t := xs (lastIdx hn)
  have hpart :
      ((Finset.univ ×ˢ Finset.univ).filter
        (fun p : Fin n × Fin n => (xs p.1 + xs p.2 + t) % n = 0)).card =
      ∑ a ∈ Finset.range n,
        (((Finset.univ ×ˢ Finset.univ).filter
          (fun p : Fin n × Fin n => (xs p.1 + xs p.2 + t) % n = 0)).filter
          (fun p => xs p.1 % n = a)).card := by
    rw [← Finset.card_biUnion]
    · congr 1
      ext p
      simp only [Finset.mem_biUnion, Finset.mem_range, Finset.mem_filter,
        Finset.mem_product, Finset.mem_univ, true_and]
      constructor
      · intro hp
        exact ⟨xs p.1 % n, Nat.mod_lt _ hn, hp, rfl⟩
      · rintro ⟨_, _, hp, _⟩
        exact hp
    · intro a _ b _ hab
      dsimp only [Function.onFun]
      rw [Finset.disjoint_left]
      intro p hpa hpb
      simp only [Finset.mem_filter] at hpa hpb
      exact hab (hpa.2.symm.trans hpb.2)
  rw [hpart]
  refine Finset.sum_congr rfl fun a _ => ?_
  have hset :
      ((Finset.univ ×ˢ Finset.univ).filter
        (fun p : Fin n × Fin n => (xs p.1 + xs p.2 + t) % n = 0)).filter
        (fun p => xs p.1 % n = a) =
      (Finset.univ.filter (fun i : Fin n => xs i % n = a)) ×ˢ
      (Finset.univ.filter (fun j : Fin n => xs j % n = partner n t a)) := by
    ext ⟨i, j⟩
    simp only [Finset.mem_filter, Finset.mem_product, Finset.mem_univ, true_and]
    constructor
    · rintro ⟨hsum, hi⟩
      refine ⟨hi, ?_⟩
      have hsum' : (a + xs j % n + t) % n = 0 := by
        rw [← hi, ← triple_mod_eq]
        exact hsum
      exact (partner_spec hn t a (xs j % n) (Nat.mod_lt _ hn)).mp hsum'
    · rintro ⟨hi, hj⟩
      refine ⟨?_, hi⟩
      have hsum' : (a + xs j % n + t) % n = 0 :=
        (partner_spec hn t a (xs j % n) (Nat.mod_lt _ hn)).mpr hj
      rw [triple_mod_eq, hi]
      exact hsum'
  rw [hset, Finset.card_product]
  rfl

/-- **Theorem 1**: Exact residue-histogram formula for `Count3`. -/
theorem count3_eq_hist_sum {n : ℕ} (hn : 0 < n) (xs : Fin n → ℕ) :
    count3 hn xs =
      (∑ a ∈ Finset.range n, hist xs a * hist xs (partner n (xs (lastIdx hn)) a)) % n := by
  unfold count3
  rw [card_pairs_eq_hist_sum hn xs]

/-! ### 2. Residue-Block Algorithm (Refutation of Conjecture 6.1) -/

/-- One block pass over residues `a ∈ [lo, hi)`:
first scan computes `hist xs a` for `a ∈ [lo, hi)` (at most `hi - lo` counters);
second scan scans `j : Fin n`, checks if `partner n t (xs j) ∈ [lo, hi)`, and adds its count. -/
def blockPass {n : ℕ} (hn : 0 < n) (xs : Fin n → ℕ) (lo hi : ℕ) : ℕ :=
  ∑ j : Fin n,
    if lo ≤ partner n (xs (lastIdx hn)) (xs j) ∧ partner n (xs (lastIdx hn)) (xs j) < hi
    then hist xs (partner n (xs (lastIdx hn)) (xs j)) else 0

/-- Full residue-block algorithm with block width `B` (`⌈n / B⌉` blocks). -/
def blockedCount3 {n : ℕ} (hn : 0 < n) (xs : Fin n → ℕ) (B : ℕ) : ℕ :=
  (∑ k ∈ Finset.range ((n + B - 1) / B), blockPass hn xs (k * B) (min ((k + 1) * B) n)) % n

lemma blockPass_eq_sum {n : ℕ} (hn : 0 < n) (xs : Fin n → ℕ) (lo hi : ℕ) (hhi : hi ≤ n) :
    blockPass hn xs lo hi =
      ∑ a ∈ Finset.Ico lo hi, hist xs a * hist xs (partner n (xs (lastIdx hn)) a) := by
  set t := xs (lastIdx hn)
  unfold blockPass
  have h1 :
      (∑ j : Fin n, if lo ≤ partner n t (xs j) ∧ partner n t (xs j) < hi
        then hist xs (partner n t (xs j)) else 0) =
      ∑ j ∈ Finset.univ.filter (fun j : Fin n => partner n t (xs j) ∈ Finset.Ico lo hi),
        hist xs (partner n t (xs j)) := by
    rw [Finset.sum_filter]
    refine Finset.sum_congr rfl fun j _ => ?_
    simp [Finset.mem_Ico]
  rw [h1]
  rw [← Finset.sum_fiberwise_of_maps_to (g := fun j : Fin n => partner n t (xs j))
    (t := Finset.Ico lo hi) (fun j hj => (Finset.mem_filter.mp hj).2)]
  refine Finset.sum_congr rfl fun a ha => ?_
  have ha_lt : a < n := lt_of_lt_of_le (Finset.mem_Ico.mp ha).2 hhi
  have hconst :
      ∑ j ∈ (Finset.univ.filter (fun j : Fin n => partner n t (xs j) ∈ Finset.Ico lo hi)).filter
        (fun j : Fin n => partner n t (xs j) = a), hist xs (partner n t (xs j)) =
      ((Finset.univ.filter (fun j : Fin n => partner n t (xs j) ∈ Finset.Ico lo hi)).filter
        (fun j : Fin n => partner n t (xs j) = a)).card * hist xs a := by
    rw [Finset.sum_congr rfl (fun j hj => by rw [(Finset.mem_filter.mp hj).2]),
      Finset.sum_const, smul_eq_mul]
  rw [hconst, mul_comm]
  congr 1
  have hfib :
      (Finset.univ.filter (fun j : Fin n => partner n t (xs j) ∈ Finset.Ico lo hi)).filter
        (fun j : Fin n => partner n t (xs j) = a) =
      Finset.univ.filter (fun j : Fin n => xs j % n = partner n t a) := by
    ext j
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    constructor
    · rintro ⟨_, hpa⟩
      have hpa' : partner n t (xs j % n) = a := by rwa [partner_mod_eq]
      rw [← hpa', partner_involutive hn t (xs j % n) (Nat.mod_lt _ hn)]
    · intro hj
      have hpa : partner n t (xs j) = a := by
        rw [← partner_mod_eq, hj, partner_involutive hn t a ha_lt]
      exact ⟨hpa ▸ ha, hpa⟩
  rw [hfib]
  rfl

/-- **Theorem 2 (Exact correctness of the residue-block counterexample)**:
For every `n ≥ 1`, every input `xs : Fin n → ℕ`, and every block width `B ≥ 1`,
`blockedCount3 hn xs B = count3 hn xs`. -/
theorem blockedCount3_eq_count3 {n : ℕ} (hn : 0 < n) (xs : Fin n → ℕ)
    {B : ℕ} (hB : 0 < B) :
    blockedCount3 hn xs B = count3 hn xs := by
  rw [count3_eq_hist_sum hn xs]
  unfold blockedCount3
  set m := (n + B - 1) / B with hmdef
  have hmn : n ≤ m * B := by
    have hlt : n + B - 1 < m * B + B := by
      rw [hmdef]
      exact Nat.lt_div_mul_add (a := n + B - 1) hB
    omega
  have hsum :
      ∑ k ∈ Finset.range m, blockPass hn xs (k * B) (min ((k + 1) * B) n) =
      ∑ a ∈ Finset.range n, hist xs a * hist xs (partner n (xs (lastIdx hn)) a) := by
    have hrewrite :
        ∑ k ∈ Finset.range m, blockPass hn xs (k * B) (min ((k + 1) * B) n) =
        ∑ k ∈ Finset.range m,
          ∑ a ∈ Finset.Ico (k * B) (min ((k + 1) * B) n),
            hist xs a * hist xs (partner n (xs (lastIdx hn)) a) :=
      Finset.sum_congr rfl fun k _ => blockPass_eq_sum hn xs _ _ (min_le_right _ _)
    rw [hrewrite, ← Finset.sum_biUnion]
    · congr 1
      ext a
      simp only [Finset.mem_biUnion, Finset.mem_range, Finset.mem_Ico]
      constructor
      · rintro ⟨_, _, _, ha2⟩
        exact lt_of_lt_of_le ha2 (min_le_right _ _)
      · intro ha
        refine ⟨a / B, ?_, Nat.div_mul_le_self a B, lt_min ?_ ha⟩
        · by_contra hge
          push_neg at hge
          have hmul : m * B ≤ (a / B) * B := Nat.mul_le_mul_right B hge
          have hle := Nat.div_mul_le_self a B
          omega
        · have h1 := Nat.lt_div_mul_add (a := a) hB
          have h2 : (a / B + 1) * B = a / B * B + B := by ring
          omega
    · intro k₁ _ k₂ _ hne
      dsimp only [Function.onFun]
      rw [Finset.disjoint_left]
      intro a ha₁ ha₂
      simp only [Finset.mem_Ico] at ha₁ ha₂
      rcases lt_or_gt_of_ne hne with hlt | hgt
      · have hle : (k₁ + 1) * B ≤ k₂ * B := Nat.mul_le_mul_right B hlt
        have : a < (k₁ + 1) * B := lt_of_lt_of_le ha₁.2 (min_le_left _ _)
        omega
      · have hle : (k₂ + 1) * B ≤ k₁ * B := Nat.mul_le_mul_right B hgt
        have : a < (k₂ + 1) * B := lt_of_lt_of_le ha₂.2 (min_le_left _ _)
        omega
  rw [hsum]

/-- Explicit resource bounds at block width `B = Nat.sqrt n`:
at most `Nat.sqrt n` counters per block and at most `Nat.sqrt n + 2` block passes,
so total time is `≤ (Nat.sqrt n + 2) * (2 * n) = O(n^{3/2})` and workspace is `≤ Nat.sqrt n` words. -/
theorem sqrt_block_bounds {n : ℕ} (hn : 0 < n) :
    let B := Nat.sqrt n
    0 < B ∧
    (∀ k, min ((k + 1) * B) n - k * B ≤ B) ∧
    (n + B - 1) / B ≤ B + 2 := by
  intro B
  have hBpos : 0 < B := Nat.sqrt_pos.mpr hn
  refine ⟨hBpos, ?_, ?_⟩
  · intro k
    have h1 : min ((k + 1) * B) n ≤ (k + 1) * B := min_le_left _ _
    have h2 : (k + 1) * B = k * B + B := by ring
    omega
  · have hlt : n < (B + 1) * (B + 1) := Nat.lt_succ_sqrt n
    have hle1 : n + B ≤ (B + 3) * B := by nlinarith
    have hle : n + B - 1 < (B + 3) * B := by omega
    have hdiv : (n + B - 1) / B < B + 3 := (Nat.div_lt_iff_lt_mul hBpos).mpr hle
    omega

/-! ### 3. Exact One-Pass Streaming Lower Bound via INDEX Reduction -/

/-- Prefix of length `n - 1` encoding a bitstring `bits : Fin (n / 12) → Bool`. -/
def indexPrefix (n : ℕ) (bits : Fin (n / 12) → Bool) (i : Fin (n - 1)) : ℕ :=
  if h : i.1 < n / 12 then
    if bits ⟨i.1, h⟩ then 2 * (n / 12 + 1 + i.1) else 4 * (n / 12) + 1
  else
    4 * (n / 12) + 1

/-- Full length-`n` input obtained by appending the query `x_n = n - (n / 12 + 1 + q)`
to `indexPrefix n bits`. -/
def indexInstance (n : ℕ) (bits : Fin (n / 12) → Bool) (q : Fin (n / 12)) (i : Fin n) : ℕ :=
  if h : i.1 < n - 1 then
    indexPrefix n bits ⟨i.1, h⟩
  else
    n - (n / 12 + 1 + q.1)

lemma indexInstance_bounds {n : ℕ} (hn12 : 12 ≤ n) (bits : Fin (n / 12) → Bool)
    (q : Fin (n / 12)) (i : Fin n) :
    1 ≤ indexInstance n bits q i ∧ indexInstance n bits q i < n := by
  have hq : q.1 < n / 12 := q.2
  unfold indexInstance indexPrefix
  split_ifs with h1 h2 h3
  · constructor <;> omega
  · constructor <;> omega
  · constructor <;> omega
  · constructor <;> omega

lemma indexInstance_last {n : ℕ} (hn12 : 12 ≤ n) (bits : Fin (n / 12) → Bool)
    (q : Fin (n / 12)) :
    indexInstance n bits q (lastIdx (by omega : 0 < n)) = n - (n / 12 + 1 + q.1) := by
  unfold indexInstance lastIdx
  simp

/-- **Theorem 3 (INDEX reduction for one-pass streaming)**:
For every `n ≥ 12`, every bitstring `bits : Fin (n / 12) → Bool`, and every query `q : Fin (n / 12)`,
`count3` on `indexInstance n bits q` equals `2` if `bits q = true` and `0` if `bits q = false`. -/
theorem index_reduction_correct {n : ℕ} (hn12 : 12 ≤ n) (bits : Fin (n / 12) → Bool)
    (q : Fin (n / 12)) :
    count3 (by omega : 0 < n) (indexInstance n bits q) = if bits q then 2 else 0 := by
  have hn : 0 < n := by omega
  set k := n / 12 + 1 + q.1 with hkdef
  set last : Fin n := lastIdx hn with hlastdef
  set qIdx : Fin n := ⟨q.1, by have := q.2; omega⟩
  have hq_lt : q.1 < n / 12 := q.2
  have hm1 : 1 ≤ n / 12 := by omega
  have h12m : 12 * (n / 12) ≤ n := by omega
  have hk_lo : n / 12 + 1 ≤ k := by omega
  have hk_hi : k ≤ 2 * (n / 12) := by omega
  have hlast_val : indexInstance n bits q last = n - k := indexInstance_last hn12 bits q
  have hpref_bounds : ∀ r : Fin n, r ≠ last →
      2 * (n / 12) + 2 ≤ indexInstance n bits q r ∧
      indexInstance n bits q r ≤ 4 * (n / 12) + 1 := by
    intro r hr
    have hrlt : r.1 < n - 1 := by
      have hne : r.1 ≠ n - 1 := fun h => hr (Fin.ext h)
      have := r.2
      omega
    unfold indexInstance indexPrefix
    rw [dif_pos hrlt]
    dsimp only
    split_ifs with hrm hbr <;> omega
  have hpref_eq_2k : ∀ r : Fin n, r ≠ last →
      (indexInstance n bits q r = 2 * k ↔ bits q = true ∧ r = qIdx) := by
    intro r hr
    have hrlt : r.1 < n - 1 := by
      have hne : r.1 ≠ n - 1 := fun h => hr (Fin.ext h)
      have := r.2
      omega
    unfold indexInstance indexPrefix
    rw [dif_pos hrlt]
    dsimp only
    constructor
    · intro heq
      split_ifs at heq with hrm hbr
      · have hrq : r.1 = q.1 := by omega
        have hfin : (⟨r.1, hrm⟩ : Fin (n / 12)) = q := Fin.ext hrq
        rw [hfin] at hbr
        exact ⟨hbr, Fin.ext hrq⟩
      · omega
      · omega
    · rintro ⟨hbq, rfl⟩
      rw [dif_pos hq_lt]
      have hfin : (⟨qIdx.1, hq_lt⟩ : Fin (n / 12)) = q := Fin.ext rfl
      rw [hfin, hbq]
      simp [qIdx, hkdef]
  have hpair_iff : ∀ p : Fin n × Fin n,
      (indexInstance n bits q p.1 + indexInstance n bits q p.2 + indexInstance n bits q last) % n = 0 ↔
        bits q = true ∧ ((p.1 = qIdx ∧ p.2 = last) ∨ (p.1 = last ∧ p.2 = qIdx)) := by
    rintro ⟨i, j⟩
    dsimp only
    rw [hlast_val]
    constructor
    · intro hmod0
      by_cases hi : i = last
      · subst hi
        rw [hlast_val] at hmod0
        by_cases hj : j = last
        · subst hj
          rw [hlast_val] at hmod0
          have h3 : (n - k + (n - k) + (n - k)) % n = n - 3 * k := by
            have heq : n - k + (n - k) + (n - k) = n * 2 + (n - 3 * k) := by omega
            rw [heq, Nat.mul_add_mod, Nat.mod_eq_of_lt (by omega)]
          omega
        · have hjb := hpref_bounds j hj
          rcases Nat.dvd_of_mod_eq_zero hmod0 with ⟨c, hc⟩
          have hc2 : c = 2 := by
            rcases c with _ | _ | _ | c
            · simp at hc; omega
            · simp at hc; omega
            · rfl
            · have : 3 * n ≤ n * (c + 1 + 1 + 1) := by nlinarith
              omega
          rw [hc2] at hc
          have hxj2k : indexInstance n bits q j = 2 * k := by omega
          obtain ⟨hbq, hjq⟩ := (hpref_eq_2k j hj).mp hxj2k
          exact ⟨hbq, Or.inr ⟨rfl, hjq⟩⟩
      · by_cases hj : j = last
        · subst hj
          rw [hlast_val] at hmod0
          have hib := hpref_bounds i hi
          rcases Nat.dvd_of_mod_eq_zero hmod0 with ⟨c, hc⟩
          have hc2 : c = 2 := by
            rcases c with _ | _ | _ | c
            · simp at hc; omega
            · simp at hc; omega
            · rfl
            · have : 3 * n ≤ n * (c + 1 + 1 + 1) := by nlinarith
              omega
          rw [hc2] at hc
          have hxi2k : indexInstance n bits q i = 2 * k := by omega
          obtain ⟨hbq, hiq⟩ := (hpref_eq_2k i hi).mp hxi2k
          exact ⟨hbq, Or.inl ⟨hiq, rfl⟩⟩
        · have hib := hpref_bounds i hi
          have hjb := hpref_bounds j hj
          rcases Nat.dvd_of_mod_eq_zero hmod0 with ⟨c, hc⟩
          rcases c with _ | _ | c
          · simp at hc; omega
          · simp at hc; omega
          · have : 2 * n ≤ n * (c + 1 + 1) := by nlinarith
            omega
    · rintro ⟨hbq, ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩⟩
      · have hqne : qIdx ≠ last := by
          intro h; have := congrArg Fin.val h; simp [qIdx, last, lastIdx] at this; omega
        have hxq : indexInstance n bits q qIdx = 2 * k :=
          (hpref_eq_2k qIdx hqne).mpr ⟨hbq, rfl⟩
        rw [hxq, hlast_val]
        have heq : 2 * k + (n - k) + (n - k) = 2 * n := by omega
        rw [heq, Nat.mul_mod_left]
      · have hqne : qIdx ≠ last := by
          intro h; have := congrArg Fin.val h; simp [qIdx, last, lastIdx] at this; omega
        have hxq : indexInstance n bits q qIdx = 2 * k :=
          (hpref_eq_2k qIdx hqne).mpr ⟨hbq, rfl⟩
        rw [hxq, hlast_val]
        have heq : n - k + 2 * k + (n - k) = 2 * n := by omega
        rw [heq, Nat.mul_mod_left]
  unfold count3
  have hfilter :
      (Finset.univ ×ˢ Finset.univ).filter
        (fun p : Fin n × Fin n =>
          (indexInstance n bits q p.1 + indexInstance n bits q p.2 +
            indexInstance n bits q (lastIdx hn)) % n = 0) =
      if bits q then {(qIdx, last), (last, qIdx)} else ∅ := by
    ext p
    simp only [Finset.mem_filter, Finset.mem_product, Finset.mem_univ, true_and]
    rw [← hlastdef, hpair_iff p]
    cases hb : bits q
    · simp
    · simp [Prod.ext_iff]
  rw [hfilter]
  cases hb : bits q
  · simp
  · simp only [↓reduceIte]
    have hne : (qIdx, last) ≠ (last, qIdx) := by
      intro h
      have := congrArg (fun p : Fin n × Fin n => p.1.1) h
      simp [qIdx, last, lastIdx] at this
      omega
    rw [Finset.card_pair hne, Nat.mod_eq_of_lt (by omega)]

/-- **Corollary (Deterministic one-pass streaming state lower bound)**:
Any deterministic one-pass streaming algorithm for known length `n ≥ 12` that summarizes
the first `n - 1` inputs into a finite state space `α` and then outputs `count3` upon
receiving `x_n` must satisfy `Fintype.card α ≥ 2 ^ (n / 12)` (i.e. at least `⌊n / 12⌋` bits). -/
theorem streaming_state_lower_bound {n : ℕ} (hn12 : 12 ≤ n) {α : Type*} [Fintype α]
    (encode : (Fin (n - 1) → ℕ) → α)
    (finish : α → ℕ → ℕ)
    (hcorrect : ∀ (bits : Fin (n / 12) → Bool) (q : Fin (n / 12)),
      finish (encode (indexPrefix n bits)) (n - (n / 12 + 1 + q.1)) =
        count3 (by omega : 0 < n) (indexInstance n bits q)) :
    2 ^ (n / 12) ≤ Fintype.card α := by
  have hinj : Function.Injective (fun bits : Fin (n / 12) → Bool => encode (indexPrefix n bits)) := by
    intro b₁ b₂ heq
    dsimp only at heq
    funext q
    have h1 := hcorrect b₁ q
    have h2 := hcorrect b₂ q
    rw [heq] at h1
    rw [index_reduction_correct hn12 b₁ q] at h1
    rw [index_reduction_correct hn12 b₂ q] at h2
    have hEq : (if b₁ q then (2 : ℕ) else 0) = (if b₂ q then 2 else 0) := h1.symm.trans h2
    cases hb1 : b₁ q <;> cases hb2 : b₂ q <;> simp [hb1, hb2] at hEq ⊢
  have hcard := Fintype.card_le_of_injective _ hinj
  simpa [Fintype.card_fun, Fintype.card_bool, Fintype.card_fin] using hcard

/-! ### 4. Exact One-Pass Streaming Upper Bound (`2n - 2` Bits via Unary Histogram) -/

/-- Separator positions of the stars-and-bars unary histogram encoding for `n - 1` items
distributed across `n` residue buckets: for `r : Fin (n - 1)`, position `r` equals
`r + ∑_{a ≤ r} h a ∈ {0, …, 2n - 3}`. -/
def separatorPos {n : ℕ} (h : Fin n → ℕ) (r : Fin (n - 1)) : ℕ :=
  r.1 + ∑ a ∈ Finset.univ.filter (fun a : Fin n => a.1 ≤ r.1), h a

lemma separatorPos_strictMono {n : ℕ} (h : Fin n → ℕ) :
    StrictMono (separatorPos h) := by
  intro r₁ r₂ hlt
  unfold separatorPos
  have hsub :
      Finset.univ.filter (fun a : Fin n => a.1 ≤ r₁.1) ⊆
      Finset.univ.filter (fun a : Fin n => a.1 ≤ r₂.1) := by
    intro a ha
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha ⊢
    exact le_trans ha (le_of_lt hlt)
  have hle :
      ∑ a ∈ Finset.univ.filter (fun a : Fin n => a.1 ≤ r₁.1), h a ≤
      ∑ a ∈ Finset.univ.filter (fun a : Fin n => a.1 ≤ r₂.1), h a :=
    Finset.sum_le_sum_of_subset hsub
  have hlt_val : r₁.1 < r₂.1 := hlt
  omega

lemma separatorPos_lt {n : ℕ} (h : Fin n → ℕ) (hsum : ∑ a : Fin n, h a = n - 1)
    (r : Fin (n - 1)) :
    separatorPos h r < 2 * n - 2 := by
  unfold separatorPos
  have hle :
      ∑ a ∈ Finset.univ.filter (fun a : Fin n => a.1 ≤ r.1), h a ≤ ∑ a : Fin n, h a :=
    Finset.sum_le_sum_of_subset (Finset.filter_subset _ _)
  have hr : r.1 < n - 1 := r.2
  omega

/-- The `(2n - 2)`-bit indicator vector of the `n - 1` separator positions in stars-and-bars
unary histogram encoding. -/
def unaryCode {n : ℕ} (h : Fin n → ℕ) : Fin (2 * n - 2) → Bool :=
  fun pos => decide (∃ r : Fin (n - 1), separatorPos h r = pos.1)

/-- **Theorem 4 (Exact `2n - 2`-bit streaming upper bound)**:
The `(2n - 2)`-bit unary histogram code `unaryCode` is injective on all residue histograms
of `n - 1` items in `n` buckets (`∑ a, h a = n - 1`). Together with `count3_eq_hist_sum`,
`2n - 2` persistent bits suffice for exact one-pass streaming with known length `n`. -/
theorem unaryCode_injective {n : ℕ}
    {h₁ h₂ : Fin n → ℕ}
    (hs₁ : ∑ a : Fin n, h₁ a = n - 1)
    (hs₂ : ∑ a : Fin n, h₂ a = n - 1)
    (heq : unaryCode h₁ = unaryCode h₂) :
    h₁ = h₂ := by
  have himg :
      Finset.univ.image (separatorPos h₁) = Finset.univ.image (separatorPos h₂) := by
    ext x
    simp only [Finset.mem_image, Finset.mem_univ, true_and]
    constructor
    · rintro ⟨r, rfl⟩
      have hx : separatorPos h₁ r < 2 * n - 2 := separatorPos_lt h₁ hs₁ r
      have hb := congrFun heq ⟨separatorPos h₁ r, hx⟩
      simp only [unaryCode, decide_eq_decide] at hb
      exact hb.mp ⟨r, rfl⟩
    · rintro ⟨r, rfl⟩
      have hx : separatorPos h₂ r < 2 * n - 2 := separatorPos_lt h₂ hs₂ r
      have hb := congrFun heq ⟨separatorPos h₂ r, hx⟩
      simp only [unaryCode, decide_eq_decide] at hb
      exact hb.mpr ⟨r, rfl⟩
  have hsep_eq : ∀ r : Fin (n - 1), separatorPos h₁ r = separatorPos h₂ r := by
    have hOrderIso :
        ∀ (f g : Fin (n - 1) → ℕ), StrictMono f → StrictMono g →
          Finset.univ.image f = Finset.univ.image g → ∀ r, f r = g r := by
      intro f g hf hg hfg r
      obtain ⟨k, hk⟩ := r
      induction k using Nat.strong_induction_on with
      | h k ih =>
        have hmem1 : f ⟨k, hk⟩ ∈ Finset.univ.image g :=
          hfg ▸ Finset.mem_image_of_mem f (Finset.mem_univ _)
        obtain ⟨r₂, _, hr₂⟩ := Finset.mem_image.mp hmem1
        have hmem2 : g ⟨k, hk⟩ ∈ Finset.univ.image f :=
          hfg.symm ▸ Finset.mem_image_of_mem g (Finset.mem_univ _)
        obtain ⟨r₁, _, hr₁⟩ := Finset.mem_image.mp hmem2
        have hr2_ge : k ≤ r₂.1 := by
          by_contra hlt
          push_neg at hlt
          have hprev : f r₂ = g r₂ := ih r₂.1 hlt r₂.2
          have hgt : f r₂ < f ⟨k, hk⟩ := hf hlt
          omega
        have hr1_ge : k ≤ r₁.1 := by
          by_contra hlt
          push_neg at hlt
          have hprev : f r₁ = g r₁ := ih r₁.1 hlt r₁.2
          have hgt : g r₁ < g ⟨k, hk⟩ := hg hlt
          omega
        have hle1 : g ⟨k, hk⟩ ≤ g r₂ := hg.monotone hr2_ge
        have hle2 : f ⟨k, hk⟩ ≤ f r₁ := hf.monotone hr1_ge
        omega
    exact hOrderIso (separatorPos h₁) (separatorPos h₂)
      (separatorPos_strictMono h₁) (separatorPos_strictMono h₂) himg
  have hpref : ∀ r : Fin (n - 1),
      ∑ a ∈ Finset.univ.filter (fun a : Fin n => a.1 ≤ r.1), h₁ a =
      ∑ a ∈ Finset.univ.filter (fun a : Fin n => a.1 ≤ r.1), h₂ a := by
    intro r
    have := hsep_eq r
    unfold separatorPos at this
    omega
  have hlt_n1 : ∀ a : Fin n, a.1 < n - 1 → h₁ a = h₂ a := by
    intro a ha
    have hcur := hpref ⟨a.1, ha⟩
    dsimp only at hcur
    by_cases ha0 : a.1 = 0
    · have hfilt : Finset.univ.filter (fun b : Fin n => b.1 ≤ a.1) = {a} := by
        ext b
        simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton]
        constructor
        · intro hle; exact Fin.ext (by omega)
        · rintro rfl; exact le_refl _
      rw [hfilt, Finset.sum_singleton, Finset.sum_singleton] at hcur
      exact hcur
    · have hprev := hpref ⟨a.1 - 1, by omega⟩
      dsimp only at hprev
      have hsplit : Finset.univ.filter (fun b : Fin n => b.1 ≤ a.1) =
          insert a (Finset.univ.filter (fun b : Fin n => b.1 ≤ a.1 - 1)) := by
        ext b
        simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_insert, Fin.ext_iff]
        omega
      have hnotmem : a ∉ Finset.univ.filter (fun b : Fin n => b.1 ≤ a.1 - 1) := by
        simp only [Finset.mem_filter, Finset.mem_univ, true_and]
        omega
      rw [hsplit, Finset.sum_insert hnotmem, Finset.sum_insert hnotmem] at hcur
      omega
  funext a
  by_cases ha : a.1 < n - 1
  · exact hlt_n1 a ha
  · have ha_val : a.1 = n - 1 := by have := a.2; omega
    have hsplit : ∀ h : Fin n → ℕ,
        ∑ b : Fin n, h b = h a + ∑ b ∈ Finset.univ.erase a, h b :=
      fun h => (Finset.add_sum_erase Finset.univ h (Finset.mem_univ a)).symm
    have herase :
        ∑ b ∈ Finset.univ.erase a, h₁ b = ∑ b ∈ Finset.univ.erase a, h₂ b := by
      refine Finset.sum_congr rfl fun b hb => ?_
      have hne : b ≠ a := Finset.ne_of_mem_erase hb
      have hblt : b.1 < n - 1 := by
        have hbv : b.1 ≠ a.1 := fun h => hne (Fin.ext h)
        have := b.2
        omega
      exact hlt_n1 b hblt
    have h1 := hsplit h₁
    have h2 := hsplit h₂
    omega

end ENTPCount3
