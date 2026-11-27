import Mathlib

/-!
# Product Depth for Cache-Restricted Pointer Chasing
(arXiv:2604.17935)

We formalize in Lean 4 + Mathlib (zero `sorry`, standard axioms only) the three core components of
the resolution of Conjecture `thm:lower` and the refutation of step `(★)`:

1. **Exact output range and readout cardinality bound**:
   - `full_output_range`: For every `1 ≤ k`, `k + 1 ≤ n`, and `v : Fin n`, there exists a
     permutation `π : Equiv.Perm (Fin n)` with `(π ^ k) ⟨0, _⟩ = v`.
   - `readout_cardinality_bound`: Any exact solver whose final query state lies in a finite set
     `Q` with `|Q| ≤ 2^(m * p)` must satisfy `n ≤ 2^(m * p)`.
   - `bandwidth_factor_eq_one`: Whenever `2 ≤ n`, `1 ≤ H`, `1 ≤ m * p`, and `n ≤ 2^(m * p)`,
     `⌈Real.logb 2 n / (H * m * p)⌉₊ = 1`.
2. **Adversarial permutation-probe lower bound (`L * s ≥ k` for `n ≥ 2 * k + 1`)**:
   - `ProbeTree n d`: Adaptive deterministic permutation-probe decision trees of depth `≤ d`.
   - `adversarial_indistinguishable_pair`: Whenever `d < k` and `2 * k + 1 ≤ n`, for EVERY
     `t : ProbeTree n d`, there exist permutations `π₁, π₂ : Equiv.Perm (Fin n)` giving the same
     output `evalTree t π₁ = evalTree t π₂` but different `k`-hop answers `(π₁ ^ k) 0 ≠ (π₂ ^ k) 0`.
   - `probe_lower_bound`: Any exact `ProbeTree n d` solver requires `k ≤ d`.
   - `depth_cache_lower_bound`: If `k ≤ L * s` with `1 ≤ s`, then `⌈(k : ℝ) / s⌉₊ ≤ L`, and
     under `n ≤ 2^(m * p)` this yields `⌈(k : ℝ) / s⌉₊ * ⌈Real.logb 2 n / (H * m * p)⌉₊ ≤ L`.
3. **Refutation of step `(★)` for `n ≥ 16`**:
   - `two_layer_s1_exact`: The explicit 2-layer, `s = 1` adaptive circuit reading `1` then `π(1)`
     computes `π^2(1)` exactly on every permutation `π : Equiv.Perm (Fin n)`, and always has
     `π(1) ∈ {0, 1, π(1)}`.
   - `trace_size_sum_eq`: Over all `π : Equiv.Perm (Fin n)`, the average trace size
     `E|T| = (∑ π, |T(π)|) / n!` equals `3 - 1 / n`.
   - `star_step_refutation`: For every `n ≥ 16`, `(3 - 1 / (n : ℝ)) / ((n : ℝ) - 1) < 1`.
   - `star_step_n16_exact`: At `n = 16`, `(3 - 1 / (16 : ℚ)) / (16 - 1) = 47 / 240 < 1`.

Machine-checked vs prose-only:
- Machine-checked: Full output range of `π^k(1)`, readout cardinality `n ≤ 2^(m*p)`, bandwidth
  factor `= 1`, the complete adaptive adversarial permutation-probe lower bound for `n ≥ 2k + 1`,
  `L ≥ ⌈k/s⌉ * ⌈log_2 n / (Hmp)⌉`, the exact 2-layer `s = 1` circuit trace expectation `3 - 1/n`,
  and the `n ≥ 16` (and exact `47/240 < 1` at `n = 16`) refutation of `(★)`.
- Prose-only (in `solutions/kv_cache_depth/results.md`): The simulation theorem compiling an
  arbitrary local-controller cache-restricted Transformer of depth `L` and cache size `s` into a
  `ProbeTree n (L * s)`, and the explicit 4-coordinate quantized MLP/attention weight matrices for
  the `k = 2, s = 1, L = 2` circuit.
-/

open BigOperators Finset

namespace KVCacheDepth

/-! ### 1. Output Range and Readout Cardinality Bound -/

/-- Build a cycle permutation from a list of distinct elements `l : List (Fin n)`, mapping each
element to the next cyclically and fixing all elements outside `l`. -/
def cyclePerm {n : ℕ} (l : List (Fin n)) (_hl : l.Nodup) : Equiv.Perm (Fin n) :=
  l.formPerm

lemma cyclePerm_apply_get {n : ℕ} {l : List (Fin n)} (hl : l.Nodup)
    {i : ℕ} (hi : i + 1 < l.length) :
    cyclePerm l hl (l[i]'(by omega)) = l[i + 1]'hi := by
  unfold cyclePerm
  have h1 := List.formPerm_apply_getElem l hl i (by omega)
  have hmod : (i + 1) % l.length = i + 1 := Nat.mod_eq_of_lt hi
  simpa [hmod] using h1

lemma cyclePerm_apply_last {n : ℕ} {l : List (Fin n)} (hl : l.Nodup) (hne : l ≠ []) :
    cyclePerm l hl (l[l.length - 1]'(Nat.sub_lt (List.length_pos_iff.2 hne) (by omega))) =
      l[0]'(List.length_pos_iff.2 hne) := by
  unfold cyclePerm
  have hpos : 0 < l.length := List.length_pos_iff.2 hne
  have h1 := List.formPerm_apply_getElem l hl (l.length - 1) (Nat.sub_lt hpos (by omega))
  have hmod : (l.length - 1 + 1) % l.length = 0 := by
    rw [Nat.sub_add_cancel hpos, Nat.mod_self]
  simpa [hmod] using h1

lemma cyclePerm_apply_not_mem {n : ℕ} {l : List (Fin n)} (hl : l.Nodup)
    {x : Fin n} (hx : x ∉ l) :
    cyclePerm l hl x = x := by
  exact List.formPerm_apply_of_notMem hx

lemma cyclePerm_pow_get_zero {n : ℕ} {l : List (Fin n)} (hl : l.Nodup) :
    ∀ j (hj : j < l.length),
      ((cyclePerm l hl) ^ j) (l[0]'(by omega)) = l[j]'hj
  | 0, _ => by simp
  | j + 1, hj => by
      rw [pow_succ', Equiv.Perm.mul_apply, cyclePerm_pow_get_zero hl j (by omega)]
      exact cyclePerm_apply_get hl hj

/-- Pick `r` distinct elements from a finite set `S` of cardinality at least `r`. -/
lemma exists_nodup_list_subset {α : Type*} [DecidableEq α] {S : Finset α} {r : ℕ}
    (hr : r ≤ S.card) :
    ∃ l : List α, l.Nodup ∧ l.length = r ∧ ∀ x ∈ l, x ∈ S := by
  obtain ⟨T, hTS, hTcard⟩ := Finset.exists_subset_card_eq hr
  refine ⟨T.toList, T.nodup_toList, by rw [Finset.length_toList, hTcard], ?_⟩
  intro x hx
  exact hTS (Finset.mem_toList.1 hx)

/-- **Full output range**: for `1 ≤ k` and `k + 1 ≤ n`, every `v : Fin n` occurs as `(π ^ k) 0`. -/
theorem full_output_range {n k : ℕ} (hk : 1 ≤ k) (hn : k + 1 ≤ n) (v : Fin n) :
    ∃ π : Equiv.Perm (Fin n), (π ^ k) ⟨0, by omega⟩ = v := by
  let zero : Fin n := ⟨0, by omega⟩
  by_cases hv : v = zero
  · refine ⟨1, ?_⟩
    simp [hv, zero]
  · let S : Finset (Fin n) := (Finset.univ.erase zero).erase v
    have hScard : k - 1 ≤ S.card := by
      have hv_mem : v ∈ Finset.univ.erase zero := Finset.mem_erase.2 ⟨hv, Finset.mem_univ v⟩
      rw [Finset.card_erase_of_mem hv_mem, Finset.card_erase_of_mem (Finset.mem_univ zero),
        Finset.card_fin]
      omega
    obtain ⟨mid, hmid_nodup, hmid_len, hmid_sub⟩ := exists_nodup_list_subset hScard
    have h0_not_mid : zero ∉ mid := fun h => (Finset.mem_erase.1 (Finset.mem_erase.1 (hmid_sub zero h)).2).1 rfl
    have hv_not_mid : v ∉ mid := fun h => (Finset.mem_erase.1 (hmid_sub v h)).1 rfl
    let l : List (Fin n) := zero :: mid ++ [v]
    have hl_nodup : l.Nodup := by
      dsimp only [l]
      rw [List.cons_append, List.nodup_cons, List.nodup_append]
      refine ⟨?_, hmid_nodup, List.nodup_singleton v, ?_⟩
      · intro h
        rcases List.mem_append.1 h with hm | hv'
        · exact h0_not_mid hm
        · exact hv (List.mem_singleton.1 hv').symm
      · intro a ha b hb hab
        rw [List.mem_singleton.1 hb] at hab
        subst hab
        exact hv_not_mid ha
    have hl_len : l.length = k + 1 := by
      simp [l, hmid_len]; omega
    refine ⟨cyclePerm l hl_nodup, ?_⟩
    have h0_eq : l[0]'(by omega) = zero := by simp [l]
    have hk_eq : l[k]'(by omega) = v := by
      have hlen1 : (zero :: mid).length = k := by simp [hmid_len]; omega
      change ((zero :: mid) ++ [v])[k]'(by simp [hmid_len]; omega) = v
      rw [List.getElem_append_right (by omega)]
      simp [hlen1]
    have hpow := cyclePerm_pow_get_zero hl_nodup k (by omega)
    rw [h0_eq, hk_eq] at hpow
    exact hpow

/-- **Readout cardinality bound**: any exact solver factoring through a state set `Q` of size at
most `2 ^ (m * p)` satisfies `n ≤ 2 ^ (m * p)`. -/
theorem readout_cardinality_bound {n k m p : ℕ} {Q : Type*} [Fintype Q]
    (hk : 1 ≤ k) (hn : k + 1 ≤ n)
    (hQ : Fintype.card Q ≤ 2 ^ (m * p))
    (eval : Equiv.Perm (Fin n) → Q) (decode : Q → Fin n)
    (hexact : ∀ π, decode (eval π) = (π ^ k) ⟨0, by omega⟩) :
    n ≤ 2 ^ (m * p) := by
  have hsurj : Function.Surjective decode := by
    intro v
    obtain ⟨π, hπ⟩ := full_output_range hk hn v
    exact ⟨eval π, by rw [hexact π, hπ]⟩
  have hcard : Fintype.card (Fin n) ≤ Fintype.card Q :=
    Fintype.card_le_of_surjective decode hsurj
  rw [Fintype.card_fin] at hcard
  exact le_trans hcard hQ

/-- **Bandwidth factor equals 1**: when `2 ≤ n ≤ 2^(m * p)` and `1 ≤ H`, `1 ≤ m * p`,
`⌈log_2 n / (H * m * p)⌉₊ = 1`. -/
theorem bandwidth_factor_eq_one {n H m p : ℕ}
    (hn2 : 2 ≤ n) (hH : 1 ≤ H) (hmp : 1 ≤ m * p) (hcap : n ≤ 2 ^ (m * p)) :
    Nat.ceil (Real.logb 2 (n : ℝ) / ((H : ℝ) * (m : ℝ) * (p : ℝ))) = 1 := by
  have hden_pos : (0 : ℝ) < (H : ℝ) * (m : ℝ) * (p : ℝ) := by
    have hH' : (1 : ℝ) ≤ (H : ℝ) := by exact_mod_cast hH
    have hmp' : (1 : ℝ) ≤ ((m * p : ℕ) : ℝ) := by exact_mod_cast hmp
    rw [Nat.cast_mul] at hmp'
    nlinarith
  have hlog_pos : 0 < Real.logb 2 (n : ℝ) := by
    have hn1 : (1 : ℝ) < (n : ℝ) := by exact_mod_cast (by omega : 1 < n)
    exact Real.logb_pos (by norm_num) hn1
  have hlog_le : Real.logb 2 (n : ℝ) ≤ (m : ℝ) * (p : ℝ) := by
    have hn_pos : (0 : ℝ) < (n : ℝ) := by positivity
    have hcapR : (n : ℝ) ≤ (2 : ℝ) ^ (m * p) := by exact_mod_cast hcap
    calc Real.logb 2 (n : ℝ)
        ≤ Real.logb 2 ((2 : ℝ) ^ (m * p)) :=
          Real.logb_le_logb_of_le (by norm_num) hn_pos hcapR
      _ = ((m * p : ℕ) : ℝ) * Real.logb 2 2 := by rw [Real.logb_pow]
      _ = (m : ℝ) * (p : ℝ) := by
          rw [Real.logb_self_eq_one (by norm_num), mul_one, Nat.cast_mul]
  have hmp_le_Hmp : (m : ℝ) * (p : ℝ) ≤ (H : ℝ) * (m : ℝ) * (p : ℝ) := by
    have hH' : (1 : ℝ) ≤ (H : ℝ) := by exact_mod_cast hH
    have hmp' : (0 : ℝ) ≤ (m : ℝ) * (p : ℝ) := by positivity
    nlinarith
  have hratio_pos : 0 < Real.logb 2 (n : ℝ) / ((H : ℝ) * (m : ℝ) * (p : ℝ)) :=
    div_pos hlog_pos hden_pos
  have hratio_le : Real.logb 2 (n : ℝ) / ((H : ℝ) * (m : ℝ) * (p : ℝ)) ≤ 1 := by
    rw [div_le_one hden_pos]
    linarith
  apply le_antisymm
  · exact Nat.ceil_le.2 (by simpa using hratio_le)
  · exact Nat.one_le_iff_ne_zero.2 (ne_of_gt (Nat.ceil_pos.2 hratio_pos))

/-! ### 2. Adversarial Permutation-Probe Lower Bound (`n ≥ 2 * k + 1`) -/

/-- Adaptive deterministic permutation-probe decision tree of depth at most `d`. -/
inductive ProbeTree (n : ℕ) : ℕ → Type
  | output {d : ℕ} (ans : Fin n) : ProbeTree n d
  | probe {d : ℕ} (addr : Fin n) (next : Fin n → ProbeTree n d) : ProbeTree n (d + 1)

/-- Evaluate a probe decision tree against a permutation `π : Equiv.Perm (Fin n)`. -/
def evalTree {n : ℕ} : {d : ℕ} → ProbeTree n d → Equiv.Perm (Fin n) → Fin n
  | _, .output ans, _ => ans
  | _, .probe addr next, π => evalTree (next (π addr)) π

/-- Consistency of a permutation `π` with an adversary state `(path, fixed)`:
`π` maps each non-last element of `path` to its successor in `path`, and fixes every element in
`fixed`. -/
def Consistent {n : ℕ} (path : List (Fin n)) (fixed : Finset (Fin n))
    (π : Equiv.Perm (Fin n)) : Prop :=
  (∀ i (hi : i + 1 < path.length), π (path[i]'(by omega)) = path[i + 1]'hi) ∧
  (∀ x ∈ fixed, π x = x)

/-- Core inductive step of the adversary: starting from any valid partial state `(path, fixed)`
with `path` nonempty starting at `0`, disjoint from `fixed`, and `path.length + fixed.card + d ≤ 2 * k`,
we can follow `t : ProbeTree n d` to a leaf and exhibit two consistent permutations `π₁, π₂`
returning the same tree output yet disagreeing on `(π ^ k) 0`. -/
theorem adversary_walk {n k : ℕ} (hn : 2 * k + 1 ≤ n) :
    ∀ {d : ℕ} (t : ProbeTree n d)
      (path : List (Fin n)) (fixed : Finset (Fin n)),
      path.Nodup →
      (hpath_pos : 0 < path.length) →
      path[0]'hpath_pos = ⟨0, by omega⟩ →
      (∀ x ∈ path, x ∉ fixed) →
      path.length + fixed.card + d ≤ k →
      ∃ π₁ π₂ : Equiv.Perm (Fin n),
        Consistent path fixed π₁ ∧
        Consistent path fixed π₂ ∧
        evalTree t π₁ = evalTree t π₂ ∧
        (π₁ ^ k) ⟨0, by omega⟩ ≠ (π₂ ^ k) ⟨0, by omega⟩
  | d, .output ans, path, fixed, hnodup, hpos, hhead, hdisj, hused_bound => by
      -- Extend `path` with `k - path.length` unused vertices, then branch to two distinct unused `a, b`
      let used : Finset (Fin n) := path.toFinset ∪ fixed
      let avail : Finset (Fin n) := Finset.univ \ used
      have hcard_used : used.card = path.length + fixed.card := by
        rw [Finset.card_union_of_disjoint]
        · rw [List.toFinset_card_of_nodup hnodup]
        · rw [Finset.disjoint_left]
          intro x hx
          exact hdisj x (List.mem_toFinset.1 hx)
      have hcard_avail : (k - path.length) + 2 ≤ avail.card := by
        rw [Finset.card_sdiff, Finset.inter_univ, Finset.card_univ, Fintype.card_fin, hcard_used]
        omega
      obtain ⟨extAll, hext_nodup, hext_len, hext_sub⟩ := exists_nodup_list_subset hcard_avail
      let m := k - path.length
      let ext := extAll.take m
      let a : Fin n := extAll[m]'(by omega)
      let b : Fin n := extAll[m + 1]'(by omega)
      have hab : a ≠ b := by
        intro h
        have hinj := (List.nodup_iff_injective_getElem.1 hext_nodup)
          (a₁ := ⟨m, by omega⟩) (a₂ := ⟨m + 1, by omega⟩) h
        have hval := congrArg Fin.val hinj
        dsimp at hval
        omega
      have hext_disj_used : ∀ x ∈ extAll, x ∉ path ∧ x ∉ fixed := by
        intro x hx
        have hmem := Finset.mem_sdiff.1 (hext_sub x hx)
        have hnot := hmem.2
        rw [Finset.mem_union, not_or, List.mem_toFinset] at hnot
        exact hnot
      have ha_not_ext : a ∉ ext := by
        intro h
        obtain ⟨i, hi, hget⟩ := List.getElem_of_mem h
        have hi' : i < m := by simp [ext, hext_len] at hi; omega
        have hget' : extAll[i]'(by omega) = extAll[m]'(by omega) := by
          simpa [ext, List.getElem_take] using hget
        have hinj := (List.nodup_iff_injective_getElem.1 hext_nodup)
          (a₁ := ⟨i, by omega⟩) (a₂ := ⟨m, by omega⟩) hget'
        have hval := congrArg Fin.val hinj
        dsimp at hval
        omega
      have hb_not_ext : b ∉ ext := by
        intro h
        obtain ⟨i, hi, hget⟩ := List.getElem_of_mem h
        have hi' : i < m := by simp [ext, hext_len] at hi; omega
        have hget' : extAll[i]'(by omega) = extAll[m + 1]'(by omega) := by
          simpa [ext, List.getElem_take] using hget
        have hinj := (List.nodup_iff_injective_getElem.1 hext_nodup)
          (a₁ := ⟨i, by omega⟩) (a₂ := ⟨m + 1, by omega⟩) hget'
        have hval := congrArg Fin.val hinj
        dsimp at hval
        omega
      have hext_nodup' : ext.Nodup := hext_nodup.sublist (List.take_sublist m extAll)
      have hext_len' : ext.length = m := by simp [ext, hext_len]; omega
      have hmk : ∀ (u : Fin n), u ∈ extAll → u ∉ ext →
          ∃ π : Equiv.Perm (Fin n), Consistent path fixed π ∧ (π ^ k) ⟨0, by omega⟩ = u := by
        intro u hu_all hu_not_ext
        let cyc : List (Fin n) := path ++ ext ++ [u]
        have hcyc_nodup : cyc.Nodup := by
          dsimp only [cyc]
          rw [List.append_assoc, List.nodup_append, List.nodup_append]
          refine ⟨hnodup, ⟨hext_nodup', List.nodup_singleton u, ?_⟩, ?_⟩
          · intro x hx y hy hxy
            rw [List.mem_singleton.1 hy] at hxy
            subst hxy
            exact hu_not_ext hx
          · intro x hx y hy hxy
            subst hxy
            rcases List.mem_append.1 hy with hy1 | hy2
            · exact (hext_disj_used x (List.mem_of_mem_take hy1)).1 hx
            · have hxu : x = u := List.mem_singleton.1 hy2
              subst hxu
              exact (hext_disj_used x hu_all).1 hx
        have hcyc_len : cyc.length = k + 1 := by
          simp [cyc, hext_len', m]; omega
        refine ⟨cyclePerm cyc hcyc_nodup, ⟨?_, ?_⟩, ?_⟩
        · intro i hi
          have hi' : i + 1 < cyc.length := by simp [cyc]; omega
          have hg1 : cyc[i]'(by omega) = path[i]'(by omega) := by
            simp [cyc, List.append_assoc, List.getElem_append_left (by omega : i < path.length)]
          have hg2 : cyc[i + 1]'hi' = path[i + 1]'hi := by
            simp [cyc, List.append_assoc, List.getElem_append_left hi]
          rw [← hg1, ← hg2]
          exact cyclePerm_apply_get hcyc_nodup hi'
        · intro x hx
          apply cyclePerm_apply_not_mem hcyc_nodup
          intro hmem
          simp only [cyc, List.mem_append, List.mem_singleton] at hmem
          rcases hmem with (hp | he) | hu
          · exact hdisj x hp hx
          · exact (hext_disj_used x (List.mem_of_mem_take he)).2 hx
          · subst hu
            exact (hext_disj_used x hu_all).2 hx
        · have h0 : cyc[0]'(by omega) = ⟨0, by omega⟩ := by
            rw [← hhead]
            simp [cyc, List.append_assoc, List.getElem_append_left hpos]
          have hk : cyc[k]'(by omega) = u := by
            have hlen_pe : (path ++ ext).length = k := by simp [hext_len', m]; omega
            dsimp only [cyc]
            rw [List.getElem_append_right (by omega)]
            simp [hlen_pe]
          have hpow := cyclePerm_pow_get_zero hcyc_nodup k (by omega)
          rw [h0, hk] at hpow
          exact hpow
      obtain ⟨π₁, hc₁, hp₁⟩ := hmk a (List.getElem_mem _) ha_not_ext
      obtain ⟨π₂, hc₂, hp₂⟩ := hmk b (List.getElem_mem _) hb_not_ext
      refine ⟨π₁, π₂, hc₁, hc₂, rfl, ?_⟩
      rw [hp₁, hp₂]
      exact hab
  | d + 1, .probe addr next, path, fixed, hnodup, hpos, hhead, hdisj, hused_bound => by
      -- Case 1: addr ∈ fixed → answer addr, keep state unchanged
      by_cases hfix : addr ∈ fixed
      · obtain ⟨π₁, π₂, hc₁, hc₂, heq, hne⟩ :=
          adversary_walk hn (next addr) path fixed hnodup hpos hhead hdisj (by omega)
        refine ⟨π₁, π₂, hc₁, hc₂, ?_, hne⟩
        simp only [evalTree, hc₁.2 addr hfix, hc₂.2 addr hfix, heq]
      · -- Case 2: addr ∈ path
        by_cases hmem : addr ∈ path
        · obtain ⟨i, hi, hget⟩ := List.getElem_of_mem hmem
          by_cases hlast : i + 1 < path.length
          · -- Non-tip vertex of path: answer is already determined as path[i + 1]
            let ans := path[i + 1]'hlast
            obtain ⟨π₁, π₂, hc₁, hc₂, heq, hne⟩ :=
              adversary_walk hn (next ans) path fixed hnodup hpos hhead hdisj (by omega)
            refine ⟨π₁, π₂, hc₁, hc₂, ?_, hne⟩
            have h1 : π₁ addr = ans := by rw [← hget]; exact hc₁.1 i hlast
            have h2 : π₂ addr = ans := by rw [← hget]; exact hc₂.1 i hlast
            simp only [evalTree, h1, h2, heq]
          · -- Tip vertex of path: pick a fresh unused vertex `u` and append it to `path`
            have hi_last : i = path.length - 1 := by omega
            let used : Finset (Fin n) := path.toFinset ∪ fixed
            let avail : Finset (Fin n) := Finset.univ \ used
            have hcard_used : used.card = path.length + fixed.card := by
              rw [Finset.card_union_of_disjoint]
              · rw [List.toFinset_card_of_nodup hnodup]
              · rw [Finset.disjoint_left]
                intro x hx
                exact hdisj x (List.mem_toFinset.1 hx)
            have havail_pos : 0 < avail.card := by
              rw [Finset.card_sdiff, Finset.inter_univ, Finset.card_univ, Fintype.card_fin,
                hcard_used]
              omega
            obtain ⟨u, hu⟩ := Finset.card_pos.1 havail_pos
            have hu_not : u ∉ path ∧ u ∉ fixed := by
              have hnot := (Finset.mem_sdiff.1 hu).2
              rw [Finset.mem_union, not_or, List.mem_toFinset] at hnot
              exact hnot
            let path' : List (Fin n) := path ++ [u]
            have hnodup' : path'.Nodup := by
              rw [List.nodup_append]
              refine ⟨hnodup, List.nodup_singleton u, ?_⟩
              intro x hx y hy hxy
              rw [List.mem_singleton.1 hy] at hxy
              subst hxy
              exact hu_not.1 hx
            have hpos' : 0 < path'.length := by simp [path']
            have hhead' : path'[0]'hpos' = ⟨0, by omega⟩ := by
              rw [← hhead]
              simp [path', List.getElem_append_left hpos]
            have hdisj' : ∀ x ∈ path', x ∉ fixed := by
              intro x hx
              rcases List.mem_append.1 hx with hp | hu'
              · exact hdisj x hp
              · rw [List.mem_singleton.1 hu']; exact hu_not.2
            have hlen' : path'.length = path.length + 1 := by simp [path']
            obtain ⟨π₁, π₂, hc₁, hc₂, heq, hne⟩ :=
              adversary_walk hn (next u) path' fixed hnodup' hpos' hhead' hdisj' (by omega)
            have hcons : ∀ π, Consistent path' fixed π → Consistent path fixed π ∧ π addr = u := by
              intro π hc
              refine ⟨⟨fun j hj => ?_, hc.2⟩, ?_⟩
              · have hj' : j + 1 < path'.length := by simp [path']; omega
                have hstep := hc.1 j hj'
                have hg1 : path'[j]'(by omega) = path[j]'(by omega) := by
                  simp [path', List.getElem_append_left (by omega : j < path.length)]
                have hg2 : path'[j + 1]'hj' = path[j + 1]'hj := by
                  simp [path', List.getElem_append_left hj]
                rw [hg1, hg2] at hstep
                exact hstep
              · have hi' : i + 1 < path'.length := by simp [path']; omega
                have hstep := hc.1 i hi'
                have hg1 : path'[i]'(by omega) = addr := by
                  rw [← hget]
                  simp [path', List.getElem_append_left hi]
                have hg2 : path'[i + 1]'hi' = u := by
                  have hsub : i + 1 - path.length = 0 := by omega
                  dsimp only [path']
                  rw [List.getElem_append_right (by omega)]
                  simp [hsub]
                rw [hg1, hg2] at hstep
                exact hstep
            rcases hcons π₁ hc₁ with ⟨hc₁', h1⟩
            rcases hcons π₂ hc₂ with ⟨hc₂', h2⟩
            refine ⟨π₁, π₂, hc₁', hc₂', ?_, hne⟩
            simp only [evalTree, h1, h2, heq]
        · -- Case 3: addr is completely fresh (neither in path nor in fixed) → make it a fixed point
          let fixed' : Finset (Fin n) := insert addr fixed
          have hdisj' : ∀ x ∈ path, x ∉ fixed' := by
            intro x hx hxf
            rcases Finset.mem_insert.1 hxf with rfl | hxf'
            · exact hmem hx
            · exact hdisj x hx hxf'
          have hcard_fixed' : fixed'.card = fixed.card + 1 := Finset.card_insert_of_notMem hfix
          obtain ⟨π₁, π₂, hc₁, hc₂, heq, hne⟩ :=
            adversary_walk hn (next addr) path fixed' hnodup hpos hhead hdisj' (by omega)
          have hcons : ∀ π, Consistent path fixed' π → Consistent path fixed π ∧ π addr = addr :=
            fun π hc => ⟨⟨hc.1, fun x hx => hc.2 x (Finset.mem_insert_of_mem hx)⟩,
              hc.2 addr (Finset.mem_insert_self addr fixed)⟩
          rcases hcons π₁ hc₁ with ⟨hc₁', h1⟩
          rcases hcons π₂ hc₂ with ⟨hc₂', h2⟩
          refine ⟨π₁, π₂, hc₁', hc₂', ?_, hne⟩
          simp only [evalTree, h1, h2, heq]

/-- **Adversarial indistinguishability**: for `n ≥ 2 * k + 1` and `d < k`, no depth-`d` adaptive
permutation-probe tree can compute `(π ^ k) 0` on all permutations `π`. -/
theorem adversarial_indistinguishable_pair {n k d : ℕ}
    (hn : 2 * k + 1 ≤ n) (hd : d < k) (t : ProbeTree n d) :
    ∃ π₁ π₂ : Equiv.Perm (Fin n),
      evalTree t π₁ = evalTree t π₂ ∧
      (π₁ ^ k) ⟨0, by omega⟩ ≠ (π₂ ^ k) ⟨0, by omega⟩ := by
  let path₀ : List (Fin n) := [⟨0, by omega⟩]
  obtain ⟨π₁, π₂, _, _, heq, hne⟩ :=
    adversary_walk hn t path₀ ∅ (List.nodup_singleton _) (by simp [path₀]) (by simp [path₀])
      (fun _ _ => Finset.notMem_empty _) (by simp [path₀]; omega)
  exact ⟨π₁, π₂, heq, hne⟩

/-- **Probe lower bound**: any exact `ProbeTree n d` for `k`-hop pointer chasing with `n ≥ 2k + 1`
must make at least `k` probes (`k ≤ d`). -/
theorem probe_lower_bound {n k d : ℕ} (hn : 2 * k + 1 ≤ n) (t : ProbeTree n d)
    (hexact : ∀ π : Equiv.Perm (Fin n), evalTree t π = (π ^ k) ⟨0, by omega⟩) :
    k ≤ d := by
  by_contra hlt
  push_neg at hlt
  obtain ⟨π₁, π₂, heq, hne⟩ := adversarial_indistinguishable_pair hn hlt t
  rw [hexact π₁, hexact π₂] at heq
  exact hne heq

/-- **Depth lower bound from `L * s ≥ k`**: for `s ≥ 1`, `L * s ≥ k` implies `L ≥ ⌈k / s⌉₊`. -/
theorem depth_ge_ceil_div {L s k : ℕ} (hs : 1 ≤ s) (hLs : k ≤ L * s) :
    Nat.ceil ((k : ℝ) / (s : ℝ)) ≤ L := by
  rw [Nat.ceil_le]
  have hs_pos : (0 : ℝ) < (s : ℝ) := by exact_mod_cast hs
  rw [div_le_iff₀ hs_pos]
  exact_mod_cast hLs

/-- **Combined product depth lower bound**: under `n ≥ 2k + 1`, `1 ≤ k`, `2 ≤ n`, `1 ≤ H`,
`1 ≤ s`, `1 ≤ m * p`, if a solver has readout cardinality `≤ 2^(m * p)` and admits an `(L * s)`-probe
simulation, then `L ≥ ⌈k / s⌉₊ * ⌈log_2 n / (H * m * p)⌉₊`. -/
theorem product_depth_lower_bound {n k L s H m p : ℕ} {Q : Type*} [Fintype Q]
    (hk : 1 ≤ k) (hn : 2 * k + 1 ≤ n) (hH : 1 ≤ H) (hs : 1 ≤ s) (hmp : 1 ≤ m * p)
    (hQ : Fintype.card Q ≤ 2 ^ (m * p))
    (eval : Equiv.Perm (Fin n) → Q) (decode : Q → Fin n)
    (hexactQ : ∀ π, decode (eval π) = (π ^ k) ⟨0, by omega⟩)
    (t : ProbeTree n (L * s))
    (hexactT : ∀ π, evalTree t π = (π ^ k) ⟨0, by omega⟩) :
    Nat.ceil ((k : ℝ) / (s : ℝ)) *
      Nat.ceil (Real.logb 2 (n : ℝ) / ((H : ℝ) * (m : ℝ) * (p : ℝ))) ≤ L := by
  have hcap : n ≤ 2 ^ (m * p) := readout_cardinality_bound hk (by omega) hQ eval decode hexactQ
  have hbw : Nat.ceil (Real.logb 2 (n : ℝ) / ((H : ℝ) * (m : ℝ) * (p : ℝ))) = 1 :=
    bandwidth_factor_eq_one (by omega) hH hmp hcap
  have hLs : k ≤ L * s := probe_lower_bound hn t hexactT
  rw [hbw, mul_one]
  exact depth_ge_ceil_div hs hLs

/-! ### 3. Refutation of Step `(★)` for `n ≥ 16` -/

/-- The 2-layer, `s = 1`, `k = 2` adaptive circuit: layer 1 reads position `0` (getting `π 0`),
layer 2 reads position `π 0` (getting `π (π 0)`). -/
def twoLayerCircuit {n : ℕ} (hn : 0 < n) (π : Equiv.Perm (Fin n)) : Fin n :=
  let v1 := π ⟨0, hn⟩
  π v1

/-- Exactness of the 2-layer `s = 1` circuit on every permutation `π`. -/
theorem two_layer_s1_exact {n : ℕ} (hn : 0 < n) (π : Equiv.Perm (Fin n)) :
    twoLayerCircuit hn π = (π ^ 2) ⟨0, hn⟩ := by
  simp [twoLayerCircuit, sq, Equiv.Perm.mul_apply]

/-- Trace set `T(π) ⊆ Fin (n + 1)` containing the query token `0`, the layer-1 selection `1`, and
the layer-2 selection `1 + π(0)` (in `Fin (n + 1)` indexing where `0` is the query token and
`1..n` are data tokens). -/
def traceSet {n : ℕ} (hn : 0 < n) (π : Equiv.Perm (Fin n)) : Finset (Fin (n + 1)) :=
  {⟨0, by omega⟩, ⟨1, by omega⟩, ⟨(π ⟨0, hn⟩).val + 1, by omega⟩}

lemma traceSet_card {n : ℕ} (hn : 0 < n) (π : Equiv.Perm (Fin n)) :
    (traceSet hn π).card = if π ⟨0, hn⟩ = ⟨0, hn⟩ then 2 else 3 := by
  unfold traceSet
  have h01 : (⟨0, by omega⟩ : Fin (n + 1)) ≠ ⟨1, by omega⟩ := by
    intro h; have := congrArg Fin.val h; dsimp at this; omega
  have h0p : (⟨0, by omega⟩ : Fin (n + 1)) ≠ ⟨(π ⟨0, hn⟩).val + 1, by omega⟩ := by
    intro h; have := congrArg Fin.val h; dsimp at this; omega
  split_ifs with hfix
  · have h1p : (⟨(π ⟨0, hn⟩).val + 1, by omega⟩ : Fin (n + 1)) = ⟨1, by omega⟩ := by
      apply Fin.ext; simp [hfix]
    rw [h1p, Finset.pair_eq_singleton]
    rw [Finset.card_insert_of_notMem (by simp only [Finset.mem_singleton, h01, not_false_eq_true]),
      Finset.card_singleton]
  · have h1p : (⟨1, by omega⟩ : Fin (n + 1)) ≠ ⟨(π ⟨0, hn⟩).val + 1, by omega⟩ := by
      intro h
      have hv : (π ⟨0, hn⟩).val = 0 := by
        have := congrArg Fin.val h; dsimp at this; omega
      exact hfix (Fin.ext hv)
    rw [Finset.card_insert_of_notMem (by simp only [Finset.mem_insert, Finset.mem_singleton,
          h01, h0p, or_self, not_false_eq_true]),
      Finset.card_insert_of_notMem (by simp only [Finset.mem_singleton, h1p, not_false_eq_true]),
      Finset.card_singleton]

/-- Exact trace expectation over uniform `π : Equiv.Perm (Fin n)`:
`∑_π |T(π)| = n! * 3 - (n - 1)!`, equivalently `(∑_π |T(π)|) / n! = 3 - 1 / n`. -/
theorem trace_size_expectation {n : ℕ} (hn : 0 < n) :
    (∑ π : Equiv.Perm (Fin n), ((traceSet hn π).card : ℝ)) /
      (Fintype.card (Equiv.Perm (Fin n)) : ℝ) = 3 - 1 / (n : ℝ) := by
  have hfix_card : (Finset.univ.filter (fun π : Equiv.Perm (Fin n) => π ⟨0, hn⟩ = ⟨0, hn⟩)).card * n =
      Fintype.card (Equiv.Perm (Fin n)) := by
    have hbij : ∀ v : Fin n,
        (Finset.univ.filter (fun π : Equiv.Perm (Fin n) => π ⟨0, hn⟩ = v)).card =
        (Finset.univ.filter (fun π : Equiv.Perm (Fin n) => π ⟨0, hn⟩ = ⟨0, hn⟩)).card := by
      intro v
      let τ : Equiv.Perm (Fin n) := Equiv.swap ⟨0, hn⟩ v
      refine Finset.card_bij (fun π _ => τ * π) ?_ ?_ ?_
      · intro π hπ
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hπ ⊢
        simp [Equiv.Perm.mul_apply, hπ, τ]
      · intro π₁ _ π₂ _ h
        exact mul_left_cancel h
      · intro σ hσ
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hσ
        refine ⟨τ * σ, ?_, by simp [τ]⟩
        simp only [Finset.mem_filter, Finset.mem_univ, true_and]
        simp [Equiv.Perm.mul_apply, hσ, τ]
    have hsum := Finset.card_eq_sum_card_fiberwise
      (s := (Finset.univ : Finset (Equiv.Perm (Fin n))))
      (t := (Finset.univ : Finset (Fin n)))
      (f := fun π => π ⟨0, hn⟩) (fun _ _ => Finset.mem_univ _)
    simp_rw [hbij] at hsum
    rw [Finset.sum_const, Finset.card_univ, Finset.card_univ, Fintype.card_fin,
      smul_eq_mul, mul_comm] at hsum
    exact hsum.symm
  have hsum_card : ∑ π : Equiv.Perm (Fin n), ((traceSet hn π).card : ℝ) =
      3 * (Fintype.card (Equiv.Perm (Fin n)) : ℝ) -
        ((Finset.univ.filter (fun π : Equiv.Perm (Fin n) => π ⟨0, hn⟩ = ⟨0, hn⟩)).card : ℝ) := by
    have h1 : ∀ π : Equiv.Perm (Fin n),
        ((traceSet hn π).card : ℝ) = 3 - if π ⟨0, hn⟩ = ⟨0, hn⟩ then (1 : ℝ) else 0 := by
      intro π
      rw [traceSet_card hn π]
      split_ifs <;> norm_num
    simp_rw [h1, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul,
      Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const, nsmul_eq_mul, mul_one]
    ring
  have hcard_pos : (0 : ℝ) < (Fintype.card (Equiv.Perm (Fin n)) : ℝ) := by
    rw [Fintype.card_perm, Fintype.card_fin]
    exact_mod_cast Nat.factorial_pos n
  have hn_pos : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
  have hfixR : ((Finset.univ.filter (fun π : Equiv.Perm (Fin n) => π ⟨0, hn⟩ = ⟨0, hn⟩)).card : ℝ) *
      (n : ℝ) = (Fintype.card (Equiv.Perm (Fin n)) : ℝ) := by
    exact_mod_cast hfix_card
  rw [hsum_card]
  field_simp [ne_of_gt hcard_pos, ne_of_gt hn_pos]
  linarith

/-- **Refutation of step `(★)` for all `n ≥ 16`**: the proposed inequality
`1 ≤ (3 - 1 / n) / (n - 1)` fails strictly for every `n ≥ 16`. -/
theorem star_step_refutation {n : ℕ} (hn : 16 ≤ n) :
    (3 - 1 / (n : ℝ)) / ((n : ℝ) - 1) < 1 := by
  have hnR : (16 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  have hden : (0 : ℝ) < (n : ℝ) - 1 := by linarith
  have hnpos : (0 : ℝ) < (n : ℝ) := by linarith
  rw [div_lt_one hden]
  have hinv : (0 : ℝ) < 1 / (n : ℝ) := div_pos one_pos hnpos
  linarith

/-- **Exact rational check at `n = 16`**: `(3 - 1/16) / (16 - 1) = 47 / 240 < 1`. -/
theorem star_step_n16_exact :
    (3 - 1 / (16 : ℚ)) / (16 - 1) = 47 / 240 ∧ (47 : ℚ) / 240 < 1 := by
  norm_num

end KVCacheDepth
