import Mathlib

/-!
# The Symmetry Group of the Binary Deletion Channel
(arXiv:2202.03633, Conjecture IV.3 and Global Group Classification)

We formalize in Lean 4 + Mathlib the complete proof of Conjecture IV.3 and the global
channel-group classification for the Binary Deletion Channel `BDC_d` (`0 < d < 1`)
from Francisco Pernice, *On the Symmetries of the Deletion Channel* (Allerton 2022,
arXiv:2202.03633):

1. **Subsequence multiplicities and channel law**:
   - `subseqCount`: Exact multiplicity `N_x(y)` of a word `y` as a subsequence of `x`.
   - `bdcProb`: Exact BDC transition probability `P(D(x) = y) = N_x(y) d^(|x|-|y|) (1-d)^|y|`.
   - `subseqCount_pos_iff_sublist`, `subseqCount_same_length`, `subseqCount_comp`,
     `subseqCount_reverse`, `subseqCount_single_bit`.
2. **Output bijection forced (`h = g⁻¹`) and subsequence-count characterization (Lemma 1)**:
   - `channelGroup_output_length_preserving`: Every output bijection `h` witnessing `g ∈ G_BDC`
     for `0 < d < 1` automatically preserves word lengths (`|h(y)| = |y|`).
   - `channelGroup_output_eq_inv`: Necessarily `h = g⁻¹` globally on `{0, 1}*`.
   - `inChannelGroup_iff_subseqCount`: `g ∈ G_BDC_d ↔ (∀ x, |g x| = |x|) ∧ (∀ x y, N_{gx}(gy) = N_x(y))`.
3. **Single-deletion moment identity and deck uniqueness for `n ≥ 3` (Lemma 2)**:
   - `subseqCount_single_deletion_eq_card`: `N_x(y)` for `|y| = |x| - 1` counts single-position
     deletions `j : Fin |x|` yielding `y`.
   - `single_deletion_moment_identity`:
     `∑_{y ∈ {0,1}^(n-1)} N_x(y) · bit(y_i) = (n - 1 - i) · bit(x_i) + (i + 1) · bit(x_{i+1})`.
   - `single_deletion_deck_unique`: For every `n ≥ 3`, the single-deletion multiplicity deck
     uniquely determines `x ∈ {0, 1}^n`.
4. **Complete classification `G_BDC ≅ V₄` and exact orbit sizes (Theorem 1 / Conjecture IV.3)**:
   - `compPerm`, `revPerm`: Bitwise complement `C` and word reversal `R` are commuting
     involutions in `G_BDC_d`.
   - `channelGroup_classification`: For every `0 < d < 1`,
     `InChannelGroup d g ↔ g = 1 ∨ g = compPerm ∨ g = revPerm ∨ g = compPerm * revPerm`.
   - `channelGroupEquivV4`: Explicit group isomorphism
     `Multiplicative (ZMod 2 × ZMod 2) ≃* ChannelGroup d hd0 hd1` (Klein four-group `V₄`).
   - `orbit_card_dichotomy`, `orbit_card_le_four`, `orbit_card_eq_two_of_len_le_two`,
     `orbit_card_attains_four`: Exact orbit cardinality is `2` if `R x ∈ {x, C x}` and `4`
     otherwise; maximum orbit size is `2` for `n ∈ {1, 2}` and `4` for all `n ≥ 3`
     (attained at `0^(n-1) 1`).

Machine-checked vs prose-only:
- Machine-checked: The entire mathematical proof of Lemma 1, Lemma 2, Theorem 1, the Klein
  four-group isomorphism `G_BDC_d ≅ V₄`, and all exact orbit-size bounds for every `0 < d < 1`
  and every blocklength `n ≥ 1`.
- Prose-only (in `solutions/bdc_orbit_symmetry/proof.pdf`): Historical remarks on subword-order
  automorphisms, endpoint `d ∈ {0, 1}` commentary, and the 8,190-word finite verification table.
-/

open BigOperators Finset

namespace BDCOrbitSymmetry

abbrev Word := List Bool

/-- Bitwise complement of a binary word. -/
def compWord (x : Word) : Word := x.map (!·)

lemma compWord_length (x : Word) : (compWord x).length = x.length := by
  simp [compWord]

lemma compWord_involutive (x : Word) : compWord (compWord x) = x := by
  simp [compWord, List.map_map, Function.comp_def]

lemma compWord_reverse (x : Word) : compWord x.reverse = (compWord x).reverse := by
  simp [compWord, List.map_reverse]

lemma compWord_ne_self {x : Word} (hx : 0 < x.length) : compWord x ≠ x := by
  cases x with
  | nil => simp at hx
  | cons a xs =>
    intro h
    simp [compWord] at h

/-- Number of increasing index selections in `x` that spell `y` (subsequence multiplicity `N_x(y)`). -/
def subseqCount : Word → Word → ℕ
  | _, [] => 1
  | [], _ :: _ => 0
  | a :: xs, b :: ys => (if a = b then subseqCount xs ys else 0) + subseqCount xs (b :: ys)

@[simp] lemma subseqCount_nil_right (x : Word) : subseqCount x [] = 1 := by
  cases x <;> rfl

@[simp] lemma subseqCount_nil_left_cons (b : Bool) (ys : Word) :
    subseqCount [] (b :: ys) = 0 := rfl

lemma subseqCount_eq_zero_of_lt : ∀ (x y : Word), x.length < y.length → subseqCount x y = 0
  | [], [], h => by simp at h
  | [], _ :: _, _ => rfl
  | _ :: _, [], h => by simp at h
  | a :: xs, b :: ys, h => by
    simp only [List.length_cons] at h
    unfold subseqCount
    have h1 : subseqCount xs ys = 0 := subseqCount_eq_zero_of_lt xs ys (by omega)
    have h2 : subseqCount xs (b :: ys) = 0 :=
      subseqCount_eq_zero_of_lt xs (b :: ys) (by simp; omega)
    simp [h1, h2]

lemma subseqCount_same_length : ∀ (x y : Word), x.length = y.length →
    subseqCount x y = if x = y then 1 else 0
  | [], [], _ => by simp
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | a :: xs, b :: ys, h => by
    simp only [List.length_cons, Nat.add_right_cancel_iff] at h
    unfold subseqCount
    have h2 : subseqCount xs (b :: ys) = 0 :=
      subseqCount_eq_zero_of_lt xs (b :: ys) (by simp [h])
    rw [h2, Nat.add_zero]
    by_cases hab : a = b
    · subst hab
      rw [if_pos rfl, subseqCount_same_length xs ys h]
      simp
    · rw [if_neg hab]
      simp [hab]

lemma subseqCount_self (x : Word) : subseqCount x x = 1 := by
  rw [subseqCount_same_length x x rfl, if_pos rfl]

lemma subseqCount_pos_iff_sublist : ∀ (x y : Word), 0 < subseqCount x y ↔ List.Sublist y x
  | x, [] => by simp
  | [], b :: ys => by simp
  | a :: xs, b :: ys => by
    unfold subseqCount
    constructor
    · intro h
      by_cases hab : a = b
      · subst hab
        simp only [↓reduceIte] at h
        by_cases h1 : 0 < subseqCount xs ys
        · exact List.Sublist.cons₂ a ((subseqCount_pos_iff_sublist xs ys).mp h1)
        · have h2 : 0 < subseqCount xs (a :: ys) := by omega
          exact List.Sublist.cons a ((subseqCount_pos_iff_sublist xs (a :: ys)).mp h2)
      · simp only [hab, ↓reduceIte, zero_add] at h
        exact List.Sublist.cons a ((subseqCount_pos_iff_sublist xs (b :: ys)).mp h)
    · intro h
      cases h with
      | cons _ hsub =>
        have hpos := (subseqCount_pos_iff_sublist xs (b :: ys)).mpr hsub
        omega
      | cons₂ _ hsub =>
        have hpos := (subseqCount_pos_iff_sublist xs ys).mpr hsub
        simp only [↓reduceIte]
        omega

lemma subseqCount_comp : ∀ (x y : Word), subseqCount (compWord x) (compWord y) = subseqCount x y
  | x, [] => by simp [compWord]
  | [], b :: ys => by simp [compWord]
  | a :: xs, b :: ys => by
    simp only [compWord, List.map_cons, subseqCount]
    have h1 := subseqCount_comp xs ys
    have h2 := subseqCount_comp xs (b :: ys)
    simp only [compWord, List.map_cons] at h1 h2
    rw [h1, h2]
    have hbeq : ((!a) = (!b)) ↔ (a = b) := by cases a <;> cases b <;> decide
    simp [hbeq]

lemma subseqCount_append_singleton : ∀ (x y : Word) (a b : Bool),
    subseqCount (x ++ [a]) (y ++ [b]) =
      (if a = b then subseqCount x y else 0) + subseqCount x (y ++ [b])
  | [], [], a, b => by
    simp [subseqCount]
  | [], c :: ys, a, b => by
    have hlt : ([] : Word).length < (ys ++ [b]).length := by simp
    have hz : subseqCount [] (ys ++ [b]) = 0 := subseqCount_eq_zero_of_lt _ _ hlt
    simp [subseqCount, hz]
  | c :: xs, [], a, b => by
    simp only [List.cons_append, List.nil_append, subseqCount, subseqCount_nil_right]
    have ih := subseqCount_append_singleton xs [] a b
    simp only [List.nil_append, subseqCount_nil_right] at ih
    rw [ih]
    by_cases hcb : c = b <;> by_cases hab : a = b <;> simp [hcb, hab]
  | c :: xs, d :: ys, a, b => by
    simp only [List.cons_append, subseqCount]
    have ih1 := subseqCount_append_singleton xs ys a b
    have ih2 := subseqCount_append_singleton xs (d :: ys) a b
    simp only [List.cons_append] at ih2
    rw [ih1, ih2]
    by_cases hcd : c = d <;> by_cases hab : a = b <;> simp only [hcd, hab, ↓reduceIte] <;> omega

lemma subseqCount_reverse : ∀ (x y : Word), subseqCount x.reverse y.reverse = subseqCount x y
  | x, [] => by simp
  | [], b :: ys => by
    simp only [List.reverse_nil, List.reverse_cons]
    have hlt : ([] : Word).length < (ys.reverse ++ [b]).length := by simp
    rw [subseqCount_eq_zero_of_lt _ _ hlt, subseqCount_nil_left_cons]
  | a :: xs, b :: ys => by
    simp only [List.reverse_cons, subseqCount_append_singleton, subseqCount]
    have h1 := subseqCount_reverse xs ys
    have h2 := subseqCount_reverse xs (b :: ys)
    simp only [List.reverse_cons] at h2
    rw [h1, h2]

lemma subseqCount_single_bit : ∀ (x : Word) (b : Bool), subseqCount x [b] = x.count b
  | [], b => by simp [subseqCount]
  | a :: xs, b => by
    simp only [subseqCount, subseqCount_nil_right, subseqCount_single_bit xs b, List.count_cons]
    cases a <;> cases b <;> simp <;> omega

/-! ### 1. Exact BDC Channel Transition Probability and Channel Group -/

/-- Exact transition probability `Pr(BDC_d(x) = y)` for deletion probability `d`. -/
def bdcProb (d : ℝ) (x y : Word) : ℝ :=
  if y.length ≤ x.length then
    (subseqCount x y : ℝ) * d ^ (x.length - y.length) * (1 - d) ^ y.length
  else 0

lemma bdcProb_pos_iff {d : ℝ} (hd0 : 0 < d) (hd1 : d < 1) (x y : Word) :
    0 < bdcProb d x y ↔ List.Sublist y x := by
  unfold bdcProb
  have h1d : 0 < 1 - d := sub_pos.mpr hd1
  constructor
  · intro h
    by_cases hlen : y.length ≤ x.length
    · rw [if_pos hlen] at h
      have hpos : (0 : ℝ) < (subseqCount x y : ℝ) := by
        by_contra hle
        push_neg at hle
        have hzero : (subseqCount x y : ℝ) = 0 := le_antisymm hle (Nat.cast_nonneg _)
        rw [hzero, zero_mul, zero_mul] at h
        exact lt_irrefl 0 h
      exact (subseqCount_pos_iff_sublist x y).mp (Nat.cast_pos.mp hpos)
    · rw [if_neg hlen] at h
      exact (lt_irrefl 0 h).elim
  · intro hsub
    have hlen : y.length ≤ x.length := hsub.length_le
    rw [if_pos hlen]
    have hcnt : (0 : ℝ) < (subseqCount x y : ℝ) :=
      Nat.cast_pos.mpr ((subseqCount_pos_iff_sublist x y).mpr hsub)
    positivity

/-- Membership in Pernice's global channel group `G_{BDC_d}` (`defn:channel-group`):
`g ∈ Aut({0,1}*)` is length-preserving and admits some `h ∈ Aut({0,1}*)` with
`h(D(g x)) =^D D(x)` for all `x ∈ {0,1}*`, i.e., `Pr(D(g x) = z) = Pr(D(x) = h(z))`. -/
def InChannelGroup (d : ℝ) (g : Equiv.Perm Word) : Prop :=
  (∀ x, (g x).length = x.length) ∧
  ∃ h : Equiv.Perm Word, ∀ x z, bdcProb d (g x) z = bdcProb d x (h z)

/-! ### 2. Lemma 1: Output Bijection Forced (`h = g⁻¹`) and Subsequence-Count Criterion -/

lemma exists_superword_of_length_le {y : Word} {n : ℕ} (h : y.length ≤ n) :
    ∃ x : Word, x.length = n ∧ List.Sublist y x := by
  refine ⟨y ++ List.replicate (n - y.length) false, ?_, ?_⟩
  · simp; omega
  · exact List.sublist_append_left y _

/-- **Lemma 1(a)**: For `0 < d < 1`, any output bijection `h` witnessing `g ∈ G_{BDC_d}`
automatically preserves word lengths. -/
theorem channelGroup_output_length_preserving {d : ℝ} (hd0 : 0 < d) (hd1 : d < 1)
    {g h : Equiv.Perm Word}
    (hg : ∀ x, (g x).length = x.length)
    (hchan : ∀ x z, bdcProb d (g x) z = bdcProb d x (h z)) :
    ∀ z, (h z).length = z.length := by
  have hle : ∀ n z, (h z).length ≤ n ↔ z.length ≤ n := by
    intro n z
    have step1 : (h z).length ≤ n ↔ ∃ x : Word, x.length = n ∧ 0 < bdcProb d x (h z) := by
      constructor
      · intro hlen
        obtain ⟨x, hxlen, hsub⟩ := exists_superword_of_length_le hlen
        exact ⟨x, hxlen, (bdcProb_pos_iff hd0 hd1 x (h z)).mpr hsub⟩
      · rintro ⟨x, hxlen, hpos⟩
        have hsub := (bdcProb_pos_iff hd0 hd1 x (h z)).mp hpos
        have := hsub.length_le
        omega
    have step2 : z.length ≤ n ↔ ∃ x : Word, x.length = n ∧ 0 < bdcProb d x z := by
      constructor
      · intro hlen
        obtain ⟨x, hxlen, hsub⟩ := exists_superword_of_length_le hlen
        exact ⟨x, hxlen, (bdcProb_pos_iff hd0 hd1 x z).mpr hsub⟩
      · rintro ⟨x, hxlen, hpos⟩
        have hsub := (bdcProb_pos_iff hd0 hd1 x z).mp hpos
        have := hsub.length_le
        omega
    rw [step1, step2]
    constructor
    · rintro ⟨x, hxlen, hpos⟩
      refine ⟨g x, by rw [hg, hxlen], ?_⟩
      rw [hchan x z]; exact hpos
    · rintro ⟨x', hx'len, hpos⟩
      refine ⟨g.symm x', ?_, ?_⟩
      · have hgx := hg (g.symm x')
        simp only [Equiv.apply_symm_apply] at hgx
        omega
      · rw [← hchan (g.symm x') z, Equiv.apply_symm_apply]
        exact hpos
  intro z
  exact le_antisymm ((hle z.length z).mpr le_rfl) ((hle (h z).length z).mp le_rfl)

/-- **Lemma 1(b)**: For `0 < d < 1`, the output bijection `h` is forced to equal `g⁻¹`. -/
theorem channelGroup_output_eq_inv {d : ℝ} (hd0 : 0 < d) (hd1 : d < 1)
    {g h : Equiv.Perm Word}
    (hg : ∀ x, (g x).length = x.length)
    (hchan : ∀ x z, bdcProb d (g x) z = bdcProb d x (h z)) :
    h = g⁻¹ := by
  have hhlen := channelGroup_output_length_preserving hd0 hd1 hg hchan
  refine Equiv.ext fun z => ?_
  have h1 : 0 < bdcProb d (g (g.symm z)) z := by
    rw [Equiv.apply_symm_apply]
    exact (bdcProb_pos_iff hd0 hd1 z z).mpr (List.Sublist.refl z)
  rw [hchan] at h1
  have hsub : List.Sublist (h z) (g.symm z) := (bdcProb_pos_iff hd0 hd1 (g.symm z) (h z)).mp h1
  have hlen : (h z).length = (g.symm z).length := by
    rw [hhlen z]
    have := hg (g.symm z)
    simp only [Equiv.apply_symm_apply] at this
    exact this
  exact hsub.eq_of_length hlen

/-- **Lemma 1(c)**: For every fixed `0 < d < 1`, `g ∈ G_{BDC_d}` if and only if `g` is
length-preserving and preserves all subsequence multiplicities `N_{gx}(gy) = N_x(y)`. -/
theorem inChannelGroup_iff_subseqCount {d : ℝ} (hd0 : 0 < d) (hd1 : d < 1)
    (g : Equiv.Perm Word) :
    InChannelGroup d g ↔
      (∀ x, (g x).length = x.length) ∧ (∀ x y, subseqCount (g x) (g y) = subseqCount x y) := by
  have h1d : 0 < 1 - d := sub_pos.mpr hd1
  constructor
  · rintro ⟨hg, h, hchan⟩
    have hh : h = g⁻¹ := channelGroup_output_eq_inv hd0 hd1 hg hchan
    refine ⟨hg, fun x y => ?_⟩
    by_cases hlen : y.length ≤ x.length
    · have hprob := hchan x (g y)
      rw [hh] at hprob
      change bdcProb d (g x) (g y) = bdcProb d x (g.symm (g y)) at hprob
      rw [Equiv.symm_apply_apply] at hprob
      unfold bdcProb at hprob
      rw [hg x, hg y, if_pos hlen, if_pos hlen] at hprob
      have hfac : 0 < d ^ (x.length - y.length) * (1 - d) ^ y.length := by positivity
      have hmul : (subseqCount (g x) (g y) : ℝ) * (d ^ (x.length - y.length) * (1 - d) ^ y.length) =
          (subseqCount x y : ℝ) * (d ^ (x.length - y.length) * (1 - d) ^ y.length) := by
        linarith
      exact Nat.cast_injective (mul_right_cancel₀ (ne_of_gt hfac) hmul)
    · have hlt : x.length < y.length := Nat.lt_of_not_ge hlen
      rw [subseqCount_eq_zero_of_lt _ _ (by rw [hg, hg]; exact hlt),
          subseqCount_eq_zero_of_lt _ _ hlt]
  · rintro ⟨hg, hsub⟩
    refine ⟨hg, g⁻¹, fun x z => ?_⟩
    change bdcProb d (g x) z = bdcProb d x (g.symm z)
    have hgz : g (g.symm z) = z := Equiv.apply_symm_apply g z
    have hlenz : (g.symm z).length = z.length := by
      have := hg (g.symm z)
      rw [hgz] at this
      exact this.symm
    unfold bdcProb
    rw [hg x, hlenz]
    by_cases hlen : z.length ≤ x.length
    · rw [if_pos hlen, if_pos hlen]
      have hc := hsub x (g.symm z)
      rw [hgz] at hc
      rw [hc]
    · rw [if_neg hlen, if_neg hlen]

/-! ### 3. Lemma 2: Single-Deletion Moment Identity and Deck Uniqueness for `n ≥ 3` -/

lemma subseqCount_single_deletion_eq_card : ∀ (x y : Word), y.length + 1 = x.length →
    subseqCount x y =
      (Finset.univ.filter (fun j : Fin x.length => x.eraseIdx j = y)).card
  | [], y, h => by simp at h
  | [a], [], _ => by
    simp [subseqCount]
  | [a], _ :: _, h => by simp at h
  | a :: x1 :: xs, [], h => by simp at h
  | a :: x1 :: xs, b :: ys, h => by
    simp only [List.length_cons, Nat.add_right_cancel_iff] at h
    unfold subseqCount
    have hsame : subseqCount (x1 :: xs) (b :: ys) = if (x1 :: xs) = (b :: ys) then 1 else 0 :=
      subseqCount_same_length (x1 :: xs) (b :: ys) (by simp [h])
    have ih := subseqCount_single_deletion_eq_card (x1 :: xs) ys (by simp [h])
    rw [Finset.card_filter]
    have hsplit : ∑ i : Fin (a :: x1 :: xs).length, (if (a :: x1 :: xs).eraseIdx ↑i = b :: ys then 1 else 0) =
        (if (x1 :: xs) = b :: ys then 1 else 0) +
        ∑ j : Fin (x1 :: xs).length, (if a :: (x1 :: xs).eraseIdx ↑j = b :: ys then 1 else 0) := by
      have := Fin.sum_univ_succ (n := (x1 :: xs).length)
        (fun i : Fin ((x1 :: xs).length + 1) => if (a :: x1 :: xs).eraseIdx ↑i = b :: ys then 1 else 0)
      simpa only [List.length_cons, Fin.val_zero, List.eraseIdx_zero, List.tail_cons,
        Fin.val_succ, List.eraseIdx_cons_succ] using this
    rw [hsplit, hsame]
    by_cases hab : a = b
    · subst hab
      rw [if_pos rfl, ih, Finset.card_filter]
      have hsum_eq : ∑ x : Fin (x1 :: xs).length, (if a :: (x1 :: xs).eraseIdx ↑x = a :: ys then 1 else 0) =
          ∑ x : Fin (x1 :: xs).length, (if (x1 :: xs).eraseIdx ↑x = ys then 1 else 0) := by
        refine Finset.sum_congr rfl fun j _ => by simp
      rw [hsum_eq]
      omega
    · rw [if_neg hab, zero_add]
      have hsum_zero : ∑ x : Fin (x1 :: xs).length, (if a :: (x1 :: xs).eraseIdx ↑x = b :: ys then 1 else 0) = 0 := by
        refine Finset.sum_eq_zero fun j _ => by simp [hab]
      rw [hsum_zero, Nat.add_zero]

/-- Bit indicator `bitVal b ∈ {0, 1}`. -/
def bitVal (b : Bool) : ℕ := if b then 1 else 0

/-- Position-`i` bit indicator of a word `y`. -/
def bitAt (y : Word) (i : ℕ) : ℕ :=
  match y[i]? with
  | some true => 1
  | _ => 0

lemma bitAt_eq_bitVal (x : Word) (i : ℕ) (hi : i < x.length) :
    bitAt x i = bitVal x[i] := by
  unfold bitAt bitVal
  rw [List.getElem?_eq_getElem hi]
  cases x[i] <;> rfl

lemma getElem?_eraseIdx_of_lt (x : Word) (j i : ℕ) (hij : i < j) :
    (x.eraseIdx j)[i]? = x[i]? := by
  induction x generalizing j i with
  | nil => simp
  | cons a xs ih =>
    cases j with
    | zero => omega
    | succ j' =>
      cases i with
      | zero => simp
      | succ i' =>
        simp only [List.eraseIdx_cons_succ, List.getElem?_cons_succ]
        exact ih j' i' (by omega)

lemma getElem?_eraseIdx_of_le (x : Word) (j i : ℕ) (hji : j ≤ i) :
    (x.eraseIdx j)[i]? = x[i + 1]? := by
  induction x generalizing j i with
  | nil => simp
  | cons a xs ih =>
    cases j with
    | zero => simp
    | succ j' =>
      cases i with
      | zero => omega
      | succ i' =>
        simp only [List.eraseIdx_cons_succ, List.getElem?_cons_succ]
        exact ih j' i' (by omega)

/-- The finite set of all binary words of length `m`, represented via `Fin m → Bool`. -/
def wordsOfLen (m : ℕ) : Finset Word :=
  Finset.univ.image (fun f : Fin m → Bool => List.ofFn f)

lemma mem_wordsOfLen {m : ℕ} {y : Word} : y ∈ wordsOfLen m ↔ y.length = m := by
  unfold wordsOfLen
  simp only [Finset.mem_image, Finset.mem_univ, true_and]
  constructor
  · rintro ⟨f, rfl⟩
    exact List.length_ofFn
  · intro hlen
    refine ⟨fun i => y[i.1]'(by omega), ?_⟩
    apply List.ext_getElem
    · simp [hlen]
    · intro i hi1 hi2
      simp

/-- **Lemma 2(a) (Single-deletion moment identity)**: For every word `x` of length `n` and
every `i < n - 1`,
`∑_{y ∈ {0,1}^(n-1)} N_x(y) · bit(y_i) = (n - 1 - i) · bit(x_i) + (i + 1) · bit(x_{i+1})`. -/
theorem single_deletion_moment_identity (x : Word) (i : ℕ) (hi : i + 1 < x.length) :
    ∑ y ∈ wordsOfLen (x.length - 1), subseqCount x y * bitAt y i =
      (x.length - 1 - i) * bitAt x i + (i + 1) * bitAt x (i + 1) := by
  have h1 : ∑ y ∈ wordsOfLen (x.length - 1), subseqCount x y * bitAt y i =
      ∑ j : Fin x.length, bitAt (x.eraseIdx j) i := by
    calc ∑ y ∈ wordsOfLen (x.length - 1), subseqCount x y * bitAt y i
        = ∑ y ∈ wordsOfLen (x.length - 1),
            (Finset.univ.filter (fun j : Fin x.length => x.eraseIdx j = y)).card * bitAt y i := by
          refine Finset.sum_congr rfl fun y hy => ?_
          have hylen : y.length = x.length - 1 := mem_wordsOfLen.mp hy
          rw [subseqCount_single_deletion_eq_card x y (by omega)]
      _ = ∑ y ∈ wordsOfLen (x.length - 1),
            ∑ j ∈ Finset.univ.filter (fun j : Fin x.length => x.eraseIdx j = y), bitAt (x.eraseIdx j) i := by
          refine Finset.sum_congr rfl fun y _ => ?_
          rw [Finset.sum_congr rfl (fun j hj => by
            have := (Finset.mem_filter.mp hj).2
            rw [this])]
          simp [Finset.sum_const, smul_eq_mul]
      _ = ∑ j : Fin x.length, bitAt (x.eraseIdx j) i := by
          rw [← Finset.sum_biUnion]
          · congr 1
            ext j
            simp only [Finset.mem_biUnion, Finset.mem_filter, Finset.mem_univ, true_and, iff_true]
            refine ⟨x.eraseIdx j, ?_, rfl⟩
            rw [mem_wordsOfLen, List.length_eraseIdx_of_lt j.2]
          · intro y1 _ y2 _ hne
            change Disjoint _ _
            rw [Finset.disjoint_left]
            intro j hj1 hj2
            exact hne ((Finset.mem_filter.mp hj1).2.symm.trans (Finset.mem_filter.mp hj2).2)
  rw [h1]
  have hsplit : ∑ j : Fin x.length, bitAt (x.eraseIdx j) i =
      (∑ j ∈ Finset.univ.filter (fun j : Fin x.length => i < j.1), bitAt (x.eraseIdx j) i) +
      (∑ j ∈ Finset.univ.filter (fun j : Fin x.length => ¬ i < j.1), bitAt (x.eraseIdx j) i) :=
    (Finset.sum_filter_add_sum_filter_not _ _ _).symm
  rw [hsplit]
  have hpart1 : ∑ j ∈ Finset.univ.filter (fun j : Fin x.length => i < j.1), bitAt (x.eraseIdx j) i =
      (x.length - 1 - i) * bitAt x i := by
    have hcongr : ∀ j ∈ Finset.univ.filter (fun j : Fin x.length => i < j.1),
        bitAt (x.eraseIdx j) i = bitAt x i := by
      intro j hj
      unfold bitAt
      rw [getElem?_eraseIdx_of_lt x j.1 i (Finset.mem_filter.mp hj).2]
    rw [Finset.sum_congr rfl hcongr, Finset.sum_const, smul_eq_mul]
    congr 1
    have hcard : (Finset.univ.filter (fun j : Fin x.length => i < j.1)).card =
        (Finset.Ioo i x.length).card := by
      refine Finset.card_bij (fun j _ => j.1) ?_ ?_ ?_
      · intro j hj
        rw [Finset.mem_Ioo]
        exact ⟨(Finset.mem_filter.mp hj).2, j.2⟩
      · intro j1 _ j2 _ h; exact Fin.ext h
      · intro m hm
        rw [Finset.mem_Ioo] at hm
        exact ⟨⟨m, hm.2⟩, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hm.1⟩, rfl⟩
    rw [hcard, Nat.card_Ioo]
    omega
  have hpart2 : ∑ j ∈ Finset.univ.filter (fun j : Fin x.length => ¬ i < j.1), bitAt (x.eraseIdx j) i =
      (i + 1) * bitAt x (i + 1) := by
    have hcongr : ∀ j ∈ Finset.univ.filter (fun j : Fin x.length => ¬ i < j.1),
        bitAt (x.eraseIdx j) i = bitAt x (i + 1) := by
      intro j hj
      unfold bitAt
      have hle : j.1 ≤ i := Nat.le_of_not_lt (Finset.mem_filter.mp hj).2
      rw [getElem?_eraseIdx_of_le x j.1 i hle]
    rw [Finset.sum_congr rfl hcongr, Finset.sum_const, smul_eq_mul]
    congr 1
    have hcard : (Finset.univ.filter (fun j : Fin x.length => ¬ i < j.1)).card =
        (Finset.range (i + 1)).card := by
      refine Finset.card_bij (fun j _ => j.1) ?_ ?_ ?_
      · intro j hj
        rw [Finset.mem_range]
        have hjlt := (Finset.mem_filter.mp hj).2
        dsimp only
        omega
      · intro j1 _ j2 _ h; exact Fin.ext h
      · intro m hm
        rw [Finset.mem_range] at hm
        refine ⟨⟨m, by omega⟩, Finset.mem_filter.mpr ⟨Finset.mem_univ _, ?_⟩, rfl⟩
        dsimp only
        omega
    rw [hcard, Finset.card_range]
  rw [hpart1, hpart2]

/-- **Lemma 2(b) (Single-deletion deck uniqueness for `n ≥ 3`)**: For any two binary words
`x, x'` of the same length `n ≥ 3`, if `N_x(y) = N_{x'}(y)` for every word `y` of length `n - 1`,
then `x = x'`. -/
theorem single_deletion_deck_unique {x x' : Word}
    (hlen : x.length = x'.length) (hn : 3 ≤ x.length)
    (hdeck : ∀ y : Word, y.length = x.length - 1 → subseqCount x y = subseqCount x' y) :
    x = x' := by
  have hmom : ∀ i, i + 1 < x.length →
      (x.length - 1 - i) * bitAt x i + (i + 1) * bitAt x (i + 1) =
      (x.length - 1 - i) * bitAt x' i + (i + 1) * bitAt x' (i + 1) := by
    intro i hi
    have h1 := single_deletion_moment_identity x i hi
    have h2 := single_deletion_moment_identity x' i (by omega)
    rw [← hlen] at h2
    rw [← h1, ← h2]
    refine Finset.sum_congr rfl fun y hy => ?_
    rw [hdeck y (mem_wordsOfLen.mp hy)]
  have hbit_le_one : ∀ (w : Word) (i : ℕ), bitAt w i ≤ 1 := by
    intro w i
    unfold bitAt
    cases h : w[i]? with
    | none => simp
    | some b => cases b <;> simp
  have hbit_eq : ∀ i, i < x.length → bitAt x i = bitAt x' i := by
    intro i
    induction i with
    | zero =>
      intro _
      have h0 := hmom 0 (by omega)
      simp only [Nat.sub_zero, Nat.zero_add, one_mul] at h0
      have bx0 := hbit_le_one x 0
      have bx1 := hbit_le_one x 1
      have by0 := hbit_le_one x' 0
      have by1 := hbit_le_one x' 1
      rcases Nat.le_one_iff_eq_zero_or_eq_one.mp bx0 with h1 | h1 <;>
        rcases Nat.le_one_iff_eq_zero_or_eq_one.mp by0 with h2 | h2
      · rw [h1, h2]
      · rw [h1, h2] at h0; omega
      · rw [h1, h2] at h0; omega
      · rw [h1, h2]
    | succ k ih =>
      intro hk
      have hprev := ih (by omega)
      have hstep := hmom k (by omega)
      rw [hprev] at hstep
      have hmul : (k + 1) * bitAt x (k + 1) = (k + 1) * bitAt x' (k + 1) := by omega
      exact Nat.eq_of_mul_eq_mul_left (Nat.succ_pos k) hmul
  apply List.ext_getElem hlen
  intro i hi1 hi2
  have hbi := hbit_eq i hi1
  rw [bitAt_eq_bitVal x i hi1, bitAt_eq_bitVal x' i hi2] at hbi
  unfold bitVal at hbi
  cases hx : x[i] <;> cases hx' : x'[i] <;> simp_all

/-! ### 4. Theorem 1: Classification `G_BDC_d ≅ V₄` and Exact Orbit Bounds -/

/-- Bitwise complement permutation `C ∈ Aut({0,1}*)`. -/
def compPerm : Equiv.Perm Word where
  toFun := compWord
  invFun := compWord
  left_inv := compWord_involutive
  right_inv := compWord_involutive

/-- Word reversal permutation `R ∈ Aut({0,1}*)`. -/
def revPerm : Equiv.Perm Word where
  toFun := List.reverse
  invFun := List.reverse
  left_inv := List.reverse_reverse
  right_inv := List.reverse_reverse

lemma compPerm_sq : compPerm * compPerm = 1 :=
  Equiv.ext fun x => compWord_involutive x

lemma revPerm_sq : revPerm * revPerm = 1 :=
  Equiv.ext fun x => List.reverse_reverse x

lemma compPerm_comm_revPerm : compPerm * revPerm = revPerm * compPerm :=
  Equiv.ext fun x => compWord_reverse x

lemma compPerm_inChannelGroup {d : ℝ} (hd0 : 0 < d) (hd1 : d < 1) :
    InChannelGroup d compPerm := by
  rw [inChannelGroup_iff_subseqCount hd0 hd1]
  exact ⟨compWord_length, subseqCount_comp⟩

lemma revPerm_inChannelGroup {d : ℝ} (hd0 : 0 < d) (hd1 : d < 1) :
    InChannelGroup d revPerm := by
  rw [inChannelGroup_iff_subseqCount hd0 hd1]
  exact ⟨fun x => List.length_reverse, subseqCount_reverse⟩

lemma inChannelGroup_one {d : ℝ} (hd0 : 0 < d) (hd1 : d < 1) :
    InChannelGroup d 1 := by
  rw [inChannelGroup_iff_subseqCount hd0 hd1]
  exact ⟨fun _ => rfl, fun _ _ => rfl⟩

lemma inChannelGroup_mul {d : ℝ} (hd0 : 0 < d) (hd1 : d < 1)
    {g₁ g₂ : Equiv.Perm Word} (h1 : InChannelGroup d g₁) (h2 : InChannelGroup d g₂) :
    InChannelGroup d (g₁ * g₂) := by
  rw [inChannelGroup_iff_subseqCount hd0 hd1] at h1 h2 ⊢
  refine ⟨fun x => by simp [h1.1, h2.1], fun x y => by simp [h1.2, h2.2]⟩

lemma inChannelGroup_inv {d : ℝ} (hd0 : 0 < d) (hd1 : d < 1)
    {g : Equiv.Perm Word} (hg : InChannelGroup d g) :
    InChannelGroup d g⁻¹ := by
  rw [inChannelGroup_iff_subseqCount hd0 hd1] at hg ⊢
  refine ⟨fun x => ?_, fun x y => ?_⟩
  · have := hg.1 (g⁻¹ x); simpa using this.symm
  · have := hg.2 (g⁻¹ x) (g⁻¹ y); simpa using this.symm

/-- Subgroup of `Equiv.Perm Word` given by `G_{BDC_d}` for `0 < d < 1`. -/
def ChannelGroup (d : ℝ) (hd0 : 0 < d) (hd1 : d < 1) : Subgroup (Equiv.Perm Word) where
  carrier := {g | InChannelGroup d g}
  one_mem' := inChannelGroup_one hd0 hd1
  mul_mem' := inChannelGroup_mul hd0 hd1
  inv_mem' := inChannelGroup_inv hd0 hd1

/-- Any length- and subsequence-count-preserving permutation fixing `A^1` and `A^2` is the identity. -/
lemma eq_one_of_fixes_len_le_two {f : Equiv.Perm Word}
    (hlen : ∀ x, (f x).length = x.length)
    (hsub : ∀ x y, subseqCount (f x) (f y) = subseqCount x y)
    (h1 : ∀ x : Word, x.length = 1 → f x = x)
    (h2 : ∀ x : Word, x.length = 2 → f x = x) :
    f = 1 := by
  have hall : ∀ n : ℕ, ∀ x : Word, x.length = n → f x = x := by
    intro n
    induction n with
    | zero =>
      intro x hx
      have hx0 : x = [] := List.eq_nil_of_length_eq_zero hx
      have hfx0 : f x = [] := List.eq_nil_of_length_eq_zero (by rw [hlen, hx])
      rw [hfx0, hx0]
    | succ m ih =>
      intro x hx
      by_cases hm : m = 0
      · exact h1 x (by omega)
      · by_cases hm1 : m = 1
        · exact h2 x (by omega)
        · have hn3 : 3 ≤ x.length := by omega
          apply single_deletion_deck_unique (hlen x) (by rw [hlen]; exact hn3)
          intro y hy
          have hylen : y.length = m := by rw [hlen, hx] at hy; omega
          have hfy : f y = y := ih y hylen
          calc subseqCount (f x) y
              = subseqCount (f x) (f y) := by rw [hfy]
            _ = subseqCount x y := hsub x y
  exact Equiv.ext fun x => hall x.length x rfl

/-- **Theorem 1(a) (Complete classification of `G_{BDC_d}`)**: For every `0 < d < 1`,
`InChannelGroup d g ↔ g ∈ {1, C, R, C ∘ R}`. -/
theorem channelGroup_classification {d : ℝ} (hd0 : 0 < d) (hd1 : d < 1)
    (g : Equiv.Perm Word) :
    InChannelGroup d g ↔
      g = 1 ∨ g = compPerm ∨ g = revPerm ∨ g = compPerm * revPerm := by
  constructor
  · intro hg
    have hlen_g : ∀ x, (g x).length = x.length := hg.1
    have hgt_len : (g [true]).length = 1 := hlen_g [true]
    have hgt_cases : g [true] = [true] ∨ g [true] = [false] := by
      cases hgx : g [true] with
      | nil => simp [hgx] at hgt_len
      | cons b rest =>
        simp [hgx] at hgt_len
        subst hgt_len
        cases b <;> simp
    have make_fix1 : ∃ c : Equiv.Perm Word, (c = 1 ∨ c = compPerm) ∧
        InChannelGroup d (c * g) ∧ (∀ x : Word, x.length = 1 → (c * g) x = x) := by
      rcases hgt_cases with ht | ht
      · refine ⟨1, Or.inl rfl, by simpa using hg, fun x hx => ?_⟩
        cases x with
        | nil => simp at hx
        | cons b rest =>
          simp at hx; subst hx
          cases b
          · have hfl : (g [false]).length = 1 := hlen_g [false]
            cases hgf : g [false] with
            | nil => simp [hgf] at hfl
            | cons b' r' =>
              simp [hgf] at hfl; subst hfl
              cases b'
              · exact hgf
              · exfalso
                have hinj := g.injective (hgf.trans ht.symm)
                simp at hinj
          · simpa using ht
      · refine ⟨compPerm, Or.inr rfl, inChannelGroup_mul hd0 hd1 (compPerm_inChannelGroup hd0 hd1) hg,
          fun x hx => ?_⟩
        cases x with
        | nil => simp at hx
        | cons b rest =>
          simp at hx; subst hx
          have hct : (compPerm * g) [true] = [true] := by
            change compWord (g [true]) = [true]
            rw [ht]; rfl
          cases b
          · have hfl : ((compPerm * g) [false]).length = 1 := by
              simp [compPerm, compWord_length, hlen_g]
            cases hgf : (compPerm * g) [false] with
            | nil => simp [hgf] at hfl
            | cons b' r' =>
              simp [hgf] at hfl; subst hfl
              cases b'
              · rfl
              · exfalso
                have hinj := (compPerm * g).injective (hgf.trans hct.symm)
                simp at hinj
          · exact hct
    obtain ⟨c, hc, hcg, hfix1⟩ := make_fix1
    set f₁ := c * g
    have hf₁_spec := (inChannelGroup_iff_subseqCount hd0 hd1 f₁).mp hcg
    have hweight : ∀ x : Word, (f₁ x).count true = x.count true := by
      intro x
      rw [← subseqCount_single_bit, ← subseqCount_single_bit]
      have := hf₁_spec.2 x [true]
      rw [hfix1 [true] rfl] at this
      exact this
    have hfix00 : f₁ [false, false] = [false, false] := by
      have hl : (f₁ [false, false]).length = 2 := hf₁_spec.1 [false, false]
      have hw : (f₁ [false, false]).count true = 0 := hweight [false, false]
      cases h : f₁ [false, false] with
      | nil => simp [h] at hl
      | cons a l1 =>
        cases l1 with
        | nil => simp [h] at hl
        | cons b l2 =>
          cases l2 with
          | nil =>
            simp [h] at hw
            cases a <;> cases b <;> simp at hw ⊢
          | cons _ _ => simp [h] at hl
    have hfix11 : f₁ [true, true] = [true, true] := by
      have hl : (f₁ [true, true]).length = 2 := hf₁_spec.1 [true, true]
      have hw : (f₁ [true, true]).count true = 2 := hweight [true, true]
      cases h : f₁ [true, true] with
      | nil => simp [h] at hl
      | cons a l1 =>
        cases l1 with
        | nil => simp [h] at hl
        | cons b l2 =>
          cases l2 with
          | nil =>
            simp [h] at hw
            cases a <;> cases b <;> simp at hw ⊢
          | cons _ _ => simp [h] at hl
    have h01_cases : f₁ [false, true] = [false, true] ∨ f₁ [false, true] = [true, false] := by
      have hl : (f₁ [false, true]).length = 2 := hf₁_spec.1 [false, true]
      have hw : (f₁ [false, true]).count true = 1 := hweight [false, true]
      cases h : f₁ [false, true] with
      | nil => simp [h] at hl
      | cons a l1 =>
        cases l1 with
        | nil => simp [h] at hl
        | cons b l2 =>
          cases l2 with
          | nil =>
            simp [h] at hw
            cases a <;> cases b <;> simp at hw ⊢
          | cons _ _ => simp [h] at hl
    have make_fix2 : ∃ r : Equiv.Perm Word, (r = 1 ∨ r = revPerm) ∧ r * f₁ = 1 := by
      rcases h01_cases with h01 | h01
      · refine ⟨1, Or.inl rfl, ?_⟩
        rw [one_mul]
        apply eq_one_of_fixes_len_le_two hf₁_spec.1 hf₁_spec.2 hfix1
        intro x hx
        cases x with
        | nil => simp at hx
        | cons a l1 =>
          cases l1 with
          | nil => simp at hx
          | cons b l2 =>
            cases l2 with
            | nil =>
              cases a <;> cases b
              · exact hfix00
              · exact h01
              · have hl : (f₁ [true, false]).length = 2 := hf₁_spec.1 [true, false]
                have hw : (f₁ [true, false]).count true = 1 := hweight [true, false]
                cases h : f₁ [true, false] with
                | nil => simp [h] at hl
                | cons a' l1' =>
                  cases l1' with
                  | nil => simp [h] at hl
                  | cons b' l2' =>
                    cases l2' with
                    | nil =>
                      simp [h] at hw
                      cases a' <;> cases b'
                      · simp at hw
                      · exfalso
                        have hinj := f₁.injective (h.trans h01.symm)
                        simp at hinj
                      · rfl
                      · simp at hw
                    | cons _ _ => simp [h] at hl
              · exact hfix11
            | cons _ _ => simp at hx
      · refine ⟨revPerm, Or.inr rfl, ?_⟩
        have hf₂_in := inChannelGroup_mul hd0 hd1 (revPerm_inChannelGroup hd0 hd1) hcg
        have hf₂_spec := (inChannelGroup_iff_subseqCount hd0 hd1 (revPerm * f₁)).mp hf₂_in
        apply eq_one_of_fixes_len_le_two hf₂_spec.1 hf₂_spec.2
        · intro x hx
          change (f₁ x).reverse = x
          rw [hfix1 x hx]
          cases x with
          | nil => simp at hx
          | cons a rest => simp at hx; subst hx; rfl
        · intro x hx
          have hr01 : (revPerm * f₁) [false, true] = [false, true] := by
            change (f₁ [false, true]).reverse = [false, true]
            rw [h01]; rfl
          have hr00 : (revPerm * f₁) [false, false] = [false, false] := by
            change (f₁ [false, false]).reverse = [false, false]
            rw [hfix00]; rfl
          have hr11 : (revPerm * f₁) [true, true] = [true, true] := by
            change (f₁ [true, true]).reverse = [true, true]
            rw [hfix11]; rfl
          have hrf_w : ((revPerm * f₁) [true, false]).count true = 1 := by
            change (f₁ [true, false]).reverse.count true = 1
            rw [List.count_reverse, hweight [true, false]]
            rfl
          cases x with
          | nil => simp at hx
          | cons a l1 =>
            cases l1 with
            | nil => simp at hx
            | cons b l2 =>
              cases l2 with
              | nil =>
                cases a <;> cases b
                · exact hr00
                · exact hr01
                · have hl : ((revPerm * f₁) [true, false]).length = 2 :=
                    hf₂_spec.1 [true, false]
                  cases h : (revPerm * f₁) [true, false] with
                  | nil => simp [h] at hl
                  | cons a' l1' =>
                    cases l1' with
                    | nil => simp [h] at hl
                    | cons b' l2' =>
                      cases l2' with
                      | nil =>
                        simp [h] at hrf_w
                        cases a' <;> cases b'
                        · simp at hrf_w
                        · exfalso
                          have hinj := (revPerm * f₁).injective (h.trans hr01.symm)
                          simp at hinj
                        · rfl
                        · simp at hrf_w
                      | cons _ _ => simp [h] at hl
                · exact hr11
              | cons _ _ => simp at hx
    obtain ⟨r, hr, hrf⟩ := make_fix2
    have hg_eq : g = c * r := by
      have h1 : (c * r) * (r * (c * g)) = c * r := by rw [hrf, mul_one]
      have hr2 : r * r = 1 := by rcases hr with rfl | rfl; exact one_mul 1; exact revPerm_sq
      have hc2 : c * c = 1 := by rcases hc with rfl | rfl; exact one_mul 1; exact compPerm_sq
      calc g = 1 * g := (one_mul g).symm
        _ = (c * c) * g := by rw [hc2]
        _ = c * (1 * (c * g)) := by simp [mul_assoc]
        _ = c * ((r * r) * (c * g)) := by rw [hr2]
        _ = (c * r) * (r * (c * g)) := by simp [mul_assoc]
        _ = c * r := h1
    rcases hc with rfl | rfl <;> rcases hr with rfl | rfl
    · left; simpa using hg_eq
    · right; right; left; simpa using hg_eq
    · right; left; simpa using hg_eq
    · right; right; right; exact hg_eq
  · rintro (rfl | rfl | rfl | rfl)
    · exact inChannelGroup_one hd0 hd1
    · exact compPerm_inChannelGroup hd0 hd1
    · exact revPerm_inChannelGroup hd0 hd1
    · exact inChannelGroup_mul hd0 hd1
        (compPerm_inChannelGroup hd0 hd1) (revPerm_inChannelGroup hd0 hd1)

/-- The four symmetries `1, C, R, CR` are pairwise distinct on `{0, 1}*`. -/
lemma four_symmetries_distinct :
    (1 : Equiv.Perm Word) ≠ compPerm ∧
    (1 : Equiv.Perm Word) ≠ revPerm ∧
    (1 : Equiv.Perm Word) ≠ compPerm * revPerm ∧
    compPerm ≠ revPerm ∧
    compPerm ≠ compPerm * revPerm ∧
    revPerm ≠ compPerm * revPerm := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩ <;> intro h
  · have := congrArg (fun p => p [false]) h; simp [compPerm, compWord] at this
  · have := congrArg (fun p => p [false, true]) h; simp [revPerm] at this
  · have := congrArg (fun p => p [false, false]) h; simp [compPerm, revPerm, compWord] at this
  · have := congrArg (fun p => p [false]) h; simp [compPerm, revPerm, compWord] at this
  · have := congrArg (fun p => p [false, true]) h; simp [compPerm, revPerm, compWord] at this
  · have := congrArg (fun p => p [false, false]) h; simp [compPerm, revPerm, compWord] at this

/-- Helper mapping `ZMod 2 × ZMod 2` to `{1, C, R, CR}` in `ChannelGroup`. -/
def v4ToChannel (d : ℝ) (hd0 : 0 < d) (hd1 : d < 1) (ab : ZMod 2 × ZMod 2) :
    ChannelGroup d hd0 hd1 :=
  if ab.1 = 0 then
    if ab.2 = 0 then ⟨1, inChannelGroup_one hd0 hd1⟩
    else ⟨revPerm, revPerm_inChannelGroup hd0 hd1⟩
  else
    if ab.2 = 0 then ⟨compPerm, compPerm_inChannelGroup hd0 hd1⟩
    else ⟨compPerm * revPerm, inChannelGroup_mul hd0 hd1
      (compPerm_inChannelGroup hd0 hd1) (revPerm_inChannelGroup hd0 hd1)⟩

/-- Explicit Klein four-group isomorphism `Multiplicative (ZMod 2 × ZMod 2) ≃* ChannelGroup d`. -/
noncomputable def channelGroupEquivV4 (d : ℝ) (hd0 : 0 < d) (hd1 : d < 1) :
    Multiplicative (ZMod 2 × ZMod 2) ≃* ChannelGroup d hd0 hd1 := by
  classical
  exact
  { toFun := fun p => v4ToChannel d hd0 hd1 (Multiplicative.toAdd p)
    invFun := fun g =>
      if (g : Equiv.Perm Word) = 1 then Multiplicative.ofAdd (0, 0)
      else if (g : Equiv.Perm Word) = compPerm then Multiplicative.ofAdd (1, 0)
      else if (g : Equiv.Perm Word) = revPerm then Multiplicative.ofAdd (0, 1)
      else Multiplicative.ofAdd (1, 1)
    left_inv := fun p => by
      have hdist := four_symmetries_distinct
      change (let g := v4ToChannel d hd0 hd1 (Multiplicative.toAdd p);
        if (g : Equiv.Perm Word) = 1 then Multiplicative.ofAdd (0, 0)
        else if (g : Equiv.Perm Word) = compPerm then Multiplicative.ofAdd (1, 0)
        else if (g : Equiv.Perm Word) = revPerm then Multiplicative.ofAdd (0, 1)
        else Multiplicative.ofAdd (1, 1)) = Multiplicative.ofAdd (Multiplicative.toAdd p)
      generalize Multiplicative.toAdd p = ab
      rcases ab with ⟨a, b⟩
      fin_cases a <;> fin_cases b
      · simp [v4ToChannel]
      · simp [v4ToChannel, hdist.2.1.symm, hdist.2.2.2.1.symm]
      · simp [v4ToChannel, hdist.1.symm]
      · simp [v4ToChannel, hdist.2.2.1.symm, hdist.2.2.2.2.1.symm, hdist.2.2.2.2.2.symm]
    right_inv := fun g => by
      have hcls := (channelGroup_classification hd0 hd1 g.1).mp g.2
      have hdist := four_symmetries_distinct
      apply Subtype.ext
      rcases hcls with h | h | h | h
      · simp [h, v4ToChannel]
      · simp [h, v4ToChannel, hdist.1.symm]
      · simp [h, v4ToChannel, hdist.2.1.symm, hdist.2.2.2.1.symm]
      · simp [h, v4ToChannel, hdist.2.2.1.symm, hdist.2.2.2.2.1.symm, hdist.2.2.2.2.2.symm]
    map_mul' := fun p q => by
      apply Subtype.ext
      apply Equiv.ext
      intro w
      change ((v4ToChannel d hd0 hd1 (Multiplicative.toAdd p + Multiplicative.toAdd q) : Equiv.Perm Word) w) =
        (((v4ToChannel d hd0 hd1 (Multiplicative.toAdd p) : Equiv.Perm Word) *
          (v4ToChannel d hd0 hd1 (Multiplicative.toAdd q) : Equiv.Perm Word)) w)
      generalize Multiplicative.toAdd p = ab1
      generalize Multiplicative.toAdd q = ab2
      rcases ab1 with ⟨a1, b1⟩
      rcases ab2 with ⟨a2, b2⟩
      have h20 : (1 : ZMod 2) + 1 = 0 := by decide
      have h10 : (1 : ZMod 2) ≠ 0 := by decide
      fin_cases a1 <;> fin_cases b1 <;> fin_cases a2 <;> fin_cases b2
      all_goals
        dsimp only [Prod.mk_add_mk]
        try rw [show (⟨0, by decide⟩ : ZMod 2) = 0 from rfl]
        try rw [show (⟨1, by decide⟩ : ZMod 2) = 1 from rfl]
        simp [v4ToChannel, h20, h10, compPerm, revPerm, compWord_involutive, compWord_reverse] }

/-- The `G_{BDC_d}`-orbit of a word `x`, as a finite set `{x, C x, R x, C R x}`. -/
def bdcOrbit (x : Word) : Finset Word :=
  {x, compWord x, x.reverse, compWord x.reverse}

lemma mem_bdcOrbit_iff {d : ℝ} (hd0 : 0 < d) (hd1 : d < 1) (x y : Word) :
    y ∈ bdcOrbit x ↔ ∃ g : Equiv.Perm Word, InChannelGroup d g ∧ g x = y := by
  unfold bdcOrbit
  simp only [Finset.mem_insert, Finset.mem_singleton]
  constructor
  · rintro (rfl | rfl | rfl | rfl)
    · exact ⟨1, inChannelGroup_one hd0 hd1, rfl⟩
    · exact ⟨compPerm, compPerm_inChannelGroup hd0 hd1, rfl⟩
    · exact ⟨revPerm, revPerm_inChannelGroup hd0 hd1, rfl⟩
    · exact ⟨compPerm * revPerm, inChannelGroup_mul hd0 hd1
        (compPerm_inChannelGroup hd0 hd1) (revPerm_inChannelGroup hd0 hd1), rfl⟩
  · rintro ⟨g, hg, rfl⟩
    rcases (channelGroup_classification hd0 hd1 g).mp hg with rfl | rfl | rfl | rfl
    · left; rfl
    · right; left; rfl
    · right; right; left; rfl
    · right; right; right; rfl

/-- **Theorem 1(b) (Exact orbit cardinality dichotomy)**: For every nonempty binary word `x`,
its `G_{BDC_d}`-orbit has cardinality `2` if `R x ∈ {x, C x}`, and cardinality `4` otherwise. -/
theorem orbit_card_dichotomy {x : Word} (hx : 0 < x.length) :
    (bdcOrbit x).card = if (x.reverse = x ∨ x.reverse = compWord x) then 2 else 4 := by
  have hcx : compWord x ≠ x := compWord_ne_self hx
  have hcrx : compWord x.reverse ≠ x.reverse :=
    compWord_ne_self (by rw [List.length_reverse]; exact hx)
  unfold bdcOrbit
  by_cases hrev : x.reverse = x ∨ x.reverse = compWord x
  · rw [if_pos hrev]
    rcases hrev with h1 | h2
    · rw [h1]
      have hset : ({x, compWord x, x, compWord x} : Finset Word) = {x, compWord x} := by
        ext w; simp [or_comm]
      rw [hset, Finset.card_pair hcx.symm]
    · rw [h2, compWord_involutive]
      have hset : ({x, compWord x, compWord x, x} : Finset Word) = {x, compWord x} := by
        ext w; simp [or_comm]
      rw [hset, Finset.card_pair hcx.symm]
  · rw [if_neg hrev]
    push_neg at hrev
    have h1 : x ≠ compWord x := hcx.symm
    have h2 : x ≠ x.reverse := hrev.1.symm
    have h3 : x ≠ compWord x.reverse := by
      intro heq
      apply hrev.2
      calc x.reverse = compWord (compWord x.reverse) := (compWord_involutive _).symm
        _ = compWord x := by rw [← heq]
    have h4 : compWord x ≠ x.reverse := hrev.2.symm
    have h5 : compWord x ≠ compWord x.reverse := by
      intro heq
      apply hrev.1
      calc x.reverse = compWord (compWord x.reverse) := (compWord_involutive _).symm
        _ = compWord (compWord x) := by rw [← heq]
        _ = x := compWord_involutive x
    have h6 : x.reverse ≠ compWord x.reverse := hcrx.symm
    rw [Finset.card_insert_of_notMem (by simp [h1, h2, h3]),
        Finset.card_insert_of_notMem (by simp [h4, h5]),
        Finset.card_pair h6]

/-- **Conjecture IV.3 (Universal upper bound)**: Every orbit of `G_{BDC_d}` has size at most `4`. -/
theorem orbit_card_le_four (x : Word) : (bdcOrbit x).card ≤ 4 := by
  by_cases hx : 0 < x.length
  · rw [orbit_card_dichotomy hx]
    split_ifs <;> omega
  · have hx0 : x = [] := List.eq_nil_of_length_eq_zero (by omega)
    subst hx0
    decide

/-- For blocklengths `n = 1` and `n = 2`, every orbit has exact cardinality `2`. -/
theorem orbit_card_eq_two_of_len_le_two {x : Word} (hx1 : 1 ≤ x.length) (hx2 : x.length ≤ 2) :
    (bdcOrbit x).card = 2 := by
  rw [orbit_card_dichotomy (by omega)]
  have hcond : x.reverse = x ∨ x.reverse = compWord x := by
    cases x with
    | nil => simp at hx1
    | cons a l1 =>
      cases l1 with
      | nil => left; rfl
      | cons b l2 =>
        cases l2 with
        | nil =>
          cases a <;> cases b <;> simp [compWord]
        | cons _ _ => simp at hx2
  rw [if_pos hcond]

/-- For every blocklength `n ≥ 3`, the word `0^(n-1) 1` attains orbit cardinality `4`. -/
theorem orbit_card_attains_four {n : ℕ} (hn : 3 ≤ n) :
    ∃ x : Word, x.length = n ∧ (bdcOrbit x).card = 4 := by
  refine ⟨List.replicate (n - 1) false ++ [true], by simp; omega, ?_⟩
  rw [orbit_card_dichotomy (by simp)]
  have hnot : ¬ ((List.replicate (n - 1) false ++ [true]).reverse =
      List.replicate (n - 1) false ++ [true] ∨
      (List.replicate (n - 1) false ++ [true]).reverse =
      compWord (List.replicate (n - 1) false ++ [true])) := by
    rintro (h1 | h2)
    · have hhead := congrArg List.head? h1
      cases hm : n - 1 with
      | zero => omega
      | succ k =>
        rw [hm] at hhead
        simp [List.reverse_append, List.reverse_replicate, List.replicate_succ] at hhead
    · have hcnt := congrArg (fun w => w.count true) h2
      dsimp only at hcnt
      rw [List.count_reverse] at hcnt
      have hcnt1 : (List.replicate (n - 1) false ++ [true]).count true = 1 := by
        simp [List.count_append, List.count_replicate]
      have hcnt2 : (compWord (List.replicate (n - 1) false ++ [true])).count true = n - 1 := by
        simp [compWord, List.map_append, List.count_append]
      omega
  rw [if_neg hnot]

end BDCOrbitSymmetry
