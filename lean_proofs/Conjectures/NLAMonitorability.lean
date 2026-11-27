import Mathlib

/-!
# Resolution of the Natural Language Autoencoder Monitorability Conjectures
(Denamganaï, arXiv:2610.18727, Section `sec:open`, Conjectures `conj:idnodisc` and `conj:paraphrase`)

We formalize in Lean 4 + Mathlib (complete proofs, standard axioms only) the mathematical cores of
both conjectures:

1. **Orthogonal codebook exact AUC values (`conj:idnodisc` parts (a), (d), and tie-vs.-`J = 0` tension)**:
   - `codebook_loss_zero`: For every `n ≥ 2` and permutation `p : Equiv.Perm (Fin n)`, the
     deterministic grounded codebook `AV(e_i) = blue (p i)` with `v(blue (p i)) = e_i` achieves
     exact generative cosine loss `J = 0`.
   - `codebook_provenance_auc_one`: Strict provenance AUC equals `1` across all `n(n - 1)` ordered
     distinct pairs.
   - `codebook_truth_auc_zero` & `codebook_truth_midrank_half`: Out-of-support rival extension
     `v(red (p i)) = e_i` attains strict truth AUC `0` and midrank truth AUC `1/2` at `J = 0`.
   - `codebook_truth_auc_one`: Out-of-support rival extension `v(red (p i)) = e_{(i+1) mod n}`
     attains strict truth AUC `1` at `J = 0`.
   - `codebook_truth_auc_half`: Mixing red (`e_i`) and green (`e_{(i+1) mod n}`) rivals equally
     attains strict truth AUC `1/2` at `J = 0`.
   - `unit_dot_basis_le_one` & `unit_dot_basis_eq_one_iff`: For any unit vector `w ∈ ℝⁿ`,
     `⟨e_i, w⟩ ≤ 1` with equality iff `w = e_i`; hence at `J = 0` strict losses are impossible
     (every `AUC < 1` witness must rely on ties) and strict truth AUC `1` holds on the open
     complement `∏_i (S^{n-1} \ {e_i})`.
2. **Sharp `2ε` paraphrase margin sandwich (`conj:paraphrase` part (a))**:
   - `paraphrase_margin_sandwich`: On any finite probability space with failure event of weight
     `≤ q` outside a good event `E` where `|e₀|, |e₁| ≤ ε`,
     `max 0 (P(M > 2ε) - q) ≤ AUC_strict ≤ min 1 (P(M > -2ε) + q)`.
   - `paraphrase_margin_sandwich_exact` & `paraphrase_auc_deviation`: When `q = 0`,
     `P(M > 2ε) ≤ AUC_strict ≤ P(M > -2ε)` and
     `|AUC_strict - P(M > 0)| ≤ P(-2ε < M ≤ 2ε)`.
   - `paraphrase_sandwich_sharp_endpoints` & `paraphrase_small_eps_reversal`: Endpoint tie
     sharpness at `M = ±2ε`, both win/loss attainability on `(-2ε, 2ε)`, and an explicit unit-vector
     reversal in `ℝ²` flipping `AUC_strict` from `1` to `0` for every `0 < ε < 1/2`.
3. **Explicit 50-token authorship counterexample (`conj:paraphrase` part (b))**:
   - `authorship50_unit_vectors`, `authorship50_hamming_distances`,
     `authorship50_zero_control_damage`, `authorship50_rival_reversal`: Five 50-token accounts
     with unit-vector reconstructions in `ℝ²` where both 1-token true controls have zero angular
     control damage, both false rivals retain 49/50 (98%) AV-authored tokens, and switching the
     single rival color word (`red` → `scarlet`) shifts the cosine score by `3/5` and flips strict
     AUC from `1` to `0`.
4. **Steganographic / audit-gap counterexamples in both directions (`conj:paraphrase` part (c))**:
   - `steg_word_identity_refutes_upper`: Lexical marker channel (`indeed`/`certainly`) achieves
     raw steganographic gap `1/2` (relative gap `1`) with zero paraphrase damage.
   - `steg_word_order_refutes_lower`: Public word-order channel (`Alice leads Bob` vs
     `Bob leads Alice`) achieves steganographic gap `0` with normalized shuffle angular damage `1`.

Machine-checked vs prose-only:
- Machine-checked: All orthogonal codebook AUC regimes (`0`, `1/2`, `1`, midrank `1/2`), `J = 0`
  tie characterization, the weighted finite-space `2ε` margin sandwich (with `q`-failure extension,
  endpoint sharpness, and unit-vector small-`ε` reversal), the 50-token red/scarlet counterexample,
  and both steganographic-gap counterexamples.
- Prose-only (in `solutions/nla_monitorability/proof.pdf`): The continuous 2-symbol planar
  Gaussian integral (`J* = 1 - 2/π`), external Gaussian noise CDF bound `Φ(B_w/σ) - 1/2`,
  product spherical measure-1 computation in `conj:idnodisc`(d), and information-theoretic
  selection-effect calculations in `conj:idnodisc`(c).
-/

open BigOperators Finset

namespace NLAMonitorability

/-! ### 1. Orthogonal Codebook Exact AUC Regimes (`conj:idnodisc` parts (a) & (d)) -/

/-- Standard Euclidean inner product on `Fin n → ℝ`. -/
def dot {n : ℕ} (u v : Fin n → ℝ) : ℝ :=
  ∑ k : Fin n, u k * v k

/-- Standard basis vector `e_i ∈ ℝⁿ`. -/
def unitBasis {n : ℕ} (i : Fin n) : Fin n → ℝ :=
  fun k => if k = i then 1 else 0

lemma dot_unitBasis_right {n : ℕ} (u : Fin n → ℝ) (i : Fin n) :
    dot u (unitBasis i) = u i := by
  unfold dot unitBasis
  simp

lemma dot_unitBasis_left {n : ℕ} (i : Fin n) (v : Fin n → ℝ) :
    dot (unitBasis i) v = v i := by
  unfold dot unitBasis
  simp

lemma dot_unitBasis_basis {n : ℕ} (i j : Fin n) :
    dot (unitBasis i) (unitBasis j) = if i = j then 1 else 0 := by
  rw [dot_unitBasis_left]
  unfold unitBasis
  rfl

lemma unitBasis_norm_sq {n : ℕ} (i : Fin n) :
    dot (unitBasis i) (unitBasis i) = 1 := by
  simp [dot_unitBasis_basis]

/-- Unit-vector Cauchy–Schwarz bound along any coordinate: for `‖w‖² = 1`, `⟨e_i, w⟩ ≤ 1`,
with equality iff `w = e_i`. This formalizes both the `J = 0` tie-necessity remark and the
open complement characterization `∏ (S^{n-1} \ {e_i})` of perfect-ranking extensions. -/
theorem unit_dot_basis_le_one {n : ℕ} (i : Fin n) (w : Fin n → ℝ) (hw : dot w w = 1) :
    dot (unitBasis i) w ≤ 1 := by
  rw [dot_unitBasis_left]
  have hsq : (w i) ^ 2 ≤ 1 := by
    have hle : w i * w i ≤ ∑ k : Fin n, w k * w k :=
      Finset.single_le_sum (fun k _ => mul_self_nonneg (w k)) (Finset.mem_univ i)
    unfold dot at hw
    have : (w i) ^ 2 = w i * w i := sq (w i)
    linarith
  nlinarith

theorem unit_dot_basis_eq_one_iff {n : ℕ} (i : Fin n) (w : Fin n → ℝ) (hw : dot w w = 1) :
    dot (unitBasis i) w = 1 ↔ w = unitBasis i := by
  constructor
  · intro h
    rw [dot_unitBasis_left] at h
    have hrest : ∑ k ∈ Finset.univ.erase i, w k * w k = 0 := by
      have hsplit : w i * w i + ∑ k ∈ Finset.univ.erase i, w k * w k = dot w w := by
        unfold dot
        exact Finset.add_sum_erase Finset.univ (fun k => w k * w k) (Finset.mem_univ i)
      rw [h, hw] at hsplit
      linarith
    funext k
    unfold unitBasis
    by_cases hki : k = i
    · subst hki; simp [h]
    · simp only [hki, ↓reduceIte]
      have hkmem : k ∈ Finset.univ.erase i := Finset.mem_erase.mpr ⟨hki, Finset.mem_univ k⟩
      have hnonneg : ∀ j ∈ Finset.univ.erase i, 0 ≤ w j * w j :=
        fun j _ => mul_self_nonneg (w j)
      have hzero : w k * w k = 0 :=
        (Finset.sum_eq_zero_iff_of_nonneg hnonneg).mp hrest k hkmem
      exact mul_self_eq_zero.mp hzero
  · intro h
    subst h
    exact unitBasis_norm_sq i

/-- Codebook message alphabet: grounded true templates `blue j`, and out-of-support
one-word false edits `red j` and `green j`. -/
inductive CodebookMsg (n : ℕ) where
  | blue : Fin n → CodebookMsg n
  | red : Fin n → CodebookMsg n
  | green : Fin n → CodebookMsg n
  deriving DecidableEq

/-- Cyclic shift `(i + 1) mod n` on `Fin n` (distinct from `i` whenever `n ≥ 2`). -/
def cyclicNext {n : ℕ} (hn : 2 ≤ n) (i : Fin n) : Fin n :=
  ⟨(i.val + 1) % n, Nat.mod_lt _ (by omega)⟩

lemma cyclicNext_ne {n : ℕ} (hn : 2 ≤ n) (i : Fin n) : cyclicNext hn i ≠ i := by
  intro h
  have hval : (i.val + 1) % n = i.val := congrArg Fin.val h
  by_cases hlt : i.val + 1 < n
  · rw [Nat.mod_eq_of_lt hlt] at hval; omega
  · have heq : i.val + 1 = n := by omega
    rw [heq, Nat.mod_self] at hval
    omega

/-- Deterministic grounded speaker `AV(e_i) = blue (p i)`. -/
def codebookAV {n : ℕ} (p : Equiv.Perm (Fin n)) (i : Fin n) : CodebookMsg n :=
  CodebookMsg.blue (p i)

/-- Decoder parameterized by its out-of-support rival maps `vRed`, `vGreen`. -/
def codebookAR {n : ℕ} (p : Equiv.Perm (Fin n))
    (vRed vGreen : Fin n → (Fin n → ℝ)) : CodebookMsg n → (Fin n → ℝ)
  | .blue j => unitBasis (p.symm j)
  | .red j => vRed (p.symm j)
  | .green j => vGreen (p.symm j)

/-- Surrogate cosine score `S(c, e_i) = ⟨e_i, v(c)⟩`. -/
def codebookScore {n : ℕ} (v : CodebookMsg n → (Fin n → ℝ)) (c : CodebookMsg n) (i : Fin n) : ℝ :=
  dot (unitBasis i) (v c)

/-- Generative cosine loss `J = 1 - (1/n) ∑_i S(AV(e_i), e_i)`. -/
noncomputable def codebookLoss {n : ℕ} (p : Equiv.Perm (Fin n)) (v : CodebookMsg n → (Fin n → ℝ)) : ℝ :=
  1 - (∑ i : Fin n, codebookScore v (codebookAV p i) i) / (n : ℝ)

/-- Every out-of-support rival extension achieves exact global minimum loss `J = 0`. -/
theorem codebook_loss_zero {n : ℕ} (hn : 2 ≤ n) (p : Equiv.Perm (Fin n))
    (vRed vGreen : Fin n → (Fin n → ℝ)) :
    codebookLoss p (codebookAR p vRed vGreen) = 0 := by
  unfold codebookLoss codebookScore codebookAV codebookAR
  have hterm : ∀ i : Fin n, dot (unitBasis i) (unitBasis (p.symm (p i))) = 1 := by
    intro i
    simp [unitBasis_norm_sq]
  simp only [hterm, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one]
  have hn0 : (n : ℝ) ≠ 0 := by positivity
  rw [div_self hn0, sub_self]

/-- Strict provenance AUC over all `n(n - 1)` ordered distinct pairs `(i, i')` equals `1`,
independent of out-of-support rival outputs. -/
theorem codebook_provenance_auc_one {n : ℕ} (hn : 2 ≤ n) (p : Equiv.Perm (Fin n))
    (vRed vGreen : Fin n → (Fin n → ℝ)) :
    let v := codebookAR p vRed vGreen
    let pairs := Finset.univ.filter (fun q : Fin n × Fin n => q.1 ≠ q.2)
    ((pairs.filter (fun q => codebookScore v (codebookAV p q.2) q.1 <
        codebookScore v (codebookAV p q.1) q.1)).card : ℝ) / (pairs.card : ℝ) = 1 := by
  intro v pairs
  have hall : ∀ q ∈ pairs,
      codebookScore v (codebookAV p q.2) q.1 < codebookScore v (codebookAV p q.1) q.1 := by
    intro q hq
    have hne : q.1 ≠ q.2 := (Finset.mem_filter.mp hq).2
    simp only [v, codebookScore, codebookAV, codebookAR, Equiv.symm_apply_apply,
      dot_unitBasis_basis, ↓reduceIte, hne]
    norm_num
  have hfilt : pairs.filter (fun q => codebookScore v (codebookAV p q.2) q.1 <
      codebookScore v (codebookAV p q.1) q.1) = pairs :=
    Finset.filter_true_of_mem hall
  rw [hfilt]
  have hnonempty : pairs.Nonempty := by
    refine ⟨(⟨0, by omega⟩, ⟨1, by omega⟩), ?_⟩
    simp [pairs]
  have hpos : (0 : ℝ) < (pairs.card : ℝ) := Nat.cast_pos.mpr hnonempty.card_pos
  exact div_self (ne_of_gt hpos)

/-- Regime 1 (`AUC_strict = 0`, `AUC_midrank = 1/2`): decoding `red (p i)` to `e_i` gives
exact ties on every state `i`, so strict truth AUC is `0` while midrank truth AUC is `1/2`. -/
theorem codebook_truth_auc_zero {n : ℕ} (hn : 2 ≤ n) (p : Equiv.Perm (Fin n))
    (vGreen : Fin n → (Fin n → ℝ)) :
    let v := codebookAR p unitBasis vGreen
    let strictWins := Finset.univ.filter (fun i : Fin n =>
      codebookScore v (CodebookMsg.red (p i)) i < codebookScore v (codebookAV p i) i)
    let midrankSum : ℝ := ∑ i : Fin n,
      if codebookScore v (CodebookMsg.red (p i)) i < codebookScore v (codebookAV p i) i then 1
      else if codebookScore v (CodebookMsg.red (p i)) i = codebookScore v (codebookAV p i) i
      then (1 : ℝ) / 2 else 0
    (strictWins.card : ℝ) / (n : ℝ) = 0 ∧ midrankSum / (n : ℝ) = 1 / 2 := by
  intro v strictWins midrankSum
  have htie : ∀ i : Fin n,
      codebookScore v (CodebookMsg.red (p i)) i = codebookScore v (codebookAV p i) i := by
    intro i
    simp [v, codebookScore, codebookAV, codebookAR]
  have hzero : strictWins = ∅ := by
    ext i
    simp [strictWins, htie i]
  have hmid : midrankSum = (n : ℝ) * (1 / 2) := by
    simp [midrankSum, htie]
  have hn0 : (n : ℝ) ≠ 0 := by positivity
  constructor
  · simp [hzero]
  · rw [hmid, mul_div_cancel_left₀ _ hn0]

/-- Regime 2 (`AUC_strict = 1`): decoding `red (p i)` to `e_{(i+1) mod n}` gives strict truth
AUC equal to `1` at the exact same loss `J = 0`. -/
theorem codebook_truth_auc_one {n : ℕ} (hn : 2 ≤ n) (p : Equiv.Perm (Fin n))
    (vGreen : Fin n → (Fin n → ℝ)) :
    let v := codebookAR p (fun i => unitBasis (cyclicNext hn i)) vGreen
    let strictWins := Finset.univ.filter (fun i : Fin n =>
      codebookScore v (CodebookMsg.red (p i)) i < codebookScore v (codebookAV p i) i)
    (strictWins.card : ℝ) / (n : ℝ) = 1 := by
  intro v strictWins
  have hall : ∀ i ∈ (Finset.univ : Finset (Fin n)),
      codebookScore v (CodebookMsg.red (p i)) i < codebookScore v (codebookAV p i) i := by
    intro i _
    have hne : i ≠ cyclicNext hn i := (cyclicNext_ne hn i).symm
    simp only [v, codebookScore, codebookAV, codebookAR, Equiv.symm_apply_apply,
      dot_unitBasis_basis, ↓reduceIte, hne]
    norm_num
  have hfilt : strictWins = Finset.univ := Finset.filter_true_of_mem hall
  rw [hfilt]
  have hn0 : (n : ℝ) ≠ 0 := by positivity
  simp [div_self hn0]

/-- Regime 3 (`AUC_strict = 1/2`): mixing tied red rivals (`v(red (p i)) = e_i`) and strictly
beaten green rivals (`v(green (p i)) = e_{(i+1) mod n}`) equally yields strict truth AUC `1/2`
at `J = 0`. -/
theorem codebook_truth_auc_half {n : ℕ} (hn : 2 ≤ n) (p : Equiv.Perm (Fin n)) :
    let v := codebookAR p unitBasis (fun i => unitBasis (cyclicNext hn i))
    let lineup : Fin n × Bool → CodebookMsg n :=
      fun q => if q.2 then CodebookMsg.green (p q.1) else CodebookMsg.red (p q.1)
    let strictWins := Finset.univ.filter (fun q : Fin n × Bool =>
      codebookScore v (lineup q) q.1 < codebookScore v (codebookAV p q.1) q.1)
    (strictWins.card : ℝ) / ((Finset.univ : Finset (Fin n × Bool)).card : ℝ) = 1 / 2 := by
  intro v lineup strictWins
  have hiff : ∀ q : Fin n × Bool,
      (codebookScore v (lineup q) q.1 < codebookScore v (codebookAV p q.1) q.1) ↔ q.2 = true := by
    intro ⟨i, b⟩
    cases b
    · simp [v, lineup, codebookScore, codebookAV, codebookAR]
    · have hne : i ≠ cyclicNext hn i := (cyclicNext_ne hn i).symm
      simp [v, lineup, codebookScore, codebookAV, codebookAR, dot_unitBasis_basis, hne]
  have hcard : strictWins.card = n := by
    have heq : strictWins = Finset.univ.image (fun i : Fin n => (i, true)) := by
      ext ⟨i, b⟩
      simp [strictWins, hiff]
    rw [heq, Finset.card_image_of_injective]
    · simp
    · intro a b hab
      exact congrArg Prod.fst hab
  have htot : (Finset.univ : Finset (Fin n × Bool)).card = 2 * n := by
    simp [mul_comm]
  rw [hcard, htot]
  have hn0 : (n : ℝ) ≠ 0 := by positivity
  push_cast
  field_simp [hn0]

/-! ### 2. Sharp `2ε` Paraphrase Margin Sandwich (`conj:paraphrase` part (a)) -/

/-- Weighted probability of a decidable event on a finite sample space. -/
def finProb {Ω : Type*} [Fintype Ω] (w : Ω → ℝ) (P : Ω → Prop) [DecidablePred P] : ℝ :=
  ∑ ω ∈ Finset.univ.filter P, w ω

lemma finProb_nonneg {Ω : Type*} [Fintype Ω] {w : Ω → ℝ} (hw : ∀ ω, 0 ≤ w ω)
    (P : Ω → Prop) [DecidablePred P] : 0 ≤ finProb w P :=
  Finset.sum_nonneg fun ω _ => hw ω

lemma finProb_le_one {Ω : Type*} [Fintype Ω] {w : Ω → ℝ} (hw : ∀ ω, 0 ≤ w ω)
    (hsum : ∑ ω : Ω, w ω = 1) (P : Ω → Prop) [DecidablePred P] : finProb w P ≤ 1 := by
  unfold finProb
  calc ∑ ω ∈ Finset.univ.filter P, w ω
      ≤ ∑ ω : Ω, w ω := Finset.sum_le_univ_sum_of_nonneg hw
    _ = 1 := hsum

lemma finProb_mono {Ω : Type*} [Fintype Ω] {w : Ω → ℝ} (hw : ∀ ω, 0 ≤ w ω)
    {P Q : Ω → Prop} [DecidablePred P] [DecidablePred Q] (h : ∀ ω, P ω → Q ω) :
    finProb w P ≤ finProb w Q := by
  unfold finProb
  apply Finset.sum_le_sum_of_subset_of_nonneg
  · intro ω hω
    exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, h ω (Finset.mem_filter.mp hω).2⟩
  · intro ω _ _; exact hw ω

/-- **Theorem 5 (`conj:paraphrase` part (a)) — General `2ε` margin sandwich with failure weight `q`**.
Suppose `S₀ = g₀ + e₀`, `S₁ = g₁ + e₁`, `M = g₀ - g₁`, and on a good set `E` with
`∑_{ω ∉ E} w(ω) ≤ q` we have `|e₀(ω)| ≤ ε` and `|e₁(ω)| ≤ ε`. Then
`max 0 (P(M > 2ε) - q) ≤ P(S₀ > S₁) ≤ min 1 (P(M > -2ε) + q)`. -/
theorem paraphrase_margin_sandwich {Ω : Type*} [Fintype Ω] [DecidableEq Ω]
    (w : Ω → ℝ) (hw : ∀ ω, 0 ≤ w ω) (hsum : ∑ ω : Ω, w ω = 1)
    (g₀ g₁ e₀ e₁ : Ω → ℝ) (ε q : ℝ) (E : Finset Ω)
    (hfail : ∑ ω ∈ Eᶜ, w ω ≤ q)
    (herr : ∀ ω ∈ E, |e₀ ω| ≤ ε ∧ |e₁ ω| ≤ ε) :
    let S₀ := fun ω => g₀ ω + e₀ ω
    let S₁ := fun ω => g₁ ω + e₁ ω
    let M := fun ω => g₀ ω - g₁ ω
    max 0 (finProb w (fun ω => 2 * ε < M ω) - q) ≤ finProb w (fun ω => S₁ ω < S₀ ω) ∧
    finProb w (fun ω => S₁ ω < S₀ ω) ≤ min 1 (finProb w (fun ω => -2 * ε < M ω) + q) := by
  classical
  intro S₀ S₁ M
  have hlower_sub : Finset.univ.filter (fun ω => 2 * ε < M ω) ⊆
      Finset.univ.filter (fun ω => S₁ ω < S₀ ω) ∪ Eᶜ := by
    intro ω hω
    have hM : 2 * ε < M ω := (Finset.mem_filter.mp hω).2
    by_cases hE : ω ∈ E
    · apply Finset.mem_union_left
      refine Finset.mem_filter.mpr ⟨Finset.mem_univ _, ?_⟩
      obtain ⟨h0, h1⟩ := herr ω hE
      have h0' := abs_le.mp h0
      have h1' := abs_le.mp h1
      dsimp only [S₀, S₁, M] at hM ⊢
      linarith
    · apply Finset.mem_union_right
      exact Finset.mem_compl.mpr hE
  have hupper_sub : Finset.univ.filter (fun ω => S₁ ω < S₀ ω) ⊆
      Finset.univ.filter (fun ω => -2 * ε < M ω) ∪ Eᶜ := by
    intro ω hω
    have hS : S₁ ω < S₀ ω := (Finset.mem_filter.mp hω).2
    by_cases hE : ω ∈ E
    · apply Finset.mem_union_left
      refine Finset.mem_filter.mpr ⟨Finset.mem_univ _, ?_⟩
      obtain ⟨h0, h1⟩ := herr ω hE
      have h0' := abs_le.mp h0
      have h1' := abs_le.mp h1
      dsimp only [S₀, S₁, M] at hS ⊢
      linarith
    · apply Finset.mem_union_right
      exact Finset.mem_compl.mpr hE
  have hsum_union : ∀ A B : Finset Ω, ∑ ω ∈ A ∪ B, w ω ≤ ∑ ω ∈ A, w ω + ∑ ω ∈ B, w ω := by
    intro A B
    have h := Finset.sum_union_inter (s₁ := A) (s₂ := B) (f := w)
    have hnn : 0 ≤ ∑ ω ∈ A ∩ B, w ω := Finset.sum_nonneg fun ω _ => hw ω
    linarith
  constructor
  · apply max_le (finProb_nonneg hw _)
    have h1 : finProb w (fun ω => 2 * ε < M ω) ≤
        finProb w (fun ω => S₁ ω < S₀ ω) + ∑ ω ∈ Eᶜ, w ω := by
      unfold finProb
      calc ∑ ω ∈ Finset.univ.filter (fun ω => 2 * ε < M ω), w ω
          ≤ ∑ ω ∈ Finset.univ.filter (fun ω => S₁ ω < S₀ ω) ∪ Eᶜ, w ω :=
            Finset.sum_le_sum_of_subset_of_nonneg hlower_sub (fun ω _ _ => hw ω)
        _ ≤ ∑ ω ∈ Finset.univ.filter (fun ω => S₁ ω < S₀ ω), w ω + ∑ ω ∈ Eᶜ, w ω :=
            hsum_union _ _
    linarith
  · apply le_min (finProb_le_one hw hsum _)
    have h2 : finProb w (fun ω => S₁ ω < S₀ ω) ≤
        finProb w (fun ω => -2 * ε < M ω) + ∑ ω ∈ Eᶜ, w ω := by
      unfold finProb
      calc ∑ ω ∈ Finset.univ.filter (fun ω => S₁ ω < S₀ ω), w ω
          ≤ ∑ ω ∈ Finset.univ.filter (fun ω => -2 * ε < M ω) ∪ Eᶜ, w ω :=
            Finset.sum_le_sum_of_subset_of_nonneg hupper_sub (fun ω _ _ => hw ω)
        _ ≤ ∑ ω ∈ Finset.univ.filter (fun ω => -2 * ε < M ω), w ω + ∑ ω ∈ Eᶜ, w ω :=
            hsum_union _ _
    linarith

/-- Exact `q = 0` paraphrase margin sandwich: `P(M > 2ε) ≤ AUC_strict ≤ P(M > -2ε)`. -/
theorem paraphrase_margin_sandwich_exact {Ω : Type*} [Fintype Ω] [DecidableEq Ω]
    (w : Ω → ℝ) (hw : ∀ ω, 0 ≤ w ω) (hsum : ∑ ω : Ω, w ω = 1)
    (g₀ g₁ e₀ e₁ : Ω → ℝ) (ε : ℝ)
    (herr : ∀ ω, |e₀ ω| ≤ ε ∧ |e₁ ω| ≤ ε) :
    let S₀ := fun ω => g₀ ω + e₀ ω
    let S₁ := fun ω => g₁ ω + e₁ ω
    let M := fun ω => g₀ ω - g₁ ω
    finProb w (fun ω => 2 * ε < M ω) ≤ finProb w (fun ω => S₁ ω < S₀ ω) ∧
    finProb w (fun ω => S₁ ω < S₀ ω) ≤ finProb w (fun ω => -2 * ε < M ω) := by
  classical
  intro S₀ S₁ M
  have h := paraphrase_margin_sandwich w hw hsum g₀ g₁ e₀ e₁ ε 0 Finset.univ
    (by simp) (fun ω _ => herr ω)
  have h1 : finProb w (fun ω => 2 * ε < M ω) ≤ finProb w (fun ω => S₁ ω < S₀ ω) := by
    have := le_trans (le_max_right _ _) h.1
    linarith
  have h2 : finProb w (fun ω => S₁ ω < S₀ ω) ≤ finProb w (fun ω => -2 * ε < M ω) := by
    have := le_trans h.2 (min_le_right _ _)
    linarith
  exact ⟨h1, h2⟩

/-- Deviation of `AUC_strict` from meaning-only `P(M > 0)` is bounded by the `(-2ε, 2ε]` margin
band probability `P(-2ε < M ≤ 2ε)`. -/
theorem paraphrase_auc_deviation {Ω : Type*} [Fintype Ω] [DecidableEq Ω]
    (w : Ω → ℝ) (hw : ∀ ω, 0 ≤ w ω) (hsum : ∑ ω : Ω, w ω = 1)
    (g₀ g₁ e₀ e₁ : Ω → ℝ) {ε : ℝ} (hε : 0 ≤ ε)
    (herr : ∀ ω, |e₀ ω| ≤ ε ∧ |e₁ ω| ≤ ε) :
    let S₀ := fun ω => g₀ ω + e₀ ω
    let S₁ := fun ω => g₁ ω + e₁ ω
    let M := fun ω => g₀ ω - g₁ ω
    |finProb w (fun ω => S₁ ω < S₀ ω) - finProb w (fun ω => 0 < M ω)| ≤
      finProb w (fun ω => -2 * ε < M ω ∧ M ω ≤ 2 * ε) := by
  classical
  intro S₀ S₁ M
  obtain ⟨hlo, hhi⟩ := paraphrase_margin_sandwich_exact w hw hsum g₀ g₁ e₀ e₁ ε herr
  have h0lo : finProb w (fun ω => 2 * ε < M ω) ≤ finProb w (fun ω => 0 < M ω) :=
    finProb_mono hw (fun ω hω => by linarith)
  have h0hi : finProb w (fun ω => 0 < M ω) ≤ finProb w (fun ω => -2 * ε < M ω) :=
    finProb_mono hw (fun ω hω => by linarith)
  have hband : finProb w (fun ω => -2 * ε < M ω) =
      finProb w (fun ω => 2 * ε < M ω) + finProb w (fun ω => -2 * ε < M ω ∧ M ω ≤ 2 * ε) := by
    unfold finProb
    rw [← Finset.sum_union]
    · congr 1
      ext ω
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_union]
      constructor
      · intro h
        by_cases hgt : 2 * ε < M ω
        · exact Or.inl hgt
        · exact Or.inr ⟨h, not_lt.mp hgt⟩
      · rintro (h | ⟨h, _⟩)
        · linarith
        · exact h
    · rw [Finset.disjoint_left]
      intro ω h1 h2
      have hgt : 2 * ε < M ω := (Finset.mem_filter.mp h1).2
      have hle : M ω ≤ 2 * ε := (Finset.mem_filter.mp h2).2.2
      linarith
  rw [abs_le]
  constructor <;> linarith

/-- **Sharpness of both `2ε` sandwich endpoints and interior reversal**:
- At `M = 2ε`, `e₀ = -ε, e₁ = ε` achieves a tie (`S₀ = S₁`, strict win false).
- At `M = -2ε`, even the most favorable errors `|e₀| ≤ ε, |e₁| ≤ ε` cannot produce a strict win.
- For every `M ∈ (-2ε, 2ε)`, extreme form errors `±ε` realize both strict win and strict loss. -/
theorem paraphrase_sandwich_sharp_endpoints {ε : ℝ} (hε : 0 ≤ ε) :
    (∀ g₀ g₁ : ℝ, g₀ - g₁ = 2 * ε →
      ∃ e₀ e₁ : ℝ, |e₀| ≤ ε ∧ |e₁| ≤ ε ∧ ¬(g₁ + e₁ < g₀ + e₀)) ∧
    (∀ g₀ g₁ e₀ e₁ : ℝ, g₀ - g₁ = -2 * ε → |e₀| ≤ ε → |e₁| ≤ ε → ¬(g₁ + e₁ < g₀ + e₀)) ∧
    (∀ g₀ g₁ : ℝ, -2 * ε < g₀ - g₁ → g₀ - g₁ < 2 * ε →
      (∃ e₀ e₁ : ℝ, |e₀| ≤ ε ∧ |e₁| ≤ ε ∧ g₁ + e₁ < g₀ + e₀) ∧
      (∃ e₀ e₁ : ℝ, |e₀| ≤ ε ∧ |e₁| ≤ ε ∧ g₀ + e₀ < g₁ + e₁)) := by
  refine ⟨?_, ?_, ?_⟩
  · intro g₀ g₁ hM
    refine ⟨-ε, ε, ?_, ?_, ?_⟩
    · rw [abs_neg, abs_of_nonneg hε]
    · rw [abs_of_nonneg hε]
    · linarith
  · intro g₀ g₁ e₀ e₁ hM h0 h1
    have h0' := abs_le.mp h0
    have h1' := abs_le.mp h1
    linarith
  · intro g₀ g₁ hgt hlt
    refine ⟨⟨ε, -ε, ?_, ?_, ?_⟩, ⟨-ε, ε, ?_, ?_, ?_⟩⟩
    · rw [abs_of_nonneg hε]
    · rw [abs_neg, abs_of_nonneg hε]
    · linarith
    · rw [abs_neg, abs_of_nonneg hε]
    · rw [abs_of_nonneg hε]
    · linarith

/-- **Arbitrarily small `ε ∈ (0, 1/2)` reverses strict AUC from `1` to `0` on unit vectors in `ℝ²`**:
At `h = e₀ ∈ ℝ²`, setting `g(true) = ε`, `g(false) = 0`, `e₀ = -ε`, `e₁ = +ε` gives unit vectors
`v₀ = (0, 1)` and `v₁ = (ε, √(1 - ε²))` with meaning margin `M = ε > 0` yet `S(c₀, h) = 0 < ε = S(c₁, h)`. -/
theorem paraphrase_small_eps_reversal {ε : ℝ} (hε0 : 0 < ε) (hε1 : ε < 1 / 2) :
    let v₀ : Fin 2 → ℝ := ![0, 1]
    let v₁ : Fin 2 → ℝ := ![ε, Real.sqrt (1 - ε ^ 2)]
    let g₀ : ℝ := ε
    let g₁ : ℝ := 0
    let e₀ : ℝ := -ε
    let e₁ : ℝ := ε
    dot v₀ v₀ = 1 ∧ dot v₁ v₁ = 1 ∧
    dot (unitBasis (0 : Fin 2)) v₀ = g₀ + e₀ ∧
    dot (unitBasis (0 : Fin 2)) v₁ = g₁ + e₁ ∧
    |e₀| ≤ ε ∧ |e₁| ≤ ε ∧
    0 < g₀ - g₁ ∧
    dot (unitBasis (0 : Fin 2)) v₀ < dot (unitBasis (0 : Fin 2)) v₁ := by
  intro v₀ v₁ g₀ g₁ e₀ e₁
  have h1m : 0 ≤ 1 - ε ^ 2 := by nlinarith
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [dot, v₀, Fin.sum_univ_two]
  · simp only [dot, v₁, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one]
    have hsq : Real.sqrt (1 - ε ^ 2) * Real.sqrt (1 - ε ^ 2) = 1 - ε ^ 2 :=
      Real.mul_self_sqrt h1m
    linarith
  · simp [dot, unitBasis, v₀, g₀, e₀]
  · simp [dot, unitBasis, v₁, g₁, e₁]
  · simp [e₀, abs_of_pos hε0]
  · simp [e₁, abs_of_pos hε0]
  · dsimp only [g₀, g₁]; linarith
  · simp [dot, unitBasis, v₀, v₁, hε0]

/-! ### 3. Explicit 50-Token Authorship Counterexample (`conj:paraphrase` part (b)) -/

/-- Vocabulary for the 50-token single-word authorship counterexample (Proposition 6). -/
inductive AuthToken where
  | blue | red | scarlet | frameA | frameB | frameC
  deriving DecidableEq

/-- A 50-token account represented as `Fin 50 → AuthToken`. -/
abbrev Account50 := Fin 50 → AuthToken

def origTrue50 : Account50 :=
  fun i => if i = 0 then .blue else .frameA

def extTruePara50 : Account50 :=
  fun i => if i = 0 then .blue else if i = 1 then .frameB else .frameA

def avSubControl50 : Account50 :=
  fun i => if i = 0 then .blue else if i = 1 then .frameC else .frameA

def nativeFalseRed50 : Account50 :=
  fun i => if i = 0 then .red else .frameA

def extFalseScarlet50 : Account50 :=
  fun i => if i = 0 then .scarlet else .frameA

/-- Number of differing token positions between two 50-token accounts. -/
def tokenHamming (a b : Account50) : ℕ :=
  (Finset.univ.filter (fun i : Fin 50 => a i ≠ b i)).card

/-- Unit-vector decoder in `ℝ²` depending solely on token 0 (the color word). -/
noncomputable def decoder50 (c : Account50) : Fin 2 → ℝ :=
  match c 0 with
  | .blue => ![3 / 5, 4 / 5]
  | .red => ![1 / 5, Real.sqrt 24 / 5]
  | .scarlet => ![4 / 5, 3 / 5]
  | _ => ![1, 0]

/-- All five 50-token accounts decode to unit vectors in `ℝ²`. -/
theorem authorship50_unit_vectors :
    dot (decoder50 origTrue50) (decoder50 origTrue50) = 1 ∧
    dot (decoder50 extTruePara50) (decoder50 extTruePara50) = 1 ∧
    dot (decoder50 avSubControl50) (decoder50 avSubControl50) = 1 ∧
    dot (decoder50 nativeFalseRed50) (decoder50 nativeFalseRed50) = 1 ∧
    dot (decoder50 extFalseScarlet50) (decoder50 extFalseScarlet50) = 1 := by
  have hred : dot (decoder50 nativeFalseRed50) (decoder50 nativeFalseRed50) = 1 := by
    simp only [dot, decoder50, nativeFalseRed50, ↓reduceIte, Fin.sum_univ_two,
      Matrix.cons_val_zero, Matrix.cons_val_one]
    have hsq : Real.sqrt 24 * Real.sqrt 24 = 24 := Real.mul_self_sqrt (by norm_num)
    nlinarith
  refine ⟨?_, ?_, ?_, hred, ?_⟩ <;>
    (simp [dot, decoder50, origTrue50, extTruePara50, avSubControl50, extFalseScarlet50,
      Fin.sum_univ_two]; norm_num)

/-- Every edited account (both true controls and both false rivals) differs from `origTrue50`
in exactly `1` of the `50` token positions (retaining 49/50 = 98% AV-authored tokens). -/
theorem authorship50_hamming_distances :
    tokenHamming origTrue50 extTruePara50 = 1 ∧
    tokenHamming origTrue50 avSubControl50 = 1 ∧
    tokenHamming origTrue50 nativeFalseRed50 = 1 ∧
    tokenHamming origTrue50 extFalseScarlet50 = 1 := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · have h : Finset.univ.filter (fun i : Fin 50 => origTrue50 i ≠ extTruePara50 i) = {1} := by
      ext i
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton,
        origTrue50, extTruePara50]
      by_cases h0 : i = 0
      · subst h0; simp
      · by_cases h1 : i = 1
        · subst h1; simp
        · simp [h0, h1]
    simp [tokenHamming, h]
  · have h : Finset.univ.filter (fun i : Fin 50 => origTrue50 i ≠ avSubControl50 i) = {1} := by
      ext i
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton,
        origTrue50, avSubControl50]
      by_cases h0 : i = 0
      · subst h0; simp
      · by_cases h1 : i = 1
        · subst h1; simp
        · simp [h0, h1]
    simp [tokenHamming, h]
  · have h : Finset.univ.filter (fun i : Fin 50 => origTrue50 i ≠ nativeFalseRed50 i) = {0} := by
      ext i
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton,
        origTrue50, nativeFalseRed50]
      by_cases h0 : i = 0
      · subst h0; simp
      · simp [h0]
    simp [tokenHamming, h]
  · have h : Finset.univ.filter (fun i : Fin 50 => origTrue50 i ≠ extFalseScarlet50 i) = {0} := by
      ext i
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton,
        origTrue50, extFalseScarlet50]
      by_cases h0 : i = 0
      · subst h0; simp
      · simp [h0]
    simp [tokenHamming, h]

/-- Both true-account controls reconstruct identically to the original (zero matched authorship
control damage), while switching the false rival from native `"red"` to external `"scarlet"`
shifts the cosine score at `e₀` by `3/5` and reverses strict preference from win (`3/5 > 1/5`)
to loss (`3/5 < 4/5`). -/
theorem authorship50_zero_control_damage_and_reversal :
    decoder50 extTruePara50 = decoder50 origTrue50 ∧
    decoder50 avSubControl50 = decoder50 origTrue50 ∧
    dot (unitBasis (0 : Fin 2)) (decoder50 origTrue50) = 3 / 5 ∧
    dot (unitBasis (0 : Fin 2)) (decoder50 nativeFalseRed50) = 1 / 5 ∧
    dot (unitBasis (0 : Fin 2)) (decoder50 extFalseScarlet50) = 4 / 5 ∧
    dot (unitBasis (0 : Fin 2)) (decoder50 extFalseScarlet50) -
      dot (unitBasis (0 : Fin 2)) (decoder50 nativeFalseRed50) = 3 / 5 ∧
    dot (unitBasis (0 : Fin 2)) (decoder50 nativeFalseRed50) <
      dot (unitBasis (0 : Fin 2)) (decoder50 origTrue50) ∧
    dot (unitBasis (0 : Fin 2)) (decoder50 origTrue50) <
      dot (unitBasis (0 : Fin 2)) (decoder50 extFalseScarlet50) := by
  have h1 : decoder50 extTruePara50 = decoder50 origTrue50 := by
    simp [decoder50, extTruePara50, origTrue50]
  have h2 : decoder50 avSubControl50 = decoder50 origTrue50 := by
    simp [decoder50, avSubControl50, origTrue50]
  have hs0 : dot (unitBasis (0 : Fin 2)) (decoder50 origTrue50) = 3 / 5 := by
    simp [dot, unitBasis, decoder50, origTrue50]
  have hsR : dot (unitBasis (0 : Fin 2)) (decoder50 nativeFalseRed50) = 1 / 5 := by
    simp [dot, unitBasis, decoder50, nativeFalseRed50]
  have hsS : dot (unitBasis (0 : Fin 2)) (decoder50 extFalseScarlet50) = 4 / 5 := by
    simp [dot, unitBasis, decoder50, extFalseScarlet50]
  refine ⟨h1, h2, hs0, hsR, hsS, ?_, ?_, ?_⟩ <;> linarith

/-! ### 4. Steganographic / Audit-Gap Counterexamples (`conj:paraphrase` part (c)) -/

/-- Raw decision-theoretic steganographic gap (Anwar et al., arXiv:2602.23163):
`(U_Rec - U_Rec^0) - (U_Sent - U_Sent^0)`. -/
def stegGap (uRec uRec0 uSent uSent0 : ℝ) : ℝ :=
  (uRec - uRec0) - (uSent - uSent0)

/-- Accuracy of a binary predictor `pred : Bool → Bool` on a uniform fair bit `Y ∈ {false, true}`. -/
noncomputable def fairBitAccuracy (pred : Bool → Bool) : ℝ :=
  ((if pred false = false then (1 : ℝ) else 0) +
   (if pred true = true then (1 : ℝ) else 0)) / 2

/-- **Word-identity channel (refutes upper bound by paraphrase damage)**:
Encoding `Y = 0` as `("text", "indeed", "today")` and `Y = 1` as `("text", "certainly", "today")`,
paraphrasing `"today" ↦ "now"` leaves the AR vector invariant (zero angular damage on every
input), while the Receiver recovers `Y` perfectly (`gain = 1/2`) and any marker-blind Sentinel
is stuck at chance (`gain = 0`), yielding raw gap `1/2` (relative gap `1`). -/
theorem steg_word_identity_refutes_upper :
    let encode : Bool → String × String × String :=
      fun y => if y then ("text", "certainly", "today") else ("text", "indeed", "today")
    let rewrite : String × String × String → String × String × String :=
      fun m => (m.1, m.2.1, "now")
    let arVec : String × String × String → (Fin 2 → ℝ) :=
      fun m => if m.2.1 = "certainly" then unitBasis 1 else unitBasis 0
    let recDec : (Fin 2 → ℝ) → Bool := fun v => decide (v 1 = 1)
    let sentDec : String × String × String → Bool := fun _ => false
    (∀ y : Bool, arVec (rewrite (encode y)) = arVec (encode y)) ∧
    (∀ y : Bool, Real.arccos (dot (arVec (encode y)) (arVec (rewrite (encode y)))) = 0) ∧
    fairBitAccuracy (fun y => recDec (arVec (encode y))) = 1 ∧
    fairBitAccuracy (fun y => sentDec (encode y)) = 1 / 2 ∧
    stegGap (fairBitAccuracy (fun y => recDec (arVec (encode y)))) (1 / 2)
      (fairBitAccuracy (fun y => sentDec (encode y))) (1 / 2) = 1 / 2 := by
  intro encode rewrite arVec recDec sentDec
  have hinv : ∀ y : Bool, arVec (rewrite (encode y)) = arVec (encode y) := by
    intro y; cases y <;> rfl
  have hang : ∀ y : Bool, Real.arccos (dot (arVec (encode y)) (arVec (rewrite (encode y)))) = 0 := by
    intro y
    rw [hinv y]
    cases y <;> simp [arVec, encode, unitBasis_norm_sq, Real.arccos_one]
  have hrec : fairBitAccuracy (fun y => recDec (arVec (encode y))) = 1 := by
    simp [fairBitAccuracy, recDec, arVec, encode, unitBasis]
  have hsent : fairBitAccuracy (fun y => sentDec (encode y)) = 1 / 2 := by
    simp [fairBitAccuracy, sentDec]
  refine ⟨hinv, hang, hrec, hsent, ?_⟩
  rw [hrec, hsent]
  unfold stegGap
  norm_num

/-- **Word-order channel (refutes lower bound by shuffle damage)**:
Encoding `Y = 0` as `("Alice", "leads", "Bob")` and `Y = 1` as `("Bob", "leads", "Alice")`,
both Receiver and Sentinel read the public leading name and achieve accuracy `1` (so
`stegGap = 0`), while reversing token order maps every AR vector to an orthogonal direction,
giving normalized shuffle angular damage `(π/2) / (π/2) = 1` on every input. -/
theorem steg_word_order_refutes_lower :
    let encode : Bool → String × String × String :=
      fun y => if y then ("Bob", "leads", "Alice") else ("Alice", "leads", "Bob")
    let revShuffle : String × String × String → String × String × String :=
      fun m => (m.2.2, m.2.1, m.1)
    let arVec : String × String × String → (Fin 2 → ℝ) :=
      fun m => if m.1 = "Bob" then unitBasis 1 else unitBasis 0
    let publicDec : String × String × String → Bool := fun m => decide (m.1 = "Bob")
    (∀ y : Bool, dot (arVec (encode y)) (arVec (revShuffle (encode y))) = 0) ∧
    (∀ y : Bool,
      Real.arccos (dot (arVec (encode y)) (arVec (revShuffle (encode y)))) / (Real.pi / 2) = 1) ∧
    fairBitAccuracy (fun y => publicDec (encode y)) = 1 ∧
    stegGap (fairBitAccuracy (fun y => publicDec (encode y))) (1 / 2)
      (fairBitAccuracy (fun y => publicDec (encode y))) (1 / 2) = 0 := by
  intro encode revShuffle arVec publicDec
  have horth : ∀ y : Bool, dot (arVec (encode y)) (arVec (revShuffle (encode y))) = 0 := by
    intro y
    cases y <;> simp [arVec, encode, revShuffle, dot_unitBasis_basis]
  have hang : ∀ y : Bool,
      Real.arccos (dot (arVec (encode y)) (arVec (revShuffle (encode y)))) / (Real.pi / 2) = 1 := by
    intro y
    rw [horth y, Real.arccos_zero]
    have hpi : Real.pi / 2 ≠ 0 := by positivity
    exact div_self hpi
  have hacc : fairBitAccuracy (fun y => publicDec (encode y)) = 1 := by
    simp [fairBitAccuracy, publicDec, encode]
  refine ⟨horth, hang, hacc, ?_⟩
  unfold stegGap
  ring

end NLAMonitorability
