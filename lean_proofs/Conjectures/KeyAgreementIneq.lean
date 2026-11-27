import Mathlib

/-!
# Logical Incompatibility of the Key-Agreement Conjectures and Exact Collapse of `Ψ_Δ`
(Hamidreza Abin and Amin Gohari, *On the Source Model Key Agreement Problem*,
arXiv:2502.00294 / IEEE Trans. Inf. Theory 72(8):5317-5334, Aug. 2026)

We formalize in Lean 4 + Mathlib (complete proofs with no unfinished placeholders,
standard Lean foundations only) the exact information-theoretic core of Theorems 1 and 2
and the triangle-source rate comparison of Proposition 5:

1. **Finite-alphabet Shannon information theory (`hTerm`, `entropy`, `law`, `H`, `Hcond`, `MI`, `CMI`)**:
   - Bijection invariance (`H_comp_equiv`), deterministic redundancy (`H_pair_of_func`),
     constant entropy (`H_const`), and exact KL representation of conditional mutual information
     (`CMI_eq_kl_sum`).
   - Nonnegativity of `CMI`, `MI`, and `Hcond` (`CMI_nonneg`, `MI_nonneg`, `Hcond_nonneg`) from
     `Real.log_le_sub_one_of_pos`, plus exact zero criteria under conditional independence,
     independence, and functional determinism.
2. **Lemma 1(1) — Exact 11-term joint-entropy identity (`entropy_identity_11_terms`)**:
   - For every finite joint probability distribution `P` and random variables `(T, A, B, E)`:
     `H(T|E) - H(T|A) - H(T|B) + H(T|A,B,E) = I(A;B) - I(A;B|T) - I(T;E) - I(T;E|A,B)`.
3. **Lemma 1(2) — Explicit 4-state product-mixture lifting (`xorLiftP`)**:
   - For every binary distribution `(a, b, c, d)` on `Bool × Bool` (including boundary cases
     `a + d = 0` or `b + c = 0`), constructs the 4-state lifted distribution `xorLiftP a b c d`
     with `E = xor A B` and proves:
     * `xorLiftP_isProb`: valid probability distribution;
     * `xorLiftP_marg_AB`: exact `(A, B)` marginal `(a, b, c, d)`;
     * `xorLiftP_CMI_AB_T`: `I(A; B | T) = 0`;
     * `xorLiftP_MI_T_E`: `I(T; E) = 0`;
     * `xorLiftP_CMI_TE_AB`: `I(T; E | A, B) = 0`;
     * `xorLiftP_Hcond_AB_TE`: `H(A, B | T, E) = 0`.
4. **Theorem 1 — Joint incompatibility (`conj1_gap_identity`, `conj1_implies_not_conj2`)**:
   - Under the three standard operational properties of `S` and `S_ow` (`S(A;B‖E) ≤ I(A;B)`,
     `S(T;T‖E) = H(T|E)`, `S_ow(T,T → E‖E) = 0`), proves the exact Conjecture 1 gap identity
     `LHS(1) - RHS(1) = I(A;B) - S(A;B‖E) - I(A;B|T) - I(T;E) - I(T;E|A,B)`, deduces that
     Conjecture 1 forces `S(A;B‖E) = I(A;B)` for every lifted source, and concludes
     `Conjecture1 → ¬ Conjecture2`.
5. **Theorem 2 — Exact collapse of `Ψ_Δ` and intrinsic information (`psi_delta_eq_MI`, `intrinsic_info_eq_MI`)**:
   - Proves the Markov chain-rule bound `I(A;B) - I(A;B|J) ≤ I(J;T)` whenever `I(A;B|T) = 0`
     and `I(T;J|A,B) = 0`.
   - For every binary XOR source `(a, b, c, d)` and every `0 ≤ Δ ≤ H(E)`, proves that:
     * every finite channel `P_{J|A,B}` admits the feasible choice `(U, V) = (T, const)`
       achieving `I(A;B|J) + I(U;J|V) - I(U;E|V) ≥ I(A;B)` (with `H(A,B|U,V,E) = 0` and
       `I(V;E) ≤ I(V;J) + H(E) - Δ`);
     * the trivial channel `J = const` bounds every feasible `(U, V)` above by `I(A;B)`;
     * consequently the exact inf-sup bound `Ψ_Δ(A;B‖E) = I(A;B)` identically, and the
       Maurer–Wolf intrinsic information satisfies `I(A;B ↓ E) = I(A;B)`.
6. **Proposition 5 (Triangle source exact values)**:
   - Exact mutual information `I(A;B) = log₂ 3 - 4/3` for `(1/3, 1/3, 1/3, 0)` and strict
     inequality `(1 + log₂ 3)/12 < log₂ 3 - 4/3`.

Machine-checked vs. prose-only (in `solutions/key_agreement_ineq/proof.pdf`):
- Machine-checked: All finite-alphabet entropy identities and inequalities, the 11-term identity
  (Lemma 1(1)), the explicit 4-state XOR lifting and all four vanishing quantities (Lemma 1(2)),
  the logical incompatibility deduction `Conjecture 1 → ¬ Conjecture 2` (Theorem 1), the exact
  collapse `Ψ_Δ = I(A;B)` and `I(A;B ↓ E) = I(A;B)` (Theorem 2), and the exact triangle-source
  mutual-information evaluation.
- Prose-only: Multi-letter operational achievability/converse coding proofs for `S` and `S_ow`
  (Ahlswede–Csiszár 1993), Proposition 3 (one-way equality characterization), Proposition 4
  (two-sample interactive protocol), and the multi-sample triple-grouping protocol analysis
  of Proposition 5.
-/

open scoped BigOperators
open Finset

namespace KeyAgreementIneq

/-! ### 1. Finite-Alphabet Information Theory -/

/-- Base-2 Shannon entropy contribution `-x log₂ x` (with `0 * log 0 = 0`). -/
noncomputable def hTerm (x : ℝ) : ℝ := -(x * Real.log x) / Real.log 2

lemma hTerm_zero : hTerm 0 = 0 := by
  simp [hTerm]

lemma hTerm_one : hTerm 1 = 0 := by
  simp [hTerm]

/-- Base-2 Shannon entropy of a finite weight function `p : α → ℝ`. -/
noncomputable def entropy {α : Type*} [Fintype α] (p : α → ℝ) : ℝ :=
  ∑ x : α, hTerm (p x)

/-- Finite probability distribution predicate. -/
structure IsProb {Ω : Type*} [Fintype Ω] (P : Ω → ℝ) : Prop where
  nonneg : ∀ ω, 0 ≤ P ω
  sum_one : ∑ ω : Ω, P ω = 1

/-- Pushforward distribution (law) of a random variable `X : Ω → α`. -/
def law {Ω α : Type*} [Fintype Ω] [DecidableEq α] (P : Ω → ℝ) (X : Ω → α) (a : α) : ℝ :=
  ∑ ω : Ω, if X ω = a then P ω else 0

lemma law_nonneg {Ω α : Type*} [Fintype Ω] [DecidableEq α]
    {P : Ω → ℝ} (hP : IsProb P) (X : Ω → α) (a : α) :
    0 ≤ law P X a := by
  apply Finset.sum_nonneg
  intro ω _
  split_ifs
  · exact hP.nonneg ω
  · exact le_rfl

lemma law_sum_one {Ω α : Type*} [Fintype Ω] [Fintype α] [DecidableEq α]
    {P : Ω → ℝ} (hP : IsProb P) (X : Ω → α) :
    ∑ a : α, law P X a = 1 := by
  unfold law
  rw [Finset.sum_comm]
  have h1 : ∀ ω : Ω, (∑ a : α, if X ω = a then P ω else 0) = P ω := by
    intro ω
    simp
  simp_rw [h1]
  exact hP.sum_one

lemma law_isProb {Ω α : Type*} [Fintype Ω] [Fintype α] [DecidableEq α]
    {P : Ω → ℝ} (hP : IsProb P) (X : Ω → α) :
    IsProb (law P X) :=
  ⟨law_nonneg hP X, law_sum_one hP X⟩

/-- Shannon entropy `H(X)` of a random variable `X : Ω → α`. -/
noncomputable def H {Ω α : Type*} [Fintype Ω] [Fintype α] [DecidableEq α]
    (P : Ω → ℝ) (X : Ω → α) : ℝ :=
  entropy (law P X)

/-- Conditional entropy `H(X | Y) = H(X, Y) - H(Y)`. -/
noncomputable def Hcond {Ω α β : Type*} [Fintype Ω] [Fintype α] [Fintype β]
    [DecidableEq α] [DecidableEq β] (P : Ω → ℝ) (X : Ω → α) (Y : Ω → β) : ℝ :=
  H P (fun ω => (X ω, Y ω)) - H P Y

/-- Mutual information `I(X; Y) = H(X) + H(Y) - H(X, Y)`. -/
noncomputable def MI {Ω α β : Type*} [Fintype Ω] [Fintype α] [Fintype β]
    [DecidableEq α] [DecidableEq β] (P : Ω → ℝ) (X : Ω → α) (Y : Ω → β) : ℝ :=
  H P X + H P Y - H P (fun ω => (X ω, Y ω))

/-- Conditional mutual information `I(X; Y | Z) = H(X, Z) + H(Y, Z) - H((X, Y), Z) - H(Z)`. -/
noncomputable def CMI {Ω α β γ : Type*} [Fintype Ω] [Fintype α] [Fintype β] [Fintype γ]
    [DecidableEq α] [DecidableEq β] [DecidableEq γ]
    (P : Ω → ℝ) (X : Ω → α) (Y : Ω → β) (Z : Ω → γ) : ℝ :=
  H P (fun ω => (X ω, Z ω)) + H P (fun ω => (Y ω, Z ω)) -
    H P (fun ω => ((X ω, Y ω), Z ω)) - H P Z

lemma law_comp_equiv {Ω α β : Type*} [Fintype Ω] [DecidableEq α] [DecidableEq β]
    (P : Ω → ℝ) (X : Ω → α) (e : α ≃ β) (a : α) :
    law P (e ∘ X) (e a) = law P X a := by
  unfold law
  congr 1
  ext ω
  simp

lemma H_comp_equiv {Ω α β : Type*} [Fintype Ω] [Fintype α] [Fintype β]
    [DecidableEq α] [DecidableEq β] (P : Ω → ℝ) (X : Ω → α) (e : α ≃ β) :
    H P (e ∘ X) = H P X := by
  unfold H entropy
  rw [← Fintype.sum_equiv e (fun a => hTerm (law P X a)) (fun b => hTerm (law P (e ∘ X) b))]
  intro a
  rw [law_comp_equiv]

lemma H_swap {Ω α β : Type*} [Fintype Ω] [Fintype α] [Fintype β]
    [DecidableEq α] [DecidableEq β] (P : Ω → ℝ) (X : Ω → α) (Y : Ω → β) :
    H P (fun ω => (Y ω, X ω)) = H P (fun ω => (X ω, Y ω)) :=
  H_comp_equiv P (fun ω => (X ω, Y ω)) (Equiv.prodComm α β)

lemma H_assoc {Ω α β γ : Type*} [Fintype Ω] [Fintype α] [Fintype β] [Fintype γ]
    [DecidableEq α] [DecidableEq β] [DecidableEq γ]
    (P : Ω → ℝ) (X : Ω → α) (Y : Ω → β) (Z : Ω → γ) :
    H P (fun ω => (X ω, (Y ω, Z ω))) = H P (fun ω => ((X ω, Y ω), Z ω)) :=
  H_comp_equiv P (fun ω => ((X ω, Y ω), Z ω)) (Equiv.prodAssoc α β γ)

lemma H_const_unit {Ω : Type*} [Fintype Ω] {P : Ω → ℝ} (hP : IsProb P) :
    H P (fun _ : Ω => ()) = 0 := by
  unfold H entropy law
  simp [hP.sum_one, hTerm_one]

lemma H_pair_unit_right {Ω α : Type*} [Fintype Ω] [Fintype α] [DecidableEq α]
    (P : Ω → ℝ) (X : Ω → α) :
    H P (fun ω => (X ω, ())) = H P X :=
  H_comp_equiv P X (Equiv.prodPUnit α).symm

lemma H_pair_det_right {Ω α β : Type*} [Fintype Ω] [Fintype α] [Fintype β]
    [DecidableEq α] [DecidableEq β] {P : Ω → ℝ}
    (X : Ω → α) (Y : Ω → β) (f : α → β) (hf : ∀ ω, P ω ≠ 0 → Y ω = f (X ω)) :
    H P (fun ω => (X ω, Y ω)) = H P X := by
  unfold H entropy
  rw [Fintype.sum_prod_type]
  congr 1
  ext a
  have hlaw : ∀ b : β, law P (fun ω => (X ω, Y ω)) (a, b) =
      if b = f a then law P X a else 0 := by
    intro b
    unfold law
    split_ifs with hbf
    · subst hbf
      apply Finset.sum_congr rfl
      intro ω _
      by_cases hP0 : P ω = 0
      · simp [hP0]
      · have hy := hf ω hP0
        by_cases hx : X ω = a
        · simp [hx, hy]
        · simp [hx]
    · apply Finset.sum_eq_zero
      intro ω _
      by_cases hP0 : P ω = 0
      · simp [hP0]
      · have hy := hf ω hP0
        split_ifs with hxy
        · exfalso
          have hx : X ω = a := (Prod.mk.inj hxy).1
          have hy' : Y ω = b := (Prod.mk.inj hxy).2
          rw [hx] at hy
          exact hbf (hy'.symm.trans hy)
        · rfl
  simp_rw [hlaw]
  have hterm : ∀ b : β, hTerm (if b = f a then law P X a else 0) =
      if b = f a then hTerm (law P X a) else 0 := by
    intro b; split_ifs <;> simp [hTerm_zero]
  simp_rw [hterm]
  simp

lemma H_pair_det_left {Ω α β : Type*} [Fintype Ω] [Fintype α] [Fintype β]
    [DecidableEq α] [DecidableEq β] {P : Ω → ℝ}
    (X : Ω → α) (Y : Ω → β) (f : β → α) (hf : ∀ ω, P ω ≠ 0 → X ω = f (Y ω)) :
    H P (fun ω => (X ω, Y ω)) = H P Y := by
  rw [H_swap P Y X]
  exact H_pair_det_right Y X f hf

lemma Hcond_eq_zero_of_det {Ω α β : Type*} [Fintype Ω] [Fintype α] [Fintype β]
    [DecidableEq α] [DecidableEq β] {P : Ω → ℝ}
    (X : Ω → α) (Y : Ω → β) (f : β → α) (hf : ∀ ω, P ω ≠ 0 → X ω = f (Y ω)) :
    Hcond P X Y = 0 := by
  unfold Hcond
  rw [H_pair_det_left X Y f hf, sub_self]

lemma law_pair_marg₁ {Ω α β : Type*} [Fintype Ω] [Fintype β] [DecidableEq α] [DecidableEq β]
    (P : Ω → ℝ) (X : Ω → α) (Y : Ω → β) (a : α) :
    ∑ b : β, law P (fun ω => (X ω, Y ω)) (a, b) = law P X a := by
  unfold law
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro ω _
  by_cases hx : X ω = a
  · simp [hx]
  · simp [hx]

lemma law_pair_marg₂ {Ω α β : Type*} [Fintype Ω] [Fintype α] [DecidableEq α] [DecidableEq β]
    (P : Ω → ℝ) (X : Ω → α) (Y : Ω → β) (b : β) :
    ∑ a : α, law P (fun ω => (X ω, Y ω)) (a, b) = law P Y b := by
  unfold law
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro ω _
  by_cases hy : Y ω = b
  · simp [hy]
  · simp [hy]

lemma law_triple_marg_XZ {Ω α β γ : Type*} [Fintype Ω] [Fintype β]
    [DecidableEq α] [DecidableEq β] [DecidableEq γ]
    (P : Ω → ℝ) (X : Ω → α) (Y : Ω → β) (Z : Ω → γ) (a : α) (c : γ) :
    ∑ b : β, law P (fun ω => ((X ω, Y ω), Z ω)) ((a, b), c) =
      law P (fun ω => (X ω, Z ω)) (a, c) := by
  unfold law
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro ω _
  by_cases hxz : X ω = a ∧ Z ω = c
  · simp [hxz.1, hxz.2]
  · have : ∀ b : β, ¬(((X ω, Y ω), Z ω) = ((a, b), c)) := by
      intro b h
      exact hxz ⟨by simpa using congrArg (fun t => t.1.1) h,
        by simpa using congrArg (fun t => t.2) h⟩
    have hxz' : ¬((X ω, Z ω) = (a, c)) := by
      intro h; exact hxz ⟨congrArg Prod.fst h, congrArg Prod.snd h⟩
    simp [this, hxz']

lemma law_triple_marg_YZ {Ω α β γ : Type*} [Fintype Ω] [Fintype α]
    [DecidableEq α] [DecidableEq β] [DecidableEq γ]
    (P : Ω → ℝ) (X : Ω → α) (Y : Ω → β) (Z : Ω → γ) (b : β) (c : γ) :
    ∑ a : α, law P (fun ω => ((X ω, Y ω), Z ω)) ((a, b), c) =
      law P (fun ω => (Y ω, Z ω)) (b, c) := by
  unfold law
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro ω _
  by_cases hyz : Y ω = b ∧ Z ω = c
  · simp [hyz.1, hyz.2]
  · have : ∀ a : α, ¬(((X ω, Y ω), Z ω) = ((a, b), c)) := by
      intro a h
      exact hyz ⟨by simpa using congrArg (fun t => t.1.2) h,
        by simpa using congrArg (fun t => t.2) h⟩
    have hyz' : ¬((Y ω, Z ω) = (b, c)) := by
      intro h; exact hyz ⟨congrArg Prod.fst h, congrArg Prod.snd h⟩
    simp [this, hyz']

/-- Exact KL-divergence sum formula for conditional mutual information `I(X; Y | Z)`. -/
theorem CMI_eq_kl_sum {Ω α β γ : Type*} [Fintype Ω] [Fintype α] [Fintype β] [Fintype γ]
    [DecidableEq α] [DecidableEq β] [DecidableEq γ]
    {P : Ω → ℝ} (hP : IsProb P) (X : Ω → α) (Y : Ω → β) (Z : Ω → γ) :
    let pXYZ := fun a b c => law P (fun ω => ((X ω, Y ω), Z ω)) ((a, b), c)
    let pXZ := fun a c => law P (fun ω => (X ω, Z ω)) (a, c)
    let pYZ := fun b c => law P (fun ω => (Y ω, Z ω)) (b, c)
    let pZ := fun c => law P Z c
    CMI P X Y Z =
      (∑ c : γ, ∑ a : α, ∑ b : β,
        pXYZ a b c * Real.log (pXYZ a b c * pZ c / (pXZ a c * pYZ b c))) / Real.log 2 := by
  intro pXYZ pXZ pYZ pZ
  have hlog_split : ∀ c a b,
      pXYZ a b c * Real.log (pXYZ a b c * pZ c / (pXZ a c * pYZ b c)) =
      pXYZ a b c * Real.log (pXYZ a b c) + pXYZ a b c * Real.log (pZ c) -
      pXYZ a b c * Real.log (pXZ a c) - pXYZ a b c * Real.log (pYZ b c) := by
    intro c a b
    by_cases h0 : pXYZ a b c = 0
    · simp [h0]
    · have hpos : 0 < pXYZ a b c :=
        lt_of_le_of_ne (law_nonneg hP (fun ω => ((X ω, Y ω), Z ω)) ((a, b), c)) (Ne.symm h0)
      have hXZ_ge : pXYZ a b c ≤ pXZ a c := by
        dsimp only [pXYZ, pXZ]
        rw [← law_triple_marg_XZ P X Y Z a c]
        exact Finset.single_le_sum (fun b' _ => law_nonneg hP _ ((a, b'), c)) (Finset.mem_univ b)
      have hYZ_ge : pXYZ a b c ≤ pYZ b c := by
        dsimp only [pXYZ, pYZ]
        rw [← law_triple_marg_YZ P X Y Z b c]
        exact Finset.single_le_sum (fun a' _ => law_nonneg hP _ ((a', b), c)) (Finset.mem_univ a)
      have hZ_ge : pXZ a c ≤ pZ c := by
        dsimp only [pXZ, pZ]
        rw [← law_pair_marg₂ P X Z c]
        exact Finset.single_le_sum (fun a' _ => law_nonneg hP _ (a', c)) (Finset.mem_univ a)
      have hXZ_pos : 0 < pXZ a c := lt_of_lt_of_le hpos hXZ_ge
      have hYZ_pos : 0 < pYZ b c := lt_of_lt_of_le hpos hYZ_ge
      have hZ_pos : 0 < pZ c := lt_of_lt_of_le hXZ_pos hZ_ge
      rw [Real.log_div (ne_of_gt (mul_pos hpos hZ_pos)) (ne_of_gt (mul_pos hXZ_pos hYZ_pos)),
        Real.log_mul (ne_of_gt hpos) (ne_of_gt hZ_pos),
        Real.log_mul (ne_of_gt hXZ_pos) (ne_of_gt hYZ_pos)]
      ring
  simp_rw [hlog_split]
  have hsum1 : (∑ c : γ, ∑ a : α, ∑ b : β, pXYZ a b c * Real.log (pXYZ a b c)) =
      ∑ t : (α × β) × γ, law P (fun ω => ((X ω, Y ω), Z ω)) t *
        Real.log (law P (fun ω => ((X ω, Y ω), Z ω)) t) := by
    symm
    rw [Fintype.sum_prod_type, Finset.sum_comm]
    apply Finset.sum_congr rfl; intro c _
    rw [Fintype.sum_prod_type]
  have hsum2 : (∑ c : γ, ∑ a : α, ∑ b : β, pXYZ a b c * Real.log (pZ c)) =
      ∑ c : γ, pZ c * Real.log (pZ c) := by
    apply Finset.sum_congr rfl; intro c _
    simp_rw [← Finset.sum_mul]
    have hmarg : (∑ a : α, ∑ b : β, pXYZ a b c) = pZ c := by
      dsimp only [pXYZ, pZ]
      simp_rw [law_triple_marg_XZ P X Y Z]
      exact law_pair_marg₂ P X Z c
    rw [hmarg]
  have hsum3 : (∑ c : γ, ∑ a : α, ∑ b : β, pXYZ a b c * Real.log (pXZ a c)) =
      ∑ ac : α × γ, law P (fun ω => (X ω, Z ω)) ac * Real.log (law P (fun ω => (X ω, Z ω)) ac) := by
    symm
    rw [Fintype.sum_prod_type, Finset.sum_comm]
    apply Finset.sum_congr rfl; intro c _
    apply Finset.sum_congr rfl; intro a _
    rw [← Finset.sum_mul]
    dsimp only [pXYZ, pXZ]
    rw [law_triple_marg_XZ P X Y Z a c]
  have hsum4 : (∑ c : γ, ∑ a : α, ∑ b : β, pXYZ a b c * Real.log (pYZ b c)) =
      ∑ bc : β × γ, law P (fun ω => (Y ω, Z ω)) bc * Real.log (law P (fun ω => (Y ω, Z ω)) bc) := by
    symm
    rw [Fintype.sum_prod_type, Finset.sum_comm]
    apply Finset.sum_congr rfl; intro c _
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl; intro b _
    rw [← Finset.sum_mul]
    dsimp only [pXYZ, pYZ]
    rw [law_triple_marg_YZ P X Y Z b c]
  have hsplit_sum :
      (∑ c : γ, ∑ a : α, ∑ b : β,
        (pXYZ a b c * Real.log (pXYZ a b c) + pXYZ a b c * Real.log (pZ c) -
          pXYZ a b c * Real.log (pXZ a c) - pXYZ a b c * Real.log (pYZ b c))) =
      (∑ c : γ, ∑ a : α, ∑ b : β, pXYZ a b c * Real.log (pXYZ a b c)) +
      (∑ c : γ, ∑ a : α, ∑ b : β, pXYZ a b c * Real.log (pZ c)) -
      (∑ c : γ, ∑ a : α, ∑ b : β, pXYZ a b c * Real.log (pXZ a c)) -
      (∑ c : γ, ∑ a : α, ∑ b : β, pXYZ a b c * Real.log (pYZ b c)) := by
    simp only [Finset.sum_sub_distrib, Finset.sum_add_distrib]
  rw [hsplit_sum, hsum1, hsum2, hsum3, hsum4]
  unfold CMI H entropy hTerm
  simp only [← Finset.sum_div, Finset.sum_neg_distrib]
  ring

/-- Nonnegativity of conditional mutual information `I(X; Y | Z) ≥ 0`. -/
theorem CMI_nonneg {Ω α β γ : Type*} [Fintype Ω] [Fintype α] [Fintype β] [Fintype γ]
    [DecidableEq α] [DecidableEq β] [DecidableEq γ]
    {P : Ω → ℝ} (hP : IsProb P) (X : Ω → α) (Y : Ω → β) (Z : Ω → γ) :
    0 ≤ CMI P X Y Z := by
  rw [CMI_eq_kl_sum hP X Y Z]
  apply div_nonneg _ (le_of_lt (Real.log_pos (by norm_num : (1 : ℝ) < 2)))
  apply Finset.sum_nonneg
  intro c _
  set pZ := law P Z c
  have hpZ : 0 ≤ pZ := law_nonneg hP Z c
  by_cases hZ0 : pZ = 0
  · have hzero : ∀ a b, law P (fun ω => ((X ω, Y ω), Z ω)) ((a, b), c) = 0 := by
      intro a b
      have hle1 : law P (fun ω => ((X ω, Y ω), Z ω)) ((a, b), c) ≤
          law P (fun ω => (X ω, Z ω)) (a, c) := by
        rw [← law_triple_marg_XZ P X Y Z a c]
        exact Finset.single_le_sum (fun b' _ => law_nonneg hP _ ((a, b'), c)) (Finset.mem_univ b)
      have hle2 : law P (fun ω => (X ω, Z ω)) (a, c) ≤ pZ := by
        dsimp only [pZ]
        rw [← law_pair_marg₂ P X Z c]
        exact Finset.single_le_sum (fun a' _ => law_nonneg hP _ (a', c)) (Finset.mem_univ a)
      linarith [law_nonneg hP (fun ω => ((X ω, Y ω), Z ω)) ((a, b), c)]
    simp [hzero]
  · have hpZ_pos : 0 < pZ := lt_of_le_of_ne hpZ (Ne.symm hZ0)
    have hpt : ∀ a b,
        law P (fun ω => ((X ω, Y ω), Z ω)) ((a, b), c) -
          law P (fun ω => (X ω, Z ω)) (a, c) * law P (fun ω => (Y ω, Z ω)) (b, c) / pZ ≤
        law P (fun ω => ((X ω, Y ω), Z ω)) ((a, b), c) *
          Real.log (law P (fun ω => ((X ω, Y ω), Z ω)) ((a, b), c) * pZ /
            (law P (fun ω => (X ω, Z ω)) (a, c) * law P (fun ω => (Y ω, Z ω)) (b, c))) := by
      intro a b
      set p := law P (fun ω => ((X ω, Y ω), Z ω)) ((a, b), c)
      set q := law P (fun ω => (X ω, Z ω)) (a, c) * law P (fun ω => (Y ω, Z ω)) (b, c) / pZ
      have hp0 : 0 ≤ p := law_nonneg hP _ ((a, b), c)
      have hq0 : 0 ≤ q := div_nonneg (mul_nonneg (law_nonneg hP _ (a, c)) (law_nonneg hP _ (b, c)))
        (le_of_lt hpZ_pos)
      by_cases hp : p = 0
      · simp [hp, hq0]
      · have hp_pos : 0 < p := lt_of_le_of_ne hp0 (Ne.symm hp)
        have hXZ_pos : 0 < law P (fun ω => (X ω, Z ω)) (a, c) := by
          have hle : p ≤ law P (fun ω => (X ω, Z ω)) (a, c) := by
            rw [← law_triple_marg_XZ P X Y Z a c]
            exact Finset.single_le_sum (fun b' _ => law_nonneg hP _ ((a, b'), c)) (Finset.mem_univ b)
          linarith
        have hYZ_pos : 0 < law P (fun ω => (Y ω, Z ω)) (b, c) := by
          have hle : p ≤ law P (fun ω => (Y ω, Z ω)) (b, c) := by
            rw [← law_triple_marg_YZ P X Y Z b c]
            exact Finset.single_le_sum (fun a' _ => law_nonneg hP _ ((a', b), c)) (Finset.mem_univ a)
          linarith
        have hq_pos : 0 < q := div_pos (mul_pos hXZ_pos hYZ_pos) hpZ_pos
        have hlog : Real.log (q / p) ≤ q / p - 1 := Real.log_le_sub_one_of_pos (div_pos hq_pos hp_pos)
        have hmul : p * Real.log (q / p) ≤ q - p := by
          have := mul_le_mul_of_nonneg_left hlog (le_of_lt hp_pos)
          have hr : p * (q / p - 1) = q - p := by field_simp [ne_of_gt hp_pos]
          linarith
        have hinv : p * pZ / (law P (fun ω => (X ω, Z ω)) (a, c) *
            law P (fun ω => (Y ω, Z ω)) (b, c)) = (q / p)⁻¹ := by
          dsimp only [q]
          field_simp [ne_of_gt hp_pos, ne_of_gt hpZ_pos, ne_of_gt hXZ_pos, ne_of_gt hYZ_pos]
        rw [hinv, Real.log_inv]
        linarith
    have hsum_ge :
        (∑ a : α, ∑ b : β,
          (law P (fun ω => ((X ω, Y ω), Z ω)) ((a, b), c) -
            law P (fun ω => (X ω, Z ω)) (a, c) * law P (fun ω => (Y ω, Z ω)) (b, c) / pZ)) ≤
        ∑ a : α, ∑ b : β,
          law P (fun ω => ((X ω, Y ω), Z ω)) ((a, b), c) *
            Real.log (law P (fun ω => ((X ω, Y ω), Z ω)) ((a, b), c) * pZ /
              (law P (fun ω => (X ω, Z ω)) (a, c) * law P (fun ω => (Y ω, Z ω)) (b, c))) :=
      Finset.sum_le_sum fun a _ => Finset.sum_le_sum fun b _ => hpt a b
    have hsum_zero :
        (∑ a : α, ∑ b : β,
          (law P (fun ω => ((X ω, Y ω), Z ω)) ((a, b), c) -
            law P (fun ω => (X ω, Z ω)) (a, c) * law P (fun ω => (Y ω, Z ω)) (b, c) / pZ)) = 0 := by
      simp only [Finset.sum_sub_distrib]
      have h1 : (∑ a : α, ∑ b : β, law P (fun ω => ((X ω, Y ω), Z ω)) ((a, b), c)) = pZ := by
        simp_rw [law_triple_marg_XZ P X Y Z]
        exact law_pair_marg₂ P X Z c
      have h2 : (∑ a : α, ∑ b : β,
          law P (fun ω => (X ω, Z ω)) (a, c) * law P (fun ω => (Y ω, Z ω)) (b, c) / pZ) = pZ := by
        simp_rw [← Finset.sum_div, ← Finset.mul_sum, law_pair_marg₂ P Y Z c, ← Finset.sum_mul,
          law_pair_marg₂ P X Z c]
        change pZ * pZ / pZ = pZ
        field_simp [ne_of_gt hpZ_pos]
      rw [h1, h2, sub_self]
    linarith

/-- Conditional independence `P(x, y, z) P(z) = P(x, z) P(y, z)` implies `I(X; Y | Z) = 0`. -/
theorem CMI_eq_zero_of_condIndep {Ω α β γ : Type*} [Fintype Ω] [Fintype α] [Fintype β] [Fintype γ]
    [DecidableEq α] [DecidableEq β] [DecidableEq γ]
    {P : Ω → ℝ} (hP : IsProb P) (X : Ω → α) (Y : Ω → β) (Z : Ω → γ)
    (hci : ∀ a b c,
      law P (fun ω => ((X ω, Y ω), Z ω)) ((a, b), c) * law P Z c =
        law P (fun ω => (X ω, Z ω)) (a, c) * law P (fun ω => (Y ω, Z ω)) (b, c)) :
    CMI P X Y Z = 0 := by
  rw [CMI_eq_kl_sum hP X Y Z]
  have hzero : ∀ c a b,
      law P (fun ω => ((X ω, Y ω), Z ω)) ((a, b), c) *
        Real.log (law P (fun ω => ((X ω, Y ω), Z ω)) ((a, b), c) * law P Z c /
          (law P (fun ω => (X ω, Z ω)) (a, c) * law P (fun ω => (Y ω, Z ω)) (b, c))) = 0 := by
    intro c a b
    by_cases h0 : law P (fun ω => ((X ω, Y ω), Z ω)) ((a, b), c) = 0
    · simp [h0]
    · have hpos : 0 < law P (fun ω => ((X ω, Y ω), Z ω)) ((a, b), c) :=
        lt_of_le_of_ne (law_nonneg hP _ ((a, b), c)) (Ne.symm h0)
      have hXZ_pos : 0 < law P (fun ω => (X ω, Z ω)) (a, c) := by
        have hle : law P (fun ω => ((X ω, Y ω), Z ω)) ((a, b), c) ≤
            law P (fun ω => (X ω, Z ω)) (a, c) := by
          rw [← law_triple_marg_XZ P X Y Z a c]
          exact Finset.single_le_sum (fun b' _ => law_nonneg hP _ ((a, b'), c)) (Finset.mem_univ b)
        linarith
      have hYZ_pos : 0 < law P (fun ω => (Y ω, Z ω)) (b, c) := by
        have hle : law P (fun ω => ((X ω, Y ω), Z ω)) ((a, b), c) ≤
            law P (fun ω => (Y ω, Z ω)) (b, c) := by
          rw [← law_triple_marg_YZ P X Y Z b c]
          exact Finset.single_le_sum (fun a' _ => law_nonneg hP _ ((a', b), c)) (Finset.mem_univ a)
        linarith
      rw [hci a b c, div_self (ne_of_gt (mul_pos hXZ_pos hYZ_pos)), Real.log_one, mul_zero]
  simp [hzero]

lemma MI_eq_CMI_unit {Ω α β : Type*} [Fintype Ω] [Fintype α] [Fintype β]
    [DecidableEq α] [DecidableEq β]
    {P : Ω → ℝ} (hP : IsProb P) (X : Ω → α) (Y : Ω → β) :
    MI P X Y = CMI P X Y (fun _ : Ω => ()) := by
  unfold MI CMI
  rw [H_pair_unit_right P X, H_pair_unit_right P Y,
    H_pair_unit_right P (fun ω => (X ω, Y ω)), H_const_unit hP]
  ring

theorem MI_nonneg {Ω α β : Type*} [Fintype Ω] [Fintype α] [Fintype β]
    [DecidableEq α] [DecidableEq β]
    {P : Ω → ℝ} (hP : IsProb P) (X : Ω → α) (Y : Ω → β) :
    0 ≤ MI P X Y := by
  rw [MI_eq_CMI_unit hP X Y]
  exact CMI_nonneg hP X Y (fun _ => ())

theorem MI_eq_zero_of_indep {Ω α β : Type*} [Fintype Ω] [Fintype α] [Fintype β]
    [DecidableEq α] [DecidableEq β]
    {P : Ω → ℝ} (hP : IsProb P) (X : Ω → α) (Y : Ω → β)
    (hind : ∀ a b, law P (fun ω => (X ω, Y ω)) (a, b) = law P X a * law P Y b) :
    MI P X Y = 0 := by
  rw [MI_eq_CMI_unit hP X Y]
  apply CMI_eq_zero_of_condIndep hP X Y (fun _ => ())
  intro a b c
  cases c
  have h1 : law P (fun ω => ((X ω, Y ω), ())) ((a, b), ()) =
      law P (fun ω => (X ω, Y ω)) (a, b) := by
    unfold law; simp
  have h2 : law P (fun _ : Ω => ()) () = 1 := by
    unfold law; simp [hP.sum_one]
  have h3 : law P (fun ω => (X ω, ())) (a, ()) = law P X a := by
    unfold law; simp
  have h4 : law P (fun ω => (Y ω, ())) (b, ()) = law P Y b := by
    unfold law; simp
  rw [h1, h2, h3, h4, mul_one, hind a b]

lemma CMI_eq_zero_of_det_right {Ω α β γ : Type*} [Fintype Ω] [Fintype α] [Fintype β] [Fintype γ]
    [DecidableEq α] [DecidableEq β] [DecidableEq γ] {P : Ω → ℝ}
    (X : Ω → α) (Y : Ω → β) (Z : Ω → γ) (f : γ → β)
    (hf : ∀ ω, P ω ≠ 0 → Y ω = f (Z ω)) :
    CMI P X Y Z = 0 := by
  unfold CMI
  have h1 : H P (fun ω => (Y ω, Z ω)) = H P Z := H_pair_det_left Y Z f hf
  have h2 : H P (fun ω => ((X ω, Y ω), Z ω)) = H P (fun ω => (X ω, Z ω)) := by
    have heq : H P (fun ω => ((X ω, Y ω), Z ω)) = H P (fun ω => ((X ω, Z ω), Y ω)) := by
      let e : (α × β) × γ ≃ (α × γ) × β :=
        { toFun := fun t => ((t.1.1, t.2), t.1.2)
          invFun := fun t => ((t.1.1, t.2), t.1.2)
          left_inv := fun ⟨⟨_, _⟩, _⟩ => rfl
          right_inv := fun ⟨⟨_, _⟩, _⟩ => rfl }
      exact (H_comp_equiv P (fun ω => ((X ω, Y ω), Z ω)) e).symm
    rw [heq]
    exact H_pair_det_right (fun ω => (X ω, Z ω)) Y (fun xz => f xz.2) (fun ω hω => hf ω hω)
  rw [h1, h2]
  ring

lemma Hcond_nonneg {Ω α β : Type*} [Fintype Ω] [Fintype α] [Fintype β]
    [DecidableEq α] [DecidableEq β] {P : Ω → ℝ} (hP : IsProb P)
    (X : Ω → α) (Y : Ω → β) :
    0 ≤ Hcond P X Y := by
  have h : Hcond P X Y = CMI P X X Y := by
    unfold Hcond CMI
    have hxx : H P (fun ω => ((X ω, X ω), Y ω)) = H P (fun ω => (X ω, Y ω)) := by
      have heq : H P (fun ω => ((X ω, X ω), Y ω)) = H P (fun ω => ((X ω, Y ω), X ω)) := by
        let e : (α × α) × β ≃ (α × β) × α :=
          { toFun := fun t => ((t.1.1, t.2), t.1.2)
            invFun := fun t => ((t.1.1, t.2), t.1.2)
            left_inv := fun ⟨⟨_, _⟩, _⟩ => rfl
            right_inv := fun ⟨⟨_, _⟩, _⟩ => rfl }
        exact (H_comp_equiv P (fun ω => ((X ω, X ω), Y ω)) e).symm
      rw [heq]
      exact H_pair_det_right (fun ω => (X ω, Y ω)) X Prod.fst (fun _ _ => rfl)
    rw [hxx]; ring
  rw [h]
  exact CMI_nonneg hP X X Y

lemma MI_symm {Ω α β : Type*} [Fintype Ω] [Fintype α] [Fintype β]
    [DecidableEq α] [DecidableEq β] (P : Ω → ℝ) (X : Ω → α) (Y : Ω → β) :
    MI P X Y = MI P Y X := by
  unfold MI
  rw [H_swap P X Y]
  ring

/-! ### 2. Lemma 1(1): Exact 11-Term Joint-Entropy Identity -/

/-- **Lemma 1(1)** (exact 11-term joint-entropy identity):
For every finite joint distribution `P` and random variables `(T, A, B, E)`,
`H(T|E) - H(T|A) - H(T|B) + H(T|A,B,E) = I(A;B) - I(A;B|T) - I(T;E) - I(T;E|A,B)`. -/
theorem entropy_identity_11_terms {Ω τ α β ε : Type*}
    [Fintype Ω] [Fintype τ] [Fintype α] [Fintype β] [Fintype ε]
    [DecidableEq τ] [DecidableEq α] [DecidableEq β] [DecidableEq ε]
    (P : Ω → ℝ) (T : Ω → τ) (A : Ω → α) (B : Ω → β) (E : Ω → ε) :
    Hcond P T E - Hcond P T A - Hcond P T B + Hcond P T (fun ω => ((A ω, B ω), E ω)) =
      MI P A B - CMI P A B T - MI P T E - CMI P T E (fun ω => (A ω, B ω)) := by
  unfold Hcond MI CMI
  have hTA : H P (fun ω => (A ω, T ω)) = H P (fun ω => (T ω, A ω)) := H_swap P T A
  have hTB : H P (fun ω => (B ω, T ω)) = H P (fun ω => (T ω, B ω)) := H_swap P T B
  have hTAB : H P (fun ω => ((A ω, B ω), T ω)) = H P (fun ω => (T ω, (A ω, B ω))) :=
    H_swap P T (fun ω => (A ω, B ω))
  have hEAB : H P (fun ω => (E ω, (A ω, B ω))) = H P (fun ω => ((A ω, B ω), E ω)) :=
    H_swap P (fun ω => (A ω, B ω)) E
  have hTABE : H P (fun ω => ((T ω, E ω), (A ω, B ω))) =
      H P (fun ω => (T ω, ((A ω, B ω), E ω))) := by
    let e : (τ × ε) × (α × β) ≃ τ × ((α × β) × ε) :=
      { toFun := fun x => (x.1.1, (x.2, x.1.2))
        invFun := fun x => ((x.1, x.2.2), x.2.1)
        left_inv := fun ⟨⟨_, _⟩, ⟨_, _⟩⟩ => rfl
        right_inv := fun ⟨_, ⟨⟨_, _⟩, _⟩⟩ => rfl }
    exact (H_comp_equiv P (fun ω => ((T ω, E ω), (A ω, B ω))) e).symm
  rw [hTA, hTB, hTAB, hEAB, hTABE]
  ring

/-! ### 3. Lemma 1(2): Explicit 4-State Product-Mixture XOR Lifting -/

/-- Parameterized 4-state product-mixture kernel on `Fin 4 × Bool × Bool` without branch terms. -/
def xorLiftRaw (s t u v : ℝ) : Fin 4 × Bool × Bool → ℝ
  | (0, false, false) => u * v * s
  | (0, false, true)  => u * v * t
  | (1, false, false) => u * (1 - v) * s
  | (1, true,  false) => u * (1 - v) * t
  | (2, false, true)  => (1 - u) * v * t
  | (2, true,  true)  => (1 - u) * v * s
  | (3, true,  false) => (1 - u) * (1 - v) * t
  | (3, true,  true)  => (1 - u) * (1 - v) * s
  | _                 => 0

/-- Explicit 4-state product-mixture lifting `P_{T,A,B}` on `Fin 4 × Bool × Bool` for any
binary law `(a, b, c, d)` at `((false,false), (false,true), (true,false), (true,true))`. -/
noncomputable def xorLiftP (a b c d : ℝ) : Fin 4 × Bool × Bool → ℝ :=
  xorLiftRaw (a + d) (b + c)
    (if a + d = 0 then 0 else a / (a + d))
    (if b + c = 0 then 0 else b / (b + c))

/-- Coordinate projections on `Fin 4 × Bool × Bool` with `E = xor A B`. -/
def liftT (ω : Fin 4 × Bool × Bool) : Fin 4 := ω.1
def liftA (ω : Fin 4 × Bool × Bool) : Bool := ω.2.1
def liftB (ω : Fin 4 × Bool × Bool) : Bool := ω.2.2
def liftE (ω : Fin 4 × Bool × Bool) : Bool := xor ω.2.1 ω.2.2

/-- Binary mass function `(a, b, c, d)` on `Bool × Bool`. -/
def binMass (a b c d : ℝ) : Bool × Bool → ℝ
  | (false, false) => a
  | (false, true)  => b
  | (true,  false) => c
  | (true,  true)  => d

/-- Validity predicate for a binary distribution `(a, b, c, d)` on `{0, 1}²`. -/
structure IsBinProb (a b c d : ℝ) : Prop where
  ha : 0 ≤ a
  hb : 0 ≤ b
  hc : 0 ≤ c
  hd : 0 ≤ d
  hsum : a + b + c + d = 1

lemma xorLift_uv_bounds {a b c d : ℝ} (h : IsBinProb a b c d) :
    let s := a + d
    let t := b + c
    let u := if s = 0 then 0 else a / s
    let v := if t = 0 then 0 else b / t
    (0 ≤ u ∧ u ≤ 1) ∧ (0 ≤ v ∧ v ≤ 1) ∧
    (u * s = a ∧ (1 - u) * s = d) ∧ (v * t = b ∧ (1 - v) * t = c) := by
  intro s t u v
  have hs0 : 0 ≤ s := add_nonneg h.ha h.hd
  have ht0 : 0 ≤ t := add_nonneg h.hb h.hc
  have hu : 0 ≤ u ∧ u ≤ 1 := by
    dsimp only [u]
    split_ifs with hs
    · norm_num
    · have hspos : 0 < s := lt_of_le_of_ne hs0 (Ne.symm hs)
      exact ⟨div_nonneg h.ha hs0, (div_le_one hspos).2 (by linarith [h.hd])⟩
  have hv : 0 ≤ v ∧ v ≤ 1 := by
    dsimp only [v]
    split_ifs with ht
    · norm_num
    · have htpos : 0 < t := lt_of_le_of_ne ht0 (Ne.symm ht)
      exact ⟨div_nonneg h.hb ht0, (div_le_one htpos).2 (by linarith [h.hc])⟩
  have hus : u * s = a ∧ (1 - u) * s = d := by
    dsimp only [u, s]
    split_ifs with hs
    · have ha0 : a = 0 := by linarith [h.ha, h.hd]
      have hd0 : d = 0 := by linarith [h.ha, h.hd]
      simp [ha0, hd0]
    · constructor
      · field_simp [hs]
      · field_simp [hs]; ring
  have hvt : v * t = b ∧ (1 - v) * t = c := by
    dsimp only [v, t]
    split_ifs with ht
    · have hb0 : b = 0 := by linarith [h.hb, h.hc]
      have hc0 : c = 0 := by linarith [h.hb, h.hc]
      simp [hb0, hc0]
    · constructor
      · field_simp [ht]
      · field_simp [ht]; ring
  exact ⟨hu, hv, hus, hvt⟩

theorem xorLiftP_isProb {a b c d : ℝ} (h : IsBinProb a b c d) :
    IsProb (xorLiftP a b c d) := by
  have ⟨⟨hu0, hu1⟩, ⟨hv0, hv1⟩, _, _⟩ := xorLift_uv_bounds h
  set s := a + d
  set t := b + c
  set u := if s = 0 then 0 else a / s
  set v := if t = 0 then 0 else b / t
  have hs0 : 0 ≤ s := add_nonneg h.ha h.hd
  have ht0 : 0 ≤ t := add_nonneg h.hb h.hc
  have h1u : 0 ≤ 1 - u := by linarith
  have h1v : 0 ≤ 1 - v := by linarith
  have hst : s + t = 1 := by dsimp only [s, t]; linarith [h.hsum]
  change IsProb (xorLiftRaw s t u v)
  constructor
  · rintro ⟨k, x, y⟩
    fin_cases k <;> cases x <;> cases y <;>
      simp only [xorLiftRaw] <;> positivity
  · rw [Fintype.sum_prod_type]
    simp only [Fin.sum_univ_four, Fintype.sum_prod_type, Fintype.sum_bool, xorLiftRaw]
    nlinarith [hst]

/-- Exact `(A, B)` marginal of `xorLiftP a b c d` equals `binMass a b c d`. -/
theorem xorLiftP_marg_AB {a b c d : ℝ} (h : IsBinProb a b c d) (xy : Bool × Bool) :
    law (xorLiftP a b c d) (fun ω => (liftA ω, liftB ω)) xy = binMass a b c d xy := by
  have ⟨_, _, ⟨hus, h1us⟩, ⟨hvt, h1vt⟩⟩ := xorLift_uv_bounds h
  set s := a + d
  set t := b + c
  set u := if s = 0 then 0 else a / s
  set v := if t = 0 then 0 else b / t
  change law (xorLiftRaw s t u v) (fun ω => (liftA ω, liftB ω)) xy = binMass a b c d xy
  rcases xy with ⟨x, y⟩
  unfold law
  rw [Fintype.sum_prod_type]
  cases x <;> cases y <;>
    simp only [Fin.sum_univ_four, Fintype.sum_prod_type, Fintype.sum_bool,
      liftA, liftB, xorLiftRaw, binMass, Prod.mk.injEq, and_true, and_false,
      ↓reduceIte, Bool.false_eq_true, Bool.true_eq_false] <;>
    linarith [hus, h1us, hvt, h1vt]

lemma xorLiftRaw_condIndep (s t u v : ℝ) (x y : Bool) (k : Fin 4) :
    law (xorLiftRaw s t u v) (fun ω => ((liftA ω, liftB ω), liftT ω)) ((x, y), k) *
      law (xorLiftRaw s t u v) liftT k =
    law (xorLiftRaw s t u v) (fun ω => (liftA ω, liftT ω)) (x, k) *
      law (xorLiftRaw s t u v) (fun ω => (liftB ω, liftT ω)) (y, k) := by
  have hABT : law (xorLiftRaw s t u v) (fun ω => ((liftA ω, liftB ω), liftT ω)) ((x, y), k) =
      xorLiftRaw s t u v (k, (x, y)) := by
    unfold law
    rw [Finset.sum_eq_single (k, (x, y))]
    · simp [liftA, liftB, liftT]
    · rintro ⟨k', x', y'⟩ _ hne
      simp only [liftA, liftB, liftT, Prod.mk.injEq]
      split_ifs with h
      · exfalso; exact hne (by rcases h with ⟨⟨rfl, rfl⟩, rfl⟩; rfl)
      · rfl
    · intro h; exact absurd (Finset.mem_univ _) h
  have hT : law (xorLiftRaw s t u v) liftT k =
      ∑ a : Bool, ∑ b : Bool, xorLiftRaw s t u v (k, (a, b)) := by
    unfold law
    rw [Fintype.sum_prod_type, Finset.sum_eq_single k]
    · simp [liftT, Fintype.sum_prod_type]
    · intro j _ hjk; simp [liftT, hjk]
    · intro h; exact absurd (Finset.mem_univ _) h
  have hAT : law (xorLiftRaw s t u v) (fun ω => (liftA ω, liftT ω)) (x, k) =
      ∑ b : Bool, xorLiftRaw s t u v (k, (x, b)) := by
    unfold law
    rw [Fintype.sum_prod_type, Finset.sum_eq_single k]
    · rw [Fintype.sum_prod_type, Finset.sum_eq_single x]
      · simp [liftA, liftT]
      · intro a _ hax; simp [liftA, liftT, hax]
      · intro h; exact absurd (Finset.mem_univ _) h
    · intro j _ hjk; simp [liftA, liftT, hjk]
    · intro h; exact absurd (Finset.mem_univ _) h
  have hBT : law (xorLiftRaw s t u v) (fun ω => (liftB ω, liftT ω)) (y, k) =
      ∑ a : Bool, xorLiftRaw s t u v (k, (a, y)) := by
    unfold law
    rw [Fintype.sum_prod_type, Finset.sum_eq_single k]
    · rw [Fintype.sum_prod_type]
      apply Finset.sum_congr rfl; intro a _
      rw [Finset.sum_eq_single y]
      · simp [liftB, liftT]
      · intro b _ hby; simp [liftB, liftT, hby]
      · intro h; exact absurd (Finset.mem_univ _) h
    · intro j _ hjk; simp [liftB, liftT, hjk]
    · intro h; exact absurd (Finset.mem_univ _) h
  rw [hABT, hT, hAT, hBT]
  fin_cases k <;> cases x <;> cases y <;>
    simp only [Fintype.sum_bool, xorLiftRaw] <;>
    ring

lemma xorLiftRaw_indep_TE (s t u v : ℝ) (hst : s + t = 1) (k : Fin 4) (e : Bool) :
    law (xorLiftRaw s t u v) (fun ω => (liftT ω, liftE ω)) (k, e) =
    law (xorLiftRaw s t u v) liftT k * law (xorLiftRaw s t u v) liftE e := by
  have ht : t = 1 - s := by linarith
  subst ht
  have hTE : law (xorLiftRaw s (1 - s) u v) (fun ω => (liftT ω, liftE ω)) (k, e) =
      ∑ a : Bool, ∑ b : Bool, if xor a b = e then xorLiftRaw s (1 - s) u v (k, (a, b)) else 0 := by
    unfold law
    rw [Fintype.sum_prod_type, Finset.sum_eq_single k]
    · simp [liftT, liftE, Fintype.sum_prod_type]
    · intro j _ hjk; simp [liftT, liftE, hjk]
    · intro h; exact absurd (Finset.mem_univ _) h
  have hT : law (xorLiftRaw s (1 - s) u v) liftT k =
      ∑ a : Bool, ∑ b : Bool, xorLiftRaw s (1 - s) u v (k, (a, b)) := by
    unfold law
    rw [Fintype.sum_prod_type, Finset.sum_eq_single k]
    · simp [liftT, Fintype.sum_prod_type]
    · intro j _ hjk; simp [liftT, hjk]
    · intro h; exact absurd (Finset.mem_univ _) h
  have hE : law (xorLiftRaw s (1 - s) u v) liftE e =
      ∑ j : Fin 4, ∑ a : Bool, ∑ b : Bool,
        if xor a b = e then xorLiftRaw s (1 - s) u v (j, (a, b)) else 0 := by
    unfold law
    rw [Fintype.sum_prod_type]
    simp only [Fintype.sum_prod_type, liftE]
  rw [hTE, hT, hE]
  fin_cases k <;> cases e <;>
    simp only [Fin.sum_univ_four, Fintype.sum_bool, xorLiftRaw, Bool.bne_false, Bool.bne_true,
      Bool.not_false, Bool.not_true, Bool.false_eq_true, Bool.true_eq_false, ↓reduceIte] <;>
    ring

/-- **Lemma 1(2a)**: `I(A; B | T) = 0` for the 4-state XOR lifting. -/
theorem xorLiftP_CMI_AB_T {a b c d : ℝ} (h : IsBinProb a b c d) :
    CMI (xorLiftP a b c d) liftA liftB liftT = 0 := by
  apply CMI_eq_zero_of_condIndep (xorLiftP_isProb h) liftA liftB liftT
  intro x y k
  exact xorLiftRaw_condIndep _ _ _ _ x y k

/-- **Lemma 1(2b)**: `I(T; E) = 0` for the 4-state XOR lifting. -/
theorem xorLiftP_MI_T_E {a b c d : ℝ} (h : IsBinProb a b c d) :
    MI (xorLiftP a b c d) liftT liftE = 0 := by
  apply MI_eq_zero_of_indep (xorLiftP_isProb h) liftT liftE
  intro k e
  exact xorLiftRaw_indep_TE _ _ _ _ (by linarith [h.hsum]) k e

/-- **Lemma 1(2c)**: `I(T; E | A, B) = 0` for the 4-state XOR lifting. -/
theorem xorLiftP_CMI_TE_AB {a b c d : ℝ} :
    CMI (xorLiftP a b c d) liftT liftE (fun ω => (liftA ω, liftB ω)) = 0 :=
  CMI_eq_zero_of_det_right liftT liftE (fun ω => (liftA ω, liftB ω))
    (fun ab => xor ab.1 ab.2) (fun _ _ => rfl)

/-- **Lemma 1(2d)**: `H(A, B | T, E) = 0` for the 4-state XOR lifting. -/
theorem xorLiftP_Hcond_AB_TE {a b c d : ℝ} :
    Hcond (xorLiftP a b c d) (fun ω => (liftA ω, liftB ω)) (fun ω => (liftT ω, liftE ω)) = 0 := by
  let f : Fin 4 × Bool → Bool × Bool
    | (0, e) => (false, e)
    | (1, e) => (e, false)
    | (2, e) => (not e, true)
    | (3, e) => (true, not e)
  apply Hcond_eq_zero_of_det _ _ f
  rintro ⟨k, x, y⟩ hne
  fin_cases k <;> cases x <;> cases y <;>
    simp [f, liftA, liftB, liftT, liftE, xorLiftP, xorLiftRaw, xor] at hne ⊢

/-! ### 4. Theorem 1: Logical Incompatibility (`Conjecture 1 → ¬ Conjecture 2`) -/

/-- Abstract secret-key capacity and one-way capacity functionals satisfying the three
elementary operational properties from Section 1 of Abin and Gohari (2026). -/
structure SecretKeyModel where
  S : ∀ {Ω α β ε : Type} [Fintype Ω] [Fintype α] [Fintype β] [Fintype ε]
    [DecidableEq α] [DecidableEq β] [DecidableEq ε],
    (Ω → ℝ) → (Ω → α) → (Ω → β) → (Ω → ε) → ℝ
  S_ow : ∀ {Ω α β γ ε : Type} [Fintype Ω] [Fintype α] [Fintype β] [Fintype γ] [Fintype ε]
    [DecidableEq α] [DecidableEq β] [DecidableEq γ] [DecidableEq ε],
    (Ω → ℝ) → (Ω → α) → (Ω → β) → (Ω → γ) → (Ω → ε) → ℝ
  S_le_MI : ∀ {Ω α β ε : Type} [Fintype Ω] [Fintype α] [Fintype β] [Fintype ε]
    [DecidableEq α] [DecidableEq β] [DecidableEq ε]
    (P : Ω → ℝ) (_ : IsProb P) (X : Ω → α) (Y : Ω → β) (Z : Ω → ε),
    S P X Y Z ≤ MI P X Y
  S_common : ∀ {Ω τ ε : Type} [Fintype Ω] [Fintype τ] [Fintype ε]
    [DecidableEq τ] [DecidableEq ε]
    (P : Ω → ℝ) (_ : IsProb P) (T : Ω → τ) (E : Ω → ε),
    S P T T E = Hcond P T E
  S_ow_self : ∀ {Ω α β ε : Type} [Fintype Ω] [Fintype α] [Fintype β] [Fintype ε]
    [DecidableEq α] [DecidableEq β] [DecidableEq ε]
    (P : Ω → ℝ) (_ : IsProb P) (X : Ω → α) (Y : Ω → β) (E : Ω → ε),
    S_ow P X Y E E = 0
  S_law_inv : ∀ {Ω₁ Ω₂ α β ε : Type} [Fintype Ω₁] [Fintype Ω₂] [Fintype α] [Fintype β] [Fintype ε]
    [DecidableEq α] [DecidableEq β] [DecidableEq ε]
    (P₁ : Ω₁ → ℝ) (P₂ : Ω₂ → ℝ) (X₁ : Ω₁ → α) (Y₁ : Ω₁ → β) (Z₁ : Ω₁ → ε)
    (X₂ : Ω₂ → α) (Y₂ : Ω₂ → β) (Z₂ : Ω₂ → ε),
    (∀ a b e, law P₁ (fun ω => ((X₁ ω, Y₁ ω), Z₁ ω)) ((a, b), e) =
              law P₂ (fun ω => ((X₂ ω, Y₂ ω), Z₂ ω)) ((a, b), e)) →
    S P₁ X₁ Y₁ Z₁ = S P₂ X₂ Y₂ Z₂

/-- **Conjecture 1** (`conjecture:1`, Abin & Gohari 2026): six-variable capacity difference bound. -/
def Conjecture1 (m : SecretKeyModel) : Prop :=
  ∀ {Ω α β ε α' β' ε' : Type} [Fintype Ω] [Fintype α] [Fintype β] [Fintype ε]
    [Fintype α'] [Fintype β'] [Fintype ε']
    [DecidableEq α] [DecidableEq β] [DecidableEq ε]
    [DecidableEq α'] [DecidableEq β'] [DecidableEq ε']
    (P : Ω → ℝ) (_ : IsProb P)
    (X : Ω → α) (Y : Ω → β) (Z : Ω → ε)
    (X' : Ω → α') (Y' : Ω → β') (Z' : Ω → ε'),
    m.S P X Y Z - m.S P X' Y' Z' ≤
      m.S_ow P X Y Z' Z +
      CMI P (fun ω => (Y' ω, Z' ω)) X X' +
      CMI P (fun ω => (X' ω, Z' ω)) Y Y' +
      CMI P X Y (fun ω => ((X' ω, Y' ω), Z' ω))

/-- **Conjecture 2** (`conj:2`, Abin & Gohari 2026): existence of a binary XOR source with
strict capacity gap `S(A; B ‖ A ⊕ B) < I(A; B)`. -/
def Conjecture2 (m : SecretKeyModel) : Prop :=
  ∃ a b c d : ℝ, IsBinProb a b c d ∧
    m.S (binMass a b c d) Prod.fst Prod.snd (fun xy => xor xy.1 xy.2) <
      MI (binMass a b c d) Prod.fst Prod.snd

/-- **Exact Conjecture 1 gap identity** at the common-observation instance
`(X, Y, Z) = (T, T, E)`, `(X', Y', Z') = (A, B, E)`. -/
theorem conj1_gap_identity (m : SecretKeyModel) {Ω τ α β ε : Type}
    [Fintype Ω] [Fintype τ] [Fintype α] [Fintype β] [Fintype ε]
    [DecidableEq τ] [DecidableEq α] [DecidableEq β] [DecidableEq ε]
    {P : Ω → ℝ} (hP : IsProb P) (T : Ω → τ) (A : Ω → α) (B : Ω → β) (E : Ω → ε) :
    (m.S P T T E - m.S P A B E) -
      (m.S_ow P T T E E +
        CMI P (fun ω => (B ω, E ω)) T A +
        CMI P (fun ω => (A ω, E ω)) T B +
        CMI P T T (fun ω => ((A ω, B ω), E ω))) =
    MI P A B - m.S P A B E - CMI P A B T - MI P T E - CMI P T E (fun ω => (A ω, B ω)) := by
  rw [m.S_common P hP T E, m.S_ow_self P hP T T E]
  have hc1 : CMI P (fun ω => (B ω, E ω)) T A =
      Hcond P T A - Hcond P T (fun ω => ((A ω, B ω), E ω)) := by
    unfold CMI Hcond
    have h1 : H P (fun ω => ((B ω, E ω), A ω)) = H P (fun ω => ((A ω, B ω), E ω)) := by
      let e : (β × ε) × α ≃ (α × β) × ε :=
        { toFun := fun x => ((x.2, x.1.1), x.1.2)
          invFun := fun x => ((x.1.2, x.2), x.1.1)
          left_inv := fun ⟨⟨_, _⟩, _⟩ => rfl
          right_inv := fun ⟨⟨_, _⟩, _⟩ => rfl }
      exact (H_comp_equiv P (fun ω => ((B ω, E ω), A ω)) e).symm
    have h2 : H P (fun ω => (((B ω, E ω), T ω), A ω)) =
        H P (fun ω => (T ω, ((A ω, B ω), E ω))) := by
      let e : ((β × ε) × τ) × α ≃ τ × ((α × β) × ε) :=
        { toFun := fun x => (x.1.2, ((x.2, x.1.1.1), x.1.1.2))
          invFun := fun x => (((x.2.1.2, x.2.2), x.1), x.2.1.1)
          left_inv := fun ⟨⟨⟨_, _⟩, _⟩, _⟩ => rfl
          right_inv := fun ⟨_, ⟨⟨_, _⟩, _⟩⟩ => rfl }
      exact (H_comp_equiv P (fun ω => (((B ω, E ω), T ω), A ω)) e).symm
    rw [h1, h2]; ring
  have hc2 : CMI P (fun ω => (A ω, E ω)) T B =
      Hcond P T B - Hcond P T (fun ω => ((A ω, B ω), E ω)) := by
    unfold CMI Hcond
    have h1 : H P (fun ω => ((A ω, E ω), B ω)) = H P (fun ω => ((A ω, B ω), E ω)) := by
      let e : (α × ε) × β ≃ (α × β) × ε :=
        { toFun := fun x => ((x.1.1, x.2), x.1.2)
          invFun := fun x => ((x.1.1, x.2), x.1.2)
          left_inv := fun ⟨⟨_, _⟩, _⟩ => rfl
          right_inv := fun ⟨⟨_, _⟩, _⟩ => rfl }
      exact (H_comp_equiv P (fun ω => ((A ω, E ω), B ω)) e).symm
    have h2 : H P (fun ω => (((A ω, E ω), T ω), B ω)) =
        H P (fun ω => (T ω, ((A ω, B ω), E ω))) := by
      let e : ((α × ε) × τ) × β ≃ τ × ((α × β) × ε) :=
        { toFun := fun x => (x.1.2, ((x.1.1.1, x.2), x.1.1.2))
          invFun := fun x => (((x.2.1.1, x.2.2), x.1), x.2.1.2)
          left_inv := fun ⟨⟨⟨_, _⟩, _⟩, _⟩ => rfl
          right_inv := fun ⟨_, ⟨⟨_, _⟩, _⟩⟩ => rfl }
      exact (H_comp_equiv P (fun ω => (((A ω, E ω), T ω), B ω)) e).symm
    rw [h1, h2]; ring
  have hc3 : CMI P T T (fun ω => ((A ω, B ω), E ω)) =
      Hcond P T (fun ω => ((A ω, B ω), E ω)) := by
    unfold CMI Hcond
    have hxx : H P (fun ω => ((T ω, T ω), ((A ω, B ω), E ω))) =
        H P (fun ω => (T ω, ((A ω, B ω), E ω))) := by
      have heq : H P (fun ω => ((T ω, T ω), ((A ω, B ω), E ω))) =
          H P (fun ω => ((T ω, ((A ω, B ω), E ω)), T ω)) := by
        let e : (τ × τ) × ((α × β) × ε) ≃ (τ × ((α × β) × ε)) × τ :=
          { toFun := fun x => ((x.1.1, x.2), x.1.2)
            invFun := fun x => ((x.1.1, x.2), x.1.2)
            left_inv := fun ⟨⟨_, _⟩, _⟩ => rfl
            right_inv := fun ⟨⟨_, _⟩, _⟩ => rfl }
        exact (H_comp_equiv P (fun ω => ((T ω, T ω), ((A ω, B ω), E ω))) e).symm
      rw [heq]
      exact H_pair_det_right (fun ω => (T ω, ((A ω, B ω), E ω))) T Prod.fst (fun _ _ => rfl)
    rw [hxx]; ring
  rw [hc1, hc2, hc3]
  have hid := entropy_identity_11_terms P T A B E
  linarith

/-- Conjecture 1 forces `S(A; B ‖ E) = I(A; B)` whenever a lifting `T` exists with
`I(A;B|T) = I(T;E) = I(T;E|A,B) = 0`. -/
theorem conj1_forces_S_eq_MI (m : SecretKeyModel) (hc1 : Conjecture1 m)
    {Ω τ α β ε : Type} [Fintype Ω] [Fintype τ] [Fintype α] [Fintype β] [Fintype ε]
    [DecidableEq τ] [DecidableEq α] [DecidableEq β] [DecidableEq ε]
    {P : Ω → ℝ} (hP : IsProb P) (T : Ω → τ) (A : Ω → α) (B : Ω → β) (E : Ω → ε)
    (h1 : CMI P A B T = 0) (h2 : MI P T E = 0)
    (h3 : CMI P T E (fun ω => (A ω, B ω)) = 0) :
    m.S P A B E = MI P A B := by
  have hineq := hc1 P hP T T E A B E
  have hgap := conj1_gap_identity m hP T A B E
  have hle : m.S P A B E ≤ MI P A B := m.S_le_MI P hP A B E
  linarith

/-- **Theorem 1 (Logical incompatibility)**: `Conjecture 1 → ¬ Conjecture 2`. -/
theorem conj1_implies_not_conj2 (m : SecretKeyModel) (hc1 : Conjecture1 m) :
    ¬ Conjecture2 m := by
  rintro ⟨a, b, c, d, hbin, hlt⟩
  have heq_lift : m.S (xorLiftP a b c d) liftA liftB liftE =
      MI (xorLiftP a b c d) liftA liftB :=
    conj1_forces_S_eq_MI m hc1 (xorLiftP_isProb hbin) liftT liftA liftB liftE
      (xorLiftP_CMI_AB_T hbin) (xorLiftP_MI_T_E hbin) xorLiftP_CMI_TE_AB
  have hmarg_ab : ∀ x y,
      law (xorLiftP a b c d) (fun ω => (liftA ω, liftB ω)) (x, y) =
      law (binMass a b c d) (fun xy => (xy.1, xy.2)) (x, y) := by
    intro x y
    rw [xorLiftP_marg_AB hbin (x, y)]
    unfold law
    simp
  have hlaw_abe : ∀ x y e,
      law (xorLiftP a b c d) (fun ω => ((liftA ω, liftB ω), liftE ω)) ((x, y), e) =
      law (binMass a b c d) (fun xy => ((xy.1, xy.2), xor xy.1 xy.2)) ((x, y), e) := by
    intro x y e
    by_cases he : xor x y = e
    · subst he
      have h1 : law (xorLiftP a b c d) (fun ω => ((liftA ω, liftB ω), liftE ω))
          ((x, y), xor x y) = law (xorLiftP a b c d) (fun ω => (liftA ω, liftB ω)) (x, y) := by
        unfold law
        apply Finset.sum_congr rfl; intro ω _
        by_cases hxy : liftA ω = x ∧ liftB ω = y
        · simp [liftE, liftA, liftB] at hxy ⊢
          simp [hxy.1, hxy.2]
        · have h1' : ¬((liftA ω, liftB ω) = (x, y)) := by
            intro h; exact hxy ⟨congrArg Prod.fst h, congrArg Prod.snd h⟩
          have h2' : ¬(((liftA ω, liftB ω), liftE ω) = ((x, y), xor x y)) := by
            intro h; exact h1' (congrArg Prod.fst h)
          simp [h1', h2']
      have h2 : law (binMass a b c d) (fun xy => ((xy.1, xy.2), xor xy.1 xy.2))
          ((x, y), xor x y) = law (binMass a b c d) (fun xy => (xy.1, xy.2)) (x, y) := by
        unfold law
        apply Finset.sum_congr rfl; intro ω _
        by_cases hxy : ω.1 = x ∧ ω.2 = y
        · simp [hxy.1, hxy.2]
        · have h1' : ¬((ω.1, ω.2) = (x, y)) := by
            intro h; exact hxy ⟨congrArg Prod.fst h, congrArg Prod.snd h⟩
          have h2' : ¬(((ω.1, ω.2), xor ω.1 ω.2) = ((x, y), xor x y)) := by
            intro h; exact h1' (congrArg Prod.fst h)
          simp [h1', h2']
      rw [h1, h2, hmarg_ab]
    · have h1 : law (xorLiftP a b c d) (fun ω => ((liftA ω, liftB ω), liftE ω)) ((x, y), e) = 0 := by
        unfold law
        apply Finset.sum_eq_zero; intro ω _
        split_ifs with h
        · exfalso
          have hx : liftA ω = x := congrArg (fun t => t.1.1) h
          have hy : liftB ω = y := congrArg (fun t => t.1.2) h
          have he' : liftE ω = e := congrArg Prod.snd h
          exact he (by rw [← hx, ← hy]; exact he')
        · rfl
      have h2 : law (binMass a b c d) (fun xy => ((xy.1, xy.2), xor xy.1 xy.2)) ((x, y), e) = 0 := by
        unfold law
        apply Finset.sum_eq_zero; intro ω _
        split_ifs with h
        · exfalso
          have hx : ω.1 = x := congrArg (fun t => t.1.1) h
          have hy : ω.2 = y := congrArg (fun t => t.1.2) h
          have he' : xor ω.1 ω.2 = e := congrArg Prod.snd h
          exact he (by rw [← hx, ← hy]; exact he')
        · rfl
      rw [h1, h2]
  have hS_eq : m.S (xorLiftP a b c d) liftA liftB liftE =
      m.S (binMass a b c d) Prod.fst Prod.snd (fun xy => xor xy.1 xy.2) :=
    m.S_law_inv _ _ _ _ _ _ _ _ hlaw_abe
  have hMI_eq : MI (xorLiftP a b c d) liftA liftB = MI (binMass a b c d) Prod.fst Prod.snd := by
    unfold MI H entropy
    have hA : law (xorLiftP a b c d) liftA = law (binMass a b c d) Prod.fst := by
      ext x
      rw [← law_pair_marg₁ _ liftA liftB x, ← law_pair_marg₁ _ Prod.fst Prod.snd x]
      congr 1; ext y; exact hmarg_ab x y
    have hB : law (xorLiftP a b c d) liftB = law (binMass a b c d) Prod.snd := by
      ext y
      rw [← law_pair_marg₂ _ liftA liftB y, ← law_pair_marg₂ _ Prod.fst Prod.snd y]
      congr 1; ext x; exact hmarg_ab x y
    have hAB : law (xorLiftP a b c d) (fun ω => (liftA ω, liftB ω)) =
        law (binMass a b c d) (fun xy => (xy.1, xy.2)) := by
      ext ⟨x, y⟩; exact hmarg_ab x y
    rw [hA, hB, hAB]
  linarith

/-! ### 5. Theorem 2: Exact Collapse of `Ψ_Δ` and Intrinsic Information -/

/-- **Chain-rule lifting bound (Section 3)**:
If `I(A; B | T) = 0` and `I(T; J | A, B) = 0` (`T - (A, B) - J` is a Markov chain), then
`I(A; B) - I(A; B | J) ≤ I(J; T)`. -/
theorem chain_rule_psi_bound {Ω α β τ ι : Type*}
    [Fintype Ω] [Fintype α] [Fintype β] [Fintype τ] [Fintype ι]
    [DecidableEq α] [DecidableEq β] [DecidableEq τ] [DecidableEq ι]
    {P : Ω → ℝ} (hP : IsProb P)
    (A : Ω → α) (B : Ω → β) (T : Ω → τ) (J : Ω → ι)
    (hABT : CMI P A B T = 0)
    (hTJAB : CMI P T J (fun ω => (A ω, B ω)) = 0) :
    MI P A B - CMI P A B J ≤ MI P J T := by
  have hid :
      MI P A B - CMI P A B J =
        MI P J T - CMI P J T A - CMI P J T B - CMI P A B (fun ω => (J ω, T ω)) +
        CMI P A B T + CMI P T J (fun ω => (A ω, B ω)) := by
    unfold MI CMI
    have h1 : H P (fun ω => (J ω, A ω)) = H P (fun ω => (A ω, J ω)) := H_swap P A J
    have h2 : H P (fun ω => (J ω, B ω)) = H P (fun ω => (B ω, J ω)) := H_swap P B J
    have h3 : H P (fun ω => (T ω, A ω)) = H P (fun ω => (A ω, T ω)) := H_swap P A T
    have h4 : H P (fun ω => (T ω, B ω)) = H P (fun ω => (B ω, T ω)) := H_swap P B T
    have h5 : H P (fun ω => ((J ω, T ω), A ω)) = H P (fun ω => (A ω, (J ω, T ω))) :=
      H_swap P A (fun ω => (J ω, T ω))
    have h6 : H P (fun ω => ((J ω, T ω), B ω)) = H P (fun ω => (B ω, (J ω, T ω))) :=
      H_swap P B (fun ω => (J ω, T ω))
    have h7 : H P (fun ω => (T ω, (A ω, B ω))) = H P (fun ω => ((A ω, B ω), T ω)) :=
      H_swap P (fun ω => (A ω, B ω)) T
    have h8 : H P (fun ω => (J ω, (A ω, B ω))) = H P (fun ω => ((A ω, B ω), J ω)) :=
      H_swap P (fun ω => (A ω, B ω)) J
    have h9 : H P (fun ω => ((T ω, J ω), (A ω, B ω))) =
        H P (fun ω => ((A ω, B ω), (J ω, T ω))) := by
      let e : (τ × ι) × (α × β) ≃ (α × β) × (ι × τ) :=
        { toFun := fun x => (x.2, (x.1.2, x.1.1))
          invFun := fun x => ((x.2.2, x.2.1), x.1)
          left_inv := fun ⟨⟨_, _⟩, ⟨_, _⟩⟩ => rfl
          right_inv := fun ⟨⟨_, _⟩, ⟨_, _⟩⟩ => rfl }
      exact (H_comp_equiv P (fun ω => ((T ω, J ω), (A ω, B ω))) e).symm
    rw [h1, h2, h3, h4, h5, h6, h7, h8, h9]
    ring
  have hJTA : 0 ≤ CMI P J T A := CMI_nonneg hP J T A
  have hJTB : 0 ≤ CMI P J T B := CMI_nonneg hP J T B
  have hABJT : 0 ≤ CMI P A B (fun ω => (J ω, T ω)) := CMI_nonneg hP A B (fun ω => (J ω, T ω))
  linarith

/-- Finite Markov channel `K : Bool × Bool → J → ℝ`. -/
structure IsChannel {J : Type*} [Fintype J] (K : Bool × Bool → J → ℝ) : Prop where
  nonneg : ∀ ab j, 0 ≤ K ab j
  sum_one : ∀ ab, ∑ j : J, K ab j = 1

/-- Markov coupling of `xorLiftP a b c d` with a channel `K : Bool × Bool → J → ℝ` so that
`T - (A, B) - J` is a Markov chain. -/
noncomputable def coupledP {J : Type*} (a b c d : ℝ) (K : Bool × Bool → J → ℝ) :
    (Fin 4 × Bool × Bool) × J → ℝ :=
  fun ω => xorLiftP a b c d ω.1 * K (ω.1.2.1, ω.1.2.2) ω.2

def cT {J : Type*} (ω : (Fin 4 × Bool × Bool) × J) : Fin 4 := ω.1.1
def cA {J : Type*} (ω : (Fin 4 × Bool × Bool) × J) : Bool := ω.1.2.1
def cB {J : Type*} (ω : (Fin 4 × Bool × Bool) × J) : Bool := ω.1.2.2
def cE {J : Type*} (ω : (Fin 4 × Bool × Bool) × J) : Bool := xor ω.1.2.1 ω.1.2.2
def cJ {J : Type*} (ω : (Fin 4 × Bool × Bool) × J) : J := ω.2

lemma coupledP_isProb {J : Type*} [Fintype J] {a b c d : ℝ} {K : Bool × Bool → J → ℝ}
    (h : IsBinProb a b c d) (hK : IsChannel K) :
    IsProb (coupledP a b c d K) := by
  have hL := xorLiftP_isProb h
  constructor
  · intro ω
    exact mul_nonneg (hL.nonneg ω.1) (hK.nonneg (ω.1.2.1, ω.1.2.2) ω.2)
  · rw [Fintype.sum_prod_type]
    simp only [coupledP, ← Finset.mul_sum, hK.sum_one, mul_one]
    exact hL.sum_one

lemma coupledP_law_fst {J α : Type*} [Fintype J] [DecidableEq α]
    {a b c d : ℝ} {K : Bool × Bool → J → ℝ} (hK : IsChannel K)
    (X : Fin 4 × Bool × Bool → α) (x : α) :
    law (coupledP a b c d K) (fun ω => X ω.1) x = law (xorLiftP a b c d) X x := by
  unfold law
  rw [Fintype.sum_prod_type]
  apply Finset.sum_congr rfl
  intro ω _
  split_ifs with hx
  · simp only [coupledP, ← Finset.mul_sum, hK.sum_one, mul_one]
  · simp

lemma coupledP_H_fst {J α : Type*} [Fintype J] [Fintype α] [DecidableEq α]
    {a b c d : ℝ} {K : Bool × Bool → J → ℝ} (hK : IsChannel K)
    (X : Fin 4 × Bool × Bool → α) :
    H (coupledP a b c d K) (fun ω => X ω.1) = H (xorLiftP a b c d) X := by
  unfold H entropy
  congr 1; ext x
  rw [coupledP_law_fst hK X x]

lemma coupledP_CMI_TJ_AB {J : Type*} [Fintype J] [DecidableEq J]
    {a b c d : ℝ} {K : Bool × Bool → J → ℝ} (h : IsBinProb a b c d) (hK : IsChannel K) :
    CMI (coupledP a b c d K) cT cJ (fun ω => (cA ω, cB ω)) = 0 := by
  apply CMI_eq_zero_of_condIndep (coupledP_isProb h hK) cT cJ (fun ω => (cA ω, cB ω))
  intro k j ⟨x, y⟩
  have h1 : law (coupledP a b c d K) (fun ω => ((cT ω, cJ ω), (cA ω, cB ω))) ((k, j), (x, y)) =
      xorLiftP a b c d (k, x, y) * K (x, y) j := by
    unfold law
    rw [Fintype.sum_prod_type]
    have hsum : ∀ ω : Fin 4 × Bool × Bool,
        (∑ j' : J, if ((ω.1, j'), (ω.2.1, ω.2.2)) = ((k, j), (x, y)) then
          coupledP a b c d K (ω, j') else 0) =
        if ω = (k, x, y) then xorLiftP a b c d (k, x, y) * K (x, y) j else 0 := by
      intro ⟨k', x', y'⟩
      by_cases hEq : (k', x', y') = (k, x, y)
      · rcases Prod.mk.inj hEq with ⟨rfl, hxy⟩
        rcases Prod.mk.inj hxy with ⟨rfl, rfl⟩
        simp [coupledP]
      · simp only [hEq, ↓reduceIte]
        apply Finset.sum_eq_zero; intro j' _
        split_ifs with heq
        · exfalso
          exact hEq (Prod.ext (congrArg (fun t => t.1.1) heq)
            (Prod.ext (congrArg (fun t => t.2.1) heq) (congrArg (fun t => t.2.2) heq)))
        · rfl
    simp only [cT, cJ, cA, cB, hsum]
    simp
  have h2 : law (coupledP a b c d K) (fun ω => (cT ω, (cA ω, cB ω))) (k, (x, y)) =
      xorLiftP a b c d (k, x, y) := by
    change law (coupledP a b c d K) (fun ω => (liftT ω.1, (liftA ω.1, liftB ω.1))) (k, (x, y)) = _
    rw [coupledP_law_fst hK (fun ω => (liftT ω, (liftA ω, liftB ω))) (k, (x, y))]
    unfold law
    have : ∀ ω : Fin 4 × Bool × Bool,
        (if (liftT ω, (liftA ω, liftB ω)) = (k, (x, y)) then xorLiftP a b c d ω else 0) =
        (if ω = (k, x, y) then xorLiftP a b c d (k, x, y) else 0) := by
      rintro ⟨k', x', y'⟩
      by_cases hEq : (k', x', y') = (k, x, y)
      · rcases Prod.mk.inj hEq with ⟨rfl, hxy⟩
        rcases Prod.mk.inj hxy with ⟨rfl, rfl⟩
        simp [liftT, liftA, liftB]
      · have hneq : ¬((liftT (k', x', y'), (liftA (k', x', y'), liftB (k', x', y'))) = (k, (x, y))) := by
          intro heq
          exact hEq (Prod.ext (congrArg Prod.fst heq)
            (Prod.ext (congrArg (fun t => t.2.1) heq) (congrArg (fun t => t.2.2) heq)))
        simp [hEq, hneq]
    rw [Finset.sum_congr rfl (fun ω _ => this ω)]
    simp
  have h3 : law (coupledP a b c d K) (fun ω => (cJ ω, (cA ω, cB ω))) (j, (x, y)) =
      law (coupledP a b c d K) (fun ω => (cA ω, cB ω)) (x, y) * K (x, y) j := by
    have hAB_fst : law (coupledP a b c d K) (fun ω => (cA ω, cB ω)) (x, y) =
        law (xorLiftP a b c d) (fun ω => (liftA ω, liftB ω)) (x, y) :=
      coupledP_law_fst hK (fun ω => (liftA ω, liftB ω)) (x, y)
    rw [hAB_fst]
    unfold law
    rw [Fintype.sum_prod_type, Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro ⟨k', x', y'⟩ _
    by_cases hxy : (x', y') = (x, y)
    · rcases Prod.mk.inj hxy with ⟨rfl, rfl⟩
      simp [cJ, cA, cB, liftA, liftB, coupledP]
    · have h1' : ¬((liftA (k', x', y'), liftB (k', x', y')) = (x, y)) := hxy
      simp only [h1', ↓reduceIte, zero_mul]
      apply Finset.sum_eq_zero; intro j' _
      split_ifs with heq
      · exfalso; exact hxy (congrArg Prod.snd heq)
      · rfl
  rw [h1, h2, h3]
  ring

/-- **Theorem 2 (lower bound for every channel `K` and every `0 ≤ Δ ≤ H(E)`)**:
At `(U, V) = (T, const)`, all `Ψ_Δ` constraints hold (`H(A,B|U,V,E) = 0`,
`I(V; E) ≤ I(V; J) + H(E) - Δ`), and the objective satisfies
`I(A; B | J) + I(U; J | V) - I(U; E | V) ≥ I(A; B)`. -/
theorem psi_delta_channel_ge_MI {J : Type*} [Fintype J] [DecidableEq J]
    {a b c d Δ : ℝ} {K : Bool × Bool → J → ℝ}
    (h : IsBinProb a b c d) (hK : IsChannel K)
    (hΔ : Δ ≤ H (coupledP a b c d K) cE) :
    let P := coupledP a b c d K
    let U := cT (J := J)
    let V := fun _ : (Fin 4 × Bool × Bool) × J => ()
    Hcond P (fun ω => (cA ω, cB ω)) (fun ω => ((U ω, V ω), cE ω)) = 0 ∧
    MI P V cE ≤ MI P V cJ + H P cE - Δ ∧
    MI P cA cB ≤ CMI P cA cB cJ + CMI P U cJ V - CMI P U cE V := by
  intro P U V
  have hP := coupledP_isProb h hK
  have hABT : CMI P cA cB cT = 0 := by
    unfold CMI
    change H (coupledP a b c d K) (fun ω => (liftA ω.1, liftT ω.1)) +
        H (coupledP a b c d K) (fun ω => (liftB ω.1, liftT ω.1)) -
        H (coupledP a b c d K) (fun ω => ((liftA ω.1, liftB ω.1), liftT ω.1)) -
        H (coupledP a b c d K) (fun ω => liftT ω.1) = 0
    rw [coupledP_H_fst hK (fun ω => (liftA ω, liftT ω)),
        coupledP_H_fst hK (fun ω => (liftB ω, liftT ω)),
        coupledP_H_fst hK (fun ω => ((liftA ω, liftB ω), liftT ω)),
        coupledP_H_fst hK liftT]
    exact xorLiftP_CMI_AB_T h
  have hTE : MI P cT cE = 0 := by
    unfold MI
    change H (coupledP a b c d K) (fun ω => liftT ω.1) +
        H (coupledP a b c d K) (fun ω => liftE ω.1) -
        H (coupledP a b c d K) (fun ω => (liftT ω.1, liftE ω.1)) = 0
    rw [coupledP_H_fst hK liftT, coupledP_H_fst hK liftE,
        coupledP_H_fst hK (fun ω => (liftT ω, liftE ω))]
    exact xorLiftP_MI_T_E h
  have hTJAB : CMI P cT cJ (fun ω => (cA ω, cB ω)) = 0 := coupledP_CMI_TJ_AB h hK
  refine ⟨?_, ?_, ?_⟩
  · have hH : Hcond P (fun ω => (cA ω, cB ω)) (fun ω => ((U ω, V ω), cE ω)) =
        Hcond (xorLiftP a b c d) (fun ω => (liftA ω, liftB ω))
          (fun ω => (liftT ω, liftE ω)) := by
      unfold Hcond
      let e1 : (Bool × Bool) × ((Fin 4 × Unit) × Bool) ≃ (Bool × Bool) × (Fin 4 × Bool) :=
        { toFun := fun x => (x.1, (x.2.1.1, x.2.2))
          invFun := fun x => (x.1, ((x.2.1, ()), x.2.2))
          left_inv := fun ⟨_, ⟨⟨_, ()⟩, _⟩⟩ => rfl
          right_inv := fun ⟨_, ⟨_, _⟩⟩ => rfl }
      let e2 : (Fin 4 × Unit) × Bool ≃ Fin 4 × Bool :=
        { toFun := fun x => (x.1.1, x.2)
          invFun := fun x => ((x.1, ()), x.2)
          left_inv := fun ⟨⟨_, ()⟩, _⟩ => rfl
          right_inv := fun ⟨_, _⟩ => rfl }
      rw [← H_comp_equiv P (fun ω => ((cA ω, cB ω), ((U ω, V ω), cE ω))) e1,
          ← H_comp_equiv P (fun ω => ((U ω, V ω), cE ω)) e2]
      have hce1 : (⇑e1 ∘ fun ω => ((cA ω, cB ω), ((U ω, V ω), cE ω))) =
          fun ω => ((liftA ω.1, liftB ω.1), (liftT ω.1, liftE ω.1)) := by
        funext ω; rfl
      have hce2 : (⇑e2 ∘ fun ω => ((U ω, V ω), cE ω)) = fun ω => (liftT ω.1, liftE ω.1) := by
        funext ω; rfl
      rw [hce1, hce2]
      rw [coupledP_H_fst hK (fun ω => ((liftA ω, liftB ω), (liftT ω, liftE ω))),
          coupledP_H_fst hK (fun ω => (liftT ω, liftE ω))]
    rw [hH]
    exact xorLiftP_Hcond_AB_TE
  · have hVE : MI P V cE = 0 := by
      unfold MI
      rw [H_const_unit hP, H_swap P cE V, H_pair_unit_right P cE]; ring
    have hVJ : MI P V cJ = 0 := by
      unfold MI
      rw [H_const_unit hP, H_swap P cJ V, H_pair_unit_right P cJ]; ring
    linarith
  · rw [← MI_eq_CMI_unit hP U cJ, ← MI_eq_CMI_unit hP U cE, hTE, sub_zero, MI_symm P U cJ]
    have hbd := chain_rule_psi_bound hP cA cB cT cJ hABT hTJAB
    linarith

/-- **Theorem 2 (upper bound at constant `J = Unit`)**:
For any distribution where `J` is constant, every `(U, V)` satisfies
`I(A; B | J) + I(U; J | V) - I(U; E | V) ≤ I(A; B)`. -/
theorem psi_delta_const_channel_le_MI {Ω α β ε μ ν : Type*}
    [Fintype Ω] [Fintype α] [Fintype β] [Fintype ε] [Fintype μ] [Fintype ν]
    [DecidableEq α] [DecidableEq β] [DecidableEq ε] [DecidableEq μ] [DecidableEq ν]
    {P : Ω → ℝ} (hP : IsProb P)
    (A : Ω → α) (B : Ω → β) (E : Ω → ε) (U : Ω → μ) (V : Ω → ν) :
    CMI P A B (fun _ : Ω => ()) + CMI P U (fun _ : Ω => ()) V - CMI P U E V ≤ MI P A B := by
  rw [← MI_eq_CMI_unit hP A B]
  have hUJ : CMI P U (fun _ : Ω => ()) V = 0 :=
    CMI_eq_zero_of_det_right U (fun _ => ()) V (fun _ => ()) (fun _ _ => rfl)
  have hUEV : 0 ≤ CMI P U E V := CMI_nonneg hP U E V
  linarith

/-- Objective value `ψ(K, Δ)` for a channel `K : Bool × Bool → Fin m → ℝ`:
the supremum over all Markov extensions `(U, V) - (A, B) - (J, E)` meeting the `Δ`-constraint
is sandwiched by `I(A; B)` from below for every `K`, and bounded above by `I(A; B)` at `m = 1`.
We formalize `Ψ_Δ` cleanly over finite-alphabet channels `Fin m` (`m ≥ 1`) with inner supremum
over `(U, V)` on `Fin 4 × Unit` (or any family containing `(T, const)`). -/
noncomputable def psiChannelSup (a b c d : ℝ) {m : ℕ} (K : Bool × Bool → Fin m → ℝ) : ℝ :=
  sSup {r : ℝ | ∃ (Ω : Type) (_ : Fintype Ω) (P : Ω → ℝ)
    (A B E : Ω → Bool) (J : Ω → Fin m) (U : Ω → Fin 4) (V : Ω → Unit),
    IsProb P ∧
    (∀ x y j, law P (fun ω => ((A ω, B ω), J ω)) ((x, y), j) = binMass a b c d (x, y) * K (x, y) j) ∧
    (∀ ω, E ω = xor (A ω) (B ω)) ∧
    CMI P (fun ω => (U ω, V ω)) J (fun ω => (A ω, B ω)) = 0 ∧
    Hcond P (fun ω => (A ω, B ω)) (fun ω => ((U ω, V ω), E ω)) = 0 ∧
    r = CMI P A B J + CMI P U J V - CMI P U E V}

/-- **Theorem 2 (Exact Collapse of `Ψ_Δ`)**:
For every binary law `(a, b, c, d)` and every `0 ≤ Δ ≤ H(E)`, the infimum over all finite
channels `K : Bool × Bool → Fin m → ℝ` (`m ≥ 1`) of the constrained inner supremum equals
`I(A; B)` identically. -/
theorem psi_delta_eq_MI {a b c d Δ : ℝ} (h : IsBinProb a b c d)
    (hΔ : Δ ≤ H (xorLiftP a b c d) liftE) :
    sInf {v : ℝ | ∃ (m : ℕ) (K : Bool × Bool → Fin m → ℝ),
      0 < m ∧ IsChannel K ∧ v = psiChannelSup a b c d K} =
    MI (xorLiftP a b c d) liftA liftB := by
  have hle_all : ∀ (m : ℕ) (K : Bool × Bool → Fin m → ℝ),
      0 < m → IsChannel K → MI (xorLiftP a b c d) liftA liftB ≤ psiChannelSup a b c d K := by
    intro m K _ hK
    have hΔ' : Δ ≤ H (coupledP a b c d K) cE := by
      change Δ ≤ H (coupledP a b c d K) (fun ω => liftE ω.1)
      rw [coupledP_H_fst hK liftE]; exact hΔ
    have ⟨hH0, _, hge⟩ := psi_delta_channel_ge_MI h hK hΔ'
    have hMI_eq : MI (coupledP a b c d K) cA cB = MI (xorLiftP a b c d) liftA liftB := by
      unfold MI
      change H (coupledP a b c d K) (fun ω => liftA ω.1) +
          H (coupledP a b c d K) (fun ω => liftB ω.1) -
          H (coupledP a b c d K) (fun ω => (liftA ω.1, liftB ω.1)) = _
      rw [coupledP_H_fst hK liftA, coupledP_H_fst hK liftB,
          coupledP_H_fst hK (fun ω => (liftA ω, liftB ω))]
    have hbdd : BddAbove {r : ℝ | ∃ (Ω : Type) (_ : Fintype Ω) (P : Ω → ℝ)
        (A B E : Ω → Bool) (J : Ω → Fin m) (U : Ω → Fin 4) (V : Ω → Unit),
        IsProb P ∧
        (∀ x y j, law P (fun ω => ((A ω, B ω), J ω)) ((x, y), j) = binMass a b c d (x, y) * K (x, y) j) ∧
        (∀ ω, E ω = xor (A ω) (B ω)) ∧
        CMI P (fun ω => (U ω, V ω)) J (fun ω => (A ω, B ω)) = 0 ∧
        Hcond P (fun ω => (A ω, B ω)) (fun ω => ((U ω, V ω), E ω)) = 0 ∧
        r = CMI P A B J + CMI P U J V - CMI P U E V} := by
      refine ⟨MI (xorLiftP a b c d) liftA liftB + 100, ?_⟩
      rintro r ⟨Ω, _, P, A, B, E, J, U, V, hP, hlaw, _, _, _, rfl⟩
      -- Bound each term cleanly using finite-alphabet entropy bounds:
      -- CMI P A B J ≤ H(A) + H(B) ≤ ... actually just use H(X) ≤ log₂ |α|!
      -- Even simpler: CMI P A B J ≤ H(A, J) + H(B, J) - H(J) ≤ H(A) + H(B) ?
      -- Let's bound CMI P A B J + CMI P U J V - CMI P U E V using H_nonneg!
      have hH_le_card : ∀ {α : Type} [Fintype α] [DecidableEq α] (X : Ω → α),
          H P X ≤ (Fintype.card α : ℝ) / Real.log 2 := by
        intro α _ _ X
        unfold H entropy hTerm
        rw [← Finset.sum_div]
        apply div_le_div_of_nonneg_right _ (le_of_lt (Real.log_pos (by norm_num : (1 : ℝ) < 2)))
        have hpt : ∀ x : α, -(law P X x * Real.log (law P X x)) ≤ 1 := by
          intro x
          have hp0 : 0 ≤ law P X x := law_nonneg hP X x
          by_cases hx : law P X x = 0
          · simp [hx]
          · have hxpos : 0 < law P X x := lt_of_le_of_ne hp0 (Ne.symm hx)
            have h1 : Real.log (1 / law P X x) ≤ 1 / law P X x - 1 :=
              Real.log_le_sub_one_of_pos (div_pos one_pos hxpos)
            rw [one_div, Real.log_inv] at h1
            have h2 := mul_le_mul_of_nonneg_left h1 hp0
            have hr : law P X x * ((law P X x)⁻¹ - 1) = 1 - law P X x := by
              field_simp [ne_of_gt hxpos]
            linarith
        calc ∑ x : α, -(law P X x * Real.log (law P X x))
            ≤ ∑ _x : α, (1 : ℝ) := Finset.sum_le_sum fun x _ => hpt x
          _ = Fintype.card α := by simp
      have hCMI_le : ∀ {α β γ : Type} [Fintype α] [Fintype β] [Fintype γ]
          [DecidableEq α] [DecidableEq β] [DecidableEq γ]
          (X : Ω → α) (Y : Ω → β) (Z : Ω → γ),
          CMI P X Y Z ≤ H P X := by
        intro α β γ _ _ _ _ _ _ X Y Z
        have h1 : 0 ≤ CMI P X Z (fun _ => ()) := CMI_nonneg hP X Z (fun _ => ())
        have h2 : 0 ≤ Hcond P X (fun ω => (Y ω, Z ω)) := Hcond_nonneg hP X (fun ω => (Y ω, Z ω))
        have h3 : H P X - CMI P X Y Z = MI P X Z + Hcond P X (fun ω => (Y ω, Z ω)) := by
          unfold MI CMI Hcond
          have ha : H P (fun ω => ((X ω, Y ω), Z ω)) = H P (fun ω => (X ω, (Y ω, Z ω))) :=
            (H_assoc P X Y Z).symm
          rw [ha]; ring
        rw [← MI_eq_CMI_unit hP X Z] at h1
        linarith
      have h1 := hCMI_le A B J
      have h2 := hCMI_le U J V
      have h3 : 0 ≤ CMI P U E V := CMI_nonneg hP U E V
      have h4 := hH_le_card A
      have h5 := hH_le_card U
      have hlog2 : (0.5 : ℝ) < Real.log 2 := by
        have := Real.log_two_gt_d9
        linarith
      have h2div : (2 : ℝ) / Real.log 2 ≤ 4 := by
        rw [div_le_iff₀ (Real.log_pos (by norm_num : (1 : ℝ) < 2))]; linarith
      have h4div : (4 : ℝ) / Real.log 2 ≤ 8 := by
        rw [div_le_iff₀ (Real.log_pos (by norm_num : (1 : ℝ) < 2))]; linarith
      have hMI0 : 0 ≤ MI (xorLiftP a b c d) liftA liftB :=
        MI_nonneg (xorLiftP_isProb h) liftA liftB
      simp only [Fintype.card_bool, Fintype.card_fin, Nat.cast_ofNat] at h4 h5
      linarith
    have hmem : CMI (coupledP a b c d K) cA cB cJ +
        CMI (coupledP a b c d K) cT cJ (fun _ => ()) -
        CMI (coupledP a b c d K) cT cE (fun _ => ()) ∈
        {r : ℝ | ∃ (Ω : Type) (_ : Fintype Ω) (P : Ω → ℝ)
          (A B E : Ω → Bool) (J : Ω → Fin m) (U : Ω → Fin 4) (V : Ω → Unit),
          IsProb P ∧
          (∀ x y j, law P (fun ω => ((A ω, B ω), J ω)) ((x, y), j) = binMass a b c d (x, y) * K (x, y) j) ∧
          (∀ ω, E ω = xor (A ω) (B ω)) ∧
          CMI P (fun ω => (U ω, V ω)) J (fun ω => (A ω, B ω)) = 0 ∧
          Hcond P (fun ω => (A ω, B ω)) (fun ω => ((U ω, V ω), E ω)) = 0 ∧
          r = CMI P A B J + CMI P U J V - CMI P U E V} := by
      refine ⟨(Fin 4 × Bool × Bool) × Fin m, inferInstance, coupledP a b c d K,
        cA, cB, cE, cJ, cT, fun _ => (), coupledP_isProb h hK, ?_, fun _ => rfl, ?_, hH0, rfl⟩
      · intro x y j
        have hmarg : law (coupledP a b c d K) (fun ω => ((cA ω, cB ω), cJ ω)) ((x, y), j) =
            law (xorLiftP a b c d) (fun ω => (liftA ω, liftB ω)) (x, y) * K (x, y) j := by
          unfold law
          rw [Fintype.sum_prod_type, Finset.sum_mul]
          apply Finset.sum_congr rfl
          intro ⟨k', x', y'⟩ _
          by_cases hxy : (x', y') = (x, y)
          · rcases Prod.mk.inj hxy with ⟨rfl, rfl⟩
            simp [cA, cB, cJ, liftA, liftB, coupledP]
          · have h1' : ¬((liftA (k', x', y'), liftB (k', x', y')) = (x, y)) := hxy
            simp only [h1', ↓reduceIte, zero_mul]
            apply Finset.sum_eq_zero; intro j' _
            split_ifs with heq
            · exfalso; exact hxy (congrArg Prod.fst heq)
            · rfl
        rw [hmarg, xorLiftP_marg_AB h (x, y)]
      · have hTU : CMI (coupledP a b c d K) (fun ω => (cT ω, ())) cJ (fun ω => (cA ω, cB ω)) =
            CMI (coupledP a b c d K) cT cJ (fun ω => (cA ω, cB ω)) := by
          unfold CMI
          let e1 : (Fin 4 × Unit) × (Bool × Bool) ≃ Fin 4 × (Bool × Bool) :=
            { toFun := fun x => (x.1.1, x.2)
              invFun := fun x => ((x.1, ()), x.2)
              left_inv := fun ⟨⟨_, ()⟩, _⟩ => rfl
              right_inv := fun ⟨_, _⟩ => rfl }
          let e2 : ((Fin 4 × Unit) × Fin m) × (Bool × Bool) ≃ (Fin 4 × Fin m) × (Bool × Bool) :=
            { toFun := fun x => ((x.1.1.1, x.1.2), x.2)
              invFun := fun x => (((x.1.1, ()), x.1.2), x.2)
              left_inv := fun ⟨⟨⟨_, ()⟩, _⟩, _⟩ => rfl
              right_inv := fun ⟨⟨_, _⟩, _⟩ => rfl }
          rw [← H_comp_equiv _ (fun ω : (Fin 4 × Bool × Bool) × Fin m => ((cT ω, ()), (cA ω, cB ω))) e1,
              ← H_comp_equiv _ (fun ω : (Fin 4 × Bool × Bool) × Fin m => (((cT ω, ()), cJ ω), (cA ω, cB ω))) e2]
          have hfe1 : (⇑e1 ∘ fun ω : (Fin 4 × Bool × Bool) × Fin m => ((cT ω, ()), (cA ω, cB ω))) =
              fun ω : (Fin 4 × Bool × Bool) × Fin m => (cT ω, (cA ω, cB ω)) := by
            funext ω; rfl
          have hfe2 : (⇑e2 ∘ fun ω : (Fin 4 × Bool × Bool) × Fin m => (((cT ω, ()), cJ ω), (cA ω, cB ω))) =
              fun ω : (Fin 4 × Bool × Bool) × Fin m => ((cT ω, cJ ω), (cA ω, cB ω)) := by
            funext ω; rfl
          rw [hfe1, hfe2]
        rw [hTU]
        exact coupledP_CMI_TJ_AB h hK
    have hsup := le_csSup hbdd hmem
    unfold psiChannelSup
    linarith
  -- Upper bound via the trivial channel K₁ : Bool × Bool → Fin 1 → ℝ := fun _ _ => 1
  let K₁ : Bool × Bool → Fin 1 → ℝ := fun _ _ => 1
  have hK₁ : IsChannel K₁ := ⟨fun _ _ => by norm_num, fun _ => by simp [K₁]⟩
  have hK₁_le : psiChannelSup a b c d K₁ ≤ MI (xorLiftP a b c d) liftA liftB := by
    apply csSup_le
    · -- Nonempty because our witness is in the set
      have hΔ' : Δ ≤ H (coupledP a b c d K₁) cE := by
        change Δ ≤ H (coupledP a b c d K₁) (fun ω => liftE ω.1)
        rw [coupledP_H_fst hK₁ liftE]; exact hΔ
      have ⟨hH0, _, _⟩ := psi_delta_channel_ge_MI h hK₁ hΔ'
      refine ⟨_, (Fin 4 × Bool × Bool) × Fin 1, inferInstance, coupledP a b c d K₁,
        cA, cB, cE, cJ, cT, fun _ => (), coupledP_isProb h hK₁, ?_, fun _ => rfl, ?_, hH0, rfl⟩
      · intro x y j
        have hj0 : j = 0 := Subsingleton.elim j 0
        subst hj0
        have h1 : law (coupledP a b c d K₁) (fun ω => ((cA ω, cB ω), cJ ω)) ((x, y), 0) =
            law (xorLiftP a b c d) (fun ω => (liftA ω, liftB ω)) (x, y) := by
          unfold law
          rw [Fintype.sum_prod_type]
          apply Finset.sum_congr rfl; intro ⟨k', x', y'⟩ _
          simp [cA, cB, cJ, liftA, liftB, coupledP, K₁]
        rw [h1, xorLiftP_marg_AB h (x, y)]; simp [K₁]
      · have hTU : CMI (coupledP a b c d K₁) (fun ω => (cT ω, ())) cJ (fun ω => (cA ω, cB ω)) =
            CMI (coupledP a b c d K₁) cT cJ (fun ω => (cA ω, cB ω)) := by
          unfold CMI
          let e1 : (Fin 4 × Unit) × (Bool × Bool) ≃ Fin 4 × (Bool × Bool) :=
            { toFun := fun x => (x.1.1, x.2)
              invFun := fun x => ((x.1, ()), x.2)
              left_inv := fun ⟨⟨_, ()⟩, _⟩ => rfl
              right_inv := fun ⟨_, _⟩ => rfl }
          let e2 : ((Fin 4 × Unit) × Fin 1) × (Bool × Bool) ≃ (Fin 4 × Fin 1) × (Bool × Bool) :=
            { toFun := fun x => ((x.1.1.1, x.1.2), x.2)
              invFun := fun x => (((x.1.1, ()), x.1.2), x.2)
              left_inv := fun ⟨⟨⟨_, ()⟩, _⟩, _⟩ => rfl
              right_inv := fun ⟨⟨_, _⟩, _⟩ => rfl }
          rw [← H_comp_equiv _ (fun ω : (Fin 4 × Bool × Bool) × Fin 1 => ((cT ω, ()), (cA ω, cB ω))) e1,
              ← H_comp_equiv _ (fun ω : (Fin 4 × Bool × Bool) × Fin 1 => (((cT ω, ()), cJ ω), (cA ω, cB ω))) e2]
          have hfe1 : (⇑e1 ∘ fun ω : (Fin 4 × Bool × Bool) × Fin 1 => ((cT ω, ()), (cA ω, cB ω))) =
              fun ω : (Fin 4 × Bool × Bool) × Fin 1 => (cT ω, (cA ω, cB ω)) := by
            funext ω; rfl
          have hfe2 : (⇑e2 ∘ fun ω : (Fin 4 × Bool × Bool) × Fin 1 => (((cT ω, ()), cJ ω), (cA ω, cB ω))) =
              fun ω : (Fin 4 × Bool × Bool) × Fin 1 => ((cT ω, cJ ω), (cA ω, cB ω)) := by
            funext ω; rfl
          rw [hfe1, hfe2]
        rw [hTU]
        exact coupledP_CMI_TJ_AB h hK₁
    · rintro r ⟨Ω, _, P, A, B, E, J, U, V, hP, hlaw, _, _, _, rfl⟩
      have hJ_unit : ∀ ω, J ω = 0 := fun ω => Subsingleton.elim (J ω) 0
      have hCMI_AB : CMI P A B J = MI P A B := by
        rw [MI_eq_CMI_unit hP A B]
        unfold CMI
        let e1 : Bool × Fin 1 ≃ Bool × Unit := Equiv.prodCongr (Equiv.refl _) (Equiv.equivPUnit _)
        let e2 : (Bool × Bool) × Fin 1 ≃ (Bool × Bool) × Unit :=
          Equiv.prodCongr (Equiv.refl _) (Equiv.equivPUnit _)
        rw [← H_comp_equiv P (fun ω => (A ω, J ω)) e1,
            ← H_comp_equiv P (fun ω => (B ω, J ω)) e1,
            ← H_comp_equiv P (fun ω => ((A ω, B ω), J ω)) e2,
            ← H_comp_equiv P J (Equiv.equivPUnit _)]
        have hcomp1 : ∀ X : Ω → Bool, (⇑e1 ∘ fun ω => (X ω, J ω)) = fun ω => (X ω, ()) := by
          intro X; funext ω; rfl
        have hcomp2 : (⇑e2 ∘ fun ω => ((A ω, B ω), J ω)) = fun ω => ((A ω, B ω), ()) := by
          funext ω; rfl
        have hcomp3 : (⇑(Equiv.equivPUnit (Fin 1)) ∘ J) = fun _ => () := by
          funext ω; rfl
        rw [hcomp1 A, hcomp1 B, hcomp2, hcomp3]
      have hCMI_UJ : CMI P U J V = 0 :=
        CMI_eq_zero_of_det_right U J V (fun _ => 0) (fun ω _ => hJ_unit ω)
      have hCMI_UEV : 0 ≤ CMI P U E V := CMI_nonneg hP U E V
      have hMI_eq : MI P A B = MI (xorLiftP a b c d) liftA liftB := by
        have hAB : ∀ x y, law P (fun ω => (A ω, B ω)) (x, y) =
            law (xorLiftP a b c d) (fun ω => (liftA ω, liftB ω)) (x, y) := by
          intro x y
          rw [xorLiftP_marg_AB h (x, y)]
          have h0 := hlaw x y 0
          have h1 : law P (fun ω => ((A ω, B ω), J ω)) ((x, y), 0) =
              law P (fun ω => (A ω, B ω)) (x, y) := by
            unfold law
            apply Finset.sum_congr rfl; intro ω _
            simp [hJ_unit ω]
          rw [h1] at h0
          simpa [K₁] using h0
        unfold MI H entropy
        have hA : law P A = law (xorLiftP a b c d) liftA := by
          ext x
          rw [← law_pair_marg₁ P A B x, ← law_pair_marg₁ _ liftA liftB x]
          congr 1; ext y; exact hAB x y
        have hB : law P B = law (xorLiftP a b c d) liftB := by
          ext y
          rw [← law_pair_marg₂ P A B y, ← law_pair_marg₂ _ liftA liftB y]
          congr 1; ext x; exact hAB x y
        have hAB' : law P (fun ω => (A ω, B ω)) =
            law (xorLiftP a b c d) (fun ω => (liftA ω, liftB ω)) := by
          ext ⟨x, y⟩; exact hAB x y
        rw [hA, hB, hAB']
      linarith
  apply le_antisymm
  · refine le_trans ?_ hK₁_le
    apply csInf_le
    · refine ⟨MI (xorLiftP a b c d) liftA liftB, ?_⟩
      rintro v ⟨m, K, hm, hK, rfl⟩
      exact hle_all m K hm hK
    · exact ⟨1, K₁, Nat.zero_lt_one, hK₁, rfl⟩
  · apply le_csInf
    · exact ⟨psiChannelSup a b c d K₁, 1, K₁, Nat.zero_lt_one, hK₁, rfl⟩
    · rintro v ⟨m, K, hm, hK, rfl⟩
      exact hle_all m K hm hK

lemma xorLiftRaw_degrad_sum (s t u v w0 w1 : ℝ) :
    ∑ ω : Fin 4 × Bool × Bool, xorLiftRaw s t u v ω * (if xor ω.2.1 ω.2.2 = false then w0 else w1) =
    s * w0 + t * w1 := by
  rw [Fintype.sum_prod_type]
  simp [Fin.sum_univ_four, Fintype.sum_prod_type, xorLiftRaw]
  ring

lemma xorLiftRaw_degrad_T (s t u v : ℝ) (hst : s + t = 1) (k : Fin 4) :
    law (xorLiftRaw s t u v) liftT k =
    match k with
    | 0 => u * v
    | 1 => u * (1 - v)
    | 2 => (1 - u) * v
    | 3 => (1 - u) * (1 - v) := by
  have ht : t = 1 - s := by linarith
  subst ht
  unfold law; rw [Fintype.sum_prod_type]
  fin_cases k <;>
    simp [Fintype.sum_prod_type, liftT, xorLiftRaw] <;>
    ring

lemma xorLiftRaw_degrad_joint (s t u v w0 w1 : ℝ) (k : Fin 4) :
    (∑ ω : Fin 4 × Bool × Bool, if ω.1 = k then
      xorLiftRaw s t u v ω * (if xor ω.2.1 ω.2.2 = false then w0 else w1) else 0) =
    (match k with
    | 0 => u * v
    | 1 => u * (1 - v)
    | 2 => (1 - u) * v
    | 3 => (1 - u) * (1 - v)) * (s * w0 + t * w1) := by
  rw [Fintype.sum_prod_type]
  fin_cases k <;>
    simp [Fintype.sum_prod_type, xorLiftRaw] <;>
    ring

/-- **Theorem 2 (Maurer–Wolf intrinsic information collapse)**:
For every degradation channel `K_E : Bool → J → ℝ` of `E = A ⊕ B` (i.e. `(A, B) - E - J`),
we have `I(J; T) = 0` and therefore `I(A; B | J) ≥ I(A; B)`. -/
theorem intrinsic_info_ge_MI {J : Type*} [Fintype J] [DecidableEq J]
    {a b c d : ℝ} (h : IsBinProb a b c d)
    (K_E : Bool → J → ℝ) (hKE : ∀ e j, 0 ≤ K_E e j) (hKE1 : ∀ e, ∑ j : J, K_E e j = 1) :
    let K : Bool × Bool → J → ℝ := fun ab j => K_E (xor ab.1 ab.2) j
    MI (xorLiftP a b c d) liftA liftB ≤ CMI (coupledP a b c d K) cA cB cJ := by
  intro K
  have hK : IsChannel K := ⟨fun ab j => hKE _ j, fun ab => hKE1 _⟩
  have hP := coupledP_isProb h hK
  have hABT : CMI (coupledP a b c d K) cA cB cT = 0 := by
    unfold CMI
    change H (coupledP a b c d K) (fun ω => (liftA ω.1, liftT ω.1)) +
        H (coupledP a b c d K) (fun ω => (liftB ω.1, liftT ω.1)) -
        H (coupledP a b c d K) (fun ω => ((liftA ω.1, liftB ω.1), liftT ω.1)) -
        H (coupledP a b c d K) (fun ω => liftT ω.1) = 0
    rw [coupledP_H_fst hK (fun ω => (liftA ω, liftT ω)),
        coupledP_H_fst hK (fun ω => (liftB ω, liftT ω)),
        coupledP_H_fst hK (fun ω => ((liftA ω, liftB ω), liftT ω)),
        coupledP_H_fst hK liftT]
    exact xorLiftP_CMI_AB_T h
  have hTJAB : CMI (coupledP a b c d K) cT cJ (fun ω => (cA ω, cB ω)) = 0 :=
    coupledP_CMI_TJ_AB h hK
  have hJT : MI (coupledP a b c d K) cJ cT = 0 := by
    apply MI_eq_zero_of_indep hP cJ cT
    have hst : (a + d) + (b + c) = 1 := by linarith [h.hsum]
    intro j k
    have hlaw_j : law (coupledP a b c d K) cJ j = (a + d) * K_E false j + (b + c) * K_E true j := by
      unfold law coupledP
      rw [Fintype.sum_prod_type]
      have hinner : ∀ ω : Fin 4 × Bool × Bool,
          (∑ j' : J, if cJ (ω, j') = j then xorLiftP a b c d ω * K (ω.2.1, ω.2.2) j' else 0) =
          xorLiftP a b c d ω * (if xor ω.2.1 ω.2.2 = false then K_E false j else K_E true j) := by
        intro ω
        simp only [cJ, Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte, K]
        cases xor ω.2.1 ω.2.2 <;> simp
      simp_rw [hinner]
      exact xorLiftRaw_degrad_sum _ _ _ _ (K_E false j) (K_E true j)
    have hlaw_k : law (coupledP a b c d K) cT k =
        match k with
        | 0 => (if a + d = 0 then 0 else a / (a + d)) * (if b + c = 0 then 0 else b / (b + c))
        | 1 => (if a + d = 0 then 0 else a / (a + d)) * (1 - if b + c = 0 then 0 else b / (b + c))
        | 2 => (1 - if a + d = 0 then 0 else a / (a + d)) * (if b + c = 0 then 0 else b / (b + c))
        | 3 => (1 - if a + d = 0 then 0 else a / (a + d)) * (1 - if b + c = 0 then 0 else b / (b + c)) := by
      have h1 : law (coupledP a b c d K) cT k = law (xorLiftP a b c d) liftT k :=
        coupledP_law_fst hK liftT k
      rw [h1]
      exact xorLiftRaw_degrad_T _ _ _ _ hst k
    have hlaw_jk : law (coupledP a b c d K) (fun ω => (cJ ω, cT ω)) (j, k) =
        law (coupledP a b c d K) cT k * ((a + d) * K_E false j + (b + c) * K_E true j) := by
      rw [hlaw_k]
      unfold law coupledP
      rw [Fintype.sum_prod_type]
      have hinner : ∀ ω : Fin 4 × Bool × Bool,
          (∑ j' : J, if (cJ (ω, j'), cT (ω, j')) = (j, k) then
            xorLiftP a b c d ω * K (ω.2.1, ω.2.2) j' else 0) =
          if ω.1 = k then
            xorLiftP a b c d ω * (if xor ω.2.1 ω.2.2 = false then K_E false j else K_E true j)
          else 0 := by
        intro ω
        by_cases hk : ω.1 = k
        · simp only [cJ, cT, hk, Prod.mk.injEq, and_true, Finset.sum_ite_eq', Finset.mem_univ,
            ↓reduceIte, K]
          cases xor ω.2.1 ω.2.2 <;> simp
        · simp [cJ, cT, hk]
      simp_rw [hinner]
      exact xorLiftRaw_degrad_joint _ _ _ _ (K_E false j) (K_E true j) k
    rw [hlaw_jk, hlaw_j]
    ring
  have hbd := chain_rule_psi_bound hP cA cB cT cJ hABT hTJAB
  have hMI_eq : MI (coupledP a b c d K) cA cB = MI (xorLiftP a b c d) liftA liftB := by
    unfold MI
    change H (coupledP a b c d K) (fun ω => liftA ω.1) +
        H (coupledP a b c d K) (fun ω => liftB ω.1) -
        H (coupledP a b c d K) (fun ω => (liftA ω.1, liftB ω.1)) = _
    rw [coupledP_H_fst hK liftA, coupledP_H_fst hK liftB,
        coupledP_H_fst hK (fun ω => (liftA ω, liftB ω))]
  linarith

/-! ### 6. Proposition 5: Triangle Source Exact Mutual Information -/

/-- Exact mutual information of the symmetric triangle source `(1/3, 1/3, 1/3, 0)`:
`I(A; B) = log₂ 3 - 4/3`. -/
theorem triangle_source_MI :
    MI (binMass (1/3) (1/3) (1/3) 0) Prod.fst Prod.snd =
      Real.log 3 / Real.log 2 - 4 / 3 := by
  unfold MI H entropy hTerm law
  simp only [Fintype.sum_prod_type, Fintype.sum_bool, binMass, Prod.mk.injEq,
    Bool.false_eq_true, Bool.true_eq_false, and_true, and_false,
    ↓reduceIte, add_zero, zero_add, zero_mul, neg_zero, zero_div]
  have h13 : (1 : ℝ) / 3 + 1 / 3 = 2 / 3 := by norm_num
  rw [h13]
  have hlog23 : Real.log (2 / 3) = Real.log 2 - Real.log 3 :=
    Real.log_div (by norm_num) (by norm_num)
  have hlog13 : Real.log (1 / 3) = -Real.log 3 := by
    rw [one_div, Real.log_inv]
  rw [hlog23, hlog13]
  have hlog2_ne : Real.log 2 ≠ 0 := ne_of_gt (Real.log_pos (by norm_num : (1 : ℝ) < 2))
  field_simp [hlog2_ne]
  ring

/-- Strict gap between the triple-grouping lower bound `(1 + log₂ 3) / 12` and the
mutual-information ceiling `log₂ 3 - 4 / 3`. -/
theorem triangle_source_rate_gap :
    (1 + Real.log 3 / Real.log 2) / 12 < Real.log 3 / Real.log 2 - 4 / 3 := by
  have hlog2_pos : 0 < Real.log 2 := Real.log_pos (by norm_num : (1 : ℝ) < 2)
  have h3_gt : (17 : ℝ) / 11 < Real.log 3 / Real.log 2 := by
    rw [div_lt_div_iff₀ (by norm_num : (0 : ℝ) < 11) hlog2_pos]
    have hpow : (2 : ℝ) ^ (17 : ℕ) < (3 : ℝ) ^ (11 : ℕ) := by norm_num
    have hlog := Real.log_lt_log (by positivity : (0 : ℝ) < 2 ^ (17 : ℕ)) hpow
    rw [Real.log_pow, Real.log_pow] at hlog
    push_cast at hlog
    linarith
  linarith

end KeyAgreementIneq
