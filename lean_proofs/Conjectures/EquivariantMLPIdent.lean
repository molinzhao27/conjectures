import Mathlib

/-!
# Resolution of the Biased-MLP Weak Identifiability Conjecture
(Shahverdi, Marchetti, Bökman, and Kohn, arXiv:2601.21645, Section 5.1, Conjecture 1 `conj:main`)

We formalize in Lean 4 + Mathlib (complete proofs, standard axioms only) the mathematical cores of
both the universal obstructions and the affirmative sigmoidal classification:

1. **Theorem 1 (Universal failure under positive hidden widths `d_i ≥ 1`)**:
   - `pos_width_constant_nonidentifiable`: For EVERY origin-passing activation `σ : ℝ → ℝ`
     (`σ 0 = 0`), EVERY positive hidden width `m ≥ 1`, and EVERY constant `c : ℝ`, the zero-hidden
     tuples `(W₁ = 0, b₁ = 0, W₂ = 0, c)` and `(W₁ = 0, b₁ = 0, W₂ = e₀, c)` both realize `x ↦ c`,
     yet no linear latent symmetry `k : (Fin m → ℝ) ≃ₗ[ℝ] (Fin m → ℝ)` relates their readout maps.
2. **Theorem 2 & Applications (Identity-realization obstruction for ReLU, exact GELU, polynomials)**:
   - `linear_symmetry_preserves_output_bias`: Any linear latent symmetry preserves the output
     bias `c' = c`.
   - `identity_realization_nonidentifiable`: Whenever ANY activation `σ` realizes `f(x) = x` at
     depth `2` and width `m`, necessarily `m ≥ 1`, the translation family
     `(b₁, c) ↦ (b₁ - t W₁, c + t)` realizes `f(x) = x` for every `t : ℝ`, and distinct `t`
     yield distinct linear-symmetry orbits.
   - `odd_part_identity_mlp`: Whenever `σ z - σ (-z) = z` for all `z`, the explicit width-2
     family `f_t(x) = σ(x - t) - σ(t - x) + t` realizes `x` identically with distinct orbits
     for distinct `t`.
   - `relu_refutes_weak_ident`: ReLU `max z 0` satisfies `relu 0 = 0` and `relu z - relu (-z) = z`,
     cannot realize `x` at width `1` (`relu_width1_cannot_realize_id`), and fails at minimal width `2`.
   - `exactGelu_refutes_weak_ident`: Exact Gaussian GELU
     `exactGelu z = z * (1/2 + (√(2π))⁻¹ ∫₀^z exp(-t²/2) dt)` satisfies `exactGelu 0 = 0` and
     `exactGelu z - exactGelu (-z) = z`, refuting weak identifiability at width `2`.
   - `poly_forward_diff_realizes_id`: Every polynomial activation `p` of degree `d ≥ 1` realizes
     `x ↦ x` at width `d` via the `(d - 1)`-st forward difference, triggering Theorem 2.
3. **Theorem 4 Core (Affirmative zero-width constants, centered logistic, and exact `tanh` recovery)**:
   - `zero_width_constant_identifiable`: Allowing hidden width `m = 0` makes every constant
     `x ↦ c` uniquely identifiable.
   - `centered_logistic_eq_half_tanh` & `plain_logistic_nonidentifiable_linear`: Centered logistic
     equals `(1/2) tanh(x/2)` (origin-passing), whereas plain logistic `s(x) = (1 + e^{-x})⁻¹`
     violates `s(0) = 0` and fails under linear latent actions via `s(x) = 1 - s(-x)`.
   - `tanh_width1_identifiable` & `centered_logistic_width1_identifiable`: Complete self-contained
     proof that any two regular (`w ≠ 0, a ≠ 0`) width-1 `tanh` (or centered-logistic) MLPs
     realizing the same function on `ℝ` are related by a UNIQUE sign `s ∈ {+1, -1}`
     (`w' = s w, b' = s b, a' = s a, c' = c`).

Machine-checked vs prose-only:
- Machine-checked: Complete Theorem 1, complete Theorem 2 (and width-2 ReLU, exact Gaussian GELU,
  and general degree-`d` polynomial realizations), plain-logistic refutation, zero-width constant
  identifiability, and complete width-1 `tanh` and centered-logistic identifiability.
- Prose-only (in `solutions/equivariant_mlp_ident/proof.pdf`): Multi-layer arbitrary-width
  Vlačić–Bölcskei (2021) regular-network reduction for Theorem 4, and the BV-sigmoidal density
  transfer (Vlačić–Bölcskei 2022) in Theorem 5.
-/

open BigOperators Finset Filter Topology

namespace EquivariantMLPIdent

/-! ### Depth-2 Scalar-Endpoint MLP Definitions and Linear Latent Symmetries -/

/-- Hidden layer map `f₁(x)_i = σ(W₁_i x + b₁_i)` for a depth-2 width-`m` MLP. -/
def layer1 (σ : ℝ → ℝ) {m : ℕ} (W₁ b₁ : Fin m → ℝ) (x : ℝ) : Fin m → ℝ :=
  fun i => σ (W₁ i * x + b₁ i)

/-- Affine readout map `f₂(z) = ∑_i W₂_i z_i + c`. -/
def layer2 {m : ℕ} (W₂ : Fin m → ℝ) (c : ℝ) (z : Fin m → ℝ) : ℝ :=
  (∑ i : Fin m, W₂ i * z i) + c

/-- End-to-end depth-2 width-`m` MLP function `f(x) = f₂(f₁(x))`. -/
def mlp2 (σ : ℝ → ℝ) {m : ℕ} (W₁ b₁ W₂ : Fin m → ℝ) (c : ℝ) (x : ℝ) : ℝ :=
  layer2 W₂ c (layer1 σ W₁ b₁ x)

/-- Two depth-2 width-`m` parameter tuples are related by a linear latent symmetry
`k : (Fin m → ℝ) ≃ₗ[ℝ] (Fin m → ℝ)` if `f₁' = k ∘ f₁` and `f₂' = f₂ ∘ k⁻¹` on the full
ambient domains. -/
def LinearSymRelated (σ : ℝ → ℝ) {m : ℕ}
    (W₁ b₁ W₂ : Fin m → ℝ) (c : ℝ)
    (W₁' b₁' W₂' : Fin m → ℝ) (c' : ℝ)
    (k : (Fin m → ℝ) ≃ₗ[ℝ] (Fin m → ℝ)) : Prop :=
  (∀ x : ℝ, layer1 σ W₁' b₁' x = k (layer1 σ W₁ b₁ x)) ∧
  (∀ z : Fin m → ℝ, layer2 W₂' c' z = layer2 W₂ c (k.symm z))

/-! ### 1. Theorem 1: Universal Failure Under Positive Hidden Widths -/

/-- **Theorem 1 (Positive-hidden-width non-identifiability for every origin-passing `σ`)**:
For EVERY `σ : ℝ → ℝ` with `σ 0 = 0`, EVERY positive width `m ≥ 1`, and EVERY constant `c : ℝ`,
the two tuples `(0, 0, 0, c)` and `(0, 0, e₀, c)` both realize `x ↦ c`, yet NO linear latent
symmetry `k : (Fin m → ℝ) ≃ₗ[ℝ] (Fin m → ℝ)` relates them. -/
theorem pos_width_constant_nonidentifiable (σ : ℝ → ℝ) (hσ0 : σ 0 = 0)
    {m : ℕ} (hm : 1 ≤ m) (c : ℝ) :
    let z0 : Fin m → ℝ := fun _ => 0
    let e0 : Fin m → ℝ := fun i => if i = ⟨0, by omega⟩ then 1 else 0
    (∀ x : ℝ, mlp2 σ z0 z0 z0 c x = c) ∧
    (∀ x : ℝ, mlp2 σ z0 z0 e0 c x = c) ∧
    (∀ k : (Fin m → ℝ) ≃ₗ[ℝ] (Fin m → ℝ), ¬ LinearSymRelated σ z0 z0 z0 c z0 z0 e0 c k) := by
  intro z0 e0
  refine ⟨?_, ?_, ?_⟩
  · intro x
    simp [mlp2, layer2, layer1, z0]
  · intro x
    simp [mlp2, layer2, layer1, z0, hσ0]
  · intro k hrel
    have h2 := hrel.2 e0
    have hlhs : layer2 e0 c e0 = c + 1 := by
      unfold layer2
      have hsum : ∑ i : Fin m, e0 i * e0 i = 1 := by
        have hterm : ∀ i : Fin m, e0 i * e0 i = if i = ⟨0, by omega⟩ then 1 else 0 := by
          intro i; dsimp only [e0]; split_ifs <;> norm_num
        simp [hterm]
      linarith
    have hrhs : layer2 z0 c (k.symm e0) = c := by
      simp [layer2, z0]
    linarith

/-! ### 2. Output-Bias Invariance, Theorem 2, ReLU, Exact GELU, and Polynomials -/

/-- **Output-bias invariance lemma**: Any linear latent symmetry leaves the readout bias
strictly invariant (`c' = c`), by evaluation at `z = 0`. -/
theorem linear_symmetry_preserves_output_bias (σ : ℝ → ℝ) {m : ℕ}
    {W₁ b₁ W₂ W₁' b₁' W₂' : Fin m → ℝ} {c c' : ℝ}
    {k : (Fin m → ℝ) ≃ₗ[ℝ] (Fin m → ℝ)}
    (h : LinearSymRelated σ W₁ b₁ W₂ c W₁' b₁' W₂' c' k) :
    c' = c := by
  have h0 := h.2 0
  simp only [map_zero, layer2, Pi.zero_apply, mul_zero, Finset.sum_const_zero, zero_add] at h0
  exact h0

/-- **Theorem 2 (Identity-realization obstruction for any activation `σ`)**:
If a depth-2 width-`m` MLP with activation `σ` realizes `f(x) = x`, then:
1. `m ≥ 1` (width 0 realizes only constants);
2. For every `t : ℝ`, shifting `(b₁, c) ↦ (b₁ - t W₁, c + t)` still realizes `f(x) = x`;
3. For every `t₁ ≠ t₂`, no linear latent symmetry relates the `t₁`-shift to the `t₂`-shift. -/
theorem identity_realization_nonidentifiable (σ : ℝ → ℝ) {m : ℕ}
    (W₁ b₁ W₂ : Fin m → ℝ) (c : ℝ)
    (hid : ∀ x : ℝ, mlp2 σ W₁ b₁ W₂ c x = x) :
    1 ≤ m ∧
    (∀ t x : ℝ, mlp2 σ W₁ (fun i => b₁ i - t * W₁ i) W₂ (c + t) x = x) ∧
    (∀ t₁ t₂ : ℝ, t₁ ≠ t₂ →
      ∀ k : (Fin m → ℝ) ≃ₗ[ℝ] (Fin m → ℝ),
        ¬ LinearSymRelated σ
            W₁ (fun i => b₁ i - t₁ * W₁ i) W₂ (c + t₁)
            W₁ (fun i => b₁ i - t₂ * W₁ i) W₂ (c + t₂) k) := by
  refine ⟨?_, ?_, ?_⟩
  · by_contra hm0
    have hm : m = 0 := by omega
    subst hm
    have h0 := hid 0
    have h1 := hid 1
    simp [mlp2, layer2] at h0 h1
    linarith
  · intro t x
    have hshift := hid (x - t)
    unfold mlp2 layer2 layer1 at hshift ⊢
    have harg : ∀ i : Fin m, W₁ i * x + (b₁ i - t * W₁ i) = W₁ i * (x - t) + b₁ i := by
      intro i; ring
    simp_rw [harg]
    linarith
  · intro t₁ t₂ hne k hrel
    have hbias := linear_symmetry_preserves_output_bias σ hrel
    exact hne (by linarith)

/-- General odd-part bypass identity: if `σ z - σ (-z) = z` for all `z : ℝ`, then the explicit
width-2 family `f_t(x) = σ(x - t) - σ(t - x) + t` realizes `x ↦ x` identically for every `t`,
producing a continuum of distinct linear-symmetry orbits. -/
theorem odd_part_identity_mlp (σ : ℝ → ℝ) (hodd : ∀ z : ℝ, σ z - σ (-z) = z) :
    let W₁ : Fin 2 → ℝ := ![1, -1]
    let W₂ : Fin 2 → ℝ := ![1, -1]
    (∀ t x : ℝ, mlp2 σ W₁ ![-t, t] W₂ t x = x) ∧
    (∀ t₁ t₂ : ℝ, t₁ ≠ t₂ →
      ∀ k : (Fin 2 → ℝ) ≃ₗ[ℝ] (Fin 2 → ℝ),
        ¬ LinearSymRelated σ W₁ ![-t₁, t₁] W₂ t₁ W₁ ![-t₂, t₂] W₂ t₂ k) := by
  intro W₁ W₂
  have hreal : ∀ t x : ℝ, mlp2 σ W₁ ![-t, t] W₂ t x = x := by
    intro t x
    simp only [mlp2, layer2, layer1, W₁, W₂, Fin.sum_univ_two, Matrix.cons_val_zero,
      Matrix.cons_val_one, one_mul, neg_mul]
    have h1 : x + -t = x - t := by ring
    have h2 : -x + t = -(x - t) := by ring
    rw [h1, h2]
    linarith [hodd (x - t)]
  refine ⟨hreal, fun t₁ t₂ hne k hrel => ?_⟩
  have hbias := linear_symmetry_preserves_output_bias σ hrel
  exact hne hbias.symm

/-- ReLU activation `max z 0`. -/
def relu (z : ℝ) : ℝ := max z 0

lemma relu_zero : relu 0 = 0 := by simp [relu]

lemma relu_odd_part (z : ℝ) : relu z - relu (-z) = z := by
  unfold relu
  rcases le_total z 0 with h | h
  · rw [max_eq_right h, max_eq_left (by linarith)]; ring
  · rw [max_eq_left h, max_eq_right (by linarith)]; ring

/-- Width-1 ReLU cannot realize `x ↦ x` (so width `2` is minimal). -/
theorem relu_width1_cannot_realize_id (W₁ b₁ W₂ : Fin 1 → ℝ) (c : ℝ) :
    ¬ (∀ x : ℝ, mlp2 relu W₁ b₁ W₂ c x = x) := by
  intro h
  set w := W₁ 0
  set b := b₁ 0
  set a := W₂ 0
  have heval : ∀ x : ℝ, a * max (w * x + b) 0 + c = x := by
    intro x
    have hx := h x
    simpa [mlp2, layer2, layer1, relu, w, b, a] using hx
  by_cases hw : w = 0
  · have h0 := heval 0
    have h1 := heval 1
    simp [hw] at h0 h1
    linarith
  · -- Pick two distinct points on the flat zero half-line `w * x + b ≤ 0`
    let x₁ : ℝ := (-b - 1) / w
    let x₂ : ℝ := (-b - 2) / w
    have hx₁ : w * x₁ + b = -1 := by
      dsimp only [x₁]; field_simp [hw]; ring
    have hx₂ : w * x₂ + b = -2 := by
      dsimp only [x₂]; field_simp [hw]; ring
    have e1 := heval x₁
    have e2 := heval x₂
    rw [hx₁, max_eq_right (by norm_num : (-1 : ℝ) ≤ 0)] at e1
    rw [hx₂, max_eq_right (by norm_num : (-2 : ℝ) ≤ 0)] at e2
    have hxeq : x₁ = x₂ := by linarith
    dsimp only [x₁, x₂] at hxeq
    have hmul := congrArg (fun u => w * u) hxeq
    field_simp [hw] at hmul
    linarith

/-- **ReLU refutation of weak identifiability** at minimal width `2`. -/
theorem relu_refutes_weak_ident :
    relu 0 = 0 ∧
    (∀ t x : ℝ, mlp2 relu ![1, -1] ![-t, t] ![1, -1] t x = x) ∧
    (∀ t₁ t₂ : ℝ, t₁ ≠ t₂ →
      ∀ k : (Fin 2 → ℝ) ≃ₗ[ℝ] (Fin 2 → ℝ),
        ¬ LinearSymRelated relu ![1, -1] ![-t₁, t₁] ![1, -1] t₁ ![1, -1] ![-t₂, t₂] ![1, -1] t₂ k) :=
  ⟨relu_zero, odd_part_identity_mlp relu relu_odd_part⟩

/-- Exact standard normal CDF `Φ(z) = 1/2 + (√(2π))⁻¹ ∫₀^z exp(-t²/2) dt`. -/
noncomputable def normalCDF (z : ℝ) : ℝ :=
  1 / 2 + (Real.sqrt (2 * Real.pi))⁻¹ * ∫ t in (0 : ℝ)..z, Real.exp (-(t ^ 2) / 2)

/-- Exact Gaussian GELU activation `GELU(z) = z Φ(z)`. -/
noncomputable def exactGelu (z : ℝ) : ℝ :=
  z * normalCDF z

lemma normalCDF_add_neg (z : ℝ) : normalCDF z + normalCDF (-z) = 1 := by
  unfold normalCDF
  have hneg : ∫ t in (0 : ℝ)..(-z), Real.exp (-(t ^ 2) / 2) =
      -(∫ t in (0 : ℝ)..z, Real.exp (-(t ^ 2) / 2)) := by
    have hcomp := intervalIntegral.integral_comp_neg (a := (0 : ℝ)) (b := z)
      (f := fun t => Real.exp (-(t ^ 2) / 2))
    simp only [neg_zero, neg_sq] at hcomp
    rw [intervalIntegral.integral_symm (-z) 0, ← hcomp]
  rw [hneg]
  ring

lemma exactGelu_zero : exactGelu 0 = 0 := by simp [exactGelu]

lemma exactGelu_odd_part (z : ℝ) : exactGelu z - exactGelu (-z) = z := by
  unfold exactGelu
  have h := normalCDF_add_neg z
  calc z * normalCDF z - (-z) * normalCDF (-z)
      = z * (normalCDF z + normalCDF (-z)) := by ring
    _ = z * 1 := by rw [h]
    _ = z := mul_one z

/-- **Exact Gaussian GELU refutation of weak identifiability** at width `2`. -/
theorem exactGelu_refutes_weak_ident :
    exactGelu 0 = 0 ∧
    (∀ t x : ℝ, mlp2 exactGelu ![1, -1] ![-t, t] ![1, -1] t x = x) ∧
    (∀ t₁ t₂ : ℝ, t₁ ≠ t₂ →
      ∀ k : (Fin 2 → ℝ) ≃ₗ[ℝ] (Fin 2 → ℝ),
        ¬ LinearSymRelated exactGelu ![1, -1] ![-t₁, t₁] ![1, -1] t₁ ![1, -1] ![-t₂, t₂] ![1, -1] t₂ k) :=
  ⟨exactGelu_zero, odd_part_identity_mlp exactGelu exactGelu_odd_part⟩

/-- Iterated forward difference `(Δ^k f)(x)` defined recursively by
`Δ^0 f = f`, `Δ^{k+1} f(x) = Δ^k f(x + 1) - Δ^k f(x)`. -/
def fwdDiff : ℕ → (ℝ → ℝ) → ℝ → ℝ
  | 0, f, x => f x
  | k + 1, f, x => fwdDiff k f (x + 1) - fwdDiff k f x

/-- Closed-form binomial expansion of `fwdDiff k f x` as a linear combination of `k + 1` shifts
`f (x + j)` (`j ∈ Fin (k + 1)`). -/
lemma fwdDiff_eq_sum (k : ℕ) (f : ℝ → ℝ) (x : ℝ) :
    fwdDiff k f x =
      ∑ j : Fin (k + 1), ((-1 : ℝ) ^ (k - j.val) * (Nat.choose k j.val : ℝ)) * f (x + (j.val : ℝ)) := by
  induction k generalizing x with
  | zero => simp [fwdDiff]
  | succ k ih =>
    simp only [fwdDiff, ih]
    have hL : ∀ n (g : ℕ → ℝ), (∑ j : Fin n, g j.val) = ∑ j ∈ Finset.range n, g j :=
      fun n g => Fin.sum_univ_eq_sum_range g n
    rw [hL (k + 1) (fun j => ((-1 : ℝ) ^ (k - j) * (Nat.choose k j : ℝ)) * f (x + 1 + (j : ℝ))),
        hL (k + 1) (fun j => ((-1 : ℝ) ^ (k - j) * (Nat.choose k j : ℝ)) * f (x + (j : ℝ))),
        hL (k + 2) (fun j => ((-1 : ℝ) ^ (k + 1 - j) * (Nat.choose (k + 1) j : ℝ)) * f (x + (j : ℝ)))]
    rw [Finset.sum_range_succ' _ (k + 1)]
    have hshift : ∑ j ∈ Finset.range (k + 1),
        ((-1 : ℝ) ^ (k - j) * (Nat.choose k j : ℝ)) * f (x + 1 + (j : ℝ)) =
        ∑ j ∈ Finset.range (k + 1),
          ((-1 : ℝ) ^ (k - j) * (Nat.choose k j : ℝ)) * f (x + ((j + 1 : ℕ) : ℝ)) := by
      refine Finset.sum_congr rfl fun j _ => ?_
      push_cast; ring_nf
    rw [hshift]
    rw [Finset.sum_range_succ _ k]
    rw [Finset.sum_range_succ' _ k]
    rw [Finset.sum_range_succ _ k]
    simp only [Nat.choose_self, Nat.choose_zero_right, Nat.sub_self, Nat.sub_zero, pow_zero,
      Nat.cast_one, one_mul]
    -- Peel off the top term j = k and bottom term j = 0, then combine interior terms j + 1 ∈ [1..k]
    have hinter : ∑ j ∈ Finset.range k,
        ((-1 : ℝ) ^ (k - j) * (Nat.choose k j : ℝ)) * f (x + ((j + 1 : ℕ) : ℝ)) -
      ∑ j ∈ Finset.range k,
        ((-1 : ℝ) ^ (k - (j + 1)) * (Nat.choose k (j + 1) : ℝ)) * f (x + ((j + 1 : ℕ) : ℝ)) =
      ∑ j ∈ Finset.range k,
        ((-1 : ℝ) ^ (k + 1 - (j + 1)) * (Nat.choose (k + 1) (j + 1) : ℝ)) * f (x + ((j + 1 : ℕ) : ℝ)) := by
      rw [← Finset.sum_sub_distrib]
      refine Finset.sum_congr rfl fun j hj => ?_
      have hjk : j < k := Finset.mem_range.mp hj
      have hsub1 : k + 1 - (j + 1) = k - j := by omega
      have hsub2 : k - j = (k - (j + 1)) + 1 := by omega
      rw [hsub1, Nat.choose_succ_succ]
      push_cast
      rw [hsub2, pow_succ]
      ring
    have hpow0 : (-1 : ℝ) ^ (k + 1) = -((-1 : ℝ) ^ k) := by rw [pow_succ]; ring
    rw [hpow0]
    have hlast : k + 1 - (k + 1) = 0 := by omega
    linarith [hinter]

/-- Forward difference of a polynomial `P : Polynomial ℝ` lowers its degree by 1 and scales
its leading coefficient by `P.natDegree` (for `P.natDegree ≥ 1`). -/
lemma fwdDiff_one_poly (P : Polynomial ℝ) (hdeg : 1 ≤ P.natDegree) :
    ∃ Q : Polynomial ℝ,
      (∀ x : ℝ, fwdDiff 1 (fun t => P.eval t) x = Q.eval x) ∧
      Q.natDegree = P.natDegree - 1 ∧
      Q.leadingCoeff = (P.natDegree : ℝ) * P.leadingCoeff := by
  let Q : Polynomial ℝ := P.comp (Polynomial.X + Polynomial.C 1) - P
  set d := P.natDegree with hd_def
  have hPne : P ≠ 0 := by
    intro h0
    have : d = 0 := by rw [hd_def, h0, Polynomial.natDegree_zero]
    omega
  have hcoeff_d_gt : ∀ m > d - 1, Q.coeff m = 0 := by
    intro m hm
    dsimp only [Q]
    rw [Polynomial.coeff_sub]
    by_cases hmd : m = d
    · subst hmd
      have hcomp_d : (P.comp (Polynomial.X + Polynomial.C 1)).coeff d = P.coeff d := by
        rw [Polynomial.comp_eq_sum_left, Polynomial.sum_def, Polynomial.finset_sum_coeff]
        rw [Finset.sum_eq_single d]
        · rw [Polynomial.coeff_C_mul]
          have hmon : ((Polynomial.X + Polynomial.C (1 : ℝ)) ^ d).Monic :=
            (Polynomial.monic_X_add_C 1).pow d
          have hnd : ((Polynomial.X + Polynomial.C (1 : ℝ)) ^ d).natDegree = d := by
            rw [Polynomial.natDegree_pow, Polynomial.natDegree_X_add_C, mul_one]
          have hcoeff1 : ((Polynomial.X + Polynomial.C (1 : ℝ)) ^ d).coeff d = 1 := by
            have hlc := hmon.leadingCoeff
            rw [Polynomial.leadingCoeff, hnd] at hlc
            exact hlc
          rw [hcoeff1, mul_one]
        · intro b hb hbd
          rw [Polynomial.coeff_C_mul]
          have hlt : b < d := lt_of_le_of_ne (Polynomial.le_natDegree_of_mem_supp b hb) hbd
          have h0 : ((Polynomial.X + Polynomial.C (1 : ℝ)) ^ b).coeff d = 0 := by
            apply Polynomial.coeff_eq_zero_of_natDegree_lt
            rw [Polynomial.natDegree_pow, Polynomial.natDegree_X_add_C, mul_one]
            exact hlt
          rw [h0, mul_zero]
        · intro hdnot
          have : P.coeff d = 0 := Polynomial.notMem_support_iff.mp hdnot
          rw [this, map_zero, zero_mul, Polynomial.coeff_zero]
      rw [hcomp_d, sub_self]
    · have hgt : d < m := by omega
      have h1 : (P.comp (Polynomial.X + Polynomial.C 1)).coeff m = 0 := by
        apply Polynomial.coeff_eq_zero_of_natDegree_lt
        rw [Polynomial.natDegree_comp, Polynomial.natDegree_X_add_C, mul_one]
        exact hgt
      have h2 : P.coeff m = 0 := Polynomial.coeff_eq_zero_of_natDegree_lt hgt
      rw [h1, h2, sub_zero]
  have hcoeff_dm1 : Q.coeff (d - 1) = (d : ℝ) * P.leadingCoeff := by
    dsimp only [Q]
    rw [Polynomial.coeff_sub, Polynomial.comp_eq_sum_left, Polynomial.sum_def,
        Polynomial.finset_sum_coeff]
    have hsplit : ∀ b ∈ P.support,
        (Polynomial.C (P.coeff b) * (Polynomial.X + Polynomial.C 1) ^ b).coeff (d - 1) =
        if b = d then P.coeff d * (d : ℝ)
        else if b = d - 1 then P.coeff (d - 1) else 0 := by
      intro b hb
      rw [Polynomial.coeff_C_mul]
      have hble : b ≤ d := Polynomial.le_natDegree_of_mem_supp b hb
      by_cases hbd : b = d
      · subst hbd
        simp only [↓reduceIte]
        have hbin : ((Polynomial.X + Polynomial.C (1 : ℝ)) ^ d).coeff (d - 1) = (d : ℝ) := by
          rw [map_one, add_pow]
          simp only [one_pow, mul_one, Polynomial.finset_sum_coeff]
          rw [Finset.sum_eq_single (d - 1)]
          · rw [Polynomial.coeff_mul_natCast, Polynomial.coeff_X_pow, if_pos rfl, one_mul]
            have hch : Nat.choose d (d - 1) = d := by
              have h1 : d = (d - 1) + 1 := by omega
              calc Nat.choose d (d - 1) = Nat.choose ((d - 1) + 1) (d - 1) := by rw [← h1]
                _ = (d - 1) + 1 := Nat.choose_succ_self_right (d - 1)
                _ = d := h1.symm
            simp [hch]
          · intro j _ hjne
            rw [Polynomial.coeff_mul_natCast, Polynomial.coeff_X_pow, if_neg hjne.symm, zero_mul]
          · intro hnot
            exact absurd (Finset.mem_range.mpr (by omega)) hnot
        rw [hbin]
      · by_cases hbd1 : b = d - 1
        · subst hbd1
          simp only [hbd, ↓reduceIte]
          have hmon : ((Polynomial.X + Polynomial.C (1 : ℝ)) ^ (d - 1)).Monic :=
            (Polynomial.monic_X_add_C 1).pow (d - 1)
          have hnd : ((Polynomial.X + Polynomial.C (1 : ℝ)) ^ (d - 1)).natDegree = d - 1 := by
            rw [Polynomial.natDegree_pow, Polynomial.natDegree_X_add_C, mul_one]
          have hcoeff1 : ((Polynomial.X + Polynomial.C (1 : ℝ)) ^ (d - 1)).coeff (d - 1) = 1 := by
            have hlc := hmon.leadingCoeff
            rw [Polynomial.leadingCoeff, hnd] at hlc
            exact hlc
          rw [hcoeff1, mul_one]
        · simp only [hbd, hbd1, ↓reduceIte]
          have hlt : b < d - 1 := by omega
          have h0 : ((Polynomial.X + Polynomial.C (1 : ℝ)) ^ b).coeff (d - 1) = 0 := by
            apply Polynomial.coeff_eq_zero_of_natDegree_lt
            rw [Polynomial.natDegree_pow, Polynomial.natDegree_X_add_C, mul_one]
            exact hlt
          rw [h0, mul_zero]
    rw [Finset.sum_congr rfl hsplit]
    have hd_mem : d ∈ P.support :=
      Polynomial.mem_support_iff.mpr (Polynomial.leadingCoeff_ne_zero.mpr hPne)
    rw [← Finset.add_sum_erase _ _ hd_mem]
    simp only [↓reduceIte]
    have hrest : ∑ b ∈ P.support.erase d,
        (if b = d then P.coeff d * (d : ℝ) else if b = d - 1 then P.coeff (d - 1) else 0) =
        P.coeff (d - 1) := by
      by_cases hm1 : d - 1 ∈ P.support.erase d
      · rw [Finset.sum_eq_single (d - 1)]
        · have hne : d - 1 ≠ d := by omega
          simp [hne]
        · intro b hb hne
          have hbd : b ≠ d := (Finset.mem_erase.mp hb).1
          simp [hbd, hne]
        · intro habs; exact absurd hm1 habs
      · have hzero : P.coeff (d - 1) = 0 := by
          by_contra hc
          have hmem : d - 1 ∈ P.support := Polynomial.mem_support_iff.mpr hc
          exact hm1 (Finset.mem_erase.mpr ⟨by omega, hmem⟩)
        rw [hzero]
        apply Finset.sum_eq_zero
        intro b hb
        have hbd : b ≠ d := (Finset.mem_erase.mp hb).1
        have hbm1 : b ≠ d - 1 := fun h => hm1 (h ▸ hb)
        simp [hbd, hbm1]
    rw [hrest, Polynomial.leadingCoeff]
    ring
  have hne : Q.coeff (d - 1) ≠ 0 := by
    rw [hcoeff_dm1]
    exact mul_ne_zero (by positivity) (Polynomial.leadingCoeff_ne_zero.mpr hPne)
  have hndQ : Q.natDegree = d - 1 := le_antisymm
    (Polynomial.natDegree_le_iff_coeff_eq_zero.mpr hcoeff_d_gt)
    (Polynomial.le_natDegree_of_ne_zero hne)
  refine ⟨Q, ?_, hndQ, ?_⟩
  · intro x
    simp [fwdDiff, Q, Polynomial.eval_sub, Polynomial.eval_comp, Polynomial.eval_add,
      Polynomial.eval_X]
  · rw [Polynomial.leadingCoeff, hndQ, hcoeff_dm1]

/-- **Corollary 3 (Every nonconstant polynomial activation of degree `d ≥ 1` realizes `x ↦ x`
at width `d` and therefore fails weak identifiability by Theorem 2)**. -/
theorem poly_forward_diff_realizes_id (P : Polynomial ℝ) (hdeg : 1 ≤ P.natDegree) :
    ∃ (W₁ b₁ W₂ : Fin P.natDegree → ℝ) (c : ℝ),
      ∀ x : ℝ, mlp2 (fun t => P.eval t) W₁ b₁ W₂ c x = x := by
  -- Induct on k = P.natDegree - 1 to show fwdDiff (d - 1) P is affine of nonzero slope
  have hiter : ∀ k (R : Polynomial ℝ), R.natDegree = k + 1 →
      ∃ a b : ℝ, a ≠ 0 ∧ ∀ x : ℝ, fwdDiff k (fun t => R.eval t) x = a * x + b := by
    intro k
    induction k with
    | zero =>
      intro R hR
      refine ⟨R.coeff 1, R.coeff 0, ?_, ?_⟩
      · have h1 : (1 : ℕ) = R.natDegree := by omega
        rw [h1, ← Polynomial.leadingCoeff]
        apply Polynomial.leadingCoeff_ne_zero.mpr
        intro h0; rw [h0, Polynomial.natDegree_zero] at hR; omega
      · intro x
        simp only [fwdDiff]
        rw [Polynomial.eq_X_add_C_of_natDegree_le_one (le_of_eq hR)]
        simp [Polynomial.eval_add, Polynomial.eval_mul, Polynomial.eval_X, Polynomial.eval_C]
    | succ k ih =>
      intro R hR
      obtain ⟨Q, hQeval, hQdeg, _⟩ := fwdDiff_one_poly R (by omega)
      have hQdeg' : Q.natDegree = k + 1 := by omega
      obtain ⟨a, b, ha, hQ⟩ := ih Q hQdeg'
      refine ⟨a, b, ha, fun x => ?_⟩
      have hstep : ∀ n (f : ℝ → ℝ) (y : ℝ),
          fwdDiff (n + 1) f y = fwdDiff n (fwdDiff 1 f) y := by
        intro n
        induction n with
        | zero => intro f y; rfl
        | succ n ihn =>
          intro f y
          rw [fwdDiff, ihn f (y + 1), ihn f y, ← fwdDiff]
      rw [hstep k (fun t => R.eval t) x]
      have hfun : fwdDiff 1 (fun t => R.eval t) = fun t => Q.eval t := funext hQeval
      rw [hfun]
      exact hQ x
  have hnd : P.natDegree = (P.natDegree - 1) + 1 := by omega
  obtain ⟨a, b, ha, haff⟩ := hiter (P.natDegree - 1) P hnd
  let k := P.natDegree - 1
  let eFin : Fin P.natDegree ≃ Fin (k + 1) := finCongr hnd
  refine ⟨fun _ => 1, fun i => ((eFin i).val : ℝ),
    fun i => ((-1 : ℝ) ^ (k - (eFin i).val) * (Nat.choose k (eFin i).val : ℝ)) / a,
    -b / a, fun x => ?_⟩
  unfold mlp2 layer2 layer1
  have hsum_eq : ∑ i : Fin P.natDegree,
      ((-1 : ℝ) ^ (k - (eFin i).val) * (Nat.choose k (eFin i).val : ℝ)) / a *
        P.eval (1 * x + ((eFin i).val : ℝ)) =
      (∑ j : Fin (k + 1),
        ((-1 : ℝ) ^ (k - j.val) * (Nat.choose k j.val : ℝ)) * P.eval (x + (j.val : ℝ))) / a := by
    rw [Equiv.sum_comp eFin (fun j : Fin (k + 1) =>
      ((-1 : ℝ) ^ (k - j.val) * (Nat.choose k j.val : ℝ)) / a * P.eval (1 * x + (j.val : ℝ)))]
    rw [Finset.sum_div]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [one_mul]; ring
  rw [hsum_eq, ← fwdDiff_eq_sum k (fun t => P.eval t) x, haff x]
  field_simp [ha]
  ring

/-! ### 3. Affirmative Zero-Width & Width-1 Sigmoidal Identifiability Core (Theorem 4) -/

/-- **Zero-width constant identifiability**: When hidden width `m = 0` is allowed, every
constant function `x ↦ c` is uniquely identifiable (`c' = c`, unique trivial symmetry). -/
theorem zero_width_constant_identifiable (σ : ℝ → ℝ)
    (W₁ b₁ W₂ W₁' b₁' W₂' : Fin 0 → ℝ) (c c' : ℝ)
    (heq : ∀ x : ℝ, mlp2 σ W₁ b₁ W₂ c x = mlp2 σ W₁' b₁' W₂' c' x) :
    c' = c ∧ ∃! k : (Fin 0 → ℝ) ≃ₗ[ℝ] (Fin 0 → ℝ),
      LinearSymRelated σ W₁ b₁ W₂ c W₁' b₁' W₂' c' k := by
  have hc : c' = c := by
    have h0 := heq 0
    simp only [mlp2, layer2, Finset.univ_eq_empty, Finset.sum_empty, zero_add] at h0
    exact h0.symm
  refine ⟨hc, LinearEquiv.refl ℝ (Fin 0 → ℝ), ?_, ?_⟩
  · constructor
    · intro x; ext i; exact Fin.elim0 i
    · intro z; simp [layer2, hc]
  · intro k _
    ext z i
    exact Fin.elim0 i

/-- Plain logistic sigmoid `s(x) = (1 + exp(-x))⁻¹`. -/
noncomputable def logistic (x : ℝ) : ℝ :=
  (1 + Real.exp (-x))⁻¹

/-- Centered logistic `s(x) - 1/2` equals `(1/2) tanh(x/2)` (and is origin-passing). -/
theorem centered_logistic_eq_half_tanh (x : ℝ) :
    logistic x - 1 / 2 = (1 / 2) * Real.tanh (x / 2) := by
  unfold logistic
  rw [Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq]
  have hden1 : 0 < 1 + Real.exp (-x) := by positivity
  have hden2 : 0 < Real.exp (x / 2) + Real.exp (-(x / 2)) := by positivity
  have hexp : Real.exp (-(x / 2)) = Real.exp (x / 2) * Real.exp (-x) := by
    rw [← Real.exp_add]; congr 1; ring
  field_simp [ne_of_gt hden1, ne_of_gt hden2]
  rw [hexp]
  ring

/-- **Plain logistic failure under linear latent actions**: `s(0) = 1/2 ≠ 0` and
`s(x) = 1 - s(-x)` yields two width-1 realizations of `s` with distinct output biases `0` and `1`,
which lie in distinct linear-symmetry orbits. -/
theorem plain_logistic_nonidentifiable_linear :
    logistic 0 ≠ 0 ∧
    (∀ x : ℝ, mlp2 logistic ![1] ![0] ![1] 0 x = logistic x) ∧
    (∀ x : ℝ, mlp2 logistic ![-1] ![0] ![-1] 1 x = logistic x) ∧
    (∀ k : (Fin 1 → ℝ) ≃ₗ[ℝ] (Fin 1 → ℝ),
      ¬ LinearSymRelated logistic ![1] ![0] ![1] 0 ![-1] ![0] ![-1] 1 k) := by
  have h0 : logistic 0 ≠ 0 := by
    unfold logistic; norm_num
  have hrefl : ∀ x : ℝ, logistic x + logistic (-x) = 1 := by
    intro x
    have h1 := centered_logistic_eq_half_tanh x
    have h2 := centered_logistic_eq_half_tanh (-x)
    have htanh_neg : Real.tanh (-x / 2) = -Real.tanh (x / 2) := by
      rw [neg_div, Real.tanh_neg]
    linarith
  refine ⟨h0, ?_, ?_, ?_⟩
  · intro x; simp [mlp2, layer2, layer1]
  · intro x
    simp only [mlp2, layer2, layer1, Fin.sum_univ_one, Matrix.cons_val_zero, neg_mul, one_mul,
      add_zero]
    linarith [hrefl x]
  · intro k hrel
    have hbias := linear_symmetry_preserves_output_bias logistic hrel
    norm_num at hbias

/-- Asymptotic limit of `Real.tanh` at `+∞` is `1`. -/
lemma tendsto_tanh_atTop : Tendsto Real.tanh atTop (𝓝 1) := by
  have hrew : ∀ x : ℝ, Real.tanh x = (1 - Real.exp (-2 * x)) / (1 + Real.exp (-2 * x)) := by
    intro x
    rw [Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq]
    have hpos1 : 0 < Real.exp x + Real.exp (-x) := by positivity
    have hpos2 : 0 < 1 + Real.exp (-2 * x) := by positivity
    have hexp : Real.exp (-x) = Real.exp x * Real.exp (-2 * x) := by
      rw [← Real.exp_add]; congr 1; ring
    rw [hexp] at hpos1 ⊢
    field_simp [ne_of_gt hpos1, ne_of_gt hpos2]
  have hfun : Real.tanh = fun x : ℝ => (1 - Real.exp (-2 * x)) / (1 + Real.exp (-2 * x)) :=
    funext hrew
  rw [hfun]
  have h0 : Tendsto (fun x : ℝ => Real.exp (-2 * x)) atTop (𝓝 0) :=
    Real.tendsto_exp_atBot.comp (tendsto_id.const_mul_atTop_of_neg (by norm_num : (-2 : ℝ) < 0))
  have hnum : Tendsto (fun x : ℝ => 1 - Real.exp (-2 * x)) atTop (𝓝 (1 - 0)) :=
    tendsto_const_nhds.sub h0
  have hden : Tendsto (fun x : ℝ => 1 + Real.exp (-2 * x)) atTop (𝓝 (1 + 0)) :=
    tendsto_const_nhds.add h0
  have hdiv := hnum.div hden (by norm_num)
  simpa using hdiv

/-- Asymptotic limit of `Real.tanh` at `-∞` is `-1`. -/
lemma tendsto_tanh_atBot : Tendsto Real.tanh atBot (𝓝 (-1)) := by
  have h := (tendsto_tanh_atTop.comp tendsto_neg_atBot_atTop).neg
  simpa [Function.comp, Real.tanh_neg] using h

/-- `Real.tanh` is strictly increasing (hence injective) on `ℝ`. -/
lemma strictMono_tanh : StrictMono Real.tanh := by
  intro x y hxy
  rw [Real.tanh_eq_sinh_div_cosh, Real.tanh_eq_sinh_div_cosh]
  have hcx : 0 < Real.cosh x := Real.cosh_pos x
  have hcy : 0 < Real.cosh y := Real.cosh_pos y
  rw [div_lt_div_iff₀ hcx hcy]
  have hdiff : Real.sinh y * Real.cosh x - Real.sinh x * Real.cosh y = Real.sinh (y - x) := by
    rw [Real.sinh_sub]; ring
  have hpos : 0 < Real.sinh (y - x) := Real.sinh_pos_iff.mpr (sub_pos.mpr hxy)
  linarith

/-- **Theorem 4 (Exact width-1 `tanh` parameter recovery and unique signed symmetry)**:
If two regular (`w ≠ 0, a ≠ 0`, `w' ≠ 0, a' ≠ 0`) width-1 `tanh` MLPs realize the same function
on `ℝ`, then there exists a UNIQUE sign `s ∈ {1, -1}` such that
`w' = s * w`, `b' = s * b`, `a' = s * a`, and `c' = c`. -/
theorem tanh_width1_identifiable {w b a c w' b' a' c' : ℝ}
    (hw : w ≠ 0) (ha : a ≠ 0) (hw' : w' ≠ 0)
    (heq : ∀ x : ℝ, a * Real.tanh (w * x + b) + c = a' * Real.tanh (w' * x + b') + c') :
    ∃! s : ℝ, (s = 1 ∨ s = -1) ∧ w' = s * w ∧ b' = s * b ∧ a' = s * a ∧ c' = c := by
  -- Compute ±∞ limits of f(x) = a * tanh(w x + b) + c
  have hlim : ∀ (u v p q : ℝ), u ≠ 0 →
      Tendsto (fun x : ℝ => p * Real.tanh (u * x + v) + q) atTop
        (𝓝 (p * (if 0 < u then 1 else -1) + q)) ∧
      Tendsto (fun x : ℝ => p * Real.tanh (u * x + v) + q) atBot
        (𝓝 (p * (if 0 < u then -1 else 1) + q)) := by
    intro u v p q hu
    rcases lt_or_gt_of_ne hu with hneg | hpos
    · simp only [not_lt.mpr (le_of_lt hneg), ↓reduceIte]
      have h1 : Tendsto (fun x : ℝ => u * x + v) atTop atBot :=
        (tendsto_id.const_mul_atTop_of_neg hneg).atBot_add tendsto_const_nhds
      have h2 : Tendsto (fun x : ℝ => u * x + v) atBot atTop :=
        (tendsto_id.const_mul_atBot_of_neg hneg).atTop_add tendsto_const_nhds
      exact ⟨((tendsto_tanh_atBot.comp h1).const_mul p).add tendsto_const_nhds,
             ((tendsto_tanh_atTop.comp h2).const_mul p).add tendsto_const_nhds⟩
    · simp only [hpos, ↓reduceIte]
      have h1 : Tendsto (fun x : ℝ => u * x + v) atTop atTop :=
        (tendsto_id.const_mul_atTop hpos).atTop_add tendsto_const_nhds
      have h2 : Tendsto (fun x : ℝ => u * x + v) atBot atBot :=
        (tendsto_id.const_mul_atBot hpos).atBot_add tendsto_const_nhds
      exact ⟨((tendsto_tanh_atTop.comp h1).const_mul p).add tendsto_const_nhds,
             ((tendsto_tanh_atBot.comp h2).const_mul p).add tendsto_const_nhds⟩
  obtain ⟨htop1, hbot1⟩ := hlim w b a c hw
  obtain ⟨htop2, hbot2⟩ := hlim w' b' a' c' hw'
  have hfun : (fun x : ℝ => a * Real.tanh (w * x + b) + c) =
      (fun x : ℝ => a' * Real.tanh (w' * x + b') + c') := funext heq
  rw [hfun] at htop1 hbot1
  have eqTop := tendsto_nhds_unique htop1 htop2
  have eqBot := tendsto_nhds_unique hbot1 hbot2
  have hc : c' = c := by
    by_cases h1 : 0 < w <;> by_cases h2 : 0 < w' <;> simp only [h1, h2, ↓reduceIte] at eqTop eqBot <;> linarith
  set s : ℝ := if (0 < w ↔ 0 < w') then 1 else -1
  have hs_pm : s = 1 ∨ s = -1 := by
    dsimp only [s]; split_ifs <;> simp
  have hs_sq : s * s = 1 := by rcases hs_pm with h | h <;> rw [h] <;> norm_num
  have ha_eq : a' = s * a := by
    dsimp only [s]
    by_cases h1 : 0 < w <;> by_cases h2 : 0 < w' <;>
      simp only [h1, h2, iff_true, iff_false, not_true_eq_false, ↓reduceIte] at eqTop eqBot ⊢ <;>
      linarith
  have htanh_eq : ∀ x : ℝ, Real.tanh (w * x + b) = Real.tanh (s * w' * x + s * b') := by
    intro x
    have hx := heq x
    rw [hc, ha_eq] at hx
    have hmul : a * Real.tanh (w * x + b) = a * (s * Real.tanh (w' * x + b')) := by linarith
    have h1 := mul_left_cancel₀ ha hmul
    have hs_in : s * Real.tanh (w' * x + b') = Real.tanh (s * w' * x + s * b') := by
      rcases hs_pm with h | h
      · rw [h]; simp
      · rw [h]; simp only [neg_one_mul, ← Real.tanh_neg]; congr 1; ring
    rw [h1, hs_in]
  have haff : ∀ x : ℝ, w * x + b = s * w' * x + s * b' := fun x =>
    strictMono_tanh.injective (htanh_eq x)
  have hb_eq : b' = s * b := by
    have h0 := haff 0
    simp only [mul_zero, zero_add] at h0
    have hmul : s * b = s * (s * b') := congrArg (fun u => s * u) h0
    have : s * (s * b') = (s * s) * b' := by ring
    rw [this, hs_sq, one_mul] at hmul
    exact hmul.symm
  have hw_eq : w' = s * w := by
    have h1 := haff 1
    have h0 := haff 0
    have hsub : w = s * w' := by linarith
    have hmul : s * w = s * (s * w') := congrArg (fun u => s * u) hsub
    have : s * (s * w') = (s * s) * w' := by ring
    rw [this, hs_sq, one_mul] at hmul
    exact hmul.symm
  refine ⟨s, ⟨hs_pm, hw_eq, hb_eq, ha_eq, hc⟩, ?_⟩
  rintro s' ⟨_, hw', _, _, _⟩
  have hzero : (s' - s) * w = 0 := by linarith
  have := (mul_eq_zero.mp hzero).resolve_right hw
  linarith

/-- **Exact width-1 centered-logistic identifiability**: immediate consequence of
`centered_logistic_eq_half_tanh` and `tanh_width1_identifiable`. -/
theorem centered_logistic_width1_identifiable {w b a c w' b' a' c' : ℝ}
    (hw : w ≠ 0) (ha : a ≠ 0) (hw' : w' ≠ 0)
    (heq : ∀ x : ℝ, a * (logistic (w * x + b) - 1 / 2) + c =
      a' * (logistic (w' * x + b') - 1 / 2) + c') :
    ∃! s : ℝ, (s = 1 ∨ s = -1) ∧ w' = s * w ∧ b' = s * b ∧ a' = s * a ∧ c' = c := by
  have heq' : ∀ x : ℝ, (a / 2) * Real.tanh ((w / 2) * x + (b / 2)) + c =
      (a' / 2) * Real.tanh ((w' / 2) * x + (b' / 2)) + c' := by
    intro x
    have hx := heq x
    rw [centered_logistic_eq_half_tanh, centered_logistic_eq_half_tanh] at hx
    have h1 : (w * x + b) / 2 = w / 2 * x + b / 2 := by ring
    have h2 : (w' * x + b') / 2 = w' / 2 * x + b' / 2 := by ring
    rw [← h1, ← h2]
    linarith
  obtain ⟨s, ⟨hs_pm, hws, hbs, has, hc⟩, huniq⟩ := tanh_width1_identifiable
    (div_ne_zero hw (by norm_num)) (div_ne_zero ha (by norm_num))
    (div_ne_zero hw' (by norm_num)) heq'
  refine ⟨s, ⟨hs_pm, by linarith, by linarith, by linarith, hc⟩, ?_⟩
  rintro s' ⟨hs'pm, hw's, hb's, ha's, hc'⟩
  exact huniq s' ⟨hs'pm, by linarith, by linarith, by linarith, hc'⟩

end EquivariantMLPIdent
