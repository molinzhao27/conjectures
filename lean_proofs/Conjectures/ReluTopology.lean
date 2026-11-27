import Mathlib

/-!
# Formalization of Beshkov's Tight Topological Expressivity Conjecture (arXiv:2610.13931)

We formalize in Lean 4 + Mathlib (zero `sorry`):
1. **Theorem 1 (Part A — Complete End-to-End Formalization)**:
   For every `d ≥ 1`, the `(d, d+1, d+1)` ReLU neural network `P_net : (Fin d → ℝ) → (Fin (d + 1) → ℝ)`
   is continuous (`continuous_P_net`) and satisfies:
   - Exact image `Set.range P_net = boundarySimplex d` and `P_net '' omegaBar d = boundarySimplex d`
     (`beshkov_part_A`), with compact convex support `omegaBar d ⊆ [-1, 2]^d`
     (`isCompact_omegaBar`, `convex_omegaBar`, `omegaBar_subset_box`) and `P_net x = 0` for `2 ≤ S_sum x`
     (`P_net_eq_zero_of_S_ge_two`).
   - **Explicit Homeomorphism (`boundarySimplex_homeomorph_sphere`)**:
     `boundarySimplex d ≃ₜ Metric.sphere (0 : EuclideanSpace ℝ (Fin (d + 1))) 1` via radial projection
     from the barycenter `c = (1/(d+2), ..., 1/(d+2))`.
2. **Theorem 2 (Part B — Geometric Engine & Impossibility for Spheres / Simplex Boundary)**:
   For any `d ≥ 1`, any `W₂ : Matrix (Fin (d + 1)) (Fin d) ℝ`, and any `b₂ : Fin (d + 1) → ℝ`:
   - Every output `reluVec (W₂.mulVec h + b₂)` lies in `reluVec '' hyperplane n α` for a nonzero `n`
     (`bottleneck_in_hyperplane`, `bottleneck_relu_in_relu_hyperplane`).
   - **Straight-Ray Escape Lemma (`straight_ray_escape`)**: For every `n ≠ 0`, `α : ℝ`, and
     `p ∉ reluVec '' hyperplane n α`, there exists a nonzero direction `v` such that the entire ray
     `{p + t • v | t ≥ 0}` avoids `reluVec '' hyperplane n α`.
   - **Geometric Impossibility Corollaries**: No `(d, d, d+1)` ReLU network can contain ANY
     Euclidean sphere `{z | ∑ j, (z j - c j)^2 = r^2}` (`r > 0`) in its image
     (`bottleneck_not_superset_sphere`), nor can it contain `boundarySimplex d`
     (`bottleneck_not_superset_boundarySimplex`).

Note: Extending Part B from geometric spheres / simplex boundaries to *arbitrary wild topological
embeddings* of `S^d` inside `reluVec '' H` uses Alexander duality / Čech cohomology (not yet in
Mathlib) and is proved in prose in `solutions/relu_topology/solution.md`.
-/

open Finset BigOperators

namespace BeshkovReluTopology

/-- Standard `max 0 u` ReLU activation. -/
def relu (u : ℝ) : ℝ := max 0 u

/-- Componentwise ReLU on `Fin m → ℝ`. -/
def reluVec {m : ℕ} (u : Fin m → ℝ) : Fin m → ℝ := fun i => relu (u i)

/-- Hidden layer of Beshkov's `(d, d+1, d+1)` network. -/
def h_layer {d : ℕ} (x : Fin d → ℝ) : Fin (d + 1) → ℝ :=
  Fin.snoc (fun i : Fin d => relu (x i)) (relu (1 - ∑ i : Fin d, x i))

/-- Sum of hidden activations `S(x) = ∑ j, h_j(x)`. -/
def S_sum {d : ℕ} (x : Fin d → ℝ) : ℝ := ∑ j : Fin (d + 1), h_layer x j

/-- Output layer `g_i(y) = max(0, y_i + 2(1 - ∑ j, y_j))`. -/
def g_layer {d : ℕ} (y : Fin (d + 1) → ℝ) : Fin (d + 1) → ℝ :=
  fun i => relu (y i + 2 * (1 - ∑ j : Fin (d + 1), y j))

/-- Full `(d, d+1, d+1)` ReLU network `P = g ∘ h`. -/
def P_net {d : ℕ} (x : Fin d → ℝ) : Fin (d + 1) → ℝ := g_layer (h_layer x)

/-- Exact hidden-layer image set `H`. -/
def H_set (d : ℕ) : Set (Fin (d + 1) → ℝ) :=
  {y | (∀ j, 0 ≤ y j) ∧ 1 ≤ ∑ j, y j ∧ (1 < ∑ j, y j → ∃ j, y j = 0)}

/-- Boundary of the standard `(d+1)`-simplex `∂Δ₀^{d+1}`. -/
def boundarySimplex (d : ℕ) : Set (Fin (d + 1) → ℝ) :=
  {z | (∀ j, 0 ≤ z j) ∧ ∑ j, z j ≤ 1 ∧ ((∃ j, z j = 0) ∨ ∑ j, z j = 1)}

/-- Compact convex support `Ω̄ = {x | S(x) ≤ 2}`. -/
def omegaBar (d : ℕ) : Set (Fin d → ℝ) := {x | S_sum x ≤ 2}

/-! ### Step 1: Exact Characterization of Hidden-Layer Image -/

lemma sum_h_layer {d : ℕ} (x : Fin d → ℝ) :
    ∑ j : Fin (d + 1), h_layer x j =
      (∑ i : Fin d, relu (x i)) + relu (1 - ∑ i : Fin d, x i) := by
  unfold h_layer
  rw [Fin.sum_univ_castSucc]
  simp [Fin.snoc]

lemma h_layer_nonneg {d : ℕ} (x : Fin d → ℝ) (j : Fin (d + 1)) : 0 ≤ h_layer x j := by
  refine Fin.lastCases ?_ ?_ j
  · simp [h_layer, relu]
  · intro i; simp [h_layer, relu]

lemma h_layer_mem_H {d : ℕ} (x : Fin d → ℝ) : h_layer x ∈ H_set d := by
  refine ⟨h_layer_nonneg x, ?_, ?_⟩
  · rw [sum_h_layer]
    have h1 : ∑ i : Fin d, x i ≤ ∑ i : Fin d, relu (x i) :=
      Finset.sum_le_sum fun i _ => le_max_right 0 (x i)
    have h2 : 1 - ∑ i : Fin d, x i ≤ relu (1 - ∑ i : Fin d, x i) := le_max_right 0 _
    linarith
  · intro hgt
    by_contra hall
    push_neg at hall
    have hpos : ∀ j, 0 < h_layer x j := fun j => lt_of_le_of_ne (h_layer_nonneg x j) (Ne.symm (hall j))
    have hxi : ∀ i : Fin d, 0 < x i := by
      intro i
      have hi := hpos i.castSucc
      simp [h_layer, relu] at hi
      exact hi
    have hlast : 0 < 1 - ∑ i : Fin d, x i := by
      have hl := hpos (Fin.last d)
      simp [h_layer, relu] at hl
      linarith
    have hsum1 : ∑ j : Fin (d + 1), h_layer x j = 1 := by
      rw [sum_h_layer]
      have hs : ∑ i : Fin d, relu (x i) = ∑ i : Fin d, x i := by
        refine Finset.sum_congr rfl fun i _ => ?_
        exact max_eq_right (le_of_lt (hxi i))
      rw [hs, relu, max_eq_right (le_of_lt hlast)]
      ring
    linarith

lemma exists_preimage_h_layer {d : ℕ} {y : Fin (d + 1) → ℝ} (hy : y ∈ H_set d) :
    ∃ x : Fin d → ℝ, h_layer x = y := by
  rcases hy with ⟨hnn, hSge, hzero⟩
  set S : ℝ := ∑ j : Fin (d + 1), y j with hSdef
  have hSsplit : S = (∑ i : Fin d, y i.castSucc) + y (Fin.last d) := by
    rw [hSdef, Fin.sum_univ_castSucc]
  by_cases hcase : S = 1 ∨ y (Fin.last d) = 0
  · -- Case 1: x_i = y_i for i < d
    refine ⟨fun i => y i.castSucc, ?_⟩
    funext j
    refine Fin.lastCases ?_ ?_ j
    · simp only [h_layer, Fin.snoc_last, relu]
      rcases hcase with hS1 | hylast
      · have : 1 - ∑ i : Fin d, y i.castSucc = y (Fin.last d) := by linarith
        rw [this, max_eq_right (hnn (Fin.last d))]
      · have : 1 - ∑ i : Fin d, y i.castSucc ≤ 0 := by linarith
        rw [hylast, max_eq_left this]
    · intro i
      simp only [h_layer, Fin.snoc_castSucc, relu]
      exact max_eq_right (hnn i.castSucc)
  · -- Case 2: S > 1 and y_last > 0, so some k < d has y_k = 0
    push_neg at hcase
    have hSgt : 1 < S := lt_of_le_of_ne hSge (Ne.symm hcase.1)
    obtain ⟨j0, hj0⟩ := hzero hSgt
    have hj0_ne : j0 ≠ Fin.last d := by
      intro heq; rw [heq] at hj0; exact hcase.2 hj0
    obtain ⟨k, rfl⟩ := Fin.exists_castSucc_eq.mpr hj0_ne
    let x : Fin d → ℝ := Function.update (fun i : Fin d => y i.castSucc) k (1 - S)
    refine ⟨x, ?_⟩
    have hxsum : ∑ i : Fin d, x i = (∑ i : Fin d, y i.castSucc) + (1 - S) := by
      rw [← Finset.add_sum_erase _ _ (Finset.mem_univ k),
          ← Finset.add_sum_erase _ (fun i : Fin d => y i.castSucc) (Finset.mem_univ k)]
      have heq_rest : ∑ i ∈ Finset.univ.erase k, x i = ∑ i ∈ Finset.univ.erase k, y i.castSucc := by
        refine Finset.sum_congr rfl fun i hi => ?_
        have hne : i ≠ k := Finset.ne_of_mem_erase hi
        simp [x, Function.update_of_ne hne]
      rw [heq_rest]
      simp [x, hj0]
      ring
    funext j
    refine Fin.lastCases ?_ ?_ j
    · simp only [h_layer, Fin.snoc_last, relu]
      have : 1 - ∑ i : Fin d, x i = y (Fin.last d) := by linarith
      rw [this, max_eq_right (hnn (Fin.last d))]
    · intro i
      simp only [h_layer, Fin.snoc_castSucc, relu]
      by_cases hik : i = k
      · subst hik
        simp only [x, Function.update_self]
        rw [hj0, max_eq_left (by linarith)]
      · simp only [x, Function.update_of_ne hik]
        exact max_eq_right (hnn i.castSucc)

/-! ### Step 2: Exact Characterization of Output-Layer Image -/

lemma g_layer_mem_boundary {d : ℕ} {y : Fin (d + 1) → ℝ} (hy : y ∈ H_set d) :
    g_layer y ∈ boundarySimplex d := by
  rcases hy with ⟨hnn, hSge, hzero⟩
  set S : ℝ := ∑ j : Fin (d + 1), y j with hSdef
  set t : ℝ := 2 * (S - 1) with htdef
  have hgnn : ∀ j, 0 ≤ g_layer y j := fun j => le_max_left 0 _
  have hg_eq : ∀ j, g_layer y j = max 0 (y j - t) := fun j => by
    simp only [g_layer, relu, htdef, hSdef]; ring_nf
  by_cases hS1 : S = 1
  · have ht0 : t = 0 := by linarith
    have hge : g_layer y = y := by
      funext j; rw [hg_eq, ht0, sub_zero, max_eq_right (hnn j)]
    rw [hge]
    exact ⟨hnn, le_of_eq hS1, Or.inr hS1⟩
  · have hSgt : 1 < S := lt_of_le_of_ne hSge (Ne.symm hS1)
    have htpos : 0 < t := by linarith
    obtain ⟨j0, hj0⟩ := hzero hSgt
    have hgz0 : g_layer y j0 = 0 := by
      rw [hg_eq, hj0, zero_sub, max_eq_left (by linarith)]
    refine ⟨hgnn, ?_, Or.inl ⟨j0, hgz0⟩⟩
    -- Bound ∑ j, g_layer y j ≤ 1 using active set A = {j | t < y j}
    let A : Finset (Fin (d + 1)) := Finset.univ.filter (fun j => t < y j)
    have hsumA : ∑ j : Fin (d + 1), g_layer y j = ∑ j ∈ A, (y j - t) := by
      rw [← Finset.sum_filter_add_sum_filter_not Finset.univ (fun j => t < y j)]
      have h1 : ∑ j ∈ A, g_layer y j = ∑ j ∈ A, (y j - t) := by
        refine Finset.sum_congr rfl fun j hj => ?_
        have hlt : t < y j := (Finset.mem_filter.mp hj).2
        rw [hg_eq, max_eq_right (by linarith)]
      have h2 : ∑ j ∈ Finset.univ.filter (fun j => ¬ t < y j), g_layer y j = 0 := by
        refine Finset.sum_eq_zero fun j hj => ?_
        have hle : y j ≤ t := not_lt.mp (Finset.mem_filter.mp hj).2
        rw [hg_eq, max_eq_left (by linarith)]
      rw [h1, h2, add_zero]
    rw [hsumA]
    by_cases hAempty : A = ∅
    · simp [hAempty]
    · have hAcard : (1 : ℝ) ≤ (A.card : ℝ) := by
        exact_mod_cast Finset.one_le_card.mpr (Finset.nonempty_iff_ne_empty.mpr hAempty)
      have hsumAleS : ∑ j ∈ A, y j ≤ S := by
        rw [hSdef]
        exact Finset.sum_le_univ_sum_of_nonneg hnn
      rw [Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul]
      nlinarith

lemma exists_preimage_g_layer {d : ℕ} (hd : 1 ≤ d) {z : Fin (d + 1) → ℝ}
    (hz : z ∈ boundarySimplex d) :
    ∃ y ∈ H_set d, (∑ j, y j ≤ 2) ∧ g_layer y = z := by
  rcases hz with ⟨hznn, hZle, hbdry⟩
  set Sz : ℝ := ∑ j : Fin (d + 1), z j with hZdef
  by_cases hZ1 : Sz = 1
  · refine ⟨z, ⟨hznn, le_of_eq hZ1.symm, fun hlt => absurd (hZdef.symm.trans hZ1) (ne_of_gt hlt)⟩, by linarith, ?_⟩
    funext j
    simp only [g_layer, relu, ← hZdef, hZ1, sub_self, mul_zero, add_zero]
    exact max_eq_right (hznn j)
  · by_cases hZ0 : Sz = 0
    · have hzall0 : ∀ j, z j = 0 := fun j =>
        (Finset.sum_eq_zero_iff_of_nonneg (fun i _ => hznn i)).mp hZ0 j (Finset.mem_univ j)
      let y : Fin (d + 1) → ℝ := fun j => if j = 0 then 2 else 0
      have hylast : y (Fin.last d) = 0 := by
        have hne : (Fin.last d : Fin (d + 1)) ≠ 0 := by
          intro h; have := congrArg Fin.val h; simp at this; omega
        simp [y, hne]
      have hyS : ∑ j : Fin (d + 1), y j = 2 := by
        simp [y]
      refine ⟨y, ⟨fun j => by dsimp [y]; split_ifs <;> linarith, by linarith,
                  fun _ => ⟨Fin.last d, hylast⟩⟩, by linarith, ?_⟩
      funext j
      rw [hzall0 j]
      simp only [g_layer, relu, hyS]
      have hyle : y j ≤ 2 := by dsimp [y]; split_ifs <;> linarith
      exact max_eq_left (by linarith)
    · -- Main case: 0 < Sz < 1
      have hZlt : Sz < 1 := lt_of_le_of_ne hZle hZ1
      have hZnn : 0 ≤ Sz := Finset.sum_nonneg fun j _ => hznn j
      have hZpos : 0 < Sz := lt_of_le_of_ne hZnn (Ne.symm hZ0)
      have hhas0 : ∃ j0, z j0 = 0 := by
        rcases hbdry with h | h
        · exact h
        · exact absurd h hZ1
      obtain ⟨j0, hj0⟩ := hhas0
      let A : Finset (Fin (d + 1)) := Finset.univ.filter (fun j => 0 < z j)
      have hA_ne : A.Nonempty := by
        by_contra h
        rw [Finset.not_nonempty_iff_eq_empty] at h
        have : Sz = 0 := Finset.sum_eq_zero fun j _ => by
          have hj : j ∉ A := by simp [h]
          have hle : z j ≤ 0 := not_lt.mp (by simpa [A] using hj)
          exact le_antisymm hle (hznn j)
        exact hZ0 this
      have hj0_notA : j0 ∉ A := by simp [A, hj0]
      set m : ℝ := (A.card : ℝ) with hmdef
      have hm1 : (1 : ℝ) ≤ m := by
        rw [hmdef]
        exact_mod_cast Finset.one_le_card.mpr hA_ne
      have h2m1_pos : 0 < 2 * m - 1 := by linarith
      set t : ℝ := 2 * (1 - Sz) / (2 * m - 1) with htdef
      have htpos : 0 < t := div_pos (by linarith) h2m1_pos
      let y : Fin (d + 1) → ℝ := fun j => if j ∈ A then z j + t else 0
      have hynn : ∀ j, 0 ≤ y j := fun j => by
        dsimp [y]; split_ifs
        · linarith [hznn j]
        · exact le_refl 0
      have hyj0 : y j0 = 0 := by simp [y, hj0_notA]
      have hsumzA : ∑ j ∈ A, z j = Sz := by
        rw [hZdef, ← Finset.sum_filter_add_sum_filter_not Finset.univ (fun j => 0 < z j)]
        have hzeroRest : ∑ j ∈ Finset.univ.filter (fun j => ¬ 0 < z j), z j = 0 := by
          refine Finset.sum_eq_zero fun j hj => ?_
          have hle : z j ≤ 0 := not_lt.mp (Finset.mem_filter.mp hj).2
          exact le_antisymm hle (hznn j)
        rw [hzeroRest, add_zero]
      have hyS : ∑ j : Fin (d + 1), y j = 1 + t / 2 := by
        have h1 : ∑ j : Fin (d + 1), y j = ∑ j ∈ A, (z j + t) := by
          rw [← Finset.sum_filter_add_sum_filter_not Finset.univ (fun j => j ∈ A)]
          have hAeq : Finset.univ.filter (fun j => j ∈ A) = A := by ext j; simp
          rw [hAeq]
          have hrest : ∑ j ∈ Finset.univ.filter (fun j => j ∉ A), y j = 0 :=
            Finset.sum_eq_zero fun j hj => by simp [y, (Finset.mem_filter.mp hj).2]
          have hmain : ∑ j ∈ A, y j = ∑ j ∈ A, (z j + t) :=
            Finset.sum_congr rfl fun j hj => by simp [y, hj]
          rw [hmain, hrest, add_zero]
        rw [h1, Finset.sum_add_distrib, hsumzA, Finset.sum_const, nsmul_eq_mul, ← hmdef, htdef]
        have hne : 2 * m - 1 ≠ 0 := ne_of_gt h2m1_pos
        rw [eq_comm]
        calc 1 + 2 * (1 - Sz) / (2 * m - 1) / 2
            = ((2 * m - 1) + (1 - Sz)) / (2 * m - 1) := by field_simp [hne]
          _ = (Sz * (2 * m - 1) + 2 * m * (1 - Sz)) / (2 * m - 1) := by ring
          _ = Sz + m * (2 * (1 - Sz) / (2 * m - 1)) := by field_simp [hne]
      have hyS_le2 : ∑ j : Fin (d + 1), y j ≤ 2 := by
        rw [hyS, htdef]
        have hnum : 1 - Sz ≤ 2 * m - 1 := by linarith
        have hdiv : (1 - Sz) / (2 * m - 1) ≤ 1 := (div_le_one h2m1_pos).mpr hnum
        have hhalf : 2 * (1 - Sz) / (2 * m - 1) / 2 = (1 - Sz) / (2 * m - 1) := by ring
        linarith
      refine ⟨y, ⟨hynn, by linarith, fun _ => ⟨j0, hyj0⟩⟩, hyS_le2, ?_⟩
      funext j
      simp only [g_layer, relu, hyS]
      have hshift : 2 * (1 - (1 + t / 2)) = -t := by ring
      rw [hshift]
      by_cases hjA : j ∈ A
      · simp only [y, hjA, ↓reduceIte]
        have : z j + t + -t = z j := by ring
        rw [this, max_eq_right (hznn j)]
      · simp only [y, hjA, ↓reduceIte, zero_add]
        have hzj0 : z j = 0 := le_antisymm (not_lt.mp (by simpa [A] using hjA)) (hznn j)
        rw [hzj0, max_eq_left (by linarith)]

/-! ### Step 3: Compact Support and Main Theorem 1 -/

lemma omegaBar_subset_box {d : ℕ} {x : Fin d → ℝ} (hx : x ∈ omegaBar d) (k : Fin d) :
    -1 ≤ x k ∧ x k ≤ 2 := by
  have hS : S_sum x ≤ 2 := hx
  unfold S_sum at hS
  rw [sum_h_layer] at hS
  constructor
  · -- x k ≥ -1 from S(x) ≥ 1 + (-x k)
    have hsum_ge : (∑ i : Fin d, x i) - x k ≤ (∑ i : Fin d, relu (x i)) - relu (x k) := by
      have h1 : ∑ i ∈ Finset.univ.erase k, x i ≤ ∑ i ∈ Finset.univ.erase k, relu (x i) :=
        Finset.sum_le_sum (fun i _ => le_max_right 0 (x i))
      have h2 := Finset.add_sum_erase Finset.univ (fun i => x i) (Finset.mem_univ k)
      have h3 := Finset.add_sum_erase Finset.univ (fun i => relu (x i)) (Finset.mem_univ k)
      linarith
    have hrl : 1 - ∑ i : Fin d, x i ≤ relu (1 - ∑ i : Fin d, x i) := le_max_right 0 _
    have hnnk : 0 ≤ relu (x k) := le_max_left 0 _
    linarith
  · -- x k ≤ 2 from x k ≤ relu(x k) ≤ S(x) ≤ 2
    have hxk : x k ≤ relu (x k) := le_max_right 0 (x k)
    have hsumk : relu (x k) ≤ ∑ i : Fin d, relu (x i) :=
      Finset.single_le_sum (fun i _ => le_max_left 0 (x i)) (Finset.mem_univ k)
    have hrl0 : 0 ≤ relu (1 - ∑ i : Fin d, x i) := le_max_left 0 _
    linarith

lemma convex_omegaBar (d : ℕ) : Convex ℝ (omegaBar d) := by
  intro x hx y hy a b ha hb hab
  show S_sum (a • x + b • y) ≤ 2
  have hS_x : S_sum x ≤ 2 := hx
  have hS_y : S_sum y ≤ 2 := hy
  unfold S_sum at hS_x hS_y ⊢
  rw [sum_h_layer] at hS_x hS_y ⊢
  have hterm : ∀ i : Fin d, relu (a * x i + b * y i) ≤ a * relu (x i) + b * relu (y i) := by
    intro i
    simp only [relu]
    refine max_le ?_ ?_
    · positivity
    · nlinarith [le_max_right 0 (x i), le_max_right 0 (y i)]
  have hsum_le : ∑ i : Fin d, relu ((a • x + b • y) i) ≤
      a * (∑ i : Fin d, relu (x i)) + b * (∑ i : Fin d, relu (y i)) := by
    calc ∑ i : Fin d, relu ((a • x + b • y) i)
        ≤ ∑ i : Fin d, (a * relu (x i) + b * relu (y i)) :=
          Finset.sum_le_sum fun i _ => hterm i
      _ = a * (∑ i : Fin d, relu (x i)) + b * (∑ i : Fin d, relu (y i)) := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
  have hlast : relu (1 - ∑ i : Fin d, (a • x + b • y) i) ≤
      a * relu (1 - ∑ i : Fin d, x i) + b * relu (1 - ∑ i : Fin d, y i) := by
    have hlin : 1 - ∑ i : Fin d, (a • x + b • y) i =
        a * (1 - ∑ i : Fin d, x i) + b * (1 - ∑ i : Fin d, y i) := by
      have h1 : ∑ i : Fin d, (a • x + b • y) i =
          a * (∑ i : Fin d, x i) + b * (∑ i : Fin d, y i) := by
        simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, Finset.sum_add_distrib,
          ← Finset.mul_sum]
      rw [h1]
      nlinarith
    rw [hlin]
    simp only [relu]
    refine max_le ?_ ?_
    · positivity
    · nlinarith [le_max_right 0 (1 - ∑ i : Fin d, x i), le_max_right 0 (1 - ∑ i : Fin d, y i)]
  nlinarith

lemma continuous_S_sum (d : ℕ) : Continuous (@S_sum d) := by
  have hfun : (@S_sum d) = fun x =>
      (∑ i : Fin d, max 0 (x i)) + max 0 (1 - ∑ i : Fin d, x i) := by
    funext x
    unfold S_sum
    rw [sum_h_layer]
    rfl
  rw [hfun]
  refine Continuous.add ?_ ?_
  · exact continuous_finset_sum _ fun i _ => continuous_const.max (continuous_apply i)
  · exact continuous_const.max (continuous_const.sub (continuous_finset_sum _ fun i _ => continuous_apply i))

lemma isCompact_omegaBar (d : ℕ) : IsCompact (omegaBar d) := by
  have hbox : IsCompact (Set.pi Set.univ fun _ : Fin d => Set.Icc (-1 : ℝ) 2) :=
    isCompact_univ_pi fun _ => isCompact_Icc
  have hclosed : IsClosed (omegaBar d) :=
    isClosed_le (continuous_S_sum d) continuous_const
  refine hbox.of_isClosed_subset hclosed fun x hx => ?_
  intro k _
  exact omegaBar_subset_box hx k

lemma P_net_eq_zero_of_S_ge_two {d : ℕ} {x : Fin d → ℝ} (hS : 2 ≤ S_sum x) :
    P_net x = 0 := by
  funext i
  simp only [P_net, g_layer, relu, Pi.zero_apply]
  have hi : h_layer x i ≤ S_sum x :=
    Finset.single_le_sum (fun j _ => h_layer_nonneg x j) (Finset.mem_univ i)
  have hSdef : ∑ j : Fin (d + 1), h_layer x j = S_sum x := rfl
  rw [hSdef]
  exact max_eq_left (by linarith)

/-- **Theorem 1 (Part A of Beshkov's Conjecture)**:
For every `d ≥ 1`, the `(d, d+1, d+1)` ReLU network `P_net` maps both `ℝ^d` and the compact convex
set `omegaBar d ⊆ [-1, 2]^d` onto the exact boundary of the standard `(d+1)`-simplex `∂Δ₀^{d+1}`. -/
theorem beshkov_part_A {d : ℕ} (hd : 1 ≤ d) :
    Set.range (@P_net d) = boundarySimplex d ∧
    (@P_net d) '' omegaBar d = boundarySimplex d := by
  have hsub_range : Set.range (@P_net d) ⊆ boundarySimplex d := by
    rintro z ⟨x, rfl⟩
    exact g_layer_mem_boundary (h_layer_mem_H x)
  have hsub_omega : boundarySimplex d ⊆ (@P_net d) '' omegaBar d := by
    intro z hz
    obtain ⟨y, hyH, hyS2, hgy⟩ := exists_preimage_g_layer hd hz
    obtain ⟨x, hxy⟩ := exists_preimage_h_layer hyH
    refine ⟨x, ?_, ?_⟩
    · show S_sum x ≤ 2
      unfold S_sum
      rw [hxy]
      exact hyS2
    · show g_layer (h_layer x) = z
      rw [hxy, hgy]
  have hsub_om_range : (@P_net d) '' omegaBar d ⊆ Set.range (@P_net d) :=
    Set.image_subset_range _ _
  exact ⟨Set.Subset.antisymm hsub_range (hsub_omega.trans hsub_om_range),
         Set.Subset.antisymm (hsub_om_range.trans hsub_range) hsub_omega⟩

/-! ### Theorem 2 (Part B): Straight-Ray Escape Lemma for `(d, d, d+1)` Bottleneck -/

/-- An affine hyperplane `{u | ∑ i, n i * u i = α}` in `ℝ^{d+1}`. -/
def hyperplane {d : ℕ} (n : Fin (d + 1) → ℝ) (α : ℝ) : Set (Fin (d + 1) → ℝ) :=
  {u | ∑ i : Fin (d + 1), n i * u i = α}

/-- Every `W₂ : Matrix (Fin (d+1)) (Fin d) ℝ` has a nonzero left kernel vector `n`,
so `W₂ h + b₂` always lies inside an affine hyperplane `hyperplane n (∑ i, n i * b₂ i)`. -/
theorem bottleneck_in_hyperplane {d : ℕ} (W₂ : Matrix (Fin (d + 1)) (Fin d) ℝ)
    (b₂ : Fin (d + 1) → ℝ) :
    ∃ n : Fin (d + 1) → ℝ, n ≠ 0 ∧
      ∀ h : Fin d → ℝ, (W₂.mulVec h + b₂) ∈ hyperplane n (∑ i, n i * b₂ i) := by
  -- The linear map `W₂ᵀ : (Fin (d+1) → ℝ) →ₗ[ℝ] (Fin d → ℝ)` has domain dim d+1 > codomain dim d.
  let f : (Fin (d + 1) → ℝ) →ₗ[ℝ] (Fin d → ℝ) := Matrix.mulVecLin W₂.transpose
  have hlt : Module.finrank ℝ (Fin d → ℝ) < Module.finrank ℝ (Fin (d + 1) → ℝ) := by
    simp
  have hker : LinearMap.ker f ≠ ⊥ := by
    intro hbot
    have hinj : Function.Injective f := LinearMap.ker_eq_bot.mp hbot
    have hle := LinearMap.finrank_le_finrank_of_injective hinj
    omega
  obtain ⟨n, hn_mem, hn_ne⟩ := Submodule.exists_mem_ne_zero_of_ne_bot hker
  refine ⟨n, hn_ne, fun h => ?_⟩
  show ∑ i : Fin (d + 1), n i * (W₂.mulVec h + b₂) i = ∑ i : Fin (d + 1), n i * b₂ i
  have hW : ∑ i : Fin (d + 1), n i * W₂.mulVec h i = 0 := by
    have hfn : W₂.transpose.mulVec n = 0 := hn_mem
    have hdot : dotProduct n (W₂.mulVec h) = dotProduct (W₂.transpose.mulVec n) h := by
      rw [Matrix.dotProduct_mulVec, Matrix.mulVec_transpose]
    have h0 : dotProduct n (W₂.mulVec h) = 0 := by rw [hdot, hfn, zero_dotProduct]
    exact h0
  calc ∑ i : Fin (d + 1), n i * (W₂.mulVec h + b₂) i
      = (∑ i : Fin (d + 1), n i * W₂.mulVec h i) + (∑ i : Fin (d + 1), n i * b₂ i) := by
        rw [← Finset.sum_add_distrib]
        refine Finset.sum_congr rfl fun i _ => ?_
        simp [mul_add]
    _ = ∑ i : Fin (d + 1), n i * b₂ i := by rw [hW, zero_add]

/-- Single joining lemma for Theorem 2: any output of a `(·, d, d+1)` ReLU layer lies inside
`reluVec '' hyperplane n α` for a nonzero normal `n`. -/
theorem bottleneck_relu_in_relu_hyperplane {d : ℕ} (W₂ : Matrix (Fin (d + 1)) (Fin d) ℝ)
    (b₂ : Fin (d + 1) → ℝ) :
    ∃ n : Fin (d + 1) → ℝ, n ≠ 0 ∧
      ∀ h : Fin d → ℝ, reluVec (W₂.mulVec h + b₂) ∈ reluVec '' hyperplane n (∑ i, n i * b₂ i) := by
  obtain ⟨n, hn, hmem⟩ := bottleneck_in_hyperplane W₂ b₂
  exact ⟨n, hn, fun h => ⟨W₂.mulVec h + b₂, hmem h, rfl⟩⟩

/-- Standard basis vector `e_k` in `Fin m → ℝ`. -/
def stdBasis {m : ℕ} (k : Fin m) : Fin m → ℝ := fun j => if j = k then 1 else 0

lemma stdBasis_ne_zero {m : ℕ} (k : Fin m) : stdBasis k ≠ 0 := by
  intro h
  have := congrFun h k
  simp [stdBasis] at this

lemma sum_mul_stdBasis {m : ℕ} (n : Fin m → ℝ) (k : Fin m) :
    ∑ i : Fin m, n i * stdBasis k i = n k := by
  simp [stdBasis]

/-- **Straight-Ray Escape Lemma (Part B core engine)**:
For every affine hyperplane `H = {u | n · u = α}` with `n ≠ 0` and every point
`p ∉ reluVec '' H`, there exists a nonzero direction `v` such that the entire ray
`{p + t • v | t ≥ 0}` avoids `reluVec '' H`. -/
theorem straight_ray_escape {d : ℕ} {n : Fin (d + 1) → ℝ} (hn : n ≠ 0) (α : ℝ)
    {p : Fin (d + 1) → ℝ} (hp : p ∉ reluVec '' hyperplane n α) :
    ∃ v : Fin (d + 1) → ℝ, v ≠ 0 ∧
      ∀ t : ℝ, 0 ≤ t → (p + t • v) ∉ reluVec '' hyperplane n α := by
  -- Helper: if some coordinate of q is < 0, then q ∉ reluVec '' S for any S
  have not_mem_of_neg : ∀ (S : Set (Fin (d + 1) → ℝ)) (q : Fin (d + 1) → ℝ),
      (∃ k, q k < 0) → q ∉ reluVec '' S := by
    rintro S q ⟨k, hqk⟩ ⟨u, _, huq⟩
    have hnn : 0 ≤ reluVec u k := le_max_left 0 (u k)
    rw [huq] at hnn
    linarith
  -- Case 1: some coordinate p k ≤ 0
  by_cases hneg : ∃ k : Fin (d + 1), p k ≤ 0
  · obtain ⟨k, hpk⟩ := hneg
    refine ⟨-stdBasis k, neg_ne_zero.mpr (stdBasis_ne_zero k), fun t ht => ?_⟩
    by_cases ht0 : t = 0
    · subst ht0; simpa using hp
    · have htpos : 0 < t := lt_of_le_of_ne ht (Ne.symm ht0)
      refine not_mem_of_neg _ (p + t • (-stdBasis k)) ⟨k, ?_⟩
      simp [stdBasis]
      linarith
  · -- Case 2: p k > 0 for all k
    push_neg at hneg
    have hnp_ne : ∑ i : Fin (d + 1), n i * p i ≠ α := by
      intro heq
      apply hp
      refine ⟨p, heq, ?_⟩
      funext i
      exact max_eq_right (le_of_lt (hneg i))
    -- Reduce to WLOG ∑ n_i p_i < α by swapping (n, α) with (-n, -α) if needed
    suffices hmain : ∀ (m : Fin (d + 1) → ℝ) (β : ℝ), m ≠ 0 →
        (∑ i, m i * p i < β) →
        ∃ v : Fin (d + 1) → ℝ, v ≠ 0 ∧
          ∀ t : ℝ, 0 ≤ t → (p + t • v) ∉ reluVec '' hyperplane m β by
      rcases lt_or_gt_of_ne hnp_ne with hlt | hgt
      · exact hmain n α hn hlt
      · have hneg_sum : ∑ i : Fin (d + 1), (-n) i * p i < -α := by
          have : ∑ i : Fin (d + 1), (-n) i * p i = -(∑ i : Fin (d + 1), n i * p i) := by
            simp [Finset.sum_neg_distrib]
          linarith
        obtain ⟨v, hv, hray⟩ := hmain (-n) (-α) (neg_ne_zero.mpr hn) hneg_sum
        refine ⟨v, hv, fun t ht hmem => hray t ht ?_⟩
        rcases hmem with ⟨u, huH, huq⟩
        refine ⟨u, ?_, huq⟩
        show ∑ i, (-n) i * u i = -α
        have hEq : ∑ i, n i * u i = α := huH
        simp [Finset.sum_neg_distrib, hEq]
    intro m β hm hmp_lt
    -- Subcase 2a: ∃ k with m k > 0 -> escape along -e_k
    by_cases hpos_m : ∃ k : Fin (d + 1), 0 < m k
    · obtain ⟨k, hmk⟩ := hpos_m
      refine ⟨-stdBasis k, neg_ne_zero.mpr (stdBasis_ne_zero k), fun t ht => ?_⟩
      set q : Fin (d + 1) → ℝ := p + t • (-stdBasis k)
      by_cases htp : p k < t
      · -- t > p k: q k < 0
        refine not_mem_of_neg _ q ⟨k, ?_⟩
        simp [q, stdBasis]
        linarith
      · -- 0 ≤ t ≤ p k: for any preimage u of q under reluVec, u j = p j (j ≠ k) and u k ≤ p k
        rintro ⟨u, huH, huq⟩
        have huj_ne : ∀ j : Fin (d + 1), j ≠ k → u j = p j := by
          intro j hjk
          have hqj : q j = p j := by simp [q, stdBasis, hjk]
          have hru : relu (u j) = p j := by
            have := congrFun huq j
            simpa [reluVec, hqj] using this
          have hpj : 0 < p j := hneg j
          by_contra hne
          have hle : u j ≤ 0 := by
            by_contra hgt; push_neg at hgt
            rw [relu, max_eq_right (le_of_lt hgt)] at hru
            exact hne hru
          rw [relu, max_eq_left hle] at hru
          linarith
        have huk_le : u k ≤ p k := by
          have hqk : q k = p k - t := by simp [q, stdBasis]; ring
          have hru : relu (u k) = p k - t := by
            have := congrFun huq k
            simpa [reluVec, hqk] using this
          have hule : u k ≤ relu (u k) := le_max_right 0 (u k)
          linarith
        -- Then ∑ m_i u_i ≤ ∑ m_i p_i < β, contradicting u ∈ H!
        have hsum_le : ∑ i : Fin (d + 1), m i * u i ≤ ∑ i : Fin (d + 1), m i * p i := by
          rw [← Finset.add_sum_erase _ _ (Finset.mem_univ k),
              ← Finset.add_sum_erase _ (fun i => m i * p i) (Finset.mem_univ k)]
          have hrest : ∑ i ∈ Finset.univ.erase k, m i * u i =
              ∑ i ∈ Finset.univ.erase k, m i * p i := by
            refine Finset.sum_congr rfl fun i hi => ?_
            rw [huj_ne i (Finset.ne_of_mem_erase hi)]
          rw [hrest]
          nlinarith
        have huEq : ∑ i : Fin (d + 1), m i * u i = β := huH
        linarith
    · -- Subcase 2b: ∀ i, m i ≤ 0, so since m ≠ 0 there is k with m k < 0 -> escape along +e_k
      push_neg at hpos_m
      have hneg_m : ∃ k : Fin (d + 1), m k < 0 := by
        by_contra hall
        push_neg at hall
        apply hm
        funext i
        exact le_antisymm (hpos_m i) (hall i)
      obtain ⟨k, hmk⟩ := hneg_m
      refine ⟨stdBasis k, stdBasis_ne_zero k, fun t ht => ?_⟩
      set q : Fin (d + 1) → ℝ := p + t • stdBasis k
      rintro ⟨u, huH, huq⟩
      -- Every q j > 0, so u = q identically
      have hu_eq_q : u = q := by
        funext j
        have hqj_pos : 0 < q j := by
          dsimp [q, stdBasis]
          split_ifs
          · linarith [hneg j]
          · linarith [hneg j]
        have hru : relu (u j) = q j := congrFun huq j
        by_contra hne
        have hle : u j ≤ 0 := by
          by_contra hgt; push_neg at hgt
          rw [relu, max_eq_right (le_of_lt hgt)] at hru
          exact hne hru
        rw [relu, max_eq_left hle] at hru
        linarith
      have hsum_q : ∑ i : Fin (d + 1), m i * q i = (∑ i, m i * p i) + t * m k := by
        have h1 : ∑ i : Fin (d + 1), m i * (p i + t * stdBasis k i) =
            (∑ i, m i * p i) + t * (∑ i, m i * stdBasis k i) := by
          rw [Finset.mul_sum, ← Finset.sum_add_distrib]
          refine Finset.sum_congr rfl fun i _ => ?_
          ring
        simpa [q, Pi.add_apply, Pi.smul_apply, smul_eq_mul, sum_mul_stdBasis] using h1
      have huEq : ∑ i : Fin (d + 1), m i * u i = β := huH
      rw [hu_eq_q, hsum_q] at huEq
      nlinarith

/-! ### Part A Completion: Continuity of `P_net` and Explicit Homeomorphism `∂Δ₀^{d+1} ≃ₜ S^d` -/

lemma continuous_h_layer (d : ℕ) : Continuous (@h_layer d) := by
  refine continuous_pi fun j => ?_
  refine Fin.lastCases ?_ ?_ j
  · simp only [h_layer, Fin.snoc_last, relu]
    exact continuous_const.max (continuous_const.sub (continuous_finset_sum _ fun i _ => continuous_apply i))
  · intro i
    simp only [h_layer, Fin.snoc_castSucc, relu]
    exact continuous_const.max (continuous_apply i)

lemma continuous_g_layer (d : ℕ) : Continuous (@g_layer d) := by
  refine continuous_pi fun i => ?_
  simp only [g_layer, relu]
  exact continuous_const.max
    ((continuous_apply i).add
      (continuous_const.mul (continuous_const.sub (continuous_finset_sum _ fun j _ => continuous_apply j))))

/-- `P_net` is continuous on `ℝ^d`. -/
theorem continuous_P_net (d : ℕ) : Continuous (@P_net d) :=
  (continuous_g_layer d).comp (continuous_h_layer d)

/-- Interior barycenter `c = (1/(d+2), …, 1/(d+2))` of the standard `(d+1)`-simplex `Δ₀^{d+1}`. -/
noncomputable def barycenter (d : ℕ) : Fin (d + 1) → ℝ :=
  fun _ => 1 / ((d : ℝ) + 2)

/-- Minkowski gauge functional of `Δ₀^{d+1} - barycenter d` (scaled by `1/(d+2)`):
`simplexGauge u = max (∑ j, u j) (max_j (-u j))`. -/
def simplexGauge {d : ℕ} (u : Fin (d + 1) → ℝ) : ℝ :=
  max (∑ j : Fin (d + 1), u j)
    (Finset.univ.sup' ⟨0, Finset.mem_univ 0⟩ (fun j : Fin (d + 1) => -(u j)))

lemma simplexGauge_pos {d : ℕ} {u : Fin (d + 1) → ℝ} (hu : u ≠ 0) :
    0 < simplexGauge u := by
  by_contra hle
  push_neg at hle
  have hsum : ∑ j : Fin (d + 1), u j ≤ 0 := le_trans (le_max_left _ _) hle
  have hsup : Finset.univ.sup' ⟨0, Finset.mem_univ 0⟩ (fun j : Fin (d + 1) => -(u j)) ≤ 0 :=
    le_trans (le_max_right _ _) hle
  have hnn : ∀ j : Fin (d + 1), 0 ≤ u j := by
    intro j
    have hj : -(u j) ≤ Finset.univ.sup' ⟨0, Finset.mem_univ 0⟩ (fun j : Fin (d + 1) => -(u j)) :=
      Finset.le_sup' (fun j : Fin (d + 1) => -(u j)) (Finset.mem_univ j)
    linarith
  have hsum0 : ∑ j : Fin (d + 1), u j = 0 :=
    le_antisymm hsum (Finset.sum_nonneg fun j _ => hnn j)
  have hall0 : ∀ j : Fin (d + 1), u j = 0 := fun j =>
    (Finset.sum_eq_zero_iff_of_nonneg (fun i _ => hnn i)).mp hsum0 j (Finset.mem_univ j)
  exact hu (funext hall0)

lemma simplexGauge_smul {d : ℕ} {c : ℝ} (hc : 0 < c) (u : Fin (d + 1) → ℝ) :
    simplexGauge (c • u) = c * simplexGauge u := by
  unfold simplexGauge
  have h1 : ∑ j : Fin (d + 1), (c • u) j = c * ∑ j : Fin (d + 1), u j := by
    simp only [Pi.smul_apply, smul_eq_mul, Finset.mul_sum]
  have h2 : Finset.univ.sup' ⟨0, Finset.mem_univ 0⟩ (fun j : Fin (d + 1) => -((c • u) j)) =
      c * Finset.univ.sup' ⟨0, Finset.mem_univ 0⟩ (fun j : Fin (d + 1) => -(u j)) := by
    apply le_antisymm
    · refine Finset.sup'_le _ _ fun j _ => ?_
      have hj := Finset.le_sup' (fun j : Fin (d + 1) => -(u j)) (Finset.mem_univ j)
      have hneg : -((c • u) j) = c * -(u j) := by simp [smul_eq_mul]
      rw [hneg]
      exact mul_le_mul_of_nonneg_left hj (le_of_lt hc)
    · obtain ⟨j₀, _, hj₀⟩ := Finset.exists_mem_eq_sup' ⟨0, Finset.mem_univ (0 : Fin (d + 1))⟩
        (fun j : Fin (d + 1) => -(u j))
      rw [hj₀]
      have hle := Finset.le_sup' (fun j : Fin (d + 1) => -((c • u) j)) (Finset.mem_univ j₀)
      have hneg : -((c • u) j₀) = c * -(u j₀) := by simp [smul_eq_mul]
      linarith
  rw [h1, h2, mul_max_of_nonneg _ _ (le_of_lt hc)]

lemma continuous_simplexGauge (d : ℕ) : Continuous (@simplexGauge d) := by
  unfold simplexGauge
  refine Continuous.max (continuous_finset_sum _ fun j _ => continuous_apply j) ?_
  -- `Finset.sup'` over a finite set of continuous functions is continuous
  suffices h : ∀ (s : Finset (Fin (d + 1))) (hs : s.Nonempty),
      Continuous (fun u : Fin (d + 1) → ℝ => s.sup' hs (fun j => -(u j))) from
    h Finset.univ ⟨0, Finset.mem_univ 0⟩
  intro s hs
  refine Finset.Nonempty.cons_induction (s := s) (hs := hs) ?_ ?_
  · intro a
    simpa using (continuous_apply a).neg
  · intro a t hat ht ih
    have heq : (fun u : Fin (d + 1) → ℝ => (Finset.cons a t hat).sup' (Finset.cons_nonempty hat) (fun j => -(u j))) =
        fun u => max (-(u a)) (t.sup' ht (fun j => -(u j))) := by
      funext u; exact Finset.sup'_cons ht (fun j => -(u j))
    rw [heq]
    exact (continuous_apply a).neg.max ih

lemma simplexGauge_sub_barycenter_of_mem {d : ℕ} {z : Fin (d + 1) → ℝ}
    (hz : z ∈ boundarySimplex d) :
    simplexGauge (z - barycenter d) = 1 / ((d : ℝ) + 2) := by
  rcases hz with ⟨hnn, hsum_le, hbdry⟩
  have hsum_b : ∑ j : Fin (d + 1), barycenter d j = ((d : ℝ) + 1) / ((d : ℝ) + 2) := by
    simp [barycenter, Finset.sum_const, nsmul_eq_mul, div_eq_mul_inv]
  have hsum_diff : ∑ j : Fin (d + 1), (z - barycenter d) j =
      (∑ j : Fin (d + 1), z j) - ((d : ℝ) + 1) / ((d : ℝ) + 2) := by
    simp only [Pi.sub_apply, Finset.sum_sub_distrib, hsum_b]
  have hd2_pos : (0 : ℝ) < (d : ℝ) + 2 := by positivity
  have h1_le : ∑ j : Fin (d + 1), (z - barycenter d) j ≤ 1 / ((d : ℝ) + 2) := by
    rw [hsum_diff]
    have : 1 - ((d : ℝ) + 1) / ((d : ℝ) + 2) = 1 / ((d : ℝ) + 2) := by
      field_simp [ne_of_gt hd2_pos]; ring
    linarith
  have h2_le : Finset.univ.sup' ⟨0, Finset.mem_univ 0⟩ (fun j : Fin (d + 1) => -((z - barycenter d) j)) ≤
      1 / ((d : ℝ) + 2) := by
    refine Finset.sup'_le _ _ fun j _ => ?_
    simp only [Pi.sub_apply, barycenter]
    linarith [hnn j]
  apply le_antisymm
  · exact max_le h1_le h2_le
  · rcases hbdry with ⟨j₀, hj₀⟩ | hsum1
    · refine le_trans ?_ (le_max_right _ _)
      have hle := Finset.le_sup' (fun j : Fin (d + 1) => -((z - barycenter d) j)) (Finset.mem_univ j₀)
      have heq : -((z - barycenter d) j₀) = 1 / ((d : ℝ) + 2) := by
        simp [Pi.sub_apply, barycenter, hj₀]
      linarith
    · refine le_trans ?_ (le_max_left _ _)
      rw [hsum_diff, hsum1]
      have : 1 - ((d : ℝ) + 1) / ((d : ℝ) + 2) = 1 / ((d : ℝ) + 2) := by
        field_simp [ne_of_gt hd2_pos]; ring
      linarith

lemma sub_barycenter_ne_zero {d : ℕ} {z : Fin (d + 1) → ℝ} (hz : z ∈ boundarySimplex d) :
    z - barycenter d ≠ 0 := by
  intro h0
  have hsg := simplexGauge_sub_barycenter_of_mem hz
  rw [h0] at hsg
  have hzero : simplexGauge (0 : Fin (d + 1) → ℝ) = 0 := by
    unfold simplexGauge
    have hsup : Finset.univ.sup' ⟨0, Finset.mem_univ 0⟩ (fun _ : Fin (d + 1) => -(0 : ℝ)) = 0 := by
      apply le_antisymm
      · refine Finset.sup'_le _ _ fun _ _ => by norm_num
      · exact le_trans (by norm_num)
          (Finset.le_sup' (fun _ : Fin (d + 1) => -(0 : ℝ)) (Finset.mem_univ 0))
    simp
  rw [hzero] at hsg
  have hd2_pos : (0 : ℝ) < (d : ℝ) + 2 := by positivity
  have : (0 : ℝ) < 1 / ((d : ℝ) + 2) := div_pos one_pos hd2_pos
  linarith

/-- **Explicit Homeomorphism (`boundarySimplex d ≃ₜ S^d`)**:
For every `d : ℕ`, the boundary `boundarySimplex d = ∂Δ₀^{d+1}` is homeomorphic to the
Euclidean unit sphere `Metric.sphere (0 : EuclideanSpace ℝ (Fin (d + 1))) 1` via radial
projection from the simplex barycenter `c = (1/(d+2), …, 1/(d+2))`. -/
noncomputable def boundarySimplex_homeomorph_sphere (d : ℕ) :
    boundarySimplex d ≃ₜ Metric.sphere (0 : EuclideanSpace ℝ (Fin (d + 1))) 1 := by
  let toE_lin : (Fin (d + 1) → ℝ) ≃L[ℝ] EuclideanSpace ℝ (Fin (d + 1)) :=
    (EuclideanSpace.equiv (Fin (d + 1)) ℝ).symm
  let toE : (Fin (d + 1) → ℝ) ≃ₜ EuclideanSpace ℝ (Fin (d + 1)) := toE_lin.toHomeomorph
  have htoE_norm_pos : ∀ z : boundarySimplex d, 0 < ‖toE (z.1 - barycenter d)‖ := by
    intro z
    rw [norm_pos_iff]
    intro h0
    apply sub_barycenter_ne_zero z.2
    have := congrArg toE.symm h0
    simpa [toE, toE_lin] using this
  have hfrom_ne : ∀ u : Metric.sphere (0 : EuclideanSpace ℝ (Fin (d + 1))) 1,
      (toE.symm u.1 : Fin (d + 1) → ℝ) ≠ 0 := by
    intro u h0
    have hu1 : ‖u.1‖ = 1 := by rw [← dist_zero_right]; exact Metric.mem_sphere.mp u.2
    have hu0 : u.1 = 0 := by
      have := congrArg toE h0
      simpa [toE, toE_lin] using this
    rw [hu0, norm_zero] at hu1
    norm_num at hu1
  have htoE_smul : ∀ (a : ℝ) (v : Fin (d + 1) → ℝ), toE (a • v) = a • toE v :=
    fun a v => toE_lin.map_smul a v
  have hfrom_mem : ∀ u : Metric.sphere (0 : EuclideanSpace ℝ (Fin (d + 1))) 1,
      barycenter d + ((((d : ℝ) + 2) * simplexGauge (toE.symm u.1))⁻¹) • (toE.symm u.1) ∈
        boundarySimplex d := by
    intro u
    set v : Fin (d + 1) → ℝ := toE.symm u.1
    have hsg_pos : 0 < simplexGauge v := simplexGauge_pos (hfrom_ne u)
    have hd2_pos : (0 : ℝ) < (d : ℝ) + 2 := by positivity
    set s : ℝ := (((d : ℝ) + 2) * simplexGauge v)⁻¹
    have hs_pos : 0 < s := inv_pos.mpr (mul_pos hd2_pos hsg_pos)
    have hs_sg : s * simplexGauge v = 1 / ((d : ℝ) + 2) := by
      dsimp [s]; field_simp [ne_of_gt hd2_pos, ne_of_gt hsg_pos]
    refine ⟨?_, ?_, ?_⟩
    · intro j
      have hj : -(v j) ≤ simplexGauge v :=
        le_trans (Finset.le_sup' (fun j => -(v j)) (Finset.mem_univ j)) (le_max_right _ _)
      have hmul : s * (-(v j)) ≤ 1 / ((d : ℝ) + 2) := by
        calc s * (-(v j)) ≤ s * simplexGauge v := mul_le_mul_of_nonneg_left hj (le_of_lt hs_pos)
          _ = 1 / ((d : ℝ) + 2) := hs_sg
      simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, barycenter]
      linarith
    · have hsum_v : ∑ j : Fin (d + 1), v j ≤ simplexGauge v := le_max_left _ _
      have hsum_b : ∑ j : Fin (d + 1), barycenter d j = ((d : ℝ) + 1) / ((d : ℝ) + 2) := by
        simp [barycenter, Finset.sum_const, nsmul_eq_mul, div_eq_mul_inv]
      have hmul : s * ∑ j : Fin (d + 1), v j ≤ 1 / ((d : ℝ) + 2) := by
        calc s * ∑ j : Fin (d + 1), v j ≤ s * simplexGauge v :=
              mul_le_mul_of_nonneg_left hsum_v (le_of_lt hs_pos)
          _ = 1 / ((d : ℝ) + 2) := hs_sg
      have htot : ∑ j : Fin (d + 1), (barycenter d + s • v) j =
          ((d : ℝ) + 1) / ((d : ℝ) + 2) + s * ∑ j : Fin (d + 1), v j := by
        simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, Finset.sum_add_distrib, hsum_b,
          ← Finset.mul_sum]
      rw [htot]
      have h1 : ((d : ℝ) + 1) / ((d : ℝ) + 2) + 1 / ((d : ℝ) + 2) = 1 := by
        field_simp [ne_of_gt hd2_pos]; ring
      linarith
    · -- Either simplexGauge v = ∑ v j (giving sum = 1) or = sup' (-v j) (giving some coord = 0)
      rcases max_choice (∑ j : Fin (d + 1), v j)
          (Finset.univ.sup' ⟨0, Finset.mem_univ 0⟩ (fun j : Fin (d + 1) => -(v j))) with hmax | hmax
      · right
        have hsum_eq : ∑ j : Fin (d + 1), v j = simplexGauge v := hmax.symm
        have hsum_b : ∑ j : Fin (d + 1), barycenter d j = ((d : ℝ) + 1) / ((d : ℝ) + 2) := by
          simp [barycenter, Finset.sum_const, nsmul_eq_mul, div_eq_mul_inv]
        have htot : ∑ j : Fin (d + 1), (barycenter d + s • v) j =
            ((d : ℝ) + 1) / ((d : ℝ) + 2) + s * ∑ j : Fin (d + 1), v j := by
          simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, Finset.sum_add_distrib, hsum_b,
            ← Finset.mul_sum]
        rw [htot, hsum_eq, hs_sg]
        field_simp [ne_of_gt hd2_pos]; ring
      · left
        obtain ⟨j₀, _, hj₀⟩ := Finset.exists_mem_eq_sup' ⟨0, Finset.mem_univ (0 : Fin (d + 1))⟩
          (fun j : Fin (d + 1) => -(v j))
        refine ⟨j₀, ?_⟩
        have hneg_eq : -(v j₀) = simplexGauge v := hj₀.symm.trans hmax.symm
        simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, barycenter]
        have hprod : s * v j₀ = -(1 / ((d : ℝ) + 2)) := by
          have : s * (-(v j₀)) = 1 / ((d : ℝ) + 2) := by rw [hneg_eq, hs_sg]
          linarith
        linarith
  refine
    { toFun := fun z =>
        ⟨‖toE (z.1 - barycenter d)‖⁻¹ • toE (z.1 - barycenter d), by
          rw [Metric.mem_sphere, dist_zero_right, norm_smul, Real.norm_eq_abs,
            abs_of_pos (inv_pos.mpr (htoE_norm_pos z)),
            inv_mul_cancel₀ (ne_of_gt (htoE_norm_pos z))]⟩
      invFun := fun u =>
        ⟨barycenter d + ((((d : ℝ) + 2) * simplexGauge (toE.symm u.1))⁻¹) • (toE.symm u.1),
          hfrom_mem u⟩
      left_inv := fun z => by
        apply Subtype.ext
        set v : Fin (d + 1) → ℝ := z.1 - barycenter d
        set r : ℝ := ‖toE v‖
        have hr_pos : 0 < r := htoE_norm_pos z
        have hd2_pos : (0 : ℝ) < (d : ℝ) + 2 := by positivity
        have hsymm : toE.symm (r⁻¹ • toE v) = r⁻¹ • v := by
          rw [← htoE_smul, toE.symm_apply_apply]
        dsimp only
        rw [hsymm, simplexGauge_smul (inv_pos.mpr hr_pos) v, simplexGauge_sub_barycenter_of_mem z.2]
        have hcoef : (((d : ℝ) + 2) * (r⁻¹ * (1 / ((d : ℝ) + 2))))⁻¹ = r := by
          field_simp [ne_of_gt hd2_pos, ne_of_gt hr_pos]
        rw [hcoef, smul_smul, mul_inv_cancel₀ (ne_of_gt hr_pos), one_smul]
        ext j; simp [v]
      right_inv := fun u => by
        apply Subtype.ext
        set v : Fin (d + 1) → ℝ := toE.symm u.1
        have hsg_pos : 0 < simplexGauge v := simplexGauge_pos (hfrom_ne u)
        have hd2_pos : (0 : ℝ) < (d : ℝ) + 2 := by positivity
        set s : ℝ := (((d : ℝ) + 2) * simplexGauge v)⁻¹
        have hs_pos : 0 < s := inv_pos.mpr (mul_pos hd2_pos hsg_pos)
        have hsub : barycenter d + s • v - barycenter d = s • v := by
          ext j; simp
        have htoE_v : toE v = u.1 := toE.apply_symm_apply u.1
        have hu1 : ‖u.1‖ = 1 := by rw [← dist_zero_right]; exact Metric.mem_sphere.mp u.2
        dsimp only
        rw [hsub, htoE_smul, htoE_v, norm_smul, Real.norm_eq_abs, abs_of_pos hs_pos, hu1, mul_one,
          smul_smul, inv_mul_cancel₀ (ne_of_gt hs_pos), one_smul]
      continuous_toFun := by
        refine Continuous.subtype_mk ?_ _
        have h1 : Continuous (fun z : boundarySimplex d => toE (z.1 - barycenter d)) :=
          toE.continuous.comp (continuous_subtype_val.sub continuous_const)
        refine Continuous.smul ?_ h1
        exact Continuous.inv₀ (continuous_norm.comp h1) fun z => ne_of_gt (htoE_norm_pos z)
      continuous_invFun := by
        refine Continuous.subtype_mk ?_ _
        have h1 : Continuous (fun u : Metric.sphere (0 : EuclideanSpace ℝ (Fin (d + 1))) 1 =>
            toE.symm u.1) := toE.symm.continuous.comp continuous_subtype_val
        have h2 : Continuous (fun u : Metric.sphere (0 : EuclideanSpace ℝ (Fin (d + 1))) 1 =>
            (((d : ℝ) + 2) * simplexGauge (toE.symm u.1))⁻¹) := by
          refine Continuous.inv₀ (continuous_const.mul ((continuous_simplexGauge d).comp h1)) ?_
          intro u
          have hd2_pos : (0 : ℝ) < (d : ℝ) + 2 := by positivity
          exact ne_of_gt (mul_pos hd2_pos (simplexGauge_pos (hfrom_ne u)))
        exact continuous_const.add (h2.smul h1) }

/-! ### Part B Completion: Geometric Impossibility for All Euclidean Spheres and `∂Δ₀^{d+1}` -/

/-- Helper: if `n ≠ 0`, every Euclidean open ball `{x | ∑ j, (x j - c j)^2 < r^2}` (`r > 0`)
contains a point `p ∉ reluVec '' hyperplane n α`. -/
lemma exists_ball_not_mem_relu_hyperplane {d : ℕ} {n : Fin (d + 1) → ℝ} (hn : n ≠ 0) (α : ℝ)
    (c : Fin (d + 1) → ℝ) {r : ℝ} (hr : 0 < r) :
    ∃ p : Fin (d + 1) → ℝ,
      (∑ j : Fin (d + 1), (p j - c j) ^ 2 < r ^ 2) ∧ p ∉ reluVec '' hyperplane n α := by
  by_cases hneg : ∃ k : Fin (d + 1), c k ≤ 0
  · obtain ⟨k, hck⟩ := hneg
    let p : Fin (d + 1) → ℝ := c + (r / 2) • (-stdBasis k)
    refine ⟨p, ?_, ?_⟩
    · have hsum : ∑ j : Fin (d + 1), (p j - c j) ^ 2 = (r / 2) ^ 2 := by
        have hterm : ∀ j : Fin (d + 1), (p j - c j) ^ 2 = if j = k then (r / 2) ^ 2 else 0 := by
          intro j; dsimp [p, stdBasis]; split_ifs <;> ring
        simp [hterm]
      rw [hsum]; nlinarith
    · rintro ⟨u, _, hup⟩
      have hnn : 0 ≤ reluVec u k := le_max_left 0 (u k)
      have hpk : p k = c k - r / 2 := by simp [p, stdBasis]; ring
      rw [hup, hpk] at hnn
      linarith
  · push_neg at hneg
    -- All c j > 0! Pick k with n k ≠ 0 and tiny δ > 0.
    have hk_ex : ∃ k : Fin (d + 1), n k ≠ 0 := by
      by_contra hall; push_neg at hall; exact hn (funext hall)
    obtain ⟨k, hnk⟩ := hk_ex
    by_cases hc_mem : c ∈ reluVec '' hyperplane n α
    · -- Since c j > 0 for all j, c ∈ reluVec '' H forces ∑ n_i c_i = α.
      -- Shift c by +(r / 2) • e_k: still all coords > 0, in ball, and ∑ n_i p_i = α + (r/2) n_k ≠ α!
      let p : Fin (d + 1) → ℝ := c + (r / 2) • stdBasis k
      have hnc : ∑ i : Fin (d + 1), n i * c i = α := by
        rcases hc_mem with ⟨u, huH, huc⟩
        have hu_eq : u = c := by
          funext j
          have hru : relu (u j) = c j := congrFun huc j
          have hcj : 0 < c j := hneg j
          by_contra hne
          have hle : u j ≤ 0 := by
            by_contra hgt; push_neg at hgt
            rw [relu, max_eq_right (le_of_lt hgt)] at hru
            exact hne hru
          rw [relu, max_eq_left hle] at hru
          linarith
        rwa [← hu_eq]
      refine ⟨p, ?_, ?_⟩
      · have hsum : ∑ j : Fin (d + 1), (p j - c j) ^ 2 = (r / 2) ^ 2 := by
          have hterm : ∀ j : Fin (d + 1), (p j - c j) ^ 2 = if j = k then (r / 2) ^ 2 else 0 := by
            intro j; dsimp [p, stdBasis]; split_ifs <;> ring
          simp [hterm]
        rw [hsum]; nlinarith
      · rintro ⟨u, huH, hup⟩
        have hu_eq : u = p := by
          funext j
          have hpj : 0 < p j := by dsimp [p, stdBasis]; split_ifs <;> linarith [hneg j]
          have hru : relu (u j) = p j := congrFun hup j
          by_contra hne
          have hle : u j ≤ 0 := by
            by_contra hgt; push_neg at hgt
            rw [relu, max_eq_right (le_of_lt hgt)] at hru
            exact hne hru
          rw [relu, max_eq_left hle] at hru
          linarith
        have hsum_p : ∑ i : Fin (d + 1), n i * p i = (∑ i, n i * c i) + (r / 2) * n k := by
          have h1 : ∑ i : Fin (d + 1), n i * (c i + (r / 2) * stdBasis k i) =
              (∑ i, n i * c i) + (r / 2) * (∑ i, n i * stdBasis k i) := by
            rw [Finset.mul_sum, ← Finset.sum_add_distrib]
            refine Finset.sum_congr rfl fun i _ => by ring
          simpa [p, Pi.add_apply, Pi.smul_apply, smul_eq_mul, sum_mul_stdBasis] using h1
        have huEq : ∑ i : Fin (d + 1), n i * u i = α := huH
        rw [hu_eq, hsum_p, hnc] at huEq
        have hprod0 : (r / 2) * n k = 0 := by linarith
        rcases mul_eq_zero.mp hprod0 with h | h
        · linarith
        · exact hnk h
    · exact ⟨c, by simp [hr], hc_mem⟩

/-- **Part B Corollary 1 (No `(d, d, d+1)` ReLU Network Can Cover Any Euclidean Sphere)**:
For EVERY `d : ℕ`, EVERY bottleneck layer `(W₂, b₂)` (`(·, d, d+1)`), EVERY center `c`, and
EVERY radius `r > 0`, the Euclidean sphere `{z | ∑ j, (z j - c j)^2 = r^2}` is NOT contained in
the range of `h ↦ reluVec (W₂.mulVec h + b₂)`! -/
theorem bottleneck_not_superset_sphere {d : ℕ} (W₂ : Matrix (Fin (d + 1)) (Fin d) ℝ)
    (b₂ : Fin (d + 1) → ℝ) (c : Fin (d + 1) → ℝ) {r : ℝ} (hr : 0 < r) :
    ¬ ({z : Fin (d + 1) → ℝ | ∑ j : Fin (d + 1), (z j - c j) ^ 2 = r ^ 2} ⊆
        Set.range (fun h : Fin d → ℝ => reluVec (W₂.mulVec h + b₂))) := by
  intro hsub
  obtain ⟨n, hn, hmem⟩ := bottleneck_relu_in_relu_hyperplane W₂ b₂
  set α : ℝ := ∑ i : Fin (d + 1), n i * b₂ i
  obtain ⟨p, hp_ball, hp_not⟩ := exists_ball_not_mem_relu_hyperplane hn α c hr
  obtain ⟨v, hv_ne, hray⟩ := straight_ray_escape hn α hp_not
  -- Along the ray q(t) = p + t • v, Q(t) = ∑ (q(t)_j - c_j)^2 starts < r^2 at t = 0 and → +∞!
  let Q : ℝ → ℝ := fun t => ∑ j : Fin (d + 1), (p j + t * v j - c j) ^ 2
  have hQ_cont : Continuous Q :=
    continuous_finset_sum _ fun j _ =>
      ((continuous_const.add (continuous_id.mul continuous_const)).sub continuous_const).pow 2
  have hQ0 : Q 0 < r ^ 2 := by simpa [Q] using hp_ball
  -- Pick k with v k ≠ 0, so Q(T) ≥ (p k + T * v k - c k)^2 ≥ r^2 for large T > 0
  have hk_ex : ∃ k : Fin (d + 1), v k ≠ 0 := by
    by_contra hall; push_neg at hall; exact hv_ne (funext hall)
  obtain ⟨k, hvk⟩ := hk_ex
  let T : ℝ := (|p k - c k| + r) / |v k| + 1
  have hT_pos : 0 < T := by positivity
  have hQT : r ^ 2 ≤ Q T := by
    have hsingle : (p k + T * v k - c k) ^ 2 ≤ Q T :=
      Finset.single_le_sum (f := fun j => (p j + T * v j - c j) ^ 2)
        (fun j _ => sq_nonneg _) (Finset.mem_univ k)
    have hTvk : |p k - c k| + r ≤ |T * v k| := by
      rw [abs_mul, abs_of_pos hT_pos]
      have h1 : (|p k - c k| + r) / |v k| * |v k| = |p k - c k| + r :=
        div_mul_cancel₀ _ (abs_ne_zero.mpr hvk)
      dsimp [T]
      nlinarith [abs_pos.mpr hvk]
    have habs : r ≤ |p k + T * v k - c k| := by
      have htri : |T * v k| ≤ |T * v k + (p k - c k)| + |p k - c k| := by
        have h1 := abs_add_le (T * v k + (p k - c k)) (-(p k - c k))
        have h2 : T * v k + (p k - c k) + -(p k - c k) = T * v k := by ring
        rw [h2, abs_neg] at h1
        exact h1
      have heq : T * v k + (p k - c k) = p k + T * v k - c k := by ring
      rw [heq] at htri
      linarith
    have hsq : r ^ 2 ≤ (p k + T * v k - c k) ^ 2 := by
      rw [← sq_abs (p k + T * v k - c k)]
      nlinarith [abs_nonneg (p k + T * v k - c k)]
    exact le_trans hsq hsingle
  -- By IVT on [0, T], ∃ t ∈ [0, T] with Q t = r^2!
  have hr2_mem : r ^ 2 ∈ Set.Icc (Q 0) (Q T) := ⟨le_of_lt hQ0, hQT⟩
  obtain ⟨t, ht_mem, hQt⟩ := intermediate_value_Icc (le_of_lt hT_pos) hQ_cont.continuousOn hr2_mem
  have hqt_sphere : p + t • v ∈ {z : Fin (d + 1) → ℝ | ∑ j : Fin (d + 1), (z j - c j) ^ 2 = r ^ 2} := by
    simpa [Q] using hQt
  obtain ⟨h, hh⟩ := hsub hqt_sphere
  exact hray t ht_mem.1 (hh ▸ hmem h)

/-- **Part B Corollary 2 (Direct Contrast with Part A: No `(d, d, d+1)` ReLU Network Can Cover `∂Δ₀^{d+1}`)**:
For EVERY `d : ℕ` and EVERY bottleneck layer `(W₂, b₂)` (`(·, d, d+1)`), `boundarySimplex d` is
NOT contained in the range of `h ↦ reluVec (W₂.mulVec h + b₂)`! -/
theorem bottleneck_not_superset_boundarySimplex {d : ℕ} (W₂ : Matrix (Fin (d + 1)) (Fin d) ℝ)
    (b₂ : Fin (d + 1) → ℝ) :
    ¬ (boundarySimplex d ⊆ Set.range (fun h : Fin d → ℝ => reluVec (W₂.mulVec h + b₂))) := by
  intro hsub
  obtain ⟨n, hn, hmem⟩ := bottleneck_relu_in_relu_hyperplane W₂ b₂
  set α : ℝ := ∑ i : Fin (d + 1), n i * b₂ i
  have hd2_pos : (0 : ℝ) < (d : ℝ) + 2 := by positivity
  -- Choose radius r = 1 / (2 * (d + 2)^2) so the entire open ball around barycenter d lies strictly inside Δ₀^{d+1}!
  set r : ℝ := 1 / (2 * ((d : ℝ) + 2) ^ 2)
  have hr_pos : 0 < r := by positivity
  obtain ⟨p, hp_ball, hp_not⟩ := exists_ball_not_mem_relu_hyperplane hn α (barycenter d) hr_pos
  have hp_coord_diff : ∀ j : Fin (d + 1), |p j - barycenter d j| < r := by
    intro j
    have hsq : (p j - barycenter d j) ^ 2 < r ^ 2 :=
      lt_of_le_of_lt
        (Finset.single_le_sum (f := fun i => (p i - barycenter d i) ^ 2)
          (fun i _ => sq_nonneg _) (Finset.mem_univ j))
        hp_ball
    exact abs_lt_of_sq_lt_sq hsq (le_of_lt hr_pos)
  have hp_pos : ∀ j : Fin (d + 1), 0 < p j := by
    intro j
    have h1 := (abs_lt.mp (hp_coord_diff j)).1
    have hr_lt : r < 1 / ((d : ℝ) + 2) := by
      dsimp [r]
      rw [div_lt_div_iff₀ (by positivity) hd2_pos]
      nlinarith
    simp only [barycenter] at h1
    linarith
  have hp_sum_lt : ∑ j : Fin (d + 1), p j < 1 := by
    have hsum_b : ∑ j : Fin (d + 1), barycenter d j = ((d : ℝ) + 1) / ((d : ℝ) + 2) := by
      simp [barycenter, Finset.sum_const, nsmul_eq_mul, div_eq_mul_inv]
    have hdiff_sum : ∑ j : Fin (d + 1), (p j - barycenter d j) < ((d : ℝ) + 1) * r := by
      calc ∑ j : Fin (d + 1), (p j - barycenter d j)
          < ∑ _j : Fin (d + 1), r :=
            Finset.sum_lt_sum_of_nonempty ⟨0, Finset.mem_univ 0⟩ fun j _ => (abs_lt.mp (hp_coord_diff j)).2
        _ = ((d : ℝ) + 1) * r := by simp [Finset.sum_const, nsmul_eq_mul]
    rw [Finset.sum_sub_distrib, hsum_b] at hdiff_sum
    have hr_sum_lt : ((d : ℝ) + 1) * r < 1 / ((d : ℝ) + 2) := by
      dsimp [r]
      have hmul : ((d : ℝ) + 1) * (1 / (2 * ((d : ℝ) + 2) ^ 2)) =
          ((d : ℝ) + 1) / (2 * ((d : ℝ) + 2) ^ 2) := by ring
      rw [hmul, div_lt_div_iff₀ (by positivity) hd2_pos]
      nlinarith
    have h1 : ((d : ℝ) + 1) / ((d : ℝ) + 2) + 1 / ((d : ℝ) + 2) = 1 := by
      field_simp [ne_of_gt hd2_pos]; ring
    linarith
  -- Apply Straight-Ray Escape from p!
  obtain ⟨v, hv_ne, hray⟩ := straight_ray_escape hn α hp_not
  -- Define signed distance to ∂Δ₀^{d+1}: f(t) = min (1 - ∑ (p + t•v)) (inf_j (p + t•v)_j)
  let f : ℝ → ℝ := fun t =>
    min (1 - ∑ j : Fin (d + 1), (p j + t * v j))
      (Finset.univ.inf' ⟨0, Finset.mem_univ 0⟩ (fun j : Fin (d + 1) => p j + t * v j))
  have hf_cont : Continuous f := by
    refine Continuous.min ?_ ?_
    · exact continuous_const.sub
        (continuous_finset_sum _ fun j _ => continuous_const.add (continuous_id.mul continuous_const))
    · suffices h : ∀ (s : Finset (Fin (d + 1))) (hs : s.Nonempty),
          Continuous (fun t : ℝ => s.inf' hs (fun j => p j + t * v j)) from
        h Finset.univ ⟨0, Finset.mem_univ 0⟩
      intro s hs
      refine Finset.Nonempty.cons_induction (s := s) (hs := hs) ?_ ?_
      · intro a; exact continuous_const.add (continuous_id.mul continuous_const)
      · intro a s' has' hs' ih
        have heq : (fun t : ℝ => (Finset.cons a s' has').inf' (Finset.cons_nonempty has')
            (fun j => p j + t * v j)) = fun t => min (p a + t * v a) (s'.inf' hs' (fun j => p j + t * v j)) := by
          funext t; exact Finset.inf'_cons hs' (fun j => p j + t * v j)
        rw [heq]
        exact (continuous_const.add (continuous_id.mul continuous_const)).min ih
  have hf0 : 0 < f 0 := by
    dsimp [f]
    refine lt_min (by simpa using sub_pos.mpr hp_sum_lt) ?_
    rw [Finset.lt_inf'_iff]
    intro j _
    simpa using hp_pos j
  -- Since simplexGauge v > 0, for large T > 0 we have f T < 0!
  have hsg_pos : 0 < simplexGauge v := simplexGauge_pos hv_ne
  let T : ℝ := 2 / simplexGauge v
  have hT_pos : 0 < T := div_pos (by norm_num) hsg_pos
  have hT_sg : T * simplexGauge v = 2 := div_mul_cancel₀ 2 (ne_of_gt hsg_pos)
  have hfT : f T ≤ 0 := by
    rcases max_choice (∑ j : Fin (d + 1), v j)
        (Finset.univ.sup' ⟨0, Finset.mem_univ 0⟩ (fun j : Fin (d + 1) => -(v j))) with hmax | hmax
    · refine le_trans (min_le_left _ _) ?_
      have hsum_v : ∑ j : Fin (d + 1), v j = simplexGauge v := hmax.symm
      have hsum_pT : ∑ j : Fin (d + 1), (p j + T * v j) =
          (∑ j, p j) + T * ∑ j, v j := by
        rw [Finset.sum_add_distrib, ← Finset.mul_sum]
      rw [hsum_pT, hsum_v, hT_sg]
      have hsum_nn : 0 ≤ ∑ j : Fin (d + 1), p j := Finset.sum_nonneg fun j _ => le_of_lt (hp_pos j)
      linarith
    · refine le_trans (min_le_right _ _) ?_
      obtain ⟨j₀, _, hj₀⟩ := Finset.exists_mem_eq_sup' ⟨0, Finset.mem_univ (0 : Fin (d + 1))⟩
        (fun j : Fin (d + 1) => -(v j))
      have hneg_eq : -(v j₀) = simplexGauge v := hj₀.symm.trans hmax.symm
      have hle := Finset.inf'_le (fun j : Fin (d + 1) => p j + T * v j) (Finset.mem_univ j₀)
      have hpj_le1 : p j₀ ≤ 1 := by
        have h1 : p j₀ ≤ ∑ j : Fin (d + 1), p j :=
          Finset.single_le_sum (fun j _ => le_of_lt (hp_pos j)) (Finset.mem_univ j₀)
        linarith
      have hTv : T * v j₀ = -2 := by
        have : T * (-(v j₀)) = 2 := by rw [hneg_eq, hT_sg]
        linarith
      linarith
  -- By IVT on [0, T], ∃ t ∈ [0, T] with f t = 0, i.e., p + t • v ∈ boundarySimplex d!
  have h0_mem : (0 : ℝ) ∈ Set.Icc (f T) (f 0) := ⟨hfT, le_of_lt hf0⟩
  obtain ⟨t, ht_mem, hft⟩ := intermediate_value_Icc' (le_of_lt hT_pos) hf_cont.continuousOn h0_mem
  have hqt_bdry : p + t • v ∈ boundarySimplex d := by
    have hmin_ge : 0 ≤ f t := le_of_eq hft.symm
    have hsum_le1 : ∑ j : Fin (d + 1), (p + t • v) j ≤ 1 := by
      have h1 := le_trans hmin_ge (min_le_left _ _)
      have heq : ∑ j : Fin (d + 1), (p + t • v) j = ∑ j, (p j + t * v j) := rfl
      linarith
    have hcoord_nn : ∀ j : Fin (d + 1), 0 ≤ (p + t • v) j := by
      intro j
      have h1 := le_trans hmin_ge (min_le_right _ _)
      have h2 := Finset.inf'_le (fun j : Fin (d + 1) => p j + t * v j) (Finset.mem_univ j)
      exact le_trans h1 h2
    refine ⟨hcoord_nn, hsum_le1, ?_⟩
    rcases min_choice (1 - ∑ j : Fin (d + 1), (p j + t * v j))
        (Finset.univ.inf' ⟨0, Finset.mem_univ 0⟩ (fun j : Fin (d + 1) => p j + t * v j)) with hmin | hmin
    · right
      have h0 : 1 - ∑ j : Fin (d + 1), (p j + t * v j) = 0 := hmin.symm.trans hft
      have heq : ∑ j : Fin (d + 1), (p + t • v) j = ∑ j, (p j + t * v j) := rfl
      linarith
    · left
      obtain ⟨j₀, _, hj₀⟩ := Finset.exists_mem_eq_sup' ⟨0, Finset.mem_univ (0 : Fin (d + 1))⟩
        (fun j : Fin (d + 1) => -(p j + t * v j))
      -- Simpler: use Finset.exists_mem_eq_inf'!
      obtain ⟨j₁, _, hj₁⟩ := Finset.exists_mem_eq_inf' ⟨0, Finset.mem_univ (0 : Fin (d + 1))⟩
        (fun j : Fin (d + 1) => p j + t * v j)
      refine ⟨j₁, ?_⟩
      exact hj₁.symm.trans (hmin.symm.trans hft)
  obtain ⟨h, hh⟩ := hsub hqt_bdry
  exact hray t ht_mem.1 (hh ▸ hmem h)

end BeshkovReluTopology
