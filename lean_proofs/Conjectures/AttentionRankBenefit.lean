import Mathlib

/-!
# Refutation of the All-Length Low-Rank Attention Conjecture
(Noah Amsel, Gilad Yehudai, and Joan Bruna, arXiv:2407.16153 / ICLR 2025, Conjecture 8)

We formalize in Lean 4 + Mathlib (complete kernel-checked proofs) the exact algebraic,
analytic, and topological cores of the 3-layer rank-1 softmax Transformer uniform-in-`N`
approximation refuting Conjecture 8, together with the explicit worst-case `N = 2` lower bound
`≥ 1` scope theorem from `solutions/attention_rank_benefit/solution.md`:

1. **Lemma 1 (Softmax weighted average `B_X(v)`: 1-Lipschitz stability & deterministic cap bound)**:
   - `softmaxW_pos`, `softmaxW_sum_one`, `mu_X_eq_avg`, `B_X_N1_exact`: Exact simplex weights,
     uniform mean `μ_X = B_X(0)`, and exact `N = 1` identity `B_X(v) = x_1`.
   - `B_X_lipschitz` (Lemma 1(2)): For EVERY `N ≥ 1`, EVERY tuple of unit vectors `X ∈ (S^{d-1})^N`,
     and ALL `v, w ∈ ℝ^d`, `‖B_X(v) - B_X(w)‖_2 ≤ ‖v - w‖_2` (proved via 1D Mean Value Theorem
     and Cauchy–Schwarz on the softmax covariance).
   - `softmax_cap_error_bound` (Lemma 1(1) deterministic core): On the high-probability cap event
     `|{i | ⟨x_i, y⟩ ≥ 1 - δ}| ≥ N p / 2` (`δ ∈ (0, 1)`, `p ∈ (0, 1]`, `β > 0`), the maximizing
     candidate `x*` satisfies `‖x* - y‖² ≤ 2δ`, the bad-candidate softmax mass satisfies
     `w ≤ (2/p) exp(-β δ)`, and `‖B_X(β y) - x*‖² ≤ 12 δ + (16/p) exp(-β δ)` uniformly in `N`.
2. **Parity-bit comparator primitive & uniform-in-`N` offset bounds**:
   - `parityComp_eq_sigmoid_and_offset`: For EVERY `N ≥ 2`, `S_N(t) = σ(t + ℓ_N)` with
     `ℓ_N = log(⌈N/2⌉ / ⌊N/2⌋) ∈ [0, log 2]`.
   - `parityComp_uniform_bounds`: Uniformly over ALL `N ≥ 2`, `1 - S_N(t) ≤ exp(-t)` and
     `S_N(t) ≤ 2 exp(t)`.
3. **Theorem 1 & Corollary 3 (3-layer rank-1 Transformer routing & uniform-in-`N` error bound)**:
   - `transformerOut_exact_N1`: At `N = 1`, the 3-layer rank-1 Transformer outputs `x_1 = f(X; y)`
     identically for all weights `(a_1, a_2, β)`.
   - `routing_gate_uniform_in_N`: For any `K ≥ 2` unit directions `{u_1, ..., u_K}` and any query
     `y` with strict winner margin `γ > 0`, explicit finite gains `a_1, a_2` (independent of `N`)
     force `|g_h(N, y) - 1_{h = h*}| ≤ η` simultaneously for ALL `N ≥ 2` and ALL `h ∈ [K]`.
   - `transformerOut_distance_to_B_X`: Using `B_X_lipschitz`, for every `N ≥ 2`,
     `‖T(X, y) - B_X(β y)‖_2 ≤ K β e + β ρ`.
   - `relu_gate_exact` (Corollary 3): Exact identity `min(s, t) = s - ReLU(s - t)` and exact
     one-hot routing `g_h(y) = 1_{h = h*}` for the position-free ReLU MLP gate when `0 < η ≤ γ`.
4. **Proposition 4 (Worst-case-over-inputs `N = 2` lower bound `≥ 1` scope theorem)**:
   - `worst_case_N2_lower_bound`: For EVERY `d ≥ 2`, EVERY continuous map
     `T : (Fin d → ℝ) → (Fin d → ℝ)` (representing `y ↦ T(e_0, -e_0, y)`), and EVERY `ε < 1`,
     there exists a unit query `y ∈ S^{d-1}` with `y 0 ≠ 0` such that
     `ε < ‖T(y) - sgn(y 0) e_0‖_2`.

Machine-checked vs prose-only:
- Machine-checked: All four algebraic, analytic, and topological theorems above for arbitrary
  `d ≥ 2`, `N ≥ 1`, and `K ≥ 2`.
- Prose-only (in `solutions/attention_rank_benefit/proof.pdf`): Rotational invariance and Chebyshev
  tail bound `Pr(M_N < Np/2) ≤ 4/(Np)` on `Unif(S^{d-1})`, dominated convergence for `N < N_0`
  and `E[e_a(y)²] → 0`, and `L²`-Minkowski combination into `sup_{N ≥ 1} E‖T - f‖² < ε`.
-/

open BigOperators Finset Filter Topology

namespace AttentionRankBenefit

/-! ### 1. Euclidean Geometry & Softmax Weighted Average `B_X(v)` (Lemma 1) -/

/-- Euclidean inner product on `ℝ^d`. -/
def innerProd {d : ℕ} (u v : Fin d → ℝ) : ℝ := ∑ j : Fin d, u j * v j

/-- Squared Euclidean norm `‖u‖₂²`. -/
def normSq {d : ℕ} (u : Fin d → ℝ) : ℝ := innerProd u u

/-- Euclidean norm `‖u‖₂`. -/
noncomputable def euclidNorm {d : ℕ} (u : Fin d → ℝ) : ℝ := Real.sqrt (normSq u)

lemma normSq_nonneg {d : ℕ} (u : Fin d → ℝ) : 0 ≤ normSq u := by
  unfold normSq innerProd
  exact Finset.sum_nonneg fun _ _ => mul_self_nonneg _

lemma euclidNorm_nonneg {d : ℕ} (u : Fin d → ℝ) : 0 ≤ euclidNorm u :=
  Real.sqrt_nonneg _

lemma innerProd_comm {d : ℕ} (u v : Fin d → ℝ) : innerProd u v = innerProd v u :=
  Finset.sum_congr rfl fun _ _ => mul_comm _ _

lemma innerProd_add_left {d : ℕ} (u v w : Fin d → ℝ) :
    innerProd (u + v) w = innerProd u w + innerProd v w := by
  unfold innerProd
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun j _ => by simp [add_mul]

lemma innerProd_sub_left {d : ℕ} (u v w : Fin d → ℝ) :
    innerProd (u - v) w = innerProd u w - innerProd v w := by
  unfold innerProd
  rw [← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun j _ => by simp [sub_mul]

lemma innerProd_smul_left {d : ℕ} (c : ℝ) (u v : Fin d → ℝ) :
    innerProd (c • u) v = c * innerProd u v := by
  unfold innerProd
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => by simp; ring

lemma innerProd_le_mul_norm {d : ℕ} (u v : Fin d → ℝ) :
    |innerProd u v| ≤ euclidNorm u * euclidNorm v := by
  have hcs : (innerProd u v) ^ 2 ≤ normSq u * normSq v := by
    have h := Finset.sum_mul_sq_le_sq_mul_sq (Finset.univ : Finset (Fin d)) u v
    simpa [innerProd, normSq, sq] using h
  unfold euclidNorm
  rw [← Real.sqrt_sq_eq_abs, ← Real.sqrt_mul (normSq_nonneg u)]
  exact Real.sqrt_le_sqrt hcs

lemma euclidNorm_add_le {d : ℕ} (u v : Fin d → ℝ) :
    euclidNorm (u + v) ≤ euclidNorm u + euclidNorm v := by
  have hsq : normSq (u + v) ≤ (euclidNorm u + euclidNorm v) ^ 2 := by
    have hexp : normSq (u + v) = normSq u + normSq v + 2 * innerProd u v := by
      unfold normSq innerProd
      have h1 : ∀ j : Fin d, (u j + v j) * (u j + v j) =
          u j * u j + v j * v j + 2 * (u j * v j) := fun j => by ring
      simp only [Pi.add_apply]
      rw [Finset.sum_congr rfl fun j _ => h1 j, Finset.sum_add_distrib,
          Finset.sum_add_distrib, ← Finset.mul_sum]
    have hle : innerProd u v ≤ euclidNorm u * euclidNorm v :=
      le_trans (le_abs_self _) (innerProd_le_mul_norm u v)
    unfold euclidNorm at hle
    rw [hexp, add_sq, euclidNorm, euclidNorm, Real.sq_sqrt (normSq_nonneg u),
        Real.sq_sqrt (normSq_nonneg v)]
    linarith
  exact Real.sqrt_le_iff.2 ⟨add_nonneg (euclidNorm_nonneg u) (euclidNorm_nonneg v), hsq⟩

lemma euclidNorm_sub_le {d : ℕ} (u v w : Fin d → ℝ) :
    euclidNorm (u - w) ≤ euclidNorm (u - v) + euclidNorm (v - w) := by
  have heq : u - w = (u - v) + (v - w) := by ext j; simp
  rw [heq]; exact euclidNorm_add_le _ _

lemma euclidNorm_smul {d : ℕ} (c : ℝ) (u : Fin d → ℝ) :
    euclidNorm (c • u) = |c| * euclidNorm u := by
  unfold euclidNorm normSq innerProd
  have h1 : ∑ j : Fin d, (c • u) j * (c • u) j = c ^ 2 * ∑ j : Fin d, u j * u j := by
    rw [Finset.mul_sum]; refine Finset.sum_congr rfl fun j _ => by simp; ring
  rw [h1, Real.sqrt_mul (sq_nonneg c), Real.sqrt_sq_eq_abs]

lemma normSq_sub_unit {d : ℕ} {x y : Fin d → ℝ} (hx : normSq x = 1) (hy : normSq y = 1) :
    normSq (x - y) = 2 - 2 * innerProd x y := by
  have hexp : normSq (x - y) = normSq x + normSq y - 2 * innerProd x y := by
    unfold normSq innerProd
    have h1 : ∀ j : Fin d, (x j - y j) * (x j - y j) =
        x j * x j + y j * y j - 2 * (x j * y j) := fun j => by ring
    simp only [Pi.sub_apply]
    rw [Finset.sum_congr rfl fun j _ => h1 j, Finset.sum_sub_distrib,
        Finset.sum_add_distrib, ← Finset.mul_sum]
  rw [hexp, hx, hy]; ring

/-- Softmax denominator `Z_X(v) = ∑_{i=1}^N exp(⟨v, x_i⟩)`. -/
noncomputable def softmaxDen {N d : ℕ} (X : Fin N → Fin d → ℝ) (v : Fin d → ℝ) : ℝ :=
  ∑ i : Fin N, Real.exp (innerProd v (X i))

/-- Softmax attention weight `p_i(v) = exp(⟨v, x_i⟩) / Z_X(v)`. -/
noncomputable def softmaxW {N d : ℕ} (X : Fin N → Fin d → ℝ) (v : Fin d → ℝ) (i : Fin N) : ℝ :=
  Real.exp (innerProd v (X i)) / softmaxDen X v

/-- Softmax-weighted candidate average `B_X(v) = ∑_{i=1}^N p_i(v) x_i`. -/
noncomputable def B_X {N d : ℕ} (X : Fin N → Fin d → ℝ) (v : Fin d → ℝ) : Fin d → ℝ :=
  fun j => ∑ i : Fin N, softmaxW X v i * X i j

/-- Candidate empirical mean `μ_X = B_X(0)`. -/
noncomputable def mu_X {N d : ℕ} (X : Fin N → Fin d → ℝ) : Fin d → ℝ :=
  B_X X 0

lemma softmaxDen_pos {N d : ℕ} (hN : 1 ≤ N) (X : Fin N → Fin d → ℝ) (v : Fin d → ℝ) :
    0 < softmaxDen X v :=
  Finset.sum_pos (fun _ _ => Real.exp_pos _) ⟨⟨0, hN⟩, Finset.mem_univ _⟩

lemma softmaxW_pos {N d : ℕ} (hN : 1 ≤ N) (X : Fin N → Fin d → ℝ) (v : Fin d → ℝ) (i : Fin N) :
    0 < softmaxW X v i :=
  div_pos (Real.exp_pos _) (softmaxDen_pos hN X v)

lemma softmaxW_sum_one {N d : ℕ} (hN : 1 ≤ N) (X : Fin N → Fin d → ℝ) (v : Fin d → ℝ) :
    ∑ i : Fin N, softmaxW X v i = 1 := by
  unfold softmaxW
  rw [← Finset.sum_div]
  exact div_self (softmaxDen_pos hN X v).ne'

/-- At `N = 1`, `B_X(v) = x_0` identically for EVERY vector `v`. -/
theorem B_X_N1_exact {d : ℕ} (X : Fin 1 → Fin d → ℝ) (v : Fin d → ℝ) :
    B_X X v = X 0 := by
  ext j
  simp only [B_X, softmaxW, softmaxDen, Fin.sum_univ_one]
  rw [div_self (Real.exp_pos _).ne', one_mul]

/-- **Lemma 1(2) (Exact 1-Lipschitz stability of `B_X`)**:
For EVERY `N ≥ 1`, EVERY tuple of unit vectors `X ∈ (S^{d-1})^N`, and ALL `v, w ∈ ℝ^d`,
`‖B_X(v) - B_X(w)‖₂ ≤ ‖v - w‖₂`. -/
theorem B_X_lipschitz {N d : ℕ} (hN : 1 ≤ N) (X : Fin N → Fin d → ℝ)
    (hX : ∀ i, normSq (X i) = 1) (v w : Fin d → ℝ) :
    euclidNorm (B_X X v - B_X X w) ≤ euclidNorm (v - w) := by
  let u : Fin d → ℝ := B_X X v - B_X X w
  let δv : Fin d → ℝ := v - w
  -- 1D function g(t) = ⟨u, B_X(w + t δv)⟩ on [0, 1].
  let g : ℝ → ℝ := fun t => innerProd u (B_X X (w + t • δv))
  have hg_diff : ∀ t : ℝ,
      let p := softmaxW X (w + t • δv)
      let a : Fin N → ℝ := fun i => innerProd u (X i)
      let b : Fin N → ℝ := fun i => innerProd δv (X i)
      HasDerivAt g
        ((∑ i : Fin N, p i * a i * b i) - (∑ i : Fin N, p i * a i) * (∑ i : Fin N, p i * b i)) t := by
    intro t p a b
    -- Express g(t) = num(t) / den(t)
    let num : ℝ → ℝ := fun s => ∑ i : Fin N, Real.exp (innerProd (w + s • δv) (X i)) * a i
    let den : ℝ → ℝ := fun s => ∑ i : Fin N, Real.exp (innerProd (w + s • δv) (X i))
    have hg_eq : g = fun s => num s / den s := by
      ext s
      simp only [g, innerProd, B_X, softmaxW, softmaxDen, num, den]
      rw [Finset.sum_div]
      trans ∑ j : Fin d, ∑ i : Fin N, u j * (Real.exp (innerProd (w + s • δv) (X i)) /
          (∑ k : Fin N, Real.exp (innerProd (w + s • δv) (X k))) * X i j)
      · refine Finset.sum_congr rfl fun j _ => Finset.mul_sum _ _ _
      · rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun i _ => ?_
        simp only [a, innerProd]
        rw [Finset.mul_sum, Finset.sum_div]
        refine Finset.sum_congr rfl fun j _ => by ring
    have hexp_d : ∀ i : Fin N,
        HasDerivAt (fun s => Real.exp (innerProd (w + s • δv) (X i)))
          (Real.exp (innerProd (w + t • δv) (X i)) * b i) t := by
      intro i
      have hlin : HasDerivAt (fun s => innerProd (w + s • δv) (X i)) (b i) t := by
        have h1 : (fun s => innerProd (w + s • δv) (X i)) =
            fun s => innerProd w (X i) + s * b i := by
          ext s; rw [innerProd_add_left, innerProd_smul_left]
        rw [h1]
        simpa using ((hasDerivAt_id t).mul_const (b i)).const_add (innerProd w (X i))
      exact hlin.exp
    have hnum_d : HasDerivAt num
        (∑ i : Fin N, Real.exp (innerProd (w + t • δv) (X i)) * b i * a i) t :=
      HasDerivAt.fun_sum fun i _ => (hexp_d i).mul_const (a i)
    have hden_d : HasDerivAt den
        (∑ i : Fin N, Real.exp (innerProd (w + t • δv) (X i)) * b i) t :=
      HasDerivAt.fun_sum fun i _ => hexp_d i
    have hden_pos : 0 < den t := softmaxDen_pos hN X (w + t • δv)
    rw [hg_eq]
    convert hnum_d.div hden_d hden_pos.ne' using 1
    simp only [p, softmaxW, softmaxDen, num, den]
    have h1 : (∑ i : Fin N, Real.exp (innerProd (w + t • δv) (X i)) /
        (∑ k : Fin N, Real.exp (innerProd (w + t • δv) (X k))) * a i * b i) =
        (∑ i : Fin N, Real.exp (innerProd (w + t • δv) (X i)) * b i * a i) /
        (∑ k : Fin N, Real.exp (innerProd (w + t • δv) (X k))) := by
      rw [Finset.sum_div]; refine Finset.sum_congr rfl fun i _ => by ring
    have h2 : (∑ i : Fin N, Real.exp (innerProd (w + t • δv) (X i)) /
        (∑ k : Fin N, Real.exp (innerProd (w + t • δv) (X k))) * a i) =
        (∑ i : Fin N, Real.exp (innerProd (w + t • δv) (X i)) * a i) /
        (∑ k : Fin N, Real.exp (innerProd (w + t • δv) (X k))) := by
      rw [Finset.sum_div]; refine Finset.sum_congr rfl fun i _ => by ring
    have h3 : (∑ i : Fin N, Real.exp (innerProd (w + t • δv) (X i)) /
        (∑ k : Fin N, Real.exp (innerProd (w + t • δv) (X k))) * b i) =
        (∑ i : Fin N, Real.exp (innerProd (w + t • δv) (X i)) * b i) /
        (∑ k : Fin N, Real.exp (innerProd (w + t • δv) (X k))) := by
      rw [Finset.sum_div]; refine Finset.sum_congr rfl fun i _ => by ring
    rw [h1, h2, h3]
    field_simp [hden_pos.ne']
  -- Bound |g'(t)| ≤ euclidNorm u * euclidNorm δv via covariance Cauchy-Schwarz!
  have hderiv_bound : ∀ t : ℝ,
      |deriv g t| ≤ euclidNorm u * euclidNorm δv := by
    intro t
    rw [(hg_diff t).deriv]
    set p := softmaxW X (w + t • δv)
    set a : Fin N → ℝ := fun i => innerProd u (X i)
    set b : Fin N → ℝ := fun i => innerProd δv (X i)
    set μa := ∑ i : Fin N, p i * a i
    set μb := ∑ i : Fin N, p i * b i
    have hp_pos : ∀ i, 0 < p i := softmaxW_pos hN X (w + t • δv)
    have hp_sum : ∑ i : Fin N, p i = 1 := softmaxW_sum_one hN X (w + t • δv)
    have hcov : (∑ i : Fin N, p i * a i * b i) - μa * μb =
        ∑ i : Fin N, p i * (a i - μa) * (b i - μb) := by
      have hexp : ∀ i : Fin N, p i * (a i - μa) * (b i - μb) =
          p i * a i * b i - μb * (p i * a i) - μa * (p i * b i) + μa * μb * p i := fun i => by ring
      rw [Finset.sum_congr rfl fun i _ => hexp i, Finset.sum_add_distrib,
          Finset.sum_sub_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
          ← Finset.mul_sum, hp_sum]
      simp only [μa, μb]; ring
    rw [hcov]
    have hcs := Finset.sum_mul_sq_le_sq_mul_sq (Finset.univ : Finset (Fin N))
      (fun i => Real.sqrt (p i) * (a i - μa)) (fun i => Real.sqrt (p i) * (b i - μb))
    have hlhs : ∑ i : Fin N, (Real.sqrt (p i) * (a i - μa)) * (Real.sqrt (p i) * (b i - μb)) =
        ∑ i : Fin N, p i * (a i - μa) * (b i - μb) := by
      refine Finset.sum_congr rfl fun i _ => ?_
      have hs := Real.mul_self_sqrt (hp_pos i).le
      calc Real.sqrt (p i) * (a i - μa) * (Real.sqrt (p i) * (b i - μb))
          = (Real.sqrt (p i) * Real.sqrt (p i)) * (a i - μa) * (b i - μb) := by ring
        _ = p i * (a i - μa) * (b i - μb) := by rw [hs]
    have hvar_a : ∑ i : Fin N, (Real.sqrt (p i) * (a i - μa)) ^ 2 ≤ normSq u := by
      have h1 : ∑ i : Fin N, (Real.sqrt (p i) * (a i - μa)) ^ 2 =
          (∑ i : Fin N, p i * (a i) ^ 2) - μa ^ 2 := by
        have h2 : ∀ i : Fin N, (Real.sqrt (p i) * (a i - μa)) ^ 2 =
            p i * (a i) ^ 2 - 2 * μa * (p i * a i) + μa ^ 2 * p i := fun i => by
          rw [mul_pow, Real.sq_sqrt (hp_pos i).le]; ring
        rw [Finset.sum_congr rfl fun i _ => h2 i, Finset.sum_add_distrib, Finset.sum_sub_distrib,
            ← Finset.mul_sum, ← Finset.mul_sum, hp_sum]
        simp only [μa]; ring
      rw [h1]
      have hEa : ∑ i : Fin N, p i * (a i) ^ 2 ≤ normSq u := by
        calc ∑ i : Fin N, p i * (a i) ^ 2
            ≤ ∑ i : Fin N, p i * normSq u := by
              refine Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left ?_ (hp_pos i).le
              have hle := innerProd_le_mul_norm u (X i)
              have hXi : euclidNorm (X i) = 1 := by simp [euclidNorm, hX i]
              rw [hXi, mul_one] at hle
              have hsq : (a i) ^ 2 ≤ (euclidNorm u) ^ 2 := by
                rw [← sq_abs]; exact sq_le_sq' (by linarith [abs_nonneg (a i)]) hle
              simpa [euclidNorm, Real.sq_sqrt (normSq_nonneg u)] using hsq
          _ = normSq u := by rw [← Finset.sum_mul, hp_sum, one_mul]
      linarith [sq_nonneg μa]
    have hvar_b : ∑ i : Fin N, (Real.sqrt (p i) * (b i - μb)) ^ 2 ≤ normSq δv := by
      have h1 : ∑ i : Fin N, (Real.sqrt (p i) * (b i - μb)) ^ 2 =
          (∑ i : Fin N, p i * (b i) ^ 2) - μb ^ 2 := by
        have h2 : ∀ i : Fin N, (Real.sqrt (p i) * (b i - μb)) ^ 2 =
            p i * (b i) ^ 2 - 2 * μb * (p i * b i) + μb ^ 2 * p i := fun i => by
          rw [mul_pow, Real.sq_sqrt (hp_pos i).le]; ring
        rw [Finset.sum_congr rfl fun i _ => h2 i, Finset.sum_add_distrib, Finset.sum_sub_distrib,
            ← Finset.mul_sum, ← Finset.mul_sum, hp_sum]
        simp only [μb]; ring
      rw [h1]
      have hEb : ∑ i : Fin N, p i * (b i) ^ 2 ≤ normSq δv := by
        calc ∑ i : Fin N, p i * (b i) ^ 2
            ≤ ∑ i : Fin N, p i * normSq δv := by
              refine Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left ?_ (hp_pos i).le
              have hle := innerProd_le_mul_norm δv (X i)
              have hXi : euclidNorm (X i) = 1 := by simp [euclidNorm, hX i]
              rw [hXi, mul_one] at hle
              have hsq : (b i) ^ 2 ≤ (euclidNorm δv) ^ 2 := by
                rw [← sq_abs]; exact sq_le_sq' (by linarith [abs_nonneg (b i)]) hle
              simpa [euclidNorm, Real.sq_sqrt (normSq_nonneg δv)] using hsq
          _ = normSq δv := by rw [← Finset.sum_mul, hp_sum, one_mul]
      linarith [sq_nonneg μb]
    rw [hlhs] at hcs
    have hsq_le : (∑ i : Fin N, p i * (a i - μa) * (b i - μb)) ^ 2 ≤ normSq u * normSq δv :=
      le_trans hcs (mul_le_mul hvar_a hvar_b (Finset.sum_nonneg fun _ _ => sq_nonneg _) (normSq_nonneg u))
    rw [← Real.sqrt_sq_eq_abs, euclidNorm, euclidNorm, ← Real.sqrt_mul (normSq_nonneg u)]
    exact Real.sqrt_le_sqrt hsq_le
  -- Apply Mean Value Theorem on [0, 1] to g!
  have hg_cont : ContinuousOn g (Set.Icc 0 1) :=
    fun t _ => (hg_diff t).continuousAt.continuousWithinAt
  have hg_diff_Ioo : DifferentiableOn ℝ g (Set.Ioo 0 1) :=
    fun t _ => (hg_diff t).differentiableAt.differentiableWithinAt
  obtain ⟨c, _, hc⟩ := exists_deriv_eq_slope g zero_lt_one hg_cont hg_diff_Ioo
  have hg1 : g 1 = innerProd u (B_X X v) := by
    simp only [g, one_smul, δv]; congr 1; ext j; simp
  have hg0 : g 0 = innerProd u (B_X X w) := by
    simp only [g, zero_smul, add_zero]
  have hdiff_val : g 1 - g 0 = normSq u := by
    rw [hg1, hg0, innerProd_comm u (B_X X v), innerProd_comm u (B_X X w), ← innerProd_sub_left,
        innerProd_comm]
    rfl
  have hnorm_sq : normSq u ≤ euclidNorm u * euclidNorm δv := by
    have hslope : g 1 - g 0 = deriv g c := by rw [hc]; ring
    rw [← hdiff_val, hslope]
    exact le_trans (le_abs_self _) (hderiv_bound c)
  by_cases hu0 : euclidNorm u = 0
  · rw [hu0]; exact euclidNorm_nonneg δv
  · have hu_pos : 0 < euclidNorm u := lt_of_le_of_ne (euclidNorm_nonneg u) (Ne.symm hu0)
    have hsq_u : normSq u = euclidNorm u * euclidNorm u := (Real.mul_self_sqrt (normSq_nonneg u)).symm
    rw [hsq_u] at hnorm_sq
    exact le_of_mul_le_mul_left hnorm_sq hu_pos

/-- **Lemma 1(1) (Deterministic cap-event error bound uniform in `N`)**:
For any `N ≥ 1`, unit candidates `X ∈ (S^{d-1})^N`, unit query `y ∈ S^{d-1}`, `δ ∈ (0, 1)`,
`p ∈ (0, 1]`, `β > 0`: if at least `N p / 2` candidates satisfy `⟨x_i, y⟩ ≥ 1 - δ`, then for any
maximizing candidate `i*` (`∀ i, ⟨x_i, y⟩ ≤ ⟨x_{i*}, y⟩`):
1. `‖x_{i*} - y‖² ≤ 2 δ`;
2. The total softmax mass on `{i | ⟨x_i, y⟩ < 1 - 2δ}` is at most `(2 / p) exp(-β δ)`;
3. `‖B_X(β y) - x_{i*}‖² ≤ 12 δ + (16 / p) exp(-β δ)`. -/
theorem softmax_cap_error_bound {N d : ℕ} (hN : 1 ≤ N)
    (X : Fin N → Fin d → ℝ) (hX : ∀ i, normSq (X i) = 1)
    (y : Fin d → ℝ) (hy : normSq y = 1)
    {δ p β : ℝ} (hδ0 : 0 < δ) (hp0 : 0 < p) (hβ0 : 0 < β)
    (hcap : ((N : ℝ) * p) / 2 ≤
      ((Finset.univ.filter fun i : Fin N => 1 - δ ≤ innerProd (X i) y).card : ℝ))
    (iStar : Fin N) (hmax : ∀ i : Fin N, innerProd (X i) y ≤ innerProd (X iStar) y) :
    let bad := Finset.univ.filter fun i : Fin N => innerProd (X i) y < 1 - 2 * δ
    normSq (X iStar - y) ≤ 2 * δ ∧
    (∑ i ∈ bad, softmaxW X (β • y) i) ≤ (2 / p) * Real.exp (-β * δ) ∧
    normSq (B_X X (β • y) - X iStar) ≤ 12 * δ + (16 / p) * Real.exp (-β * δ) := by
  intro bad
  let good := Finset.univ.filter fun i : Fin N => 1 - δ ≤ innerProd (X i) y
  have hN_pos : (0 : ℝ) < N := by positivity
  -- 1. Since |good| ≥ N p / 2 > 0, good is nonempty, so ⟨x_{i*}, y⟩ ≥ 1 - δ, hence ‖x_{i*} - y‖² ≤ 2δ!
  have hgood_ne : good.Nonempty := by
    by_contra hemp
    rw [Finset.not_nonempty_iff_eq_empty] at hemp
    have : (good.card : ℝ) = 0 := by simp [hemp]
    have hpos : 0 < ((N : ℝ) * p) / 2 := by positivity
    linarith
  obtain ⟨i₀, hi₀⟩ := hgood_ne
  have hi₀_ge : 1 - δ ≤ innerProd (X i₀) y := (Finset.mem_filter.1 hi₀).2
  have hiStar_ge : 1 - δ ≤ innerProd (X iStar) y := le_trans hi₀_ge (hmax i₀)
  have h1 : normSq (X iStar - y) ≤ 2 * δ := by
    rw [normSq_sub_unit (hX iStar) hy]; linarith
  -- 2. Softmax mass on bad is ≤ (2 / p) exp(-β δ).
  have hden_ge : ((N : ℝ) * p / 2) * Real.exp (β * (1 - δ)) ≤ softmaxDen X (β • y) := by
    have hsub : ∑ i ∈ good, Real.exp (innerProd (β • y) (X i)) ≤ softmaxDen X (β • y) :=
      Finset.sum_le_univ_sum_of_nonneg fun i => (Real.exp_pos _).le
    have hlb : ∑ i ∈ good, Real.exp (β * (1 - δ)) ≤
        ∑ i ∈ good, Real.exp (innerProd (β • y) (X i)) := by
      refine Finset.sum_le_sum fun i hi => Real.exp_le_exp.2 ?_
      rw [innerProd_smul_left, innerProd_comm]
      exact mul_le_mul_of_nonneg_left (Finset.mem_filter.1 hi).2 hβ0.le
    rw [Finset.sum_const, nsmul_eq_mul] at hlb
    have hcard : ((N : ℝ) * p / 2) * Real.exp (β * (1 - δ)) ≤
        (good.card : ℝ) * Real.exp (β * (1 - δ)) :=
      mul_le_mul_of_nonneg_right hcap (Real.exp_pos _).le
    exact le_trans hcard (le_trans hlb hsub)
  have hnum_le : ∑ i ∈ bad, Real.exp (innerProd (β • y) (X i)) ≤
      (N : ℝ) * Real.exp (β * (1 - 2 * δ)) := by
    calc ∑ i ∈ bad, Real.exp (innerProd (β • y) (X i))
        ≤ ∑ _i ∈ bad, Real.exp (β * (1 - 2 * δ)) := by
          refine Finset.sum_le_sum fun i hi => Real.exp_le_exp.2 ?_
          rw [innerProd_smul_left, innerProd_comm]
          exact mul_le_mul_of_nonneg_left (Finset.mem_filter.1 hi).2.le hβ0.le
      _ = (bad.card : ℝ) * Real.exp (β * (1 - 2 * δ)) := by rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ (N : ℝ) * Real.exp (β * (1 - 2 * δ)) := by
          apply mul_le_mul_of_nonneg_right _ (Real.exp_pos _).le
          have : bad.card ≤ N := by
            simpa using Finset.card_le_univ bad
          exact_mod_cast this
  have h2 : ∑ i ∈ bad, softmaxW X (β • y) i ≤ (2 / p) * Real.exp (-β * δ) := by
    unfold softmaxW
    rw [← Finset.sum_div]
    have hden_pos : 0 < softmaxDen X (β • y) := softmaxDen_pos hN X (β • y)
    have hgood_den_pos : 0 < ((N : ℝ) * p / 2) * Real.exp (β * (1 - δ)) := by positivity
    calc (∑ i ∈ bad, Real.exp (innerProd (β • y) (X i))) / softmaxDen X (β • y)
        ≤ ((N : ℝ) * Real.exp (β * (1 - 2 * δ))) / (((N : ℝ) * p / 2) * Real.exp (β * (1 - δ))) := by
          exact div_le_div₀ (by positivity) hnum_le hgood_den_pos hden_ge
      _ = (2 / p) * (Real.exp (β * (1 - 2 * δ)) / Real.exp (β * (1 - δ))) := by
          field_simp [hN_pos.ne', hp0.ne', (Real.exp_pos _).ne']
      _ = (2 / p) * Real.exp (-β * δ) := by
          rw [← Real.exp_sub]; congr 2; ring
  -- 3. Convexity of z ↦ ‖z - y‖² gives ‖B_X(β y) - y‖² ≤ ∑_i p_i ‖x_i - y‖² ≤ 4δ + 4 w!
  set pW := softmaxW X (β • y)
  have hpW_nn : ∀ i, 0 ≤ pW i := fun i => (softmaxW_pos hN X (β • y) i).le
  have hpW_sum : ∑ i : Fin N, pW i = 1 := softmaxW_sum_one hN X (β • y)
  have hjensen : normSq (B_X X (β • y) - y) ≤ ∑ i : Fin N, pW i * normSq (X i - y) := by
    unfold normSq innerProd
    have hcoord : ∀ j : Fin d, (B_X X (β • y) j - y j) ^ 2 ≤
        ∑ i : Fin N, pW i * (X i j - y j) ^ 2 := by
      intro j
      have hdiff_j : B_X X (β • y) j - y j = ∑ i : Fin N, pW i * (X i j - y j) := by
        simp only [B_X, pW, mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, hpW_sum, one_mul]
      rw [hdiff_j]
      have hcs := Finset.sum_mul_sq_le_sq_mul_sq (Finset.univ : Finset (Fin N))
        (fun i => Real.sqrt (pW i)) (fun i => Real.sqrt (pW i) * (X i j - y j))
      have hlhs : ∑ i : Fin N, Real.sqrt (pW i) * (Real.sqrt (pW i) * (X i j - y j)) =
          ∑ i : Fin N, pW i * (X i j - y j) :=
        Finset.sum_congr rfl fun i _ => by rw [← mul_assoc, Real.mul_self_sqrt (hpW_nn i)]
      have hr1 : ∑ i : Fin N, (Real.sqrt (pW i)) ^ 2 = 1 := by
        rw [Finset.sum_congr rfl fun i _ => Real.sq_sqrt (hpW_nn i)]; exact hpW_sum
      have hr2 : ∑ i : Fin N, (Real.sqrt (pW i) * (X i j - y j)) ^ 2 =
          ∑ i : Fin N, pW i * (X i j - y j) ^ 2 :=
        Finset.sum_congr rfl fun i _ => by rw [mul_pow, Real.sq_sqrt (hpW_nn i)]
      rw [hlhs, hr1, hr2, one_mul] at hcs
      exact hcs
    calc ∑ j : Fin d, (B_X X (β • y) - y) j * (B_X X (β • y) - y) j
        ≤ ∑ j : Fin d, ∑ i : Fin N, pW i * (X i j - y j) ^ 2 := by
          refine Finset.sum_le_sum fun j _ => by
            simpa [Pi.sub_apply, sq] using hcoord j
      _ = ∑ i : Fin N, pW i * ∑ j : Fin d, (X i - y) j * (X i - y) j := by
          rw [Finset.sum_comm]
          refine Finset.sum_congr rfl fun i _ => ?_
          rw [Finset.mul_sum]; refine Finset.sum_congr rfl fun j _ => by simp [sq]
  have hBy_le : normSq (B_X X (β • y) - y) ≤ 4 * δ + 4 * ∑ i ∈ bad, pW i := by
    have hsplit := Finset.sum_filter_add_sum_filter_not (s := (Finset.univ : Finset (Fin N)))
      (p := fun i : Fin N => innerProd (X i) y < 1 - 2 * δ) (f := fun i => pW i * normSq (X i - y))
    have hbad_le : ∑ i ∈ bad, pW i * normSq (X i - y) ≤ 4 * ∑ i ∈ bad, pW i := by
      rw [Finset.mul_sum]
      refine Finset.sum_le_sum fun i _ => ?_
      have h4 : normSq (X i - y) ≤ 4 := by
        rw [normSq_sub_unit (hX i) hy]
        have hle := innerProd_le_mul_norm (X i) y
        have h1' : euclidNorm (X i) = 1 := by simp [euclidNorm, hX i]
        have h2' : euclidNorm y = 1 := by simp [euclidNorm, hy]
        rw [h1', h2', mul_one] at hle
        linarith [(abs_le.1 hle).1]
      nlinarith [hpW_nn i]
    have hok_le : ∑ i ∈ Finset.univ.filter (fun i : Fin N => ¬ innerProd (X i) y < 1 - 2 * δ),
        pW i * normSq (X i - y) ≤ 4 * δ := by
      calc ∑ i ∈ Finset.univ.filter (fun i : Fin N => ¬ innerProd (X i) y < 1 - 2 * δ),
            pW i * normSq (X i - y)
          ≤ ∑ i ∈ Finset.univ.filter (fun i : Fin N => ¬ innerProd (X i) y < 1 - 2 * δ),
            pW i * (4 * δ) := by
            refine Finset.sum_le_sum fun i hi => mul_le_mul_of_nonneg_left ?_ (hpW_nn i)
            have hge : 1 - 2 * δ ≤ innerProd (X i) y := not_lt.1 (Finset.mem_filter.1 hi).2
            rw [normSq_sub_unit (hX i) hy]; linarith
        _ ≤ ∑ i : Fin N, pW i * (4 * δ) :=
            Finset.sum_le_univ_sum_of_nonneg fun i => mul_nonneg (hpW_nn i) (by linarith)
        _ = 4 * δ := by rw [← Finset.sum_mul, hpW_sum, one_mul]
    linarith [hjensen, hsplit, hbad_le, hok_le]
  have htri_sq : normSq (B_X X (β • y) - X iStar) ≤
      2 * normSq (B_X X (β • y) - y) + 2 * normSq (X iStar - y) := by
    unfold normSq innerProd
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_le_sum fun j _ => ?_
    simp only [Pi.sub_apply]
    nlinarith [sq_nonneg ((B_X X (β • y) j - y j) + (X iStar j - y j))]
  have h3 : normSq (B_X X (β • y) - X iStar) ≤ 12 * δ + (16 / p) * Real.exp (-β * δ) := by
    have hmul8 : 8 * ((2 / p) * Real.exp (-β * δ)) = (16 / p) * Real.exp (-β * δ) := by ring
    linarith [htri_sq, hBy_le, h1, h2, hmul8]
  exact ⟨h1, h2, h3⟩

/-! ### 2. Parity-Bit Comparator Primitive and Uniform-in-`N` Offset Bounds -/

/-- Number of odd-indexed candidates `n₁ = ⌈N/2⌉ = (N + 1) / 2`. -/
def parityN1 (N : ℕ) : ℕ := (N + 1) / 2

/-- Number of even-indexed candidates `n₀ = ⌊N/2⌋ = N / 2`. -/
def parityN0 (N : ℕ) : ℕ := N / 2

/-- Parity-bit comparator output `S_N(t) = (n₁ exp t) / (n₀ + n₁ exp t)`. -/
noncomputable def parityComp (N : ℕ) (t : ℝ) : ℝ :=
  ((parityN1 N : ℝ) * Real.exp t) / ((parityN0 N : ℝ) + (parityN1 N : ℝ) * Real.exp t)

/-- Standard logistic sigmoid `σ(z) = (1 + exp(-z))⁻¹`. -/
noncomputable def sigmoid (z : ℝ) : ℝ := 1 / (1 + Real.exp (-z))

/-- **Exact sigmoid representation and uniform offset bound `ℓ_N ∈ [0, log 2]` for all `N ≥ 2`**. -/
theorem parityComp_eq_sigmoid_and_offset {N : ℕ} (hN : 2 ≤ N) (t : ℝ) :
    let ell := Real.log ((parityN1 N : ℝ) / (parityN0 N : ℝ))
    parityComp N t = sigmoid (t + ell) ∧ 0 ≤ ell ∧ ell ≤ Real.log 2 := by
  intro ell
  have hn0 : 1 ≤ parityN0 N := by unfold parityN0; omega
  have hn1 : parityN0 N ≤ parityN1 N := by unfold parityN0 parityN1; omega
  have hn1_le : parityN1 N ≤ 2 * parityN0 N := by unfold parityN0 parityN1; omega
  have hn0_pos : (0 : ℝ) < parityN0 N := by exact_mod_cast (show 0 < parityN0 N by omega)
  have hn1_pos : (0 : ℝ) < parityN1 N := by exact_mod_cast (show 0 < parityN1 N by omega)
  have hratio_ge1 : 1 ≤ (parityN1 N : ℝ) / (parityN0 N : ℝ) := by
    rw [one_le_div hn0_pos]; exact_mod_cast hn1
  have hratio_le2 : (parityN1 N : ℝ) / (parityN0 N : ℝ) ≤ 2 := by
    rw [div_le_iff₀ hn0_pos]; exact_mod_cast hn1_le
  have hratio_pos : 0 < (parityN1 N : ℝ) / (parityN0 N : ℝ) := by positivity
  have heq : parityComp N t = sigmoid (t + ell) := by
    unfold parityComp sigmoid ell
    rw [neg_add, Real.exp_add, Real.exp_neg, Real.exp_neg, Real.exp_log hratio_pos]
    have hexp_pos : 0 < Real.exp t := Real.exp_pos t
    field_simp [hn0_pos.ne', hn1_pos.ne', hexp_pos.ne']
    ring
  refine ⟨heq, Real.log_nonneg hratio_ge1, Real.log_le_log hratio_pos hratio_le2⟩

/-- **Uniform-in-`N` comparator tail bounds for all `N ≥ 2`**:
`0 < S_N(t) < 1`, `1 - S_N(t) ≤ exp(-t)`, and `S_N(t) ≤ 2 exp(t)`. -/
theorem parityComp_uniform_bounds {N : ℕ} (hN : 2 ≤ N) (t : ℝ) :
    0 < parityComp N t ∧ parityComp N t < 1 ∧
    1 - parityComp N t ≤ Real.exp (-t) ∧
    parityComp N t ≤ 2 * Real.exp t := by
  have hn0 : 1 ≤ parityN0 N := by unfold parityN0; omega
  have hn1 : parityN0 N ≤ parityN1 N := by unfold parityN0 parityN1; omega
  have hn1_le : parityN1 N ≤ 2 * parityN0 N := by unfold parityN0 parityN1; omega
  have hn0_pos : (0 : ℝ) < parityN0 N := by exact_mod_cast (show 0 < parityN0 N by omega)
  have hn1_pos : (0 : ℝ) < parityN1 N := by exact_mod_cast (show 0 < parityN1 N by omega)
  have hn0_le_n1 : (parityN0 N : ℝ) ≤ parityN1 N := by exact_mod_cast hn1
  have hn1_le_2n0 : (parityN1 N : ℝ) ≤ 2 * parityN0 N := by exact_mod_cast hn1_le
  have hden_pos : 0 < (parityN0 N : ℝ) + (parityN1 N : ℝ) * Real.exp t := by positivity
  have h0lt : 0 < parityComp N t := div_pos (by positivity) hden_pos
  have hlt1 : parityComp N t < 1 := by
    unfold parityComp; rw [div_lt_one hden_pos]; linarith
  have hupper : 1 - parityComp N t ≤ Real.exp (-t) := by
    have h1m : 1 - parityComp N t =
        (parityN0 N : ℝ) / ((parityN0 N : ℝ) + (parityN1 N : ℝ) * Real.exp t) := by
      unfold parityComp; field_simp [hden_pos.ne']; ring
    rw [h1m, Real.exp_neg, div_le_iff₀ hden_pos]
    have hmul : (parityN0 N : ℝ) ≤ (parityN1 N : ℝ) * Real.exp t * (Real.exp t)⁻¹ := by
      rw [mul_assoc, mul_inv_cancel₀ (Real.exp_pos t).ne', mul_one]; exact hn0_le_n1
    have hnn : 0 ≤ (parityN0 N : ℝ) * (Real.exp t)⁻¹ := by positivity
    linarith
  have hlower : parityComp N t ≤ 2 * Real.exp t := by
    unfold parityComp
    rw [div_le_iff₀ hden_pos]
    have hnn : 0 ≤ 2 * Real.exp t * ((parityN1 N : ℝ) * Real.exp t) := by positivity
    nlinarith [Real.exp_pos t]
  exact ⟨h0lt, hlt1, hupper, hlower⟩

/-! ### 3. Three-Layer Rank-1 Transformer Routing & Uniform-in-`N` Approximation (Theorem 1) -/

/-- Layer 1 win counts `c_h = ∑_{k ≠ h} S_N(a₁ ⟨u_h - u_k, y⟩)`. -/
noncomputable def layer1Wins {K d : ℕ} (U : Fin K → Fin d → ℝ) (a1 : ℝ) (N : ℕ)
    (y : Fin d → ℝ) (h : Fin K) : ℝ :=
  ∑ k ∈ Finset.univ.erase h, parityComp N (a1 * innerProd (U h - U k) y)

/-- Layer 2 winner gates `g_h = S_N(a₂ (c_h - (K - 3/2)))`. -/
noncomputable def layer2Gates {K d : ℕ} (U : Fin K → Fin d → ℝ) (a1 a2 : ℝ) (N : ℕ)
    (y : Fin d → ℝ) (h : Fin K) : ℝ :=
  parityComp N (a2 * (layer1Wins U a1 N y h - ((K : ℝ) - 3 / 2)))

/-- Layer 3 rank-1 Transformer readout `T(X, y) = μ_X + ∑_h (B_X(β g_h u_h) - μ_X)`. -/
noncomputable def transformerOut {K N d : ℕ} (U : Fin K → Fin d → ℝ)
    (a1 a2 β : ℝ) (X : Fin N → Fin d → ℝ) (y : Fin d → ℝ) : Fin d → ℝ :=
  mu_X X + ∑ h : Fin K, (B_X X ((β * layer2Gates U a1 a2 N y h) • U h) - mu_X X)

/-- **Exact retrieval at `N = 1`**: For any direction net `U` and ANY weights `(a₁, a₂, β)`,
the 3-layer rank-1 Transformer outputs `x₀` identically (zero error). -/
theorem transformerOut_exact_N1 {K d : ℕ} (U : Fin K → Fin d → ℝ)
    (a1 a2 β : ℝ) (X : Fin 1 → Fin d → ℝ) (y : Fin d → ℝ) :
    transformerOut U a1 a2 β X y = X 0 := by
  unfold transformerOut mu_X
  simp [B_X_N1_exact]

/-- **Theorem 1 (Uniform-in-`N` routing gate convergence with explicit finite gains)**:
For any `K ≥ 2` directions `U`, query `y` with strict winner `h*` and margin `γ > 0`
(`∀ k ≠ h*, γ ≤ ⟨u_{h*} - u_k, y⟩`), and any tolerance `η > 0`:
taking `a₁ ≥ log(8 K) / γ` and `a₂ ≥ 4 log(2 / η)` guarantees
`|g_h(N, y) - 1_{h = h*}| ≤ η` simultaneously for EVERY `N ≥ 2` and EVERY `h : Fin K`! -/
theorem routing_gate_uniform_in_N {K d : ℕ} (hK : 2 ≤ K) (U : Fin K → Fin d → ℝ)
    (y : Fin d → ℝ) (hStar : Fin K) {γ η : ℝ} (hγ : 0 < γ) (hη : 0 < η)
    (hmargin : ∀ k : Fin K, k ≠ hStar → γ ≤ innerProd (U hStar - U k) y)
    {a1 a2 : ℝ} (ha1 : Real.log (8 * (K : ℝ)) / γ ≤ a1)
    (ha2 : 4 * Real.log (2 / η) ≤ a2) (ha2_nn : 0 ≤ a2)
    {N : ℕ} (hN : 2 ≤ N) (h : Fin K) :
    |layer2Gates U a1 a2 N y h - (if h = hStar then 1 else 0)| ≤ η := by
  have hK_pos : (0 : ℝ) < K := by positivity
  have ha1_pos : 0 < a1 := by
    have h8K : (1 : ℝ) < 8 * (K : ℝ) := by
      have : (2 : ℝ) ≤ K := by exact_mod_cast hK
      linarith
    exact lt_of_lt_of_le (div_pos (Real.log_pos h8K) hγ) ha1
  have hexp_a1 : Real.exp (-(a1 * γ)) ≤ 1 / (8 * (K : ℝ)) := by
    have hge : Real.log (8 * (K : ℝ)) ≤ a1 * γ := (div_le_iff₀ hγ).1 ha1
    calc Real.exp (-(a1 * γ))
        ≤ Real.exp (-Real.log (8 * (K : ℝ))) := Real.exp_le_exp.2 (by linarith)
      _ = 1 / (8 * (K : ℝ)) := by rw [Real.exp_neg, Real.exp_log (by positivity), one_div]
  rcases eq_or_ne hStar h with rfl | hh'
  · simp only [↓reduceIte]
    -- Winner hStar has c_{hStar} ≥ K - 1 - (K - 1) / (8K) ≥ K - 5/4!
    have hc_ge : (K : ℝ) - 5 / 4 ≤ layer1Wins U a1 N y hStar := by
      unfold layer1Wins
      have hterm : ∀ k ∈ Finset.univ.erase hStar,
          1 - 1 / (8 * (K : ℝ)) ≤ parityComp N (a1 * innerProd (U hStar - U k) y) := by
        intro k hk
        have hkne : k ≠ hStar := (Finset.mem_erase.1 hk).1
        have hcomp := (parityComp_uniform_bounds hN (a1 * innerProd (U hStar - U k) y)).2.2.1
        have harg : a1 * γ ≤ a1 * innerProd (U hStar - U k) y :=
          mul_le_mul_of_nonneg_left (hmargin k hkne) ha1_pos.le
        have hexp_le : Real.exp (-(a1 * innerProd (U hStar - U k) y)) ≤ 1 / (8 * (K : ℝ)) :=
          le_trans (Real.exp_le_exp.2 (by linarith)) hexp_a1
        linarith
      have hsum_ge := Finset.sum_le_sum hterm
      rw [Finset.sum_const, Finset.card_erase_of_mem (Finset.mem_univ hStar),
          Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hsum_ge
      have hK1 : ((K - 1 : ℕ) : ℝ) = (K : ℝ) - 1 := by
        rw [Nat.cast_sub (by omega)]; simp
      rw [hK1] at hsum_ge
      have hbound : (K : ℝ) - 5 / 4 ≤ ((K : ℝ) - 1) * (1 - 1 / (8 * (K : ℝ))) := by
        have hKge2 : (2 : ℝ) ≤ K := by exact_mod_cast hK
        have hfrac : ((K : ℝ) - 1) / (8 * (K : ℝ)) ≤ 1 / 4 := by
          rw [div_le_div_iff₀ (by positivity) (by norm_num)]; linarith
        have heq : ((K : ℝ) - 1) * (1 - 1 / (8 * (K : ℝ))) =
            (K : ℝ) - 1 - ((K : ℝ) - 1) / (8 * (K : ℝ)) := by ring
        linarith
      linarith
    have hgate := parityComp_uniform_bounds hN (a2 * (layer1Wins U a1 N y hStar - ((K : ℝ) - 3 / 2)))
    rw [abs_of_neg (by unfold layer2Gates; linarith [hgate.2.1]), neg_sub]
    have harg_ge : a2 / 4 ≤ a2 * (layer1Wins U a1 N y hStar - ((K : ℝ) - 3 / 2)) := by
      nlinarith
    have hlog_ge : Real.log (2 / η) ≤ a2 / 4 := by linarith
    calc 1 - layer2Gates U a1 a2 N y hStar
        ≤ Real.exp (-(a2 * (layer1Wins U a1 N y hStar - ((K : ℝ) - 3 / 2)))) := hgate.2.2.1
      _ ≤ Real.exp (-Real.log (2 / η)) := Real.exp_le_exp.2 (by linarith)
      _ = η / 2 := by rw [Real.exp_neg, Real.exp_log (by positivity)]; field_simp
      _ ≤ η := by linarith
  · have hh : h ≠ hStar := hh'.symm
    simp only [hh, ↓reduceIte, sub_zero]
    -- Loser h ≠ hStar loses to hStar (contribution ≤ 2/(8K) = 1/(4K)) and has ≤ K - 2 other terms (< 1 each)!
    have hc_le : layer1Wins U a1 N y h ≤ (K : ℝ) - 7 / 4 := by
      unfold layer1Wins
      have hmem : hStar ∈ Finset.univ.erase h := Finset.mem_erase.2 ⟨Ne.symm hh, Finset.mem_univ _⟩
      rw [← Finset.add_sum_erase _ _ hmem]
      have hloss : parityComp N (a1 * innerProd (U h - U hStar) y) ≤ 1 / 4 := by
        have hcomp := (parityComp_uniform_bounds hN (a1 * innerProd (U h - U hStar) y)).2.2.2
        have hneg : innerProd (U h - U hStar) y = -innerProd (U hStar - U h) y := by
          unfold innerProd
          rw [← Finset.sum_neg_distrib]
          refine Finset.sum_congr rfl fun j _ => by simp only [Pi.sub_apply]; ring
        have harg : a1 * innerProd (U h - U hStar) y ≤ -(a1 * γ) := by
          rw [hneg]; nlinarith [hmargin h hh]
        have hexp_le : Real.exp (a1 * innerProd (U h - U hStar) y) ≤ 1 / (8 * (K : ℝ)) :=
          le_trans (Real.exp_le_exp.2 harg) hexp_a1
        have h2K : 2 * (1 / (8 * (K : ℝ))) ≤ 1 / 4 := by
          have hKge2 : (2 : ℝ) ≤ K := by exact_mod_cast hK
          rw [show 2 * (1 / (8 * (K : ℝ))) = 1 / (4 * (K : ℝ)) by ring,
              div_le_div_iff₀ (by positivity) (by norm_num)]
          linarith
        linarith
      have hrest : ∑ k ∈ (Finset.univ.erase h).erase hStar,
          parityComp N (a1 * innerProd (U h - U k) y) ≤ (K : ℝ) - 2 := by
        calc ∑ k ∈ (Finset.univ.erase h).erase hStar,
              parityComp N (a1 * innerProd (U h - U k) y)
            ≤ ∑ _k ∈ (Finset.univ.erase h).erase hStar, (1 : ℝ) :=
              Finset.sum_le_sum fun k _ => (parityComp_uniform_bounds hN _).2.1.le
          _ = (((Finset.univ.erase h).erase hStar).card : ℝ) := by simp
          _ = (K : ℝ) - 2 := by
              rw [Finset.card_erase_of_mem hmem, Finset.card_erase_of_mem (Finset.mem_univ h),
                  Finset.card_univ, Fintype.card_fin]
              have hsub : K - 1 - 1 = K - 2 := by omega
              rw [hsub, Nat.cast_sub hK]; simp
      linarith
    have hgate := parityComp_uniform_bounds hN (a2 * (layer1Wins U a1 N y h - ((K : ℝ) - 3 / 2)))
    unfold layer2Gates
    rw [abs_of_pos hgate.1]
    have harg_le : a2 * (layer1Wins U a1 N y h - ((K : ℝ) - 3 / 2)) ≤ -(a2 / 4) := by
      nlinarith
    have hlog_ge : Real.log (2 / η) ≤ a2 / 4 := by linarith
    calc layer2Gates U a1 a2 N y h
        ≤ 2 * Real.exp (a2 * (layer1Wins U a1 N y h - ((K : ℝ) - 3 / 2))) := hgate.2.2.2
      _ ≤ 2 * Real.exp (-Real.log (2 / η)) :=
          mul_le_mul_of_nonneg_left (Real.exp_le_exp.2 (by linarith)) (by norm_num)
      _ = η := by rw [Real.exp_neg, Real.exp_log (by positivity)]; field_simp

/-- **Theorem 1 (Deterministic distance bound `‖T(X, y) - B_X(β y)‖₂ ≤ K β e + β ρ`)**:
For every `N ≥ 1`, unit candidates `X`, unit directions `U`, and query `y` with
`‖u_{h*} - y‖₂ ≤ ρ` and `∀ h, |g_h - 1_{h = h*}| ≤ e`:
`‖T(X, y) - B_X(β y)‖₂ ≤ K β e + β ρ`. -/
theorem transformerOut_distance_to_B_X {K N d : ℕ} (hN : 1 ≤ N)
    (U : Fin K → Fin d → ℝ) (hU : ∀ h, normSq (U h) = 1)
    (X : Fin N → Fin d → ℝ) (hX : ∀ i, normSq (X i) = 1)
    (y : Fin d → ℝ) (hStar : Fin K) {a1 a2 β e ρ : ℝ}
    (hβ : 0 ≤ β)
    (hnet : euclidNorm (U hStar - y) ≤ ρ)
    (hgate_err : ∀ h : Fin K, |layer2Gates U a1 a2 N y h - (if h = hStar then 1 else 0)| ≤ e) :
    euclidNorm (transformerOut U a1 a2 β X y - B_X X (β • y)) ≤ (K : ℝ) * β * e + β * ρ := by
  set g := layer2Gates U a1 a2 N y
  have hdecomp : transformerOut U a1 a2 β X y - B_X X (β • y) =
      (B_X X ((β * g hStar) • U hStar) - B_X X (β • y)) +
      ∑ h ∈ Finset.univ.erase hStar, (B_X X ((β * g h) • U h) - B_X X 0) := by
    unfold transformerOut mu_X
    rw [← Finset.add_sum_erase _ _ (Finset.mem_univ hStar)]
    ext j; simp only [Pi.add_apply, Pi.sub_apply, Finset.sum_apply]; ring
  rw [hdecomp]
  -- Triangle inequality on Finset sum
  have hsum_tri : ∀ (s : Finset (Fin K)) (F : Fin K → Fin d → ℝ),
      euclidNorm (∑ h ∈ s, F h) ≤ ∑ h ∈ s, euclidNorm (F h) := by
    classical
    intro s F
    induction s using Finset.induction_on with
    | empty =>
      simp only [Finset.sum_empty, euclidNorm, normSq, innerProd, Pi.zero_apply, mul_zero,
        Finset.sum_const_zero, Real.sqrt_zero, le_refl]
    | insert a s ha ih =>
      rw [Finset.sum_insert ha, Finset.sum_insert ha]
      exact le_trans (euclidNorm_add_le _ _) (add_le_add_left ih _)
  have hmain : euclidNorm (B_X X ((β * g hStar) • U hStar) - B_X X (β • y)) ≤ β * (e + ρ) := by
    calc euclidNorm (B_X X ((β * g hStar) • U hStar) - B_X X (β • y))
        ≤ euclidNorm ((β * g hStar) • U hStar - β • y) := B_X_lipschitz hN X hX _ _
      _ = β * euclidNorm (g hStar • U hStar - y) := by
          have h1 : (β * g hStar) • U hStar - β • y = β • (g hStar • U hStar - y) := by
            ext j; simp; ring
          rw [h1, euclidNorm_smul, abs_of_nonneg hβ]
      _ ≤ β * (e + ρ) := by
          apply mul_le_mul_of_nonneg_left _ hβ
          have hsplit : g hStar • U hStar - y = (g hStar - 1) • U hStar + (U hStar - y) := by
            ext j; simp; ring
          rw [hsplit]
          calc euclidNorm ((g hStar - 1) • U hStar + (U hStar - y))
              ≤ euclidNorm ((g hStar - 1) • U hStar) + euclidNorm (U hStar - y) :=
                euclidNorm_add_le _ _
            _ = |g hStar - 1| * euclidNorm (U hStar) + euclidNorm (U hStar - y) := by
                rw [euclidNorm_smul]
            _ = |g hStar - 1| + euclidNorm (U hStar - y) := by
                simp [euclidNorm, hU hStar]
            _ ≤ e + ρ := add_le_add (by simpa using hgate_err hStar) hnet
  have hlosers : ∑ h ∈ Finset.univ.erase hStar, euclidNorm (B_X X ((β * g h) • U h) - B_X X 0) ≤
      ((K : ℝ) - 1) * (β * e) := by
    calc ∑ h ∈ Finset.univ.erase hStar, euclidNorm (B_X X ((β * g h) • U h) - B_X X 0)
        ≤ ∑ _h ∈ Finset.univ.erase hStar, (β * e) := by
          refine Finset.sum_le_sum fun h hh => ?_
          have hne : h ≠ hStar := (Finset.mem_erase.1 hh).1
          calc euclidNorm (B_X X ((β * g h) • U h) - B_X X 0)
              ≤ euclidNorm ((β * g h) • U h - 0) := B_X_lipschitz hN X hX _ _
            _ = |β * g h| * euclidNorm (U h) := by rw [sub_zero, euclidNorm_smul]
            _ = β * |g h| := by simp [euclidNorm, hU h, abs_mul, abs_of_nonneg hβ]
            _ ≤ β * e := mul_le_mul_of_nonneg_left (by simpa [hne] using hgate_err h) hβ
      _ = ((Finset.univ.erase hStar).card : ℝ) * (β * e) := by rw [Finset.sum_const, nsmul_eq_mul]
      _ = ((K : ℝ) - 1) * (β * e) := by
          rw [Finset.card_erase_of_mem (Finset.mem_univ hStar), Finset.card_univ, Fintype.card_fin]
          have hK1 : 1 ≤ K := Nat.pos_of_ne_zero (fun h0 => by subst h0; exact Fin.elim0 hStar)
          rw [Nat.cast_sub hK1]; simp
  calc euclidNorm ((B_X X ((β * g hStar) • U hStar) - B_X X (β • y)) +
          ∑ h ∈ Finset.univ.erase hStar, (B_X X ((β * g h) • U h) - B_X X 0))
      ≤ euclidNorm (B_X X ((β * g hStar) • U hStar) - B_X X (β • y)) +
        euclidNorm (∑ h ∈ Finset.univ.erase hStar, (B_X X ((β * g h) • U h) - B_X X 0)) :=
        euclidNorm_add_le _ _
    _ ≤ β * (e + ρ) + ((K : ℝ) - 1) * (β * e) :=
        add_le_add hmain (le_trans (hsum_tri _ _) hlosers)
    _ = (K : ℝ) * β * e + β * ρ := by ring

/-- **Corollary 3 (Exact position-free ReLU MLP gate)**:
`min(s, t) = s - ReLU(s - t)`, and the clipped ReLU gate
`g_h(y) = min(1, max(0, min_{k ≠ h} ⟨u_h - u_k, y⟩ / η))` equals `1_{h = h*}` whenever
`0 < η ≤ γ`. -/
theorem relu_gate_exact {K d : ℕ} (hK : 2 ≤ K) (U : Fin K → Fin d → ℝ)
    (y : Fin d → ℝ) (hStar : Fin K) {γ η : ℝ} (hη : 0 < η) (hηγ : η ≤ γ)
    (hmargin : ∀ k : Fin K, k ≠ hStar → γ ≤ innerProd (U hStar - U k) y) :
    (∀ s t : ℝ, min s t = s - max 0 (s - t)) ∧
    ∀ h : Fin K,
      have hne : (Finset.univ.erase h).Nonempty := by
        have hcard : 0 < (Finset.univ.erase h).card := by
          rw [Finset.card_erase_of_mem (Finset.mem_univ h), Finset.card_univ, Fintype.card_fin]
          omega
        exact Finset.card_pos.1 hcard
      min 1 (max 0 ((Finset.univ.erase h).inf' hne (fun k => innerProd (U h - U k) y / η))) =
        if h = hStar then 1 else 0 := by
  constructor
  · intro s t
    rcases le_total s t with hst | hts
    · rw [min_eq_left hst, max_eq_left (by linarith)]; ring
    · rw [min_eq_right hts, max_eq_right (by linarith)]; ring
  · intro h hne
    rcases eq_or_ne h hStar with rfl | hh
    · simp only [↓reduceIte]
      have hge1 : 1 ≤ (Finset.univ.erase h).inf' hne
          (fun k => innerProd (U h - U k) y / η) := by
        rw [Finset.le_inf'_iff]
        intro k hk
        rw [one_le_div hη]
        exact le_trans hηγ (hmargin k (Finset.mem_erase.1 hk).1)
      rw [max_eq_right (by linarith), min_eq_left hge1]
    · simp only [hh, ↓reduceIte]
      have hmem : hStar ∈ Finset.univ.erase h := Finset.mem_erase.2 ⟨Ne.symm hh, Finset.mem_univ _⟩
      have hle0 : (Finset.univ.erase h).inf' hne (fun k => innerProd (U h - U k) y / η) ≤ 0 := by
        refine le_trans (Finset.inf'_le _ hmem) ?_
        have hneg : innerProd (U h - U hStar) y = -innerProd (U hStar - U h) y := by
          unfold innerProd
          rw [← Finset.sum_neg_distrib]
          refine Finset.sum_congr rfl fun j _ => by simp only [Pi.sub_apply]; ring
        rw [hneg]
        exact div_nonpos_of_nonpos_of_nonneg (by linarith [hmargin h hh]) hη.le
      rw [max_eq_left hle0, min_eq_right zero_le_one]

/-! ### 4. Proposition 4: Worst-Case `N = 2` Lower Bound `≥ 1` Scope Theorem -/

/-- **Proposition 4 (Impossibility of worst-case-over-inputs approximation at `N = 2`)**:
For EVERY dimension `d ≥ 2`, EVERY continuous map `T : (Fin d → ℝ) → (Fin d → ℝ)` (representing
`y ↦ T(e₀, -e₀, y)` for antipodal candidates `±e₀`), and EVERY `ε < 1`, there exists a unit
query `y ∈ S^{d-1}` with `y 0 ≠ 0` such that `ε < ‖T(y) - sgn(y 0) e₀‖₂`! -/
theorem worst_case_N2_lower_bound {d : ℕ} (hd : 2 ≤ d)
    (T : (Fin d → ℝ) → (Fin d → ℝ)) (hT : Continuous T) {ε : ℝ} (hε : ε < 1) :
    let e0 : Fin d → ℝ := Pi.single ⟨0, by omega⟩ 1
    ∃ y : Fin d → ℝ, normSq y = 1 ∧ y ⟨0, by omega⟩ ≠ 0 ∧
      ε < euclidNorm (T y - (if 0 < y ⟨0, by omega⟩ then e0 else -e0)) := by
  intro e0
  let i0 : Fin d := ⟨0, by omega⟩
  let i1 : Fin d := ⟨1, by omega⟩
  let e1 : Fin d → ℝ := Pi.single i1 1
  have he0_eq : e0 = Pi.single i0 1 := rfl
  have hi01 : i0 ≠ i1 := by intro h; have := congrArg Fin.val h; simp [i0, i1] at this
  -- First, by triangle inequality on T(e1), max(‖T(e1) - e0‖, ‖T(e1) + e0‖) ≥ 1!
  have hnorm_2e0 : euclidNorm (e0 - (-e0)) = 2 := by
    have h1 : e0 - (-e0) = (2 : ℝ) • e0 := by ext j; simp; ring
    have he0 : normSq e0 = 1 := by
      unfold normSq innerProd
      rw [he0_eq, Finset.sum_eq_single i0]
      · simp
      · intro j _ hj; simp [Pi.single_eq_of_ne hj]
      · intro h; exact (h (Finset.mem_univ i0)).elim
    rw [h1, euclidNorm_smul, abs_of_pos (by norm_num : (0 : ℝ) < 2), euclidNorm, he0,
        Real.sqrt_one, mul_one]
  have hmax_ge1 : 1 ≤ euclidNorm (T e1 - e0) ∨ 1 ≤ euclidNorm (T e1 - (-e0)) := by
    have htri : euclidNorm (e0 - (-e0)) ≤ euclidNorm (e0 - T e1) + euclidNorm (T e1 - (-e0)) :=
      euclidNorm_sub_le e0 (T e1) (-e0)
    have hsym : euclidNorm (e0 - T e1) = euclidNorm (T e1 - e0) := by
      have : e0 - T e1 = (-1 : ℝ) • (T e1 - e0) := by ext j; simp
      rw [this, euclidNorm_smul]; norm_num
    rw [hnorm_2e0, hsym] at htri
    by_contra hlt
    push_neg at hlt
    linarith
  -- Continuity of θ ↦ euclidNorm (T(cos θ e1 + s sin θ e0) - s e0) at θ = 0!
  have hnorm_cont : Continuous (euclidNorm (d := d)) := by
    unfold euclidNorm normSq innerProd
    exact Real.continuous_sqrt.comp (continuous_finset_sum _ fun j _ =>
      (continuous_apply j).mul (continuous_apply j))
  have hunit_curve : ∀ (s : ℝ) (θ : ℝ), s ^ 2 = 1 →
      normSq (Real.cos θ • e1 + (s * Real.sin θ) • e0) = 1 := by
    intro s θ hs
    unfold normSq innerProd
    have hterm : ∀ j : Fin d,
        (Real.cos θ • e1 + (s * Real.sin θ) • e0) j *
        (Real.cos θ • e1 + (s * Real.sin θ) • e0) j =
        (if j = i1 then Real.cos θ ^ 2 else 0) + (if j = i0 then Real.sin θ ^ 2 else 0) := by
      intro j
      simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, he0_eq, e1]
      by_cases hj1 : j = i1
      · subst hj1; simp [hi01.symm]; ring
      · by_cases hj0 : j = i0
        · subst hj0; simp [hj1]; nlinarith [hs]
        · simp [hj1, hj0]
    rw [Finset.sum_congr rfl fun j _ => hterm j, Finset.sum_add_distrib]
    simp [Real.cos_sq_add_sin_sq θ]
  rcases hmax_ge1 with hpos | hneg
  · -- Approach e1 from y_+(θ) = cos θ e1 + sin θ e0 (θ ↓ 0)
    let γ : ℝ → (Fin d → ℝ) := fun θ => Real.cos θ • e1 + (1 * Real.sin θ) • e0
    have hγ_cont : Continuous γ := by fun_prop
    have hf_cont : Continuous (fun θ => euclidNorm (T (γ θ) - e0)) :=
      hnorm_cont.comp ((hT.comp hγ_cont).sub continuous_const)
    have hf0 : ε < euclidNorm (T (γ 0) - e0) := by
      have hγ0 : γ 0 = e1 := by ext j; simp [γ]
      rw [hγ0]; linarith
    have hopen : IsOpen {θ : ℝ | ε < euclidNorm (T (γ θ) - e0)} :=
      isOpen_lt continuous_const hf_cont
    obtain ⟨δ, hδpos, hδball⟩ := Metric.isOpen_iff.1 hopen 0 hf0
    let θ := min (δ / 2) (Real.pi / 4)
    have hθ_pos : 0 < θ := lt_min (by linarith) (by positivity)
    have hθ_lt_pi : θ < Real.pi := lt_of_le_of_lt (min_le_right _ _) (by linarith [Real.pi_pos])
    have hθ_mem : θ ∈ Metric.ball (0 : ℝ) δ := by
      rw [Metric.mem_ball, Real.dist_eq, sub_zero, abs_of_pos hθ_pos]
      exact lt_of_le_of_lt (min_le_left _ _) (by linarith)
    have hsin_pos : 0 < Real.sin θ := Real.sin_pos_of_pos_of_lt_pi hθ_pos hθ_lt_pi
    have hy0 : γ θ i0 = Real.sin θ := by
      simp [γ, he0_eq, e1, Pi.single_eq_of_ne hi01]
    refine ⟨γ θ, hunit_curve 1 θ (by norm_num), by rw [hy0]; exact hsin_pos.ne', ?_⟩
    have hif : (if 0 < γ θ i0 then e0 else -e0) = e0 := by rw [hy0, if_pos hsin_pos]
    rw [hif]
    exact hδball hθ_mem
  · -- Approach e1 from y_-(θ) = cos θ e1 - sin θ e0 (θ ↓ 0)
    let γ : ℝ → (Fin d → ℝ) := fun θ => Real.cos θ • e1 + ((-1) * Real.sin θ) • e0
    have hγ_cont : Continuous γ := by fun_prop
    have hf_cont : Continuous (fun θ => euclidNorm (T (γ θ) - (-e0))) :=
      hnorm_cont.comp ((hT.comp hγ_cont).sub continuous_const)
    have hf0 : ε < euclidNorm (T (γ 0) - (-e0)) := by
      have hγ0 : γ 0 = e1 := by ext j; simp [γ]
      rw [hγ0]; linarith
    have hopen : IsOpen {θ : ℝ | ε < euclidNorm (T (γ θ) - (-e0))} :=
      isOpen_lt continuous_const hf_cont
    obtain ⟨δ, hδpos, hδball⟩ := Metric.isOpen_iff.1 hopen 0 hf0
    let θ := min (δ / 2) (Real.pi / 4)
    have hθ_pos : 0 < θ := lt_min (by linarith) (by positivity)
    have hθ_lt_pi : θ < Real.pi := lt_of_le_of_lt (min_le_right _ _) (by linarith [Real.pi_pos])
    have hθ_mem : θ ∈ Metric.ball (0 : ℝ) δ := by
      rw [Metric.mem_ball, Real.dist_eq, sub_zero, abs_of_pos hθ_pos]
      exact lt_of_le_of_lt (min_le_left _ _) (by linarith)
    have hsin_pos : 0 < Real.sin θ := Real.sin_pos_of_pos_of_lt_pi hθ_pos hθ_lt_pi
    have hy0 : γ θ i0 = -Real.sin θ := by
      simp [γ, he0_eq, e1, Pi.single_eq_of_ne hi01]
    refine ⟨γ θ, hunit_curve (-1) θ (by norm_num), by rw [hy0]; linarith, ?_⟩
    have hif : (if 0 < γ θ i0 then e0 else -e0) = -e0 := by rw [hy0, if_neg (by linarith)]
    rw [hif]
    exact hδball hθ_mem

end AttentionRankBenefit
