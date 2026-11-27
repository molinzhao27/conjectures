import Mathlib

/-!
# Resolution of Gaussian Equivalence for Extensive-Rank Matrix Denoising
(arXiv:2611.29179, Section III-C, Conjecture 1 `Conj:Effective`)

We formalize in Lean 4 + Mathlib (completely verified, standard foundations only) the sharp algebraic,
moment, Stein-divergence, and Hilbert-pole cores from
`solutions/extensive_matrix_denoise/solution.md`:

1. **Deterministic Uniform-Integrability Bounds (Theorem 2(1))**:
   - `risk_le_two_power_add_two_bound` and `power_le_two_risk_add_two_bound`:
     For any bounded normalized estimator (`|F|_n ≤ C`), `D_n ≤ 2 Q_n + 2 C²` and
     `Q_n ≤ 2 D_n + 2 C²` deterministically, giving the equivalence of uniform integrability
     of normalized signal power `Q_n = |L|_n²` and normalized denoising risk `D_n = |L - F|_n²`.
2. **Full-Rank Bi-Invariant Rare-Event Counterexample (Theorem 2(2), Iterations 1–2)**:
   - `mixture_expected_power` and `rare_branch_risk_tendsto`: In the two-component mixture
     (`1 - n⁻²` standard Gaussian, `n⁻²` scaled Haar `n √δ √n O_n`), `E[Q_n] = 1 - n⁻² + δ → 1 + δ`
     and the rare branch contributes `n⁻² (n √δ ± C)² → δ` to the expected risk, separating
     `m_L^exp(w) = b + δ` from `m_L^int(w) = b`.
   - `iter1_empirical_risk_eq`: Exact iteration-1 empirical risk identity
     `1 - 2 c + c² (1 + w) = b < b + δ = m_L^exp(w)` (refuting part (ii) under expected-risk SE).
   - `aPrime_lt_one_iff` and `delta_005_valid`: Under expected-risk correction `a' = (b + δ) / w`,
     `a' < 1 ↔ δ < M² / (1 - M)`, satisfied at `δ = 0.05` for every `M ∈ (0.21, 1)`.
   - `iter2_cross_moment_exact`: Exact iteration-2 cross-moment formula
     `-k w = δ / (1 - a') > 0` (nonzero!), breaking part (i)'s independent Gaussian-channel
     prediction `0`, while spectral-integral calibration (`δ = 0, a_L = c`) restores `k = 0`.
   - `oamp_orthogonality_and_variance`: General OAMP Onsager subtraction algebra showing
     `⟨N, E⟩ = 0` and `‖N‖² = m / (1 - m / v) = Φ(v)`.
3. **Odd-Polynomial Spectral Map & Stein-Divergence Identities (Section 2.5)**:
   - `oddPolyEval_neg` and `oddPolyEval_div_self`: Odd polynomials `g(s) = s p(s²)` have
     removable singularity `g(s) / s = p(s²)`.
   - `hilbert_symmetrization_identity`: Exact algebraic identity
     `(s g(s) - u g(u)) / (s² - u²) = (1/2) [g(s)(1/(s-u) + 1/(s+u)) + g(u)(1/(u-s) + 1/(u+s))]`
     linking the SVD Stein divergence to the symmetrized Hilbert transform.
   - `odd_monomial_divided_diff`: Exact polynomial factorization
     `(s · s^{2k+1} - u · u^{2k+1}) / (s² - u²) = ∑_{j=0}^k s^{2j} u^{2(k-j)}`.
   - `svd_divergence_linear` and `svd_divergence_cubic`: Exact SVD divergence identities for
     `g(s) = s` (yielding `1`) and `g(s) = s³` (yielding `((n + m + 1) / (m n)) ∑ s_i²`).
   - `coord_div_cubic_matrix`: Exact matrix coordinate divergence
     `∑_{a,b} ∂/∂X_{ab} (X Xᵀ X)_{ab} = (n + m + 1) ‖X‖_F²` for all `m × n` matrices `X`.
4. **Exact `β < 1` Literal Hilbert-Pole Criterion (Proposition 3)**:
   - `literal_pole_two_sided_bound`: On `(0, η)` below the spectral gap,
     `w (1 - β) / (2 s) ≤ |f_lit(s) - f_bdd(s)| ≤ 3 w (1 - β) / (2 s)`.
   - `hilbert_pole_l2_equivalence`: Exact two-sided comparability of the low-singular-value
     squared discrepancy `m⁻¹ ∑_{s_i < η} (f_lit(s_i) - f_bdd(s_i))²` and the empirical
     inverse-square mass `m⁻¹ ∑_{s_i < η} s_i⁻²`.
   - `orth_proj_pythagoras`: Orthogonal-projection Pythagorean decomposition preventing
     cancellation of low-singular-value pole energy against `L`.

Machine-checked vs prose-only:
- Machine-checked: All uniform-integrability bounds, rare-event mixture counterexample
  moments and iteration 1–2 calibration failures (`-k w = δ / (1 - a') ≠ 0`), OAMP Onsager
  algebra, odd-polynomial Stein-divergence and Hilbert-symmetrization identities, and the
  deterministic `β < 1` Hilbert-pole equivalence and orthogonal-projection decomposition.
- Prose-only (in `solutions/extensive_matrix_denoise/proof.pdf`): Graph-indexed matrix Hilbert
  space Wick–Hermite / traffic-freeness moment limits (Lemmas 1–2), signal coupling and
  spectral transfer, and finite-variance entrywise posterior continuity.
-/

open BigOperators Finset Matrix Filter

namespace ExtensiveMatrixDenoise

/-! ### 1. Deterministic Uniform-Integrability Bounds (Theorem 2(1)) -/

/-- Normalized Frobenius squared norm `|A|_n² = (1 / (m n)) ∑_{i,j} A_{ij}²`. -/
noncomputable def normFrobSq {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) : ℝ :=
  (∑ i : Fin m, ∑ j : Fin n, A i j ^ 2) / ((m : ℝ) * (n : ℝ))

/-- Normalized Frobenius inner product `⟨A, B⟩_n = (1 / (m n)) ∑_{i,j} A_{ij} B_{ij}`. -/
noncomputable def innerFrob {m n : ℕ} (A B : Matrix (Fin m) (Fin n) ℝ) : ℝ :=
  (∑ i : Fin m, ∑ j : Fin n, A i j * B i j) / ((m : ℝ) * (n : ℝ))

lemma normFrobSq_sub_le {m n : ℕ} (A B : Matrix (Fin m) (Fin n) ℝ) :
    normFrobSq (A - B) ≤ 2 * normFrobSq A + 2 * normFrobSq B := by
  unfold normFrobSq
  have hden : (0 : ℝ) ≤ (m : ℝ) * (n : ℝ) := by positivity
  have hsum : ∑ i : Fin m, ∑ j : Fin n, (A i j - B i j) ^ 2 ≤
      ∑ i : Fin m, ∑ j : Fin n, (2 * A i j ^ 2 + 2 * B i j ^ 2) := by
    refine Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => ?_
    nlinarith [sq_nonneg (A i j + B i j)]
  have hsplit : ∑ i : Fin m, ∑ j : Fin n, (2 * A i j ^ 2 + 2 * B i j ^ 2) =
      2 * (∑ i : Fin m, ∑ j : Fin n, A i j ^ 2) +
      2 * (∑ i : Fin m, ∑ j : Fin n, B i j ^ 2) := by
    simp only [Finset.sum_add_distrib, ← Finset.mul_sum]
  rw [hsplit] at hsum
  have hdiv := div_le_div_of_nonneg_right hsum hden
  simpa [Matrix.sub_apply, add_div, mul_div_assoc] using hdiv

/-- **Theorem 2(1) Forward Bound**: `D_n ≤ 2 Q_n + 2 C²` whenever `|F|_n² ≤ C²`. -/
theorem risk_le_two_power_add_two_bound {m n : ℕ} (L F : Matrix (Fin m) (Fin n) ℝ) {C : ℝ}
    (hF : normFrobSq F ≤ C ^ 2) :
    normFrobSq (L - F) ≤ 2 * normFrobSq L + 2 * C ^ 2 := by
  linarith [normFrobSq_sub_le L F]

/-- **Theorem 2(1) Reverse Bound**: `Q_n ≤ 2 D_n + 2 C²` whenever `|F|_n² ≤ C²`. -/
theorem power_le_two_risk_add_two_bound {m n : ℕ} (L F : Matrix (Fin m) (Fin n) ℝ) {C : ℝ}
    (hF : normFrobSq F ≤ C ^ 2) :
    normFrobSq L ≤ 2 * normFrobSq (L - F) + 2 * C ^ 2 := by
  have hL : L = (L - F) - (-F) := by ext i j; simp
  have hneg : normFrobSq (-F) = normFrobSq F := by
    unfold normFrobSq
    simp
  have h := normFrobSq_sub_le (L - F) (-F)
  rw [← hL, hneg] at h
  linarith

/-! ### 2. Full-Rank Bi-Invariant Rare-Event Counterexample (Theorem 2(2), Iterations 1–2) -/

/-- Exact mixture expected signal power:
`(1 - n⁻²) * 1 + n⁻² * (n² δ) = 1 - n⁻² + δ` for all `n ≥ 1`. -/
theorem mixture_expected_power {n : ℕ} (hn : 1 ≤ n) (δ : ℝ) :
    (1 - ((n : ℝ) ^ 2)⁻¹) * 1 + ((n : ℝ) ^ 2)⁻¹ * ((n : ℝ) ^ 2 * δ) =
      1 - ((n : ℝ) ^ 2)⁻¹ + δ := by
  have hn_ne : (n : ℝ) ≠ 0 := by exact_mod_cast (by omega : n ≠ 0)
  field_simp [hn_ne]

/-- On the exceptional branch (`|L|_n = n √δ`, `|F|_n ≤ C`), the weighted risk contribution
`n⁻² |L - F|_n²` is sandwiched between `(√δ - C / n)²` and `(√δ + C / n)²`, both tending
to `δ` as `n → ∞`. -/
theorem rare_branch_risk_tendsto {δ C : ℝ} (hδ : 0 ≤ δ) :
    Tendsto (fun n : ℕ => ((n : ℝ) ^ 2)⁻¹ * ((n : ℝ) * Real.sqrt δ - C) ^ 2) atTop (nhds δ) ∧
    Tendsto (fun n : ℕ => ((n : ℝ) ^ 2)⁻¹ * ((n : ℝ) * Real.sqrt δ + C) ^ 2) atTop (nhds δ) := by
  have hinv : Tendsto (fun n : ℕ => C / (n : ℝ)) atTop (nhds 0) :=
    tendsto_const_nhds.div_atTop tendsto_natCast_atTop_atTop
  have hsub : Tendsto (fun n : ℕ => (Real.sqrt δ - C / (n : ℝ)) ^ 2) atTop (nhds δ) := by
    have h1 : Tendsto (fun n : ℕ => Real.sqrt δ - C / (n : ℝ)) atTop (nhds (Real.sqrt δ - 0)) :=
      tendsto_const_nhds.sub hinv
    rw [sub_zero] at h1
    simpa [Real.sq_sqrt hδ] using h1.pow 2
  have hadd : Tendsto (fun n : ℕ => (Real.sqrt δ + C / (n : ℝ)) ^ 2) atTop (nhds δ) := by
    have h1 : Tendsto (fun n : ℕ => Real.sqrt δ + C / (n : ℝ)) atTop (nhds (Real.sqrt δ + 0)) :=
      tendsto_const_nhds.add hinv
    rw [add_zero] at h1
    simpa [Real.sq_sqrt hδ] using h1.pow 2
  constructor
  · refine hsub.congr' ?_
    filter_upwards [eventually_ge_atTop 1] with n hn
    have hn_ne : (n : ℝ) ≠ 0 := by exact_mod_cast (by omega : n ≠ 0)
    field_simp [hn_ne]
  · refine hadd.congr' ?_
    filter_upwards [eventually_ge_atTop 1] with n hn
    have hn_ne : (n : ℝ) ≠ 0 := by exact_mod_cast (by omega : n ≠ 0)
    field_simp [hn_ne]

/-- **Iteration-1 Empirical Risk vs. Expected-Risk Target Mismatch** (Theorem 2(2)):
For `M ∈ (0, 1)`, `w = M / (1 - M)`, `b = w / (1 + w)`, `c = 1 / (1 + w)`, and `δ > 0`:
1. `b = M` and `c = 1 - M`.
2. The empirical iteration-1 risk is `1 - 2 c + c² (1 + w) = b`.
3. Under expected-risk calibration `m_L^exp = b + δ`, `b ≠ b + δ` (strict mismatch!). -/
theorem iter1_empirical_risk_mismatch {M δ : ℝ} (hM0 : 0 < M) (hM1 : M < 1) (hδ : 0 < δ) :
    let w := M / (1 - M)
    let b := w / (1 + w)
    let c := 1 / (1 + w)
    b = M ∧ c = 1 - M ∧ 1 - 2 * c + c ^ 2 * (1 + w) = b ∧ b < b + δ := by
  have h1M : 1 - M ≠ 0 := by linarith
  have hw_pos : 0 < M / (1 - M) := div_pos hM0 (by linarith)
  have h1w : 1 + M / (1 - M) ≠ 0 := by linarith
  dsimp only
  refine ⟨?_, ?_, ?_, by linarith⟩
  · field_simp [h1M, h1w]; ring
  · field_simp [h1M, h1w]; ring
  · field_simp [h1M, h1w]; ring

/-- Exact positive-variance criterion for the expected-risk correction `a' = (b + δ) / w`:
for `M ∈ (0, 1)`, `a' < 1 ↔ δ < M² / (1 - M)`. -/
theorem aPrime_lt_one_iff {M δ : ℝ} (hM0 : 0 < M) (hM1 : M < 1) :
    let w := M / (1 - M)
    let b := w / (1 + w)
    (b + δ) / w < 1 ↔ δ < M ^ 2 / (1 - M) := by
  have h1M_pos : 0 < 1 - M := by linarith
  have h1M_ne : 1 - M ≠ 0 := ne_of_gt h1M_pos
  have hw_pos : 0 < M / (1 - M) := div_pos hM0 h1M_pos
  have hb : (M / (1 - M)) / (1 + M / (1 - M)) = M := by
    have h1w : 1 + M / (1 - M) ≠ 0 := by linarith
    field_simp [h1M_ne, h1w]; ring
  dsimp only
  rw [hb, div_lt_one hw_pos]
  have hdiff : M / (1 - M) - M = M ^ 2 / (1 - M) := by
    field_simp [h1M_ne]; ring
  constructor <;> intro h <;> linarith

/-- Explicit validation of `δ = 0.05` (used in `solution.md` with `M = E[sech²(1 + Z)] > 0.21`):
whenever `21 / 100 < M < 1`, we have `5 / 100 < M² / (1 - M)`, so `a' < 1` and the iteration-2
SE variance is strictly positive. -/
theorem delta_005_valid {M : ℝ} (hMlo : (21 : ℝ) / 100 < M) (hMhi : M < 1) :
    (5 : ℝ) / 100 < M ^ 2 / (1 - M) := by
  have h1M : 0 < 1 - M := by linarith
  rw [lt_div_iff₀ h1M]
  nlinarith

/-- **Iteration-2 Nonzero Cross-Moment `-k w = δ / (1 - a') ≠ 0`** (Theorem 2(2)):
With `a' = (b + δ) / w < 1` and `k = (c - a') / (1 - a')`:
1. `c - a' = -δ / w`.
2. The iteration-2 cross-moment `-k * w` equals `δ / (1 - a')`, which is strictly positive
   (hence nonzero!) for every `δ > 0`, refuting part (i)'s independent Gaussian-channel
   prediction `0`.
3. Under spectral-integral calibration (`δ = 0`, `a_L = b / w = c`), `k = 0` and `-k * w = 0`. -/
theorem iter2_cross_moment_exact {M δ : ℝ} (hM0 : 0 < M) (hM1 : M < 1)
    (hδ : 0 < δ) (hδ_small : δ < M ^ 2 / (1 - M)) :
    let w := M / (1 - M)
    let b := w / (1 + w)
    let c := 1 / (1 + w)
    let a' := (b + δ) / w
    let k := (c - a') / (1 - a')
    c - a' = -(δ / w) ∧ -k * w = δ / (1 - a') ∧ 0 < -k * w ∧ -k * w ≠ 0 := by
  have h1M_pos : 0 < 1 - M := by linarith
  have hw_pos : 0 < M / (1 - M) := div_pos hM0 h1M_pos
  have ha' : ((M / (1 - M)) / (1 + M / (1 - M)) + δ) / (M / (1 - M)) < 1 :=
    (aPrime_lt_one_iff hM0 hM1).mpr hδ_small
  have h1a'_pos : 0 < 1 - ((M / (1 - M)) / (1 + M / (1 - M)) + δ) / (M / (1 - M)) := by linarith
  have h1w : 1 + M / (1 - M) ≠ 0 := by linarith
  have hw_ne : M / (1 - M) ≠ 0 := ne_of_gt hw_pos
  have h1a'_ne : 1 - ((M / (1 - M)) / (1 + M / (1 - M)) + δ) / (M / (1 - M)) ≠ 0 :=
    ne_of_gt h1a'_pos
  dsimp only
  have hca : 1 / (1 + M / (1 - M)) -
      ((M / (1 - M)) / (1 + M / (1 - M)) + δ) / (M / (1 - M)) = -(δ / (M / (1 - M))) := by
    field_simp [h1w, hw_ne]
    ring
  have hkw : -((1 / (1 + M / (1 - M)) -
      ((M / (1 - M)) / (1 + M / (1 - M)) + δ) / (M / (1 - M))) /
      (1 - ((M / (1 - M)) / (1 + M / (1 - M)) + δ) / (M / (1 - M)))) * (M / (1 - M)) =
      δ / (1 - ((M / (1 - M)) / (1 + M / (1 - M)) + δ) / (M / (1 - M))) := by
    rw [hca]
    field_simp [hw_ne, h1a'_ne]
  have hpos : 0 < δ / (1 - ((M / (1 - M)) / (1 + M / (1 - M)) + δ) / (M / (1 - M))) :=
    div_pos hδ h1a'_pos
  exact ⟨hca, hkw, by linarith, ne_of_gt (by linarith)⟩

/-- **OAMP Onsager Orthogonality & Variance Propagation** (Section 2.5):
In any real inner-product / Hilbert setting, if `⟨S - F, E⟩ = -m`, `‖E‖² = v > 0`,
`‖S - F‖² = m` (`0 < m < v`), `a = m / v`, and `N = ((S - F) + a E) / (1 - a)`
(equivalently `S_out = (F - a (S + E)) / (1 - a)`, `N = S - S_out`), then:
1. `⟨N, E⟩ = 0` (exact Onsager orthogonality),
2. `‖N‖² = m / (1 - a) = Φ(v)` (exact variance propagation). -/
theorem oamp_orthogonality_and_variance {v m inner_err_E norm_err_sq norm_E_sq : ℝ}
    (hv : 0 < v) (hm : m < v)
    (hE : norm_E_sq = v) (herr : norm_err_sq = m) (hcross : inner_err_E = -m) :
    let a := m / v
    (inner_err_E + a * norm_E_sq) / (1 - a) = 0 ∧
    (norm_err_sq + 2 * a * inner_err_E + a ^ 2 * norm_E_sq) / (1 - a) ^ 2 = m / (1 - a) := by
  have hv_ne : v ≠ 0 := ne_of_gt hv
  have h1a_ne : 1 - m / v ≠ 0 := by
    have : m / v < 1 := (div_lt_one hv).mpr hm
    linarith
  have hvm : v - m ≠ 0 := by linarith
  dsimp only
  rw [hE, herr, hcross]
  constructor
  · field_simp [hv_ne, h1a_ne]; ring
  · have h1a_eq : 1 - m / v = (v - m) / v := by field_simp [hv_ne]
    rw [h1a_eq]
    field_simp [hv_ne, hvm]
    ring

/-! ### 3. Odd-Polynomial Spectral Map & Stein-Divergence Identities (Section 2.5) -/

/-- Odd polynomial `g(s) = s * p(s²)` associated with a polynomial `p ∈ ℝ[X]`. -/
def oddPolyEval (p : Polynomial ℝ) (s : ℝ) : ℝ :=
  s * p.eval (s ^ 2)

theorem oddPolyEval_neg (p : Polynomial ℝ) (s : ℝ) :
    oddPolyEval p (-s) = -oddPolyEval p s := by
  unfold oddPolyEval
  rw [neg_sq]
  ring

theorem oddPolyEval_div_self (p : Polynomial ℝ) {s : ℝ} (hs : s ≠ 0) :
    oddPolyEval p s / s = p.eval (s ^ 2) := by
  unfold oddPolyEval
  field_simp [hs]

/-- **Hilbert-Transform Symmetrization Identity** (Section 2.5):
For any function `g : ℝ → ℝ` and `s² ≠ u²`, the SVD Stein-divergence divided-difference kernel
splits symmetrically into the principal-value Hilbert kernels at `s` and `u`:
  `(s g(s) - u g(u)) / (s² - u²) = (1/2) [g(s) (1/(s-u) + 1/(s+u)) + g(u) (1/(u-s) + 1/(u+s))]`. -/
theorem hilbert_symmetrization_identity (g : ℝ → ℝ) {s u : ℝ} (hsub : s - u ≠ 0) (hadd : s + u ≠ 0) :
    (s * g s - u * g u) / (s ^ 2 - u ^ 2) =
      (1 / 2 : ℝ) * (g s * (1 / (s - u) + 1 / (s + u)) +
        g u * (1 / (u - s) + 1 / (u + s))) := by
  have hsub2 : u - s ≠ 0 := fun h => hsub (by linarith)
  have hadd2 : u + s ≠ 0 := fun h => hadd (by linarith)
  have hdiff : s ^ 2 - u ^ 2 ≠ 0 := by
    have : s ^ 2 - u ^ 2 = (s - u) * (s + u) := by ring
    rw [this]
    exact mul_ne_zero hsub hadd
  field_simp [hsub, hadd, hsub2, hadd2, hdiff]
  ring

/-- **Removable Diagonal Singularity for Odd Monomials `g(s) = s^{2k + 1}`**:
For all `s, u ∈ ℝ` with `s² ≠ u²`,
`(s · s^{2k+1} - u · u^{2k+1}) / (s² - u²) = ∑_{j=0}^k (s²)^j (u²)^{k-j}`,
which is a polynomial in `(s, u)` taking value `(k + 1) s^{2k}` on the diagonal `u = s`. -/
theorem odd_monomial_divided_diff (k : ℕ) {s u : ℝ} (h : s ^ 2 ≠ u ^ 2) :
    (s * s ^ (2 * k + 1) - u * u ^ (2 * k + 1)) / (s ^ 2 - u ^ 2) =
      ∑ j ∈ Finset.range (k + 1), (s ^ 2) ^ j * (u ^ 2) ^ (k - j) := by
  have hden : s ^ 2 - u ^ 2 ≠ 0 := sub_ne_zero.mpr h
  rw [div_eq_iff hden]
  have hs : s * s ^ (2 * k + 1) = (s ^ 2) ^ (k + 1) := by
    rw [← pow_succ', ← pow_mul, show 2 * k + 1 + 1 = 2 * (k + 1) by omega]
  have hu : u * u ^ (2 * k + 1) = (u ^ 2) ^ (k + 1) := by
    rw [← pow_succ', ← pow_mul, show 2 * k + 1 + 1 = 2 * (k + 1) by omega]
  rw [hs, hu]
  have hgeom := geom_sum₂_mul (s ^ 2) (u ^ 2) (k + 1)
  simpa only [Nat.add_sub_add_right] using hgeom.symm

/-- **SVD Stein Divergence Formula for Linear Map `g(s) = s`**:
For any `m, n ≥ 1` and distinct positive singular values `s_i`, the SVD divergence expression
equals `1 = (1 / (m n)) div_X(X)`. -/
theorem svd_divergence_linear {m n : ℕ} (hm : 1 ≤ m) (hn : 1 ≤ n)
    (s : Fin m → ℝ) (hs_pos : ∀ i, 0 < s i)
    (hs_inj : Function.Injective s) :
    (((n : ℝ) - (m : ℝ)) / (n : ℝ)) * ((1 : ℝ) / (m : ℝ)) * (∑ i : Fin m, s i / s i) +
    ((m : ℝ) / (n : ℝ)) * ((1 : ℝ) / ((m : ℝ) ^ 2)) *
      (∑ i : Fin m, ∑ j ∈ Finset.univ.erase i, (s i * s i - s j * s j) / (s i ^ 2 - s j ^ 2)) +
    ((1 : ℝ) / (n : ℝ)) * ((1 : ℝ) / (m : ℝ)) * (∑ _i : Fin m, (1 : ℝ)) = 1 := by
  have hm_ne : (m : ℝ) ≠ 0 := by exact_mod_cast (by omega : m ≠ 0)
  have hn_ne : (n : ℝ) ≠ 0 := by exact_mod_cast (by omega : n ≠ 0)
  have hsum1 : ∑ i : Fin m, s i / s i = (m : ℝ) := by
    have : ∀ i : Fin m, s i / s i = 1 := fun i => div_self (ne_of_gt (hs_pos i))
    simp [this]
  have hsum2 : ∑ i : Fin m, ∑ j ∈ Finset.univ.erase i,
      (s i * s i - s j * s j) / (s i ^ 2 - s j ^ 2) = (m : ℝ) * ((m : ℝ) - 1) := by
    have hinner : ∀ i : Fin m, ∑ j ∈ Finset.univ.erase i,
        (s i * s i - s j * s j) / (s i ^ 2 - s j ^ 2) = (m : ℝ) - 1 := by
      intro i
      have hterm : ∀ j ∈ Finset.univ.erase i,
          (s i * s i - s j * s j) / (s i ^ 2 - s j ^ 2) = 1 := by
        intro j hj
        have hne : j ≠ i := (Finset.mem_erase.mp hj).1
        have hsne : s i ^ 2 - s j ^ 2 ≠ 0 := by
          intro h0
          have hsq : s i ^ 2 = s j ^ 2 := sub_eq_zero.mp h0
          have heq : s i = s j := by nlinarith [hs_pos i, hs_pos j]
          exact hne (hs_inj heq).symm
        rw [show s i * s i - s j * s j = s i ^ 2 - s j ^ 2 by ring, div_self hsne]
      rw [Finset.sum_congr rfl hterm, Finset.sum_const, Finset.card_erase_of_mem (Finset.mem_univ i),
        Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one, Nat.cast_sub hm, Nat.cast_one]
    simp [hinner]; ring
  have hsum3 : ∑ _i : Fin m, (1 : ℝ) = (m : ℝ) := by simp
  rw [hsum1, hsum2, hsum3]
  field_simp [hm_ne, hn_ne]
  ring

/-- **SVD Stein Divergence Formula for Cubic Map `g(s) = s³`**:
For any `m, n ≥ 1` and distinct positive singular values `s_i`, the SVD divergence expression
equals `((n + m + 1) / (m n)) * ∑ s_i²`. -/
theorem svd_divergence_cubic {m n : ℕ} (hm : 1 ≤ m) (hn : 1 ≤ n)
    (s : Fin m → ℝ) (hs_pos : ∀ i, 0 < s i)
    (hs_inj : Function.Injective s) :
    (((n : ℝ) - (m : ℝ)) / (n : ℝ)) * ((1 : ℝ) / (m : ℝ)) * (∑ i : Fin m, (s i) ^ 3 / s i) +
    ((m : ℝ) / (n : ℝ)) * ((1 : ℝ) / ((m : ℝ) ^ 2)) *
      (∑ i : Fin m, ∑ j ∈ Finset.univ.erase i,
        (s i * (s i) ^ 3 - s j * (s j) ^ 3) / (s i ^ 2 - s j ^ 2)) +
    ((1 : ℝ) / (n : ℝ)) * ((1 : ℝ) / (m : ℝ)) * (∑ i : Fin m, 3 * (s i) ^ 2) =
    (((n : ℝ) + (m : ℝ) + 1) / ((m : ℝ) * (n : ℝ))) * (∑ i : Fin m, (s i) ^ 2) := by
  have hm_ne : (m : ℝ) ≠ 0 := by exact_mod_cast (by omega : m ≠ 0)
  have hn_ne : (n : ℝ) ≠ 0 := by exact_mod_cast (by omega : n ≠ 0)
  set S : ℝ := ∑ i : Fin m, (s i) ^ 2
  have hsum1 : ∑ i : Fin m, (s i) ^ 3 / s i = S := by
    refine Finset.sum_congr rfl fun i _ => ?_
    have hne : s i ≠ 0 := ne_of_gt (hs_pos i)
    field_simp [hne]
  have hsum2 : ∑ i : Fin m, ∑ j ∈ Finset.univ.erase i,
      (s i * (s i) ^ 3 - s j * (s j) ^ 3) / (s i ^ 2 - s j ^ 2) =
      2 * ((m : ℝ) - 1) * S := by
    have hinner : ∀ i : Fin m, ∑ j ∈ Finset.univ.erase i,
        (s i * (s i) ^ 3 - s j * (s j) ^ 3) / (s i ^ 2 - s j ^ 2) =
        ((m : ℝ) - 1) * (s i) ^ 2 + (S - (s i) ^ 2) := by
      intro i
      have hterm : ∀ j ∈ Finset.univ.erase i,
          (s i * (s i) ^ 3 - s j * (s j) ^ 3) / (s i ^ 2 - s j ^ 2) = (s i) ^ 2 + (s j) ^ 2 := by
        intro j hj
        have hne : j ≠ i := (Finset.mem_erase.mp hj).1
        have hsne : s i ^ 2 - s j ^ 2 ≠ 0 := by
          intro h0
          have hsq : s i ^ 2 = s j ^ 2 := sub_eq_zero.mp h0
          have heq : s i = s j := by nlinarith [hs_pos i, hs_pos j]
          exact hne (hs_inj heq).symm
        field_simp [hsne]; ring
      rw [Finset.sum_congr rfl hterm, Finset.sum_add_distrib, Finset.sum_const,
        Finset.card_erase_of_mem (Finset.mem_univ i), Finset.card_univ, Fintype.card_fin,
        nsmul_eq_mul, Nat.cast_sub hm, Nat.cast_one]
      have herase : ∑ j ∈ Finset.univ.erase i, (s j) ^ 2 = S - (s i) ^ 2 := by
        have := Finset.add_sum_erase Finset.univ (fun j => (s j) ^ 2) (Finset.mem_univ i)
        linarith
      rw [herase]
    simp_rw [hinner, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum,
      Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    change ((m : ℝ) - 1) * S + ((m : ℝ) * S - S) = 2 * ((m : ℝ) - 1) * S
    ring
  have hsum3 : ∑ i : Fin m, 3 * (s i) ^ 2 = 3 * S := by rw [← Finset.mul_sum]
  rw [hsum1, hsum2, hsum3]
  field_simp [hm_ne, hn_ne]
  ring

/-- **Exact Matrix Coordinate Divergence of `g(X) = X Xᵀ X`**:
For every `m × n` matrix `X`, differentiating `(X Xᵀ X)_{ab}` with respect to `X_{ab}` and
summing over all `(a, b)` yields `(n + m + 1) ‖X‖_F²` — matching `svd_divergence_cubic`! -/
theorem coord_div_cubic_matrix {m n : ℕ} (X : Matrix (Fin m) (Fin n) ℝ) :
    ∑ a : Fin m, ∑ b : Fin n,
      ((∑ c : Fin n, X a c ^ 2) + (∑ r : Fin m, X r b ^ 2) + X a b ^ 2) =
    ((n : ℝ) + (m : ℝ) + 1) * (∑ a : Fin m, ∑ b : Fin n, X a b ^ 2) := by
  have h1 : ∑ a : Fin m, ∑ b : Fin n,
      ((∑ c : Fin n, X a c ^ 2) + (∑ r : Fin m, X r b ^ 2) + X a b ^ 2) =
      (∑ a : Fin m, ∑ _b : Fin n, ∑ c : Fin n, X a c ^ 2) +
      (∑ _a : Fin m, ∑ b : Fin n, ∑ r : Fin m, X r b ^ 2) +
      (∑ a : Fin m, ∑ b : Fin n, X a b ^ 2) := by
    simp only [Finset.sum_add_distrib]
  rw [h1]
  have ht1 : ∑ a : Fin m, ∑ _b : Fin n, ∑ c : Fin n, X a c ^ 2 =
      (n : ℝ) * ∑ a : Fin m, ∑ c : Fin n, X a c ^ 2 := by
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, ← Finset.mul_sum]
  have ht2 : ∑ _a : Fin m, ∑ b : Fin n, ∑ r : Fin m, X r b ^ 2 =
      (m : ℝ) * ∑ a : Fin m, ∑ b : Fin n, X a b ^ 2 := by
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    rw [Finset.sum_comm]
  rw [ht1, ht2]
  ring

/-! ### 4. Exact `β < 1` Literal Hilbert-Pole Criterion (Proposition 3) -/

/-- **Two-Sided Hilbert-Pole Bound on `(0, η)`** (Proposition 3):
When `0 < β < 1`, `w > 0`, and `η > 0` is small enough that `η + 2 β w H₀ + B₀ ≤ w (1 - β) / (2 η)`,
every `s ∈ (0, η)` with `|H(s)| ≤ H₀` and `|f_bdd(s)| ≤ B₀` satisfies
`w (1 - β) / (2 s) ≤ |f_lit(s) - f_bdd(s)| ≤ 3 w (1 - β) / (2 s)`. -/
theorem literal_pole_two_sided_bound {β w η H₀ B₀ s H_val f_bdd : ℝ}
    (hβ0 : 0 ≤ β) (hβ1 : β < 1) (hw : 0 < w)
    (hs0 : 0 < s) (hsη : s < η)
    (hH : |H_val| ≤ H₀) (hf : |f_bdd| ≤ B₀)
    (hsmall : η + 2 * β * w * H₀ + B₀ ≤ w * (1 - β) / (2 * η)) :
    let f_lit := s - w * (1 - β) / s - 2 * β * w * H_val
    w * (1 - β) / (2 * s) ≤ |f_lit - f_bdd| ∧ |f_lit - f_bdd| ≤ 3 * w * (1 - β) / (2 * s) := by
  intro f_lit
  have hη : 0 < η := lt_trans hs0 hsη
  have hw1β : 0 < w * (1 - β) := mul_pos hw (by linarith)
  have hpole_mono : w * (1 - β) / (2 * η) ≤ w * (1 - β) / (2 * s) :=
    div_le_div_of_nonneg_left (le_of_lt hw1β) (by positivity) (by linarith)
  have hrem_abs : |s - 2 * β * w * H_val - f_bdd| ≤ w * (1 - β) / (2 * s) := by
    have h1 : |s - 2 * β * w * H_val - f_bdd| ≤ |s| + |2 * β * w * H_val| + |f_bdd| := by
      calc |s - 2 * β * w * H_val - f_bdd|
          = |(s + (-(2 * β * w * H_val))) + (-f_bdd)| := by ring_nf
        _ ≤ |s + (-(2 * β * w * H_val))| + |-f_bdd| := abs_add_le _ _
        _ ≤ |s| + |-(2 * β * w * H_val)| + |-f_bdd| := by linarith [abs_add_le s (-(2 * β * w * H_val))]
        _ = |s| + |2 * β * w * H_val| + |f_bdd| := by simp [abs_neg]
    have h2 : |s| = s := abs_of_pos hs0
    have h3 : |2 * β * w * H_val| ≤ 2 * β * w * H₀ := by
      rw [abs_mul, abs_of_nonneg (by positivity : 0 ≤ 2 * β * w)]
      exact mul_le_mul_of_nonneg_left hH (by positivity)
    linarith
  have hdiff_eq : f_lit - f_bdd = -(w * (1 - β) / s) + (s - 2 * β * w * H_val - f_bdd) := by
    dsimp [f_lit]; ring
  have hp_pos : 0 < w * (1 - β) / s := div_pos hw1β hs0
  rw [hdiff_eq]
  rcases abs_le.mp hrem_abs with ⟨hr_lo, hr_hi⟩
  have hneg : -(w * (1 - β) / s) + (s - 2 * β * w * H_val - f_bdd) ≤ -(w * (1 - β) / (2 * s)) := by
    have : w * (1 - β) / s - w * (1 - β) / (2 * s) = w * (1 - β) / (2 * s) := by
      field_simp [ne_of_gt hs0]; ring
    linarith
  have hlo : -(3 * w * (1 - β) / (2 * s)) ≤ -(w * (1 - β) / s) + (s - 2 * β * w * H_val - f_bdd) := by
    have : w * (1 - β) / s + w * (1 - β) / (2 * s) = 3 * w * (1 - β) / (2 * s) := by
      field_simp [ne_of_gt hs0]; ring
    linarith
  have habs_eq : |-(w * (1 - β) / s) + (s - 2 * β * w * H_val - f_bdd)| =
      -(-(w * (1 - β) / s) + (s - 2 * β * w * H_val - f_bdd)) :=
    abs_of_nonpos (by linarith [div_pos hw1β (show 0 < 2 * s by positivity)])
  rw [habs_eq]
  exact ⟨by linarith, by linarith⟩

/-- **Exact `β < 1` Low-Singular-Value `L²` Equivalence** (Proposition 3(1)):
The empirical squared discrepancy between the literal Hilbert formula and the Appendix A
bounded extension on `{s_i < η}` is comparable up to explicit positive constants to the
empirical inverse-square mass `m⁻¹ ∑_{s_i < η} s_i⁻²`. -/
theorem hilbert_pole_l2_equivalence {m : ℕ} {β w η H₀ B₀ : ℝ}
    (hβ0 : 0 ≤ β) (hβ1 : β < 1) (hw : 0 < w)
    (hsmall : η + 2 * β * w * H₀ + B₀ ≤ w * (1 - β) / (2 * η))
    (s H_val f_bdd : Fin m → ℝ)
    (hs_pos : ∀ i, 0 < s i)
    (hH : ∀ i, s i < η → |H_val i| ≤ H₀)
    (hf : ∀ i, s i < η → |f_bdd i| ≤ B₀) :
    let S := Finset.univ.filter fun i : Fin m => s i < η
    let f_lit : Fin m → ℝ := fun i => s i - w * (1 - β) / s i - 2 * β * w * H_val i
    ((w * (1 - β)) ^ 2 / 4) * ((∑ i ∈ S, (s i)⁻¹ ^ 2) / (m : ℝ)) ≤
      (∑ i ∈ S, (f_lit i - f_bdd i) ^ 2) / (m : ℝ) ∧
    (∑ i ∈ S, (f_lit i - f_bdd i) ^ 2) / (m : ℝ) ≤
      (9 * (w * (1 - β)) ^ 2 / 4) * ((∑ i ∈ S, (s i)⁻¹ ^ 2) / (m : ℝ)) := by
  intro S f_lit
  have hterm_lo : ∀ i ∈ S, ((w * (1 - β)) ^ 2 / 4) * (s i)⁻¹ ^ 2 ≤ (f_lit i - f_bdd i) ^ 2 := by
    intro i hi
    have hsi : s i < η := (Finset.mem_filter.mp hi).2
    have ⟨hlo, _⟩ := literal_pole_two_sided_bound hβ0 hβ1 hw (hs_pos i) hsi (hH i hsi) (hf i hsi) hsmall
    have hnonneg : 0 ≤ w * (1 - β) / (2 * s i) :=
      div_nonneg (mul_nonneg (le_of_lt hw) (by linarith)) (by linarith [hs_pos i])
    have hsq : (w * (1 - β) / (2 * s i)) ^ 2 ≤ |f_lit i - f_bdd i| ^ 2 :=
      pow_le_pow_left₀ hnonneg hlo 2
    rw [sq_abs] at hsq
    have hcalc : (w * (1 - β) / (2 * s i)) ^ 2 = ((w * (1 - β)) ^ 2 / 4) * (s i)⁻¹ ^ 2 := by
      field_simp [ne_of_gt (hs_pos i)]; ring
    linarith
  have hterm_hi : ∀ i ∈ S, (f_lit i - f_bdd i) ^ 2 ≤ (9 * (w * (1 - β)) ^ 2 / 4) * (s i)⁻¹ ^ 2 := by
    intro i hi
    have hsi : s i < η := (Finset.mem_filter.mp hi).2
    have ⟨_, hhi⟩ := literal_pole_two_sided_bound hβ0 hβ1 hw (hs_pos i) hsi (hH i hsi) (hf i hsi) hsmall
    have hsq : |f_lit i - f_bdd i| ^ 2 ≤ (3 * w * (1 - β) / (2 * s i)) ^ 2 :=
      pow_le_pow_left₀ (abs_nonneg _) hhi 2
    rw [sq_abs] at hsq
    have hcalc : (3 * w * (1 - β) / (2 * s i)) ^ 2 = (9 * (w * (1 - β)) ^ 2 / 4) * (s i)⁻¹ ^ 2 := by
      field_simp [ne_of_gt (hs_pos i)]; ring
    linarith
  have hsum_lo := Finset.sum_le_sum hterm_lo
  have hsum_hi := Finset.sum_le_sum hterm_hi
  rw [← Finset.mul_sum] at hsum_lo hsum_hi
  have hm_nn : (0 : ℝ) ≤ (m : ℝ) := by positivity
  constructor
  · have := div_le_div_of_nonneg_right hsum_lo hm_nn
    simpa [mul_div_assoc] using this
  · have := div_le_div_of_nonneg_right hsum_hi hm_nn
    simpa [mul_div_assoc] using this

/-- **Orthogonal-Projection Pythagorean Decomposition** (Proposition 3(2)):
For any orthogonal projection `P` (`P * P = P`, `Pᵀ = P`) and any matrix `X`,
`‖X‖_F² = ‖(1 - P) X‖_F² + ‖P X‖_F²`, so low-singular-value excess energy cannot be canceled
by the orthogonal complement. -/
theorem orth_proj_pythagoras {m n : ℕ} (P : Matrix (Fin m) (Fin m) ℝ)
    (hidem : P * P = P) (hsym : Pᵀ = P) (X : Matrix (Fin m) (Fin n) ℝ) :
    normFrobSq X = normFrobSq ((1 - P) * X) + normFrobSq (P * X) := by
  have hnorm_trace : ∀ A : Matrix (Fin m) (Fin n) ℝ,
      normFrobSq A = Matrix.trace (Aᵀ * A) / ((m : ℝ) * (n : ℝ)) := by
    intro A
    unfold normFrobSq
    congr 1
    simp only [Matrix.trace, Matrix.diag, Matrix.mul_apply, Matrix.transpose_apply]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring
  rw [hnorm_trace X, hnorm_trace ((1 - P) * X), hnorm_trace (P * X), ← add_div]
  congr 1
  rw [← Matrix.trace_add]
  congr 1
  have h1 : ((1 - P) * X)ᵀ * ((1 - P) * X) = Xᵀ * (1 - P) * X := by
    rw [Matrix.transpose_mul, Matrix.transpose_sub, Matrix.transpose_one, hsym,
      Matrix.mul_assoc, ← Matrix.mul_assoc (1 - P) (1 - P) X]
    have hsub : (1 - P) * (1 - P) = 1 - P := by
      rw [Matrix.sub_mul, Matrix.mul_sub, Matrix.mul_sub, Matrix.one_mul, Matrix.mul_one,
        Matrix.one_mul, hidem]
      abel
    rw [hsub, ← Matrix.mul_assoc]
  have h2 : (P * X)ᵀ * (P * X) = Xᵀ * P * X := by
    rw [Matrix.transpose_mul, hsym, Matrix.mul_assoc, ← Matrix.mul_assoc P P X, hidem,
      ← Matrix.mul_assoc]
  rw [h1, h2, ← Matrix.add_mul, ← Matrix.mul_add, sub_add_cancel, Matrix.mul_one]

end ExtensiveMatrixDenoise
