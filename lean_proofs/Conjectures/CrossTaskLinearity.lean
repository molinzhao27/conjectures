import Mathlib

/-!
# Refutation of Cross-Task Linearity Transitivity
(arXiv:2402.03660, ICML 2024, Conjecture 4.1)

We formalize in Lean 4 + Mathlib the refutation and exact characterizations of Cross-Task
Linearity (CTL) transitivity from Zhanpeng Zhou et al., *On the Emergence of Cross-Task
Linearity in Pretraining-Finetuning Paradigm* (ICML 2024, arXiv:2402.03660):

1. **Collinear one-layer ReLU refutation (Theorem 1)**:
   - `collinear_ctl_AB`, `collinear_ctl_BC`: For `F_d((u, v); x) = (ReLU(ux), ReLU(vx), d)`
     and collinear checkpoints `A = (M, -M)`, `B = (0, 0)`, `C = (-M, M)` (`M > 0, d > 0`),
     exact CTL holds on both `(A, B)` and `(B, C)` for every `x > 0` and `t ∈ [0, 1]`.
   - `collinear_AC_midpoint_diagnostics`, `collinear_AC_midpoint_error_unit_M`: At `t = 1/2`
     on `(A, C)`, the Euclidean gap equals `M * x / √2` (`x / √2` at `M = 1`),
     `cosine = d / √(d² + (Mx)²/2)`, and `coef = d² / (d² + (Mx)²/2)`.
   - `collinear_AC_asymptotics`: As `M → ∞` (for fixed `d, x > 0`), error diverges to `+∞`
     and `cosine → 0` while both premise pairs remain exact.
2. **Shared-checkpoint one-step GD construction (Proposition 2)**:
   - `pretrain_unique_global_min`: On `D₀ = {(1/2, 1), (1/2, 2)}` with `d = 1/10`, `P = (1, 1)`
     is the unique global minimizer of `L_{D₀}` over `ℝ × ℝ`.
   - `downstream_gradients_at_P`: Exact partial derivatives at `P = (1, 1)` equal `(0, 1)`,
     `(1/2, 1/2)`, `(1, 0)`.
   - `gd_step_and_task_opt`: One gradient step `P - 2 ∇L(P)` produces `A = (1, -1)`,
     `B = (0, 0)`, `C = (-1, 1)`, each globally minimizing its task loss over `ℝ × ℝ`.
3. **Exact one-layer ReLU transitivity iff criterion (Theorem 4.1)**:
   - `relu_chord_max_gap`: When `a * c < 0`, the gap attains its exact maximum
     `|a * c| / (|a| + |c|) > 0` at `t* = a / (a - c) ∈ (0, 1)`.
   - `relu_chord_iff`: `ReLU((1-t)a + tc) = (1-t)ReLU(a) + t ReLU(c)` for all `t ∈ [0, 1]`
     if and only if `0 ≤ a * c`.
   - `one_layer_transitivity_failure_iff`: Under exact premises `0 ≤ a * b` and `0 ≤ b * c`,
     `(a, c)` fails exact CTL if and only if `b = 0 ∧ a * c < 0`.
4. **Strictly active two-layer counterexample (Proposition 3)**:
   - `two_layer_strictly_active_conv`: For `A = (1+r, 1)`, `B = (1, 1)`, `C = (1, 1+s)`
     (`r, s > 0`), every neuron is strictly active throughout `conv(A, B, C)`.
   - `two_layer_strictly_active_counterexample`: Exact layer-1 and layer-2 CTL on `(A, B)`
     and `(B, C)`, exact layer-1 CTL on `(A, C)`, and exact layer-2 gap `t(1-t)rsx` on `(A, C)`.
   - `two_layer_pos_ctl_iff`: On `(0, ∞)²`, exact CTL holds iff `(u_q - u_p)(v_q - v_p) = 0`.

Machine-checked vs prose-only:
- Machine-checked: Theorem 1, Proposition 2, Proposition 3, and Theorem 4.1.
- Prose-only (in `solutions/cross_task_linearity/proof.pdf`): General depth-`L` inductive
  layer-order criterion (Theorem 4.2), Zorn's-lemma maximal affine region characterization
  (Theorem 4.3), and the rational grid verification script.
-/

open BigOperators Finset Filter Topology

namespace CrossTaskLinearity

/-- ReLU activation `σ(s) = max(s, 0)`. -/
def relu (s : ℝ) : ℝ := max s 0

lemma relu_of_nonneg {s : ℝ} (hs : 0 ≤ s) : relu s = s := max_eq_left hs
lemma relu_of_nonpos {s : ℝ} (hs : s ≤ 0) : relu s = 0 := max_eq_right hs
lemma relu_nonneg (s : ℝ) : 0 ≤ relu s := le_max_right s 0

/-! ### 1. Collinear One-Layer ReLU Counterexample (Theorem 1) -/

/-- One-layer 3-unit feature map `F_d((u, v); x) = (ReLU(ux), ReLU(vx), d)`. -/
def feat1 (d : ℝ) (θ : ℝ × ℝ) (x : ℝ) : Fin 3 → ℝ :=
  ![relu (θ.1 * x), relu (θ.2 * x), d]

/-- Linear interpolation of two parameter pairs. -/
def interp (t : ℝ) (p q : ℝ × ℝ) : ℝ × ℝ :=
  ((1 - t) * p.1 + t * q.1, (1 - t) * p.2 + t * q.2)

/-- Linear interpolation of two 3D feature vectors. -/
def interp3 (t : ℝ) (u v : Fin 3 → ℝ) : Fin 3 → ℝ :=
  fun i => (1 - t) * u i + t * v i

/-- Squared Euclidean norm on `Fin 3 → ℝ`. -/
def normSq3 (v : Fin 3 → ℝ) : ℝ :=
  v 0 ^ 2 + v 1 ^ 2 + v 2 ^ 2

/-- Euclidean norm on `Fin 3 → ℝ`. -/
noncomputable def norm3 (v : Fin 3 → ℝ) : ℝ :=
  Real.sqrt (normSq3 v)

/-- Inner product on `Fin 3 → ℝ`. -/
def dot3 (u v : Fin 3 → ℝ) : ℝ :=
  u 0 * v 0 + u 1 * v 1 + u 2 * v 2

/-- Cosine similarity diagnostic on `Fin 3 → ℝ`. -/
noncomputable def cosine3 (h g : Fin 3 → ℝ) : ℝ :=
  dot3 h g / (norm3 h * norm3 g)

/-- Coefficient diagnostic `⟨h, g⟩ / ‖g‖₂²` on `Fin 3 → ℝ`. -/
noncomputable def coef3 (h g : Fin 3 → ℝ) : ℝ :=
  dot3 h g / normSq3 g

/-- **Theorem 1(a)**: Exact CTL holds on `(A, B)` for `A = (M, -M)`, `B = (0, 0)`,
for every `M ≥ 0`, `x ≥ 0`, and `t ∈ [0, 1]`. -/
theorem collinear_ctl_AB {M d x t : ℝ} (hM : 0 ≤ M) (hx : 0 ≤ x)
    (_ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    feat1 d (interp t (M, -M) (0, 0)) x =
      interp3 t (feat1 d (M, -M) x) (feat1 d (0, 0) x) := by
  have h1t : 0 ≤ 1 - t := sub_nonneg.mpr ht1
  have hMx : 0 ≤ M * x := mul_nonneg hM hx
  ext i
  fin_cases i
  · change relu (((1 - t) * M + t * 0) * x) = (1 - t) * relu (M * x) + t * relu (0 * x)
    rw [show ((1 - t) * M + t * 0) * x = (1 - t) * (M * x) by ring,
        relu_of_nonneg (mul_nonneg h1t hMx), relu_of_nonneg hMx,
        zero_mul, relu_of_nonneg le_rfl]
    ring
  · change relu (((1 - t) * -M + t * 0) * x) = (1 - t) * relu ((-M) * x) + t * relu (0 * x)
    rw [show ((1 - t) * -M + t * 0) * x = -((1 - t) * (M * x)) by ring,
        show (-M) * x = -(M * x) by ring,
        relu_of_nonpos (by nlinarith), relu_of_nonpos (by linarith),
        zero_mul, relu_of_nonneg le_rfl]
    ring
  · change d = (1 - t) * d + t * d
    ring

/-- **Theorem 1(b)**: Exact CTL holds on `(B, C)` for `B = (0, 0)`, `C = (-M, M)`,
for every `M ≥ 0`, `x ≥ 0`, and `t ∈ [0, 1]`. -/
theorem collinear_ctl_BC {M d x t : ℝ} (hM : 0 ≤ M) (hx : 0 ≤ x)
    (ht0 : 0 ≤ t) (_ht1 : t ≤ 1) :
    feat1 d (interp t (0, 0) (-M, M)) x =
      interp3 t (feat1 d (0, 0) x) (feat1 d (-M, M) x) := by
  have hMx : 0 ≤ M * x := mul_nonneg hM hx
  ext i
  fin_cases i
  · change relu (((1 - t) * 0 + t * -M) * x) = (1 - t) * relu (0 * x) + t * relu ((-M) * x)
    rw [show ((1 - t) * 0 + t * -M) * x = -(t * (M * x)) by ring,
        show (-M) * x = -(M * x) by ring,
        zero_mul, relu_of_nonneg le_rfl, relu_of_nonpos (by nlinarith), relu_of_nonpos (by linarith)]
    ring
  · change relu (((1 - t) * 0 + t * M) * x) = (1 - t) * relu (0 * x) + t * relu (M * x)
    rw [show ((1 - t) * 0 + t * M) * x = t * (M * x) by ring,
        relu_of_nonneg (mul_nonneg ht0 hMx), relu_of_nonneg hMx,
        zero_mul, relu_of_nonneg le_rfl]
    ring
  · change d = (1 - t) * d + t * d
    ring

/-- **Theorem 1(c)**: On `(A, C)` at `t = 1/2`, the Euclidean interpolation gap equals `M * x / √2`,
`cosine = d / √(d² + (Mx)²/2)`, and `coef = d² / (d² + (Mx)²/2)`. -/
theorem collinear_AC_midpoint_diagnostics {M d x : ℝ} (hM : 0 < M) (hd : 0 < d) (hx : 0 < x) :
    let h := feat1 d (interp (1 / 2) (M, -M) (-M, M)) x
    let g := interp3 (1 / 2) (feat1 d (M, -M) x) (feat1 d (-M, M) x)
    norm3 (fun i => h i - g i) = M * x / Real.sqrt 2 ∧
    cosine3 h g = d / Real.sqrt (d ^ 2 + (M * x) ^ 2 / 2) ∧
    coef3 h g = d ^ 2 / (d ^ 2 + (M * x) ^ 2 / 2) := by
  intro h g
  have hMx : 0 ≤ M * x := by positivity
  have hh : h = ![0, 0, d] := by
    ext i; fin_cases i
    · change relu (((1 - 1 / 2 : ℝ) * M + 1 / 2 * -M) * x) = 0
      rw [show ((1 - 1 / 2 : ℝ) * M + 1 / 2 * -M) * x = 0 by ring, relu_of_nonneg le_rfl]
    · change relu (((1 - 1 / 2 : ℝ) * -M + 1 / 2 * M) * x) = 0
      rw [show ((1 - 1 / 2 : ℝ) * -M + 1 / 2 * M) * x = 0 by ring, relu_of_nonneg le_rfl]
    · rfl
  have hg : g = ![M * x / 2, M * x / 2, d] := by
    ext i; fin_cases i
    · change (1 - 1 / 2 : ℝ) * relu (M * x) + 1 / 2 * relu ((-M) * x) = M * x / 2
      rw [relu_of_nonneg hMx, show (-M) * x = -(M * x) by ring, relu_of_nonpos (by linarith)]
      ring
    · change (1 - 1 / 2 : ℝ) * relu ((-M) * x) + 1 / 2 * relu (M * x) = M * x / 2
      rw [show (-M) * x = -(M * x) by ring, relu_of_nonpos (by linarith), relu_of_nonneg hMx]
      ring
    · change (1 - 1 / 2 : ℝ) * d + 1 / 2 * d = d
      ring
  have hnorm_diff : norm3 (fun i => h i - g i) = M * x / Real.sqrt 2 := by
    rw [hh, hg]
    unfold norm3 normSq3
    have hsq : (0 - M * x / 2) ^ 2 + (0 - M * x / 2) ^ 2 + (d - d) ^ 2 =
        (M * x) ^ 2 / 2 := by ring
    change Real.sqrt ((0 - M * x / 2) ^ 2 + (0 - M * x / 2) ^ 2 + (d - d) ^ 2) = _
    rw [hsq, Real.sqrt_div (sq_nonneg _), Real.sqrt_sq hMx]
  have hdot : dot3 h g = d ^ 2 := by
    rw [hh, hg]; unfold dot3; change 0 * (M * x / 2) + 0 * (M * x / 2) + d * d = d ^ 2; ring
  have hnh : norm3 h = d := by
    rw [hh]; unfold norm3 normSq3
    change Real.sqrt (0 ^ 2 + 0 ^ 2 + d ^ 2) = d
    rw [show (0 : ℝ) ^ 2 + 0 ^ 2 + d ^ 2 = d ^ 2 by ring, Real.sqrt_sq (le_of_lt hd)]
  have hngSq : normSq3 g = d ^ 2 + (M * x) ^ 2 / 2 := by
    rw [hg]; unfold normSq3
    change (M * x / 2) ^ 2 + (M * x / 2) ^ 2 + d ^ 2 = d ^ 2 + (M * x) ^ 2 / 2
    ring
  refine ⟨hnorm_diff, ?_, ?_⟩
  · unfold cosine3
    rw [hdot, hnh, norm3, hngSq]
    have hdne : d ≠ 0 := ne_of_gt hd
    have hpos : 0 < Real.sqrt (d ^ 2 + (M * x) ^ 2 / 2) :=
      Real.sqrt_pos.mpr (by positivity)
    field_simp [hdne, ne_of_gt hpos]
  · unfold coef3
    rw [hdot, hngSq]

/-- At `M = 1`, the `(A, C)` midpoint gap is exactly `x / √2` (unbounded in `x`). -/
theorem collinear_AC_midpoint_error_unit_M {d x : ℝ} (hd : 0 < d) (hx : 0 < x) :
    norm3 (fun i =>
      feat1 d (interp (1 / 2) (1, -1) (-1, 1)) x i -
      interp3 (1 / 2) (feat1 d (1, -1) x) (feat1 d (-1, 1) x) i) = x / Real.sqrt 2 := by
  have h := (collinear_AC_midpoint_diagnostics one_pos hd hx).1
  simpa using h

/-- As `M → ∞` (for fixed `d, x > 0`), the `(A, C)` midpoint error diverges to `+∞` and
`cosine → 0`, while both premise pairs `(A, B)` and `(B, C)` remain exact. -/
theorem collinear_AC_asymptotics {d x : ℝ} (_hd : 0 < d) (hx : 0 < x) :
    Tendsto (fun M => M * x / Real.sqrt 2) atTop atTop ∧
    Tendsto (fun M => d / Real.sqrt (d ^ 2 + (M * x) ^ 2 / 2)) atTop (nhds 0) := by
  constructor
  · have hpos : 0 < x / Real.sqrt 2 := by positivity
    have heq : (fun M : ℝ => M * x / Real.sqrt 2) = fun M => M * (x / Real.sqrt 2) := by
      ext M; ring
    rw [heq]
    exact Filter.Tendsto.atTop_mul_const hpos tendsto_id
  · refine tendsto_const_nhds.div_atTop ?_
    have hsqrt_top : Tendsto Real.sqrt atTop atTop := by
      rw [tendsto_atTop_atTop]
      intro B
      refine ⟨max B 0 ^ 2, fun y hy => ?_⟩
      exact le_trans (le_max_left B 0) (by
        rw [← Real.sqrt_sq (le_max_right B 0)]
        exact Real.sqrt_le_sqrt hy)
    refine hsqrt_top.comp ?_
    have h1 : Tendsto (fun M : ℝ => (M * x) ^ 2 / 2) atTop atTop := by
      have heq : (fun M : ℝ => (M * x) ^ 2 / 2) = fun M => (M * M) * (x ^ 2 / 2) := by
        ext M; ring
      rw [heq]
      exact (tendsto_id.atTop_mul_atTop₀ tendsto_id).atTop_mul_const (by positivity)
    exact tendsto_const_nhds.add_atTop h1

/-! ### 2. Shared-Checkpoint One-Step Gradient Descent Construction (Proposition 2) -/

/-- Squared error `(1/2) ‖F_d(θ; x) - e_y‖₂²` for class `y ∈ Fin 3`. -/
noncomputable def sqLossExample (d : ℝ) (θ : ℝ × ℝ) (x : ℝ) (y : Fin 3) : ℝ :=
  (1 / 2 : ℝ) * normSq3 (fun i => feat1 d θ x i - if i = y then 1 else 0)

/-- Pretraining loss on `D₀ = {(1/2, class 0), (1/2, class 1)}`. -/
noncomputable def lossPretrain (d : ℝ) (θ : ℝ × ℝ) : ℝ :=
  (sqLossExample d θ (1 / 2) 0 + sqLossExample d θ (1 / 2) 1) / 2

/-- Downstream task losses on `D_A = {(1, 0)}`, `D_B = {(3/5, 2), (4/5, 2)}`, `D_C = {(1, 1)}`. -/
noncomputable def lossDA (d : ℝ) (θ : ℝ × ℝ) : ℝ := sqLossExample d θ 1 0
noncomputable def lossDB (d : ℝ) (θ : ℝ × ℝ) : ℝ :=
  (sqLossExample d θ (3 / 5) 2 + sqLossExample d θ (4 / 5) 2) / 2
noncomputable def lossDC (d : ℝ) (θ : ℝ × ℝ) : ℝ := sqLossExample d θ 1 1

/-- **Proposition 2(a)**: `P = (1, 1)` is the unique global minimizer of `lossPretrain d` over `ℝ × ℝ`. -/
theorem pretrain_unique_global_min (d : ℝ) (θ : ℝ × ℝ) :
    lossPretrain d (1, 1) ≤ lossPretrain d θ ∧
      (lossPretrain d θ = lossPretrain d (1, 1) → θ = (1, 1)) := by
  have hform : ∀ p : ℝ × ℝ, lossPretrain d p =
      (1 / 2) * (relu (p.1 * (1 / 2)) - 1 / 2) ^ 2 +
      (1 / 2) * (relu (p.2 * (1 / 2)) - 1 / 2) ^ 2 + 1 / 4 + d ^ 2 / 2 := by
    intro p
    unfold lossPretrain sqLossExample normSq3 feat1
    simp
    ring
  have hP : lossPretrain d (1, 1) = 1 / 4 + d ^ 2 / 2 := by
    rw [hform]
    rw [show (1 : ℝ) * (1 / 2) = 1 / 2 by norm_num, relu_of_nonneg (by norm_num)]
    ring
  rw [hform θ, hP]
  constructor
  · have h1 : 0 ≤ (relu (θ.1 * (1 / 2)) - 1 / 2) ^ 2 := sq_nonneg _
    have h2 : 0 ≤ (relu (θ.2 * (1 / 2)) - 1 / 2) ^ 2 := sq_nonneg _
    linarith
  · intro heq
    have h1 : 0 ≤ (relu (θ.1 * (1 / 2)) - 1 / 2) ^ 2 := sq_nonneg _
    have h2 : 0 ≤ (relu (θ.2 * (1 / 2)) - 1 / 2) ^ 2 := sq_nonneg _
    have hs1 : (relu (θ.1 * (1 / 2)) - 1 / 2) ^ 2 = 0 := by linarith
    have hs2 : (relu (θ.2 * (1 / 2)) - 1 / 2) ^ 2 = 0 := by linarith
    have hr1 : relu (θ.1 * (1 / 2)) = 1 / 2 := by nlinarith
    have hr2 : relu (θ.2 * (1 / 2)) = 1 / 2 := by nlinarith
    have hpos1 : 0 < θ.1 * (1 / 2) := by
      by_contra hle; push_neg at hle
      rw [relu_of_nonpos hle] at hr1; norm_num at hr1
    have hpos2 : 0 < θ.2 * (1 / 2) := by
      by_contra hle; push_neg at hle
      rw [relu_of_nonpos hle] at hr2; norm_num at hr2
    rw [relu_of_nonneg (le_of_lt hpos1)] at hr1
    rw [relu_of_nonneg (le_of_lt hpos2)] at hr2
    ext <;> linarith

/-- **Proposition 2(b)**: Exact partial derivatives of `lossDA`, `lossDB`, `lossDC` at `P = (1, 1)`
equal `(0, 1)`, `(1/2, 1/2)`, `(1, 0)`. -/
theorem downstream_gradients_at_P (d : ℝ) :
    (HasDerivAt (fun u => lossDA d (u, 1)) 0 1 ∧ HasDerivAt (fun v => lossDA d (1, v)) 1 1) ∧
    (HasDerivAt (fun u => lossDB d (u, 1)) (1 / 2) 1 ∧ HasDerivAt (fun v => lossDB d (1, v)) (1 / 2) 1) ∧
    (HasDerivAt (fun u => lossDC d (u, 1)) 1 1 ∧ HasDerivAt (fun v => lossDC d (1, v)) 0 1) := by
  have hEv : ∀ c : ℝ, 0 < c → ∀ᶠ t in nhds (1 : ℝ), relu (t * c) = t * c := by
    intro c hc
    filter_upwards [eventually_gt_nhds (show (0 : ℝ) < 1 by norm_num)] with t ht
    exact relu_of_nonneg (by positivity)
  have hEv1 : ∀ᶠ t in nhds (1 : ℝ), relu t = t := by
    filter_upwards [eventually_gt_nhds (show (0 : ℝ) < 1 by norm_num)] with t ht
    exact relu_of_nonneg (le_of_lt ht)
  refine ⟨⟨?_, ?_⟩, ⟨?_, ?_⟩, ⟨?_, ?_⟩⟩
  · have hpoly : HasDerivAt (fun u : ℝ => (1 / 2 : ℝ) * ((u - 1) ^ 2 + 1 ^ 2 + d ^ 2)) 0 1 := by
      have h1 := (((((hasDerivAt_id (1 : ℝ)).sub_const 1).pow 2).add_const (1 ^ 2 : ℝ)).add_const (d ^ 2)).const_mul (1 / 2 : ℝ)
      convert h1 using 1; norm_num
    refine hpoly.congr_of_eventuallyEq ?_
    filter_upwards [hEv1] with u hu
    unfold lossDA sqLossExample normSq3 feat1
    simp [hu, relu_of_nonneg zero_le_one]
  · have hpoly : HasDerivAt (fun v : ℝ => (1 / 2 : ℝ) * (0 ^ 2 + v ^ 2 + d ^ 2)) 1 1 := by
      have h1 := (((((hasDerivAt_id (1 : ℝ)).pow 2).const_add (0 ^ 2 : ℝ)).add_const (d ^ 2))).const_mul (1 / 2 : ℝ)
      convert h1 using 1; norm_num
    refine hpoly.congr_of_eventuallyEq ?_
    filter_upwards [hEv1] with v hv
    unfold lossDA sqLossExample normSq3 feat1
    simp [hv, relu_of_nonneg zero_le_one]
  · have hpoly : HasDerivAt (fun u : ℝ =>
        ((1 / 2 : ℝ) * ((u * (3 / 5)) ^ 2 + (3 / 5) ^ 2 + (d - 1) ^ 2) +
         (1 / 2 : ℝ) * ((u * (4 / 5)) ^ 2 + (4 / 5) ^ 2 + (d - 1) ^ 2)) / 2) (1 / 2) 1 := by
      have h1 := (((hasDerivAt_id (1 : ℝ)).mul_const (3 / 5 : ℝ)).pow 2)
      have h2 := (((hasDerivAt_id (1 : ℝ)).mul_const (4 / 5 : ℝ)).pow 2)
      have h3 := ((((h1.add_const ((3 / 5 : ℝ) ^ 2)).add_const ((d - 1) ^ 2)).const_mul (1 / 2 : ℝ)).add
        (((h2.add_const ((4 / 5 : ℝ) ^ 2)).add_const ((d - 1) ^ 2)).const_mul (1 / 2 : ℝ))).div_const 2
      convert h3 using 1; norm_num
    refine hpoly.congr_of_eventuallyEq ?_
    filter_upwards [hEv (3 / 5) (by norm_num), hEv (4 / 5) (by norm_num)] with u hu3 hu4
    unfold lossDB sqLossExample normSq3 feat1
    simp [hu3, hu4, relu_of_nonneg (show (0 : ℝ) ≤ 3 / 5 by norm_num),
      relu_of_nonneg (show (0 : ℝ) ≤ 4 / 5 by norm_num)]
  · have hpoly : HasDerivAt (fun v : ℝ =>
        ((1 / 2 : ℝ) * ((3 / 5) ^ 2 + (v * (3 / 5)) ^ 2 + (d - 1) ^ 2) +
         (1 / 2 : ℝ) * ((4 / 5) ^ 2 + (v * (4 / 5)) ^ 2 + (d - 1) ^ 2)) / 2) (1 / 2) 1 := by
      have h1 := (((hasDerivAt_id (1 : ℝ)).mul_const (3 / 5 : ℝ)).pow 2)
      have h2 := (((hasDerivAt_id (1 : ℝ)).mul_const (4 / 5 : ℝ)).pow 2)
      have h3 := ((((h1.const_add ((3 / 5 : ℝ) ^ 2)).add_const ((d - 1) ^ 2)).const_mul (1 / 2 : ℝ)).add
        (((h2.const_add ((4 / 5 : ℝ) ^ 2)).add_const ((d - 1) ^ 2)).const_mul (1 / 2 : ℝ))).div_const 2
      convert h3 using 1; norm_num
    refine hpoly.congr_of_eventuallyEq ?_
    filter_upwards [hEv (3 / 5) (by norm_num), hEv (4 / 5) (by norm_num)] with v hv3 hv4
    unfold lossDB sqLossExample normSq3 feat1
    simp [hv3, hv4, relu_of_nonneg (show (0 : ℝ) ≤ 3 / 5 by norm_num),
      relu_of_nonneg (show (0 : ℝ) ≤ 4 / 5 by norm_num)]
  · have hpoly : HasDerivAt (fun u : ℝ => (1 / 2 : ℝ) * (u ^ 2 + 0 ^ 2 + d ^ 2)) 1 1 := by
      have h1 := ((((hasDerivAt_id (1 : ℝ)).pow 2).add_const (0 ^ 2 : ℝ)).add_const (d ^ 2)).const_mul (1 / 2 : ℝ)
      convert h1 using 1; norm_num
    refine hpoly.congr_of_eventuallyEq ?_
    filter_upwards [hEv1] with u hu
    unfold lossDC sqLossExample normSq3 feat1
    simp [hu, relu_of_nonneg zero_le_one]
  · have hpoly : HasDerivAt (fun v : ℝ => (1 / 2 : ℝ) * (1 ^ 2 + (v - 1) ^ 2 + d ^ 2)) 0 1 := by
      have h1 := (((((hasDerivAt_id (1 : ℝ)).sub_const 1).pow 2).const_add (1 ^ 2 : ℝ)).add_const (d ^ 2)).const_mul (1 / 2 : ℝ)
      convert h1 using 1; norm_num
    refine hpoly.congr_of_eventuallyEq ?_
    filter_upwards [hEv1] with v hv
    unfold lossDC sqLossExample normSq3 feat1
    simp [hv, relu_of_nonneg zero_le_one]

/-- **Proposition 2(c)**: One GD step with `η = 2` from `P = (1, 1)` produces
`A = (1, -1)`, `B = (0, 0)`, `C = (-1, 1)`, each globally minimizing its task loss. -/
theorem gd_step_and_task_opt (d : ℝ) (θ : ℝ × ℝ) :
    ((1 : ℝ) - 2 * 0, (1 : ℝ) - 2 * 1) = ((1 : ℝ), (-1 : ℝ)) ∧
    ((1 : ℝ) - 2 * (1 / 2), (1 : ℝ) - 2 * (1 / 2)) = ((0 : ℝ), (0 : ℝ)) ∧
    ((1 : ℝ) - 2 * 1, (1 : ℝ) - 2 * 0) = ((-1 : ℝ), (1 : ℝ)) ∧
    lossDA d (1, -1) ≤ lossDA d θ ∧
    lossDB d (0, 0) ≤ lossDB d θ ∧
    lossDC d (-1, 1) ≤ lossDC d θ := by
  refine ⟨by norm_num, by norm_num, by norm_num, ?_, ?_, ?_⟩
  · unfold lossDA sqLossExample normSq3 feat1
    simp [relu_of_nonneg zero_le_one, relu_of_nonpos (show (-1 : ℝ) ≤ 0 by norm_num)]
    nlinarith [sq_nonneg (relu θ.1 - 1), sq_nonneg (relu θ.2)]
  · unfold lossDB sqLossExample normSq3 feat1
    simp [relu_of_nonneg le_rfl]
    nlinarith [sq_nonneg (relu (θ.1 * (3 / 5))), sq_nonneg (relu (θ.2 * (3 / 5))),
      sq_nonneg (relu (θ.1 * (4 / 5))), sq_nonneg (relu (θ.2 * (4 / 5)))]
  · unfold lossDC sqLossExample normSq3 feat1
    simp [relu_of_nonneg zero_le_one, relu_of_nonpos (show (-1 : ℝ) ≤ 0 by norm_num)]
    nlinarith [sq_nonneg (relu θ.1), sq_nonneg (relu θ.2 - 1)]

/-! ### 3. Exact One-Layer ReLU Transitivity Criterion (Theorem 4.1) -/

/-- Exact chord gap `((1-t)σ(a) + tσ(c)) - σ((1-t)a + tc)`. -/
def reluGap (a c t : ℝ) : ℝ :=
  (1 - t) * relu a + t * relu c - relu ((1 - t) * a + t * c)

/-- **Theorem 4.1(a)**: When `a * c < 0`, the ReLU chord gap attains its exact maximum
`|a * c| / (|a| + |c|) > 0` at `t* = a / (a - c) ∈ (0, 1)`. -/
theorem relu_chord_max_gap {a c : ℝ} (hac : a * c < 0) :
    let tStar := a / (a - c)
    0 < tStar ∧ tStar < 1 ∧
    reluGap a c tStar = |a * c| / (|a| + |c|) ∧
    0 < reluGap a c tStar ∧
    ∀ t ∈ Set.Icc (0 : ℝ) 1, reluGap a c t ≤ |a * c| / (|a| + |c|) := by
  intro tStar
  have hcases : (0 < a ∧ c < 0) ∨ (a < 0 ∧ 0 < c) := by
    rcases lt_trichotomy a 0 with ha | rfl | ha
    · right; exact ⟨ha, by nlinarith⟩
    · simp at hac
    · left; exact ⟨ha, by nlinarith⟩
  rcases hcases with ⟨ha, hc⟩ | ⟨ha, hc⟩
  · have hsub : 0 < a - c := by linarith
    have ht0 : 0 < tStar := div_pos ha hsub
    have ht1 : tStar < 1 := (div_lt_one hsub).mpr (by linarith)
    have hval : reluGap a c tStar = |a * c| / (|a| + |c|) := by
      unfold reluGap
      have hlin : (1 - tStar) * a + tStar * c = 0 := by
        dsimp [tStar]; field_simp [ne_of_gt hsub]; ring
      rw [hlin, relu_of_nonneg le_rfl, relu_of_nonneg (le_of_lt ha), relu_of_nonpos (le_of_lt hc)]
      rw [abs_of_neg hac, abs_of_pos ha, abs_of_neg hc, show a + -c = a - c by ring]
      dsimp [tStar]; field_simp [ne_of_gt hsub]; ring
    have hpos : 0 < reluGap a c tStar := by
      rw [hval]; exact div_pos (abs_pos.mpr (ne_of_lt hac)) (by rw [abs_of_pos ha, abs_of_neg hc]; linarith)
    refine ⟨ht0, ht1, hval, hpos, fun t ⟨h0, h1⟩ => ?_⟩
    unfold reluGap
    rw [relu_of_nonneg (le_of_lt ha), relu_of_nonpos (le_of_lt hc),
        abs_of_neg hac, abs_of_pos ha, abs_of_neg hc]
    by_cases hcomb : 0 ≤ (1 - t) * a + t * c
    · rw [relu_of_nonneg hcomb, le_div_iff₀ (by linarith)]
      nlinarith
    · rw [relu_of_nonpos (le_of_not_ge hcomb), le_div_iff₀ (by linarith)]
      nlinarith
  · have hsub : a - c < 0 := by linarith
    have ht0 : 0 < tStar := div_pos_of_neg_of_neg ha hsub
    have ht1 : tStar < 1 := by
      dsimp [tStar]
      rw [div_lt_iff_of_neg hsub]
      linarith
    have hval : reluGap a c tStar = |a * c| / (|a| + |c|) := by
      unfold reluGap
      have hlin : (1 - tStar) * a + tStar * c = 0 := by
        dsimp [tStar]; field_simp [ne_of_lt hsub]; ring
      rw [hlin, relu_of_nonneg le_rfl, relu_of_nonpos (le_of_lt ha), relu_of_nonneg (le_of_lt hc)]
      rw [abs_of_neg hac, abs_of_neg ha, abs_of_pos hc, show -a + c = -(a - c) by ring]
      dsimp [tStar]; field_simp [ne_of_lt hsub]; ring
    have hpos : 0 < reluGap a c tStar := by
      rw [hval]; exact div_pos (abs_pos.mpr (ne_of_lt hac)) (by rw [abs_of_neg ha, abs_of_pos hc]; linarith)
    refine ⟨ht0, ht1, hval, hpos, fun t ⟨h0, h1⟩ => ?_⟩
    unfold reluGap
    rw [relu_of_nonpos (le_of_lt ha), relu_of_nonneg (le_of_lt hc),
        abs_of_neg hac, abs_of_neg ha, abs_of_pos hc]
    by_cases hcomb : 0 ≤ (1 - t) * a + t * c
    · rw [relu_of_nonneg hcomb, le_div_iff₀ (by linarith)]
      nlinarith
    · rw [relu_of_nonpos (le_of_not_ge hcomb), le_div_iff₀ (by linarith)]
      nlinarith

/-- **Theorem 4.1(b)**: Exact ReLU chord identity holds for all `t ∈ [0, 1]` iff `0 ≤ a * c`. -/
theorem relu_chord_iff (a c : ℝ) :
    (∀ t ∈ Set.Icc (0 : ℝ) 1, relu ((1 - t) * a + t * c) = (1 - t) * relu a + t * relu c) ↔
      0 ≤ a * c := by
  constructor
  · intro h
    by_contra hlt
    push_neg at hlt
    obtain ⟨ht0, ht1, _, hpos, _⟩ := relu_chord_max_gap hlt
    have heq := h (a / (a - c)) ⟨le_of_lt ht0, le_of_lt ht1⟩
    unfold reluGap at hpos
    linarith
  · intro hac t ⟨ht0, ht1⟩
    have h1t : 0 ≤ 1 - t := sub_nonneg.mpr ht1
    rcases le_or_gt 0 a with ha | ha
    · rcases eq_or_lt_of_le ha with rfl | ha'
      · simp only [mul_zero, zero_add, relu_of_nonneg le_rfl]
        rcases le_or_gt 0 c with hc | hc
        · rw [relu_of_nonneg (by positivity), relu_of_nonneg hc]
        · rw [relu_of_nonpos (by nlinarith), relu_of_nonpos (le_of_lt hc), mul_zero]
      · have hc : 0 ≤ c := by nlinarith
        rw [relu_of_nonneg (by nlinarith), relu_of_nonneg ha, relu_of_nonneg hc]
    · have hc : c ≤ 0 := by nlinarith
      rw [relu_of_nonpos (by nlinarith), relu_of_nonpos (le_of_lt ha), relu_of_nonpos hc]
      ring

/-- **Theorem 4.1(c) (Exact one-layer ReLU transitivity criterion)**: Given exact CTL on `(a, b)`
and `(b, c)`, exact CTL fails on `(a, c)` if and only if `b = 0 ∧ a * c < 0`. -/
theorem one_layer_transitivity_failure_iff {a b c : ℝ}
    (hab : ∀ t ∈ Set.Icc (0 : ℝ) 1, relu ((1 - t) * a + t * b) = (1 - t) * relu a + t * relu b)
    (hbc : ∀ t ∈ Set.Icc (0 : ℝ) 1, relu ((1 - t) * b + t * c) = (1 - t) * relu b + t * relu c) :
    (¬ ∀ t ∈ Set.Icc (0 : ℝ) 1, relu ((1 - t) * a + t * c) = (1 - t) * relu a + t * relu c) ↔
      (b = 0 ∧ a * c < 0) := by
  rw [relu_chord_iff] at hab hbc ⊢
  push_neg
  constructor
  · intro hac
    refine ⟨?_, hac⟩
    by_contra hb
    have hbb : 0 < b ^ 2 := sq_pos_of_ne_zero hb
    have hprod : 0 ≤ (a * b) * (b * c) := mul_nonneg hab hbc
    have heq : (a * b) * (b * c) = (a * c) * b ^ 2 := by ring
    rw [heq] at hprod
    nlinarith
  · exact And.right

/-! ### 4. Strictly Active Two-Layer Counterexample (Proposition 3) -/

/-- Two-layer scalar ReLU network: layer 1 `h((u,v); x) = σ(ux)`, layer 2 `y((u,v); x) = σ(v·h)`. -/
def layer1 (θ : ℝ × ℝ) (x : ℝ) : ℝ := relu (θ.1 * x)
def layer2 (θ : ℝ × ℝ) (x : ℝ) : ℝ := relu (θ.2 * layer1 θ x)

/-- **Proposition 3(a)**: Every convex combination of `A = (1+r, 1)`, `B = (1, 1)`, `C = (1, 1+s)`
(`r, s > 0`) keeps both neurons strictly active for every `x > 0`. -/
theorem two_layer_strictly_active_conv {r s x α β γ : ℝ}
    (hr : 0 < r) (hs : 0 < s) (hx : 0 < x)
    (hα : 0 ≤ α) (_hβ : 0 ≤ β) (hγ : 0 ≤ γ) (hsum : α + β + γ = 1) :
    let u := α * (1 + r) + β * 1 + γ * 1
    let v := α * 1 + β * 1 + γ * (1 + s)
    0 < u * x ∧ 0 < v * relu (u * x) := by
  intro u v
  have hu : 1 ≤ u := by dsimp [u]; nlinarith
  have hv : 1 ≤ v := by dsimp [v]; nlinarith
  have hux : 0 < u * x := by positivity
  rw [relu_of_nonneg (le_of_lt hux)]
  exact ⟨hux, by positivity⟩

/-- **Proposition 3(b)**: Exact layerwise CTL on `(A, B)` and `(B, C)`, exact layer-1 CTL on
`(A, C)`, and exact positive layer-2 gap `t(1 - t)rsx` on `(A, C)` (with maximum `rsx / 4` at `t = 1/2`). -/
theorem two_layer_strictly_active_counterexample {r s x t : ℝ}
    (hr : 0 < r) (hs : 0 < s) (hx : 0 < x) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    let A : ℝ × ℝ := (1 + r, 1)
    let B : ℝ × ℝ := (1, 1)
    let C : ℝ × ℝ := (1, 1 + s)
    layer1 (interp t A B) x = (1 - t) * layer1 A x + t * layer1 B x ∧
    layer2 (interp t A B) x = (1 - t) * layer2 A x + t * layer2 B x ∧
    layer1 (interp t B C) x = (1 - t) * layer1 B x + t * layer1 C x ∧
    layer2 (interp t B C) x = (1 - t) * layer2 B x + t * layer2 C x ∧
    layer1 (interp t A C) x = (1 - t) * layer1 A x + t * layer1 C x ∧
    layer2 (interp t A C) x - ((1 - t) * layer2 A x + t * layer2 C x) =
      t * (1 - t) * r * s * x := by
  intro A B C
  have h1t : 0 ≤ 1 - t := sub_nonneg.mpr ht1
  have hL : ∀ p : ℝ × ℝ, 0 < p.1 → 0 < p.2 →
      layer1 p x = p.1 * x ∧ layer2 p x = p.1 * p.2 * x := by
    intro p hp1 hp2
    have h1 : layer1 p x = p.1 * x := relu_of_nonneg (by positivity)
    refine ⟨h1, ?_⟩
    unfold layer2
    rw [h1, relu_of_nonneg (by positivity)]
    ring
  have hA := hL A (by linarith) one_pos
  have hB := hL B one_pos one_pos
  have hC := hL C one_pos (by linarith)
  have hAB := hL (interp t A B) (by simp [interp, A, B]; nlinarith) (by simp [interp, A, B])
  have hBC := hL (interp t B C) (by simp [interp, B, C]) (by simp [interp, B, C]; nlinarith)
  have hAC := hL (interp t A C) (by simp [interp, A, C]; nlinarith) (by simp [interp, A, C]; nlinarith)
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hAB.1, hA.1, hB.1]; simp [interp, A, B]; ring
  · rw [hAB.2, hA.2, hB.2]; simp [interp, A, B]; ring
  · rw [hBC.1, hB.1, hC.1]; simp [interp, B, C]; ring
  · rw [hBC.2, hB.2, hC.2]; simp [interp, B, C]; ring
  · rw [hAC.1, hA.1, hC.1]; simp [interp, A, C]; ring
  · rw [hAC.2, hA.2, hC.2]; simp [interp, A, C]; ring

/-- On the strictly positive cone `(0, ∞)²`, exact layer-2 CTL holds on `(p, q)` iff
`(q.1 - p.1) * (q.2 - p.2) = 0`. -/
theorem two_layer_pos_ctl_iff {p q : ℝ × ℝ} {x : ℝ}
    (hp1 : 0 < p.1) (hp2 : 0 < p.2) (hq1 : 0 < q.1) (hq2 : 0 < q.2) (hx : 0 < x) :
    (∀ t ∈ Set.Icc (0 : ℝ) 1,
      layer2 (interp t p q) x = (1 - t) * layer2 p x + t * layer2 q x) ↔
      (q.1 - p.1) * (q.2 - p.2) = 0 := by
  have hval : ∀ t ∈ Set.Icc (0 : ℝ) 1,
      layer2 (interp t p q) x - ((1 - t) * layer2 p x + t * layer2 q x) =
        -(t * (1 - t) * (q.1 - p.1) * (q.2 - p.2) * x) := by
    intro t ⟨ht0, ht1⟩
    have h1t : 0 ≤ 1 - t := sub_nonneg.mpr ht1
    have hu : 0 ≤ (1 - t) * p.1 + t * q.1 := by nlinarith
    have hv : 0 ≤ (1 - t) * p.2 + t * q.2 := by nlinarith
    unfold layer2 layer1 interp
    rw [relu_of_nonneg (mul_nonneg hu (le_of_lt hx)),
        relu_of_nonneg (mul_nonneg hv (mul_nonneg hu (le_of_lt hx))),
        relu_of_nonneg (mul_nonneg (le_of_lt hp1) (le_of_lt hx)),
        relu_of_nonneg (mul_nonneg (le_of_lt hp2) (mul_nonneg (le_of_lt hp1) (le_of_lt hx))),
        relu_of_nonneg (mul_nonneg (le_of_lt hq1) (le_of_lt hx)),
        relu_of_nonneg (mul_nonneg (le_of_lt hq2) (mul_nonneg (le_of_lt hq1) (le_of_lt hx)))]
    ring
  constructor
  · intro h
    have hmid := h (1 / 2) ⟨by norm_num, by norm_num⟩
    have hv := hval (1 / 2) ⟨by norm_num, by norm_num⟩
    rw [hmid, sub_self] at hv
    have hxne : x ≠ 0 := ne_of_gt hx
    nlinarith
  · intro hprod t ht
    have hv := hval t ht
    have hzero : -(t * (1 - t) * (q.1 - p.1) * (q.2 - p.2) * x) = 0 := by
      calc -(t * (1 - t) * (q.1 - p.1) * (q.2 - p.2) * x)
          = -(t * (1 - t) * ((q.1 - p.1) * (q.2 - p.2)) * x) := by ring
        _ = 0 := by rw [hprod]; ring
    linarith

end CrossTaskLinearity
