# Refutation of Conjecture 6.2 (Static and Dynamical Head-Subspace Incidence)

**Target paper:** Jingchuan Xiao, Xinyi Sui, and Cihan Ruan, *Flag Varieties: A Geometric Framework for Deep Network Alignment*, [arXiv:2605.09861](https://arxiv.org/abs/2605.09861), Section 6, Conjecture 6.2 (`conj:static_head_incidence`).  
**Companion files:** [`solutions/flag_variety_alignment/explore.py`](explore.py), [`solutions/flag_variety_alignment/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture 6.2 (`conj:static_head_incidence`) — Static claim: FALSE.** A standard two-head softmax attention layer ($d = h = 2$, $r = d_v = 1$, 16 trainable parameters, no biases) trained by squared error with positive $L_2$ weight decay ($\lambda = 1/100$) has two nonzero, mutually orthogonal query heads at **every global minimizer** of an explicit orthogonal-routing task, giving overlap $R = 0$ against the Haar baseline $r/d = 1/2$.
- **Open attracting basin:** The full 16-variable Hessian at the global minimizer is strictly positive definite ($\nabla^2 F \succ 0$), so gradient flow converges to the $R = 0$ minimizer from an open set of initial weights.
- **Target dependence vs. input covariance:** On the exact same full-rank anisotropic inputs ($w = (2, 1)$), changing only the targets forces $R = 1$ at every global minimizer. Thus the input covariance alone cannot determine the sign of $R - r/d$.
- **Conjecture 6.2 — Dynamical claim: FALSE.** Perturbing only one query vector of the isotropic orthogonal minimizer yields an explicit initial condition with $\dot{R}(0) < 0$; taking tilt $t = 1.1c$ starts above the Haar baseline ($R(0) \approx 0.5475 > 1/2$) with $\dot{R}(0) \approx -0.2967$ and decreases to $R \to 0$ along full jointly trained gradient flow.
- **Scope note:** These counterexamples refute both universal statements of Conjecture 6.2. Whether excess overlap $R > r/d$ is typical under a specified distribution of large language model training runs remains an empirical question.

---

## 1. Conjecture, Haar Baseline, and Projector Criterion

Suppress the layer index $\ell$ and write $r := d_k$. For $h$ attention heads with full-column-rank query matrices $W_i \in \mathbb{R}^{d \times r}$, let $U_i \in \mathbb{R}^{d \times r}$ have orthonormal columns spanning $\operatorname{col}(W_i) \in \operatorname{Gr}(r, d)$ and let $P_i := U_i U_i^\top$ be the orthogonal projector onto $\operatorname{col}(W_i)$. The paper's Frobenius overlap observable is
$$
R := \frac{1}{h(h-1)r}\sum_{i \neq j} \|P_i P_j\|_F^2
= \frac{1}{h(h-1)r}\sum_{i \neq j} \|U_i^\top U_j\|_F^2
= \frac{1}{h(h-1)r}\sum_{i \neq j} \operatorname{tr}(P_i P_j) \in [0, 1].
$$

**Conjecture 6.2 (`conj:static_head_incidence`, arXiv:2605.09861, Section 6).** *For attention-based networks trained with weight decay, terminal query-head subspaces exhibit higher mean pairwise Frobenius overlap $R^{(\ell)}$ than Haar-random subspaces of the same dimensions. A stronger dynamical version—predicting that $R^{(\ell)}$ increases monotonically during training—requires training-time checkpoints and is left for future work.*

*(Note: The source specifies no common full-attention training loss, initialization law, or task distribution for Conjecture 6.2; its toy Proposition 6.1 holds the sum $W_Q^{(1)} + W_Q^{(2)}$ fixed, and its GPT-2 small / Llama measurements examine terminal checkpoints.)*

**Proposition 1 (Haar mean and exact projector identity).**
1. If $P_1, \dots, P_h$ are independent Haar-random rank-$r$ orthogonal projectors in $\mathbb{R}^d$, then $\mathbb{E}[P_i] = (r/d)I_d$ and $\mathbb{E}[R] = r/d$.
2. For any full-rank tuple $(W_1, \dots, W_h)$ with aggregate projector $S := \sum_{i=1}^h P_i$,
   $$
   h(h-1)r\Bigl(R - \frac{r}{d}\Bigr) = \Bigl\|S - \frac{hr}{d}I_d\Bigr\|_F^2 - hr\Bigl(1 - \frac{r}{d}\Bigr).
   $$

*Proof.* By orthogonal invariance, $\mathbb{E}[P_i]$ commutes with $O(d)$ and is therefore scalar; taking traces yields $\mathbb{E}[P_i] = (r/d)I_d$. Independence gives $\mathbb{E}[\operatorname{tr}(P_i P_j)] = \operatorname{tr}((r/d)^2 I_d) = r^2/d$ for $i \neq j$, hence $\mathbb{E}[R] = r/d$. For part (2), since $P_i^2 = P_i$ and $\operatorname{tr}(S) = hr$,
$$
\Bigl\|S - \frac{hr}{d}I_d\Bigr\|_F^2 = \operatorname{tr}(S^2) - \frac{h^2 r^2}{d}
= hr + h(h-1)r R - \frac{h^2 r^2}{d}
= hr\Bigl(1 - \frac{r}{d}\Bigr) + h(h-1)r\Bigl(R - \frac{r}{d}\Bigr). \quad \square
$$

**Why weight decay does not automatically force alignment.** For a differentiable data loss $L$ regularized by $\frac{\lambda}{2}\sum_i \|W_i\|_F^2$ (plus penalties on other parameters), stationarity gives $\nabla_{W_i} L = -\lambda W_i$ separately for each head, so at a full-rank stationary point the criterion in Proposition 1 applies equally to the individual data-gradient column spaces. Along Euclidean gradient flow $\dot{W}_i = -\nabla_{W_i} L - \lambda W_i$, differentiating $P_i = W_i W_i^\dagger$ at full column rank yields
$$
\dot{P}_i = -(I_d - P_i)(\nabla_{W_i} L) W_i^\dagger - \bigl[(I_d - P_i)(\nabla_{W_i} L) W_i^\dagger\bigr]^\top.
$$
Because $(I_d - P_i) W_i = 0$, the direct radial decay term $-\lambda W_i$ drops out identically from $\dot{P}_i$. In Proposition 6.1 of the paper, fixing $W_Q^{(1)} + W_Q^{(2)}$ lets the parallelogram identity force head equality; in full softmax attention, redistributing query matrices while fixing their sum generally changes the output function and increases data loss.

---

## 2. Two-Head Softmax Layer and Two Tasks

Set $d = h = 2$ and $r = d_v = 1$, so the Haar baseline is $r/d = 1/2$. Each head $i \in \{1, 2\}$ has freely trainable vectors $q_i, k_i, v_i, o_i \in \mathbb{R}^2$ (where $o_i^\top$ is row $i$ of the output projection), giving 16 real parameters and no biases. A query token $x \in \mathbb{R}^2$ attends to two memory tokens $(m, -m)$ (equivalently a masked self-attention row on $(m, -m, x)$). Head $i$ has logits $(s_i, -s_i)$ with $s_i = (q_i^\top x)(k_i^\top m)$ and softmax weight difference $\tanh(s_i)$, so the exact layer output is
$$
f(x, m) = \sum_{i=1}^2 o_i (v_i^\top m) \tanh\bigl((q_i^\top x)(k_i^\top m)\bigr).
$$
With standard basis vectors $e_1, e_2 \in \mathbb{R}^2$, consider the squared-error objective with $L_2$ weight decay $\lambda = 1/100$:
$$
F(\theta) = \frac{1}{2}\sum_{l,j=1}^2 w_l \|f(e_l, e_j) - y_{lj}\|^2
+ \frac{\lambda}{2}\sum_{i=1}^2 \bigl(\|q_i\|^2 + \|k_i\|^2 + \|v_i\|^2 + \|o_i\|^2\bigr).
$$
Because $f$ is odd in $x$, replacing each $(e_l, e_j, y_{lj})$ by $\pm(e_l, e_j, y_{lj})$ with half weight leaves $F$ unchanged and yields centered query covariance $\operatorname{diag}(w_1, w_2)/(w_1 + w_2)$: isotropic for $w = (1, 1)$ and full-rank anisotropic for $w = (2, 1)$. The objective supervises only the mixed output $f$ (no head is assigned to a task) and is coercive since $\lambda > 0$, so global minimizers exist.

| Task | Targets $y_{lj}$ | Query weights $w$ | $R$ at every global minimizer |
|---|---|---|---:|
| Orthogonal tasks | $\mathbf{1}_{l=j} e_j$ | $(1,1)$ or $(2,1)$ | $0$ |
| Shared query direction | $\mathbf{1}_{l=1} e_j$ | $(2,1)$ | $1$ |

---

## 3. Orthogonal Tasks: $R = 0 < 1/2$ and Open Attracting Basin

**Theorem 2 (Global optimality of orthogonal heads and strict local convexity).** *For $y_{lj} = \mathbf{1}_{l=j} e_j$, $\lambda = 1/100$, and either $w = (1, 1)$ or $w = (2, 1)$:*
1. *Every global minimizer of $F$ has two nonzero, mutually orthogonal query vectors $q_1 \perp q_2$, hence $R = 0 < 1/2$.*
2. *The positive-coordinate global minimizer has strictly positive definite $16 \times 16$ Hessian $\nabla^2 F \succ 0$, and therefore attracts gradient flow from an open neighborhood of initial weights.*

*Proof.* Write $q_{ij} := q_i^\top e_j$ (and similarly $k_{ij}, v_{ij}, o_{ij}$). For each coordinate $j \in \{1, 2\}$, retain only the scalar prediction $z_j := e_j^\top f(e_j, e_j)$ against target $1$, and set
$$
A_j := \sum_{i=1}^2 |o_{ij} v_{ij}|, \qquad B_j := \sum_{i=1}^2 |q_{ij} k_{ij}|.
$$
By AM–GM, the coordinate-$j$ weight-decay penalty is at least $\lambda(A_j + B_j)$. By monotonicity of $\tanh$ on $[0, \infty)$,
$$
|z_j| \le \sum_{i=1}^2 |o_{ij} v_{ij}| \tanh |q_{ij} k_{ij}| \le A_j \tanh B_j.
$$
Setting $b = B_j$ and $a = |z_j| / \tanh b \le A_j$ when $B_j > 0$ (and $a = b = 0$ otherwise), and using $(1 - z_j)^2 \ge (1 - |z_j|)^2$, we obtain
$$
F(\theta) \ge m(w_1) + m(w_2), \qquad
m(w) := \min_{a,b \ge 0}\Bigl\{\frac{w}{2}(1 - a\tanh b)^2 + \lambda(a + b)\Bigr\}.
$$
Each scalar minimum is strictly active: the trial $b = 1, a = 1/\tanh 1$ gives cost $\lambda(1 + 1/\tanh 1) < 0.0232$, whereas $ab = 0$ costs at least $w/2 \ge 0.5$. Let $(a_j, b_j)$ minimize the scalar problem for $w_j$. Then the assignment
$$
q_j = k_j = \sqrt{b_j}\, e_j, \qquad v_j = o_j = \sqrt{a_j}\, e_j \qquad (j = 1, 2)
$$
attains $m(w_1) + m(w_2)$ exactly, with all discarded components and zero-target examples having zero residual.

*Equality forces specialization.* At any global minimizer, each coordinate $j$ attains its active lower bound, so $A_j, B_j, |z_j| > 0$. By strict increase of $\tanh$ on $[0, \infty)$, every head $i$ with $|o_{ij} v_{ij}| > 0$ must satisfy $|q_{ij} k_{ij}| = B_j$. Hence a single head carries all nonzero weight at coordinate $j$, and equality in AM–GM forces all four entries $(q_{i'j}, k_{i'j}, v_{i'j}, o_{i'j})$ of the other head $i'$ to vanish. If the same head $i$ carried both $j = 1$ and $j = 2$, then $f(e_1, e_1)$ would be a nonzero multiple of $o_i$, whose second coordinate $o_{i2} \neq 0$ would produce a strictly positive discarded error on $e_2^\top f(e_1, e_1) - 0$. Therefore distinct heads carry the two coordinates, so $q_1, q_2$ are nonzero multiples of $e_1, e_2$ (up to head swap), giving $R = 0$.

*Strict positive definiteness of the 16-variable Hessian.* Put $s := \tanh b \in (0, 1)$ and $\delta := \lambda/w \in \{0.005, 0.01\}$. Minimizing the scalar objective over $a \ge 0$ gives $a(b) = (s - \delta)/s^2$ on $s > \delta$ (and $a = 0$ otherwise), with reduced derivative
$$
\frac{d}{db}\Bigl[\frac{w}{2}(1 - a(b)s)^2 + \lambda(a(b) + b)\Bigr]
= \lambda\,\frac{2s^3 - \delta s^2 - s + \delta}{s^3}.
$$
Let $p(s) := 2s^3 - \delta s^2 - s + \delta$. For $\delta \in (0, 1/4)$, $p(-\infty) = -\infty$, $p(\delta) = \delta^3 > 0$, $p(1/2) = \frac{3}{4}\delta - \frac{1}{4} < 0$, and $p(1) = 1 > 0$. Thus $p$ has three simple roots in $(-\infty, 0)$, $(\delta, 1/2)$, and $(1/2, 1)$. Only the root in $(1/2, 1)$ is a local minimum; there $p'(s) > 0$, so the reduced second derivative in $b$ is strictly positive. Together with the $a$-curvature $w s^2 > 0$, the 2-variable Hessian in $(a, b)$ is positive definite.

For the four active entries of head $j$, reparameterize locally by $q_{jj} = \sqrt{b_j} e^u$, $k_{jj} = \sqrt{b_j} e^{-u}$, $v_{jj} = \sqrt{a_j} e^\eta$, $o_{jj} = \sqrt{a_j} e^{-\eta}$. The data loss depends only on $(a_j, b_j)$, while the weight-decay term becomes $\lambda[b_j \cosh(2u) + a_j \cosh(2\eta)]$, contributing strictly positive curvature in $(u, \eta)$. For each retained scalar error $e_j^\top f(e_j, e_j) - 1$, the inactive head enters as $o_{i'j} v_{i'j} \tanh(q_{i'j} k_{i'j}) = O(\|\text{inactive}\|^4)$, so its second derivative at $0$ vanishes; those eight inactive parameters receive strictly positive quadratic curvature $\lambda I_8$ from weight decay. Every discarded residual vanishes at the minimizer, so its squared error adds a positive semidefinite Hessian. Thus $\nabla^2 F \succ 0$ on $\mathbb{R}^{16}$. By continuity, $F$ is strongly convex in a ball around the minimizer, where gradient flow contracts distance and converges from an open set of initializations. $\square$

---

## 4. Shared Query Direction ($R = 1$) and Negative Dynamical Drift ($\dot{R} < 0$)

**Proposition 3 (Shared query direction on identical anisotropic inputs).** *Keep $w = (2, 1)$, $\lambda = 1/100$, and the same tokens and loss, but set $y_{lj} = \mathbf{1}_{l=1} e_j$. Then every global minimizer of $F$ has both query vectors nonzero and collinear with $e_1$, hence $R = 1 > 1/2$.*

*Proof.* At any global minimizer, $q_{12} = q_{22} = 0$: zeroing both entries preserves $f(e_1, e_j)$, makes $f(e_2, e_j) = 0 = y_{2j}$ exact, and strictly lowers weight decay if either entry were nonzero. Moreover, both $q_1$ and $q_2$ must be nonzero: if at most one head has nonzero query, then $f(e_1, e_1)$ and $f(e_1, e_2)$ lie in a single one-dimensional line with projector $P$, forcing $\sum_{j=1}^2 \|(I_2 - P)e_j\|^2 = 1$ and $F \ge w_1/2 = 1$. By contrast, $q_j = e_1$, $k_j = e_j$, $v_j = o_j = e_j/\sqrt{\tanh 1}$ ($j = 1, 2$) fits all four targets with zero data error and total cost $2\lambda(1 + 1/\tanh 1) < 0.0463 < 1$. Hence both query lines equal $\operatorname{span}(e_1)$ and $R = 1$. $\square$

**Proposition 4 (Strict decrease of overlap under gradient flow).** *On the isotropic orthogonal task ($w = (1, 1)$), initialize at the positive global minimizer of Theorem 2 except with $q_1$ tilted to $q_1(0) = c e_1 + t e_2$, where $c = \sqrt{b_1} > 0$ and $t > 0$. Along the full 16-parameter Euclidean gradient flow $\dot{\theta} = -\nabla F(\theta)$, at $s = 0$ one has $\dot{q}_{11}(0) = 0$, $\dot{q}_2(0) = 0$, and*
$$
\dot{t}(0) = -\lambda t - w_2 a_1^2 c \tanh(ct)\operatorname{sech}^2(ct) < 0,
\qquad
R(0) = \frac{t^2}{c^2 + t^2},
\qquad
\dot{R}(0) = \frac{2 t c^2}{(c^2 + t^2)^2}\,\dot{t}(0) < 0.
$$

*Proof.* Only the sample $(e_2, e_1)$ changes its prediction, from $0$ to $a_1 \tanh(ct) e_1$, while its target is $0$. The data gradients on all other samples still cancel weight decay in $q_{11}$ and $q_2$, and the new residual $a_1 \tanh(ct) e_1$ has derivative with respect to $q_{11}$ and $q_2$ equal to zero (since $e_2^\top e_1 = 0$ and $k_{21} = 0$). Differentiating $\frac{w_2}{2} a_1^2 \tanh^2(ct) + \frac{\lambda}{2} t^2$ with respect to $t$ gives the formula for $\dot{t}(0) < 0$. Since $q_2(0) = c e_2$, $R = q_{12}^2/(q_{11}^2 + q_{12}^2)$, and differentiating at $s = 0$ yields $\dot{R}(0) < 0$. $\square$

By continuity, $R$ decreases strictly on a positive time interval and for all sufficiently close initializations. Choosing $t = 1.1c$ places the starting point strictly above the Haar baseline ($R(0) = 1.21/2.21 \approx 0.5475 > 1/2$).

---

## 5. Reproducible Verification

Run from the repository root:

```sh
.venv/bin/python -B solutions/flag_variety_alignment/explore.py
```

The script checks the $\tanh$ forward formula against explicit two-token `softmax`, finite-differences all 16 gradient entries, verifies the projector identity of Proposition 1 on five rank-3 frames in $\mathbb{R}^7$, solves the stationarity cubic, and runs L-BFGS-B from 12 starts per regime (3 dense perturbations of the dedicated solution + 9 unstructured Gaussian starts, fixed seed `260509861`):

| Query weights $w$ | Task | Certified / reference $F$ | Best $F$ in 12 runs | Runs reaching reference | $R$ in those runs |
|---|---|---:|---:|---:|---:|
| $(1,1)$ | Orthogonal | 0.0457110274 | 0.0457110274 | 11/12 | 0 |
| $(2,1)$ | Orthogonal | 0.0457612963 | 0.0457612963 | 12/12 | 0 |
| $(2,1)$ | Shared | 0.0458115652 | 0.0458115652 | 12/12 | 1 |

Integrating the full 16-variable gradient flow (`DOP853`) from the $t = 1.1c$ tilt yields $\dot{R}(0) = -0.2966638584$, $R(0) = 0.547511312$, $R(1) = 0.290792478$, $R(10) = 1.29 \times 10^{-8}$, and $R(800) = 0$ with final objective gap $2.78 \times 10^{-17}$ from the certified global minimum.
