# Proof of Virtual-Weight Identifiability & Dimension Formula for Conjecture 3.10 (`conj:deepsmax`, arXiv:2408.17221)

- **Conjecture Source**: Nathan W. Henry, Giovanni Luca Marchetti, Kathlén Kohn, *"Geometry of Lightning Self-Attention: Identifiability and Dimension"*, [arXiv:2408.17221](https://arxiv.org/abs/2408.17221) (v3 Jun 2026; ICLR 2025), Section 3.4, Conjecture 3.10 (`conj:deepsmax`) and Dimension Formula (`eq:conjform`).
- **Domain**: Machine Learning Theory / Algebraic & Differential Geometry of Transformers (Deep Softmax Self-Attention).
- **Verdict**: **PROVED for all $C^{1,1}$ normalizers with $\mathcal{S}'(0) \neq 0$** (in particular softmax $\mathcal{S}(x) = e^{x/\tau}$, $\tau > 0$), for all depths $l \ge 1$, all sequence lengths $t \ge 2$, and all positive architectures $\mathbf{d}, \mathbf{a}$: virtual weights $(\mathbf{M}, L)$ are uniquely identifiable on the explicit Zariski-open locus $\mathcal{U} = \{L \neq 0,\ M_r + M_r^\top \neq 0\ \forall 1 \le r \le l-1\}$, which is both necessary and sufficient for singleton fibers when $t = 2$ with softmax; combining Theorem 1 with real-analytic constant-rank and Henry–Marchetti–Kohn's `matfact`/`M_fibers` lemmas gives the neuromanifold dimension formula (`eq:conjform`) for $l \ge 2$ (for $l = 1$, $\delta$ in `eq:conjform` is undefined in the paper). Note that non-analytic injective normalizers with $\mathcal{S}'(0) = 0$ (e.g., $\mathcal{S}(x) = e^{x^3}$) or non-smooth $\mathcal{S}$ remain outside the scope of Theorem 1.
- **Prior Art & Novelty Note**:
  - **Prior art (`arXiv:2610.09620`):** P. V. Konda, *"The Identifiability and Observability of Deep Normalized Attention"* ([arXiv:2610.09620](https://arxiv.org/abs/2610.09620), Oct 7, 2026, unrefereed v1) independently claims a proof of virtual-weight identifiability on the same locus $\mathcal{U}$ for positive nonconstant real-analytic normalizers (including softmax), together with the $t = 2$ skew-collapse sharpness, an immersion theorem on $\mathcal{U}$, and the equal-width local rank formula.
  - **Genuinely novel contributions here:** (i) our **complete, zero-`sorry` Lean 4 + Mathlib machine-checked proof** of virtual-weight deep softmax identifiability (`deep_softmax_identifiability`) for all $l \ge 1, t \ge 2$ and first-layer $t = 2$ skew collapse (`t2_skew_collapse`); (ii) our **non-analytic $C^{1,1}$ normalizer proof** requiring only $C^{1,1}$ regularity near $0$ and $\mathcal{S}'(0) \neq 0$ (explicitly left open by Konda); and (iii) the unequal-width derivation of `eq:conjform` via real-analytic generic rank combined with Henry–Marchetti–Kohn's `matfact`/`M_fibers` factorization.

---

## 1. Setup and Virtual-Weight Reparametrization

Fix integers $l \ge 1$ (depth), $t \ge 2$ (sequence length), and $d_0, d_l \ge 1$. Let $\mathcal{S}\colon \mathbb{R} \to \mathbb{R}_{>0}$ be $C^2$ on $(-\delta_0, \delta_0)$ with $\mathcal{S}'(0) \neq 0$, and set $f(x) := \log \mathcal{S}(x)$ and $\alpha := f'(0) = \mathcal{S}'(0)/\mathcal{S}(0) \neq 0$.

For virtual weights $\mathbf{W} = (\mathbf{M}, L) \in (\mathbb{R}^{d_0 \times d_0})^l \times \mathbb{R}^{d_l \times d_0}$ (where $L = V_l \cdots V_1$ and $M_r = L_{r-1}^\top A_r L_{r-1}$) and input $X = (x_1, \dots, x_t) \in \mathbb{R}^{d_0 \times t}$, define $Z^{(0)}(X) := X$ and inductively for $r = 1, \dots, l$:
$$
Z^{(r)}(X)_i := \frac{\sum_{j=1}^t \mathcal{S}\!\left((Z^{(r-1)}_i)^\top M_r Z^{(r-1)}_j\right) Z^{(r-1)}_j}{\sum_{k=1}^t \mathcal{S}\!\left((Z^{(r-1)}_i)^\top M_r Z^{(r-1)}_k\right)}, \qquad 1 \le i \le t.
$$
By induction on $r$, $X^{(r)} = L_r Z^{(r)}(X)$ and the full $l$-layer normalized self-attention map is $\varphi_{(\mathbf{M}, L)}(X) = L\, Z^{(l)}(X)$.

---

## 2. Main Theorem (Global Identifiability on an Explicit Zariski-Open Locus)

**Theorem 1.** *Let $l \ge 1$, $t \ge 2$, $d_0, d_l \ge 1$, and let $\mathcal{S}\colon \mathbb{R} \to \mathbb{R}_{>0}$ be $C^2$ near $0$ with $\mathcal{S}'(0) \neq 0$. Let $\mathbf{W} = (\mathbf{M}, L)$ and $\mathbf{W}' = (\mathbf{M}', L')$ in $(\mathbb{R}^{d_0 \times d_0})^l \times \mathbb{R}^{d_l \times d_0}$. Assume:*
1. *$L \neq 0$,*
2. *$M_r + M_r^\top \neq 0$ (i.e., $M_r$ is not skew-symmetric) for every $1 \le r \le l - 1$.*

*If $\varphi_{\mathbf{W}}(X) = \varphi_{\mathbf{W}'}(X)$ for all $X$ in some open neighborhood of $0 \in \mathbb{R}^{d_0 \times t}$, then $\mathbf{W} = \mathbf{W}'$ (i.e., $L = L'$ and $M_r = M'_r$ for all $1 \le r \le l$).*

### Proof

Set $\sigma(z) := \frac{1}{1 + (t - 1)e^{-z}} \in (0, 1)$, so $\sigma(0) = \frac{1}{t}$, $\sigma'(0) = \frac{t - 1}{t^2} > 0$, and $\frac{\mathcal{S}(p)}{\mathcal{S}(p) + (t - 1)\mathcal{S}(q)} = \sigma(f(p) - f(q))$.

**Step 1 (Recovery of $L$).** On constant-token inputs $X = (x, \dots, x)$, induction yields $Z^{(r)}(X)_i = x$ for all $r, i$, so $\varphi_{\mathbf{W}}(x, \dots, x)_1 = L x = L' x$ near $0$, giving $L = L'$.

**Step 2 (Two-Cluster Reduction).** Fix $y, v \in \mathbb{R}^{d_0}$ and $\eta \in \mathbb{R}$. For $u > 0$, set $\mu := \sqrt{u}$ and
$$
X(u) := \big(\mu(\eta y + v),\; \mu \eta y,\; \dots,\; \mu \eta y\big) \in \mathbb{R}^{d_0 \times t}.
$$
Define $a_0(u) := 1$, $b_0(u) := 0$, and for $1 \le r \le l$:
$$
p_{r,1}(u) = u(\eta y + a_{r-1} v)^\top M_r (\eta y + a_{r-1} v), \quad q_{r,1}(u) = u(\eta y + a_{r-1} v)^\top M_r (\eta y + b_{r-1} v),
$$
$$
p_{r,2}(u) = u(\eta y + b_{r-1} v)^\top M_r (\eta y + a_{r-1} v), \quad q_{r,2}(u) = u(\eta y + b_{r-1} v)^\top M_r (\eta y + b_{r-1} v),
$$
$$
w_{r,k}(u) = \sigma(f(p_{r,k}(u)) - f(q_{r,k}(u))) \in (0, 1), \quad a_r = b_{r-1} + d_{r-1} w_{r,1}, \quad b_r = b_{r-1} + d_{r-1} w_{r,2},
$$
where $d_r := a_r - b_r$.

*Lemma 1 (Two-Cluster Invariance & Contraction).* For all $u > 0$ and $0 \le r \le l$:
1. $Z^{(r)}(X(u))_1 = \mu(\eta y + a_r(u) v)$ and $Z^{(r)}(X(u))_i = \mu(\eta y + b_r(u) v)$ for all $2 \le i \le t$.
2. $a_r(u), b_r(u) \in [0, 1]$, and nested convex-hull contraction gives $|a_l(u) - b_r(u)| \le |d_r(u)|$.

*Proof.* Immediate induction on $r$, since $w_{r,k} \in (0, 1)$ makes $a_r, b_r$ convex combinations of $a_{r-1}, b_{r-1}$, so $[\min(a_l, b_l), \max(a_l, b_l)] \subseteq \dots \subseteq [\min(a_r, b_r), \max(a_r, b_r)] \subseteq [0, 1]$. $\square$

**Step 3 (Exact Small-$u$ Asymptotics).** Let $\beta_r := v^\top M_r v$, $\gamma_r := y^\top M_r v$, $e_r := \frac{3^r - 1}{2}$ (so $e_0 = 0$, $e_r = 3e_{r-1} + 1$, $2e_{r-1} + 1 = 3^{r-1}$), $c_0 := 1$, $c_r := (\sigma'(0)\alpha)\beta_r c_{r-1}^3 = (\sigma'(0)\alpha)^{e_r}\prod_{j=1}^r \beta_j^{3^{r-j}}$, and $b_0(0) := 0$, $b_r(0) := \frac{1}{t}$ ($r \ge 1$).

*Lemma 2 (Small-$u$ Expansion).* For every $r \in \{1, \dots, l\}$, as $u \to 0^+$:
1. $a_r(u) = \frac{1}{t} + O(u)$ and $b_r(u) = \frac{1}{t} + O(u)$;
2. $d_r(u) = c_r\, u^{e_r} + O(u^{e_r + 1})$;
3. $b_r(u) - b_{r-1}(u) - \frac{1}{t} d_{r-1}(u) = \sigma'(0)\alpha\, c_{r-1}^2(\eta \gamma_r + b_{r-1}(0)\beta_r)\, u^{3^{r-1}} + O(u^{3^{r-1} + 1})$.

*Proof.* By induction on $r \ge 1$. Since $a_{r-1}, b_{r-1} \in [0, 1]$, $|p_{r,k}(u)|, |q_{r,k}(u)| \le Cu$, and $f'$ is Lipschitz on $[-\delta_0/2, \delta_0/2]$. The logit differences $\Delta_k := p_{r,k} - q_{r,k}$ satisfy
$$
\Delta_2(u) = u d_{r-1}(\eta \gamma_r + b_{r-1}\beta_r) = c_{r-1}(\eta\gamma_r + b_{r-1}(0)\beta_r) u^{e_{r-1}+1} + O(u^{e_{r-1}+2}),
$$
$$
\Delta_1(u) - \Delta_2(u) = u d_{r-1}^2 \beta_r = c_{r-1}^2 \beta_r u^{2e_{r-1}+1} + O(u^{2e_{r-1}+2}), \qquad q_{r,1} - q_{r,2} = O(u^{e_{r-1}+1}).
$$
Setting $g_k := f(p_{r,k}) - f(q_{r,k})$, the Mean Value Theorem gives $g_k = (\alpha + O(u))\Delta_k = \alpha\Delta_k + O(u^{e_{r-1}+2})$, so $w_{r,k} = \sigma(g_k) = \frac{1}{t} + O(u)$, giving (1), and
$$
\sigma(g_2) - \frac{1}{t} = \sigma'(0)\alpha c_{r-1}(\eta\gamma_r + b_{r-1}(0)\beta_r)u^{e_{r-1}+1} + O(u^{e_{r-1}+2}),
$$
which multiplied by $d_{r-1} = c_{r-1}u^{e_{r-1}} + O(u^{e_{r-1}+1})$ yields (3). For (2), applying the Mean Value Theorem to $\psi(s) = f(q_{r,2} + s(q_{r,1}-q_{r,2}) + \Delta_2 + s(\Delta_1-\Delta_2)) - f(q_{r,2} + s(q_{r,1}-q_{r,2}))$ on $[0, 1]$ gives
$$
g_1 - g_2 = f'(q_\theta + \Delta_\theta)(\Delta_1 - \Delta_2) + \big(f'(q_\theta + \Delta_\theta) - f'(q_\theta)\big)(q_{r,1} - q_{r,2}) = \alpha c_{r-1}^2 \beta_r u^{2e_{r-1}+1} + O(u^{2e_{r-1}+2}),
$$
since $|f'(q_\theta + \Delta_\theta) - f'(q_\theta)| \cdot |q_{r,1}-q_{r,2}| \le K_f |\Delta_\theta| \cdot |q_{r,1}-q_{r,2}| = O(u^{2e_{r-1}+2})$. By MVT on $\sigma$, $\sigma(g_1) - \sigma(g_2) = (\sigma'(0) + O(u))(g_1 - g_2)$, and multiplying by $d_{r-1}$ yields $d_r(u) = c_r u^{e_r} + O(u^{e_r+1})$. $\square$

**Step 4 (Layer-by-Layer Recovery).** Whenever $L v \neq 0$, $L = L'$ and Lemma 1(1) imply $a_l(u) = a'_l(u)$ for all small $u > 0$ and all $\eta \in \mathbb{R}$.
- **Base $r = 1$:** For $l = 1$, $a_1(u) - \frac{1}{t} = \sigma'(0)\alpha(\eta y^\top M_1 v + v^\top M_1 v) u + O(u^2)$. For $l \ge 2$, $|a_l - b_2| \le |d_2| = O(u^4)$ and $b_2 = b_1 + \frac{1}{t} d_1 + O(u^3)$, so $a_l(u) - \frac{1}{t} = \sigma'(0)\alpha(\eta y^\top M_1 v + \frac{1}{t} v^\top M_1 v) u + O(u^2)$. Comparing coefficients of $u$ at $\eta = 1$ minus $\eta = 0$ gives $y^\top M_1 v = y^\top M'_1 v$ for all $y$ and all $v \notin \ker(L)$ (dense since $L \neq 0$), so $M_1 = M'_1$.
- **Inductive step ($2 \le r \le l$):** Assume $M_j = M'_j$ for $1 \le j \le r - 1$. The set $U_r = \{v \in \mathbb{R}^{d_0} : Lv \neq 0 \text{ and } v^\top M_j v \neq 0\ \forall j < r\}$ is nonempty Zariski-open (hence dense) because $L \neq 0$ and each $M_j + M_j^\top \neq 0$. For $v \in U_r$, $b_{r-1} = b'_{r-1}$, $d_{r-1} = d'_{r-1}$, and $c_{r-1} = c'_{r-1} \neq 0$. Since $r \ge 2$, the crucial exponent gap
  $$e_r = \frac{3^r - 1}{2} = 3^{r-1} + \frac{3^{r-1} - 1}{2} \ge 3^{r-1} + 1$$
  implies $|a_l(u) - b_r(u)| \le |d_r(u)| = O(u^{3^{r-1}+1})$. Hence by Lemma 2(3),
  $$a_l(u) - b_{r-1}(u) - \frac{1}{t} d_{r-1}(u) = \sigma'(0)\alpha\, c_{r-1}^2\Big(\eta (y^\top M_r v) + \frac{1}{t} (v^\top M_r v)\Big) u^{3^{r-1}} + O(u^{3^{r-1}+1}).$$
  Dividing by $u^{3^{r-1}}$, letting $u \to 0^+$, and taking $\eta = 1$ minus $\eta = 0$ yields $y^\top M_r v = y^\top M'_r v$ on the dense set $U_r$, hence $M_r = M'_r$. $\blacksquare$

---

## 3. Sharpness ($t = 2$), Conjecture `conj:deepsmax`, and Dimension Formula

- **Proposition 2 (Sharpness for $t = 2$ with softmax $\mathcal{S}(x) = e^{x/\tau}$):** The fiber of $\varphi_{(\mathbf{M}, L)}$ is a singleton **if and only if** $L \neq 0$ and $M_r + M_r^\top \neq 0$ for all $1 \le r \le l - 1$. Indeed, if $M_r^\top = -M_r$ ($r < l$), then on any $X \in \mathbb{R}^{d_0 \times 2}$, $z_1^\top M_r z_1 = z_2^\top M_r z_2 = 0$ and $z_2^\top M_r z_1 = -z_1^\top M_r z_2$, so $\frac{1}{1+e^{c/\tau}} = \frac{e^{-c/\tau}}{e^{-c/\tau}+1}$ collapses $Z^{(r)}_1 = Z^{(r)}_2$ identically, making the network independent of $M_{r+1}, \dots, M_l$.
- **Corollary 3 (Generic Identifiability of $(\mathbf{Q}, \mathbf{K}, \mathbf{V})$ & Dimension Formula `eq:conjform`):**
  1. The Zariski-open locus $\mathcal{U} = \{L \neq 0,\ M_r + M_r^\top \neq 0\ \forall r < l\}$ intersects the realizable virtual-weight image $\mathcal{V}_{\mathbf{d}, \mathbf{a}} = \operatorname{im}((\mathbf{Q}, \mathbf{K}, \mathbf{V}) \mapsto (\mathbf{M}, L))$ nonemptily (witnessed by $V_r = Q_r = K_r = E_{1,1}$). Intersecting $\mathcal{U}$ with the generic full-rank locus required by Henry–Marchetti–Kohn's Lemmas `matfact` and `M_fibers` ($\operatorname{rk}(A_r) = \alpha_r$, $\operatorname{rk}(L) = \delta$, etc.) recovers the raw $(\mathbf{Q}, \mathbf{K}, \mathbf{V})$ parameters up to the gauge symmetries of those lemmas (without the $l$ scaling symmetries of unnormalized attention, because normalization fixes the scale of each $M_r$).
  2. **Dimension formula (`eq:conjform`) for real-analytic $\mathcal{S}$ (e.g., softmax) and $l \ge 2$:** Let $\Psi : \mathcal{V}_{\mathbf{d}, \mathbf{a}} \to \operatorname{Map}(\mathbb{R}^{d_0 \times t}, \mathbb{R}^{d_l \times t})$ be the virtual-weight-to-function map. Because $\Psi$ is real-analytic and injective on the dense open subset $\mathcal{U} \cap \mathcal{V}_{\mathbf{d}, \mathbf{a}}$, its differential $d\Psi$ must have full generic rank equal to $\dim \mathcal{V}_{\mathbf{d}, \mathbf{a}}$ (otherwise, by the constant-rank theorem, $\Psi$ would have positive-dimensional fibers on a nonempty open set, contradicting Theorem 1). Evaluating $\dim \mathcal{V}_{\mathbf{d}, \mathbf{a}}$ as in Corollary `dimneur` of Henry–Marchetti–Kohn — with the $l$ unnormalized scaling redundancies removed — yields `eq:conjform` for all $l \ge 2$ and arbitrary (possibly unequal) widths. (Note: for $l = 1$, $\delta$ in `eq:conjform` is not defined in the paper; and a direct proof of the stronger *everywhere immersion* claim on $\mathcal{U}$ is not written out here, though Konda `arXiv:2610.09620` Theorem 3 / Prop. C.1 claims it for real-analytic normalizers.)

Computational verification: [`verify_attention.py`](./verify_attention.py).
