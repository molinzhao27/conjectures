# Refutation of the Cascading Minimization Conjecture for Quasi-Homogeneous Networks

**Target paper:** Daniel Kunin, Atsushi Yamamura, Chao Ma, and Surya Ganguli, *The Asymmetric Maximum Margin Bias of Quasi-Homogeneous Neural Networks*, ICLR 2023, [arXiv:2210.03820](https://arxiv.org/abs/2210.03820), Section 4, Conjecture `conj:cascading` (Cascading Minimization).  
**Companion files:** [`solutions/quasi_homog_cascade/explore.py`](explore.py), [`solutions/quasi_homog_cascade/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture `conj:cascading`: FALSE** under Assumptions A1–A6.
- **Explicit two-parameter counterexample (Theorem 1):** The smooth polynomial classifier $f(x; a, b) = x_1 a b + x_2 a^3$ has the *unique* diagonal rate matrix $\Lambda = \operatorname{diag}(1/3, 2/3)$. On two training examples $(e_1, +1)$ and $(-e_2, -1)$, every positive initialization $(a_0, b_0) \in (0, \infty)^2$ yields a global exponential-loss gradient-flow trajectory satisfying A1–A6 with both raw signed margins diverging to $+\infty$ and $\mathcal{L}(t) \to 0$. Yet the $\Lambda$-normalized limit is $\bar{\theta} = (\sqrt{3}, 0)$, at which $q_1(\bar{\theta}) = 0$. Every finite rescaling $\psi_\alpha(\bar{\theta})$ ($\alpha \in \mathbb{R}$) therefore violates the primal constraint $q_1(\theta) \ge 1$, precluding KKT optimality for *every* cascade threshold $\tilde{\lambda} \in \mathbb{R}_+$ (whether $\tilde{\lambda}$ is unrestricted or required to be an attained rate).
- **Structural mechanism:** Unlike decoupled additive ensembles of homogeneous networks (Nacson et al., 2019), multiplicative coupling between coordinates of distinct scaling rates ($ab$ paired with $a^3$) enables gradient flow to push the normalized highest-rate coordinate to zero even though that coordinate remains strictly necessary to separate the data.
- **Sign and quantifier completeness (Proposition 2 & Theorem 3):** We also expose a vacuity issue in the printed quantifier $\tilde{\lambda} \in \mathbb{R}_+$ (any limit with strictly positive margins becomes a trivial KKT point for $\tilde{\lambda} > \lambda_{\max}$ with zero multipliers), which our counterexample completely bypasses, and we resolve the missing-minus-sign typo in printed Assumption A4 by supplying a second explicit counterexample for the literal plus sign.

---

## 1. Setup, Assumptions A1–A6, and Quantifier Audit

Following Kunin et al. ([arXiv:2210.03820](https://arxiv.org/abs/2210.03820), Section 3–4), consider a model $f(x; \theta)$ trained on $\{(x_i, y_i)\}_{i=1}^n$ with $y_i \in \{-1, +1\}$ and signed margins $q_i(\theta) := y_i f(x_i; \theta)$. Assumptions A1–A6 are:

1. **A1 (Quasi-homogeneity):** There exists a nonzero diagonal positive semidefinite matrix $\Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_p)$ such that $f(x; \psi_\alpha \theta) = e^\alpha f(x; \theta)$ for all $x, \theta$ and all $\alpha \in \mathbb{R}$, where $\psi_\alpha \theta := e^{\alpha \Lambda}\theta$.
2. **A2 (Regularity):** For each fixed $x$, $f(x; \cdot)$ is locally Lipschitz and admits a chain rule.
3. **A3 (Exponential loss):** $\mathcal{L}(\theta) := \frac{1}{n} \sum_{i=1}^n e^{-q_i(\theta)}$.
4. **A4 (Gradient flow):** The printed display writes $\dot{\theta} \in \partial^\circ_\theta \mathcal{L}(\theta)$ for $t > 0$ (missing the minus sign), whereas the paper's setup defines gradient descent ($\dot{\theta} = -\nabla \mathcal{L}(\theta)$ in the smooth case) and its dynamics lemma uses $-\dot{\mathcal{L}} = \|\dot{\theta}\|_2^2$. We refute the conjecture under **both** signs.
5. **A5 (Separability):** There exists $t_0 \ge 0$ with $\mathcal{L}(\theta(t_0)) < 1/n$.
6. **A6 (Directional convergence):** The limit $\bar{\theta} := \lim_{t \to \infty} \hat{\theta}(t)$ exists, where $\hat{\theta}(t) := \psi_{-\tau(t)}\theta(t)$ with $\tau(t)$ chosen so that $\|\hat{\theta}(t)\|_\Lambda^2 := \sum_j \lambda_j \hat{\theta}_j(t)^2 = 1$.

**Conjecture `conj:cascading` (Cascading Minimization).** *Under A1–A6, there exist $\tilde{\lambda} \in \mathbb{R}_+$ and $\alpha \in \mathbb{R}$ such that $\psi_\alpha(\bar{\theta})$ is a first-order KKT point of*
$$
\min_\theta\; \frac{1}{2}\sum_{j : \lambda_j = \tilde{\lambda}} \lambda_j \theta_j^2
\quad\text{subject to}\quad
q_i(\theta) \ge 1\;\; (i \in [n]), \qquad
\theta_j = 0\;\; (\lambda_j > \tilde{\lambda}).
\tag{1}
$$

**Proposition 1 (Necessary and sufficient condition for the printed quantifier).** *Because $q_i(\psi_\alpha \bar{\theta}) = e^\alpha q_i(\bar{\theta})$ for every $\alpha \in \mathbb{R}$, strict positivity $q_i(\bar{\theta}) > 0$ for all $i \in [n]$ is necessary for $\psi_\alpha(\bar{\theta})$ to be feasible for (1) at any $\tilde{\lambda}$. Conversely, under the printed quantifier $\tilde{\lambda} \in \mathbb{R}_+$, strict positivity is also sufficient: choosing any $\tilde{\lambda} > \max_j \lambda_j$ makes the objective of (1) identically zero with no equality constraints, and choosing $\alpha := -\log\bigl(\min_i q_i(\bar{\theta})\bigr)$ makes $\psi_\alpha(\bar{\theta})$ feasible with all Lagrange multipliers equal to zero.*

Thus any genuine refutation of `conj:cascading` *must* violate primal feasibility by producing $q_i(\bar{\theta}) \le 0$ for some $i \in [n]$, which simultaneously refutes both the printed quantifier $\tilde{\lambda} \in \mathbb{R}_+$ and any tightened version restricting $\tilde{\lambda}$ to attained rates $\{\lambda_j\}$.

---

## 2. Counterexample for Gradient Descent

**Theorem 1 (Refutation of `conj:cascading` under gradient descent).** *Let $f(x; a, b) := x_1 a b + x_2 a^3$ on $\mathbb{R}^2$ with training set $\{(e_1, +1), (-e_2, -1)\}$.*
1. *$f$ satisfies A1 with the unique diagonal rate matrix $\Lambda = \operatorname{diag}(1/3, 2/3)$, and satisfies A2.*
2. *For every positive initialization $(a_0, b_0) \in (0, \infty)^2$, the exponential-loss gradient descent flow $\dot{\theta} = -\nabla \mathcal{L}(\theta)$ exists globally on $[0, \infty)$ and satisfies A3, A4 (descent), A5, and A6, with $a(t), b(t) \to +\infty$, $b(t)/a(t) \to 1$, $\mathcal{L}(\theta(t)) \to 0$, and $\Lambda$-normalized limit $\bar{\theta} = (\sqrt{3}, 0)$.*
3. *For every $\alpha \in \mathbb{R}$ and every $\tilde{\lambda} \in \mathbb{R}_+$, $\psi_\alpha(\bar{\theta})$ has $q_1(\psi_\alpha \bar{\theta}) = 0 < 1$ and is therefore infeasible (hence not a KKT point) for (1).*

*Proof.* **Step 1 (A1 and A2).** The model is polynomial in $(a, b)$, so A2 holds everywhere. The two signed margins are $q_1(a, b) = ab$ and $q_2(a, b) = a^3$. Under $\psi_\alpha(a, b) = (e^{\alpha/3}a, e^{2\alpha/3}b)$, both $ab$ and $a^3$ scale by $e^\alpha$. Conversely, requiring $q_1(e^{\alpha\lambda_a}a, e^{\alpha\lambda_b}b) = e^\alpha ab$ and $q_2(e^{\alpha\lambda_a}a, e^{\alpha\lambda_b}b) = e^\alpha a^3$ for all $a, b, \alpha$ forces $\lambda_a + \lambda_b = 1$ and $3\lambda_a = 1$, uniquely determining $\Lambda = \operatorname{diag}(1/3, 2/3)$.

**Step 2 (Global existence and positivity).** The exponential loss is $\mathcal{L}(a, b) = \frac{1}{2}(e^{-ab} + e^{-a^3})$, so gradient descent is the smooth ODE
$$
\dot{a} = \frac{1}{2}\bigl(b e^{-ab} + 3a^2 e^{-a^3}\bigr), \qquad
\dot{b} = \frac{1}{2} a e^{-ab}.
\tag{2}
$$
Starting from $a_0, b_0 > 0$, both velocities are strictly positive on $(0, \infty)^2$, so $a(t) \ge a_0 > 0$ and $b(t) \ge b_0 > 0$ are strictly increasing on the maximal interval $[0, T_{\max})$. By energy dissipation, $\int_0^t \|\dot{\theta}(s)\|_2^2\,ds = \mathcal{L}(\theta(0)) - \mathcal{L}(\theta(t)) \le \mathcal{L}(\theta(0))$, so Cauchy–Schwarz gives
$$
\|\theta(t) - \theta(0)\|_2 \le \int_0^t \|\dot{\theta}(s)\|_2\,ds \le \sqrt{t\,\mathcal{L}(\theta(0))}.
$$
Hence $\theta(t)$ cannot escape to infinity in finite time, so $T_{\max} = \infty$.

**Step 3 (Exact cancellation, A5, and asymptotic ratio $b/a \to 1$).** Define $D(t) := a(t)^2 - b(t)^2$. Differentiating along (2) cancels the $e^{-ab}$ terms identically:
$$
\dot{D} = 2a\dot{a} - 2b\dot{b} = 3a^3 e^{-a^3} > 0.
\tag{3}
$$
Thus $D(t) \ge D(0) = a_0^2 - b_0^2$, which rearranges to
$$
b(t)^2 \le a(t)^2 + b_0^2 - a_0^2 \qquad (\forall t \ge 0).
$$
If $a(t)$ were bounded above by $M$, then $b(t)$ would also be bounded above by some $M'$, and $\dot{a}(t) \ge \frac{1}{2} b_0 e^{-M M'} > 0$ would contradict boundedness of $a$. Hence $a(t) \to \infty$, $b(t) = O(a(t))$, and both $ab \ge b_0 a \to \infty$ and $a^3 \to \infty$. Consequently $\mathcal{L}(\theta(t)) \to 0$, verifying A5 for every positive start (at $(1, 1)$, $\mathcal{L}(0) = e^{-1} < 1/2$ already at $t = 0$).

In fact, $b(t)/a(t) \to 1$: because $a$ is strictly increasing and $\dot{a} > \frac{1}{2} b_0 e^{-ab}$, dividing (3) by $\dot{a}$ gives
$$
0 < \frac{dD}{da} = \frac{3a^3 e^{-a^3}}{\dot{a}} \le \frac{6a^3}{b_0} \exp(ab - a^3).
$$
Since $b = O(a)$, we have $ab - a^3 \le -\frac{1}{2} a^3$ for all sufficiently large $a$, so the right-hand side is integrable on $[a_0, \infty)$. Thus $D(t) \to D_\infty \in \mathbb{R}$, whence $b(t)^2 / a(t)^2 = 1 - D(t)/a(t)^2 \to 1$.

**Step 4 ($\Lambda$-normalized limit and failure of feasibility).** Write $r(t) := e^{-\tau(t)/3} > 0$. The normalization $\|\hat{\theta}\|_\Lambda^2 = \frac{1}{3}\hat{a}^2 + \frac{2}{3}\hat{b}^2 = 1$ reads
$$
(\hat{a}, \hat{b}) = (a r,\, b r^2), \qquad (a r)^2 + 2(b r^2)^2 = 3.
$$
Therefore $a r < \sqrt{3}$, i.e., $r < \sqrt{3}/a$, which yields
$$
0 < \hat{b}(t) = b(t) r(t)^2 < \frac{3 b(t)}{a(t)^2} \xrightarrow[t \to \infty]{} 0,
$$
since $b(t) = O(a(t))$ and $a(t) \to \infty$. By positivity and $\hat{a}^2 + 2\hat{b}^2 = 3$, it follows that $\hat{a}(t) \to \sqrt{3}$, proving A6 with $\bar{\theta} = (\sqrt{3}, 0)$. For every $\alpha \in \mathbb{R}$, $\psi_\alpha(\bar{\theta}) = (e^{\alpha/3}\sqrt{3}, 0)$ satisfies $q_1(\psi_\alpha \bar{\theta}) = 0 < 1$, violating primal feasibility of (1) for every $\tilde{\lambda} \in \mathbb{R}_+$. $\square$

*Remark (Failure of A7 and contrast with homogeneous ensembles).* Every point along the trajectory separates the data ($\mathcal{L} \to 0$), yet the normalized highest-rate coordinate $\hat{b} \to 0$ vanishes in the limit while $b$ remains indispensable for $q_1 = ab > 0$. This shows precisely why Assumption A7 (conditional separability in Theorem `theorem:KKT` of Kunin et al., 2023) cannot be removed. By contrast, in decoupled ensembles of homogeneous networks (Nacson et al., 2019), setting the highest-rate sub-model to zero leaves an independent lower-rate sub-model that can still separate the data on its own.

---

## 3. Counterexample for the Literal Printed Sign in A4

For completeness, we confirm that `conj:cascading` also fails if Assumption A4 is read literally with its printed plus sign ($\dot{\theta} = +\nabla \mathcal{L}(\theta)$ in the smooth case).

**Theorem 2 (Refutation under the literal plus sign in A4).** *Take $f(x; a, b) = x_1 a b + x_2 a^3$ with unique rate matrix $\Lambda = \operatorname{diag}(1/3, 2/3)$ and training examples $\{(e_1, +1), (-e_1, -1)\}$. From $(a_0, b_0) = (1, 1)$, the literal-sign flow $\dot{\theta} = +\nabla \mathcal{L}(\theta)$ exists globally on $[0, \infty)$ and satisfies A1–A6, with $\Lambda$-normalized limit $\bar{\theta} = (0, \sqrt{3/2})$, at which $q_1(\psi_\alpha \bar{\theta}) = q_2(\psi_\alpha \bar{\theta}) = 0$ for every $\alpha \in \mathbb{R}$.*

*Proof.* Both signed margins equal $q_1 = q_2 = ab$, so $\mathcal{L}(a, b) = e^{-ab}$. At $(1, 1)$, $\mathcal{L}(0) = e^{-1} < 1/2$, and continuity preserves $\mathcal{L}(t) < 1/2$ on a positive interval, verifying A5. The smooth ODE $\dot{\theta} = +\nabla\mathcal{L}(\theta)$ is $(\dot{a}, \dot{b}) = -e^{-ab}(b, a)$. By uniqueness and symmetry of the initial data $(1, 1)$, $a(t) = b(t) = s(t)$ where $\dot{s} = -s e^{-s^2}$ and $s(0) = 1$. Since $0 < s \le 1$, we have $-1 \le \frac{d}{dt}(\log s) = -e^{-s^2} \le -e^{-1}$, giving global existence and the two-sided bound
$$
e^{-t} \le s(t) \le e^{-t/e} \qquad (\forall t \ge 0),
$$
so $s(t) \to 0^+$. Writing $r = e^{-\tau/3} > 0$, the normalization $(s r)^2 + 2(s r^2)^2 = 3$ gives $\hat{b} = s r^2 \le \sqrt{3/2}$, whence $\hat{a}^2 = s^2 r^2 = s\hat{b} \le s\sqrt{3/2} \to 0$ and $\hat{b} \to \sqrt{3/2}$. This verifies A6 with $\bar{\theta} = (0, \sqrt{3/2})$. Both margins vanish at $\psi_\alpha(\bar{\theta}) = (0, e^{2\alpha/3}\sqrt{3/2})$ for every $\alpha \in \mathbb{R}$. $\square$

---

## 4. Reproducible Numerical Verification

Run from the repository root:

```sh
.venv/bin/python -B solutions/quasi_homog_cascade/explore.py
```

The script [`explore.py`](explore.py) checks quasi-homogeneity, integrates the descent ODE (2) in physical time to $t = 10^6$ (DOP853, `rtol=2e-11, atol=1e-13`) for three positive initializations, cross-checks against the exact orbit equation $da/db = b/a + 3a\exp(ab - a^3)$ (which avoids physical-time velocity underflow), continues the orbit to $b = 10^4$, and separately integrates the literal-sign ODE to $t = 16$:

| Flow / initialization | Terminal checkpoint | Raw $\min_i q_i$ | Normalized $\min_i q_i(\hat{\theta})$ |
|---|---:|---:|---:|
| Descent, $(a_0, b_0) = (1.0, 1.0)$ | $b = 10^4$ | $1.00000 \times 10^8$ | $5.19615 \times 10^{-4}$ |
| Descent, $(a_0, b_0) = (1.5, 1.0)$ | $b = 10^4$ | $1.00000 \times 10^8$ | $5.19615 \times 10^{-4}$ |
| Descent, $(a_0, b_0) = (1.0, 1.25)$ | $b = 10^4$ | $1.00000 \times 10^8$ | $5.19615 \times 10^{-4}$ |
| Literal A4 sign, $(1.0, 1.0)$ | $t = 16$ | — | $6.32124 \times 10^{-4}$ |

All assertions pass with exit status `0`.

---

## References

- D. Kunin, A. Yamamura, C. Ma, and S. Ganguli, *The Asymmetric Maximum Margin Bias of Quasi-Homogeneous Neural Networks*, ICLR 2023, [arXiv:2210.03820](https://arxiv.org/abs/2210.03820).
- M. S. Nacson, S. Gunasekar, J. D. Lee, N. Srebro, and D. Soudry, *Lexicographic and Depth-Sensitive Margins in Homogeneous and Non-Homogeneous Deep Models*, ICML 2019, [arXiv:1905.07325](https://arxiv.org/abs/1905.07325).
