# A $2\times 2$ Counterexample to Moving-Reference I-Projection Selection

**Target paper:** Luis Leal, *Which Nash Equilibrium? Solver-Dependent Selection on Zero-Sum Nash Polytopes*, [arXiv:2606.28308](https://arxiv.org/abs/2606.28308) (June 2026), Section 7, Conjecture 1 (`conj:iproj`).  
**Companion files:** [`solutions/nash_iproj_selection/explore.py`](explore.py), [`solutions/nash_iproj_selection/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture 1 (`conj:iproj`): FALSE**, including its uniform-reference / maximum-entropy specialization.
- **Explicit $2\times 2$ counterexample:** For the zero-sum matrix game
  $$
  A = \begin{pmatrix} 0 & 2 \\ 0 & -1 \end{pmatrix},
  $$
  the Nash polytope is $\mathcal{N}(A) = [1/3, 1] \times \{0\}$ (writing $p$ for row $1$ and $q$ for column $2$), and the I-projection of any interior reference $(p_0, q_0)$ is $\Pi_{\mathcal{N}}(p_0, q_0) = (\max\{1/3, p_0\}, 0)$. In particular, uniform initialization projects to the unique maximum-entropy equilibrium $\sigma^{\mathrm{ME}} = (1/2, 0)$.
- **Universal strict discrepancy across all four moving-reference regimes (Theorems 1–4):** For every interior initialization with $p_0 \in (1/3, 1)$ and $q_0 \in (0, 1)$, the deployed limit is a Nash equilibrium $(p_\infty, 0) \in \mathcal{N}(A)$ with **$p_\infty > p_0$ strictly** under:
  1. **Exact anchored QRE resets** at every fixed temperature $\lambda > 0$ (e.g., from uniform start at $\lambda = 1$, $p_\infty \approx 0.7632330195 > 0.5$), and the limit **depends on $\lambda$** ($u_\infty(h=1) > 0.35 > 0.017 > u_\infty(h=16)$);
  2. **Simultaneous discrete MMD (`eq:mmd`) with periodic resets** for every learning rate $\eta > 0$, temperature $\lambda > 0$, and reset period $m \ge 1$ (full last-iterate convergence);
  3. **Small-step continuous-time flow with periodic resets** every $T > 0$ at any $\lambda > 0$;
  4. **Continuously reset flow ($\lambda = 0$)**, where an exact first integral characterizes the uniform-start limit as the unique root $p \approx 0.8656496057 \in (1/2, 1)$ of $p(1 - p)^2 = 1/64$.
- **Fixed-anchor vs. moving-reference contrast:** Taking a *single* fixed-anchor exact solve from $\rho_0$ as $\lambda \downarrow 0$ (and likewise the exact-reset limit as $\lambda \downarrow 0$ on this game) *does* recover $\Pi_{\mathcal{N}}(\rho_0)$. The refutation applies to every fixed $\lambda > 0$ (and every $\eta > 0, m \ge 1, T > 0$, and continuous reset), which is the moving-reference R-NaD / MMD algorithm defined in arXiv:2606.28308.
- **Structural explanation (Proposition 5):** Appending strictly dominated column(s) to the paper's own `asym_safe` core breaks the tangent-orthogonality invariant $z^\top \log(x_t/\rho_0) = 0$ while leaving $\mathcal{N}(A)$ and $\Pi_{\mathcal{N}}(\rho_0)$ unchanged, explaining why Leal's 180-game random ensemble (containing no unused columns) never observed the discrepancy.

---

## 1. Model, Nash Polytope, and I-Projection

In arXiv:2606.28308 (Section 2, `eq:me`, `eq:iproj`, `eq:mmd`), the I-projection of an interior reference $\rho_0$ onto the Nash polytope $\mathcal{N}(G)$ minimizes $\mathrm{KL}(\sigma \| \rho_0)$ over $\sigma \in \mathcal{N}(G)$. For a two-player matrix game, the paper averages entropy/KL over information sets by $1/|\mathcal{I}| = 1/2$, whereas the unaveraged sum $D(\sigma \| \rho_0) := D(x \| x_0) + D(y \| y_0)$ has the exact same minimizers. Under uniform $\rho_0$, $\Pi_{\mathcal{N}}(\mathrm{unif})$ is the maximum-entropy equilibrium $\sigma^{\mathrm{ME}}$.

Let the row player maximize and the column player minimize $x^\top A y$ for
$$
A = \begin{pmatrix} 0 & 2 \\ 0 & -1 \end{pmatrix}.
$$
Write $x = (p, 1 - p)^\top$ and $y = (1 - q, q)^\top$, where $p$ is the probability of row $1$ and $q$ is the probability of column $2$. The expected row payoff is $q(3p - 1)$. Column $1$ guarantees payoff $0$, and row strategy $p = 1/3$ also guarantees $0$, so the game value is $0$ and the row maximin set is $p \in [1/3, 1]$. For every $q > 0$, pure row $1$ earns $2q > 0$, so the column minimax strategy is uniquely $q = 0$. Hence
$$
\mathcal{N}(A) = [1/3, 1] \times \{0\}, \qquad
\Pi_{\mathcal{N}}(p_0, q_0) = \bigl(\max\{1/3, p_0\},\, 0\bigr)
$$
for every interior reference $(p_0, q_0) \in (0, 1)^2$: on $\mathcal{N}(A)$ the column KL term $-\log(1 - q_0)$ is constant, while the Bernoulli row KL $p \log(p/p_0) + (1 - p)\log((1 - p)/(1 - p_0))$ is strictly convex with unique unconstrained minimum at $p_0$. In particular, $\Pi_{\mathcal{N}}(1/2, 1/2) = \sigma^{\mathrm{ME}} = (1/2, 0)$.

Throughout Sections 2–4, fix an interior initial profile $(p_0, q_0)$ with $1/3 < p_0 < 1$ and $0 < q_0 < 1$, and start both the policy and the reference at $(p_0, q_0)$. Put
$$
g(z) := \frac{1}{1 + e^{-z}}, \qquad
u := \log\frac{p}{1 - p}, \qquad
v := \log\frac{q}{1 - q}, \qquad
d := 3p_0 - 1 > 0.
$$
The relevant payoff differences are $(Ay)_1 - (Ay)_2 = 3q = 3g(v)$ for the row player and $(-A^\top x)_2 - (-A^\top x)_1 = -(3p - 1) = -(3g(u) - 1)$ for the column player.

---

## 2. Exact Anchored QRE Resets and Temperature Dependence

Fix temperature $\lambda > 0$ and set $h := 1/\lambda > 0$. At a fixed point of the anchored MMD update (`eq:mmd`)—equivalently, the unique saddle point of $x^\top A y - \lambda D(x \| x_n) + \lambda D(y \| y_n)$—taking log-odds cancels the normalization and learning rate, so resetting the anchor to that fixed point defines
$$
u_{n+1} = u_n + 3h\,g(v_{n+1}), \qquad
v_{n+1} = v_n - h\bigl(3g(u_{n+1}) - 1\bigr).
\tag{1}
$$

**Theorem 1 (Exact anchored QRE resets).** *For every $h = 1/\lambda > 0$ and every interior start with $p_0 \in (1/3, 1)$, $q_0 \in (0, 1)$:*
1. *Each step $(u_{n+1}, v_{n+1})$ of (1) exists uniquely.*
2. *The sequence converges to a Nash equilibrium $(g(u_\infty), 0) \in \mathcal{N}(A)$ with*
   $$
   0 < u_1 - u_0 \le u_\infty - u_0 \le \frac{3h e^{v_0}}{e^{hd} - 1} < \infty,
   $$
   *so $g(u_\infty) > p_0 = \Pi_{\mathcal{N}}(p_0, q_0)_1$.*
3. *From uniform initialization $(p_0, q_0) = (1/2, 1/2)$, the limit strictly depends on $\lambda$: $u_\infty(h = 1) \ge 3/(1 + e^2) > 0.35 > 0.017 > 48/(e^8 - 1) \ge u_\infty(h = 16)$.*

*Proof.* Substituting the first equation of (1) into the second gives the scalar equation $\Phi(b) := b - v_n + h\{3g(u_n + 3h g(b)) - 1\} = 0$ for $b = v_{n+1}$. Because $g$ is strictly increasing and bounded, $\Phi'(b) \ge 1$ everywhere and $\Phi(b) \to \pm\infty$ as $b \to \pm\infty$, giving unique existence of $(u_{n+1}, v_{n+1})$.

Since $g(v_{n+1}) > 0$, we have $u_{n+1} > u_n \ge u_0$ for all $n \ge 0$ by induction, so $3g(u_{n+1}) - 1 \ge 3p_0 - 1 = d > 0$ and therefore $v_{n+1} \le v_n - hd$, yielding $v_k \le v_0 - khd$ for all $k \ge 1$. Using $g(b) < e^b$,
$$
0 < u_1 - u_0 \le u_\infty - u_0 = 3h \sum_{k=1}^\infty g(v_k) \le 3h \sum_{k=1}^\infty e^{v_0 - khd} = \frac{3h e^{v_0}}{e^{hd} - 1} < \infty.
$$
Thus $u_n \uparrow u_\infty \in (u_0, \infty)$ and $v_n \to -\infty$, so $(g(u_n), g(v_n)) \to (g(u_\infty), 0) \in \mathcal{N}(A)$ with $g(u_\infty) > p_0$.

At uniform start ($u_0 = v_0 = 0$, $d = 1/2$): for $h = 1$, $v_1 = -(3g(u_1) - 1) \ge -2$, so $u_\infty \ge u_1 = 3g(v_1) \ge 3/(1 + e^2) > 0.35$; for $h = 16$, the upper bound gives $u_\infty \le 48/(e^8 - 1) < 0.017$. $\square$

*Fixed-anchor comparison:* A *single* fixed-anchor solve from $\rho_0$ satisfies $0 < u_1 - u_0 \le 3h e^{v_0 - hd} \to 0$ and $q_1 \to 0$ as $\lambda \downarrow 0$ ($h \to \infty$), and indeed the moving-reference bound $3h e^{v_0}/(e^{hd} - 1)$ also vanishes as $\lambda \downarrow 0$. Thus vanishing-regularization fixed-anchor theory (McKelvey & Palfrey 1995; Turocy 2005; Cuturi 2013; Weed 2018) recovers $\Pi_{\mathcal{N}}(\rho_0)$, whereas moving-reference R-NaD at any fixed $\lambda > 0$ does not.

---

## 3. Simultaneous Discrete MMD With Periodic Resets

Fix learning rate $\eta > 0$, temperature $\lambda > 0$, $c := (1 + \eta\lambda)^{-1} \in (0, 1)$, and reset period $m \ge 1$. Within reset block $n$, hold anchor logits $(U_n, V_n)$ and start the policy at $(u_0, v_0) = (U_n, V_n)$. One step of the paper's simultaneous update (`eq:mmd`) in log-odds is
$$
u_{i+1} - U_n = c\bigl(u_i - U_n + 3\eta g(v_i)\bigr), \qquad
v_{i+1} - V_n = c\bigl(v_i - V_n - \eta(3g(u_i) - 1)\bigr),
$$
and after $m$ steps the anchor resets to $(U_{n+1}, V_{n+1}) := (u_m, v_m)$.

**Theorem 2 (Discrete MMD with periodic resets).** *For every $\eta > 0$, $\lambda > 0$, $m \ge 1$, and interior initialization with $p_0 \in (1/3, 1)$, $q_0 \in (0, 1)$, the full last-iterate sequence converges to a Nash equilibrium $(g(U_\infty), 0) \in \mathcal{N}(A)$ with $g(U_\infty) > p_0$.*

*Proof.* Unrolling the linear recurrence within block $n$ gives, for $1 \le j \le m$,
$$
u_j - U_n = 3 c\eta \sum_{i=0}^{j-1} c^{j-1-i} g(v_i), \qquad
v_j - V_n = -c\eta \sum_{i=0}^{j-1} c^{j-1-i}\bigl(3g(u_i) - 1\bigr).
$$
Since $1 - c = c\eta\lambda$, the weights sum to $s_j := c\eta \sum_{i=0}^{j-1} c^i = (1 - c^j)/\lambda > 0$. Because $g(v_i) > 0$, we have $u_j > U_n$ for all $1 \le j \le m$; in particular $U_{n+1} > U_n \ge U_0 = \operatorname{logit}(p_0)$ by induction on $n$. Consequently $3g(u_i) - 1 \ge d$ for every $0 \le i < m$ in every block, so $v_j \le V_n - d s_j \le V_n$, and hence $u_j - U_n \le 3 s_j e^{V_n}$.

Setting $s := s_m > 0$, the block endpoints satisfy
$$
V_n \le V_0 - n d s, \qquad 0 < U_{n+1} - U_n \le 3 s e^{V_0 - n d s}.
$$
Geometric summation implies $U_n \uparrow U_\infty \in (U_0, \infty)$ and $V_n \to -\infty$. The within-block inequalities $v_j \le V_n$ and $0 \le u_j - U_n \le 3s e^{V_n}$ then force $v_j \to -\infty$ and $u_j - U_n \to 0$ uniformly over $0 \le j \le m$ as $n \to \infty$, proving convergence of the full last-iterate sequence to $(g(U_\infty), 0)$ with $g(U_\infty) > p_0$. $\square$

---

## 4. Continuous-Time Flows (Periodic and Continuous Resets)

Scaling physical time by $t = k\eta$ and letting $\eta \downarrow 0$ with fixed anchor $(U, V)$ yields the globally Lipschitz vector field
$$
\dot u = 3g(v) - \lambda(u - U), \qquad \dot v = -(3g(u) - 1) - \lambda(v - V).
\tag{2}
$$

**Theorem 3 (Continuous-time flow with periodic resets).** *For every $\lambda > 0$, reset interval $T > 0$, and interior start with $p_0 \in (1/3, 1)$, $q_0 \in (0, 1)$, the trajectory of (2) with resets $(U_{n+1}, V_{n+1}) = (u(T), v(T))$ converges to $(g(U_\infty), 0) \in \mathcal{N}(A)$ with $g(U_\infty) > p_0$.*

*Proof.* By variation of constants within block $n$ from $(U_n, V_n)$,
$$
u(t) - U_n = 3 \int_0^t e^{-\lambda(t - r)} g(v(r))\,dr, \qquad
v(t) - V_n = -\int_0^t e^{-\lambda(t - r)}\bigl(3g(u(r)) - 1\bigr)\,dr.
$$
Writing $s(t) := (1 - e^{-\lambda t})/\lambda$, the first identity gives $u(t) > U_n$ for $t > 0$, so $U_{n+1} > U_n \ge U_0$ by induction, $3g(u(r)) - 1 \ge d$ throughout every block, $v(t) \le V_n - d s(t) \le V_n$, and $0 \le u(t) - U_n \le 3 s(t) e^{V_n}$. With $s := s(T) > 0$, the exact same geometric bounds $V_n \le V_0 - nds$ and $0 < U_{n+1} - U_n \le 3s e^{V_0 - nds}$ yield convergence of the entire trajectory to $(g(U_\infty), 0)$ with $g(U_\infty) > p_0$. $\square$

**Theorem 4 (Continuously reset flow).** *Setting $(U, V) = (u, v)$ continuously (equivalently $\lambda = 0$ in (2)) gives $\dot u = 3g(v)$, $\dot v = 1 - 3g(u)$. For every $p_0 \in (1/3, 1)$ and $q_0 \in (0, 1)$, $(g(u(t)), g(v(t))) \to (g(u_\infty), 0)$ with $g(u_\infty) > p_0$. From uniform initialization $(0, 0)$, $p_\infty = g(u_\infty)$ is the unique root in $(1/2, 1)$ of*
$$
p(1 - p)^2 = \frac{1}{64} \qquad (p_\infty \approx 0.8656496057).
$$

*Proof.* Along the flow, $\dot u = 3g(v) > 0$ implies $u(t) > u_0$, hence $\dot v \le -d$, $v(t) \le v_0 - dt$, and $\dot u \le 3 e^{v_0 - dt}$, which is integrable on $[0, \infty)$; thus $u(t) \uparrow u_\infty \in (u_0, \infty)$ and $v(t) \to -\infty$. Define $F(u) := -u + 3\log(1 + e^u)$ and $J(v) := \log(1 + e^v)$. Since $F'(u) = 3g(u) - 1$ and $J'(v) = g(v)$,
$$
\frac{d}{dt}\bigl(F(u) + 3J(v)\bigr) = (3g(u) - 1)(3g(v)) + 3g(v)(1 - 3g(u)) = 0.
$$
At $(u_0, v_0) = (0, 0)$, $F(0) + 3J(0) = 6\log 2$. As $t \to \infty$, $J(v(t)) \to 0$, so $e^{F(u_\infty)} = 64$. Substituting $e^u = p/(1 - p)$ gives $e^{F(u)} = \bigl(p(1 - p)^2\bigr)^{-1}$, hence $p_\infty(1 - p_\infty)^2 = 1/64$. Because $p \mapsto p(1 - p)^2$ is strictly decreasing on $(1/3, 1)$, this equation has a unique root on $(1/2, 1)$, namely $p_\infty \approx 0.8656496057$. $\square$

---

## 5. Structural Mechanism: Tangent Orthogonality and Unused Columns

Why did Leal's 180-game random ensemble in arXiv:2606.28308 observe exact agreement with $\sigma^{\mathrm{ME}}$? Consider a matrix game $A$ whose row Nash face has a zero-sum tangent vector $z$ ($\sum_i z_i = 0$) along which first-order I-projection optimality requires $z^\top \log(x/\rho_0) = 0$.

**Proposition 5 (Tangent-orthogonality invariant and dominated-column perturbation).**
1. *If $z^\top A = 0$ and $\sum_i z_i = 0$, then the linear log-ratio functional $z^\top \log(x_t / x_0)$ is identically preserved along moving-reference discrete MMD (for all $\eta, \lambda, m$).*
2. *Consider the $3\times 3$ and $3\times 4$ games with columns $w = (2, -1, 0)^\top$, $-w$, $b = (2, 1, 1)^\top$ (and $a = (4, 3, 2)^\top$ in $3\times 4$). Both games have value $0$, Nash polytope $\mathcal{N}(A) = \{(t, 2t, 1 - 3t) : 0 \le t \le 1/3\} \times \{(1/2, 1/2, 0[, 0])\}$, and unique I-projection $t = R/(1 + 3R)$ where $R = (r_1 r_2^2 / 4)^{1/3}/r_3$ for any positive initial row policy $r$. The tangent $z = (1, 2, -3)^\top$ satisfies $z^\top w = 0$ (exact invariance on the $3\times 2$ core), whereas $z^\top b = 1 > 0$ and $z^\top a = 4 > 0$, so transient mass on the strictly dominated column(s) permanently increases $z^\top \log(x_t / r)$ across resets and drives the limit strictly away from $\Pi_{\mathcal{N}}$.*

*Proof.* In a single discrete MMD step with anchor $X_n$ and normalization constant $\gamma$, $\log x_{i+1} - X_n = c(\log x_i - X_n + \eta A y_i) + \gamma \mathbf{1}$. Taking the inner product with $z$ kills $\gamma \mathbf{1}$ since $z^\top \mathbf{1} = 0$, leaving
$$
z^\top(\log x_{i+1} - X_n) = c\, z^\top(\log x_i - X_n) + c\eta\,(z^\top A y_i).
$$
When $z^\top A = 0$, $z^\top \log x$ never changes from $z^\top \log x_0$. For the $3\times 3$ and $3\times 4$ games, $(1/2, 1/2, 0[, 0])$ achieves payoff $0$ against every row and strictly dominates $b$ and $a$ (which have strictly positive entries); row optimality requires $x^\top w = 0$, giving $x = (t, 2t, 1 - 3t)$ ($0 \le t \le 1/3$), and differentiating $D(x \| r)$ in $t$ yields $t = R/(1 + 3R)$. Since $z^\top b = 1 > 0$ and $z^\top a = 4 > 0$, every step with positive mass on $b$ or $a$ adds a strictly positive drift to $z^\top \log x$, which each reset locks into the new anchor. $\square$

---

## 6. Reproducible Numerical Verification

Run from the repository root:

```sh
.venv/bin/python -B solutions/nash_iproj_selection/explore.py
```

The script [`explore.py`](explore.py) runs **32 deterministic test cases** (three initial references across four exact-solve temperatures, three discrete MMD configurations, and two periodic-flow settings; plus the algebraic continuously reset root and four $3\times 3$ / $3\times 4$ tilted-game checks). All 32 cases pass with final $\mathrm{NashConv} < 10^{-10}$. Selected uniform-start results on the $2\times 2$ counterexample:

| Regime | Parameters | Deployed $p_\infty$ | I-projection $p$ |
|---|---|---:|---:|
| Exact anchored QRE reset | $\lambda = 1$ ($h = 1$) | `0.7632330195` | `0.5000000000` |
| Exact anchored QRE reset | $\lambda = 1/16$ ($h = 16$) | `0.5034167483` | `0.5000000000` |
| Discrete MMD, periodic reset | $\eta = 0.15,\; \lambda = 1,\; m = 7$ | `0.8534454367` | `0.5000000000` |
| Continuous flow, periodic reset | $\lambda = 1,\; T = 1$ | `0.8428003635` | `0.5000000000` |
| Continuously reset flow | $\lambda = 0$ (algebraic root) | `0.8656496057` | `0.5000000000` |

On the $3\times 3$ and $3\times 4$ tilted games from uniform initialization, the analytic I-projection has $t = 0.2179883530$, whereas discrete MMD ($\eta = 0.12, \lambda = 1, m = 25$) converges to $t = 0.2232759106$ ($3\times 3$) and $t = 0.2270167967$ ($3\times 4$).

---

## References

- M. Cuturi, *Sinkhorn Distances: Lightspeed Computation of Optimal Transport*, NeurIPS 2013.
- L. Leal, *Which Nash Equilibrium? Solver-Dependent Selection on Zero-Sum Nash Polytopes*, [arXiv:2606.28308](https://arxiv.org/abs/2606.28308), June 2026.
- R. D. McKelvey and T. R. Palfrey, *Quantal Response Equilibria for Normal Form Games*, Games and Economic Behavior 10(1):6–38, 1995.
- J. Pérolat, R. Munos, J.-B. Lespiau, S. Omidshafiei, M. Rowland, P. Ortega, N. Burch, T. Anthony, D. Balduzzi, B. De Vylder, G. Piliouras, M. Lanctot, and K. Tuyls, *From Poincaré Recurrence to Convergence in Imperfect Information Games: Finding Equilibrium via Regularization*, ICML 2021.
- J. Pérolat et al., *Mastering the game of Stratego with model-free multiagent reinforcement learning*, Science 378(6623):990–996, 2022.
- S. Sokota, R. D'Orazio, J. Z. Kolter, N. Loizou, M. Lanctot, I. Mitliagkas, N. Brown, and C. Kroer, *A Unified Approach to Reinforcement Learning, Quantal Response Equilibria, and Two-Player Zero-Sum Games*, ICLR 2023.
- T. L. Turocy, *A Dynamic Homotopy Interpretation of the Logistic Quantal Response Equilibrium Correspondence*, Games and Economic Behavior 51(2):243–263, 2005.
- J. Weed, *An explicit analysis of the entropic penalty in linear programming*, COLT 2018.
