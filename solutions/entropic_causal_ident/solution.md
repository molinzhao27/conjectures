# Refutation of the Entropic Causal Identifiability Conjecture

**Target papers:** Spencer Compton, Murat Kocaoglu, Kristjan Greenewald, and Dmitriy Katz, *Entropic Causal Inference: Identifiability and Finite Sample Results*, [arXiv:2101.03501](https://arxiv.org/abs/2101.03501) (NeurIPS 2020), Section 2, Conjecture 1 (`conj:entropic`), restating Conjecture 1 of Murat Kocaoglu, Alexandros G. Dimakis, Sriram Vishwanath, and Babak Hassibi, *Entropic Causal Inference*, AAAI 2017 ([arXiv:1611.04035](https://arxiv.org/abs/1611.04035)).  
**Companion files:** [`solutions/entropic_causal_ident/explore.py`](explore.py), [`solutions/entropic_causal_ident/proof.pdf`](proof.pdf).

---

## Verdict

All entropies and logarithms are natural (in nats). Given a structural causal model $Y = f(X, E)$ with $X \in [n]$, $Y \in [n]$, $E \in [m]$ ($X \perp\!\!\!\perp E$), let $B_m$ be the minimum Shannon entropy $H(\tilde E)$ over all backward models $X = g(Y, \tilde E)$ with $\tilde E \perp\!\!\!\perp Y$ and $|\mathrm{supp}(\tilde E)| \le m$ (setting $B_m := +\infty$ if none exists). Conjecture 1 (`conj:entropic`) asserts that when $p(X) \sim \mathrm{Unif}(\Delta_n)$, $p(E) \sim \mathrm{Unif}(\{q \in \Delta_m : H(q) \le \log n + c\})$ for fixed $c = O(1)$, and $f : [n] \times [m] \to [n]$ is uniform and independent of $(p, q)$, the total-entropy gap
$$
G_m := H(Y) + B_m - H(X) - H(E)
$$
is strictly positive with probability tending to $1$ as $n \to \infty$.

**Conjecture 1 (`conj:entropic`) is FALSE**, both in the printed form with backward alphabet $[m]$ and in the unrestricted-alphabet Shannon formulation of Kocaoglu et al. (AAAI 2017):

| Forward noise alphabet $m$ | Entropy cap $\log n + c$ | Asymptotic gap $G_m$ | Result |
|---|---|---|---|
| $m = n^2$ | $\log n + 2$ ($c = 2$) | $G_m \le -\gamma + o_P(1)$ ($\gamma = 0.57721566\dots$) | Refutes `conj:entropic` for *every* table $f$ (Theorem 3) |
| $m = \lceil e^{n^5} \rceil$ | $\log n$ ($c = 0$) | $G_m \to -D(0) = -0.8515044932\dots$ in prob. | Exact four-entropy asymptotics (Theorem 6) |
| $m = n^{1024}$ | $\log n$ ($c = 0$) | $G_m < -0.073$ with prob. $\to 1$ | Polynomial counterexample at zero offset (Theorem 7) |

- **Scope clarification:** Conjecture 1 explicitly compares the forward score $H(X) + H(E)$ (using the **sampled** exogenous variable $E$) against $H(Y) + B_m$ (the minimum over backward SCMs $\tilde E \perp\!\!\!\perp Y$). Comparing the *minimized* exogenous entropies in both directions ($\min H(E') + H(X)$ versus $\min H(\tilde E) + H(Y)$) is a separate question. Likewise, the appendix simulations in Compton et al. use an adaptive Dirichlet concentration parameter $\alpha \neq 1$ (which alters the density on the accepted sublevel set) and compare greedy fits in both directions (where a forward greedy fit can have lower entropy than the sampled $E$).
- **Other alphabet regimes (Section 5):** For fixed $m$ and for $m = \lfloor n^\alpha \rfloor$ with $0 < \alpha < 2$, the alphabet cap makes $B_m = +\infty$ with high probability due to an explicit rational-independence support obstruction (Proposition 8). For fixed powers $\alpha \ge 2$, $\log n - d - o_P(1) \le B_m \le \log n + U + o_P(1)$, while the exact $O(1)$ additive constant of $B_m$ remains undetermined here.

---

## 1. Exact Statement and Coupling Reformulation

In Section 2 of Compton et al. (`arXiv:2101.03501`), $p(X)$ is uniform on the simplex $\Delta_n$ (equivalently $\mathrm{Dirichlet}(1,\dots,1)$), $p(E)$ is uniform with respect to normalized simplex volume on $\{q \in \Delta_m : H(q) \le \log n + c\}$ for a fixed additive constant $c = O(1)$, and $f : [n] \times [m] \to [n]$ is uniform and independent of $(p, q)$. No growth law is imposed on $m$, and $g : [n] \times [m] \to [n]$. Kocaoglu et al. (AAAI 2017, Theorem 1 and Conjecture 1) proved Rényi-0 (cardinality) identifiability under a generic-mechanism hypothesis and posed the Shannon-entropy conjecture with an unrestricted backward alphabet; every counterexample below uses at most $m$ backward atoms and therefore refutes both formulations.

For each $y$ with $p(y) > 0$, write $P_y := p(X \mid Y = y)$. Any backward SCM $X = g(Y, \tilde E)$ with $\tilde E \perp\!\!\!\perp Y$ induces a coupling of the family $\{P_y\}_{y \in \mathrm{supp}(Y)}$ whose atoms are the response vectors $(g(y, \tilde e))_y$ weighted by $p(\tilde e)$; conversely, any coupling on at most $m$ atoms yields such a backward SCM, and merging identical response vectors only decreases entropy. Hence $B_m$ equals the minimum coupling entropy of $\{P_y\}$ over couplings supported on at most $m$ atoms, and $B_m \ge \max_{y \in \mathrm{supp}(Y)} H(P_y)$. Moreover, both $H(X) + H(E)$ and $H(Y) + B_m$ are at least $H(X, Y)$ by independence and deterministic processing.

---

## 2. Finite Greedy Coupling Bound and the Quadratic Counterexample

**Lemma 1 (Finite greedy coupling bound).** *For any collection of distributions $\{P_y\}_{y \in \mathrm{supp}(Y)}$ on $[n]$ ($|\mathrm{supp}(Y)| \le n$), there exists a coupling supported on at most $1 + \sum_y (|\mathrm{supp}(P_y)| - 1) \le 1 + n(n-1)$ atoms whose Shannon entropy is strictly less than $\log n + 1$.*

*Proof.* Maintain residual nonnegative vectors $R_y$ initialized to $P_y$, sharing a common total mass $R \in (0, 1]$. At each step, choose $x_y^* \in \arg\max_x R_y(x)$ for each $y$, emit a shared atom on response vector $(x_y^*)_y$ with weight $w := \min_y R_y(x_y^*)$, and subtract $w$ from $R_y(x_y^*)$ in every row $y$. Since each residual row has at most $n$ nonzero entries summing to $R$, we have $w \ge R/n$. Every nonfinal step zeroes at least one entry across all rows, and the final step zeroes the last entry in every row simultaneously, so at most $1 + \sum_y (|\mathrm{supp}(P_y)| - 1) \le 1 + n(n-1)$ atoms are emitted.

Let $J \in \{1, 2, \dots\}$ be the step index sampled with the emitted weights. After $k$ steps, $\Pr(J > k) \le (1 - 1/n)^k$, so $\mathbb{E}[J] = \sum_{k \ge 0} \Pr(J > k) \le n$. Comparing $J$ to $Q \sim \mathrm{Geom}(1/n)$ via $D_{\mathrm{KL}}(P_J \| Q) \ge 0$ gives
$$
H(J) \le \log n - (\mathbb{E}[J] - 1)\log\!\left(1 - \frac{1}{n}\right) \le \log n - (n-1)\log\!\left(1 - \frac{1}{n}\right) < \log n + 1,
$$
where the final inequality holds for $n = 1$ ($0 < 1$) and for $n \ge 2$ since $(n-1)\log(1 + \frac{1}{n-1}) < 1$. $\square$

**Lemma 2 (Volume concentration near the entropy cap).** *Set $t := \log n + c$ (fixed $c \in \mathbb{R}$) and $L := \log m$. Assume $m / (n L^3) \to \infty$ (which holds for $m = \lfloor n^\alpha \rfloor$ with any fixed $\alpha > 1$ and for $m = \lceil e^{n^5} \rceil$). Then for every fixed $\varepsilon > 0$,*
$$
\Pr\bigl(H(q) \le t - \varepsilon \mid H(q) \le t\bigr) \longrightarrow 0.
$$

*Proof.* On $\{q \in \Delta_m : H(q) \le t\}$, we have $t \ge \sum_j q_j \log(1/q_j) \ge \log(1/\max_j q_j)$, so $\max_j q_j \ge e^{-t} = e^{-c}/n$. Partition $\{q \in \Delta_m : H(q) \le t - \varepsilon\}$ (ignoring measure-zero ties) into $m$ pieces according to $i = \arg\max_j q_j$. On piece $i$, dilate about the vertex $e_i$ via
$$
T_i(q) := e_i + (1 + \delta)(q - e_i), \qquad \delta := \frac{1}{n L^2}.
$$
Its $i$-th coordinate is $q_i - \delta(1 - q_i) \ge e^{-c}/n - 1/(n L^2) \ge 0$ for large $n$, so $T_i(q) \in \Delta_m$. The total variation distance is $\|T_i(q) - q\|_{\mathrm{TV}} = \delta(1 - q_i) \le \delta$. Coupling a draw from $q$ with a draw from $T_i(q)$ and revealing their disagreement indicator gives $|H(T_i(q)) - H(q)| \le h(\delta) + \delta \log m = o(1)$, where $h(x) := -x\log x - (1-x)\log(1-x)$. Hence for large $n$, each $T_i$ injects piece $i$ into $\{H(q) \le t\}$ with $(m-1)$-dimensional Jacobian $(1+\delta)^{m-1}$, and the $m$ images overlap at most $m$ times. Thus
$$
\Pr\bigl(H(q) \le t - \varepsilon \mid H(q) \le t\bigr) \le m(1 + \delta)^{-(m-1)} \longrightarrow 0,
$$
since $(m-1)\delta = \Omega(m/(n L^2)) \gg \log m$. $\square$

**Theorem 3 (Quadratic-alphabet refutation).** *Take $m = n^2$ and any fixed offset $c > 2 - \gamma$ (in particular $c = 2$), where $\gamma = 0.57721566\dots$ is the Euler–Mascheroni constant. Then for every deterministic (hence also uniform random) forward table $f : [n] \times [n^2] \to [n]$,*
$$
G_m \le 2 - \gamma - c + o_P(1).
$$
*In particular, at $c = 2$, $G_m \le -\gamma + o_P(1) < 0$ with probability tending to $1$, refuting Conjecture 1 (`conj:entropic`).*

*Proof.* Write $p_x = A_x / \sum_{j=1}^n A_j$ for i.i.d. $\mathrm{Exp}(1)$ variables $A_1, \dots, A_n$. By the law of large numbers for $A$ and $A \log A$, $H(X) = \log n - d + o_P(1)$, where $d := \mathbb{E}[A \log A] = 1 - \gamma$. By Lemma 2, $H(E) = \log n + c - o_P(1)$. Because $1 + n(n-1) \le n^2 = m$, Lemma 1 gives $B_m < \log n + 1$, and trivially $H(Y) \le \log n$. Combining these four bounds gives $G_m \le (\log n) + (\log n + 1) - (\log n - d) - (\log n + c) + o_P(1) = 1 + d - c + o_P(1) = 2 - \gamma - c + o_P(1)$. $\square$

---

## 3. Structure of the Conditioned Noise and Random-Map Output Entropy

**Lemma 4 (Single dominant atom and uniform tail).** *Let $c \in \mathbb{R}$ be fixed, $L := \log m$, and $b := L^3 / m$. Assume either $m = \lfloor n^\alpha \rfloor$ for fixed $\alpha > 1$ or $m = \lceil e^{n^5} \rceil$. Conditioned on $H(q) \le \log n + c$:*
1. *With probability tending to $1$, the second-largest coordinate of $q$ is at most $b$.*
2. *Let $1 - s := \max_j q_j$ and $i := \arg\max_j q_j$. Then $s \to 1/\alpha$ in probability when $m = \lfloor n^\alpha \rfloor$, and $s \sim (\log n)/n^5$ in probability when $m = \lceil e^{n^5} \rceil$.*
3. *Let $F(x) := f(x, i)$ (a uniform random map independent of $p, q$) and $\epsilon_n := (\log n)^{-2}$. With probability tending to $1$, uniformly over all $x, y \in [n]$,*
   $$
   p(y \mid x) = (1 - s)\,\mathbf{1}_{F(x) = y} + \frac{s}{n}\bigl(1 + O(\epsilon_n)\bigr). \tag{1}
   $$

*Proof.* (1) Partition the bad event $\{\text{second-largest coordinate} \ge b\}$ by the indices $(i, j)$ of the two largest coordinates ($q_i \ge q_j = v \ge b$). Transfer $v/2$ from coordinate $j$ to coordinate $i$: this linear map has simplex Jacobian $1/2$ and reduces entropy by at least $v\bigl(\frac{3}{2}\log\frac{3}{2} + \frac{1}{2}\log\frac{1}{2}\bigr) > v/4$ (by differentiation in $q_i$, the minimum entropy drop occurs at $q_i = v$). Next dilate about $e_i$ by $1 + \eta$ with $\eta := b / (100 L)$; for large $n$ the image remains in $\Delta_m$ and its entropy increases by at most $h(\eta) + \eta L < b/4 \le v/4$. Hence each bad piece maps injectively into $\{H(q) \le \log n + c\}$ with Jacobian $\frac{1}{2}(1+\eta)^{m-1}$, giving bad probability at most $2m^2 (1 + \eta)^{-(m-1)} \to 0$ since $m \eta = L^2 / 100 \gg \log m$.

(2) On the good event $\max_{j \neq i} q_j \le b$, we have $s \log(1/b) \le -\sum_{j \neq i} q_j \log q_j \le H(q) \le h(s) + s \log(m-1)$. Since $\log(1/b) = L - 3\log L = (1 - o(1))L$ and $H(q) = \log n + c - o_P(1)$ by Lemma 2, solving for $s$ gives $s = (\log n + c + o_P(1))/L$, yielding both limits.

(3) Conditional on $q$, for each $(x, y)$ the tail contribution $\sum_{e \neq i} q_e \mathbf{1}_{f(x,e)=y}$ is a sum of independent variables in $[0, b]$ with mean $s/n$ and variance at most $bs/n$. Bernstein's inequality and a union bound over all $n^2$ pairs $(x, y)$ give failure probability at most $2n^2 \exp(-C \epsilon_n^2 s / (nb))$ on the good event. Since $\epsilon_n^2 s / (nb) \sim (\log n)^{-4} m s / (n L^3) \gg \log n$ in both regimes, (1) holds with probability tending to $1$. $\square$

**Proposition 5 (Output entropy of the random-map mixture).** *Let $Z := \sum_{j=1}^K A_j$ where $K \sim \mathrm{Pois}(1)$ is independent of i.i.d. $A_j \sim \mathrm{Exp}(1)$ (with $0 \log 0 := 0$), and define*
$$
D(u) := \mathbb{E}\bigl[(u + (1-u)Z)\log(u + (1-u)Z)\bigr] \qquad (u \in [0, 1]).
$$
*Under the conditions of Lemma 4 with $s \to u \in [0, 1)$ in probability, $H(Y) = \log n - D(u) + o_P(1)$. Moreover, the empty-bin fraction of $F$ tends to $e^{-1}$ in probability, the maximum occupancy of $F$ is $K_* = O_P(\log n)$, and*
$$
D(0) = e^{-1}\sum_{k=1}^\infty \frac{k\,\psi(k+1)}{k!} = 0.8515044932\ldots,
\qquad \psi(k+1) = \sum_{j=1}^k \frac{1}{j} - \gamma.
$$

*Proof.* Replacing the uniform-tail error $1 + O(\epsilon_n)$ in (1) by $1$ perturbs the output distribution $p(Y)$ in total variation by $O(\epsilon_n)$, changing $H(Y)$ by $O(h(\epsilon_n) + \epsilon_n \log n) = O((\log n)^{-1}) = o(1)$. For the idealized mixture $p_0(y) = \frac{s}{n} + (1-s)\frac{W_y}{S_n}$ where $W_y := \sum_{x: F(x)=y} A_x$ and $S_n := \sum_{x=1}^n A_x$, note that $S_n / n \to 1$ in probability. For any single bin $y$, $W_y \Rightarrow Z$ by the binomial-to-Poisson limit; for two distinct bins $(y_1, y_2)$, $(W_{y_1}, W_{y_2}) \Rightarrow (Z_1, Z_2)$ (independent copies) by multinomial convergence and independent marks. Every fixed moment of $W_y$ is uniformly bounded (conditioning on the $\mathrm{Bin}(n, 1/n)$ bin count), so the empirical average $n^{-1}\sum_{y=1}^n \phi_u(W_y)$ of $\phi_u(z) := (u + (1-u)z)\log(u + (1-u)z)$ converges in probability to $D(u)$ by the two-bin variance calculation and uniform integrability. Absorbing $S_n / n \to 1$ and $s \to u$ by truncating $W_y$ at a large constant and using the uniform moment bounds gives $H(Y) = \log n - D(u) + o_P(1)$. The same two-bin calculation shows the empty-bin fraction tends to $e^{-1}$, and a binomial tail union bound yields $K_* = O_P(\log n)$. Finally, conditioning on $K = k \ge 1$ (so $Z \sim \mathrm{Gamma}(k, 1)$) gives $\mathbb{E}[Z \log Z \mid K = k] = k\,\psi(k+1)$, yielding the series for $D(0)$. $\square$

---

## 4. Exact Zero-Offset Counterexamples

### 4.1 Super-Exponential Alphabet: Exact Four-Entropy Asymptotics ($m = \lceil e^{n^5} \rceil$)

**Theorem 6 (Exact gap at $m = \lceil e^{n^5} \rceil$).** *Let $m = \lceil e^{n^5} \rceil$ and fix any $c \in \mathbb{R}$. Then up to $o_P(1)$,*
$$
H(X) = \log n - d, \qquad
H(E) = \log n + c, \qquad
H(Y) = \log n - D(0), \qquad
B_m = \log n - d.
$$
*Consequently, $G_m \to -D(0) - c$ in probability. At zero offset ($c = 0$), $G_m \to -0.8515044932\dots$ in probability, and reversals occur for every fixed $c > -D(0)$.*

*Proof.* Lemma 2, Lemma 4 ($s \sim (\log n)/n^5 \to 0$), and Proposition 5 give $H(X)$, $H(E)$, and $H(Y)$. For $B_m$, note that on every empty $F$-bin $y$, (1) gives $P_y(x) = p_x(1 + O(\epsilon_n))$ uniformly in $x$. Since empty bins exist with high probability, $B_m \ge \max_y H(P_y) \ge H(X) - o_P(1)$.

For the matching upper bound, observe that $\min_x p_x \ge n^{-3}$ with high probability (by a union bound on the $\mathrm{Beta}(1, n-1)$ marginal). At every occupied $F$-bin $y$, the preimages contribute at least $(1-s)n^{-3}$ to $p(y)$ while the entire tail contributes $O(s/n)$, so the posterior $P_y$ puts all but $a_y' = O(sn^2)$ mass on its preimages; let $a_* := \max_{y \text{ occ.}} a_y' = O_P(sn^2)$. Define $v_x := \min_{y \text{ empty}} P_y(x)$ for each $x \in [n]$; uniformly, $v_x / p_x = 1 + O(\epsilon_n)$ and $\sum_x v_x = 1 - O(\epsilon_n)$.

Process $x = 1, \dots, n$ sequentially: in each empty bin allocate weight $v_x$ to $x$, and in each occupied bin allocate $v_x$ to any preimage whose residual mass is at least $v_x$. Whenever the common remaining mass $R$ satisfies $R > a_* + K_* \max_x v_x$, the residual mass on the at most $K_*$ preimages of any occupied bin exceeds $K_* \max_x v_x$, so by pigeonhole at least one preimage has residual mass at least $v_x$. Stop when $R \le a_* + K_* \max_x v_x$ or the list ends. The remaining mass satisfies
$$
R = O(\epsilon_n + sn^2 + K_* \max_x p_x) = o_P(1/\log n),
$$
since $\max_x p_x = O_P((\log n)/n)$ and $K_* = O_P(\log n)$. Couple the normalized residual rows via Lemma 1 using at most $1 + n(n-1)$ additional atoms (total atoms $\le n + 1 + n(n-1) \le m$). The initial atoms contribute at most $H(X) + o_P(1)$ to the entropy (since $v_x / p_x = 1 + O(\epsilon_n)$ uniformly), and the residual coupling contributes at most $R(\log n + 1 - \log R) = o_P(1)$. Hence $B_m \le H(X) + o_P(1) = \log n - d + o_P(1)$. $\square$

### 4.2 Polynomial Alphabet at Exact Zero Offset ($m = n^{1024}$, $c = 0$)

**Theorem 7 (Polynomial-alphabet zero-offset refutation).** *Take $m = n^{1024}$ and cap $H(E) \le \log n$ ($c = 0$). Then*
$$
B_m \le \log n + U + o_P(1), \qquad U = 0.3466016619\ldots,
$$
*and $G_m \le U + d - D(1/1024) + o_P(1) < -0.07354 + o_P(1) < -0.073$ with probability tending to $1$.*

*Proof.* Set $u := 1/1024$. By Lemma 2, Lemma 4, and Proposition 5, $H(X) = \log n - d + o_P(1)$, $H(E) = \log n - o_P(1)$, and $H(Y) = \log n - D(u) + o_P(1)$.

At atom size $w := z/n$ ($z > 0$ fixed), row $P_y$ can pack $\sum_{x=1}^n \lfloor P_y(x)/w \rfloor$ atoms of size $w$, leaving unusable remainder
$$
R_y(z) := 1 - w\sum_{x=1}^n \left\lfloor \frac{P_y(x)}{w} \right\rfloor.
$$
We claim that $\max_{y \in [n]} R_y(z) \to r(z) := 1 - z/(e^z - 1)$ in probability for every fixed $z > 0$. Write $a_y := (s/n)/p(y) \in [0, 1 + O(\epsilon_n)]$. Outside its at most $K_* = O_P(\log n)$ preimages, row $P_y$ satisfies $P_y(x) = a_y p_x(1 + O(\epsilon_n))$ uniformly. For ideal diffuse entries $a p_x$, the limiting remainder is
$$
r_a(z) := a - z\,\mathbb{E}\!\left\lfloor \frac{a A}{z} \right\rfloor = a - \frac{z}{e^{z/a} - 1} \quad (a > 0), \qquad r_0(z) := 0.
$$
By the law of large numbers on a finite grid of $a \in [0, 2]$, monotonicity in $a$, and continuity of $a \mapsto \mathbb{E}\lfloor aA/z \rfloor$, $n^{-1}\sum_{x=1}^n \lfloor a n p_x / z \rfloor \to \mathbb{E}\lfloor aA/z \rfloor$ uniformly in $a \in [0, 2]$; bracketing $a$ absorbs both $S_n / n \to 1$ and the uniform $(1 + O(\epsilon_n))$ error. Each preimage alters the fractional remainder by at most $w = z/n$, so all preimages together alter $R_y(z)$ by at most $K_* z / n = o_P(1)$. Hence $R_y(z) = r_{a_y}(z) + o_P(1)$ uniformly in $y$. Since $\frac{\partial}{\partial a} r_a(z) = 1 - \bigl(\frac{v}{2\sinh(v/2)}\bigr)^2 \ge 0$ (where $v = z/a$), $r_a(z)$ increases in $a$; empty bins have $a_y = 1 + O(\epsilon_n)$ and exist with high probability, proving $\max_y R_y(z) \to r(z)$.

Now run a dyadic-scale allocation over $z_j := 2^j$ in descending order $j = 5, 4, \dots, -12$. At scale $j$, emit as many shared atoms of size $2^j / n$ as every row can simultaneously accommodate, placing them in any entries with sufficient residual mass. Because every earlier allocation was an integer multiple of $2^j / n$, the common residual mass remaining after scale $j$ is **exactly** $\max_y R_y(2^j)$, independent of prior placement choices. Couple the final normalized residual rows via Lemma 1. Total emitted atoms are at most $4096 n + 1 + n(n-1) \le m$. Setting $r_j := r(2^j)$ and $r_6 := 1$, subtracting $\log n$ yields $B_m \le \log n + U + o_P(1)$, where
$$
U := -\sum_{j=-12}^{5} (r_{j+1} - r_j)\log(2^j) + r_{-12}(1 - \log r_{-12}) = 0.3466016619\ldots
$$

Finally, for every $z \ge 0$, $(u + (1-u)z)\log(u + (1-u)z) \ge u\log u + (1-u)z\log((1-u)z)$; taking expectations (with $\mathbb{E}[Z] = 1$) gives
$$
D(u) \ge (1 - u)D(0) - h(u) > 0.84292.
$$
Therefore $G_m \le U + d - D(u) + o_P(1) < -0.07354 + o_P(1) < -0.073$ with probability tending to $1$ (certified in `explore.py` via 50-digit interval arithmetic using the first 30 strictly positive terms of the series for $D(0)$). $\square$

---

## 5. Complete Phase Diagram Across Alphabet Regimes

At zero offset ($c = 0$), the backward cardinality cap $[m]$ produces distinct regimes:

**Proposition 8 (Support obstruction for small $m$).** *Fix a forward noise vector $q$ and table $f$, set $v_{yx} := p(y \mid x)$, $L_y(A) := \sum_x v_{yx} A_x$, and $s_y := |\{x : v_{yx} > 0\}|$. For any subset $S \subseteq [n]$ of output bins whose linear forms $\{L_y\}_{y \in S}$ are pairwise nonproportional, any backward coupling requires at least $1 + \sum_{y \in S}(s_y - 1)$ atoms almost surely over $A_1, \dots, A_n \stackrel{\text{i.i.d.}}{\sim} \mathrm{Exp}(1)$.*

*Proof.* We show that $1$ together with all positive row entries $v_{yx} A_x / L_y(A)$ ($y \in S$, omitting one positive entry per row) are linearly independent over $\mathbb{Q}$ almost surely. Given a nontrivial rational linear relation, clear denominators by multiplying by $\prod_{y \in S} L_y(A)$ in $\mathbb{R}[A_1, \dots, A_n]$ and reduce modulo each distinct irreducible linear factor $L_y(A)$. Because the factors are pairwise relatively prime, the row-$y$ numerator must be divisible by $L_y(A)$; since one positive entry of row $y$ is absent, that quotient must be zero, killing every row coefficient and finally the constant term. Thus every nontrivial rational relation vanishes on a Lebesgue-null set, and there are countably many rational relations. Since $1$ and every posterior entry are subset sums of the backward atom masses, at least $1 + \sum_{y \in S}(s_y - 1)$ atoms are required. $\square$

Combining Proposition 8 with Sections 2–4 gives the full picture:
- **Fixed $m$:** $H(E)$ retains its $\mathrm{Dirichlet}(1,\dots,1)$ entropy law (mean $\sum_{j=1}^m 1/j - 1$, non-concentrated for $m > 1$), $H(X) = \log n - d + o_P(1)$, and conditional on $q$, $H(Y) = \log n - \mathbb{E}[Z_q \log Z_q \mid q] + o_P(1)$ where $Z_q = \sum_{e=1}^m q_e Z_e$ with i.i.d. copies $Z_e$ of $Z$. Since the first noise column alone has a bin with $> m$ preimages with high probability (fraction $\to \Pr(\mathrm{Pois}(1) > m) > 0$), $B_m = +\infty$ with high probability. (With an unrestricted backward alphabet, Compton et al.'s $\Omega(\log\log n)$ lower bound applies.)
- **$m = \lfloor n^\alpha \rfloor$, $0 < \alpha \le 1$:** The cap is inactive, $H(E) = \log m - d + o_P(1)$, and $H(Y) = \log n + o_P(1)$ (expected $\chi^2$ distance to uniform is $(n-1)\mathbb{E}[\sum p_x^2]\mathbb{E}[\sum q_e^2] = 4(n-1)/[(n+1)(m+1)] \to 0$). Two specified bins have $\mathrm{Bin}(n, 1 - (1 - 1/n)^m)$ support sizes summing to $> (1+\delta)m$ and distinct supports with high probability, so Proposition 8 yields $B_m = +\infty$ with high probability.
- **$m = \lfloor n^\alpha \rfloor$, $1 < \alpha < 2$:** Lemmas 2, 4 and Proposition 5 give $H(E) = \log n + o_P(1)$ and $H(Y) = \log n - D(1/\alpha) + o_P(1)$. All $n^2$ cells are hit with high probability, and for every pair of output bins some noise column hits both on distinct inputs (giving a $\pm 1$ minor at that vertex, hence nonzero almost everywhere under the conditioned law). Proposition 8 applied to all $n$ bins requires $1 + n(n-1) > m$ atoms, so $B_m = +\infty$ with high probability.
- **$m = \lfloor n^\alpha \rfloor$, $\alpha \ge 2$ (fixed offset $c$):** $H(E) = \log n + c + o_P(1)$, $H(Y) = \log n - D(1/\alpha) + o_P(1)$, and $\log n - d - o_P(1) \le B_m \le \log n + U + o_P(1)$ (if the raw dyadic coupling exceeds $m$ atoms, minimizing concave entropy over the coupling polytope face supported on its response vectors attains a vertex of support at most $1 + n(n-1) \le m$). The exact $O(1)$ additive constant of $B_m$ for fixed $\alpha \ge 2$ remains undetermined here.

---

## 6. Reproducible Verification and Attribution

Run the verification script from the project root:

```sh
.venv/bin/python -B solutions/entropic_causal_ident/explore.py
```

With seed `210103501`, the script runs direct rejection-sampled trials from $\mathrm{Dirichlet}(1,\dots,1)$ under the exact cap $H(E) \le \log n + c$, checks posterior reconstruction, majorization lower bounds (MEC lower bound), greedy upper bounds, exact rational optimization for binary ($n = 2$, 400 trials: 337 feasible, 226 strict reversals) and ternary ($n = 3$, 48 trials: 12 feasible, certified by 100-digit interval arithmetic) models, 50-digit interval certification of $U + d - D(1/1024) < -0.07354$, and 128 surrogate trials exercising the single-map-plus-uniform-tail coupling at $n = 8, 16, 32, 64$ (mean $H_{\text{greedy}} - H(X)$ equal to $0.04954, 0.01329, 0.00485, 0.00148$ nats). Representative direct-trial means:

| $n$ | $m$ | $c$ | Trials | Mean $H(X)$ | Mean $H(E)$ | Mean $H(Y)$ | Mean MEC lower | Mean greedy | Feasible greedy reversals |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 2 | 2 | 64 | 2.3860 | 0.4850 | 2.2342 | 1.2365 | 1.5373 | 0 |
| 32 | 32 | 0 | 32 | 3.0571 | 3.0736 | 3.4146 | 2.7420 | 3.1643 | 0 |
| 4 | 16 | 2 | 64 | 1.0870 | 2.3639 | 1.3141 | 1.1935 | 1.5510 | 64 / 64 |
| 8 | 64 | 2 | 64 | 1.6981 | 3.7392 | 2.0531 | 1.8121 | 2.2722 | 64 / 64 |
| 10 | 100 | 2 | 64 | 1.9250 | 4.1928 | 2.2861 | 2.0348 | 2.5111 | 64 / 64 |
| 12 | 144 | 2 | 64 | 2.0932 | 4.4581 | 2.4710 | 2.1994 | 2.6754 | 64 / 64 |

All 256 direct quadratic trials at $c = 2$ ($n = 4, 8, 10, 12$; 901 proposals for 64 accepted draws at $(12, 144)$) yielded feasible reversing witnesses.

**References.**
- S. Compton, M. Kocaoglu, K. Greenewald, and D. Katz, *Entropic Causal Inference: Identifiability and Finite Sample Results*, NeurIPS 2020, [arXiv:2101.03501](https://arxiv.org/abs/2101.03501).
- M. Kocaoglu, A. G. Dimakis, S. Vishwanath, and B. Hassibi, *Entropic Causal Inference*, AAAI 2017, [arXiv:1611.04035](https://arxiv.org/abs/1611.04035).
