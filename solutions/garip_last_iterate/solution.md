# Global Last-Iterate Convergence of Cumulative-Average Matrix GARIP

**Target paper:** Can Savcı, *GARIP: A Running-Average Moving Reference for Last-Iterate Self-Play in Two-Player Zero-Sum Games*, [arXiv:2606.22688](https://arxiv.org/abs/2606.22688) (June 2026), Section 4, Conjecture `conj:li` (Conjecture 1).  
**Companion files:** [`solutions/garip_last_iterate/explore.py`](explore.py), [`solutions/garip_last_iterate/proof.pdf`](proof.pdf).

---

## Verdict

- **Small-$(\beta, \eta)$ global last-iterate convergence (`conj:li`, primary claim): PROVED.** For every finite two-player zero-sum matrix game (including games with non-unique equilibria or boundary-only equilibria), there is an explicit rectangle of positive constant stepsizes $(\beta, \eta)$—requiring no relative scaling between $\beta$ and $\eta$—on which both the GARIP last iterate $z_t$ and its cumulative average $a_t$ converge to the **same Nash equilibrium** from every interior starting point $z_0 > 0$. The limiting equilibrium may depend on $z_0$.
- **Asymptotic premature-consensus plateau (`conj:li`, secondary regime claim): REFUTED asymptotically.** For **every** $\beta \in (0, 1)$ and $\eta > 0$, in any game admitting a fully mixed Nash equilibrium, asymptotic consensus ($\|z_t - a_t\| \to 0$) is incompatible with exploitability remaining bounded away from zero:
  $$
  \|z_t - a_t\| \to 0 \quad \Longrightarrow \quad \liminf_{t \to \infty} \operatorname{gap}(z_t) = 0.
  $$
  Likewise, at any $\beta \in (0, 1)$ and $\eta > 0$, any converged limit $z_t \to u$ from an interior start must be a Nash equilibrium. Thus the large-$\beta$ plateaus observed empirically in arXiv:2606.22688 are finite-horizon transients rather than asymptotic non-Nash consensus states.
- **Open scope:** Sharp stepsize thresholds, quantitative convergence rates, finite-horizon transient durations ($\beta^\star(d)$), unconditional convergence outside the certified rectangle (1), and fixed-rate EMA deep-RL dynamics remain open.

---

## 1. Algorithm, Strict Complementarity, and Certified Rectangle

Following Algorithm 1 and Section 4 of Savcı (arXiv:2606.22688), represent a strategy pair by $z = (x, y) \in \Delta_m \times \Delta_n$, its ascent payoff vector by $g(z) := (Ay, -A^\top x)$, and the cumulative average by $a_t := (t+1)^{-1} \sum_{j=0}^t z_j$. Initialize $a_0 = z_0 > 0$ (full support) and $g_{-1} = 0$. With $g_t := g(z_t)$, $\alpha_t := 1/(t+2)$, and exponential normalization applied separately within each player's simplex,
$$
h_t \propto z_t \exp\bigl(\eta(2g_t - g_{t-1})\bigr), \qquad
z_{t+1} = (1 - \beta) h_t + \beta a_t, \qquad
a_{t+1} = (1 - \alpha_t) a_t + \alpha_t z_{t+1}.
$$
*(Why raw half-step KL monotonicity fails:* Conjecture `conj:li` proposes reducing convergence via $\Phi_{t+1/2} \le \Phi_t - D_t$ ($D_t \ge 0$), where $\Phi_t = D(p \| z_t)$. However, in matching pennies against the uniform Nash pair $p$, the linear payoff sum in $D(p \| h_0) - D(p \| z_0)$ vanishes while strict Jensen makes the log-normalizer sum strictly exceed $2\eta\langle z_0, g(z_0)\rangle = 0$ at every nonequilibrium interior start $z_0$. We instead pair the optimistic OMWU energy of Wei et al., 2021 with the exact coordinate representation of the anchor.)*

Let $S$ be the union of the supports of all Nash equilibria across both player blocks, let $v$ be the game value, and let $M := \max(1, \max_{i,j}|A_{ij}|)$. By Goldman–Tucker strict complementarity (1956)—established directly via minimax LP duality: for each row $i \notin S$, maximizing $x_i$ subject to $x \ge 0$, $\mathbf{1}^\top x = 1$, $A^\top x \ge v\mathbf{1}$ has optimum $0$, so dual feasibility gives $q \ge 0, \lambda \in \mathbb{R}$ with $\lambda \mathbf{1} - Aq \ge e_i$ and $\lambda = v \mathbf{1}^\top q$, whence $q \ne 0$ and $q / (\mathbf{1}^\top q)$ is an optimal column strategy with strictly positive slack at $i$; averaging such witnesses across all $i \notin S$, outside columns, and support witnesses—there exists a Nash equilibrium $p^0$ with positive mass on all of $S$ and strictly positive payoff slack on every action outside $S$. Set
$$
\mu := \min_{i \in S} p_i^0 > 0, \qquad
\gamma := \min\bigl(\{v - (A p_y^0)_i : i \notin S\} \cup \{((p_x^0)^\top A)_j - v : j \notin S\}\bigr) \in (0, \infty]
$$
(taking $\gamma = +\infty$ when $S$ is full). Our certified stepsize rectangle is
$$
0 < c := \frac{\beta}{1 - \beta} \le \frac{\mu}{8}, \qquad
0 < \eta \le \min\!\left(\frac{1}{64 M},\, \frac{\gamma}{1200 M^2}\right).
\tag{1}
$$
Throughout the proof, $p$ denotes any Nash equilibrium sufficiently close to $p^0$ that $p_i \ge \mu/2$ for all $i \in S$ and every outside slack is at least $\gamma/2$. Writing $m(z) := \sum_{i \notin S} z_i$ (omitting $\gamma m$ terms when $S$ is full), skew-symmetry $\langle g(z), z \rangle = 0$ and optimality of $p$ give
$$
\langle g(z), z - p \rangle = \langle g(p), z - p \rangle \le -\frac{\gamma}{2} m(z).
\tag{2}
$$

---

## 2. Implicit Anchor Correction and Dissipative Potential

For a single step, abbreviate $z = z_t$, $w = z_{t+1}$, $a = a_t$, $a' = a_{t+1}$, $\alpha = \alpha_t$. Define $s_i := c(1 - a_i/w_i)$ and $L_i := \log(1 + s_i)$. From $w = (1-\beta)h + \beta a$, we obtain $h_i = w_i(1 + s_i)$ and $\sum_i w_i s_i = 0$ on each player block, hence
$$
\log\frac{w_i}{z_i} = \eta(2g_t - g_{t-1})_i - L_i - \log Z,
\tag{3}
$$
with the block normalizer $Z$.

**Lemma 1 (Uniform ratio and correction bounds).** *Under (1), for all $t \ge 0$ and every coordinate $i$, $|s_{t,i}| \le 12\eta M \le 3/16 < 1/2$, $z_{t,i}/a_{t,i} \ge l := \beta/[1 - (1-\beta)k]$, and $w_{t,i} \le 2 z_{t,i}$, where $K := e^{6\eta M}$ and $k := K^{-1}$.*

*Proof.* Since $\|2g_t - g_{t-1}\|_\infty \le 3M$, we have $k \le h_i / z_i \le K$. Initially $z_0 / a_0 = 1 \ge l$. Inductively, $w_i/a_i = (1-\beta)(h_i/z_i)(z_i/a_i) + \beta \ge (1-\beta)kl + \beta = l$, and convex blending $a'_i = (1-\alpha)a_i + \alpha w_i$ preserves $w_i/a'_i \ge l$, which gives $s_i \ge k - 1$. If $(1-\beta)K < 1$, the same induction yields $w_i/a_i \le \beta/[1 - (1-\beta)K]$ and $s_i \le K - 1$; otherwise $(1-\beta)K \ge 1$ directly implies $s_i \le c = \beta/(1-\beta) \le K - 1$. Also $w_i/z_i \le (1-\beta)K + \beta/l = 1 + (1-\beta)(K-k)$. Since $6\eta M \le 3/32$, elementary bounds give $K - 1 \le 12\eta M \le 3/16$ and $1 + (1-\beta)(K-k) < 2$. $\square$

**Lemma 2 (Anchor KL dissipation).** *Let $D(p \| z)$ sum the two player KL divergences, and set $H_t := D(p \| a_t)$. Under (1),*
$$
-\langle L, w - p \rangle \le \frac{c}{\alpha}(H_t - H_{t+1}) - c(1 - \alpha)\sum_i \frac{p_i(w_i - a_i)^2}{w_i a'_i} + \sum_i w_i s_i^2.
\tag{4}
$$
*Consequently, on $S$ a dissipation of at least $\frac{c\mu}{8}\sum_{i \in S} (w_i - a_i)^2 / w_i$ remains, while off $S$ we have $\sum_{i \notin S} w_i s_i^2 \le 288\eta^2 M^2 m(z) \le \frac{\eta\gamma}{4} m(z)$.*

*Proof.* For $r := w_i / a_i > 0$, applying $\log u \ge 1 - 1/u$ to $u = 1 + \alpha(r-1) = a'_i / a_i$ gives
$$
\frac{1}{\alpha}\log\bigl(1 + \alpha(r - 1)\bigr) - \Bigl(1 - \frac{1}{r}\Bigr)
\ge \frac{r - 1}{1 + \alpha(r-1)} - \frac{r - 1}{r}
= \frac{(1 - \alpha)(r - 1)^2}{r(1 + \alpha(r-1))}
= \frac{(1 - \alpha)(w_i - a_i)^2}{w_i a'_i}.
$$
Using $L_i \le s_i$ on the $p$-terms, $L_i \ge s_i - s_i^2$ (valid for $s_i \ge -1/2$) on the $w$-terms, $\sum_i w_i s_i = 0$, and $H_t - H_{t+1} = \sum_i p_i \log(a'_i / a_i)$ gives (4). On $S$, $w_i s_i^2 = c^2 (w_i - a_i)^2 / w_i$, while $p_i \ge \mu/2$, $1 - \alpha \ge 1/2$, $a'_i \le 1$, and $c \le \mu/8$ leave at least $c(\mu/4 - c)(w_i - a_i)^2/w_i \ge \frac{c\mu}{8}(w_i - a_i)^2/w_i$. Off $S$, Lemma 1 yields $\sum_{i \notin S} w_i s_i^2 \le 2 \cdot (12\eta M)^2 m(z) = 288\eta^2 M^2 m(z) \le 0.24 \eta \gamma m(z) < \frac{\eta\gamma}{4} m(z)$ by (1). $\square$

Let $\|z\| := \sum_i |z_i|$ across both blocks and $d_t := \|z_t - z_{t-1}\|$. For $t \ge 1$, define the optimistic energy
$$
E_t(p) := D(p \| z_t) - \eta\langle g_{t-1}, z_t - p \rangle + \frac{d_t^2}{16}.
$$
By blockwise Pinsker's inequality $D(p \| z) \ge \|z - p\|^2/4$, $\|g(z) - g(z')\|_\infty \le M\|z - z'\|$, (2), and Young's inequality on $\eta M d_t \|z_t - p\|$ (with $2\eta^2 M^2 < 1/32$), every Nash comparator satisfies
$$
E_t(p) \ge \frac{1}{8}\|z_t - p\|^2 + \frac{1}{32} d_t^2.
\tag{5}
$$

**Proposition 1 (Harmonic-scale Lyapunov monotonicity).** *Under (1), for every $t \ge 1$ and every Nash pair $p$ in the specified neighborhood of $p^0$,*
$$
E_{t+1}(p) - E_t(p) \le \frac{c}{\alpha_t}(H_t - H_{t+1}) - R_t,
\qquad
R_t := \frac{d_t^2 + d_{t+1}^2}{32} + \frac{c\mu}{8}\sum_{i \in S}(w_i - a_i)^2 + \frac{\eta\gamma}{4} m(z_t).
\tag{6}
$$
*Consequently, $V_t(p) := H_t(p) + (\alpha_t / c) E_t(p)$ satisfies $V_{t+1}(p) \le V_t(p) - (\alpha_t / c) R_t$, $\sum_{t=1}^\infty \alpha_t R_t < \infty$, $a_{t,i}$ and $z_{t,i}$ are uniformly bounded below on $S$, and $\lim_{t \to \infty} H_t(p)$ exists for every $p$ in that neighborhood of $p^0$.*

*Proof.* From (3), the exact KL step identity is
$$
D(p \| w) - D(p \| z) = \eta\langle 2g_t - g_{t-1}, w - p \rangle - D(w \| z) - \langle L, w - p \rangle.
$$
Subtracting $\eta\langle g_t, w - p \rangle - \eta\langle g_{t-1}, z - p \rangle$ leaves $\eta\langle g_t, z - p \rangle + \eta\langle g_t - g_{t-1}, w - z \rangle$. The second term is at most $\frac{\eta M}{2}(d_t^2 + d_{t+1}^2) \le \frac{1}{128}(d_t^2 + d_{t+1}^2)$. Since $-D(w \| z) \le -d_{t+1}^2 / 4$, the lag terms sum to $(-1/4 + 1/16 + \eta M/2) d_{t+1}^2 + (-1/16 + \eta M/2) d_t^2 \le -(d_t^2 + d_{t+1}^2)/32$. Combining with (2), Lemma 2, and $w_i \le 1$ yields (6). Multiplying by $\alpha_t / c$ and using $\alpha_{t+1} \le \alpha_t$ and $E_{t+1} \ge 0$ gives $V_{t+1} - V_t \le -(\alpha_t / c) R_t$. Since $V_t \ge 0$, $V_t$ converges and $\sum \alpha_t R_t < \infty$. Boundedness of $H_t(p^0)$ bounds $a_{t,i}$ away from zero on $S$; Lemma 1 then bounds $z_{t,i} \ge l a_{t,i}$ away from zero on $S$, keeping $E_t(p)$ bounded, so $(\alpha_t/c) E_t(p) \to 0$ and $H_t(p)$ converges. $\square$

---

## 3. Anchor Convergence, Invasion Lemma, and Last-Iterate Convergence

**Theorem 1 (Global last-iterate convergence on the certified rectangle).** *Under (1), from every interior initialization $z_0 > 0$, there exists a Nash equilibrium $u$ such that $\lim_{t \to \infty} a_t = \lim_{t \to \infty} z_t = u$.*

*Proof.* **Step 1 (Cluster points lie in the equilibrium affine slice $B$).** Any bounded nonnegative sequence $b_t$ with $\sum b_t/(t+2) < \infty$ has Cesàro mean tending to $0$ (discard a prefix so the tail is $< \varepsilon$, then bound the remaining mean through $T$ by $\frac{T+2}{T+1}\sum_{t \ge t_0} \frac{b_t}{t+2}$). By (6), $a_t$ loses all mass outside $S$, and for every $\varepsilon > 0$ the set $\{t : R_t > \varepsilon\}$ has finite harmonic mass and natural density $0$.

Let $u$ be any cluster point of $a_t$. Since $\|a_{t+1} - a_t\| \le 4\alpha_t$, every neighborhood of $u$ is visited with infinite harmonic mass (each exit across an annulus consumes fixed positive harmonic mass). We may therefore select a subsequence with $a_t \to u$ and $R_t \to 0$. Along this subsequence, $w, z_t, z_{t-1} \to u$ by (6), so continuity of the exponential update gives $h = u$. Thus supported row payoffs are constant and supported column payoffs are constant; since $a_{t,i}$ is bounded below on $S$ and vanishes off $S$, $\operatorname{supp}(u) = S$. Moreover $(p_x^0)^\top A u_y = v$ because $u_y$ is supported on optimal columns, all of which bind against $p_x^0$; hence the supported payoff constants are $v$ and $-v$.

**Step 2 (Uniqueness of the anchor limit and natural-density convergence of $z_t$).** Every cluster point belongs to the affine space $B$ of normalized pairs supported on $S$ with $(Ay)_i = v$ on supported rows and $(x^\top A)_j = v$ on supported columns. By positivity of $p^0$ on $S$ and strict slack off $S$, a relative neighborhood of $p^0$ in $B$ consists entirely of Nash pairs. For any two cluster points $u, v' \in B$, convergence of $H_t(p)$ yields $\sum_{i \in S} p_i(\log u_i - \log v'_i) = 0$ throughout that neighborhood. Subtracting its values at $p^0$ and $p^0 + \delta(u - v')$ (for small $\delta > 0$) gives
$$
\sum_{i \in S} (u_i - v'_i)(\log u_i - \log v'_i) = 0,
$$
forcing $u = v'$. Thus $a_t \to u$. By (6), $z_t \to u$ in natural density (and similarly for any finite collection of time shifts).

**Step 3 (Invasion lemma: $u$ is a Nash equilibrium).** Suppose an outside action $i \notin S$ earns strictly more than the common supported ascent payoff against $u$. At $(z_t, z_{t-1}) = (u, u)$, the continuous exponential factor $h_{t,i}/z_{t,i}$ exceeds $1$, so it is at least $1 + \delta$ ($\delta > 0$) on a fixed neighborhood of $(u, u)$. Call steps in this neighborhood *good*. Every fixed-length window of $J$ consecutive steps consists entirely of good steps with natural density $1$.

Writing $r_t := z_{t,i}/a_{t,i}$, a good step gives $W_t := z_{t+1,i}/a_{t,i} \ge F(r_t) := \beta + (1-\beta)(1+\delta) r_t$, and $r_{t+1} = W_t / [1 + \alpha_t(W_t - 1)]$ (increasing in $W_t$). Iterating $F$ from $0$ strictly exceeds $1$ after a finite number $J$ of steps (since $F(1) > 1$). Because the intermediate denominators $1 + \alpha_t(W - 1) \to 1$ uniformly over the finite recursion from $0$, for sufficiently large $t$ every $J$-step good window ends with $W_t \ge 1 + e$ for some fixed $e > 0$. At such endpoints, $\log(a_{t+1,i}/a_{t,i}) \ge \log(1 + e\alpha_t) \ge \frac{e}{2}\alpha_t$; at all other steps, $\log(a_{t+1,i}/a_{t,i}) \ge \log(1 - \alpha_t) \ge -2\alpha_t$. By partial summation, any set of natural density $0$ has harmonic mass $o(\log T)$ through time $T$. Summing yields $\log a_{T,i} \to +\infty$, contradicting $a_{T,i} \le 1$. Thus $u$ satisfies all Nash inequalities.

**Step 4 (Ordinary last-iterate convergence $z_t \to u$).** Evaluate the fast energy (5) at $p = u$, and split the anchor term in (3) without dividing by $u_i$ off $S$:
$$
-\langle L, w - u \rangle = -\langle L, w - a \rangle - \langle L, a - u \rangle.
$$
Since $|s_i| \le 12\eta M < 1/2$ and $a_t \to u$, $-\langle L, a - u \rangle \to 0$. In the first term, $L_i$ shares the sign of $w_i - a_i$ and $|L_i| \ge |s_i|/2 = c|w_i - a_i|/(2w_i) \ge \frac{c}{2}|w_i - a_i|$. Using $\langle g_t, z - u \rangle \le 0$ and the same KL/lag bounds as in Proposition 1 gives
$$
E_{t+1}(u) \le E_t(u) - Q_t + o(1), \qquad Q_t := \frac{d_t^2 + d_{t+1}^2}{32} + \frac{c}{2}\sum_i (w_i - a_i)^2.
$$
Along any subsequence with $Q_t \to 0$, we have $w_t - a_t \to 0$ and $d_{t+1}, d_t \to 0$, hence $z_t \to u$ and $E_t(u) \to 0$ (since $u_i > 0$ only on $S$, where $z_{t,i}$ is bounded below). Thus for every $\varepsilon > 0$, whenever $E_t(u) \ge \varepsilon$ at sufficiently large $t$, $Q_t$ exceeds a positive constant $q(\varepsilon) > 0$. Because $E_t(u) \ge 0$ decreases by at least $q(\varepsilon)/2$ above $\varepsilon$ and rises by at most $o(1)$ below $\varepsilon$, we obtain $\limsup_{t \to \infty} E_t(u) \le \varepsilon$. Sending $\varepsilon \downarrow 0$ and invoking (5) proves $z_t \to u$. $\square$

*(Remark: Step 3 applies at **any** $\beta \in (0, 1)$ and $\eta > 0$ whenever $z_t \to u$ from an interior start, since $a_t \to u$ and every late step is good; thus a converged non-Nash point is impossible.)*

---

## 4. Refutation of Asymptotic Premature-Consensus Plateaus

**Theorem 2 (No positive-exploitability asymptotic consensus plateau).** *Fix any $\beta \in (0, 1)$ and $\eta > 0$, and suppose the game admits a fully mixed Nash equilibrium $p > 0$. If $\|z_t - a_t\| \to 0$ from an interior initialization $z_0 > 0$, then*
$$
\liminf_{t \to \infty} \Bigl[\max_i (A y_t)_i - \min_j (x_t^\top A)_j\Bigr] = 0.
$$

*Proof.* Let $D_i(z)$ denote the ascent payoff of action $i$ minus its player's average ascent payoff at $z$, and put $q := 1 - \beta$. Skew-symmetry and full mixing of $p$ yield $\sum_i p_i D_i(z) = 0$ for every $z$. Consensus $\|z_t - a_t\| \to 0$ together with $a_{t+1} - a_t = \alpha_t(z_{t+1} - a_t)$ implies $z_{t+1} - z_t \to 0$ and hence $h_t - z_t \to 0$. At every cluster point $z$ of $\{z_t\}$, the exponential update fixes $z$, so supported payoffs equal each block's mean payoff. By compactness and continuity of the exponential formula,
$$
\frac{h_{t,i}}{z_{t,i}} = \exp\bigl(\eta D_i(z_t)\bigr) + o(1)
$$
uniformly over all coordinates (including vanishing ones).

Fix an integer $J \ge 2$ and compare the $J$-step ratio recursion for $z_{t+1,i}/a_{t,i}$ against the monotone lower-bound recursion started at ratio $0$. Over any $J$-step window, this lower recursion stays bounded, intermediate denominators $1 + \alpha(W - 1) \to 1$ uniformly, and $D_i$ varies by $o(1)$, giving
$$
\frac{z_{t+1,i}}{a_{t,i}} \ge W_{t,i}^{\mathrm{lb}} := \beta \sum_{k=0}^{J-1} q^k \exp\bigl(k\eta D_i(z_t)\bigr) - o(1).
$$
Suppose for contradiction that the Nash gap were eventually at least $e > 0$. Then at every late $t$, some action has $D_i(z_t) \ge e/2$. Using $\sum_i p_i = 2$, $\sum_i p_i D_i = 0$, and $e^s \ge 1 + s$, we have $\sum_i p_i \exp(k\eta D_i) \ge 2$ for every $k \ge 0$, and at $k = 1$ the strict excess is at least $b := (\min_i p_i)\bigl(e^{\eta e/2} - 1 - \eta e/2\bigr) > 0$. Choosing $J \ge 2$ with $2 q^J < \beta q b / 2$, we obtain
$$
\sum_i p_i (W_{t,i}^{\mathrm{lb}} - 1)
\ge \beta \sum_{k=0}^{J-1} 2 q^k + \beta q b - 2 - o(1)
= \beta q b - 2 q^J - o(1) > \frac{\beta q b}{2} - o(1).
$$
Because $W_{t,i}^{\mathrm{lb}}$ is uniformly bounded above and below, inserting $W_{t,i}^{\mathrm{lb}}$ into the increasing function $\log(1 + \alpha_t(W - 1))$ and Taylor expanding in $\alpha_t$ yields
$$
\sum_i p_i \log\frac{a_{t+1,i}}{a_{t,i}} \ge \alpha_t \sum_i p_i (W_{t,i}^{\mathrm{lb}} - 1) - O(\alpha_t^2) \ge \frac{\beta q b}{4}\,\alpha_t
$$
for all sufficiently large $t$. Summing over $t$ gives $\sum_i p_i \log a_{t,i} \to +\infty$, contradicting $a_{t,i} \le 1$. $\square$

---

## 5. Reproducible Numerical Verification

Run from the repository root:

```sh
.venv/bin/python -B solutions/garip_last_iterate/explore.py
```

The script [`explore.py`](explore.py) tests 7 matrix games $\times$ 2 interior starts $\times$ 6 parameter pairs (84 trajectories of 7,000 updates each, including 56 trajectories inside the certified rectangle (1) spanning opposite scale separations and steps of order $10^{-7}$). At every step it verifies Lemma `lem:klconv` and, inside (1), the exact Lyapunov inequality (6). It also confirms that the raw matching-pennies half-step KL increases by $0.0002957693$ on certified runs.

| Game | Practical gap ($\beta=0.015, \eta=0.18/M$): start $\to$ 7,000 steps | Large-anchor gap ($\beta=0.45, \eta=0.01/M$) | Max scaled residual of (6) |
|---|---|---|---:|
| Matching pennies | $[0.6914, 1.1896] \to [0.0000465, 0.0001455]$ | $[0.6581, 1.2440]$ | $-3.8 \times 10^{-19}$ |
| Rock–paper–scissors | $[0.8772, 1.4701] \to [0.0002607, 0.0006858]$ | $[0.8653, 1.5108]$ | $-3.5 \times 10^{-19}$ |
| Asymmetric | $[2.7715, 2.6426] \to [0.0004590, 0.0004276]$ | $[2.7691, 2.6495]$ | $-4.9 \times 10^{-18}$ |
| Duplicate actions | $[1.2239, 1.1133] \to [0.0002489, 0.0003616]$ | $[1.1211, 1.2181]$ | $-1.9 \times 10^{-18}$ |
| Dominated actions | $[2.3498, 2.4967] \to [0.0012715, 0.0007127]$ | $[2.3060, 2.4151]$ | $-1.3 \times 10^{-11}$ |
| Weak boundary | $[0.8000, 0.9687] \to [0.0025485, 0.0004371]$ | $[0.8000, 0.9672]$ | $1.1 \times 10^{-17}$ |
| Zero game | $[0, 0] \to [0, 0]$ | $[0, 0]$ | $5.6 \times 10^{-17}$ |

All assertions passed (worst positive scaled residual $5.6 \times 10^{-17}$).

---

## References

- C. Savcı, *GARIP: A Running-Average Moving Reference for Last-Iterate Self-Play in Two-Player Zero-Sum Games*, [arXiv:2606.22688](https://arxiv.org/abs/2606.22688), June 2026.
- A. J. Goldman and A. W. Tucker, *Theory of Linear Programming*, in *Linear Inequalities and Related Systems*, Annals of Mathematics Studies **38**, Princeton University Press, 53–97, 1956.
- C. Daskalakis and I. Panageas, *Last-Iterate Convergence: Zero-Sum Games and Constrained Min-Max Optimization*, ITCS 2019.
- C.-Y. Wei, C.-W. Lee, M. Zhang, and H. Luo, *Linear Last-Iterate Convergence in Constrained Saddle-Point Optimization*, ICLR 2021.
