# Proof of Optimality of the Greedy Cutting Schedule for Randomized Littlestone Dimension of Intervals

**Target paper:** Miit Daga and Swarna Priya Ramu, *How many mistakes does it take to learn an interval? Exact Littlestone dimensions of intervals and their unions*, [arXiv:2611.28727](https://arxiv.org/abs/2611.28727) (November 2026), Section 7, Conjecture `conj:rl`.  
**Companion files:** [`solutions/interval_littlestone/explore.py`](explore.py), [`solutions/interval_littlestone/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture `conj:rl`: PROVED.**
  - For every finite multiset $S$ of positive block lengths, the greedy canonical cutting schedule attains the exact randomized Littlestone dimension $\mathrm{RL}(J_S)$ of the interval-union position $J_S$ (Theorem 1).
  - In particular, for every integer $m \ge 0$, the randomized Littlestone dimension of discrete intervals on $m$ points satisfies the exact recurrence
    $$
    \mathrm{RL}(\mathrm{Int}_m) = R(m + 1),
    $$
    where $R(1) = 0$ and $R(p) = 1 - 2^{1 - p} + \tfrac{1}{2}\bigl(R(\lfloor p/2 \rfloor) + R(\lceil p/2 \rceil)\bigr)$ for $p \ge 2$ (Theorem 2).
- **Proof mechanism:** A dyadic layer decomposition expresses the piecewise-linear interpolation $\tilde\lambda(x)$ of $\log_2 x$ as a sum of $[0, 1]$-valued layer rewards $h_d(x) = Q_d(x) - 2 Q_{2d}(x)$ ($d = 2^k$). Every single-rod canonical tree is *packed* at every dyadic scale with exact mass $Q_d(x)$. Comparing any alternative cut $u + v = p$ to the canonical split $a + a' = p$ in the presence of an arbitrary multiset $T$ of other rods, packed majorization bounds the future continuation gain by $\frac{1}{2}\sum_d E_d$, which telescopes *identically* to the immediate-reward loss $c(a, a') - c(u, v)$.
- **Formalization note:** The source paper formalizes its deterministic theorems in Lean 4; the present proof is purely finitary and rational, making it directly amenable to integration into the same Lean 4 development.

---

## 1. Reduction to Discounted Rod Cutting

For a finite multiset $S$ of positive integers, place one disjoint ordered block of $g$ points for each occurrence of $g \in S$. The hypothesis class $J_S$ consists of the empty set together with all nonempty intervals contained inside a single block; in particular $\mathrm{Int}_m = J_{\{m\}}$ (with just the empty hypothesis when $m = 0$). Represent each block of $g$ points by a *rod* of measure $p := g + 1 \ge 2$; unit rods ($p = 1$) carry no queryable points and may be kept or dropped freely. All unions of rods below are multiset unions.

For a nonempty finite hypothesis class $H$, its randomized Littlestone dimension $\mathrm{RL}(H)$ is half the maximum expected branch length of a binary query tree shattered by $H$ under uniform random branch choices, satisfying the recursion
$$
\mathrm{RL}(H) = \max\Bigl(\{0\} \cup \Bigl\{\tfrac{1}{2}\bigl(1 + \mathrm{RL}(H_x^0) + \mathrm{RL}(H_x^1)\bigr) : x \text{ splits } H\Bigr\}\Bigr),
$$
where $H_x^b := \{h \in H : h(x) = b\}$ must both be nonempty. On disjoint domains, $\mathrm{RL}(H_1 \otimes H_2) = \mathrm{RL}(H_1) + \mathrm{RL}(H_2)$ (by induction on $|H_1| + |H_2|$, since any query splits one factor while leaving the other intact).

Following Section 7 of arXiv:2611.28727, write $\tilde\lambda(x)$ for the piecewise-linear interpolation of $\log_2 x$ at powers of two:
$$
\tilde\lambda(x) := t + \frac{x - 2^t}{2^t} \qquad (x \ge 1,\; t = \lfloor \log_2 x \rfloor).
$$
A chain of $c$ endpoint thresholds has $\mathrm{RL} = \tfrac{1}{2}\tilde\lambda(c)$: every split leaves chains of sizes $u, v \ge 1$ with $u + v = c$; concavity and $\tilde\lambda(2x) = 1 + \tilde\lambda(x)$ give $\tilde\lambda(u) + \tilde\lambda(v) \le 2\tilde\lambda(c) - 2$, attained by $u = \lfloor c/2 \rfloor, v = \lceil c/2 \rceil$ since both lie in one linear segment of $\tilde\lambda$.

Querying the $u$-th point of a block of rod measure $p$ ($1 \le u \le p - 1$, $v := p - u \ge 1$) yields:
- On a positive answer ($1$): independent left and right endpoint threshold chains of sizes $u$ and $v$ on disjoint remaining coordinates, with value $\tfrac{1}{2}(\tilde\lambda(u) + \tilde\lambda(v))$.
- On a negative answer ($0$): the queried block splits into two rods of measures $u$ and $v$, while all other rods $T$ persist.

Writing $V(P) := \mathrm{RL}(J_S)$ for the rod multiset $P$, the Bellman recursion (`eq:rlcut` in arXiv:2611.28727) is
$$
V(P) = \max_{\substack{p \in P,\, p \ge 2 \\ u + v = p,\; u, v \ge 1}} \frac{1}{2}\Bigl(c(u, v) + V\bigl((P \setminus \{p\}) \cup \{u, v\}\bigr)\Bigr),
\qquad c(u, v) := 1 + \frac{\tilde\lambda(u) + \tilde\lambda(v)}{2},
\tag{1}
$$
with $V(\varnothing) = 0$. Every legal cut reduces the rank $\sum_{r \in P}(r - 1)$ by $1$.

For $p \ge 2$, put $b := 2^{\lfloor \log_2 p \rfloor - 1}$ and define the **canonical split** of $p$ by
$$
a := \max(b,\, p - 2b), \qquad a' := p - a.
$$
Both $a$ and $a'$ lie in $[b, 2b]$, and at least one equals $b$ or $2b$. Because $\tilde\lambda$ is affine on $[b, 2b]$ with $\tilde\lambda(2x) = 1 + \tilde\lambda(x)$, we have $c(a, a') = \tilde\lambda(p)$. The **greedy schedule** always applies the canonical split to a largest available rod. Recursively splitting canonically down to units associates to any multiset $P$ a finite binary *canonical forest*. Since every child has strictly smaller measure than its parent, greedy processes all internal nodes of this forest in non-increasing order of measure (breaking ties arbitrarily). Hence the greedy value $G(P)$ equals the discounted sum $\sum_{i \ge 1} 2^{-i} \tilde\lambda(p_i^\downarrow)$ over the internal-node measures $p_1^\downarrow \ge p_2^\downarrow \ge \cdots$ of the canonical forest of $P$.

---

## 2. Dyadic Layer Decomposition and Packedness

For every dyadic scale $d \in \{1, 2, 4, 8, \dots\}$ and $x \ge 1$, define
$$
Q_d(x) := \max\!\left(0,\, \frac{x}{d} - 1\right), \qquad
h_d(x) := Q_d(x) - 2 Q_{2d}(x) = \min\bigl(1,\, Q_d(x)\bigr).
$$
One checks the identity $Q_d(x) - 2 Q_{2d}(x) = \min(1, Q_d(x))$ directly across the three regimes $x \le d$, $d < x \le 2d$, and $x > 2d$. When $2^t \le x < 2^{t+1}$, the first $t$ layers ($d = 1, \dots, 2^{t-1}$) equal $1$, the layer $d = 2^t$ equals $(x - 2^t)/2^t$, and all higher layers vanish. Summing over all dyadic $d$ (a finite sum for each fixed $x$) gives
$$
\tilde\lambda(x) = \sum_d h_d(x), \qquad
\sum_d Q_d(x) = 2(x - 1) - \tilde\lambda(x).
\tag{2}
$$
For any finite list $X$ of nonnegative numbers, write $D(X) := \sum_{i \ge 1} 2^{-i} X_i^\downarrow$, where $X_1^\downarrow \ge X_2^\downarrow \ge \cdots$ sorts $X$ in descending order with zero padding. Let $A_d(P)$ denote the multiset of values $h_d(r)$ over all internal nodes $r$ of the canonical forest of $P$. Because every $h_d$ is nondecreasing in $r$, the same non-increasing ordering of internal-node measures sorts every layer simultaneously, so
$$
G(P) = \sum_d D\bigl(A_d(P)\bigr).
\tag{3}
$$

Call a list of entries in $[0, 1]$ **packed** if, after sorting in descending order and dropping zeros, it consists of ones followed by at most one entry strictly between $0$ and $1$.

**Lemma 1 (Single-rod packedness).** *For every integer $x \ge 1$ and every dyadic $d = 2^k$ ($k \ge 0$), the list $A_d(\{x\})$ is packed and has total mass (sum of entries) $Q_d(x)$.*

*Proof.* We induct on $x$. If $x \le d$, every node in the canonical tree of $x$ has measure $\le d$, so $A_d(\{x\})$ has only zero entries and mass $0 = Q_d(x)$. If $d < x < 2d$ (which can occur only for $d \ge 2$), then $b = d/2$, so both canonical children $a, a'$ lie in $[d/2, d]$ and contribute zero; only the root contributes, with value $h_d(x) = x/d - 1 \in (0, 1)$, which is packed of mass $Q_d(x)$.

If $x \ge 2d$, then $b = 2^{\lfloor \log_2 x \rfloor - 1} \ge d$, so both canonical children $a, a'$ satisfy $a, a' \ge b \ge d$, and at least one of them is a power of two $\ge b \ge d$, hence divisible by $d$. The root contributes $h_d(x) = 1$, and
$$
1 + Q_d(a) + Q_d(a') = 1 + \left(\frac{a}{d} - 1\right) + \left(\frac{a'}{d} - 1\right) = \frac{x}{d} - 1 = Q_d(x).
$$
By induction, $A_d(\{a\})$ and $A_d(\{a'\})$ are packed of masses $Q_d(a)$ and $Q_d(a')$. The child divisible by $d$ has integer mass $Q_d \in \mathbb{Z}_{\ge 0}$, so its packed list consists solely of ones (and zeros). Combining it with the other packed child list and the root entry $1$ produces a packed list of mass $Q_d(x)$. $\square$

**Lemma 2 (Per-layer split comparison).** *Let $p \ge 2$, let $a + a' = p$ be its canonical split, and let $u + v = p$ ($u, v \ge 1$) be any legal split. For every dyadic $d = 2^k$, the list $A_d(\{a, a'\})$ is packed with mass $M_d := Q_d(a) + Q_d(a')$, and the list $A_d(\{u, v\})$ has mass $M_d + E_d$ for some excess $E_d \ge 0$.*

*Proof.* Since $p \ge 2$, $b = 2^{\lfloor \log_2 p \rfloor - 1} \ge 1$ is a power of two, so every dyadic $d$ satisfies either $d \le b$ or $d \ge 2b$.
- If $d \le b$: both $a, a' \ge b \ge d$, and one of them is a power of two $\ge b \ge d$, hence divisible by $d$. By Lemma 1, that child's layer has only unit positive entries, so $A_d(\{a, a'\}) = A_d(\{a\}) \cup A_d(\{a'\})$ is packed with mass
  $$
  M_d = \left(\frac{a}{d} - 1\right) + \left(\frac{a'}{d} - 1\right) = \frac{p}{d} - 2 \le Q_d(u) + Q_d(v) = M_d + E_d,
  $$
  where the inequality drops the positive parts in $Q_d(u)$ and $Q_d(v)$.
- If $d \ge 2b$: both $a, a' \le 2b \le d$, so $A_d(\{a, a'\})$ vanishes identically ($M_d = 0$, trivially packed), and $E_d = Q_d(u) + Q_d(v) \ge 0$. $\square$

---

## 3. Packed Majorization and Optimality of Greedy

**Lemma 3 (Packed majorization under union and discounting).** *Let $A$ be a packed list of mass $M$, let $B$ be any list with entries in $[0, 1]$ of mass $M + E$ ($E \ge 0$), and let $C$ be any list with entries in $[0, 1]$. Then*
$$
D(C \cup B) - D(C \cup A) \le \frac{E}{2}.
\tag{4}
$$

*Proof.* For any nonnegative list $X$ and integer $k \ge 0$, let $s_k(X) := \sum_{i=1}^k X_i^\downarrow$ (zero-padded, $s_0 = 0$). Because $A$ is packed of mass $M$, $s_k(A) = \min(k, M)$ for every $k \ge 0$. Because every entry of $B$ lies in $[0, 1]$ and its entries sum to $M + E$,
$$
s_k(B) \le \min(k,\, M + E) \le \min(k, M) + E = s_k(A) + E \qquad (\forall k \ge 0).
$$
Taking the $k$ largest entries of a union chooses $j$ largest from $C$ and $k - j$ largest from $X$ for some $0 \le j \le k$, so
$$
s_k(C \cup X) = \max_{0 \le j \le k} \bigl\{s_j(C) + s_{k - j}(X)\bigr\}.
$$
Therefore $s_k(C \cup B) \le s_k(C \cup A) + E$ for every $k \ge 1$. By Abel summation,
$$
D(X) = \sum_{i \ge 1} 2^{-i} X_i^\downarrow = \sum_{k \ge 1} 2^{-k - 1} s_k(X),
$$
since $X_i^\downarrow$ appears in $s_k(X)$ for all $k \ge i$ with total weight $\sum_{k \ge i} 2^{-k - 1} = 2^{-i}$. Because $\sum_{k \ge 1} 2^{-k - 1} = 1/2$,
$$
D(C \cup B) - D(C \cup A) = \sum_{k \ge 1} 2^{-k - 1}\bigl(s_k(C \cup B) - s_k(C \cup A)\bigr) \le \frac{E}{2}. \qquad \square
$$

**Theorem 1 (Optimality of the greedy schedule).** *For every finite multiset $P$ of rods, $V(P) = G(P)$. Consequently, the greedy schedule attains $\mathrm{RL}(J_S)$ for every finite multiset $S$ of positive block lengths.*

*Proof.* We induct on the rank $\sum_{r \in P}(r - 1)$. When the rank is $0$, no rod has measure $\ge 2$ and $V(P) = G(P) = 0$.

Now suppose $P$ has positive rank, and consider any legal first cut splitting some $p \in P$ ($p \ge 2$) into $u + v = p$ ($u, v \ge 1$). Write $T := P \setminus \{p\}$ and let $a + a' = p$ be the canonical split of $p$. Applying Lemma 3 at every dyadic scale $d$ with $C = A_d(T)$, $A = A_d(\{a, a'\})$, and $B = A_d(\{u, v\})$, and summing via (3) and Lemma 2, gives
$$
G(T \cup \{u, v\}) - G(T \cup \{a, a'\}) \le \frac{1}{2} \sum_d E_d.
$$
By the mass identity in (2) and $u + v = a + a' = p$,
$$
\begin{aligned}
\frac{1}{2}\sum_d E_d
&= \frac{1}{2}\sum_d \bigl(Q_d(u) + Q_d(v) - Q_d(a) - Q_d(a')\bigr) \\
&= \frac{1}{2}\bigl(\tilde\lambda(a) + \tilde\lambda(a') - \tilde\lambda(u) - \tilde\lambda(v)\bigr)
= c(a, a') - c(u, v).
\end{aligned}
\tag{5}
$$
Since $T \cup \{u, v\}$ has rank strictly smaller than $P$, the induction hypothesis gives $V(T \cup \{u, v\}) = G(T \cup \{u, v\})$. Combining (1), (5), and $c(a, a') = \tilde\lambda(p)$, the value of cutting $p$ into $(u, v)$ is bounded by
$$
\frac{1}{2}\bigl(c(u, v) + V(T \cup \{u, v\})\bigr)
= \frac{1}{2}\bigl(c(u, v) + G(T \cup \{u, v\})\bigr)
\le \frac{1}{2}\bigl(\tilde\lambda(p) + G(T \cup \{a, a'\})\bigr).
$$
The right-hand side is the discounted sum obtained by placing the internal node $p$ first and all remaining internal nodes of the canonical forest of $T \cup \{p\}$ in non-increasing order of measure. By rearrangement (since the discount weights $2^{-i}$ decrease strictly and $\tilde\lambda$ is nondecreasing), placing *all* internal nodes of the canonical forest of $T \cup \{p\}$ in non-increasing order attains at least as much, and that sorted schedule is precisely greedy on $T \cup \{p\} = P$. Thus every legal first cut has value at most $G(P)$, so $V(P) \le G(P)$. Since greedy is a legal schedule, $V(P) \ge G(P)$ as well, proving $V(P) = G(P)$. $\square$

---

## 4. Exact Recurrence for a Single Interval Class

**Theorem 2 (Single-interval formula).** *For every integer $p \ge 1$, $G(\{p\}) = R(p)$, where $R(1) = 0$ and*
$$
R(p) = 1 - 2^{1 - p} + \frac{1}{2}\Bigl(R(\lfloor p/2 \rfloor) + R(\lceil p/2 \rceil)\Bigr) \qquad (p \ge 2).
$$
*In particular, $\mathrm{RL}(\mathrm{Int}_m) = R(m + 1)$ for every integer $m \ge 0$.*

*Proof.* Define $H(y) := 0$ for $0 \le y \le 1$, set $H(n) := 1 - 2^{1 - n}$ for every integer $n \ge 1$, and interpolate $H$ linearly between consecutive integers. For a packed list of mass $(y - 1)_+$ ($y \ge 1$), writing $y = n + \theta$ with $n = \lfloor y \rfloor \ge 1$ and $\theta \in [0, 1)$, the list has $n - 1$ ones followed by $\theta$, so its discounted sum is
$$
\sum_{i = 1}^{n - 1} 2^{-i} + \theta\,2^{-n} = (1 - 2^{1 - n}) + \theta\bigl((1 - 2^{-n}) - (1 - 2^{1 - n})\bigr) = H(y).
$$
By Lemma 1 and (3), for every integer $p \ge 1$,
$$
G(\{p\}) = \sum_{j = 0}^\infty H(p / 2^j).
\tag{6}
$$
At $p = 1$, every term vanishes, giving $G(\{1\}) = 0 = R(1)$. For $p \ge 2$, separating the $j = 0$ term $H(p) = 1 - 2^{1 - p}$ from (6) leaves
$$
G(\{p\}) = 1 - 2^{1 - p} + \sum_{j = 0}^\infty H\!\left(\frac{p/2}{2^j}\right).
$$
For every $j \ge 0$, the function $x \mapsto H(x / 2^j)$ has breakpoints only at integer multiples of $2^j \in \mathbb{Z}_{\ge 1}$, hence is affine on the interval $[\lfloor p/2 \rfloor, \lceil p/2 \rceil]$ of length at most $1$ centered at $p/2$. Therefore
$$
H\!\left(\frac{p/2}{2^j}\right) = \frac{1}{2}\left(H\!\left(\frac{\lfloor p/2 \rfloor}{2^j}\right) + H\!\left(\frac{\lceil p/2 \rceil}{2^j}\right)\right) \qquad (\forall j \ge 0).
$$
Summing over $j \ge 0$ yields $G(\{p\}) = 1 - 2^{1 - p} + \tfrac{1}{2}\bigl(G(\{\lfloor p/2 \rfloor\}) + G(\{\lceil p/2 \rceil\})\bigr)$, matching the recurrence for $R(p)$ identically. Combining with Theorem 1 gives $\mathrm{RL}(\mathrm{Int}_m) = V(\{m + 1\}) = G(\{m + 1\}) = R(m + 1)$ for all $m \ge 0$. $\square$

---

## 5. Reproducible Verification

Run from the repository root:

```sh
.venv/bin/python -B solutions/interval_littlestone/explore.py
```

The script [`explore.py`](explore.py) performs exact integer/rational verification:
- Confirms exact Bellman optimality equals the greedy heap schedule on all **204,226 multisets** of non-unit rods with total measure $\le 50$, evaluating **2,977,224 distinct legal first cuts**.
- Confirms single rods match $R(m + 1)$ through $m = 49$ for the unrestricted Bellman DP and through $m = 511$ for direct greedy simulation.
- Confirms the generic shattered-tree recursion on explicit interval hypothesis classes agrees with $R(m + 1)$ for all $0 \le m \le 10$.

All assertions passed.

---

## References

- M. Daga and S. P. Ramu, *How many mistakes does it take to learn an interval? Exact Littlestone dimensions of intervals and their unions*, [arXiv:2611.28727](https://arxiv.org/abs/2611.28727), November 2026.
