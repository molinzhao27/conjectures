# Unbounded Separation Between $\mathrm{UB}$ and $\mathrm{ILB}$ for Temporal Feedback Graphs

**Target paper:** Khashayar Gatmiry and Jon Schneider, *Adversarial Online Learning with Temporal Feedback Graphs*, [COLT 2024, PMLR v247](https://proceedings.mlr.press/v247/gatmiry24b.html) (Conjecture 12) / [arXiv:2407.00571](https://arxiv.org/abs/2407.00571) (July 2024, Section 4.1, Conjecture 1, `conj:tight`).  
**Companion files:** [`solutions/temporal_feedback_regret/explore.py`](explore.py), [`solutions/temporal_feedback_regret/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture 12 (COLT 2024) / Conjecture 1 (`conj:tight`, arXiv:2407.00571): FALSE.**
- **Explicit polynomial separation (Theorem 1):** For every integer $k \ge 1$, the $k$-fold lexicographic power $\mathcal{S}_k$ of the directed triangle $C_3$ on $T = 3^k$ rounds is a regular vertex-transitive tournament satisfying
  $$
  \mathrm{UB}(\mathcal{S}_k) = \frac{3^k}{\sqrt{2^k}}, \qquad
  \mathrm{ILB}(\mathcal{S}_k) = \frac{3^k}{\sqrt{(3^k - 1)/2}}, \qquad
  \frac{\mathrm{UB}(\mathcal{S}_k)}{\mathrm{ILB}(\mathcal{S}_k)} = \sqrt{\frac{3^k - 1}{2^{k+1}}} \sim \frac{1}{\sqrt{2}}\left(\frac{3}{2}\right)^{k/2} = \Theta\!\left(T^{(1 - \log_3 2)/2}\right) \to \infty.
  $$
  Because $(1 - \log_3 2)/2 \approx 0.1845 > 0$, no universal constant $\gamma$ can bound $\mathrm{UB}(\mathcal{S})/\mathrm{ILB}(\mathcal{S})$.
- **Scope remarks:**
  1. *Optimization programs vs. minimax regret:* Theorem 1 refutes the conjectured constant-factor equivalence between the two convex programs $\mathrm{UB}$ and $\mathrm{ILB}$; determining the true minimax regret on $\mathcal{S}_k$ is a separate question.
  2. *Cyclic vs. acyclic graphs:* The paper's model explicitly allows directed cycles and feedback from future rounds (Section 1–2 of arXiv:2407.00571), so $\mathcal{S}_k$ is fully within scope. Whether $\mathrm{UB}/\mathrm{ILB}$ is uniformly bounded on non-transitive *acyclic* (causal / past-only) feedback graphs remains an open follow-up question.
  3. *Open vs. closed neighborhood constraints:* The paper's `ILB` constraints use the open visible sets $S_t$ ($I \cap S_t \neq \varnothing$). Even if closed neighborhoods $S_t \cup \{t\}$ were substituted, the feasible region would shrink and $\mathrm{ILB}$ could only decrease, so the unbounded separation persists.

---

## 1. Definitions and Convex Programs

A temporal feedback graph $\mathcal{S}$ on $T$ rounds assigns to each $t \in [T]$ a visible set $S_t \subseteq [T] \setminus \{t\}$; write $s \to t$ when $s \in S_t$. (As stated in Sections 1–2 of Gatmiry and Schneider, directed cycles and feedback from future rounds are allowed.) An *order* is a sequence $(t_1, \dots, t_m)$ of distinct rounds such that $t_i \to t_j$ for all $1 \le i < j \le m$. Let $\mathcal{C}$ denote the collection of maximal orders under subsequence extension.

1. **Upper-bound program $\mathrm{UB}(\mathcal{S})$:** Choose nonnegative weights $\lambda_{C, t} \ge 0$ supported on $t \in C$ with $\sum_{C \in \mathcal{C}} \lambda_{C, t} = 1$ for every $t \in [T]$, and minimize
   $$
   \mathrm{UB}(\mathcal{S}) := \min_\lambda \sum_{C \in \mathcal{C}} \|\lambda_C\|_2.
   $$
2. **Independent-set lower-bound program $\mathrm{ILB}(\mathcal{S})$:** Let $\mathcal{I}(\mathcal{S})$ be the set of nonempty independent sets of the underlying undirected graph of $\mathcal{S}$ (the empty set contributes to neither objective nor constraints). Writing $z_I := w_I^2 \ge 0$, maximize
   $$
   \mathrm{ILB}(\mathcal{S}) := \max_z \sum_{t=1}^T \sqrt{\sum_{I \in \mathcal{I}(\mathcal{S}) :\, t \in I} z_I}
   \qquad \text{subject to} \qquad
   \sum_{I \in \mathcal{I}(\mathcal{S}) :\, I \cap S_t \neq \varnothing} z_I \le 1 \quad (\forall t \in [T]).
   $$

Conjecture 12 of COLT 2024 (Conjecture 1, `conj:tight`, in arXiv:2407.00571) posits the existence of a universal constant $\gamma$ such that $\mathrm{UB}(\mathcal{S}) / \mathrm{ILB}(\mathcal{S}) \le \gamma$ for every temporal feedback graph $\mathcal{S}$.

---

## 2. The Lexicographic Triangle Power and Exact Separation

For $k \ge 1$, define $\mathcal{S}_k$ on the vertex set $V_k := (\mathbb{Z}/3\mathbb{Z})^k$ of size $T = 3^k$ by declaring $x \to y$ ($x \neq y$) if and only if, at the first coordinate $j \in \{1, \dots, k\}$ where $x_j \neq y_j$, we have $y_j - x_j \equiv 1 \pmod 3$. (This iterated blow-up of $C_3$ is a classical extremal tournament; cf. Stearns 1959 and Erdős & Moser 1964.)

**Lemma 1 (Structure and maximum order length of $\mathcal{S}_k$).** *For every $k \ge 1$:*
1. *$\mathcal{S}_k$ is a vertex-transitive tournament in which every vertex has indegree and outdegree $d_k := (3^k - 1)/2$.*
2. *The maximum length of an order (transitive subtournament) in $\mathcal{S}_k$ is $a_k := 2^k$.*

*Proof.* For any distinct $x, y \in V_k$, at the first differing coordinate $j$ we have $y_j - x_j \in \{1, 2\} \pmod 3$, so $y_j - x_j \equiv 1$ and $x_j - y_j \equiv 1$ are mutually exclusive and exhaustive; thus $\mathcal{S}_k$ is a tournament. Coordinatewise addition $x \mapsto x + c$ ($c \in V_k$) preserves the first differing coordinate and its difference, hence is a transitive automorphism group of $\mathcal{S}_k$. Therefore all vertices share the same indegree and outdegree; counting all $T(T - 1)/2$ directed edges by head or tail gives $d_k = (T - 1)/2 = (3^k - 1)/2$.

For the maximum order length, set $a_0 := 1$ for the one-vertex graph $V_0$. Partition $V_k$ into three blocks $B_0, B_1, B_2$ by the first coordinate. Each $B_r$ induces an isomorphic copy of $\mathcal{S}_{k-1}$, and all cross-block edges run $B_0 \to B_1 \to B_2 \to B_0$. Any order meeting all three blocks would contain vertices $u \in B_0, v \in B_1, w \in B_2$ forming a directed $3$-cycle, which is impossible since all edges in an order point forward. Thus any order intersects at most two blocks, contributing at most $a_{k-1}$ vertices from each, so $a_k \le 2 a_{k-1}$. Conversely, concatenating a maximum order of $B_0$ followed by a maximum order of $B_1$ yields a valid order of length $2 a_{k-1}$ because every edge from $B_0$ to $B_1$ points forward. Hence $a_k = 2 a_{k-1} = 2^k$. $\square$

**Theorem 1 (Unbounded $\mathrm{UB}/\mathrm{ILB}$ ratio).** *For every integer $k \ge 1$,*
$$
\mathrm{UB}(\mathcal{S}_k) = \frac{3^k}{\sqrt{2^k}}, \qquad
\mathrm{ILB}(\mathcal{S}_k) = \frac{3^k}{\sqrt{(3^k - 1)/2}}, \qquad
\frac{\mathrm{UB}(\mathcal{S}_k)}{\mathrm{ILB}(\mathcal{S}_k)} = \sqrt{\frac{3^k - 1}{2^{k+1}}}.
$$
*In particular, $\mathrm{UB}(\mathcal{S}_k) / \mathrm{ILB}(\mathcal{S}_k) \to \infty$ as $k \to \infty$, refuting Conjecture 12 / Conjecture `conj:tight`.*

*Proof.* Write $T = 3^k$, $a = a_k = 2^k$, and $d = d_k = (3^k - 1)/2$.

**Exact value of $\mathrm{UB}(\mathcal{S}_k)$.** For any feasible assignment $\lambda$, Cauchy–Schwarz on each $C \in \mathcal{C}$ gives
$$
T = \sum_{C \in \mathcal{C}} \sum_{t \in C} \lambda_{C, t} \le \sum_{C \in \mathcal{C}} \sqrt{|C|}\,\|\lambda_C\|_2 \le \sqrt{a} \sum_{C \in \mathcal{C}} \|\lambda_C\|_2,
$$
so $\mathrm{UB}(\mathcal{S}_k) \ge T / \sqrt{a}$. For equality, let $\mathcal{O}$ be the orbit of a single maximum order $C^*$ (length $a$) under translations in $V_k$. Every $C \in \mathcal{O}$ has maximum length $a$ and is therefore maximal ($C \in \mathcal{C}$). Let $m := |\mathcal{O}|$. By vertex transitivity, every vertex $t \in V_k$ lies in the same number $r$ of orders in $\mathcal{O}$, and double counting incidences gives $T r = m a$. Setting $\lambda_{C, t} := 1/r$ for $C \in \mathcal{O}, t \in C$ and $0$ otherwise is feasible and achieves cost
$$
\sum_{C \in \mathcal{O}} \|\lambda_C\|_2 = m \cdot \frac{\sqrt{a}}{r} = \frac{T}{\sqrt{a}}.
$$

**Exact value of $\mathrm{ILB}(\mathcal{S}_k)$.** Because $\mathcal{S}_k$ is a tournament, its underlying undirected graph is the complete graph $K_T$, so every nonempty independent set is a singleton $\{s\}$ (in particular $\mathrm{ILB}(\mathcal{S}_k) = \mathrm{LB}(\mathcal{S}_k)$). Writing $z_s := z_{\{s\}}$, the constraint at $t$ is $\sum_{s \in S_t} z_s \le 1$. Since every $s \in V_k$ belongs to $S_t$ for precisely $d$ vertices $t$ (its outdegree), summing all $T$ constraints gives $d \sum_{s \in V_k} z_s \le T$. By Cauchy–Schwarz,
$$
\sum_{s \in V_k} \sqrt{z_s} \le \sqrt{T \sum_{s \in V_k} z_s} \le \frac{T}{\sqrt{d}}.
$$
Setting $z_s := 1/d$ for all $s \in V_k$ satisfies $\sum_{s \in S_t} z_s = |S_t|/d = 1$ for every $t$ and attains $T/\sqrt{d}$. Dividing $\mathrm{UB}(\mathcal{S}_k)$ by $\mathrm{ILB}(\mathcal{S}_k)$ yields $\sqrt{(3^k - 1)/2^{k+1}} \to \infty$. $\square$

---

## 3. General Vertex-Transitive Orbit Formulas and Verification

**Proposition 2 (Exact values on vertex-transitive directed graphs).** *Let $\mathcal{S}$ be any vertex-transitive directed graph on $T$ vertices with maximum order length $a$, and suppose every vertex has at least one observer ($B(I) := \{t \in [T] : I \cap S_t \neq \varnothing\} \neq \varnothing$ for every nonempty $I \in \mathcal{I}(\mathcal{S})$). Then*
$$
\mathrm{UB}(\mathcal{S})^2 = \frac{T^2}{a}, \qquad
\mathrm{ILB}(\mathcal{S})^2 = T^2 \max_{I \in \mathcal{I}(\mathcal{S})} \frac{|I|}{|B(I)|}.
$$

*Proof.* The orbit argument in Theorem 1 (using the full automorphism group of $\mathcal{S}$ acting on the orbit of a maximum *sequence*, which remains valid even when bidirectional edges exist) gives $\mathrm{UB}(\mathcal{S}) = T/\sqrt{a}$. For $\mathrm{ILB}(\mathcal{S})$, let $h := \max_{I \in \mathcal{I}(\mathcal{S})} |I|/|B(I)|$. Summing the $T$ constraints gives $\sum_I |B(I)| z_I \le T$, so Cauchy–Schwarz yields
$$
\left(\sum_{t=1}^T \sqrt{\sum_{I \ni t} z_I}\right)^2
\le T \sum_{t=1}^T \sum_{I \ni t} z_I
= T \sum_I |I| z_I
\le T h \sum_I |B(I)| z_I \le T^2 h.
$$
Conversely, let $I^*$ attain $h$, and distribute total weight $T / |B(I^*)|$ uniformly across all distinct images of $I^*$ under the automorphism group. By vertex transitivity, every vertex receives the same constraint load (average $1$, hence exactly $1$) and the same objective summand, making both inequalities tight. $\square$

### Reproducible Verification

Run from the repository root:

```sh
.venv/bin/python -B solutions/temporal_feedback_regret/explore.py
```

The script [`explore.py`](explore.py) implements exact rational evaluation of Proposition 2 (`fractions.Fraction`) via bitmask DP for maximum order length and exhaustive independent-set enumeration:
- Checks all **246 nonempty circulant step sets** on $n = 3, \dots, 8$ vertices, **4 seeded random circulant tournaments** ($n = 11, 13, 17, 19$), the Kneser graph $KG(5, 2)$, a $3$-layer graph, a complete bipartite graph, and the wrapped butterfly graph.
- Verifies tournament regularity and exact values on the triangle powers $\mathcal{S}_k$ for $k = 1, 2, 3, 4, 5$:

| $k$ | $T = 3^k$ | $a_k = 2^k$ | $d_k = (3^k-1)/2$ | $\mathrm{UB}^2$ | $\mathrm{ILB}^2$ | $(\mathrm{UB}/\mathrm{ILB})^2$ |
|---:|---:|---:|---:|---:|---:|---:|
| $1$ | $3$ | $2$ | $1$ | $9/2$ | $9$ | $1/2$ |
| $2$ | $9$ | $4$ | $4$ | $81/4$ | $81/4$ | $1$ |
| $3$ | $27$ | $8$ | $13$ | $729/8$ | $729/13$ | $13/8$ |
| $4$ | $81$ | $16$ | $40$ | $6561/16$ | $6561/40$ | $5/2$ |
| $5$ | $243$ | $32$ | $121$ | $59049/32$ | $59049/121$ | $121/32$ |

All assertions passed.

---

## References

- P. Erdős and L. Moser, *On the representation of directed graphs as unions of orderings*, Magyar Tud. Akad. Mat. Kutató Int. Közl. 9:125–132, 1964.
- K. Gatmiry and J. Schneider, *Adversarial Online Learning with Temporal Feedback Graphs*, Proceedings of the 37th Annual Conference on Learning Theory (COLT 2024), PMLR 247:1866–1896, 2024; preprint [arXiv:2407.00571](https://arxiv.org/abs/2407.00571), July 2024.
- R. Stearns, *The voting problem*, American Mathematical Monthly 66(9):761–763, 1959.
