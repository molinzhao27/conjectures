# Logarithmic Adaptive Identification Implies Bounded Synthesized Halving

**Target paper:** Ganghua Wang and Shaddin Dughmi, *On the Sample Complexity of Active Learning with Membership Queries*, [arXiv:2609.27241](https://arxiv.org/abs/2609.27241) (September 2026), Section 4, Conjecture `conj:halve`.  
**Companion files:** [`solutions/active_mq_complexity/explore.py`](explore.py), [`solutions/active_mq_complexity/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture `conj:halve`: PROVED.** For any binary concept class $\mathcal{C} \subseteq \{0, 1\}^{\mathcal{X}}$, if $\mathrm{AIC}(\mathcal{C}, m) = O(\ln m)$, then $\mathrm{SHD}(\mathcal{C}, m) = O(1)$ (Theorem 1).
- **Exact scope:** The proof handles arbitrary version spaces $V \subseteq \mathcal{C}$ with unrestricted, arbitrarily large fibers over each sample labeling $y \in V_S$. Normalizing $\mathrm{AIC}(\mathcal{C}, m) \le a \log_2(m + 1)$ for all $m \ge 1$ with $a \ge 1$, the resulting constant bound $t = t(a)$ depends only on $a$ (and is independent of the sample size $m$, the sample $S$, and the version space $V$).
- **Quantitative remark:** The two-stage $\epsilon$-net proof yields a non-elementary (at least doubly exponential) upper bound $t(a)$ in terms of $a$; determining the optimal quantitative dependence of $t(a)$ on $a$ remains open. Wang and Dughmi ([arXiv:2609.27241](https://arxiv.org/abs/2609.27241), Theorem `thm:halve_near`) previously established the near-optimal regime $\mathrm{AIC}(m) \le \log_2 m + B$.

---

## 1. Definitions and Main Theorem

Let $\mathcal{X}$ be a domain and $\mathcal{C} \subseteq \{0, 1\}^{\mathcal{X}}$ a binary concept class. For a finite sample $S \in \mathcal{X}^m$ and a version space $V \subseteq \mathcal{C}$, denote the set of realizable $S$-labelings by $V_S := \{h|_S : h \in V\}$, and for any query batch $U \in \mathcal{X}^t$ and answer pattern $b \in \{0, 1\}^U$, write $V[U = b] := \{h \in V : h|_U = b\}$.

Following Section 4 of [arXiv:2609.27241](https://arxiv.org/abs/2609.27241):
- The **adaptive identification complexity** $\mathrm{AIC}(\mathcal{C}, m)$ is the supremum over $S \in \mathcal{X}^m$ of the minimum depth $t \ge 0$ of a deterministic membership-query decision tree (querying arbitrary points of $\mathcal{X}$) that uniquely determines $h|_S$ for every $h \in \mathcal{C}$.
- The **synthesized halving dimension** $\mathrm{SHD}(\mathcal{C}, m)$ is the supremum over $S \in \mathcal{X}^m$ of the minimum $t \ge 0$ such that for every $V \subseteq \mathcal{C}$, there exists a fixed (non-adaptive) batch $U \in \mathcal{X}^t$ satisfying
  $$
  |(V[U = b])_S| \le \lceil |V_S| / 2 \rceil \qquad (\forall b \in \{0, 1\}^U).
  $$
  (Checking all patterns $b \in \{0, 1\}^U$ is equivalent to the paper's $\sup_{f \in \mathcal{C}} |\{\mathrm{VS}_{U,f}(V)\}_S| \le \lceil |V_S|/2 \rceil$, since unrealized patterns give $0$; batches of length $< t$ may be padded with repeated queries.)

Note that $\mathrm{SHD}(\mathcal{C}, 0) = 0$ trivially, and whenever $|V_S| \le 1$, zero queries suffice. Moreover, since $\mathrm{AIC}(\mathcal{C}, m) \le m$ for every $m \ge 1$ (querying $S$ directly always suffices), the asymptotic hypothesis $\mathrm{AIC}(\mathcal{C}, m) = O(\ln m)$ implies the existence of a constant $a \ge 1$ such that
$$
\mathrm{AIC}(\mathcal{C}, m) \le a \log_2(m + 1) \qquad (\forall m \ge 1).
$$

**Theorem 1 (Resolution of `conj:halve`).** *Fix $a \ge 1$. If a binary concept class $\mathcal{C} \subseteq \{0, 1\}^{\mathcal{X}}$ satisfies $\mathrm{AIC}(\mathcal{C}, m) \le a \log_2(m + 1)$ for every $m \ge 1$, then there exists a finite constant $t = t(a)$ such that $\mathrm{SHD}(\mathcal{C}, m) \le t$ for every $m \ge 0$.*

---

## 2. Two Standard Sampling Facts

For a set system $\mathcal{R}$ on a ground set $\Omega$, its *growth function* $\Pi_{\mathcal{R}}(s)$ is the maximum number of distinct intersections $\{R \cap E : R \in \mathcal{R}\}$ over $s$-element subsets $E \subseteq \Omega$, and its *VC dimension* is the largest $d$ such that $\Pi_{\mathcal{R}}(d) = 2^d$.

**Lemma 2 (Primal and dual growth; Assouad, 1983; Sauer, 1972; Shelah, 1972).** *Under the hypothesis of Theorem 1 with $a \ge 1$:*
1. *For every $s \ge 1$, the primal growth satisfies $\Pi_{\mathcal{C}}(s) \le 2^{\mathrm{AIC}(\mathcal{C}, s)} \le (s + 1)^a$. In particular, $\mathrm{VC}(\mathcal{C}) \le v := \max\{j \in \mathbb{N} : 2^j \le (j + 1)^a\} < \infty$.*
2. *The dual family of evaluation sets $\{E_x : x \in \mathcal{X}\}$ on $\mathcal{C}$, where $E_x := \{h \in \mathcal{C} : h(x) = 1\}$, has VC dimension at most $d := 2^{v + 1} - 1 < 2^{v + 1}$ (Assouad, 1983), and its growth function satisfies $\Pi_{\{E_x\}}(s) \le (s + 1)^d$ for every $s \ge 0$.*

*Proof.* Counting leaves of a depth-$\mathrm{AIC}(\mathcal{C}, s)$ decision tree gives $|\mathcal{C}_S| \le 2^{\mathrm{AIC}(\mathcal{C}, s)} \le (s + 1)^a$, bounding $v = \mathrm{VC}(\mathcal{C}) < \infty$. For the dual bound, if $\{E_x\}$ shattered $2^{v+1}$ concepts, index those concepts by $\{0, 1\}^{v+1}$; for each bit position $r \in [v + 1]$, pick $x_r \in \mathcal{X}$ realizing the $r$-th coordinate across all $2^{v+1}$ concepts. Then $\mathcal{C}$ shatters $\{x_1, \dots, x_{v+1}\}$, contradicting $\mathrm{VC}(\mathcal{C}) \le v$. Finally, for any family of VC dimension $\le d$, the Sauer–Shelah–Perles lemma gives $\Pi(s) \le \sum_{j=0}^d \binom{s}{j}$ (induct on $s + d$ by deleting the last point: traces with both extensions have VC dimension $\le d - 1$). Encoding subsets of $[s]$ of size $\le d$ as sorted padded $d$-tuples from $s + 1$ symbols gives $\sum_{j=0}^d \binom{s}{j} \le (s + 1)^d$. $\square$

**Lemma 3 (Finite $\epsilon$-net hitting lemma; Haussler and Welzl, 1987).** *Let $P$ be a probability distribution on a finite ground set $\Omega$, let $\mathcal{R} \subseteq 2^\Omega$, and let $n \ge 1$ and $\epsilon \in (0, 1]$ satisfy $n \epsilon \ge 8$. Then an i.i.d. sample $X \sim P^n$ misses some range $R \in \mathcal{R}$ of mass $P(R) \ge \epsilon$ with probability at most*
$$
2\, \Pi_{\mathcal{R}}(2n)\, \exp(-\epsilon n / 4).
$$

*Proof.* Let $E_1$ be the event that $X = (X_1, \dots, X_n)$ misses some $R \in \mathcal{R}$ with $P(R) \ge \epsilon$. Draw an independent ghost sample $X' = (X'_1, \dots, X'_n) \sim P^n$, and let $E_2$ be the event that there exists $R \in \mathcal{R}$ with $P(R) \ge \epsilon$, $R \cap X = \varnothing$, and $|\{j \in [n] : X'_j \in R\}| \ge \epsilon n / 2$. Conditional on $X \in E_1$, pick a deterministic $R_X \in \mathcal{R}$ witnessing $E_1$ and let $p := P(R_X) \ge \epsilon$. For $Z \sim \mathrm{Bin}(n, p)$, Chebyshev's inequality gives
$$
\mathbb{P}(Z < \epsilon n / 2) \le \mathbb{P}\bigl(|Z - np| \ge np / 2\bigr) \le \frac{4 np(1 - p)}{(np)^2} \le \frac{4}{n \epsilon} \le \frac{1}{2},
$$
so $\mathbb{P}(E_2) \ge \frac{1}{2} \mathbb{P}(E_1)$. By exchangeability of $(X, X') \sim P^{2n}$, condition on the ordered $2n$-tuple $Y = (Y_1, \dots, Y_{2n})$ and apply a uniformly random partition into two halves of size $n$. The set of index patterns $\{j \in [2n] : Y_j \in R\}$ as $R$ varies over $\mathcal{R}$ has cardinality at most $\Pi_{\mathcal{R}}(2n)$ (absorbing repeated draws in $Y$). For any fixed index pattern of size $m \ge \epsilon n / 2$, the hypergeometric probability that the first half avoids all $m$ positions is
$$
\prod_{j = 0}^{n - 1} \left(1 - \frac{m}{2n - j}\right) \le \left(1 - \frac{\epsilon}{4}\right)^n \le e^{-\epsilon n / 4}.
$$
Taking a union bound over the at most $\Pi_{\mathcal{R}}(2n)$ index patterns and multiplying by $2$ completes the proof. $\square$

---

## 3. Proof of Theorem 1

Fix $m \ge 1$, a sample $S \in \mathcal{X}^m$, and a version space $V \subseteq \mathcal{C}$. Set $Y := V_S$ and $N := |Y|$.

**Small-$N$ base case ($N < 8$).** If $N \le 1$, zero queries suffice. If $2 \le N < 8$, repeatedly pick a coordinate of $S$ that separates two labelings in a nonsingleton part of the current partition of $Y$; each query increases the number of parts by at least $1$, so at most $N - 1 \le 6 < 8$ queries isolate every element of $Y$ completely. Hence any $t \ge 8$ handles $N < 8$. Assume henceforth that $N \ge 8$.

**Stage 1: Dual pair-hitting and random identification trees.** Let $\Omega_{\mathrm{pair}} := \{(y, z) \in Y \times Y : y \neq z\}$, which is finite of size $N(N - 1)$. For each $(y, z) \in \Omega_{\mathrm{pair}}$, fix one coordinate $s(y, z) \in S$ where $y$ and $z$ differ. For an integer $k \ge 1$ (depending only on $a$, chosen below), draw $k$ ordered pairs $(y_1, z_1), \dots, (y_k, z_k)$ independently and uniformly from $\Omega_{\mathrm{pair}}$, and let $S_{\mathrm{train}} := (s(y_1, z_1), \dots, s(y_k, z_k)) \in \mathcal{X}^k$. Choose deterministically an optimal identification tree $T = T(S_{\mathrm{train}})$ for $S_{\mathrm{train}}$, of depth at most
$$
D := \lceil a \log_2(k + 1) \rceil.
$$
Crucially, **for every training pair $(y_r, z_r)$ and every choice of concepts $f, g \in V$ extending $y_r, z_r$**, $f$ and $g$ must reach different leaves of $T$, because every leaf of $T$ uniquely determines the concept's values on all $k$ coordinates of $S_{\mathrm{train}}$. Moreover, the distribution of $T$ depends only on $Y$ and the chosen coordinates $s(y, z)$, not on any choice of fiber representatives.

Now fix any finite transversal $W \subseteq V$ consisting of $h > N / 2$ concepts with pairwise distinct restrictions to $S$. We claim that with probability at least $1/2$ over the draw of the $k$ training pairs, every leaf of $T$ contains strictly fewer than $h / 2$ members of $W$.

To prove this, consider all conjunctions $L$ of at most $D$ query-answer literals $(x_j = b_j)$. For each such $L$, define a range $R_W(L) \subseteq \Omega_{\mathrm{pair}}$ consisting of all pairs $(y, z)$ such that both $y$ and $z$ are realized in $W$ and their unique representatives $f_y, f_z \in W$ both satisfy $L$. Let $\mathcal{R}_W$ be the family of all such pair ranges. On any $l$ pairs in $\Omega_{\mathrm{pair}}$, at most $2l$ representatives from $W$ participate; by Lemma 2(2), a single literal (either $E_x$ or its complement) induces at most $2(2l + 1)^d$ patterns on those representatives. Allowing a padding symbol for conjunctions of length $< D$, the growth function of $\mathcal{R}_W$ satisfies
$$
\Pi_{\mathcal{R}_W}(l) \le \bigl[1 + 2(2l + 1)^d\bigr]^D \qquad (\forall l \ge 1).
$$
If a conjunction $L$ is satisfied by $c \ge h / 2$ members of $W$, then $c \ge N / 4$ and $c - 1 \ge N / 4 - 1 \ge N / 8$ (using $N \ge 8$), so the uniform mass of $R_W(L)$ on $\Omega_{\mathrm{pair}}$ is
$$
\frac{c(c - 1)}{N(N - 1)} \ge \frac{(N / 4)(N / 8)}{N^2} = \frac{1}{32}.
$$
Since $\log \bigl([1 + 2(4k + 1)^d]^D\bigr) = O_a((\log k)^2)$ grows polylogarithmically in $k$ while $k / 128$ grows linearly, we can choose an integer $k = k(a)$ depending only on $a$ such that $k / 32 \ge 8$ and
$$
2\bigl[1 + 2(4k + 1)^d\bigr]^D e^{-k / 128} < \frac{1}{2}.
$$
By Lemma 3 (with $\epsilon = 1/32$), with probability at least $1/2$ the $k$ training pairs hit every range $R_W(L) \in \mathcal{R}_W$ of mass $\ge 1/32$. Because each leaf transcript of $T$ is a conjunction $L$ of length $\le D$ whose range $R_W(L)$ contains *no* training pair (by the separation property above), on this event every leaf of $T$ contains fewer than $h / 2$ members of $W$—even though $T$ depends on the training sample.

**Query distribution $Q$ and heavy pairs.** Define a single probability distribution $Q$ on $\mathcal{X}$, **independent of $W$**, as follows: draw the random tree $T$, list the queries at its internal nodes (at most $2^D - 1$ nodes), pad the list with an arbitrary query to length $M := 2^D$, and output a uniformly random entry of the list. Because $Y$ and $k$ are finite, $Q$ has finite support $\Omega_Q \subset \mathcal{X}$.

Fix any transversal $W \subseteq V$ of size $h > N / 2$, and draw an ordered distinct pair $(f, g)$ uniformly from $W$, independently of $T$. On the good-tree event (probability $\ge 1/2$), conditional on $f \in W$, strictly fewer than $h / 2$ members of $W$ share $f$'s leaf, so at least $h - \lfloor (h - 1)/2 \rfloor > h/2$ of the $h - 1$ choices for $g \neq f$ land in a different leaf from $f$. Hence $f$ and $g$ reach different leaves with probability at least $\frac{1}{2} \cdot \frac{h/2}{h-1} > \frac{1}{4}$. Whenever $f$ and $g$ reach different leaves of $T$, they disagree at the internal node where their root-to-leaf paths diverge, which is sampled with probability at least $1/M$. Therefore
$$
\mathbb{E}_{\substack{f, g \in W \\ f \neq g}}\bigl[Q(\{x \in \mathcal{X} : f(x) \neq g(x)\})\bigr] \ge \frac{1}{4M}.
$$
Consequently, **every** transversal $W \subseteq V$ of size $h > N / 2$ contains a pair $f, g \in W$ whose disagreement set $D_{f,g} := \{x \in \mathcal{X} : f(x) \neq g(x)\}$ has $Q$-mass at least $\eta := \frac{1}{4M} = 2^{-D-2}$.

**Stage 2: Primal hitting of heavy disagreement sets.** Consider the range family $\mathcal{D} := \{D_{f,g} \cap \Omega_Q : f, g \in \mathcal{C}\}$ on the finite ground set $\Omega_Q$. On any $s$ queries, the pair $(f, g)$ takes at most $\Pi_{\mathcal{C}}(s)^2 \le (s + 1)^{2a}$ joint labelings by Lemma 2(1), so $\Pi_{\mathcal{D}}(s) \le (s + 1)^{2a}$. Choose an integer $t = t(a) \ge 8$ depending only on $a$ such that $t \eta \ge 8$ and
$$
2 (2t + 1)^{2a} e^{-\eta t / 4} < 1.
$$
By Lemma 3 applied to $(\Omega_Q, Q, \mathcal{D})$ with $\epsilon = \eta$, a random batch $U \sim Q^t$ hits *all* disagreement sets of $Q$-mass $\ge \eta$ with strictly positive probability; hence there exists a deterministic batch $U \in \mathcal{X}^t$ hitting every such disagreement set.

We claim that $U$ halves $V_S$ under every answer pattern $b \in \{0, 1\}^U$. Suppose for contradiction that some pattern $b$ satisfies $|(V[U = b])_S| > \lceil N / 2 \rceil \ge N / 2$. Choose one surviving representative in $V[U = b]$ for each labeling in $(V[U = b])_S$, forming a transversal $W \subseteq V[U = b]$ of size $h > N / 2$. By Stage 1, $W$ contains a pair $f, g$ with $Q(D_{f,g}) \ge \eta$; by Stage 2, $U$ hits $D_{f,g}$, contradicting $f|_U = b = g|_U$. Thus $\mathrm{SHD}(\mathcal{C}, m) \le t(a)$ for all $m \ge 0$. $\square$

---

## 4. Reconciliation with Wang–Dughmi's Finite Example and Verification

In [arXiv:2609.27241](https://arxiv.org/abs/2609.27241) (Section 4), Wang and Dughmi construct a sequence of finite concept classes $\mathcal{C}_n$ with logarithmic adaptive depth on a large sample of size $m = n 2^n$ yet logarithmic halving batch size. This does not contradict Theorem 1: within $\mathcal{C}_n$, the $n$ auxiliary bit queries of a single block are shattered, so exact identification on that length-$n$ sample requires $n$ queries. Hence any uniform bound $\mathrm{AIC}(\mathcal{C}_n, m) \le a_n \log_2(m + 1)$ must satisfy $a_n \ge n / \log_2(n + 1) \to \infty$ as $n \to \infty$.

### Reproducible Finite Checks

Run from the repository root:

```sh
.venv/bin/python -B solutions/active_mq_complexity/explore.py
```

The script [`explore.py`](explore.py) computes exact optimal adaptive depth and minimum non-adaptive halving-batch size across 447 samples and 1,339 sample/version-space cases spanning nine concept families (`PG(2,2)`, `AG(2,3)`, `RM(1,3)`, radius-1 Hamming balls, bipartite edge incidence, products and disjoint unions of threshold chains, binary tree subtrees, and three gated search blocks), preserving full fibers over every sample labeling. It also executes the exact two-stage sampling mechanism on a 4-label, 8-concept full-fiber class by aggregating all $4^9 = 262{,}144$ sequences of separating coordinates: across all 48 transversals of size $> 2$, the minimum good-tree probability is $32385/32768 > 1/2$, every transversal contains a heavy pair, and a batch of $3$ queries hits all heavy disagreements and halves the full-fiber version space. All assertions passed.

---

## References

- P. Assouad, *Densité et dimension*, Annales de l'Institut Fourier 33(3):233–282, 1983.
- D. Haussler and E. Welzl, *$\epsilon$-nets and simplex range queries*, Discrete & Computational Geometry 2(2):127–151, 1987.
- N. Sauer, *On the density of families of sets*, Journal of Combinatorial Theory, Series A 13(1):145–147, 1972.
- S. Shelah, *A combinatorial problem; stability and order for models and theories in infinitary languages*, Pacific Journal of Mathematics 41(1):247–261, 1972.
- G. Wang and S. Dughmi, *On the Sample Complexity of Active Learning with Membership Queries*, [arXiv:2609.27241](https://arxiv.org/abs/2609.27241), September 2026.
